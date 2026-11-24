//! Host-RAM tier of prefix reuse, engine side: what is copied between the device and the store
//! (`cache_ram.rs`), and when.
//!
//! This is a child module of `engine::qwen3` (declared there with `#[path]`).
//!
//! The device tier (`prefix_cache.rs`) resumes a prefill from state the card still holds. This
//! tier keeps a copy of that state in host memory and puts it back when the card has lost it:
//! after another conversation, after another model, after the release of the card. A restore
//! ends with the device and its table in the state they had when the entry was saved (cut to
//! the position the request resumes from), and the request then runs as a device hit: the same
//! tiles, on the same rows, from the same state. Nothing here computes a row.
//!
//! What one saved prefix is, on this card and this engine:
//! - **K/V rows** of the prompt's whole prefill tiles up to its last kept position. They are
//!   positional and lie in every attention layer's two caches, chunk by chunk and head by head;
//!   a tile's rows are one host segment, laid out as the device ranges follow each other.
//! - **The recurrent state** at the kept positions the device holds (a slot each).
//! - **The drafter's context rings**, whole, with the first position valid in them: a drafter
//!   only proposes tokens, so they cannot change the output, but without them a restored turn
//!   would draft from the rows of its own short tail alone.
//!
//! When it is saved: after the response, while no request waits for the card (the server calls
//! [`Qwen3Engine::idle_work`]); a save gives way to a request that arrives and goes on after
//! it. A request that is about to overwrite kept positions that are not saved yet copies them
//! out first, and so does the release of the card. A conversation's next turn extends its
//! entry: only the new tiles, the new positions and the rings are copied.
//!
//! When it is restored: when an entry offers a later resume point than the device does and
//! copying is faster than prefilling the rows in between, by the engine's own measured rates
//! (`hc_copy_wins`). Device tiles that still equal the entry's (nothing wrote them since the
//! last copy) are not copied again.
//!
//! Every decision is the verified core's (`verified/host_cache.bend`, the host-tier section of
//! `verified/kv_pool.bend`); the copies go through the transfer ring (`synapse/xfer.rs`).

use std::collections::BTreeMap as Map;
use std::sync::Arc;

use super::prefix_cache::{ImageId, PrefixCache};
use super::*;
use crate::cache_ram::{digest128, EngineId, HostCache, SharedCache};
use crate::synapse::xfer::{Piece, XferRing, XferStats};
use crate::verified::{
    hc_cache_span, hc_cache_tiles, hc_calibrate_bytes, hc_clean_hi, hc_clean_join_hi,
    hc_clean_join_lo, hc_clean_lo, hc_copy_wins, hc_layout, hc_loses_unsaved, hc_must_flush,
    hc_new_tiles, hc_piece_dev_off, hc_piece_host_off, hc_piece_inside, hc_piece_len,
    hc_pieces_per_tile, hc_range_end, hc_range_inside, hc_restore_bytes, hc_ring_due,
    hc_ring_floor, hc_ring_restore, hc_rows_added, hc_rows_complete, hc_save_ok, hc_saved_whole,
    hc_tile_bytes, hc_tile_chunk, hc_tile_clean, hc_tile_copied, hc_tile_slot, hc_tiles_fill_chunk,
    hc_tiles_per_chunk, hc_written_hi, hc_written_lo, pc_boundary, pc_request_may_reuse, HcLayout,
};

/// Tiles whose rows one copy call takes (a waiting request is noticed between ring buffers
/// inside a call; rows are committed tile by tile either way).
const ROUND_TILES: u64 = 64;
/// A copy below this does not update a measured rate (its fixed costs would dominate).
const RATE_MIN_BYTES: u64 = 32 * 1024 * 1024;
/// Bytes the rates are first measured with, when the engine is loaded.
const CALIBRATE_BYTES: u64 = 128 * 1024 * 1024;

/// Host bytes the first copies of a loaded engine take at most (the buffer of
/// [`HostTier::calibrate`], mapped and freed again at each load): part of what a server plans
/// beside its store.
pub fn first_copies_bytes() -> u64 {
    crate::hostmem::AnonMap::mapped_len(CALIBRATE_BYTES)
}
/// Rows of a BF16 K/V block: the engine writes cache rows in blocks of this many, so a tile
/// whose rows are to be one range per head is a whole number of them. A cache of another dtype
/// has its own block (`kv_vperm_rows`, [`RowLayout::with_block`]).
const KV_BLOCK_ROWS: u64 = 128;

/// How long the card is left idle after a response before the save starts: a request that
/// arrives inside it (a client that answers a tool call at once) never waits for a copy.
/// Tuning, not a rule: the save is correct at any value.
pub const IDLE_GRACE: std::time::Duration = std::time::Duration::from_millis(30);

/// Where the rows of a request's prompt came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tier {
    /// All prefilled.
    #[default]
    Prefilled,
    /// The first rows were still on the device.
    Device,
    /// The first rows were copied back from host memory.
    Host,
}

impl Tier {
    /// The tier of a request that reused `reused` rows, `restored`: after a restore from host
    /// memory (the verified `hc_tier`).
    pub fn of(restored: bool, reused: u64) -> Self {
        match crate::verified::hc_tier(restored, reused) {
            crate::verified::HcTier::HcPrefilled => Self::Prefilled,
            crate::verified::HcTier::HcFromDevice => Self::Device,
            crate::verified::HcTier::HcFromHost => Self::Host,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Prefilled => "prefilled",
            Self::Device => "device",
            Self::Host => "host",
        }
    }
}

/// What the tier did for one request.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TierReport {
    pub tier: Tier,
    /// The restore, inside the request's time to first token.
    pub restore_ms: f64,
    pub restore_bytes: u64,
    /// Rows the restore made resumable beyond what the device held.
    pub restore_rows: u64,
    /// A save the request paid for before it overwrote kept positions.
    pub save_ms: f64,
    pub save_bytes: u64,
}

/// What one save did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SaveReport {
    pub bytes: u64,
    pub ms: f64,
    /// Everything the device held is in host memory now.
    pub complete: bool,
    /// A waiting request cut it short.
    pub yielded: bool,
}

/// Measured rates of this engine: copies in bytes per second, prefill in rows per second
/// (0 = not measured yet).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rates {
    pub to_host: u64,
    pub to_device: u64,
    pub prefill_rows: u64,
}

/// What a loaded engine's tier measured and what its entries cost, with its last request.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TierFacts {
    pub rates: Rates,
    /// Host bytes per saved prompt row (K/V rows), per kept position, and of the drafter rings.
    pub bytes_per_row: u64,
    pub bytes_per_state: u64,
    pub ring_bytes: u64,
    pub last: TierReport,
}

/// The device side of a copy (the transfer ring on the card; a plain memory copy in tests).
pub trait Copier {
    /// Device to host. `stop` is asked between ring buffers; returns the pieces complete.
    fn to_host(
        &mut self,
        pieces: &[Piece],
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<(usize, XferStats), String>;
    /// Host to device; complete on return.
    fn to_device(&mut self, pieces: &[Piece]) -> Result<XferStats, String>;
}

/// The transfer ring of a loaded engine as a [`Copier`].
pub(super) struct RingCopier<'a> {
    pub ring: &'a mut XferRing,
    pub dev: &'a HpuDevice,
}

impl Copier for RingCopier<'_> {
    fn to_host(
        &mut self,
        pieces: &[Piece],
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<(usize, XferStats), String> {
        // a save reads rows the compute stream wrote: nothing may still be queued on it
        // SAFETY: the stream belongs to `dev`.
        unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
            .map_err(|e| format!("host tier: sync before a save: {e}"))?;
        // SAFETY: the pieces come from `HostTier`: host ranges inside live segments of the
        // store (held under its lock), device ranges inside this engine's planned buffers,
        // which no launch writes between requests.
        unsafe { self.ring.d2h(self.dev, pieces, stop) }
            .map_err(|e| format!("host tier: copy to host: {e}"))
    }

    fn to_device(&mut self, pieces: &[Piece]) -> Result<XferStats, String> {
        // SAFETY: as above; nothing reads the device ranges until the request's prefill, which
        // is enqueued after this returns.
        unsafe { self.ring.h2d(self.dev, pieces) }
            .map_err(|e| format!("host tier: copy to the device: {e}"))
    }
}

/// Where the K/V rows of a prompt lie on the device: every attention layer's key and value
/// cache is `chunks` chunks of `chunk_rows` rows, and inside a chunk each of `heads` heads
/// holds its rows one after the other (`row_bytes` each). Every size and offset of it is the
/// verified core's (`verified/host_cache.bend`, section 16); this type holds the numbers and
/// adds the offsets to addresses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowLayout {
    /// Device address of each cache, in name order.
    pub caches: Vec<u64>,
    pub chunks: u64,
    pub chunk_rows: u64,
    /// Tiles of one chunk (`hc_tiles_per_chunk`); they fill it exactly.
    pub tiles_per_chunk: u64,
    pub heads: u64,
    pub row_bytes: u64,
    pub tile: u64,
}

impl RowLayout {
    /// The layout of BF16 caches of `cache_bytes` each ([`Self::with_block`] with the BF16 block).
    pub fn new(
        caches: Vec<u64>,
        cache_bytes: u64,
        chunks: u64,
        chunk_rows: u64,
        heads: u64,
        row_bytes: u64,
        tile: u64,
    ) -> Result<Self, String> {
        Self::with_block(caches, cache_bytes, chunks, chunk_rows, heads, row_bytes, tile, KV_BLOCK_ROWS)
    }

    /// The layout of caches of `cache_bytes` each whose V rows are permuted in blocks of `block`
    /// rows. An error says why this engine's caches cannot be copied tile by tile.
    #[allow(clippy::too_many_arguments)]
    pub fn with_block(
        caches: Vec<u64>,
        cache_bytes: u64,
        chunks: u64,
        chunk_rows: u64,
        heads: u64,
        row_bytes: u64,
        tile: u64,
        block: u64,
    ) -> Result<Self, String> {
        // whether these caches can be copied tile by tile is the verified `hc_layout`'s decision
        match hc_layout(
            caches.len() as u64,
            chunks,
            chunk_rows,
            heads,
            row_bytes,
            tile,
            block,
            cache_bytes,
        ) {
            HcLayout::HcLayoutOk => {}
            HcLayout::HcLayoutEmpty => return Err("no K/V caches to copy".into()),
            HcLayout::HcLayoutTile => {
                return Err(format!(
                    "a prefill tile of {tile} rows does not divide a K/V chunk of {chunk_rows} rows into whole {block}-row blocks"
                ))
            }
            HcLayout::HcLayoutSize => {
                return Err(format!(
                    "a K/V cache of {cache_bytes} bytes is not {chunks} chunks x {heads} heads x {chunk_rows} rows x {row_bytes} bytes"
                ))
            }
        }
        // The offsets of a piece count whole tiles of a chunk, so the tiles must fill a chunk
        // exactly (a tile that divides the chunk does); the laws that keep every piece inside
        // its cache are stated for such chunks.
        let tiles_per_chunk = hc_tiles_per_chunk(chunk_rows, tile);
        if !hc_tiles_fill_chunk(tiles_per_chunk, tile, chunk_rows) {
            return Err(format!(
                "{tiles_per_chunk} tiles of {tile} rows do not fill a K/V chunk of {chunk_rows} rows"
            ));
        }
        Ok(Self {
            caches,
            chunks,
            chunk_rows,
            tiles_per_chunk,
            heads,
            row_bytes,
            tile,
        })
    }

    /// Host bytes of one tile's rows.
    pub fn tile_bytes(&self) -> u64 {
        hc_tile_bytes(
            self.caches.len() as u64,
            self.heads,
            self.tile,
            self.row_bytes,
        )
    }

    /// Device ranges of one tile: one per cache and head.
    pub fn pieces_per_tile(&self) -> usize {
        hc_pieces_per_tile(self.caches.len() as u64, self.heads) as usize
    }

    /// Tiles the caches hold.
    pub fn tiles(&self) -> u64 {
        hc_cache_tiles(self.chunks, self.tiles_per_chunk)
    }

    /// Bytes of one cache.
    pub fn cache_bytes(&self) -> u64 {
        hc_cache_span(
            self.chunks,
            self.heads,
            self.tiles_per_chunk,
            self.tile,
            self.row_bytes,
        )
    }

    /// Appends the pieces of tile `t`, whose rows live at `host`: per cache, per head, the
    /// tile's rows of that head. Where a piece starts in its cache and in the tile's segment,
    /// and how long it is, are `hc_piece_dev_off`, `hc_piece_host_off` and `hc_piece_len`;
    /// a piece is used only when `hc_piece_inside` says that the place is the tile's and one
    /// the layout has (its laws then keep the piece inside the cache and inside the segment).
    /// An error names a tile the caches have no place for; nothing was appended for it.
    pub(crate) fn tile_pieces(
        &self,
        t: u64,
        host: *mut u8,
        out: &mut Vec<Piece>,
    ) -> Result<(), String> {
        let (chunk, slot) = (
            hc_tile_chunk(t, self.tiles_per_chunk),
            hc_tile_slot(t, self.tiles_per_chunk),
        );
        let caches = self.caches.len() as u64;
        let len = hc_piece_len(self.tile, self.row_bytes) as usize;
        let mut pieces = Vec::with_capacity(self.pieces_per_tile());
        for (cache, &addr) in self.caches.iter().enumerate() {
            for head in 0..self.heads {
                if !hc_piece_inside(
                    t,
                    chunk,
                    self.chunks,
                    slot,
                    self.tiles_per_chunk,
                    head,
                    self.heads,
                    cache as u64,
                    caches,
                ) {
                    return Err(format!(
                        "host tier: the K/V caches have no place for tile {t} ({} chunks of {} tiles)",
                        self.chunks, self.tiles_per_chunk
                    ));
                }
                let dev_off = hc_piece_dev_off(
                    chunk,
                    slot,
                    head,
                    self.heads,
                    self.tiles_per_chunk,
                    self.tile,
                    self.row_bytes,
                );
                let host_off =
                    hc_piece_host_off(cache as u64, head, self.heads, self.tile, self.row_bytes);
                pieces.push(Piece {
                    dev: addr + dev_off,
                    // SAFETY: `host_off + len <= tile_bytes()`, the size of the tile's segment
                    // (`hc_piece_within_segment`).
                    host: unsafe { host.add(host_off as usize) },
                    len,
                });
            }
        }
        out.extend(pieces);
        Ok(())
    }
}

/// Bytes of device ranges `ranges` (`(address, bytes)`) laid one after the other.
fn ranges_bytes(ranges: &[(u64, u64)]) -> u64 {
    ranges
        .iter()
        .fold(0, |end, &(_, bytes)| hc_range_end(end, bytes))
}

/// The pieces of device ranges `ranges` (`(address, bytes)`), whose bytes follow each other at
/// `host` in a segment of `seg_bytes` (`hc_range_end`). A range that would end past the
/// segment is an error (`hc_range_inside`), and no piece is returned.
fn range_pieces(
    ranges: &[(u64, u64)],
    host: *mut u8,
    seg_bytes: u64,
) -> Result<Vec<Piece>, String> {
    let mut off = 0u64;
    let mut out = Vec::with_capacity(ranges.len());
    for &(dev, bytes) in ranges {
        if !hc_range_inside(off, bytes, seg_bytes) {
            return Err(format!(
                "host tier: a device range of {bytes} bytes at offset {off} ends past its segment of {seg_bytes} bytes"
            ));
        }
        out.push(Piece {
            dev,
            // SAFETY: `off + bytes <= seg_bytes`, checked above.
            host: unsafe { host.add(off as usize) },
            len: bytes as usize,
        });
        off = hc_range_end(off, bytes);
    }
    Ok(out)
}

/// The tier of one loaded engine.
pub struct HostTier {
    cache: Arc<SharedCache>,
    id: EngineId,
    rows: RowLayout,
    /// Device ranges of each kept-position slot (the state buffers of every recurrent layer).
    slots: Vec<Vec<(u64, u64)>>,
    state_bytes: u64,
    /// Device ranges of the drafter's context rings (empty: no drafter that keeps them).
    ring: Vec<(u64, u64)>,
    ring_bytes: u64,
    /// Device tiles that still equal an entry's: entry -> the interval `[lo, hi)`.
    clean: Map<u64, (u64, u64)>,
    /// The entry whose ring image the device's rings equal.
    ring_owner: Option<u64>,
    /// The device may hold kept positions host memory lacks.
    unsaved: bool,
    rates: Rates,
    last: TierReport,
    /// The running request's rows were restored from host memory.
    restored: bool,
    /// Rows the last finished request took over (0: its prompt was prefilled whole).
    last_reused: u64,
    /// The entry a save that a request cut short was writing.
    partial: Option<u64>,
    /// Write counts of the entries, shared by every lane's share of the tier (`--parallel`):
    /// each save into an entry raises its count.
    writes: Arc<std::sync::Mutex<Map<u64, u64>>>,
    /// The write count of each entry this share's `clean` range or `ring_owner` was recorded at.
    seen: Map<u64, u64>,
    verbose: bool,
}

impl HostTier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cache: Arc<SharedCache>,
        id: EngineId,
        rows: RowLayout,
        slots: Vec<Vec<(u64, u64)>>,
        ring: Vec<(u64, u64)>,
        verbose: bool,
    ) -> Self {
        let state_bytes = slots.first().map_or(0, |s| ranges_bytes(s));
        let ring_bytes = ranges_bytes(&ring);
        Self {
            cache,
            id,
            rows,
            slots,
            state_bytes,
            ring,
            ring_bytes,
            clean: Map::new(),
            ring_owner: None,
            unsaved: false,
            rates: Rates::default(),
            last: TierReport::default(),
            restored: false,
            last_reused: 0,
            partial: None,
            writes: Arc::new(std::sync::Mutex::new(Map::new())),
            seen: Map::new(),
            verbose,
        }
    }

    pub fn rates(&self) -> Rates {
        self.rates
    }

    /// The device may hold kept positions host memory lacks.
    pub fn has_unsaved(&self) -> bool {
        self.unsaved
    }

    /// [lanes] This share with the entries' write counts of `other`'s (every lane's share of one
    /// engine counts the same writes).
    pub fn sharing_writes_of(mut self, other: &HostTier) -> Self {
        self.writes = Arc::clone(&other.writes);
        self
    }

    /// Forgets what this share recorded of an entry another share has written since (the
    /// verified `ln_tier_record_trusted`): its clean tiles and its ring image.
    fn void_rewritten(&mut self) {
        let writes = self.writes.lock().unwrap_or_else(|e| e.into_inner());
        let count = |id: &u64| writes.get(id).copied().unwrap_or(0);
        let stale: Vec<u64> = self
            .clean
            .keys()
            .chain(self.ring_owner.iter())
            .copied()
            .filter(|id| {
                !crate::verified::ln_tier_record_trusted(
                    self.seen.get(id).copied().unwrap_or(0),
                    count(id),
                )
            })
            .collect();
        drop(writes);
        for id in stale {
            self.clean.remove(&id);
            if self.ring_owner == Some(id) {
                self.ring_owner = None;
            }
            self.seen.remove(&id);
        }
    }

    /// This share writes entry `id`: its count goes up, and this share's record is of the new one.
    fn note_write(&mut self, id: u64) {
        let mut writes = self.writes.lock().unwrap_or_else(|e| e.into_inner());
        let c = writes.entry(id).or_insert(0);
        *c += 1;
        self.seen.insert(id, *c);
    }

    /// The device now holds entry `id` as it stands: this share's record is of its current count.
    fn note_read(&mut self, id: u64) {
        let writes = self.writes.lock().unwrap_or_else(|e| e.into_inner());
        self.seen.insert(id, writes.get(&id).copied().unwrap_or(0));
    }

    pub fn set_rates(&mut self, rates: Rates) {
        self.rates = rates;
    }

    /// What the tier did for the last request.
    pub fn last(&self) -> TierReport {
        self.last
    }

    pub fn facts(&self) -> TierFacts {
        let (bytes_per_row, bytes_per_state) = self.bytes_per_row_and_state();
        TierFacts {
            rates: self.rates,
            bytes_per_row,
            bytes_per_state,
            ring_bytes: self.ring_bytes,
            last: self.last,
        }
    }

    /// Host bytes per saved prompt row (K/V rows only), and per kept position.
    pub fn bytes_per_row_and_state(&self) -> (u64, u64) {
        (
            HostCache::charge_of(self.rows.tile_bytes()) / self.rows.tile.max(1),
            HostCache::charge_of(self.state_bytes),
        )
    }

    fn note_rate(slot: &mut u64, st: &XferStats) {
        if st.bytes >= RATE_MIN_BYTES && st.rate() > 0 {
            *slot = st.rate();
        }
    }

    /// Measures the two copy rates through `copier` with the first bytes of the first K/V
    /// cache, which are read and written back unchanged.
    ///
    /// The host buffer they go through is asked for now, like every other host memory of the
    /// tier (`HostCache::admit_beside`: against what the host has available at this moment,
    /// with the store's entries leaving first when that is short). A buffer that is refused
    /// is an error with the numbers, and the engine serves without the tier.
    pub fn calibrate(&mut self, copier: &mut dyn Copier) -> Result<(), String> {
        let bytes = hc_calibrate_bytes(CALIBRATE_BYTES, self.rows.cache_bytes());
        let mut leave = {
            let mut store = self.cache.inner.lock().unwrap_or_else(|e| e.into_inner());
            let leave = store.admit_beside(
                "the buffer of its first copies",
                crate::hostmem::AnonMap::mapped_len(bytes),
            );
            self.cache.publish(&store);
            leave?
        };
        let mut map = crate::hostmem::AnonMap::new(bytes, &mut leave)?;
        let pieces = [Piece {
            dev: self.rows.caches[0],
            host: map.as_mut_ptr(),
            len: bytes as usize,
        }];
        let (_, down) = copier.to_host(&pieces, &mut || false)?;
        let up = copier.to_device(&pieces)?;
        self.rates.to_host = down.rate();
        self.rates.to_device = up.rate();
        Ok(())
    }

    /// Start of a request, before its prefix plan. Copies out kept positions the request is
    /// about to drop when they are not saved yet, then restores from host memory when an entry
    /// offers a later resume point than the device and the copy is worth it. Returns what
    /// [`PrefixCache::plan_from`] takes: `(common token prefix, shared rows)` of the prompt and
    /// the restored entry's prompt, or `None` when nothing was restored.
    pub fn begin_request(
        &mut self,
        table: &mut PrefixCache,
        copier: &mut dyn Copier,
        prompt: &[i32],
        images: &[ImageId],
        wanted: bool,
        rows_zeroed: bool,
    ) -> Result<Option<(u64, u64)>, String> {
        self.last = TierReport::default();
        self.restored = false;
        // [lanes] nothing of an entry another lane rewrote since is trusted
        self.void_rewritten();
        if rows_zeroed {
            // the reset zeroed the K/V rows: nothing on the device equals anything saved
            self.clean.clear();
            self.ring_owner = None;
            self.unsaved = false;
        }
        let peek = table.peek(prompt, images, wanted, rows_zeroed);
        let kept_rows = table.kept_rows();
        if hc_must_flush(self.unsaved, self.last_reused, peek.cap, kept_rows) {
            let rep = self.save(table, copier, &mut || false)?;
            self.last.save_ms = rep.ms;
            self.last.save_bytes = rep.bytes;
        } else if hc_loses_unsaved(self.unsaved, peek.cap, kept_rows) {
            // The request drops kept positions of a prompt that was only ever prefilled once:
            // it does not pay for their save. A save of them that a request cut short has left
            // rows with no position to resume from; they go.
            if let Some(id) = self.partial.take() {
                let mut store = self.cache.inner.lock().unwrap_or_else(|e| e.into_inner());
                store.drop_unresumable(id);
                store.settle();
                self.cache.publish(&store);
                self.clean.remove(&id);
            }
            self.unsaved = false;
        }

        // a request that may take nothing over from the device (a cold one, or one after the
        // K/V rows were zeroed: its plan forgets the table) gets nothing copied back for it
        if !pc_request_may_reuse(wanted, rows_zeroed, table.enabled()) {
            return Ok(None);
        }
        let tile = self.rows.tile;
        let cache = Arc::clone(&self.cache);
        let mut store = cache.inner.lock().unwrap_or_else(|e| e.into_inner());
        self.clean.retain(|id, _| store.contains(*id));
        self.seen.retain(|id, _| store.contains(*id));
        // (the store first, then the write counts: the order every share takes the two locks in)
        self.writes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|id, _| store.contains(*id));
        let Some(found) = store.find(self.id, tile, prompt, images, wanted, table.enabled()) else {
            return Ok(None);
        };
        // what the restore would copy: the entry's tiles the device no longer holds, the state
        // at the resume point, and the drafter rings unless they are still this entry's
        let (lo, hi) = self.clean.get(&found.entry).copied().unwrap_or((0, 0));
        let linked = self.clean.contains_key(&found.entry);
        let copy_tiles: Vec<u64> = (0..found.tiles)
            .filter(|&t| !hc_tile_clean(linked, t, lo, hi))
            .collect();
        let ring_copy = hc_ring_restore(
            self.ring_bytes,
            store.ring_serves(found.entry, found.tiles),
            self.ring_owner == Some(found.entry),
        );
        let bytes = hc_restore_bytes(
            copy_tiles.len() as u64,
            self.rows.tile_bytes(),
            self.state_bytes,
            ring_copy,
            self.ring_bytes,
        );
        let gained = found.pos.saturating_sub(peek.resume);
        if !hc_copy_wins(bytes, self.rates.to_device, gained, self.rates.prefill_rows) {
            return Ok(None);
        }
        let Some(view) = store.begin_read(found.entry, found.tiles) else {
            return Ok(None);
        };
        let t0 = Instant::now();
        // the table describes the restored rows from here on; the state's slot is in the
        // middle of a write until the copy has ended
        let ring_from = hc_ring_floor(
            self.ring_bytes,
            view.ring.is_some(),
            view.ring.map_or(0, |(_, from)| from),
            found.pos,
        );
        let slot = table.adopt_restored(&view.tokens, &view.images, found.tiles, ring_from);
        let pieces = (|| -> Result<Vec<Piece>, String> {
            let mut pieces: Vec<Piece> = Vec::new();
            for &t in &copy_tiles {
                self.rows
                    .tile_pieces(t, view.rows[t as usize] as *mut u8, &mut pieces)?;
            }
            if let Some(s) = slot {
                pieces.extend(range_pieces(
                    &self.slots[s],
                    view.state as *mut u8,
                    self.state_bytes,
                )?);
            }
            if let (true, Some((ptr, _))) = (ring_copy, view.ring) {
                pieces.extend(range_pieces(&self.ring, ptr as *mut u8, self.ring_bytes)?);
            }
            Ok(pieces)
        })();
        let res = pieces.and_then(|pieces| copier.to_device(&pieces));
        match res {
            Err(e) => {
                // nothing of a copy that failed may be read later: not the entry, not the
                // device rows it was to fill
                store.end_read(found.entry, false);
                store.settle();
                cache.publish(&store);
                table.restore_done(0);
                table.clear();
                self.clean.clear();
                self.ring_owner = None;
                Err(e)
            }
            Ok(st) => {
                table.restore_done(found.tiles);
                store.end_read(found.entry, true);
                // the tiles just written are this entry's and no other's
                for (id, iv) in self.clean.iter_mut() {
                    if *id != found.entry {
                        *iv = (
                            hc_clean_lo(iv.0, iv.1, 0, found.tiles),
                            hc_clean_hi(iv.0, iv.1, 0, found.tiles),
                        );
                    }
                }
                self.clean.insert(
                    found.entry,
                    (
                        hc_clean_join_lo(lo, hi, 0, found.tiles),
                        hc_clean_join_hi(lo, hi, 0, found.tiles),
                    ),
                );
                if ring_copy {
                    self.ring_owner = Some(found.entry);
                } else if self.ring_owner != Some(found.entry) {
                    self.ring_owner = None;
                }
                self.note_read(found.entry);
                Self::note_rate(&mut self.rates.to_device, &st);
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let s = store.stats_mut();
                s.restores += 1;
                s.restore_bytes += st.bytes;
                s.restore_s += ms / 1e3;
                store.settle();
                cache.publish(&store);
                self.restored = true;
                self.last.restore_ms = ms;
                self.last.restore_bytes = st.bytes;
                self.last.restore_rows = gained;
                if self.verbose {
                    eprintln!(
                        "[cache-ram] restored entry {}: resume at row {} ({} of {} tiles copied, {:.1} MiB in {:.1} ms, {:.2} GiB/s; the device alone offered row {})",
                        found.entry,
                        found.pos,
                        copy_tiles.len(),
                        found.tiles,
                        st.bytes as f64 / (1u64 << 20) as f64,
                        ms,
                        st.rate() as f64 / (1u64 << 30) as f64,
                        peek.resume,
                    );
                }
                Ok(Some((found.lcp, found.cap)))
            }
        }
    }

    /// End of a request: what it wrote no longer equals anything saved, and what it left on the
    /// device is to be saved. `resume`: the row its prefill started at; `written_end`: one past
    /// the last row it may have written; `prefilled` rows took `prefill_s` seconds.
    pub fn end_request(
        &mut self,
        ok: bool,
        resume: u64,
        written_end: u64,
        prefilled: u64,
        prefill_s: f64,
    ) {
        if !ok {
            // the next reset zeroes the rows it may have written
            self.clean.clear();
            self.ring_owner = None;
            self.unsaved = false;
            return;
        }
        self.last.tier = Tier::of(self.restored, resume);
        self.last_reused = resume;
        let tile = self.rows.tile;
        let (w0, w1) = (
            hc_written_lo(resume, tile),
            hc_written_hi(written_end, tile),
        );
        for iv in self.clean.values_mut() {
            *iv = (
                hc_clean_lo(iv.0, iv.1, w0, w1),
                hc_clean_hi(iv.0, iv.1, w0, w1),
            );
        }
        self.ring_owner = None;
        self.unsaved = true;
        // the prefill rate of this engine, from a prefill of at least one whole tile
        if prefilled >= tile && prefill_s > 0.0 {
            self.rates.prefill_rows = (prefilled as f64 / prefill_s) as u64;
        }
    }

    /// Copies what the device holds for its prompt and host memory lacks: the rows of the tiles
    /// up to the last kept position, the state at every kept position, the drafter rings.
    /// `stop` is asked between pieces; a save it cuts short has committed what finished and
    /// goes on at the next call.
    pub fn save(
        &mut self,
        table: &PrefixCache,
        copier: &mut dyn Copier,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<SaveReport, String> {
        let mut rep = SaveReport::default();
        if !self.unsaved {
            rep.complete = true;
            return Ok(rep);
        }
        // [lanes] nothing of an entry another lane rewrote since is trusted (the save extends
        // what this lane's record says the entry already has)
        self.void_rewritten();
        let kept = table.held_slots();
        let (held, images) = table.held();
        let tile = self.rows.tile;
        let Some(&(upto, _)) = kept.last() else {
            self.unsaved = false;
            rep.complete = true;
            return Ok(rep);
        };
        if !hc_save_ok(upto, tile, held.len() as u64, self.rows.tiles()) {
            self.unsaved = false;
            return Ok(rep);
        }
        let t0 = Instant::now();
        let tile_bytes = self.rows.tile_bytes();
        let per_tile = self.rows.pieces_per_tile();
        let cache = Arc::clone(&self.cache);
        let mut store = cache.inner.lock().unwrap_or_else(|e| e.into_inner());
        self.clean.retain(|id, _| store.contains(*id));
        self.seen.retain(|id, _| store.contains(*id));
        // (the store first, then the write counts: the order every share takes the two locks in)
        self.writes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|id, _| store.contains(*id));
        // the store charges the save for what it adds to the entry it continues, and for
        // nothing that entry keeps
        let kept_tiles: Vec<u64> = kept.iter().map(|&(k, _)| k).collect();
        let Some(target) = store.open_save(
            self.id,
            tile,
            tile_bytes,
            held,
            images,
            upto,
            &kept_tiles,
            table.slot_count(),
            self.state_bytes,
            self.ring_bytes,
        ) else {
            // no room for this prefix at all: not asked again until the device holds another
            self.unsaved = false;
            store.settle();
            cache.publish(&store);
            return Ok(rep);
        };
        let entry = target.entry;
        // [lanes] the entry changes: every other lane's record of it is void from now on
        self.note_write(entry);
        // an entry cut to the tiles the prompt shares: its tiles past them are gone
        if let Some(iv) = self.clean.get_mut(&entry) {
            *iv = (
                hc_clean_lo(iv.0, iv.1, target.have_tiles, u64::MAX),
                hc_clean_hi(iv.0, iv.1, target.have_tiles, u64::MAX),
            );
        }
        let mut complete = true;
        let fail = |store: &mut HostCache, this: &mut Self, e: String| -> String {
            store.fail(entry);
            this.clean.remove(&entry);
            if this.ring_owner == Some(entry) {
                this.ring_owner = None;
            }
            store.settle();
            cache.publish(store);
            e
        };

        // rows, tile by tile: a round of the tiles the entry still lacks (`hc_new_tiles`), until
        // none is left or a round stops short
        let mut have = target.have_tiles;
        while complete {
            let n = hc_new_tiles(upto, have).min(ROUND_TILES);
            if n == 0 {
                break;
            }
            let Some(segs) = store.begin_rows(entry, n, tile_bytes) else {
                complete = false;
                break;
            };
            let mut pieces: Vec<Piece> = Vec::with_capacity(n as usize * per_tile);
            for (i, &seg) in segs.iter().enumerate() {
                if let Err(e) = self.rows.tile_pieces(have + i as u64, seg, &mut pieces) {
                    return Err(fail(&mut store, self, e));
                }
            }
            let (done, st) = match copier.to_host(&pieces, stop) {
                Ok(x) => x,
                Err(e) => return Err(fail(&mut store, self, e)),
            };
            // the tiles of the batch whose pieces all arrived (`hc_tile_copied`), in order
            let done_tiles = (0..n)
                .take_while(|&i| hc_tile_copied(i, per_tile as u64, done as u64))
                .count() as u64;
            let now = store.end_rows(entry, have + done_tiles);
            rep.bytes += st.bytes;
            Self::note_rate(&mut self.rates.to_host, &st);
            if hc_rows_added(have, now) {
                let (lo, hi) = self.clean.get(&entry).copied().unwrap_or((have, have));
                self.clean.insert(
                    entry,
                    (
                        hc_clean_join_lo(lo, hi, have, now),
                        hc_clean_join_hi(lo, hi, have, now),
                    ),
                );
            }
            if !hc_rows_complete(now, have + n) {
                complete = false;
                rep.yielded = true;
            }
            have = now;
        }

        // the state at every kept position the entry's rows reach
        if complete && store.contains(entry) {
            // the states the entry lacks and will keep (`hc_state_due`); one that would leave
            // again when the entry is trimmed below is not copied
            let due = crate::cache_ram::states_due(
                &store.states_of(entry),
                &kept_tiles,
                store.tiles_of(entry),
                table.slot_count(),
            );
            for &(k, slot) in &kept {
                if !due.contains(&k) {
                    continue;
                }
                if stop() {
                    complete = false;
                    rep.yielded = true;
                    break;
                }
                let Some(ptr) = store.begin_piece(entry, self.state_bytes) else {
                    complete = false;
                    break;
                };
                let pieces = match range_pieces(&self.slots[slot], ptr, self.state_bytes) {
                    Ok(p) => p,
                    Err(e) => return Err(fail(&mut store, self, e)),
                };
                let (done, st) = match copier.to_host(&pieces, stop) {
                    Ok(x) => x,
                    Err(e) => return Err(fail(&mut store, self, e)),
                };
                let whole = store.end_state(entry, k, done as u64, pieces.len() as u64);
                rep.bytes += st.bytes;
                if !whole {
                    complete = false;
                    rep.yielded = true;
                    break;
                }
            }
            store.trim_states(entry, table.slot_count(), upto);
        }

        // The drafter rings, whole, when the entry's image falls short of what the entry keeps
        // (`hc_ring_due`). That is read from the entry, not from what this call added: a save
        // that gave way before or during this copy takes the image at the next call, although
        // that call finds no tile and no position left to copy.
        let image = store.ring_tiles(entry);
        if store.contains(entry)
            && hc_ring_due(
                self.ring_bytes,
                complete,
                image.is_some(),
                image.unwrap_or(0),
                store.last_state_of(entry),
            )
        {
            if stop() {
                complete = false;
                rep.yielded = true;
            } else if let Some(ptr) = store.begin_piece(entry, self.ring_bytes) {
                let pieces = match range_pieces(&self.ring, ptr, self.ring_bytes) {
                    Ok(p) => p,
                    Err(e) => return Err(fail(&mut store, self, e)),
                };
                let (done, st) = match copier.to_host(&pieces, stop) {
                    Ok(x) => x,
                    Err(e) => return Err(fail(&mut store, self, e)),
                };
                let whole =
                    store.end_ring(entry, table.ring_from(), done as u64, pieces.len() as u64);
                rep.bytes += st.bytes;
                if whole {
                    self.ring_owner = Some(entry);
                } else {
                    complete = false;
                    rep.yielded = true;
                }
            } else {
                complete = false;
            }
        }

        // complete: nothing was cut short, and the image covers the entry's last kept position
        // (`hc_saved_whole`)
        let image = store.ring_tiles(entry);
        rep.complete = store.contains(entry)
            && hc_saved_whole(
                complete,
                self.ring_bytes,
                image.is_some(),
                image.unwrap_or(0),
                store.last_state_of(entry),
            );
        rep.ms = t0.elapsed().as_secs_f64() * 1e3;
        // a save cut short goes on at the next call; one that found no room does not ask again
        self.unsaved = rep.yielded;
        self.partial = rep.yielded.then_some(entry);
        let s = store.stats_mut();
        if rep.bytes > 0 {
            s.saves += 1;
            s.save_bytes += rep.bytes;
            s.save_s += rep.ms / 1e3;
        }
        if rep.yielded {
            s.yielded += 1;
        }
        if self.verbose && (rep.bytes > 0 || !rep.complete) {
            eprintln!(
                "[cache-ram] saved into entry {entry}: {} rows, {} kept position(s){}; {:.1} MiB copied in {:.1} ms ({:.2} GiB/s){}; the cache holds {:.2} of {:.2} GiB in {} entr{}",
                pc_boundary(store.tiles_of(entry), tile),
                store.states_of(entry).len(),
                if store.has_ring(entry) { ", drafter rings" } else { "" },
                rep.bytes as f64 / (1u64 << 20) as f64,
                rep.ms,
                if rep.ms > 0.0 { rep.bytes as f64 / (1u64 << 30) as f64 / (rep.ms / 1e3) } else { 0.0 },
                if rep.yielded {
                    " (a request took the card: the rest follows)"
                } else if !rep.complete {
                    " (no room for the rest)"
                } else {
                    ""
                },
                store.held_bytes() as f64 / (1u64 << 30) as f64,
                store.budget_bytes() as f64 / (1u64 << 30) as f64,
                store.stats().entries,
                if store.stats().entries == 1 { "y" } else { "ies" },
            );
        }
        store.settle();
        cache.publish(&store);
        Ok(rep)
    }
}

/// What must be the same on two engines for the bytes of a saved entry to mean the same on
/// both: the K/V caches (how many, their names in the order their rows are laid out in a
/// segment, their chunks, heads, rows and row bytes), the prefill tile, and the sizes of the
/// ranges of a kept position's state and of the drafter's rings, each in copy order. Never a
/// device address: an engine that is loaded again (after the release of the card, after
/// another model) may get its buffers elsewhere, and what it saved before is still its own.
pub fn layout_fingerprint(
    cache_names: &[String],
    rows: &RowLayout,
    state: &[(u64, u64)],
    ring: &[(u64, u64)],
) -> String {
    let sizes = |ranges: &[(u64, u64)]| {
        ranges
            .iter()
            .map(|&(_, bytes)| bytes.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "caches={}[{}]|chunks={}|chunk_rows={}|heads={}|row_bytes={}|tile={}|state=[{}]|ring=[{}]",
        rows.caches.len(),
        cache_names.join(","),
        rows.chunks,
        rows.chunk_rows,
        rows.heads,
        rows.row_bytes,
        rows.tile,
        sizes(state),
        sizes(ring),
    )
}

/// The identity the store stamps entries with: a digest of what decides the bytes of a row or
/// of a state on this engine (`engine`, [`engine_text`]) and of the layout the copies use
/// (`layout`, [`layout_fingerprint`]).
pub fn identity_of(engine: &str, layout: &str) -> EngineId {
    digest128(format!("{engine}|layout={layout}").as_bytes())
}

/// Names, sizes and modification times of the files of `dir` (`all`), or of its weight and
/// configuration files only.
fn files_of(dir: &std::path::Path, all: bool) -> String {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let keep = all
                || name.ends_with(".safetensors")
                || name.ends_with(".json")
                || name.ends_with(".txt");
            let md = e.metadata().ok()?;
            if !md.is_file() {
                return None;
            }
            let mtime = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            keep.then(|| format!("{name}:{}:{mtime}", md.len()))
        })
        .collect();
    v.sort();
    v.join(",")
}

/// The FP8 files of a DSpine drafter in `draft_dir`, as a load finds them: its activation-scale
/// table (the one the run options name, else the one in its default FP8 directory) by its
/// content, and the files of that table's directory, which is where the drafter keeps its
/// quantised matrices, by name, size and modification time. A drafter only proposes tokens, so
/// none of this can change an answer; it decides what the saved context rings hold.
pub(super) fn drafter_fp8_text(
    draft_dir: &std::path::Path,
    step: &crate::engine::step_opts::StepOpts,
) -> String {
    let table = spec::dspine::act_table_path(draft_dir, step);
    let content = match std::fs::read(&table) {
        Ok(bytes) => {
            let d = digest128(&bytes);
            format!("{}:{:016x}{:016x}", bytes.len(), d[0], d[1])
        }
        Err(_) => "absent".into(),
    };
    format!(
        "table={content}|files={}",
        table.parent().map_or(String::new(), |d| files_of(d, true))
    )
}

/// Everything that decides the bytes of a row or of a state on a loaded engine, as text: the
/// model's files, the precisions, the context and the tile, the kernels, the drafter, the
/// vision tower.
///
/// Every input is named as the load resolved it, never only as the setting that may name it:
/// the FP8 activation-scale table and the cached FP8 matrices come from the load itself
/// (`Qwen3Engine::fp8_id`: the table as parsed, the cache files as used), wherever the
/// setting, an environment variable or the default place pointed; the prefill levers are the
/// resolved ones; the drafter's FP8 files are read where its load reads them.
pub(super) fn engine_text(cfg: &Qwen3Config, eng: &Qwen3Engine) -> String {
    let drafter = cfg.spec.as_ref().map(|sc| {
        let dir = match &sc.method {
            spec::SpecMethod::DSpine { draft_model } => format!(
                "{}|fp8={}",
                files_of(draft_model, false),
                drafter_fp8_text(draft_model, &cfg.step)
            ),
            spec::SpecMethod::DFlash2 { draft_model }
            | spec::SpecMethod::Assistant { draft_model } => files_of(draft_model, false),
            _ => String::new(),
        };
        format!("{sc:?}|{dir}")
    });
    format!(
        "model={}|files={}|spec={:?}|ctx={}|tile={}|prefill={:?}|precision={:?}|fp8={}|kv8={}|moe_routed={}|reset={:?}|step={:?}|kernels={:?}|kernel_files={}|driver={}|drafter={:?}|ctx_rows={}|vision={}/{}/{}",
        cfg.model_dir.display(),
        files_of(&cfg.model_dir, false),
        eng.spec,
        eng.max_ctx,
        eng.prefill_tile,
        // the levers as resolved (`--prefill`, else `GAUDIUM_PREFILL`, else the defaults)
        eng.pf.opts,
        cfg.precision,
        eng.fp8_id,
        // [kv8] the K/V dtype is in `spec`; the scale tables' calibration here
        eng.kv8_id,
        cfg.moe_routed,
        cfg.request_reset,
        cfg.step,
        eng.elf_hashes,
        // every kernel and glue file a recipe may be built from (the tower's and the
        // drafter's too): a rebuilt kernel is another engine
        files_of(&cfg.kernels_dir, true),
        eng.driver_version,
        drafter,
        eng.prefix.ctx_rows(),
        cfg.vision_max_patches,
        cfg.vision_max_tokens,
        cfg.vision_gelu_bf16,
    )
}

/// Builds the tier of a loaded engine and its transfer ring, when the configuration has a
/// store, the engine reuses prefixes and its buffers can be copied tile by tile. One line says
/// what it is, or why this engine has none.
pub(super) fn attach(eng: &mut Qwen3Engine, cfg: &Qwen3Config) -> Result<(), String> {
    let Some(cache) = cfg.host_cache.clone() else {
        return Ok(());
    };
    if !eng.prefix.enabled() {
        if cfg.verbose {
            eprintln!("[cache-ram] off for this engine: it keeps no position to resume a prefill from (prefix reuse is off)");
        }
        return Ok(());
    }
    let (cache_names, rows, slots, ring) = match layout_of_active(eng) {
        Ok(l) => l,
        Err(why) => {
            eprintln!("[cache-ram] off for this engine: {why}");
            return Ok(());
        }
    };
    // the identity never holds a device address (`layout_fingerprint`)
    let layout = layout_fingerprint(
        &cache_names,
        &rows,
        slots.first().map_or(&[][..], |s| s.as_slice()),
        &ring,
    );
    // [lanes] every other lane's share of the tier: its own device addresses, the same layout
    // (and so the same identity: an entry one lane saved is restored into any lane)
    let n_lanes = eng.lanes.as_ref().map_or(1, |rt| rt.lanes());
    let mut others = Vec::with_capacity(n_lanes.saturating_sub(1));
    for lane in 1..n_lanes {
        super::lanes_rt::activate(eng, lane);
        let l = layout_of_active(eng).and_then(|(names, rows, slots, ring)| {
            let same = layout_fingerprint(
                &names,
                &rows,
                slots.first().map_or(&[][..], |s| s.as_slice()),
                &ring,
            ) == layout;
            if same {
                Ok((rows, slots, ring))
            } else {
                Err(format!("lane {lane}'s device layout differs from lane 0's"))
            }
        });
        others.push(l);
    }
    if n_lanes > 1 {
        super::lanes_rt::activate(eng, 0);
    }
    let others = match others.into_iter().collect::<Result<Vec<_>, String>>() {
        Ok(o) => o,
        Err(why) => {
            eprintln!("[cache-ram] off for this engine: {why}");
            return Ok(());
        }
    };
    let id = identity_of(&engine_text(cfg, eng), &layout);
    // The ring: a few buffers of untouched memory, each mapped for the device on its own (the
    // one pinned host memory of the tier, planned when the server started). An engine that
    // cannot get it serves without the tier: no model fails to load for lack of a cache.
    //
    // Its memory is asked for now, against what the host and the control groups of the
    // process have left at this moment (the plan at start was made before any model was
    // loaded): the store's entries leave first when that is short, and a ring that still
    // does not fit is refused, with the numbers, before a page of it exists.
    let ring_leave = {
        let mut store = cache.inner.lock().unwrap_or_else(|e| e.into_inner());
        let leave = store.admit_beside(
            "its pinned transfer ring",
            XferRing::planned_bytes(
                crate::synapse::xfer::RING_BUFS,
                crate::synapse::xfer::RING_BUF_BYTES,
            ),
        );
        cache.publish(&store);
        leave
    };
    let ring_leave = match ring_leave {
        Ok(leave) => leave,
        Err(e) => {
            eprintln!(
                "[cache-ram] off for this engine: {e} ({})",
                crate::hostmem::HostMemory::read().describe()
            );
            return Ok(());
        }
    };
    let t0 = Instant::now();
    let mut xfer = match XferRing::new(
        &eng.dev,
        crate::synapse::xfer::RING_BUFS,
        crate::synapse::xfer::RING_BUF_BYTES,
        crate::synapse::xfer::RingKind::MappedHuge,
        crate::synapse::xfer::COPY_THREADS,
        ring_leave,
    ) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[cache-ram] off for this engine: the pinned transfer ring could not be mapped ({e})");
            return Ok(());
        }
    };
    let ring_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut tier = HostTier::new(Arc::clone(&cache), id, rows, slots, ring, cfg.verbose);
    if let Err(e) = tier.calibrate(&mut RingCopier {
        ring: &mut xfer,
        dev: &eng.dev,
    }) {
        eprintln!(
            "[cache-ram] off for this engine: its first copies through the transfer ring could not be made: {e} ({})",
            crate::hostmem::HostMemory::read().describe()
        );
        return Ok(());
    }
    if cfg.verbose {
        let st = cache.stats();
        let (per_row, per_state) = tier.bytes_per_row_and_state();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "[cache-ram] on: budget {:.2} GiB ({:.2} GiB held in {} entr{}); {:.1} KiB per saved row + {:.1} MiB per kept position{}; transfer ring {} MiB pinned in {:.1} ms; measured {:.2} GiB/s to host, {:.2} GiB/s to the device",
            gib(st.budget_bytes),
            gib(st.held_bytes),
            st.entries,
            if st.entries == 1 { "y" } else { "ies" },
            per_row as f64 / 1024.0,
            per_state as f64 / (1u64 << 20) as f64,
            if tier.ring_bytes > 0 {
                format!(" + {:.1} MiB of drafter rings", tier.ring_bytes as f64 / (1u64 << 20) as f64)
            } else {
                String::new()
            },
            xfer.pinned_bytes() >> 20,
            ring_ms,
            gib(tier.rates.to_host),
            gib(tier.rates.to_device),
        );
    }
    let rates = tier.facts().rates;
    eng.xfer = Some(xfer);
    eng.host = Some(tier);
    let first = eng.host.as_ref().expect("lane 0's share");
    let shares: Vec<HostTier> = others
        .into_iter()
        .map(|(rows, slots, ring)| {
            let mut t = HostTier::new(Arc::clone(&cache), id, rows, slots, ring, cfg.verbose)
                .sharing_writes_of(first);
            t.set_rates(rates);
            t
        })
        .collect();
    for (i, t) in shares.into_iter().enumerate() {
        super::lanes_rt::activate(eng, i + 1);
        eng.host = Some(t);
    }
    if n_lanes > 1 {
        super::lanes_rt::activate(eng, 0);
        if cfg.verbose {
            eprintln!("[cache-ram] every one of the {n_lanes} lanes saves and restores through the tier");
        }
    }
    Ok(())
}

/// The device layout the host tier copies for the active lane: the trunk K/V cache names and rows,
/// the kept positions' state ranges and the drafter's context rings.
#[allow(clippy::type_complexity)]
fn layout_of_active(
    eng: &Qwen3Engine,
) -> Result<(Vec<String>, RowLayout, Vec<Vec<(u64, u64)>>, Vec<(u64, u64)>), String> {
    // the trunk K/V caches: one size, chunk-major, `n_kv_heads` heads of BF16 rows
    let cache_names: Vec<String> = eng.kv_addr.keys().cloned().collect();
    let caches: Vec<u64> = eng.kv_addr.values().map(|&(a, _)| a).collect();
    let cache_bytes = eng.kv_addr.values().next().map_or(0, |&(_, b)| b);
    if eng.kv_addr.values().any(|&(_, b)| b != cache_bytes) {
        return Err("its K/V caches differ in size".into());
    }
    // [kv8] one head's row of one position in the caches' own dtype, and its V block
    let kv_dt = super::kv8::trunk_dtype(&eng.spec);
    let rows = RowLayout::with_block(
        caches,
        cache_bytes,
        eng.max_chunks,
        KV_CHUNK,
        eng.spec.n_kv_heads,
        crate::core::kv_cache::row_bytes(1, eng.spec.head_dim, kv_dt),
        eng.prefill_tile,
        crate::verified::kv_vperm_rows(kv_dt),
    )?;
    let slots: Vec<Vec<(u64, u64)>> = eng
        .prefix_slots
        .iter()
        .map(|m| {
            m.iter()
                .flat_map(|(l, &(s, cs))| {
                    let st = &eng.gdn_state[l];
                    [(s, st.s_bytes), (cs, st.cs_bytes)]
                })
                .collect()
        })
        .collect();
    let ring: Vec<(u64, u64)> = match eng.specdec.as_ref() {
        Some(sd) if sd.context_rows() > 0 => sd.cache_regions(),
        _ => Vec::new(),
    };
    Ok((cache_names, rows, slots, ring))
}

/// The end of the engine's request on the active lane: what it wrote no longer equals anything
/// saved; what it left is to be saved (after the response: `idle_work`). Returns the tier's report
/// of the request.
pub(super) fn end_request(
    eng: &mut Qwen3Engine,
    res: &Result<Qwen3GenReport, String>,
    reused: u64,
    written_end: u64,
) -> Option<TierReport> {
    let tier = eng.host.as_mut()?;
    let (prefilled, secs) = res.as_ref().map_or((0, 0.0), |r| {
        (
            (r.prompt_tokens as u64).saturating_sub(reused),
            // the restore is inside the prefill clock, not part of the prefill itself
            (r.ttft_ms - tier.last().restore_ms - tier.last().save_ms).max(0.0) / 1e3,
        )
    });
    let cancelled = res.as_ref().is_ok_and(|r| r.stopped_by_cancel);
    tier.end_request(
        res.is_ok(),
        reused,
        written_end,
        if cancelled { 0 } else { prefilled },
        secs,
    );
    Some(tier.last())
}

/// [`HostTier::begin_request`] for the engine's request.
pub(super) fn begin_request(
    eng: &mut Qwen3Engine,
    prompt: &[i32],
    wanted: bool,
    rows_zeroed: bool,
) -> Result<Option<(u64, u64)>, String> {
    let Qwen3Engine {
        host,
        xfer,
        prefix,
        prefix_images,
        dev,
        ..
    } = eng;
    let (Some(tier), Some(ring)) = (host.as_mut(), xfer.as_mut()) else {
        return Ok(None);
    };
    tier.begin_request(
        prefix,
        &mut RingCopier { ring, dev },
        prompt,
        prefix_images,
        wanted,
        rows_zeroed,
    )
}

/// The save between requests (`stop`: a request waits) and before the card is released.
pub(super) fn save(
    eng: &mut Qwen3Engine,
    stop: &mut dyn FnMut() -> bool,
) -> Result<SaveReport, String> {
    let Qwen3Engine {
        host,
        xfer,
        prefix,
        dev,
        ..
    } = eng;
    let (Some(tier), Some(ring)) = (host.as_mut(), xfer.as_mut()) else {
        return Ok(SaveReport {
            complete: true,
            ..SaveReport::default()
        });
    };
    tier.save(prefix, &mut RingCopier { ring, dev }, stop)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: u64 = 512;

    /// A device made of host memory: K/V caches, state slots and drafter rings at made-up
    /// addresses. "Prefilling" a prompt writes, for every row and state, bytes that depend on
    /// the tokens up to there, so two prompts agree exactly where a real prefill would.
    struct FakeDev {
        mem: Vec<u8>,
        layout: RowLayout,
        slots: Vec<Vec<(u64, u64)>>,
        ring: Vec<(u64, u64)>,
        /// bytes copied to the device / to the host
        up: u64,
        down: u64,
        fail_next: bool,
    }

    const BASE: u64 = 0x1000_0000;

    impl FakeDev {
        fn new(n_slots: usize, with_ring: bool) -> Self {
            // 2 caches x 2 chunks x 2 heads x 4096 rows x 16 bytes
            let (chunks, chunk_rows, heads, row_bytes) = (2u64, 4096u64, 2u64, 16u64);
            let cache_bytes = chunks * heads * chunk_rows * row_bytes;
            let mut at = BASE;
            let mut take = |bytes: u64| {
                let a = at;
                at += bytes;
                a
            };
            let caches = vec![take(cache_bytes), take(cache_bytes)];
            let slots = (0..n_slots)
                .map(|_| vec![(take(4096), 4096u64), (take(512), 512u64)])
                .collect();
            let ring = if with_ring {
                vec![(take(8192), 8192u64)]
            } else {
                Vec::new()
            };
            let layout =
                RowLayout::new(caches, cache_bytes, chunks, chunk_rows, heads, row_bytes, T)
                    .unwrap();
            Self {
                mem: vec![0u8; (at - BASE) as usize],
                layout,
                slots,
                ring,
                up: 0,
                down: 0,
                fail_next: false,
            }
        }

        fn at(&mut self, dev: u64, len: usize) -> &mut [u8] {
            let o = (dev - BASE) as usize;
            &mut self.mem[o..o + len]
        }

        /// A value that depends on `tokens[..=pos]`.
        fn mark(tokens: &[i32], pos: usize) -> u64 {
            let mut h = 0x9E37_79B9_7F4A_7C15u64;
            for &t in &tokens[..=pos.min(tokens.len() - 1)] {
                h = (h ^ t as u32 as u64)
                    .wrapping_mul(0x100_0000_01B3)
                    .rotate_left(23);
            }
            h ^ pos as u64
        }

        /// Writes the rows `[from, to)` of `tokens` as a prefill would.
        fn write_rows(&mut self, tokens: &[i32], from: u64, to: u64) {
            let l = self.layout.clone();
            for row in from..to {
                let (chunk, in_chunk) = (row / l.chunk_rows, row % l.chunk_rows);
                let v = Self::mark(tokens, row as usize);
                for (c, &addr) in l.caches.iter().enumerate() {
                    for h in 0..l.heads {
                        let dev = addr
                            + chunk * l.heads * l.chunk_rows * l.row_bytes
                            + (h * l.chunk_rows + in_chunk) * l.row_bytes;
                        let word = v ^ ((c as u64) << 56) ^ (h << 48);
                        let bytes = [word.to_le_bytes(), (!word).to_le_bytes()].concat();
                        self.at(dev, l.row_bytes as usize).copy_from_slice(&bytes);
                    }
                }
            }
        }

        /// Writes the state after `tiles` whole tiles of `tokens` into slot `slot`.
        fn write_state(&mut self, tokens: &[i32], tiles: u64, slot: usize) {
            let v = Self::mark(tokens, (tiles * T) as usize - 1) ^ 0x5555;
            for (dev, bytes) in self.slots[slot].clone() {
                let fill: Vec<u8> = (0..bytes)
                    .map(|i| (v >> (8 * (i % 8))) as u8 ^ i as u8)
                    .collect();
                self.at(dev, bytes as usize).copy_from_slice(&fill);
            }
        }

        fn rows_of(&mut self, from: u64, to: u64) -> Vec<u8> {
            let l = self.layout.clone();
            let mut out = Vec::new();
            for t in from / T..to / T {
                let mut p = Vec::new();
                l.tile_pieces(t, std::ptr::null_mut(), &mut p).unwrap();
                for piece in p {
                    out.extend_from_slice(self.at(piece.dev, piece.len));
                }
            }
            out
        }

        fn state_of(&mut self, slot: usize) -> Vec<u8> {
            let mut out = Vec::new();
            for (dev, bytes) in self.slots[slot].clone() {
                out.extend_from_slice(self.at(dev, bytes as usize));
            }
            out
        }
    }

    impl Copier for FakeDev {
        fn to_host(
            &mut self,
            pieces: &[Piece],
            stop: &mut dyn FnMut() -> bool,
        ) -> Result<(usize, XferStats), String> {
            if std::mem::take(&mut self.fail_next) {
                return Err("device error".into());
            }
            let mut st = XferStats::default();
            let mut done = 0;
            for (i, p) in pieces.iter().enumerate() {
                // a "ring buffer" every 8 pieces
                if i % 8 == 0 && stop() {
                    break;
                }
                let src = self.at(p.dev, p.len).as_ptr();
                unsafe { std::ptr::copy_nonoverlapping(src, p.host, p.len) };
                st.bytes += p.len as u64;
                done = i + 1;
            }
            self.down += st.bytes;
            st.wall_s = 1e-3;
            Ok((done, st))
        }

        fn to_device(&mut self, pieces: &[Piece]) -> Result<XferStats, String> {
            if std::mem::take(&mut self.fail_next) {
                return Err("device error".into());
            }
            let mut st = XferStats::default();
            for p in pieces {
                let dst = self.at(p.dev, p.len).as_mut_ptr();
                unsafe { std::ptr::copy_nonoverlapping(p.host as *const u8, dst, p.len) };
                st.bytes += p.len as u64;
            }
            self.up += st.bytes;
            st.wall_s = 1e-3;
            Ok(st)
        }
    }

    fn prompt(n: usize, salt: i32) -> Vec<i32> {
        (0..n as i32).map(|i| i.wrapping_mul(31) ^ salt).collect()
    }

    struct Rig {
        dev: FakeDev,
        table: PrefixCache,
        tier: HostTier,
        cache: Arc<SharedCache>,
    }

    fn rig_on(cache: Arc<SharedCache>, n_slots: u64, with_ring: bool, id: EngineId) -> Rig {
        let dev = FakeDev::new(n_slots as usize, with_ring);
        let table = PrefixCache::new(T, n_slots, 1, true, if with_ring { 8960 } else { 0 });
        let mut tier = HostTier::new(
            Arc::clone(&cache),
            id,
            dev.layout.clone(),
            dev.slots.clone(),
            dev.ring.clone(),
            false,
        );
        // rates as measured on the card: a copy is far faster than a prefill
        tier.set_rates(Rates {
            to_host: 10 << 30,
            to_device: 16 << 30,
            prefill_rows: 6500,
        });
        Rig {
            dev,
            table,
            tier,
            cache,
        }
    }

    fn rig(budget: u64, n_slots: u64) -> Rig {
        rig_on(
            Arc::new(SharedCache::new(budget, 0, false, false)),
            n_slots,
            false,
            [7, 7],
        )
    }

    impl Rig {
        /// Runs one request as the engine does: the tier, the plan, a prefill that writes what
        /// a real one would, the end. Returns `(resume, tier report)`.
        fn request(&mut self, p: &[i32], wanted: bool) -> (u64, TierReport) {
            let restored = self
                .tier
                .begin_request(&mut self.table, &mut self.dev, p, &[], wanted, false)
                .unwrap();
            let plan = self.table.plan_from(p, &[], wanted, false, restored);
            let resume = plan.resume;
            self.dev.write_rows(p, resume, p.len() as u64);
            for &(k, slot) in &plan.captures {
                self.dev.write_state(p, k, slot);
            }
            self.table.tiles_done(p, &[], p.len() as u64 / T);
            self.table.prefill_end(p, &[], p.len() as u64);
            // some generated rows past the prompt, as decode writes them
            let written_end = p.len() as u64 + 40;
            let junk = prompt(written_end as usize, -99);
            self.dev.write_rows(&junk, p.len() as u64, written_end);
            self.table.request_end(true, written_end);
            self.tier
                .end_request(true, resume, written_end, p.len() as u64 - resume, 0.5);
            (resume, self.tier.last())
        }

        fn idle(&mut self) -> SaveReport {
            self.tier
                .save(&self.table, &mut self.dev, &mut || false)
                .unwrap()
        }

        /// The device rows and the resume state a cold prefill of `p` up to `resume` leaves.
        fn cold(p: &[i32], resume: u64, n_slots: usize) -> (Vec<u8>, Vec<u8>) {
            let mut d = FakeDev::new(n_slots, false);
            d.write_rows(p, 0, resume);
            d.write_state(p, resume / T, 0);
            (d.rows_of(0, resume), d.state_of(0))
        }

        /// The state the table resumes from for position `resume`.
        fn resume_state(&mut self, resume: u64) -> Vec<u8> {
            let slot = self
                .table
                .held_slots()
                .iter()
                .find(|&&(k, _)| k * T == resume)
                .map(|&(_, s)| s)
                .expect("a slot holds the resume position");
            self.dev.state_of(slot)
        }
    }

    #[test]
    fn a_lane_never_trusts_its_record_of_an_entry_another_lane_rewrote() {
        // two lanes' shares of one store (one engine): lane B's save cuts and rewrites the entry
        // lane A's device still matches in part
        let cache = Arc::new(SharedCache::new(256 << 20, 0, false, false));
        let mut a = rig_on(Arc::clone(&cache), 4, false, [7, 7]);
        let mut b = rig_on(Arc::clone(&cache), 4, false, [7, 7]);
        b.tier = std::mem::replace(
            &mut b.tier,
            HostTier::new(Arc::clone(&cache), [7, 7], b.dev.layout.clone(), b.dev.slots.clone(), Vec::new(), false),
        )
        .sharing_writes_of(&a.tier);
        let p = prompt(8 * 512 + 100, 1);
        a.request(&p, true);
        assert!(a.idle().complete);
        // B's conversation shares P's first four tiles, then differs
        let mut q = p[..4 * 512].to_vec();
        q.extend(prompt(4 * 512 + 100, 2));
        b.request(&q, true);
        assert!(b.idle().complete);
        assert_eq!(cache.stats().entries, 1, "B's save continued and cut the entry A saved");
        // Q's next turn lands on lane A, whose device still holds P's rows past the fourth tile
        let mut q2 = q.clone();
        q2.extend(prompt(300, 3));
        let restored = a
            .tier
            .begin_request(&mut a.table, &mut a.dev, &q2, &[], true, false)
            .unwrap();
        assert!(restored.is_some());
        let plan = a.table.plan_from(&q2, &[], true, false, restored);
        assert!(plan.resume > 4 * 512, "the entry gives Q's rows back past the shared ones");
        // the rows A resumes from are Q's, as a cold prefill of Q leaves them
        let (rows, state) = Rig::cold(&q2, plan.resume, 4);
        assert!(a.dev.rows_of(0, plan.resume) == rows);
        assert!(a.resume_state(plan.resume) == state);
    }

    #[test]
    fn a_conversation_comes_back_from_host_memory_after_another_took_the_card() {
        let mut r = rig(64 << 20, 4);
        let a1 = prompt(6 * 512 + 100, 1);
        assert_eq!(r.request(&a1, true).1.tier, Tier::Prefilled);
        assert!(r.idle().complete);
        // another conversation overwrites the first rows
        let b = prompt(3 * 512 + 50, 2);
        assert_eq!(r.request(&b, true).0, 0);
        r.idle();
        // the first one's next turn: the device has nothing of it, host memory does
        let mut a2 = a1.clone();
        a2.extend(prompt(700, 5));
        let up_before = r.dev.up;
        let restored = r
            .tier
            .begin_request(&mut r.table, &mut r.dev, &a2, &[], true, false)
            .unwrap();
        assert!(restored.is_some());
        let rep = r.tier.last();
        let plan = r.table.plan_from(&a2, &[], true, false, restored);
        assert_eq!(plan.resume, 6 * 512);
        // the rows and the state the prefill resumes from are the ones a cold prefill leaves
        let (rows, state) = Rig::cold(&a2, plan.resume, 4);
        assert!(r.dev.rows_of(0, plan.resume) == rows);
        assert!(r.resume_state(plan.resume) == state);
        // only the tiles the other conversation wrote were copied back (4 of 6: its prompt and
        // generated rows reach into the fourth), and one state
        let tile_bytes = r.dev.layout.tile_bytes();
        assert_eq!(r.dev.up - up_before, 4 * tile_bytes + 4608);
        assert_eq!(rep.restore_bytes, 4 * tile_bytes + 4608);
    }

    #[test]
    fn a_reloaded_engine_restores_what_its_predecessor_saved_and_another_engine_does_not() {
        let cache = Arc::new(SharedCache::new(64 << 20, 0, false, false));
        let mut r = rig_on(Arc::clone(&cache), 4, true, [7, 7]);
        let a1 = prompt(5 * 512 + 100, 1);
        r.request(&a1, true);
        assert!(r.idle().complete);
        assert_eq!(cache.stats().entries, 1);
        // the card is released: a new engine of the same identity, an empty device
        let mut r2 = rig_on(Arc::clone(&cache), 4, true, [7, 7]);
        let mut a2 = a1.clone();
        a2.extend(prompt(300, 5));
        let (resume, rep) = r2.request(&a2, true);
        assert_eq!((resume, rep.tier), (5 * 512, Tier::Host));
        // every tile, the state and the rings were copied
        let tile_bytes = r2.dev.layout.tile_bytes();
        assert_eq!(rep.restore_bytes, 5 * tile_bytes + 4608 + 8192);
        // an engine of another identity prefills
        let mut r3 = rig_on(Arc::clone(&cache), 4, true, [7, 8]);
        assert_eq!(r3.request(&a2, true), (0, TierReport::default()));
        // and so does a request that asks for a cold prefill
        let mut r4 = rig_on(Arc::clone(&cache), 4, true, [7, 7]);
        assert_eq!(r4.request(&a2, false).0, 0);
    }

    #[test]
    fn the_next_turn_copies_only_what_is_new_and_a_turn_on_the_device_copies_nothing_back() {
        let mut r = rig(64 << 20, 4);
        let a1 = prompt(4 * 512 + 100, 1);
        r.request(&a1, true);
        r.idle();
        let down1 = r.dev.down;
        let tile_bytes = r.dev.layout.tile_bytes();
        assert_eq!(down1, 4 * tile_bytes + 3 * 4608);
        // the next turn resumes on the device: no restore
        let mut a2 = a1.clone();
        a2.extend(prompt(1100, 5));
        let (resume, rep) = r.request(&a2, true);
        assert_eq!(
            (resume, rep.tier, rep.restore_bytes),
            (4 * 512, Tier::Device, 0)
        );
        assert_eq!(r.dev.up, 0);
        // its save copies the two new tiles and the new positions only
        let rep = r.idle();
        assert!(rep.complete);
        assert_eq!(r.dev.down - down1, rep.bytes);
        assert_eq!(rep.bytes, 2 * tile_bytes + 2 * 4608);
        assert_eq!(r.cache.stats().entries, 1);
        // nothing left to save: the next idle call copies nothing
        assert_eq!(r.idle().bytes, 0);
    }

    #[test]
    fn the_drafter_rings_are_copied_again_only_when_the_entry_grew() {
        let cache = Arc::new(SharedCache::new(64 << 20, 0, false, false));
        let mut r = rig_on(Arc::clone(&cache), 4, true, [7, 7]);
        let a1 = prompt(4 * 512 + 100, 1);
        r.request(&a1, true);
        let tile_bytes = r.dev.layout.tile_bytes();
        // the first save takes the rings with the rows and the positions
        assert_eq!(r.idle().bytes, 4 * tile_bytes + 3 * 4608 + 8192);
        // the same prompt again adds nothing: its save copies nothing, rings included
        r.request(&a1, true);
        let rep = r.idle();
        assert!(rep.complete);
        assert_eq!(rep.bytes, 0);
        // a turn that adds tiles takes the rings again
        let mut a2 = a1.clone();
        a2.extend(prompt(1100, 5));
        r.request(&a2, true);
        assert_eq!(r.idle().bytes, 2 * tile_bytes + 2 * 4608 + 8192);
        // an image serves the positions the entry had when it was taken, and no later one: a
        // save cut short after the new rows leaves the old image for the old positions only
        let mut a3 = a2.clone();
        a3.extend(prompt(1100, 6));
        r.request(&a3, true);
        let mut asked = 0;
        let rep = r
            .tier
            .save(&r.table, &mut r.dev, &mut || {
                asked += 1;
                asked > 3
            })
            .unwrap();
        assert!(rep.yielded);
        let store = cache.inner.lock().unwrap();
        let id = store.find([7, 7], T, &a3, &[], true, true).map(|f| f.entry);
        if let Some(id) = id {
            assert!(store.ring_serves(id, 6));
            assert!(!store.ring_serves(id, 8));
        }
    }

    #[test]
    fn a_save_cut_short_at_the_rings_takes_them_at_the_next_idle_time() {
        // A conversation whose first turn is saved (rows, positions, rings), after its second
        // turn: two new tiles, two new positions, and an image that covers the first turn only.
        let scene = || {
            let cache = Arc::new(SharedCache::new(64 << 20, 0, false, false));
            let mut r = rig_on(Arc::clone(&cache), 4, true, [7, 7]);
            let a1 = prompt(4 * 512 + 100, 1);
            r.request(&a1, true);
            assert!(r.idle().complete);
            let mut a2 = a1.clone();
            a2.extend(prompt(1100, 5));
            r.request(&a2, true);
            // what the drafter's rings hold now
            let (ring_at, ring_len) = r.dev.ring[0];
            r.dev.at(ring_at, ring_len as usize).fill(0xAB);
            (r, a2)
        };
        let tile_bytes = FakeDev::new(1, false).layout.tile_bytes();
        // how often a whole save of that turn asks whether a request waits: the last time but
        // one is the question before the ring copy, the last one is asked inside it
        let asks = {
            let (mut r, _) = scene();
            let mut n = 0usize;
            let rep = r
                .tier
                .save(&r.table, &mut r.dev, &mut || {
                    n += 1;
                    false
                })
                .unwrap();
            assert!(rep.complete);
            assert_eq!(rep.bytes, 2 * tile_bytes + 2 * 4608 + 8192);
            n
        };
        for cut_at in [asks - 1, asks] {
            let (mut r, a2) = scene();
            // a request arrives exactly before the ring copy, or during it
            let mut n = 0usize;
            let rep = r
                .tier
                .save(&r.table, &mut r.dev, &mut || {
                    n += 1;
                    n >= cut_at
                })
                .unwrap();
            assert!(rep.yielded && !rep.complete, "cut at ask {cut_at}");
            assert_eq!(rep.bytes, 2 * tile_bytes + 2 * 4608);
            assert!(r.tier.has_unsaved());
            {
                // the entry holds the new rows and positions, and the old image
                let store = r.cache.inner.lock().unwrap();
                let id = store.find([7, 7], T, &a2, &[], true, true).unwrap().entry;
                assert_eq!((store.tiles_of(id), store.ring_tiles(id)), (6, Some(4)));
                assert!(!store.ring_serves(id, 6));
            }
            // The next idle time finds no tile and no position left to copy. It takes the
            // image all the same, because the one the entry has does not reach its last kept
            // position; only then is the save complete.
            let rep = r.idle();
            assert!(rep.complete, "cut at ask {cut_at}");
            assert_eq!(rep.bytes, 8192);
            assert!(!r.tier.has_unsaved());
            // one more idle time copies nothing
            assert_eq!(r.idle().bytes, 0);
            // the conversation comes back on an engine loaded again: rows, state and rings,
            // and its drafter has the context of the whole restored prefix
            let mut r2 = rig_on(Arc::clone(&r.cache), 4, true, [7, 7]);
            let mut a3 = a2.clone();
            a3.extend(prompt(300, 6));
            let restored = r2
                .tier
                .begin_request(&mut r2.table, &mut r2.dev, &a3, &[], true, false)
                .unwrap();
            assert!(restored.is_some());
            assert_eq!(r2.tier.last().restore_bytes, 6 * tile_bytes + 4608 + 8192);
            let plan = r2.table.plan_from(&a3, &[], true, false, restored);
            assert_eq!((plan.resume, plan.ring_flo), (6 * 512, 0));
            let (ring_at, ring_len) = r2.dev.ring[0];
            assert!(r2
                .dev
                .at(ring_at, ring_len as usize)
                .iter()
                .all(|&b| b == 0xAB));
        }
    }

    #[test]
    fn a_save_does_not_copy_a_position_its_entry_would_drop_at_once() {
        // four slots on the device, so an entry keeps four positions
        let mut r = rig(64 << 20, 4);
        let tile_bytes = r.dev.layout.tile_bytes();
        // three turns of a conversation, each one tile further: the entry keeps the last four
        // positions
        let mut p = prompt(8 * 512 + 100, 1);
        r.request(&p, true);
        assert!(r.idle().complete);
        for turn in 0..2 {
            p.extend(prompt(512, 10 + turn));
            r.request(&p, true);
            assert!(r.idle().complete);
        }
        // Another conversation takes the card. A cold run of the first one's next turn then
        // keeps its own ladder on the device, with a position far back that is not among the
        // entry's last four.
        r.request(&prompt(3 * 512 + 50, 2), true);
        r.idle();
        p.extend(prompt(512, 20));
        r.request(&p, false);
        let on_device: Vec<u64> = r.table.held_slots().iter().map(|&(k, _)| k).collect();
        let entry = {
            let store = r.cache.inner.lock().unwrap();
            let id = store.find([7, 7], T, &p, &[], true, true).unwrap().entry;
            (id, store.states_of(id))
        };
        let early: Vec<u64> = on_device
            .iter()
            .copied()
            .filter(|k| k < entry.1.first().unwrap())
            .collect();
        assert!(
            !early.is_empty(),
            "the device keeps {on_device:?}, the entry {:?}",
            entry.1
        );
        // the save takes the new tile and the new last position, and not the early position
        // (which would be copied and dropped again by every save from here on)
        let rep = r.idle();
        assert!(rep.complete);
        assert_eq!(rep.bytes, tile_bytes + 4608);
        let states = r.cache.inner.lock().unwrap().states_of(entry.0);
        assert_eq!(states.len(), 4);
        assert!(early.iter().all(|k| !states.contains(k)));
        // the same turn again, on the device: nothing is left to copy
        r.request(&p, true);
        let rep = r.idle();
        assert!(rep.complete);
        assert_eq!(rep.bytes, 0);
    }

    #[test]
    fn the_pieces_of_every_tile_lie_where_the_layout_says_inside_cache_and_segment() {
        // (caches, chunks, rows of a chunk, heads, row bytes, tile)
        for (n_caches, chunks, chunk_rows, heads, row_bytes, tile) in [
            (2u64, 2u64, 4096u64, 2u64, 16u64, 512u64),
            (3, 3, 1024, 4, 8, 256),
            (1, 1, 512, 1, 32, 128),
            (2, 5, 2048, 3, 24, 1024),
        ] {
            let cache_bytes = chunks * heads * chunk_rows * row_bytes;
            let at: Vec<u64> = (0..n_caches)
                .map(|c| 0x1000_0000 + c * 3 * cache_bytes)
                .collect();
            let l = RowLayout::new(
                at.clone(),
                cache_bytes,
                chunks,
                chunk_rows,
                heads,
                row_bytes,
                tile,
            )
            .unwrap();
            assert_eq!(l.cache_bytes(), cache_bytes);
            assert_eq!(l.tiles(), chunks * chunk_rows / tile);
            assert_eq!(l.tile_bytes(), n_caches * heads * tile * row_bytes);
            assert_eq!(l.pieces_per_tile() as u64, n_caches * heads);
            let len = tile * row_bytes;
            let mut seg = vec![0u8; l.tile_bytes() as usize];
            let seg_at = seg.as_mut_ptr();
            let mut dev_ranges: Vec<u64> = Vec::new();
            for t in 0..l.tiles() {
                let mut p = Vec::new();
                l.tile_pieces(t, seg_at, &mut p).unwrap();
                assert_eq!(p.len(), l.pieces_per_tile());
                // the arithmetic this replaced, written out
                let row = t * tile;
                let (chunk, in_chunk) = (row / chunk_rows, row % chunk_rows);
                let chunk_bytes = heads * chunk_rows * row_bytes;
                let mut off = 0u64;
                for (c, &addr) in at.iter().enumerate() {
                    for h in 0..heads {
                        let piece = p[c * heads as usize + h as usize];
                        assert_eq!(
                            piece.dev,
                            addr + chunk * chunk_bytes + (h * chunk_rows + in_chunk) * row_bytes
                        );
                        assert_eq!(piece.host as u64 - seg_at as u64, off);
                        assert_eq!(piece.len as u64, len);
                        // inside its cache, and inside the tile's segment
                        assert!(piece.dev >= addr && piece.dev + len <= addr + cache_bytes);
                        assert!(off + len <= l.tile_bytes());
                        dev_ranges.push(piece.dev);
                        off += len;
                    }
                }
                // the pieces of the tile fill its segment without a gap or an overlap
                assert_eq!(off, l.tile_bytes());
            }
            // over all tiles, the pieces cover every cache exactly once
            dev_ranges.sort_unstable();
            let mut expect = Vec::new();
            for &addr in &at {
                expect.extend((0..cache_bytes / len).map(|i| addr + i * len));
            }
            assert!(dev_ranges == expect);
            // a tile the caches have no place for is refused, and nothing is appended for it
            let mut p = Vec::new();
            assert!(l.tile_pieces(l.tiles(), seg_at, &mut p).is_err());
            assert!(l.tile_pieces(l.tiles() + 7, seg_at, &mut p).is_err());
            assert!(p.is_empty());
        }
    }

    #[test]
    fn ranges_of_a_state_follow_each_other_and_never_leave_their_segment() {
        let ranges = [(0x9000u64, 4096u64), (0x100u64, 512), (0x7000, 100)];
        assert_eq!(ranges_bytes(&ranges), 4708);
        let mut seg = vec![0u8; 4708];
        let at = seg.as_mut_ptr();
        let p = range_pieces(&ranges, at, 4708).unwrap();
        let got: Vec<(u64, u64, usize)> = p
            .iter()
            .map(|x| (x.dev, x.host as u64 - at as u64, x.len))
            .collect();
        assert_eq!(
            got,
            vec![(0x9000, 0, 4096), (0x100, 4096, 512), (0x7000, 4608, 100)]
        );
        // a segment one byte short is refused, with no piece to copy
        assert!(range_pieces(&ranges, at, 4707).is_err());
        assert!(range_pieces(&ranges[..2], at, 4607).is_err());
        assert!(range_pieces(&[], at, 0).unwrap().is_empty());
    }

    #[test]
    fn a_request_pays_for_a_save_only_of_a_conversation_that_went_on() {
        let mut r = rig(64 << 20, 4);
        // a prompt that was prefilled once, then another conversation at once (no idle time):
        // the request pays for nothing, and nothing of the first prompt is kept
        let x = prompt(4 * 512 + 100, 7);
        r.request(&x, true);
        let b = prompt(2 * 512 + 50, 2);
        let (_, rep) = r.request(&b, true);
        assert_eq!((rep.save_bytes, r.dev.down), (0, 0));
        assert_eq!(r.cache.stats().entries, 0);
        // a conversation that went on (its second turn took rows over), then another one at
        // once: its new rows and positions are copied out first, on that request's time
        let a1 = prompt(4 * 512 + 100, 1);
        r.request(&a1, true);
        r.idle();
        let mut a2 = a1.clone();
        a2.extend(prompt(1100, 5));
        assert_eq!(r.request(&a2, true).1.tier, Tier::Device);
        let down = r.dev.down;
        let (_, rep) = r.request(&b, true);
        let tile_bytes = r.dev.layout.tile_bytes();
        assert_eq!(rep.save_bytes, 2 * tile_bytes + 2 * 4608);
        assert_eq!(r.dev.down - down, rep.save_bytes);
        // a request of the conversation the card holds pays for nothing
        let mut b2 = b.clone();
        b2.extend(prompt(600, 3));
        assert_eq!(r.request(&b2, true).1.save_bytes, 0);
        // and the first conversation's third turn comes back from host memory, exactly
        r.idle();
        let mut a3 = a2.clone();
        a3.extend(prompt(200, 6));
        let restored = r
            .tier
            .begin_request(&mut r.table, &mut r.dev, &a3, &[], true, false)
            .unwrap();
        let plan = r.table.plan_from(&a3, &[], true, false, restored);
        assert_eq!(plan.resume, 6 * 512);
        let (rows, state) = Rig::cold(&a3, plan.resume, 4);
        assert!(r.dev.rows_of(0, plan.resume) == rows);
        assert!(r.resume_state(plan.resume) == state);
    }

    #[test]
    fn a_save_that_was_cut_short_and_never_finished_leaves_no_rows_behind() {
        let mut r = rig(64 << 20, 4);
        let a1 = prompt(6 * 512 + 100, 1);
        r.request(&a1, true);
        // the idle save of a prompt that was prefilled once is cut short after some tiles
        let mut asked = 0;
        let rep = r
            .tier
            .save(&r.table, &mut r.dev, &mut || {
                asked += 1;
                asked > 2
            })
            .unwrap();
        assert!(rep.yielded);
        assert_eq!(r.cache.stats().entries, 1);
        // the request that cut it short is another conversation: the device loses the rest,
        // the request does not pay for it, and the rows already copied (nothing could ever
        // resume from them) are given back
        let b = prompt(2 * 512 + 50, 2);
        let (_, rep) = r.request(&b, true);
        assert_eq!(rep.save_bytes, 0);
        assert_eq!(r.cache.stats().entries, 0);
        assert_eq!(r.cache.stats().held_bytes, 0);
    }

    #[test]
    fn a_save_cut_short_by_a_waiting_request_goes_on_later_and_never_shows_half_a_tile() {
        let mut r = rig(64 << 20, 4);
        let a1 = prompt(6 * 512 + 100, 1);
        r.request(&a1, true);
        // a request is waiting after two "ring buffers": the save stops between pieces
        let mut asked = 0;
        let rep = r
            .tier
            .save(&r.table, &mut r.dev, &mut || {
                asked += 1;
                asked > 2
            })
            .unwrap();
        assert!(rep.yielded && !rep.complete);
        // whole tiles only: 16 pieces of 4 per tile
        assert_eq!(r.cache.stats().held_rows, 4 * 512);
        assert_eq!(r.cache.stats().yielded, 1);
        // the rest follows at the next idle call
        let rep = r.idle();
        assert!(rep.complete);
        assert_eq!(r.cache.stats().held_rows, 6 * 512);
        // and the entry restores exactly
        let mut r2 = rig_on(Arc::clone(&r.cache), 4, false, [7, 7]);
        let mut a2 = a1.clone();
        a2.extend(prompt(300, 5));
        let (resume, _) = r2.request(&a2, true);
        assert_eq!(resume, 6 * 512);
        let (rows, _) = Rig::cold(&a2, resume, 4);
        assert!(r2.dev.rows_of(0, resume) == rows);
    }

    #[test]
    fn a_failed_copy_leaves_no_entry_and_no_device_state_to_read() {
        let mut r = rig(64 << 20, 4);
        let a1 = prompt(4 * 512 + 100, 1);
        r.request(&a1, true);
        r.dev.fail_next = true;
        assert!(r.tier.save(&r.table, &mut r.dev, &mut || false).is_err());
        assert_eq!(r.cache.stats().entries, 0);
        assert_eq!(r.cache.stats().held_bytes, 0);
        // a restore that fails drops the entry and everything the table held
        r.tier.unsaved = true;
        r.idle();
        assert_eq!(r.cache.stats().entries, 1);
        let mut r2 = rig_on(Arc::clone(&r.cache), 4, false, [7, 7]);
        let mut a2 = a1.clone();
        a2.extend(prompt(300, 5));
        r2.dev.fail_next = true;
        assert!(r2
            .tier
            .begin_request(&mut r2.table, &mut r2.dev, &a2, &[], true, false)
            .is_err());
        assert_eq!(r2.cache.stats().entries, 0);
        assert!(r2.table.held_slots().is_empty());
        assert_eq!(r2.table.plan(&a2, &[], true, false).resume, 0);
    }

    #[test]
    fn a_copy_that_would_take_longer_than_the_prefill_is_not_made() {
        let mut r = rig(64 << 20, 4);
        let a1 = prompt(4 * 512 + 100, 1);
        r.request(&a1, true);
        r.idle();
        let mut r2 = rig_on(Arc::clone(&r.cache), 4, false, [7, 7]);
        // a link so slow that prefilling 2048 rows is faster than copying them
        r2.tier.set_rates(Rates {
            to_host: 1,
            to_device: 1,
            prefill_rows: 6500,
        });
        let mut a2 = a1.clone();
        a2.extend(prompt(300, 5));
        assert_eq!(r2.request(&a2, true), (0, TierReport::default()));
        assert_eq!(r2.dev.up, 0);
    }

    #[test]
    fn a_full_cache_evicts_and_a_prefix_larger_than_the_budget_is_not_saved() {
        // room for about six tiles and their states
        let tile_bytes = FakeDev::new(1, false).layout.tile_bytes();
        let seg = HostCache::charge_of(tile_bytes);
        let state = HostCache::charge_of(4608);
        let mut r = rig(6 * seg + 6 * state, 2);
        let a = prompt(3 * 512 + 100, 1);
        r.request(&a, true);
        assert!(r.idle().complete);
        let b = prompt(3 * 512 + 100, 2);
        r.request(&b, true);
        assert!(r.idle().complete);
        assert_eq!(r.cache.stats().entries, 2);
        // a third conversation: the one used least recently leaves
        let c = prompt(3 * 512 + 100, 3);
        r.request(&c, true);
        assert!(r.idle().complete);
        let st = r.cache.stats();
        assert_eq!((st.entries, st.evictions), (2, 1));
        assert!(st.held_bytes <= st.budget_bytes);
        // a prefix that cannot fit at all is not saved, evicts nothing and fails no request
        let big = prompt(12 * 512 + 100, 4);
        let (resume, _) = r.request(&big, true);
        assert_eq!(resume, 0);
        let rep = r.idle();
        assert!(!rep.complete && rep.bytes == 0);
        let st = r.cache.stats();
        assert_eq!((st.entries, st.refused), (2, 1));
    }

    #[test]
    fn a_next_turn_fits_a_nearly_full_cache_when_only_its_new_part_does() {
        let tile_bytes = FakeDev::new(1, false).layout.tile_bytes();
        let seg = HostCache::charge_of(tile_bytes);
        let state = HostCache::charge_of(4608);
        // the first turn leaves four tiles and three kept positions; the budget has room for
        // exactly what the next turn adds (two tiles, two positions) and not a byte more
        let cache = Arc::new(SharedCache::new(
            4 * seg + 3 * state + 2 * seg + 2 * state,
            0,
            false,
            false,
        ));
        let mut r = rig_on(Arc::clone(&cache), 4, false, [7, 7]);
        let a1 = prompt(4 * 512 + 100, 1);
        r.request(&a1, true);
        assert!(r.idle().complete);
        assert_eq!(cache.stats().held_bytes, 4 * seg + 3 * state);
        let mut a2 = a1.clone();
        a2.extend(prompt(1100, 5));
        assert_eq!(r.request(&a2, true).0, 4 * 512);
        // The save is charged for the two tiles and the two positions it adds. The positions
        // the entry keeps are in the ledger already: charged again, the save would not fit, the
        // entry would stay at the first turn and the conversation would prefill on its return.
        let rep = r.idle();
        assert!(rep.complete);
        assert_eq!(rep.bytes, 2 * tile_bytes + 2 * 4608);
        let st = cache.stats();
        assert_eq!((st.entries, st.refused, st.evictions), (1, 0, 0));
        assert_eq!(st.held_rows, 6 * 512);
        assert!(st.held_bytes <= st.budget_bytes);
        // it comes back from host memory at its last kept position after the card was released
        let mut r2 = rig_on(Arc::clone(&cache), 4, false, [7, 7]);
        let mut a3 = a2.clone();
        a3.extend(prompt(300, 6));
        let (resume, rep) = r2.request(&a3, true);
        assert_eq!((resume, rep.tier), (6 * 512, Tier::Host));
    }

    #[test]
    fn a_request_after_the_rows_were_zeroed_gets_nothing_copied_back() {
        let mut r = rig(64 << 20, 4);
        let a1 = prompt(6 * 512 + 100, 1);
        r.request(&a1, true);
        assert!(r.idle().complete);
        let b = prompt(3 * 512 + 50, 2);
        r.request(&b, true);
        r.idle();
        // the reset zeroed the K/V rows before this request: its plan forgets the table and
        // prefills everything, so nothing is copied for it
        let mut a2 = a1.clone();
        a2.extend(prompt(700, 5));
        let restored = r
            .tier
            .begin_request(&mut r.table, &mut r.dev, &a2, &[], true, true)
            .unwrap();
        assert!(restored.is_none());
        assert_eq!(r.dev.up, 0);
        let plan = r.table.plan_from(&a2, &[], true, true, restored);
        assert_eq!(plan.resume, 0);
        // the entry is still there for the turn after it
        assert_eq!(r.cache.stats().entries, 2);
    }

    #[test]
    fn the_identity_ignores_device_addresses_and_follows_everything_that_must_match() {
        /// One engine's layout: the K/V caches at `at`, with these dimensions; a kept
        /// position's state ranges and the drafter rings as `(address, bytes)`.
        #[derive(Clone)]
        struct L {
            names: Vec<&'static str>,
            at: Vec<u64>,
            dims: (u64, u64, u64, u64, u64),
            state: Vec<(u64, u64)>,
            ring: Vec<(u64, u64)>,
        }
        fn fingerprint(l: &L) -> String {
            let (chunks, chunk_rows, heads, row_bytes, tile) = l.dims;
            let rows = RowLayout::new(
                l.at.clone(),
                chunks * heads * chunk_rows * row_bytes,
                chunks,
                chunk_rows,
                heads,
                row_bytes,
                tile,
            )
            .unwrap();
            let names: Vec<String> = l.names.iter().map(|n| n.to_string()).collect();
            layout_fingerprint(&names, &rows, &l.state, &l.ring)
        }
        let id = |l: &L| identity_of("the engine", &fingerprint(l));
        let base = L {
            names: vec!["L0_kcache", "L0_vcache"],
            at: vec![0x1000_0000, 0x1400_0000],
            dims: (2, 4096, 2, 16, 512),
            state: vec![(0x2000_0000, 4096), (0x2000_1000, 512)],
            ring: vec![(0x3000_0000, 8192)],
        };
        // the same engine loaded again with every buffer somewhere else: the same identity,
        // so it finds the entries it saved before
        let moved = L {
            at: vec![0x7700_0000, 0x5500_0000],
            state: vec![(0x9000_0000, 4096), (0x4000_0000, 512)],
            ring: vec![(0x1234_5000, 8192)],
            ..base.clone()
        };
        assert_eq!(fingerprint(&moved), fingerprint(&base));
        assert_eq!(id(&moved), id(&base));
        // no address is in what is hashed
        let text = fingerprint(&base);
        for addr in [
            0x1000_0000u64,
            0x1400_0000,
            0x2000_0000,
            0x2000_1000,
            0x3000_0000,
        ] {
            assert!(
                !text.contains(&addr.to_string()) && !text.contains(&format!("{addr:x}")),
                "{text}"
            );
        }
        // each thing that must match for the bytes to mean the same changes it
        let with = |f: &dyn Fn(&mut L)| {
            let mut l = base.clone();
            f(&mut l);
            id(&l)
        };
        let others = [
            // the number of caches
            with(&|l| {
                l.names.push("L1_kcache");
                l.at.push(0x1800_0000);
            }),
            // their order
            with(&|l| l.names.swap(0, 1)),
            // another cache in the same place
            with(&|l| l.names[1] = "L1_kcache"),
            // chunks, rows of a chunk, heads, row bytes, tile
            with(&|l| l.dims.0 = 4),
            with(&|l| l.dims.1 = 8192),
            with(&|l| l.dims.2 = 4),
            with(&|l| l.dims.3 = 32),
            with(&|l| l.dims.4 = 256),
            // the size of a state range, their order, their number
            with(&|l| l.state[0].1 = 8192),
            with(&|l| l.state.swap(0, 1)),
            with(&|l| l.state.push((0x2000_2000, 512))),
            // the rings: their size, and whether there are any
            with(&|l| l.ring[0].1 = 16384),
            with(&|l| l.ring.clear()),
        ];
        for (i, other) in others.iter().enumerate() {
            assert_ne!(*other, id(&base), "change {i} left the identity as it was");
            for (j, third) in others.iter().enumerate().skip(i + 1) {
                assert_ne!(other, third, "changes {i} and {j} give one identity");
            }
        }
        // and so does anything else that decides the bytes (the engine's own text)
        assert_ne!(
            identity_of("another engine", &fingerprint(&base)),
            id(&base)
        );
    }

    #[test]
    fn the_first_copies_ask_for_their_buffer_and_an_engine_that_cannot_have_it_gets_no_tier() {
        use std::sync::atomic::{AtomicU64, Ordering};
        const MIB: u64 = 1 << 20;
        // a made-up host: what it has, less what the store has mapped
        static HOST: AtomicU64 = AtomicU64::new(0);
        fn available(held: u64) -> u64 {
            HOST.load(Ordering::Relaxed).saturating_sub(held)
        }
        HOST.store(64 * MIB, Ordering::Relaxed);
        let cache = Arc::new(SharedCache::on_host(64 * MIB, 0, available));
        let mut r = rig_on(Arc::clone(&cache), 4, false, [7, 7]);
        // the buffer is one huge page here (these caches are small); with room it is taken,
        // the rates are measured, and nothing of it is left
        r.tier.calibrate(&mut r.dev).unwrap();
        assert!(r.dev.down > 0 && r.dev.up == r.dev.down);
        assert_eq!(cache.stats().held_bytes, 0);
        // a conversation is saved
        let p = prompt(3 * 512 + 100, 1);
        r.request(&p, true);
        assert!(r.idle().complete);
        let held = cache.stats().held_bytes;
        assert!(held >= 4 * MIB);
        // The engine is loaded again on a host that has 1 MiB left beside the store: less than
        // the buffer. The store gives way: its entry leaves, and the copies are made.
        HOST.store(held + MIB, Ordering::Relaxed);
        let mut r2 = rig_on(Arc::clone(&cache), 4, false, [7, 7]);
        r2.tier.calibrate(&mut r2.dev).unwrap();
        assert!(r2.dev.down > 0);
        assert_eq!(
            (
                cache.stats().entries,
                cache.stats().evictions,
                cache.stats().held_bytes
            ),
            (0, 1, 0)
        );
        // A host that cannot hold the buffer even with the store empty: refused with the
        // numbers before a byte is copied. The engine then serves without the tier.
        HOST.store(MIB, Ordering::Relaxed);
        let mut r3 = rig_on(Arc::clone(&cache), 4, false, [7, 7]);
        let e = r3.tier.calibrate(&mut r3.dev).unwrap_err();
        assert!(
            e.contains("the buffer of its first copies needs 0.00 GiB of host memory, and 0.00 GiB are available now"),
            "{e}"
        );
        assert_eq!((r3.dev.down, r3.dev.up), (0, 0));
        // what a server plans for these copies beside its store
        assert_eq!(first_copies_bytes(), 128 * MIB);
    }

    #[test]
    fn the_identity_follows_the_fp8_table_a_load_reads_wherever_it_is_found() {
        use crate::engine::fp8_pack::{
            act_path_with, act_table_id, default_fp8_dir, ACT_TABLE_FILE,
        };
        let root = std::env::temp_dir().join(format!("gaudium-fp8-id-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let model = root.join("models").join("M");
        let default_place = default_fp8_dir(&model).join(ACT_TABLE_FILE);
        let by_env = root.join("from-env").join("scales.txt");
        let by_setting = root.join("from-setting").join("scales.txt");
        let write = |path: &std::path::Path, k: i32| {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(
                path,
                format!("L0_h {k} 12.5 1.25\nL0_core 2 7 0.5\nL0_h2 3 12 1\nL0_m 1 3 0.25\n"),
            )
            .unwrap();
        };
        for p in [&default_place, &by_env, &by_setting] {
            write(p, 3);
        }
        // the identity of the engine a load gives, from the table that load reads: resolved as
        // the load resolves it, parsed as the load parses it
        let id = |cfg: &Qwen3Config, env: &dyn Fn(&str) -> Option<PathBuf>| {
            let path = act_path_with(cfg, env);
            let act = crate::fp8::load_act_scales(&path).unwrap();
            (
                path,
                identity_of(
                    &format!("the engine|fp8=table={}", act_table_id(&act)),
                    "the layout",
                ),
            )
        };
        let no_env = |_: &str| -> Option<PathBuf> { None };
        let env = {
            let by_env = by_env.clone();
            move |name: &str| (name == "GAUDIUM_FP8_ACT").then(|| by_env.clone())
        };
        let plain = Qwen3Config::new(model.clone(), root.join("kernels"));
        let mut set = plain.clone();
        set.fp8_act = Some(by_setting.clone());
        // (configuration, environment, the file that load reads, the two it does not)
        type Env<'a> = &'a dyn Fn(&str) -> Option<PathBuf>;
        type Case<'a> = (&'a Qwen3Config, Env<'a>, &'a PathBuf, [&'a PathBuf; 2]);
        let cases: [Case; 3] = [
            (&set, &env, &by_setting, [&by_env, &default_place]),
            (&plain, &env, &by_env, [&by_setting, &default_place]),
            (&plain, &no_env, &default_place, [&by_setting, &by_env]),
        ];
        for (cfg, env, read, others) in cases {
            for p in [&default_place, &by_env, &by_setting] {
                write(p, 3);
            }
            let (path, before) = id(cfg, env);
            assert_eq!(&path, read);
            // a table that the load does not read changes nothing
            for other in others {
                write(other, 5);
            }
            assert_eq!(id(cfg, env).1, before, "{}", read.display());
            // the one it reads, with another scale: another engine
            write(read, 4);
            assert_ne!(id(cfg, env).1, before, "{}", read.display());
            // the same scales again: the same engine, whenever the file was written
            write(read, 3);
            assert_eq!(id(cfg, env).1, before, "{}", read.display());
        }
        // the same scales give one identity wherever the table lies; a BF16 load names none
        for p in [&default_place, &by_env, &by_setting] {
            write(p, 3);
        }
        assert_eq!(id(&set, &env).1, id(&plain, &no_env).1);
        assert_eq!(act_table_id(&std::collections::BTreeMap::new()), "none");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_identity_follows_the_drafter_s_fp8_files() {
        use crate::engine::step_opts::StepOpts;
        let root = std::env::temp_dir().join(format!("gaudium-drafter-id-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let draft = root.join("Drafter-b16");
        let fp8_dir = spec::dspine::default_fp8_dir(&draft);
        std::fs::create_dir_all(&draft).unwrap();
        std::fs::create_dir_all(&fp8_dir).unwrap();
        let table = fp8_dir.join(spec::dspine::ACT_TABLE_FILE);
        let opts = StepOpts::default();
        // no table: named as absent
        let none = drafter_fp8_text(&draft, &opts);
        assert!(none.starts_with("table=absent|"), "{none}");
        std::fs::write(&table, "ds_L0_x 3 12 1\n").unwrap();
        let first = drafter_fp8_text(&draft, &opts);
        assert_ne!(first, none);
        // another scale in the table, or another cached matrix beside it: another text
        std::fs::write(&table, "ds_L0_x 4 12 1\n").unwrap();
        let second = drafter_fp8_text(&draft, &opts);
        assert_ne!(second, first);
        std::fs::write(fp8_dir.join("W-0123.w8"), [1u8, 2, 3]).unwrap();
        let third = drafter_fp8_text(&draft, &opts);
        assert_ne!(third, second);
        // a table the run options name is the one that counts
        let elsewhere = root.join("elsewhere").join("ds.txt");
        std::fs::create_dir_all(elsewhere.parent().unwrap()).unwrap();
        std::fs::write(&elsewhere, "ds_L0_x 3 12 1\n").unwrap();
        let named = StepOpts {
            draft_fp8_act: Some(elsewhere.clone()),
            ..StepOpts::default()
        };
        let by_option = drafter_fp8_text(&draft, &named);
        assert_ne!(by_option, third);
        std::fs::write(&table, "ds_L0_x 9 12 1\n").unwrap();
        assert_eq!(drafter_fp8_text(&draft, &named), by_option);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn today_s_tuning_values() {
        // pinned here, not in a law: each may change with measurements
        use crate::synapse::xfer::{COPY_THREADS, RING_BUFS, RING_BUF_BYTES};
        assert_eq!(IDLE_GRACE, std::time::Duration::from_millis(30));
        assert_eq!((RING_BUFS, RING_BUF_BYTES, COPY_THREADS), (4, 64 << 20, 2));
        assert_eq!(
            (ROUND_TILES, RATE_MIN_BYTES, CALIBRATE_BYTES),
            (64, 32 << 20, 128 << 20)
        );
        assert_eq!(crate::verified::hc_default_share(), 4);
        // a prompt continues an entry when it shares at least as many tiles as it drops
        use crate::verified::hc_continues;
        assert!(hc_continues(5, 6) && hc_continues(3, 6) && !hc_continues(2, 6));
        // a request pays for a save only of a conversation whose last request took rows over
        use crate::verified::hc_must_flush;
        assert!(hc_must_flush(true, 512, 0, 2048) && !hc_must_flush(true, 0, 0, 2048));
        // the rings are copied when the entry has no image, or one that does not reach its
        // last kept position; a save is whole only when neither is the case
        use crate::verified::hc_ring_short;
        assert!(hc_ring_short(false, 6, 6) && hc_ring_short(true, 4, 6));
        assert!(!hc_ring_short(true, 6, 6) && !hc_ring_short(true, 8, 6));
        assert!(hc_saved_whole(true, 8192, true, 6, 6) && hc_saved_whole(true, 0, false, 0, 6));
        assert!(!hc_saved_whole(true, 8192, true, 4, 6) && !hc_saved_whole(false, 0, false, 0, 6));
        // ... and copied back unless the device's rings are that image already
        assert!(hc_ring_restore(8192, true, false) && !hc_ring_restore(8192, true, true));
        // a restore with an image takes the image's first valid position, one without takes
        // the resume point
        assert_eq!(hc_ring_floor(8192, true, 700, 2048), 700);
        assert_eq!(hc_ring_floor(8192, false, 700, 2048), 2048);
        assert_eq!(hc_ring_floor(0, true, 700, 2048), 2048);
        // tiles of a batch count as copied one whole tile at a time (4 pieces to a tile)
        let copied = |done| (0..3).take_while(|&i| hc_tile_copied(i, 4, done)).count();
        assert_eq!(
            (copied(0), copied(3), copied(4), copied(11), copied(12)),
            (0, 0, 1, 2, 3)
        );
        // a copy wins when it takes less time than the prefill it replaces: 2 GiB at
        // 14 GiB/s against 32,256 rows at 6,500 rows per second
        assert!(hc_copy_wins(2 << 30, 14 << 30, 32_256, 6_500));
        assert!(!hc_copy_wins(2 << 30, 100 << 20, 512, 6_500));
    }

    #[test]
    fn row_layout_follows_the_caches_chunk_by_chunk_and_head_by_head() {
        let l =
            RowLayout::new(vec![1000, 900_000], 2 * 2 * 4096 * 16, 2, 4096, 2, 16, 512).unwrap();
        assert_eq!(l.tile_bytes(), 2 * 2 * 512 * 16);
        assert_eq!((l.pieces_per_tile(), l.tiles()), (4, 16));
        let mut p = Vec::new();
        // tile 9: chunk 1, rows 512..1024 of it
        l.tile_pieces(9, std::ptr::null_mut(), &mut p).unwrap();
        let chunk = 2 * 4096 * 16u64;
        assert_eq!(
            p.iter().map(|x| (x.dev, x.len)).collect::<Vec<_>>(),
            vec![
                (1000 + chunk + 512 * 16, 8192),
                (1000 + chunk + (4096 + 512) * 16, 8192),
                (900_000 + chunk + 512 * 16, 8192),
                (900_000 + chunk + (4096 + 512) * 16, 8192),
            ]
        );
        assert_eq!(p[3].host as usize, 3 * 8192);
        // layouts the copies cannot follow are refused with the reason
        assert!(RowLayout::new(vec![0], 2 * 2 * 4096 * 16, 2, 4096, 2, 16, 384).is_err());
        assert!(RowLayout::new(vec![0], 12345, 2, 4096, 2, 16, 512).is_err());
        assert!(RowLayout::new(vec![], 0, 2, 4096, 2, 16, 512).is_err());
        // [kv8] FP8 caches permute V rows in 256-row blocks: a 128-row tile would split one
        let fp8 = crate::verified::kv_vperm_rows(crate::verified::KvDtype::KvFp8);
        assert!(RowLayout::with_block(vec![0], 2 * 2 * 4096 * 16, 2, 4096, 2, 16, 128, fp8).is_err());
        assert!(RowLayout::with_block(vec![0], 2 * 2 * 4096 * 16, 2, 4096, 2, 16, 256, fp8).is_ok());
        assert!(RowLayout::new(vec![0], 2 * 2 * 4096 * 16, 2, 4096, 2, 16, 128).is_ok());
    }
}
