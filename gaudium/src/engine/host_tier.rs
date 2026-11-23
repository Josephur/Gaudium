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
    hc_clean_hi, hc_clean_join_hi, hc_clean_join_lo, hc_clean_lo, hc_copy_wins, hc_must_flush,
    hc_ring_stale, hc_rows_saveable, hc_tile_clean, hc_written_hi, hc_written_lo, pc_boundary,
};

/// Tiles whose rows one copy call takes (a waiting request is noticed between ring buffers
/// inside a call; rows are committed tile by tile either way).
const ROUND_TILES: u64 = 64;
/// A copy below this does not update a measured rate (its fixed costs would dominate).
const RATE_MIN_BYTES: u64 = 32 * 1024 * 1024;
/// Bytes the rates are first measured with, when the engine is loaded.
const CALIBRATE_BYTES: u64 = 128 * 1024 * 1024;

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
/// holds its rows one after the other (`row_bytes` each).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowLayout {
    /// Device address of each cache, in name order.
    pub caches: Vec<u64>,
    pub chunks: u64,
    pub chunk_rows: u64,
    pub heads: u64,
    pub row_bytes: u64,
    pub tile: u64,
}

impl RowLayout {
    /// The layout of caches of `cache_bytes` each. An error says why this engine's caches
    /// cannot be copied tile by tile.
    pub fn new(
        caches: Vec<u64>,
        cache_bytes: u64,
        chunks: u64,
        chunk_rows: u64,
        heads: u64,
        row_bytes: u64,
        tile: u64,
    ) -> Result<Self, String> {
        if caches.is_empty() || tile == 0 || chunks == 0 || heads == 0 || row_bytes == 0 {
            return Err("no K/V caches to copy".into());
        }
        if !chunk_rows.is_multiple_of(tile) || !tile.is_multiple_of(128) {
            return Err(format!(
                "a prefill tile of {tile} rows does not divide a K/V chunk of {chunk_rows} rows into whole 128-row blocks"
            ));
        }
        let expect = chunks
            .saturating_mul(heads)
            .saturating_mul(chunk_rows)
            .saturating_mul(row_bytes);
        if cache_bytes != expect {
            return Err(format!(
                "a K/V cache of {cache_bytes} bytes is not {chunks} chunks x {heads} heads x {chunk_rows} rows x {row_bytes} bytes"
            ));
        }
        Ok(Self {
            caches,
            chunks,
            chunk_rows,
            heads,
            row_bytes,
            tile,
        })
    }

    /// Host bytes of one tile's rows.
    pub fn tile_bytes(&self) -> u64 {
        self.caches.len() as u64 * self.heads * self.tile * self.row_bytes
    }

    /// Device ranges of one tile: one per cache and head.
    pub fn pieces_per_tile(&self) -> usize {
        self.caches.len() * self.heads as usize
    }

    /// Tiles the caches hold.
    pub fn tiles(&self) -> u64 {
        self.chunks * self.chunk_rows / self.tile
    }

    /// Appends the pieces of tile `t`, whose rows live at `host`: per cache, per head, the
    /// tile's rows of that head. The host bytes follow each other in that order.
    pub fn tile_pieces(&self, t: u64, host: *mut u8, out: &mut Vec<Piece>) {
        let row = t * self.tile;
        let (chunk, in_chunk) = (row / self.chunk_rows, row % self.chunk_rows);
        let len = (self.tile * self.row_bytes) as usize;
        let chunk_bytes = self.heads * self.chunk_rows * self.row_bytes;
        let mut off = 0usize;
        for &addr in &self.caches {
            for h in 0..self.heads {
                out.push(Piece {
                    dev: addr
                        + chunk * chunk_bytes
                        + (h * self.chunk_rows + in_chunk) * self.row_bytes,
                    // SAFETY: `off + len <= tile_bytes()`, the size of the tile's segment.
                    host: unsafe { host.add(off) },
                    len,
                });
                off += len;
            }
        }
    }
}

/// The pieces of device ranges `ranges` (`(address, bytes)`), whose bytes follow each other at
/// `host`.
fn range_pieces(ranges: &[(u64, u64)], host: *mut u8) -> Vec<Piece> {
    let mut off = 0usize;
    ranges
        .iter()
        .map(|&(dev, bytes)| {
            let p = Piece {
                dev,
                // SAFETY: the segment at `host` has the size of the sum of the ranges.
                host: unsafe { host.add(off) },
                len: bytes as usize,
            };
            off += bytes as usize;
            p
        })
        .collect()
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
        let state_bytes = slots
            .first()
            .map_or(0, |s| s.iter().map(|&(_, b)| b).sum());
        let ring_bytes = ring.iter().map(|&(_, b)| b).sum();
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
    pub fn calibrate(&mut self, copier: &mut dyn Copier) -> Result<(), String> {
        let bytes = CALIBRATE_BYTES.min(self.rows.chunks * self.rows.heads * self.rows.chunk_rows * self.rows.row_bytes);
        let mut map = crate::hostmem::AnonMap::new(bytes)?;
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
        if rows_zeroed {
            // the reset zeroed the K/V rows: nothing on the device equals anything saved
            self.clean.clear();
            self.ring_owner = None;
            self.unsaved = false;
        }
        let peek = table.peek(prompt, images, wanted, rows_zeroed);
        if hc_must_flush(self.unsaved, peek.cap, table.kept_rows()) {
            let rep = self.save(table, copier, &mut || false)?;
            self.last.save_ms = rep.ms;
            self.last.save_bytes = rep.bytes;
        }

        let tile = self.rows.tile;
        let cache = Arc::clone(&self.cache);
        let mut store = cache.inner.lock().unwrap_or_else(|e| e.into_inner());
        self.clean.retain(|id, _| store.contains(*id));
        let Some(found) = store.find(self.id, tile, prompt, images, wanted, table.enabled())
        else {
            return Ok(None);
        };
        // what the restore would copy: the entry's tiles the device no longer holds, the state
        // at the resume point, and the drafter rings unless they are still this entry's
        let (lo, hi) = self.clean.get(&found.entry).copied().unwrap_or((0, 0));
        let linked = self.clean.contains_key(&found.entry);
        let copy_tiles: Vec<u64> = (0..found.tiles)
            .filter(|&t| !hc_tile_clean(linked, t, lo, hi))
            .collect();
        let ring_copy = self.ring_bytes > 0
            && store.ring_serves(found.entry, found.tiles)
            && self.ring_owner != Some(found.entry);
        let bytes = copy_tiles.len() as u64 * self.rows.tile_bytes()
            + self.state_bytes
            + if ring_copy { self.ring_bytes } else { 0 };
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
        let ring_from = match view.ring {
            Some((_, from)) if self.ring_bytes > 0 => from,
            _ => found.pos,
        };
        let slot = table.adopt_restored(&view.tokens, &view.images, found.tiles, ring_from);
        let mut pieces: Vec<Piece> = Vec::new();
        for &t in &copy_tiles {
            self.rows
                .tile_pieces(t, view.rows[t as usize] as *mut u8, &mut pieces);
        }
        if let Some(s) = slot {
            pieces.extend(range_pieces(&self.slots[s], view.state as *mut u8));
        }
        if let (true, Some((ptr, _))) = (ring_copy, view.ring) {
            pieces.extend(range_pieces(&self.ring, ptr as *mut u8));
        }
        let res = copier.to_device(&pieces);
        match res {
            Err(e) => {
                // nothing of a copy that failed may be read later: not the entry, not the
                // device rows it was to fill
                store.end_read(found.entry, false);
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
                Self::note_rate(&mut self.rates.to_device, &st);
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let s = store.stats_mut();
                s.restores += 1;
                s.restore_bytes += st.bytes;
                s.restore_s += ms / 1e3;
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
        let tile = self.rows.tile;
        let (w0, w1) = (hc_written_lo(resume, tile), hc_written_hi(written_end, tile));
        for iv in self.clean.values_mut() {
            *iv = (hc_clean_lo(iv.0, iv.1, w0, w1), hc_clean_hi(iv.0, iv.1, w0, w1));
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
        let kept = table.held_slots();
        let (held, images) = table.held();
        let tile = self.rows.tile;
        let Some(&(upto, _)) = kept.last() else {
            self.unsaved = false;
            rep.complete = true;
            return Ok(rep);
        };
        if !hc_rows_saveable(upto, tile, held.len() as u64) || upto > self.rows.tiles() {
            self.unsaved = false;
            return Ok(rep);
        }
        let t0 = Instant::now();
        let tile_bytes = self.rows.tile_bytes();
        let per_tile = self.rows.pieces_per_tile();
        let cache = Arc::clone(&self.cache);
        let mut store = cache.inner.lock().unwrap_or_else(|e| e.into_inner());
        self.clean.retain(|id, _| store.contains(*id));
        let extra = kept.len() as u64 * HostCache::charge_of(self.state_bytes)
            + if self.ring_bytes > 0 {
                HostCache::charge_of(self.ring_bytes)
            } else {
                0
            };
        let Some(target) =
            store.open_save(self.id, tile, tile_bytes, held, images, upto, extra)
        else {
            // no room for this prefix at all: not asked again until the device holds another
            self.unsaved = false;
            cache.publish(&store);
            return Ok(rep);
        };
        let entry = target.entry;
        // an entry cut to the tiles the prompt shares: its tiles past them are gone
        if let Some(iv) = self.clean.get_mut(&entry) {
            *iv = (
                hc_clean_lo(iv.0, iv.1, target.have_tiles, u64::MAX),
                hc_clean_hi(iv.0, iv.1, target.have_tiles, u64::MAX),
            );
        }
        let mut complete = true;
        // the entry gained rows or positions in this save (its ring image is then stale)
        let mut grew = false;
        let fail = |store: &mut HostCache, this: &mut Self, e: String| -> String {
            store.fail(entry);
            this.clean.remove(&entry);
            if this.ring_owner == Some(entry) {
                this.ring_owner = None;
            }
            cache.publish(store);
            e
        };

        // rows, tile by tile
        let mut have = target.have_tiles;
        while have < upto && complete {
            let n = (upto - have).min(ROUND_TILES);
            let Some(segs) = store.begin_rows(entry, n, tile_bytes) else {
                complete = false;
                break;
            };
            let mut pieces: Vec<Piece> = Vec::with_capacity(n as usize * per_tile);
            for (i, &seg) in segs.iter().enumerate() {
                self.rows.tile_pieces(have + i as u64, seg, &mut pieces);
            }
            let (done, st) = match copier.to_host(&pieces, stop) {
                Ok(x) => x,
                Err(e) => return Err(fail(&mut store, self, e)),
            };
            let now = store.end_rows(entry, have + (done / per_tile) as u64);
            rep.bytes += st.bytes;
            Self::note_rate(&mut self.rates.to_host, &st);
            if now > have {
                grew = true;
                let (lo, hi) = self.clean.get(&entry).copied().unwrap_or((have, have));
                self.clean.insert(
                    entry,
                    (
                        hc_clean_join_lo(lo, hi, have, now),
                        hc_clean_join_hi(lo, hi, have, now),
                    ),
                );
            }
            if now < have + n {
                complete = false;
                rep.yielded = true;
            }
            have = now;
        }

        // the state at every kept position the entry's rows reach
        if complete && store.contains(entry) {
            let have_states = store.states_of(entry);
            for &(k, slot) in &kept {
                if have_states.contains(&k) || k > store.tiles_of(entry) {
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
                let pieces = range_pieces(&self.slots[slot], ptr);
                let (done, st) = match copier.to_host(&pieces, stop) {
                    Ok(x) => x,
                    Err(e) => return Err(fail(&mut store, self, e)),
                };
                let whole = done == pieces.len();
                store.end_state(entry, k, whole);
                rep.bytes += st.bytes;
                grew |= whole;
                if !whole {
                    complete = false;
                    rep.yielded = true;
                    break;
                }
            }
            store.trim_states(entry, table.slot_count(), upto);
        }

        // the drafter rings, whole, when the entry has no image or outgrew the one it has
        if complete
            && self.ring_bytes > 0
            && store.contains(entry)
            && hc_ring_stale(store.has_ring(entry), grew)
        {
            if stop() {
                complete = false;
                rep.yielded = true;
            } else if let Some(ptr) = store.begin_piece(entry, self.ring_bytes) {
                let pieces = range_pieces(&self.ring, ptr);
                let (done, st) = match copier.to_host(&pieces, stop) {
                    Ok(x) => x,
                    Err(e) => return Err(fail(&mut store, self, e)),
                };
                let whole = done == pieces.len();
                store.end_ring(entry, table.ring_from(), whole);
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

        rep.complete = complete && store.contains(entry);
        rep.ms = t0.elapsed().as_secs_f64() * 1e3;
        // a save cut short goes on at the next call; one that found no room does not ask again
        self.unsaved = rep.yielded;
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
        cache.publish(&store);
        Ok(rep)
    }
}

/// Identity of a loaded engine for the store: a digest of everything that decides the bytes of
/// a row or of a state (the model's files, the precisions, the context and the tile, the
/// kernels, the drafter, the vision tower), and of the layout the copies use.
pub(super) fn engine_identity(cfg: &Qwen3Config, eng: &Qwen3Engine, layout: &str) -> EngineId {
    /// Names, sizes and modification times of the weight and configuration files of `dir`.
    fn files_of(dir: &std::path::Path) -> String {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let keep = name.ends_with(".safetensors")
                    || name.ends_with(".json")
                    || name.ends_with(".txt");
                let md = e.metadata().ok()?;
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
    let file_of = |p: &Option<PathBuf>| -> String {
        p.as_ref().map_or(String::new(), |p| {
            let md = std::fs::metadata(p).ok();
            format!(
                "{}:{}:{}",
                p.display(),
                md.as_ref().map_or(0, |m| m.len()),
                md.and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos())
            )
        })
    };
    let drafter = cfg.spec.as_ref().map(|sc| {
        let dir = match &sc.method {
            spec::SpecMethod::DSpine { draft_model }
            | spec::SpecMethod::DFlash2 { draft_model }
            | spec::SpecMethod::Assistant { draft_model } => files_of(draft_model),
            _ => String::new(),
        };
        format!("{sc:?}|{dir}")
    });
    let text = format!(
        "model={}|files={}|spec={:?}|ctx={}|tile={}|prefill={:?}|precision={:?}|fp8_act={}|fp8_cache={:?}|moe_routed={}|reset={:?}|step={:?}|kernels={:?}|driver={}|drafter={:?}|ctx_rows={}|vision={}/{}/{}|layout={layout}",
        cfg.model_dir.display(),
        files_of(&cfg.model_dir),
        eng.spec,
        eng.max_ctx,
        eng.prefill_tile,
        cfg.prefill,
        cfg.precision,
        file_of(&cfg.fp8_act),
        cfg.fp8_cache,
        cfg.moe_routed,
        cfg.request_reset,
        cfg.step,
        eng.elf_hashes,
        eng.driver_version,
        drafter,
        eng.prefix.ctx_rows(),
        cfg.vision_max_patches,
        cfg.vision_max_tokens,
        cfg.vision_gelu_bf16,
    );
    digest128(text.as_bytes())
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
    // the trunk K/V caches: one size, chunk-major, `n_kv_heads` heads of BF16 rows
    let caches: Vec<u64> = eng.kv_addr.values().map(|&(a, _)| a).collect();
    let cache_bytes = eng.kv_addr.values().next().map_or(0, |&(_, b)| b);
    if eng.kv_addr.values().any(|&(_, b)| b != cache_bytes) {
        eprintln!("[cache-ram] off for this engine: its K/V caches differ in size");
        return Ok(());
    }
    let rows = match RowLayout::new(
        caches,
        cache_bytes,
        eng.max_chunks,
        KV_CHUNK,
        eng.spec.n_kv_heads,
        eng.spec.head_dim * 2,
        eng.prefill_tile,
    ) {
        Ok(r) => r,
        Err(why) => {
            eprintln!("[cache-ram] off for this engine: {why}");
            return Ok(());
        }
    };
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
    let layout = format!(
        "{rows:?}|slots={}|state={:?}|ring={:?}",
        slots.len(),
        slots.first().map(|s| s.iter().map(|&(_, b)| b).collect::<Vec<_>>()),
        ring.iter().map(|&(_, b)| b).collect::<Vec<_>>()
    );
    let id = engine_identity(cfg, eng, &layout);
    // The ring: a few buffers of untouched memory, each mapped for the device on its own (the
    // one pinned host memory of the tier, planned when the server started). An engine that
    // cannot get it serves without the tier: no model fails to load for lack of a cache.
    let t0 = Instant::now();
    let mut xfer = match XferRing::new(
        &eng.dev,
        crate::synapse::xfer::RING_BUFS,
        crate::synapse::xfer::RING_BUF_BYTES,
        crate::synapse::xfer::RingKind::MappedHuge,
        crate::synapse::xfer::COPY_THREADS,
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
        eprintln!("[cache-ram] off for this engine: the first copy through the transfer ring failed ({e})");
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
    eng.xfer = Some(xfer);
    eng.host = Some(tier);
    Ok(())
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
                h = (h ^ t as u32 as u64).wrapping_mul(0x100_0000_01B3).rotate_left(23);
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
                let fill: Vec<u8> = (0..bytes).map(|i| (v >> (8 * (i % 8))) as u8 ^ i as u8).collect();
                self.at(dev, bytes as usize).copy_from_slice(&fill);
            }
        }

        fn rows_of(&mut self, from: u64, to: u64) -> Vec<u8> {
            let l = self.layout.clone();
            let mut out = Vec::new();
            for t in from / T..to / T {
                let mut p = Vec::new();
                l.tile_pieces(t, std::ptr::null_mut(), &mut p);
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
        rig_on(Arc::new(SharedCache::new(budget, 0, false)), n_slots, false, [7, 7])
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
        let cache = Arc::new(SharedCache::new(64 << 20, 0, false));
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
        assert_eq!((resume, rep.tier, rep.restore_bytes), (4 * 512, Tier::Device, 0));
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
        let cache = Arc::new(SharedCache::new(64 << 20, 0, false));
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
    fn a_request_that_would_overwrite_unsaved_positions_copies_them_out_first() {
        let mut r = rig(64 << 20, 4);
        let a1 = prompt(4 * 512 + 100, 1);
        r.request(&a1, true);
        // no idle time: another conversation arrives at once
        let b = prompt(2 * 512 + 50, 2);
        let (_, rep) = r.request(&b, true);
        assert!(rep.save_bytes > 0);
        assert_eq!(r.cache.stats().entries, 1);
        // a request of the same conversation pays for nothing
        let mut b2 = b.clone();
        b2.extend(prompt(600, 3));
        assert_eq!(r.request(&b2, true).1.save_bytes, 0);
        // and the first conversation is back from host memory
        let mut a2 = a1.clone();
        a2.extend(prompt(200, 5));
        r.idle();
        let (resume, rep) = r.request(&a2, true);
        assert_eq!((resume, rep.tier), (4 * 512, Tier::Host));
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
        assert!(r
            .tier
            .save(&r.table, &mut r.dev, &mut || false)
            .is_err());
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
    fn row_layout_follows_the_caches_chunk_by_chunk_and_head_by_head() {
        let l = RowLayout::new(vec![1000, 900_000], 2 * 2 * 4096 * 16, 2, 4096, 2, 16, 512).unwrap();
        assert_eq!(l.tile_bytes(), 2 * 2 * 512 * 16);
        assert_eq!((l.pieces_per_tile(), l.tiles()), (4, 16));
        let mut p = Vec::new();
        // tile 9: chunk 1, rows 512..1024 of it
        l.tile_pieces(9, std::ptr::null_mut(), &mut p);
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
    }
}
