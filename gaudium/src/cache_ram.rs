//! Host-RAM tier of prefix reuse: the store (`--cache-ram`).
//!
//! Prefix reuse on the device (`engine/prefix_cache.rs`) works while the card still holds a
//! conversation. This store keeps what a later request can resume from in host memory, so a
//! conversation outlives another one in between, a switch of the model on the card and the
//! release of the card. It belongs to the server, not to an engine: an engine that is loaded
//! again finds the entries its predecessor saved, each stamped with the identity of the engine
//! that made it.
//!
//! An entry holds, for one prompt:
//! - the K/V rows of its first whole prefill tiles, one host segment per tile;
//! - the recurrent state after some of those tiles (the positions the engine kept on the
//!   device), one segment each;
//! - optionally the image of the drafter's context rings, with the first position valid in it;
//! - the prompt's tokens and image runs, by which a later prompt is compared with it.
//!
//! The store knows no device. The engine (`engine/host_tier.rs`) asks it where bytes go and
//! copies them through its transfer ring. Every decision here is the verified core's
//! (`verified/host_cache.bend`, and the host-tier section of `verified/kv_pool.bend`): which
//! entry a prompt may resume from and where, whether a prompt continues an entry, the life of
//! an entry with its bounded commit, who leaves when the budget is full, and the byte ledger
//! (`allocated + free == budget`, the locked rules of the block pool counted in bytes).
//!
//! Memory: a segment is an anonymous mapping of its own (`hostmem::AnonMap`). The budget is
//! charged for what is mapped, and pages exist only once written. No segment is mapped without
//! a leave (`hostmem::Admitted`): the room an allocation is given ([`HostCache::make_room`])
//! is a leave for its fresh memory, asked for against what the host and the control groups of
//! the process have left at that moment. The memory an engine takes beside the store (its
//! pinned transfer ring, the buffer of its first copies) is asked for here too
//! ([`HostCache::admit_beside`]), because the store can make room for it by evicting.
//! Segments of entries that
//! leave during a save are handed to that save first (a cache that is full evicts on every
//! save, and a segment that is reused costs neither an unmap nor fresh pages: measured on the
//! Gaudi 2 host, memory that was written before takes a copy at about twice the rate of fresh
//! memory); what the save does not take goes back to the system when it ends
//! ([`HostCache::settle`]). A spare segment is taken again only by an allocation of exactly its
//! size, so it counts as memory the save has only then (`hc_spare_credit`); when the host
//! cannot give what an allocation needs, the spare segments of other sizes go back to the
//! system before anything else happens (`hc_spares_return`). Only a segment that was written
//! whole becomes a spare one: a segment a copy did not finish goes back to the system at once,
//! so a spare segment is always memory the process really has. Nothing here is pinned.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::core::kv_pool::BlockPoolTracker;
use crate::engine::qwen3::prefix_cache::{shared_rows, ImageId};
use crate::hostmem::{Admitted, AnonMap};
use crate::verified::{
    hc_abort, hc_begin_read, hc_begin_write, hc_busy, hc_can_evict, hc_continue_pick,
    hc_copy_whole, hc_end_write, hc_evict_before, hc_host_counted, hc_may_read, hc_may_restore,
    hc_pins, hc_read_done, hc_readable, hc_resumable, hc_resume_better, hc_resume_valid,
    hc_ring_cover_after_cut, hc_ring_due, hc_ring_serves, hc_room, hc_save_need, hc_shared_tiles,
    hc_spare_credit, hc_spares_return, hc_state_due, hc_state_kept, hc_state_leaves_first,
    hc_state_may_leave, hc_state_survives_cut, hc_state_wanted, hc_states_over, hc_take_now,
    hc_tiles, hc_truncate, hm_fresh, pc_boundary, HcEntry, HcRoom,
};

/// Identity of the engine an entry was made by: a digest of everything that decides the bytes
/// of a row (model files, precisions, context, tile, drafter, kernels).
pub type EngineId = [u64; 2];

/// 128-bit digest of a byte string (not cryptographic: it tells engine configurations of one
/// process apart).
pub fn digest128(bytes: &[u8]) -> [u64; 2] {
    let mix = |mut x: u64| {
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        x ^ (x >> 33)
    };
    let (mut a, mut b) = (
        0x9E37_79B9_7F4A_7C15u64 ^ bytes.len() as u64,
        0xC2B2_AE3D_27D4_EB4Fu64,
    );
    let (words, rest) = bytes.as_chunks::<8>();
    for w in words {
        let v = u64::from_le_bytes(*w);
        a = mix(a ^ v).wrapping_add(0x632B_E59B_D9B4_E019);
        b = (b.rotate_left(29) ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    let mut tail = 0u64;
    for (k, &x) in rest.iter().enumerate() {
        tail |= u64::from(x) << (8 * k);
    }
    a = mix(a ^ tail);
    b = (b.rotate_left(29) ^ tail).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    [mix(a ^ b.rotate_left(17)), mix(b ^ a.rotate_left(41))]
}

/// One host segment.
#[derive(Debug)]
struct Seg {
    map: AnonMap,
}

impl Seg {
    fn charged(&self) -> u64 {
        self.map.len() as u64
    }
}

#[derive(Debug)]
struct RingImage {
    seg: Seg,
    /// First position whose row in the image is that position's.
    from: u64,
    /// Whole tiles the entry held when the image was taken: it holds the prompt's rows up to
    /// there, and serves no resume point past them.
    tiles: u64,
}

#[derive(Debug)]
struct Entry {
    engine: EngineId,
    life: HcEntry,
    tile: u64,
    /// The prompt the rows belong to (the whole prompt held when the entry was last saved).
    tokens: Vec<i32>,
    images: Vec<ImageId>,
    /// K/V rows of tile `t` at index `t`; `hc_tiles(life)` of them are complete, a write in
    /// flight has appended more.
    rows: Vec<Seg>,
    /// `(whole tiles, state)`, ascending.
    states: Vec<(u64, Seg)>,
    ring: Option<RingImage>,
    /// A state or ring image a write in flight fills.
    pending: Option<Seg>,
    /// Tick of the last save or restore.
    used: u64,
}

impl Entry {
    fn bytes(&self) -> u64 {
        self.rows.iter().map(Seg::charged).sum::<u64>()
            + self.states.iter().map(|(_, s)| s.charged()).sum::<u64>()
            + self.ring.as_ref().map_or(0, |r| r.seg.charged())
            + self.pending.as_ref().map_or(0, Seg::charged)
    }
}

/// Counters of the store since the server started, and its state now.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CacheStats {
    pub budget_bytes: u64,
    pub held_bytes: u64,
    pub entries: u64,
    /// Prompt rows whose K/V rows the entries hold.
    pub held_rows: u64,
    /// Kept positions (recurrent states) the entries hold.
    pub held_states: u64,
    pub saves: u64,
    pub save_bytes: u64,
    pub save_s: f64,
    pub restores: u64,
    pub restore_bytes: u64,
    pub restore_s: f64,
    pub evictions: u64,
    /// Saves that found no room even with every other entry gone.
    pub refused: u64,
    /// Saves a waiting request cut short (they go on after it).
    pub yielded: u64,
}

/// Where a prompt can resume from host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Found {
    pub entry: u64,
    /// The kept state to resume from: after this many whole tiles, at this row position.
    pub tiles: u64,
    pub pos: u64,
    /// Common token prefix with the entry's prompt, and what is left of it at the first image
    /// that differs.
    pub lcp: u64,
    pub cap: u64,
}

/// The entry a save writes into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveTarget {
    pub entry: u64,
    /// Whole tiles the entry already holds for the prompt being saved.
    pub have_tiles: u64,
    /// Kept positions (whole tiles) it already holds the state of.
    pub have_states: Vec<u64>,
}

/// What a restore reads: host pointers, valid until [`HostCache::end_read`].
#[derive(Debug, Clone)]
pub struct ReadView {
    /// K/V rows of tile `t` at index `t` (the first `tiles` of the entry).
    pub rows: Vec<*const u8>,
    pub state: *const u8,
    /// The drafter ring image and its first valid position.
    pub ring: Option<(*const u8, u64)>,
    pub tokens: Vec<i32>,
    pub images: Vec<ImageId>,
}

/// Of the kept positions `kept` of a save into an entry that holds the states `have` and rows up
/// to `entry_tiles` whole tiles and keeps at most `cap` positions: the ones whose state the save
/// copies and is charged for (the verified `hc_state_due`). The entry lacks them, its rows reach
/// them, and they are among the `cap` positions furthest into the prompt of everything the entry
/// would then hold; a state that would leave again at the end of the same save is not copied.
pub fn states_due(have: &[u64], kept: &[u64], entry_tiles: u64, cap: u64) -> Vec<u64> {
    // every position the entry would hold if the save copied all it could
    let mut all: Vec<u64> = have.to_vec();
    for &k in kept {
        if hc_state_wanted(all.contains(&k), k, entry_tiles) {
            all.push(k);
        }
    }
    kept.iter()
        .copied()
        .filter(|&k| {
            // positions that leave after this one (`hc_state_leaves_first`, the order of the trim)
            let above = all.iter().filter(|&&u| hc_state_leaves_first(k, u)).count() as u64;
            hc_state_due(have.contains(&k), k, entry_tiles, above, cap)
        })
        .collect()
}

/// Where a store reads what the host can give.
#[derive(Debug, Clone, Copy)]
enum HostSource {
    /// Nowhere: the budget alone bounds the store (tests of other things).
    Unasked,
    /// The system, read at every asking (`hostmem::HostMemory::read`).
    Live { budget_named: bool },
    /// A made-up host: asked with the bytes the store has mapped.
    #[cfg(test)]
    Made {
        available: fn(u64) -> u64,
        limits_known: bool,
        budget_named: bool,
    },
}

/// The store. One per server; engines reach it through [`SharedCache`].
#[derive(Debug)]
pub struct HostCache {
    /// Bytes given out and bytes free: the verified block-pool ledger, counted in bytes.
    ledger: BlockPoolTracker,
    entries: BTreeMap<u64, Entry>,
    next_id: u64,
    tick: u64,
    stats: CacheStats,
    verbose: bool,
    /// Where the store reads what the host can give this process now. Read before the store
    /// grows, and before an engine takes memory beside it.
    host: HostSource,
    /// Segments of entries that left during the copy in progress, free for it to take; emptied
    /// by [`Self::settle`]. Not charged to the budget, still the store's memory.
    spare: Vec<Seg>,
}

// SAFETY: the segments are plain host memory owned by the store; the raw pointers it hands out
// are used by the one engine thread that holds the store's lock.
unsafe impl Send for HostCache {}

impl HostCache {
    pub fn new(budget_bytes: u64, verbose: bool) -> Self {
        Self {
            ledger: BlockPoolTracker::new(budget_bytes),
            entries: BTreeMap::new(),
            next_id: 0,
            tick: 0,
            stats: CacheStats {
                budget_bytes,
                ..CacheStats::default()
            },
            verbose,
            host: HostSource::Unasked,
            spare: Vec::new(),
        }
    }

    /// A store that also asks the host what it can give before it grows: a save evicts the
    /// store's own entries first (their memory goes back to the system at once) and is refused
    /// when even that is not enough, so the store never takes memory the host does not have.
    pub fn watching_host(mut self) -> Self {
        self.host = HostSource::Live {
            budget_named: false,
        };
        self
    }

    /// The operator named this store's budget (`--cache-ram <GB>`), or did not. It matters
    /// when a limit of the process cannot be read (`hc_host_counted`).
    pub fn set_budget_named(&mut self, named: bool) {
        if let HostSource::Live { budget_named } = &mut self.host {
            *budget_named = named;
        }
    }

    /// A made-up host for tests: `available` is asked with the bytes the store has mapped.
    #[cfg(test)]
    fn set_host(&mut self, available: fn(u64) -> u64) {
        self.host = HostSource::Made {
            available,
            limits_known: true,
            budget_named: false,
        };
    }

    /// One reading of the host: the bytes it has available for this process now, whether the
    /// limits of the process are established, and whether the operator named the budget. The
    /// spare segments are still mapped, so a made-up host is told of them as held.
    fn host_reading(&self) -> (u64, bool, bool) {
        match self.host {
            HostSource::Unasked => (u64::MAX, true, true),
            HostSource::Live { budget_named } => {
                let mem = crate::hostmem::HostMemory::read();
                (mem.available(), mem.limits_known(), budget_named)
            }
            #[cfg(test)]
            HostSource::Made {
                available,
                limits_known,
                budget_named,
            } => (available(self.mapped_bytes()), limits_known, budget_named),
        }
    }

    /// What the host can give this process now, as the store may count it (`hc_host_counted`:
    /// nothing under a limit that could not be read, unless the budget was named).
    /// `u64::MAX`: not asked; the budget alone bounds the store.
    fn host_now(&self) -> u64 {
        let (available, known, named) = self.host_reading();
        hc_host_counted(known, named, available)
    }

    /// Bytes the store has mapped: what its entries hold, and the spare segments of the copy
    /// in progress.
    pub fn mapped_bytes(&self) -> u64 {
        self.ledger.allocated().saturating_add(self.spare_bytes())
    }

    fn spare_bytes(&self) -> u64 {
        self.spare.iter().map(Seg::charged).sum()
    }

    /// Spare segments of exactly `charge` bytes: the ones an allocation of that size takes
    /// again.
    fn spares_of(&self, charge: u64) -> u64 {
        self.spare.iter().filter(|s| s.charged() == charge).count() as u64
    }

    /// Bytes of the spare segments an allocation of `charge` bytes cannot take again.
    fn spare_bytes_not_of(&self, charge: u64) -> u64 {
        self.spare
            .iter()
            .filter(|s| s.charged() != charge)
            .map(Seg::charged)
            .sum()
    }

    /// The spare segments an allocation of `charge` bytes cannot take again go back to the
    /// system now.
    fn return_spares_not_of(&mut self, charge: u64) {
        self.spare.retain(|s| s.charged() == charge);
    }

    /// A copy has ended: the segments it did not take go back to the system. Between copies
    /// the store's memory is what its entries hold.
    pub fn settle(&mut self) {
        self.spare.clear();
    }

    pub fn budget_bytes(&self) -> u64 {
        self.ledger.capacity()
    }

    pub fn held_bytes(&self) -> u64 {
        self.ledger.allocated()
    }

    pub fn stats(&self) -> CacheStats {
        let mut s = self.stats;
        s.budget_bytes = self.ledger.capacity();
        s.held_bytes = self.ledger.allocated();
        s.entries = self.entries.len() as u64;
        s.held_rows = self
            .entries
            .values()
            .map(|e| pc_boundary(hc_tiles(e.life), e.tile))
            .sum();
        s.held_states = self.entries.values().map(|e| e.states.len() as u64).sum();
        s
    }

    pub fn stats_mut(&mut self) -> &mut CacheStats {
        &mut self.stats
    }

    /// Whole tiles entry `id` holds complete (0: no such entry).
    pub fn tiles_of(&self, id: u64) -> u64 {
        self.entries.get(&id).map_or(0, |e| hc_tiles(e.life))
    }

    pub fn contains(&self, id: u64) -> bool {
        self.entries.contains_key(&id)
    }

    /// Bytes a segment for `bytes` is charged.
    pub fn charge_of(bytes: u64) -> u64 {
        AnonMap::mapped_len(bytes)
    }

    /// A segment for `bytes`, charged to the budget: a spare one of that size, which is memory
    /// the store already has, else a fresh mapping taken out of `leave` (the leave
    /// [`Self::make_room`] gave for this allocation; a mapping it does not cover is refused).
    fn alloc(&mut self, bytes: u64, leave: &mut Admitted) -> Option<Seg> {
        let charge = Self::charge_of(bytes);
        self.ledger.alloc(charge).ok()?;
        // a spare segment of this size first: every byte a copy reads later is written by it
        if let Some(i) = self.spare.iter().position(|s| s.charged() == charge) {
            return Some(self.spare.swap_remove(i));
        }
        match AnonMap::new(bytes, leave) {
            Ok(map) => Some(Seg { map }),
            Err(e) => {
                let _ = self.ledger.free_blocks(charge);
                eprintln!("[cache-ram] {e}");
                None
            }
        }
    }

    /// The budget gets the segment's bytes back; the segment itself stays for the copy in
    /// progress to take ([`Self::settle`] unmaps what is left). Only for a segment that was
    /// written whole (it held rows, a state or an image of an entry): a spare segment counts as
    /// memory the process has, which is true of written pages only.
    fn release(&mut self, seg: Seg) {
        let _ = self.ledger.free_blocks(seg.charged());
        self.spare.push(seg);
    }

    /// The budget gets the segment's bytes back and the segment goes back to the system now:
    /// for a segment a copy did not finish. Some of its pages were never written, so it is not
    /// memory the process has and must not be counted as such.
    fn discard(&mut self, seg: Seg) {
        let _ = self.ledger.free_blocks(seg.charged());
        drop(seg);
    }

    fn remove(&mut self, id: u64) {
        if let Some(mut e) = self.entries.remove(&id) {
            // what a copy was still writing (the tiles past the ones the entry holds whole,
            // the piece that is pending) goes back to the system; the rest was written whole
            // and stays for the copy in progress to take
            let whole = (hc_tiles(e.life) as usize).min(e.rows.len());
            let unfinished: Vec<Seg> = e.rows.drain(whole..).chain(e.pending.take()).collect();
            let mut segs: Vec<Seg> = std::mem::take(&mut e.rows);
            segs.extend(e.states.drain(..).map(|(_, s)| s));
            segs.extend(e.ring.take().map(|r| r.seg));
            for s in segs {
                self.release(s);
            }
            for s in unfinished {
                self.discard(s);
            }
        }
    }

    /// Bytes in entries that may leave to make room for `target`.
    fn evictable(&self, target: Option<u64>) -> u64 {
        self.entries
            .iter()
            .filter(|(id, e)| hc_can_evict(hc_pins(e.life), hc_busy(e.life), Some(**id) == target))
            .map(|(_, e)| e.bytes())
            .sum()
    }

    /// Makes room for `count` segments of `seg` charged bytes each in entry `target`, or says
    /// that it cannot. What may be taken at once is the verified `hc_take_now`: the budget's
    /// free bytes, and no more than the host has available plus the spare segments of that size
    /// the allocation takes again (`hc_spare_credit`). When the host cannot give the rest, the
    /// spare segments of other sizes go back to the system first (`hc_spares_return`: they are
    /// mapped memory nothing here can use); then the entries used least recently leave. A
    /// request that cannot be met even so evicts nothing.
    ///
    /// The room comes as a leave for the fresh memory the allocation maps: what it needs
    /// beyond the spare segments it takes again (`hm_fresh`), admitted against the host figure
    /// this very answer was computed from. `None`: no room.
    fn make_room(&mut self, target: Option<u64>, seg: u64, count: u64) -> Option<Admitted> {
        let need = seg.saturating_mul(count);
        loop {
            let host = self.host_now();
            let reusable = hc_spare_credit(self.spares_of(seg), count, seg);
            let room = hc_room(
                need,
                hc_take_now(self.ledger.free(), host, reusable),
                self.evictable(target),
            );
            if room == HcRoom::HcFits {
                return Admitted::ask(hm_fresh(need, reusable), host);
            }
            if hc_spares_return(need, host, reusable, self.spare_bytes_not_of(seg)) {
                self.return_spares_not_of(seg);
                continue;
            }
            if room == HcRoom::HcRefuse || !self.evict_one(target, need) {
                return None;
            }
        }
    }

    /// The entry used least recently among those that may leave (never `target`, never one a
    /// copy reads or writes) leaves. `false`: there is none.
    fn evict_one(&mut self, target: Option<u64>, need: u64) -> bool {
        let mut victim: Option<(u64, u64)> = None;
        for (&id, e) in &self.entries {
            if !hc_can_evict(hc_pins(e.life), hc_busy(e.life), Some(id) == target) {
                continue;
            }
            if victim.is_none_or(|(_, used)| hc_evict_before(e.used, used)) {
                victim = Some((id, e.used));
            }
        }
        let Some((id, _)) = victim else { return false };
        if self.verbose {
            let e = &self.entries[&id];
            eprintln!(
                "[cache-ram] evicted entry {id}: {} rows, {:.2} GiB (room for {:.2} GiB)",
                pc_boundary(hc_tiles(e.life), e.tile),
                e.bytes() as f64 / (1u64 << 30) as f64,
                need as f64 / (1u64 << 30) as f64
            );
        }
        self.remove(id);
        self.stats.evictions += 1;
        true
    }

    /// Leave for `need` bytes of host memory beside the store's own segments, for `what` (an
    /// engine's pinned transfer ring, the buffer of its first copies): asked for against what
    /// the host has available now. When that is short, the entries used least recently leave
    /// first, as for a save, and their memory goes back to the system at once; what cannot be
    /// had even with every entry gone is refused with the numbers, and nothing leaves for it
    /// (the verified `hc_room`, then `hm_admit`).
    pub fn admit_beside(&mut self, what: &str, need: u64) -> Result<Admitted, String> {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        loop {
            // one reading for both figures: what is available, and what the entries would
            // give back, which counts only where the reading itself does
            let (available, known, named) = self.host_reading();
            let host = hc_host_counted(known, named, available);
            let evictable = hc_host_counted(known, named, self.evictable(None));
            let refused = |host: u64| {
                format!(
                    "{what} needs {:.2} GiB of host memory, and {:.2} GiB are available now{}",
                    gib(need),
                    gib(host),
                    match evictable {
                        0 => String::new(),
                        b => format!(" (the cache could give back {:.2} GiB)", gib(b)),
                    }
                )
            };
            match hc_room(need, host, evictable) {
                HcRoom::HcFits => return Admitted::ask(need, host).ok_or_else(|| refused(host)),
                HcRoom::HcRefuse => return Err(refused(host)),
                HcRoom::HcEvict => {
                    if !self.evict_one(None, need) {
                        return Err(refused(host));
                    }
                    // the entry's segments go back to the system before the host is asked again
                    self.settle();
                }
            }
        }
    }

    // ---- lookup ----

    /// The entry and kept position a request can resume from: among the positions the entries
    /// of this engine offer for the prompt, the one the verified order prefers
    /// (`hc_resume_better`: the furthest, and among equals the entry used more recently).
    pub fn find(
        &self,
        engine: EngineId,
        tile: u64,
        prompt: &[i32],
        images: &[ImageId],
        wanted: bool,
        supported: bool,
    ) -> Option<Found> {
        let p = prompt.len() as u64;
        let mut best: Option<(Found, u64)> = None;
        for (&id, e) in &self.entries {
            let same = e.engine == engine && e.tile == tile;
            if !hc_may_restore(wanted, supported, same, hc_readable(e.life)) {
                continue;
            }
            let (lcp, cap) = shared_rows(prompt, images, &e.tokens, &e.images);
            let saved_rows = pc_boundary(hc_tiles(e.life), tile);
            for &(k, _) in &e.states {
                let pos = pc_boundary(k, tile);
                if !hc_resume_valid(pos, cap, p, tile, saved_rows) {
                    continue;
                }
                if best.is_none_or(|(f, used)| hc_resume_better(pos, e.used, f.pos, used)) {
                    best = Some((
                        Found {
                            entry: id,
                            tiles: k,
                            pos,
                            lcp,
                            cap,
                        },
                        e.used,
                    ));
                }
            }
        }
        best.map(|(f, _)| f)
    }

    /// Does entry `id` hold a drafter ring image?
    pub fn has_ring(&self, id: u64) -> bool {
        self.entries.get(&id).is_some_and(|e| e.ring.is_some())
    }

    /// Whole tiles the ring image of entry `id` stands for (`None`: it has no image).
    pub fn ring_tiles(&self, id: u64) -> Option<u64> {
        self.entries
            .get(&id)
            .and_then(|e| e.ring.as_ref())
            .map(|r| r.tiles)
    }

    /// Whole tiles of the last kept position of entry `id` (0: it keeps none).
    pub fn last_state_of(&self, id: u64) -> u64 {
        self.entries
            .get(&id)
            .and_then(|e| e.states.last())
            .map_or(0, |(k, _)| *k)
    }

    /// Does entry `id` hold a ring image that serves a resume after `tiles` whole tiles?
    pub fn ring_serves(&self, id: u64, tiles: u64) -> bool {
        self.entries
            .get(&id)
            .and_then(|e| e.ring.as_ref())
            .is_some_and(|r| hc_ring_serves(r.tiles, tiles))
    }

    // ---- restore ----

    /// A restore starts to read entry `id` up to the state after `tiles` whole tiles.
    pub fn begin_read(&mut self, id: u64, tiles: u64) -> Option<ReadView> {
        self.tick += 1;
        let tick = self.tick;
        let e = self.entries.get_mut(&id)?;
        if !hc_may_read(e.life, tiles) {
            return None;
        }
        let state = e.states.iter().find(|(k, _)| *k == tiles)?.1.map.as_ptr();
        e.life = hc_begin_read(e.life);
        e.used = tick;
        let rows = pc_boundary(tiles, e.tile) as usize;
        Some(ReadView {
            rows: e.rows[..tiles as usize]
                .iter()
                .map(|s| s.map.as_ptr())
                .collect(),
            state,
            ring: e
                .ring
                .as_ref()
                .filter(|r| hc_ring_serves(r.tiles, tiles))
                .map(|r| (r.seg.map.as_ptr(), r.from)),
            tokens: e.tokens[..rows.min(e.tokens.len())].to_vec(),
            images: e
                .images
                .iter()
                .copied()
                .filter(|im| im.start < rows as u64)
                .collect(),
        })
    }

    /// The restore ended. A copy that failed leaves nothing of the entry to read later.
    pub fn end_read(&mut self, id: u64, ok: bool) {
        let Some(e) = self.entries.get_mut(&id) else {
            return;
        };
        e.life = hc_read_done(e.life, ok);
        if !hc_readable(e.life) {
            self.remove(id);
        }
    }

    // ---- save ----

    /// Chooses the entry a save of the prompt `held` writes into: the entry of this engine that
    /// the prompt continues (`hc_continue_pick`; it is cut to the tiles they share), else a new
    /// one. The save is to reach `upto_tiles` whole tiles and the state at the positions `kept`
    /// (whole tiles each; the entry keeps at most `state_cap` positions), with segments of
    /// `tile_bytes`, `state_bytes` and `ring_bytes` (0: no drafter rings). It is charged for what it adds to the entry and for nothing the
    /// entry keeps (`hc_save_need`). `None`: that has no room even with every other entry
    /// gone, and nothing was changed.
    #[allow(clippy::too_many_arguments)]
    pub fn open_save(
        &mut self,
        engine: EngineId,
        tile: u64,
        tile_bytes: u64,
        held: &[i32],
        images: &[ImageId],
        upto_tiles: u64,
        kept: &[u64],
        state_cap: u64,
        state_bytes: u64,
        ring_bytes: u64,
    ) -> Option<SaveTarget> {
        // the entry this prompt continues: the one it shares the most whole tiles with
        // (`have`: those tiles, 0 while there is none)
        let mut cont: Option<u64> = None;
        let mut have = 0u64;
        for (&id, e) in &self.entries {
            let (_, cap) = shared_rows(held, images, &e.tokens, &e.images);
            let tiles = hc_tiles(e.life);
            let common = hc_shared_tiles(cap, tile, tiles);
            let same = e.engine == engine && e.tile == tile;
            if hc_continue_pick(same, hc_readable(e.life), common, tiles, have) {
                cont = Some(id);
                have = common;
            }
        }
        // what the cut gives back, and what the entry keeps of what the save is to reach
        let entry = cont.map(|id| &self.entries[&id]);
        let cut_bytes = entry.map_or(0, |e| {
            e.rows[(have as usize).min(e.rows.len())..]
                .iter()
                .map(Seg::charged)
                .sum::<u64>()
                + e.states
                    .iter()
                    .filter(|(k, _)| !hc_state_survives_cut(*k, have))
                    .map(|(_, s)| s.charged())
                    .sum::<u64>()
        });
        // the states the entry keeps through the cut, and the kept positions the save does
        // not copy (it holds them, or they would leave again at once: `states_due`)
        let have_states: Vec<u64> = entry.map_or_else(Vec::new, |e| {
            e.states
                .iter()
                .map(|(k, _)| *k)
                .filter(|&k| hc_state_survives_cut(k, have))
                .collect()
        });
        let due = states_due(&have_states, kept, upto_tiles, state_cap);
        let held_states = kept.iter().filter(|k| !due.contains(k)).count() as u64;
        // The tiles the entry's ring image still stands for once the entry is cut: a new image
        // is copied, and charged, when that falls short of the save's last kept position
        // (`hc_ring_due`, the rule the save itself goes by).
        let image_tiles = entry
            .and_then(|e| e.ring.as_ref())
            .map(|r| hc_ring_cover_after_cut(r.tiles, have));
        let n_kept = kept.len() as u64;
        let need = hc_save_need(
            upto_tiles,
            have,
            Self::charge_of(tile_bytes),
            n_kept,
            held_states,
            Self::charge_of(state_bytes),
            hc_ring_due(
                ring_bytes,
                true,
                image_tiles.is_some(),
                image_tiles.unwrap_or(0),
                upto_tiles,
            ),
            Self::charge_of(ring_bytes),
        );
        let target = cont;
        // Everything the cut gives back counts here: a segment of it is taken again, or goes
        // back to the system before fresh memory is mapped (`make_room`).
        let free = hc_take_now(
            self.ledger.free().saturating_add(cut_bytes),
            self.host_now(),
            cut_bytes.saturating_add(self.spare_bytes()),
        );
        if hc_room(need, free, self.evictable(target)) == HcRoom::HcRefuse {
            self.stats.refused += 1;
            if self.verbose {
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                eprintln!(
                    "[cache-ram] not saved: a prefix of {} rows needs {:.2} GiB more; the budget is {:.2} GiB with {:.2} GiB in entries that cannot leave now{}",
                    pc_boundary(upto_tiles, tile),
                    gib(need),
                    gib(self.ledger.capacity()),
                    gib(self.ledger.allocated().saturating_sub(self.evictable(target))),
                    match self.host {
                        HostSource::Unasked => String::new(),
                        _ => format!(
                            ", and the host has {:.2} GiB available",
                            gib(self.host_now())
                        ),
                    },
                );
            }
            return None;
        }
        self.tick += 1;
        let tick = self.tick;
        let id = match cont {
            Some(id) => {
                let e = self.entries.get_mut(&id).expect("entry");
                e.life = hc_truncate(e.life, have);
                let keep = hc_tiles(e.life);
                let mut gone: Vec<Seg> =
                    e.rows.drain((keep as usize).min(e.rows.len())..).collect();
                let (stay, cut): (Vec<_>, Vec<_>) = e
                    .states
                    .drain(..)
                    .partition(|(k, _)| hc_state_survives_cut(*k, keep));
                e.states = stay;
                gone.extend(cut.into_iter().map(|(_, s)| s));
                if let Some(r) = e.ring.as_mut() {
                    r.tiles = hc_ring_cover_after_cut(r.tiles, keep);
                }
                e.tokens = held.to_vec();
                e.images = images.to_vec();
                e.used = tick;
                for s in gone {
                    self.release(s);
                }
                id
            }
            None => {
                self.next_id += 1;
                self.entries.insert(
                    self.next_id,
                    Entry {
                        engine,
                        life: HcEntry::HcVacant,
                        tile,
                        tokens: held.to_vec(),
                        images: images.to_vec(),
                        rows: Vec::new(),
                        states: Vec::new(),
                        ring: None,
                        pending: None,
                        used: tick,
                    },
                );
                self.next_id
            }
        };
        let e = &self.entries[&id];
        Some(SaveTarget {
            entry: id,
            have_tiles: hc_tiles(e.life),
            have_states: e.states.iter().map(|(k, _)| *k).collect(),
        })
    }

    /// A save starts to copy the rows of the next `n` tiles into entry `id`: their segments,
    /// in tile order. `None`: no room (nothing was changed).
    pub fn begin_rows(&mut self, id: u64, n: u64, tile_bytes: u64) -> Option<Vec<*mut u8>> {
        if !self.entries.contains_key(&id) {
            return None;
        }
        let mut leave = self.make_room(Some(id), Self::charge_of(tile_bytes), n)?;
        let mut segs = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match self.alloc(tile_bytes, &mut leave) {
                Some(s) => segs.push(s),
                None => {
                    // nothing was written to these yet
                    for s in segs {
                        self.discard(s);
                    }
                    return None;
                }
            }
        }
        let e = self.entries.get_mut(&id).expect("entry");
        let have = hc_tiles(e.life);
        e.life = hc_begin_write(e.life, have.saturating_add(n));
        let ptrs = segs.iter_mut().map(|s| s.map.as_mut_ptr()).collect();
        e.rows.extend(segs);
        Some(ptrs)
    }

    /// The copy of rows ended with `done_tiles` whole tiles of the entry complete in host
    /// memory (counted from tile 0): the entry holds those, and the segments of the others go.
    /// Returns the tiles it holds now.
    pub fn end_rows(&mut self, id: u64, done_tiles: u64) -> u64 {
        let Some(e) = self.entries.get_mut(&id) else {
            return 0;
        };
        e.life = hc_end_write(e.life, done_tiles);
        let keep = hc_tiles(e.life);
        let gone: Vec<Seg> = e.rows.drain((keep as usize).min(e.rows.len())..).collect();
        let empty = !hc_readable(e.life);
        // the tiles the copy did not finish
        for s in gone {
            self.discard(s);
        }
        if empty {
            self.remove(id);
        }
        keep
    }

    /// A save starts to copy one more piece of entry `id` (a state, or the ring image) of
    /// `bytes`: where it goes. `None`: no room.
    pub fn begin_piece(&mut self, id: u64, bytes: u64) -> Option<*mut u8> {
        if !self.entries.contains_key(&id) {
            return None;
        }
        let mut leave = self.make_room(Some(id), Self::charge_of(bytes), 1)?;
        let mut seg = self.alloc(bytes, &mut leave)?;
        let ptr = seg.map.as_mut_ptr();
        let e = self.entries.get_mut(&id).expect("entry");
        if let Some(old) = e.pending.replace(seg) {
            let _ = self.ledger.free_blocks(old.charged());
        }
        let e = self.entries.get_mut(&id).expect("entry");
        e.life = hc_begin_write(e.life, hc_tiles(e.life));
        Some(ptr)
    }

    /// The copy of a state ended with `done` of its `all` pieces in host memory. The entry
    /// holds the state after `tiles` whole tiles from now on (replacing one it had for that
    /// position) when the verified `hc_state_kept` says so: a whole copy, at a position inside
    /// its rows. Returns whether it does.
    pub fn end_state(&mut self, id: u64, tiles: u64, done: u64, all: u64) -> bool {
        let Some(e) = self.entries.get_mut(&id) else {
            return false;
        };
        e.life = hc_end_write(e.life, hc_tiles(e.life));
        let mut gone: Vec<Seg> = Vec::new();
        let mut unfinished: Option<Seg> = None;
        let mut kept = false;
        if let Some(seg) = e.pending.take() {
            kept = hc_state_kept(done, all, tiles, hc_tiles(e.life));
            if kept {
                if let Some(i) = e.states.iter().position(|(k, _)| *k == tiles) {
                    gone.push(e.states.remove(i).1);
                }
                let at = e.states.partition_point(|(k, _)| *k < tiles);
                e.states.insert(at, (tiles, seg));
            } else {
                unfinished = Some(seg);
            }
        }
        for s in gone {
            self.release(s);
        }
        if let Some(s) = unfinished {
            self.discard(s);
        }
        kept
    }

    /// The copy of the ring image ended with `done` of its `all` pieces in host memory. A whole
    /// copy (`hc_copy_whole`) is the entry's image from now on, valid from position `from` for
    /// the tiles the entry holds now. Any other is dropped and leaves the image the entry had,
    /// which goes on serving the tiles it was taken with. Returns whether the image was taken.
    pub fn end_ring(&mut self, id: u64, from: u64, done: u64, all: u64) -> bool {
        let Some(e) = self.entries.get_mut(&id) else {
            return false;
        };
        e.life = hc_end_write(e.life, hc_tiles(e.life));
        let mut gone: Vec<Seg> = Vec::new();
        let mut unfinished: Option<Seg> = None;
        let mut whole = false;
        if let Some(seg) = e.pending.take() {
            whole = hc_copy_whole(done, all);
            if whole {
                let tiles = hc_tiles(e.life);
                gone.extend(
                    e.ring
                        .replace(RingImage { seg, from, tiles })
                        .map(|r| r.seg),
                );
            } else {
                unfinished = Some(seg);
            }
        }
        for s in gone {
            self.release(s);
        }
        if let Some(s) = unfinished {
            self.discard(s);
        }
        whole
    }

    /// Drops the kept positions of entry `id` beyond `cap`, the ones nearest the start of the
    /// prompt first. Never the one after `keep_tiles` whole tiles (`hc_state_may_leave`).
    pub fn trim_states(&mut self, id: u64, cap: u64, keep_tiles: u64) {
        let Some(e) = self.entries.get_mut(&id) else {
            return;
        };
        let mut gone: Vec<Seg> = Vec::new();
        while hc_states_over(e.states.len() as u64, cap) {
            let mut first: Option<usize> = None;
            for (i, (k, _)) in e.states.iter().enumerate() {
                if !hc_state_may_leave(*k, keep_tiles) {
                    continue;
                }
                if first.is_none_or(|f| hc_state_leaves_first(*k, e.states[f].0)) {
                    first = Some(i);
                }
            }
            let Some(i) = first else { break };
            gone.push(e.states.remove(i).1);
        }
        for s in gone {
            self.release(s);
        }
    }

    /// Entry `id` was left by a save that will not go on (the device no longer holds what it
    /// was copying): it goes when nothing can resume from it.
    pub fn drop_unresumable(&mut self, id: u64) {
        let gone = self
            .entries
            .get(&id)
            .is_some_and(|e| !hc_resumable(e.states.len() as u64));
        if gone {
            self.remove(id);
        }
    }

    /// A copy into entry `id` failed: nothing of it may be read later.
    pub fn fail(&mut self, id: u64) {
        if let Some(e) = self.entries.get_mut(&id) {
            e.life = hc_abort(hc_begin_write(e.life, hc_tiles(e.life)));
        }
        self.remove(id);
    }

    /// Kept positions (whole tiles) of entry `id`.
    pub fn states_of(&self, id: u64) -> Vec<u64> {
        self.entries
            .get(&id)
            .map(|e| e.states.iter().map(|(k, _)| *k).collect())
            .unwrap_or_default()
    }

    /// Bytes entry `id` is charged.
    pub fn bytes_of(&self, id: u64) -> u64 {
        self.entries.get(&id).map_or(0, Entry::bytes)
    }
}

/// The store as the server shares it: the store itself, held by the engine for the length of a
/// copy, and a copy of its counters that status readers take without waiting for one.
#[derive(Debug)]
pub struct SharedCache {
    pub inner: Mutex<HostCache>,
    snap: Mutex<CacheStats>,
    /// Pinned host bytes an engine's transfer ring takes while it is loaded (planned at start).
    pub ring_bytes: u64,
}

impl SharedCache {
    /// `watch_host`: the store asks the host what it can give before it grows (a server's
    /// store does; a test's has no reason to depend on the machine it runs on).
    pub fn new(budget_bytes: u64, ring_bytes: u64, verbose: bool, watch_host: bool) -> Self {
        let cache = HostCache::new(budget_bytes, verbose);
        let cache = if watch_host {
            cache.watching_host()
        } else {
            cache
        };
        let snap = cache.stats();
        Self {
            inner: Mutex::new(cache),
            snap: Mutex::new(snap),
            ring_bytes,
        }
    }

    /// A store on a made-up host: `available` is asked with the bytes the store has mapped.
    #[cfg(test)]
    pub fn on_host(budget_bytes: u64, ring_bytes: u64, available: fn(u64) -> u64) -> Self {
        let mut cache = HostCache::new(budget_bytes, false);
        cache.set_host(available);
        let snap = cache.stats();
        Self {
            inner: Mutex::new(cache),
            snap: Mutex::new(snap),
            ring_bytes,
        }
    }

    /// [`HostCache::set_budget_named`] for a store that is being set up.
    pub fn budget_named(self, named: bool) -> Self {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_budget_named(named);
        self
    }

    /// The counters as of the last finished save or restore.
    pub fn stats(&self) -> CacheStats {
        *self.snap.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Publishes the counters of `cache` (the engine calls it before it lets go of the store).
    pub fn publish(&self, cache: &HostCache) {
        *self.snap.lock().unwrap_or_else(|e| e.into_inner()) = cache.stats();
    }
}

/// The budget a server runs with, from what the operator asked for (`None`: the default share)
/// and what the host can give now: `Ok((bytes, how it was chosen))`, 0 = off. `beside_bytes`:
/// what a loaded engine takes beside the store (its pinned transfer ring and the buffer of its
/// first copies). A budget that was asked for and cannot be honoured is an error that says so
/// with the numbers.
///
/// When the limits of this process are not established (`HostMemory::limits_known`), no
/// default budget is taken (the verified `hc_default_allowed`): the tier is off unless the
/// operator names a budget, which is then checked against what could be read.
pub fn resolve_budget(
    asked: Option<u64>,
    beside_bytes: u64,
    mem: &crate::hostmem::HostMemory,
) -> Result<(u64, String), String> {
    use crate::verified::{
        hc_beside_pinned, hc_budget_fits, hc_default_allowed, hc_default_budget, hc_default_share,
    };
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let available = mem.available();
    match asked {
        Some(0) => Ok((0, "off (--cache-ram 0)".into())),
        Some(b) => {
            if !hc_budget_fits(b, beside_bytes, available) {
                return Err(format!(
                    "--cache-ram {:.2} GiB cannot be honoured: with {:.2} GiB for the copies of a loaded engine it needs {:.2} GiB, and {:.2} GiB are available ({}). Lower --cache-ram, or free host memory",
                    gib(b),
                    gib(beside_bytes),
                    gib(b.saturating_add(beside_bytes)),
                    gib(available),
                    mem.describe()
                ));
            }
            Ok((b, format!("--cache-ram; {}", mem.describe())))
        }
        None if !hc_default_allowed(mem.limits_known()) => Ok((
            0,
            format!(
                "off: the memory limit of this process is not established ({}), and a default budget could be more than it allows; --cache-ram <GB> turns it on with a budget you name",
                mem.unread.as_deref().unwrap_or("unknown")
            ),
        )),
        None => {
            let b = hc_default_budget(hc_beside_pinned(available, beside_bytes));
            Ok((
                b,
                format!(
                    "the default: one part in {} of the {:.2} GiB available at start; --cache-ram <GB> sets it, 0 turns it off; {}",
                    hc_default_share(),
                    gib(available),
                    mem.describe()
                ),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const T: u64 = 512;
    /// Bytes of one tile's rows and of one state in these tests (whole huge pages).
    const TB: u64 = 4 * MIB;
    const SB: u64 = 2 * MIB;
    const ENG: EngineId = [1, 2];

    fn prompt(n: usize, salt: i32) -> Vec<i32> {
        (0..n as i32).map(|i| i.wrapping_mul(31) ^ salt).collect()
    }

    /// Saves the first `tiles` tiles of `p` with states after each of `at`, as an engine would:
    /// every copy completes.
    fn save(c: &mut HostCache, p: &[i32], tiles: u64, at: &[u64]) -> Option<u64> {
        let t = c.open_save(ENG, T, TB, p, &[], tiles, at, 8, SB, 0)?;
        let n = tiles.saturating_sub(t.have_tiles);
        if n > 0 {
            let segs = c.begin_rows(t.entry, n, TB)?;
            for (i, s) in segs.iter().enumerate() {
                // each tile's rows are marked with its index and the prompt's first token
                unsafe { s.write_bytes((t.have_tiles as usize + i) as u8 ^ p[0] as u8, 16) };
            }
            assert_eq!(c.end_rows(t.entry, tiles), tiles);
        }
        for &k in at {
            if t.have_states.contains(&k) {
                continue;
            }
            let ptr = c.begin_piece(t.entry, SB)?;
            unsafe { ptr.write_bytes(k as u8, 16) };
            assert!(c.end_state(t.entry, k, 1, 1));
        }
        Some(t.entry)
    }

    #[test]
    fn a_saved_prefix_is_found_by_a_prompt_that_shares_it_and_only_inside_what_it_shares() {
        let mut c = HostCache::new(256 * MIB, false);
        let p = prompt(6 * 512 + 100, 0);
        let id = save(&mut c, &p, 6, &[3, 6]).unwrap();
        assert_eq!(c.tiles_of(id), 6);
        // the bytes held are whole segments: 6 tiles and 2 states
        assert_eq!(c.held_bytes(), 6 * TB + 2 * SB);
        // an appended turn resumes at the last kept position
        let mut p2 = p.clone();
        p2.extend(prompt(300, 9));
        let f = c.find(ENG, T, &p2, &[], true, true).unwrap();
        assert_eq!((f.entry, f.pos, f.lcp), (id, 6 * 512, p.len() as u64));
        // a prompt that leaves the saved one inside tile 4 resumes at the position before it
        let mut p3 = p[..4 * 512 + 7].to_vec();
        p3.extend(prompt(900, 5));
        assert_eq!(c.find(ENG, T, &p3, &[], true, true).unwrap().pos, 3 * 512);
        // a prompt that ends at a kept position cannot resume there (its last rows are prefilled)
        assert_eq!(
            c.find(ENG, T, &p[..6 * 512], &[], true, true).unwrap().pos,
            3 * 512
        );
        // nothing shared, a cold request, another engine, another tile width: no entry
        assert!(c.find(ENG, T, &prompt(4000, 77), &[], true, true).is_none());
        assert!(c.find(ENG, T, &p2, &[], false, true).is_none());
        assert!(c.find(ENG, T, &p2, &[], true, false).is_none());
        assert!(c.find([1, 3], T, &p2, &[], true, true).is_none());
        assert!(c.find(ENG, 256, &p2, &[], true, true).is_none());
    }

    #[test]
    fn another_picture_stops_a_restore_before_it() {
        let mut c = HostCache::new(256 * MIB, false);
        let p = prompt(6 * 512 + 100, 0);
        let img = |px: u64| ImageId {
            start: 2 * 512 + 40,
            len: 256,
            rows: 16,
            cols: 16,
            pixels: [px, 1],
        };
        let t = c
            .open_save(ENG, T, TB, &p, &[img(1)], 6, &[2, 6], 8, SB, 0)
            .unwrap();
        c.begin_rows(t.entry, 6, TB).unwrap();
        c.end_rows(t.entry, 6);
        for k in [2u64, 6] {
            c.begin_piece(t.entry, SB).unwrap();
            c.end_state(t.entry, k, 1, 1);
        }
        let mut p2 = p.clone();
        p2.extend(prompt(300, 9));
        // the same picture: resume past it; another one, or none: before its run
        assert_eq!(
            c.find(ENG, T, &p2, &[img(1)], true, true).unwrap().pos,
            6 * 512
        );
        let f = c.find(ENG, T, &p2, &[img(2)], true, true).unwrap();
        assert_eq!((f.pos, f.cap), (2 * 512, 2 * 512 + 40));
        assert_eq!(c.find(ENG, T, &p2, &[], true, true).unwrap().pos, 2 * 512);
    }

    #[test]
    fn a_next_turn_extends_its_entry_and_a_branch_cuts_it() {
        let mut c = HostCache::new(256 * MIB, false);
        let p1 = prompt(4 * 512 + 100, 0);
        let id = save(&mut c, &p1, 4, &[4]).unwrap();
        let before = c.held_bytes();
        // the next turn shares every tile: the same entry grows by the new tiles only
        let mut p2 = p1.clone();
        p2.extend(prompt(1100, 3));
        assert_eq!(save(&mut c, &p2, 6, &[6]), Some(id));
        assert_eq!(c.stats().entries, 1);
        assert_eq!(c.held_bytes(), before + 2 * TB + SB);
        assert_eq!(c.states_of(id), vec![4, 6]);
        // a branch that leaves the prompt inside tile 5 continues it too (it shares 5 of 6
        // tiles): the entry is cut to them and the state past the branch point is dropped
        let mut p3 = p2[..5 * 512 + 9].to_vec();
        p3.extend(prompt(800, 8));
        assert_eq!(save(&mut c, &p3, 6, &[6]), Some(id));
        assert_eq!(c.tiles_of(id), 6);
        assert_eq!(c.states_of(id), vec![4, 6]);
        // the old branch can still resume where the two are the same
        assert_eq!(c.find(ENG, T, &p2, &[], true, true).unwrap().pos, 4 * 512);
    }

    #[test]
    fn another_conversation_gets_an_entry_of_its_own() {
        let mut c = HostCache::new(256 * MIB, false);
        // two conversations with the same first tile (a system prompt)
        let sys = prompt(600, 1);
        let mut a = sys.clone();
        a.extend(prompt(5 * 512, 2));
        let mut b = sys.clone();
        b.extend(prompt(3 * 512, 3));
        let ia = save(&mut c, &a, 5, &[5]).unwrap();
        let ib = save(&mut c, &b, 3, &[3]).unwrap();
        assert_ne!(ia, ib);
        assert_eq!(c.stats().entries, 2);
        // each comes back to its own rows
        assert_eq!(
            c.find(ENG, T, &a, &[], true, true).unwrap(),
            Found {
                entry: ia,
                tiles: 5,
                pos: 5 * 512,
                lcp: a.len() as u64,
                cap: a.len() as u64
            }
        );
        assert_eq!(c.find(ENG, T, &b, &[], true, true).unwrap().entry, ib);
    }

    #[test]
    fn a_full_budget_evicts_the_entry_used_least_recently_and_refuses_what_cannot_fit() {
        // room for 5 tiles and 2 states
        let mut c = HostCache::new(5 * TB + 2 * SB, false);
        let a = save(&mut c, &prompt(2 * 512 + 50, 1), 2, &[2]).unwrap();
        let b = save(&mut c, &prompt(2 * 512 + 50, 2), 2, &[2]).unwrap();
        assert_eq!(c.held_bytes(), 4 * TB + 2 * SB);
        // a is used again, so b is the older one
        let v = c.begin_read(a, 2).unwrap();
        assert_eq!(v.rows.len(), 2);
        c.end_read(a, true);
        // a third conversation needs 2 tiles and a state: b leaves, a stays
        let d = save(&mut c, &prompt(2 * 512 + 50, 3), 2, &[2]).unwrap();
        assert!(c.contains(a) && !c.contains(b) && c.contains(d));
        assert_eq!(c.stats().evictions, 1);
        assert!(c.held_bytes() <= c.budget_bytes());
        // the entry that left gave its segments to the save that needed the room: all three
        // were taken (two tiles, one state), none is left over
        assert_eq!(c.spare_bytes(), 0);
        c.settle();
        // an entry that is cut gives its tail back when the copy has ended
        let p = prompt(2 * 512 + 50, 3);
        let mut branch = p[..512 + 9].to_vec();
        branch.extend(prompt(300, 8));
        assert_eq!(save(&mut c, &branch, 1, &[1]), Some(d));
        assert_eq!(c.spare_bytes(), TB);
        c.settle();
        assert_eq!(c.spare_bytes(), 0);
        assert_eq!(c.held_bytes(), 3 * TB + 2 * SB);
        // a prefix larger than the whole budget is refused and evicts nothing for it
        assert!(save(&mut c, &prompt(9 * 512 + 50, 4), 9, &[9]).is_none());
        assert!(c.contains(a) && c.contains(d));
        assert_eq!(c.stats().refused, 1);
        // the ledger is whole: given out + free = budget
        assert_eq!(c.ledger.allocated() + c.ledger.free(), c.budget_bytes());
    }

    /// Memory other processes hold on the pretend host of the test below.
    static OTHERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[test]
    fn a_host_short_of_memory_bounds_the_store_below_its_budget() {
        use std::sync::atomic::Ordering;
        // a host of 20 MiB; the store's budget (64 MiB) is more than it can ever give
        fn available(held: u64) -> u64 {
            (20 * MIB).saturating_sub(OTHERS.load(Ordering::Relaxed) + held)
        }
        let mut c = HostCache::new(64 * MIB, false);
        c.set_host(available);
        let a = save(&mut c, &prompt(512 + 50, 1), 1, &[1]).unwrap();
        let b = save(&mut c, &prompt(512 + 50, 2), 1, &[1]).unwrap();
        assert_eq!(c.held_bytes(), 2 * (TB + SB));
        // 8 MiB are left on the host: a third conversation of 6 MiB fits without evicting
        let d = save(&mut c, &prompt(512 + 50, 3), 1, &[1]).unwrap();
        assert_eq!((c.stats().entries, c.stats().evictions), (3, 0));
        // another process takes what is left: the next save makes its room out of the store's
        // own entries (the one used least recently), not out of memory the host lacks
        OTHERS.store(2 * MIB, Ordering::Relaxed);
        let e = save(&mut c, &prompt(512 + 50, 4), 1, &[1]).unwrap();
        assert!(!c.contains(a) && c.contains(b) && c.contains(d) && c.contains(e));
        assert_eq!(c.stats().evictions, 1);
        assert!(c.held_bytes() + OTHERS.load(Ordering::Relaxed) <= 20 * MIB);
        // a prefix this host could not hold even with the store empty is refused, and nothing
        // is evicted for it
        assert!(save(&mut c, &prompt(5 * 512 + 50, 5), 5, &[5]).is_none());
        assert_eq!(
            (c.stats().entries, c.stats().evictions, c.stats().refused),
            (3, 1, 1)
        );
    }

    #[test]
    fn spare_segments_of_another_size_go_back_to_the_host_before_fresh_memory_is_mapped() {
        // a host of 20 MiB; the budget (64 MiB) is not what bounds the store
        fn available(held: u64) -> u64 {
            (20 * MIB).saturating_sub(held)
        }
        const HOST: u64 = 20 * MIB;
        let mut c = HostCache::new(64 * MIB, false);
        c.set_host(available);
        // an entry of one engine: three tiles of 4 MiB and a state (14 MiB)
        let a = save(&mut c, &prompt(3 * 512 + 50, 1), 3, &[3]).unwrap();
        assert_eq!(c.mapped_bytes(), 3 * TB + SB);
        // A save of another engine, whose tiles are 6 MiB: two of them and a state (14 MiB).
        // The host has 6 MiB left, so the first entry leaves, and none of its segments is of a
        // size these tiles can take again.
        const TB2: u64 = 6 * MIB;
        let p = prompt(2 * 512 + 50, 2);
        let t = c
            .open_save([9, 9], T, TB2, &p, &[], 2, &[2], 8, SB, 0)
            .unwrap();
        let segs = c.begin_rows(t.entry, 2, TB2).unwrap();
        assert!(!c.contains(a));
        assert_eq!(c.stats().evictions, 1);
        // They went back to the system before the tiles were mapped: the store never has more
        // mapped than the host can give. (Counted as free while still mapped, they would stand
        // beside the new tiles: 26 MiB on a host of 20.)
        assert_eq!(c.spare_bytes(), 0);
        assert_eq!(c.mapped_bytes(), 2 * TB2);
        assert!(c.mapped_bytes() <= HOST);
        for s in &segs {
            unsafe { s.write_bytes(7, TB2 as usize) };
        }
        assert_eq!(c.end_rows(t.entry, 2), 2);
        c.begin_piece(t.entry, SB).unwrap();
        assert!(c.end_state(t.entry, 2, 1, 1));
        assert!(c.mapped_bytes() <= HOST);
        c.settle();
        assert_eq!(c.held_bytes(), 2 * TB2 + SB);
        // A save that cannot fit even with the spare segments returned and every entry gone is
        // refused, and evicts nothing for it.
        assert!(save(&mut c, &prompt(5 * 512 + 50, 3), 5, &[5]).is_none());
        assert!(c.contains(t.entry));
        assert_eq!((c.stats().refused, c.stats().evictions), (1, 1));
    }

    #[test]
    fn a_host_with_room_leaves_the_spare_segments_to_the_save() {
        // the budget bounds this store (5 tiles and 2 states); the host has room to spare
        fn available(held: u64) -> u64 {
            (256 * MIB).saturating_sub(held)
        }
        let mut c = HostCache::new(5 * TB + 2 * SB, false);
        c.set_host(available);
        let a = save(&mut c, &prompt(2 * 512 + 50, 1), 2, &[2]).unwrap();
        let b = save(&mut c, &prompt(2 * 512 + 50, 2), 2, &[2]).unwrap();
        // a third conversation: the older entry leaves for its two tiles
        let p = prompt(2 * 512 + 50, 3);
        let t = c.open_save(ENG, T, TB, &p, &[], 2, &[2], 8, SB, 0).unwrap();
        c.begin_rows(t.entry, 2, TB).unwrap();
        assert!(!c.contains(a) && c.contains(b));
        // its tiles were taken again; its state segment, which the tiles could not use, is
        // still there for the state that follows (nothing made the store return it)
        assert_eq!(c.spare_bytes(), SB);
        c.end_rows(t.entry, 2);
        c.begin_piece(t.entry, SB).unwrap();
        assert!(c.end_state(t.entry, 2, 1, 1));
        assert_eq!(c.spare_bytes(), 0);
        assert_eq!(c.held_bytes(), 4 * TB + 2 * SB);
    }

    #[test]
    fn an_extension_is_charged_only_for_what_it_adds() {
        // room for six tiles and three states, to the byte
        let mut c = HostCache::new(6 * TB + 3 * SB, false);
        let p1 = prompt(4 * 512 + 100, 0);
        let id = save(&mut c, &p1, 4, &[2, 4]).unwrap();
        assert_eq!(c.held_bytes(), 4 * TB + 2 * SB);
        // The next turn: two more tiles and one more position. The budget has exactly that
        // left. The two states the entry keeps are in the ledger already and are not copied
        // again, so they are not charged again.
        let mut p2 = p1.clone();
        p2.extend(prompt(1100, 3));
        assert_eq!(save(&mut c, &p2, 6, &[2, 4, 6]), Some(id));
        assert_eq!(
            (c.stats().refused, c.stats().evictions, c.stats().entries),
            (0, 0, 1)
        );
        assert_eq!(c.held_bytes(), c.budget_bytes());
        assert_eq!((c.tiles_of(id), c.states_of(id)), (6, vec![2, 4, 6]));
        // a save that has nothing to add is not refused by a full budget
        let t = c
            .open_save(ENG, T, TB, &p2, &[], 6, &[2, 4, 6], 8, SB, 0)
            .unwrap();
        assert_eq!((t.entry, t.have_tiles), (id, 6));
        assert_eq!(c.stats().refused, 0);
        // what really does not fit is still refused, and the entry stays as it is
        let mut p3 = p2.clone();
        p3.extend(prompt(600, 4));
        assert!(save(&mut c, &p3, 7, &[2, 4, 6, 7]).is_none());
        assert_eq!(c.stats().refused, 1);
        assert_eq!((c.tiles_of(id), c.states_of(id)), (6, vec![2, 4, 6]));
        // a branch gives back what it cuts: three tiles and a state go, three tiles and a
        // state come, in a budget that has nothing free
        let mut p4 = p2[..3 * 512 + 9].to_vec();
        p4.extend(prompt(3 * 512, 8));
        assert_eq!(save(&mut c, &p4, 6, &[2, 6]), Some(id));
        assert_eq!(c.stats().refused, 1);
        assert_eq!((c.tiles_of(id), c.states_of(id)), (6, vec![2, 6]));
        c.settle();
        assert_eq!(c.held_bytes(), 6 * TB + 2 * SB);
    }

    #[test]
    fn a_ring_image_is_charged_while_it_does_not_reach_the_last_kept_position() {
        const RB: u64 = 2 * MIB;
        let p = prompt(6 * 512 + 100, 0);
        // The first four tiles of a prompt with their position and the rings; then two more
        // tiles and a position, where the save gives way before it takes the rings again.
        let first_turn = |c: &mut HostCache| {
            let t = c
                .open_save(ENG, T, TB, &p, &[], 4, &[4], 8, SB, RB)
                .unwrap();
            c.begin_rows(t.entry, 4, TB).unwrap();
            c.end_rows(t.entry, 4);
            c.begin_piece(t.entry, SB).unwrap();
            assert!(c.end_state(t.entry, 4, 1, 1));
            c.begin_piece(t.entry, RB).unwrap();
            assert!(c.end_ring(t.entry, 0, 1, 1));
            c.settle();
            t.entry
        };
        // room for six tiles, two states, an image, and a second image while the first is
        // replaced
        let mut c = HostCache::new(6 * TB + 2 * SB + 2 * RB, false);
        let id = first_turn(&mut c);
        assert_eq!(c.ring_tiles(id), Some(4));
        let t = c
            .open_save(ENG, T, TB, &p, &[], 6, &[4, 6], 8, SB, RB)
            .unwrap();
        assert_eq!((t.entry, t.have_tiles), (id, 4));
        c.begin_rows(id, 2, TB).unwrap();
        c.end_rows(id, 6);
        c.begin_piece(id, SB).unwrap();
        assert!(c.end_state(id, 6, 1, 1));
        c.settle();
        assert_eq!((c.last_state_of(id), c.ring_tiles(id)), (6, Some(4)));
        assert_eq!(c.budget_bytes() - c.held_bytes(), RB);
        // The save that goes on has no tile and no position left to add. It is charged for the
        // image it still owes, and there is exactly room for that.
        assert!(c
            .open_save(ENG, T, TB, &p, &[], 6, &[4, 6], 8, SB, RB)
            .is_some());
        c.begin_piece(id, RB).unwrap();
        assert!(c.end_ring(id, 0, 1, 1));
        c.settle();
        assert_eq!(c.ring_tiles(id), Some(6));
        assert_eq!(c.held_bytes(), 6 * TB + 2 * SB + RB);
        // With an image that reaches the last kept position the same save asks for nothing:
        // a budget with nothing free does not refuse it.
        let mut full = HostCache::new(6 * TB + 2 * SB + RB, false);
        let fid = first_turn(&mut full);
        // (in this budget the two new tiles and the position fit, a second image beside the
        // first does not: the extension is refused before anything is cut or copied)
        assert!(full
            .open_save(ENG, T, TB, &p, &[], 6, &[4, 6], 8, SB, RB)
            .is_none());
        assert_eq!(full.stats().refused, 1);
        assert_eq!((full.tiles_of(fid), full.ring_tiles(fid)), (4, Some(4)));
        // an engine without rings is charged for none
        assert!(full
            .open_save(ENG, T, TB, &p, &[], 6, &[4, 6], 8, SB, 0)
            .is_some());
        assert_eq!(full.stats().refused, 1);
        assert_eq!((c.stats().refused, c.stats().evictions), (0, 0));
    }

    #[test]
    fn a_save_copies_only_the_states_its_entry_will_keep() {
        // an entry with eight positions (its cap) and rows up to tile 18; the device keeps the
        // new last position, one the entry has, and an early one the entry dropped long ago
        let have: Vec<u64> = (10..=17).collect();
        assert_eq!(states_due(&have, &[3, 11, 18], 18, 8), vec![18]);
        // (the early one would leave again at once: eight positions lie above it)
        // with room for one more position than that, it still would; with room for two, it stays
        assert_eq!(states_due(&have, &[3, 11, 18], 18, 9), vec![18]);
        assert_eq!(states_due(&have, &[3, 11, 18], 18, 10), vec![3, 18]);
        // a position past the entry's rows is never copied, and a new entry takes the last ones
        assert_eq!(states_due(&have, &[18, 19], 18, 8), vec![18]);
        assert_eq!(states_due(&[], &[1, 5, 7, 8], 8, 2), vec![7, 8]);
        assert_eq!(states_due(&[], &[1, 5, 7, 8], 8, 8), vec![1, 5, 7, 8]);
        // the charge follows: the early position is not paid for
        let mut c = HostCache::new(256 * MIB, false);
        let p = prompt(18 * 512 + 100, 0);
        let id = save(&mut c, &p, 17, &have).unwrap();
        assert_eq!(c.states_of(id).len(), 8);
        let held = c.held_bytes();
        // room for one tile and one state exactly
        let mut tight = HostCache::new(held + TB + SB, false);
        let tid = save(&mut tight, &p, 17, &have).unwrap();
        let t = tight
            .open_save(ENG, T, TB, &p, &[], 18, &[3, 11, 18], 8, SB, 0)
            .unwrap();
        assert_eq!((t.entry, tight.stats().refused), (tid, 0));
    }

    #[test]
    fn among_equal_resume_points_the_entry_used_last_is_chosen() {
        let mut c = HostCache::new(256 * MIB, false);
        // two conversations with the same first tile (a system prompt), each with a kept
        // position at its end
        let sys = prompt(600, 1);
        let mut a = sys.clone();
        a.extend(prompt(5 * 512, 2));
        let mut b = sys.clone();
        b.extend(prompt(3 * 512, 3));
        let ia = save(&mut c, &a, 5, &[1, 5]).unwrap();
        let ib = save(&mut c, &b, 3, &[1, 3]).unwrap();
        assert_ne!(ia, ib);
        // a third prompt shares only that tile: both entries offer row 512, and the one used
        // last is taken
        let mut d = sys.clone();
        d.extend(prompt(900, 4));
        let f = c.find(ENG, T, &d, &[], true, true).unwrap();
        assert_eq!((f.entry, f.pos), (ib, 512));
        c.begin_read(ia, 1).unwrap();
        c.end_read(ia, true);
        assert_eq!(c.find(ENG, T, &d, &[], true, true).unwrap().entry, ia);
        // a further point is taken whichever entry was used last
        c.begin_read(ia, 1).unwrap();
        c.end_read(ia, true);
        let mut b2 = b.clone();
        b2.extend(prompt(300, 9));
        let f = c.find(ENG, T, &b2, &[], true, true).unwrap();
        assert_eq!((f.entry, f.pos), (ib, 3 * 512));
    }

    #[test]
    fn today_s_choices_and_charges() {
        // which place to resume from: the further point, then the entry used last
        assert!(hc_resume_better(1024, 1, 512, 9));
        assert!(!hc_resume_better(512, 9, 1024, 1));
        assert!(hc_resume_better(512, 9, 512, 1) && !hc_resume_better(512, 1, 512, 9));
        // which entry a save goes into: one it continues, the one it shares the most with
        assert!(hc_continue_pick(true, true, 4, 6, 3));
        assert!(!hc_continue_pick(true, true, 3, 6, 3));
        assert!(!hc_continue_pick(true, true, 2, 6, 0));
        assert!(!hc_continue_pick(false, true, 4, 6, 3) && !hc_continue_pick(true, false, 4, 6, 3));
        // what a save adds: two new tiles, one new state of three, a ring image
        assert_eq!(hc_save_need(6, 4, TB, 3, 2, SB, true, SB), 2 * TB + 2 * SB);
        assert_eq!(hc_save_need(6, 6, TB, 3, 3, SB, false, SB), 0);
        // the rings are copied (and charged) for an entry without an image and for one whose
        // image does not reach its last kept position, on an engine that has rings
        assert!(hc_ring_due(SB, true, false, 0, 6) && hc_ring_due(SB, true, true, 4, 6));
        assert!(!hc_ring_due(SB, true, true, 6, 6) && !hc_ring_due(0, true, false, 0, 6));
        assert!(!hc_ring_due(SB, false, true, 4, 6));
        // spare segments count for the allocations that take them again
        assert_eq!(hc_spare_credit(3, 2, TB), 2 * TB);
        assert_eq!(hc_spare_credit(1, 2, TB), TB);
        assert_eq!(hc_take_now(10 * MIB, 4 * MIB, 2 * MIB), 6 * MIB);
        assert_eq!(hc_take_now(5 * MIB, 4 * MIB, 2 * MIB), 5 * MIB);
        // ... and the others go back when the host is short, not before
        assert!(hc_spares_return(12 * MIB, 6 * MIB, 0, 14 * MIB));
        assert!(!hc_spares_return(12 * MIB, 12 * MIB, 0, 14 * MIB));
        assert!(!hc_spares_return(12 * MIB, 6 * MIB, 0, 0));
    }

    #[test]
    fn a_save_cut_short_keeps_the_tiles_that_finished_and_a_failed_one_leaves_nothing() {
        let mut c = HostCache::new(256 * MIB, false);
        let p = prompt(6 * 512 + 100, 0);
        let t = c.open_save(ENG, T, TB, &p, &[], 6, &[6], 8, SB, 0).unwrap();
        assert!(c.begin_rows(t.entry, 6, TB).is_some());
        // while the rows are written the entry cannot be read, found or evicted
        assert!(c.find(ENG, T, &p, &[], true, true).is_none());
        assert!(c.begin_read(t.entry, 1).is_none());
        assert_eq!(c.evictable(None), 0);
        // a request took the card after 4 tiles: the entry holds exactly those
        assert_eq!(c.end_rows(t.entry, 4), 4);
        assert_eq!(c.held_bytes(), 4 * TB);
        // without a state nothing can resume from it yet
        assert!(c.find(ENG, T, &p, &[], true, true).is_none());
        // the save goes on later: the same entry, from the tiles it has
        let t2 = c.open_save(ENG, T, TB, &p, &[], 6, &[6], 8, SB, 0).unwrap();
        assert_eq!((t2.entry, t2.have_tiles), (t.entry, 4));
        c.begin_rows(t2.entry, 2, TB).unwrap();
        assert_eq!(c.end_rows(t2.entry, 6), 6);
        // a state whose copy did not finish is not kept; one that finished is
        c.begin_piece(t2.entry, SB).unwrap();
        // (one of its two pieces did not arrive)
        assert!(!c.end_state(t2.entry, 6, 1, 2));
        assert!(c.states_of(t2.entry).is_empty());
        assert_eq!(c.held_bytes(), 6 * TB);
        c.begin_piece(t2.entry, SB).unwrap();
        assert!(c.end_state(t2.entry, 6, 2, 2));
        assert_eq!(c.find(ENG, T, &p, &[], true, true).unwrap().pos, 6 * 512);
        // a copy that failed drops the whole entry and gives its bytes back
        c.begin_rows(t2.entry, 1, TB).unwrap();
        c.fail(t2.entry);
        assert!(!c.contains(t2.entry));
        assert_eq!(c.held_bytes(), 0);
        // so does a restore that failed
        let id = save(&mut c, &p, 6, &[6]).unwrap();
        assert!(c.begin_read(id, 6).is_some());
        c.end_read(id, false);
        assert!(!c.contains(id));
        assert_eq!(c.held_bytes(), 0);
    }

    #[test]
    fn a_save_cut_short_before_its_first_tile_leaves_no_entry() {
        let mut c = HostCache::new(256 * MIB, false);
        let p = prompt(3 * 512 + 100, 0);
        let t = c.open_save(ENG, T, TB, &p, &[], 3, &[3], 8, SB, 0).unwrap();
        c.begin_rows(t.entry, 3, TB).unwrap();
        assert_eq!(c.end_rows(t.entry, 0), 0);
        assert!(!c.contains(t.entry));
        assert_eq!(c.held_bytes(), 0);
    }

    #[test]
    fn an_entry_keeps_no_more_positions_than_its_cap_and_never_drops_the_last_one() {
        let mut c = HostCache::new(256 * MIB, false);
        let p = prompt(8 * 512 + 100, 0);
        let id = save(&mut c, &p, 8, &[1, 2, 4, 6, 8]).unwrap();
        c.trim_states(id, 3, 8);
        assert_eq!(c.states_of(id), vec![4, 6, 8]);
        c.trim_states(id, 1, 4);
        assert_eq!(c.states_of(id), vec![4]);
        assert_eq!(c.held_bytes(), 8 * TB + SB);
    }

    #[test]
    fn the_ring_image_is_replaced_whole_or_dropped() {
        let mut c = HostCache::new(256 * MIB, false);
        let p = prompt(2 * 512 + 100, 0);
        let id = save(&mut c, &p, 2, &[2]).unwrap();
        c.begin_piece(id, SB).unwrap();
        assert!(c.end_ring(id, 700, 1, 1));
        assert!(c.has_ring(id));
        assert_eq!(c.begin_read(id, 2).unwrap().ring.unwrap().1, 700);
        c.end_read(id, true);
        let held = c.held_bytes();
        // a newer image replaces it
        c.begin_piece(id, SB).unwrap();
        assert!(c.end_ring(id, 900, 1, 1));
        assert_eq!(c.held_bytes(), held);
        // a copy that did not finish drops the piece and keeps the entry
        c.begin_piece(id, SB).unwrap();
        assert!(!c.end_ring(id, 950, 0, 1));
        assert_eq!(c.held_bytes(), held);
        assert!(c.contains(id));
    }

    #[test]
    fn the_budget_is_checked_against_what_the_host_can_give() {
        let gib = 1u64 << 30;
        let mem = crate::hostmem::HostMemory::of_host(40 * gib);
        let ring = 256 * MIB;
        // asked for and fits
        assert_eq!(
            resolve_budget(Some(16 * gib), ring, &mem).unwrap().0,
            16 * gib
        );
        // asked for and does not fit: an error with both numbers
        let e = resolve_budget(Some(40 * gib), ring, &mem).unwrap_err();
        assert!(e.contains("40.00 GiB") && e.contains("40.25 GiB"), "{e}");
        // off
        assert_eq!(resolve_budget(Some(0), ring, &mem).unwrap().0, 0);
        // the default never exceeds what is available (today: one part in four)
        let d = resolve_budget(None, ring, &mem).unwrap().0;
        assert!(d <= 40 * gib);
        assert_eq!(d, (40 * gib - ring) / 4);
        assert_eq!(crate::verified::hc_default_share(), 4);
    }

    #[test]
    fn a_limit_that_cannot_be_read_takes_no_default_budget() {
        let gib = 1u64 << 30;
        let beside = 384 * MIB;
        let mut mem = crate::hostmem::HostMemory::of_host(40 * gib);
        mem.unread = Some("memory.max of its own group cannot be read (Permission denied)".into());
        // no default: the cache is off, and the line says why and how to turn it on
        let (b, how) = resolve_budget(None, beside, &mem).unwrap();
        assert_eq!(b, 0);
        assert!(
            how.starts_with("off: the memory limit of this process is not established (memory.max of its own group cannot be read")
                && how.contains("--cache-ram <GB> turns it on"),
            "{how}"
        );
        // a budget the operator names is honoured, checked against what could be read
        let (b, how) = resolve_budget(Some(8 * gib), beside, &mem).unwrap();
        assert_eq!(b, 8 * gib);
        assert!(how.contains("not established"), "{how}");
        assert!(resolve_budget(Some(40 * gib), beside, &mem).is_err());
        assert_eq!(resolve_budget(Some(0), beside, &mem).unwrap().0, 0);
        // with the limits established the default is the share, as before
        mem.unread = None;
        assert_eq!(
            resolve_budget(None, beside, &mem).unwrap().0,
            (40 * gib - beside) / 4
        );
        // today's rule in one line
        assert!(crate::verified::hc_default_allowed(true));
        assert!(!crate::verified::hc_default_allowed(false));
    }

    #[test]
    fn memory_beside_the_store_is_asked_for_and_entries_leave_for_it() {
        // a host of 20 MiB; the budget (64 MiB) is not what bounds the store
        fn available(held: u64) -> u64 {
            (20 * MIB).saturating_sub(held)
        }
        let mut c = HostCache::new(64 * MIB, false);
        c.set_host(available);
        // an empty store: what the host has is what can be had
        assert_eq!(c.admit_beside("a ring", 20 * MIB).unwrap().left(), 20 * MIB);
        assert!(c.admit_beside("a ring", 20 * MIB + 1).is_err());
        // two entries, 10 MiB and 6 MiB: 4 MiB of the host are left
        let a = save(&mut c, &prompt(2 * 512 + 50, 1), 2, &[2]).unwrap();
        let b = save(&mut c, &prompt(512 + 50, 2), 1, &[1]).unwrap();
        c.settle();
        assert_eq!(c.mapped_bytes(), 16 * MIB);
        // what fits as things are evicts nothing
        assert_eq!(c.admit_beside("a buffer", 4 * MIB).unwrap().left(), 4 * MIB);
        assert_eq!((c.stats().evictions, c.stats().entries), (0, 2));
        // More than the host has left: the entry used least recently leaves, its memory goes
        // back to the system, and then the 8 MiB are there.
        let leave = c.admit_beside("its pinned transfer ring", 8 * MIB).unwrap();
        assert_eq!(leave.left(), 8 * MIB);
        assert!(!c.contains(a) && c.contains(b));
        assert_eq!((c.stats().evictions, c.spare_bytes()), (1, 0));
        assert_eq!(c.mapped_bytes(), 6 * MIB);
        // More than the host has even with every entry gone: refused with the numbers, and
        // nothing leaves for it.
        let e = c
            .admit_beside("its pinned transfer ring", 21 * MIB)
            .unwrap_err();
        assert!(
            e.contains("its pinned transfer ring needs 0.02 GiB of host memory, and 0.01 GiB are available now")
                && e.contains("the cache could give back 0.01 GiB"),
            "{e}"
        );
        assert!(c.contains(b));
        assert_eq!(c.stats().evictions, 1);
        // an entry a copy reads does not leave for it either
        c.begin_read(b, 1).unwrap();
        assert!(c.admit_beside("a buffer", 16 * MIB).is_err());
        assert!(c.contains(b));
        c.end_read(b, true);
        assert!(c.admit_beside("a buffer", 16 * MIB).is_ok());
        assert!(!c.contains(b));
        // a store that asks no host (tests of other things) admits what it is asked
        let mut plain = HostCache::new(64 * MIB, false);
        assert!(plain.admit_beside("a ring", 1 << 40).is_ok());
    }

    #[test]
    fn a_limit_that_stops_being_readable_stops_the_growth_of_an_unnamed_budget() {
        // a host with room to spare; the budget (64 MiB) is not what bounds the store
        fn available(held: u64) -> u64 {
            (256 * MIB).saturating_sub(held)
        }
        let unread = |named: bool| HostSource::Made {
            available,
            limits_known: false,
            budget_named: named,
        };
        let mut c = HostCache::new(64 * MIB, false);
        c.set_host(available);
        let a = save(&mut c, &prompt(2 * 512 + 50, 1), 2, &[2]).unwrap();
        c.settle();
        let mapped = c.mapped_bytes();
        assert_eq!(mapped, 2 * TB + SB);
        // The limit of the process can no longer be read, and nobody named the budget: the
        // store takes no fresh memory. Nothing is asked for beside it, and no entry leaves
        // for that (what it would give back does not count either).
        c.host = unread(false);
        let e = c
            .admit_beside("its pinned transfer ring", 2 * MIB)
            .unwrap_err();
        assert!(e.contains("0.00 GiB are available now"), "{e}");
        assert!(c.contains(a));
        assert_eq!(c.stats().evictions, 0);
        // A save goes on with the memory the store has: the older entry leaves for it and
        // its segments are taken again; nothing more is mapped.
        let b = save(&mut c, &prompt(2 * 512 + 50, 2), 2, &[2]).unwrap();
        assert!(!c.contains(a) && c.contains(b));
        c.settle();
        assert_eq!(c.mapped_bytes(), mapped);
        // a larger one does not fit in what the store has, and is refused
        assert!(save(&mut c, &prompt(3 * 512 + 50, 3), 3, &[3]).is_none());
        assert!(c.contains(b));
        // A budget the operator named is the operator's word: the store goes on counting what
        // could be read.
        c.host = unread(true);
        assert!(c.admit_beside("its pinned transfer ring", 2 * MIB).is_ok());
        assert!(save(&mut c, &prompt(3 * 512 + 50, 3), 3, &[3]).is_some());
        assert_eq!(c.stats().entries, 2);
        // today's rule in one place
        use crate::verified::hc_host_counted as counted;
        assert_eq!((counted(true, false, 7), counted(false, true, 7)), (7, 7));
        assert_eq!(counted(false, false, 7), 0);
    }

    #[test]
    fn a_segment_a_copy_did_not_finish_goes_back_to_the_system_at_once() {
        fn available(held: u64) -> u64 {
            (20 * MIB).saturating_sub(held)
        }
        let mut c = HostCache::new(64 * MIB, false);
        c.set_host(available);
        let p = prompt(4 * 512 + 50, 1);
        let t = c.open_save(ENG, T, TB, &p, &[], 4, &[4], 8, SB, 0).unwrap();
        // four tiles are mapped for the copy (16 of the host's 20 MiB); a fifth would not be
        assert!(c.begin_rows(t.entry, 6, TB).is_none());
        assert_eq!(c.mapped_bytes(), 0);
        let segs = c.begin_rows(t.entry, 4, TB).unwrap();
        assert_eq!(c.mapped_bytes(), 4 * TB);
        unsafe { segs[0].write_bytes(1, TB as usize) };
        // The copy is cut after one tile. The three it did not finish were never written:
        // they are not memory the store has, so they are unmapped, not kept as spare segments
        // (which count as memory at hand for the next allocation).
        assert_eq!(c.end_rows(t.entry, 1), 1);
        assert_eq!((c.spare_bytes(), c.mapped_bytes()), (0, TB));
        // the same for a state whose copy is cut
        c.begin_piece(t.entry, SB).unwrap();
        assert_eq!(c.mapped_bytes(), TB + SB);
        assert!(!c.end_state(t.entry, 1, 0, 1));
        assert_eq!((c.spare_bytes(), c.mapped_bytes()), (0, TB));
        // and for everything a failed copy was writing
        let segs = c.begin_rows(t.entry, 2, TB).unwrap();
        assert_eq!(segs.len(), 2);
        c.fail(t.entry);
        assert_eq!(c.mapped_bytes(), c.spare_bytes());
        assert!(c.spare_bytes() <= TB);
        c.settle();
        assert_eq!((c.mapped_bytes(), c.held_bytes()), (0, 0));
    }

    #[test]
    fn digests_tell_configurations_apart() {
        assert_eq!(
            digest128(b"model a, ctx 4096"),
            digest128(b"model a, ctx 4096")
        );
        assert_ne!(
            digest128(b"model a, ctx 4096"),
            digest128(b"model a, ctx 8192")
        );
        assert_ne!(digest128(b""), digest128(b"\0"));
        assert_ne!(digest128(b"abcdefgh"), digest128(b"abcdefgi"));
    }
}
