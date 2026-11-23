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
//! charged for what is mapped, pages exist only once written, and a segment that leaves goes
//! back to the system at once. Nothing here is pinned.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::core::kv_pool::BlockPoolTracker;
use crate::engine::qwen3::prefix_cache::{shared_rows, ImageId};
use crate::hostmem::AnonMap;
use crate::verified::{
    hc_abort, hc_begin_read, hc_begin_write, hc_busy, hc_can_evict, hc_continues, hc_end_read,
    hc_end_write, hc_evict_before, hc_may_restore, hc_pins, hc_readable, hc_resume_valid,
    hc_ring_cover_after_cut, hc_ring_serves, hc_room, hc_shared_tiles, hc_state_leaves_first,
    hc_states_over, hc_tiles, hc_truncate, pc_boundary, HcEntry, HcRoom,
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
        }
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

    fn alloc(&mut self, bytes: u64) -> Option<Seg> {
        let charge = Self::charge_of(bytes);
        self.ledger.alloc(charge).ok()?;
        match AnonMap::new(bytes) {
            Ok(map) => Some(Seg { map }),
            Err(e) => {
                let _ = self.ledger.free_blocks(charge);
                eprintln!("[cache-ram] {e}");
                None
            }
        }
    }

    fn release(&mut self, seg: Seg) {
        let _ = self.ledger.free_blocks(seg.charged());
    }

    fn remove(&mut self, id: u64) {
        if let Some(mut e) = self.entries.remove(&id) {
            let mut segs: Vec<Seg> = e.rows.drain(..).collect();
            segs.extend(e.states.drain(..).map(|(_, s)| s));
            segs.extend(e.ring.take().map(|r| r.seg));
            segs.extend(e.pending.take());
            for s in segs {
                self.release(s);
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

    /// Makes `need` bytes free for `target`, evicting the entries used least recently, or says
    /// that it cannot (and then evicts nothing).
    fn make_room(&mut self, need: u64, target: Option<u64>) -> bool {
        loop {
            match hc_room(need, self.ledger.free(), self.evictable(target)) {
                HcRoom::HcFits => return true,
                HcRoom::HcRefuse => return false,
                HcRoom::HcEvict => {}
            }
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
        }
    }

    // ---- lookup ----

    /// The entry and kept position a request can resume from: the furthest position any entry
    /// of this engine offers for the prompt (the more recently used entry among equals).
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
                let better = match &best {
                    None => true,
                    Some((f, used)) => pos > f.pos || (pos == f.pos && e.used > *used),
                };
                if better {
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
        if !hc_readable(e.life) || tiles > hc_tiles(e.life) {
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
        e.life = if ok {
            hc_end_read(e.life)
        } else {
            hc_abort(e.life)
        };
        if !hc_readable(e.life) {
            self.remove(id);
        }
    }

    // ---- save ----

    /// Chooses the entry a save of the prompt `held` (with `upto_tiles` whole tiles to keep)
    /// writes into: the entry of this engine that the prompt continues (it is cut to the tiles
    /// they share), else a new one. `extra` = bytes of the states and the ring image the save
    /// wants beside the rows. `None`: the save has no room even with every other entry gone,
    /// and nothing was changed.
    #[allow(clippy::too_many_arguments)]
    pub fn open_save(
        &mut self,
        engine: EngineId,
        tile: u64,
        tile_bytes: u64,
        held: &[i32],
        images: &[ImageId],
        upto_tiles: u64,
        extra: u64,
    ) -> Option<SaveTarget> {
        // the entry this prompt continues: the one it shares the most whole tiles with
        let mut cont: Option<(u64, u64)> = None;
        for (&id, e) in &self.entries {
            if e.engine != engine || e.tile != tile || !hc_readable(e.life) {
                continue;
            }
            let (_, cap) = shared_rows(held, images, &e.tokens, &e.images);
            let tiles = hc_tiles(e.life);
            let common = hc_shared_tiles(cap, tile, tiles);
            if hc_continues(common, tiles) && cont.is_none_or(|(_, c)| common > c) {
                cont = Some((id, common));
            }
        }
        let (have, cut_bytes) = match cont {
            Some((id, common)) => {
                let e = &self.entries[&id];
                let cut: u64 = e.rows[(common as usize).min(e.rows.len())..]
                    .iter()
                    .map(Seg::charged)
                    .sum::<u64>()
                    + e.states
                        .iter()
                        .filter(|(k, _)| *k > common)
                        .map(|(_, s)| s.charged())
                        .sum::<u64>();
                (common, cut)
            }
            None => (0, 0),
        };
        let need = upto_tiles
            .saturating_sub(have)
            .saturating_mul(Self::charge_of(tile_bytes))
            .saturating_add(extra);
        let target = cont.map(|(id, _)| id);
        if hc_room(
            need,
            self.ledger.free().saturating_add(cut_bytes),
            self.evictable(target),
        ) == HcRoom::HcRefuse
        {
            self.stats.refused += 1;
            if self.verbose {
                eprintln!(
                    "[cache-ram] not saved: a prefix of {} rows needs {:.2} GiB, the budget is {:.2} GiB with {:.2} GiB in entries that cannot leave now",
                    pc_boundary(upto_tiles, tile),
                    need as f64 / (1u64 << 30) as f64,
                    self.ledger.capacity() as f64 / (1u64 << 30) as f64,
                    (self.ledger.allocated().saturating_sub(self.evictable(target))) as f64
                        / (1u64 << 30) as f64,
                );
            }
            return None;
        }
        self.tick += 1;
        let tick = self.tick;
        let id = match cont {
            Some((id, common)) => {
                let e = self.entries.get_mut(&id).expect("entry");
                e.life = hc_truncate(e.life, common);
                let keep = hc_tiles(e.life);
                let mut gone: Vec<Seg> = e.rows.drain((keep as usize).min(e.rows.len())..).collect();
                let (kept, cut): (Vec<_>, Vec<_>) =
                    e.states.drain(..).partition(|(k, _)| *k <= keep);
                e.states = kept;
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
        let charge = Self::charge_of(tile_bytes).saturating_mul(n);
        if !self.entries.contains_key(&id) || !self.make_room(charge, Some(id)) {
            return None;
        }
        let mut segs = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match self.alloc(tile_bytes) {
                Some(s) => segs.push(s),
                None => {
                    for s in segs {
                        self.release(s);
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
        for s in gone {
            self.release(s);
        }
        if empty {
            self.remove(id);
        }
        keep
    }

    /// A save starts to copy one more piece of entry `id` (a state, or the ring image) of
    /// `bytes`: where it goes. `None`: no room.
    pub fn begin_piece(&mut self, id: u64, bytes: u64) -> Option<*mut u8> {
        if !self.entries.contains_key(&id) || !self.make_room(Self::charge_of(bytes), Some(id)) {
            return None;
        }
        let mut seg = self.alloc(bytes)?;
        let ptr = seg.map.as_mut_ptr();
        let e = self.entries.get_mut(&id).expect("entry");
        if let Some(old) = e.pending.replace(seg) {
            let _ = self.ledger.free_blocks(old.charged());
        }
        let e = self.entries.get_mut(&id).expect("entry");
        e.life = hc_begin_write(e.life, hc_tiles(e.life));
        Some(ptr)
    }

    /// The copy of a state ended. `complete`: all of it is in host memory; then the entry holds
    /// the state after `tiles` whole tiles (replacing one it had for that position).
    pub fn end_state(&mut self, id: u64, tiles: u64, complete: bool) {
        let Some(e) = self.entries.get_mut(&id) else {
            return;
        };
        e.life = hc_end_write(e.life, hc_tiles(e.life));
        let mut gone: Vec<Seg> = Vec::new();
        if let Some(seg) = e.pending.take() {
            if complete && tiles <= hc_tiles(e.life) {
                if let Some(i) = e.states.iter().position(|(k, _)| *k == tiles) {
                    gone.push(e.states.remove(i).1);
                }
                let at = e.states.partition_point(|(k, _)| *k < tiles);
                e.states.insert(at, (tiles, seg));
            } else {
                gone.push(seg);
            }
        }
        for s in gone {
            self.release(s);
        }
    }

    /// The copy of the ring image ended. `complete`: all of it is in host memory; then it is
    /// the entry's image, valid from position `from` for the tiles the entry holds now. An
    /// incomplete copy is dropped and leaves the image the entry had, which goes on serving
    /// the tiles it was taken with.
    pub fn end_ring(&mut self, id: u64, from: u64, complete: bool) {
        let Some(e) = self.entries.get_mut(&id) else {
            return;
        };
        e.life = hc_end_write(e.life, hc_tiles(e.life));
        let mut gone: Vec<Seg> = Vec::new();
        if let Some(seg) = e.pending.take() {
            if complete {
                let tiles = hc_tiles(e.life);
                gone.extend(e.ring.replace(RingImage { seg, from, tiles }).map(|r| r.seg));
            } else {
                gone.push(seg);
            }
        }
        for s in gone {
            self.release(s);
        }
    }

    /// Drops the kept positions of entry `id` beyond `cap`, the ones nearest the start of the
    /// prompt first. Never the one after `keep_tiles` whole tiles.
    pub fn trim_states(&mut self, id: u64, cap: u64, keep_tiles: u64) {
        let Some(e) = self.entries.get_mut(&id) else {
            return;
        };
        let mut gone: Vec<Seg> = Vec::new();
        while hc_states_over(e.states.len() as u64, cap) {
            let mut first: Option<usize> = None;
            for (i, (k, _)) in e.states.iter().enumerate() {
                if *k == keep_tiles {
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
    pub fn new(budget_bytes: u64, ring_bytes: u64, verbose: bool) -> Self {
        let cache = HostCache::new(budget_bytes, verbose);
        let snap = cache.stats();
        Self {
            inner: Mutex::new(cache),
            snap: Mutex::new(snap),
            ring_bytes,
        }
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
/// and what the host can give now: `Ok((bytes, how it was chosen))`, 0 = off. A budget that was
/// asked for and cannot be honoured is an error that says so with the numbers.
pub fn resolve_budget(
    asked: Option<u64>,
    ring_bytes: u64,
    mem: &crate::hostmem::HostMemory,
) -> Result<(u64, String), String> {
    use crate::verified::{hc_beside_pinned, hc_budget_fits, hc_default_budget, hc_default_share};
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let available = mem.available();
    match asked {
        Some(0) => Ok((0, "off (--cache-ram 0)".into())),
        Some(b) => {
            if !hc_budget_fits(b, ring_bytes, available) {
                return Err(format!(
                    "--cache-ram {:.2} GiB cannot be honoured: with {:.2} GiB of pinned transfer buffers it needs {:.2} GiB, and {:.2} GiB are available ({}). Lower --cache-ram, or free host memory",
                    gib(b),
                    gib(ring_bytes),
                    gib(b.saturating_add(ring_bytes)),
                    gib(available),
                    mem.describe()
                ));
            }
            Ok((b, format!("--cache-ram; {}", mem.describe())))
        }
        None => {
            let b = hc_default_budget(hc_beside_pinned(available, ring_bytes));
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
        let t = c.open_save(ENG, T, TB, p, &[], tiles, at.len() as u64 * SB)?;
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
            c.end_state(t.entry, k, true);
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
        assert_eq!(c.find(ENG, T, &p[..6 * 512], &[], true, true).unwrap().pos, 3 * 512);
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
        let t = c.open_save(ENG, T, TB, &p, &[img(1)], 6, 2 * SB).unwrap();
        c.begin_rows(t.entry, 6, TB).unwrap();
        c.end_rows(t.entry, 6);
        for k in [2u64, 6] {
            c.begin_piece(t.entry, SB).unwrap();
            c.end_state(t.entry, k, true);
        }
        let mut p2 = p.clone();
        p2.extend(prompt(300, 9));
        // the same picture: resume past it; another one, or none: before its run
        assert_eq!(c.find(ENG, T, &p2, &[img(1)], true, true).unwrap().pos, 6 * 512);
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
        assert_eq!(c.find(ENG, T, &a, &[], true, true).unwrap(), Found {
            entry: ia,
            tiles: 5,
            pos: 5 * 512,
            lcp: a.len() as u64,
            cap: a.len() as u64
        });
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
        // a prefix larger than the whole budget is refused and evicts nothing for it
        assert!(save(&mut c, &prompt(9 * 512 + 50, 4), 9, &[9]).is_none());
        assert!(c.contains(a) && c.contains(d));
        assert_eq!(c.stats().refused, 1);
        // the ledger is whole: given out + free = budget
        assert_eq!(c.ledger.allocated() + c.ledger.free(), c.budget_bytes());
    }

    #[test]
    fn a_save_cut_short_keeps_the_tiles_that_finished_and_a_failed_one_leaves_nothing() {
        let mut c = HostCache::new(256 * MIB, false);
        let p = prompt(6 * 512 + 100, 0);
        let t = c.open_save(ENG, T, TB, &p, &[], 6, SB).unwrap();
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
        let t2 = c.open_save(ENG, T, TB, &p, &[], 6, SB).unwrap();
        assert_eq!((t2.entry, t2.have_tiles), (t.entry, 4));
        c.begin_rows(t2.entry, 2, TB).unwrap();
        assert_eq!(c.end_rows(t2.entry, 6), 6);
        // a state whose copy did not finish is not kept; one that finished is
        c.begin_piece(t2.entry, SB).unwrap();
        c.end_state(t2.entry, 6, false);
        assert!(c.states_of(t2.entry).is_empty());
        assert_eq!(c.held_bytes(), 6 * TB);
        c.begin_piece(t2.entry, SB).unwrap();
        c.end_state(t2.entry, 6, true);
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
        let t = c.open_save(ENG, T, TB, &p, &[], 3, SB).unwrap();
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
        c.end_ring(id, 700, true);
        assert!(c.has_ring(id));
        assert_eq!(c.begin_read(id, 2).unwrap().ring.unwrap().1, 700);
        c.end_read(id, true);
        let held = c.held_bytes();
        // a newer image replaces it
        c.begin_piece(id, SB).unwrap();
        c.end_ring(id, 900, true);
        assert_eq!(c.held_bytes(), held);
        // a copy that did not finish drops the piece and keeps the entry
        c.begin_piece(id, SB).unwrap();
        c.end_ring(id, 950, false);
        assert_eq!(c.held_bytes(), held);
        assert!(c.contains(id));
    }

    #[test]
    fn the_budget_is_checked_against_what_the_host_can_give() {
        let gib = 1u64 << 30;
        let mem = crate::hostmem::HostMemory {
            host_available: 40 * gib,
            cgroup_limit: None,
            cgroup_in_use: None,
        };
        let ring = 256 * MIB;
        // asked for and fits
        assert_eq!(resolve_budget(Some(16 * gib), ring, &mem).unwrap().0, 16 * gib);
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
    fn digests_tell_configurations_apart() {
        assert_eq!(digest128(b"model a, ctx 4096"), digest128(b"model a, ctx 4096"));
        assert_ne!(digest128(b"model a, ctx 4096"), digest128(b"model a, ctx 8192"));
        assert_ne!(digest128(b""), digest128(b"\0"));
        assert_ne!(digest128(b"abcdefgh"), digest128(b"abcdefgi"));
    }
}
