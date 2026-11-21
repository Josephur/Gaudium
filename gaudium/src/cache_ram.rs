//! Host-RAM prefix cache tier (`prefix_cache.hpp` + `cache_ram.hpp`).
//!
//! Backed directly by the formally verified Bend 2 core (`core::kv_pool::BlockPoolTracker`,
//! `blocks_needed`, `can_evict_entry`, `coalesce_slab_runs`, `is_prefix_reuse_valid`).

use std::collections::BTreeMap;

use crate::core::kv_pool::{
    blocks_needed, can_evict_entry, coalesce_slab_runs, is_prefix_reuse_valid, BlockPoolTracker,
    PrefixReuseQuery, SlabRun, BLOCKS_PER_SLAB, BLOCK_BYTES, SLAB_BYTES,
};

// ============================================================================
// 1. Prefix Cache Planner (`prefix_cache.hpp`)
// ============================================================================

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^= x >> 33;
    x
}

/// Byte hash matching `pfx::hash_bytes`.
pub fn hash_bytes(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    let (chunks, rem) = bytes.as_chunks::<8>();
    for c in chunks {
        let w = u64::from_le_bytes(*c);
        h = splitmix64(h ^ w).wrapping_add(0x632B_E59B_D9B4_E019);
    }
    let mut tail = 0u64;
    for (k, &b) in rem.iter().enumerate() {
        tail |= (b as u64) << (8 * k);
    }
    splitmix64(h ^ tail ^ (bytes.len() as u64))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageKey {
    pub span0: usize,
    pub ntok: usize,
    pub hash: u64,
}

/// Builds per-position prompt keys where image-pad spans carry bit 63 set so they NEVER collide with token IDs.
pub fn make_keys(prompt: &[i32], imgs: &[ImageKey]) -> Vec<u64> {
    let mut k: Vec<u64> = prompt.iter().map(|&tok| (tok as u32) as u64).collect();
    for im in imgs {
        let hbytes = im.hash.to_le_bytes();
        for j in 0..im.ntok {
            let idx = im.span0 + j;
            if idx >= k.len() {
                break;
            }
            let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((j as u64) + 1);
            k[idx] = (1u64 << 63) | (hash_bytes(&hbytes, seed) >> 1);
        }
    }
    k
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snap {
    pub pos: usize,
    pub slot: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefixPlan {
    pub lcp: usize,
    pub reuse: usize,
    pub restore_slot: Option<usize>,
    pub take: Vec<Snap>,
    pub keep: Vec<Snap>,
    pub why: String,
}

#[derive(Debug, Clone, Default)]
pub struct PrefixCache {
    pub nslots: usize,
    pub tile: usize,
    pub keys: Vec<u64>,
    pub mtp_upto: usize,
    pub df_upto: usize,
    pub snaps: Vec<Snap>,
    pub kv_dirty: usize,
}

impl PrefixCache {
    pub fn new(nslots: usize, tile: usize) -> Self {
        Self {
            nslots,
            tile: tile.max(1),
            ..Self::default()
        }
    }

    pub fn keep_pos(&self) -> usize {
        self.snaps.last().map(|s| s.pos).unwrap_or(0)
    }

    pub fn plan(
        &self,
        new_keys: &[u64],
        want: bool,
        need_mtp: bool,
        need_df: bool,
        hints: &[usize],
    ) -> PrefixPlan {
        let mut p = PrefixPlan::default();
        let prompt_len = new_keys.len();
        if self.nslots == 0 {
            p.why = "prefix cache off".into();
            return p;
        }
        let m = self.keys.len().min(prompt_len);
        let mut lcp = 0usize;
        while lcp < m && self.keys[lcp] == new_keys[lcp] {
            lcp += 1;
        }
        p.lcp = lcp;

        if !want {
            p.why = "cache_prompt false".into();
        } else if self.snaps.is_empty() {
            p.why = if self.keys.is_empty() {
                "no earlier prompt".into()
            } else {
                "no snapshot held".into()
            };
        } else {
            for snap in self.snaps.iter().rev() {
                let valid = is_prefix_reuse_valid(PrefixReuseQuery {
                    snap_pos: snap.pos as u64,
                    lcp: lcp as u64,
                    prompt_len: prompt_len as u64,
                    tile: self.tile as u64,
                    need_mtp,
                    mtp_upto: self.mtp_upto as u64,
                    need_dflash: need_df,
                    dflash_upto: self.df_upto as u64,
                });
                if valid {
                    p.reuse = snap.pos;
                    p.restore_slot = Some(snap.slot);
                    break;
                }
            }
            if p.reuse == 0 {
                p.why = if lcp < self.tile + 1 {
                    format!("common prefix {lcp} < one tile")
                } else {
                    "no snapshot inside the common prefix / for this drafter".into()
                };
            }
        }

        // New snapshot candidates strictly above `p.reuse`, up to `prompt_len - 1`
        let mut cands = Vec::new();
        let mut add_cand = |h: usize| {
            if prompt_len == 0 {
                return;
            }
            let s = (h / self.tile) * self.tile;
            if s >= self.tile && s > p.reuse && s < prompt_len {
                cands.push(s);
            }
        };
        if prompt_len > 0 {
            add_cand(prompt_len - 1);
        }
        for &h in hints {
            add_cand(h);
        }
        cands.sort_unstable();
        cands.dedup();

        let mut all: Vec<(usize, Option<usize>)> = self
            .snaps
            .iter()
            .filter(|s| s.pos <= p.reuse)
            .map(|s| (s.pos, Some(s.slot)))
            .collect();
        for c in cands {
            all.push((c, None));
        }
        all.sort_by_key(|a| std::cmp::Reverse(a.0));
        all.truncate(self.nslots);

        let mut used = vec![false; self.nslots];
        for &(_, slot) in &all {
            if let Some(s) = slot {
                if s < used.len() {
                    used[s] = true;
                }
            }
        }
        for entry in &mut all {
            if entry.1.is_none() {
                if let Some(free_slot) = used.iter().position(|&u| !u) {
                    used[free_slot] = true;
                    entry.1 = Some(free_slot);
                    p.take.push(Snap {
                        pos: entry.0,
                        slot: free_slot,
                    });
                }
            }
        }
        for &(pos, slot) in &all {
            if let Some(s) = slot {
                p.keep.push(Snap { pos, slot: s });
            }
        }
        p.take.sort_by_key(|s| s.pos);
        p.keep.sort_by_key(|s| s.pos);
        p
    }

    pub fn commit(
        &mut self,
        plan: &PrefixPlan,
        new_keys: Vec<u64>,
        fills_mtp: bool,
        fills_df: bool,
    ) {
        let prompt_len = new_keys.len();
        self.mtp_upto = if fills_mtp {
            prompt_len
        } else {
            self.mtp_upto.min(plan.reuse)
        };
        self.df_upto = if fills_df {
            prompt_len
        } else {
            self.df_upto.min(plan.reuse)
        };
        self.keys = new_keys;
        self.snaps = plan.keep.clone();
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.snaps.clear();
        self.mtp_upto = 0;
        self.df_upto = 0;
    }
}

// ============================================================================
// 2. Host-RAM 2 MiB Block / 1 GiB Slab Pool (`cache_ram.hpp`)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub is_snap: bool,
    pub a: usize,
    pub b: usize,
    pub bytes: u64,
    pub blocks: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct RamEntry {
    pub id: u64,
    pub gen: u64,
    pub pc: PrefixCache,
    pub e: usize,
    pub segs: Vec<Segment>,
    pub tick: u64,
    pub pins: u64,
    pub busy: bool,
}

impl RamEntry {
    pub fn block_count(&self) -> u64 {
        self.segs.iter().map(|s| s.blocks.len() as u64).sum()
    }

    pub fn data_bytes(&self) -> u64 {
        self.segs.iter().map(|s| s.bytes).sum()
    }

    pub fn has_snap_seg(&self, pos: usize) -> bool {
        self.segs.iter().any(|s| s.is_snap && s.a == pos)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamTotals {
    pub saves: u64,
    pub extends: u64,
    pub hits: u64,
    pub evictions: u64,
    pub replaced: u64,
    pub refused: u64,
    pub covered: u64,
}

/// Checked host-RAM prefix cache pool backed by verified `BlockPoolTracker`.
pub struct RamPool {
    tracker: BlockPoolTracker,
    used: Vec<bool>,
    hint: usize,
    tick: u64,
    next_id: u64,
    entries: BTreeMap<u64, RamEntry>,
    pub totals: RamTotals,
}

impl RamPool {
    pub fn new(budget_bytes: u64) -> Option<Self> {
        let nblocks = budget_bytes / BLOCK_BYTES;
        if nblocks == 0 {
            return None;
        }
        Some(Self {
            tracker: BlockPoolTracker::new(nblocks),
            used: vec![false; nblocks as usize],
            hint: 0,
            tick: 0,
            next_id: 0,
            entries: BTreeMap::new(),
            totals: RamTotals::default(),
        })
    }

    pub fn nblocks(&self) -> u64 {
        self.tracker.capacity()
    }

    pub fn free_blocks(&self) -> u64 {
        self.tracker.free()
    }

    pub fn allocated_blocks(&self) -> u64 {
        self.tracker.allocated()
    }

    pub fn budget_bytes(&self) -> u64 {
        self.nblocks() * BLOCK_BYTES
    }

    pub fn used_bytes(&self) -> u64 {
        self.allocated_blocks() * BLOCK_BYTES
    }

    pub fn slab_count(&self) -> usize {
        self.budget_bytes().div_ceil(SLAB_BYTES) as usize
    }

    pub fn slab_bytes(&self, slab_idx: usize) -> u64 {
        let off = (slab_idx as u64) * SLAB_BYTES;
        SLAB_BYTES.min(self.budget_bytes().saturating_sub(off))
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn tokens_held(&self) -> usize {
        self.entries.values().map(|e| e.e).sum()
    }

    pub fn get(&self, id: u64) -> Option<&RamEntry> {
        self.entries.get(&id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut RamEntry> {
        self.entries.get_mut(&id)
    }

    pub fn touch(&mut self, id: u64) {
        self.tick += 1;
        if let Some(e) = self.entries.get_mut(&id) {
            e.tick = self.tick;
        }
    }

    pub fn create(&mut self) -> u64 {
        self.next_id += 1;
        self.tick += 1;
        let id = self.next_id;
        self.entries.insert(
            id,
            RamEntry {
                id,
                gen: 0,
                pc: PrefixCache::new(8, 512),
                e: 0,
                segs: Vec::new(),
                tick: self.tick,
                pins: 0,
                busy: true,
            },
        );
        id
    }

    pub fn alloc_blocks(&mut self, bytes: u64) -> Option<Vec<u64>> {
        let need = blocks_needed(bytes, BLOCK_BYTES);
        if self.tracker.alloc(need).is_err() {
            return None;
        }
        let mut out = Vec::with_capacity(need as usize);
        let cap = self.used.len();
        let mut b = self.hint;
        while b < cap && (out.len() as u64) < need {
            if !self.used[b] {
                self.used[b] = true;
                out.push(b as u64);
            }
            b += 1;
        }
        debug_assert_eq!(out.len() as u64, need);
        while self.hint < cap && self.used[self.hint] {
            self.hint += 1;
        }
        Some(out)
    }

    pub fn release_blocks(&mut self, blocks: &[u64]) {
        for &b in blocks {
            let idx = b as usize;
            if idx < self.used.len() && self.used[idx] {
                self.used[idx] = false;
                if idx < self.hint {
                    self.hint = idx;
                }
            }
        }
        let _ = self.tracker.free_blocks(blocks.len() as u64);
    }

    pub fn remove(&mut self, id: u64) -> bool {
        let Some(entry) = self.entries.remove(&id) else {
            return false;
        };
        for seg in &entry.segs {
            self.release_blocks(&seg.blocks);
        }
        true
    }

    pub fn best(
        &self,
        keys: &[u64],
        need_mtp: bool,
        need_df: bool,
        hints: &[usize],
        lmax: usize,
    ) -> Option<(u64, usize)> {
        let mut best_id = None;
        let mut best_reuse = 0usize;
        let mut best_tick = 0u64;
        for entry in self.entries.values() {
            if entry.busy || entry.e > lmax {
                continue;
            }
            let r = entry.pc.plan(keys, true, need_mtp, need_df, hints).reuse;
            if r > best_reuse || (r > 0 && r == best_reuse && entry.tick > best_tick) {
                best_reuse = r;
                best_id = Some(entry.id);
                best_tick = entry.tick;
            }
        }
        best_id.map(|id| (id, best_reuse))
    }

    pub fn prefix_of(xk: &[u64], xe: usize, yk: &[u64], ye: usize) -> bool {
        if xe > ye || xk.len() <= xe || yk.len() <= xe {
            return false;
        }
        xk[..=xe] == yk[..=xe]
    }

    pub fn covering(&self, pc: &PrefixCache, e: usize) -> Option<u64> {
        for y in self.entries.values() {
            if y.busy || !Self::prefix_of(&pc.keys, e, &y.pc.keys, y.e) {
                continue;
            }
            let all_snaps = pc.snaps.iter().all(|s| y.has_snap_seg(s.pos));
            if all_snaps && y.pc.mtp_upto >= pc.mtp_upto.min(e) && y.pc.df_upto >= pc.df_upto.min(e)
            {
                return Some(y.id);
            }
        }
        None
    }

    pub fn prefixes_of(&self, target_id: u64) -> Vec<u64> {
        let Some(y) = self.entries.get(&target_id) else {
            return Vec::new();
        };
        self.entries
            .values()
            .filter(|x| {
                x.id != target_id
                    && can_evict_entry(x.pins, x.busy)
                    && Self::prefix_of(&x.pc.keys, x.e, &y.pc.keys, y.e)
            })
            .map(|x| x.id)
            .collect()
    }

    /// Evicts LRU unpinned/non-busy entries (never `keep_id`) until `free_blocks() >= need`.
    /// Atomic: if even evicting all eligible entries cannot reach `need`, evicts nothing and returns `false`.
    pub fn make_room<F>(&mut self, need: u64, keep_id: Option<u64>, mut on_evict: F) -> bool
    where
        F: FnMut(u64),
    {
        if need <= self.free_blocks() {
            return true;
        }
        let mut eligible: Vec<(u64, u64, u64)> = self
            .entries
            .values()
            .filter(|e| Some(e.id) != keep_id && can_evict_entry(e.pins, e.busy))
            .map(|e| (e.tick, e.id, e.block_count()))
            .collect();
        let can_free: u64 = self.free_blocks() + eligible.iter().map(|&(_, _, b)| b).sum::<u64>();
        if can_free < need {
            return false;
        }
        eligible.sort_by_key(|&(tick, _, _)| tick);
        for (_, id, _) in eligible {
            if need <= self.free_blocks() {
                break;
            }
            on_evict(id);
            self.totals.evictions += 1;
            self.remove(id);
        }
        need <= self.free_blocks()
    }

    /// Coalesces block IDs into contiguous slab-bounded DMA runs via verified `coalesce_slab_runs`.
    pub fn extents(&self, blocks: &[u64], bytes: u64) -> Vec<SlabRun> {
        coalesce_slab_runs(blocks, bytes, BLOCK_BYTES, BLOCKS_PER_SLAB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1 << 20;

    fn synth_keys(conv: u64, n: usize) -> Vec<u64> {
        (0..n).map(|i| conv * 1_000_003 + (i as u64)).collect()
    }

    fn put_entry(
        pool: &mut RamPool,
        keys: Vec<u64>,
        positions: &[usize],
        row_bytes: u64,
        snap_bytes: u64,
    ) -> Option<u64> {
        let id = pool.create();
        let e = *positions.last().unwrap();
        let band_bytes = row_bytes * (e as u64);
        let band_blocks = pool.alloc_blocks(band_bytes)?;
        let mut segs = vec![Segment {
            is_snap: false,
            a: 0,
            b: e,
            bytes: band_bytes,
            blocks: band_blocks,
        }];
        let mut snaps = Vec::new();
        for (slot, &pos) in positions.iter().enumerate() {
            let sblocks = pool.alloc_blocks(snap_bytes)?;
            segs.push(Segment {
                is_snap: true,
                a: pos,
                b: pos,
                bytes: snap_bytes,
                blocks: sblocks,
            });
            snaps.push(Snap { pos, slot });
        }
        let ent = pool.get_mut(id).unwrap();
        ent.pc.tile = 512;
        ent.pc.nslots = 8;
        ent.pc.mtp_upto = keys.len();
        ent.pc.keys = keys;
        ent.pc.snaps = snaps;
        ent.e = e;
        ent.segs = segs;
        ent.busy = false;
        ent.gen += 1;
        Some(id)
    }

    #[test]
    fn mirrors_all_seven_scenarios_of_cache_ram_test_cpp() {
        let mut p = RamPool::new(64 * MB).unwrap();
        assert_eq!(
            (p.nblocks(), p.free_blocks(), p.budget_bytes()),
            (32, 32, 64 * MB)
        );

        // 1. Accounting in whole 2 MB blocks, snapshots included
        let ka = synth_keys(1, 3000);
        let kb = synth_keys(2, 3000);
        let a = put_entry(&mut p, ka.clone(), &[1024, 2560], 2048, 3 * MB).unwrap();
        assert_eq!(p.get(a).unwrap().block_count(), 7);
        assert_eq!(p.free_blocks(), 25);
        assert_eq!(p.used_bytes(), 14 * MB);
        assert_eq!(p.tokens_held(), 2560);

        // 2. Planner rule: [0, s] equal, s <= P - 1, drafter bounds, context cap
        assert_eq!(p.best(&ka, true, false, &[], 1 << 20), Some((a, 2560)));
        assert_eq!(
            p.best(&ka[..2561], true, false, &[], 1 << 20),
            Some((a, 2560))
        );
        assert_eq!(
            p.best(&ka[..2560], true, false, &[], 1 << 20),
            Some((a, 1024))
        );
        let mut ka3 = ka.clone();
        ka3[2000] ^= 1;
        assert_eq!(p.best(&ka3, true, false, &[], 1 << 20), Some((a, 1024)));
        let mut ka4 = ka.clone();
        ka4[1024] ^= 1;
        assert_eq!(p.best(&ka4, true, false, &[], 1 << 20), None);
        assert_eq!(p.best(&kb, true, false, &[], 1 << 20), None);
        assert_eq!(p.best(&ka, false, true, &[], 1 << 20), None); // no DFlash coverage
        assert_eq!(p.best(&ka, true, false, &[], 2000), None); // lmax too small

        // 3. LRU eviction, pin & keep guards
        let b = put_entry(&mut p, kb, &[2560], 2048, 3 * MB).unwrap();
        assert_eq!(p.free_blocks(), 20);
        let kc = synth_keys(3, 9000);
        let c = put_entry(&mut p, kc, &[8192], 2048, 3 * MB).unwrap();
        assert_eq!((p.get(c).unwrap().block_count(), p.free_blocks()), (10, 10));
        p.touch(a); // LRU order now: b, c, a
        let mut gone = Vec::new();
        assert!(p.make_room(12, None, |id| gone.push(id)));
        assert_eq!(gone, vec![b]);
        assert_eq!(p.free_blocks(), 15);
        assert!(p.get(b).is_none());

        p.get_mut(c).unwrap().pins += 1;
        gone.clear();
        assert!(!p.make_room(24, None, |id| gone.push(id))); // 15 + 7 < 24 -> nothing evicted
        assert!(gone.is_empty() && p.get(a).is_some());
        assert!(p.make_room(20, None, |id| gone.push(id)));
        assert_eq!(gone, vec![a]);
        assert_eq!(p.free_blocks(), 22);
        p.get_mut(c).unwrap().pins -= 1;

        gone.clear();
        assert!(!p.make_room(30, Some(c), |id| gone.push(id))); // keep entry never evicted
        assert!(gone.is_empty());
        assert!(p.make_room(32, None, |id| gone.push(id)));
        assert_eq!(
            (gone, p.free_blocks(), p.entry_count(), p.totals.evictions),
            (vec![c], 32, 0, 3)
        );

        // 4. Prefix-of-longer-entry & covering
        let kd1 = synth_keys(4, 2100);
        let kd2 = synth_keys(4, 5000);
        let d1 = put_entry(&mut p, kd1.clone(), &[2048], 2048, 3 * MB).unwrap();
        let d2 = put_entry(&mut p, kd2.clone(), &[2048, 4608], 2048, 3 * MB).unwrap();
        assert_eq!(p.prefixes_of(d2), vec![d1]);
        assert!(p.prefixes_of(d1).is_empty());
        let mut q = PrefixCache::new(8, 512);
        q.keys = kd1.clone();
        q.snaps = vec![Snap { pos: 2048, slot: 0 }];
        q.mtp_upto = 2100;
        let cov = p.covering(&q, 2048);
        assert!(cov == Some(d1) || cov == Some(d2));
        q.snaps = vec![Snap { pos: 1024, slot: 0 }, Snap { pos: 2048, slot: 1 }];
        p.remove(d1);
        assert_eq!(p.covering(&q, 2048), None); // d2 lacks snapshot at 1024
        let mut kd3 = kd2.clone();
        kd3[2048] ^= 1;
        assert!(!RamPool::prefix_of(&kd3, 2048, &kd2, 4608));
        assert!(RamPool::prefix_of(&kd1, 2048, &kd2, 4608));

        // 5 & 6. Host runs & block return
        let ex = p.extents(&[3, 4, 5, 9, 10], 4 * BLOCK_BYTES + 100);
        assert_eq!(ex.len(), 2);
        assert_eq!(ex[0].byte_len, 3 * BLOCK_BYTES);
        assert_eq!(ex[1].byte_len, BLOCK_BYTES + 100);
        p.remove(d2);
        assert_eq!((p.free_blocks(), p.used_bytes()), (32, 0));

        // 7. Multi-slab 1 GiB boundary split (510, 511 | 512, 513)
        let big = RamPool::new(3 * SLAB_BYTES).unwrap();
        assert_eq!((big.slab_count(), big.slab_bytes(2)), (3, SLAB_BYTES));
        let cross = big.extents(&[510, 511, 512, 513], 4 * BLOCK_BYTES);
        assert_eq!(cross.len(), 2);
        assert_eq!((cross[0].slab_idx, cross[0].byte_len), (0, 2 * BLOCK_BYTES));
        assert_eq!(
            (
                cross[1].slab_idx,
                cross[1].byte_offset_in_slab,
                cross[1].byte_len
            ),
            (1, 0, 2 * BLOCK_BYTES)
        );
    }
}
