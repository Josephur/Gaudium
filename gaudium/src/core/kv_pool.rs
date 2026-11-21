//! Rust Shell wrappers for KV-cache / host-RAM block pool accounting & prefix reuse rules.
//!
//! All conservation, OOM, double-free, eviction, slab-boundary, and prefix-reuse decisions
//! delegate directly to the build-time verified Bend 2 core (`verified/kv_pool.bend`).

use crate::verified::{
    pfx_reuse_valid, pool_alloc, pool_block_in_bounds, pool_blocks_for, pool_can_alloc,
    pool_can_evict, pool_free, pool_init, pool_same_slab_adjacent, pool_slab_coords, pool_total,
    PoolAcct,
};

/// Host-RAM cache block size (2 MiB, mirrors `cram::Pool::BLOCK`).
pub const BLOCK_BYTES: u64 = 2 * 1024 * 1024;

/// Host-RAM slab size (1 GiB, mirrors `cram::Pool::SLAB`).
pub const SLAB_BYTES: u64 = 1024 * 1024 * 1024;

/// Blocks per 1 GiB slab (512).
pub const BLOCKS_PER_SLAB: u64 = SLAB_BYTES / BLOCK_BYTES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    OutOfBlocks { requested: u64, free: u64 },
    DoubleFree { requested: u64, allocated: u64 },
    BlockOutOfBounds { block_id: u64, capacity: u64 },
}

/// Contiguous byte run within a single host-RAM slab (never crosses a 1 GiB slab boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabRun {
    pub slab_idx: u64,
    pub first_block_in_slab: u64,
    pub byte_offset_in_slab: u64,
    pub byte_len: u64,
}

/// Checked block-pool accounting guard (`allocated + free == capacity` proven in Bend 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockPoolTracker {
    acct: PoolAcct,
}

impl BlockPoolTracker {
    pub fn new(capacity_blocks: u64) -> Self {
        Self {
            acct: pool_init(capacity_blocks),
        }
    }

    pub fn allocated(&self) -> u64 {
        let PoolAcct::PoolAcct(a, _) = self.acct;
        a
    }

    pub fn free(&self) -> u64 {
        let PoolAcct::PoolAcct(_, f) = self.acct;
        f
    }

    pub fn capacity(&self) -> u64 {
        pool_total(self.acct)
    }

    pub fn can_alloc(&self, req_blocks: u64) -> bool {
        pool_can_alloc(self.acct, req_blocks)
    }

    pub fn alloc(&mut self, req_blocks: u64) -> Result<(), PoolError> {
        if !pool_can_alloc(self.acct, req_blocks) {
            return Err(PoolError::OutOfBlocks {
                requested: req_blocks,
                free: self.free(),
            });
        }
        self.acct = pool_alloc(self.acct, req_blocks);
        Ok(())
    }

    pub fn free_blocks(&mut self, rel_blocks: u64) -> Result<(), PoolError> {
        if rel_blocks > self.allocated() {
            return Err(PoolError::DoubleFree {
                requested: rel_blocks,
                allocated: self.allocated(),
            });
        }
        self.acct = pool_free(self.acct, rel_blocks);
        Ok(())
    }

    pub fn check_block_id(&self, block_id: u64) -> Result<(), PoolError> {
        if pool_block_in_bounds(block_id, self.capacity()) {
            Ok(())
        } else {
            Err(PoolError::BlockOutOfBounds {
                block_id,
                capacity: self.capacity(),
            })
        }
    }
}

/// True iff a host-RAM cache entry is eligible for LRU eviction (`pins == 0 && !busy`).
pub fn can_evict_entry(pins: u64, busy: bool) -> bool {
    pool_can_evict(pins, busy)
}

/// Ceiling block count for `bytes` given `block_bytes` (div-by-zero safe).
pub fn blocks_needed(bytes: u64, block_bytes: u64) -> u64 {
    pool_blocks_for(bytes, block_bytes)
}

/// Maps a global `block_id` into `(slab_index, block_in_slab)`.
pub fn slab_coords(block_id: u64, blocks_per_slab: u64) -> (u64, u64) {
    pool_slab_coords(block_id, blocks_per_slab)
}

/// True iff `prev_block` and `cur_block` can coalesce into one contiguous DMA run inside one slab.
pub fn same_slab_adjacent(prev_block: u64, cur_block: u64, blocks_per_slab: u64) -> bool {
    pool_same_slab_adjacent(prev_block, cur_block, blocks_per_slab)
}

/// Coalesces block IDs into slab-bounded host byte runs (mirrors `cram::Pool::extents`).
pub fn coalesce_slab_runs(
    blocks: &[u64],
    total_bytes: u64,
    block_bytes: u64,
    blocks_per_slab: u64,
) -> Vec<SlabRun> {
    if blocks.is_empty() || total_bytes == 0 || block_bytes == 0 {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut remaining = total_bytes;
    let mut idx = 0usize;
    while idx < blocks.len() && remaining > 0 {
        let start_blk = blocks[idx];
        let (slab_idx, first_in_slab) = slab_coords(start_blk, blocks_per_slab);
        let mut take = remaining.min(block_bytes);
        remaining -= take;
        let mut prev_blk = start_blk;
        idx += 1;
        while idx < blocks.len()
            && remaining > 0
            && same_slab_adjacent(prev_blk, blocks[idx], blocks_per_slab)
        {
            let chunk = remaining.min(block_bytes);
            take += chunk;
            remaining -= chunk;
            prev_blk = blocks[idx];
            idx += 1;
        }
        runs.push(SlabRun {
            slab_idx,
            first_block_in_slab: first_in_slab,
            byte_offset_in_slab: first_in_slab.saturating_mul(block_bytes),
            byte_len: take,
        });
    }
    runs
}

/// Parameters for checking whether a prefix-cache snapshot at `snap_pos` is valid for reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixReuseQuery {
    pub snap_pos: u64,
    pub lcp: u64,
    pub prompt_len: u64,
    pub tile: u64,
    pub need_mtp: bool,
    pub mtp_upto: u64,
    pub need_dflash: bool,
    pub dflash_upto: u64,
}

/// Verified prefix-cache reuse rule (`prefix_cache.hpp`: `0 < s < lcp`, `s < P`, tile-aligned).
pub fn is_prefix_reuse_valid(q: PrefixReuseQuery) -> bool {
    pfx_reuse_valid(
        q.snap_pos,
        q.lcp,
        q.prompt_len,
        q.tile,
        q.need_mtp,
        q.mtp_upto,
        q.need_dflash,
        q.dflash_upto,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1 << 20;

    #[test]
    fn pool_conservation_oom_and_double_free() {
        // Mirrors cache_ram_test.cpp: 64 MB = 32 blocks of 2 MB
        let mut pool = BlockPoolTracker::new(32);
        assert_eq!(pool.capacity(), 32);
        assert_eq!(pool.allocated(), 0);
        assert_eq!(pool.free(), 32);

        // Entry A: 5 MB rows -> 3 blocks, 2 x 3 MB snapshots -> 2 blocks each = 7 blocks
        let b_rows = blocks_needed(2048 * 2560, BLOCK_BYTES);
        let b_snap = blocks_needed(3 * MB, BLOCK_BYTES);
        assert_eq!(b_rows, 3);
        assert_eq!(b_snap, 2);
        assert_eq!(blocks_needed(0, BLOCK_BYTES), 0);
        assert_eq!(blocks_needed(5 * MB, 0), 0);

        pool.alloc(b_rows + 2 * b_snap).unwrap();
        assert_eq!(pool.allocated(), 7);
        assert_eq!(pool.free(), 25);
        assert_eq!(pool.allocated() + pool.free(), 32);

        // OOM rejected without mutating pool
        assert_eq!(
            pool.alloc(26),
            Err(PoolError::OutOfBlocks {
                requested: 26,
                free: 25
            })
        );
        assert_eq!(pool.allocated(), 7);
        assert_eq!(pool.free(), 25);

        // Double-free / over-free rejected without mutating pool
        assert_eq!(
            pool.free_blocks(8),
            Err(PoolError::DoubleFree {
                requested: 8,
                allocated: 7
            })
        );
        assert_eq!(pool.allocated(), 7);
        assert_eq!(pool.free(), 25);

        // Valid free restores full capacity
        pool.free_blocks(7).unwrap();
        assert_eq!(pool.allocated(), 0);
        assert_eq!(pool.free(), 32);

        // Block ID bounds
        assert!(pool.check_block_id(0).is_ok());
        assert!(pool.check_block_id(31).is_ok());
        assert_eq!(
            pool.check_block_id(32),
            Err(PoolError::BlockOutOfBounds {
                block_id: 32,
                capacity: 32
            })
        );
    }

    #[test]
    fn eviction_guards_protect_pinned_and_busy_entries() {
        assert!(can_evict_entry(0, false));
        assert!(!can_evict_entry(1, false));
        assert!(!can_evict_entry(0, true));
        assert!(!can_evict_entry(3, true));
    }

    #[test]
    fn slab_runs_split_on_holes_and_1gb_boundaries() {
        // Mirrors cache_ram_test.cpp tests 5 & 7
        let runs = coalesce_slab_runs(&[3, 4, 5, 9, 10], 4 * BLOCK_BYTES + 100, BLOCK_BYTES, BLOCKS_PER_SLAB);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].slab_idx, 0);
        assert_eq!(runs[0].first_block_in_slab, 3);
        assert_eq!(runs[0].byte_len, 3 * BLOCK_BYTES);
        assert_eq!(runs[1].slab_idx, 0);
        assert_eq!(runs[1].first_block_in_slab, 9);
        assert_eq!(runs[1].byte_len, BLOCK_BYTES + 100);

        // Cross-slab boundary at 511 -> 512 (1 GiB boundary with 512 blocks/slab)
        assert!(!same_slab_adjacent(511, 512, BLOCKS_PER_SLAB));
        assert!(same_slab_adjacent(510, 511, BLOCKS_PER_SLAB));
        assert!(same_slab_adjacent(512, 513, BLOCKS_PER_SLAB));
        assert_eq!(slab_coords(511, BLOCKS_PER_SLAB), (0, 511));
        assert_eq!(slab_coords(512, BLOCKS_PER_SLAB), (1, 0));

        let cross = coalesce_slab_runs(&[510, 511, 512, 513], 4 * BLOCK_BYTES, BLOCK_BYTES, BLOCKS_PER_SLAB);
        assert_eq!(cross.len(), 2);
        assert_eq!(cross[0].slab_idx, 0);
        assert_eq!(cross[0].first_block_in_slab, 510);
        assert_eq!(cross[0].byte_len, 2 * BLOCK_BYTES);
        assert_eq!(cross[1].slab_idx, 1);
        assert_eq!(cross[1].first_block_in_slab, 0);
        assert_eq!(cross[1].byte_offset_in_slab, 0);
        assert_eq!(cross[1].byte_len, 2 * BLOCK_BYTES);
    }

    #[test]
    fn prefix_reuse_rule_enforces_last_tile_and_drafter_bounds() {
        // Mirrors cache_ram_test.cpp test 2: tile = 512, snapshot at 2560
        let base = PrefixReuseQuery {
            snap_pos: 2560,
            lcp: 3000,
            prompt_len: 2561, // P - 1 == 2560 => valid
            tile: 512,
            need_mtp: true,
            mtp_upto: 3000,
            need_dflash: false,
            dflash_upto: 0,
        };
        assert!(is_prefix_reuse_valid(base));

        // Prompt length == snap_pos (2560): last tile must run => refused
        assert!(!is_prefix_reuse_valid(PrefixReuseQuery {
            prompt_len: 2560,
            ..base
        }));
        // Diverges at token 2560 (lcp == 2560) => refused
        assert!(!is_prefix_reuse_valid(PrefixReuseQuery {
            lcp: 2560,
            ..base
        }));
        // lcp == 0 => safe (no unsigned underflow) and refused
        assert!(!is_prefix_reuse_valid(PrefixReuseQuery {
            lcp: 0,
            ..base
        }));
        // Unaligned snapshot => refused
        assert!(!is_prefix_reuse_valid(PrefixReuseQuery {
            snap_pos: 2500,
            ..base
        }));
        // Missing DFlash drafter coverage => refused
        assert!(!is_prefix_reuse_valid(PrefixReuseQuery {
            need_dflash: true,
            dflash_upto: 2048,
            ..base
        }));
        // Sufficient DFlash drafter coverage => accepted
        assert!(is_prefix_reuse_valid(PrefixReuseQuery {
            need_dflash: true,
            dflash_upto: 2560,
            ..base
        }));
    }
}
