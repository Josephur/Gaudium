//! Rust Shell wrappers for KV-cache / host-RAM block pool accounting & prefix reuse rules.
//!
//! All conservation, OOM, double-free, eviction, slab-boundary, and prefix-reuse decisions
//! delegate directly to the build-time verified Bend 2 core (`verified/kv_pool.bend`).

use crate::verified::{
    pfx_reuse_valid, pool_alloc, pool_can_alloc, pool_can_evict, pool_free, pool_init, pool_total,
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
}

/// True iff a host-RAM cache entry is eligible for LRU eviction (`pins == 0 && !busy`).
pub fn can_evict_entry(pins: u64, busy: bool) -> bool {
    pool_can_evict(pins, busy)
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

        // Entry A: 7 blocks
        pool.alloc(7).unwrap();
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
    }

    #[test]
    fn eviction_guards_protect_pinned_and_busy_entries() {
        assert!(can_evict_entry(0, false));
        assert!(!can_evict_entry(1, false));
        assert!(!can_evict_entry(0, true));
        assert!(!can_evict_entry(3, true));
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
        assert!(!is_prefix_reuse_valid(PrefixReuseQuery { lcp: 0, ..base }));
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
