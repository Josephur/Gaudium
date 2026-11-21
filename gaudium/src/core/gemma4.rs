//! Rust Shell wrappers for Gemma 4 deinterleaved channel layouts, attention/multimodal
//! dimension guards, RoPE tables, and atomic chunked host-memory pinning.
//!
//! All dimension predicates and host-pinning transitions delegate to `verified/gemma4_bounds.bend`.

use crate::verified::{
    g4_audio_ctx_valid, g4_heads_valid, g4_kv_share_valid, g4_lane, g4_nat, g4_ple_valid,
    g4_rope_vecs, g4_vision_heads_valid, g4_vperm, pin_add_extent, pin_huge_ratio_ok,
    pin_is_usable, pin_mapped_count, pin_rollback, PinState,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4Error {
    InvalidHeads { kind: &'static str, nh: u64, nkv: u64, dh: u64, pairs: u64 },
    InvalidPle { ple_dim: u64, ple_vocab: u64, vocab: u64 },
    InvalidKvShare { n_layers: u64, kv_shared: u64 },
    InvalidAudioCtx { left: u64, right: u64 },
    InvalidVisionHeads { v_heads: u64, v_head_dim: u64, v_hidden: u64 },
    InvalidHostPinState(PinState),
}

/// Position of channel `c` inside its 128-channel block (`g4l::lane`).
pub fn lane_in_block(c: u64) -> u64 {
    g4_lane(c)
}

/// Natural channel for TPC vector half `h in {0, 1}`, lane `j in 0..64` (`g4l::nat`).
pub fn natural_channel(h: u64, j: u64) -> u64 {
    g4_nat(h, j)
}

/// V row index for key slot `j` (`g4l::vperm`).
pub fn v_row_perm(j: u64) -> u64 {
    g4_vperm(j)
}

/// Number of 128-lane RoPE cos/sin table vectors: `ceil(pairs / 128)`.
pub fn rope_table_vecs(pairs: u64) -> u64 {
    g4_rope_vecs(pairs)
}

/// Converts a natural-order per-channel f32 table `{DH}` (`DH % 128 == 0`) into the Gemma 4 TPC view.
pub fn deinterleave_channel_table(natural: &[f32]) -> Vec<f32> {
    assert_eq!(natural.len() % 128, 0, "Gemma 4 channel table length must be a multiple of 128");
    let mut out = vec![0.0f32; natural.len()];
    let blocks = natural.len() / 128;
    for b in 0..blocks {
        for h in 0..2u64 {
            for j in 0..64u64 {
                let nat_c = natural_channel(h, j) as usize;
                out[128 * b + 64 * (h as usize) + (j as usize)] = natural[128 * b + nat_c];
            }
        }
    }
    out
}

/// Gemma 4 RoPE generator (supports full and proportional rotation, matching `g4l::Rope`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4Rope {
    pub head_dim: u64,
    pub pairs: u64,
    pub theta: f64,
}

impl Gemma4Rope {
    pub fn rotated_pairs(rope_type: &str, partial_factor: f64, head_dim: u64) -> u64 {
        if rope_type == "proportional" {
            ((partial_factor * (head_dim as f64)) as u64) / 2
        } else {
            head_dim / 2
        }
    }

    pub fn table_vecs(&self) -> u64 {
        rope_table_vecs(self.pairs)
    }

    pub fn freq(&self, i: u64) -> f64 {
        self.theta.powf(-2.0 * (i as f64) / (self.head_dim as f64))
    }

    /// Generates `(cos, sin)` tables of shape `[hi - lo, 128 * table_vecs()]` in Gemma 4 TPC view.
    pub fn build_rows<F>(&self, lo: u64, hi: u64, mut pos_of: F) -> (Vec<f32>, Vec<f32>)
    where
        F: FnMut(u64) -> u64,
    {
        let rv = self.table_vecs();
        let width = (128 * rv) as usize;
        let nrows = hi.saturating_sub(lo) as usize;
        let mut cos_tab = vec![1.0f32; width * nrows];
        let mut sin_tab = vec![0.0f32; width * nrows];
        for r in lo..hi {
            let p = pos_of(r) as f64;
            let base = ((r - lo) as usize) * width;
            for b in 0..rv {
                for h in 0..2u64 {
                    for j in 0..64u64 {
                        let i = 128 * b + natural_channel(h, j);
                        if i >= self.pairs {
                            continue;
                        }
                        let angle = p * self.freq(i);
                        let idx = base + (128 * b + 64 * h + j) as usize;
                        cos_tab[idx] = angle.cos() as f32;
                        sin_tab[idx] = angle.sin() as f32;
                    }
                }
            }
        }
        (cos_tab, sin_tab)
    }
}

pub fn validate_gemma4_heads(
    kind: &'static str,
    nh: u64,
    nkv: u64,
    dh: u64,
    pairs: u64,
) -> Result<(), Gemma4Error> {
    if g4_heads_valid(nh, nkv, dh, pairs) {
        Ok(())
    } else {
        Err(Gemma4Error::InvalidHeads {
            kind,
            nh,
            nkv,
            dh,
            pairs,
        })
    }
}

pub fn validate_gemma4_ple(ple_dim: u64, ple_vocab: u64, vocab: u64) -> Result<(), Gemma4Error> {
    if g4_ple_valid(ple_dim, ple_vocab, vocab) {
        Ok(())
    } else {
        Err(Gemma4Error::InvalidPle {
            ple_dim,
            ple_vocab,
            vocab,
        })
    }
}

pub fn validate_gemma4_kv_share(n_layers: u64, kv_shared: u64) -> Result<(), Gemma4Error> {
    if g4_kv_share_valid(n_layers, kv_shared) {
        Ok(())
    } else {
        Err(Gemma4Error::InvalidKvShare {
            n_layers,
            kv_shared,
        })
    }
}

pub fn validate_gemma4_audio_ctx(left: u64, right: u64) -> Result<(), Gemma4Error> {
    if g4_audio_ctx_valid(left, right) {
        Ok(())
    } else {
        Err(Gemma4Error::InvalidAudioCtx { left, right })
    }
}

pub fn validate_gemma4_vision_heads(
    v_heads: u64,
    v_head_dim: u64,
    v_hidden: u64,
) -> Result<(), Gemma4Error> {
    if g4_vision_heads_valid(v_heads, v_head_dim, v_hidden) {
        Ok(())
    } else {
        Err(Gemma4Error::InvalidVisionHeads {
            v_heads,
            v_head_dim,
            v_hidden,
        })
    }
}

/// Atomic chunked host-pinning guard (mirrors `common.hpp` `HugeMem` / `pin_extents`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPinGuard {
    state: PinState,
}

impl Default for HostPinGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl HostPinGuard {
    pub const fn new() -> Self {
        Self {
            state: PinState::PinUnmapped,
        }
    }

    pub const fn state(&self) -> PinState {
        self.state
    }

    pub fn mapped_extents(&self) -> u64 {
        pin_mapped_count(self.state)
    }

    pub fn is_usable(&self) -> bool {
        pin_is_usable(self.state)
    }

    pub fn record_extent(&mut self, is_last: bool) -> Result<PinState, Gemma4Error> {
        let prev = self.state;
        let next = pin_add_extent(prev, is_last);
        if next == prev {
            return Err(Gemma4Error::InvalidHostPinState(prev));
        }
        self.state = next;
        Ok(next)
    }

    pub fn rollback(&mut self) -> PinState {
        self.state = pin_rollback(self.state);
        self.state
    }

    pub fn huge_ratio_meets_min(huge_bytes: u64, total_bytes: u64, min_pct: u64) -> bool {
        pin_huge_ratio_ok(huge_bytes, total_bytes, min_pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_lane_and_nat_are_exact_inverses() {
        let mut seen_nat = [false; 128];
        for h in 0..2u64 {
            for j in 0..64u64 {
                let c = natural_channel(h, j);
                assert!(c < 128);
                assert!(!seen_nat[c as usize]);
                seen_nat[c as usize] = true;
                // TPC bf16 load deinterleaves lane positions: even positions -> h=0, odd -> h=1
                assert_eq!(lane_in_block(c), 2 * j + h);
            }
        }
        // vperm is a bijection on 0..256
        let mut seen_v = [false; 256];
        for j in 0..256u64 {
            let p = v_row_perm(j) as usize;
            assert!(p < 256 && !seen_v[p]);
            seen_v[p] = true;
        }
    }

    #[test]
    fn gemma4_model_family_guards_match_checkpoints() {
        // Gemma 4 31B: sliding 16Q/8KV dh=256, global 16Q/4KV dh=512
        validate_gemma4_heads("sliding", 16, 8, 256, 128).unwrap();
        validate_gemma4_heads("global", 16, 4, 512, 128).unwrap();
        assert!(validate_gemma4_heads("bad", 16, 8, 128, 64).is_err());
        assert!(validate_gemma4_heads("bad", 16, 0, 256, 128).is_err());

        // PLE & KV sharing (E2B / E4B)
        validate_gemma4_ple(0, 0, 262_144).unwrap();
        validate_gemma4_ple(256, 262_144, 262_144).unwrap();
        assert!(validate_gemma4_ple(256, 1000, 262_144).is_err());
        assert!(validate_gemma4_ple(100, 262_144, 262_144).is_err());

        validate_gemma4_kv_share(35, 20).unwrap();
        assert!(validate_gemma4_kv_share(35, 35).is_err());
        assert!(validate_gemma4_kv_share(0, 0).is_err());

        // Causal audio & vision
        validate_gemma4_audio_ctx(13, 0).unwrap();
        assert!(validate_gemma4_audio_ctx(13, 1).is_err());
        assert!(validate_gemma4_audio_ctx(1, 0).is_err());
        validate_gemma4_vision_heads(16, 72, 1152).unwrap();
        assert!(validate_gemma4_vision_heads(16, 70, 1120).is_err()); // 70 % 4 != 0

        // RoPE table at pos 0 is cos=1, sin=0
        let rope = Gemma4Rope {
            head_dim: 512,
            pairs: Gemma4Rope::rotated_pairs("proportional", 0.25, 512),
            theta: 1_000_000.0,
        };
        assert_eq!(rope.pairs, 64);
        assert_eq!(rope.table_vecs(), 1);
        let (c, s) = rope.build_rows(0, 2, |r| r);
        assert_eq!(c.len(), 256);
        assert_eq!(s.len(), 256);
        assert!(c[..128].iter().all(|&x| (x - 1.0).abs() < 1e-6));
        assert!(s[..128].iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn host_pin_guard_enforces_atomic_rollback() {
        let mut pin = HostPinGuard::new();
        assert!(!pin.is_usable());
        assert_eq!(pin.mapped_extents(), 0);

        pin.record_extent(false).unwrap();
        pin.record_extent(false).unwrap();
        assert_eq!(pin.mapped_extents(), 2);
        assert!(!pin.is_usable()); // partial is never usable

        // Rollback clears mapped count to 0 and absorbs further adds
        pin.rollback();
        assert_eq!(pin.mapped_extents(), 0);
        assert!(!pin.is_usable());
        assert!(pin.record_extent(true).is_err());

        // Full mapping succeeds cleanly
        let mut ok = HostPinGuard::new();
        ok.record_extent(false).unwrap();
        ok.record_extent(true).unwrap();
        assert!(ok.is_usable());
        assert_eq!(ok.mapped_extents(), 2);
        assert!(HostPinGuard::huge_ratio_meets_min(60, 100, 50));
        assert!(!HostPinGuard::huge_ratio_meets_min(49, 100, 50));
    }
}
