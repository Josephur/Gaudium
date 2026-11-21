//! Bit-exact CPU reference of the Gaudium on-card Gumbel-max sampler (`samp_ref.hpp`).
//!
//! Uses identical IEEE-754 single-precision arithmetic order, identical murmur3 position-keyed
//! Gumbel noise, identical `mlog`/`mexp` polynomials, identical 16-ary 4-round BF16 order-key
//! threshold search, and identical 64-lane reduction tree as the Gaudi 2 TPC sampler kernels.

use crate::core::budget::bounded_top_k;
use crate::fp8::bf16_to_f32;

pub const BLK: usize = 1024; // vocab block (8 bf16 vectors of 128 lanes)
pub const NTH: usize = 16; // test thresholds per search round
pub const NIT: usize = 4; // rounds per phase (16-ary over 16-bit bf16 keys)
pub const DELTA: f32 = 20.5;
pub const NEG: f32 = -1e30;
pub const PINF: f32 = 3.0e38;

// Device sampler flag bits (`spi[3]`)
pub const F_TOPP: i32 = 1;
pub const F_MINP: i32 = 2;
pub const F_COUPLED: i32 = 4;
pub const F_DTRUNC: i32 = 8;
pub const F_GREEDY: i32 = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: u64,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub coupled: bool,
    pub draft_trunc: bool,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            coupled: true,
            draft_trunc: true,
        }
    }
}

impl SampleParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature.is_nan() || self.temperature <= 0.0
    }

    pub fn to_dev(&self, vocab: u64) -> DevSampleParams {
        let mut d = DevSampleParams::default();
        if self.is_greedy() {
            d.f[0] = 1.0;
            d.f[1] = DELTA;
            d.f[2] = 1.0;
            d.f[3] = 0.0;
            d.i[1] = (self.seed & 0xFFFF_FFFF) as u32 as i32;
            d.i[2] = (self.seed >> 32) as u32 as i32;
            d.i[3] = F_GREEDY;
            return d;
        }
        let inv_t = 1.0f32 / self.temperature;
        d.f[0] = inv_t;
        d.f[1] = DELTA / inv_t;
        d.f[2] = self.top_p;
        d.f[3] = if self.min_p > 0.0 {
            ((self.min_p as f64).ln() as f32) / inv_t
        } else {
            0.0
        };
        let clamped_k = if self.top_k > 0 {
            bounded_top_k(self.top_k, vocab)
        } else {
            0
        };
        d.i[0] = clamped_k.min(i32::MAX as u64) as i32;
        d.i[1] = (self.seed & 0xFFFF_FFFF) as u32 as i32;
        d.i[2] = (self.seed >> 32) as u32 as i32;
        let mut flags = 0i32;
        if self.top_p > 0.0 && self.top_p < 1.0 {
            flags |= F_TOPP;
        }
        if self.min_p > 0.0 {
            flags |= F_MINP;
        }
        if self.coupled {
            flags |= F_COUPLED;
        }
        if self.draft_trunc {
            flags |= F_DTRUNC;
        }
        d.i[3] = flags;
        d
    }
}

/// Device parameter tensors: `spf f32 {8}`, `spi i32 {8}`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DevSampleParams {
    pub f: [f32; 8],
    pub i: [i32; 8],
}

// ---- Murmur3 finaliser over Weyl-keyed position counter ----
#[inline]
pub fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

#[inline]
pub fn row_key(seed_lo: u32, seed_hi: u32, seq_pos: u32) -> u32 {
    let base = fmix32(seed_lo ^ fmix32(seed_hi ^ 0x5BD1_E995));
    fmix32(base.wrapping_add(seq_pos.wrapping_mul(0x9E37_79B9)))
}

#[inline]
pub fn elem_hash(rk: u32, v: u32) -> u32 {
    fmix32(rk ^ v.wrapping_mul(0x27D4_EB2F))
}

#[inline]
pub fn u01(h: u32) -> f32 {
    let u = ((h >> 9) as f32) + 0.5f32;
    u * 1.192_092_9e-7f32 // 2^-23
}

// ---- Polynomial log / exp matching the Gaudi 2 TPC sampler kernels bit-for-bit ----
const LC: [f32; 9] = [
    1.0,
    -0.499_999_88,
    0.333_341_84,
    -0.250_021_55,
    0.199_569_78,
    -0.165_606_96,
    0.149_490_95,
    -0.143_773_3,
    0.087_439_455,
];
const EC: [f32; 6] = [
    1.0,
    0.5,
    0.166_665_05,
    0.041_666_35,
    0.008_369_151,
    0.001_394_111_1,
];
const LN2F: f32 = std::f32::consts::LN_2;
const LOG2EF: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const RND: f32 = 12_582_912.0;

#[inline]
pub fn mlog(y: f32) -> f32 {
    let b = y.to_bits() as i32;
    let mut e = (b >> 23) - 127;
    let mut mb = (b & 0x007F_FFFF) | 0x3F80_0000;
    if mb > 0x3FB5_04F3 {
        mb -= 0x0080_0000;
        e += 1;
    }
    let t = f32::from_bits(mb as u32) - 1.0f32;
    let mut q = LC[8];
    for &c in LC[..8].iter().rev() {
        q *= t;
        q += c;
    }
    let p = q * t;
    let r = (e as f32) * LN2F;
    r + p
}

#[inline]
pub fn mexp(x: f32) -> f32 {
    let y = x * LOG2EF;
    let n = (y + RND) - RND;
    let mut r = x - (n * LN2_HI);
    r -= n * LN2_LO;
    let mut q = EC[5];
    for &c in EC[..5].iter().rev() {
        q *= r;
        q += c;
    }
    let p = (q * r) + 1.0f32;
    let ni = n as i32;
    let sc = f32::from_bits(((ni + 127) as u32) << 23);
    p * sc
}

#[inline]
pub fn gumbel(rk: u32, v: u32) -> f32 {
    let u = u01(elem_hash(rk, v));
    let e = -mlog(u);
    -mlog(e)
}

// ---- BF16 16-bit order-preserving keys ----
#[inline]
pub fn key_of_bits(b: u32) -> i32 {
    if (b & 0x8000) != 0 {
        ((!b) & 0xFFFF) as i32
    } else {
        (b | 0x8000) as i32
    }
}

#[inline]
pub fn bits_of_key(k: i32) -> u32 {
    let ku = k as u32;
    if (ku & 0x8000) != 0 {
        ku & 0x7FFF
    } else {
        (!ku) & 0xFFFF
    }
}

#[inline]
pub fn val_of_key(k: i32) -> f32 {
    f32::from_bits(bits_of_key(k) << 16)
}

#[inline]
pub fn key_of(f: f32) -> i32 {
    key_of_bits(f.to_bits() >> 16)
}

#[inline]
pub fn key_ceil(f: f32) -> i32 {
    let mut k = key_of(f);
    if val_of_key(k) < f {
        k += 1;
    }
    k
}

/// 64-lane butterfly reduction tree matching TPC `REDUCE64`.
pub fn reduce64(a: &[f32; 64]) -> f32 {
    let mut v = [0.0f32; 16];
    for i in 0..16 {
        let s1 = a[i] + a[16 + i];
        let s2 = a[32 + i] + a[48 + i];
        v[i] = s1 + s2;
    }
    let mut d = 8usize;
    while d >= 1 {
        let mut w = [0.0f32; 16];
        for i in 0..16 {
            w[i] = v[i] + v[i ^ d];
        }
        v = w;
        d >>= 1;
    }
    v[0]
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchState {
    pub lo: i32,
    pub hi: i32,
    pub done: bool,
    pub target: f32,
    pub max_logit: f32,
}

impl Default for SearchState {
    fn default() -> Self {
        Self {
            lo: 0,
            hi: 0,
            done: true,
            target: 0.0,
            max_logit: 0.0,
        }
    }
}

pub fn round_thresholds(lo: i32, hi: i32) -> ([i32; NTH], [f32; NTH]) {
    let n = hi - lo + 1;
    let mut tk = [0i32; NTH];
    let mut tv = [PINF; NTH];
    for j in 0..NTH {
        if n <= NTH as i32 {
            tk[j] = lo + (j as i32);
            tv[j] = if (j as i32) < n {
                val_of_key(tk[j])
            } else {
                PINF
            };
        } else {
            let k = lo + ((j as i64) * (n as i64) / (NTH as i64)) as i32;
            tk[j] = k;
            tv[j] = val_of_key(k);
        }
    }
    (tk, tv)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleResult {
    pub token: usize,
    pub tau_key: i32,
    pub thr: f32,
    pub state: SearchState,
    pub z_best: f32,
    pub second: Option<usize>,
    pub z_second: f32,
}

fn block_max(logits_bf16: &[u16], b: usize) -> f32 {
    let start = b * BLK;
    let end = logits_bf16.len().min(start + BLK);
    let mut mx = NEG;
    for &bits in &logits_bf16[start..end] {
        mx = mx.max(bf16_to_f32(bits));
    }
    mx
}

fn block_partial(
    logits_bf16: &[u16],
    b: usize,
    tv: &[f32; NTH],
    phase_b: bool,
    m: f32,
    inv_t: f32,
) -> [f32; NTH] {
    let mut acc = [[0.0f32; 64]; NTH];
    let rem = logits_bf16.len().saturating_sub(b * BLK);
    let nv = (rem / 128).min(8);
    for q in 0..nv {
        for h in 0..2 {
            for k in 0..64 {
                let v = b * BLK + 128 * q + 2 * k + h;
                let l = bf16_to_f32(logits_bf16[v]);
                let w = if phase_b {
                    let mut x = (l - m) * inv_t;
                    if x < -30.0 {
                        x = -30.0;
                    }
                    mexp(x)
                } else {
                    1.0
                };
                for (acc_row, &t_val) in acc.iter_mut().zip(tv.iter()) {
                    acc_row[k] += if l >= t_val { w } else { 0.0 };
                }
            }
        }
    }
    // Handle any non-128-aligned tail in host unit tests gracefully
    let tail_start = b * BLK + nv * 128;
    let tail_end = logits_bf16.len().min((b + 1) * BLK);
    for (offset, &bits) in logits_bf16[tail_start..tail_end].iter().enumerate() {
        let l = bf16_to_f32(bits);
        let lane = offset % 64;
        let w = if phase_b {
            let mut x = (l - m) * inv_t;
            if x < -30.0 {
                x = -30.0;
            }
            mexp(x)
        } else {
            1.0
        };
        for (acc_row, &t_val) in acc.iter_mut().zip(tv.iter()) {
            acc_row[lane] += if l >= t_val { w } else { 0.0 };
        }
    }
    let mut out = [0.0f32; NTH];
    for j in 0..NTH {
        out[j] = reduce64(&acc[j]);
    }
    out
}

fn search_round(
    logits_bf16: &[u16],
    round: usize,
    prev: SearchState,
    dev: &DevSampleParams,
    trunc_active: bool,
    bm: &[f32],
) -> SearchState {
    let m = prev.max_logit;
    let inv_t = dev.f[0];
    let (lo, hi, done) = if round == 0 {
        (
            key_ceil(m - dev.f[1]),
            key_of(m),
            !(dev.i[0] > 0) || !trunc_active,
        )
    } else if round == NIT {
        (
            prev.lo,
            key_of(m),
            (dev.i[3] & F_TOPP) == 0 || !trunc_active,
        )
    } else {
        (prev.lo, prev.hi, prev.done)
    };
    let mut s = SearchState {
        lo,
        hi,
        done,
        target: if round > NIT { prev.target } else { 0.0 },
        max_logit: m,
    };
    if done {
        s.done = true;
        return s;
    }
    let phase_b = round >= NIT;
    let (tk, tv) = round_thresholds(lo, hi);
    let mut tot = [0.0f32; NTH];
    for (b, &bmax) in bm.iter().enumerate() {
        if bmax >= tv[0] || bmax.is_nan() {
            let p16 = block_partial(logits_bf16, b, &tv, phase_b, m, inv_t);
            for (t_slot, &p_val) in tot.iter_mut().zip(p16.iter()) {
                *t_slot += p_val;
            }
        }
    }
    let target = if round == NIT {
        dev.f[2] * tot[0]
    } else {
        s.target
    };
    let mut cnt = 0i32;
    for &t_val in &tot {
        if phase_b {
            if t_val >= target {
                cnt += 1;
            }
        } else if t_val >= (dev.i[0] as f32) {
            cnt += 1;
        }
    }
    let j_idx = cnt - 1;
    let n = hi - lo + 1;
    if j_idx < 0 {
        s.lo = lo;
        s.hi = lo;
        s.done = true;
    } else if n <= NTH as i32 {
        let idx = j_idx as usize;
        s.lo = tk[idx];
        s.hi = tk[idx];
        s.done = true;
    } else {
        let idx = j_idx as usize;
        s.lo = tk[idx];
        s.hi = if idx < NTH - 1 { tk[idx + 1] - 1 } else { hi };
        s.done = s.lo == s.hi;
    }
    s.target = if phase_b { target } else { 0.0 };
    s
}

/// Samples one vocab row of BF16 logits (`role = 0` target/verify, `role = 1` draft).
pub fn sample_row(
    logits_bf16: &[u16],
    dev: &DevSampleParams,
    role: i32,
    seq_pos: u32,
    searched: bool,
) -> SampleResult {
    assert!(!logits_bf16.is_empty());
    let coupled = (dev.i[3] & F_COUPLED) != 0;
    let dtrunc = (dev.i[3] & F_DTRUNC) != 0;
    let greedy = (dev.i[3] & F_GREEDY) != 0;
    let noise = !greedy && (role == 0 || coupled);
    let trunc_active = !greedy && (role == 0 || (coupled && dtrunc));

    let nb = logits_bf16.len().div_ceil(BLK);
    let mut bm = vec![NEG; nb];
    let mut m = NEG;
    for (b, slot) in bm.iter_mut().enumerate() {
        let mx = block_max(logits_bf16, b);
        *slot = mx;
        m = m.max(mx);
    }
    let mut st = SearchState {
        max_logit: m,
        ..SearchState::default()
    };
    if searched {
        for round in 0..(2 * NIT) {
            st = search_round(logits_bf16, round, st, dev, trunc_active, &bm);
        }
    }
    let inv_t = dev.f[0];
    let tau = if !noise {
        key_of(m)
    } else {
        let mut t = if trunc_active && searched {
            st.lo
        } else {
            key_ceil(m - dev.f[1])
        };
        if trunc_active && (dev.i[3] & F_MINP) != 0 {
            let km = key_ceil(m + dev.f[3]);
            if km > t {
                t = km;
            }
        }
        t
    };
    let thr = val_of_key(tau);
    let rk = row_key(dev.i[1] as u32, dev.i[2] as u32, seq_pos);
    let mut best = NEG;
    let mut bi = 0usize;
    let mut best2 = NEG;
    let mut bi2 = None;

    for (b, &bmax) in bm.iter().enumerate() {
        if bmax < thr {
            continue;
        }
        let start = b * BLK;
        let end = logits_bf16.len().min(start + BLK);
        for (offset, &bits) in logits_bf16[start..end].iter().enumerate() {
            let v = start + offset;
            let l = bf16_to_f32(bits);
            if l.is_nan() || l < thr {
                continue;
            }
            let x = (l - m) * inv_t;
            let z = if noise { x + gumbel(rk, v as u32) } else { x };
            if z > best {
                if best > NEG {
                    best2 = best;
                    bi2 = Some(bi);
                }
                best = z;
                bi = v;
            } else if z > best2 {
                best2 = z;
                bi2 = Some(v);
            }
        }
    }

    SampleResult {
        token: bi,
        tau_key: tau,
        thr,
        state: st,
        z_best: best,
        second: bi2,
        z_second: best2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::f32_to_bf16_rne;

    #[test]
    fn bf16_order_keys_strictly_monotone() {
        let vals = [-100.0f32, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0];
        let keys: Vec<i32> = vals
            .iter()
            .map(|&v| key_of(bf16_to_f32(f32_to_bf16_rne(v))))
            .collect();
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "order violated: {} >= {}", w[0], w[1]);
        }
        for (&v, &k) in vals.iter().zip(&keys) {
            let bv = bf16_to_f32(f32_to_bf16_rne(v));
            assert_eq!(val_of_key(k), bv);
        }
    }

    #[test]
    fn greedy_keeps_smallest_id_on_exact_ties() {
        let mut logits = vec![f32_to_bf16_rne(-5.0); 256];
        logits[42] = f32_to_bf16_rne(3.5);
        logits[17] = f32_to_bf16_rne(3.5); // tie with smaller index
        logits[200] = f32_to_bf16_rne(3.5);
        let dev = SampleParams::default().to_dev(256);
        let res = sample_row(&logits, &dev, 0, 10, true);
        assert_eq!(res.token, 17);
    }

    #[test]
    fn position_keyed_gumbel_sampling_is_deterministic_and_respects_topk() {
        let mut logits = vec![f32_to_bf16_rne(-10.0); 256];
        logits[10] = f32_to_bf16_rne(5.0);
        logits[20] = f32_to_bf16_rne(4.75);
        logits[30] = f32_to_bf16_rne(4.5);
        logits[40] = f32_to_bf16_rne(0.0);

        let params = SampleParams {
            temperature: 0.8,
            top_k: 2,
            top_p: 0.95,
            min_p: 0.0,
            seed: 12345,
            coupled: true,
            draft_trunc: true,
        };
        let dev = params.to_dev(256);
        // Same position -> identical token across repeated verifications (speculative exactness!)
        let r1 = sample_row(&logits, &dev, 0, 7, true);
        let r2 = sample_row(&logits, &dev, 0, 7, true);
        assert_eq!(r1.token, r2.token);
        assert!(
            r1.token == 10 || r1.token == 20,
            "top_k=2 violated: got {}",
            r1.token
        );

        // Verify polynomial accuracy for mlog / mexp
        assert!((mlog(0.5) - 0.5f32.ln()).abs() < 1e-5);
        assert!((mexp(-2.5) - (-2.5f32).exp()).abs() < 1e-5);
    }
}
