//! Gaudi 2 FP8 (`fp8_143` = E4M3, bias 7, max finite 240.0, exp 15 = NaN/Inf) & BF16 numerics.
//!
//! Mirrors `fp8_quant.hpp`: per-output-channel symmetric power-of-two weight quantization,
//! static activation scale tables, and atomic disk cache persistence.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::thread;

/// Gaudi 2 `fp8_143` maximum finite representable magnitude (`S.1110.111` = 240.0).
pub const FP8_MAX: f32 = 240.0;

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round-to-nearest-even `f32 -> bf16`.
#[inline]
pub fn f32_to_bf16_rne(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((u >> 16) | 0x0040) as u16; // preserve quiet NaN
    }
    let rounded = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (rounded >> 16) as u16
}

/// Round-to-nearest-even `f32 -> Gaudi 2 fp8_143` (E4M3, bias 7, max finite 240.0 = `0x77`).
/// Magnitudes >= 248.0 saturate to `0x77 | sign`; NaN encodes as `0x7F | sign`.
#[inline]
pub fn f32_to_e4m3(f: f32) -> u8 {
    let sign: u8 = if f.is_sign_negative() { 0x80 } else { 0x00 };
    let a = f.abs();
    if a.is_nan() {
        return 0x7F | sign;
    }
    if a >= 248.0 {
        return 0x77 | sign;
    }
    let min_normal = f32::from_bits((127 - 6) << 23); // 2^-6
    if a < min_normal {
        let m = (a * 512.0).round_ties_even() as i32;
        if m >= 8 {
            return sign | (1 << 3);
        }
        return sign | (m as u8);
    }
    let bits = a.to_bits();
    let mut e = ((bits >> 23) as i32) - 127;
    let frac_bits = bits & 0x007F_FFFF;
    let frac = (frac_bits as f32) / 8_388_608.0; // in [0, 1)
    let mut mant = (frac * 8.0).round_ties_even() as i32;
    if mant == 8 {
        mant = 0;
        e += 1;
    }
    let be = e + 7;
    if be > 15 || (be == 15 && mant > 6) {
        return 0x77 | sign;
    }
    sign | ((be as u8) << 3) | (mant as u8)
}

/// Exact decode of Gaudi 2 `fp8_143` to `f32`.
#[inline]
pub fn e4m3_to_f32(v: u8) -> f32 {
    let s = if (v & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let be = ((v >> 3) & 0x0F) as i32;
    let mant = (v & 0x07) as f32;
    if be == 0 {
        return s * mant * (1.0f32 / 512.0f32); // mant * 2^-9
    }
    if be == 15 && (v & 0x07) == 7 {
        return f32::NAN;
    }
    let exp_scale = f32::from_bits(((be - 7 + 127) as u32) << 23);
    s * (1.0 + mant / 8.0) * exp_scale
}

#[inline]
pub fn pow2_ceil(s: f32) -> f32 {
    2.0f32.powi(s.log2().ceil() as i32)
}

#[inline]
pub fn pow2_floor(s: f32) -> f32 {
    2.0f32.powi(s.log2().floor() as i32)
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RowStats {
    pub amax_min: f32,
    pub amax_max: f32,
    pub log2_mean: f64,
    pub n_subnormal: usize,
}

/// Per-output-channel symmetric power-of-two quantization of a BF16 `[n, k]` weight matrix.
/// Returns `(w8 [n*k], scale [n], stats)` where `scale[row]` is the power-of-two dequant multiplier.
pub fn quantize_rows_bf16(
    weights_bf16: &[u16],
    n: usize,
    k: usize,
    nthreads: usize,
) -> (Vec<u8>, Vec<f32>, RowStats) {
    assert_eq!(weights_bf16.len(), n * k);
    let mut w8 = vec![0u8; n * k];
    let mut scale = vec![1.0f32; n];
    if n == 0 || k == 0 {
        return (w8, scale, RowStats::default());
    }
    let threads = nthreads.clamp(1, n);
    let mut amaxv = vec![0.0f32; n];
    let mut sub_counts = vec![0usize; threads];

    let rows_per = n.div_ceil(threads);
    thread::scope(|s| {
        let mut w8_rem = w8.as_mut_slice();
        let mut sc_rem = scale.as_mut_slice();
        let mut am_rem = amaxv.as_mut_slice();
        let mut row_start = 0usize;
        for sub_out in &mut sub_counts {
            if row_start >= n {
                break;
            }
            let chunk_rows = rows_per.min(n - row_start);
            let (w8_chunk, w8_rest) = w8_rem.split_at_mut(chunk_rows * k);
            let (sc_chunk, sc_rest) = sc_rem.split_at_mut(chunk_rows);
            let (am_chunk, am_rest) = am_rem.split_at_mut(chunk_rows);
            w8_rem = w8_rest;
            sc_rem = sc_rest;
            am_rem = am_rest;
            let src_chunk = &weights_bf16[row_start * k..(row_start + chunk_rows) * k];
            row_start += chunk_rows;

            s.spawn(move || {
                let mut nsub = 0usize;
                for r in 0..chunk_rows {
                    let row = &src_chunk[r * k..(r + 1) * k];
                    let mut amax = 0.0f32;
                    for &b in row {
                        amax = amax.max(bf16_to_f32(b).abs());
                    }
                    let sc = if amax > 0.0 { pow2_ceil(amax / FP8_MAX) } else { 1.0 };
                    sc_chunk[r] = sc;
                    am_chunk[r] = amax;
                    let inv = 1.0 / sc;
                    let dst = &mut w8_chunk[r * k..(r + 1) * k];
                    for c in 0..k {
                        let q = f32_to_e4m3(bf16_to_f32(row[c]) * inv);
                        dst[c] = q;
                        if (q & 0x78) == 0 && (q & 0x07) != 0 {
                            nsub += 1;
                        }
                    }
                }
                *sub_out = nsub;
            });
        }
    });

    let mut amax_min = f32::INFINITY;
    let mut amax_max = 0.0f32;
    let mut log2_sum = 0.0f64;
    for &a in &amaxv {
        amax_min = amax_min.min(a);
        amax_max = amax_max.max(a);
        if a > 0.0 {
            log2_sum += (a as f64).log2();
        }
    }
    let stats = RowStats {
        amax_min,
        amax_max,
        log2_mean: log2_sum / (n as f64),
        n_subnormal: sub_counts.into_iter().sum(),
    };
    (w8, scale, stats)
}

/// Static per-site activation scale entry (`<site> <k> <amax> <rms>`, `cs = 2^k`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ActScale {
    pub k: i32,
    pub amax: f32,
    pub rms: f32,
}

impl ActScale {
    pub fn cs(self) -> f32 {
        2.0f32.powi(self.k)
    }

    pub fn sa(self) -> f32 {
        2.0f32.powi(-self.k)
    }

    pub fn compute_k(amax: f32, margin: f32) -> i32 {
        if !(amax > 0.0) {
            return 0;
        }
        (FP8_MAX / (amax * margin)).log2().floor() as i32
    }
}

pub fn load_act_scales(path: &Path) -> io::Result<BTreeMap<String, ActScale>> {
    let text = fs::read_to_string(path)?;
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(site), Some(k), Some(amax), Some(rms)) = (it.next(), it.next(), it.next(), it.next()) else {
            continue;
        };
        if let (Ok(k), Ok(amax), Ok(rms)) = (k.parse::<i32>(), amax.parse::<f32>(), rms.parse::<f32>()) {
            map.insert(site.to_owned(), ActScale { k, amax, rms });
        }
    }
    Ok(map)
}

pub fn store_act_scales(path: &Path, scales: &BTreeMap<String, ActScale>) -> io::Result<()> {
    let mut out = String::new();
    for (site, a) in scales {
        out.push_str(&format!("{site} {} {} {}\n", a.k, a.amax, a.rms));
    }
    write_atomic(path, out.as_bytes())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

pub fn cache_store_fp8(dir: &Path, name: &str, w8: &[u8], scale: &[f32]) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    write_atomic(&dir.join(format!("{name}.w8")), w8)?;
    let mut sbytes = Vec::with_capacity(scale.len() * 4);
    for &s in scale {
        sbytes.extend_from_slice(&s.to_le_bytes());
    }
    write_atomic(&dir.join(format!("{name}.scale")), &sbytes)
}

pub fn cache_load_fp8(dir: &Path, name: &str, n: usize, k: usize) -> io::Result<Option<(Vec<u8>, Vec<f32>)>> {
    let p_w8 = dir.join(format!("{name}.w8"));
    let p_sc = dir.join(format!("{name}.scale"));
    if !p_w8.exists() || !p_sc.exists() {
        return Ok(None);
    }
    let w8 = fs::read(&p_w8)?;
    let sc_bytes = fs::read(&p_sc)?;
    if w8.len() != n * k || sc_bytes.len() != n * 4 {
        return Ok(None);
    }
    let mut scale = Vec::with_capacity(n);
    for c in sc_bytes.chunks_exact(4) {
        scale.push(f32::from_le_bytes(c.try_into().unwrap()));
    }
    Ok(Some((w8, scale)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaudi2_e4m3_encoding_and_decoding_matches_hardware_spec() {
        // Max finite code is 0x77 => +240.0; 0xF7 => -240.0
        assert_eq!(f32_to_e4m3(240.0), 0x77);
        assert_eq!(e4m3_to_f32(0x77), 240.0);
        assert_eq!(f32_to_e4m3(-240.0), 0xF7);
        assert_eq!(e4m3_to_f32(0xF7), -240.0);

        // Saturation >= 248.0 -> 240.0 (0x77)
        assert_eq!(f32_to_e4m3(248.0), 0x77);
        assert_eq!(f32_to_e4m3(1000.0), 0x77);
        assert_eq!(f32_to_e4m3(-500.0), 0xF7);

        // Zero, subnormal, normal, NaN
        assert_eq!(f32_to_e4m3(0.0), 0x00);
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(f32_to_e4m3(1.0), 0x38); // bias 7 -> exp 7 << 3 = 0x38
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x01), 1.0 / 512.0); // smallest positive subnormal 2^-9
        assert_eq!(f32_to_e4m3(1.0 / 512.0), 0x01);
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert_eq!(f32_to_e4m3(f32::NAN) & 0x7F, 0x7F);
    }

    #[test]
    fn row_quantization_uses_exact_pow2_scales() {
        let vals = [0.5f32, -1.0, 2.0, -3.5, 0.0, 0.0, 0.0, 0.0];
        let bf16: Vec<u16> = vals.iter().map(|&v| f32_to_bf16_rne(v)).collect();
        let (w8, scales, stats) = quantize_rows_bf16(&bf16, 2, 4, 2);
        // Row 0 amax = 3.5 -> ceil_pow2(3.5 / 240) = 2^-6 = 1/64
        assert_eq!(scales[0], 2.0f32.powi(-6));
        // Row 1 all zero -> scale = 1.0
        assert_eq!(scales[1], 1.0);
        assert_eq!(stats.amax_max, 3.5);
        for c in 0..4 {
            let recon = e4m3_to_f32(w8[c]) * scales[0];
            assert!((recon - vals[c]).abs() <= 0.1, "col {c}: got {recon}, want {}", vals[c]);
        }
        assert!(w8[4..8].iter().all(|&b| b == 0));
    }
}
