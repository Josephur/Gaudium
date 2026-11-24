//! Host decoders of the ggml tensor types, to `f32` and to BF16.
//!
//! Each decoder follows the reference dequantisation of its type value for value, in `f32`, in
//! the same operation order (Rust does not contract `a * b + c` into a fused multiply-add), so
//! the results are bit-identical to the reference vectors in `tests/data/gguf/`. BF16 output is
//! the `f32` value rounded to nearest even ([`crate::fp8::f32_to_bf16_rne`]); a BF16 tensor is
//! copied bit for bit.
//!
//! Block layouts (little-endian; `d`, `dmin`: binary16 unless noted):
//! - `Q4_0`  d | 16 B nibbles: `(q - 8) * d`, low nibbles are weights 0..16, high 16..32
//! - `Q4_1`  d | m | 16 B: `q * d + m`
//! - `Q5_0`  d | u32 high bits | 16 B: `(q5 - 16) * d`
//! - `Q5_1`  d | m | u32 high bits | 16 B: `q5 * d + m`
//! - `Q8_0`  d | 32 x i8: `q * d`;  `Q8_1` d | s | 32 x i8: `q * d`
//! - `Q2_K`  16 B scales | 64 B 2-bit | d | dmin
//! - `Q3_K`  32 B high mask | 64 B 2-bit | 12 B 6-bit scales | d
//! - `Q4_K`  d | dmin | 12 B 6-bit (scale, min) pairs | 128 B nibbles
//! - `Q5_K`  d | dmin | 12 B pairs | 32 B high bits | 128 B nibbles
//! - `Q6_K`  128 B low nibbles | 64 B high 2-bit | 16 x i8 scales | d
//! - `Q8_K`  d (f32) | 256 x i8 | 16 x i16 sums (unused): `q * d`
//! - `IQ4_NL` d | 16 B: `d * K[q]`;  `IQ4_XS` d | u16 high scale bits | 4 B low | 128 B
//! - `MXFP4` e (E8M0) | 16 B: `F[q] * 2^(e - 128)` (F = doubled E2M1 values, hence the half)
//! - `NVFP4` 4 x UE4M3 | 32 B: per 16 weights, `F[q] * ue4m3(d_s) / 2`; low nibbles of the 8
//!   bytes of a sub-block are its weights 0..8, high nibbles 8..16
//! - `F8_E4M3` one OCP E4M3 byte per weight (bias 7; 0x7F / 0xFF are NaN)

use std::thread;

use super::{err, GgmlType, Gguf, GgufError, TensorInfo};
use crate::fp8::f32_to_bf16_rne;
use crate::verified as v;

/// IEEE binary16 -> binary32, exact.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 31) as u32;
    let m = (h & 1023) as u32;
    let u = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal: normalise
            let mut sh = 0u32;
            let mut mm = m;
            while mm & 1024 == 0 {
                mm <<= 1;
                sh += 1;
            }
            s | ((113 - sh) << 23) | ((mm & 1023) << 13)
        }
    } else if e == 31 {
        s | 0x7F80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(u)
}

/// Doubled E2M1 values (sign in bit 3), shared by MXFP4 and NVFP4.
const FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
/// The 16 non-uniform 4-bit levels of IQ4_NL / IQ4_XS.
const IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// E8M0 exponent byte -> `2^(e - 127) / 2` (the MXFP4 block scale, halved for doubled values).
#[inline]
pub fn e8m0_half(e: u8) -> f32 {
    let bits = if e < 2 {
        0x0020_0000u32 << e
    } else {
        ((e - 1) as u32) << 23
    };
    f32::from_bits(bits)
}

/// UE4M3 scale byte of NVFP4 -> value / 2 (halved for doubled values). The sign bit is
/// ignored; 0x00 and 0x7F give 0.
#[inline]
pub fn ue4m3_half(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 7) as f32;
    let raw = if exp == 0 {
        man * (1.0 / 512.0)
    } else {
        (1.0 + man / 8.0) * f32::from_bits(((exp - 7 + 127) as u32) << 23)
    };
    raw * 0.5
}

/// OCP E4M3 (bias 7, no infinities, 0x7F / 0xFF NaN) -> f32, exact.
#[inline]
pub fn e4m3fn_to_f32(x: u8) -> f32 {
    let ax = x & 0x7F;
    if ax == 0x7F {
        return f32::NAN;
    }
    let exp = ((ax >> 3) & 0xF) as u32;
    let man = (ax & 7) as u32;
    let val = if exp == 0 {
        man as f32 * (1.0 / 512.0)
    } else {
        f32::from_bits(((exp + 120) << 23) | (man << 20))
    };
    f32::from_bits(val.to_bits() | (((x & 0x80) as u32) << 24))
}

#[inline]
fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
#[inline]
fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
#[inline]
fn h(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16le(b, at))
}

/// The 6-bit (scale, min) pair `j` of a K-quant super-block's 12 packed bytes.
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let qs = &b[2..18];
    for j in 0..16 {
        y[j] = ((qs[j] & 0xF) as i32 - 8) as f32 * d;
        y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
    }
}

fn q4_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (h(b, 0), h(b, 2));
    let qs = &b[4..20];
    for j in 0..16 {
        y[j] = (qs[j] & 0xF) as f32 * d + m;
        y[j + 16] = (qs[j] >> 4) as f32 * d + m;
    }
}

fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let qh = u32le(b, 2);
    let qs = &b[6..22];
    for j in 0..16 {
        let h0 = (((qh >> j) << 4) & 0x10) as u8;
        let h1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = (((qs[j] & 0xF) | h0) as i32 - 16) as f32 * d;
        y[j + 16] = (((qs[j] >> 4) | h1) as i32 - 16) as f32 * d;
    }
}

fn q5_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (h(b, 0), h(b, 2));
    let qh = u32le(b, 4);
    let qs = &b[8..24];
    for j in 0..16 {
        let h0 = (((qh >> j) << 4) & 0x10) as u8;
        let h1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = ((qs[j] & 0xF) | h0) as f32 * d + m;
        y[j + 16] = ((qs[j] >> 4) | h1) as f32 * d + m;
    }
}

fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..32 {
        y[j] = b[2 + j] as i8 as f32 * d;
    }
}

fn q8_1(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..32 {
        y[j] = b[4 + j] as i8 as f32 * d;
    }
}

fn q2_k(b: &[u8], y: &mut [f32]) {
    let scales = &b[0..16];
    let d = h(b, 80);
    let dmin = h(b, 82);
    let mut o = 0;
    let mut is = 0;
    let mut q = 16;
    for _ in 0..2 {
        let mut shift = 0;
        for _ in 0..4 {
            for half in 0..2 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    y[o] = dl * (((b[q + l + 16 * half] >> shift) & 3) as i8) as f32 - ml;
                    o += 1;
                }
            }
            shift += 2;
        }
        q += 32;
    }
}

fn q3_k(b: &[u8], y: &mut [f32]) {
    let hm = &b[0..32];
    let d_all = h(b, 108);
    const K1: u32 = 0x0303_0303;
    const K2: u32 = 0x0f0f_0f0f;
    let mut aux = [u32le(b, 96), u32le(b, 100), u32le(b, 104), 0u32];
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & K2) | (((tmp >> 4) & K1) << 4);
    aux[3] = ((aux[1] >> 4) & K2) | (((tmp >> 6) & K1) << 4);
    aux[0] = (aux[0] & K2) | ((tmp & K1) << 4);
    aux[1] = (aux[1] & K2) | (((tmp >> 2) & K1) << 4);
    let mut scales = [0i8; 16];
    for (i, a) in aux.iter().enumerate() {
        for (k, byte) in a.to_le_bytes().iter().enumerate() {
            scales[4 * i + k] = *byte as i8;
        }
    }
    let mut o = 0;
    let mut is = 0;
    let mut m: u8 = 1;
    let mut q = 32;
    for _ in 0..2 {
        let mut shift = 0;
        for _ in 0..4 {
            for half in 0..2 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let k = l + 16 * half;
                    let lo = ((b[q + k] >> shift) & 3) as i32;
                    let hi = if hm[k] & m != 0 { 0 } else { 4 };
                    y[o] = dl * (lo - hi) as f32;
                    o += 1;
                }
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

fn q4_k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let scales = &b[4..16];
    let mut o = 0;
    let mut q = 16;
    let mut is = 0;
    for _ in 0..4 {
        let (sc, mn) = scale_min_k4(is, scales);
        let (d1, m1) = (d * sc as f32, dmin * mn as f32);
        let (sc, mn) = scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * sc as f32, dmin * mn as f32);
        for l in 0..32 {
            y[o + l] = d1 * (b[q + l] & 0xF) as f32 - m1;
        }
        for l in 0..32 {
            y[o + 32 + l] = d2 * (b[q + l] >> 4) as f32 - m2;
        }
        o += 64;
        q += 32;
        is += 2;
    }
}

fn q5_k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let scales = &b[4..16];
    let qh = &b[16..48];
    let mut ql = 48;
    let mut o = 0;
    let mut is = 0;
    let (mut u1, mut u2) = (1u8, 2u8);
    for _ in 0..4 {
        let (sc, mn) = scale_min_k4(is, scales);
        let (d1, m1) = (d * sc as f32, dmin * mn as f32);
        let (sc, mn) = scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * sc as f32, dmin * mn as f32);
        for l in 0..32 {
            let hb = if qh[l] & u1 != 0 { 16 } else { 0 };
            y[o + l] = d1 * ((b[ql + l] & 0xF) + hb) as f32 - m1;
        }
        for l in 0..32 {
            let hb = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[o + 32 + l] = d2 * ((b[ql + l] >> 4) + hb) as f32 - m2;
        }
        ql += 32;
        o += 64;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn q6_k(b: &[u8], y: &mut [f32]) {
    let d = h(b, 208);
    let (mut ql, mut qh, mut sc, mut o) = (0usize, 128usize, 192usize, 0usize);
    for _ in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((b[ql + l] & 0xF) | ((b[qh + l] & 3) << 4)) as i32 - 32;
            let q2 = ((b[ql + l + 32] & 0xF) | (((b[qh + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((b[ql + l] >> 4) | (((b[qh + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((b[ql + l + 32] >> 4) | (((b[qh + l] >> 6) & 3) << 4)) as i32 - 32;
            let s = |k: usize| b[sc + is + k] as i8 as f32;
            y[o + l] = d * s(0) * q1 as f32;
            y[o + l + 32] = d * s(2) * q2 as f32;
            y[o + l + 64] = d * s(4) * q3 as f32;
            y[o + l + 96] = d * s(6) * q4 as f32;
        }
        o += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

fn q8_k(b: &[u8], y: &mut [f32]) {
    let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    for j in 0..256 {
        y[j] = d * b[4 + j] as i8 as f32;
    }
}

fn iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..16 {
        y[j] = d * IQ4NL[(b[2 + j] & 0xF) as usize] as f32;
        y[j + 16] = d * IQ4NL[(b[2 + j] >> 4) as usize] as f32;
    }
}

fn iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let sh = u16le(b, 2);
    for ib in 0..8 {
        let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
            | ((((sh >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let qs = &b[8 + 16 * ib..8 + 16 * ib + 16];
        for j in 0..16 {
            y[32 * ib + j] = dl * IQ4NL[(qs[j] & 0xF) as usize] as f32;
            y[32 * ib + j + 16] = dl * IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
}

fn mxfp4(b: &[u8], y: &mut [f32]) {
    let d = e8m0_half(b[0]);
    for j in 0..16 {
        y[j] = FP4[(b[1 + j] & 0xF) as usize] as f32 * d;
        y[j + 16] = FP4[(b[1 + j] >> 4) as usize] as f32 * d;
    }
}

fn nvfp4(b: &[u8], y: &mut [f32]) {
    for s in 0..4 {
        let d = ue4m3_half(b[s]);
        let qs = &b[4 + 8 * s..4 + 8 * s + 8];
        for j in 0..8 {
            y[16 * s + j] = FP4[(qs[j] & 0xF) as usize] as f32 * d;
            y[16 * s + j + 8] = FP4[(qs[j] >> 4) as usize] as f32 * d;
        }
    }
}

/// Decodes whole blocks of type `ty`: `src` holds `n` blocks, `dst` their `n * weights` values.
/// Integer types give their values as `f32` (`I32` / `I64` round to nearest); `F64` rounds.
pub fn decode_blocks(ty: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), String> {
    let ti = ty.info();
    if !ti.decodable {
        return Err(format!("no host decoder for ggml type {} yet", ti.name));
    }
    let (bw, bb) = (ti.block_weights as usize, ti.block_bytes as usize);
    if !src.len().is_multiple_of(bb) || dst.len() != src.len() / bb * bw {
        return Err(format!(
            "{}: {} source bytes and {} values are not whole blocks",
            ti.name,
            src.len(),
            dst.len()
        ));
    }
    let blocks = src.chunks_exact(bb);
    match ty {
        GgmlType::F32 => blocks
            .zip(dst)
            .for_each(|(b, y)| *y = f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        GgmlType::F16 => blocks.zip(dst).for_each(|(b, y)| *y = h(b, 0)),
        GgmlType::BF16 => blocks
            .zip(dst)
            .for_each(|(b, y)| *y = f32::from_bits((u16le(b, 0) as u32) << 16)),
        GgmlType::F64 => blocks
            .zip(dst)
            .for_each(|(b, y)| *y = f64::from_le_bytes(b[..8].try_into().unwrap()) as f32),
        GgmlType::I8 => blocks.zip(dst).for_each(|(b, y)| *y = b[0] as i8 as f32),
        GgmlType::I16 => blocks
            .zip(dst)
            .for_each(|(b, y)| *y = i16::from_le_bytes([b[0], b[1]]) as f32),
        GgmlType::I32 => blocks
            .zip(dst)
            .for_each(|(b, y)| *y = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32),
        GgmlType::I64 => blocks
            .zip(dst)
            .for_each(|(b, y)| *y = i64::from_le_bytes(b[..8].try_into().unwrap()) as f32),
        GgmlType::F8_E4M3 => blocks.zip(dst).for_each(|(b, y)| *y = e4m3fn_to_f32(b[0])),
        _ => {
            let f: fn(&[u8], &mut [f32]) = match ty {
                GgmlType::Q4_0 => q4_0,
                GgmlType::Q4_1 => q4_1,
                GgmlType::Q5_0 => q5_0,
                GgmlType::Q5_1 => q5_1,
                GgmlType::Q8_0 => q8_0,
                GgmlType::Q8_1 => q8_1,
                GgmlType::Q2_K => q2_k,
                GgmlType::Q3_K => q3_k,
                GgmlType::Q4_K => q4_k,
                GgmlType::Q5_K => q5_k,
                GgmlType::Q6_K => q6_k,
                GgmlType::Q8_K => q8_k,
                GgmlType::IQ4_NL => iq4_nl,
                GgmlType::IQ4_XS => iq4_xs,
                GgmlType::MXFP4 => mxfp4,
                GgmlType::NVFP4 => nvfp4,
                _ => return Err(format!("no host decoder for ggml type {} yet", ti.name)),
            };
            for (b, y) in blocks.zip(dst.chunks_exact_mut(bw)) {
                f(b, y);
            }
        }
    }
    Ok(())
}

/// The stored bytes of rows `[r0, r1)` of `t` and the number of values they hold.
fn row_bytes<'g>(
    g: &'g Gguf,
    t: &TensorInfo,
    r0: u64,
    r1: u64,
) -> Result<(&'g [u8], u64), GgufError> {
    let rb = t.row_bytes();
    if !v::gg_rows_ok(r0, r1, rb, t.nbytes) {
        return Err(err(
            &g.files[t.shard].path.display().to_string(),
            format!(
                "tensor {}: rows [{r0}, {r1}) outside its {} rows",
                t.name,
                t.rows()
            ),
        ));
    }
    let data = g.tensor_data(t);
    let (b0, b1) = (
        v::gg_elem_offset(r0, rb) as usize,
        v::gg_elem_offset(r1, rb) as usize,
    );
    Ok((&data[b0..b1], v::gg_elem_offset(r1 - r0, t.row_len())))
}

fn check_out(g: &Gguf, t: &TensorInfo, have: usize, need: u64) -> Result<(), GgufError> {
    if have as u64 != need {
        return Err(err(
            &g.files[t.shard].path.display().to_string(),
            format!(
                "tensor {}: output holds {have} values, the rows hold {need}",
                t.name
            ),
        ));
    }
    Ok(())
}

/// Rows `[r0, r1)` of `t` as `f32`: `out[(r - r0) * row_len + c]`. A row is `dims[0]` values.
pub fn rows_f32(
    g: &Gguf,
    t: &TensorInfo,
    r0: u64,
    r1: u64,
    out: &mut [f32],
) -> Result<(), GgufError> {
    let (src, need) = row_bytes(g, t, r0, r1)?;
    check_out(g, t, out.len(), need)?;
    decode_blocks(t.ty, src, out).map_err(|m| {
        err(
            &g.files[t.shard].path.display().to_string(),
            format!("tensor {}: {m}", t.name),
        )
    })
}

/// Rows `[r0, r1)` of `t` as BF16 (round to nearest even from the `f32` value; a BF16 tensor's
/// stored bits unchanged), each value first multiplied by `scale` in `f32` when one is given.
pub fn rows_bf16(
    g: &Gguf,
    t: &TensorInfo,
    r0: u64,
    r1: u64,
    scale: Option<f32>,
    out: &mut [u16],
) -> Result<(), GgufError> {
    let (src, need) = row_bytes(g, t, r0, r1)?;
    check_out(g, t, out.len(), need)?;
    if t.ty == GgmlType::BF16 && scale.is_none() {
        for (o, b) in out.iter_mut().zip(src.chunks_exact(2)) {
            *o = u16le(b, 0);
        }
        return Ok(());
    }
    let rl = t.row_len() as usize;
    let rb = t.row_bytes() as usize;
    let mut f = vec![0f32; rl];
    for (o, b) in out.chunks_exact_mut(rl).zip(src.chunks_exact(rb)) {
        decode_blocks(t.ty, b, &mut f).map_err(|m| {
            err(
                &g.files[t.shard].path.display().to_string(),
                format!("tensor {}: {m}", t.name),
            )
        })?;
        match scale {
            Some(s) => o
                .iter_mut()
                .zip(&f)
                .for_each(|(o, &x)| *o = f32_to_bf16_rne(x * s)),
            None => o
                .iter_mut()
                .zip(&f)
                .for_each(|(o, &x)| *o = f32_to_bf16_rne(x)),
        }
    }
    Ok(())
}

/// [`rows_bf16`] split over up to `threads` threads (std threads, one contiguous run of rows
/// each). The result is identical to the single-threaded decode.
pub fn rows_bf16_par(
    g: &Gguf,
    t: &TensorInfo,
    r0: u64,
    r1: u64,
    scale: Option<f32>,
    threads: usize,
    out: &mut [u16],
) -> Result<(), GgufError> {
    let (_, need) = row_bytes(g, t, r0, r1)?;
    check_out(g, t, out.len(), need)?;
    let rows = r1 - r0;
    let chunk = v::gg_part_chunk(rows, threads.max(1) as u64).max(1);
    if threads <= 1 || rows <= 1 {
        return rows_bf16(g, t, r0, r1, scale, out);
    }
    let per = v::gg_elem_offset(chunk, t.row_len()) as usize;
    thread::scope(|s| {
        let jobs: Vec<_> = out
            .chunks_mut(per)
            .enumerate()
            .map(|(i, o)| {
                let lo = v::gg_part_lo(i as u64, rows, chunk);
                let n = o.len() as u64 / t.row_len();
                s.spawn(move || rows_bf16(g, t, r0 + lo, r0 + lo + n, scale, o))
            })
            .collect();
        jobs.into_iter()
            .map(|j| {
                j.join()
                    .unwrap_or_else(|_| Err(err("", "decode thread panicked")))
            })
            .collect::<Result<(), _>>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn same(a: f32, b: f32) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    /// Every reference vector in `tests/data/gguf/*.vec` (written by `make_fixtures.py` from the
    /// reference's numpy dequantiser) decodes bit for bit.
    #[test]
    fn decoders_match_the_reference_vectors() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/gguf");
        let mut seen = Vec::new();
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_none_or(|x| x != "vec") {
                continue;
            }
            let b = std::fs::read(&p).unwrap();
            assert_eq!(&b[..4], b"GQV1");
            let id = u32::from_le_bytes(b[4..8].try_into().unwrap());
            let n = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
            let ty = GgmlType::from_id(id).unwrap();
            let ti = ty.info();
            let (bw, bb) = (ti.block_weights as usize, ti.block_bytes as usize);
            let raw = &b[12..12 + n * bb];
            let want: Vec<f32> = b[12 + n * bb..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(want.len(), n * bw, "{}", ti.name);
            let mut got = vec![0f32; n * bw];
            decode_blocks(ty, raw, &mut got).unwrap();
            let bad = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| !same(**g, **w))
                .count();
            let first = got.iter().zip(&want).position(|(g, w)| !same(*g, *w));
            assert_eq!(
                bad,
                0,
                "{}: {bad} of {} values differ, first at {first:?}",
                ti.name,
                want.len()
            );
            seen.push(ti.name);
        }
        seen.sort();
        let mut want = vec![
            "BF16", "F16", "F8_E4M3", "IQ4_NL", "IQ4_XS", "MXFP4", "NVFP4", "Q2_K", "Q3_K", "Q4_0",
            "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0",
        ];
        want.sort();
        assert_eq!(seen, want);
    }

    /// Types the reference has no numpy dequantiser for: vectors derived by hand from the layout.
    #[test]
    fn hand_derived_vectors() {
        // Q8_1: d = 0.5, s (unused) = 7.0, q = -16..16
        let mut b = vec![0x00, 0x38, 0x00, 0x47];
        b.extend((0..32).map(|i| (i as i8 - 16) as u8));
        let mut y = vec![0f32; 32];
        decode_blocks(GgmlType::Q8_1, &b, &mut y).unwrap();
        assert_eq!(
            y,
            (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect::<Vec<_>>()
        );
        // Q8_K: d = 0.25 (f32), q = -128..128, block sums ignored
        let mut b = 0.25f32.to_le_bytes().to_vec();
        b.extend((0..256).map(|i| (i as i32 - 128) as i8 as u8));
        b.extend([0xAB; 32]);
        let mut y = vec![0f32; 256];
        decode_blocks(GgmlType::Q8_K, &b, &mut y).unwrap();
        assert_eq!(
            y,
            (0..256)
                .map(|i| (i as f32 - 128.0) * 0.25)
                .collect::<Vec<_>>()
        );
        // integers and F64
        let mut y = vec![0f32; 2];
        decode_blocks(GgmlType::I8, &[0xFF, 0x05], &mut y).unwrap();
        assert_eq!(y, vec![-1.0, 5.0]);
        decode_blocks(GgmlType::I16, &[0x00, 0x80, 0x10, 0x00], &mut y).unwrap();
        assert_eq!(y, vec![-32768.0, 16.0]);
        let b: Vec<u8> = [-7i32, 1 << 24]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        decode_blocks(GgmlType::I32, &b, &mut y).unwrap();
        assert_eq!(y, vec![-7.0, 16777216.0]);
        let b: Vec<u8> = [-3i64, 9].iter().flat_map(|x| x.to_le_bytes()).collect();
        decode_blocks(GgmlType::I64, &b, &mut y).unwrap();
        assert_eq!(y, vec![-3.0, 9.0]);
        let b: Vec<u8> = [0.1f64, -2.5]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        decode_blocks(GgmlType::F64, &b, &mut y).unwrap();
        assert_eq!(y, vec![0.1f64 as f32, -2.5]);
    }

    #[test]
    fn scale_formats() {
        // UE4M3 (halved): 0x38 = 1.0 -> 0.5; subnormal 0x01 = 2^-9 -> 2^-10; sign ignored; 0x7F -> 0
        assert_eq!(ue4m3_half(0x38), 0.5);
        assert_eq!(ue4m3_half(0x01), 2f32.powi(-10));
        assert_eq!(ue4m3_half(0xB8), 0.5);
        assert_eq!(ue4m3_half(0x7F), 0.0);
        assert_eq!(ue4m3_half(0x7E), 448.0 * 0.5);
        // E8M0 (halved): 127 -> 0.5, 0 -> 2^-128, 1 -> 2^-127
        assert_eq!(e8m0_half(127), 0.5);
        assert_eq!(e8m0_half(0), f32::from_bits(0x0020_0000)); // 2^-128
        assert_eq!(e8m0_half(1), f32::from_bits(0x0040_0000)); // 2^-127
                                                               // E4M3FN: 0x7E = 448, 0x01 = 2^-9, 0x80 = -0, NaN at 0x7F / 0xFF
        assert_eq!(e4m3fn_to_f32(0x7E), 448.0);
        assert_eq!(e4m3fn_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(e4m3fn_to_f32(0xC0), -2.0);
        assert!(e4m3fn_to_f32(0x80).is_sign_negative());
        assert!(e4m3fn_to_f32(0x7F).is_nan() && e4m3fn_to_f32(0xFF).is_nan());
        // binary16: every value round-trips against the standard conversion
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            let e = (h >> 10) & 31;
            if e == 31 && h & 1023 != 0 {
                assert!(f.is_nan());
            } else if e == 0 {
                assert_eq!(
                    f,
                    ((h & 1023) as f32) * 2f32.powi(-24) * if h & 0x8000 != 0 { -1.0 } else { 1.0 }
                );
            }
        }
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn grid_types_are_refused_by_name() {
        let ty = GgmlType::from_id(16).unwrap(); // IQ2_XXS
        let e = decode_blocks(ty, &[0u8; 66], &mut [0f32; 256]).unwrap_err();
        assert!(e.contains("IQ2_XXS"), "{e}");
        assert!(decode_blocks(GgmlType::Q8_0, &[0u8; 33], &mut [0f32; 32]).is_err());
    }
}
