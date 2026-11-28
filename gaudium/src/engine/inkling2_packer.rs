//! Inkling 2 weight packer: the exact HBM byte image of every device tensor the graphs of
//! `engine::inkling2` declare, from the publisher's checkpoint.
//!
//! - Projections are borrowed or concatenated unchanged, except the value projection, whose rows
//!   go in **lane order** inside each block of 128 so that the f32 partial outputs of the
//!   `P . V` product arrive in the kernels' split layout.
//! - Per-channel f32 tables (clip scales, convolution taps, mix queries, the router's selection
//!   bias) are packed in the split layout: lanes `[0, 64)` of a block of 128 hold its even
//!   channels, `[64, 128)` the odd ones.
//! - The relative-position projection becomes a block-diagonal matrix over eight heads whose
//!   rows are reversed and padded (`verified::ik_bias_src`), so a softmax block loads the
//!   biases of 64 consecutive keys with one vector load.
//! - Expert matrices stored as 4-bit codes with a scale per 16 values and a per-expert factor
//!   (NVFP4), or as 8-bit floats with a power-of-two scale per 32 values (MXFP8), are decoded to
//!   BF16 with the publisher's arithmetic: `(code * scale) * (amax / denominator)` resp.
//!   `value * 2^(scale - 127)` in f32, rounded to nearest even.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;

use super::IkDims;
use crate::core::gemma4::{lane_in_block, natural_channel};
use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};
use crate::safetensors::{SafetensorsStore, TensorMeta};
use crate::verified::{
    ik_bias_src, ik_f8_row_ok, ik_f8_row_scales, ik_nib_row_bytes, ik_nib_row_scales, ik_row_admit,
    ik_row_at, ik_stacked_bytes, ikr_dn_tiles, ikr_gu_tiles, ikr_scale_bytes, ikr_scale_lane_pos,
    ikr_scale_vec_off, ikr_src_row, ikr_w4_code_bytes, ikr_w4_col_off, ikr_w8_code_bytes,
    ikr_w8_col_off,
};

/// Values of the sixteen 4-bit codes (sign, two exponent bits, one mantissa bit). Code 8 is a
/// negative zero in the format; it is stored as the same weight, a positive zero, so that every
/// source of one model gives one byte image.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Value of an 8-bit float with 4 exponent and 3 mantissa bits (bias 7, no infinities).
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    if e == 0xF && (b & 7) == 7 {
        return f32::NAN;
    }
    let mag = if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    sign * mag
}

fn e4m3_table() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (b, v) in t.iter_mut().enumerate() {
        *v = e4m3_to_f32(b as u8);
    }
    t
}

/// Decodes one row of 4-bit codes (two per byte, the even element in the low nibble) with one
/// 8-bit float scale per 16 values and the expert's factor, to BF16.
pub fn decode_nib_row(codes: &[u8], scales: &[u8], factor: f32, f8: &[f32; 256], out: &mut [u8]) {
    for (bi, &sb) in scales.iter().enumerate() {
        let s = f8[sb as usize];
        for k in 0..8 {
            let byte = codes[bi * 8 + k];
            let lo = (E2M1[(byte & 15) as usize] * s) * factor;
            let hi = (E2M1[(byte >> 4) as usize] * s) * factor;
            let o = (bi * 16 + 2 * k) * 2;
            out[o..o + 2].copy_from_slice(&f32_to_bf16_rne(lo).to_le_bytes());
            out[o + 2..o + 4].copy_from_slice(&f32_to_bf16_rne(hi).to_le_bytes());
        }
    }
}

/// Decodes one row of 8-bit floats with one power-of-two scale `2^(scale - 127)` per 32 values,
/// to BF16.
pub fn decode_f8_row(vals: &[u8], scales: &[u8], f8: &[f32; 256], out: &mut [u8]) {
    for (bi, &sb) in scales.iter().enumerate() {
        let s = 2f32.powi(i32::from(sb) - 127);
        for k in 0..32 {
            let v = f8[vals[bi * 32 + k] as usize] * s;
            let o = (bi * 32 + k) * 2;
            out[o..o + 2].copy_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
        }
    }
}

/// Rows of a stacked expert tensor in the publisher's row order, decoded to BF16.
pub trait ExpertRows: Sync {
    /// Decodes row `r` of expert `e` into `out` (the row's values as little-endian BF16).
    fn decode(&self, e: usize, r: usize, out: &mut [u8]) -> Result<(), String>;
}

/// Quantized format for one Inkling 2 layer's routed expert pools on-card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IkExpertFmt {
    W4,
    W8,
}

/// Verified dimensions and exact byte extents of one layer's packed routed expert code/scale pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RoutedLayerExtents {
    pub blk: u64,
    pub nblk: u64,
    pub inner_cols: u64,
    pub code_dims: [u64; 4],
    pub scale_dims: [u64; 4],
    pub code_bytes: u64,
    pub scale_bytes: u64,
}

pub(super) fn routed_layer_extents(
    fmt: IkExpertFmt,
    in_cols: u64,
    tiles: u64,
    ne: u64,
) -> Result<RoutedLayerExtents, String> {
    if tiles == 0 || ne == 0 {
        return Err(format!(
            "routed expert extents require non-zero tiles and experts, got tiles={tiles} ne={ne}"
        ));
    }
    match fmt {
        IkExpertFmt::W4 => {
            let inner_cols = ik_nib_row_bytes(in_cols);
            if inner_cols == 0 {
                return Err(format!(
                    "routed W4 packing requires non-zero columns % 16 == 0, got {in_cols}"
                ));
            }
            let nblk = ik_nib_row_scales(in_cols);
            let code_bytes = ikr_w4_code_bytes(in_cols, tiles, ne)
                .ok_or_else(|| "routed W4 code pool byte size overflows u64".to_string())?;
            let scale_bytes = ikr_scale_bytes(in_cols, 16, tiles, ne)
                .ok_or_else(|| "routed W4 scale pool byte size overflows u64".to_string())?;
            Ok(RoutedLayerExtents {
                blk: 16,
                nblk,
                inner_cols,
                code_dims: [256, inner_cols, tiles, ne],
                scale_dims: [256, nblk, tiles, ne],
                code_bytes,
                scale_bytes,
            })
        }
        IkExpertFmt::W8 => {
            if !ik_f8_row_ok(in_cols) {
                return Err(format!(
                    "routed W8 packing requires non-zero columns % 32 == 0, got {in_cols}"
                ));
            }
            let nblk = ik_f8_row_scales(in_cols);
            let code_bytes = ikr_w8_code_bytes(in_cols, tiles, ne)
                .ok_or_else(|| "routed W8 code pool byte size overflows u64".to_string())?;
            let scale_bytes = ikr_scale_bytes(in_cols, 32, tiles, ne)
                .ok_or_else(|| "routed W8 scale pool byte size overflows u64".to_string())?;
            Ok(RoutedLayerExtents {
                blk: 32,
                nblk,
                inner_cols: in_cols,
                code_dims: [256, in_cols, tiles, ne],
                scale_dims: [256, nblk, tiles, ne],
                code_bytes,
                scale_bytes,
            })
        }
    }
}

/// Round-to-nearest-even IEEE 754 `f32 -> fp16` (`u16` bit pattern).
#[inline]
pub fn f32_to_f16_rne(f: f32) -> u16 {
    let u = f.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xFF) as i32;
    let mant = u & 0x007F_FFFF;
    if exp == 0xFF {
        if mant != 0 {
            return sign | 0x7E00 | ((mant >> 13) as u16).max(1);
        }
        return sign | 0x7C00;
    }
    if exp == 0 {
        return sign;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let sig = 0x0080_0000 | mant;
        let sh = (14 - e) as u32;
        let half_sig = sig >> sh;
        let rem = sig & ((1 << sh) - 1);
        let mid = 1 << (sh - 1);
        let round = u32::from(rem > mid || (rem == mid && (half_sig & 1) != 0));
        return sign | ((half_sig + round) as u16);
    }
    let mut m = mant >> 13;
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) != 0) {
        m += 1;
        if m >= 1024 {
            let ne = e + 1;
            if ne >= 31 {
                return sign | 0x7C00;
            }
            return sign | ((ne as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | (m as u16)
}

/// Exact IEEE 754 `fp16` (`u16`) -> `f32`.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x03FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let s = if sign != 0 { -1.0f32 } else { 1.0f32 };
        return s * (mant as f32) * 2f32.powi(-24);
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// 256-byte `int8(2 * E2M1)` LUT replicated across all 16-byte SIMD groups for `gaudium_ik_w4_*`.
pub fn ik_nvfp4_lut() -> Vec<u8> {
    const LV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let mut out = Vec::with_capacity(256);
    for _ in 0..16 {
        for &v in &LV {
            out.push(v as u8);
        }
    }
    out
}

#[inline]
fn routed_src_row(is_gu: bool, tt: u64, lane: u64) -> u64 {
    ikr_src_row(is_gu, tt, lane)
}

fn pack_w8_rows_with<S, M, F>(
    ne: u64,
    tiles: u64,
    in_cols: u64,
    is_gu: bool,
    panic_msg: &'static str,
    make_scratch: M,
    decode_f32: F,
) -> Result<(Vec<u8>, Vec<u8>), String>
where
    M: Fn() -> S + Sync,
    F: Fn(&mut S, usize, u64, &mut [f32]) -> Result<(), String> + Sync,
{
    let ext = routed_layer_extents(IkExpertFmt::W8, in_cols, tiles, ne)?;
    let exp = routed_layer_extents(IkExpertFmt::W8, in_cols, tiles, 1)?;
    let nblk = ext.nblk;
    let code_bytes = ext.code_bytes;
    let scale_bytes = ext.scale_bytes;
    let exp_code = exp.code_bytes as usize;
    let exp_scale = exp.scale_bytes as usize;

    let col_offs_per_tile: Vec<Vec<usize>> = (0..tiles)
        .map(|tt| {
            (0..in_cols)
                .map(|col| {
                    ikr_w8_col_off(0, tiles, in_cols, tt, col)
                        .map(|o| o as usize)
                        .ok_or_else(|| "routed W8 col offset overflow".to_string())
                })
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let vec_offs_per_tile: Vec<Vec<usize>> = (0..tiles)
        .map(|tt| {
            (0..nblk)
                .map(|b| {
                    ikr_scale_vec_off(0, tiles, nblk, tt, b)
                        .map(|o| o as usize * 2)
                        .ok_or_else(|| "routed W8 scale offset overflow".to_string())
                })
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;

    let mut codes = vec![0u8; code_bytes as usize];
    let mut scales = vec![0u8; scale_bytes as usize];
    let scale_pos: [usize; 256] = std::array::from_fn(|i| ikr_scale_lane_pos(i as u64) as usize);

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 32);
    let per = (ne as usize).div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for (chunk_idx, (c_chunk, s_chunk)) in codes
            .chunks_mut(per * exp_code)
            .zip(scales.chunks_mut(per * exp_scale))
            .enumerate()
        {
            let e0 = chunk_idx * per;
            let cnt = c_chunk.len() / exp_code;
            let make_scratch = &make_scratch;
            let decode_f32 = &decode_f32;
            let col_offs_ref = &col_offs_per_tile;
            let vec_offs_ref = &vec_offs_per_tile;
            let sp = &scale_pos;
            handles.push(sc.spawn(move || -> Result<(), String> {
                let mut scratch = make_scratch();
                let mut row_vals = vec![0f32; in_cols as usize];
                for le in 0..cnt {
                    let e = e0 + le;
                    let c_dst = &mut c_chunk[le * exp_code..(le + 1) * exp_code];
                    let s_dst = &mut s_chunk[le * exp_scale..(le + 1) * exp_scale];
                    for tt in 0..tiles as usize {
                        let col_offs = &col_offs_ref[tt];
                        let vec_offs = &vec_offs_ref[tt];
                        for i in 0..256usize {
                            let row = routed_src_row(is_gu, tt as u64, i as u64);
                            decode_f32(&mut scratch, e, row, &mut row_vals)?;
                            for b in 0..nblk as usize {
                                let base_c = b * 32;
                                let vals = &row_vals[base_c..base_c + 32];
                                let mut amax = 0.0f32;
                                for &v in vals {
                                    amax = amax.max(v.abs());
                                }
                                let s_h = if amax == 0.0 {
                                    0u16
                                } else {
                                    f32_to_f16_rne(amax / 127.0)
                                };
                                let sv = f16_to_f32(s_h);
                                let inv = if sv == 0.0 { 0.0 } else { 1.0 / sv };
                                let sb = vec_offs[b] + sp[i] * 2;
                                s_dst[sb..sb + 2].copy_from_slice(&s_h.to_le_bytes());
                                for c in 0..32 {
                                    let q = (vals[c] * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
                                    c_dst[col_offs[base_c + c] + i] = q as u8;
                                }
                            }
                        }
                    }
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().map_err(|_| panic_msg.to_string())??;
        }
        Ok::<(), String>(())
    })?;
    Ok((codes, scales))
}

fn pack_w8_from_rows(
    ex: &dyn ExpertRows,
    ne: u64,
    tiles: u64,
    in_cols: u64,
    is_gu: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    pack_w8_rows_with(
        ne,
        tiles,
        in_cols,
        is_gu,
        "routed W8 packer thread panicked",
        || vec![0u8; in_cols as usize * 2],
        |bf_buf, e, row, vals| {
            ex.decode(e, row as usize, bf_buf)?;
            for (dst, ch) in vals.iter_mut().zip(bf_buf.chunks_exact(2)) {
                *dst = bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]]));
            }
            Ok(())
        },
    )
}

/// Where the packer reads a checkpoint: its tensors under the publisher's names (the part after
/// `model.llm.`) and in the publisher's shapes and element order, whatever the file format.
pub trait IkSource: Sync {
    /// A BF16 tensor of exactly `shape`, as bytes.
    fn bf16(&self, suffix: &str, shape: &[u64]) -> Result<Cow<'_, [u8]>, String>;
    /// A small tensor of `elems` values as f32.
    fn f32s(&self, suffix: &str, elems: u64) -> Result<Vec<f32>, String>;
    /// The stacked expert tensor `layers.<l>.mlp.experts.<key>_weight` (`key`: `w13`, whose even
    /// rows are gate rows and odd rows up rows, or `w2`) of `ne` experts x `rows` x `cols`.
    fn experts(
        &self,
        l: usize,
        key: &str,
        ne: u64,
        rows: u64,
        cols: u64,
    ) -> Result<Box<dyn ExpertRows + '_>, String>;
    /// Preferred routed-kernel weight format for layer `l`.
    fn expert_fmt(&self, _l: usize) -> Result<IkExpertFmt, String> {
        Ok(IkExpertFmt::W8)
    }
    /// Packs `ne` routed experts of layer `l` into native W4 (`(u8, f16)`) or W8 (`(i8, f16)`) pools.
    fn pack_routed_experts(
        &self,
        l: usize,
        key: &str,
        ne: u64,
        out_rows: u64,
        in_cols: u64,
        is_gu: bool,
    ) -> Result<(Vec<u8>, Vec<u8>, IkExpertFmt), String> {
        let tiles = if is_gu {
            ikr_gu_tiles(out_rows / 2)
        } else {
            ikr_dn_tiles(out_rows)
        };
        let ex = self.experts(l, key, ne, out_rows, in_cols)?;
        let (codes, scales) = pack_w8_from_rows(ex.as_ref(), ne, tiles, in_cols, is_gu)?;
        Ok((codes, scales, IkExpertFmt::W8))
    }
}

/// A stacked expert tensor of the publisher's checkpoint in its stored form.
struct StExperts<'a> {
    w: &'a [u8],
    scale: &'a [u8],
    /// Per-expert factor of the 4-bit form; `None`: the 8-bit float form.
    factor: Option<Vec<f32>>,
    rows: usize,
    cols: usize,
    f8: [f32; 256],
}

/// Row `r` of expert `e` of a stacked tensor of `rows` rows of `rb` bytes per expert
/// (`verified::ik_row_at`); `None` when the row is not inside `bytes`.
fn stacked_row(bytes: &[u8], e: usize, rows: usize, r: usize, rb: usize) -> Option<&[u8]> {
    let at = ik_row_at(e as u64, rows as u64, r as u64, rb as u64)?;
    let at = usize::try_from(at).ok()?;
    bytes.get(at..at.checked_add(rb)?)
}

/// Whether `len` bytes hold the bytes a stacked tensor's shape `[ne, rows, rb]` names
/// (`verified::ik_stacked_bytes`), so every row `stacked_row` reads below `ne` and `rows` is inside
/// them (`verified::ik_row_at_in_tensor`).
fn holds_stacked_rows(len: usize, ne: u64, rows: u64, rb: u64) -> bool {
    ik_stacked_bytes(ne, rows, rb).is_some_and(|n| n <= len as u64)
}

impl ExpertRows for StExperts<'_> {
    fn decode(&self, e: usize, r: usize, out: &mut [u8]) -> Result<(), String> {
        let (rb, rs) = match self.factor {
            Some(_) => (
                ik_nib_row_bytes(self.cols as u64) as usize,
                ik_nib_row_scales(self.cols as u64) as usize,
            ),
            None => (self.cols, ik_f8_row_scales(self.cols as u64) as usize),
        };
        // a row at or past the expert's row count would read the next expert's rows
        // (`verified::ik_row_admit_in_expert`)
        let (true, Some(w), Some(s)) = (
            ik_row_admit(self.rows as u64, r as u64),
            stacked_row(self.w, e, self.rows, r, rb),
            stacked_row(self.scale, e, self.rows, r, rs),
        ) else {
            return Err(format!(
                "expert {e} row {r} is not inside its stacked tensor ({} rows an expert)",
                self.rows
            ));
        };
        match &self.factor {
            Some(f) => {
                let factor = *f
                    .get(e)
                    .ok_or_else(|| format!("expert {e} has no factor ({} experts)", f.len()))?;
                decode_nib_row(w, s, factor, &self.f8, out)
            }
            None => decode_f8_row(w, s, &self.f8, out),
        }
        Ok(())
    }
}

/// The publisher's safetensors checkpoint as a packer source.
pub struct StSource<'a> {
    store: &'a SafetensorsStore,
    prefix: &'static str,
    /// The 4-bit experts' per-expert factor from scale terms rounded to BF16 first (what the
    /// publisher's reference loader ends up computing), instead of the stored f32 values.
    amax_bf16: bool,
}

impl<'a> StSource<'a> {
    pub fn new(store: &'a SafetensorsStore) -> Self {
        Self {
            store,
            prefix: "model.llm.",
            amax_bf16: false,
        }
    }

    /// Decode the 4-bit experts with their scale terms rounded to BF16 first (see the field).
    pub fn with_amax_bf16(mut self, on: bool) -> Self {
        self.amax_bf16 = on;
        self
    }

    fn amax_per_expert(&self, suffix: &str, ne: u64) -> Result<Vec<f32>, String> {
        let (full, t) = self.meta(suffix)?;
        let n: u64 = t.shape.iter().product();
        if n == ne {
            return self.f32s(suffix, ne);
        }
        if n == 1 {
            let v = self.f32s(suffix, 1)?[0];
            return Ok(vec![v; ne as usize]);
        }
        Err(format!(
            "checkpoint tensor '{full}' has {n} values ({:?}), expected 1 or {ne}",
            t.shape
        ))
    }

    fn meta(&self, suffix: &str) -> Result<(String, &'a TensorMeta), String> {
        let full = format!("{}{}", self.prefix, suffix);
        let meta = self
            .store
            .get(&full)
            .map_err(|e| format!("missing checkpoint tensor '{full}': {e}"))?;
        Ok((full, meta))
    }

    fn mapped(&self, full: &str) -> Result<&'a [u8], String> {
        self.store
            .tensor_bytes(full)
            .map_err(|e| format!("read '{full}': {e}"))?
            .ok_or_else(|| format!("checkpoint tensor '{full}' is not mapped"))
    }

    fn raw(&self, suffix: &str, dtype: &str, shape: &[u64]) -> Result<&'a [u8], String> {
        let (full, t) = self.meta(suffix)?;
        if t.dtype != dtype || t.shape != shape {
            return Err(format!(
                "checkpoint tensor '{full}' is {} {:?}, expected {dtype} {shape:?}",
                t.dtype, t.shape
            ));
        }
        self.mapped(&full)
    }
}

impl IkSource for StSource<'_> {
    fn bf16(&self, suffix: &str, shape: &[u64]) -> Result<Cow<'_, [u8]>, String> {
        Ok(Cow::Borrowed(self.raw(suffix, "BF16", shape)?))
    }

    fn f32s(&self, suffix: &str, elems: u64) -> Result<Vec<f32>, String> {
        let (full, t) = self.meta(suffix)?;
        let bytes = self.mapped(&full)?;
        let n: u64 = t.shape.iter().product();
        if n != elems {
            return Err(format!(
                "checkpoint tensor '{full}' has {n} values ({:?}), expected {elems}",
                t.shape
            ));
        }
        match t.dtype.as_str() {
            "BF16" => Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
                .collect()),
            "F32" => Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()),
            other => Err(format!(
                "checkpoint tensor '{full}' has dtype {other}, expected BF16 or F32"
            )),
        }
    }

    fn experts(
        &self,
        l: usize,
        key: &str,
        ne: u64,
        rows: u64,
        cols: u64,
    ) -> Result<Box<dyn ExpertRows + '_>, String> {
        let base = format!("layers.{l}.mlp.experts.{key}_weight");
        let (full, t) = self.meta(&base)?;
        let w = self.mapped(&full)?;
        let (scale_dtype, factor, row_bytes, row_scales) = match t.dtype.as_str() {
            "U8" => {
                let rb = ik_nib_row_bytes(cols);
                if rb == 0 {
                    return Err(format!(
                        "expert tensor '{full}': {cols} columns are not whole blocks of 16 codes"
                    ));
                }
                let amax = self.amax_per_expert(&format!("{base}_amax"), ne)?;
                let den = self.f32s(&format!("{base}_outer_scale_denominator"), 1)?[0];
                let rf = |x: f32| {
                    if self.amax_bf16 {
                        bf16_to_f32(f32_to_bf16_rne(x))
                    } else {
                        x
                    }
                };
                let f: Vec<f32> = amax.iter().map(|&a| rf(a) / rf(den)).collect();
                ("F8_E4M3", Some(f), rb, ik_nib_row_scales(cols))
            }
            "F8_E4M3" => {
                if !ik_f8_row_ok(cols) {
                    return Err(format!(
                        "expert tensor '{full}': {cols} columns are not whole blocks of 32 values"
                    ));
                }
                ("U8", None, cols, ik_f8_row_scales(cols))
            }
            other => {
                return Err(format!(
                    "expert tensor '{full}' has dtype {other}: expected 4-bit codes (U8) or 8-bit floats (F8_E4M3)"
                ))
            }
        };
        if t.shape != [ne, rows, row_bytes] {
            return Err(format!(
                "expert tensor '{full}' has shape {:?}, expected {:?}",
                t.shape,
                [ne, rows, row_bytes]
            ));
        }
        let scale = self.raw(&format!("{base}_scale"), scale_dtype, &[ne, rows, row_scales])?;
        // the header's offsets may give a tensor fewer bytes than its shape names: refuse it here
        // rather than meet a row past its end while decoding
        for (what, bytes, rb) in [("", w, row_bytes), ("_scale", scale, row_scales)] {
            if !holds_stacked_rows(bytes.len(), ne, rows, rb) {
                return Err(format!(
                    "expert tensor '{full}{what}' holds {} bytes, fewer than its {ne} x {rows} rows of {rb}",
                    bytes.len()
                ));
            }
        }
        Ok(Box::new(StExperts {
            w,
            scale,
            factor,
            rows: rows as usize,
            cols: cols as usize,
            f8: e4m3_table(),
        }))
    }

    fn expert_fmt(&self, l: usize) -> Result<IkExpertFmt, String> {
        let (_, w13) = self.meta(&format!("layers.{l}.mlp.experts.w13_weight"))?;
        let (_, w2) = self.meta(&format!("layers.{l}.mlp.experts.w2_weight"))?;
        if w13.dtype == "U8" && w2.dtype == "U8" {
            Ok(IkExpertFmt::W4)
        } else {
            Ok(IkExpertFmt::W8)
        }
    }

    fn pack_routed_experts(
        &self,
        l: usize,
        key: &str,
        ne: u64,
        out_rows: u64,
        in_cols: u64,
        is_gu: bool,
    ) -> Result<(Vec<u8>, Vec<u8>, IkExpertFmt), String> {
        let base = format!("layers.{l}.mlp.experts.{key}_weight");
        let (full, t) = self.meta(&base)?;
        let tiles = if is_gu {
            ikr_gu_tiles(out_rows / 2)
        } else {
            ikr_dn_tiles(out_rows)
        };
        if tiles == 0 {
            return Err(format!("routed expert '{full}' produced 0 tiles"));
        }
        match t.dtype.as_str() {
            "U8" if self.expert_fmt(l)? == IkExpertFmt::W4 => {
                let rb = ik_nib_row_bytes(in_cols);
                if rb == 0 {
                    return Err(format!(
                        "expert tensor '{full}': {in_cols} columns are not whole blocks of 16 codes"
                    ));
                }
                if t.shape != [ne, out_rows, rb] {
                    return Err(format!(
                        "expert tensor '{full}' has shape {:?}, expected {:?}",
                        t.shape,
                        [ne, out_rows, rb]
                    ));
                }
                let rs = ik_nib_row_scales(in_cols);
                let w = self.mapped(&full)?;
                let scale_raw = self.raw(&format!("{base}_scale"), "F8_E4M3", &[ne, out_rows, rs])?;
                for (what, bytes, bpr) in [("", w, rb), ("_scale", scale_raw, rs)] {
                    if !holds_stacked_rows(bytes.len(), ne, out_rows, bpr) {
                        return Err(format!(
                            "expert tensor '{full}{what}' holds {} bytes, fewer than {ne} x {out_rows} rows of {bpr}",
                            bytes.len()
                        ));
                    }
                }
                let amax = self.amax_per_expert(&format!("{base}_amax"), ne)?;
                let den = self.f32s(&format!("{base}_outer_scale_denominator"), 1)?[0];
                let rf = |x: f32| {
                    if self.amax_bf16 {
                        bf16_to_f32(f32_to_bf16_rne(x))
                    } else {
                        x
                    }
                };
                let factors: Vec<f32> = amax.iter().map(|&a| 0.5 * (rf(a) / rf(den))).collect();
                let f8 = e4m3_table();

                let ext = routed_layer_extents(IkExpertFmt::W4, in_cols, tiles, ne)?;
                let exp = routed_layer_extents(IkExpertFmt::W4, in_cols, tiles, 1)?;
                let nblk = ext.nblk;
                let cp_cnt = ext.inner_cols;
                let code_bytes = ext.code_bytes;
                let scale_bytes = ext.scale_bytes;
                let exp_code = exp.code_bytes as usize;
                let exp_scale = exp.scale_bytes as usize;

                let col_offs_per_tile: Vec<Vec<usize>> = (0..tiles)
                    .map(|tt| {
                        (0..in_cols)
                            .map(|col| {
                                ikr_w4_col_off(0, tiles, cp_cnt, tt, col)
                                    .map(|o| o as usize)
                                    .ok_or_else(|| "routed W4 col offset overflow".to_string())
                            })
                            .collect::<Result<_, _>>()
                    })
                    .collect::<Result<_, _>>()?;
                let vec_offs_per_tile: Vec<Vec<usize>> = (0..tiles)
                    .map(|tt| {
                        (0..nblk)
                            .map(|b| {
                                ikr_scale_vec_off(0, tiles, nblk, tt, b)
                                    .map(|o| o as usize * 2)
                                    .ok_or_else(|| "routed W4 scale offset overflow".to_string())
                            })
                            .collect::<Result<_, _>>()
                    })
                    .collect::<Result<_, _>>()?;

                let mut codes = vec![0u8; code_bytes as usize];
                let mut scales = vec![0u8; scale_bytes as usize];
                let scale_pos: [usize; 256] =
                    std::array::from_fn(|i| ikr_scale_lane_pos(i as u64) as usize);

                let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 32);
                let per = (ne as usize).div_ceil(threads).max(1);
                let full_ref = &full;
                std::thread::scope(|sc| {
                    let mut handles = Vec::new();
                    for (chunk_idx, (c_chunk, s_chunk)) in codes
                        .chunks_mut(per * exp_code)
                        .zip(scales.chunks_mut(per * exp_scale))
                        .enumerate()
                    {
                        let e0 = chunk_idx * per;
                        let cnt = c_chunk.len() / exp_code;
                        let fac_ref = &factors;
                        let f8_ref = &f8;
                        let sp = &scale_pos;
                        let col_offs_ref = &col_offs_per_tile;
                        let vec_offs_ref = &vec_offs_per_tile;
                        handles.push(sc.spawn(move || -> Result<(), String> {
                            for le in 0..cnt {
                                let e = e0 + le;
                                let fac = fac_ref[e];
                                let c_dst = &mut c_chunk[le * exp_code..(le + 1) * exp_code];
                                let s_dst = &mut s_chunk[le * exp_scale..(le + 1) * exp_scale];
                                for tt in 0..tiles as usize {
                                    let col_offs = &col_offs_ref[tt];
                                    let vec_offs = &vec_offs_ref[tt];
                                    for i in 0..256usize {
                                        let row = routed_src_row(is_gu, tt as u64, i as u64);
                                        if !ik_row_admit(out_rows, row) {
                                            return Err(format!("expert {e} row {row} out of range"));
                                        }
                                        let w_row = stacked_row(w, e, out_rows as usize, row as usize, rb as usize)
                                            .ok_or_else(|| format!("short expert '{full_ref}' row {row}"))?;
                                        let s_row = stacked_row(scale_raw, e, out_rows as usize, row as usize, rs as usize)
                                            .ok_or_else(|| format!("short expert '{full_ref}_scale' row {row}"))?;
                                        let byte_off = i >> 1;
                                        let shift = 4 * (i & 1);
                                        for col in 0..in_cols as usize {
                                            let nib = (w_row[col >> 1] >> (4 * (col & 1))) & 0x0F;
                                            c_dst[col_offs[col] + byte_off] |= nib << shift;
                                        }
                                        for b in 0..nblk as usize {
                                            let s_h = f32_to_f16_rne(f8_ref[s_row[b] as usize] * fac);
                                            let sb = vec_offs[b] + sp[i] * 2;
                                            s_dst[sb..sb + 2].copy_from_slice(&s_h.to_le_bytes());
                                        }
                                    }
                                }
                            }
                            Ok(())
                        }));
                    }
                    for h in handles {
                        h.join().map_err(|_| "routed W4 packer thread panicked".to_string())??;
                    }
                    Ok::<(), String>(())
                })?;
                Ok((codes, scales, IkExpertFmt::W4))
            }
            "F8_E4M3" => {
                if !ik_f8_row_ok(in_cols) {
                    return Err(format!(
                        "expert tensor '{full}': {in_cols} columns are not whole blocks of 32 values"
                    ));
                }
                if t.shape != [ne, out_rows, in_cols] {
                    return Err(format!(
                        "expert tensor '{full}' has shape {:?}, expected {:?}",
                        t.shape,
                        [ne, out_rows, in_cols]
                    ));
                }
                let rs = ik_f8_row_scales(in_cols);
                let w = self.mapped(&full)?;
                let scale_raw = self.raw(&format!("{base}_scale"), "U8", &[ne, out_rows, rs])?;
                for (what, bytes, bpr) in [("", w, in_cols), ("_scale", scale_raw, rs)] {
                    if !holds_stacked_rows(bytes.len(), ne, out_rows, bpr) {
                        return Err(format!(
                            "expert tensor '{full}{what}' holds {} bytes, fewer than {ne} x {out_rows} rows of {bpr}",
                            bytes.len()
                        ));
                    }
                }
                let f8 = e4m3_table();
                let nblk = rs;
                let full_ref = &full;
                let (codes, scales) = pack_w8_rows_with(
                    ne,
                    tiles,
                    in_cols,
                    is_gu,
                    "routed F8->W8 packer thread panicked",
                    || (),
                    |_, e, row, row_vals| {
                        if !ik_row_admit(out_rows, row) {
                            return Err(format!("expert {e} row {row} out of range"));
                        }
                        let w_row = stacked_row(w, e, out_rows as usize, row as usize, in_cols as usize)
                            .ok_or_else(|| format!("short expert '{full_ref}' row {row}"))?;
                        let s_row = stacked_row(scale_raw, e, out_rows as usize, row as usize, rs as usize)
                            .ok_or_else(|| format!("short expert '{full_ref}_scale' row {row}"))?;
                        for b in 0..nblk as usize {
                            let ds = 2f32.powi(i32::from(s_row[b]) - 127);
                            let base_c = b * 32;
                            for c in 0..32 {
                                row_vals[base_c + c] = f8[w_row[base_c + c] as usize] * ds;
                            }
                        }
                        Ok(())
                    },
                )?;
                Ok((codes, scales, IkExpertFmt::W8))
            }
            _ => {
                let ex = self.experts(l, key, ne, out_rows, in_cols)?;
                let (codes, scales) = pack_w8_from_rows(ex.as_ref(), ne, tiles, in_cols, is_gu)?;
                Ok((codes, scales, IkExpertFmt::W8))
            }
        }
    }
}

/// A job over a range of output rows: `(first row, the rows' bytes)`.
type RowJob<'a> = dyn Fn(usize, &mut [u8]) -> Result<(), String> + Sync + 'a;

/// Runs `job(first_row, rows_slice)` over `out` split into row ranges on the host's threads.
fn par_rows(out: &mut [u8], row_bytes: usize, job: &RowJob<'_>) -> Result<(), String> {
    let rows = out.len() / row_bytes;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 32);
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        let handles: Vec<_> = out
            .chunks_mut(per * row_bytes)
            .enumerate()
            .map(|(i, chunk)| sc.spawn(move || job(i * per, chunk)))
            .collect();
        let mut res = Ok(());
        for h in handles {
            let r = h
                .join()
                .unwrap_or_else(|_| Err("a weight decode thread panicked".into()));
            if res.is_ok() {
                res = r;
            }
        }
        res
    })
}

/// Weight packer for the Inkling 2 family.
pub struct IkPacker<'a> {
    src: &'a dyn IkSource,
    d: &'a IkDims,
    max_ctx: u64,
    routed_cache: Mutex<HashMap<String, Vec<u8>>>,
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn i32s_to_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// A per-channel table of whole blocks of 128 in the split layout of a tensor that holds its
/// channels in the checkpoint's order: lane `64 h + j` of a block holds channel `2 j + h`.
pub fn split_table(natural: &[f32]) -> Vec<f32> {
    assert_eq!(natural.len() % 128, 0, "a channel table is whole blocks of 128");
    let mut out = vec![0f32; natural.len()];
    for (b, blk) in natural.chunks(128).enumerate() {
        for (c, &v) in blk.iter().enumerate() {
            out[128 * b + lane_in_block(c as u64) as usize] = v;
        }
    }
    out
}

/// The same for a BF16 tensor whose rows are in lane order: after the converting load, lane
/// `j` of vector `h` holds the channel at position `2 j + h` of the lane-ordered block.
pub fn split_table_lane(natural: &[f32]) -> Vec<f32> {
    assert_eq!(natural.len() % 128, 0, "a channel table is whole blocks of 128");
    let mut out = vec![0f32; natural.len()];
    for (b, blk) in natural.chunks(128).enumerate() {
        for h in 0..2u64 {
            for j in 0..64u64 {
                out[128 * b + 64 * h as usize + j as usize] = blk[natural_channel(h, j) as usize];
            }
        }
    }
    out
}

/// Rows of `row_bytes` bytes in lane order inside each block of 128 rows.
pub fn lane_rows(src: &[u8], row_bytes: usize) -> Vec<u8> {
    let rows = src.len() / row_bytes;
    assert_eq!(rows % 128, 0, "lane order needs whole blocks of 128 rows");
    let mut out = vec![0u8; src.len()];
    for r in 0..rows {
        let dst = (r / 128) * 128 + lane_in_block((r % 128) as u64) as usize;
        out[dst * row_bytes..(dst + 1) * row_bytes]
            .copy_from_slice(&src[r * row_bytes..(r + 1) * row_bytes]);
    }
    out
}

/// The position scale of the global layers' queries and relative states, in f32 as the
/// reference computes it: `1 + alpha ln(max((pos + 1) / floor, 1))`.
pub fn tau_table(floor: u64, alpha: f32, lmax: u64) -> Vec<f32> {
    (0..lmax)
        .map(|p| {
            let x = ((p + 1) as f32 / floor.max(1) as f32).max(1.0);
            1.0 + alpha * x.ln()
        })
        .collect()
}

impl<'a> IkPacker<'a> {
    pub fn new(src: &'a dyn IkSource, d: &'a IkDims, max_ctx: u64) -> Self {
        Self {
            src,
            d,
            max_ctx,
            routed_cache: Mutex::new(HashMap::new()),
        }
    }

    fn routed_part(
        &self,
        l: usize,
        key: &str,
        out_rows: u64,
        in_cols: u64,
        is_gu: bool,
        want_scales: bool,
    ) -> Result<Vec<u8>, String> {
        let tag = if is_gu { "W13" } else { "W2" };
        let want_name = format!("L{l}_{tag}_{}", if want_scales { "s" } else { "c" });
        if let Some(hit) = self
            .routed_cache
            .lock()
            .expect("routed cache poisoned")
            .remove(&want_name)
        {
            return Ok(hit);
        }
        let other_name = format!("L{l}_{tag}_{}", if want_scales { "c" } else { "s" });
        let (codes, scales, _) =
            self.src
                .pack_routed_experts(l, key, self.d.ne, out_rows, in_cols, is_gu)?;
        let (ret, store) = if want_scales {
            (scales, codes)
        } else {
            (codes, scales)
        };
        self.routed_cache
            .lock()
            .expect("routed cache poisoned")
            .insert(other_name, store);
        Ok(ret)
    }

    fn raw(&self, suffix: &str, _dtype: &str, shape: &[u64]) -> Result<Cow<'a, [u8]>, String> {
        self.src.bf16(suffix, shape)
    }

    fn f32s(&self, suffix: &str, elems: u64) -> Result<Vec<f32>, String> {
        self.src.f32s(suffix, elems)
    }

    /// The wide gate (`up == false`) or up matrix `{hid, ne ef}`: expert `e`'s channel `i` is
    /// row `2 i` (gate) or `2 i + 1` (up) of its stored gate-up matrix.
    fn pack_gate_up(&self, l: usize, up: bool) -> Result<Vec<u8>, String> {
        let d = self.d;
        let src = self.src.experts(l, "w13", d.ne, 2 * d.ef, d.hid)?;
        let (ef, row) = (d.ef as usize, d.hid as usize * 2);
        let mut out = vec![0u8; d.ne as usize * ef * row];
        par_rows(&mut out, row, &|first, chunk| {
            for (k, o) in chunk.chunks_mut(row).enumerate() {
                let n = first + k;
                src.decode(n / ef, 2 * (n % ef) + usize::from(up), o)?;
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// The wide down matrix `{ne ef, hid}`: output row `n` holds, expert after expert, the
    /// `ef` values of the experts' row `n`.
    fn pack_down(&self, l: usize) -> Result<Vec<u8>, String> {
        let d = self.d;
        let src = self.src.experts(l, "w2", d.ne, d.hid, d.ef)?;
        let (ef2, ne) = (d.ef as usize * 2, d.ne as usize);
        let row = ne * ef2;
        let mut out = vec![0u8; d.hid as usize * row];
        par_rows(&mut out, row, &|first, chunk| {
            for (k, o) in chunk.chunks_mut(row).enumerate() {
                for e in 0..ne {
                    src.decode(e, first + k, &mut o[e * ef2..(e + 1) * ef2])?;
                }
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// Convolution taps of `c` channels as the f32 table `{c, taps}` (column = lag), from the
    /// checkpoint's tap-major (`lag_major`: `[taps, c]`, index = lag) or channel-major form
    /// (`[c, 1, taps]`, the last tap weights the row itself).
    fn conv_table(&self, suffix: &str, c: u64, lag_major: bool, lane: bool) -> Result<Vec<u8>, String> {
        let taps = self.d.taps as usize;
        let c = c as usize;
        let w = self.f32s(suffix, (c * taps) as u64)?;
        let mut out = Vec::with_capacity(c * taps);
        for lag in 0..taps {
            let nat: Vec<f32> = (0..c)
                .map(|ch| {
                    if lag_major {
                        w[lag * c + ch]
                    } else {
                        w[ch * taps + (taps - 1 - lag)]
                    }
                })
                .collect();
            out.extend(if lane { split_table_lane(&nat) } else { split_table(&nat) });
        }
        Ok(f32s_to_bytes(&out))
    }

    /// The block-diagonal, reversed and padded relative-position projection `{128, 8 bw, 1}`.
    fn pack_proj(&self, l: usize) -> Result<Vec<u8>, String> {
        let d = self.d;
        let (ext, bw, dr) = (d.extent as usize, d.bw as usize, d.d_rel as usize);
        let proj = self.raw(
            &format!("layers.{l}.attn.rel_logits_proj.proj"),
            "BF16",
            &[d.d_rel, d.extent],
        )?;
        let mut out = vec![0u8; 8 * bw * 128 * 2];
        for h in 0..8 {
            for x in 0..bw {
                let dist = ik_bias_src(d.extent, x as u64) as usize;
                if dist >= ext {
                    continue;
                }
                let row = (h * bw + x) * 128 * 2;
                for c in 0..dr {
                    let s = (c * ext + dist) * 2;
                    let o = row + (dr * h + c) * 2;
                    out[o..o + 2].copy_from_slice(&proj[s..s + 2]);
                }
            }
        }
        Ok(out)
    }

    fn concat_rows(&self, parts: &[&[u8]], total_rows: u64, row_bytes: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity((total_rows * row_bytes) as usize);
        for p in parts {
            out.extend_from_slice(p);
        }
        out.resize((total_rows * row_bytes) as usize, 0);
        out
    }

    /// A mix query table `{hid, nq}` in the split layout from `nq` natural rows.
    fn query_table(rows: &[Vec<f32>]) -> Vec<u8> {
        let mut out = Vec::new();
        for r in rows {
            out.extend(split_table(r));
        }
        f32s_to_bytes(&out)
    }

    /// Packs the device tensor `name`.
    pub fn pack(&self, name: &str) -> Result<Cow<'a, [u8]>, String> {
        let d = self.d;
        let hid = d.hid;
        let bf = "BF16";
        match name {
            "embed" => return self.raw("embed.weight", bf, &[d.vocab_in, hid]),
            "w_embed" => return self.raw("embed_norm.weight", bf, &[hid]),
            "w_final" => return self.raw("norm.weight", bf, &[hid]),
            "W_head" => {
                let w = self.raw("unembed.weight", bf, &[d.vocab_in, hid])?;
                let keep = (d.vocab_out * hid * 2) as usize;
                return Ok(Cow::Owned(self.concat_rows(&[&w[..keep]], d.vocab_pad, hid * 2)));
            }
            "q_fin" => {
                let q = self.f32s("attn_res.output_query", hid)?;
                return Ok(Cow::Owned(Self::query_table(&[q])));
            }
            "tau" => {
                return Ok(Cow::Owned(f32s_to_bytes(&tau_table(
                    d.log_floor,
                    d.log_alpha,
                    self.max_ctx,
                ))))
            }
            "ctl_fin" => {
                let ns = d.fin_states().len();
                return Ok(Cow::Owned(i32s_to_bytes(&[ns as i32, 0, 0, 0, 0, 0, 0, 0])));
            }
            "ctl_msg" => {
                return Ok(Cow::Owned(i32s_to_bytes(&[
                    d.n_msg_states as i32,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ])))
            }
            "ik_nvfp4_lut" => return Ok(Cow::Owned(ik_nvfp4_lut())),
            _ => {}
        }
        if let Some(rest) = name.strip_prefix('A') {
            if let Some((idx, "ctl")) = rest.split_once('_') {
                let a: usize = idx.parse().map_err(|_| format!("bad tensor name '{name}'"))?;
                let app = d.apps.get(a).ok_or_else(|| format!("no application {a}"))?;
                return Ok(Cow::Owned(i32s_to_bytes(&[
                    app.ns_attn as i32,
                    i32::from(app.layer == 0),
                    i32::from(app.opens_block),
                    0,
                    0,
                    0,
                    0,
                    0,
                ])));
            }
        }
        let (l, what) = name
            .strip_prefix('L')
            .and_then(|r| r.split_once('_'))
            .and_then(|(i, w)| i.parse::<usize>().ok().map(|i| (i, w)))
            .ok_or_else(|| format!("no packer for device tensor '{name}'"))?;
        let sliding = d.layer_sliding(l);
        let lp = format!("layers.{l}.");
        let at = format!("{lp}attn.");
        let (nh, nkv) = (d.nh, d.nkv);
        let kvw = 128 * nkv;
        Ok(match what {
            "w_q" | "w_k" | "w_v" => {
                let src = if sliding {
                    format!("{lp}attn_norm.weight")
                } else {
                    format!("{lp}{}_input_norm.weight", &what[2..])
                };
                self.raw(&src, bf, &[hid])?
            }
            "w_mlp" => self.raw(&format!("{lp}mlp_norm.weight"), bf, &[hid])?,
            "W_qx" => {
                let q = self.raw(&format!("{at}wq_du.weight"), bf, &[128 * nh, hid])?;
                let g = self.raw(&format!("{at}gate_attn.weight"), bf, &[128 * nh, hid])?;
                let r = self.raw(&format!("{at}wr_du.weight"), bf, &[d.d_rel * nh, hid])?;
                Cow::Owned(self.concat_rows(&[&q, &g, &r], d.qw, hid * 2))
            }
            "W_sink" => self.raw(&format!("{at}attn_sink_proj.weight"), bf, &[nh, hid])?,
            "W_kx" | "W_vx" => {
                let is_v = what == "W_vx";
                let kv = if is_v { "v" } else { "k" };
                let w = self.raw(&format!("{at}w{kv}_dv.weight"), bf, &[kvw, hid])?;
                let main: Cow<'a, [u8]> = if is_v {
                    Cow::Owned(lane_rows(&w, (hid * 2) as usize))
                } else {
                    w
                };
                if sliding {
                    main
                } else {
                    let taps = self.raw(
                        &format!("{at}{kv}_sconv.weight_proj"),
                        bf,
                        &[d.taps * nkv, hid],
                    )?;
                    Cow::Owned(self.concat_rows(&[&main, &taps], d.kxw(false), hid * 2))
                }
            }
            "qsc" => {
                let s = self.f32s(&format!("{at}q_l2_clip.scale"), 128 * nh)?;
                Cow::Owned(f32s_to_bytes(&split_table(&s)))
            }
            "rsc" => {
                let s = self.f32s(&format!("{at}rel_l2_clip.scale"), d.d_rel * nh)?;
                Cow::Owned(f32s_to_bytes(&split_table(&s)))
            }
            "PROJ" => Cow::Owned(self.pack_proj(l)?),
            "kcw" | "vcw" => {
                let is_v = what == "vcw";
                let kv = if is_v { "v" } else { "k" };
                if sliding {
                    Cow::Owned(self.conv_table(&format!("{at}{kv}_sconv.weight"), kvw, false, is_v)?)
                } else {
                    Cow::Owned(self.conv_table(&format!("{at}{kv}_sconv.static_w"), kvw, true, is_v)?)
                }
            }
            "W_o" => self.raw(&format!("{at}wo_ud.weight"), bf, &[hid, 128 * nh])?,
            "acw" => Cow::Owned(self.conv_table(&format!("{lp}attn_sconv.weight"), hid, false, false)?),
            "mcw" => Cow::Owned(self.conv_table(&format!("{lp}mlp_sconv.weight"), hid, false, false)?),
            "q_attn" => {
                let nq = if sliding { 1 } else { 3 };
                if l == 0 {
                    // the first layer has no mix (its control word bypasses it)
                    Cow::Owned(vec![0u8; (hid * nq * 4) as usize])
                } else {
                    let q = self.f32s(&format!("attn_res.attn_queries.{}", l - 1), hid * nq)?;
                    let rows: Vec<Vec<f32>> = q.chunks(hid as usize).map(<[f32]>::to_vec).collect();
                    Cow::Owned(Self::query_table(&rows))
                }
            }
            "q_mlp" => {
                let q = self.f32s(&format!("attn_res.mlp_queries.{l}"), hid)?;
                Cow::Owned(Self::query_table(&[q]))
            }
            "q_msg" => {
                let app = d
                    .apps
                    .iter()
                    .find(|a| a.layer == l && a.looped)
                    .ok_or_else(|| format!("layer {l} is not in the looped span"))?;
                let span = d.apps.iter().filter(|a| a.looped && a.pass == 0).count() as u64;
                let qm = self.f32s("middle_cycle_cross_pass_queries", 2 * span * hid)?;
                let kv = self.f32s("middle_cycle_cross_pass_kv_queries", 2 * span * hid)?;
                let h = hid as usize;
                let row = |t: &[f32], i: usize| t[i * h..(i + 1) * h].to_vec();
                let li = app.loop_index;
                // the order of the gates: query, key, value, MLP
                Cow::Owned(Self::query_table(&[
                    row(&qm, 2 * li),
                    row(&kv, 2 * li),
                    row(&kv, 2 * li + 1),
                    row(&qm, 2 * li + 1),
                ]))
            }
            "W_router" => self.raw(&format!("{lp}mlp.gate.weight"), bf, &[d.ne, hid])?,
            "rb" => {
                let b = self.f32s(&format!("{lp}mlp.gate.bias"), d.ne)?;
                Cow::Owned(f32s_to_bytes(&split_table(&b)))
            }
            "W_gate" => Cow::Owned(self.pack_gate_up(l, false)?),
            "W_up" => Cow::Owned(self.pack_gate_up(l, true)?),
            "W_down" => Cow::Owned(self.pack_down(l)?),
            "W13_c" => Cow::Owned(self.routed_part(l, "w13", 2 * d.ef, hid, true, false)?),
            "W13_s" => Cow::Owned(self.routed_part(l, "w13", 2 * d.ef, hid, true, true)?),
            "W2_c" => Cow::Owned(self.routed_part(l, "w2", hid, d.ef, false, false)?),
            "W2_s" => Cow::Owned(self.routed_part(l, "w2", hid, d.ef, false, true)?),
            _ => {
                let ctrl = |which: &str| match which {
                    "q" => Some("attn"),
                    "k" => Some("k"),
                    "v" => Some("v"),
                    "mlp" => Some("mlp"),
                    _ => None,
                };
                if let Some(c) = what.strip_prefix("W_gd_").and_then(ctrl) {
                    self.raw(
                        &format!("{lp}controllers.{c}.down_weight"),
                        bf,
                        &[d.rank, 2 * hid],
                    )?
                } else if let Some(c) = what.strip_prefix("W_gf_").and_then(ctrl) {
                    let f = self.raw(&format!("{lp}controllers.{c}.forget_weight"), bf, &[hid, d.rank])?;
                    let g = self.raw(&format!("{lp}controllers.{c}.gain_weight"), bf, &[hid, d.rank])?;
                    let rb = (d.rank * 2) as usize;
                    if d.fg_f32 {
                        // an f32 result in the split layout: the rows in lane order
                        let mut out = lane_rows(&f, rb);
                        out.extend(lane_rows(&g, rb));
                        Cow::Owned(out)
                    } else {
                        Cow::Owned(self.concat_rows(&[&f, &g], 2 * hid, d.rank * 2))
                    }
                } else {
                    return Err(format!("no packer for device tensor '{name}'"));
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exponent 7, mantissa 0
        assert_eq!(e4m3_to_f32(0x3C), 1.5);
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal: 1/8 * 2^-6
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // largest finite: 1.75 * 2^8
        assert!(e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn nib_row_decodes_low_nibble_first() {
        let f8 = e4m3_table();
        // one block of 16 codes: bytes hold (even, odd) = (low, high) nibbles; scale 2.0 (0x40); factor 0.5
        let codes = [0x21u8, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F];
        let mut out = [0u8; 32];
        decode_nib_row(&codes, &[0x40], 0.5, &f8, &mut out);
        let vals: Vec<f32> = out
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        // codes 1,2,3,...: E2M1[1] = 0.5, [2] = 1.0, ... each times 2.0 * 0.5
        let want: Vec<f32> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
            .iter()
            .map(|&c| E2M1[c])
            .collect();
        assert_eq!(vals, want);
    }

    #[test]
    fn stacked_rows_stay_inside_their_tensor() {
        // 2 experts x 3 rows of 4 bytes: row r of expert e at (3 e + r) 4
        let t: Vec<u8> = (0..24).collect();
        assert_eq!(stacked_row(&t, 0, 3, 0, 4), Some(&t[0..4]));
        assert_eq!(stacked_row(&t, 1, 3, 2, 4), Some(&t[20..24]));
        // a row past the tensor's bytes, and offsets that overflow u64, are refused
        assert_eq!(stacked_row(&t, 2, 3, 0, 4), None);
        assert_eq!(stacked_row(&t[..23], 1, 3, 2, 4), None);
        assert_eq!(stacked_row(&t, usize::MAX, 3, 0, 4), None);
        assert!(holds_stacked_rows(24, 2, 3, 4));
        assert!(!holds_stacked_rows(23, 2, 3, 4));
        assert!(!holds_stacked_rows(24, u64::MAX, 3, 4));
    }

    #[test]
    fn a_short_stacked_tensor_is_an_error_not_a_panic() {
        // one expert of two 32-value rows in the 8-bit float form: 64 value bytes and 2 scale
        // bytes, with the value tensor one byte short
        let ex = StExperts {
            w: &[0x38; 63],
            scale: &[127; 2],
            factor: None,
            rows: 2,
            cols: 32,
            f8: e4m3_table(),
        };
        let mut out = [0u8; 64];
        assert!(ex.decode(0, 0, &mut out).is_ok());
        let e = ex.decode(0, 1, &mut out).unwrap_err();
        assert!(e.contains("not inside"), "{e}");
        // two whole experts: a row past the expert's row count is refused, not read from the
        // next expert's bytes
        let ex = StExperts {
            w: &[0x38; 128],
            scale: &[127; 4],
            ..ex
        };
        assert!(ex.decode(1, 1, &mut out).is_ok());
        assert!(ex.decode(0, 2, &mut out).is_err());
        assert!(ex.decode(2, 0, &mut out).is_err());
    }

    #[test]
    fn f8_row_applies_the_power_of_two_scale() {
        let f8 = e4m3_table();
        let vals = [0x38u8; 32]; // 1.0
        let mut out = [0u8; 64];
        decode_f8_row(&vals, &[125], &f8, &mut out);
        let v = bf16_to_f32(u16::from_le_bytes([out[0], out[1]]));
        assert_eq!(v, 0.25);
    }

    #[test]
    fn split_tables_and_lane_rows_agree_with_the_load_convention() {
        let nat: Vec<f32> = (0..128).map(|c| c as f32).collect();
        let s = split_table(&nat);
        assert_eq!((s[0], s[1], s[64], s[65]), (0.0, 2.0, 1.0, 3.0));
        // lane order: position p < 64 holds channel 2 p, p >= 64 channel 2 (p - 64) + 1
        let rows: Vec<u8> = (0..128u8).collect();
        let lr = lane_rows(&rows, 1);
        assert_eq!((lr[0], lr[1], lr[64], lr[65]), (0, 2, 1, 3));
        // a converting load of a lane-ordered block gives v1[j] = position 2 j, v2[j] = position 2 j + 1
        let sl = split_table_lane(&nat);
        assert_eq!(sl[0], f32::from(lr[0]));
        assert_eq!(sl[1], f32::from(lr[2]));
        assert_eq!(sl[64], f32::from(lr[1]));
        assert_eq!(sl[64 + 33], f32::from(lr[67]));
    }

    #[test]
    fn tau_is_one_below_the_floor_and_grows_with_the_logarithm() {
        let t = tau_table(8192, 0.1, 20000);
        assert_eq!(t[0], 1.0);
        assert_eq!(t[8191], 1.0);
        assert!(t[8192] > 1.0 && t[8192] < 1.0001);
        assert!((t[16383] - (1.0 + 0.1 * 2f32.ln())).abs() < 1e-6);
    }

    #[test]
    fn f16_roundtrip_and_rne() {
        for &v in &[0.0f32, -0.0, 1.0, -1.5, 6.0, 448.0, 65504.0, 2f32.powi(-14), 2f32.powi(-24)] {
            let h = f32_to_f16_rne(v);
            let back = f16_to_f32(h);
            assert_eq!(back.to_bits(), v.to_bits(), "v={v}");
        }
        // ties-to-even at midpoint between 1.0 (0x3C00) and 1.0 + 2^-10 (0x3C01)
        let mid_even = 1.0f32 + 0.5 * 2f32.powi(-10);
        assert_eq!(f32_to_f16_rne(mid_even), 0x3C00);
        let mid_odd = 1.0f32 + 1.5 * 2f32.powi(-10);
        assert_eq!(f32_to_f16_rne(mid_odd), 0x3C02);
        // overflow clamps to infinity
        assert_eq!(f32_to_f16_rne(100_000.0), 0x7C00);
        assert_eq!(f32_to_f16_rne(-100_000.0), 0xFC00);
    }

    fn ref_matvec_bf16(ex: &dyn ExpertRows, e: usize, rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        let mut bf = vec![0u8; cols * 2];
        let mut out = vec![0f32; rows];
        for r in 0..rows {
            ex.decode(e, r, &mut bf).unwrap();
            let mut acc = 0.0f32;
            for (c, ch) in bf.chunks_exact(2).enumerate() {
                let w = bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]]));
                acc += w * x[c];
            }
            out[r] = acc;
        }
        out
    }

    #[test]
    fn routed_w4_packing_matches_decoded_gu_and_dn_matvec() {
        let lut_bytes = ik_nvfp4_lut();
        let lut: Vec<f32> = lut_bytes.iter().map(|&b| f32::from(b as i8)).collect();

        // GU: ne=2, ef=128 (1 tile), in_cols=64 (4 blocks of 16)
        let (ne, ef, cols) = (2usize, 128usize, 64usize);
        let out_rows = 2 * ef;
        let rb = cols / 2;
        let rs = cols / 16;
        let mut w_raw = vec![0u8; ne * out_rows * rb];
        let mut s_raw = vec![0u8; ne * out_rows * rs];
        for (idx, b) in w_raw.iter_mut().enumerate() {
            *b = (((idx * 7 + 3) & 0x0F) | (((idx * 11 + 5) & 0x0F) << 4)) as u8;
        }
        // Include E4M3 exponent 15 (0x78 -> 256.0, 0x7F? skip NaN) to test exp==15 scale survival
        let valid_f8: Vec<u8> = (0x20..=0x7Eu8).collect();
        for (idx, b) in s_raw.iter_mut().enumerate() {
            *b = valid_f8[idx % valid_f8.len()];
        }
        let factors = [0.25f32, 0.5f32]; // 0.5 * (amax / den)
        let f8 = e4m3_table();
        let ex = StExperts {
            w: &w_raw,
            scale: &s_raw,
            factor: Some(factors.to_vec()),
            rows: out_rows,
            cols,
            f8,
        };
        let x: Vec<f32> = (0..cols).map(|c| ((c as f32) - 31.5) * 0.03).collect();

        // Pack GU directly using same W4 kernel layout helper via StSource logic
        let tiles = ikr_gu_tiles(ef as u64);
        let nblk = (cols / 16) as u64;
        let cp_cnt = (cols / 2) as u64;
        let code_bytes = ikr_w4_code_bytes(cols as u64, tiles, ne as u64).unwrap() as usize;
        let scale_bytes = ikr_scale_bytes(cols as u64, 16, tiles, ne as u64).unwrap() as usize;
        let exp_code = ikr_w4_code_bytes(cols as u64, tiles, 1).unwrap() as usize;
        let exp_scale = ikr_scale_bytes(cols as u64, 16, tiles, 1).unwrap() as usize;
        let mut codes = vec![0u8; code_bytes];
        let mut scales = vec![0u8; scale_bytes];
        for e in 0..ne {
            let fac = 0.5 * factors[e];
            let c_dst = &mut codes[e * exp_code..(e + 1) * exp_code];
            let s_dst = &mut scales[e * exp_scale..(e + 1) * exp_scale];
            for tt in 0..tiles {
                for i in 0..256usize {
                    let row = routed_src_row(true, tt, i as u64) as usize;
                    let w_row = stacked_row(&w_raw, e, out_rows, row, rb).unwrap();
                    let s_row = stacked_row(&s_raw, e, out_rows, row, rs).unwrap();
                    for col in 0..cols {
                        let nib = (w_row[col >> 1] >> (4 * (col & 1))) & 0x0F;
                        let off = ikr_w4_col_off(0, tiles, cp_cnt, tt, col as u64).unwrap() as usize;
                        c_dst[off + (i >> 1)] |= nib << (4 * (i & 1));
                    }
                    for b in 0..nblk as usize {
                        let s_h = f32_to_f16_rne(f8[s_row[b] as usize] * fac);
                        let vo = ikr_scale_vec_off(0, tiles, nblk, tt, b as u64).unwrap() as usize * 2;
                        let sb = vo + ikr_scale_lane_pos(i as u64) as usize * 2;
                        s_dst[sb..sb + 2].copy_from_slice(&s_h.to_le_bytes());
                    }
                }
            }
        }

        // Verify reconstructed GU weights and matvec against ref_matvec_bf16
        let mut bf_buf = vec![0u8; cols * 2];
        for e in 0..ne {
            let ref_y = ref_matvec_bf16(&ex, e, out_rows, cols, &x);
            let c_exp = &codes[e * exp_code..(e + 1) * exp_code];
            let s_exp = &scales[e * exp_scale..(e + 1) * exp_scale];
            for i in 0..256usize {
                let row = routed_src_row(true, 0, i as u64) as usize;
                ex.decode(e, row, &mut bf_buf).unwrap();
                let mut acc = 0.0f32;
                let mut l1_x = 0.0f32;
                for col in 0..cols {
                    let off = ikr_w4_col_off(0, tiles, cp_cnt, 0, col as u64).unwrap() as usize;
                    let byte = c_exp[off + (i >> 1)];
                    let nib = ((byte >> (4 * (i & 1))) & 0x0F) as usize;
                    let b = col / 16;
                    let vo = ikr_scale_vec_off(0, tiles, nblk, 0, b as u64).unwrap() as usize * 2;
                    let sb = vo + ikr_scale_lane_pos(i as u64) as usize * 2;
                    let sh = u16::from_le_bytes([s_exp[sb], s_exp[sb + 1]]);
                    let sv = f16_to_f32(sh);
                    let w_rec = lut[nib] * sv;
                    let w_ref = bf16_to_f32(u16::from_le_bytes([bf_buf[col * 2], bf_buf[col * 2 + 1]]));
                    assert!(
                        (w_rec - w_ref).abs() <= 0.01 * w_ref.abs().max(1e-4),
                        "W4 GU e={e} row={row} col={col}: w_rec={w_rec} w_ref={w_ref}"
                    );
                    l1_x += w_ref.abs() * x[col].abs();
                    acc += w_rec * x[col];
                }
                let diff = (acc - ref_y[row]).abs();
                assert!(
                    diff <= 0.01 * l1_x.max(1e-4),
                    "GU e={e} row={row}: acc={acc} ref={} diff={diff}",
                    ref_y[row]
                );
            }
        }
    }

    #[test]
    fn routed_w8_packing_matches_decoded_gu_and_dn_matvec() {
        struct DummyEx {
            data: Vec<u16>,
            rows: usize,
            cols: usize,
        }
        impl ExpertRows for DummyEx {
            fn decode(&self, e: usize, r: usize, out: &mut [u8]) -> Result<(), String> {
                let start = (e * self.rows + r) * self.cols;
                for (i, o) in out.chunks_exact_mut(2).enumerate() {
                    o.copy_from_slice(&self.data[start + i].to_le_bytes());
                }
                Ok(())
            }
        }

        for is_gu in [true, false] {
            let (ne, cols, out_rows) = if is_gu { (2usize, 64usize, 256usize) } else { (2usize, 128usize, 256usize) };
            let tiles = if is_gu { ikr_gu_tiles(128) } else { ikr_dn_tiles(256) };
            let nblk = (cols / 32) as u64;
            let mut data = vec![0u16; ne * out_rows * cols];
            for (idx, d) in data.iter_mut().enumerate() {
                let v = (((idx % 97) as f32) - 48.0) * 0.015;
                *d = f32_to_bf16_rne(v);
            }
            let ex = DummyEx { data, rows: out_rows, cols };
            let (codes, scales) = pack_w8_from_rows(&ex, ne as u64, tiles, cols as u64, is_gu).unwrap();
            let exp_code = ikr_w8_code_bytes(cols as u64, tiles, 1).unwrap() as usize;
            let exp_scale = ikr_scale_bytes(cols as u64, 32, tiles, 1).unwrap() as usize;
            let x: Vec<f32> = (0..cols).map(|c| ((c as f32) - 30.0) * 0.05).collect();
            let mut bf_buf = vec![0u8; cols * 2];

            for e in 0..ne {
                let ref_y = ref_matvec_bf16(&ex, e, out_rows, cols, &x);
                let c_exp = &codes[e * exp_code..(e + 1) * exp_code];
                let s_exp = &scales[e * exp_scale..(e + 1) * exp_scale];
                for i in 0..256usize {
                    let row = routed_src_row(is_gu, 0, i as u64) as usize;
                    ex.decode(e, row, &mut bf_buf).unwrap();
                    let mut acc = 0.0f32;
                    let mut l1_bound = 0.0f32;
                    for col in 0..cols {
                        let off = ikr_w8_col_off(0, tiles, cols as u64, 0, col as u64).unwrap() as usize;
                        let q = c_exp[off + i] as i8;
                        let b = col / 32;
                        let vo = ikr_scale_vec_off(0, tiles, nblk, 0, b as u64).unwrap() as usize * 2;
                        let sb = vo + ikr_scale_lane_pos(i as u64) as usize * 2;
                        let sh = u16::from_le_bytes([s_exp[sb], s_exp[sb + 1]]);
                        let sv = f16_to_f32(sh);
                        let w_rec = f32::from(q) * sv;
                        let w_ref = bf16_to_f32(u16::from_le_bytes([bf_buf[col * 2], bf_buf[col * 2 + 1]]));
                        assert!(
                            (w_rec - w_ref).abs() <= 0.55 * sv + 1e-5,
                            "W8 is_gu={is_gu} e={e} row={row} col={col}: w_rec={w_rec} w_ref={w_ref} sv={sv}"
                        );
                        l1_bound += 0.55 * sv * x[col].abs();
                        acc += w_rec * x[col];
                    }
                    let diff = (acc - ref_y[row]).abs();
                    assert!(
                        diff <= l1_bound + 1e-4,
                        "W8 is_gu={is_gu} e={e} row={row}: acc={acc} ref={} diff={diff} bound={l1_bound}",
                        ref_y[row]
                    );
                }
            }
        }
    }

    #[test]
    fn ikr_use_routed_defaults_to_routed_only_for_valid_inkling2() {
        use crate::verified::{ikr_use_routed, IkMoeForm};
        assert!(ikr_use_routed(true, IkMoeForm::IkMoeAuto, true));
        assert!(!ikr_use_routed(true, IkMoeForm::IkMoeAuto, false));
        assert!(!ikr_use_routed(true, IkMoeForm::IkMoeDense, true));
        assert!(ikr_use_routed(true, IkMoeForm::IkMoeRouted, true));
        assert!(!ikr_use_routed(true, IkMoeForm::IkMoeRouted, false));
        assert!(!ikr_use_routed(false, IkMoeForm::IkMoeAuto, true));
    }

    #[test]
    fn routed_layer_extents_uses_verified_guards_and_sizes() {
        let w4 = routed_layer_extents(IkExpertFmt::W4, 2304, 9, 128).unwrap();
        assert_eq!((w4.blk, w4.nblk, w4.inner_cols), (16, 144, 1152));
        assert_eq!(w4.code_dims, [256, 1152, 9, 128]);
        assert_eq!(w4.scale_dims, [256, 144, 9, 128]);
        assert_eq!(w4.code_bytes, 256 * 1152 * 9 * 128);
        assert_eq!(w4.scale_bytes, 256 * 144 * 9 * 128 * 2);

        let w8 = routed_layer_extents(IkExpertFmt::W8, 2304, 9, 128).unwrap();
        assert_eq!((w8.blk, w8.nblk, w8.inner_cols), (32, 72, 2304));
        assert_eq!(w8.code_dims, [256, 2304, 9, 128]);
        assert_eq!(w8.scale_dims, [256, 72, 9, 128]);
        assert_eq!(w8.code_bytes, 256 * 2304 * 9 * 128);
        assert_eq!(w8.scale_bytes, 256 * 72 * 9 * 128 * 2);

        assert!(routed_layer_extents(IkExpertFmt::W8, 0, 9, 128).is_err());
        assert!(routed_layer_extents(IkExpertFmt::W4, 0, 9, 128).is_err());
        assert!(routed_layer_extents(IkExpertFmt::W8, 16, 9, 128).is_err());
        assert!(routed_layer_extents(IkExpertFmt::W4, 8, 9, 128).is_err());
        assert!(routed_layer_extents(IkExpertFmt::W4, 2304, 0, 128).is_err());
        assert!(routed_layer_extents(IkExpertFmt::W8, 2304, 9, 0).is_err());
    }

    #[test]
    fn routed_src_row_bijects_all_256_lanes_per_tile() {
        for is_gu in [false, true] {
            for tt in [0u64, 3u64] {
                let mut seen = std::collections::BTreeSet::new();
                for lane in 0..256u64 {
                    let row = routed_src_row(is_gu, tt, lane);
                    assert!(row >= tt * 256 && row < (tt + 1) * 256);
                    assert!(seen.insert(row));
                }
                assert_eq!(seen.len(), 256);
            }
        }
    }
}
