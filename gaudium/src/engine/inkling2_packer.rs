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

use super::IkDims;
use crate::core::gemma4::{lane_in_block, natural_channel};
use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};
use crate::safetensors::{SafetensorsStore, TensorMeta};
use crate::verified::{ik_bias_src, ik_nib_row_bytes, ik_nib_row_scales};

/// Values of the sixteen 4-bit codes (sign, two exponent bits, one mantissa bit).
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
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

/// A stacked expert tensor of the checkpoint: `ne` experts of `rows` rows of `cols` values.
struct ExpertSrc<'a> {
    name: String,
    w: &'a [u8],
    scale: &'a [u8],
    /// Per-expert factor of the 4-bit form; `None`: the 8-bit float form.
    factor: Option<Vec<f32>>,
    ne: usize,
    rows: usize,
    cols: usize,
}

impl ExpertSrc<'_> {
    fn row_bytes(&self) -> usize {
        if self.factor.is_some() {
            self.cols / 2
        } else {
            self.cols
        }
    }

    fn row_scales(&self) -> usize {
        if self.factor.is_some() {
            self.cols / 16
        } else {
            self.cols / 32
        }
    }

    /// Decodes row `r` of expert `e` into `out` (`cols` BF16 values).
    fn decode(&self, e: usize, r: usize, f8: &[f32; 256], out: &mut [u8]) {
        let (rb, rs) = (self.row_bytes(), self.row_scales());
        let wi = (e * self.rows + r) * rb;
        let si = (e * self.rows + r) * rs;
        match &self.factor {
            Some(f) => decode_nib_row(&self.w[wi..wi + rb], &self.scale[si..si + rs], f[e], f8, out),
            None => decode_f8_row(&self.w[wi..wi + rb], &self.scale[si..si + rs], f8, out),
        }
    }
}

/// Runs `job(first_row, rows_slice)` over `out` split into row ranges on the host's threads.
fn par_rows(out: &mut [u8], row_bytes: usize, job: &(dyn Fn(usize, &mut [u8]) + Sync)) {
    let rows = out.len() / row_bytes;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 32);
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (i, chunk) in out.chunks_mut(per * row_bytes).enumerate() {
            sc.spawn(move || job(i * per, chunk));
        }
    });
}

/// Weight packer for the Inkling 2 family.
pub struct IkPacker<'a> {
    store: &'a SafetensorsStore,
    d: &'a IkDims,
    prefix: &'static str,
    max_ctx: u64,
    f8: [f32; 256],
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
        for h in 0..2 {
            for j in 0..64 {
                out[128 * b + 64 * h + j] = blk[2 * j + h];
            }
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
    pub fn new(store: &'a SafetensorsStore, d: &'a IkDims, max_ctx: u64) -> Self {
        Self {
            store,
            d,
            prefix: "model.llm.",
            max_ctx,
            f8: e4m3_table(),
        }
    }

    fn meta(&self, suffix: &str) -> Result<(String, &'a TensorMeta), String> {
        let full = format!("{}{}", self.prefix, suffix);
        let meta = self
            .store
            .get(&full)
            .map_err(|e| format!("missing checkpoint tensor '{full}': {e}"))?;
        Ok((full, meta))
    }

    fn raw(&self, suffix: &str, dtype: &str, shape: &[u64]) -> Result<&'a [u8], String> {
        let (full, t) = self.meta(suffix)?;
        if t.dtype != dtype || t.shape != shape {
            return Err(format!(
                "checkpoint tensor '{full}' is {} {:?}, expected {dtype} {shape:?}",
                t.dtype, t.shape
            ));
        }
        self.store
            .tensor_bytes(&full)
            .map_err(|e| format!("read '{full}': {e}"))?
            .ok_or_else(|| format!("checkpoint tensor '{full}' is not mapped"))
    }

    /// A small BF16 or F32 tensor of `elems` values as f32.
    fn f32s(&self, suffix: &str, elems: u64) -> Result<Vec<f32>, String> {
        let (full, t) = self.meta(suffix)?;
        let bytes = self
            .store
            .tensor_bytes(&full)
            .map_err(|e| format!("read '{full}': {e}"))?
            .ok_or_else(|| format!("checkpoint tensor '{full}' is not mapped"))?;
        let n: u64 = t.shape.iter().product();
        if n != elems {
            return Err(format!(
                "checkpoint tensor '{full}' has {n} values ({:?}), expected {elems}",
                t.shape
            ));
        }
        match t.dtype.as_str() {
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            other => Err(format!("checkpoint tensor '{full}' has dtype {other}, expected BF16 or F32")),
        }
    }

    /// The stacked expert tensor `layers.<l>.mlp.experts.<key>_weight` in its stored form.
    fn experts(&self, l: usize, key: &str, rows: u64, cols: u64) -> Result<ExpertSrc<'a>, String> {
        let base = format!("layers.{l}.mlp.experts.{key}_weight");
        let (full, t) = self.meta(&base)?;
        let ne = self.d.ne;
        let w = self
            .store
            .tensor_bytes(&full)
            .map_err(|e| format!("read '{full}': {e}"))?
            .ok_or_else(|| format!("checkpoint tensor '{full}' is not mapped"))?;
        let (scale_dtype, factor, row_bytes, row_scales) = match t.dtype.as_str() {
            "U8" => {
                let rb = ik_nib_row_bytes(cols);
                if rb == 0 {
                    return Err(format!(
                        "expert tensor '{full}': {cols} columns are not whole blocks of 16 codes"
                    ));
                }
                let amax = self.f32s(&format!("{base}_amax"), ne)?;
                let den = self.f32s(&format!("{base}_outer_scale_denominator"), 1)?[0];
                let f: Vec<f32> = amax.iter().map(|&a| a / den).collect();
                ("F8_E4M3", Some(f), rb, ik_nib_row_scales(cols))
            }
            "F8_E4M3" => {
                if !cols.is_multiple_of(32) {
                    return Err(format!(
                        "expert tensor '{full}': {cols} columns are not whole blocks of 32 values"
                    ));
                }
                ("U8", None, cols, cols / 32)
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
        Ok(ExpertSrc {
            name: full,
            w,
            scale,
            factor,
            ne: ne as usize,
            rows: rows as usize,
            cols: cols as usize,
        })
    }

    /// The wide gate (`up == false`) or up matrix `{hid, ne ef}`: expert `e`'s channel `i` is
    /// row `2 i` (gate) or `2 i + 1` (up) of its stored gate-up matrix.
    fn pack_gate_up(&self, l: usize, up: bool) -> Result<Vec<u8>, String> {
        let d = self.d;
        let src = self.experts(l, "w13", 2 * d.ef, d.hid)?;
        let (ef, row) = (d.ef as usize, d.hid as usize * 2);
        let mut out = vec![0u8; src.ne * ef * row];
        let f8 = &self.f8;
        par_rows(&mut out, row, &|first, chunk| {
            for (k, o) in chunk.chunks_mut(row).enumerate() {
                let n = first + k;
                src.decode(n / ef, 2 * (n % ef) + usize::from(up), f8, o);
            }
        });
        let _ = &src.name;
        Ok(out)
    }

    /// The wide down matrix `{ne ef, hid}`: output row `n` holds, expert after expert, the
    /// `ef` values of the experts' row `n`.
    fn pack_down(&self, l: usize) -> Result<Vec<u8>, String> {
        let d = self.d;
        let src = self.experts(l, "w2", d.hid, d.ef)?;
        let (ef2, ne) = (d.ef as usize * 2, src.ne);
        let row = ne * ef2;
        let mut out = vec![0u8; d.hid as usize * row];
        let f8 = &self.f8;
        par_rows(&mut out, row, &|first, chunk| {
            for (k, o) in chunk.chunks_mut(row).enumerate() {
                for e in 0..ne {
                    src.decode(e, first + k, f8, &mut o[e * ef2..(e + 1) * ef2]);
                }
            }
        });
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
            "embed" => return Ok(Cow::Borrowed(self.raw("embed.weight", bf, &[d.vocab_in, hid])?)),
            "w_embed" => return Ok(Cow::Borrowed(self.raw("embed_norm.weight", bf, &[hid])?)),
            "w_final" => return Ok(Cow::Borrowed(self.raw("norm.weight", bf, &[hid])?)),
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
                let ns = if d.n_apps == d.apps.len() {
                    d.final_states.len()
                } else {
                    d.wires[d.n_apps - 1].mlp_states.len()
                };
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
                Cow::Borrowed(self.raw(&src, bf, &[hid])?)
            }
            "w_mlp" => Cow::Borrowed(self.raw(&format!("{lp}mlp_norm.weight"), bf, &[hid])?),
            "W_qx" => {
                let q = self.raw(&format!("{at}wq_du.weight"), bf, &[128 * nh, hid])?;
                let g = self.raw(&format!("{at}gate_attn.weight"), bf, &[128 * nh, hid])?;
                let r = self.raw(&format!("{at}wr_du.weight"), bf, &[d.d_rel * nh, hid])?;
                Cow::Owned(self.concat_rows(&[q, g, r], d.qw, hid * 2))
            }
            "W_sink" => Cow::Borrowed(self.raw(&format!("{at}attn_sink_proj.weight"), bf, &[nh, hid])?),
            "W_kx" | "W_vx" => {
                let is_v = what == "W_vx";
                let kv = if is_v { "v" } else { "k" };
                let w = self.raw(&format!("{at}w{kv}_dv.weight"), bf, &[kvw, hid])?;
                let main: Cow<'_, [u8]> = if is_v {
                    Cow::Owned(lane_rows(w, (hid * 2) as usize))
                } else {
                    Cow::Borrowed(w)
                };
                if sliding {
                    match main {
                        Cow::Borrowed(b) => Cow::Borrowed(b),
                        Cow::Owned(v) => Cow::Owned(v),
                    }
                } else {
                    let taps = self.raw(
                        &format!("{at}{kv}_sconv.weight_proj"),
                        bf,
                        &[d.taps * nkv, hid],
                    )?;
                    Cow::Owned(self.concat_rows(&[&main, taps], d.kxw(false), hid * 2))
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
            "W_o" => Cow::Borrowed(self.raw(&format!("{at}wo_ud.weight"), bf, &[hid, 128 * nh])?),
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
            "W_router" => Cow::Borrowed(self.raw(&format!("{lp}mlp.gate.weight"), bf, &[d.ne, hid])?),
            "rb" => {
                let b = self.f32s(&format!("{lp}mlp.gate.bias"), d.ne)?;
                Cow::Owned(f32s_to_bytes(&split_table(&b)))
            }
            "W_gate" => Cow::Owned(self.pack_gate_up(l, false)?),
            "W_up" => Cow::Owned(self.pack_gate_up(l, true)?),
            "W_down" => Cow::Owned(self.pack_down(l)?),
            _ => {
                let ctrl = |which: &str| match which {
                    "q" => Some("attn"),
                    "k" => Some("k"),
                    "v" => Some("v"),
                    "mlp" => Some("mlp"),
                    _ => None,
                };
                if let Some(c) = what.strip_prefix("W_gd_").and_then(ctrl) {
                    Cow::Borrowed(self.raw(
                        &format!("{lp}controllers.{c}.down_weight"),
                        bf,
                        &[d.rank, 2 * hid],
                    )?)
                } else if let Some(c) = what.strip_prefix("W_gf_").and_then(ctrl) {
                    let f = self.raw(&format!("{lp}controllers.{c}.forget_weight"), bf, &[hid, d.rank])?;
                    let g = self.raw(&format!("{lp}controllers.{c}.gain_weight"), bf, &[hid, d.rank])?;
                    let rb = (d.rank * 2) as usize;
                    if d.fg_f32 {
                        // an f32 result in the split layout: the rows in lane order
                        let mut out = lane_rows(f, rb);
                        out.extend(lane_rows(g, rb));
                        Cow::Owned(out)
                    } else {
                        Cow::Owned(self.concat_rows(&[f, g], 2 * hid, d.rank * 2))
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
}
