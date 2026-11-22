//! Gemma 4 weight packer and RoPE table generator.
//!
//! Produces the exact HBM byte image of every device tensor the Gemma 4 graphs
//! (`engine::gemma4`) declare, straight from the zero-copy mmap'd Safetensors shards:
//!
//! - `W_qg` / `W_k` / `W_v`: projection rows stored in **lane order** inside each 128-row block
//!   (`core::gemma4::lane_in_block`), so every GEMM output arrives in the order the attention
//!   front-end kernel loads it and no shuffle runs on the card.
//! - `qn_w` / `kn_w`: per-head norm weights as f32 tables in the kernels' deinterleaved view.
//! - RoPE cos/sin tables `{128 * rv, Lmax}` per attention kind (sliding: every pair rotated;
//!   global: "proportional", the leading pairs only, the rest `cos 1 / sin 0`).
//! - Every other matrix is borrowed unchanged from the checkpoint (`Cow::Borrowed`).

use std::borrow::Cow;

use crate::core::gemma4::{deinterleave_channel_table, lane_in_block, natural_channel};
use crate::engine::gemma4::G4Dims;
use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};
use crate::model_spec::ModelSpec;
use crate::safetensors::{SafetensorsStore, TensorMeta};

/// Weight packer for the Gemma 4 family (E2B / E4B / 12B / 31B text decoders, BF16 checkpoints).
pub struct Gemma4Packer<'a> {
    store: &'a SafetensorsStore,
    spec: &'a ModelSpec,
    dims: &'a G4Dims,
}

fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bf16_scalar(v: f32) -> Vec<u8> {
    f32_to_bf16_rne(v).to_le_bytes().to_vec()
}

/// Destination row of projection row `r` (lane order inside each 128-row block).
pub fn lane_order_row(r: u64) -> u64 {
    (r / 128) * 128 + lane_in_block(r % 128)
}

/// Builds the `(cos, sin)` f32 tables `{128 * rv, lmax}` of one attention kind in the kernels'
/// view: element `128 b + 64 h + j` of a row holds channel pair `128 b + nat(h, j)`, which turns
/// by `pos * theta^(-2 i / head_dim)`; pairs at or beyond `pairs` are not rotated.
pub fn build_g4_rope_tables(
    head_dim: u64,
    pairs: u64,
    theta: f64,
    lmax: usize,
) -> (Vec<u8>, Vec<u8>) {
    let rv = pairs.div_ceil(128) as usize;
    let width = 128 * rv;
    // Per-lane frequency (`None`: an unrotated lane inside a rotated vector).
    let mut lane_freq: Vec<Option<f64>> = vec![None; width];
    for b in 0..rv {
        for h in 0..2u64 {
            for j in 0..64u64 {
                let i = 128 * (b as u64) + natural_channel(h, j);
                if i < pairs {
                    lane_freq[128 * b + (64 * h + j) as usize] =
                        Some(theta.powf(-2.0 * (i as f64) / (head_dim as f64)));
                }
            }
        }
    }
    let mut c = vec![1.0f32; width * lmax];
    let mut s = vec![0.0f32; width * lmax];
    for pos in 0..lmax {
        let p = pos as f64;
        let base = pos * width;
        for (lane, f) in lane_freq.iter().enumerate() {
            if let Some(freq) = f {
                let a = p * freq;
                c[base + lane] = a.cos() as f32;
                s[base + lane] = a.sin() as f32;
            }
        }
    }
    (f32_slice_to_bytes(&c), f32_slice_to_bytes(&s))
}

impl<'a> Gemma4Packer<'a> {
    pub fn new(store: &'a SafetensorsStore, spec: &'a ModelSpec, dims: &'a G4Dims) -> Self {
        Self { store, spec, dims }
    }

    fn tensor_ref(&self, suffix: &str) -> Result<(String, &'a TensorMeta), String> {
        let full = format!("{}{}", self.spec.text_prefix, suffix);
        let meta = self
            .store
            .get(&full)
            .map_err(|e| format!("missing safetensors weight '{full}': {e}"))?;
        Ok((full, meta))
    }

    fn raw_bytes(&self, name: &str) -> Result<&'a [u8], String> {
        self.store
            .tensor_bytes(name)
            .map_err(|e| format!("read '{name}': {e}"))?
            .ok_or_else(|| format!("weight '{name}' is lazy/unmapped (expected mapped tensor)"))
    }

    /// Borrows a BF16 tensor of exactly `shape` unchanged from the mmap.
    fn borrow_bf16(&self, suffix: &str, shape: &[u64]) -> Result<Cow<'a, [u8]>, String> {
        let (full, t) = self.tensor_ref(suffix)?;
        if t.dtype != "BF16" {
            return Err(format!(
                "weight '{full}' has dtype {}, expected BF16",
                t.dtype
            ));
        }
        if t.shape != shape {
            return Err(format!(
                "weight '{full}' has shape {:?}, expected {:?}",
                t.shape, shape
            ));
        }
        Ok(Cow::Borrowed(self.raw_bytes(&full)?))
    }

    /// Borrows a BF16 vector by element count (accepts `[n]` and scalar-shaped `[]` / `[1]`).
    fn borrow_bf16_elems(&self, suffix: &str, elems: u64) -> Result<Cow<'a, [u8]>, String> {
        let (full, t) = self.tensor_ref(suffix)?;
        if t.dtype != "BF16" || t.nbytes as u64 != elems * 2 {
            return Err(format!(
                "weight '{full}' has dtype {} and {} bytes, expected BF16 with {} elements",
                t.dtype, t.nbytes, elems
            ));
        }
        Ok(Cow::Borrowed(self.raw_bytes(&full)?))
    }

    /// Projection `{rows, hid}` with its rows moved to lane order inside each 128-row block.
    fn lane_order_rows(&self, suffix: &str, rows: u64) -> Result<Cow<'a, [u8]>, String> {
        let hid = self.dims.hid;
        let src = self.borrow_bf16(suffix, &[rows, hid])?;
        let row_bytes = (hid as usize) * 2;
        let mut dst = vec![0u8; src.len()];
        for r in 0..rows {
            let s_off = (r as usize) * row_bytes;
            let d_off = (lane_order_row(r) as usize) * row_bytes;
            dst[d_off..d_off + row_bytes].copy_from_slice(&src[s_off..s_off + row_bytes]);
        }
        Ok(Cow::Owned(dst))
    }

    /// Per-head norm weight `{dh}` as an f32 table in the kernels' deinterleaved view.
    fn head_norm_table(&self, suffix: &str, dh: u64) -> Result<Cow<'a, [u8]>, String> {
        let src = self.borrow_bf16_elems(suffix, dh)?;
        let mut w = Vec::with_capacity(dh as usize);
        for &ch in src.as_chunks::<2>().0 {
            w.push(bf16_to_f32(u16::from_le_bytes(ch)));
        }
        Ok(Cow::Owned(f32_slice_to_bytes(&deinterleave_channel_table(
            &w,
        ))))
    }

    /// RoPE tables of the global (`full_attention`) layers: `rope_c` / `rope_s`.
    pub fn global_rope_tables(&self, lmax: usize) -> (Vec<u8>, Vec<u8>) {
        let d = self.dims;
        build_g4_rope_tables(d.g_dh, d.g_pairs, d.g_theta, lmax)
    }

    /// RoPE tables of the sliding layers: `rope_sc` / `rope_ss`.
    pub fn sliding_rope_tables(&self, lmax: usize) -> (Vec<u8>, Vec<u8>) {
        let d = self.dims;
        build_g4_rope_tables(d.s_dh, d.s_pairs, d.s_theta, lmax)
    }

    /// Packs a named device weight tensor into its exact HBM byte representation.
    pub fn pack_weight(&self, name: &str) -> Result<Cow<'a, [u8]>, String> {
        let d = self.dims;
        let hid = d.hid;
        let vocab = d.vocab;
        let pw = (d.n_layers as u64) * d.ple;

        match name {
            "embed_table" => return self.borrow_bf16("embed_tokens.weight", &[vocab, hid]),
            "embed_scale" => return Ok(Cow::Owned(bf16_scalar((hid as f32).sqrt()))),
            "w_final" => return self.borrow_bf16_elems("norm.weight", hid),
            _ => {}
        }
        if d.ple > 0 {
            match name {
                "ple_table" => {
                    return self.borrow_bf16("embed_tokens_per_layer.weight", &[vocab, pw])
                }
                "ple_scale" => return Ok(Cow::Owned(bf16_scalar((d.ple as f32).sqrt()))),
                "W_plm" => {
                    return self.borrow_bf16("per_layer_model_projection.weight", &[pw, hid])
                }
                "plm_scale" => return Ok(Cow::Owned(bf16_scalar((hid as f64).powf(-0.5) as f32))),
                "ple_mix" => return Ok(Cow::Owned(bf16_scalar(2.0f64.powf(-0.5) as f32))),
                "w_plm" => {
                    return self.borrow_bf16_elems("per_layer_projection_norm.weight", d.ple)
                }
                _ => {}
            }
        }

        let rest = name
            .strip_prefix('L')
            .ok_or_else(|| format!("unknown Gemma 4 weight tensor '{name}'"))?;
        let sep = rest
            .find('_')
            .ok_or_else(|| format!("malformed layer weight tensor '{name}'"))?;
        let l: usize = rest[..sep]
            .parse()
            .map_err(|_| format!("invalid layer index in '{name}'"))?;
        if l >= d.n_layers {
            return Err(format!("layer index out of range in '{name}'"));
        }
        let what = &rest[sep + 1..];
        let lp = format!("layers.{l}.");
        let ap = format!("{lp}self_attn.");
        let glob = d.is_global(l);
        let shared = d.is_shared(l);
        let (nkv, dh) = d.heads(glob);
        let nh = d.nh;
        let inter = d.layer_inter(l);

        match what {
            "w_in" => self.borrow_bf16_elems(&format!("{lp}input_layernorm.weight"), hid),
            "w_pa" => self.borrow_bf16_elems(&format!("{lp}post_attention_layernorm.weight"), hid),
            "w_post" => {
                self.borrow_bf16_elems(&format!("{lp}pre_feedforward_layernorm.weight"), hid)
            }
            "w_pf" => {
                self.borrow_bf16_elems(&format!("{lp}post_feedforward_layernorm.weight"), hid)
            }
            "ls" => self.borrow_bf16_elems(&format!("{lp}layer_scalar"), 1),
            "W_gate" => self.borrow_bf16(&format!("{lp}mlp.gate_proj.weight"), &[inter, hid]),
            "W_up" => self.borrow_bf16(&format!("{lp}mlp.up_proj.weight"), &[inter, hid]),
            "W_down" => self.borrow_bf16(&format!("{lp}mlp.down_proj.weight"), &[hid, inter]),
            "W_o" => self.borrow_bf16(&format!("{ap}o_proj.weight"), &[hid, nh * dh]),
            "W_qg" => self.lane_order_rows(&format!("{ap}q_proj.weight"), nh * dh),
            "qn_w" => self.head_norm_table(&format!("{ap}q_norm.weight"), dh),
            "W_k" if !shared => self.lane_order_rows(&format!("{ap}k_proj.weight"), nkv * dh),
            "W_v" if !shared && !(glob && d.keqv) => {
                self.lane_order_rows(&format!("{ap}v_proj.weight"), nkv * dh)
            }
            "kn_w" if !shared => self.head_norm_table(&format!("{ap}k_norm.weight"), dh),
            "W_pg" if d.ple > 0 => {
                self.borrow_bf16(&format!("{lp}per_layer_input_gate.weight"), &[d.ple, hid])
            }
            "W_pp" if d.ple > 0 => {
                self.borrow_bf16(&format!("{lp}per_layer_projection.weight"), &[hid, d.ple])
            }
            "w_pl" if d.ple > 0 => {
                self.borrow_bf16_elems(&format!("{lp}post_per_layer_input_norm.weight"), hid)
            }
            _ => Err(format!("unsupported Gemma 4 weight tensor '{name}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rd(b: &[u8], i: usize) -> f32 {
        f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
    }

    #[test]
    fn lane_order_rows_are_a_permutation_of_each_block() {
        let mut seen = vec![false; 512];
        for r in 0..512u64 {
            let d = lane_order_row(r);
            assert_eq!(d / 128, r / 128, "a row stays inside its 128-row block");
            assert!(!seen[d as usize]);
            seen[d as usize] = true;
        }
        // even channels fill the first half of a block, odd channels the second
        assert_eq!(lane_order_row(0), 0);
        assert_eq!(lane_order_row(2), 1);
        assert_eq!(lane_order_row(1), 64);
        assert_eq!(lane_order_row(129), 128 + 64);
    }

    #[test]
    fn sliding_rope_rotates_every_pair_and_global_only_the_leading_quarter() {
        // sliding: head 256, 128 pairs, theta 1e4
        let (c, s) = build_g4_rope_tables(256, 128, 1.0e4, 3);
        assert_eq!(c.len(), 128 * 3 * 4);
        for lane in 0..128 {
            assert_eq!(rd(&c, lane), 1.0);
            assert_eq!(rd(&s, lane), 0.0);
        }
        // position 1, lane (h = 0, j = 0) is channel pair 0: angle 1 rad
        assert!((rd(&c, 128) - 1.0f32.cos()).abs() < 1e-6);
        assert!((rd(&s, 128) - 1.0f32.sin()).abs() < 1e-6);
        // lane (h = 1, j = 0) is channel pair nat(1, 0) = 2: frequency 1e4^(-4 / 256)
        let f2 = 1.0e4f64.powf(-4.0 / 256.0);
        assert!((rd(&c, 128 + 64) - (f2.cos() as f32)).abs() < 1e-6);

        // global: head 512, int(0.25 * 512 / 2) = 64 rotated pairs, theta 1e6; divisor is the head dim
        let (gc, gs) = build_g4_rope_tables(512, 64, 1.0e6, 2);
        assert_eq!(gc.len(), 128 * 2 * 4);
        let mut rotated = 0;
        for lane in 0..128usize {
            let h = (lane / 64) as u64;
            let j = (lane % 64) as u64;
            let i = natural_channel(h, j);
            let (cv, sv) = (rd(&gc, 128 + lane), rd(&gs, 128 + lane));
            if i < 64 {
                let f = 1.0e6f64.powf(-2.0 * (i as f64) / 512.0);
                assert!((cv - (f.cos() as f32)).abs() < 1e-6);
                assert!((sv - (f.sin() as f32)).abs() < 1e-6);
                rotated += 1;
            } else {
                assert_eq!(cv, 1.0);
                assert_eq!(sv, 0.0);
            }
        }
        assert_eq!(rotated, 64);
    }
}
