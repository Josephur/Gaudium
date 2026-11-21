//! Model weight packer & RoPE table generator (`pack_qwen35.hpp`).
//!
//! Reads Safetensors shards via zero-copy mmap (`SafetensorsStore`), applies Gaudi 2
//! deinterleaved lane permutations verified in `gaudium/verified/tensor_bounds.bend`
//! (`verified::lane128` / `verified::lane256`), and produces device-ready byte slices
//! (`Cow::Borrowed` directly from mmap for un-permuted matrices to avoid multi-GiB host copies).

use std::borrow::Cow;

use crate::fp8::bf16_to_f32;
use crate::model_spec::ModelSpec;
use crate::safetensors::{SafetensorsStore, TensorMeta};
use crate::verified::{lane128, lane256};

/// Weight packer for Qwen3 & Qwen3.5/3.8 hybrid (Gated DeltaNet + Full Attention) BF16 models.
pub struct WeightPacker<'a> {
    store: &'a SafetensorsStore,
    spec: &'a ModelSpec,
    /// Permute `W_v` rows per KV head via `lane128`/`lane256` for Gaudium's chunked TPC attention kernels.
    pub chunked_attn: bool,
}

impl<'a> WeightPacker<'a> {
    pub fn new(store: &'a SafetensorsStore, spec: &'a ModelSpec) -> Self {
        Self {
            store,
            spec,
            chunked_attn: true,
        }
    }

    fn tensor_ref(&self, suffix: &str) -> Result<(String, &'a TensorMeta), String> {
        let full = format!("{}{}", self.spec.text_prefix, suffix);
        let meta = self
            .store
            .get(&full)
            .map_err(|e| format!("missing safetensors weight '{full}': {e}"))?;
        Ok((full, meta))
    }

    fn need_bf16(
        &self,
        name: &str,
        t: &TensorMeta,
        d0: u64,
        d1: Option<u64>,
    ) -> Result<(), String> {
        if t.dtype != "BF16" {
            return Err(format!(
                "weight '{name}' has dtype {}, expected BF16",
                t.dtype
            ));
        }
        let expected: Vec<u64> = match d1 {
            Some(c) => vec![d0, c],
            None => vec![d0],
        };
        if t.shape != expected {
            return Err(format!(
                "weight '{name}' has shape {:?}, expected {:?}",
                t.shape, expected
            ));
        }
        Ok(())
    }

    fn raw_bytes(&self, name: &str) -> Result<&'a [u8], String> {
        self.store
            .tensor_bytes(name)
            .map_err(|e| format!("read '{name}': {e}"))?
            .ok_or_else(|| format!("weight '{name}' is lazy/unmapped (expected mapped tensor)"))
    }

    /// Reads a 1-D BF16 or F32 tensor and converts every element to `f32`.
    fn read_1d_as_f32(
        &self,
        name: &str,
        t: &TensorMeta,
        expected_len: u64,
    ) -> Result<Vec<f32>, String> {
        if t.shape != [expected_len] {
            return Err(format!(
                "weight '{name}' has shape {:?}, expected [{expected_len}]",
                t.shape
            ));
        }
        let bytes = self.raw_bytes(name)?;
        match t.dtype.as_str() {
            "BF16" => {
                let mut out = Vec::with_capacity(expected_len as usize);
                for &chunk in bytes.as_chunks::<2>().0 {
                    out.push(bf16_to_f32(u16::from_le_bytes(chunk)));
                }
                Ok(out)
            }
            "F32" => {
                let mut out = Vec::with_capacity(expected_len as usize);
                for &chunk in bytes.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(chunk));
                }
                Ok(out)
            }
            other => Err(format!(
                "weight '{name}' has dtype {other}, expected BF16 or F32"
            )),
        }
    }

    /// Packs a named device weight tensor into its exact HBM byte representation.
    /// Returns `Cow::Borrowed` directly from the mmap shard whenever no layout transform is needed.
    pub fn pack_weight(&self, name: &str) -> Result<Cow<'a, [u8]>, String> {
        let m = self.spec;
        let hid = m.hidden;
        let inter = m.intermediate;
        let vocab = m.vocab;
        let nh = m.n_heads;
        let nkv = m.n_kv_heads;
        let dh = m.head_dim;
        let qg = m.qg_dim();
        let kvd = m.kv_dim();

        match name {
            "embed_table" => {
                let (full, t) = self.tensor_ref("embed_tokens.weight")?;
                self.need_bf16(&full, t, vocab, Some(hid))?;
                return Ok(Cow::Borrowed(self.raw_bytes(&full)?));
            }
            "W_lm" => {
                let full = "lm_head.weight";
                let t = self
                    .store
                    .get(full)
                    .map_err(|e| format!("missing safetensors weight '{full}': {e}"))?;
                self.need_bf16(full, t, vocab, Some(hid))?;
                return Ok(Cow::Borrowed(self.raw_bytes(full)?));
            }
            "w_final" => {
                let (full, t) = self.tensor_ref("norm.weight")?;
                self.need_bf16(&full, t, hid, None)?;
                return Ok(Cow::Borrowed(self.raw_bytes(&full)?));
            }
            "norm_ones" => {
                let mut buf = Vec::with_capacity((hid as usize) * 2);
                for _ in 0..hid {
                    buf.extend_from_slice(&0x3F80u16.to_le_bytes());
                }
                return Ok(Cow::Owned(buf));
            }
            "zero_slot" => {
                return Ok(Cow::Owned(vec![0u8; 4]));
            }
            _ => {}
        }

        // Parse layer-prefixed name `L{idx}_{what}`
        let rest = name
            .strip_prefix('L')
            .ok_or_else(|| format!("unknown weight tensor '{name}'"))?;
        let sep = rest
            .find('_')
            .ok_or_else(|| format!("malformed layer weight tensor '{name}'"))?;
        let layer_idx: usize = rest[..sep]
            .parse()
            .map_err(|_| format!("invalid layer index in '{name}'"))?;
        let what = &rest[sep + 1..];
        let lpfx = format!("layers.{layer_idx}.");
        let is_attn = m.is_full_attention_layer(layer_idx);

        // Shared norm + dense MLP weights
        match what {
            "w_in" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}input_layernorm.weight"))?;
                self.need_bf16(&full, t, hid, None)?;
                return Ok(Cow::Borrowed(self.raw_bytes(&full)?));
            }
            "w_post" => {
                let (full, t) =
                    self.tensor_ref(&format!("{lpfx}post_attention_layernorm.weight"))?;
                self.need_bf16(&full, t, hid, None)?;
                return Ok(Cow::Borrowed(self.raw_bytes(&full)?));
            }
            "W_gate" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}mlp.gate_proj.weight"))?;
                self.need_bf16(&full, t, inter, Some(hid))?;
                return Ok(Cow::Borrowed(self.raw_bytes(&full)?));
            }
            "W_up" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}mlp.up_proj.weight"))?;
                self.need_bf16(&full, t, inter, Some(hid))?;
                return Ok(Cow::Borrowed(self.raw_bytes(&full)?));
            }
            "W_down" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}mlp.down_proj.weight"))?;
                self.need_bf16(&full, t, hid, Some(inter))?;
                return Ok(Cow::Borrowed(self.raw_bytes(&full)?));
            }
            _ => {}
        }

        if is_attn {
            match what {
                "W_qg" => {
                    let (full, t) = self.tensor_ref(&format!("{lpfx}self_attn.q_proj.weight"))?;
                    self.need_bf16(&full, t, qg, Some(hid))?;
                    let src = self.raw_bytes(&full)?;
                    if !m.attn_gate {
                        return Ok(Cow::Borrowed(src));
                    }
                    // Gated Qwen3.5: de-interleave per-head [Q_h | G_h] into [Q_0..Q_{NH-1} | G_0..G_{NH-1}]
                    let row_bytes = (hid as usize) * 2;
                    let head_bytes = (dh as usize) * row_bytes;
                    let mut dst = vec![0u8; src.len()];
                    for h in 0..(nh as usize) {
                        let src_q = (2 * h) * head_bytes;
                        let src_g = (2 * h + 1) * head_bytes;
                        let dst_q = h * head_bytes;
                        let dst_g = ((nh as usize) + h) * head_bytes;
                        dst[dst_q..dst_q + head_bytes]
                            .copy_from_slice(&src[src_q..src_q + head_bytes]);
                        dst[dst_g..dst_g + head_bytes]
                            .copy_from_slice(&src[src_g..src_g + head_bytes]);
                    }
                    Ok(Cow::Owned(dst))
                }
                "W_k" => {
                    let (full, t) = self.tensor_ref(&format!("{lpfx}self_attn.k_proj.weight"))?;
                    self.need_bf16(&full, t, kvd, Some(hid))?;
                    Ok(Cow::Borrowed(self.raw_bytes(&full)?))
                }
                "W_v" => {
                    let (full, t) = self.tensor_ref(&format!("{lpfx}self_attn.v_proj.weight"))?;
                    self.need_bf16(&full, t, kvd, Some(hid))?;
                    let src = self.raw_bytes(&full)?;
                    if !self.chunked_attn {
                        return Ok(Cow::Borrowed(src));
                    }
                    let row_bytes = (hid as usize) * 2;
                    let mut dst = vec![0u8; src.len()];
                    for g in 0..(nkv as usize) {
                        for c in 0..(dh as usize) {
                            let dst_c = if dh == 128 {
                                lane128(c as u64) as usize
                            } else {
                                lane256(c as u64) as usize
                            };
                            let s_off = (g * (dh as usize) + c) * row_bytes;
                            let d_off = (g * (dh as usize) + dst_c) * row_bytes;
                            dst[d_off..d_off + row_bytes]
                                .copy_from_slice(&src[s_off..s_off + row_bytes]);
                        }
                    }
                    Ok(Cow::Owned(dst))
                }
                "qn_w" | "kn_w" => {
                    let sub = if what == "qn_w" {
                        "self_attn.q_norm.weight"
                    } else {
                        "self_attn.k_norm.weight"
                    };
                    let (full, t) = self.tensor_ref(&format!("{lpfx}{sub}"))?;
                    let w = self.read_1d_as_f32(&full, t, dh)?;
                    let offset = if m.norm_plus1 { 1.0f32 } else { 0.0f32 };
                    let mut perm = vec![0.0f32; dh as usize];
                    for (i, &val) in w.iter().enumerate().take(dh as usize) {
                        let dst_i = if dh == 128 {
                            lane128(i as u64) as usize
                        } else {
                            lane256(i as u64) as usize
                        };
                        perm[dst_i] = offset + val;
                    }
                    Ok(Cow::Owned(f32_slice_to_bytes(&perm)))
                }
                "W_o" => {
                    let (full, t) = self.tensor_ref(&format!("{lpfx}self_attn.o_proj.weight"))?;
                    self.need_bf16(&full, t, hid, Some(nh * dh))?;
                    Ok(Cow::Borrowed(self.raw_bytes(&full)?))
                }
                _ => Err(format!("unsupported full-attention weight '{name}'")),
            }
        } else {
            // Gated DeltaNet (`linear_attn.*`) weights
            let hv = m.gdn_v_heads;
            let dl = m.gdn_v_dim;
            let qkv = m.gdn_qkv_dim();
            let qkvz = m.gdn_qkvz_dim();
            let hvp = m.gdn_hvp();
            let ba = m.gdn_ba_dim();
            let row_bytes = (hid as usize) * 2;

            match what {
                "W_qkvz" => {
                    let (fa, ta) =
                        self.tensor_ref(&format!("{lpfx}linear_attn.in_proj_qkv.weight"))?;
                    self.need_bf16(&fa, ta, qkv, Some(hid))?;
                    let (fz, tz) =
                        self.tensor_ref(&format!("{lpfx}linear_attn.in_proj_z.weight"))?;
                    self.need_bf16(&fz, tz, hv * dl, Some(hid))?;
                    let a_bytes = self.raw_bytes(&fa)?;
                    let z_bytes = self.raw_bytes(&fz)?;
                    let mut dst = Vec::with_capacity((qkvz as usize) * row_bytes);
                    dst.extend_from_slice(a_bytes);
                    dst.extend_from_slice(z_bytes);
                    Ok(Cow::Owned(dst))
                }
                "W_ba" => {
                    let (fb, tb) =
                        self.tensor_ref(&format!("{lpfx}linear_attn.in_proj_b.weight"))?;
                    self.need_bf16(&fb, tb, hv, Some(hid))?;
                    let (fa, ta) =
                        self.tensor_ref(&format!("{lpfx}linear_attn.in_proj_a.weight"))?;
                    self.need_bf16(&fa, ta, hv, Some(hid))?;
                    let b_bytes = self.raw_bytes(&fb)?;
                    let a_bytes = self.raw_bytes(&fa)?;
                    let mut dst = vec![0u8; (ba as usize) * row_bytes];
                    for j in 0..(hv as usize) {
                        let b_off = j * row_bytes;
                        let a_off = j * row_bytes;
                        let d_b = (2 * j) * row_bytes;
                        let d_a = (2 * j + 1) * row_bytes;
                        dst[d_b..d_b + row_bytes]
                            .copy_from_slice(&b_bytes[b_off..b_off + row_bytes]);
                        dst[d_a..d_a + row_bytes]
                            .copy_from_slice(&a_bytes[a_off..a_off + row_bytes]);
                    }
                    Ok(Cow::Owned(dst))
                }
                "conv_w" => {
                    let (fc, tc) = self.tensor_ref(&format!("{lpfx}linear_attn.conv1d.weight"))?;
                    if tc.dtype != "BF16" || tc.shape != [qkv, 1, 4] {
                        return Err(format!(
                            "weight '{fc}' has dtype {} shape {:?}, expected BF16 [{qkv}, 1, 4]",
                            tc.dtype, tc.shape
                        ));
                    }
                    let src = self.raw_bytes(&fc)?;
                    let mut f32_out = vec![0.0f32; (qkv as usize) * 4];
                    let elems = src.as_chunks::<2>().0;
                    for c in 0..(qkv as usize) {
                        let cd = (c / 128) * 128 + (lane128((c % 128) as u64) as usize);
                        for tap in 0..4usize {
                            let bits = u16::from_le_bytes(elems[c * 4 + tap]);
                            f32_out[tap * (qkv as usize) + cd] = bf16_to_f32(bits);
                        }
                    }
                    Ok(Cow::Owned(f32_slice_to_bytes(&f32_out)))
                }
                "conv_b" => Ok(Cow::Owned(vec![0u8; (qkv as usize) * 4])),
                "A_log" => {
                    let (fl, tl) = self.tensor_ref(&format!("{lpfx}linear_attn.A_log"))?;
                    let mut v = self.read_1d_as_f32(&fl, tl, hv)?;
                    v.resize(hvp as usize, 0.0f32);
                    Ok(Cow::Owned(f32_slice_to_bytes(&v)))
                }
                "dt_bias" => {
                    let (fd, td) = self.tensor_ref(&format!("{lpfx}linear_attn.dt_bias"))?;
                    let mut v = self.read_1d_as_f32(&fd, td, hv)?;
                    v.resize(hvp as usize, 0.0f32);
                    Ok(Cow::Owned(f32_slice_to_bytes(&v)))
                }
                "norm_w" => {
                    let (fnw, tnw) = self.tensor_ref(&format!("{lpfx}linear_attn.norm.weight"))?;
                    let w = self.read_1d_as_f32(&fnw, tnw, dl)?;
                    let mut perm = vec![0.0f32; dl as usize];
                    for (i, &val) in w.iter().enumerate().take(dl as usize) {
                        perm[lane128(i as u64) as usize] = val;
                    }
                    Ok(Cow::Owned(f32_slice_to_bytes(&perm)))
                }
                "W_o" => {
                    let (fo, to) =
                        self.tensor_ref(&format!("{lpfx}linear_attn.out_proj.weight"))?;
                    self.need_bf16(&fo, to, hid, Some(hv * dl))?;
                    Ok(Cow::Borrowed(self.raw_bytes(&fo)?))
                }
                _ => Err(format!("unsupported linear-attention weight '{name}'")),
            }
        }
    }

    /// Builds `{128, lmax}` F32 `rope_c` and `rope_s` tables matching `attn_prep_gaudi2.c`.
    pub fn build_rope_tables(&self, lmax: usize) -> (Vec<u8>, Vec<u8>) {
        let theta = self.spec.rope_theta;
        let rot = self.spec.rotary_dim as usize;
        let np = rot / 4;

        let mut c = vec![1.0f32; 128 * lmax];
        let mut s = vec![0.0f32; 128 * lmax];

        for pos in 0..lmax {
            let base = pos * 128;
            let p = pos as f64;
            for j in 0..np {
                let fe = theta.powf(-2.0 * (2 * j) as f64 / rot as f64);
                let fo = theta.powf(-2.0 * (2 * j + 1) as f64 / rot as f64);
                let ce = (p * fe).cos() as f32;
                let se = (p * fe).sin() as f32;
                let co = (p * fo).cos() as f32;
                let so = (p * fo).sin() as f32;

                c[base + j] = ce;
                c[base + np + j] = ce;
                s[base + j] = -se;
                s[base + np + j] = se;

                c[base + 64 + j] = co;
                c[base + 64 + np + j] = co;
                s[base + 64 + j] = -so;
                s[base + 64 + np + j] = so;
            }
        }

        (f32_slice_to_bytes(&c), f32_slice_to_bytes(&s))
    }
}

fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_qwen3_8b_and_qwen38_27b() {
        let cfg8 = r#"{
          "model_type": "qwen3",
          "hidden_size": 4096,
          "intermediate_size": 12288,
          "num_hidden_layers": 36,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "vocab_size": 151936,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "tie_word_embeddings": false
        }"#;
        let spec8 = ModelSpec::from_config_json(cfg8).unwrap();
        let store = SafetensorsStore::empty();
        let packer8 = WeightPacker::new(&store, &spec8);
        let (rc, rs) = packer8.build_rope_tables(2);
        let read_f32 = |b: &[u8], i: usize| {
            f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
        };
        for lane in 0..128 {
            assert_eq!(read_f32(&rc, lane), 1.0);
            assert_eq!(read_f32(&rs, lane), 0.0);
        }
        assert!((read_f32(&rc, 128) - 1.0f32.cos()).abs() < 1e-6);
        assert!((read_f32(&rc, 128 + 32) - 1.0f32.cos()).abs() < 1e-6);
        assert!((read_f32(&rs, 128) + 1.0f32.sin()).abs() < 1e-6);
        assert!((read_f32(&rs, 128 + 32) - 1.0f32.sin()).abs() < 1e-6);

        // Qwen3.8-27B partial RoPE (rotary_dim=64 -> NP=16; lanes [32..64] and [96..128] stay 1.0 / 0.0)
        let cfg27 = r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
          }
        }"#;
        let spec27 = ModelSpec::from_config_json(cfg27).unwrap();
        assert_eq!(spec27.rotary_dim, 64);
        assert_eq!(spec27.gdn_qkv_dim(), 10240);
        assert_eq!(spec27.gdn_qkvz_dim(), 16384);
        assert_eq!(spec27.gdn_ba_dim(), 128);
        let packer27 = WeightPacker::new(&store, &spec27);
        let (rc27, rs27) = packer27.build_rope_tables(2);
        assert!((read_f32(&rc27, 128) - 1.0f32.cos()).abs() < 1e-6);
        assert!((read_f32(&rc27, 128 + 16) - 1.0f32.cos()).abs() < 1e-6);
        assert_eq!(read_f32(&rc27, 128 + 32), 1.0);
        assert_eq!(read_f32(&rs27, 128 + 32), 0.0);
    }
}
