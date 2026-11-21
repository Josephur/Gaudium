//! Model weight packer & RoPE table generator (`pack_qwen35.hpp`).
//!
//! Reads Safetensors shards via zero-copy mmap (`SafetensorsStore`), applies Gaudi 2
//! deinterleaved lane permutations verified in `gaudium/verified/tensor_bounds.bend`
//! (`core::shapes::lane128` / `lane256`), and produces raw device-ready byte buffers for H2D upload.

use crate::fp8::bf16_to_f32;
use crate::model_spec::ModelSpec;
use crate::safetensors::{SafetensorsStore, TensorMeta};
use crate::verified::{lane128, lane256};

/// Weight packer for Qwen3 / Qwen3.5 full-attention models (BF16 weights + F32 norms/RoPE).
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
            .ok_or_else(|| format!("weight '{name}' is lazy/unmapped (expected mapped BF16)"))
    }

    /// Reads a 1-D BF16 tensor and converts every element to `f32`.
    fn read_bf16_as_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let bytes = self.raw_bytes(name)?;
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for &chunk in bytes.as_chunks::<2>().0 {
            out.push(bf16_to_f32(u16::from_le_bytes(chunk)));
        }
        Ok(out)
    }

    /// Packs a named device weight tensor into its exact HBM byte representation.
    pub fn pack_weight(&self, name: &str) -> Result<Vec<u8>, String> {
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
                return Ok(self.raw_bytes(&full)?.to_vec());
            }
            "W_lm" => {
                let full = "lm_head.weight";
                let t = self
                    .store
                    .get(full)
                    .map_err(|e| format!("missing safetensors weight '{full}': {e}"))?;
                self.need_bf16(full, t, vocab, Some(hid))?;
                return Ok(self.raw_bytes(full)?.to_vec());
            }
            "w_final" => {
                let (full, t) = self.tensor_ref("norm.weight")?;
                self.need_bf16(&full, t, hid, None)?;
                return Ok(self.raw_bytes(&full)?.to_vec());
            }
            "norm_ones" => {
                let mut buf = Vec::with_capacity((hid as usize) * 2);
                for _ in 0..hid {
                    buf.extend_from_slice(&0x3F80u16.to_le_bytes());
                }
                return Ok(buf);
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

        match what {
            "w_in" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}input_layernorm.weight"))?;
                self.need_bf16(&full, t, hid, None)?;
                Ok(self.raw_bytes(&full)?.to_vec())
            }
            "w_post" => {
                let (full, t) =
                    self.tensor_ref(&format!("{lpfx}post_attention_layernorm.weight"))?;
                self.need_bf16(&full, t, hid, None)?;
                Ok(self.raw_bytes(&full)?.to_vec())
            }
            "W_gate" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}mlp.gate_proj.weight"))?;
                self.need_bf16(&full, t, inter, Some(hid))?;
                Ok(self.raw_bytes(&full)?.to_vec())
            }
            "W_up" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}mlp.up_proj.weight"))?;
                self.need_bf16(&full, t, inter, Some(hid))?;
                Ok(self.raw_bytes(&full)?.to_vec())
            }
            "W_down" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}mlp.down_proj.weight"))?;
                self.need_bf16(&full, t, hid, Some(inter))?;
                Ok(self.raw_bytes(&full)?.to_vec())
            }
            "W_qg" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}self_attn.q_proj.weight"))?;
                self.need_bf16(&full, t, qg, Some(hid))?;
                let src = self.raw_bytes(&full)?;
                if !m.attn_gate {
                    return Ok(src.to_vec());
                }
                let row_bytes = (hid as usize) * 2;
                let head_bytes = (dh as usize) * row_bytes;
                let mut dst = vec![0u8; src.len()];
                for h in 0..(nh as usize) {
                    let src_q = (2 * h) * head_bytes;
                    let src_g = (2 * h + 1) * head_bytes;
                    let dst_q = h * head_bytes;
                    let dst_g = ((nh as usize) + h) * head_bytes;
                    dst[dst_q..dst_q + head_bytes].copy_from_slice(&src[src_q..src_q + head_bytes]);
                    dst[dst_g..dst_g + head_bytes].copy_from_slice(&src[src_g..src_g + head_bytes]);
                }
                Ok(dst)
            }
            "W_k" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}self_attn.k_proj.weight"))?;
                self.need_bf16(&full, t, kvd, Some(hid))?;
                Ok(self.raw_bytes(&full)?.to_vec())
            }
            "W_v" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}self_attn.v_proj.weight"))?;
                self.need_bf16(&full, t, kvd, Some(hid))?;
                let src = self.raw_bytes(&full)?;
                if !self.chunked_attn {
                    return Ok(src.to_vec());
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
                Ok(dst)
            }
            "qn_w" | "kn_w" => {
                let sub = if what == "qn_w" {
                    "self_attn.q_norm.weight"
                } else {
                    "self_attn.k_norm.weight"
                };
                let (full, t) = self.tensor_ref(&format!("{lpfx}{sub}"))?;
                self.need_bf16(&full, t, dh, None)?;
                let w = self.read_bf16_as_f32(&full)?;
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
                let mut out = Vec::with_capacity(perm.len() * 4);
                for &f in &perm {
                    out.extend_from_slice(&f.to_le_bytes());
                }
                Ok(out)
            }
            "W_o" => {
                let (full, t) = self.tensor_ref(&format!("{lpfx}self_attn.o_proj.weight"))?;
                self.need_bf16(&full, t, hid, Some(nh * dh))?;
                Ok(self.raw_bytes(&full)?.to_vec())
            }
            _ => Err(format!("unsupported layer weight '{name}'")),
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

        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        };
        (to_bytes(&c), to_bytes(&s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_qwen3_8b_pos0_and_pos1() {
        let cfg = r#"{
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
        let spec = ModelSpec::from_config_json(cfg).unwrap();
        let store = SafetensorsStore::empty();
        let packer = WeightPacker::new(&store, &spec);
        let (rc, rs) = packer.build_rope_tables(2);
        assert_eq!(rc.len(), 2 * 128 * 4);
        assert_eq!(rs.len(), 2 * 128 * 4);
        let read_f32 = |b: &[u8], i: usize| {
            f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
        };
        // pos=0: cos=1, sin=0 across all 128 lanes
        for lane in 0..128 {
            assert_eq!(read_f32(&rc, lane), 1.0);
            assert_eq!(read_f32(&rs, lane), 0.0);
        }
        // pos=1, j=0 (freq=1.0): cos(1) at lanes 0 & 32; -sin(1) at lane 0, +sin(1) at lane 32
        assert!((read_f32(&rc, 128) - 1.0f32.cos()).abs() < 1e-6);
        assert!((read_f32(&rc, 128 + 32) - 1.0f32.cos()).abs() < 1e-6);
        assert!((read_f32(&rs, 128) + 1.0f32.sin()).abs() < 1e-6);
        assert!((read_f32(&rs, 128 + 32) - 1.0f32.sin()).abs() < 1e-6);
    }
}
