//! Inkling 2 from a GGUF file: the model's geometry from the file's metadata and its tensors
//! presented to the weight packer under the publisher's names, shapes and element order
//! ([`IkSource`]), so a GGUF build and the publisher's checkpoint go through one packer.
//!
//! How the GGUF form relates to the publisher's (established from the files themselves and the
//! public converter's description of its transforms):
//! - 2-D projections keep their bytes (`[out, in]` row-major); small tensors are stored as F32.
//! - The fused gate / up expert matrix is stored de-interleaved (`[E, 2 I, C]`: the `I` gate rows,
//!   then the `I` up rows); the publisher interleaves them (row `2 i` gate, `2 i + 1` up).
//! - Expert matrices are 4-bit blocks with a per-expert scale tensor (`<name>.scale`), or BF16.
//! - Convolution taps are `[C, taps]` with the last tap on the row itself for every layer kind
//!   (the publisher stores the global layers' static taps tap-major, first tap on the row itself).
//! - The relative-position projection is stored transposed (`[extent, d_rel]`).
//! - The message queries of the looped layers are one `[4, C]` tensor per layer in the order
//!   query, key, value, MLP (the publisher stores two tensors of interleaved pairs).

use std::borrow::Cow;
use std::sync::Arc;

use super::packer::{ExpertRows, IkSource};
use crate::fp8::f32_to_bf16_rne;
use crate::gguf::adapter::companion_scale;
use crate::gguf::dequant::{rows_bf16, rows_f32};
use crate::gguf::{GgmlType, Gguf, TensorInfo};

/// Whether a GGUF file holds a model of this family.
pub fn is_family(g: &Gguf) -> bool {
    g.arch().starts_with("inkling2")
}

/// The `config.json` text equivalent to a GGUF file's metadata (the fields `ModelSpec` reads).
pub fn config_json(g: &Gguf) -> Result<String, String> {
    let arch = g.arch().to_string();
    if !is_family(g) {
        return Err(format!("GGUF architecture '{arch}' is not an Inkling 2 model"));
    }
    let u = |k: &str| -> Result<u64, String> {
        g.arch_kv(k)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("GGUF metadata key {arch}.{k} is missing or not a number"))
    };
    let f = |k: &str| -> Result<f64, String> {
        g.arch_kv(k)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("GGUF metadata key {arch}.{k} is missing or not a number"))
    };
    // per-layer arrays (one value per layer) or one value for every layer
    let first = |k: &str| -> Result<u64, String> {
        let v = g
            .arch_kv(k)
            .ok_or_else(|| format!("GGUF metadata key {arch}.{k} is missing"))?;
        v.as_u64()
            .or_else(|| v.as_array().and_then(|a| a.get(0)).and_then(|x| x.as_u64()))
            .ok_or_else(|| format!("GGUF metadata key {arch}.{k} is not a number or a list of numbers"))
    };
    let n_layers = u("block_count")?;
    let pattern = g
        .arch_kv("attention.sliding_window_pattern")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("GGUF metadata key {arch}.attention.sliding_window_pattern is missing"))?;
    if pattern.len() != n_layers {
        return Err(format!(
            "GGUF metadata: {} sliding-window flags for {n_layers} layers",
            pattern.len()
        ));
    }
    let mut local = Vec::new();
    for l in 0..n_layers {
        let sliding = pattern
            .get(l)
            .and_then(|v| v.as_bool())
            .ok_or_else(|| "GGUF metadata: a sliding-window flag is not a boolean".to_string())?;
        if sliding {
            local.push(l.to_string());
        }
    }
    let looped = g.arch_kv("loop.count").is_some();
    let loop_cfg = if looped {
        format!(
            r#","loop_config":{{"core_start":{},"core_end":{},"num_passes":{},"gate_rank":{}}}"#,
            u("loop.start")?,
            u("loop.end")?,
            u("loop.count")?,
            u("loop.gate_rank")?
        )
    } else {
        String::new()
    };
    let eos = g
        .get_u64("tokenizer.ggml.eos_token_id")
        .ok_or("GGUF metadata key tokenizer.ggml.eos_token_id is missing")?;
    Ok(format!(
        r#"{{"architectures":["{arch}"],"model_type":"{arch}","eos_token_id":{eos},
"text_config":{{"model_type":"{arch}_text","model_max_length":{ctx},"hidden_size":{hid},"num_hidden_layers":{n_layers},
"vocab_size":{vocab},"unpadded_vocab_size":{uvocab},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"head_dim":{dh},
"d_rel":{d_rel},"rel_extent":{ext},"log_scaling_n_floor":{lfloor},"log_scaling_alpha":{lalpha},"rms_norm_eps":{eps},
"qk_l2_clip_threshold":{qk},"rel_l2_clip_threshold":{rel},"local_layer_ids":[{local}],"sconv_kernel_size":{taps},
"final_logit_softcapping":{cap},"sliding_window_size":{win},"n_layers_per_attn_res_block":{block},
"n_routed_experts":{ne},"num_experts_per_tok":{ek},"intermediate_size":{ef},"mlp_gate_softcap":{gcap},
"mlp_up_softcap":{ucap},"route_scale":{rscale}{loop_cfg}}}}}"#,
        ctx = u("context_length")?,
        hid = u("embedding_length")?,
        vocab = u("vocab_size")?,
        uvocab = u("vocab_size_unpadded")?,
        nh = first("attention.head_count")?,
        nkv = first("attention.head_count_kv")?,
        dh = u("attention.key_length")?,
        d_rel = u("attention.relative_length")?,
        ext = u("attention.relative_extent")?,
        lfloor = u("attention.log_scale_floor")?,
        lalpha = f("attention.log_scale_alpha")?,
        eps = f("attention.layer_norm_rms_epsilon")?,
        qk = f("attention.qk_clip")?,
        rel = f("attention.relative_clip")?,
        local = local.join(","),
        taps = u("shortconv.l_cache")?,
        cap = f("final_logit_softcapping")?,
        win = u("attention.sliding_window")?,
        block = u("attn_res.block_size")?,
        ne = u("expert_count")?,
        ek = u("expert_used_count")?,
        ef = u("expert_feed_forward_length")?,
        gcap = f("activation.situ_beta")?,
        ucap = f("activation.situ_linear_beta")?,
        rscale = f("expert_weights_scale")?,
    ))
}

/// A GGUF file as a packer source.
pub struct GgufSource {
    g: Arc<Gguf>,
    loop_start: u64,
    taps: u64,
}

/// The GGUF tensor name of a publisher tensor that keeps its shape and element order.
fn direct_name(suffix: &str) -> Option<String> {
    Some(match suffix {
        "embed.weight" => "token_embd.weight".into(),
        "embed_norm.weight" => "token_embd_norm.weight".into(),
        "unembed.weight" => "output.weight".into(),
        "norm.weight" => "output_norm.weight".into(),
        "attn_res.output_query" => "output_res_query.weight".into(),
        _ => {
            if let Some(i) = suffix.strip_prefix("attn_res.attn_queries.") {
                // the list starts at the second layer
                return Some(format!("blk.{}.attn_res_query.weight", i.parse::<u64>().ok()? + 1));
            }
            if let Some(l) = suffix.strip_prefix("attn_res.mlp_queries.") {
                return Some(format!("blk.{l}.ffn_res_query.weight"));
            }
            let (l, what) = suffix.strip_prefix("layers.")?.split_once('.')?;
            let tail = match what {
                "attn.wq_du.weight" => "attn_q.weight",
                "attn.wk_dv.weight" => "attn_k.weight",
                "attn.wv_dv.weight" => "attn_v.weight",
                "attn.wo_ud.weight" => "attn_output.weight",
                "attn.gate_attn.weight" => "attn_gate.weight",
                "attn.wr_du.weight" => "attn_rel.weight",
                "attn.attn_sink_proj.weight" => "attn_sink_proj.weight",
                "attn.q_l2_clip.scale" => "attn_q_clip.weight",
                "attn.rel_l2_clip.scale" => "attn_rel_clip.weight",
                "attn.k_sconv.weight" => "attn_k_conv.weight",
                "attn.v_sconv.weight" => "attn_v_conv.weight",
                "attn.k_sconv.weight_proj" => "attn_k_conv_proj.weight",
                "attn.v_sconv.weight_proj" => "attn_v_conv_proj.weight",
                "attn_norm.weight" => "attn_norm.weight",
                "q_input_norm.weight" => "attn_q_inp_norm.weight",
                "k_input_norm.weight" => "attn_k_inp_norm.weight",
                "v_input_norm.weight" => "attn_v_inp_norm.weight",
                "attn_sconv.weight" => "attn_post_conv.weight",
                "mlp_sconv.weight" => "ffn_post_conv.weight",
                "mlp_norm.weight" => "ffn_norm.weight",
                "mlp.gate.weight" => "ffn_gate_inp.weight",
                "mlp.gate.bias" => "exp_probs_b.bias",
                _ => {
                    let rest = what.strip_prefix("controllers.")?;
                    let (kind, w) = rest.split_once('.')?;
                    let k = match kind {
                        "attn" => "q",
                        "k" => "k",
                        "v" => "v",
                        "mlp" => "ffn",
                        _ => return None,
                    };
                    let w = match w {
                        "down_weight" => "down",
                        "forget_weight" => "forget",
                        "gain_weight" => "gain",
                        _ => return None,
                    };
                    return Some(format!("blk.{l}.loop_{k}_ctrl_{w}.weight"));
                }
            };
            format!("blk.{l}.{tail}")
        }
    })
}

impl GgufSource {
    pub fn new(g: Arc<Gguf>) -> Result<Self, String> {
        let arch = g.arch().to_string();
        let u = |k: &str| g.arch_kv(k).and_then(|v| v.as_u64());
        Ok(Self {
            loop_start: u("loop.start").unwrap_or(0),
            taps: u("shortconv.l_cache")
                .ok_or_else(|| format!("GGUF metadata key {arch}.shortconv.l_cache is missing"))?,
            g,
        })
    }

    fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.g.get(name).map_err(|e| e.to_string())
    }

    /// Every value of a (small) tensor as f32, row-major.
    fn all_f32(&self, name: &str, elems: u64) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?;
        if t.n_elems != elems {
            return Err(format!(
                "GGUF tensor '{name}' has {} values ({:?}), expected {elems}",
                t.n_elems, t.dims
            ));
        }
        let mut out = vec![0f32; elems as usize];
        rows_f32(&self.g, t, 0, t.rows(), &mut out).map_err(|e| e.to_string())?;
        Ok(out)
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&f32_to_bf16_rne(x).to_le_bytes());
    }
    out
}

impl IkSource for GgufSource {
    fn bf16(&self, suffix: &str, shape: &[u64]) -> Result<Cow<'_, [u8]>, String> {
        let elems: u64 = shape.iter().product();
        // the relative-position projection is stored transposed
        if let Some(l) = suffix
            .strip_prefix("layers.")
            .and_then(|r| r.strip_suffix(".attn.rel_logits_proj.proj"))
        {
            let [d_rel, ext] = shape else {
                return Err(format!("'{suffix}': a relative projection is 2-D, not {shape:?}"));
            };
            let name = format!("blk.{l}.attn_rel_bias.weight");
            let t = self.tensor(&name)?;
            if t.shape_row_major() != [*ext, *d_rel] {
                return Err(format!(
                    "GGUF tensor '{name}' has shape {:?}, expected {:?}",
                    t.shape_row_major(),
                    [ext, d_rel]
                ));
            }
            let v = self.all_f32(&name, elems)?;
            let (d, e) = (*d_rel as usize, *ext as usize);
            let mut tr = vec![0f32; v.len()];
            for x in 0..e {
                for c in 0..d {
                    tr[c * e + x] = v[x * d + c];
                }
            }
            return Ok(Cow::Owned(bf16_bytes(&tr)));
        }
        let name = direct_name(suffix)
            .ok_or_else(|| format!("no GGUF tensor is known for checkpoint tensor '{suffix}'"))?;
        let t = self.tensor(&name)?;
        if t.shape_row_major() != shape {
            return Err(format!(
                "GGUF tensor '{name}' has shape {:?}, expected {shape:?}",
                t.shape_row_major()
            ));
        }
        if t.ty == GgmlType::BF16 {
            return Ok(Cow::Borrowed(self.g.tensor_data(t)));
        }
        Ok(Cow::Owned(bf16_bytes(&self.all_f32(&name, elems)?)))
    }

    fn f32s(&self, suffix: &str, elems: u64) -> Result<Vec<f32>, String> {
        // the message queries: two publisher tensors of interleaved pairs from one tensor per layer
        let kv = suffix == "middle_cycle_cross_pass_kv_queries";
        if kv || suffix == "middle_cycle_cross_pass_queries" {
            let hid = self.tensor("token_embd_norm.weight")?.n_elems;
            if hid == 0 || !elems.is_multiple_of(2 * hid) {
                return Err(format!("'{suffix}': {elems} values are not pairs of rows of {hid}"));
            }
            let span = elems / (2 * hid);
            let h = hid as usize;
            let mut out = Vec::with_capacity(elems as usize);
            for i in 0..span {
                let name = format!("blk.{}.loop_msg_query.weight", self.loop_start + i);
                let q = self.all_f32(&name, 4 * hid)?;
                // rows: query, key, value, MLP
                let (a, b) = if kv { (1, 2) } else { (0, 3) };
                out.extend_from_slice(&q[a * h..(a + 1) * h]);
                out.extend_from_slice(&q[b * h..(b + 1) * h]);
            }
            return Ok(out);
        }
        // a global layer's static taps: tap-major with the first tap on the row itself
        if let Some((l, kvn)) = suffix
            .strip_prefix("layers.")
            .and_then(|r| r.strip_suffix("_sconv.static_w"))
            .and_then(|r| r.split_once(".attn."))
        {
            let name = format!("blk.{l}.attn_{kvn}_conv.weight");
            let v = self.all_f32(&name, elems)?;
            let taps = self.taps as usize;
            let c = v.len() / taps;
            let mut out = vec![0f32; v.len()];
            for lag in 0..taps {
                for ch in 0..c {
                    out[lag * c + ch] = v[ch * taps + (taps - 1 - lag)];
                }
            }
            return Ok(out);
        }
        let name = direct_name(suffix)
            .ok_or_else(|| format!("no GGUF tensor is known for checkpoint tensor '{suffix}'"))?;
        self.all_f32(&name, elems)
    }

    fn experts(
        &self,
        l: usize,
        key: &str,
        ne: u64,
        rows: u64,
        cols: u64,
    ) -> Result<Box<dyn ExpertRows + '_>, String> {
        let (name, fused) = match key {
            "w13" => (format!("blk.{l}.ffn_gate_up_exps.weight"), true),
            "w2" => (format!("blk.{l}.ffn_down_exps.weight"), false),
            other => return Err(format!("unknown expert tensor '{other}'")),
        };
        let t = self.tensor(&name)?;
        if t.shape_row_major() != [ne, rows, cols] {
            return Err(format!(
                "GGUF tensor '{name}' has shape {:?}, expected {:?}",
                t.shape_row_major(),
                [ne, rows, cols]
            ));
        }
        let scale = companion_scale(&self.g, &name).map_err(|e| e.to_string())?;
        if let Some(s) = &scale {
            if s.len() as u64 != ne {
                return Err(format!("GGUF tensor '{name}': {} scales for {ne} experts", s.len()));
            }
        }
        Ok(Box::new(GgufExperts {
            g: &self.g,
            t,
            scale,
            rows,
            fused,
        }))
    }
}

struct GgufExperts<'a> {
    g: &'a Gguf,
    t: &'a TensorInfo,
    scale: Option<Arc<[f32]>>,
    rows: u64,
    /// The gate / up matrix: stored as the gate rows then the up rows.
    fused: bool,
}

impl ExpertRows for GgufExperts<'_> {
    fn decode(&self, e: usize, r: usize, out: &mut [u8]) -> Result<(), String> {
        let r = r as u64;
        let row = if self.fused {
            (r % 2) * (self.rows / 2) + r / 2
        } else {
            r
        };
        let g0 = e as u64 * self.rows + row;
        let mut tmp = vec![0u16; out.len() / 2];
        rows_bf16(self.g, self.t, g0, g0 + 1, self.scale.as_ref().map(|s| s[e]), &mut tmp)
            .map_err(|x| x.to_string())?;
        for (o, v) in out.as_chunks_mut::<2>().0.iter_mut().zip(&tmp) {
            *o = v.to_le_bytes();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::inkling2::packer::{IkPacker, StSource};
    use crate::engine::inkling2::IkDims;
    use crate::model_spec::ModelSpec;
    use crate::safetensors::SafetensorsStore;

    /// The GGUF build and the publisher's checkpoint of one model give the same device tensors.
    /// Needs both files: `GAUDIUM_INKLING2_DIR` (the checkpoint directory) and
    /// `GAUDIUM_INKLING2_GGUF` (the GGUF file).
    #[test]
    #[ignore = "needs GAUDIUM_INKLING2_DIR and GAUDIUM_INKLING2_GGUF (the same model in both forms)"]
    fn gguf_and_checkpoint_pack_the_same_device_tensors() {
        let (Some(dir), Some(file)) = (
            std::env::var_os("GAUDIUM_INKLING2_DIR"),
            std::env::var_os("GAUDIUM_INKLING2_GGUF"),
        ) else {
            panic!("set GAUDIUM_INKLING2_DIR and GAUDIUM_INKLING2_GGUF");
        };
        let g = Arc::new(Gguf::open(std::path::Path::new(&file)).unwrap());
        let spec_g = ModelSpec::from_config_json(&config_json(&g).unwrap()).unwrap();
        let spec_s =
            ModelSpec::from_config_file(std::path::Path::new(&dir).join("config.json")).unwrap();
        // the geometry the engine reads is the same from both
        let (a, b) = (spec_g.inkling2.clone().unwrap(), spec_s.inkling2.clone().unwrap());
        assert_eq!(
            (a.loop_start, a.loop_end, a.passes, a.gate_rank, a.block, a.d_rel, a.rel_extent),
            (b.loop_start, b.loop_end, b.passes, b.gate_rank, b.block, b.d_rel, b.rel_extent)
        );
        assert_eq!(
            (a.window, a.conv_taps, a.log_floor, a.unpadded_vocab),
            (b.window, b.conv_taps, b.log_floor, b.unpadded_vocab)
        );
        assert_eq!(spec_g.layer_types, spec_s.layer_types);
        assert_eq!(
            (spec_g.hidden, spec_g.n_layers, spec_g.n_heads, spec_g.n_kv_heads, spec_g.n_experts),
            (spec_s.hidden, spec_s.n_layers, spec_s.n_heads, spec_s.n_kv_heads, spec_s.n_experts)
        );
        assert_eq!(spec_g.vocab, spec_s.vocab);
        let d = IkDims::from_spec(&spec_s, 1).unwrap();
        let store = SafetensorsStore::open(std::path::Path::new(&dir)).unwrap();
        let st = StSource::new(&store);
        let gg = GgufSource::new(Arc::clone(&g)).unwrap();
        let (ps, pg) = (IkPacker::new(&st, &d, 16384), IkPacker::new(&gg, &d, 16384));
        let mut names: Vec<String> = ["embed", "w_embed", "w_final", "W_head", "q_fin", "tau"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // a sliding layer, a global one, a looped sliding and a looped global one, an 8-bit one
        let per_layer = [
            "w_q", "w_k", "w_v", "w_mlp", "W_qx", "W_sink", "W_kx", "W_vx", "qsc", "rsc", "PROJ",
            "kcw", "vcw", "W_o", "acw", "mcw", "q_attn", "q_mlp", "W_router", "rb", "W_gate",
            "W_up", "W_down",
        ];
        for l in [0usize, 5, 6, 11, 20] {
            for w in per_layer {
                names.push(format!("L{l}_{w}"));
            }
        }
        for l in [6usize, 11] {
            names.push(format!("L{l}_q_msg"));
            for k in ["q", "k", "v", "mlp"] {
                names.push(format!("L{l}_W_gd_{k}"));
                names.push(format!("L{l}_W_gf_{k}"));
            }
        }
        let mut bad = Vec::new();
        for n in &names {
            let (x, y) = (ps.pack(n).unwrap(), pg.pack(n).unwrap());
            if x.len() != y.len() {
                bad.push(format!("{n}: {} vs {} bytes", x.len(), y.len()));
                continue;
            }
            let diff = x
                .as_chunks::<2>()
                .0
                .iter()
                .zip(y.as_chunks::<2>().0)
                .filter(|(p, q)| p != q)
                .count();
            if diff > 0 {
                bad.push(format!("{n}: {diff} of {} 16-bit words differ", x.len() / 2));
            }
        }
        assert!(bad.is_empty(), "device tensors differ:\n{}", bad.join("\n"));
    }

    #[test]
    fn checkpoint_names_map_to_gguf_names() {
        let d = |s: &str| direct_name(s).unwrap();
        assert_eq!(d("embed.weight"), "token_embd.weight");
        assert_eq!(d("attn_res.attn_queries.4"), "blk.5.attn_res_query.weight");
        assert_eq!(d("attn_res.mlp_queries.4"), "blk.4.ffn_res_query.weight");
        assert_eq!(d("layers.7.attn.wq_du.weight"), "blk.7.attn_q.weight");
        assert_eq!(d("layers.11.attn.k_sconv.weight_proj"), "blk.11.attn_k_conv_proj.weight");
        assert_eq!(d("layers.3.mlp.gate.bias"), "blk.3.exp_probs_b.bias");
        assert_eq!(d("layers.9.controllers.attn.down_weight"), "blk.9.loop_q_ctrl_down.weight");
        assert_eq!(d("layers.9.controllers.mlp.gain_weight"), "blk.9.loop_ffn_ctrl_gain.weight");
        assert!(direct_name("layers.9.something_else").is_none());
    }
}
