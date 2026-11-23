//! A GGUF of the Qwen3.5 / 3.8 family (`general.architecture` `qwen35` or `qwen35moe`) under
//! the checkpoint names the Qwen packers read (`model.language_model.layers.<l>....`,
//! `lm_head.weight`, `mtp.*`).
//!
//! What the GGUF converter does to the checkpoint tensors (read from the converter for facts),
//! and what the views here undo:
//! - names: `token_embd` / `output` / `output_norm` / `blk.<l>.<role>`; the MTP block is
//!   appended as `blk.<n_layers>` with `nextn.*` for the MTP fusion and its norms;
//! - RMSNorm weights are stored as `1 + w` (every norm but the gated DeltaNet norm): the view
//!   subtracts 1 in `f32` (exact whenever `|w| >= 2^-16`);
//! - `A_log` is stored as `-exp(A_log)`: the view takes `ln(-x)` in `f64`;
//! - the conv1d `[C, 1, K]` is stored squeezed (`{K, C}`);
//! - with more value heads than key heads, the value heads are stored tiled `(r, k)` instead of
//!   grouped `(k, r)` in the V rows of `in_proj_qkv`, the rows of `in_proj_z` / `in_proj_a` /
//!   `in_proj_b`, the V channels of conv1d, `A_log`, `dt_bias` and the columns of `out_proj`
//!   (index map: [`crate::verified::gg_vhead_src`]);
//! - routed experts are stacked (`{H, F, E}`): one view per expert selects its rows.
//!
//! Not done here: a ModelSpec from the GGUF metadata, a tokenizer from `tokenizer.ggml.*`, an
//! MTP head from a separate file, and the packers' reads of lazy tensors; until those exist the
//! Qwen engine does not start from a `.gguf` (see the pull request of issue #20).

use std::sync::Arc;

use super::adapter::{register, Registered, ViewSpec};
use super::{err, Gguf, GgufError, TensorInfo};
use crate::safetensors::SafetensorsStore;
use crate::verified as v;

const P: &str = "model.language_model.";

/// Dimensions the map needs from the metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35Dims {
    /// Trunk layers (block_count minus the MTP blocks).
    pub n_layers: u64,
    pub mtp_layers: u64,
    pub hidden: u64,
    pub k_heads: u64,
    pub v_heads: u64,
    pub k_dim: u64,
    pub v_dim: u64,
    /// Leading q and k entries of the fused qkv projection (2 x key heads x key dim).
    pub qk: u64,
}

pub fn is_family(g: &Gguf) -> bool {
    matches!(g.arch(), "qwen35" | "qwen35moe")
}

/// Reads and checks the dimensions.
pub fn dims(g: &Gguf) -> Result<Qwen35Dims, GgufError> {
    let src = g.source.display().to_string();
    if !is_family(g) {
        return Err(err(
            &src,
            format!("architecture {:?} is not qwen35 / qwen35moe", g.arch()),
        ));
    }
    let u = |k: &str| g.arch_kv(k).and_then(|x| x.as_u64()).unwrap_or(0);
    let blocks = u("block_count");
    let mtp = u("nextn_predict_layers");
    let (hk, hv, dk, inner) = (
        u("ssm.group_count"),
        u("ssm.time_step_rank"),
        u("ssm.state_size"),
        u("ssm.inner_size"),
    );
    let hidden = u("embedding_length");
    // at least one trunk layer; value heads a whole multiple of key heads; inner = value heads x value dim
    if !v::gg_span_ok(mtp, 1, blocks)
        || hidden == 0
        || dk == 0
        || !v::gg_blocks_ok(hv, hk)
        || !v::gg_blocks_ok(inner, hv)
    {
        return Err(err(
            &src,
            format!("block_count {blocks} / nextn_predict_layers {mtp} / embedding_length {hidden} / ssm.* ({hk}, {hv}, {dk}, {inner}) missing or inconsistent"),
        ));
    }
    let qk = hk
        .checked_mul(dk)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| {
            err(
                &src,
                format!("ssm.group_count {hk} x ssm.state_size {dk} overflows"),
            )
        })?;
    Ok(Qwen35Dims {
        n_layers: blocks - mtp,
        mtp_layers: mtp,
        hidden,
        k_heads: hk,
        v_heads: hv,
        k_dim: dk,
        v_dim: inner / hv,
        qk,
    })
}

/// Checkpoint index -> stored index over `n` entries: `base` leading entries in place, then the
/// value heads (`d` values each) from tiled to grouped order. Refused unless it permutes `0..n`.
fn vhead_map(n: u64, base: u64, d: &Qwen35Dims, head: u64, name: &str) -> Result<Vec<u64>, String> {
    let r = d.v_heads / d.k_heads;
    let mut seen = vec![false; n as usize];
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let s = v::gg_vhead_src(i, base, d.k_heads, r, head).filter(|&s| s < n);
        match s {
            Some(s) if !seen[s as usize] => {
                seen[s as usize] = true;
                out.push(s);
            }
            _ => {
                return Err(format!(
                    "{name}: ssm.* metadata does not match its {n} entries"
                ))
            }
        }
    }
    Ok(out)
}

fn minus_one() -> super::adapter::ValueFn {
    Arc::new(|r: &mut [f32]| r.iter_mut().for_each(|x| *x -= 1.0))
}

fn neg_log() -> super::adapter::ValueFn {
    Arc::new(|r: &mut [f32]| r.iter_mut().for_each(|x| *x = (-(*x as f64)).ln() as f32))
}

/// The checkpoint tensors one GGUF tensor becomes (empty: not part of this family's checkpoint).
pub fn map_tensor(
    g: &Gguf,
    d: &Qwen35Dims,
    t: &TensorInfo,
) -> Result<Vec<(String, ViewSpec)>, GgufError> {
    let src = g.files[t.shard].path.display().to_string();
    let bad = |m: String| err(&src, format!("tensor {} {:?}: {m}", t.name, t.dims));
    let n = t.name.as_str();
    let mat = |name: String| -> Result<Vec<(String, ViewSpec)>, GgufError> {
        if t.dims.len() != 2 {
            return Err(bad("expected a matrix".into()));
        }
        Ok(vec![(name, ViewSpec::whole())])
    };
    let vec1 = |name: String, spec: ViewSpec| -> Result<Vec<(String, ViewSpec)>, GgufError> {
        if t.dims.len() != 1 {
            return Err(bad("expected a vector".into()));
        }
        Ok(vec![(name, spec)])
    };
    match n {
        "token_embd.weight" => return mat(format!("{P}embed_tokens.weight")),
        "output.weight" => return mat("lm_head.weight".into()),
        "output_norm.weight" => {
            return vec1(
                format!("{P}norm.weight"),
                ViewSpec::whole().map_values(minus_one()),
            )
        }
        _ => {}
    }
    let Some(rest) = n.strip_prefix("blk.") else {
        return Ok(Vec::new());
    };
    let Some((b, role)) = rest.split_once('.') else {
        return Ok(Vec::new());
    };
    let Ok(b) = b.parse::<u64>() else {
        return Ok(Vec::new());
    };
    let blocks = d.n_layers + d.mtp_layers;
    if b >= blocks {
        return Ok(Vec::new());
    }
    let l = if b < d.n_layers {
        format!("{P}layers.{b}.")
    } else {
        format!("mtp.layers.{}.", b - d.n_layers)
    };
    let tiled = d.v_heads > d.k_heads;
    let qk = d.qk;
    let rows = t.rows();
    let rmap = |base: u64, head: u64| -> Result<Option<Arc<[u64]>>, GgufError> {
        if !tiled {
            return Ok(None);
        }
        Ok(Some(
            vhead_map(rows, base, d, head, n).map_err(bad)?.into(),
        ))
    };
    let cmap = |base: u64, head: u64| -> Result<Option<Arc<[u32]>>, GgufError> {
        if !tiled {
            return Ok(None);
        }
        let m = vhead_map(t.row_len(), base, d, head, n).map_err(bad)?;
        Ok(Some(
            m.into_iter().map(|x| x as u32).collect::<Vec<_>>().into(),
        ))
    };
    let norm = |name: String| vec1(name, ViewSpec::whole().map_values(minus_one()));
    let plain = |name: &str| mat(format!("{l}{name}"));
    let with_rows =
        |name: &str, m: Option<Arc<[u64]>>| -> Result<Vec<(String, ViewSpec)>, GgufError> {
            if t.dims.len() != 2 {
                return Err(bad("expected a matrix".into()));
            }
            Ok(vec![(
                format!("{l}{name}"),
                ViewSpec {
                    row_map: m,
                    ..ViewSpec::whole()
                },
            )])
        };
    Ok(match role {
        "attn_norm.weight" => norm(format!("{l}input_layernorm.weight"))?,
        "post_attention_norm.weight" => norm(format!("{l}post_attention_layernorm.weight"))?,
        "attn_q_norm.weight" => norm(format!("{l}self_attn.q_norm.weight"))?,
        "attn_k_norm.weight" => norm(format!("{l}self_attn.k_norm.weight"))?,
        "ffn_gate.weight" => plain("mlp.gate_proj.weight")?,
        "ffn_up.weight" => plain("mlp.up_proj.weight")?,
        "ffn_down.weight" => plain("mlp.down_proj.weight")?,
        "attn_q.weight" => plain("self_attn.q_proj.weight")?,
        "attn_k.weight" => plain("self_attn.k_proj.weight")?,
        "attn_v.weight" => plain("self_attn.v_proj.weight")?,
        "attn_output.weight" => plain("self_attn.o_proj.weight")?,
        "attn_qkv.weight" => with_rows("linear_attn.in_proj_qkv.weight", rmap(qk, d.v_dim)?)?,
        "attn_gate.weight" => with_rows("linear_attn.in_proj_z.weight", rmap(0, d.v_dim)?)?,
        "ssm_beta.weight" => with_rows("linear_attn.in_proj_b.weight", rmap(0, 1)?)?,
        "ssm_alpha.weight" => with_rows("linear_attn.in_proj_a.weight", rmap(0, 1)?)?,
        "ssm_out.weight" => {
            if t.dims.len() != 2 {
                return Err(bad("expected a matrix".into()));
            }
            vec![(
                format!("{l}linear_attn.out_proj.weight"),
                ViewSpec {
                    col_map: cmap(0, d.v_dim)?,
                    ..ViewSpec::whole()
                },
            )]
        }
        "ssm_a" => vec1(
            format!("{l}linear_attn.A_log"),
            ViewSpec {
                col_map: cmap(0, 1)?,
                ..ViewSpec::whole().map_values(neg_log())
            },
        )?,
        "ssm_dt.bias" => vec1(
            format!("{l}linear_attn.dt_bias"),
            ViewSpec {
                col_map: cmap(0, 1)?,
                ..ViewSpec::whole()
            },
        )?,
        "ssm_norm.weight" => vec1(format!("{l}linear_attn.norm.weight"), ViewSpec::whole())?,
        "ssm_conv1d.weight" => {
            if t.dims.len() != 2 {
                return Err(bad("expected {kernel, channels}".into()));
            }
            let spec = ViewSpec {
                row_map: rmap(qk, d.v_dim)?,
                ..ViewSpec::whole().with_shape(vec![t.dims[1], 1, t.dims[0]])
            };
            vec![(format!("{l}linear_attn.conv1d.weight"), spec)]
        }
        "ffn_gate_inp.weight" => plain("mlp.gate.weight")?,
        "ffn_gate_inp_shexp.weight" => {
            // {H} or {H, 1}: both occur
            if !(t.dims.len() == 1 || (t.dims.len() == 2 && t.dims[1] == 1)) {
                return Err(bad("expected a vector".into()));
            }
            vec![(
                format!("{l}mlp.shared_expert_gate.weight"),
                ViewSpec::whole().with_shape(vec![1, t.dims[0]]),
            )]
        }
        "ffn_gate_shexp.weight" => plain("mlp.shared_expert.gate_proj.weight")?,
        "ffn_up_shexp.weight" => plain("mlp.shared_expert.up_proj.weight")?,
        "ffn_down_shexp.weight" => plain("mlp.shared_expert.down_proj.weight")?,
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            if t.dims.len() != 3 {
                return Err(bad("expected {in, out, experts}".into()));
            }
            let pj = match role {
                "ffn_gate_exps.weight" => "gate_proj",
                "ffn_up_exps.weight" => "up_proj",
                _ => "down_proj",
            };
            (0..t.dims[2])
                .map(|e| {
                    (
                        format!("{l}mlp.experts.{e}.{pj}.weight"),
                        ViewSpec::expert(e, t.dims[1]),
                    )
                })
                .collect()
        }
        "nextn.eh_proj.weight" => mat("mtp.fc.weight".into())?,
        "nextn.enorm.weight" => norm("mtp.pre_fc_norm_embedding.weight".into())?,
        "nextn.hnorm.weight" => norm("mtp.pre_fc_norm_hidden.weight".into())?,
        "nextn.shared_head_norm.weight" => norm("mtp.norm.weight".into())?,
        _ => Vec::new(),
    })
}

/// Registers every tensor of a Qwen3.5-family GGUF under its checkpoint name.
pub fn register_qwen35(
    store: &mut SafetensorsStore,
    g: &Arc<Gguf>,
) -> Result<Registered, GgufError> {
    let d = dims(g)?;
    register(store, g, |g, t| map_tensor(g, &d, t))
}
