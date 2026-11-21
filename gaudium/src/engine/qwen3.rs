//! Intel Gaudi 2 execution engine for Qwen3 & Qwen3.5/3.8 dense models
//! (`Qwen3-8B` BF16 full-attention AND `Qwen3.8-27B` BF16 hybrid Gated DeltaNet + full-attention).
//!
//! Architecture ("Verified Core, Rust Shell"):
//! - All HBM capacity checks & 5-category accounting go through `core::hbm` (`HbmLedgerGuard`) and
//!   `AllocGuard` (zero dynamic device allocation after sealing).
//! - All context/token admission and remaining-budget bounds go through `core::budget::SequenceBudget`.
//! - All channel lane permutations (`W_v`, `qn_w`, `kn_w`, `conv_w`, `norm_w`) use Bend-verified
//!   `verified::lane128` / `verified::lane256`.
//! - All Synapse graphs are constructed via `GraphBuilder` (100% fingerprint coverage, normalized
//!   TPC-C ELF hashing, immediate `synGraphDestroy`, pre-validated zero-alloc `PreparedLaunch`).
//!
//! Execution design:
//! - **Prefill**: generic single-layer recipes (`attn_layer` and `linear_layer`, `T <= MAX_PREFILL_TILE`)
//!   reused across all layers by binding per-layer weights, KV caches, and ping-pong GDN states,
//!   followed by a shared `T = 1` readout recipe bound directly to the last prompt token's hidden row.
//! - **Decode**: one monolithic `n_layers`-deep step recipe (`T = 1`) with fused
//!   `gaudium_fused_add_norm` and ping-pong `PreparedLaunch`es (`[parity 0, parity 1]`) —
//!   **one `synLaunch` per token** with zero per-step allocations or state copies.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::core::budget::SequenceBudget;
use crate::engine::packer::WeightPacker;
use crate::kernels::KernelEnv;
use crate::live_report::LiveReporter;
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::safetensors::SafetensorsStore;
use crate::sampler::{sample_row, SampleParams};
use crate::synapse::device::{HbmArena, HpuDevice, PinnedBuf};
use crate::synapse::ffi::{SynDataType, SynapseApi};
use crate::synapse::graph::{CompiledRecipe, GraphBuilder, PreparedLaunch, TensorId};
use crate::synapse::ops::u32_params;
use crate::synapse::recipe_cache::RecipeCache;
use crate::verified::{align_up, HbmCat};

/// KV cache chunk length (matches `CHUNK = 4096` in `attn_prep_gaudi2.c`).
pub const KV_CHUNK: u64 = 4096;
/// Maximum prefill tile width pre-planned into the activation scratch arena.
pub const MAX_PREFILL_TILE: u64 = 256;
/// Minimum pre-planned shared workspace bytes so lazily compiled remainder tiles never need HBM allocs.
const MIN_SHARED_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
/// Reusable host pinned staging buffer size for chunked weight uploads & KV/state zeroing.
const STAGE_BUF_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub max_ctx: u64,
    pub cache_dir: Option<PathBuf>,
    pub no_cache: bool,
    pub prefill_tile: u64,
    pub verbose: bool,
}

impl Qwen3Config {
    pub fn new(model_dir: PathBuf, kernels_dir: PathBuf) -> Self {
        Self {
            model_dir,
            kernels_dir,
            max_ctx: KV_CHUNK,
            cache_dir: None,
            no_cache: false,
            prefill_tile: MAX_PREFILL_TILE,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3GenParams {
    pub max_new: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u64,
    pub min_p: f32,
    pub seed: u64,
    pub stop_on_eos: bool,
    pub extra_eos_ids: Vec<i32>,
}

impl Default for Qwen3GenParams {
    fn default() -> Self {
        Self {
            max_new: 64,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
            stop_on_eos: true,
            extra_eos_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3GenReport {
    pub prompt_tokens: usize,
    pub generated_tokens: Vec<i32>,
    pub ttft_ms: f64,
    pub prefill_tok_s: f64,
    pub decode_ms: f64,
    pub decode_tok_s: f64,
    pub ms_per_step: f64,
    pub stopped_by_eos: bool,
    pub stopped_by_cancel: bool,
}

/// Required kernel ELFs for plain Qwen3 (`DH=128, NKV=8, GQ=4`).
const QWEN3_G8Q4_ELFS: &[&str] = &[
    "GAUDIUM_A128PREP_G8Q4_ELF",
    "GAUDIUM_A128POST_G8Q4_ELF",
    "GAUDIUM_A128COMB_ELF",
    "GAUDIUM_ASOFTC_ELF",
    "GAUDIUM_ANORMP_ELF",
];

/// Required kernel ELFs for hybrid Qwen3.5/3.8 dense (`DH=256, NKV=4, GQ=6, HK=16, HV=48`).
const QWEN35_27B_ELFS: &[&str] = &[
    "GAUDIUM_APREP_ELF",
    "GAUDIUM_APOSTF_ELF",
    "GAUDIUM_ACOMB_ELF",
    "GAUDIUM_ASOFTC_ELF",
    "GAUDIUM_ANORM_ELF",
    "GAUDIUM_PREPS_ELF",
    "GAUDIUM_PREPSD_ELF",
    "GAUDIUM_GATE_ELF",
    "GAUDIUM_GDNS_ELF",
    "GAUDIUM_POST_ELF",
    "GAUDIUM_POSTD_ELF",
];

// ============================================================================
// Pure Graph Builders (host-testable without HPU hardware)
// ============================================================================

/// Helper: plain `Qwen3RMSNorm` (`!norm_plus1`) or zero-centered (`norm_plus1`) using Synapse built-ins.
fn add_rmsnorm(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    w_name: &str,
    y_name: &str,
    t: u64,
) -> TensorId {
    let hid = spec.hidden;
    let eps = spec.rms_eps as f32;
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Bf16, &[hid]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[hid, t]);
    let ir = gb.add_persistent_tensor(format!("{y_name}_ir"), SynDataType::Float, &[1, t]);
    if spec.norm_plus1 {
        gb.op_rms_norm_bf16(format!("{y_name}_norm"), x, w, y, ir, eps, true);
    } else {
        let ones = gb.get_or_add_persistent_tensor("norm_ones", SynDataType::Bf16, &[hid]);
        let yn = gb.add_persistent_tensor(format!("{y_name}_n"), SynDataType::Bf16, &[hid, t]);
        gb.op_rms_norm_bf16(format!("{y_name}_norm"), x, ones, yn, ir, eps, false);
        let w2 = gb.add_intermediate(format!("{y_name}_w2"), SynDataType::Bf16, &[hid, 1]);
        gb.op_reshape(format!("{y_name}_wrs"), w, w2);
        gb.op_mult_bf16(format!("{y_name}_wmul"), yn, w2, y);
    }
    y
}

/// Helper: fused residual add + RMSNorm (`gaudium_fused_add_norm`).
#[allow(clippy::too_many_arguments)]
fn add_fused_add_norm(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    a: TensorId,
    b: TensorId,
    w_name: &str,
    sum_name: &str,
    y_name: &str,
    t: u64,
) -> (TensorId, TensorId) {
    let hid = spec.hidden;
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Bf16, &[hid]);
    let sum = gb.add_persistent_tensor(sum_name, SynDataType::Bf16, &[hid, t]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[hid, t]);
    let params = if spec.norm_plus1 {
        u32_params(&[t as u32])
    } else {
        u32_params(&[t as u32, 1u32])
    };
    gb.add_node(
        "gaudium_fused_add_norm",
        format!("{sum_name}_an"),
        &[a, b, w],
        &[sum, y],
        &params,
    );
    (sum, y)
}

/// Helper: BF16 linear projection `y {N, T} = W {K, N}^T . x {K, T}`.
fn add_linear_bf16(
    gb: &mut GraphBuilder,
    x: TensorId,
    w_name: &str,
    k: u64,
    n: u64,
    t: u64,
    y_name: &str,
) -> TensorId {
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Bf16, &[k, n]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[n, t]);
    gb.op_gemm(format!("{y_name}_gemm"), x, w, y, false, true);
    y
}

/// Shared post-attention residual + norm + SwiGLU MLP + tail residual (+ optional fused next norm).
#[allow(clippy::too_many_arguments)]
fn append_post_and_mlp(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    attn: TensorId,
    lpfx: &str,
    t: u64,
    fused_anorm: bool,
    wnext_norm: Option<(&str, &str)>,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let inter = spec.intermediate;

    let (x2, h2) = if fused_anorm {
        add_fused_add_norm(
            gb,
            spec,
            x,
            attn,
            &format!("{lpfx}w_post"),
            &format!("{lpfx}x2"),
            &format!("{lpfx}h2"),
            t,
        )
    } else {
        let x2 = gb.add_persistent_tensor(format!("{lpfx}x2"), SynDataType::Bf16, &[hid, t]);
        gb.op_add_bf16(format!("{lpfx}x2_add"), x, attn, x2);
        let h2 = add_rmsnorm(
            gb,
            spec,
            x2,
            &format!("{lpfx}w_post"),
            &format!("{lpfx}h2"),
            t,
        );
        (x2, h2)
    };

    let gate = add_linear_bf16(
        gb,
        h2,
        &format!("{lpfx}W_gate"),
        hid,
        inter,
        t,
        &format!("{lpfx}gate"),
    );
    let up = add_linear_bf16(
        gb,
        h2,
        &format!("{lpfx}W_up"),
        hid,
        inter,
        t,
        &format!("{lpfx}up"),
    );
    let act = gb.add_persistent_tensor(format!("{lpfx}act"), SynDataType::Bf16, &[inter, t]);
    gb.op_silu_bf16(format!("{lpfx}silu"), gate, act);
    let m = gb.add_persistent_tensor(format!("{lpfx}m"), SynDataType::Bf16, &[inter, t]);
    gb.op_mult_bf16(format!("{lpfx}mul"), act, up, m);
    let mo = add_linear_bf16(
        gb,
        m,
        &format!("{lpfx}W_down"),
        inter,
        hid,
        t,
        &format!("{lpfx}mo"),
    );

    if fused_anorm {
        if let Some((next_w, next_h)) = wnext_norm {
            let (out, h_next) =
                add_fused_add_norm(gb, spec, x2, mo, next_w, &format!("{lpfx}out"), next_h, t);
            return (out, Some(h_next));
        }
    }
    let out = gb.add_persistent_tensor(format!("{lpfx}out"), SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16(format!("{lpfx}out_add"), x2, mo, out);
    (out, None)
}

/// Appends one full-attention layer (`DH = 128` plain Qwen3 OR `DH = 256` gated Qwen3.5/3.8).
#[allow(clippy::too_many_arguments)]
fn append_attn_layer(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    cursor: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lpfx: &str,
    t: u64,
    nch: u64,
    fused_anorm: bool,
    wnext_norm: Option<(&str, &str)>,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let nh = spec.n_heads;
    let nkv = spec.n_kv_heads;
    let dh = spec.head_dim;
    let qg_dim = spec.qg_dim();
    let kvd = spec.kv_dim();
    let gq = nh / nkv;
    let r = gq * t;
    let b = nkv * nch;

    let g_aprep = if dh == 128 {
        "gaudium_attn128_prep"
    } else {
        "gaudium_attn_prep"
    };
    let g_acomb = if dh == 128 {
        "gaudium_attn128_combine"
    } else {
        "gaudium_attn_combine"
    };
    let g_apost = if dh == 128 {
        "gaudium_attn128_post"
    } else {
        "gaudium_attn_post_f32"
    };

    let h = match h_pre {
        Some(h_in) => h_in,
        None => add_rmsnorm(gb, spec, x, &format!("{lpfx}w_in"), &format!("{lpfx}h"), t),
    };

    let qg = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_qg"),
        hid,
        qg_dim,
        t,
        &format!("{lpfx}qg"),
    );
    let k = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_k"),
        hid,
        kvd,
        t,
        &format!("{lpfx}k"),
    );
    let v = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_v"),
        hid,
        kvd,
        t,
        &format!("{lpfx}v"),
    );

    let qn_w = gb.get_or_add_persistent_tensor(format!("{lpfx}qn_w"), SynDataType::Float, &[dh]);
    let kn_w = gb.get_or_add_persistent_tensor(format!("{lpfx}kn_w"), SynDataType::Float, &[dh]);
    let qb = gb.add_persistent_tensor(format!("{lpfx}qb"), SynDataType::Bf16, &[dh, r, b]);
    let kcache = gb.add_persistent_tensor(
        format!("{lpfx}kcache"),
        SynDataType::Bf16,
        &[dh, KV_CHUNK, b],
    );
    let vcache = gb.add_persistent_tensor(
        format!("{lpfx}vcache"),
        SynDataType::Bf16,
        &[dh, KV_CHUNK, b],
    );
    let prep_params = u32_params(&[0u32]); // coff = 0
    gb.add_node(
        g_aprep,
        format!("{lpfx}attn_prep"),
        &[qg, k, v, qn_w, kn_w, rope_c, rope_s, cursor],
        &[qb, kcache, vcache],
        &prep_params,
    );

    let scores = gb.add_persistent_tensor(
        format!("{lpfx}scores"),
        SynDataType::Float,
        &[KV_CHUNK, r, b],
    );
    gb.op_batch_gemm(format!("{lpfx}qk"), qb, kcache, scores, false, true);

    let pm = gb.add_persistent_tensor(format!("{lpfx}P"), SynDataType::Bf16, &[KV_CHUNK, r, b]);
    let stats =
        gb.add_persistent_tensor(format!("{lpfx}stats"), SynDataType::Float, &[64, 2, r, b]);
    let sm_params = u32_params(&[t as u32, nch as u32, 0u32, 0u32]);
    gb.add_node(
        "gaudium_attn_softmax_chunk",
        format!("{lpfx}softmax"),
        &[scores, cursor],
        &[pm, stats],
        &sm_params,
    );

    let ao_part =
        gb.add_persistent_tensor(format!("{lpfx}ao_part"), SynDataType::Float, &[dh, r, b]);
    gb.op_batch_gemm(format!("{lpfx}pv"), pm, vcache, ao_part, false, false);

    let ao = gb.add_persistent_tensor(format!("{lpfx}ao"), SynDataType::Float, &[dh, r, nkv]);
    let comb_params = u32_params(&[nch as u32]);
    gb.add_node(
        g_acomb,
        format!("{lpfx}combine"),
        &[ao_part, stats],
        &[ao],
        &comb_params,
    );

    let core = gb.add_persistent_tensor(format!("{lpfx}core"), SynDataType::Bf16, &[nh * dh, t]);
    gb.add_node(g_apost, format!("{lpfx}attn_post"), &[ao, qg], &[core], &[]);

    let attn = add_linear_bf16(
        gb,
        core,
        &format!("{lpfx}W_o"),
        nh * dh,
        hid,
        t,
        &format!("{lpfx}attn"),
    );

    append_post_and_mlp(gb, spec, x, attn, lpfx, t, fused_anorm, wnext_norm)
}

/// Appends one Gated DeltaNet (`linear_attention`) layer (`Qwen3.5/3.8`).
#[allow(clippy::too_many_arguments)]
fn append_linear_layer(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    slot: TensorId,
    lpfx: &str,
    t: u64,
    fused_anorm: bool,
    wnext_norm: Option<(&str, &str)>,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let hv = spec.gdn_v_heads;
    let dl = spec.gdn_v_dim;
    let qkv = spec.gdn_qkv_dim();
    let qkvz_dim = spec.gdn_qkvz_dim();
    let hvp = spec.gdn_hvp();
    let ba_dim = spec.gdn_ba_dim();

    let h = match h_pre {
        Some(h_in) => h_in,
        None => add_rmsnorm(gb, spec, x, &format!("{lpfx}w_in"), &format!("{lpfx}h"), t),
    };

    let qkvz = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_qkvz"),
        hid,
        qkvz_dim,
        t,
        &format!("{lpfx}qkvz"),
    );
    let ba = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_ba"),
        hid,
        ba_dim,
        t,
        &format!("{lpfx}ba"),
    );

    let conv_w =
        gb.get_or_add_persistent_tensor(format!("{lpfx}conv_w"), SynDataType::Float, &[qkv, 4]);
    let conv_b =
        gb.get_or_add_persistent_tensor(format!("{lpfx}conv_b"), SynDataType::Float, &[qkv]);
    let cs_in = gb.add_persistent_tensor(format!("{lpfx}cs_in"), SynDataType::Float, &[qkv, 3, 1]);
    let cs_out = gb.add_persistent_tensor(format!("{lpfx}cs_out"), SynDataType::Float, &[qkv, 3]);
    let cs_ckpt =
        gb.add_persistent_tensor(format!("{lpfx}cs_ckpt"), SynDataType::Float, &[qkv, 3, 1]);

    let q = gb.add_persistent_tensor(format!("{lpfx}q"), SynDataType::Float, &[dl, hv, t]);
    let k = gb.add_persistent_tensor(format!("{lpfx}k"), SynDataType::Float, &[dl, hv, t]);
    let v = gb.add_persistent_tensor(format!("{lpfx}v"), SynDataType::Float, &[dl, hv, t]);
    let zero_flag = u32_params(&[0u32]);
    gb.add_node(
        "gaudium_gdn_prep",
        format!("{lpfx}gdn_prep"),
        &[qkvz, conv_w, cs_in, conv_b, slot],
        &[q, k, v, cs_out, cs_ckpt],
        &zero_flag,
    );

    let a_log = gb.get_or_add_persistent_tensor(format!("{lpfx}A_log"), SynDataType::Float, &[hvp]);
    let dt_bias =
        gb.get_or_add_persistent_tensor(format!("{lpfx}dt_bias"), SynDataType::Float, &[hvp]);
    let eg = gb.add_persistent_tensor(format!("{lpfx}eg"), SynDataType::Float, &[hvp, t]);
    let beta = gb.add_persistent_tensor(format!("{lpfx}beta"), SynDataType::Float, &[hvp, t]);
    gb.add_node(
        "gaudium_gdn_gate",
        format!("{lpfx}gdn_gate"),
        &[ba, a_log, dt_bias],
        &[eg, beta],
        &[],
    );

    let s_in =
        gb.add_persistent_tensor(format!("{lpfx}S_in"), SynDataType::Float, &[64, 256, hv, 1]);
    let s_out =
        gb.add_persistent_tensor(format!("{lpfx}S_out"), SynDataType::Float, &[64, 256, hv]);
    let s_ckpt = gb.add_persistent_tensor(
        format!("{lpfx}S_ckpt"),
        SynDataType::Float,
        &[64, 256, hv, 1],
    );
    let o = gb.add_persistent_tensor(format!("{lpfx}o"), SynDataType::Float, &[dl, hv, t]);
    gb.add_node(
        "gaudium_gdn_f32",
        format!("{lpfx}gdn"),
        &[s_in, q, k, v, eg, beta, slot],
        &[o, s_out, s_ckpt],
        &zero_flag,
    );

    let norm_w =
        gb.get_or_add_persistent_tensor(format!("{lpfx}norm_w"), SynDataType::Float, &[dl]);
    let core = gb.add_persistent_tensor(format!("{lpfx}core"), SynDataType::Bf16, &[hv * dl, t]);
    gb.add_node(
        "gaudium_gdn_post",
        format!("{lpfx}gdn_post"),
        &[o, qkvz, norm_w],
        &[core],
        &zero_flag,
    );

    let attn = add_linear_bf16(
        gb,
        core,
        &format!("{lpfx}W_o"),
        hv * dl,
        hid,
        t,
        &format!("{lpfx}attn"),
    );

    append_post_and_mlp(gb, spec, x, attn, lpfx, t, fused_anorm, wnext_norm)
}

/// Builds the embedding gather graph `g_qwen_embed_T{t}`: `ids {T}` -> `x0 {HID, T}`.
pub fn build_embed_graph(spec: &ModelSpec, t: u64) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_qwen_embed_T{t}"));
    let table =
        gb.add_persistent_tensor("embed_table", SynDataType::Bf16, &[spec.hidden, spec.vocab]);
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[t]);
    let x0 = gb.add_persistent_tensor("x0", SynDataType::Bf16, &[spec.hidden, t]);
    gb.op_gather_bf16("embed", table, ids, x0, 1);
    gb
}

/// Builds the shared `T = 1` readout graph `g_qwen_head_T1`.
pub fn build_head_t1_graph(spec: &ModelSpec) -> GraphBuilder {
    let mut gb = GraphBuilder::new("g_qwen_head_T1");
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[spec.hidden, 1]);
    let h_final = add_rmsnorm(&mut gb, spec, x_in, "w_final", "h_final", 1);
    let lm_w = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    let logits = add_linear_bf16(&mut gb, h_final, lm_w, spec.hidden, spec.vocab, 1, "logits");
    let tokens = gb.add_persistent_tensor("tokens", SynDataType::Int32, &[1, 1]);
    gb.op_argmax_bf16("argmax", logits, tokens, 0);
    gb
}

/// Builds the generic single-layer full-attention prefill graph `g_qwen_attn_T{t}_C{nch}`.
pub fn build_prefill_attn_layer_graph(
    spec: &ModelSpec,
    t: u64,
    lmax: u64,
    nch: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_qwen_attn_T{t}_C{nch}"));
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[spec.hidden, t]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let rope_c = gb.add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let _ = append_attn_layer(
        &mut gb, spec, x_in, None, cursor, rope_c, rope_s, "L_", t, nch, false, None,
    );
    gb
}

/// Builds the generic single-layer Gated DeltaNet prefill graph `g_qwen_lin_T{t}`.
pub fn build_prefill_linear_layer_graph(
    spec: &ModelSpec,
    t: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_qwen_lin_T{t}"));
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[spec.hidden, t]);
    let slot = gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]);
    let _ = append_linear_layer(&mut gb, spec, x_in, None, slot, "L_", t, false, None);
    gb
}

/// Builds the monolithic `n_layers`-deep decode step graph `g_qwen_step_C{nch}` (`T = 1`)
/// supporting both pure full-attention (`Qwen3-8B`) and hybrid GDN + attention (`Qwen3.8-27B`).
pub fn build_decode_step_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_qwen_step_C{nch}"));
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }

    let table = gb.get_or_add_persistent_tensor(
        "embed_table",
        SynDataType::Bf16,
        &[spec.hidden, spec.vocab],
    );
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[1]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let slot = if spec.n_linear_layers() > 0 {
        Some(gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]))
    } else {
        None
    };

    let mut x = gb.add_persistent_tensor("x0", SynDataType::Bf16, &[spec.hidden, 1]);
    gb.op_gather_bf16("embed", table, ids, x, 1);

    let mut h_pre: Option<TensorId> = None;
    let n_layers = spec.n_layers as usize;
    for l in 0..n_layers {
        let lpfx = format!("L{l}_");
        let next_w_owned;
        let next_h_owned;
        let wnext = if l + 1 < n_layers {
            next_w_owned = format!("L{}_w_in", l + 1);
            next_h_owned = format!("L{}_h", l + 1);
            Some((next_w_owned.as_str(), next_h_owned.as_str()))
        } else {
            next_w_owned = "w_final".to_string();
            next_h_owned = "h_final".to_string();
            Some((next_w_owned.as_str(), next_h_owned.as_str()))
        };

        let (out_x, next_h) = if spec.is_full_attention_layer(l) {
            append_attn_layer(
                &mut gb, spec, x, h_pre, cursor, rope_c, rope_s, &lpfx, 1, nch, true, wnext,
            )
        } else {
            append_linear_layer(
                &mut gb,
                spec,
                x,
                h_pre,
                slot.expect("zero_slot allocated"),
                &lpfx,
                1,
                true,
                wnext,
            )
        };
        x = out_x;
        h_pre = next_h;
    }

    let h_final = h_pre.expect("fused final norm produced by last layer");
    let lm_w = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    let logits = add_linear_bf16(&mut gb, h_final, lm_w, spec.hidden, spec.vocab, 1, "logits");
    let tokens = gb.add_persistent_tensor("tokens", SynDataType::Int32, &[1, 1]);
    gb.op_argmax_bf16("argmax", logits, tokens, 0);
    gb
}

// ============================================================================
// Live Gaudi 2 Qwen3 / Qwen3.5 / Qwen3.8 Engine
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct GdnLayerState {
    s: [u64; 2],
    s_bytes: u64,
    cs: [u64; 2],
    cs_bytes: u64,
}

/// Live Intel Gaudi 2 execution engine for Qwen3 (`Qwen3-8B`) and Qwen3.5/3.8 (`Qwen3.8-27B`) BF16 models.
///
/// Field declaration order ensures all `CompiledRecipe`s and `PinnedBuf`s are dropped
/// **before** `dev: HpuDevice` (which drains all streams and releases `/dev/accel/accel0`).
pub struct Qwen3Engine {
    spec: ModelSpec,
    max_ctx: u64,
    max_chunks: u64,
    prefill_tile: u64,
    driver_version: String,
    cache: RecipeCache,
    elf_hashes: Vec<(String, String)>,

    // Recipes (must drop before `dev`)
    head_t1: CompiledRecipe,
    step_recipes: Vec<CompiledRecipe>, // index = nch - 1
    embed_recipes: HashMap<u64, CompiledRecipe>,
    attn_layer_recipes: HashMap<(u64, u64), CompiledRecipe>, // (T, nch)
    lin_layer_recipes: HashMap<u64, CompiledRecipe>,         // T

    // Reusable pinned host staging buffers (must drop before `dev`)
    stage_io: PinnedBuf,   // small scalar + logits D2H/H2D
    stage_zero: PinnedBuf, // 64 MiB zero buffer for KV/state reset

    // HBM address maps (all sub-allocated from sealed arenas owned by `dev`)
    weights_addr: BTreeMap<String, u64>,
    kv_addr: BTreeMap<String, (u64, u64)>, // L{l}_kcache / vcache -> (addr, bytes)
    gdn_state: BTreeMap<usize, GdnLayerState>, // linear layer idx -> ping-pong S & cs buffers
    pf_attn_scratch: BTreeMap<String, u64>,
    pf_lin_scratch: BTreeMap<String, u64>,
    pf_x0: u64,
    pf_x1: u64,
    pf_ids: u64,
    pf_cursor: u64,
    head_scratch: BTreeMap<String, u64>,
    step_launches: Vec<[PreparedLaunch; 2]>, // [nch - 1][gdn_parity]
    step_ids: u64,
    step_cursor: u64,
    step_tokens: u64,
    step_logits: u64,
    hbm_bytes: [u64; 5],
    live: LiveReporter,

    // Device handle (dropped LAST!)
    dev: HpuDevice,
}

// SAFETY: `Qwen3Engine` exclusively owns its SynapseAI device context, streams, compiled recipes,
// pinned host buffers, and pre-validated launch descriptors. Moving ownership across threads (e.g.
// behind a `Mutex` in the HTTP server) is safe because no handle is accessed concurrently.
unsafe impl Send for Qwen3Engine {}

impl Qwen3Engine {
    /// Initializes the Gaudi 2 device, validates the model against the verified core, compiles/loads
    /// core recipes, plans & seals HBM arenas (`AllocGuard`), and uploads Safetensors weights.
    pub fn load(cfg: &Qwen3Config) -> Result<Self, String> {
        let t_start = Instant::now();
        let live =
            LiveReporter::from_env("gaudium-rust", env!("CARGO_PKG_VERSION"), "engine", true);
        let model_name = cfg
            .model_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("gaudium-model")
            .to_string();
        live.set_model(&model_name);
        live.set_phase("loading", "opening device and compiling recipes");

        let spec = ModelSpec::from_config_file(cfg.model_dir.join("config.json"))
            .map_err(|e| format!("load config.json: {e}"))?;
        spec.validate()
            .map_err(|e| format!("verified ModelSpec validation failed: {e}"))?;

        let required_elfs: &[&str] = match spec.family {
            ModelFamily::Qwen3
                if spec.n_linear_layers() == 0
                    && !spec.is_moe()
                    && spec.head_dim == 128
                    && spec.n_kv_heads == 8
                    && spec.n_heads == 32 =>
            {
                QWEN3_G8Q4_ELFS
            }
            ModelFamily::Qwen35Dense
                if !spec.is_moe()
                    && spec.head_dim == 256
                    && spec.n_kv_heads == 4
                    && spec.n_heads == 24
                    && spec.gdn_k_heads == 16
                    && spec.gdn_v_heads == 48 =>
            {
                QWEN35_27B_ELFS
            }
            _ => {
                return Err(format!(
                    "Qwen3Engine currently supports dense Qwen3 (DH=128 NKV=8 NH=32) and dense Qwen3.5/3.8 hybrid (DH=256 NKV=4 NH=24 HK=16 HV=48); got family={:?}, DH={}, NKV={}, NH={}, HK={}, HV={}, moe={}",
                    spec.family,
                    spec.head_dim,
                    spec.n_kv_heads,
                    spec.n_heads,
                    spec.gdn_k_heads,
                    spec.gdn_v_heads,
                    spec.is_moe()
                ));
            }
        };

        let has_gdn = spec.n_linear_layers() > 0;
        let max_ctx = align_up(cfg.max_ctx.max(KV_CHUNK), KV_CHUNK);
        let max_chunks = max_ctx / KV_CHUNK;
        let prefill_tile = cfg.prefill_tile.clamp(1, MAX_PREFILL_TILE);

        // 2. Configure TPC-C kernel ELFs & Synapse GC glue BEFORE opening SynapseAI
        let kenv = KernelEnv::configure_from_dir(&cfg.kernels_dir)?;
        let elf_hashes = kenv.fingerprint_hashes(required_elfs)?;

        // 3. Load SynapseAI & open Gaudi 2 device
        let api =
            Arc::new(SynapseApi::load_default().map_err(|e| format!("load libSynapse.so: {e}"))?);
        let driver_version = api.driver_version().unwrap_or_else(|_| "unknown".into());
        let mut dev = HpuDevice::open_gaudi2(Arc::clone(&api))
            .map_err(|e| format!("open Gaudi 2 device: {e}"))?;
        let (free_hbm, total_hbm) = dev
            .memory_info()
            .map_err(|e| format!("query HBM info: {e}"))?;

        let cache = if cfg.no_cache {
            RecipeCache::disabled()
        } else if let Some(dir) = &cfg.cache_dir {
            RecipeCache::at_dir(dir)
        } else {
            RecipeCache::from_env()
        };

        if cfg.verbose {
            eprintln!(
                "[gaudium] device=Gaudi2 driver={} HBM={:.2}/{:.2} GiB family={:?} layers={} (attn={}, lin={}) max_ctx={} ({} chunks)",
                driver_version,
                free_hbm as f64 / (1024.0 * 1024.0 * 1024.0),
                total_hbm as f64 / (1024.0 * 1024.0 * 1024.0),
                spec.family,
                spec.n_layers,
                spec.n_full_attention_layers(),
                spec.n_linear_layers(),
                max_ctx,
                max_chunks
            );
        }

        // 4. Build graph specs to calculate EXACT HBM layout up front
        let head_gb = build_head_t1_graph(&spec);
        let pf_attn_max_gb =
            build_prefill_attn_layer_graph(&spec, prefill_tile, max_ctx, max_chunks, &elf_hashes);
        let pf_lin_max_gb = if has_gdn {
            Some(build_prefill_linear_layer_graph(
                &spec,
                prefill_tile,
                &elf_hashes,
            ))
        } else {
            None
        };
        let step_gbs: Vec<GraphBuilder> = (1..=max_chunks)
            .map(|nch| build_decode_step_graph(&spec, max_ctx, nch, &elf_hashes))
            .collect();

        let mut weight_specs: BTreeMap<String, u64> = BTreeMap::new();
        let mut kv_specs: BTreeMap<String, u64> = BTreeMap::new();
        let mut step_act_specs: BTreeMap<String, u64> = BTreeMap::new();

        for (name, _dt, _dims, bytes) in step_gbs.last().unwrap().persistent_tensor_specs() {
            if is_kv_tensor(&name) {
                kv_specs.insert(name, bytes);
            } else if is_gdn_state_tensor(&name) {
                // Allocated as ping-pong pairs in `gdn_state` below!
                continue;
            } else if is_weight_or_table(&name) {
                weight_specs.insert(name, bytes);
            } else {
                step_act_specs.insert(name, bytes);
            }
        }
        if !spec.norm_plus1 {
            weight_specs
                .entry("norm_ones".to_string())
                .or_insert(spec.hidden * 2);
        }
        weight_specs
            .entry("w_final".to_string())
            .or_insert(spec.hidden * 2);
        if has_gdn {
            weight_specs.entry("zero_slot".to_string()).or_insert(4);
        }

        // Prefill scratch specs (sized for `prefill_tile` & `max_chunks`)
        let mut pf_attn_act_specs: BTreeMap<String, u64> = BTreeMap::new();
        for (name, _dt, _dims, bytes) in pf_attn_max_gb.persistent_tensor_specs() {
            if name == "x_in"
                || name == "L_out"
                || name == "cursor"
                || name == "rope_c"
                || name == "rope_s"
                || name == "norm_ones"
                || is_generic_layer_weight(&name)
                || name == "L_kcache"
                || name == "L_vcache"
            {
                continue;
            }
            pf_attn_act_specs.insert(name, bytes);
        }

        let mut pf_lin_act_specs: BTreeMap<String, u64> = BTreeMap::new();
        if let Some(ref gb) = pf_lin_max_gb {
            for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
                if name == "x_in"
                    || name == "L_out"
                    || name == "zero_slot"
                    || name == "norm_ones"
                    || is_generic_layer_weight(&name)
                    || matches!(name.as_str(), "L_S_in" | "L_S_out" | "L_cs_in" | "L_cs_out")
                {
                    continue;
                }
                pf_lin_act_specs.insert(name, bytes);
            }
        }

        let mut head_act_specs: BTreeMap<String, u64> = BTreeMap::new();
        for (name, _dt, _dims, bytes) in head_gb.persistent_tensor_specs() {
            if name == "x_in" || is_weight_or_table(&name) {
                continue;
            }
            head_act_specs.insert(name, bytes);
        }

        // 5. Compile core recipes BEFORE HBM allocation so we know exact workspace requirements
        let t_comp = Instant::now();
        let mut head_t1 = head_gb
            .compile(Arc::clone(&api), &cache, &driver_version)
            .map_err(|e| format!("compile head_T1: {e}"))?;
        let mut step_recipes = Vec::with_capacity(step_gbs.len());
        let mut max_ws = head_t1.ws_size;
        for (idx, gb) in step_gbs.iter().enumerate() {
            let r = gb
                .compile(Arc::clone(&api), &cache, &driver_version)
                .map_err(|e| format!("compile decode step (chunks={}): {e}", idx + 1))?;
            if cfg.verbose {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                    r.name,
                    r.from_cache,
                    r.ws_size as f64 / (1024.0 * 1024.0)
                );
            }
            max_ws = max_ws.max(r.ws_size);
            step_recipes.push(r);
        }

        let mut attn_layer_recipes: HashMap<(u64, u64), CompiledRecipe> = HashMap::new();
        let mut lin_layer_recipes: HashMap<u64, CompiledRecipe> = HashMap::new();
        let mut embed_recipes: HashMap<u64, CompiledRecipe> = HashMap::new();
        if prefill_tile > 1 {
            let pf_c1_gb =
                build_prefill_attn_layer_graph(&spec, prefill_tile, max_ctx, 1, &elf_hashes);
            let r_pf = pf_c1_gb
                .compile(Arc::clone(&api), &cache, &driver_version)
                .map_err(|e| format!("compile prefill attn layer T={prefill_tile} C=1: {e}"))?;
            max_ws = max_ws.max(r_pf.ws_size);
            if cfg.verbose {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                    r_pf.name,
                    r_pf.from_cache,
                    r_pf.ws_size as f64 / (1024.0 * 1024.0)
                );
            }
            attn_layer_recipes.insert((prefill_tile, 1), r_pf);

            if let Some(ref lin_gb) = pf_lin_max_gb {
                let r_lin = lin_gb
                    .compile(Arc::clone(&api), &cache, &driver_version)
                    .map_err(|e| format!("compile prefill lin layer T={prefill_tile}: {e}"))?;
                max_ws = max_ws.max(r_lin.ws_size);
                if cfg.verbose {
                    eprintln!(
                        "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                        r_lin.name,
                        r_lin.from_cache,
                        r_lin.ws_size as f64 / (1024.0 * 1024.0)
                    );
                }
                lin_layer_recipes.insert(prefill_tile, r_lin);
            }

            let emb_gb = build_embed_graph(&spec, prefill_tile);
            let r_emb = emb_gb
                .compile(Arc::clone(&api), &cache, &driver_version)
                .map_err(|e| format!("compile embed T={prefill_tile}: {e}"))?;
            max_ws = max_ws.max(r_emb.ws_size);
            embed_recipes.insert(prefill_tile, r_emb);
        }
        if cfg.verbose {
            eprintln!(
                "[gaudium] core recipes ready in {:.2}s (max_ws={:.2} MiB)",
                t_comp.elapsed().as_secs_f64(),
                max_ws as f64 / (1024.0 * 1024.0)
            );
        }

        // 6. Compute exact arena sizes & check HBM fit via verified `core::hbm`!
        let align256 = |n: u64| align_up(n.max(256), 256);
        let weights_bytes: u64 = weight_specs.values().map(|&b| align256(b)).sum();

        let s_one_bytes = if has_gdn {
            64 * 256 * spec.gdn_v_heads * 4
        } else {
            0
        };
        let cs_one_bytes = if has_gdn {
            spec.gdn_qkv_dim() * 3 * 4
        } else {
            0
        };
        let n_lin = spec.n_linear_layers() as u64;
        let gdn_state_bytes = n_lin * (2 * align256(s_one_bytes) + 2 * align256(cs_one_bytes));
        let kv_bytes: u64 = kv_specs.values().map(|&b| align256(b)).sum::<u64>() + gdn_state_bytes;

        let pf_x_bytes = align256(spec.hidden * prefill_tile * 2);
        let pf_ids_bytes = align256(prefill_tile * 4);
        let pf_cursor_bytes = 256u64;
        let act_bytes: u64 = pf_x_bytes * 2
            + pf_ids_bytes
            + pf_cursor_bytes
            + pf_attn_act_specs
                .values()
                .map(|&b| align256(b))
                .sum::<u64>()
            + pf_lin_act_specs.values().map(|&b| align256(b)).sum::<u64>()
            + head_act_specs.values().map(|&b| align256(b)).sum::<u64>()
            + step_act_specs.values().map(|&b| align256(b)).sum::<u64>();
        let ws_bytes = align256(max_ws.max(MIN_SHARED_WORKSPACE_BYTES));

        let planned_total = weights_bytes
            .saturating_add(kv_bytes)
            .saturating_add(act_bytes)
            .saturating_add(ws_bytes);
        let headroom = 512 * 1024 * 1024;
        dev.check_plan_fits(planned_total, headroom)
            .map_err(|e| format!("HBM plan check failed: {e}"))?;

        if cfg.verbose {
            eprintln!(
                "[gaudium] HBM plan fits: weights={:.2} GiB, kv+state={:.2} GiB, acts={:.2} MiB, ws={:.2} MiB",
                weights_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                act_bytes as f64 / (1024.0 * 1024.0),
                ws_bytes as f64 / (1024.0 * 1024.0),
            );
        }

        // 7. Allocate arenas and seal AllocGuard
        let mut weights_arena = dev
            .alloc_arena(weights_bytes, HbmCat::CatWeights, "weights_arena")
            .map_err(|e| format!("alloc weights arena: {e}"))?;
        let mut kv_arena = dev
            .alloc_arena(kv_bytes, HbmCat::CatKv, "kv_state_arena")
            .map_err(|e| format!("alloc kv/state arena: {e}"))?;
        let mut act_arena = dev
            .alloc_arena(act_bytes, HbmCat::CatScratch, "act_arena")
            .map_err(|e| format!("alloc act arena: {e}"))?;

        let suballoc = |arena: &mut HbmArena, bytes: u64| -> Result<u64, String> {
            let (addr, _off) = arena
                .suballoc(bytes.max(256), 256)
                .map_err(|e| format!("arena suballoc: {e}"))?;
            Ok(addr)
        };

        let mut weights_addr = BTreeMap::new();
        for (name, &bytes) in &weight_specs {
            weights_addr.insert(name.clone(), suballoc(&mut weights_arena, bytes)?);
        }
        let mut kv_addr = BTreeMap::new();
        for (name, &bytes) in &kv_specs {
            kv_addr.insert(name.clone(), (suballoc(&mut kv_arena, bytes)?, bytes));
        }
        let mut gdn_state = BTreeMap::new();
        for l in 0..(spec.n_layers as usize) {
            if !spec.is_full_attention_layer(l) {
                let s0 = suballoc(&mut kv_arena, s_one_bytes)?;
                let s1 = suballoc(&mut kv_arena, s_one_bytes)?;
                let cs0 = suballoc(&mut kv_arena, cs_one_bytes)?;
                let cs1 = suballoc(&mut kv_arena, cs_one_bytes)?;
                gdn_state.insert(
                    l,
                    GdnLayerState {
                        s: [s0, s1],
                        s_bytes: s_one_bytes,
                        cs: [cs0, cs1],
                        cs_bytes: cs_one_bytes,
                    },
                );
            }
        }

        let pf_x0 = suballoc(&mut act_arena, pf_x_bytes)?;
        let pf_x1 = suballoc(&mut act_arena, pf_x_bytes)?;
        let pf_ids = suballoc(&mut act_arena, pf_ids_bytes)?;
        let pf_cursor = suballoc(&mut act_arena, pf_cursor_bytes)?;

        let mut pf_attn_scratch = BTreeMap::new();
        for (name, &bytes) in &pf_attn_act_specs {
            pf_attn_scratch.insert(name.clone(), suballoc(&mut act_arena, bytes)?);
        }
        let mut pf_lin_scratch = BTreeMap::new();
        for (name, &bytes) in &pf_lin_act_specs {
            pf_lin_scratch.insert(name.clone(), suballoc(&mut act_arena, bytes)?);
        }
        let mut head_scratch = BTreeMap::new();
        for (name, &bytes) in &head_act_specs {
            head_scratch.insert(name.clone(), suballoc(&mut act_arena, bytes)?);
        }
        let mut step_act_addr = BTreeMap::new();
        for (name, &bytes) in &step_act_specs {
            step_act_addr.insert(name.clone(), suballoc(&mut act_arena, bytes)?);
        }

        dev.seal_with_shared_workspace(ws_bytes)
            .map_err(|e| format!("seal HBM allocations: {e}"))?;

        let kv_only_bytes = kv_bytes.saturating_sub(gdn_state_bytes);
        let hbm_bytes = [
            weights_bytes,
            kv_only_bytes,
            gdn_state_bytes,
            act_bytes,
            ws_bytes,
        ];
        live.set_hbm_bytes(
            hbm_bytes[0],
            hbm_bytes[1],
            hbm_bytes[2],
            hbm_bytes[3],
            hbm_bytes[4],
        );
        live.set_phase("loading", "uploading weights to HBM");

        head_t1
            .bind_shared_workspace(&dev)
            .map_err(|e| format!("bind head_T1 ws: {e}"))?;
        for r in &mut step_recipes {
            r.bind_shared_workspace(&dev)
                .map_err(|e| format!("bind step ws: {e}"))?;
        }
        for r in attn_layer_recipes.values_mut() {
            r.bind_shared_workspace(&dev)
                .map_err(|e| format!("bind attn layer ws: {e}"))?;
        }
        for r in lin_layer_recipes.values_mut() {
            r.bind_shared_workspace(&dev)
                .map_err(|e| format!("bind lin layer ws: {e}"))?;
        }
        for r in embed_recipes.values_mut() {
            r.bind_shared_workspace(&dev)
                .map_err(|e| format!("bind embed ws: {e}"))?;
        }

        // 8. Upload weights + RoPE tables from zero-copy mmap Safetensors into `weights_arena`
        let t_up = Instant::now();
        let store =
            SafetensorsStore::open(&cfg.model_dir).map_err(|e| format!("open safetensors: {e}"))?;
        let packer = WeightPacker::new(&store, &spec);

        let mut stage_upload = dev
            .alloc_pinned(STAGE_BUF_BYTES)
            .map_err(|e| format!("alloc pinned upload stage: {e}"))?;

        let (rope_c_bytes, rope_s_bytes) = packer.build_rope_tables(max_ctx as usize);
        if let Some(&addr) = weights_addr.get("rope_c") {
            dev.upload_h2d_chunked(&rope_c_bytes, addr, &mut stage_upload)
                .map_err(|e| format!("upload rope_c: {e}"))?;
        }
        if let Some(&addr) = weights_addr.get("rope_s") {
            dev.upload_h2d_chunked(&rope_s_bytes, addr, &mut stage_upload)
                .map_err(|e| format!("upload rope_s: {e}"))?;
        }

        let mut uploaded_bytes = (rope_c_bytes.len() + rope_s_bytes.len()) as u64;
        for (name, &addr) in &weights_addr {
            if name == "rope_c" || name == "rope_s" {
                continue;
            }
            let expected = weight_specs[name];
            let data = packer.pack_weight(name)?;
            if data.len() as u64 != expected {
                return Err(format!(
                    "weight '{name}' packed size {} != expected graph tensor size {expected}",
                    data.len()
                ));
            }
            dev.upload_h2d_chunked(&data, addr, &mut stage_upload)
                .map_err(|e| format!("upload weight '{name}': {e}"))?;
            uploaded_bytes += expected;
        }
        stage_upload.as_mut_slice().fill(0);
        let stage_zero = stage_upload;

        let logits_bytes = (spec.vocab as usize) * 2;
        let stage_io_bytes = logits_bytes.max((prefill_tile as usize) * 4).max(4096);
        let stage_io = dev
            .alloc_pinned(stage_io_bytes)
            .map_err(|e| format!("alloc pinned io stage: {e}"))?;

        if cfg.verbose {
            let dt = t_up.elapsed().as_secs_f64();
            eprintln!(
                "[gaudium] uploaded {:.2} GiB weights in {:.2}s ({:.2} GiB/s); total init {:.2}s",
                uploaded_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                dt,
                (uploaded_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / dt.max(1e-6),
                t_start.elapsed().as_secs_f64()
            );
        }

        // Pre-validate decode-step launches for BOTH GDN ping-pong parities (0 and 1)!
        let mut base_step_addrs = BTreeMap::new();
        for (k, &v) in &weights_addr {
            base_step_addrs.insert(k.clone(), v);
        }
        for (k, &(v, _)) in &kv_addr {
            base_step_addrs.insert(k.clone(), v);
        }
        for (k, &v) in &step_act_addr {
            base_step_addrs.insert(k.clone(), v);
        }

        let mut step_launches = Vec::with_capacity(step_recipes.len());
        for r in &step_recipes {
            let build_for_parity = |parity: usize| -> Result<PreparedLaunch, String> {
                let mut pool = base_step_addrs.clone();
                for (&l, st) in &gdn_state {
                    pool.insert(format!("L{l}_S_in"), st.s[parity]);
                    pool.insert(format!("L{l}_S_out"), st.s[1 - parity]);
                    pool.insert(format!("L{l}_cs_in"), st.cs[parity]);
                    pool.insert(format!("L{l}_cs_out"), st.cs[1 - parity]);
                }
                let exact = select_exact_bindings(r, &pool)?;
                r.prepare_launch(&exact)
                    .map_err(|e| format!("prepare decode step launch (parity={parity}): {e}"))
            };
            step_launches.push([build_for_parity(0)?, build_for_parity(1)?]);
        }

        let step_ids = *step_act_addr.get("ids").ok_or("missing step 'ids'")?;
        let step_cursor = *step_act_addr.get("cursor").ok_or("missing step 'cursor'")?;
        let step_tokens = *step_act_addr.get("tokens").ok_or("missing step 'tokens'")?;
        let step_logits = *step_act_addr.get("logits").ok_or("missing step 'logits'")?;

        live.set_phase("idle", "ready");

        Ok(Self {
            spec,
            max_ctx,
            max_chunks,
            prefill_tile,
            driver_version,
            cache,
            elf_hashes,
            head_t1,
            step_recipes,
            embed_recipes,
            attn_layer_recipes,
            lin_layer_recipes,
            stage_io,
            stage_zero,
            weights_addr,
            kv_addr,
            gdn_state,
            pf_attn_scratch,
            pf_lin_scratch,
            pf_x0,
            pf_x1,
            pf_ids,
            pf_cursor,
            head_scratch,
            step_launches,
            step_ids,
            step_cursor,
            step_tokens,
            step_logits,
            hbm_bytes,
            live,
            dev,
        })
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn max_ctx(&self) -> u64 {
        self.max_ctx
    }

    /// Exact 5-category HBM ledger bytes `[weights, kv, state, scratch, workspace]`.
    pub fn hbm_bytes(&self) -> [u64; 5] {
        self.hbm_bytes
    }

    /// Zeroes all KV caches and GDN initial recurrent/conv states (slot 0) before a new request.
    fn clear_request_state(&self) -> Result<(), String> {
        for &(addr, bytes) in self.kv_addr.values() {
            self.dev
                .zero_hbm_chunked(addr, bytes, &self.stage_zero)
                .map_err(|e| format!("zero kv cache: {e}"))?;
        }
        for st in self.gdn_state.values() {
            self.dev
                .zero_hbm_chunked(st.s[0], st.s_bytes, &self.stage_zero)
                .map_err(|e| format!("zero GDN S state: {e}"))?;
            self.dev
                .zero_hbm_chunked(st.cs[0], st.cs_bytes, &self.stage_zero)
                .map_err(|e| format!("zero GDN conv state: {e}"))?;
        }
        Ok(())
    }

    fn ensure_embed_recipe(&mut self, t: u64) -> Result<(), String> {
        if self.embed_recipes.contains_key(&t) {
            return Ok(());
        }
        let gb = build_embed_graph(&self.spec, t);
        let mut r = gb
            .compile(
                Arc::clone(self.dev.api()),
                &self.cache,
                &self.driver_version,
            )
            .map_err(|e| format!("compile embed T={t}: {e}"))?;
        r.bind_shared_workspace(&self.dev)
            .map_err(|e| format!("bind embed T={t} ws: {e}"))?;
        self.embed_recipes.insert(t, r);
        Ok(())
    }

    fn ensure_attn_layer_recipe(&mut self, t: u64, nch: u64) -> Result<(), String> {
        if self.attn_layer_recipes.contains_key(&(t, nch)) {
            return Ok(());
        }
        let gb = build_prefill_attn_layer_graph(&self.spec, t, self.max_ctx, nch, &self.elf_hashes);
        let mut r = gb
            .compile(
                Arc::clone(self.dev.api()),
                &self.cache,
                &self.driver_version,
            )
            .map_err(|e| format!("compile attn layer T={t} C={nch}: {e}"))?;
        r.bind_shared_workspace(&self.dev)
            .map_err(|e| format!("bind attn layer T={t} C={nch} ws: {e}"))?;
        self.attn_layer_recipes.insert((t, nch), r);
        Ok(())
    }

    fn ensure_lin_layer_recipe(&mut self, t: u64) -> Result<(), String> {
        if self.lin_layer_recipes.contains_key(&t) {
            return Ok(());
        }
        let gb = build_prefill_linear_layer_graph(&self.spec, t, &self.elf_hashes);
        let mut r = gb
            .compile(
                Arc::clone(self.dev.api()),
                &self.cache,
                &self.driver_version,
            )
            .map_err(|e| format!("compile lin layer T={t}: {e}"))?;
        r.bind_shared_workspace(&self.dev)
            .map_err(|e| format!("bind lin layer T={t} ws: {e}"))?;
        self.lin_layer_recipes.insert(t, r);
        Ok(())
    }

    fn write_step_inputs(&mut self, token_id: i32, cursor_val: i32) -> Result<(), String> {
        let buf = self.stage_io.as_mut_slice();
        buf[0..4].copy_from_slice(&token_id.to_le_bytes());
        buf[64..68].copy_from_slice(&cursor_val.to_le_bytes());
        self.dev
            .copy_h2d_pinned_batch_sync(
                &self.stage_io,
                &[(0, 4, self.step_ids), (64, 4, self.step_cursor)],
            )
            .map_err(|e| format!("write step inputs H2D: {e}"))
    }

    fn write_i32_scalar(&mut self, hbm_addr: u64, val: i32) -> Result<(), String> {
        self.stage_io.as_mut_slice()[..4].copy_from_slice(&val.to_le_bytes());
        self.dev
            .copy_h2d_pinned_sync(&self.stage_io, 4, hbm_addr)
            .map_err(|e| format!("write i32 H2D: {e}"))
    }

    fn read_i32_scalar(&mut self, hbm_addr: u64) -> Result<i32, String> {
        self.dev
            .copy_d2h_pinned_sync(hbm_addr, &mut self.stage_io, 4)
            .map_err(|e| format!("read i32 D2H: {e}"))?;
        let b = &self.stage_io.as_slice()[..4];
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn sample_from_logits(
        &mut self,
        logits_hbm: u64,
        params: &Qwen3GenParams,
        seq_pos: u32,
    ) -> Result<i32, String> {
        let vocab = self.spec.vocab as usize;
        let bytes = vocab * 2;
        self.dev
            .copy_d2h_pinned_sync(logits_hbm, &mut self.stage_io, bytes)
            .map_err(|e| format!("read logits D2H: {e}"))?;
        let raw = &self.stage_io.as_slice()[..bytes];
        let mut bf16 = Vec::with_capacity(vocab);
        for &ch in raw.as_chunks::<2>().0 {
            bf16.push(u16::from_le_bytes(ch));
        }
        let sp = SampleParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            min_p: params.min_p,
            seed: params.seed,
            coupled: true,
            draft_trunc: true,
        };
        let dev_sp = sp.to_dev(self.spec.vocab);
        let res = sample_row(&bf16, &dev_sp, 0, seq_pos, true);
        Ok(res.token as i32)
    }

    /// Runs full prefill + autoregressive decode for `prompt_ids`, invoking `on_token` for every
    /// generated token ID as soon as it is produced on the Gaudi 2 card.
    pub fn generate(
        &mut self,
        prompt_ids: &[i32],
        params: &Qwen3GenParams,
        on_token: impl FnMut(i32),
    ) -> Result<Qwen3GenReport, String> {
        self.generate_cancellable(prompt_ids, params, None, on_token)
    }

    /// Like `generate`, but checks `cancel` cleanly between prefill tiles and after each synced
    /// decode step so no Synapse launch is ever left queued on cancellation.
    pub fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &Qwen3GenParams,
        cancel: Option<&AtomicBool>,
        mut on_token: impl FnMut(i32),
    ) -> Result<Qwen3GenReport, String> {
        if prompt_ids.is_empty() {
            return Err("prompt must contain at least 1 token".into());
        }

        let mut budget =
            SequenceBudget::admit(prompt_ids.len() as u64, Some(params.max_new), self.max_ctx)
                .map_err(|e| format!("SequenceBudget rejected prompt/max_new: {e:?}"))?;

        let mut eos_ids = self.spec.eos_token_ids.clone();
        for &id in &params.extra_eos_ids {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
        let is_eos = |tok: i32| eos_ids.contains(&tok);
        let is_cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));

        let method_label = if params.temperature > 0.0 {
            format!("K=0 BF16 T={:.2}", params.temperature)
        } else {
            "K=0 BF16 greedy".to_string()
        };
        self.live
            .begin_request(prompt_ids.len() as u64, &method_label, "");

        let res = self.run_generate_inner(
            prompt_ids,
            params,
            &mut budget,
            &is_eos,
            &is_cancelled,
            &mut on_token,
        );
        self.live.end_request();
        res
    }

    fn run_generate_inner(
        &mut self,
        prompt_ids: &[i32],
        params: &Qwen3GenParams,
        budget: &mut SequenceBudget,
        is_eos: &impl Fn(i32) -> bool,
        is_cancelled: &impl Fn() -> bool,
        on_token: &mut impl FnMut(i32),
    ) -> Result<Qwen3GenReport, String> {
        self.clear_request_state()?;
        let mut gdn_parity: usize = 0;

        let t_prefill = Instant::now();
        let mut generated = Vec::with_capacity(budget.max_new() as usize);
        let mut stopped_by_eos = false;
        let mut stopped_by_cancel = false;

        // ---- PREFILL PHASE ----
        let first_tok: i32 = if prompt_ids.len() == 1 {
            self.write_step_inputs(prompt_ids[0], 0)?;
            self.step_recipes[0]
                .launch_prepared(&self.dev, &self.step_launches[0][gdn_parity])
                .map_err(|e| format!("launch step (prompt len 1): {e}"))?;
            unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                .map_err(|e| format!("sync compute (prompt len 1): {e}"))?;
            gdn_parity ^= 1;
            if params.temperature > 0.0 {
                self.sample_from_logits(self.step_logits, params, 0)?
            } else {
                self.read_i32_scalar(self.step_tokens)?
            }
        } else {
            let mut pos = 0usize;
            let n_layers = self.spec.n_layers as usize;
            let has_gdn = !self.gdn_state.is_empty();
            let mut last_tile_len = 0u64;

            while pos < prompt_ids.len() {
                if is_cancelled() {
                    stopped_by_cancel = true;
                    break;
                }
                let tile_len = (prompt_ids.len() - pos).min(self.prefill_tile as usize) as u64;
                let end_pos = (pos as u64) + tile_len;
                let nch = end_pos.div_ceil(KV_CHUNK).clamp(1, self.max_chunks);
                last_tile_len = tile_len;

                self.ensure_embed_recipe(tile_len)?;
                self.ensure_attn_layer_recipe(tile_len, nch)?;
                if has_gdn {
                    self.ensure_lin_layer_recipe(tile_len)?;
                }

                let tile_bytes = (tile_len as usize) * 4;
                for (k, &id) in prompt_ids[pos..pos + tile_len as usize].iter().enumerate() {
                    self.stage_io.as_mut_slice()[k * 4..k * 4 + 4]
                        .copy_from_slice(&id.to_le_bytes());
                }
                self.dev
                    .copy_h2d_pinned_sync(&self.stage_io, tile_bytes, self.pf_ids)
                    .map_err(|e| format!("upload prefill ids: {e}"))?;
                self.write_i32_scalar(self.pf_cursor, pos as i32)?;

                // Launch embed(tile_len) -> pf_x0
                let mut emb_pool = BTreeMap::new();
                emb_pool.insert("embed_table".to_string(), self.weights_addr["embed_table"]);
                emb_pool.insert("ids".to_string(), self.pf_ids);
                emb_pool.insert("x0".to_string(), self.pf_x0);
                let emb_recipe = &self.embed_recipes[&tile_len];
                let emb_bind = select_exact_bindings(emb_recipe, &emb_pool)?;
                emb_recipe
                    .launch_checked(&self.dev, &emb_bind)
                    .map_err(|e| format!("launch embed T={tile_len}: {e}"))?;

                // Prepare reusable pools for attention & linear layers
                let attn_recipe = &self.attn_layer_recipes[&(tile_len, nch)];
                let mut apool = BTreeMap::new();
                for (k, &v) in &self.pf_attn_scratch {
                    apool.insert(k.clone(), v);
                }
                apool.insert("cursor".to_string(), self.pf_cursor);
                apool.insert("rope_c".to_string(), self.weights_addr["rope_c"]);
                apool.insert("rope_s".to_string(), self.weights_addr["rope_s"]);
                if let Some(&ones) = self.weights_addr.get("norm_ones") {
                    apool.insert("norm_ones".to_string(), ones);
                }

                let mut lpool = BTreeMap::new();
                if has_gdn {
                    for (k, &v) in &self.pf_lin_scratch {
                        lpool.insert(k.clone(), v);
                    }
                    lpool.insert("zero_slot".to_string(), self.weights_addr["zero_slot"]);
                    if let Some(&ones) = self.weights_addr.get("norm_ones") {
                        lpool.insert("norm_ones".to_string(), ones);
                    }
                }

                for l in 0..n_layers {
                    let (xin, xout) = if l % 2 == 0 {
                        (self.pf_x0, self.pf_x1)
                    } else {
                        (self.pf_x1, self.pf_x0)
                    };

                    if self.spec.is_full_attention_layer(l) {
                        apool.insert("x_in".to_string(), xin);
                        apool.insert("L_out".to_string(), xout);
                        for suffix in [
                            "w_in", "w_post", "W_qg", "W_k", "W_v", "qn_w", "kn_w", "W_o",
                            "W_gate", "W_up", "W_down",
                        ] {
                            let real_key = format!("L{l}_{suffix}");
                            apool.insert(format!("L_{suffix}"), self.weights_addr[&real_key]);
                        }
                        apool.insert(
                            "L_kcache".to_string(),
                            self.kv_addr[&format!("L{l}_kcache")].0,
                        );
                        apool.insert(
                            "L_vcache".to_string(),
                            self.kv_addr[&format!("L{l}_vcache")].0,
                        );
                        let abind = select_exact_bindings(attn_recipe, &apool)?;
                        attn_recipe.launch_checked(&self.dev, &abind).map_err(|e| {
                            format!("launch prefill attn layer {l} T={tile_len}: {e}")
                        })?;
                    } else {
                        let lin_recipe = &self.lin_layer_recipes[&tile_len];
                        lpool.insert("x_in".to_string(), xin);
                        lpool.insert("L_out".to_string(), xout);
                        for suffix in [
                            "w_in", "w_post", "W_qkvz", "W_ba", "conv_w", "conv_b", "A_log",
                            "dt_bias", "norm_w", "W_o", "W_gate", "W_up", "W_down",
                        ] {
                            let real_key = format!("L{l}_{suffix}");
                            lpool.insert(format!("L_{suffix}"), self.weights_addr[&real_key]);
                        }
                        let st = self.gdn_state[&l];
                        lpool.insert("L_S_in".to_string(), st.s[gdn_parity]);
                        lpool.insert("L_S_out".to_string(), st.s[1 - gdn_parity]);
                        lpool.insert("L_cs_in".to_string(), st.cs[gdn_parity]);
                        lpool.insert("L_cs_out".to_string(), st.cs[1 - gdn_parity]);
                        let lbind = select_exact_bindings(lin_recipe, &lpool)?;
                        lin_recipe.launch_checked(&self.dev, &lbind).map_err(|e| {
                            format!("launch prefill lin layer {l} T={tile_len}: {e}")
                        })?;
                    }
                }

                unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                    .map_err(|e| format!("sync prefill tile at pos={pos}: {e}"))?;
                if has_gdn {
                    gdn_parity ^= 1;
                }
                pos += tile_len as usize;
            }

            if stopped_by_cancel {
                let ttft_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
                return Ok(Qwen3GenReport {
                    prompt_tokens: prompt_ids.len(),
                    generated_tokens: generated,
                    ttft_ms,
                    prefill_tok_s: 0.0,
                    decode_ms: 0.0,
                    decode_tok_s: 0.0,
                    ms_per_step: 0.0,
                    stopped_by_eos: false,
                    stopped_by_cancel: true,
                });
            }

            // Launch shared head_T1 directly on the last prompt token's hidden row!
            let final_x_base = if n_layers.is_multiple_of(2) {
                self.pf_x0
            } else {
                self.pf_x1
            };
            let last_row_addr = final_x_base + (last_tile_len - 1) * self.spec.hidden * 2;

            let mut hpool = BTreeMap::new();
            for (k, &v) in &self.head_scratch {
                hpool.insert(k.clone(), v);
            }
            hpool.insert("x_in".to_string(), last_row_addr);
            hpool.insert("w_final".to_string(), self.weights_addr["w_final"]);
            if let Some(&ones) = self.weights_addr.get("norm_ones") {
                hpool.insert("norm_ones".to_string(), ones);
            }
            let lm_w = if self.spec.tie_word_embeddings {
                "embed_table"
            } else {
                "W_lm"
            };
            hpool.insert(lm_w.to_string(), self.weights_addr[lm_w]);

            let hbind = select_exact_bindings(&self.head_t1, &hpool)?;
            self.head_t1
                .launch_checked(&self.dev, &hbind)
                .map_err(|e| format!("launch head_T1: {e}"))?;
            unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                .map_err(|e| format!("sync head_T1: {e}"))?;

            if params.temperature > 0.0 {
                self.sample_from_logits(
                    self.head_scratch["logits"],
                    params,
                    (prompt_ids.len() - 1) as u32,
                )?
            } else {
                self.read_i32_scalar(self.head_scratch["tokens"])?
            }
        };

        let ttft_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let prefill_tok_s = (prompt_ids.len() as f64) / (ttft_ms / 1000.0).max(1e-9);

        if budget.remaining() > 0 && !is_cancelled() {
            budget.record_emitted(1);
            generated.push(first_tok);
            self.live.first_token(Some(ttft_ms), 1);
            on_token(first_tok);
            if params.stop_on_eos && is_eos(first_tok) {
                stopped_by_eos = true;
            }
        } else if is_cancelled() {
            stopped_by_cancel = true;
        }

        // ---- DECODE PHASE (1 monolithic synLaunch per token, ping-pong GDN parity!) ----
        let t_decode = Instant::now();
        let mut cur_tok = first_tok;
        let mut decode_steps = 0usize;

        while !stopped_by_eos && !stopped_by_cancel && budget.remaining() > 0 {
            if is_cancelled() {
                stopped_by_cancel = true;
                break;
            }
            let cursor_val = (prompt_ids.len() + decode_steps) as u64;
            let nch = (cursor_val + 1)
                .div_ceil(KV_CHUNK)
                .clamp(1, self.max_chunks) as usize;

            self.write_step_inputs(cur_tok, cursor_val as i32)?;

            self.step_recipes[nch - 1]
                .launch_prepared(&self.dev, &self.step_launches[nch - 1][gdn_parity])
                .map_err(|e| {
                    format!("launch decode step {decode_steps} (cursor={cursor_val}): {e}")
                })?;
            unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                .map_err(|e| format!("sync decode step {decode_steps}: {e}"))?;
            gdn_parity ^= 1;

            let next_tok = if params.temperature > 0.0 {
                self.sample_from_logits(self.step_logits, params, cursor_val as u32)?
            } else {
                self.read_i32_scalar(self.step_tokens)?
            };

            decode_steps += 1;
            budget.record_emitted(1);
            generated.push(next_tok);
            self.live.step(1, 0, 0);
            on_token(next_tok);
            cur_tok = next_tok;

            if params.stop_on_eos && is_eos(next_tok) {
                stopped_by_eos = true;
            }
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let decode_tok_s = if decode_steps > 0 {
            (decode_steps as f64) / (decode_ms / 1000.0).max(1e-9)
        } else {
            0.0
        };
        let ms_per_step = if decode_steps > 0 {
            decode_ms / (decode_steps as f64)
        } else {
            0.0
        };

        Ok(Qwen3GenReport {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: generated,
            ttft_ms,
            prefill_tok_s,
            decode_ms,
            decode_tok_s,
            ms_per_step,
            stopped_by_eos,
            stopped_by_cancel,
        })
    }
}

fn select_exact_bindings(
    recipe: &CompiledRecipe,
    pool: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, u64>, String> {
    let mut exact = BTreeMap::new();
    for t in recipe.tensors() {
        let &addr = pool.get(&t.name).ok_or_else(|| {
            format!(
                "recipe {} requires persistent tensor '{}' which has no allocated HBM buffer",
                recipe.name, t.name
            )
        })?;
        exact.insert(t.name.clone(), addr);
    }
    Ok(exact)
}

fn is_kv_tensor(name: &str) -> bool {
    name.ends_with("_kcache") || name.ends_with("_vcache")
}

fn is_gdn_state_tensor(name: &str) -> bool {
    name.ends_with("_S_in")
        || name.ends_with("_S_out")
        || name.ends_with("_cs_in")
        || name.ends_with("_cs_out")
}

fn is_weight_or_table(name: &str) -> bool {
    if matches!(
        name,
        "embed_table" | "W_lm" | "w_final" | "norm_ones" | "rope_c" | "rope_s" | "zero_slot"
    ) {
        return true;
    }
    if let Some(rest) = name.strip_prefix('L') {
        if let Some(sep) = rest.find('_') {
            if rest[..sep].chars().all(|c| c.is_ascii_digit()) && !rest[..sep].is_empty() {
                return matches!(
                    &rest[sep + 1..],
                    "w_in"
                        | "w_post"
                        | "W_qg"
                        | "W_k"
                        | "W_v"
                        | "qn_w"
                        | "kn_w"
                        | "W_qkvz"
                        | "W_ba"
                        | "conv_w"
                        | "conv_b"
                        | "A_log"
                        | "dt_bias"
                        | "norm_w"
                        | "W_o"
                        | "W_gate"
                        | "W_up"
                        | "W_down"
                );
            }
        }
    }
    false
}

fn is_generic_layer_weight(name: &str) -> bool {
    matches!(
        name,
        "L_w_in"
            | "L_w_post"
            | "L_W_qg"
            | "L_W_k"
            | "L_W_v"
            | "L_qn_w"
            | "L_kn_w"
            | "L_W_qkvz"
            | "L_W_ba"
            | "L_conv_w"
            | "L_conv_b"
            | "L_A_log"
            | "L_dt_bias"
            | "L_norm_w"
            | "L_W_o"
            | "L_W_gate"
            | "L_W_up"
            | "L_W_down"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_8b_spec() -> ModelSpec {
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
        ModelSpec::from_config_json(cfg).unwrap()
    }

    fn qwen38_27b_spec() -> ModelSpec {
        let cfg = r#"{
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
        ModelSpec::from_config_json(cfg).unwrap()
    }

    #[test]
    fn qwen3_8b_graphs_build_and_fingerprint_deterministically() {
        let spec = qwen3_8b_spec();
        spec.validate().unwrap();
        let elfs = vec![("GAUDIUM_A128PREP_G8Q4_ELF".into(), "deadbeef".into())];

        let emb = build_embed_graph(&spec, 16);
        let head = build_head_t1_graph(&spec);
        let pf = build_prefill_attn_layer_graph(&spec, 16, 4096, 1, &elfs);
        let step1 = build_decode_step_graph(&spec, 4096, 1, &elfs);
        let step2 = build_decode_step_graph(&spec, 4096, 1, &elfs);

        assert_eq!(step1.fingerprint("1.24.2"), step2.fingerprint("1.24.2"));
        assert_ne!(pf.fingerprint("1.24.2"), step1.fingerprint("1.24.2"));
        assert_eq!(emb.persistent_tensor_specs().len(), 3);
        assert!(head
            .persistent_tensor_specs()
            .iter()
            .any(|t| t.0 == "tokens"));

        let kv_count = step1
            .persistent_tensor_specs()
            .iter()
            .filter(|t| is_kv_tensor(&t.0))
            .count();
        assert_eq!(kv_count, 72); // 36 layers * (kcache + vcache)
    }

    #[test]
    fn qwen38_27b_hybrid_graphs_cover_48_gdn_and_16_attn_layers() {
        let spec = qwen38_27b_spec();
        spec.validate().unwrap();
        assert_eq!(spec.n_full_attention_layers(), 16);
        assert_eq!(spec.n_linear_layers(), 48);

        let elfs = vec![("GAUDIUM_GDNS_ELF".into(), "cafebabe".into())];
        let pf_lin = build_prefill_linear_layer_graph(&spec, 32, &elfs);
        assert!(pf_lin
            .persistent_tensor_specs()
            .iter()
            .any(|t| t.0 == "L_S_in"));
        assert!(pf_lin
            .persistent_tensor_specs()
            .iter()
            .any(|t| t.0 == "L_S_out"));

        let step = build_decode_step_graph(&spec, 4096, 1, &elfs);
        let specs = step.persistent_tensor_specs();
        let kv_count = specs.iter().filter(|t| is_kv_tensor(&t.0)).count();
        let gdn_s_in = specs.iter().filter(|t| t.0.ends_with("_S_in")).count();
        let gdn_cs_in = specs.iter().filter(|t| t.0.ends_with("_cs_in")).count();
        assert_eq!(kv_count, 32); // 16 full-attention layers * 2
        assert_eq!(gdn_s_in, 48); // 48 linear-attention layers
        assert_eq!(gdn_cs_in, 48);
    }
}
