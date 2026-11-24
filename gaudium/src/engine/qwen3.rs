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
use crate::engine::qwen3_traffic;
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

pub mod prefill;
// prefix reuse across requests: kept positions, their device slots
#[path = "prefix_cache.rs"]
pub mod prefix_cache;
// prefix reuse from host memory: what is copied between the device and the store, and when
#[path = "host_tier.rs"]
pub mod host_tier;
#[path = "request_reset.rs"]
mod request_reset;
// [kv8] FP8 K/V caches of the trunk's attention
#[path = "kv8.rs"]
pub mod kv8;
mod vision_hooks;

/// KV cache chunk length (matches `CHUNK = 4096` in `attn_prep_gaudi2.c`).
pub const KV_CHUNK: u64 = 4096;
/// Maximum prefill tile width pre-planned into the activation scratch arena.
pub const MAX_PREFILL_TILE: u64 = prefill::TILE_MAX;
/// Minimum pre-planned shared workspace bytes so lazily compiled remainder tiles never need HBM allocs.
const MIN_SHARED_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
/// Reusable host pinned staging buffer size for chunked weight uploads.
const STAGE_BUF_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub max_ctx: u64,
    pub cache_dir: Option<PathBuf>,
    pub no_cache: bool,
    pub prefill_tile: u64,
    /// [prefill] Prefill levers, `k=v[,k=v..]` (`engine/qwen3/prefill.rs`); `None`:
    /// `GAUDIUM_PREFILL`, else the defaults.
    pub prefill: Option<String>,
    pub verbose: bool,
    /// Largest image (in patches) the vision tower is planned for; 0 = no image input.
    pub vision_max_patches: u64,
    /// Image tokens one request may hold (0 = the tokens of one largest image).
    pub vision_max_tokens: u64,
    /// Patch counts whose tower recipes are compiled at start-up (others on first use).
    pub vision_precompile: Vec<u64>,
    /// Further patch counts whose tower recipes `build_cache` compiles into the recipe cache
    /// (image sizes a load compiles on their first use). A load ignores them.
    pub vision_cache_sizes: Vec<u64>,
    /// Evaluate the tower's GELU with the BF16 kernel (default: in F32).
    pub vision_gelu_bf16: bool,
    /// Speculative decoding (`--spec`); `None` = the non-speculative `K = 0` engine.
    pub spec: Option<spec::SpecConfig>,
    /// [moe/fp8] Device precision of the projection / MLP weight matrices (`engine/fp8_graph.rs`).
    pub precision: crate::engine::fp8_graph::WeightPrecision,
    /// [moe/fp8] Activation-scale table of the FP8 graphs. Default: `GAUDIUM_FP8_ACT`, else
    /// `<model_dir>-gaudium-pc/act_scales_lc.txt`.
    pub fp8_act: Option<PathBuf>,
    /// [moe/fp8] Optional directory caching the quantised FP8 matrices (or `GAUDIUM_FP8_CACHE`).
    pub fp8_cache: Option<PathBuf>,
    /// [moe/fp8] Calibration load: BF16 graphs, every prefill records the GEMM input sites.
    pub fp8_calibrate: bool,
    /// [moe] Routed MoE form: stacked BF16 experts read through the vendor expert-select node
    /// (decode reads only the routed experts); default: the dense-masked wide MLP.
    pub moe_routed: bool,
    /// Prompt-tail tile lengths `1..=N` whose prefill recipes are loaded (compiled on a recipe
    /// cache miss) while the engine loads, so the first request of a prompt length does not pay
    /// for them inside its time to first token. 0 = on first use (the default).
    pub prefill_warm_rows: u64,
    /// Per-request device state reset: `auto` (default), `dirty` or `full`
    /// (`engine/request_reset.rs`); `None` reads `GAUDIUM_REQUEST_RESET`.
    pub request_reset: Option<String>,
    /// [step] Step-recipe options: DeltaNet verify state, drafter precision / readout width,
    /// device traces (`engine/step_opts.rs`).
    pub step: crate::engine::step_opts::StepOpts,
    /// Prefix reuse across requests: positions of the previous prompt whose recurrent state is
    /// kept on the device so that a later request resumes there (`engine/prefix_cache.rs`).
    /// 0 = off. A load keeps as many as fit beside the rest of its plan.
    pub prefix_slots: u64,
    /// The host-RAM tier of prefix reuse: the server's store (`cache_ram.rs`). `None`: this
    /// engine keeps nothing in host memory.
    pub host_cache: Option<Arc<crate::cache_ram::SharedCache>>,
    /// [kv8] Element type of the trunk's K/V caches (`--kv-cache`).
    pub kv_dtype: crate::verified::KvDtype,
    /// [kv8] The K/V calibration an FP8 load builds its scale tables from (`--kv-scales`);
    /// `None`: `<model_dir>-gaudium-pc/kv_scales.txt`.
    pub kv_scales: Option<PathBuf>,
    /// [kv8] Calibration load (`gaudium --kv-calibrate`): BF16 caches, read back after every
    /// prefill for their per-head maxima.
    pub kv_calibrate: bool,

    /// Requests served at once (`--parallel`): lanes with their own K/V rows, recurrent state
    /// and prefix table, decoded together in batched steps. 1 = one request at a time.
    pub lanes: u64,
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
            prefill: None,
            verbose: false,
            vision_max_patches: 0,
            vision_max_tokens: 0,
            vision_precompile: Vec::new(),
            vision_cache_sizes: Vec::new(),
            vision_gelu_bf16: false,
            spec: None,
            precision: crate::engine::fp8_graph::WeightPrecision::Auto,
            fp8_act: None,
            fp8_cache: None,
            fp8_calibrate: false,
            moe_routed: false,
            prefill_warm_rows: 0,
            request_reset: None,
            step: crate::engine::step_opts::StepOpts::from_env(),
            prefix_slots: 0,
            host_cache: None,
            kv_dtype: crate::verified::KvDtype::KvBf16,
            kv_scales: None,
            kv_calibrate: false,

            lanes: 1,
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
    /// Teacher forcing for exactness checks: when set, generated position `i` continues with
    /// `force_tokens[i]` instead of the engine's own greedy token, and every position where the
    /// two differ is recorded in `Qwen3GenReport::forks` with both BF16 logits.
    pub force_tokens: Option<Vec<i32>>,
    /// Sampled speculative requests: the drafter draws its drafts with the verify's position
    /// noise on its own scores (DSpine's coupled walk). Never changes the tokens, only how many
    /// drafts are accepted. `false`: the drafter's greedy chain.
    pub coupled: bool,
    /// Coupled drafts also apply top-k / top-p / min-p to the drafter's candidates.
    pub draft_trunc: bool,
    /// Debug check: every row the card samples is sampled again on the host from the card's own
    /// logits with the bit-exact CPU twin; rows and mismatches are counted in the report.
    /// Runs one launch at a time.
    pub sample_check: bool,
    /// Run this request on the non-speculative `K = 0` path of an engine loaded with `--spec`
    /// (the exactness reference of the speculative stream, without a second engine load).
    pub non_speculative: bool,
    /// Measurement runs on an engine loaded with several `--draft-k` widths: verify exactly this
    /// many drafts per launch (one of the listed widths) instead of the width policy's choice.
    pub spec_k: Option<u64>,
    /// [step] Measurement: verify exactly this many drafts in every launch of the request (one
    /// of the engine's `--draft-k` widths) instead of letting the width policy choose.
    pub force_draft_k: Option<u64>,
    /// [step] Run the `K = 0` decode of this request as the plain loop (one host round trip per
    /// token) even where the enqueue-ahead loop applies: the reference of `--compare-k0`.
    pub k0_plain: bool,
    /// The request may resume its prefill from state the engine holds for a prefix of its prompt
    /// (`false`: a cold prefill from position 0, the reference a resumed one is identical to).
    pub cache_prompt: bool,
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
            force_tokens: None,
            coupled: true,
            draft_trunc: true,
            sample_check: false,
            non_speculative: false,
            spec_k: None,
            force_draft_k: None,
            k0_plain: false,
            cache_prompt: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
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
    /// Teacher-forced positions where the engine's greedy token differed from the forced one.
    pub forks: Vec<ForkRecord>,
    /// Decode launches after the first token (`K = 0`: one per token).
    pub launches: u64,
    /// Decode tokens per launch (`K = 0`: 1.0).
    pub tokens_per_launch: f64,
    /// Decode method label, e.g. `K=0 BF16 greedy` or `dspine K=15 BF16 greedy`.
    pub method: String,
    /// Speculative runs: `P(draft i accepted | verified)` per draft position; empty at `K = 0`.
    pub accept_per_pos: Vec<f64>,
    /// Speculative runs: drafts verified over the request, and how many of them were accepted.
    pub drafted: u64,
    pub draft_accepted: u64,
    /// `sample_check`: rows sampled again on the host, and how many differed from the card.
    pub sample_check_rows: u64,
    pub sample_check_mismatches: u64,
    /// Part of `ttft_ms` spent loading or compiling recipes this request used for the first
    /// time in the process (a prefill tile length, a step family).
    pub ttft_recipe_ms: f64,
    /// Prompt rows taken over from the previous request (not prefilled); 0 = a cold prefill.
    /// `prefill_tok_s` counts the rows that were prefilled.
    pub reused_tokens: u64,
    /// Rows of the prompt that equal the prompt the engine held (0 without prefix reuse).
    pub prefix_common: u64,
    /// Where the reused rows came from, and what the host-RAM tier copied for this request
    /// (`None`: the engine has no host tier).
    pub host_tier: Option<host_tier::TierReport>,
}

/// One greedy-vs-reference disagreement observed under teacher forcing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForkRecord {
    /// Index into the generated stream (0 = first generated token).
    pub pos: usize,
    pub engine_token: i32,
    pub forced_token: i32,
    /// BF16 logits of the two tokens at this position, widened to f32.
    pub engine_logit: f32,
    pub forced_logit: f32,
}

/// What a request's prefill left (`Qwen3Engine::run_prefill`).
pub(super) enum Prefilled {
    /// The first token is read out.
    First(PrefillFirst),
    /// The request was cancelled during its prefill (its report).
    Cancelled(Qwen3GenReport),
    /// A one-row prompt left to its first decode step.
    OneRow { gdn_parity: usize, ttft_recipe_ms: f64 },
}

/// The end of a prefill that read out the first token.
pub(super) struct PrefillFirst {
    /// The token as the readout gave it (before any teacher forcing).
    pub first_tok: i32,
    pub first_logits_hbm: u64,
    pub ttft_ms: f64,
    pub prefill_tok_s: f64,
    pub ttft_recipe_ms: f64,
    /// Ping-pong buffer the recurrent state is in after the prefill.
    pub gdn_parity: usize,
    /// A sampled request with the on-card sampler.
    pub samp_on: bool,
}

#[path = "qwen3_samp.rs"]
mod samp_rt;
#[path = "spec.rs"]
pub mod spec;
// [moe hook] routed-form prefill: the routed-experts recipe of a tile
#[path = "moe_rt.rs"]
mod moe_rt;
// [moe hook] recipe pre-compilation without the card
#[path = "qwen3_build_cache.rs"]
mod build_cache_rt;
pub use build_cache_rt::build_cache;
// [step] device-trace windows of the decode loops
#[path = "qwen3_halfprobe.rs"]
mod halfprobe;
#[path = "qwen3_prof.rs"]
mod prof_rt;
// [prefill] two tiles in flight
#[path = "qwen3_wave.rs"]
mod wave_rt;
// [step] enqueue-ahead greedy K=0 decode (token fed back on the card)
#[path = "qwen3_k0pipe.rs"]
mod k0pipe;
// [lanes] several requests at once, batched steps (`--parallel`)
#[path = "qwen3_lanes.rs"]
pub mod lanes_rt;

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
pub(crate) fn add_fused_add_norm(
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
pub(crate) fn add_linear_bf16(
    gb: &mut GraphBuilder,
    x: TensorId,
    w_name: &str,
    k: u64,
    n: u64,
    t: u64,
    y_name: &str,
) -> TensorId {
    // [fp8 hook] the quantisable matrices of an FP8 graph: site cast + fp8_gemm (same names)
    if gb.fp8_weights() && crate::engine::fp8_graph::is_fp8_matrix(w_name) {
        return crate::engine::fp8_graph::add_linear_fp8(gb, x, w_name, k, n, t, y_name);
    }
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

    // [moe hook] router + dense-masked wide MLP + layer tail replace the dense SwiGLU MLP
    if spec.is_moe() {
        return crate::engine::moe::append_moe_mlp(
            gb,
            spec,
            x2,
            h2,
            lpfx,
            t,
            fused_anorm,
            wnext_norm,
        );
    }

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
    let dh = spec.head_dim;
    gb.set_fp8_weights(spec.fp8_weights); // [fp8 hook]
    let (qg, k, v) = append_attn_proj(gb, spec, x, h_pre, lpfx, t);
    let core = append_attn_core(gb, spec, qg, k, v, cursor, rope_c, rope_s, lpfx, lpfx, t, nch);
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

/// The input norm (unless fused into the previous layer: `h_pre`) and the q / K / V projections
/// of a full-attention layer over `t` rows: `(qg {qg_dim, t}, k {kv_dim, t}, v {kv_dim, t})`.
pub(crate) fn append_attn_proj(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    lpfx: &str,
    t: u64,
) -> (TensorId, TensorId, TensorId) {
    let hid = spec.hidden;
    let qg_dim = spec.qg_dim();
    let kvd = spec.kv_dim();
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
    (qg, k, v)
}

/// The attention core of a full-attention layer over `t` rows at `cursor`: q / K / V preparation
/// (K and V written into the cache), the chunked scores over `nch` chunks, softmax, PV, the
/// combine of the chunk partials and the output gate -> `core {nh * dh, t}`. `wpfx` names the
/// layer's weights; `spfx` the core's own tensors and nodes (the cache among them): the layer's
/// prefix for one stream, one prefix per lane in a batched step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_attn_core(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    qg: TensorId,
    k: TensorId,
    v: TensorId,
    cursor: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    wpfx: &str,
    spfx: &str,
    t: u64,
    nch: u64,
) -> TensorId {
    let nh = spec.n_heads;
    let nkv = spec.n_kv_heads;
    let dh = spec.head_dim;
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

    let qn_w = gb.get_or_add_persistent_tensor(format!("{wpfx}qn_w"), SynDataType::Float, &[dh]);
    let kn_w = gb.get_or_add_persistent_tensor(format!("{wpfx}kn_w"), SynDataType::Float, &[dh]);
    // [kv8] the cache dtype of this layer (q block, caches and P) and its scale table (the
    // layer's, shared by every slot)
    let kv = kv8::AttnKv::of(gb, spec, wpfx);
    let qb = gb.add_persistent_tensor(format!("{spfx}qb"), kv.dt, &[dh, r, b]);
    let kcache =
        gb.add_persistent_tensor(format!("{spfx}kcache"), kv.dt, &[dh, KV_CHUNK, b]);
    let vcache =
        gb.add_persistent_tensor(format!("{spfx}vcache"), kv.dt, &[dh, KV_CHUNK, b]);
    let prep_params = u32_params(&[0u32]); // coff = 0
    gb.add_node(
        g_aprep,
        format!("{spfx}attn_prep"),
        &kv.inputs(&[qg, k, v, qn_w, kn_w, rope_c, rope_s, cursor]),
        &[qb, kcache, vcache],
        &prep_params,
    );

    let scores = gb.add_persistent_tensor(
        format!("{spfx}scores"),
        SynDataType::Float,
        &[KV_CHUNK, r, b],
    );
    gb.op_batch_gemm(format!("{spfx}qk"), qb, kcache, scores, false, true);

    let pm = gb.add_persistent_tensor(format!("{spfx}P"), kv.dt, &[KV_CHUNK, r, b]);
    let stats =
        gb.add_persistent_tensor(format!("{spfx}stats"), SynDataType::Float, &[64, 2, r, b]);
    let sm_params = u32_params(&[t as u32, nch as u32, 0u32, 0u32]);
    gb.add_node(
        "gaudium_attn_softmax_chunk",
        format!("{spfx}softmax"),
        &kv.inputs(&[scores, cursor]),
        &[pm, stats],
        &sm_params,
    );

    let ao_part =
        gb.add_persistent_tensor(format!("{spfx}ao_part"), SynDataType::Float, &[dh, r, b]);
    gb.op_batch_gemm(format!("{spfx}pv"), pm, vcache, ao_part, false, false);

    let ao = gb.add_persistent_tensor(format!("{spfx}ao"), SynDataType::Float, &[dh, r, nkv]);
    let comb_params = u32_params(&[nch as u32]);
    gb.add_node(
        g_acomb,
        format!("{spfx}combine"),
        &[ao_part, stats],
        &[ao],
        &comb_params,
    );

    let core = gb.add_persistent_tensor(format!("{spfx}core"), SynDataType::Bf16, &[nh * dh, t]);
    gb.add_node(g_apost, format!("{spfx}attn_post"), &[ao, qg], &[core], &[]);
    core
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
    gb.set_fp8_weights(spec.fp8_weights); // [fp8 hook]
    let (qkvz, ba) = append_linear_proj(gb, spec, x, h_pre, lpfx, t);
    let core = append_gdn_core(gb, spec, qkvz, ba, slot, lpfx, lpfx, t);
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

/// The input norm (unless fused into the previous layer: `h_pre`) and the input projections of a
/// Gated DeltaNet layer over `t` rows: `(qkvz {qkvz_dim, t}, ba {ba_dim, t})`.
pub(crate) fn append_linear_proj(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    lpfx: &str,
    t: u64,
) -> (TensorId, TensorId) {
    let hid = spec.hidden;
    let qkvz_dim = spec.gdn_qkvz_dim();
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
    (qkvz, ba)
}

/// The recurrence of a Gated DeltaNet layer over `t` rows: the causal conv (its state `cs_in` ->
/// `cs_out`), the gates, the gated delta rule (state `S_in` -> `S_out`, both read and written at
/// checkpoint slot `slot`) and the gated norm -> `core {hv * dl, t}`. `wpfx` names the layer's
/// weights; `spfx` the core's own tensors and nodes (the states among them): the layer's prefix
/// for one stream, one prefix per lane in a batched step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_gdn_core(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    qkvz: TensorId,
    ba: TensorId,
    slot: TensorId,
    wpfx: &str,
    spfx: &str,
    t: u64,
) -> TensorId {
    let hv = spec.gdn_v_heads;
    let dl = spec.gdn_v_dim;
    let qkv = spec.gdn_qkv_dim();
    let hvp = spec.gdn_hvp();

    let conv_w =
        gb.get_or_add_persistent_tensor(format!("{wpfx}conv_w"), SynDataType::Float, &[qkv, 4]);
    let conv_b =
        gb.get_or_add_persistent_tensor(format!("{wpfx}conv_b"), SynDataType::Float, &[qkv]);
    let cs_in = gb.add_persistent_tensor(format!("{spfx}cs_in"), SynDataType::Float, &[qkv, 3, 1]);
    let cs_out = gb.add_persistent_tensor(format!("{spfx}cs_out"), SynDataType::Float, &[qkv, 3]);
    let cs_ckpt =
        gb.add_persistent_tensor(format!("{spfx}cs_ckpt"), SynDataType::Float, &[qkv, 3, 1]);

    let q = gb.add_persistent_tensor(format!("{spfx}q"), SynDataType::Float, &[dl, hv, t]);
    let k = gb.add_persistent_tensor(format!("{spfx}k"), SynDataType::Float, &[dl, hv, t]);
    let v = gb.add_persistent_tensor(format!("{spfx}v"), SynDataType::Float, &[dl, hv, t]);
    let zero_flag = u32_params(&[0u32]);
    gb.add_node(
        "gaudium_gdn_prep",
        format!("{spfx}gdn_prep"),
        &[qkvz, conv_w, cs_in, conv_b, slot],
        &[q, k, v, cs_out, cs_ckpt],
        &zero_flag,
    );

    let a_log = gb.get_or_add_persistent_tensor(format!("{wpfx}A_log"), SynDataType::Float, &[hvp]);
    let dt_bias =
        gb.get_or_add_persistent_tensor(format!("{wpfx}dt_bias"), SynDataType::Float, &[hvp]);
    let eg = gb.add_persistent_tensor(format!("{spfx}eg"), SynDataType::Float, &[hvp, t]);
    let beta = gb.add_persistent_tensor(format!("{spfx}beta"), SynDataType::Float, &[hvp, t]);
    gb.add_node(
        "gaudium_gdn_gate",
        format!("{spfx}gdn_gate"),
        &[ba, a_log, dt_bias],
        &[eg, beta],
        &[],
    );

    let s_in =
        gb.add_persistent_tensor(format!("{spfx}S_in"), SynDataType::Float, &[64, 256, hv, 1]);
    let s_out =
        gb.add_persistent_tensor(format!("{spfx}S_out"), SynDataType::Float, &[64, 256, hv]);
    let s_ckpt = gb.add_persistent_tensor(
        format!("{spfx}S_ckpt"),
        SynDataType::Float,
        &[64, 256, hv, 1],
    );
    let o = gb.add_persistent_tensor(format!("{spfx}o"), SynDataType::Float, &[dl, hv, t]);
    gb.add_node(
        "gaudium_gdn_f32",
        format!("{spfx}gdn"),
        &[s_in, q, k, v, eg, beta, slot],
        &[o, s_out, s_ckpt],
        &zero_flag,
    );

    let norm_w =
        gb.get_or_add_persistent_tensor(format!("{wpfx}norm_w"), SynDataType::Float, &[dl]);
    let core = gb.add_persistent_tensor(format!("{spfx}core"), SynDataType::Bf16, &[hv * dl, t]);
    gb.add_node(
        "gaudium_gdn_post",
        format!("{spfx}gdn_post"),
        &[o, qkvz, norm_w],
        &[core],
        &zero_flag,
    );
    core
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
    build_head_t1_graph_as(spec, None)
}

/// The readout of a sampled request (`g_qwen_shead_T1`): the on-card sampler in place of the
/// argmax, at the sequence index in `samp_pos0`. `elf_hashes` = the sampler kernels.
pub fn build_sampled_head_t1_graph(
    spec: &ModelSpec,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    build_head_t1_graph_as(spec, Some(elf_hashes))
}

fn build_head_t1_graph_as(spec: &ModelSpec, sampled: Option<&[(String, String)]>) -> GraphBuilder {
    let mut gb = GraphBuilder::new(if sampled.is_some() {
        "g_qwen_shead_T1"
    } else {
        "g_qwen_head_T1"
    });
    for (k, h) in sampled.unwrap_or(&[]) {
        gb.record_kernel_elf_hash(k, h);
    }
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[spec.hidden, 1]);
    let h_final = add_rmsnorm(&mut gb, spec, x_in, "w_final", "h_final", 1);
    let lm_w = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    let logits = add_linear_bf16(&mut gb, h_final, lm_w, spec.hidden, spec.vocab, 1, "logits");
    if sampled.is_some() {
        let pos0 =
            gb.add_persistent_tensor(crate::engine::sampler::SAMP_POS0, SynDataType::Int32, &[1]);
        let at = crate::engine::sampler::SamplerAt {
            base: pos0,
            off: 0,
            role: 0,
        };
        crate::engine::sampler::append_sampler(&mut gb, logits, spec.vocab, 1, &at, "tokens");
        return gb;
    }
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
    build_decode_step_graph_as(spec, lmax, nch, elf_hashes, false)
}

/// The decode step of a sampled request (`g_qwen_sstep_C{nch}`): the same trunk, the on-card
/// sampler in place of the argmax (the row samples the token at sequence index `cursor + 1`).
/// `elf_hashes` must include the sampler kernels.
pub fn build_sampled_decode_step_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    build_decode_step_graph_as(spec, lmax, nch, elf_hashes, true)
}

fn build_decode_step_graph_as(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(if sampled {
        format!("g_qwen_sstep_C{nch}")
    } else {
        format!("g_qwen_step_C{nch}")
    });
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
    if sampled {
        let at = crate::engine::sampler::SamplerAt {
            base: cursor,
            off: 1,
            role: 0,
        };
        crate::engine::sampler::append_sampler(&mut gb, logits, spec.vocab, 1, &at, "tokens");
        return gb;
    }
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
    /// The FP8 inputs this load used (the activation-scale table as parsed, the cached
    /// matrices as found): part of the identity the host-RAM tier stamps entries with.
    fp8_id: String,

    // Recipes (must drop before `dev`)
    head_t1: CompiledRecipe,
    step_recipes: Vec<CompiledRecipe>, // index = nch - 1
    embed_recipes: HashMap<u64, CompiledRecipe>,
    attn_layer_recipes: HashMap<(u64, u64), CompiledRecipe>, // (T, nch)
    lin_layer_recipes: HashMap<u64, CompiledRecipe>,         // T
    /// [prefill] two-tile wavefront recipes of the full tile (empty: the wavefront is off)
    wave_recipes: HashMap<prefill::WaveKey, CompiledRecipe>,
    // [moe hook] routed-experts prefill recipes (owns recipes: must drop before `dev`)
    moe_rx: moe_rt::RoutedPrefill,

    // Reusable pinned host staging buffers (must drop before `dev`)
    stage_io: PinnedBuf, // small scalar + logits D2H/H2D

    // HBM address maps (all sub-allocated from sealed arenas owned by `dev`)
    weights_addr: BTreeMap<String, u64>,
    kv_addr: BTreeMap<String, (u64, u64)>, // L{l}_kcache / vcache -> (addr, bytes)
    gdn_state: BTreeMap<usize, GdnLayerState>, // linear layer idx -> ping-pong S & cs buffers
    /// [prefill] tile-recipe forms, scratch region and per-recipe scratch layouts
    pf: prefill::TileRt,
    pf_x0: u64,
    pf_x1: u64,
    pf_ids: u64,
    pf_cursor: u64,
    head_scratch: BTreeMap<String, u64>,
    step_launches: Vec<[PreparedLaunch; 2]>, // [nch - 1][gdn_parity]
    /// [step] the same recipes bound for enqueue-ahead K=0 decode: parity `p` reads its token
    /// from `[step_ids, step_tokens][p]` and its cursor from `[step_cursor, step_cursor_b][p]`,
    /// and writes its token to the other token buffer. Empty: the plain loop only.
    step_fb_launches: Vec<[PreparedLaunch; 2]>,
    step_cursor_b: u64,
    step_ids: u64,
    step_cursor: u64,
    step_tokens: u64,
    step_logits: u64,
    hbm_bytes: [u64; 5],
    /// Per-request reset bookkeeping (mode, KV high-water mark, last reset time).
    reset: request_reset::ResetState,
    /// Prefix reuse: the table of kept positions, the device buffers of its slots, and what
    /// the running request does with them.
    prefix: prefix_cache::PrefixCache,
    prefix_slots: Vec<prefix_cache::SlotAddrs>,
    prefix_plan: prefix_cache::Plan,
    prefix_images: Vec<prefix_cache::ImageId>,
    /// Prefix reuse from host memory (`engine/host_tier.rs`) and the pinned ring its copies go
    /// through (host memory mapped for the device: must drop before `dev`).
    host: Option<host_tier::HostTier>,
    xfer: Option<crate::synapse::xfer::XferRing>,
    /// Print recipe / timing diagnostics (`Qwen3Config::verbose`).
    verbose: bool,
    live: LiveReporter,
    // Speculative decoding runtime (owns recipes: must drop before `dev`)
    specdec: Option<spec::SpecRuntime>,
    // On-card sampler of the non-speculative path (owns recipes: must drop before `dev`);
    // `None`: the vocabulary or the kernel directory cannot carry it (host sampler instead).
    samp: Option<samp_rt::SampRt>,

    // Image input (recipes + pinned staging inside; must drop before `dev`)
    vision: Option<crate::vision::vit::VisionTower>,
    vision_req: vision_hooks::VisionRequestState,
    /// [fp8 hook] calibration load: running maxima of the GEMM input sites
    calib: Option<crate::engine::fp8_graph::ActCalibrator>,
    /// [kv8] K/V calibration load: running per-head maxima of the trunk's caches
    kv_calib: Option<crate::core::kv_cache::KvCalibration>,
    /// [kv8] What the FP8 scale tables were made from (`none` for BF16 caches): part of the
    /// identity the host-RAM tier stamps entries with.
    kv8_id: String,
    /// [step] device-trace windows (`--profile-steps`)
    prof: Option<prof_rt::ProfRt>,
    /// [lanes] the lanes of an engine loaded with `--parallel N > 1` (owns recipes: must drop
    /// before `dev`); `None`: one request at a time.
    lanes: Option<lanes_rt::LaneRt>,

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
        Self::load_or_plan(cfg, false)?
            .ok_or_else(|| "internal error: the load produced no engine".to_string())
    }

    /// Fills the recipe cache for `cfg` and prints its exact device memory plan WITHOUT opening
    /// the card (graph compilation needs only an initialized SynapseAI): every recipe a load
    /// compiles, plus the step and prefill recipes of every KV chunk count that a load only takes
    /// from the cache. Safe to run while another process owns the card.
    pub fn build_cache(cfg: &Qwen3Config) -> Result<(), String> {
        Self::load_or_plan(cfg, true).map(|_| ())
    }

    /// `plan_only`: stop after the recipes and the memory plan (`Ok(None)`), no device.
    fn load_or_plan(cfg: &Qwen3Config, plan_only: bool) -> Result<Option<Self>, String> {
        let t_start = Instant::now();
        let live = if plan_only {
            LiveReporter::disabled()
        } else {
            LiveReporter::from_env("gaudium-rust", env!("CARGO_PKG_VERSION"), "engine", true)
        };
        let model_name = cfg
            .model_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("gaudium-model")
            .to_string();
        live.set_model(&model_name);
        live.set_phase("loading", "opening device and compiling recipes");

        let mut spec = ModelSpec::from_config_file(cfg.model_dir.join("config.json"))
            .map_err(|e| format!("load config.json: {e}"))?;
        spec.validate()
            .map_err(|e| format!("verified ModelSpec validation failed: {e}"))?;
        // [moe/fp8 hook] run precision (and the FP8 calibration table) before any graph is built
        let fp8_act_table = crate::engine::fp8_pack::resolve_precision(cfg, &mut spec)?;
        // [kv8] the trunk's K/V dtype (and the FP8 calibration), before any graph is built
        let kv8_cal = kv8::resolve(cfg, &mut spec)?;

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
            // [moe hook] Qwen3.5 MoE (DH=256 NKV=2 NH=16 HK=16 HV=32, dense-masked experts)
            ModelFamily::Qwen35Moe if crate::engine::moe::supported_shape(&spec) => {
                crate::engine::moe::QWEN35_MOE_ELFS
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
        let reset_mode = request_reset::ResetMode::from_config(cfg.request_reset.as_deref())?;
        // [lanes] several requests at once: lanes need batched steps for this engine and its
        // drafter (the verified choice the server makes too); a calibration load has one lane
        let lane_drafter = match cfg.spec.as_ref().map(|sc| &sc.method) {
            None => crate::verified::LaneDrafter::LdNone,
            Some(spec::SpecMethod::DSpine { .. }) => crate::verified::LaneDrafter::LdDSpine,
            Some(_) => crate::verified::LaneDrafter::LdOther,
        };
        // (a calibration load: of the FP8 weights' activation sites, or of the K/V caches)
        let lanes_ok = crate::verified::ln_lanes_ok(
            crate::engine::lanes_supported(&spec),
            lane_drafter,
            cfg.fp8_calibrate || cfg.kv_calibrate,
        );
        let n_lanes =
            crate::verified::ln_served_lanes(cfg.lanes, crate::engine::MAX_LANES, lanes_ok) as usize;
        if (n_lanes as u64) < cfg.lanes {
            eprintln!(
                "[gaudium] lanes: {n_lanes} of the {} asked for (this engine, drafter or load has no batched steps for more)",
                cfg.lanes
            );
        }
        let max_ctx = align_up(cfg.max_ctx.max(KV_CHUNK), KV_CHUNK);
        let max_chunks = max_ctx / KV_CHUNK;
        let pf_opts = prefill::PrefillOpts::from_config(cfg)?;

        // 2. Configure TPC-C kernel ELFs & Synapse GC glue BEFORE opening SynapseAI
        let kenv = KernelEnv::configure_from_dir(&cfg.kernels_dir)?;
        let mut required_elfs: Vec<&str> = required_elfs.to_vec();
        if crate::verified::kv_is_fp8(crate::engine::qwen3::kv8::trunk_dtype(&spec)) {
            required_elfs.extend_from_slice(kv8::KV8_ELFS); // [kv8]
        }
        let elf_hashes = kenv.fingerprint_hashes(&required_elfs)?;
        // [prefill] the tile forms of this model; `prefill_tile` is the widest candidate until
        // the HBM plan below settles the width
        let mut pf = prefill::TileRt::new(pf_opts, &spec, &kenv, &elf_hashes)?;
        let prefill_tile = pf.tile();

        // 3. Load SynapseAI & open Gaudi 2 device
        let api =
            Arc::new(SynapseApi::load_default().map_err(|e| format!("load libSynapse.so: {e}"))?);
        let driver_version = api.driver_version().unwrap_or_else(|_| "unknown".into());
        let t_open = Instant::now();
        // Declared before every recipe: recipes are dropped before the session / the device.
        let _compile_session = if plan_only {
            Some(
                crate::synapse::device::CompileSession::open(Arc::clone(&api))
                    .map_err(|e| format!("initialize SynapseAI (no device): {e}"))?,
            )
        } else {
            None
        };
        let dev_opt = if plan_only {
            None
        } else {
            Some(
                HpuDevice::open_gaudi2(Arc::clone(&api))
                    .map_err(|e| format!("open Gaudi 2 device: {e}"))?,
            )
        };
        let (free_hbm, total_hbm) = match &dev_opt {
            Some(dev) => dev
                .memory_info()
                .map_err(|e| format!("query HBM info: {e}"))?,
            None => (0, 0),
        };
        // [step] device traces: the profiler's own device buffer is planned with the rest
        let prof = match &dev_opt {
            Some(dev) if !cfg.step.profile.is_empty() => Some(
                crate::synapse::profiler::Profiler::new(dev.device_id(), cfg.step.profile.clone())
                    .map_err(|e| format!("profiler: {e}"))?,
            ),
            _ => None,
        };
        let prof_bytes = match &prof {
            Some(p) => align_up(
                p.required_bytes()
                    .map_err(|e| format!("profiler memory query: {e}"))?,
                256,
            ),
            None => 0,
        };
        let open_s = t_open.elapsed().as_secs_f64();

        let cache = if cfg.no_cache {
            RecipeCache::disabled()
        } else if let Some(dir) = &cfg.cache_dir {
            RecipeCache::at_dir(dir)
        } else {
            RecipeCache::from_env()
        };

        if cfg.verbose || plan_only {
            if plan_only {
                eprintln!("[gaudium] build-cache: SynapseAI initialized without a device; the card is not opened");
            }
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
        // request sampler parameters (written per sampled request; a few bytes)
        for (name, bytes) in crate::engine::sampler::sampler_param_specs() {
            step_act_specs.insert(name.to_string(), bytes);
        }
        // [step] second cursor buffer of the enqueue-ahead K=0 decode
        step_act_specs.insert("cursor_b".to_string(), 4);
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
        // Cached step recipes are loaded on several threads; the rest compile here in order.
        let mut step_cached = crate::synapse::graph::load_cached_parallel(
            step_gbs.len(),
            crate::synapse::graph::recipe_load_threads(),
            &api,
            &cache,
            &driver_version,
            |i| step_gbs[i].clone(),
        );
        for (idx, gb) in step_gbs.iter().enumerate() {
            let r = match std::mem::replace(&mut step_cached[idx], Ok(None)) {
                Ok(Some(r)) => r,
                _ => gb
                    .compile(Arc::clone(&api), &cache, &driver_version)
                    .map_err(|e| format!("compile decode step (chunks={}): {e}", idx + 1))?,
            };
            if cfg.verbose && (!r.from_cache || idx == 0 || idx + 1 == step_gbs.len()) {
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
        let mut wave_recipes: HashMap<prefill::WaveKey, CompiledRecipe> = HashMap::new();
        let mut embed_recipes: HashMap<u64, CompiledRecipe> = HashMap::new();
        // [moe hook] routed form: the routed-experts prefill recipe of a full tile
        let mut moe_rx = moe_rt::RoutedPrefill::prepare(
            &spec,
            prefill_tile,
            &api,
            &cache,
            &driver_version,
            cfg.verbose,
        )?;
        max_ws = max_ws.max(moe_rx.max_ws());
        // Sampled recipe families (separate from the greedy ones above, which stay as they are).
        // The step of every chunk count is compiled now for a small context of a non-speculative
        // engine; otherwise the first and the last, the others on first use.
        let samp_eager: Vec<u64> = if cfg.spec.is_none() && max_chunks <= 8 {
            (1..=max_chunks).collect()
        } else if max_chunks > 1 {
            vec![1, max_chunks]
        } else {
            vec![1]
        };
        let mut samp = samp_rt::SampRt::prepare(
            &spec,
            &kenv,
            &elf_hashes,
            max_ctx,
            &samp_eager,
            &api,
            &cache,
            &driver_version,
            cfg.verbose,
        )?;
        if let Some(s) = &samp {
            max_ws = max_ws.max(s.max_ws());
        }
        let mut specdec = match &cfg.spec {
            Some(sc) => Some(spec::SpecRuntime::prepare(
                sc,
                &spec,
                &kenv,
                &elf_hashes,
                max_ctx,
                max_chunks,
                prefill_tile,
                &api,
                &cache,
                &driver_version,
                cfg.verbose || plan_only,
                samp.is_some(),
                plan_only,
                &cfg.step,
            )?),
            None => None,
        };
        if let Some(sd) = &specdec {
            max_ws = max_ws.max(sd.max_ws());
        }
        // [lanes] the batched step recipes and their buffers
        let lane_plan = if n_lanes > 1 {
            let p = lanes_rt::LanePlan::prepare(
                n_lanes,
                &spec,
                max_ctx,
                max_chunks,
                &elf_hashes,
                &kenv,
                samp.is_some(),
                &|name| weight_specs.contains_key(name) || is_weight_or_table(name),
                &api,
                &cache,
                &driver_version,
                cfg.verbose || plan_only,
                plan_only,
            )?;
            max_ws = max_ws.max(p.max_ws);
            Some(p)
        } else {
            None
        };
        let recipes_s = t_comp.elapsed().as_secs_f64();
        if cfg.verbose || plan_only {
            eprintln!(
                "[gaudium] core recipes ready in {:.2}s (max_ws={:.2} MiB)",
                t_comp.elapsed().as_secs_f64(),
                max_ws as f64 / (1024.0 * 1024.0)
            );
        }

        // Vision tower (opt-in): planned from its graph descriptions, recipes for the image
        // sizes known now compiled before the HBM plan so their workspace is counted.
        let mut vision = vision_hooks::plan_vision(&spec, cfg)?;
        if let Some(v) = vision.as_mut() {
            max_ws = max_ws.max(v.precompile(&api, &cache, &driver_version, cfg.verbose)?);
            if plan_only && !cfg.vision_cache_sizes.is_empty() {
                let t0 = Instant::now();
                let (hit, compiled, ws) =
                    v.compile_into_cache(&api, &cache, &driver_version, &cfg.vision_cache_sizes)?;
                eprintln!(
                    "[gaudium] vision tower recipes of {} further image sizes: {hit} from the cache, {compiled} compiled ({:.2}s); largest workspace {:.2} MiB",
                    cfg.vision_cache_sizes.len(),
                    t0.elapsed().as_secs_f64(),
                    ws as f64 / (1024.0 * 1024.0)
                );
                if ws > max_ws {
                    return Err(format!(
                        "a vision tower recipe of a smaller image needs a workspace of {ws} bytes, more than the {max_ws} bytes planned from the largest image"
                    ));
                }
            }
        }
        let (vis_weights_bytes, vis_act_bytes) = vision.as_ref().map_or((0, 0), |v| {
            (v.memory_plan().weights_bytes(), v.memory_plan().act_bytes())
        });

        // 6. Compute exact arena sizes & check HBM fit via verified `core::hbm`!
        let align256 = |n: u64| align_up(n.max(256), 256);
        let weights_bytes: u64 =
            weight_specs.values().map(|&b| align256(b)).sum::<u64>() + vis_weights_bytes;

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
        // Speculative steps keep one state checkpoint per verify row: each buffer is a ring of slots.
        let gdn_slots = specdec.as_ref().map_or(1, |sd| sd.gdn_slots());
        // [step] the replay form of the verify rows keeps one recurrent state per buffer
        let s_ring_bytes = s_one_bytes * specdec.as_ref().map_or(1, |sd| sd.gdn_state_slots());
        let cs_ring_bytes = cs_one_bytes * gdn_slots;
        let gdn_state_bytes = n_lin * (2 * align256(s_ring_bytes) + 2 * align256(cs_ring_bytes));
        let kv_bytes: u64 = kv_specs.values().map(|&b| align256(b)).sum::<u64>() + gdn_state_bytes;
        // [lanes] every lane holds what lane 0 holds (its caches and state rings), and the
        // batched steps have buffers of their own
        let kv_all = crate::verified::ln_lane_arena_bytes(kv_bytes, 0, n_lanes as u64);
        let lane_act_bytes = lane_plan.as_ref().map_or(0, |p| p.act_bytes());

        let pf_x_bytes = align256(spec.hidden * prefill_tile * 2);
        let pf_ids_bytes = align256(prefill_tile * 4);
        let pf_cursor_bytes = 256u64;
        let pf_const_bytes = pf.const_bytes();
        // [prefill] the wavefront's second buffer set, when this load can run two tiles in flight
        pf.set_wave_taps(match (&specdec, cfg.fp8_calibrate || spec.is_moe()) {
            (_, true) => None,
            (Some(sd), false) => sd.wave_taps(),
            (None, false) => Some(0),
        });
        let pf_wave_bytes = align256(pf.wave_buf_bytes(&spec, prefill_tile));
        let act_fixed_bytes: u64 = pf_x_bytes * 2
            + pf_ids_bytes
            + pf_cursor_bytes
            + pf_const_bytes
            + pf_wave_bytes
            + head_act_specs.values().map(|&b| align256(b)).sum::<u64>()
            + step_act_specs.values().map(|&b| align256(b)).sum::<u64>()
            + vis_act_bytes
            + lane_act_bytes;
        let spec_bytes = specdec.as_ref().map_or([0u64; 3], |sd| sd.planned_bytes());
        let headroom = 512 * 1024 * 1024;

        // [prefill] the widest tile whose recipe scratch fits beside the rest of this plan, and
        // that width's full-tile recipes (their workspaces are part of the plan)
        let (pf_scratch_bytes, pf_ws) = {
            let other = weights_bytes
                .saturating_add(kv_all)
                .saturating_add(act_fixed_bytes)
                .saturating_add(spec_bytes.iter().sum());
            let fits = |scratch: u64, ws: u64| {
                let ws = align256(max_ws.max(ws).max(MIN_SHARED_WORKSPACE_BYTES));
                match &dev_opt {
                    Some(dev) => dev
                        .check_plan_fits(
                            other.saturating_add(align256(scratch)).saturating_add(ws),
                            headroom,
                        )
                        .is_ok(),
                    // plan only (the card is not opened): the widest tile; a load that finds it
                    // does not fit settles on a narrower one and compiles that width's recipes
                    None => true,
                }
            };
            pf.settle(
                &spec,
                max_ctx,
                max_chunks,
                &elf_hashes,
                &api,
                &cache,
                &driver_version,
                &mut attn_layer_recipes,
                &mut lin_layer_recipes,
                &mut embed_recipes,
                &mut wave_recipes,
                cfg.verbose,
                &fits,
            )?
        };
        let pf_scratch_bytes = align256(pf_scratch_bytes);
        let prefill_tile = pf.tile();
        max_ws = max_ws.max(pf_ws);

        // Last-tile recipes: a prompt whose length is not a multiple of the tile ends in a tile
        // of 1 .. tile-1 rows, which has its own embed / attention / DeltaNet recipes. Those of
        // the first KV chunk (every prompt below one chunk, i.e. most chat requests) are taken
        // from the recipe cache here, on several threads, so a new prompt length does not load
        // or compile them inside a request; `build_cache` compiles them, and also the last-tile
        // attention recipes of the full context to learn the largest workspace one can need.
        if prefill_tile > 1 {
            let t_tail = Instant::now();
            let tails: Vec<u64> = (1..prefill_tile).collect();
            let threads = crate::synapse::graph::recipe_load_threads();
            let n = tails.len();
            // jobs: [attn (T, 1)] ++ [embed T] ++ [lin T, hybrid models only]
            let kinds = if has_gdn { 3 } else { 2 };
            let describe = |pf: &prefill::TileRt, i: usize| -> GraphBuilder {
                let t = tails[i % n];
                match i / n {
                    0 => pf.attn_graph(&spec, t, max_ctx, 1, &elf_hashes),
                    1 => build_embed_graph(&spec, t),
                    _ => pf.lin_graph(&spec, t, &elf_hashes),
                }
            };
            let mut loaded = crate::synapse::graph::load_cached_parallel(
                kinds * n,
                threads,
                &api,
                &cache,
                &driver_version,
                |i| describe(&pf, i),
            );
            let (mut hits, mut built, mut tail_ws) = (0u64, 0u64, 0u64);
            for (i, slot) in loaded.iter_mut().enumerate() {
                let t = tails[i % n];
                let r = match std::mem::replace(slot, Ok(None)) {
                    Ok(Some(r)) => Some(r),
                    _ if plan_only => Some(
                        describe(&pf, i)
                            .compile(Arc::clone(&api), &cache, &driver_version)
                            .map_err(|e| format!("compile last-tile recipe T={t}: {e}"))?,
                    ),
                    _ => None,
                };
                let Some(r) = r else { continue };
                if r.from_cache {
                    hits += 1;
                } else {
                    built += 1;
                }
                tail_ws = tail_ws.max(r.ws_size);
                // a tile recipe's scratch tensors get their places in the planned region
                match i / n {
                    0 => {
                        let gb = describe(&pf, i);
                        pf.register_attn(&gb, t, 1)?;
                        attn_layer_recipes.insert((t, 1), r);
                    }
                    1 => {
                        embed_recipes.insert(t, r);
                    }
                    _ => {
                        let gb = describe(&pf, i);
                        pf.register_lin(&gb, t)?;
                        lin_layer_recipes.insert(t, r);
                    }
                }
            }
            // full-context last tiles: compiled by `build_cache` only, to measure their workspace
            let mut far_ws = 0u64;
            if plan_only && max_chunks > 1 {
                for &t in &tails {
                    let r = pf
                        .attn_graph(&spec, t, max_ctx, max_chunks, &elf_hashes)
                        .compile(Arc::clone(&api), &cache, &driver_version)
                        .map_err(|e| format!("compile last-tile attn T={t} C={max_chunks}: {e}"))?;
                    far_ws = far_ws.max(r.ws_size);
                }
            }
            max_ws = max_ws.max(tail_ws).max(far_ws);
            if cfg.verbose || plan_only {
                eprintln!(
                    "[gaudium] last-tile recipes (1..{} rows, first chunk): {hits} from the cache, {built} compiled, {} left for first use ({:.2}s); largest workspace {:.2} MiB{}",
                    prefill_tile - 1,
                    (kinds * n) as u64 - hits - built,
                    t_tail.elapsed().as_secs_f64(),
                    tail_ws as f64 / (1024.0 * 1024.0),
                    if plan_only && max_chunks > 1 {
                        format!(
                            ", at {max_chunks} chunks {:.2} MiB",
                            far_ws as f64 / (1024.0 * 1024.0)
                        )
                    } else {
                        String::new()
                    }
                );
            }
        }
        let act_bytes: u64 = act_fixed_bytes + pf_scratch_bytes;
        let ws_bytes = align256(max_ws.max(MIN_SHARED_WORKSPACE_BYTES));

        let base_total = weights_bytes
            .saturating_add(kv_all)
            .saturating_add(act_bytes)
            .saturating_add(ws_bytes)
            .saturating_add(prof_bytes)
            .saturating_add(spec_bytes.iter().sum());
        // Prefix reuse: device slots for the recurrent state at kept positions of the previous
        // prompt. They are planned last: everything above (the context first of all) is
        // settled, and the slots take what is left, down to none.
        let pfx_slot_bytes = n_lin * (align256(s_one_bytes) + align256(cs_one_bytes));
        let pfx_why_not = if !has_gdn {
            Some("this model has no recurrent layers to keep a state of")
        } else if spec.is_moe() {
            Some("not validated for mixture-of-experts models")
        } else if cfg.fp8_calibrate {
            Some("a calibration load prefills every row")
        } else if reset_mode != request_reset::ResetMode::Auto {
            Some("the request reset mode zeroes the K/V rows of every request")
        } else if specdec.as_ref().is_some_and(|sd| !sd.prefix_reuse_ok()) {
            Some("not validated for this drafter")
        } else {
            None
        };
        let pfx_wanted = if pfx_why_not.is_none() {
            cfg.prefix_slots
        } else {
            0
        };
        // [lanes] N lanes of the requested context fit, or the load is refused with the numbers
        // (the context of every lane is planned before anything that gives way to it)
        if n_lanes > 1 && dev_opt.is_some() {
            use crate::core::hbm::usable_hbm_bytes;
            let usable = usable_hbm_bytes(total_hbm, total_hbm.saturating_sub(free_hbm), headroom);
            let shared = crate::verified::ln_plan_shared(base_total, n_lanes as u64, kv_bytes);
            if let Err(short) =
                crate::core::lanes::plan_check(n_lanes as u64, kv_bytes, shared, usable)
            {
                let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
                let fit = crate::core::lanes::lanes_fit(kv_bytes, shared, usable);
                return Err(format!(
                    "--parallel {n_lanes} at {max_ctx} tokens per lane does not fit: each lane needs {:.2} GiB (K/V caches and DeltaNet state), the shared part (weights, step buffers, workspace) {:.2} GiB, together {:.2} GiB of the {:.2} GiB the card has usable, {:.2} GiB short. At this context {fit} lane(s) fit; or lower the context per lane (--max-ctx / --model-ctx)",
                    gib(kv_bytes),
                    gib(shared),
                    gib(shared.saturating_add(kv_all)),
                    gib(usable),
                    gib(short),
                ));
            }
        }
        // prefix reuse: the same count of kept positions in every lane
        let lanes_u = n_lanes as u64;
        let pfx_granted = match &dev_opt {
            Some(_) => {
                use crate::core::hbm::{grant_prefix_slots, usable_hbm_bytes};
                let usable =
                    usable_hbm_bytes(total_hbm, total_hbm.saturating_sub(free_hbm), headroom);
                let all = crate::verified::ln_lane_pfx_ask(pfx_wanted, lanes_u);
                let fit = grant_prefix_slots(all, pfx_slot_bytes, base_total, usable);
                let granted = crate::core::lanes::prefix_per_lane(
                    crate::verified::pc_grant_checked(fit, pfx_slot_bytes, base_total, usable),
                    lanes_u,
                );
                if granted < pfx_wanted {
                    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
                    eprintln!(
                        "[gaudium] prefix reuse: {granted} of {pfx_wanted} kept positions{} fit beside the context ({:.2} GiB each; the rest of the plan takes {:.2} of the {:.2} GiB the card has usable){}",
                        if n_lanes > 1 { " per lane" } else { "" },
                        gib(pfx_slot_bytes),
                        gib(base_total),
                        gib(usable),
                        if granted == 0 {
                            ": every request prefills its whole prompt"
                        } else {
                            ""
                        }
                    );
                }
                granted
            }
            // plan only (the card is not opened): what a load asks for
            None => pfx_wanted,
        };
        if let Some(why) = pfx_why_not.filter(|_| cfg.prefix_slots > 0) {
            eprintln!("[gaudium] prefix reuse: off ({why})");
        }
        let pfx_bytes = crate::verified::ln_lane_pfx_bytes(pfx_granted, pfx_slot_bytes, lanes_u);
        let kv_arena_bytes = crate::verified::ln_kv_arena_total(kv_all, pfx_bytes);
        let planned_total = base_total.saturating_add(pfx_bytes);
        // [live] the same plan, buffer by buffer under named parts (the live record's
        // `hbm_detail`, and the lines below the plan total)
        let mem_report = {
            use crate::engine::mem_report::{trunk_weight_part, MemReport};
            let mut r = MemReport::new();
            r.add_map(HbmCat::CatWeights, &weight_specs, trunk_weight_part);
            // [lanes] every lane's caches and rings (lane 0 first)
            for _ in 0..n_lanes {
                r.add_map(HbmCat::CatKv, &kv_specs, |_| "kv-target");
                r.add(
                    HbmCat::CatCheckpoints,
                    "rings",
                    "DeltaNet state rings",
                    n_lin * 2 * align256(s_ring_bytes),
                );
                r.add(
                    HbmCat::CatCheckpoints,
                    "conv-rings",
                    "DeltaNet conv-state rings",
                    n_lin * 2 * align256(cs_ring_bytes),
                );
            }
            if let Some(lp) = lane_plan.as_ref() {
                r.add_map(HbmCat::CatOther, &lp.act_specs, |_| "lanes");
                r.add(HbmCat::CatOther, "lanes", "step inputs and outputs", lp.io_bytes);
                r.set_source(
                    "lanes",
                    format!(
                        "{n_lanes} lanes: the batched decode step's buffers ({n_lanes} slots at {max_ctx} tokens)"
                    ),
                );
            }
            for _ in 0..pfx_granted * lanes_u {
                r.add(
                    HbmCat::CatCheckpoints,
                    "prefix",
                    "kept position: DeltaNet state",
                    n_lin * align256(s_one_bytes),
                );
                r.add(
                    HbmCat::CatCheckpoints,
                    "prefix",
                    "kept position: DeltaNet conv state",
                    n_lin * align256(cs_one_bytes),
                );
            }
            if pfx_granted > 0 {
                r.set_source(
                    "prefix",
                    format!(
                        "prefix reuse: the recurrent state at {pfx_granted} kept positions of the previous prompt{}",
                        if plan_only {
                            " (asked for; a load keeps as many as fit beside the rest)"
                        } else {
                            ""
                        }
                    ),
                );
            }
            // the prefill region, as the buffers of the tile recipe that needs the most of it
            // (every tile recipe lays its scratch out in this one region)
            let arena_items = pf.plan_items(&spec, prefill_tile, max_ctx, max_chunks);
            let arena_sum: u64 = arena_items.iter().map(|x| x.1).sum();
            if arena_sum > 0 && arena_sum <= pf_scratch_bytes {
                for (name, bytes) in &arena_items {
                    r.add(HbmCat::CatScratch, "arena", name, *bytes);
                }
                r.add(
                    HbmCat::CatScratch,
                    "arena",
                    "unused by the largest recipe",
                    pf_scratch_bytes - arena_sum,
                );
            } else {
                r.add(HbmCat::CatScratch, "arena", "arena", pf_scratch_bytes);
            }
            r.set_source(
                "arena",
                format!(
                    "prefill scratch of a {prefill_tile}-row tile at {max_ctx} tokens, shared by every tile recipe"
                ),
            );
            r.add(HbmCat::CatOther, "prefill", "tile rows x2", pf_x_bytes * 2);
            r.add(HbmCat::CatOther, "prefill", "tile ids", pf_ids_bytes);
            r.add(HbmCat::CatOther, "prefill", "tile cursor", pf_cursor_bytes);
            r.add(HbmCat::CatOther, "prefill", "tile constants", pf_const_bytes);
            r.add(
                HbmCat::CatOther,
                "prefill",
                "second tile set (two tiles in flight)",
                pf_wave_bytes,
            );
            r.add_map(HbmCat::CatOther, &head_act_specs, |_| "activations");
            r.add_map(HbmCat::CatOther, &step_act_specs, |_| "activations");
            r.add(HbmCat::CatOther, "workspace", "shared recipe workspace", ws_bytes);
            r.add(HbmCat::CatOther, "profile", "profiler trace buffer", prof_bytes);
            if let Some(v) = vision.as_ref() {
                // the whole tower under one part, so one number says what vision costs
                let plan = v.memory_plan();
                r.add_map(HbmCat::CatOther, &plan.weights, |_| "vision");
                r.add_map(HbmCat::CatOther, &plan.acts, |_| "vision");
                let xbuf = align256(plan.xbuf_bytes);
                r.add(HbmCat::CatOther, "vision", "block rows A", xbuf);
                r.add(HbmCat::CatOther, "vision", "block rows B", xbuf);
                r.add(
                    HbmCat::CatOther,
                    "vision",
                    "image embeddings",
                    align256(plan.emb_bytes),
                );
                r.set_source(
                    "vision",
                    format!(
                        "vision tower, up to {} patches ({} image tokens): weights {:.2} GiB + activations {:.2} GiB",
                        v.max_patches(),
                        v.max_tokens(),
                        vis_weights_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                        vis_act_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                    ),
                );
            }
            r.set_source(
                "activations",
                format!(
                    "single-row decode step and readout: scores and P of every attention layer at {max_ctx} tokens, state checkpoints"
                ),
            );
            if let Some(sd) = specdec.as_ref() {
                r.set_source(
                    "verify",
                    format!(
                        "verify rows of the speculative step: scores and P of every attention layer at {max_ctx} tokens"
                    ),
                );
                let [w, kv, acts] = sd.planned_maps();
                r.add_map(HbmCat::CatWeights, w, |_| "drafter");
                r.add_map(HbmCat::CatKv, kv, |_| "kv-drafter");
                r.add_map(
                    HbmCat::CatOther,
                    acts,
                    crate::engine::mem_report::spec_act_part,
                );
            }
            r
        };
        // every planned byte is in the report exactly once, or the report is not used
        let mem_report_ok = mem_report.total_bytes() == planned_total;
        debug_assert!(mem_report_ok, "the itemised memory report misses the plan");
        if !mem_report_ok {
            eprintln!(
                "[gaudium] the itemised memory report ({} bytes) differs from the plan ({planned_total} bytes): reporting the five categories only",
                mem_report.total_bytes()
            );
        }
        // The whole plan, before the fit check, so a refused configuration shows where it went.
        if cfg.verbose || plan_only {
            let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
            eprintln!(
                "[gaudium] HBM plan: total {:.2} GiB = trunk weights {:.2} + trunk KV {:.2} ({}) + state rings {:.2} ({} slots) + prefix reuse {:.2} ({} kept positions{}) + trunk activations {:.2} + drafter weights {:.2} + drafter caches {:.2} + drafter/verify activations {:.2} + workspace {:.2}{}",
                gib(planned_total),
                gib(weights_bytes),
                gib(kv_all.saturating_sub(gdn_state_bytes * lanes_u)),
                crate::core::kv_cache::kv_dtype_name(kv8::trunk_dtype(&spec)),
                gib(gdn_state_bytes * lanes_u),
                gdn_slots,
                gib(pfx_bytes),
                pfx_granted,
                if n_lanes > 1 {
                    format!(" per lane; {n_lanes} lanes")
                } else {
                    String::new()
                },
                gib(act_bytes),
                gib(spec_bytes[0]),
                gib(spec_bytes[1]),
                gib(spec_bytes[2]),
                gib(ws_bytes),
                if plan_only {
                    " (card not opened: free HBM unknown, headroom 0.50 GiB)".to_string()
                } else {
                    format!(
                        "; card free {:.2} of {:.2} GiB, headroom {:.2}",
                        gib(free_hbm),
                        gib(total_hbm),
                        gib(headroom)
                    )
                }
            );
            if mem_report_ok {
                for line in mem_report.log_lines() {
                    eprintln!("[gaudium] {line}");
                }
            }
        }
        let Some(mut dev) = dev_opt else {
            if pf.opts.half_probe > 0 {
                halfprobe::compile_only(
                    &spec,
                    prefill_tile,
                    max_ctx,
                    &pf.form,
                    &elf_hashes,
                    pf.elf_tile(),
                    &api,
                    &driver_version,
                )?;
            }
            eprintln!(
                "[gaudium] build-cache done in {:.1}s: recipes for max_ctx={} are in the cache; no engine loaded",
                t_start.elapsed().as_secs_f64(),
                max_ctx
            );
            return Ok(None);
        };
        dev.check_plan_fits(planned_total, headroom)
            .map_err(|e| format!("HBM plan check failed: {e}"))?;

        if cfg.verbose {
            eprintln!(
                "[gaudium] HBM plan fits: weights={:.2} GiB, kv+state={:.2} GiB, acts={:.2} MiB, ws={:.2} MiB",
                weights_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                kv_arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                act_bytes as f64 / (1024.0 * 1024.0),
                ws_bytes as f64 / (1024.0 * 1024.0),
            );
            if let Some(v) = vision.as_ref() {
                eprintln!(
                    "[gaudium] HBM plan vision (included above): weights={:.2} MiB, acts={:.2} MiB (max {} patches, {} image tokens)",
                    vis_weights_bytes as f64 / (1024.0 * 1024.0),
                    vis_act_bytes as f64 / (1024.0 * 1024.0),
                    v.max_patches(),
                    v.max_tokens(),
                );
            }
        }

        // 7. Allocate arenas and seal AllocGuard
        let mut weights_arena = dev
            .alloc_arena(weights_bytes, HbmCat::CatWeights, "weights_arena")
            .map_err(|e| format!("alloc weights arena: {e}"))?;
        let mut kv_arena = dev
            .alloc_arena(kv_arena_bytes, HbmCat::CatKv, "kv_state_arena")
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
                let s0 = suballoc(&mut kv_arena, s_ring_bytes)?;
                let s1 = suballoc(&mut kv_arena, s_ring_bytes)?;
                let cs0 = suballoc(&mut kv_arena, cs_ring_bytes)?;
                let cs1 = suballoc(&mut kv_arena, cs_ring_bytes)?;
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

        // prefix reuse: per kept position, the state buffers of every recurrent layer
        let mut prefix_slots: Vec<prefix_cache::SlotAddrs> = Vec::new();
        for _ in 0..pfx_granted {
            let mut m = prefix_cache::SlotAddrs::new();
            for l in 0..(spec.n_layers as usize) {
                if !spec.is_full_attention_layer(l) {
                    let s = suballoc(&mut kv_arena, s_one_bytes)?;
                    let cs = suballoc(&mut kv_arena, cs_one_bytes)?;
                    m.insert(l, (s, cs));
                }
            }
            prefix_slots.push(m);
        }

        // [lanes] lanes 1..N: the same buffers as lane 0's, each with its own prefix table
        let other_lanes = if n_lanes > 1 {
            let ctx_rows = specdec.as_ref().map_or(0, |sd| sd.context_rows());
            let mut sa = |bytes: u64| suballoc(&mut kv_arena, bytes);
            lanes_rt::allocate_lanes(
                n_lanes,
                &spec,
                &kv_specs,
                s_ring_bytes,
                s_one_bytes,
                cs_ring_bytes,
                cs_one_bytes,
                pfx_granted,
                &|| {
                    prefix_cache::PrefixCache::new(
                        prefill_tile,
                        pfx_granted,
                        pfx_slot_bytes,
                        pfx_why_not.is_none(),
                        ctx_rows,
                    )
                },
                &|| request_reset::ResetState::new(reset_mode, cfg.verbose),
                &mut sa,
            )?
        } else {
            Vec::new()
        };

        let pf_x0 = suballoc(&mut act_arena, pf_x_bytes)?;
        let pf_x1 = suballoc(&mut act_arena, pf_x_bytes)?;
        let pf_ids = suballoc(&mut act_arena, pf_ids_bytes)?;
        let pf_cursor = suballoc(&mut act_arena, pf_cursor_bytes)?;

        // [prefill] one scratch region for every tile recipe (+ the cast constant)
        let pf_scratch_base = suballoc(&mut act_arena, pf_scratch_bytes)?;
        let pf_const_one = if pf_const_bytes > 0 {
            suballoc(&mut act_arena, pf_const_bytes)?
        } else {
            0
        };
        pf.set_addresses(pf_scratch_base, pf_const_one);
        if pf_wave_bytes > 0 {
            let base = suballoc(&mut act_arena, pf_wave_bytes)?;
            pf.set_wave_addresses(&spec, prefill_tile, base, [pf_x0, pf_x1], pf_ids, pf_cursor);
        }
        let mut head_scratch = BTreeMap::new();
        for (name, &bytes) in &head_act_specs {
            head_scratch.insert(name.clone(), suballoc(&mut act_arena, bytes)?);
        }
        let mut step_act_addr = BTreeMap::new();
        for (name, &bytes) in &step_act_specs {
            step_act_addr.insert(name.clone(), suballoc(&mut act_arena, bytes)?);
        }
        // [lanes] the batched step's buffers and its inputs / outputs
        let lane_alloc = match lane_plan.as_ref() {
            Some(lp) => {
                let mut act = BTreeMap::new();
                for (name, &bytes) in &lp.act_specs {
                    act.insert(name.clone(), suballoc(&mut act_arena, bytes)?);
                }
                let mut io = BTreeMap::new();
                for name in lanes_rt::io_names(lp.n) {
                    io.insert(name, suballoc(&mut act_arena, lanes_rt::IO_SCALAR)?);
                }
                Some((act, io))
            }
            None => None,
        };
        if let Some(v) = vision.as_mut() {
            v.allocate(&mut weights_arena, &mut act_arena)?;
        }

        if let Some(sd) = &mut specdec {
            sd.allocate(&mut dev)?;
            if cfg.verbose {
                eprintln!(
                    "[gaudium] spec HBM plan: weights={:.2} GiB, drafter kv={:.2} MiB, acts={:.2} MiB, state rings={} slots ({:.2} GiB)",
                    spec_bytes[0] as f64 / (1024.0 * 1024.0 * 1024.0),
                    spec_bytes[1] as f64 / (1024.0 * 1024.0),
                    spec_bytes[2] as f64 / (1024.0 * 1024.0),
                    gdn_slots,
                    gdn_state_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
        }

        if prof_bytes > 0 {
            let addr = dev
                .dmalloc(prof_bytes, HbmCat::CatScratch, "profiler_trace_buffer")
                .map_err(|e| format!("alloc profiler buffer: {e}"))?;
            prof.as_ref()
                .expect("profiler planned")
                .set_user_buffer(addr)
                .map_err(|e| format!("profiler buffer: {e}"))?;
        }
        dev.seal_with_shared_workspace(ws_bytes)
            .map_err(|e| format!("seal HBM allocations: {e}"))?;

        // Every K/V cache and state ring starts as zeros (a masked row must be finite): filled
        // on the device while the weights upload, ahead of any launch on the compute stream.
        let t_zero = Instant::now();
        let mut zero_regions = vec![(kv_arena.base(), kv_arena_bytes)];
        if let Some(sd) = &specdec {
            zero_regions.extend(sd.cache_regions());
        }
        let zeroed_bytes = request_reset::enqueue_load_zero(&dev, &zero_regions)?;
        let zero_enqueue_s = t_zero.elapsed().as_secs_f64();

        let kv_only_bytes = kv_all.saturating_sub(gdn_state_bytes * n_lanes as u64);
        let hbm_bytes = [
            weights_bytes + spec_bytes[0],
            kv_only_bytes + spec_bytes[1],
            gdn_state_bytes * n_lanes as u64 + pfx_bytes,
            act_bytes + spec_bytes[2],
            ws_bytes,
        ];
        if mem_report_ok {
            // the live record groups the plan by what a part is for: `scratch` is the prefill
            // arena only, and the vision tower, the drafter's buffers, the step activations,
            // the prefill tile buffers and the workspace are named parts of `other`
            let mut mem_report = mem_report;
            mem_report.set_source(
                "target",
                format!(
                    "{model_name}, {} weights, {} K/V",
                    if spec.fp8_weights { "FP8" } else { "BF16" },
                    if crate::verified::kv_is_fp8(crate::engine::qwen3::kv8::trunk_dtype(&spec)) { "FP8" } else { "BF16" }
                ),
            );
            if let Some(sc) = cfg.spec.as_ref() {
                mem_report.set_source("drafter", format!("{} drafter", sc.method_name()));
            }
            let t = mem_report.totals();
            live.set_hbm_bytes(t[0], t[1], t[2], t[3], t[4]);
            live.set_hbm_detail(mem_report.detail_json());
            live.set_hbm_kinds(mem_report.kinds_json());
            live.set_sources(mem_report.sources_json());
        } else {
            live.set_hbm_bytes(
                hbm_bytes[0],
                hbm_bytes[1],
                hbm_bytes[2],
                hbm_bytes[3],
                hbm_bytes[4],
            );
        }
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
        for r in wave_recipes.values_mut() {
            r.bind_shared_workspace(&dev)
                .map_err(|e| format!("bind wavefront ws: {e}"))?;
        }

        // 8. Upload weights + RoPE tables from zero-copy mmap Safetensors into `weights_arena`
        let before_upload_s = t_start.elapsed().as_secs_f64();
        let t_up = Instant::now();
        let store =
            SafetensorsStore::open(&cfg.model_dir).map_err(|e| format!("open safetensors: {e}"))?;
        // what the host tier's identity says about this load's FP8 inputs: the table as parsed
        // (now), the cache files as used (after the upload)
        let fp8_table_id = crate::engine::fp8_pack::act_table_id(&fp8_act_table);
        // [moe/fp8 hook] FP8 checkpoints / FP8 device weights / MoE wide MLPs
        let ext_packer = crate::engine::fp8_pack::ExtPacker::new(
            &store,
            &spec,
            fp8_act_table,
            crate::engine::fp8_pack::cache_dir_of(cfg),
        );
        let mut packer = WeightPacker::new(&store, &spec);
        packer.ext = Some(&ext_packer);
        packer.kv8 = kv8_cal.as_ref().map(|(c, _)| c); // [kv8] the layers' scale tables

        let mut stage_upload = dev
            .alloc_pinned(STAGE_BUF_BYTES)
            .map_err(|e| format!("alloc pinned upload stage: {e}"))?;

        let t_rope = Instant::now();
        let (rope_c_bytes, rope_s_bytes) = packer.build_rope_tables(max_ctx as usize);
        let rope_build_s = t_rope.elapsed().as_secs_f64();
        if let Some(&addr) = weights_addr.get("rope_c") {
            dev.upload_h2d_chunked(&rope_c_bytes, addr, &mut stage_upload)
                .map_err(|e| format!("upload rope_c: {e}"))?;
        }
        if let Some(&addr) = weights_addr.get("rope_s") {
            dev.upload_h2d_chunked(&rope_s_bytes, addr, &mut stage_upload)
                .map_err(|e| format!("upload rope_s: {e}"))?;
        }

        let mut uploaded_bytes = (rope_c_bytes.len() + rope_s_bytes.len()) as u64;
        // Trunk weights: packed on this thread, copied into a pinned ring by a few threads and
        // handed to the DMA without waiting for it (`synapse/upload.rs`).
        let mut pack_s = 0.0f64;
        let up_threads = crate::synapse::upload::H2dStreamer::default_threads();
        let mut up = crate::synapse::upload::H2dStreamer::new(
            &dev,
            crate::synapse::upload::RING_BUFS,
            crate::synapse::upload::RING_BUF_BYTES,
            up_threads,
        )
        .map_err(|e| format!("alloc pinned upload ring: {e}"))?;
        for (name, &addr) in &weights_addr {
            if name == "rope_c" || name == "rope_s" {
                continue;
            }
            let expected = weight_specs[name];
            let t_pack = Instant::now();
            // a concatenation of checkpoint tensors goes up part by part, with no host copy
            if let Some(parts) = packer.concat_parts(name)? {
                pack_s += t_pack.elapsed().as_secs_f64();
                let total: u64 = parts.iter().map(|p| p.len() as u64).sum();
                if total != expected {
                    return Err(format!(
                        "weight '{name}' parts hold {total} bytes != expected graph tensor size {expected}"
                    ));
                }
                let mut at = addr;
                for part in parts {
                    up.put(part, at)
                        .map_err(|e| format!("upload weight '{name}': {e}"))?;
                    at += part.len() as u64;
                }
                uploaded_bytes += expected;
                continue;
            }
            let data = packer.pack_weight(name)?;
            pack_s += t_pack.elapsed().as_secs_f64();
            if data.len() as u64 != expected {
                return Err(format!(
                    "weight '{name}' packed size {} != expected graph tensor size {expected}",
                    data.len()
                ));
            }
            up.put(&data, addr)
                .map_err(|e| format!("upload weight '{name}': {e}"))?;
            uploaded_bytes += expected;
        }
        let up_stats = up
            .finish()
            .map_err(|e| format!("finish trunk weight upload: {e}"))?;
        let trunk_upload_s = t_up.elapsed().as_secs_f64();
        if cfg.verbose {
            let gib = up_stats.bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            eprintln!(
                "[gaudium] trunk weight upload: {:.2} GiB in {:.2}s ({:.2} GiB/s) = RoPE tables {:.2}s + packing {:.2}s + staging copy {:.2}s + direct file reads {:.2}s ({:.2} GiB not in the page cache) on {} threads + waiting for the link {:.2}s; {} transfers",
                gib,
                trunk_upload_s,
                gib / trunk_upload_s.max(1e-6),
                rope_build_s,
                pack_s,
                up_stats.copy_s,
                up_stats.direct_s,
                up_stats.direct_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                up_threads,
                up_stats.wait_s,
                up_stats.transfers
            );
        }
        if let Some(v) = vision.as_mut() {
            v.finish_load(&dev)?;
            uploaded_bytes += v.upload_weights(&dev, &store, &mut stage_upload)?;
        }
        // The sampler parameter tensors are bound like tables by every sampled recipe family
        // (they live in the activation arena; `samp_rsel` is the constant row selector 0).
        for (name, _) in crate::engine::sampler::sampler_param_specs() {
            weights_addr.insert(name.to_string(), step_act_addr[name]);
        }
        dev.upload_h2d_chunked(
            &[0u8; 4],
            weights_addr[crate::engine::sampler::SAMP_RSEL],
            &mut stage_upload,
        )
        .map_err(|e| format!("upload samp_rsel: {e}"))?;
        // [lanes] the batched steps bind the weights as the single-stream steps do
        let mut lane_rt = match (lane_plan, lane_alloc) {
            (Some(lp), Some((act, io))) => Some(lanes_rt::LaneRt::new(
                lp,
                other_lanes,
                act,
                &io,
                reset_mode,
            )),
            _ => None,
        };
        if let Some(rt) = lane_rt.as_mut() {
            rt.finish(&dev, &weights_addr)?;
        }
        let t_spec_fin = Instant::now();
        if let Some(sd) = &mut specdec {
            sd.finish(
                &dev,
                &spec,
                &packer,
                &mut stage_upload,
                cfg.cache_dir.as_deref().filter(|_| !cfg.no_cache),
                &weights_addr,
                &kv_addr,
                &gdn_state,
                pf_cursor,
            )?;
        }
        let spec_finish_s = t_spec_fin.elapsed().as_secs_f64();
        pf.finish_load(&dev)?; // [prefill]
                               // [moe hook] routed form: bind the routed-experts prefill recipe, zero its padded rows
        moe_rx.finish(
            &dev,
            &spec,
            prefill_tile,
            [&pf.attn_pool(prefill_tile, 1), &pf.lin_pool(prefill_tile)],
            &stage_upload,
        )?;
        drop(stage_upload); // the upload staging memory is not needed after the load
        ext_packer.log_summary(cfg.verbose); // [moe/fp8 hook]
        let fp8_id = format!(
            "table={fp8_table_id}|cache_files={}",
            ext_packer.cache_files_id()
        );

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

        // SAFETY: the compute stream belongs to `dev`.
        unsafe { dev.sync_stream(dev.compute_stream()) }
            .map_err(|e| format!("zero device state at load: {e}"))?;
        if cfg.verbose {
            eprintln!(
                "[gaudium] KV caches + state rings zeroed on the device: {:.2} GiB (enqueue {:.1} ms; complete within {:.2}s of the seal, overlapped with the upload)",
                zeroed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                zero_enqueue_s * 1000.0,
                t_zero.elapsed().as_secs_f64()
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
        // The binding pool of each parity is the same for every chunk count: built once.
        let parity_pool = |parity: usize| -> BTreeMap<String, u64> {
            let mut pool = base_step_addrs.clone();
            for (&l, st) in &gdn_state {
                pool.insert(format!("L{l}_S_in"), st.s[parity]);
                pool.insert(format!("L{l}_S_out"), st.s[1 - parity]);
                pool.insert(format!("L{l}_cs_in"), st.cs[parity]);
                pool.insert(format!("L{l}_cs_out"), st.cs[1 - parity]);
            }
            pool
        };
        let parity_pools = [parity_pool(0), parity_pool(1)];
        for r in &step_recipes {
            let build_for_parity = |parity: usize| -> Result<PreparedLaunch, String> {
                let exact = select_exact_bindings(r, &parity_pools[parity])?;
                r.prepare_launch(&exact)
                    .map_err(|e| format!("prepare decode step launch (parity={parity}): {e}"))
            };
            step_launches.push([build_for_parity(0)?, build_for_parity(1)?]);
        }

        if let Some(s) = samp.as_mut() {
            s.finish(&dev, &base_step_addrs, &gdn_state)?;
        }

        let step_ids = *step_act_addr.get("ids").ok_or("missing step 'ids'")?;
        let step_cursor = *step_act_addr.get("cursor").ok_or("missing step 'cursor'")?;
        let step_tokens = *step_act_addr.get("tokens").ok_or("missing step 'tokens'")?;
        let step_logits = *step_act_addr.get("logits").ok_or("missing step 'logits'")?;
        // [step] enqueue-ahead bindings of the same step recipes (`engine/qwen3_k0pipe.rs`)
        let step_cursor_b = step_act_addr["cursor_b"];
        let mut step_fb_launches = Vec::new();
        if cfg.step.k0_pipeline {
            for r in &step_recipes {
                let build_for_parity = |parity: usize| -> Result<PreparedLaunch, String> {
                    let mut pool = base_step_addrs.clone();
                    for (&l, st) in &gdn_state {
                        pool.insert(format!("L{l}_S_in"), st.s[parity]);
                        pool.insert(format!("L{l}_S_out"), st.s[1 - parity]);
                        pool.insert(format!("L{l}_cs_in"), st.cs[parity]);
                        pool.insert(format!("L{l}_cs_out"), st.cs[1 - parity]);
                    }
                    if parity == 1 {
                        pool.insert("ids".to_string(), step_tokens);
                        pool.insert("tokens".to_string(), step_ids);
                        pool.insert("cursor".to_string(), step_cursor_b);
                    }
                    let exact = select_exact_bindings(r, &pool)?;
                    r.prepare_launch(&exact).map_err(|e| {
                        format!("prepare enqueue-ahead decode step launch (parity={parity}): {e}")
                    })
                };
                step_fb_launches.push([build_for_parity(0)?, build_for_parity(1)?]);
            }
        }

        live.set_phase("idle", "ready");
        if cfg.verbose {
            let total = t_start.elapsed().as_secs_f64();
            let upload_all = t_up.elapsed().as_secs_f64();
            eprintln!(
                "[gaudium] start-up {:.2}s = device open {:.2} + recipes {:.2} + plan/alloc/other {:.2} + trunk weights {:.2} + drafter/vision weights, tables and launches {:.2} + launch preparation {:.2}",
                total,
                open_s,
                recipes_s,
                (before_upload_s - open_s - recipes_s).max(0.0),
                trunk_upload_s,
                spec_finish_s,
                (upload_all - trunk_upload_s - spec_finish_s).max(0.0),
            );
        }

        let mut eng = Self {
            spec,
            max_ctx,
            max_chunks,
            prefill_tile,
            driver_version,
            cache,
            elf_hashes,
            fp8_id,
            head_t1,
            step_recipes,
            embed_recipes,
            attn_layer_recipes,
            lin_layer_recipes,
            wave_recipes,
            moe_rx,
            stage_io,
            weights_addr,
            kv_addr,
            gdn_state,
            pf,
            pf_x0,
            pf_x1,
            pf_ids,
            pf_cursor,
            head_scratch,
            step_launches,
            step_fb_launches,
            step_cursor_b,
            step_ids,
            step_cursor,
            step_tokens,
            step_logits,
            hbm_bytes,
            reset: request_reset::ResetState::new(reset_mode, cfg.verbose),
            prefix: prefix_cache::PrefixCache::new(
                prefill_tile,
                pfx_granted,
                pfx_slot_bytes,
                pfx_why_not.is_none(),
                specdec.as_ref().map_or(0, |sd| sd.context_rows()),
            ),
            prefix_slots,
            prefix_plan: prefix_cache::Plan::default(),
            prefix_images: Vec::new(),
            host: None,
            xfer: None,
            verbose: cfg.verbose,
            live,
            vision,
            vision_req: vision_hooks::VisionRequestState::default(),
            specdec,
            samp,
            calib: cfg
                .fp8_calibrate
                .then(crate::engine::fp8_graph::ActCalibrator::new),
            kv_calib: cfg
                .kv_calibrate
                .then(crate::core::kv_cache::KvCalibration::default),
            kv8_id: kv8_cal
                .as_ref()
                .map_or_else(|| "none".to_string(), |(c, _)| c.id()),
            prof: prof.map(prof_rt::ProfRt::new),
            lanes: lane_rt,
            dev,
        };
        eng.warm_prefill_tiles(cfg.prefill_warm_rows, cfg.verbose)?;
        // (the host tier keeps one prefix cache, not one per lane)
        if crate::verified::ln_host_tier_ok(n_lanes as u64, false) {
            host_tier::attach(&mut eng, cfg)?;
        } else if cfg.host_cache.is_some() {
            eprintln!(
                "[gaudium] host-RAM tier of prefix reuse: off for an engine with {n_lanes} lanes (each lane keeps its prefixes on the card)"
            );
        }
        Ok(Some(eng))
    }

    /// Loads (or compiles into the recipe cache) the single-chunk prefill recipes of every
    /// prompt-tail length `1..=rows`. A prompt's last tile has `prompt_len mod prefill_tile` rows
    /// and one recipe set per length; without this the first request of a length loads them
    /// inside its time to first token (tens of milliseconds from the recipe cache, hundreds on a
    /// cache miss, measured on Gaudi 2 with the 64-layer hybrid model).
    fn warm_prefill_tiles(&mut self, rows: u64, verbose: bool) -> Result<(), String> {
        let rows = rows.min(self.prefill_tile.saturating_sub(1));
        if rows == 0 {
            return Ok(());
        }
        let t0 = Instant::now();
        let has_gdn = !self.gdn_state.is_empty();
        for t in 1..=rows {
            self.ensure_embed_recipe(t)?;
            self.ensure_attn_layer_recipe(t, 1)?;
            if has_gdn {
                self.ensure_lin_layer_recipe(t)?;
            }
            moe_rt::ensure(self, t)?; // [moe hook] routed-experts recipe of the tail's padded rows
        }
        if verbose {
            eprintln!(
                "[gaudium] prefill recipes of prompt-tail lengths 1..={rows} ready in {:.2}s",
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn max_ctx(&self) -> u64 {
        self.max_ctx
    }

    /// Reader of this engine's measured live `profile` block.
    pub fn live_profile(&self) -> crate::live_report::ProfileHandle {
        self.live.profile_handle()
    }

    /// Exact 5-category HBM ledger bytes `[weights, kv, state, scratch, workspace]`.
    pub fn hbm_bytes(&self) -> [u64; 5] {
        self.hbm_bytes
    }

    /// [fp8 hook] The activation-scale table of a calibration load (`Qwen3Config::fp8_calibrate`).
    pub fn fp8_calibration(&self) -> Option<BTreeMap<String, crate::fp8::ActScale>> {
        self.calib.as_ref().map(|c| c.table())
    }

    /// Wall time of the reset that preceded the last request, in ms (it runs before the prefill
    /// timer starts, so it is not part of `ttft_ms`).
    pub fn last_reset_ms(&self) -> f64 {
        self.reset.last_ms()
    }

    /// Prefix reuse: the kept positions and what the requests so far took over (`None`: this
    /// load keeps none).
    pub fn prefix_stats(&self) -> Option<prefix_cache::PrefixStats> {
        self.prefix.enabled().then(|| self.prefix.stats())
    }

    /// Work for the time between two requests: copies what the device holds for its prompt
    /// into the host-RAM tier. `waiting` says how many requests wait for the card; the copy
    /// stops between pieces once one does and goes on at the next call. An error is a failed
    /// device copy: the entry it wrote is gone, the engine goes on.
    pub fn idle_work(&mut self, waiting: &dyn Fn() -> u64) {
        if !self.host.as_ref().is_some_and(|t| t.has_unsaved()) {
            return;
        }
        let mut stop = || !crate::verified::hc_idle_may_run(waiting());
        // A client that sends its next request as soon as it has the answer (a tool loop) must
        // not find the card inside a copy: the save starts only once the card has been idle
        // for a moment, and a request that arrives before that takes the card at once.
        let t0 = Instant::now();
        while t0.elapsed() < host_tier::IDLE_GRACE {
            if stop() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if let Err(e) = host_tier::save(self, &mut stop) {
            eprintln!("[cache-ram] save failed: {e}");
        }
    }

    /// Before the card is released (idle release, another model, shutdown): everything the
    /// device holds for its prompt goes to the host-RAM tier, and every stream is drained.
    pub fn before_release(&mut self) {
        if let Err(e) = host_tier::save(self, &mut || false) {
            eprintln!("[cache-ram] save before the release failed: {e}");
        }
        let _ = self.dev.sync_all();
    }

    /// The host-RAM tier's measured rates, its sizes and what it did for the last request
    /// (`None`: this engine has none).
    pub fn host_tier_facts(&self) -> Option<host_tier::TierFacts> {
        self.host.as_ref().map(|t| t.facts())
    }

    /// Requests this engine runs at once (`--parallel`).
    pub fn lanes(&self) -> usize {
        self.lanes.as_ref().map_or(1, |rt| rt.lanes())
    }

    /// Serves the requests of `feed` in the engine's lanes (`lanes_rt`); an engine loaded with
    /// one lane runs them one at a time.
    pub fn serve_lanes(&mut self, feed: &mut dyn crate::engine::LaneFeed) -> Result<(), String> {
        if self.lanes.is_some() {
            lanes_rt::serve(self, feed)
        } else {
            crate::engine::serve_lanes_serially(self, feed)
        }
    }

    /// A recipe that was not ready when a request needed it: the request waited for it.
    fn note_late_recipe(&self, r: &CompiledRecipe, since: Instant) {
        if self.verbose {
            eprintln!(
                "[gaudium] recipe {} was not ready: the request waited {:.1} ms for it ({})",
                r.name,
                since.elapsed().as_secs_f64() * 1000.0,
                if r.from_cache {
                    "loaded from the recipe cache"
                } else {
                    "compiled"
                }
            );
        }
    }

    fn ensure_embed_recipe(&mut self, t: u64) -> Result<(), String> {
        if self.embed_recipes.contains_key(&t) {
            return Ok(());
        }
        let t0 = Instant::now();
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
        self.note_late_recipe(&r, t0);
        self.embed_recipes.insert(t, r);
        Ok(())
    }

    fn ensure_attn_layer_recipe(&mut self, t: u64, nch: u64) -> Result<(), String> {
        if self.attn_layer_recipes.contains_key(&(t, nch)) {
            return Ok(());
        }
        let t0 = Instant::now();
        let gb = self
            .pf
            .attn_graph(&self.spec, t, self.max_ctx, nch, &self.elf_hashes);
        self.pf.register_attn(&gb, t, nch)?; // [prefill] scratch layout in the planned region
        let mut r = gb
            .compile(
                Arc::clone(self.dev.api()),
                &self.cache,
                &self.driver_version,
            )
            .map_err(|e| format!("compile attn layer T={t} C={nch}: {e}"))?;
        r.bind_shared_workspace(&self.dev)
            .map_err(|e| format!("bind attn layer T={t} C={nch} ws: {e}"))?;
        self.note_late_recipe(&r, t0);
        self.attn_layer_recipes.insert((t, nch), r);
        Ok(())
    }

    fn ensure_lin_layer_recipe(&mut self, t: u64) -> Result<(), String> {
        if self.lin_layer_recipes.contains_key(&t) {
            return Ok(());
        }
        let t0 = Instant::now();
        let gb = self.pf.lin_graph(&self.spec, t, &self.elf_hashes);
        self.pf.register_lin(&gb, t)?; // [prefill] scratch layout in the planned region
        let mut r = gb
            .compile(
                Arc::clone(self.dev.api()),
                &self.cache,
                &self.driver_version,
            )
            .map_err(|e| format!("compile lin layer T={t}: {e}"))?;
        r.bind_shared_workspace(&self.dev)
            .map_err(|e| format!("bind lin layer T={t} ws: {e}"))?;
        self.note_late_recipe(&r, t0);
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

    #[allow(dead_code)]
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
        let dev_sp = samp_rt::dev_params(params, self.spec.vocab);
        let res = sample_row(&bf16, &dev_sp, 0, seq_pos, true);
        Ok(res.token as i32)
    }

    /// The token of a readout row: read from the card (greedy argmax, or the on-card sampler of
    /// a sampled request, which `sample_check` compares with the host twin), or sampled on the
    /// host when the engine has no on-card sampler. `n` = sequence index of the token.
    fn read_row_token(
        &mut self,
        params: &Qwen3GenParams,
        tokens_hbm: u64,
        logits_hbm: u64,
        n: u32,
    ) -> Result<i32, String> {
        if params.temperature > 0.0 && self.samp.is_none() {
            return self.sample_from_logits(logits_hbm, params, n);
        }
        let tok = self.read_i32_scalar(tokens_hbm)?;
        if params.temperature > 0.0 && params.sample_check {
            samp_rt::check_row(self, logits_hbm, n, tok)?;
        }
        Ok(tok)
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

        let wprec = if self.spec.fp8_weights { "FP8" } else { "BF16" }; // [fp8 hook]
        let method_label = if let Some(sd) = &self.specdec {
            sd.method_label(params.temperature).replace(
                " BF16 ",
                if self.spec.fp8_weights {
                    " FP8 "
                } else {
                    " BF16 "
                },
            )
        } else if params.temperature > 0.0 {
            format!("K=0 {wprec} T={:.2}", params.temperature)
        } else {
            format!("K=0 {wprec} greedy")
        };
        self.live
            .begin_request(prompt_ids.len() as u64, &method_label, "");
        spec::readout::arm(self, params.temperature > 0.0); // [readout hook] step byte model
        self.live.prefill_progress(0);

        let res = self.run_generate_inner(
            prompt_ids,
            params,
            &mut budget,
            &is_eos,
            &is_cancelled,
            &mut on_token,
        );
        self.live.end_request();
        request_reset::end_request(self, prompt_ids.len() as u64, &res);
        // prefix reuse: what the request leaves for the next one (nothing after an error)
        let written_end = res.as_ref().map_or(0, |r| {
            request_reset::request_high_water(
                prompt_ids.len() as u64,
                r.generated_tokens.len() as u64,
                self.max_ctx,
            )
        });
        self.prefix.request_end(res.is_ok(), written_end);
        let (reused, common) = (self.prefix_plan.resume, self.prefix_plan.lcp);
        // host tier: what the request wrote no longer equals anything saved; what it left is
        // to be saved (after the response: `idle_work`)
        if let Some(tier) = self.host.as_mut() {
            let (prefilled, secs) = res.as_ref().map_or((0, 0.0), |r| {
                (
                    (r.prompt_tokens as u64).saturating_sub(reused),
                    // the restore is inside the prefill clock, not part of the prefill itself
                    (r.ttft_ms - tier.last().restore_ms - tier.last().save_ms).max(0.0) / 1e3,
                )
            });
            let cancelled = res.as_ref().is_ok_and(|r| r.stopped_by_cancel);
            tier.end_request(
                res.is_ok(),
                reused,
                written_end,
                if cancelled { 0 } else { prefilled },
                secs,
            );
        }
        let tier_report = self.host.as_ref().map(|t| t.last());
        let res = res.map(|mut r| {
            r.reused_tokens = reused;
            r.prefix_common = common;
            r.host_tier = tier_report;
            r
        });
        // [prefill] measurement of the half-layer recipe forms, once, after the first request
        if res.is_ok() && self.pf.opts.half_probe > 0 {
            let n = std::mem::take(&mut self.pf.opts.half_probe);
            halfprobe::run(self, n)?;
        }
        res
    }

    /// The prefill of a request: the per-request reset, the prefix plan, the image tower, every
    /// prefill tile and the readout of the first token. `defer_one_row`: a one-row prompt is not
    /// run here (a lane runs it as its first decode step, `Prefilled::OneRow`).
    pub(super) fn run_prefill(
        &mut self,
        prompt_ids: &[i32],
        params: &Qwen3GenParams,
        budget: &mut SequenceBudget,
        is_cancelled: &dyn Fn() -> bool,
        defer_one_row: bool,
    ) -> Result<Prefilled, String> {
        // (before the reset: whether it zeroes K/V rows decides what the request may take over)
        let rows_zeroed = self.reset.zeroes_rows();
        self.prefix_plan = prefix_cache::Plan::default();
        request_reset::begin_request(self, prompt_ids.len() as u64, budget.max_new())?;
        spec::begin_request(self, prompt_ids);
        prof_rt::begin_request(self);
        let mut gdn_parity: usize = 0;
        // A sampled request runs the sampled recipe families; its parameters go to the card now.
        let samp_on = params.temperature > 0.0 && self.samp.is_some();
        if let Some(s) = self.samp.as_mut() {
            s.reset_checks();
        }
        if samp_on {
            samp_rt::begin_request(self, params, prompt_ids.len())?;
        }
        // first-use recipe loads / compiles inside the time to first token
        let mut ttft_recipe_ms = 0.0f64;

        let t_prefill = Instant::now();
        // Image input: the prompt's layout first (the prefix plan compares its image runs).
        let vis_prepared = self.vision_prepare(prompt_ids)?;
        // Prefix reuse: where this prefill starts and which kept positions it reads and writes
        // (`engine/prefix_cache.rs`); a cold plan launches exactly what the engine always did.
        self.prefix_images = vis_prepared
            .as_ref()
            .map(|(images, layout)| prefix_cache::image_ids(images, layout))
            .unwrap_or_default();
        let reuse_wanted = params.cache_prompt
            && self.calib.is_none()
            && !self.pf.opts.profile
            && self.pf.opts.half_probe == 0;
        // Host tier: rows and state the card has lost come back from host memory first, when
        // that is faster than prefilling them (`engine/host_tier.rs`); the plan then finds them
        // as if they had never left.
        let restored = host_tier::begin_request(self, prompt_ids, reuse_wanted, rows_zeroed)?;
        self.prefix_plan = self.prefix.plan_from(
            prompt_ids,
            &self.prefix_images,
            reuse_wanted,
            rows_zeroed,
            restored,
        );
        let resume = self.prefix_plan.resume as usize;
        spec::set_context_floor(self, self.prefix_plan.ring_flo)?;
        if resume > 0 {
            self.live.set_cached(resume as u64);
        }
        if self.verbose && self.prefix.enabled() {
            eprintln!(
                "[gaudium] prefix reuse: {} of {} prompt rows taken over ({} rows equal the held prompt{}); this request keeps {} position(s); held now {:?}",
                resume,
                prompt_ids.len(),
                self.prefix_plan.lcp,
                if self.prefix_plan.cap < self.prefix_plan.lcp {
                    format!(", {} before the first image that differs", self.prefix_plan.cap)
                } else {
                    String::new()
                },
                self.prefix_plan.captures.len(),
                self.prefix.held_boundaries(),
            );
        }
        // Tower launches + this request's slot-indexed RoPE rows (inside TTFT). `None` for a
        // text-only request, which launches exactly what it did before.
        let vis_layout =
            self.vision_begin_request(vis_prepared, prompt_ids.len(), budget.max_new(), resume)?;
        let generated: Vec<i32> = Vec::new();
        let mut stopped_by_cancel = false;
        let mut first_logits_hbm = self.step_logits;

        // ---- PREFILL PHASE ----
        if prompt_ids.len() == 1 && defer_one_row {
            // a one-row prompt runs as a decode step: nothing of it is a prefill tile's row
            self.prefix.prefill_end(prompt_ids, &self.prefix_images, 0);
            return Ok(Prefilled::OneRow {
                gdn_parity,
                ttft_recipe_ms,
            });
        }
        let first_tok: i32 = if prompt_ids.len() == 1 {
            self.write_step_inputs(prompt_ids[0], 0)?;
            if samp_on {
                ttft_recipe_ms += samp_rt::ensure_step(self, 1)?;
                samp_rt::launch_step(self, 1, gdn_parity)?;
            } else {
                self.step_recipes[0]
                    .launch_prepared(&self.dev, &self.step_launches[0][gdn_parity])
                    .map_err(|e| format!("launch step (prompt len 1): {e}"))?;
            }
            unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                .map_err(|e| format!("sync compute (prompt len 1): {e}"))?;
            gdn_parity ^= 1;
            // a one-row prompt runs as a decode step: nothing of it is a prefill tile's row
            self.prefix.prefill_end(prompt_ids, &self.prefix_images, 0);
            self.read_row_token(params, self.step_tokens, self.step_logits, 1)?
        } else {
            let mut pos = resume;
            let n_layers = self.spec.n_layers as usize;
            let has_gdn = !self.gdn_state.is_empty();
            let mut last_tile_len = 0u64;
            let pf_profile = self.pf.opts.profile;
            self.pf.begin_request();
            // [profile] index of the next prefill launch (`--profile-steps pA-B`)
            let mut pf_launch: u64 = 0;

            // [prefill] the full tiles the verified schedule runs two in flight; the single-tile
            // loop below takes the rest (an odd full tile, the partial last tile)
            let mut wave_final_x: Option<u64> = None;
            if let Some(w) = wave_rt::run(
                self,
                prompt_ids,
                resume,
                gdn_parity,
                vis_layout.as_ref(),
                is_cancelled,
                &mut pf_launch,
            )? {
                pos = resume + w.tiles * self.prefill_tile as usize;
                // the wavefront synchronised: the positions its finished tiles wrote are held
                self.prefix.tiles_done(
                    prompt_ids,
                    &self.prefix_images,
                    pos as u64 / self.prefill_tile.max(1),
                );
                if w.cancelled {
                    stopped_by_cancel = true;
                } else {
                    last_tile_len = self.prefill_tile;
                    wave_final_x = Some(w.final_x);
                    // an even number of tiles leaves every recurrent state in the buffer it
                    // started from; an odd count would flip it like the single-tile loop does
                    if has_gdn && w.tiles % 2 == 1 {
                        gdn_parity ^= 1;
                    }
                }
            }

            while pos < prompt_ids.len() && !stopped_by_cancel {
                if is_cancelled() {
                    stopped_by_cancel = true;
                    break;
                }
                wave_final_x = None;
                let tile_len = (prompt_ids.len() - pos).min(self.prefill_tile as usize) as u64;
                let end_pos = (pos as u64) + tile_len;
                let nch = end_pos.div_ceil(KV_CHUNK).clamp(1, self.max_chunks);
                last_tile_len = tile_len;

                let t_recipes = Instant::now();
                self.ensure_embed_recipe(tile_len)?;
                self.ensure_attn_layer_recipe(tile_len, nch)?;
                if has_gdn {
                    self.ensure_lin_layer_recipe(tile_len)?;
                }
                moe_rt::ensure(self, tile_len)?; // [moe hook] routed-experts recipe of the tile
                ttft_recipe_ms += t_recipes.elapsed().as_secs_f64() * 1000.0;

                // the state buffers of this tile's recurrent layers (a tile at a kept position
                // reads or writes that position's slot), resolved before its first launch
                let tile_idx = pos as u64 / self.prefill_tile.max(1);
                let mut tile_state: BTreeMap<usize, prefix_cache::StateIo> = BTreeMap::new();
                for (&l, st) in &self.gdn_state {
                    tile_state.insert(
                        l,
                        prefix_cache::state_io(
                            st,
                            &self.prefix_slots,
                            &self.prefix_plan,
                            self.prefill_tile,
                            l,
                            tile_idx,
                            gdn_parity,
                        )?,
                    );
                }

                let t_tile = prefill::now();
                let tile_bytes = (tile_len as usize) * 4;
                for (k, &id) in prompt_ids[pos..pos + tile_len as usize].iter().enumerate() {
                    self.stage_io.as_mut_slice()[k * 4..k * 4 + 4]
                        .copy_from_slice(&id.to_le_bytes());
                }
                // the tile's ids and its cursor in one synchronised upload
                let cur_off = tile_bytes.next_multiple_of(64);
                self.stage_io.as_mut_slice()[cur_off..cur_off + 4]
                    .copy_from_slice(&(pos as i32).to_le_bytes());
                self.dev
                    .copy_h2d_pinned_batch_sync(
                        &self.stage_io,
                        &[(0, tile_bytes, self.pf_ids), (cur_off, 4, self.pf_cursor)],
                    )
                    .map_err(|e| format!("upload prefill ids / cursor: {e}"))?;
                self.pf.timers.upload_ms += prefill::ms_since(t_tile);
                let t_submit = prefill::now();
                // profile runs: device time of one launch, by synchronising right after it
                let prof = |dev: &HpuDevice, t0: std::time::Instant| -> Result<f64, String> {
                    unsafe { dev.sync_stream(dev.compute_stream()) }
                        .map_err(|e| format!("sync prefill profile: {e}"))?;
                    Ok(prefill::ms_since(t0))
                };

                // Launch embed(tile_len) -> pf_x0
                let mut emb_pool = BTreeMap::new();
                emb_pool.insert("embed_table".to_string(), self.weights_addr["embed_table"]);
                emb_pool.insert("ids".to_string(), self.pf_ids);
                emb_pool.insert("x0".to_string(), self.pf_x0);
                let emb_recipe = &self.embed_recipes[&tile_len];
                let emb_bind = select_exact_bindings(emb_recipe, &emb_pool)?;
                let t_l = prefill::now();
                prof_rt::before_prefill_launch(&mut self.prof, &self.dev, pf_launch)?;
                pf_launch += 1;
                emb_recipe
                    .launch_checked(&self.dev, &emb_bind)
                    .map_err(|e| format!("launch embed T={tile_len}: {e}"))?;
                if pf_profile {
                    let ms = prof(&self.dev, t_l)?;
                    self.pf.timers.class(&format!("embed T{tile_len}"), ms);
                }
                if let Some(layout) = vis_layout.as_ref() {
                    // image-pad rows of this tile take the tower's embeddings (device copy)
                    self.vision_splice_tile(layout, pos, tile_len as usize, self.pf_x0)?;
                }

                // Prepare reusable pools for attention & linear layers
                // [moe hook] routed form: where this tile's layer recipes leave their scratch
                let (rx_attn_scratch, rx_lin_scratch) =
                    (self.pf.attn_pool(tile_len, nch), self.pf.lin_pool(tile_len));
                let attn_recipe = &self.attn_layer_recipes[&(tile_len, nch)];
                let mut apool = self.pf.attn_pool(tile_len, nch);
                apool.insert("cursor".to_string(), self.pf_cursor);
                apool.insert("rope_c".to_string(), self.weights_addr["rope_c"]);
                apool.insert("rope_s".to_string(), self.weights_addr["rope_s"]);
                if let Some(&ones) = self.weights_addr.get("norm_ones") {
                    apool.insert("norm_ones".to_string(), ones);
                }

                let mut lpool = BTreeMap::new();
                if has_gdn {
                    lpool = self.pf.lin_pool(tile_len);
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
                    let (xin, xout) = match &self.specdec {
                        Some(sd) => sd.prefill_io(l, xin, xout),
                        None => (xin, xout),
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
                        // [moe/fp8 hook] router, FP8 weight scales, site cast scales of layer l
                        crate::engine::moe::bind_ext_layer_weights(
                            &self.weights_addr,
                            l,
                            &mut apool,
                        );
                        let abind = select_exact_bindings(attn_recipe, &apool)?;
                        let t_l = prefill::now();
                        prof_rt::before_prefill_launch(&mut self.prof, &self.dev, pf_launch)?;
                        pf_launch += 1;
                        attn_recipe.launch_checked(&self.dev, &abind).map_err(|e| {
                            format!("launch prefill attn layer {l} T={tile_len}: {e}")
                        })?;
                        if pf_profile {
                            let ms = prof(&self.dev, t_l)?;
                            self.pf
                                .timers
                                .class(&format!("attn T{tile_len} C{nch}"), ms);
                        }
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
                        let io = tile_state[&l];
                        lpool.insert("L_S_in".to_string(), io.s_in);
                        lpool.insert("L_S_out".to_string(), io.s_out);
                        lpool.insert("L_cs_in".to_string(), io.cs_in);
                        lpool.insert("L_cs_out".to_string(), io.cs_out);
                        // [moe/fp8 hook] router, FP8 weight scales, site cast scales of layer l
                        crate::engine::moe::bind_ext_layer_weights(
                            &self.weights_addr,
                            l,
                            &mut lpool,
                        );
                        let lbind = select_exact_bindings(lin_recipe, &lpool)?;
                        let t_l = prefill::now();
                        prof_rt::before_prefill_launch(&mut self.prof, &self.dev, pf_launch)?;
                        pf_launch += 1;
                        lin_recipe.launch_checked(&self.dev, &lbind).map_err(|e| {
                            format!("launch prefill lin layer {l} T={tile_len}: {e}")
                        })?;
                        if pf_profile {
                            let ms = prof(&self.dev, t_l)?;
                            self.pf.timers.class(&format!("lin T{tile_len}"), ms);
                        }
                    }
                    // [moe hook] routed form: this layer's routed experts and tail -> xout
                    self.moe_rx.launch_layer(
                        &self.dev,
                        if self.spec.is_full_attention_layer(l) {
                            &rx_attn_scratch
                        } else {
                            &rx_lin_scratch
                        },
                        &self.weights_addr,
                        self.prefill_tile,
                        l,
                        tile_len,
                        xout,
                    )?;
                    // [fp8 hook] calibration load: read back this layer's GEMM input sites
                    if let Some(cal) = self.calib.as_mut() {
                        unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                            .map_err(|e| format!("sync calibration layer {l}: {e}"))?;
                        let scratch = if self.spec.is_full_attention_layer(l) {
                            &apool
                        } else {
                            &lpool
                        };
                        crate::engine::fp8_graph::calibrate_layer(
                            cal, &self.dev, &self.spec, scratch, l, tile_len,
                        )?;
                    }
                }

                if self.specdec.is_some() {
                    let t_l = prefill::now();
                    spec::prefill_tile_hook(self, pos as u64, tile_len)?;
                    if pf_profile {
                        let ms = prof(&self.dev, t_l)?;
                        self.pf.timers.class("drafter context", ms);
                    }
                }
                self.pf.timers.submit_ms += prefill::ms_since(t_submit);
                let t_wait = prefill::now();
                unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                    .map_err(|e| format!("sync prefill tile at pos={pos}: {e}"))?;
                self.pf.timers.wait_ms += prefill::ms_since(t_wait);
                self.pf.timers.tiles += 1;
                self.pf.timers.launches += n_layers as u64 + 1;
                if has_gdn {
                    gdn_parity ^= 1;
                }
                pos += tile_len as usize;
                // synchronised: a whole tile's end is a position the next request can resume at
                if tile_len == self.prefill_tile {
                    self.prefix.tiles_done(
                        prompt_ids,
                        &self.prefix_images,
                        pos as u64 / self.prefill_tile.max(1),
                    );
                }
                self.live.prefill_tile(
                    pos as u64,
                    n_layers as u64 + 1,
                    (tile_len << 16) | nch,
                    || {
                        // [prefill] the byte model of the tile forms actually launched
                        let attn = (self.spec.n_full_attention_layers() > 0).then(|| {
                            self.pf
                                .attn_graph(&self.spec, tile_len, self.max_ctx, nch, &[])
                        });
                        let lin = has_gdn.then(|| self.pf.lin_graph(&self.spec, tile_len, &[]));
                        qwen3_traffic::prefill_tile_of(
                            &self.spec,
                            tile_len,
                            nch,
                            attn.as_ref(),
                            lin.as_ref(),
                        )
                    },
                );
            }

            prof_rt::end_prefill(&mut self.prof, &self.dev, pf_launch)?;
            // rows [0, pos) are this prompt's now (all of it, or the tiles a cancelled prefill
            // finished); positions whose tile did not finish are dropped
            self.prefix
                .prefill_end(prompt_ids, &self.prefix_images, pos as u64);
            if stopped_by_cancel {
                let ttft_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
                return Ok(Prefilled::Cancelled(Qwen3GenReport {
                    prompt_tokens: prompt_ids.len(),
                    generated_tokens: generated,
                    ttft_ms,
                    prefill_tok_s: 0.0,
                    decode_ms: 0.0,
                    decode_tok_s: 0.0,
                    ms_per_step: 0.0,
                    stopped_by_eos: false,
                    stopped_by_cancel: true,
                    forks: Vec::new(),
                    launches: 0,
                    tokens_per_launch: 0.0,
                    method: String::new(),
                    accept_per_pos: Vec::new(),
                    ..Qwen3GenReport::default()
                }));
            }

            // Launch shared head_T1 directly on the last prompt token's hidden row!
            let final_x_base = match wave_final_x {
                // the prompt ended on a tile of the wavefront: its rows, on its buffer set
                Some(x) => x,
                None if n_layers.is_multiple_of(2) => self.pf_x0,
                None => self.pf_x1,
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
            if samp_on {
                for (name, _) in crate::engine::sampler::sampler_param_specs() {
                    hpool.insert(name.to_string(), self.weights_addr[name]);
                }
            }

            // a sampled request reads out through the sampled head (same buffers, the on-card
            // sampler at sequence index `prompt_len` in place of the argmax)
            let head = match self.samp.as_ref().filter(|_| samp_on) {
                Some(s) => s.head(),
                None => &self.head_t1,
            };
            let hbind = select_exact_bindings(head, &hpool)?;
            head.launch_checked(&self.dev, &hbind)
                .map_err(|e| format!("launch head_T1: {e}"))?;
            unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                .map_err(|e| format!("sync head_T1: {e}"))?;
            first_logits_hbm = self.head_scratch["logits"];

            self.read_row_token(
                params,
                self.head_scratch["tokens"],
                self.head_scratch["logits"],
                prompt_ids.len() as u32,
            )?
        };

        let ttft_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        // rows this request prefilled (a resumed prefill takes the rest over)
        let prefilled = prompt_ids.len() - resume.min(prompt_ids.len());
        let prefill_tok_s = (prefilled as f64) / (ttft_ms / 1000.0).max(1e-9);
        if prompt_ids.len() > 1 {
            self.pf.report(prefilled, ttft_ms); // [prefill] profile / stats runs only
        }
        Ok(Prefilled::First(PrefillFirst {
            first_tok,
            first_logits_hbm,
            ttft_ms,
            prefill_tok_s,
            ttft_recipe_ms,
            gdn_parity,
            samp_on,
        }))
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
        let pf = match self.run_prefill(prompt_ids, params, budget, is_cancelled, false)? {
            Prefilled::First(f) => f,
            Prefilled::Cancelled(rep) => return Ok(rep),
            Prefilled::OneRow { .. } => {
                return Err("internal: a deferred one-row prompt outside a lane".into())
            }
        };
        let PrefillFirst {
            first_tok,
            first_logits_hbm,
            ttft_ms,
            prefill_tok_s,
            ttft_recipe_ms,
            mut gdn_parity,
            samp_on,
        } = pf;
        let mut generated = Vec::with_capacity(budget.max_new() as usize);
        let mut stopped_by_eos = false;
        let mut stopped_by_cancel = false;
        let mut forks: Vec<ForkRecord> = Vec::new();
        let first_tok = self.apply_force(params, 0, first_tok, first_logits_hbm, &mut forks)?;

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

        // ---- SPECULATIVE DECODE PHASE (requests of an engine loaded with `--spec`) ----
        if self
            .specdec
            .as_ref()
            .is_some_and(|sd| params.temperature <= 0.0 || sd.handles_sampled())
            && !params.non_speculative
            && !stopped_by_eos
            && !stopped_by_cancel
            && !generated.is_empty()
        {
            let method = self
                .specdec
                .as_ref()
                .map(|sd| {
                    sd.method_label(params.temperature).replace(
                        " BF16 ",
                        if self.spec.fp8_weights {
                            " FP8 "
                        } else {
                            " BF16 "
                        },
                    )
                })
                .unwrap_or_default();
            let out = spec::decode(
                self,
                prompt_ids.len(),
                params,
                budget,
                is_eos,
                is_cancelled,
                on_token,
                &mut generated,
                gdn_parity,
            )?;
            let launches = out.stats.launches;
            let (sample_check_rows, sample_check_mismatches) =
                self.samp.as_ref().map_or((0, 0), |s| {
                    (s.check_rows + s.walk_rows, s.check_bad + s.walk_bad)
                });
            return Ok(Qwen3GenReport {
                sample_check_rows,
                sample_check_mismatches,
                ttft_recipe_ms,
                prompt_tokens: prompt_ids.len(),
                generated_tokens: generated,
                ttft_ms,
                prefill_tok_s,
                decode_ms: out.decode_ms,
                decode_tok_s: if out.decode_tokens > 0 {
                    out.decode_tokens as f64 / (out.decode_ms / 1000.0).max(1e-9)
                } else {
                    0.0
                },
                ms_per_step: if launches > 0 {
                    out.decode_ms / launches as f64
                } else {
                    0.0
                },
                stopped_by_eos: out.stopped_by_eos,
                stopped_by_cancel: out.stopped_by_cancel,
                forks,
                launches,
                tokens_per_launch: if launches > 0 {
                    out.decode_tokens as f64 / launches as f64
                } else {
                    0.0
                },
                method,
                accept_per_pos: out.stats.accept_per_pos(),
                drafted: out.stats.widths.iter().map(|(k, n)| k * n).sum(),
                draft_accepted: out
                    .stats
                    .accept_hist
                    .iter()
                    .enumerate()
                    .map(|(a, n)| a as u64 * n)
                    .sum(),
                // set by the caller from the request's prefix plan
                reused_tokens: 0,
                prefix_common: 0,
                host_tier: None,
            });
        }

        // ---- DECODE PHASE (1 monolithic synLaunch per token, ping-pong GDN parity!) ----
        let t_decode = Instant::now();
        let mut cur_tok = first_tok;
        let mut decode_steps = 0usize;
        // [step] greedy requests decode enqueue-ahead (same recipes, token fed back on the card)
        let mut piped_ms: Option<f64> = None;
        if !samp_on
            && k0pipe::usable(self, params)
            && !stopped_by_eos
            && !stopped_by_cancel
            && budget.remaining() > 0
        {
            let out = k0pipe::decode(
                self,
                prompt_ids.len(),
                params,
                budget,
                is_eos,
                is_cancelled,
                on_token,
                &mut generated,
                gdn_parity,
                first_tok,
            )?;
            decode_steps = out.steps;
            stopped_by_eos = out.eos;
            stopped_by_cancel = out.cancel;
            piped_ms = Some(out.decode_ms);
        }

        while piped_ms.is_none() && !stopped_by_eos && !stopped_by_cancel && budget.remaining() > 0
        {
            if is_cancelled() {
                stopped_by_cancel = true;
                break;
            }
            let cursor_val = (prompt_ids.len() + decode_steps) as u64;
            let nch = (cursor_val + 1)
                .div_ceil(KV_CHUNK)
                .clamp(1, self.max_chunks) as usize;
            prof_rt::before_launch(self, decode_steps as u64, 0)?;

            self.write_step_inputs(cur_tok, cursor_val as i32)?;

            if samp_on {
                samp_rt::ensure_step(self, nch as u64)?;
            }
            self.live.launch_begin_with(nch as u64, || {
                qwen3_traffic::decode_step(&self.spec, self.max_ctx, nch as u64)
            });
            if samp_on {
                samp_rt::launch_step(self, nch as u64, gdn_parity).map_err(|e| {
                    format!("decode step {decode_steps} (cursor={cursor_val}): {e}")
                })?;
            } else {
                self.step_recipes[nch - 1]
                    .launch_prepared(&self.dev, &self.step_launches[nch - 1][gdn_parity])
                    .map_err(|e| {
                        format!("launch decode step {decode_steps} (cursor={cursor_val}): {e}")
                    })?;
            }
            self.live.launch_submitted();
            unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
                .map_err(|e| format!("sync decode step {decode_steps}: {e}"))?;
            gdn_parity ^= 1;

            // the row at `cursor_val` yields the token at sequence index `cursor_val + 1`
            let next_tok = self.read_row_token(
                params,
                self.step_tokens,
                self.step_logits,
                cursor_val as u32 + 1,
            )?;

            let next_tok = self.apply_force(
                params,
                generated.len(),
                next_tok,
                self.step_logits,
                &mut forks,
            )?;

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

        let decode_ms = piped_ms.unwrap_or_else(|| t_decode.elapsed().as_secs_f64() * 1000.0);
        if piped_ms.is_none() {
            prof_rt::end_request(self, decode_steps as u64)?;
        }
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
            forks,
            launches: decode_steps as u64,
            tokens_per_launch: if decode_steps > 0 { 1.0 } else { 0.0 },
            method: {
                let wprec = if self.spec.fp8_weights { "FP8" } else { "BF16" };
                if params.temperature > 0.0 {
                    format!("K=0 {wprec} T={:.2}", params.temperature)
                } else {
                    format!("K=0 {wprec} greedy")
                }
            },
            accept_per_pos: Vec::new(),
            sample_check_rows: self.samp.as_ref().map_or(0, |s| s.check_rows),
            sample_check_mismatches: self.samp.as_ref().map_or(0, |s| s.check_bad),
            ttft_recipe_ms,
            ..Qwen3GenReport::default()
        })
    }

    /// Under teacher forcing, compares the engine's greedy token at generated position `pos`
    /// with the forced one, records a fork (with both logits) when they differ, and returns the
    /// token generation continues with.
    fn apply_force(
        &mut self,
        params: &Qwen3GenParams,
        pos: usize,
        engine_tok: i32,
        logits_hbm: u64,
        forks: &mut Vec<ForkRecord>,
    ) -> Result<i32, String> {
        let Some(forced) = params
            .force_tokens
            .as_ref()
            .and_then(|f| f.get(pos).copied())
        else {
            return Ok(engine_tok);
        };
        if forced == engine_tok {
            return Ok(engine_tok);
        }
        let vocab = self.spec.vocab as usize;
        self.dev
            .copy_d2h_pinned_sync(logits_hbm, &mut self.stage_io, vocab * 2)
            .map_err(|e| format!("read logits D2H (fork trace): {e}"))?;
        let raw = self.stage_io.as_slice();
        let logit_of = |tok: i32| -> f32 {
            let i = tok as usize;
            if tok < 0 || i >= vocab {
                return f32::NAN;
            }
            crate::fp8::bf16_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]))
        };
        forks.push(ForkRecord {
            pos,
            engine_token: engine_tok,
            forced_token: forced,
            engine_logit: logit_of(engine_tok),
            forced_logit: logit_of(forced),
        });
        Ok(forced)
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

pub(crate) fn is_weight_or_table(name: &str) -> bool {
    if matches!(
        name,
        "embed_table" | "W_lm" | "w_final" | "norm_ones" | "rope_c" | "rope_s" | "zero_slot"
    ) {
        return true;
    }
    if let Some(rest) = name.strip_prefix('L') {
        if let Some(sep) = rest.find('_') {
            if rest[..sep].chars().all(|c| c.is_ascii_digit()) && !rest[..sep].is_empty() {
                // [moe/fp8 hook] router, shared gate, FP8 weight scales and site cast scales
                if crate::engine::moe::is_ext_layer_weight(&rest[sep + 1..]) {
                    return true;
                }
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

pub(crate) fn is_generic_layer_weight(name: &str) -> bool {
    // [moe/fp8 hook] the generic per-layer names of the extra layer weights
    if name
        .strip_prefix("L_")
        .is_some_and(crate::engine::moe::is_ext_layer_weight)
    {
        return true;
    }
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
