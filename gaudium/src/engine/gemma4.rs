//! Intel Gaudi 2 execution engine for the Gemma 4 text decoders (E2B, E4B, 12B, 31B): BF16 or
//! FP8 projection / MLP weights (`gemma4_fp8`), non-speculative decode, and speculative decode
//! with the size's own drafter checkpoint or a published block drafter (child module
//! `gemma4_spec`).
//!
//! What the family needs from the chip, and how this engine lays it out:
//!
//! - **Two layer kinds, two kinds of state.** A *global* (`full_attention`) layer is causal over
//!   the whole context: its K / V live in chunk-major caches of `Lmax` rows that every decode step
//!   reads through the MME (`batch_gemm`), one bucket of 4096-row chunks at a time. A *sliding*
//!   layer sees the last `W` positions only, so its K / V live in a **ring** of `RING` slots
//!   (`slot = position mod RING`, `RING` the power of two at or above `W + rows_of_a_pass - 1`):
//!   bounded HBM and a bounded number of key rows read per step, whatever the context length.
//!   The ring softmax masks by ring distance, so rows of a pass that lie ahead of a row and stale
//!   rows of an earlier request are never attended.
//! - **Layers without keys of their own** (the small sizes' `num_kv_shared_layers`): such a layer
//!   projects queries only and attends the ring / cache of the last layer of its kind that has
//!   keys, which already ran the same rows earlier in the pass. No K / V state at all for them.
//! - **Per-layer input embeddings** (the small sizes): one gather of a `{layers x 256}`-wide table
//!   row per token plus one GEMM of the input embedding gives every layer's 256-value input.
//!   A prefill tile computes them once into a layer-major buffer that each layer's recipe reads
//!   its block of; a decode step splits them inside its one recipe.
//! - **Readout**: tied embedding table as the lm_head (one copy in HBM), the logit soft cap on the
//!   card, and the family's own arg-max (lowest id among equal maxima, any sign) because the cap
//!   merges neighbouring BF16 logits into exact ties.
//!
//! Execution follows the engine pattern of this crate: every graph goes through `GraphBuilder`
//! (full fingerprint coverage, recipe cache), device memory is summed up front, allocated exactly
//! and sealed, prefill runs generic single-layer recipes bound per layer at launch, and decode is
//! one monolithic `synLaunch` per token with a pre-validated binding list.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::core::budget::SequenceBudget;
use crate::core::gemma4::Gemma4Rope;
use crate::engine::fp8_graph::{add_linear_fp8, is_fp8_aux_suffix, is_fp8_matrix, ActCalibrator};
use crate::engine::gemma4_fp8::G4Fp8Packer;
use crate::engine::gemma4_packer::Gemma4Packer;
use crate::engine::{EngineConfig, GenParams, GenReport};
use crate::json::{parse_json, JsonValue};
use crate::kernels::KernelEnv;
use crate::live_report::LiveReporter;
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::safetensors::SafetensorsStore;
use crate::sampler::{sample_row, SampleParams};
use crate::synapse::device::{HbmArena, HpuDevice, PinnedBuf};
use crate::synapse::ffi::{SynDataType, SynapseApi};
use crate::synapse::graph::{CompiledRecipe, GraphBuilder, PreparedLaunch, TensorId};
use crate::synapse::ops::{as_param_bytes, u32_params, TransposeParams};
use crate::synapse::recipe_cache::RecipeCache;
use crate::verified::{align_up, g4v_arena_step, g4v_part_bytes, g4v_take_bytes, HbmCat};

#[path = "gemma4_spec.rs"]
pub mod spec;
// Image input: the image path (a tower or the unified embedder), its preprocessing, and the
// hand-over of the soft tokens to the prefill.
#[path = "gemma4_vision.rs"]
pub mod vision;
// The prefill of a prompt with images: the embedding stage of image rows and the two-way image
// blocks of the sliding layers.
#[path = "gemma4_units.rs"]
mod units;
// The prefill recipes a request launches (picked here for the prefill and for `--build-cache`),
// and `--build-cache`.
#[path = "gemma4_recipes.rs"]
mod recipes;

/// Rows of one chunk of a global layer's K / V cache (the chunked softmax kernel's `C`).
pub const G4_CHUNK: u64 = 4096;
/// Largest prefill tile planned into the activation scratch.
pub const G4_MAX_PREFILL_TILE: u64 = 256;
/// Smallest prefill tile the memory planner falls back to before refusing a configuration.
const G4_MIN_PREFILL_TILE: u64 = 16;
// --- readout: the decode step's byte model (host-only) ---
#[path = "gemma4_readout.rs"]
mod readout;

/// Largest ring the ring softmax kernel stages.
const G4_MAX_RING: u64 = 4096;
/// Shared recipe workspace floor, so lazily compiled recipes never need a device allocation.
const MIN_SHARED_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
/// Reusable pinned host staging buffer for chunked weight uploads and state zeroing.
const STAGE_BUF_BYTES: usize = 64 * 1024 * 1024;
/// HBM kept free beside the plan.
const HBM_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;
const RMS_EPS: f32 = 1e-6;

/// Kernel ELFs every Gemma 4 graph may be handed by the graph-compiler glue.
const GEMMA4_ELFS: &[&str] = &[
    "GAUDIUM_GAPREP_ELF",
    "GAUDIUM_GAPREPK_ELF",
    "GAUDIUM_GAQPREP_ELF",
    "GAUDIUM_GASOFTR_ELF",
    "GAUDIUM_ASOFTC_ELF",
    "GAUDIUM_GACOMB4_ELF",
    "GAUDIUM_GACOMB8_ELF",
    "GAUDIUM_GAPOST_ELF",
    "GAUDIUM_GAGELU_ELF",
    "GAUDIUM_GASOFTCAP_ELF",
    "GAUDIUM_GAARGMAX_ELF",
    "GAUDIUM_GARMSNORM_ELF",
];

/// Which RMS-norm node the layer graphs use.
///
/// The vendor norm's `1 / rms` of a row can move by an f32 step with the number of rows in the
/// pass; the family's own row norm (`gaudium_ga_rmsnorm`) gives a row the same bits in any pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G4NormMode {
    /// Vendor norm node everywhere.
    Vendor,
    /// Row norm in prefill tile recipes, vendor node in the decode step.
    Tile,
    /// Row norm everywhere.
    Row,
}

impl G4NormMode {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "vendor" => Ok(Self::Vendor),
            "tile" => Ok(Self::Tile),
            "row" => Ok(Self::Row),
            other => Err(format!(
                "GAUDIUM_G4_NORM={other}: expected row, tile or vendor"
            )),
        }
    }

    fn row_norm(self, tile_recipe: bool) -> bool {
        match self {
            Self::Vendor => false,
            Self::Tile => tile_recipe,
            Self::Row => true,
        }
    }
}

/// The four layer shapes a Gemma 4 decoder can hold; one generic prefill recipe per shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum G4LayerKind {
    Sliding,
    SlidingShared,
    Global,
    GlobalShared,
}

impl G4LayerKind {
    pub fn is_global(self) -> bool {
        matches!(self, Self::Global | Self::GlobalShared)
    }

    pub fn is_shared(self) -> bool {
        matches!(self, Self::SlidingShared | Self::GlobalShared)
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Sliding => "s",
            Self::SlidingShared => "ss",
            Self::Global => "g",
            Self::GlobalShared => "gs",
        }
    }
}

/// Everything the graphs and the packer derive from `config.json`, validated once.
#[derive(Debug, Clone)]
pub struct G4Dims {
    pub hid: u64,
    pub inter: u64,
    /// MLP width of a layer without keys of its own (`use_double_wide_mlp`: twice `inter`).
    pub inter_shared: u64,
    pub vocab: u64,
    pub n_layers: usize,
    pub nh: u64,
    pub s_nkv: u64,
    pub s_dh: u64,
    pub g_nkv: u64,
    pub g_dh: u64,
    pub window: u64,
    pub ring: u64,
    /// Global layers have no `v_proj`: V is the unweighted norm of the `k_proj` rows.
    pub keqv: bool,
    pub softcap: f32,
    /// Per-layer input width (0: the model has none).
    pub ple: u64,
    /// First layer without keys of its own (`n_layers` when every layer has keys).
    pub first_shared: usize,
    pub src_s: Option<usize>,
    pub src_g: Option<usize>,
    pub s_pairs: u64,
    pub g_pairs: u64,
    pub s_theta: f64,
    pub g_theta: f64,
    pub norm: G4NormMode,
    /// Projection and MLP matrices run as per-output-channel FP8 GEMMs (`engine/fp8_graph.rs`).
    pub fp8: bool,
    global: Vec<bool>,
}

impl G4Dims {
    /// Derives and checks the family's dims. `max_pass_rows` is the widest pass (prefill tile)
    /// a sliding layer's ring must hold beside its window.
    pub fn from_spec(
        spec: &ModelSpec,
        max_pass_rows: u64,
        norm: G4NormMode,
    ) -> Result<Self, String> {
        if spec.family != ModelFamily::Gemma4 {
            return Err(format!("not a Gemma 4 model: family {:?}", spec.family));
        }
        let g4 = spec
            .gemma4
            .as_ref()
            .ok_or("Gemma 4 model without its gemma4 spec block")?;
        let n_layers = spec.n_layers as usize;
        if n_layers == 0 || spec.layer_types.len() != n_layers {
            return Err(format!(
                "layer_types has {} entries for num_hidden_layers={n_layers}",
                spec.layer_types.len()
            ));
        }
        let mut global = Vec::with_capacity(n_layers);
        for (l, t) in spec.layer_types.iter().enumerate() {
            match t.as_str() {
                "full_attention" => global.push(true),
                "sliding_attention" => global.push(false),
                other => {
                    return Err(format!(
                        "layer_types[{l}] = \"{other}\": a Gemma 4 layer is sliding_attention or full_attention"
                    ))
                }
            }
        }
        if !spec.tie_word_embeddings {
            return Err("tie_word_embeddings=false: the readout reads the embedding table".into());
        }
        if spec.fp8_quantized {
            return Err("quantization_config present: this engine runs BF16 checkpoints".into());
        }
        if !spec.hidden.is_multiple_of(128) || !spec.vocab.is_multiple_of(256) {
            return Err(format!(
                "hidden_size {} must be a multiple of 128 and vocab_size {} of 256",
                spec.hidden, spec.vocab
            ));
        }
        if (spec.rms_eps - 1e-6).abs() > 1e-12 {
            return Err(format!(
                "rms_norm_eps {}: the attention front-end kernel has eps 1e-6 as a literal",
                spec.rms_eps
            ));
        }
        if !(g4.s_rope_type.is_empty() || g4.s_rope_type == "default")
            || (g4.s_partial - 1.0).abs() > 1e-12
        {
            return Err(format!(
                "sliding rope_type \"{}\" / partial factor {}: the sliding tables rotate every pair",
                g4.s_rope_type, g4.s_partial
            ));
        }
        if !(2..=G4_MAX_RING).contains(&g4.sliding_window) {
            return Err(format!(
                "sliding_window {}: 2..{G4_MAX_RING}",
                g4.sliding_window
            ));
        }
        if g4.softcap < 0.0 || (g4.softcap > 0.0 && g4.softcap < 1.0) {
            return Err(format!("final_logit_softcapping {}", g4.softcap));
        }
        if g4.ple_dim % 128 != 0 || g4.ple_dim > 1024 {
            return Err(format!(
                "hidden_size_per_layer_input {}: a multiple of 128, at most 1024",
                g4.ple_dim
            ));
        }
        if g4.ple_dim > 0 && g4.ple_vocab != spec.vocab {
            return Err(format!(
                "vocab_size_per_layer_input {} differs from vocab_size {}",
                g4.ple_vocab, spec.vocab
            ));
        }
        if g4.kv_shared as usize >= n_layers {
            return Err(format!(
                "num_kv_shared_layers {} of {n_layers} layers: at least one layer needs keys",
                g4.kv_shared
            ));
        }
        let first_shared = n_layers - g4.kv_shared as usize;
        let (mut src_s, mut src_g) = (None, None);
        for (l, &is_g) in global.iter().enumerate().take(first_shared) {
            if is_g {
                src_g = Some(l);
            } else {
                src_s = Some(l);
            }
        }
        for (l, &is_g) in global.iter().enumerate().skip(first_shared) {
            if (is_g && src_g.is_none()) || (!is_g && src_s.is_none()) {
                return Err(format!(
                    "layer {l} has no keys of its own and no layer of its kind below layer {first_shared} to read them from"
                ));
            }
        }
        if !global.iter().take(first_shared).any(|&g| g) {
            return Err("no full_attention layer with keys of its own".into());
        }

        let need = g4.sliding_window + max_pass_rows.max(1) - 1;
        let mut ring = 256u64;
        while ring < need {
            ring *= 2;
        }
        if ring > G4_MAX_RING {
            return Err(format!(
                "sliding window {} + a pass of {max_pass_rows} rows needs a ring of {ring} slots (at most {G4_MAX_RING})",
                g4.sliding_window
            ));
        }

        let s_pairs = Gemma4Rope::rotated_pairs(&g4.s_rope_type, g4.s_partial, spec.head_dim);
        let g_pairs = Gemma4Rope::rotated_pairs(&g4.g_rope_type, g4.g_partial, g4.global_head_dim);
        // One table vector pair per 128 rotated pairs; the kernels take at most half a head.
        for (kind, pairs, dh) in [
            ("sliding", s_pairs, spec.head_dim),
            ("global", g_pairs, g4.global_head_dim),
        ] {
            if pairs == 0 || pairs.div_ceil(128) > dh / 256 {
                return Err(format!(
                    "{kind} RoPE: {pairs} rotated pairs for a head of {dh} channels"
                ));
            }
        }

        Ok(Self {
            hid: spec.hidden,
            inter: spec.intermediate,
            inter_shared: spec.intermediate * if g4.double_wide_mlp { 2 } else { 1 },
            vocab: spec.vocab,
            n_layers,
            nh: spec.n_heads,
            s_nkv: spec.n_kv_heads,
            s_dh: spec.head_dim,
            g_nkv: g4.global_kv_heads,
            g_dh: g4.global_head_dim,
            window: g4.sliding_window,
            ring,
            keqv: g4.k_eq_v,
            softcap: g4.softcap as f32,
            ple: g4.ple_dim,
            first_shared,
            src_s,
            src_g,
            s_pairs,
            g_pairs,
            s_theta: g4.s_theta,
            g_theta: g4.g_theta,
            norm,
            fp8: spec.fp8_weights,
            global,
        })
    }

    pub fn is_global(&self, l: usize) -> bool {
        self.global[l]
    }

    pub fn is_shared(&self, l: usize) -> bool {
        l >= self.first_shared
    }

    pub fn kind(&self, l: usize) -> G4LayerKind {
        match (self.is_global(l), self.is_shared(l)) {
            (false, false) => G4LayerKind::Sliding,
            (false, true) => G4LayerKind::SlidingShared,
            (true, false) => G4LayerKind::Global,
            (true, true) => G4LayerKind::GlobalShared,
        }
    }

    /// MLP width of layer `l`.
    pub fn layer_inter(&self, l: usize) -> u64 {
        if self.is_shared(l) {
            self.inter_shared
        } else {
            self.inter
        }
    }

    /// `(kv heads, head width)` of a layer kind.
    pub fn heads(&self, glob: bool) -> (u64, u64) {
        if glob {
            (self.g_nkv, self.g_dh)
        } else {
            (self.s_nkv, self.s_dh)
        }
    }

    /// The layer whose K / V layer `l` attends (itself unless it has no keys of its own).
    pub fn kv_source(&self, l: usize) -> usize {
        if !self.is_shared(l) {
            l
        } else if self.is_global(l) {
            self.src_g.expect("checked in from_spec")
        } else {
            self.src_s.expect("checked in from_spec")
        }
    }

    fn rope_width(&self, glob: bool) -> u64 {
        128 * if glob { self.g_pairs } else { self.s_pairs }.div_ceil(128)
    }

    /// Every layer kind present, in a stable order.
    pub fn kinds(&self) -> Vec<G4LayerKind> {
        let mut v: Vec<G4LayerKind> = (0..self.n_layers).map(|l| self.kind(l)).collect();
        v.sort();
        v.dedup();
        v
    }
}

/// Decode-step bucket (chunks of the global caches a step reads): the smallest rung of the ladder
/// `1, 2, 3, 4, 6, 8, 12, 16, ...` that holds `need` chunks, capped at `max_chunks`.
pub fn step_bucket(need: u64, max_chunks: u64) -> u64 {
    let need = need.clamp(1, max_chunks);
    let mut p = 1u64;
    loop {
        if p >= need {
            return p.min(max_chunks);
        }
        let mid = p + p / 2;
        if p >= 2 && mid >= need {
            return mid.min(max_chunks);
        }
        p *= 2;
    }
}

// ============================================================================
// Pure graph builders (host-testable without the card)
// ============================================================================

fn kv_names(glob: bool) -> (&'static str, &'static str) {
    if glob {
        ("kcache", "vcache")
    } else {
        ("kring", "vring")
    }
}

fn rope_names(glob: bool) -> (&'static str, &'static str) {
    if glob {
        ("rope_c", "rope_s")
    } else {
        ("rope_sc", "rope_ss")
    }
}

/// Gemma 4 RMS norm with a plain weight, rounded once: `y = bf16(x * rsqrt(mean(x^2) + eps) * w)`
/// over the `c` channels of each of the `cols` rows.
fn g4_norm(
    gb: &mut GraphBuilder,
    x: TensorId,
    w_name: &str,
    y_name: &str,
    c: u64,
    cols: u64,
    row: bool,
) -> TensorId {
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Bf16, &[c]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[c, cols]);
    if row {
        gb.add_node(
            "gaudium_ga_rmsnorm",
            format!("{y_name}_norm"),
            &[x, w],
            &[y],
            &RMS_EPS.to_le_bytes(),
        );
    } else {
        let ir = gb.add_persistent_tensor(format!("{y_name}_ir"), SynDataType::Float, &[1, cols]);
        gb.op_rms_norm_bf16(format!("{y_name}_norm"), x, w, y, ir, RMS_EPS, false);
    }
    y
}

/// BF16 linear projection `y {n, t} = W {k, n}^T . x {k, t}`.
fn g4_linear(
    gb: &mut GraphBuilder,
    x: TensorId,
    w_name: &str,
    k: u64,
    n: u64,
    t: u64,
    y_name: &str,
) -> TensorId {
    if gb.fp8_weights() && is_fp8_matrix(w_name) {
        return add_linear_fp8(gb, x, w_name, k, n, t, y_name);
    }
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Bf16, &[k, n]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[n, t]);
    gb.op_gemm(format!("{y_name}_gemm"), x, w, y, false, true);
    y
}

/// A new graph of this family: kernel fingerprints, and for FP8 weights the FP8 GEMM form with
/// the graph compiler's persistent-output reuse off. (The compiler splits an FP8 GEMM into an MME
/// node writing unscaled sums and a scaling node, and may park those sums in the memory of a
/// persistent output a later node writes; a ring or a cache is written only in part by its
/// kernel, so the rest would keep them.)
fn g4_graph(d: &G4Dims, name: String, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = GraphBuilder::new(name);
    record_elfs(&mut gb, elf_hashes);
    gb.set_fp8_weights(d.fp8);
    gb.set_noreuse(d.fp8);
    gb
}

/// A second view of `x` with other dims (no data moves: a workspace tensor behind a reshape).
fn g4_view(
    gb: &mut GraphBuilder,
    x: TensorId,
    name: &str,
    dtype: SynDataType,
    dims: &[u64],
) -> TensorId {
    let v = gb.add_intermediate(name, dtype, dims);
    gb.op_reshape(format!("{name}_rs"), x, v);
    v
}

/// Where one layer's attention state and inputs come from.
struct G4LayerIo<'a> {
    /// Tensor-name prefix of the layer's weights and activations (`"L7_"`, or `"L_"` generic).
    lpfx: &'a str,
    /// Prefix of the K / V tensors it attends (its own, or its key source's).
    kv_pfx: &'a str,
    cursor: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    /// The layer's per-layer input `{ple, t}`.
    pli: Option<TensorId>,
    /// A sliding layer of a tile with image rows: positions after its own that each row also
    /// attends (`ext {t}`, the rest of its image block; the ring softmax with an extension).
    ext: Option<TensorId>,
    /// Only the layer's key / value write (a later tile's keys put into the ring before an
    /// earlier tile's image block attends them): the graph ends after the attention front end.
    kv_only: bool,
}

/// Appends one Gemma 4 decoder layer over `t` rows at positions `cursor + row`:
///
/// ```text
/// h = norm_in(x);  q, k [, v] = W h;  q / K = rope(norm(.) w),  V = norm(v or k)
/// a = softmax(q K^T + mask) V                      (scale 1; f32 scores and sums)
/// x2 = x + norm_pa(W_o a)
/// xs = x2 + norm_pf(W_down (gelu_tanh(W_gate n) * W_up n)),   n = norm_post(x2)
/// [xs = xs + norm_pl(W_pp (gelu_tanh(W_pg xs) * pli))]         per-layer input
/// out = xs * layer_scalar
/// ```
#[allow(clippy::too_many_arguments)]
fn append_g4_layer(
    gb: &mut GraphBuilder,
    d: &G4Dims,
    x: TensorId,
    io: &G4LayerIo<'_>,
    glob: bool,
    shared: bool,
    inter: u64,
    t: u64,
    nch: u64,
    row: bool,
) -> TensorId {
    let l = io.lpfx;
    let hid = d.hid;
    let nh = d.nh;
    let (nkv, dh) = d.heads(glob);
    let gq = nh / nkv;
    let r = gq * t;
    let (c, nch) = if glob { (G4_CHUNK, nch) } else { (d.ring, 1) };
    let b = nkv * nch;
    let keqv = glob && d.keqv;
    let (kk, vk) = kv_names(glob);

    let h = g4_norm(gb, x, &format!("{l}w_in"), &format!("{l}h"), hid, t, row);
    let q = g4_linear(
        gb,
        h,
        &format!("{l}W_qg"),
        hid,
        nh * dh,
        t,
        &format!("{l}qg"),
    );

    let kcache;
    let vcache;
    let qb;
    if shared {
        let qn_w = gb.get_or_add_persistent_tensor(format!("{l}qn_w"), SynDataType::Float, &[dh]);
        qb = gb.add_persistent_tensor(format!("{l}qb"), SynDataType::Bf16, &[dh, r, b]);
        kcache = gb.get_or_add_persistent_tensor(
            format!("{}{kk}", io.kv_pfx),
            SynDataType::Bf16,
            &[dh, c, b],
        );
        vcache = gb.get_or_add_persistent_tensor(
            format!("{}{vk}", io.kv_pfx),
            SynDataType::Bf16,
            &[dh, c, b],
        );
        // node params: coff = 0, same = 0 (row t at position cursor + t)
        gb.add_node(
            "gaudium_ga_qprep",
            format!("{l}attn_qprep"),
            &[q, qn_w, io.rope_c, io.rope_s, io.cursor],
            &[qb],
            &u32_params(&[0, 0]),
        );
    } else {
        let k = g4_linear(
            gb,
            h,
            &format!("{l}W_k"),
            hid,
            nkv * dh,
            t,
            &format!("{l}k"),
        );
        let v = if keqv {
            None
        } else {
            Some(g4_linear(
                gb,
                h,
                &format!("{l}W_v"),
                hid,
                nkv * dh,
                t,
                &format!("{l}v"),
            ))
        };
        let qn_w = gb.get_or_add_persistent_tensor(format!("{l}qn_w"), SynDataType::Float, &[dh]);
        let kn_w = gb.get_or_add_persistent_tensor(format!("{l}kn_w"), SynDataType::Float, &[dh]);
        qb = gb.add_persistent_tensor(format!("{l}qb"), SynDataType::Bf16, &[dh, r, b]);
        kcache = gb.get_or_add_persistent_tensor(
            format!("{}{kk}", io.kv_pfx),
            SynDataType::Bf16,
            &[dh, c, b],
        );
        vcache = gb.get_or_add_persistent_tensor(
            format!("{}{vk}", io.kv_pfx),
            SynDataType::Bf16,
            &[dh, c, b],
        );
        let mut ins = vec![q, k];
        if let Some(v) = v {
            ins.push(v);
        }
        ins.extend_from_slice(&[qn_w, kn_w, io.rope_c, io.rope_s, io.cursor]);
        // node params: coff = 0, ring (1: one chunk of RING slots, slot = position mod RING)
        gb.add_node(
            "gaudium_ga_prep",
            format!("{l}attn_prep"),
            &ins,
            &[qb, kcache, vcache],
            &u32_params(&[0, u32::from(!glob)]),
        );
    }
    if io.kv_only {
        return qb;
    }

    let scores = gb.add_persistent_tensor(format!("{l}scores"), SynDataType::Float, &[c, r, b]);
    gb.op_batch_gemm(format!("{l}qk"), qb, kcache, scores, false, true);
    let pm = gb.add_persistent_tensor(format!("{l}P"), SynDataType::Bf16, &[c, r, b]);
    let stats = gb.add_persistent_tensor(format!("{l}stats"), SynDataType::Float, &[64, 2, r, b]);
    if glob {
        // node params: T, nch, coff, mode 0 (a row sees positions up to its own)
        gb.add_node(
            "gaudium_attn_softmax_chunk",
            format!("{l}softmax"),
            &[scores, io.cursor],
            &[pm, stats],
            &u32_params(&[t as u32, nch as u32, 0, 0]),
        );
    } else if let Some(ext) = io.ext {
        // node params: T, coff, W, mode (not read): row t at its own position, seeing the
        // positions (p - W, p + ext[t]]
        gb.add_node(
            "gaudium_ga_softmax_ringx",
            format!("{l}softmax"),
            &[scores, io.cursor, ext],
            &[pm, stats],
            &u32_params(&[t as u32, 0, d.window as u32, 0]),
        );
    } else {
        // node params: T, coff, W, mode 0 (row t at its own position, masked by ring distance)
        gb.add_node(
            "gaudium_ga_softmax_ring",
            format!("{l}softmax"),
            &[scores, io.cursor],
            &[pm, stats],
            &u32_params(&[t as u32, 0, d.window as u32, 0]),
        );
    }
    let ao_part = gb.add_persistent_tensor(format!("{l}ao_part"), SynDataType::Float, &[dh, r, b]);
    gb.op_batch_gemm(format!("{l}pv"), pm, vcache, ao_part, false, false);
    let ao = gb.add_persistent_tensor(format!("{l}ao"), SynDataType::Float, &[dh, r, nkv]);
    gb.add_node(
        "gaudium_ga_combine",
        format!("{l}combine"),
        &[ao_part, stats],
        &[ao],
        &u32_params(&[nch as u32]),
    );
    let core = gb.add_persistent_tensor(format!("{l}core"), SynDataType::Bf16, &[nh * dh, t]);
    gb.add_node(
        "gaudium_ga_post",
        format!("{l}attn_post"),
        &[ao],
        &[core],
        &[],
    );
    let attn = g4_linear(
        gb,
        core,
        &format!("{l}W_o"),
        nh * dh,
        hid,
        t,
        &format!("{l}attn"),
    );

    let pa = g4_norm(
        gb,
        attn,
        &format!("{l}w_pa"),
        &format!("{l}pa"),
        hid,
        t,
        row,
    );
    let x2 = gb.add_persistent_tensor(format!("{l}x2"), SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16(format!("{l}x2_add"), x, pa, x2);
    let h2 = g4_norm(
        gb,
        x2,
        &format!("{l}w_post"),
        &format!("{l}h2"),
        hid,
        t,
        row,
    );
    let gate = g4_linear(
        gb,
        h2,
        &format!("{l}W_gate"),
        hid,
        inter,
        t,
        &format!("{l}gate"),
    );
    let up = g4_linear(
        gb,
        h2,
        &format!("{l}W_up"),
        hid,
        inter,
        t,
        &format!("{l}up"),
    );
    let m = gb.add_persistent_tensor(format!("{l}m"), SynDataType::Bf16, &[inter, t]);
    gb.add_node(
        "gaudium_ga_gelu",
        format!("{l}gelu"),
        &[gate, up],
        &[m],
        &[],
    );
    let mo = g4_linear(
        gb,
        m,
        &format!("{l}W_down"),
        inter,
        hid,
        t,
        &format!("{l}mlp"),
    );
    let pf = g4_norm(gb, mo, &format!("{l}w_pf"), &format!("{l}pf"), hid, t, row);
    let mut xs = gb.add_persistent_tensor(format!("{l}xs"), SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16(format!("{l}xs_add"), x2, pf, xs);

    if let Some(pli) = io.pli {
        let pd = d.ple;
        let pg = g4_linear(gb, xs, &format!("{l}W_pg"), hid, pd, t, &format!("{l}pg"));
        let pmul = gb.add_persistent_tensor(format!("{l}pm"), SynDataType::Bf16, &[pd, t]);
        gb.add_node(
            "gaudium_ga_gelu",
            format!("{l}pgelu"),
            &[pg, pli],
            &[pmul],
            &[],
        );
        let po = g4_linear(gb, pmul, &format!("{l}W_pp"), pd, hid, t, &format!("{l}pp"));
        let pl = g4_norm(gb, po, &format!("{l}w_pl"), &format!("{l}pl"), hid, t, row);
        let xp = gb.add_persistent_tensor(format!("{l}xp"), SynDataType::Bf16, &[hid, t]);
        gb.op_add_bf16(format!("{l}xp_add"), xs, pl, xp);
        xs = xp;
    }

    let ls = gb.get_or_add_persistent_tensor(format!("{l}ls"), SynDataType::Bf16, &[1, 1]);
    let out = gb.add_persistent_tensor(format!("{l}out"), SynDataType::Bf16, &[hid, t]);
    gb.op_mult_bf16(format!("{l}lscale"), xs, ls, out);
    out
}

/// Appends the scaled embedding gather: `x0 = bf16(table[id] * bf16(sqrt(hidden)))`.
fn append_g4_embed(gb: &mut GraphBuilder, d: &G4Dims, ids: TensorId, t: u64) -> TensorId {
    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[d.hid, d.vocab]);
    let sc = gb.get_or_add_persistent_tensor("embed_scale", SynDataType::Bf16, &[1, 1]);
    let e = gb.add_persistent_tensor("x0g", SynDataType::Bf16, &[d.hid, t]);
    let x0 = gb.add_persistent_tensor("x0", SynDataType::Bf16, &[d.hid, t]);
    gb.op_gather_bf16("embed", table, ids, e, 1);
    gb.op_mult_bf16("embed_scale", e, sc, x0);
    x0
}

/// Appends the per-layer inputs of a pass's `t` rows:
///
/// ```text
/// token part    table[id] * sqrt(ple)                       (one table row holds every layer's block)
/// context part  norm(W_plm x0 * hidden^-0.5) per block of `ple` values
/// pli = bf16(bf16(context + token) * 2^-0.5)
/// ```
///
/// `layer_major`: returns one persistent tensor `ple_out {ple, t, layers}` (a prefill tile: each
/// layer's recipe reads its block). Otherwise one `{ple, t}` workspace view per layer (a step).
pub(crate) fn append_g4_ple(
    gb: &mut GraphBuilder,
    d: &G4Dims,
    x0: TensorId,
    ids: TensorId,
    t: u64,
    row: bool,
    layer_major: bool,
) -> Vec<TensorId> {
    let pd = d.ple;
    let nl = d.n_layers as u64;
    let pw = pd * nl;
    let bf = SynDataType::Bf16;
    let table = gb.get_or_add_persistent_tensor("ple_table", bf, &[pw, d.vocab]);
    let sc = gb.get_or_add_persistent_tensor("ple_scale", bf, &[1, 1]);
    let ps = gb.get_or_add_persistent_tensor("plm_scale", bf, &[1, 1]);
    let mx = gb.get_or_add_persistent_tensor("ple_mix", bf, &[1, 1]);
    let t0 = gb.add_persistent_tensor("ple_t0", bf, &[pw, t]);
    let tk = gb.add_persistent_tensor("ple_tok", bf, &[pw, t]);
    gb.op_gather_bf16("ple_gather", table, ids, t0, 1);
    gb.op_mult_bf16("ple_tscale", t0, sc, tk);
    let pr = g4_linear(gb, x0, "W_plm", d.hid, pw, t, "ple_proj");
    let pp = gb.add_persistent_tensor("ple_pscaled", bf, &[pw, t]);
    gb.op_mult_bf16("ple_pscale", pr, ps, pp);
    // the norm runs over each block of `ple` values: the same memory seen as {ple, layers * t}
    let ppv = g4_view(gb, pp, "ple_pv", bf, &[pd, nl * t]);
    let tkv = g4_view(gb, tk, "ple_tv", bf, &[pd, nl * t]);
    let pn = g4_norm(gb, ppv, "w_plm", "ple_pn", pd, nl * t, row);
    let sm = gb.add_persistent_tensor("ple_sum", bf, &[pd, nl * t]);
    let all = gb.add_persistent_tensor("ple_all", bf, &[pd, nl * t]);
    gb.op_add_bf16("ple_add", pn, tkv, sm);
    gb.op_mult_bf16("ple_mixmul", sm, mx, all);
    let o3 = g4_view(gb, all, "ple_o3", bf, &[pd, nl, t]);
    if layer_major {
        let lm = gb.add_persistent_tensor("ple_out", bf, &[pd, t, nl]);
        let tp = TransposeParams {
            permutation: [0, 2, 1, 0, 0],
            tensor_dim: 3,
        };
        gb.add_node(
            "transpose",
            "ple_layer_major",
            &[o3],
            &[lm],
            as_param_bytes(&tp),
        );
        return vec![lm];
    }
    let parts: Vec<TensorId> = (0..nl)
        .map(|l| gb.add_intermediate(format!("ple_part{l}"), bf, &[pd, 1, t]))
        .collect();
    gb.op_split("ple_split", o3, &parts, 1);
    parts
        .iter()
        .enumerate()
        .map(|(l, &p)| g4_view(gb, p, &format!("ple_l{l}"), bf, &[pd, t]))
        .collect()
}

/// Appends the readout of `t` final-norm rows: tied lm_head GEMM, logit soft cap, arg-max.
/// Returns `(logits, tokens)`.
fn append_g4_head(
    gb: &mut GraphBuilder,
    d: &G4Dims,
    h_final: TensorId,
    t: u64,
) -> (TensorId, TensorId) {
    let capped = d.softcap > 0.0;
    let raw_name = if capped { "logits_raw" } else { "logits" };
    let mut logits = g4_linear(gb, h_final, "embed_table", d.hid, d.vocab, t, raw_name);
    if capped {
        let c = gb.add_persistent_tensor("logits", SynDataType::Bf16, &[d.vocab, t]);
        gb.add_node(
            "gaudium_ga_softcap",
            "softcap",
            &[logits],
            &[c],
            &d.softcap.to_le_bytes(),
        );
        logits = c;
    }
    let tokens = gb.add_persistent_tensor("tokens", SynDataType::Int32, &[1, t]);
    gb.add_node("gaudium_ga_argmax", "argmax", &[logits], &[tokens], &[]);
    (logits, tokens)
}

fn record_elfs(gb: &mut GraphBuilder, elf_hashes: &[(String, String)]) {
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
}

/// Prefill tile front end `g_g4_embed_T{t}`: `ids {t}` -> `x0 {hid, t}` and, for a model with
/// per-layer inputs, `ple_out {ple, t, layers}`.
pub fn build_embed_graph(d: &G4Dims, t: u64, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_g4_embed_T{t}"));
    record_elfs(&mut gb, elf_hashes);
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[t]);
    let x0 = append_g4_embed(&mut gb, d, ids, t);
    if d.ple > 0 {
        let _ = append_g4_ple(&mut gb, d, x0, ids, t, d.norm.row_norm(true), true);
    }
    gb
}

/// Generic single-layer prefill recipe of one layer kind: `g_g4_<kind>_T{t}[_C{nch}]`.
pub fn build_layer_graph(
    d: &G4Dims,
    kind: G4LayerKind,
    t: u64,
    lmax: u64,
    nch: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let glob = kind.is_global();
    let name = if glob {
        format!("g_g4_{}_T{t}_C{nch}", kind.tag())
    } else {
        format!("g_g4_{}_T{t}", kind.tag())
    };
    let mut gb = g4_graph(d, name, elf_hashes);
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[d.hid, t]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let (rc_name, rs_name) = rope_names(glob);
    let rw = d.rope_width(glob);
    let rope_c = gb.add_persistent_tensor(rc_name, SynDataType::Float, &[rw, lmax]);
    let rope_s = gb.add_persistent_tensor(rs_name, SynDataType::Float, &[rw, lmax]);
    let pli =
        (d.ple > 0).then(|| gb.add_persistent_tensor("L_pli", SynDataType::Bf16, &[d.ple, t]));
    let io = G4LayerIo {
        lpfx: "L_",
        kv_pfx: "L_",
        cursor,
        rope_c,
        rope_s,
        pli,
        ext: None,
        kv_only: false,
    };
    let inter = if kind.is_shared() {
        d.inter_shared
    } else {
        d.inter
    };
    let _ = append_g4_layer(
        &mut gb,
        d,
        x_in,
        &io,
        glob,
        kind.is_shared(),
        inter,
        t,
        nch,
        d.norm.row_norm(true),
    );
    gb
}

/// A sliding layer of a tile with image rows (`g_g4_sx_T{t}`: each row also attends the rest
/// of its image block, `ext`) or the key / value write of a sliding layer alone (`g_g4_skv_T{t}`:
/// a later tile's keys into the ring before an earlier tile's block attends them). Only the
/// sizes whose decoder lets image rows attend both ways (`use_bidirectional_attention: vision`)
/// build them; they have keys in every layer.
pub fn build_sliding_vis_graph(
    d: &G4Dims,
    t: u64,
    lmax: u64,
    kv_only: bool,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let name = if kv_only {
        format!("g_g4_skv_T{t}")
    } else {
        format!("g_g4_sx_T{t}")
    };
    let mut gb = g4_graph(d, name, elf_hashes);
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[d.hid, t]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let (rc_name, rs_name) = rope_names(false);
    let rw = d.rope_width(false);
    let rope_c = gb.add_persistent_tensor(rc_name, SynDataType::Float, &[rw, lmax]);
    let rope_s = gb.add_persistent_tensor(rs_name, SynDataType::Float, &[rw, lmax]);
    let ext = (!kv_only).then(|| gb.add_persistent_tensor("ext", SynDataType::Int32, &[t]));
    let io = G4LayerIo {
        lpfx: "L_",
        kv_pfx: "L_",
        cursor,
        rope_c,
        rope_s,
        pli: None,
        ext,
        kv_only,
    };
    let _ = append_g4_layer(
        &mut gb,
        d,
        x_in,
        &io,
        false,
        false,
        d.inter,
        t,
        1,
        d.norm.row_norm(true),
    );
    gb
}

/// Final norm over a prefill tile `g_g4_final_T{t}`: `x_in {hid, t}` -> `h_final {hid, t}`.
pub fn build_final_graph(d: &G4Dims, t: u64, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_g4_final_T{t}"));
    record_elfs(&mut gb, elf_hashes);
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[d.hid, t]);
    let _ = g4_norm(
        &mut gb,
        x_in,
        "w_final",
        "h_final",
        d.hid,
        t,
        d.norm.row_norm(true),
    );
    gb
}

/// One-row readout `g_g4_head_T1`: `h_in {hid, 1}` (a final-norm row) -> `logits`, `tokens`.
pub fn build_head_graph(d: &G4Dims, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = GraphBuilder::new("g_g4_head_T1");
    record_elfs(&mut gb, elf_hashes);
    let h_in = gb.add_persistent_tensor("h_in", SynDataType::Bf16, &[d.hid, 1]);
    let _ = append_g4_head(&mut gb, d, h_in, 1);
    gb
}

/// Monolithic decode step `g_g4_step_C{nch}` (one row): embedding, per-layer inputs, every layer,
/// final norm, readout. `nch`: chunks of the global caches the step reads (its bucket).
pub fn build_step_graph(
    d: &G4Dims,
    lmax: u64,
    nch: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = g4_graph(d, format!("g_g4_step_C{nch}"), elf_hashes);
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[1]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let (h_final, _) = append_g4_trunk(&mut gb, d, ids, cursor, lmax, nch, 1);
    let _ = append_g4_head(&mut gb, d, h_final, 1);
    gb
}

/// Appends the whole decoder over `t` rows at positions `cursor + row`: embedding, per-layer
/// inputs, every layer (each writing its rows into its ring / cache), final norm. Returns
/// `(h_final {hid, t}, [rope_c, rope_s, rope_sc, rope_ss])` (the sliding tables when present).
fn append_g4_trunk(
    gb: &mut GraphBuilder,
    d: &G4Dims,
    ids: TensorId,
    cursor: TensorId,
    lmax: u64,
    nch: u64,
    t: u64,
) -> (TensorId, G4Rope) {
    let row = d.norm.row_norm(false);
    let f32t = SynDataType::Float;
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", f32t, &[d.rope_width(true), lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", f32t, &[d.rope_width(true), lmax]);
    let has_sliding = (0..d.n_layers).any(|l| !d.is_global(l));
    let sliding_rope = has_sliding.then(|| {
        (
            gb.get_or_add_persistent_tensor("rope_sc", f32t, &[d.rope_width(false), lmax]),
            gb.get_or_add_persistent_tensor("rope_ss", f32t, &[d.rope_width(false), lmax]),
        )
    });

    let mut x = append_g4_embed(gb, d, ids, t);
    let pli = if d.ple > 0 {
        append_g4_ple(gb, d, x, ids, t, row, false)
    } else {
        Vec::new()
    };
    for l in 0..d.n_layers {
        let glob = d.is_global(l);
        let lpfx = format!("L{l}_");
        let kv_pfx = format!("L{}_", d.kv_source(l));
        let (rc, rs) = if glob {
            (rope_c, rope_s)
        } else {
            sliding_rope.expect("sliding tables declared")
        };
        let io = G4LayerIo {
            lpfx: &lpfx,
            kv_pfx: &kv_pfx,
            cursor,
            rope_c: rc,
            rope_s: rs,
            pli: pli.get(l).copied(),
            ext: None,
            kv_only: false,
        };
        x = append_g4_layer(
            gb,
            d,
            x,
            &io,
            glob,
            d.is_shared(l),
            d.layer_inter(l),
            t,
            nch,
            row,
        );
    }
    let h_final = g4_norm(gb, x, "w_final", "h_final", d.hid, t, row);
    (
        h_final,
        G4Rope {
            global: (rope_c, rope_s),
            sliding: sliding_rope,
        },
    )
}

/// RoPE table tensors of a graph that holds the whole decoder.
struct G4Rope {
    global: (TensorId, TensorId),
    sliding: Option<(TensorId, TensorId)>,
}

// ----------------------------------------------------------------------------
// Tensor roles (which arena a graph tensor's buffer belongs to)
// ----------------------------------------------------------------------------

const GLOBAL_WEIGHTS: &[&str] = &[
    "embed_table",
    "embed_scale",
    "w_final",
    "rope_c",
    "rope_s",
    "rope_sc",
    "rope_ss",
    "ple_table",
    "ple_scale",
    "W_plm",
    "plm_scale",
    "ple_mix",
    "w_plm",
];

const LAYER_WEIGHTS: &[&str] = &[
    "w_in", "w_pa", "w_post", "w_pf", "ls", "W_qg", "W_k", "W_v", "qn_w", "kn_w", "W_o", "W_gate",
    "W_up", "W_down", "W_pg", "W_pp", "w_pl",
];

const KV_SUFFIXES: &[&str] = &["kcache", "vcache", "kring", "vring"];

/// Splits `"L12_W_qg"` into `(Some(12), "W_qg")` and the generic `"L_W_qg"` into `(None, "W_qg")`.
fn split_layer_name(name: &str) -> Option<(Option<usize>, &str)> {
    let rest = name.strip_prefix('L')?;
    let sep = rest.find('_')?;
    let idx = &rest[..sep];
    if idx.is_empty() {
        return Some((None, &rest[sep + 1..]));
    }
    if idx.chars().all(|c| c.is_ascii_digit()) {
        return Some((idx.parse().ok(), &rest[sep + 1..]));
    }
    None
}

fn is_weight_name(name: &str) -> bool {
    GLOBAL_WEIGHTS.contains(&name)
        || spec::is_spec_weight(name)
        || split_layer_name(name)
            .is_some_and(|(_, what)| LAYER_WEIGHTS.contains(&what) || is_fp8_aux_suffix(what))
}

fn is_kv_name(name: &str) -> bool {
    split_layer_name(name).is_some_and(|(_, what)| KV_SUFFIXES.contains(&what))
}

/// Tensors of a prefill recipe that the driver binds to buffers of its own (not shared scratch).
fn is_prefill_special(name: &str) -> bool {
    matches!(
        name,
        "x_in" | "L_out" | "cursor" | "ids" | "x0" | "ple_out" | "L_pli" | "h_final" | "h_in"
    )
}

/// One buffer of `bytes` out of a sealed arena: it starts on the arena's step (`g4v_arena_step`)
/// and takes what `g4v_take_bytes` says (at least one step), which the plan's `g4v_part_bytes`
/// holds.
fn arena_take(arena: &mut HbmArena, bytes: u64) -> Result<u64, String> {
    let (addr, _off) = arena
        .suballoc(g4v_take_bytes(bytes), g4v_arena_step())
        .map_err(|e| format!("arena suballoc: {e}"))?;
    Ok(addr)
}

/// Bytes of the prefill cursor buffer (one `i32`, given a whole buffer).
const PF_CURSOR_BYTES: u64 = 256;

fn gib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn mib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

/// Exact device-memory plan of one configuration, computed from the graphs before any allocation.
#[derive(Debug, Clone, Default)]
struct G4Plan {
    weights: BTreeMap<String, u64>,
    /// Global layers' K / V caches (grow with the context).
    kv: BTreeMap<String, u64>,
    /// Sliding layers' K / V rings (bounded).
    rings: BTreeMap<String, u64>,
    step_acts: BTreeMap<String, u64>,
    pf_scratch: BTreeMap<String, u64>,
    pf_x_bytes: u64,
    pf_ids_bytes: u64,
    pf_ple_bytes: u64,
    pf_hfin_bytes: u64,
    /// Image input: the image path's own weights and activations.
    vis: Option<vision::run::G4vPlan>,
    /// Image input: the tiles of a prefill unit beyond the first (two layer buffers, ids, cursor
    /// and extension table each).
    unit_extra: u64,
}

impl G4Plan {
    fn build(
        d: &G4Dims,
        max_ctx: u64,
        max_chunks: u64,
        tile: u64,
        sp: Option<&spec::G4SpecCfg>,
        vis: Option<&units::VisLoad>,
    ) -> Self {
        let mut p = Self::default();
        let step = build_step_graph(d, max_ctx, max_chunks, &[]);
        for (name, _dt, _dims, bytes) in step.persistent_tensor_specs() {
            if is_weight_name(&name) {
                p.weights.insert(name, bytes);
            } else if is_kv_name(&name) {
                if name.ends_with("ring") {
                    p.rings.insert(name, bytes);
                } else {
                    p.kv.insert(name, bytes);
                }
            } else {
                p.step_acts.insert(name, bytes);
            }
        }
        let mut graphs = vec![
            build_embed_graph(d, tile, &[]),
            build_final_graph(d, tile, &[]),
            build_head_graph(d, &[]),
        ];
        for kind in d.kinds() {
            graphs.push(build_layer_graph(d, kind, tile, max_ctx, max_chunks, &[]));
        }
        if let Some(v) = vis {
            // the embedding stage of image rows (its `vis_feat` is the request's feature buffer)
            // and the two-way block recipes
            graphs.extend(v.sizing_graphs(d, tile, max_ctx));
            p.vis = Some(v.geo.plan_bytes());
            p.unit_extra = v.unit_tiles.saturating_sub(1);
        }
        for gb in &graphs {
            for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
                if is_prefill_special(&name) || is_weight_name(&name) || is_kv_name(&name) {
                    continue;
                }
                let e = p.pf_scratch.entry(name).or_insert(0);
                *e = (*e).max(bytes);
            }
        }
        if let Some(sc) = sp {
            // the speculative step and first-block recipes: drafter weights and tables, and the
            // activations of the widest verify (same names as the one-row step, more rows)
            for gb in sc.sizing_graphs(d, max_ctx, max_chunks) {
                for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
                    if spec::is_bound_tensor(&name) || is_kv_name(&name) {
                        continue;
                    }
                    let map = if spec::is_spec_state(&name) {
                        // a block drafter's context: bounded rings, caches of the context's length
                        if name.ends_with("ring") {
                            &mut p.rings
                        } else {
                            &mut p.kv
                        }
                    } else if is_weight_name(&name) {
                        &mut p.weights
                    } else {
                        &mut p.step_acts
                    };
                    let e = map.entry(name).or_insert(0);
                    *e = (*e).max(bytes);
                }
            }
            for (name, bytes) in sc.table_bytes() {
                p.weights.insert(name, bytes);
            }
            for (name, bytes) in sc.state_bytes() {
                p.step_acts.insert(name, bytes);
            }
        }
        p.pf_x_bytes = d.hid * tile * 2;
        p.pf_ids_bytes = tile * 4;
        p.pf_ple_bytes = d.ple * tile * (d.n_layers as u64) * 2;
        p.pf_hfin_bytes = d.hid * tile * 2;
        p
    }

    fn sum(m: &BTreeMap<String, u64>) -> u64 {
        m.values().map(|&b| g4v_part_bytes(b)).sum()
    }

    fn weights_bytes(&self) -> u64 {
        Self::sum(&self.weights) + self.vis.as_ref().map_or(0, |v| v.weights_bytes())
    }

    fn kv_bytes(&self) -> u64 {
        Self::sum(&self.kv)
    }

    fn ring_bytes(&self) -> u64 {
        Self::sum(&self.rings)
    }

    fn act_bytes(&self) -> u64 {
        Self::sum(&self.step_acts)
            + Self::sum(&self.pf_scratch)
            + 2 * g4v_part_bytes(self.pf_x_bytes)
            + g4v_part_bytes(self.pf_ids_bytes)
            + g4v_part_bytes(PF_CURSOR_BYTES)
            + g4v_part_bytes(self.pf_ple_bytes)
            + g4v_part_bytes(self.pf_hfin_bytes)
            + self.vis.as_ref().map_or(0, |v| v.act_bytes())
            + units::unit_bytes(self.unit_extra, self.pf_x_bytes, self.pf_ids_bytes)
    }

    fn total(&self, ws_bytes: u64) -> u64 {
        self.weights_bytes()
            .saturating_add(self.kv_bytes())
            .saturating_add(self.ring_bytes())
            .saturating_add(self.act_bytes())
            .saturating_add(ws_bytes)
    }
}

/// Host-only device-memory plan of a Gemma 4 configuration, in bytes:
/// `[weights + tables, global K/V caches, sliding rings, activations]` (the shared recipe
/// workspace comes on top). Exactly what `Gemma4Engine::load` allocates for the same inputs.
pub fn hbm_plan_bytes(
    spec: &ModelSpec,
    max_ctx: u64,
    prefill_tile: u64,
    norm: G4NormMode,
) -> Result<[u64; 4], String> {
    let max_ctx = align_up(max_ctx.max(G4_CHUNK), G4_CHUNK);
    let tile = prefill_tile.clamp(1, G4_MAX_PREFILL_TILE);
    let d = G4Dims::from_spec(spec, tile, norm)?;
    let p = G4Plan::build(&d, max_ctx, max_ctx / G4_CHUNK, tile, None, None);
    Ok([
        p.weights_bytes(),
        p.kv_bytes(),
        p.ring_bytes(),
        p.act_bytes(),
    ])
}

pub(crate) fn select_exact_bindings(
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

/// Reads `max_position_embeddings` and the generation config's extra end-of-sequence ids.
fn read_model_limits(cfg: &EngineConfig) -> (u64, Vec<i32>) {
    let mut max_pos = 0u64;
    if let Ok(txt) = std::fs::read_to_string(cfg.model_dir.join("config.json")) {
        if let Ok(root) = parse_json(&txt) {
            let tc = root.get("text_config").unwrap_or(&root);
            max_pos = tc
                .get("max_position_embeddings")
                .and_then(JsonValue::as_u64)
                .unwrap_or(0);
        }
    }
    let mut eos = Vec::new();
    if let Ok(txt) = std::fs::read_to_string(cfg.model_dir.join("generation_config.json")) {
        if let Ok(root) = parse_json(&txt) {
            if let Some(v) = root.get("eos_token_id") {
                if let Some(id) = v.as_i64() {
                    eos.push(id as i32);
                } else if let Some(arr) = v.as_array() {
                    eos.extend(arr.iter().filter_map(JsonValue::as_i64).map(|x| x as i32));
                }
            }
        }
    }
    (max_pos, eos)
}

// ============================================================================
// Live Gaudi 2 Gemma 4 engine
// ============================================================================

/// Live Intel Gaudi 2 execution engine for Gemma 4 text decoders (BF16 or FP8 weights; `K = 0`
/// or speculative with the size's own drafter).
///
/// Field declaration order ensures all `CompiledRecipe`s and `PinnedBuf`s are dropped
/// **before** `dev: HpuDevice` (which drains all streams and releases the device).
pub struct Gemma4Engine {
    spec: ModelSpec,
    dims: G4Dims,
    max_ctx: u64,
    max_chunks: u64,
    prefill_tile: u64,
    eos_ids: Vec<i32>,
    verbose: bool,
    driver_version: String,
    cache: RecipeCache,
    elf_hashes: Vec<(String, String)>,

    // Recipes (must drop before `dev`)
    head: CompiledRecipe,
    step_recipes: BTreeMap<u64, (CompiledRecipe, PreparedLaunch)>, // bucket (chunks)
    embed_recipes: HashMap<u64, CompiledRecipe>,                   // T
    final_recipes: HashMap<u64, CompiledRecipe>,                   // T
    layer_recipes: HashMap<(G4LayerKind, u64, u64), CompiledRecipe>, // (kind, T, chunks; 0 sliding)
    /// Per layer: the pre-validated prefill launch of its current `(T, chunks)`.
    pf_prepared: Vec<Option<(u64, u64, PreparedLaunch)>>,
    /// Speculative decoding state (`--spec`): step recipes and the drafter (must drop before `dev`).
    specrt: Option<spec::G4SpecRt>,
    /// FP8 calibration load: running maxima of every layer's GEMM input sites.
    calib: Option<ActCalibrator>,
    /// Greedy requests use the drafter (off: they run the non-speculative step, for self-checks).
    spec_on: bool,

    // Reusable pinned host staging buffer (must drop before `dev`)
    stage_io: PinnedBuf,

    // HBM address maps (sub-allocated from sealed arenas owned by `dev`)
    weights_addr: BTreeMap<String, u64>,
    kv_addr: BTreeMap<String, (u64, u64)>,
    step_act_addr: BTreeMap<String, u64>,
    pf_scratch: BTreeMap<String, u64>,
    pf_x: [u64; 2],
    pf_ids: u64,
    pf_cursor: u64,
    pf_ple: u64,
    pf_hfin: u64,
    hbm_bytes: [u64; 5],
    live: LiveReporter,

    // Image input (recipes and pinned memory: must drop before `dev`)
    vision: Option<vision::run::G4Vision>,
    vis_load: Option<units::VisLoad>,
    vis_recipes: HashMap<(units::VisRecipe, u64), CompiledRecipe>,
    /// Buffers of the tiles of a prefill unit beyond the first.
    pf_unit: Vec<units::TileBufs>,

    // Device handle (dropped LAST)
    dev: HpuDevice,
}

// SAFETY: `Gemma4Engine` exclusively owns its SynapseAI device context, streams, compiled recipes,
// pinned host buffers and launch descriptors; no handle is accessed concurrently, so moving
// ownership across threads (behind a `Mutex` in the server) is safe.
unsafe impl Send for Gemma4Engine {}

impl Gemma4Engine {
    /// Opens the card, validates the model, plans and seals device memory, uploads the weights.
    pub fn load(cfg: &EngineConfig) -> Result<Self, String> {
        Self::load_or_plan(cfg, None)?
            .ok_or_else(|| "internal error: the load produced no engine".to_string())
    }

    /// Fills the recipe cache for `cfg` and prints its device memory plan WITHOUT opening the
    /// card (graph compilation needs only an initialized SynapseAI): every recipe a load
    /// compiles, and the ones it compiles on first use (`recipes::build_further`), for the
    /// configuration and for `requests`. Safe to run while another process owns the card.
    pub fn build_cache(cfg: &EngineConfig, requests: &[crate::engine::BuildRequest]) -> Result<(), String> {
        if cfg.no_cache {
            return Err("--build-cache writes the recipe cache: drop --no-cache".into());
        }
        Self::load_or_plan(cfg, Some(requests)).map(|_| ())
    }

    /// `build`: the requests of a `--build-cache` run; stop after the recipes and the memory plan
    /// (`Ok(None)`), with no device.
    fn load_or_plan(
        cfg: &EngineConfig,
        build: Option<&[crate::engine::BuildRequest]>,
    ) -> Result<Option<Self>, String> {
        let t_start = Instant::now();
        let plan_only = build.is_some();
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
        let ple_dim = spec
            .gemma4
            .as_ref()
            .ok_or("Gemma4Engine: config.json is not a Gemma 4 model")?
            .ple_dim;
        // Weight precision. FP8 needs this model's own activation-scale table; it is read (and a
        // missing or foreign one refused) before the device is opened.
        // (`auto` with an explicitly named table means FP8 for this family: its checkpoints are
        // BF16, so a table is only ever named to run them in FP8.)
        let mut pcfg = cfg.clone();
        if cfg.precision == crate::engine::fp8_graph::WeightPrecision::Auto
            && cfg.fp8_act.is_some()
            && !cfg.fp8_calibrate
        {
            pcfg.precision = crate::engine::fp8_graph::WeightPrecision::Fp8;
        }
        let act = crate::engine::fp8_pack::resolve_precision(&pcfg, &mut spec)?;
        let fp8 = spec.fp8_weights;

        // BF16: the reference streams of the sizes with per-layer inputs were made with the row
        // norm in prefill tiles, the others with the vendor node everywhere. FP8: the row norm
        // everywhere. The vendor node's `1 / rms` of a row can move by an f32 step with the row
        // count of the pass; an FP8 cast (3 mantissa bits) turns that into different codes, and a
        // multi-row verify would then leave the one-row stream. A calibration pass measures the
        // graphs the FP8 load will run.
        let norm = match std::env::var("GAUDIUM_G4_NORM") {
            Ok(v) if !v.is_empty() => {
                let m = G4NormMode::parse(&v)?;
                if fp8 && m != G4NormMode::Row {
                    return Err(format!(
                        "GAUDIUM_G4_NORM={v} with FP8 weights: FP8 graphs need the row norm (a row's values must not depend on the row count of its pass)"
                    ));
                }
                m
            }
            _ if fp8 || cfg.fp8_calibrate => G4NormMode::Row,
            _ if ple_dim > 0 => G4NormMode::Tile,
            _ => G4NormMode::Vendor,
        };
        let mut spec_cfg = match &cfg.spec {
            Some(sc) => Some(spec::G4SpecCfg::open(sc)?),
            None => None,
        };

        let (max_pos, gen_eos) = read_model_limits(cfg);
        let max_ctx = align_up(cfg.max_ctx.max(G4_CHUNK), G4_CHUNK);
        if max_pos > 0 && max_ctx > max_pos {
            return Err(format!(
                "--max-ctx {max_ctx} exceeds the model's max_position_embeddings {max_pos}"
            ));
        }
        let max_chunks = max_ctx / G4_CHUNK;
        let mut eos_ids = spec.eos_token_ids.clone();
        for id in gen_eos {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }

        // Configure kernel ELFs and the graph-compiler glue BEFORE SynapseAI is loaded.
        let kenv = KernelEnv::configure_from_dir(&cfg.kernels_dir)?;
        let elf_hashes = kenv.fingerprint_hashes(GEMMA4_ELFS)?;
        if let Some(sc) = spec_cfg.as_mut() {
            // the speculative graphs alone carry the extra kernels: the fingerprints of the
            // non-speculative recipes are the same with and without a drafter
            sc.elf_hashes = elf_hashes.clone();
            let extra = sc.extra_elfs();
            sc.elf_hashes.extend(kenv.fingerprint_hashes(&extra)?);
        }
        if !kenv.gc_kernel_path.contains("libgaudium_ga.so") {
            return Err(format!(
                "the Gemma 4 glue library libgaudium_ga.so is missing from {}",
                cfg.kernels_dir.display()
            ));
        }

        let api =
            Arc::new(SynapseApi::load_default().map_err(|e| format!("load libSynapse.so: {e}"))?);
        let driver_version = api.driver_version().unwrap_or_else(|_| "unknown".into());
        // `--build-cache`: SynapseAI without a device. Declared before every recipe: recipes are
        // dropped before the session.
        let _compile_session = if plan_only {
            Some(
                crate::synapse::device::CompileSession::open(Arc::clone(&api))
                    .map_err(|e| format!("initialize SynapseAI (no device): {e}"))?,
            )
        } else {
            None
        };
        let dev_opt = if plan_only {
            eprintln!("[gaudium] build-cache: SynapseAI initialized without a device; the card is not opened");
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

        let cache = if cfg.no_cache {
            RecipeCache::disabled()
        } else if let Some(dir) = &cfg.cache_dir {
            RecipeCache::at_dir(dir)
        } else {
            RecipeCache::from_env()
        };

        // image input: the image path's share of the plan (none without `--vision-max-patches`,
        // a checkpoint without an image path, or a calibration load)
        let vis_geo = if spec.has_vision && !cfg.fp8_calibrate {
            vision::run::G4vGeo::plan(cfg)?
        } else {
            None
        };
        // the drafter must be one that runs beside the image path (the verified table): the
        // size's assistant reads the target's states and keys only, a block drafter embeds
        // prompt rows of its own
        let drafter_kind = spec_cfg
            .as_ref()
            .map(|s| if s.reads_prompt_rows() { "dflash2" } else { "mtp" });
        if vis_geo.is_some() && !crate::core::vision::drafter_runs_with_vision(drafter_kind, spec.family) {
            return Err("image input with a block drafter is not implemented: load without --vision-max-patches or with the size's own -assistant drafter".into());
        }

        // ---- Plan: the largest prefill tile whose exact plan fits the card ----
        let mut tile = cfg.prefill_tile.clamp(1, G4_MAX_PREFILL_TILE);
        let (dims, plan, vis_load) = loop {
            // a sliding layer's ring holds its window beside the widest pass (tile or verify)
            let pass_rows = tile.max(spec_cfg.as_ref().map_or(1, |s| s.max_rows()));
            let d = G4Dims::from_spec(&spec, pass_rows, norm)?;
            if let Some(sc) = spec_cfg.as_mut() {
                sc.tile = tile;
                sc.check_target(&d)?;
            }
            let vl = match &vis_geo {
                Some(g) => Some(units::VisLoad::new(g.clone(), &spec, &d, tile)?),
                None => None,
            };
            let p = G4Plan::build(&d, max_ctx, max_chunks, tile, spec_cfg.as_ref(), vl.as_ref());
            let total = p.total(MIN_SHARED_WORKSPACE_BYTES);
            // `--build-cache`: the widest tile (a load that finds it does not fit settles on a
            // narrower one and compiles that width's recipes on the card)
            let fits = match &dev_opt {
                Some(dev) => dev.check_plan_fits(total, HBM_HEADROOM_BYTES).map(|_| ()),
                None => Ok(()),
            };
            match fits {
                Ok(()) => break (d, p, vl),
                Err(e) if tile <= G4_MIN_PREFILL_TILE => {
                    return Err(format!(
                        "HBM plan does not fit at max_ctx={max_ctx} even with a {tile}-row prefill tile: planned {:.2} GiB (weights {:.2}, global KV {:.2}, rings {:.2}, activations {:.2}, workspace {:.2}) of {:.2} GiB free: {e}",
                        gib(total),
                        gib(p.weights_bytes()),
                        gib(p.kv_bytes()),
                        gib(p.ring_bytes()),
                        gib(p.act_bytes()),
                        gib(MIN_SHARED_WORKSPACE_BYTES),
                        gib(free_hbm),
                    ));
                }
                Err(_) => {
                    if cfg.verbose {
                        eprintln!(
                            "[gaudium] gemma4: plan with a {tile}-row prefill tile needs {:.2} GiB (activations {:.2} GiB); trying {}",
                            gib(total),
                            gib(p.act_bytes()),
                            tile / 2
                        );
                    }
                    tile /= 2;
                }
            }
        };
        let prefill_tile = tile;

        if cfg.verbose {
            eprintln!(
                "[gaudium] device=Gaudi2 driver={} HBM={:.2}/{:.2} GiB family=Gemma4 weights={} layers={} (global={}, sliding={}, without keys={}) hidden={} heads={}q s={}x{} g={}x{} keqv={} ple={} window={} ring={} norm={:?} max_ctx={} ({} chunks) tile={}",
                driver_version,
                gib(free_hbm),
                gib(total_hbm),
                if dims.fp8 { "FP8" } else { "BF16" },
                dims.n_layers,
                (0..dims.n_layers).filter(|&l| dims.is_global(l)).count(),
                (0..dims.n_layers).filter(|&l| !dims.is_global(l)).count(),
                dims.n_layers - dims.first_shared,
                dims.hid,
                dims.nh,
                dims.s_nkv,
                dims.s_dh,
                dims.g_nkv,
                dims.g_dh,
                dims.keqv,
                dims.ple,
                dims.window,
                dims.ring,
                dims.norm,
                max_ctx,
                max_chunks,
                prefill_tile,
            );
            if let Some(sc) = &spec_cfg {
                eprintln!("[gaudium] {}", sc.describe(&dims));
            }
        }

        // ---- Compile the recipes every request needs, before any device allocation ----
        let t_comp = Instant::now();
        let note = |r: &CompiledRecipe| {
            if cfg.verbose {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                    r.name,
                    r.from_cache,
                    mib(r.ws_size)
                );
            }
        };
        let mut head = build_head_graph(&dims, &elf_hashes)
            .compile(Arc::clone(&api), &cache, &driver_version)
            .map_err(|e| format!("compile head: {e}"))?;
        note(&head);
        let mut step1 = build_step_graph(&dims, max_ctx, 1, &elf_hashes)
            .compile(Arc::clone(&api), &cache, &driver_version)
            .map_err(|e| format!("compile decode step (1 chunk): {e}"))?;
        note(&step1);
        let mut max_ws = head.ws_size.max(step1.ws_size);
        // FP8 graphs compile with the persistent-output reuse off, so their intermediates live in
        // the workspace: compile the widest prefill recipes now, before the workspace is sized.
        let mut pre_layers: Vec<((G4LayerKind, u64, u64), CompiledRecipe)> = Vec::new();
        if dims.fp8 {
            for kind in dims.kinds() {
                let mut chunk_counts = vec![1u64];
                if kind.is_global() && max_chunks > 1 {
                    chunk_counts.push(max_chunks);
                }
                for nch in chunk_counts {
                    let r = build_layer_graph(&dims, kind, prefill_tile, max_ctx, nch, &elf_hashes)
                        .compile(Arc::clone(&api), &cache, &driver_version)
                        .map_err(|e| format!("compile prefill layer {kind:?}: {e}"))?;
                    note(&r);
                    max_ws = max_ws.max(r.ws_size);
                    pre_layers.push((recipes::layer_key(kind, prefill_tile, nch), r));
                }
            }
        }
        let specrt = match spec_cfg {
            Some(sc) => {
                let rt = spec::G4SpecRt::prepare(
                    sc,
                    &dims,
                    max_ctx,
                    &api,
                    &cache,
                    &driver_version,
                    cfg.verbose,
                )?;
                max_ws = max_ws.max(rt.max_ws());
                Some(rt)
            }
            None => None,
        };
        // image input: the image path's largest recipes and the decoder's image-tile recipes at
        // the planned tile, so the shared workspace covers them
        let mut vision = vis_load
            .as_ref()
            .map(|vl| -> Result<vision::run::G4Vision, String> {
                let elf = kenv.fingerprint_hashes(vision::graphs::G4V_ELFS)?;
                Ok(vision::run::G4Vision::new(vl.geo.clone(), elf))
            })
            .transpose()?;
        let mut pre_vis: Vec<((units::VisRecipe, u64), CompiledRecipe)> = Vec::new();
        if let (Some(v), Some(vl)) = (vision.as_mut(), vis_load.as_ref()) {
            max_ws = max_ws.max(v.compile_planned(&api, &cache, &driver_version)?);
            let mut elf = elf_hashes.clone();
            elf.extend_from_slice(v.elf());
            let mut gbs = vec![(
                units::VisRecipe::Embed,
                vision::graphs::build_embed_vis_graph(&dims, vl.geo.feat_cols, prefill_tile, &elf),
            )];
            if vl.bidir {
                for (kind, kv) in [
                    (units::VisRecipe::SlidingExt, false),
                    (units::VisRecipe::SlidingKv, true),
                ] {
                    gbs.push((kind, build_sliding_vis_graph(&dims, prefill_tile, max_ctx, kv, &elf)));
                }
            }
            for (kind, gb) in gbs {
                let r = gb
                    .compile(Arc::clone(&api), &cache, &driver_version)
                    .map_err(|e| format!("compile vision {kind:?}: {e}"))?;
                note(&r);
                max_ws = max_ws.max(r.ws_size);
                pre_vis.push(((kind, prefill_tile), r));
            }
            if cfg.verbose {
                eprintln!(
                    "[gaudium] gemma4 vision: {:?} path, images of up to {} soft tokens, {} per request, feature buffer {} columns, two-way blocks {} (units of up to {} tiles), {:.2} GiB weights + {:.2} GiB buffers",
                    vl.geo.d.kind,
                    vision::soft_budget(&vl.geo.spec),
                    vl.geo.max_tokens,
                    vl.geo.feat_cols,
                    vl.bidir,
                    vl.unit_tiles,
                    gib(plan.vis.as_ref().map_or(0, |p| p.weights_bytes())),
                    gib(plan.vis.as_ref().map_or(0, |p| p.act_bytes())),
                );
            }
        }
        if cfg.verbose {
            eprintln!(
                "[gaudium] core recipes ready in {:.2}s (max_ws={:.2} MiB)",
                t_comp.elapsed().as_secs_f64(),
                mib(max_ws)
            );
        }

        // ---- Exact arena sizes, final fit check, allocation, seal ----
        let ws_bytes = align_up(max_ws.max(MIN_SHARED_WORKSPACE_BYTES), 256);
        let weights_bytes = plan.weights_bytes();
        let kv_bytes = plan.kv_bytes();
        let ring_bytes = plan.ring_bytes();
        let act_bytes = plan.act_bytes();
        let planned_total = plan.total(ws_bytes);
        let Some(mut dev) = dev_opt else {
            // `--build-cache`: the plan, and the recipes a load compiles on first use
            eprintln!(
                "[gaudium] build-cache: HBM plan {:.2} GiB with a {prefill_tile}-row prefill tile (weights {:.2}, global KV {:.2}, rings {:.2}, activations {:.2}, workspace {:.2}); the card is not opened, so its free memory is unknown",
                gib(planned_total),
                gib(weights_bytes),
                gib(kv_bytes),
                gib(ring_bytes),
                gib(act_bytes),
                gib(ws_bytes),
            );
            recipes::build_further(
                &recipes::BuildCtx {
                    api: &api,
                    cache: &cache,
                    driver: &driver_version,
                    dims: &dims,
                    max_ctx,
                    max_chunks,
                    tile: prefill_tile,
                    elf: &elf_hashes,
                    vision: vision.as_ref().zip(vis_load.as_ref()),
                    specrt: specrt.as_ref(),
                    ws_bytes,
                    verbose: cfg.verbose,
                },
                build.unwrap_or(&[]),
            )?;
            eprintln!(
                "[gaudium] build-cache: done in {:.1}s",
                t_start.elapsed().as_secs_f64()
            );
            return Ok(None);
        };
        dev.check_plan_fits(planned_total, HBM_HEADROOM_BYTES)
            .map_err(|e| format!("HBM plan check failed: {e}"))?;
        if cfg.verbose {
            eprintln!(
                "[gaudium] HBM plan fits: total={:.2} GiB of {:.2} GiB free | weights={:.2} GiB, global_kv={:.2} GiB, rings={:.2} GiB, acts={:.2} GiB (step {:.1} MiB, prefill scratch {:.1} MiB), ws={:.2} MiB",
                gib(planned_total),
                gib(free_hbm),
                gib(weights_bytes),
                gib(kv_bytes),
                gib(ring_bytes),
                gib(act_bytes),
                mib(G4Plan::sum(&plan.step_acts)),
                mib(G4Plan::sum(&plan.pf_scratch)),
                mib(ws_bytes),
            );
        }

        let mut weights_arena = dev
            .alloc_arena(weights_bytes, HbmCat::CatWeights, "g4_weights_arena")
            .map_err(|e| format!("alloc weights arena: {e}"))?;
        let mut kv_arena = dev
            .alloc_arena(kv_bytes, HbmCat::CatKv, "g4_global_kv_arena")
            .map_err(|e| format!("alloc global KV arena: {e}"))?;
        let mut ring_arena = if ring_bytes > 0 {
            Some(
                dev.alloc_arena(ring_bytes, HbmCat::CatKv, "g4_ring_arena")
                    .map_err(|e| format!("alloc ring arena: {e}"))?,
            )
        } else {
            None
        };
        let mut act_arena = dev
            .alloc_arena(act_bytes, HbmCat::CatScratch, "g4_act_arena")
            .map_err(|e| format!("alloc act arena: {e}"))?;

        let mut weights_addr = BTreeMap::new();
        for (name, &bytes) in &plan.weights {
            weights_addr.insert(name.clone(), arena_take(&mut weights_arena, bytes)?);
        }
        let mut kv_addr = BTreeMap::new();
        for (name, &bytes) in &plan.kv {
            kv_addr.insert(name.clone(), (arena_take(&mut kv_arena, bytes)?, bytes));
        }
        if let Some(arena) = ring_arena.as_mut() {
            for (name, &bytes) in &plan.rings {
                kv_addr.insert(name.clone(), (arena_take(arena, bytes)?, bytes));
            }
        }
        let mut step_act_addr = BTreeMap::new();
        for (name, &bytes) in &plan.step_acts {
            step_act_addr.insert(name.clone(), arena_take(&mut act_arena, bytes)?);
        }
        let mut pf_scratch = BTreeMap::new();
        for (name, &bytes) in &plan.pf_scratch {
            pf_scratch.insert(name.clone(), arena_take(&mut act_arena, bytes)?);
        }
        let pf_x = [
            arena_take(&mut act_arena, plan.pf_x_bytes)?,
            arena_take(&mut act_arena, plan.pf_x_bytes)?,
        ];
        let pf_ids = arena_take(&mut act_arena, plan.pf_ids_bytes)?;
        let pf_cursor = arena_take(&mut act_arena, PF_CURSOR_BYTES)?;
        let pf_ple = arena_take(&mut act_arena, plan.pf_ple_bytes)?;
        let pf_hfin = arena_take(&mut act_arena, plan.pf_hfin_bytes)?;
        // image input: the image path's weights and buffers, the unit tiles' buffers
        let mut vis_w_addr = BTreeMap::new();
        let mut vis_act_addr = BTreeMap::new();
        let mut vis_x = [0u64; 2];
        if let Some(vp) = plan.vis.as_ref() {
            for (name, &bytes) in &vp.weights {
                vis_w_addr.insert(name.clone(), arena_take(&mut weights_arena, bytes)?);
            }
            for (name, &bytes) in &vp.acts {
                vis_act_addr.insert(name.clone(), arena_take(&mut act_arena, bytes)?);
            }
            if vp.x_bytes > 0 {
                vis_x = [
                    arena_take(&mut act_arena, vp.x_bytes)?,
                    arena_take(&mut act_arena, vp.x_bytes)?,
                ];
            }
        }
        let mut pf_unit = Vec::new();
        for _ in 0..plan.unit_extra {
            pf_unit.push(units::TileBufs {
                x: [
                    arena_take(&mut act_arena, plan.pf_x_bytes)?,
                    arena_take(&mut act_arena, plan.pf_x_bytes)?,
                ],
                ids: arena_take(&mut act_arena, plan.pf_ids_bytes)?,
                cursor: arena_take(&mut act_arena, PF_CURSOR_BYTES)?,
                ext: arena_take(&mut act_arena, plan.pf_ids_bytes)?,
            });
        }

        dev.seal_with_shared_workspace(ws_bytes)
            .map_err(|e| format!("seal HBM allocations: {e}"))?;

        let hbm_bytes = [weights_bytes, kv_bytes, ring_bytes, act_bytes, ws_bytes];
        live.set_hbm_bytes(
            hbm_bytes[0],
            hbm_bytes[1],
            hbm_bytes[2],
            hbm_bytes[3],
            hbm_bytes[4],
        );
        live.set_phase("loading", "uploading weights to HBM");

        head.bind_shared_workspace(&dev)
            .map_err(|e| format!("bind head ws: {e}"))?;
        step1
            .bind_shared_workspace(&dev)
            .map_err(|e| format!("bind step ws: {e}"))?;
        for (_, r) in pre_layers.iter_mut() {
            r.bind_shared_workspace(&dev)
                .map_err(|e| format!("bind {} ws: {e}", r.name))?;
        }
        for (_, r) in pre_vis.iter_mut() {
            r.bind_shared_workspace(&dev)
                .map_err(|e| format!("bind {} ws: {e}", r.name))?;
        }
        if let Some(v) = vision.as_mut() {
            v.bind_workspaces(&dev)?;
        }

        // ---- Upload weights and RoPE tables from the mmap'd checkpoint ----
        let t_up = Instant::now();
        let store =
            SafetensorsStore::open(&cfg.model_dir).map_err(|e| format!("open safetensors: {e}"))?;
        let packer = Gemma4Packer::new(&store, &spec, &dims);
        let fpacker = G4Fp8Packer::new(
            &packer,
            &dims,
            act,
            crate::engine::fp8_pack::cache_dir_of(cfg),
        );
        let mut stage_upload = dev
            .alloc_pinned(STAGE_BUF_BYTES)
            .map_err(|e| format!("alloc pinned upload stage: {e}"))?;

        let mut uploaded_bytes = 0u64;
        for (c_name, s_name, glob) in [("rope_c", "rope_s", true), ("rope_sc", "rope_ss", false)] {
            if !weights_addr.contains_key(c_name) {
                continue;
            }
            let (c, s) = if glob {
                packer.global_rope_tables(max_ctx as usize)
            } else {
                packer.sliding_rope_tables(max_ctx as usize)
            };
            for (name, data) in [(c_name, &c), (s_name, &s)] {
                if data.len() as u64 != plan.weights[name] {
                    return Err(format!(
                        "RoPE table '{name}' has {} bytes, the graph tensor {}",
                        data.len(),
                        plan.weights[name]
                    ));
                }
                dev.upload_h2d_chunked(data, weights_addr[name], &mut stage_upload)
                    .map_err(|e| format!("upload {name}: {e}"))?;
                uploaded_bytes += data.len() as u64;
            }
        }
        for (name, &addr) in &weights_addr {
            if name.starts_with("rope_") {
                continue;
            }
            let expected = plan.weights[name];
            let data: std::borrow::Cow<'_, [u8]> = if spec::is_spec_weight(name) {
                let rt = specrt
                    .as_ref()
                    .ok_or_else(|| format!("tensor '{name}' without a speculative runtime"))?;
                std::borrow::Cow::Owned(rt.pack_weight(name, expected)?)
            } else {
                fpacker.pack(name)?
            };
            if data.len() as u64 != expected {
                return Err(format!(
                    "weight '{name}' packed size {} != graph tensor size {expected}",
                    data.len()
                ));
            }
            dev.upload_h2d_chunked(&data, addr, &mut stage_upload)
                .map_err(|e| format!("upload weight '{name}': {e}"))?;
            uploaded_bytes += expected;
        }
        if let Some(v) = vision.as_mut() {
            for (name, &addr) in &vis_w_addr {
                let data = vision::run::pack_weight(&store, &v.geo, name)?;
                let expected = plan.vis.as_ref().map_or(0, |p| p.weights[name]);
                if data.len() as u64 != expected {
                    return Err(format!(
                        "vision weight '{name}' packed size {} != graph tensor size {expected}",
                        data.len()
                    ));
                }
                dev.upload_h2d_chunked(&data, addr, &mut stage_upload)
                    .map_err(|e| format!("upload vision weight '{name}': {e}"))?;
                uploaded_bytes += expected;
            }
            let feat = *pf_scratch
                .get("vis_feat")
                .ok_or("the plan has no feature buffer")?;
            v.attach(&store, &dev, vis_w_addr, vis_act_addr, vis_x, feat)?;
        }
        let fstats = fpacker.stats();
        drop(fpacker);
        drop(store);
        if cfg.verbose {
            let dt = t_up.elapsed().as_secs_f64();
            eprintln!(
                "[gaudium] uploaded {:.2} GiB weights in {:.2}s ({:.2} GiB/s)",
                gib(uploaded_bytes),
                dt,
                gib(uploaded_bytes) / dt.max(1e-6),
            );
            if fstats.matrices > 0 {
                eprintln!(
                    "[gaudium] FP8 weights: {} matrices ({:.2} G codes, {} from the cache), {:.1} s of host quantisation / cache reads",
                    fstats.matrices,
                    fstats.codes as f64 / 1e9,
                    fstats.from_cache,
                    fstats.seconds
                );
            }
        }

        // ---- Zero the K / V state once: a masked key must be a finite number ----
        // (a masked slot's probability is exactly 0, and 0 x finite = 0 in the P . V GEMM; a
        // request never reads a slot it has not written, so no per-request clear is needed.)
        stage_upload.as_mut_slice().fill(0);
        let stage_zero = stage_upload;
        let t_zero = Instant::now();
        for &(addr, bytes) in kv_addr.values() {
            dev.zero_hbm_chunked(addr, bytes, &stage_zero)
                .map_err(|e| format!("zero K/V state: {e}"))?;
        }
        if cfg.verbose {
            eprintln!(
                "[gaudium] zeroed {:.2} GiB of K/V state in {:.2}s; total init {:.2}s",
                gib(kv_bytes + ring_bytes),
                t_zero.elapsed().as_secs_f64(),
                t_start.elapsed().as_secs_f64()
            );
        }

        drop(stage_zero);

        let stage_io_bytes = ((spec.vocab as usize) * 2)
            .max((prefill_tile as usize) * 4)
            .max(4096);
        let stage_io = dev
            .alloc_pinned(stage_io_bytes)
            .map_err(|e| format!("alloc pinned io stage: {e}"))?;

        let n_layers = dims.n_layers;
        let mut engine = Self {
            spec,
            dims,
            max_ctx,
            max_chunks,
            prefill_tile,
            eos_ids,
            verbose: cfg.verbose,
            driver_version,
            cache,
            elf_hashes,
            head,
            step_recipes: BTreeMap::new(),
            embed_recipes: HashMap::new(),
            final_recipes: HashMap::new(),
            layer_recipes: pre_layers.into_iter().collect(),
            pf_prepared: (0..n_layers).map(|_| None).collect(),
            specrt,
            calib: cfg.fp8_calibrate.then(ActCalibrator::new),
            spec_on: true,
            stage_io,
            weights_addr,
            kv_addr,
            step_act_addr,
            pf_scratch,
            pf_x,
            pf_ids,
            pf_cursor,
            pf_ple,
            pf_hfin,
            hbm_bytes,
            live,
            vision,
            vis_load,
            vis_recipes: pre_vis.into_iter().collect(),
            pf_unit,
            dev,
        };
        let prepared = engine.prepare_step(&step1)?;
        engine.step_recipes.insert(1, (step1, prepared));
        spec::finish(&mut engine)?;
        engine.live.set_phase("idle", "ready");
        Ok(Some(engine))
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// The image settings when the engine was loaded with image input.
    pub fn vision_spec(&self) -> Option<&crate::vision::VisionSpec> {
        self.vision.as_ref().map(|v| v.spec())
    }

    /// Hands the images of the next request to the engine, in prompt order.
    pub fn set_request_images(
        &mut self,
        images: Vec<crate::vision::PreparedImage>,
    ) -> Result<(), String> {
        match self.vision.as_mut() {
            Some(v) => v.set_request_images(images),
            None if images.is_empty() => Ok(()),
            None => Err("this engine was loaded without image input".into()),
        }
    }

    /// Image timings of the last request (`None`: it had no images).
    pub fn last_vision_report(&self) -> Option<crate::vision::VisionReport> {
        self.vision.as_ref().and_then(|v| v.last_report())
    }

    /// Soft tokens of the last image request read back from HBM (`tokens x hidden` BF16).
    pub fn read_vision_embeddings(&self) -> Result<Vec<u8>, String> {
        self.vision
            .as_ref()
            .ok_or("this engine was loaded without image input")?
            .read_features(&self.dev)
    }

    pub fn max_ctx(&self) -> u64 {
        self.max_ctx
    }

    /// Prefill tile the memory plan settled on.
    pub fn prefill_tile(&self) -> u64 {
        self.prefill_tile
    }

    /// Reader of this engine's measured live `profile` block.
    pub fn live_profile(&self) -> crate::live_report::ProfileHandle {
        self.live.profile_handle()
    }

    /// The speculative runtime this engine was loaded with (`None`: a `K = 0` engine). The
    /// size's own drafter drafts one token per pass, as an MTP head does (`mtp`); a block
    /// drafter is reported under the block drafters' name.
    pub fn spec_info(&self) -> Option<super::qwen3::spec::readout::SpecInfo> {
        let cfg = self.specrt.as_ref()?.cfg();
        let mut draft_k = cfg.ks.clone();
        draft_k.sort_unstable_by(|a, b| b.cmp(a));
        Some(super::qwen3::spec::readout::SpecInfo {
            method: match cfg.drafter {
                spec::G4Drafter::Assist(_) => "mtp",
                spec::G4Drafter::DFlash(_) => "dflash2",
            },
            draft_k,
            samples: true,
            tree: None,
            drafter_matrix: None,
        })
    }

    /// Exact HBM ledger bytes `[weights, global K/V caches, sliding rings, scratch, workspace]`.
    pub fn hbm_bytes(&self) -> [u64; 5] {
        self.hbm_bytes
    }

    /// The activation-scale table of a calibration load (`EngineConfig::fp8_calibrate`): the
    /// running maxima over every prefill so far. `None`: not a calibration load.
    pub fn fp8_calibration(&self) -> Option<BTreeMap<String, crate::fp8::ActScale>> {
        self.calib.as_ref().map(ActCalibrator::table)
    }

    /// Turns the loaded drafter off or on for the requests that follow (a self-check runs a
    /// request both ways in one process).
    pub fn set_speculative(&mut self, on: bool) -> Result<(), String> {
        if self.specrt.is_none() {
            return Err("this engine was loaded without a drafter (--spec)".into());
        }
        self.spec_on = on;
        Ok(())
    }

    fn precision_label(&self) -> &'static str {
        if self.dims.fp8 {
            "FP8"
        } else {
            "BF16"
        }
    }

    fn compile(&self, gb: &GraphBuilder, what: &str) -> Result<CompiledRecipe, String> {
        let t0 = Instant::now();
        let mut r = gb
            .compile(
                Arc::clone(self.dev.api()),
                &self.cache,
                &self.driver_version,
            )
            .map_err(|e| format!("compile {what}: {e}"))?;
        r.bind_shared_workspace(&self.dev)
            .map_err(|e| format!("bind {what} workspace: {e}"))?;
        if self.verbose {
            eprintln!(
                "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                r.name,
                r.from_cache,
                mib(r.ws_size),
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(r)
    }

    /// Pre-validates the one binding list of a decode-step recipe.
    fn prepare_step(&self, r: &CompiledRecipe) -> Result<PreparedLaunch, String> {
        let mut pool = BTreeMap::new();
        for (k, &v) in &self.weights_addr {
            pool.insert(k.clone(), v);
        }
        for (k, &(v, _)) in &self.kv_addr {
            pool.insert(k.clone(), v);
        }
        for (k, &v) in &self.step_act_addr {
            pool.insert(k.clone(), v);
        }
        let exact = select_exact_bindings(r, &pool)?;
        r.prepare_launch(&exact)
            .map_err(|e| format!("prepare decode step launch: {e}"))
    }

    fn ensure_step(&mut self, bucket: u64) -> Result<(), String> {
        if self.step_recipes.contains_key(&bucket) {
            return Ok(());
        }
        let gb = build_step_graph(&self.dims, self.max_ctx, bucket, &self.elf_hashes);
        let r = self.compile(&gb, &format!("decode step ({bucket} chunks)"))?;
        let prepared = self.prepare_step(&r)?;
        self.step_recipes.insert(bucket, (r, prepared));
        Ok(())
    }

    /// Compiles (or loads) a prefill recipe on first use.
    fn ensure_pf(&mut self, r: recipes::PfRecipe) -> Result<(), String> {
        use recipes::PfRecipe;
        match r {
            PfRecipe::Embed(t) => self.ensure_embed(t),
            PfRecipe::Final(t) => self.ensure_final(t),
            PfRecipe::Layer(kind, t, nch) => self.ensure_layer(kind, t, nch),
            PfRecipe::Vis(kind, t) => self.ensure_vis(kind, t),
        }
    }

    fn pf_graph(&self, r: recipes::PfRecipe) -> Result<GraphBuilder, String> {
        recipes::pf_graph(&self.dims, self.max_ctx, &self.elf_hashes, None, r)
    }

    fn ensure_embed(&mut self, t: u64) -> Result<(), String> {
        if !self.embed_recipes.contains_key(&t) {
            let gb = self.pf_graph(recipes::PfRecipe::Embed(t))?;
            let r = self.compile(&gb, &format!("embed T={t}"))?;
            self.embed_recipes.insert(t, r);
        }
        Ok(())
    }

    fn ensure_final(&mut self, t: u64) -> Result<(), String> {
        if !self.final_recipes.contains_key(&t) {
            let gb = self.pf_graph(recipes::PfRecipe::Final(t))?;
            let r = self.compile(&gb, &format!("final norm T={t}"))?;
            self.final_recipes.insert(t, r);
        }
        Ok(())
    }

    fn ensure_layer(&mut self, kind: G4LayerKind, t: u64, nch: u64) -> Result<(), String> {
        let key = recipes::layer_key(kind, t, nch);
        if !self.layer_recipes.contains_key(&key) {
            let gb = self.pf_graph(recipes::PfRecipe::Layer(kind, t, nch))?;
            let r = self.compile(&gb, &format!("prefill layer {kind:?} T={t} C={nch}"))?;
            self.layer_recipes.insert(key, r);
        }
        Ok(())
    }

    /// Where layer `l` of a prefill tile reads its input and writes its output: the two tile
    /// buffers in turn, except that a layer a block drafter taps writes its output to the tap's
    /// buffer (and the next layer reads it from there).
    fn prefill_io(&self, l: usize) -> (u64, u64) {
        let (x_in, x_out) = (self.pf_x[l % 2], self.pf_x[(l + 1) % 2]);
        match &self.specrt {
            Some(rt) => rt.prefill_io(&self.step_act_addr, l, x_in, x_out),
            None => (x_in, x_out),
        }
    }

    /// Launches layer `l` of a prefill tile of `t` rows (`nch`: chunks a global layer reads).
    fn launch_prefill_layer(&mut self, l: usize, t: u64, nch: u64) -> Result<(), String> {
        let d = &self.dims;
        let kind = d.kind(l);
        let glob = kind.is_global();
        let key = recipes::layer_key(kind, t, nch);
        let nkey = key.2;
        let recipe = &self.layer_recipes[&key];
        let fresh = !matches!(&self.pf_prepared[l], Some((pt, pn, _)) if *pt == t && *pn == nkey);
        if fresh {
            let mut pool = self.pf_scratch.clone();
            let (x_in, x_out) = self.prefill_io(l);
            pool.insert("x_in".into(), x_in);
            pool.insert("L_out".into(), x_out);
            pool.insert("cursor".into(), self.pf_cursor);
            let (rc, rs) = rope_names(glob);
            pool.insert(rc.into(), self.weights_addr[rc]);
            pool.insert(rs.into(), self.weights_addr[rs]);
            if d.ple > 0 {
                pool.insert("L_pli".into(), self.pf_ple + (l as u64) * d.ple * t * 2);
            }
            // every device tensor of layer `l` (weights, and for FP8 its scales) under its
            // generic name
            let lp = format!("L{l}_");
            for (name, &addr) in self
                .weights_addr
                .range(lp.clone()..)
                .take_while(|(name, _)| name.starts_with(&lp))
            {
                pool.insert(format!("L_{}", &name[lp.len()..]), addr);
            }
            let src = d.kv_source(l);
            let (kk, vk) = kv_names(glob);
            pool.insert(format!("L_{kk}"), self.kv_addr[&format!("L{src}_{kk}")].0);
            pool.insert(format!("L_{vk}"), self.kv_addr[&format!("L{src}_{vk}")].0);
            let exact = select_exact_bindings(recipe, &pool)?;
            let prepared = recipe
                .prepare_launch(&exact)
                .map_err(|e| format!("prepare prefill layer {l} T={t}: {e}"))?;
            self.pf_prepared[l] = Some((t, nkey, prepared));
        }
        let prepared = &self.pf_prepared[l].as_ref().expect("prepared above").2;
        recipe
            .launch_prepared(&self.dev, prepared)
            .map_err(|e| format!("launch prefill layer {l} T={t} C={nch}: {e}"))
    }

    fn sync_compute(&self, what: &str) -> Result<(), String> {
        // SAFETY: the compute stream is owned by `self.dev` and alive for its lifetime.
        unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
            .map_err(|e| format!("sync {what}: {e}"))
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
        params: &GenParams,
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
        Ok(sample_row(&bf16, &dev_sp, 0, seq_pos, true).token as i32)
    }

    /// Runs prefill + autoregressive decode, calling `on_token` per generated token. `cancel` is
    /// checked only between synced launches, so no Synapse launch is ever left queued.
    pub fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        cancel: Option<&AtomicBool>,
        mut on_token: impl FnMut(i32),
    ) -> Result<GenReport, String> {
        // image input: the request's images (taken even when the request fails) against its
        // placeholder rows
        let images = match self.vision.as_mut() {
            Some(v) => v.begin_request(prompt_ids)?,
            None => Vec::new(),
        };
        if !images.is_empty() && self.calib.is_some() {
            return Err("a calibration load takes text prompts only".into());
        }
        if prompt_ids.is_empty() {
            return Err("prompt must contain at least 1 token".into());
        }
        // this engine picks its tokens without the penalties (`sampler_penalty`)
        if !crate::verified::pen_path_allowed(params.penalties.active(), false) {
            return Err("sampling penalties and logit_bias are not applied by the Gemma 4 engine yet".into());
        }
        if let Some(&bad) = prompt_ids
            .iter()
            .find(|&&id| id < 0 || (id as u64) >= self.spec.vocab)
        {
            return Err(format!(
                "prompt token id {bad} is outside the vocabulary (0..{})",
                self.spec.vocab
            ));
        }
        let mut budget =
            SequenceBudget::admit(prompt_ids.len() as u64, Some(params.max_new), self.max_ctx)
                .map_err(|e| format!("SequenceBudget rejected prompt/max_new: {e:?}"))?;

        let mut eos_ids = self.eos_ids.clone();
        for &id in &params.extra_eos_ids {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
        let is_eos = |tok: i32| eos_ids.contains(&tok);
        let is_cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));

        let prec = self.precision_label();
        let method_label = if let Some(rt) = self.specrt.as_ref().filter(|_| self.spec_on) {
            rt.method_label(self.dims.fp8, params.temperature)
        } else if params.temperature > 0.0 {
            format!("K=0 {prec} T={:.2}", params.temperature)
        } else {
            format!("K=0 {prec} greedy")
        };
        self.live
            .begin_request(prompt_ids.len() as u64, &method_label, "");
        self.live.prefill_progress(0);
        let res = self.run_generate_inner(
            prompt_ids,
            &images,
            params,
            &mut budget,
            &is_eos,
            &is_cancelled,
            &mut on_token,
        );
        self.live.end_request();
        if let Some(v) = self.vision.as_mut() {
            v.req = None;
        }
        if res.is_err() {
            // never leave a launch queued behind an error return
            let _ = self.dev.sync_all();
        }
        res
    }

    fn run_generate_inner(
        &mut self,
        prompt_ids: &[i32],
        images: &[crate::vision::PreparedImage],
        params: &GenParams,
        budget: &mut SequenceBudget,
        is_eos: &impl Fn(i32) -> bool,
        is_cancelled: &impl Fn() -> bool,
        on_token: &mut impl FnMut(i32),
    ) -> Result<GenReport, String> {
        let n_layers = self.dims.n_layers;
        let hid = self.dims.hid;
        let kinds = self.dims.kinds();
        let t_prefill = Instant::now();
        let mut generated = Vec::with_capacity(budget.max_new() as usize);
        let mut stopped_by_eos = false;
        let mut stopped_by_cancel = false;
        let cancelled_report = |ttft_ms: f64| GenReport {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: Vec::new(),
            ttft_ms,
            prefill_tok_s: 0.0,
            decode_ms: 0.0,
            decode_tok_s: 0.0,
            ms_per_step: 0.0,
            stopped_by_eos: false,
            stopped_by_cancel: true,
            ..GenReport::default()
        };

        // ---- PREFILL: tiles of up to `prefill_tile` rows through per-layer recipes ----
        let mut pos = 0usize;
        let mut last_tile = 0u64;
        // time spent compiling / loading recipes on first use: reported, not counted as prefill rate
        let mut recipe_ms = 0.0f64;
        // image input: the image path runs first, its soft tokens land in the feature buffer;
        // the prefill then reads them at the image rows
        let mut last_out = None;
        if !images.is_empty() {
            let v = self.vision.as_mut().expect("images are taken by the image path");
            v.encode(
                &self.dev,
                self.dev.api(),
                &self.cache,
                &self.driver_version,
                images,
            )?;
            if let Some(r) = v.last_report() {
                recipe_ms += r.compile_ms;
            }
            match self.prefill_vis(prompt_ids, is_cancelled)? {
                Some((t, out, ms)) => {
                    last_tile = t;
                    last_out = Some(out);
                    recipe_ms += ms;
                    pos = prompt_ids.len();
                }
                None => return Ok(cancelled_report(t_prefill.elapsed().as_secs_f64() * 1000.0)),
            }
        }
        while pos < prompt_ids.len() {
            if is_cancelled() {
                return Ok(cancelled_report(t_prefill.elapsed().as_secs_f64() * 1000.0));
            }
            // (the tiles of `units::split_tiles`, which `--build-cache` plans with)
            let t = (prompt_ids.len() - pos).min(self.prefill_tile as usize) as u64;
            let end_pos = pos as u64 + t;
            let nch = recipes::tile_chunks(end_pos, self.max_chunks);
            last_tile = t;

            let t_rec = Instant::now();
            self.ensure_pf(recipes::embed_recipe(false, t))?;
            for r in recipes::layer_recipes(&kinds, t, nch, false) {
                self.ensure_pf(r)?;
            }
            recipe_ms += t_rec.elapsed().as_secs_f64() * 1000.0;

            let tile_bytes = (t as usize) * 4;
            for (k, &id) in prompt_ids[pos..pos + t as usize].iter().enumerate() {
                self.stage_io.as_mut_slice()[k * 4..k * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
            self.dev
                .copy_h2d_pinned_sync(&self.stage_io, tile_bytes, self.pf_ids)
                .map_err(|e| format!("upload prefill ids: {e}"))?;
            self.write_i32_scalar(self.pf_cursor, pos as i32)?;

            // embedding (+ every layer's per-layer input) of the tile's rows
            {
                let mut pool = self.pf_scratch.clone();
                for name in GLOBAL_WEIGHTS {
                    if let Some(&addr) = self.weights_addr.get(*name) {
                        pool.insert((*name).to_string(), addr);
                    }
                }
                pool.insert("ids".into(), self.pf_ids);
                pool.insert("x0".into(), self.pf_x[0]);
                pool.insert("ple_out".into(), self.pf_ple);
                let r = &self.embed_recipes[&t];
                let bind = select_exact_bindings(r, &pool)?;
                r.launch_checked(&self.dev, &bind)
                    .map_err(|e| format!("launch embed T={t}: {e}"))?;
            }
            for l in 0..n_layers {
                self.launch_prefill_layer(l, t, nch)?;
                if self.calib.is_some() {
                    // the generic layer recipe's site buffers hold layer l's rows until layer
                    // l + 1 runs
                    self.sync_compute("calibration layer")?;
                    let mut cal = self.calib.take().expect("checked");
                    let res = crate::engine::gemma4_fp8::calibrate_g4_layer(
                        &mut cal,
                        &self.dev,
                        &self.dims,
                        &self.pf_scratch,
                        l,
                        t,
                    );
                    self.calib = Some(cal);
                    res?;
                }
            }
            // a block drafter's context of the tile's rows (from the tapped layer outputs)
            spec::prefill_tile_hook(self, pos as u64, t)?;
            self.sync_compute("prefill tile")?;
            pos += t as usize;
            self.live.prefill_progress(pos as u64);
        }

        // final norm over the last tile, then the readout of its last row
        let t_rec = Instant::now();
        self.ensure_final(last_tile)?;
        recipe_ms += t_rec.elapsed().as_secs_f64() * 1000.0;
        {
            let mut pool = self.pf_scratch.clone();
            pool.insert(
                "x_in".into(),
                last_out.unwrap_or_else(|| self.prefill_io(n_layers - 1).1),
            );
            pool.insert("w_final".into(), self.weights_addr["w_final"]);
            pool.insert("h_final".into(), self.pf_hfin);
            let r = &self.final_recipes[&last_tile];
            let bind = select_exact_bindings(r, &pool)?;
            r.launch_checked(&self.dev, &bind)
                .map_err(|e| format!("launch final norm T={last_tile}: {e}"))?;
        }
        {
            let mut pool = self.pf_scratch.clone();
            pool.insert("h_in".into(), self.pf_hfin + (last_tile - 1) * hid * 2);
            pool.insert("embed_table".into(), self.weights_addr["embed_table"]);
            let bind = select_exact_bindings(&self.head, &pool)?;
            self.head
                .launch_checked(&self.dev, &bind)
                .map_err(|e| format!("launch head: {e}"))?;
        }
        self.sync_compute("prefill readout")?;
        let first_tok = if params.temperature > 0.0 {
            self.sample_from_logits(
                self.pf_scratch["logits"],
                params,
                (prompt_ids.len() - 1) as u32,
            )?
        } else {
            self.read_i32_scalar(self.pf_scratch["tokens"])?
        };

        let ttft_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let prefill_tok_s = (prompt_ids.len() as f64) / ((ttft_ms - recipe_ms).max(1e-6) / 1000.0);
        if self.verbose {
            eprintln!(
                "[gaudium] gemma4 prefill: {} tok, {} tiles of <= {} rows, {:.1} ms to first token of which {:.1} ms first-use recipe compile/load (excluded from the prefill rate: {:.1} tok/s)",
                prompt_ids.len(),
                prompt_ids.len().div_ceil(self.prefill_tile as usize),
                self.prefill_tile,
                ttft_ms,
                recipe_ms,
                prefill_tok_s
            );
        }

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

        // ---- DECODE, speculative: an engine loaded with a drafter ----
        if self.specrt.is_some()
            && self.spec_on
            && !stopped_by_eos
            && !stopped_by_cancel
            && !generated.is_empty()
            && budget.remaining() > 0
        {
            let method = self
                .specrt
                .as_ref()
                .map(|rt| rt.method_label(self.dims.fp8, params.temperature))
                .unwrap_or_default();
            let h_addr = self.pf_hfin + (last_tile - 1) * hid * 2;
            let out = if params.temperature > 0.0 {
                spec::decode_sampled(
                    self,
                    prompt_ids.len(),
                    h_addr,
                    params,
                    budget,
                    is_eos,
                    is_cancelled,
                    on_token,
                    &mut generated,
                )?
            } else {
                spec::decode(
                    self,
                    prompt_ids.len(),
                    h_addr,
                    params,
                    budget,
                    is_eos,
                    is_cancelled,
                    on_token,
                    &mut generated,
                )?
            };
            let launches = out.stats.launches;
            return Ok(GenReport {
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
                ..GenReport::default()
            });
        }

        // ---- DECODE: one monolithic launch per token ----
        let step_ids = *self.step_act_addr.get("ids").ok_or("missing step 'ids'")?;
        let step_cursor = *self
            .step_act_addr
            .get("cursor")
            .ok_or("missing step 'cursor'")?;
        let step_tokens = *self
            .step_act_addr
            .get("tokens")
            .ok_or("missing step 'tokens'")?;
        let step_logits = *self
            .step_act_addr
            .get("logits")
            .ok_or("missing step 'logits'")?;

        // a bucket compiled on first use is not decode time: build the first one before the clock
        let mut bucket = step_bucket(
            (prompt_ids.len() as u64 + 1).div_ceil(G4_CHUNK),
            self.max_chunks,
        );
        if !stopped_by_eos && !stopped_by_cancel && budget.remaining() > 0 {
            self.ensure_step(bucket)?;
        }

        let t_decode = Instant::now();
        let mut compile_ms = 0.0f64;
        let mut cur_tok = first_tok;
        let mut decode_steps = 0usize;
        while !stopped_by_eos && !stopped_by_cancel && budget.remaining() > 0 {
            if is_cancelled() {
                stopped_by_cancel = true;
                break;
            }
            let cursor_val = (prompt_ids.len() + decode_steps) as u64;
            let need = step_bucket((cursor_val + 1).div_ceil(G4_CHUNK), self.max_chunks);
            if need != bucket {
                let t0 = Instant::now();
                self.ensure_step(need)?;
                compile_ms += t0.elapsed().as_secs_f64() * 1000.0;
                bucket = need;
            }

            let buf = self.stage_io.as_mut_slice();
            buf[0..4].copy_from_slice(&cur_tok.to_le_bytes());
            buf[64..68].copy_from_slice(&(cursor_val as i32).to_le_bytes());
            self.dev
                .copy_h2d_pinned_batch_sync(
                    &self.stage_io,
                    &[(0, 4, step_ids), (64, 4, step_cursor)],
                )
                .map_err(|e| format!("write step inputs H2D: {e}"))?;

            let (recipe, prepared) = &self.step_recipes[&bucket];
            // [readout hook] the launch mark, with the step's byte model
            self.live.launch_begin_with(bucket, || {
                readout::decode_step(&self.dims, self.max_ctx, bucket)
            });
            recipe.launch_prepared(&self.dev, prepared).map_err(|e| {
                format!("launch decode step {decode_steps} (cursor={cursor_val}): {e}")
            })?;
            self.live.launch_submitted();
            self.sync_compute("decode step")?;

            let next_tok = if params.temperature > 0.0 {
                self.sample_from_logits(step_logits, params, cursor_val as u32)?
            } else {
                self.read_i32_scalar(step_tokens)?
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

        let decode_ms = (t_decode.elapsed().as_secs_f64() * 1000.0 - compile_ms).max(0.0);
        let (decode_tok_s, ms_per_step) = if decode_steps > 0 {
            (
                (decode_steps as f64) / (decode_ms / 1000.0).max(1e-9),
                decode_ms / (decode_steps as f64),
            )
        } else {
            (0.0, 0.0)
        };

        Ok(GenReport {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: generated,
            ttft_ms,
            prefill_tok_s,
            decode_ms,
            decode_tok_s,
            ms_per_step,
            stopped_by_eos,
            stopped_by_cancel,
            launches: decode_steps as u64,
            tokens_per_launch: if decode_steps > 0 { 1.0 } else { 0.0 },
            method: if params.temperature > 0.0 {
                format!("K=0 {} T={:.2}", self.precision_label(), params.temperature)
            } else {
                format!("K=0 {} greedy", self.precision_label())
            },
            ..GenReport::default()
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn spec_json(extra_tc: &str, layers: usize, global_every: usize) -> String {
        let types: Vec<&str> = (0..layers)
            .map(|l| {
                if (l + 1) % global_every == 0 {
                    "\"full_attention\""
                } else {
                    "\"sliding_attention\""
                }
            })
            .collect();
        format!(
            r#"{{
              "architectures": ["Gemma4ForConditionalGeneration"],
              "model_type": "gemma4",
              "tie_word_embeddings": true,
              "eos_token_id": [1, 106],
              "text_config": {{
                "model_type": "gemma4_text",
                "vocab_size": 262144, "num_hidden_layers": {layers},
                "head_dim": 256, "global_head_dim": 512, "final_logit_softcapping": 30.0,
                "rms_norm_eps": 1e-06,
                "layer_types": [{}],
                "rope_parameters": {{
                    "sliding_attention": {{"rope_theta": 10000.0, "rope_type": "default"}},
                    "full_attention": {{"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}}
                }},
                {extra_tc}
              }}
            }}"#,
            types.join(", ")
        )
    }

    pub(crate) fn e2b() -> ModelSpec {
        let tc = r#""hidden_size": 1536, "intermediate_size": 6144, "num_attention_heads": 8,
            "num_key_value_heads": 1, "num_global_key_value_heads": null, "sliding_window": 512,
            "attention_k_eq_v": false, "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144, "num_kv_shared_layers": 20,
            "use_double_wide_mlp": true"#;
        ModelSpec::from_config_json(&spec_json(tc, 35, 5)).unwrap()
    }

    pub(crate) fn b31() -> ModelSpec {
        let tc = r#""hidden_size": 5376, "intermediate_size": 21504, "num_attention_heads": 32,
            "num_key_value_heads": 16, "num_global_key_value_heads": 4, "sliding_window": 1024,
            "attention_k_eq_v": true, "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 262144, "num_kv_shared_layers": 0,
            "use_double_wide_mlp": false"#;
        ModelSpec::from_config_json(&spec_json(tc, 60, 6)).unwrap()
    }

    #[test]
    fn the_plan_covers_what_the_arena_takes() {
        use crate::verified::g4v_add_part;
        // two 257-byte buffers: the second starts on the next step, so with today's 256-byte
        // step the arena needs 769 bytes, which an unaligned count (514) would miss
        let plan = g4v_add_part(g4v_add_part(0, 257), 257);
        let mut a = HbmArena::new(0, plan);
        arena_take(&mut a, 257).unwrap();
        assert_eq!(arena_take(&mut a, 257).unwrap(), 2 * g4v_arena_step());
        assert_eq!(a.used(), 769);
        // sequences of sizes around the step, empty buffers included: an arena of exactly the
        // plan's bytes holds every buffer of the plan
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for round in 0..200 {
            let sizes: Vec<u64> = (0..1 + round % 17)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let s = g4v_arena_step();
                    match (seed >> 60) % 4 {
                        0 => (seed >> 20) % 4,
                        1 => s - 2 + (seed >> 20) % 4,
                        2 => (seed >> 20) % (8 * s),
                        _ => (seed >> 20) % (1 << 22),
                    }
                })
                .collect();
            let plan = sizes.iter().fold(0, |t, &b| g4v_add_part(t, b));
            let mut a = HbmArena::new(0, plan);
            for &b in &sizes {
                arena_take(&mut a, b).unwrap_or_else(|e| panic!("{sizes:?}: {e}"));
            }
            assert!(a.used() <= plan);
        }
    }

    #[test]
    fn dims_of_the_small_and_the_large_size() {
        let s = e2b();
        s.validate().unwrap();
        let d = G4Dims::from_spec(&s, 256, G4NormMode::Tile).unwrap();
        assert_eq!(d.ring, 1024); // window 512 + a 256-row tile
        assert_eq!(d.first_shared, 15);
        assert_eq!((d.src_s, d.src_g), (Some(13), Some(14)));
        assert_eq!(d.kv_source(3), 3);
        assert_eq!(d.kv_source(20), 13);
        assert_eq!(d.kv_source(34), 14);
        assert_eq!(d.layer_inter(14), 6144);
        assert_eq!(d.layer_inter(15), 12288);
        assert_eq!((d.s_pairs, d.g_pairs), (128, 64));
        assert_eq!(d.heads(true), (1, 512));
        assert_eq!(d.kinds().len(), 4);

        let b = b31();
        b.validate().unwrap();
        let d = G4Dims::from_spec(&b, 256, G4NormMode::Vendor).unwrap();
        assert_eq!(d.ring, 2048);
        assert_eq!(d.first_shared, 60);
        assert!(d.keqv);
        assert_eq!(d.kinds(), vec![G4LayerKind::Sliding, G4LayerKind::Global]);
        // a pass too wide for the ring kernel is refused with the numbers
        assert!(G4Dims::from_spec(&b, 4096, G4NormMode::Vendor).is_err());
    }

    #[test]
    fn step_bucket_ladder_covers_every_chunk_count() {
        assert_eq!(step_bucket(1, 64), 1);
        assert_eq!(step_bucket(2, 64), 2);
        assert_eq!(step_bucket(3, 64), 3);
        assert_eq!(step_bucket(5, 64), 6);
        assert_eq!(step_bucket(7, 64), 8);
        assert_eq!(step_bucket(9, 64), 12);
        assert_eq!(step_bucket(33, 64), 48);
        assert_eq!(step_bucket(49, 64), 64);
        assert_eq!(step_bucket(64, 64), 64);
        assert_eq!(step_bucket(5, 5), 5);
        for max in [1u64, 2, 4, 8, 32, 64] {
            let mut prev = 0;
            for need in 1..=max {
                let b = step_bucket(need, max);
                assert!(b >= need && b <= max && b >= prev);
                prev = b;
            }
        }
    }

    #[test]
    fn step_graph_of_the_small_size_shares_keys_and_carries_per_layer_inputs() {
        let s = e2b();
        let d = G4Dims::from_spec(&s, 256, G4NormMode::Tile).unwrap();
        let step = build_step_graph(&d, 4096, 1, &[]);
        let specs = step.persistent_tensor_specs();
        let names: Vec<&str> = specs.iter().map(|t| t.0.as_str()).collect();
        // 15 layers with keys: 12 sliding rings + 3 global caches, K and V each
        assert_eq!(names.iter().filter(|n| n.ends_with("ring")).count(), 24);
        assert_eq!(names.iter().filter(|n| n.ends_with("cache")).count(), 6);
        assert!(!names.contains(&"L20_W_k"));
        assert!(names.contains(&"L20_W_pg") && names.contains(&"ple_table"));
        // a layer without keys has the doubled MLP
        let gate = |l: usize| {
            specs
                .iter()
                .find(|t| t.0 == format!("L{l}_W_gate"))
                .unwrap()
                .2
                .clone()
        };
        assert_eq!(gate(14), vec![1536, 6144]);
        assert_eq!(gate(15), vec![1536, 12288]);
        // every tensor has exactly one role
        for n in &names {
            assert!(!(is_weight_name(n) && is_kv_name(n)), "{n}");
        }
        assert_eq!(
            step.fingerprint("x"),
            build_step_graph(&d, 4096, 1, &[]).fingerprint("x")
        );
        assert_ne!(
            step.fingerprint("x"),
            build_step_graph(&d, 8192, 2, &[]).fingerprint("x")
        );
    }

    #[test]
    fn plan_of_the_large_size_matches_the_shapes() {
        let s = b31();
        let d = G4Dims::from_spec(&s, 256, G4NormMode::Vendor).unwrap();
        let p = G4Plan::build(&d, 4096, 1, 256, None, None);
        // 50 sliding layers: K and V rings of 16 heads x 256 x 2048 slots, BF16
        assert_eq!(p.rings.len(), 100);
        assert_eq!(p.ring_bytes(), 100 * 16 * 256 * 2048 * 2);
        // 10 global layers: K and V caches of 4 heads x 512 x 4096 rows
        assert_eq!(p.kv.len(), 20);
        assert_eq!(p.kv_bytes(), 20 * 4 * 512 * 4096 * 2);
        // k = v: no W_v on a global layer, one on a sliding layer
        assert!(!p.weights.contains_key("L5_W_v"));
        assert!(p.weights.contains_key("L4_W_v"));
        assert!(p.weights.contains_key("rope_sc") && p.weights.contains_key("rope_c"));
        // the prefill scratch holds no weight, K/V or driver-bound tensor
        for n in p.pf_scratch.keys() {
            assert!(
                !is_weight_name(n) && !is_kv_name(n) && !is_prefill_special(n),
                "{n}"
            );
        }
        // the full context: the global caches grow 64-fold, the rings do not
        let p64 = G4Plan::build(&d, 262_144, 64, 256, None, None);
        assert_eq!(p64.kv_bytes(), 64 * p.kv_bytes());
        assert_eq!(p64.ring_bytes(), p.ring_bytes());
        // generic layer recipes name every weight of their kind with the generic prefix
        let lg = build_layer_graph(&d, G4LayerKind::Global, 256, 4096, 1, &[]);
        let names: Vec<String> = lg
            .persistent_tensor_specs()
            .into_iter()
            .map(|t| t.0)
            .collect();
        assert!(names.contains(&"L_kcache".to_string()) && !names.contains(&"L_W_v".to_string()));
        assert_eq!(split_layer_name("L_W_qg"), Some((None, "W_qg")));
        assert_eq!(split_layer_name("L12_kring"), Some((Some(12), "kring")));
        assert_eq!(split_layer_name("logits"), None);
    }
}
