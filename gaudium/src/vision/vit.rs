//! Vision tower on the card: three recipes per patch count `N`, launched per image.
//!
//! - `g_vit_patch_N`: `x0 = pix {row, N} . W_patch + b + pos {H, N}`. The patch convolution has
//!   stride = kernel, so it is one MME GEMM over the processor's patch rows.
//! - `g_vit_block_N`: ONE generic block recipe, launched `depth` times and bound at launch to
//!   block `l`'s weights (`vb_*` -> `V<l>_*`). A whole tower in one graph would compile
//!   superlinearly; a generic block compiles once per image size. Inside: LayerNorm, five GEMMs
//!   (q, k, v and the `rotate_half` twins of q / k), the 2-D rotary product in F32, heads moved
//!   to the batch dimension, per-head `batch_gemm` scores in F32, softmax, `P . V`, projection,
//!   residual, LayerNorm, MLP with GELU, residual.
//! - `g_vit_merge_N`: LayerNorm, view `{4H, N / 4}` (patch rows arrive in 2x2 merge order, so
//!   four consecutive rows are one block), fc1, GELU, fc2 -> decoder embeddings `{HID, N / 4}`.
//!
//! Every activation is a persistent tensor, so the tower's device memory is known exactly from
//! the graph description before anything is compiled or allocated: weights and activation
//! buffers (sized for the largest admitted image) join the engine's up-front HBM plan, and
//! recipes for other image sizes compiled later bind the same buffers and the shared workspace.
//! Attention is bidirectional inside one image; images run one after another.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use crate::safetensors::SafetensorsStore;
use crate::synapse::device::{HbmArena, HpuDevice, PinnedBuf};
use crate::synapse::ffi::{SynDataType, SynapseApi};
use crate::synapse::graph::{CompiledRecipe, GraphBuilder, PreparedLaunch, TensorId};
use crate::synapse::ops::{as_param_bytes, TransposeParams};
use crate::synapse::recipe_cache::RecipeCache;
use crate::verified::align_up;
use crate::vision::mrope::{mrope_rows, PromptLayout};
use crate::vision::packer::{pos_embed, vit_rope_tables, VisionPacker};
use crate::vision::preprocess::PreparedImage;
use crate::vision::VisionSpec;

/// `ns_LayerNormKernel::Params` (`layer_norm_fwd_bf16`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LayerNormParams {
    eps_valid: u8,
    _pad: [u8; 3],
    eps: f32,
}

/// `ns_GeluKernel::Params` (`gelu_fwd_f32`): `approximation` selects the tanh form.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GeluParams {
    approximation: u8,
}

/// `ns_Softmax::Params` (`softmax_fwd_f32`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SoftmaxParams {
    dim: i32,
}

/// `ns_CastKernel::Params` (`cast_bf16_to_f32`, `cast_f32_to_bf16`); 0 = round half to even.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CastParams {
    round_mode: u32,
}

const _: () = {
    assert!(std::mem::size_of::<LayerNormParams>() == 8);
    assert!(std::mem::offset_of!(LayerNormParams, eps) == 4);
    assert!(std::mem::size_of::<GeluParams>() == 1);
    assert!(std::mem::size_of::<SoftmaxParams>() == 4);
    assert!(std::mem::size_of::<CastParams>() == 4);
    assert!(cfg!(target_endian = "little"));
};

const LAYER_NORM_EPS: f32 = 1e-6;

/// Weight kinds of one block, as `vb_<kind>` in the generic recipe and `V<l>_<kind>` on the card.
pub const BLOCK_WEIGHTS: &[&str] = &[
    "n1_w", "n1_b", "q_W", "q_b", "k_W", "k_b", "v_W", "v_b", "qr_W", "qr_b", "kr_W", "kr_b",
    "o_W", "o_b", "n2_w", "n2_b", "fc1_W", "fc1_b", "fc2_W", "fc2_b",
];

/// Tensors bound per launch to the block ping-pong buffers or the embedding buffer.
const CHAIN_TENSORS: &[&str] = &["v_x0", "vb_x_in", "vb_x_out", "vm_x_in", "v_merged"];

/// Rows of the request RoPE table kept valid past `prompt + max_new` (speculative drafts read a
/// few slots ahead of the accepted cursor).
pub const ROPE_ROW_MARGIN: usize = 64;

/// Host pinned staging buffer for per-image inputs and RoPE rows.
const VISION_STAGE_BYTES: usize = 8 * 1024 * 1024;

fn generic_block_weight(name: &str) -> Option<&str> {
    let kind = name.strip_prefix("vb_")?;
    BLOCK_WEIGHTS.contains(&kind).then_some(kind)
}

/// Graph construction helper for the tower's recipes.
struct Vb<'a> {
    gb: &'a mut GraphBuilder,
    vs: &'a VisionSpec,
    n: u64,
}

impl Vb<'_> {
    fn act(&mut self, name: impl Into<String>, dt: SynDataType, dims: &[u64]) -> TensorId {
        self.gb.add_persistent_tensor(name, dt, dims)
    }

    /// `y {nout, cols} = W {k, nout}^T . x {k, cols} + b {nout, 1}`.
    fn lin(&mut self, x: TensorId, w: &str, k: u64, nout: u64, cols: u64, y: &str) -> TensorId {
        let wt =
            self.gb
                .get_or_add_persistent_tensor(format!("{w}_W"), SynDataType::Bf16, &[k, nout]);
        let b =
            self.gb
                .get_or_add_persistent_tensor(format!("{w}_b"), SynDataType::Bf16, &[nout, 1]);
        let g = self.act(format!("{y}_mm"), SynDataType::Bf16, &[nout, cols]);
        let o = self.act(y, SynDataType::Bf16, &[nout, cols]);
        self.gb.op_gemm(format!("{y}_gemm"), x, wt, g, false, true);
        self.gb.op_add_bf16(format!("{y}_bias"), g, b, o);
        o
    }

    /// LayerNorm over the feature dimension (bias and weight in F32). The kernel takes 4-D
    /// `{C, cols, 1, 1}`, hence the reshape views around it.
    fn ln(&mut self, x: TensorId, w: &str, c: u64, cols: u64, y: &str) -> TensorId {
        let b = self
            .gb
            .get_or_add_persistent_tensor(format!("{w}_b"), SynDataType::Float, &[c]);
        let g = self
            .gb
            .get_or_add_persistent_tensor(format!("{w}_w"), SynDataType::Float, &[c]);
        let x4 = self
            .gb
            .add_intermediate(format!("{y}_x4"), SynDataType::Bf16, &[c, cols, 1, 1]);
        let y4 = self
            .gb
            .add_intermediate(format!("{y}_y4"), SynDataType::Bf16, &[c, cols, 1, 1]);
        let mean =
            self.gb
                .add_intermediate(format!("{y}_mean"), SynDataType::Float, &[1, cols, 1, 1]);
        let rstd =
            self.gb
                .add_intermediate(format!("{y}_rstd"), SynDataType::Float, &[1, cols, 1, 1]);
        let o = self.act(y, SynDataType::Bf16, &[c, cols]);
        self.gb.op_reshape(format!("{y}_r4"), x, x4);
        let p = LayerNormParams {
            eps_valid: 1,
            _pad: [0; 3],
            eps: LAYER_NORM_EPS,
        };
        self.gb.add_node(
            "layer_norm_fwd_bf16",
            format!("{y}_ln"),
            &[x4, b, g],
            &[y4, mean, rstd],
            as_param_bytes(&p),
        );
        self.gb.op_reshape(format!("{y}_r2"), y4, o);
        o
    }

    fn cast(&mut self, x: TensorId, to: SynDataType, dims: &[u64], y: &str) -> TensorId {
        let o = self.act(y, to, dims);
        let guid = if to == SynDataType::Float {
            "cast_bf16_to_f32"
        } else {
            "cast_f32_to_bf16"
        };
        let p = CastParams { round_mode: 0 };
        self.gb
            .add_node(guid, format!("{y}_cast"), &[x], &[o], as_param_bytes(&p));
        o
    }

    /// GELU evaluated in F32 (cast, `gelu_fwd_f32`, cast back): the BF16 kernel's tanh form
    /// costs about a percent of accuracy per block, the F32 one a few percent of tower time.
    /// `VisionSpec::gelu_bf16` selects the BF16 kernel instead.
    fn gelu(&mut self, x: TensorId, tanh_approx: bool, c: u64, cols: u64, y: &str) -> TensorId {
        if self.vs.gelu_bf16 {
            let o = self.act(y, SynDataType::Bf16, &[c, cols]);
            let aux = self.act(format!("{y}_aux"), SynDataType::Bf16, &[c, cols]);
            let p = GeluParams {
                approximation: u8::from(tanh_approx),
            };
            self.gb.add_node(
                "gelu_fwd_bf16",
                format!("{y}_gelu"),
                &[x],
                &[o, aux],
                as_param_bytes(&p),
            );
            return o;
        }
        let xf = self.cast(x, SynDataType::Float, &[c, cols], &format!("{y}_xf"));
        let of = self.act(format!("{y}_of"), SynDataType::Float, &[c, cols]);
        let aux = self.act(format!("{y}_aux"), SynDataType::Float, &[c, cols]);
        let p = GeluParams {
            approximation: u8::from(tanh_approx),
        };
        self.gb.add_node(
            "gelu_fwd_f32",
            format!("{y}_gelu"),
            &[xf],
            &[of, aux],
            as_param_bytes(&p),
        );
        self.cast(of, SynDataType::Bf16, &[c, cols], y)
    }

    fn add(
        &mut self,
        a: TensorId,
        b: TensorId,
        dt: SynDataType,
        dims: &[u64],
        y: &str,
    ) -> TensorId {
        let o = self.act(y, dt, dims);
        let guid = if dt == SynDataType::Bf16 {
            "add_fwd_bf16"
        } else {
            "add_fwd_f32"
        };
        self.gb
            .add_node(guid, format!("{y}_add"), &[a, b], &[o], &[]);
        o
    }

    fn swap_heads_params() -> TransposeParams {
        TransposeParams {
            permutation: [0, 2, 1, 0, 0],
            tensor_dim: 3,
        }
    }

    /// `{H, N}` -> `{dh, nh, N}` -> `{dh, N, nh}`: heads become the batch dimension.
    fn heads(&mut self, x: TensorId, y: &str) -> TensorId {
        let (dh, nh, n) = (self.vs.head_dim(), self.vs.n_heads, self.n);
        let r = self.act(format!("{y}_r3"), SynDataType::Bf16, &[dh, nh, n]);
        self.gb.op_reshape(format!("{y}_reshape"), x, r);
        let t = self.act(format!("{y}_h"), SynDataType::Bf16, &[dh, n, nh]);
        let p = Self::swap_heads_params();
        self.gb.add_node(
            "transpose",
            format!("{y}_transpose"),
            &[r],
            &[t],
            as_param_bytes(&p),
        );
        t
    }

    /// `{dh, N, nh}` -> `{dh, nh, N}` -> `{H, N}`.
    fn unheads(&mut self, t: TensorId, y: &str) -> TensorId {
        let (dh, nh, n, h) = (self.vs.head_dim(), self.vs.n_heads, self.n, self.vs.hidden);
        let r = self.act(format!("{y}_t3"), SynDataType::Bf16, &[dh, nh, n]);
        let p = Self::swap_heads_params();
        self.gb.add_node(
            "transpose",
            format!("{y}_transpose"),
            &[t],
            &[r],
            as_param_bytes(&p),
        );
        let o = self.act(y, SynDataType::Bf16, &[h, n]);
        self.gb.op_reshape(format!("{y}_reshape"), r, o);
        o
    }

    /// Rotary product in F32 on `{H, N}`: `e = x * cos + rotate_half(x) * sin`, where
    /// `rotate_half(x)` comes from the folded weight rows; then BF16 and heads to the batch dim.
    fn rope_heads(
        &mut self,
        x: TensorId,
        xr: TensorId,
        cs: TensorId,
        sn: TensorId,
        y: &str,
    ) -> TensorId {
        let dims = [self.vs.hidden, self.n];
        let xf = self.cast(x, SynDataType::Float, &dims, &format!("{y}_f"));
        let xrf = self.cast(xr, SynDataType::Float, &dims, &format!("{y}_rf"));
        let a = self.act(format!("{y}_xc"), SynDataType::Float, &dims);
        let b = self.act(format!("{y}_xs"), SynDataType::Float, &dims);
        self.gb
            .add_node("mult_fwd_f32", format!("{y}_mc"), &[xf, cs], &[a], &[]);
        self.gb
            .add_node("mult_fwd_f32", format!("{y}_ms"), &[xrf, sn], &[b], &[]);
        let e = self.add(a, b, SynDataType::Float, &dims, &format!("{y}_e"));
        let eb = self.cast(e, SynDataType::Bf16, &dims, &format!("{y}_eb"));
        self.heads(eb, y)
    }
}

/// `g_vit_patch_N{n}`: patch rows -> patch embeddings + position rows.
pub fn build_vit_patch_graph(vs: &VisionSpec, n: u64) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_vit_patch_N{n}"));
    let (h, row) = (vs.hidden, vs.patch_row());
    let mut b = Vb { gb: &mut gb, vs, n };
    let pix = b.act("v_pix", SynDataType::Bf16, &[row, n]);
    let pos = b.act("v_pos", SynDataType::Bf16, &[h, n]);
    let x = b.lin(pix, "V_patch", row, h, n, "v_pe");
    b.add(x, pos, SynDataType::Bf16, &[h, n], "v_x0");
    gb
}

/// `g_vit_block_N{n}`: one generic pre-norm transformer block with 2-D rotary attention.
pub fn build_vit_block_graph(vs: &VisionSpec, n: u64) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_vit_block_N{n}"));
    let (h, inter, nh, dh) = (vs.hidden, vs.intermediate, vs.n_heads, vs.head_dim());
    let mut b = Vb { gb: &mut gb, vs, n };
    let x = b.act("vb_x_in", SynDataType::Bf16, &[h, n]);
    let cq = b.act("v_cq", SynDataType::Float, &[h, n]);
    let sq = b.act("v_sq", SynDataType::Float, &[h, n]);
    let ck = b.act("v_ck", SynDataType::Float, &[h, n]);
    let sk = b.act("v_sk", SynDataType::Float, &[h, n]);

    let h1 = b.ln(x, "vb_n1", h, n, "vb_h1");
    let q = b.lin(h1, "vb_q", h, h, n, "vb_q");
    let k = b.lin(h1, "vb_k", h, h, n, "vb_k");
    let v = b.lin(h1, "vb_v", h, h, n, "vb_v");
    let qr = b.lin(h1, "vb_qr", h, h, n, "vb_qr");
    let kr = b.lin(h1, "vb_kr", h, h, n, "vb_kr");
    let qh = b.rope_heads(q, qr, cq, sq, "vb_qe");
    let kh = b.rope_heads(k, kr, ck, sk, "vb_ke");
    let vh = b.heads(v, "vb_vh");

    let scores = b.act("vb_scores", SynDataType::Float, &[n, n, nh]);
    b.gb.op_batch_gemm("vb_qk", qh, kh, scores, false, true);
    let pf = b.act("vb_pf", SynDataType::Float, &[n, n, nh]);
    let sp = SoftmaxParams { dim: 0 };
    b.gb.add_node(
        "softmax_fwd_f32",
        "vb_softmax",
        &[scores],
        &[pf],
        as_param_bytes(&sp),
    );
    let pb = b.cast(pf, SynDataType::Bf16, &[n, n, nh], "vb_pb");
    let ao = b.act("vb_ao", SynDataType::Bf16, &[dh, n, nh]);
    b.gb.op_batch_gemm("vb_pv", pb, vh, ao, false, false);
    let a = b.unheads(ao, "vb_attn");
    let o = b.lin(a, "vb_o", h, h, n, "vb_o");
    let x2 = b.add(x, o, SynDataType::Bf16, &[h, n], "vb_x2");

    let h2 = b.ln(x2, "vb_n2", h, n, "vb_h2");
    let f1 = b.lin(h2, "vb_fc1", h, inter, n, "vb_f1");
    let g = b.gelu(f1, vs.gelu_tanh, inter, n, "vb_g");
    let f2 = b.lin(g, "vb_fc2", inter, h, n, "vb_f2");
    b.add(x2, f2, SynDataType::Bf16, &[h, n], "vb_x_out");
    gb
}

/// `g_vit_merge_N{n}`: LayerNorm, 2x2 merge view, MLP to the decoder width.
pub fn build_vit_merge_graph(vs: &VisionSpec, n: u64) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_vit_merge_N{n}"));
    let (h, mw, cols) = (vs.hidden, vs.merged_width(), n / vs.patches_per_token());
    let mut b = Vb { gb: &mut gb, vs, n };
    let x = b.act("vm_x_in", SynDataType::Bf16, &[h, n]);
    let hn = b.ln(x, "V_m_n", h, n, "vm_h");
    let r = b.act("vm_r", SynDataType::Bf16, &[mw, cols]);
    b.gb.op_reshape("vm_reshape", hn, r);
    let f1 = b.lin(r, "V_m_fc1", mw, mw, cols, "vm_f1");
    let g = b.gelu(f1, false, mw, cols, "vm_g");
    b.lin(g, "V_m_fc2", mw, vs.out_hidden, cols, "v_merged");
    gb
}

/// What the tower holds on the card, derived from the graph descriptions alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VisionMemoryPlan {
    /// Device weight name -> bytes (a block weight once per block).
    pub weights: BTreeMap<String, u64>,
    /// Activation / input buffer name -> bytes at the largest patch count.
    pub acts: BTreeMap<String, u64>,
    /// One block ping-pong buffer `{H, max_patches}` BF16 (two are allocated).
    pub xbuf_bytes: u64,
    /// Embedding buffer `{HID, max_tokens}` BF16.
    pub emb_bytes: u64,
}

fn align256(n: u64) -> u64 {
    align_up(n.max(256), 256)
}

impl VisionMemoryPlan {
    pub fn new(vs: &VisionSpec, max_patches: u64, max_tokens: u64) -> Self {
        let mut plan = Self {
            xbuf_bytes: vs.hidden * max_patches * 2,
            emb_bytes: vs.out_hidden * max_tokens * 2,
            ..Self::default()
        };
        for gb in [
            build_vit_patch_graph(vs, max_patches),
            build_vit_block_graph(vs, max_patches),
            build_vit_merge_graph(vs, max_patches),
        ] {
            for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
                if CHAIN_TENSORS.contains(&name.as_str()) {
                    continue;
                }
                if let Some(kind) = generic_block_weight(&name) {
                    for l in 0..vs.depth {
                        plan.weights.insert(format!("V{l}_{kind}"), bytes);
                    }
                } else if name.starts_with("V_") {
                    plan.weights.insert(name, bytes);
                } else {
                    let e = plan.acts.entry(name).or_insert(0);
                    *e = (*e).max(bytes);
                }
            }
        }
        plan
    }

    /// Bytes taken from the weights arena (256-byte aligned sub-allocations).
    pub fn weights_bytes(&self) -> u64 {
        self.weights.values().map(|&b| align256(b)).sum()
    }

    /// Bytes taken from the activation arena.
    pub fn act_bytes(&self) -> u64 {
        self.acts.values().map(|&b| align256(b)).sum::<u64>()
            + 2 * align256(self.xbuf_bytes)
            + align256(self.emb_bytes)
    }
}

struct RecipeSet {
    patch: CompiledRecipe,
    block: CompiledRecipe,
    merge: CompiledRecipe,
    patch_launch: Option<PreparedLaunch>,
    block_launches: Vec<PreparedLaunch>,
}

/// The vision tower of one engine: plan, device buffers, recipes per patch count, and the
/// request-scoped RoPE row bookkeeping of the decoder's slot-indexed table.
pub struct VisionTower {
    vs: VisionSpec,
    max_patches: u64,
    max_tokens: u64,
    plan: VisionMemoryPlan,
    precompile: Vec<u64>,

    // recipes and pinned memory (dropped before the device by the owning engine's field order)
    sets: HashMap<u64, RecipeSet>,
    stage: Option<PinnedBuf>,

    weights_addr: BTreeMap<String, u64>,
    act_addr: BTreeMap<String, u64>,
    xbuf: [u64; 2],
    emb_addr: u64,

    pos_table: Vec<u16>,
    /// Grid whose position rows and rotary tables are in the device input buffers.
    tables_grid: Option<(usize, usize)>,
    /// Launches of the previous image may still read the shared input buffers.
    compute_pending: bool,
    /// Decoder RoPE rows `[lo, hi)` currently holding a vision request's positions.
    rope_dirty: Option<(usize, usize)>,
    compile_ms: f64,
}

impl VisionTower {
    /// Plans the tower for images of up to `max_patches` patches and requests of up to
    /// `max_tokens` image tokens. Nothing is compiled or allocated here.
    pub fn plan(
        vs: VisionSpec,
        decoder_hidden: u64,
        max_patches: u64,
        max_tokens: u64,
        precompile: &[u64],
    ) -> Result<Self, String> {
        vs.validate()?;
        if vs.out_hidden != decoder_hidden {
            return Err(format!(
                "vision out_hidden_size {} != decoder hidden size {decoder_hidden}",
                vs.out_hidden
            ));
        }
        let ppt = vs.patches_per_token();
        if max_patches < ppt || !max_patches.is_multiple_of(ppt) {
            return Err(format!(
                "vision max patches {max_patches} must be a positive multiple of {ppt}"
            ));
        }
        let max_tokens = max_tokens.max(max_patches / ppt);
        let mut pre: Vec<u64> = precompile.to_vec();
        pre.sort_unstable();
        pre.dedup();
        for &n in &pre {
            if n == 0 || !n.is_multiple_of(ppt) || n > max_patches {
                return Err(format!(
                    "image of {n} patches does not fit the planned vision buffers ({max_patches} patches)"
                ));
            }
        }
        let plan = VisionMemoryPlan::new(&vs, max_patches, max_tokens);
        Ok(Self {
            vs,
            max_patches,
            max_tokens,
            plan,
            precompile: pre,
            sets: HashMap::new(),
            stage: None,
            weights_addr: BTreeMap::new(),
            act_addr: BTreeMap::new(),
            xbuf: [0; 2],
            emb_addr: 0,
            pos_table: Vec::new(),
            tables_grid: None,
            compute_pending: false,
            rope_dirty: None,
            compile_ms: 0.0,
        })
    }

    pub fn spec(&self) -> &VisionSpec {
        &self.vs
    }

    pub fn max_patches(&self) -> u64 {
        self.max_patches
    }

    pub fn max_tokens(&self) -> u64 {
        self.max_tokens
    }

    pub fn memory_plan(&self) -> &VisionMemoryPlan {
        &self.plan
    }

    /// HBM address of the request's image embeddings `{HID, max_tokens}` BF16, rows in prompt
    /// order (`ImageSpan::emb_row`). Drafters read image-slot embeddings from here.
    pub fn embeddings_addr(&self) -> u64 {
        self.emb_addr
    }

    fn compile_set(
        vs: &VisionSpec,
        n: u64,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
    ) -> Result<RecipeSet, String> {
        let c = |gb: GraphBuilder| -> Result<CompiledRecipe, String> {
            gb.compile(Arc::clone(api), cache, driver_version)
                .map_err(|e| format!("compile vision recipe for {n} patches: {e}"))
        };
        Ok(RecipeSet {
            patch: c(build_vit_patch_graph(vs, n))?,
            block: c(build_vit_block_graph(vs, n))?,
            merge: c(build_vit_merge_graph(vs, n))?,
            patch_launch: None,
            block_launches: Vec::new(),
        })
    }

    /// Compiles (or loads) the recipes of the patch counts known at start-up, before the HBM
    /// plan is made. Returns the largest workspace they need.
    pub fn precompile(
        &mut self,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
        verbose: bool,
    ) -> Result<u64, String> {
        let mut max_ws = 0u64;
        for n in self.precompile.clone() {
            let t0 = Instant::now();
            let set = Self::compile_set(&self.vs, n, api, cache, driver_version)?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            self.compile_ms += ms;
            for r in [&set.patch, &set.block, &set.merge] {
                max_ws = max_ws.max(r.ws_size);
                if verbose {
                    eprintln!(
                        "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                        r.name,
                        r.from_cache,
                        r.ws_size as f64 / (1024.0 * 1024.0)
                    );
                }
            }
            self.sets.insert(n, set);
        }
        Ok(max_ws)
    }

    /// Sub-allocates every weight and buffer of the plan from the engine's arenas (pre-seal).
    pub fn allocate(
        &mut self,
        weights_arena: &mut HbmArena,
        act_arena: &mut HbmArena,
    ) -> Result<(), String> {
        let sub = |arena: &mut HbmArena, bytes: u64| -> Result<u64, String> {
            arena
                .suballoc(bytes.max(256), 256)
                .map(|(addr, _)| addr)
                .map_err(|e| format!("vision arena suballoc: {e}"))
        };
        for (name, &bytes) in &self.plan.weights {
            self.weights_addr
                .insert(name.clone(), sub(weights_arena, bytes)?);
        }
        for (name, &bytes) in &self.plan.acts {
            self.act_addr.insert(name.clone(), sub(act_arena, bytes)?);
        }
        self.xbuf = [
            sub(act_arena, self.plan.xbuf_bytes)?,
            sub(act_arena, self.plan.xbuf_bytes)?,
        ];
        self.emb_addr = sub(act_arena, self.plan.emb_bytes)?;
        Ok(())
    }

    fn addr_of(&self, name: &str) -> Result<u64, String> {
        self.weights_addr
            .get(name)
            .or_else(|| self.act_addr.get(name))
            .copied()
            .ok_or_else(|| format!("vision tensor '{name}' has no planned HBM buffer"))
    }

    /// Binds the shared workspace and pre-validates the fixed launches of one recipe set.
    fn finish_set(&self, dev: &HpuDevice, set: &mut RecipeSet) -> Result<(), String> {
        for r in [&mut set.patch, &mut set.block, &mut set.merge] {
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind vision recipe workspace: {e}"))?;
        }
        let mut pb = BTreeMap::new();
        for t in set.patch.tensors() {
            let addr = if t.name == "v_x0" {
                self.xbuf[0]
            } else {
                self.addr_of(&t.name)?
            };
            pb.insert(t.name.clone(), addr);
        }
        set.patch_launch = Some(
            set.patch
                .prepare_launch(&pb)
                .map_err(|e| format!("prepare vision patch launch: {e}"))?,
        );
        set.block_launches.clear();
        for l in 0..self.vs.depth as usize {
            let mut bb = BTreeMap::new();
            for t in set.block.tensors() {
                let addr = match t.name.as_str() {
                    "vb_x_in" => self.xbuf[l & 1],
                    "vb_x_out" => self.xbuf[(l + 1) & 1],
                    name => match generic_block_weight(name) {
                        Some(kind) => self.addr_of(&format!("V{l}_{kind}"))?,
                        None => self.addr_of(name)?,
                    },
                };
                bb.insert(t.name.clone(), addr);
            }
            set.block_launches.push(
                set.block
                    .prepare_launch(&bb)
                    .map_err(|e| format!("prepare vision block {l} launch: {e}"))?,
            );
        }
        Ok(())
    }

    /// After the engine sealed its allocations: workspaces and launch descriptors of the
    /// precompiled sets, and the pinned staging buffer.
    pub fn finish_load(&mut self, dev: &HpuDevice) -> Result<(), String> {
        let mut sets = std::mem::take(&mut self.sets);
        let res = sets
            .values_mut()
            .try_for_each(|set| self.finish_set(dev, set));
        self.sets = sets;
        res?;
        self.stage = Some(
            dev.alloc_pinned(VISION_STAGE_BYTES)
                .map_err(|e| format!("alloc pinned vision stage: {e}"))?,
        );
        Ok(())
    }

    /// Uploads every tower weight and keeps a host copy of the position table.
    pub fn upload_weights(
        &mut self,
        dev: &HpuDevice,
        store: &SafetensorsStore,
        stage: &mut PinnedBuf,
    ) -> Result<u64, String> {
        let packer = VisionPacker::new(store, &self.vs);
        let mut total = 0u64;
        for (name, &addr) in &self.weights_addr {
            let data = packer.pack(name)?;
            let expected = self.plan.weights[name];
            if data.len() as u64 != expected {
                return Err(format!(
                    "vision weight '{name}' packed size {} != graph tensor size {expected}",
                    data.len()
                ));
            }
            dev.upload_h2d_chunked(&data, addr, stage)
                .map_err(|e| format!("upload vision weight '{name}': {e}"))?;
            total += expected;
        }
        self.pos_table = packer.pos_table()?;
        Ok(total)
    }

    /// Recipes for images of `n` patches: compiled on first use into the shared workspace.
    pub fn ensure(
        &mut self,
        dev: &HpuDevice,
        cache: &RecipeCache,
        driver_version: &str,
        n: u64,
    ) -> Result<(), String> {
        if self.sets.contains_key(&n) {
            return Ok(());
        }
        let ppt = self.vs.patches_per_token();
        if n == 0 || !n.is_multiple_of(ppt) {
            return Err(format!("image of {n} patches is not a multiple of {ppt}"));
        }
        if n > self.max_patches {
            return Err(format!(
                "image of {n} patches exceeds the {} patches the vision buffers were planned for (--vision-max-patches)",
                self.max_patches
            ));
        }
        let t0 = Instant::now();
        let mut set = Self::compile_set(&self.vs, n, dev.api(), cache, driver_version)?;
        self.finish_set(dev, &mut set)?;
        self.compile_ms += t0.elapsed().as_secs_f64() * 1000.0;
        self.sets.insert(n, set);
        Ok(())
    }

    fn upload<T: Copy>(
        dev: &HpuDevice,
        stage: &mut PinnedBuf,
        data: &[T],
        addr: u64,
        what: &str,
    ) -> Result<(), String> {
        // SAFETY: `T` is a plain numeric type (u16 / f32) without padding; the host is little
        // endian (asserted at compile time), which is the device byte order.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        dev.upload_h2d_chunked(bytes, addr, stage)
            .map_err(|e| format!("upload vision {what}: {e}"))
    }

    /// Enqueues one image on the compute stream: uploads its inputs, then patch embedding,
    /// `depth` block launches and the merger, whose output lands at row `emb_row` of the
    /// embedding buffer. The caller synchronizes the compute stream after the last image
    /// (`sync`). Returns the host preparation + upload time in ms.
    pub fn encode_image(
        &mut self,
        dev: &HpuDevice,
        cache: &RecipeCache,
        driver_version: &str,
        img: &PreparedImage,
        emb_row: usize,
    ) -> Result<f64, String> {
        let n = img.n_patches() as u64;
        if img.row_len as u64 != self.vs.patch_row()
            || img.pixels.len() != img.n_patches() * img.row_len
        {
            return Err("prepared image does not match this model's patch layout".into());
        }
        if (emb_row + img.n_tokens()) as u64 > self.max_tokens {
            return Err(format!(
                "request needs {} image tokens, the vision embedding buffer holds {}",
                emb_row + img.n_tokens(),
                self.max_tokens
            ));
        }
        self.ensure(dev, cache, driver_version, n)?;
        // the previous image's launches read the shared input buffers
        self.sync(dev)?;

        let t0 = Instant::now();
        let mut stage = self.stage.take().ok_or("vision tower is not loaded")?;
        let grid = (img.grid_h, img.grid_w);
        // position rows and rotary tables depend on the grid only: an image with the grid of
        // the previous one reuses what is already in the device buffers
        let need_tables = self.tables_grid != Some(grid);
        let a_pix = self.addr_of("v_pix")?;
        let a_pos = self.addr_of("v_pos")?;
        let a_rot = [
            self.addr_of("v_cq")?,
            self.addr_of("v_sq")?,
            self.addr_of("v_ck")?,
            self.addr_of("v_sk")?,
        ];
        let (vs, pos_table) = (&self.vs, &self.pos_table);
        let up = (|| -> Result<(), String> {
            Self::upload(dev, &mut stage, &img.pixels, a_pix, "pixels")?;
            if need_tables {
                let pos = pos_embed(vs, pos_table, grid.0, grid.1);
                Self::upload(dev, &mut stage, &pos, a_pos, "position rows")?;
                let t = vit_rope_tables(vs, grid.0, grid.1);
                Self::upload(dev, &mut stage, &t.cos_q, a_rot[0], "rotary table")?;
                Self::upload(dev, &mut stage, &t.sin_q, a_rot[1], "rotary table")?;
                Self::upload(dev, &mut stage, &t.cos_k, a_rot[2], "rotary table")?;
                Self::upload(dev, &mut stage, &t.sin_k, a_rot[3], "rotary table")?;
            }
            Ok(())
        })();
        self.stage = Some(stage);
        if need_tables {
            self.tables_grid = up.is_ok().then_some(grid);
        }
        up?;
        let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let depth = self.vs.depth as usize;
        let out = self.emb_addr + (emb_row as u64) * self.vs.out_hidden * 2;
        let set = &self.sets[&n];
        let mut mb = BTreeMap::new();
        for t in set.merge.tensors() {
            let addr = match t.name.as_str() {
                "vm_x_in" => self.xbuf[depth & 1],
                "v_merged" => out,
                name => self.addr_of(name)?,
            };
            mb.insert(t.name.clone(), addr);
        }
        let merge_launch = set
            .merge
            .prepare_launch(&mb)
            .map_err(|e| format!("prepare vision merge launch: {e}"))?;

        self.compute_pending = true;
        let patch_launch = set
            .patch_launch
            .as_ref()
            .ok_or("vision recipes not bound")?;
        set.patch
            .launch_prepared(dev, patch_launch)
            .map_err(|e| format!("launch vision patch embedding: {e}"))?;
        for (l, bl) in set.block_launches.iter().enumerate() {
            set.block
                .launch_prepared(dev, bl)
                .map_err(|e| format!("launch vision block {l}: {e}"))?;
        }
        set.merge
            .launch_prepared(dev, &merge_launch)
            .map_err(|e| format!("launch vision merger: {e}"))?;
        Ok(prep_ms)
    }

    /// Waits for every queued tower launch.
    pub fn sync(&mut self, dev: &HpuDevice) -> Result<(), String> {
        if self.compute_pending {
            // SAFETY: the compute stream belongs to `dev` and lives as long as it.
            unsafe { dev.sync_stream(dev.compute_stream()) }
                .map_err(|e| format!("sync vision tower: {e}"))?;
            self.compute_pending = false;
        }
        Ok(())
    }

    /// Rewrites the decoder's slot-indexed RoPE rows for the next request and restores rows a
    /// previous vision request changed. `layout` is `None` for a text-only request (a no-op
    /// unless rows are still dirty). Rows `[first image slot, row_hi)` take the layout's M-RoPE
    /// positions. Returns the number of rows uploaded.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_request_rope(
        &mut self,
        dev: &HpuDevice,
        rope_c_addr: u64,
        rope_s_addr: u64,
        theta: f64,
        rotary_dim: usize,
        layout: Option<&PromptLayout>,
        row_hi: usize,
    ) -> Result<u64, String> {
        let new = layout
            .and_then(|l| l.first_image_slot())
            .filter(|&lo| lo < row_hi)
            .map(|lo| (lo, row_hi));
        let (lo, hi) = match (new, self.rope_dirty) {
            (None, None) => return Ok(0),
            (Some(n), None) => n,
            (None, Some(d)) => d,
            (Some(n), Some(d)) => (n.0.min(d.0), n.1.max(d.1)),
        };
        let mut stage = self.stage.take().ok_or("vision tower is not loaded")?;
        let rows_per_chunk = (stage.len() / (128 * 4)).max(1);
        let sections = self.vs.mrope_section;
        let res = (|| -> Result<(), String> {
            let mut a = lo;
            while a < hi {
                let b = (a + rows_per_chunk).min(hi);
                let (c, s) = mrope_rows(theta, rotary_dim, sections, a, b, |slot, sec| {
                    match (new, layout) {
                        (Some((nlo, nhi)), Some(l)) if slot >= nlo && slot < nhi => {
                            l.position(slot, sec)
                        }
                        _ => slot as i64,
                    }
                });
                let off = (a as u64) * 128 * 4;
                dev.upload_h2d_chunked(&c, rope_c_addr + off, &mut stage)
                    .map_err(|e| format!("upload RoPE cos rows: {e}"))?;
                dev.upload_h2d_chunked(&s, rope_s_addr + off, &mut stage)
                    .map_err(|e| format!("upload RoPE sin rows: {e}"))?;
                a = b;
            }
            Ok(())
        })();
        self.stage = Some(stage);
        // rows outside the new range now hold text positions again; on failure everything
        // touched stays marked dirty so the next request rewrites it
        self.rope_dirty = if res.is_ok() { new } else { Some((lo, hi)) };
        res?;
        Ok((hi - lo) as u64)
    }

    pub fn compile_ms(&self) -> f64 {
        self.compile_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::test_util::qwen38_vision_spec;

    fn bytes_of(gb: &GraphBuilder, name: &str) -> u64 {
        gb.persistent_tensor_specs()
            .into_iter()
            .find(|t| t.0 == name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
            .3
    }

    #[test]
    fn tower_graphs_have_the_expected_io_shapes() {
        let vs = qwen38_vision_spec();
        let n = 256u64;
        let patch = build_vit_patch_graph(&vs, n);
        assert_eq!(bytes_of(&patch, "v_pix"), 1536 * n * 2);
        assert_eq!(bytes_of(&patch, "V_patch_W"), 1536 * 1152 * 2);
        assert_eq!(bytes_of(&patch, "v_x0"), 1152 * n * 2);

        let block = build_vit_block_graph(&vs, n);
        assert_eq!(bytes_of(&block, "vb_scores"), n * n * 16 * 4);
        assert_eq!(bytes_of(&block, "vb_q_W"), 1152 * 1152 * 2);
        assert_eq!(bytes_of(&block, "vb_fc1_W"), 1152 * 4304 * 2);
        assert_eq!(bytes_of(&block, "vb_n1_w"), 1152 * 4);
        assert_eq!(bytes_of(&block, "v_cq"), 1152 * n * 4);
        // every generic weight kind appears exactly once
        let specs = block.persistent_tensor_specs();
        for kind in BLOCK_WEIGHTS {
            let name = format!("vb_{kind}");
            assert_eq!(specs.iter().filter(|t| t.0 == name).count(), 1, "{name}");
        }
        // no activation is mistaken for a weight
        for t in &specs {
            if generic_block_weight(&t.0).is_none() {
                assert!(!t.0.starts_with("V_"), "{}", t.0);
            }
        }

        let merge = build_vit_merge_graph(&vs, n);
        assert_eq!(bytes_of(&merge, "v_merged"), 5120 * (n / 4) * 2);
        assert_eq!(bytes_of(&merge, "V_m_fc2_W"), 4608 * 5120 * 2);
        assert_eq!(bytes_of(&merge, "vm_r"), 4608 * (n / 4) * 2);

        // the BF16 GELU variant is a different graph (and recipe)
        let mut vb = vs.clone();
        vb.gelu_bf16 = true;
        assert_ne!(
            block.fingerprint("1.24.2"),
            build_vit_block_graph(&vb, n).fingerprint("1.24.2")
        );
        assert_eq!(
            bytes_of(&build_vit_block_graph(&vb, n), "vb_g_aux"),
            4304 * n * 2
        );

        // fingerprints are deterministic and depend on the patch count
        assert_eq!(
            block.fingerprint("1.24.2"),
            build_vit_block_graph(&vs, n).fingerprint("1.24.2")
        );
        assert_ne!(
            block.fingerprint("1.24.2"),
            build_vit_block_graph(&vs, 560).fingerprint("1.24.2")
        );
    }

    #[test]
    fn memory_plan_counts_every_block_weight_and_scales_with_patches() {
        let vs = qwen38_vision_spec();
        let plan = VisionMemoryPlan::new(&vs, 1176, 294);
        // 20 weights per block x 27 blocks + patch (2) + merger (6)
        assert_eq!(plan.weights.len(), 20 * 27 + 2 + 6);
        let block_w: u64 = 5 * (1152 * 1152 * 2 + 1152 * 2)       // q, k, v, qr, kr
            + (1152 * 1152 * 2 + 1152 * 2)                         // o
            + (1152 * 4304 * 2 + 4304 * 2)                         // fc1
            + (4304 * 1152 * 2 + 1152 * 2)                         // fc2
            + 4 * 1152 * 4; // two LayerNorms, F32
        let other: u64 = 1536 * 1152 * 2 + 1152 * 2                // patch
            + 2 * 1152 * 4                                         // merger norm
            + 4608 * 4608 * 2 + 4608 * 2
            + 4608 * 5120 * 2 + 5120 * 2;
        let raw: u64 = plan.weights.values().sum();
        assert_eq!(raw, 27 * block_w + other);
        assert_eq!(plan.xbuf_bytes, 1152 * 1176 * 2);
        assert_eq!(plan.emb_bytes, 5120 * 294 * 2);
        assert_eq!(plan.acts["vb_scores"], 1176 * 1176 * 16 * 4);
        for chain in CHAIN_TENSORS {
            assert!(!plan.acts.contains_key(*chain));
        }
        // a smaller cap needs strictly less activation memory, the same weights
        let small = VisionMemoryPlan::new(&vs, 256, 64);
        assert_eq!(small.weights, plan.weights);
        assert!(small.act_bytes() < plan.act_bytes());
        // every buffer of a smaller image fits the buffers planned for the larger one
        for (name, bytes) in &small.acts {
            assert!(plan.acts[name] >= *bytes, "{name}");
        }
    }

    #[test]
    fn plan_refuses_bad_caps() {
        let vs = qwen38_vision_spec();
        assert!(VisionTower::plan(vs.clone(), 5120, 0, 0, &[]).is_err());
        assert!(VisionTower::plan(vs.clone(), 5120, 258, 0, &[]).is_err());
        assert!(VisionTower::plan(vs.clone(), 4096, 256, 0, &[]).is_err());
        assert!(VisionTower::plan(vs.clone(), 5120, 256, 0, &[560]).is_err());
        let t = VisionTower::plan(vs, 5120, 1176, 0, &[256, 1176, 256]).unwrap();
        assert_eq!(t.max_tokens(), 294);
        assert_eq!(t.precompile, vec![256, 1176]);
    }
}
