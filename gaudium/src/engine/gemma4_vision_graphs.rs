//! Graphs of the Gemma 4 image path: the tower's three recipes per patch count, the unified
//! embedder's one recipe of fixed rows, and the decoder's embedding stage for tiles with image
//! rows.
//!
//! Where the reference rounds, the graphs round: every norm is computed in f32 and rounded to
//! BF16 once; the rotary product and the residual sums are BF16 operations; scores, softmax and
//! the weighted sum of the values stay in f32 with one rounding of the output; the pooled mean
//! is rounded to BF16 before its scaling. A cast to BF16 that feeds another elementwise node is
//! followed by `gaudium_ga_copy` (the graph compiler would otherwise let the consumer read the
//! unrounded value). Two arrangements give the matrix engine the work:
//!
//! - `rotate_half` inside each axis half of a head is a signed permutation of its channels: one
//!   `{dh, dh}` GEMM with entries 0 / +1 / -1 on the normed rows, exact;
//! - the host hands the patches over in pooling order (`super::pooling_order`), so the sum of a
//!   block is one GEMM of the view `{pool^2 hid, N / pool^2}` with a 0 / 1 matrix.

use super::{G4vDims, G4vKind};
use crate::engine::gemma4::G4Dims;
use crate::synapse::ffi::SynDataType;
use crate::synapse::graph::{GraphBuilder, TensorId};
use crate::synapse::ops::{as_param_bytes, TransposeParams};

/// Kernel ELFs the image path's recipes may be handed (kept apart from the text recipes'
/// fingerprints, so a text-only load keys its recipes exactly as before).
pub const G4V_ELFS: &[&str] = &[
    "GAUDIUM_GARMSNORM_ELF",
    "GAUDIUM_GACLAMP_ELF",
    "GAUDIUM_GACOPY_ELF",
    "GAUDIUM_GAGELU_ELF",
    "GAUDIUM_GALNORM_ELF",
    "GAUDIUM_GASOFTRX_ELF",
];

/// `ns_Softmax::Params`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SoftmaxParams {
    dim: i32,
}

/// `ns_CastKernel::Params`; 0 = round half to even.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CastParams {
    round_mode: u32,
}

const _: () = {
    assert!(std::mem::size_of::<SoftmaxParams>() == 4);
    assert!(std::mem::size_of::<CastParams>() == 4);
};

/// LayerNorm epsilon of the unified embedder (the library's default: its config has no field).
pub const LN_EPS: f32 = 1e-5;

/// Weight kinds of one encoder layer: `vb_<kind>` in the generic recipe, `V<l>_<kind>` on the
/// card. Norm weights over the tower width are BF16 (the row-norm kernel converts them like the
/// row), the per-head q / k norm weights f32.
pub const LAYER_WEIGHTS: &[&str] = &[
    "in_w", "pa_w", "pfin_w", "pf_w", "qn_w", "kn_w", "q_W", "k_W", "v_W", "o_W", "gate_W", "up_W",
    "down_W",
];
/// The clamped projections of a small size's layer, and the four bounds of each.
pub const CLAMPED: &[&str] = &["q", "k", "v", "o", "gate", "up", "down"];
pub const BOUNDS: &[&str] = &["imin", "imax", "omin", "omax"];

/// Tensors bound per launch to the chain's buffers (never planned by name).
pub const CHAIN_TENSORS: &[&str] = &["v_x0", "vb_x_in", "vb_x_out", "vp_x_in", "v_feat"];

/// Whether a tensor of the image path's recipes is a weight (uploaded once at load).
pub fn is_weight(name: &str) -> bool {
    if name.starts_with("V_") || name.starts_with("U_") {
        return true;
    }
    match name.strip_prefix("vb_") {
        Some(k) => LAYER_WEIGHTS.contains(&k) || BOUNDS.iter().any(|b| k.ends_with(b)),
        None => false,
    }
}

/// Whether a device tensor of the engine's plan is an image-path weight (`V_*`, `V<l>_*`, `U_*`).
pub fn is_vision_weight_name(name: &str) -> bool {
    name.starts_with("V_")
        || name.starts_with("U_")
        || (name.starts_with('V') && name[1..].starts_with(|c: char| c.is_ascii_digit()))
}

struct Vb<'a> {
    gb: &'a mut GraphBuilder,
    d: &'a G4vDims,
    n: u64,
}

impl Vb<'_> {
    fn act(&mut self, name: impl Into<String>, dt: SynDataType, dims: &[u64]) -> TensorId {
        self.gb.add_persistent_tensor(name, dt, dims)
    }

    fn w(&mut self, name: impl Into<String>, dt: SynDataType, dims: &[u64]) -> TensorId {
        self.gb.get_or_add_persistent_tensor(name, dt, dims)
    }

    fn view(&mut self, x: TensorId, dims: &[u64], dt: SynDataType, y: &str) -> TensorId {
        let v = self.gb.add_intermediate(y, dt, dims);
        self.gb.op_reshape(format!("{y}_rs"), x, v);
        v
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

    /// f32 -> BF16 where the model rounds: the cast, then a copy the compiler cannot see through.
    fn round(&mut self, x: TensorId, dims: &[u64], y: &str) -> TensorId {
        let r = self.cast(x, SynDataType::Bf16, dims, &format!("{y}_r"));
        let o = self.act(y, SynDataType::Bf16, dims);
        self.gb
            .add_node("gaudium_ga_copy", format!("{y}_fix"), &[r], &[o], &[]);
        o
    }

    fn bin(&mut self, guid: &str, a: TensorId, b: TensorId, dt: SynDataType, dims: &[u64], y: &str) -> TensorId {
        let o = self.act(y, dt, dims);
        self.gb
            .add_node(guid, format!("{y}_{guid}"), &[a, b], &[o], &[]);
        o
    }

    /// `bf16(x rsqrt(mean(x^2) + eps) w)` over the `c` channels of each column (`c` a multiple of
    /// 128), one rounding.
    fn norm_w(&mut self, x: TensorId, w: &str, c: u64, cols: u64, y: &str) -> TensorId {
        let wt = self.w(w, SynDataType::Bf16, &[c]);
        let o = self.act(y, SynDataType::Bf16, &[c, cols]);
        self.gb.add_node(
            "gaudium_ga_rmsnorm",
            format!("{y}_norm"),
            &[x, wt],
            &[o],
            &self.d.eps.to_le_bytes(),
        );
        o
    }

    /// `min(max(x, lo), hi)` with the bounds `<w>_<lo>` / `<w>_<hi>` (BF16, 128 lanes).
    fn clamp(&mut self, x: TensorId, w: &str, lo: &str, hi: &str, dims: &[u64], y: &str) -> TensorId {
        let l = self.w(format!("{w}_{lo}"), SynDataType::Bf16, &[128]);
        let h = self.w(format!("{w}_{hi}"), SynDataType::Bf16, &[128]);
        let o = self.act(y, SynDataType::Bf16, dims);
        self.gb
            .add_node("gaudium_ga_clamp", format!("{y}_clamp"), &[x, l, h], &[o], &[]);
        o
    }

    /// `y {nout, cols} = W {k, nout}^T . x {k, cols}`, no bias. A tower with clamped linears
    /// bounds the input and the output of every encoder projection (`vb_*`) by its own scalars.
    fn proj(&mut self, x: TensorId, w: &str, k: u64, nout: u64, cols: u64, y: &str) -> TensorId {
        let clamped = self.d.clipped && w.starts_with("vb_");
        let xin = if clamped {
            self.clamp(x, w, "imin", "imax", &[k, cols], &format!("{y}_ci"))
        } else {
            x
        };
        let wt = self.w(format!("{w}_W"), SynDataType::Bf16, &[k, nout]);
        let raw = if clamped { format!("{y}_raw") } else { y.to_string() };
        let o = self.act(raw, SynDataType::Bf16, &[nout, cols]);
        self.gb.op_gemm(format!("{y}_gemm"), xin, wt, o, false, true);
        if clamped {
            self.clamp(o, w, "omin", "omax", &[nout, cols], y)
        } else {
            o
        }
    }

    /// `{dh, nh, n}` <-> `{dh, n, nh}`.
    fn swap() -> TransposeParams {
        TransposeParams {
            permutation: [0, 2, 1, 0, 0],
            tensor_dim: 3,
        }
    }

    /// Head rows `{dh, nh n}` -> `{dh, n, nh}`: the heads become the batch dimension.
    fn heads(&mut self, x: TensorId, y: &str) -> TensorId {
        let (dh, nh, n) = (self.d.dh, self.d.nh, self.n);
        let r = self.view(x, &[dh, nh, n], SynDataType::Bf16, &format!("{y}_r3"));
        let t = self.act(format!("{y}_h"), SynDataType::Bf16, &[dh, n, nh]);
        let p = Self::swap();
        self.gb
            .add_node("transpose", format!("{y}_transpose"), &[r], &[t], as_param_bytes(&p));
        t
    }

    /// `{dh, n, nh}` -> `{hid, n}`.
    fn unheads(&mut self, t: TensorId, y: &str) -> TensorId {
        let (dh, nh, n, h) = (self.d.dh, self.d.nh, self.n, self.d.hid);
        let r = self.act(format!("{y}_t3"), SynDataType::Bf16, &[dh, nh, n]);
        let p = Self::swap();
        self.gb
            .add_node("transpose", format!("{y}_transpose"), &[t], &[r], as_param_bytes(&p));
        self.view(r, &[h, n], SynDataType::Bf16, y)
    }

    /// Per-head norm over `dh` channels of `{dh, cols}`: the vendor norm with a unit weight gives
    /// `1 / rms` per column (and the weightless norm itself); with a weight, `(x / rms) w` in f32
    /// and one rounding (the vendor's own weighted output rounds twice).
    fn head_norm(&mut self, x: TensorId, w: Option<&str>, cols: u64, y: &str) -> TensorId {
        let dh = self.d.dh;
        let one = self.w("V_oned", SynDataType::Bf16, &[dh]);
        let u = self.act(
            if w.is_some() { format!("{y}_u") } else { y.to_string() },
            SynDataType::Bf16,
            &[dh, cols],
        );
        let ir = self.act(format!("{y}_ir"), SynDataType::Float, &[1, cols]);
        self.gb
            .op_rms_norm_bf16(format!("{y}_norm"), x, one, u, ir, self.d.eps, false);
        let Some(w) = w else {
            return u;
        };
        let f = SynDataType::Float;
        let xf = self.cast(x, f, &[dh, cols], &format!("{y}_xf"));
        let xn = self.bin("mult_fwd_f32", xf, ir, f, &[dh, cols], &format!("{y}_xn"));
        let wt = self.w(w, f, &[dh, 1]);
        let xw = self.bin("mult_fwd_f32", xn, wt, f, &[dh, cols], &format!("{y}_xw"));
        self.round(xw, &[dh, cols], y)
    }

    /// q or k rows `{hid, n}`: per-head norm with its weight, the 2-D rotary product
    /// `e = n cos + rotate_half(n) sin` (BF16 products and sum), heads to the batch dimension.
    fn qk(&mut self, x: TensorId, wn: &str, rot: TensorId, cs: TensorId, sn: TensorId, y: &str) -> TensorId {
        let (dh, c) = (self.d.dh, self.d.nh * self.n);
        let xc = self.view(x, &[dh, c], SynDataType::Bf16, &format!("{y}_c"));
        let n = self.head_norm(xc, Some(wn), c, &format!("{y}_n"));
        let bf = SynDataType::Bf16;
        let rn = self.act(format!("{y}_rot"), bf, &[dh, c]);
        self.gb.op_gemm(format!("{y}_rotg"), n, rot, rn, false, true);
        let a = self.bin("mult_fwd_bf16", n, cs, bf, &[dh, c], &format!("{y}_xc"));
        let b = self.bin("mult_fwd_bf16", rn, sn, bf, &[dh, c], &format!("{y}_xs"));
        let e = self.bin("add_fwd_bf16", a, b, bf, &[dh, c], &format!("{y}_e"));
        self.heads(e, y)
    }
}

fn new_graph(name: String, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = GraphBuilder::new(name);
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    gb
}

/// `g_g4v_patch_N<n>`: patch rows `v_pix {3 p p, n}` and position rows `v_pos {hid, n}` ->
/// `v_x0 {hid, n}` (the bias-free patch linear plus the positions, a BF16 sum).
pub fn build_patch_graph(d: &G4vDims, n: u64, elf: &[(String, String)]) -> GraphBuilder {
    let mut gb = new_graph(format!("g_g4v_patch_N{n}"), elf);
    let bf = SynDataType::Bf16;
    let mut b = Vb { gb: &mut gb, d, n };
    let pix = b.act("v_pix", bf, &[d.row_len(), n]);
    let pos = b.act("v_pos", bf, &[d.hid, n]);
    let pe = b.proj(pix, "V_patch", d.row_len(), d.hid, n, "v_pe");
    b.bin("add_fwd_bf16", pe, pos, bf, &[d.hid, n], "v_x0");
    gb
}

/// `g_g4v_block_N<n>`: one generic encoder layer `vb_x_in -> vb_x_out`, bound at launch to
/// layer `l`'s weights (`vb_*` -> `V<l>_*`):
///
/// ```text
/// h = norm(x);  q, k, v = W h;  q / k = rope(norm(.) w) per head,  v = norm(v)
/// a = softmax(q k^T) v               (scale 1, every patch sees every patch, f32)
/// x2 = x + norm(W_o a);  out = x2 + norm(W_down (gelu_tanh(W_gate h2) * W_up h2)),  h2 = norm(x2)
/// ```
pub fn build_block_graph(d: &G4vDims, n: u64, elf: &[(String, String)]) -> GraphBuilder {
    let mut gb = new_graph(format!("g_g4v_block_N{n}"), elf);
    let (bf, f) = (SynDataType::Bf16, SynDataType::Float);
    let (h, nh, dh, inter) = (d.hid, d.nh, d.dh, d.inter);
    let mut b = Vb { gb: &mut gb, d, n };
    let x = b.act("vb_x_in", bf, &[h, n]);
    let cs = b.act("v_cos", bf, &[dh, nh * n]);
    let sn = b.act("v_sin", bf, &[dh, nh * n]);
    let rot = b.w("V_rot_W", bf, &[dh, dh]);

    let h1 = b.norm_w(x, "vb_in_w", h, n, "vb_h1");
    let q = b.proj(h1, "vb_q", h, h, n, "vb_q");
    let k = b.proj(h1, "vb_k", h, h, n, "vb_k");
    let v = b.proj(h1, "vb_v", h, h, n, "vb_v");
    let qh = b.qk(q, "vb_qn_w", rot, cs, sn, "vb_qe");
    let kh = b.qk(k, "vb_kn_w", rot, cs, sn, "vb_ke");
    let vc = b.view(v, &[dh, nh * n], bf, "vb_vc");
    let vn = b.head_norm(vc, None, nh * n, "vb_vn");
    let vh = b.heads(vn, "vb_vh");

    let scores = b.act("vb_scores", f, &[n, n, nh]);
    b.gb.op_batch_gemm("vb_qk", qh, kh, scores, false, true);
    let pr = b.act("vb_prob", f, &[n, n, nh]);
    let sp = SoftmaxParams { dim: 0 };
    b.gb.add_node("softmax_fwd_f32", "vb_softmax", &[scores], &[pr], as_param_bytes(&sp));
    let vf = b.cast(vh, f, &[dh, n, nh], "vb_vf");
    let aof = b.act("vb_aof", f, &[dh, n, nh]);
    b.gb.op_batch_gemm("vb_pv", pr, vf, aof, false, false);
    // its consumers are a transpose and a GEMM: nothing to fuse the rounding away
    let ao = b.cast(aof, bf, &[dh, n, nh], "vb_ao");
    let a = b.unheads(ao, "vb_attn");
    let o = b.proj(a, "vb_o", h, h, n, "vb_o");
    let pa = b.norm_w(o, "vb_pa_w", h, n, "vb_pa");
    let x2 = b.bin("add_fwd_bf16", x, pa, bf, &[h, n], "vb_x2");

    let h2 = b.norm_w(x2, "vb_pfin_w", h, n, "vb_h2");
    let gt = b.proj(h2, "vb_gate", h, inter, n, "vb_gt");
    let up = b.proj(h2, "vb_up", h, inter, n, "vb_upo");
    // bf16(bf16(gelu_tanh(gate)) * up): the family's activation kernel, two roundings
    let m = b.act("vb_m", bf, &[inter, n]);
    b.gb.add_node("gaudium_ga_gelu", "vb_gelu", &[gt, up], &[m], &[]);
    let dn = b.proj(m, "vb_down", inter, h, n, "vb_dn");
    let pf = b.norm_w(dn, "vb_pf_w", h, n, "vb_pf");
    b.bin("add_fwd_bf16", x2, pf, bf, &[h, n], "vb_x_out");
    gb
}

/// `g_g4v_pool_N<n>`: the encoder's output `vp_x_in {hid, n}` (patches in pooling order) ->
/// `v_feat {out, n / pool^2}`: the mean of every block, rounded, times `sqrt(hid)`, standardised
/// where the checkpoint has the two vectors, rounded; then the weightless norm and the projection.
pub fn build_pool_graph(d: &G4vDims, n: u64, elf: &[(String, String)]) -> GraphBuilder {
    let mut gb = new_graph(format!("g_g4v_pool_N{n}"), elf);
    let (bf, f) = (SynDataType::Bf16, SynDataType::Float);
    let (h, p2) = (d.hid, d.ppt());
    let m = n / p2;
    let mut b = Vb { gb: &mut gb, d, n };
    let x = b.act("vp_x_in", bf, &[h, n]);
    let r = b.view(x, &[h * p2, m], bf, "vp_r");
    let wp = b.w("V_pool_W", bf, &[h * p2, h]);
    // f32 sums of BF16 rows: exact products, f32 accumulation
    let s = b.act("vp_sum", f, &[h, m]);
    b.gb.op_gemm("vp_sumg", r, wp, s, false, true);
    let inv = b.w("V_pool_s", f, &[1, 1]);
    let mean = b.bin("mult_fwd_f32", s, inv, f, &[h, m], "vp_mean");
    let mb = b.round(mean, &[h, m], "vp_mb");
    let mf = b.cast(mb, f, &[h, m], "vp_mf");
    let root = b.w("V_root", f, &[1, 1]);
    let mut a = b.bin("mult_fwd_f32", mf, root, f, &[h, m], "vp_sc");
    if d.standardize {
        let nb = b.w("V_std_nb", f, &[h, 1]);
        a = b.bin("add_fwd_f32", a, nb, f, &[h, m], "vp_sb");
        let ss = b.w("V_std_s", f, &[h, 1]);
        a = b.bin("mult_fwd_f32", a, ss, f, &[h, m], "vp_ss");
    }
    let y = b.round(a, &[h, m], "vp_y");
    let nrm = b.norm_w(y, "V_one", h, m, "vp_n");
    b.proj(nrm, "V_proj", h, d.out, m, "v_feat");
    gb
}

/// `gaudium_ga_lnorm` over `c` channels (weight and bias f32 in the converted-lane order).
fn lnorm(b: &mut Vb<'_>, x: TensorId, w: &str, c: u64, cols: u64, y: &str) -> TensorId {
    let f = SynDataType::Float;
    let g = b.w(format!("{w}_g"), f, &[c]);
    let bb = b.w(format!("{w}_b"), f, &[c]);
    let o = b.act(y, SynDataType::Bf16, &[c, cols]);
    b.gb.add_node(
        "gaudium_ga_lnorm",
        format!("{y}_ln"),
        &[x, g, bb],
        &[o],
        &LN_EPS.to_le_bytes(),
    );
    o
}

/// `g_g4u_img_R<r>`: `r` merged patches `u_px {3 (p pool)^2, r}` and their position rows
/// `u_pos {e, r}` -> soft tokens `v_feat {out, r}`:
///
/// ```text
/// h = LN1(x);  e = bf16(W_dense h + b) (f32 sum and bias, one rounding);  y = LN3(LN2(e) + pos)
/// soft = W_proj rmsnorm(y)        (a weightless norm)
/// ```
pub fn build_unified_graph(d: &G4vDims, r: u64, elf: &[(String, String)]) -> GraphBuilder {
    debug_assert_eq!(d.kind, G4vKind::Unified);
    let mut gb = new_graph(format!("g_g4u_img_R{r}"), elf);
    let (bf, f) = (SynDataType::Bf16, SynDataType::Float);
    let (row, e) = (d.row_len(), d.hid);
    let mut b = Vb { gb: &mut gb, d, n: r };
    let px = b.act("u_px", bf, &[row, r]);
    let pos = b.act("u_pos", bf, &[e, r]);
    let h1 = lnorm(&mut b, px, "U_ln1", row, r, "u_h1");
    let wd = b.w("U_dense_W", bf, &[row, e]);
    let s = b.act("u_dsum", f, &[e, r]);
    b.gb.op_gemm("u_dense", h1, wd, s, false, true);
    let bias = b.w("U_dense_b", f, &[e, 1]);
    let sb = b.bin("add_fwd_f32", s, bias, f, &[e, r], "u_dsb");
    let e1 = b.round(sb, &[e, r], "u_e");
    let h2 = lnorm(&mut b, e1, "U_ln2", e, r, "u_h2");
    let hp = b.bin("add_fwd_bf16", h2, pos, bf, &[e, r], "u_hp");
    let h3 = lnorm(&mut b, hp, "U_ln3", e, r, "u_h3");
    let nrm = b.norm_w(h3, "U_one", e, r, "u_n");
    let wpj = b.w("U_proj_W", bf, &[e, d.out]);
    let o = b.act("v_feat", bf, &[d.out, r]);
    b.gb.op_gemm("u_proj_gemm", nrm, wpj, o, false, true);
    gb
}

/// `g_g4_embed_vis_T<t>`: the decoder's embedding stage of a tile with image rows:
/// `x0 = bf16(table[id] scale) keep + feat[fidx]` (a text row keeps its embedding and reads the
/// zero column; an image row has keep 0 and its own feature column; both terms exact), then the
/// per-layer inputs of the small sizes from that `x0` and the ids (the pad id at image rows, as
/// the reference gathers them).
pub fn build_embed_vis_graph(td: &G4Dims, feat_cols: u64, t: u64, elf: &[(String, String)]) -> GraphBuilder {
    let mut gb = new_graph(format!("g_g4_embed_vis_T{t}"), elf);
    let (bf, i32t) = (SynDataType::Bf16, SynDataType::Int32);
    let ids = gb.add_persistent_tensor("ids", i32t, &[t]);
    let table = gb.get_or_add_persistent_tensor("embed_table", bf, &[td.hid, td.vocab]);
    let sc = gb.get_or_add_persistent_tensor("embed_scale", bf, &[1, 1]);
    let e = gb.add_persistent_tensor("x0g", bf, &[td.hid, t]);
    let es = gb.add_persistent_tensor("vis_xs", bf, &[td.hid, t]);
    gb.op_gather_bf16("embed", table, ids, e, 1);
    gb.op_mult_bf16("embed_scale", e, sc, es);
    let feat = gb.add_persistent_tensor("vis_feat", bf, &[td.hid, feat_cols]);
    let fidx = gb.add_persistent_tensor("vis_fidx", i32t, &[t]);
    let fg = gb.add_persistent_tensor("vis_fg", bf, &[td.hid, t]);
    gb.op_gather_bf16("vis_gather", feat, fidx, fg, 1);
    let keep = gb.add_persistent_tensor("vis_keep", bf, &[1, t]);
    let xk = gb.add_persistent_tensor("vis_xk", bf, &[td.hid, t]);
    gb.op_mult_bf16("vis_keep_mult", es, keep, xk);
    let x0 = gb.add_persistent_tensor("x0", bf, &[td.hid, t]);
    gb.op_add_bf16("vis_select", xk, fg, x0);
    if td.ple > 0 {
        let _ = crate::engine::gemma4::append_g4_ple(&mut gb, td, x0, ids, t, td.norm.row_norm(true), true);
    }
    gb
}
