//! MTP (multi-token prediction) drafter on the card: the target checkpoint's own MTP head drafts
//! the next tokens from the hidden rows a step has just verified.
//!
//! The head (`mtp.*` tensors of the target checkpoint, hybrid Qwen3.5 / 3.8 only) is one fusion
//! plus one full-attention layer with its own K/V cache over the whole sequence:
//!
//! ```text
//! row at position p:  x = W_fc_e . norm_e(embed(token[p + 1])) + W_fc_h . norm_h(h_final[p])
//!                     -> MTP layer (attends to its own rows <= p) -> mtp norm -> lm_head
//!                     -> argmax = a guess for token[p + 2]
//! ```
//!
//! What the card runs:
//! - **prefill** (`g_mtp_ctx_T*_C*`, one launch per prompt tile): the MTP layer's K/V rows of the
//!   prompt positions, from the trunk's last residual of the tile and the tile's tokens shifted
//!   by one;
//! - **first block** (`g_mtp_first_*`, once per request): the row of the last prompt position
//!   (its next token is the prefill's token) and the first drafts;
//! - **every step** (appended to the step recipe after the accept): the *catch-up* rows of the
//!   positions the step verified (ids = the verify argmaxes, `h` = the verify's `h_final` rows),
//!   written in place into the MTP cache, then `K - 1` recursive drafts. Draft 1 is the catch-up
//!   argmax of the accepted row; draft `i >= 2` feeds the previous draft's token and the previous
//!   pass's layer output back in at position `cursor_out + i - 2`.
//!
//! The catch-up rows past the accepted one hold wrong context, exactly like the trunk's rejected
//! verify rows: the next step's catch-up overwrites them before any row may attend to them (the
//! mask is by position). The recursive drafts must not overwrite the cache rows the next step
//! still reads as history, so they run the *fresh-window* attention builds: the cache is read
//! below `cursor_out` only and the drafts' own K/V rows live in a 256-slot window that each pass
//! copies forward. A draft therefore costs the same at any context length.
//!
//! The drafter only proposes: the verify decides every token, so nothing here affects exactness.
//!
//! Vision note (not built here): at image positions the fusion must read the vision embedding
//! row, not the embedding of the image-pad token. The fusion takes its embedding rows as a
//! tensor (`append_fuse`), so a vision prefill only swaps the gather of the context recipe for
//! the merged embedding rows. Positions are RoPE positions of the trunk (`cursor` based), never
//! cache slots.

use std::borrow::Cow;

use super::super::*;
use super::SpecRuntime;
use crate::tree::TreeShape;

/// Kernel ELFs the MTP passes need beyond the trunk's and the verify's.
pub const MTP_ELFS: &[&str] = &[
    "GAUDIUM_APREPF_ELF",
    "GAUDIUM_ASOFTF_ELF",
    "GAUDIUM_ACOMBF_ELF",
    "GAUDIUM_SEL_ELF",
    "GAUDIUM_NEXTIDS_ELF",
];

/// [`MTP_ELFS`] for the target's attention shape: the fresh-window attention front end is built
/// per `(KV heads, query heads per KV head)` shape, and the MoE family's is `g2q8`.
pub fn mtp_elfs(spec: &ModelSpec) -> Vec<&'static str> {
    MTP_ELFS
        .iter()
        .map(|&e| {
            if e == "GAUDIUM_APREPF_ELF" && spec.n_kv_heads == 2 && spec.n_heads == 16 {
                "GAUDIUM_APREPF_G2Q8_ELF"
            } else {
                e
            }
        })
        .collect()
}

/// Kernel ELFs of the tree candidates (top-4 readouts, candidate table, verify ids).
pub const MTP_TREE_ELFS: &[&str] = &[
    "GAUDIUM_MTOPK1_ELF",
    "GAUDIUM_MTOPK2_ELF",
    "GAUDIUM_TREECAND_ELF",
    "GAUDIUM_TREEIDS_ELF",
];

/// Slots of the drafts' fresh window (the window softmax walks 256-slot blocks).
const NF: u64 = 256;
/// Candidates per drafted depth (the top-k kernels' build) and candidate-table columns.
const KT: u64 = 4;
const CM: u64 = 16;
/// Most drafts one step may chain (candidate-table columns `1..CM`).
pub const MTP_MAX_DRAFTS: u64 = CM - 1;

/// Weight / table names of the MTP head (bound from this module's weight arena).
pub fn is_mtp_weight(name: &str) -> bool {
    matches!(
        name,
        "mtp_w_e"
            | "mtp_w_h"
            | "mtp_w_norm"
            | "mtp_W_fc_e"
            | "mtp_W_fc_h"
            | "mtp_fpos"
            | "mtp_fz_k"
            | "mtp_fz_v"
            | "mtp_W_lmd"
            | "mtp_dv_ids"
    ) || name.strip_prefix("M_").is_some_and(|s| {
        matches!(
            s,
            "w_in"
                | "w_post"
                | "W_qg"
                | "W_k"
                | "W_v"
                | "qn_w"
                | "kn_w"
                | "W_o"
                | "W_gate"
                | "W_up"
                | "W_down"
        ) || crate::engine::moe::is_ext_layer_weight(s)
            && !s.ends_with("_cs")
            && !s.ends_with("_sa")
    })
}

/// The MTP layer's K/V cache (zeroed per request with the other drafter caches).
pub fn is_mtp_kv(name: &str) -> bool {
    matches!(name, "M_kcache" | "M_vcache")
}

/// Tensors of this module's graphs that are bound to buffers owned by the engine.
fn is_external(name: &str) -> bool {
    matches!(name, "mtp_ctx_x" | "ctx_cursor" | "mtp_h0")
}

/// Moves this module's weights and caches out of the scratch plan the generic pass put them in.
pub fn reclassify(
    weights: &mut BTreeMap<String, u64>,
    kv: &mut BTreeMap<String, u64>,
    act: &mut BTreeMap<String, u64>,
) {
    let names: Vec<String> = act.keys().cloned().collect();
    for name in names {
        if is_mtp_weight(&name) {
            let b = act.remove(&name).unwrap();
            let e = weights.entry(name).or_insert(0);
            *e = (*e).max(b);
        } else if is_mtp_kv(&name) {
            let b = act.remove(&name).unwrap();
            let e = kv.entry(name).or_insert(0);
            *e = (*e).max(b);
        } else if is_external(&name) {
            act.remove(&name);
        }
    }
}

// ============================================================================
// Graph passes
// ============================================================================

fn lm_weight(spec: &ModelSpec) -> &'static str {
    if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    }
}

/// The readout of MTP rows: `(weight name, rows)`. `nvd > 0` reads only `nvd` rows of the
/// `lm_head` (a copy, `mtp_W_lmd`, of the rows [`draft_vocab_ids`]): drafts are guesses, so a
/// token outside the subset only costs an acceptance. Readout indices map back to token ids
/// through the table `mtp_dv_ids` ([`dv_map`]).
fn draft_head(spec: &ModelSpec, nvd: u64) -> (&'static str, u64) {
    if nvd > 0 {
        ("mtp_W_lmd", nvd)
    } else {
        (lm_weight(spec), spec.vocab)
    }
}

/// Ids a block of control tokens is assumed to span from the lowest EOS id on.
pub(super) const DV_TAIL: u64 = 128;

/// Token ids of the draft vocabulary of `nvd` rows: the lowest ids (the tokenizer's merge
/// order puts frequent tokens first) and, when it lies beyond them, the block of control tokens
/// that starts at the lowest EOS id (end-of-turn and reasoning tags are drafted often).
pub fn draft_vocab_ids(spec: &ModelSpec, nvd: u64) -> Vec<i32> {
    let tail = spec
        .eos_token_ids
        .iter()
        .min()
        .map(|&e| e as u64)
        .filter(|&e| nvd >= DV_TAIL && e + DV_TAIL > nvd && e < spec.vocab);
    match tail {
        Some(e0) => (0..nvd - DV_TAIL)
            .chain((e0..e0 + DV_TAIL).map(|id| id.min(spec.vocab - 1)))
            .map(|id| id as i32)
            .collect(),
        None => (0..nvd as i32).collect(),
    }
}

/// Draft-vocabulary indices `li` (`dims`) -> token ids, as the persistent tensor `name`.
fn dv_map(gb: &mut GraphBuilder, li: TensorId, dims: &[u64], name: &str, nvd: u64) -> TensorId {
    let n: u64 = dims.iter().product();
    let tab = gb.get_or_add_persistent_tensor("mtp_dv_ids", SynDataType::Int32, &[1, nvd]);
    let flat = gb.add_intermediate(format!("{name}_flat"), SynDataType::Int32, &[n]);
    gb.op_reshape(format!("{name}_rs"), li, flat);
    let o2 = gb.add_intermediate(format!("{name}_o2"), SynDataType::Int32, &[1, n]);
    gb.add_node(
        "gather_fwd_i32",
        format!("{name}_dvmap"),
        &[tab, flat],
        &[o2],
        crate::synapse::ops::as_param_bytes(&crate::synapse::ops::GatherParams { axis: 1 }),
    );
    let out = gb.add_persistent_tensor(name, SynDataType::Int32, dims);
    gb.op_reshape(format!("{name}_rs_out"), o2, out);
    out
}

/// Argmax of `{V', t}` logits as token ids `{1, t}` named `name`.
fn append_argmax(
    gb: &mut GraphBuilder,
    logits: TensorId,
    name: &str,
    t: u64,
    nvd: u64,
) -> TensorId {
    if nvd == 0 {
        let tok = gb.add_persistent_tensor(name, SynDataType::Int32, &[1, t]);
        gb.op_argmax_bf16(format!("{name}_argmax"), logits, tok, 0);
        return tok;
    }
    let li = gb.add_persistent_tensor(format!("{name}_li"), SynDataType::Int32, &[1, t]);
    gb.op_argmax_bf16(format!("{name}_argmax"), logits, li, 0);
    dv_map(gb, li, &[1, t], name, nvd)
}

/// The MTP fusion over `t` rows: `x = W_fc_e . norm_e(e) + W_fc_h . norm_h(h)` and the MTP
/// layer's input norm of `x` (one fused node). `e` = embedding rows of the rows' next tokens.
fn append_fuse(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    e: TensorId,
    h: TensorId,
    ap: &str,
    t: u64,
) -> (TensorId, TensorId) {
    let hid = spec.hidden;
    let en = add_rmsnorm(gb, spec, e, "mtp_w_e", &format!("{ap}en"), t);
    let hn = add_rmsnorm(gb, spec, h, "mtp_w_h", &format!("{ap}hn"), t);
    let fe = add_linear_bf16(gb, en, "mtp_W_fc_e", hid, hid, t, &format!("{ap}fe"));
    let fh = add_linear_bf16(gb, hn, "mtp_W_fc_h", hid, hid, t, &format!("{ap}fh"));
    add_fused_add_norm(
        gb,
        spec,
        fe,
        fh,
        "M_w_in",
        &format!("{ap}x"),
        &format!("{ap}h"),
        t,
    )
}

/// Rows `t` of the MTP head written in place into the MTP cache at `cursor .. cursor + t`
/// (prefix `M_`): returns `(layer output, argmax tokens {1, t}, logits {V, t})`.
#[allow(clippy::too_many_arguments)]
fn append_inplace_rows(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    ids: TensorId,
    h_rows: TensorId,
    cursor: TensorId,
    t: u64,
    nch: u64,
    lmax: u64,
    nvd: u64,
) -> (TensorId, TensorId, TensorId) {
    let hid = spec.hidden;
    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, spec.vocab]);
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let e = gb.add_persistent_tensor("M_e", SynDataType::Bf16, &[hid, t]);
    gb.op_gather_bf16("M_embed", table, ids, e, 1);
    let (x, h64) = append_fuse(gb, spec, e, h_rows, "M_", t);
    let (out, hn) = append_attn_layer(
        gb,
        spec,
        x,
        Some(h64),
        cursor,
        rope_c,
        rope_s,
        "M_",
        t,
        nch,
        true,
        Some(("mtp_w_norm", "M_hfin")),
    );
    let hn = hn.expect("fused MTP norm");
    let (lm_w, lm_rows) = draft_head(spec, nvd);
    let logits = add_linear_bf16(gb, hn, lm_w, hid, lm_rows, t, "M_logits");
    let tok = append_argmax(gb, logits, "M_tok", t, nvd);
    (out, tok, logits)
}

/// Where a recursive draft sits: its position is `pos_cursor + pos_off`, the MTP cache is valid
/// below `pos_cursor + big_off`, and its K/V row goes to window slot `slot`.
struct DraftAt {
    kf_in: TensorId,
    vf_in: TensorId,
    pos_cursor: TensorId,
    pos_off: u32,
    big_off: u32,
    slot: u32,
}

struct DraftOut {
    out: TensorId,
    tok: TensorId,
    logits: TensorId,
    kf: TensorId,
    vf: TensorId,
}

/// One recursive draft (one row): fusion of `ids` and `h_in`, the MTP layer with the fresh-window
/// attention (cache below the step's new cursor + the window of this step's earlier drafts), the
/// MTP norm, `lm_head`, argmax.
#[allow(clippy::too_many_arguments)]
fn append_draft_pass(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    ap: &str,
    ids: TensorId,
    h_in: TensorId,
    at: &DraftAt,
    nch: u64,
    lmax: u64,
    nvd: u64,
) -> DraftOut {
    let hid = spec.hidden;
    let inter = spec.intermediate;
    let nh = spec.n_heads;
    let nkv = spec.n_kv_heads;
    let dh = spec.head_dim;
    let r = nh / nkv;
    let b = nkv * nch;
    let n = |s: &str| format!("{ap}{s}");

    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, spec.vocab]);
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let zero = gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]);
    let fpos = gb.get_or_add_persistent_tensor("mtp_fpos", SynDataType::Int32, &[NF]);

    let e = gb.add_persistent_tensor(n("e"), SynDataType::Bf16, &[hid, 1]);
    gb.op_gather_bf16(n("embed"), table, ids, e, 1);
    let (x, h) = append_fuse(gb, spec, e, h_in, ap, 1);

    let qg = add_linear_bf16(gb, h, "M_W_qg", hid, spec.qg_dim(), 1, &n("qg"));
    let k = add_linear_bf16(gb, h, "M_W_k", hid, spec.kv_dim(), 1, &n("k"));
    let v = add_linear_bf16(gb, h, "M_W_v", hid, spec.kv_dim(), 1, &n("v"));
    let qn_w = gb.get_or_add_persistent_tensor("M_qn_w", SynDataType::Float, &[dh]);
    let kn_w = gb.get_or_add_persistent_tensor("M_kn_w", SynDataType::Float, &[dh]);
    let kbig = gb.get_or_add_persistent_tensor("M_kcache", SynDataType::Bf16, &[dh, KV_CHUNK, b]);
    let vbig = gb.get_or_add_persistent_tensor("M_vcache", SynDataType::Bf16, &[dh, KV_CHUNK, b]);

    let qb = gb.add_persistent_tensor(n("qb"), SynDataType::Bf16, &[dh, r, b]);
    let qb_f = gb.add_persistent_tensor(n("qb_f"), SynDataType::Bf16, &[dh, r, nkv]);
    let kf = gb.add_persistent_tensor(n("kf"), SynDataType::Bf16, &[dh, NF, nkv]);
    let vf = gb.add_persistent_tensor(n("vf"), SynDataType::Bf16, &[dh, NF, nkv]);
    // fresh-window build: params = position offset, window slot, chunk length of the cache
    gb.add_node(
        "gaudium_attn_prep",
        n("attn_prep"),
        &[
            qg,
            k,
            v,
            qn_w,
            kn_w,
            rope_c,
            rope_s,
            at.pos_cursor,
            at.kf_in,
            at.vf_in,
        ],
        &[qb, qb_f, kf, vf],
        &u32_params(&[at.pos_off, at.slot, KV_CHUNK as u32]),
    );

    // the cache below the new cursor (mode 1: no row of this pass is in it)
    let scores = gb.add_persistent_tensor(n("scores"), SynDataType::Float, &[KV_CHUNK, r, b]);
    gb.op_batch_gemm(n("qk"), qb, kbig, scores, false, true);
    let pm = gb.add_persistent_tensor(n("P"), SynDataType::Bf16, &[KV_CHUNK, r, b]);
    let stats = gb.add_persistent_tensor(n("stats"), SynDataType::Float, &[64, 2, r, b]);
    gb.add_node(
        "gaudium_attn_softmax_chunk",
        n("softmax"),
        &[scores, at.pos_cursor],
        &[pm, stats],
        &u32_params(&[1, nch as u32, at.big_off, 1]),
    );
    let ao_part = gb.add_persistent_tensor(n("ao_part"), SynDataType::Float, &[dh, r, b]);
    gb.op_batch_gemm(n("pv"), pm, vbig, ao_part, false, false);

    // the window: slot j is valid for the draft in slot s iff j <= s (table = slot index, query
    // position = the slot, read through a zero cursor)
    let sf = gb.add_persistent_tensor(n("scores_f"), SynDataType::Float, &[NF, r, nkv]);
    gb.op_batch_gemm(n("qk_f"), qb_f, kf, sf, false, true);
    let pf = gb.add_persistent_tensor(n("P_f"), SynDataType::Bf16, &[NF, r, nkv]);
    let stf = gb.add_persistent_tensor(n("stats_f"), SynDataType::Float, &[64, 2, r, nkv]);
    gb.add_node(
        "gaudium_attn_softmax_fresh",
        n("softmax_f"),
        &[sf, fpos, zero],
        &[pf, stf],
        &u32_params(&[1, at.slot]),
    );
    let aof = gb.add_persistent_tensor(n("ao_f"), SynDataType::Float, &[dh, r, nkv]);
    gb.op_batch_gemm(n("pv_f"), pf, vf, aof, false, false);

    let ao = gb.add_persistent_tensor(n("ao"), SynDataType::Float, &[dh, r, nkv]);
    gb.add_node(
        "gaudium_attn_combine",
        n("combine"),
        &[ao_part, stats, aof, stf],
        &[ao],
        &u32_params(&[nch as u32]),
    );
    let core = gb.add_persistent_tensor(n("core"), SynDataType::Bf16, &[nh * dh, 1]);
    gb.add_node(
        "gaudium_attn_post_f32",
        n("attn_post"),
        &[ao, qg],
        &[core],
        &[],
    );
    let attn = add_linear_bf16(gb, core, "M_W_o", nh * dh, hid, 1, &n("attn"));

    // residual + norm + SwiGLU MLP + residual, fused with the MTP output norm
    let (x2, h2) = add_fused_add_norm(gb, spec, x, attn, "M_w_post", &n("x2"), &n("h2"), 1);
    let (out, hn) = if spec.is_moe() {
        // a MoE target's MTP layer is a MoE layer too: the head's experts under this pass's names
        let (out, hn) = crate::engine::moe::append_moe_mlp_as(
            gb,
            spec,
            x2,
            h2,
            "M_",
            ap,
            1,
            true,
            Some(("mtp_w_norm", &n("hfin"))),
        );
        (out, hn.expect("fused MTP norm"))
    } else {
        let gate = add_linear_bf16(gb, h2, "M_W_gate", hid, inter, 1, &n("gate"));
        let up = add_linear_bf16(gb, h2, "M_W_up", hid, inter, 1, &n("up"));
        let act = gb.add_persistent_tensor(n("act"), SynDataType::Bf16, &[inter, 1]);
        gb.op_silu_bf16(n("silu"), gate, act);
        let m = gb.add_persistent_tensor(n("m"), SynDataType::Bf16, &[inter, 1]);
        gb.op_mult_bf16(n("mul"), act, up, m);
        let mo = add_linear_bf16(gb, m, "M_W_down", inter, hid, 1, &n("mo"));
        add_fused_add_norm(gb, spec, x2, mo, "mtp_w_norm", &n("out"), &n("hfin"), 1)
    };

    let (lm_w, lm_rows) = draft_head(spec, nvd);
    let logits = add_linear_bf16(gb, hn, lm_w, hid, lm_rows, 1, &n("logits"));
    let tok = append_argmax(gb, logits, &n("tok"), 1, nvd);
    DraftOut {
        out,
        tok,
        logits,
        kf,
        vf,
    }
}

/// Inputs of the MTP section of a recipe.
pub struct MtpIn {
    /// In-place rows (the catch-up of a step, or the single row of the first block).
    pub t: u64,
    /// Drafts produced (the next step's spine): draft 1 from the in-place rows, the rest recursive.
    pub kd: u64,
    pub nch: u64,
    pub lmax: u64,
    /// Rows of the draft readout (0 = the full `lm_head`).
    pub nvd: u64,
    /// `{t}` next tokens of the in-place rows, `{HID, t}` their `h_final` rows.
    pub ids: TensorId,
    pub h_rows: TensorId,
    /// `{1}`: the in-place row the drafts continue from (the accepted row).
    pub sel: TensorId,
    /// Position of in-place row 0.
    pub cursor: TensorId,
    /// Position tensor the recursive drafts count from, and their offsets: draft `i` sits at
    /// `draft_cursor + draft_off + i - 2`; the cache is valid below `draft_cursor + draft_off`.
    pub draft_cursor: TensorId,
    pub draft_off: u32,
}

/// What the MTP section produced: the in-place argmaxes / logits and the recursive drafts.
pub struct MtpOut {
    pub c_tok: TensorId,
    pub c_logits: TensorId,
    pub d_toks: Vec<TensorId>,
    pub d_logits: Vec<TensorId>,
}

/// Appends the in-place rows and the `kd - 1` recursive drafts.
pub fn append_mtp(gb: &mut GraphBuilder, spec: &ModelSpec, i: &MtpIn) -> MtpOut {
    let hid = spec.hidden;
    let (c_out, c_tok, c_logits) = append_inplace_rows(
        gb, spec, i.ids, i.h_rows, i.cursor, i.t, i.nch, i.lmax, i.nvd,
    );
    let zero = gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]);
    let dh = spec.head_dim;
    let nkv = spec.n_kv_heads;
    let mut prev_k = gb.get_or_add_persistent_tensor("mtp_fz_k", SynDataType::Bf16, &[dh, NF, nkv]);
    let mut prev_v = gb.get_or_add_persistent_tensor("mtp_fz_v", SynDataType::Bf16, &[dh, NF, nkv]);
    let mut prev_out = c_out;
    let mut prev_tok = c_tok;
    let mut d_toks = Vec::new();
    let mut d_logits = Vec::new();
    for d in 2..=i.kd {
        let ap = format!("d{d}_");
        let ids = gb.add_persistent_tensor(format!("{ap}ids"), SynDataType::Int32, &[1]);
        let h_in = if d == 2 {
            gb.add_node(
                "gaudium_sel",
                format!("{ap}pick_id"),
                &[prev_tok, i.sel],
                &[ids],
                &[],
            );
            let h = gb.add_persistent_tensor(format!("{ap}h_in"), SynDataType::Bf16, &[hid, 1]);
            gb.op_gather_bf16(format!("{ap}pick_h"), prev_out, i.sel, h, 1);
            h
        } else {
            gb.add_node(
                "gaudium_sel",
                format!("{ap}pick_id"),
                &[prev_tok, zero],
                &[ids],
                &[],
            );
            prev_out
        };
        let at = DraftAt {
            kf_in: prev_k,
            vf_in: prev_v,
            pos_cursor: i.draft_cursor,
            pos_off: i.draft_off + (d as u32 - 2),
            big_off: i.draft_off,
            slot: d as u32 - 2,
        };
        let o = append_draft_pass(gb, spec, &ap, ids, h_in, &at, i.nch, i.lmax, i.nvd);
        prev_k = o.kf;
        prev_v = o.vf;
        prev_out = o.out;
        prev_tok = o.tok;
        d_toks.push(o.tok);
        d_logits.push(o.logits);
    }
    MtpOut {
        c_tok,
        c_logits,
        d_toks,
        d_logits,
    }
}

/// The next step's ids `{kd + 1}` of a chain: `[tokens[a], draft 1 .. draft kd]`.
pub fn append_chain_next_ids(
    gb: &mut GraphBuilder,
    tokens: TensorId,
    a: TensorId,
    m: &MtpOut,
    kd: u64,
    out_name: &str,
) -> TensorId {
    let rec = match m.d_toks.len() {
        0 => m.c_tok,
        1 => m.d_toks[0],
        n => {
            let rec = gb.add_persistent_tensor("mtp_drafts", SynDataType::Int32, &[1, n as u64]);
            gb.op_concat("mtp_drafts_cat", &m.d_toks, rec, 1);
            rec
        }
    };
    let next = gb.add_persistent_tensor(out_name, SynDataType::Int32, &[kd + 1]);
    gb.add_node(
        "gaudium_next_ids",
        "mtp_next_ids",
        &[tokens, a, m.c_tok, rec],
        &[next],
        &[],
    );
    next
}

/// Top-4 candidates (ids, softmax probabilities) per row of `{V, t}` logits.
fn append_topk(
    gb: &mut GraphBuilder,
    logits: TensorId,
    name: &str,
    t: u64,
    nvd: u64,
) -> (TensorId, TensorId) {
    let ch = (48 / t).max(1);
    let pv = gb.add_persistent_tensor(format!("{name}_pv"), SynDataType::Float, &[64, KT, ch, t]);
    let pi = gb.add_persistent_tensor(format!("{name}_pi"), SynDataType::Int32, &[64, KT, ch, t]);
    let ps = gb.add_persistent_tensor(format!("{name}_ps"), SynDataType::Float, &[64, ch, t]);
    gb.add_node(
        "gaudium_mtp_topk1",
        format!("{name}_topk1"),
        &[logits],
        &[pv, pi, ps],
        &[],
    );
    let ids_name = if nvd > 0 {
        format!("{name}_lids")
    } else {
        format!("{name}_ids")
    };
    let ids = gb.add_persistent_tensor(ids_name, SynDataType::Int32, &[KT, t]);
    let vals = gb.add_persistent_tensor(format!("{name}_vals"), SynDataType::Float, &[KT, t]);
    let probs = gb.add_persistent_tensor(format!("{name}_probs"), SynDataType::Float, &[KT, t]);
    gb.add_node(
        "gaudium_mtp_topk2",
        format!("{name}_topk2"),
        &[pv, pi, ps],
        &[ids, vals, probs],
        &[],
    );
    if nvd > 0 {
        let mapped = dv_map(gb, ids, &[KT, t], &format!("{name}_ids"), nvd);
        return (mapped, probs);
    }
    (ids, probs)
}

/// The next step's ids `{N}` of a tree shape: the spine is the drafts, a rank-`r` leaf at depth
/// `j` the `(r + 1)`-th candidate of the readout that produced spine draft `j`.
#[allow(clippy::too_many_arguments)]
pub fn append_tree_next_ids(
    gb: &mut GraphBuilder,
    tokens: TensorId,
    a: TensorId,
    dep: TensorId,
    m: &MtpOut,
    t_c: u64,
    kd: u64,
    depth_tab: TensorId,
    rank_tab: TensorId,
    n_nodes: u64,
    nvd: u64,
    out_name: &str,
) -> TensorId {
    let (c_top, c_prob) = append_topk(gb, m.c_logits, "mtp_ctop", t_c, nvd);
    let (d_tok, d_top, d_prob) = match m.d_toks.len() {
        0 => (m.c_tok, c_top, c_prob),
        n => {
            let mut tops = Vec::new();
            let mut probs = Vec::new();
            for (j, &lg) in m.d_logits.iter().enumerate() {
                let (ids, pr) = append_topk(gb, lg, &format!("mtp_dtop{}", j + 2), 1, nvd);
                tops.push(ids);
                probs.push(pr);
            }
            if n == 1 {
                (m.d_toks[0], tops[0], probs[0])
            } else {
                let n = n as u64;
                let tok = gb.add_persistent_tensor("mtp_drafts", SynDataType::Int32, &[1, n]);
                gb.op_concat("mtp_drafts_cat", &m.d_toks, tok, 1);
                let top = gb.add_persistent_tensor("mtp_dtop", SynDataType::Int32, &[KT, n]);
                gb.op_concat("mtp_dtop_cat", &tops, top, 1);
                let prob = gb.add_persistent_tensor("mtp_dprob", SynDataType::Float, &[KT, n]);
                gb.op_concat("mtp_dprob_cat", &probs, prob, 1);
                (tok, top, prob)
            }
        }
    };
    let cand = gb.add_persistent_tensor("mtp_cand", SynDataType::Int32, &[KT, CM]);
    let cprob = gb.add_persistent_tensor("mtp_cprob", SynDataType::Float, &[KT, CM]);
    gb.add_node(
        "gaudium_tree_cand",
        "mtp_tree_cand",
        &[tokens, a, dep, m.c_tok, c_top, d_tok, d_top, c_prob, d_prob],
        &[cand, cprob],
        &u32_params(&[kd as u32]),
    );
    let next = gb.add_persistent_tensor(out_name, SynDataType::Int32, &[n_nodes]);
    gb.add_node(
        "gaudium_tree_ids",
        "mtp_tree_ids",
        &[cand, depth_tab, rank_tab],
        &[next],
        &[],
    );
    next
}

/// A chain step with the MTP drafter: the verify / accept of `t` rows (the proven chain graph,
/// unchanged), the catch-up of those rows, `t - 2` recursive drafts, the next ids. The 1-row
/// fallback step near the end of the context only hands its token on.
pub fn build_chain_step_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    t: u64,
    ns: u64,
    nvd: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    build_chain_step_graph_as(spec, lmax, nch, t, ns, nvd, elf_hashes, false)
}

/// [`build_chain_step_graph`], or (`sampled`) the chain step of a sampled request: the verify
/// rows sample on the card at their own positions (`g_spec_mtps_*`), the catch-up rows are fed
/// those sampled tokens, and the drafts stay the head's greedy guesses.
#[allow(clippy::too_many_arguments)]
pub fn build_chain_step_graph_as(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    t: u64,
    ns: u64,
    nvd: u64,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    build_chain_step_graph_kd(
        spec,
        lmax,
        nch,
        t,
        t.saturating_sub(1),
        ns,
        nvd,
        elf_hashes,
        sampled,
    )
}

/// [`build_chain_step_graph_as`] with the drafts for the next step (`kd`) chosen apart from this
/// step's verify rows: a runtime with several verify widths drafts its widest block in every
/// step, so the width of the next step can be chosen when that step is enqueued (recipes
/// `g_spec_mtp*_T{t}_D{kd}_C{nch}`). `kd == t - 1` is the fixed-width step.
#[allow(clippy::too_many_arguments)]
pub fn build_chain_step_graph_kd(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    t: u64,
    kd: u64,
    ns: u64,
    nvd: u64,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    let mut gb = super::build_spec_step_graph_as(spec, lmax, nch, t, ns, None, elf_hashes, sampled);
    let fam = if sampled { "mtps" } else { "mtp" };
    gb.set_recipe_name(if t > 1 && kd != t - 1 {
        format!("g_spec_{fam}_T{t}_D{kd}_C{nch}")
    } else {
        format!("g_spec_{fam}_T{t}_C{nch}")
    });
    let tokens = gb.get_or_add_persistent_tensor("tokens", SynDataType::Int32, &[1, t]);
    let a_out = gb.get_or_add_persistent_tensor("a_out", SynDataType::Int32, &[1]);
    if t == 1 {
        let next = gb.add_persistent_tensor("next_ids", SynDataType::Int32, &[1]);
        gb.add_node(
            "gaudium_sel",
            "mtp_next_ids",
            &[tokens, a_out],
            &[next],
            &[],
        );
        return gb;
    }
    let cursor = gb.get_or_add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let cursor_out = gb.get_or_add_persistent_tensor("cursor_out", SynDataType::Int32, &[1]);
    let h_final = gb.get_or_add_persistent_tensor("h_final", SynDataType::Bf16, &[spec.hidden, t]);
    let zero = gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]);
    let c_ids = gb.add_persistent_tensor("mtp_c_ids", SynDataType::Int32, &[t]);
    gb.add_node(
        "gaudium_sel",
        "mtp_c_ids_sel",
        &[tokens, zero],
        &[c_ids],
        &[],
    );
    let m = append_mtp(
        &mut gb,
        spec,
        &MtpIn {
            t,
            kd,
            nch,
            lmax,
            nvd,
            ids: c_ids,
            h_rows: h_final,
            sel: a_out,
            cursor,
            draft_cursor: cursor_out,
            draft_off: 0,
        },
    );
    append_chain_next_ids(&mut gb, tokens, a_out, &m, kd, "next_ids");
    gb
}

/// MTP K/V rows of a prefill tile: `mtp_ctx_x` = the trunk's last residual of the tile,
/// `mtp_ctx_ids` = the tile's tokens shifted by one, `ctx_cursor` = the tile's first position.
pub fn build_ctx_graph(
    spec: &ModelSpec,
    t: u64,
    lmax: u64,
    nch: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_mtp_ctx_T{t}_C{nch}"));
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let hid = spec.hidden;
    let nkv = spec.n_kv_heads;
    let dh = spec.head_dim;
    let r = (spec.n_heads / nkv) * t;
    let x_in = gb.add_persistent_tensor("mtp_ctx_x", SynDataType::Bf16, &[hid, t]);
    let ids = gb.add_persistent_tensor("mtp_ctx_ids", SynDataType::Int32, &[t]);
    let cursor = gb.add_persistent_tensor("ctx_cursor", SynDataType::Int32, &[1]);
    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, spec.vocab]);
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let h = add_rmsnorm(&mut gb, spec, x_in, "w_final", "mc_hf", t);
    let e = gb.add_persistent_tensor("mc_e", SynDataType::Bf16, &[hid, t]);
    gb.op_gather_bf16("mc_embed", table, ids, e, 1);
    let (_x, h64) = append_fuse(&mut gb, spec, e, h, "mc_", t);
    let qg = add_linear_bf16(&mut gb, h64, "M_W_qg", hid, spec.qg_dim(), t, "mc_qg");
    let k = add_linear_bf16(&mut gb, h64, "M_W_k", hid, spec.kv_dim(), t, "mc_k");
    let v = add_linear_bf16(&mut gb, h64, "M_W_v", hid, spec.kv_dim(), t, "mc_v");
    let qn_w = gb.get_or_add_persistent_tensor("M_qn_w", SynDataType::Float, &[dh]);
    let kn_w = gb.get_or_add_persistent_tensor("M_kn_w", SynDataType::Float, &[dh]);
    // only K / V are kept: one q block (never read) beside the cache of `nch` chunks
    let qb = gb.add_persistent_tensor("mc_qb", SynDataType::Bf16, &[dh, r, nkv]);
    let kc = gb.add_persistent_tensor("M_kcache", SynDataType::Bf16, &[dh, KV_CHUNK, nkv * nch]);
    let vc = gb.add_persistent_tensor("M_vcache", SynDataType::Bf16, &[dh, KV_CHUNK, nkv * nch]);
    gb.add_node(
        "gaudium_attn_prep",
        "mc_attn_prep",
        &[qg, k, v, qn_w, kn_w, rope_c, rope_s, cursor],
        &[qb, kc, vc],
        &u32_params(&[0u32]),
    );
    gb
}

/// The first block of a request: the MTP row of the last prompt position `mtp_cur0` (next token
/// `mtp_tok0` = the prefill's token, `mtp_h0` = the prefill head's `h_final`), the recursive
/// drafts, and the first step's ids in `mtp_first_ids`.
pub fn build_first_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    kd: u64,
    nvd: u64,
    tree: Option<&TreeShape>,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let tag = tree.map_or_else(|| format!("K{kd}"), super::tree_step::shape_key);
    let mut gb = GraphBuilder::new(format!("g_mtp_first_{tag}_C{nch}"));
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let tok0 = gb.add_persistent_tensor("mtp_tok0", SynDataType::Int32, &[1, 1]);
    let cur0 = gb.add_persistent_tensor("mtp_cur0", SynDataType::Int32, &[1]);
    let h0 = gb.add_persistent_tensor("mtp_h0", SynDataType::Bf16, &[spec.hidden, 1]);
    let zero = gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]);
    let ids0 = gb.add_persistent_tensor("mtp_ids0", SynDataType::Int32, &[1]);
    gb.add_node("gaudium_sel", "mtp_ids0_sel", &[tok0, zero], &[ids0], &[]);
    let m = append_mtp(
        &mut gb,
        spec,
        &MtpIn {
            t: 1,
            kd,
            nch,
            lmax,
            nvd,
            ids: ids0,
            h_rows: h0,
            sel: zero,
            cursor: cur0,
            draft_cursor: cur0,
            draft_off: 1,
        },
    );
    match tree {
        None => {
            append_chain_next_ids(&mut gb, tok0, zero, &m, kd, "mtp_first_ids");
        }
        Some(shape) => {
            let n = shape.n_nodes() as u64;
            let key = super::tree_step::shape_key(shape);
            let depth = gb.get_or_add_persistent_tensor(
                format!("trt_{key}_depth"),
                SynDataType::Int32,
                &[n],
            );
            let rank = gb.get_or_add_persistent_tensor(
                format!("trt_{key}_rank"),
                SynDataType::Int32,
                &[n],
            );
            append_tree_next_ids(
                &mut gb,
                tok0,
                zero,
                zero,
                &m,
                1,
                kd,
                depth,
                rank,
                n,
                nvd,
                "mtp_first_ids",
            );
        }
    }
    gb
}

// ============================================================================
// Loaded drafter
// ============================================================================

/// The MTP head of a loaded engine: its own recipes and bindings.
pub struct Mtp {
    /// Drafts per step (the chain width, or the tree's spine).
    pub kd: u64,
    /// Rows of the draft readout (0 = the full `lm_head`).
    pub nvd: u64,
    tree: Option<TreeShape>,
    /// Ids of the first step (`kd + 1`, or the tree's nodes).
    n_first: usize,
    ctx: HashMap<(u64, u64), CompiledRecipe>,
    first: HashMap<u64, CompiledRecipe>,
    pool: BTreeMap<String, u64>,
}

impl Mtp {
    /// Checks that the target can carry the MTP passes (the checkpoint itself is checked when
    /// its tensors are packed).
    pub fn new(
        spec: &ModelSpec,
        kd: u64,
        nvd: u64,
        tree: Option<&TreeShape>,
    ) -> Result<Self, String> {
        if !nvd.is_multiple_of(128) || nvd > spec.vocab {
            return Err(format!(
                "--draft-vocab {nvd}: a multiple of 128, at most the vocabulary ({})",
                spec.vocab
            ));
        }
        let moe_ok =
            spec.family == ModelFamily::Qwen35Moe && crate::engine::moe::supported_shape(spec);
        if !(spec.family == ModelFamily::Qwen35Dense || moe_ok) || spec.head_dim != 256 {
            return Err(format!(
                "--spec mtp: the MTP drafter needs a hybrid Qwen3.5 / 3.8 target (dense or mixture-of-experts) with 256-channel attention heads (family {:?}, head_dim {})",
                spec.family, spec.head_dim
            ));
        }
        if spec.n_linear_layers() == 0 {
            return Err("--spec mtp: target without linear-attention layers".into());
        }
        if kd == 0 || kd > MTP_MAX_DRAFTS {
            return Err(format!(
                "--spec mtp: 1..={MTP_MAX_DRAFTS} drafts per step (got {kd})"
            ));
        }
        Ok(Self {
            kd,
            nvd,
            tree: tree.cloned(),
            n_first: tree.map_or(kd as usize + 1, |s| s.n_nodes()),
            ctx: HashMap::new(),
            first: HashMap::new(),
            pool: BTreeMap::new(),
        })
    }

    /// Graphs whose persistent tensors size this module's device plan.
    pub fn sizing_graphs(
        &self,
        spec: &ModelSpec,
        lmax: u64,
        max_chunks: u64,
        prefill_tile: u64,
        elf_hashes: &[(String, String)],
    ) -> Vec<GraphBuilder> {
        vec![
            build_ctx_graph(spec, prefill_tile, lmax, max_chunks, elf_hashes),
            build_first_graph(
                spec,
                lmax,
                max_chunks,
                self.kd,
                self.nvd,
                self.tree.as_ref(),
                elf_hashes,
            ),
        ]
    }

    /// The head's own recipes of KV chunk count `nch` that a request compiles on first use
    /// (device-less pre-compilation): the prompt-tile rows and the first block.
    pub fn lazy_graphs(
        &self,
        spec: &ModelSpec,
        lmax: u64,
        prefill_tile: u64,
        nch: u64,
        elf_hashes: &[(String, String)],
    ) -> Vec<GraphBuilder> {
        vec![
            build_ctx_graph(spec, prefill_tile, lmax, nch, elf_hashes),
            build_first_graph(
                spec,
                lmax,
                nch,
                self.kd,
                self.nvd,
                self.tree.as_ref(),
                elf_hashes,
            ),
        ]
    }

    /// Compiles the recipes whose workspace must be known before the allocation seal.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_eager(
        &mut self,
        spec: &ModelSpec,
        lmax: u64,
        prefill_tile: u64,
        elf_hashes: &[(String, String)],
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
        verbose: bool,
    ) -> Result<u64, String> {
        let first = build_first_graph(
            spec,
            lmax,
            1,
            self.kd,
            self.nvd,
            self.tree.as_ref(),
            elf_hashes,
        )
        .compile(Arc::clone(api), cache, driver_version)
        .map_err(|e| format!("compile mtp first block: {e}"))?;
        let ctx = build_ctx_graph(spec, prefill_tile, lmax, 1, elf_hashes)
            .compile(Arc::clone(api), cache, driver_version)
            .map_err(|e| format!("compile mtp ctx T={prefill_tile}: {e}"))?;
        if verbose {
            for r in [&first, &ctx] {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                    r.name,
                    r.from_cache,
                    r.ws_size as f64 / (1024.0 * 1024.0)
                );
            }
        }
        let ws = first.ws_size.max(ctx.ws_size);
        self.first.insert(1, first);
        self.ctx.insert((prefill_tile, 1), ctx);
        Ok(ws)
    }

    /// Device bytes of one MTP weight / table.
    fn pack<'a>(
        name: &str,
        nvd: u64,
        spec: &ModelSpec,
        packer: &WeightPacker<'a>,
        layer: &WeightPacker<'a>,
    ) -> Result<Cow<'a, [u8]>, String> {
        let store = packer.store();
        let raw = |tensor: &str, shape: &[u64]| -> Result<&'a [u8], String> {
            let meta = store
                .get(tensor)
                .map_err(|e| format!("the checkpoint has no MTP head tensor '{tensor}': {e}"))?;
            if meta.dtype != "BF16" || meta.shape != shape {
                return Err(format!(
                    "MTP tensor '{tensor}': dtype {} shape {:?}, expected BF16 {shape:?}",
                    meta.dtype, meta.shape
                ));
            }
            store
                .tensor_bytes(tensor)
                .map_err(|e| format!("read '{tensor}': {e}"))?
                .ok_or_else(|| format!("MTP tensor '{tensor}' is not mapped"))
        };
        let hid = spec.hidden;
        match name {
            "mtp_w_e" => Ok(Cow::Borrowed(raw(
                "mtp.pre_fc_norm_embedding.weight",
                &[hid],
            )?)),
            "mtp_w_h" => Ok(Cow::Borrowed(raw("mtp.pre_fc_norm_hidden.weight", &[hid])?)),
            "mtp_w_norm" => Ok(Cow::Borrowed(raw("mtp.norm.weight", &[hid])?)),
            "mtp_W_fc_e" | "mtp_W_fc_h" => {
                // fc.weight [HID, 2 HID] = [W_e | W_h] per output row
                let src = raw("mtp.fc.weight", &[hid, 2 * hid])?;
                let half = (hid as usize) * 2;
                let off = if name == "mtp_W_fc_e" { 0 } else { half };
                let mut dst = Vec::with_capacity((hid as usize) * half);
                for row in src.chunks_exact(2 * half) {
                    dst.extend_from_slice(&row[off..off + half]);
                }
                Ok(Cow::Owned(dst))
            }
            "mtp_W_lmd" => {
                // the lm_head rows of the draft vocabulary, in table order
                let full = packer.pack_weight(lm_weight(spec))?;
                let row = (hid * 2) as usize;
                let mut dst = Vec::with_capacity(nvd as usize * row);
                for id in draft_vocab_ids(spec, nvd) {
                    let o = id as usize * row;
                    dst.extend_from_slice(&full[o..o + row]);
                }
                Ok(Cow::Owned(dst))
            }
            "mtp_dv_ids" => Ok(Cow::Owned(
                draft_vocab_ids(spec, nvd)
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect(),
            )),
            "mtp_fpos" => Ok(Cow::Owned(
                (0..NF as i32).flat_map(|i| i.to_le_bytes()).collect(),
            )),
            "mtp_fz_k" | "mtp_fz_v" => Ok(Cow::Owned(vec![
                0u8;
                (spec.head_dim * NF * spec.n_kv_heads * 2)
                    as usize
            ])),
            _ => {
                let what = name
                    .strip_prefix("M_")
                    .ok_or_else(|| format!("unknown MTP weight '{name}'"))?;
                layer.pack_weight(&format!("L0_{what}"))
            }
        }
    }

    /// After the seal: uploads the MTP head and binds its eager recipes.
    /// `pool` = every address an MTP recipe may bind (target weights + this module's buffers).
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &mut self,
        dev: &HpuDevice,
        spec: &ModelSpec,
        packer: &WeightPacker<'_>,
        stage: &mut PinnedBuf,
        weight_specs: &BTreeMap<String, u64>,
        addr: &BTreeMap<String, u64>,
        pool: BTreeMap<String, u64>,
        verbose: bool,
    ) -> Result<(), String> {
        let t0 = Instant::now();
        // the MTP layer is one full-attention layer stored as `mtp.layers.0.*`
        let mut lspec = spec.clone();
        lspec.text_prefix = "mtp.".into();
        lspec.layer_types = vec!["full_attention".into()];
        // The head runs in BF16 in every graph (its matrices are not `L<l>_*` names, so an FP8
        // trunk leaves them alone): a block-FP8 checkpoint's head matrices are dequantised here.
        lspec.fp8_weights = false;
        let ext =
            crate::engine::fp8_pack::ExtPacker::new(packer.store(), &lspec, BTreeMap::new(), None);
        let mut layer = WeightPacker::new(packer.store(), &lspec);
        layer.ext = Some(&ext);
        let mut uploaded = 0u64;
        for (name, &bytes) in weight_specs {
            if !is_mtp_weight(name) {
                continue;
            }
            let data = Self::pack(name, self.nvd, spec, packer, &layer)?;
            if data.len() as u64 != bytes {
                return Err(format!(
                    "MTP weight '{name}' packed size {} != graph tensor size {bytes}",
                    data.len()
                ));
            }
            dev.upload_h2d_chunked(&data, addr[name], stage)
                .map_err(|e| format!("upload MTP weight '{name}': {e}"))?;
            uploaded += bytes;
        }
        if verbose {
            eprintln!(
                "[gaudium] mtp: uploaded {:.2} GiB MTP head (tensors mtp.* of the target checkpoint) in {:.2}s",
                uploaded as f64 / (1u64 << 30) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        self.pool = pool;
        for r in self.first.values_mut().chain(self.ctx.values_mut()) {
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind {} ws: {e}", r.name))?;
        }
        Ok(())
    }
}

/// After the trunk layers of a prefill tile: the MTP layer's K/V rows of the tile.
pub(super) fn prefill_tile(eng: &mut Qwen3Engine, pos: u64, tile_len: u64) -> Result<(), String> {
    let nch = (pos + tile_len).div_ceil(KV_CHUNK).clamp(1, eng.max_chunks);
    let x_final = if (eng.spec.n_layers as usize).is_multiple_of(2) {
        eng.pf_x0
    } else {
        eng.pf_x1
    };
    let Qwen3Engine {
        specdec,
        dev,
        spec,
        cache,
        driver_version,
        stage_io,
        pf_cursor,
        ..
    } = eng;
    let rt: &mut SpecRuntime = specdec.as_mut().expect("spec runtime");
    let SpecRuntime {
        mtp,
        prompt,
        elf_hashes,
        lmax,
        ..
    } = rt;
    let m = mtp.as_mut().expect("mtp drafter");
    if let std::collections::hash_map::Entry::Vacant(slot) = m.ctx.entry((tile_len, nch)) {
        let mut r = build_ctx_graph(spec, tile_len, *lmax, nch, elf_hashes)
            .compile(Arc::clone(dev.api()), cache, driver_version)
            .map_err(|e| format!("compile mtp ctx T={tile_len} C={nch}: {e}"))?;
        r.bind_shared_workspace(dev)
            .map_err(|e| format!("bind mtp ctx T={tile_len} C={nch} ws: {e}"))?;
        slot.insert(r);
    }
    // the rows' next tokens; the last prompt position has none yet (the first block redoes it)
    let n = tile_len as usize;
    let buf = stage_io.as_mut_slice();
    for i in 0..n {
        let id = prompt.get(pos as usize + i + 1).copied().unwrap_or(0);
        buf[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
    }
    dev.copy_h2d_pinned_sync(stage_io, n * 4, m.pool["mtp_ctx_ids"])
        .map_err(|e| format!("mtp ctx ids H2D: {e}"))?;
    let r = &m.ctx[&(tile_len, nch)];
    let mut pool = m.pool.clone();
    pool.insert("mtp_ctx_x".into(), x_final);
    pool.insert("ctx_cursor".into(), *pf_cursor);
    let exact = select_exact_bindings(r, &pool)?;
    r.launch_checked(dev, &exact)
        .map_err(|e| format!("launch mtp ctx T={tile_len} C={nch}: {e}"))
}

/// The first step's ids: `[first_tok, drafts ..]` (a tree: its nodes), drafted from the last
/// prompt position `prompt_len - 1`. Returns `blk` ids (padded).
pub(super) fn first_block(
    eng: &mut Qwen3Engine,
    first_tok: i32,
    prompt_len: u64,
    blk: usize,
) -> Result<Vec<i32>, String> {
    let mut ids = vec![first_tok; blk.max(1)];
    for v in ids.iter_mut().skip(1) {
        *v = 0;
    }
    let Some(&h0) = eng.head_scratch.get("h_final") else {
        return Ok(ids);
    };
    if prompt_len < 2 {
        // a one-token prompt ran the step recipe, not the prefill head: no hidden row to draft from
        return Ok(ids);
    }
    let nch = prompt_len.div_ceil(KV_CHUNK).clamp(1, eng.max_chunks);
    let Qwen3Engine {
        specdec,
        dev,
        spec,
        cache,
        driver_version,
        stage_io,
        ..
    } = eng;
    let rt: &mut SpecRuntime = specdec.as_mut().expect("spec runtime");
    let SpecRuntime {
        mtp,
        elf_hashes,
        lmax,
        ..
    } = rt;
    let m = mtp.as_mut().expect("mtp drafter");
    if let std::collections::hash_map::Entry::Vacant(slot) = m.first.entry(nch) {
        let mut r = build_first_graph(spec, *lmax, nch, m.kd, m.nvd, m.tree.as_ref(), elf_hashes)
            .compile(Arc::clone(dev.api()), cache, driver_version)
            .map_err(|e| format!("compile mtp first block C={nch}: {e}"))?;
        r.bind_shared_workspace(dev)
            .map_err(|e| format!("bind mtp first block C={nch} ws: {e}"))?;
        slot.insert(r);
    }
    let buf = stage_io.as_mut_slice();
    buf[0..4].copy_from_slice(&first_tok.to_le_bytes());
    buf[64..68].copy_from_slice(&((prompt_len - 1) as i32).to_le_bytes());
    dev.copy_h2d_pinned_batch_sync(
        stage_io,
        &[(0, 4, m.pool["mtp_tok0"]), (64, 4, m.pool["mtp_cur0"])],
    )
    .map_err(|e| format!("mtp first block H2D: {e}"))?;
    let r = &m.first[&nch];
    let mut pool = m.pool.clone();
    pool.insert("mtp_h0".into(), h0);
    let exact = select_exact_bindings(r, &pool)?;
    r.launch_checked(dev, &exact)
        .map_err(|e| format!("launch mtp first block: {e}"))?;
    unsafe { dev.sync_stream(dev.compute_stream()) }
        .map_err(|e| format!("sync mtp first block: {e}"))?;
    let n = m.n_first.min(ids.len());
    dev.copy_d2h_pinned_sync(m.pool["mtp_first_ids"], stage_io, n * 4)
        .map_err(|e| format!("mtp first block D2H: {e}"))?;
    for (dst, &c) in ids
        .iter_mut()
        .zip(stage_io.as_slice()[..n * 4].as_chunks::<4>().0)
    {
        *dst = i32::from_le_bytes(c);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_27b() -> ModelSpec {
        let cfg = r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "vocab_size": 248320, "rms_norm_eps": 1e-6, "full_attention_interval": 4,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
          }
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    fn dims(gb: &GraphBuilder, n: &str) -> Option<Vec<u64>> {
        gb.persistent_tensor_specs()
            .iter()
            .find(|t| t.0 == n)
            .map(|t| t.2.clone())
    }

    #[test]
    fn names_are_classified_for_the_device_plan() {
        for w in [
            "M_W_qg",
            "M_w_in",
            "M_qn_w",
            "mtp_W_fc_e",
            "mtp_fpos",
            "mtp_fz_v",
        ] {
            assert!(is_mtp_weight(w), "{w}");
        }
        for a in ["M_qg", "M_out", "M_kcache", "d2_kf", "mtp_c_ids", "M_hfin"] {
            assert!(!is_mtp_weight(a), "{a}");
        }
        assert!(is_mtp_kv("M_kcache") && is_mtp_kv("M_vcache") && !is_mtp_kv("M_qb"));
        let mut w = BTreeMap::new();
        let mut kv = BTreeMap::new();
        let mut act: BTreeMap<String, u64> = [
            ("M_W_k", 10u64),
            ("M_kcache", 20),
            ("M_qg", 30),
            ("mtp_ctx_x", 40),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        reclassify(&mut w, &mut kv, &mut act);
        assert_eq!(w.get("M_W_k"), Some(&10));
        assert_eq!(kv.get("M_kcache"), Some(&20));
        assert_eq!(act.keys().cloned().collect::<Vec<_>>(), vec!["M_qg"]);
    }

    #[test]
    fn chain_step_keeps_the_verify_and_adds_catch_up_and_drafts() {
        let spec = target_27b();
        let gb = build_chain_step_graph(&spec, 8192, 2, 5, 5, 0, &[]);
        // the verify part is the chain graph: same tensors, same dims
        assert_eq!(dims(&gb, "tokens"), Some(vec![1, 5]));
        assert_eq!(dims(&gb, "L0_S_ckpt"), Some(vec![64, 256, 48, 5]));
        // catch-up over the 5 verified rows, in place in the MTP cache
        assert_eq!(dims(&gb, "M_kcache"), Some(vec![256, 4096, 8]));
        assert_eq!(dims(&gb, "M_tok"), Some(vec![1, 5]));
        // drafts 2..4 run on the fresh window; draft 5 does not exist (K = 4)
        assert_eq!(dims(&gb, "d2_kf"), Some(vec![256, 256, 4]));
        assert_eq!(dims(&gb, "d4_tok"), Some(vec![1, 1]));
        assert!(dims(&gb, "d5_tok").is_none());
        assert_eq!(dims(&gb, "next_ids"), Some(vec![5]));
        assert_eq!(dims(&gb, "mtp_drafts"), Some(vec![1, 3]));
        // the 1-row fallback hands its token on and drafts nothing
        let g1 = build_chain_step_graph(&spec, 8192, 2, 1, 5, 0, &[]);
        assert_eq!(dims(&g1, "next_ids"), Some(vec![1]));
        assert!(dims(&g1, "M_tok").is_none());
        // K = 1 and K = 2 need no concat
        let g2 = build_chain_step_graph(&spec, 4096, 1, 2, 5, 0, &[]);
        assert!(dims(&g2, "mtp_drafts").is_none() && dims(&g2, "d2_tok").is_none());
        assert_eq!(dims(&g2, "next_ids"), Some(vec![2]));
    }

    #[test]
    fn context_and_first_block_share_the_cache_and_the_head_weights() {
        let spec = target_27b();
        let ctx = build_ctx_graph(&spec, 256, 16384, 4, &[]);
        assert_eq!(dims(&ctx, "M_kcache"), Some(vec![256, 4096, 16]));
        // one q block whatever the chunk count
        assert_eq!(dims(&ctx, "mc_qb"), Some(vec![256, 6 * 256, 4]));
        let first = build_first_graph(&spec, 16384, 1, 4, 0, None, &[]);
        assert_eq!(dims(&first, "mtp_first_ids"), Some(vec![5]));
        assert_eq!(dims(&first, "M_tok"), Some(vec![1, 1]));
        assert_eq!(dims(&first, "mtp_h0"), Some(vec![5120, 1]));
        for g in [&ctx, &first] {
            assert_eq!(dims(g, "M_W_qg"), Some(vec![5120, 12288]));
            assert_eq!(dims(g, "mtp_W_fc_h"), Some(vec![5120, 5120]));
        }
    }

    #[test]
    fn refuses_targets_without_an_mtp_layer_shape() {
        let cfg = r#"{"model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288,
          "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
          "head_dim": 128, "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
          "tie_word_embeddings": false}"#;
        let dense = ModelSpec::from_config_json(cfg).unwrap();
        assert!(Mtp::new(&dense, 4, 0, None).is_err());
        let spec = target_27b();
        assert!(Mtp::new(&spec, 4, 0, None).is_ok());
        assert!(Mtp::new(&spec, 0, 0, None).is_err());
        assert!(Mtp::new(&spec, 4, 100, None).is_err());
        // a draft vocabulary shrinks the MTP readouts only: the verify keeps the full head
        let g = build_chain_step_graph(&spec, 4096, 1, 5, 5, 98304, &[]);
        assert_eq!(dims(&g, "mtp_W_lmd"), Some(vec![5120, 98304]));
        assert_eq!(dims(&g, "M_logits"), Some(vec![98304, 5]));
        assert_eq!(dims(&g, "d2_logits"), Some(vec![98304, 1]));
        assert_eq!(dims(&g, "mtp_dv_ids"), Some(vec![1, 98304]));
        assert_eq!(dims(&g, "M_tok_li"), Some(vec![1, 5]));
        assert_eq!(dims(&g, "M_tok"), Some(vec![1, 5]));
        // the subset: the lowest ids, then the control-token block at the lowest EOS id
        let mut s2 = spec.clone();
        s2.eos_token_ids = vec![248046, 248044];
        let ids = draft_vocab_ids(&s2, 98304);
        assert_eq!(ids.len(), 98304);
        assert_eq!(ids[98304 - 129], 98304 - 129);
        assert_eq!(ids[98304 - 128], 248044);
        assert_eq!(ids[98304 - 126], 248046);
        s2.eos_token_ids = vec![7];
        assert_eq!(draft_vocab_ids(&s2, 256), (0..256).collect::<Vec<i32>>());
        assert_eq!(dims(&g, "logits"), Some(vec![248320, 5]));
        assert!(Mtp::new(&spec, MTP_MAX_DRAFTS + 1, 0, None).is_err());
    }
}
