//! Tree verify: a speculative step whose rows are the nodes of a draft *tree* (a spine plus
//! alternative leaves) instead of one chain.
//!
//! Row 0 is the root (the last accepted token), rows `1..=D` the spine (the drafter's first
//! choices), the remaining rows leaves that hang off a spine node (other candidates for the same
//! position). One launch `g_spec_tr*_<shape>_C<nch>` does:
//! - **verify** of all `N` rows: node `t` is written to KV *slot* `cursor + t` but rotated at RoPE
//!   *position* `cursor + depth[t]` (slot and position are separate on purpose: after an image
//!   they differ at sequence level too); attention rows see the history and their own ancestors
//!   only (a 32-bit ancestor mask per row); Gated DeltaNet rows resume from their parent's state
//!   and leave one state checkpoint per node;
//! - **accept** (`gaudium_accept_tree`): the longest root path whose every node equals the
//!   target's argmax at its parent -> `a_out` (row of the last accepted node = checkpoint slot
//!   to resume from), `cursor_out = cursor + d + 1`, the emitted tokens, and the *path move*:
//!   when the path ends in a leaf, that leaf's KV rows sit at its tree slot, not at the slot of
//!   its position;
//! - the next step's attention prep applies the pending move (`tfix_in`, alternating with
//!   `tfix_out` by step parity) before it writes its own rows, so the cache is again
//!   `slot == position` below the cursor. State needs no move: the checkpoint slot is `a`.
//!
//! Exactness: every emitted token is a verify argmax, so the stream follows the target for any
//! drafts. Spine rows run the same arithmetic as chain rows. A leaf row reads its own key from a
//! different column of the attention than a chain row would, which changes the summation order
//! of its softmax and of its P.V product: its outputs can differ from the chain's in the last
//! bit, and at an exact BF16 tie of the two best logits that can pick the other token. Tree
//! verify is therefore an opt-in mode; a run given a reference stream reports the first
//! divergence with the logit gap (`fork` line).
//!
//! Shapes are static caterpillars (`crate::tree`): leaves have no children, because the DeltaNet
//! kernels keep one live state per head and evaluate a leaf from its parent's state without
//! writing it back.

use super::super::*;
use super::{
    ensure_step, mtp, read_i32s, write_i32s, OracleRequest, SpecDecodeOut, SpecMethod, SpecStats,
};
use crate::fp8::bf16_to_f32;
use crate::tree::{make_shape, walk_tree, TreeShape};

/// Entries of the emitted record: tokens, `-1` .., `[EMN - 2]` = accepted depth, `[EMN - 1]` = `a`.
pub const EMN: u64 = 32;
/// Path moves a step may leave for the next one (a caterpillar path ends in at most one leaf).
pub const FIXN: u64 = 1;
/// Most nodes of a shape (the emitted record must hold `depth + 1` tokens before its tail).
pub const TREE_MAX_NODES: u64 = 24;

/// Kernel ELFs of the tree builds for `spec` (beyond the trunk's own).
pub fn tree_elfs(spec: &ModelSpec) -> Vec<&'static str> {
    let mut v = vec!["GAUDIUM_ACCEPTTR_ELF", "GAUDIUM_ASOFTCT_ELF"];
    v.push(if spec.head_dim == 128 {
        "GAUDIUM_A128PREPT_G8Q4_ELF"
    } else {
        "GAUDIUM_APREPT_ELF"
    });
    if spec.n_linear_layers() > 0 {
        v.push("GAUDIUM_GDNT_ELF");
        v.push("GAUDIUM_PREPT_ELF");
    }
    if crate::verified::kv_is_fp8(crate::engine::qwen3::kv8::trunk_dtype(&spec)) {
        v.extend_from_slice(crate::engine::qwen3::kv8::KV8_TREE_ELFS); // [kv8]
    }
    v
}

/// Recipe / tensor name key of a shape (`4:1-2@2` -> `4_1_2_2`).
pub fn shape_key(shape: &TreeShape) -> String {
    shape
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Tree state of a loaded engine: the verify shape and the one-row shape used when the full
/// shape no longer fits in the context.
pub struct TreeRt {
    pub shape: TreeShape,
    pub root: TreeShape,
    /// The MTP head drafts the tree on the card (else: host oracle).
    pub mtp: bool,
    /// Reference streams (one per request): the first divergence from them is reported.
    pub refs: Vec<Vec<i32>>,
}

impl TreeRt {
    pub fn new(method: &SpecMethod, spec: &ModelSpec) -> Result<Option<Self>, String> {
        let (shape, mtp, refs) = match method {
            SpecMethod::OracleTree { shape, requests } => (
                shape,
                false,
                requests.iter().map(|r| r.stream.clone()).collect(),
            ),
            SpecMethod::Mtp {
                tree: Some(shape),
                reference,
                ..
            } => (shape, true, reference.clone()),
            _ => return Ok(None),
        };
        if shape.dynamic {
            return Err(format!(
                "--tree {}: device-planned shapes are not built yet (static shapes only)",
                shape.name
            ));
        }
        shape
            .validate_caterpillar(TREE_MAX_NODES, mtp::MTP_MAX_DRAFTS, 4)
            .map_err(|e| format!("--tree {}: {e}", shape.name))?;
        if shape.spine() == 0 {
            return Err(format!("--tree {}: the spine needs a node", shape.name));
        }
        if mtp && shape.nodes.iter().any(|n| n.rank > 3) {
            return Err(format!(
                "--tree {}: the MTP readout keeps 4 candidates per depth (ranks 1..4)",
                shape.name
            ));
        }
        if spec.head_dim == 128 && (spec.n_kv_heads != 8 || spec.n_heads != 32) {
            return Err("--tree: no tree attention build for this head shape".into());
        }
        Ok(Some(Self {
            shape: shape.clone(),
            root: make_shape(0, &[], "0"),
            mtp,
            refs,
        }))
    }

    /// The shape a step of `t` rows verifies.
    pub fn shape_of(&self, t: u64) -> &TreeShape {
        if t == self.shape.n_nodes() as u64 {
            &self.shape
        } else {
            &self.root
        }
    }

    /// Device tables `(name, values)` of both shapes.
    pub fn tables(&self) -> Vec<(String, Vec<i32>)> {
        let mut out = Vec::new();
        for s in [&self.shape, &self.root] {
            let key = shape_key(s);
            out.push((format!("trt_{key}_parent"), s.parent_tab()));
            out.push((format!("trt_{key}_depth"), s.depth_tab()));
            out.push((format!("trt_{key}_rank"), s.rank_tab()));
            out.push((
                format!("trt_{key}_anc"),
                s.anc_tab().iter().map(|&m| m as i32).collect(),
            ));
        }
        out
    }
}

/// Moves the shape tables into the weight plan and drops the parity-bound path-move tensors.
pub fn reclassify(weights: &mut BTreeMap<String, u64>, act: &mut BTreeMap<String, u64>) {
    let names: Vec<String> = act.keys().cloned().collect();
    for name in names {
        if name.starts_with("trt_") {
            let b = act.remove(&name).unwrap();
            let e = weights.entry(name).or_insert(0);
            *e = (*e).max(b);
        } else if name == "tfix_in" || name == "tfix_out" {
            act.remove(&name);
        }
    }
}

// ============================================================================
// Graph builders
// ============================================================================

/// Shape tables of a tree step graph.
struct Tabs {
    parent: TensorId,
    depth: TensorId,
    rank: TensorId,
    anc: TensorId,
}

/// One full-attention layer over the `t` nodes of a tree: slots `cursor + row`, positions
/// `cursor + depth[row]`, the pending path move first, ancestor-masked softmax.
#[allow(clippy::too_many_arguments)]
fn append_attn_layer_tree(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    cursor: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    tabs: &Tabs,
    fix: TensorId,
    lpfx: &str,
    t: u64,
    nch: u64,
    wnext_norm: Option<(&str, &str)>,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let nh = spec.n_heads;
    let nkv = spec.n_kv_heads;
    let dh = spec.head_dim;
    let r = (nh / nkv) * t;
    let b = nkv * nch;
    let (g_aprep, g_acomb, g_apost) = if dh == 128 {
        (
            "gaudium_attn128_prep",
            "gaudium_attn128_combine",
            "gaudium_attn128_post",
        )
    } else {
        (
            "gaudium_attn_prep",
            "gaudium_attn_combine",
            "gaudium_attn_post_f32",
        )
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
        spec.qg_dim(),
        t,
        &format!("{lpfx}qg"),
    );
    let k = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_k"),
        hid,
        spec.kv_dim(),
        t,
        &format!("{lpfx}k"),
    );
    let v = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_v"),
        hid,
        spec.kv_dim(),
        t,
        &format!("{lpfx}v"),
    );
    let qn_w = gb.get_or_add_persistent_tensor(format!("{lpfx}qn_w"), SynDataType::Float, &[dh]);
    let kn_w = gb.get_or_add_persistent_tensor(format!("{lpfx}kn_w"), SynDataType::Float, &[dh]);
    // [kv8] the cache dtype of this layer (q block, caches and P) and its scale table
    let kv = crate::engine::qwen3::kv8::AttnKv::of(gb, spec, lpfx);
    let qb = gb.add_persistent_tensor(format!("{lpfx}qb"), kv.dt, &[dh, r, b]);
    let kcache =
        gb.add_persistent_tensor(format!("{lpfx}kcache"), kv.dt, &[dh, KV_CHUNK, b]);
    let vcache =
        gb.add_persistent_tensor(format!("{lpfx}vcache"), kv.dt, &[dh, KV_CHUNK, b]);
    gb.add_node(
        g_aprep,
        format!("{lpfx}attn_prep"),
        &kv.inputs(&[
            qg, k, v, qn_w, kn_w, rope_c, rope_s, cursor, tabs.depth, fix,
        ]),
        &[qb, kcache, vcache],
        &u32_params(&[0u32]),
    );
    let scores = gb.add_persistent_tensor(
        format!("{lpfx}scores"),
        SynDataType::Float,
        &[KV_CHUNK, r, b],
    );
    gb.op_batch_gemm(format!("{lpfx}qk"), qb, kcache, scores, false, true);
    let pm = gb.add_persistent_tensor(format!("{lpfx}P"), kv.dt, &[KV_CHUNK, r, b]);
    let stats =
        gb.add_persistent_tensor(format!("{lpfx}stats"), SynDataType::Float, &[64, 2, r, b]);
    gb.add_node(
        "gaudium_attn_softmax_chunk",
        format!("{lpfx}softmax"),
        &kv.inputs(&[scores, cursor, tabs.anc]),
        &[pm, stats],
        &u32_params(&[t as u32, nch as u32, 0u32, 0u32]),
    );
    let ao_part =
        gb.add_persistent_tensor(format!("{lpfx}ao_part"), SynDataType::Float, &[dh, r, b]);
    gb.op_batch_gemm(format!("{lpfx}pv"), pm, vcache, ao_part, false, false);
    let ao = gb.add_persistent_tensor(format!("{lpfx}ao"), SynDataType::Float, &[dh, r, nkv]);
    gb.add_node(
        g_acomb,
        format!("{lpfx}combine"),
        &[ao_part, stats],
        &[ao],
        &u32_params(&[nch as u32]),
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
    append_post_and_mlp(gb, spec, x, attn, lpfx, t, true, wnext_norm)
}

/// One Gated DeltaNet layer over the `t` nodes of a tree: the conv and recurrent states of a
/// node come from its parent, one checkpoint per node (`S_ckpt` / `cs_ckpt`), the step's start
/// state from slot `slot[0]` of the `ns`-slot rings.
#[allow(clippy::too_many_arguments)]
fn append_linear_layer_tree(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    slot: TensorId,
    parent: TensorId,
    lpfx: &str,
    t: u64,
    ns: u64,
    wnext_norm: Option<(&str, &str)>,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let hv = spec.gdn_v_heads;
    let dl = spec.gdn_v_dim;
    let qkv = spec.gdn_qkv_dim();
    let hvp = spec.gdn_hvp();
    let n = |s: &str| format!("{lpfx}{s}");

    let h = match h_pre {
        Some(h_in) => h_in,
        None => add_rmsnorm(gb, spec, x, &n("w_in"), &n("h"), t),
    };
    let qkvz = add_linear_bf16(gb, h, &n("W_qkvz"), hid, spec.gdn_qkvz_dim(), t, &n("qkvz"));
    let ba = add_linear_bf16(gb, h, &n("W_ba"), hid, spec.gdn_ba_dim(), t, &n("ba"));

    let conv_w = gb.get_or_add_persistent_tensor(n("conv_w"), SynDataType::Float, &[qkv, 4]);
    let conv_b = gb.get_or_add_persistent_tensor(n("conv_b"), SynDataType::Float, &[qkv]);
    let cs_in = gb.add_persistent_tensor(n("cs_in"), SynDataType::Float, &[qkv, 3, ns]);
    let cs_out = gb.add_persistent_tensor(n("cs_out"), SynDataType::Float, &[qkv, 3]);
    let cs_ckpt = gb.add_persistent_tensor(n("cs_ckpt"), SynDataType::Float, &[qkv, 3, t]);
    let q = gb.add_persistent_tensor(n("q"), SynDataType::Float, &[dl, hv, t]);
    let k = gb.add_persistent_tensor(n("k"), SynDataType::Float, &[dl, hv, t]);
    let v = gb.add_persistent_tensor(n("v"), SynDataType::Float, &[dl, hv, t]);
    gb.add_node(
        "gaudium_gdn_prep",
        n("gdn_prep"),
        &[qkvz, conv_w, cs_in, conv_b, slot, parent],
        &[q, k, v, cs_out, cs_ckpt],
        &u32_params(&[2 | 16]),
    );

    let a_log = gb.get_or_add_persistent_tensor(n("A_log"), SynDataType::Float, &[hvp]);
    let dt_bias = gb.get_or_add_persistent_tensor(n("dt_bias"), SynDataType::Float, &[hvp]);
    let eg = gb.add_persistent_tensor(n("eg"), SynDataType::Float, &[hvp, t]);
    let beta = gb.add_persistent_tensor(n("beta"), SynDataType::Float, &[hvp, t]);
    gb.add_node(
        "gaudium_gdn_gate",
        n("gdn_gate"),
        &[ba, a_log, dt_bias],
        &[eg, beta],
        &[],
    );

    let s_in = gb.add_persistent_tensor(n("S_in"), SynDataType::Float, &[64, 256, hv, ns]);
    let s_out = gb.add_persistent_tensor(n("S_out"), SynDataType::Float, &[64, 256, hv]);
    let s_ckpt = gb.add_persistent_tensor(n("S_ckpt"), SynDataType::Float, &[64, 256, hv, t]);
    let o = gb.add_persistent_tensor(n("o"), SynDataType::Float, &[dl, hv, t]);
    gb.add_node(
        "gaudium_gdn_f32",
        n("gdn"),
        &[s_in, q, k, v, eg, beta, slot, parent],
        &[o, s_out, s_ckpt],
        &u32_params(&[1 | 16]),
    );

    let norm_w = gb.get_or_add_persistent_tensor(n("norm_w"), SynDataType::Float, &[dl]);
    let core = gb.add_persistent_tensor(n("core"), SynDataType::Bf16, &[hv * dl, t]);
    gb.add_node(
        "gaudium_gdn_post",
        n("gdn_post"),
        &[o, qkvz, norm_w],
        &[core],
        &u32_params(&[0u32]),
    );
    let attn = add_linear_bf16(gb, core, &n("W_o"), hv * dl, hid, t, &n("attn"));
    append_post_and_mlp(gb, spec, x, attn, lpfx, t, true, wnext_norm)
}

/// Builds the tree step graph of `shape` over `nch` KV chunks. `mtp` (its draft-readout rows,
/// 0 = the full head) appends the MTP drafter: its catch-up on the accepted path, the spine
/// drafts, and the next tree's ids.
pub fn build_step_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    shape: &TreeShape,
    ns: u64,
    mtp: Option<u64>,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let t = shape.n_nodes() as u64;
    let spine = shape.spine() as u64;
    let key = shape_key(shape);
    let kind = if mtp.is_some() { "mtp" } else { "vf" };
    let mut gb = GraphBuilder::new(format!("g_spec_tr{kind}_{key}_C{nch}"));
    // Select the weight precision for the whole step graph (see `build_spec_step_graph`).
    gb.set_fp8_weights(spec.fp8_weights);
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let hid = spec.hidden;
    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, spec.vocab]);
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[t]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let slot = if spec.n_linear_layers() > 0 {
        Some(gb.add_persistent_tensor("a_in", SynDataType::Int32, &[1]))
    } else {
        None
    };
    let fix_in = gb.add_persistent_tensor("tfix_in", SynDataType::Int32, &[2 * FIXN]);
    let tab = |gb: &mut GraphBuilder, what: &str| {
        gb.get_or_add_persistent_tensor(format!("trt_{key}_{what}"), SynDataType::Int32, &[t])
    };
    let tabs = Tabs {
        parent: tab(&mut gb, "parent"),
        depth: tab(&mut gb, "depth"),
        rank: tab(&mut gb, "rank"),
        anc: tab(&mut gb, "anc"),
    };

    let mut x = gb.add_persistent_tensor("x0", SynDataType::Bf16, &[hid, t]);
    gb.op_gather_bf16("embed", table, ids, x, 1);
    let mut h_pre: Option<TensorId> = None;
    let n_layers = spec.n_layers as usize;
    for l in 0..n_layers {
        let lpfx = format!("L{l}_");
        let (next_w, next_h) = if l + 1 < n_layers {
            (format!("L{}_w_in", l + 1), format!("L{}_h", l + 1))
        } else {
            ("w_final".to_string(), "h_final".to_string())
        };
        let wnext = Some((next_w.as_str(), next_h.as_str()));
        let (out_x, next_hid) = if spec.is_full_attention_layer(l) {
            append_attn_layer_tree(
                &mut gb, spec, x, h_pre, cursor, rope_c, rope_s, &tabs, fix_in, &lpfx, t, nch,
                wnext,
            )
        } else {
            append_linear_layer_tree(
                &mut gb,
                spec,
                x,
                h_pre,
                slot.expect("a_in declared for hybrid models"),
                tabs.parent,
                &lpfx,
                t,
                ns,
                wnext,
            )
        };
        x = out_x;
        h_pre = next_hid;
    }
    let h_final = h_pre.expect("fused final norm produced by last layer");
    let lm_w = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    let logits = add_linear_bf16(&mut gb, h_final, lm_w, hid, spec.vocab, t, "logits");
    let tokens = gb.add_persistent_tensor("tokens", SynDataType::Int32, &[1, t]);
    gb.op_argmax_bf16("argmax", logits, tokens, 0);

    let tc = spine + 1;
    let a_out = gb.add_persistent_tensor("a_out", SynDataType::Int32, &[1]);
    let cursor_out = gb.add_persistent_tensor("cursor_out", SynDataType::Int32, &[1]);
    let emitted = gb.add_persistent_tensor("emitted", SynDataType::Int32, &[EMN]);
    let ncommit = gb.add_persistent_tensor("tr_ncommit", SynDataType::Int32, &[1]);
    let fix_out = gb.add_persistent_tensor("tfix_out", SynDataType::Int32, &[2 * FIXN]);
    let dep = gb.add_persistent_tensor("tr_dep", SynDataType::Int32, &[1]);
    let c_ids = gb.add_persistent_tensor("tr_c_ids", SynDataType::Int32, &[tc]);
    let pidx = gb.add_persistent_tensor("tr_pidx", SynDataType::Int32, &[tc]);
    gb.add_node(
        "gaudium_accept_tree",
        "accept",
        &[ids, tokens, cursor, tabs.parent, tabs.depth],
        &[
            a_out, cursor_out, emitted, ncommit, fix_out, dep, c_ids, pidx,
        ],
        &[],
    );
    let Some(nvd) = mtp else {
        return gb;
    };
    if spine == 0 {
        // the one-row step near the end of the context only hands its token on
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
    // MTP catch-up on the accepted path (rows gathered by the accept kernel), spine drafts, and
    // the next tree's ids from the drafts and their runner-up candidates
    let h_path = gb.add_persistent_tensor("mtp_hpath", SynDataType::Bf16, &[hid, tc]);
    gb.op_gather_bf16("mtp_hpath_g", h_final, pidx, h_path, 1);
    let m = mtp::append_mtp(
        &mut gb,
        spec,
        &mtp::MtpIn {
            t: tc,
            kd: spine,
            nch,
            lmax,
            nvd,
            ids: c_ids,
            h_rows: h_path,
            sel: dep,
            cursor,
            draft_cursor: cursor_out,
            draft_off: 0,
        },
    );
    mtp::append_tree_next_ids(
        &mut gb, tokens, a_out, dep, &m, tc, spine, tabs.depth, tabs.rank, t, nvd, "next_ids",
    );
    gb
}

// ============================================================================
// Host side
// ============================================================================

/// Oracle drafts on a tree for the step whose root is the last of the `emitted` tokens:
/// spine nodes carry the known continuation, except at corrupted indices, where the spine node
/// is wrong and the first leaf of that depth (if the shape has one) carries the true token.
pub fn oracle_tree_ids(
    o: &OracleRequest,
    shape: &TreeShape,
    emitted: &[i32],
    blk: usize,
) -> Vec<i32> {
    let n = emitted.len();
    let mut ids = vec![*emitted.last().expect("root token"); blk.max(shape.n_nodes())];
    for (t, node) in shape.nodes.iter().enumerate().skip(1) {
        let idx = n + node.depth - 1;
        let truth = o.stream.get(idx).copied().unwrap_or(0);
        let corrupt = o.corrupt.contains(&idx);
        let wrong = if truth == 0 { 1 } else { truth - 1 };
        ids[t] = match (node.rank, corrupt) {
            (0, false) => truth,
            (0, true) => wrong,
            (1, true) => truth,
            (r, _) => truth + 1 + r as i32,
        };
    }
    ids
}

/// `(top-1 id, its logit, runner-up id, its logit)` of one BF16 logits row (ties to the lower id).
fn top2(row: &[u8]) -> (usize, f32, usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = (0usize, f32::NEG_INFINITY);
    for (i, &c) in row.as_chunks::<2>().0.iter().enumerate() {
        let v = bf16_to_f32(u16::from_le_bytes(c));
        if v > best.1 {
            second = best;
            best = (i, v);
        } else if v > second.1 {
            second = (i, v);
        }
    }
    (best.0, best.1, second.0, second.1)
}

/// Tree decode phase: continues after the prefill produced `generated[0]` at position
/// `prompt_len`, with the DeltaNet state of the prompt in slot 0 of ring `gdn_parity`.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode(
    eng: &mut Qwen3Engine,
    prompt_len: usize,
    params: &Qwen3GenParams,
    budget: &mut SequenceBudget,
    is_eos: &impl Fn(i32) -> bool,
    is_cancelled: &impl Fn() -> bool,
    on_token: &mut impl FnMut(i32),
    generated: &mut Vec<i32>,
    gdn_parity: usize,
) -> Result<SpecDecodeOut, String> {
    let t_decode = Instant::now();
    let first_tok = *generated.last().expect("prefill produced the first token");
    let n0 = generated.len();
    let mut p = gdn_parity;
    let mut cursor = prompt_len as u64;
    let mut stats = SpecStats::default();
    let mut stopped_by_eos = false;
    let mut stopped_by_cancel = false;

    let (shape, root, blk, addr_tokens, addr_logits, oracle, reference, verbose) = {
        let rt = eng.specdec.as_mut().expect("spec runtime");
        let idx = rt.request_idx;
        rt.request_idx += 1;
        let oracle = match &rt.cfg.method {
            SpecMethod::OracleTree { requests, .. } => Some(
                requests
                    .get(idx)
                    .cloned()
                    .ok_or_else(|| format!("oracle drafter: no stream for request {idx}"))?,
            ),
            _ => None,
        };
        let tr = rt.tree.as_ref().expect("tree runtime");
        (
            tr.shape.clone(),
            tr.root.clone(),
            (rt.act_specs["sp_ids0"] / 4) as usize,
            rt.addr["tokens"],
            rt.addr["logits"],
            oracle,
            tr.refs.get(idx).cloned(),
            rt.verbose,
        )
    };
    let n_main = shape.n_nodes() as u64;
    let spine = shape.spine();
    stats.accept_hist = vec![0; spine + 1];
    stats.verified = vec![0; spine];
    let mut leaf_accepts = 0u64;
    let mut leaf_by_depth = vec![0u64; spine + 1];
    let mut fork_reported = false;

    let (a_addr, cur_addr, ids_addr, fix_addr) = {
        let rt = eng.specdec.as_ref().unwrap();
        (
            rt.addr[&format!("sp_a{p}")],
            rt.addr[&format!("sp_cur{p}")],
            rt.addr[&format!("sp_ids{p}")],
            rt.addr[&format!("sp_fix{p}")],
        )
    };
    let first_ids: Vec<i32> = match &oracle {
        Some(o) => oracle_tree_ids(o, &shape, generated, blk),
        None => mtp::first_block(eng, first_tok, cursor, blk)?,
    };
    let nofix = vec![-1i32; (2 * FIXN) as usize];
    write_i32s(
        eng,
        &[
            (a_addr, &[0]),
            (cur_addr, &[cursor as i32]),
            (ids_addr, &first_ids),
            (fix_addr, &nofix),
        ],
    )?;
    let mut cur_ids = first_ids;

    // The host reads device values of a step (ids, argmaxes, a logits row) only for the oracle
    // and for a reference check; those runs keep one launch in flight. Otherwise nothing the
    // host launches depends on a device value and the next step is enqueued while one runs.
    let check = oracle.is_some() || reference.is_some();
    let depth: usize = if check { 1 } else { 2 };
    let emit_addr = {
        let rt = eng.specdec.as_ref().unwrap();
        [rt.addr["sp_emit0"], rt.addr["sp_emit1"]]
    };
    let events = if depth > 1 {
        let mk = || {
            eng.dev
                .create_event(false)
                .map_err(|e| format!("create step event: {e}"))
        };
        Some([mk()?, mk()?])
    } else {
        None
    };
    struct Flight {
        rows: u64,
        k: u64,
        p: usize,
        launched: Instant,
    }
    let mut flights: std::collections::VecDeque<Flight> = std::collections::VecDeque::new();
    let mut last_done = Instant::now();
    let mut step_ms_sum = 0.0f64;
    // From here on a launch may be queued: every error path drains the device before returning.
    macro_rules! tryd {
        ($e:expr) => {
            match $e {
                Ok(v) => v,
                Err(e) => {
                    let _ = eng.dev.sync_all();
                    return Err(e);
                }
            }
        };
    }

    loop {
        // ---- enqueue ----
        while flights.len() < depth {
            if is_cancelled() {
                stopped_by_cancel = true;
                break;
            }
            let ahead: u64 = flights.iter().map(|f| f.k + 1).sum();
            if budget.remaining() <= ahead {
                break;
            }
            let cur_lo = cursor + flights.len() as u64;
            let cur_hi = cursor + ahead;
            let rows = if cur_hi + n_main <= eng.max_ctx {
                n_main
            } else if cur_hi < eng.max_ctx {
                1
            } else {
                break;
            };
            // every tree step binds the chunks a full shape reaches: the pending path move of
            // the previous step may read a slot up to `n_main - 1` rows past its cursor
            let chunks = |cur: u64| (cur + n_main).div_ceil(KV_CHUNK).clamp(1, eng.max_chunks);
            let nch = chunks(cur_hi);
            if chunks(cur_lo) != nch {
                break;
            }
            tryd!(ensure_step(eng, rows, nch));
            let t_launch = Instant::now();
            {
                let step = &eng.specdec.as_ref().unwrap().steps[&(rows, nch)];
                tryd!(step
                    .recipe
                    .launch_prepared(&eng.dev, &step.launches[p])
                    .map_err(|e| format!("launch tree step (cursor<={cur_hi}, rows={rows}): {e}")));
            }
            if let Some(ev) = &events {
                tryd!(ev[p]
                    .record_compute(&eng.dev)
                    .map_err(|e| format!("record step event: {e}")));
            }
            stats.host_ms[0] += t_launch.elapsed().as_secs_f64() * 1000.0;
            flights.push_back(Flight {
                rows,
                k: if rows == n_main { spine as u64 } else { 0 },
                p,
                launched: t_launch,
            });
            p ^= 1;
        }
        let Some(f) = flights.pop_front() else {
            break;
        };
        if stopped_by_cancel {
            flights.push_front(f);
            break;
        }

        // ---- collect the oldest step ----
        let t_wait = Instant::now();
        match &events {
            Some(ev) => tryd!(ev[f.p]
                .synchronize()
                .map_err(|e| format!("wait tree step (cursor={cursor}): {e}"))),
            None => tryd!(unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
                .map_err(|e| format!("sync tree step (cursor={cursor}): {e}"))),
        }
        stats.host_ms[1] += t_wait.elapsed().as_secs_f64() * 1000.0;
        let t_read = Instant::now();
        let em = tryd!(read_i32s(eng, emit_addr[f.p], EMN as usize));
        let d = em[EMN as usize - 2];
        let a = em[EMN as usize - 1];
        let n_emit = em[..EMN as usize - 2]
            .iter()
            .take_while(|&&v| v >= 0)
            .count();
        if d < 0 || a < 0 || a as u64 >= f.rows || n_emit != d as usize + 1 {
            let _ = eng.dev.sync_all();
            return Err(format!(
                "tree step at cursor {cursor}: inconsistent accept record (d {d}, a {a}, emitted {em:?})"
            ));
        }
        let (d, a) = (d as usize, a as usize);
        let step_shape = if f.rows == n_main { &shape } else { &root };
        if check {
            // the device accept must equal the host walk on the step's own ids and argmaxes
            let tokens = tryd!(read_i32s(eng, addr_tokens, f.rows as usize));
            if oracle.is_none() {
                let ids_addr = eng.specdec.as_ref().unwrap().addr[&format!("sp_ids{}", f.p)];
                cur_ids = tryd!(read_i32s(eng, ids_addr, f.rows as usize));
            }
            let path = walk_tree(step_shape, &cur_ids[..f.rows as usize], &tokens);
            let want_d = path.len() - 1;
            let want_a = *path.last().unwrap();
            let ok = want_d == d
                && want_a == a
                && em[d] == tokens[a]
                && (0..d).all(|i| em[i] == cur_ids[path[i + 1]]);
            if !ok {
                let _ = eng.dev.sync_all();
                return Err(format!(
                    "tree step at cursor {cursor}: device accept (d {d}, a {a}, emitted {:?}) disagrees with the host walk (path {path:?}; ids {:?}, tokens {tokens:?})",
                    &em[..n_emit],
                    &cur_ids[..f.rows as usize]
                ));
            }
            // first divergence from the reference stream, with the logit gap at the row that
            // produced the token (emitted token i is the argmax of path node i)
            if let (Some(rf), false) = (&reference, fork_reported) {
                let base = generated.len();
                for i in 0..n_emit {
                    let Some(&want) = rf.get(base + i) else { break };
                    if em[i] == want {
                        continue;
                    }
                    let row = path[i];
                    let vb = (eng.spec.vocab * 2) as usize;
                    tryd!(eng
                        .dev
                        .copy_d2h_pinned_sync(
                            addr_logits + (row as u64) * vb as u64,
                            &mut eng.stage_io,
                            vb
                        )
                        .map_err(|e| format!("fork logits D2H: {e}")));
                    let raw = &eng.stage_io.as_slice()[..vb];
                    let (t1, v1, t2, v2) = top2(raw);
                    let lw = raw
                        .as_chunks::<2>()
                        .0
                        .get(want as usize)
                        .map_or(f32::NAN, |&c| bf16_to_f32(u16::from_le_bytes(c)));
                    eprintln!(
                        "[gaudium] fork: generated index {} (position {}): emitted {} but the reference has {want}; verify row {row} (depth {}, {}): top-1 {t1} = {v1}, top-2 {t2} = {v2}, reference token logit {lw}, gap to the reference {}",
                        base + i,
                        cursor as usize + i + 1,
                        em[i],
                        step_shape.nodes[row].depth,
                        if row > step_shape.spine() { "leaf" } else { "spine" },
                        v1 - lw
                    );
                    fork_reported = true;
                    break;
                }
            }
        }
        stats.host_ms[2] += t_read.elapsed().as_secs_f64() * 1000.0;
        let now = Instant::now();
        let step_ms = now.duration_since(last_done.max(f.launched)).as_secs_f64() * 1000.0;
        last_done = now;
        step_ms_sum += step_ms;

        stats.launches += 1;
        if d < stats.accept_hist.len() {
            stats.accept_hist[d] += 1;
        }
        for i in 0..f.k as usize {
            stats.verified[i] += 1;
        }
        *stats.widths.entry(f.rows - 1).or_insert(0) += 1;
        if a > step_shape.spine() {
            leaf_accepts += 1;
            leaf_by_depth[d] += 1;
        }

        let mut step_emitted = 0u64;
        for &tok in &em[..n_emit] {
            if budget.remaining() == 0 {
                break;
            }
            budget.record_emitted(1);
            generated.push(tok);
            step_emitted += 1;
            on_token(tok);
            if params.stop_on_eos && is_eos(tok) {
                stopped_by_eos = true;
                break;
            }
        }
        stats.emitted += step_emitted;
        eng.live.step(step_emitted, f.k, d as u64);
        if stopped_by_eos || budget.remaining() == 0 {
            break;
        }

        cursor += n_emit as u64;
        if let Some(o) = &oracle {
            cur_ids = oracle_tree_ids(o, &shape, generated, blk);
            let ids_addr = eng.specdec.as_ref().unwrap().addr[&format!("sp_ids{p}")];
            tryd!(write_i32s(eng, &[(ids_addr, &cur_ids)]));
        }
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    if !flights.is_empty() {
        let t_drain = Instant::now();
        unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
            .map_err(|e| format!("drain tree steps: {e}"))?;
        stats.drain_ms = t_drain.elapsed().as_secs_f64() * 1000.0;
    }

    let decode_tokens = generated.len() - n0;
    if verbose && stats.launches > 0 {
        let n = stats.launches as f64;
        eprintln!(
            "[gaudium] tree {} ({} rows, spine {}): accepted-depth histogram {:?}, leaf accepts {} (by depth {:?}), {:.2} ms/step; host split per launch: enqueue {:.3} ms, wait {:.3} ms, read-back {:.3} ms; rows-1:launches {:?}; drained {:.1} ms{}",
            shape.name,
            n_main,
            spine,
            stats.accept_hist,
            leaf_accepts,
            &leaf_by_depth[1..],
            step_ms_sum / n,
            stats.host_ms[0] / n,
            stats.host_ms[1] / n,
            stats.host_ms[2] / n,
            stats.widths,
            stats.drain_ms,
            if reference.is_some() && !fork_reported {
                "; no divergence from the reference stream"
            } else {
                ""
            }
        );
    }
    eng.specdec.as_mut().unwrap().last_stats = stats.clone();
    Ok(SpecDecodeOut {
        decode_ms,
        decode_tokens,
        stopped_by_eos,
        stopped_by_cancel,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::parse_tree_spec;

    fn qwen38_27b_spec() -> ModelSpec {
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

    fn qwen3_8b_spec() -> ModelSpec {
        let cfg = r#"{"model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288,
          "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
          "head_dim": 128, "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
          "tie_word_embeddings": false}"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    fn dims(gb: &GraphBuilder, n: &str) -> Option<Vec<u64>> {
        gb.persistent_tensor_specs()
            .iter()
            .find(|t| t.0 == n)
            .map(|t| t.2.clone())
    }

    #[test]
    fn tree_step_graph_has_node_rows_tables_and_the_path_move() {
        let spec = qwen38_27b_spec();
        let shape = parse_tree_spec("4:1-2@2").unwrap();
        assert_eq!(shape.n_nodes(), 7);
        let gb = build_step_graph(&spec, 8192, 2, &shape, 7, None, &[]);
        assert_eq!(dims(&gb, "ids"), Some(vec![7]));
        assert_eq!(dims(&gb, "tokens"), Some(vec![1, 7]));
        assert_eq!(dims(&gb, "emitted"), Some(vec![EMN]));
        assert_eq!(dims(&gb, "tfix_in"), Some(vec![2 * FIXN]));
        assert_eq!(dims(&gb, "tfix_out"), Some(vec![2 * FIXN]));
        assert_eq!(dims(&gb, "trt_4_1_2_2_anc"), Some(vec![7]));
        assert_eq!(dims(&gb, "L0_S_in"), Some(vec![64, 256, 48, 7]));
        assert_eq!(dims(&gb, "L0_S_ckpt"), Some(vec![64, 256, 48, 7]));
        assert_eq!(dims(&gb, "L3_kcache"), Some(vec![256, 4096, 8]));
        assert_eq!(dims(&gb, "tr_pidx"), Some(vec![5]));
        assert!(dims(&gb, "next_ids").is_none());
        // with the MTP drafter: catch-up on the path (spine + 1 rows), next ids = the tree
        let gm = build_step_graph(&spec, 8192, 2, &shape, 7, Some(0), &[]);
        assert_eq!(dims(&gm, "mtp_hpath"), Some(vec![5120, 5]));
        assert_eq!(dims(&gm, "M_tok"), Some(vec![1, 5]));
        assert_eq!(dims(&gm, "next_ids"), Some(vec![7]));
        assert_eq!(dims(&gm, "mtp_cand"), Some(vec![4, 16]));
        assert_ne!(gb.fingerprint("x"), gm.fingerprint("x"));
        // the one-row shape still takes the pending move and hands its token on
        let root = make_shape(0, &[], "0");
        let g0 = build_step_graph(&spec, 8192, 2, &root, 7, Some(0), &[]);
        assert_eq!(dims(&g0, "ids"), Some(vec![1]));
        assert_eq!(dims(&g0, "tfix_in"), Some(vec![2 * FIXN]));
        assert_eq!(dims(&g0, "next_ids"), Some(vec![1]));
        assert_eq!(dims(&g0, "L0_S_in"), Some(vec![64, 256, 48, 7]));
    }

    #[test]
    fn attention_only_target_builds_a_tree_step_without_state() {
        let spec = qwen3_8b_spec();
        let shape = parse_tree_spec("3:1-3@2").unwrap();
        let gb = build_step_graph(&spec, 4096, 1, &shape, 7, None, &[]);
        assert!(dims(&gb, "a_in").is_none());
        assert_eq!(dims(&gb, "L0_kcache"), Some(vec![128, 4096, 8]));
        assert_eq!(dims(&gb, "L0_qb"), Some(vec![128, 4 * 7, 8]));
        assert!(tree_elfs(&spec).contains(&"GAUDIUM_A128PREPT_G8Q4_ELF"));
        assert!(!tree_elfs(&spec).contains(&"GAUDIUM_GDNT_ELF"));
    }

    #[test]
    fn oracle_puts_the_true_token_on_the_leaf_of_a_corrupted_depth() {
        let shape = parse_tree_spec("3:1-2@2").unwrap(); // rows: root, s1, s2, s3, leaf@1, leaf@2
        let o = OracleRequest {
            stream: vec![10, 11, 12, 13, 14, 15, 16],
            corrupt: vec![2],
        };
        // one token generated so far: the root is stream[0], depth j drafts stream[j]
        let ids = oracle_tree_ids(&o, &shape, &[10], 6);
        assert_eq!(ids, vec![10, 11, 11, 13, 13, 12]);
        // target argmaxes along the true stream: the walk leaves the spine at depth 2
        let mut tokens = vec![0i32; 6];
        tokens[0] = 11; // after the root
        tokens[1] = 12; // after spine 1
        tokens[5] = 13; // after the leaf at depth 2
        let path = walk_tree(&shape, &ids, &tokens);
        assert_eq!(path, vec![0, 1, 5]);
        // no leaf at depth 3: a corrupted depth 3 just ends the path
        let o3 = OracleRequest {
            stream: o.stream.clone(),
            corrupt: vec![3],
        };
        let ids3 = oracle_tree_ids(&o3, &shape, &[10], 6);
        tokens[2] = 13;
        assert_eq!(walk_tree(&shape, &ids3, &tokens), vec![0, 1, 2]);
    }

    #[test]
    fn shapes_are_checked_and_keyed() {
        let spec = qwen38_27b_spec();
        let shape = parse_tree_spec("4:1-2@2").unwrap();
        assert_eq!(shape_key(&shape), "4_1_2_2");
        let m = SpecMethod::OracleTree {
            requests: Vec::new(),
            shape: shape.clone(),
        };
        let tr = TreeRt::new(&m, &spec).unwrap().unwrap();
        assert_eq!(tr.shape_of(7).n_nodes(), 7);
        assert_eq!(tr.shape_of(1).n_nodes(), 1);
        let tabs = tr.tables();
        let anc = &tabs.iter().find(|t| t.0 == "trt_4_1_2_2_anc").unwrap().1;
        // leaf at depth 1 (row 5): itself and the root; leaf at depth 2 (row 6): itself, s1, root
        assert_eq!(anc[5], (1 << 5) | 1);
        assert_eq!(anc[6], (1 << 6) | 0b11);
        let planned = SpecMethod::OracleTree {
            requests: Vec::new(),
            shape: parse_tree_spec("d7@4").unwrap(),
        };
        assert!(TreeRt::new(&planned, &spec).is_err());
        assert!(TreeRt::new(
            &SpecMethod::Oracle {
                requests: Vec::new()
            },
            &spec
        )
        .unwrap()
        .is_none());
        assert_eq!(top2(&[0x00, 0x3f, 0x80, 0x3f, 0x80, 0x3f]).0, 1);
    }
}
