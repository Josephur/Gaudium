//! A published block drafter (the `DFlashDraftModel` form: no convolutions, no candidate
//! selector) behind a Gemma 4 target, as graph passes of the speculative step.
//!
//! This is a child module of `engine::gemma4::spec` (declared there with `#[path]`).
//!
//! The drafter never runs the context tokens. What it computes:
//! - **context** (incremental): for every verified target row, `hidden_norm(fc(concat of the
//!   target's tapped layer outputs))`, then each drafter layer's own K / V projections of that
//!   row (key norm with a weight, RoPE at the row's position). A layer with a sliding window
//!   keeps its context K / V in a **ring** (`slot = position mod ring`): bounded HBM and a bounded
//!   pass whatever the context; a full-attention layer keeps a cache of the context's length.
//! - **block**: rows `[anchor, mask, mask, ..]` at the positions from the anchor on, embedded as
//!   plain rows of the target's table, through the layers (pre-norm attention and SwiGLU MLP). A
//!   windowed layer's row sees the context inside its window and the block rows up to its own; a
//!   full layer's row sees the whole context and the whole block.
//! - **readout**: final norm, the target's tied table, arg-max per row: row `t >= 1` drafts the
//!   token of its own position (the soft cap is monotone and left out).
//!
//! The attention body is the 128-channel block-drafter form (32 query / 8 KV heads) with its own
//! RoPE tables; the ring forms of its context writer and softmax come from the family's glue
//! library. The drafter only proposes: the verify decides every token.

use super::*;

// the 128-channel block-drafter tables (RoPE rows, per-channel parameters in lane order)
use crate::engine::qwen3::spec::dflash2::packer_dflash2 as tables;

/// Kernel ELFs of the drafter passes.
pub(in super::super) const DFLASH_ELFS: &[&str] = &[
    "GAUDIUM_DFAPREP_ELF",
    "GAUDIUM_DFCTX_ELF",
    "GAUDIUM_DFSOFT_ELF",
    "GAUDIUM_DFPOST_ELF",
    "GAUDIUM_GADFSOFTR_ELF",
    "GAUDIUM_GADFCTXR_ELF",
];

/// Drafter attention shape fixed by the block-attention kernels: 32 query / 8 KV heads x 128.
const D_DH: u64 = 128;
const D_NQ: u64 = 32;
const D_NKV: u64 = 8;
const D_QD: u64 = D_NQ * D_DH;
const D_KVD: u64 = D_NKV * D_DH;
/// Block rows the block-attention front end stages keys for.
const MAX_BLOCK: u64 = 16;
/// Rows the RoPE tables carry past the context so a block starting near its end stays in range.
const ROPE_PAD: u64 = 256;

const LAYER_WEIGHTS: &[&str] = &[
    "w_in", "w_post", "W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down", "qn_w", "kn_w",
];
const LAYER_STATE: &[&str] = &["kring", "vring", "kcache", "vcache"];

/// Splits `"D3_W_q"` into `(3, "W_q")`.
fn layer_suffix(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix('D')?;
    let sep = rest.find('_')?;
    let l = rest[..sep].parse().ok()?;
    Some((l, &rest[sep + 1..]))
}

/// A drafter weight or table tensor of this form.
pub(in super::super) fn is_weight(name: &str) -> bool {
    matches!(
        name,
        "df_W_fc"
            | "df_w_hn"
            | "df_w_final"
            | "df_nofix"
            | "df_one"
            | "df_masks"
            | "rope128_c"
            | "rope128_s"
    ) || layer_suffix(name).is_some_and(|(_, what)| LAYER_WEIGHTS.contains(&what))
}

/// A drafter context ring or cache.
pub(in super::super) fn is_state(name: &str) -> bool {
    layer_suffix(name).is_some_and(|(_, what)| LAYER_STATE.contains(&what))
}

/// The checkpoint contract read from `<draft dir>/config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct G4DFlash {
    pub dir: PathBuf,
    pub hid: u64,
    pub inter: u64,
    pub vocab: u64,
    /// Block rows the checkpoint was trained with (the anchor and `block - 1` drafts).
    pub block: u64,
    pub mask_id: i32,
    /// Sliding window of the windowed layers, in positions.
    pub window: u64,
    pub theta: f64,
    /// Target layers whose output feeds the drafter's context.
    pub taps: Vec<usize>,
    /// Per layer: full attention (a context cache) instead of a sliding window (a ring).
    pub full: Vec<bool>,
}

impl G4DFlash {
    pub fn from_config_json(dir: &Path, text: &str) -> Result<Self, String> {
        let bad = |what: &str| format!("block drafter {}: {what}", dir.display());
        let j = parse_json(text).map_err(|e| bad(&format!("config.json: {e:?}")))?;
        let is_form = j
            .get("architectures")
            .and_then(JsonValue::as_array)
            .is_some_and(|a| a.iter().any(|x| x.as_str() == Some("DFlashDraftModel")));
        if !is_form {
            return Err(bad("architectures does not name DFlashDraftModel"));
        }
        let dc = j
            .get("dflash_config")
            .ok_or_else(|| bad("no dflash_config"))?;
        if dc.get("conv_group_size").is_some() || dc.get("selector_rank").is_some() {
            return Err(bad(
                "convolutions / candidate selector (the DFlash2 form) are not built for a Gemma 4 target",
            ));
        }
        let num = |o: &JsonValue, k: &str| o.get(k).and_then(JsonValue::as_u64).unwrap_or(0);
        if num(&j, "head_dim") != D_DH
            || num(&j, "num_attention_heads") != D_NQ
            || num(&j, "num_key_value_heads") != D_NKV
        {
            return Err(bad(
                "heads must be 32 query / 8 KV x 128 (fixed in the block-attention kernels)",
            ));
        }
        let act = j
            .get("hidden_act")
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if act != "silu" {
            return Err(bad(&format!("hidden_act \"{act}\"")));
        }
        let mut full = Vec::new();
        for x in j
            .get("layer_types")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| bad("no layer_types"))?
        {
            match x.as_str() {
                Some("full_attention") => full.push(true),
                Some("sliding_attention") => full.push(false),
                other => return Err(bad(&format!("layer type {other:?}"))),
            }
        }
        let taps: Vec<usize> = dc
            .get("target_layer_ids")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| bad("no dflash_config.target_layer_ids"))?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let c = Self {
            dir: dir.to_path_buf(),
            hid: num(&j, "hidden_size"),
            inter: num(&j, "intermediate_size"),
            vocab: num(&j, "vocab_size"),
            block: dc
                .get("block_size")
                .and_then(JsonValue::as_u64)
                .unwrap_or_else(|| num(&j, "block_size")),
            mask_id: dc
                .get("mask_token_id")
                .and_then(JsonValue::as_i64)
                .ok_or_else(|| bad("no dflash_config.mask_token_id"))? as i32,
            window: num(&j, "sliding_window"),
            theta: j
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .or_else(|| j.get("rope_theta"))
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| bad("no rope_theta"))?,
            taps,
            full,
        };
        if c.full.len() as u64 != num(&j, "num_hidden_layers") || c.full.is_empty() {
            return Err(bad("layer_types does not have num_hidden_layers entries"));
        }
        if c.hid == 0 || c.inter == 0 || c.taps.is_empty() {
            return Err(bad("hidden_size / intermediate_size / target_layer_ids"));
        }
        if !(2..=MAX_BLOCK).contains(&c.block) {
            return Err(bad(&format!("block_size {} (2..={MAX_BLOCK})", c.block)));
        }
        if c.full.iter().any(|&f| !f) && (c.window < 2 || c.window > G4_MAX_RING / 2) {
            return Err(bad(&format!("sliding_window {}", c.window)));
        }
        Ok(c)
    }

    pub fn check_target(&self, d: &G4Dims, max_rows: u64) -> Result<(), String> {
        let bad = |what: String| format!("block drafter {}: {what}", self.dir.display());
        if self.hid != d.hid || self.vocab != d.vocab {
            return Err(bad(format!(
                "hidden {} / vocab {} are not the target's {} / {}",
                self.hid, self.vocab, d.hid, d.vocab
            )));
        }
        if let Some(&l) = self.taps.iter().find(|&&l| l >= d.n_layers) {
            return Err(bad(format!(
                "target_layer_ids names layer {l} of a {}-layer target",
                d.n_layers
            )));
        }
        if max_rows > self.block {
            return Err(bad(format!(
                "--draft-k {}: the checkpoint drafts {} tokens per block",
                max_rows - 1,
                self.block - 1
            )));
        }
        Ok(())
    }

    /// Slots of a windowed layer's context ring: a power of two holding the window beside the
    /// widest context pass (`pass_rows`: a prefill tile or the verify rows).
    pub fn ring(&self, pass_rows: u64) -> u64 {
        let mut r = 256u64;
        while r < self.window + pass_rows.max(64) {
            r *= 2;
        }
        r
    }

    pub fn weight_bytes(&self) -> u64 {
        let h = self.hid;
        let per_layer = 2 * h * 2
            + 2 * D_QD * h * 2
            + 2 * D_KVD * h * 2
            + 3 * h * self.inter * 2
            + 2 * D_DH * 4;
        h * h * self.taps.len() as u64 * 2 + 2 * h * 2 + per_layer * self.full.len() as u64
    }
}

/// The drafter's RoPE tables (`{128, lmax + pad}`, one row per position).
fn rope_tables(gb: &mut GraphBuilder, lmax: u64) -> (TensorId, TensorId) {
    let n = lmax + ROPE_PAD;
    (
        gb.get_or_add_persistent_tensor("rope128_c", SynDataType::Float, &[128, n]),
        gb.get_or_add_persistent_tensor("rope128_s", SynDataType::Float, &[128, n]),
    )
}

/// A drafter layer's context K / V state tensors (a ring, or a cache of `lmax` rows).
fn state(
    gb: &mut GraphBuilder,
    c: &G4DFlash,
    l: usize,
    lmax: u64,
    ring: u64,
) -> (TensorId, TensorId, u64) {
    let (kn, vn, rows) = if c.full[l] {
        ("kcache", "vcache", lmax)
    } else {
        ("kring", "vring", ring)
    };
    let dims = [D_DH, rows, D_NKV];
    (
        gb.get_or_add_persistent_tensor(format!("D{l}_{kn}"), SynDataType::Bf16, &dims),
        gb.get_or_add_persistent_tensor(format!("D{l}_{vn}"), SynDataType::Bf16, &dims),
        rows,
    )
}

/// Standard RMS norm with a plain weight over `t` rows (the vendor node: a draft never decides a
/// token, so its rounding is free to follow the fastest node).
fn d_norm(gb: &mut GraphBuilder, x: TensorId, w: &str, y: &str, hid: u64, t: u64) -> TensorId {
    g4_norm(gb, x, w, y, hid, t, false)
}

/// Context K / V of every drafter layer for `t` target rows at positions `cursor + depth[row]`
/// (`hs[i]`: the output of tapped target layer `i`).
#[allow(clippy::too_many_arguments)]
fn append_ctx(
    gb: &mut GraphBuilder,
    c: &G4DFlash,
    hs: &[TensorId],
    t: u64,
    cursor: TensorId,
    depth: TensorId,
    lmax: u64,
    ring: u64,
) {
    let hid = c.hid;
    let fci = hid * hs.len() as u64;
    let (rc, rs) = rope_tables(gb, lmax);
    let nofix = gb.get_or_add_persistent_tensor("df_nofix", SynDataType::Int32, &[2]);
    let cat = gb.add_persistent_tensor("ctx_cat", SynDataType::Bf16, &[fci, t]);
    gb.op_concat("ctx_concat", hs, cat, 0);
    let f = g4_linear(gb, cat, "df_W_fc", fci, hid, t, "ctx_f");
    let cn = d_norm(gb, f, "df_w_hn", "ctx_cn", hid, t);
    for l in 0..c.full.len() {
        let lp = format!("D{l}_");
        let kc = g4_linear(
            gb,
            cn,
            &format!("{lp}W_k"),
            hid,
            D_KVD,
            t,
            &format!("ctx_{lp}k"),
        );
        let vc = g4_linear(
            gb,
            cn,
            &format!("{lp}W_v"),
            hid,
            D_KVD,
            t,
            &format!("ctx_{lp}v"),
        );
        let kn_w =
            gb.get_or_add_persistent_tensor(format!("{lp}kn_w"), SynDataType::Float, &[D_DH]);
        let (ks, vs, _) = state(gb, c, l, lmax, ring);
        gb.add_node(
            if c.full[l] {
                "gaudium_df_ctx"
            } else {
                "gaudium_ga_dfctxr"
            },
            format!("ctx_{lp}kv"),
            &[kc, vc, kn_w, rc, rs, cursor, depth, nofix],
            &[ks, vs],
            &[],
        );
    }
}

/// One drafter layer over the `t` block rows at positions `cur + row`.
#[allow(clippy::too_many_arguments)]
fn append_layer(
    gb: &mut GraphBuilder,
    c: &G4DFlash,
    x: TensorId,
    l: usize,
    t: u64,
    cur: TensorId,
    lmax: u64,
    ring: u64,
) -> TensorId {
    let bf = SynDataType::Bf16;
    let f32t = SynDataType::Float;
    let hid = c.hid;
    let lp = format!("D{l}_");
    let r4 = 4 * t;
    let (rc, rs) = rope_tables(gb, lmax);
    let nofix = gb.get_or_add_persistent_tensor("df_nofix", SynDataType::Int32, &[2]);
    let h = d_norm(gb, x, &format!("{lp}w_in"), &format!("{lp}h"), hid, t);
    let q = g4_linear(gb, h, &format!("{lp}W_q"), hid, D_QD, t, &format!("{lp}q"));
    let k = g4_linear(gb, h, &format!("{lp}W_k"), hid, D_KVD, t, &format!("{lp}k"));
    let v = g4_linear(gb, h, &format!("{lp}W_v"), hid, D_KVD, t, &format!("{lp}v"));
    let qn_w = gb.get_or_add_persistent_tensor(format!("{lp}qn_w"), f32t, &[D_DH]);
    let kn_w = gb.get_or_add_persistent_tensor(format!("{lp}kn_w"), f32t, &[D_DH]);
    let qb = gb.add_persistent_tensor(format!("{lp}qb"), bf, &[D_DH, r4, D_NKV]);
    let kblk = gb.add_persistent_tensor(format!("{lp}kblk"), bf, &[D_DH, 128, D_NKV]);
    let vblk = gb.add_persistent_tensor(format!("{lp}vblk"), bf, &[D_DH, 128, D_NKV]);
    gb.add_node(
        "gaudium_df_attn_prep",
        format!("{lp}attn_prep"),
        &[q, k, v, qn_w, kn_w, rc, rs, cur],
        &[qb, kblk, vblk],
        &[],
    );
    let (ks, vs, rows) = state(gb, c, l, lmax, ring);
    let sc = gb.add_persistent_tensor(format!("{lp}sc"), f32t, &[rows, r4, D_NKV]);
    let sb = gb.add_persistent_tensor(format!("{lp}sb"), f32t, &[128, r4, D_NKV]);
    gb.op_batch_gemm(format!("{lp}qk_ctx"), qb, ks, sc, false, true);
    gb.op_batch_gemm(format!("{lp}qk_blk"), qb, kblk, sb, false, true);
    let pc = gb.add_persistent_tensor(format!("{lp}Pc"), bf, &[rows, r4, D_NKV]);
    let pb = gb.add_persistent_tensor(format!("{lp}Pb"), bf, &[128, r4, D_NKV]);
    if c.full[l] {
        // node params: T, window (one nothing reaches: every context position, the whole block)
        gb.add_node(
            "gaudium_df_softmax",
            format!("{lp}softmax"),
            &[sc, sb, cur, nofix],
            &[pc, pb],
            &u32_params(&[t as u32, (lmax + G4_CHUNK) as u32]),
        );
    } else {
        // node params: T, window (context inside the window, block rows up to the row's own)
        gb.add_node(
            "gaudium_ga_dfsoftr",
            format!("{lp}softmax"),
            &[sc, sb, cur, nofix],
            &[pc, pb],
            &u32_params(&[t as u32, c.window as u32]),
        );
    }
    let aoc = gb.add_persistent_tensor(format!("{lp}aoc"), bf, &[D_DH, r4, D_NKV]);
    let aob = gb.add_persistent_tensor(format!("{lp}aob"), bf, &[D_DH, r4, D_NKV]);
    let ao = gb.add_persistent_tensor(format!("{lp}ao"), bf, &[D_DH, r4, D_NKV]);
    gb.op_batch_gemm(format!("{lp}pv_ctx"), pc, vs, aoc, false, false);
    gb.op_batch_gemm(format!("{lp}pv_blk"), pb, vblk, aob, false, false);
    gb.op_add_bf16(format!("{lp}ao_add"), aoc, aob, ao);
    let core = gb.add_persistent_tensor(format!("{lp}core"), bf, &[D_QD, t]);
    gb.add_node(
        "gaudium_df_post",
        format!("{lp}attn_post"),
        &[ao],
        &[core],
        &[],
    );
    let attn = g4_linear(
        gb,
        core,
        &format!("{lp}W_o"),
        D_QD,
        hid,
        t,
        &format!("{lp}attn"),
    );
    let x2 = gb.add_persistent_tensor(format!("{lp}x2"), bf, &[hid, t]);
    gb.op_add_bf16(format!("{lp}x2_add"), x, attn, x2);
    let h2 = d_norm(gb, x2, &format!("{lp}w_post"), &format!("{lp}h2"), hid, t);
    let gate = g4_linear(
        gb,
        h2,
        &format!("{lp}W_gate"),
        hid,
        c.inter,
        t,
        &format!("{lp}gate"),
    );
    let up = g4_linear(
        gb,
        h2,
        &format!("{lp}W_up"),
        hid,
        c.inter,
        t,
        &format!("{lp}up"),
    );
    let act = gb.add_persistent_tensor(format!("{lp}act"), bf, &[c.inter, t]);
    gb.op_silu_bf16(format!("{lp}silu"), gate, act);
    let m = gb.add_persistent_tensor(format!("{lp}m"), bf, &[c.inter, t]);
    gb.op_mult_bf16(format!("{lp}swiglu"), act, up, m);
    let mo = g4_linear(
        gb,
        m,
        &format!("{lp}W_down"),
        c.inter,
        hid,
        t,
        &format!("{lp}mlp"),
    );
    let out = gb.add_persistent_tensor(format!("{lp}out"), bf, &[hid, t]);
    gb.op_add_bf16(format!("{lp}out_add"), x2, mo, out);
    out
}

/// The block `ids_blk {tb}` (`[anchor, mask ..]`) at positions `cur + row` -> the `tb - 1` drafts
/// (the arg-max of rows `1..tb`).
#[allow(clippy::too_many_arguments)]
fn append_block(
    gb: &mut GraphBuilder,
    c: &G4DFlash,
    ids_blk: TensorId,
    tb: u64,
    cur: TensorId,
    lmax: u64,
    ring: u64,
) -> TensorId {
    let i32t = SynDataType::Int32;
    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[c.hid, c.vocab]);
    // plain rows of the target's table (no embedding scale): what the drafter was trained on
    let mut x = gb.add_persistent_tensor("blk_x0", SynDataType::Bf16, &[c.hid, tb]);
    gb.op_gather_bf16("blk_embed", table, ids_blk, x, 1);
    for l in 0..c.full.len() {
        x = append_layer(gb, c, x, l, tb, cur, lmax, ring);
    }
    let hf = d_norm(gb, x, "df_w_final", "blk_hf", c.hid, tb);
    let logits = g4_linear(gb, hf, "embed_table", c.hid, c.vocab, tb, "blk_logits");
    let tok = gb.add_persistent_tensor("blk_tok", i32t, &[1, tb]);
    gb.add_node("gaudium_ga_argmax", "blk_argmax", &[logits], &[tok], &[]);
    // rows 1.. are the drafts: the anchor's row has none
    let one = gb.get_or_add_persistent_tensor("df_one", i32t, &[1]);
    let drafts = gb.add_persistent_tensor("blk_drafts", i32t, &[tb - 1]);
    gb.add_node("gaudium_sel", "blk_drafts", &[tok, one], &[drafts], &[]);
    drafts
}

/// The drafter section of a step recipe, after the accept: the context of the `t` verified rows,
/// then the block anchored on the accepted token at `cursor_out` -> `next_ids {kd + 1}`.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn append_step(
    gb: &mut GraphBuilder,
    d: &G4Dims,
    c: &G4DFlash,
    t: u64,
    kd: u64,
    cursor: TensorId,
    depth: TensorId,
    tokens: TensorId,
    a_out: TensorId,
    cursor_out: TensorId,
    next_ids: TensorId,
    lmax: u64,
    ring: u64,
) {
    let i32t = SynDataType::Int32;
    let taps: Vec<TensorId> = c
        .taps
        .iter()
        .map(|&l| {
            gb.get_or_add_persistent_tensor(format!("L{l}_out"), SynDataType::Bf16, &[d.hid, t])
        })
        .collect();
    append_ctx(gb, c, &taps, t, cursor, depth, lmax, ring);
    if kd == 0 {
        gb.add_node(
            "gaudium_sel",
            "next_ids",
            &[tokens, a_out],
            &[next_ids],
            &[],
        );
        return;
    }
    let tok0 = gb.add_persistent_tensor("g4_tok0", i32t, &[1]);
    gb.add_node("gaudium_sel", "g4_tok0_sel", &[tokens, a_out], &[tok0], &[]);
    let masks = gb.get_or_add_persistent_tensor("df_masks", i32t, &[kd]);
    let ids_blk = gb.add_persistent_tensor("df_ids_blk", i32t, &[kd + 1]);
    gb.op_concat("df_ids", &[tok0, masks], ids_blk, 0);
    let drafts = append_block(gb, c, ids_blk, kd + 1, cursor_out, lmax, ring);
    gb.op_concat("next_ids", &[tok0, drafts], next_ids, 0);
}

/// First block after a prefill: `b0_blk {kd + 1}` (the host's `[token, mask ..]`) at `b0_cur`
/// -> `b0_ids = [b0_tok, kd drafts]`.
pub(in super::super) fn append_block0(
    gb: &mut GraphBuilder,
    c: &G4DFlash,
    kd: u64,
    lmax: u64,
    ring: u64,
) {
    let i32t = SynDataType::Int32;
    let tok = gb.add_persistent_tensor("b0_tok", i32t, &[1]);
    let blk = gb.add_persistent_tensor("b0_blk", i32t, &[kd + 1]);
    let cur = gb.add_persistent_tensor("b0_cur", i32t, &[1]);
    let drafts = append_block(gb, c, blk, kd + 1, cur, lmax, ring);
    let ids = gb.add_persistent_tensor("b0_ids", i32t, &[kd + 1]);
    gb.op_concat("b0_ids", &[tok, drafts], ids, 0);
}

/// Drafter context for a prefill tile of `t` rows `g_g4df_ctx_T{t}` (`df_tap{i}`: the tile's
/// tapped layer outputs; `ctx_cursor`: the tile's first position).
pub fn build_ctx_graph(
    d: &G4Dims,
    c: &G4DFlash,
    t: u64,
    lmax: u64,
    ring: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = g4_graph(d, format!("g_g4df_ctx_T{t}"), elf_hashes);
    let hs: Vec<TensorId> = (0..c.taps.len())
        .map(|i| gb.add_persistent_tensor(format!("df_tap{i}"), SynDataType::Bf16, &[d.hid, t]))
        .collect();
    let cursor = gb.add_persistent_tensor("ctx_cursor", SynDataType::Int32, &[1]);
    let depth = gb.get_or_add_persistent_tensor("sp_depth", SynDataType::Int32, &[t]);
    append_ctx(&mut gb, c, &hs, t, cursor, depth, lmax, ring);
    gb
}

/// Device byte image of drafter weight / table tensor `name` (`bytes`: the graph tensor's size,
/// which sizes the tables).
pub(in super::super) fn pack_weight(
    store: &SafetensorsStore,
    c: &G4DFlash,
    name: &str,
    bytes: u64,
) -> Result<Vec<u8>, String> {
    let i32s = |vals: Vec<i32>| {
        vals.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let need = |full: &str, shape: &[u64]| -> Result<&[u8], String> {
        let t = store
            .get(full)
            .map_err(|e| format!("drafter tensor '{full}': {e}"))?;
        if t.dtype != "BF16" || t.shape != shape {
            return Err(format!(
                "drafter tensor '{full}' is {} {:?}, expected BF16 {shape:?}",
                t.dtype, t.shape
            ));
        }
        store
            .tensor_bytes(full)
            .map_err(|e| format!("read '{full}': {e}"))?
            .ok_or_else(|| format!("drafter tensor '{full}' is not mapped"))
    };
    let raw = |full: &str, shape: &[u64]| need(full, shape).map(<[u8]>::to_vec);
    let (hid, inter) = (c.hid, c.inter);
    match name {
        "df_W_fc" => return raw("fc.weight", &[hid, hid * c.taps.len() as u64]),
        "df_w_hn" => return raw("hidden_norm.weight", &[hid]),
        "df_w_final" => return raw("norm.weight", &[hid]),
        "df_nofix" => return Ok(i32s(vec![-1, -1])),
        "df_one" => return Ok(i32s(vec![1])),
        "df_masks" => return Ok(i32s(vec![c.mask_id; (bytes / 4) as usize])),
        "rope128_c" | "rope128_s" => {
            let (rc, rs) = tables::rope128_tables(c.theta, (bytes / (128 * 4)) as usize);
            return Ok(tables::f32_bytes(if name == "rope128_c" {
                &rc
            } else {
                &rs
            }));
        }
        _ => {}
    }
    let Some((l, what)) = layer_suffix(name).filter(|&(l, _)| l < c.full.len()) else {
        return Err(format!("block drafter: no packing rule for '{name}'"));
    };
    let lp = format!("layers.{l}.");
    let lane = |full: &str| need(full, &[D_DH]).map(tables::lane_f32);
    match what {
        "w_in" => raw(&format!("{lp}input_layernorm.weight"), &[hid]),
        "w_post" => raw(&format!("{lp}post_attention_layernorm.weight"), &[hid]),
        "W_q" => raw(&format!("{lp}self_attn.q_proj.weight"), &[D_QD, hid]),
        "W_k" => raw(&format!("{lp}self_attn.k_proj.weight"), &[D_KVD, hid]),
        "W_v" => raw(&format!("{lp}self_attn.v_proj.weight"), &[D_KVD, hid]),
        "W_o" => raw(&format!("{lp}self_attn.o_proj.weight"), &[hid, D_QD]),
        "W_gate" => raw(&format!("{lp}mlp.gate_proj.weight"), &[inter, hid]),
        "W_up" => raw(&format!("{lp}mlp.up_proj.weight"), &[inter, hid]),
        "W_down" => raw(&format!("{lp}mlp.down_proj.weight"), &[hid, inter]),
        "qn_w" => lane(&format!("{lp}self_attn.q_norm.weight")),
        "kn_w" => lane(&format!("{lp}self_attn.k_norm.weight")),
        _ => Err(format!("block drafter: no packing rule for '{name}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) const CFG_12B: &str = r#"{
        "architectures": ["DFlashDraftModel"],
        "block_size": 16,
        "dflash_config": {"mask_token_id": 4, "target_layer_ids": [1, 10, 19, 27, 36, 45]},
        "head_dim": 128, "hidden_act": "silu", "hidden_size": 3840, "intermediate_size": 7680,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "full_attention"],
        "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "sliding_window": 2048, "vocab_size": 262144
    }"#;

    #[test]
    fn config_contract_is_read_and_checked() {
        let c = G4DFlash::from_config_json(Path::new("d"), CFG_12B).unwrap();
        assert_eq!((c.hid, c.inter, c.block, c.mask_id), (3840, 7680, 16, 4));
        assert_eq!(c.taps, vec![1, 10, 19, 27, 36, 45]);
        assert_eq!(c.full, vec![false, false, false, false, true]);
        assert_eq!((c.window, c.theta), (2048, 1.0e6));
        // the ring holds the window beside a context pass
        assert_eq!(c.ring(8), 4096);
        assert_eq!(c.ring(256), 4096);
        // the selector form is another drafter
        let sel = CFG_12B.replace(
            "\"mask_token_id\": 4,",
            "\"mask_token_id\": 4, \"selector_rank\": 256,",
        );
        assert!(G4DFlash::from_config_json(Path::new("d"), &sel).is_err());
        let heads = CFG_12B.replace("\"num_attention_heads\": 32", "\"num_attention_heads\": 16");
        assert!(G4DFlash::from_config_json(Path::new("d"), &heads).is_err());
    }

    #[test]
    fn tensor_names_have_one_role() {
        for n in [
            "D0_W_q",
            "D4_kn_w",
            "df_W_fc",
            "rope128_s",
            "df_masks",
            "df_one",
        ] {
            assert!(is_weight(n) && !is_state(n), "{n}");
        }
        for n in ["D0_kring", "D3_vring", "D4_kcache", "D4_vcache"] {
            assert!(is_state(n) && !is_weight(n), "{n}");
        }
        for n in [
            "D0_h",
            "D0_qb",
            "D2_sc",
            "ctx_D1_k",
            "blk_logits",
            "df_ids_blk",
            "df_tap0",
        ] {
            assert!(!is_weight(n) && !is_state(n), "{n}");
        }
    }
}
