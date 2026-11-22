//! DFlash2 block drafter on the card (checkpoints of the `DFlash2DraftModel` family: a small
//! non-causal transformer with two-tap dynamic convolutions and a codebook candidate selector).
//!
//! The drafter never embeds or runs the context tokens: it reads the target's own residual stream.
//! What one draft block computes, for an anchor token at position `p` and `block - 1` mask rows:
//! - **context** (incremental): per drafter layer, K/V of `hidden_norm(fc(concat(target taps)))`
//!   for the target rows the verify just produced, written to the layer's context cache
//!   (`gaudium_df_ctx`: per-head key norm, RoPE at the row's position, cache slot of the row);
//! - **block pass**: `[anchor, mask ..]` embedded with the target's table through `nl` layers. A
//!   layer wraps both its attention and its MLP in a dynamic grouped convolution over the block
//!   rows (`gaudium_df_conv`: per row and 16-channel group, coefficients projected from the
//!   normalised layer input, one tap on the row itself and one on the row before it), and attends
//!   to the context cache inside a sliding window plus every row of the block, in both directions
//!   (`gaudium_df_softmax`: one softmax over the two key sets);
//! - **head**: final norm, the target's `lm_head`, top-16 per row (`gaudium_df_topk1/2`), the
//!   selector lattice `unary[c] + <pred[prev] * h, succ[c]>` over consecutive rows
//!   (`gaudium_df_score`) and the greedy walk through it (`gaudium_df_walk`).
//!
//! The drafter only proposes: the verify decides every token, so nothing here affects exactness.
//!
//! Vision: the context is the target's residual stream, so an image position contributes what the
//! target computed from its vision embedding (never a pad-token embedding), and the block ids are
//! generated tokens and masks only. RoPE rows are addressed by KV slot and hold the rotation of
//! the slot's position (`packer_dflash2::rope128_rows`); a text request has slot = position.
//!
//! Geometry (block size, taps, mask id, conv layout, selector rank, window) comes from the
//! checkpoint's `config.json`; anything the device graph does not implement fails at load.

use std::borrow::Cow;
use std::path::Path;

use super::super::*;
use crate::json::{parse_json, JsonValue};

#[path = "packer_dflash2.rs"]
pub(crate) mod packer_dflash2;

/// Kernel ELFs of the drafter passes.
pub const DFLASH2_ELFS: &[&str] = &[
    "GAUDIUM_DFCONV_ELF",
    "GAUDIUM_DFAPREP_ELF",
    "GAUDIUM_DFCTX_ELF",
    "GAUDIUM_DFSOFT_ELF",
    "GAUDIUM_DFPOST_ELF",
    "GAUDIUM_DFTOPK1_ELF",
    "GAUDIUM_DFTOPK2_ELF",
    "GAUDIUM_DFSCORE_ELF",
    "GAUDIUM_DFWALK_ELF",
    "GAUDIUM_DFIDS_ELF",
    "GAUDIUM_DFNEXT_ELF",
];

/// Drafter attention shape fixed by the block-attention kernels: 32 query / 8 KV heads x 128.
const D_DH: u64 = 128;
const D_NQ: u64 = 32;
const D_NKV: u64 = 8;
const D_QD: u64 = D_NQ * D_DH;
const D_KVD: u64 = D_NKV * D_DH;
/// Candidates per row and selector rank fixed by the top-k / score / walk kernels.
const TOPK: u64 = 16;
const RANK: u64 = 256;
/// Vocabulary chunks of the two-stage top-16 kernels.
const TOPK_CH: u64 = 32;
/// Block rows the conv / softmax / walk kernels were validated for.
const BLOCK: u64 = 8;
/// Channels per convolution group and taps per side fixed by the conv kernel.
const CONV_GROUP: u64 = 16;
const CONV_TAPS: u64 = 2;
/// Rows the RoPE tables carry past the context so a block starting near its end stays in range.
pub const ROPE_PAD: u64 = 256;
/// Window value of a checkpoint without a sliding window (no context key is ever below it).
const NO_WINDOW: u64 = (u32::MAX / 4) as u64;

/// Device tensors of the block readout's draft vocabulary: the `lm_head` rows and the
/// row -> token id table (uploaded by [`DFlash2::upload_draft_vocab`], not by the weight loop).
pub const DV_HEAD: &str = "df_W_lmd";
pub const DV_IDS: &str = "df_dv_ids";

/// The checkpoint contract read from `<draft dir>/config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlash2Cfg {
    /// Block rows: the anchor plus `block - 1` drafts.
    pub block: u64,
    pub nl: usize,
    /// Width of a convolution's coefficient projection: 2 sides x 2 taps x `hidden / 16` groups.
    pub kp: u64,
    pub mask_id: i32,
    /// Sliding window of the context attention in positions.
    pub window: u64,
    pub rope_theta: f64,
    /// Target layers whose output residual feeds the drafter's context.
    pub taps: Vec<usize>,
    pub hidden: u64,
    pub intermediate: u64,
    pub vocab: u64,
    /// Run option, not part of the checkpoint: rows of the block readout's draft vocabulary
    /// (`df_W_lmd`, a copy of those `lm_head` rows; 0 = the full `lm_head`).
    pub nvd: u64,
}

impl DFlash2Cfg {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let j = parse_json(text).map_err(|e| format!("dflash2 config: {e}"))?;
        let fail = |m: &str| format!("dflash2 config: {m}");
        let num = |o: &JsonValue, k: &str| -> Result<u64, String> {
            o.get(k)
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| format!("dflash2 config: missing integer '{k}'"))
        };
        let dc = j
            .get("dflash_config")
            .ok_or_else(|| fail("missing dflash_config"))?;
        if dc.get("conv_group_size").is_none() || dc.get("selector_rank").is_none() {
            return Err(fail(
                "no convolutions / candidate selector (a block drafter of another form is not this drafter)",
            ));
        }
        if num(dc, "conv_group_size")? != CONV_GROUP || num(dc, "conv_kernel_size")? != CONV_TAPS {
            return Err(fail(
                "conv layout must be groups of 16 channels with 2 taps (fixed in the conv kernel)",
            ));
        }
        if num(dc, "selector_rank")? != RANK || num(dc, "selector_top_k")? != TOPK {
            return Err(fail(
                "selector must be rank 256 over the top 16 (fixed in the score / walk kernels)",
            ));
        }
        let block = num(dc, "block_size")?;
        if block != BLOCK {
            return Err(fail(
                "block_size must be 8 (the block kernels draft 8 rows)",
            ));
        }
        if j.get("is_causal").and_then(JsonValue::as_bool) == Some(true) {
            return Err(fail(
                "is_causal must be false (the block attends in both directions)",
            ));
        }
        let nl = num(&j, "num_hidden_layers")? as usize;
        if let Some(lt) = j.get("layer_types").and_then(JsonValue::as_array) {
            if lt.len() != nl || lt.iter().any(|v| v.as_str() != Some("sliding_attention")) {
                return Err(fail("every layer must be of type sliding_attention"));
            }
        }
        if num(&j, "num_attention_heads")? != D_NQ
            || num(&j, "num_key_value_heads")? != D_NKV
            || num(&j, "head_dim")? != D_DH
        {
            return Err(fail("drafter attention must be 32 / 8 heads x 128"));
        }
        let eps = j
            .get("rms_norm_eps")
            .and_then(JsonValue::as_f64)
            .unwrap_or(0.0);
        if (eps - 1e-6).abs() > 1e-12 {
            return Err(fail("rms_norm_eps must be 1e-6 (fixed in the kernels)"));
        }
        let taps: Vec<usize> = dc
            .get("target_layer_ids")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| fail("missing dflash_config.target_layer_ids"))?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let window = match j.get("use_sliding_window").and_then(JsonValue::as_bool) {
            Some(false) => NO_WINDOW,
            _ => j
                .get("sliding_window")
                .and_then(JsonValue::as_u64)
                .unwrap_or(NO_WINDOW),
        };
        let rope_theta = j
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| j.get("rope_theta"))
            .and_then(JsonValue::as_f64)
            .ok_or_else(|| fail("missing rope_theta"))?;
        let hidden = num(&j, "hidden_size")?;
        let cfg = Self {
            block,
            nl,
            kp: hidden / CONV_GROUP * 2 * CONV_TAPS,
            mask_id: num(dc, "mask_token_id")? as i32,
            window,
            rope_theta,
            taps,
            hidden,
            intermediate: num(&j, "intermediate_size")?,
            vocab: num(&j, "vocab_size")?,
            nvd: 0,
        };
        if cfg.taps.is_empty() || cfg.nl == 0 || cfg.window <= cfg.block {
            return Err(fail("taps / layers / window layout"));
        }
        Ok(cfg)
    }

    /// The drafter reads the target's hidden states and shares its embedding table and `lm_head`.
    pub fn check_target(&self, spec: &ModelSpec) -> Result<(), String> {
        if self.hidden != spec.hidden || self.vocab != spec.vocab {
            return Err(format!(
                "dflash2: drafter dims (hidden {}, vocab {}) differ from the target's ({}, {})",
                self.hidden, self.vocab, spec.hidden, spec.vocab
            ));
        }
        if !self.hidden.is_multiple_of(128) || !self.vocab.is_multiple_of(128) {
            return Err(
                "dflash2: hidden and vocab must be multiples of 128 (conv / top-16 kernels)".into(),
            );
        }
        if self.mask_id < 0 || self.mask_id as u64 >= spec.vocab {
            return Err(format!(
                "dflash2: mask_token_id {} outside the target's vocabulary",
                self.mask_id
            ));
        }
        let n = spec.n_layers as usize;
        if self.taps.iter().any(|&t| t + 1 >= n) {
            return Err(format!(
                "dflash2: target_layer_ids {:?} must lie below the target's last layer ({n} layers)",
                self.taps
            ));
        }
        Ok(())
    }
}

fn drafter_layer_suffix(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix('D')?;
    let sep = rest.find('_')?;
    if sep == 0 || !rest[..sep].bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((rest[..sep].parse().ok()?, &rest[sep + 1..]))
}

/// Context caches of the drafter: zeroed per request, planned as KV.
pub fn is_drafter_kv(name: &str) -> bool {
    drafter_layer_suffix(name).is_some_and(|(_, s)| s == "kcache" || s == "vcache")
}

/// Tensors uploaded once from the checkpoint (or computed from its config).
pub fn is_drafter_weight(name: &str) -> bool {
    if matches!(
        name,
        "df_W_fc"
            | "df_w_hn"
            | "df_w_final"
            | "df_W_hp"
            | "df_pred"
            | "df_succ"
            | "df_nofix"
            | "rope128_c"
            | "rope128_s"
    ) {
        return true;
    }
    drafter_layer_suffix(name).is_some_and(|(_, s)| {
        matches!(
            s,
            "w_in"
                | "w_post"
                | "W_q"
                | "W_k"
                | "W_v"
                | "W_o"
                | "W_gate"
                | "W_up"
                | "W_down"
                | "W_ac"
                | "W_mc"
                | "qn_w"
                | "kn_w"
                | "ac_base0"
                | "ac_base1"
                | "mc_base0"
                | "mc_base1"
        )
    })
}

// ============================================================================
// Graph passes
// ============================================================================

fn d_rmsnorm(
    gb: &mut GraphBuilder,
    x: TensorId,
    w_name: &str,
    y_name: &str,
    hid: u64,
    t: u64,
) -> TensorId {
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Bf16, &[hid]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[hid, t]);
    let ir = gb.add_persistent_tensor(format!("{y_name}_ir"), SynDataType::Float, &[1, t]);
    gb.op_rms_norm_bf16(format!("{y_name}_norm"), x, w, y, ir, 1e-6, false);
    y
}

/// A convolution's coefficients for `t` rows: the kernel projection of `h` as f32 `{kp, t}`
/// (the projection's bf16 result converted, as the checkpoint's own arithmetic rounds it).
fn conv_coef(
    gb: &mut GraphBuilder,
    dc: &DFlash2Cfg,
    h: TensorId,
    w_name: &str,
    y_name: &str,
    t: u64,
) -> TensorId {
    let cb = add_linear_bf16(gb, h, w_name, dc.hidden, dc.kp, t, &format!("{y_name}_bf"));
    let y = gb.add_persistent_tensor(y_name, SynDataType::Float, &[dc.kp, t]);
    // ns_CastKernel::Params { round_mode = CAST_ROUND_HALF_NE (0) }
    gb.add_node(
        "cast_bf16_to_f32",
        format!("{y_name}_cast"),
        &[cb],
        &[y],
        &u32_params(&[0u32]),
    );
    y
}

/// `y = conv(x)` with side `side` of the coefficients (0 = before the sublayer, 1 = after it):
/// `y[t, c] = (base[0][c] + coef[t, 0, c / 16]) x[t, c] + (base[1][c] + coef[t, 1, c / 16]) x[t - 1, c]`,
/// the second tap absent on the block's first row.
#[allow(clippy::too_many_arguments)]
fn conv(
    gb: &mut GraphBuilder,
    dc: &DFlash2Cfg,
    x: TensorId,
    coef: TensorId,
    base_name: &str,
    side: u64,
    y_name: &str,
    t: u64,
) -> TensorId {
    let base = gb.get_or_add_persistent_tensor(
        format!("{base_name}{side}"),
        SynDataType::Float,
        &[dc.hidden, 2],
    );
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[dc.hidden, t]);
    // scalars: coefficient offset of this side, offset of the second tap inside a side
    gb.add_node(
        "gaudium_df_conv",
        format!("{y_name}_conv"),
        &[x, coef, base],
        &[y],
        &u32_params(&[(side * dc.kp / 2) as u32, (dc.kp / 4) as u32]),
    );
    y
}

/// Context K/V of every drafter layer for `t` target rows (`hs[i]` = output residual of tap `i`).
#[allow(clippy::too_many_arguments)]
pub fn append_ctx_pass(
    gb: &mut GraphBuilder,
    dc: &DFlash2Cfg,
    hs: &[TensorId],
    t: u64,
    cursor: TensorId,
    depth: TensorId,
    fix: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
) {
    let hid = dc.hidden;
    let fci = hid * hs.len() as u64;
    let cat = gb.add_persistent_tensor("ctx_cat", SynDataType::Bf16, &[fci, t]);
    gb.op_concat("ctx_concat", hs, cat, 0);
    let f = add_linear_bf16(gb, cat, "df_W_fc", fci, hid, t, "ctx_f");
    let cn = d_rmsnorm(gb, f, "df_w_hn", "ctx_cn", hid, t);
    for l in 0..dc.nl {
        let lp = format!("D{l}_");
        let kc = add_linear_bf16(
            gb,
            cn,
            &format!("{lp}W_k"),
            hid,
            D_KVD,
            t,
            &format!("ctx_{lp}k"),
        );
        let vc = add_linear_bf16(
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
        let kcache = gb.get_or_add_persistent_tensor(
            format!("{lp}kcache"),
            SynDataType::Bf16,
            &[D_DH, lmax, D_NKV],
        );
        let vcache = gb.get_or_add_persistent_tensor(
            format!("{lp}vcache"),
            SynDataType::Bf16,
            &[D_DH, lmax, D_NKV],
        );
        gb.add_node(
            "gaudium_df_ctx",
            format!("ctx_{lp}kv"),
            &[kc, vc, kn_w, rope_c, rope_s, cursor, depth, fix],
            &[kcache, vcache],
            &[],
        );
    }
}

/// One drafter layer over the block: residual `x {hid, block}` -> the layer's output residual.
#[allow(clippy::too_many_arguments)]
fn append_layer(
    gb: &mut GraphBuilder,
    dc: &DFlash2Cfg,
    x: TensorId,
    l: usize,
    cur_blk: TensorId,
    fix: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
) -> TensorId {
    let t = dc.block;
    let hid = dc.hidden;
    let inter = dc.intermediate;
    let r4 = (D_NQ / D_NKV) * t;
    let lp = format!("D{l}_");

    // ---- attention, between the two sides of its convolution ----
    let h = d_rmsnorm(gb, x, &format!("{lp}w_in"), &format!("{lp}h"), hid, t);
    let cf = conv_coef(gb, dc, h, &format!("{lp}W_ac"), &format!("{lp}acoef"), t);
    let hc = conv(
        gb,
        dc,
        h,
        cf,
        &format!("{lp}ac_base"),
        0,
        &format!("{lp}hc"),
        t,
    );
    let q = add_linear_bf16(gb, hc, &format!("{lp}W_q"), hid, D_QD, t, &format!("{lp}q"));
    let k = add_linear_bf16(
        gb,
        hc,
        &format!("{lp}W_k"),
        hid,
        D_KVD,
        t,
        &format!("{lp}k"),
    );
    let v = add_linear_bf16(
        gb,
        hc,
        &format!("{lp}W_v"),
        hid,
        D_KVD,
        t,
        &format!("{lp}v"),
    );
    let qn_w = gb.get_or_add_persistent_tensor(format!("{lp}qn_w"), SynDataType::Float, &[D_DH]);
    let kn_w = gb.get_or_add_persistent_tensor(format!("{lp}kn_w"), SynDataType::Float, &[D_DH]);
    let qb = gb.add_persistent_tensor(format!("{lp}qb"), SynDataType::Bf16, &[D_DH, r4, D_NKV]);
    let kblk =
        gb.add_persistent_tensor(format!("{lp}kblk"), SynDataType::Bf16, &[D_DH, 128, D_NKV]);
    let vblk =
        gb.add_persistent_tensor(format!("{lp}vblk"), SynDataType::Bf16, &[D_DH, 128, D_NKV]);
    gb.add_node(
        "gaudium_df_attn_prep",
        format!("{lp}attn_prep"),
        &[q, k, v, qn_w, kn_w, rope_c, rope_s, cur_blk],
        &[qb, kblk, vblk],
        &[],
    );
    let kcache = gb.get_or_add_persistent_tensor(
        format!("{lp}kcache"),
        SynDataType::Bf16,
        &[D_DH, lmax, D_NKV],
    );
    let vcache = gb.get_or_add_persistent_tensor(
        format!("{lp}vcache"),
        SynDataType::Bf16,
        &[D_DH, lmax, D_NKV],
    );
    let sc = gb.add_persistent_tensor(format!("{lp}sc"), SynDataType::Float, &[lmax, r4, D_NKV]);
    let sb = gb.add_persistent_tensor(format!("{lp}sb"), SynDataType::Float, &[128, r4, D_NKV]);
    gb.op_batch_gemm(format!("{lp}qk_ctx"), qb, kcache, sc, false, true);
    gb.op_batch_gemm(format!("{lp}qk_blk"), qb, kblk, sb, false, true);
    let pc = gb.add_persistent_tensor(format!("{lp}Pc"), SynDataType::Bf16, &[lmax, r4, D_NKV]);
    let pb = gb.add_persistent_tensor(format!("{lp}Pb"), SynDataType::Bf16, &[128, r4, D_NKV]);
    gb.add_node(
        "gaudium_df_softmax",
        format!("{lp}softmax"),
        &[sc, sb, cur_blk, fix],
        &[pc, pb],
        &u32_params(&[t as u32, dc.window as u32]),
    );
    let aoc = gb.add_persistent_tensor(format!("{lp}aoc"), SynDataType::Bf16, &[D_DH, r4, D_NKV]);
    let aob = gb.add_persistent_tensor(format!("{lp}aob"), SynDataType::Bf16, &[D_DH, r4, D_NKV]);
    let ao = gb.add_persistent_tensor(format!("{lp}ao"), SynDataType::Bf16, &[D_DH, r4, D_NKV]);
    gb.op_batch_gemm(format!("{lp}pv_ctx"), pc, vcache, aoc, false, false);
    gb.op_batch_gemm(format!("{lp}pv_blk"), pb, vblk, aob, false, false);
    gb.op_add_bf16(format!("{lp}ao_add"), aoc, aob, ao);
    let core = gb.add_persistent_tensor(format!("{lp}core"), SynDataType::Bf16, &[D_QD, t]);
    gb.add_node(
        "gaudium_df_post",
        format!("{lp}attn_post"),
        &[ao],
        &[core],
        &[],
    );
    let attn = add_linear_bf16(
        gb,
        core,
        &format!("{lp}W_o"),
        D_QD,
        hid,
        t,
        &format!("{lp}attn"),
    );
    let attn2 = conv(
        gb,
        dc,
        attn,
        cf,
        &format!("{lp}ac_base"),
        1,
        &format!("{lp}attn2"),
        t,
    );
    let x2 = gb.add_persistent_tensor(format!("{lp}x2"), SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16(format!("{lp}x2_add"), x, attn2, x2);

    // ---- SwiGLU MLP, between the two sides of its convolution ----
    let h2 = d_rmsnorm(gb, x2, &format!("{lp}w_post"), &format!("{lp}h2"), hid, t);
    let cm = conv_coef(gb, dc, h2, &format!("{lp}W_mc"), &format!("{lp}mcoef"), t);
    let h2c = conv(
        gb,
        dc,
        h2,
        cm,
        &format!("{lp}mc_base"),
        0,
        &format!("{lp}h2c"),
        t,
    );
    let gt = add_linear_bf16(
        gb,
        h2c,
        &format!("{lp}W_gate"),
        hid,
        inter,
        t,
        &format!("{lp}gate"),
    );
    let up = add_linear_bf16(
        gb,
        h2c,
        &format!("{lp}W_up"),
        hid,
        inter,
        t,
        &format!("{lp}up"),
    );
    let act = gb.add_persistent_tensor(format!("{lp}act"), SynDataType::Bf16, &[inter, t]);
    gb.op_silu_bf16(format!("{lp}silu"), gt, act);
    let m = gb.add_persistent_tensor(format!("{lp}m"), SynDataType::Bf16, &[inter, t]);
    gb.op_mult_bf16(format!("{lp}swiglu"), act, up, m);
    let mo = add_linear_bf16(
        gb,
        m,
        &format!("{lp}W_down"),
        inter,
        hid,
        t,
        &format!("{lp}mlp"),
    );
    let mo2 = conv(
        gb,
        dc,
        mo,
        cm,
        &format!("{lp}mc_base"),
        1,
        &format!("{lp}mlp2"),
        t,
    );
    let out = gb.add_persistent_tensor(format!("{lp}out"), SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16(format!("{lp}out_add"), x2, mo2, out);
    out
}

pub struct BlockOut {
    pub drafts: TensorId,
    pub alts: TensorId,
    pub alts2: TensorId,
}

/// The block `[anchor, mask x (block - 1)]` at positions `cur_blk + t` -> `block - 1` drafts.
#[allow(clippy::too_many_arguments)]
pub fn append_block_pass(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    dc: &DFlash2Cfg,
    ids_blk: TensorId,
    cur_blk: TensorId,
    fix: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
) -> BlockOut {
    let t = dc.block;
    let s = t - 1;
    let hid = dc.hidden;
    let vocab = spec.vocab;
    let table = gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, vocab]);
    let mut x = gb.add_persistent_tensor("blk_x0", SynDataType::Bf16, &[hid, t]);
    gb.op_gather_bf16("blk_embed", table, ids_blk, x, 1);
    for l in 0..dc.nl {
        x = append_layer(gb, dc, x, l, cur_blk, fix, rope_c, rope_s, lmax);
    }
    let hf = d_rmsnorm(gb, x, "df_w_final", "blk_hf", hid, t);

    let lm_name = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    // Candidates come from the draft vocabulary when one is set (`nvd` rows of the `lm_head`;
    // the top-16 row indices map back to token ids through `df_dv_ids`); the selector tables
    // are indexed by token id either way. A token outside the subset can only cost an acceptance.
    let dv = dc.nvd > 0;
    let logits = if dv {
        add_linear_bf16(gb, hf, DV_HEAD, hid, dc.nvd, t, "blk_logits")
    } else {
        add_linear_bf16(gb, hf, lm_name, hid, vocab, t, "blk_logits")
    };
    let pv = gb.add_persistent_tensor("blk_pv", SynDataType::Float, &[64, TOPK, TOPK_CH, t]);
    let pi = gb.add_persistent_tensor("blk_pi", SynDataType::Int32, &[64, TOPK, TOPK_CH, t]);
    gb.add_node("gaudium_df_topk1", "blk_topk1", &[logits], &[pv, pi], &[]);
    let cand_name = if dv { "blk_cand_l" } else { "blk_cand" };
    let mut cand = gb.add_persistent_tensor(cand_name, SynDataType::Int32, &[TOPK, t]);
    let unary = gb.add_persistent_tensor("blk_unary", SynDataType::Float, &[TOPK, t]);
    gb.add_node(
        "gaudium_df_topk2",
        "blk_topk2",
        &[pv, pi],
        &[cand, unary],
        &[],
    );
    if dv {
        cand = super::dspine::append_dv_map(gb, cand, &[TOPK, t], "blk_cand", DV_IDS, dc.nvd);
    }

    let hp = add_linear_bf16(gb, hf, "df_W_hp", hid, RANK, t, "blk_hproj");
    let pred = gb.get_or_add_persistent_tensor("df_pred", SynDataType::Bf16, &[RANK, vocab]);
    let succ = gb.get_or_add_persistent_tensor("df_succ", SynDataType::Bf16, &[RANK, vocab]);
    let scores = gb.add_persistent_tensor("blk_scores", SynDataType::Float, &[TOPK, TOPK, s]);
    gb.add_node(
        "gaudium_df_score",
        "blk_score",
        &[cand, unary, hp, pred, succ, ids_blk],
        &[scores],
        &[],
    );
    let drafts = gb.add_persistent_tensor("blk_drafts", SynDataType::Int32, &[s]);
    let alts = gb.add_persistent_tensor("blk_alts", SynDataType::Int32, &[s]);
    let alts2 = gb.add_persistent_tensor("blk_alts2", SynDataType::Int32, &[s]);
    gb.add_node(
        "gaudium_df_walk",
        "blk_walk",
        &[scores, cand],
        &[drafts, alts, alts2],
        &[],
    );
    BlockOut {
        drafts,
        alts,
        alts2,
    }
}

fn rope_tables(gb: &mut GraphBuilder, lmax: u64) -> (TensorId, TensorId) {
    let n = lmax + ROPE_PAD;
    (
        gb.get_or_add_persistent_tensor("rope128_c", SynDataType::Float, &[128, n]),
        gb.get_or_add_persistent_tensor("rope128_s", SynDataType::Float, &[128, n]),
    )
}

/// The drafter section of a step recipe, after the accept: context update for the `t` verified
/// rows, the next block anchored on the bonus token at `cursor_out`, and the next step's ids.
#[allow(clippy::too_many_arguments)]
pub fn append_step_drafter(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    dc: &DFlash2Cfg,
    taps: &[TensorId],
    t: u64,
    cursor: TensorId,
    depth: TensorId,
    tokens: TensorId,
    a_out: TensorId,
    cursor_out: TensorId,
    lmax: u64,
) {
    let (rc, rs) = rope_tables(gb, lmax);
    let nofix = gb.get_or_add_persistent_tensor("df_nofix", SynDataType::Int32, &[2]);
    append_ctx_pass(gb, dc, taps, t, cursor, depth, nofix, rc, rs, lmax);
    let ids_blk = gb.add_persistent_tensor("df_ids_blk", SynDataType::Int32, &[dc.block]);
    gb.add_node(
        "gaudium_df_ids",
        "df_ids",
        &[tokens, a_out],
        &[ids_blk],
        &u32_params(&[dc.mask_id as u32]),
    );
    let out = append_block_pass(gb, spec, dc, ids_blk, cursor_out, nofix, rc, rs, lmax);
    let next_ids = gb.add_persistent_tensor("next_ids", SynDataType::Int32, &[dc.block]);
    gb.add_node(
        "gaudium_df_next",
        "next_ids",
        &[tokens, a_out, out.drafts, out.alts, out.alts2],
        &[next_ids],
        &u32_params(&[(dc.block - 1) as u32]),
    );
}

/// The speculative step recipe with this drafter: the verify + accept graph of the core, then the
/// drafter section on the step's own tensors (taps = the verify rows' layer outputs).
pub fn build_step_graph(
    spec: &ModelSpec,
    dc: &DFlash2Cfg,
    lmax: u64,
    nch: u64,
    t: u64,
    ns: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    build_step_graph_as(spec, dc, lmax, nch, t, ns, elf_hashes, false)
}

/// [`build_step_graph`], or (`sampled`) the step of a sampled request: the verify rows sample on
/// the card at their own positions (`g_spec_df2s_*`); the drafts stay the drafter's greedy walk.
#[allow(clippy::too_many_arguments)]
pub fn build_step_graph_as(
    spec: &ModelSpec,
    dc: &DFlash2Cfg,
    lmax: u64,
    nch: u64,
    t: u64,
    ns: u64,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    let mut gb = super::build_spec_step_graph_as(spec, lmax, nch, t, ns, None, elf_hashes, sampled);
    gb.set_recipe_name(if sampled {
        format!("g_spec_df2s_T{t}_C{nch}")
    } else {
        format!("g_spec_df2_T{t}_C{nch}")
    });
    let i32s = SynDataType::Int32;
    let taps: Vec<TensorId> = dc
        .taps
        .iter()
        .map(|&l| {
            gb.get_or_add_persistent_tensor(
                format!("L{l}_out"),
                SynDataType::Bf16,
                &[spec.hidden, t],
            )
        })
        .collect();
    let cursor = gb.get_or_add_persistent_tensor("cursor", i32s, &[1]);
    let depth = gb.get_or_add_persistent_tensor("sp_depth", i32s, &[t]);
    let tokens = gb.get_or_add_persistent_tensor("tokens", i32s, &[1, t]);
    let a_out = gb.get_or_add_persistent_tensor("a_out", i32s, &[1]);
    let cursor_out = gb.get_or_add_persistent_tensor("cursor_out", i32s, &[1]);
    append_step_drafter(
        &mut gb, spec, dc, &taps, t, cursor, depth, tokens, a_out, cursor_out, lmax,
    );
    gb
}

/// Drafter context for a prefill tile of `t` rows (`df_tap{i}` = the tile's tap outputs).
pub fn build_ctx_graph(
    spec: &ModelSpec,
    dc: &DFlash2Cfg,
    t: u64,
    lmax: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_spec_df2ctx_T{t}"));
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let hs: Vec<TensorId> = (0..dc.taps.len())
        .map(|i| {
            gb.add_persistent_tensor(format!("df_tap{i}"), SynDataType::Bf16, &[spec.hidden, t])
        })
        .collect();
    let cursor = gb.add_persistent_tensor("ctx_cursor", SynDataType::Int32, &[1]);
    let depth = gb.get_or_add_persistent_tensor("sp_depth", SynDataType::Int32, &[t]);
    let nofix = gb.get_or_add_persistent_tensor("df_nofix", SynDataType::Int32, &[2]);
    let (rc, rs) = rope_tables(&mut gb, lmax);
    append_ctx_pass(&mut gb, dc, &hs, t, cursor, depth, nofix, rc, rs, lmax);
    gb
}

/// The first draft block after the prefill (anchor = the prefill's token at position `P`).
pub fn build_block0_graph(
    spec: &ModelSpec,
    dc: &DFlash2Cfg,
    lmax: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = GraphBuilder::new("g_spec_df2blk0");
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let ids_blk = gb.add_persistent_tensor("df_ids_pre", SynDataType::Int32, &[dc.block]);
    let cur_blk = gb.add_persistent_tensor("df_cur_pre", SynDataType::Int32, &[1]);
    let nofix = gb.get_or_add_persistent_tensor("df_nofix", SynDataType::Int32, &[2]);
    let (rc, rs) = rope_tables(&mut gb, lmax);
    let _ = append_block_pass(&mut gb, spec, dc, ids_blk, cur_blk, nofix, rc, rs, lmax);
    gb
}

// ============================================================================
// Loaded drafter
// ============================================================================

/// A DFlash2 checkpoint opened for one target model: config, mapped weights, and its own recipes.
pub struct DFlash2 {
    pub cfg: DFlash2Cfg,
    store: SafetensorsStore,
    ctx_recipes: HashMap<u64, CompiledRecipe>,
    blk0: Option<CompiledRecipe>,
    blk0_launch: Option<PreparedLaunch>,
    pool: BTreeMap<String, u64>,
    /// Rows of the eagerly compiled context recipe (the engine's prefill tile).
    full_tile: u64,
}

impl DFlash2 {
    pub fn open(dir: &Path, spec: &ModelSpec) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("read {}/config.json: {e}", dir.display()))?;
        let cfg = DFlash2Cfg::from_json(&text)?;
        cfg.check_target(spec)?;
        let store = SafetensorsStore::open(dir)
            .map_err(|e| format!("open drafter safetensors in {}: {e}", dir.display()))?;
        Ok(Self {
            cfg,
            store,
            ctx_recipes: HashMap::new(),
            blk0: None,
            blk0_launch: None,
            pool: BTreeMap::new(),
            full_tile: 0,
        })
    }

    fn need(&self, name: &str, shape: &[u64]) -> Result<&[u8], String> {
        let meta = self
            .store
            .get(name)
            .map_err(|e| format!("drafter tensor '{name}': {e}"))?;
        if meta.dtype != "BF16" || meta.shape != shape {
            return Err(format!(
                "drafter tensor '{name}' is {} {:?}, expected BF16 {shape:?}",
                meta.dtype, meta.shape
            ));
        }
        self.store
            .tensor_bytes(name)
            .map_err(|e| format!("drafter tensor '{name}': {e}"))?
            .ok_or_else(|| format!("drafter tensor '{name}' is not mapped"))
    }

    /// Device bytes of one drafter weight tensor.
    fn pack(&self, name: &str, lmax: u64) -> Result<Cow<'_, [u8]>, String> {
        let c = &self.cfg;
        let (hid, inter, vocab) = (c.hidden, c.intermediate, c.vocab);
        let raw = |n: &str, shape: &[u64]| self.need(n, shape).map(Cow::Borrowed);
        let sel = "candidate_selector.";
        match name {
            "df_W_fc" => return raw("fc.weight", &[hid, hid * c.taps.len() as u64]),
            "df_w_hn" => return raw("hidden_norm.weight", &[hid]),
            "df_w_final" => return raw("norm.weight", &[hid]),
            "df_W_hp" => return raw(&format!("{sel}hidden_projection.weight"), &[RANK, hid]),
            "df_pred" => return raw(&format!("{sel}predecessor_codebook"), &[vocab, RANK]),
            "df_succ" => return raw(&format!("{sel}successor_codebook"), &[vocab, RANK]),
            "df_nofix" => {
                return Ok(Cow::Owned(
                    [-1i32, -1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                ))
            }
            "rope128_c" | "rope128_s" => {
                let (rc, rs) =
                    packer_dflash2::rope128_tables(c.rope_theta, (lmax + ROPE_PAD) as usize);
                let t = if name == "rope128_c" { rc } else { rs };
                return Ok(Cow::Owned(packer_dflash2::f32_bytes(&t)));
            }
            _ => {}
        }
        let Some((l, what)) = drafter_layer_suffix(name) else {
            return Err(format!("dflash2: no packing rule for '{name}'"));
        };
        let lp = format!("layers.{l}.");
        let lane = |n: &str| {
            self.need(n, &[D_DH])
                .map(|b| Cow::Owned(packer_dflash2::lane_f32(b)))
        };
        let base = |conv: &str, side: usize| {
            self.need(&format!("{lp}{conv}.base_kernel"), &[2, CONV_TAPS, hid])
                .map(|b| Cow::Owned(packer_dflash2::conv_base(b, side, hid as usize)))
        };
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
            "W_ac" => raw(
                &format!("{lp}attention_conv.kernel_projection.weight"),
                &[c.kp, hid],
            ),
            "W_mc" => raw(
                &format!("{lp}mlp_conv.kernel_projection.weight"),
                &[c.kp, hid],
            ),
            "qn_w" => lane(&format!("{lp}self_attn.q_norm.weight")),
            "kn_w" => lane(&format!("{lp}self_attn.k_norm.weight")),
            "ac_base0" => base("attention_conv", 0),
            "ac_base1" => base("attention_conv", 1),
            "mc_base0" => base("mlp_conv", 0),
            "mc_base1" => base("mlp_conv", 1),
            _ => Err(format!("dflash2: no packing rule for '{name}'")),
        }
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
        let blk0 = build_block0_graph(spec, &self.cfg, lmax, elf_hashes)
            .compile(Arc::clone(api), cache, driver_version)
            .map_err(|e| format!("compile dflash2 block0: {e}"))?;
        let ctx = build_ctx_graph(spec, &self.cfg, prefill_tile, lmax, elf_hashes)
            .compile(Arc::clone(api), cache, driver_version)
            .map_err(|e| format!("compile dflash2 ctx T={prefill_tile}: {e}"))?;
        if verbose {
            for r in [&blk0, &ctx] {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                    r.name,
                    r.from_cache,
                    r.ws_size as f64 / (1024.0 * 1024.0)
                );
            }
        }
        let ws = blk0.ws_size.max(ctx.ws_size);
        self.blk0 = Some(blk0);
        self.ctx_recipes.insert(prefill_tile, ctx);
        self.full_tile = prefill_tile;
        Ok(ws)
    }

    /// After the seal: uploads the draft vocabulary of the block readout (the `lm_head` rows of
    /// [`super::mtp::draft_vocab_ids`] and the row -> token id table), when one is set.
    pub fn upload_draft_vocab(
        &self,
        dev: &HpuDevice,
        spec: &ModelSpec,
        packer: &WeightPacker<'_>,
        stage: &mut PinnedBuf,
        addr: &BTreeMap<String, u64>,
    ) -> Result<(), String> {
        if self.cfg.nvd == 0 {
            return Ok(());
        }
        let ids = super::mtp::draft_vocab_ids(spec, self.cfg.nvd);
        let full = packer.pack_weight(if spec.tie_word_embeddings {
            "embed_table"
        } else {
            "W_lm"
        })?;
        let row = (spec.hidden * 2) as usize;
        let mut rows = Vec::with_capacity(ids.len() * row);
        for &id in &ids {
            let o = id as usize * row;
            rows.extend_from_slice(&full[o..o + row]);
        }
        dev.upload_h2d_chunked(&rows, addr[DV_HEAD], stage)
            .map_err(|e| format!("upload {DV_HEAD}: {e}"))?;
        let idb: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
        dev.upload_h2d_chunked(&idb, addr[DV_IDS], stage)
            .map_err(|e| format!("upload {DV_IDS}: {e}"))
    }

    /// After the seal: uploads the drafter weights and prepares its launches.
    /// `pool` = every address a drafter recipe may bind (target weights + this module's buffers).
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &mut self,
        dev: &HpuDevice,
        spec: &ModelSpec,
        stage: &mut PinnedBuf,
        weight_specs: &BTreeMap<String, u64>,
        addr: &BTreeMap<String, u64>,
        pool: &BTreeMap<String, u64>,
        lmax: u64,
        verbose: bool,
    ) -> Result<(), String> {
        let t0 = Instant::now();
        let mut uploaded = 0u64;
        for (name, &bytes) in weight_specs {
            if !is_drafter_weight(name) {
                continue;
            }
            let data = self.pack(name, lmax)?;
            if data.len() as u64 != bytes {
                return Err(format!(
                    "drafter weight '{name}' packed size {} != graph tensor size {bytes}",
                    data.len()
                ));
            }
            dev.upload_h2d_chunked(&data, addr[name], stage)
                .map_err(|e| format!("upload drafter weight '{name}': {e}"))?;
            uploaded += bytes;
        }
        if verbose {
            eprintln!(
                "[gaudium] dflash2: uploaded {:.2} GiB drafter weights + tables in {:.2}s",
                uploaded as f64 / (1u64 << 30) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        self.pool = pool.clone();
        // The tap buffers start as zeros: a full-tile context pass over a short tile must only
        // ever read finite rows (see `launch_ctx`).
        let tap_bytes = (spec.hidden * self.full_tile * 2) as usize;
        let zeros = vec![0u8; tap_bytes];
        for i in 0..self.cfg.taps.len() {
            dev.upload_h2d_chunked(&zeros, addr[&format!("df_tap{i}")], stage)
                .map_err(|e| format!("zero drafter tap {i}: {e}"))?;
        }
        let mut blk0 = self.blk0.take().expect("block0 compiled before the seal");
        blk0.bind_shared_workspace(dev)
            .map_err(|e| format!("bind dflash2 block0 ws: {e}"))?;
        let exact = select_exact_bindings(&blk0, &self.pool)?;
        self.blk0_launch = Some(
            blk0.prepare_launch(&exact)
                .map_err(|e| format!("prepare dflash2 block0: {e}"))?,
        );
        self.blk0 = Some(blk0);
        for r in self.ctx_recipes.values_mut() {
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind dflash2 ctx ws: {e}"))?;
        }
        Ok(())
    }

    /// Prefill routing of the target-layer taps: a tapped layer writes its output to the tap
    /// buffer and the next layer reads it from there.
    pub fn prefill_io(
        &self,
        addr: &BTreeMap<String, u64>,
        l: usize,
        xin: u64,
        xout: u64,
    ) -> (u64, u64) {
        let tap = |layer: usize| {
            self.cfg
                .taps
                .iter()
                .position(|&t| t == layer)
                .map(|i| addr[&format!("df_tap{i}")])
        };
        let xin = if l > 0 {
            tap(l - 1).unwrap_or(xin)
        } else {
            xin
        };
        (xin, tap(l).unwrap_or(xout))
    }

    /// Launches the context pass for the prefill tile of `tile_len` rows at position `pos` (taps
    /// already in place).
    ///
    /// A short last tile runs through the full-tile recipe when its rows fit below the context
    /// end: the rows past `tile_len` are older (finite) tap rows, and the cache slots they write
    /// lie at positions the prompt has not reached, which the block attention masks and the decode
    /// steps overwrite before they become visible. That keeps one context recipe per engine
    /// instead of one per prompt length.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_ctx(
        &mut self,
        dev: &HpuDevice,
        spec: &ModelSpec,
        cache: &RecipeCache,
        driver_version: &str,
        elf_hashes: &[(String, String)],
        lmax: u64,
        pos: u64,
        tile_len: u64,
    ) -> Result<(), String> {
        let tile_len = if pos + self.full_tile <= lmax {
            self.full_tile
        } else {
            tile_len
        };
        if !self.ctx_recipes.contains_key(&tile_len) {
            let mut r = build_ctx_graph(spec, &self.cfg, tile_len, lmax, elf_hashes)
                .compile(Arc::clone(dev.api()), cache, driver_version)
                .map_err(|e| format!("compile dflash2 ctx T={tile_len}: {e}"))?;
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind dflash2 ctx T={tile_len} ws: {e}"))?;
            self.ctx_recipes.insert(tile_len, r);
        }
        let r = &self.ctx_recipes[&tile_len];
        let exact = select_exact_bindings(r, &self.pool)?;
        r.launch_checked(dev, &exact)
            .map_err(|e| format!("launch dflash2 ctx T={tile_len}: {e}"))
    }

    /// Drafts the first block: anchor `t0` at position `cur`. Returns `block - 1` drafts.
    pub fn launch_block0(
        &mut self,
        dev: &HpuDevice,
        stage_io: &mut PinnedBuf,
        t0: i32,
        cur: i32,
    ) -> Result<Vec<i32>, String> {
        let blk = self.cfg.block as usize;
        let buf = stage_io.as_mut_slice();
        for j in 0..blk {
            let v = if j == 0 { t0 } else { self.cfg.mask_id };
            buf[j * 4..j * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let cur_off = (blk * 4).next_multiple_of(64);
        buf[cur_off..cur_off + 4].copy_from_slice(&cur.to_le_bytes());
        dev.copy_h2d_pinned_batch_sync(
            stage_io,
            &[
                (0, blk * 4, self.pool["df_ids_pre"]),
                (cur_off, 4, self.pool["df_cur_pre"]),
            ],
        )
        .map_err(|e| format!("dflash2 block0 H2D: {e}"))?;
        let r = self.blk0.as_ref().expect("block0 recipe");
        r.launch_prepared(dev, self.blk0_launch.as_ref().expect("block0 launch"))
            .map_err(|e| format!("launch dflash2 block0: {e}"))?;
        unsafe { dev.sync_stream(dev.compute_stream()) }
            .map_err(|e| format!("sync dflash2 block0: {e}"))?;
        let n = blk - 1;
        dev.copy_d2h_pinned_sync(self.pool["blk_drafts"], stage_io, n * 4)
            .map_err(|e| format!("dflash2 block0 D2H: {e}"))?;
        Ok(stage_io.as_slice()[..n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG_27B: &str = r#"{
      "architectures": ["DFlash2DraftModel"], "is_causal": false,
      "dflash_config": {
        "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248070,
        "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 19, 33, 47, 61]
      },
      "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
      "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                      "sliding_attention", "sliding_attention"],
      "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
      "num_target_layers": 64, "rms_norm_eps": 1e-06,
      "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
      "sliding_window": 2048, "tie_word_embeddings": false, "use_sliding_window": true,
      "vocab_size": 248320
    }"#;

    fn target_27b() -> ModelSpec {
        ModelSpec::from_config_json(
            r#"{"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": {"model_type": "qwen3_5_text", "hidden_size": 5120,
                "intermediate_size": 17408, "num_hidden_layers": 64, "num_attention_heads": 24,
                "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                "rms_norm_eps": 1e-6, "full_attention_interval": 4, "linear_num_key_heads": 16,
                "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn config_contract_is_read_and_checked() {
        let c = DFlash2Cfg::from_json(CFG_27B).unwrap();
        assert_eq!(c.block, 8);
        assert_eq!(c.nl, 5);
        assert_eq!(c.kp, 1280);
        assert_eq!(c.mask_id, 248070);
        assert_eq!(c.window, 2048);
        assert_eq!(c.taps, vec![5, 19, 33, 47, 61]);
        c.check_target(&target_27b()).unwrap();

        // a checkpoint that turns the window off attends to the whole context
        let full = CFG_27B.replace(
            "\"use_sliding_window\": true",
            "\"use_sliding_window\": false",
        );
        assert!(DFlash2Cfg::from_json(&full).unwrap().window > 1 << 20);
        // anything the device graph does not implement fails loudly
        for (from, to) in [
            ("\"block_size\": 8", "\"block_size\": 16"),
            ("\"conv_group_size\": 16", "\"conv_group_size\": 8"),
            ("\"conv_kernel_size\": 2", "\"conv_kernel_size\": 3"),
            ("\"selector_rank\": 256", "\"selector_rank\": 128"),
            ("\"selector_top_k\": 16", "\"selector_top_k\": 8"),
            ("\"is_causal\": false", "\"is_causal\": true"),
            ("\"head_dim\": 128", "\"head_dim\": 256"),
            (
                "\"sliding_attention\", \"sliding_attention\",\n",
                "\"full_attention\", \"sliding_attention\",\n",
            ),
        ] {
            assert!(CFG_27B.contains(from), "{from}");
            assert!(
                DFlash2Cfg::from_json(&CFG_27B.replace(from, to)).is_err(),
                "{from}"
            );
        }
        // a block drafter without convolutions and selector is another drafter
        let plain = CFG_27B
            .replace("\"conv_group_size\": 16, ", "")
            .replace("\"selector_rank\": 256, ", "");
        assert!(DFlash2Cfg::from_json(&plain).is_err());
        // a drafter for another target is refused
        let other = CFG_27B.replace("\"hidden_size\": 5120", "\"hidden_size\": 4096");
        assert!(DFlash2Cfg::from_json(&other)
            .unwrap()
            .check_target(&target_27b())
            .is_err());
    }

    #[test]
    fn tensor_names_are_classified_for_the_device_plan() {
        assert!(is_drafter_kv("D0_kcache") && is_drafter_kv("D4_vcache"));
        assert!(!is_drafter_kv("L3_kcache") && !is_drafter_kv("D0_qb"));
        for w in [
            "df_W_fc",
            "df_w_final",
            "df_W_hp",
            "df_pred",
            "df_succ",
            "D2_W_gate",
            "D0_W_ac",
            "D4_W_mc",
            "D1_ac_base0",
            "D3_mc_base1",
            "D0_qn_w",
            "rope128_s",
        ] {
            assert!(is_drafter_weight(w), "{w}");
        }
        for a in [
            "D0_q",
            "D0_sc",
            "D0_acoef",
            "D0_acoef_bf",
            "D0_hc",
            "blk_drafts",
            "blk_hf",
            "ctx_cat",
            "L0_W_o",
            "D0_kcache",
        ] {
            assert!(!is_drafter_weight(a), "{a}");
        }
    }

    #[test]
    fn step_graph_appends_the_drafter_to_the_verify_core() {
        let spec = target_27b();
        let dc = DFlash2Cfg::from_json(CFG_27B).unwrap();
        let gb = build_step_graph(&spec, &dc, 4096, 1, 6, 8, &[]);
        let specs = gb.persistent_tensor_specs();
        let count = |n: &str| specs.iter().filter(|t| t.0 == n).count();
        // the drafter binds the verify core's own tensors (no second tensor of the same name)
        for n in [
            "L5_out",
            "L61_out",
            "cursor",
            "cursor_out",
            "tokens",
            "a_out",
            "sp_depth",
            "embed_table",
            "W_lm",
            "D0_kcache",
            "D4_vcache",
            "D0_W_k",
        ] {
            assert_eq!(count(n), 1, "{n}");
        }
        let dims = |n: &str| specs.iter().find(|t| t.0 == n).map(|t| t.2.clone());
        // a 6-row verify still drafts the whole block
        assert_eq!(dims("next_ids"), Some(vec![8]));
        assert_eq!(dims("ctx_cat"), Some(vec![5 * 5120, 6]));
        assert_eq!(dims("D0_acoef"), Some(vec![1280, 8]));
        assert_eq!(dims("D0_ac_base1"), Some(vec![5120, 2]));
        assert_eq!(dims("blk_scores"), Some(vec![16, 16, 7]));
        assert_eq!(dims("df_pred"), Some(vec![256, 248320]));
        assert_eq!(dims("rope128_c"), Some(vec![128, 4096 + ROPE_PAD]));
        assert_eq!(dims("D0_Pc"), Some(vec![4096, 32, 8]));
        // a recipe of its own, distinct from the verify-only step of the same shape
        let vf = super::super::build_spec_step_graph(&spec, 4096, 1, 6, 8, None, &[]);
        assert_ne!(gb.fingerprint("x"), vf.fingerprint("x"));
        assert_eq!(
            gb.fingerprint("x"),
            build_step_graph(&spec, &dc, 4096, 1, 6, 8, &[]).fingerprint("x")
        );
    }

    #[test]
    fn every_weight_of_the_graphs_has_a_packing_rule_name() {
        let spec = target_27b();
        let dc = DFlash2Cfg::from_json(CFG_27B).unwrap();
        let mut names = std::collections::BTreeSet::new();
        for gb in [
            build_step_graph(&spec, &dc, 4096, 1, 8, 8, &[]),
            build_ctx_graph(&spec, &dc, 128, 4096, &[]),
            build_block0_graph(&spec, &dc, 4096, &[]),
        ] {
            for (name, ..) in gb.persistent_tensor_specs() {
                if is_drafter_weight(&name) {
                    names.insert(name);
                }
            }
        }
        // 9 global tensors + 17 per layer
        assert_eq!(names.len(), 9 + 17 * 5);
    }
}
