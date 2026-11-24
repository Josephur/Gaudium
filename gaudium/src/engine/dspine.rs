//! DSpine block drafter on the card (checkpoints of the `DFlashDraftModel` family with a
//! selected-prefix code bridge and a serial exact-sender refinement).
//!
//! What one draft block computes, for an anchor token at position `p` and `block - 1` mask rows:
//! - **context** (incremental): per drafter layer, K/V of `hidden_norm(fc(concat(target taps)))`
//!   for the target rows the verify just produced, written to the layer's context cache;
//! - **block pass**: `nl` plain Qwen3 layers over the block (attention over the context cache and
//!   causally over the block), each followed by the code bridge (`gaudium_ds_read` -> read GEMM ->
//!   `gaudium_ds_codes` -> gate/predecessor/value GEMM -> `gaudium_ds_msg` -> write GEMM ->
//!   `gaudium_ds_finish`, which also applies the next layer's input norm);
//! - **tail**: the target's `lm_head` on the final rows, top-16 per row, the serial table that
//!   redoes the last bridge write with the exact code of the actual predecessor candidate
//!   (`gaudium_ds_base`, `gaudium_ds_score` over the tables TG / TSV), and the greedy lattice walk.
//!
//! The drafter only proposes: the verify decides every token, so nothing here affects exactness.
//!
//! Vision: the context pass reads only the target's tapped residual rows, so an image position
//! contributes what the target computed from its vision embedding (never a pad-token
//! embedding); the block rows are generated tokens and masks. Like DFlash2's, the drafter's RoPE
//! rows are addressed and rotated by KV slot. Checked by the vision regression suite
//! (`--vision-suite`).
//!
//! Geometry (block size, taps, mask id, code size, rank, layers, window) comes from the
//! checkpoint's `config.json`; anything the device graph does not implement fails at load.

use std::borrow::Cow;
use std::path::Path;

use super::super::*;
use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};
use crate::json::{parse_json, JsonValue};
use crate::synapse::recipe_cache::Sha256;
use crate::verified::{ds_ctx_rows, ds_draft_mapped, ds_draft_rows, ds_ring_pick, ds_ring_rows, lane256};

#[path = "packer_dspine.rs"]
mod packer_dspine;
pub use packer_dspine::build_serial_tables;

/// Kernel ELFs of the drafter passes.
pub const DSPINE_ELFS: &[&str] = &[
    "GAUDIUM_DFAPREP_ELF",
    "GAUDIUM_DFCTX_ELF",
    "GAUDIUM_DSSOFT_ELF",
    "GAUDIUM_DFPOST_ELF",
    "GAUDIUM_DFTOPK1_ELF",
    "GAUDIUM_DFTOPK2_ELF",
    "GAUDIUM_DFWALK_ELF",
    "GAUDIUM_DFIDS_ELF",
    "GAUDIUM_DFNEXT_ELF",
    "GAUDIUM_DSREAD_ELF",
    "GAUDIUM_DSCODES_ELF",
    "GAUDIUM_DSMSG_ELF",
    "GAUDIUM_DSFINISH_ELF",
    "GAUDIUM_DSBASE_ELF",
    "GAUDIUM_DSSCORE_ELF",
];

/// Kernel ELFs of the ring form of the context caches ([`DSpineCfg::ring`]), served by the step
/// glue library.
pub const DSPINE_RING_ELFS: &[&str] = &["GAUDIUM_DSCTXR_ELF", "GAUDIUM_DSSOFTR_ELF"];

/// Slots a context ring keeps beyond a window and the rows of one context pass: the softmax
/// kernel sweeps 128-aligned slot ranges, and the two ranges of a wrapped window must not meet.
pub const RING_GUARD: u64 = 256;

/// Rows of a context ring for a sliding window of `window` positions when one context pass
/// writes at most `pass_rows` rows: a whole number of 128-row blocks holding the window, the
/// rows a pass may write past the cursor, and the guard.
pub fn ring_rows(window: u64, pass_rows: u64) -> u64 {
    ds_ring_rows(window, pass_rows, RING_GUARD)
}

/// Drafter attention shape fixed by the shared block-attention kernels: 32 query / 8 KV heads x 128.
const D_DH: u64 = 128;
const D_NKV: u64 = 8;
const D_QD: u64 = 4096;
const D_KVD: u64 = 1024;
const TOPK: u64 = 16;
/// Vocabulary chunks of the two-stage top-16 kernels.
const TOPK_CH: u64 = 32;
/// Rows the RoPE tables carry past the context so a block starting near its end stays in range.
pub const ROPE_PAD: u64 = 256;

/// The checkpoint contract read from `<draft dir>/config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct DSpineCfg {
    /// Block rows: the anchor plus `block - 1` drafts.
    pub block: u64,
    pub nl: usize,
    pub code_size: u64,
    pub rank: u64,
    pub mask_id: i32,
    /// Sliding window of the context attention in positions (`u32::MAX / 4` = unbounded).
    pub window: u64,
    pub rope_theta: f64,
    /// Target layers whose output residual feeds the drafter's context.
    pub taps: Vec<usize>,
    pub hidden: u64,
    pub intermediate: u64,
    pub vocab: u64,
    /// Run option, not part of the checkpoint: rows of the block readout's draft vocabulary
    /// (`ds_W_lmd`, a copy of those `lm_head` rows; 0 = the full `lm_head` as it is, BF16;
    /// the vocabulary's size = every row, as the FP8 copy of an FP8 readout).
    pub nvd: u64,
    /// Run option: rows of the context rings (0 = linear caches of the engine's context size).
    /// The context attention reads at most `window` positions back, so a ring of
    /// [`ring_rows`] slots (slot = position mod rows) holds everything a block can see: the
    /// caches, the score / probability tensors and the two context GEMMs of every layer then
    /// have the window's size instead of the engine's context size.
    pub ring: u64,
    /// Run option: the layers' weight GEMMs run as FP8 (`<W>_f8` codes with one power-of-two
    /// scale per output channel, one static cast per input site). Drafts only: the verify, the
    /// context pass and the bridge stay BF16.
    pub fp8: bool,
    /// Run option: the draft-vocabulary readout is an FP8 GEMM too (`ds_W_lmd_f8`, the cast of
    /// the final hidden rows as site `D<nl>_xin`). Needs `nvd > 0`, `fp8` and that site's scale.
    pub fp8_head: bool,
}

impl DSpineCfg {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let j = parse_json(text).map_err(|e| format!("dspine config: {e}"))?;
        let fail = |m: &str| format!("dspine config: {m}");
        let num = |o: &JsonValue, k: &str| -> Result<u64, String> {
            o.get(k)
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| format!("dspine config: missing integer '{k}'"))
        };
        let dc = j
            .get("dflash_config")
            .ok_or_else(|| fail("missing dflash_config"))?;
        let br = dc
            .get("selected_prefix_code_bridge")
            .ok_or_else(|| fail("missing dflash_config.selected_prefix_code_bridge"))?;

        let block = num(&j, "block_size")?;
        let nl = num(&j, "num_hidden_layers")? as usize;
        let taps: Vec<usize> = dc
            .get("target_layer_ids")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| fail("missing dflash_config.target_layer_ids"))?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let window = j
            .get("sliding_window")
            .and_then(JsonValue::as_u64)
            .unwrap_or((u32::MAX / 4) as u64);
        let rope_theta = j
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| j.get("rope_theta"))
            .and_then(JsonValue::as_f64)
            .ok_or_else(|| fail("missing rope_theta"))?;

        // The serial last-write refinement may be named by either key; both must agree.
        let mut topk = 0u64;
        if let Some(sp) = j.get("code_bridge_second_pass") {
            topk = num(sp, "head_topk")?;
            if sp.get("mode").and_then(JsonValue::as_str) != Some("serial_table")
                || num(sp, "first_exact")? as usize != nl - 1
            {
                return Err(fail(
                    "code_bridge_second_pass must be serial_table redoing the last write",
                ));
            }
        }
        if let Some(lr) = dc.get("selected_prefix_last_write_refine") {
            let k2 = num(lr, "head_topk")?;
            if lr.get("kind").and_then(JsonValue::as_str) != Some("exact_sender") {
                return Err(fail(
                    "selected_prefix_last_write_refine must be kind exact_sender",
                ));
            }
            if topk != 0 && k2 != topk {
                return Err(fail("the two refinement widths disagree"));
            }
            topk = k2;
        }
        if topk != TOPK {
            return Err(fail(
                "refinement head_topk must be 16 (16-lane top-k / score / walk kernels)",
            ));
        }

        let s = |o: &JsonValue, k: &str| o.get(k).and_then(JsonValue::as_str).map(str::to_owned);
        let b = |o: &JsonValue, k: &str| o.get(k).and_then(JsonValue::as_bool);
        if s(dc, "selected_prefix_decode_mode").as_deref() != Some("serial")
            || b(dc, "draft_block_causal") != Some(true)
            || b(dc, "selected_prefix_terminal") != Some(false)
            || b(dc, "selected_prefix_feature_writes") != Some(true)
            || s(dc, "selected_prefix_feature_write_mode").as_deref() != Some("independent")
        {
            return Err(fail(
                "unsupported bridge mode (need serial decode, causal block, independent writes, terminal off)",
            ));
        }
        let writers: Vec<u64> = dc
            .get("selected_prefix_feature_layers")
            .and_then(JsonValue::as_array)
            .map(|a| a.iter().filter_map(JsonValue::as_u64).collect())
            .unwrap_or_default();
        if writers != (1..=nl as u64).collect::<Vec<_>>() {
            return Err(fail("every draft layer must write through the bridge"));
        }
        if num(&j, "num_attention_heads")? != 32
            || num(&j, "num_key_value_heads")? != D_NKV
            || num(&j, "head_dim")? != D_DH
        {
            return Err(fail("drafter attention must be 32 / 8 heads x 128"));
        }
        let eps = j
            .get("rms_norm_eps")
            .and_then(JsonValue::as_f64)
            .unwrap_or(0.0);
        let beps = br.get("eps").and_then(JsonValue::as_f64).unwrap_or(0.0);
        if (eps - 1e-6).abs() > 1e-12 || (beps - 1e-6).abs() > 1e-12 {
            return Err(fail("eps must be 1e-6 (fixed in the kernels)"));
        }
        let cfg = Self {
            block,
            nl,
            code_size: num(br, "code_size")?,
            rank: num(br, "rank")?,
            mask_id: num(dc, "mask_token_id")? as i32,
            window,
            rope_theta,
            taps,
            hidden: num(&j, "hidden_size")?,
            intermediate: num(&j, "intermediate_size")?,
            vocab: num(&j, "vocab_size")?,
            nvd: 0,
            ring: 0,
            fp8: false,
            fp8_head: false,
        };
        if !(2..=32).contains(&cfg.block) {
            return Err(fail("block_size outside 2..=32"));
        }
        if cfg.taps.is_empty()
            || !cfg.code_size.is_multiple_of(128)
            || !cfg.rank.is_multiple_of(128)
            || cfg.rank > 512
            || cfg.nl == 0
        {
            return Err(fail(
                "taps / code_size / rank layout (code_size and rank multiples of 128, rank <= 512)",
            ));
        }
        Ok(cfg)
    }

    /// Rows of a context cache in an engine of `lmax` context positions: the ring's when the
    /// context is kept as rings, else one row per position.
    pub fn ctx_rows(&self, lmax: u64) -> u64 {
        ds_ctx_rows(self.ring, lmax)
    }

    /// Chooses the form of the context caches for an engine of `lmax` positions whose context
    /// passes write at most `pass_rows` rows: rings when the checkpoint's window is bounded, the
    /// ring kernels are there (`have_ring`) and a ring is smaller than a linear cache.
    pub fn plan_context(&mut self, lmax: u64, pass_rows: u64, have_ring: bool) {
        let rows = ring_rows(self.window.min(lmax), pass_rows.max(self.block));
        self.ring = ds_ring_pick(have_ring, self.window, lmax, rows);
    }

    /// The drafter reads the target's hidden states and shares its `lm_head`.
    pub fn check_target(&self, spec: &ModelSpec) -> Result<(), String> {
        if self.hidden != spec.hidden
            || self.vocab != spec.vocab
            || !self.hidden.is_multiple_of(128)
        {
            return Err(format!(
                "dspine: drafter dims (hidden {}, vocab {}) differ from the target's ({}, {})",
                self.hidden, self.vocab, spec.hidden, spec.vocab
            ));
        }
        if !self.vocab.is_multiple_of(128) {
            return Err("dspine: vocab must be a multiple of 128 (top-16 kernel)".into());
        }
        let n = spec.n_layers as usize;
        if self.taps.iter().any(|&t| t + 1 >= n) {
            return Err(format!(
                "dspine: target_layer_ids {:?} must lie below the target's last layer ({n} layers)",
                self.taps
            ));
        }
        Ok(())
    }
}

pub fn is_drafter_kv(name: &str) -> bool {
    drafter_layer_suffix(name).is_some_and(|s| s == "kcache" || s == "vcache")
}

fn drafter_layer_suffix(name: &str) -> Option<&str> {
    let rest = name.strip_prefix('D')?;
    let sep = rest.find('_')?;
    if sep == 0 || !rest[..sep].bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&rest[sep + 1..])
}

pub fn is_drafter_weight(name: &str) -> bool {
    if matches!(
        name,
        "df_W_fc" | "df_w_hn" | "df_nofix" | "rope128_c" | "rope128_s"
    ) || name.starts_with("ds_")
    {
        return true;
    }
    drafter_layer_suffix(name).is_some_and(|s| {
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
                | "qn_w"
                | "kn_w"
        ) || fp8_matrix_of(s).is_some()
    })
}

/// The matrices of a drafter layer that run as FP8 when [`DSpineCfg::fp8`] is set.
const FP8_MATS: [&str; 7] = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"];
/// GEMM input sites of a drafter layer: the layer input (q / k / v), the attention core (o),
/// the post-attention norm (gate / up) and the gated product (down).
pub const FP8_SITES: [&str; 4] = ["xin", "core", "h2", "m"];

/// `"W_q_f8"` -> `("W_q", false)`, `"W_q_f8s"` -> `("W_q", true)` (codes / per-channel scales).
fn fp8_matrix_of(suffix: &str) -> Option<(&str, bool)> {
    let (w, scales) = match suffix.strip_suffix("_f8s") {
        Some(w) => (w, true),
        None => (suffix.strip_suffix("_f8")?, false),
    };
    FP8_MATS.contains(&w).then_some((w, scales))
}

/// Calibration-table name of site `what` of drafter layer `l` (`D<l>_<what>`).
pub fn fp8_site(l: usize, what: &str) -> String {
    format!("D{l}_{what}")
}

/// A drafter-layer projection `y {n, t} = W {k, n}^T . x {k, t}`: BF16, or FP8 with the cast of
/// input site `site` shared by every GEMM that reads it.
#[allow(clippy::too_many_arguments)]
/// A drafter-layer projection `y {n, t} = W {k, n}^T . x {k, t}`: BF16, or FP8 with the cast of
/// input site `site` shared by every GEMM that reads it (the cast named after `pfx`, one per lane
/// in a batched step).
#[allow(clippy::too_many_arguments)]
fn d_gemm_p(
    gb: &mut GraphBuilder,
    pfx: &str,
    dc: &DSpineCfg,
    x: TensorId,
    w_name: &str,
    k: u64,
    n: u64,
    t: u64,
    y_name: &str,
    site: &str,
) -> TensorId {
    if !dc.fp8 {
        return add_linear_bf16(gb, x, w_name, k, n, t, y_name);
    }
    use crate::synapse::ops::{as_param_bytes, ConvertToFp8Params, Fp8GemmV2Params};
    let xq_name = format!("{pfx}{site}_xq8");
    let xq = match gb.find_persistent_tensor(&xq_name) {
        Some(id) => id,
        None => {
            let cs =
                gb.get_or_add_persistent_tensor(format!("ds_{site}_cs"), SynDataType::Float, &[1]);
            let xq = gb.add_persistent_tensor(xq_name, SynDataType::Fp8_143, &[k, t]);
            // round half to even (`CAST_ROUND_HALF_NE`)
            let cp = ConvertToFp8Params { round_mode: 0 };
            gb.add_node(
                "convert_to_fp8_bf16",
                format!("{pfx}{site}_cast8"),
                &[x, cs],
                &[xq],
                as_param_bytes(&cp),
            );
            xq
        }
    };
    let sa = gb.get_or_add_persistent_tensor(format!("ds_{site}_sa"), SynDataType::Float, &[1]);
    let w = gb.get_or_add_persistent_tensor(format!("{w_name}_f8"), SynDataType::Fp8_143, &[k, n]);
    let sb = gb.get_or_add_persistent_tensor(format!("{w_name}_f8s"), SynDataType::Float, &[n]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[n, t]);
    let gp = Fp8GemmV2Params {
        transpose_a: 0,
        transpose_b: 1,
        is_hw_aligned: 0,
        _pad: 0,
        scale_method_hash_id: 0,
    };
    gb.add_node(
        "fp8_gemm_bf16",
        format!("{y_name}_gemm8"),
        &[xq, w, sa, sb],
        &[y],
        as_param_bytes(&gp),
    );
    y
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

/// Context K/V of every drafter layer for `t` target rows (`hs[i]` = output residual of tap `i`).
#[allow(clippy::too_many_arguments)]
pub fn append_ctx_pass(
    gb: &mut GraphBuilder,
    dc: &DSpineCfg,
    hs: &[TensorId],
    t: u64,
    cursor: TensorId,
    depth: TensorId,
    fix: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
) {
    append_ctx_pass_p(gb, "", dc, hs, t, cursor, depth, fix, rope_c, rope_s, lmax)
}

/// [`append_ctx_pass`] with the pass's own tensors and the context caches named after `pfx`
/// (one lane's in a batched step); the weights keep their names.
#[allow(clippy::too_many_arguments)]
pub fn append_ctx_pass_p(
    gb: &mut GraphBuilder,
    pfx: &str,
    dc: &DSpineCfg,
    hs: &[TensorId],
    t: u64,
    cursor: TensorId,
    depth: TensorId,
    fix: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
) {
    let slot = CtxSlot {
        pfx: pfx.to_string(),
        t,
        cursor,
        fix,
    };
    append_ctx_pass_slots(gb, pfx, dc, hs, &[slot], depth, rope_c, rope_s, lmax)
}

/// One slot of a drafter pass that runs over several lanes' rows in one launch: the prefix of
/// its own tensors (which also names its lane's context caches), its rows and its inputs.
#[derive(Debug, Clone)]
pub struct CtxSlot {
    pub pfx: String,
    /// The slot's target rows: its columns of the taps, in slot order.
    pub t: u64,
    pub cursor: TensorId,
    /// The lane's context bound.
    pub fix: TensorId,
}

/// `x {d, sum widths}` as one tensor per slot of `widths[j]` columns (intermediates named by
/// `name(j)`); a single slot is `x` itself.
fn split_slot_cols(
    gb: &mut GraphBuilder,
    x: TensorId,
    dt: SynDataType,
    d: u64,
    widths: &[u64],
    split_name: &str,
    name: impl Fn(usize) -> String,
) -> Vec<TensorId> {
    if widths.len() == 1 {
        return vec![x];
    }
    let parts: Vec<TensorId> = widths
        .iter()
        .enumerate()
        .map(|(j, &w)| gb.add_intermediate(name(j), dt, &[d, w]))
        .collect();
    gb.op_split(split_name, x, &parts, 1);
    parts
}

/// The slots' tensors `{d, w_j}` side by side as one `{d, sum w_j}` (an intermediate `name`);
/// a single slot's is its own tensor.
fn cat_slot_cols(
    gb: &mut GraphBuilder,
    parts: &[TensorId],
    dt: SynDataType,
    dims: &[u64],
    axis: u32,
    name: &str,
) -> TensorId {
    if parts.len() == 1 {
        return parts[0];
    }
    let out = gb.add_intermediate(name, dt, dims);
    gb.op_concat(format!("{name}_cat"), parts, out, axis);
    out
}

/// The context pass of several slots in one launch (`hs[i]` = tap `i` over every slot's rows,
/// slot by slot): the fc projection, its norm and every layer's context K/V projection run once
/// over all rows, so each weight is read once; each slot's K/V rows are then written to its own
/// lane's caches at its own cursor. `bpfx` names the tensors of the shared part; with one slot
/// whose prefix it is, this is the single-stream pass.
#[allow(clippy::too_many_arguments)]
pub fn append_ctx_pass_slots(
    gb: &mut GraphBuilder,
    bpfx: &str,
    dc: &DSpineCfg,
    hs: &[TensorId],
    slots: &[CtxSlot],
    depth: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
) {
    let hid = dc.hidden;
    let fci = hid * hs.len() as u64;
    let widths: Vec<u64> = slots.iter().map(|s| s.t).collect();
    let t: u64 = widths.iter().sum();
    let cat = gb.add_persistent_tensor(format!("{bpfx}ctx_cat"), SynDataType::Bf16, &[fci, t]);
    gb.op_concat(format!("{bpfx}ctx_concat"), hs, cat, 0);
    let f = add_linear_bf16(gb, cat, "df_W_fc", fci, hid, t, &format!("{bpfx}ctx_f"));
    let cn = d_rmsnorm(gb, f, "df_w_hn", &format!("{bpfx}ctx_cn"), hid, t);
    for l in 0..dc.nl {
        let lp = format!("D{l}_");
        let kc = add_linear_bf16(
            gb,
            cn,
            &format!("{lp}W_k"),
            hid,
            D_KVD,
            t,
            &format!("{bpfx}ctx_{lp}k"),
        );
        let vc = add_linear_bf16(
            gb,
            cn,
            &format!("{lp}W_v"),
            hid,
            D_KVD,
            t,
            &format!("{bpfx}ctx_{lp}v"),
        );
        let kn_w =
            gb.get_or_add_persistent_tensor(format!("{lp}kn_w"), SynDataType::Float, &[D_DH]);
        let kcs = split_slot_cols(gb, kc, SynDataType::Bf16, D_KVD, &widths, &format!("{bpfx}ctx_{lp}k_split"), |j| {
            format!("{}ctx_{lp}k", slots[j].pfx)
        });
        let vcs = split_slot_cols(gb, vc, SynDataType::Bf16, D_KVD, &widths, &format!("{bpfx}ctx_{lp}v_split"), |j| {
            format!("{}ctx_{lp}v", slots[j].pfx)
        });
        let rows = dc.ctx_rows(lmax);
        for (j, s) in slots.iter().enumerate() {
            let pfx = &s.pfx;
            let kcache = gb.get_or_add_persistent_tensor(
                format!("{pfx}{lp}kcache"),
                SynDataType::Bf16,
                &[D_DH, rows, D_NKV],
            );
            let vcache = gb.get_or_add_persistent_tensor(
                format!("{pfx}{lp}vcache"),
                SynDataType::Bf16,
                &[D_DH, rows, D_NKV],
            );
            gb.add_node(
                if dc.ring > 0 {
                    "gaudium_ds_ctxr"
                } else {
                    "gaudium_df_ctx"
                },
                format!("{pfx}ctx_{lp}kv"),
                &[kcs[j], vcs[j], kn_w, rope_c, rope_s, s.cursor, depth, s.fix],
                &[kcache, vcache],
                &[],
            );
        }
    }
}

pub struct BlockOut {
    pub drafts: TensorId,
    pub alts: TensorId,
    pub alts2: TensorId,
}

/// The pre-write state of the last layer that the serial tail reads.
struct Stash {
    raw: TensorId,
    rms: TensorId,
    pk: TensorId,
}

/// One slot of a block pass over several lanes in one launch: the prefix of its own tensors
/// (which also names its lane's context caches), its block's ids and position, its context
/// bound, and its sampler parameters `samp` (`(samp_f, samp_i)` names, for the coupled walk).
#[derive(Debug, Clone)]
pub struct BlkSlot {
    pub pfx: String,
    pub ids_blk: TensorId,
    pub cur_blk: TensorId,
    pub fix: TensorId,
    pub samp: (String, String),
}

/// One drafter layer plus its code bridge over the blocks of `slots`, side by side (`block`
/// rows each). `x_in` = the normalised layer input, `resid` = the residual stream; returns `(next normalised input, written stream,
/// pre-write state)`. The projections, the MLP, the bridge's read and its GEMMs run once over
/// every slot's rows (each weight read once); each slot's attention runs on its own block and
/// its lane's context caches. With one slot whose prefix is `bpfx` this is the single-stream
/// layer.
#[allow(clippy::too_many_arguments)]
fn append_spine_layer(
    gb: &mut GraphBuilder,
    bpfx: &str,
    slots: &[BlkSlot],
    dc: &DSpineCfg,
    x_in: TensorId,
    resid: TensorId,
    l: usize,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
    vocab: u64,
) -> (TensorId, TensorId, Stash) {
    let tb = dc.block;
    let t = tb * slots.len() as u64;
    let hid = dc.hidden;
    let inter = dc.intermediate;
    let r4 = 4 * tb;
    let lp = format!("D{l}_");
    // the layer's own tensors: one name set per launch (`bpfx`), the attention's per slot
    let ap = format!("{bpfx}D{l}_");
    let pfx = bpfx;

    let s_xin = fp8_site(l, "xin");
    let q = d_gemm_p(
        gb,
        pfx,
        dc,
        x_in,
        &format!("{lp}W_q"),
        hid,
        D_QD,
        t,
        &format!("{ap}q"),
        &s_xin,
    );
    let k = d_gemm_p(
        gb,
        pfx,
        dc,
        x_in,
        &format!("{lp}W_k"),
        hid,
        D_KVD,
        t,
        &format!("{ap}k"),
        &s_xin,
    );
    let v = d_gemm_p(
        gb,
        pfx,
        dc,
        x_in,
        &format!("{lp}W_v"),
        hid,
        D_KVD,
        t,
        &format!("{ap}v"),
        &s_xin,
    );
    let qn_w = gb.get_or_add_persistent_tensor(format!("{lp}qn_w"), SynDataType::Float, &[D_DH]);
    let kn_w = gb.get_or_add_persistent_tensor(format!("{lp}kn_w"), SynDataType::Float, &[D_DH]);
    let widths = vec![tb; slots.len()];
    let qs = split_slot_cols(gb, q, SynDataType::Bf16, D_QD, &widths, &format!("{ap}q_split"), |j| {
        format!("{}D{l}_q", slots[j].pfx)
    });
    let ks = split_slot_cols(gb, k, SynDataType::Bf16, D_KVD, &widths, &format!("{ap}k_split"), |j| {
        format!("{}D{l}_k", slots[j].pfx)
    });
    let vs = split_slot_cols(gb, v, SynDataType::Bf16, D_KVD, &widths, &format!("{ap}v_split"), |j| {
        format!("{}D{l}_v", slots[j].pfx)
    });
    // each slot's attention on its own block and its lane's context caches, stage by stage across
    // the slots (one slot's matrix stage beside another's vector stage)
    let rows = dc.ctx_rows(lmax);
    let sap = |sl: &BlkSlot| format!("{}D{l}_", sl.pfx);
    let mut prep = Vec::with_capacity(slots.len());
    for (j, sl) in slots.iter().enumerate() {
        let ap = sap(sl);
        let qb = gb.add_persistent_tensor(format!("{ap}qb"), SynDataType::Bf16, &[D_DH, r4, D_NKV]);
        let kblk =
            gb.add_persistent_tensor(format!("{ap}kblk"), SynDataType::Bf16, &[D_DH, 128, D_NKV]);
        let vblk =
            gb.add_persistent_tensor(format!("{ap}vblk"), SynDataType::Bf16, &[D_DH, 128, D_NKV]);
        gb.add_node(
            "gaudium_df_attn_prep",
            format!("{ap}attn_prep"),
            &[qs[j], ks[j], vs[j], qn_w, kn_w, rope_c, rope_s, sl.cur_blk],
            &[qb, kblk, vblk],
            &[],
        );
        prep.push((qb, kblk, vblk));
    }
    let mut scs = Vec::with_capacity(slots.len());
    for (sl, &(qb, kblk, _)) in slots.iter().zip(&prep) {
        let ap = sap(sl);
        // the context keys: one row per position, or the ring of the window (`dc.ring`)
        let kcache = gb.get_or_add_persistent_tensor(
            format!("{ap}kcache"),
            SynDataType::Bf16,
            &[D_DH, rows, D_NKV],
        );
        let vcache = gb.get_or_add_persistent_tensor(
            format!("{ap}vcache"),
            SynDataType::Bf16,
            &[D_DH, rows, D_NKV],
        );
        let sc = gb.add_persistent_tensor(format!("{ap}sc"), SynDataType::Float, &[rows, r4, D_NKV]);
        let sb = gb.add_persistent_tensor(format!("{ap}sb"), SynDataType::Float, &[128, r4, D_NKV]);
        gb.op_batch_gemm(format!("{ap}qk_ctx"), qb, kcache, sc, false, true);
        gb.op_batch_gemm(format!("{ap}qk_blk"), qb, kblk, sb, false, true);
        scs.push((sc, sb, vcache));
    }
    let mut ps = Vec::with_capacity(slots.len());
    for (sl, &(sc, sb, _)) in slots.iter().zip(&scs) {
        let ap = sap(sl);
        let pc = gb.add_persistent_tensor(format!("{ap}Pc"), SynDataType::Bf16, &[rows, r4, D_NKV]);
        let pb = gb.add_persistent_tensor(format!("{ap}Pb"), SynDataType::Bf16, &[128, r4, D_NKV]);
        gb.add_node(
            if dc.ring > 0 {
                "gaudium_ds_softr"
            } else {
                "gaudium_ds_softmax"
            },
            format!("{ap}softmax"),
            &[sc, sb, sl.cur_blk, sl.fix],
            &[pc, pb],
            &u32_params(&[tb as u32, dc.window as u32]),
        );
        ps.push((pc, pb));
    }
    let mut aos = Vec::with_capacity(slots.len());
    for (((sl, &(pc, pb)), &(_, _, vcache)), &(_, _, vblk)) in slots.iter().zip(&ps).zip(&scs).zip(&prep) {
        let ap = sap(sl);
        let aoc = gb.add_persistent_tensor(format!("{ap}aoc"), SynDataType::Bf16, &[D_DH, r4, D_NKV]);
        let aob = gb.add_persistent_tensor(format!("{ap}aob"), SynDataType::Bf16, &[D_DH, r4, D_NKV]);
        let ao = gb.add_persistent_tensor(format!("{ap}ao"), SynDataType::Bf16, &[D_DH, r4, D_NKV]);
        gb.op_batch_gemm(format!("{ap}pv_ctx"), pc, vcache, aoc, false, false);
        gb.op_batch_gemm(format!("{ap}pv_blk"), pb, vblk, aob, false, false);
        gb.op_add_bf16(format!("{ap}ao_add"), aoc, aob, ao);
        aos.push(ao);
    }
    let mut cores = Vec::with_capacity(slots.len());
    for (sl, &ao) in slots.iter().zip(&aos) {
        let ap = sap(sl);
        let core = gb.add_persistent_tensor(format!("{ap}core"), SynDataType::Bf16, &[D_QD, tb]);
        gb.add_node(
            "gaudium_df_post",
            format!("{ap}attn_post"),
            &[ao],
            &[core],
            &[],
        );
        cores.push(core);
    }
    let core = cat_slot_cols(gb, &cores, SynDataType::Bf16, &[D_QD, t], 1, &format!("{ap}core_all"));
    let attn = d_gemm_p(
        gb,
        pfx,
        dc,
        core,
        &format!("{lp}W_o"),
        D_QD,
        hid,
        t,
        &format!("{ap}attn"),
        &fp8_site(l, "core"),
    );

    let h = gb.add_persistent_tensor(format!("{ap}x2"), SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16(format!("{ap}x2_add"), resid, attn, h);
    let h2 = d_rmsnorm(gb, h, &format!("{lp}w_post"), &format!("{ap}h2"), hid, t);
    let s_h2 = fp8_site(l, "h2");
    let gt = d_gemm_p(
        gb,
        pfx,
        dc,
        h2,
        &format!("{lp}W_gate"),
        hid,
        inter,
        t,
        &format!("{ap}gate"),
        &s_h2,
    );
    let up = d_gemm_p(
        gb,
        pfx,
        dc,
        h2,
        &format!("{lp}W_up"),
        hid,
        inter,
        t,
        &format!("{ap}up"),
        &s_h2,
    );
    let act = gb.add_persistent_tensor(format!("{ap}act"), SynDataType::Bf16, &[inter, t]);
    gb.op_silu_bf16(format!("{ap}silu"), gt, act);
    let m = gb.add_persistent_tensor(format!("{ap}m"), SynDataType::Bf16, &[inter, t]);
    gb.op_mult_bf16(format!("{ap}swiglu"), act, up, m);
    let mo = d_gemm_p(
        gb,
        pfx,
        dc,
        m,
        &format!("{lp}W_down"),
        inter,
        hid,
        t,
        &format!("{ap}mlp"),
        &fp8_site(l, "m"),
    );
    let raw = gb.add_persistent_tensor(format!("{ap}out"), SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16(format!("{ap}out_add"), h, mo, raw);

    // ---- code bridge ----
    // (the read, the three GEMMs and the norms run over every slot's rows; the code, message
    // and finish kernels treat row 0 of a block as its anchor, so they run per slot)
    let (c, r) = (dc.code_size, dc.rank);
    let nrm = gb.add_persistent_tensor(format!("{ap}bn"), SynDataType::Bf16, &[hid, t]);
    let rms = gb.add_persistent_tensor(format!("{ap}brms"), SynDataType::Float, &[64, t]);
    gb.add_node(
        "gaudium_ds_read",
        format!("{ap}bread"),
        &[raw],
        &[nrm, rms],
        &[],
    );
    let z = add_linear_bf16(gb, nrm, &format!("ds_R{l}"), hid, c, t, &format!("{ap}bz"));
    let cb = gb.get_or_add_persistent_tensor("ds_cb", SynDataType::Bf16, &[c, vocab]);
    let zs = split_slot_cols(gb, z, SynDataType::Bf16, c, &widths, &format!("{ap}bz_split"), |j| {
        format!("{}D{l}_bz", slots[j].pfx)
    });
    let mut codes_s = Vec::with_capacity(slots.len());
    for (j, sl) in slots.iter().enumerate() {
        let sp = format!("{}D{l}_", sl.pfx);
        let codes = gb.add_persistent_tensor(format!("{sp}bcodes"), SynDataType::Bf16, &[c, tb]);
        gb.add_node(
            "gaudium_ds_codes",
            format!("{sp}bcode"),
            &[zs[j], cb, sl.ids_blk],
            &[codes],
            &[],
        );
        codes_s.push(codes);
    }
    let codes = cat_slot_cols(gb, &codes_s, SynDataType::Bf16, &[c, t], 1, &format!("{ap}bcodes_all"));
    let pk = add_linear_bf16(gb, codes, "ds_PW", c, 3 * r, t, &format!("{ap}bpk"));
    let gbias = gb.get_or_add_persistent_tensor("ds_gb", SynDataType::Float, &[r]);
    let pks = split_slot_cols(gb, pk, SynDataType::Bf16, 3 * r, &widths, &format!("{ap}bpk_split"), |j| {
        format!("{}D{l}_bpk", slots[j].pfx)
    });
    let mut msg_s = Vec::with_capacity(slots.len());
    for (j, sl) in slots.iter().enumerate() {
        let sp = format!("{}D{l}_", sl.pfx);
        let msg = gb.add_persistent_tensor(format!("{sp}bmsg"), SynDataType::Bf16, &[r, tb]);
        gb.add_node(
            "gaudium_ds_msg",
            format!("{sp}bmsgk"),
            &[pks[j], gbias],
            &[msg],
            &[],
        );
        msg_s.push(msg);
    }
    let msg = cat_slot_cols(gb, &msg_s, SynDataType::Bf16, &[r, t], 1, &format!("{ap}bmsg_all"));
    let delta = add_linear_bf16(
        gb,
        msg,
        &format!("ds_W{l}"),
        r,
        hid,
        t,
        &format!("{ap}bdelta"),
    );
    let wn = gb.get_or_add_persistent_tensor(format!("ds_wnext{l}"), SynDataType::Float, &[hid]);
    let raws = split_slot_cols(gb, raw, SynDataType::Bf16, hid, &widths, &format!("{ap}out_split"), |j| {
        format!("{}D{l}_out", slots[j].pfx)
    });
    let deltas = split_slot_cols(gb, delta, SynDataType::Bf16, hid, &widths, &format!("{ap}bdelta_split"), |j| {
        format!("{}D{l}_bdelta", slots[j].pfx)
    });
    let rmss = split_slot_cols(gb, rms, SynDataType::Float, 64, &widths, &format!("{ap}brms_split"), |j| {
        format!("{}D{l}_brms", slots[j].pfx)
    });
    let (mut outp_s, mut nx_s) = (Vec::with_capacity(slots.len()), Vec::with_capacity(slots.len()));
    for (j, sl) in slots.iter().enumerate() {
        let sp = format!("{}D{l}_", sl.pfx);
        let outp = gb.add_persistent_tensor(format!("{sp}bout"), SynDataType::Bf16, &[hid, tb]);
        let nx = gb.add_persistent_tensor(format!("{sp}bnx"), SynDataType::Bf16, &[hid, tb]);
        gb.add_node(
            "gaudium_ds_finish",
            format!("{sp}bfin"),
            &[raws[j], deltas[j], rmss[j], wn],
            &[outp, nx],
            &[],
        );
        outp_s.push(outp);
        nx_s.push(nx);
    }
    let outp = cat_slot_cols(gb, &outp_s, SynDataType::Bf16, &[hid, t], 1, &format!("{ap}bout_all"));
    let nx = cat_slot_cols(gb, &nx_s, SynDataType::Bf16, &[hid, t], 1, &format!("{ap}bnx_all"));
    (nx, outp, Stash { raw, rms, pk })
}

/// Score calibration scalar of the coupled walk (`cs_t = CWALK_ALPHA / rms_t`). The serial
/// scores are the refined row's logits times its RMS; dividing by the row's pre-write RMS makes
/// them logits up to a ratio near 1, and a further scalar was best at 1.0 in the project's
/// offline study of this drafter family.
pub const CWALK_ALPHA: f32 = 1.0;

/// The block `[anchor, mask x (block - 1)]` at positions `cur_blk + t` -> `block - 1` drafts.
///
/// `coupled` (the recipes of sampled requests): the serial walk is `gaudium_ds_cwalk`, which
/// draws every draft as the verify row that will check it draws its token: the argmax over the
/// row's 16 candidates of the calibrated score over the temperature plus the sampler's noise
/// `g(seed, n, candidate)`, `n = cur_blk + row` = the draft's sequence index, with the request's
/// support rules applied to the 16 values. A request with coupling off walks greedily in the
/// same kernel (a flag of `samp_i`). The drafts stay proposals either way.
#[allow(clippy::too_many_arguments)]
pub fn append_block_pass(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    dc: &DSpineCfg,
    ids_blk: TensorId,
    cur_blk: TensorId,
    fix: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
    coupled: bool,
) -> BlockOut {
    use crate::engine::sampler::{SAMP_F, SAMP_I};
    append_block_pass_p(
        gb, "", spec, dc, ids_blk, cur_blk, fix, rope_c, rope_s, lmax, coupled, (SAMP_F, SAMP_I),
    )
}

/// [`append_block_pass`] with the pass's own tensors named after `pfx` and the coupled walk
/// reading the request parameters `samp` (`(samp_f, samp_i)` names): one lane's block in a
/// batched step.
#[allow(clippy::too_many_arguments)]
pub fn append_block_pass_p(
    gb: &mut GraphBuilder,
    pfx: &str,
    spec: &ModelSpec,
    dc: &DSpineCfg,
    ids_blk: TensorId,
    cur_blk: TensorId,
    fix: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
    coupled: bool,
    samp: (&str, &str),
) -> BlockOut {
    let slot = BlkSlot {
        pfx: pfx.to_string(),
        ids_blk,
        cur_blk,
        fix,
        samp: (samp.0.to_string(), samp.1.to_string()),
    };
    append_block_pass_slots(gb, pfx, spec, dc, &[slot], rope_c, rope_s, lmax, coupled)
        .pop()
        .expect("one slot")
}

/// The draft blocks of several slots in one launch: the embedding, every layer's projections,
/// MLP and bridge GEMMs, the final norm and the readout run once over every slot's block rows
/// (each weight read once); each slot's attention, candidates' base and score, and walk run on
/// its own rows, its lane's context caches and its own parameters. `bpfx` names the tensors of
/// the shared part; with one slot whose prefix it is, this is the single-stream pass. Returns
/// each slot's drafts.
#[allow(clippy::too_many_arguments)]
pub fn append_block_pass_slots(
    gb: &mut GraphBuilder,
    bpfx: &str,
    spec: &ModelSpec,
    dc: &DSpineCfg,
    slots: &[BlkSlot],
    rope_c: TensorId,
    rope_s: TensorId,
    lmax: u64,
    coupled: bool,
) -> Vec<BlockOut> {
    let tb = dc.block;
    let s = tb - 1;
    let t = tb * slots.len() as u64;
    let hid = dc.hidden;
    let vocab = spec.vocab;
    let pfx = bpfx;
    let widths = vec![tb; slots.len()];
    let ids_s: Vec<TensorId> = slots.iter().map(|sl| sl.ids_blk).collect();
    let ids_all = cat_slot_cols(gb, &ids_s, SynDataType::Int32, &[t], 0, &format!("{pfx}blk_ids_all"));
    let table = gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, vocab]);
    let x = gb.add_persistent_tensor(format!("{pfx}blk_x0"), SynDataType::Bf16, &[hid, t]);
    gb.op_gather_bf16(format!("{pfx}blk_embed"), table, ids_all, x, 1);
    let mut resid = x;
    let mut x_in = d_rmsnorm(gb, x, "D0_w_in", &format!("{pfx}D0_h"), hid, t);
    let mut stash = None;
    for l in 0..dc.nl {
        let (nx, outp, st) = append_spine_layer(
            gb, pfx, slots, dc, x_in, resid, l, rope_c, rope_s, lmax, vocab,
        );
        x_in = nx;
        resid = outp;
        stash = Some(st);
    }
    let st = stash.expect("at least one drafter layer");
    let hf = x_in; // norm(written stream of the last layer): the final hidden rows

    let lm_name = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    // Candidates come from the draft vocabulary when one is set: the readout streams `nvd`
    // rows of the `lm_head` instead of all of it, and the top-16 row indices map back to token
    // ids through `ds_dv_ids`. The serial tables (base, TG, TSV) are indexed by token id either
    // way. A token outside the subset can only cost an acceptance.
    let dv = dc.nvd > 0;
    let logits = if dv && dc.fp8 && dc.fp8_head {
        let site = fp8_site(dc.nl, "xin");
        d_gemm_p(gb, pfx, dc, hf, "ds_W_lmd", hid, dc.nvd, t, &format!("{pfx}blk_logits"), &site)
    } else if dv {
        add_linear_bf16(gb, hf, "ds_W_lmd", hid, dc.nvd, t, &format!("{pfx}blk_logits"))
    } else {
        add_linear_bf16(gb, hf, lm_name, hid, vocab, t, &format!("{pfx}blk_logits"))
    };
    let pv = gb.add_persistent_tensor(format!("{pfx}blk_pv"), SynDataType::Float, &[64, TOPK, TOPK_CH, t]);
    let pi = gb.add_persistent_tensor(format!("{pfx}blk_pi"), SynDataType::Int32, &[64, TOPK, TOPK_CH, t]);
    gb.add_node("gaudium_df_topk1", format!("{pfx}blk_topk1"), &[logits], &[pv, pi], &[]);
    // a subset's row indices map back to token ids; every row in id order needs no map
    let mapped = ds_draft_mapped(dc.nvd, vocab);
    let cand_name = format!("{pfx}{}", if mapped { "blk_cand_l" } else { "blk_cand" });
    let mut cand = gb.add_persistent_tensor(cand_name.as_str(), SynDataType::Int32, &[TOPK, t]);
    let unary = gb.add_persistent_tensor(format!("{pfx}blk_unary"), SynDataType::Float, &[TOPK, t]);
    gb.add_node(
        "gaudium_df_topk2",
        format!("{pfx}blk_topk2"),
        &[pv, pi],
        &[cand, unary],
        &[],
    );
    if mapped {
        cand = append_dv_map(gb, cand, &[TOPK, t], &format!("{pfx}blk_cand"), "ds_dv_ids", dc.nvd);
    }
    // per slot: its rows of the candidates and of the last layer's pre-write state
    let cands = split_slot_cols(gb, cand, SynDataType::Int32, TOPK, &widths, &format!("{pfx}blk_cand_split"), |j| {
        format!("{}blk_cand_s", slots[j].pfx)
    });
    let raws = split_slot_cols(gb, st.raw, SynDataType::Bf16, hid, &widths, &format!("{pfx}blk_raw_split"), |j| {
        format!("{}blk_raw", slots[j].pfx)
    });
    let pks = split_slot_cols(gb, st.pk, SynDataType::Bf16, 3 * dc.rank, &widths, &format!("{pfx}blk_pk_split"), |j| {
        format!("{}blk_pk", slots[j].pfx)
    });
    let rmss = split_slot_cols(gb, st.rms, SynDataType::Float, 64, &widths, &format!("{pfx}blk_rms_split"), |j| {
        format!("{}blk_rms", slots[j].pfx)
    });

    let mut outs = Vec::with_capacity(slots.len());
    for (j, sl) in slots.iter().enumerate() {
        let pfx = sl.pfx.as_str();
        let (cand, t) = (cands[j], tb);
        let st = Stash {
            raw: raws[j],
            rms: rmss[j],
            pk: pks[j],
        };
        let (ids_blk, cur_blk) = (sl.ids_blk, sl.cur_blk);
        let samp = (sl.samp.0.as_str(), sl.samp.1.as_str());
        let nw = gb.get_or_add_persistent_tensor("ds_nw", SynDataType::Float, &[hid]);
        let lm = gb.get_or_add_persistent_tensor(lm_name, SynDataType::Bf16, &[hid, vocab]);
        let base = gb.add_persistent_tensor(format!("{pfx}blk_base"), SynDataType::Float, &[TOPK, t]);
        gb.add_node(
            "gaudium_ds_base",
            format!("{pfx}blk_base_k"),
            &[st.raw, nw, cand, lm],
            &[base],
            &[],
        );
        let r = dc.rank;
        let gbias = gb.get_or_add_persistent_tensor("ds_gb", SynDataType::Float, &[r]);
        let tsv = gb.get_or_add_persistent_tensor("ds_TSV", SynDataType::Bf16, &[2 * r, vocab]);
        let tg = gb.get_or_add_persistent_tensor("ds_TG", SynDataType::Bf16, &[r, vocab]);
        let scores = gb.add_persistent_tensor(format!("{pfx}blk_scores"), SynDataType::Float, &[TOPK, TOPK, s]);
        gb.add_node(
            "gaudium_ds_score",
            format!("{pfx}blk_score"),
            &[cand, base, st.pk, st.rms, gbias, tsv, tg, ids_blk],
            &[scores],
            &[],
        );
        let drafts = gb.add_persistent_tensor(format!("{pfx}blk_drafts"), SynDataType::Int32, &[s]);
        let alts = gb.add_persistent_tensor(format!("{pfx}blk_alts"), SynDataType::Int32, &[s]);
        let alts2 = gb.add_persistent_tensor(format!("{pfx}blk_alts2"), SynDataType::Int32, &[s]);
        if coupled {
            let spf = gb.get_or_add_persistent_tensor(samp.0, SynDataType::Float, &[8]);
            let spi = gb.get_or_add_persistent_tensor(samp.1, SynDataType::Int32, &[8]);
            // the row scales the kernel used (read back by the host twin under `sample_check`)
            let cs = gb.add_persistent_tensor(format!("{pfx}blk_cwcs"), SynDataType::Float, &[64, s]);
            gb.add_node(
                "gaudium_ds_cwalk",
                format!("{pfx}blk_cwalk"),
                &[scores, cand, st.rms, spf, spi, cur_blk],
                &[drafts, alts, alts2, cs],
                &u32_params(&[CWALK_ALPHA.to_bits()]),
            );
        } else {
            gb.add_node(
                "gaudium_df_walk",
                format!("{pfx}blk_walk"),
                &[scores, cand],
                &[drafts, alts, alts2],
                &[],
            );
        }
        outs.push(BlockOut {
            drafts,
            alts,
            alts2,
        });
    }
    outs
}

/// Draft-vocabulary row indices `li` (`dims`) -> token ids through the table `table` `{1, nvd}`,
/// as the persistent tensor `name`.
pub(super) fn append_dv_map(
    gb: &mut GraphBuilder,
    li: TensorId,
    dims: &[u64],
    name: &str,
    table: &str,
    nvd: u64,
) -> TensorId {
    let n: u64 = dims.iter().product();
    let tab = gb.get_or_add_persistent_tensor(table, SynDataType::Int32, &[1, nvd]);
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
    dc: &DSpineCfg,
    taps: &[TensorId],
    t: u64,
    cursor: TensorId,
    depth: TensorId,
    tokens: TensorId,
    a_out: TensorId,
    cursor_out: TensorId,
    lmax: u64,
    coupled: bool,
) {
    use crate::engine::sampler::{SAMP_F, SAMP_I};
    append_step_drafter_p(
        gb, "", spec, dc, taps, t, cursor, depth, tokens, a_out, cursor_out, lmax, coupled,
        (SAMP_F, SAMP_I),
    )
}

/// [`append_step_drafter`] for one lane of a batched step: its own tensors, context caches,
/// context bound and next ids named after `pfx`, its sampler parameters `samp`.
#[allow(clippy::too_many_arguments)]
pub fn append_step_drafter_p(
    gb: &mut GraphBuilder,
    pfx: &str,
    spec: &ModelSpec,
    dc: &DSpineCfg,
    taps: &[TensorId],
    t: u64,
    cursor: TensorId,
    depth: TensorId,
    tokens: TensorId,
    a_out: TensorId,
    cursor_out: TensorId,
    lmax: u64,
    coupled: bool,
    samp: (&str, &str),
) {
    let slot = StepSlot {
        pfx: pfx.to_string(),
        samp: (samp.0.to_string(), samp.1.to_string()),
        t,
        cursor,
        tokens,
        a_out,
        cursor_out,
    };
    append_step_drafter_slots(gb, pfx, spec, dc, taps, &[slot], depth, lmax, coupled)
}

/// One slot of the drafter section of a batched step: the prefix of its own tensors (which
/// also names its lane's context caches, context bound and next ids), its sampler parameters,
/// its verify rows and what its accept produced.
#[derive(Debug, Clone)]
pub struct StepSlot {
    pub pfx: String,
    pub samp: (String, String),
    /// Verify rows of the slot: its columns of the taps, in slot order.
    pub t: u64,
    pub cursor: TensorId,
    pub tokens: TensorId,
    pub a_out: TensorId,
    pub cursor_out: TensorId,
}

/// [`append_step_drafter`] for several slots in one launch: the context pass over every slot's
/// verify rows and the next blocks of every slot side by side, each drafter weight read once
/// ([`append_ctx_pass_slots`], [`append_block_pass_slots`]); `taps[i]` = tap `i` over every
/// slot's rows, slot by slot. With one slot whose prefix is `bpfx` this is the single-stream
/// drafter section.
#[allow(clippy::too_many_arguments)]
pub fn append_step_drafter_slots(
    gb: &mut GraphBuilder,
    bpfx: &str,
    spec: &ModelSpec,
    dc: &DSpineCfg,
    taps: &[TensorId],
    slots: &[StepSlot],
    depth: TensorId,
    lmax: u64,
    coupled: bool,
) {
    let (rc, rs) = rope_tables(gb, lmax);
    let nofix: Vec<TensorId> = slots
        .iter()
        .map(|sl| gb.get_or_add_persistent_tensor(format!("{}df_nofix", sl.pfx), SynDataType::Int32, &[2]))
        .collect();
    let ctx: Vec<CtxSlot> = slots
        .iter()
        .zip(&nofix)
        .map(|(sl, &fix)| CtxSlot {
            pfx: sl.pfx.clone(),
            t: sl.t,
            cursor: sl.cursor,
            fix,
        })
        .collect();
    append_ctx_pass_slots(gb, bpfx, dc, taps, &ctx, depth, rc, rs, lmax);
    let mut blk = Vec::with_capacity(slots.len());
    for (sl, &fix) in slots.iter().zip(&nofix) {
        let pfx = &sl.pfx;
        let ids_blk = gb.add_persistent_tensor(format!("{pfx}df_ids_blk"), SynDataType::Int32, &[dc.block]);
        gb.add_node(
            "gaudium_df_ids",
            format!("{pfx}df_ids"),
            &[sl.tokens, sl.a_out],
            &[ids_blk],
            &u32_params(&[dc.mask_id as u32]),
        );
        blk.push(BlkSlot {
            pfx: pfx.clone(),
            ids_blk,
            cur_blk: sl.cursor_out,
            fix,
            samp: sl.samp.clone(),
        });
    }
    let outs = append_block_pass_slots(gb, bpfx, spec, dc, &blk, rc, rs, lmax, coupled);
    for (sl, out) in slots.iter().zip(outs) {
        let pfx = &sl.pfx;
        let next_ids = gb.add_persistent_tensor(format!("{pfx}next_ids"), SynDataType::Int32, &[dc.block]);
        gb.add_node(
            "gaudium_df_next",
            format!("{pfx}next_ids"),
            &[sl.tokens, sl.a_out, out.drafts, out.alts, out.alts2],
            &[next_ids],
            &u32_params(&[(dc.block - 1) as u32]),
        );
    }
}

/// Drafter context for a prefill tile of `t` rows (`df_tap{i}` = the tile's tap outputs).
pub fn build_ctx_graph(
    spec: &ModelSpec,
    dc: &DSpineCfg,
    t: u64,
    lmax: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_spec_dsctx_T{t}"));
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
/// `coupled`: the first block of a sampled request (`g_spec_dsblk0s`, the coupled walk).
pub fn build_block0_graph(
    spec: &ModelSpec,
    dc: &DSpineCfg,
    lmax: u64,
    elf_hashes: &[(String, String)],
    coupled: bool,
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(if coupled {
        "g_spec_dsblk0s"
    } else {
        "g_spec_dsblk0"
    });
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let ids_blk = gb.add_persistent_tensor("df_ids_pre", SynDataType::Int32, &[dc.block]);
    let cur_blk = gb.add_persistent_tensor("df_cur_pre", SynDataType::Int32, &[1]);
    let nofix = gb.get_or_add_persistent_tensor("df_nofix", SynDataType::Int32, &[2]);
    let (rc, rs) = rope_tables(&mut gb, lmax);
    let _ = append_block_pass(
        &mut gb, spec, dc, ids_blk, cur_blk, nofix, rc, rs, lmax, coupled,
    );
    gb
}

// ============================================================================
// Loaded drafter
// ============================================================================

/// A DSpine checkpoint opened for one target model: config, mapped weights, and its own recipes.
pub struct DSpine {
    pub cfg: DSpineCfg,
    store: SafetensorsStore,
    ctx_recipes: HashMap<u64, CompiledRecipe>,
    blk0: Option<CompiledRecipe>,
    blk0_launch: Option<PreparedLaunch>,
    /// First block of a sampled request (the coupled walk); `None`: not built.
    blk0s: Option<CompiledRecipe>,
    blk0s_launch: Option<PreparedLaunch>,
    pool: BTreeMap<String, u64>,
    /// Rows of the eagerly compiled context recipe (the engine's prefill tile).
    full_tile: u64,
    /// FP8 layers: the calibrated cast scale of every GEMM input site (`D<l>_<site>`).
    act: BTreeMap<String, crate::fp8::ActScale>,
    /// FP8 layers: directory caching the quantised matrices (None: quantise at every load).
    f8_dir: Option<std::path::PathBuf>,
    /// Calibration run: where the table goes, and the running maxima of the GEMM input sites.
    calib: Option<(std::path::PathBuf, crate::engine::fp8_graph::ActCalibrator)>,
    /// The context bound on the device (`lim[0]` of the ring softmax; -1 = none, as loaded).
    ctx_floor: i32,
    /// [lanes] the pool's lane-owned entries (context caches, context bound) change with the
    /// active lane: the first block binds them at each launch instead of a prepared launch.
    lanes_on: bool,
}

/// [lanes] What one lane owns of the drafter: the addresses of its context caches and context
/// bound (by the pool's names), and the bound's value on the device.
#[derive(Debug, Clone, Default)]
pub struct DSpineLane {
    pub pool: BTreeMap<String, u64>,
    pub ctx_floor: i32,
}

/// Default location of a drafter's FP8 files (activation-scale table, quantised matrices).
pub fn default_fp8_dir(draft_dir: &Path) -> std::path::PathBuf {
    let mut name = draft_dir
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push("-gaudium-pc");
    draft_dir.with_file_name(name)
}

/// File name of a drafter's activation-scale table inside its FP8 directory.
pub const ACT_TABLE_FILE: &str = "act_scales_ds.txt";

/// The drafter's activation-scale table a load looks for: the one the run options name, else
/// the one in the drafter's default FP8 directory.
pub fn act_table_path(draft_dir: &Path, opts: &crate::engine::step_opts::StepOpts) -> std::path::PathBuf {
    opts.draft_fp8_act
        .clone()
        .unwrap_or_else(|| default_fp8_dir(draft_dir).join(ACT_TABLE_FILE))
}

impl DSpine {
    /// Applies the run options: the draft vocabulary of the block readout and, when the
    /// drafter's activation-scale table is there (or `force_fp8` demands it), FP8 layers.
    /// Returns a description for the load log.
    pub fn configure(
        &mut self,
        spec: &ModelSpec,
        draft_dir: &Path,
        opts: &crate::engine::step_opts::StepOpts,
    ) -> Result<String, String> {
        // The block proposes from the whole vocabulary unless the run names a subset: a token
        // outside a subset can never be drafted, and outside English text and code a large
        // share of the tokens lies above any "lowest ids" cut.
        self.cfg.nvd = match opts.draft_vocab {
            None => 0,
            Some(_) => opts.draft_rows(spec.vocab, super::mtp::DV_TAIL)?,
        };
        let table = act_table_path(draft_dir, opts);
        let want = match opts.draft_fp8 {
            _ if opts.draft_fp8_calibrate.is_some() => false,
            Some(v) => v,
            None => table.is_file(),
        };
        if want {
            let act = crate::fp8::load_act_scales(&table)
                .map_err(|e| format!("dspine FP8 table {}: {e}", table.display()))?;
            for l in 0..self.cfg.nl {
                for what in FP8_SITES {
                    let site = fp8_site(l, what);
                    if !act.contains_key(&site) {
                        return Err(format!(
                            "dspine FP8 table {} has no scale for site {site}",
                            table.display()
                        ));
                    }
                }
            }
            // the readout's site is optional: a table calibrated without it keeps a BF16 head.
            // A full-vocabulary FP8 readout is the "subset" of every row (half the bytes of the
            // BF16 `lm_head` per block).
            self.cfg.fp8_head = act.contains_key(&fp8_site(self.cfg.nl, "xin"));
            self.cfg.nvd = ds_draft_rows(self.cfg.nvd, spec.vocab, self.cfg.fp8_head);
            self.act = act;
            self.f8_dir = table.parent().map(Path::to_path_buf);
            self.cfg.fp8 = true;
        }
        if let Some(out) = &opts.draft_fp8_calibrate {
            self.calib = Some((out.clone(), crate::engine::fp8_graph::ActCalibrator::new()));
        }
        let vocab = if ds_draft_mapped(self.cfg.nvd, spec.vocab) {
            format!("draft vocabulary {} of {} rows", self.cfg.nvd, spec.vocab)
        } else {
            "full-vocabulary readout".to_string()
        };
        Ok(if self.cfg.fp8 {
            format!(
                "{vocab}, FP8 layers{} ({})",
                if self.cfg.fp8_head {
                    " and FP8 readout"
                } else {
                    ""
                },
                table.display()
            )
        } else {
            format!("{vocab}, BF16 layers")
        })
    }

    pub fn open(dir: &Path, spec: &ModelSpec) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("read {}/config.json: {e}", dir.display()))?;
        let cfg = DSpineCfg::from_json(&text)?;
        cfg.check_target(spec)?;
        let store = SafetensorsStore::open(dir)
            .map_err(|e| format!("open drafter safetensors in {}: {e}", dir.display()))?;
        let d = Self {
            cfg,
            store,
            ctx_recipes: HashMap::new(),
            blk0: None,
            blk0_launch: None,
            blk0s: None,
            blk0s_launch: None,
            pool: BTreeMap::new(),
            full_tile: 0,
            act: BTreeMap::new(),
            f8_dir: None,
            calib: None,
            ctx_floor: -1,
            lanes_on: false,
        };
        d.check_checkpoint()?;
        Ok(d)
    }

    /// The parts of the contract only the tensors can confirm.
    fn check_checkpoint(&self) -> Result<(), String> {
        for k in [
            "draft_block_causal_version",
            "last_write_refine_version",
            "selected_prefix_bridge.independent_write_version",
            "selected_prefix_bridge.terminal_disabled_version",
        ] {
            let Some(meta) = self.store.find(k) else {
                continue;
            };
            let bytes = self.bytes(k)?;
            if meta.dtype != "I64" || bytes.len() < 8 || bytes[..8] != 1i64.to_le_bytes() {
                return Err(format!("dspine: unsupported {k}"));
            }
        }
        if self
            .store
            .find("selected_prefix_bridge.out.weight")
            .is_some()
            && self
                .bytes("selected_prefix_bridge.out.weight")?
                .as_chunks::<2>()
                .0
                .iter()
                .any(|&c| u16::from_le_bytes(c) & 0x7fff != 0)
        {
            return Err(
                "dspine: terminal-disabled checkpoint with a nonzero terminal projection".into(),
            );
        }
        Ok(())
    }

    fn bytes(&self, name: &str) -> Result<&[u8], String> {
        self.store
            .tensor_bytes(name)
            .map_err(|e| format!("drafter tensor '{name}': {e}"))?
            .ok_or_else(|| format!("drafter tensor '{name}' is not mapped"))
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
        self.bytes(name)
    }

    fn f32s(&self, name: &str, shape: &[u64]) -> Result<Vec<f32>, String> {
        Ok(self
            .need(name, shape)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| bf16_to_f32(u16::from_le_bytes(c)))
            .collect())
    }

    /// Per-channel f32 parameter in the deinterleaved lane order of each 128-channel block.
    fn lane_f32(&self, name: &str, n: u64) -> Result<Vec<u8>, String> {
        let w = self.f32s(name, &[n])?;
        let mut out = vec![0f32; n as usize];
        for (i, &x) in w.iter().enumerate() {
            out[lane256(i as u64) as usize] = x;
        }
        Ok(out.iter().flat_map(|x| x.to_le_bytes()).collect())
    }

    /// Device bytes of one drafter weight tensor (HF `[out, in]` matrices are used as `{in, out}`
    /// operands of transposed-B GEMMs, so they are copied as they are).
    fn pack<'a>(
        &'a self,
        name: &str,
        lmax: u64,
        tables: &'a SerialTables,
    ) -> Result<Cow<'a, [u8]>, String> {
        let c = &self.cfg;
        let (hid, inter, vocab, cs, r) = (c.hidden, c.intermediate, c.vocab, c.code_size, c.rank);
        let nl = c.nl as u64;
        let b = "selected_prefix_bridge.";
        let raw = |n: &str, shape: &[u64]| self.need(n, shape).map(Cow::Borrowed);
        match name {
            "df_W_fc" => return raw("fc.weight", &[hid, hid * c.taps.len() as u64]),
            "df_w_hn" => return raw("hidden_norm.weight", &[hid]),
            "df_nofix" => {
                return Ok(Cow::Owned(
                    [-1i32, -1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                ))
            }
            "rope128_c" | "rope128_s" => {
                let (rc, rs) =
                    packer_dspine::rope128_tables(c.rope_theta, (lmax + ROPE_PAD) as usize);
                let t = if name == "rope128_c" { rc } else { rs };
                return Ok(Cow::Owned(t.iter().flat_map(|x| x.to_le_bytes()).collect()));
            }
            "ds_nw" => return self.lane_f32("norm.weight", hid).map(Cow::Owned),
            "ds_cb" => return raw(&format!("{b}codebook"), &[vocab, cs]),
            "ds_gb" => return self.lane_f32(&format!("{b}gate.bias"), r).map(Cow::Owned),
            n if n.starts_with("ds_D") && (n.ends_with("_cs") || n.ends_with("_sa")) => {
                let site = &n[3..n.len() - 3];
                let a = self
                    .act
                    .get(site)
                    .ok_or_else(|| format!("dspine: no FP8 activation scale for site {site}"))?;
                let v = if n.ends_with("_cs") { a.cs() } else { a.sa() };
                return Ok(Cow::Owned(v.to_le_bytes().to_vec()));
            }
            "ds_TG" => return Ok(Cow::Borrowed(&tables.tg)),
            "ds_TSV" => return Ok(Cow::Borrowed(&tables.tsv)),
            "ds_PW" => {
                let mut out = Vec::with_capacity((3 * r * cs * 2) as usize);
                for k in ["gate.weight", "predecessor_gate.weight", "prev.weight"] {
                    out.extend_from_slice(self.need(&format!("{b}{k}"), &[r, cs])?);
                }
                return Ok(Cow::Owned(out));
            }
            _ => {}
        }
        if let Some(l) = name
            .strip_prefix("ds_wnext")
            .and_then(|s| s.parse::<u64>().ok())
        {
            let src = if l + 1 < nl {
                format!("layers.{}.input_layernorm.weight", l + 1)
            } else {
                "norm.weight".to_string()
            };
            return self.lane_f32(&src, hid).map(Cow::Owned);
        }
        if let Some(l) = name
            .strip_prefix("ds_R")
            .and_then(|s| s.parse::<u64>().ok())
        {
            if l + 1 == nl {
                return raw(&format!("{b}read_weight"), &[cs, hid]);
            }
            let all = self.need(&format!("{b}early_read_weights"), &[nl - 1, cs, hid])?;
            let n = (cs * hid * 2) as usize;
            return Ok(Cow::Borrowed(&all[l as usize * n..(l as usize + 1) * n]));
        }
        if let Some(l) = name
            .strip_prefix("ds_W")
            .and_then(|s| s.parse::<u64>().ok())
        {
            let all = self.need(&format!("{b}feature_write_weights"), &[nl, hid, r])?;
            let n = (hid * r * 2) as usize;
            return Ok(Cow::Borrowed(&all[l as usize * n..(l as usize + 1) * n]));
        }
        let Some(what) = drafter_layer_suffix(name) else {
            return Err(format!("dspine: no packing rule for '{name}'"));
        };
        let l = &name[1..name.find('_').unwrap()];
        let lp = format!("layers.{l}.");
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
            "qn_w" => self
                .lane_f32(&format!("{lp}self_attn.q_norm.weight"), D_DH)
                .map(Cow::Owned),
            "kn_w" => self
                .lane_f32(&format!("{lp}self_attn.k_norm.weight"), D_DH)
                .map(Cow::Owned),
            _ => Err(format!("dspine: no packing rule for '{name}'")),
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
        coupled_elf_hashes: Option<&[(String, String)]>,
    ) -> Result<u64, String> {
        let mut ws_s = 0u64;
        if let Some(hashes) = coupled_elf_hashes {
            let r = build_block0_graph(spec, &self.cfg, lmax, hashes, true)
                .compile(Arc::clone(api), cache, driver_version)
                .map_err(|e| format!("compile dspine block0 (sampled): {e}"))?;
            if verbose {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                    r.name,
                    r.from_cache,
                    r.ws_size as f64 / (1024.0 * 1024.0)
                );
            }
            ws_s = r.ws_size;
            self.blk0s = Some(r);
        }
        let blk0 = build_block0_graph(spec, &self.cfg, lmax, elf_hashes, false)
            .compile(Arc::clone(api), cache, driver_version)
            .map_err(|e| format!("compile dspine block0: {e}"))?;
        let ctx = build_ctx_graph(spec, &self.cfg, prefill_tile, lmax, elf_hashes)
            .compile(Arc::clone(api), cache, driver_version)
            .map_err(|e| format!("compile dspine ctx T={prefill_tile}: {e}"))?;
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
        let ws = blk0.ws_size.max(ctx.ws_size).max(ws_s);
        self.blk0 = Some(blk0);
        self.ctx_recipes.insert(prefill_tile, ctx);
        self.full_tile = prefill_tile;
        Ok(ws)
    }

    /// After the seal: uploads the drafter weights and serial tables and prepares its launches.
    /// `pool` = every address a drafter recipe may bind (target weights + this module's buffers).
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &mut self,
        dev: &HpuDevice,
        spec: &ModelSpec,
        packer: &WeightPacker<'_>,
        stage: &mut PinnedBuf,
        cache_dir: Option<&Path>,
        weight_specs: &BTreeMap<String, u64>,
        addr: &BTreeMap<String, u64>,
        pool: &BTreeMap<String, u64>,
        lmax: u64,
        verbose: bool,
    ) -> Result<(), String> {
        let t0 = Instant::now();
        let tables = self.serial_tables(spec, packer, cache_dir, verbose)?;
        let t_tab = t0.elapsed().as_secs_f64();
        let mut uploaded = 0u64;
        // scales of the FP8 matrices quantised so far (`<W>_f8` sorts before `<W>_f8s`)
        let mut f8_scales: HashMap<String, Vec<f32>> = HashMap::new();
        let (mut f8_hit, mut f8_built) = (0u32, 0u32);
        for (name, &bytes) in weight_specs {
            if !is_drafter_weight(name) {
                continue;
            }
            let data: Cow<'_, [u8]> = match name.as_str() {
                "ds_W_lmd" | "ds_dv_ids" | "ds_W_lmd_f8" => {
                    let ids = super::mtp::draft_vocab_ids(spec, self.cfg.nvd);
                    if name == "ds_dv_ids" {
                        Cow::Owned(ids.iter().flat_map(|i| i.to_le_bytes()).collect())
                    } else {
                        let full = packer.pack_weight(if spec.tie_word_embeddings {
                            "embed_table"
                        } else {
                            "W_lm"
                        })?;
                        let row = (spec.hidden * 2) as usize;
                        let mut dst = Vec::with_capacity(ids.len() * row);
                        for id in ids {
                            let o = id as usize * row;
                            dst.extend_from_slice(&full[o..o + row]);
                        }
                        if name == "ds_W_lmd" {
                            Cow::Owned(dst)
                        } else {
                            // the FP8 readout: one power-of-two scale per draft row
                            let (w8, sc, hit) = self.fp8_quantise(
                                "ds_W_lmd",
                                &dst,
                                self.cfg.nvd as usize,
                                spec.hidden as usize,
                            );
                            if hit {
                                f8_hit += 1;
                            } else {
                                f8_built += 1;
                            }
                            f8_scales.insert(name.clone(), sc);
                            Cow::Owned(w8)
                        }
                    }
                }
                "ds_W_lmd_f8s" => {
                    let sc = f8_scales
                        .remove("ds_W_lmd_f8")
                        .ok_or("dspine: scales of ds_W_lmd_f8 not built")?;
                    Cow::Owned(sc.iter().flat_map(|x| x.to_le_bytes()).collect())
                }
                n => match drafter_layer_suffix(n).and_then(fp8_matrix_of) {
                    Some((_, true)) => {
                        let base = &n[..n.len() - 1];
                        let sc = f8_scales
                            .remove(base)
                            .ok_or_else(|| format!("dspine: scales of {base} not built"))?;
                        Cow::Owned(sc.iter().flat_map(|x| x.to_le_bytes()).collect())
                    }
                    Some((_, false)) => {
                        let (w8, sc, hit) = self.fp8_matrix(&n[..n.len() - 3], lmax, &tables)?;
                        if hit {
                            f8_hit += 1;
                        } else {
                            f8_built += 1;
                        }
                        f8_scales.insert(n.to_string(), sc);
                        Cow::Owned(w8)
                    }
                    None => self.pack(n, lmax, &tables)?,
                },
            };
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
                "[gaudium] dspine: serial tables {:.2}s, uploaded {:.2} GiB drafter weights + tables in {:.2}s",
                t_tab,
                uploaded as f64 / (1u64 << 30) as f64,
                t0.elapsed().as_secs_f64() - t_tab
            );
            if f8_hit + f8_built > 0 {
                eprintln!(
                    "[gaudium] dspine: {} FP8 matrices ({f8_hit} from the cache, {f8_built} quantised)",
                    f8_hit + f8_built
                );
            }
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
            .map_err(|e| format!("bind dspine block0 ws: {e}"))?;
        let exact = select_exact_bindings(&blk0, &self.pool)?;
        self.blk0_launch = Some(
            blk0.prepare_launch(&exact)
                .map_err(|e| format!("prepare dspine block0: {e}"))?,
        );
        self.blk0 = Some(blk0);
        if let Some(mut r) = self.blk0s.take() {
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind dspine block0 (sampled) ws: {e}"))?;
            let exact = select_exact_bindings(&r, &self.pool)?;
            self.blk0s_launch = Some(
                r.prepare_launch(&exact)
                    .map_err(|e| format!("prepare dspine block0 (sampled): {e}"))?,
            );
            self.blk0s = Some(r);
        }
        for r in self.ctx_recipes.values_mut() {
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind dspine ctx ws: {e}"))?;
        }
        Ok(())
    }

    /// The FP8 form of the drafter matrix `name` (`D<l>_W_x`): codes `[n * k]` in the BF16
    /// matrix's own layout and one power-of-two scale per output channel; `true` when it came
    /// from the cache directory. A cached file is keyed by a sampled hash of its BF16 source.
    fn fp8_matrix(
        &self,
        name: &str,
        lmax: u64,
        tables: &SerialTables,
    ) -> Result<(Vec<u8>, Vec<f32>, bool), String> {
        let src = self.pack(name, lmax, tables)?;
        let suffix = drafter_layer_suffix(name).unwrap_or("");
        let (hid, inter) = (self.cfg.hidden, self.cfg.intermediate);
        // HF `[out, in]`
        let (n, k) = match suffix {
            "W_q" => (D_QD, hid),
            "W_k" | "W_v" => (D_KVD, hid),
            "W_o" => (hid, D_QD),
            "W_gate" | "W_up" => (inter, hid),
            "W_down" => (hid, inter),
            _ => return Err(format!("dspine: '{name}' is not an FP8 matrix")),
        };
        Ok(self.fp8_quantise(name, &src, n as usize, k as usize))
    }

    /// Quantises the BF16 matrix `src` (`n` rows of `k`) per row, through the cache directory.
    fn fp8_quantise(
        &self,
        name: &str,
        src: &[u8],
        n: usize,
        k: usize,
    ) -> (Vec<u8>, Vec<f32>, bool) {
        let mut h = Sha256::new();
        h.update_str("gaudium-dspine-fp8-v1");
        h.update_u64(n as u64);
        h.update_u64(k as u64);
        for chunk in src.chunks(1 << 14) {
            h.update(&chunk[..chunk.len().min(32)]);
        }
        let key = format!("{name}-{}", &h.finalize_hex()[..16]);
        if let Some(dir) = &self.f8_dir {
            if let Ok(Some((w8, sc))) = crate::fp8::cache_load_fp8(dir, &key, n, k) {
                return (w8, sc, true);
            }
        }
        let w: Vec<u16> = src
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect();
        let threads = std::thread::available_parallelism().map_or(4, |t| t.get().min(16));
        let (w8, sc, _) = crate::fp8::quantize_rows_bf16(&w, n, k, threads);
        if let Some(dir) = &self.f8_dir {
            // best effort: a read-only model directory only costs the quantisation at each load
            let _ = crate::fp8::cache_store_fp8(dir, &key, &w8, &sc);
        }
        (w8, sc, false)
    }

    /// TG / TSV: a fixed function of the checkpoint and the target's `lm_head`, computed on the
    /// host once and kept beside the recipe cache.
    fn serial_tables(
        &self,
        spec: &ModelSpec,
        packer: &WeightPacker<'_>,
        cache_dir: Option<&Path>,
        verbose: bool,
    ) -> Result<SerialTables, String> {
        let c = &self.cfg;
        let (hid, vocab, cs, r) = (c.hidden, c.vocab, c.code_size, c.rank);
        let b = "selected_prefix_bridge.";
        let lm = packer.pack_weight(if spec.tie_word_embeddings {
            "embed_table"
        } else {
            "W_lm"
        })?;
        let cb = self.need(&format!("{b}codebook"), &[vocab, cs])?;
        let nw = self.f32s("norm.weight", &[hid])?;
        let fw = self.f32s(&format!("{b}feature_write_weights"), &[c.nl as u64, hid, r])?;
        let w_last = &fw[(c.nl - 1) * (hid * r) as usize..];
        let mut pv = self.f32s(&format!("{b}predecessor_gate.weight"), &[r, cs])?;
        pv.extend(self.f32s(&format!("{b}prev.weight"), &[r, cs])?);

        let mut h = Sha256::new();
        h.update_str("gaudium-dspine-serial-tables-v1");
        for d in [hid, vocab, cs, r] {
            h.update_u64(d);
        }
        for bytes in [&lm[..], cb] {
            h.update_u64(bytes.len() as u64);
            for chunk in bytes.chunks(1 << 16) {
                h.update(&chunk[..chunk.len().min(64)]);
            }
        }
        for f in [&nw[..], w_last, &pv[..]] {
            for x in f.iter().step_by(17) {
                h.update(&x.to_le_bytes());
            }
        }
        let key = h.finalize_hex();
        let (tg_bytes, tsv_bytes) = ((vocab * r * 2) as usize, (vocab * 2 * r * 2) as usize);
        let path = cache_dir.map(|d| d.join(format!("dspine-tables-{}.bin", &key[..24])));
        if let Some(p) = &path {
            if let Ok(data) = std::fs::read(p) {
                if data.len() == tg_bytes + tsv_bytes {
                    let tsv = data[tg_bytes..].to_vec();
                    let mut tg = data;
                    tg.truncate(tg_bytes);
                    return Ok(SerialTables { tg, tsv });
                }
            }
        }
        let (tg, tsv) = build_serial_tables(
            &lm,
            &nw,
            w_last,
            cb,
            &pv,
            vocab as usize,
            hid as usize,
            r as usize,
            cs as usize,
        );
        if let Some(p) = &path {
            // write-then-rename: a concurrent or interrupted process never leaves a short file
            let tmp = p.with_extension(format!("tmp{}", std::process::id()));
            let ok = std::fs::write(&tmp, [&tg[..], &tsv[..]].concat()).is_ok()
                && std::fs::rename(&tmp, p).is_ok();
            if !ok {
                let _ = std::fs::remove_file(&tmp);
                if verbose {
                    eprintln!(
                        "[gaudium] dspine: could not store serial tables at {}",
                        p.display()
                    );
                }
            }
        }
        Ok(SerialTables { tg, tsv })
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
        self.launch_ctx_over(
            dev,
            spec,
            cache,
            driver_version,
            elf_hashes,
            lmax,
            pos,
            tile_len,
            None,
        )
    }

    /// [`Self::launch_ctx`] with some of the pass's buffers taken from `over` (tensor name ->
    /// address) instead of the drafter's own: the tap rows and cursor of a tile that the
    /// two-tile prefill loop keeps on its second buffer set.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_ctx_over(
        &mut self,
        dev: &HpuDevice,
        spec: &ModelSpec,
        cache: &RecipeCache,
        driver_version: &str,
        elf_hashes: &[(String, String)],
        lmax: u64,
        pos: u64,
        tile_len: u64,
        over: Option<&BTreeMap<String, u64>>,
    ) -> Result<(), String> {
        let tile_len = if pos + self.full_tile <= lmax {
            self.full_tile
        } else {
            tile_len
        };
        if !self.ctx_recipes.contains_key(&tile_len) {
            let mut r = build_ctx_graph(spec, &self.cfg, tile_len, lmax, elf_hashes)
                .compile(Arc::clone(dev.api()), cache, driver_version)
                .map_err(|e| format!("compile dspine ctx T={tile_len}: {e}"))?;
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind dspine ctx T={tile_len} ws: {e}"))?;
            self.ctx_recipes.insert(tile_len, r);
        }
        let r = &self.ctx_recipes[&tile_len];
        let mut exact = select_exact_bindings(r, &self.pool)?;
        for (name, &addr) in over.into_iter().flatten() {
            if let Some(a) = exact.get_mut(name) {
                *a = addr;
            }
        }
        r.launch_checked(dev, &exact)
            .map_err(|e| format!("launch dspine ctx T={tile_len}: {e}"))
    }

    /// Sets the first position whose context ring row is valid for the request that follows
    /// (`gaudium_ds_softr` reads it as `lim[0]`; a position below it is never attended). Only the
    /// ring form has this bound: with one cache row per position nothing is ever overwritten by
    /// a later position, and the linear kernels read the same tensor as a tree fix-up pair that
    /// must stay `(-1, -1)`. The value is uploaded when it changes.
    pub fn set_context_floor(
        &mut self,
        dev: &HpuDevice,
        stage_io: &mut PinnedBuf,
        floor: u64,
    ) -> Result<(), String> {
        if self.cfg.ring == 0 {
            return Ok(());
        }
        let v = i32::try_from(floor).map_err(|_| format!("dspine context floor {floor}"))?;
        // 0 and "no bound" are the same to the kernel; keep the load-time value for a cold request
        let v = if v == 0 { -1 } else { v };
        if self.ctx_floor == v {
            return Ok(());
        }
        let &addr = self
            .pool
            .get("df_nofix")
            .ok_or("dspine: no context bound tensor")?;
        let buf = stage_io.as_mut_slice();
        buf[..4].copy_from_slice(&v.to_le_bytes());
        buf[4..8].copy_from_slice(&(-1i32).to_le_bytes());
        dev.copy_h2d_pinned_sync(stage_io, 8, addr)
            .map_err(|e| format!("dspine context floor H2D: {e}"))?;
        self.ctx_floor = v;
        Ok(())
    }

    /// [lanes] Names of the pool entries a lane owns: its context caches and context bound.
    pub fn lane_pool_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .pool
            .keys()
            .filter(|n| is_drafter_kv(n))
            .cloned()
            .collect();
        v.push("df_nofix".to_string());
        v
    }

    /// [lanes] Exchanges the lane-owned pool entries and the cached context bound with `lane`
    /// (the active lane's values go into `lane`, `lane`'s become the drafter's).
    pub fn swap_lane(&mut self, lane: &mut DSpineLane) {
        self.lanes_on = true;
        for (name, addr) in lane.pool.iter_mut() {
            if let Some(a) = self.pool.get_mut(name) {
                std::mem::swap(a, addr);
            }
        }
        std::mem::swap(&mut self.ctx_floor, &mut lane.ctx_floor);
    }

    /// A calibration run reads the block's site buffers back after every launch.
    pub fn calibrating(&self) -> bool {
        self.calib.is_some()
    }

    /// Calibration run: folds the block the last launch drafted (its four GEMM input sites per
    /// layer, read from the card) into the running maxima. The caller has synchronised the
    /// compute stream. `addr` = this module's buffer addresses by tensor name.
    pub fn calibrate_observe(
        &mut self,
        dev: &HpuDevice,
        addr: &BTreeMap<String, u64>,
    ) -> Result<(), String> {
        let Some((_, cal)) = self.calib.as_mut() else {
            return Ok(());
        };
        let c = &self.cfg;
        for l in 0..c.nl {
            let xin = if l == 0 {
                "D0_h".to_string()
            } else {
                format!("D{}_bnx", l - 1)
            };
            for (what, tensor, features) in [
                ("xin", xin, c.hidden),
                ("core", format!("D{l}_core"), D_QD),
                ("h2", format!("D{l}_h2"), c.hidden),
                ("m", format!("D{l}_m"), c.intermediate),
            ] {
                let &a = addr
                    .get(&tensor)
                    .ok_or_else(|| format!("dspine calibration: no buffer '{tensor}'"))?;
                cal.observe(dev, &fp8_site(l, what), a, features, c.block)?;
            }
        }
        // the readout's input: the final hidden rows (the last layer's normalised written stream)
        let hf = format!("D{}_bnx", c.nl - 1);
        let &a = addr
            .get(&hf)
            .ok_or_else(|| format!("dspine calibration: no buffer '{hf}'"))?;
        cal.observe(dev, &fp8_site(c.nl, "xin"), a, c.hidden, c.block)?;
        cal.add_rows(c.block);
        Ok(())
    }

    /// Calibration run: writes the table gathered so far (`<site> <k> <amax> <rms>` per line).
    pub fn calibrate_store(&self, verbose: bool) -> Result<(), String> {
        let Some((path, cal)) = self.calib.as_ref() else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        crate::fp8::store_act_scales(path, &cal.table())
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        if verbose {
            eprintln!(
                "[gaudium] dspine FP8 calibration: {} sites over {} block rows -> {}",
                cal.table().len(),
                cal.rows_seen(),
                path.display()
            );
        }
        Ok(())
    }

    /// Whether the coupled-walk recipes of sampled requests were built.
    pub fn has_coupled(&self) -> bool {
        self.blk0s.is_some()
    }

    /// Drafts the first block: anchor `t0` at position `cur`. Returns `block - 1` drafts.
    /// `sampled`: the first block of a sampled request (the coupled walk, when built).
    pub fn launch_block0(
        &mut self,
        dev: &HpuDevice,
        stage_io: &mut PinnedBuf,
        t0: i32,
        cur: i32,
        sampled: bool,
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
        .map_err(|e| format!("dspine block0 H2D: {e}"))?;
        let (r, launch) = match (&self.blk0s, &self.blk0s_launch) {
            (Some(r), Some(l)) if sampled => (r, l),
            _ => (
                self.blk0.as_ref().expect("block0 recipe"),
                self.blk0_launch.as_ref().expect("block0 launch"),
            ),
        };
        if self.lanes_on {
            let exact = select_exact_bindings(r, &self.pool)?;
            r.launch_checked(dev, &exact)
        } else {
            r.launch_prepared(dev, launch)
        }
        .map_err(|e| format!("launch dspine block0: {e}"))?;
        unsafe { dev.sync_stream(dev.compute_stream()) }
            .map_err(|e| format!("sync dspine block0: {e}"))?;
        let n = blk - 1;
        dev.copy_d2h_pinned_sync(self.pool["blk_drafts"], stage_io, n * 4)
            .map_err(|e| format!("dspine block0 D2H: {e}"))?;
        Ok(stage_io.as_slice()[..n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c))
            .collect())
    }
}

/// Device bytes of the serial-refinement tables: `tg` = `[V][R]`, `tsv` = `[V][2R]` (bf16).
pub struct SerialTables {
    tg: Vec<u8>,
    tsv: Vec<u8>,
}

/// bf16 round trip used at the tables' storage points.
#[inline]
pub(super) fn bf16_round(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_rne(x))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const CFG_27B: &str = r#"{
      "architectures": ["DFlashDraftModel"], "block_size": 16,
      "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"},
      "dflash_config": {
        "draft_block_causal": true, "mask_token_id": 248070,
        "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-06, "kappa": 10.0, "rank": 512},
        "selected_prefix_decode_mode": "serial",
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_writes": true,
        "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
        "selected_prefix_terminal": false,
        "target_layer_ids": [1, 16, 31, 46, 61]
      },
      "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
      "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
      "sliding_window": 8192, "vocab_size": 248320
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
        let c = DSpineCfg::from_json(CFG_27B).unwrap();
        assert_eq!(c.block, 16);
        assert_eq!(c.nl, 5);
        assert_eq!((c.code_size, c.rank), (1024, 512));
        assert_eq!(c.mask_id, 248070);
        assert_eq!(c.window, 8192);
        assert_eq!(c.taps, vec![1, 16, 31, 46, 61]);
        c.check_target(&target_27b()).unwrap();

        // a checkpoint without a sliding window attends to the whole context
        let full = CFG_27B.replace("\"sliding_window\": 8192", "\"sliding_window\": null");
        assert!(DSpineCfg::from_json(&full).unwrap().window > 1 << 20);
        // anything the device graph does not implement fails loudly
        for (from, to) in [
            ("\"head_topk\": 16, \"kind\"", "\"head_topk\": 8, \"kind\""),
            (
                "\"selected_prefix_terminal\": false",
                "\"selected_prefix_terminal\": true",
            ),
            (
                "\"draft_block_causal\": true",
                "\"draft_block_causal\": false",
            ),
            ("\"rank\": 512", "\"rank\": 640"),
            ("[1, 2, 3, 4, 5]", "[1, 2, 3]"),
        ] {
            assert!(
                DSpineCfg::from_json(&CFG_27B.replace(from, to)).is_err(),
                "{from}"
            );
        }
        // a drafter for another target is refused
        let other = CFG_27B.replace("\"hidden_size\": 5120", "\"hidden_size\": 4096");
        assert!(DSpineCfg::from_json(&other)
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
            "ds_TG",
            "ds_R3",
            "ds_wnext4",
            "D2_W_gate",
            "D0_qn_w",
            "rope128_s",
        ] {
            assert!(is_drafter_weight(w), "{w}");
        }
        for a in [
            "D0_q",
            "D0_sc",
            "blk_drafts",
            "ctx_cat",
            "L0_W_o",
            "D0_kcache",
        ] {
            assert!(!is_drafter_weight(a), "{a}");
        }
    }

    #[test]
    fn context_rings_are_chosen_by_window_kernels_and_size() {
        let mut dc = DSpineCfg::from_json(CFG_27B).unwrap();
        assert_eq!(dc.window, 8192);
        // serving context: the window is far shorter, a ring holds window + tile + guard
        dc.plan_context(262_144, 512, true);
        assert_eq!(dc.ring, 8960);
        assert_eq!(dc.ring % 128, 0);
        assert_eq!(dc.ctx_rows(262_144), 8960);
        // no ring kernels in the kernel directory: linear caches
        dc.plan_context(262_144, 512, false);
        assert_eq!((dc.ring, dc.ctx_rows(262_144)), (0, 262_144));
        // a context no longer than the window, or one a ring would not be smaller than
        dc.plan_context(4096, 512, true);
        assert_eq!((dc.ring, dc.ctx_rows(4096)), (0, 4096));
        dc.plan_context(8192, 512, true);
        assert_eq!(dc.ring, 0);
        dc.plan_context(8960, 512, true);
        assert_eq!(dc.ring, 0);
        dc.plan_context(12_288, 512, true);
        assert_eq!(dc.ring, 8960);
    }

    /// The slot arithmetic of the ring kernels (`gaudium_ds_ctxr`, `gaudium_ds_softr`), written
    /// with the verified window functions, against the positions a block row may read.
    #[test]
    fn ring_slots_of_a_window_are_its_positions() {
        use crate::verified::{ds_win_low, ds_win_rows, ds_win_wrap};
        use std::collections::BTreeSet;
        for (rg, w, tmax) in [(384u64, 100u64, 16u64), (512, 200, 16), (1024, 513, 64), (768, 64, 128)] {
            assert!(rg % 128 == 0 && rg >= w + tmax + 128, "the glue's condition on a ring");
            for cur in (0..3 * rg + 7).step_by(37).chain([rg - 1, rg, rg + 1, 2 * rg]) {
                for flo in [0u64, cur / 2, cur] {
                    let span = cur.saturating_sub(flo);
                    // the slot after the newest context row, in 1..=rg
                    let hi = (cur + rg - 1) % rg + 1;
                    // slots the pass of this step writes
                    let written: BTreeSet<u64> = (0..tmax).map(|t| (cur + t) % rg).collect();
                    for t in 0..tmax.min(16) {
                        let nv = ds_win_rows(w, t, span);
                        let (low, wrap) = (ds_win_low(nv, hi), ds_win_wrap(nv, hi));
                        let slots: BTreeSet<u64> = (hi - low..hi).chain(rg - wrap..rg).collect();
                        let want: BTreeSet<u64> = (cur - nv..cur).map(|p| p % rg).collect();
                        assert_eq!(slots, want, "rg {rg} w {w} cur {cur} flo {flo} t {t}");
                        assert_eq!(slots.len() as u64, nv, "no slot is read twice");
                        assert!(slots.is_disjoint(&written), "a pass never writes a visible slot");
                        // the oldest visible position: the window's start or the request's
                        assert_eq!(cur - nv, (cur + t).saturating_sub(w - 1).max(flo).min(cur));
                    }
                }
            }
        }
    }

    #[test]
    fn draft_readout_rows_follow_the_request_and_the_fp8_copy() {
        use crate::verified::{ds_draft_mapped, ds_draft_rows};
        let v = 248_320u64;
        // no subset asked: every row with an FP8 copy of the readout, the plain readout without
        assert_eq!(ds_draft_rows(0, v, true), v);
        assert_eq!(ds_draft_rows(0, v, false), 0);
        // a subset asked is kept either way
        assert_eq!(ds_draft_rows(98_304, v, true), 98_304);
        assert_eq!(ds_draft_rows(98_304, v, false), 98_304);
        // only a proper subset maps rows to token ids
        assert!(ds_draft_mapped(98_304, v));
        assert!(!ds_draft_mapped(v, v) && !ds_draft_mapped(0, v));
    }

    #[test]
    fn draft_vocabulary_and_fp8_layers_change_only_the_drafter() {
        let spec = target_27b();
        let base = DSpineCfg::from_json(CFG_27B).unwrap();
        assert_eq!((base.nvd, base.fp8), (0, false));
        let mut dc = base.clone();
        dc.nvd = 98_304;
        dc.fp8 = true;
        let g0 = super::super::build_spec_step_graph(&spec, 4096, 1, 8, 16, Some(&base), &[]);
        let g1 = super::super::build_spec_step_graph(&spec, 4096, 1, 8, 16, Some(&dc), &[]);
        let s0 = g0.persistent_tensor_specs();
        let s1 = g1.persistent_tensor_specs();
        let d1 = |n: &str| s1.iter().find(|t| t.0 == n).map(|t| (t.1, t.2.clone()));
        // the block readout streams the draft rows; candidates map back to token ids
        assert_eq!(
            d1("ds_W_lmd"),
            Some((SynDataType::Bf16, vec![5120, 98_304]))
        );
        assert_eq!(
            d1("blk_logits"),
            Some((SynDataType::Bf16, vec![98_304, 16]))
        );
        assert_eq!(d1("ds_dv_ids"), Some((SynDataType::Int32, vec![1, 98_304])));
        assert_eq!(d1("blk_cand_l"), Some((SynDataType::Int32, vec![16, 16])));
        assert_eq!(d1("blk_cand"), Some((SynDataType::Int32, vec![16, 16])));
        // the verify readout keeps the full head
        assert_eq!(d1("logits"), Some((SynDataType::Bf16, vec![248_320, 8])));
        // FP8 layers: codes + per-channel scales per matrix, one cast per input site
        assert_eq!(
            d1("D0_W_gate_f8"),
            Some((SynDataType::Fp8_143, vec![5120, 17_408]))
        );
        assert_eq!(
            d1("D0_W_gate_f8s"),
            Some((SynDataType::Float, vec![17_408]))
        );
        assert_eq!(
            d1("D2_h2_xq8"),
            Some((SynDataType::Fp8_143, vec![5120, 16]))
        );
        assert_eq!(s1.iter().filter(|t| t.0 == "D2_h2_xq8").count(), 1);
        assert_eq!(d1("ds_D4_m_cs"), Some((SynDataType::Float, vec![1])));
        // the context pass keeps the BF16 key / value projections; the other BF16 layer
        // matrices are gone from the graph (and from the device plan)
        assert!(d1("D0_W_k").is_some() && d1("D0_W_v").is_some());
        for gone in ["D0_W_q", "D0_W_o", "D0_W_gate", "D0_W_up", "D0_W_down"] {
            assert!(d1(gone).is_none(), "{gone}");
            assert!(s0.iter().any(|t| t.0 == gone), "{gone}");
        }
        // every tensor the options add is planned as a drafter weight or as scratch
        for w in [
            "ds_W_lmd",
            "ds_dv_ids",
            "D0_W_q_f8",
            "D3_W_down_f8s",
            "ds_D1_xin_sa",
        ] {
            assert!(is_drafter_weight(w), "{w}");
        }
        assert!(!is_drafter_weight("D2_h2_xq8") && !is_drafter_weight("blk_cand_l"));
        assert_eq!(fp8_matrix_of("W_q_f8"), Some(("W_q", false)));
        assert_eq!(fp8_matrix_of("W_down_f8s"), Some(("W_down", true)));
        assert_eq!(fp8_matrix_of("kn_w"), None);
        // the trunk of both graphs is the same node list
        let trunk = |g: &GraphBuilder| -> Vec<String> {
            g.node_io_views()
                .iter()
                .filter(|n| n.name.starts_with('L') || n.name == "accept")
                .map(|n| format!("{}:{}", n.guid, n.name))
                .collect()
        };
        assert_eq!(trunk(&g0), trunk(&g1));
        assert_ne!(g0.fingerprint("x"), g1.fingerprint("x"));
        assert_eq!(
            default_fp8_dir(Path::new("/m/Drafter-b16")),
            Path::new("/m/Drafter-b16-gaudium-pc")
        );
    }

        #[test]
    fn step_graph_with_drafter_shares_caches_between_context_and_block() {
        let spec = target_27b();
        let dc = DSpineCfg::from_json(CFG_27B).unwrap();
        let gb = super::super::build_spec_step_graph(&spec, 4096, 1, 8, 16, Some(&dc), &[]);
        // the sampled family: same trunk and drafter, the sampler and the coupled walk instead
        // of the argmax and the greedy walk; the greedy graph is untouched by its existence
        let gs =
            super::super::build_spec_step_graph_as(&spec, 4096, 1, 8, 16, Some(&dc), &[], true);
        let guids = |g: &GraphBuilder| -> Vec<String> {
            g.node_io_views()
                .iter()
                .map(|n| n.guid.to_string())
                .collect()
        };
        let (gg, sg) = (guids(&gb), guids(&gs));
        assert!(
            gg.iter().any(|g| g == "argmax_fwd_bf16") && gg.iter().any(|g| g == "gaudium_df_walk")
        );
        assert!(!gg
            .iter()
            .any(|g| g.starts_with("gaudium_samp_") || g == "gaudium_ds_cwalk"));
        assert!(!sg
            .iter()
            .any(|g| g == "argmax_fwd_bf16" || g == "gaudium_df_walk"));
        assert_eq!(sg.iter().filter(|g| *g == "gaudium_ds_cwalk").count(), 1);
        assert_eq!(sg.iter().filter(|g| *g == "gaudium_samp_gumbel").count(), 1);
        assert_eq!(sg.iter().filter(|g| *g == "gaudium_accept_t").count(), 1);
        let ss = gs.persistent_tensor_specs();
        let sdims = |n: &str| ss.iter().find(|t| t.0 == n).map(|t| t.2.clone());
        assert_eq!(sdims("blk_cwcs"), Some(vec![64, 15]));
        assert_eq!(sdims("tokens"), Some(vec![1, 8]));
        assert_eq!(sdims("samp_i"), Some(vec![8]));
        let cw = gs.node_io_views();
        let cw = cw.iter().find(|n| n.guid == "gaudium_ds_cwalk").unwrap();
        assert_eq!(cw.inputs[5].name, "cursor_out");
        assert_eq!(cw.inputs[2].name, "D4_brms");
        let specs = gb.persistent_tensor_specs();
        let count = |n: &str| specs.iter().filter(|t| t.0 == n).count();
        // one tensor per cache: written by the context pass, read by the block pass
        assert_eq!(count("D0_kcache"), 1);
        assert_eq!(count("D4_vcache"), 1);
        assert_eq!(count("W_lm"), 1);
        let dims = |n: &str| specs.iter().find(|t| t.0 == n).map(|t| t.2.clone());
        assert_eq!(dims("next_ids"), Some(vec![16]));
        assert_eq!(dims("ctx_cat"), Some(vec![5 * 5120, 8]));
        assert_eq!(dims("blk_scores"), Some(vec![16, 16, 15]));
        assert_eq!(dims("ds_TSV"), Some(vec![1024, 248320]));
        assert_eq!(dims("rope128_c"), Some(vec![128, 4096 + ROPE_PAD]));
        assert_eq!(dims("D0_Pc"), Some(vec![4096, 64, 8]));
    }
}
