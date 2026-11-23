//! The Inkling 2 block drafter (the checkpoint's `mtp.safetensors`) on the card: the speculative
//! drafter of [`Inkling2Engine`]. A child module of `engine::inkling2` (declared there with
//! `#[path]`).
//!
//! The drafter is a small bidirectional decoder over one block of rows (the anchor, the last
//! committed token, and `block - 1` mask rows at the positions after it), followed by a recurrent
//! head that corrects the mask rows' logits one after another. What it computes and where:
//!
//! - **Context** (one row per committed position, kept between blocks): four hidden states of the
//!   main model at that position (the inputs of three layer applications and the final norm
//!   output) are mixed with softmax weights and normalised once per drafter layer
//!   (`gaudium_ik_dmix`); each layer projects its own keys and values of the row and stores them by
//!   position (`gaudium_ik_dkv`: key norm, rotary position). A layer with a window keeps a ring of
//!   a power-of-two number of slots, the global layer a chunk-major cache of the whole context.
//!   The context recipe runs on the tap tensors a pass of the main model left behind: per prefill
//!   tile, and after every verify pass on the verify's own rows (rows a verify rejected are
//!   written too; they sit at positions at or beyond the next anchor, which no block reads as
//!   context, and are written again when those positions are committed).
//! - **Block** (one recipe per round, after the host has read the verify's tokens): the anchor's
//!   normed embedding and the mask rows through the layers (pre-norm attention over the layer's
//!   context state and the block's own keys, without a causal mask inside the block; SwiGLU), the
//!   final norm, base logits of every row over the main model's unembedding, then the recurrent
//!   head inside the same recipe: per mask row a 256-wide gated state driven by the previously
//!   drafted token's row of `markov_w1`, a latent folded through the unembedding into a vocabulary
//!   bias, and the arg-max. The previous token is a device tensor (the arg-max before it), so the
//!   chain has no host round trip.
//!
//! The drafter only proposes: every emitted token is the verify pass's own.

use super::*;
use crate::safetensors::SafetensorsStore;
use crate::verified::{
    mtp_ctx_chunks, mtp_ctx_seen, mtp_draft_count, mtp_frnn_row, mtp_ring_ok, mtp_ring_slots,
};

/// The drafter checkpoint beside the main model's files.
pub const MTP_FILE: &str = "mtp.safetensors";

/// Kernel ELFs of the drafter's graphs (beside the family's own).
pub(super) const MTP_ELFS: &[&str] = &[
    "GAUDIUM_IKDMIX_ELF",
    "GAUDIUM_IKDKV_ELF",
    "GAUDIUM_IKDQPREP_ELF",
    "GAUDIUM_IKDSOFT_ELF",
    "GAUDIUM_IKDCOMB_ELF",
    "GAUDIUM_IKDSWIGLU_ELF",
    "GAUDIUM_IKMSTEP_ELF",
    "GAUDIUM_IKMARGMAX_ELF",
    "GAUDIUM_GARMSNORM_ELF",
];

/// Channels of a head (fixed in the drafter kernels).
const DH: u64 = 128;
/// Rows of one level of the rotary tables (`pos = 256 hi + lo`).
const ROPE_LO: u64 = 256;
/// Main-model tensors a drafter graph reads by their own names.
const MAIN_SHARED: &[&str] = &["W_head", "embed", "w_embed"];

/// The drafter's contract, read from `mtp_config` of the checkpoint's `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct MtpConfig {
    pub hid: u64,
    pub nh: u64,
    pub nkv: u64,
    pub ff: u64,
    /// Per layer: attends a window of the context (a ring) instead of all of it.
    pub local: Vec<bool>,
    pub window: u64,
    /// Rows of a block: the anchor and `block - 1` mask rows.
    pub block: u64,
    pub rank: u64,
    pub theta: f64,
    pub eps: f32,
    /// Main-model layer applications whose input feeds the context; the application count itself
    /// names the final norm output.
    pub taps: Vec<u64>,
}

impl MtpConfig {
    /// Reads `mtp_config` from the text of a `config.json`.
    pub fn from_config_json(text: &str) -> Result<Self, String> {
        let bad = |what: &str| format!("Inkling 2 drafter: mtp_config: {what}");
        let root = parse_json(text).map_err(|e| bad(&format!("config.json: {e:?}")))?;
        let m = root
            .get("mtp_config")
            .ok_or_else(|| bad("config.json has no mtp_config"))?;
        let num = |k: &str| m.get(k).and_then(JsonValue::as_u64);
        let need = |k: &str| num(k).ok_or_else(|| bad(&format!("no {k}")));
        let n_layers = need("num_hidden_layers")?;
        let list = |k: &str| -> Result<Vec<u64>, String> {
            Ok(m.get(k)
                .and_then(JsonValue::as_array)
                .ok_or_else(|| bad(&format!("no {k}")))?
                .iter()
                .filter_map(JsonValue::as_u64)
                .collect())
        };
        let local_ids = list("local_layer_ids")?;
        let kind = m
            .get("markov_head_type")
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if kind != "rnn_hidden" {
            return Err(bad(&format!(
                "markov_head_type \"{kind}\" (rnn_hidden is built)"
            )));
        }
        for flag in ["q_bias", "o_bias"] {
            if m.get(flag).and_then(JsonValue::as_bool).unwrap_or(false) {
                return Err(bad(&format!(
                    "{flag} is set (the drafter is built without biases)"
                )));
            }
        }
        let c = Self {
            hid: need("hidden_size")?,
            nh: need("num_attention_heads")?,
            nkv: need("num_key_value_heads")?,
            ff: need("intermediate_size")?,
            local: (0..n_layers).map(|l| local_ids.contains(&l)).collect(),
            window: need("sliding_window_size")?,
            block: need("block_size")?,
            rank: need("markov_rank")?,
            theta: m
                .get("rope_theta")
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| bad("no rope_theta"))?,
            eps: m
                .get("rms_norm_eps")
                .and_then(JsonValue::as_f64)
                .unwrap_or(1e-6) as f32,
            taps: list("target_layer_ids")?,
        };
        if num("head_dim") != Some(DH) {
            return Err(bad("head_dim must be 128 (fixed in the drafter kernels)"));
        }
        if c.hid == 0 || c.hid % 128 != 0 || c.ff == 0 || c.ff % 128 != 0 {
            return Err(bad(
                "hidden_size / intermediate_size must be whole blocks of 128",
            ));
        }
        if c.nkv == 0 || c.nh == 0 || c.nh % c.nkv != 0 {
            return Err(bad(
                "heads must be a whole number of query heads per key / value head",
            ));
        }
        if c.rank == 0 || c.rank % 128 != 0 {
            return Err(bad("markov_rank must be whole blocks of 128"));
        }
        if !(2..=64).contains(&c.block) {
            return Err(bad(&format!("block_size {} (2..=64)", c.block)));
        }
        if c.local.is_empty() || c.taps.len() != 4 {
            return Err(bad(
                "the drafter needs layers and exactly four target_layer_ids",
            ));
        }
        if c.local.iter().any(|&l| l) && c.window == 0 {
            return Err(bad("a windowed layer needs sliding_window_size > 0"));
        }
        Ok(c)
    }

    /// Where each of the four taps is in the main model's passes.
    pub fn check_target(&self, d: &IkDims) -> Result<[MtpTap; 4], String> {
        if self.hid != d.hid {
            return Err(format!(
                "Inkling 2 drafter: hidden size {} is not the model's {}",
                self.hid, d.hid
            ));
        }
        if d.n_apps != d.apps.len() {
            return Err(
                "Inkling 2 drafter: needs every layer application (GAUDIUM_IK_APPS cuts them)"
                    .into(),
            );
        }
        let mut out = [MtpTap::Final; 4];
        for (o, &a) in out.iter_mut().zip(&self.taps) {
            *o = if (a as usize) < d.apps.len() {
                MtpTap::Source(d.wires[a as usize].x_in)
            } else if a as usize == d.apps.len() {
                MtpTap::Final
            } else {
                return Err(format!(
                    "Inkling 2 drafter: target_layer_ids names application {a} of {}",
                    d.apps.len()
                ));
            };
        }
        Ok(out)
    }
}

/// One of the main model's states the drafter's context is computed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpTap {
    /// A source of the wiring (the input of a layer application).
    Source(usize),
    /// The final norm output.
    Final,
}

impl MtpTap {
    /// The decode step's tensor that holds this tap.
    fn step_name(self) -> String {
        match self {
            MtpTap::Source(0) => "x0".into(),
            MtpTap::Source(s) => format!("A{}_out", s - 1),
            MtpTap::Final => "h_final".into(),
        }
    }
}

/// The drafter's geometry in one engine configuration.
#[derive(Debug, Clone)]
pub struct MtpGeom {
    pub c: MtpConfig,
    /// Slots of a windowed layer's context ring.
    pub ring: u64,
    /// Rows of the widest context pass (a prefill tile or a verify pass).
    pub ctx_rows: u64,
    pub max_ctx: u64,
    pub max_chunks: u64,
    /// Rows of the coarse rotary table.
    pub rope_hi: u64,
    pub vocab_in: u64,
    pub vocab_out: u64,
    pub vocab_pad: u64,
}

impl MtpGeom {
    pub fn new(c: MtpConfig, d: &IkDims, max_ctx: u64, ctx_rows: u64) -> Result<Self, String> {
        let ring = mtp_ring_slots(c.window, ctx_rows);
        if !mtp_ring_ok(ring, c.window, ctx_rows) {
            return Err(format!(
                "Inkling 2 drafter: no context ring holds a window of {} beside passes of {ctx_rows} rows",
                c.window
            ));
        }
        // positions the block rows can reach: the context and one block past it
        let max_pos = max_ctx + c.block;
        Ok(Self {
            ring,
            ctx_rows,
            max_ctx,
            max_chunks: max_ctx / IK_CHUNK,
            rope_hi: max_pos / ROPE_LO + 2,
            vocab_in: d.vocab_in,
            vocab_out: d.vocab_out,
            vocab_pad: d.vocab_pad,
            c,
        })
    }

    fn kvw(&self) -> u64 {
        2 * DH * self.c.nkv
    }

    fn gq(&self) -> u64 {
        self.c.nh / self.c.nkv
    }

    /// Drafts one block yields for a verify pass of `k` drafts.
    pub fn drafts(&self, k: u64) -> u64 {
        mtp_draft_count(self.c.block, k)
    }
}

// ============================================================================
// Graph builders (host-testable without the card)
// ============================================================================

fn rope(gb: &mut GraphBuilder, g: &MtpGeom) -> (TensorId, TensorId) {
    let f = SynDataType::Float;
    (
        gb.get_or_add_persistent_tensor("mtw_rope_hi", f, &[DH, 2, g.rope_hi]),
        gb.get_or_add_persistent_tensor("mtw_rope_lo", f, &[DH, 2, ROPE_LO]),
    )
}

/// A layer's context keys / values: a ring of `ring` slots, or the cache's first `nch` chunks.
fn state(gb: &mut GraphBuilder, g: &MtpGeom, l: usize, nch: u64) -> (TensorId, TensorId, u64, u64) {
    let (c, b, nx) = if g.c.local[l] {
        (g.ring, g.c.nkv, 1)
    } else {
        (IK_CHUNK, g.c.nkv * nch, nch)
    };
    let dims = [DH, c, b];
    (
        gb.get_or_add_persistent_tensor(format!("mts_L{l}_k"), SynDataType::Bf16, &dims),
        gb.get_or_add_persistent_tensor(format!("mts_L{l}_v"), SynDataType::Bf16, &dims),
        c,
        nx,
    )
}

fn rms(
    gb: &mut GraphBuilder,
    x: TensorId,
    w: &str,
    y: &str,
    hid: u64,
    t: u64,
    eps: f32,
) -> TensorId {
    let w = gb.get_or_add_persistent_tensor(w, SynDataType::Bf16, &[hid]);
    let out = gb.add_persistent_tensor(y, SynDataType::Bf16, &[hid, t]);
    gb.add_node(
        "gaudium_ga_rmsnorm",
        format!("{y}_norm"),
        &[x, w],
        &[out],
        &eps.to_le_bytes(),
    );
    out
}

/// Context recipe `g_ikd_ctx_T{t}`: the four taps `mti_tap0..3 {hid, t}` of rows at positions
/// `mti_ccur + row` -> every layer's context keys / values, and the Markov head's mixes `mt_c3`.
pub fn build_ctx_graph(g: &MtpGeom, t: u64, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let c = &g.c;
    let bf = SynDataType::Bf16;
    let mut gb = ik_graph(
        format!("g_ikd_ctx_T{t}_C{}_R{}", g.max_chunks, g.ring),
        elf_hashes,
    );
    let taps: Vec<TensorId> = (0..4)
        .map(|j| gb.add_persistent_tensor(format!("mti_tap{j}"), bf, &[c.hid, t]))
        .collect();
    let cur = gb.add_persistent_tensor("mti_ccur", SynDataType::Int32, &[1]);
    let w = gb.get_or_add_persistent_tensor("mtw_wmix", bf, &[c.hid, 4]);
    let alpha = gb.get_or_add_persistent_tensor("mtw_alpha", SynDataType::Float, &[16]);
    let mixes: Vec<TensorId> = (0..4)
        .map(|l| gb.add_persistent_tensor(format!("mt_c{l}"), bf, &[c.hid, t]))
        .collect();
    gb.add_node(
        "gaudium_ik_dmix",
        "mt_mix",
        &[taps[0], taps[1], taps[2], taps[3], w, alpha],
        &mixes,
        &u32_params(&[f32_bits(c.eps)]),
    );
    let (hi, lo) = rope(&mut gb, g);
    for l in 0..c.local.len() {
        let kv = ik_linear(
            &mut gb,
            mixes[l.min(2)],
            &format!("mtw_L{l}_Wkvc"),
            c.hid,
            g.kvw(),
            t,
            &format!("mt_ckv{l}"),
        );
        let kn = gb.get_or_add_persistent_tensor(format!("mtw_L{l}_knc"), bf, &[DH]);
        let (ks, vs, _, _) = state(&mut gb, g, l, g.max_chunks);
        gb.add_node(
            "gaudium_ik_dkv",
            format!("mt_kv{l}"),
            &[kv, kn, hi, lo, cur],
            &[ks, vs],
            &u32_params(&[u32::from(c.local[l]), f32_bits(c.eps)]),
        );
    }
    gb
}

/// One drafter layer over the block rows at positions `ctl[0] + row`.
fn append_layer(
    gb: &mut GraphBuilder,
    g: &MtpGeom,
    x: TensorId,
    l: usize,
    nch: u64,
    ctl: TensorId,
) -> TensorId {
    let c = &g.c;
    let (bf, f32t) = (SynDataType::Bf16, SynDataType::Float);
    let t = c.block;
    let p = format!("mt_L{l}_");
    let (hi, lo) = rope(gb, g);
    let a = rms(
        gb,
        x,
        &format!("mtw_L{l}_win"),
        &format!("{p}a"),
        c.hid,
        t,
        c.eps,
    );
    let q = ik_linear(
        gb,
        a,
        &format!("mtw_L{l}_Wq"),
        c.hid,
        DH * c.nh,
        t,
        &format!("{p}q"),
    );
    let kv = ik_linear(
        gb,
        a,
        &format!("mtw_L{l}_Wkv"),
        c.hid,
        g.kvw(),
        t,
        &format!("{p}kv"),
    );
    let (ks, vs, cs, nx) = state(gb, g, l, nch);
    let r = g.gq() * t;
    let qn = gb.get_or_add_persistent_tensor(format!("mtw_L{l}_qn"), bf, &[DH]);
    let kn = gb.get_or_add_persistent_tensor(format!("mtw_L{l}_kn"), bf, &[DH]);
    let qb = gb.add_persistent_tensor(format!("{p}qb"), bf, &[DH, r, c.nkv * nx]);
    let qbb = gb.add_persistent_tensor(format!("{p}qbb"), bf, &[DH, r, c.nkv]);
    let kb = gb.add_persistent_tensor(format!("{p}kb"), bf, &[DH, 128, c.nkv]);
    let vb = gb.add_persistent_tensor(format!("{p}vb"), bf, &[DH, 128, c.nkv]);
    gb.add_node(
        "gaudium_ik_dqprep",
        format!("{p}qprep"),
        &[q, kv, qn, kn, hi, lo, ctl],
        &[qb, qbb, kb, vb],
        &u32_params(&[f32_bits(c.eps)]),
    );
    let sc = gb.add_persistent_tensor(format!("{p}sc"), f32t, &[cs, r, c.nkv * nx]);
    gb.op_batch_gemm(format!("{p}qk_ctx"), qb, ks, sc, false, true);
    let sb = gb.add_persistent_tensor(format!("{p}sb"), f32t, &[128, r, c.nkv]);
    gb.op_batch_gemm(format!("{p}qk_blk"), qbb, kb, sb, false, true);
    let pc = gb.add_persistent_tensor(format!("{p}Pc"), bf, &[cs, r, c.nkv * nx]);
    let pb = gb.add_persistent_tensor(format!("{p}Pb"), bf, &[128, r, c.nkv]);
    let st = gb.add_persistent_tensor(format!("{p}st"), f32t, &[64, 2, r, c.nkv * (nx + 1)]);
    // how many context positions the rows see comes from the host (`ctl[1 + local]`)
    gb.add_node(
        "gaudium_ik_dsoft",
        format!("{p}softmax"),
        &[sc, sb, ctl],
        &[pc, pb, st],
        &u32_params(&[
            t as u32,
            u32::from(c.local[l]),
            u32::from(c.local[l]),
            f32_bits((DH as f32).powf(-0.5)),
        ]),
    );
    let aoc = gb.add_persistent_tensor(format!("{p}aoc"), f32t, &[DH, r, c.nkv * nx]);
    gb.op_batch_gemm(format!("{p}pv_ctx"), pc, vs, aoc, false, false);
    let aob = gb.add_persistent_tensor(format!("{p}aob"), f32t, &[DH, r, c.nkv]);
    gb.op_batch_gemm(format!("{p}pv_blk"), pb, vb, aob, false, false);
    let core = gb.add_persistent_tensor(format!("{p}core"), bf, &[DH * c.nh, t]);
    gb.add_node(
        "gaudium_ik_dcomb",
        format!("{p}comb"),
        &[aoc, aob, st],
        &[core],
        &u32_params(&[t as u32]),
    );
    let o = ik_linear(
        gb,
        core,
        &format!("mtw_L{l}_Wo"),
        DH * c.nh,
        c.hid,
        t,
        &format!("{p}o"),
    );
    let h1 = gb.add_persistent_tensor(format!("{p}h1"), bf, &[c.hid, t]);
    gb.op_add_bf16(format!("{p}res1"), x, o, h1);
    let b = rms(
        gb,
        h1,
        &format!("mtw_L{l}_wpost"),
        &format!("{p}b"),
        c.hid,
        t,
        c.eps,
    );
    let gu = ik_linear(
        gb,
        b,
        &format!("mtw_L{l}_Wgu"),
        c.hid,
        2 * c.ff,
        t,
        &format!("{p}gu"),
    );
    let act = gb.add_persistent_tensor(format!("{p}act"), bf, &[c.ff, t]);
    gb.add_node(
        "gaudium_ik_dswiglu",
        format!("{p}swiglu"),
        &[gu],
        &[act],
        &[],
    );
    let m = ik_linear(
        gb,
        act,
        &format!("mtw_L{l}_Wd"),
        c.ff,
        c.hid,
        t,
        &format!("{p}m"),
    );
    let out = gb.add_persistent_tensor(format!("{p}out"), bf, &[c.hid, t]);
    gb.op_add_bf16(format!("{p}res2"), h1, m, out);
    out
}

/// Block recipe `g_ikd_blk_C{nch}`: the anchor `mt_anchor` at position `mt_ctl[0]`, the first
/// numbers of context positions the layers see in `mt_ctl[1]` (global) and `mt_ctl[2]` (windowed) (the global
/// layer reads `nch` chunks of its cache), the Markov context from column `mt_fidx` of `mt_c3`
/// -> `mt_drafts {1, block - 1}`.
pub fn build_block_graph(g: &MtpGeom, nch: u64, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let c = &g.c;
    let (bf, i32t) = (SynDataType::Bf16, SynDataType::Int32);
    let b = c.block;
    let mut gb = ik_graph(
        format!("g_ikd_blk_B{b}_C{nch}_R{}_X{}", g.ring, g.ctx_rows),
        elf_hashes,
    );
    let anchor = gb.add_persistent_tensor("mt_anchor", i32t, &[1]);
    let fidx = gb.add_persistent_tensor("mt_fidx", i32t, &[b]);
    let ctl = gb.add_persistent_tensor("mt_ctl", i32t, &[8]);
    // the anchor's normed embedding, then the mask rows
    let table = gb.get_or_add_persistent_tensor("embed", bf, &[c.hid, g.vocab_in]);
    let e0 = gb.add_persistent_tensor("mt_e0", bf, &[c.hid, 1]);
    gb.op_gather_bf16("mt_embed", table, anchor, e0, 1);
    let x0 = rms(&mut gb, e0, "w_embed", "mt_x0", c.hid, 1, c.eps);
    let mask = gb.get_or_add_persistent_tensor("mtw_mask", bf, &[c.hid, b - 1]);
    let mut x = gb.add_persistent_tensor("mt_x", bf, &[c.hid, b]);
    gb.op_concat("mt_block_in", &[x0, mask], x, 1);
    for l in 0..c.local.len() {
        x = append_layer(&mut gb, g, x, l, nch, ctl);
    }
    let z = rms(&mut gb, x, "mtw_norm", "mt_z", c.hid, b, c.eps);
    let logits = ik_linear(&mut gb, z, "W_head", c.hid, g.vocab_pad, b, "mt_logits");
    // the Markov head's context row, beside every block row
    let c3 = gb.get_or_add_persistent_tensor("mt_c3", bf, &[c.hid, g.ctx_rows]);
    let fb = gb.add_persistent_tensor("mt_fb", bf, &[c.hid, b]);
    gb.op_gather_bf16("mt_frnn", c3, fidx, fb, 1);
    let zf = gb.add_persistent_tensor("mt_zf", bf, &[2 * c.hid, b]);
    gb.op_concat("mt_zf_cat", &[z, fb], zf, 0);
    let rk = c.rank;
    let ctxp = ik_linear_f32(&mut gb, zf, "mtw_Wzf", 2 * c.hid, 3 * rk, b, "mt_ctxp");
    let rb = gb.get_or_add_persistent_tensor("mtw_rnnb", SynDataType::Float, &[3 * rk]);
    let w1 = gb.get_or_add_persistent_tensor("mtw_w1", bf, &[rk, g.vocab_in]);
    let bfold = gb.get_or_add_persistent_tensor("mtw_bf", bf, &[g.vocab_pad]);
    let mut s = gb.get_or_add_persistent_tensor("mtw_zero", bf, &[rk, 1]);
    let mut prev = anchor;
    let mut drafts = Vec::new();
    for i in 1..b {
        let w1p = gb.add_persistent_tensor(format!("mt_w1p{i}"), bf, &[rk, 1]);
        gb.op_gather_bf16(format!("mt_w1g{i}"), w1, prev, w1p, 1);
        let sw = gb.add_persistent_tensor(format!("mt_sw{i}"), bf, &[2 * rk, 1]);
        gb.op_concat(format!("mt_sw_cat{i}"), &[s, w1p], sw, 0);
        let swp = ik_linear_f32(
            &mut gb,
            sw,
            "mtw_Wsw",
            2 * rk,
            3 * rk,
            1,
            &format!("mt_swp{i}"),
        );
        let sn = gb.add_persistent_tensor(format!("mt_s{i}"), bf, &[rk, 1]);
        let lat = gb.add_persistent_tensor(format!("mt_lat{i}"), bf, &[rk, 1]);
        gb.add_node(
            "gaudium_ik_mstep",
            format!("mt_step{i}"),
            &[ctxp, swp, rb, s],
            &[sn, lat],
            &u32_params(&[i as u32]),
        );
        let bias = ik_linear(
            &mut gb,
            lat,
            "mtw_fold",
            rk,
            g.vocab_pad,
            1,
            &format!("mt_bias{i}"),
        );
        let di = gb.add_persistent_tensor(format!("mt_d{i}"), i32t, &[1, 1]);
        gb.add_node(
            "gaudium_ik_margmax",
            format!("mt_argmax{i}"),
            &[logits, bias, bfold],
            &[di],
            &u32_params(&[i as u32, g.vocab_out as u32]),
        );
        prev = ik_view(&mut gb, di, &format!("mt_dv{i}"), i32t, &[1]);
        s = sn;
        drafts.push(di);
    }
    let out = gb.add_persistent_tensor("mt_drafts", i32t, &[1, b - 1]);
    gb.op_concat("mt_drafts_cat", &drafts, out, 1);
    gb
}

/// One-off recipe `g_ikd_fold`: the recurrent head's latent map folded through the main model's
/// unembedding, `mtw_fold = unembed . hidden_out.W` `{rank, vocab_pad}` and
/// `mtw_bf = unembed . hidden_out.b` (run once at load, on the card).
pub fn build_fold_graph(g: &MtpGeom, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let c = &g.c;
    let mut gb = ik_graph(format!("g_ikd_fold_V{}", g.vocab_pad), elf_hashes);
    let u = gb.get_or_add_persistent_tensor("W_head", SynDataType::Bf16, &[c.hid, g.vocab_pad]);
    let _ = ik_linear(
        &mut gb,
        u,
        "mtw_howT",
        c.hid,
        c.rank,
        g.vocab_pad,
        "mtw_fold",
    );
    let _ = ik_linear(&mut gb, u, "mtw_hob", c.hid, 1, g.vocab_pad, "mtw_bf");
    gb
}

// ============================================================================
// Tensor roles and the memory plan
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Weight,
    State,
    /// Bound at launch to buffers of the main model.
    External,
    Act,
}

fn role(name: &str) -> Role {
    if name.starts_with("mtw_") {
        Role::Weight
    } else if name.starts_with("mts_") {
        Role::State
    } else if name.starts_with("mti_") || MAIN_SHARED.contains(&name) {
        Role::External
    } else {
        Role::Act
    }
}

/// Weights the card computes at load instead of the host.
fn computed(name: &str) -> bool {
    matches!(name, "mtw_fold" | "mtw_bf")
}

/// The drafter's share of the up-front device-memory plan, in bytes per tensor.
#[derive(Debug, Clone, Default)]
pub struct MtpPlan {
    pub weights: BTreeMap<String, u64>,
    pub state: BTreeMap<String, u64>,
    pub acts: BTreeMap<String, u64>,
}

impl MtpPlan {
    pub fn build(g: &MtpGeom) -> Self {
        let mut p = Self::default();
        let graphs = [
            build_ctx_graph(g, g.ctx_rows, &[]),
            build_block_graph(g, g.max_chunks, &[]),
            build_fold_graph(g, &[]),
        ];
        for gb in &graphs {
            for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
                let map = match role(&name) {
                    Role::Weight => &mut p.weights,
                    Role::State => &mut p.state,
                    Role::Act => &mut p.acts,
                    Role::External => continue,
                };
                let e = map.entry(name).or_insert(0);
                *e = (*e).max(bytes);
            }
        }
        p
    }

    fn sum(m: &BTreeMap<String, u64>) -> u64 {
        m.values().map(|&b| align256(b)).sum()
    }

    pub fn weights_bytes(&self) -> u64 {
        Self::sum(&self.weights)
    }

    pub fn state_bytes(&self) -> u64 {
        Self::sum(&self.state)
    }

    pub fn act_bytes(&self) -> u64 {
        Self::sum(&self.acts)
    }

    pub fn total(&self) -> u64 {
        self.weights_bytes() + self.state_bytes() + self.act_bytes()
    }
}

// ============================================================================
// Weights
// ============================================================================

/// Stored position `k` of a block of 128 channels holds channel `perm(k)`: the even channels,
/// then the odd ones (the family's split layout of f32 rows).
fn split_perm(k: usize) -> usize {
    if k < 64 {
        2 * k
    } else {
        2 * (k - 64) + 1
    }
}

fn bf16_to_f32(b: [u8; 2]) -> f32 {
    crate::fp8::bf16_to_f32(u16::from_le_bytes(b))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Rows of a row-major BF16 matrix `rows x cols` reordered so that, inside every block of 128
/// rows, stored row `k` is row `split_perm(k)` (`cols` values each).
fn split_rows(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let rb = cols * 2;
    let mut out = vec![0u8; rows * rb];
    for r in 0..rows {
        let s = (r & !127) + split_perm(r & 127);
        out[r * rb..(r + 1) * rb].copy_from_slice(&src[s * rb..(s + 1) * rb]);
    }
    out
}

/// The two-level rotary tables (`ik_mtp.h`): `{128, 2, n}` f32, column `n` the angle `step n f`.
pub fn rope_table(theta: f64, step: u64, n: u64) -> Vec<f32> {
    let mut out = vec![0f32; (DH * 2 * n) as usize];
    for col in 0..n as usize {
        for lane in 0..DH as usize {
            let (half, j) = (lane / 64, lane % 64);
            let f = 2 * (j % 32) + half;
            let inv = theta.powf(-(2.0 * f as f64) / DH as f64);
            let ang = (col as f64) * (step as f64) * inv;
            out[col * 256 + lane] = ang.cos() as f32;
            out[col * 256 + 128 + lane] = ang.sin() as f32;
        }
    }
    out
}

/// Packs the drafter's weights from `mtp.safetensors`.
pub(super) struct MtpPacker<'a> {
    st: &'a SafetensorsStore,
    g: &'a MtpGeom,
}

impl<'a> MtpPacker<'a> {
    pub fn new(st: &'a SafetensorsStore, g: &'a MtpGeom) -> Self {
        Self { st, g }
    }

    fn raw(&self, name: &str, shape: &[u64]) -> Result<&'a [u8], String> {
        let t = self
            .st
            .get(name)
            .map_err(|e| format!("drafter tensor '{name}': {e}"))?;
        if t.dtype != "BF16" || t.shape != shape {
            return Err(format!(
                "drafter tensor '{name}' is {} {:?}, expected BF16 {shape:?}",
                t.dtype, t.shape
            ));
        }
        self.st
            .tensor_bytes(name)
            .map_err(|e| format!("read '{name}': {e}"))?
            .ok_or_else(|| format!("drafter tensor '{name}' is not mapped"))
    }

    fn f32s(&self, name: &str, shape: &[u64]) -> Result<Vec<f32>, String> {
        Ok(self
            .raw(name, shape)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16_to_f32(*b))
            .collect())
    }

    /// `softmax(fuse_weight)` per row, in f64.
    pub fn alpha(&self) -> Result<Vec<f32>, String> {
        let w = self.f32s("fuse_weight", &[4, 4])?;
        let mut out = Vec::with_capacity(16);
        for r in 0..4 {
            let row: Vec<f64> = w[4 * r..4 * r + 4].iter().map(|&x| x as f64).collect();
            let m = row.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = row.iter().map(|x| (x - m).exp()).collect();
            let s: f64 = e.iter().sum();
            out.extend(e.iter().map(|x| (x / s) as f32));
        }
        Ok(out)
    }

    /// The device image of weight `name` (`None`: computed on the card at load).
    pub fn pack(&self, name: &str) -> Result<Option<Vec<u8>>, String> {
        if computed(name) {
            return Ok(None);
        }
        let c = &self.g.c;
        let (h, rk) = (c.hid, c.rank);
        let kvh = DH * c.nkv;
        let v = |x: &[u8]| Ok(Some(x.to_vec()));
        match name {
            "mtw_alpha" => return Ok(Some(f32_bytes(&self.alpha()?))),
            "mtw_wmix" => {
                let mut out = Vec::new();
                for l in 0..3 {
                    out.extend_from_slice(
                        self.raw(&format!("layers.{l}.dflare_norm.weight"), &[h])?,
                    );
                }
                out.extend_from_slice(self.raw("markov_head.dflare_norm.weight", &[h])?);
                return Ok(Some(out));
            }
            "mtw_rope_hi" => {
                return Ok(Some(f32_bytes(&rope_table(
                    c.theta,
                    ROPE_LO,
                    self.g.rope_hi,
                ))))
            }
            "mtw_rope_lo" => return Ok(Some(f32_bytes(&rope_table(c.theta, 1, ROPE_LO)))),
            "mtw_mask" => {
                return Ok(Some(
                    self.raw("mask_embed", &[h])?.repeat((c.block - 1) as usize),
                ))
            }
            "mtw_norm" => return v(self.raw("norm.weight", &[h])?),
            "mtw_zero" => return Ok(Some(vec![0u8; (rk * 2) as usize])),
            "mtw_w1" => return v(self.raw("markov_head.markov_w1.weight", &[self.g.vocab_in, rk])?),
            "mtw_hob" => return v(self.raw("markov_head.hidden_out.bias", &[h])?),
            "mtw_howT" => {
                let w = self.raw("markov_head.hidden_out.weight", &[h, rk])?;
                let mut out = vec![0u8; w.len()];
                for i in 0..h as usize {
                    for j in 0..rk as usize {
                        let (s, d) = ((i * rk as usize + j) * 2, (j * h as usize + i) * 2);
                        out[d..d + 2].copy_from_slice(&w[s..s + 2]);
                    }
                }
                return Ok(Some(out));
            }
            "mtw_Wzf" | "mtw_Wsw" => {
                let cols = 2 * rk + 2 * h;
                let w = self.raw("markov_head.rnn_proj.weight", &[3 * rk, cols])?;
                let rows = split_rows(w, (3 * rk) as usize, cols as usize);
                let (lo, hi) = if name == "mtw_Wsw" {
                    (0, 2 * rk)
                } else {
                    (2 * rk, cols)
                };
                let mut out = Vec::with_capacity(((hi - lo) * 3 * rk * 2) as usize);
                for r in 0..(3 * rk) as usize {
                    let base = r * cols as usize * 2;
                    out.extend_from_slice(&rows[base + lo as usize * 2..base + hi as usize * 2]);
                }
                return Ok(Some(out));
            }
            "mtw_rnnb" => {
                let b = self.f32s("markov_head.rnn_proj.bias", &[3 * rk])?;
                let p: Vec<f32> = (0..b.len())
                    .map(|r| b[(r & !127) + split_perm(r & 127)])
                    .collect();
                return Ok(Some(f32_bytes(&p)));
            }
            _ => {}
        }
        let Some((l, what)) = name
            .strip_prefix("mtw_L")
            .and_then(|r| r.split_once('_'))
            .and_then(|(l, w)| l.parse::<usize>().ok().map(|l| (l, w)))
            .filter(|&(l, _)| l < c.local.len())
        else {
            return Err(format!("Inkling 2 drafter: no packing rule for '{name}'"));
        };
        let p = format!("layers.{l}.");
        let a = format!("{p}self_attn.");
        // keys as stored, values with their channels in the split order of each head
        let kv = |k: &str, vn: &str| -> Result<Option<Vec<u8>>, String> {
            let mut out = self.raw(&format!("{a}{k}"), &[kvh, h])?.to_vec();
            out.extend(split_rows(
                self.raw(&format!("{a}{vn}"), &[kvh, h])?,
                kvh as usize,
                h as usize,
            ));
            Ok(Some(out))
        };
        match what {
            "Wkvc" => kv("k_proj_ctx.weight", "v_proj_ctx.weight"),
            "Wkv" => kv("k_proj.weight", "v_proj.weight"),
            "knc" => v(self.raw(&format!("{a}k_norm_ctx.weight"), &[DH])?),
            "qn" => v(self.raw(&format!("{a}q_norm.weight"), &[DH])?),
            "kn" => v(self.raw(&format!("{a}k_norm.weight"), &[DH])?),
            "Wq" => v(self.raw(&format!("{a}q_proj.weight"), &[DH * c.nh, h])?),
            "Wo" => v(self.raw(&format!("{a}o_proj.weight"), &[h, DH * c.nh])?),
            "win" => v(self.raw(&format!("{p}input_layernorm.weight"), &[h])?),
            "wpost" => v(self.raw(&format!("{p}post_attention_layernorm.weight"), &[h])?),
            "Wgu" => {
                let mut out = self
                    .raw(&format!("{p}mlp.gate_proj.weight"), &[c.ff, h])?
                    .to_vec();
                out.extend_from_slice(self.raw(&format!("{p}mlp.up_proj.weight"), &[c.ff, h])?);
                Ok(Some(out))
            }
            "Wd" => v(self.raw(&format!("{p}mlp.down_proj.weight"), &[h, c.ff])?),
            _ => Err(format!("Inkling 2 drafter: no packing rule for '{name}'")),
        }
    }
}

// ============================================================================
// Runtime
// ============================================================================

/// Acceptance and timing of the drafter over one request.
#[derive(Debug, Default, Clone)]
pub struct MtpStats {
    pub blocks: u64,
    /// `accepted_at[j]`: blocks whose draft `j` was accepted (with every draft before it).
    pub accepted_at: Vec<u64>,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub ctx_ms: f64,
}

/// The drafter's device state: addresses, recipes, the Markov context's row.
pub(super) struct MtpState {
    pub g: MtpGeom,
    pub taps: [MtpTap; 4],
    elf_hashes: Vec<(String, String)>,
    pub plan: MtpPlan,
    addr: BTreeMap<String, u64>,
    ctx_recipes: HashMap<u64, CompiledRecipe>,
    /// Prepared context launches: `(rows, decode)` (a decode launch reads the verify's taps, a
    /// prefill launch the tile's row buffers).
    ctx_prepared: HashMap<(u64, bool), PreparedLaunch>,
    blk: BTreeMap<u64, (CompiledRecipe, PreparedLaunch)>,
    /// Column of `mt_c3` that holds the Markov context of the next block (the last committed
    /// position before its anchor).
    pub fidx: u64,
    pub stats: MtpStats,
    pub dump_dir: Option<std::path::PathBuf>,
}

impl MtpState {
    pub fn new(
        g: MtpGeom,
        taps: [MtpTap; 4],
        elf_hashes: Vec<(String, String)>,
        plan: MtpPlan,
    ) -> Self {
        Self {
            g,
            taps,
            elf_hashes,
            plan,
            addr: BTreeMap::new(),
            ctx_recipes: HashMap::new(),
            ctx_prepared: HashMap::new(),
            blk: BTreeMap::new(),
            fidx: 0,
            stats: MtpStats::default(),
            dump_dir: std::env::var_os("GAUDIUM_IK_MTP_DUMP").map(std::path::PathBuf::from),
        }
    }

    /// Records the address of a planned tensor.
    pub fn set_addr(&mut self, name: &str, addr: u64) {
        self.addr.insert(name.to_string(), addr);
    }

    pub fn addr(&self, name: &str) -> u64 {
        self.addr[name]
    }

    /// `(address, bytes)` of every context state tensor.
    pub fn state_extents(&self) -> Vec<(u64, u64)> {
        self.plan
            .state
            .iter()
            .map(|(n, &b)| (self.addr[n], b))
            .collect()
    }
}

impl Inkling2Engine {
    pub(super) fn mtp_ref(&self) -> &MtpState {
        self.mtp.as_ref().expect("drafter configured")
    }

    pub(super) fn mtp_mut(&mut self) -> &mut MtpState {
        self.mtp.as_mut().expect("drafter configured")
    }

    /// The binding pool of a drafter recipe: the main model's shared tensors and every planned
    /// drafter tensor.
    fn mtp_pool(&self) -> BTreeMap<String, u64> {
        let mut pool = self.mtp_ref().addr.clone();
        for n in MAIN_SHARED {
            pool.insert((*n).to_string(), self.weights_addr[*n]);
        }
        pool
    }

    /// Uploads the drafter's weights, zeroes its context state and folds the head's latent map
    /// through the unembedding on the card.
    pub(super) fn mtp_init(
        &mut self,
        model_dir: &std::path::Path,
        stage: &mut PinnedBuf,
    ) -> Result<(), String> {
        let path = model_dir.join(MTP_FILE);
        let store = SafetensorsStore::open_file(&path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        let st = self.mtp_ref();
        let packer = MtpPacker::new(&store, &st.g);
        let mut uploads = Vec::new();
        for (name, &bytes) in &st.plan.weights {
            if let Some(data) = packer.pack(name)? {
                if data.len() as u64 != bytes {
                    return Err(format!(
                        "drafter weight '{name}' packed size {} != graph tensor size {bytes}",
                        data.len()
                    ));
                }
                uploads.push((st.addr[name], data));
            }
        }
        for (addr, data) in uploads {
            self.dev
                .upload_h2d_chunked(&data, addr, stage)
                .map_err(|e| format!("upload drafter weight: {e}"))?;
        }
        stage.as_mut_slice().fill(0);
        for (addr, bytes) in self.mtp_ref().state_extents() {
            self.dev
                .zero_hbm_chunked(addr, bytes, stage)
                .map_err(|e| format!("zero drafter state: {e}"))?;
        }
        let gb = build_fold_graph(&self.mtp_ref().g, &self.mtp_ref().elf_hashes);
        let r = self.compile(&gb, "drafter fold")?;
        let bind = select_exact_bindings(&r, &self.mtp_pool())?;
        r.launch_checked(&self.dev, &bind)
            .map_err(|e| format!("launch drafter fold: {e}"))?;
        self.sync_compute("drafter fold")?;
        Ok(())
    }

    fn mtp_ensure_ctx(&mut self, t: u64, decode: bool) -> Result<(), String> {
        if !self.mtp_ref().ctx_recipes.contains_key(&t) {
            let gb = build_ctx_graph(&self.mtp_ref().g, t, &self.mtp_ref().elf_hashes);
            let r = self.compile(&gb, &format!("drafter context T={t}"))?;
            self.mtp_mut().ctx_recipes.insert(t, r);
        }
        if self.mtp_ref().ctx_prepared.contains_key(&(t, decode)) {
            return Ok(());
        }
        let mut pool = self.mtp_pool();
        let taps = self.mtp_ref().taps;
        for (j, tap) in taps.iter().enumerate() {
            let addr = if decode {
                self.step_act_addr[&tap.step_name()].0
            } else {
                match *tap {
                    MtpTap::Source(s) => self.src_buf(s),
                    MtpTap::Final => self.pf_hfin,
                }
            };
            pool.insert(format!("mti_tap{j}"), addr);
        }
        pool.insert(
            "mti_ccur".into(),
            if decode {
                self.step_act_addr["cursor"].0
            } else {
                self.pf_cursor
            },
        );
        let r = &self.mtp_ref().ctx_recipes[&t];
        let exact = select_exact_bindings(r, &pool)?;
        let prepared = r
            .prepare_launch(&exact)
            .map_err(|e| format!("prepare drafter context T={t}: {e}"))?;
        self.mtp_mut().ctx_prepared.insert((t, decode), prepared);
        Ok(())
    }

    /// Queues the drafter's context of `t` rows whose taps the last pass left behind (`decode`:
    /// a verify pass; otherwise a prefill tile, whose final norm output is in `pf_hfin`). The
    /// rows' positions are in the pass's cursor. Nothing is synced.
    pub(super) fn mtp_ctx(&mut self, t: u64, decode: bool) -> Result<f64, String> {
        let tc = Instant::now();
        self.mtp_ensure_ctx(t, decode)?;
        let compile_ms = tc.elapsed().as_secs_f64() * 1000.0;
        let st = self.mtp_ref();
        st.ctx_recipes[&t]
            .launch_prepared(&self.dev, &st.ctx_prepared[&(t, decode)])
            .map_err(|e| format!("launch drafter context T={t}: {e}"))?;
        Ok(compile_ms)
    }

    /// Sets the column of the last context pass that holds the Markov context of the next block:
    /// the row of the last committed position before the next anchor (`row` of a pass of `rows`).
    pub(super) fn mtp_set_frnn(&mut self, row: u64, rows: u64) {
        self.mtp_mut().fidx = mtp_frnn_row(row, rows);
    }

    fn mtp_ensure_block(&mut self, nch: u64) -> Result<(), String> {
        if self.mtp_ref().blk.contains_key(&nch) {
            return Ok(());
        }
        let gb = build_block_graph(&self.mtp_ref().g, nch, &self.mtp_ref().elf_hashes);
        let r = self.compile(&gb, &format!("drafter block ({nch} chunks)"))?;
        let exact = select_exact_bindings(&r, &self.mtp_pool())?;
        let prepared = r
            .prepare_launch(&exact)
            .map_err(|e| format!("prepare drafter block: {e}"))?;
        self.mtp_mut().blk.insert(nch, (r, prepared));
        Ok(())
    }

    /// The chunks of the global layer's cache a block at anchor position `p` reads.
    fn mtp_bucket(&self, p: u64) -> u64 {
        step_bucket(mtp_ctx_chunks(p, IK_CHUNK), self.max_chunks)
    }

    /// Compiles the block recipe a block at anchor position `p` needs (first use is not decode
    /// time). Returns the milliseconds spent.
    pub(super) fn mtp_prepare_block(&mut self, p: u64) -> Result<f64, String> {
        let t0 = Instant::now();
        let nch = self.mtp_bucket(p);
        self.mtp_ensure_block(nch)?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    }

    /// Drafts the block anchored on `anchor` at position `p` (the context must hold every
    /// position before `p`): returns the `block - 1` drafted tokens.
    pub(super) fn mtp_draft(&mut self, anchor: i32, p: u64) -> Result<Vec<i32>, String> {
        let nch = self.mtp_bucket(p);
        self.mtp_ensure_block(nch)?;
        let (b, window, fidx) = {
            let st = self.mtp_ref();
            (st.g.c.block as usize, st.g.c.window, st.fidx)
        };
        let seen_w = mtp_ctx_seen(p, window, true);
        let seen_g = mtp_ctx_seen(p, window, false);
        let (a_anchor, a_fidx, a_ctl, a_out) = {
            let st = self.mtp_ref();
            (
                st.addr("mt_anchor"),
                st.addr("mt_fidx"),
                st.addr("mt_ctl"),
                st.addr("mt_drafts"),
            )
        };
        let buf = self.stage_io.as_mut_slice();
        buf[..4].copy_from_slice(&anchor.to_le_bytes());
        for k in 0..b {
            buf[64 + 4 * k..68 + 4 * k].copy_from_slice(&(fidx as i32).to_le_bytes());
        }
        // ctl: [p, context positions the global layer sees, a windowed layer sees]
        let ctl = [p as i32, seen_g as i32, seen_w as i32, 0, 0, 0, 0, 0];
        for (k, v) in ctl.iter().enumerate() {
            buf[512 + 4 * k..516 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        self.dev
            .copy_h2d_pinned_batch_sync(
                &self.stage_io,
                &[(0, 4, a_anchor), (64, 4 * b, a_fidx), (512, 32, a_ctl)],
            )
            .map_err(|e| format!("write drafter inputs H2D: {e}"))?;
        {
            let st = self.mtp_ref();
            let (r, prepared) = &st.blk[&nch];
            r.launch_prepared(&self.dev, prepared)
                .map_err(|e| format!("launch drafter block (p={p}): {e}"))?;
        }
        self.sync_compute("drafter block")?;
        let n = b - 1;
        self.dev
            .copy_d2h_pinned_sync(a_out, &mut self.stage_io, n * 4)
            .map_err(|e| format!("read drafts D2H: {e}"))?;
        let drafts: Vec<i32> = self.stage_io.as_slice()[..n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        if let Some(dir) = self.mtp_mut().dump_dir.take() {
            self.mtp_dump(&dir)?;
        }
        Ok(drafts)
    }

    /// Debug (`GAUDIUM_IK_MTP_DUMP`): writes every drafter activation of the last block and the
    /// first rows of every context state to `dir`.
    fn mtp_dump(&mut self, dir: &std::path::Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let cap = self.stage_io.as_slice().len();
        let st = self.mtp_ref();
        let list: Vec<(String, u64, u64)> = st
            .plan
            .acts
            .iter()
            .chain(st.plan.state.iter())
            .map(|(n, &b)| (n.clone(), st.addr[n], b.min(64 << 20)))
            .collect();
        for (name, addr, bytes) in list {
            let mut out = Vec::with_capacity(bytes as usize);
            let mut off = 0u64;
            while off < bytes {
                let n = ((bytes - off) as usize).min(cap);
                self.dev
                    .copy_d2h_pinned_sync(addr + off, &mut self.stage_io, n)
                    .map_err(|e| format!("dump '{name}': {e}"))?;
                out.extend_from_slice(&self.stage_io.as_slice()[..n]);
                off += n as u64;
            }
            std::fs::write(dir.join(format!("{name}.bin")), &out)
                .map_err(|e| format!("write dump '{name}': {e}"))?;
        }
        Ok(())
    }

    /// Records one block's outcome.
    pub(super) fn mtp_record(&mut self, n_drafts: usize, accepted: usize) {
        let s = &mut self.mtp_mut().stats;
        if s.accepted_at.len() < n_drafts {
            s.accepted_at.resize(n_drafts, 0);
        }
        s.blocks += 1;
        for a in s.accepted_at.iter_mut().take(accepted) {
            *a += 1;
        }
    }

    /// Starts a request's statistics.
    pub(super) fn mtp_begin_request(&mut self) {
        self.mtp_mut().stats = MtpStats::default();
    }

    /// The request's acceptance and timing, one line.
    pub(super) fn mtp_summary(&self) -> String {
        let s = &self.mtp_ref().stats;
        let n = s.blocks.max(1) as f64;
        let per: Vec<String> = s
            .accepted_at
            .iter()
            .map(|&a| format!("{:.3}", a as f64 / n))
            .collect();
        let mean: f64 = s.accepted_at.iter().map(|&a| a as f64).sum::<f64>() / n;
        format!(
            "blocks={} accepted/block={mean:.3} accept-rate per draft position=[{}] ms/draft-launch={:.3} ms/verify-launch={:.3} ms/context-launch(queue)={:.3}",
            s.blocks,
            per.join(", "),
            s.draft_ms / n,
            s.verify_ms / n,
            s.ctx_ms / n
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MTP_JSON: &str = r#"{"mtp_config": {"hidden_size": 2304, "num_hidden_layers": 3,
        "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 9216, "q_bias": false, "o_bias": false, "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0, "local_layer_ids": [0, 2], "sliding_window_size": 1024,
        "block_size": 8, "markov_rank": 256, "markov_head_type": "rnn_hidden",
        "target_layer_ids": [5, 23, 41, 48]}}"#;

    fn mini_dims(rows: u64) -> IkDims {
        let local: Vec<String> = (0..24u32)
            .filter(|l| (l + 1) % 6 != 0)
            .map(|l| l.to_string())
            .collect();
        let json = format!(
            r#"{{"architectures":["Inkling2LoopForConditionalGeneration"],"model_type":"inkling2_loop","eos_token_id":200006,
            "text_config":{{"model_type":"inkling2_loop_text","hidden_size":2304,"num_hidden_layers":24,"vocab_size":201024,
            "num_attention_heads":24,"num_key_value_heads":8,"head_dim":128,"d_rel":16,"rel_extent":1024,"log_scaling_n_floor":8192,
            "log_scaling_alpha":0.1,"rms_norm_eps":1e-06,"qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,
            "local_layer_ids":[{}],"sconv_kernel_size":4,"unpadded_vocab_size":200058,"final_logit_softcapping":30.0,
            "sliding_window_size":1024,"n_layers_per_attn_res_block":6,"n_routed_experts":128,"num_experts_per_tok":8,
            "intermediate_size":1152,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,
            "loop_config":{{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64}}}}}}"#,
            local.join(",")
        );
        IkDims::from_spec(&ModelSpec::from_config_json(&json).unwrap(), rows).unwrap()
    }

    #[test]
    fn config_and_taps_of_the_mini() {
        let c = MtpConfig::from_config_json(MTP_JSON).unwrap();
        assert_eq!(
            (c.hid, c.nh, c.nkv, c.ff, c.block, c.rank),
            (2304, 24, 8, 9216, 8, 256)
        );
        assert_eq!(c.local, vec![true, false, true]);
        let d = mini_dims(256);
        let taps = c.check_target(&d).unwrap();
        // the inputs of applications 5, 23, 41 (the outputs of 4, 22, 40) and the final norm output
        assert_eq!(
            taps,
            [
                MtpTap::Source(5),
                MtpTap::Source(23),
                MtpTap::Source(41),
                MtpTap::Final
            ]
        );
        assert_eq!(taps[1].step_name(), "A22_out");
        assert_eq!(taps[3].step_name(), "h_final");
        let bad = MTP_JSON.replace("rnn_hidden", "linear");
        assert!(MtpConfig::from_config_json(&bad).is_err());
    }

    #[test]
    fn geometry_and_plan_of_the_mini() {
        let c = MtpConfig::from_config_json(MTP_JSON).unwrap();
        let d = mini_dims(256);
        let g = MtpGeom::new(c, &d, 262144, 256).unwrap();
        // the ring holds the window beside the widest context pass
        assert!(g.ring >= 1024 + 256 && g.ring.is_power_of_two());
        assert_eq!(g.drafts(7), 7);
        assert_eq!(g.drafts(15), 7);
        assert_eq!(g.drafts(3), 3);
        let p = MtpPlan::build(&g);
        // the global layer's cache: 8 heads x 128 channels x K and V x 2 bytes per position
        assert_eq!(
            p.state["mts_L1_k"] + p.state["mts_L1_v"],
            2 * 8 * 128 * 2 * 262144
        );
        assert_eq!(p.state["mts_L0_k"], 128 * g.ring * 8 * 2);
        assert!(p.weights.contains_key("mtw_fold") && p.weights.contains_key("mtw_L2_Wgu"));
        assert!(!p.acts.contains_key("W_head") && !p.weights.contains_key("W_head"));
        assert!(!p.acts.keys().any(|n| n.starts_with("mti_")));
        let w = p.weights_bytes();
        assert!(w > 700_000_000 && w < 800_000_000, "weights {w}");
    }

    #[test]
    fn every_planned_weight_packs_to_its_size() {
        // needs the checkpoint: GAUDIUM_INKLING2_DIR=<dir with config.json and mtp.safetensors>
        let Some(dir) = std::env::var_os("GAUDIUM_INKLING2_DIR").map(std::path::PathBuf::from)
        else {
            eprintln!("skipped: needs GAUDIUM_INKLING2_DIR");
            return;
        };
        let c =
            MtpConfig::from_config_json(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let d = mini_dims(256);
        let g = MtpGeom::new(c, &d, 8192, 256).unwrap();
        let st = SafetensorsStore::open_file(dir.join(MTP_FILE)).unwrap();
        let pk = MtpPacker::new(&st, &g);
        let p = MtpPlan::build(&g);
        for (name, &bytes) in &p.weights {
            match pk.pack(name).unwrap() {
                Some(v) => assert_eq!(v.len() as u64, bytes, "{name}"),
                None => assert!(computed(name), "{name}"),
            }
        }
        // the mixes' weights are a softmax per row
        let a = pk.alpha().unwrap();
        for r in 0..4 {
            assert!((a[4 * r..4 * r + 4].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        // a value row of the context projection is stored in the split order of its head
        let wkvc = pk.pack("mtw_L0_Wkvc").unwrap().unwrap();
        let v = st
            .tensor_bytes("layers.0.self_attn.v_proj_ctx.weight")
            .unwrap()
            .unwrap();
        let rb = 2304 * 2;
        let base = 1024 * rb;
        assert_eq!(&wkvc[base + rb..base + 2 * rb], &v[2 * rb..3 * rb]);
        assert_eq!(&wkvc[base + 64 * rb..base + 65 * rb], &v[rb..2 * rb]);
    }

    #[test]
    fn layouts() {
        assert_eq!(
            (
                split_perm(0),
                split_perm(1),
                split_perm(63),
                split_perm(64),
                split_perm(127)
            ),
            (0, 2, 126, 1, 127)
        );
        let t = rope_table(1.0e6, 1, 3);
        // column 0 is the zero angle; lane 1 of v1 is frequency 2, lane 32 frequency 0 again
        assert_eq!((t[0], t[128]), (1.0, 0.0));
        let f2 = 1.0e6f64.powf(-4.0 / 128.0);
        assert!((t[256 + 1] as f64 - f2.cos()).abs() < 1e-7);
        assert!((t[256 + 128 + 1] as f64 - f2.sin()).abs() < 1e-7);
        assert_eq!(t[256 + 32], 1.0f64.cos() as f32);
        // the two levels compose to the angle of the position
        let (hi, lo) = (rope_table(1.0e6, 256, 4), rope_table(1.0e6, 1, 256));
        let (pos, lane) = (3 * 256 + 17, 64 + 5);
        let (h, l) = (pos / 256, pos % 256);
        let c = hi[h * 256 + lane] * lo[l * 256 + lane]
            - hi[h * 256 + 128 + lane] * lo[l * 256 + 128 + lane];
        let f = 1.0e6f64.powf(-(2.0 * 11.0) / 128.0);
        assert!((c as f64 - (pos as f64 * f).cos()).abs() < 1e-6);
    }
}
