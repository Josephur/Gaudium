//! Speculative decoding for the Gemma 4 engine with the size's own drafter checkpoint.
//!
//! This is a child module of `engine::gemma4` (declared there with `#[path]`), so it reuses that
//! module's private graph builders and engine fields without widening their visibility.
//!
//! **The drafter** (a size's `-assistant` checkpoint) is a four-layer decoder of the family's
//! layer body with no key or value projections of its own: every layer attends, with its own
//! queries, the *target's* K / V state as it is on the card: a sliding layer the ring of the
//! target's last sliding layer that has keys, a global layer the chunked cache of its last global
//! layer, positions below the cursor only. One pass drafts one token:
//!
//! ```text
//! x  = W_pre [ target_embed(token) ; h ]        h: the target's final-norm row at the last
//! 4 layers, final norm -> y                        validated position, then the previous pass's
//! draft = argmax(y . table^T);  h' = W_post y      h'; every pass at the same position
//! ```
//!
//! **One launch of a step recipe** `g_g4v_T{T}_D{Kd}_C{nch}` does
//! - *verify*: `T` rows (row 0 = the last accepted token, rows `1..T` = drafts) through the whole
//!   decoder. The rows write the rings and caches exactly as a one-row step does; the ring
//!   softmax masks by ring distance, so a row never sees the pass's later rows nor the stale rows
//!   a rejected draft left beyond the cursor, and nothing is undone after a rejection;
//! - *accept* (`gaudium_accept_t`, chain): the longest draft prefix equal to the verify arg-max
//!   chain -> `a_out`, `cursor_out`, `emitted`;
//! - *draft*: `Kd` drafter passes from the accepted row (token = its arg-max, hidden = its
//!   final-norm row) -> `next_ids = [that token, Kd drafts]`.
//!
//! All step state stays on the card and alternates between two buffers by step parity (`ids` /
//! `next_ids`, `cursor` / `cursor_out`, `emitted`), so the host only launches, waits and reads
//! the `emitted` row, and the next step is enqueued while the current one runs. The width of a
//! step's verify is the number of drafts the step before it made, so an adaptive width set is a
//! set of `(T, Kd)` recipes: the policy picks `Kd`, which becomes the verify width one step later.
//!
//! Exactness: tokens are always the verify arg-max chain, so the stream equals the engine's own
//! non-speculative `K = 0` stream whatever the drafts are. A sampled request verifies the same
//! way with the acceptance on the host ([`decode_sampled`]): the verify rows are sampled in order
//! with the position-seeded sampler of the non-speculative loop, so its stream is that loop's
//! stream of the same seed.

use super::*;
use crate::engine::qwen3::spec::{SpecConfig, SpecMethod, SpecStats, SPEC_MAX_ROWS};
use crate::synapse::ops::GatherParams;
use crate::tree::{choose_shape, make_shape, TreeCost, TreeShape, TreeStats};
use std::path::{Path, PathBuf};

#[path = "gemma4_dflash.rs"]
pub mod dflash;

/// Kernel ELFs a speculative step needs beyond the decoder's own.
const SPEC_ELFS: &[&str] = &["GAUDIUM_ACCEPTT_ELF", "GAUDIUM_SEL_ELF"];

/// `ns_TopkNode::Params` (`topk`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct TopkParams {
    bsw: u32,
    axis: u32,
    bottom_k: u32,
}

/// A drafter or shared table tensor (uploaded like a weight, owned by this module).
pub(super) fn is_spec_weight(name: &str) -> bool {
    name.starts_with("gd") || name == "sp_parent" || name == "sp_depth" || dflash::is_weight(name)
}

/// K / V state a drafter owns on the card (a block drafter's context rings and caches).
pub(super) fn is_spec_state(name: &str) -> bool {
    dflash::is_state(name)
}

/// Step / first-block tensors bound to parity buffers or to another recipe's buffer at launch.
pub(super) fn is_bound_tensor(name: &str) -> bool {
    matches!(
        name,
        "ids"
            | "next_ids"
            | "cursor"
            | "cursor_out"
            | "emitted"
            | "b0_h"
            | "b0_cur"
            | "b0_ids"
            | "ctx_cursor"
    )
}

// ============================================================================
// The drafter checkpoint
// ============================================================================

/// Configuration of a Gemma 4 drafter checkpoint (`config.json` of the `-assistant` directory).
#[derive(Debug, Clone, PartialEq)]
pub struct G4Assist {
    pub dir: PathBuf,
    pub hid: u64,
    pub inter: u64,
    /// The target's hidden size the pre / post projections map from / to.
    pub backbone: u64,
    pub vocab: u64,
    pub nh: u64,
    /// Per layer: a global layer (reads the target's last global cache).
    pub glob: Vec<bool>,
    window: u64,
    s_nkv: u64,
    s_dh: u64,
    g_nkv: Option<u64>,
    g_dh: u64,
    /// Ordered-embedding readout: `(clusters, clusters a draft looks into, tokens per cluster)`.
    pub ordered: Option<(u64, u64, u64)>,
}

impl G4Assist {
    /// Parses a drafter `config.json` text. Anything this engine does not build is refused by name.
    pub fn from_config_json(dir: &Path, text: &str) -> Result<Self, String> {
        let bad = |what: &str| format!("Gemma 4 drafter {}: {what}", dir.display());
        let root = parse_json(text).map_err(|e| bad(&format!("config.json: {e:?}")))?;
        let mt = root
            .get("model_type")
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if mt != "gemma4_assistant" && mt != "gemma4_unified_assistant" {
            return Err(bad(&format!(
                "model_type \"{mt}\" is not gemma4_assistant / gemma4_unified_assistant"
            )));
        }
        let tc = root
            .get("text_config")
            .ok_or_else(|| bad("no text_config"))?;
        let num = |o: &JsonValue, k: &str| o.get(k).and_then(JsonValue::as_u64).unwrap_or(0);
        let nl = num(tc, "num_hidden_layers");
        let a = Self {
            dir: dir.to_path_buf(),
            hid: num(tc, "hidden_size"),
            inter: num(tc, "intermediate_size"),
            backbone: num(&root, "backbone_hidden_size"),
            vocab: num(tc, "vocab_size"),
            nh: num(tc, "num_attention_heads"),
            glob: {
                let lt = tc
                    .get("layer_types")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| bad("no layer_types"))?;
                let mut g = Vec::with_capacity(lt.len());
                for x in lt {
                    match x.as_str() {
                        Some("full_attention") => g.push(true),
                        Some("sliding_attention") => g.push(false),
                        other => return Err(bad(&format!("layer type {other:?}"))),
                    }
                }
                g
            },
            window: num(tc, "sliding_window"),
            s_nkv: num(tc, "num_key_value_heads"),
            s_dh: num(tc, "head_dim"),
            g_nkv: tc
                .get("num_global_key_value_heads")
                .and_then(JsonValue::as_u64),
            g_dh: num(tc, "global_head_dim"),
            ordered: if root
                .get("use_ordered_embeddings")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
            {
                let nc = num(&root, "num_centroids");
                let tk = num(&root, "centroid_intermediate_top_k");
                let vocab = num(tc, "vocab_size");
                if nc == 0 || tk == 0 || tk > nc || !vocab.is_multiple_of(nc) {
                    return Err(bad(&format!(
                        "use_ordered_embeddings with num_centroids {nc}, centroid_intermediate_top_k {tk}"
                    )));
                }
                Some((nc, tk, vocab / nc))
            } else {
                None
            },
        };
        if a.hid == 0 || !a.hid.is_multiple_of(128) || a.inter == 0 || a.nh == 0 || nl == 0 {
            return Err(bad(
                "hidden_size / intermediate_size / num_attention_heads / num_hidden_layers",
            ));
        }
        if a.glob.len() as u64 != nl {
            return Err(bad("layer_types does not have num_hidden_layers entries"));
        }
        if num(tc, "num_kv_shared_layers") != nl {
            return Err(bad(
                "num_kv_shared_layers is not num_hidden_layers: a drafter layer with keys of its own",
            ));
        }
        if num(tc, "hidden_size_per_layer_input") != 0 {
            return Err(bad("per-layer input embeddings on a drafter"));
        }
        if tc
            .get("final_logit_softcapping")
            .is_some_and(|v| v.as_f64().is_some())
        {
            return Err(bad("final_logit_softcapping on a drafter"));
        }
        let act = tc
            .get("hidden_activation")
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if act != "gelu_pytorch_tanh" {
            return Err(bad(&format!("hidden_activation \"{act}\"")));
        }
        if let Some((nc, tk, vpc)) = a.ordered {
            // the candidate arg-max kernel takes whole 256-lane blocks; the centroid GEMM any width
            if !(tk * vpc).is_multiple_of(256) || nc < tk {
                return Err(bad(&format!(
                    "ordered readout over {tk} clusters of {vpc} tokens"
                )));
            }
        }
        Ok(a)
    }

    pub fn open(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("read drafter config {}: {e}", path.display()))?;
        Self::from_config_json(dir, &text)
    }

    /// The drafter's queries meet the target's keys: every head shape must be the target's.
    pub fn check_target(&self, d: &G4Dims, max_rows: u64) -> Result<(), String> {
        let bad = |what: String| format!("Gemma 4 drafter {}: {what}", self.dir.display());
        if self.backbone != d.hid {
            return Err(bad(format!(
                "backbone_hidden_size {} is not the target's hidden size {}",
                self.backbone, d.hid
            )));
        }
        if self.vocab != d.vocab {
            return Err(bad(format!(
                "vocab_size {} is not the target's {}",
                self.vocab, d.vocab
            )));
        }
        if self.window != d.window {
            return Err(bad(format!(
                "sliding_window {} differs from the target's {}",
                self.window, d.window
            )));
        }
        if self.s_nkv != d.s_nkv || self.s_dh != d.s_dh || !self.nh.is_multiple_of(d.s_nkv) {
            return Err(bad(format!(
                "sliding head shape {}q / {}kv x {} differs from the target's kv {} x {}",
                self.nh, self.s_nkv, self.s_dh, d.s_nkv, d.s_dh
            )));
        }
        if self.g_dh != d.g_dh
            || !self.nh.is_multiple_of(d.g_nkv)
            || self.g_nkv.is_some_and(|n| n != d.g_nkv)
        {
            return Err(bad(format!(
                "global head shape {}q / {:?}kv x {} differs from the target's kv {} x {}",
                self.nh, self.g_nkv, self.g_dh, d.g_nkv, d.g_dh
            )));
        }
        if self.glob.iter().any(|&g| !g) && d.src_s.is_none() {
            return Err(bad(
                "a sliding drafter layer, but the target has no sliding layer".into(),
            ));
        }
        // a drafter pass sees the `window + 1` most recent validated keys of the ring, which also
        // holds up to `max_rows - 1` rows beyond the cursor
        if d.ring < d.window + max_rows + 1 {
            return Err(bad(format!(
                "a ring of {} slots is too small for the window {} beside {max_rows} verify rows",
                d.ring, d.window
            )));
        }
        Ok(())
    }

    fn dh(&self, l: usize) -> u64 {
        if self.glob[l] {
            self.g_dh
        } else {
            self.s_dh
        }
    }

    /// Bytes of the checkpoint's device tensors (reported at load).
    pub fn weight_bytes(&self) -> u64 {
        let h = self.hid;
        let mut b = self.vocab * h * 2 + 3 * self.backbone * h * 2 + h * 2;
        for l in 0..self.glob.len() {
            let dh = self.dh(l);
            b += h * 2 * 4 + 2 + 3 * h * self.inter * 2 + 2 * self.nh * dh * h * 2 + dh * 4;
        }
        if let Some((nc, _, _)) = self.ordered {
            b += nc * h * 2 + self.vocab * 4;
        }
        b
    }
}

/// The drafter of a speculative Gemma 4 load.
#[derive(Debug, Clone)]
pub enum G4Drafter {
    /// The size's own drafter checkpoint (`-assistant`): one pass per drafted token.
    Assist(G4Assist),
    /// A published block drafter (`DFlashDraftModel`): one block pass per step.
    DFlash(dflash::G4DFlash),
}

/// A validated speculative configuration of a Gemma 4 load.
#[derive(Debug, Clone)]
pub struct G4SpecCfg {
    pub drafter: G4Drafter,
    /// Draft counts per launch, widest first (one value: fixed width).
    pub ks: Vec<u64>,
    /// Kernel fingerprints of the speculative graphs (the decoder's plus this module's).
    pub elf_hashes: Vec<(String, String)>,
    /// Rows of the engine's prefill tile (a block drafter's context pass covers a tile).
    pub tile: u64,
}

impl G4SpecCfg {
    /// Reads the drafter named by a `--spec` configuration. The drafter kind is read off the
    /// checkpoint's own `config.json`, whichever method name carried its directory.
    pub fn open(sc: &SpecConfig) -> Result<Self, String> {
        let dir = match &sc.method {
            SpecMethod::Assistant { draft_model }
            | SpecMethod::DSpine { draft_model }
            | SpecMethod::DFlash2 { draft_model } => draft_model,
            _ => {
                return Err(format!(
                    "--spec {}: a Gemma 4 model drafts with a drafter checkpoint (--spec assistant, or --spec dflash2 --draft-model <dir> for a block drafter)",
                    sc.method_name()
                ))
            }
        };
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("read drafter config {}: {e}", path.display()))?;
        let is_block = parse_json(&text).ok().is_some_and(|j| {
            j.get("architectures")
                .and_then(JsonValue::as_array)
                .is_some_and(|a| a.iter().any(|x| x.as_str() == Some("DFlashDraftModel")))
                || j.get("dflash_config").is_some()
                || j.get("speculators_config").is_some()
        });
        let drafter = if is_block {
            G4Drafter::DFlash(dflash::G4DFlash::from_config_json(dir, &text)?)
        } else {
            G4Drafter::Assist(G4Assist::from_config_json(dir, &text)?)
        };
        Self::new(drafter, &sc.draft_k)
    }

    pub fn new(drafter: G4Drafter, draft_k: &[u64]) -> Result<Self, String> {
        if draft_k.is_empty() {
            return Err("--draft-k needs at least one width".into());
        }
        let mut ks: Vec<u64> = Vec::new();
        for &k in draft_k {
            if k == 0 || k + 1 > SPEC_MAX_ROWS {
                return Err(format!(
                    "--draft-k {k}: a speculative step verifies 1..={} drafts",
                    SPEC_MAX_ROWS - 1
                ));
            }
            if !ks.contains(&k) {
                ks.push(k);
            }
        }
        ks.sort_unstable_by(|a, b| b.cmp(a));
        if matches!(drafter, G4Drafter::DFlash(_)) && ks.len() > 1 {
            return Err(
                "--draft-k: one width with a block drafter (its block is paid for whatever the verify width)"
                    .into(),
            );
        }
        Ok(Self {
            drafter,
            ks,
            elf_hashes: Vec::new(),
            tile: G4_MAX_PREFILL_TILE,
        })
    }

    pub fn kmax(&self) -> u64 {
        self.ks[0]
    }

    /// Rows of the widest verify pass.
    pub fn max_rows(&self) -> u64 {
        self.kmax() + 1
    }

    /// Method name of the reports.
    pub fn method_name(&self) -> &'static str {
        match self.drafter {
            G4Drafter::Assist(_) => "assistant",
            G4Drafter::DFlash(_) => "dflash",
        }
    }

    pub fn draft_dir(&self) -> &Path {
        match &self.drafter {
            G4Drafter::Assist(a) => &a.dir,
            G4Drafter::DFlash(c) => &c.dir,
        }
    }

    /// Kernel ELFs the speculative graphs need beyond the decoder's.
    pub(super) fn extra_elfs(&self) -> Vec<&'static str> {
        let mut v = SPEC_ELFS.to_vec();
        if matches!(self.drafter, G4Drafter::DFlash(_)) {
            v.extend_from_slice(dflash::DFLASH_ELFS);
        }
        v
    }

    /// The drafter's head shapes, taps and width against the target's.
    pub(super) fn check_target(&self, d: &G4Dims) -> Result<(), String> {
        match &self.drafter {
            G4Drafter::Assist(a) => a.check_target(d, self.max_rows()),
            G4Drafter::DFlash(c) => c.check_target(d, self.max_rows()),
        }
    }

    /// Slots of a block drafter's context rings (0: the drafter has none).
    fn df_ring(&self) -> u64 {
        match &self.drafter {
            G4Drafter::DFlash(c) => c.ring(self.tile.max(self.max_rows())),
            G4Drafter::Assist(_) => 0,
        }
    }

    /// One line on the drafter for the load log.
    pub(super) fn describe(&self, d: &G4Dims) -> String {
        match &self.drafter {
            G4Drafter::Assist(a) => format!(
                "drafter {}: {} layers x {} (MLP {}, {} query heads), {} readout, {:.2} GiB; drafts per launch {:?}; reads the target's L{} ring and L{} cache",
                a.dir.display(),
                a.glob.len(),
                a.hid,
                a.inter,
                a.nh,
                if a.ordered.is_some() {
                    "ordered-embedding"
                } else {
                    "full-table"
                },
                gib(a.weight_bytes()),
                self.ks,
                d.src_s.map_or(-1, |l| l as i64),
                d.src_g.map_or(-1, |l| l as i64),
            ),
            G4Drafter::DFlash(c) => format!(
                "block drafter {}: {} layers x {} (MLP {}), block {} (drafting {}), window {} in rings of {} slots, {} full-attention layer(s), taps {:?}, mask id {}, {:.2} GiB",
                c.dir.display(),
                c.full.len(),
                c.hid,
                c.inter,
                c.block,
                self.kmax(),
                c.window,
                self.df_ring(),
                c.full.iter().filter(|&&f| f).count(),
                c.taps,
                c.mask_id,
                gib(c.weight_bytes()),
            ),
        }
    }

    /// The graphs whose persistent tensors bound every speculative recipe of this configuration
    /// (every dimension grows with the rows, the drafts and the chunk count).
    pub(super) fn sizing_graphs(
        &self,
        d: &G4Dims,
        lmax: u64,
        max_chunks: u64,
    ) -> Vec<GraphBuilder> {
        let k = self.kmax();
        let mut v = vec![
            build_spec_step_graph(d, self, lmax, max_chunks, k + 1, k),
            build_spec_step_graph(d, self, lmax, max_chunks, 1, 0),
            build_block0_graph(d, self, lmax, max_chunks, k),
        ];
        if let G4Drafter::DFlash(c) = &self.drafter {
            v.push(dflash::build_ctx_graph(
                d,
                c,
                self.tile,
                lmax,
                self.df_ring(),
                &self.elf_hashes,
            ));
        }
        v
    }

    /// The chain tables of the accept kernel (`parent[t] = t - 1`, `depth[t] = t`); a block
    /// drafter's context pass reads `depth` for every row of a prefill tile.
    pub(super) fn table_bytes(&self) -> Vec<(String, u64)> {
        let n = SPEC_MAX_ROWS.max(self.tile);
        vec![("sp_parent".into(), n * 4), ("sp_depth".into(), n * 4)]
    }

    /// Step state that alternates by parity, and the first block's host-written token.
    pub(super) fn state_bytes(&self) -> Vec<(String, u64)> {
        let mut v = Vec::new();
        for p in 0..2 {
            v.push((format!("sp_ids{p}"), SPEC_MAX_ROWS * 4));
            v.push((format!("sp_cur{p}"), 4));
            v.push((format!("sp_emit{p}"), SPEC_MAX_ROWS * 4));
        }
        v.push(("b0_tok".into(), 4));
        v
    }
}

// ============================================================================
// Graph builders (host-testable without the card)
// ============================================================================

/// The target tensors a drafter pass reads, as declared in the graph under construction.
struct AssistKv {
    ring: Option<(TensorId, TensorId)>,
    cache: (TensorId, TensorId),
    rope: G4Rope,
}

/// Declares (or finds) the target's K / V state and RoPE tables the drafter attends.
fn assist_kv(gb: &mut GraphBuilder, d: &G4Dims, lmax: u64, nch: u64) -> AssistKv {
    let bf = SynDataType::Bf16;
    let f32t = SynDataType::Float;
    let lg = d
        .src_g
        .expect("a global layer with keys (checked in from_spec)");
    let cache = (
        gb.get_or_add_persistent_tensor(
            format!("L{lg}_kcache"),
            bf,
            &[d.g_dh, G4_CHUNK, d.g_nkv * nch],
        ),
        gb.get_or_add_persistent_tensor(
            format!("L{lg}_vcache"),
            bf,
            &[d.g_dh, G4_CHUNK, d.g_nkv * nch],
        ),
    );
    let ring = d.src_s.map(|ls| {
        (
            gb.get_or_add_persistent_tensor(format!("L{ls}_kring"), bf, &[d.s_dh, d.ring, d.s_nkv]),
            gb.get_or_add_persistent_tensor(format!("L{ls}_vring"), bf, &[d.s_dh, d.ring, d.s_nkv]),
        )
    });
    let global = (
        gb.get_or_add_persistent_tensor("rope_c", f32t, &[d.rope_width(true), lmax]),
        gb.get_or_add_persistent_tensor("rope_s", f32t, &[d.rope_width(true), lmax]),
    );
    let sliding = ring.is_some().then(|| {
        (
            gb.get_or_add_persistent_tensor("rope_sc", f32t, &[d.rope_width(false), lmax]),
            gb.get_or_add_persistent_tensor("rope_ss", f32t, &[d.rope_width(false), lmax]),
        )
    });
    AssistKv {
        ring,
        cache,
        rope: G4Rope { global, sliding },
    }
}

/// Appends one drafter pass: `(tok {1}, h {backbone, 1})` at position `cursor` (keys below it are
/// in view) -> `(draft {1}, h' {backbone, 1})`. `pre`: name prefix of the pass's activations.
#[allow(clippy::too_many_arguments)]
fn append_assist_pass(
    gb: &mut GraphBuilder,
    d: &G4Dims,
    a: &G4Assist,
    kv: &AssistKv,
    pre: &str,
    tok: TensorId,
    h: TensorId,
    cursor: TensorId,
    nch: u64,
) -> (TensorId, TensorId) {
    let bf = SynDataType::Bf16;
    let f32t = SynDataType::Float;
    let i32t = SynDataType::Int32;
    let (hd, bh) = (a.hid, a.backbone);
    let norm = |gb: &mut GraphBuilder, x: TensorId, w: &str, y: &str| {
        g4_norm(gb, x, w, &format!("{pre}{y}"), hd, 1, false)
    };

    // input: the target's scaled embedding of the token beside the hidden state
    let table = gb.get_or_add_persistent_tensor("embed_table", bf, &[bh, a.vocab]);
    let sc = gb.get_or_add_persistent_tensor("embed_scale", bf, &[1, 1]);
    let e0 = gb.add_persistent_tensor(format!("{pre}e0"), bf, &[bh, 1]);
    let e = gb.add_persistent_tensor(format!("{pre}e"), bf, &[bh, 1]);
    gb.op_gather_bf16(format!("{pre}embed"), table, tok, e0, 1);
    gb.op_mult_bf16(format!("{pre}embed_scale"), e0, sc, e);
    let xin = gb.add_persistent_tensor(format!("{pre}xin"), bf, &[2 * bh, 1]);
    gb.op_concat(format!("{pre}xin_cat"), &[e, h], xin, 0);
    let mut x = g4_linear(gb, xin, "gd_pre", 2 * bh, hd, 1, &format!("{pre}x0"));

    for (l, &glob) in a.glob.iter().enumerate() {
        let w = format!("gd{l}_");
        let p = format!("l{l}_");
        let (nkv, dh) = d.heads(glob);
        let gq = a.nh / nkv;
        let (c, nchl) = if glob { (G4_CHUNK, nch) } else { (d.ring, 1) };
        let b = nkv * nchl;
        let (kc, vc) = if glob {
            kv.cache
        } else {
            kv.ring.expect("checked in check_target")
        };
        let (rc, rs) = if glob {
            kv.rope.global
        } else {
            kv.rope.sliding.expect("sliding tables declared")
        };
        let hn = norm(gb, x, &format!("{w}w_in"), &format!("{p}h"));
        let q = g4_linear(
            gb,
            hn,
            &format!("{w}W_qg"),
            hd,
            a.nh * dh,
            1,
            &format!("{pre}{p}q"),
        );
        let qn_w = gb.get_or_add_persistent_tensor(format!("{w}qn_w"), f32t, &[dh]);
        let qb = gb.add_persistent_tensor(format!("{pre}{p}qb"), bf, &[dh, gq, b]);
        // node params: coff = 0, same = 1 (every row at the cursor's position)
        gb.add_node(
            "gaudium_ga_qprep",
            format!("{pre}{p}qprep"),
            &[q, qn_w, rc, rs, cursor],
            &[qb],
            &u32_params(&[0, 1]),
        );
        let scores = gb.add_persistent_tensor(format!("{pre}{p}scores"), f32t, &[c, gq, b]);
        gb.op_batch_gemm(format!("{pre}{p}qk"), qb, kc, scores, false, true);
        let pm = gb.add_persistent_tensor(format!("{pre}{p}P"), bf, &[c, gq, b]);
        let stats = gb.add_persistent_tensor(format!("{pre}{p}stats"), f32t, &[64, 2, gq, b]);
        if glob {
            // node params: T, nch, coff, mode 1 (positions below the cursor)
            gb.add_node(
                "gaudium_attn_softmax_chunk",
                format!("{pre}{p}softmax"),
                &[scores, cursor],
                &[pm, stats],
                &u32_params(&[1, nch as u32, 0, 1]),
            );
        } else {
            // node params: T, coff, W, mode 1 (the W + 1 most recent validated positions)
            gb.add_node(
                "gaudium_ga_softmax_ring",
                format!("{pre}{p}softmax"),
                &[scores, cursor],
                &[pm, stats],
                &u32_params(&[1, 0, d.window as u32 + 1, 1]),
            );
        }
        let aop = gb.add_persistent_tensor(format!("{pre}{p}ao_part"), f32t, &[dh, gq, b]);
        gb.op_batch_gemm(format!("{pre}{p}pv"), pm, vc, aop, false, false);
        let ao = gb.add_persistent_tensor(format!("{pre}{p}ao"), f32t, &[dh, gq, nkv]);
        gb.add_node(
            "gaudium_ga_combine",
            format!("{pre}{p}combine"),
            &[aop, stats],
            &[ao],
            &u32_params(&[nchl as u32]),
        );
        let core = gb.add_persistent_tensor(format!("{pre}{p}core"), bf, &[a.nh * dh, 1]);
        gb.add_node(
            "gaudium_ga_post",
            format!("{pre}{p}attn_post"),
            &[ao],
            &[core],
            &[],
        );
        let attn = g4_linear(
            gb,
            core,
            &format!("{w}W_o"),
            a.nh * dh,
            hd,
            1,
            &format!("{pre}{p}attn"),
        );
        let pa = norm(gb, attn, &format!("{w}w_pa"), &format!("{p}pa"));
        let x2 = gb.add_persistent_tensor(format!("{pre}{p}x2"), bf, &[hd, 1]);
        gb.op_add_bf16(format!("{pre}{p}x2_add"), x, pa, x2);
        let h2 = norm(gb, x2, &format!("{w}w_post"), &format!("{p}h2"));
        let gate = g4_linear(
            gb,
            h2,
            &format!("{w}W_gate"),
            hd,
            a.inter,
            1,
            &format!("{pre}{p}gate"),
        );
        let up = g4_linear(
            gb,
            h2,
            &format!("{w}W_up"),
            hd,
            a.inter,
            1,
            &format!("{pre}{p}up"),
        );
        let m = gb.add_persistent_tensor(format!("{pre}{p}m"), bf, &[a.inter, 1]);
        gb.add_node(
            "gaudium_ga_gelu",
            format!("{pre}{p}gelu"),
            &[gate, up],
            &[m],
            &[],
        );
        let mo = g4_linear(
            gb,
            m,
            &format!("{w}W_down"),
            a.inter,
            hd,
            1,
            &format!("{pre}{p}mlp"),
        );
        let pf = norm(gb, mo, &format!("{w}w_pf"), &format!("{p}pf"));
        let xs = gb.add_persistent_tensor(format!("{pre}{p}xs"), bf, &[hd, 1]);
        gb.op_add_bf16(format!("{pre}{p}xs_add"), x2, pf, xs);
        let ls = gb.get_or_add_persistent_tensor(format!("{w}ls"), bf, &[1, 1]);
        let out = gb.add_persistent_tensor(format!("{pre}{p}out"), bf, &[hd, 1]);
        gb.op_mult_bf16(format!("{pre}{p}lscale"), xs, ls, out);
        x = out;
    }

    let y = norm(gb, x, "gd_w_final", "y");
    let tok_out = gb.add_persistent_tensor(format!("{pre}tok"), i32t, &[1, 1]);
    match a.ordered {
        None => {
            // the drafter's own tied table over the whole vocabulary (no soft cap)
            let logits = g4_linear(gb, y, "gd_table", hd, a.vocab, 1, &format!("{pre}logits"));
            gb.add_node(
                "gaudium_ga_argmax",
                format!("{pre}argmax"),
                &[logits],
                &[tok_out],
                &[],
            );
        }
        Some((nc, tk, vpc)) => {
            // ordered embeddings: the `tk` clusters whose centroid scores highest, then the best
            // of their `vpc` tokens each
            let ncand = tk * vpc;
            let gp = GatherParams { axis: 1 };
            let cl = g4_linear(gb, y, "gd_centroids", hd, nc, 1, &format!("{pre}cl"));
            let tv = gb.add_persistent_tensor(format!("{pre}ctop_v"), bf, &[tk, 1]);
            let ti = gb.add_persistent_tensor(format!("{pre}ctop_i"), i32t, &[tk, 1]);
            let tp = TopkParams {
                bsw: tk as u32,
                axis: 0,
                bottom_k: 0,
            };
            gb.add_node(
                "topk",
                format!("{pre}ctop"),
                &[cl],
                &[tv, ti],
                as_param_bytes(&tp),
            );
            let ti1 = g4_view(gb, ti, &format!("{pre}ctop_i1"), i32t, &[tk]);
            let ord = gb.get_or_add_persistent_tensor("gd_order", i32t, &[vpc, nc]);
            let sel = gb.add_persistent_tensor(format!("{pre}csel"), i32t, &[vpc, tk]);
            gb.add_node(
                "gather_fwd_i32",
                format!("{pre}csel_gather"),
                &[ord, ti1],
                &[sel],
                as_param_bytes(&gp),
            );
            let sel_flat = g4_view(gb, sel, &format!("{pre}csel_flat"), i32t, &[ncand]);
            let tab = gb.get_or_add_persistent_tensor("gd_table", bf, &[hd, a.vocab]);
            let emb = gb.add_persistent_tensor(format!("{pre}cemb"), bf, &[hd, ncand]);
            gb.op_gather_bf16(format!("{pre}cemb_gather"), tab, sel_flat, emb, 1);
            let lg = gb.add_persistent_tensor(format!("{pre}clogits"), bf, &[ncand, 1]);
            gb.op_gemm(format!("{pre}clogits_gemm"), y, emb, lg, false, true);
            let am = gb.add_persistent_tensor(format!("{pre}cbest"), i32t, &[1, 1]);
            gb.add_node(
                "gaudium_ga_argmax",
                format!("{pre}cargmax"),
                &[lg],
                &[am],
                &[],
            );
            let sel_row = g4_view(gb, sel, &format!("{pre}csel_row"), i32t, &[1, ncand]);
            let am1 = g4_view(gb, am, &format!("{pre}cbest1"), i32t, &[1]);
            gb.add_node(
                "gather_fwd_i32",
                format!("{pre}ctok_gather"),
                &[sel_row, am1],
                &[tok_out],
                as_param_bytes(&gp),
            );
        }
    }
    let tok1 = g4_view(gb, tok_out, &format!("{pre}tok1"), i32t, &[1]);
    let h_out = g4_linear(gb, y, "gd_post", hd, bh, 1, &format!("{pre}hout"));
    (tok1, h_out)
}

/// Appends `kd` chained drafter passes from `(tok, h)` and returns `[tok, draft 1, .., draft kd]`
/// as the `{1}` id tensors of a concatenation.
#[allow(clippy::too_many_arguments)]
fn append_assist_chain(
    gb: &mut GraphBuilder,
    d: &G4Dims,
    a: &G4Assist,
    kv: &AssistKv,
    tok: TensorId,
    h: TensorId,
    cursor: TensorId,
    nch: u64,
    kd: u64,
) -> Vec<TensorId> {
    let mut parts = vec![tok];
    let (mut tok, mut h) = (tok, h);
    for j in 1..=kd {
        let (t1, h1) = append_assist_pass(gb, d, a, kv, &format!("gp{j}_"), tok, h, cursor, nch);
        parts.push(t1);
        tok = t1;
        h = h1;
    }
    parts
}

/// Speculative step `g_g4v_T{t}_D{kd}_C{nch}`: verify `t` rows, accept on the card, draft `kd`
/// tokens for the next step (module docs). `kd = 0`: `next_ids` is the accepted token alone.
pub fn build_spec_step_graph(
    d: &G4Dims,
    sc: &G4SpecCfg,
    lmax: u64,
    nch: u64,
    t: u64,
    kd: u64,
) -> GraphBuilder {
    let i32t = SynDataType::Int32;
    let tag = match sc.drafter {
        G4Drafter::Assist(_) => "v",
        G4Drafter::DFlash(_) => "df",
    };
    let mut gb = g4_graph(d, format!("g_g4{tag}_T{t}_D{kd}_C{nch}"), &sc.elf_hashes);
    let ids = gb.add_persistent_tensor("ids", i32t, &[t]);
    let cursor = gb.add_persistent_tensor("cursor", i32t, &[1]);
    let (h_final, _) = append_g4_trunk(&mut gb, d, ids, cursor, lmax, nch, t);
    let (_, tokens) = append_g4_head(&mut gb, d, h_final, t);

    // chain tables: parent[t] = t - 1, depth[t] = t (the shared iota buffers)
    let parent = gb.get_or_add_persistent_tensor("sp_parent", i32t, &[t]);
    let depth = gb.get_or_add_persistent_tensor("sp_depth", i32t, &[t]);
    let a_out = gb.add_persistent_tensor("a_out", i32t, &[1]);
    let cursor_out = gb.add_persistent_tensor("cursor_out", i32t, &[1]);
    let emitted = gb.add_persistent_tensor("emitted", i32t, &[t]);
    let fix_out = gb.add_persistent_tensor("fix_out", i32t, &[2]);
    gb.add_node(
        "gaudium_accept_t",
        "accept",
        &[ids, tokens, cursor, parent, depth],
        &[a_out, cursor_out, emitted, fix_out],
        &[],
    );

    let next_ids = gb.add_persistent_tensor("next_ids", i32t, &[kd + 1]);
    let a = match &sc.drafter {
        G4Drafter::Assist(a) => a,
        G4Drafter::DFlash(c) => {
            dflash::append_step(
                &mut gb,
                d,
                c,
                t,
                kd,
                cursor,
                depth,
                tokens,
                a_out,
                cursor_out,
                next_ids,
                lmax,
                sc.df_ring(),
            );
            return gb;
        }
    };
    if kd == 0 {
        gb.add_node(
            "gaudium_sel",
            "next_ids",
            &[tokens, a_out],
            &[next_ids],
            &[],
        );
        return gb;
    }
    // the drafter starts from the accepted row: its arg-max and its final-norm state
    let tok0 = gb.add_persistent_tensor("g4_tok0", i32t, &[1]);
    gb.add_node("gaudium_sel", "g4_tok0_sel", &[tokens, a_out], &[tok0], &[]);
    let h0 = gb.add_persistent_tensor("g4_h0", SynDataType::Bf16, &[d.hid, 1]);
    gb.op_gather_bf16("g4_h0_pick", h_final, a_out, h0, 1);
    let kv = assist_kv(&mut gb, d, lmax, nch);
    let parts = append_assist_chain(&mut gb, d, a, &kv, tok0, h0, cursor_out, nch, kd);
    gb.op_concat("next_ids", &parts, next_ids, 0);
    gb
}

/// First block after a prefill `g_g4v_blk0_D{kd}_C{nch}`: `kd` drafter passes from the last
/// prompt row (`b0_tok`: the prefill's token, `b0_h`: that row's final-norm state, `b0_cur`: the
/// token's position) -> `b0_ids = [b0_tok, kd drafts]`, the first step's ids.
pub fn build_block0_graph(
    d: &G4Dims,
    sc: &G4SpecCfg,
    lmax: u64,
    nch: u64,
    kd: u64,
) -> GraphBuilder {
    let i32t = SynDataType::Int32;
    let a = match &sc.drafter {
        G4Drafter::Assist(a) => a,
        G4Drafter::DFlash(c) => {
            // a block drafter reads its own context state: one recipe for every chunk count
            let mut gb = g4_graph(d, format!("g_g4df_blk0_D{kd}"), &sc.elf_hashes);
            dflash::append_block0(&mut gb, c, kd, lmax, sc.df_ring());
            return gb;
        }
    };
    let mut gb = g4_graph(d, format!("g_g4v_blk0_D{kd}_C{nch}"), &sc.elf_hashes);
    let tok = gb.add_persistent_tensor("b0_tok", i32t, &[1]);
    let h = gb.add_persistent_tensor("b0_h", SynDataType::Bf16, &[d.hid, 1]);
    let cur = gb.add_persistent_tensor("b0_cur", i32t, &[1]);
    let kv = assist_kv(&mut gb, d, lmax, nch);
    let parts = append_assist_chain(&mut gb, d, a, &kv, tok, h, cur, nch, kd);
    let ids = gb.add_persistent_tensor("b0_ids", i32t, &[kd + 1]);
    gb.op_concat("b0_ids", &parts, ids, 0);
    gb
}

// ============================================================================
// Drafter weights
// ============================================================================

/// Device byte image of drafter / table tensor `name` (the forms of the target's packer: `W_qg`
/// rows in lane order inside each 128-row block, `qn_w` as an f32 table in the kernels' view).
fn pack_assist_weight(
    store: &SafetensorsStore,
    a: &G4Assist,
    name: &str,
) -> Result<Vec<u8>, String> {
    use crate::core::gemma4::deinterleave_channel_table;
    use crate::engine::gemma4_packer::lane_order_row;
    use crate::fp8::bf16_to_f32;

    let bytes = |full: &str, dtype: &str, elems: u64, width: u64| -> Result<&[u8], String> {
        let t = store
            .get(full)
            .map_err(|e| format!("drafter tensor '{full}': {e}"))?;
        if t.dtype != dtype || t.nbytes as u64 != elems * width {
            return Err(format!(
                "drafter tensor '{full}' is {} with {} bytes, expected {dtype} with {elems} elements",
                t.dtype, t.nbytes
            ));
        }
        store
            .tensor_bytes(full)
            .map_err(|e| format!("read '{full}': {e}"))?
            .ok_or_else(|| format!("drafter tensor '{full}' is not mapped"))
    };
    let bf = |full: &str, elems: u64| bytes(full, "BF16", elems, 2).map(<[u8]>::to_vec);
    let (h, bh) = (a.hid, a.backbone);
    match name {
        "gd_table" => return bf("model.embed_tokens.weight", a.vocab * h),
        "gd_pre" => return bf("pre_projection.weight", h * 2 * bh),
        "gd_post" => return bf("post_projection.weight", bh * h),
        "gd_w_final" => return bf("model.norm.weight", h),
        "gd_centroids" => {
            let (nc, _, _) = a.ordered.ok_or("gd_centroids on a full-table drafter")?;
            return bf("masked_embedding.centroids.weight", nc * h);
        }
        "gd_order" => {
            // int32 {vpc, clusters}: column c holds the token ids of cluster c, in the
            // checkpoint's order
            let src = bytes("masked_embedding.token_ordering", "I64", a.vocab, 8)?;
            let mut out = Vec::with_capacity(a.vocab as usize * 4);
            for &c in src.as_chunks::<8>().0 {
                let id = i64::from_le_bytes(c);
                if id < 0 || id as u64 >= a.vocab {
                    return Err(format!(
                        "drafter token_ordering holds id {id} outside the vocabulary"
                    ));
                }
                out.extend_from_slice(&(id as i32).to_le_bytes());
            }
            return Ok(out);
        }
        _ => {}
    }
    let rest = name
        .strip_prefix("gd")
        .ok_or_else(|| format!("unknown drafter tensor '{name}'"))?;
    let sep = rest
        .find('_')
        .ok_or_else(|| format!("malformed drafter tensor '{name}'"))?;
    let l: usize = rest[..sep]
        .parse()
        .map_err(|_| format!("invalid layer index in '{name}'"))?;
    if l >= a.glob.len() {
        return Err(format!("layer index out of range in '{name}'"));
    }
    let lp = format!("model.layers.{l}.");
    let qd = a.nh * a.dh(l);
    match &rest[sep + 1..] {
        "w_in" => bf(&format!("{lp}input_layernorm.weight"), h),
        "w_pa" => bf(&format!("{lp}post_attention_layernorm.weight"), h),
        "w_post" => bf(&format!("{lp}pre_feedforward_layernorm.weight"), h),
        "w_pf" => bf(&format!("{lp}post_feedforward_layernorm.weight"), h),
        "ls" => bf(&format!("{lp}layer_scalar"), 1),
        "W_gate" => bf(&format!("{lp}mlp.gate_proj.weight"), a.inter * h),
        "W_up" => bf(&format!("{lp}mlp.up_proj.weight"), a.inter * h),
        "W_down" => bf(&format!("{lp}mlp.down_proj.weight"), h * a.inter),
        "W_o" => bf(&format!("{lp}self_attn.o_proj.weight"), h * qd),
        "W_qg" => {
            let src = bytes(&format!("{lp}self_attn.q_proj.weight"), "BF16", qd * h, 2)?;
            let rb = h as usize * 2;
            let mut dst = vec![0u8; src.len()];
            for r in 0..qd {
                let (s, t) = (r as usize * rb, lane_order_row(r) as usize * rb);
                dst[t..t + rb].copy_from_slice(&src[s..s + rb]);
            }
            Ok(dst)
        }
        "qn_w" => {
            let src = bytes(&format!("{lp}self_attn.q_norm.weight"), "BF16", a.dh(l), 2)?;
            let w: Vec<f32> = src
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| bf16_to_f32(u16::from_le_bytes(c)))
                .collect();
            Ok(deinterleave_channel_table(&w)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        other => Err(format!("unsupported drafter tensor 'gd{l}_{other}'")),
    }
}

// ============================================================================
// Width policy
// ============================================================================

/// Picks the number of drafts a step makes for the step after it, from the measured acceptance
/// per draft position and the measured launch time of each width (`crate::tree`: expected tokens
/// per millisecond; a width is launched [`G4Widths::MIN_TIMED`] times before its time is used).
#[derive(Debug, Clone)]
pub struct G4Widths {
    shapes: Vec<TreeShape>,
    keys: Vec<String>,
    stats: TreeStats,
    cost: TreeCost,
}

impl G4Widths {
    pub const MIN_TIMED: usize = 3;

    /// `ks`: the candidate draft counts, widest first.
    pub fn new(ks: &[u64]) -> Self {
        let kmax = ks.first().copied().unwrap_or(1) as usize;
        Self {
            keys: ks.iter().map(|k| format!("chain{k}")).collect(),
            shapes: ks
                .iter()
                .map(|&k| make_shape(k as usize, &[], &format!("chain{k}")))
                .collect(),
            stats: TreeStats::new(kmax, 1),
            cost: TreeCost::default(),
        }
    }

    fn timed(&self, i: usize) -> usize {
        self.cost.measured.get(&self.keys[i]).map_or(0, |m| m.1)
    }

    /// The draft count to make next.
    pub fn choose(&self) -> u64 {
        if self.shapes.len() == 1 {
            return self.shapes[0].spine() as u64;
        }
        if let Some(i) = (0..self.shapes.len()).find(|&i| self.timed(i) < Self::MIN_TIMED) {
            return self.shapes[i].spine() as u64;
        }
        let keys: Vec<&str> = self.keys.iter().map(String::as_str).collect();
        choose_shape(&self.stats, &self.shapes, &self.cost, None, &keys)
            .map_or(self.shapes[0].spine() as u64, |c| {
                self.shapes[c].spine() as u64
            })
    }

    /// A step verified `k` drafts and accepted the first `accepted`. `ms`: its launch time when
    /// the step also drafted `k` tokens (a width's own steady-state cost), else `None`.
    pub fn observe(&mut self, k: u64, accepted: usize, ms: Option<f64>) {
        if k == 0 {
            return;
        }
        let shape = match self.shapes.iter().position(|s| s.spine() as u64 == k) {
            Some(i) => {
                if let Some(ms) = ms {
                    self.cost.observe(&self.keys[i], ms);
                }
                self.shapes[i].clone()
            }
            None => make_shape(k as usize, &[], "chain"),
        };
        self.stats.observe(&shape, accepted, accepted);
    }
}

// ============================================================================
// Runtime
// ============================================================================

struct StepRecipe {
    recipe: CompiledRecipe,
    launches: [PreparedLaunch; 2],
}

/// Speculative decoding state of a loaded Gemma 4 engine: the drafter and the step recipes.
///
/// Built around the engine's allocation seal: [`G4SpecRt::prepare`] (drafter checkpoint, eager
/// recipes and their workspace need, before any allocation) and [`G4SpecRt::finish`] (launch
/// descriptors, after the seal). The device memory it needs is part of the engine's plan
/// (`G4Plan::build`).
pub struct G4SpecRt {
    cfg: G4SpecCfg,
    /// The drafter checkpoint, kept mapped until its tensors are uploaded.
    store: Option<SafetensorsStore>,
    /// `(rows, drafts, chunks)` -> step recipe with both parities' launches.
    steps: HashMap<(u64, u64, u64), StepRecipe>,
    pending: Vec<((u64, u64, u64), CompiledRecipe)>,
    /// `(drafts, chunks)` -> first-block recipe.
    blk0: HashMap<(u64, u64), CompiledRecipe>,
    /// Tile rows -> a block drafter's prefill context recipe.
    ctx: HashMap<u64, CompiledRecipe>,
    max_ws: u64,
    verbose: bool,
    pub last_stats: SpecStats,
}

impl G4SpecRt {
    pub fn cfg(&self) -> &G4SpecCfg {
        &self.cfg
    }

    pub fn max_ws(&self) -> u64 {
        self.max_ws
    }

    pub fn method_label(&self, fp8: bool, temperature: f32) -> String {
        let ks: Vec<String> = self.cfg.ks.iter().map(u64::to_string).collect();
        format!(
            "{} K={} {} {}",
            self.cfg.method_name(),
            ks.join(","),
            if fp8 { "FP8" } else { "BF16" },
            if temperature > 0.0 {
                format!("T={temperature:.2}")
            } else {
                "greedy".to_string()
            }
        )
    }

    /// Opens the drafter checkpoint and compiles, for the first chunk bucket, every step recipe
    /// a request of this width set launches (so none compiles inside a short request).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        cfg: G4SpecCfg,
        d: &G4Dims,
        max_ctx: u64,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
        verbose: bool,
    ) -> Result<Self, String> {
        let store = SafetensorsStore::open(cfg.draft_dir())
            .map_err(|e| format!("open drafter {}: {e}", cfg.draft_dir().display()))?;
        let mut rt = Self {
            cfg,
            store: Some(store),
            steps: HashMap::new(),
            pending: Vec::new(),
            blk0: HashMap::new(),
            ctx: HashMap::new(),
            max_ws: 0,
            verbose,
            last_stats: SpecStats::default(),
        };
        let ks = rt.cfg.ks.clone();
        let note = |r: &CompiledRecipe, t0: Instant| {
            if verbose {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                    r.name,
                    r.from_cache,
                    mib(r.ws_size),
                    t0.elapsed().as_secs_f64()
                );
            }
        };
        for &kv in &ks {
            for &kd in &ks {
                let t0 = Instant::now();
                let r = build_spec_step_graph(d, &rt.cfg, max_ctx, 1, kv + 1, kd)
                    .compile(Arc::clone(api), cache, driver_version)
                    .map_err(|e| format!("compile spec step T={} D={kd}: {e}", kv + 1))?;
                note(&r, t0);
                rt.max_ws = rt.max_ws.max(r.ws_size);
                rt.pending.push(((kv + 1, kd, 1), r));
            }
        }
        for &kd in &ks {
            let t0 = Instant::now();
            let r = build_block0_graph(d, &rt.cfg, max_ctx, 1, kd)
                .compile(Arc::clone(api), cache, driver_version)
                .map_err(|e| format!("compile first block D={kd}: {e}"))?;
            note(&r, t0);
            rt.max_ws = rt.max_ws.max(r.ws_size);
            rt.blk0.insert((kd, 1), r);
        }
        Ok(rt)
    }

    /// Device byte image of one of this module's weight tensors (`bytes`: the size planned for
    /// it, which sizes the tables).
    pub(super) fn pack_weight(&self, name: &str, bytes: u64) -> Result<Vec<u8>, String> {
        if name == "sp_depth" || name == "sp_parent" {
            let off = i32::from(name == "sp_parent");
            return Ok((0..(bytes / 4) as i32)
                .flat_map(|i| (i - off).to_le_bytes())
                .collect());
        }
        let store = self
            .store
            .as_ref()
            .ok_or("the drafter checkpoint is already closed")?;
        match &self.cfg.drafter {
            G4Drafter::Assist(a) => pack_assist_weight(store, a, name),
            G4Drafter::DFlash(c) => dflash::pack_weight(store, c, name, bytes),
        }
    }

    /// Prefill routing of a block drafter's taps: a tapped layer writes its output to the tap's
    /// buffer and the next layer reads it from there.
    pub(super) fn prefill_io(
        &self,
        acts: &BTreeMap<String, u64>,
        l: usize,
        x_in: u64,
        x_out: u64,
    ) -> (u64, u64) {
        let G4Drafter::DFlash(c) = &self.cfg.drafter else {
            return (x_in, x_out);
        };
        let tap = |layer: usize| {
            c.taps
                .iter()
                .position(|&t| t == layer)
                .and_then(|i| acts.get(&format!("df_tap{i}")).copied())
        };
        let x_in = if l > 0 {
            tap(l - 1).unwrap_or(x_in)
        } else {
            x_in
        };
        (x_in, tap(l).unwrap_or(x_out))
    }

    /// Unmaps the drafter checkpoint once its tensors are on the card.
    pub(super) fn close_store(&mut self) {
        self.store = None;
    }
}

/// Every buffer a speculative recipe can be bound to by its own tensor name.
fn base_pool(eng: &Gemma4Engine) -> BTreeMap<String, u64> {
    let mut pool: BTreeMap<String, u64> = eng.weights_addr.clone();
    for (k, &(v, _)) in &eng.kv_addr {
        pool.insert(k.clone(), v);
    }
    for (k, &v) in &eng.step_act_addr {
        pool.insert(k.clone(), v);
    }
    pool
}

/// Pre-validates a step recipe's launches for both parities (parity `p` reads the buffers `p`
/// and writes the buffers `1 - p`).
fn install_step(
    eng: &Gemma4Engine,
    rt: &mut G4SpecRt,
    key: (u64, u64, u64),
    mut recipe: CompiledRecipe,
) -> Result<(), String> {
    recipe
        .bind_shared_workspace(&eng.dev)
        .map_err(|e| format!("bind {} workspace: {e}", recipe.name))?;
    let base = base_pool(eng);
    let prep = |p: usize| -> Result<PreparedLaunch, String> {
        let q = 1 - p;
        let mut pool = base.clone();
        pool.insert("ids".into(), base[&format!("sp_ids{p}")]);
        pool.insert("next_ids".into(), base[&format!("sp_ids{q}")]);
        pool.insert("cursor".into(), base[&format!("sp_cur{p}")]);
        pool.insert("cursor_out".into(), base[&format!("sp_cur{q}")]);
        pool.insert("emitted".into(), base[&format!("sp_emit{p}")]);
        let exact = select_exact_bindings(&recipe, &pool)?;
        recipe
            .prepare_launch(&exact)
            .map_err(|e| format!("prepare {} (parity {p}): {e}", recipe.name))
    };
    let launches = [prep(0)?, prep(1)?];
    rt.steps.insert(key, StepRecipe { recipe, launches });
    Ok(())
}

/// After the seal and the uploads: binds workspaces and pre-validates the eager recipes.
pub(super) fn finish(eng: &mut Gemma4Engine) -> Result<(), String> {
    let Some(mut rt) = eng.specrt.take() else {
        return Ok(());
    };
    rt.close_store();
    let res = (|| -> Result<(), String> {
        for (key, r) in std::mem::take(&mut rt.pending) {
            install_step(eng, &mut rt, key, r)?;
        }
        for r in rt.blk0.values_mut() {
            r.bind_shared_workspace(&eng.dev)
                .map_err(|e| format!("bind {} workspace: {e}", r.name))?;
        }
        Ok(())
    })();
    eng.specrt = Some(rt);
    res
}

/// After the decoder layers of one prefill tile: a block drafter's context K / V of the tile's
/// `t` rows at positions `pos ..` (enqueued behind the layers; nothing is read back).
pub(super) fn prefill_tile_hook(eng: &mut Gemma4Engine, pos: u64, t: u64) -> Result<(), String> {
    if !matches!(
        eng.specrt.as_ref().map(|rt| &rt.cfg.drafter),
        Some(G4Drafter::DFlash(_))
    ) {
        return Ok(());
    }
    let mut rt = eng.specrt.take().expect("checked");
    let res = (|| -> Result<(), String> {
        let G4Drafter::DFlash(c) = &rt.cfg.drafter else {
            return Ok(());
        };
        if !rt.ctx.contains_key(&t) {
            let t0 = Instant::now();
            let mut r = dflash::build_ctx_graph(
                &eng.dims,
                c,
                t,
                eng.max_ctx,
                rt.cfg.df_ring(),
                &rt.cfg.elf_hashes,
            )
            .compile(Arc::clone(eng.dev.api()), &eng.cache, &eng.driver_version)
            .map_err(|e| format!("compile drafter context T={t}: {e}"))?;
            r.bind_shared_workspace(&eng.dev)
                .map_err(|e| format!("bind {} workspace: {e}", r.name))?;
            if rt.verbose {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                    r.name,
                    r.from_cache,
                    mib(r.ws_size),
                    t0.elapsed().as_secs_f64()
                );
            }
            rt.ctx.insert(t, r);
        }
        let r = &rt.ctx[&t];
        let mut pool = base_pool(eng);
        pool.insert("ctx_cursor".into(), eng.pf_cursor);
        let exact = select_exact_bindings(r, &pool)?;
        r.launch_checked(&eng.dev, &exact)
            .map_err(|e| format!("launch drafter context T={t} at {pos}: {e}"))
    })();
    eng.specrt = Some(rt);
    res
}

/// Compiles (or loads) the step recipe `(t, kd, nch)` on first use.
fn ensure_step(eng: &mut Gemma4Engine, t: u64, kd: u64, nch: u64) -> Result<(), String> {
    let mut rt = eng.specrt.take().expect("spec runtime");
    let res = (|| -> Result<(), String> {
        if rt.steps.contains_key(&(t, kd, nch)) {
            return Ok(());
        }
        let t0 = Instant::now();
        let recipe = build_spec_step_graph(&eng.dims, &rt.cfg, eng.max_ctx, nch, t, kd)
            .compile(Arc::clone(eng.dev.api()), &eng.cache, &eng.driver_version)
            .map_err(|e| format!("compile spec step T={t} D={kd} C={nch}: {e}"))?;
        if rt.verbose {
            eprintln!(
                "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                recipe.name,
                recipe.from_cache,
                mib(recipe.ws_size),
                t0.elapsed().as_secs_f64()
            );
        }
        install_step(eng, &mut rt, (t, kd, nch), recipe)
    })();
    eng.specrt = Some(rt);
    res
}

/// Compiles (or loads) the first-block recipe `(kd, nch)` on first use.
fn ensure_block0(eng: &mut Gemma4Engine, kd: u64, nch: u64) -> Result<(), String> {
    let mut rt = eng.specrt.take().expect("spec runtime");
    let res = (|| -> Result<(), String> {
        if !rt.blk0.contains_key(&(kd, nch)) {
            let mut r = build_block0_graph(&eng.dims, &rt.cfg, eng.max_ctx, nch, kd)
                .compile(Arc::clone(eng.dev.api()), &eng.cache, &eng.driver_version)
                .map_err(|e| format!("compile first block D={kd} C={nch}: {e}"))?;
            r.bind_shared_workspace(&eng.dev)
                .map_err(|e| format!("bind {} workspace: {e}", r.name))?;
            rt.blk0.insert((kd, nch), r);
        }
        Ok(())
    })();
    eng.specrt = Some(rt);
    res
}

/// Hands a step its inputs in the parity-0 buffers: row 0 = `anchor` at position `cursor`, and
/// `kd` drafts made from it by the first-block recipe (`h_addr`: the final-norm row that
/// predicted `anchor`). Enqueued, not waited for: the step launched next reads what it writes.
fn stage_block(
    eng: &mut Gemma4Engine,
    anchor: i32,
    cursor: u64,
    kd: u64,
    nch: u64,
    h_addr: u64,
) -> Result<(), String> {
    let addr = |eng: &Gemma4Engine, name: &str| -> Result<u64, String> {
        eng.step_act_addr
            .get(name)
            .copied()
            .ok_or_else(|| format!("speculative buffer '{name}' is not allocated"))
    };
    let (cur0, ids0) = (addr(eng, "sp_cur0")?, addr(eng, "sp_ids0")?);
    let (tok_val, cur_val) = ([anchor], [cursor as i32]);
    if kd == 0 {
        return write_i32s(eng, &[(ids0, &tok_val), (cur0, &cur_val)]);
    }
    // a block drafter's block: the token and the masks its rows start from
    let blk = match &eng.specrt.as_ref().expect("spec runtime").cfg.drafter {
        G4Drafter::DFlash(c) => {
            let mut ids = vec![c.mask_id; kd as usize + 1];
            ids[0] = anchor;
            Some((addr(eng, "b0_blk")?, ids))
        }
        G4Drafter::Assist(_) => None,
    };
    let mut writes: Vec<(u64, &[i32])> = vec![(addr(eng, "b0_tok")?, &tok_val), (cur0, &cur_val)];
    if let Some((a, ids)) = &blk {
        writes.push((*a, ids));
    }
    write_i32s(eng, &writes)?;
    launch_block0(eng, kd, nch, h_addr)
}

/// Launches the first block: `kd` drafts from the last prompt row into the parity-0 buffers.
/// The host has written `b0_tok` and `sp_cur0`; nothing is read back.
fn launch_block0(eng: &mut Gemma4Engine, kd: u64, nch: u64, h_addr: u64) -> Result<(), String> {
    ensure_block0(eng, kd, nch)?;
    let rt = eng.specrt.take().expect("spec runtime");
    let res = (|| -> Result<(), String> {
        let r = &rt.blk0[&(kd, nch)];
        let mut pool = base_pool(eng);
        pool.insert("b0_h".into(), h_addr);
        pool.insert("b0_cur".into(), pool["sp_cur0"]);
        pool.insert("b0_ids".into(), pool["sp_ids0"]);
        let exact = select_exact_bindings(r, &pool)?;
        r.launch_checked(&eng.dev, &exact)
            .map_err(|e| format!("launch first block D={kd} C={nch}: {e}"))
    })();
    eng.specrt = Some(rt);
    res
}

fn write_i32s(eng: &mut Gemma4Engine, writes: &[(u64, &[i32])]) -> Result<(), String> {
    let mut transfers = Vec::with_capacity(writes.len());
    let mut off = 0usize;
    let buf = eng.stage_io.as_mut_slice();
    for &(addr, vals) in writes {
        for (i, v) in vals.iter().enumerate() {
            buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        transfers.push((off, vals.len() * 4, addr));
        off += (vals.len() * 4).next_multiple_of(64);
    }
    eng.dev
        .copy_h2d_pinned_batch_sync(&eng.stage_io, &transfers)
        .map_err(|e| format!("spec H2D: {e}"))
}

fn read_i32s(eng: &mut Gemma4Engine, addr: u64, n: usize) -> Result<Vec<i32>, String> {
    eng.dev
        .copy_d2h_pinned_sync(addr, &mut eng.stage_io, n * 4)
        .map_err(|e| format!("spec D2H: {e}"))?;
    Ok(eng.stage_io.as_slice()[..n * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| i32::from_le_bytes(c))
        .collect())
}

/// Result of the speculative decode phase, folded into the engine's report by the caller.
pub(super) struct G4SpecOut {
    pub decode_ms: f64,
    pub decode_tokens: usize,
    pub stopped_by_eos: bool,
    pub stopped_by_cancel: bool,
    pub stats: SpecStats,
}

/// Greedy speculative decode: continues after the prefill produced `generated[last]` at position
/// `prompt_len`; `h_addr` is the final-norm row of the last prompt position.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode(
    eng: &mut Gemma4Engine,
    prompt_len: usize,
    h_addr: u64,
    params: &GenParams,
    budget: &mut SequenceBudget,
    is_eos: &impl Fn(i32) -> bool,
    is_cancelled: &impl Fn() -> bool,
    on_token: &mut impl FnMut(i32),
    generated: &mut Vec<i32>,
) -> Result<G4SpecOut, String> {
    let first_tok = *generated.last().expect("prefill produced the first token");
    let n0 = generated.len();
    let mut cursor = prompt_len as u64;
    let mut stats = SpecStats::default();
    let mut stopped_by_eos = false;
    let mut stopped_by_cancel = false;

    let ks = eng.specrt.as_ref().expect("spec runtime").cfg.ks.clone();
    let kmax = ks[0] as usize;
    let mut policy = G4Widths::new(&ks);
    stats.accept_hist = vec![0; kmax + 1];
    stats.verified = vec![0; kmax];
    let addr = |eng: &Gemma4Engine, name: &str| -> Result<u64, String> {
        eng.step_act_addr
            .get(name)
            .copied()
            .ok_or_else(|| format!("speculative buffer '{name}' is not allocated"))
    };
    let emit_addr = [addr(eng, "sp_emit0")?, addr(eng, "sp_emit1")?];
    let max_chunks = eng.max_chunks;
    let bucket = |end: u64| step_bucket(end.div_ceil(G4_CHUNK), max_chunks);

    // ---- first block: drafts for the first step, from the last prompt row ----
    // (recipes compiled on first use are not decode time: the clock starts after them)
    let room0 = eng.max_ctx.saturating_sub(cursor);
    let kd0 = policy
        .choose()
        .min(room0.saturating_sub(1))
        .min(budget.remaining().saturating_sub(1));
    let nch0 = bucket(cursor + kd0 + 1);
    ensure_step(eng, kd0 + 1, policy.choose(), nch0)?;
    if kd0 > 0 {
        ensure_block0(eng, kd0, nch0)?;
    }
    let t_decode = Instant::now();
    let mut compile_ms = 0.0f64;
    stage_block(eng, first_tok, cursor, kd0, nch0, h_addr)?;
    // drafts waiting on the card for the next step to enqueue
    let mut avail = kd0;
    let mut p = 0usize;

    let mk_event = || {
        eng.dev
            .create_event(false)
            .map_err(|e| format!("create step event: {e}"))
    };
    let events = [mk_event()?, mk_event()?];
    struct Flight {
        k: u64,
        kd: u64,
        p: usize,
        launched: Instant,
    }
    let mut flights: std::collections::VecDeque<Flight> = std::collections::VecDeque::new();
    let mut last_done = Instant::now();
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
        // ---- enqueue (two launches in flight: the host's work hides under device time) ----
        while flights.len() < 2 {
            if is_cancelled() {
                stopped_by_cancel = true;
                break;
            }
            let ahead: u64 = flights.iter().map(|f| f.k + 1).sum();
            // never enqueue behind a step that may already complete the request's budget
            if budget.remaining() <= ahead {
                break;
            }
            // the cursor of the step to launch lies in [cur_lo, cur_hi] until the flights land
            let cur_lo = cursor + flights.len() as u64;
            let cur_hi = cursor + ahead;
            let room = eng.max_ctx.saturating_sub(cur_hi);
            if room == 0 {
                break;
            }
            let t = (avail + 1).min(room);
            let k = t - 1;
            let kd = policy.choose().min(room.saturating_sub(t + 1));
            let nch = bucket(cur_hi + t);
            if bucket(cur_lo + t) != nch {
                // the chunk bucket this step needs depends on a result still in flight
                break;
            }
            let t_rec = Instant::now();
            tryd!(ensure_step(eng, t, kd, nch));
            let t_launch = Instant::now();
            compile_ms += t_launch.duration_since(t_rec).as_secs_f64() * 1000.0;
            {
                let step = &eng.specrt.as_ref().expect("spec runtime").steps[&(t, kd, nch)];
                tryd!(step
                    .recipe
                    .launch_prepared(&eng.dev, &step.launches[p])
                    .map_err(|e| format!("launch spec step (cursor<={cur_hi}, T={t}): {e}")));
            }
            tryd!(events[p]
                .record_compute(&eng.dev)
                .map_err(|e| format!("record step event: {e}")));
            stats.host_ms[0] += t_launch.elapsed().as_secs_f64() * 1000.0;
            flights.push_back(Flight {
                k,
                kd,
                p,
                launched: t_launch,
            });
            p ^= 1;
            avail = kd;
        }
        let Some(f) = flights.pop_front() else {
            break;
        };
        if stopped_by_cancel {
            flights.push_front(f);
            break;
        }

        // ---- collect the oldest step ----
        let t = f.k + 1;
        let t_wait = Instant::now();
        tryd!(events[f.p]
            .synchronize()
            .map_err(|e| format!("wait spec step (cursor={cursor}): {e}")));
        stats.host_ms[1] += t_wait.elapsed().as_secs_f64() * 1000.0;
        let t_read = Instant::now();
        let emitted = tryd!(read_i32s(eng, emit_addr[f.p], t as usize));
        let n_emit = emitted.iter().take_while(|&&v| v >= 0).count();
        if n_emit == 0 {
            let _ = eng.dev.sync_all();
            return Err(format!(
                "spec step at cursor {cursor}: the accept kernel emitted no token ({emitted:?})"
            ));
        }
        let a = n_emit - 1;
        stats.host_ms[2] += t_read.elapsed().as_secs_f64() * 1000.0;
        // steady state: one step per completion interval; the first step counts from its launch
        let now = Instant::now();
        let step_ms = now.duration_since(last_done.max(f.launched)).as_secs_f64() * 1000.0;
        last_done = now;

        stats.launches += 1;
        stats.accept_hist[a] += 1;
        for v in stats.verified.iter_mut().take(f.k as usize) {
            *v += 1;
        }
        *stats.widths.entry(f.k).or_insert(0) += 1;
        policy.observe(f.k, a, (f.k == f.kd).then_some(step_ms));

        let mut step_emitted = 0u64;
        for &tok in &emitted[..n_emit] {
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
        eng.live.step(step_emitted, f.k, a as u64);
        if stopped_by_eos || budget.remaining() == 0 {
            break;
        }
        cursor += n_emit as u64;
    }
    // Decode time ends with the last token; a step enqueued ahead of an EOS / cancel still runs
    // on the card: wait for it and drop its tokens.
    let decode_ms = (t_decode.elapsed().as_secs_f64() * 1000.0 - compile_ms).max(0.0);
    if !flights.is_empty() {
        let t_drain = Instant::now();
        eng.sync_compute("drain spec steps")?;
        stats.drain_ms = t_drain.elapsed().as_secs_f64() * 1000.0;
    }

    let decode_tokens = generated.len() - n0;
    if eng.verbose && stats.launches > 0 {
        let n = stats.launches as f64;
        eprintln!(
            "[gaudium] spec host split per launch: enqueue {:.3} ms, wait {:.3} ms, read-back {:.3} ms; widths K:launches {:?}; first-use recipes {:.1} ms (not decode time); drained after the last token: {:.1} ms",
            stats.host_ms[0] / n,
            stats.host_ms[1] / n,
            stats.host_ms[2] / n,
            stats.widths,
            compile_ms,
            stats.drain_ms
        );
    }
    if let Some(rt) = eng.specrt.as_mut() {
        rt.last_stats = stats.clone();
    }
    Ok(G4SpecOut {
        decode_ms,
        decode_tokens,
        stopped_by_eos,
        stopped_by_cancel,
        stats,
    })
}

/// Speculative decode of a sampled request.
///
/// The verify recipe is the greedy one without drafter passes; the acceptance moves to the host.
/// Per launch the host enqueues the drafts of the next block (first-block recipe, from the last
/// sampled token and the final-norm row that predicted it) and the verify step behind them, then
/// samples the verify rows in order with the engine's position-seeded sampler (row `i` samples
/// the token after position `cursor + i`, exactly as the non-speculative loop does) and accepts
/// drafts while they equal the samples. The stream is therefore the non-speculative stream of
/// the same seed whatever the drafts are; the drafts themselves stay the greedy chain. One
/// launch at a time: the next block needs the sampled token.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_sampled(
    eng: &mut Gemma4Engine,
    prompt_len: usize,
    h_addr: u64,
    params: &GenParams,
    budget: &mut SequenceBudget,
    is_eos: &impl Fn(i32) -> bool,
    is_cancelled: &impl Fn() -> bool,
    on_token: &mut impl FnMut(i32),
    generated: &mut Vec<i32>,
) -> Result<G4SpecOut, String> {
    let n0 = generated.len();
    let mut cursor = prompt_len as u64;
    let mut stats = SpecStats::default();
    let mut stopped_by_eos = false;
    let mut stopped_by_cancel = false;

    let ks = eng.specrt.as_ref().expect("spec runtime").cfg.ks.clone();
    let kmax = ks[0] as usize;
    let mut policy = G4Widths::new(&ks);
    stats.accept_hist = vec![0; kmax + 1];
    stats.verified = vec![0; kmax];
    let addr = |eng: &Gemma4Engine, name: &str| -> Result<u64, String> {
        eng.step_act_addr
            .get(name)
            .copied()
            .ok_or_else(|| format!("speculative buffer '{name}' is not allocated"))
    };
    let (ids0, logits, h_final) = (
        addr(eng, "sp_ids0")?,
        addr(eng, "logits")?,
        addr(eng, "h_final")?,
    );
    let row_bytes = eng.dims.vocab * 2;
    let hid_bytes = eng.dims.hid * 2;
    let max_chunks = eng.max_chunks;
    let bucket = |end: u64| step_bucket(end.div_ceil(G4_CHUNK), max_chunks);

    let mut anchor = *generated.last().expect("prefill produced the first token");
    let mut h_addr = h_addr;
    let t_decode = Instant::now();
    let mut compile_ms = 0.0f64;
    while !stopped_by_eos && budget.remaining() > 0 {
        if is_cancelled() {
            stopped_by_cancel = true;
            break;
        }
        let room = eng.max_ctx.saturating_sub(cursor);
        if room == 0 {
            break;
        }
        let kd = policy
            .choose()
            .min(room - 1)
            .min(budget.remaining().saturating_sub(1));
        let t = kd + 1;
        let nch = bucket(cursor + t);
        let t_rec = Instant::now();
        ensure_step(eng, t, 0, nch)?;
        if kd > 0 {
            ensure_block0(eng, kd, nch)?;
        }
        let t_step = Instant::now();
        compile_ms += t_step.duration_since(t_rec).as_secs_f64() * 1000.0;

        // drafts for this step, and the verify behind them
        stage_block(eng, anchor, cursor, kd, nch, h_addr)?;
        {
            let step = &eng.specrt.as_ref().expect("spec runtime").steps[&(t, 0, nch)];
            step.recipe
                .launch_prepared(&eng.dev, &step.launches[0])
                .map_err(|e| format!("launch spec step (cursor={cursor}, T={t}): {e}"))?;
        }
        let t_launched = t_step.elapsed().as_secs_f64() * 1000.0;
        eng.sync_compute("sampled spec step")?;
        let t_synced = t_step.elapsed().as_secs_f64() * 1000.0;

        // host-sampled accept: rows are read and sampled only up to the first mismatch
        let ids = read_i32s(eng, ids0, t as usize)?;
        let mut toks: Vec<i32> = Vec::with_capacity(t as usize);
        for i in 0..t as usize {
            let tok = eng.sample_from_logits(
                logits + i as u64 * row_bytes,
                params,
                (cursor + i as u64) as u32,
            )?;
            toks.push(tok);
            if i + 1 >= t as usize || ids[i + 1] != tok {
                break;
            }
        }
        let n_emit = toks.len();
        let a = n_emit - 1;
        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
        stats.host_ms[0] += t_launched;
        stats.host_ms[1] += t_synced - t_launched;
        stats.host_ms[2] += step_ms - t_synced;
        stats.launches += 1;
        stats.accept_hist[a] += 1;
        for v in stats.verified.iter_mut().take(kd as usize) {
            *v += 1;
        }
        *stats.widths.entry(kd).or_insert(0) += 1;
        policy.observe(kd, a, Some(step_ms));

        let mut step_emitted = 0u64;
        for &tok in &toks {
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
        eng.live.step(step_emitted, kd, a as u64);
        cursor += n_emit as u64;
        anchor = toks[a];
        h_addr = h_final + a as u64 * hid_bytes;
    }

    let decode_ms = (t_decode.elapsed().as_secs_f64() * 1000.0 - compile_ms).max(0.0);
    let decode_tokens = generated.len() - n0;
    if eng.verbose && stats.launches > 0 {
        let n = stats.launches as f64;
        eprintln!(
            "[gaudium] spec host split per launch (sampled): stage + enqueue {:.3} ms, wait {:.3} ms, read + sample rows {:.3} ms; widths K:launches {:?}; first-use recipes {:.1} ms (not decode time)",
            stats.host_ms[0] / n,
            stats.host_ms[1] / n,
            stats.host_ms[2] / n,
            stats.widths,
            compile_ms
        );
    }
    if let Some(rt) = eng.specrt.as_mut() {
        rt.last_stats = stats.clone();
    }
    Ok(G4SpecOut {
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

    fn assist_json(
        hid: u64,
        nh: u64,
        s_nkv: u64,
        g_nkv: &str,
        bh: u64,
        window: u64,
        ordered: bool,
    ) -> String {
        format!(
            r#"{{
              "architectures": ["Gemma4AssistantForCausalLM"],
              "backbone_hidden_size": {bh},
              "centroid_intermediate_top_k": 32,
              "model_type": "gemma4_assistant",
              "num_centroids": 2048,
              "text_config": {{
                "final_logit_softcapping": null,
                "global_head_dim": 512, "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": {hid}, "hidden_size_per_layer_input": 0,
                "intermediate_size": {},
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "num_attention_heads": {nh},
                "num_global_key_value_heads": {g_nkv},
                "num_hidden_layers": 4,
                "num_key_value_heads": {s_nkv},
                "num_kv_shared_layers": 4,
                "sliding_window": {window},
                "vocab_size": 262144
              }},
              "use_ordered_embeddings": {ordered}
            }}"#,
            hid * 8
        )
    }

    fn target(tc: &str, layers: usize, every: usize) -> ModelSpec {
        let types: Vec<&str> = (0..layers)
            .map(|l| {
                if (l + 1) % every == 0 {
                    "\"full_attention\""
                } else {
                    "\"sliding_attention\""
                }
            })
            .collect();
        ModelSpec::from_config_json(&format!(
            r#"{{
              "architectures": ["Gemma4ForConditionalGeneration"],
              "model_type": "gemma4", "tie_word_embeddings": true, "eos_token_id": [1, 106],
              "text_config": {{
                "model_type": "gemma4_text", "vocab_size": 262144, "num_hidden_layers": {layers},
                "head_dim": 256, "global_head_dim": 512, "final_logit_softcapping": 30.0,
                "rms_norm_eps": 1e-06, "layer_types": [{}],
                "rope_parameters": {{
                    "sliding_attention": {{"rope_theta": 10000.0, "rope_type": "default"}},
                    "full_attention": {{"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}}
                }},
                {tc}
              }}
            }}"#,
            types.join(", ")
        ))
        .unwrap()
    }

    fn b12() -> ModelSpec {
        target(
            r#""hidden_size": 3840, "intermediate_size": 15360, "num_attention_heads": 16,
            "num_key_value_heads": 8, "num_global_key_value_heads": 1, "sliding_window": 1024,
            "attention_k_eq_v": true, "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 262144, "num_kv_shared_layers": 0,
            "use_double_wide_mlp": false"#,
            48,
            6,
        )
    }

    fn e2b() -> ModelSpec {
        target(
            r#""hidden_size": 1536, "intermediate_size": 6144, "num_attention_heads": 8,
            "num_key_value_heads": 1, "num_global_key_value_heads": null, "sliding_window": 512,
            "attention_k_eq_v": false, "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144, "num_kv_shared_layers": 20,
            "use_double_wide_mlp": true"#,
            35,
            5,
        )
    }

    fn names(gb: &GraphBuilder) -> BTreeMap<String, (Vec<u64>, u64)> {
        gb.persistent_tensor_specs()
            .into_iter()
            .map(|(n, _, dims, b)| (n, (dims, b)))
            .collect()
    }

    #[test]
    fn drafter_config_is_checked_against_the_target() {
        let dir = Path::new("drafter");
        let a = G4Assist::from_config_json(dir, &assist_json(1024, 16, 8, "1", 3840, 1024, false))
            .unwrap();
        assert_eq!((a.hid, a.inter, a.backbone, a.nh), (1024, 8192, 3840, 16));
        assert_eq!(a.glob, vec![false, false, false, true]);
        assert!(a.ordered.is_none());
        let d = G4Dims::from_spec(&b12(), 256, G4NormMode::Vendor).unwrap();
        a.check_target(&d, 9).unwrap();
        // another backbone, another window, another kv head count: refused by name
        for (bh, w, nkv) in [(5376u64, 1024u64, 8u64), (3840, 512, 8), (3840, 1024, 4)] {
            let o = G4Assist::from_config_json(dir, &assist_json(1024, 16, nkv, "1", bh, w, false))
                .unwrap();
            assert!(o.check_target(&d, 9).is_err());
        }
        // a checkpoint that is not a drafter of this family
        assert!(G4Assist::from_config_json(dir, r#"{"model_type": "gemma4"}"#).is_err());
        // the small sizes' ordered readout: 2048 clusters of 128 tokens, 32 looked into
        let o = G4Assist::from_config_json(dir, &assist_json(256, 4, 1, "null", 1536, 512, true))
            .unwrap();
        assert_eq!(o.ordered, Some((2048, 32, 128)));
        let de = G4Dims::from_spec(&e2b(), 256, G4NormMode::Tile).unwrap();
        o.check_target(&de, 5).unwrap();
    }

    #[test]
    fn widths_are_validated_and_sorted() {
        let a = G4Assist::from_config_json(
            Path::new("d"),
            &assist_json(1024, 16, 8, "1", 3840, 1024, false),
        )
        .unwrap();
        let c = G4SpecCfg::new(G4Drafter::Assist(a.clone()), &[2, 6, 4, 6]).unwrap();
        assert_eq!(c.ks, vec![6, 4, 2]);
        assert_eq!((c.kmax(), c.max_rows()), (6, 7));
        assert!(G4SpecCfg::new(G4Drafter::Assist(a.clone()), &[]).is_err());
        assert!(G4SpecCfg::new(G4Drafter::Assist(a.clone()), &[0]).is_err());
        assert!(G4SpecCfg::new(G4Drafter::Assist(a), &[SPEC_MAX_ROWS]).is_err());
    }

    #[test]
    fn step_graph_verifies_rows_and_drafts_from_the_target_state() {
        let d = G4Dims::from_spec(&b12(), 256, G4NormMode::Vendor).unwrap();
        let a = G4Assist::from_config_json(
            Path::new("d"),
            &assist_json(1024, 16, 8, "1", 3840, 1024, false),
        )
        .unwrap();
        let sc = G4SpecCfg::new(G4Drafter::Assist(a), &[4]).unwrap();
        let g = build_spec_step_graph(&d, &sc, 8192, 2, 5, 4);
        let n = names(&g);
        // verify rows: every decoder activation has 5 rows; one logits row per verify row
        assert_eq!(n["ids"].0, vec![5]);
        assert_eq!(n["L0_h"].0, vec![3840, 5]);
        assert_eq!(n["logits"].0, vec![262144, 5]);
        assert_eq!(n["emitted"].0, vec![5]);
        assert_eq!(n["next_ids"].0, vec![5]);
        // the drafter reads the target's last sliding ring and last global cache: no K / V of its own
        assert_eq!(n["L46_kring"].0, vec![256, 2048, 8]);
        assert_eq!(n["L47_kcache"].0, vec![512, 4096, 2]);
        assert_eq!(
            n.keys()
                .filter(|k| k.ends_with("ring") || k.ends_with("cache"))
                .count(),
            2 * 48
        );
        // 16 query heads over 8 sliding kv heads / 1 global kv head (two chunks)
        assert_eq!(n["gp1_l0_qb"].0, vec![256, 2, 8]);
        assert_eq!(n["gp4_l3_qb"].0, vec![512, 16, 2]);
        assert_eq!(n["gd_table"].0, vec![1024, 262144]);
        assert_eq!(n["gd_pre"].0, vec![7680, 1024]);
        assert!(n.contains_key("gp4_hout") && !n.contains_key("gp5_hout"));
        // every drafter tensor is a spec weight, no decoder tensor is
        for k in n.keys() {
            assert_eq!(
                is_spec_weight(k),
                k.starts_with("gd") || k.starts_with("sp_"),
                "{k}"
            );
        }
        // a step that drafts nothing has no drafter tensor at all
        let g0 = build_spec_step_graph(&d, &sc, 8192, 2, 1, 0);
        let n0 = names(&g0);
        assert!(!n0
            .keys()
            .any(|k| k.starts_with("gd") || k.starts_with("gp")));
        assert_eq!(n0["next_ids"].0, vec![1]);
        assert_ne!(g.fingerprint("x"), g0.fingerprint("x"));
        // the one-row verify computes exactly the tensors of the non-speculative step
        let k0 = names(&build_step_graph(&d, 8192, 2, &[]));
        for (k, v) in &k0 {
            assert_eq!(n0.get(k), Some(v), "{k}");
        }
    }

    #[test]
    fn sizing_graphs_bound_every_recipe_of_the_width_set() {
        let d = G4Dims::from_spec(&e2b(), 256, G4NormMode::Tile).unwrap();
        let a = G4Assist::from_config_json(
            Path::new("d"),
            &assist_json(256, 4, 1, "null", 1536, 512, true),
        )
        .unwrap();
        let c = G4SpecCfg::new(G4Drafter::Assist(a), &[4, 2]).unwrap();
        let mut bound: BTreeMap<String, u64> = BTreeMap::new();
        for g in c.sizing_graphs(&d, 8192, 2) {
            for (n, _, _, b) in g.persistent_tensor_specs() {
                let e = bound.entry(n).or_insert(0);
                *e = (*e).max(b);
            }
        }
        for (t, kd, nch) in [
            (5, 4, 1),
            (5, 2, 2),
            (3, 4, 2),
            (3, 2, 1),
            (2, 1, 2),
            (1, 3, 1),
        ] {
            for (n, _, _, b) in
                build_spec_step_graph(&d, &c, 8192, nch, t, kd).persistent_tensor_specs()
            {
                assert!(bound.get(&n).is_some_and(|&m| b <= m), "{n} T={t} D={kd}");
            }
        }
        for (n, _, _, b) in build_block0_graph(&d, &c, 8192, 2, 3).persistent_tensor_specs() {
            assert!(bound.get(&n).is_some_and(|&m| b <= m), "{n}");
        }
        // the ordered readout: candidates of 32 clusters x 128 tokens, scored by one GEMM
        let n = names(&build_block0_graph(&d, &c, 8192, 2, 1));
        assert_eq!(n["gd_order"].0, vec![128, 2048]);
        assert_eq!(n["gp1_cemb"].0, vec![256, 4096]);
        assert_eq!(n["gp1_clogits"].0, vec![4096, 1]);
        assert_eq!(n["b0_ids"].0, vec![2]);
        // the small size's drafter attends the layers its own key-less layers read
        assert!(n.contains_key("L13_kring") && n.contains_key("L14_kcache"));
    }

    #[test]
    fn width_policy_tries_every_width_then_follows_acceptance() {
        let mut p = G4Widths::new(&[6, 2]);
        assert_eq!(p.choose(), 6);
        for _ in 0..G4Widths::MIN_TIMED {
            p.observe(6, 0, Some(40.0));
        }
        assert_eq!(p.choose(), 2);
        // an off-width step informs the acceptance only
        p.observe(2, 0, None);
        assert_eq!(p.choose(), 2);
        for _ in 0..G4Widths::MIN_TIMED {
            p.observe(2, 0, Some(32.0));
        }
        // nothing is accepted: the cheaper step wins
        for _ in 0..40 {
            p.observe(
                p.choose(),
                0,
                Some(if p.choose() == 6 { 40.0 } else { 32.0 }),
            );
        }
        assert_eq!(p.choose(), 2);
        // every draft accepted at either width: six drafts at 40 ms beat two at 32 ms
        let mut p = G4Widths::new(&[6, 2]);
        for _ in 0..60 {
            p.observe(6, 6, Some(40.0));
            p.observe(2, 2, Some(32.0));
        }
        assert_eq!(p.choose(), 6);
        // one width: no policy
        assert_eq!(G4Widths::new(&[4]).choose(), 4);
    }

    #[test]
    fn block_drafter_step_taps_the_target_and_keeps_bounded_context_state() {
        let d = G4Dims::from_spec(&b12(), 256, G4NormMode::Vendor).unwrap();
        let cfg = r#"{
            "architectures": ["DFlashDraftModel"], "block_size": 16,
            "dflash_config": {"mask_token_id": 4, "target_layer_ids": [1, 10, 19, 27, 36, 45]},
            "head_dim": 128, "hidden_act": "silu", "hidden_size": 3840, "intermediate_size": 7680,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                            "sliding_attention", "full_attention"],
            "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
            "rope_parameters": {"rope_theta": 1000000}, "sliding_window": 2048,
            "vocab_size": 262144
        }"#;
        let c = dflash::G4DFlash::from_config_json(Path::new("d"), cfg).unwrap();
        c.check_target(&d, 8).unwrap();
        assert!(c.check_target(&d, 17).is_err());
        assert!(G4SpecCfg::new(G4Drafter::DFlash(c.clone()), &[7, 3]).is_err());
        let sc = G4SpecCfg::new(G4Drafter::DFlash(c), &[7]).unwrap();
        let n = names(&build_spec_step_graph(&d, &sc, 8192, 2, 8, 7));
        // context of the 8 verified rows from six tapped layer outputs
        assert_eq!(n["ctx_cat"].0, vec![6 * 3840, 8]);
        assert_eq!(n["L45_out"].0, vec![3840, 8]);
        // windowed layers keep rings, the full layer a cache of the context's length
        assert_eq!(n["D0_kring"].0, vec![128, 4096, 8]);
        assert_eq!(n["D4_kcache"].0, vec![128, 8192, 8]);
        assert_eq!(n["D3_sc"].0, vec![4096, 32, 8]);
        assert_eq!(n["D4_sc"].0, vec![8192, 32, 8]);
        // the block: the accepted token and seven masks, one draft per mask row
        assert_eq!(n["df_ids_blk"].0, vec![8]);
        assert_eq!(n["blk_logits"].0, vec![262144, 8]);
        assert_eq!(n["blk_drafts"].0, vec![7]);
        assert_eq!(n["next_ids"].0, vec![8]);
        assert_eq!(n["rope128_c"].0, vec![128, 8192 + 256]);
        for k in n.keys() {
            assert!(!(is_spec_weight(k) && is_spec_state(k)), "{k}");
        }
        // every recipe of the load is bounded by the sizing graphs
        let mut bound: BTreeMap<String, u64> = BTreeMap::new();
        for g in sc.sizing_graphs(&d, 8192, 2) {
            for (name, _, _, b) in g.persistent_tensor_specs() {
                let e = bound.entry(name).or_insert(0);
                *e = (*e).max(b);
            }
        }
        let others = [
            build_spec_step_graph(&d, &sc, 8192, 1, 8, 7),
            build_spec_step_graph(&d, &sc, 8192, 2, 3, 2),
            build_block0_graph(&d, &sc, 8192, 1, 5),
            dflash::build_ctx_graph(&d, sc_dflash(&sc), 52, 8192, 4096, &[]),
        ];
        for g in &others {
            for (name, _, _, b) in g.persistent_tensor_specs() {
                assert!(bound.get(&name).is_some_and(|&m| b <= m), "{name}");
            }
        }
        // the first block starts from host-written ids and writes the first step's ids
        let b0 = names(&build_block0_graph(&d, &sc, 8192, 1, 7));
        assert_eq!(b0["b0_blk"].0, vec![8]);
        assert_eq!(b0["b0_ids"].0, vec![8]);
        assert!(!b0.contains_key("b0_h"));
    }

    fn sc_dflash(sc: &G4SpecCfg) -> &dflash::G4DFlash {
        match &sc.drafter {
            G4Drafter::DFlash(c) => c,
            G4Drafter::Assist(_) => panic!("not a block drafter"),
        }
    }
}
