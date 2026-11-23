//! Speculative decoding core for the dense Qwen3 / Qwen3.5 / Qwen3.8 engine: multi-row verify step
//! recipes with per-row Gated DeltaNet checkpoints, on-device acceptance, and the decode loop.
//!
//! This is a child module of `engine::qwen3` (declared there with `#[path]`), so it reuses that
//! module's private graph helpers and engine fields without widening their visibility.
//!
//! One launch of a step recipe `g_spec_*_T{T}_C{nch}` does, for `T = K + 1` rows:
//! - **verify**: row 0 = the last accepted token, rows `1..=K` = drafts; embed -> every trunk
//!   layer over `T` rows (causal attention writing KV rows `cursor .. cursor + T`; Gated DeltaNet
//!   layers starting from checkpoint slot `a_in[0]` of the previous step's state ring and writing
//!   one state checkpoint per row into the other ring) -> final norm -> `lm_head` -> argmax;
//! - **accept** (`gaudium_accept_t`): the longest draft prefix with `ids[i + 1] == tokens[i]` ->
//!   `a_out` (row to resume from), `cursor_out = cursor + a + 1`, `emitted` (accepted drafts, then
//!   the bonus token `tokens[a]`, then `-1`);
//! - **draft** (optional, a device drafter such as DSpine): the drafter's context update for the
//!   verified rows and its next block -> `next_ids` (row 0 = the bonus token).
//!
//! All step state stays on the card and alternates between two buffers by step parity (`ids` /
//! `next_ids`, `cursor` / `cursor_out`, `a_in` / `a_out`, `emitted`, the two DeltaNet state
//! rings), so the host only launches, waits, and reads the `emitted` row. With a device drafter
//! nothing the host launches depends on a device value, so the next step is enqueued while the
//! current one runs.
//!
//! Exactness: tokens are always the verify argmax chain, so the stream equals the engine's own
//! non-speculative `K = 0` stream whatever the drafts are; a wrong draft only shortens the
//! accepted prefix. Rows after the accepted prefix leave KV rows and state checkpoints behind
//! that the next step overwrites (KV) or never selects (checkpoints).
//!
//! Sampled requests run the same loop on a second recipe family (`g_spec_*s_T{T}_C{nch}`): the
//! on-card sampler (`engine::sampler`) replaces the argmax, so verify row `i` at cursor `c`
//! draws the token at sequence index `c + i + 1` with that index's noise. The token at an index
//! is a function of (prefix, seed, index) only, so the accept kernel is unchanged ("draft equals
//! the row's token") and the stream equals the `K = 0` stream of the same seed by construction.
//! Greedy requests never launch that family; their graphs and fingerprints are untouched.

use super::*;
use crate::tree::{choose_shape, make_shape, TreeCost, TreeShape, TreeStats};

#[path = "dflash2.rs"]
pub mod dflash2;
#[path = "dspine.rs"]
pub mod dspine;
// --- mtptree: the MTP drafter and tree verify ---
#[path = "mtp.rs"]
pub mod mtp;
#[path = "tree_step.rs"]
pub mod tree_step;
// --- step: checkpoint-free DeltaNet verify rows (`--gdn-verify replay`) ---
#[path = "spec_replay.rs"]
pub mod replay;
// --- step: DeltaNet verify rows on the MME (`--gdn-verify mme`, not token-identical) ---
#[path = "spec_vmme.rs"]
pub mod vmme;
// --- readout: runtime description and the step launches' byte model (host-only) ---
#[path = "spec_readout.rs"]
pub mod readout;

/// Most verify rows a step may carry (the decode builds of the DeltaNet kernels handle `T <= 32`).
pub const SPEC_MAX_ROWS: u64 = 32;

/// Row recurrence with vector-loaded gate scalars (checkpoint form; served by the step glue).
pub const GDNV_ELF: &str = "GAUDIUM_GDNV_ELF";

/// Kernel ELFs every speculative step needs beyond the trunk's own.
const SPEC_VERIFY_ELFS: &[&str] = &["GAUDIUM_ACCEPTT_ELF"];

/// One request of the host-side oracle drafter (test mode): the known continuation and the
/// generated-token indices whose draft is deliberately replaced by a wrong token.
#[derive(Debug, Clone, Default)]
pub struct OracleRequest {
    pub stream: Vec<i32>,
    pub corrupt: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum SpecMethod {
    /// Drafts come from a known token stream on the host (verify / accept / rollback test).
    Oracle { requests: Vec<OracleRequest> },
    /// The DSpine block drafter runs on the card inside the step recipe.
    DSpine { draft_model: PathBuf },
    /// The DFlash2 block drafter runs on the card inside the step recipe.
    DFlash2 { draft_model: PathBuf },
    /// The target's own MTP head drafts on the card: a chain of `draft_k` tokens, or (`tree`)
    /// a caterpillar tree whose leaves are the head's runner-up candidates. `reference` (tree
    /// only): known streams, one per request, whose first divergence is reported.
    /// `draft_vocab` > 0: the head's readouts cover only the first `draft_vocab` token ids.
    Mtp {
        tree: Option<TreeShape>,
        reference: Vec<Vec<i32>>,
        draft_vocab: u64,
    },
    /// Host oracle drafts laid out on a draft tree (tree verify / accept / path-move test).
    OracleTree {
        requests: Vec<OracleRequest>,
        shape: TreeShape,
    },
    /// A Gemma 4 size's own drafter checkpoint (`-assistant`): a small decoder that attends the
    /// target's K / V state, run on the card inside the step recipe (`engine/gemma4_spec.rs`).
    Assistant { draft_model: PathBuf },
}

#[derive(Debug, Clone)]
pub struct SpecConfig {
    pub method: SpecMethod,
    /// Drafts verified per launch. One value = fixed width; several = the adaptive policy's set.
    pub draft_k: Vec<u64>,
}

impl SpecConfig {
    pub fn method_name(&self) -> &'static str {
        match self.method {
            SpecMethod::Oracle { .. } => "oracle",
            SpecMethod::DSpine { .. } => "dspine",
            SpecMethod::DFlash2 { .. } => "dflash2",
            SpecMethod::Mtp { tree: None, .. } => "mtp",
            SpecMethod::Mtp { .. } => "mtp-tree",
            SpecMethod::OracleTree { .. } => "oracle-tree",
            SpecMethod::Assistant { .. } => "assistant",
        }
    }
}

/// Host twin of the chain branch of `gaudium_accept_t`: accepted draft count for one step.
pub fn chain_accept(ids: &[i32], tokens: &[i32]) -> usize {
    let mut a = 0usize;
    while a + 1 < ids.len() && a < tokens.len() && ids[a + 1] == tokens[a] {
        a += 1;
    }
    a
}

/// Verify-width policy of a chain drafter: the candidate widths as chain shapes of the shared
/// tree policy (`crate::tree`: per-position conditional acceptance with a pooled tail for depths
/// not tested yet, expected tokens per millisecond).
///
/// The cost side is a card measurement: every width is launched [`WidthPolicy::MIN_TIMED`] times
/// first (widest first), after which its own measured launch time is used. The measured times
/// outlive a request ([`WidthPolicy::with_cost`] / [`WidthPolicy::into_cost`]): a loaded engine
/// times its widths once, in its first request, and later requests choose from their first
/// launch on (a width costs the same whatever the prompt; what a request has to learn is its
/// own acceptance).
#[derive(Debug, Clone)]
pub struct WidthPolicy {
    shapes: Vec<TreeShape>,
    keys: Vec<String>,
    stats: TreeStats,
    cost: TreeCost,
    /// Steps since draft position `j` was last tested (a step tests positions up to its first
    /// miss, at most its width). An estimate older than [`WidthPolicy::MAX_AGE`] steps is not
    /// trusted: the position falls back to the pooled rate, so a deep position that failed a few
    /// times long ago cannot keep the wide step from being tried again.
    age: Vec<usize>,
}

impl WidthPolicy {
    /// Launches of a width before its measured time replaces "try it".
    pub const MIN_TIMED: usize = 3;
    /// Steps after which an untested position's own acceptance estimate expires.
    pub const MAX_AGE: usize = 48;

    pub fn new(ks: &[u64]) -> Self {
        let mut ks = ks.to_vec();
        ks.sort_unstable_by(|a, b| b.cmp(a));
        let kmax = ks.first().copied().unwrap_or(1) as usize;
        Self {
            keys: ks.iter().map(|k| format!("chain{k}")).collect(),
            shapes: ks
                .iter()
                .map(|&k| make_shape(k as usize, &[], &format!("chain{k}")))
                .collect(),
            stats: TreeStats::new(kmax, 1),
            cost: TreeCost::default(),
            age: vec![0; kmax + 2],
        }
    }

    /// The policy over `ks` with launch times measured earlier (by a previous request).
    pub fn with_cost(ks: &[u64], cost: TreeCost) -> Self {
        let mut p = Self::new(ks);
        p.cost = cost;
        p
    }

    /// The launch times measured so far, for the next request's policy.
    pub fn into_cost(self) -> TreeCost {
        self.cost
    }

    fn timed(&self, i: usize) -> usize {
        self.cost.measured.get(&self.keys[i]).map_or(0, |m| m.1)
    }

    /// The `K` to verify next among the widths whose `K + 1` rows fit in `room`; 0 if none fits.
    pub fn choose(&self, room: u64) -> u64 {
        self.choose_with_pending(room, &[])
    }

    /// [`Self::choose`] with the widths of the launches still in flight (`pending`), which count
    /// as timed already: with enqueue-ahead a width would otherwise be launched once more per
    /// launch in flight than its timing needs.
    pub fn choose_with_pending(&self, room: u64, pending: &[u64]) -> u64 {
        let fits: Vec<usize> = (0..self.shapes.len())
            .filter(|&i| self.shapes[i].n_nodes() as u64 <= room)
            .collect();
        let in_flight = |i: usize| {
            let k = self.shapes[i].spine() as u64;
            pending.iter().filter(|&&p| p == k).count()
        };
        if let Some(&i) = fits
            .iter()
            .find(|&&i| self.timed(i) + in_flight(i) < Self::MIN_TIMED)
        {
            return self.shapes[i].spine() as u64;
        }
        let cands: Vec<TreeShape> = fits.iter().map(|&i| self.shapes[i].clone()).collect();
        let keys: Vec<&str> = fits.iter().map(|&i| self.keys[i].as_str()).collect();
        // expired per-position estimates count as untested (the pooled rate stands in)
        let mut stats = self.stats.clone();
        for (j, &age) in self.age.iter().enumerate().skip(2) {
            if age > Self::MAX_AGE && j < stats.cn.len() {
                stats.cn[j] = 0;
            }
        }
        choose_shape(&stats, &cands, &self.cost, None, &keys).map_or(0, |c| cands[c].spine() as u64)
    }

    /// A step verified `k` drafts, the first `accepted` of them were right, and it took `ms`.
    pub fn observe(&mut self, k: u64, accepted: usize, ms: f64) {
        if let Some(i) = self.shapes.iter().position(|s| s.spine() as u64 == k) {
            self.stats.observe(&self.shapes[i], accepted, accepted);
            self.cost.observe(&self.keys[i], ms);
            let tested = (k as usize).min(accepted + 1);
            for (j, age) in self.age.iter_mut().enumerate().skip(1) {
                *age = if j <= tested { 0 } else { *age + 1 };
            }
        }
    }
}

// ============================================================================
// Graph builders
// ============================================================================

/// Gated DeltaNet layer over `t` verify rows with per-row state checkpoints: the recurrent and
/// conv states are read from slot `slot[0]` of `{.., ns}` rings and one checkpoint per row is
/// written to `S_ckpt` / `cs_ckpt` (flags: per-row checkpoints, final `S_out` / `cs_out` skipped).
#[allow(clippy::too_many_arguments)]
fn append_linear_layer_ckpt(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    slot: TensorId,
    lpfx: &str,
    t: u64,
    ns: u64,
    wnext_norm: Option<(&str, &str)>,
    fast_rows: bool,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let hv = spec.gdn_v_heads;
    let dl = spec.gdn_v_dim;
    let qkv = spec.gdn_qkv_dim();
    let qkvz_dim = spec.gdn_qkvz_dim();
    let hvp = spec.gdn_hvp();
    let ba_dim = spec.gdn_ba_dim();

    let h = match h_pre {
        Some(h_in) => h_in,
        None => add_rmsnorm(gb, spec, x, &format!("{lpfx}w_in"), &format!("{lpfx}h"), t),
    };
    let qkvz = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_qkvz"),
        hid,
        qkvz_dim,
        t,
        &format!("{lpfx}qkvz"),
    );
    let ba = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_ba"),
        hid,
        ba_dim,
        t,
        &format!("{lpfx}ba"),
    );

    let conv_w =
        gb.get_or_add_persistent_tensor(format!("{lpfx}conv_w"), SynDataType::Float, &[qkv, 4]);
    let conv_b =
        gb.get_or_add_persistent_tensor(format!("{lpfx}conv_b"), SynDataType::Float, &[qkv]);
    let cs_in = gb.add_persistent_tensor(format!("{lpfx}cs_in"), SynDataType::Float, &[qkv, 3, ns]);
    let cs_out = gb.add_persistent_tensor(format!("{lpfx}cs_out"), SynDataType::Float, &[qkv, 3]);
    let cs_ckpt =
        gb.add_persistent_tensor(format!("{lpfx}cs_ckpt"), SynDataType::Float, &[qkv, 3, t]);
    let q = gb.add_persistent_tensor(format!("{lpfx}q"), SynDataType::Float, &[dl, hv, t]);
    let k = gb.add_persistent_tensor(format!("{lpfx}k"), SynDataType::Float, &[dl, hv, t]);
    let v = gb.add_persistent_tensor(format!("{lpfx}v"), SynDataType::Float, &[dl, hv, t]);
    gb.add_node(
        "gaudium_gdn_prep",
        format!("{lpfx}gdn_prep"),
        &[qkvz, conv_w, cs_in, conv_b, slot],
        &[q, k, v, cs_out, cs_ckpt],
        &u32_params(&[2 | 16]),
    );

    let a_log = gb.get_or_add_persistent_tensor(format!("{lpfx}A_log"), SynDataType::Float, &[hvp]);
    let dt_bias =
        gb.get_or_add_persistent_tensor(format!("{lpfx}dt_bias"), SynDataType::Float, &[hvp]);
    let eg = gb.add_persistent_tensor(format!("{lpfx}eg"), SynDataType::Float, &[hvp, t]);
    let beta = gb.add_persistent_tensor(format!("{lpfx}beta"), SynDataType::Float, &[hvp, t]);
    gb.add_node(
        "gaudium_gdn_gate",
        format!("{lpfx}gdn_gate"),
        &[ba, a_log, dt_bias],
        &[eg, beta],
        &[],
    );

    let s_in = gb.add_persistent_tensor(
        format!("{lpfx}S_in"),
        SynDataType::Float,
        &[64, 256, hv, ns],
    );
    let s_out =
        gb.add_persistent_tensor(format!("{lpfx}S_out"), SynDataType::Float, &[64, 256, hv]);
    let s_ckpt = gb.add_persistent_tensor(
        format!("{lpfx}S_ckpt"),
        SynDataType::Float,
        &[64, 256, hv, t],
    );
    let o = gb.add_persistent_tensor(format!("{lpfx}o"), SynDataType::Float, &[dl, hv, t]);
    // [step] the same rows through the kernel build with vector-loaded gate scalars when the
    // step family carries it (bit-identical outputs)
    let gdn_guid = if fast_rows {
        "gaudium_gdn_f32v"
    } else {
        "gaudium_gdn_f32"
    };
    gb.add_node(
        gdn_guid,
        format!("{lpfx}gdn"),
        &[s_in, q, k, v, eg, beta, slot],
        &[o, s_out, s_ckpt],
        &u32_params(&[1 | 16]),
    );

    let norm_w =
        gb.get_or_add_persistent_tensor(format!("{lpfx}norm_w"), SynDataType::Float, &[dl]);
    let core = gb.add_persistent_tensor(format!("{lpfx}core"), SynDataType::Bf16, &[hv * dl, t]);
    gb.add_node(
        "gaudium_gdn_post",
        format!("{lpfx}gdn_post"),
        &[o, qkvz, norm_w],
        &[core],
        &u32_params(&[0u32]),
    );

    let attn = add_linear_bf16(
        gb,
        core,
        &format!("{lpfx}W_o"),
        hv * dl,
        hid,
        t,
        &format!("{lpfx}attn"),
    );
    append_post_and_mlp(gb, spec, x, attn, lpfx, t, true, wnext_norm)
}

/// Builds the speculative step graph for `t` verify rows over `nch` KV chunks.
///
/// `ns` = slots of the DeltaNet state rings (the widest step's rows), `drafter` = the device
/// drafter appended after the accept (None: verify + accept only, the host supplies `ids`).
pub fn build_spec_step_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    t: u64,
    ns: u64,
    drafter: Option<&dspine::DSpineCfg>,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    build_spec_step_graph_as(spec, lmax, nch, t, ns, drafter, elf_hashes, false)
}

/// [`build_spec_step_graph`], or (`sampled`) the step of a sampled request: every verify row
/// samples on the card at its own sequence index (row `i` -> index `cursor + i + 1`), and a
/// DSpine drafter walks coupled to that noise. `elf_hashes` must then include the sampler
/// kernels (and the coupled walk's).
#[allow(clippy::too_many_arguments)]
pub fn build_spec_step_graph_as(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    t: u64,
    ns: u64,
    drafter: Option<&dspine::DSpineCfg>,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    let kind = if drafter.is_some() { "ds" } else { "vf" };
    let fam = if sampled { "s" } else { "" };
    // [step] the replay form of the DeltaNet verify rows is a family of its own (other kernel,
    // other state buffers): its recipes carry an `r` in the name
    let gdn_replay = replay::enabled(elf_hashes);
    // [step] and so are the MME verify rows (`m`), the opt-in form that may leave K=0
    let gdn_mme = vmme::enabled(elf_hashes);
    let fam = if gdn_mme {
        format!("{fam}m")
    } else if gdn_replay {
        format!("{fam}r")
    } else {
        fam.to_string()
    };
    let mut vm_layers: Vec<vmme::VmLayer> = Vec::new();
    // [step] checkpoint form with the vector-scalar row kernel (same family name: same graph
    // but for the kernel, which the fingerprint records)
    let gdn_fast = elf_hashes.iter().any(|(k, _)| k == GDNV_ELF);
    let mut gb = GraphBuilder::new(format!("g_spec_{kind}{fam}_T{t}_C{nch}"));
    // The checkpointing DeltaNet layer builder below does not go through the K=0 layer builders
    // that select the weight precision, so select it for the whole step graph here.
    gb.set_fp8_weights(spec.fp8_weights);
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }

    let table = gb.get_or_add_persistent_tensor(
        "embed_table",
        SynDataType::Bf16,
        &[spec.hidden, spec.vocab],
    );
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[t]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let slot = if spec.n_linear_layers() > 0 {
        Some(gb.add_persistent_tensor("a_in", SynDataType::Int32, &[1]))
    } else {
        None
    };

    let mut x = gb.add_persistent_tensor("x0", SynDataType::Bf16, &[spec.hidden, t]);
    gb.op_gather_bf16("embed", table, ids, x, 1);

    let mut h_pre: Option<TensorId> = None;
    let mut layer_out: Vec<TensorId> = Vec::with_capacity(spec.n_layers as usize);
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
            append_attn_layer(
                &mut gb, spec, x, h_pre, cursor, rope_c, rope_s, &lpfx, t, nch, true, wnext,
            )
        } else if gdn_mme {
            vmme::append_linear_layer_vmme(
                &mut gb,
                spec,
                x,
                h_pre,
                slot.expect("a_in declared for hybrid models"),
                &lpfx,
                t,
                ns,
                wnext,
                &mut vm_layers,
            )
        } else if gdn_replay {
            replay::append_linear_layer_replay(
                &mut gb,
                spec,
                x,
                h_pre,
                slot.expect("a_in declared for hybrid models"),
                cursor,
                &lpfx,
                t,
                ns,
                wnext,
            )
        } else {
            append_linear_layer_ckpt(
                &mut gb,
                spec,
                x,
                h_pre,
                slot.expect("a_in declared for hybrid models"),
                &lpfx,
                t,
                ns,
                wnext,
                gdn_fast,
            )
        };
        x = out_x;
        h_pre = next_hid;
        layer_out.push(out_x);
    }

    let h_final = h_pre.expect("fused final norm produced by last layer");
    let lm_w = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    let logits = add_linear_bf16(&mut gb, h_final, lm_w, spec.hidden, spec.vocab, t, "logits");
    let tokens = if sampled {
        let at = crate::engine::sampler::SamplerAt {
            base: cursor,
            off: 1,
            role: 0,
        };
        crate::engine::sampler::append_sampler(&mut gb, logits, spec.vocab, t, &at, "tokens")
    } else {
        let tokens = gb.add_persistent_tensor("tokens", SynDataType::Int32, &[1, t]);
        gb.op_argmax_bf16("argmax", logits, tokens, 0);
        tokens
    };

    // chain tables: parent[t] = t - 1, depth[t] = t (bound to the shared iota buffers)
    let parent = gb.get_or_add_persistent_tensor("sp_parent", SynDataType::Int32, &[t]);
    let depth = gb.get_or_add_persistent_tensor("sp_depth", SynDataType::Int32, &[t]);
    let a_out = gb.add_persistent_tensor("a_out", SynDataType::Int32, &[1]);
    let cursor_out = gb.add_persistent_tensor("cursor_out", SynDataType::Int32, &[1]);
    let emitted = gb.add_persistent_tensor("emitted", SynDataType::Int32, &[t]);
    let fix_out = gb.add_persistent_tensor("fix_out", SynDataType::Int32, &[2]);
    gb.add_node(
        "gaudium_accept_t",
        "accept",
        &[ids, tokens, cursor, parent, depth],
        &[a_out, cursor_out, emitted, fix_out],
        &[],
    );
    // [step] MME verify rows: the step's one state update per DeltaNet layer
    if gdn_mme {
        vmme::append_commits(&mut gb, spec, &vm_layers, a_out, t);
    }

    if let Some(dc) = drafter {
        let taps: Vec<TensorId> = dc.taps.iter().map(|&l| layer_out[l]).collect();
        dspine::append_step_drafter(
            &mut gb, spec, dc, &taps, t, cursor, depth, tokens, a_out, cursor_out, lmax, sampled,
        );
    }
    gb
}

/// The step graph of a runtime's device drafter: DFlash2 extends the verify-only graph itself
/// (`dflash2::build_step_graph`); every other case is [`build_spec_step_graph`].
#[allow(clippy::too_many_arguments)]
fn build_step_graph_for(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    t: u64,
    ns: u64,
    drafter: Option<&dspine::DSpineCfg>,
    dflash2: Option<&dflash2::DFlash2Cfg>,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    match dflash2 {
        Some(dc) => dflash2::build_step_graph_as(spec, dc, lmax, nch, t, ns, elf_hashes, sampled),
        None => build_spec_step_graph_as(spec, lmax, nch, t, ns, drafter, elf_hashes, sampled),
    }
}

// ============================================================================
// Runtime
// ============================================================================

struct StepRecipe {
    recipe: CompiledRecipe,
    launches: [PreparedLaunch; 2],
}

/// Per-request speculative decode statistics.
#[derive(Debug, Clone, Default)]
pub struct SpecStats {
    pub launches: u64,
    pub emitted: u64,
    /// `hist[a]` = launches that accepted exactly `a` drafts.
    pub accept_hist: Vec<u64>,
    /// `verified[i]` = launches whose verify covered draft position `i`.
    pub verified: Vec<u64>,
    pub widths: BTreeMap<u64, u64>,
    /// Host wall time split over the request: `[launch enqueue, wait for the step, read back]` ms.
    pub host_ms: [f64; 3],
    /// Time spent after the last token waiting for a step that was enqueued ahead of the stop.
    pub drain_ms: f64,
}

impl SpecStats {
    /// `P(draft i accepted | draft i verified)` per draft position.
    pub fn accept_per_pos(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for (i, &n) in self.verified.iter().enumerate() {
            if n == 0 {
                break;
            }
            let hits: u64 = self.accept_hist.iter().skip(i + 1).sum();
            out.push(hits as f64 / n as f64);
        }
        out
    }
}

/// Speculative decoding state of a loaded engine: step recipes, the drafter, and the HBM it owns.
///
/// Built in three phases around the engine's allocation seal: [`SpecRuntime::prepare`] (graphs,
/// eager recipes, exact byte needs), [`SpecRuntime::allocate`] (arenas, before the seal) and
/// [`SpecRuntime::finish`] (uploads and launch descriptors, after the seal).
pub struct SpecRuntime {
    cfg: SpecConfig,
    /// Verify widths in rows (`K + 1`), widest first; always ends with the 1-row fallback.
    widths: Vec<u64>,
    ns: u64,
    lmax: u64,
    elf_hashes: Vec<(String, String)>,
    drafter: Option<dspine::DSpine>,
    /// The DFlash2 drafter (`SpecMethod::DFlash2`); at most one device drafter is loaded.
    dflash2: Option<dflash2::DFlash2>,
    steps: HashMap<(u64, u64), StepRecipe>,
    pending_steps: HashMap<(u64, u64), CompiledRecipe>,
    /// Step recipes of sampled requests (the on-card sampler in the verify); empty maps and
    /// `sampled_steps == false` when this runtime cannot carry them.
    ssteps: HashMap<(u64, u64), StepRecipe>,
    pending_ssteps: HashMap<(u64, u64), CompiledRecipe>,
    sampled_steps: bool,
    /// `elf_hashes` plus the sampler kernels (and the coupled walk): the sampled family's.
    samp_elf_hashes: Vec<(String, String)>,
    weight_specs: BTreeMap<String, u64>,
    kv_specs: BTreeMap<String, u64>,
    act_specs: BTreeMap<String, u64>,
    addr: BTreeMap<String, u64>,
    kv_addr: Vec<(u64, u64)>,
    max_ws: u64,
    request_idx: usize,
    verbose: bool,
    pub last_stats: SpecStats,
    // --- mtptree ---
    mtp: Option<mtp::Mtp>,
    tree: Option<tree_step::TreeRt>,
    /// Prompt of the running request (the MTP prefill feeds each row its next token).
    prompt: Vec<i32>,
    /// [step] the step families run the DeltaNet verify rows in the replay form.
    gdn_replay: bool,
    /// [step] MME verify rows: the per-request conversion of the prefill's state blobs.
    vmconv: Option<vmme::VmConv>,
    /// [step] Launch times of the verify widths measured so far, `[greedy, sampled]` family:
    /// what a request's width policy starts from.
    width_cost: [TreeCost; 2],
}

fn align256(n: u64) -> u64 {
    align_up(n.max(256), 256)
}

/// Step-recipe key of an MTP chain step that verifies `t` rows and drafts `kd` tokens for the
/// next step. The MTP drafts of a step are verified by the NEXT step, so with several verify
/// widths a step's draft count is chosen apart from its own rows. `kd == t - 1` (the fixed-width
/// step) is the plain row count, as for every other method.
fn mtp_step_key(t: u64, kd: u64) -> u64 {
    if kd + 1 == t {
        t
    } else {
        t | ((kd + 1) << 8)
    }
}

/// Drafts an MTP step that verifies `k` drafts produces when the width policy wants `want` next:
/// `want` itself when it is not wider than `k`, else the next listed width above `k` (a wider
/// choice is reached one listed width per step). So the step recipes of a width set are its
/// fixed-width steps plus one "widen" step per width, all compiled when the engine loads.
/// `ks` = the listed widths; `k == 0` is the 1-row fallback, which drafts nothing.
fn mtp_next_drafts(ks: &[u64], k: u64, want: u64) -> u64 {
    if k == 0 {
        0
    } else if want <= k {
        want
    } else {
        ks.iter()
            .copied()
            .filter(|&w| w > k && w <= want)
            .min()
            .unwrap_or(want)
    }
}

/// `(verify rows, drafts for the next step)` of a step-recipe key ([`mtp_step_key`]).
fn mtp_key_parts(key: u64) -> (u64, u64) {
    let t = key & 0xff;
    match key >> 8 {
        0 => (t, t.saturating_sub(1)),
        e => (t, e - 1),
    }
}

/// Step tensors bound by step parity or to engine-owned buffers rather than by their own name.
fn is_parity_tensor(name: &str) -> bool {
    matches!(
        name,
        "ids" | "next_ids" | "cursor" | "cursor_out" | "a_in" | "a_out" | "emitted"
    ) || name.ends_with("_S_in")
        || name.ends_with("_S_ckpt")
        || name.ends_with("_cs_in")
        || name.ends_with("_cs_ckpt")
        || replay::is_parity_name(name)
        || vmme::is_parity_name(name)
}

impl SpecRuntime {
    pub fn gdn_slots(&self) -> u64 {
        self.ns
    }

    /// Slots of the DeltaNet recurrent-state buffers: one per verify row in the checkpoint form,
    /// a single state in the replay form (the conv-state rings keep [`Self::gdn_slots`]).
    pub fn gdn_state_slots(&self) -> u64 {
        if self.gdn_replay || self.vmconv.is_some() {
            1
        } else {
            self.ns
        }
    }

    pub fn max_ws(&self) -> u64 {
        self.max_ws
    }

    pub fn method_label(&self, temperature: f32) -> String {
        let ks: Vec<String> = self
            .widths
            .iter()
            .filter(|&&t| t > 1)
            .map(|t| (t - 1).to_string())
            .collect();
        let mode = if temperature > 0.0 {
            format!("T={temperature:.2}")
        } else {
            "greedy".to_string()
        };
        if let Some(tr) = &self.tree {
            let dv = self
                .mtp
                .as_ref()
                .filter(|m| m.nvd > 0)
                .map_or(String::new(), |m| format!(" draft-vocab {}", m.nvd));
            return format!(
                "{} {} BF16 {mode}{dv}",
                self.cfg.method_name(),
                tr.shape.name
            );
        }
        if let Some(m) = self.mtp.as_ref().filter(|m| m.nvd > 0) {
            return format!(
                "{} K={} BF16 {mode} draft-vocab {}",
                self.cfg.method_name(),
                ks.join(","),
                m.nvd
            );
        }
        // [step] the opt-in form that may leave K=0 is named in every report of a run
        let form = if self.vmconv.is_some() {
            " mme-verify"
        } else {
            ""
        };
        format!(
            "{} K={} BF16 {mode}{form}",
            self.cfg.method_name(),
            ks.join(",")
        )
    }

    /// The step graph of `t` rows over `nch` chunks for the MTP and tree methods (None: the
    /// chain graph of `build_spec_step_graph`).
    fn own_step_graph(
        &self,
        spec: &ModelSpec,
        nch: u64,
        t: u64,
        sampled: bool,
    ) -> Option<GraphBuilder> {
        // an MTP chain step's key may carry its draft count ([`mtp_step_key`])
        let (t, kd) = mtp_key_parts(t);
        if sampled {
            // tree steps are greedy-only; a sampled request of an MTP chain runs its own family
            return self.mtp.as_ref().filter(|_| self.tree.is_none()).map(|m| {
                mtp::build_chain_step_graph_kd(
                    spec,
                    self.lmax,
                    nch,
                    t,
                    kd,
                    self.ns,
                    m.nvd,
                    &self.samp_elf_hashes,
                    true,
                )
            });
        }
        if let Some(tr) = &self.tree {
            return Some(tree_step::build_step_graph(
                spec,
                self.lmax,
                nch,
                tr.shape_of(t),
                self.ns,
                self.mtp.as_ref().filter(|_| tr.mtp).map(|m| m.nvd),
                &self.elf_hashes,
            ));
        }
        self.mtp.as_ref().map(|m| {
            mtp::build_chain_step_graph_kd(
                spec,
                self.lmax,
                nch,
                t,
                kd,
                self.ns,
                m.nvd,
                &self.elf_hashes,
                false,
            )
        })
    }

    /// The step graph of `t` rows over `nch` chunks, of the greedy or the sampled family.
    fn family_step_graph(
        &self,
        spec: &ModelSpec,
        nch: u64,
        t: u64,
        dcfg: Option<&dspine::DSpineCfg>,
        df2cfg: Option<&dflash2::DFlash2Cfg>,
        sampled: bool,
    ) -> GraphBuilder {
        let hashes = if sampled {
            &self.samp_elf_hashes
        } else {
            &self.elf_hashes
        };
        self.own_step_graph(spec, nch, t, sampled)
            .unwrap_or_else(|| {
                build_step_graph_for(
                    spec, self.lmax, nch, t, self.ns, dcfg, df2cfg, hashes, sampled,
                )
            })
    }

    /// The step graph `ensure_step_as` compiles on first use of a key (device-less
    /// pre-compilation of a context's later chunk counts).
    pub(super) fn lazy_step_graph(
        &self,
        spec: &ModelSpec,
        nch: u64,
        key: u64,
        sampled: bool,
    ) -> GraphBuilder {
        let dcfg = self.drafter.as_ref().map(|d| &d.cfg);
        let df2cfg = self.dflash2.as_ref().map(|d| &d.cfg);
        self.family_step_graph(spec, nch, key, dcfg, df2cfg, sampled)
    }

    /// The drafter's own recipes of KV chunk count `nch` that a request compiles on first use
    /// (device-less pre-compilation). Empty for drafters whose recipes do not depend on it here.
    pub(super) fn lazy_drafter_graphs(
        &self,
        spec: &ModelSpec,
        prefill_tile: u64,
        nch: u64,
    ) -> Vec<GraphBuilder> {
        match &self.mtp {
            Some(m) => m.lazy_graphs(spec, self.lmax, prefill_tile, nch, &self.elf_hashes),
            None => Vec::new(),
        }
    }

    /// Keys of the step recipes a load compiled (greedy family) and whether the sampled family
    /// exists.
    pub(super) fn loaded_step_keys(&self) -> (Vec<u64>, bool) {
        let mut keys: Vec<u64> = self.pending_steps.keys().map(|k| k.0).collect();
        keys.sort_unstable();
        keys.dedup();
        (keys, self.sampled_steps)
    }

    /// Sampled requests run the speculative loop when this runtime has the sampled step family
    /// (a device drafter on chain steps, the sampler kernels present); with the host oracle or a
    /// draft tree they keep the non-speculative loop.
    pub fn handles_sampled(&self) -> bool {
        self.sampled_steps
    }

    /// `(address, bytes)` of every drafter context cache (valid after [`Self::allocate`]).
    pub(super) fn cache_regions(&self) -> Vec<(u64, u64)> {
        self.kv_addr.clone()
    }

    /// Row layout of the context cache `name`: the block drafters keep `{128, rows, 8}` bf16
    /// head-major caches (256 bytes per row), the MTP head a chunk-major one like the trunk's.
    fn cache_layout(&self, name: &str, bytes: u64, max_chunks: u64) -> request_reset::CacheLayout {
        if name.starts_with('D') {
            let heads = 8;
            request_reset::CacheLayout::HeadMajor {
                heads,
                rows: bytes / (heads * 256),
            }
        } else {
            request_reset::CacheLayout::ChunkMajor { chunks: max_chunks }
        }
    }

    /// The buffers behind `planned_bytes`, by name: `[weights, kv, activations]`.
    pub fn planned_maps(&self) -> [&BTreeMap<String, u64>; 3] {
        [&self.weight_specs, &self.kv_specs, &self.act_specs]
    }

    /// `[weights, kv, scratch]` bytes this module adds to the device plan.
    pub fn planned_bytes(&self) -> [u64; 3] {
        let sum = |m: &BTreeMap<String, u64>| m.values().map(|&b| align256(b)).sum::<u64>();
        [
            sum(&self.weight_specs),
            sum(&self.kv_specs),
            sum(&self.act_specs),
        ]
    }

    /// Phase 1 (device open, nothing allocated): validates the configuration, builds the sizing
    /// graphs, compiles the step recipes of the first KV chunk, and records exact byte needs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        cfg: &SpecConfig,
        spec: &ModelSpec,
        kenv: &KernelEnv,
        trunk_elf_hashes: &[(String, String)],
        max_ctx: u64,
        max_chunks: u64,
        prefill_tile: u64,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
        verbose: bool,
        sampler_on: bool,
        compile_all: bool,
        opts: &crate::engine::step_opts::StepOpts,
    ) -> Result<Self, String> {
        if cfg.draft_k.is_empty() {
            return Err("--draft-k needs at least one width".into());
        }
        let mut widths: Vec<u64> = Vec::new();
        for &k in &cfg.draft_k {
            if k == 0 || k + 1 > SPEC_MAX_ROWS {
                return Err(format!(
                    "--draft-k {k}: a speculative step verifies 1..={} drafts",
                    SPEC_MAX_ROWS - 1
                ));
            }
            if !widths.contains(&(k + 1)) {
                widths.push(k + 1);
            }
        }
        widths.sort_unstable_by(|a, b| b.cmp(a));
        widths.push(1);
        let t_max = widths[0];

        let mut extra: Vec<&str> = SPEC_VERIFY_ELFS.to_vec();
        let mut df2: Option<dflash2::DFlash2> = None;
        let drafter = match &cfg.method {
            SpecMethod::Oracle { .. } => None,
            SpecMethod::DFlash2 { draft_model } => {
                let mut d = dflash2::DFlash2::open(draft_model, spec)?;
                d.cfg.nvd = opts.draft_rows(spec.vocab, mtp::DV_TAIL)?;
                if verbose && d.cfg.nvd > 0 {
                    eprintln!(
                        "[gaudium] dflash2 drafter: draft vocabulary {} of {} rows",
                        d.cfg.nvd, spec.vocab
                    );
                }
                if t_max > d.cfg.block {
                    return Err(format!(
                        "--draft-k {}: the DFlash2 checkpoint drafts {} tokens per block",
                        t_max - 1,
                        d.cfg.block - 1
                    ));
                }
                extra.extend_from_slice(dflash2::DFLASH2_ELFS);
                df2 = Some(d);
                None
            }
            SpecMethod::DSpine { draft_model } => {
                let mut d = dspine::DSpine::open(draft_model, spec)?;
                let how = d.configure(spec, draft_model, opts)?;
                // context caches: rings of the sliding window when the ring kernels are there
                let have_ring = dspine::DSPINE_RING_ELFS
                    .iter()
                    .all(|e| kenv.present_elfs.contains_key(*e))
                    && kenv.step_glue_hash().is_some();
                if opts.draft_ring == Some(true) && !have_ring {
                    return Err(format!(
                        "--draft-ring 1: the ring kernels ({}) or the step glue library {} are missing from the kernels directory",
                        dspine::DSPINE_RING_ELFS.join(", "),
                        crate::kernels::STEP_GLUE
                    ));
                }
                d.cfg.plan_context(
                    max_ctx,
                    prefill_tile.max(t_max),
                    opts.draft_ring.unwrap_or(have_ring),
                );
                if verbose {
                    eprintln!("[gaudium] dspine drafter: {how}");
                    if d.cfg.ring > 0 {
                        eprintln!(
                            "[gaudium] dspine drafter: context rings of {} rows (window {}) instead of {} rows per cache",
                            d.cfg.ring, d.cfg.window, max_ctx
                        );
                    }
                }
                if t_max > d.cfg.block {
                    return Err(format!(
                        "--draft-k {}: the DSpine checkpoint drafts {} tokens per block",
                        t_max - 1,
                        d.cfg.block - 1
                    ));
                }
                extra.extend_from_slice(dspine::DSPINE_ELFS);
                if d.cfg.ring > 0 {
                    extra.extend_from_slice(dspine::DSPINE_RING_ELFS);
                }
                Some(d)
            }
            // --- mtptree ---
            SpecMethod::Mtp { tree, .. } => {
                if tree.is_some() && cfg.draft_k.len() != 1 {
                    return Err("--spec mtp --tree: one tree shape per engine".into());
                }
                extra.extend(mtp::mtp_elfs(spec));
                if tree.is_some() {
                    extra.extend_from_slice(mtp::MTP_TREE_ELFS);
                }
                None
            }
            SpecMethod::OracleTree { .. } => None,
            SpecMethod::Assistant { .. } => {
                return Err(
                    "--spec assistant is the Gemma 4 family's drafter; this model family drafts with dspine, dflash2 or mtp"
                        .into(),
                )
            }
        };
        let tree = tree_step::TreeRt::new(&cfg.method, spec)?;
        if tree.is_some() {
            extra.retain(|e| !SPEC_VERIFY_ELFS.contains(e));
            extra.extend(tree_step::tree_elfs(spec));
        }
        let mtp = match &cfg.method {
            SpecMethod::Mtp {
                tree: shape,
                draft_vocab,
                ..
            } => Some(mtp::Mtp::new(
                spec,
                shape.as_ref().map_or(t_max - 1, |s| s.spine() as u64),
                *draft_vocab,
                shape.as_ref(),
            )?),
            _ => None,
        };
        // [step] DeltaNet verify rows without per-row state checkpoints (chain steps only: a
        // draft tree keeps the checkpoint form). The replay kernel and its glue join the kernel
        // list of the step families, which is what selects the form in the graph builder.
        let gdn_replay = opts.gdn_verify == crate::engine::step_opts::GdnVerify::Replay
            && spec.n_linear_layers() > 0
            && tree.is_none();
        if gdn_replay {
            extra.push(replay::GDNR_ELF);
        }
        let gdn_mme = opts.gdn_verify == crate::engine::step_opts::GdnVerify::Mme
            && spec.n_linear_layers() > 0
            && tree.is_none();
        if gdn_mme {
            extra.extend_from_slice(&[vmme::VREC_ELF, vmme::VCOMMIT_ELF, vmme::RELAY_ELF]);
        }
        // checkpoint form: the vector-scalar row kernel when it is installed (or demanded)
        let have_fast = kenv.present_elfs.contains_key(GDNV_ELF) && kenv.step_glue_hash().is_some();
        let gdn_fast = spec.n_linear_layers() > 0
            && tree.is_none()
            && !gdn_replay
            && !gdn_mme
            && opts.gdn_fast_rows.unwrap_or(have_fast);
        if gdn_fast {
            extra.push(GDNV_ELF);
        }
        let mut elf_hashes = trunk_elf_hashes.to_vec();
        let ring_ctx = drafter.as_ref().is_some_and(|d| d.cfg.ring > 0);
        if gdn_replay || gdn_fast || ring_ctx {
            elf_hashes.push(kenv.step_glue_hash().ok_or_else(|| {
                format!(
                    "--gdn-verify replay / --gdn-fast-rows 1 / --draft-ring 1: the step glue library {} is missing from the kernels directory",
                    crate::kernels::STEP_GLUE
                )
            })??);
        }
        for (k, h) in kenv.fingerprint_hashes(&extra)? {
            if !elf_hashes.iter().any(|(n, _)| *n == k) {
                elf_hashes.push((k, h));
            }
        }

        // The sampled step family: chain steps of a device drafter, when the engine carries the
        // on-card sampler. DSpine's sampled recipes walk coupled to the sampler's noise, so they
        // also need that kernel.
        let need_cwalk = drafter.is_some();
        let have_cwalk = kenv
            .present_elfs
            .contains_key(crate::engine::sampler::CWALK_ELF);
        let sampled_steps = sampler_on
            && tree.is_none()
            && (drafter.is_some() || df2.is_some() || mtp.is_some())
            && (!need_cwalk || have_cwalk);
        if verbose && sampler_on && need_cwalk && !have_cwalk {
            eprintln!(
                "[gaudium] spec: kernel {} is missing: sampled requests run non-speculatively",
                crate::engine::sampler::CWALK_ELF
            );
        }
        let mut samp_elf_hashes = elf_hashes.clone();
        if sampled_steps {
            let mut more: Vec<&str> = crate::engine::sampler::SAMPLER_ELFS.to_vec();
            if need_cwalk {
                more.push(crate::engine::sampler::CWALK_ELF);
            }
            for (k, h) in kenv.fingerprint_hashes(&more)? {
                if !samp_elf_hashes.iter().any(|(n, _)| *n == k) {
                    samp_elf_hashes.push((k, h));
                }
            }
        }

        let dcfg = drafter.as_ref().map(|d| &d.cfg);
        let df2cfg = df2.as_ref().map(|d| &d.cfg);
        let ns = t_max;
        let mut rt = Self {
            cfg: cfg.clone(),
            widths,
            ns,
            lmax: max_ctx,
            elf_hashes,
            drafter: None,
            dflash2: None,
            steps: HashMap::new(),
            pending_steps: HashMap::new(),
            ssteps: HashMap::new(),
            pending_ssteps: HashMap::new(),
            sampled_steps,
            samp_elf_hashes,
            weight_specs: BTreeMap::new(),
            kv_specs: BTreeMap::new(),
            act_specs: BTreeMap::new(),
            addr: BTreeMap::new(),
            kv_addr: Vec::new(),
            max_ws: 0,
            request_idx: 0,
            verbose,
            last_stats: SpecStats::default(),
            mtp,
            tree,
            prompt: Vec::new(),
            gdn_replay,
            vmconv: None,
            width_cost: [TreeCost::default(), TreeCost::default()],
        };
        if gdn_mme {
            let r = vmme::build_conv_graph(spec, &rt.elf_hashes)
                .compile(Arc::clone(api), cache, driver_version)
                .map_err(|e| format!("compile the DeltaNet state conversion: {e}"))?;
            rt.max_ws = rt.max_ws.max(r.ws_size);
            rt.vmconv = Some(vmme::VmConv {
                recipe: r,
                launches: None,
            });
            if verbose {
                eprintln!(
                    "[gaudium] spec: DeltaNet verify rows on the MME (opt-in: streams may leave K=0 at near-ties of the logits)"
                );
            }
        }
        if gdn_replay {
            for (name, bytes) in replay::buffer_specs(spec, ns) {
                rt.act_specs.insert(name, bytes);
            }
            if verbose {
                eprintln!(
                    "[gaudium] spec: DeltaNet verify rows in the replay form (no per-row state checkpoints)"
                );
            }
        }
        if gdn_fast && verbose {
            eprintln!(
                "[gaudium] spec: DeltaNet verify rows with vector-loaded gate scalars (gaudium_gdn_f32v)"
            );
        }

        // Exact byte needs from the widest graphs (every dimension grows with T and nch).
        let sizing = rt.family_step_graph(spec, max_chunks, t_max, dcfg, df2cfg, false);
        rt.absorb_specs(&sizing);
        // The sampled family binds the same buffers; what it adds (the coupled walk's row
        // scales) is planned from its own widest graph.
        if rt.sampled_steps {
            rt.absorb_specs(&rt.family_step_graph(spec, max_chunks, t_max, dcfg, df2cfg, true));
        }
        if let Some(d) = &df2 {
            rt.absorb_specs(&dflash2::build_ctx_graph(
                spec,
                &d.cfg,
                prefill_tile,
                max_ctx,
                &rt.elf_hashes,
            ));
            rt.absorb_specs(&dflash2::build_block0_graph(
                spec,
                &d.cfg,
                max_ctx,
                &rt.elf_hashes,
            ));
            for i in 0..d.cfg.taps.len() {
                rt.act_specs
                    .insert(format!("df_tap{i}"), spec.hidden * prefill_tile * 2);
            }
        }
        // --- mtptree: the MTP head's own graphs, the tree tables, the path-move pair ---
        if let Some(m) = &rt.mtp {
            for g in m.sizing_graphs(spec, max_ctx, max_chunks, prefill_tile, &rt.elf_hashes) {
                rt.absorb_specs(&g);
            }
            mtp::reclassify(&mut rt.weight_specs, &mut rt.kv_specs, &mut rt.act_specs);
        }
        if rt.tree.is_some() {
            if let Some(g) = rt.own_step_graph(spec, max_chunks, 1, false) {
                rt.absorb_specs(&g);
            }
            tree_step::reclassify(&mut rt.weight_specs, &mut rt.act_specs);
            for name in ["sp_fix0", "sp_fix1"] {
                rt.act_specs.insert(name.into(), 2 * tree_step::FIXN * 4);
            }
        }
        if let Some(d) = &drafter {
            rt.absorb_specs(&dspine::build_ctx_graph(
                spec,
                &d.cfg,
                prefill_tile,
                max_ctx,
                &rt.elf_hashes,
            ));
            rt.absorb_specs(&dspine::build_block0_graph(
                spec,
                &d.cfg,
                max_ctx,
                &rt.elf_hashes,
                false,
            ));
            for i in 0..d.cfg.taps.len() {
                rt.act_specs
                    .insert(format!("df_tap{i}"), spec.hidden * prefill_tile * 2);
            }
        }
        // step-parity pairs, the chain tables, and the host-written first block
        let blk = drafter.as_ref().map_or(t_max, |d| d.cfg.block).max(t_max);
        let blk = df2.as_ref().map_or(blk, |d| d.cfg.block.max(blk));
        for name in ["sp_ids0", "sp_ids1"] {
            rt.act_specs.insert(name.into(), blk * 4);
        }
        for name in ["sp_a0", "sp_a1", "sp_cur0", "sp_cur1"] {
            rt.act_specs.insert(name.into(), 4);
        }
        for name in ["sp_emit0", "sp_emit1"] {
            rt.act_specs.insert(name.into(), SPEC_MAX_ROWS * 4);
        }
        let iota = prefill_tile.max(SPEC_MAX_ROWS).max(blk);
        rt.weight_specs.insert("sp_parent".into(), iota * 4);
        rt.weight_specs.insert("sp_depth".into(), iota * 4);

        // Step recipes: every width at every chunk count, so no step compiles inside a request.
        // A small context compiles them all here (as the engine does for its own step recipes).
        // A large one compiles the first and the last chunk count and takes the ones between
        // from the recipe cache when they are there (`Qwen3Engine::build_cache` fills it, with
        // `compile_all`); one that is not cached compiles on first use.
        const EAGER_ALL_CHUNKS: u64 = 8;
        let t_steps = Instant::now();
        let (mut hits, mut built, mut left) = (0u64, 0u64, 0u64);
        // `(rows, chunk count, sampled family)`: the greedy steps, then the sampled ones.
        let families: &[bool] = if rt.sampled_steps {
            &[false, true]
        } else {
            &[false]
        };
        // An MTP chain with several widths also runs its "widen" steps (`mtp_next_drafts`).
        let mut step_keys: Vec<u64> = rt.widths.clone();
        if rt.mtp.is_some() && rt.tree.is_none() && rt.widths.len() > 2 {
            let ks: Vec<u64> = rt
                .widths
                .iter()
                .filter(|&&t| t > 1)
                .map(|t| t - 1)
                .collect();
            let kmax = ks.iter().copied().max().unwrap_or(0);
            for &k in &ks {
                let kd = mtp_next_drafts(&ks, k, kmax);
                if kd != k {
                    step_keys.push(mtp_step_key(k + 1, kd));
                }
            }
        }
        let mut jobs: Vec<(u64, u64, bool)> = Vec::new();
        for &sampled in families {
            for &t in &step_keys {
                jobs.extend((1..=max_chunks).map(|nch| (t, nch, sampled)));
            }
        }
        // Cached recipes first, on several threads (the chain graphs are plain data; the MTP and
        // tree graphs are described by the runtime itself and are taken one by one).
        let chain = rt.tree.is_none() && rt.mtp.is_none();
        let mut cached: Vec<Option<CompiledRecipe>> = if chain {
            let (hashes, samp_hashes) = (&rt.elf_hashes, &rt.samp_elf_hashes);
            crate::synapse::graph::load_cached_parallel(
                jobs.len(),
                crate::synapse::graph::recipe_load_threads(),
                api,
                cache,
                driver_version,
                |i| {
                    let (t, nch, sampled) = jobs[i];
                    build_step_graph_for(
                        spec,
                        max_ctx,
                        nch,
                        t,
                        ns,
                        dcfg,
                        df2cfg,
                        if sampled { samp_hashes } else { hashes },
                        sampled,
                    )
                },
            )
            .into_iter()
            .zip(&jobs)
            .map(|(r, &(t, nch, _))| r.map_err(|e| format!("load spec step T={t} C={nch}: {e}")))
            .collect::<Result<_, String>>()?
        } else {
            jobs.iter().map(|_| None).collect()
        };
        for (i, &(t, nch, sampled)) in jobs.iter().enumerate() {
            let must =
                compile_all || max_chunks <= EAGER_ALL_CHUNKS || nch == 1 || nch == max_chunks;
            let t0 = Instant::now();
            let r = match cached[i].take() {
                Some(r) => Some(r),
                None if must || !chain => {
                    let gb = rt.family_step_graph(spec, nch, t, dcfg, df2cfg, sampled);
                    if must {
                        Some(
                            gb.compile(Arc::clone(api), cache, driver_version)
                                .map_err(|e| format!("compile spec step T={t} C={nch}: {e}"))?,
                        )
                    } else {
                        gb.load_cached(api, cache, driver_version)
                            .map_err(|e| format!("load spec step T={t} C={nch}: {e}"))?
                    }
                }
                None => None,
            };
            let Some(r) = r else {
                left += 1;
                continue;
            };
            if r.from_cache {
                hits += 1;
            } else {
                built += 1;
            }
            if verbose && (!r.from_cache || nch == 1 || nch == max_chunks) {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                    r.name,
                    r.from_cache,
                    r.ws_size as f64 / (1024.0 * 1024.0),
                    t0.elapsed().as_secs_f64()
                );
            }
            rt.max_ws = rt.max_ws.max(r.ws_size);
            if sampled {
                rt.pending_ssteps.insert((t, nch), r);
            } else {
                rt.pending_steps.insert((t, nch), r);
            }
        }
        if verbose {
            eprintln!(
                "[gaudium] spec step recipes ({} families x {} step shapes x {} chunk counts): {hits} from the cache, {built} compiled, {left} left for first use ({:.2}s)",
                families.len(),
                step_keys.len(),
                max_chunks,
                t_steps.elapsed().as_secs_f64()
            );
        }
        let mut drafter = drafter;
        if let Some(d) = &mut drafter {
            let ws = d.compile_eager(
                spec,
                max_ctx,
                prefill_tile,
                &rt.elf_hashes,
                api,
                cache,
                driver_version,
                verbose,
                rt.sampled_steps.then_some(rt.samp_elf_hashes.as_slice()),
            )?;
            rt.max_ws = rt.max_ws.max(ws);
        }
        rt.drafter = drafter;
        if let Some(d) = &mut df2 {
            let ws = d.compile_eager(
                spec,
                max_ctx,
                prefill_tile,
                &rt.elf_hashes,
                api,
                cache,
                driver_version,
                verbose,
            )?;
            rt.max_ws = rt.max_ws.max(ws);
        }
        rt.dflash2 = df2;
        // --- mtptree ---
        if let Some(mut m) = rt.mtp.take() {
            let ws = m.compile_eager(
                spec,
                max_ctx,
                prefill_tile,
                &rt.elf_hashes,
                api,
                cache,
                driver_version,
                verbose,
            )?;
            rt.max_ws = rt.max_ws.max(ws);
            rt.mtp = Some(m);
        }
        Ok(rt)
    }

    /// Sorts a graph's persistent tensors into this module's weight / KV / scratch needs. Tensors
    /// owned by the engine (trunk weights, trunk KV, the DeltaNet rings) and the parity-bound
    /// names are skipped.
    fn absorb_specs(&mut self, gb: &GraphBuilder) {
        for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
            if is_parity_tensor(&name)
                || is_weight_or_table(&name)
                || crate::engine::sampler::is_sampler_param(&name)
                || (is_kv_tensor(&name) && name.starts_with('L'))
            {
                continue;
            }
            let map = if dspine::is_drafter_kv(&name) {
                &mut self.kv_specs
            } else if dspine::is_drafter_weight(&name)
                || dflash2::is_drafter_weight(&name)
                || name == dflash2::DV_HEAD
                || name == dflash2::DV_IDS
                || name == "sp_parent"
                || name == "sp_depth"
            {
                &mut self.weight_specs
            } else {
                &mut self.act_specs
            };
            let e = map.entry(name).or_insert(0);
            *e = (*e).max(bytes);
        }
    }

    /// Phase 2 (before the allocation seal): allocates exactly the planned arenas.
    pub(super) fn allocate(&mut self, dev: &mut HpuDevice) -> Result<(), String> {
        let [wb, kb, ab] = self.planned_bytes();
        let mut place = |specs: &BTreeMap<String, u64>,
                         bytes: u64,
                         cat: HbmCat,
                         label: &str,
                         addr: &mut BTreeMap<String, u64>|
         -> Result<(), String> {
            if bytes == 0 {
                return Ok(());
            }
            let mut arena = dev
                .alloc_arena(bytes, cat, label)
                .map_err(|e| format!("alloc {label}: {e}"))?;
            for (name, &b) in specs {
                let (a, _off) = arena
                    .suballoc(b.max(256), 256)
                    .map_err(|e| format!("{label} suballoc '{name}': {e}"))?;
                addr.insert(name.clone(), a);
            }
            Ok(())
        };
        let mut addr = BTreeMap::new();
        place(
            &self.weight_specs,
            wb,
            HbmCat::CatWeights,
            "spec_weights_arena",
            &mut addr,
        )?;
        place(
            &self.kv_specs,
            kb,
            HbmCat::CatKv,
            "spec_kv_arena",
            &mut addr,
        )?;
        place(
            &self.act_specs,
            ab,
            HbmCat::CatScratch,
            "spec_act_arena",
            &mut addr,
        )?;
        self.kv_addr = self.kv_specs.iter().map(|(n, &b)| (addr[n], b)).collect();
        self.addr = addr;
        Ok(())
    }

    /// Binding pool of a step launch for step parity `p` (reads buffers `p`, writes `1 - p`).
    fn step_pool(
        &self,
        p: usize,
        weights_addr: &BTreeMap<String, u64>,
        kv_addr: &BTreeMap<String, (u64, u64)>,
        gdn_state: &BTreeMap<usize, GdnLayerState>,
    ) -> BTreeMap<String, u64> {
        let mut pool: BTreeMap<String, u64> = weights_addr.clone();
        for (k, &(v, _)) in kv_addr {
            pool.insert(k.clone(), v);
        }
        for (k, &v) in &self.addr {
            pool.insert(k.clone(), v);
        }
        let q = 1 - p;
        pool.insert("ids".into(), self.addr[&format!("sp_ids{p}")]);
        pool.insert("next_ids".into(), self.addr[&format!("sp_ids{q}")]);
        pool.insert("cursor".into(), self.addr[&format!("sp_cur{p}")]);
        pool.insert("cursor_out".into(), self.addr[&format!("sp_cur{q}")]);
        pool.insert("a_in".into(), self.addr[&format!("sp_a{p}")]);
        pool.insert("a_out".into(), self.addr[&format!("sp_a{q}")]);
        pool.insert("emitted".into(), self.addr[&format!("sp_emit{p}")]);
        // --- mtptree: the tree steps' pending path move alternates like the other step state ---
        if let (Some(&fi), Some(&fo)) = (
            self.addr.get(&format!("sp_fix{p}")),
            self.addr.get(&format!("sp_fix{q}")),
        ) {
            pool.insert("tfix_in".into(), fi);
            pool.insert("tfix_out".into(), fo);
        }
        for (&l, st) in gdn_state {
            pool.insert(format!("L{l}_S_in"), st.s[p]);
            pool.insert(format!("L{l}_S_ckpt"), st.s[q]);
            pool.insert(format!("L{l}_cs_in"), st.cs[p]);
            pool.insert(format!("L{l}_cs_ckpt"), st.cs[q]);
        }
        if self.gdn_replay {
            replay::bind(&mut pool, &self.addr, gdn_state, p);
        }
        if self.vmconv.is_some() {
            // The request's conversion wrote the matrix states into the buffer the prefill did
            // NOT leave its blobs in, so the matrix states alternate opposite to the step
            // parity (which the conv-state rings and the step scalars keep following).
            for (&l, st) in gdn_state {
                pool.insert(format!("L{l}_mS_in"), st.s[q]);
                pool.insert(format!("L{l}_mS_out"), st.s[p]);
            }
        }
        pool
    }

    /// Binds a compiled step recipe (greedy or sampled family) and prepares its two parity
    /// launches. `pools` = the binding pools of step parity 0 and 1 (`step_pool`): the same for
    /// every step recipe, so a caller installing many builds them once.
    fn install_step(
        &mut self,
        sampled: bool,
        key: (u64, u64),
        mut recipe: CompiledRecipe,
        dev: &HpuDevice,
        pools: &[BTreeMap<String, u64>; 2],
    ) -> Result<(), String> {
        recipe
            .bind_shared_workspace(dev)
            .map_err(|e| format!("bind {} ws: {e}", recipe.name))?;
        let prep = |p: usize| -> Result<PreparedLaunch, String> {
            let pool = &pools[p];
            let exact = select_exact_bindings(&recipe, pool)?;
            recipe
                .prepare_launch(&exact)
                .map_err(|e| format!("prepare {} (parity {p}): {e}", recipe.name))
        };
        let launches = [prep(0)?, prep(1)?];
        let map = if sampled {
            &mut self.ssteps
        } else {
            &mut self.steps
        };
        map.insert(key, StepRecipe { recipe, launches });
        Ok(())
    }

    /// Phase 3 (after the seal and the trunk upload): binds workspaces, uploads the tables and the
    /// drafter, and pre-validates the launches of the eager recipes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish(
        &mut self,
        dev: &HpuDevice,
        spec: &ModelSpec,
        packer: &WeightPacker<'_>,
        stage: &mut PinnedBuf,
        cache_dir: Option<&std::path::Path>,
        weights_addr: &BTreeMap<String, u64>,
        kv_addr: &BTreeMap<String, (u64, u64)>,
        gdn_state: &BTreeMap<usize, GdnLayerState>,
        pf_cursor: u64,
    ) -> Result<(), String> {
        let n = (self.weight_specs["sp_depth"] / 4) as i32;
        let depth: Vec<u8> = (0..n).flat_map(|i| i.to_le_bytes()).collect();
        let parent: Vec<u8> = (0..n).flat_map(|i| (i - 1).to_le_bytes()).collect();
        dev.upload_h2d_chunked(&depth, self.addr["sp_depth"], stage)
            .map_err(|e| format!("upload sp_depth: {e}"))?;
        dev.upload_h2d_chunked(&parent, self.addr["sp_parent"], stage)
            .map_err(|e| format!("upload sp_parent: {e}"))?;

        if let Some(mut d) = self.drafter.take() {
            let mut pool: BTreeMap<String, u64> = weights_addr.clone();
            for (k, &v) in &self.addr {
                pool.insert(k.clone(), v);
            }
            pool.insert("ctx_cursor".into(), pf_cursor);
            d.finish(
                dev,
                spec,
                packer,
                stage,
                cache_dir,
                &self.weight_specs,
                &self.addr,
                &pool,
                self.lmax,
                self.verbose,
            )?;
            self.drafter = Some(d);
        }
        if let Some(mut d) = self.dflash2.take() {
            let mut pool: BTreeMap<String, u64> = weights_addr.clone();
            for (k, &v) in &self.addr {
                pool.insert(k.clone(), v);
            }
            pool.insert("ctx_cursor".into(), pf_cursor);
            d.upload_draft_vocab(dev, spec, packer, stage, &self.addr)?;
            d.finish(
                dev,
                spec,
                stage,
                &self.weight_specs,
                &self.addr,
                &pool,
                self.lmax,
                self.verbose,
            )?;
            self.dflash2 = Some(d);
        }

        // --- mtptree: shape tables and the MTP head ---
        if let Some(tr) = &self.tree {
            for (name, vals) in tr.tables() {
                let Some(&a) = self.addr.get(&name) else {
                    continue;
                };
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                dev.upload_h2d_chunked(&bytes, a, stage)
                    .map_err(|e| format!("upload {name}: {e}"))?;
            }
        }
        if let Some(mut m) = self.mtp.take() {
            let mut pool: BTreeMap<String, u64> = weights_addr.clone();
            for (k, &(v, _)) in kv_addr {
                pool.insert(k.clone(), v);
            }
            for (k, &v) in &self.addr {
                pool.insert(k.clone(), v);
            }
            pool.insert("ctx_cursor".into(), pf_cursor);
            m.finish(
                dev,
                spec,
                packer,
                stage,
                &self.weight_specs,
                &self.addr,
                pool,
                self.verbose,
            )?;
            self.mtp = Some(m);
        }

        if let Some(vc) = self.vmconv.as_mut() {
            vc.finish(dev, gdn_state)?;
        }
        let pending: Vec<((u64, u64), CompiledRecipe)> = self.pending_steps.drain().collect();
        let pools = [
            self.step_pool(0, weights_addr, kv_addr, gdn_state),
            self.step_pool(1, weights_addr, kv_addr, gdn_state),
        ];
        for (key, r) in pending {
            self.install_step(false, key, r, dev, &pools)?;
        }
        let pending: Vec<((u64, u64), CompiledRecipe)> = self.pending_ssteps.drain().collect();
        for (key, r) in pending {
            self.install_step(true, key, r, dev, &pools)?;
        }
        Ok(())
    }

    /// Drafter tap rows a prefill tile carries when two tiles are in flight (the wavefront keeps
    /// a second set of them); `None`: this drafter's prefill hook reads buffers the two-tile
    /// loop does not duplicate, so its prompts stay with the single-tile loop.
    pub(super) fn wave_taps(&self) -> Option<usize> {
        if self.mtp.is_some() || self.dflash2.is_some() {
            return None;
        }
        Some(self.drafter.as_ref().map_or(0, |d| d.cfg.taps.len()))
    }

    /// [`Self::prefill_io`] for a tile on the wavefront's second buffer set: `taps` are that
    /// set's tap rows, in tap order.
    pub(super) fn prefill_io_set1(&self, l: usize, xin: u64, xout: u64, taps: &[u64]) -> (u64, u64) {
        let Some(d) = &self.drafter else {
            return (xin, xout);
        };
        let tap = |layer: usize| {
            d.cfg
                .taps
                .iter()
                .position(|&t| t == layer)
                .and_then(|i| taps.get(i).copied())
        };
        let xin = if l > 0 {
            tap(l - 1).unwrap_or(xin)
        } else {
            xin
        };
        (xin, tap(l).unwrap_or(xout))
    }

    /// Prefill routing for the drafter's target-layer taps: a tapped layer writes its output to
    /// the tap buffer and the next layer reads it from there.
    pub(super) fn prefill_io(&self, l: usize, xin: u64, xout: u64) -> (u64, u64) {
        if let Some(d) = &self.dflash2 {
            return d.prefill_io(&self.addr, l, xin, xout);
        }
        let Some(d) = &self.drafter else {
            return (xin, xout);
        };
        let tap = |layer: usize| {
            d.cfg
                .taps
                .iter()
                .position(|&t| t == layer)
                .map(|i| self.addr[&format!("df_tap{i}")])
        };
        let xin = if l > 0 {
            tap(l - 1).unwrap_or(xin)
        } else {
            xin
        };
        (xin, tap(l).unwrap_or(xout))
    }
}

// ============================================================================
// Engine hooks (the engine's private fields are visible here)
// ============================================================================

/// After the trunk layers of one prefill tile: the drafter's context K/V for the tile's rows.
pub(super) fn prefill_tile_hook(
    eng: &mut Qwen3Engine,
    pos: u64,
    tile_len: u64,
) -> Result<(), String> {
    // --- mtptree ---
    if eng.specdec.as_ref().is_some_and(|rt| rt.mtp.is_some()) {
        return mtp::prefill_tile(eng, pos, tile_len);
    }
    let Qwen3Engine {
        specdec,
        dev,
        spec,
        cache,
        driver_version,
        ..
    } = eng;
    let Some(rt) = specdec.as_mut() else {
        return Ok(());
    };
    if let Some(d) = rt.dflash2.as_mut() {
        return d.launch_ctx(
            dev,
            spec,
            cache,
            driver_version,
            &rt.elf_hashes,
            rt.lmax,
            pos,
            tile_len,
        );
    }
    let Some(d) = rt.drafter.as_mut() else {
        return Ok(());
    };
    d.launch_ctx(
        dev,
        spec,
        cache,
        driver_version,
        &rt.elf_hashes,
        rt.lmax,
        pos,
        tile_len,
    )
}

/// [`prefill_tile_hook`] for a tile on the wavefront's second buffer set: the drafter's context
/// pass reads that set's tap rows and cursor (`over` = tensor name -> address).
pub(super) fn prefill_tile_hook_set1(
    eng: &mut Qwen3Engine,
    pos: u64,
    tile_len: u64,
    over: &BTreeMap<String, u64>,
) -> Result<(), String> {
    let Qwen3Engine {
        specdec,
        dev,
        spec,
        cache,
        driver_version,
        ..
    } = eng;
    let Some(rt) = specdec.as_mut() else {
        return Ok(());
    };
    if rt.mtp.is_some() || rt.dflash2.is_some() {
        return Err("the wavefront has no second buffer set for this drafter".into());
    }
    let Some(d) = rt.drafter.as_mut() else {
        return Ok(());
    };
    d.launch_ctx_over(
        dev,
        spec,
        cache,
        driver_version,
        &rt.elf_hashes,
        rt.lmax,
        pos,
        tile_len,
        Some(over),
    )
}

/// Start of a request: the prompt, for drafters whose prefill needs more than the tile's rows.
pub(super) fn begin_request(eng: &mut Qwen3Engine, prompt_ids: &[i32]) {
    if let Some(rt) = eng.specdec.as_mut() {
        if rt.mtp.is_some() {
            rt.prompt.clear();
            rt.prompt.extend_from_slice(prompt_ids);
        }
    }
}

/// The drafter's context caches as `(row layout, address, bytes)`, for the request reset.
pub(super) fn request_caches(eng: &Qwen3Engine) -> Vec<(request_reset::CacheLayout, u64, u64)> {
    let Some(rt) = eng.specdec.as_ref() else {
        return Vec::new();
    };
    rt.kv_specs
        .keys()
        .zip(&rt.kv_addr)
        .map(|(name, &(addr, bytes))| (rt.cache_layout(name, bytes, eng.max_chunks), addr, bytes))
        .collect()
}

fn ensure_step(eng: &mut Qwen3Engine, t: u64, nch: u64) -> Result<(), String> {
    ensure_step_as(eng, false, t, nch)
}

/// Makes the step of `t` rows over `nch` chunks launchable (greedy or sampled family); a first
/// use of a chunk count that was not compiled at load compiles into the shared workspace.
fn ensure_step_as(eng: &mut Qwen3Engine, sampled: bool, t: u64, nch: u64) -> Result<(), String> {
    let rt = eng.specdec.as_ref().expect("spec runtime");
    let have = if sampled { &rt.ssteps } else { &rt.steps };
    if have.contains_key(&(t, nch)) {
        return Ok(());
    }
    let t0 = Instant::now();
    let dcfg = rt.drafter.as_ref().map(|d| &d.cfg);
    let df2cfg = rt.dflash2.as_ref().map(|d| &d.cfg);
    let gb = rt.family_step_graph(&eng.spec, nch, t, dcfg, df2cfg, sampled);
    let recipe = gb
        .compile(Arc::clone(eng.dev.api()), &eng.cache, &eng.driver_version)
        .map_err(|e| format!("compile spec step T={t} C={nch}: {e}"))?;
    let (name, from_cache) = (recipe.name.clone(), recipe.from_cache);
    let Qwen3Engine {
        specdec,
        dev,
        weights_addr,
        kv_addr,
        gdn_state,
        ..
    } = eng;
    let rt = specdec.as_mut().expect("spec runtime");
    let pools = [
        rt.step_pool(0, weights_addr, kv_addr, gdn_state),
        rt.step_pool(1, weights_addr, kv_addr, gdn_state),
    ];
    rt.install_step(sampled, (t, nch), recipe, dev, &pools)?;
    if eng.verbose {
        eprintln!(
            "[gaudium] recipe {name} was not ready: the request waited {:.1} ms for it ({})",
            t0.elapsed().as_secs_f64() * 1000.0,
            if from_cache {
                "loaded from the recipe cache"
            } else {
                "compiled"
            }
        );
    }
    Ok(())
}

fn write_i32s(eng: &mut Qwen3Engine, writes: &[(u64, &[i32])]) -> Result<(), String> {
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

fn read_i32s(eng: &mut Qwen3Engine, addr: u64, n: usize) -> Result<Vec<i32>, String> {
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
pub(super) struct SpecDecodeOut {
    pub decode_ms: f64,
    pub decode_tokens: usize,
    pub stopped_by_eos: bool,
    pub stopped_by_cancel: bool,
    pub stats: SpecStats,
}

/// Speculative decode phase: continues after the prefill produced `generated[0]` at position
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
    // A sampled request (only routed here when `handles_sampled`) launches the sampled step
    // family; everything else in this loop is the same.
    let sampled = params.temperature > 0.0;
    let check = sampled && params.sample_check;
    // --- mtptree: tree steps have their own loop ---
    if eng.specdec.as_ref().is_some_and(|rt| rt.tree.is_some()) {
        return tree_step::decode(
            eng,
            prompt_len,
            params,
            budget,
            is_eos,
            is_cancelled,
            on_token,
            generated,
            gdn_parity,
        );
    }
    let t_decode = Instant::now();
    let first_tok = *generated.last().expect("prefill produced the first token");
    let n0 = generated.len();
    let mut p = gdn_parity;
    // [step] MME verify rows: the prefill left blobs in state buffer `p`; the steps read the
    // matrix layout, converted into the other buffer (see `step_pool`)
    if let Some(vc) = eng.specdec.as_ref().and_then(|rt| rt.vmconv.as_ref()) {
        vc.launch(&eng.dev, p)?;
    }
    let mut cursor = prompt_len as u64;
    let mut stats = SpecStats::default();
    let mut stopped_by_eos = false;
    let mut stopped_by_cancel = false;

    let (widths, blk, addr_tokens, oracle) = {
        let rt = eng.specdec.as_mut().expect("spec runtime");
        let oracle = match &rt.cfg.method {
            SpecMethod::Oracle { requests } => {
                Some(requests.get(rt.request_idx).cloned().ok_or_else(|| {
                    format!("oracle drafter: no stream for request {}", rt.request_idx)
                })?)
            }
            SpecMethod::DSpine { .. } | SpecMethod::DFlash2 { .. } => None,
            SpecMethod::Mtp { .. } | SpecMethod::OracleTree { .. } => None,
            SpecMethod::Assistant { .. } => None,
        };
        rt.request_idx += 1;
        (
            rt.widths.clone(),
            (rt.act_specs["sp_ids0"] / 4) as usize,
            rt.addr["tokens"],
            oracle,
        )
    };
    let mut ks: Vec<u64> = widths.iter().filter(|&&t| t > 1).map(|t| t - 1).collect();
    // [step] a request may pin one of the engine's widths (width-policy measurements)
    if let Some(k) = params.force_draft_k.filter(|k| ks.contains(k)) {
        ks = vec![k];
    }
    // launch times of the widths measured by earlier requests of this engine (per step family)
    let mut policy = WidthPolicy::with_cost(
        &ks,
        eng.specdec.as_ref().unwrap().width_cost[usize::from(sampled)].clone(),
    );
    let kmax = ks.iter().copied().max().unwrap_or(0) as usize;
    // a request may pin one of the engine's widths (measurement runs)
    let force_k = params.spec_k.filter(|k| ks.contains(k));
    // MTP chain: a step verifies drafts the step before it produced (the first block drafts the
    // widest), and the width chosen at its enqueue is also what IT drafts for the next step
    let is_mtp = eng.specdec.as_ref().is_some_and(|rt| rt.mtp.is_some());
    let mut mtp_avail = kmax as u64;
    stats.accept_hist = vec![0; kmax + 1];
    stats.verified = vec![0; kmax];

    // Oracle drafts for the step whose row 0 is generated token index `n - 1`.
    let oracle_ids = |o: &OracleRequest, gen: &[i32]| -> Vec<i32> {
        let n = gen.len();
        let mut ids = vec![*gen.last().unwrap(); blk];
        for (j, slot) in ids.iter_mut().enumerate().skip(1) {
            let idx = n + j - 1;
            let mut d = o.stream.get(idx).copied().unwrap_or(0);
            if o.corrupt.contains(&idx) {
                d = if d == 0 { 1 } else { d - 1 };
            }
            *slot = d;
        }
        ids
    };

    // Step-parity scalars of the first step, and its ids.
    let (a_addr, cur_addr, ids_addr) = {
        let rt = eng.specdec.as_ref().unwrap();
        (
            rt.addr[&format!("sp_a{p}")],
            rt.addr[&format!("sp_cur{p}")],
            rt.addr[&format!("sp_ids{p}")],
        )
    };
    let first_ids: Vec<i32> = if let Some(o) = &oracle {
        oracle_ids(o, generated)
    } else if eng.specdec.as_ref().is_some_and(|rt| rt.mtp.is_some()) {
        // --- mtptree: the MTP head's first block ---
        mtp::first_block(eng, first_tok, cursor, blk)?
    } else {
        let drafts = {
            let Qwen3Engine {
                specdec,
                dev,
                stage_io,
                ..
            } = eng;
            let rt = specdec.as_mut().unwrap();
            if let Some(d) = rt.dflash2.as_mut() {
                d.launch_block0(dev, stage_io, first_tok, cursor as i32)?
            } else {
                let d = rt.drafter.as_mut().expect("drafter");
                d.launch_block0(dev, stage_io, first_tok, cursor as i32, sampled)?
            }
        };
        if check {
            check_coupled_walk(eng, cursor as u32)?;
        }
        calibrate_drafter(eng)?;
        let mut ids = vec![first_tok];
        ids.extend_from_slice(&drafts);
        ids.resize(blk, 0);
        ids
    };
    // [step] replay form: the first step replays nothing (its cursor equals `sp_rp_first`)
    let cur0_addr = eng
        .specdec
        .as_ref()
        .and_then(|rt| rt.addr.get("sp_rp_first").copied());
    let cur0 = [cursor as i32];
    let mut first_writes: Vec<(u64, &[i32])> =
        vec![(a_addr, &[0]), (cur_addr, &cur0), (ids_addr, &first_ids)];
    if let Some(a) = cur0_addr {
        first_writes.push((a, &cur0));
    }
    write_i32s(eng, &first_writes)?;
    drop(first_writes);
    let mut cur_ids = first_ids;

    // A device drafter keeps every step input on the card, so the next step is enqueued while
    // the current one runs (two launches in flight, the host's enqueue and read-back hidden under
    // device time). A host drafter needs each result before it can write the next ids.
    // The sample check reads each step's logits back before the next step overwrites them.
    // A drafter calibration run reads the block's site buffers after every step.
    let calibrating = eng
        .specdec
        .as_ref()
        .and_then(|rt| rt.drafter.as_ref())
        .is_some_and(|d| d.calibrating());
    let depth: usize = if oracle.is_some() || check || calibrating {
        1
    } else {
        2
    };
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
        k: u64,
        p: usize,
        launched: Instant,
    }
    let mut flights: std::collections::VecDeque<Flight> = std::collections::VecDeque::new();
    let mut last_done = Instant::now();
    // [step] completion-interval sums per verify width (verbose report)
    let mut width_ms: BTreeMap<u64, (f64, u64)> = BTreeMap::new();
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
            // Do not enqueue ahead of a step that may already complete the request's budget:
            // the extra launch would run for nothing and hold the card after the last token.
            if budget.remaining() <= flights.iter().map(|f| f.k + 1).sum::<u64>() {
                break;
            }
            // the cursor of the step to launch lies in [cur_lo, cur_hi] until the flights land
            let cur_lo = cursor + flights.len() as u64;
            let cur_hi = cursor + flights.iter().map(|f| f.k + 1).sum::<u64>();
            let room = eng.max_ctx.saturating_sub(cur_hi);
            if room == 0 {
                break;
            }
            // [step] a device-trace window opens / closes on a drained stream only
            if !tryd!(prof_rt::before_launch(
                eng,
                stats.launches + flights.len() as u64,
                flights.len()
            )) {
                break;
            }
            let want = match force_k {
                Some(fk) if fk < room => fk,
                Some(_) => 0,
                None if ks.len() > 1 => {
                    let pending: Vec<u64> = flights.iter().map(|f| f.k).collect();
                    policy.choose_with_pending(room, &pending)
                }
                None if ks[0] < room => ks[0],
                None => 0,
            };
            // `tk` = the step recipe's key: its rows, or an MTP step's rows and draft count
            let (k, tk, drafts_next) = if is_mtp {
                // verify at most the chosen width of the drafted block (a narrower choice takes
                // effect at once, a wider one a step later); `want == 0` near the end of the
                // context is the 1-row fallback, which drafts nothing
                let k = mtp_avail.min(want);
                let kd = mtp_next_drafts(&ks, k, want);
                (k, mtp_step_key(k + 1, kd), kd)
            } else {
                (want, want + 1, want)
            };
            let t = k + 1;
            let chunks = |cur: u64| (cur + t).div_ceil(KV_CHUNK).clamp(1, eng.max_chunks);
            let nch = chunks(cur_hi);
            if chunks(cur_lo) != nch {
                // the chunk count this step needs depends on a result still in flight
                break;
            }
            tryd!(ensure_step_as(eng, sampled, tk, nch));
            let t_launch = Instant::now();
            {
                let rt = eng.specdec.as_ref().unwrap();
                let step = if sampled {
                    &rt.ssteps[&(tk, nch)]
                } else {
                    &rt.steps[&(tk, nch)]
                };
                // No host launch marks here: with two launches in flight a mark would be paired
                // with the `live.step` of the older launch. The reporter times these steps by
                // their token arrivals and takes their bytes from the armed launch model
                // (`spec::readout::arm`), which a pending mark would switch off.
                tryd!(step
                    .recipe
                    .launch_prepared(&eng.dev, &step.launches[p])
                    .map_err(|e| format!("launch spec step (cursor<={cur_hi}, T={t}): {e}")));
            }
            if let Some(ev) = &events {
                tryd!(ev[p]
                    .record_compute(&eng.dev)
                    .map_err(|e| format!("record step event: {e}")));
            }
            stats.host_ms[0] += t_launch.elapsed().as_secs_f64() * 1000.0;
            flights.push_back(Flight {
                k,
                p,
                launched: t_launch,
            });
            mtp_avail = drafts_next; // what the step just enqueued leaves for the next one
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
        let (k, t) = (f.k, f.k + 1);
        let t_wait = Instant::now();
        match &events {
            Some(ev) => tryd!(ev[f.p]
                .synchronize()
                .map_err(|e| format!("wait spec step (cursor={cursor}): {e}"))),
            None => tryd!(unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
                .map_err(|e| format!("sync spec step (cursor={cursor}): {e}"))),
        }
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
        if oracle.is_some() {
            // test mode: the device accept must equal the host twin on the verify argmaxes
            let tokens = tryd!(read_i32s(eng, addr_tokens, t as usize));
            let want = chain_accept(&cur_ids[..t as usize], &tokens);
            if want != a || emitted[a] != tokens[a] || emitted[..a] != cur_ids[1..=a] {
                return Err(format!(
                    "spec step at cursor {cursor}: device accept {a} / emitted {emitted:?} disagrees with the host twin {want} (ids {:?}, tokens {tokens:?})",
                    &cur_ids[..t as usize]
                ));
            }
        }
        if check {
            // debug: every verify row of this step against the host twin, on the card's logits
            // (row `i` at `cursor` samples sequence index `cursor + i + 1`), then the next
            // block's coupled walk (anchored on the bonus token at `cursor + n_emit`)
            let tokens = tryd!(read_i32s(eng, addr_tokens, t as usize));
            let addr_logits = eng.specdec.as_ref().unwrap().addr["logits"];
            let row_bytes = eng.spec.vocab * 2;
            for (i, &tok) in tokens.iter().enumerate() {
                tryd!(samp_rt::check_row(
                    eng,
                    addr_logits + i as u64 * row_bytes,
                    (cursor + i as u64 + 1) as u32,
                    tok
                ));
            }
            if t > 1 {
                tryd!(check_coupled_walk(eng, (cursor + n_emit as u64) as u32));
            }
        }
        if calibrating && t > 1 {
            tryd!(calibrate_drafter(eng));
        }
        stats.host_ms[2] += t_read.elapsed().as_secs_f64() * 1000.0;
        // steady state: one step per completion interval; the first step counts from its launch
        let now = Instant::now();
        let step_ms = now.duration_since(last_done.max(f.launched)).as_secs_f64() * 1000.0;
        last_done = now;

        stats.launches += 1;
        stats.accept_hist[a] += 1;
        for i in 0..k as usize {
            stats.verified[i] += 1;
        }
        *stats.widths.entry(k).or_insert(0) += 1;
        policy.observe(k, a, step_ms);
        // the first launches of a request start on an idle card and are not steady-state steps
        if stats.launches > 2 {
            let e = width_ms.entry(k).or_insert((0.0, 0));
            e.0 += step_ms;
            e.1 += 1;
        }

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
        eng.live.step(step_emitted, k, a as u64);
        if stopped_by_eos || budget.remaining() == 0 {
            break;
        }

        cursor += n_emit as u64;
        if let Some(o) = &oracle {
            // the next step's row 0 is the bonus token; its drafts are the known continuation
            cur_ids = oracle_ids(o, generated);
            let ids_addr = eng.specdec.as_ref().unwrap().addr[&format!("sp_ids{p}")];
            tryd!(write_i32s(eng, &[(ids_addr, &cur_ids)]));
        }
    }
    // Decode time ends with the last token; a step enqueued ahead of an EOS / cancel still runs
    // on the card: wait for it and drop its tokens.
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    if !flights.is_empty() {
        let t_drain = Instant::now();
        unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
            .map_err(|e| format!("drain spec steps: {e}"))?;
        stats.drain_ms = t_drain.elapsed().as_secs_f64() * 1000.0;
    }
    prof_rt::end_request(eng, stats.launches + flights.len() as u64)?;
    if calibrating {
        let rt = eng.specdec.as_ref().unwrap();
        if let Some(d) = rt.drafter.as_ref() {
            d.calibrate_store(rt.verbose)?;
        }
    }

    let decode_tokens = generated.len() - n0;
    if eng.specdec.as_ref().unwrap().verbose && stats.launches > 0 {
        let n = stats.launches as f64;
        eprintln!(
            "[gaudium] spec host split per launch: enqueue {:.3} ms, wait {:.3} ms, read-back {:.3} ms; widths K:launches {:?}; drained after the last token: {:.1} ms",
            stats.host_ms[0] / n,
            stats.host_ms[1] / n,
            stats.host_ms[2] / n,
            stats.widths,
            stats.drain_ms
        );
        let per: Vec<String> = width_ms
            .iter()
            .map(|(k, (ms, c))| format!("K={k}: {:.3} ms x{c}", ms / *c as f64))
            .collect();
        eprintln!(
            "[gaudium] spec step time by verify width (completion interval, first 2 launches excluded): {}",
            per.join(", ")
        );
    }
    eng.specdec.as_mut().unwrap().last_stats = stats.clone();
    eng.specdec.as_mut().unwrap().width_cost[usize::from(sampled)] = policy.into_cost();
    Ok(SpecDecodeOut {
        decode_ms,
        decode_tokens,
        stopped_by_eos,
        stopped_by_cancel,
        stats,
    })
}

/// Drafter calibration run: reads the GEMM input sites of the block the last launch drafted.
fn calibrate_drafter(eng: &mut Qwen3Engine) -> Result<(), String> {
    let Qwen3Engine { specdec, dev, .. } = eng;
    let Some(rt) = specdec.as_mut() else {
        return Ok(());
    };
    let SpecRuntime { drafter, addr, .. } = rt;
    match drafter.as_mut() {
        Some(d) if d.calibrating() => d.calibrate_observe(dev, addr),
        _ => Ok(()),
    }
}

/// `sample_check` of a coupled DSpine block: walks the block again on the host from the card's
/// own scores, candidates and row scales (the CPU twin of `gaudium_ds_cwalk`) and compares the
/// drafts. `base` = the block anchor's sequence index. No-op for other drafters.
fn check_coupled_walk(eng: &mut Qwen3Engine, base: u32) -> Result<(), String> {
    let Some(rt) = eng.specdec.as_ref() else {
        return Ok(());
    };
    let Some(d) = rt.drafter.as_ref().filter(|d| d.has_coupled()) else {
        return Ok(());
    };
    let s_rows = (d.cfg.block - 1) as usize;
    let (a_sc, a_cand, a_cs, a_dr) = (
        rt.addr["blk_scores"],
        rt.addr["blk_cand"],
        rt.addr["blk_cwcs"],
        rt.addr["blk_drafts"],
    );
    let dev_p = eng.samp.as_ref().expect("on-card sampler").dev;
    if (dev_p.i[3] & crate::sampler::F_COUPLED) == 0 {
        return Ok(());
    }
    let f32s = |eng: &mut Qwen3Engine, addr: u64, n: usize| -> Result<Vec<f32>, String> {
        eng.dev
            .copy_d2h_pinned_sync(addr, &mut eng.stage_io, n * 4)
            .map_err(|e| format!("sample-check D2H: {e}"))?;
        Ok(eng.stage_io.as_slice()[..n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect())
    };
    let scores = f32s(eng, a_sc, 16 * 16 * s_rows)?;
    let cs_all = f32s(eng, a_cs, 64 * s_rows)?;
    let cs: Vec<f32> = (0..s_rows).map(|s| cs_all[s * 64]).collect();
    let cand = read_i32s(eng, a_cand, 16 * (s_rows + 1))?;
    let card = read_i32s(eng, a_dr, s_rows)?;
    let host = crate::sampler::ds_cwalk(&scores, &cand, &cs, s_rows, &dev_p, base);
    let bad = host.iter().zip(&card).filter(|(h, c)| h != c).count() as u64;
    let rt = eng.samp.as_mut().expect("on-card sampler");
    rt.walk_rows += s_rows as u64;
    rt.walk_bad += bad;
    if bad > 0 && rt.walk_bad <= 8 * s_rows as u64 {
        eprintln!(
            "[gaudium] SAMPLE-CHECK coupled walk mismatch at anchor index {base}: card {card:?}, host twin {host:?}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen38_27b_spec() -> ModelSpec {
        let cfg = r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
          }
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    fn qwen3_8b_spec() -> ModelSpec {
        let cfg = r#"{
          "model_type": "qwen3",
          "hidden_size": 4096,
          "intermediate_size": 12288,
          "num_hidden_layers": 36,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "vocab_size": 151936,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "tie_word_embeddings": false
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    #[test]
    fn chain_accept_is_the_longest_matching_prefix() {
        // ids[0] = last token, ids[1..] = drafts; tokens[i] = argmax after row i
        assert_eq!(chain_accept(&[5, 6, 7, 8], &[6, 7, 8, 9]), 3);
        assert_eq!(chain_accept(&[5, 6, 7, 8], &[6, 9, 8, 9]), 1);
        assert_eq!(chain_accept(&[5, 6, 7, 8], &[0, 7, 8, 9]), 0);
        assert_eq!(chain_accept(&[5], &[6]), 0);
        // a later match after a miss must not extend the prefix
        assert_eq!(chain_accept(&[5, 1, 7, 8], &[6, 7, 8, 9]), 0);
    }

    #[test]
    fn hybrid_step_graph_has_rings_checkpoints_and_accept() {
        let spec = qwen38_27b_spec();
        let elfs = vec![("GAUDIUM_ACCEPTT_ELF".to_string(), "00".to_string())];
        let gb = build_spec_step_graph(&spec, 4096, 1, 16, 16, None, &elfs);
        let specs = gb.persistent_tensor_specs();
        let dims = |n: &str| specs.iter().find(|t| t.0 == n).map(|t| t.2.clone());
        assert_eq!(dims("L0_S_in"), Some(vec![64, 256, 48, 16]));
        assert_eq!(dims("L0_S_ckpt"), Some(vec![64, 256, 48, 16]));
        assert_eq!(dims("L0_cs_in"), Some(vec![10240, 3, 16]));
        assert_eq!(dims("L0_cs_ckpt"), Some(vec![10240, 3, 16]));
        assert_eq!(dims("tokens"), Some(vec![1, 16]));
        assert_eq!(dims("emitted"), Some(vec![16]));
        assert_eq!(dims("a_in"), Some(vec![1]));
        assert_eq!(
            specs.iter().filter(|t| t.0.ends_with("_S_ckpt")).count(),
            48
        );
        // a narrower step of the same family reads the same rings (same slot count)
        let gb8 = build_spec_step_graph(&spec, 4096, 1, 8, 16, None, &elfs);
        let specs8 = gb8.persistent_tensor_specs();
        let d8 = |n: &str| specs8.iter().find(|t| t.0 == n).map(|t| t.2.clone());
        assert_eq!(d8("L0_S_in"), Some(vec![64, 256, 48, 16]));
        assert_eq!(d8("L0_S_ckpt"), Some(vec![64, 256, 48, 8]));
        assert_ne!(gb.fingerprint("x"), gb8.fingerprint("x"));
        assert_eq!(
            gb.fingerprint("x"),
            build_spec_step_graph(&spec, 4096, 1, 16, 16, None, &elfs).fingerprint("x")
        );
    }

    #[test]
    fn attention_only_step_graph_has_no_state_slot() {
        let spec = qwen3_8b_spec();
        let gb = build_spec_step_graph(&spec, 8192, 2, 8, 8, None, &[]);
        let specs = gb.persistent_tensor_specs();
        assert!(!specs.iter().any(|t| t.0 == "a_in"));
        assert!(specs.iter().any(|t| t.0 == "a_out"));
        let kc = specs.iter().find(|t| t.0 == "L0_kcache").unwrap();
        assert_eq!(kc.2, vec![128, 4096, 16]);
    }

    #[test]
    fn width_policy_prefers_wide_when_drafts_hold_and_narrow_when_they_miss() {
        let mut p = WidthPolicy::new(&[7, 15]);
        // every width is timed on the card before the policy compares them, widest first
        for _ in 0..WidthPolicy::MIN_TIMED {
            assert_eq!(p.choose(4096), 15);
            p.observe(15, 15, 32.2);
        }
        for _ in 0..WidthPolicy::MIN_TIMED {
            assert_eq!(p.choose(4096), 7);
            p.observe(7, 7, 30.2);
        }
        // drafts that hold deep: the wide step wins
        for _ in 0..40 {
            p.observe(15, 15, 32.2);
        }
        assert_eq!(p.choose(4096), 15);
        // drafts that break after one token: the narrow step wins, and keeps winning while the
        // narrow steps confirm it (untested depths fall back to the pooled rate, not to a prior)
        for _ in 0..80 {
            p.observe(15, 1, 32.2);
        }
        assert_eq!(p.choose(4096), 7);
        for _ in 0..80 {
            p.observe(7, 1, 30.2);
        }
        assert_eq!(p.choose(4096), 7);
        // fully accepted narrow steps make the wide one worth it again
        for _ in 0..80 {
            p.observe(7, 7, 30.2);
        }
        assert_eq!(p.choose(4096), 15);
        // near the end of the context only widths that fit are offered
        assert_eq!(p.choose(8), 7);
        assert_eq!(p.choose(3), 0);
    }

    #[test]
    fn mtp_widening_goes_one_listed_width_per_step() {
        let ks = [6u64, 4, 3, 2, 1];
        // narrower or equal: at once
        assert_eq!(mtp_next_drafts(&ks, 4, 2), 2);
        assert_eq!(mtp_next_drafts(&ks, 4, 4), 4);
        // wider: the next listed width above the verified one, never past the wanted one
        assert_eq!(mtp_next_drafts(&ks, 1, 6), 2);
        assert_eq!(mtp_next_drafts(&ks, 4, 6), 6);
        assert_eq!(mtp_next_drafts(&ks, 2, 3), 3);
        // the 1-row fallback drafts nothing
        assert_eq!(mtp_next_drafts(&ks, 0, 6), 0);
        // a single width never widens
        assert_eq!(mtp_next_drafts(&[4], 4, 4), 4);
        // a step verifies `min(drafted, wanted)`, so `wanted >= verified`: every reachable pair
        // is a fixed-width step or the one widen step of its width
        let mut keys = std::collections::BTreeSet::new();
        for &k in &ks {
            for &w in ks.iter().filter(|&&w| w >= k) {
                keys.insert(mtp_step_key(k + 1, mtp_next_drafts(&ks, k, w)));
            }
        }
        assert_eq!(keys.len(), ks.len() + (ks.len() - 1));
    }

    #[test]
    fn mtp_step_keys_carry_rows_and_drafts() {
        // the fixed-width step keeps the plain row count every other method uses
        for t in 1..=SPEC_MAX_ROWS {
            assert_eq!(mtp_step_key(t, t - 1), t);
            assert_eq!(mtp_key_parts(t), (t, t - 1));
        }
        for t in 2..=SPEC_MAX_ROWS {
            for kd in 1..=mtp::MTP_MAX_DRAFTS {
                let key = mtp_step_key(t, kd);
                assert_eq!(mtp_key_parts(key), (t, kd));
                assert_eq!(key == t, kd + 1 == t);
            }
        }
    }

    #[test]
    fn width_policy_times_each_width_once_per_engine_and_counts_launches_in_flight() {
        // launches still in flight count towards a width's timing
        let mut p = WidthPolicy::new(&[7, 15]);
        assert_eq!(p.choose_with_pending(4096, &[]), 15);
        assert_eq!(p.choose_with_pending(4096, &[15, 15]), 15);
        assert_eq!(p.choose_with_pending(4096, &[15, 15, 15]), 7);
        p.observe(15, 15, 20.9);
        assert_eq!(p.choose_with_pending(4096, &[15, 15]), 7);
        for _ in 0..2 {
            p.observe(15, 15, 20.9);
        }
        for _ in 0..WidthPolicy::MIN_TIMED {
            p.observe(7, 7, 18.8);
        }
        // the next request starts from the measured times: no width is launched just to be timed,
        // and with no acceptance seen yet the narrow step is the better bet at these costs
        let mut q = WidthPolicy::with_cost(&[7, 15], p.into_cost());
        assert_eq!(q.choose(4096), 7);
        // drafts that hold through the narrow block make the wide step worth its rows at once
        q.observe(7, 7, 18.8);
        q.observe(7, 7, 18.8);
        assert_eq!(q.choose(4096), 15);
        // drafts that break early keep it narrow
        let mut r = WidthPolicy::with_cost(&[7, 15], q.into_cost());
        for _ in 0..4 {
            r.observe(7, 1, 18.8);
        }
        assert_eq!(r.choose(4096), 7);
    }

    #[test]
    fn width_policy_forgets_old_estimates_of_positions_it_no_longer_tests() {
        let mut p = WidthPolicy::new(&[7, 15]);
        for _ in 0..WidthPolicy::MIN_TIMED {
            p.observe(15, 15, 20.9);
            p.observe(7, 7, 18.8);
        }
        // the drafts break right after the narrow block for a while: position 8's own estimate
        // sinks and the wide step is not worth its rows
        for _ in 0..30 {
            p.observe(15, 7, 20.9);
        }
        assert_eq!(p.choose(4096), 7);
        // narrow steps that are fully accepted never test positions 8..15 again; while the old
        // estimates are fresh they keep the policy narrow ...
        for _ in 0..10 {
            p.observe(7, 7, 18.8);
        }
        assert_eq!(p.choose(4096), 7);
        // ... and once they have expired the pooled rate of the positions still tested stands in
        for _ in 0..WidthPolicy::MAX_AGE {
            p.observe(7, 7, 18.8);
        }
        assert_eq!(p.choose(4096), 15);
    }

    #[test]
    fn stats_report_conditional_acceptance_per_position() {
        let s = SpecStats {
            launches: 4,
            emitted: 0,
            accept_hist: vec![1, 1, 2],
            verified: vec![4, 4],
            widths: BTreeMap::new(),
            host_ms: [0.0; 3],
            drain_ms: 0.0,
        };
        assert_eq!(s.accept_per_pos(), vec![0.75, 0.5]);
    }
}
