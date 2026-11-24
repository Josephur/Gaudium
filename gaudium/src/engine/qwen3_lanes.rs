//! Lanes of the dense Qwen3 / Qwen3.5 / Qwen3.8 engine (`--parallel N`): several requests at
//! once, each in a lane of its own, the decoding lanes stepped together in one launch.
//!
//! This is a child module of `engine::qwen3` (declared there with `#[path]`), so it reuses that
//! module's private graph helpers and engine fields without widening their visibility.
//!
//! What a lane owns: the K/V caches of every attention layer, the two ping-pong buffers of every
//! DeltaNet layer's recurrent and conv state, the prefix table with its kept positions, and the
//! per-request reset bookkeeping. They are planned up front for every lane (`load_or_plan`), so a
//! configuration of N lanes at the requested context either fits or is refused at load.
//!
//! The engine's own fields (`kv_addr`, `gdn_state`, `prefix`, ...) always hold one lane, the
//! active one; `activate` swaps another lane in. A lane's prefill is therefore the single-stream
//! prefill, unchanged, run with that lane active.
//!
//! One batched step `g_qwen_bstep_B{b}_C{nch}` decodes `b` lanes (slots `0..b`, any lanes in any
//! order) by one token each:
//! - the trunk (norms, every projection, the MLPs, the readout GEMM) runs once over `b` columns,
//!   so each weight is read from HBM once for all lanes (decode is bound by that read);
//! - per slot, on its own column: the attention core over the slot's lane cache at the slot's own
//!   cursor (`s{j}_cursor`), and the DeltaNet recurrence from the slot's lane state. Every slot
//!   covers `nch` chunks, the most any slot needs: a slot's chunks past its cursor are fully
//!   masked, which the softmax writes as `m = -1e30, s = 0, P = 0` and the combine adds as exact
//!   zeros, so a slot computes the bits it computes alone. The slot table is checked by the
//!   verified `ln_slot_fits` before every launch;
//! - the readout: the argmax per column, or (`g_qwen_sbstep_*`) per slot the on-card sampler
//!   with that lane's own parameters (`s{j}_samp_f` / `s{j}_samp_i`) at the slot's sequence index.
//!
//! Exactness: a slot's column goes through the same nodes as the single-stream step's one column;
//! what has to hold is that a GEMM's result for a column does not depend on how many columns the
//! launch has. That is measured on the card (each lane against the same request run alone).

use super::*;
use crate::core::lanes::{
    batch_chunks, slot_fits, spec_batched, step_chunk_counts, step_chunks, LaneTable, Offer,
};
use crate::engine::{FeedHead, LaneDone, LaneFeed, LaneJob, LaneOut, LaneView};

/// Name prefix of slot `j`'s own tensors in a batched step.
pub fn slot_pfx(j: u64) -> String {
    format!("s{j}_")
}

/// Name of a batched step recipe.
pub fn batched_step_name(b: u64, nch: u64, sampled: bool) -> String {
    format!("g_qwen_{}bstep_B{b}_C{nch}", if sampled { "s" } else { "" })
}

/// `b` columns of `x {d, b}` as `b` tensors `{d, 1}` (intermediates).
fn split_cols(
    gb: &mut GraphBuilder,
    x: TensorId,
    dt: SynDataType,
    d: u64,
    b: u64,
    name: &str,
) -> Vec<TensorId> {
    let parts: Vec<TensorId> = (0..b)
        .map(|j| gb.add_intermediate(format!("{name}_c{j}"), dt, &[d, 1]))
        .collect();
    gb.op_split(format!("{name}_split"), x, &parts, 1);
    parts
}

/// The batched decode step of `b` slots over `nch` KV chunks (see the module text).
pub fn build_batched_step_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    b: u64,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    let mut gb = GraphBuilder::new(batched_step_name(b, nch, sampled));
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let hid = spec.hidden;
    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, spec.vocab]);
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[b]);
    let cursors: Vec<TensorId> = (0..b)
        .map(|j| {
            gb.add_persistent_tensor(format!("{}cursor", slot_pfx(j)), SynDataType::Int32, &[1])
        })
        .collect();
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let slot = (spec.n_linear_layers() > 0)
        .then(|| gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]));

    let mut x = gb.add_persistent_tensor("x0", SynDataType::Bf16, &[hid, b]);
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
        gb.set_fp8_weights(spec.fp8_weights); // [fp8 hook]
        let attn = if spec.is_full_attention_layer(l) {
            let (nh, dh) = (spec.n_heads, spec.head_dim);
            let (qg, k, v) = append_attn_proj(&mut gb, spec, x, h_pre, &lpfx, b);
            let qgs = split_cols(
                &mut gb,
                qg,
                SynDataType::Bf16,
                spec.qg_dim(),
                b,
                &format!("{lpfx}qg"),
            );
            let ks = split_cols(
                &mut gb,
                k,
                SynDataType::Bf16,
                spec.kv_dim(),
                b,
                &format!("{lpfx}k"),
            );
            let vs = split_cols(
                &mut gb,
                v,
                SynDataType::Bf16,
                spec.kv_dim(),
                b,
                &format!("{lpfx}v"),
            );
            // every slot on its own rows, cursor and caches, stage by stage across the slots
            let slots: Vec<AttnSlot> = (0..b as usize)
                .map(|j| AttnSlot {
                    qg: qgs[j],
                    k: ks[j],
                    v: vs[j],
                    cursor: cursors[j],
                    spfx: format!("{}{lpfx}", slot_pfx(j as u64)),
                    t: 1,
                })
                .collect();
            let cores = append_attn_core_slots(&mut gb, spec, &slots, rope_c, rope_s, &lpfx, nch);
            let core = gb.add_intermediate(format!("{lpfx}core"), SynDataType::Bf16, &[nh * dh, b]);
            gb.op_concat(format!("{lpfx}core_cat"), &cores, core, 1);
            add_linear_bf16(
                &mut gb,
                core,
                &format!("{lpfx}W_o"),
                nh * dh,
                hid,
                b,
                &format!("{lpfx}attn"),
            )
        } else {
            let (hv, dl) = (spec.gdn_v_heads, spec.gdn_v_dim);
            let (qkvz, ba) = append_linear_proj(&mut gb, spec, x, h_pre, &lpfx, b);
            let qs = split_cols(
                &mut gb,
                qkvz,
                SynDataType::Bf16,
                spec.gdn_qkvz_dim(),
                b,
                &format!("{lpfx}qkvz"),
            );
            let bs = split_cols(
                &mut gb,
                ba,
                SynDataType::Bf16,
                spec.gdn_ba_dim(),
                b,
                &format!("{lpfx}ba"),
            );
            let cores: Vec<TensorId> = (0..b as usize)
                .map(|j| {
                    let spfx = format!("{}{lpfx}", slot_pfx(j as u64));
                    append_gdn_core(
                        &mut gb,
                        spec,
                        qs[j],
                        bs[j],
                        slot.expect("zero_slot declared for hybrid models"),
                        &lpfx,
                        &spfx,
                        1,
                    )
                })
                .collect();
            let core = gb.add_intermediate(format!("{lpfx}core"), SynDataType::Bf16, &[hv * dl, b]);
            gb.op_concat(format!("{lpfx}core_cat"), &cores, core, 1);
            add_linear_bf16(
                &mut gb,
                core,
                &format!("{lpfx}W_o"),
                hv * dl,
                hid,
                b,
                &format!("{lpfx}attn"),
            )
        };
        let (out_x, next) = append_post_and_mlp(
            &mut gb,
            spec,
            x,
            attn,
            &lpfx,
            b,
            true,
            Some((next_w.as_str(), next_h.as_str())),
        );
        x = out_x;
        h_pre = next;
    }
    let h_final = h_pre.expect("fused final norm produced by last layer");
    let lm_w = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    let logits = add_linear_bf16(&mut gb, h_final, lm_w, hid, spec.vocab, b, "logits");
    let tokens = gb.add_persistent_tensor("tokens", SynDataType::Int32, &[1, b]);
    if sampled {
        let cols = split_cols(&mut gb, logits, SynDataType::Bf16, spec.vocab, b, "logits");
        let toks: Vec<TensorId> = (0..b as usize)
            .map(|j| {
                let sp = slot_pfx(j as u64);
                let tok = gb.add_intermediate(format!("{sp}tokens"), SynDataType::Int32, &[1, 1]);
                let at = crate::engine::sampler::SamplerAt {
                    base: cursors[j],
                    off: 1,
                    role: 0,
                };
                crate::engine::sampler::append_sampler_as(
                    &mut gb,
                    cols[j],
                    spec.vocab,
                    1,
                    &at,
                    &format!("{sp}tokens"),
                    &format!("{sp}samp_f"),
                    &format!("{sp}samp_i"),
                    Some(tok),
                )
            })
            .collect();
        gb.op_concat("tokens_cat", &toks, tokens, 1);
    } else {
        gb.op_argmax_bf16("argmax", logits, tokens, 0);
    }
    gb
}

/// A tensor of a batched step bound to the lane in one of its slots: `(slot, the lane's own
/// name of the buffer)`: its caches and states (by their single-stream names), its cursor and
/// its sampler parameters.
pub fn slot_bound(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix('s')?;
    let sep = rest.find('_')?;
    let j: usize = rest[..sep].parse().ok()?;
    let own = &rest[sep + 1..];
    let lane_owned = matches!(own, "cursor" | "samp_f" | "samp_i")
        || is_kv_tensor(own)
        || is_gdn_state_tensor(own);
    lane_owned.then_some((j, own))
}

/// The step's inputs and outputs written and read by the host every step.
fn is_step_io(name: &str) -> bool {
    matches!(name, "ids" | "tokens")
}

/// Which batched step a loaded engine's lanes take.
#[derive(Debug, Clone)]
pub(super) enum LaneFamily {
    /// Plain decode, one row per slot ([`build_batched_step_graph`]; the kernels' hashes).
    K0 { hashes: Vec<(String, String)> },
    /// Speculative: verify rows, accept and the DSpine draft per slot (`spec::lanes`).
    Spec(spec::lanes::SpecLaneGraph),
}

impl LaneFamily {
    /// The plain family's kernel hashes: the trunk's, and the sampler's when it is there.
    pub fn k0(trunk: &[(String, String)], kenv: &KernelEnv, sampler: bool) -> Result<Self, String> {
        let mut hashes = trunk.to_vec();
        if sampler {
            for (k, h) in kenv.fingerprint_hashes(crate::engine::sampler::SAMPLER_ELFS)? {
                if !hashes.iter().any(|(e, _)| *e == k) {
                    hashes.push((k, h));
                }
            }
        }
        Ok(Self::K0 { hashes })
    }

    fn graph(&self, spec: &ModelSpec, lmax: u64, key: &StepKey) -> GraphBuilder {
        let (ts, nch, sampled) = (&key.0, key.1, key.2);
        match self {
            Self::K0 { hashes } => {
                build_batched_step_graph(spec, lmax, nch, ts.len() as u64, hashes, sampled)
            }
            Self::Spec(g) => g.build(spec, ts, nch, sampled),
        }
    }

    fn name(&self, key: &StepKey) -> String {
        let (ts, nch, sampled) = (&key.0, key.1, key.2);
        match self {
            Self::K0 { .. } => batched_step_name(ts.len() as u64, nch, sampled),
            Self::Spec(g) => {
                spec::lanes::batched_spec_name(ts, nch, sampled, spec::replay::enabled(&g.hashes))
            }
        }
    }

    /// Rows a slot writes per step at its narrow and at its wide width (the same without a
    /// width of its own per lane).
    fn rows(&self) -> (u64, u64) {
        match self {
            Self::K0 { .. } => (1, 1),
            Self::Spec(g) => (g.narrow, g.wide),
        }
    }

    /// The slot rows of every step of `b` slots (widest first).
    fn slot_rows(&self, b: u64) -> Vec<Vec<u64>> {
        match self {
            Self::K0 { .. } => vec![vec![1; b as usize]],
            Self::Spec(g) => g.slot_rows(b),
        }
    }

    /// Is this tensor bound to a slot's lane?
    fn bound(&self, name: &str) -> bool {
        match self {
            Self::K0 { .. } => slot_bound(name).is_some(),
            Self::Spec(_) => spec::lanes::spec_slot_bound(name).is_some(),
        }
    }
}

/// One lane's own device buffers and request bookkeeping while the lane is not the active one.
pub(super) struct LaneBufs {
    kv_addr: BTreeMap<String, (u64, u64)>,
    gdn_state: BTreeMap<usize, GdnLayerState>,
    reset: request_reset::ResetState,
    prefix: prefix_cache::PrefixCache,
    prefix_slots: Vec<prefix_cache::SlotAddrs>,
    prefix_plan: prefix_cache::Plan,
    prefix_images: Vec<prefix_cache::ImageId>,
    /// The lane's share of the speculative runtime (`None` without a drafter).
    pub spec: Option<spec::SpecLane>,
    /// Where that share was allocated (until the runtime is finished and `spec` is made).
    spec_addr: BTreeMap<String, u64>,
}

impl LaneBufs {
    /// What the active lane leaves in its own entry (nothing: its buffers are the engine's).
    fn placeholder(mode: request_reset::ResetMode) -> Self {
        Self {
            kv_addr: BTreeMap::new(),
            gdn_state: BTreeMap::new(),
            reset: request_reset::ResetState::new(mode, false),
            prefix: prefix_cache::PrefixCache::off(),
            prefix_slots: Vec::new(),
            prefix_plan: prefix_cache::Plan::default(),
            prefix_images: Vec::new(),
            spec: None,
            spec_addr: BTreeMap::new(),
        }
    }
}

/// The plan of the lanes beside the single-stream engine, before anything is allocated.
pub(super) struct LanePlan {
    pub n: usize,
    /// Buffers of the batched steps (scratch shared by every lane), by name.
    pub act_specs: BTreeMap<String, u64>,
    /// Step inputs / outputs and per-lane sampler parameters, in bytes (`io_names`, one
    /// buffer each).
    pub io_bytes: u64,
    /// Recipes compiled at load (their workspaces are part of the plan).
    pub eager: Vec<(StepKey, CompiledRecipe)>,
    pub max_ws: u64,
    pub sampled: bool,
    pub family: LaneFamily,
    /// The batched recipes bypass the recipe cache (`--profile-steps`).
    pub fresh: bool,
}

/// Bytes of one scalar buffer of the step io (each one its own 256-aligned buffer).
pub const IO_SCALAR: u64 = 256;

/// The step io buffers of `n` lanes: the ids and tokens of a step, one cursor per slot, and two
/// sampler parameter buffers per lane.
pub fn io_names(n: usize) -> Vec<String> {
    let mut v = vec!["io_ids".to_string(), "io_tokens".to_string()];
    v.extend((0..n).map(|j| format!("io_cursor{j}")));
    for l in 0..n {
        v.push(format!("io_samp_f{l}"));
        v.push(format!("io_samp_i{l}"));
    }
    v
}

impl LanePlan {
    /// Plans `n` lanes: the buffers of the widest batched step (`n` slots, every chunk), and the
    /// recipes a load compiles (every width at one chunk, and the widest at every chunk, which
    /// sizes the shared workspace). `weights`: names the single-stream engine binds as weights.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        n: usize,
        spec: &ModelSpec,
        lmax: u64,
        max_chunks: u64,
        family: LaneFamily,
        sampler: bool,
        weights: &dyn Fn(&str) -> bool,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
        verbose: bool,
        compile_all: bool,
        fresh: bool,
    ) -> Result<Self, String> {
        // a profiled load compiles its batched recipes here (node names reach a trace only from
        // recipes compiled in the process); the engine's other recipes still load from the cache
        let off = RecipeCache::disabled();
        let cache = if fresh { &off } else { cache };
        let widest = family.graph(spec, lmax, &(vec![family.rows().1; n], max_chunks, sampler));
        let mut act_specs = BTreeMap::new();
        for (name, _dt, _dims, bytes) in widest.persistent_tensor_specs() {
            if weights(&name)
                || crate::engine::sampler::is_sampler_param(&name)
                || family.bound(&name)
                || is_step_io(&name)
            {
                continue;
            }
            act_specs.insert(name, bytes);
        }
        let io_bytes = crate::verified::ln_lane_io_bytes(io_names(n).len() as u64, IO_SCALAR);
        // every slot-rows combination of every width at one chunk (and with `compile_all` at
        // every chunk count), and the widest step at every chunk (it sizes the workspace)
        let widest_rows = vec![family.rows().1; n];
        let mut keys: Vec<StepKey> = Vec::new();
        let fams: &[bool] = if sampler { &[false, true] } else { &[false] };
        for &sm in fams {
            let chunk_counts = if compile_all {
                step_chunk_counts(max_chunks)
            } else {
                vec![1]
            };
            for nch in chunk_counts {
                for b in 1..=n as u64 {
                    for ts in family.slot_rows(b) {
                        keys.push((ts, nch, sm));
                    }
                }
            }
            if !compile_all && max_chunks > 1 {
                keys.push((widest_rows.clone(), max_chunks, sm));
            }
        }
        let t0 = Instant::now();
        let describe = |i: usize| -> GraphBuilder { family.graph(spec, lmax, &keys[i]) };
        let mut cached = crate::synapse::graph::load_cached_parallel(
            keys.len(),
            crate::synapse::graph::recipe_load_threads(),
            api,
            cache,
            driver_version,
            describe,
        );
        let (mut hits, mut built, mut max_ws) = (0u64, 0u64, 0u64);
        let mut eager = Vec::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            let r = match std::mem::replace(&mut cached[i], Ok(None)) {
                Ok(Some(r)) => r,
                _ => describe(i)
                    .compile(Arc::clone(api), cache, driver_version)
                    .map_err(|e| {
                        format!(
                            "compile batched step {}: {e}",
                            family.name(key)
                        )
                    })?,
            };
            if r.from_cache {
                hits += 1;
            } else {
                built += 1;
            }
            max_ws = max_ws.max(r.ws_size);
            eager.push((key.clone(), r));
        }
        if verbose {
            eprintln!(
                "[gaudium] lanes: {n} lanes, batched step recipes {hits} from the cache, {built} compiled ({:.1}s); largest workspace {:.2} MiB; step buffers {:.2} GiB",
                t0.elapsed().as_secs_f64(),
                max_ws as f64 / (1024.0 * 1024.0),
                act_specs
                    .values()
                    .fold(0, |acc, &b| crate::verified::ln_lane_bufs_add(acc, b)) as f64
                    / (1024.0 * 1024.0 * 1024.0)
            );
        }
        Ok(Self {
            n,
            act_specs,
            io_bytes,
            eager,
            max_ws,
            sampled: sampler,
            family,
            fresh,
        })
    }

    /// Bytes of the step buffers and the io, as the act arena allocates them.
    pub fn act_bytes(&self) -> u64 {
        // (the io first, then every step buffer as the arena allocates it)
        self.act_specs
            .values()
            .fold(self.io_bytes, |acc, &b| crate::verified::ln_lane_bufs_add(acc, b))
    }
}

/// Key of a batched step recipe: every slot's rows (widest first), the chunk count, sampled.
pub(super) type StepKey = (Vec<u64>, u64, bool);

/// Key of a prepared batched launch: the recipe, and the lane and state parity of every slot.
type LaunchKey = (StepKey, Vec<u8>, Vec<u8>);

/// Lanes of a loaded engine.
pub(super) struct LaneRt {
    n: usize,
    /// The lane whose buffers are the engine's own fields now.
    active: usize,
    bufs: Vec<LaneBufs>,
    /// Per lane: its sampler parameter buffers `(samp_f, samp_i)`.
    samp_bufs: Vec<(u64, u64)>,
    io_ids: u64,
    io_tokens: u64,
    io_cursor: Vec<u64>,
    /// Weights, step buffers and the io: what every batched launch binds the same way.
    base: BTreeMap<String, u64>,
    pub family: LaneFamily,
    sampled: bool,
    recipes: HashMap<StepKey, CompiledRecipe>,
    launches: HashMap<LaunchKey, PreparedLaunch>,
    pub table: LaneTable,
    /// Batched launches since the load (the lane windows of `--profile-steps` count them).
    launched: u64,
    /// Batched recipes compiled on first use bypass the recipe cache (`--profile-steps`).
    fresh: bool,
    /// Where a lane session's wall time goes (report only: `--verbose`).
    timing: StepTiming,
    /// The measured step times the lanes' width choices use.
    widths: WidthCosts,
    /// Timed events around the speculative launches, `(start, end)` (two pairs: one launch may
    /// be queued behind the other), and the pair the next launch records.
    events: Vec<(crate::synapse::device::DeviceEvent, crate::synapse::device::DeviceEvent)>,
    next_event: usize,
}

/// What the lanes' width choices know of a batched speculative step's time (measured,
/// exponentially averaged): per slot count and chunk count, the launch with every slot at its
/// narrow width; per slot count, the extra time of one slot at its wide width.
#[derive(Debug, Default)]
struct WidthCosts {
    narrow_ms: HashMap<(u64, u64), f64>,
    extra_ms: HashMap<u64, f64>,
    /// Slot steps of the session at the wide width, and in all (report only).
    wide_slots: u64,
    slots: u64,
}

impl WidthCosts {
    /// Weight of a new launch in the averages.
    const ALPHA: f64 = 0.2;

    fn ema(slot: &mut Option<f64>, x: f64) {
        *slot = Some(match *slot {
            Some(m) => m + Self::ALPHA * (x - m),
            None => x,
        });
    }

    /// A launch of `b` slots over `nch` chunks, `wide` of them at their wide width, took `ms`.
    fn observe(&mut self, b: u64, nch: u64, wide: u64, ms: f64) {
        self.wide_slots += wide;
        self.slots += b;
        if wide == 0 {
            let mut m = self.narrow_ms.get(&(b, nch)).copied();
            Self::ema(&mut m, ms);
            self.narrow_ms.insert((b, nch), m.unwrap_or(ms));
        } else if let Some(&t0) = self.narrow_ms.get(&(b, nch)) {
            let mut m = self.extra_ms.get(&b).copied();
            Self::ema(&mut m, ((ms - t0) / wide as f64).max(0.0));
            self.extra_ms.insert(b, m.unwrap_or(0.0));
        }
    }

    /// What the model holds, for the session report (and a fresh count of the slot steps).
    fn report(&mut self) -> Option<String> {
        if self.wide_slots == 0 && self.extra_ms.is_empty() {
            return None;
        }
        let mut bs: Vec<u64> = self.extra_ms.keys().copied().collect();
        bs.sort_unstable();
        let per_b: Vec<String> = bs
            .iter()
            .map(|b| {
                format!(
                    "{b} slots: {:.2} ms narrow, +{:.2} ms a wide slot",
                    self.narrow_ms.get(&(*b, 1)).copied().unwrap_or(f64::NAN),
                    self.extra_ms[b]
                )
            })
            .collect();
        let line = format!(
            "[gaudium] lanes: widths: {} of {} slot steps wide; {}",
            self.wide_slots,
            self.slots,
            per_b.join("; ")
        );
        self.wide_slots = 0;
        self.slots = 0;
        Some(line)
    }

    /// `(time of the narrow step, extra time of one wide slot)` in ms for `b` slots over `nch`
    /// chunks: `None` until the narrow step was measured. The extra time of a slot count not
    /// measured yet is the nearest measured slot count's (a wide slot's own work hardly depends
    /// on the others), and 0 until any was (so a lane that expects more from the wide step tries
    /// it).
    fn estimate(&self, b: u64, nch: u64) -> Option<(f64, f64)> {
        let t0 = *self.narrow_ms.get(&(b, nch))?;
        let extra = self.extra_ms.get(&b).copied().or_else(|| {
            self.extra_ms
                .iter()
                .min_by_key(|(&k, _)| k.abs_diff(b))
                .map(|(_, &v)| v)
        });
        Some((t0, extra.unwrap_or(0.0)))
    }
}

/// The wall time of a lane session's batched launches, split at the launch and its sync.
#[derive(Debug, Default)]
struct StepTiming {
    launches: u64,
    /// Before the launch: the slot table, the recipe and its prepared bindings, the inputs.
    before_ms: f64,
    /// From the launch to the end of its sync: the device.
    device_ms: f64,
    /// After the sync: the read-back of each slot's tokens.
    after_ms: f64,
    /// From one launch's read-back to the next launch's start: the loop's own work (tokens,
    /// admissions, prefills of new requests).
    between_ms: f64,
    last_end: Option<Instant>,
}

impl StepTiming {
    fn begin(&mut self) -> Instant {
        let now = Instant::now();
        if let Some(e) = self.last_end {
            self.between_ms += (now - e).as_secs_f64() * 1e3;
        }
        now
    }

    fn end(&mut self, t0: Instant, launched: Instant, synced: Instant) {
        let now = Instant::now();
        self.launches += 1;
        self.before_ms += (launched - t0).as_secs_f64() * 1e3;
        self.device_ms += (synced - launched).as_secs_f64() * 1e3;
        self.after_ms += (now - synced).as_secs_f64() * 1e3;
        self.last_end = Some(now);
    }

    /// The split per launch, then a fresh count.
    fn report(&mut self) -> Option<String> {
        let n = self.launches;
        let line = (n > 0).then(|| {
            let per = |ms: f64| ms / n as f64;
            format!(
                "[gaudium] lanes: {n} batched launches: per launch {:.2} ms before the launch, {:.2} ms on the device, {:.2} ms reading back, {:.2} ms between launches",
                per(self.before_ms),
                per(self.device_ms),
                per(self.after_ms),
                per(self.between_ms)
            )
        });
        *self = Self::default();
        line
    }
}

/// Prepared launches kept before the cache is cleared.
const LAUNCH_CACHE: usize = 256;

impl LaneRt {
    /// After the device plan is allocated: lane 0 is the engine's own buffers, `others` lanes
    /// `1..n`; `act_addr` the step buffers, `io` the io buffers by name ([`io_names`]).
    pub fn new(
        plan: LanePlan,
        others: Vec<LaneBufs>,
        act_addr: BTreeMap<String, u64>,
        io: &BTreeMap<String, u64>,
        reset_mode: request_reset::ResetMode,
    ) -> Self {
        let n = plan.n;
        let io_ids = io["io_ids"];
        let io_tokens = io["io_tokens"];
        let io_cursor: Vec<u64> = (0..n).map(|j| io[&format!("io_cursor{j}")]).collect();
        let samp_bufs: Vec<(u64, u64)> = (0..n)
            .map(|l| (io[&format!("io_samp_f{l}")], io[&format!("io_samp_i{l}")]))
            .collect();
        let mut bufs = vec![LaneBufs::placeholder(reset_mode)];
        bufs.extend(others);
        let mut base = act_addr;
        base.insert("ids".to_string(), io_ids);
        base.insert("tokens".to_string(), io_tokens);
        Self {
            n,
            active: 0,
            bufs,
            samp_bufs,
            io_ids,
            io_tokens,
            io_cursor,
            base,
            family: plan.family,
            sampled: plan.sampled,
            recipes: plan.eager.into_iter().collect(),
            launches: HashMap::new(),
            table: LaneTable::new(n),
            launched: 0,
            fresh: plan.fresh,
            timing: StepTiming::default(),
            widths: WidthCosts::default(),
            events: Vec::new(),
            next_event: 0,
        }
    }

    pub fn lanes(&self) -> usize {
        self.n
    }

    /// After the speculative runtime is finished: every lane's share of it (lane 0's is the
    /// runtime's own). Returns the context bound buffers of lanes `1..n`, which start as loaded.
    pub fn attach_spec(&mut self, sd: &spec::SpecRuntime) -> Vec<u64> {
        // the drafter's weights and the step tables: bound the same way in every slot
        for (n, a) in sd.weight_addrs() {
            self.base.entry(n).or_insert(a);
        }
        let mut bounds = Vec::new();
        let names: BTreeMap<String, u64> =
            sd.lane_buffers().into_iter().map(|(n, _)| (n, 0)).collect();
        for (i, b) in self.bufs.iter_mut().enumerate() {
            if i == self.active {
                b.spec = Some(sd.lane_at(names.clone()));
            } else {
                if let Some(&a) = b.spec_addr.get("df_nofix") {
                    bounds.push(a);
                }
                b.spec = Some(sd.lane_at(std::mem::take(&mut b.spec_addr)));
            }
        }
        bounds
    }

    /// After the seal: binds the workspaces; `weights` = every address the engine binds as a
    /// weight or table.
    pub fn finish(
        &mut self,
        dev: &HpuDevice,
        weights: &BTreeMap<String, u64>,
    ) -> Result<(), String> {
        for r in self.recipes.values_mut() {
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind {} ws: {e}", r.name))?;
        }
        for (k, &v) in weights {
            self.base.entry(k.clone()).or_insert(v);
        }
        if matches!(self.family, LaneFamily::Spec(_)) {
            let ev = || {
                dev.create_event(true)
                    .map_err(|e| format!("lanes: create a step event: {e}"))
            };
            for _ in 0..2 {
                self.events.push((ev()?, ev()?));
            }
        }
        Ok(())
    }
}

/// The lane buffers for lanes `1..n`, allocated like lane 0's (the engine's own).
#[allow(clippy::too_many_arguments)]
pub(super) fn allocate_lanes(
    n: usize,
    spec: &ModelSpec,
    kv_specs: &BTreeMap<String, u64>,
    s_ring_bytes: u64,
    s_one_bytes: u64,
    cs_ring_bytes: u64,
    cs_one_bytes: u64,
    pfx_per_lane: u64,
    spec_bufs: &[(String, u64)],
    prefix: &dyn Fn() -> prefix_cache::PrefixCache,
    reset: &dyn Fn() -> request_reset::ResetState,
    suballoc: &mut dyn FnMut(u64) -> Result<u64, String>,
) -> Result<Vec<LaneBufs>, String> {
    let mut out = Vec::with_capacity(n.saturating_sub(1));
    for _ in 1..n {
        let mut kv_addr = BTreeMap::new();
        for (name, &bytes) in kv_specs {
            kv_addr.insert(name.clone(), (suballoc(bytes)?, bytes));
        }
        let mut gdn_state = BTreeMap::new();
        for l in 0..(spec.n_layers as usize) {
            if !spec.is_full_attention_layer(l) {
                let s0 = suballoc(s_ring_bytes)?;
                let s1 = suballoc(s_ring_bytes)?;
                let cs0 = suballoc(cs_ring_bytes)?;
                let cs1 = suballoc(cs_ring_bytes)?;
                gdn_state.insert(
                    l,
                    GdnLayerState {
                        s: [s0, s1],
                        s_bytes: s_one_bytes,
                        cs: [cs0, cs1],
                        cs_bytes: cs_one_bytes,
                    },
                );
            }
        }
        let mut prefix_slots = Vec::new();
        for _ in 0..pfx_per_lane {
            let mut m = prefix_cache::SlotAddrs::new();
            for l in 0..(spec.n_layers as usize) {
                if !spec.is_full_attention_layer(l) {
                    let s = suballoc(s_one_bytes)?;
                    let cs = suballoc(cs_one_bytes)?;
                    m.insert(l, (s, cs));
                }
            }
            prefix_slots.push(m);
        }
        let mut spec_addr = BTreeMap::new();
        for (name, bytes) in spec_bufs {
            spec_addr.insert(name.clone(), suballoc(*bytes)?);
        }
        out.push(LaneBufs {
            kv_addr,
            gdn_state,
            reset: reset(),
            prefix: prefix(),
            prefix_slots,
            prefix_plan: prefix_cache::Plan::default(),
            prefix_images: Vec::new(),
            spec: None,
            spec_addr,
        });
    }
    Ok(out)
}

/// Swaps the engine's own lane fields with `b`.
fn swap_fields(eng: &mut Qwen3Engine, b: &mut LaneBufs) {
    std::mem::swap(&mut eng.kv_addr, &mut b.kv_addr);
    std::mem::swap(&mut eng.gdn_state, &mut b.gdn_state);
    std::mem::swap(&mut eng.reset, &mut b.reset);
    std::mem::swap(&mut eng.prefix, &mut b.prefix);
    std::mem::swap(&mut eng.prefix_slots, &mut b.prefix_slots);
    std::mem::swap(&mut eng.prefix_plan, &mut b.prefix_plan);
    std::mem::swap(&mut eng.prefix_images, &mut b.prefix_images);
    if let (Some(sd), Some(sl)) = (eng.specdec.as_mut(), b.spec.as_mut()) {
        sd.swap_lane(sl);
    }
}

/// Makes `lane` the engine's active lane (its buffers become the engine's own fields).
fn activate(eng: &mut Qwen3Engine, lane: usize) {
    let Some(mut rt) = eng.lanes.take() else {
        return;
    };
    if rt.active != lane {
        let cur = rt.active;
        // the active lane's data back into its entry, then the wanted lane's data out of its own
        swap_fields(eng, &mut rt.bufs[cur]);
        swap_fields(eng, &mut rt.bufs[lane]);
        rt.active = lane;
    }
    eng.lanes = Some(rt);
}

/// What one lane's request is doing.
struct Run {
    id: u64,
    params: Qwen3GenParams,
    cancel: Arc<AtomicBool>,
    out: Box<dyn FnMut(LaneOut) + Send>,
    exclusive: bool,
    prompt_len: u64,
    budget: SequenceBudget,
    eos: Vec<i32>,
    /// Rows written so far: the next step reads `cur_tok` at this position.
    cursor: u64,
    cur_tok: i32,
    parity: usize,
    /// A one-row prompt: the next step yields the first token.
    first_pending: bool,
    sampled: bool,
    generated: Vec<i32>,
    stopped_by_eos: bool,
    stopped_by_cancel: bool,
    t_start: Instant,
    ttft_ms: f64,
    prefill_tok_s: f64,
    ttft_recipe_ms: f64,
    t_decode: Option<Instant>,
    steps: u64,
    reused: u64,
    /// Drafts verified, and accepted, over the request (speculative lanes).
    drafted: u64,
    accepted: u64,
    /// The request's acceptance per draft position at the lanes' two widths (speculative lanes
    /// with a width of their own; it only estimates what a wider step would bring).
    policy: Option<spec::WidthPolicy>,
}

impl Run {
    /// The lane's request ends after this token (or before the next one).
    fn done(&self) -> bool {
        crate::core::lanes::run_done(
            self.stopped_by_eos,
            self.stopped_by_cancel,
            self.budget.remaining(),
        )
    }

    /// After a step that wrote `rows` rows at the cursor and emitted `emitted` tokens: the
    /// cursor and the recurrent-state parity of the next step (the verified transition), and the
    /// step count of the report.
    fn advance(&mut self, emitted: u64, rows: u64) {
        self.cursor = crate::verified::ln_lane_cursor_next(self.cursor, emitted, rows);
        self.parity = crate::verified::ln_lane_parity_next(self.parity as u64) as usize;
        self.steps += 1;
    }

    /// Notes a cancellation of the request; whether it ends is then [`Run::done`]'s decision.
    fn observe_cancel(&mut self) {
        self.stopped_by_cancel |= self.cancel.load(Ordering::Relaxed);
    }

    /// Takes one generated token as the single-stream decode loop does.
    fn emit(&mut self, tok: i32) {
        if self.first_pending {
            self.first_pending = false;
            self.ttft_ms = self.t_start.elapsed().as_secs_f64() * 1000.0;
            self.t_decode = Some(Instant::now());
        }
        self.budget.record_emitted(1);
        self.generated.push(tok);
        (self.out)(LaneOut::Token(tok));
        self.cur_tok = tok;
        if self.params.stop_on_eos && self.eos.contains(&tok) {
            self.stopped_by_eos = true;
        }
    }
}

/// The view of the lanes `/slots` shows.
fn view(rt: &LaneRt, runs: &[Option<Run>]) -> Vec<LaneView> {
    (0..rt.n)
        .map(|i| match &runs[i] {
            Some(r) => LaneView {
                state: rt.table.state_name(i),
                job: Some(r.id),
                prompt_tokens: r.prompt_len,
                generated: r.generated.len() as u64,
                reused: r.reused,
                busy: rt.table.is_busy(i),
            },
            None => LaneView {
                state: rt.table.state_name(i),
                busy: rt.table.is_busy(i),
                ..LaneView::default()
            },
        })
        .collect()
}

/// What lane `lane` offers a request of `prompt` (routing).
fn lane_reuse(eng: &Qwen3Engine, lane: usize, prompt: &[i32], params: &Qwen3GenParams) -> Offer {
    let rt = eng.lanes.as_ref().expect("lanes");
    let (prefix, reset) = if rt.active == lane {
        (&eng.prefix, &eng.reset)
    } else {
        (&rt.bufs[lane].prefix, &rt.bufs[lane].reset)
    };
    let p = prefix.peek(prompt, &[], params.cache_prompt, reset.zeroes_rows());
    let held = prefix.held().0.len() as u64;
    Offer {
        resume: p.resume,
        shared: p.cap,
        loss: crate::verified::ln_route_loss(held, p.cap),
    }
}

fn done_event(report: Result<Qwen3GenReport, String>, eng: &Qwen3Engine, images: bool) -> LaneOut {
    LaneOut::Done(Box::new(LaneDone {
        report,
        vision: if images {
            eng.last_vision_report().cloned()
        } else {
            None
        },
        prefix: eng.prefix_stats(),
        host_tier: None,
    }))
}

/// Admits `job` into the free lane `lane` and runs its prefill. `Ok(None)`: the request already
/// ended (its `Done` is sent).
fn start_lane(eng: &mut Qwen3Engine, lane: usize, job: LaneJob) -> Result<Option<Run>, String> {
    let exclusive = job.exclusive();
    let LaneJob {
        id,
        prompt,
        params,
        images,
        cancel,
        mut out,
    } = job;
    activate(eng, lane);
    out(LaneOut::Started { lane });
    let fail = |eng: &mut Qwen3Engine, out: &mut Box<dyn FnMut(LaneOut) + Send>, e: String| {
        let res: Result<Qwen3GenReport, String> = Err(e);
        request_reset::end_request(eng, prompt.len() as u64, &res);
        eng.prefix.request_end(false, 0);
        out(done_event(res, eng, exclusive));
    };
    if prompt.is_empty() {
        out(done_event(
            Err("prompt must contain at least 1 token".into()),
            eng,
            false,
        ));
        return Ok(None);
    }
    let mut budget =
        match SequenceBudget::admit(prompt.len() as u64, Some(params.max_new), eng.max_ctx) {
            Ok(b) => b,
            Err(e) => {
                out(done_event(
                    Err(format!("SequenceBudget rejected prompt/max_new: {e:?}")),
                    eng,
                    false,
                ));
                return Ok(None);
            }
        };
    let sampled = params.temperature > 0.0;
    let can_sample = eng.lanes.as_ref().is_some_and(|rt| rt.sampled);
    if !crate::verified::sampling_request_admitted(can_sample, sampled) {
        out(done_event(
            Err("a sampled request in a lane needs the on-card sampler, which this model or kernel directory lacks".into()),
            eng,
            false,
        ));
        return Ok(None);
    }
    let mut eos = eng.spec.eos_token_ids.clone();
    for &tok in &params.extra_eos_ids {
        if !eos.contains(&tok) {
            eos.push(tok);
        }
    }
    if let Some(im) = images.filter(|v| !v.is_empty()) {
        if let Err(e) = eng.set_request_images(im) {
            out(done_event(Err(e), eng, false));
            return Ok(None);
        }
    }
    // the lane's sampler parameters, read by its slot of a sampled step
    if sampled {
        let d = samp_rt::dev_params(&params, eng.spec.vocab);
        let (af, ai) = eng.lanes.as_ref().expect("lanes").samp_bufs[lane];
        let buf = eng.stage_io.as_mut_slice();
        for (i, v) in d.f.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in d.i.iter().enumerate() {
            buf[64 + i * 4..64 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        eng.dev
            .copy_h2d_pinned_batch_sync(&eng.stage_io, &[(0, 32, af), (64, 32, ai)])
            .map_err(|e| format!("write lane sampler parameters H2D: {e}"))?;
    }
    let wprec = if eng.spec.fp8_weights { "FP8" } else { "BF16" };
    let label = if sampled {
        format!("K=0 {wprec} T={:.2}", params.temperature)
    } else {
        format!("K=0 {wprec} greedy")
    };
    // the live record follows the lane session: it begins with the first busy lane (a request
    // admitted beside others must not reset what they report) and ends with the last
    if live_alone(eng) {
        eng.live.begin_request(prompt.len() as u64, &label, "");
        eng.live.prefill_progress(0);
    }
    let t_start = Instant::now();
    let is_cancelled = || cancel.load(Ordering::Relaxed);
    // (a one-row prompt runs as the single-stream step, bound to this lane)
    let pre = eng.run_prefill(&prompt, &params, &mut budget, &is_cancelled, false);
    let mut run = Run {
        id,
        params,
        cancel: Arc::clone(&cancel),
        out,
        exclusive,
        prompt_len: prompt.len() as u64,
        budget,
        eos,
        cursor: 0,
        cur_tok: prompt[0],
        parity: 0,
        first_pending: false,
        sampled,
        generated: Vec::new(),
        stopped_by_eos: false,
        stopped_by_cancel: false,
        t_start,
        ttft_ms: 0.0,
        prefill_tok_s: 0.0,
        ttft_recipe_ms: 0.0,
        t_decode: None,
        steps: 0,
        reused: eng.prefix_plan.resume,
        drafted: 0,
        accepted: 0,
        policy: match eng.lanes.as_ref().map(|rt| rt.family.rows()) {
            Some((narrow, wide)) if wide > narrow => {
                Some(spec::WidthPolicy::new(&[narrow - 1, wide - 1]))
            }
            _ => None,
        },
    };
    match pre {
        Err(e) => {
            if live_alone(eng) {
                eng.live.end_request();
            }
            fail(eng, &mut run.out, e);
            Ok(None)
        }
        Ok(Prefilled::Cancelled(rep)) => {
            run.stopped_by_cancel = true;
            run.ttft_ms = rep.ttft_ms;
            Ok(Some(run))
        }
        Ok(Prefilled::OneRow { .. }) if eng.specdec.is_some() => {
            // (`run_prefill` defers a one-row prompt only when asked to, which lanes never do: a
            // speculative lane's step buffers are written after its first token)
            eng.live.end_request();
            fail(
                eng,
                &mut run.out,
                "internal: a speculative lane got a deferred one-row prompt".into(),
            );
            Ok(None)
        }
        Ok(Prefilled::OneRow {
            gdn_parity,
            ttft_recipe_ms,
        }) => {
            run.parity = gdn_parity;
            run.ttft_recipe_ms = ttft_recipe_ms;
            run.first_pending = true;
            Ok(Some(run))
        }
        Ok(Prefilled::First(f)) => {
            run.parity = f.gdn_parity;
            run.cursor = run.prompt_len;
            run.ttft_ms = f.ttft_ms;
            run.prefill_tok_s = f.prefill_tok_s;
            run.ttft_recipe_ms = f.ttft_recipe_ms;
            run.t_decode = Some(Instant::now());
            run.observe_cancel();
            if !run.done() {
                eng.live.first_token(Some(f.ttft_ms), 1);
                run.emit(f.first_tok);
            }
            if !run.done() && eng.specdec.is_some() {
                // a speculative lane: the first block of drafts, and the lane's step buffers
                // of this parity (row 0 = the first token, at the cursor)
                // (the first step's ids at the widest rows a lane may verify at)
                let t = eng.lanes.as_ref().expect("lanes").family.rows().1 as usize;
                let drafts = {
                    let Qwen3Engine {
                        specdec,
                        dev,
                        stage_io,
                        ..
                    } = eng;
                    specdec.as_mut().expect("spec runtime").lane_block0(
                        dev,
                        stage_io,
                        f.first_tok,
                        run.cursor,
                        run.sampled,
                    )?
                };
                let mut ids = vec![f.first_tok];
                ids.extend_from_slice(&drafts);
                ids.resize(t, 0);
                let sd = eng.specdec.as_ref().expect("spec runtime");
                let p = run.parity;
                let (a, c, i) = (
                    sd.step_addr(&format!("sp_a{p}")),
                    sd.step_addr(&format!("sp_cur{p}")),
                    sd.step_addr(&format!("sp_ids{p}")),
                );
                let cur0 = [run.cursor as i32];
                let mut writes: Vec<(u64, &[i32])> = vec![(a, &[0]), (c, &cur0), (i, &ids)];
                // the replay form: the lane's first step replays nothing (its cursor is the
                // first-step cursor)
                if let Some(r) = sd.lane_rp_first() {
                    writes.push((r, &cur0));
                }
                spec::write_i32s(eng, &writes)?;
            }
            Ok(Some(run))
        }
    }
}

/// The draft widths a speculative lane verifies at, as the method label shows them (`7` or
/// `7/15`).
fn lane_ks(g: &spec::lanes::SpecLaneGraph) -> String {
    if g.wide == g.narrow {
        format!("{}", g.narrow - 1)
    } else {
        format!("{}/{}", g.narrow - 1, g.wide - 1)
    }
}

/// Is the lane being started or ended the only busy one (the live record's lifecycle)?
fn live_alone(eng: &Qwen3Engine) -> bool {
    eng.lanes.as_ref().is_none_or(|rt| rt.table.busy() <= 1)
}

/// Ends lane `lane`'s request: its report goes out and the lane keeps what it wrote.
fn end_lane(eng: &mut Qwen3Engine, lane: usize, mut run: Run) {
    activate(eng, lane);
    let decode_ms = run
        .t_decode
        .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
    let wprec = if eng.spec.fp8_weights { "FP8" } else { "BF16" };
    let n_dec = run.generated.len().saturating_sub(1) as u64;
    let report = Qwen3GenReport {
        prompt_tokens: run.prompt_len as usize,
        generated_tokens: std::mem::take(&mut run.generated),
        ttft_ms: run.ttft_ms,
        prefill_tok_s: run.prefill_tok_s,
        decode_ms,
        decode_tok_s: if run.steps > 0 {
            n_dec as f64 / (decode_ms / 1000.0).max(1e-9)
        } else {
            0.0
        },
        ms_per_step: if run.steps > 0 {
            decode_ms / run.steps as f64
        } else {
            0.0
        },
        stopped_by_eos: run.stopped_by_eos,
        stopped_by_cancel: run.stopped_by_cancel,
        launches: run.steps,
        tokens_per_launch: if run.steps > 0 {
            n_dec as f64 / run.steps as f64
        } else {
            0.0
        },
        drafted: run.drafted,
        draft_accepted: run.accepted,
        method: match (eng.lanes.as_ref().map(|rt| &rt.family), run.sampled) {
            (Some(LaneFamily::Spec(g)), true) => format!(
                "dspine K={} {wprec} T={:.2}",
                lane_ks(g),
                run.params.temperature
            ),
            (Some(LaneFamily::Spec(g)), false) => format!("dspine K={} {wprec} greedy", lane_ks(g)),
            (_, true) => format!("K=0 {wprec} T={:.2}", run.params.temperature),
            (_, false) => format!("K=0 {wprec} greedy"),
        },
        ttft_recipe_ms: run.ttft_recipe_ms,
        ..Qwen3GenReport::default()
    };
    let res = Ok(report);
    request_reset::end_request(eng, run.prompt_len, &res);
    let written_end = res.as_ref().map_or(0, |r| {
        request_reset::request_high_water(
            run.prompt_len,
            r.generated_tokens.len() as u64,
            eng.max_ctx,
        )
    });
    eng.prefix.request_end(true, written_end);
    let (reused, common) = (eng.prefix_plan.resume, eng.prefix_plan.lcp);
    let res = res.map(|mut r| {
        r.reused_tokens = reused;
        r.prefix_common = common;
        r
    });
    if live_alone(eng) {
        eng.live.end_request();
    }
    let ev = done_event(res, eng, run.exclusive);
    (run.out)(ev);
}

/// The single-row step of a one-row prompt on the active lane (its caches and states bound at
/// the launch). Returns the ms a first use of the sampled step took.
pub(super) fn launch_one_row(
    eng: &mut Qwen3Engine,
    parity: usize,
    sampled: bool,
) -> Result<f64, String> {
    let mut pool = eng.step_base.clone();
    for (k, &(v, _)) in &eng.kv_addr {
        pool.insert(k.clone(), v);
    }
    let next = crate::verified::ln_lane_parity_next(parity as u64) as usize;
    for (&l, st) in &eng.gdn_state {
        pool.insert(format!("L{l}_S_in"), st.s[parity]);
        pool.insert(format!("L{l}_S_out"), st.s[next]);
        pool.insert(format!("L{l}_cs_in"), st.cs[parity]);
        pool.insert(format!("L{l}_cs_out"), st.cs[next]);
    }
    if sampled {
        let ms = samp_rt::ensure_step(eng, 1)?;
        samp_rt::launch_step_with(eng, 1, &pool)?;
        return Ok(ms);
    }
    let r = &eng.step_recipes[0];
    let exact = select_exact_bindings(r, &pool)?;
    r.launch_checked(&eng.dev, &exact)
        .map_err(|e| format!("launch step (prompt len 1, lane): {e}"))?;
    Ok(0.0)
}

/// Makes the batched step `key` launchable; the ms a first use took.
fn ensure_recipe(eng: &mut Qwen3Engine, key: &StepKey) -> Result<f64, String> {
    if eng
        .lanes
        .as_ref()
        .is_some_and(|rt| rt.recipes.contains_key(key))
    {
        return Ok(0.0);
    }
    let t0 = Instant::now();
    let Qwen3Engine {
        lanes,
        dev,
        spec,
        cache,
        driver_version,
        max_ctx,
        ..
    } = eng;
    let rt = lanes.as_mut().expect("lanes");
    let off = RecipeCache::disabled();
    let cache = if rt.fresh { &off } else { &*cache };
    let mut r = rt
        .family
        .graph(spec, *max_ctx, key)
        .compile(Arc::clone(dev.api()), cache, driver_version)
        .map_err(|e| format!("compile batched step {}: {e}", rt.family.name(key)))?;
    r.bind_shared_workspace(dev)
        .map_err(|e| format!("bind {} ws: {e}", r.name))?;
    if eng.verbose {
        eprintln!(
            "[gaudium] lanes: recipe {} was not ready: the step waited {:.1} ms for it",
            r.name,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    rt.recipes.insert(key.clone(), r);
    Ok(t0.elapsed().as_secs_f64() * 1000.0)
}

/// The key of the prepared launch of batched step `key` with `lanes[j]` (state parity
/// `parities[j]`) in slot `j`.
fn launch_key(key: &StepKey, lanes: &[usize], parities: &[usize]) -> LaunchKey {
    (
        key.clone(),
        lanes.iter().map(|&l| l as u8).collect(),
        parities.iter().map(|&p| p as u8).collect(),
    )
}

/// The prepared launch of batched step `key` with `lanes[j]` (state parity `parities[j]`) in
/// slot `j`.
fn prepare(
    eng: &mut Qwen3Engine,
    key: &StepKey,
    lanes: &[usize],
    parities: &[usize],
) -> Result<(), String> {
    let lk = launch_key(key, lanes, parities);
    if eng
        .lanes
        .as_ref()
        .is_some_and(|rt| rt.launches.contains_key(&lk))
    {
        return Ok(());
    }
    let rt = eng.lanes.as_ref().expect("lanes");
    let mut pool = rt.base.clone();
    let spec_family = matches!(rt.family, LaneFamily::Spec(_));
    let replay = matches!(&rt.family, LaneFamily::Spec(g) if spec::replay::enabled(&g.hashes));
    for (j, (&lane, &p)) in lanes.iter().zip(parities).enumerate() {
        let sp = slot_pfx(j as u64);
        // the state buffer the step writes: the one its lane's next step reads
        let q = crate::verified::ln_lane_parity_next(p as u64) as usize;
        pool.insert(format!("{sp}samp_f"), rt.samp_bufs[lane].0);
        pool.insert(format!("{sp}samp_i"), rt.samp_bufs[lane].1);
        let (kv, gdn) = if rt.active == lane {
            (&eng.kv_addr, &eng.gdn_state)
        } else {
            (&rt.bufs[lane].kv_addr, &rt.bufs[lane].gdn_state)
        };
        for (name, &(addr, _)) in kv {
            pool.insert(format!("{sp}{name}"), addr);
        }
        if spec_family {
            // the lane's step buffers alternate by parity: read `p`, write `q`
            let own = |name: &str| -> u64 {
                if rt.active == lane {
                    eng.specdec.as_ref().expect("spec runtime").step_addr(name)
                } else {
                    rt.bufs[lane].spec.as_ref().expect("lane spec buffers").addr[name]
                }
            };
            for (slot_name, own_name) in [
                ("ids", format!("sp_ids{p}")),
                ("next_ids", format!("sp_ids{q}")),
                ("cursor", format!("sp_cur{p}")),
                ("cursor_out", format!("sp_cur{q}")),
                ("a_in", format!("sp_a{p}")),
                ("a_out", format!("sp_a{q}")),
                ("emitted", format!("sp_emit{p}")),
                ("df_nofix", "df_nofix".to_string()),
            ] {
                pool.insert(format!("{sp}{slot_name}"), own(&own_name));
            }
            let drafter_caches: Vec<(String, u64)> = if rt.active == lane {
                let sd = eng.specdec.as_ref().expect("spec runtime");
                sd.shared_addrs()
                    .iter()
                    .filter(|(n, _)| spec::dspine::is_drafter_kv(n))
                    .map(|(n, &a)| (n.clone(), a))
                    .collect()
            } else {
                rt.bufs[lane]
                    .spec
                    .as_ref()
                    .expect("lane spec buffers")
                    .addr
                    .iter()
                    .filter(|(n, _)| spec::dspine::is_drafter_kv(n))
                    .map(|(n, &a)| (n.clone(), a))
                    .collect()
            };
            for (n, a) in drafter_caches {
                pool.insert(format!("{sp}{n}"), a);
            }
            for (&l, st) in gdn {
                pool.insert(format!("{sp}L{l}_S_in"), st.s[p]);
                pool.insert(format!("{sp}L{l}_S_ckpt"), st.s[q]);
                pool.insert(format!("{sp}L{l}_cs_in"), st.cs[p]);
                pool.insert(format!("{sp}L{l}_cs_ckpt"), st.cs[q]);
            }
            if replay {
                // the replay form: the state the step starts from is written once (`S_base`),
                // the update buffers alternate by parity like the rest of the lane's step state
                pool.insert(format!("{sp}rp_first"), own("sp_rp_first"));
                for &l in gdn.keys() {
                    pool.insert(format!("{sp}L{l}_S_base"), gdn[&l].s[q]);
                    for what in ["rpk", "rpd", "rpg"] {
                        pool.insert(format!("{sp}L{l}_{what}_in"), own(&format!("sp_{what}{p}_L{l}")));
                        pool.insert(format!("{sp}L{l}_{what}_out"), own(&format!("sp_{what}{q}_L{l}")));
                    }
                }
            }
        } else {
            pool.insert(format!("{sp}cursor"), rt.io_cursor[j]);
            for (&l, st) in gdn {
                pool.insert(format!("{sp}L{l}_S_in"), st.s[p]);
                pool.insert(format!("{sp}L{l}_S_out"), st.s[q]);
                pool.insert(format!("{sp}L{l}_cs_in"), st.cs[p]);
                pool.insert(format!("{sp}L{l}_cs_out"), st.cs[q]);
            }
        }
    }
    let recipe = &rt.recipes[key];
    let exact = select_exact_bindings(recipe, &pool)?;
    let prepared = recipe
        .prepare_launch(&exact)
        .map_err(|e| format!("prepare {}: {e}", recipe.name))?;
    let rt = eng.lanes.as_mut().expect("lanes");
    if rt.launches.len() >= LAUNCH_CACHE {
        rt.launches.clear();
    }
    rt.launches.insert(lk, prepared);
    Ok(())
}

/// One batched step of `lanes` (all of one readout kind): every lane's next token.
fn step(
    eng: &mut Qwen3Engine,
    lanes: &[usize],
    runs: &[Option<Run>],
    sampled: bool,
) -> Result<Vec<i32>, String> {
    let t0 = eng.lanes.as_mut().expect("lanes").timing.begin();
    let b = lanes.len() as u64;
    let slots: Vec<(u64, u64)> = lanes
        .iter()
        .map(|&l| (runs[l].as_ref().expect("a decoding lane runs").cursor, 1))
        .collect();
    let nch = step_chunks(batch_chunks(&slots, KV_CHUNK), eng.max_chunks);
    for (j, &(c, r)) in slots.iter().enumerate() {
        if !slot_fits(c, r, nch, KV_CHUNK, eng.max_chunks) {
            return Err(format!(
                "batched step: slot {j} (lane {}, rows to {}) does not fit {nch} chunks of a lane of {} chunks",
                lanes[j],
                c + r,
                eng.max_chunks
            ));
        }
    }
    let key: StepKey = (vec![1; lanes.len()], nch, sampled);
    ensure_recipe(eng, &key)?;
    let parities: Vec<usize> = lanes
        .iter()
        .map(|&l| runs[l].as_ref().expect("a decoding lane runs").parity)
        .collect();
    prepare(eng, &key, lanes, &parities)?;
    // ids and cursors in one synchronised upload
    let (io_ids, io_tokens, io_cursor) = {
        let rt = eng.lanes.as_ref().expect("lanes");
        (rt.io_ids, rt.io_tokens, rt.io_cursor.clone())
    };
    let mut copies = vec![(0usize, 4 * lanes.len(), io_ids)];
    {
        // token ids first, then one block per slot's cursor (`ln_stage_cursor`: never inside
        // the ids, never overlapping each other)
        let last = crate::core::lanes::stage_cursor(b, b.saturating_sub(1)) as usize + 4;
        if last > eng.stage_io.len() {
            return Err(format!(
                "batched step: the inputs of {b} slots need {last} staging bytes, the buffer has {}",
                eng.stage_io.len()
            ));
        }
        let buf = eng.stage_io.as_mut_slice();
        for (j, &l) in lanes.iter().enumerate() {
            let r = runs[l].as_ref().expect("a decoding lane runs");
            buf[4 * j..4 * j + 4].copy_from_slice(&r.cur_tok.to_le_bytes());
            let off = crate::core::lanes::stage_cursor(b, j as u64) as usize;
            buf[off..off + 4].copy_from_slice(&(r.cursor as i32).to_le_bytes());
            copies.push((off, 4, io_cursor[j]));
        }
    }
    eng.dev
        .copy_h2d_pinned_batch_sync(&eng.stage_io, &copies)
        .map_err(|e| format!("write batched step inputs H2D: {e}"))?;
    let lk = launch_key(&key, lanes, &parities);
    lane_launch_hook(eng)?;
    {
        let rt = eng.lanes.as_ref().expect("lanes");
        rt.recipes[&key]
            .launch_prepared(&eng.dev, &rt.launches[&lk])
            .map_err(|e| format!("launch {}: {e}", batched_step_name(b, nch, sampled)))?;
    }
    let launched = Instant::now();
    // SAFETY: the compute stream belongs to `eng.dev`.
    unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
        .map_err(|e| format!("sync {}: {e}", batched_step_name(b, nch, sampled)))?;
    let synced = Instant::now();
    eng.dev
        .copy_d2h_pinned_sync(io_tokens, &mut eng.stage_io, 4 * lanes.len())
        .map_err(|e| format!("read batched step tokens D2H: {e}"))?;
    let raw = &eng.stage_io.as_slice()[..4 * lanes.len()];
    let toks = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    eng.lanes.as_mut().expect("lanes").timing.end(t0, launched, synced);
    Ok(toks)
}

/// Before every batched launch: the device-trace windows of `--profile-steps` (by the count of
/// batched launches since the load).
fn lane_launch_hook(eng: &mut Qwen3Engine) -> Result<(), String> {
    let idx = {
        let rt = eng.lanes.as_mut().expect("lanes");
        rt.launched += 1;
        rt.launched - 1
    };
    prof_rt::before_lane_launch(eng, idx)
}

/// A batched speculative launch on the device, not collected yet.
struct LaneLaunch {
    /// `(lane, rows)` of every slot, widest first.
    lanes: Vec<(usize, u64)>,
    /// The state parity every slot's lane read (its emitted tokens are in that parity's buffer).
    parities: Vec<usize>,
    nch: u64,
    /// Its completion event (`LaneRt::events`).
    event: usize,
    t0: Instant,
    launched: Instant,
}

/// Launches a batched speculative step of `lanes` (`(lane, rows)`, widest first; all of one
/// readout kind) whose slots read state parity `parities[j]` at a cursor of at most
/// `cursors[j]` (the slot table is checked for those rows).
fn spec_launch(
    eng: &mut Qwen3Engine,
    lanes: &[(usize, u64)],
    cursors: &[u64],
    parities: &[usize],
    sampled: bool,
) -> Result<LaneLaunch, String> {
    let t0 = eng.lanes.as_mut().expect("lanes").timing.begin();
    let slots: Vec<(u64, u64)> = lanes.iter().zip(cursors).map(|(&(_, t), &c)| (c, t)).collect();
    let nch = step_chunks(batch_chunks(&slots, KV_CHUNK), eng.max_chunks);
    for (j, &(c, r)) in slots.iter().enumerate() {
        if !slot_fits(c, r, nch, KV_CHUNK, eng.max_chunks) {
            return Err(format!(
                "batched speculative step: slot {j} (lane {}, rows to {}) does not fit {nch} chunks of a lane of {} chunks",
                lanes[j].0,
                c + r,
                eng.max_chunks
            ));
        }
    }
    let key: StepKey = (lanes.iter().map(|&(_, t)| t).collect(), nch, sampled);
    ensure_recipe(eng, &key)?;
    let ids: Vec<usize> = lanes.iter().map(|&(l, _)| l).collect();
    prepare(eng, &key, &ids, parities)?;
    let lk = launch_key(&key, &ids, parities);
    lane_launch_hook(eng)?;
    let rt = eng.lanes.as_mut().expect("lanes");
    let event = rt.next_event;
    // (the start event fires when the launch starts on the device: behind a launch queued
    // before it, when that one ends)
    if let Some((ev, _)) = rt.events.get(event) {
        ev.record_compute(&eng.dev)
            .map_err(|e| format!("record a lane step event: {e}"))?;
    }
    rt.recipes[&key]
        .launch_prepared(&eng.dev, &rt.launches[&lk])
        .map_err(|e| format!("launch {}: {e}", rt.family.name(&key)))?;
    if let Some((_, ev)) = rt.events.get(event) {
        ev.record_compute(&eng.dev)
            .map_err(|e| format!("record a lane step event: {e}"))?;
    }
    rt.next_event = (event + 1) % rt.events.len().max(1);
    Ok(LaneLaunch {
        lanes: lanes.to_vec(),
        parities: parities.to_vec(),
        nch,
        event,
        t0,
        launched: Instant::now(),
    })
}

/// Waits for launch `l` and reads every slot's emitted tokens (its accepted drafts, then the
/// bonus token; a negative value ends the list), in the order of its lanes.
fn spec_collect(eng: &mut Qwen3Engine, l: &LaneLaunch) -> Result<Vec<Vec<i32>>, String> {
    {
        let rt = eng.lanes.as_ref().expect("lanes");
        match rt.events.get(l.event) {
            Some((_, ev)) => ev
                .synchronize()
                .map_err(|e| format!("wait for a batched speculative step: {e}"))?,
            // SAFETY: the compute stream belongs to `eng.dev`.
            None => unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
                .map_err(|e| format!("sync batched speculative step: {e}"))?,
        }
    }
    let synced = Instant::now();
    let mut out = Vec::with_capacity(l.lanes.len());
    for (&(lane, t), &p) in l.lanes.iter().zip(&l.parities) {
        let name = format!("sp_emit{p}");
        let addr = {
            let rt = eng.lanes.as_ref().expect("lanes");
            if rt.active == lane {
                eng.specdec.as_ref().expect("spec runtime").step_addr(&name)
            } else {
                rt.bufs[lane].spec.as_ref().expect("lane spec buffers").addr[&name]
            }
        };
        out.push(spec::read_i32s(eng, addr, t as usize)?);
    }
    let rt = eng.lanes.as_mut().expect("lanes");
    rt.timing.end(l.t0, l.launched, synced);
    // the launch's own device time, for the lanes' width choices
    if let Some((a, b)) = rt.events.get(l.event) {
        if let Ok(ns) = crate::synapse::device::DeviceEvent::elapsed_ns(a, b) {
            let wide = l.lanes.iter().filter(|&&(_, t)| t > rt.family.rows().0).count() as u64;
            rt.widths.observe(l.lanes.len() as u64, l.nch, wide, ns as f64 / 1e6);
        }
    }
    Ok(out)
}

/// One step of the speculative lanes `group` (all of one readout kind), and the next one queued
/// ahead of it when the verified rule allows (`waiting`: a request waits for a lane; `one_group`:
/// the group is every decoding lane): a launch queued in the previous call (`ahead`) is this step
/// when it has exactly this group's lanes. A lane whose request ended in a step drops the step queued
/// behind it, which is collected before the lane is released.
fn spec_group_step(
    eng: &mut Qwen3Engine,
    runs: &mut [Option<Run>],
    group: &[usize],
    sampled: bool,
    waiting: bool,
    one_group: bool,
    ahead: &mut Option<LaneLaunch>,
) -> Result<(), String> {
    // a lane without room in its context for a batched step's rows takes single-row steps to
    // the end of its request, as a single stream does
    let (narrow, _) = eng.lanes.as_ref().expect("lanes").family.rows();
    let max_ctx = eng.max_ctx;
    let parity = |runs: &[Option<Run>], l: usize| runs[l].as_ref().expect("a decoding lane runs").parity;
    if let Some(a) = ahead.take() {
        // the step in flight: queue the one after it (same lanes, at most its rows further on,
        // the other state parity) unless the group changed or a lane may leave the batched steps
        let mut lanes_a: Vec<usize> = a.lanes.iter().map(|&(l, _)| l).collect();
        lanes_a.sort_unstable();
        let mut lanes_g = group.to_vec();
        lanes_g.sort_unstable();
        let bound: BTreeMap<usize, u64> = a
            .lanes
            .iter()
            .map(|&(l, t)| (l, runs[l].as_ref().expect("a decoding lane runs").cursor + t))
            .collect();
        let next = if crate::core::lanes::queue_ahead(
            waiting,
            one_group && lanes_a == lanes_g,
            bound.values().all(|&c| spec_batched(c, narrow, max_ctx)),
        ) {
            let lanes = a.lanes.iter().map(|&(l, _)| l).collect::<Vec<_>>();
            let with_rows = lane_rows_of(eng, &lanes, runs, &|l| bound[&l]);
            let cursors: Vec<u64> = with_rows.iter().map(|&(l, _)| bound[&l]).collect();
            let parities: Vec<usize> = with_rows
                .iter()
                .map(|&(l, _)| {
                    let p = a.parities[a.lanes.iter().position(|&(x, _)| x == l).expect("its lane")];
                    crate::verified::ln_lane_parity_next(p as u64) as usize
                })
                .collect();
            Some(spec_launch(eng, &with_rows, &cursors, &parities, sampled)?)
        } else {
            None
        };
        let em = spec_collect(eng, &a)?;
        apply_spec(eng, runs, a.lanes.iter().zip(em).map(|(&(l, t), e)| (l, t, e)).collect())?;
        if let Some(b) = next {
            let ended = b.lanes.iter().any(|&(l, _)| runs[l].as_ref().is_none_or(|r| r.done()));
            if crate::core::lanes::ahead_lands(ended) {
                // a request ended: the step behind it lands before the lane is released
                let em = spec_collect(eng, &b)?;
                let live = b
                    .lanes
                    .iter()
                    .zip(em)
                    .filter(|((l, _), _)| runs[*l].as_ref().is_some_and(|r| !r.done()))
                    .map(|(&(l, t), e)| (l, t, e))
                    .collect();
                apply_spec(eng, runs, live)?;
            } else {
                *ahead = Some(b);
            }
        }
        return Ok(());
    }
    let (stepping, single): (Vec<usize>, Vec<usize>) = group.iter().copied().partition(|&l| {
        let run = runs[l].as_ref().expect("a decoding lane runs");
        spec_batched(run.cursor, narrow, max_ctx)
    });
    let mut emitted: Vec<(usize, u64, Vec<i32>)> = Vec::with_capacity(group.len());
    if !stepping.is_empty() {
        let with_rows = lane_rows_of(eng, &stepping, runs, &|l| {
            runs[l].as_ref().expect("a decoding lane runs").cursor
        });
        let cursors: Vec<u64> = with_rows
            .iter()
            .map(|&(l, _)| runs[l].as_ref().expect("a decoding lane runs").cursor)
            .collect();
        let parities: Vec<usize> = with_rows.iter().map(|&(l, _)| parity(runs, l)).collect();
        let a = spec_launch(eng, &with_rows, &cursors, &parities, sampled)?;
        // (every lane's rows of this step fit its context: it is in `stepping`)
        if crate::core::lanes::queue_ahead(waiting, one_group && single.is_empty(), true) {
            // collected (and the next step queued) by the next call
            *ahead = Some(a);
            return Ok(());
        }
        let em = spec_collect(eng, &a)?;
        emitted.extend(with_rows.iter().zip(em).map(|(&(l, t), e)| (l, t, e)));
    }
    for &lane in &single {
        emitted.push((lane, 1, single_row_step(eng, lane, runs)?));
    }
    apply_spec(eng, runs, emitted)
}

/// A speculative step's results: per lane `(lane, rows it verified, emitted tokens)`.
fn apply_spec(
    eng: &mut Qwen3Engine,
    runs: &mut [Option<Run>],
    emitted: Vec<(usize, u64, Vec<i32>)>,
) -> Result<(), String> {
    for (lane, rows, em) in emitted {
        let run = runs[lane].as_mut().expect("a decoding lane runs");
        let n_emit = em.iter().take_while(|&&v| v >= 0).count();
        if n_emit == 0 {
            return Err(format!(
                "lane {lane}: the accept kernel emitted no token at cursor {} ({em:?})",
                run.cursor
            ));
        }
        run.advance(n_emit as u64, rows);
        run.drafted += rows - 1;
        run.accepted += n_emit as u64 - 1;
        if rows > 1 {
            if let Some(p) = run.policy.as_mut() {
                p.observe(rows - 1, n_emit - 1, 0.0);
            }
        }
        run.observe_cancel();
        for &tok in &em[..n_emit] {
            if run.done() {
                break;
            }
            run.emit(tok);
        }
        if run.done() {
            eng.lanes
                .as_mut()
                .expect("lanes")
                .table
                .finish(lane)
                .map_err(|e| format!("lanes: {e}"))?;
        }
    }
    Ok(())
}

/// The rows each of the speculative lanes `stepping` verifies in its next batched step, widest
/// first: a lane goes wide when the extra tokens it expects from the wide step beat the batch's
/// expected tokens per unit of time over the wide step's extra time (the verified rule), and its
/// wide rows fit its rings and its context.
/// `cursor_of(lane)`: the lane's cursor, or a bound of it for a step queued ahead.
fn lane_rows_of(
    eng: &Qwen3Engine,
    stepping: &[usize],
    runs: &[Option<Run>],
    cursor_of: &dyn Fn(usize) -> u64,
) -> Vec<(usize, u64)> {
    let rt = eng.lanes.as_ref().expect("lanes");
    let (narrow, wide) = rt.family.rows();
    let ring = eng.specdec.as_ref().map_or(narrow, |sd| sd.gdn_slots());
    let run = |l: usize| runs[l].as_ref().expect("a decoding lane runs");
    // the time of this step with every lane narrow, and of one more lane wide
    let narrow_slots: Vec<(u64, u64)> = stepping.iter().map(|&l| (cursor_of(l), narrow)).collect();
    let nch = step_chunks(batch_chunks(&narrow_slots, KV_CHUNK), eng.max_chunks);
    let est = (wide > narrow)
        .then(|| rt.widths.estimate(stepping.len() as u64, nch))
        .flatten();
    // expected tokens per lane at each width, in thousandths
    let milli = |x: f64| (x.max(0.0) * 1000.0).round() as u64;
    let expect = |l: usize, rows: u64| run(l).policy.as_ref().map_or(0.0, |p| p.expected_tokens(rows - 1));
    let total: u64 = stepping.iter().map(|&l| milli(expect(l, narrow))).sum();
    let mut out: Vec<(usize, u64)> = stepping
        .iter()
        .map(|&l| {
            let go = est.is_some_and(|(t0, extra)| {
                let gain = milli(expect(l, wide)).saturating_sub(milli(expect(l, narrow)));
                crate::core::lanes::lane_goes_wide(gain, milli(extra), total, milli(t0))
            });
            let rows = crate::core::lanes::lane_step_rows(go, narrow, wide, ring, cursor_of(l), eng.max_ctx);
            (l, rows)
        })
        .collect();
    // (the slots of a step are ordered widest first: one recipe per row combination)
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}

/// One single-row step of speculative lane `lane` (near the end of its context): its emitted
/// tokens.
fn single_row_step(
    eng: &mut Qwen3Engine,
    lane: usize,
    runs: &[Option<Run>],
) -> Result<Vec<i32>, String> {
    let run = runs[lane].as_ref().expect("a decoding lane runs");
    let nch = step_chunks(batch_chunks(&[(run.cursor, 1)], KV_CHUNK), eng.max_chunks);
    if !slot_fits(run.cursor, 1, nch, KV_CHUNK, eng.max_chunks) {
        return Err(format!(
            "single-row step: lane {lane} (row {}) does not fit {nch} chunks of a lane of {} chunks",
            run.cursor, eng.max_chunks
        ));
    }
    activate(eng, lane);
    let samp = eng.lanes.as_ref().expect("lanes").samp_bufs[lane];
    spec::lane_single_row_step(eng, run.sampled, run.parity, nch, samp)
}

/// Serves the requests of `feed` in the engine's lanes until the feed ends the session.
pub(super) fn serve(eng: &mut Qwen3Engine, feed: &mut dyn LaneFeed) -> Result<(), String> {
    let n = eng.lanes.as_ref().expect("lanes").n;
    let mut runs: Vec<Option<Run>> = (0..n).map(|_| None).collect();
    let res = serve_loop(eng, feed, &mut runs);
    if let Some(line) = eng.lanes.as_mut().expect("lanes").timing.report() {
        if eng.verbose {
            eprintln!("{line}");
        }
    }
    if let Some(line) = eng.lanes.as_mut().expect("lanes").widths.report() {
        if eng.verbose {
            eprintln!("{line}");
        }
    }
    let launched = eng.lanes.as_ref().expect("lanes").launched;
    let res = res.and(prof_rt::end_lanes(eng, launched));
    if let Err(e) = &res {
        // a launch queued ahead lands first (never leave launches queued behind an error)
        let _ = eng.dev.sync_all();
        // every request taken gets its end: the lanes still busy end with the error
        for (lane, slot) in runs.iter_mut().enumerate() {
            if let Some(mut run) = slot.take() {
                activate(eng, lane);
                let r: Result<Qwen3GenReport, String> = Err(e.clone());
                request_reset::end_request(eng, run.prompt_len, &r);
                eng.prefix.request_end(false, 0);
                let ev = done_event(r, eng, run.exclusive);
                (run.out)(ev);
            }
        }
        let rt = eng.lanes.as_mut().expect("lanes");
        if let Err(le) = rt.table.fail_all() {
            // (every move of fail_all is allowed from a busy state: this is an internal error)
            eprintln!("[gaudium] internal: lanes: {le}; the lane table is reset to free lanes");
            rt.table = LaneTable::new(n);
        }
        let v = view(rt, &runs);
        feed.lanes(&v);
    }
    res
}

fn publish(eng: &Qwen3Engine, feed: &mut dyn LaneFeed, runs: &[Option<Run>]) {
    let v = view(eng.lanes.as_ref().expect("lanes"), runs);
    feed.lanes(&v);
}

fn serve_loop(
    eng: &mut Qwen3Engine,
    feed: &mut dyn LaneFeed,
    runs: &mut [Option<Run>],
) -> Result<(), String> {
    // a speculative launch queued ahead, not collected yet
    let mut ahead: Option<LaneLaunch> = None;
    loop {
        // 1. admission: the oldest waiting request, while the lane gate lets it in
        let mut waiting = false;
        loop {
            let idle = eng.lanes.as_ref().expect("lanes").table.busy() == 0;
            let exclusive = match feed.head(idle) {
                FeedHead::End => {
                    if let Some(a) = ahead.take() {
                        // (a session ends with no lane busy, so nothing is queued; collected
                        // anyway so that no launch outlives the session)
                        spec_collect(eng, &a)?;
                    }
                    return Ok(());
                }
                FeedHead::Empty => break,
                FeedHead::Job { exclusive } => exclusive,
            };
            waiting = true;
            if !eng
                .lanes
                .as_ref()
                .expect("lanes")
                .table
                .can_admit(exclusive)
            {
                break;
            }
            let Some(job) = feed.take() else { break };
            let lane = {
                let rt = eng.lanes.as_ref().expect("lanes");
                let prompt = job.prompt.clone();
                let params = job.params.clone();
                rt.table.route(|i| lane_reuse(eng, i, &prompt, &params))
            };
            let Some(lane) = lane else {
                return Err("internal: the lane gate admitted a request with no free lane".into());
            };
            eng.lanes
                .as_mut()
                .expect("lanes")
                .table
                .admit(lane, exclusive)
                .map_err(|e| format!("lanes: {e}"))?;
            publish(eng, feed, runs);
            match start_lane(eng, lane, job)? {
                None => {
                    let t = &mut eng.lanes.as_mut().expect("lanes").table;
                    t.fail(lane).map_err(|e| format!("lanes: {e}"))?;
                    t.release(lane).map_err(|e| format!("lanes: {e}"))?;
                }
                Some(run) => {
                    let done = run.done();
                    runs[lane] = Some(run);
                    let t = &mut eng.lanes.as_mut().expect("lanes").table;
                    if done {
                        t.finish(lane).map_err(|e| format!("lanes: {e}"))?;
                    } else {
                        t.prefilled(lane).map_err(|e| format!("lanes: {e}"))?;
                    }
                }
            }
            publish(eng, feed, runs);
        }

        // 2. one step of every decoding lane, greedy and sampled lanes in their own launches.
        // A step queued ahead belongs to the lanes it was launched for: when the batch is no
        // longer exactly those lanes (a request joined meanwhile), it lands first, so the groups
        // below are formed from what the lanes hold now.
        if let Some(a) = ahead.take() {
            let mut queued: Vec<usize> = a.lanes.iter().map(|&(l, _)| l).collect();
            queued.sort_unstable();
            let mut now = eng.lanes.as_ref().expect("lanes").table.batch();
            now.sort_unstable();
            if queued == now {
                ahead = Some(a);
            } else {
                let em = spec_collect(eng, &a)?;
                let live = a
                    .lanes
                    .iter()
                    .zip(em)
                    .filter(|((l, _), _)| runs[*l].as_ref().is_some_and(|r| !r.done()))
                    .map(|(&(l, t), e)| (l, t, e))
                    .collect();
                apply_spec(eng, runs, live)?;
            }
        }
        let batch = eng.lanes.as_ref().expect("lanes").table.batch();
        for sampled in [false, true] {
            let group: Vec<usize> = batch
                .iter()
                .copied()
                .filter(|&l| {
                    runs[l]
                        .as_ref()
                        .is_some_and(|r| r.sampled == sampled && !r.done())
                })
                .collect();
            if group.is_empty() {
                continue;
            }
            if eng.specdec.is_some() {
                // the next step may be queued ahead while this one runs (its inputs are on the
                // device already): one group of speculative lanes, nothing waiting
                let one_group = batch.len() == group.len();
                spec_group_step(eng, runs, &group, sampled, waiting, one_group, &mut ahead)?;
                eng.live.step(group.len() as u64, 0, 0);
                continue;
            }
            let toks = step(eng, &group, runs, sampled)?;
            for (&lane, tok) in group.iter().zip(toks) {
                let run = runs[lane].as_mut().expect("a decoding lane runs");
                run.advance(1, 1);
                run.observe_cancel();
                if !run.done() {
                    run.emit(tok);
                }
                if run.done() {
                    eng.lanes
                        .as_mut()
                        .expect("lanes")
                        .table
                        .finish(lane)
                        .map_err(|e| format!("lanes: {e}"))?;
                }
            }
            eng.live.step(group.len() as u64, 0, 0);
        }

        // 3. the lanes whose request ended deliver it and are free again
        let drained = eng.lanes.as_ref().expect("lanes").table.drained();
        for lane in drained.iter().copied() {
            if let Some(run) = runs[lane].take() {
                end_lane(eng, lane, run);
            }
            eng.lanes
                .as_mut()
                .expect("lanes")
                .table
                .release(lane)
                .map_err(|e| format!("lanes: {e}"))?;
        }
        if !drained.is_empty() || !batch.is_empty() {
            publish(eng, feed, runs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_costs_learn_the_narrow_step_first_then_the_extra_of_a_wide_slot() {
        let mut w = WidthCosts::default();
        assert_eq!(w.estimate(4, 1), None);
        // a wide launch before any narrow one of that shape teaches nothing yet
        w.observe(4, 1, 2, 30.0);
        assert_eq!(w.estimate(4, 1), None);
        w.observe(4, 1, 0, 20.0);
        assert_eq!(w.estimate(4, 1), Some((20.0, 0.0)));
        w.observe(4, 1, 2, 24.0);
        assert_eq!(w.estimate(4, 1), Some((20.0, 2.0)));
        // exponential averages (today's weight 0.2)
        w.observe(4, 1, 0, 30.0);
        let (t0, extra) = w.estimate(4, 1).unwrap();
        assert!((t0 - 22.0).abs() < 1e-9 && (extra - 2.0).abs() < 1e-9);
        // other shapes are their own; a slot count without a measured wide slot takes the
        // nearest one's extra time
        assert_eq!(w.estimate(4, 2), None);
        assert_eq!(w.estimate(3, 1), None);
        w.observe(3, 1, 0, 18.0);
        assert_eq!(w.estimate(3, 1), Some((18.0, 2.0)));
    }

    fn qwen38_27b_spec() -> ModelSpec {
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
    fn batched_step_with_fp8_kv_gives_each_slot_its_caches_and_shares_each_layers_scale_table() {
        let mut spec = qwen38_27b_spec();
        spec.kv_dtype = crate::verified::KvDtype::KvFp8;
        let elfs = vec![("GAUDIUM_GDNS_ELF".to_string(), "aa".to_string())];
        let g = build_batched_step_graph(&spec, 16384, 2, 3, &elfs, false);
        let specs = g.persistent_tensor_specs();
        let find = |n: &str| specs.iter().find(|s| s.0 == n);
        // one scale table per attention layer, a layer table bound like the weights
        let table = find("L3_kv8").expect("layer 3's scale table");
        assert!(is_weight_or_table("L3_kv8"));
        assert!(slot_bound("L3_kv8").is_none());
        for j in 0..3 {
            assert!(find(&format!("s{j}_L3_kv8")).is_none(), "no per-slot table");
            let k = find(&format!("s{j}_L3_kcache")).expect("the slot's cache");
            assert_eq!(k.1, SynDataType::Fp8_143, "FP8 caches per slot");
            assert_eq!(slot_bound(&format!("s{j}_L3_kcache")).map(|b| b.0), Some(j));
        }
        // the same table as the single-stream step's
        let solo = build_decode_step_graph(&spec, 16384, 2, &elfs);
        let s = solo.persistent_tensor_specs();
        let solo_table = s.iter().find(|x| x.0 == "L3_kv8").expect("solo table");
        assert_eq!((table.1, &table.2, table.3), (solo_table.1, &solo_table.2, solo_table.3));
    }

    #[test]
    fn batched_step_shares_the_trunk_and_gives_each_slot_its_own_lane_buffers() {
        let spec = qwen38_27b_spec();
        let elfs = vec![("GAUDIUM_GDNS_ELF".to_string(), "aa".to_string())];
        let g = build_batched_step_graph(&spec, 16384, 2, 3, &elfs, false);
        assert_eq!(g.recipe_name(), "g_qwen_bstep_B3_C2");
        let specs = g.persistent_tensor_specs();
        let dims = |n: &str| specs.iter().find(|s| s.0 == n).map(|s| s.2.clone());
        // one weight set, read by the trunk over 3 columns
        assert_eq!(dims("L0_W_qkvz").map(|d| d.len()), Some(2));
        assert_eq!(dims("L3_qg"), Some(vec![spec.qg_dim(), 3]));
        assert_eq!(dims("ids"), Some(vec![3]));
        assert_eq!(dims("tokens"), Some(vec![1, 3]));
        // per slot: its own cursor, caches (two chunks) and states
        for j in 0..3 {
            assert_eq!(dims(&format!("s{j}_cursor")), Some(vec![1]));
            assert_eq!(dims(&format!("s{j}_L3_kcache")), Some(vec![256, 4096, 8]));
            assert!(dims(&format!("s{j}_L0_S_in")).is_some());
        }
        assert!(dims("s3_cursor").is_none());
        // the solo step's weight names, no new weight
        let solo = build_decode_step_graph(&spec, 16384, 2, &elfs);
        for (name, ..) in solo.persistent_tensor_specs() {
            if is_weight_or_table(&name) {
                assert!(dims(&name).is_some(), "{name}");
            }
        }
        for (name, ..) in &specs {
            if is_weight_or_table(name) {
                assert!(solo.find_persistent_tensor(name).is_some(), "{name}");
            }
        }
        // what binds to a lane, by the lane's own buffer names
        assert_eq!(slot_bound("s2_L3_kcache"), Some((2, "L3_kcache")));
        assert_eq!(slot_bound("s0_L0_S_in"), Some((0, "L0_S_in")));
        assert_eq!(slot_bound("s1_cursor"), Some((1, "cursor")));
        assert_eq!(slot_bound("s1_L3_scores"), None);
        assert_eq!(slot_bound("samp_rsel"), None);
        let nodes = g.node_io_views();
        let count = |guid: &str| nodes.iter().filter(|n| n.guid == guid).count();
        assert_eq!(count("gaudium_attn_prep"), 16 * 3);
        assert_eq!(count("gaudium_gdn_f32"), 48 * 3);
        assert_eq!(count("argmax_fwd_bf16"), 1);
    }

    #[test]
    fn sampled_batched_step_reads_each_lanes_parameters_at_its_cursor() {
        let spec = qwen38_27b_spec();
        let g = build_batched_step_graph(&spec, 16384, 1, 2, &[], true);
        assert_eq!(g.recipe_name(), "g_qwen_sbstep_B2_C1");
        let views = g.node_io_views();
        let gum: Vec<_> = views
            .iter()
            .filter(|n| n.guid == "gaudium_samp_gumbel")
            .collect();
        assert_eq!(gum.len(), 2);
        for (j, n) in gum.iter().enumerate() {
            assert_eq!(n.inputs[2].name, format!("s{j}_cursor"));
            assert_eq!(n.inputs[3].name, format!("s{j}_samp_f"));
            assert_eq!(n.inputs[4].name, format!("s{j}_samp_i"));
        }
        assert!(!views.iter().any(|n| n.guid == "argmax_fwd_bf16"));
    }
}
