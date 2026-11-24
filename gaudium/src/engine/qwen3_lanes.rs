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
    batch_chunks, slot_fits, step_chunk_counts, step_chunks, LaneTable, Offer,
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
            let cores: Vec<TensorId> = (0..b as usize)
                .map(|j| {
                    let spfx = format!("{}{lpfx}", slot_pfx(j as u64));
                    append_attn_core(
                        &mut gb, spec, qgs[j], ks[j], vs[j], cursors[j], rope_c, rope_s, &lpfx,
                        &spfx, 1, nch,
                    )
                })
                .collect();
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

/// One lane's own device buffers and request bookkeeping while the lane is not the active one.
pub(super) struct LaneBufs {
    kv_addr: BTreeMap<String, (u64, u64)>,
    gdn_state: BTreeMap<usize, GdnLayerState>,
    reset: request_reset::ResetState,
    prefix: prefix_cache::PrefixCache,
    prefix_slots: Vec<prefix_cache::SlotAddrs>,
    prefix_plan: prefix_cache::Plan,
    prefix_images: Vec<prefix_cache::ImageId>,
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
    pub eager: Vec<((u64, u64, bool), CompiledRecipe)>,
    pub max_ws: u64,
    pub sampled: bool,
    pub elf_hashes: Vec<(String, String)>,
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
        trunk_elf_hashes: &[(String, String)],
        kenv: &KernelEnv,
        sampler: bool,
        weights: &dyn Fn(&str) -> bool,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
        verbose: bool,
        compile_all: bool,
    ) -> Result<Self, String> {
        let mut elf_hashes = trunk_elf_hashes.to_vec();
        if sampler {
            for (k, h) in kenv.fingerprint_hashes(crate::engine::sampler::SAMPLER_ELFS)? {
                if !elf_hashes.iter().any(|(e, _)| *e == k) {
                    elf_hashes.push((k, h));
                }
            }
        }
        let widest =
            build_batched_step_graph(spec, lmax, max_chunks, n as u64, &elf_hashes, sampler);
        let mut act_specs = BTreeMap::new();
        for (name, _dt, _dims, bytes) in widest.persistent_tensor_specs() {
            if weights(&name)
                || crate::engine::sampler::is_sampler_param(&name)
                || slot_bound(&name).is_some()
                || is_step_io(&name)
            {
                continue;
            }
            act_specs.insert(name, bytes);
        }
        let io_bytes = IO_SCALAR * io_names(n).len() as u64;
        let mut keys: Vec<(u64, u64, bool)> = Vec::new();
        let fams: &[bool] = if sampler { &[false, true] } else { &[false] };
        for &sm in fams {
            if compile_all {
                for nch in step_chunk_counts(max_chunks) {
                    for b in 1..=n as u64 {
                        keys.push((b, nch, sm));
                    }
                }
            } else {
                for b in 1..=n as u64 {
                    keys.push((b, 1, sm));
                }
                if max_chunks > 1 {
                    keys.push((n as u64, max_chunks, sm));
                }
            }
        }
        let t0 = Instant::now();
        let describe = |i: usize| -> GraphBuilder {
            let (b, nch, sm) = keys[i];
            build_batched_step_graph(spec, lmax, nch, b, &elf_hashes, sm)
        };
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
        for (i, &key) in keys.iter().enumerate() {
            let r = match std::mem::replace(&mut cached[i], Ok(None)) {
                Ok(Some(r)) => r,
                _ => describe(i)
                    .compile(Arc::clone(api), cache, driver_version)
                    .map_err(|e| {
                        format!(
                            "compile batched step {}: {e}",
                            batched_step_name(key.0, key.1, key.2)
                        )
                    })?,
            };
            if r.from_cache {
                hits += 1;
            } else {
                built += 1;
            }
            max_ws = max_ws.max(r.ws_size);
            eager.push((key, r));
        }
        if verbose {
            eprintln!(
                "[gaudium] lanes: {n} lanes, batched step recipes {hits} from the cache, {built} compiled ({:.1}s); largest workspace {:.2} MiB; step buffers {:.2} GiB",
                t0.elapsed().as_secs_f64(),
                max_ws as f64 / (1024.0 * 1024.0),
                act_specs.values().map(|&b| align_up(b.max(256), 256)).sum::<u64>() as f64
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
            elf_hashes,
        })
    }

    /// Bytes of the step buffers and the io, as the act arena allocates them.
    pub fn act_bytes(&self) -> u64 {
        self.act_specs
            .values()
            .map(|&b| align_up(b.max(256), 256))
            .sum::<u64>()
            + self.io_bytes
    }
}

/// Key of a prepared batched launch: the recipe, and the lane and state parity of every slot.
type LaunchKey = (u64, u64, bool, Vec<u8>, Vec<u8>);

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
    elf_hashes: Vec<(String, String)>,
    sampled: bool,
    recipes: HashMap<(u64, u64, bool), CompiledRecipe>,
    launches: HashMap<LaunchKey, PreparedLaunch>,
    pub table: LaneTable,
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
            elf_hashes: plan.elf_hashes,
            sampled: plan.sampled,
            recipes: plan.eager.into_iter().collect(),
            launches: HashMap::new(),
            table: LaneTable::new(n),
        }
    }

    pub fn lanes(&self) -> usize {
        self.n
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
        out.push(LaneBufs {
            kv_addr,
            gdn_state,
            reset: reset(),
            prefix: prefix(),
            prefix_slots,
            prefix_plan: prefix_cache::Plan::default(),
            prefix_images: Vec::new(),
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
    if sampled && !eng.lanes.as_ref().is_some_and(|rt| rt.sampled) {
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
    eng.live.begin_request(prompt.len() as u64, &label, "");
    eng.live.prefill_progress(0);
    let t_start = Instant::now();
    let is_cancelled = || cancel.load(Ordering::Relaxed);
    let pre = eng.run_prefill(&prompt, &params, &mut budget, &is_cancelled, true);
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
    };
    match pre {
        Err(e) => {
            eng.live.end_request();
            fail(eng, &mut run.out, e);
            Ok(None)
        }
        Ok(Prefilled::Cancelled(rep)) => {
            run.stopped_by_cancel = true;
            run.ttft_ms = rep.ttft_ms;
            Ok(Some(run))
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
            if run.budget.remaining() > 0 && !run.cancel.load(Ordering::Relaxed) {
                eng.live.first_token(Some(f.ttft_ms), 1);
                run.emit(f.first_tok);
            } else if run.cancel.load(Ordering::Relaxed) {
                run.stopped_by_cancel = true;
            }
            Ok(Some(run))
        }
    }
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
        tokens_per_launch: if run.steps > 0 { 1.0 } else { 0.0 },
        method: if run.sampled {
            format!("K=0 {wprec} T={:.2}", run.params.temperature)
        } else {
            format!("K=0 {wprec} greedy")
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
    eng.live.end_request();
    let ev = done_event(res, eng, run.exclusive);
    (run.out)(ev);
}

/// Makes the batched step `(b, nch, sampled)` launchable; the ms a first use took.
fn ensure_recipe(eng: &mut Qwen3Engine, key: (u64, u64, bool)) -> Result<f64, String> {
    if eng
        .lanes
        .as_ref()
        .is_some_and(|rt| rt.recipes.contains_key(&key))
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
    let (b, nch, sm) = key;
    let mut r = build_batched_step_graph(spec, *max_ctx, nch, b, &rt.elf_hashes, sm)
        .compile(Arc::clone(dev.api()), cache, driver_version)
        .map_err(|e| {
            format!(
                "compile batched step {}: {e}",
                batched_step_name(b, nch, sm)
            )
        })?;
    r.bind_shared_workspace(dev)
        .map_err(|e| format!("bind {} ws: {e}", r.name))?;
    if eng.verbose {
        eprintln!(
            "[gaudium] lanes: recipe {} was not ready: the step waited {:.1} ms for it",
            batched_step_name(b, nch, sm),
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    rt.recipes.insert(key, r);
    Ok(t0.elapsed().as_secs_f64() * 1000.0)
}

/// The prepared launch of batched step `key` with `lanes[j]` (state parity `parities[j]`) in
/// slot `j`.
fn prepare(
    eng: &mut Qwen3Engine,
    key: (u64, u64, bool),
    lanes: &[usize],
    parities: &[usize],
) -> Result<(), String> {
    let lk: LaunchKey = (
        key.0,
        key.1,
        key.2,
        lanes.iter().map(|&l| l as u8).collect(),
        parities.iter().map(|&p| p as u8).collect(),
    );
    if eng
        .lanes
        .as_ref()
        .is_some_and(|rt| rt.launches.contains_key(&lk))
    {
        return Ok(());
    }
    let rt = eng.lanes.as_ref().expect("lanes");
    let mut pool = rt.base.clone();
    for (j, (&lane, &p)) in lanes.iter().zip(parities).enumerate() {
        let sp = slot_pfx(j as u64);
        pool.insert(format!("{sp}cursor"), rt.io_cursor[j]);
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
        for (&l, st) in gdn {
            pool.insert(format!("{sp}L{l}_S_in"), st.s[p]);
            pool.insert(format!("{sp}L{l}_S_out"), st.s[1 - p]);
            pool.insert(format!("{sp}L{l}_cs_in"), st.cs[p]);
            pool.insert(format!("{sp}L{l}_cs_out"), st.cs[1 - p]);
        }
    }
    let recipe = &rt.recipes[&key];
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
    let key = (b, nch, sampled);
    ensure_recipe(eng, key)?;
    let parities: Vec<usize> = lanes
        .iter()
        .map(|&l| runs[l].as_ref().expect("a decoding lane runs").parity)
        .collect();
    prepare(eng, key, lanes, &parities)?;
    // ids and cursors in one synchronised upload
    let (io_ids, io_tokens, io_cursor) = {
        let rt = eng.lanes.as_ref().expect("lanes");
        (rt.io_ids, rt.io_tokens, rt.io_cursor.clone())
    };
    let mut copies = vec![(0usize, 4 * lanes.len(), io_ids)];
    {
        let buf = eng.stage_io.as_mut_slice();
        for (j, &l) in lanes.iter().enumerate() {
            let r = runs[l].as_ref().expect("a decoding lane runs");
            buf[4 * j..4 * j + 4].copy_from_slice(&r.cur_tok.to_le_bytes());
            let off = 64 * (1 + j);
            buf[off..off + 4].copy_from_slice(&(r.cursor as i32).to_le_bytes());
            copies.push((off, 4, io_cursor[j]));
        }
    }
    eng.dev
        .copy_h2d_pinned_batch_sync(&eng.stage_io, &copies)
        .map_err(|e| format!("write batched step inputs H2D: {e}"))?;
    let lk: LaunchKey = (
        b,
        nch,
        sampled,
        lanes.iter().map(|&l| l as u8).collect(),
        parities.iter().map(|&p| p as u8).collect(),
    );
    {
        let rt = eng.lanes.as_ref().expect("lanes");
        rt.recipes[&key]
            .launch_prepared(&eng.dev, &rt.launches[&lk])
            .map_err(|e| format!("launch {}: {e}", batched_step_name(b, nch, sampled)))?;
    }
    // SAFETY: the compute stream belongs to `eng.dev`.
    unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
        .map_err(|e| format!("sync {}: {e}", batched_step_name(b, nch, sampled)))?;
    eng.dev
        .copy_d2h_pinned_sync(io_tokens, &mut eng.stage_io, 4 * lanes.len())
        .map_err(|e| format!("read batched step tokens D2H: {e}"))?;
    let raw = &eng.stage_io.as_slice()[..4 * lanes.len()];
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect())
}

/// Serves the requests of `feed` in the engine's lanes until the feed ends the session.
pub(super) fn serve(eng: &mut Qwen3Engine, feed: &mut dyn LaneFeed) -> Result<(), String> {
    let n = eng.lanes.as_ref().expect("lanes").n;
    let mut runs: Vec<Option<Run>> = (0..n).map(|_| None).collect();
    let res = serve_loop(eng, feed, &mut runs);
    if let Err(e) = &res {
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
        rt.table = LaneTable::new(n);
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
    loop {
        // 1. admission: the oldest waiting request, while the lane gate lets it in
        loop {
            let idle = eng.lanes.as_ref().expect("lanes").table.busy() == 0;
            let exclusive = match feed.head(idle) {
                FeedHead::End => return Ok(()),
                FeedHead::Empty => break,
                FeedHead::Job { exclusive } => exclusive,
            };
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

        // 2. one step of every decoding lane, greedy and sampled lanes in their own launches
        let batch = eng.lanes.as_ref().expect("lanes").table.batch();
        for sampled in [false, true] {
            let group: Vec<usize> = batch
                .iter()
                .copied()
                .filter(|&l| runs[l].as_ref().is_some_and(|r| r.sampled == sampled))
                .collect();
            if group.is_empty() {
                continue;
            }
            let toks = step(eng, &group, runs, sampled)?;
            for (&lane, tok) in group.iter().zip(toks) {
                let run = runs[lane].as_mut().expect("a decoding lane runs");
                run.parity ^= 1;
                run.cursor += 1;
                run.steps += 1;
                if run.cancel.load(Ordering::Relaxed) {
                    run.stopped_by_cancel = true;
                } else {
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
