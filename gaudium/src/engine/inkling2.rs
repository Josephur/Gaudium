//! Intel Gaudi 2 execution engine for the Inkling 2 decoders (`inkling2`, `inkling2_loop`): BF16
//! weights with the checkpoint's 4-bit / 8-bit expert matrices reproduced exactly, non-speculative
//! decode, tiled prefill.
//!
//! What the family needs from the chip, and how this engine lays it out:
//!
//! - **Layer applications, not layers.** A middle span of the layers runs several times per
//!   token with shared weights. A layer's weights are bound once (`L<l>_*`); everything a pass
//!   over it keeps (K / V state, convolution history, activations) belongs to the *application*
//!   (`A<a>_*`), so the three passes of a looped layer have three caches.
//! - **Two attention kinds, two kinds of K / V state.** A *sliding* layer keeps the last
//!   positions in a ring of a power-of-two number of slots; a *global* layer keeps every
//!   position in chunk-major caches read one bucket of 4096-row chunks at a time. Neither
//!   has a rotary position code: position enters through a relative-position bias the layer
//!   projects from each row (a tiny block-diagonal GEMM gives every head its bias against the
//!   backward distance, stored reversed so a softmax block loads 64 of them in key order),
//!   through a per-position scale of the global layers' queries, and through short causal
//!   convolutions over the projected keys, values, the attention output and the expert output.
//! - **Short convolutions read rings of rows by position.** A pass first saves its projected
//!   rows into a ring (`gaudium_ik_save`), then convolves out of the ring: the rows before the
//!   pass and the rows of the pass come from one place, and a rejected draft row is simply
//!   written again.
//! - **Block attention residuals.** A layer's input is a softmax mix, by learned queries, of the
//!   block states (the inputs of every sixth layer) and its own partial state; from the second
//!   pass on, messages mixed from the previous pass's block states go through low-rank gates.
//! - **Dense-masked experts.** A layer's experts are one wide MLP on the matrix engine; the
//!   router's dense weight table zeroes the blocks a row did not select and the down GEMM
//!   accumulates the weighted sum. The matrix engine reads each weight once whatever the row
//!   count, so a pass of several rows costs little more than one row.
//!
//! Execution follows the engine pattern of this crate: every graph goes through `GraphBuilder`
//! (full fingerprint coverage, recipe cache), device memory is summed up front, allocated exactly
//! and sealed, prefill runs generic single-application recipes bound per application at launch,
//! and decode is one monolithic `synLaunch` per token with a pre-validated binding list.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::core::budget::SequenceBudget;
use crate::core::inkling2::{
    final_states, hist_rows, msg_states, ring_slots, schedule, source_buffers, wiring, IkApp,
    IkWire, IK_MAX_STATES,
};
use crate::engine::qwen3::spec::{chain_accept, SpecConfig, SpecMethod};
use crate::engine::{EngineConfig, GenParams, GenReport};
use crate::json::{parse_json, JsonValue};
use crate::kernels::{KernelEnv, INKLING2_GLUE};
use crate::live_report::LiveReporter;
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::sampler::{sample_row, SampleParams};
use crate::synapse::device::{HbmArena, HpuDevice, PinnedBuf};
use crate::synapse::ffi::{SynDataType, SynapseApi};
use crate::synapse::graph::{CompiledRecipe, GraphBuilder, PreparedLaunch, TensorId};
use crate::synapse::ops::u32_params;
use crate::synapse::recipe_cache::RecipeCache;
use crate::verified::{
    align_up, ik_bias_off, ik_bias_width, ik_gate_off, ik_kx_width, ik_qx_width, ik_rel_off,
    ik_tile_last_row, ik_tile_rows, ik_tile_take, HbmCat,
};

#[path = "inkling2_packer.rs"]
pub mod packer;

#[path = "inkling2_gguf.rs"]
pub mod gguf_source;

// Image input (vision tower, preprocessing, hand-over to the prefill).
#[path = "inkling2_vision.rs"]
pub mod vision;

// Prefix reuse: taking over the rows the card holds from the request before.
#[path = "inkling2_reuse.rs"]
mod reuse;

/// Rows of one chunk of a global layer's K / V cache (the softmax kernel stages one chunk).
pub const IK_CHUNK: u64 = 4096;
/// Largest prefill tile planned into the activation scratch.
pub const IK_MAX_PREFILL_TILE: u64 = 256;
/// Smallest prefill tile the memory planner falls back to before refusing a configuration.
const IK_MIN_PREFILL_TILE: u64 = 8;
/// Shared recipe workspace floor, so lazily compiled recipes never need a device allocation.
const MIN_SHARED_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
/// Reusable pinned host staging buffer for chunked weight uploads and state zeroing.
const STAGE_BUF_BYTES: usize = 64 * 1024 * 1024;
/// HBM kept free beside the plan.
const HBM_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

/// Kernel ELFs every Inkling 2 graph may be handed by the graph-compiler glue.
const INKLING2_ELFS: &[&str] = &[
    "GAUDIUM_IKMIX_ELF",
    "GAUDIUM_IKMIXF_ELF",
    "GAUDIUM_IKGIN_ELF",
    "GAUDIUM_IKMGATE_ELF",
    "GAUDIUM_IKMGATEF_ELF",
    "GAUDIUM_IKNORM_ELF",
    "GAUDIUM_IKSAVE_ELF",
    "GAUDIUM_IKCONVA_ELF",
    "GAUDIUM_IKCONVM_ELF",
    "GAUDIUM_IKQPREP_ELF",
    "GAUDIUM_IKKVPREP_ELF",
    "GAUDIUM_IKSOFT_ELF",
    "GAUDIUM_IKCOMB_ELF",
    "GAUDIUM_IKROUTE_ELF",
    "GAUDIUM_IKACT_ELF",
    "GAUDIUM_IKARGMAX_ELF",
    "GAUDIUM_GARMSNORM_ELF",
];

/// Geometry of one Inkling 2 configuration: the model's shapes and the sizes that depend on the
/// widest pass (`rows`).
#[derive(Debug, Clone)]
pub struct IkDims {
    pub hid: u64,
    pub nh: u64,
    pub nkv: u64,
    pub gq: u64,
    pub ne: u64,
    pub ek: u64,
    pub ef: u64,
    pub taps: u64,
    pub d_rel: u64,
    pub extent: u64,
    pub window: u64,
    pub rank: u64,
    /// Embedding rows, rows the model can emit, rows of the device unembedding (whole blocks).
    pub vocab_in: u64,
    pub vocab_out: u64,
    pub vocab_pad: u64,
    pub n_layers: usize,
    pub apps: Vec<IkApp>,
    pub wires: Vec<IkWire>,
    pub final_states: Vec<usize>,
    pub n_msg_states: u64,
    /// Slots of a sliding layer's K / V ring; rows of a convolution's history ring.
    pub ring: u64,
    pub hist: u64,
    /// Channels of the query-side projection, of one head's bias table, and the table's offset.
    pub qw: u64,
    pub bw: u64,
    pub boff: u64,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub gate_cap: f32,
    pub up_cap: f32,
    pub route_scale: f32,
    pub log_floor: u64,
    pub log_alpha: f32,
    /// The message gates' second projection accumulates in f32 (as the reference does).
    pub fg_f32: bool,
    /// Applications a debug run builds (all of them unless cut short).
    pub n_apps: usize,
}

impl IkDims {
    /// Builds the geometry for passes of at most `rows` rows.
    pub fn from_spec(spec: &ModelSpec, rows: u64) -> Result<Self, String> {
        let ik = spec
            .inkling2
            .as_ref()
            .ok_or("Inkling 2 engine: config.json is not an Inkling 2 model")?;
        spec.validate().map_err(|e| e.to_string())?;
        let apps = schedule(spec, ik);
        let (wires, fin) = wiring(&apps);
        debug_assert_eq!(fin.len() as u64, final_states(spec, ik));
        let n_apps = apps.len();
        Ok(Self {
            hid: spec.hidden,
            nh: spec.n_heads,
            nkv: spec.n_kv_heads,
            gq: spec.n_heads / spec.n_kv_heads,
            ne: spec.n_experts,
            ek: spec.experts_per_tok,
            ef: spec.moe_intermediate,
            taps: ik.conv_taps,
            d_rel: ik.d_rel,
            extent: ik.rel_extent,
            window: ik.window,
            rank: ik.gate_rank,
            vocab_in: spec.vocab,
            vocab_out: ik.unpadded_vocab,
            vocab_pad: align_up(ik.unpadded_vocab, 128),
            n_layers: spec.n_layers as usize,
            apps,
            wires,
            final_states: fin,
            n_msg_states: msg_states(ik),
            ring: ring_slots(ik, rows)?,
            hist: hist_rows(ik, rows)?,
            qw: ik_qx_width(spec.n_heads, ik.d_rel),
            bw: ik_bias_width(ik.rel_extent),
            boff: ik_bias_off(ik.rel_extent),
            eps: spec.rms_eps as f32,
            qk_clip: ik.qk_clip as f32,
            rel_clip: ik.rel_clip as f32,
            gate_cap: ik.gate_softcap as f32,
            up_cap: ik.up_softcap as f32,
            route_scale: ik.route_scale as f32,
            log_floor: ik.log_floor,
            log_alpha: ik.log_alpha as f32,
            fg_f32: true,
            n_apps,
        })
    }

    /// Channels of a layer's key / value projection (the global layers add the convolution taps).
    pub fn kxw(&self, sliding: bool) -> u64 {
        ik_kx_width(self.nkv, if sliding { 0 } else { self.taps })
    }

    /// Whether layer `l` attends a sliding window.
    pub fn layer_sliding(&self, l: usize) -> bool {
        self.apps
            .iter()
            .find(|a| a.layer == l)
            .is_some_and(|a| a.sliding)
    }

    /// Whether layer `l` is part of the looped span (it has message gates).
    pub fn layer_looped(&self, l: usize) -> bool {
        self.apps.iter().any(|a| a.layer == l && a.looped) && self.apps.iter().any(|a| a.msgs)
    }

    /// The block states the final mix reads: those behind the last layer, or, for a debug run cut
    /// short, the ones its last application's MLP mix saw.
    pub fn fin_states(&self) -> &[usize] {
        if self.n_apps == self.apps.len() {
            &self.final_states
        } else {
            &self.wires[self.n_apps - 1].mlp_states
        }
    }

    /// The applications this configuration runs.
    pub fn run_apps(&self) -> &[IkApp] {
        &self.apps[..self.n_apps]
    }

    /// The distinct application kinds of the prefill recipes.
    pub fn kinds(&self) -> Vec<IkKind> {
        let mut v = Vec::new();
        for a in self.run_apps() {
            let k = IkKind {
                sliding: a.sliding,
                msgs: a.msgs,
            };
            if !v.contains(&k) {
                v.push(k);
            }
        }
        v
    }
}

/// The shape of an application's recipe: its attention kind and whether it receives messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IkKind {
    pub sliding: bool,
    pub msgs: bool,
}

impl IkKind {
    fn tag(self) -> &'static str {
        match (self.sliding, self.msgs) {
            (true, false) => "s",
            (true, true) => "sm",
            (false, false) => "g",
            (false, true) => "gm",
        }
    }
}

/// Decode-step bucket (chunks of the global caches a step reads): the smallest rung of the ladder
/// `1, 2, 3, 4, 6, 8, 12, 16, ...` that holds `need` chunks, capped at `max_chunks`.
pub fn step_bucket(need: u64, max_chunks: u64) -> u64 {
    crate::engine::gemma4::step_bucket(need, max_chunks)
}

// ============================================================================
// Pure graph builders (host-testable without the card)
// ============================================================================

fn f32_bits(v: f32) -> u32 {
    v.to_bits()
}

/// BF16 linear projection `y {n, t} = W {k, n}^T . x {k, t}`.
fn ik_linear(
    gb: &mut GraphBuilder,
    x: TensorId,
    w_name: &str,
    k: u64,
    n: u64,
    t: u64,
    y_name: &str,
) -> TensorId {
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Bf16, &[k, n]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[n, t]);
    gb.op_gemm(format!("{y_name}_gemm"), x, w, y, false, true);
    y
}

/// A second view of `x` with other dims (no data moves: a workspace tensor behind a reshape).
fn ik_view(
    gb: &mut GraphBuilder,
    x: TensorId,
    name: &str,
    dtype: SynDataType,
    dims: &[u64],
) -> TensorId {
    let v = gb.add_intermediate(name, dtype, dims);
    gb.op_reshape(format!("{name}_rs"), x, v);
    v
}

/// Linear projection of BF16 rows with an f32 result `y {n, t}` (a batch of one on the matrix
/// engine, whose batched product accumulates into f32). The weight tensor is `{k, n, 1}`.
fn ik_linear_f32(
    gb: &mut GraphBuilder,
    x: TensorId,
    w_name: &str,
    k: u64,
    n: u64,
    t: u64,
    y_name: &str,
) -> TensorId {
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Bf16, &[k, n, 1]);
    let x3 = ik_view(gb, x, &format!("{y_name}_x3"), SynDataType::Bf16, &[k, t, 1]);
    let y3 = gb.add_persistent_tensor(format!("{y_name}3"), SynDataType::Float, &[n, t, 1]);
    gb.op_batch_gemm(format!("{y_name}_gemm"), x3, w, y3, false, true);
    ik_view(gb, y3, y_name, SynDataType::Float, &[n, t])
}

fn record_elfs(gb: &mut GraphBuilder, elf_hashes: &[(String, String)]) {
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
}

fn ik_graph(name: String, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = GraphBuilder::new(name);
    record_elfs(&mut gb, elf_hashes);
    gb
}

/// Where one application's inputs and state come from.
struct IkIo<'a> {
    /// Tensor-name prefix of the layer's weights (`"L7_"`, or `"L_"` generic).
    lpfx: &'a str,
    /// Prefix of the application's state, control words and activations (`"A19_"`, or `"A_"`).
    apfx: &'a str,
    x_in: TensorId,
    /// The four state inputs of the mixes (slots beyond the application's count hold any tensor).
    states: [TensorId; 4],
    /// Message sources (the previous pass's block states, then its last output) and their
    /// control words; `None` for an application without messages.
    msg: Option<([TensorId; 4], TensorId, TensorId)>,
    cursor: TensorId,
    tau: TensorId,
}

/// Appends one message gate: `out {hid, t, 1} = (1 - tanh(f)) h + tanh(g) m` with the gate rows
/// projected from the normalised pair `[h | m]` through the rank-`rank` bottleneck.
#[allow(clippy::too_many_arguments)]
fn append_ik_gate(
    gb: &mut GraphBuilder,
    d: &IkDims,
    io: &IkIo<'_>,
    which: &str,
    h: TensorId,
    jh: u32,
    m: TensorId,
    jm: u32,
    t: u64,
) -> TensorId {
    let (l, a, hid) = (io.lpfx, io.apfx, d.hid);
    let gin = gb.add_persistent_tensor(format!("{a}gi_{which}"), SynDataType::Bf16, &[2 * hid, t]);
    gb.add_node(
        "gaudium_ik_gin",
        format!("{a}gin_{which}"),
        &[h, m],
        &[gin],
        &u32_params(&[jh, jm, f32_bits(d.eps)]),
    );
    let gd = ik_linear(
        gb,
        gin,
        &format!("{l}W_gd_{which}"),
        2 * hid,
        d.rank,
        t,
        &format!("{a}gd_{which}"),
    );
    let fg = if d.fg_f32 {
        ik_linear_f32(
            gb,
            gd,
            &format!("{l}W_gf_{which}"),
            d.rank,
            2 * hid,
            t,
            &format!("{a}fg_{which}"),
        )
    } else {
        ik_linear(
            gb,
            gd,
            &format!("{l}W_gf_{which}"),
            d.rank,
            2 * hid,
            t,
            &format!("{a}fg_{which}"),
        )
    };
    let out = gb.add_persistent_tensor(format!("{a}go_{which}"), SynDataType::Float, &[hid, t, 1]);
    gb.add_node(
        "gaudium_ik_mgate",
        format!("{a}mgate_{which}"),
        &[h, m, fg],
        &[out],
        &u32_params(&[jh, jm]),
    );
    out
}

/// Appends one layer application over `t` rows at positions `cursor + row`:
///
/// ```text
/// [m = mix4(previous pass's block states, its output)]                   messages (later passes)
/// h_q, h_k, h_v = mix(block states, x)       [gated by m]                attention input
/// q, gate, rel, sink = W norm(h_q);  k = W norm(h_k);  v = W norm(h_v)
/// K = clip(conv(k)),  V = norm(conv(v))                                   into the ring / cache
/// a = softmax(q K^T / sqrt(dh) + bias(rel) | sink) V;   attn = W_o (a sigmoid(gate))
/// r = conv(attn) [+ x unless the layer opens a block]                     f32 residual stream
/// h = mix(block states [+ x], r)             [gated by m]                MLP input
/// out = bf16(r + conv(experts(norm(h))))
/// ```
fn append_ik_app(
    gb: &mut GraphBuilder,
    d: &IkDims,
    io: &IkIo<'_>,
    kind: IkKind,
    t: u64,
    nch: u64,
) -> TensorId {
    let (l, a, hid) = (io.lpfx, io.apfx, d.hid);
    let f32t = SynDataType::Float;
    let bf = SynDataType::Bf16;
    let sliding = kind.sliding;
    let ctl = gb.get_or_add_persistent_tensor(format!("{a}ctl"), SynDataType::Int32, &[8]);
    let mix_params = u32_params(&[f32_bits(d.eps_mix()), f32_bits((hid as f32).powf(-0.5))]);

    // messages of the previous pass: one mix per query (query, key, value, MLP)
    let msgs = io.msg.as_ref().map(|(p, ph, ctlp)| {
        let q = gb.get_or_add_persistent_tensor(format!("{l}q_msg"), f32t, &[hid, 4]);
        let m = gb.add_persistent_tensor(format!("{a}msg"), f32t, &[hid, t, 4]);
        gb.add_node(
            "gaudium_ik_mix",
            format!("{a}mix_msg"),
            &[p[0], p[1], p[2], p[3], *ph, q, *ctlp],
            &[m],
            &mix_params,
        );
        m
    });

    // attention input: one mixed state (sliding) or one each for the query, key and value (global)
    let nq = if sliding { 1 } else { 3 };
    let q_attn = gb.get_or_add_persistent_tensor(format!("{l}q_attn"), f32t, &[hid, nq]);
    let ha = gb.add_persistent_tensor(format!("{a}ha"), f32t, &[hid, t, nq]);
    gb.add_node(
        "gaudium_ik_mix",
        format!("{a}mix_attn"),
        &[
            io.states[0],
            io.states[1],
            io.states[2],
            io.states[3],
            io.x_in,
            q_attn,
            ctl,
        ],
        &[ha],
        &mix_params,
    );
    let mut xin = [TensorId(0); 3];
    for (j, which) in ["q", "k", "v"].into_iter().enumerate() {
        let jh = if sliding { 0 } else { j as u32 };
        if j > 0 && sliding && msgs.is_none() {
            // a sliding layer without messages projects all three from one normed row
            xin[j] = xin[0];
            continue;
        }
        let (src, js) = match msgs {
            Some(m) => (append_ik_gate(gb, d, io, which, ha, jh, m, j as u32, t), 0),
            None => (ha, jh),
        };
        let w = gb.get_or_add_persistent_tensor(format!("{l}w_{which}"), bf, &[hid]);
        let y = gb.add_persistent_tensor(format!("{a}x{which}"), bf, &[hid, t]);
        gb.add_node(
            "gaudium_ik_norm",
            format!("{a}norm_{which}"),
            &[src, w],
            &[y],
            &u32_params(&[js, f32_bits(d.eps)]),
        );
        xin[j] = y;
    }

    // projections
    let kxw = d.kxw(sliding);
    let qx = ik_linear(gb, xin[0], &format!("{l}W_qx"), hid, d.qw, t, &format!("{a}qx"));
    let sink = ik_linear(gb, xin[0], &format!("{l}W_sink"), hid, d.nh, t, &format!("{a}sink"));
    let kx = ik_linear(gb, xin[1], &format!("{l}W_kx"), hid, kxw, t, &format!("{a}kx"));
    let vx = ik_linear(gb, xin[2], &format!("{l}W_vx"), hid, kxw, t, &format!("{a}vx"));

    // the projected rows by position, for the short convolutions
    let kvw = 128 * d.nkv;
    let kraw = gb.get_or_add_persistent_tensor(format!("{a}kraw"), bf, &[kvw, d.hist]);
    let vraw = gb.get_or_add_persistent_tensor(format!("{a}vraw"), bf, &[kvw, d.hist]);
    gb.add_node("gaudium_ik_save", format!("{a}save_k"), &[kx, io.cursor], &[kraw], &u32_params(&[0]));
    gb.add_node("gaudium_ik_save", format!("{a}save_v"), &[vx, io.cursor], &[vraw], &u32_params(&[0]));

    // queries and relative states
    let (c, nchx) = if sliding { (d.ring, 1) } else { (IK_CHUNK, nch) };
    let b = d.nkv * nchx;
    let r = d.gq * t;
    let ng = d.nh / 8;
    let qsc = gb.get_or_add_persistent_tensor(format!("{l}qsc"), f32t, &[128, d.nh]);
    let rsc = gb.get_or_add_persistent_tensor(format!("{l}rsc"), f32t, &[128, ng]);
    let qb = gb.add_persistent_tensor(format!("{a}qb"), bf, &[128, r, b]);
    let rs = gb.add_persistent_tensor(format!("{a}rs"), bf, &[128, ng * t]);
    gb.add_node(
        "gaudium_ik_qprep",
        format!("{a}qprep"),
        &[qx, qsc, rsc, io.tau, io.cursor],
        &[qb, rs],
        &u32_params(&[
            0,
            d.nkv as u32,
            ik_rel_off(d.nh) as u32,
            u32::from(!sliding),
            f32_bits(d.qk_clip),
            f32_bits(d.rel_clip),
        ]),
    );

    // keys and values into the ring / cache
    let (kk, vk) = if sliding { ("kring", "vring") } else { ("kcache", "vcache") };
    let kc = gb.get_or_add_persistent_tensor(format!("{a}{kk}"), bf, &[128, c, b]);
    let vc = gb.get_or_add_persistent_tensor(format!("{a}{vk}"), bf, &[128, c, b]);
    let kcw = gb.get_or_add_persistent_tensor(format!("{l}kcw"), f32t, &[kvw, d.taps]);
    let vcw = gb.get_or_add_persistent_tensor(format!("{l}vcw"), f32t, &[kvw, d.taps]);
    let isd_kv = if sliding {
        ((d.taps + 1) as f32).powf(-0.5)
    } else {
        ((2 * d.taps + 1) as f32).powf(-0.5)
    };
    gb.add_node(
        "gaudium_ik_kvprep",
        format!("{a}kvprep"),
        &[kraw, vraw, kx, vx, kcw, vcw, io.cursor],
        &[kc, vc],
        &u32_params(&[
            0,
            u32::from(sliding),
            u32::from(!sliding),
            kvw as u32,
            f32_bits(isd_kv),
            f32_bits(d.qk_clip),
        ]),
    );

    // relative-position bias of every head against the backward distance (reversed, padded)
    let rs3 = ik_view(gb, rs, &format!("{a}rs3"), bf, &[128, ng * t, 1]);
    let proj = gb.get_or_add_persistent_tensor(format!("{l}PROJ"), bf, &[128, 8 * d.bw, 1]);
    let bias3 = gb.add_persistent_tensor(format!("{a}bias3"), f32t, &[8 * d.bw, ng * t, 1]);
    gb.op_batch_gemm(format!("{a}bias_gemm"), rs3, proj, bias3, false, true);
    let bias = ik_view(gb, bias3, &format!("{a}bias"), f32t, &[8 * d.bw, ng * t]);

    // scores, softmax numerator, values
    let scores = gb.add_persistent_tensor(format!("{a}scores"), f32t, &[c, r, b]);
    gb.op_batch_gemm(format!("{a}qk"), qb, kc, scores, false, true);
    let pm = gb.add_persistent_tensor(format!("{a}P"), bf, &[c, r, b]);
    let stats = gb.add_persistent_tensor(format!("{a}stats"), f32t, &[64, 2, r, b]);
    gb.add_node(
        "gaudium_ik_softmax",
        format!("{a}softmax"),
        &[scores, bias, io.cursor],
        &[pm, stats],
        &u32_params(&[
            t as u32,
            0,
            if sliding { d.window as u32 } else { 0 },
            u32::from(sliding),
            nchx as u32,
            d.gq as u32,
            d.boff as u32,
            f32_bits(128f32.powf(-0.5)),
        ]),
    );
    let ao_part = gb.add_persistent_tensor(format!("{a}ao_part"), f32t, &[128, r, b]);
    gb.op_batch_gemm(format!("{a}pv"), pm, vc, ao_part, false, false);
    let core = gb.add_persistent_tensor(format!("{a}core"), bf, &[128 * d.nh, t]);
    gb.add_node(
        "gaudium_ik_combine",
        format!("{a}combine"),
        &[ao_part, stats, qx, sink],
        &[core],
        &u32_params(&[nchx as u32, ik_gate_off(d.nh) as u32]),
    );
    let attn = ik_linear(gb, core, &format!("{l}W_o"), 128 * d.nh, hid, t, &format!("{a}attn"));

    // convolution over the attention output, and the f32 residual stream
    let isd = ((d.taps + 1) as f32).powf(-0.5);
    let araw = gb.get_or_add_persistent_tensor(format!("{a}araw"), bf, &[hid, d.hist]);
    gb.add_node("gaudium_ik_save", format!("{a}save_a"), &[attn, io.cursor], &[araw], &u32_params(&[0]));
    let acw = gb.get_or_add_persistent_tensor(format!("{l}acw"), f32t, &[hid, d.taps]);
    let resid = gb.add_persistent_tensor(format!("{a}resid"), f32t, &[hid, t]);
    gb.add_node(
        "gaudium_ik_conv",
        format!("{a}conv_a"),
        &[araw, acw, io.x_in, ctl, io.cursor],
        &[resid],
        &u32_params(&[0, f32_bits(isd)]),
    );

    // MLP input
    let q_mlp = gb.get_or_add_persistent_tensor(format!("{l}q_mlp"), f32t, &[hid, 1]);
    let hm = gb.add_persistent_tensor(format!("{a}hm"), f32t, &[hid, t, 1]);
    gb.add_node(
        "gaudium_ik_mix",
        format!("{a}mix_mlp"),
        &[
            io.states[0],
            io.states[1],
            io.states[2],
            io.states[3],
            resid,
            q_mlp,
            ctl,
            io.x_in,
        ],
        &[hm],
        &mix_params,
    );
    let hsrc = match msgs {
        Some(m) => append_ik_gate(gb, d, io, "mlp", hm, 0, m, 3, t),
        None => hm,
    };
    let w_mlp = gb.get_or_add_persistent_tensor(format!("{l}w_mlp"), bf, &[hid]);
    let h2 = gb.add_persistent_tensor(format!("{a}h2"), bf, &[hid, t]);
    gb.add_node(
        "gaudium_ik_norm",
        format!("{a}norm_mlp"),
        &[hsrc, w_mlp],
        &[h2],
        &u32_params(&[0, f32_bits(d.eps)]),
    );

    // router and dense-masked experts
    let wide = d.ne * d.ef;
    let rl = ik_linear(gb, h2, &format!("{l}W_router"), hid, d.ne, t, &format!("{a}rl"));
    let rb = gb.get_or_add_persistent_tensor(format!("{l}rb"), f32t, &[d.ne]);
    let wd2 = gb.add_persistent_tensor(format!("{a}wd2"), f32t, &[d.ne / 2, 2, t]);
    gb.add_node(
        "gaudium_ik_route",
        format!("{a}route"),
        &[rl, rb],
        &[wd2],
        &u32_params(&[d.ek as u32, f32_bits(d.route_scale)]),
    );
    let g = ik_linear(gb, h2, &format!("{l}W_gate"), hid, wide, t, &format!("{a}g"));
    let u = ik_linear(gb, h2, &format!("{l}W_up"), hid, wide, t, &format!("{a}u"));
    let act = gb.add_persistent_tensor(format!("{a}act"), bf, &[wide, t]);
    gb.add_node(
        "gaudium_ik_moe_act",
        format!("{a}moe_act"),
        &[g, u, wd2],
        &[act],
        &u32_params(&[d.ef as u32, f32_bits(d.gate_cap), f32_bits(d.up_cap)]),
    );
    let mo = ik_linear(gb, act, &format!("{l}W_down"), wide, hid, t, &format!("{a}mo"));

    // convolution over the expert output and the layer's output
    let mraw = gb.get_or_add_persistent_tensor(format!("{a}mraw"), bf, &[hid, d.hist]);
    gb.add_node("gaudium_ik_save", format!("{a}save_m"), &[mo, io.cursor], &[mraw], &u32_params(&[0]));
    let mcw = gb.get_or_add_persistent_tensor(format!("{l}mcw"), f32t, &[hid, d.taps]);
    let out = gb.add_persistent_tensor(format!("{a}out"), bf, &[hid, t]);
    gb.add_node(
        "gaudium_ik_conv",
        format!("{a}conv_m"),
        &[mraw, mcw, resid, io.cursor],
        &[out],
        &u32_params(&[0, f32_bits(isd)]),
    );
    out
}

impl IkDims {
    /// `eps` of the unweighted norm inside the block-residual mixes.
    fn eps_mix(&self) -> f32 {
        1.0e-6
    }
}

/// Appends the normed embedding gather: `x0 = norm(table[id])`.
fn append_ik_embed(gb: &mut GraphBuilder, d: &IkDims, ids: TensorId, t: u64) -> TensorId {
    let bf = SynDataType::Bf16;
    let table = gb.get_or_add_persistent_tensor("embed", bf, &[d.hid, d.vocab_in]);
    let w = gb.get_or_add_persistent_tensor("w_embed", bf, &[d.hid]);
    let e = gb.add_persistent_tensor("x0g", bf, &[d.hid, t]);
    let x0 = gb.add_persistent_tensor("x0", bf, &[d.hid, t]);
    gb.op_gather_bf16("embed_gather", table, ids, e, 1);
    gb.add_node(
        "gaudium_ga_rmsnorm",
        "embed_norm",
        &[e, w],
        &[x0],
        &d.eps.to_le_bytes(),
    );
    x0
}

/// Appends the final mix of the block states and the last layer's output, and the final norm.
fn append_ik_final(
    gb: &mut GraphBuilder,
    d: &IkDims,
    states: [TensorId; 4],
    x: TensorId,
    t: u64,
) -> TensorId {
    let f32t = SynDataType::Float;
    let bf = SynDataType::Bf16;
    let ctl = gb.get_or_add_persistent_tensor("ctl_fin", SynDataType::Int32, &[8]);
    let q = gb.get_or_add_persistent_tensor("q_fin", f32t, &[d.hid, 1]);
    let hf = gb.add_persistent_tensor("hf", f32t, &[d.hid, t, 1]);
    gb.add_node(
        "gaudium_ik_mix",
        "mix_fin",
        &[states[0], states[1], states[2], states[3], x, q, ctl],
        &[hf],
        &u32_params(&[f32_bits(d.eps_mix()), f32_bits((d.hid as f32).powf(-0.5))]),
    );
    let w = gb.get_or_add_persistent_tensor("w_final", bf, &[d.hid]);
    let h = gb.add_persistent_tensor("h_final", bf, &[d.hid, t]);
    gb.add_node(
        "gaudium_ik_norm",
        "norm_fin",
        &[hf, w],
        &[h],
        &u32_params(&[0, f32_bits(d.eps)]),
    );
    h
}

/// Appends the readout: logits over the padded unembedding and the arg-max over the rows the
/// model can emit. (The logit soft cap is monotonic: the greedy token needs no cap; a sampled
/// request applies it on the host to the row it reads.)
fn append_ik_head(gb: &mut GraphBuilder, d: &IkDims, h_final: TensorId, t: u64) -> (TensorId, TensorId) {
    let logits = ik_linear(gb, h_final, "W_head", d.hid, d.vocab_pad, t, "logits");
    let tokens = gb.add_persistent_tensor("tokens", SynDataType::Int32, &[1, t]);
    gb.add_node(
        "gaudium_ik_argmax",
        "argmax",
        &[logits],
        &[tokens],
        &u32_params(&[d.vocab_out as u32]),
    );
    (logits, tokens)
}

/// Picks the four state inputs of a mix: the given tensors, padded with `fill`.
fn four(states: &[TensorId], fill: TensorId) -> [TensorId; 4] {
    let mut out = [fill; 4];
    for (o, s) in out.iter_mut().zip(states) {
        *o = *s;
    }
    out
}

/// Monolithic decode step `g_ik_step_T{t}_C{nch}`: embedding, every layer application, the final
/// mix and norm, the readout. `nch`: chunks of the global caches the step reads (its bucket).
pub fn build_step_graph(
    d: &IkDims,
    lmax: u64,
    nch: u64,
    t: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = ik_graph(format!("g_ik_step_T{t}_C{nch}_A{}", d.n_apps), elf_hashes);
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[t]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let tau = gb.get_or_add_persistent_tensor("tau", SynDataType::Float, &[lmax]);
    let x0 = append_ik_embed(&mut gb, d, ids, t);
    let mut src = vec![x0];
    let ctl_msg = d
        .run_apps()
        .iter()
        .any(|a| a.msgs)
        .then(|| gb.get_or_add_persistent_tensor("ctl_msg", SynDataType::Int32, &[8]));
    for app in d.run_apps() {
        let w = &d.wires[app.index];
        let x_in = src[w.x_in];
        let st: Vec<TensorId> = w.attn_states.iter().map(|&s| src[s]).collect();
        let msg = match (app.msgs, w.msg_prev) {
            (true, Some(prev)) => {
                let ms: Vec<TensorId> = w.msg_states.iter().map(|&s| src[s]).collect();
                Some((four(&ms, src[prev]), src[prev], ctl_msg.expect("declared above")))
            }
            _ => None,
        };
        let lpfx = format!("L{}_", app.layer);
        let apfx = format!("A{}_", app.index);
        let io = IkIo {
            lpfx: &lpfx,
            apfx: &apfx,
            x_in,
            states: four(&st, x_in),
            msg,
            cursor,
            tau,
        };
        let kind = IkKind {
            sliding: app.sliding,
            msgs: app.msgs,
        };
        let out = append_ik_app(&mut gb, d, &io, kind, t, nch);
        src.push(out);
    }
    let last = *src.last().expect("at least the embedding");
    let fin: Vec<TensorId> = d.fin_states().iter().map(|&s| src[s]).collect();
    let h_final = append_ik_final(&mut gb, d, four(&fin, last), last, t);
    let _ = append_ik_head(&mut gb, d, h_final, t);
    gb
}

/// Prefill tile front end `g_ik_embed_T{t}`: `ids {t}` -> `x0 {hid, t}`.
pub fn build_embed_graph(d: &IkDims, t: u64, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = ik_graph(format!("g_ik_embed_T{t}"), elf_hashes);
    let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[t]);
    let _ = append_ik_embed(&mut gb, d, ids, t);
    gb
}

/// Generic single-application prefill recipe of one kind: `g_ik_<kind>_T{t}[_C{nch}]`.
/// Inputs under generic names: `x_in`, the block states `S0..S3`, for a kind with messages the
/// sources `P0..P3`, `ph` and `ctl_msg`; the layer's weights `L_*`; the application's state and
/// control words `A_*`. Output `A_out`.
pub fn build_app_graph(
    d: &IkDims,
    kind: IkKind,
    t: u64,
    lmax: u64,
    nch: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let name = if kind.sliding {
        format!("g_ik_{}_T{t}", kind.tag())
    } else {
        format!("g_ik_{}_T{t}_C{nch}", kind.tag())
    };
    let mut gb = ik_graph(name, elf_hashes);
    let bf = SynDataType::Bf16;
    let x_in = gb.add_persistent_tensor("x_in", bf, &[d.hid, t]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let tau = gb.get_or_add_persistent_tensor("tau", SynDataType::Float, &[lmax]);
    let mut states = [x_in; 4];
    for (i, s) in states.iter_mut().enumerate() {
        *s = gb.add_persistent_tensor(format!("S{i}"), bf, &[d.hid, t]);
    }
    let msg = kind.msgs.then(|| {
        let mut p = [x_in; 4];
        for (i, s) in p.iter_mut().enumerate() {
            *s = gb.add_persistent_tensor(format!("P{i}"), bf, &[d.hid, t]);
        }
        let ph = gb.add_persistent_tensor("ph", bf, &[d.hid, t]);
        let ctlp = gb.get_or_add_persistent_tensor("ctl_msg", SynDataType::Int32, &[8]);
        (p, ph, ctlp)
    });
    let io = IkIo {
        lpfx: "L_",
        apfx: "A_",
        x_in,
        states,
        msg,
        cursor,
        tau,
    };
    let _ = append_ik_app(&mut gb, d, &io, kind, t, nch);
    gb
}

/// Final mix and norm over a prefill tile `g_ik_final_T{t}`: the block states `S0..S3` and the
/// last layer's output `x_in` -> `h_final {hid, t}`.
pub fn build_final_graph(d: &IkDims, t: u64, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = ik_graph(format!("g_ik_final_T{t}"), elf_hashes);
    let bf = SynDataType::Bf16;
    let x_in = gb.add_persistent_tensor("x_in", bf, &[d.hid, t]);
    let mut states = [x_in; 4];
    for (i, s) in states.iter_mut().enumerate() {
        *s = gb.add_persistent_tensor(format!("S{i}"), bf, &[d.hid, t]);
    }
    let _ = append_ik_final(&mut gb, d, states, x_in, t);
    gb
}

/// One-row readout `g_ik_head_T1`: `h_in {hid, 1}` (a final-norm row) -> `logits`, `tokens`.
pub fn build_head_graph(d: &IkDims, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = ik_graph("g_ik_head_T1".into(), elf_hashes);
    let h_in = gb.add_persistent_tensor("h_in", SynDataType::Bf16, &[d.hid, 1]);
    let _ = append_ik_head(&mut gb, d, h_in, 1);
    gb
}

// ----------------------------------------------------------------------------
// Tensor roles (which arena a graph tensor's buffer belongs to)
// ----------------------------------------------------------------------------

const GLOBAL_WEIGHTS: &[&str] = &[
    "embed", "w_embed", "tau", "q_fin", "w_final", "W_head", "ctl_fin", "ctl_msg",
];
const KV_SUFFIXES: &[&str] = &["kcache", "vcache", "kring", "vring"];
const HIST_SUFFIXES: &[&str] = &["kraw", "vraw", "araw", "mraw"];

/// Splits `"L12_W_qx"` into `('L', Some(12), "W_qx")` and the generic `"A_ctl"` into
/// `('A', None, "ctl")`.
fn split_name(name: &str) -> Option<(char, Option<usize>, &str)> {
    let kind = name.chars().next().filter(|c| *c == 'L' || *c == 'A')?;
    let rest = &name[1..];
    let sep = rest.find('_')?;
    let idx = &rest[..sep];
    if idx.is_empty() {
        return Some((kind, None, &rest[sep + 1..]));
    }
    if idx.chars().all(|c| c.is_ascii_digit()) {
        return Some((kind, idx.parse().ok(), &rest[sep + 1..]));
    }
    None
}

fn is_weight_name(name: &str) -> bool {
    GLOBAL_WEIGHTS.contains(&name)
        || matches!(split_name(name), Some(('L', _, _)) | Some(('A', _, "ctl")))
}

fn is_kv_name(name: &str) -> bool {
    matches!(split_name(name), Some(('A', _, what)) if KV_SUFFIXES.contains(&what))
}

fn is_hist_name(name: &str) -> bool {
    matches!(split_name(name), Some(('A', _, what)) if HIST_SUFFIXES.contains(&what))
}

/// Tensors of a prefill recipe that the driver binds to buffers of its own (not shared scratch).
fn is_prefill_special(name: &str) -> bool {
    matches!(
        name,
        "x_in" | "A_out" | "cursor" | "ids" | "x0" | "h_final" | "h_in" | "ph"
    ) || (name.len() == 2
        && (name.starts_with('S') || name.starts_with('P'))
        && name.as_bytes()[1].is_ascii_digit())
}

fn align256(n: u64) -> u64 {
    align_up(n.max(256), 256)
}

fn gib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn mib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

/// What a speculative decode did after the first token.
#[derive(Debug, Default)]
struct SpecOutcome {
    launches: u64,
    drafted: u64,
    accepted: u64,
    decode_ms: f64,
    stopped_by_eos: bool,
    stopped_by_cancel: bool,
}

/// Unused state and message slots of a prefill recipe are bound to buffers of their own (a
/// launch refuses one address under two tensors).
const PF_DUMMIES: usize = 8;

/// Exact device-memory plan of one configuration, computed from the graphs before any allocation.
#[derive(Debug, Clone, Default)]
struct IkPlan {
    weights: BTreeMap<String, u64>,
    /// Global layers' K / V caches (grow with the context).
    kv: BTreeMap<String, u64>,
    /// Sliding layers' K / V rings and the convolutions' history rings (bounded).
    rings: BTreeMap<String, u64>,
    step_acts: BTreeMap<String, u64>,
    pf_scratch: BTreeMap<String, u64>,
    /// Bytes of one prefill row buffer `{hid, tile}`, and how many hold layer outputs.
    pf_x_bytes: u64,
    pf_bufs: usize,
    pf_ids_bytes: u64,
}

impl IkPlan {
    /// `step_rows`: the widest decode-step pass (1: the one-row step only).
    fn build(d: &IkDims, max_ctx: u64, max_chunks: u64, tile: u64, step_rows: u64) -> Self {
        let mut p = Self::default();
        let mut widths = vec![1u64];
        if step_rows > 1 {
            widths.push(step_rows);
        }
        for t in widths {
            let step = build_step_graph(d, max_ctx, max_chunks, t, &[]);
            for (name, _dt, _dims, bytes) in step.persistent_tensor_specs() {
                let map = if is_weight_name(&name) {
                    &mut p.weights
                } else if is_kv_name(&name) {
                    if name.ends_with("ring") {
                        &mut p.rings
                    } else {
                        &mut p.kv
                    }
                } else if is_hist_name(&name) {
                    &mut p.rings
                } else {
                    &mut p.step_acts
                };
                let e = map.entry(name).or_insert(0);
                *e = (*e).max(bytes);
            }
        }
        let mut graphs = vec![
            build_embed_graph(d, tile, &[]),
            build_final_graph(d, tile, &[]),
            build_head_graph(d, &[]),
        ];
        for kind in d.kinds() {
            graphs.push(build_app_graph(d, kind, tile, max_ctx, max_chunks, &[]));
        }
        for gb in &graphs {
            for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
                if is_prefill_special(&name)
                    || is_weight_name(&name)
                    || is_kv_name(&name)
                    || is_hist_name(&name)
                {
                    continue;
                }
                let e = p.pf_scratch.entry(name).or_insert(0);
                *e = (*e).max(bytes);
            }
        }
        let (_, n) = source_buffers(&d.wires[..d.n_apps], d.fin_states());
        p.pf_bufs = n;
        p.pf_x_bytes = d.hid * tile * 2;
        p.pf_ids_bytes = tile * 4;
        p
    }

    fn sum(m: &BTreeMap<String, u64>) -> u64 {
        m.values().map(|&b| align256(b)).sum()
    }

    fn weights_bytes(&self) -> u64 {
        Self::sum(&self.weights)
    }

    fn kv_bytes(&self) -> u64 {
        Self::sum(&self.kv)
    }

    fn ring_bytes(&self) -> u64 {
        Self::sum(&self.rings)
    }

    fn act_bytes(&self) -> u64 {
        Self::sum(&self.step_acts)
            + Self::sum(&self.pf_scratch)
            + (self.pf_bufs + PF_DUMMIES + 1) as u64 * align256(self.pf_x_bytes)
            + align256(self.pf_ids_bytes)
            + 256
    }

    fn total(&self, ws_bytes: u64) -> u64 {
        self.weights_bytes()
            .saturating_add(self.kv_bytes())
            .saturating_add(self.ring_bytes())
            .saturating_add(self.act_bytes())
            .saturating_add(ws_bytes)
    }
}

/// Host-only device-memory plan of an Inkling 2 configuration, in bytes:
/// `[weights + tables, global K/V caches, rings, activations]` (the shared recipe workspace comes
/// on top). Exactly what `Inkling2Engine::load` allocates for the same inputs.
pub fn hbm_plan_bytes(spec: &ModelSpec, max_ctx: u64, prefill_tile: u64) -> Result<[u64; 4], String> {
    let max_ctx = align_up(max_ctx.max(IK_CHUNK), IK_CHUNK);
    let tile = prefill_tile.clamp(1, IK_MAX_PREFILL_TILE);
    let d = IkDims::from_spec(spec, tile)?;
    let p = IkPlan::build(&d, max_ctx, max_ctx / IK_CHUNK, tile, 1);
    Ok([p.weights_bytes(), p.kv_bytes(), p.ring_bytes(), p.act_bytes()])
}

fn select_exact_bindings(
    recipe: &CompiledRecipe,
    pool: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, u64>, String> {
    let mut exact = BTreeMap::new();
    for t in recipe.tensors() {
        let &addr = pool.get(&t.name).ok_or_else(|| {
            format!(
                "recipe {} requires persistent tensor '{}' which has no allocated HBM buffer",
                recipe.name, t.name
            )
        })?;
        exact.insert(t.name.clone(), addr);
    }
    Ok(exact)
}

/// Reads the context limit (`model_max_length` or `max_position_embeddings` of the text config)
/// and the generation config's extra end-of-sequence ids.
fn read_model_limits(cfg: &EngineConfig) -> (u64, Vec<i32>) {
    let mut max_pos = 0u64;
    if let Ok(txt) = std::fs::read_to_string(cfg.model_dir.join("config.json")) {
        if let Ok(root) = parse_json(&txt) {
            let tc = root.get("text_config").unwrap_or(&root);
            for key in ["max_position_embeddings", "model_max_length"] {
                if let Some(v) = tc.get(key).and_then(JsonValue::as_u64) {
                    max_pos = max_pos.max(v);
                }
            }
        }
    }
    let mut eos = Vec::new();
    for file in ["generation_config.json", "config.json"] {
        if let Ok(txt) = std::fs::read_to_string(cfg.model_dir.join(file)) {
            if let Ok(root) = parse_json(&txt) {
                if let Some(v) = root.get("eos_token_id") {
                    if let Some(id) = v.as_i64() {
                        eos.push(id as i32);
                    } else if let Some(arr) = v.as_array() {
                        eos.extend(arr.iter().filter_map(JsonValue::as_i64).map(|x| x as i32));
                    }
                }
            }
        }
    }
    eos.sort_unstable();
    eos.dedup();
    (max_pos, eos)
}

// ============================================================================
// Live Gaudi 2 Inkling 2 engine
// ============================================================================

/// Live Intel Gaudi 2 execution engine for Inkling 2 decoders (BF16 weights, `K = 0`).
///
/// Field declaration order ensures all `CompiledRecipe`s and `PinnedBuf`s are dropped
/// **before** `dev: HpuDevice` (which drains all streams and releases the device).
pub struct Inkling2Engine {
    spec: ModelSpec,
    dims: IkDims,
    max_ctx: u64,
    max_chunks: u64,
    prefill_tile: u64,
    eos_ids: Vec<i32>,
    verbose: bool,
    driver_version: String,
    cache: RecipeCache,
    elf_hashes: Vec<(String, String)>,
    /// Debug: prefill the prompt one row at a time with the decode step.
    step_prefill: bool,
    /// Debug: directory the step's tensors are written to after the prompt.
    dump_dir: Option<std::path::PathBuf>,
    /// Debug: rows of the passes a greedy request is run again with (0: off).
    verify_check: u64,
    /// Speculative decoding: the drafter and the rows of a verify pass (the last token and its
    /// drafts); `None`: every request is `K = 0`.
    draft: Option<SpecConfig>,
    spec_rows: u64,
    /// Requests served so far (the host oracle drafter's streams are per request).
    requests: usize,
    // reuse: the rows the card holds for the next request, and the copy of the rings
    reuse: reuse::IkReuse,

    // Recipes (must drop before `dev`)
    head: CompiledRecipe,
    step_recipes: BTreeMap<(u64, u64), (CompiledRecipe, PreparedLaunch)>, // (rows, bucket)
    embed_recipes: HashMap<u64, CompiledRecipe>,                   // T
    final_recipes: HashMap<u64, CompiledRecipe>,                   // T
    app_recipes: HashMap<(IkKind, u64, u64), CompiledRecipe>,      // (kind, T, chunks; 0 sliding)
    /// Per application: the pre-validated prefill launch of its current `(T, chunks)`.
    pf_prepared: Vec<Option<(u64, u64, PreparedLaunch)>>,

    // Reusable pinned host staging buffer (must drop before `dev`)
    stage_io: PinnedBuf,
    // vision: the image tower and its request state (recipes and pinned memory: before `dev`)
    vision: Option<vision::IkVision>,

    // HBM address maps (sub-allocated from sealed arenas owned by `dev`)
    weights_addr: BTreeMap<String, u64>,
    kv_addr: BTreeMap<String, (u64, u64)>,
    step_act_addr: BTreeMap<String, (u64, u64)>,
    pf_scratch: BTreeMap<String, u64>,
    /// Row buffers of a prefill tile: the layer outputs' buffers, then the dummies.
    pf_buf: Vec<u64>,
    pf_dummy: Vec<u64>,
    /// The buffer of each source of the wiring.
    buf_of_src: Vec<usize>,
    pf_ids: u64,
    pf_cursor: u64,
    pf_hfin: u64,
    hbm_bytes: [u64; 5],
    live: LiveReporter,

    // Device handle (dropped LAST)
    dev: HpuDevice,
}

// SAFETY: `Inkling2Engine` exclusively owns its SynapseAI device context, streams, compiled
// recipes, pinned host buffers and launch descriptors; no handle is accessed concurrently, so
// moving ownership across threads (behind a `Mutex` in the server) is safe.
unsafe impl Send for Inkling2Engine {}

impl Inkling2Engine {
    /// Opens the card, validates the model, plans and seals device memory, uploads the weights.
    pub fn load(cfg: &EngineConfig) -> Result<Self, String> {
        let t_start = Instant::now();
        let live =
            LiveReporter::from_env("gaudium-rust", env!("CARGO_PKG_VERSION"), "engine", true);
        let model_name = cfg
            .model_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("gaudium-model")
            .to_string();
        live.set_model(&model_name);
        live.set_phase("loading", "opening device and compiling recipes");

        // A GGUF file (or a directory of its shards) carries the geometry in its metadata.
        let gguf = if crate::gguf::looks_like_gguf(&cfg.model_dir) {
            Some(Arc::new(
                crate::gguf::Gguf::open(&cfg.model_dir).map_err(|e| format!("open GGUF: {e}"))?,
            ))
        } else {
            None
        };
        let spec = match &gguf {
            Some(g) => ModelSpec::from_config_json(&gguf_source::config_json(g)?)
                .map_err(|e| format!("GGUF metadata: {e}"))?,
            None => ModelSpec::from_config_file(cfg.model_dir.join("config.json"))
                .map_err(|e| format!("load config.json: {e}"))?,
        };
        if spec.family != ModelFamily::Inkling2 {
            return Err("Inkling2Engine: the model is not an Inkling 2 model".into());
        }
        spec.validate()
            .map_err(|e| format!("verified ModelSpec validation failed: {e}"))?;
        if cfg.precision == crate::engine::fp8_graph::WeightPrecision::Fp8 {
            return Err(
                "Inkling 2 runs its checkpoint's own precision in BF16; FP8 weights are not available for this family yet"
                    .into(),
            );
        }
        // Speculative decoding: the rows of a verify pass are the last token and its drafts.
        let spec_cfg = cfg.spec.clone();
        let spec_rows = match &spec_cfg {
            Some(sc) => {
                if !matches!(sc.method, SpecMethod::Oracle { .. }) {
                    return Err(format!(
                        "Inkling 2: the '{}' drafter is not wired up for this family yet (the host oracle drafter is)",
                        sc.method_name()
                    ));
                }
                let k = sc.draft_k.iter().copied().max().unwrap_or(7);
                if k == 0 || k > 63 {
                    return Err(format!("Inkling 2: --draft-k {k} is outside 1..63"));
                }
                k + 1
            }
            None => 1,
        };

        let (max_pos, gen_eos) = match &gguf {
            Some(g) => (
                g.arch_kv("context_length").and_then(|v| v.as_u64()).unwrap_or(0),
                g.get_u64("tokenizer.ggml.eos_token_id")
                    .map(|e| vec![e as i32])
                    .unwrap_or_default(),
            ),
            None => read_model_limits(cfg),
        };
        let max_ctx = align_up(cfg.max_ctx.max(IK_CHUNK), IK_CHUNK);
        if max_pos > 0 && max_ctx > max_pos {
            return Err(format!(
                "--max-ctx {max_ctx} exceeds the model's context limit {max_pos}"
            ));
        }
        let max_chunks = max_ctx / IK_CHUNK;
        let mut eos_ids = spec.eos_token_ids.clone();
        for id in gen_eos {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
        if let Some(ik) = &spec.inkling2 {
            // padding ids are never emitted as text
            let _ = ik;
        }

        // Debug knobs (documented in `--help`).
        let apps_cut = std::env::var("GAUDIUM_IK_APPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let step_prefill = std::env::var("GAUDIUM_IK_STEP_PREFILL").is_ok_and(|v| v == "1");
        let dump_dir = std::env::var_os("GAUDIUM_IK_DUMP").map(std::path::PathBuf::from);
        // Debug: after a greedy request, its tokens are run again in passes of this many rows and
        // every row's logits are compared with the one-row step's, bit for bit.
        let verify_check = std::env::var("GAUDIUM_IK_VERIFY_CHECK")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&t| t > 1 && t <= 64)
            .unwrap_or(0);
        let step_rows = verify_check.max(1).max(spec_rows);

        // Configure kernel ELFs and the graph-compiler glue BEFORE SynapseAI is loaded.
        let kenv = KernelEnv::configure_from_dir(&cfg.kernels_dir)?;
        if !kenv.gc_kernel_path.contains(INKLING2_GLUE) {
            return Err(format!(
                "the Inkling 2 glue library {INKLING2_GLUE} is missing from {}",
                cfg.kernels_dir.display()
            ));
        }
        let mut elf_hashes = kenv.fingerprint_hashes(INKLING2_ELFS)?;
        elf_hashes.push(kenv.glue_hash(INKLING2_GLUE)?);

        let api =
            Arc::new(SynapseApi::load_default().map_err(|e| format!("load libSynapse.so: {e}"))?);
        let driver_version = api.driver_version().unwrap_or_else(|_| "unknown".into());
        let mut dev = HpuDevice::open_gaudi2(Arc::clone(&api))
            .map_err(|e| format!("open Gaudi 2 device: {e}"))?;
        let (free_hbm, total_hbm) = dev
            .memory_info()
            .map_err(|e| format!("query HBM info: {e}"))?;

        let cache = if cfg.no_cache {
            RecipeCache::disabled()
        } else if let Some(dir) = &cfg.cache_dir {
            RecipeCache::at_dir(dir)
        } else {
            RecipeCache::from_env()
        };

        // vision: the image tower's share of the plan (none without image input)
        // (a GGUF file carries no tower: text only)
        let mut vision = match gguf {
            Some(_) => None,
            None => vision::IkVision::plan(cfg, &IkDims::from_spec(&spec, IK_MAX_PREFILL_TILE)?)?,
        };
        let vis_bytes = vision.as_ref().map_or(0, |v| v.plan_bytes().0 + v.plan_bytes().1);

        // reuse: on with `--prefix-cache` > 0 (the debug modes that write rows out of order
        // keep it off)
        let reuse_on = cfg.prefix_slots > 0 && !step_prefill && verify_check == 0;

        // ---- Plan: the largest prefill tile whose exact plan fits the card ----
        let mut tile = cfg.prefill_tile.clamp(1, IK_MAX_PREFILL_TILE);
        let (dims, plan) = loop {
            // rings hold their window beside the widest pass (a prefill tile or a multi-row step)
            let mut d = IkDims::from_spec(&spec, tile.max(step_rows))?;
            if let Some(n) = apps_cut {
                d.n_apps = n.clamp(1, d.apps.len());
            }
            let p = IkPlan::build(&d, max_ctx, max_chunks, tile, step_rows);
            // reuse: the copy of the rings is as large as the rings
            let total = p.total(MIN_SHARED_WORKSPACE_BYTES)
                + vis_bytes
                + if reuse_on { p.ring_bytes() } else { 0 };
            match dev.check_plan_fits(total, HBM_HEADROOM_BYTES) {
                Ok(_) => break (d, p),
                Err(e) if tile <= IK_MIN_PREFILL_TILE => {
                    return Err(format!(
                        "HBM plan does not fit at max_ctx={max_ctx} even with a {tile}-row prefill tile: planned {:.2} GiB (weights {:.2}, global KV {:.2}, rings {:.2}, activations {:.2}, workspace {:.2}) of {:.2} GiB free: {e}",
                        gib(total),
                        gib(p.weights_bytes()),
                        gib(p.kv_bytes()),
                        gib(p.ring_bytes()),
                        gib(p.act_bytes()),
                        gib(MIN_SHARED_WORKSPACE_BYTES),
                        gib(free_hbm),
                    ));
                }
                Err(_) => {
                    if cfg.verbose {
                        eprintln!(
                            "[gaudium] inkling2: plan with a {tile}-row prefill tile needs {:.2} GiB (activations {:.2} GiB); trying {}",
                            gib(total),
                            gib(p.act_bytes()),
                            tile / 2
                        );
                    }
                    tile /= 2;
                }
            }
        };
        let prefill_tile = tile;

        if cfg.verbose {
            eprintln!(
                "[gaudium] device=Gaudi2 driver={} HBM={:.2}/{:.2} GiB family=Inkling2 weights=BF16 layers={} applications={} (global={}, sliding={}) hidden={} heads={}q/{}kv experts={} (top {}) x {} window={} ring={} hist={} max_ctx={} ({} chunks) tile={}",
                driver_version,
                gib(free_hbm),
                gib(total_hbm),
                dims.n_layers,
                dims.n_apps,
                dims.run_apps().iter().filter(|a| !a.sliding).count(),
                dims.run_apps().iter().filter(|a| a.sliding).count(),
                dims.hid,
                dims.nh,
                dims.nkv,
                dims.ne,
                dims.ek,
                dims.ef,
                dims.window,
                dims.ring,
                dims.hist,
                max_ctx,
                max_chunks,
                prefill_tile,
            );
        }

        // ---- Compile the recipes every request needs, before any device allocation ----
        let t_comp = Instant::now();
        let note = |r: &CompiledRecipe| {
            if cfg.verbose {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                    r.name,
                    r.from_cache,
                    mib(r.ws_size)
                );
            }
        };
        let mut head = build_head_graph(&dims, &elf_hashes)
            .compile(Arc::clone(&api), &cache, &driver_version)
            .map_err(|e| format!("compile head: {e}"))?;
        note(&head);
        let mut step1 = build_step_graph(&dims, max_ctx, 1, 1, &elf_hashes)
            .compile(Arc::clone(&api), &cache, &driver_version)
            .map_err(|e| format!("compile decode step (1 chunk): {e}"))?;
        note(&step1);
        // vision: the tower's largest planned recipes, so the shared workspace covers them
        let vis_ws = match vision.as_mut() {
            Some(v) => v.compile_planned(&api, &cache, &driver_version, cfg.verbose)?,
            None => 0,
        };
        let max_ws = head.ws_size.max(step1.ws_size).max(vis_ws);
        if cfg.verbose {
            eprintln!(
                "[gaudium] core recipes ready in {:.2}s (max_ws={:.2} MiB)",
                t_comp.elapsed().as_secs_f64(),
                mib(max_ws)
            );
        }

        // ---- Exact arena sizes, final fit check, allocation, seal ----
        let ws_bytes = align256(max_ws.max(MIN_SHARED_WORKSPACE_BYTES));
        let weights_bytes = plan.weights_bytes();
        let kv_bytes = plan.kv_bytes();
        let ring_bytes = plan.ring_bytes();
        let snap_bytes = if reuse_on { ring_bytes } else { 0 };
        let act_bytes = plan.act_bytes();
        let planned_total = plan.total(ws_bytes) + vis_bytes + snap_bytes;
        dev.check_plan_fits(planned_total, HBM_HEADROOM_BYTES)
            .map_err(|e| format!("HBM plan check failed: {e}"))?;
        if cfg.verbose {
            eprintln!(
                "[gaudium] HBM plan fits: total={:.2} GiB of {:.2} GiB free | weights={:.2} GiB, global_kv={:.2} GiB, rings={:.2} GiB, acts={:.2} GiB (step {:.1} MiB, prefill scratch {:.1} MiB), ws={:.2} MiB",
                gib(planned_total),
                gib(free_hbm),
                gib(weights_bytes),
                gib(kv_bytes),
                gib(ring_bytes),
                gib(act_bytes),
                mib(IkPlan::sum(&plan.step_acts)),
                mib(IkPlan::sum(&plan.pf_scratch)),
                mib(ws_bytes),
            );
        }

        let mut weights_arena = dev
            .alloc_arena(weights_bytes, HbmCat::CatWeights, "ik_weights_arena")
            .map_err(|e| format!("alloc weights arena: {e}"))?;
        let mut kv_arena = dev
            .alloc_arena(kv_bytes.max(256), HbmCat::CatKv, "ik_global_kv_arena")
            .map_err(|e| format!("alloc global KV arena: {e}"))?;
        let mut ring_arena = dev
            .alloc_arena((ring_bytes + snap_bytes).max(256), HbmCat::CatKv, "ik_ring_arena")
            .map_err(|e| format!("alloc ring arena: {e}"))?;
        let mut act_arena = dev
            .alloc_arena(act_bytes, HbmCat::CatScratch, "ik_act_arena")
            .map_err(|e| format!("alloc act arena: {e}"))?;

        let suballoc = |arena: &mut HbmArena, bytes: u64| -> Result<u64, String> {
            let (addr, _off) = arena
                .suballoc(bytes.max(256), 256)
                .map_err(|e| format!("arena suballoc: {e}"))?;
            Ok(addr)
        };

        let mut weights_addr = BTreeMap::new();
        for (name, &bytes) in &plan.weights {
            weights_addr.insert(name.clone(), suballoc(&mut weights_arena, bytes)?);
        }
        let mut kv_addr = BTreeMap::new();
        for (name, &bytes) in &plan.kv {
            kv_addr.insert(name.clone(), (suballoc(&mut kv_arena, bytes)?, bytes));
        }
        for (name, &bytes) in &plan.rings {
            kv_addr.insert(name.clone(), (suballoc(&mut ring_arena, bytes)?, bytes));
        }
        // reuse: the copy of the rings lies behind them
        let mut reuse = reuse::IkReuse::new(
            reuse_on,
            dims.ring,
            dims.window,
            dims.hist,
            dims.taps,
        );
        if reuse_on {
            let (lo, used) = (ring_arena.base(), ring_arena.used());
            if used != ring_bytes {
                return Err(format!(
                    "ring arena holds {used} bytes where the plan has {ring_bytes}"
                ));
            }
            reuse.bind(lo, ring_bytes, suballoc(&mut ring_arena, ring_bytes)?);
        }
        let mut step_act_addr = BTreeMap::new();
        for (name, &bytes) in &plan.step_acts {
            step_act_addr.insert(name.clone(), (suballoc(&mut act_arena, bytes)?, bytes));
        }
        let mut pf_scratch = BTreeMap::new();
        for (name, &bytes) in &plan.pf_scratch {
            pf_scratch.insert(name.clone(), suballoc(&mut act_arena, bytes)?);
        }
        let mut pf_buf = Vec::with_capacity(plan.pf_bufs);
        for _ in 0..plan.pf_bufs {
            pf_buf.push(suballoc(&mut act_arena, plan.pf_x_bytes)?);
        }
        let mut pf_dummy = Vec::with_capacity(PF_DUMMIES);
        for _ in 0..PF_DUMMIES {
            pf_dummy.push(suballoc(&mut act_arena, plan.pf_x_bytes)?);
        }
        let pf_hfin = suballoc(&mut act_arena, plan.pf_x_bytes)?;
        let pf_ids = suballoc(&mut act_arena, plan.pf_ids_bytes)?;
        let pf_cursor = suballoc(&mut act_arena, 256)?;
        let (buf_of_src, _) = source_buffers(&dims.wires[..dims.n_apps], dims.fin_states());
        if let Some(v) = vision.as_mut() {
            v.allocate(&mut dev)?;
        }

        dev.seal_with_shared_workspace(ws_bytes)
            .map_err(|e| format!("seal HBM allocations: {e}"))?;

        let (vis_w, vis_a) = vision.as_ref().map_or((0, 0), |v| v.plan_bytes());
        let hbm_bytes = [
            weights_bytes + vis_w,
            kv_bytes,
            ring_bytes + snap_bytes,
            act_bytes + vis_a,
            ws_bytes,
        ];
        live.set_hbm_bytes(
            hbm_bytes[0],
            hbm_bytes[1],
            hbm_bytes[2],
            hbm_bytes[3],
            hbm_bytes[4],
        );
        live.set_phase("loading", "uploading weights to HBM");

        head.bind_shared_workspace(&dev)
            .map_err(|e| format!("bind head ws: {e}"))?;
        step1
            .bind_shared_workspace(&dev)
            .map_err(|e| format!("bind step ws: {e}"))?;

        // ---- Upload weights and tables from the mmap'd checkpoint ----
        let t_up = Instant::now();
        let store;
        let st_source;
        let gg_source;
        let source: &dyn packer::IkSource = match &gguf {
            Some(g) => {
                gg_source = gguf_source::GgufSource::new(Arc::clone(g))?;
                &gg_source
            }
            None => {
                store = crate::safetensors::SafetensorsStore::open(&cfg.model_dir)
                    .map_err(|e| format!("open safetensors: {e}"))?;
                st_source = packer::StSource::new(&store)
                    .with_amax_bf16(std::env::var("GAUDIUM_IK_AMAX_BF16").is_ok_and(|v| v == "1"));
                &st_source
            }
        };
        let packer = packer::IkPacker::new(source, &dims, max_ctx);
        let mut stage_upload = dev
            .alloc_pinned(STAGE_BUF_BYTES)
            .map_err(|e| format!("alloc pinned upload stage: {e}"))?;
        let mut uploaded_bytes = 0u64;
        for (name, &addr) in &weights_addr {
            let expected = plan.weights[name];
            let data = packer.pack(name)?;
            if data.len() as u64 != expected {
                return Err(format!(
                    "weight '{name}' packed size {} != graph tensor size {expected}",
                    data.len()
                ));
            }
            dev.upload_h2d_chunked(&data, addr, &mut stage_upload)
                .map_err(|e| format!("upload weight '{name}': {e}"))?;
            uploaded_bytes += expected;
        }
        // vision: the tower's weights come from the checkpoint's safetensors file
        if let Some(v) = vision.as_mut() {
            let vstore = crate::safetensors::SafetensorsStore::open(&cfg.model_dir)
                .map_err(|e| format!("open safetensors (vision tower): {e}"))?;
            uploaded_bytes += v.upload(&dev, &vstore, &mut stage_upload)?;
        }
        if cfg.verbose {
            let dt = t_up.elapsed().as_secs_f64();
            eprintln!(
                "[gaudium] uploaded {:.2} GiB weights in {:.2}s ({:.2} GiB/s)",
                gib(uploaded_bytes),
                dt,
                gib(uploaded_bytes) / dt.max(1e-6),
            );
        }

        // ---- Zero the K / V state and the history rings once: a masked key must be a finite number ----
        stage_upload.as_mut_slice().fill(0);
        let stage_zero = stage_upload;
        let t_zero = Instant::now();
        for &(addr, bytes) in kv_addr.values() {
            dev.zero_hbm_chunked(addr, bytes, &stage_zero)
                .map_err(|e| format!("zero K/V state: {e}"))?;
        }
        if cfg.verbose {
            eprintln!(
                "[gaudium] zeroed {:.2} GiB of K/V state in {:.2}s; total init {:.2}s",
                gib(kv_bytes + ring_bytes),
                t_zero.elapsed().as_secs_f64(),
                t_start.elapsed().as_secs_f64()
            );
        }
        drop(stage_zero);

        let stage_io_bytes = ((dims.vocab_pad as usize) * 2)
            .max((prefill_tile as usize) * 4)
            .max(16 * 1024 * 1024);
        let stage_io = dev
            .alloc_pinned(stage_io_bytes)
            .map_err(|e| format!("alloc pinned io stage: {e}"))?;

        let n_apps = dims.n_apps;
        let mut engine = Self {
            spec,
            dims,
            max_ctx,
            max_chunks,
            prefill_tile,
            eos_ids,
            verbose: cfg.verbose,
            driver_version,
            cache,
            elf_hashes,
            step_prefill,
            dump_dir,
            verify_check,
            draft: spec_cfg,
            spec_rows,
            requests: 0,
            reuse,
            head,
            step_recipes: BTreeMap::new(),
            embed_recipes: HashMap::new(),
            final_recipes: HashMap::new(),
            app_recipes: HashMap::new(),
            pf_prepared: (0..n_apps).map(|_| None).collect(),
            stage_io,
            vision,
            weights_addr,
            kv_addr,
            step_act_addr,
            pf_scratch,
            pf_buf,
            pf_dummy,
            buf_of_src,
            pf_ids,
            pf_cursor,
            pf_hfin,
            hbm_bytes,
            live,
            dev,
        };
        let prepared = engine.prepare_step(&step1)?;
        engine.step_recipes.insert((1, 1), (step1, prepared));
        engine.live.set_phase("idle", "ready");
        Ok(engine)
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn max_ctx(&self) -> u64 {
        self.max_ctx
    }

    /// The prefill tile the memory plan settled on.
    pub fn prefill_tile(&self) -> u64 {
        self.prefill_tile
    }

    pub fn live_profile(&self) -> crate::live_report::ProfileHandle {
        self.live.profile_handle()
    }

    /// Exact 5-category HBM ledger bytes `[weights, global KV, rings, activations, workspace]`.
    pub fn hbm_bytes(&self) -> [u64; 5] {
        self.hbm_bytes
    }

    fn compile(&self, gb: &GraphBuilder, what: &str) -> Result<CompiledRecipe, String> {
        let t0 = Instant::now();
        let mut r = gb
            .compile(
                Arc::clone(self.dev.api()),
                &self.cache,
                &self.driver_version,
            )
            .map_err(|e| format!("compile {what}: {e}"))?;
        r.bind_shared_workspace(&self.dev)
            .map_err(|e| format!("bind {what} workspace: {e}"))?;
        if self.verbose {
            eprintln!(
                "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                r.name,
                r.from_cache,
                mib(r.ws_size),
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(r)
    }

    /// Pre-validates the one binding list of a decode-step recipe.
    fn prepare_step(&self, r: &CompiledRecipe) -> Result<PreparedLaunch, String> {
        let mut pool = BTreeMap::new();
        for (k, &v) in &self.weights_addr {
            pool.insert(k.clone(), v);
        }
        for (k, &(v, _)) in &self.kv_addr {
            pool.insert(k.clone(), v);
        }
        for (k, &(v, _)) in &self.step_act_addr {
            pool.insert(k.clone(), v);
        }
        let exact = select_exact_bindings(r, &pool)?;
        r.prepare_launch(&exact)
            .map_err(|e| format!("prepare decode step launch: {e}"))
    }

    fn ensure_step(&mut self, rows: u64, bucket: u64) -> Result<(), String> {
        if self.step_recipes.contains_key(&(rows, bucket)) {
            return Ok(());
        }
        let gb = build_step_graph(&self.dims, self.max_ctx, bucket, rows, &self.elf_hashes);
        let r = self.compile(&gb, &format!("decode step ({rows} rows, {bucket} chunks)"))?;
        let prepared = self.prepare_step(&r)?;
        self.step_recipes.insert((rows, bucket), (r, prepared));
        Ok(())
    }

    fn ensure_embed(&mut self, t: u64) -> Result<(), String> {
        if !self.embed_recipes.contains_key(&t) {
            let gb = build_embed_graph(&self.dims, t, &self.elf_hashes);
            let r = self.compile(&gb, &format!("embed T={t}"))?;
            self.embed_recipes.insert(t, r);
        }
        Ok(())
    }

    fn ensure_final(&mut self, t: u64) -> Result<(), String> {
        if !self.final_recipes.contains_key(&t) {
            let gb = build_final_graph(&self.dims, t, &self.elf_hashes);
            let r = self.compile(&gb, &format!("final mix T={t}"))?;
            self.final_recipes.insert(t, r);
        }
        Ok(())
    }

    fn ensure_app(&mut self, kind: IkKind, t: u64, nch: u64) -> Result<(), String> {
        let key = (kind, t, if kind.sliding { 0 } else { nch });
        if !self.app_recipes.contains_key(&key) {
            let gb = build_app_graph(&self.dims, kind, t, self.max_ctx, nch, &self.elf_hashes);
            let r = self.compile(&gb, &format!("prefill application {} T={t} C={nch}", kind.tag()))?;
            self.app_recipes.insert(key, r);
        }
        Ok(())
    }

    /// The row buffer that holds source `s` of the wiring during a prefill tile.
    fn src_buf(&self, s: usize) -> u64 {
        self.pf_buf[self.buf_of_src[s]]
    }

    /// Launches application `a` of a prefill tile of `t` rows (`nch`: chunks a global layer reads).
    fn launch_prefill_app(&mut self, a: usize, t: u64, nch: u64) -> Result<(), String> {
        let app = self.dims.apps[a];
        let kind = IkKind {
            sliding: app.sliding,
            msgs: app.msgs,
        };
        let nkey = if kind.sliding { 0 } else { nch };
        let fresh = !matches!(&self.pf_prepared[a], Some((pt, pn, _)) if *pt == t && *pn == nkey);
        if fresh {
            let recipe = &self.app_recipes[&(kind, t, nkey)];
            let w = &self.dims.wires[a];
            let mut pool = self.pf_scratch.clone();
            pool.insert("x_in".into(), self.src_buf(w.x_in));
            pool.insert("A_out".into(), self.src_buf(a + 1));
            pool.insert("cursor".into(), self.pf_cursor);
            pool.insert("tau".into(), self.weights_addr["tau"]);
            for i in 0..IK_MAX_STATES as usize {
                let addr = match w.attn_states.get(i) {
                    Some(&s) => self.src_buf(s),
                    None => self.pf_dummy[i],
                };
                pool.insert(format!("S{i}"), addr);
            }
            if kind.msgs {
                for i in 0..IK_MAX_STATES as usize {
                    let addr = match w.msg_states.get(i) {
                        Some(&s) => self.src_buf(s),
                        None => self.pf_dummy[IK_MAX_STATES as usize + i],
                    };
                    pool.insert(format!("P{i}"), addr);
                }
                let prev = w
                    .msg_prev
                    .ok_or_else(|| format!("application {a} has messages but no previous pass"))?;
                pool.insert("ph".into(), self.src_buf(prev));
                pool.insert("ctl_msg".into(), self.weights_addr["ctl_msg"]);
            }
            // the layer's weights and the application's state and control words, under their
            // generic names
            let lp = format!("L{}_", app.layer);
            for (name, &addr) in self
                .weights_addr
                .range(lp.clone()..)
                .take_while(|(name, _)| name.starts_with(&lp))
            {
                pool.insert(format!("L_{}", &name[lp.len()..]), addr);
            }
            let ap = format!("A{a}_");
            pool.insert("A_ctl".into(), self.weights_addr[&format!("{ap}ctl")]);
            for (name, &(addr, _)) in self
                .kv_addr
                .range(ap.clone()..)
                .take_while(|(name, _)| name.starts_with(&ap))
            {
                pool.insert(format!("A_{}", &name[ap.len()..]), addr);
            }
            let exact = select_exact_bindings(recipe, &pool)?;
            let prepared = recipe
                .prepare_launch(&exact)
                .map_err(|e| format!("prepare prefill application {a} T={t}: {e}"))?;
            self.pf_prepared[a] = Some((t, nkey, prepared));
        }
        let recipe = &self.app_recipes[&(kind, t, nkey)];
        let prepared = &self.pf_prepared[a].as_ref().expect("prepared above").2;
        recipe
            .launch_prepared(&self.dev, prepared)
            .map_err(|e| format!("launch prefill application {a} T={t} C={nch}: {e}"))
    }

    fn sync_compute(&self, what: &str) -> Result<(), String> {
        // SAFETY: the compute stream is owned by `self.dev` and alive for its lifetime.
        unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
            .map_err(|e| format!("sync {what}: {e}"))
    }

    fn write_i32_scalar(&mut self, hbm_addr: u64, val: i32) -> Result<(), String> {
        self.stage_io.as_mut_slice()[..4].copy_from_slice(&val.to_le_bytes());
        self.dev
            .copy_h2d_pinned_sync(&self.stage_io, 4, hbm_addr)
            .map_err(|e| format!("write i32 H2D: {e}"))
    }

    fn read_i32_scalar(&mut self, hbm_addr: u64) -> Result<i32, String> {
        self.dev
            .copy_d2h_pinned_sync(hbm_addr, &mut self.stage_io, 4)
            .map_err(|e| format!("read i32 D2H: {e}"))?;
        let b = &self.stage_io.as_slice()[..4];
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Samples from a logits row on the card: the row is read back, the soft cap is applied in
    /// f32 as the reference does, and the host sampler draws from the rows the model can emit.
    fn sample_from_logits(
        &mut self,
        logits_hbm: u64,
        params: &GenParams,
        seq_pos: u32,
    ) -> Result<i32, String> {
        self.sample_logits_row(logits_hbm, 0, params, seq_pos)
    }

    /// [`Self::sample_from_logits`] for row `row` of a logits tensor of several rows.
    fn sample_logits_row(
        &mut self,
        logits_hbm: u64,
        row: usize,
        params: &GenParams,
        seq_pos: u32,
    ) -> Result<i32, String> {
        let vocab = self.dims.vocab_out as usize;
        let bytes = vocab * 2;
        let logits_hbm = logits_hbm + (row as u64) * self.dims.vocab_pad * 2;
        self.dev
            .copy_d2h_pinned_sync(logits_hbm, &mut self.stage_io, bytes)
            .map_err(|e| format!("read logits D2H: {e}"))?;
        let raw = &self.stage_io.as_slice()[..bytes];
        let cap = self
            .spec
            .inkling2
            .as_ref()
            .map_or(0.0, |ik| ik.logit_softcap as f32);
        let mut bf16 = Vec::with_capacity(vocab);
        for &ch in raw.as_chunks::<2>().0 {
            let bits = u16::from_le_bytes(ch);
            if cap > 0.0 {
                let x = crate::fp8::bf16_to_f32(bits);
                bf16.push(crate::fp8::f32_to_bf16_rne((x / cap).tanh() * cap));
            } else {
                bf16.push(bits);
            }
        }
        let sp = SampleParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            min_p: params.min_p,
            seed: params.seed,
            coupled: true,
            draft_trunc: true,
        };
        let dev_sp = sp.to_dev(self.dims.vocab_out);
        Ok(sample_row(&bf16, &dev_sp, 0, seq_pos, true).token as i32)
    }

    /// Writes every tensor of the one-row decode step (as the last launch left it) to `dir`.
    fn dump_step(&mut self, dir: &std::path::Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let cap = self.stage_io.as_slice().len();
        // the activations, and the head of every K / V state and history ring
        let list: Vec<(String, u64, u64)> = self
            .step_act_addr
            .iter()
            .map(|(n, &(a, b))| (n.clone(), a, b))
            .chain(
                self.kv_addr
                    .iter()
                    .map(|(n, &(a, b))| (n.clone(), a, b.min(64 << 20))),
            )
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

    /// Runs one pass of the decode step over `toks` at positions `cursor ..` (the recipe of
    /// `toks.len()` rows must be compiled) and returns each row's arg-max token.
    fn run_rows(&mut self, toks: &[i32], cursor: u64, bucket: u64) -> Result<Vec<i32>, String> {
        let rows = toks.len();
        // reuse: every row of the pass is written, accepted or not
        self.reuse.will_write(cursor + rows as u64);
        let (ids, cur, tokens) = (
            self.step_act_addr["ids"].0,
            self.step_act_addr["cursor"].0,
            self.step_act_addr["tokens"].0,
        );
        let buf = self.stage_io.as_mut_slice();
        for (k, &t) in toks.iter().enumerate() {
            buf[k * 4..k * 4 + 4].copy_from_slice(&t.to_le_bytes());
        }
        let coff = (rows * 4).next_multiple_of(64);
        buf[coff..coff + 4].copy_from_slice(&(cursor as i32).to_le_bytes());
        self.dev
            .copy_h2d_pinned_batch_sync(&self.stage_io, &[(0, rows * 4, ids), (coff, 4, cur)])
            .map_err(|e| format!("write step inputs H2D: {e}"))?;
        let (recipe, prepared) = &self.step_recipes[&(rows as u64, bucket)];
        self.live.launch_begin();
        recipe
            .launch_prepared(&self.dev, prepared)
            .map_err(|e| format!("launch decode step (cursor={cursor}, {rows} rows): {e}"))?;
        self.live.launch_submitted();
        self.sync_compute("decode step")?;
        self.dev
            .copy_d2h_pinned_sync(tokens, &mut self.stage_io, rows * 4)
            .map_err(|e| format!("read step tokens D2H: {e}"))?;
        Ok(self.stage_io.as_slice()[..rows * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect())
    }

    /// Runs one decode step for `tok` at position `cursor` and returns the next token.
    fn run_step(
        &mut self,
        tok: i32,
        cursor: u64,
        bucket: u64,
        params: &GenParams,
    ) -> Result<i32, String> {
        let greedy = self.run_rows(&[tok], cursor, bucket)?[0];
        if params.temperature > 0.0 {
            let logits = self.step_act_addr["logits"].0;
            self.sample_from_logits(logits, params, cursor as u32)
        } else {
            Ok(greedy)
        }
    }

    /// Reads back the raw logits of the first `rows` rows of the last step pass.
    fn read_step_logits(&mut self, rows: usize) -> Result<Vec<u8>, String> {
        let row_bytes = self.dims.vocab_pad as usize * 2;
        let addr = self.step_act_addr["logits"].0;
        let cap = self.stage_io.as_slice().len() / row_bytes * row_bytes;
        let total = rows * row_bytes;
        let mut out = Vec::with_capacity(total);
        while out.len() < total {
            let n = (total - out.len()).min(cap);
            self.dev
                .copy_d2h_pinned_sync(addr + out.len() as u64, &mut self.stage_io, n)
                .map_err(|e| format!("read logits D2H: {e}"))?;
            out.extend_from_slice(&self.stage_io.as_slice()[..n]);
        }
        Ok(out)
    }

    /// Debug check behind `GAUDIUM_IK_VERIFY_CHECK`: runs the tokens of a finished greedy request
    /// again in passes of `rows` rows and compares every row's logits with the one-row step's.
    fn verify_check_run(
        &mut self,
        prompt_len: usize,
        generated: &[i32],
        step_logits: &[Vec<u8>],
    ) -> Result<(), String> {
        let rows = self.verify_check as usize;
        let row_bytes = self.dims.vocab_pad as usize * 2;
        let (mut passes, mut bad_tokens, mut bad_rows, mut bad_values) = (0u64, 0u64, 0u64, 0u64);
        let mut pass_ms = 0.0f64;
        let mut b = 0usize;
        // step i reads generated[i] at position prompt_len + i and gives generated[i + 1]
        while b + rows < generated.len() && b + rows <= step_logits.len() {
            let cursor = (prompt_len + b) as u64;
            let bucket = step_bucket((cursor + rows as u64).div_ceil(IK_CHUNK), self.max_chunks);
            self.ensure_step(rows as u64, bucket)?;
            let t0 = Instant::now();
            let got = self.run_rows(&generated[b..b + rows], cursor, bucket)?;
            pass_ms += t0.elapsed().as_secs_f64() * 1000.0;
            let logits = self.read_step_logits(rows)?;
            for k in 0..rows {
                if got[k] != generated[b + k + 1] {
                    bad_tokens += 1;
                }
                let a = &logits[k * row_bytes..(k + 1) * row_bytes];
                let diff = a
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(step_logits[b + k].as_chunks::<2>().0)
                    .filter(|(x, y)| x != y)
                    .count() as u64;
                if diff > 0 {
                    bad_rows += 1;
                    bad_values += diff;
                }
            }
            passes += 1;
            b += rows;
        }
        eprintln!(
            "[gaudium] VERIFY-CHECK rows={rows}: {passes} passes ({:.2} ms per pass), {} rows; tokens differing from the one-row stream: {bad_tokens}; rows with any differing logit: {bad_rows} ({bad_values} values)",
            pass_ms / (passes.max(1) as f64),
            passes * rows as u64
        );
        Ok(())
    }

    /// Speculative decode after the first token: each launch verifies the last token and its
    /// drafts in one pass of `spec_rows` rows and emits the tokens the pass itself gives up to and
    /// including the first one a draft missed. Every emitted token is the pass's own arg-max (or
    /// its own sample at that position), and a pass of several rows gives each row the logits a
    /// one-row step gives it, so the stream is the `K = 0` stream.
    #[allow(clippy::too_many_arguments)]
    fn decode_spec(
        &mut self,
        prompt_len: usize,
        params: &GenParams,
        budget: &mut SequenceBudget,
        is_eos: &impl Fn(i32) -> bool,
        is_cancelled: &impl Fn() -> bool,
        on_token: &mut impl FnMut(i32),
        generated: &mut Vec<i32>,
    ) -> Result<SpecOutcome, String> {
        let rows = self.spec_rows as usize;
        let k = rows - 1;
        let req = self.requests;
        let oracle = match self.draft.as_ref().map(|s| &s.method) {
            Some(SpecMethod::Oracle { requests }) => requests.get(req).cloned().unwrap_or_default(),
            _ => return Err("Inkling 2: no drafter for this request".into()),
        };
        let logits = self.step_act_addr["logits"].0;
        let mut out = SpecOutcome::default();
        let mut bucket = 0u64;
        let t0 = Instant::now();
        let mut compile_ms = 0.0f64;
        while budget.remaining() > 0 {
            if is_cancelled() {
                out.stopped_by_cancel = true;
                break;
            }
            // the last emitted token is generated[gi]; it sits at position prompt_len + gi
            let gi = generated.len() - 1;
            let cursor = (prompt_len + gi) as u64;
            let room = (self.max_ctx - cursor) as usize;
            if room < rows {
                // the pass would write past the context: finish with one-row steps
                let b1 = step_bucket((cursor + 1).div_ceil(IK_CHUNK), self.max_chunks);
                self.ensure_step(1, b1)?;
                let tok = self.run_step(generated[gi], cursor, b1, params)?;
                out.launches += 1;
                budget.record_emitted(1);
                generated.push(tok);
                self.live.step(1, 0, 0);
                on_token(tok);
                if params.stop_on_eos && is_eos(tok) {
                    out.stopped_by_eos = true;
                    break;
                }
                continue;
            }
            // drafts of the host oracle: the known continuation, with the marked ones made wrong
            let mut toks = vec![generated[gi]];
            for j in 0..k {
                let Some(&t) = oracle.stream.get(gi + 1 + j) else { break };
                let wrong = oracle.corrupt.contains(&(gi + 1 + j));
                toks.push(if wrong { (t + 1) % self.dims.vocab_out as i32 } else { t });
            }
            let n_drafts = toks.len() - 1;
            toks.resize(rows, 0);
            let need = step_bucket((cursor + rows as u64).div_ceil(IK_CHUNK), self.max_chunks);
            if need != bucket {
                let tc = Instant::now();
                self.ensure_step(rows as u64, need)?;
                compile_ms += tc.elapsed().as_secs_f64() * 1000.0;
                bucket = need;
            }
            let mut got = self.run_rows(&toks, cursor, bucket)?;
            if params.temperature > 0.0 {
                // a sampled request: each row's token is its position's own sample
                for (r, g) in got.iter_mut().enumerate().take(n_drafts + 1) {
                    *g = self.sample_logits_row(logits, r, params, (cursor + r as u64) as u32)?;
                }
            }
            let accepted = chain_accept(&toks[..=n_drafts], &got[..=n_drafts]);
            out.launches += 1;
            out.drafted += n_drafts as u64;
            out.accepted += accepted as u64;
            let mut emitted = 0u64;
            for &tok in &got[..=accepted] {
                if budget.remaining() == 0 {
                    break;
                }
                budget.record_emitted(1);
                generated.push(tok);
                on_token(tok);
                emitted += 1;
                if params.stop_on_eos && is_eos(tok) {
                    out.stopped_by_eos = true;
                    break;
                }
            }
            self.live.step(emitted, n_drafts as u64, accepted as u64);
            if out.stopped_by_eos {
                break;
            }
        }
        out.decode_ms = (t0.elapsed().as_secs_f64() * 1000.0 - compile_ms).max(0.0);
        Ok(out)
    }

    /// Runs prefill + autoregressive decode, calling `on_token` per generated token. `cancel` is
    /// checked only between synced launches, so no Synapse launch is ever left queued.
    pub fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        cancel: Option<&AtomicBool>,
        mut on_token: impl FnMut(i32),
    ) -> Result<GenReport, String> {
        // vision: the request's images (taken even when the request fails) against its
        // placeholder rows
        if let Some(v) = self.vision.as_mut() {
            v.begin_request(prompt_ids, self.step_prefill)?;
        }
        if prompt_ids.is_empty() {
            return Err("prompt must contain at least 1 token".into());
        }
        if let Some(&bad) = prompt_ids
            .iter()
            .find(|&&id| id < 0 || (id as u64) >= self.dims.vocab_in)
        {
            return Err(format!(
                "prompt token id {bad} is outside the vocabulary (0..{})",
                self.dims.vocab_in
            ));
        }
        let mut budget =
            SequenceBudget::admit(prompt_ids.len() as u64, Some(params.max_new), self.max_ctx)
                .map_err(|e| format!("SequenceBudget rejected prompt/max_new: {e:?}"))?;

        let mut eos_ids = self.eos_ids.clone();
        for &id in &params.extra_eos_ids {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
        let is_eos = |tok: i32| eos_ids.contains(&tok);
        let is_cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));

        let method_label = if params.temperature > 0.0 {
            format!("K=0 BF16 T={:.2}", params.temperature)
        } else {
            "K=0 BF16 greedy".to_string()
        };
        self.live
            .begin_request(prompt_ids.len() as u64, &method_label, "");
        self.live.prefill_progress(0);
        let res = self.run_generate_inner(
            prompt_ids,
            params,
            &mut budget,
            &is_eos,
            &is_cancelled,
            &mut on_token,
            &method_label,
        );
        self.live.end_request();
        match &res {
            // reuse: the rows of the tokens fed during the decode are held too
            Ok(rep) => self.reuse_request_done(prompt_ids.len(), &rep.generated_tokens),
            // never leave a launch queued behind an error return
            Err(_) => {
                let _ = self.dev.sync_all();
            }
        }
        res
    }

    /// Prefills `prompt_ids` in tiles and returns the first generated token with the time spent
    /// on first-use recipe compiles.
    fn prefill_tiles(
        &mut self,
        prompt_ids: &[i32],
        start: usize,
        params: &GenParams,
        is_cancelled: &impl Fn() -> bool,
    ) -> Result<Option<(i32, f64)>, String> {
        let n_apps = self.dims.n_apps;
        let hid = self.dims.hid;
        let kinds = self.dims.kinds();
        // reuse: the rows before `start` are taken over from the card
        let mut pos = start;
        let mut last_tile = 0u64;
        let mut last_valid = 0u64;
        let mut recipe_ms = 0.0f64;
        // vision: the tower runs first, its features land in the feature buffer
        self.ikv_run_tower()?;
        while pos < prompt_ids.len() {
            if is_cancelled() {
                return Ok(None);
            }
            // `tv` prompt rows run as a tile of `t` rows (a ladder of sizes: the rows past the prompt
            // are computed, written past the cursor and never read; the next pass writes them again)
            let tv = ik_tile_take((prompt_ids.len() - pos) as u64, self.prefill_tile);
            let t = ik_tile_rows(tv, self.prefill_tile, self.max_ctx - pos as u64);
            let end_pos = pos as u64 + t;
            let nch = step_bucket(end_pos.div_ceil(IK_CHUNK), self.max_chunks);
            self.reuse.will_write(end_pos);
            last_tile = t;
            last_valid = tv;

            let t_rec = Instant::now();
            self.ensure_embed(t)?;
            for &kind in &kinds {
                self.ensure_app(kind, t, nch)?;
            }
            recipe_ms += t_rec.elapsed().as_secs_f64() * 1000.0;

            let tile_bytes = (t as usize) * 4;
            self.stage_io.as_mut_slice()[..tile_bytes].fill(0);
            for (k, &id) in prompt_ids[pos..pos + tv as usize].iter().enumerate() {
                self.stage_io.as_mut_slice()[k * 4..k * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
            self.dev
                .copy_h2d_pinned_sync(&self.stage_io, tile_bytes, self.pf_ids)
                .map_err(|e| format!("upload prefill ids: {e}"))?;
            self.write_i32_scalar(self.pf_cursor, pos as i32)?;

            // embedding of the tile's rows into the buffer of source 0 (vision: image rows take
            // their features)
            if !self.ikv_embed_tile(prompt_ids, pos, t)? {
                let mut pool = self.pf_scratch.clone();
                for name in ["embed", "w_embed"] {
                    pool.insert(name.to_string(), self.weights_addr[name]);
                }
                pool.insert("ids".into(), self.pf_ids);
                pool.insert("x0".into(), self.src_buf(0));
                let r = &self.embed_recipes[&t];
                let bind = select_exact_bindings(r, &pool)?;
                r.launch_checked(&self.dev, &bind)
                    .map_err(|e| format!("launch embed T={t}: {e}"))?;
            }
            for a in 0..n_apps {
                self.launch_prefill_app(a, t, nch)?;
            }
            self.sync_compute("prefill tile")?;
            pos += tv as usize;
            self.reuse.rows_in(pos);
            self.live.prefill_progress(pos as u64);
        }

        // final mix and norm over the last tile, then the readout of its last row
        let t_rec = Instant::now();
        self.ensure_final(last_tile)?;
        recipe_ms += t_rec.elapsed().as_secs_f64() * 1000.0;
        {
            let fin: Vec<usize> = self.dims.fin_states().to_vec();
            let mut pool = self.pf_scratch.clone();
            pool.insert("x_in".into(), self.src_buf(n_apps));
            for i in 0..IK_MAX_STATES as usize {
                let addr = match fin.get(i) {
                    Some(&s) => self.src_buf(s),
                    None => self.pf_dummy[i],
                };
                pool.insert(format!("S{i}"), addr);
            }
            for name in ["ctl_fin", "q_fin", "w_final"] {
                pool.insert(name.to_string(), self.weights_addr[name]);
            }
            pool.insert("h_final".into(), self.pf_hfin);
            let r = &self.final_recipes[&last_tile];
            let bind = select_exact_bindings(r, &pool)?;
            r.launch_checked(&self.dev, &bind)
                .map_err(|e| format!("launch final mix T={last_tile}: {e}"))?;
        }
        {
            let mut pool = self.pf_scratch.clone();
            pool.insert("h_in".into(), self.pf_hfin + ik_tile_last_row(last_valid) * hid * 2);
            pool.insert("W_head".into(), self.weights_addr["W_head"]);
            let bind = select_exact_bindings(&self.head, &pool)?;
            self.head
                .launch_checked(&self.dev, &bind)
                .map_err(|e| format!("launch head: {e}"))?;
        }
        self.sync_compute("prefill readout")?;
        let first_tok = if params.temperature > 0.0 {
            self.sample_from_logits(
                self.pf_scratch["logits"],
                params,
                (prompt_ids.len() - 1) as u32,
            )?
        } else {
            self.read_i32_scalar(self.pf_scratch["tokens"])?
        };
        Ok(Some((first_tok, recipe_ms)))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generate_inner(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        budget: &mut SequenceBudget,
        is_eos: &impl Fn(i32) -> bool,
        is_cancelled: &impl Fn() -> bool,
        on_token: &mut impl FnMut(i32),
        method: &str,
    ) -> Result<GenReport, String> {
        let t_prefill = Instant::now();
        let mut generated = Vec::with_capacity(budget.max_new() as usize);
        let mut stopped_by_eos = false;
        let mut stopped_by_cancel = false;
        let cancelled_report = |ttft_ms: f64| GenReport {
            prompt_tokens: prompt_ids.len(),
            ttft_ms,
            stopped_by_cancel: true,
            ..GenReport::default()
        };

        // reuse: where the prefill starts (0: nothing is taken over)
        let (start, prefix_common) = self.reuse_begin(prompt_ids, params.cache_prompt)?;

        // ---- PREFILL ----
        let (first_tok, recipe_ms) = if self.step_prefill {
            // debug: the prompt one row at a time through the decode step
            let mut tok = 0i32;
            for (i, &id) in prompt_ids.iter().enumerate() {
                if is_cancelled() {
                    return Ok(cancelled_report(t_prefill.elapsed().as_secs_f64() * 1000.0));
                }
                let bucket = step_bucket((i as u64 + 1).div_ceil(IK_CHUNK), self.max_chunks);
                self.ensure_step(1, bucket)?;
                tok = self.run_step(id, i as u64, bucket, params)?;
                self.live.prefill_progress(i as u64 + 1);
            }
            (tok, 0.0)
        } else {
            match self.prefill_tiles(prompt_ids, start, params, is_cancelled)? {
                Some(v) => v,
                None => return Ok(cancelled_report(t_prefill.elapsed().as_secs_f64() * 1000.0)),
            }
        };
        if let Some(dir) = self.dump_dir.clone() {
            self.dump_step(&dir)?;
        }

        let ttft_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let prefilled = prompt_ids.len() - start;
        let prefill_tok_s = (prefilled as f64) / ((ttft_ms - recipe_ms).max(1e-6) / 1000.0);
        if self.verbose {
            eprintln!(
                "[gaudium] inkling2 prefill: {} tok ({start} taken over), tiles of <= {} rows, {:.1} ms to first token of which {:.1} ms first-use recipe compile/load (excluded from the prefill rate: {:.1} tok/s)",
                prefilled,
                self.prefill_tile,
                ttft_ms,
                recipe_ms,
                prefill_tok_s
            );
        }

        if budget.remaining() > 0 && !is_cancelled() {
            budget.record_emitted(1);
            generated.push(first_tok);
            self.live.first_token(Some(ttft_ms), 1);
            on_token(first_tok);
            if params.stop_on_eos && is_eos(first_tok) {
                stopped_by_eos = true;
            }
        } else if is_cancelled() {
            stopped_by_cancel = true;
        }
        // reuse: the prompt is in (after the first token left, so the copy is not in its time)
        self.reuse_prompt_in()?;

        // ---- DECODE, speculative: an engine loaded with a drafter ----
        if self.draft.is_some()
            && !stopped_by_eos
            && !stopped_by_cancel
            && !generated.is_empty()
            && budget.remaining() > 0
        {
            let out = self.decode_spec(
                prompt_ids.len(),
                params,
                budget,
                is_eos,
                is_cancelled,
                on_token,
                &mut generated,
            )?;
            self.requests += 1;
            let decode_tokens = generated.len().saturating_sub(1);
            let k = self.spec_rows - 1;
            return Ok(GenReport {
                prompt_tokens: prompt_ids.len(),
                reused_tokens: start as u64,
                prefix_common: prefix_common as u64,
                generated_tokens: generated,
                ttft_ms,
                prefill_tok_s,
                decode_ms: out.decode_ms,
                decode_tok_s: decode_tokens as f64 / (out.decode_ms / 1000.0).max(1e-9),
                ms_per_step: out.decode_ms / (out.launches.max(1) as f64),
                stopped_by_eos: out.stopped_by_eos,
                stopped_by_cancel: out.stopped_by_cancel,
                launches: out.launches,
                tokens_per_launch: decode_tokens as f64 / (out.launches.max(1) as f64),
                method: format!(
                    "{} K={k} BF16 {}",
                    self.draft.as_ref().map_or("", |s| s.method_name()),
                    if params.temperature > 0.0 { "sampled" } else { "greedy" }
                ),
                drafted: out.drafted,
                draft_accepted: out.accepted,
                ..GenReport::default()
            });
        }
        self.requests += 1;

        // ---- DECODE: one monolithic launch per token ----
        // a bucket compiled on first use is not decode time: build the first one before the clock
        let mut bucket = step_bucket(
            (prompt_ids.len() as u64 + 1).div_ceil(IK_CHUNK),
            self.max_chunks,
        );
        if !stopped_by_eos && !stopped_by_cancel && budget.remaining() > 0 {
            self.ensure_step(1, bucket)?;
        }
        let keep_logits = self.verify_check > 0 && params.temperature <= 0.0;
        let mut step_logits: Vec<Vec<u8>> = Vec::new();
        let t_decode = Instant::now();
        let mut compile_ms = 0.0f64;
        let mut cur_tok = first_tok;
        let mut decode_steps = 0usize;
        while !stopped_by_eos && !stopped_by_cancel && budget.remaining() > 0 {
            if is_cancelled() {
                stopped_by_cancel = true;
                break;
            }
            let cursor_val = (prompt_ids.len() + decode_steps) as u64;
            let need = step_bucket((cursor_val + 1).div_ceil(IK_CHUNK), self.max_chunks);
            if need != bucket {
                let t0 = Instant::now();
                self.ensure_step(1, need)?;
                compile_ms += t0.elapsed().as_secs_f64() * 1000.0;
                bucket = need;
            }
            let next_tok = self.run_step(cur_tok, cursor_val, bucket, params)?;
            if keep_logits {
                step_logits.push(self.read_step_logits(1)?);
            }
            decode_steps += 1;
            budget.record_emitted(1);
            generated.push(next_tok);
            self.live.step(1, 0, 0);
            on_token(next_tok);
            cur_tok = next_tok;
            if params.stop_on_eos && is_eos(next_tok) {
                stopped_by_eos = true;
            }
        }

        let decode_ms = (t_decode.elapsed().as_secs_f64() * 1000.0 - compile_ms).max(0.0);
        if keep_logits && !stopped_by_cancel {
            self.verify_check_run(prompt_ids.len(), &generated, &step_logits)?;
        }
        let (decode_tok_s, ms_per_step) = if decode_steps > 0 {
            (
                (decode_steps as f64) / (decode_ms / 1000.0).max(1e-9),
                decode_ms / (decode_steps as f64),
            )
        } else {
            (0.0, 0.0)
        };
        Ok(GenReport {
            prompt_tokens: prompt_ids.len(),
            reused_tokens: start as u64,
            prefix_common: prefix_common as u64,
            generated_tokens: generated,
            ttft_ms,
            prefill_tok_s,
            decode_ms,
            decode_tok_s,
            ms_per_step,
            stopped_by_eos,
            stopped_by_cancel,
            launches: decode_steps as u64,
            tokens_per_launch: if decode_steps > 0 { 1.0 } else { 0.0 },
            method: method.to_string(),
            ..GenReport::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini() -> ModelSpec {
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
        ModelSpec::from_config_json(&json).unwrap()
    }

    #[test]
    fn dims_of_the_mini() {
        let d = IkDims::from_spec(&mini(), 1).unwrap();
        assert_eq!((d.hid, d.nh, d.nkv, d.gq), (2304, 24, 8, 3));
        assert_eq!((d.ne, d.ek, d.ef), (128, 8, 1152));
        assert_eq!((d.apps.len(), d.n_layers), (48, 24));
        assert_eq!((d.ring, d.hist), (2048, 128));
        assert_eq!((d.qw, d.bw, d.boff), (6528, 1152, 1087));
        assert_eq!((d.kxw(true), d.kxw(false)), (1024, 1152));
        assert_eq!((d.vocab_in, d.vocab_out, d.vocab_pad), (201024, 200058, 200064));
        assert_eq!(d.kinds().len(), 4);
    }

    #[test]
    fn step_graph_has_one_state_set_per_application() {
        let d = IkDims::from_spec(&mini(), 1).unwrap();
        let gb = build_step_graph(&d, 8192, 2, 1, &[]);
        let specs = gb.persistent_tensor_specs();
        let dims_of = |n: &str| {
            specs
                .iter()
                .find(|(name, ..)| name == n)
                .unwrap_or_else(|| panic!("tensor {n}"))
                .2
                .clone()
        };
        // a sliding application keeps a ring, a global one a chunked cache of its bucket
        assert_eq!(dims_of("A0_kring"), vec![128, 2048, 8]);
        assert_eq!(dims_of("A5_kcache"), vec![128, 4096, 16]);
        // the three passes of the looped global layer 11 have three caches and share its weights
        for a in [11, 23, 35] {
            assert_eq!(dims_of(&format!("A{a}_vcache")), vec![128, 4096, 16]);
        }
        assert_eq!(dims_of("L11_W_qx"), vec![2304, 6528]);
        assert_eq!(specs.iter().filter(|(n, ..)| n.ends_with("_W_qx")).count(), 24);
        // messages and their gates only in the later passes
        assert!(specs.iter().any(|(n, ..)| n == "A18_msg"));
        assert!(!specs.iter().any(|(n, ..)| n == "A6_msg"));
        assert_eq!(dims_of("L6_W_gf_q"), vec![64, 4608, 1]);
        assert_eq!(dims_of("A18_go_mlp"), vec![2304, 1, 1]);
        // the wide expert matrices and the readout
        assert_eq!(dims_of("L0_W_gate"), vec![2304, 147456]);
        assert_eq!(dims_of("L0_W_down"), vec![147456, 2304]);
        assert_eq!(dims_of("logits"), vec![200064, 1]);
        assert_eq!(dims_of("A5_bias3"), vec![9216, 3, 1]);
    }

    #[test]
    fn plan_of_the_mini_matches_the_shapes() {
        let spec = mini();
        let [w, kv, rings, acts] = hbm_plan_bytes(&spec, 262144, 64).unwrap();
        // 24 layers of experts in BF16 are 49 GB of the weights
        assert!(w > 50_000_000_000 && w < 56_000_000_000, "weights {w}");
        // 8 global applications x K and V x 8 heads x 128 channels x 2 bytes per position
        assert_eq!(kv, 8 * 2 * 8 * 128 * 2 * 262144);
        assert!(rings < 1_000_000_000, "rings {rings}");
        assert!(acts > 0);
        // every name the plan sorted is one the roles know
        let d = IkDims::from_spec(&spec, 64).unwrap();
        let p = IkPlan::build(&d, 262144, 64, 64, 1);
        assert!(p.weights.keys().all(|n| is_weight_name(n)));
        assert!(p.kv.keys().all(|n| is_kv_name(n) && n.ends_with("cache")));
        assert!(p.pf_scratch.keys().all(|n| !is_weight_name(n) && !is_kv_name(n) && !is_hist_name(n)));
        assert!(p.pf_bufs <= 8);
    }

    #[test]
    fn names_split_into_roles() {
        assert_eq!(split_name("L12_W_qx"), Some(('L', Some(12), "W_qx")));
        assert_eq!(split_name("A_ctl"), Some(('A', None, "ctl")));
        assert!(is_weight_name("A7_ctl") && is_weight_name("tau") && is_weight_name("L3_rb"));
        assert!(is_kv_name("A5_kcache") && is_kv_name("A_vring") && !is_kv_name("A5_kraw"));
        assert!(is_hist_name("A5_kraw") && is_hist_name("A_mraw"));
        assert!(is_prefill_special("S3") && is_prefill_special("P0") && is_prefill_special("ph"));
        assert!(!is_prefill_special("A_scores"));
    }
}
