//! Zero-dependency OpenAI-compatible HTTP/SSE + Gaudium Web UI server.
//!
//! Serves:
//! - Gaudium Web UI static assets (`GET /`, `/sw.js`, `/css/*`, `/js/*`, `/img/*`, `/vendor/*`)
//! - OpenAI-compatible chat & model endpoints (`/v1/models`, `/v1/chat/completions`, `/props`,
//!   `/health`, `/tokenize`, `/detokenize`, `/slots`, `/models/load`, `/models/unload`)
//! - Gaudium UI telemetry & card-lifecycle endpoints (`/gaudium/ui`, `/gaudium/engine`,
//!   `/gaudium/live`, `/gaudium/events`, `/gaudium/count_tokens`, `/v1/chat/completions/control`)
//! - A2UI interfaces as a per-request option (`--a2ui`, the request field `gaudium_a2ui`,
//!   `/gaudium/a2ui`, instructions hot-reloaded from `--a2ui-prompt <file>`; see `a2ui.rs`)
//! - The MCP proxy for the Web UI's MCP servers (`--mcp-proxy`, `/cors-proxy`, `/gaudium/proxy`;
//!   see `proxy.rs`)
//!
//! HTTP and HTTPS share the port (`--tls-cert`, `--tls-key`; see `conn.rs`): the first byte of a
//! connection says which of the two the client speaks.
//!
//! Single-card safety & verified core invariants:
//! - Uses `core::lifecycle::can_switch_resident_model` before switching resident models or parking.
//! - Uses `core::lifecycle::RequestGuard` and `core::budget::SequenceBudget` per request.
//! - Supports `--lazy` startup and `--idle-park` automatic card release so `/dev/accel/accel0` is
//!   only held while actively serving or within the idle window.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::core::budget::SequenceBudget;
use crate::core::effort::EffortLevels;
use crate::core::lifecycle::{can_switch_resident_model, RequestGuard};
use crate::core::tool_calls::{finish_reason_name, generation_end, stop_reason_name, GenEnd};
use crate::core::vision::{drafter_runs_with_vision, planned_image_tokens, MediaLimits};
use crate::engine::qwen3::{Qwen3Config, Qwen3GenParams, KV_CHUNK};
use crate::engine::{load_engine, GenEngine, LaneDone, LaneJob, LaneOut};
use crate::json::{self, JsonValue};
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::model_text::ModelText;
use crate::tokenizer::{StreamDecoder, Tokenizer};
use crate::verified::{BackendState, ReqEvent};
use crate::vision::media::{MediaCounts, RequestMedia, VisionServe};
use crate::vision::VisionSpec;

mod a2ui;
mod anthropic;
mod chat;
mod conn;
mod jinja;
mod lanes;
mod outparse;
mod proxy;
mod readout;
mod stream_out;

use chat::{ChatPrompt, ChatTemplate};
use conn::Conn;
pub use proxy::ProxyOptions;
use outparse::{Delta, OutputFormat, OutputParser};
use readout::{DrafterFacts, RunFacts, Timings};
use stream_out::StreamOut;

/// Largest request body. Images and video frames travel inside the request as base64.
const MAX_REQUEST_BODY_BYTES: usize = 256 * 1024 * 1024;

/// Server defaults of the vision tower (`--vision-max-patches`, `--vision-max-tokens`): images
/// of up to 4096 patches (1024 x 1024 pixels; larger ones are resized down to it) and 16384
/// image tokens per request.
pub const DEFAULT_VISION_MAX_PATCHES: u64 = 4096;
pub const DEFAULT_VISION_MAX_TOKENS: u64 = 16_384;

#[derive(Debug, Clone)]
pub struct ModelSpecEntry {
    pub id: String,
    pub dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub models: Vec<ModelSpecEntry>,
    pub kernels_dir: PathBuf,
    pub ui_dir: Option<PathBuf>,
    pub max_ctx: u64,
    pub cache_dir: Option<PathBuf>,
    pub no_cache: bool,
    pub lazy: bool,
    pub idle_park_secs: u64,
    /// Card-queue settings file (`queue.conf`), re-read on every turn decision; missing = defaults.
    pub queue_config: Option<PathBuf>,
    pub api_key: Option<String>,
    /// Offer A2UI: a request that declares renderer capabilities (`gaudium_a2ui`) gets the
    /// instructions that teach the model to write interface blocks.
    pub a2ui: bool,
    /// Read the A2UI instructions from this file instead of the built-in text, again whenever it
    /// changes on disk (a file that fails its checks is not served; see `a2ui.rs`).
    pub a2ui_prompt: Option<PathBuf>,
    pub verbose: bool,
    /// Per-model options keyed by served model id.
    pub per_model: Vec<(String, ModelRunOptions)>,
    /// Draft widths for models with a drafter (one value = fixed, several = adaptive).
    pub draft_k: Vec<u64>,
    /// Load the vision tower of models that have one (image and video input).
    pub vision: bool,
    /// Largest image the vision tower is planned for, in patches (its device buffers grow with
    /// the square of this); larger images are resized down to it.
    pub vision_max_patches: u64,
    /// Image tokens one request may hold (the tower's embedding buffer).
    pub vision_max_tokens: u64,
    /// Prefix reuse across requests: positions of the previous prompt whose recurrent state an
    /// engine keeps on the card (0 = off; a load keeps as many as fit beside the context).
    pub prefix_slots: u64,
    /// The MCP proxy (`--mcp-proxy` and its options).
    pub mcp_proxy: ProxyOptions,
    /// PEM certificate chain of the HTTPS listener (`--tls-cert`).
    pub tls_cert: Option<PathBuf>,
    /// PEM private key of the HTTPS listener (`--tls-key`). Only its path is held here.
    pub tls_key: Option<PathBuf>,
    /// Budget of the host-RAM tier of prefix reuse in bytes (`--cache-ram`): `None` = a share
    /// of the host memory available at start, `Some(0)` = off.
    pub cache_ram_bytes: Option<u64>,
    /// Requests a model runs at once (`--parallel`): lanes of their own, decoded together.
    /// Models whose engine or drafter has no batched steps are served with one lane.
    pub parallel: u64,
}

/// Token ids an own-MTP drafter reads out per draft (the old engine's measured default: the same
/// acceptance as the full vocabulary at about 1 ms less per launch).
const OWN_MTP_DRAFT_VOCAB: u64 = 98_304;

/// Optional per-model run configuration (speculative drafter, FP8 tables).
#[derive(Debug, Clone, Default)]
pub struct ModelRunOptions {
    /// Drafter checkpoint: enables speculative decoding for this model.
    pub draft_model: Option<PathBuf>,
    /// Weight precision instead of the automatic one (`fp8` quantises a BF16 checkpoint at
    /// load, with its activation-scale table from `fp8_act`).
    pub precision: Option<crate::engine::fp8_graph::WeightPrecision>,
    /// Speculative decoding with the model's own MTP head (no drafter checkpoint).
    pub own_mtp: bool,
    /// Mixture-of-experts model: the routed expert form instead of the dense-masked one.
    pub moe_routed: bool,
    /// Context size for this model instead of the server-wide one (still clamped to the
    /// model's own limit).
    pub max_ctx: Option<u64>,
    /// FP8 activation-scale table for an FP8 checkpoint.
    pub fp8_act: Option<PathBuf>,
    /// Directory caching the quantised FP8 matrices.
    pub fp8_cache: Option<PathBuf>,
    /// The vision tower of this model instead of the server-wide setting: its largest image in
    /// patches, or 0 for no tower (text only).
    pub vision_patches: Option<u64>,
    /// The reasoning-effort levels instead of the ones read from the model's chat template.
    pub effort_levels: Option<EffortLevels>,
    /// [kv8] The K/V cache dtype of this model (`None`: BF16).
    pub kv_dtype: Option<crate::verified::KvDtype>,
    /// [kv8] The K/V calibration of an FP8 K/V load (`None`: the model's default table).
    pub kv_scales: Option<PathBuf>,
}

struct RegisteredModel {
    id: String,
    dir: PathBuf,
    spec: ModelSpec,
    tokenizer: Arc<Tokenizer>,
    extra_eos_ids: Vec<i32>,
    max_ctx: u64,
    /// Gemma 4 only: the model's template closes an empty thought channel after the generation
    /// prompt of a non-thinking request.
    gemma_closes_thought: bool,
    /// The model's own chat template; `None`: the built-in text templates (no tool calling).
    chat: Option<ChatTemplate>,
    /// How the family writes its reasoning and its tool calls.
    format: OutputFormat,
    /// Sampler settings of a request that names none: the model's own `generation_config.json`.
    gen_defaults: SamplerDefaults,
    /// Image and video input: what the model's vision tower is loaded for. `None`: text only.
    vision: Option<VisionServe>,
    /// The model's own context limit (`ModelText::max_positions`), read once at registration.
    n_ctx_train: Option<u64>,
    /// Lanes the model is served with (`--parallel`; 1 = one request at a time).
    lanes: usize,
}

/// Sampler settings a request falls back to (temperature, top-k, top-p).
#[derive(Debug, Clone, Copy, PartialEq)]
struct SamplerDefaults {
    temperature: f32,
    top_k: u64,
    top_p: f32,
}

impl SamplerDefaults {
    const FALLBACK: Self = Self {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.8,
    };

    /// Reads `generation_config.json` text; a missing or unusable key keeps the fallback value.
    fn from_generation_config(text: &str) -> Self {
        let mut d = Self::FALLBACK;
        if let Ok(v) = crate::json::parse_json(text) {
            if let Some(t) = v.get("temperature").and_then(|x| x.as_f64()) {
                if t.is_finite() && t >= 0.0 {
                    d.temperature = t as f32;
                }
            }
            if let Some(k) = v.get("top_k").and_then(|x| x.as_u64()) {
                d.top_k = k;
            }
            if let Some(p) = v.get("top_p").and_then(|x| x.as_f64()) {
                if p > 0.0 && p <= 1.0 {
                    d.top_p = p as f32;
                }
            }
        }
        d
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Lifecycle {
    Parked,
    Loading(usize),
    Ready(usize),
    Stopping,
    Failed(String),
}

impl Lifecycle {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Parked => "parked",
            Self::Loading(_) => "loading",
            Self::Ready(_) => "ready",
            Self::Stopping => "stopping",
            Self::Failed(_) => "failed",
        }
    }

    fn as_backend_state(&self) -> BackendState {
        match self {
            Self::Parked => BackendState::BeParked,
            Self::Loading(_) => BackendState::BeLoading,
            Self::Ready(_) => BackendState::BeReady,
            Self::Stopping => BackendState::BeStopping,
            Self::Failed(_) => BackendState::BeFailed,
        }
    }
}

struct EngineSlot {
    engine: Option<Box<dyn GenEngine>>,
    resident_idx: Option<usize>,
    lifecycle: Lifecycle,
    detail: String,
    since: Instant,
    last_activity: Instant,
    admin_parked: bool,
    hold: bool,
    active_requests: u64,
    /// Chat requests that arrived and are waiting for the card (a model load or another
    /// request), by model index.
    waiting: Vec<u64>,
    switches: u64,
    last_switch_s: Option<f64>,
    last_warmup_s: Option<f64>,
    last_live_json: Option<String>,
    /// The newest `request` event (for `/gaudium/live`, which a polling client reads).
    last_request_json: Option<String>,
    hbm_bytes: Option<[u64; 5]>,
    /// How the resident model runs, read from its loaded engine.
    facts: Option<RunFacts>,
    /// Prefix reuse of the resident engine as of its last request (`None`: no engine, or one
    /// without it).
    prefix: Option<crate::engine::qwen3::prefix_cache::PrefixStats>,
    /// The resident engine's host-RAM tier as of its last request (`None`: no engine, or one
    /// without it).
    host_tier: Option<crate::engine::qwen3::host_tier::TierFacts>,
    /// A chat request ended and the engine has not had its idle time since
    /// (`AppState::idle_work`).
    idle_due: bool,
}

struct EventHubState {
    seq: u64,
    ring: VecDeque<(u64, String)>,
    closed: bool,
}

struct EventHub {
    mu: Mutex<EventHubState>,
    cv: Condvar,
}

impl EventHub {
    fn new() -> Self {
        Self {
            mu: Mutex::new(EventHubState {
                seq: 0,
                ring: VecDeque::new(),
                closed: false,
            }),
            cv: Condvar::new(),
        }
    }

    fn publish(&self, json: String) -> u64 {
        let mut g = self.mu.lock().unwrap();
        g.seq += 1;
        let seq = g.seq;
        g.ring.push_back((seq, json));
        while g.ring.len() > 256 {
            g.ring.pop_front();
        }
        self.cv.notify_all();
        seq
    }

    fn since(&self, after: u64, wait: Duration) -> (Vec<(u64, String)>, bool) {
        let g = self.mu.lock().unwrap();
        let (g, _) = self
            .cv
            .wait_timeout_while(g, wait, |st| st.seq <= after && !st.closed)
            .unwrap();
        let evs = g
            .ring
            .iter()
            .filter(|(seq, _)| *seq > after)
            .cloned()
            .collect();
        (evs, g.closed)
    }

    fn last_seq(&self) -> u64 {
        self.mu.lock().unwrap().seq
    }

    fn close(&self) {
        self.mu.lock().unwrap().closed = true;
        self.cv.notify_all();
    }
}

struct AppState {
    cfg: ServerConfig,
    models: Vec<RegisteredModel>,
    slot: Mutex<EngineSlot>,
    /// Serializes device-level load/generate/unload operations without blocking status reads.
    card_op: Mutex<()>,
    cancels: Mutex<HashMap<String, Arc<AtomicBool>>>,
    next_req_id: AtomicU64,
    hub: EventHub,
    /// Data types of a model's step graph by `(model, fp8 weights, routed experts, FP8 K/V)`.
    dtypes:
        Mutex<HashMap<(usize, bool, bool, bool), Option<crate::engine::qwen3_traffic::StepDtypes>>>,
    /// Chat requests take turns on the card through this queue (model load + generation in one turn).
    turns: TurnQueue,
    /// The MCP proxy (off unless the server was started with `--mcp-proxy`).
    proxy: proxy::McpProxy,
    /// The certificate of the HTTPS listener; `None`: plain HTTP only.
    tls: Option<crate::tls::TlsServer>,
    /// The host-RAM tier of prefix reuse (`--cache-ram`): entries outlive the engine that made
    /// them (another model, the idle release). `None`: off.
    host_cache: Option<Arc<crate::cache_ram::SharedCache>>,
    /// The lane queues of models served with `--parallel` lanes.
    lanes: lanes::LaneHub,
}

impl AppState {
    /// The model loaded and ready on the card, if any.
    fn ready_model(&self) -> Option<usize> {
        match self.slot.lock().unwrap().lifecycle {
            Lifecycle::Ready(i) => Some(i),
            _ => None,
        }
    }

    fn find_model(&self, requested: Option<&str>) -> usize {
        let Some(name) = requested.filter(|s| !s.is_empty()) else {
            return self.slot.lock().unwrap().resident_idx.unwrap_or(0);
        };
        if let Some(i) = self.models.iter().position(|m| m.id == name) {
            return i;
        }
        if let Some(i) = self
            .models
            .iter()
            .position(|m| m.id.eq_ignore_ascii_case(name))
        {
            return i;
        }
        0
    }

    fn publish_engine_event(&self) {
        let hello = self.hello_json();
        let eng = self.engine_status_json(true);
        self.hub.publish(hello);
        self.hub.publish(eng);
    }

    /// The drafter the configuration gives model `idx` ([`planned_drafter_of`]).
    fn planned_drafter(&self, idx: usize) -> Option<(Option<PathBuf>, &'static str, Vec<u64>)> {
        planned_drafter_of(&self.cfg, &self.models[idx].id)
    }

    /// How model `idx` runs: read from its engine while it is resident, else what the
    /// configuration will load (`from_engine == false`).
    fn run_facts(&self, idx: usize) -> RunFacts {
        {
            let s = self.slot.lock().unwrap();
            if let (Some(f), Lifecycle::Ready(r)) = (&s.facts, &s.lifecycle) {
                if *r == idx {
                    return f.clone();
                }
            }
        }
        let m = &self.models[idx];
        RunFacts {
            from_engine: false,
            // the configured precision, else the engine's automatic one: FP8 graphs for a
            // block-FP8 checkpoint
            fp8_weights: {
                use crate::engine::fp8_graph::WeightPrecision as P;
                let configured = self
                    .cfg
                    .per_model
                    .iter()
                    .find(|(id, _)| *id == m.id)
                    .and_then(|(_, o)| o.precision);
                match configured {
                    Some(P::Fp8) => true,
                    Some(P::Bf16) => false,
                    _ => m.spec.fp8_quantized,
                }
            },
            moe_routed: self
                .cfg
                .per_model
                .iter()
                .any(|(id, o)| *id == m.id && o.moe_routed),
            kv_dtype: self
                .cfg
                .per_model
                .iter()
                .find(|(id, _)| *id == m.id)
                .and_then(|(_, o)| o.kv_dtype)
                .unwrap_or(crate::verified::KvDtype::KvBf16),
            drafter: self.planned_drafter(idx).map(|(dir, kind, mut ks)| {
                ks.sort_unstable_by(|a, b| b.cmp(a));
                DrafterFacts {
                    kind: kind.to_string(),
                    draft_k: ks,
                    // every device drafter verifies sampled requests too
                    samples: true,
                    tree: None,
                    matrix: None,
                    checkpoint: dir.as_deref().and_then(dir_name),
                }
            }),
            vision: m.vision.is_some(),
            hbm_bytes: None,
        }
    }

    /// Data types of model `idx`'s step graph under `facts` (built once per precision / form).
    fn step_dtypes(
        &self,
        idx: usize,
        facts: &RunFacts,
    ) -> Option<crate::engine::qwen3_traffic::StepDtypes> {
        let key = (
            idx,
            facts.fp8_weights,
            facts.moe_routed,
            crate::verified::kv_is_fp8(facts.kv_dtype),
        );
        if let Some(v) = self.dtypes.lock().unwrap().get(&key) {
            return v.clone();
        }
        let spec = readout::run_spec(&self.models[idx].spec, facts);
        // a graph description only; a shape the builders refuse yields no data types
        let v = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::engine::qwen3_traffic::step_dtypes(&spec)
        }))
        .ok()
        .flatten();
        self.dtypes.lock().unwrap().insert(key, v.clone());
        v
    }

    fn model_desc_with(&self, idx: usize, facts: &RunFacts) -> String {
        let m = &self.models[idx];
        let dt = self.step_dtypes(idx, facts);
        readout::model_desc_json(
            &m.id,
            &dir_name(&m.dir).unwrap_or_else(|| m.id.clone()),
            &m.spec,
            m.max_ctx,
            facts,
            dt.as_ref(),
        )
    }

    fn engine_info_json(&self, idx: usize, startup_s: Option<f64>) -> String {
        let m = &self.models[idx];
        let facts = self.run_facts(idx);
        let desc = self.model_desc_with(idx, &facts);
        let (greedy, sampled, loaded, greedy_only) = readout::method_set(facts.drafter.as_ref());
        let startup_str = startup_s
            .filter(|_| facts.from_engine)
            .map(|v| format!(",\"startup_s\":{v:.1}"))
            .unwrap_or_default();
        format!(
            "{{\"model\":{},\"precision\":{},\"n_ctx\":{},\"greedy\":{},\"sampled\":{},\
             \"methods\":{},\"greedy_only\":[{}],\"speculative\":{},\"build\":{}{startup_str},\
             \"model_desc\":{desc}}}",
            jq(&m.id),
            jq(readout::precision_word(&facts)),
            m.max_ctx,
            jq(&greedy),
            jq(&sampled),
            jq(&loaded),
            greedy_only
                .iter()
                .map(|g| jq(g))
                .collect::<Vec<_>>()
                .join(","),
            facts.drafter.is_some(),
            jq(&format!("gaudium-rust {}", env!("CARGO_PKG_VERSION"))),
        )
    }

    fn hello_json(&self) -> String {
        let (res_idx, state_str, warmup_s) = {
            let s = self.slot.lock().unwrap();
            (
                s.resident_idx.unwrap_or(0),
                s.lifecycle.as_str(),
                s.last_warmup_s,
            )
        };
        let m = &self.models[res_idx];
        let facts = self.run_facts(res_idx);
        let info = self.engine_info_json(res_idx, warmup_s);
        let desc = self.model_desc_with(res_idx, &facts);
        let (greedy, sampled, loaded, greedy_only) = readout::method_set(facts.drafter.as_ref());
        format!(
            "{{\"type\":\"hello\",\"backend\":\"engine\",\"model\":{},\"n_ctx\":{},\"vision\":{},\
             \"speculative\":{},\"precision\":{},\
             \"methods\":{{\"greedy\":{},\"sampled\":{},\"loaded\":{},\"greedy_only\":[{}]}},\
             \"engine_info\":{info},\"model_desc\":{desc},\
             \"build\":{},\"supports_inject\":false,\"mock_card\":false,\
             \"state\":{},\"ui\":{{\"root\":\"gaudium\",\"gaudium\":\"/\",\"llama\":\"/\"}},\
             \"api\":{{\"events\":\"/gaudium/events\",\"live\":\"/gaudium/live\",\"engine\":\"/gaudium/engine\",\
             \"count_tokens\":\"/gaudium/count_tokens\",\"version\":1}}}}",
            jq(&m.id),
            m.max_ctx,
            facts.vision,
            facts.drafter.is_some(),
            jq(readout::precision_word(&facts)),
            jq(&greedy),
            jq(&sampled),
            jq(&loaded),
            greedy_only.iter().map(|g| jq(g)).collect::<Vec<_>>().join(","),
            jq(&format!("gaudium-rust {}", env!("CARGO_PKG_VERSION"))),
            jq(state_str),
        )
    }

    fn engine_status_json(&self, include_type: bool) -> String {
        // per-model facts first: `run_facts` takes the slot lock itself
        let per_model: Vec<(bool, &'static str, String, String)> = (0..self.models.len())
            .map(|i| {
                let f = self.run_facts(i);
                let (g, smp, _, _) = readout::method_set(f.drafter.as_ref());
                (f.drafter.is_some(), readout::precision_word(&f), g, smp)
            })
            .collect();
        let s = self.slot.lock().unwrap();
        let res_idx = s.resident_idx.unwrap_or(0);
        let state_for_s = s.since.elapsed().as_secs();
        let idle_for_s = s.last_activity.elapsed().as_secs();
        let active_reqs = s.active_requests;
        let waiting_total: u64 = s.waiting.iter().sum();
        let state_str = s.lifecycle.as_str();
        let detail = s.detail.clone();
        let hold = s.hold;
        let admin_parked = s.admin_parked;
        let switches = s.switches;
        let last_switch_s = s.last_switch_s;
        let last_warmup_s = s.last_warmup_s;
        let last_error = match &s.lifecycle {
            Lifecycle::Failed(err) => err.clone(),
            _ => String::new(),
        };
        let resident_id = self.models[res_idx].id.clone();
        let switching_to = match s.lifecycle {
            Lifecycle::Loading(idx) if Some(idx) != s.resident_idx => {
                Some(self.models[idx].id.clone())
            }
            _ => None,
        };
        let mut models_arr = String::from("[");
        for (i, m) in self.models.iter().enumerate() {
            if i > 0 {
                models_arr.push(',');
            }
            let mstate = match &s.lifecycle {
                Lifecycle::Ready(r) if *r == i => "ready",
                Lifecycle::Loading(r) if *r == i => "loading",
                Lifecycle::Failed(_) if s.resident_idx == Some(i) => "failed",
                _ => "parked",
            };
            let is_res = s.resident_idx == Some(i) && matches!(s.lifecycle, Lifecycle::Ready(_));
            models_arr.push_str(&format!(
                "{{\"id\":{},\"state\":{},\"resident\":{},\"running\":{},\"waiting\":{},\"n_ctx\":{},\"vision\":{},\"speculative\":{},\"precision\":{},\"greedy\":{},\"sampled\":{}}}",
                jq(&m.id),
                jq(mstate),
                is_res,
                if is_res { active_reqs } else { 0 },
                s.waiting.get(i).copied().unwrap_or(0),
                m.max_ctx,
                m.vision.is_some(),
                per_model[i].0,
                jq(per_model[i].1),
                jq(&per_model[i].2),
                jq(&per_model[i].3),
            ));
        }
        models_arr.push(']');
        drop(s);

        let info = self.engine_info_json(res_idx, last_warmup_s);
        let holder_str = if state_str == "ready" || state_str == "loading" {
            "\"server/gaudium-rust\""
        } else {
            "\"\""
        };
        let card_holder_str = if state_str == "ready" || state_str == "loading" {
            "\"server/gaudium-rust\""
        } else {
            "null"
        };
        let warmup_val = last_warmup_s
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "null".into());
        // the last measured start of this process; unknown (null) before the first one
        let expected_ready = warmup_val.clone();
        // this server's only idle rule: it releases the card after `--idle-park` seconds idle
        let idle_yield = if self.cfg.idle_park_secs > 0 {
            format!("\"idle_yield_s\":{},", self.cfg.idle_park_secs)
        } else {
            String::new()
        };

        let mut out = String::with_capacity(1024);
        out.push('{');
        if include_type {
            out.push_str("\"type\":\"engine\",");
        }
        out.push_str(&format!(
            "\"available\":true,\"mock\":false,\"state\":{},\"detail\":{},\"state_for_s\":{},\
             \"holder\":{holder_str},\"owner\":\"server/gaudium-rust\",\"waiting_jobs\":{waiting_total},\
             \"idle_s\":{idle_for_s},\"idle_for_s\":{idle_for_s},{idle_yield}\
             \"idle_release_s\":{},\"idle_park_s\":{},\
             \"expected_ready_s\":{expected_ready},\"last_warmup_s\":{warmup_val},\"last_start_to_ready_s\":{warmup_val},\
             \"last_error\":{},\"info\":{info},\
             \"hold\":{hold},\"admin_parked\":{admin_parked},\"running\":{active_reqs},\"residency\":\"swap\",\"resident\":{},\
             \"switching_to\":{},\"switches\":{switches},\"last_switch_s\":{},\"models\":{models_arr},\
             \"card\":{{\"owner\":\"server/gaudium-rust\",\"holder\":{card_holder_str},\"waiting_jobs\":{waiting_total}}}}}",
            jq(state_str),
            jq(&detail),
            state_for_s,
            self.cfg.idle_park_secs,
            self.cfg.idle_park_secs,
            jq(&last_error),
            jq(&resident_id),
            switching_to.as_deref().map(jq).unwrap_or_else(|| "null".into()),
            last_switch_s
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "null".into()),
        ));
        out
    }

    fn live_snapshot_json(&self) -> String {
        let (last_live, last_req) = {
            let s = self.slot.lock().unwrap();
            (s.last_live_json.clone(), s.last_request_json.clone())
        };
        let mut out = format!(
            "{{\"hello\":{},\"engine\":{}",
            self.hello_json(),
            self.engine_status_json(true),
        );
        if let Some(r) = last_req {
            out.push_str(&format!(",\"request\":{r}"));
        }
        if let Some(live) = last_live {
            out.push_str(&format!(",\"live\":{live}"));
        }
        out.push_str(&format!(",\"seq\":{}}}", self.hub.last_seq()));
        out
    }

    /// The slots of the resident model: one per lane (`--parallel`), each with its state and its
    /// request, and what the model runs as.
    fn slots_json(&self) -> String {
        let (active, idx) = {
            let s = self.slot.lock().unwrap();
            (s.active_requests, s.resident_idx.unwrap_or(0))
        };
        let m = &self.models[idx];
        let facts = self.run_facts(idx);
        let (greedy, sampled, _, _) = readout::method_set(facts.drafter.as_ref());
        let view = if m.lanes > 1 {
            self.lanes.view(idx)
        } else {
            Vec::new()
        };
        let mut out = String::from("[");
        for lane in 0..m.lanes.max(1) {
            if lane > 0 {
                out.push(',');
            }
            let v = view.get(lane);
            let state = v.map_or(
                if m.lanes <= 1 && active > 0 {
                    "busy"
                } else {
                    "free"
                },
                |v| v.state,
            );
            let processing = if m.lanes > 1 {
                state != "free"
            } else {
                active > 0
            };
            out.push_str(&format!(
                "{{\"id\":{lane},\"is_processing\":{processing},\"state\":{},\"model\":{},\"n_ctx\":{},\
                 \"speculative\":{},\"precision\":{},\"greedy\":{},\"sampled\":{},\"loaded\":{}",
                jq(state),
                jq(&m.id),
                m.max_ctx,
                facts.drafter.is_some(),
                jq(readout::precision_word(&facts)),
                jq(&greedy),
                jq(&sampled),
                facts.from_engine,
            ));
            if let Some(v) = v.filter(|v| v.job.is_some()) {
                out.push_str(&format!(
                    ",\"job\":{},\"prompt_tokens\":{},\"reused_tokens\":{},\"generated\":{}",
                    v.job.unwrap_or(0),
                    v.prompt_tokens,
                    v.reused,
                    v.generated
                ));
            }
            out.push('}');
        }
        out.push(']');
        out
    }

    /// Publishes a `request` event and keeps it for `/gaudium/live`.
    fn publish_request(&self, json: String) {
        self.slot.lock().unwrap().last_request_json = Some(json.clone());
        self.hub.publish(json);
    }

    fn model_entry_json(&self, idx: usize) -> String {
        let m = &self.models[idx];
        let facts = self.run_facts(idx);
        let (greedy, sampled, _, _) = readout::method_set(facts.drafter.as_ref());
        let s = self.slot.lock().unwrap();
        let (status_val, resident) = match &s.lifecycle {
            Lifecycle::Ready(r) if *r == idx => ("loaded", true),
            Lifecycle::Loading(r) if *r == idx => ("loading", false),
            Lifecycle::Failed(_) if s.resident_idx == Some(idx) => ("failed", false),
            _ => ("unloaded", false),
        };
        drop(s);

        let n_ctx_train = m.n_ctx_train.map_or("null".to_string(), |n| n.to_string());
        let mut out = format!(
            "{{\"id\":{},\"object\":\"model\",\"owned_by\":\"gaudium\",\"created\":0,\
             \"context_length\":{},\"architecture\":{{\"input_modalities\":[\"text\"]}},\
             \"meta\":{{\"n_ctx_train\":{n_ctx_train},\"speculative\":{},\"precision\":{},\
             \"greedy\":{},\"sampled\":{},\"family\":{},\"layers\":{},\"hidden\":{}",
            jq(&m.id),
            m.max_ctx,
            facts.drafter.is_some(),
            jq(readout::precision_word(&facts)),
            jq(&greedy),
            jq(&sampled),
            jq(&format!("{:?}", m.spec.family)),
            m.spec.n_layers,
            m.spec.hidden,
        );
        if let Some(d) = &facts.drafter {
            out.push_str(&format!(",\"drafter\":{}", jq(&d.kind)));
        }
        out.push('}');
        if self.models.len() > 1 {
            out.push_str(&format!(
                ",\"status\":{{\"value\":{}}},\"resident\":{}",
                jq(status_val),
                resident
            ));
        }
        out.push('}');
        out
    }

    fn props_json(&self, idx: usize, include_router: bool) -> String {
        let m = &self.models[idx];
        let speculative = self.run_facts(idx).drafter.is_some();
        let gd = m.gen_defaults;
        let (d_temp, d_top_k, d_top_p) = (gd.temperature, gd.top_k, gd.top_p);
        // tools reach the model through its own chat template
        let tools = m.chat.as_ref().is_some_and(|t| t.supports_tools);
        // the reasoning-effort levels of the template (the verified plan keeps each of them)
        let effort = m.chat.as_ref().map_or(EffortLevels::NONE, |t| t.effort);
        let effort_on = effort.supported();
        let effort_levels = format!(
            "[{}]",
            effort.advertised().into_iter().map(jq).collect::<Vec<_>>().join(",")
        );
        let family = jq(template_family(m.spec.family));
        // what a request's images and videos must fit (the Web UI sizes its estimates by it)
        let vision = m.vision.is_some();
        let video = m.vision.as_ref().is_some_and(|v| v.spec.video.is_some());
        let vision_limits = match &m.vision {
            Some(v) => format!(
                "{{\"max_patches\":{},\"max_tokens\":{},\"patch\":{},\"merge\":{},\"min_pixels\":{},\
                 \"max_pixels\":{},\"max_file_bytes\":{},\"max_items\":{},\"max_request_bytes\":{},\"video\":{}}}",
                v.max_patches,
                v.max_tokens,
                v.spec.patch,
                v.spec.merge,
                v.spec.min_pixels,
                v.spec.max_pixels,
                v.limits.max_bytes,
                v.limits.max_items,
                MAX_REQUEST_BODY_BYTES,
                match v.spec.video {
                    Some(x) => format!(
                        "{{\"fps\":{},\"min_frames\":{},\"max_frames\":{},\"min_pixels\":{},\"max_pixels\":{}}}",
                        x.fps, x.min_frames, x.max_frames, x.min_pixels, x.max_pixels
                    ),
                    None => "null".into(),
                },
            ),
            None => "null".into(),
        };
        let mut out = format!(
            "{{\"default_generation_settings\":{{\"n_ctx\":{},\"params\":{{\"seed\":-1,\"temperature\":{d_temp},\
             \"top_k\":{d_top_k},\"top_p\":{d_top_p},\"min_p\":0.0,\"max_tokens\":-1,\"n_predict\":-1,\"stream\":false,\
             \"timings_per_token\":false,\"reasoning_format\":\"auto\",\"speculative\":{speculative},\
             \"samplers\":[\"top_k\",\"top_p\",\"min_p\",\"temperature\"]}}}},\
             \"total_slots\":{},\"model_alias\":{},\"model_path\":{},\
             \"modalities\":{{\"vision\":{vision},\"audio\":false,\"video\":{video}}},\
             \"vision\":{vision_limits},\
             \"endpoint_slots\":true,\"endpoint_props\":false,\"endpoint_metrics\":false,\"ui\":{},\
             \"ui_settings\":{{}},\"chat_template\":{family},\
             \"chat_template_caps\":{{\"supports_string_content\":true,\"supports_typed_content\":true,\
             \"supports_tools\":{tools},\"supports_tool_calls\":{tools},\"supports_parallel_tool_calls\":{tools},\
             \"supports_system_role\":true,\"supports_preserve_reasoning\":true,\"supports_reasoning_effort\":{effort_on},\
             \"supports_object_arguments\":false,\"reasoning_effort_levels\":{effort_levels}}},\
             \"bos_token\":{},\"eos_token\":{},\"build_info\":{},\"is_sleeping\":false,\
             \"cors_proxy_enabled\":{},\"a2ui\":{}",
            m.max_ctx,
            m.lanes.max(1),
            jq(&m.id),
            jq(&m.dir.display().to_string()),
            self.cfg.ui_dir.is_some(),
            jq(m.chat.as_ref().map_or("", |t| t.token("bos_token"))),
            jq(m.chat.as_ref().map_or("<|im_end|>", |t| t.token("eos_token"))),
            jq(&format!("gaudium-rust {}", env!("CARGO_PKG_VERSION"))),
            self.proxy.enabled(),
            a2ui::capabilities_json(self.cfg.a2ui, false),
        );
        // what this model is and how it runs (the same description the event stream carries)
        out.push_str(&format!(
            ",\"model_desc\":{}",
            self.model_desc_with(idx, &self.run_facts(idx))
        ));
        // prefix reuse of this model's engine while it is on the card (null: not loaded, or an
        // engine that prefills every prompt whole)
        let prefix = {
            let s = self.slot.lock().unwrap();
            match (&s.lifecycle, s.prefix) {
                (Lifecycle::Ready(r), Some(p)) if *r == idx => readout::prefix_json(&p),
                _ => "null".to_string(),
            }
        };
        out.push_str(&format!(",\"prefix_cache\":{prefix}"));
        // the host-RAM tier: what it holds for every model (null: off), and the resident
        // engine's measured rates while this model is on the card
        let cache_ram = match &self.host_cache {
            Some(c) => {
                let facts = {
                    let s = self.slot.lock().unwrap();
                    match (&s.lifecycle, s.host_tier) {
                        (Lifecycle::Ready(r), Some(f)) if *r == idx => Some(f),
                        _ => None,
                    }
                };
                readout::cache_ram_json(&c.stats(), facts.as_ref())
            }
            None => "null".to_string(),
        };
        out.push_str(&format!(",\"cache_ram\":{cache_ram}"));
        if include_router && self.models.len() > 1 {
            out.push_str(",\"role\":\"router\",\"max_instances\":1,\"models_autoload\":true");
        }
        out.push('}');
        out
    }

    /// Ensures `target_idx` is loaded and resident on Gaudi 2, cleanly parking any previous model.
    fn ensure_loaded(&self, target_idx: usize, force_unpark: bool) -> Result<(), String> {
        let _op = self.card_op.lock().unwrap();
        let need_drop_prev;
        {
            let mut s = self.slot.lock().unwrap();
            if s.admin_parked && !force_unpark {
                return Err("engine is parked (card released); click Start or POST /gaudium/engine {\"action\":\"start\"} to resume".into());
            }
            if matches!(s.lifecycle, Lifecycle::Ready(cur) if cur == target_idx)
                && s.engine.is_some()
            {
                s.last_activity = Instant::now();
                return Ok(());
            }
            if s.active_requests > 0 {
                return Err("cannot switch resident model while a request is active".into());
            }
            need_drop_prev = s.engine.is_some();
            if need_drop_prev {
                s.lifecycle = Lifecycle::Stopping;
                s.detail = format!(
                    "releasing card before loading {}",
                    self.models[target_idx].id
                );
                s.since = Instant::now();
                // what the card holds for its conversation goes to host memory first
                if let Some(e) = s.engine.as_mut() {
                    e.before_release();
                }
                s.engine = None; // drops Qwen3Engine -> drains streams & releases /dev/accel/accel0
                s.hbm_bytes = None;
                s.facts = None;
                s.prefix = None;
                s.host_tier = None;
                s.lifecycle = Lifecycle::Parked;
            }
            if !can_switch_resident_model(s.active_requests, s.lifecycle.as_backend_state()) {
                return Err("verified single-resident guard refused card switch".into());
            }
            s.admin_parked = false;
            s.resident_idx = Some(target_idx);
            s.lifecycle = Lifecycle::Loading(target_idx);
            s.detail = format!("loading {} onto Gaudi 2", self.models[target_idx].id);
            s.since = Instant::now();
        }
        self.publish_engine_event();

        if need_drop_prev {
            // Brief grace interval for the accelerator driver's post-release reset cycle.
            thread::sleep(Duration::from_millis(1200));
        }

        let m = &self.models[target_idx];
        let t0 = Instant::now();
        let (mut qcfg, drafter_dir) = engine_config_of(&self.cfg, &m.id, &m.dir, m.max_ctx);
        qcfg.host_cache = self.host_cache.clone();

        match load_engine(&qcfg) {
            Ok(eng) => {
                let dt = t0.elapsed().as_secs_f64();
                let hbm = eng.hbm_bytes();
                let prefix = eng.prefix_stats();
                let host_tier = eng.host_tier_facts();
                // what the readout says about this model comes from the engine that loaded it
                let facts = RunFacts {
                    from_engine: true,
                    fp8_weights: eng.spec().fp8_weights,
                    moe_routed: eng.spec().moe_routed,
                    kv_dtype: eng.spec().kv_dtype,
                    drafter: eng.spec_info().map(|i| {
                        DrafterFacts::from_engine(&i, drafter_dir.as_deref().and_then(dir_name))
                    }),
                    vision: eng.vision_spec().is_some(),
                    hbm_bytes: Some(hbm),
                };
                let mut s = self.slot.lock().unwrap();
                if need_drop_prev {
                    s.switches += 1;
                    s.last_switch_s = Some(dt);
                }
                s.last_warmup_s = Some(dt);
                s.engine = Some(eng);
                s.resident_idx = Some(target_idx);
                s.lifecycle = Lifecycle::Ready(target_idx);
                s.detail = format!("{} ready ({:.1}s init)", m.id, dt);
                s.since = Instant::now();
                s.last_activity = Instant::now();
                s.hbm_bytes = Some(hbm);
                s.facts = Some(facts);
                s.prefix = prefix;
                s.host_tier = host_tier;
                drop(s);
                self.publish_engine_event();
                if self.cfg.verbose {
                    eprintln!("[server] model '{}' ready on Gaudi 2 in {:.2}s", m.id, dt);
                }
                Ok(())
            }
            Err(e) => {
                // a plan that does not fit says what the tower takes of it
                let e = match (&m.vision, e.contains("HBM")) {
                    (Some(v), true) => format!(
                        "{e} (this model is loaded with a vision tower for images of {} patches: a smaller --model-vision {}=<patches>, or =off, needs less device memory)",
                        v.max_patches, m.id
                    ),
                    _ => e,
                };
                let mut s = self.slot.lock().unwrap();
                s.engine = None;
                s.facts = None;
                s.lifecycle = Lifecycle::Failed(e.clone());
                s.detail = format!("load failed: {e}");
                s.since = Instant::now();
                drop(s);
                self.publish_engine_event();
                Err(e)
            }
        }
    }

    /// The engine's idle time after a chat request (the host-RAM tier saves what the card holds
    /// for its conversation). Called when any connection has ended, on that connection's
    /// thread, and done only while the card is free: a request that already holds or waits
    /// for the card comes first, and the work gives way to one that arrives.
    ///
    /// The work a chat request left pending stays pending until a caller can really start it
    /// (the verified `hc_idle_starts`, `hc_idle_left`). A connection that ends while the
    /// request still holds the card finds the card busy and changes nothing; the request's own
    /// connection, or whichever ends next, then does the work.
    fn idle_work(&self) {
        if self.host_cache.is_none() {
            return;
        }
        // not while another request has the card (it saves what it would overwrite itself)
        let Ok(_op) = self.card_op.try_lock() else {
            return;
        };
        let taken = {
            let mut s = self.slot.lock().unwrap();
            let ready = matches!(s.lifecycle, Lifecycle::Ready(_)) && s.engine.is_some();
            let starts = crate::verified::hc_idle_starts(s.idle_due, s.active_requests, ready);
            s.idle_due = crate::verified::hc_idle_left(s.idle_due, starts);
            if starts {
                s.engine.take()
            } else {
                None
            }
        };
        let Some(mut engine) = taken else { return };
        let waiting = || -> u64 { self.slot.lock().unwrap().waiting.iter().sum() };
        engine.idle_work(&waiting);
        let host_tier = engine.host_tier_facts();
        let mut s = self.slot.lock().unwrap();
        s.engine = Some(engine);
        s.host_tier = host_tier;
    }

    /// Cleanly drains and releases `/dev/accel/accel0`.
    fn park(&self, admin_parked: bool, reason: &str) -> Result<(), String> {
        let _op = self.card_op.lock().unwrap();
        let mut s = self.slot.lock().unwrap();
        if s.active_requests > 0 {
            return Err("cannot release card while a request is active".into());
        }
        s.admin_parked = admin_parked;
        let had_engine = s.engine.is_some();
        if had_engine {
            s.lifecycle = Lifecycle::Stopping;
            s.detail = reason.to_string();
            s.since = Instant::now();
            // what the card holds for its conversation goes to host memory first, so the next
            // turn after the release does not prefill it again (not when the server is
            // stopping: the store goes with the process)
            if let Some(e) = s.engine.as_mut().filter(|_| !crate::signals::term_requested()) {
                e.before_release();
            }
            s.engine = None; // drains streams & releases device + unlinks cardlock holder/live/shm
            s.hbm_bytes = None;
            s.facts = None;
            s.prefix = None;
            s.host_tier = None;
        }
        s.lifecycle = Lifecycle::Parked;
        s.detail = reason.to_string();
        s.since = Instant::now();
        drop(s);
        self.publish_engine_event();
        if had_engine && self.cfg.verbose {
            eprintln!("[server] card released ({reason})");
        }
        Ok(())
    }
}

/// The host-RAM tier's store of a server with this configuration (`None`: off): its budget
/// from `--cache-ram`, or the default share of the host memory available now, checked against
/// what the host and its control groups can give beside what a loaded engine takes for its
/// copies (the pinned transfer ring, and the buffer of its first copies at each load). A
/// budget that was asked for and cannot be honoured is an error with the numbers. This is the
/// plan; each of those allocations is asked for again when it is made
/// (`hostmem::Admitted`).
fn host_cache_of(cfg: &ServerConfig) -> Result<Option<Arc<crate::cache_ram::SharedCache>>, String> {
    use crate::synapse::xfer::{XferRing, RING_BUFS, RING_BUF_BYTES};
    if cfg.prefix_slots == 0 {
        if cfg.cache_ram_bytes.is_some_and(|b| b > 0) {
            eprintln!("[server] cache-ram: off (--prefix-cache 0: no engine keeps a position to resume a prefill from)");
        }
        return Ok(None);
    }
    let ring = XferRing::planned_bytes(RING_BUFS, RING_BUF_BYTES);
    let first_copies = crate::engine::qwen3::host_tier::first_copies_bytes();
    let mem = crate::hostmem::HostMemory::read();
    let (budget, how) =
        crate::cache_ram::resolve_budget(cfg.cache_ram_bytes, ring + first_copies, &mem)?;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    if budget == 0 {
        eprintln!("[server] cache-ram: {how}");
        return Ok(None);
    }
    eprintln!(
        "[server] cache-ram: {:.2} GiB of host memory for prefixes the card no longer holds ({how}); taken as prefixes are saved, plus {} MiB of pinned transfer buffers while a model is loaded and {} MiB for the first copies at each load",
        gib(budget),
        ring >> 20,
        first_copies >> 20,
    );
    // a budget the operator named stays theirs when a limit stops being readable later
    Ok(Some(Arc::new(
        crate::cache_ram::SharedCache::new(budget, ring, cfg.verbose, true)
            .budget_named(cfg.cache_ram_bytes.is_some()),
    )))
}

/// Starts the Gaudium HTTP/SSE + Web UI server (blocks until terminated).
/// The drafter the configuration gives the served model `id`: `(checkpoint, kind, draft widths)`.
/// The drafter kind is read off the checkpoint directory name: a block drafter (`dflash2`,
/// one width), a model's own per-token drafter (`mtp`, a `-assistant` checkpoint) or a
/// DSpine drafter (`dspine`, an adaptive pair of widths). `--draft-k` sets the DSpine widths.
///
/// A model configured to draft with its own MTP head has no checkpoint (`None`): 4 drafts
/// per launch, 3 in the routed mixture-of-experts form (its measured best widths).
fn planned_drafter_of(
    cfg: &ServerConfig,
    id: &str,
) -> Option<(Option<PathBuf>, &'static str, Vec<u64>)> {
    let (_, opts) = cfg.per_model.iter().find(|(k, _)| k == id)?;
    if opts.own_mtp {
        let dir = cfg.models.iter().find(|m| m.id == id).map(|m| m.dir.as_path());
        return Some((None, "mtp", vec![own_mtp_width(dir, opts.moe_routed)]));
    }
    let dir = opts.draft_model.as_ref()?;
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let (kind, draft_k) = if name.contains("dflash") {
        ("dflash2", vec![7])
    } else if name.ends_with("-assistant") {
        ("mtp", vec![4])
    } else if !cfg.draft_k.is_empty() {
        ("dspine", cfg.draft_k.clone())
    } else {
        ("dspine", vec![7, 15])
    };
    Some((Some(dir.clone()), kind, draft_k))
}

/// Lanes the configuration serves model `id` with (`--parallel`): its engine and its drafter must
/// have batched steps (the dense Qwen3 / Qwen3.5 / Qwen3.8 engine), else one lane, with the
/// reason.
fn served_lanes(cfg: &ServerConfig, id: &str, spec: &ModelSpec) -> (usize, Option<&'static str>) {
    use crate::verified::{ln_lanes_ok, ln_served_lanes, LaneDrafter};
    let drafter = match planned_drafter_of(cfg, id) {
        None => LaneDrafter::LdNone,
        Some((_, "dspine", _)) => LaneDrafter::LdDSpine,
        Some(_) => LaneDrafter::LdOther,
    };
    let engine_ok = crate::engine::lanes_supported(spec);
    // (the server never calibrates)
    let n = ln_served_lanes(
        cfg.parallel,
        crate::engine::MAX_LANES,
        ln_lanes_ok(engine_ok, drafter, false),
    ) as usize;
    let why = (cfg.parallel > 1 && n <= 1).then_some(if engine_ok {
        "its drafter has no batched steps yet"
    } else {
        "its engine has no batched steps"
    });
    (n, why)
}

/// [`served_lanes`] at registration, saying why a model asked for lanes gets one.
fn lanes_of(cfg: &ServerConfig, id: &str, spec: &ModelSpec) -> usize {
    let (n, why) = served_lanes(cfg, id, spec);
    if let Some(why) = why {
        eprintln!(
            "[server] model '{id}': served with one lane (--parallel {}: {why})",
            cfg.parallel
        );
    }
    n
}

/// Drafts per launch of a model's own MTP drafter (`--draft-model <id>=mtp`): the widths that
/// measured best on the card. Inkling 2's block drafter fills a block of 7 drafts in one launch
/// that costs about 3% of a verify pass, so it runs at the full block.
fn own_mtp_width(dir: Option<&Path>, moe_routed: bool) -> u64 {
    static INKLING: std::sync::OnceLock<Mutex<HashMap<PathBuf, bool>>> = std::sync::OnceLock::new();
    let inkling = dir.is_some_and(|d| {
        let mut seen = INKLING
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *seen.entry(d.to_path_buf()).or_insert_with(|| {
            ModelText::open(d)
                .and_then(|t| t.spec())
                .is_ok_and(|s| s.family == ModelFamily::Inkling2)
        })
    });
    match (inkling, moe_routed) {
        (true, _) => 7,
        (false, true) => 3,
        (false, false) => 4,
    }
}

/// `--effort-levels`: the model's reasoning-effort levels instead of the ones read from its chat
/// template. A level the template raises on is refused.
fn effort_override(
    chat: Option<&mut ChatTemplate>,
    levels: EffortLevels,
) -> Result<(), String> {
    let Some(t) = chat else {
        return match levels.supported() {
            true => Err("the model has no chat template".into()),
            false => Ok(()),
        };
    };
    if let Some(bad) = levels
        .advertised()
        .into_iter()
        .find(|l| !t.effort_renders(l))
    {
        return Err(format!(
            "the model's chat template raises on reasoning_effort '{bad}'"
        ));
    }
    t.effort = levels;
    Ok(())
}

/// The `chat_template` family name `/props` reports.
fn template_family(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen3 => "qwen3",
        ModelFamily::Qwen35Dense | ModelFamily::Qwen35Moe => "qwen3.5",
        ModelFamily::Gemma4 | ModelFamily::Gemma4Assistant => "gemma4",
        ModelFamily::Inkling2 => "inkling2",
    }
}

/// The context the served model `id` runs at: `--max-ctx` (or the model's own `--model-ctx`) in
/// whole KV chunks, clamped to the model's own limit.
fn served_max_ctx(cfg: &ServerConfig, id: &str, dir: &Path) -> u64 {
    let max_ctx = crate::verified::align_up(cfg.max_ctx.max(KV_CHUNK), KV_CHUNK);
    crate::engine::clamp_ctx_to_model(
        cfg.per_model
            .iter()
            .find(|(k, _)| k == id)
            .and_then(|(_, o)| o.max_ctx)
            .map_or(max_ctx, |c| {
                crate::verified::align_up(c.max(KV_CHUNK), KV_CHUNK)
            }),
        dir,
    )
}

/// The vision tower the served model `id` is loaded with: `(largest image in patches, image
/// tokens per request)`, or `None` for text only. A model has a tower when its checkpoint has
/// one, its engine runs it (the Qwen3.5 / 3.8 and Inkling 2 families), its drafter is one that runs with image
/// input (the verified `vis_drafter_ok`), and neither `--no-vision` nor the model's own
/// `--model-vision <id>=off` turned it off. The token count is held to the context.
fn vision_plan_of(cfg: &ServerConfig, id: &str, dir: &Path, max_ctx: u64) -> Option<(u64, u64)> {
    let own = cfg
        .per_model
        .iter()
        .find(|(k, _)| k == id)
        .and_then(|(_, o)| o.vision_patches);
    let patches = match own {
        Some(p) => p,
        None if cfg.vision => cfg.vision_max_patches,
        None => 0,
    };
    if patches == 0 {
        return None;
    }
    // vision: a checkpoint with a tower, or a GGUF model with a projector file beside it
    let spec = crate::vision::model_with_tower(dir)?;
    let runs_tower = matches!(
        spec.family,
        ModelFamily::Qwen35Dense | ModelFamily::Qwen35Moe | ModelFamily::Inkling2
    );
    if !runs_tower {
        return None;
    }
    let drafter = planned_drafter_of(cfg, id).map(|(_, kind, _)| kind);
    if !drafter_runs_with_vision(drafter, spec.family) {
        return None;
    }
    let tokens = planned_image_tokens(
        cfg.vision_max_tokens,
        crate::core::budget::advertised_served_ctx(max_ctx),
    );
    (tokens > 0).then_some((patches, tokens))
}

/// What requests to the served model `id` may carry as images and videos (`None`: text only).
/// An error is a configuration the tower cannot be loaded with.
fn vision_serve_of(
    cfg: &ServerConfig,
    id: &str,
    dir: &Path,
    max_ctx: u64,
) -> Result<Option<VisionServe>, String> {
    let Some((patches, tokens)) = vision_plan_of(cfg, id, dir, max_ctx) else {
        return Ok(None);
    };
    let spec = VisionSpec::from_model_dir(dir)?;
    let ppt = spec.patches_per_token();
    if !patches.is_multiple_of(ppt) {
        return Err(format!(
            "the largest image ({patches} patches) must be a multiple of {ppt} patches"
        ));
    }
    VisionServe::new(spec, patches, tokens, MediaLimits::default()).map(Some)
}

/// The engine configuration the served model `id` is loaded with, and its drafter checkpoint.
/// The one place that turns the server's options into an engine configuration: a model load and
/// [`build_cache`] both take it from here, so the recipes one compiles are the recipes the
/// other looks up.
fn engine_config_of(
    cfg: &ServerConfig,
    id: &str,
    dir: &Path,
    max_ctx: u64,
) -> (Qwen3Config, Option<PathBuf>) {
    let mut qcfg = Qwen3Config::new(dir.to_path_buf(), cfg.kernels_dir.clone());
    qcfg.max_ctx = max_ctx;
    qcfg.cache_dir = cfg.cache_dir.clone();
    qcfg.no_cache = cfg.no_cache;
    qcfg.verbose = cfg.verbose;
    qcfg.prefix_slots = cfg.prefix_slots;
    qcfg.lanes = ModelText::open(dir)
        .and_then(|t| t.spec())
        .map_or(1, |spec| served_lanes(cfg, id, &spec).0 as u64);
    if let Some((patches, tokens)) = vision_plan_of(cfg, id, dir, max_ctx) {
        // The tower's buffers are planned for the largest image. Its recipes are compiled at
        // load for that size (so the shared workspace covers them); other image sizes compile
        // on their first use and stay in the recipe cache.
        qcfg.vision_max_patches = patches;
        qcfg.vision_max_tokens = tokens;
        qcfg.vision_precompile = vec![patches];
        // `--build-cache` also compiles the sizes that usual pictures come to
        if let Ok(Some(serve)) = vision_serve_of(cfg, id, dir, max_ctx) {
            qcfg.vision_cache_sizes = serve.common_patch_counts();
        }
    }
    if let Some((_, opts)) = cfg.per_model.iter().find(|(k, _)| k == id) {
        qcfg.fp8_act = opts.fp8_act.clone();
        qcfg.fp8_cache = opts.fp8_cache.clone();
        qcfg.moe_routed = opts.moe_routed;
        if let Some(d) = opts.kv_dtype {
            qcfg.kv_dtype = d;
        }
        qcfg.kv_scales = opts.kv_scales.clone();
        if let Some(p) = opts.precision {
            qcfg.precision = p;
        }
    }
    let drafter_dir = planned_drafter_of(cfg, id).and_then(|(dir, kind, draft_k)| {
        use crate::engine::qwen3::spec::{SpecConfig, SpecMethod};
        let method = match (kind, dir.clone()) {
            ("dflash2", Some(draft_model)) => SpecMethod::DFlash2 { draft_model },
            ("mtp", Some(draft_model)) => SpecMethod::Assistant { draft_model },
            // the model's own MTP head; draft readouts over the low token ids and the
            // control-token block only (drafts are guesses: the output is unchanged)
            ("mtp", None) => SpecMethod::Mtp {
                tree: None,
                reference: Vec::new(),
                draft_vocab: OWN_MTP_DRAFT_VOCAB,
            },
            (_, Some(draft_model)) => SpecMethod::DSpine { draft_model },
            (_, None) => return None,
        };
        qcfg.spec = Some(SpecConfig { method, draft_k });
        dir
    });
    (qcfg, drafter_dir)
}

/// `--serve ... --build-cache`: compiles into the recipe cache every recipe the served models
/// load, WITHOUT opening the card and without listening (safe beside a running server). `only`
/// limits it to those model ids. Each model takes the engine configuration a load of it takes
/// ([`engine_config_of`]). A model family without an off-card build is skipped with a note.
pub fn build_cache(cfg: &ServerConfig, only: &[String]) -> Result<(), String> {
    if cfg.models.is_empty() {
        return Err("at least one --model <dir> is required for --serve --build-cache".into());
    }
    if let Some(id) = only
        .iter()
        .find(|o| !cfg.models.iter().any(|m| &m.id == *o))
    {
        return Err(format!(
            "--build-cache-only '{id}': no --model with that id"
        ));
    }
    let mut failed: Vec<String> = Vec::new();
    for entry in cfg
        .models
        .iter()
        .filter(|m| only.is_empty() || only.contains(&m.id))
    {
        let max_ctx = served_max_ctx(cfg, &entry.id, &entry.dir);
        let (qcfg, _) = engine_config_of(cfg, &entry.id, &entry.dir, max_ctx);
        eprintln!(
            "[server] build-cache: model '{}' at context {max_ctx}",
            entry.id
        );
        let family = ModelText::open(&entry.dir)
            .and_then(|t| t.spec())
            .map(|s| s.family)
            .map_err(|e| format!("model description of '{}': {e}", entry.id))?;
        if !matches!(
            family,
            ModelFamily::Qwen3 | ModelFamily::Qwen35Dense | ModelFamily::Qwen35Moe
        ) {
            eprintln!(
                "[server] build-cache: model '{}' skipped: no off-card build for family {family:?} (its recipes compile on its first load)",
                entry.id
            );
            continue;
        }
        if let Err(e) = crate::engine::build_cache(&qcfg) {
            eprintln!("[server] build-cache: model '{}' FAILED: {e}", entry.id);
            failed.push(entry.id.clone());
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!("build-cache failed for: {}", failed.join(", ")))
    }
}

/// The HTTPS listener's certificate, as the flags and this machine allow
/// (`verified/tls_listener.bend`, `ls_start`): `None` serves plain HTTP only; an error refuses
/// the start. The key file is read by the TLS library; neither its contents nor the library's
/// view of it appear in a message.
fn start_tls(cfg: &ServerConfig) -> Result<Option<crate::tls::TlsServer>, String> {
    use crate::core::listener;
    let (cert, key) = (cfg.tls_cert.as_deref(), cfg.tls_key.as_deref());
    // the library is only looked for when HTTPS was asked for
    let lib = (cert.is_some() || key.is_some()).then(crate::tls::availability);
    let server = match (cert, key, &lib) {
        (Some(c), Some(k), Some(Ok(_))) => Some(crate::tls::TlsServer::new(c, k)),
        _ => None,
    };
    let start = listener::tls_start(
        cert.is_some(),
        key.is_some(),
        matches!(lib, Some(Ok(_))),
        matches!(server, Some(Ok(_))),
    );
    if listener::start_refused(start) {
        return Err(match server {
            Some(Err(e)) => format!("HTTPS listener (--tls-cert, --tls-key): {e}"),
            _ => "--tls-cert and --tls-key go together: give both files, or neither".into(),
        });
    }
    if listener::start_warns(start) {
        eprintln!(
            "[server] HTTPS was asked for (--tls-cert, --tls-key) and is NOT offered: {}. Serving plain HTTP only.",
            lib.as_ref().and_then(|l| l.as_ref().err()).cloned().unwrap_or_default()
        );
    }
    if !listener::serves_tls(start) {
        return Ok(None);
    }
    if let Some(Ok(version)) = &lib {
        eprintln!("[server] HTTPS on: {version}");
    }
    Ok(server.and_then(Result::ok))
}

/// Longest wait for a connection before the accept loop looks at the stop request again.
const ACCEPT_WAIT_MS: i32 = 200;

pub fn run_server(cfg: ServerConfig) -> Result<(), String> {
    if cfg.models.is_empty() {
        return Err("at least one --model <dir> is required for --serve".into());
    }
    // Settings that can refuse the start are checked before anything is loaded.
    let proxy = proxy::McpProxy::new(&cfg.mcp_proxy, cfg.api_key.is_some())?;
    let tls = start_tls(&cfg)?;
    // Shared stop-request handler; re-asserted by the device layer after every card open, because
    // the runtime replaces signal dispositions while initializing.
    crate::signals::install();
    // The readout reads each engine's launch feed and profile block in-process, whether or not
    // a live-record directory is configured.
    crate::live_report::enable_in_memory();
    if let Some(p) = &cfg.a2ui_prompt {
        a2ui::use_prompt_file(p.clone());
    }

    let mut models = Vec::with_capacity(cfg.models.len());
    for entry in &cfg.models {
        // a checkpoint directory or a GGUF file: its text resources from one place
        let text = ModelText::open(&entry.dir).map_err(|e| format!("model '{}': {e}", entry.id))?;
        let spec = text
            .spec()
            .map_err(|e| format!("model description of '{}': {e}", entry.id))?;
        spec.validate()
            .map_err(|e| format!("invalid ModelSpec for '{}': {e}", entry.id))?;
        let tok = text
            .tokenizer()
            .map_err(|e| format!("load tokenizer for '{}': {e}", entry.id))?;
        let mut extra_eos_ids = Vec::new();
        // `<|tool_response>`: where a Gemma 4 reply hands over to the tool results of its calls
        for special in ["<|im_end|>", "<|endoftext|>", "<turn|>", "<|tool_response>"] {
            if let Some(id) = tok.token_id(special) {
                extra_eos_ids.push(id);
            }
        }
        let mut chat = match ChatTemplate::load_model(&text) {
            Ok(Some((t, note))) => {
                if let Some(note) = note {
                    eprintln!(
                        "[server] model '{}': {note}; reasoning and tool calls are parsed in the default form",
                        entry.id
                    );
                }
                Some(t)
            }
            Ok(None) => {
                eprintln!(
                    "[server] model '{}' has no chat template: built-in text template, no tool calling",
                    entry.id
                );
                None
            }
            Err(e) => {
                eprintln!(
                    "[server] model '{}': chat template not usable ({e}): built-in text template, no tool calling",
                    entry.id
                );
                None
            }
        };
        if let Some(levels) = cfg
            .per_model
            .iter()
            .find(|(k, _)| k == &entry.id)
            .and_then(|(_, o)| o.effort_levels)
        {
            effort_override(chat.as_mut(), levels)
                .map_err(|e| format!("--effort-levels {}: {e}", entry.id))?;
        }
        if let Some(t) = chat.as_ref().filter(|t| t.effort.supported()) {
            eprintln!(
                "[server] model '{}': reasoning effort levels {}",
                entry.id,
                t.effort.advertised().join(", ")
            );
        }
        let format = match &chat {
            Some(t) => t.format.clone(),
            None => builtin_format(spec.family),
        };
        // the tokens that end a reply of the model's own output form (a block form: its
        // end-of-sequence token and the other roles' headers)
        for marker in format.stop_markers() {
            if let Some(id) = tok.token_id(marker).filter(|id| !extra_eos_ids.contains(id)) {
                extra_eos_ids.push(id);
            }
        }
        // `--max-ctx` (or the model's own `--model-ctx`), clamped to the model's own limit.
        let max_ctx = served_max_ctx(&cfg, &entry.id, &entry.dir);
        let vision = vision_serve_of(&cfg, &entry.id, &entry.dir, max_ctx)
            .map_err(|e| format!("vision input of '{}': {e}", entry.id))?;
        let runs_tower = matches!(
            spec.family,
            ModelFamily::Qwen35Dense | ModelFamily::Qwen35Moe | ModelFamily::Inkling2
        );
        if vision.is_none() && spec.has_vision && runs_tower && cfg.vision {
            if let Some((_, kind, _)) =
planned_drafter_of(&cfg, &entry.id).filter(|d| {
                    !drafter_runs_with_vision(Some(d.1), spec.family)
                })
            {
                eprintln!(
                    "[server] model '{}': served text only: its {kind} drafter has not passed the vision regression suite for this model family (plain decode, DSpine, a dense Qwen model's DFlash2 drafter and MTP head, and Inkling 2's own drafter have)",
                    entry.id
                );
            }
        }
        if let Some(v) = &vision {
            eprintln!(
                "[server] model '{}': image{} input, images up to {} patches ({} pixels), {} image tokens per request",
                entry.id,
                if v.spec.video.is_some() { " and video" } else { "" },
                v.max_patches,
                v.spec.max_pixels,
                v.max_tokens,
            );
        }
        let lanes = lanes_of(&cfg, &entry.id, &spec);
        models.push(RegisteredModel {
            id: entry.id.clone(),
            dir: entry.dir.clone(),
            spec,
            chat,
            format,
            tokenizer: Arc::new(tok),
            extra_eos_ids,
            max_ctx,
            vision,
            gen_defaults: text
                .generation_config()
                .map(|t| SamplerDefaults::from_generation_config(&t))
                .unwrap_or(SamplerDefaults::FALLBACK),
            gemma_closes_thought: text
                .chat_template_source()
                .map(|t| crate::tokenizer::gemma4_template_closes_thought(&t))
                .unwrap_or(false),
            n_ctx_train: text.max_positions(),
            lanes,
        });
    }

    let host_cache = host_cache_of(&cfg)?;

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    let _ = listener.set_nonblocking(true);

    let n_models = models.len();
    let state = Arc::new(AppState {
        cfg: cfg.clone(),
        models,
        slot: Mutex::new(EngineSlot {
            engine: None,
            resident_idx: Some(0),
            lifecycle: Lifecycle::Parked,
            detail: if cfg.lazy {
                "ready (lazy: will load on first request)".into()
            } else {
                "initializing".into()
            },
            since: Instant::now(),
            last_activity: Instant::now(),
            admin_parked: false,
            hold: false,
            active_requests: 0,
            waiting: vec![0; n_models],
            switches: 0,
            last_switch_s: None,
            last_warmup_s: None,
            last_live_json: None,
            last_request_json: None,
            hbm_bytes: None,
            facts: None,
            prefix: None,
            host_tier: None,
            idle_due: false,
        }),
        card_op: Mutex::new(()),
        cancels: Mutex::new(HashMap::new()),
        next_req_id: AtomicU64::new(1),
        hub: EventHub::new(),
        dtypes: Mutex::new(HashMap::new()),
        turns: TurnQueue::default(),
        proxy,
        tls,
        host_cache,
        lanes: lanes::LaneHub::new(n_models),
    });

    // Optional eager warm-up in a background thread so HTTP/UI responds immediately!
    if !cfg.lazy {
        let st = Arc::clone(&state);
        thread::Builder::new()
            .name("gaudium-eager-load".into())
            .spawn(move || {
                if let Err(e) = st.ensure_loaded(0, true) {
                    eprintln!("[server] initial model load failed: {e}");
                }
            })
            .ok();
    }

    // Idle auto-park watchdog thread (keeps card free when unused!).
    if cfg.idle_park_secs > 0 {
        let st = Arc::clone(&state);
        let limit = Duration::from_secs(cfg.idle_park_secs);
        thread::Builder::new()
            .name("gaudium-idle-park".into())
            .spawn(move || loop {
                if crate::signals::term_requested() {
                    break;
                }
                thread::sleep(Duration::from_secs(5));
                let should_park = {
                    let s = st.slot.lock().unwrap();
                    matches!(s.lifecycle, Lifecycle::Ready(_))
                        && !s.hold
                        && s.active_requests == 0
                        && s.last_activity.elapsed() >= limit
                };
                if should_park {
                    let _ = st.park(false, "idle auto-release (will reload on next chat)");
                }
            })
            .ok();
    }

    if let Some(line) = state.proxy.summary(cfg.api_key.is_some()) {
        eprintln!("[server] {line}");
    }
    eprintln!(
        "[server] listening on http://{0}/{1} (models: {2}, lazy={3}, idle_park={4}s, ui={5})",
        addr,
        if state.tls.is_some() {
            format!(" and https://{addr}/ (same port)")
        } else {
            String::new()
        },
        state
            .models
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        cfg.lazy,
        cfg.idle_park_secs,
        cfg.ui_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".into()),
    );

    while !crate::signals::term_requested() {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                let st = Arc::clone(&state);
                thread::spawn(move || handle_connection(st, stream));
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Nothing to accept: wait for a connection, not for a sleep tick (a request is
                // taken when it arrives). The timeout bounds how long a stop request waits.
                crate::signals::wait_readable(
                    std::os::fd::AsRawFd::as_raw_fd(&listener),
                    ACCEPT_WAIT_MS,
                );
            }
            Err(e) => {
                eprintln!("[server] accept error: {e}");
                thread::sleep(Duration::from_millis(50));
            }
        }
    }

    // Graceful shutdown on SIGINT / SIGTERM: close SSE streams and release card + cardlock!
    state.hub.close();
    let _ = state.park(false, "server shutdown");
    Ok(())
}

// ============================================================================
// Minimal Zero-Dep HTTP/1.1 Parser & Router
// ============================================================================

struct HttpRequest {
    method: String,
    path: String,
    query: HashMap<String, String>,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut Conn) -> Result<Option<HttpRequest>, String> {
    let mut reader = BufReader::new(stream);
    let mut req_line = String::new();
    let n = reader
        .read_line(&mut req_line)
        .map_err(|e| format!("read request line: {e}"))?;
    if n == 0 {
        return Ok(None);
    }
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err("malformed HTTP request line".into());
    }
    let method = parts[0].to_uppercase();
    let raw_target = parts[1];
    let (raw_path, raw_query) = raw_target
        .split_once('?')
        .map_or((raw_target, ""), |(p, q)| (p, q));
    let path = percent_decode(raw_path);
    let query = parse_query(raw_query);

    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        let r = reader
            .read_line(&mut line)
            .map_err(|e| format!("read header: {e}"))?;
        if r == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    let mut body = Vec::new();
    let is_chunked = headers
        .get("transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
    if is_chunked {
        loop {
            let mut sz_line = String::new();
            reader
                .read_line(&mut sz_line)
                .map_err(|e| format!("read chunk size: {e}"))?;
            let hex = sz_line.trim().split(';').next().unwrap_or("0");
            let sz = usize::from_str_radix(hex, 16)
                .map_err(|_| format!("invalid chunk size '{hex}'"))?;
            if sz == 0 {
                break;
            }
            if body.len().saturating_add(sz) > MAX_REQUEST_BODY_BYTES {
                return Err("request body exceeds 32 MiB limit".into());
            }
            let start = body.len();
            body.resize(start + sz, 0);
            reader
                .read_exact(&mut body[start..])
                .map_err(|e| format!("read chunk bytes: {e}"))?;
            let mut crlf = [0u8; 2];
            let _ = reader.read_exact(&mut crlf);
        }
    } else if let Some(cl_str) = headers.get("content-length") {
        let cl: usize = cl_str.parse().map_err(|_| "invalid Content-Length")?;
        if cl > MAX_REQUEST_BODY_BYTES {
            return Err("request body exceeds 32 MiB limit".into());
        }
        body.resize(cl, 0);
        if cl > 0 {
            reader
                .read_exact(&mut body)
                .map_err(|e| format!("read body: {e}"))?;
        }
    }

    Ok(Some(HttpRequest {
        method,
        path,
        query,
        headers,
        body,
    }))
}

fn write_response_with_body(
    stream: &mut Conn,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
    send_body: bool,
) {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, HEAD, POST, DELETE, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type, Authorization, X-Api-Key\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    if send_body && !body.is_empty() {
        let _ = stream.write_all(body);
    }
    let _ = stream.flush();
}

fn write_response(
    stream: &mut Conn,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) {
    write_response_with_body(stream, status, reason, content_type, body, true);
}

fn write_json(stream: &mut Conn, status: u16, json: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        503 => "Service Unavailable",
        529 => "Overloaded",
        _ => "Internal Server Error",
    };
    write_response(
        stream,
        status,
        reason,
        "application/json; charset=utf-8",
        json.as_bytes(),
    );
}

fn write_error(stream: &mut Conn, status: u16, err_type: &str, msg: &str) {
    let body = format!(
        "{{\"error\":{{\"message\":{},\"type\":{},\"code\":{}}}}}",
        jq(msg),
        jq(err_type),
        status
    );
    write_json(stream, status, &body);
}

fn write_sse_headers(stream: &mut Conn) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream; charset=utf-8\r\n\
                Transfer-Encoding: chunked\r\n\
                Cache-Control: no-cache, no-store\r\n\
                X-Accel-Buffering: no\r\n\
                Access-Control-Allow-Origin: *\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

fn write_sse_chunk(stream: &mut Conn, data: &[u8]) -> std::io::Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    // one write per chunk (one TLS record on an HTTPS connection)
    let mut out = Vec::with_capacity(data.len() + 16);
    out.extend_from_slice(format!("{:x}\r\n", data.len()).as_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(b"\r\n");
    stream.write_all(&out)?;
    stream.flush()
}

fn write_sse_end(stream: &mut Conn) {
    let _ = stream.write_all(b"0\r\n\r\n");
    let _ = stream.flush();
}

fn check_auth(cfg: &ServerConfig, req: &HttpRequest) -> bool {
    let Some(expected) = cfg.api_key.as_deref() else {
        return true;
    };
    if let Some(auth) = req.headers.get("authorization") {
        if let Some(tok) = auth.strip_prefix("Bearer ") {
            if tok.trim() == expected {
                return true;
            }
        }
    }
    req.headers
        .get("x-api-key")
        .is_some_and(|k| k.trim() == expected)
}

/// Serves one accepted socket: plain HTTP, or TLS when the client opens with a handshake and the
/// server has a certificate.
fn handle_connection(state: Arc<AppState>, tcp: TcpStream) {
    if let Some(stream) = Conn::open(tcp, state.tls.as_ref()) {
        handle_request(Arc::clone(&state), stream);
    }
    // The connection is closed: nothing of a response waits for what follows. After a chat
    // request the engine now gets its idle time (the host-RAM tier's save).
    state.idle_work();
}

fn handle_request(state: Arc<AppState>, mut stream: Conn) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let req = match read_http_request(&mut stream) {
        Ok(Some(r)) => r,
        Ok(None) => return,
        Err(e) => {
            write_error(&mut stream, 400, "invalid_request_error", &e);
            return;
        }
    };
    let _ = stream.set_read_timeout(None);

    if req.method == "OPTIONS" {
        write_response(&mut stream, 204, "No Content", "text/plain", &[]);
        return;
    }

    let is_api = req.path.starts_with("/v1/")
        || req.path.starts_with("/gaudium/")
        || matches!(
            req.path.as_str(),
            "/props"
                | "/models"
                | "/models/load"
                | "/models/unload"
                | "/chat/completions"
                | "/apply-template"
                | "/tokenize"
                | "/detokenize"
                | "/slots"
        );
    if is_api && !check_auth(&state.cfg, &req) {
        write_error(
            &mut stream,
            401,
            "authentication_error",
            "invalid or missing API key",
        );
        return;
    }

    match (req.method.as_str(), req.path.as_str()) {
        // every method: the proxy answers the ones it does not forward itself
        (_, "/cors-proxy") => {
            let key_ok = check_auth(&state.cfg, &req);
            state.proxy.handle(&req, key_ok, &mut stream);
        }
        ("GET", "/gaudium/proxy") => {
            write_json(&mut stream, 200, &state.proxy.status_json());
        }
        ("GET", "/health") | ("GET", "/v1/health") => {
            write_json(&mut stream, 200, "{\"status\":\"ok\"}");
        }
        ("GET", "/props") => {
            let model_q = req.query.get("model").map(|s| s.as_str());
            let idx = state.find_model(model_q);
            let json = state.props_json(idx, model_q.is_none());
            write_json(&mut stream, 200, &json);
        }
        ("GET", "/v1/models") | ("GET", "/models") => {
            let mut arr = String::from("{\"object\":\"list\",\"data\":[");
            for i in 0..state.models.len() {
                if i > 0 {
                    arr.push(',');
                }
                arr.push_str(&state.model_entry_json(i));
            }
            arr.push_str("]}");
            write_json(&mut stream, 200, &arr);
        }
        ("GET", p) if p.starts_with("/v1/models/") => {
            let id = &p["/v1/models/".len()..];
            let idx = state.find_model(Some(id));
            write_json(&mut stream, 200, &state.model_entry_json(idx));
        }
        ("POST", "/models/load") => handle_model_load(&state, &req, &mut stream, true),
        ("POST", "/models/unload") => handle_model_load(&state, &req, &mut stream, false),
        ("GET", "/gaudium/ui") => {
            let body = format!(
                "{{\"root\":\"gaudium\",\"gaudium\":\"/\",\"llama\":\"/\",\"events\":true,\"files\":{}}}",
                if state.cfg.ui_dir.is_some() { 1 } else { 0 }
            );
            write_json(&mut stream, 200, &body);
        }
        ("GET", "/gaudium/a2ui") => {
            write_json(
                &mut stream,
                200,
                &a2ui::capabilities_json(state.cfg.a2ui, true),
            );
        }
        ("GET", "/gaudium/engine") => {
            write_json(&mut stream, 200, &state.engine_status_json(false));
        }
        ("POST", "/gaudium/engine") => handle_engine_action(&state, &req, &mut stream),
        ("GET", "/gaudium/live") => {
            write_json(&mut stream, 200, &state.live_snapshot_json());
        }
        ("GET", "/gaudium/events") => handle_events_sse(&state, &mut stream),
        ("GET", "/slots") => {
            write_json(&mut stream, 200, &state.slots_json());
        }
        ("POST", "/v1/streams/lookup") => write_json(&mut stream, 200, "[]"),
        ("POST", "/tokenize") => handle_tokenize(&state, &req, &mut stream),
        ("POST", "/detokenize") => handle_detokenize(&state, &req, &mut stream),
        ("POST", "/gaudium/count_tokens") => handle_count_tokens(&state, &req, &mut stream),
        ("POST", "/v1/chat/completions/control") => handle_chat_control(&state, &req, &mut stream),
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            handle_chat_completions(&state, &req, &mut stream);
        }
        ("POST", "/v1/messages") => handle_messages(&state, &req, &mut stream),
        ("POST", "/v1/messages/count_tokens") => handle_messages_count(&state, &req, &mut stream),
        ("POST", "/apply-template") => handle_apply_template(&state, &req, &mut stream),
        ("GET", _) => handle_static_ui(&state, &req.path, &mut stream, true),
        ("HEAD", _) => handle_static_ui(&state, &req.path, &mut stream, false),
        _ => write_error(
            &mut stream,
            404,
            "not_found_error",
            &format!("unknown endpoint {} {}", req.method, req.path),
        ),
    }
}

fn handle_events_sse(state: &AppState, stream: &mut Conn) {
    if write_sse_headers(stream).is_err() {
        return;
    }
    let mut seq = state.hub.last_seq();
    let init = [state.hello_json(), state.engine_status_json(true)];
    for msg in &init {
        if write_sse_chunk(stream, format!("data: {msg}\n\n").as_bytes()).is_err() {
            return;
        }
    }
    let last_live = state.slot.lock().unwrap().last_live_json.clone();
    if let Some(live) = last_live {
        if write_sse_chunk(stream, format!("data: {live}\n\n").as_bytes()).is_err() {
            return;
        }
    }
    loop {
        let (evs, closed) = state.hub.since(seq, Duration::from_secs(10));
        if closed {
            break;
        }
        if evs.is_empty() {
            if write_sse_chunk(stream, b":\n\n").is_err() {
                break;
            }
            continue;
        }
        for (s, payload) in evs {
            seq = s;
            if write_sse_chunk(stream, format!("data: {payload}\n\n").as_bytes()).is_err() {
                return;
            }
        }
    }
    write_sse_end(stream);
}

fn handle_engine_action(state: &Arc<AppState>, req: &HttpRequest, stream: &mut Conn) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON");
        return;
    };
    let action = val
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = val.get("model").and_then(|v| v.as_str());
    let idx = state.find_model(model);

    let result: Result<String, String> = match action.as_str() {
        "start" | "unpark" | "load" => {
            {
                let mut s = state.slot.lock().unwrap();
                if s.active_requests == 0 {
                    s.resident_idx = Some(idx);
                    s.admin_parked = false;
                }
            }
            state.publish_engine_event();
            let st_clone = Arc::clone(state);
            thread::spawn(move || {
                let _ = st_clone.ensure_loaded(idx, true);
            });
            Ok(format!("loading {}", state.models[idx].id))
        }
        "park" | "stop" => state
            .park(true, "parked by admin")
            .map(|()| "parked".to_string()),
        "release" => state
            .park(false, "released by admin")
            .map(|()| "released".to_string()),
        "hold" => {
            state.slot.lock().unwrap().hold = true;
            state.publish_engine_event();
            Ok("hold enabled".to_string())
        }
        "unhold" => {
            state.slot.lock().unwrap().hold = false;
            state.publish_engine_event();
            Ok("hold disabled".to_string())
        }
        _ => Err(format!(
            "unknown action '{action}' (expected start, unpark, load, park, release, hold, unhold)"
        )),
    };

    match result {
        Ok(msg) => write_json(
            stream,
            200,
            &format!(
                "{{\"success\":true,\"ok\":true,\"message\":{},\"status\":{}}}",
                jq(&msg),
                state.engine_status_json(false)
            ),
        ),
        Err(msg) => write_error(stream, 409, "invalid_request_error", &msg),
    }
}

fn handle_model_load(state: &AppState, req: &HttpRequest, stream: &mut Conn, load: bool) {
    let body_str = String::from_utf8_lossy(&req.body);
    let val = json::parse_json(&body_str).unwrap_or(JsonValue::Null);
    let model = val.get("model").and_then(|v| v.as_str());
    let idx = state.find_model(model);
    if load {
        match state.ensure_loaded(idx, true) {
            Ok(()) => write_json(stream, 200, "{\"success\":true}"),
            Err(e) => write_error(stream, 503, "server_error", &e),
        }
    } else {
        match state.park(false, "unloaded via /models/unload") {
            Ok(()) => write_json(stream, 200, "{\"success\":true}"),
            Err(e) => write_error(stream, 409, "invalid_request_error", &e),
        }
    }
}

fn handle_tokenize(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON");
        return;
    };
    let Some(content) = val.get("content").and_then(|v| v.as_str()) else {
        write_error(
            stream,
            400,
            "invalid_request_error",
            "'content' must be a string",
        );
        return;
    };
    let idx = state.find_model(val.get("model").and_then(|v| v.as_str()));
    let parse_special = val
        .get("parse_special")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let ids = state.models[idx].tokenizer.encode(content, parse_special);
    let list = ids
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    write_json(stream, 200, &format!("{{\"tokens\":[{list}]}}"));
}

fn handle_detokenize(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON");
        return;
    };
    let Some(arr) = val.get("tokens").and_then(|v| v.as_array()) else {
        write_error(
            stream,
            400,
            "invalid_request_error",
            "'tokens' must be an array",
        );
        return;
    };
    let ids: Vec<i32> = arr
        .iter()
        .filter_map(|v| v.as_i64().map(|x| x as i32))
        .collect();
    let idx = state.find_model(val.get("model").and_then(|v| v.as_str()));
    let text = state.models[idx].tokenizer.decode(&ids, false);
    write_json(stream, 200, &format!("{{\"content\":{}}}", jq(&text)));
}

fn handle_count_tokens(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON");
        return;
    };
    let idx = state.find_model(val.get("model").and_then(|v| v.as_str()));
    let m = &state.models[idx];
    let prompt = match prepare_prompt(state.cfg.a2ui, m, &val) {
        Ok(p) => p,
        Err(e) => {
            write_error(stream, 400, "invalid_request_error", &e);
            return;
        }
    };
    // image and video sizes come from the files' headers: no pixels are decoded for a count
    let (ids, media) = match tokenize_prompt(m, &prompt, false) {
        Ok(x) => x,
        Err(e) => {
            write_error(stream, 400, "invalid_request_error", &e);
            return;
        }
    };
    let n_prompt = ids.len() as u64;
    let req_max = val.get("max_tokens").and_then(|v| v.as_u64());
    let (fits, max_tokens) = match SequenceBudget::admit(n_prompt, req_max, m.max_ctx) {
        Ok(b) => (true, b.max_new()),
        Err(_) => (false, 0),
    };
    let c = media.map(|md| md.counts()).unwrap_or_default();
    // `return_tokens`: the prompt's token ids too (the ids a chat request with this body runs)
    let tokens = if val.get("return_tokens").and_then(|v| v.as_bool()) == Some(true) {
        format!(",\"tokens\":{}", ids_json(&ids))
    } else {
        String::new()
    };
    write_json(
        stream,
        200,
        &format!(
            "{{\"n_prompt_tokens\":{n_prompt},\"n_ctx\":{},\"max_tokens\":{max_tokens},\"fits\":{fits},\
             \"images\":{},\"videos\":{},\"image_tokens\":{}{tokens}}}",
            m.max_ctx, c.images, c.videos, c.tokens
        ),
    );
}

/// How far a request for the loaded model may jump the queue: no request is passed over once it has waited
/// `max_wait` or been passed over `max_skips` times. Read from the card-queue settings file
/// (`queue.conf`, keys `group_max_wait_min` and `group_max_skips`), the same file the card lock reads.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TurnLimits {
    max_wait: Duration,
    max_skips: u32,
}

impl Default for TurnLimits {
    fn default() -> Self {
        Self {
            max_wait: Duration::from_secs(45 * 60),
            max_skips: 3,
        }
    }
}

impl TurnLimits {
    /// `key=value` lines, `#` starts a comment; a missing file, key or unparsable value keeps the default.
    fn parse(text: &str) -> Self {
        let mut l = Self::default();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("");
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let Ok(v) = v.trim().parse::<u64>() else {
                continue;
            };
            match k.trim() {
                "group_max_wait_min" => l.max_wait = Duration::from_secs(v.saturating_mul(60)),
                "group_max_skips" => l.max_skips = u32::try_from(v).unwrap_or(u32::MAX),
                _ => {}
            }
        }
        l
    }

    fn read(path: Option<&Path>) -> Self {
        path.and_then(|p| fs::read_to_string(p).ok())
            .map(|t| Self::parse(&t))
            .unwrap_or_default()
    }
}

/// The chat server's card queue. Requests get the card in arrival order, except that a request for the
/// model already loaded goes ahead of older requests for another model, so the card is not unloaded and
/// reloaded between them. The model index is the whole configuration: every load option is fixed per model
/// at startup, and per-request settings (sampling, method) never reload the card. No request is passed over
/// beyond the `TurnLimits` of the settings file (the verified `turn_rank`: a due waiter goes first).
struct TurnQueue {
    mu: Mutex<TurnState>,
    cv: Condvar,
    /// The clock the verified turn rules compare arrivals on (ns since this instant).
    epoch: Instant,
}

impl Default for TurnQueue {
    fn default() -> Self {
        Self {
            mu: Mutex::default(),
            cv: Condvar::new(),
            epoch: Instant::now(),
        }
    }
}

#[derive(Default)]
struct TurnState {
    busy: bool,
    next_id: u64,
    waiting: Vec<TurnWaiter>,
}

struct TurnWaiter {
    id: u64,
    model: usize,
    since: Instant,
    skips: u32,
}

/// May no request pass `w` over any more (the verified `turn_due`)?
fn waiter_due(w: &TurnWaiter, now: Instant, lim: TurnLimits) -> bool {
    let ms = |d: Duration| u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
    crate::core::lanes::waiter_due(
        ms(now.duration_since(w.since)),
        ms(lim.max_wait),
        u64::from(w.skips),
        u64::from(lim.max_skips),
    )
}

/// The waiter that gets the card next: its position in `waiting` (arrival order).
fn next_turn(
    waiting: &[TurnWaiter],
    ready: Option<usize>,
    now: Instant,
    lim: TurnLimits,
) -> Option<usize> {
    crate::core::lanes::next_turn_of(
        waiting
            .iter()
            .map(|w| (waiter_due(w, now, lim), ready == Some(w.model))),
    )
}

/// Holding the card: dropping it hands the card to the next waiter.
struct Turn<'a>(&'a TurnQueue);

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.0.mu.lock().unwrap().busy = false;
        self.0.cv.notify_all();
    }
}

impl TurnQueue {
    /// Waits for this request's turn on the card. `ready` reads the model loaded now (it takes the slot
    /// lock; this queue's lock is always taken first).
    fn acquire(
        &self,
        model: usize,
        ready: impl Fn() -> Option<usize>,
        limits: impl Fn() -> TurnLimits,
    ) -> Turn<'_> {
        self.acquire_from(model, Instant::now(), ready, limits)
    }

    /// [`Self::acquire`] for a waiter that has waited since `since` (a lane session queues at its
    /// oldest request's arrival): it takes its place in arrival order.
    fn acquire_from(
        &self,
        model: usize,
        since: Instant,
        ready: impl Fn() -> Option<usize>,
        limits: impl Fn() -> TurnLimits,
    ) -> Turn<'_> {
        let mut st = self.mu.lock().unwrap();
        let id = st.next_id;
        st.next_id += 1;
        let at = st.waiting.partition_point(|w| w.since <= since);
        st.waiting.insert(
            at,
            TurnWaiter {
                id,
                model,
                since,
                skips: 0,
            },
        );
        loop {
            if !st.busy {
                if let Some(pos) = next_turn(&st.waiting, ready(), Instant::now(), limits()) {
                    if st.waiting[pos].id == id {
                        // the waiters it takes the card before, that arrived first, were passed
                        // over once more (the verified rule a lane session's admissions use)
                        let at = self.clock(st.waiting[pos].since);
                        for w in &mut st.waiting[..pos] {
                            if crate::core::lanes::passes_over(self.clock(w.since), at) {
                                w.skips += 1;
                            }
                        }
                        st.waiting.remove(pos);
                        st.busy = true;
                        return Turn(self);
                    }
                }
            }
            // a timed wait, so a waiter that becomes due by waiting is noticed without a release
            st = self.cv.wait_timeout(st, Duration::from_secs(1)).unwrap().0;
        }
    }

    /// An arrival on the clock the verified turn rules compare (ns since the queue began).
    fn clock(&self, t: Instant) -> u64 {
        u64::try_from(t.saturating_duration_since(self.epoch).as_nanos()).unwrap_or(u64::MAX)
    }

    /// [lanes] Would admitting a request that arrived at `since` pass over a waiter whose turn
    /// is due? (A lane session admits nothing more then.) Every waiter counts, not only the
    /// first: a lane session that queued again stands among them in arrival order.
    fn blocks(&self, since: Instant, lim: TurnLimits) -> bool {
        let st = self.mu.lock().unwrap();
        let now = Instant::now();
        st.waiting.iter().any(|w| {
            crate::core::lanes::admission_blocked(
                waiter_due(w, now, lim),
                self.clock(w.since),
                self.clock(since),
            )
        })
    }

    /// [lanes] A lane session admitted a request that arrived at `since`: every waiter it passes
    /// over (one that arrived first) was passed over once more, as when a request of the loaded
    /// model takes the card before them.
    fn passed_over(&self, since: Instant) {
        let at = self.clock(since);
        let mut st = self.mu.lock().unwrap();
        for w in st.waiting.iter_mut() {
            if crate::core::lanes::passes_over(self.clock(w.since), at) {
                w.skips += 1;
            }
        }
    }
}

/// One chat request's place in the card queue: counted in `EngineSlot::waiting` from its arrival
/// until it holds the card, and removed on any early exit (load error, client gone).
struct WaitTicket<'a> {
    state: &'a AppState,
    idx: usize,
    counted: bool,
}

impl<'a> WaitTicket<'a> {
    fn enter(state: &'a AppState, idx: usize) -> Self {
        if let Some(w) = state.slot.lock().unwrap().waiting.get_mut(idx) {
            *w += 1;
        }
        state.publish_engine_event();
        Self {
            state,
            idx,
            counted: true,
        }
    }

    /// Leaves the queue under the caller's slot lock, so `running` and `waiting` move together.
    fn leave(&mut self, s: &mut EngineSlot) {
        if std::mem::take(&mut self.counted) {
            if let Some(w) = s.waiting.get_mut(self.idx) {
                *w = w.saturating_sub(1);
            }
        }
    }
}

impl Drop for WaitTicket<'_> {
    fn drop(&mut self) {
        if self.counted {
            let mut s = self.state.slot.lock().unwrap();
            self.leave(&mut s);
            drop(s);
            self.state.publish_engine_event();
        }
    }
}

/// The lane session of model `idx` (`--parallel`): takes the card's turn, loads the model and
/// hands the model's waiting requests to the engine's lanes until no lane is busy and nothing
/// waits. Started by the request that found no session running (`lanes::LaneHub::submit`). When
/// the oldest waiter for the card becomes due, the session lets its lanes drain, hands the turn
/// back and waits for its next one.
fn drive_lanes(state: Arc<AppState>, idx: usize) {
    let limits = || TurnLimits::read(state.cfg.queue_config.as_deref());
    loop {
        // queued at the arrival of the oldest request it serves, so a session that handed the
        // card back keeps its requests' place in line
        let since = state.lanes.oldest(idx).unwrap_or_else(Instant::now);
        let turn = state
            .turns
            .acquire_from(idx, since, || state.ready_model(), limits);
        if let Err(e) = state.ensure_loaded(idx, false) {
            drop(turn);
            state.lanes.fail_waiting(idx, &e);
            return;
        }
        let op = state.card_op.lock().unwrap();
        let engine = {
            let mut s = state.slot.lock().unwrap();
            match s.engine.take() {
                Some(e) if s.resident_idx == Some(idx) => {
                    // the session counts as one active request while it runs (no idle release)
                    s.active_requests += 1;
                    s.last_activity = Instant::now();
                    Some(e)
                }
                other => {
                    s.engine = other;
                    None
                }
            }
        };
        let Some(mut engine) = engine else {
            drop(op);
            drop(turn);
            state
                .lanes
                .fail_waiting(idx, "model unloaded concurrently; retry request");
            return;
        };
        state.publish_engine_event();
        let mut feed = state.lanes.feed(idx);
        let st = &state;
        let mut last_busy = 1u64;
        feed.on_lanes = Some(Box::new(move |busy| {
            let busy = busy.max(1);
            if busy != last_busy {
                last_busy = busy;
                {
                    let mut s = st.slot.lock().unwrap();
                    s.active_requests = busy;
                    s.last_activity = Instant::now();
                }
                st.publish_engine_event();
            }
        }));
        // the queue settings are read once per turn, like the turn queue's own
        let lim = limits();
        feed.blocks = Some(Box::new(move |since| st.turns.blocks(since, lim)));
        feed.on_admit = Some(Box::new(move |since| st.turns.passed_over(since)));
        let res = engine.serve_lanes(&mut feed);
        let (ended, yielded) = (feed.ended, feed.yielded);
        drop(feed);
        let prefix = engine.prefix_stats();
        let host_tier = engine.host_tier_facts();
        {
            let mut s = state.slot.lock().unwrap();
            s.engine = Some(engine);
            s.prefix = prefix;
            s.host_tier = host_tier;
            s.idle_due = state.host_cache.is_some();
            s.active_requests = 0;
            s.last_activity = Instant::now();
        }
        drop(op);
        drop(turn);
        if let Err(e) = &res {
            eprintln!(
                "[server] lane session of '{}' failed: {e}",
                state.models[idx].id
            );
        }
        state.publish_engine_event();
        if res.is_ok() && yielded {
            // the oldest waiter takes the card; this session's requests wait for its next turn
            continue;
        }
        if !ended {
            // the engine stopped before its feed ended the session: what still waits fails
            let why = res
                .err()
                .unwrap_or_else(|| "the lane session ended".to_string());
            state.lanes.fail_waiting(idx, &why);
            state.publish_engine_event();
        }
        return;
    }
}

fn handle_chat_control(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON");
        return;
    };
    let id = val.get("id").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(flag) = state.cancels.lock().unwrap().get(id) {
        flag.store(true, Ordering::Relaxed);
        write_json(stream, 200, "{\"ok\":true}");
    } else {
        write_json(stream, 200, "{\"ok\":false}");
    }
}

/// Which API a chat request came in on: the shape of its reply and of its errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Api {
    /// `/v1/chat/completions`.
    Chat,
    /// `/v1/messages`; thinking blocks carry their text unless `show_thinking` is off.
    Messages { show_thinking: bool },
}

impl Api {
    fn error(self, stream: &mut Conn, status: u16, err_type: &str, msg: &str) {
        match self {
            Api::Chat => write_error(stream, status, err_type, msg),
            Api::Messages { .. } => {
                // an engine that cannot take the request now is "overloaded" to these clients
                let status = if status == 503 { 529 } else { status };
                write_json(stream, status, &anthropic::error_json(status, msg));
            }
        }
    }

    /// An error after the event stream began (framed; the caller ends the stream).
    fn stream_error(self, status: u16, msg: &str) -> String {
        match self {
            Api::Chat => format!(
                "data: {{\"error\":{{\"message\":{},\"type\":\"server_error\"}}}}\n\ndata: [DONE]\n\n",
                jq(msg)
            ),
            Api::Messages { .. } => anthropic::sse("error", &anthropic::error_json(status, msg)),
        }
    }
}

/// The prompt of a chat request: rendered with the model's own chat template, else with the
/// built-in text template of its family (which has no tools).
fn prepare_prompt(
    a2ui_enabled: bool,
    m: &RegisteredModel,
    body: &JsonValue,
) -> Result<ChatPrompt, String> {
    // a request that declares A2UI renderer capabilities gets the instructions in its system turn
    let (with_a2ui, a2ui_note) = match a2ui::apply(a2ui_enabled, body)? {
        a2ui::Applied::Injected(b) => (Some(b), None),
        a2ui::Applied::Unchanged { note } => (None, note),
    };
    let body = with_a2ui.as_ref().unwrap_or(body);
    let mut prompt = render_prompt(m, body)?;
    prompt.ignored.extend(a2ui_note);
    Ok(prompt)
}

fn render_prompt(m: &RegisteredModel, body: &JsonValue) -> Result<ChatPrompt, String> {
    if let Some(t) = &m.chat {
        return chat::render_chat(t, body);
    }
    if chat::has_media_parts(body) {
        return Err(
            "this model has no chat template, and the built-in text template takes no images or videos".into(),
        );
    }
    let text = build_model_chat_prompt(m, body, m.spec.family == ModelFamily::Qwen35Dense)?;
    let reasoning_open = match (
        text.rfind(&m.format.think_open),
        text.rfind(&m.format.think_close),
    ) {
        (Some(open), Some(close)) => close < open,
        (Some(_), None) => true,
        _ => false,
    };
    let mut ignored = Vec::new();
    if body
        .get("tools")
        .and_then(JsonValue::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        ignored.push("tools (this model has no chat template)".to_string());
    }
    Ok(ChatPrompt {
        text,
        reasoning_open,
        tools: None,
        forced_prefix: String::new(),
        single_call: false,
        continue_final: false,
        ignored,
        media: Vec::new(),
    })
}

/// The token ids of a rendered prompt and its prepared images and videos. Each video is written
/// into the text as its timed temporal patches first; after tokenising, every image / video pad
/// is expanded to its tower input's token count. `with_pixels` false computes sizes only (a
/// token count without decoding pixels).
fn tokenize_prompt(
    m: &RegisteredModel,
    prompt: &ChatPrompt,
    with_pixels: bool,
) -> Result<(Vec<i32>, Option<RequestMedia>), String> {
    if prompt.media.is_empty() {
        return Ok((m.tokenizer.encode(&prompt.text, true), None));
    }
    let Some(serve) = &m.vision else {
        return Err(if m.spec.has_vision {
            "this model is served without its vision tower (text only)".to_string()
        } else {
            "this model takes no images or videos (text only)".to_string()
        });
    };
    let media = RequestMedia::prepare(&prompt.media, serve, with_pixels)?;
    let text = media.expand_text(&prompt.text)?;
    let ids = media.expand_ids(&m.tokenizer.encode(&text, true), &serve.spec)?;
    Ok((ids, Some(media)))
}

fn handle_chat_completions(state: &Arc<AppState>, req: &HttpRequest, stream: &mut Conn) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON body");
        return;
    };
    run_chat(state, &val, stream, Api::Chat, Vec::new());
}

/// `POST /v1/messages`: the request is translated into a chat request and runs on the chat path.
fn handle_messages(state: &Arc<AppState>, req: &HttpRequest, stream: &mut Conn) {
    let api = Api::Messages {
        show_thinking: true,
    };
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        api.error(stream, 400, "invalid_request_error", "invalid JSON body");
        return;
    };
    match anthropic::translate(&val) {
        Ok(t) => run_chat(
            state,
            &t.chat,
            stream,
            Api::Messages {
                show_thinking: t.show_thinking,
            },
            t.ignored,
        ),
        Err(e) => api.error(stream, 400, "invalid_request_error", &e),
    }
}

/// `POST /v1/messages/count_tokens`: the prompt size of a Messages request without running it.
fn handle_messages_count(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let api = Api::Messages {
        show_thinking: true,
    };
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        api.error(stream, 400, "invalid_request_error", "invalid JSON body");
        return;
    };
    let rendered = anthropic::translate(&val).and_then(|t| {
        let m = &state.models[state.find_model(t.chat.get("model").and_then(|v| v.as_str()))];
        let p = prepare_prompt(state.cfg.a2ui, m, &t.chat)?;
        tokenize_prompt(m, &p, false).map(|(ids, _)| ids.len())
    });
    match rendered {
        Ok(n) => write_json(stream, 200, &format!("{{\"input_tokens\":{n}}}")),
        Err(e) => api.error(stream, 400, "invalid_request_error", &e),
    }
}

/// `POST /apply-template`: the prompt text a chat request renders to.
fn handle_apply_template(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON body");
        return;
    };
    let m = &state.models[state.find_model(val.get("model").and_then(|v| v.as_str()))];
    match prepare_prompt(state.cfg.a2ui, m, &val) {
        Ok(p) => write_json(stream, 200, &format!("{{\"prompt\":{}}}", jq(&p.text))),
        Err(e) => write_error(stream, 400, "invalid_request_error", &e),
    }
}

/// Runs one chat request: prompt, card turn, generation, and the reply in the shape of `api`.
/// `ignored`: request fields the caller already found accepted but not applied.
fn run_chat(
    state: &Arc<AppState>,
    val: &JsonValue,
    stream: &mut Conn,
    api: Api,
    mut ignored: Vec<String>,
) {
    let t_arrival = Instant::now();
    let idx = state.find_model(val.get("model").and_then(|v| v.as_str()));
    let m = &state.models[idx];
    let prompt = match prepare_prompt(state.cfg.a2ui, m, val) {
        Ok(p) => p,
        Err(e) => {
            api.error(stream, 400, "invalid_request_error", &e);
            return;
        }
    };
    ignored.extend(prompt.ignored.iter().cloned());
    // Images and videos are decoded and prepared here, on the request's own thread, before the
    // request waits for the card.
    let (prompt_ids, media) = match tokenize_prompt(m, &prompt, true) {
        Ok(x) => x,
        Err(e) => {
            api.error(stream, 400, "invalid_request_error", &e);
            return;
        }
    };
    let media_counts: MediaCounts = media.as_ref().map(|md| md.counts()).unwrap_or_default();
    let media_host_ms = media.as_ref().map_or(0.0, |md| md.host_ms);
    let n_images = media_counts.images + media_counts.videos;
    let n_prompt = prompt_ids.len() as u64;
    let stream_mode = val.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    // `reasoning_format: "none"` leaves the reasoning in the content
    let split_reasoning = val.get("reasoning_format").and_then(|v| v.as_str()) != Some("none");
    let temperature = val
        .get("temperature")
        .and_then(|v| v.as_f64())
        .map_or(m.gen_defaults.temperature, |t| t as f32);
    let top_p = val
        .get("top_p")
        .and_then(|v| v.as_f64())
        .map_or(m.gen_defaults.top_p, |p| p as f32);
    let top_k = val
        .get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(m.gen_defaults.top_k);
    // Sampled drafts follow the verify's noise unless the request turns that off.
    let coupled = val.get("coupled").and_then(|v| v.as_bool()).unwrap_or(true);
    let draft_trunc = val
        .get("draft_trunc")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let min_p = val.get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let seed = val.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(42, |d| d.as_nanos() as u64)
    });
    let req_max = val
        .get("max_tokens")
        .or_else(|| val.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .filter(|&v| v > 0);
    // A request that names no limit may generate to the end of the model's context.
    let budget = match SequenceBudget::admit(n_prompt, req_max, m.max_ctx) {
        Ok(b) => b,
        Err(e) => {
            let msg = match api {
                Api::Chat => format!(
                    "prompt ({n_prompt} tokens) exceeds context budget {}: {e:?}",
                    m.max_ctx
                ),
                // the words these clients match to compact their context
                Api::Messages { .. } => format!(
                    "prompt is too long: {n_prompt} tokens > {} maximum",
                    crate::core::budget::advertised_served_ctx(m.max_ctx).saturating_sub(1)
                ),
            };
            api.error(stream, 400, "invalid_request_error", &msg);
            return;
        }
    };

    let stop_seqs: Vec<String> = match val.get("stop") {
        Some(JsonValue::Str(s)) if !s.is_empty() => vec![s.clone()],
        Some(JsonValue::Array(arr)) => arr
            .iter()
            .filter_map(|x| x.as_str().filter(|s| !s.is_empty()).map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };
    let is_greedy = temperature <= 0.0;

    // The method this request takes follows from the model's drafter and from greedy / sampled;
    // a request may name it (`gaudium_method`) but can not pick another one.
    let (m_greedy, m_sampled, m_loaded, _) =
        readout::method_set(state.run_facts(idx).drafter.as_ref());
    let method = if is_greedy { m_greedy } else { m_sampled };
    if let Some(want) = val
        .get("gaudium_method")
        .and_then(|v| v.as_str())
        .filter(|w| !w.is_empty() && *w != "default")
    {
        if want != method {
            api.error(
                stream,
                400,
                "invalid_request_error",
                &format!(
                    "method '{want}' is not available for this request: this engine runs {m_loaded} \
                     ({} requests: {method})",
                    if is_greedy { "greedy" } else { "sampled" }
                ),
            );
            return;
        }
    }

    let req_num = state.next_req_id.fetch_add(1, Ordering::Relaxed);
    let req_id = match api {
        Api::Chat => format!("chatcmpl-gaudium-{req_num}"),
        Api::Messages { .. } => format!("msg_gaudium_{req_num}"),
    };
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let max_new = budget.max_new().max(1);
    // the fields every `request` event of this job starts with
    let req_base = |phase: &str, queued_ms: f64| {
        format!(
            "\"type\":\"request\",\"job\":{req_num},\"phase\":{},\"background\":false,\
             \"method\":{},\"greedy\":{is_greedy},\"prompt_tokens\":{n_prompt},\"images\":{n_images},\
             \"max_new\":{max_new},\"queued_ms\":{queued_ms:.1}",
            jq(phase),
            jq(&method),
        )
    };
    state.publish_request(format!("{{{}}}", req_base("queued", 0.0)));
    // Counts this request in the card queue until it holds the card; any early return leaves it.
    let mut ticket = WaitTicket::enter(state, idx);
    // A model served with lanes: the request joins the model's lane session, which takes the
    // card's turn and loads the model (`drive_lanes`). Otherwise the request takes them itself.
    let lane_mode = m.lanes > 1;
    // One turn covers the model load and the generation, so no request for another model can swap the card
    // between them; a request for the model already on the card may go ahead of older ones (TurnQueue).
    let turn = (!lane_mode).then(|| {
        state.turns.acquire(
            idx,
            || state.ready_model(),
            || TurnLimits::read(state.cfg.queue_config.as_deref()),
        )
    });

    // Ensure target model is loaded on Gaudi 2 before starting generation.
    if let Err(e) = (!lane_mode)
        .then(|| state.ensure_loaded(idx, false))
        .unwrap_or(Ok(()))
    {
        state.publish_request(format!(
            "{{{},\"message\":{}}}",
            req_base("error", t_arrival.elapsed().as_secs_f64() * 1e3),
            jq(&e)
        ));
        api.error(stream, 503, "unavailable_error", &e);
        return;
    }

    let cancel_flag = Arc::new(AtomicBool::new(false));
    state
        .cancels
        .lock()
        .unwrap()
        .insert(req_id.clone(), Arc::clone(&cancel_flag));

    // Verified RequestGuard tracks request lifecycle transitions!
    let mut rguard = RequestGuard::new();

    if stream_mode && write_sse_headers(stream).is_err() {
        state.cancels.lock().unwrap().remove(&req_id);
        let _ = rguard.cancel();
        return;
    }

    let params = Qwen3GenParams {
        max_new,
        temperature,
        top_p,
        top_k,
        min_p,
        seed,
        coupled,
        draft_trunc,
        stop_on_eos: true,
        extra_eos_ids: m.extra_eos_ids.clone(),
        // `cache_prompt: false` asks for a whole prefill (the reference a resumed one equals)
        cache_prompt: val
            .get("cache_prompt")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        ..Qwen3GenParams::default()
    };

    // The engine takes the request: alone (the engine out of the slot for this request), or in
    // a lane of the model's lane session.
    let mut media = media;
    let mut lane_rx: Option<std::sync::mpsc::Receiver<LaneOut>> = None;
    let mut _op = None;
    let (mut engine, hbm_bytes) = if lane_mode {
        let (tx, rx) = std::sync::mpsc::channel();
        let job = LaneJob {
            id: req_num,
            prompt: prompt_ids.clone(),
            params: params.clone(),
            images: media.take().map(|md| md.into_tower_inputs()),
            cancel: Arc::clone(&cancel_flag),
            out: Box::new(move |ev| {
                let _ = tx.send(ev);
            }),
        };
        if state.lanes.submit(idx, job) {
            let st = Arc::clone(state);
            if let Err(e) = thread::Builder::new()
                .name("gaudium-lanes".into())
                .spawn(move || drive_lanes(st, idx))
            {
                state
                    .lanes
                    .fail_waiting(idx, &format!("start the lane session: {e}"));
            }
        }
        // the request waits for a lane; a session that cannot start ends it here
        let err = match rx.recv() {
            Ok(LaneOut::Started { .. }) => None,
            Ok(LaneOut::Done(d)) => Some(d.report.err().unwrap_or_default()),
            Ok(LaneOut::Token(_)) | Err(_) => Some("the lane session ended".to_string()),
        };
        if let Some(e) = err {
            state.cancels.lock().unwrap().remove(&req_id);
            let _ = rguard.apply(ReqEvent::EvReqFail);
            state.publish_request(format!(
                "{{{},\"message\":{}}}",
                req_base("error", t_arrival.elapsed().as_secs_f64() * 1e3),
                jq(&e)
            ));
            if stream_mode {
                let _ = write_sse_chunk(stream, api.stream_error(503, &e).as_bytes());
                write_sse_end(stream);
            } else {
                api.error(stream, 503, "unavailable_error", &e);
            }
            return;
        }
        let hbm = {
            let mut s = state.slot.lock().unwrap();
            ticket.leave(&mut s);
            s.last_activity = Instant::now();
            s.hbm_bytes
        };
        state.publish_engine_event();
        let _ = rguard.apply(ReqEvent::EvReqStartPrefill);
        lane_rx = Some(rx);
        (None, hbm)
    } else {
        // Acquire card op lock for exclusive Gaudi 2 execution.
        _op = Some(state.card_op.lock().unwrap());
        {
            let mut s = state.slot.lock().unwrap();
            s.active_requests += 1;
            ticket.leave(&mut s);
            s.last_activity = Instant::now();
        }
        state.publish_engine_event();
        let _ = rguard.apply(ReqEvent::EvReqStartPrefill);

        // Take engine temporarily out of slot so status endpoints never block while generating!
        let mut s = state.slot.lock().unwrap();
        match s.engine.take() {
            Some(e) if s.resident_idx == Some(idx) => (Some(e), s.hbm_bytes),
            other => {
                s.engine = other;
                s.active_requests = s.active_requests.saturating_sub(1);
                drop(s);
                state.cancels.lock().unwrap().remove(&req_id);
                let msg = "model unloaded concurrently; retry request";
                state.publish_request(format!(
                    "{{{},\"message\":{}}}",
                    req_base("error", t_arrival.elapsed().as_secs_f64() * 1e3),
                    jq(msg)
                ));
                if stream_mode {
                    let _ = write_sse_chunk(stream, api.stream_error(503, msg).as_bytes());
                    write_sse_end(stream);
                } else {
                    api.error(stream, 503, "unavailable_error", msg);
                }
                return;
            }
        }
    };
    // From here the engine has the request: everything before was the wait for the card (the
    // model load, or another request).
    let queued_ms = t_arrival.elapsed().as_secs_f64() * 1e3;
    state.publish_request(format!("{{{}}}", req_base("prefill", queued_ms)));

    let mut decoder = StreamDecoder::new(&m.tokenizer);
    // Generated text -> reasoning, content and tool calls (and the request's stop strings).
    let mut parser = OutputParser::new(
        prompt.reasoning_open,
        split_reasoning,
        prompt.tools.clone(),
        stop_seqs,
        m.format.clone(),
    );
    if prompt.single_call {
        parser.set_max_calls(1);
    }
    let format = &m.format;
    let mut started_decode = false;
    let tok_ref = Arc::clone(&m.tokenizer);
    let model_id = m.id.clone();
    let timings_per_token = val
        .get("timings_per_token")
        .and_then(|v| v.as_bool())
        .unwrap_or(stream_mode);
    let return_progress = val
        .get("return_progress")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // The engine's launch feed and profile block (both measured by the engine's own reporter).
    // (lanes: the engine's launch feed belongs to the whole session, not to one request)
    let live = engine
        .as_ref()
        .and_then(|e| e.live_profile())
        .filter(|h| h.is_live());
    let out_stream = if stream_mode {
        stream.try_clone().ok()
    } else {
        None
    };
    let stream_ok = !stream_mode || out_stream.is_some();
    let out = Arc::new(Mutex::new(StreamOut::new(
        out_stream,
        &req_id,
        &model_id,
        created,
        timings_per_token,
        return_progress,
        n_prompt,
        live.is_some(),
        Arc::clone(&cancel_flag),
    )));
    if !stream_ok {
        cancel_flag.store(true, Ordering::Relaxed);
    }
    {
        // The role chunk goes out now, so a client times its request from the moment the engine
        // took it.
        let mut o = out.lock().unwrap();
        let init_chunk = match api {
            Api::Chat => format!(
                "data: {{{},\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"\"}},\"finish_reason\":null}}]}}\n\n",
                o.chunk_head(),
            ),
            Api::Messages { show_thinking } => {
                o.anth = Some(anthropic::Builder::new(show_thinking));
                anthropic::Builder::message_start(&req_id, &model_id, n_prompt)
            }
        };
        o.write(&init_chunk);
        // A forced tool call's opening is already in the prompt: the parser reads it first.
        if !prompt.forced_prefix.is_empty() {
            o.add(parser.feed(&prompt.forced_prefix));
        }
    }
    if let Some(h) = &live {
        let sink_out = Arc::clone(&out);
        h.set_launch_sink(Some(Box::new(move |ev| {
            sink_out.lock().unwrap().on_event(ev);
        })));
    }

    let mut last_live_pub = Instant::now();
    let mut lane_done: Option<Box<LaneDone>> = None;
    let report = {
        let mut on_token = |tok_id: i32| {
            let piece = if tok_ref.is_special(tok_id) {
                // a special token is text only when it is part of the family's tags
                let text = String::from_utf8_lossy(tok_ref.token_bytes(tok_id)).into_owned();
                if format.is_marker_piece(&text) {
                    decoder.flush() + &text
                } else {
                    String::new()
                }
            } else {
                decoder.push(tok_id)
            };
            let delta = if piece.is_empty() {
                Delta::default()
            } else {
                parser.feed(&piece)
            };
            // a stop string, or the call limit: the generation ends here
            if parser.stopped() {
                cancel_flag.store(true, Ordering::Relaxed);
            }
            let (timings, seen) = {
                let mut o = out.lock().unwrap();
                o.push_token(delta);
                (o.timings(), o.seen_tokens())
            };
            if !started_decode {
                started_decode = true;
                let _ = rguard.apply(ReqEvent::EvReqPrefillDone);
                state.publish_request(format!(
                    "{{{},\"timings\":{}}}",
                    req_base("decode", queued_ms),
                    timings.json()
                ));
            }
            let now = Instant::now();
            if seen >= 3 && now.duration_since(last_live_pub).as_millis() >= 400 {
                last_live_pub = now;
                if let Some(h) = &live {
                    let rec = readout::live_record_json(
                        "decode",
                        &method,
                        h.request_rates().as_ref(),
                        hbm_bytes,
                        h.snapshot().as_deref(),
                    );
                    state.hub.publish(format!(
                    "{{\"type\":\"live\",\"job\":{req_num},\"source\":\"engine\",\"record\":{rec}}}"
                ));
                }
            }
        };
        match engine.as_mut() {
            Some(engine) => {
                // The request's tower inputs go to the engine first; it runs the tower inside
                // its prefill.
                let images_set = match media.take() {
                    Some(md) => engine.set_request_images(md.into_tower_inputs()),
                    None => Ok(()),
                };
                images_set.and_then(|()| {
                    engine.generate_cancellable(
                        &prompt_ids,
                        &params,
                        Some(&cancel_flag),
                        &mut on_token,
                    )
                })
            }
            None => {
                // a lane: tokens and the end come from the lane session
                let rx = lane_rx.take().expect("a lane request has its channel");
                loop {
                    match rx.recv() {
                        Ok(LaneOut::Token(t)) => on_token(t),
                        Ok(LaneOut::Started { .. }) => {}
                        Ok(LaneOut::Done(mut d)) => {
                            let r = std::mem::replace(&mut d.report, Err(String::new()));
                            lane_done = Some(d);
                            break r;
                        }
                        Err(_) => break Err("the lane session ended".to_string()),
                    }
                }
            }
        }
    };
    if let Some(h) = &live {
        h.set_launch_sink(None);
    }
    // the tower's own timings of this request (kept apart from the decoder's prefill)
    let (vision_report, prefix_stats, host_tier) = match (engine.as_ref(), lane_done.as_ref()) {
        (Some(e), _) => (
            e.last_vision_report().filter(|_| n_images > 0),
            e.prefix_stats(),
            e.host_tier_facts(),
        ),
        (None, Some(d)) => (
            d.vision.clone().filter(|_| n_images > 0),
            d.prefix,
            d.host_tier,
        ),
        (None, None) => (None, None, None),
    };

    if let Some(engine) = engine.take() {
        // Return engine cleanly to slot & decrement active counter!
        let mut s = state.slot.lock().unwrap();
        s.engine = Some(engine);
        s.prefix = prefix_stats;
        s.host_tier = host_tier;
        // once this connection is closed, the engine gets its idle time (`AppState::idle_work`)
        s.idle_due = state.host_cache.is_some();
        s.active_requests = s.active_requests.saturating_sub(1);
        s.last_activity = Instant::now();
    } else {
        let mut s = state.slot.lock().unwrap();
        if prefix_stats.is_some() {
            s.prefix = prefix_stats;
        }
        s.last_activity = Instant::now();
    }
    drop(turn);
    state.cancels.lock().unwrap().remove(&req_id);

    // Flush the decoder's and the parser's tail, after any output the last launch left unwritten
    let tail = decoder.flush();
    let mut tail_delta = parser.feed(&tail);
    tail_delta.append(parser.finish());
    let mut out = out.lock().unwrap();
    out.add(tail_delta);
    let pending = out.take_pending();
    let builder = out.anth.take();
    let first_token_ms = out.first_token_ms();

    match report {
        Ok(rep) => {
            // How the generation ended and what the client is told: the verified selection.
            let has_calls = !parser.tool_calls().is_empty();
            let stop_string = !parser.matched_stop().is_empty();
            let end = generation_end(
                stop_string,
                parser.stopped() && !stop_string,
                rep.stopped_by_eos,
                rep.stopped_by_cancel,
            );
            if end == GenEnd::GeCancelled {
                let _ = rguard.cancel();
            } else {
                let _ = rguard.apply(ReqEvent::EvReqFinishStop);
            }
            let finish_reason = finish_reason_name(end, has_calls);
            // the readout's phases are stop / length / cancelled
            let phase = if has_calls && end != GenEnd::GeCancelled {
                "stop"
            } else {
                finish_reason
            };
            let n_gen = rep.generated_tokens.len() as u64;
            let summary = live.as_ref().and_then(|h| h.request_summary());
            // Launches: the engine's report; an engine whose report carries none is taken at its
            // launch feed; with neither, the count is unknown and left out.
            let launches = if rep.launches > 0 || n_gen <= 1 {
                Some(rep.launches)
            } else {
                summary.as_ref().map(|s| s.launches)
            };
            let timings = Timings {
                prompt_ms: rep.ttft_ms,
                predicted_n: n_gen,
                predicted_ms: rep.decode_ms,
                launches,
                draft_n: rep.drafted,
                draft_n_accepted: rep.draft_accepted,
                // the prompt as reused + prefilled rows
                ..Timings::of_prompt(rep.prompt_tokens as u64, rep.reused_tokens)
            };
            let timings_json = timings.json();

            let profile = live.as_ref().and_then(|h| h.final_snapshot());
            let final_live = format!(
                "{{\"type\":\"live\",\"job\":{req_num},\"source\":\"engine\",\"record\":{}}}",
                readout::live_record_json(
                    "decode",
                    &method,
                    summary.as_ref(),
                    hbm_bytes,
                    profile.as_deref(),
                )
            );
            state.slot.lock().unwrap().last_live_json = Some(final_live.clone());
            state.hub.publish(final_live);

            let usage = format!(
                "{{\"prompt_tokens\":{},\"completion_tokens\":{n_gen},\"total_tokens\":{},\
                 \"prompt_tokens_details\":{{\"cached_tokens\":{}}}}}",
                rep.prompt_tokens,
                rep.prompt_tokens as u64 + n_gen,
                timings.cache_n,
            );
            let seed_json = if is_greedy {
                "null".to_string()
            } else {
                seed.to_string()
            };
            // The request's details. Times: `queued_ms` request received -> the engine took it
            // (model load, other requests); `reset_ms` the engine took it -> its prefill began
            // (per-request state reset); `prefill_ms` the engine's own prefill clock up to the
            // first token; `ttft_ms` the engine took it -> first token (reset + prefill).
            let mut details = format!(
                "{{\"method\":{},\"engine_method\":{},\"prefill_ms\":{:.2},\"decode_loop_ms\":{:.2},\
                 \"queued_ms\":{queued_ms:.2}",
                jq(&method),
                jq(&rep.method),
                rep.ttft_ms,
                rep.decode_ms,
            );
            if let Some(t) = first_token_ms {
                details.push_str(&format!(
                    ",\"ttft_ms\":{t:.2},\"reset_ms\":{:.2}",
                    (t - rep.ttft_ms).max(0.0)
                ));
            }
            if let Some(l) = launches {
                details.push_str(&format!(",\"steps\":{l}"));
                if l > 0 {
                    details.push_str(&format!(
                        ",\"ms_per_step\":{:.3},\"tok_per_step\":{:.3},\"decode_tok_s\":{:.2}",
                        rep.decode_ms / l as f64,
                        n_gen.saturating_sub(1) as f64 / l as f64,
                        rep.decode_tok_s,
                    ));
                }
            }
            if let Some(sum) = summary.as_ref().filter(|s| !s.log.is_empty()) {
                details.push_str(&format!(",\"aseq\":{}", jq(&readout::aseq(&sum.log))));
            }
            if let Some(sec) = readout::spec_section_json(&timings) {
                details.push_str(&format!(",\"spec\":{sec}"));
            }
            if has_calls {
                details.push_str(&format!(",\"tool_calls\":{}", parser.tool_calls().len()));
            }
            // prefix reuse: rows taken over from the previous request, and how many rows of
            // the prompt equal the one the engine held (a resume point lies below that)
            if prefix_stats.is_some_and(|p| p.slots > 0) {
                // `tier`: where the reused rows came from (device, host, or prefilled: none
                // were reused); `host`: what the host-RAM tier copied for this request
                details.push_str(&format!(
                    ",\"prefix\":{{\"hit\":{},\"reused_tokens\":{},\"common_tokens\":{},\"tier\":\"{}\"",
                    timings.cache_n > 0,
                    timings.cache_n,
                    rep.prefix_common,
                    readout::tier_name(rep.host_tier.as_ref(), timings.cache_n),
                ));
                if let Some(t) = rep.host_tier.as_ref() {
                    details.push_str(&format!(",\"host\":{}", readout::tier_json(t)));
                }
                details.push('}');
            }
            if n_images > 0 {
                details.push_str(&format!(
                    ",\"vision\":{}",
                    readout::vision_section_json(
                        &media_counts,
                        media_host_ms,
                        vision_report.as_ref(),
                        rep.ttft_ms,
                    )
                ));
            }
            // `gaudium_return_tokens`: the generated token ids (exactness checks between runs)
            if val.get("gaudium_return_tokens").and_then(|v| v.as_bool()) == Some(true) {
                details.push_str(&format!(",\"tokens\":{}", ids_json(&rep.generated_tokens)));
            }
            if !ignored.is_empty() {
                details.push_str(&format!(
                    ",\"ignored\":[{}]",
                    ignored.iter().map(|i| jq(i)).collect::<Vec<_>>().join(",")
                ));
            }
            details.push_str(&format!(
                ",\"sampling\":{{\"greedy\":{is_greedy},\"temperature\":{temperature:.3},\"top_k\":{top_k},\
                 \"top_p\":{top_p:.3},\"min_p\":{min_p:.3},\"seed\":{seed_json}}}}}"
            ));
            state.publish_request(format!(
                "{{{},\"timings\":{timings_json},\"details\":{details}}}",
                req_base(phase, queued_ms)
            ));

            if let Api::Messages { show_thinking } = api {
                let mut b = builder.unwrap_or_else(|| anthropic::Builder::new(show_thinking));
                let stop = stop_reason_name(end, has_calls);
                let usage = anthropic::usage_json_cached(
                    rep.prompt_tokens as u64,
                    rep.reused_tokens,
                    n_gen,
                );
                let extra = format!(",\"timings\":{timings_json},\"gaudium\":{details}");
                let events = b.feed(&pending);
                if stream_mode {
                    out.write(&events);
                    out.write(&b.finish(stop, parser.matched_stop(), &usage, &extra));
                    if !out.write_failed {
                        write_sse_end(stream);
                    }
                } else {
                    b.close();
                    let msg = b.message(
                        &req_id,
                        &model_id,
                        stop,
                        parser.matched_stop(),
                        &usage,
                        &extra,
                    );
                    write_json(stream, 200, &msg);
                }
            } else if stream_mode {
                let final_chunk = format!(
                    "data: {{{},\"choices\":[{{\"index\":0,\"delta\":{},\"finish_reason\":{}}}],\
                     \"usage\":{usage},\"timings\":{timings_json},\"gaudium\":{details}}}\n\ndata: [DONE]\n\n",
                    out.chunk_head(),
                    stream_out::delta_json(&pending),
                    jq(finish_reason),
                );
                out.write(&final_chunk);
                if !out.write_failed {
                    write_sse_end(stream);
                }
            } else {
                // a reply that is only tool calls has no content
                let content = if parser.content().is_empty() && has_calls {
                    "null".to_string()
                } else {
                    jq(parser.content())
                };
                let mut msg = format!("{{\"role\":\"assistant\",\"content\":{content}");
                if !parser.reasoning().is_empty() {
                    msg.push_str(&format!(
                        ",\"reasoning_content\":{}",
                        jq(parser.reasoning())
                    ));
                }
                if has_calls {
                    msg.push_str(&format!(
                        ",\"tool_calls\":{}",
                        chat::tool_calls_json(parser.tool_calls()).dump()
                    ));
                }
                msg.push('}');
                let resp = format!(
                    "{{\"id\":{},\"object\":\"chat.completion\",\"created\":{created},\"model\":{},\
                     \"choices\":[{{\"index\":0,\"message\":{msg},\"finish_reason\":{}}}],\
                     \"usage\":{usage},\"timings\":{timings_json},\"gaudium\":{details}}}",
                    jq(&req_id),
                    jq(&model_id),
                    jq(finish_reason),
                );
                write_json(stream, 200, &resp);
            }
        }
        Err(e) => {
            let _ = rguard.apply(ReqEvent::EvReqFail);
            state.publish_request(format!(
                "{{{},\"message\":{}}}",
                req_base("error", queued_ms),
                jq(&e)
            ));
            if stream_mode {
                out.write(&api.stream_error(500, &e));
                write_sse_end(stream);
            } else {
                api.error(stream, 500, "server_error", &e);
            }
        }
    }
}

// ============================================================================
// Static Gaudium UI File Server
// ============================================================================

fn resolve_static_rel(url_path: &str) -> &str {
    match url_path {
        "/" | "" | "/ui" | "/ui/" => "index.html",
        "/sw.js" | "/ui/sw-reset.js" => "sw-reset.js",
        other => other
            .strip_prefix("/ui/")
            .unwrap_or_else(|| other.trim_start_matches('/')),
    }
}

fn handle_static_ui(state: &AppState, url_path: &str, stream: &mut Conn, send_body: bool) {
    let Some(ui_root) = state.cfg.ui_dir.as_deref() else {
        write_error(
            stream,
            404,
            "not_found_error",
            "Web UI disabled (pass --ui-dir <path>)",
        );
        return;
    };

    let rel = resolve_static_rel(url_path);
    let Some(safe_path) = safe_join(ui_root, rel) else {
        write_error(stream, 400, "invalid_request_error", "invalid static path");
        return;
    };

    let target = if safe_path.is_dir() {
        safe_path.join("index.html")
    } else {
        safe_path
    };
    match fs::read(&target) {
        Ok(bytes) => {
            let mime = mime_type_for(&target);
            write_response_with_body(stream, 200, "OK", mime, &bytes, send_body);
        }
        Err(_) => write_error(
            stream,
            404,
            "not_found_error",
            &format!("asset not found: {url_path}"),
        ),
    }
}

fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    if rel.contains('\0') {
        return None;
    }
    let mut out = root.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(seg) => out.push(seg),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(out)
}

fn mime_type_for(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" | "webmanifest" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "txt" | "md" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

// ============================================================================
// Built-in Text Templates (models without a chat template)
// ============================================================================

/// Builds the chat prompt in the registered model's own template family.
fn build_model_chat_prompt(
    m: &RegisteredModel,
    body: &JsonValue,
    preopen_think: bool,
) -> Result<String, String> {
    if m.spec.family == ModelFamily::Gemma4 {
        build_gemma4_chat_prompt(
            body,
            requested_thinking(body, false),
            m.gemma_closes_thought,
        )
    } else {
        build_chat_prompt(body, preopen_think)
    }
}

/// `chat_template_kwargs.enable_thinking` of a request body, else the family's `default`.
fn requested_thinking(body: &JsonValue, default: bool) -> bool {
    body.get("chat_template_kwargs")
        .and_then(|v| v.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

/// Builds a Gemma 4 turn-template prompt from an OpenAI `/v1/chat/completions` JSON body, as the
/// model's chat template renders a text conversation without tools:
/// - a system turn first when the conversation starts with a system message or thinking is on;
///   with thinking it opens with `<|think|>`;
/// - one turn per further message (`assistant` is `model`);
/// - the generation prompt `<|turn>model\n`, followed by an empty closed thought channel for a
///   non-thinking request when the model's template writes one (`close_empty_thought`).
///
/// A thinking reply opens its thought channel itself; [`ChannelSplitter`] takes it apart.
pub fn build_gemma4_chat_prompt(
    body: &JsonValue,
    enable_thinking: bool,
    close_empty_thought: bool,
) -> Result<String, String> {
    let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) else {
        return Err("'messages' must be a non-empty array".into());
    };
    if msgs.is_empty() {
        return Err("'messages' must not be empty".into());
    }
    fn role_of(msg: &JsonValue) -> &str {
        msg.get("role").and_then(|v| v.as_str()).unwrap_or("user")
    }
    let first_is_system = matches!(role_of(&msgs[0]), "system" | "developer");
    let mut out = String::from("<bos>");
    if enable_thinking || first_is_system {
        out.push_str("<|turn>system\n");
        if enable_thinking {
            out.push_str("<|think|>\n");
        }
        if first_is_system {
            out.push_str(extract_message_text(msgs[0].get("content")).trim());
        }
        out.push_str("<turn|>\n");
    }
    for msg in &msgs[usize::from(first_is_system)..] {
        let role = match role_of(msg) {
            "assistant" => "model",
            other => other,
        };
        let content = extract_message_text(msg.get("content"));
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content.trim());
        out.push_str("<turn|>\n");
    }
    out.push_str("<|turn>model\n");
    if !enable_thinking && close_empty_thought {
        out.push_str("<|channel>thought\n<channel|>");
    }
    Ok(out)
}

/// The reasoning tags of a family's built-in text template (a model without a chat template).
fn builtin_format(family: ModelFamily) -> OutputFormat {
    let mut f = OutputFormat::default();
    if family == ModelFamily::Gemma4 {
        // Gemma 4 writes its reasoning as a channel between two special tokens
        f.think_open = "<|channel>thought\n".into();
        f.think_close = "<channel|>".into();
    }
    f
}

/// Builds a Qwen3 / Qwen3.5 ChatML prompt from an OpenAI `/v1/chat/completions` JSON body.
pub fn build_chat_prompt(body: &JsonValue, preopen_think: bool) -> Result<String, String> {
    let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) else {
        return Err("'messages' must be a non-empty array".into());
    };
    if msgs.is_empty() {
        return Err("'messages' must not be empty".into());
    }
    let enable_thinking = body
        .get("chat_template_kwargs")
        .and_then(|v| v.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let mut out = String::new();
    for msg in msgs {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = extract_message_text(msg.get("content"));
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    if !enable_thinking {
        out.push_str("<think>\n\n</think>\n\n");
    } else if preopen_think {
        out.push_str("<think>\n");
    }
    Ok(out)
}

fn extract_message_text(content: Option<&JsonValue>) -> String {
    match content {
        Some(JsonValue::Str(s)) => s.clone(),
        Some(JsonValue::Array(parts)) => {
            let mut joined = String::new();
            for p in parts {
                if p.get("type").and_then(|v| v.as_str()) == Some("text") {
                    if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                        joined.push_str(t);
                    }
                }
            }
            joined
        }
        _ => String::new(),
    }
}

/// Token ids as a JSON array.
fn ids_json(ids: &[i32]) -> String {
    let list: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
    format!("[{}]", list.join(","))
}

fn jq(s: &str) -> String {
    readout::jq(s)
}

/// Last component of a path, as a name (never a whole path) for descriptions.
fn dir_name(p: &Path) -> Option<String> {
    p.file_name().and_then(|n| n.to_str()).map(str::to_string)
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn parse_query(q: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in q.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(
            percent_decode(&k.replace('+', " ")),
            percent_decode(&v.replace('+', " ")),
        );
    }
    map
}

/// A model's own chat template, for the one-turn prompts of `gaudium --generate --chat` with a
/// model whose family has no built-in chat form.
pub struct OneTurnTemplate(chat::ChatTemplate);

impl OneTurnTemplate {
    /// The model's template; `None`: it has none.
    pub fn load(text: &ModelText) -> Result<Option<Self>, String> {
        Ok(chat::ChatTemplate::load_model(text)?.map(|(t, _)| Self(t)))
    }

    /// The prompt of one user turn (after an optional system turn), with reasoning on or off.
    pub fn render(&self, user: &str, system: Option<&str>, think: bool) -> Result<String, String> {
        let mut msgs = Vec::new();
        if let Some(sys) = system.filter(|s| !s.is_empty()) {
            msgs.push(format!("{{\"role\":\"system\",\"content\":{}}}", jq(sys)));
        }
        msgs.push(format!("{{\"role\":\"user\",\"content\":{}}}", jq(user)));
        let effort = if think { "" } else { ",\"reasoning_effort\":\"none\"" };
        let body = json::parse_json(&format!("{{\"messages\":[{}]{effort}}}", msgs.join(",")))
            .map_err(|e| e.to_string())?;
        chat::render_chat(&self.0, &body).map(|p| p.text)
    }
}

/// The prompt text a chat-completions request body renders to with the chat template of the
/// model at `model_dir`, a checkpoint directory or a GGUF file (`gaudium --apply-template`).
pub fn apply_chat_template(model_dir: &Path, body_json: &str) -> Result<String, String> {
    let Some((template, note)) = chat::ChatTemplate::load(model_dir)? else {
        return Err(format!("{} has no chat template", model_dir.display()));
    };
    if let Some(note) = note {
        eprintln!("[template] {note}");
    }
    let body = json::parse_json(body_json).map_err(|e| e.to_string())?;
    chat::render_chat(&template, &body).map(|p| p.text)
}

/// Parses `--model` spec: either `/path/to/ModelDir` (alias = basename) or `alias=/path/to/ModelDir`.
pub fn parse_model_arg(raw: &str) -> ModelSpecEntry {
    if let Some((alias, path)) = raw.split_once('=') {
        if !alias.is_empty() && !path.is_empty() {
            return ModelSpecEntry {
                id: alias.to_string(),
                dir: PathBuf::from(path),
            };
        }
    }
    let dir = PathBuf::from(raw);
    let id = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("gaudium")
        .to_string();
    ModelSpecEntry { id, dir }
}

/// Helper to keep unused import warning clean when tests build cleanly.
#[allow(dead_code)]
fn _keep_btree(_: BTreeMap<String, String>) {}

#[cfg(test)]
mod http_tests;

#[cfg(test)]
mod tests {
    use super::{next_turn, TurnLimits, TurnWaiter};
    use std::time::{Duration, Instant};

    #[test]
    fn one_turn_prompts_render_through_the_model_template() {
        let d = std::env::temp_dir().join(format!("gaudium-oneturn-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let text = crate::model_text::ModelText::open(&d).unwrap();
        assert!(super::OneTurnTemplate::load(&text).unwrap().is_none());
        std::fs::write(
            d.join("chat_template.jinja"),
            "{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}think={{ enable_thinking | default('unset') }}",
        )
        .unwrap();
        let t = super::OneTurnTemplate::load(&text).unwrap().unwrap();
        assert_eq!(t.render("Hi \"x\"", None, true).unwrap(), "[user:Hi \"x\"]think=unset");
        assert_eq!(
            t.render("Hi", Some("Be brief."), false).unwrap(),
            "[system:Be brief.][user:Hi]think=False"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    fn waiters(models: &[usize], now: Instant) -> Vec<TurnWaiter> {
        models
            .iter()
            .enumerate()
            .map(|(i, &model)| TurnWaiter {
                id: i as u64,
                model,
                since: now,
                skips: 0,
            })
            .collect()
    }

    #[test]
    fn turn_queue_runs_the_loaded_model_first_within_its_limits() {
        let now = Instant::now();
        let lim = TurnLimits::default();
        let mut w = waiters(&[1, 0, 0], now);
        assert_eq!(
            next_turn(&w, Some(0), now, lim),
            Some(1),
            "the loaded model goes ahead"
        );
        assert_eq!(
            next_turn(&w, Some(1), now, lim),
            Some(0),
            "the oldest already matches"
        );
        assert_eq!(
            next_turn(&w, None, now, lim),
            Some(0),
            "nothing loaded: arrival order"
        );
        assert_eq!(
            next_turn(&w, Some(2), now, lim),
            Some(0),
            "no waiter for the loaded model"
        );
        w[0].skips = lim.max_skips;
        assert_eq!(
            next_turn(&w, Some(0), now, lim),
            Some(0),
            "passed over too often"
        );
        let mut w = waiters(&[1, 0], now);
        w[0].since = now - lim.max_wait - Duration::from_secs(1);
        assert_eq!(next_turn(&w, Some(0), now, lim), Some(0), "waited too long");
        w[0].since = now - lim.max_wait + Duration::from_secs(60);
        assert_eq!(
            next_turn(&w, Some(0), now, lim),
            Some(1),
            "inside the wait limit"
        );
        assert_eq!(next_turn(&[], Some(0), now, lim), None);
    }

    #[test]
    fn lane_admissions_pass_over_older_waiters_until_one_is_due() {
        let q = TurnQueue::default();
        let lim = TurnLimits::default();
        let t0 = Instant::now();
        let later = |ms: u64| t0 + Duration::from_millis(ms);
        q.mu.lock().unwrap().waiting = waiters(&[1], t0);
        assert!(!q.blocks(later(5), lim), "a fresh waiter blocks nothing");
        // a request that arrived before the waiter does not pass it over
        q.passed_over(q.epoch);
        assert_eq!(q.mu.lock().unwrap().waiting[0].skips, 0);
        for k in 1..=lim.max_skips {
            q.passed_over(later(u64::from(k)));
        }
        assert_eq!(q.mu.lock().unwrap().waiting[0].skips, lim.max_skips);
        assert!(q.blocks(later(10), lim), "due: a younger request may not pass it over");
        assert!(
            !q.blocks(t0, lim),
            "a request that arrived with it (or before) is not blocked by it"
        );
        q.mu.lock().unwrap().waiting[0] = TurnWaiter {
            id: 0,
            model: 1,
            since: t0,
            skips: 0,
        };
        let mut st = q.mu.lock().unwrap();
        st.waiting[0].since = Instant::now() - lim.max_wait;
        let w0 = st.waiting[0].since;
        drop(st);
        assert!(q.blocks(Instant::now(), lim), "waited as long as the limit allows");
        assert!(!q.blocks(w0, lim));
        q.mu.lock().unwrap().waiting.clear();
        assert!(!q.blocks(later(10), lim), "nobody waits");
    }

    #[test]
    fn a_due_waiter_behind_one_that_is_not_due_is_never_passed_over() {
        let now = Instant::now();
        let lim = TurnLimits::default();
        // a lane session queued again (model 0, no skips yet) stands before a waiter that is due
        let mut w = waiters(&[0, 1], now);
        w[1].skips = lim.max_skips;
        assert_eq!(next_turn(&w, Some(0), now, lim), Some(1), "the due waiter goes first");
        assert_eq!(next_turn(&w, None, now, lim), Some(1));
        w[1].skips = 0;
        assert_eq!(next_turn(&w, Some(1), now, lim), Some(1), "then the loaded model");
        assert_eq!(next_turn(&w, Some(2), now, lim), Some(0), "then arrival order");
        let q = TurnQueue::default();
        let t0 = Instant::now();
        {
            let mut st = q.mu.lock().unwrap();
            st.waiting = waiters(&[0, 1], t0);
            st.waiting[1].since = t0 + Duration::from_millis(5);
            st.waiting[1].skips = lim.max_skips;
        }
        assert!(q.blocks(t0 + Duration::from_millis(9), lim), "a due waiter second in line blocks");
        assert!(!q.blocks(t0 + Duration::from_millis(5), lim), "not a request as old as it");
    }

    #[test]
    fn a_lane_waiter_queued_from_an_earlier_arrival_takes_its_place_in_line() {
        let q = TurnQueue::default();
        let t0 = Instant::now();
        {
            let mut st = q.mu.lock().unwrap();
            st.waiting = waiters(&[1, 2], t0 + Duration::from_millis(10));
            st.waiting[1].since = t0 + Duration::from_millis(30);
            st.next_id = 2;
        }
        let lim = TurnLimits::default();
        // model 0 is loaded, so its waiter goes ahead; it stood between the two, so it passes
        // over only the one that arrived before it
        let turn = q.acquire_from(0, t0 + Duration::from_millis(20), || Some(0), || lim);
        {
            let st = q.mu.lock().unwrap();
            assert_eq!(st.waiting.iter().map(|w| w.model).collect::<Vec<_>>(), vec![1, 2]);
            assert_eq!(st.waiting.iter().map(|w| w.skips).collect::<Vec<_>>(), vec![1, 0]);
        }
        drop(turn);
    }

    #[test]
    fn turn_limits_read_the_queue_settings_file() {
        let d = TurnLimits::default();
        assert_eq!(d.max_wait, Duration::from_secs(45 * 60));
        assert_eq!(d.max_skips, 3);
        let l = TurnLimits::parse(
            "# card queue\ngroup_max_wait_min = 10  # minutes\ngroup_max_skips=5\n",
        );
        assert_eq!(
            l,
            TurnLimits {
                max_wait: Duration::from_secs(600),
                max_skips: 5
            }
        );
        assert_eq!(
            TurnLimits::parse("group_max_wait_min=abc\nother=1\n"),
            d,
            "bad values keep defaults"
        );
        assert_eq!(TurnLimits::read(None), d);
    }

    use super::*;

    #[test]
    fn sampler_defaults_come_from_generation_config() {
        let d = SamplerDefaults::from_generation_config(
            r#"{"do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}"#,
        );
        assert_eq!(
            d,
            SamplerDefaults {
                temperature: 1.0,
                top_k: 20,
                top_p: 0.95
            }
        );
        // Missing or unusable keys keep the fallback values.
        let d = SamplerDefaults::from_generation_config(r#"{"top_p": 7.0}"#);
        assert_eq!(d, SamplerDefaults::FALLBACK);
        assert_eq!(
            SamplerDefaults::from_generation_config("not json"),
            SamplerDefaults::FALLBACK
        );
    }

    #[test]
    fn build_chat_prompt_formats_multiturn_and_no_think() {
        let body = json::parse_json(
            r#"{
              "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "Hi!"}]}
              ],
              "chat_template_kwargs": {"enable_thinking": false}
            }"#,
        )
        .unwrap();
        let prompt = build_chat_prompt(&body, true).unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn gemma4_prompt_follows_the_template_with_and_without_thinking() {
        let body = json::parse_json(
            r#"{"messages": [
                {"role": "system", "content": " Be brief. "},
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": [{"type": "text", "text": "Again?"}]}
            ]}"#,
        )
        .unwrap();
        assert_eq!(
            build_gemma4_chat_prompt(&body, false, true).unwrap(),
            "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi!<turn|>\n<|turn>model\nHello.<turn|>\n\
             <|turn>user\nAgain?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
        // thinking: the system turn opens with the think token and no channel is pre-closed
        assert_eq!(
            build_gemma4_chat_prompt(&body, true, true).unwrap(),
            "<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n<|turn>user\nHi!<turn|>\n<|turn>model\nHello.<turn|>\n\
             <|turn>user\nAgain?<turn|>\n<|turn>model\n"
        );
        let plain =
            json::parse_json(r#"{"messages": [{"role": "user", "content": "Hi"}]}"#).unwrap();
        assert_eq!(
            build_gemma4_chat_prompt(&plain, true, false).unwrap(),
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        );
        assert_eq!(
            build_gemma4_chat_prompt(&plain, false, false).unwrap(),
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\n"
        );
        assert!(!requested_thinking(&plain, false) && requested_thinking(&plain, true));
        let off =
            json::parse_json(r#"{"chat_template_kwargs": {"enable_thinking": false}}"#).unwrap();
        assert!(!requested_thinking(&off, true));
    }

    #[test]
    fn builtin_formats_and_api_error_shapes() {
        assert_eq!(
            builtin_format(ModelFamily::Qwen35Dense),
            OutputFormat::default()
        );
        let g = builtin_format(ModelFamily::Gemma4);
        assert_eq!(
            (g.think_open.as_str(), g.think_close.as_str()),
            ("<|channel>thought\n", "<channel|>")
        );
        assert!(Api::Chat
            .stream_error(500, "x")
            .ends_with("data: [DONE]\n\n"));
        let e = Api::Messages {
            show_thinking: true,
        }
        .stream_error(503, "busy");
        assert!(e.starts_with("event: error\ndata: {\"type\":\"error\""));
    }

    #[test]
    fn safe_join_blocks_path_traversal() {
        let root = Path::new("/srv/ui");
        assert_eq!(resolve_static_rel("/"), "index.html");
        assert_eq!(resolve_static_rel("/ui/"), "index.html");
        assert_eq!(resolve_static_rel("/sw.js"), "sw-reset.js");
        assert_eq!(resolve_static_rel("/ui/js/main.js"), "js/main.js");
        assert_eq!(
            resolve_static_rel("/ui/vendor/preact/preact.module.js"),
            "vendor/preact/preact.module.js"
        );
        assert_eq!(
            safe_join(root, resolve_static_rel("/ui/js/app.js")),
            Some(PathBuf::from("/srv/ui/js/app.js"))
        );
        assert_eq!(safe_join(root, "../etc/passwd"), None);
        assert_eq!(safe_join(root, "js/../../secret"), None);
        assert_eq!(safe_join(root, "/etc/passwd"), None);
    }

    #[test]
    fn effort_levels_override_only_levels_the_template_renders() {
        let template = || ChatTemplate::from_parts(chat::EFFORT_TEMPLATE, None).unwrap().0;
        let levels = |l: &str| EffortLevels::parse(l).unwrap();
        let mut t = template();
        effort_override(Some(&mut t), levels("low,xhigh")).unwrap();
        assert_eq!(t.effort.advertised(), ["low", "xhigh"]);
        effort_override(Some(&mut t), levels("none")).unwrap();
        assert!(!t.effort.supported());
        // the template raises on high: refused, the detected levels stay
        let mut t = template();
        assert!(effort_override(Some(&mut t), levels("low,high"))
            .unwrap_err()
            .contains("'high'"));
        assert_eq!(t.effort.advertised(), ["low", "medium", "xhigh"]);
        // no chat template: no levels to give
        assert!(effort_override(None, levels("low")).is_err());
        assert!(effort_override(None, levels("none")).is_ok());
        assert_eq!(template_family(ModelFamily::Gemma4Assistant), "gemma4");
    }
}
