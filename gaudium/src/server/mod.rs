//! Zero-dependency OpenAI-compatible HTTP/SSE + Gaudium Web UI server.
//!
//! Serves:
//! - Gaudium Web UI static assets (`GET /`, `/sw.js`, `/css/*`, `/js/*`, `/img/*`, `/vendor/*`)
//! - OpenAI-compatible chat & model endpoints (`/v1/models`, `/v1/chat/completions`, `/props`,
//!   `/health`, `/tokenize`, `/detokenize`, `/slots`, `/models/load`, `/models/unload`)
//! - Gaudium UI telemetry & card-lifecycle endpoints (`/gaudium/ui`, `/gaudium/engine`,
//!   `/gaudium/live`, `/gaudium/events`, `/gaudium/count_tokens`, `/v1/chat/completions/control`)
//! - A2UI interfaces as a per-request option (`--a2ui`, the request field `gaudium_a2ui`,
//!   `/gaudium/a2ui`; see `a2ui.rs`)
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
use crate::core::lifecycle::{can_switch_resident_model, RequestGuard};
use crate::core::tool_calls::{finish_reason_name, generation_end, stop_reason_name, GenEnd};
use crate::engine::qwen3::{Qwen3Config, Qwen3GenParams, KV_CHUNK};
use crate::engine::{load_engine, GenEngine};
use crate::json::{self, JsonValue};
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::tokenizer::{StreamDecoder, Tokenizer};
use crate::verified::{BackendState, ReqEvent};

mod a2ui;
mod anthropic;
mod chat;
mod jinja;
mod outparse;
mod readout;
mod stream_out;

use chat::{ChatPrompt, ChatTemplate};
use outparse::{Delta, OutputFormat, OutputParser};
use readout::{DrafterFacts, RunFacts, Timings};
use stream_out::StreamOut;

const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;

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
    pub verbose: bool,
    /// Per-model options keyed by served model id.
    pub per_model: Vec<(String, ModelRunOptions)>,
    /// Draft widths for models with a drafter (one value = fixed, several = adaptive).
    pub draft_k: Vec<u64>,
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
    /// Data types of a model's step graph by `(model, fp8 weights, routed experts)`.
    dtypes: Mutex<HashMap<(usize, bool, bool), Option<crate::engine::qwen3_traffic::StepDtypes>>>,
    /// Chat requests take turns on the card through this queue (model load + generation in one turn).
    turns: TurnQueue,
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
            vision: false,
            hbm_bytes: None,
        }
    }

    /// Data types of model `idx`'s step graph under `facts` (built once per precision / form).
    fn step_dtypes(
        &self,
        idx: usize,
        facts: &RunFacts,
    ) -> Option<crate::engine::qwen3_traffic::StepDtypes> {
        let key = (idx, facts.fp8_weights, facts.moe_routed);
        if let Some(v) = self.dtypes.lock().unwrap().get(&key) {
            return v.clone();
        }
        let mut spec = self.models[idx].spec.clone();
        spec.fp8_weights = facts.fp8_weights;
        spec.moe_routed = facts.moe_routed;
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
            "{{\"type\":\"hello\",\"backend\":\"engine\",\"model\":{},\"n_ctx\":{},\"vision\":false,\
             \"speculative\":{},\"precision\":{},\
             \"methods\":{{\"greedy\":{},\"sampled\":{},\"loaded\":{},\"greedy_only\":[{}]}},\
             \"engine_info\":{info},\"model_desc\":{desc},\
             \"build\":{},\"supports_inject\":false,\"mock_card\":false,\
             \"state\":{},\"ui\":{{\"root\":\"gaudium\",\"gaudium\":\"/\",\"llama\":\"/\"}},\
             \"api\":{{\"events\":\"/gaudium/events\",\"live\":\"/gaudium/live\",\"engine\":\"/gaudium/engine\",\
             \"count_tokens\":\"/gaudium/count_tokens\",\"version\":1}}}}",
            jq(&m.id),
            m.max_ctx,
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
                "{{\"id\":{},\"state\":{},\"resident\":{},\"running\":{},\"waiting\":{},\"n_ctx\":{},\"vision\":false,\"speculative\":{},\"precision\":{},\"greedy\":{},\"sampled\":{}}}",
                jq(&m.id),
                jq(mstate),
                is_res,
                if is_res { active_reqs } else { 0 },
                s.waiting.get(i).copied().unwrap_or(0),
                m.max_ctx,
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

    /// The one slot of this server: whether it is busy, and what the resident model runs as.
    fn slots_json(&self) -> String {
        let (active, idx) = {
            let s = self.slot.lock().unwrap();
            (s.active_requests, s.resident_idx.unwrap_or(0))
        };
        let m = &self.models[idx];
        let facts = self.run_facts(idx);
        let (greedy, sampled, _, _) = readout::method_set(facts.drafter.as_ref());
        format!(
            "[{{\"id\":0,\"is_processing\":{},\"model\":{},\"n_ctx\":{},\"speculative\":{},\
             \"precision\":{},\"greedy\":{},\"sampled\":{},\"loaded\":{}}}]",
            active > 0,
            jq(&m.id),
            m.max_ctx,
            facts.drafter.is_some(),
            jq(readout::precision_word(&facts)),
            jq(&greedy),
            jq(&sampled),
            facts.from_engine,
        )
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

        let n_ctx_train = crate::engine::model_max_positions(&m.dir)
            .map_or("null".to_string(), |n| n.to_string());
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
        let mut out = format!(
            "{{\"default_generation_settings\":{{\"n_ctx\":{},\"params\":{{\"seed\":-1,\"temperature\":{d_temp},\
             \"top_k\":{d_top_k},\"top_p\":{d_top_p},\"min_p\":0.0,\"max_tokens\":-1,\"n_predict\":-1,\"stream\":false,\
             \"timings_per_token\":false,\"reasoning_format\":\"auto\",\"speculative\":{speculative},\
             \"samplers\":[\"top_k\",\"top_p\",\"min_p\",\"temperature\"]}}}},\
             \"total_slots\":1,\"model_alias\":{},\"model_path\":{},\
             \"modalities\":{{\"vision\":false,\"audio\":false,\"video\":false}},\
             \"endpoint_slots\":true,\"endpoint_props\":false,\"endpoint_metrics\":false,\"ui\":{},\
             \"ui_settings\":{{}},\"chat_template\":\"qwen3\",\
             \"chat_template_caps\":{{\"supports_string_content\":true,\"supports_typed_content\":true,\
             \"supports_tools\":{tools},\"supports_tool_calls\":{tools},\"supports_parallel_tool_calls\":{tools},\
             \"supports_system_role\":true,\"supports_preserve_reasoning\":true,\"supports_reasoning_effort\":false,\
             \"supports_object_arguments\":false,\"reasoning_effort_levels\":[]}},\
             \"bos_token\":{},\"eos_token\":{},\"build_info\":{},\"is_sleeping\":false,\
             \"cors_proxy_enabled\":false,\"a2ui\":{}",
            m.max_ctx,
            jq(&m.id),
            jq(&m.dir.display().to_string()),
            self.cfg.ui_dir.is_some(),
            jq(m.chat.as_ref().map_or("", |t| t.token("bos_token"))),
            jq(m.chat.as_ref().map_or("<|im_end|>", |t| t.token("eos_token"))),
            jq(&format!("gaudium-rust {}", env!("CARGO_PKG_VERSION"))),
            a2ui::capabilities_json(self.cfg.a2ui, false),
        );
        // what this model is and how it runs (the same description the event stream carries)
        out.push_str(&format!(
            ",\"model_desc\":{}",
            self.model_desc_with(idx, &self.run_facts(idx))
        ));
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
                s.engine = None; // drops Qwen3Engine -> drains streams & releases /dev/accel/accel0
                s.hbm_bytes = None;
                s.facts = None;
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
        let (qcfg, drafter_dir) = engine_config_of(&self.cfg, &m.id, &m.dir, m.max_ctx);

        match load_engine(&qcfg) {
            Ok(eng) => {
                let dt = t0.elapsed().as_secs_f64();
                let hbm = eng.hbm_bytes();
                // what the readout says about this model comes from the engine that loaded it
                let facts = RunFacts {
                    from_engine: true,
                    fp8_weights: eng.spec().fp8_weights,
                    moe_routed: eng.spec().moe_routed,
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
                drop(s);
                self.publish_engine_event();
                if self.cfg.verbose {
                    eprintln!("[server] model '{}' ready on Gaudi 2 in {:.2}s", m.id, dt);
                }
                Ok(())
            }
            Err(e) => {
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
            s.engine = None; // drains streams & releases device + unlinks cardlock holder/live/shm
            s.hbm_bytes = None;
            s.facts = None;
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
        return Some((None, "mtp", vec![if opts.moe_routed { 3 } else { 4 }]));
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

/// The context the served model `id` runs at: `--max-ctx` (or the model's own `--model-ctx`) in
/// whole KV chunks, clamped to the model's own limit.
fn served_max_ctx(cfg: &ServerConfig, id: &str, dir: &Path) -> u64 {
    let max_ctx = crate::verified::align_up(cfg.max_ctx.max(KV_CHUNK), KV_CHUNK);
    crate::engine::clamp_ctx_to_model(
        cfg.per_model
            .iter()
            .find(|(k, _)| k == id)
            .and_then(|(_, o)| o.max_ctx)
            .map_or(max_ctx, |c| crate::verified::align_up(c.max(KV_CHUNK), KV_CHUNK)),
        dir,
    )
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
    if let Some((_, opts)) = cfg.per_model.iter().find(|(k, _)| k == id) {
        qcfg.fp8_act = opts.fp8_act.clone();
        qcfg.fp8_cache = opts.fp8_cache.clone();
        qcfg.moe_routed = opts.moe_routed;
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
    if let Some(id) = only.iter().find(|o| !cfg.models.iter().any(|m| &m.id == *o)) {
        return Err(format!("--build-cache-only '{id}': no --model with that id"));
    }
    let mut failed: Vec<String> = Vec::new();
    for entry in cfg.models.iter().filter(|m| only.is_empty() || only.contains(&m.id)) {
        let max_ctx = served_max_ctx(cfg, &entry.id, &entry.dir);
        let (qcfg, _) = engine_config_of(cfg, &entry.id, &entry.dir, max_ctx);
        eprintln!("[server] build-cache: model '{}' at context {max_ctx}", entry.id);
        let family = ModelSpec::from_config_file(entry.dir.join("config.json"))
            .map(|s| s.family)
            .map_err(|e| format!("load config.json for '{}': {e}", entry.id))?;
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

pub fn run_server(cfg: ServerConfig) -> Result<(), String> {
    if cfg.models.is_empty() {
        return Err("at least one --model <dir> is required for --serve".into());
    }
    // Shared stop-request handler; re-asserted by the device layer after every card open, because
    // the runtime replaces signal dispositions while initializing.
    crate::signals::install();
    // The readout reads each engine's launch feed and profile block in-process, whether or not
    // a live-record directory is configured.
    crate::live_report::enable_in_memory();

    let mut models = Vec::with_capacity(cfg.models.len());
    for entry in &cfg.models {
        let spec = ModelSpec::from_config_file(entry.dir.join("config.json"))
            .map_err(|e| format!("load config.json for '{}': {e}", entry.id))?;
        spec.validate()
            .map_err(|e| format!("invalid ModelSpec for '{}': {e}", entry.id))?;
        let tok = Tokenizer::from_model_dir(&entry.dir)
            .map_err(|e| format!("load tokenizer.json for '{}': {e}", entry.id))?;
        let mut extra_eos_ids = Vec::new();
        // `<|tool_response>`: where a Gemma 4 reply hands over to the tool results of its calls
        for special in ["<|im_end|>", "<|endoftext|>", "<turn|>", "<|tool_response>"] {
            if let Some(id) = tok.token_id(special) {
                extra_eos_ids.push(id);
            }
        }
        let chat = match ChatTemplate::load(&entry.dir) {
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
        let format = match &chat {
            Some(t) => t.format.clone(),
            None => builtin_format(spec.family),
        };
        models.push(RegisteredModel {
            id: entry.id.clone(),
            dir: entry.dir.clone(),
            spec,
            chat,
            format,
            tokenizer: Arc::new(tok),
            extra_eos_ids,
            // `--max-ctx` (or the model's own `--model-ctx`), clamped to the model's own limit.
            max_ctx: served_max_ctx(&cfg, &entry.id, &entry.dir),
            gen_defaults: std::fs::read_to_string(entry.dir.join("generation_config.json"))
                .map(|t| SamplerDefaults::from_generation_config(&t))
                .unwrap_or(SamplerDefaults::FALLBACK),
            gemma_closes_thought: std::fs::read_to_string(entry.dir.join("chat_template.jinja"))
                .map(|t| crate::tokenizer::gemma4_template_closes_thought(&t))
                .unwrap_or(false),
        });
    }

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
        }),
        card_op: Mutex::new(()),
        cancels: Mutex::new(HashMap::new()),
        next_req_id: AtomicU64::new(1),
        hub: EventHub::new(),
        dtypes: Mutex::new(HashMap::new()),
        turns: TurnQueue::default(),
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

    eprintln!(
        "[server] listening on http://{}/ (models: {}, lazy={}, idle_park={}s, ui={})",
        addr,
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
                thread::sleep(Duration::from_millis(25));
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

fn read_http_request(stream: &TcpStream) -> Result<Option<HttpRequest>, String> {
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
    stream: &mut TcpStream,
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
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) {
    write_response_with_body(stream, status, reason, content_type, body, true);
}

fn write_json(stream: &mut TcpStream, status: u16, json: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
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

fn write_error(stream: &mut TcpStream, status: u16, err_type: &str, msg: &str) {
    let body = format!(
        "{{\"error\":{{\"message\":{},\"type\":{},\"code\":{}}}}}",
        jq(msg),
        jq(err_type),
        status
    );
    write_json(stream, status, &body);
}

fn write_sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
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

fn write_sse_chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let prefix = format!("{:x}\r\n", data.len());
    stream.write_all(prefix.as_bytes())?;
    stream.write_all(data)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn write_sse_end(stream: &mut TcpStream) {
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

fn handle_connection(state: Arc<AppState>, mut stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let req = match read_http_request(&stream) {
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

fn handle_events_sse(state: &AppState, stream: &mut TcpStream) {
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

fn handle_engine_action(state: &Arc<AppState>, req: &HttpRequest, stream: &mut TcpStream) {
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

fn handle_model_load(state: &AppState, req: &HttpRequest, stream: &mut TcpStream, load: bool) {
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

fn handle_tokenize(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
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

fn handle_detokenize(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
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

fn handle_count_tokens(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
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
    let ids = m.tokenizer.encode(&prompt.text, true);
    let n_prompt = ids.len() as u64;
    let req_max = val.get("max_tokens").and_then(|v| v.as_u64());
    let (fits, max_tokens) = match SequenceBudget::admit(n_prompt, req_max, m.max_ctx) {
        Ok(b) => (true, b.max_new()),
        Err(_) => (false, 0),
    };
    write_json(
        stream,
        200,
        &format!(
            "{{\"n_prompt_tokens\":{n_prompt},\"n_ctx\":{},\"max_tokens\":{max_tokens},\"fits\":{fits}}}",
            m.max_ctx
        ),
    );
}

/// How far a request for the loaded model may jump the queue: the oldest request is never passed over once
/// it has waited `max_wait` or been passed over `max_skips` times. Read from the card-queue settings file
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
/// at startup, and per-request settings (sampling, method) never reload the card. The oldest request is
/// never passed over beyond the `TurnLimits` of the settings file.
#[derive(Default)]
struct TurnQueue {
    mu: Mutex<TurnState>,
    cv: Condvar,
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

/// The waiter that gets the card next: its position in `waiting` (arrival order).
fn next_turn(
    waiting: &[TurnWaiter],
    ready: Option<usize>,
    now: Instant,
    lim: TurnLimits,
) -> Option<usize> {
    let first = waiting.first()?;
    let due = now.duration_since(first.since) >= lim.max_wait || first.skips >= lim.max_skips;
    match ready {
        Some(m) if first.model != m && !due => {
            Some(waiting.iter().position(|w| w.model == m).unwrap_or(0))
        }
        _ => Some(0),
    }
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
        let mut st = self.mu.lock().unwrap();
        let id = st.next_id;
        st.next_id += 1;
        st.waiting.push(TurnWaiter {
            id,
            model,
            since: Instant::now(),
            skips: 0,
        });
        loop {
            if !st.busy {
                if let Some(pos) = next_turn(&st.waiting, ready(), Instant::now(), limits()) {
                    if st.waiting[pos].id == id {
                        for w in &mut st.waiting[..pos] {
                            w.skips += 1;
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

fn handle_chat_control(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
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
    fn error(self, stream: &mut TcpStream, status: u16, err_type: &str, msg: &str) {
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
    })
}

fn handle_chat_completions(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON body");
        return;
    };
    run_chat(state, &val, stream, Api::Chat, Vec::new());
}

/// `POST /v1/messages`: the request is translated into a chat request and runs on the chat path.
fn handle_messages(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
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
fn handle_messages_count(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
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
        prepare_prompt(state.cfg.a2ui, m, &t.chat).map(|p| m.tokenizer.encode(&p.text, true).len())
    });
    match rendered {
        Ok(n) => write_json(stream, 200, &format!("{{\"input_tokens\":{n}}}")),
        Err(e) => api.error(stream, 400, "invalid_request_error", &e),
    }
}

/// `POST /apply-template`: the prompt text a chat request renders to.
fn handle_apply_template(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
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
    state: &AppState,
    val: &JsonValue,
    stream: &mut TcpStream,
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
    let prompt_ids = m.tokenizer.encode(&prompt.text, true);
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
             \"method\":{},\"greedy\":{is_greedy},\"prompt_tokens\":{n_prompt},\"images\":0,\
             \"max_new\":{max_new},\"queued_ms\":{queued_ms:.1}",
            jq(phase),
            jq(&method),
        )
    };
    state.publish_request(format!("{{{}}}", req_base("queued", 0.0)));
    // Counts this request in the card queue until it holds the card; any early return leaves it.
    let mut ticket = WaitTicket::enter(state, idx);
    // One turn covers the model load and the generation, so no request for another model can swap the card
    // between them; a request for the model already on the card may go ahead of older ones (TurnQueue).
    let turn = state.turns.acquire(
        idx,
        || state.ready_model(),
        || TurnLimits::read(state.cfg.queue_config.as_deref()),
    );

    // Ensure target model is loaded on Gaudi 2 before starting generation.
    if let Err(e) = state.ensure_loaded(idx, false) {
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
        ..Qwen3GenParams::default()
    };

    // Acquire card op lock for exclusive Gaudi 2 execution.
    let _op = state.card_op.lock().unwrap();
    {
        let mut s = state.slot.lock().unwrap();
        s.active_requests += 1;
        ticket.leave(&mut s);
        s.last_activity = Instant::now();
    }
    state.publish_engine_event();
    let _ = rguard.apply(ReqEvent::EvReqStartPrefill);

    // Take engine temporarily out of slot so status endpoints never block while generating!
    let (mut engine, hbm_bytes) = {
        let mut s = state.slot.lock().unwrap();
        match s.engine.take() {
            Some(e) if s.resident_idx == Some(idx) => (e, s.hbm_bytes),
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
    let live = engine.live_profile().filter(|h| h.is_live());
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
    let report =
        engine.generate_cancellable(&prompt_ids, &params, Some(&cancel_flag), &mut |tok_id| {
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
        });
    if let Some(h) = &live {
        h.set_launch_sink(None);
    }

    // Return engine cleanly to slot & decrement active counter!
    {
        let mut s = state.slot.lock().unwrap();
        s.engine = Some(engine);
        s.active_requests = s.active_requests.saturating_sub(1);
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
                prompt_n: rep.prompt_tokens as u64,
                prompt_ms: rep.ttft_ms,
                predicted_n: n_gen,
                predicted_ms: rep.decode_ms,
                launches,
                draft_n: rep.drafted,
                draft_n_accepted: rep.draft_accepted,
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
                "{{\"prompt_tokens\":{},\"completion_tokens\":{n_gen},\"total_tokens\":{}}}",
                rep.prompt_tokens,
                rep.prompt_tokens as u64 + n_gen
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
                let usage = anthropic::usage_json(rep.prompt_tokens as u64, n_gen);
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

fn handle_static_ui(state: &AppState, url_path: &str, stream: &mut TcpStream, send_body: bool) {
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

/// The prompt text a chat-completions request body renders to with the chat template of the
/// model in `model_dir` (`gaudium --apply-template`).
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
}
