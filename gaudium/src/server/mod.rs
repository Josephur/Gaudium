//! Zero-dependency OpenAI-compatible HTTP/SSE + Gaudium Web UI server.
//!
//! Serves:
//! - Gaudium Web UI static assets (`GET /`, `/sw.js`, `/css/*`, `/js/*`, `/img/*`, `/vendor/*`)
//! - OpenAI-compatible chat & model endpoints (`/v1/models`, `/v1/chat/completions`, `/props`,
//!   `/health`, `/tokenize`, `/detokenize`, `/slots`, `/models/load`, `/models/unload`)
//! - Gaudium UI telemetry & card-lifecycle endpoints (`/gaudium/ui`, `/gaudium/engine`,
//!   `/gaudium/live`, `/gaudium/events`, `/gaudium/count_tokens`, `/v1/chat/completions/control`)
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
use crate::engine::qwen3::{Qwen3Config, Qwen3Engine, Qwen3GenParams, KV_CHUNK};
use crate::json::{self, JsonValue};
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::tokenizer::{StreamDecoder, Tokenizer};
use crate::verified::{BackendState, ReqEvent};

const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_MAX_TOKENS: u64 = 2048;

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
    pub api_key: Option<String>,
    pub verbose: bool,
}

struct RegisteredModel {
    id: String,
    dir: PathBuf,
    spec: ModelSpec,
    tokenizer: Arc<Tokenizer>,
    extra_eos_ids: Vec<i32>,
    max_ctx: u64,
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
    engine: Option<Qwen3Engine>,
    resident_idx: Option<usize>,
    lifecycle: Lifecycle,
    detail: String,
    since: Instant,
    last_activity: Instant,
    admin_parked: bool,
    hold: bool,
    active_requests: u64,
    switches: u64,
    last_switch_s: Option<f64>,
    hbm_bytes: Option<[u64; 5]>,
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
}

impl AppState {
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
        let payload = self.engine_status_json(true);
        self.hub.publish(payload);
    }

    fn hello_json(&self) -> String {
        let (res_idx, state_str) = {
            let s = self.slot.lock().unwrap();
            (s.resident_idx.unwrap_or(0), s.lifecycle.as_str())
        };
        let m = &self.models[res_idx];
        format!(
            "{{\"type\":\"hello\",\"backend\":\"engine\",\"model\":{},\"n_ctx\":{},\"vision\":false,\
             \"speculative\":false,\"build\":{},\"supports_inject\":false,\"mock_card\":false,\
             \"state\":{},\"ui\":{{\"root\":\"gaudium\",\"gaudium\":\"/\",\"llama\":\"/\"}},\
             \"api\":{{\"events\":\"/gaudium/events\",\"live\":\"/gaudium/live\",\"engine\":\"/gaudium/engine\",\
             \"count_tokens\":\"/gaudium/count_tokens\",\"version\":1}}}}",
            jq(&m.id),
            m.max_ctx,
            jq(&format!("gaudium-rust {}", env!("CARGO_PKG_VERSION"))),
            jq(state_str),
        )
    }

    fn engine_status_json(&self, include_type: bool) -> String {
        let s = self.slot.lock().unwrap();
        let res_idx = s.resident_idx.unwrap_or(0);
        let state_for_s = s.since.elapsed().as_secs();
        let idle_for_s = s.last_activity.elapsed().as_secs();
        let active_reqs = s.active_requests;
        let state_str = s.lifecycle.as_str();
        let detail = s.detail.clone();
        let hold = s.hold;
        let admin_parked = s.admin_parked;
        let switches = s.switches;
        let last_switch_s = s.last_switch_s;
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
                "{{\"id\":{},\"state\":{},\"resident\":{},\"running\":{},\"waiting\":0,\"n_ctx\":{},\"vision\":false,\"speculative\":false}}",
                jq(&m.id),
                jq(mstate),
                is_res,
                if is_res { active_reqs } else { 0 },
                m.max_ctx,
            ));
        }
        models_arr.push(']');
        drop(s);

        let mut out = String::with_capacity(512);
        out.push('{');
        if include_type {
            out.push_str("\"type\":\"engine\",");
        }
        out.push_str(&format!(
            "\"state\":{},\"detail\":{},\"state_for_s\":{},\"idle_for_s\":{},\"idle_park_s\":{},\
             \"hold\":{},\"admin_parked\":{},\"running\":{},\"residency\":\"swap\",\"resident\":{},\
             \"switching_to\":{},\"switches\":{},\"last_switch_s\":{},\"models\":{},\"card\":{{\"owner\":\"gaudium-rust\",\"holder\":{}}}}}",
            jq(state_str),
            jq(&detail),
            state_for_s,
            idle_for_s,
            self.cfg.idle_park_secs,
            hold,
            admin_parked,
            active_reqs,
            jq(&resident_id),
            switching_to.as_deref().map(jq).unwrap_or_else(|| "null".into()),
            switches,
            last_switch_s
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "null".into()),
            models_arr,
            if state_str == "ready" || state_str == "loading" {
                "\"gaudium-rust\""
            } else {
                "null"
            },
        ));
        out
    }

    fn live_snapshot_json(&self) -> String {
        format!(
            "{{\"hello\":{},\"engine\":{}}}",
            self.hello_json(),
            self.engine_status_json(true),
        )
    }

    fn model_entry_json(&self, idx: usize) -> String {
        let m = &self.models[idx];
        let s = self.slot.lock().unwrap();
        let (status_val, resident) = match &s.lifecycle {
            Lifecycle::Ready(r) if *r == idx => ("loaded", true),
            Lifecycle::Loading(r) if *r == idx => ("loading", false),
            Lifecycle::Failed(_) if s.resident_idx == Some(idx) => ("failed", false),
            _ => ("unloaded", false),
        };
        drop(s);

        let mut out = format!(
            "{{\"id\":{},\"object\":\"model\",\"owned_by\":\"gaudium\",\"created\":0,\
             \"context_length\":{},\"architecture\":{{\"input_modalities\":[\"text\"]}},\
             \"meta\":{{\"n_ctx_train\":262144,\"speculative\":false,\"family\":{},\"layers\":{},\"hidden\":{}}}",
            jq(&m.id),
            m.max_ctx,
            jq(&format!("{:?}", m.spec.family)),
            m.spec.n_layers,
            m.spec.hidden,
        );
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
        let mut out = format!(
            "{{\"default_generation_settings\":{{\"n_ctx\":{},\"params\":{{\"seed\":-1,\"temperature\":0.7,\
             \"top_k\":20,\"top_p\":0.8,\"min_p\":0.0,\"max_tokens\":-1,\"n_predict\":-1,\"stream\":false,\
             \"timings_per_token\":false,\"reasoning_format\":\"auto\",\"speculative\":false,\
             \"samplers\":[\"top_k\",\"top_p\",\"min_p\",\"temperature\"]}}}},\
             \"total_slots\":1,\"model_alias\":{},\"model_path\":{},\
             \"modalities\":{{\"vision\":false,\"audio\":false,\"video\":false}},\
             \"endpoint_slots\":true,\"endpoint_props\":false,\"endpoint_metrics\":false,\"ui\":{},\
             \"ui_settings\":{{}},\"chat_template\":\"qwen3\",\
             \"chat_template_caps\":{{\"supports_string_content\":true,\"supports_typed_content\":true,\
             \"supports_tools\":false,\"supports_tool_calls\":false,\"supports_parallel_tool_calls\":false,\
             \"supports_system_role\":true,\"supports_preserve_reasoning\":true,\"supports_reasoning_effort\":false,\
             \"supports_object_arguments\":false,\"reasoning_effort_levels\":[]}},\
             \"bos_token\":\"\",\"eos_token\":\"<|im_end|>\",\"build_info\":{},\"is_sleeping\":false,\
             \"cors_proxy_enabled\":false",
            m.max_ctx,
            jq(&m.id),
            jq(&m.dir.display().to_string()),
            self.cfg.ui_dir.is_some(),
            jq(&format!("gaudium-rust {}", env!("CARGO_PKG_VERSION"))),
        );
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
                s.lifecycle = Lifecycle::Parked;
            }
            if !can_switch_resident_model(s.active_requests, s.lifecycle.as_backend_state()) {
                return Err("verified single-resident guard refused card switch".into());
            }
            s.admin_parked = false;
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
        let mut qcfg = Qwen3Config::new(m.dir.clone(), self.cfg.kernels_dir.clone());
        qcfg.max_ctx = m.max_ctx;
        qcfg.cache_dir = self.cfg.cache_dir.clone();
        qcfg.no_cache = self.cfg.no_cache;
        qcfg.verbose = self.cfg.verbose;

        match Qwen3Engine::load(&qcfg) {
            Ok(eng) => {
                let dt = t0.elapsed().as_secs_f64();
                let hbm = eng.hbm_bytes();
                let mut s = self.slot.lock().unwrap();
                let swapped = s.resident_idx.is_some() && s.resident_idx != Some(target_idx);
                if swapped {
                    s.switches += 1;
                    s.last_switch_s = Some(dt);
                }
                s.engine = Some(eng);
                s.resident_idx = Some(target_idx);
                s.lifecycle = Lifecycle::Ready(target_idx);
                s.detail = format!("{} ready ({:.1}s init)", m.id, dt);
                s.since = Instant::now();
                s.last_activity = Instant::now();
                s.hbm_bytes = Some(hbm);
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
            s.engine = None; // drains streams & releases device
            s.hbm_bytes = None;
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
pub fn run_server(cfg: ServerConfig) -> Result<(), String> {
    if cfg.models.is_empty() {
        return Err("at least one --model <dir> is required for --serve".into());
    }

    let max_ctx = crate::verified::align_up(cfg.max_ctx.max(KV_CHUNK), KV_CHUNK);
    let mut models = Vec::with_capacity(cfg.models.len());
    for entry in &cfg.models {
        let spec = ModelSpec::from_config_file(entry.dir.join("config.json"))
            .map_err(|e| format!("load config.json for '{}': {e}", entry.id))?;
        spec.validate()
            .map_err(|e| format!("invalid ModelSpec for '{}': {e}", entry.id))?;
        let tok = Tokenizer::from_model_dir(&entry.dir)
            .map_err(|e| format!("load tokenizer.json for '{}': {e}", entry.id))?;
        let mut extra_eos_ids = Vec::new();
        for special in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = tok.token_id(special) {
                extra_eos_ids.push(id);
            }
        }
        models.push(RegisteredModel {
            id: entry.id.clone(),
            dir: entry.dir.clone(),
            spec,
            tokenizer: Arc::new(tok),
            extra_eos_ids,
            max_ctx,
        });
    }

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;

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
            switches: 0,
            last_switch_s: None,
            hbm_bytes: None,
        }),
        card_op: Mutex::new(()),
        cancels: Mutex::new(HashMap::new()),
        next_req_id: AtomicU64::new(1),
        hub: EventHub::new(),
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

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let st = Arc::clone(&state);
                thread::spawn(move || handle_connection(st, stream));
            }
            Err(e) => {
                eprintln!("[server] accept error: {e}");
            }
        }
    }
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

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type, Authorization, X-Api-Key\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    if !body.is_empty() {
        let _ = stream.write_all(body);
    }
    let _ = stream.flush();
}

fn write_json(stream: &mut TcpStream, status: u16, json: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        409 => "Conflict",
        503 => "Service Unavailable",
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
                Cache-Control: no-cache, no-store\r\n\
                X-Accel-Buffering: no\r\n\
                Access-Control-Allow-Origin: *\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes())?;
    stream.flush()
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
        ("GET", "/gaudium/engine") => {
            write_json(&mut stream, 200, &state.engine_status_json(false));
        }
        ("POST", "/gaudium/engine") => handle_engine_action(&state, &req, &mut stream),
        ("GET", "/gaudium/live") => {
            write_json(&mut stream, 200, &state.live_snapshot_json());
        }
        ("GET", "/gaudium/events") => handle_events_sse(&state, &mut stream),
        ("GET", "/slots") => {
            let active = state.slot.lock().unwrap().active_requests;
            write_json(
                &mut stream,
                200,
                &format!("[{{\"id\":0,\"is_processing\":{}}}]", active > 0),
            );
        }
        ("POST", "/v1/streams/lookup") => write_json(&mut stream, 200, "[]"),
        ("POST", "/tokenize") => handle_tokenize(&state, &req, &mut stream),
        ("POST", "/detokenize") => handle_detokenize(&state, &req, &mut stream),
        ("POST", "/gaudium/count_tokens") => handle_count_tokens(&state, &req, &mut stream),
        ("POST", "/v1/chat/completions/control") => handle_chat_control(&state, &req, &mut stream),
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            handle_chat_completions(&state, &req, &mut stream);
        }
        ("GET", _) => handle_static_ui(&state, &req.path, &mut stream),
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
        if stream
            .write_all(format!("data: {msg}\n\n").as_bytes())
            .is_err()
            || stream.flush().is_err()
        {
            return;
        }
    }
    loop {
        let (evs, closed) = state.hub.since(seq, Duration::from_secs(10));
        if closed {
            break;
        }
        if evs.is_empty() {
            if stream.write_all(b":\n\n").is_err() || stream.flush().is_err() {
                break;
            }
            continue;
        }
        for (s, payload) in evs {
            seq = s;
            if stream
                .write_all(format!("data: {payload}\n\n").as_bytes())
                .is_err()
                || stream.flush().is_err()
            {
                return;
            }
        }
    }
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
                "{{\"ok\":true,\"message\":{},\"status\":{}}}",
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
    let preopen_think = m.spec.family == ModelFamily::Qwen35Dense;
    let prompt_text = match build_chat_prompt(&val, preopen_think) {
        Ok(p) => p,
        Err(e) => {
            write_error(stream, 400, "invalid_request_error", &e);
            return;
        }
    };
    let ids = m.tokenizer.encode(&prompt_text, true);
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

fn handle_chat_completions(state: &AppState, req: &HttpRequest, stream: &mut TcpStream) {
    let body_str = String::from_utf8_lossy(&req.body);
    let Ok(val) = json::parse_json(&body_str) else {
        write_error(stream, 400, "invalid_request_error", "invalid JSON body");
        return;
    };
    let idx = state.find_model(val.get("model").and_then(|v| v.as_str()));
    let m = &state.models[idx];
    let preopen_think = m.spec.family == ModelFamily::Qwen35Dense;
    let prompt_text = match build_chat_prompt(&val, preopen_think) {
        Ok(p) => p,
        Err(e) => {
            write_error(stream, 400, "invalid_request_error", &e);
            return;
        }
    };
    let prompt_ids = m.tokenizer.encode(&prompt_text, true);
    let n_prompt = prompt_ids.len() as u64;
    let stream_mode = val.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let enable_thinking = val
        .get("chat_template_kwargs")
        .and_then(|v| v.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let temperature = val
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;
    let top_p = val.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
    let top_k = val.get("top_k").and_then(|v| v.as_u64()).unwrap_or(20);
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
    let budget = match SequenceBudget::admit(
        n_prompt,
        Some(req_max.unwrap_or(DEFAULT_MAX_TOKENS)),
        m.max_ctx,
    ) {
        Ok(b) => b,
        Err(e) => {
            write_error(
                stream,
                400,
                "invalid_request_error",
                &format!(
                    "prompt ({n_prompt} tokens) exceeds context budget {}: {e:?}",
                    m.max_ctx
                ),
            );
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

    // Ensure target model is loaded on Gaudi 2 before starting generation.
    if let Err(e) = state.ensure_loaded(idx, false) {
        write_error(stream, 503, "unavailable_error", &e);
        return;
    }

    let req_num = state.next_req_id.fetch_add(1, Ordering::Relaxed);
    let req_id = format!("chatcmpl-gaudium-{req_num}");
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

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

    let mut params = Qwen3GenParams {
        max_new: budget.max_new(),
        temperature,
        top_p,
        top_k,
        min_p,
        seed,
        stop_on_eos: true,
        extra_eos_ids: m.extra_eos_ids.clone(),
    };
    if params.max_new == 0 {
        params.max_new = 1;
    }

    // Acquire card op lock for exclusive Gaudi 2 execution.
    let _op = state.card_op.lock().unwrap();
    {
        let mut s = state.slot.lock().unwrap();
        s.active_requests += 1;
        s.last_activity = Instant::now();
    }
    let _ = rguard.apply(ReqEvent::EvReqStartPrefill);

    // Take engine temporarily out of slot so status endpoints never block while generating!
    let mut engine = {
        let mut s = state.slot.lock().unwrap();
        match s.engine.take() {
            Some(e) if s.resident_idx == Some(idx) => e,
            other => {
                s.engine = other;
                s.active_requests = s.active_requests.saturating_sub(1);
                drop(s);
                state.cancels.lock().unwrap().remove(&req_id);
                let msg = "model unloaded concurrently; retry request";
                if stream_mode {
                    let _ = stream.write_all(
                        format!(
                            "data: {{\"error\":{{\"message\":{}}}}}\n\ndata: [DONE]\n\n",
                            jq(msg)
                        )
                        .as_bytes(),
                    );
                } else {
                    write_error(stream, 503, "unavailable_error", msg);
                }
                return;
            }
        }
    };

    let mut decoder = StreamDecoder::new(&m.tokenizer);
    let mut splitter = ThinkSplitter::new(enable_thinking, preopen_think);
    let mut full_reasoning = String::new();
    let mut full_content = String::new();
    let mut write_failed = false;
    let mut hit_stop = false;
    let mut started_decode = false;
    let tok_ref = Arc::clone(&m.tokenizer);
    let model_id = m.id.clone();

    if stream_mode {
        // Send initial role chunk immediately so UI shows the assistant turn is active.
        let init_chunk = format!(
            "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{},\
             \"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"\"}},\"finish_reason\":null}}]}}\n\n",
            jq(&req_id),
            jq(&model_id),
        );
        if stream.write_all(init_chunk.as_bytes()).is_err() || stream.flush().is_err() {
            write_failed = true;
            cancel_flag.store(true, Ordering::Relaxed);
        }
    }

    let report = engine.generate_cancellable(&prompt_ids, &params, Some(&cancel_flag), |tok_id| {
        if !started_decode {
            started_decode = true;
            let _ = rguard.apply(ReqEvent::EvReqPrefillDone);
        }
        if tok_ref.is_special(tok_id) {
            return;
        }
        let piece = decoder.push(tok_id);
        if piece.is_empty() {
            return;
        }
        let (r_delta, c_delta) = splitter.push(&piece);
        if !r_delta.is_empty() {
            full_reasoning.push_str(&r_delta);
        }
        if !c_delta.is_empty() {
            full_content.push_str(&c_delta);
            if !stop_seqs.is_empty() && stop_seqs.iter().any(|st| full_content.ends_with(st)) {
                hit_stop = true;
                cancel_flag.store(true, Ordering::Relaxed);
            }
        }
        if stream_mode && !write_failed && (!r_delta.is_empty() || !c_delta.is_empty()) {
            let mut delta = String::from("{");
            let mut first = true;
            if !r_delta.is_empty() {
                delta.push_str(&format!("\"reasoning_content\":{}", jq(&r_delta)));
                first = false;
            }
            if !c_delta.is_empty() {
                if !first {
                    delta.push(',');
                }
                delta.push_str(&format!("\"content\":{}", jq(&c_delta)));
            }
            delta.push('}');
            let chunk = format!(
                "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{},\
                 \"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":null}}]}}\n\n",
                jq(&req_id),
                jq(&model_id),
            );
            if stream.write_all(chunk.as_bytes()).is_err() || stream.flush().is_err() {
                write_failed = true;
                cancel_flag.store(true, Ordering::Relaxed);
            }
        }
    });

    // Return engine cleanly to slot & decrement active counter!
    {
        let mut s = state.slot.lock().unwrap();
        s.engine = Some(engine);
        s.active_requests = s.active_requests.saturating_sub(1);
        s.last_activity = Instant::now();
    }
    state.cancels.lock().unwrap().remove(&req_id);

    // Flush decoder + splitter tail
    let tail = decoder.flush();
    let (r_tail, c_tail) = splitter.finish(&tail);
    if !r_tail.is_empty() {
        full_reasoning.push_str(&r_tail);
    }
    if !c_tail.is_empty() {
        full_content.push_str(&c_tail);
    }

    match report {
        Ok(rep) => {
            if rep.stopped_by_cancel && !hit_stop {
                let _ = rguard.cancel();
            } else {
                let _ = rguard.apply(ReqEvent::EvReqFinishStop);
            }
            let finish_reason = if rep.stopped_by_eos || hit_stop {
                "stop"
            } else if rep.stopped_by_cancel {
                "cancelled"
            } else {
                "length"
            };
            let n_gen = rep.generated_tokens.len();
            let timings = format!(
                "{{\"prompt_n\":{},\"prompt_ms\":{:.2},\"prompt_per_token_ms\":{:.3},\"prompt_per_second\":{:.2},\
                 \"predicted_n\":{},\"predicted_ms\":{:.2},\"predicted_per_token_ms\":{:.3},\"predicted_per_second\":{:.2},\"cache_n\":0}}",
                rep.prompt_tokens,
                rep.ttft_ms,
                rep.ttft_ms / (rep.prompt_tokens.max(1) as f64),
                rep.prefill_tok_s,
                n_gen,
                rep.decode_ms,
                rep.ms_per_step,
                rep.decode_tok_s,
            );
            let usage = format!(
                "{{\"prompt_tokens\":{},\"completion_tokens\":{n_gen},\"total_tokens\":{}}}",
                rep.prompt_tokens,
                rep.prompt_tokens + n_gen
            );
            let gaudium_meta = format!(
                "{{\"method\":\"K=0 BF16\",\"steps\":{},\"ttft_ms\":{:.2},\"decode_tok_s\":{:.2}}}",
                n_gen.saturating_sub(1),
                rep.ttft_ms,
                rep.decode_tok_s
            );

            if stream_mode {
                if !write_failed {
                    let mut tail_delta = String::from("{");
                    let mut first = true;
                    if !r_tail.is_empty() {
                        tail_delta.push_str(&format!("\"reasoning_content\":{}", jq(&r_tail)));
                        first = false;
                    }
                    if !c_tail.is_empty() {
                        if !first {
                            tail_delta.push(',');
                        }
                        tail_delta.push_str(&format!("\"content\":{}", jq(&c_tail)));
                    }
                    tail_delta.push('}');
                    let final_chunk = format!(
                        "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{},\
                         \"choices\":[{{\"index\":0,\"delta\":{tail_delta},\"finish_reason\":{}}}],\
                         \"usage\":{usage},\"timings\":{timings},\"gaudium\":{gaudium_meta}}}\n\ndata: [DONE]\n\n",
                        jq(&req_id),
                        jq(&model_id),
                        jq(finish_reason),
                    );
                    let _ = stream.write_all(final_chunk.as_bytes());
                    let _ = stream.flush();
                }
            } else {
                let mut msg = format!("{{\"role\":\"assistant\",\"content\":{}", jq(&full_content));
                if !full_reasoning.is_empty() {
                    msg.push_str(&format!(",\"reasoning_content\":{}", jq(&full_reasoning)));
                }
                msg.push('}');
                let resp = format!(
                    "{{\"id\":{},\"object\":\"chat.completion\",\"created\":{created},\"model\":{},\
                     \"choices\":[{{\"index\":0,\"message\":{msg},\"finish_reason\":{}}}],\
                     \"usage\":{usage},\"timings\":{timings},\"gaudium\":{gaudium_meta}}}",
                    jq(&req_id),
                    jq(&model_id),
                    jq(finish_reason),
                );
                write_json(stream, 200, &resp);
            }
        }
        Err(e) => {
            let _ = rguard.apply(ReqEvent::EvReqFail);
            if stream_mode {
                let _ = stream.write_all(
                    format!(
                        "data: {{\"error\":{{\"message\":{},\"type\":\"server_error\"}}}}\n\ndata: [DONE]\n\n",
                        jq(&e)
                    )
                    .as_bytes(),
                );
                let _ = stream.flush();
            } else {
                write_error(stream, 500, "server_error", &e);
            }
        }
    }
}

// ============================================================================
// Static Gaudium UI File Server
// ============================================================================

fn handle_static_ui(state: &AppState, url_path: &str, stream: &mut TcpStream) {
    let Some(ui_root) = state.cfg.ui_dir.as_deref() else {
        write_error(
            stream,
            404,
            "not_found_error",
            "Web UI disabled (pass --ui-dir <path>)",
        );
        return;
    };

    let rel = match url_path {
        "/" | "" => "index.html",
        "/sw.js" => "sw-reset.js",
        other => other.trim_start_matches('/'),
    };
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
            write_response(stream, 200, "OK", mime, &bytes);
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
        "js" | "mjs" => "application/javascript; charset=utf-8",
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
// ChatML Prompt Builder & Streaming `<think>...</think>` Splitter
// ============================================================================

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

/// Streaming parser separating `<think>...</think>` blocks into `(reasoning_content, content)`.
#[derive(Debug)]
pub struct ThinkSplitter {
    state: ThinkState,
    buf: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkState {
    Start,
    InThink,
    Content,
}

impl ThinkSplitter {
    pub fn new(enable_thinking: bool, preopened: bool) -> Self {
        Self {
            state: if !enable_thinking {
                ThinkState::Content
            } else if preopened {
                ThinkState::InThink
            } else {
                ThinkState::Start
            },
            buf: String::new(),
        }
    }

    pub fn push(&mut self, piece: &str) -> (String, String) {
        let mut reasoning = String::new();
        let mut content = String::new();
        self.buf.push_str(piece);

        loop {
            match self.state {
                ThinkState::Start => {
                    let trimmed = self.buf.trim_start_matches(['\n', '\r', ' ']);
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some(rest) = trimmed.strip_prefix("<think>") {
                        let rest = rest.trim_start_matches('\n').to_string();
                        self.buf = rest;
                        self.state = ThinkState::InThink;
                        continue;
                    }
                    if "<think>".starts_with(trimmed) {
                        break;
                    }
                    // No `<think>` opening tag -> normal content
                    content.push_str(&self.buf);
                    self.buf.clear();
                    self.state = ThinkState::Content;
                    break;
                }
                ThinkState::InThink => {
                    if let Some(idx) = self.buf.find("</think>") {
                        reasoning.push_str(&self.buf[..idx]);
                        let after = self.buf[idx + "</think>".len()..]
                            .trim_start_matches('\n')
                            .to_string();
                        self.buf = after;
                        self.state = ThinkState::Content;
                        continue;
                    }
                    // Emit everything except any trailing prefix of "</think>"
                    let keep = trailing_tag_prefix_len(&self.buf, "</think>");
                    let emit_len = self.buf.len() - keep;
                    if emit_len > 0 {
                        reasoning.push_str(&self.buf[..emit_len]);
                        self.buf.drain(..emit_len);
                    }
                    break;
                }
                ThinkState::Content => {
                    content.push_str(&self.buf);
                    self.buf.clear();
                    break;
                }
            }
        }
        (reasoning, content)
    }

    pub fn finish(&mut self, tail: &str) -> (String, String) {
        let (mut r, mut c) = self.push(tail);
        if !self.buf.is_empty() {
            match self.state {
                ThinkState::InThink => r.push_str(&self.buf),
                ThinkState::Start | ThinkState::Content => c.push_str(&self.buf),
            }
            self.buf.clear();
        }
        (r, c)
    }
}

fn trailing_tag_prefix_len(haystack: &str, tag: &str) -> usize {
    let max_k = haystack.len().min(tag.len().saturating_sub(1));
    for k in (1..=max_k).rev() {
        if haystack.is_char_boundary(haystack.len() - k) && haystack.ends_with(&tag[..k]) {
            return k;
        }
    }
    0
}

fn jq(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
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
mod tests {
    use super::*;

    #[test]
    fn think_splitter_handles_split_tags_and_plain_content() {
        let mut s = ThinkSplitter::new(true, false);
        let mut r = String::new();
        let mut c = String::new();
        for piece in [
            "<thi",
            "nk>\nStep 1",
            " done</th",
            "ink>\n\nHello",
            " world!",
        ] {
            let (rd, cd) = s.push(piece);
            r.push_str(&rd);
            c.push_str(&cd);
        }
        let (rt, ct) = s.finish("");
        r.push_str(&rt);
        c.push_str(&ct);
        assert_eq!(r, "Step 1 done");
        assert_eq!(c, "Hello world!");

        // Pre-opened thinking (Qwen3.5/3.8)
        let mut pre = ThinkSplitter::new(true, true);
        let (pr1, pc1) = pre.push("reasoning</think>\n\nAnswer");
        let (pr2, pc2) = pre.finish("");
        assert_eq!(format!("{pr1}{pr2}"), "reasoning");
        assert_eq!(format!("{pc1}{pc2}"), "Answer");

        let mut plain = ThinkSplitter::new(true, false);
        let (r2, c2) = plain.push("Just a direct answer");
        assert!(r2.is_empty());
        assert_eq!(c2, "Just a direct answer");
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
    fn safe_join_blocks_path_traversal() {
        let root = Path::new("/srv/ui");
        assert_eq!(
            safe_join(root, "js/app.js"),
            Some(PathBuf::from("/srv/ui/js/app.js"))
        );
        assert_eq!(safe_join(root, "../etc/passwd"), None);
        assert_eq!(safe_join(root, "js/../../secret"), None);
        assert_eq!(safe_join(root, "/etc/passwd"), None);
    }
}
