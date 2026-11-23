//! Live telemetry reporter for the Gaudi card status dashboard (`schema: 1` live-stats protocol).
//!
//! Writes a single atomic JSON status file (`<dir>/<tag>.<pid>.json`) refreshed on every phase
//! transition, at most once per second during decode, and on a 1-second background heartbeat so
//! long recipe compiles or weight uploads stay live.
//!
//! A reporter created with the heartbeat has one writer thread, and that thread does every file
//! write after the first: a status change only wakes it. A file write can block in the filesystem
//! for tens of milliseconds (replacing a file whose previous version is still being written out),
//! and the thread that reports a change is the one running the request. Who writes is the verified
//! `live_write_route`.
//!
//! Controlled strictly through standard environment variables (zero site-specific defaults):
//! - `GAUDI_LIVE=0`: disables reporting unconditionally.
//! - `GAUDI_LIVE_DIR`: target directory for live JSON records.
//! - `GAUDIUM_LIVE_ROOT`: fallback root (uses `<root>/.cardlock/live` if `GAUDI_LIVE_DIR` is unset).
//! - `GAUDI_LIVE_TAG`: optional filename prefix (defaults to hostname or `"job"`).
//! - `GAUDI_CARD_OWNER`: optional owner label shown in the dashboard.
//!
//! Every operation is infallible and becomes a zero-cost no-op when reporting is disabled or the
//! target directory does not exist.
//!
//! # Calls an engine makes per request
//!
//! ```text
//! live.begin_request(prompt_tokens, method, notes);   // phase prefill, request clock starts
//! live.set_cached(reused_tokens);                     // optional: prompt rows not prefilled
//! live.prefill_tile(done, launches, key, || traffic); // per synchronised tile (or prefill_progress(done))
//! live.first_token(Some(ttft_ms), 1);                 // phase decode, TTFT
//! loop {
//!     live.launch_begin_with(key, || traffic);        // right before the launch call (or launch_begin())
//!     /* launch */
//!     live.launch_submitted();                        // right after the launch call returns
//!     /* tokens reach the host */
//!     live.step(emitted, drafted, accepted);          // once per consumed launch; greedy: (1, 0, 0)
//! }
//! live.end_request();                                 // phase idle; or end_request_with("cancelled" | "error")
//! ```
//!
//! `step` counts are per launch: `emitted` tokens appended to the output (accepted drafts plus the
//! bonus token), `drafted` draft tokens the launch verified, `accepted` drafts it kept. They give
//! `tok_per_launch`, `ms_per_launch` and `accept_rate` over the rolling ~5 s window. `set_method`
//! changes the method label mid-request. Every exit path of a request (end, cancel, error) must
//! reach `end_request` / `end_request_with`, or the record stays in its last phase.
//!
//! # Record fields beyond the base protocol (all optional for consumers)
//!
//! - `request_seq`: counts `begin_request` calls of this process, so two consecutive requests
//!   with the same prompt length and method are told apart.
//! - `prefill_pos`, `prefill_tok_s` (rolling ~5 s), `prefill_tok_s_run` (rows prefilled / time
//!   so far; rows prefilled / TTFT from the first token on), `prefill_eta_s`: present once an
//!   engine reports prefill progress. While such an engine prefills, `context_pos` is the prefill
//!   cursor instead of `prompt_tokens + gen_tokens`.
//! - `outcome`: how the last request ended, while the phase is `idle`.
//! - `profile`: built by the reporter from the launch marks and the engine's byte model
//!   (`crate::live_profile`) unless the engine supplies its own with `set_profile`. During
//!   prefill it carries the running `prefill` entry only.
//!
//! # In-process readers
//!
//! A server in the same process reads the same measurements through a [`ProfileHandle`]: the
//! `profile` block, the request's per-launch log ([`RequestSummary`]) and, pushed on the engine's
//! thread, one [`LaunchEvent`] per `first_token` / `step` call ([`ProfileHandle::set_launch_sink`]).
//! [`enable_in_memory`] keeps this state for reporters created without a live directory (no file
//! is written then); without that call such a reporter stays a no-op.
//!
//! A decode loop that sets no launch marks (more than one launch in flight) can hand the reporter
//! a [`LaunchModel`]: the byte model of the launch that `step` reports, so the `profile` block
//! still carries the HBM flow and the step split, timed by the token arrivals alone.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::env;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::live_profile::{StepSample as LaunchSample, StepTimeline, Traffic};
use crate::verified::{live_write_route, LiveWrite};

/// Process-wide: reporters created without a live directory keep their state in memory.
static IN_MEMORY: AtomicBool = AtomicBool::new(false);

/// Makes every reporter created from here on keep its request state in memory even when no live
/// directory is configured (or `GAUDI_LIVE=0`), so an in-process reader (`ProfileHandle`) gets
/// the profile block and the launch feed. No file is written for such a reporter.
pub fn enable_in_memory() {
    IN_MEMORY.store(true, Ordering::Relaxed);
}

/// Most launches the per-request launch log keeps (a request can not run more decode launches
/// than the largest context has positions).
const LAUNCH_LOG_MAX: usize = 1 << 20;

/// What a [`LaunchEvent`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchKind {
    /// Prefill progress (`prefill_progress` / `prefill_tile`): `prefill_pos` prompt tokens are
    /// in the model's context.
    Prefill,
    /// The first token is on the host: the prefill ended (`first_token`).
    First,
    /// One decode launch was consumed (`step`).
    Step,
}

/// One `first_token` / `step` call with the request's running totals, as the engine reported it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaunchEvent {
    pub kind: LaunchKind,
    /// Tokens generated so far, the first token included.
    pub gen_tokens: u64,
    /// Decode launches consumed so far (the prefill's readout is not one).
    pub launches: u64,
    /// Drafts verified / accepted so far.
    pub drafted: u64,
    pub accepted: u64,
    /// This launch alone: tokens it added, drafts it verified, drafts it kept (0 for `First`).
    pub launch_emitted: u64,
    pub launch_drafted: u64,
    pub launch_accepted: u64,
    /// First token on the host -> this launch's tokens on the host, on the reporter's clock.
    pub decode_ms: f64,
    /// The engine's time to first token, as passed to `first_token` (else measured from
    /// `begin_request`); 0 before the first token.
    pub ttft_ms: f64,
    /// Prompt tokens in the model's context so far (the whole prompt from the first token on).
    pub prefill_pos: u64,
    /// `begin_request` -> this event, on the reporter's clock.
    pub request_ms: f64,
}

/// Receiver of the launch feed; called on the engine's thread, outside the reporter's locks.
pub type LaunchSink = Box<dyn FnMut(&LaunchEvent) + Send>;

/// Byte model of the launches a decode loop reports with `step` but does not mark with
/// `launch_begin_with` (see the module documentation).
pub trait LaunchModel: Send {
    /// Cache key of the launch that verified `drafted` drafts with row 0 at context position
    /// `pos`; `None` when this launch is not modelled. Keys must be stable for the process.
    fn key(&self, drafted: u64, pos: u64) -> Option<u64>;
    /// The traffic of one launch with that key (built once per key, then cached).
    fn build(&self, key: u64) -> Traffic;
}

/// Keeps a [`LaunchModel`]'s keys apart from the keys of marked launches and prefill tiles.
const MODEL_KEY_BIT: u64 = 1 << 62;

/// One consumed decode launch of the current request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaunchRec {
    /// First token on the host -> this launch's tokens on the host.
    pub decode_ms: f32,
    pub emitted: u16,
    pub drafted: u16,
    pub accepted: u16,
}

/// The current (or last) request as the engine reported it: totals, rolling rates and the
/// per-launch log. Every rate is `None` until the engine's calls can give it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestSummary {
    pub request_seq: u64,
    pub phase: String,
    pub method: String,
    pub prompt_tokens: Option<u64>,
    pub gen_tokens: u64,
    pub launches: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub ttft_ms: Option<f64>,
    /// Rolling ~5 s window, first token excluded.
    pub decode_tok_s: Option<f64>,
    /// Whole request so far, first token excluded.
    pub decode_tok_s_run: Option<f64>,
    pub tok_per_launch: Option<f64>,
    pub ms_per_launch: Option<f64>,
    pub accept_rate: Option<f64>,
    pub log: Vec<LaunchRec>,
}

#[derive(Debug, Clone, Copy)]
struct StepSample {
    t_s: f64,
    decode_tokens: u64,
    launches: u64,
    drafted: u64,
    accepted: u64,
}

struct State {
    engine: String,
    version: String,
    source: String,
    owner: String,
    model: String,
    method: String,
    notes: String,
    phase: String,
    hbm_detail: String,
    hbm_kinds: String,
    sources: String,
    profile: String,
    host_parts: BTreeMap<String, (f64, bool)>,

    pid: u32,
    prompt_tokens: Option<u64>,
    prompt_cached: u64,
    gen_tokens: u64,
    first_n: u64,
    launches: u64,
    drafted: u64,
    accepted: u64,

    ttft_ms: Option<f64>,
    rate: Option<f64>,
    rate_run: Option<f64>,
    tok_per_launch: Option<f64>,
    ms_per_launch: Option<f64>,
    accept_rate: Option<f64>,

    origin: Instant,
    t_begin_s: f64,
    t_first_s: f64,
    last_write_s: f64,

    hbm_gb: Option<[f64; 5]>,
    hbm_used_bytes: Option<u64>,
    samples: VecDeque<StepSample>,

    // Request lifecycle detail: prefill progress, per-launch host timing, byte model.
    req_seq: u64,
    notes_is_status: bool,
    outcome: String,
    prefill_reporting: bool,
    prefill_seen: bool,
    prefill_done: u64,
    prefill_rate: Option<f64>,
    prefill_rate_run: Option<f64>,
    prefill_samples: VecDeque<(f64, u64)>,
    t_launch_begin: Option<f64>,
    t_launch_submitted: Option<f64>,
    t_prev_tokens: Option<f64>,
    traffic_cache: HashMap<u64, Arc<Traffic>>,
    cur_traffic: Option<(u64, Arc<Traffic>)>,
    timeline: StepTimeline,
    profile_auto: bool,
    launch_log: Vec<LaunchRec>,
}

struct Shared {
    state: Mutex<State>,
    /// `false`: an in-memory reporter (no live directory); nothing is written or removed.
    write_files: bool,
    sink: Mutex<Option<LaunchSink>>,
    model: Mutex<Option<Box<dyn LaunchModel>>>,
    write_lock: Mutex<()>,
    /// The writer thread's mailbox.
    wake: (Mutex<Wake>, Condvar),
    /// The writer thread runs: status changes are handed to it instead of written by the caller.
    has_writer: AtomicBool,
    /// The thread that wrote the record last.
    #[cfg(test)]
    last_writer: Mutex<Option<thread::ThreadId>>,
    path: PathBuf,
    tmp_path: PathBuf,
    lock_dir: Option<PathBuf>,
    holder_path: Option<PathBuf>,
    holder_tmp_path: Option<PathBuf>,
    shm_path: Option<PathBuf>,
    since_unix: u64,
    where_tag: String,
    load_at_lock: String,
    lock_logged: Mutex<bool>,
}

/// What the writer thread is asked to do.
#[derive(Debug, Default)]
struct Wake {
    /// The reporter is being dropped.
    stop: bool,
    /// A forced write is waiting (several requests before the thread runs make one write of the
    /// newest state).
    forced: bool,
}

/// Thread-safe, RAII live-stats writer. Unlinks its status record cleanly when dropped.
pub struct LiveReporter {
    shared: Option<Arc<Shared>>,
    heartbeat: Option<JoinHandle<()>>,
}

impl LiveReporter {
    /// Creates a reporter configured from `GAUDI_LIVE*` environment variables.
    /// Returns a no-op reporter if `GAUDI_LIVE=0` or no existing live directory is configured.
    pub fn from_env(
        engine: impl Into<String>,
        engine_version: impl Into<String>,
        source: impl Into<String>,
        heartbeat: bool,
    ) -> Self {
        let dir = if env::var("GAUDI_LIVE").as_deref() == Ok("0") {
            None
        } else {
            env::var_os("GAUDI_LIVE_DIR")
                .map(PathBuf::from)
                .or_else(|| {
                    env::var_os("GAUDIUM_LIVE_ROOT")
                        .map(|r| PathBuf::from(r).join(".cardlock").join("live"))
                })
                .filter(|d| d.is_dir())
        };
        match dir {
            Some(dir) => {
                Self::in_dir_with_shm(dir, engine, engine_version, source, heartbeat, true)
            }
            None if IN_MEMORY.load(Ordering::Relaxed) => {
                Self::in_memory(engine, engine_version, source)
            }
            None => Self::disabled(),
        }
    }

    /// A reporter that keeps its state for in-process readers and writes no file.
    pub fn in_memory(
        engine: impl Into<String>,
        engine_version: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        let state = State::new(
            engine.into(),
            engine_version.into(),
            source.into(),
            String::new(),
        );
        Self {
            shared: Some(Arc::new(Shared {
                state: Mutex::new(state),
                write_files: false,
                sink: Mutex::new(None),
                model: Mutex::new(None),
                write_lock: Mutex::new(()),
                wake: (Mutex::new(Wake::default()), Condvar::new()),
                has_writer: AtomicBool::new(false),
                #[cfg(test)]
                last_writer: Mutex::new(None),
                path: PathBuf::new(),
                tmp_path: PathBuf::new(),
                lock_dir: None,
                holder_path: None,
                holder_tmp_path: None,
                shm_path: None,
                since_unix: 0,
                where_tag: String::new(),
                load_at_lock: String::new(),
                lock_logged: Mutex::new(false),
            })),
            heartbeat: None,
        }
    }

    /// Creates a reporter targeting a specific directory (primarily for tests or explicit config).
    pub fn in_dir(
        dir: impl AsRef<Path>,
        engine: impl Into<String>,
        engine_version: impl Into<String>,
        source: impl Into<String>,
        heartbeat: bool,
    ) -> Self {
        Self::in_dir_with_shm(dir, engine, engine_version, source, heartbeat, false)
    }

    fn in_dir_with_shm(
        dir: impl AsRef<Path>,
        engine: impl Into<String>,
        engine_version: impl Into<String>,
        source: impl Into<String>,
        heartbeat: bool,
        enable_shm: bool,
    ) -> Self {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Self::disabled();
        }
        let raw_tag = env::var("GAUDI_LIVE_TAG")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(read_hostname)
            .unwrap_or_else(|| "job".to_string());
        let tag: String = raw_tag
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let pid = std::process::id();
        let path = dir.join(format!("{tag}.{pid}.json"));
        let tmp_path = dir.join(format!(".{tag}.{pid}.json.tmp"));
        let engine_str: String = engine.into();
        let owner = env::var("GAUDI_CARD_OWNER")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("server/{engine_str}"));

        let lock_dir = env::var_os("GAUDI_CARDLOCK_DIR")
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .or_else(|| {
                if dir.file_name().and_then(|s| s.to_str()) == Some("live") {
                    dir.parent().filter(|p| p.is_dir()).map(Path::to_path_buf)
                } else {
                    None
                }
            });
        let holder_path = lock_dir.as_ref().map(|d| d.join("holder"));
        let holder_tmp_path = lock_dir
            .as_ref()
            .map(|d| d.join(format!(".holder.{pid}.tmp")));
        let shm_path = if enable_shm && env::var("GAUDI_CARD_NO_HBM_REPORT").as_deref() != Ok("1") {
            let accel_idx = env::var("GAUDI_CARD_ACCEL_INDEX")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "0".into());
            let shm_dir = Path::new("/dev/shm");
            if shm_dir.is_dir() {
                Some(shm_dir.join(format!("mem_usage_accel{accel_idx}")))
            } else {
                None
            }
        } else {
            None
        };
        let since_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let where_tag = detect_where_tag();
        let load_at_lock = read_loadavg();

        let state = State::new(engine_str, engine_version.into(), source.into(), owner);

        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            write_files: true,
            sink: Mutex::new(None),
            model: Mutex::new(None),
            write_lock: Mutex::new(()),
            wake: (Mutex::new(Wake::default()), Condvar::new()),
            has_writer: AtomicBool::new(false),
            #[cfg(test)]
            last_writer: Mutex::new(None),
            path,
            tmp_path,
            lock_dir,
            holder_path,
            holder_tmp_path,
            shm_path,
            since_unix,
            where_tag,
            load_at_lock,
            lock_logged: Mutex::new(false),
        });
        // The first record and the card-lock holder exist before this returns (the caller opens
        // the device next); every later write is the writer thread's, when there is one.
        flush(&shared, true, false);

        let hb_handle = if heartbeat {
            let s = Arc::clone(&shared);
            thread::Builder::new()
                .name("gaudium-live-hb".into())
                .spawn(move || writer_loop(&s))
                .ok()
        } else {
            None
        };
        shared
            .has_writer
            .store(hb_handle.is_some(), Ordering::Release);

        Self {
            shared: Some(shared),
            heartbeat: hb_handle,
        }
    }

    pub fn disabled() -> Self {
        Self {
            shared: None,
            heartbeat: None,
        }
    }

    /// Whether this reporter writes live records (an in-memory reporter does not).
    pub fn is_enabled(&self) -> bool {
        self.shared.as_ref().is_some_and(|s| s.write_files)
    }

    pub fn path(&self) -> Option<&Path> {
        self.shared
            .as_ref()
            .filter(|s| s.write_files)
            .map(|s| s.path.as_path())
    }

    pub fn set_model(&self, model: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().model = model.into();
            request_write(s, true);
        }
    }

    pub fn set_phase(&self, phase: impl Into<String>, notes: impl Into<String>) {
        if let Some(s) = &self.shared {
            {
                let mut st = s.state.lock().unwrap();
                st.phase = phase.into();
                let n = notes.into();
                if !n.is_empty() {
                    st.notes = n;
                    st.notes_is_status = true;
                }
            }
            request_write(s, true);
        }
    }

    pub fn set_notes(&self, notes: impl Into<String>) {
        if let Some(s) = &self.shared {
            let mut st = s.state.lock().unwrap();
            st.notes = notes.into();
            st.notes_is_status = false;
        }
    }

    /// Sets the 5-category HBM ledger values from exact byte counts (converted to decimal GB).
    pub fn set_hbm_bytes(&self, weights: u64, kv: u64, state: u64, scratch: u64, other: u64) {
        if let Some(s) = &self.shared {
            let total = weights
                .saturating_add(kv)
                .saturating_add(state)
                .saturating_add(scratch)
                .saturating_add(other);
            let mut st = s.state.lock().unwrap();
            st.hbm_used_bytes = Some(total);
            st.hbm_gb = Some([
                weights as f64 / 1e9,
                kv as f64 / 1e9,
                state as f64 / 1e9,
                scratch as f64 / 1e9,
                other as f64 / 1e9,
            ]);
            drop(st);
            request_write(s, true);
        }
    }

    /// Sets the 5-category HBM ledger values in decimal GB (`bytes / 1e9`).
    pub fn set_hbm_gb(
        &self,
        weights_gb: f64,
        kv_gb: f64,
        state_gb: f64,
        scratch_gb: f64,
        other_gb: f64,
    ) {
        if let Some(s) = &self.shared {
            let total_gb = (weights_gb + kv_gb + state_gb + scratch_gb + other_gb).max(0.0);
            let mut st = s.state.lock().unwrap();
            st.hbm_used_bytes = Some((total_gb * 1e9) as u64);
            st.hbm_gb = Some([weights_gb, kv_gb, state_gb, scratch_gb, other_gb]);
            drop(st);
            request_write(s, true);
        }
    }

    /// Sets an optional JSON object for `"hbm_detail"`.
    pub fn set_hbm_detail(&self, json_obj: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().hbm_detail = json_obj.into();
        }
    }

    /// Sets an optional JSON object for `"hbm_kinds"`: under each category and named part of
    /// `hbm_detail`, the part's buffers by kind as `[[kind, GB, buffers], ...]`.
    pub fn set_hbm_kinds(&self, json_obj: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().hbm_kinds = json_obj.into();
        }
    }

    /// Sets an optional JSON object for `"sources"`.
    pub fn set_sources(&self, json_obj: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().sources = json_obj.into();
        }
    }

    /// Sets an optional JSON object for `"profile"` (level-0 HBM flow & step split).
    pub fn set_profile(&self, json_obj: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().profile = json_obj.into();
        }
    }

    /// Records a named component of process host RAM (in decimal GB); `gb <= 0.0` removes it.
    pub fn set_host_part(&self, name: impl Into<String>, gb: f64, file_backed: bool) {
        if let Some(s) = &self.shared {
            let mut st = s.state.lock().unwrap();
            let k = name.into();
            if gb > 0.0 {
                st.host_parts.insert(k, (gb, file_backed));
            } else {
                st.host_parts.remove(&k);
            }
        }
    }

    pub fn begin_request(
        &self,
        prompt_tokens: u64,
        method: impl Into<String>,
        notes: impl Into<String>,
    ) {
        if let Some(s) = &self.shared {
            {
                let mut st = s.state.lock().unwrap();
                st.phase = "prefill".to_string();
                st.prompt_tokens = Some(prompt_tokens);
                st.prompt_cached = 0;
                st.gen_tokens = 0;
                st.first_n = 0;
                st.launches = 0;
                st.drafted = 0;
                st.accepted = 0;
                st.ttft_ms = None;
                st.rate = None;
                st.rate_run = None;
                st.tok_per_launch = None;
                st.ms_per_launch = None;
                st.accept_rate = None;
                st.method = method.into();
                let n = notes.into();
                let has_notes = !n.is_empty();
                if !n.is_empty() {
                    st.notes = n;
                }
                st.t_begin_s = st.origin.elapsed().as_secs_f64();
                st.profile.clear();
                st.samples.clear();
                st.on_begin_request(has_notes);
            }
            request_write(s, true);
        }
    }

    pub fn set_cached(&self, cached_tokens: u64) {
        if let Some(s) = &self.shared {
            {
                let mut st = s.state.lock().unwrap();
                st.prompt_cached = cached_tokens;
                st.prefill_done = st.prefill_done.max(cached_tokens);
            }
            request_write(s, true);
        }
    }

    pub fn first_token(&self, ttft_ms: Option<f64>, tokens: u64) {
        if let Some(s) = &self.shared {
            let ev = {
                let mut st = s.state.lock().unwrap();
                let t = st.origin.elapsed().as_secs_f64();
                st.ttft_ms = Some(ttft_ms.unwrap_or_else(|| (t - st.t_begin_s) * 1e3));
                st.phase = "decode".to_string();
                st.gen_tokens = tokens;
                st.first_n = tokens;
                st.t_first_s = t;
                st.samples.clear();
                st.samples.push_back(StepSample {
                    t_s: t,
                    decode_tokens: 0,
                    launches: 0,
                    drafted: 0,
                    accepted: 0,
                });
                st.on_first_token(t);
                st.launch_event(LaunchKind::First, t, (0, 0, 0))
            };
            notify(s, &ev);
            request_write(s, true);
        }
    }

    pub fn step(&self, emitted: u64, drafted: u64, accepted: u64) {
        let Some(s) = &self.shared else { return };
        // An unmarked launch takes its byte model from the engine's `LaunchModel`, if it set one.
        let modelled = {
            let (unmarked, pos) = {
                let st = s.state.lock().unwrap();
                (
                    st.t_launch_begin.is_none(),
                    st.prompt_tokens.unwrap_or(0) + st.gen_tokens.saturating_sub(1),
                )
            };
            if unmarked {
                let key = s
                    .model
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|m| m.key(drafted, pos));
                key.map(|k| {
                    let ck = k | MODEL_KEY_BIT;
                    (
                        ck,
                        cached_traffic(s, ck, || {
                            s.model
                                .lock()
                                .unwrap()
                                .as_ref()
                                .map_or_else(Traffic::default, |m| m.build(k))
                        }),
                    )
                })
            } else {
                None
            }
        };
        let (due, ev) = {
            let mut st = s.state.lock().unwrap();
            let t = st.origin.elapsed().as_secs_f64();
            if st.t_launch_begin.is_none() {
                st.cur_traffic = modelled;
                if st.cur_traffic.is_some() {
                    st.profile_auto = true;
                }
            }
            if st.samples.is_empty() {
                st.t_first_s = t;
                st.samples.push_back(StepSample {
                    t_s: t,
                    decode_tokens: 0,
                    launches: 0,
                    drafted: 0,
                    accepted: 0,
                });
            }
            st.gen_tokens += emitted;
            st.launches += 1;
            st.drafted += drafted;
            st.accepted += accepted;
            let prev_d = st.samples.back().map_or(0, |b| b.decode_tokens);
            let launches = st.launches;
            let drafted_tot = st.drafted;
            let accepted_tot = st.accepted;
            st.samples.push_back(StepSample {
                t_s: t,
                decode_tokens: prev_d + emitted,
                launches,
                drafted: drafted_tot,
                accepted: accepted_tot,
            });
            while st.samples.len() > 2 && st.samples[1].t_s <= t - 5.0 {
                st.samples.pop_front();
            }
            if st.samples.len() > 4096 {
                st.samples.pop_front();
            }
            let r = *st.samples.front().unwrap();
            let l = *st.samples.back().unwrap();
            let dt = l.t_s - r.t_s;
            let nl = l.launches.saturating_sub(r.launches);
            let dd = l.decode_tokens.saturating_sub(r.decode_tokens);
            let dr = l.drafted.saturating_sub(r.drafted);
            let da = l.accepted.saturating_sub(r.accepted);

            st.rate = if dt > 0.0 { Some(dd as f64 / dt) } else { None };
            st.tok_per_launch = if nl > 0 {
                Some(dd as f64 / nl as f64)
            } else {
                None
            };
            st.ms_per_launch = if nl > 0 {
                Some(dt * 1e3 / nl as f64)
            } else {
                None
            };
            st.accept_rate = if dr > 0 {
                Some(da as f64 / dr as f64)
            } else {
                None
            };
            let run_dt = t - st.t_first_s;
            st.rate_run = if run_dt > 0.0 {
                Some(st.gen_tokens.saturating_sub(st.first_n) as f64 / run_dt)
            } else {
                None
            };
            st.on_step(t);
            st.phase = "decode".to_string();
            if st.launch_log.len() < LAUNCH_LOG_MAX {
                let decode_ms = ((t - st.t_first_s) * 1e3).max(0.0) as f32;
                st.launch_log.push(LaunchRec {
                    decode_ms,
                    emitted: emitted.min(u64::from(u16::MAX)) as u16,
                    drafted: drafted.min(u64::from(u16::MAX)) as u16,
                    accepted: accepted.min(u64::from(u16::MAX)) as u16,
                });
            }
            (
                (t - st.last_write_s) >= 1.0,
                st.launch_event(LaunchKind::Step, t, (emitted, drafted, accepted)),
            )
        };
        notify(s, &ev);
        if due {
            request_write(s, false);
        }
    }

    /// Sets (or clears) the byte model of the launches the decode loop does not mark; see
    /// [`LaunchModel`]. Call it before the request's first `step`.
    pub fn set_launch_model(&self, model: Option<Box<dyn LaunchModel>>) {
        if let Some(s) = &self.shared {
            *s.model.lock().unwrap() = model;
        }
    }

    pub fn end_request(&self) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().phase = "idle".to_string();
            s.state.lock().unwrap().on_end_request("done");
            request_write(s, true);
        }
    }

    /// Like `end_request`, recording how the request ended (`"done"`, `"cancelled"`, `"error"`,
    /// ...) in the record's `outcome` field. Call exactly one of the two on every exit path of a
    /// request (normal end, cancellation, error return), after the last launch is synchronised.
    pub fn end_request_with(&self, outcome: &str) {
        if let Some(s) = &self.shared {
            {
                let mut st = s.state.lock().unwrap();
                st.phase = "idle".to_string();
                st.on_end_request(outcome);
            }
            request_write(s, true);
        }
    }

    /// Changes the method label during a request (e.g. an adaptive speculative mode switching
    /// its draft length); written with the next record.
    pub fn set_method(&self, method: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().method = method.into();
        }
    }

    /// Prefill progress: `done_tokens` prompt tokens of the current request are now in the
    /// model's context (prefix reuse included). Call once per synchronised prefill tile, never
    /// per launch. The record's `context_pos` follows it while the phase is `prefill`, and
    /// `prefill_tok_s` (rolling ~5 s) / `prefill_tok_s_run` (whole prefill so far) are derived
    /// from the call times. Written at most once per second.
    pub fn prefill_progress(&self, done_tokens: u64) {
        let Some(s) = &self.shared else { return };
        let (first, ev) = {
            let mut st = s.state.lock().unwrap();
            let t = st.origin.elapsed().as_secs_f64();
            let first = st.on_prefill(t, done_tokens);
            (first, st.launch_event(LaunchKind::Prefill, t, (0, 0, 0)))
        };
        notify(s, &ev);
        request_write(s, first);
    }

    /// `prefill_progress` plus the byte model of the tile that just finished: its `launches`
    /// launches together moved the traffic `build` returns (built once per `key`, e.g. a key of
    /// tile width and cache chunk count, then reused). Feeds the profile block's `prefill` entry.
    pub fn prefill_tile(
        &self,
        done_tokens: u64,
        launches: u64,
        key: u64,
        build: impl FnOnce() -> Traffic,
    ) {
        let Some(s) = &self.shared else { return };
        let tr = cached_traffic(s, key | PREFILL_KEY_BIT, build);
        let (first, ev) = {
            let mut st = s.state.lock().unwrap();
            let t = st.origin.elapsed().as_secs_f64();
            st.timeline.prefill_launches(&tr, launches);
            st.profile_auto = true;
            let first = st.on_prefill(t, done_tokens);
            (first, st.launch_event(LaunchKind::Prefill, t, (0, 0, 0)))
        };
        notify(s, &ev);
        request_write(s, first);
    }

    /// Marks the host time right before the launch call of a decode step (the verify launch of a
    /// speculative step, or the first launch of a group that one `step` call consumes). Together
    /// with `launch_submitted` and `step` it gives the profile block's host timing: launch call,
    /// wait for the tokens, launch-outstanding time and the idle gap between launches. Costs one
    /// clock read under an uncontended lock; no device call.
    pub fn launch_begin(&self) {
        if let Some(s) = &self.shared {
            let mut st = s.state.lock().unwrap();
            st.profile_auto = true;
            st.t_launch_begin = Some(st.origin.elapsed().as_secs_f64());
            st.t_launch_submitted = None;
        }
    }

    /// `launch_begin` plus the byte model of the launch: `build` returns the traffic of one
    /// launch of the step recipe (built once per `key`, e.g. the recipe's chunk count and draft
    /// length, then reused). Feeds the profile block's HBM GB/s and step split.
    pub fn launch_begin_with(&self, key: u64, build: impl FnOnce() -> Traffic) {
        let Some(s) = &self.shared else { return };
        let same = {
            let st = s.state.lock().unwrap();
            st.cur_traffic.as_ref().is_some_and(|c| c.0 == key)
        };
        if !same {
            let tr = cached_traffic(s, key, build);
            s.state.lock().unwrap().cur_traffic = Some((key, tr));
        }
        self.launch_begin();
    }

    /// Marks the host time right after the launch call returned.
    pub fn launch_submitted(&self) {
        if let Some(s) = &self.shared {
            let mut st = s.state.lock().unwrap();
            st.t_launch_submitted = Some(st.origin.elapsed().as_secs_f64());
        }
    }

    /// The current `profile` block (level-0 timing and byte model), if one has been built.
    pub fn profile_snapshot(&self) -> Option<String> {
        let s = self.shared.as_ref()?;
        let st = s.state.lock().unwrap();
        if st.profile.is_empty() {
            None
        } else {
            Some(st.profile.clone())
        }
    }

    /// A cheap cloneable reader of the measured `profile` block, for consumers that cannot hold
    /// the engine while it generates (e.g. an event stream fed from the token callback).
    pub fn profile_handle(&self) -> ProfileHandle {
        ProfileHandle {
            shared: self.shared.clone(),
        }
    }
}

/// Read-only view of a [`LiveReporter`]'s measured `profile` block.
#[derive(Clone)]
pub struct ProfileHandle {
    shared: Option<Arc<Shared>>,
}

impl ProfileHandle {
    /// The current `profile` JSON object, if the engine has built one.
    pub fn snapshot(&self) -> Option<String> {
        let s = self.shared.as_ref()?;
        let st = s.state.lock().unwrap();
        if st.profile.is_empty() {
            None
        } else {
            Some(st.profile.clone())
        }
    }

    /// Like `snapshot`, after rebuilding the block over the newest launches (the block is
    /// otherwise rebuilt at most every 0.5 s, so a request's last launches may be missing).
    pub fn final_snapshot(&self) -> Option<String> {
        let s = self.shared.as_ref()?;
        let mut st = s.state.lock().unwrap();
        if st.profile_auto && st.timeline.steps() > 0 {
            let t = st.origin.elapsed().as_secs_f64();
            st.timeline.invalidate();
            st.timeline.refresh(t, "decode");
            st.profile = st.timeline.block_str().to_string();
        }
        if st.profile.is_empty() {
            None
        } else {
            Some(st.profile.clone())
        }
    }

    /// Whether the engine's reporter keeps request state (so the launch feed will arrive).
    pub fn is_live(&self) -> bool {
        self.shared.is_some()
    }

    /// Sets (or clears) the receiver of the launch feed. The sink runs on the engine's thread
    /// inside `first_token` / `step`, so it must be quick and must not call back into the
    /// engine. Clear it when the request ends.
    pub fn set_launch_sink(&self, sink: Option<LaunchSink>) {
        if let Some(s) = &self.shared {
            *s.sink.lock().unwrap() = sink;
        }
    }

    /// The current (or last) request's totals, rolling rates and per-launch log.
    pub fn request_summary(&self) -> Option<RequestSummary> {
        self.summary(true)
    }

    /// `request_summary` without the per-launch log (cheap enough to call while decoding).
    pub fn request_rates(&self) -> Option<RequestSummary> {
        self.summary(false)
    }

    fn summary(&self, with_log: bool) -> Option<RequestSummary> {
        let s = self.shared.as_ref()?;
        let st = s.state.lock().unwrap();
        Some(RequestSummary {
            request_seq: st.req_seq,
            phase: st.phase.clone(),
            method: st.method.clone(),
            prompt_tokens: st.prompt_tokens,
            gen_tokens: st.gen_tokens,
            launches: st.launches,
            drafted: st.drafted,
            accepted: st.accepted,
            ttft_ms: st.ttft_ms,
            decode_tok_s: st.rate,
            decode_tok_s_run: st.rate_run,
            tok_per_launch: st.tok_per_launch,
            ms_per_launch: st.ms_per_launch,
            accept_rate: st.accept_rate,
            log: if with_log {
                st.launch_log.clone()
            } else {
                Vec::new()
            },
        })
    }
}

/// Hands one launch event to the registered sink, outside the state lock.
fn notify(s: &Shared, ev: &LaunchEvent) {
    if let Some(f) = s.sink.lock().unwrap().as_mut() {
        f(ev);
    }
}

/// Keeps prefill tile keys apart from decode step keys in the shared traffic cache.
const PREFILL_KEY_BIT: u64 = 1 << 63;

/// Looks `key` up in the reporter's traffic cache, building it outside the state lock on a miss.
fn cached_traffic(s: &Shared, key: u64, build: impl FnOnce() -> Traffic) -> Arc<Traffic> {
    if let Some(tr) = s.state.lock().unwrap().traffic_cache.get(&key) {
        return Arc::clone(tr);
    }
    let tr = Arc::new(build());
    let mut st = s.state.lock().unwrap();
    if st.traffic_cache.len() >= 4096 {
        st.traffic_cache.clear();
    }
    Arc::clone(st.traffic_cache.entry(key).or_insert(tr))
}

fn fmt_eta(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

impl State {
    fn new(engine: String, version: String, source: String, owner: String) -> Self {
        Self {
            engine,
            version,
            source,
            owner,
            model: String::new(),
            method: String::new(),
            notes: String::new(),
            phase: "idle".to_string(),
            hbm_detail: String::new(),
            hbm_kinds: String::new(),
            sources: String::new(),
            profile: String::new(),
            host_parts: BTreeMap::new(),
            pid: std::process::id(),
            prompt_tokens: None,
            prompt_cached: 0,
            gen_tokens: 0,
            first_n: 0,
            launches: 0,
            drafted: 0,
            accepted: 0,
            ttft_ms: None,
            rate: None,
            rate_run: None,
            tok_per_launch: None,
            ms_per_launch: None,
            accept_rate: None,
            origin: Instant::now(),
            t_begin_s: 0.0,
            t_first_s: 0.0,
            last_write_s: -1e9,
            hbm_gb: None,
            hbm_used_bytes: None,
            samples: VecDeque::new(),
            req_seq: 0,
            notes_is_status: false,
            outcome: String::new(),
            prefill_reporting: false,
            prefill_seen: false,
            prefill_done: 0,
            prefill_rate: None,
            prefill_rate_run: None,
            prefill_samples: VecDeque::new(),
            t_launch_begin: None,
            t_launch_submitted: None,
            t_prev_tokens: None,
            traffic_cache: HashMap::new(),
            cur_traffic: None,
            timeline: StepTimeline::default(),
            profile_auto: false,
            launch_log: Vec::new(),
        }
    }

    fn launch_event(&self, kind: LaunchKind, t: f64, launch: (u64, u64, u64)) -> LaunchEvent {
        LaunchEvent {
            kind,
            gen_tokens: self.gen_tokens,
            launches: self.launches,
            drafted: self.drafted,
            accepted: self.accepted,
            launch_emitted: launch.0,
            launch_drafted: launch.1,
            launch_accepted: launch.2,
            decode_ms: if kind == LaunchKind::Prefill {
                0.0
            } else {
                ((t - self.t_first_s) * 1e3).max(0.0)
            },
            ttft_ms: self.ttft_ms.unwrap_or(0.0),
            prefill_pos: self.prefill_done,
            request_ms: ((t - self.t_begin_s) * 1e3).max(0.0),
        }
    }

    fn on_begin_request(&mut self, has_notes: bool) {
        self.req_seq += 1;
        if has_notes {
            self.notes_is_status = false;
        } else if self.notes_is_status {
            // a status note of an earlier phase ("ready", "uploading ...") is not this request's
            self.notes.clear();
            self.notes_is_status = false;
        }
        self.outcome.clear();
        self.prefill_seen = false;
        self.prefill_done = 0;
        self.prefill_rate = None;
        self.prefill_rate_run = None;
        self.prefill_samples.clear();
        self.t_launch_begin = None;
        self.t_launch_submitted = None;
        self.t_prev_tokens = None;
        self.cur_traffic = None;
        self.timeline.reset();
        self.profile_auto = false;
        self.launch_log.clear();
    }

    /// Returns whether this is the request's first progress report (written at once).
    fn on_prefill(&mut self, t: f64, done_tokens: u64) -> bool {
        let first = !self.prefill_seen;
        self.prefill_seen = true;
        self.prefill_reporting = true;
        let prompt = self.prompt_tokens.unwrap_or(u64::MAX);
        let done = done_tokens.min(prompt).max(self.prompt_cached.min(prompt));
        self.prefill_done = done;
        if self.prefill_samples.is_empty() {
            self.prefill_samples
                .push_back((self.t_begin_s, self.prompt_cached.min(done)));
        }
        self.prefill_samples.push_back((t, done));
        while self.prefill_samples.len() > 2 && self.prefill_samples[1].0 <= t - 5.0 {
            self.prefill_samples.pop_front();
        }
        let (rt, rd) = self.prefill_samples[0];
        self.prefill_rate = if t - rt > 0.0 {
            Some(done.saturating_sub(rd) as f64 / (t - rt))
        } else {
            None
        };
        let run_dt = t - self.t_begin_s;
        self.prefill_rate_run = if run_dt > 0.0 {
            Some(done.saturating_sub(self.prompt_cached) as f64 / run_dt)
        } else {
            None
        };
        if self.profile_auto {
            // running prefill entry of the profile block: modelled bytes so far / time so far
            self.timeline.prefill_running(run_dt * 1e3);
            if self.timeline.refresh(t, "prefill") {
                self.profile = self.timeline.block_str().to_string();
            }
        }
        first
    }

    fn on_first_token(&mut self, t: f64) {
        if let Some(p) = self.prompt_tokens {
            self.prefill_done = p;
            if let Some(ttft) = self.ttft_ms.filter(|&x| x > 0.0) {
                // the request's prefill rate: rows actually prefilled / time to first token
                let r = p.saturating_sub(self.prompt_cached) as f64 / (ttft / 1e3);
                self.prefill_rate_run = Some(r);
                self.prefill_rate = Some(r);
            }
        }
        self.prefill_samples.clear();
        self.t_prev_tokens = Some(t);
        self.t_launch_begin = None;
        self.t_launch_submitted = None;
        if self.profile_auto {
            self.timeline.prefill_done(self.ttft_ms.unwrap_or(0.0));
            self.timeline.refresh(t, "decode");
            self.profile = self.timeline.block_str().to_string();
            // the first decode launch replaces the prefill-only block at once
            self.timeline.invalidate();
        }
    }

    fn on_step(&mut self, t: f64) {
        if let Some(tb) = self.t_launch_begin.take() {
            let ts = self.t_launch_submitted.take().unwrap_or(tb);
            let prev = self.t_prev_tokens;
            self.timeline.add(LaunchSample {
                t_s: t,
                wall_ms: (t - prev.unwrap_or(tb)) * 1e3,
                dev_ms: (t - tb) * 1e3,
                gap_ms: prev.map(|p| ((tb - p) * 1e3).max(0.0)),
                enqueue_us: (ts - tb) * 1e6,
                launch_us: (ts - tb) * 1e6,
                wait_ms: (t - ts) * 1e3,
                marked: true,
                traffic: self.cur_traffic.as_ref().map(|c| Arc::clone(&c.1)),
            });
            if self.timeline.refresh(t, "decode") {
                self.profile = self.timeline.block_str().to_string();
            }
        } else if let (Some(prev), Some(cur)) = (self.t_prev_tokens, self.cur_traffic.as_ref()) {
            // no launch marks: the token inter-arrival time is the only timing of this launch
            self.timeline.add(LaunchSample {
                t_s: t,
                wall_ms: (t - prev) * 1e3,
                dev_ms: 0.0,
                gap_ms: None,
                enqueue_us: 0.0,
                launch_us: 0.0,
                wait_ms: 0.0,
                marked: false,
                traffic: Some(Arc::clone(&cur.1)),
            });
            if self.timeline.refresh(t, "decode") {
                self.profile = self.timeline.block_str().to_string();
            }
        }
        self.t_prev_tokens = Some(t);
    }

    fn on_end_request(&mut self, outcome: &str) {
        self.outcome = outcome.to_string();
        self.t_launch_begin = None;
        self.t_launch_submitted = None;
        self.t_prev_tokens = None;
        self.prefill_samples.clear();
    }

    /// Position of the next token in the model's context: the prefill cursor while an engine
    /// that reports prefill progress is prefilling, `prompt + generated` otherwise.
    fn context_pos_now(&self, prompt: u64) -> u64 {
        if self.phase == "prefill" && self.prefill_reporting {
            self.prefill_done.max(self.prompt_cached).min(prompt)
        } else {
            prompt + self.gen_tokens
        }
    }

    /// Optional record fields beyond the base protocol (consumers ignore unknown fields).
    fn push_lifecycle_fields(&self, j: &mut String) {
        j.push_str(&format!(",\"request_seq\":{}", self.req_seq));
        if self.prefill_reporting && self.req_seq > 0 {
            let prompt = self.prompt_tokens.unwrap_or(0);
            let done = self.prefill_done.min(prompt);
            let eta = match (self.phase.as_str(), self.prefill_rate) {
                ("prefill", Some(r)) if r > 0.0 => Some(prompt.saturating_sub(done) as f64 / r),
                _ => None,
            };
            j.push_str(&format!(
                ",\"prefill_pos\":{done},\"prefill_tok_s\":{},\"prefill_tok_s_run\":{},\"prefill_eta_s\":{}",
                json_opt_f64(self.prefill_rate, 1),
                json_opt_f64(self.prefill_rate_run, 1),
                json_opt_f64(eta, 1),
            ));
        }
        if !self.outcome.is_empty() {
            j.push_str(",\"outcome\":");
            j.push_str(&json_quote(&self.outcome));
        }
    }

    /// One-line progress text for the card-lock holder file, when there is something to say.
    fn progress_text(&self) -> Option<String> {
        let prompt = self.prompt_tokens?;
        match self.phase.as_str() {
            "prefill" if self.prefill_reporting => {
                let done = self.prefill_done.min(prompt);
                let pct = if prompt > 0 {
                    100.0 * done as f64 / prompt as f64
                } else {
                    100.0
                };
                let mut s = format!("prefill {done}/{prompt} tok ({pct:.1}%)");
                if let Some(r) = self.prefill_rate.filter(|&r| r > 0.0) {
                    s.push_str(&format!(
                        ", {r:.0} tok/s, ETA {}",
                        fmt_eta(prompt.saturating_sub(done) as f64 / r)
                    ));
                }
                Some(s)
            }
            "prefill" => Some(format!("prefill {prompt} tok")),
            "decode" => {
                let mut s = format!("decode {} tok", self.gen_tokens);
                if let Some(r) = self.rate {
                    s.push_str(&format!(", {r:.1} tok/s"));
                }
                if let Some(tpl) = self.tok_per_launch.filter(|_| self.drafted > 0) {
                    s.push_str(&format!(", {tpl:.2} tok/launch"));
                }
                if let Some(a) = self.accept_rate {
                    s.push_str(&format!(", accept {:.0}%", 100.0 * a));
                }
                s.push_str(&format!(", ctx {}", prompt + self.gen_tokens));
                Some(s)
            }
            "idle" if !self.outcome.is_empty() => {
                let mut s = format!(
                    "idle, last request {}: {} tok",
                    self.outcome, self.gen_tokens
                );
                if let Some(r) = self.rate_run {
                    s.push_str(&format!(" at {r:.1} tok/s"));
                }
                Some(s)
            }
            _ => None,
        }
    }
}

impl Drop for LiveReporter {
    fn drop(&mut self) {
        if let Some(s) = &self.shared {
            let (lk, cv) = &s.wake;
            lk.lock().unwrap().stop = true;
            cv.notify_all();
        }
        if let Some(handle) = self.heartbeat.take() {
            // the thread writes a waiting forced record before it ends
            let _ = handle.join();
        }
        if let Some(s) = &self.shared {
            s.has_writer.store(false, Ordering::Release);
        }
        if let Some(s) = self.shared.as_ref().filter(|s| s.write_files) {
            let _ = fs::remove_file(&s.path);
            let _ = fs::remove_file(&s.tmp_path);
            if let Some(shm) = &s.shm_path {
                let _ = fs::remove_file(shm);
            }
            if let Some(hp) = &s.holder_path {
                let _ = fs::remove_file(hp);
            }
            if let Some(ht) = &s.holder_tmp_path {
                let _ = fs::remove_file(ht);
            }
            if let Some(lock_dir) = &s.lock_dir {
                let now_unix = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                let lr_tmp = lock_dir.join(format!(".last_release.{}.tmp", std::process::id()));
                let lr = lock_dir.join("last_release");
                if let Ok(mut f) = fs::File::create(&lr_tmp) {
                    let _ = writeln!(f, "{now_unix}");
                    let _ = fs::set_permissions(&lr_tmp, fs::Permissions::from_mode(0o666));
                    drop(f);
                    let _ = fs::rename(&lr_tmp, &lr);
                }
                let owner = s.state.lock().unwrap().owner.clone();
                if !owner.is_empty() {
                    append_lock_log(lock_dir, now_unix, &s.where_tag, &format!("UNLOCK {owner}"));
                }
            }
        }
    }
}

fn read_hostname() -> Option<String> {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn detect_where_tag() -> String {
    let in_container = Path::new("/run/systemd/container").exists()
        || fs::read_to_string("/proc/1/environ").is_ok_and(|e| e.contains("container=lxc"));
    if in_container {
        read_hostname().unwrap_or_else(|| "container".to_string())
    } else {
        "host".to_string()
    }
}

fn read_loadavg() -> String {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().map(str::to_string))
        .unwrap_or_else(|| "0.00".to_string())
}

/// Converts Unix epoch seconds to `YYYY-MM-DD HH:MM:SSZ` (UTC) using Howard Hinnant's civil-from-days algorithm.
fn format_utc_timestamp(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let rem = unix_secs % 86_400;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!("{year:04}-{m:02}-{d:02} {hour:02}:{min:02}:{sec:02}Z")
}

fn append_lock_log(lock_dir: &Path, unix_secs: u64, where_tag: &str, msg: &str) {
    let log_path = lock_dir.join("log");
    let ts = format_utc_timestamp(unix_secs);
    if let Ok(mut f) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        let _ = writeln!(f, "{ts} [{where_tag}] {msg}");
        let _ = fs::set_permissions(&log_path, fs::Permissions::from_mode(0o666));
    }
}

fn read_proc_ram_gb() -> Option<[Option<f64>; 5]> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    let keys = ["VmRSS:", "RssAnon:", "RssFile:", "RssShmem:", "VmSwap:"];
    let mut vals = [None; 5];
    for line in text.lines() {
        for (i, key) in keys.iter().enumerate() {
            if let Some(rest) = line.strip_prefix(key) {
                if let Some(num_str) = rest.split_whitespace().next() {
                    if let Ok(kb) = num_str.parse::<f64>() {
                        vals[i] = Some(kb * 1024.0 / 1e9);
                    }
                }
            }
        }
    }
    vals[0].map(|_| vals)
}

fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_opt_f64(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) if x.is_finite() && x >= 0.0 => format!("{x:.prec$}"),
        _ => "null".to_string(),
    }
}

/// A status change is to reach the record. With a writer thread the caller only wakes it (a
/// forced change) or leaves the change to its next periodic write; without one the caller writes.
fn request_write(shared: &Shared, force: bool) {
    if !shared.write_files {
        return;
    }
    match live_write_route(shared.has_writer.load(Ordering::Acquire), force) {
        LiveWrite::LwInline => flush(shared, force, false),
        LiveWrite::LwWake => {
            let (lk, cv) = &shared.wake;
            let mut w = lk.lock().unwrap();
            if !w.forced {
                w.forced = true;
                cv.notify_one();
            }
        }
        LiveWrite::LwDefer => {}
    }
}

/// The writer thread: a forced write as soon as one is asked for, else the 1-second heartbeat
/// write. A forced write that is still waiting when the reporter is dropped is written first.
fn writer_loop(shared: &Shared) {
    let (lk, cv) = &shared.wake;
    let mut w = lk.lock().unwrap();
    loop {
        if w.forced {
            w.forced = false;
            drop(w);
            flush(shared, true, false);
            w = lk.lock().unwrap();
            continue;
        }
        if w.stop {
            break;
        }
        let (guard, timeout) = cv.wait_timeout(w, Duration::from_secs(1)).unwrap();
        w = guard;
        if timeout.timed_out() && !w.forced && !w.stop {
            drop(w);
            flush(shared, false, true);
            w = lk.lock().unwrap();
        }
    }
}

fn flush(shared: &Shared, force: bool, heartbeat: bool) {
    if !shared.write_files {
        return;
    }
    let _wg = shared.write_lock.lock().unwrap();
    #[cfg(test)]
    {
        *shared.last_writer.lock().unwrap() = Some(thread::current().id());
    }
    let host_ram = read_proc_ram_gb();
    let wall_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let now_unix = wall_s as u64;

    let (payload, holder_payload, hbm_used_bytes, log_lock_note) = {
        let mut st = shared.state.lock().unwrap();
        let t = st.origin.elapsed().as_secs_f64();
        if !force && !heartbeat && (t - st.last_write_s) < 1.0 {
            return;
        }
        if heartbeat && (t - st.last_write_s) < 0.5 {
            return;
        }
        st.last_write_s = t;

        let mut j = String::with_capacity(768);
        j.push_str("{\"schema\":1,\"engine\":");
        j.push_str(&json_quote(&st.engine));
        j.push_str(",\"engine_version\":");
        j.push_str(&json_quote(&st.version));
        j.push_str(",\"source\":");
        j.push_str(&json_quote(&st.source));
        j.push_str(",\"owner\":");
        j.push_str(&json_quote(&st.owner));
        j.push_str(&format!(
            ",\"pid\":{},\"ts\":{:.3},\"phase\":",
            st.pid, wall_s
        ));
        j.push_str(&json_quote(&st.phase));

        if !st.model.is_empty() {
            j.push_str(",\"model\":");
            j.push_str(&json_quote(&st.model));
        }
        if !st.method.is_empty() {
            j.push_str(",\"method\":");
            j.push_str(&json_quote(&st.method));
        }

        match st.prompt_tokens {
            Some(p) => j.push_str(&format!(",\"prompt_tokens\":{p}")),
            None => j.push_str(",\"prompt_tokens\":null"),
        }
        j.push_str(&format!(
            ",\"prompt_cached\":{},\"gen_tokens\":{}",
            st.prompt_cached, st.gen_tokens
        ));
        match st.prompt_tokens {
            Some(p) => j.push_str(&format!(",\"context_pos\":{}", st.context_pos_now(p))),
            None => j.push_str(",\"context_pos\":null"),
        }

        j.push_str(&format!(
            ",\"decode_tok_s\":{},\"decode_tok_s_run\":{},\"ttft_ms\":{}",
            json_opt_f64(st.rate, 2),
            json_opt_f64(st.rate_run, 2),
            json_opt_f64(st.ttft_ms, 1),
        ));
        j.push_str(&format!(
            ",\"tok_per_launch\":{},\"ms_per_launch\":{},\"accept_rate\":{}",
            json_opt_f64(st.tok_per_launch, 3),
            json_opt_f64(st.ms_per_launch, 2),
            json_opt_f64(st.accept_rate, 3),
        ));
        let running = if st.phase == "idle" { 0 } else { 1 };
        j.push_str(&format!(",\"requests_running\":{running}"));
        st.push_lifecycle_fields(&mut j);

        if let Some(h) = st.hbm_gb {
            j.push_str(&format!(
                ",\"hbm\":{{\"weights_gb\":{},\"kv_gb\":{},\"state_gb\":{},\"scratch_gb\":{},\"other_gb\":{}}}",
                json_opt_f64(Some(h[0]), 3),
                json_opt_f64(Some(h[1]), 3),
                json_opt_f64(Some(h[2]), 3),
                json_opt_f64(Some(h[3]), 3),
                json_opt_f64(Some(h[4]), 3),
            ));
            if !st.hbm_detail.is_empty() {
                j.push_str(",\"hbm_detail\":");
                j.push_str(&st.hbm_detail);
            }
            if !st.hbm_kinds.is_empty() {
                j.push_str(",\"hbm_kinds\":");
                j.push_str(&st.hbm_kinds);
            }
        }
        if !st.sources.is_empty() {
            j.push_str(",\"sources\":");
            j.push_str(&st.sources);
        }
        if let Some(hr) = host_ram {
            j.push_str(&format!(
                ",\"host_ram\":{{\"rss_gb\":{},\"anon_gb\":{},\"file_gb\":{},\"shmem_gb\":{},\"swap_gb\":{},\"parts\":{{",
                json_opt_f64(hr[0], 3),
                json_opt_f64(hr[1], 3),
                json_opt_f64(hr[2], 3),
                json_opt_f64(hr[3], 3),
                json_opt_f64(hr[4], 3),
            ));
            for (idx, (k, &(gb, file_backed))) in st.host_parts.iter().enumerate() {
                if idx > 0 {
                    j.push(',');
                }
                j.push_str(&format!(
                    "{}:{{\"gb\":{},\"file\":{}}}",
                    json_quote(k),
                    json_opt_f64(Some(gb), 3),
                    if file_backed { "true" } else { "false" }
                ));
            }
            j.push_str("}}");
        }
        if !st.profile.is_empty() {
            j.push_str(",\"profile\":");
            j.push_str(&st.profile);
        }
        j.push_str(",\"notes\":");
        j.push_str(&json_quote(&st.notes));
        j.push_str("}\n");

        let model_label = if st.model.is_empty() {
            st.engine.as_str()
        } else {
            st.model.as_str()
        };
        let note_line = if st.notes.is_empty() {
            format!("{model_label} ({})", st.phase)
        } else {
            format!("{model_label} ({}: {})", st.phase, st.notes)
        };
        let progress_line = if st.notes.is_empty() {
            st.phase.clone()
        } else {
            format!("{}: {}", st.phase, st.notes)
        };
        let progress_line = st.progress_text().unwrap_or(progress_line);
        let holder = if shared.holder_path.is_some() {
            Some(format!(
                "owner={}\nwhere={}\npid={}\nsince={}\nlease_until={}\nnote={}\nload_at_lock={}\nlane=normal\nprogress={}\n",
                st.owner,
                shared.where_tag,
                st.pid,
                shared.since_unix,
                now_unix + 1800,
                note_line.replace('\n', " "),
                shared.load_at_lock,
                progress_line.replace('\n', " "),
            ))
        } else {
            None
        };
        let log_lock = if shared.lock_dir.is_some() && !st.model.is_empty() {
            let mut logged = shared.lock_logged.lock().unwrap();
            if !*logged {
                *logged = true;
                Some((st.owner.clone(), note_line.replace('\n', " ")))
            } else {
                None
            }
        } else {
            None
        };

        (j, holder, st.hbm_used_bytes, log_lock)
    };

    if let Ok(mut f) = fs::File::create(&shared.tmp_path) {
        let ok = f.write_all(payload.as_bytes()).is_ok();
        let _ = fs::set_permissions(&shared.tmp_path, fs::Permissions::from_mode(0o666));
        drop(f);
        if ok {
            let _ = fs::rename(&shared.tmp_path, &shared.path);
        } else {
            let _ = fs::remove_file(&shared.tmp_path);
        }
    }

    if let (Some(hp), Some(ht), Some(hbody)) =
        (&shared.holder_path, &shared.holder_tmp_path, holder_payload)
    {
        if let Ok(mut f) = fs::File::create(ht) {
            let ok = f.write_all(hbody.as_bytes()).is_ok();
            let _ = fs::set_permissions(ht, fs::Permissions::from_mode(0o666));
            drop(f);
            if ok {
                let _ = fs::rename(ht, hp);
            } else {
                let _ = fs::remove_file(ht);
            }
        }
    }

    if let (Some(lock_dir), Some((owner, note))) = (&shared.lock_dir, log_lock_note) {
        append_lock_log(
            lock_dir,
            now_unix,
            &shared.where_tag,
            &format!(
                "LOCK {owner} lease 30m load {} waited=0 note: {note}",
                shared.load_at_lock
            ),
        );
    }

    if let (Some(shm_path), Some(used_bytes)) = (&shared.shm_path, hbm_used_bytes) {
        let mut shm_buf = [0u8; 24];
        shm_buf[0..8].copy_from_slice(&1u64.to_ne_bytes());
        shm_buf[8..16].copy_from_slice(&now_unix.to_ne_bytes());
        shm_buf[16..24].copy_from_slice(&used_bytes.to_ne_bytes());
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(shm_path)
        {
            let _ = f.write_all(&shm_buf);
            let _ = fs::set_permissions(shm_path, fs::Permissions::from_mode(0o644));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn writes_schema_v1_json_and_unlinks_on_drop() {
        let root = env::temp_dir().join(format!("gaudium_live_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let dir = root.join("live");
        fs::create_dir_all(&dir).unwrap();
        let holder_path = root.join("holder");
        let log_path = root.join("log");
        let last_release_path = root.join("last_release");

        let record_path;
        {
            let rep = LiveReporter::in_dir(&dir, "gaudium-rust", "v0.1", "engine", false);
            assert!(rep.is_enabled());
            record_path = rep.path().unwrap().to_path_buf();
            assert!(record_path.exists());
            assert!(
                holder_path.exists(),
                "cardlock holder must be created while engine is active"
            );

            rep.set_model("Qwen3.8-27B");
            rep.set_hbm_bytes(
                50_000_000_000,
                500_000_000,
                100_000_000,
                400_000_000,
                256_000_000,
            );
            rep.set_hbm_detail("{\"other_gb\":{\"workspace\":0.256}}");
            rep.set_hbm_kinds("{\"weights_gb\":{\"target\":[[\"W_up\",30.5,64]]}}");
            rep.begin_request(39, "K=0 BF16", "code");
            rep.first_token(Some(35.2), 1);
            thread::sleep(Duration::from_millis(5));
            rep.step(1, 0, 0);
            rep.end_request();

            let raw = fs::read_to_string(&record_path).unwrap();
            let v = json::parse_json(&raw).expect("valid JSON output");
            assert_eq!(v.get("schema").and_then(|x| x.as_u64()), Some(1));
            assert_eq!(
                v.get("engine").and_then(|x| x.as_str()),
                Some("gaudium-rust")
            );
            assert_eq!(v.get("model").and_then(|x| x.as_str()), Some("Qwen3.8-27B"));
            assert_eq!(v.get("method").and_then(|x| x.as_str()), Some("K=0 BF16"));
            assert_eq!(v.get("phase").and_then(|x| x.as_str()), Some("idle"));
            assert_eq!(v.get("prompt_tokens").and_then(|x| x.as_u64()), Some(39));
            assert_eq!(
                v.get("hbm_detail")
                    .and_then(|d| d.get("other_gb"))
                    .and_then(|o| o.get("workspace"))
                    .and_then(|x| x.as_f64()),
                Some(0.256)
            );
            let kinds = v
                .get("hbm_kinds")
                .and_then(|d| d.get("weights_gb"))
                .and_then(|o| o.get("target"))
                .and_then(|a| a.as_array())
                .expect("hbm_kinds is written beside hbm_detail");
            assert_eq!(kinds[0].as_array().unwrap()[0].as_str(), Some("W_up"));
            assert_eq!(v.get("gen_tokens").and_then(|x| x.as_u64()), Some(2));
            assert_eq!(v.get("context_pos").and_then(|x| x.as_u64()), Some(41));
            let hbm = v.get("hbm").expect("hbm object present");
            assert!((hbm.get("weights_gb").unwrap().as_f64().unwrap() - 50.0).abs() < 1e-3);
            assert!((hbm.get("kv_gb").unwrap().as_f64().unwrap() - 0.5).abs() < 1e-3);

            let holder_txt = fs::read_to_string(&holder_path).unwrap();
            assert!(holder_txt.contains("owner="));
            assert!(holder_txt.contains("Qwen3.8-27B"));
        }

        assert!(
            !record_path.exists(),
            "live status file must be unlinked on drop"
        );
        assert!(
            !holder_path.exists(),
            "cardlock holder must be unlinked on drop"
        );
        assert!(
            last_release_path.exists(),
            "last_release timestamp must be written on drop"
        );
        let log_txt = fs::read_to_string(&log_path).unwrap();
        assert!(log_txt.contains(" LOCK ") && log_txt.contains(" UNLOCK "));
        let _ = fs::remove_dir_all(&root);
    }

    fn read_record(path: &Path) -> json::JsonValue {
        json::parse_json(&fs::read_to_string(path).unwrap()).expect("valid JSON record")
    }

    fn holder_progress(root: &Path) -> String {
        fs::read_to_string(root.join("holder"))
            .unwrap()
            .lines()
            .find_map(|l| l.strip_prefix("progress=").map(str::to_string))
            .unwrap_or_default()
    }

    fn tiny_traffic(bytes: f64) -> Traffic {
        let mut tr = Traffic::default();
        tr.bytes[0][0] = bytes * 0.75;
        tr.bytes[2][0] = bytes * 0.25;
        tr.read = bytes;
        tr.flops[0] = 1e9;
        tr.nodes = 2;
        tr
    }

    /// Rewinds the once-per-second write gate so the next non-forced write goes out.
    fn open_write_gate(rep: &LiveReporter) {
        let mut st = rep.shared.as_ref().unwrap().state.lock().unwrap();
        st.last_write_s = -1e9;
        st.timeline.invalidate();
    }

    #[test]
    fn request_lifecycle_reports_prefill_progress_decode_rates_and_idle() {
        let root = env::temp_dir().join(format!("gaudium_live_lifecycle_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let dir = root.join("live");
        fs::create_dir_all(&dir).unwrap();
        let rep = LiveReporter::in_dir(&dir, "gaudium-rust", "v0.1", "engine", false);
        let path = rep.path().unwrap().to_path_buf();
        let u = |v: &json::JsonValue, k: &str| v.get(k).and_then(|x| x.as_u64());
        let f = |v: &json::JsonValue, k: &str| v.get(k).and_then(|x| x.as_f64());
        let st =
            |v: &json::JsonValue, k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);

        // ---- load -> idle
        rep.set_model("Qwen3-8B");
        rep.set_phase("loading", "uploading weights to HBM");
        assert_eq!(st(&read_record(&path), "phase").as_deref(), Some("loading"));
        rep.set_phase("idle", "ready");
        let v = read_record(&path);
        assert_eq!(st(&v, "phase").as_deref(), Some("idle"));
        assert_eq!(u(&v, "requests_running"), Some(0));
        assert_eq!(u(&v, "request_seq"), Some(0));
        assert!(
            v.get("prefill_pos").is_none(),
            "no prefill fields before a request"
        );

        // ---- request 1: prefill of 1000 tokens in tiles of 256
        rep.begin_request(1000, "K=0 BF16 greedy", "");
        let v = read_record(&path);
        assert_eq!(st(&v, "phase").as_deref(), Some("prefill"));
        assert_eq!(u(&v, "request_seq"), Some(1));
        assert_eq!(u(&v, "gen_tokens"), Some(0));
        assert_eq!(
            st(&v, "notes").as_deref(),
            Some(""),
            "the load status note is not the request's note"
        );
        assert!(v.get("profile").is_none());

        let mut builds = 0u32;
        for done in [256u64, 512, 768, 1000] {
            thread::sleep(Duration::from_millis(4));
            open_write_gate(&rep);
            rep.prefill_tile(done, 37, (256 << 16) | 1, || {
                builds += 1;
                tiny_traffic(1e9)
            });
            let v = read_record(&path);
            assert_eq!(st(&v, "phase").as_deref(), Some("prefill"));
            assert_eq!(u(&v, "prefill_pos"), Some(done));
            assert_eq!(
                u(&v, "context_pos"),
                Some(done),
                "context follows the prefill cursor"
            );
            assert_eq!(u(&v, "prompt_tokens"), Some(1000));
            let run = f(&v, "prefill_tok_s_run").expect("running prefill rate");
            let win = f(&v, "prefill_tok_s").expect("windowed prefill rate");
            assert!(run > 0.0 && win > 0.0);
            assert!(v.get("decode_tok_s").unwrap().as_f64().is_none());
            assert!(v.get("ttft_ms").unwrap().as_f64().is_none());
            if done < 1000 {
                assert!(f(&v, "prefill_eta_s").is_some(), "an ETA while prefilling");
            }
            let pf = v
                .get("profile")
                .and_then(|p| p.get("prefill"))
                .expect("running entry");
            assert!(f(pf, "gb").unwrap() >= 1.0 && f(pf, "gbps").unwrap() > 0.0);
            assert!(v.get("profile").unwrap().get("busy_pct").is_none());
        }
        assert_eq!(builds, 1, "the tile byte model is built once per key");
        let p = holder_progress(&root);
        assert!(
            p.starts_with("prefill 1000/1000 tok (100.0%)"),
            "progress line: {p}"
        );

        // ---- first token: prefill -> decode
        rep.first_token(Some(500.0), 1);
        let v = read_record(&path);
        assert_eq!(st(&v, "phase").as_deref(), Some("decode"));
        assert_eq!(f(&v, "ttft_ms"), Some(500.0));
        assert_eq!(u(&v, "gen_tokens"), Some(1));
        assert_eq!(u(&v, "context_pos"), Some(1001));
        assert_eq!(
            f(&v, "prefill_tok_s_run"),
            Some(2000.0),
            "1000 tokens / 0.5 s"
        );
        assert!(v.get("prefill_eta_s").unwrap().as_f64().is_none());
        let pf = v
            .get("profile")
            .and_then(|p| p.get("prefill"))
            .expect("prefill entry");
        assert_eq!(u(pf, "launches"), Some(4 * 37));
        assert_eq!(f(pf, "gb"), Some(4.0));
        assert_eq!(f(pf, "gbps"), Some(8.0));

        // ---- decode: 40 greedy launches
        let mut step_builds = 0u32;
        for _ in 0..40 {
            rep.launch_begin_with(1, || {
                step_builds += 1;
                tiny_traffic(2e9)
            });
            thread::sleep(Duration::from_micros(200));
            rep.launch_submitted();
            thread::sleep(Duration::from_millis(2));
            rep.step(1, 0, 0);
            thread::sleep(Duration::from_micros(300));
        }
        assert_eq!(step_builds, 1, "the step byte model is built once per key");
        open_write_gate(&rep);
        rep.launch_begin_with(1, || tiny_traffic(2e9));
        rep.launch_submitted();
        thread::sleep(Duration::from_millis(2));
        rep.step(1, 0, 0);
        let v = read_record(&path);
        assert_eq!(st(&v, "phase").as_deref(), Some("decode"));
        assert_eq!(u(&v, "gen_tokens"), Some(42));
        assert_eq!(u(&v, "context_pos"), Some(1042));
        let rate = f(&v, "decode_tok_s").unwrap();
        let rate_run = f(&v, "decode_tok_s_run").unwrap();
        assert!(rate > 50.0 && rate < 500.0, "decode rate {rate}");
        assert!(
            (rate - rate_run).abs() / rate < 0.01,
            "a young request: window = whole run"
        );
        assert_eq!(f(&v, "tok_per_launch"), Some(1.0));
        let ms = f(&v, "ms_per_launch").unwrap();
        assert!(
            (ms * rate / 1000.0 - 1.0).abs() < 0.01,
            "ms/launch x tok/s = tok/launch"
        );
        assert!(
            v.get("accept_rate").unwrap().as_f64().is_none(),
            "greedy: no acceptance"
        );
        let pr = v.get("profile").expect("profile block");
        let step_ms = f(pr, "step_ms").unwrap();
        let dev_ms = f(pr, "dev_ms").unwrap();
        assert!(
            dev_ms > 2.0 && dev_ms <= step_ms,
            "dev {dev_ms} step {step_ms}"
        );
        let busy = f(pr, "busy_pct").unwrap();
        assert!((busy + f(pr, "idle_pct").unwrap() - 100.0).abs() < 0.02);
        assert!(f(pr, "gap_ms_mean").unwrap() >= 0.3);
        let host = pr.get("host").unwrap();
        assert!(f(host, "launch_us").unwrap() >= 200.0);
        assert!(f(host, "wait_ms").unwrap() >= 2.0);
        let hbm = pr.get("hbm").unwrap();
        assert_eq!(f(hbm, "gb_per_step"), Some(2.0));
        let gbps = f(hbm, "gbps").unwrap();
        assert!(
            (gbps * step_ms / 1000.0 - 2.0).abs() < 0.05,
            "GB/s x step = GB/step"
        );
        assert!((f(hbm, "weights_gbps").unwrap() - gbps).abs() < 0.11);
        let split = pr.get("split_ms").unwrap();
        assert!((f(split, "verify").unwrap() - 0.75 * dev_ms).abs() < 0.01);
        assert!((f(split, "readout").unwrap() - 0.25 * dev_ms).abs() < 0.01);
        assert_eq!(f(split, "drafts"), Some(0.0));
        assert!(
            pr.get("prefill").is_some(),
            "the prefill entry stays during decode"
        );
        assert!(holder_progress(&root).starts_with("decode 42 tok, "));

        // ---- end: decode -> idle, last numbers kept
        rep.end_request();
        let v = read_record(&path);
        assert_eq!(st(&v, "phase").as_deref(), Some("idle"));
        assert_eq!(u(&v, "requests_running"), Some(0));
        assert_eq!(st(&v, "outcome").as_deref(), Some("done"));
        assert_eq!(u(&v, "gen_tokens"), Some(42));
        assert!(f(&v, "decode_tok_s_run").is_some());
        assert!(holder_progress(&root).starts_with("idle, last request done: 42 tok at "));

        // ---- request 2: speculative steps, then cancelled
        rep.begin_request(10, "spec-chain K=4", "story");
        let v = read_record(&path);
        assert_eq!(st(&v, "phase").as_deref(), Some("prefill"));
        assert_eq!(u(&v, "request_seq"), Some(2));
        assert_eq!(u(&v, "context_pos"), Some(0), "nothing prefilled yet");
        assert!(v.get("outcome").is_none());
        assert!(
            v.get("profile").is_none(),
            "the previous request's profile is gone"
        );
        assert!(v.get("decode_tok_s").unwrap().as_f64().is_none());
        rep.prefill_progress(10);
        rep.first_token(None, 1);
        for accepted in [4u64, 0, 2, 2] {
            thread::sleep(Duration::from_millis(2));
            rep.launch_begin();
            rep.launch_submitted();
            rep.step(accepted + 1, 4, accepted);
        }
        rep.set_method("spec-chain K=2");
        rep.end_request_with("cancelled");
        let v = read_record(&path);
        assert_eq!(st(&v, "phase").as_deref(), Some("idle"));
        assert_eq!(st(&v, "outcome").as_deref(), Some("cancelled"));
        assert_eq!(st(&v, "method").as_deref(), Some("spec-chain K=2"));
        assert_eq!(st(&v, "notes").as_deref(), Some("story"));
        assert_eq!(u(&v, "gen_tokens"), Some(1 + 12));
        assert_eq!(f(&v, "tok_per_launch"), Some(3.0));
        assert_eq!(f(&v, "accept_rate"), Some(0.5), "8 accepted of 16 drafted");
        let pr = v.get("profile").expect("timing without a byte model");
        assert!(pr.get("busy_pct").is_some() && pr.get("hbm").is_none());

        // ---- request 3: cancelled during prefill, straight back to idle
        rep.begin_request(5000, "K=0 BF16 greedy", "");
        rep.prefill_progress(256);
        rep.end_request_with("cancelled");
        let v = read_record(&path);
        assert_eq!(st(&v, "phase").as_deref(), Some("idle"));
        assert_eq!(u(&v, "prefill_pos"), Some(256));
        assert_eq!(
            u(&v, "context_pos"),
            Some(5000),
            "idle: prompt + generated, as before"
        );
        assert_eq!(u(&v, "gen_tokens"), Some(0));
        assert!(v.get("ttft_ms").unwrap().as_f64().is_none());

        drop(rep);
        let _ = fs::remove_dir_all(&root);
    }

    struct FixedModel(f64);
    impl LaunchModel for FixedModel {
        fn key(&self, drafted: u64, pos: u64) -> Option<u64> {
            (drafted > 0).then_some((drafted << 16) | (pos / 4096))
        }
        fn build(&self, key: u64) -> Traffic {
            tiny_traffic(self.0 * (key >> 16) as f64)
        }
    }

    #[test]
    fn in_memory_reporter_feeds_launches_and_models_unmarked_steps_without_files() {
        let rep = LiveReporter::in_memory("gaudium-rust", "v0.1", "engine");
        assert!(!rep.is_enabled(), "an in-memory reporter writes no record");
        assert!(rep.path().is_none());
        let h = rep.profile_handle();
        assert!(h.is_live());
        let seen: Arc<Mutex<Vec<LaunchEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let seen2 = Arc::clone(&seen);
        h.set_launch_sink(Some(Box::new(move |ev| seen2.lock().unwrap().push(*ev))));
        rep.set_launch_model(Some(Box::new(FixedModel(1e9))));

        rep.begin_request(100, "dspine K=15,7 FP8 greedy", "");
        rep.prefill_progress(64);
        rep.first_token(Some(250.0), 1);
        // three speculative launches without launch marks: (emitted, drafted, accepted)
        for (e, d, a) in [(4u64, 7u64, 3u64), (1, 7, 0), (16, 15, 15)] {
            thread::sleep(Duration::from_millis(3));
            rep.step(e, d, a);
        }
        h.set_launch_sink(None);
        rep.step(2, 7, 1); // after the sink is cleared: logged, not delivered

        let evs = seen.lock().unwrap().clone();
        assert_eq!(evs[0].kind, LaunchKind::Prefill);
        assert_eq!((evs[0].prefill_pos, evs[0].gen_tokens), (64, 0));
        let evs = evs[1..].to_vec();
        assert_eq!(evs.len(), 4);
        assert_eq!(
            evs[0].prefill_pos, 100,
            "the whole prompt is in context at the first token"
        );
        assert_eq!(evs[0].kind, LaunchKind::First);
        assert_eq!((evs[0].gen_tokens, evs[0].launches), (1, 0));
        assert!((evs[0].ttft_ms - 250.0).abs() < 1e-9);
        assert_eq!(evs[1].kind, LaunchKind::Step);
        assert_eq!((evs[1].gen_tokens, evs[1].launches), (5, 1));
        assert_eq!((evs[1].launch_emitted, evs[1].launch_accepted), (4, 3));
        assert_eq!((evs[3].gen_tokens, evs[3].launches), (22, 3));
        assert_eq!((evs[3].drafted, evs[3].accepted), (29, 18));
        assert!(evs[3].decode_ms > evs[1].decode_ms && evs[1].decode_ms > 0.0);

        let sum = h.request_summary().unwrap();
        assert_eq!(sum.gen_tokens, 24);
        assert_eq!(sum.launches, 4);
        assert_eq!(
            sum.log.iter().map(|l| l.emitted).collect::<Vec<_>>(),
            vec![4, 1, 16, 2]
        );
        assert_eq!(
            sum.log.iter().map(|l| l.accepted).collect::<Vec<_>>(),
            vec![3, 0, 15, 1]
        );
        assert_eq!(sum.ttft_ms, Some(250.0));
        assert!(sum.accept_rate.is_some() && sum.tok_per_launch.is_some());

        // the unmarked launches still give an HBM flow and a split over the step time
        let prof = json::parse_json(&h.final_snapshot().unwrap()).unwrap();
        assert_eq!(prof.get("window_steps").and_then(|x| x.as_u64()), Some(4));
        assert!(prof.get("dev_ms").is_none() && prof.get("busy_pct").is_none());
        let hbm = prof.get("hbm").expect("modelled HBM flow");
        let gb = hbm.get("gb_per_step").and_then(|x| x.as_f64()).unwrap();
        // launches of 7, 7, 15 and 7 drafts at 1 GB per draft
        assert!((gb - 9.0).abs() < 1e-6, "gb_per_step {gb}");
        assert!(prof.get("split_ms").is_some());

        // a new request starts an empty log
        rep.end_request();
        rep.begin_request(10, "K=0 BF16 greedy", "");
        assert!(h.request_summary().unwrap().log.is_empty());
    }

    #[test]
    fn marked_launches_ignore_the_launch_model() {
        let rep = LiveReporter::in_memory("gaudium-rust", "v0.1", "engine");
        let h = rep.profile_handle();
        rep.set_launch_model(Some(Box::new(FixedModel(1e9))));
        rep.begin_request(10, "K=0 BF16 greedy", "");
        rep.first_token(Some(20.0), 1);
        for _ in 0..3 {
            rep.launch_begin_with(1, || tiny_traffic(5e9));
            rep.launch_submitted();
            thread::sleep(Duration::from_millis(2));
            rep.step(1, 0, 0);
        }
        let prof = json::parse_json(&h.final_snapshot().unwrap()).unwrap();
        assert!(prof.get("dev_ms").is_some() && prof.get("busy_pct").is_some());
        let gb = prof
            .get("hbm")
            .and_then(|x| x.get("gb_per_step"))
            .and_then(|x| x.as_f64())
            .unwrap();
        assert!((gb - 5.0).abs() < 1e-6);
    }

    #[test]
    fn a_reporter_without_a_directory_is_a_no_op_unless_in_memory_is_enabled() {
        let rep = LiveReporter::disabled();
        assert!(!rep.profile_handle().is_live());
        rep.begin_request(1, "x", "");
        rep.first_token(None, 1);
        rep.step(1, 0, 0);
        assert!(rep.profile_handle().request_summary().is_none());
        assert!(rep.profile_handle().final_snapshot().is_none());
    }

    #[test]
    fn engines_without_progress_reports_keep_the_base_protocol_fields() {
        let root = env::temp_dir().join(format!("gaudium_live_legacy_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let dir = root.join("live");
        fs::create_dir_all(&dir).unwrap();
        let rep = LiveReporter::in_dir(&dir, "gaudium-rust", "v0.1", "engine", false);
        let path = rep.path().unwrap().to_path_buf();
        rep.begin_request(300, "greedy", "explain");
        let v = read_record(&path);
        assert_eq!(v.get("context_pos").and_then(|x| x.as_u64()), Some(300));
        assert!(v.get("prefill_pos").is_none());
        rep.first_token(Some(100.0), 1);
        rep.step(1, 0, 0);
        rep.end_request();
        let v = read_record(&path);
        assert_eq!(v.get("context_pos").and_then(|x| x.as_u64()), Some(302));
        assert!(
            v.get("profile").is_none(),
            "no launch marks, no profile block"
        );
        assert!(rep.profile_snapshot().is_none());
        drop(rep);
        let _ = fs::remove_dir_all(&root);
    }

    /// Polls `cond` for up to 5 s: the writer thread writes shortly after it is woken.
    fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(5) {
            if cond() {
                return true;
            }
            thread::sleep(Duration::from_millis(2));
        }
        cond()
    }

    #[test]
    fn status_writes_go_to_the_writer_thread_when_there_is_one() {
        // today's routing (the laws only bound it)
        assert!(matches!(live_write_route(true, true), LiveWrite::LwWake));
        assert!(matches!(live_write_route(true, false), LiveWrite::LwDefer));
        assert!(matches!(
            live_write_route(false, true),
            LiveWrite::LwInline
        ));
        assert!(matches!(
            live_write_route(false, false),
            LiveWrite::LwInline
        ));
    }

    #[test]
    fn a_reporter_with_a_writer_thread_never_writes_on_the_reporting_thread() {
        let root = env::temp_dir().join(format!("gaudium_live_writer_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let dir = root.join("live");
        fs::create_dir_all(&dir).unwrap();
        let me = thread::current().id();

        let rep = LiveReporter::in_dir(&dir, "gaudium-rust", "v0.1", "engine", true);
        let path = rep.path().unwrap().to_path_buf();
        let shared = Arc::clone(rep.shared.as_ref().unwrap());
        // the first record and the card-lock holder are written before the constructor returns
        assert!(path.exists());
        assert!(root.join("holder").exists());
        assert_eq!(*shared.last_writer.lock().unwrap(), Some(me));
        assert!(shared.has_writer.load(Ordering::Acquire));

        // one whole request, every call of it on this thread
        rep.set_model("Qwen3.8-27B");
        rep.begin_request(39, "K=0 BF16", "");
        rep.prefill_progress(0);
        rep.prefill_progress(39);
        rep.first_token(Some(35.2), 1);
        rep.step(1, 0, 0);
        rep.end_request();

        // the newest state reaches the file, and it was not this thread that wrote it
        let newest = || {
            let Ok(raw) = fs::read_to_string(&path) else {
                return false;
            };
            let Ok(v) = json::parse_json(&raw) else {
                return false;
            };
            v.get("model").and_then(|x| x.as_str()) == Some("Qwen3.8-27B")
                && v.get("phase").and_then(|x| x.as_str()) == Some("idle")
                && v.get("request_seq").and_then(|x| x.as_u64()) == Some(1)
                && v.get("outcome").and_then(|x| x.as_str()) == Some("done")
        };
        assert!(wait_for(newest), "the writer thread writes the newest state");
        let writer = shared.last_writer.lock().unwrap().expect("a write");
        assert_ne!(writer, me, "a request's thread does no file write");

        drop(rep);
        assert!(!path.exists(), "the record is removed on drop");
        assert!(!shared.has_writer.load(Ordering::Acquire));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_reporter_without_a_writer_thread_writes_on_the_reporting_thread() {
        let root = env::temp_dir().join(format!("gaudium_live_inline_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let dir = root.join("live");
        fs::create_dir_all(&dir).unwrap();
        let rep = LiveReporter::in_dir(&dir, "gaudium-rust", "v0.1", "engine", false);
        let path = rep.path().unwrap().to_path_buf();
        let shared = Arc::clone(rep.shared.as_ref().unwrap());
        assert!(!shared.has_writer.load(Ordering::Acquire));
        rep.set_phase("load", "weights");
        // written before the call returned
        let v = read_record(&path);
        assert_eq!(v.get("phase").and_then(|x| x.as_str()), Some("load"));
        assert_eq!(
            *shared.last_writer.lock().unwrap(),
            Some(thread::current().id())
        );
        drop(rep);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_forced_write_still_waiting_at_drop_is_written_first() {
        let root = env::temp_dir().join(format!("gaudium_live_drop_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let dir = root.join("live");
        fs::create_dir_all(&dir).unwrap();
        for _ in 0..20 {
            let _ = fs::remove_file(root.join("log"));
            {
                let rep = LiveReporter::in_dir(&dir, "gaudium-rust", "v0.1", "engine", true);
                // the card-lock log gets its LOCK line with the first record that names a model
                rep.set_model("Qwen3.8-27B");
            }
            let log = fs::read_to_string(root.join("log")).expect("lock log");
            let lock = log.find("] LOCK ").expect("LOCK line before the reporter ended");
            let unlock = log.find("] UNLOCK ").expect("UNLOCK line");
            assert!(lock < unlock, "{log}");
        }
        let _ = fs::remove_dir_all(&root);
    }
}
