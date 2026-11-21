//! Live telemetry reporter for the Gaudi card status dashboard (`schema: 1` live-stats protocol).
//!
//! Writes a single atomic JSON status file (`<dir>/<tag>.<pid>.json`) refreshed on every phase
//! transition, at most once per second during decode, and on a 1-second background heartbeat so
//! long recipe compiles or weight uploads stay live.
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

use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    samples: VecDeque<StepSample>,
}

struct Shared {
    state: Mutex<State>,
    write_lock: Mutex<()>,
    stop: (Mutex<bool>, Condvar),
    path: PathBuf,
    tmp_path: PathBuf,
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
        if env::var("GAUDI_LIVE").as_deref() == Ok("0") {
            return Self::disabled();
        }
        let dir = env::var_os("GAUDI_LIVE_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("GAUDIUM_LIVE_ROOT")
                    .map(|r| PathBuf::from(r).join(".cardlock").join("live"))
            });
        let Some(dir) = dir else {
            return Self::disabled();
        };
        if !dir.is_dir() {
            return Self::disabled();
        }
        Self::in_dir(dir, engine, engine_version, source, heartbeat)
    }

    /// Creates a reporter targeting a specific directory (primarily for tests or explicit config).
    pub fn in_dir(
        dir: impl AsRef<Path>,
        engine: impl Into<String>,
        engine_version: impl Into<String>,
        source: impl Into<String>,
        heartbeat: bool,
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
        let owner = env::var("GAUDI_CARD_OWNER").unwrap_or_default();

        let state = State {
            engine: engine.into(),
            version: engine_version.into(),
            source: source.into(),
            owner,
            model: String::new(),
            method: String::new(),
            notes: String::new(),
            phase: "idle".to_string(),
            hbm_detail: String::new(),
            sources: String::new(),
            profile: String::new(),
            host_parts: BTreeMap::new(),
            pid,
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
            samples: VecDeque::new(),
        };

        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            write_lock: Mutex::new(()),
            stop: (Mutex::new(false), Condvar::new()),
            path,
            tmp_path,
        });
        flush(&shared, true, false);

        let hb_handle = if heartbeat {
            let s = Arc::clone(&shared);
            thread::Builder::new()
                .name("gaudium-live-hb".into())
                .spawn(move || {
                    let (lk, cv) = &s.stop;
                    let mut stopped = lk.lock().unwrap();
                    while !*stopped {
                        let (guard, _) = cv.wait_timeout(stopped, Duration::from_secs(1)).unwrap();
                        stopped = guard;
                        if *stopped {
                            break;
                        }
                        drop(stopped);
                        flush(&s, false, true);
                        stopped = lk.lock().unwrap();
                    }
                })
                .ok()
        } else {
            None
        };

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

    pub fn is_enabled(&self) -> bool {
        self.shared.is_some()
    }

    pub fn path(&self) -> Option<&Path> {
        self.shared.as_ref().map(|s| s.path.as_path())
    }

    pub fn set_model(&self, model: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().model = model.into();
            flush(s, true, false);
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
                }
            }
            flush(s, true, false);
        }
    }

    pub fn set_notes(&self, notes: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().notes = notes.into();
        }
    }

    /// Sets the 5-category HBM ledger values from exact byte counts (converted to decimal GB).
    pub fn set_hbm_bytes(&self, weights: u64, kv: u64, state: u64, scratch: u64, other: u64) {
        self.set_hbm_gb(
            weights as f64 / 1e9,
            kv as f64 / 1e9,
            state as f64 / 1e9,
            scratch as f64 / 1e9,
            other as f64 / 1e9,
        );
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
            s.state.lock().unwrap().hbm_gb =
                Some([weights_gb, kv_gb, state_gb, scratch_gb, other_gb]);
            flush(s, true, false);
        }
    }

    /// Sets an optional JSON object for `"hbm_detail"`.
    pub fn set_hbm_detail(&self, json_obj: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().hbm_detail = json_obj.into();
        }
    }

    /// Sets an optional JSON object for `"sources"`.
    pub fn set_sources(&self, json_obj: impl Into<String>) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().sources = json_obj.into();
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
                if !n.is_empty() {
                    st.notes = n;
                }
                st.t_begin_s = st.origin.elapsed().as_secs_f64();
                st.profile.clear();
                st.samples.clear();
            }
            flush(s, true, false);
        }
    }

    pub fn set_cached(&self, cached_tokens: u64) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().prompt_cached = cached_tokens;
            flush(s, true, false);
        }
    }

    pub fn first_token(&self, ttft_ms: Option<f64>, tokens: u64) {
        if let Some(s) = &self.shared {
            {
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
            }
            flush(s, true, false);
        }
    }

    pub fn step(&self, emitted: u64, drafted: u64, accepted: u64) {
        let Some(s) = &self.shared else { return };
        let due = {
            let mut st = s.state.lock().unwrap();
            let t = st.origin.elapsed().as_secs_f64();
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
            st.phase = "decode".to_string();
            (t - st.last_write_s) >= 1.0
        };
        if due {
            flush(s, false, false);
        }
    }

    pub fn end_request(&self) {
        if let Some(s) = &self.shared {
            s.state.lock().unwrap().phase = "idle".to_string();
            flush(s, true, false);
        }
    }
}

impl Drop for LiveReporter {
    fn drop(&mut self) {
        if let Some(s) = &self.shared {
            let (lk, cv) = &s.stop;
            *lk.lock().unwrap() = true;
            cv.notify_all();
        }
        if let Some(handle) = self.heartbeat.take() {
            let _ = handle.join();
        }
        if let Some(s) = &self.shared {
            let _ = fs::remove_file(&s.path);
            let _ = fs::remove_file(&s.tmp_path);
        }
    }
}

fn read_hostname() -> Option<String> {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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

fn flush(shared: &Shared, force: bool, heartbeat: bool) {
    let _wg = shared.write_lock.lock().unwrap();
    let host_ram = read_proc_ram_gb();
    let wall_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());

    let payload = {
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
            Some(p) => j.push_str(&format!(",\"context_pos\":{}", p + st.gen_tokens)),
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
        j
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn writes_schema_v1_json_and_unlinks_on_drop() {
        let dir = env::temp_dir().join(format!("gaudium_live_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let record_path;
        {
            let rep = LiveReporter::in_dir(&dir, "gaudium-rust", "v0.1", "engine", false);
            assert!(rep.is_enabled());
            record_path = rep.path().unwrap().to_path_buf();
            assert!(record_path.exists());

            rep.set_model("Qwen3.8-27B");
            rep.set_hbm_bytes(
                50_000_000_000,
                500_000_000,
                100_000_000,
                400_000_000,
                256_000_000,
            );
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
            assert_eq!(v.get("gen_tokens").and_then(|x| x.as_u64()), Some(2));
            assert_eq!(v.get("context_pos").and_then(|x| x.as_u64()), Some(41));
            let hbm = v.get("hbm").expect("hbm object present");
            assert!((hbm.get("weights_gb").unwrap().as_f64().unwrap() - 50.0).abs() < 1e-3);
            assert!((hbm.get("kv_gb").unwrap().as_f64().unwrap() - 0.5).abs() < 1e-3);
        }

        assert!(
            !record_path.exists(),
            "live status file must be unlinked on drop"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
