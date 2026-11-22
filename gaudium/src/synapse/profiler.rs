//! Device trace of chosen decode launches through the SynapseAI profiler API.
//!
//! One window per process: the profiler starts before decode launch `first`, stops after launch
//! `last` completed, and the retrieved events (one per engine activity of a graph node: MME, TPC,
//! DMA) are folded into a per-launch breakdown by node family and engine class, the idle time no
//! engine covers, and the node transitions that idle time sits in.
//!
//! Requirements of the API (vendor headers `synapse_api.h` / `synapse_common_types.h`):
//! `HABANA_PROFILE=<template>` must be in the environment before `synInitialize`
//! ([`setup_env`]), and node names only reach the trace from recipes compiled in that process,
//! so a profiling run compiles its recipes fresh (recipe cache off).
//!
//! A profiled run is never a timing result: the instrumentation and the window edges perturb the
//! launches. Its use is the split of a launch, read next to an unprofiled run's launch time.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt::Write as _;

use super::ffi::{syn_check, SynDeviceId, SynStatus, SynapseError};

/// Profiler template: start / stop by API, events kept in memory, no post-graph metadata.
pub const TEMPLATE: &str = "profile_api_light";

const SYN_TRACE_DEVICE: u32 = 3;
const SYN_TRACE_FORMAT_TEF: u32 = 1;
/// `sizeof(synTraceEvent)` and the field offsets used here (x86-64, vendor header layout).
const EV_SIZE: usize = 208;
const OFF_NAME: usize = 0;
const OFF_TYPE: usize = 16;
const OFF_TS: usize = 32;
const OFF_DUR: usize = 48;
const OFF_ENGINE_TYPE: usize = 64;
const OFF_ENGINE_INDEX: usize = 68;
const OFF_ARG0: usize = 80;

/// Puts the profiler template into the environment unless one is set already. Must run before
/// the device is opened.
pub fn setup_env() {
    if std::env::var_os("HABANA_PROFILE").is_none() {
        // SAFETY: called from the single-threaded CLI start-up, before any other thread exists.
        unsafe { std::env::set_var("HABANA_PROFILE", TEMPLATE) };
    }
}

/// Decode launches `[first, last]` (0-based) of request number `request` (0-based count of the
/// engine's generate calls) to trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileWindow {
    pub first: u64,
    pub last: u64,
    pub request: u64,
    /// Where the raw events go as tab-separated text (None: summary only).
    pub dump: Option<std::path::PathBuf>,
}

impl ProfileWindow {
    /// `A-B`, `A-B@R`, each optionally followed by `:<events file>`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let (head, dump) = match s.split_once(':') {
            Some((r, f)) => (r, Some(std::path::PathBuf::from(f))),
            None => (s, None),
        };
        let bad = || format!("--profile-steps '{s}': expected A-B[@request][:<events file>]");
        let (range, request) = match head.split_once('@') {
            Some((r, q)) => (r, q.trim().parse::<u64>().map_err(|_| bad())?),
            None => (head, 0),
        };
        let (a, b) = range.split_once('-').ok_or_else(bad)?;
        let first: u64 = a.trim().parse().map_err(|_| bad())?;
        let last: u64 = b.trim().parse().map_err(|_| bad())?;
        if last < first || last - first > 63 {
            return Err(format!(
                "--profile-steps '{s}': need first <= last and at most 64 launches"
            ));
        }
        Ok(Self {
            first,
            last,
            request,
            dump,
        })
    }

    pub fn launches(&self) -> u64 {
        self.last - self.first + 1
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

type QueryMemFn = unsafe extern "C" fn(SynDeviceId, *mut u32) -> SynStatus;
type SetBufferFn = unsafe extern "C" fn(SynDeviceId, *mut c_void) -> SynStatus;
type StartStopFn = unsafe extern "C" fn(u32, SynDeviceId) -> SynStatus;
type GetTraceFn =
    unsafe extern "C" fn(u32, SynDeviceId, u32, *mut c_void, *mut usize, *mut usize) -> SynStatus;

struct Fns {
    query_mem: QueryMemFn,
    set_buffer: SetBufferFn,
    start: StartStopFn,
    stop: StartStopFn,
    get_trace: GetTraceFn,
}

impl Fns {
    /// Resolves the profiler entry points from the already loaded SynapseAI library.
    fn load() -> Result<Self, SynapseError> {
        let mut cands: Vec<String> = Vec::new();
        if let Ok(p) = std::env::var("GAUDIUM_SYNAPSE_LIB") {
            if !p.trim().is_empty() {
                cands.push(p);
            }
        }
        cands.push("/usr/lib/habanalabs/libSynapse.so".into());
        cands.push("libSynapse.so".into());
        let mut handle = std::ptr::null_mut();
        for c in &cands {
            let Ok(cs) = CString::new(c.as_str()) else {
                continue;
            };
            // SAFETY: valid C string; RTLD_NOW | RTLD_LOCAL on a library the engine has loaded.
            handle = unsafe { dlopen(cs.as_ptr(), 2) };
            if !handle.is_null() {
                break;
            }
        }
        if handle.is_null() {
            return Err(SynapseError::LibraryLoad(
                "profiler: libSynapse.so not found".into(),
            ));
        }
        let sym = |name: &str| -> Result<*mut c_void, SynapseError> {
            let cs = CString::new(name).map_err(|_| SynapseError::MissingSymbol(name.into()))?;
            // SAFETY: live dlopen handle and NUL-terminated name.
            let p = unsafe { dlsym(handle, cs.as_ptr()) };
            if p.is_null() {
                Err(SynapseError::MissingSymbol(name.into()))
            } else {
                Ok(p)
            }
        };
        // SAFETY: the signatures follow `synapse_api.h` of the SDK the offsets above were read from.
        unsafe {
            Ok(Self {
                query_mem: std::mem::transmute::<*mut c_void, QueryMemFn>(sym(
                    "synProfilerQueryRequiredMemory",
                )?),
                set_buffer: std::mem::transmute::<*mut c_void, SetBufferFn>(sym(
                    "synProfilerSetUserBuffer",
                )?),
                start: std::mem::transmute::<*mut c_void, StartStopFn>(sym("synProfilerStart")?),
                stop: std::mem::transmute::<*mut c_void, StartStopFn>(sym("synProfilerStop")?),
                get_trace: std::mem::transmute::<*mut c_void, GetTraceFn>(sym(
                    "synProfilerGetTrace",
                )?),
            })
        }
    }
}

/// One engine activity of the trace.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceEvent {
    pub name: String,
    /// Engine class: `MME`, `TPC`, `DMA`, ...
    pub class: String,
    /// Engine instance (TEF process id and thread id).
    pub inst: (u32, u32),
    pub ts_us: f64,
    pub dur_us: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    Open,
    Done,
}

/// The profiling session of a loaded engine: one or more windows, opened one at a time.
pub struct Profiler {
    fns: Fns,
    dev: SynDeviceId,
    windows: Vec<(ProfileWindow, State)>,
}

impl Profiler {
    pub fn new(dev: SynDeviceId, windows: Vec<ProfileWindow>) -> Result<Self, SynapseError> {
        Ok(Self {
            fns: Fns::load()?,
            dev,
            windows: windows.into_iter().map(|w| (w, State::Idle)).collect(),
        })
    }

    /// Device bytes the profiler wants from the caller (0: none), to be planned with the rest.
    pub fn required_bytes(&self) -> Result<u64, SynapseError> {
        let mut need = 0u32;
        // SAFETY: valid device id and output pointer.
        syn_check("synProfilerQueryRequiredMemory", unsafe {
            (self.fns.query_mem)(self.dev, &mut need)
        })?;
        Ok(u64::from(need))
    }

    pub fn set_user_buffer(&self, addr: u64) -> Result<(), SynapseError> {
        // SAFETY: `addr` is a device allocation of at least `required_bytes()`.
        syn_check("synProfilerSetUserBuffer", unsafe {
            (self.fns.set_buffer)(self.dev, addr as *mut c_void)
        })
    }

    /// The open window, if any.
    pub fn open_window(&self) -> Option<&ProfileWindow> {
        self.windows
            .iter()
            .find(|(_, st)| *st == State::Open)
            .map(|(w, _)| w)
    }

    /// The open window ends before launch `idx` of request `req` (or belongs to another request).
    pub fn wants_stop(&self, req: u64, idx: u64) -> bool {
        self.open_window()
            .is_some_and(|w| w.request != req || idx > w.last)
    }

    /// A window that has not run yet starts at launch `idx` of request `req`.
    pub fn wants_start(&self, req: u64, idx: u64) -> bool {
        self.windows
            .iter()
            .any(|(w, st)| *st == State::Idle && w.request == req && w.first == idx)
    }

    /// Starts the trace of the window at launch `idx` of request `req`. The caller has drained
    /// the compute stream and closed any open window.
    pub fn start(&mut self, req: u64, idx: u64) -> Result<(), SynapseError> {
        let Some(i) = self
            .windows
            .iter()
            .position(|(w, st)| *st == State::Idle && w.request == req && w.first == idx)
        else {
            return Ok(());
        };
        // SAFETY: device open, profiler template set before `synInitialize`.
        syn_check("synProfilerStart", unsafe {
            (self.fns.start)(SYN_TRACE_DEVICE, self.dev)
        })?;
        self.windows[i].1 = State::Open;
        Ok(())
    }

    /// Stops the open trace and returns its window and events. The caller has drained the
    /// compute stream.
    pub fn stop(&mut self) -> Result<Option<(ProfileWindow, Vec<TraceEvent>)>, SynapseError> {
        let Some(i) = self.windows.iter().position(|(_, st)| *st == State::Open) else {
            return Ok(None);
        };
        self.windows[i].1 = State::Done;
        let window = self.windows[i].0.clone();
        // SAFETY: a started session on this device.
        syn_check("synProfilerStop", unsafe {
            (self.fns.stop)(SYN_TRACE_DEVICE, self.dev)
        })?;
        let (mut size, mut n) = (0usize, 0usize);
        // SAFETY: a null buffer asks for the sizes only.
        syn_check("synProfilerGetTrace(size)", unsafe {
            (self.fns.get_trace)(
                SYN_TRACE_DEVICE,
                self.dev,
                SYN_TRACE_FORMAT_TEF,
                std::ptr::null_mut(),
                &mut size,
                &mut n,
            )
        })?;
        if size == 0 || n == 0 {
            return Ok(Some((window, Vec::new())));
        }
        // 16-byte aligned storage: the entries hold `long double` fields
        let mut buf: Vec<u128> = vec![0; size / 16 + 8];
        let mut size2 = buf.len() * 16;
        let mut n2 = 0usize;
        // SAFETY: `buf` holds `size2` writable bytes.
        syn_check("synProfilerGetTrace", unsafe {
            (self.fns.get_trace)(
                SYN_TRACE_DEVICE,
                self.dev,
                SYN_TRACE_FORMAT_TEF,
                buf.as_mut_ptr().cast(),
                &mut size2,
                &mut n2,
            )
        })?;
        let bytes =
            // SAFETY: plain bytes of the buffer the API filled.
            unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len() * 16) };
        Ok(Some((window, decode_events(bytes, n2.min(n)))))
    }
}

/// x87 80-bit extended value (little endian, 10 significant bytes) as `f64`.
fn x87_to_f64(b: &[u8]) -> f64 {
    let mant = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let se = u16::from_le_bytes(b[8..10].try_into().unwrap());
    if mant == 0 {
        return 0.0;
    }
    let exp = i32::from(se & 0x7fff) - 16383 - 63;
    let v = (mant as f64) * 2f64.powi(exp);
    if se & 0x8000 != 0 {
        -v
    } else {
        v
    }
}

fn cstr_at(bytes: &[u8], off: usize) -> String {
    let p = usize::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as *const c_char;
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the API returns NUL-terminated strings inside the same buffer (its string pool).
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Engine class of a TEF process name (`MME0`, `TPC12`, `EDMA 3`, `DCORE0 TPC 2`, ...).
pub fn class_of(pname: &str) -> String {
    let u = pname.to_ascii_uppercase();
    for k in [
        "MME", "TPC", "EDMA", "PDMA", "KDMA", "DMA", "ROT", "NIC", "ARC",
    ] {
        if u.contains(k) {
            return if k.ends_with("DMA") { "DMA" } else { k }.to_string();
        }
    }
    if pname.is_empty() {
        "engine".into()
    } else {
        pname.into()
    }
}

fn decode_events(bytes: &[u8], n: usize) -> Vec<TraceEvent> {
    let mut pname: BTreeMap<u32, String> = BTreeMap::new();
    struct Raw {
        name: String,
        ph: u8,
        pid: u32,
        tid: u32,
        ts: f64,
        dur: f64,
    }
    let mut raw: Vec<Raw> = Vec::with_capacity(n);
    for i in 0..n {
        let e = &bytes[i * EV_SIZE..(i + 1) * EV_SIZE];
        let ph = e[OFF_TYPE];
        let pid = u32::from_le_bytes(e[OFF_ENGINE_TYPE..OFF_ENGINE_TYPE + 4].try_into().unwrap());
        let tid = u32::from_le_bytes(
            e[OFF_ENGINE_INDEX..OFF_ENGINE_INDEX + 4]
                .try_into()
                .unwrap(),
        );
        let name = cstr_at(e, OFF_NAME);
        if ph == b'M' {
            if name == "process_name" {
                pname.insert(pid, cstr_at(e, OFF_ARG0));
            }
            continue;
        }
        if !matches!(ph, b'X' | b'B' | b'E') {
            continue;
        }
        raw.push(Raw {
            name,
            ph,
            pid,
            tid,
            ts: x87_to_f64(&e[OFF_TS..OFF_TS + 10]),
            dur: x87_to_f64(&e[OFF_DUR..OFF_DUR + 10]),
        });
    }
    // begin / end pairs become complete events
    let mut open: BTreeMap<(u32, u32, String), f64> = BTreeMap::new();
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let class = class_of(pname.get(&r.pid).map_or("", String::as_str));
        match r.ph {
            b'X' => out.push(TraceEvent {
                name: r.name,
                class,
                inst: (r.pid, r.tid),
                ts_us: r.ts,
                dur_us: r.dur,
            }),
            b'B' => {
                open.insert((r.pid, r.tid, r.name), r.ts);
            }
            _ => {
                if let Some(t0) = open.remove(&(r.pid, r.tid, r.name.clone())) {
                    out.push(TraceEvent {
                        name: r.name,
                        class,
                        inst: (r.pid, r.tid),
                        ts_us: t0,
                        dur_us: r.ts - t0,
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| a.ts_us.total_cmp(&b.ts_us));
    out
}

/// Graph node of an event name: compiler suffixes (`_bundle_<n>/op_<k>`, `_complex/...`) removed.
pub fn node_of_event(ev: &str) -> &str {
    let mut end = ev.len();
    for m in ["_bundle_", "_complex/", "/"] {
        if let Some(k) = ev.find(m) {
            end = end.min(k);
        }
    }
    &ev[..end]
}

/// Node family: the layer number after a leading `L` / `D` folded to `#` (`L12_qkvz_gemm` ->
/// `L#_qkvz_gemm`).
pub fn family_of(node: &str) -> String {
    let b = node.as_bytes();
    if b.len() > 1 && (b[0] == b'L' || b[0] == b'D') && b[1].is_ascii_digit() {
        let rest = node[1..].trim_start_matches(|c: char| c.is_ascii_digit());
        return format!("{}#{rest}", &node[..1]);
    }
    node.to_string()
}

fn union_len(iv: &mut [(f64, f64)]) -> f64 {
    iv.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut tot, mut a, mut b) = (0.0, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(s, e) in iv.iter() {
        if s > b {
            if b > a {
                tot += b - a;
            }
            a = s;
            b = e;
        } else if e > b {
            b = e;
        }
    }
    if b > a {
        tot += b - a;
    }
    tot
}

/// The breakdown of a trace over `launches` launches, as report lines.
pub fn summarize(events: &[TraceEvent], launches: u64) -> String {
    let mut s = String::new();
    if events.is_empty() {
        s.push_str("[profile] no device events (was HABANA_PROFILE set before the device opened, and were the recipes compiled in this process?)\n");
        return s;
    }
    let n = launches.max(1) as f64;
    let w0 = events.iter().map(|e| e.ts_us).fold(f64::INFINITY, f64::min);
    let w1 = events
        .iter()
        .map(|e| e.ts_us + e.dur_us)
        .fold(f64::NEG_INFINITY, f64::max);
    let win = w1 - w0;
    let mut all: Vec<(f64, f64)> = events
        .iter()
        .map(|e| (e.ts_us, e.ts_us + e.dur_us))
        .collect();
    let busy = union_len(&mut all);
    let _ = writeln!(
        s,
        "[profile] window {:.3} ms over {launches} launches = {:.3} ms/launch; {} device events; some engine active {:.1}% ({:.3} ms/launch idle)",
        win / 1e3,
        win / 1e3 / n,
        events.len(),
        100.0 * busy / win,
        (win - busy) / 1e3 / n
    );
    // engine classes
    let mut by_class: BTreeMap<&str, Vec<(f64, f64)>> = BTreeMap::new();
    let mut insts: BTreeMap<&str, BTreeSet<(u32, u32)>> = BTreeMap::new();
    for e in events {
        by_class
            .entry(e.class.as_str())
            .or_default()
            .push((e.ts_us, e.ts_us + e.dur_us));
        insts.entry(e.class.as_str()).or_default().insert(e.inst);
    }
    for (c, iv) in &mut by_class {
        let sum: f64 = iv.iter().map(|x| x.1 - x.0).sum();
        let u = union_len(iv);
        let _ = writeln!(
            s,
            "[profile]   {c:<5} {:>2} instances: any busy {:>5.1}% = {:.3} ms/launch, engine time {:.3} ms/launch",
            insts[c].len(),
            100.0 * u / win,
            u / 1e3 / n,
            sum / 1e3 / n
        );
    }
    // node families
    struct Fam {
        iv: Vec<(f64, f64)>,
        cls: BTreeSet<String>,
        n: u64,
    }
    let mut fam: BTreeMap<String, Fam> = BTreeMap::new();
    for e in events {
        let f = fam.entry(family_of(node_of_event(&e.name))).or_insert(Fam {
            iv: Vec::new(),
            cls: BTreeSet::new(),
            n: 0,
        });
        f.iv.push((e.ts_us, e.ts_us + e.dur_us));
        f.cls.insert(e.class.clone());
        f.n += 1;
    }
    let mut rows: Vec<(f64, String, String, u64)> = fam
        .into_iter()
        .map(|(k, mut f)| {
            let cls: Vec<&str> = f.cls.iter().map(String::as_str).collect();
            (union_len(&mut f.iv), k, cls.join("+"), f.n)
        })
        .collect();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    let _ = writeln!(
        s,
        "[profile] node families (wall = union of the family's events, so overlapping MME spans count once inside a family but twice across families):"
    );
    for (w, k, cls, cnt) in rows.iter().take(60) {
        let _ = writeln!(
            s,
            "[profile]   {:>8.3} ms/launch {:>5.1}%  {cls:<8} {k}  ({:.0} events/launch)",
            w / 1e3 / n,
            100.0 * w / win,
            *cnt as f64 / n
        );
    }
    // idle gaps by transition
    let mut ivn: Vec<(f64, f64, &str)> = events
        .iter()
        .map(|e| (e.ts_us, e.ts_us + e.dur_us, e.name.as_str()))
        .collect();
    ivn.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut gaps: BTreeMap<String, (f64, u64)> = BTreeMap::new();
    let (mut end, mut last) = (f64::NEG_INFINITY, "");
    let mut gap_tot = 0.0;
    for &(t0, t1, name) in &ivn {
        if end > f64::NEG_INFINITY && t0 > end + 0.05 {
            let key = format!(
                "{} -> {}",
                family_of(node_of_event(last)),
                family_of(node_of_event(name))
            );
            let g = gaps.entry(key).or_insert((0.0, 0));
            g.0 += t0 - end;
            g.1 += 1;
            gap_tot += t0 - end;
        }
        if t1 > end {
            end = t1;
            last = name;
        }
    }
    let mut gv: Vec<(f64, u64, String)> = gaps.into_iter().map(|(k, v)| (v.0, v.1, k)).collect();
    gv.sort_by(|a, b| b.0.total_cmp(&a.0));
    let _ = writeln!(
        s,
        "[profile] idle (no engine active) {:.3} ms/launch, by transition:",
        gap_tot / 1e3 / n
    );
    for (us, cnt, k) in gv.iter().take(24) {
        let _ = writeln!(
            s,
            "[profile]   {:>8.1} us/launch ({:.1} per launch, {:.2} us each)  {k}",
            us / n,
            *cnt as f64 / n,
            us / *cnt as f64
        );
    }
    s
}

/// Raw events as tab-separated text: `ts_us dur_us class pid tid name`.
pub fn dump_tsv(events: &[TraceEvent]) -> String {
    let mut s = String::with_capacity(events.len() * 64);
    for e in events {
        let _ = writeln!(
            s,
            "{:.3}\t{:.3}\t{}\t{}\t{}\t{}",
            e.ts_us, e.dur_us, e.class, e.inst.0, e.inst.1, e.name
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_spec_parses_and_bounds() {
        let w = ProfileWindow::parse("24-27").unwrap();
        assert_eq!((w.first, w.last, w.request, w.launches()), (24, 27, 0, 4));
        assert!(w.dump.is_none());
        let w = ProfileWindow::parse("3-3@2:/x/ev.tsv").unwrap();
        assert_eq!((w.first, w.request), (3, 2));
        assert_eq!(w.dump.as_deref(), Some(std::path::Path::new("/x/ev.tsv")));
        assert!(ProfileWindow::parse("3-4@x").is_err());
        assert!(ProfileWindow::parse("5-2").is_err());
        assert!(ProfileWindow::parse("0-100").is_err());
        assert!(ProfileWindow::parse("abc").is_err());
    }

    #[test]
    fn x87_values_convert() {
        // 1.0 = mantissa 0x8000_0000_0000_0000, exponent 16383
        let mut b = [0u8; 10];
        b[7] = 0x80;
        b[8..10].copy_from_slice(&16383u16.to_le_bytes());
        assert_eq!(x87_to_f64(&b), 1.0);
        // 1234.5 = 0x9A50 << 48 at exponent 16383 + 10
        let m = ((1234.5f64 / 1024.0) * (1u64 << 63) as f64) as u64;
        b[0..8].copy_from_slice(&m.to_le_bytes());
        b[8..10].copy_from_slice(&(16383u16 + 10).to_le_bytes());
        assert_eq!(x87_to_f64(&b), 1234.5);
        b[8..10].copy_from_slice(&((16383u16 + 10) | 0x8000).to_le_bytes());
        assert_eq!(x87_to_f64(&b), -1234.5);
        assert_eq!(x87_to_f64(&[0u8; 10]), 0.0);
    }

    #[test]
    fn names_fold_to_families() {
        assert_eq!(node_of_event("L3_qkvz_gemm_bundle_32/op_1"), "L3_qkvz_gemm");
        assert_eq!(
            node_of_event("L0_x2_add_complex/add_fwd_bf16_0"),
            "L0_x2_add"
        );
        assert_eq!(node_of_event("accept"), "accept");
        assert_eq!(family_of("L12_qkvz_gemm"), "L#_qkvz_gemm");
        assert_eq!(family_of("D4_softmax"), "D#_softmax");
        assert_eq!(family_of("logits_gemm"), "logits_gemm");
        assert_eq!(family_of("Lx"), "Lx");
        assert_eq!(class_of("DCORE0 TPC 2"), "TPC");
        assert_eq!(class_of("EDMA 3"), "DMA");
        assert_eq!(class_of("MME0"), "MME");
    }

    #[test]
    fn summary_counts_union_time_and_gaps() {
        let ev = |name: &str, class: &str, tid: u32, ts: f64, dur: f64| TraceEvent {
            name: name.into(),
            class: class.into(),
            inst: (1, tid),
            ts_us: ts,
            dur_us: dur,
        };
        let events = vec![
            ev("L0_qkvz_gemm_bundle_1/op_0", "MME", 0, 0.0, 100.0),
            ev("L0_qkvz_gemm_bundle_1/op_1", "MME", 1, 50.0, 100.0),
            ev("L0_gdn", "TPC", 0, 160.0, 40.0),
            ev("L1_qkvz_gemm_bundle_2/op_0", "MME", 0, 200.0, 100.0),
        ];
        let s = summarize(&events, 1);
        // window 300 us, busy 150 + 40 + 100 = 290, one 10 us gap after the first GEMM family
        assert!(s.contains("window 0.300 ms"), "{s}");
        assert!(s.contains("0.010 ms/launch idle"), "{s}");
        assert!(s.contains("L#_qkvz_gemm -> L#_gdn"), "{s}");
        assert!(s.contains("   0.250 ms/launch"), "{s}");
        let tsv = dump_tsv(&events);
        assert_eq!(tsv.lines().count(), 4);
        assert!(tsv.starts_with("0.000\t100.000\tMME\t1\t0\tL0_qkvz_gemm_bundle_1/op_0"));
    }
}
