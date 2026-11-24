//! Device-trace windows of the decode loops (`--profile-steps`): a child module of
//! `engine::qwen3`, so it reaches the engine's device and profiler without widening visibility.
//!
//! A decode loop calls [`before_launch`] with the index of the launch it is about to enqueue and
//! the number of launches still in flight; a window opens and closes only on a drained compute
//! stream, so `false` means "collect what is in flight first, then ask again".

use super::*;
use crate::synapse::profiler::{dump_tsv, summarize, Profiler};

/// Request counter and profiler of a loaded engine.
pub(super) struct ProfRt {
    pub prof: Profiler,
    /// Generate calls started on this engine, minus one (the running request's number).
    pub request: u64,
    started: bool,
    /// Launch index the open window started at.
    open_at: u64,
    /// The open window counts a lane session's batched launches: the prefills of requests
    /// admitted inside it leave it open (their launches show as their own node families).
    lane_open: bool,
}

impl ProfRt {
    pub fn new(prof: Profiler) -> Self {
        Self {
            prof,
            request: 0,
            started: false,
            open_at: 0,
            lane_open: false,
        }
    }
}

/// Start of a request (every generate call).
pub(super) fn begin_request(eng: &mut Qwen3Engine) {
    if let Some(p) = eng.prof.as_mut() {
        if p.started {
            p.request += 1;
        }
        p.started = true;
    }
}

fn report(p: &mut ProfRt, dev: &HpuDevice, upto: u64) -> Result<(), String> {
    unsafe { dev.sync_stream(dev.compute_stream()) }
        .map_err(|e| format!("profile: drain before stop: {e}"))?;
    p.lane_open = false;
    let Some((w, events)) = p.prof.stop().map_err(|e| format!("profile stop: {e}"))? else {
        return Ok(());
    };
    let launches = upto.min(w.last + 1).saturating_sub(p.open_at);
    eprintln!(
        "[profile] request {} launches {}-{} (a profiled run is not a timing result)",
        w.request,
        w.first,
        w.first + launches.saturating_sub(1)
    );
    eprint!("{}", summarize(&events, launches));
    if let Some(path) = &w.dump {
        match std::fs::write(path, dump_tsv(&events)) {
            Ok(()) => eprintln!("[profile] events written to {}", path.display()),
            Err(e) => eprintln!("[profile] could not write {}: {e}", path.display()),
        }
    }
    Ok(())
}

/// Before enqueueing launch `idx` of the running request with `in_flight` launches not yet
/// collected. Returns `false` when a window edge needs those collected first.
pub(super) fn before_launch(
    eng: &mut Qwen3Engine,
    idx: u64,
    in_flight: usize,
) -> Result<bool, String> {
    let Qwen3Engine { prof, dev, .. } = eng;
    let Some(p) = prof.as_mut() else {
        return Ok(true);
    };
    let req = p.request;
    if p.prof.wants_stop(req, idx, false) {
        if in_flight > 0 {
            return Ok(false);
        }
        report(p, dev, idx)?;
    }
    if p.prof.wants_start(req, idx, false) {
        if in_flight > 0 {
            return Ok(false);
        }
        unsafe { dev.sync_stream(dev.compute_stream()) }
            .map_err(|e| format!("profile: drain before start: {e}"))?;
        p.prof
            .start(req, idx, false)
            .map_err(|e| format!("profile start: {e}"))?;
        p.open_at = idx;
    }
    Ok(true)
}

/// Before enqueueing prefill launch `idx` (the tile loop's launches in order: the embedding, then
/// one per layer, tile after tile) of the running request. A window edge drains the compute
/// stream here: a profiled prefill is not a timing run.
pub(super) fn before_prefill_launch(
    prof: &mut Option<ProfRt>,
    dev: &HpuDevice,
    idx: u64,
) -> Result<(), String> {
    let Some(p) = prof.as_mut() else {
        return Ok(());
    };
    if p.lane_open {
        return Ok(());
    }
    let req = p.request;
    if p.prof.wants_stop(req, idx, true) {
        report(p, dev, idx)?;
    }
    if p.prof.wants_start(req, idx, true) {
        unsafe { dev.sync_stream(dev.compute_stream()) }
            .map_err(|e| format!("profile: drain before start: {e}"))?;
        p.prof
            .start(req, idx, true)
            .map_err(|e| format!("profile start: {e}"))?;
        p.open_at = idx;
    }
    Ok(())
}

/// End of the prefill tile loop after `launched` launches: closes a prefill window still open.
pub(super) fn end_prefill(
    prof: &mut Option<ProfRt>,
    dev: &HpuDevice,
    launched: u64,
) -> Result<(), String> {
    let Some(p) = prof.as_mut() else {
        return Ok(());
    };
    if !p.lane_open && p.prof.open_window().is_some_and(|w| w.prefill) {
        report(p, dev, launched)?;
    }
    Ok(())
}

/// End of a decode loop that launched `launched` steps: closes a window the request ended in.
pub(super) fn end_request(eng: &mut Qwen3Engine, launched: u64) -> Result<(), String> {
    let Qwen3Engine { prof, dev, .. } = eng;
    let Some(p) = prof.as_mut() else {
        return Ok(());
    };
    if p.prof.open_window().is_some() {
        report(p, dev, launched)?;
    }
    Ok(())
}

/// Before batched lane launch `idx` (`--parallel` lanes, counted from the engine's load; the lane
/// loop collects every launch before the next, so none is in flight). The windows of request 0
/// count these launches.
pub(super) fn before_lane_launch(eng: &mut Qwen3Engine, idx: u64) -> Result<(), String> {
    let Qwen3Engine { prof, dev, .. } = eng;
    let Some(p) = prof.as_mut() else {
        return Ok(());
    };
    if p.prof.wants_stop(0, idx, false) {
        report(p, dev, idx)?;
    }
    if p.prof.wants_start(0, idx, false) {
        unsafe { dev.sync_stream(dev.compute_stream()) }
            .map_err(|e| format!("profile: drain before start: {e}"))?;
        p.prof
            .start(0, idx, false)
            .map_err(|e| format!("profile start: {e}"))?;
        p.open_at = idx;
        p.lane_open = p.prof.open_window().is_some();
    }
    Ok(())
}

/// End of a lane session after `launched` batched launches: closes a window still open.
pub(super) fn end_lanes(eng: &mut Qwen3Engine, launched: u64) -> Result<(), String> {
    let Qwen3Engine { prof, dev, .. } = eng;
    let Some(p) = prof.as_mut() else {
        return Ok(());
    };
    if p.prof.open_window().is_some_and(|w| !w.prefill) {
        report(p, dev, launched)?;
    }
    Ok(())
}
