//! Per-token log-probability readout of the Qwen3 engine (`Qwen3GenParams::logprobs`): a child
//! module of `engine::qwen3`.
//!
//! The numbers come from the target model's own logits row of each emitted token: the row the
//! card's argmax or on-card sampler picked the token from (never a drafter's), read back to the
//! host and reduced there by `crate::logprobs`. Nothing changes on the card for a request that
//! does not ask: same recipes, same launches, same depth of the launch queue.
//!
//! How a row reaches the host:
//!
//! - **Synchronised readouts** (the prefill's first token, the plain `K = 0` loop): the compute
//!   stream has drained, the row is copied D2H at once (one row, `2 * vocab` bytes).
//! - **Launches queued ahead** (the enqueue-ahead `K = 0` loop, the speculative verify loop):
//!   the next launch overwrites the logits buffer while the host collects the previous one, so
//!   each launch is followed on the compute stream by a D2H copy of its logits rows into its
//!   parity's slot of a pinned host ring. Stream order makes the copy read what the launch wrote
//!   and keeps the next launch from writing before the copy is done; the event the loop already
//!   records after the launch covers the copy. A slot is read at the launch's collect, before the
//!   loop enqueues the launch that reuses its parity. A verify copies all its rows (`t` of them),
//!   because how many are accepted is known only on the card.
//!
//! The ring is allocated once when the model loads, sized for the widest launch of the engine
//! (`lp_ring_rows`). Its size, the slot offsets, the row offsets and the fit checks (the slot
//! and the device buffer a copy reads from) are the verified `lp_*` functions.

use super::*;
use crate::logprobs::{self, TokenLogprob};

/// Two slots (one per launch parity) of `cap_rows` logits rows each, in pinned host memory.
/// Allocated once when the model loads, sized for the widest launch of the engine, and kept
/// for the engine's lifetime: a request never allocates or frees it.
pub(super) struct LpRing {
    buf: PinnedBuf,
    cap_rows: u64,
    row_bytes: u64,
}

impl LpRing {
    /// The ring of an engine whose verify launches have at most `blk` rows (0: no drafter) and
    /// whose logits rows are `row_bytes` bytes (`lp_ring_rows`, `lp_ring_bytes`).
    pub(super) fn alloc(dev: &HpuDevice, blk: u64, row_bytes: u64) -> Result<Self, String> {
        let cap_rows = crate::verified::lp_ring_rows(blk);
        let bytes = crate::verified::lp_ring_bytes(cap_rows, row_bytes);
        let len = usize::try_from(bytes)
            .map_err(|_| format!("logprob ring of {bytes} bytes does not fit host memory"))?;
        let buf = dev
            .alloc_pinned(len)
            .map_err(|e| format!("alloc pinned logprob ring ({bytes} bytes): {e}"))?;
        Ok(Self {
            buf,
            cap_rows,
            row_bytes,
        })
    }

    /// Rows of each slot.
    pub(super) fn cap_rows(&self) -> u64 {
        self.cap_rows
    }

    /// Pinned host bytes of the ring.
    pub(super) fn bytes(&self) -> u64 {
        crate::verified::lp_ring_bytes(self.cap_rows, self.row_bytes)
    }

    /// Byte range of a copy of `t` rows in slot `p`, when the copy fits the slot (`lp_slot_ok`,
    /// `lp_slot_start`, `lp_copy_bytes`).
    fn slot_range(&self, p: usize, t: u64) -> Result<(usize, usize), String> {
        if !crate::verified::lp_slot_ok(p as u64, t, self.cap_rows) {
            return Err(format!(
                "logprob readout: a launch of {t} rows does not fit slot {p} of the ring's {}-row slots",
                self.cap_rows
            ));
        }
        let at = crate::verified::lp_slot_start(p as u64, self.cap_rows, self.row_bytes);
        let bytes = crate::verified::lp_copy_bytes(t, self.row_bytes);
        Ok((at as usize, bytes as usize))
    }
}

/// The request asks for log-probabilities.
pub(super) fn wanted(params: &Qwen3GenParams) -> bool {
    params.logprobs.is_some()
}

/// Before a loop's first launch: every launch of the loop (at most `rows` rows) fits a slot of
/// the ring (`lp_copy_fits`). The ring is sized at load for the engine's widest launch, so this
/// refuses only a loop the load did not plan for, before anything is queued.
pub(super) fn loop_fits(eng: &Qwen3Engine, rows: u64) -> Result<(), String> {
    if crate::verified::lp_copy_fits(rows, eng.lp_ring.cap_rows) {
        return Ok(());
    }
    Err(format!(
        "logprob readout: launches of {rows} rows do not fit the ring's {}-row slots",
        eng.lp_ring.cap_rows
    ))
}

/// Queues, on the compute stream right after a launch, the copy of its `t` logits rows from the
/// device logits buffer at `src_hbm` (`src_bytes` bytes) into slot `p` of the ring.
pub(super) fn enqueue_rows(
    eng: &mut Qwen3Engine,
    src_hbm: u64,
    src_bytes: u64,
    t: u64,
    p: usize,
) -> Result<(), String> {
    let Qwen3Engine { lp_ring, dev, .. } = eng;
    let (at, bytes) = lp_ring.slot_range(p, t)?;
    if !crate::verified::lp_src_fits(t, lp_ring.row_bytes, src_bytes) {
        return Err(format!(
            "logprob readout: a launch of {t} rows reads past its {src_bytes}-byte logits buffer"
        ));
    }
    dev.enqueue_d2h_compute(src_hbm, bytes, &mut lp_ring.buf, at)
        .map_err(|e| format!("queue the logprob rows D2H: {e}"))
}

/// The entries of the `tokens` of a collected launch whose `t` rows are in slot `p` (row `i`
/// gave token `i`).
///
/// The other launch (and its copy) may still be queued when this runs: a panic in the
/// reduction drains the device before it unwinds out of the decode loop, as every error does.
pub(super) fn slot_entries(
    eng: &Qwen3Engine,
    p: usize,
    t: u64,
    tokens: &[i32],
    params: &Qwen3GenParams,
) -> Result<Vec<TokenLogprob>, String> {
    let Some(tap) = params.logprobs.as_ref() else {
        return Ok(Vec::new());
    };
    let ring = &eng.lp_ring;
    let (at, bytes) = ring.slot_range(p, t)?;
    let copy = ring
        .buf
        .as_slice()
        .get(at..)
        .and_then(|s| s.get(..bytes))
        .ok_or("logprob readout: the slot lies outside the host ring")?;
    let dev = samp_rt::dev_params(params, eng.spec.vocab);
    let rb = ring.row_bytes as usize;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        logprobs::step_logprobs(copy, rb, t, tokens, &dev, tap.top)
    }))
    .map_err(|panic| {
        // nothing may stay queued against the device when the panic leaves the loop
        let _ = eng.dev.sync_all();
        std::panic::resume_unwind(panic)
    })
}

/// Reads the row at `logits_hbm` now (the compute stream is synchronised) and returns the entry
/// of `token`, which that row gave. `None`: the request does not ask.
///
/// A host-picked request (penalties or a grammar: `host_picked`) picked `token` on the host from
/// its penalised and grammar-masked row (`pick_penalised`), so its entry comes from that row,
/// the distribution the token was actually drawn from, and not from the raw logits on the card:
/// a token the grammar refuses is at -inf there, outside the support, and never an alternative.
pub(super) fn row_now(
    eng: &mut Qwen3Engine,
    logits_hbm: u64,
    token: i32,
    params: &Qwen3GenParams,
) -> Result<Option<TokenLogprob>, String> {
    let pen_row = eng.lp_pen_row.take();
    let Some(tap) = params.logprobs.as_ref() else {
        return Ok(None);
    };
    let dev = samp_rt::dev_params(params, eng.spec.vocab);
    if params.host_picked() {
        let row = pen_row
            .ok_or("internal: a host-picked token reached the logprob readout without its row")?;
        return Ok(Some(logprobs::row_logprobs_u16(&row, &dev, token, tap.top)));
    }
    let rb = eng.lp_ring.row_bytes as usize;
    eng.dev
        .copy_d2h_pinned_sync(logits_hbm, &mut eng.stage_io, rb)
        .map_err(|e| format!("read the logprob row D2H: {e}"))?;
    Ok(Some(logprobs::row_logprobs(
        &eng.stage_io.as_slice()[..rb],
        &dev,
        token,
        tap.top,
    )))
}

/// Hands an entry to the request's tap (before its token goes out).
pub(super) fn push(params: &Qwen3GenParams, e: Option<TokenLogprob>) {
    if let (Some(tap), Some(e)) = (params.logprobs.as_ref(), e) {
        tap.push(e);
    }
}
