//! Enqueue-ahead greedy `K = 0` decode: a child module of `engine::qwen3`.
//!
//! The plain `K = 0` loop is a host round trip per token: write the token and the cursor, launch,
//! wait, read the token back. Measured on the 64-layer hybrid model (device trace, FP8 weights),
//! that leaves the card idle for about 0.6 ms of every 15.7 ms step.
//!
//! Here the next step is enqueued while the current one runs, with the same step recipes: only
//! the bindings alternate. A step of parity `p` reads its token from buffer `tok[p]` and writes
//! its argmax to `tok[1 - p]`, which is the buffer the next step (parity `1 - p`) reads its
//! token from, so the token never visits the host on its way into the next step. The cursor is a
//! host-known sequence: step `n` reads `cur[p]`, written by the host after step `n - 2` (the
//! last user of that buffer) has been collected and before step `n` is enqueued. The host reads
//! step `n`'s token from `tok[1 - p]` after it completes; step `n + 2`, the next writer of that
//! buffer, is enqueued only after that read.
//!
//! The recipes, their graphs and every value they compute are those of the plain loop: the
//! stream is the plain loop's stream by construction.
//!
//! Used for greedy requests of the dense models without teacher forcing; everything else keeps
//! the plain loop (the host is needed between steps there).

use super::*;

pub(super) struct PipeOut {
    pub steps: usize,
    pub decode_ms: f64,
    pub eos: bool,
    pub cancel: bool,
}

/// Can this request decode enqueue-ahead?
pub(super) fn usable(eng: &Qwen3Engine, params: &Qwen3GenParams) -> bool {
    params.temperature <= 0.0
        && !params.k0_plain
        && params.force_tokens.is_none()
        && !eng.spec.is_moe()
        && eng.step_fb_launches.len() == eng.step_recipes.len()
        && !eng.step_fb_launches.is_empty()
}

/// Decodes from `first_tok` (the prefill's token, at sequence index `prompt_len`) until the
/// budget, an EOS or a cancel. `gdn_parity` = the parity holding the prompt's DeltaNet state.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode(
    eng: &mut Qwen3Engine,
    prompt_len: usize,
    params: &Qwen3GenParams,
    budget: &mut SequenceBudget,
    is_eos: &impl Fn(i32) -> bool,
    is_cancelled: &impl Fn() -> bool,
    on_token: &mut impl FnMut(i32),
    generated: &mut Vec<i32>,
    gdn_parity: usize,
    first_tok: i32,
) -> Result<PipeOut, String> {
    let t0 = Instant::now();
    let tok_addr = [eng.step_ids, eng.step_tokens];
    let cur_addr = [eng.step_cursor, eng.step_cursor_b];
    let mk = |eng: &Qwen3Engine| {
        eng.dev
            .create_event(false)
            .map_err(|e| format!("create step event: {e}"))
    };
    let events = [mk(eng)?, mk(eng)?];
    let mut p = gdn_parity;
    // the first step's token and cursor
    {
        let buf = eng.stage_io.as_mut_slice();
        buf[0..4].copy_from_slice(&first_tok.to_le_bytes());
        buf[64..68].copy_from_slice(&(prompt_len as i32).to_le_bytes());
        eng.dev
            .copy_h2d_pinned_batch_sync(&eng.stage_io, &[(0, 4, tok_addr[p]), (64, 4, cur_addr[p])])
            .map_err(|e| format!("write first step inputs H2D: {e}"))?;
    }

    let mut out = PipeOut {
        steps: 0,
        decode_ms: 0.0,
        eos: false,
        cancel: false,
    };
    let mut launched = 0usize;
    let mut flights: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    // From here on a launch may be queued: every error path drains the device before returning.
    macro_rules! tryd {
        ($e:expr) => {
            match $e {
                Ok(v) => v,
                Err(e) => {
                    let _ = eng.dev.sync_all();
                    return Err(e);
                }
            }
        };
    }
    loop {
        while flights.len() < 2 {
            if is_cancelled() {
                out.cancel = true;
                break;
            }
            // every launch in flight yields one token
            if budget.remaining() <= flights.len() as u64 {
                break;
            }
            if !tryd!(prof_rt::before_launch(eng, launched as u64, flights.len())) {
                break;
            }
            let cursor_val = (prompt_len + launched) as u64;
            let nch = (cursor_val + 1).div_ceil(KV_CHUNK).clamp(1, eng.max_chunks) as usize;
            if launched > 0 {
                // the last step that read this buffer (two launches back) has been collected
                tryd!(eng.write_i32_scalar(cur_addr[p], cursor_val as i32));
            }
            eng.live.launch_begin_with(nch as u64, || {
                qwen3_traffic::decode_step(&eng.spec, eng.max_ctx, nch as u64)
            });
            tryd!(eng.step_recipes[nch - 1]
                .launch_prepared(&eng.dev, &eng.step_fb_launches[nch - 1][p])
                .map_err(|e| format!("launch decode step {launched} (cursor={cursor_val}): {e}")));
            eng.live.launch_submitted();
            tryd!(events[p]
                .record_compute(&eng.dev)
                .map_err(|e| format!("record step event: {e}")));
            flights.push_back(p);
            p ^= 1;
            launched += 1;
        }
        let Some(fp) = flights.pop_front() else {
            break;
        };
        if out.cancel {
            flights.push_front(fp);
            break;
        }
        tryd!(events[fp]
            .synchronize()
            .map_err(|e| format!("wait decode step {}: {e}", out.steps)));
        let tok = tryd!(eng.read_i32_scalar(tok_addr[1 - fp]));
        out.steps += 1;
        budget.record_emitted(1);
        generated.push(tok);
        eng.live.step(1, 0, 0);
        on_token(tok);
        if params.stop_on_eos && is_eos(tok) {
            out.eos = true;
            break;
        }
        if budget.remaining() == 0 {
            break;
        }
    }
    // Decode time ends with the last token; a step enqueued ahead of an EOS / cancel still runs
    // on the card: wait for it and drop its token.
    out.decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if !flights.is_empty() {
        unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
            .map_err(|e| format!("drain decode steps: {e}"))?;
    }
    prof_rt::end_request(eng, launched as u64)?;
    Ok(out)
}
