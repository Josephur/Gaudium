//! Per-request device state reset of the dense Qwen3 / Qwen3.5 / Qwen3.8 engine.
//!
//! This is a child module of `engine::qwen3` (declared there with `#[path]`), so it reads the
//! engine's private address maps without widening their visibility.
//!
//! What a new request needs from the device, and why (kernel sources under `kernels/tpc`):
//! - **Gated DeltaNet start state** (slot 0 of ring 0 of every linear layer, recurrent and conv):
//!   read by the first prefill tile (or the first step of a one-token prompt) as the state before
//!   position 0, so it must be zero. Every other ring slot is written by the step before the one
//!   that reads it.
//! - **Trunk KV rows at or past the cursor**: never needed. `gaudium_attn_softmax_chunk` selects
//!   by position (`pos < cursor + t + 1`): a masked key gets `P = 0` whatever its score is, the
//!   masked tail of the boundary chunk and every chunk past it are zero-stored without loading
//!   their scores, and the `P x V` GEMM then adds exact zeros for those rows. A stale row left by
//!   an earlier request is in the same position as the rows a rejected draft leaves behind inside
//!   one request, which the speculative steps already rely on. The one requirement is that a
//!   masked row holds a finite value (`0 x NaN` is not 0): every cache is zeroed on the device
//!   once at load, and after that only the engine's own K/V rows are ever written to it.
//! - **Drafter context caches** (DSpine / DFlash2 `D{l}_kcache|vcache`, the MTP head's cache):
//!   masked the same way (`gaudium_df_softmax`: valid iff `lo <= pos < cursor`, `P = 0`
//!   elsewhere), and a drafter only proposes tokens, so they cannot change the output at all.
//!
//! The default mode therefore zeroes only the DeltaNet start state per request, on the device
//! (`synMemsetD32Async` on the compute stream, one stream sync for the whole reset). The KV rows
//! a request may have written are still tracked as a high-water mark: after a request that ended
//! with an error the next reset also zeroes everything below it, and `GAUDIUM_REQUEST_RESET`
//! selects that (`dirty`) or a zero fill of every cache (`full`) for every request.

use super::*;

/// Rows past `prompt + generated` a request may have written: the widest verify block of a step
/// that was already enqueued when the request stopped, the step before it, and the drafter's
/// full-tile context pass over a short last prefill tile.
const HW_SLACK_ROWS: u64 = 512;

/// Environment variable selecting the reset mode.
pub(super) const RESET_ENV: &str = "GAUDIUM_REQUEST_RESET";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResetMode {
    /// DeltaNet start state only; the written KV rows too after a request that ended in an error.
    Auto,
    /// Also the KV / drafter-cache rows earlier requests may have written (high-water mark).
    Dirty,
    /// Every KV cache, state ring and drafter cache, whole.
    Full,
}

impl ResetMode {
    pub(super) fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "dirty" => Some(Self::Dirty),
            "full" => Some(Self::Full),
            _ => None,
        }
    }

    /// The configured mode, else the one named by `GAUDIUM_REQUEST_RESET`, else `auto`.
    pub(super) fn from_config(configured: Option<&str>) -> Result<Self, String> {
        let from_env = std::env::var(RESET_ENV).ok();
        let Some(v) = configured.or(from_env.as_deref()) else {
            return Ok(Self::Auto);
        };
        Self::parse(v).ok_or_else(|| {
            format!("request reset mode '{v}' ({RESET_ENV} / --request-reset): expected auto, dirty or full")
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Dirty => "dirty",
            Self::Full => "full",
        }
    }
}

/// How the rows of one K/V cache are laid out in its buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CacheLayout {
    /// `{DH, KV_CHUNK, heads x chunks}`, chunk-major: chunk `c` is one contiguous byte range.
    ChunkMajor { chunks: u64 },
    /// `{DH, rows, heads}`, head-major: each head holds `rows` contiguous rows.
    HeadMajor { heads: u64, rows: u64 },
}

/// Byte ranges `(offset, len)` of a cache of `cache_bytes` that can hold rows `[0, dirty_rows)`.
///
/// Rounded outward to what the kernels address as a unit: whole chunks of a chunk-major cache,
/// whole 128-row blocks of a head-major one (V rows are stored permuted inside such a block).
pub(super) fn dirty_ranges(
    layout: CacheLayout,
    cache_bytes: u64,
    dirty_rows: u64,
) -> Vec<(u64, u64)> {
    if dirty_rows == 0 || cache_bytes == 0 {
        return Vec::new();
    }
    match layout {
        CacheLayout::ChunkMajor { chunks } => {
            let chunks = chunks.max(1);
            let n = dirty_rows.div_ceil(KV_CHUNK).min(chunks);
            vec![(0, (cache_bytes / chunks) * n)]
        }
        CacheLayout::HeadMajor { heads, rows } => {
            let (heads, rows) = (heads.max(1), rows.max(1));
            let per_head = cache_bytes / heads;
            let row_bytes = per_head / rows;
            let n = dirty_rows.next_multiple_of(128).min(rows);
            if n == rows {
                return vec![(0, cache_bytes)];
            }
            (0..heads).map(|h| (h * per_head, n * row_bytes)).collect()
        }
    }
}

/// Upper bound of the rows a request with `prompt` rows and at most `new_tokens` generated ones
/// can have written, capped at the context capacity.
pub(super) fn request_high_water(prompt: u64, new_tokens: u64, max_ctx: u64) -> u64 {
    prompt
        .saturating_add(new_tokens)
        .saturating_add(HW_SLACK_ROWS)
        .min(max_ctx)
}

/// Reset bookkeeping of one loaded engine.
#[derive(Debug)]
pub(super) struct ResetState {
    mode: ResetMode,
    verbose: bool,
    /// Rows `[0, kv_hw)` of the KV / drafter caches may hold rows written since they were last
    /// zeroed; every row at or above it is zero.
    kv_hw: u64,
    /// `kv_hw` as the running request found it (after its own reset).
    held_hw: u64,
    /// Set while a request runs: still set at the next reset when it ended with an error.
    unclean: bool,
    /// Wall time of the last reset, enqueue to completion on the device.
    last_ms: f64,
    /// Device bytes the last reset zeroed.
    last_bytes: u64,
}

impl ResetState {
    pub(super) fn new(mode: ResetMode, verbose: bool) -> Self {
        Self {
            mode,
            verbose,
            kv_hw: 0,
            held_hw: 0,
            unclean: false,
            last_ms: 0.0,
            last_bytes: 0,
        }
    }

    pub(super) fn last_ms(&self) -> f64 {
        self.last_ms
    }
}

/// Enqueues the load-time zero fill of every K/V cache and state ring (`regions` = `(addr,
/// bytes)`), so that a masked row never holds anything but zero or a row the engine wrote. The
/// fill runs on the compute stream while the weights upload on the host-to-device stream; any
/// later launch is ordered after it.
pub(super) fn enqueue_load_zero(dev: &HpuDevice, regions: &[(u64, u64)]) -> Result<u64, String> {
    let mut total = 0u64;
    for &(addr, bytes) in regions {
        dev.zero_hbm_compute(addr, bytes)
            .map_err(|e| format!("zero device state at load: {e}"))?;
        total += bytes;
    }
    Ok(total)
}

/// What one reset covers beyond the DeltaNet start state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    StateOnly,
    Dirty,
    Full,
}

fn scope_of(mode: ResetMode, unclean: bool) -> Scope {
    match (mode, unclean) {
        (ResetMode::Full, _) => Scope::Full,
        (ResetMode::Dirty, _) | (ResetMode::Auto, true) => Scope::Dirty,
        (ResetMode::Auto, false) => Scope::StateOnly,
    }
}

/// The reset before a request of `prompt_len` rows that may generate `max_new` tokens.
pub(super) fn begin_request(
    eng: &mut Qwen3Engine,
    prompt_len: u64,
    max_new: u64,
) -> Result<(), String> {
    let t0 = Instant::now();
    let scope = scope_of(eng.reset.mode, eng.reset.unclean);
    let dirty_rows = eng.reset.kv_hw;
    let ring_slots = eng.specdec.as_ref().map_or(1, |sd| sd.gdn_slots());

    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for st in eng.gdn_state.values() {
        if scope == Scope::Full {
            for p in 0..2 {
                ranges.push((st.s[p], st.s_bytes * ring_slots));
                ranges.push((st.cs[p], st.cs_bytes * ring_slots));
            }
        } else {
            ranges.push((st.s[0], st.s_bytes));
            ranges.push((st.cs[0], st.cs_bytes));
        }
    }
    if scope != Scope::StateOnly {
        let rows = if scope == Scope::Full {
            eng.max_ctx
        } else {
            dirty_rows
        };
        let trunk = CacheLayout::ChunkMajor {
            chunks: eng.max_chunks,
        };
        for &(addr, bytes) in eng.kv_addr.values() {
            for (off, len) in dirty_ranges(trunk, bytes, rows) {
                ranges.push((addr + off, len));
            }
        }
        for (layout, addr, bytes) in spec::request_caches(eng) {
            for (off, len) in dirty_ranges(layout, bytes, rows) {
                ranges.push((addr + off, len));
            }
        }
    }

    // Enqueue every fill, then wait once. A failed enqueue still waits for what was queued.
    let mut bytes = 0u64;
    let mut res: Result<(), String> = Ok(());
    for &(addr, len) in &ranges {
        if let Err(e) = eng.dev.zero_hbm_compute(addr, len) {
            res = Err(format!("request reset: {e}"));
            break;
        }
        bytes += len;
    }
    // SAFETY: the compute stream belongs to `eng.dev`.
    let synced = unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
        .map_err(|e| format!("request reset: sync: {e}"));
    res.and(synced)?;

    let kept = if scope == Scope::StateOnly {
        dirty_rows
    } else {
        0
    };
    let st = &mut eng.reset;
    st.held_hw = kept;
    // Until the request reports what it wrote: everything its budget allows.
    st.kv_hw = kept.max(request_high_water(prompt_len, max_new, eng.max_ctx));
    st.unclean = true;
    st.last_ms = t0.elapsed().as_secs_f64() * 1000.0;
    st.last_bytes = bytes;
    if st.verbose {
        eprintln!(
            "[gaudium] request reset: {:.2} ms, {:.1} MiB zeroed on the device in {} fills (mode {}{}; KV rows below {} {})",
            st.last_ms,
            bytes as f64 / (1024.0 * 1024.0),
            ranges.len(),
            st.mode.name(),
            if scope == Scope::Dirty && st.mode == ResetMode::Auto {
                ", after a failed request"
            } else {
                ""
            },
            dirty_rows,
            if scope == Scope::StateOnly {
                "kept (masked)"
            } else {
                "zeroed"
            },
        );
    }
    Ok(())
}

/// End of a request: a completed one replaces its budget bound by the rows it really wrote; a
/// failed one keeps the bound and makes the next reset zero them.
pub(super) fn end_request(
    eng: &mut Qwen3Engine,
    prompt_len: u64,
    res: &Result<Qwen3GenReport, String>,
) {
    let max_ctx = eng.max_ctx;
    let st = &mut eng.reset;
    if let Ok(report) = res {
        let wrote = request_high_water(prompt_len, report.generated_tokens.len() as u64, max_ctx);
        st.kv_hw = st.held_hw.max(wrote);
        st.unclean = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_names_parse_and_reject_unknown() {
        assert_eq!(ResetMode::parse(""), Some(ResetMode::Auto));
        assert_eq!(ResetMode::parse("auto"), Some(ResetMode::Auto));
        assert_eq!(ResetMode::parse(" Dirty "), Some(ResetMode::Dirty));
        assert_eq!(ResetMode::parse("FULL"), Some(ResetMode::Full));
        assert_eq!(ResetMode::parse("off"), None);
    }

    #[test]
    fn scope_follows_mode_and_the_previous_outcome() {
        assert_eq!(scope_of(ResetMode::Auto, false), Scope::StateOnly);
        assert_eq!(scope_of(ResetMode::Auto, true), Scope::Dirty);
        assert_eq!(scope_of(ResetMode::Dirty, false), Scope::Dirty);
        assert_eq!(scope_of(ResetMode::Full, false), Scope::Full);
        assert_eq!(scope_of(ResetMode::Full, true), Scope::Full);
    }

    #[test]
    fn chunk_major_cache_zeroes_a_prefix_of_whole_chunks() {
        // 27B trunk cache at 8 chunks: DH 256 x 4096 rows x 4 heads x 2 bytes per chunk
        let chunk = 256 * 4096 * 4 * 2u64;
        let l = CacheLayout::ChunkMajor { chunks: 8 };
        assert!(dirty_ranges(l, 8 * chunk, 0).is_empty());
        assert_eq!(dirty_ranges(l, 8 * chunk, 1), vec![(0, chunk)]);
        assert_eq!(dirty_ranges(l, 8 * chunk, 4096), vec![(0, chunk)]);
        assert_eq!(dirty_ranges(l, 8 * chunk, 4097), vec![(0, 2 * chunk)]);
        // never past the cache, whatever the mark says
        assert_eq!(dirty_ranges(l, 8 * chunk, 1 << 40), vec![(0, 8 * chunk)]);
    }

    #[test]
    fn head_major_cache_zeroes_whole_128_row_blocks_per_head() {
        // drafter cache {128, 8192, 8} bf16: 256 bytes per row, 8 heads
        let (rows, heads) = (8192u64, 8u64);
        let bytes = 128 * 2 * rows * heads;
        let l = CacheLayout::HeadMajor { heads, rows };
        let r = dirty_ranges(l, bytes, 300);
        assert_eq!(r.len(), 8);
        // 300 rows -> 384 (three blocks of 128) per head
        assert_eq!(r[0], (0, 384 * 256));
        assert_eq!(r[1], (rows * 256, 384 * 256));
        assert_eq!(r[7].0 + r[7].1, 7 * rows * 256 + 384 * 256);
        // a mark at or past the cache rows is one fill of the whole cache
        assert_eq!(dirty_ranges(l, bytes, 8192), vec![(0, bytes)]);
        assert_eq!(dirty_ranges(l, bytes, 8100), vec![(0, bytes)]);
        // every range is a multiple of 4 bytes (the device fill works in 32-bit elements)
        assert!(r.iter().all(|&(o, n)| o % 4 == 0 && n % 4 == 0));
    }

    #[test]
    fn high_water_adds_the_step_slack_and_stops_at_the_context() {
        assert_eq!(
            request_high_water(20, 236, 131072),
            20 + 236 + HW_SLACK_ROWS
        );
        assert_eq!(request_high_water(130_000, 4096, 131072), 131072);
        assert_eq!(request_high_water(u64::MAX, 5, 4096), 4096);
    }
}
