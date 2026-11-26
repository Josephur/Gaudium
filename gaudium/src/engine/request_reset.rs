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
use crate::verified as v;

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

    fn to_verified(self) -> v::RrMode {
        match self {
            Self::Auto => v::RrMode::RrModeAuto,
            Self::Dirty => v::RrMode::RrModeDirty,
            Self::Full => v::RrMode::RrModeFull,
        }
    }

    fn from_verified(m: v::RrMode) -> Self {
        match m {
            v::RrMode::RrModeAuto => Self::Auto,
            v::RrMode::RrModeDirty => Self::Dirty,
            v::RrMode::RrModeFull => Self::Full,
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
            let bytes = v::rr_chunk_dirty_bytes(cache_bytes, chunks, dirty_rows, KV_CHUNK);
            if bytes == 0 {
                Vec::new()
            } else {
                vec![(0, bytes)]
            }
        }
        CacheLayout::HeadMajor { heads, rows } => {
            let heads_c = heads.max(1);
            let per_head = cache_bytes / heads_c;
            let n = v::rr_head_dirty_rows(rows, dirty_rows, 128);
            if n == 0 {
                return Vec::new();
            }
            if v::rr_head_full_span(rows, n) {
                return vec![(0, cache_bytes)];
            }
            let span = v::rr_head_span_bytes(cache_bytes, heads, rows, n);
            (0..heads_c).map(|h| (h * per_head, span)).collect()
        }
    }
}

/// Upper bound of the rows a request with `prompt` rows and at most `new_tokens` generated ones
/// can have written, capped at the context capacity.
pub(super) fn request_high_water(prompt: u64, new_tokens: u64, max_ctx: u64) -> u64 {
    v::rr_high_water(prompt, new_tokens, HW_SLACK_ROWS, max_ctx)
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

    /// Does the reset of the next request zero K/V rows (a reset mode that does so for every
    /// request, or the reset after a request that ended with an error)? A request after such a
    /// reset cannot take anything over from the one before it.
    pub(super) fn zeroes_rows(&self) -> bool {
        !v::rr_is_state_only(self.mode.to_verified(), self.unclean)
    }

    fn as_verified(&self) -> v::RrState {
        v::RrState::RrState(
            self.mode.to_verified(),
            self.kv_hw,
            self.held_hw,
            self.unclean,
        )
    }

    fn apply_verified(&mut self, next: v::RrState) {
        self.mode = ResetMode::from_verified(v::rr_state_mode(next));
        self.kv_hw = v::rr_state_hw(next);
        self.held_hw = v::rr_state_held(next);
        self.unclean = v::rr_state_unclean(next);
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

impl Scope {
    fn to_verified(self) -> v::RrScope {
        match self {
            Self::StateOnly => v::RrScope::RrScopeState,
            Self::Dirty => v::RrScope::RrScopeDirty,
            Self::Full => v::RrScope::RrScopeFull,
        }
    }

    fn from_verified(sc: v::RrScope) -> Self {
        match sc {
            v::RrScope::RrScopeState => Self::StateOnly,
            v::RrScope::RrScopeDirty => Self::Dirty,
            v::RrScope::RrScopeFull => Self::Full,
        }
    }
}

fn scope_of(mode: ResetMode, unclean: bool) -> Scope {
    Scope::from_verified(v::rr_scope(mode.to_verified(), unclean))
}

fn push_gdn_zero_ranges(
    ranges: &mut Vec<(u64, u64)>,
    scope: Scope,
    s_addrs: [u64; 2],
    s_one_bytes: u64,
    s_slots: u64,
    cs_addrs: [u64; 2],
    cs_one_bytes: u64,
    cs_slots: u64,
) {
    let sc = scope.to_verified();
    let sb = v::rr_gdn_zero_bytes(sc, s_one_bytes, s_slots);
    let csb = v::rr_gdn_zero_bytes(sc, cs_one_bytes, cs_slots);
    if v::rr_zero_both_rings(sc) {
        for p in 0..2 {
            ranges.push((s_addrs[p], sb));
            ranges.push((cs_addrs[p], csb));
        }
    } else {
        ranges.push((s_addrs[0], sb));
        ranges.push((cs_addrs[0], csb));
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
    let s_slots = eng.specdec.as_ref().map_or(1, |sd| sd.gdn_state_slots());
    let cs_slots = eng.specdec.as_ref().map_or(1, |sd| sd.gdn_slots());

    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for st in eng.gdn_state.values() {
        push_gdn_zero_ranges(
            &mut ranges,
            scope,
            st.s,
            st.s_bytes,
            s_slots,
            st.cs,
            st.cs_bytes,
            cs_slots,
        );
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

    let st = &mut eng.reset;
    let next = v::rr_begin(
        st.as_verified(),
        prompt_len,
        max_new,
        HW_SLACK_ROWS,
        eng.max_ctx,
    );
    st.apply_verified(next);
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
        let next = v::rr_end_ok(
            st.as_verified(),
            prompt_len,
            report.generated_tokens.len() as u64,
            HW_SLACK_ROWS,
            max_ctx,
        );
        st.apply_verified(next);
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

    #[test]
    fn full_reset_uses_distinct_gdn_state_and_conv_slot_counts() {
        let mut ranges = Vec::new();
        // Recurrent state (`s`) keeps 1 slot under replay while conv state (`cs`) has 32 slots.
        push_gdn_zero_ranges(
            &mut ranges,
            Scope::Full,
            [0x1000, 0x2000],
            4096,
            1,
            [0x3000, 0x4000],
            512,
            32,
        );
        assert_eq!(
            ranges,
            vec![
                (0x1000, 4096),
                (0x3000, 512 * 32),
                (0x2000, 4096),
                (0x4000, 512 * 32),
            ]
        );

        let mut state_only = Vec::new();
        push_gdn_zero_ranges(
            &mut state_only,
            Scope::StateOnly,
            [0x1000, 0x2000],
            4096,
            1,
            [0x3000, 0x4000],
            512,
            32,
        );
        assert_eq!(state_only, vec![(0x1000, 4096), (0x3000, 512)]);
    }

    #[test]
    fn head_and_chunk_dirty_ranges_handle_u64_max_without_overflow() {
        let head = CacheLayout::HeadMajor {
            heads: 8,
            rows: 8192,
        };
        let bytes = 8 * 8192 * 256u64;
        assert_eq!(dirty_ranges(head, bytes, u64::MAX), vec![(0, bytes)]);

        let chunk = CacheLayout::ChunkMajor { chunks: 8 };
        let cbytes = 8 * 1024 * 1024u64;
        assert_eq!(dirty_ranges(chunk, cbytes, u64::MAX), vec![(0, cbytes)]);
    }

    #[test]
    fn reset_state_escalates_to_dirty_after_failed_request() {
        let mut st = ResetState::new(ResetMode::Auto, false);
        assert!(!st.zeroes_rows());

        let next = v::rr_begin(st.as_verified(), 100, 200, HW_SLACK_ROWS, 4096);
        st.apply_verified(next);
        assert!(st.unclean);
        assert!(st.zeroes_rows());

        let done = v::rr_end_ok(st.as_verified(), 100, 20, HW_SLACK_ROWS, 4096);
        st.apply_verified(done);
        assert!(!st.unclean);
        assert!(!st.zeroes_rows());
        assert_eq!(st.kv_hw, 120 + HW_SLACK_ROWS);
    }
}
