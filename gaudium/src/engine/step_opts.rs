//! Options of the decode step recipes that are not part of a model's or a drafter's contract:
//! how the verify rows of the Gated DeltaNet layers keep their state, the precision and the
//! readout width of an on-card drafter, and the device trace of chosen launches.
//!
//! Defaults come from [`StepOpts::from_env`], so a process that builds its engine configuration
//! without knowing these options (the server) can still be steered through the environment; the
//! command line overrides them.

use std::path::PathBuf;

use crate::synapse::profiler::ProfileWindow;

/// Draft-vocabulary rows of an on-card block drafter's readout when nothing else is asked for:
/// the lowest token ids (BPE merge order, most frequent first) plus the control-token block.
pub const DEFAULT_DRAFT_VOCAB: u64 = 98_304;

/// How a speculative step carries the Gated DeltaNet recurrent state across its verify rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GdnVerify {
    /// One state checkpoint per verify row; the next step starts from the accepted row's
    /// checkpoint. Token-identical to `K = 0`.
    #[default]
    Checkpoint,
    /// No per-row checkpoints: every row keeps its rank-1 update, and the next step applies the
    /// accepted rows' updates to its start state again, with the row kernel's own arithmetic in
    /// the row kernel's own order. Token-identical to `K = 0`.
    Replay,
    /// The rows read the step's start state through batched GEMMs on the MME and the accepted
    /// rows are committed in one closed form. Faster per verify row, but NOT token-identical to
    /// `K = 0`: another fp32 summation order, so a stream may fork at a near-tie of the logits.
    /// Opt-in only.
    Mme,
}

impl GdnVerify {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "checkpoint" | "ckpt" => Some(Self::Checkpoint),
            "replay" => Some(Self::Replay),
            "mme" => Some(Self::Mme),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Checkpoint => "checkpoint",
            Self::Replay => "replay",
            Self::Mme => "mme",
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepOpts {
    /// Device traces of decode launches (`--profile-steps`); recipes compile fresh when set.
    pub profile: Vec<ProfileWindow>,
    /// Rows of the block drafter's readout (DSpine, DFlash2): `None` = [`DEFAULT_DRAFT_VOCAB`]
    /// when the vocabulary is larger, `Some(0)` = the full vocabulary.
    pub draft_vocab: Option<u64>,
    /// FP8 weight GEMMs in the DSpine drafter's layers: `None` = on when the drafter's
    /// activation-scale table exists, `Some(false)` = BF16.
    pub draft_fp8: Option<bool>,
    /// The drafter's activation-scale table (default `<draft dir>-gaudium-pc/act_scales_ds.txt`).
    pub draft_fp8_act: Option<PathBuf>,
    /// Calibration run: the drafter stays BF16, its GEMM input sites are read back after every
    /// launch, and the table is written here when the process ends.
    pub draft_fp8_calibrate: Option<PathBuf>,
    /// Context caches of a block drafter with a sliding window as rings of that window instead
    /// of one row per context position (drafts only; the step then costs the same at every
    /// context size). `None`: when the ring kernels and their glue library are installed.
    pub draft_ring: Option<bool>,
    /// DeltaNet state handling of the verify rows.
    pub gdn_verify: GdnVerify,
    /// `gdn_verify` was named (`--gdn-verify`, `GAUDIUM_GDN_VERIFY`); otherwise a load of several
    /// lanes takes the replay form (`engine::lane_verify`).
    pub gdn_verify_asked: bool,
    /// Checkpoint form: run the rows with the kernel build that takes its per-row gate scalars as
    /// vector loads (bit-identical, fewer slow scalar loads from device memory). `None`: when
    /// that kernel and its glue library are installed.
    pub gdn_fast_rows: Option<bool>,
    /// Greedy `K = 0` requests of the dense models decode enqueue-ahead: the next step is queued
    /// while the current one runs, its token fed back on the card (same recipes, same tokens).
    pub k0_pipeline: bool,
}

impl Default for StepOpts {
    fn default() -> Self {
        Self {
            profile: Vec::new(),
            draft_vocab: None,
            draft_fp8: None,
            draft_fp8_act: None,
            draft_fp8_calibrate: None,
            draft_ring: None,
            gdn_verify: GdnVerify::default(),
            gdn_verify_asked: false,
            gdn_fast_rows: None,
            k0_pipeline: true,
        }
    }
}

impl StepOpts {
    /// Defaults, with the environment's overrides: `GAUDIUM_DRAFT_VOCAB` (rows, 0 = full),
    /// `GAUDIUM_DRAFT_FP8` (0 / 1), `GAUDIUM_DRAFT_FP8_ACT` (table path), `GAUDIUM_DRAFT_RING`
    /// (0 / 1), `GAUDIUM_GDN_VERIFY` (`checkpoint` / `replay` / `mme`), `GAUDIUM_GDN_FAST_ROWS`
    /// (0 / 1), `GAUDIUM_K0_PIPELINE` (0 / 1). A value that does not parse is ignored.
    pub fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let switch = |k: &str| {
            var(k).and_then(|v| match v.trim() {
                "0" => Some(false),
                "1" => Some(true),
                _ => None,
            })
        };
        Self {
            profile: Vec::new(),
            draft_vocab: var("GAUDIUM_DRAFT_VOCAB").and_then(|v| v.trim().parse().ok()),
            draft_fp8: var("GAUDIUM_DRAFT_FP8").and_then(|v| match v.trim() {
                "0" => Some(false),
                "1" => Some(true),
                _ => None,
            }),
            draft_fp8_act: var("GAUDIUM_DRAFT_FP8_ACT").map(PathBuf::from),
            draft_fp8_calibrate: None,
            draft_ring: switch("GAUDIUM_DRAFT_RING"),
            gdn_verify: var("GAUDIUM_GDN_VERIFY")
                .and_then(|v| GdnVerify::parse(v.trim()))
                .unwrap_or_default(),
            gdn_verify_asked: var("GAUDIUM_GDN_VERIFY").is_some_and(|v| GdnVerify::parse(v.trim()).is_some()),
            gdn_fast_rows: var("GAUDIUM_GDN_FAST_ROWS").and_then(|v| match v.trim() {
                "0" => Some(false),
                "1" => Some(true),
                _ => None,
            }),
            k0_pipeline: var("GAUDIUM_K0_PIPELINE").is_none_or(|v| v.trim() != "0"),
        }
    }

    /// Readout rows of a block drafter over a vocabulary of `vocab` ids with `n_ctl` control
    /// tokens that must stay reachable: 0 = the full vocabulary.
    pub fn draft_rows(&self, vocab: u64, n_ctl: u64) -> Result<u64, String> {
        let n = self.draft_vocab.unwrap_or(DEFAULT_DRAFT_VOCAB);
        if n == 0 || n >= vocab {
            return Ok(0);
        }
        if !n.is_multiple_of(128) || n < 1024 || n <= n_ctl {
            return Err(format!(
                "draft vocabulary {n}: need 0 or a multiple of 128 in [1024, {vocab}) above the {n_ctl} control tokens"
            ));
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_rows_default_and_bounds() {
        let o = StepOpts::default();
        assert_eq!(o.draft_rows(248_320, 33), Ok(DEFAULT_DRAFT_VOCAB));
        // a vocabulary no larger than the default keeps the full readout
        assert_eq!(o.draft_rows(65_536, 33), Ok(0));
        let full = StepOpts {
            draft_vocab: Some(0),
            ..StepOpts::default()
        };
        assert_eq!(full.draft_rows(248_320, 33), Ok(0));
        let odd = StepOpts {
            draft_vocab: Some(1000),
            ..StepOpts::default()
        };
        assert!(odd.draft_rows(248_320, 33).is_err());
        let small = StepOpts {
            draft_vocab: Some(65_536),
            ..StepOpts::default()
        };
        assert_eq!(small.draft_rows(248_320, 33), Ok(65_536));
    }

    #[test]
    fn gdn_verify_names_round_trip() {
        for m in [GdnVerify::Checkpoint, GdnVerify::Replay, GdnVerify::Mme] {
            assert_eq!(GdnVerify::parse(m.name()), Some(m));
        }
        assert_eq!(GdnVerify::parse("fast"), None);
        assert_eq!(GdnVerify::default(), GdnVerify::Checkpoint);
    }
}
