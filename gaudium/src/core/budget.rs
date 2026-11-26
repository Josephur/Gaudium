//! Rust Shell wrappers for token/context budget arithmetic & sampler compatibility lattice.
//!
//! All prompt admission, context clamping, remaining-budget monotonicity, KV high-water
//! capping, and mixed-batch sampler lattice decisions delegate to `verified/token_budget.bend`.

use crate::verified::{
    clamp_top_k, ctx_for_prompt as v_ctx_for_prompt, ctx_reserve, effective_max_new,
    prompt_admitted, prompt_room, remaining_budget, served_ctx,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    PromptRejected { prompt_len: u64, served_ctx: u64 },
}

/// Returns the fixed engine KV reserve (`CTX_RESERVE = 128`).
pub fn engine_ctx_reserve() -> u64 {
    ctx_reserve()
}

/// Returns the advertised served context (`lmax - 128`, saturating at 0).
pub fn advertised_served_ctx(lmax: u64) -> u64 {
    served_ctx(lmax)
}

/// The context capacity a load sizes for its longest prompt of `prompt_len` tokens and one
/// generated token (`ctx_for_prompt`): any capacity at least this large, rounded up to whole
/// chunks or not, admits that prompt.
pub fn ctx_for_prompt(prompt_len: u64) -> u64 {
    v_ctx_for_prompt(prompt_len)
}

/// Checked per-sequence token budget tracker (never underflows remaining budget).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceBudget {
    lmax: u64,
    prompt_len: u64,
    max_new: u64,
    sent: u64,
}

impl SequenceBudget {
    /// Admits a prompt and computes its clamped `max_new` budget (`req_max_new == None` => full room).
    pub fn admit(
        prompt_len: u64,
        req_max_new: Option<u64>,
        lmax: u64,
    ) -> Result<Self, BudgetError> {
        if !prompt_admitted(prompt_len, lmax) {
            return Err(BudgetError::PromptRejected {
                prompt_len,
                served_ctx: served_ctx(lmax),
            });
        }
        let max_new = match req_max_new {
            None => effective_max_new(true, 0, prompt_len, lmax),
            Some(req) => effective_max_new(false, req, prompt_len, lmax),
        };
        Ok(Self {
            lmax,
            prompt_len,
            max_new,
            sent: 0,
        })
    }

    pub const fn lmax(&self) -> u64 {
        self.lmax
    }

    pub const fn prompt_len(&self) -> u64 {
        self.prompt_len
    }

    pub const fn max_new(&self) -> u64 {
        self.max_new
    }

    pub const fn sent(&self) -> u64 {
        self.sent
    }

    pub fn room(&self) -> u64 {
        prompt_room(self.prompt_len, self.lmax)
    }

    pub fn remaining(&self) -> u64 {
        remaining_budget(self.sent, self.max_new)
    }

    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }

    /// Records `count` emitted tokens and returns the updated non-underflowing remaining budget.
    pub fn record_emitted(&mut self, count: u64) -> u64 {
        self.sent = self.sent.saturating_add(count);
        self.remaining()
    }
}

/// Clamps `top_k` safely to `<= vocab`.
pub fn bounded_top_k(top_k: u64, vocab: u64) -> u64 {
    clamp_top_k(top_k, vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_load_sized_for_its_longest_prompt_admits_it() {
        use crate::verified::align_up;
        // a 4000-token prompt: 4001 rows rounded to one chunk would leave 3968 served positions
        for p in [1u64, 3967, 3968, 4000, 4095, 4096, 32_000] {
            let lmax = align_up(ctx_for_prompt(p), 4096);
            assert!(
                SequenceBudget::admit(p, Some(1), lmax).is_ok(),
                "prompt {p}, lmax {lmax}"
            );
        }
        assert_eq!(align_up(ctx_for_prompt(4000), 4096), 8192);
        assert_eq!(align_up(ctx_for_prompt(3967), 4096), 4096);
    }

    #[test]
    fn served_ctx_and_prompt_admission_boundaries() {
        assert_eq!(engine_ctx_reserve(), 128);
        assert_eq!(advertised_served_ctx(4096), 3968);
        assert_eq!(advertised_served_ctx(32768), 32640);
        assert_eq!(advertised_served_ctx(262144), 262016);
        assert_eq!(advertised_served_ctx(64), 0); // saturates safely, no underflow

        // Empty prompt always rejected
        assert_eq!(
            SequenceBudget::admit(0, Some(64), 4096),
            Err(BudgetError::PromptRejected {
                prompt_len: 0,
                served_ctx: 3968
            })
        );

        // Prompt at exact served_ctx rejected (no room left for >= 1 token)
        assert_eq!(
            SequenceBudget::admit(3968, Some(1), 4096),
            Err(BudgetError::PromptRejected {
                prompt_len: 3968,
                served_ctx: 3968
            })
        );

        // Prompt at served_ctx - 1 admitted with room == 1
        let b = SequenceBudget::admit(3967, Some(500), 4096).unwrap();
        assert_eq!(b.room(), 1);
        assert_eq!(b.max_new(), 1); // clamped to available room
        assert_eq!(b.remaining(), 1);

        // Unbounded max_new gets full remaining room
        let unb = SequenceBudget::admit(1000, None, 4096).unwrap();
        assert_eq!(unb.max_new(), 2968);
    }

    #[test]
    fn remaining_budget_monotone_and_kv_hw_clamped() {
        let mut b = SequenceBudget::admit(100, Some(5), 4096).unwrap();
        assert_eq!(b.remaining(), 5);
        assert_eq!(b.record_emitted(3), 2);
        assert!(!b.is_exhausted());
        assert_eq!(b.record_emitted(2), 0);
        assert!(b.is_exhausted());
        // Overshooting sent past max_new clamps cleanly at 0 (no underflow)
        assert_eq!(b.record_emitted(4), 0);
        assert!(b.is_exhausted());
    }

    #[test]
    fn top_k_clamping_and_reserve() {
        assert_eq!(engine_ctx_reserve(), 128);
        // Top-k clamping
        assert_eq!(bounded_top_k(50, 248_320), 50);
        assert_eq!(bounded_top_k(500_000, 248_320), 248_320);
        assert_eq!(bounded_top_k(10, 0), 0);
    }
}
