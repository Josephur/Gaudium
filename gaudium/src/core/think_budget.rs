//! Rust Shell wrapper for the thinking budget of `verified/think_budget.bend`.
//!
//! A request with a thinking budget (`thinking.budget_tokens` of the Messages API) holds one
//! [`ThinkBudget`]: each generated token is shown to [`ThinkBudget::step`] before it reaches the
//! reply, and the step says whether it goes on, whether the end-of-thinking marker takes its
//! place, or whether the reply ends. The cap, the step, the tokens the reply has written after it
//! and the budget left after the marker all come from the verified core; this type only holds
//! the counts between steps.

use crate::verified::{remaining_budget, think_act, think_cap, think_sent, ThinkAct};

/// What to do with one generated token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkStep {
    /// The token goes on into the reply.
    Keep,
    /// The token is dropped; the end-of-thinking marker is written in its place (as if
    /// generated), and the reply continues with its answer.
    Close,
    /// The token is dropped and the reply ends: the marker no longer fits in its token budget.
    End,
}

/// The thinking budget of one reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkBudget {
    cap: u64,
    close_len: u64,
    max_new: u64,
    /// reasoning tokens kept
    kept: u64,
    /// tokens of the reply so far, the forced marker included
    sent: u64,
    forced: bool,
}

impl ThinkBudget {
    /// `budget`: the request's thinking budget; `max_new`: the reply's token budget;
    /// `close_len`: the end-of-thinking marker's length in tokens.
    pub fn new(budget: u64, max_new: u64, close_len: u64) -> Self {
        Self {
            cap: think_cap(budget, max_new, close_len),
            close_len,
            max_new,
            kept: 0,
            sent: 0,
            forced: false,
        }
    }

    /// The step for the next generated token. `in_thought`: the reply is inside its reasoning
    /// span before this token.
    pub fn step(&mut self, in_thought: bool) -> ThinkStep {
        let room = remaining_budget(self.sent, self.max_new);
        let act = think_act(
            in_thought,
            self.kept,
            self.cap,
            self.close_len,
            room,
            self.forced,
        );
        self.sent = think_sent(act.clone(), self.sent, self.close_len);
        match act {
            ThinkAct::TaFeed(n) => {
                self.kept = n;
                ThinkStep::Keep
            }
            ThinkAct::TaClose => {
                self.forced = true;
                ThinkStep::Close
            }
            ThinkAct::TaEnd => ThinkStep::End,
        }
    }

    /// What the reply may still generate (after a forced marker: the answer's budget).
    pub fn remaining(&self) -> u64 {
        remaining_budget(self.sent, self.max_new)
    }

    /// The reasoning tokens kept so far.
    pub fn kept(&self) -> u64 {
        self.kept
    }

    /// The tokens of the reply so far, the forced marker included.
    pub fn sent(&self) -> u64 {
        self.sent
    }

    /// The marker was forced.
    pub fn forced(&self) -> bool {
        self.forced
    }

    /// The reasoning cap.
    pub fn cap(&self) -> u64 {
        self.cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Today's cap: the budget, cut to leave the marker room in max_new.
    #[test]
    fn the_cap_is_the_budget_unless_the_marker_would_not_fit() {
        assert_eq!(ThinkBudget::new(1024, 4096, 1).cap(), 1024);
        assert_eq!(ThinkBudget::new(1024, 1025, 1).cap(), 1024);
        assert_eq!(ThinkBudget::new(1024, 1024, 1).cap(), 1023);
        assert_eq!(ThinkBudget::new(1024, 2, 3).cap(), 0);
    }

    #[test]
    fn a_reply_that_reasons_to_the_cap_gets_the_marker_then_its_answer_budget() {
        let mut b = ThinkBudget::new(4, 10, 2);
        for _ in 0..4 {
            assert_eq!(b.step(true), ThinkStep::Keep);
        }
        assert_eq!(b.kept(), 4);
        // the fifth reasoning token is replaced by the marker
        assert_eq!(b.step(true), ThinkStep::Close);
        assert_eq!((b.sent(), b.remaining()), (6, 4));
        // the answer goes on to the end of max_new
        for _ in 0..4 {
            assert_eq!(b.step(false), ThinkStep::Keep);
        }
        assert_eq!(b.remaining(), 0);
        // reasoning that opens again after the forced marker ends the reply
        assert_eq!(b.step(true), ThinkStep::End);
    }

    #[test]
    fn a_reply_that_closes_its_own_reasoning_is_never_cut() {
        let mut b = ThinkBudget::new(4, 10, 2);
        for _ in 0..3 {
            assert_eq!(b.step(true), ThinkStep::Keep);
        }
        for _ in 0..7 {
            assert_eq!(b.step(false), ThinkStep::Keep);
        }
        assert!(!b.forced());
        assert_eq!(b.remaining(), 0);
    }

    #[test]
    fn a_marker_that_no_longer_fits_ends_the_reply() {
        // two tokens before the reasoning opens leave no room for the marker at the cap
        let mut b = ThinkBudget::new(8, 10, 2);
        assert_eq!(b.step(false), ThinkStep::Keep);
        assert_eq!(b.step(false), ThinkStep::Keep);
        for _ in 0..8 {
            assert_eq!(b.step(true), ThinkStep::Keep);
        }
        assert_eq!(b.step(true), ThinkStep::End);
        assert!(!b.forced());
    }
}
