//! `n` choices of one chat completion (`n` on `POST /v1/chat/completions`).
//!
//! A request with `n` choices runs the generations the verified `ch_runs` gives it: one per
//! sampled choice, each with a seed of its own ([`choice_seed`]), and one for all the choices of a
//! greedy request (the engine's greedy stream does not depend on what runs beside it, so every
//! choice is a copy of that one generation). With lanes (`--parallel`) the generations are queued
//! together and the ones that find a free lane run beside each other; a lane that takes one copies
//! the start of the prompt from the lane that prefilled it (`ln_fork_pick`), so the prompt is
//! prefilled once and each later generation prefills only the rows after the last whole tile.
//! Without lanes they run one after another in the one lane, which resumes each later one from the
//! position its prefix table kept.
//!
//! The generations of a request form a group ([`ChoiceGroup`]): each has a stop flag of its own
//! (a stop string ends that choice only), a cancel of the request stops all of them, and the
//! request answers only once every one delivered its end (`ch_step`, `ch_done`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::json::JsonValue;
use crate::verified::{
    ch_cancelled, ch_choice_seed, ch_done, ch_n_ok, ch_new, ch_run_of, ch_seed_top, ch_step,
    ChEvent, ChGroup,
};

/// The most choices `--max-choices` may allow.
pub const MAX_CHOICES: u64 = 64;

/// Choices a request may ask for when the command line names no bound.
pub const DEFAULT_MAX_CHOICES: u64 = 8;

/// The seed of choice `i` of a request whose seed is `seed`: the verified `ch_choice_seed`, which
/// today counts the request's seed up by `i` and wraps past `u64::MAX` to 0 (`ch_seed_top`).
/// Choice 0 runs with the request's own seed, so `n = 1` is the request it always was, no two
/// choices of one request share a seed, and choice `i` is token-identical to the same request
/// sent alone with the seed it reports (and `n = 1`).
pub fn choice_seed(seed: u64, i: u64) -> u64 {
    ch_choice_seed(seed, i, ch_seed_top())
}

/// The generation choice `c` reads, of the `runs` (at least one) the request's choices take: the
/// verified `ch_run_of`, which keeps it below `runs` and gives every choice that has a generation
/// of its own that one.
pub fn run_of(c: usize, runs: usize) -> usize {
    ch_run_of(c as u64, runs as u64) as usize
}

/// The choices, of `n`, that generation `r` of `runs` writes: those that read it ([`run_of`]).
pub fn choices_of(r: usize, runs: usize, n: usize) -> Vec<usize> {
    (0..n).filter(|&c| run_of(c, runs) == r).collect()
}

/// The completion tokens of `n` choices whose generations generated `per_run` tokens each: every
/// choice counts the tokens it returns, those of the generation it reads.
pub fn completion_tokens(per_run: &[u64], n: usize) -> u64 {
    (0..n).map(|c| per_run[run_of(c, per_run.len())]).sum()
}

/// The `n` of a request body: absent or `null` is one choice; anything else must be a whole
/// number the verified `ch_n_ok` admits under `max`. The error is the 400 message.
pub fn requested_n(body: &JsonValue, max: u64) -> Result<u64, String> {
    let n = match body.get("n") {
        None | Some(JsonValue::Null) => return Ok(1),
        Some(v) => v
            .as_u64()
            .ok_or_else(|| format!("'n' must be a whole number of choices from 1 to {max}"))?,
    };
    if ch_n_ok(n, max) {
        Ok(n)
    } else {
        Err(format!(
            "'n' = {n} choices: this server returns 1 to {max} choices per request (--max-choices)"
        ))
    }
}

/// The generations of one request: their stop flags and the verified group state, under one
/// lock (a cancel and a renewed flag never miss each other).
pub(super) struct ChoiceGroup {
    inner: Mutex<GroupInner>,
}

struct GroupInner {
    flags: Vec<Arc<AtomicBool>>,
    group: ChGroup,
}

impl ChoiceGroup {
    pub fn new(runs: usize) -> Self {
        Self {
            inner: Mutex::new(GroupInner {
                flags: (0..runs)
                    .map(|_| Arc::new(AtomicBool::new(false)))
                    .collect(),
                group: ch_new(runs as u64),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GroupInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The current stop flag of generation `r` (the engine checks it between synced launches).
    pub fn flag(&self, r: usize) -> Arc<AtomicBool> {
        Arc::clone(&self.lock().flags[r])
    }

    /// Cancels the request: every generation is told to stop. Idempotent.
    pub fn cancel_all(&self) {
        let mut g = self.lock();
        g.group = ch_step(g.group, ChEvent::ChCancelAll);
        if ch_cancelled(g.group) {
            for f in &g.flags {
                f.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Stops generation `r` where it is and gives it a fresh flag for what it runs next (a reply
    /// resumed after its thinking budget's forced close). The fresh flag is already set when
    /// the request was cancelled; the returned `bool` says whether it was.
    pub fn renew(&self, r: usize) -> (Arc<AtomicBool>, bool) {
        let mut g = self.lock();
        let cancelled = ch_cancelled(g.group);
        g.flags[r].store(true, Ordering::Relaxed);
        let fresh = Arc::new(AtomicBool::new(cancelled));
        g.flags[r] = Arc::clone(&fresh);
        (fresh, cancelled)
    }

    /// One generation delivered its end.
    pub fn run_ended(&self) {
        let mut g = self.lock();
        g.group = ch_step(g.group, ChEvent::ChRunEnded);
    }

    /// Every generation delivered its end: the request may answer.
    pub fn done(&self) -> bool {
        ch_done(self.lock().group)
    }

    /// The request was cancelled (a cancel, a DELETE of its response, or the client gone).
    pub fn cancelled(&self) -> bool {
        ch_cancelled(self.lock().group)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn choice_seeds_start_at_the_request_seed_and_wrap() {
        assert_eq!(choice_seed(42, 0), 42);
        assert_eq!(choice_seed(42, 2), 44);
        assert_eq!(ch_seed_top(), u64::MAX, "seeds are 64-bit");
        assert_eq!(choice_seed(u64::MAX, 1), 0);
        assert_eq!(choice_seed(u64::MAX - 1, 1), u64::MAX);
        assert_eq!(choice_seed(u64::MAX - 1, 3), 1);
        assert_eq!(choice_seed(0, u64::MAX), u64::MAX);
        for (s, i) in [
            (7u64, 5u64),
            (u64::MAX - 2, 9),
            (u64::MAX, u64::MAX),
            (1 << 63, 1 << 63),
        ] {
            assert_eq!(choice_seed(s, i), s.wrapping_add(i), "seed + i modulo 2^64");
        }
        let seeds: Vec<u64> = (0..8).map(|i| choice_seed(u64::MAX - 3, i)).collect();
        let mut uniq = seeds.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            seeds.len(),
            "choices of a request never share a seed"
        );
    }

    #[test]
    fn every_choice_reads_one_generation_whatever_the_generation_count() {
        // every count of generations from one to n, the middle ones too: each choice is written
        // by exactly one generation, its own when it has one, and usage counts what each returns
        for n in 1..=6usize {
            for runs in 1..=n {
                let mut writers = vec![0usize; n];
                for r in 0..runs {
                    let cs = choices_of(r, runs, n);
                    assert!(!cs.is_empty(), "generation {r} of {runs} is read by a choice");
                    for c in cs {
                        assert_eq!(run_of(c, runs), r);
                        writers[c] += 1;
                    }
                }
                assert!(writers.iter().all(|&w| w == 1), "n={n} runs={runs}: {writers:?}");
                for c in 0..runs {
                    assert_eq!(run_of(c, runs), c, "a choice with a generation of its own");
                }
                let per: Vec<u64> = (0..runs as u64).map(|r| 10 + r).collect();
                let want: u64 = (0..n).map(|c| per[run_of(c, runs)]).sum();
                assert_eq!(completion_tokens(&per, n), want);
            }
        }
        // today's two shapes: one generation for every choice, or one each
        assert_eq!(choices_of(0, 1, 3), vec![0, 1, 2]);
        assert_eq!(completion_tokens(&[7], 3), 21);
        assert_eq!(choices_of(1, 3, 3), vec![1]);
        assert_eq!(completion_tokens(&[1, 2, 3], 3), 6);
        // a middle count: two generations for five choices, never interleaved into one choice
        assert_eq!(choices_of(0, 2, 5), vec![0, 2, 4]);
        assert_eq!(choices_of(1, 2, 5), vec![1, 3]);
        assert_eq!(completion_tokens(&[4, 9], 5), 4 + 9 + 4 + 9 + 4);
    }

    #[test]
    fn n_is_one_by_default_and_refused_outside_the_bound() {
        let n = |s: &str| requested_n(&json::parse_json(s).unwrap(), 4);
        assert_eq!(n("{}"), Ok(1));
        assert_eq!(n(r#"{"n":null}"#), Ok(1));
        assert_eq!(n(r#"{"n":3}"#), Ok(3));
        assert_eq!(n(r#"{"n":4}"#), Ok(4));
        let e = n(r#"{"n":5}"#).unwrap_err();
        assert!(
            e.contains("5") && e.contains("1 to 4") && e.contains("--max-choices"),
            "{e}"
        );
        assert!(n(r#"{"n":0}"#).is_err());
        assert!(n(r#"{"n":-1}"#).is_err());
        assert!(n(r#"{"n":1.5}"#).is_err());
        assert!(n(r#"{"n":"2"}"#).is_err());
    }

    #[test]
    fn a_group_answers_after_every_run_and_cancel_stops_them_all() {
        let g = ChoiceGroup::new(3);
        assert!(!g.done());
        g.run_ended();
        g.cancel_all();
        g.cancel_all();
        assert!(g.cancelled());
        assert!((0..3).all(|r| g.flag(r).load(Ordering::Relaxed)));
        assert!(
            !g.done(),
            "a cancel waits for the runs still in their lanes"
        );
        g.run_ended();
        g.run_ended();
        assert!(g.done());
        g.run_ended();
        assert!(g.done(), "an end past the count changes nothing");
    }

    #[test]
    fn a_renewed_flag_stops_the_old_run_and_follows_the_request_cancel() {
        let g = ChoiceGroup::new(1);
        let old = g.flag(0);
        let (fresh, cancelled) = g.renew(0);
        assert!(!cancelled);
        assert!(old.load(Ordering::Relaxed), "the cut generation is stopped");
        assert!(!fresh.load(Ordering::Relaxed));
        assert!(Arc::ptr_eq(&fresh, &g.flag(0)), "the generation runs on under the fresh flag");
        // a cancel after the renew reaches the fresh flag
        g.cancel_all();
        assert!(fresh.load(Ordering::Relaxed));
        // a renew after a cancel starts cancelled, and says so
        let (later, cancelled) = g.renew(0);
        assert!(cancelled && later.load(Ordering::Relaxed));
        assert!(!g.done(), "a renew is not an end");
    }
}
