//! Lanes of one loaded engine (`--parallel N`): several requests at once, each in a lane of its
//! own (its own K/V rows, recurrent state and prefix table), decoded together in batched steps.
//!
//! Every decision delegates to the verified core: the lane lifecycle, routing and the slot table
//! of a step to `verified/lanes.bend`, admission (how many lanes may be busy, an image request
//! alone) to `lane_can_admit` in `verified/lifecycle.bend`. This module keeps the table and asks.

use crate::verified::{
    lane_can_admit, ln_allows, ln_batch_chunks, ln_in_batch, ln_is_drain, ln_is_free, ln_lanes_fit,
    ln_plan_bytes, ln_plan_fits, ln_plan_shortfall, ln_prefix_per_lane, ln_route_better,
    ln_route_eligible, ln_run_done, ln_session_admits, ln_session_ends, ln_session_starts,
    ln_session_yields, ln_slot_fits, ln_step, ln_step_chunks, turn_due, LaneEvent, LaneState,
};

/// What a free lane offers a request (routing).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Offer {
    /// Prompt rows the lane would take over (its prefill resumes there).
    pub resume: u64,
    /// Rows of the lane's held prompt that the request shares.
    pub shared: u64,
    /// Rows of the lane's held prompt the request would destroy (held, not shared).
    pub loss: u64,
}

/// A transition the lane lifecycle does not allow (the lane is left as it was).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneError {
    pub lane: usize,
    pub from: LaneState,
    pub event: LaneEvent,
}

impl std::fmt::Display for LaneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lane {}: {:?} does not allow {:?}",
            self.lane, self.from, self.event
        )
    }
}

/// The lanes of one engine: their states, whether a lane runs an exclusive (image) request,
/// and when each was last used (routing prefers the lane used longest ago between equals).
#[derive(Debug, Clone)]
pub struct LaneTable {
    states: Vec<LaneState>,
    exclusive: Vec<bool>,
    stamps: Vec<u64>,
    clock: u64,
}

impl LaneTable {
    /// `n` free lanes (a count from the verified `ln_served_lanes`, at least one).
    pub fn new(n: usize) -> Self {
        Self {
            states: vec![LaneState::LnFree; n],
            exclusive: vec![false; n],
            stamps: vec![0; n],
            clock: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn state(&self, lane: usize) -> LaneState {
        self.states[lane]
    }

    /// Lanes that hold a request (any state but free).
    pub fn busy(&self) -> u64 {
        self.states.iter().filter(|&&s| !ln_is_free(s)).count() as u64
    }

    /// A busy lane runs a request that must run alone.
    pub fn exclusive_active(&self) -> bool {
        self.states
            .iter()
            .zip(&self.exclusive)
            .any(|(&s, &x)| x && !ln_is_free(s))
    }

    /// May a request (`exclusive`: it has images and must run alone) be admitted now?
    pub fn can_admit(&self, exclusive: bool) -> bool {
        lane_can_admit(
            self.busy(),
            self.len() as u64,
            self.exclusive_active(),
            exclusive,
        )
    }

    /// The lane a request goes to: among the lanes routing may use, the best by the verified
    /// order. `reuse(lane)` = what that lane offers the request: [`Offer`]. `None`: no lane is
    /// free.
    pub fn route(&self, reuse: impl Fn(usize) -> Offer) -> Option<usize> {
        let mut best: Option<(usize, Offer)> = None;
        for (i, &s) in self.states.iter().enumerate() {
            if !ln_route_eligible(s) {
                continue;
            }
            let o = reuse(i);
            best = match best {
                Some((b, ob))
                    if !ln_route_better(
                        o.resume,
                        o.loss,
                        o.shared,
                        self.stamps[i],
                        ob.resume,
                        ob.loss,
                        ob.shared,
                        self.stamps[b],
                    ) =>
                {
                    Some((b, ob))
                }
                _ => Some((i, o)),
            };
        }
        best.map(|(i, _)| i)
    }

    fn apply(&mut self, lane: usize, event: LaneEvent) -> Result<LaneState, LaneError> {
        let from = self.states[lane];
        if !ln_allows(from, event) {
            return Err(LaneError { lane, from, event });
        }
        let next = ln_step(from, event);
        self.states[lane] = next;
        Ok(next)
    }

    /// A request enters a free lane.
    pub fn admit(&mut self, lane: usize, exclusive: bool) -> Result<(), LaneError> {
        self.apply(lane, LaneEvent::EvLnAdmit)?;
        self.exclusive[lane] = exclusive;
        self.clock += 1;
        self.stamps[lane] = self.clock;
        Ok(())
    }

    /// The lane's prompt is prefilled: it decodes from now on.
    pub fn prefilled(&mut self, lane: usize) -> Result<(), LaneError> {
        self.apply(lane, LaneEvent::EvLnPrefilled).map(|_| ())
    }

    /// The lane's request ended (stop, length or cancel).
    pub fn finish(&mut self, lane: usize) -> Result<(), LaneError> {
        self.apply(lane, LaneEvent::EvLnFinish).map(|_| ())
    }

    /// The lane's request failed.
    pub fn fail(&mut self, lane: usize) -> Result<(), LaneError> {
        self.apply(lane, LaneEvent::EvLnFail).map(|_| ())
    }

    /// The lane's report is delivered: the lane is free again.
    pub fn release(&mut self, lane: usize) -> Result<(), LaneError> {
        self.apply(lane, LaneEvent::EvLnRelease)?;
        self.exclusive[lane] = false;
        Ok(())
    }

    /// Lanes that take part in the next decode step, in lane order.
    pub fn batch(&self) -> Vec<usize> {
        (0..self.len())
            .filter(|&i| ln_in_batch(self.states[i]))
            .collect()
    }

    /// Lanes whose request ended and whose report waits to be delivered.
    pub fn drained(&self) -> Vec<usize> {
        (0..self.len())
            .filter(|&i| ln_is_drain(self.states[i]))
            .collect()
    }

    /// Does lane `lane` hold a request?
    pub fn is_busy(&self, lane: usize) -> bool {
        !ln_is_free(self.states[lane])
    }

    /// Short name of a lane's state (`/slots`).
    pub fn state_name(&self, lane: usize) -> &'static str {
        match self.states[lane] {
            LaneState::LnFree => "free",
            LaneState::LnPrefill => "prefill",
            LaneState::LnDecode => "decode",
            LaneState::LnDrain => "done",
        }
    }
}

/// The device plan of `n` lanes of `lane_bytes` each beside `shared` bytes: `Ok(total)` when it
/// fits in `usable` bytes, `Err(shortfall)` otherwise.
pub fn plan_check(n: u64, lane_bytes: u64, shared: u64, usable: u64) -> Result<u64, u64> {
    if ln_plan_fits(n, lane_bytes, shared, usable) {
        Ok(ln_plan_bytes(n, lane_bytes, shared))
    } else {
        Err(ln_plan_shortfall(n, lane_bytes, shared, usable))
    }
}

/// Chunks of `chunk` rows a batched step must cover for slots writing rows up to
/// `cursor + rows` (exclusive) each: at least one.
pub fn batch_chunks(slots: &[(u64, u64)], chunk: u64) -> u64 {
    slots
        .iter()
        .fold(0, |acc, &(c, r)| ln_batch_chunks(acc, c, r, chunk))
}

/// Chunks a batched step covers when its slots need `need` (a lane owns `lane_chunks`).
pub fn step_chunks(need: u64, lane_chunks: u64) -> u64 {
    ln_step_chunks(need, lane_chunks)
}

/// Every chunk count a batched step can cover in a lane of `lane_chunks` (one recipe each).
pub fn step_chunk_counts(lane_chunks: u64) -> Vec<u64> {
    let mut v: Vec<u64> = (1..=lane_chunks)
        .map(|need| ln_step_chunks(need, lane_chunks))
        .collect();
    v.dedup();
    v
}

/// Lanes of `lane_bytes` that fit beside `shared` bytes in `usable` bytes.
pub fn lanes_fit(lane_bytes: u64, shared: u64, usable: u64) -> u64 {
    ln_lanes_fit(lane_bytes, shared, usable)
}

/// Kept prefix positions per lane when `granted` fit for all `n` lanes.
pub fn prefix_per_lane(granted: u64, n: u64) -> u64 {
    ln_prefix_per_lane(granted, n)
}

/// Has a lane's request ended (end-of-sequence token, cancelled, or no budget left)?
pub fn run_done(eos: bool, cancelled: bool, remaining: u64) -> bool {
    ln_run_done(eos, cancelled, remaining)
}

/// Does a running lane session admit a waiting request while the oldest waiter for the card is
/// `due`?
pub fn session_admits(due: bool) -> bool {
    ln_session_admits(due)
}

/// Does a lane session with its lanes `idle` hand the card's turn back, the oldest waiter being
/// `due`?
pub fn session_yields(idle: bool, due: bool) -> bool {
    ln_session_yields(idle, due)
}

/// Is a waiter for the card due (no request passes it over any more)? `waited` / `max_wait` in
/// ms, `skips` / `max_skips` = times passed over.
pub fn waiter_due(waited: u64, max_wait: u64, skips: u64, max_skips: u64) -> bool {
    turn_due(waited, max_wait, skips, max_skips)
}

/// A request arrives for a model whose lane session is `active` or not: does it start one?
pub fn session_starts(active: bool) -> bool {
    ln_session_starts(active)
}

/// Does a lane session end, with no lane busy (`idle`) and a request `waiting` or not?
pub fn session_ends(idle: bool, waiting: bool) -> bool {
    ln_session_ends(idle, waiting)
}

/// Byte offset of slot `j`'s cursor in the host staging buffer of a step of `b` slots (after the
/// `b` token ids).
pub fn stage_cursor(b: u64, j: u64) -> u64 {
    crate::verified::ln_stage_cursor(b, j)
}

/// May a slot (`cursor`, `rows`) run in a step over `nch` chunks of a lane of `lane_chunks`?
pub fn slot_fits(cursor: u64, rows: u64, nch: u64, chunk: u64, lane_chunks: u64) -> bool {
    ln_slot_fits(cursor, rows, nch, chunk, lane_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lane_holds_one_request_from_admission_to_release() {
        let mut t = LaneTable::new(2);
        assert!(t.can_admit(false));
        t.admit(0, false).unwrap();
        assert!(
            t.admit(0, false).is_err(),
            "a busy lane takes no second request"
        );
        assert_eq!(t.busy(), 1);
        assert!(t.batch().is_empty(), "a prefilling lane is not stepped");
        t.prefilled(0).unwrap();
        assert_eq!(t.batch(), vec![0]);
        assert!(t.release(0).is_err(), "a decoding lane is not freed");
        t.finish(0).unwrap();
        assert!(t.batch().is_empty());
        assert_eq!(t.drained(), vec![0]);
        t.release(0).unwrap();
        assert_eq!(t.busy(), 0);
        assert_eq!(t.state_name(0), "free");
    }

    #[test]
    fn admission_follows_the_lane_gate() {
        let mut t = LaneTable::new(2);
        t.admit(0, false).unwrap();
        assert!(t.can_admit(false));
        assert!(!t.can_admit(true), "an image request waits for every lane");
        t.admit(1, false).unwrap();
        assert!(!t.can_admit(false), "all lanes busy");
        for l in 0..2 {
            t.prefilled(l).unwrap();
            t.finish(l).unwrap();
            t.release(l).unwrap();
        }
        t.admit(1, true).unwrap();
        assert!(t.exclusive_active());
        assert!(!t.can_admit(false), "nothing runs beside an image request");
        t.fail(1).unwrap();
        assert!(t.exclusive_active(), "until its report is delivered");
        t.release(1).unwrap();
        assert!(!t.exclusive_active());
        assert!(t.can_admit(true));
    }

    #[test]
    fn routing_weighs_rows_taken_over_against_rows_destroyed() {
        let off = |resume, shared, held: u64| Offer {
            resume,
            shared,
            loss: held - shared,
        };
        let mut t = LaneTable::new(3);
        // lane 1 used first, then lane 0, lane 2 never
        for l in [1, 0] {
            t.admit(l, false).unwrap();
            t.prefilled(l).unwrap();
            t.finish(l).unwrap();
            t.release(l).unwrap();
        }
        assert_eq!(t.route(|_| Offer::default()), Some(2), "the unused lane");
        // the lane holding the conversation
        assert_eq!(
            t.route(|l| if l == 0 {
                off(5120, 5200, 5200)
            } else {
                Offer::default()
            }),
            Some(0)
        );
        // a lane whose held conversation only shares a prefix: taking over 5120 rows would
        // destroy 11000 held ones, so an empty lane is the better place
        assert_eq!(
            t.route(|l| if l == 0 {
                off(5120, 5152, 16152)
            } else {
                Offer::default()
            }),
            Some(2)
        );
        // equal nets: more shared rows (then the lane used longest ago, above)
        assert_eq!(
            t.route(|l| if l == 1 {
                off(0, 70, 70)
            } else {
                Offer::default()
            }),
            Some(1)
        );
        assert_eq!(
            t.route(|l| off(0, u64::from(l == 0) * 70, u64::from(l == 0) * 70)),
            Some(0)
        );
        // a lane that would destroy held rows loses to one that destroys none
        assert_eq!(
            t.route(|l| if l == 0 { off(0, 0, 70) } else { off(0, 0, 0) }),
            Some(2)
        );
        t.admit(2, false).unwrap();
        assert_eq!(
            t.route(|_| off(5, 5, 5)),
            Some(1),
            "used longer ago than lane 0"
        );
        t.admit(1, false).unwrap();
        t.admit(0, false).unwrap();
        assert_eq!(t.route(|_| Offer::default()), None);
    }

    #[test]
    fn plan_and_slot_checks() {
        assert_eq!(plan_check(4, 10, 20, 60), Ok(60));
        assert_eq!(plan_check(4, 10, 20, 59), Err(1));
        assert_eq!(lanes_fit(10, 20, 59), 3);
        assert_eq!(lanes_fit(10, 70, 59), 0);
        assert_eq!(prefix_per_lane(17, 4), 4);
        assert!(
            run_done(false, false, 0) && run_done(true, false, 9) && !run_done(false, false, 1)
        );
        assert!(session_starts(false) && !session_starts(true));
        assert!(
            session_ends(true, false) && !session_ends(true, true) && !session_ends(false, false)
        );
        assert_eq!(batch_chunks(&[], 4096), 0);
        assert_eq!(batch_chunks(&[(0, 1)], 4096), 1);
        assert_eq!(batch_chunks(&[(4095, 1), (10, 1)], 4096), 1);
        assert_eq!(batch_chunks(&[(4095, 2), (10, 1)], 4096), 2);
        assert_eq!(batch_chunks(&[(100, 16), (8191, 1)], 4096), 2);
        assert_eq!(
            (1..=9).map(|n| step_chunks(n, 8)).collect::<Vec<_>>(),
            vec![1, 2, 4, 4, 8, 8, 8, 8, 8]
        );
        assert_eq!(step_chunk_counts(8), vec![1, 2, 4, 8]);
        assert_eq!(step_chunk_counts(6), vec![1, 2, 4, 6]);
        assert_eq!(step_chunk_counts(1), vec![1]);
        assert!(slot_fits(4095, 1, 1, 4096, 4));
        assert!(!slot_fits(4096, 1, 1, 4096, 4));
        assert!(
            !slot_fits(0, 1, 5, 4096, 4),
            "the step reaches past the lane"
        );
    }
}
