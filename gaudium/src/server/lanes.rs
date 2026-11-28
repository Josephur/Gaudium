//! The server side of `--parallel N`: chat requests for a model served with lanes wait in that
//! model's queue, in arrival order, and one lane session per model (a driver thread holding the
//! card's turn) hands them to the engine's lanes (`GenEngine::serve_lanes`).
//!
//! A session starts when a request arrives and none runs for its model, and ends when no lane is
//! busy and nothing waits; both decisions are made under the queue's lock, so a request either
//! joins the running session or starts the next one, never neither.
//!
//! A session's admissions pass over the older waiters of the card's turn queue (requests of other
//! models), as a request of the loaded model does there. Once its next admission would pass over
//! a waiter whose turn is due, the session admits nothing more, lets its lanes drain, hands the
//! card's turn back and waits for its next one (queued at its oldest request's arrival), its
//! requests staying queued. What a session does next is the verified `ln_session_next`.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use crate::engine::{FeedHead, LaneDone, LaneFeed, LaneJob, LaneOut, LaneView};
use crate::verified::SessionMove;

#[derive(Default)]
struct ModelLanes {
    /// A session runs for this model (its driver holds, or waits for, the card's turn).
    active: bool,
    /// Waiting requests with their arrival.
    queue: VecDeque<(Instant, LaneJob)>,
    /// The lanes as the running session last reported them (empty: no session ran yet).
    view: Vec<LaneView>,
}

/// The lane queues of every served model, by model index.
pub(super) struct LaneHub {
    mu: Mutex<Vec<ModelLanes>>,
}

impl LaneHub {
    pub fn new(models: usize) -> Self {
        Self {
            mu: Mutex::new((0..models).map(|_| ModelLanes::default()).collect()),
        }
    }

    /// Queues every job of `jobs` for model `idx`, one after the other with nothing between them
    /// (the generations of one request: the first prefills the prompt, the ones that find a lane
    /// beside it copy it from there). `true`: no session runs for the model, so the caller starts
    /// one (`super::drive_lanes`), which will take these jobs.
    pub fn submit_all(&self, idx: usize, jobs: Vec<LaneJob>) -> bool {
        let mut g = self.mu.lock().unwrap();
        let m = &mut g[idx];
        let now = Instant::now();
        m.queue.extend(jobs.into_iter().map(|j| (now, j)));
        let start = crate::core::lanes::session_starts(m.active);
        m.active = true;
        start
    }

    /// Arrival of model `idx`'s oldest waiting request.
    pub fn oldest(&self, idx: usize) -> Option<Instant> {
        self.mu.lock().unwrap()[idx].queue.front().map(|(t, _)| *t)
    }

    /// The lanes of model `idx` as its session last reported them.
    pub fn view(&self, idx: usize) -> Vec<LaneView> {
        self.mu.lock().unwrap()[idx].view.clone()
    }

    /// Ends a session of model `idx` that could not start or did not end through its feed: every
    /// waiting request gets `err`, and the next arrival starts a new session.
    pub fn fail_waiting(&self, idx: usize, err: &str) {
        let jobs: Vec<LaneJob> = {
            let mut g = self.mu.lock().unwrap();
            let m = &mut g[idx];
            m.active = false;
            m.view.iter_mut().for_each(|v| *v = LaneView::free());
            m.queue.drain(..).map(|(_, j)| j).collect()
        };
        for mut j in jobs {
            (j.out)(LaneOut::Done(Box::new(LaneDone {
                report: Err(err.to_string()),
                vision: None,
                prefix: None,
                host_tier: None,
            })));
        }
    }

    /// The feed of a session of model `idx`.
    pub fn feed(&self, idx: usize) -> HubFeed<'_> {
        HubFeed {
            hub: self,
            idx,
            ended: false,
            yielded: false,
            on_lanes: None,
            blocks: None,
            on_admit: None,
        }
    }
}

impl LaneView {
    fn free() -> Self {
        Self {
            state: "free",
            ..Self::default()
        }
    }
}

/// What a lane session reads its requests from.
pub(super) struct HubFeed<'a> {
    hub: &'a LaneHub,
    idx: usize,
    /// The feed ended the session (`FeedHead::End`): from then on it hands out nothing.
    pub ended: bool,
    /// The feed stopped the session to hand the card's turn back (`FeedHead::End` with requests
    /// still queued): the session's driver waits for its next turn and serves them then.
    pub yielded: bool,
    /// Called after the lanes changed, outside the queue's lock: the busy lanes.
    pub on_lanes: Option<Box<dyn FnMut(u64) + 'a>>,
    /// Would admitting a request that arrived then pass over a waiter for the card whose turn is
    /// due? (None: never.)
    pub blocks: Option<Box<dyn Fn(Instant) -> bool + 'a>>,
    /// Called after a request is admitted, outside the queue's lock, with its arrival: the older
    /// waiters for the card were passed over once more.
    pub on_admit: Option<Box<dyn FnMut(Instant) + 'a>>,
}

impl LaneFeed for HubFeed<'_> {
    fn head(&mut self, idle: bool) -> FeedHead {
        if self.ended || self.yielded {
            return FeedHead::End;
        }
        loop {
            // the turn queue's lock is never taken under the hub's: the front request's arrival
            // is read first, then whether admitting it is blocked, then the decision is made on
            // the same front (a request that arrived in between is decided on the next round)
            let front = self.hub.oldest(self.idx);
            let blocked = front.is_some_and(|t| self.blocks.as_ref().is_some_and(|f| f(t)));
            let mut g = self.hub.mu.lock().unwrap();
            let m = &mut g[self.idx];
            let now_front = m.queue.front().map(|(t, _)| *t);
            if now_front != front {
                continue;
            }
            return match crate::core::lanes::session_next(idle, front.is_some(), blocked) {
                SessionMove::SmEnd => {
                    m.active = false;
                    self.ended = true;
                    FeedHead::End
                }
                SessionMove::SmYield => {
                    self.yielded = true;
                    FeedHead::End
                }
                SessionMove::SmWait => FeedHead::Empty,
                SessionMove::SmAdmit => match m.queue.front() {
                    Some((_, j)) => FeedHead::Job {
                        exclusive: j.exclusive(),
                    },
                    None => FeedHead::Empty,
                },
            };
        }
    }

    fn take(&mut self) -> Option<LaneJob> {
        if self.ended || self.yielded {
            return None;
        }
        let (since, job) = self.hub.mu.lock().unwrap()[self.idx].queue.pop_front()?;
        if let Some(f) = self.on_admit.as_mut() {
            f(since);
        }
        Some(job)
    }

    fn lanes(&mut self, view: &[LaneView]) {
        let busy = view.iter().filter(|v| v.busy).count() as u64;
        self.hub.mu.lock().unwrap()[self.idx].view = view.to_vec();
        if let Some(f) = self.on_lanes.as_mut() {
            f(busy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::GenParams;
    use std::sync::atomic::AtomicBool;
    use std::sync::{mpsc, Arc};

    fn job(id: u64, images: bool) -> (LaneJob, mpsc::Receiver<LaneOut>) {
        let (tx, rx) = mpsc::channel();
        let j = LaneJob {
            id,
            prompt: vec![1, 2, 3],
            params: GenParams::default(),
            images: images.then(Vec::new),
            cancel: Arc::new(AtomicBool::new(false)),
            out: Box::new(move |ev| {
                let _ = tx.send(ev);
            }),
        };
        (j, rx)
    }

    #[test]
    fn a_request_joins_the_running_session_or_starts_the_next() {
        let hub = LaneHub::new(1);
        let (a, _ra) = job(1, false);
        let (b, _rb) = job(2, false);
        assert!(hub.submit_all(0, vec![a]), "the first request starts a session");
        assert!(!hub.submit_all(0, vec![b]), "the second joins it");
        let mut feed = hub.feed(0);
        assert_eq!(feed.head(true), FeedHead::Job { exclusive: false });
        assert_eq!(feed.take().map(|j| j.id), Some(1));
        assert_eq!(feed.take().map(|j| j.id), Some(2));
        assert_eq!(
            feed.head(false),
            FeedHead::Empty,
            "busy lanes keep the session"
        );
        assert_eq!(feed.head(true), FeedHead::End);
        assert!(feed.take().is_none());
        assert_eq!(feed.head(true), FeedHead::End, "an ended feed stays ended");
        let (c, _rc) = job(3, false);
        assert!(hub.submit_all(0, vec![c]), "after the end, a new session");
    }

    #[test]
    fn the_jobs_of_one_request_queue_together_in_order() {
        let hub = LaneHub::new(1);
        let (a, _ra) = job(1, false);
        assert!(hub.submit_all(0, vec![a]));
        let (b0, _rb0) = job(20, false);
        let (b1, _rb1) = job(21, false);
        let (b2, _rb2) = job(22, false);
        assert!(!hub.submit_all(0, vec![b0, b1, b2]), "they join the running session");
        let mut feed = hub.feed(0);
        let ids: Vec<u64> = std::iter::from_fn(|| feed.take().map(|j| j.id)).collect();
        assert_eq!(ids, vec![1, 20, 21, 22]);
    }

    #[test]
    fn a_session_hands_the_turn_back_once_a_waiter_is_due_and_keeps_its_requests() {
        use std::cell::{Cell, RefCell};
        let hub = LaneHub::new(1);
        let (a, _ra) = job(1, false);
        let (b, _rb) = job(2, false);
        assert!(hub.submit_all(0, vec![a]));
        let due = Cell::new(false);
        let admitted = RefCell::new(Vec::new());
        let mut feed = hub.feed(0);
        feed.blocks = Some(Box::new(|_| due.get()));
        feed.on_admit = Some(Box::new(|since| admitted.borrow_mut().push(since)));
        assert_eq!(feed.head(true), FeedHead::Job { exclusive: false });
        assert_eq!(feed.take().map(|j| j.id), Some(1));
        assert_eq!(admitted.borrow().len(), 1, "an admission passes the older waiters over");
        assert!(!hub.submit_all(0, vec![b]), "the second request joins the running session");
        due.set(true);
        assert_eq!(
            feed.head(false),
            FeedHead::Empty,
            "a due waiter stops admissions; the busy lane keeps the turn"
        );
        assert_eq!(feed.head(true), FeedHead::End, "drained: the turn goes back");
        assert!(feed.yielded && !feed.ended);
        assert!(feed.take().is_none());
        drop(feed);
        let (c, _rc) = job(3, false);
        assert!(!hub.submit_all(0, vec![c]), "the yielded session still runs: no second one");
        let mut next = hub.feed(0);
        assert_eq!(next.head(true), FeedHead::Job { exclusive: false });
        assert_eq!(next.take().map(|j| j.id), Some(2), "its queue kept the arrival order");
    }

    #[test]
    fn an_empty_image_list_is_not_exclusive_and_waiting_requests_fail_together() {
        let hub = LaneHub::new(2);
        let (a, ra) = job(1, true);
        assert!(!a.exclusive());
        assert!(hub.submit_all(1, vec![a]));
        hub.fail_waiting(1, "no card");
        assert_eq!(hub.feed(1).head(false), FeedHead::Empty);
        match ra.recv().unwrap() {
            LaneOut::Done(d) => assert_eq!(d.report.unwrap_err(), "no card"),
            _ => panic!("expected the end of the request"),
        }
        let (b, _rb) = job(2, false);
        assert!(
            hub.submit_all(1, vec![b]),
            "a failed session lets the next arrival start one"
        );
    }
}
