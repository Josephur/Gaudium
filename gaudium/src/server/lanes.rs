//! The server side of `--parallel N`: chat requests for a model served with lanes wait in that
//! model's queue, in arrival order, and one lane session per model (a driver thread holding the
//! card's turn) hands them to the engine's lanes (`GenEngine::serve_lanes`).
//!
//! A session starts when a request arrives and none runs for its model, and ends when no lane is
//! busy and nothing waits; both decisions are made under the queue's lock, so a request either
//! joins the running session or starts the next one, never neither.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::engine::{FeedHead, LaneDone, LaneFeed, LaneJob, LaneOut, LaneView};

#[derive(Default)]
struct ModelLanes {
    /// A session runs for this model (its driver holds, or waits for, the card's turn).
    active: bool,
    queue: VecDeque<LaneJob>,
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

    /// Queues `job` for model `idx`. `true`: no session runs for the model, so the caller starts
    /// one (`super::drive_lanes`), which will take this job.
    pub fn submit(&self, idx: usize, job: LaneJob) -> bool {
        let mut g = self.mu.lock().unwrap();
        let m = &mut g[idx];
        m.queue.push_back(job);
        let start = crate::core::lanes::session_starts(m.active);
        m.active = true;
        start
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
            m.queue.drain(..).collect()
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
            on_lanes: None,
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
    /// Called after the lanes changed, outside the queue's lock: the busy lanes.
    pub on_lanes: Option<Box<dyn FnMut(u64) + 'a>>,
}

impl LaneFeed for HubFeed<'_> {
    fn head(&mut self, idle: bool) -> FeedHead {
        if self.ended {
            return FeedHead::End;
        }
        let mut g = self.hub.mu.lock().unwrap();
        let m = &mut g[self.idx];
        let waiting = !m.queue.is_empty();
        if crate::core::lanes::session_ends(idle, waiting) {
            m.active = false;
            self.ended = true;
            return FeedHead::End;
        }
        match m.queue.front() {
            Some(j) => FeedHead::Job {
                exclusive: j.exclusive(),
            },
            None => FeedHead::Empty,
        }
    }

    fn take(&mut self) -> Option<LaneJob> {
        if self.ended {
            return None;
        }
        self.hub.mu.lock().unwrap()[self.idx].queue.pop_front()
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
        assert!(hub.submit(0, a), "the first request starts a session");
        assert!(!hub.submit(0, b), "the second joins it");
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
        assert!(hub.submit(0, c), "after the end, a new session");
    }

    #[test]
    fn an_empty_image_list_is_not_exclusive_and_waiting_requests_fail_together() {
        let hub = LaneHub::new(2);
        let (a, ra) = job(1, true);
        assert!(!a.exclusive());
        assert!(hub.submit(1, a));
        hub.fail_waiting(1, "no card");
        assert_eq!(hub.feed(1).head(false), FeedHead::Empty);
        match ra.recv().unwrap() {
            LaneOut::Done(d) => assert_eq!(d.report.unwrap_err(), "no card"),
            _ => panic!("expected the end of the request"),
        }
        let (b, _rb) = job(2, false);
        assert!(
            hub.submit(1, b),
            "a failed session lets the next arrival start one"
        );
    }
}
