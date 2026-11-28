//! Recent chat requests, as the server reports them:
//!
//! - `GET /gaudium/requests` (served only with `--debug-endpoints`): the newest requests, each
//!   with what it is (job, reply id, endpoint, model, streamed or not, arrival) and the newest
//!   `request` event it published;
//! - `POST /v1/streams/lookup`: the streamed replies that are still being written.
//!
//! The table only reports. Nothing in the server reads it to decide anything, so it stays in the
//! Rust shell (AGENTS.md section 3a: telemetry that is only reported). It holds no prompt or reply
//! text: only what the `request` events of `/gaudium/events` already carry.
//!
//! A request is entered when it is accepted (its `queued` event) and counts as running until its
//! handler returns. The [`Tracked`] handle marks it ended when it is dropped, so a request that
//! leaves early (the client went away, a write failed, a panic) is never reported as running.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::jq;

/// Requests the table keeps. When a new one arrives at a full table, the oldest request that has
/// ended makes room; only when every kept request is still running does the oldest one go.
pub(super) const RECENT_REQUESTS: usize = 64;

/// What a request is, fixed when the server accepts it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RequestMeta {
    /// The server's job number (`job` in the `request` events).
    pub job: u64,
    /// The reply id the client is given (`chatcmpl-gaudium-N`, `msg_gaudium_N`, `cmpl-gaudium-N`): the id that
    /// `/v1/chat/completions/control` cancels by and that `/v1/streams/lookup` filters on.
    pub id: String,
    /// The API the request came in on: `chat` (chat completions), `messages`, `completions`
    /// (text completions), `completion` (the native completion shape) or `responses`.
    pub endpoint: &'static str,
    /// The served model id.
    pub model: String,
    /// The reply is streamed (server-sent events).
    pub stream: bool,
    /// Arrival, in whole seconds since the Unix epoch.
    pub created: u64,
}

/// One kept request.
#[derive(Debug)]
struct Entry {
    meta: RequestMeta,
    /// The handler has not returned yet.
    running: bool,
    /// The newest `request` event of this request (a JSON object); `None` before the first.
    last_event: Option<String>,
}

impl Entry {
    /// The request as one JSON object (the shape of both endpoints' list items).
    fn json(&self) -> String {
        let m = &self.meta;
        format!(
            "{{\"job\":{},\"id\":{},\"endpoint\":{},\"model\":{},\"stream\":{},\"created\":{},\
             \"running\":{},\"event\":{}}}",
            m.job,
            jq(&m.id),
            jq(m.endpoint),
            jq(&m.model),
            m.stream,
            m.created,
            self.running,
            self.last_event.as_deref().unwrap_or("null"),
        )
    }
}

/// The bounded table of recent requests (oldest first).
pub(super) struct RecentRequests {
    cap: usize,
    entries: Mutex<VecDeque<Entry>>,
}

impl RecentRequests {
    /// A table that keeps at most `cap` requests (at least one).
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            cap,
            entries: Mutex::new(VecDeque::with_capacity(cap)),
        }
    }

    /// Enters an accepted request. It counts as running until the returned handle is dropped.
    pub fn begin(&self, meta: RequestMeta) -> Tracked<'_> {
        let job = meta.job;
        let mut entries = self.entries.lock().unwrap();
        while entries.len() >= self.cap {
            // the oldest request that has ended goes first; with every kept request still
            // running, the oldest one goes, so the table never grows past its capacity
            let victim = entries.iter().position(|e| !e.running).unwrap_or(0);
            entries.remove(victim);
        }
        entries.push_back(Entry {
            meta,
            running: true,
            last_event: None,
        });
        Tracked { table: self, job }
    }

    /// Keeps `event` (a `request` event, a JSON object) as the newest event of request `job`.
    /// A job the table no longer holds is left out.
    pub fn record(&self, job: u64, event: &str) {
        let mut entries = self.entries.lock().unwrap();
        // the newest requests are at the back, and an event is nearly always of a recent one
        if let Some(e) = entries.iter_mut().rev().find(|e| e.meta.job == job) {
            e.last_event = Some(event.to_string());
        }
    }

    /// Marks request `job` ended (its handler returned).
    fn end(&self, job: u64) {
        let mut entries = self.entries.lock().unwrap();
        if let Some(e) = entries.iter_mut().rev().find(|e| e.meta.job == job) {
            e.running = false;
        }
    }

    /// `GET /gaudium/requests`: every kept request, newest first.
    pub fn list_json(&self) -> String {
        let entries = self.entries.lock().unwrap();
        let items: Vec<String> = entries.iter().rev().map(Entry::json).collect();
        format!(
            "{{\"object\":\"list\",\"capacity\":{},\"data\":[{}]}}",
            self.cap,
            items.join(",")
        )
    }

    /// `POST /v1/streams/lookup`: the streamed replies still running, newest first. With `ids`,
    /// only those whose reply id is one of them.
    pub fn running_streams_json(&self, ids: Option<&[String]>) -> String {
        let entries = self.entries.lock().unwrap();
        let items: Vec<String> = entries
            .iter()
            .rev()
            .filter(|e| e.running && e.meta.stream)
            .filter(|e| ids.is_none_or(|ids| ids.iter().any(|id| *id == e.meta.id)))
            .map(Entry::json)
            .collect();
        format!("[{}]", items.join(","))
    }
}

/// A request the table holds as running; dropping it marks the request ended.
pub(super) struct Tracked<'a> {
    table: &'a RecentRequests,
    job: u64,
}

impl Drop for Tracked<'_> {
    fn drop(&mut self) {
        self.table.end(self.job);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{parse_json, JsonValue};

    fn meta(job: u64, stream: bool) -> RequestMeta {
        RequestMeta {
            job,
            id: format!("chatcmpl-gaudium-{job}"),
            endpoint: "chat",
            model: "m".into(),
            stream,
            created: 1_700_000_000 + job,
        }
    }

    fn list(t: &RecentRequests) -> Vec<JsonValue> {
        let v = parse_json(&t.list_json()).unwrap();
        v.get("data").unwrap().as_array().unwrap().to_vec()
    }

    fn jobs(items: &[JsonValue]) -> Vec<i64> {
        items
            .iter()
            .map(|i| match i.get("job") {
                Some(JsonValue::Int(n)) => *n,
                other => panic!("job is {other:?}"),
            })
            .collect()
    }

    #[test]
    fn a_request_runs_until_its_handle_is_dropped_and_keeps_its_newest_event() {
        let t = RecentRequests::new(4);
        let h = t.begin(meta(1, true));
        let items = list(&t);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].get("running"), Some(&JsonValue::Bool(true)));
        assert_eq!(items[0].get("event"), Some(&JsonValue::Null));

        t.record(1, r#"{"type":"request","job":1,"phase":"queued"}"#);
        t.record(1, r#"{"type":"request","job":1,"phase":"decode"}"#);
        let items = list(&t);
        let ev = items[0].get("event").unwrap();
        assert_eq!(ev.get("phase").and_then(|p| p.as_str()), Some("decode"));

        drop(h);
        let items = list(&t);
        assert_eq!(items[0].get("running"), Some(&JsonValue::Bool(false)));
        // the last event stays after the request ended
        let ev = items[0].get("event").unwrap();
        assert_eq!(ev.get("phase").and_then(|p| p.as_str()), Some("decode"));
    }

    #[test]
    fn the_list_is_newest_first_with_its_capacity() {
        let t = RecentRequests::new(8);
        for job in 1..=3 {
            drop(t.begin(meta(job, false)));
        }
        let v = parse_json(&t.list_json()).unwrap();
        assert_eq!(v.get("capacity"), Some(&JsonValue::Int(8)));
        assert_eq!(jobs(&list(&t)), vec![3, 2, 1]);
    }

    #[test]
    fn a_full_table_drops_its_oldest_ended_request_before_a_running_one() {
        let t = RecentRequests::new(3);
        let _running = t.begin(meta(1, true));
        drop(t.begin(meta(2, false)));
        drop(t.begin(meta(3, false)));
        // full: the oldest ended request (2) makes room, the running one (1) stays
        drop(t.begin(meta(4, false)));
        assert_eq!(jobs(&list(&t)), vec![4, 3, 1]);
    }

    #[test]
    fn a_table_of_running_requests_stays_bounded() {
        let t = RecentRequests::new(2);
        let a = t.begin(meta(1, true));
        let b = t.begin(meta(2, true));
        let c = t.begin(meta(3, true));
        assert_eq!(jobs(&list(&t)), vec![3, 2]);
        // the dropped request's handle ending later changes nothing
        drop(a);
        assert_eq!(jobs(&list(&t)), vec![3, 2]);
        let running = parse_json(&t.running_streams_json(None)).unwrap();
        assert_eq!(jobs(running.as_array().unwrap()), vec![3, 2]);
        drop((b, c));
    }

    #[test]
    fn events_of_a_job_the_table_dropped_are_left_out() {
        let t = RecentRequests::new(1);
        drop(t.begin(meta(1, false)));
        drop(t.begin(meta(2, false)));
        t.record(1, r#"{"phase":"stop"}"#);
        let items = list(&t);
        assert_eq!(jobs(&items), vec![2]);
        assert_eq!(items[0].get("event"), Some(&JsonValue::Null));
    }

    #[test]
    fn lookup_lists_only_streams_still_running_and_filters_by_reply_id() {
        let t = RecentRequests::new(8);
        let _streaming = t.begin(meta(1, true));
        let _plain = t.begin(meta(2, false));
        drop(t.begin(meta(3, true)));
        let _other = t.begin(meta(4, true));

        let all = parse_json(&t.running_streams_json(None)).unwrap();
        assert_eq!(jobs(all.as_array().unwrap()), vec![4, 1]);

        // an ended stream (3) is not listed even when its id is asked for
        let ids = vec![
            "chatcmpl-gaudium-1".to_string(),
            "chatcmpl-gaudium-3".into(),
        ];
        let some = parse_json(&t.running_streams_json(Some(ids.as_slice()))).unwrap();
        assert_eq!(jobs(some.as_array().unwrap()), vec![1]);

        let none = parse_json(&t.running_streams_json(Some(&[][..]))).unwrap();
        assert!(none.as_array().unwrap().is_empty());
    }

    #[test]
    fn a_capacity_of_zero_keeps_one_request() {
        let t = RecentRequests::new(0);
        assert_eq!(t.cap, 1);
        drop(t.begin(meta(1, false)));
        drop(t.begin(meta(2, false)));
        assert_eq!(jobs(&list(&t)), vec![2]);
    }
}
