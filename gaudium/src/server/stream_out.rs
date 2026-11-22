//! Writer of one streamed chat completion: the text of a decode launch goes out as ONE
//! server-sent chunk when that launch is complete, with the request's cumulative `timings`.
//!
//! A client that draws tokens per decode step reads consecutive chunks' `timings.predicted_n`
//! and `timings.launches`, so a chunk must never show part of a launch. Two things tell this
//! writer where a launch ends:
//!
//! - the engine's token callback ([`StreamOut::push_token`]), once per generated token;
//! - the engine's launch feed ([`StreamOut::on_event`], `live_report::LaunchEvent`), once per
//!   consumed launch, carrying the tokens generated so far.
//!
//! The two arrive in either order (a one-token step reports its launch before its token, a
//! speculative step hands over its tokens and then reports the launch), so text is written when
//! the reported launches cover every token seen. An engine without a launch feed gets one chunk
//! per token and no launch counts.

use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::readout::{jq, Timings};
use crate::live_report::{LaunchEvent, LaunchKind};

/// Least time between two prefill progress chunks.
const PROGRESS_EVERY_MS: u128 = 100;

pub(super) struct StreamOut {
    /// `None`: a non-streaming request (nothing is written; counters still run).
    stream: Option<TcpStream>,
    /// `"id":..,"object":"chat.completion.chunk","created":..,"model":..` of every chunk.
    chunk_head: String,
    timings_per_token: bool,
    return_progress: bool,
    n_prompt: u64,
    /// The engine reports its launches (else: one chunk per token).
    feed: bool,
    cancel: Arc<AtomicBool>,
    reasoning: String,
    content: String,
    /// Tokens handed over by the token callback.
    seen: u64,
    /// Tokens covered by the launches reported so far.
    covered: u64,
    last: Option<LaunchEvent>,
    started: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    last_progress: Option<Instant>,
    pub write_failed: bool,
    /// Text chunks written (the role chunk and progress chunks are not counted).
    pub chunks: u64,
}

impl StreamOut {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream: Option<TcpStream>,
        req_id: &str,
        model_id: &str,
        created: u64,
        timings_per_token: bool,
        return_progress: bool,
        n_prompt: u64,
        feed: bool,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        Self {
            stream,
            chunk_head: format!(
                "\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{}",
                jq(req_id),
                jq(model_id)
            ),
            timings_per_token,
            return_progress,
            n_prompt,
            feed,
            cancel,
            reasoning: String::new(),
            content: String::new(),
            seen: 0,
            covered: 0,
            last: None,
            started: Instant::now(),
            first_token_at: None,
            last_token_at: None,
            last_progress: None,
            write_failed: false,
            chunks: 0,
        }
    }

    pub fn chunk_head(&self) -> &str {
        &self.chunk_head
    }

    /// Tokens the engine has handed over so far.
    pub fn seen_tokens(&self) -> u64 {
        self.seen
    }

    /// Server-clock time from this writer's creation (the engine took the request) to the first
    /// token; `None` before it.
    pub fn first_token_ms(&self) -> Option<f64> {
        self.first_token_at
            .map(|t| t.duration_since(self.started).as_secs_f64() * 1e3)
    }

    /// The request's cumulative timings as of the newest complete launch.
    pub fn timings(&self) -> Timings {
        match (&self.last, self.feed) {
            (Some(ev), true) if ev.kind != LaunchKind::Prefill => Timings {
                prompt_n: self.n_prompt,
                prompt_ms: ev.ttft_ms,
                predicted_n: ev.gen_tokens,
                predicted_ms: ev.decode_ms,
                launches: Some(ev.launches),
                draft_n: ev.drafted,
                draft_n_accepted: ev.accepted,
            },
            (_, true) => Timings {
                prompt_n: self.n_prompt,
                ..Timings::default()
            },
            // no launch feed: this writer's own clock around the token callback; the prompt
            // time then includes whatever the engine did before its prefill
            (_, false) => Timings {
                prompt_n: self.n_prompt,
                prompt_ms: self.first_token_ms().unwrap_or(0.0),
                predicted_n: self.seen,
                predicted_ms: match (self.first_token_at, self.last_token_at) {
                    (Some(a), Some(b)) => b.duration_since(a).as_secs_f64() * 1e3,
                    _ => 0.0,
                },
                launches: None,
                draft_n: 0,
                draft_n_accepted: 0,
            },
        }
    }

    /// One generated token: its reasoning / content text (either may be empty, e.g. a special
    /// token or bytes held back by the UTF-8 decoder).
    pub fn push_token(&mut self, reasoning: &str, content: &str) {
        let now = Instant::now();
        self.seen += 1;
        self.first_token_at.get_or_insert(now);
        self.last_token_at = Some(now);
        self.reasoning.push_str(reasoning);
        self.content.push_str(content);
        if !self.feed || self.seen <= self.covered {
            self.flush();
        }
    }

    /// One event of the engine's launch feed.
    pub fn on_event(&mut self, ev: &LaunchEvent) {
        if ev.kind == LaunchKind::Prefill {
            self.progress(ev);
            return;
        }
        self.last = Some(*ev);
        self.covered = ev.gen_tokens;
        if self.seen >= self.covered {
            self.flush();
        }
    }

    fn progress(&mut self, ev: &LaunchEvent) {
        if !self.return_progress || self.stream.is_none() {
            return;
        }
        let now = Instant::now();
        if self
            .last_progress
            .is_some_and(|t| now.duration_since(t).as_millis() < PROGRESS_EVERY_MS)
            && ev.prefill_pos < self.n_prompt
        {
            return;
        }
        self.last_progress = Some(now);
        let chunk = format!(
            "data: {{{},\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":null}}],\
             \"prompt_progress\":{{\"total\":{},\"cache\":0,\"processed\":{},\"time_ms\":{:.1}}}}}\n\n",
            self.chunk_head,
            self.n_prompt,
            ev.prefill_pos.min(self.n_prompt),
            ev.request_ms,
        );
        self.write(&chunk);
    }

    /// Takes the text not written yet (the caller puts it into the final chunk).
    pub fn take_pending(&mut self) -> (String, String) {
        (
            std::mem::take(&mut self.reasoning),
            std::mem::take(&mut self.content),
        )
    }

    fn flush(&mut self) {
        let has_text = !self.reasoning.is_empty() || !self.content.is_empty();
        if self.stream.is_none() || (!has_text && !self.timings_per_token) {
            return;
        }
        let mut delta = String::from("{");
        if !self.reasoning.is_empty() {
            delta.push_str(&format!("\"reasoning_content\":{}", jq(&self.reasoning)));
        }
        if !self.content.is_empty() {
            if delta.len() > 1 {
                delta.push(',');
            }
            delta.push_str(&format!("\"content\":{}", jq(&self.content)));
        }
        delta.push('}');
        self.reasoning.clear();
        self.content.clear();
        let timings = if self.timings_per_token {
            format!(",\"timings\":{}", self.timings().json())
        } else {
            String::new()
        };
        let chunk = format!(
            "data: {{{},\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":null}}]{timings}}}\n\n",
            self.chunk_head
        );
        self.chunks += 1;
        self.write(&chunk);
    }

    /// Writes one already-framed SSE payload; a failed write cancels the request.
    pub fn write(&mut self, payload: &str) {
        if self.write_failed {
            return;
        }
        if let Some(s) = self.stream.as_mut() {
            if super::write_sse_chunk(s, payload.as_bytes()).is_err() {
                self.write_failed = true;
                self.cancel.store(true, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{self, JsonValue};
    use std::io::Read;
    use std::net::TcpListener;

    fn pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (b, _) = l.accept().unwrap();
        (a, b)
    }

    /// Reads everything the writer sent and returns the JSON payload of every `data:` line.
    fn payloads(mut rx: TcpStream) -> Vec<JsonValue> {
        let mut raw = String::new();
        rx.read_to_string(&mut raw).unwrap();
        raw.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|d| json::parse_json(d).expect("valid chunk JSON"))
            .collect()
    }

    fn ev(
        kind: LaunchKind,
        gen_tokens: u64,
        launches: u64,
        drafted: u64,
        accepted: u64,
        ms: f64,
    ) -> LaunchEvent {
        LaunchEvent {
            kind,
            gen_tokens,
            launches,
            drafted,
            accepted,
            launch_emitted: 0,
            launch_drafted: 0,
            launch_accepted: 0,
            decode_ms: ms,
            ttft_ms: 80.0,
            prefill_pos: 50,
            request_ms: 0.0,
        }
    }

    fn writer(tx: TcpStream, feed: bool, progress: bool) -> StreamOut {
        StreamOut::new(
            Some(tx),
            "chatcmpl-1",
            "m",
            7,
            true,
            progress,
            50,
            feed,
            Arc::new(AtomicBool::new(false)),
        )
    }

    fn timing(v: &JsonValue, k: &str) -> Option<u64> {
        v.get("timings")
            .and_then(|t| t.get(k))
            .and_then(JsonValue::as_u64)
    }

    fn content(v: &JsonValue) -> String {
        v.get("choices")
            .and_then(JsonValue::as_array)
            .and_then(|c| c[0].get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_string()
    }

    #[test]
    fn speculative_launches_go_out_as_one_chunk_each() {
        let (tx, rx) = pair();
        let mut w = writer(tx, true, false);
        // the first token: reported, then handed over
        w.on_event(&ev(LaunchKind::First, 1, 0, 0, 0, 0.0));
        w.push_token("", "A");
        // launch 1 commits three tokens, then reports itself
        for t in ["b", "c", "d"] {
            w.push_token("", t);
        }
        w.on_event(&ev(LaunchKind::Step, 4, 1, 7, 2, 30.0));
        // launch 2: one token whose bytes the decoder held back
        w.push_token("", "");
        w.on_event(&ev(LaunchKind::Step, 5, 2, 14, 2, 60.0));
        // launch 3: two tokens
        w.push_token("", "e");
        w.push_token("", "f");
        w.on_event(&ev(LaunchKind::Step, 7, 3, 21, 3, 90.0));
        assert_eq!(w.chunks, 4);
        assert_eq!(w.take_pending(), (String::new(), String::new()));
        drop(w);

        let c = payloads(rx);
        assert_eq!(c.len(), 4);
        assert_eq!(
            c.iter().map(content).collect::<Vec<_>>(),
            vec!["A", "bcd", "", "ef"]
        );
        let seq: Vec<(u64, u64)> = c
            .iter()
            .map(|v| {
                (
                    timing(v, "predicted_n").unwrap(),
                    timing(v, "launches").unwrap(),
                )
            })
            .collect();
        assert_eq!(seq, vec![(1, 0), (4, 1), (5, 2), (7, 3)]);
        assert_eq!(timing(&c[3], "draft_n"), Some(21));
        assert_eq!(timing(&c[3], "draft_n_accepted"), Some(3));
        assert_eq!(timing(&c[0], "prompt_n"), Some(50));
        // every chunk shows whole launches: tokens per launch = the differences
        let bars: Vec<u64> = seq.windows(2).map(|w| w[1].0 - w[0].0).collect();
        assert_eq!(bars, vec![3, 1, 2]);
    }

    #[test]
    fn one_token_steps_report_the_launch_before_the_token() {
        let (tx, rx) = pair();
        let mut w = writer(tx, true, false);
        w.on_event(&ev(LaunchKind::First, 1, 0, 0, 0, 0.0));
        w.push_token("think", "");
        for (i, t) in ["x", "y"].iter().enumerate() {
            w.on_event(&ev(
                LaunchKind::Step,
                2 + i as u64,
                1 + i as u64,
                0,
                0,
                25.0 * (i + 1) as f64,
            ));
            w.push_token("", t);
        }
        drop(w);
        let c = payloads(rx);
        let seq: Vec<(u64, u64)> = c
            .iter()
            .map(|v| {
                (
                    timing(v, "predicted_n").unwrap(),
                    timing(v, "launches").unwrap(),
                )
            })
            .collect();
        assert_eq!(seq, vec![(1, 0), (2, 1), (3, 2)]);
        let d0 = c[0].get("choices").and_then(JsonValue::as_array).unwrap()[0]
            .get("delta")
            .unwrap();
        assert_eq!(
            d0.get("reasoning_content").and_then(JsonValue::as_str),
            Some("think")
        );
        assert_eq!(content(&c[1]), "x");
    }

    #[test]
    fn without_a_launch_feed_every_token_is_a_chunk_and_no_launch_count_is_claimed() {
        let (tx, rx) = pair();
        let mut w = writer(tx, false, false);
        w.push_token("", "a");
        w.push_token("", "b");
        drop(w);
        let c = payloads(rx);
        assert_eq!(c.len(), 2);
        assert_eq!(timing(&c[1], "predicted_n"), Some(2));
        assert!(c[1].get("timings").unwrap().get("launches").is_none());
        assert!(c[1].get("timings").unwrap().get("draft_n").is_none());
    }

    #[test]
    fn prefill_progress_is_sent_only_when_asked_for() {
        let (tx, rx) = pair();
        let mut w = writer(tx, true, true);
        let mut p = ev(LaunchKind::Prefill, 0, 0, 0, 0, 0.0);
        p.prefill_pos = 25;
        p.request_ms = 12.0;
        w.on_event(&p);
        p.prefill_pos = 50; // the last tile is always reported
        w.on_event(&p);
        assert_eq!(w.timings().predicted_n, 0, "progress is not a launch");
        drop(w);
        let c = payloads(rx);
        assert_eq!(c.len(), 2);
        let pr = c[1].get("prompt_progress").unwrap();
        assert_eq!(pr.get("total").and_then(JsonValue::as_u64), Some(50));
        assert_eq!(pr.get("processed").and_then(JsonValue::as_u64), Some(50));

        let (tx, rx) = pair();
        let mut w = writer(tx, true, false);
        w.on_event(&p);
        drop(w);
        assert!(payloads(rx).is_empty());
    }
}
