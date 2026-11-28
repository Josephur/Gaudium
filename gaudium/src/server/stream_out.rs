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

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use super::anthropic::Builder;
use super::conn::Conn;
use super::outparse::Delta;
use super::readout::{jq, Timings};
use crate::live_report::{LaunchEvent, LaunchKind};

/// Least time between two prefill progress chunks.
const PROGRESS_EVERY_MS: u128 = 100;

/// The connection of one streamed reply, shared by everything that writes to it: the request's
/// own writer, the engine's launch feed and the keep-alive. Each write takes the lock, so a
/// keep-alive line never lands inside another chunk, and the time of the last write tells the
/// keep-alive when the stream has been quiet for a whole heartbeat.
#[derive(Clone)]
pub(super) struct SseSink {
    inner: Arc<(Mutex<SinkState>, Condvar)>,
}

struct SinkState {
    conn: Conn,
    last_write: Instant,
    /// A write failed (the client went away): every later write fails at once.
    failed: bool,
    /// The stream ended, or its writer is gone: the keep-alive writes nothing more.
    done: bool,
}

impl SseSink {
    /// A sink on a connection whose response head is already written.
    pub fn new(conn: Conn) -> Self {
        Self {
            inner: Arc::new((
                Mutex::new(SinkState {
                    conn,
                    last_write: Instant::now(),
                    failed: false,
                    done: false,
                }),
                Condvar::new(),
            )),
        }
    }

    fn state(&self) -> MutexGuard<'_, SinkState> {
        self.inner.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes one already-framed SSE payload as one chunk.
    pub fn write(&self, payload: &[u8]) -> io::Result<()> {
        let mut s = self.state();
        if s.failed || s.done {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        match super::write_sse_chunk(&mut s.conn, payload) {
            Ok(()) => {
                s.last_write = Instant::now();
                Ok(())
            }
            Err(e) => {
                s.failed = true;
                self.inner.1.notify_all();
                Err(e)
            }
        }
    }

    /// A write failed: the client went away.
    pub fn failed(&self) -> bool {
        self.state().failed
    }

    /// Ends the stream: the terminating chunk, unless a write failed. Nothing is written after
    /// it, by the keep-alive or anyone else.
    pub fn end(&self) {
        let mut s = self.state();
        if !s.failed && !s.done {
            super::write_sse_end(&mut s.conn);
        }
        s.done = true;
        self.inner.1.notify_all();
    }

    /// Starts writing `line` whenever `every` passes without a write, until the stream ends or
    /// the returned guard is dropped. A keep-alive write that fails sets `cancel` (the client
    /// went away). `None`: the keep-alive thread could not start (the stream still works).
    pub fn keep_alive(
        &self,
        line: String,
        every: Duration,
        cancel: Arc<AtomicBool>,
    ) -> Option<KeepAlive> {
        let sink = self.clone();
        thread::Builder::new()
            .name("gaudium-sse-keepalive".into())
            .spawn(move || sink.keep_alive_loop(line.as_bytes(), every, &cancel))
            .ok()
            .map(|_| KeepAlive(self.clone()))
    }

    fn keep_alive_loop(&self, line: &[u8], every: Duration, cancel: &AtomicBool) {
        let every = every.max(Duration::from_millis(1));
        let mut s = self.state();
        loop {
            if s.done || s.failed {
                return;
            }
            let quiet = s.last_write.elapsed();
            if quiet < every {
                s = self
                    .inner
                    .1
                    .wait_timeout(s, every - quiet)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
                continue;
            }
            // under the lock: no other chunk is half written
            if super::write_sse_chunk(&mut s.conn, line).is_err() {
                s.failed = true;
                cancel.store(true, Ordering::Relaxed);
                return;
            }
            s.last_write = Instant::now();
        }
    }
}

/// The keep-alive of a streamed reply; dropping it stops it (the stream itself stays open).
pub(super) struct KeepAlive(SseSink);

impl Drop for KeepAlive {
    fn drop(&mut self) {
        let mut s = self.0.state();
        s.done = true;
        self.0.inner.1.notify_all();
    }
}

/// The shape of the text chunks a stream carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkForm {
    /// Chat-completion chunks (`choices[].delta`).
    Chat,
    /// Text-completion chunks (`choices[].text`).
    Text,
    /// The native completion shape (`content`, `stop`).
    Native,
}

pub(super) struct StreamOut {
    /// `None`: a non-streaming request (nothing is written; counters still run).
    stream: Option<SseSink>,
    /// `"id":..,"object":..,"created":..,"model":..` of every chunk (not of native chunks).
    chunk_head: String,
    form: ChunkForm,
    timings_per_token: bool,
    return_progress: bool,
    n_prompt: u64,
    /// The engine reports its launches (else: one chunk per token).
    feed: bool,
    cancel: Arc<AtomicBool>,
    /// Parsed output not written yet.
    pending: Delta,
    /// Block-structured framing (the Messages API) instead of chat-completion chunks.
    pub anth: Option<Builder>,
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
        stream: Option<SseSink>,
        req_id: &str,
        model_id: &str,
        created: u64,
        timings_per_token: bool,
        return_progress: bool,
        n_prompt: u64,
        feed: bool,
        cancel: Arc<AtomicBool>,
        form: ChunkForm,
    ) -> Self {
        let object = match form {
            ChunkForm::Chat => "chat.completion.chunk",
            ChunkForm::Text | ChunkForm::Native => "text_completion",
        };
        Self {
            stream,
            chunk_head: format!(
                "\"id\":{},\"object\":\"{object}\",\"created\":{created},\"model\":{}",
                jq(req_id),
                jq(model_id)
            ),
            form,
            timings_per_token,
            return_progress,
            n_prompt,
            feed,
            cancel,
            pending: Delta::default(),
            anth: None,
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
                prompt_ms: ev.ttft_ms,
                predicted_n: ev.gen_tokens,
                predicted_ms: ev.decode_ms,
                launches: Some(ev.launches),
                draft_n: ev.drafted,
                draft_n_accepted: ev.accepted,
                ..Timings::of_prompt(self.n_prompt, ev.prompt_cached)
            },
            (_, true) => Timings::of_prompt(
                self.n_prompt,
                self.last.as_ref().map_or(0, |ev| ev.prompt_cached),
            ),
            // no launch feed: this writer's own clock around the token callback; the prompt
            // time then includes whatever the engine did before its prefill
            (_, false) => Timings {
                prompt_ms: self.first_token_ms().unwrap_or(0.0),
                predicted_n: self.seen,
                predicted_ms: match (self.first_token_at, self.last_token_at) {
                    (Some(a), Some(b)) => b.duration_since(a).as_secs_f64() * 1e3,
                    _ => 0.0,
                },
                launches: None,
                draft_n: 0,
                draft_n_accepted: 0,
                // without the engine's feed the reused rows are only known at the end
                ..Timings::of_prompt(self.n_prompt, 0)
            },
        }
    }

    /// Parsed output that belongs to no generated token (a forced call's opening).
    pub fn add(&mut self, d: Delta) {
        self.pending.append(d);
    }

    /// One generated token: what the output parser made of it (it may be empty, e.g. a special
    /// token, bytes held back by the UTF-8 decoder or text the parser holds).
    pub fn push_token(&mut self, d: Delta) {
        let now = Instant::now();
        self.seen += 1;
        self.first_token_at.get_or_insert(now);
        self.last_token_at = Some(now);
        self.pending.append(d);
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
        let progress = format!(
            "\"prompt_progress\":{{\"total\":{},\"cache\":{},\"processed\":{},\"time_ms\":{:.1}}}",
            self.n_prompt,
            ev.prompt_cached.min(self.n_prompt),
            ev.prefill_pos.min(self.n_prompt),
            ev.request_ms,
        );
        let chunk = self.text_chunk(None, &format!(",{progress}"));
        self.write(&chunk);
    }

    /// Takes the output not written yet (the caller puts it into the final chunk).
    pub fn take_pending(&mut self) -> Delta {
        std::mem::take(&mut self.pending)
    }

    fn flush(&mut self) {
        // whole calls are for the final message; the stream carries their fragments
        self.pending.tool_calls.clear();
        let has_text = !self.pending.is_empty();
        if self.stream.is_none() {
            return;
        }
        if let Some(b) = self.anth.as_mut() {
            let events = b.feed(&std::mem::take(&mut self.pending));
            if !events.is_empty() {
                self.chunks += 1;
                self.write(&events);
            }
            return;
        }
        if !has_text && !self.timings_per_token {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        let timings = if self.timings_per_token {
            format!(",\"timings\":{}", self.timings().json())
        } else {
            String::new()
        };
        let chunk = self.text_chunk(Some(&pending), &timings);
        self.chunks += 1;
        self.write(&chunk);
    }

    /// One framed chunk of this stream's form that has not ended: `d` its text (`None`: no
    /// text), `extra` further top-level members (each led by a comma).
    pub fn text_chunk(&self, d: Option<&Delta>, extra: &str) -> String {
        match self.form {
            ChunkForm::Chat => format!(
                "data: {{{},\"choices\":[{{\"index\":0,\"delta\":{},\"finish_reason\":null}}]{extra}}}\n\n",
                self.chunk_head,
                d.map_or_else(|| "{}".to_string(), delta_json),
            ),
            ChunkForm::Text => format!(
                "data: {{{},\"choices\":[{{\"index\":0,\"text\":{},\"logprobs\":null,\"finish_reason\":null}}]{extra}}}\n\n",
                self.chunk_head,
                jq(&d.map(plain_text).unwrap_or_default()),
            ),
            ChunkForm::Native => format!(
                "data: {{\"content\":{},\"stop\":false{extra}}}\n\n",
                jq(&d.map(plain_text).unwrap_or_default()),
            ),
        }
    }

    /// Writes one already-framed SSE payload; a failed write cancels the request.
    pub fn write(&mut self, payload: &str) {
        if self.write_failed {
            return;
        }
        if let Some(s) = self.stream.as_ref() {
            if s.write(payload.as_bytes()).is_err() {
                self.write_failed = true;
                self.cancel.store(true, Ordering::Relaxed);
            }
        }
    }
}

/// The text of a completion's output: a completion splits off no reasoning and parses no tool
/// calls, so this is its content (with any reasoning first).
pub(super) fn plain_text(d: &Delta) -> String {
    let mut s = d.reasoning.clone();
    s.push_str(&d.content);
    s
}

/// The `delta` object of a chat-completion chunk: reasoning, content and tool-call fragments
/// (a call's first fragment carries its index, id, type and name).
pub(super) fn delta_json(d: &Delta) -> String {
    let mut parts = Vec::new();
    if !d.reasoning.is_empty() {
        parts.push(format!("\"reasoning_content\":{}", jq(&d.reasoning)));
    }
    if !d.content.is_empty() {
        parts.push(format!("\"content\":{}", jq(&d.content)));
    }
    if !d.tool_deltas.is_empty() {
        let calls: Vec<String> = d
            .tool_deltas
            .iter()
            .map(|t| {
                if t.header {
                    format!(
                        "{{\"index\":{},\"id\":{},\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":{}}}}}",
                        t.index,
                        jq(&t.id),
                        jq(&t.name),
                        jq(&t.arguments)
                    )
                } else {
                    format!(
                        "{{\"index\":{},\"function\":{{\"arguments\":{}}}}}",
                        t.index,
                        jq(&t.arguments)
                    )
                }
            })
            .collect();
        parts.push(format!("\"tool_calls\":[{}]", calls.join(",")));
    }
    format!("{{{}}}", parts.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{self, JsonValue};
    use std::io::Read;
    use std::net::{TcpListener, TcpStream};

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
            prompt_cached: 0,
            request_ms: 0.0,
        }
    }

    fn writer(tx: TcpStream, feed: bool, progress: bool) -> StreamOut {
        StreamOut::new(
            Some(SseSink::new(tx.into())),
            "chatcmpl-1",
            "m",
            7,
            true,
            progress,
            50,
            feed,
            Arc::new(AtomicBool::new(false)),
            ChunkForm::Chat,
        )
    }

    fn text(reasoning: &str, content: &str) -> Delta {
        Delta {
            reasoning: reasoning.into(),
            content: content.into(),
            ..Delta::default()
        }
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
        w.push_token(text("", "A"));
        // launch 1 commits three tokens, then reports itself
        for t in ["b", "c", "d"] {
            w.push_token(text("", t));
        }
        w.on_event(&ev(LaunchKind::Step, 4, 1, 7, 2, 30.0));
        // launch 2: one token whose bytes the decoder held back
        w.push_token(text("", ""));
        w.on_event(&ev(LaunchKind::Step, 5, 2, 14, 2, 60.0));
        // launch 3: two tokens
        w.push_token(text("", "e"));
        w.push_token(text("", "f"));
        w.on_event(&ev(LaunchKind::Step, 7, 3, 21, 3, 90.0));
        assert_eq!(w.chunks, 4);
        assert!(w.take_pending().is_empty());
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
        w.push_token(text("think", ""));
        for (i, t) in ["x", "y"].iter().enumerate() {
            w.on_event(&ev(
                LaunchKind::Step,
                2 + i as u64,
                1 + i as u64,
                0,
                0,
                25.0 * (i + 1) as f64,
            ));
            w.push_token(text("", t));
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
        w.push_token(text("", "a"));
        w.push_token(text("", "b"));
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

    #[test]
    fn tool_call_fragments_go_out_with_their_launch() {
        use super::super::outparse::ToolDelta;
        let (tx, rx) = pair();
        let mut w = writer(tx, true, false);
        // a forced call's opening belongs to no token: it goes out with the first launch
        w.add(Delta {
            tool_deltas: vec![ToolDelta {
                index: 0,
                header: true,
                id: "call_x".into(),
                name: "f".into(),
                arguments: String::new(),
            }],
            ..Delta::default()
        });
        w.on_event(&ev(LaunchKind::First, 1, 0, 0, 0, 0.0));
        let mut d = text("", "");
        d.push_tool_delta(ToolDelta {
            index: 0,
            arguments: "{\"a\":".into(),
            ..ToolDelta::default()
        });
        d.push_tool_delta(ToolDelta {
            index: 0,
            arguments: "1}".into(),
            ..ToolDelta::default()
        });
        w.push_token(d);
        assert!(w.take_pending().is_empty());
        drop(w);
        let c = payloads(rx);
        assert_eq!(c.len(), 1);
        let calls = c[0].get("choices").and_then(JsonValue::as_array).unwrap()[0]
            .get("delta")
            .and_then(|d| d.get("tool_calls"))
            .and_then(JsonValue::as_array)
            .unwrap()
            .to_vec();
        // fragments of one call join: the header carries the arguments that came with it
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].get("index").and_then(JsonValue::as_u64), Some(0));
        assert_eq!(
            calls[0].get("id").and_then(JsonValue::as_str),
            Some("call_x")
        );
        assert_eq!(
            calls[0].get("type").and_then(JsonValue::as_str),
            Some("function")
        );
        let f0 = calls[0].get("function").unwrap();
        assert_eq!(f0.get("name").and_then(JsonValue::as_str), Some("f"));
        assert_eq!(
            f0.get("arguments").and_then(JsonValue::as_str),
            Some("{\"a\":1}")
        );
        // a later fragment of the same call carries only its index and text
        let later = Delta {
            tool_deltas: vec![ToolDelta {
                index: 0,
                arguments: ",".into(),
                ..ToolDelta::default()
            }],
            ..Delta::default()
        };
        let j = json::parse_json(&delta_json(&later)).unwrap();
        let t = &j.get("tool_calls").and_then(JsonValue::as_array).unwrap()[0];
        assert!(t.get("id").is_none() && t.get("type").is_none());
        assert_eq!(
            t.get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(JsonValue::as_str),
            Some(",")
        );
    }

    #[test]
    fn keep_alive_goes_out_only_when_the_stream_is_quiet_and_never_after_its_end() {
        let (tx, mut rx) = pair();
        rx.set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let sink = SseSink::new(tx.into());
        let cancel = Arc::new(AtomicBool::new(false));
        let every = Duration::from_millis(60);
        let guard = sink
            .keep_alive(": ka\n\n".into(), every, Arc::clone(&cancel))
            .expect("the keep-alive thread starts");
        // writes closer together than the heartbeat: no keep-alive between them
        for i in 0..10 {
            sink.write(format!("data: {i}\n\n").as_bytes()).unwrap();
            thread::sleep(Duration::from_millis(15));
        }
        // a quiet stream gets one keep-alive per heartbeat
        thread::sleep(every * 4);
        sink.end();
        // nothing is written after the end, by the keep-alive or by the writer
        thread::sleep(every * 2);
        assert!(sink.write(b"data: late\n\n").is_err());
        drop(guard);
        drop(sink);
        let mut raw = String::new();
        rx.read_to_string(&mut raw).unwrap();
        let first_ka = raw.find(": ka").expect("a keep-alive");
        let last_data = raw.rfind("data: 9").unwrap();
        assert!(first_ka > last_data, "{raw:?}");
        let n = raw.matches(": ka").count();
        assert!((2..=5).contains(&n), "{n} keep-alives: {raw:?}");
        assert!(raw.ends_with("0\r\n\r\n"), "{raw:?}");
        assert!(!raw.contains("late"));
        assert!(!cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn a_keep_alive_to_a_client_that_left_cancels_the_request() {
        let (tx, rx) = pair();
        drop(rx);
        let sink = SseSink::new(tx.into());
        let cancel = Arc::new(AtomicBool::new(false));
        let _guard = sink.keep_alive(
            ": ka\n\n".into(),
            Duration::from_millis(10),
            Arc::clone(&cancel),
        );
        let t = Instant::now();
        while !cancel.load(Ordering::Relaxed) && t.elapsed() < Duration::from_secs(5) {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            cancel.load(Ordering::Relaxed),
            "the failed write cancels the request"
        );
        // and the writer's next write fails at once
        assert!(sink.write(b"data: x\n\n").is_err());
    }
}
