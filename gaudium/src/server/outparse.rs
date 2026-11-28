//! Streaming output parser: generated text -> reasoning, content and tool calls.
//!
//! 1. Stop strings (`stop`): matched on the raw generated text; text that could still become a
//!    stop string is held back; the match and everything after it are dropped.
//! 2. Reasoning: text between the family's reasoning tags goes to `reasoning`, the rest to
//!    `content`. A request whose prompt ends inside the opening tag starts in reasoning.
//!    Whitespace around the tags is trimmed, as the chat template trims it when the turn is
//!    rendered back.
//! 3. Tool calls (when the request has tools), streamed as fragments: the call's header
//!    `{index, id, name}` as soon as the name is complete, then its `arguments` JSON text piece by
//!    piece. The concatenated fragments equal [`ToolCall::arguments`] of the finished call. Three
//!    forms are read:
//!    - parameter blocks,
//!      `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>`:
//!      string-typed parameters stream character by character (one leading and one trailing
//!      newline are dropped, exactly what the template adds when it renders the call back),
//!      other parameters arrive whole at `</parameter>`, typed by the tool's JSON schema;
//!    - a JSON object, `<tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>`, read
//!      when its closing tag arrives;
//!    - the call form `<open>NAME{key:value,...}<close>` of a model whose
//!      `tokenizer_config.json` carries a `response_template` (unquoted keys, strings between
//!      the template's delimiters, bare numbers / true / false / null, nested `{}` and `[]`),
//!      transcribed to JSON as it arrives.
//!
//!    A block that turns out not to be a call is given back as content. A call cut by the end of
//!    generation is closed and kept (its header was already streamed).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::tool_calls::{hold_back_limit, release_len, CallEvent, CallStream};
use crate::json::{escape_json_into, parse_json, JsonValue};
use crate::verified::MbState;

mod blocks;

const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const P_OPEN: &str = "<parameter=";
const P_CLOSE: &str = "</parameter>";

/// How a model family writes its tool calls between the tool tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallStyle {
    /// `<function=NAME><parameter=P>VALUE</parameter>...</function>`.
    Params,
    /// `{"name": NAME, "arguments": {...}}`.
    Json,
    /// `NAME{key:value,...}` right after the opening tag.
    Call,
    /// Typed message blocks the model opens itself (`blocks`): reasoning, text and calls each in
    /// a block of their own, a call as `{"name": NAME, "args": {...}}`.
    Blocks,
}

/// How a model family writes its reasoning and its tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputFormat {
    pub think_open: String,
    pub think_close: String,
    pub tool_open: String,
    pub tool_close: String,
    pub style: CallStyle,
    /// Call form: the string delimiters.
    pub str_open: String,
    pub str_close: String,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self {
            think_open: "<think>".into(),
            think_close: "</think>".into(),
            tool_open: "<tool_call>".into(),
            tool_close: "</tool_call>".into(),
            style: CallStyle::Params,
            str_open: String::new(),
            str_close: String::new(),
        }
    }
}

impl OutputFormat {
    /// The form a model directory names: its `tokenizer_config.json` `response_template`
    /// (reasoning tags; tool calls as `<literal>(?P<name>\w+)` with JSON-like content and string
    /// delimiters), else the default tags, with JSON-object calls when the chat template shows
    /// that form. The second value says what was not understood (the form is then the default).
    pub fn detect(
        tokenizer_config: Option<&JsonValue>,
        chat_template: &str,
    ) -> (Self, Option<String>) {
        if blocks::is_block_template(chat_template) {
            // the template's own block markers say everything (a `response_template` of the
            // tokenizer configuration describes the same blocks)
            return (blocks::block_format(), None);
        }
        let mut f = Self::default();
        if !chat_template.contains(FN_OPEN) && chat_template.contains("\"arguments\"") {
            f.style = CallStyle::Json;
        }
        let Some(rt) = tokenizer_config.and_then(|c| c.get("response_template")) else {
            return (f, None);
        };
        let no = |why: &str| (Self::default(), Some(format!("response_template: {why}")));
        let Some(fields) = rt.get("fields").filter(|x| x.as_object().is_some()) else {
            return no("no fields");
        };
        let text = |o: &JsonValue, k: &str| {
            o.get(k)
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string()
        };
        if let Some(t) = fields.get("thinking") {
            let (open, close) = (text(t, "open"), text(t, "close"));
            if open.is_empty() || close.is_empty() {
                return no("thinking needs a literal open and close");
            }
            f.think_open = open;
            f.think_close = close;
        }
        if let Some(t) = fields.get("tool_calls") {
            const GROUP: &str = "(?P<name>\\w+)";
            let pattern = text(t, "open_pattern");
            let close = text(t, "close");
            let literal = pattern.strip_suffix(GROUP).and_then(regex_literal);
            let Some(literal) = literal.filter(|l| !l.is_empty()) else {
                return no("tool_calls is not the form <literal>NAME{json}<close>");
            };
            if close.is_empty() || text(t, "content") != "json" {
                return no("tool_calls is not the form <literal>NAME{json}<close>");
            }
            f.tool_open = literal;
            f.tool_close = close;
            f.style = CallStyle::Call;
            let delims = t
                .get("content_args")
                .and_then(|a| a.get("string_delims"))
                .and_then(JsonValue::as_array)
                .and_then(|a| a.first())
                .and_then(JsonValue::as_array)
                .filter(|p| p.len() == 2)
                .and_then(|p| p[0].as_str().zip(p[1].as_str()));
            (f.str_open, f.str_close) = match delims {
                Some((a, b)) if !a.is_empty() && !b.is_empty() => (a.to_string(), b.to_string()),
                _ => ("\"".to_string(), "\"".to_string()),
            };
        }
        (f, None)
    }

    /// The opening of a forced call (`tool_choice` required: `name` empty; or a named function):
    /// what the content must start with.
    pub fn forced_prefix(&self, name: &str) -> String {
        let named = !name.is_empty();
        match self.style {
            // a call without a name cannot be opened: its name comes first
            CallStyle::Blocks if !named => String::new(),
            CallStyle::Blocks => blocks::forced_call_prefix(name),
            CallStyle::Call => format!(
                "{}{}",
                self.tool_open,
                if named {
                    format!("{name}{{")
                } else {
                    String::new()
                }
            ),
            CallStyle::Json => format!(
                "{}\n{{\"name\": \"{}",
                self.tool_open,
                if named {
                    format!("{name}\", \"arguments\": ")
                } else {
                    String::new()
                }
            ),
            // unnamed: the opening tag only, so the model writes `<function=NAME>` whole (cut
            // after `<function=` it does not continue with a bare name)
            CallStyle::Params if !named => format!("{}\n", self.tool_open),
            CallStyle::Params => format!("{}\n{FN_OPEN}{name}>\n", self.tool_open),
        }
    }

    /// Whether this form can write the opening of a forced call (`name` empty: `tool_choice`
    /// required, any function).
    pub fn can_force(&self, name: &str) -> bool {
        !self.forced_prefix(name).is_empty()
    }

    /// The tokens that end a reply of this form besides the model's own end-of-sequence ids.
    pub fn stop_markers(&self) -> &'static [&'static str] {
        match self.style {
            CallStyle::Blocks => &blocks::STOP_MARKERS,
            _ => &[],
        }
    }

    /// Whether `text` is part of one of this form's tags (a special token whose text must reach
    /// the parser).
    pub fn is_marker_piece(&self, text: &str) -> bool {
        if self.style == CallStyle::Blocks {
            return blocks::is_marker(text);
        }
        !text.is_empty()
            && [
                &self.think_open,
                &self.think_close,
                &self.tool_open,
                &self.tool_close,
                &self.str_open,
                &self.str_close,
            ]
            .iter()
            .any(|m| m.contains(text))
    }
}

/// A regular expression that is one literal: its text (`None` when it has an operator).
fn regex_literal(re: &str) -> Option<String> {
    let mut o = String::new();
    let mut it = re.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some(e) if !e.is_alphanumeric() => o.push(e),
                _ => return None,
            }
        } else if ".*+?()[]{}^$|".contains(c) {
            return None;
        } else {
            o.push(c);
        }
    }
    Some(o)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON object text.
    pub arguments: String,
}

/// One streamed fragment of a tool call; `header` marks the first fragment of the call (id and
/// name).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolDelta {
    pub index: usize,
    pub header: bool,
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Delta {
    pub reasoning: String,
    pub content: String,
    /// Calls completed in this delta (whole).
    pub tool_calls: Vec<ToolCall>,
    /// Streamed fragments.
    pub tool_deltas: Vec<ToolDelta>,
}

impl Delta {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty()
            && self.content.is_empty()
            && self.tool_calls.is_empty()
            && self.tool_deltas.is_empty()
    }

    pub fn append(&mut self, o: Delta) {
        self.reasoning.push_str(&o.reasoning);
        self.content.push_str(&o.content);
        self.tool_calls.extend(o.tool_calls);
        for x in o.tool_deltas {
            self.push_tool_delta(x);
        }
    }

    /// Adds a fragment, joining it to the previous fragment of the same call.
    pub fn push_tool_delta(&mut self, x: ToolDelta) {
        match self.tool_deltas.last_mut() {
            Some(last) if last.index == x.index && !x.header => {
                last.arguments.push_str(&x.arguments)
            }
            _ => self.tool_deltas.push(x),
        }
    }
}

fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\n' | b'\r' | b'\t')
}

fn ws_prefix_len(s: &str) -> usize {
    s.bytes().take_while(|c| is_ws(*c)).count()
}

fn ws_suffix_start(s: &str) -> usize {
    s.len() - s.bytes().rev().take_while(|c| is_ws(*c)).count()
}

/// Length of the longest suffix of `s` that is a proper prefix of `tag`.
fn partial_suffix(s: &str, tag: &str) -> usize {
    // the verified bound: a proper prefix of the tag, no longer than the text
    let m = hold_back_limit(s.len(), tag.len());
    (1..=m)
        .rev()
        .find(|&k| {
            s.is_char_boundary(s.len() - k)
                && tag.is_char_boundary(k)
                && s.as_bytes()[s.len() - k..] == tag.as_bytes()[..k]
        })
        .unwrap_or(0)
}

fn trim_one_newline(v: &str) -> &str {
    let v = v.strip_prefix('\n').unwrap_or(v);
    v.strip_suffix('\n').unwrap_or(v)
}

fn jesc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    escape_json_into(&mut o, s);
    o
}

fn jstr(s: &str) -> String {
    format!("\"{}\"", jesc(s))
}

fn find_min(hay: &str, needles: &[&str]) -> Option<(usize, usize)> {
    needles
        .iter()
        .enumerate()
        .filter_map(|(i, n)| hay.find(n).map(|p| (p, i)))
        .min()
}

/// `call_<24 letters and digits>`.
fn random_call_id() -> String {
    static STATE: AtomicU64 = AtomicU64::new(0);
    const ALNUM: &[u8; 62] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let next = || {
        // splitmix64 over a counter seeded with the clock
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0x9e37_79b9, |d| d.as_nanos() as u64);
        let _ = STATE.compare_exchange(0, seed | 1, Ordering::Relaxed, Ordering::Relaxed);
        let mut z = STATE.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut s = String::from("call_");
    while s.len() < 5 + 24 {
        let mut r = next();
        for _ in 0..8 {
            if s.len() < 5 + 24 {
                s.push(ALNUM[(r % 62) as usize] as char);
            }
            r /= 62;
        }
    }
    s
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Reasoning,
    Content,
    Tool,
}

/// Where the parameter-block (and JSON-object) reader is inside a tool block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ts {
    Open,
    Name,
    Body,
    PName,
    Value,
    After,
    Json,
}

/// Where the call-form reader is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cs {
    Name,
    Key,
    Value,
    Str,
    Scalar,
    Next,
    Close,
}

pub struct OutputParser {
    f: OutputFormat,
    stops: Vec<String>,
    stop_buf: String,
    matched_stop: String,
    stopped: bool,
    split: bool,
    tools_on: bool,
    tools: Option<JsonValue>,
    state: State,
    /// Text after the stop filter not yet routed (partial tags).
    buf: String,
    /// Trailing whitespace held back.
    ws_hold: String,
    /// The whitespace that stood before the tool block being parsed.
    tool_ws: String,
    at_reasoning_start: bool,
    at_content_start: bool,
    content: String,
    reasoning: String,
    calls: Vec<ToolCall>,
    /// The last call was closed by the end of generation, not by the model ([`Self::repaired`]).
    repaired: bool,
    /// The verified lifecycle of this reply's tool calls (header before fragments, call limit).
    stream: CallStream,
    // the tool call being parsed
    ts: Ts,
    /// Unprocessed text inside the current tool block.
    tb: String,
    /// The whole raw block (given back as content if it is not a call).
    raw: String,
    cur: ToolCall,
    nparams: usize,
    ptype: String,
    vraw: String,
    vemit: usize,
    // the call form: the open containers, whether each already has a member, the separator owed
    cs: Cs,
    cstack: Vec<u8>,
    cfilled: Vec<bool>,
    csep: String,
    /// Call form: the string being read is streamed (else held and cast when it closes).
    str_streamed: bool,
    /// Message blocks: where the verified block reader is, the open block's name and the open
    /// call's text.
    mb: MbState,
    mb_name: String,
    mb_call: String,
}

impl OutputParser {
    /// `reasoning_open`: the prompt ended inside the reasoning opening tag. `split_reasoning`
    /// false leaves reasoning in the content. `tools`: the request's tool list (argument types),
    /// `None` when the request has no tools (tool blocks are then plain content).
    pub fn new(
        reasoning_open: bool,
        split_reasoning: bool,
        tools: Option<JsonValue>,
        stops: Vec<String>,
        format: OutputFormat,
    ) -> Self {
        Self {
            f: format,
            stops,
            stop_buf: String::new(),
            matched_stop: String::new(),
            stopped: false,
            split: split_reasoning,
            tools_on: tools.is_some(),
            tools,
            state: if reasoning_open && split_reasoning {
                State::Reasoning
            } else {
                State::Content
            },
            buf: String::new(),
            ws_hold: String::new(),
            tool_ws: String::new(),
            at_reasoning_start: true,
            at_content_start: split_reasoning,
            content: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
            repaired: false,
            stream: CallStream::new(),
            ts: Ts::Open,
            tb: String::new(),
            raw: String::new(),
            cur: ToolCall::default(),
            nparams: 0,
            ptype: String::new(),
            vraw: String::new(),
            vemit: 0,
            cs: Cs::Name,
            cstack: Vec::new(),
            cfilled: Vec::new(),
            csep: String::new(),
            str_streamed: false,
            mb: Self::block_start(reasoning_open),
            mb_name: String::new(),
            mb_call: String::new(),
        }
    }

    /// `parallel_tool_calls` false: stop after the first finished call.
    pub fn set_max_calls(&mut self, n: usize) {
        self.stream.set_limit(n as u64);
    }

    /// A stop string matched (or the call limit was reached): the caller ends generation.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// The last call was cut by the end of generation and closed by the parser ([`Self::finish`]),
    /// not finished by the model: its arguments may be incomplete.
    pub fn repaired(&self) -> bool {
        self.repaired
    }

    /// The stop string that matched (empty: none did).
    pub fn matched_stop(&self) -> &str {
        &self.matched_stop
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn reasoning(&self) -> &str {
        &self.reasoning
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// The text routed so far leaves the reply inside its reasoning span (the thinking budget
    /// counts the tokens generated there). Text held back as a possible tag is not routed yet.
    pub fn in_reasoning(&self) -> bool {
        match self.f.style {
            CallStyle::Blocks => self.mb == MbState::MsThink,
            _ => self.state == State::Reasoning,
        }
    }

    pub fn feed(&mut self, text: &str) -> Delta {
        let mut d = Delta::default();
        if self.stopped {
            return d;
        }
        let pass;
        let mut hit = false;
        if self.stops.is_empty() {
            pass = text.to_string();
        } else {
            self.stop_buf.push_str(text);
            let best = self
                .stops
                .iter()
                .filter_map(|s| self.stop_buf.find(s.as_str()).map(|p| (p, s)))
                .min_by_key(|(p, _)| *p);
            if let Some((p, s)) = best {
                self.matched_stop = s.clone();
                pass = self.stop_buf[..p].to_string();
                self.stop_buf.clear();
                hit = true;
            } else {
                let hold = self
                    .stops
                    .iter()
                    .map(|s| partial_suffix(&self.stop_buf, s))
                    .max()
                    .unwrap_or(0);
                let cut = release_len(self.stop_buf.len(), hold);
                pass = self.stop_buf[..cut].to_string();
                self.stop_buf.drain(..cut);
            }
        }
        // a stop string ends generation, but the text before it is still routed
        self.route(&pass, &mut d, false);
        self.stopped |= hit;
        d
    }

    /// End of generation: flushes held-back text.
    pub fn finish(&mut self) -> Delta {
        let mut d = Delta::default();
        let rest = if self.stopped {
            String::new()
        } else {
            std::mem::take(&mut self.stop_buf)
        };
        self.stop_buf.clear();
        let was_stopped = std::mem::take(&mut self.stopped);
        self.route(&rest, &mut d, true);
        self.stopped |= was_stopped;
        d
    }

    fn emit_reasoning(&mut self, s: &str, d: &mut Delta) {
        let mut s = s;
        if self.at_reasoning_start {
            s = &s[ws_prefix_len(s)..];
            if s.is_empty() {
                return;
            }
            self.at_reasoning_start = false;
        }
        let joined = std::mem::take(&mut self.ws_hold) + s;
        let e = ws_suffix_start(&joined);
        self.ws_hold = joined[e..].to_string();
        d.reasoning.push_str(&joined[..e]);
        self.reasoning.push_str(&joined[..e]);
    }

    fn emit_content(&mut self, s: &str, d: &mut Delta) {
        let mut s = s;
        if self.at_content_start {
            s = &s[ws_prefix_len(s)..];
            if s.is_empty() {
                return;
            }
            self.at_content_start = false;
        }
        if self.tools_on {
            // hold trailing whitespace: it is dropped if a tool call follows
            let joined = std::mem::take(&mut self.ws_hold) + s;
            let e = ws_suffix_start(&joined);
            self.ws_hold = joined[e..].to_string();
            d.content.push_str(&joined[..e]);
            self.content.push_str(&joined[..e]);
        } else {
            d.content.push_str(s);
            self.content.push_str(s);
        }
    }

    fn route(&mut self, text: &str, d: &mut Delta, last: bool) {
        if self.f.style == CallStyle::Blocks {
            return self.route_blocks(text, d, last);
        }
        self.buf.push_str(text);
        loop {
            if self.stopped {
                // the call limit ended the generation
                self.buf.clear();
                return;
            }
            match self.state {
                State::Reasoning => {
                    if let Some(p) = self.buf.find(self.f.think_close.as_str()) {
                        let head = self.buf[..p].to_string();
                        self.emit_reasoning(&head, d);
                        self.ws_hold.clear();
                        self.buf.drain(..p + self.f.think_close.len());
                        self.state = State::Content;
                        self.at_content_start = true;
                        continue;
                    }
                    let hold = if last {
                        0
                    } else {
                        partial_suffix(&self.buf, &self.f.think_close)
                    };
                    let cut = release_len(self.buf.len(), hold);
                    let head = self.buf[..cut].to_string();
                    self.emit_reasoning(&head, d);
                    self.buf.drain(..cut);
                    if last {
                        self.ws_hold.clear();
                    }
                    return;
                }
                State::Content => {
                    self.route_content(d, last);
                    if self.state == State::Content {
                        return;
                    }
                }
                State::Tool => {
                    let more = std::mem::take(&mut self.buf);
                    self.tb.push_str(&more);
                    self.raw.push_str(&more);
                    while self.state == State::Tool
                        && if self.f.style == CallStyle::Call {
                            self.call_step(d, last)
                        } else {
                            self.tool_step(d, last)
                        }
                    {}
                    if self.state == State::Tool {
                        if !last {
                            return;
                        }
                        if !self.stream.is_streaming() {
                            // no complete name: the text is content
                            self.abandon_call(d);
                            continue;
                        }
                        // cut by the end of generation: keep the streamed call
                        self.close_call(d);
                        self.state = State::Content;
                    }
                }
            }
        }
    }

    fn route_content(&mut self, d: &mut Delta, last: bool) {
        if self.at_content_start && self.split {
            let k = ws_prefix_len(&self.buf);
            self.buf.drain(..k);
            if self.buf.is_empty() {
                return;
            }
            if self.buf.starts_with(self.f.think_open.as_str()) {
                self.buf.drain(..self.f.think_open.len());
                self.state = State::Reasoning;
                self.at_reasoning_start = true;
                return;
            }
            if !last
                && self.buf.len() < self.f.think_open.len()
                && self.f.think_open.starts_with(self.buf.as_str())
            {
                return;
            }
        }
        if !self.tools_on {
            let all = std::mem::take(&mut self.buf);
            self.emit_content(&all, d);
            return;
        }
        if let Some(p) = self.buf.find(self.f.tool_open.as_str()) {
            let head = self.buf[..p].to_string();
            self.emit_content(&head, d);
            // whitespace before a call is dropped; the call form gives it back if the block is not a call
            self.tool_ws = if self.f.style == CallStyle::Call {
                std::mem::take(&mut self.ws_hold)
            } else {
                String::new()
            };
            self.ws_hold.clear();
            self.buf.drain(..p + self.f.tool_open.len());
            self.state = State::Tool;
            self.ts = Ts::Open;
            self.tb.clear();
            self.raw = self.f.tool_open.clone();
            self.cur = ToolCall::default();
            self.stream.apply(CallEvent::CeOpen);
            self.nparams = 0;
            self.cs = Cs::Name;
            self.cstack.clear();
            self.cfilled.clear();
            self.csep.clear();
            self.vraw.clear();
            return;
        }
        let hold = if last {
            0
        } else {
            partial_suffix(&self.buf, &self.f.tool_open)
        };
        let cut = release_len(self.buf.len(), hold);
        let head = self.buf[..cut].to_string();
        self.emit_content(&head, d);
        self.buf.drain(..cut);
        if last && !self.ws_hold.is_empty() {
            let ws = std::mem::take(&mut self.ws_hold);
            d.content.push_str(&ws);
            self.content.push_str(&ws);
        }
    }

    // ------------------------------------------------------------------------- tool calls

    fn send_header(&mut self, d: &mut Delta) {
        if !self.stream.apply(CallEvent::CeHeader) {
            return;
        }
        self.cur.id = random_call_id();
        d.push_tool_delta(ToolDelta {
            index: self.stream.index(),
            header: true,
            id: self.cur.id.clone(),
            name: self.cur.name.clone(),
            arguments: String::new(),
        });
    }

    fn tool_args(&mut self, d: &mut Delta, s: &str) {
        // fragments go out only while the call's header is out
        if s.is_empty() || !self.stream.apply(CallEvent::CeFragment) {
            return;
        }
        self.cur.arguments.push_str(s);
        d.push_tool_delta(ToolDelta {
            index: self.stream.index(),
            arguments: s.to_string(),
            ..ToolDelta::default()
        });
    }

    /// Records the finished call; `true` when it reached the call limit.
    fn push_call(&mut self, d: &mut Delta) -> bool {
        if self.stream.apply(CallEvent::CeFinish) {
            let call = std::mem::take(&mut self.cur);
            self.calls.push(call.clone());
            d.tool_calls.push(call);
        }
        self.stream.stopped()
    }

    fn value_flush(&mut self, d: &mut Delta, closing: bool) {
        if self.ptype == "string" {
            if self.vraw.is_empty() {
                if closing {
                    self.tool_args(d, "\"");
                }
                return;
            }
            let start = usize::from(self.vraw.starts_with('\n'));
            let mut end = self.vraw.len();
            // closing: the template's newline; else: maybe it
            if end > start && self.vraw.ends_with('\n') {
                end -= 1;
            }
            self.vemit = self.vemit.max(start);
            if end > self.vemit {
                let piece = jesc(&self.vraw[self.vemit..end]);
                self.tool_args(d, &piece);
                self.vemit = end;
            }
            if closing {
                self.tool_args(d, "\"");
            }
            return;
        }
        if !closing {
            return;
        }
        let raw = trim_one_newline(&self.vraw).to_string();
        let ty = self.ptype.as_str();
        let text = match parse_json(&raw) {
            Ok(j)
                if matches!(
                    ty,
                    "" | "object" | "array" | "number" | "integer" | "boolean" | "null"
                ) =>
            {
                j.dump()
            }
            Ok(_) => jstr(&raw),
            Err(_) if ty == "boolean" && raw.eq_ignore_ascii_case("true") => "true".into(),
            Err(_) if ty == "boolean" && raw.eq_ignore_ascii_case("false") => "false".into(),
            Err(_) => jstr(&raw),
        };
        self.tool_args(d, &text);
    }

    /// Closes a call cut by the end of generation.
    fn close_call(&mut self, d: &mut Delta) {
        if self.f.style == CallStyle::Call {
            let tail = self.call_close_open();
            self.tool_args(d, &tail);
        } else {
            if self.ts == Ts::Value {
                self.value_flush(d, true);
            }
            if matches!(self.ts, Ts::Body | Ts::Value | Ts::PName) {
                self.tool_args(d, if self.nparams > 0 { "}" } else { "{}" });
            }
        }
        let before = self.calls.len();
        if self.push_call(d) {
            self.stopped = true;
        }
        self.repaired = self.calls.len() > before;
        self.tb.clear();
        self.raw.clear();
        self.ts = Ts::Open;
    }

    /// The block is not a call: its text goes back to the content.
    fn abandon_call(&mut self, d: &mut Delta) {
        let eaten = self.raw.len() - self.raw.len().min(self.tb.len());
        let consumed = std::mem::take(&mut self.tool_ws) + &self.raw[..eaten];
        d.content.push_str(&consumed);
        self.content.push_str(&consumed);
        self.buf = std::mem::take(&mut self.tb) + &self.buf;
        self.raw.clear();
        self.stream.apply(CallEvent::CeAbandon);
        self.state = State::Content;
        self.at_content_start = false;
    }

    /// The call is complete at its closing tag: back to content with the text after it.
    fn end_block(&mut self, d: &mut Delta, rest: String) {
        let limit = self.push_call(d);
        self.raw.clear();
        self.tb.clear();
        self.ts = Ts::Open;
        self.cs = Cs::Name;
        self.state = State::Content;
        self.at_content_start = true;
        self.buf = rest;
        if limit {
            self.stopped = true;
        }
    }

    /// One step of the parameter-block reader; `false`: it needs more text (or left the block).
    fn tool_step(&mut self, d: &mut Delta, last: bool) -> bool {
        let tool_close = self.f.tool_close.clone();
        match self.ts {
            Ts::Open => {
                let k = ws_prefix_len(&self.tb);
                self.tb.drain(..k);
                if self.tb.is_empty() {
                    return false;
                }
                if self.tb.starts_with('{') {
                    self.ts = Ts::Json;
                    return true;
                }
                let n = self.tb.len().min(FN_OPEN.len());
                if self.tb.as_bytes()[..n] != FN_OPEN.as_bytes()[..n] {
                    self.abandon_call(d);
                    return false;
                }
                if n < FN_OPEN.len() {
                    return false;
                }
                self.tb.drain(..FN_OPEN.len());
                self.ts = Ts::Name;
                true
            }
            Ts::Json => {
                // the whole object, read when its closing tag (or the end of generation) arrives
                let end = match self.tb.find(tool_close.as_str()) {
                    Some(p) => p,
                    None if last => self.tb.len(),
                    None => return false,
                };
                let parsed = parse_json(self.tb[..end].trim()).ok().and_then(|j| {
                    let name = j.get("name")?.as_str()?.trim().to_string();
                    let args = match j.get("arguments").or_else(|| j.get("parameters")) {
                        None | Some(JsonValue::Null) => "{}".to_string(),
                        Some(a @ JsonValue::Object(_)) => a.dump(),
                        // arguments written as a JSON string of an object
                        Some(JsonValue::Str(s)) => match parse_json(s) {
                            Ok(a @ JsonValue::Object(_)) => a.dump(),
                            _ => return None,
                        },
                        Some(_) => return None,
                    };
                    (!name.is_empty()).then_some((name, args))
                });
                let Some((name, args)) = parsed else {
                    self.abandon_call(d);
                    return false;
                };
                self.cur.name = name;
                self.send_header(d);
                self.tool_args(d, &args);
                let rest = self.tb[(end + tool_close.len()).min(self.tb.len())..].to_string();
                self.end_block(d, rest);
                false
            }
            Ts::Name => {
                let Some(p) = self.tb.find('>') else {
                    return false;
                };
                let name = self.tb[..p]
                    .trim_end_matches([' ', '\n', '\r', '\t'])
                    .to_string();
                if name.is_empty() {
                    self.abandon_call(d);
                    return false;
                }
                self.cur.name = name;
                self.tb.drain(..=p);
                self.ts = Ts::Body;
                self.send_header(d);
                true
            }
            Ts::Body => {
                let Some((m, which)) = find_min(&self.tb, &[P_OPEN, FN_CLOSE, &tool_close]) else {
                    let hold = [P_OPEN, FN_CLOSE, tool_close.as_str()]
                        .iter()
                        .map(|t| partial_suffix(&self.tb, t))
                        .max()
                        .unwrap_or(0);
                    let cut = release_len(self.tb.len(), hold);
                    self.tb.drain(..cut);
                    return false;
                };
                if which == 0 {
                    self.tb.drain(..m + P_OPEN.len());
                    self.ts = Ts::PName;
                    return true;
                }
                self.tool_args(d, if self.nparams > 0 { "}" } else { "{}" });
                // a closing tool tag without `</function>` is left for `After`
                self.tb
                    .drain(..if which == 1 { m + FN_CLOSE.len() } else { m });
                self.ts = Ts::After;
                true
            }
            Ts::PName => {
                let Some(p) = self.tb.find('>') else {
                    return false;
                };
                let pname = self.tb[..p].to_string();
                self.tb.drain(..=p);
                self.ptype = self.param_type(&self.cur.name, &pname);
                self.vraw.clear();
                self.vemit = 0;
                let lead = format!(
                    "{}{}:{}",
                    if self.nparams > 0 { "," } else { "{" },
                    jstr(&pname),
                    if self.ptype == "string" { "\"" } else { "" }
                );
                self.tool_args(d, &lead);
                self.nparams += 1;
                self.ts = Ts::Value;
                true
            }
            Ts::Value => {
                // a value ends at `</parameter>` (or, missing that, at `</function>` or the tool close)
                if let Some((t, which)) = find_min(&self.tb, &[P_CLOSE, FN_CLOSE, &tool_close]) {
                    let head = self.tb[..t].to_string();
                    self.vraw.push_str(&head);
                    self.value_flush(d, true);
                    self.tb
                        .drain(..if which == 0 { t + P_CLOSE.len() } else { t });
                    self.ts = Ts::Body;
                    return true;
                }
                let hold = [P_CLOSE, FN_CLOSE, tool_close.as_str()]
                    .iter()
                    .map(|t| partial_suffix(&self.tb, t))
                    .max()
                    .unwrap_or(0);
                let cut = release_len(self.tb.len(), hold);
                let head = self.tb[..cut].to_string();
                self.vraw.push_str(&head);
                self.tb.drain(..cut);
                self.value_flush(d, false);
                false
            }
            Ts::After => {
                // after `</function>`: another `<function=` in the same block is a further call;
                // else everything up to the closing tag is ignored
                let p = self.tb.find(tool_close.as_str());
                let q = self.tb.find(FN_OPEN);
                if let Some(q) = q.filter(|q| p.is_none_or(|p| *q < p)) {
                    if self.push_call(d) {
                        self.stopped = true;
                        self.tb.clear();
                        self.raw.clear();
                        self.state = State::Content;
                        return false;
                    }
                    self.stream.apply(CallEvent::CeOpen);
                    self.nparams = 0;
                    self.tb.drain(..q + FN_OPEN.len());
                    self.ts = Ts::Name;
                    return true;
                }
                let Some(p) = p else {
                    let hold = partial_suffix(&self.tb, &tool_close)
                        .max(partial_suffix(&self.tb, FN_OPEN));
                    let cut = release_len(self.tb.len(), hold);
                    self.tb.drain(..cut);
                    return false;
                };
                let rest = self.tb[p + tool_close.len()..].to_string();
                self.end_block(d, rest);
                false
            }
        }
    }

    // ---------------------------------------------------------------------- the call form
    // NAME{key:value,...}: the arguments are transcribed to JSON as they arrive. `cstack` holds
    // the open containers; a member's comma is written when the member starts (so a trailing
    // comma in the model's text leaves none in the JSON).

    fn call_begin_value(&mut self) {
        if let Some(filled) = self.cfilled.last_mut() {
            if *filled {
                self.csep = ",".into();
            }
            *filled = true;
        }
    }

    /// A bare value ended: its JSON text (JSON scalars as they are, anything else a string).
    fn call_scalar_text(&mut self) -> String {
        let v = std::mem::take(&mut self.vraw);
        let v = v.trim_matches([' ', '\n', '\r', '\t']);
        let mut json = matches!(
            parse_json(v),
            Ok(JsonValue::Int(_) | JsonValue::Float(_) | JsonValue::Bool(_) | JsonValue::Null)
        );
        // the tool's schema says string: 7 stays "7"
        if self.cstack.len() == 1 && self.ptype == "string" && !v.is_empty() {
            json = false;
        }
        if json {
            v.to_string()
        } else {
            jstr(v)
        }
    }

    /// The JSON text that closes what a cut call left open.
    fn call_close_open(&mut self) -> String {
        let mut o = String::new();
        match self.cs {
            Cs::Str if self.str_streamed => {
                o.push_str(&jesc(&self.tb));
                o.push('"');
            }
            Cs::Str => {
                // a held (typed) string: as text
                let held = std::mem::take(&mut self.vraw) + &self.tb;
                o.push_str(&jstr(&held));
            }
            Cs::Scalar => {
                let more = self.tb.clone();
                self.vraw.push_str(&more);
                o.push_str(&self.call_scalar_text());
            }
            // a key without its value
            Cs::Value if self.cstack.last() == Some(&b'{') => o.push_str("null"),
            _ => {}
        }
        for c in self.cstack.iter().rev() {
            o.push(if *c == b'{' { '}' } else { ']' });
        }
        if self.cs == Cs::Name {
            o = "{}".into();
        }
        self.cstack.clear();
        self.cfilled.clear();
        self.csep.clear();
        self.cs = Cs::Name;
        o
    }

    fn call_pop(&mut self, d: &mut Delta) {
        let close = if self.cstack.pop() == Some(b'{') {
            "}"
        } else {
            "]"
        };
        self.tool_args(d, close);
        self.cfilled.pop();
        self.csep.clear();
        self.cs = if self.cstack.is_empty() {
            Cs::Close
        } else {
            Cs::Next
        };
    }

    fn call_skip_ws(&mut self) {
        let k = ws_prefix_len(&self.tb);
        self.tb.drain(..k);
    }

    /// One step of the call-form reader; `false`: it needs more text (or left the block).
    fn call_step(&mut self, d: &mut Delta, last: bool) -> bool {
        let so = self.f.str_open.clone();
        let sc = self.f.str_close.clone();
        // the text may still become a string delimiter
        let maybe_delim = |tb: &str| !last && tb.len() < so.len() && so.starts_with(tb);
        match self.cs {
            Cs::Name => {
                let k = self
                    .tb
                    .bytes()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == b'_')
                    .count();
                if k == self.tb.len() {
                    if last || k > 256 {
                        self.abandon_call(d);
                    }
                    return false;
                }
                if k == 0 || self.tb.as_bytes()[k] != b'{' {
                    self.abandon_call(d);
                    return false;
                }
                self.cur.name = self.tb[..k].to_string();
                self.tb.drain(..=k);
                self.send_header(d);
                self.tool_args(d, "{");
                self.cstack = vec![b'{'];
                self.cfilled = vec![false];
                self.csep.clear();
                self.cs = Cs::Key;
                true
            }
            Cs::Key => {
                // a key (bare, or between the string delimiters), or the end of the object
                self.call_skip_ws();
                let Some(&c0) = self.tb.as_bytes().first() else {
                    return false;
                };
                if c0 == b'}' {
                    self.tb.drain(..1);
                    self.call_pop(d);
                    return true;
                }
                if c0 == b',' {
                    self.tb.drain(..1);
                    return true;
                }
                let (key, used) = if self.tb.starts_with(so.as_str()) {
                    let Some(e) = self.tb[so.len()..].find(sc.as_str()).map(|e| e + so.len())
                    else {
                        return false;
                    };
                    let after = e + sc.len();
                    let c = after + ws_prefix_len(&self.tb[after..]);
                    if c == self.tb.len() {
                        return false;
                    }
                    if self.tb.as_bytes()[c] != b':' {
                        // not a key: dropped
                        self.tb.drain(..after);
                        return true;
                    }
                    (self.tb[so.len()..e].to_string(), c + 1)
                } else {
                    if maybe_delim(&self.tb) {
                        return false;
                    }
                    let Some(c) = self.tb.find([':', '}']) else {
                        return false;
                    };
                    if self.tb.as_bytes()[c] == b'}' {
                        // text without a colon before the end: dropped
                        self.tb.drain(..c);
                        return true;
                    }
                    (
                        self.tb[..c]
                            .trim_end_matches([' ', '\n', '\r', '\t'])
                            .to_string(),
                        c + 1,
                    )
                };
                self.tb.drain(..used);
                self.call_begin_value();
                let lead = format!("{}{}:", std::mem::take(&mut self.csep), jstr(&key));
                self.tool_args(d, &lead);
                self.ptype = if self.cstack.len() == 1 {
                    self.param_type(&self.cur.name, &key)
                } else {
                    String::new()
                };
                self.cs = Cs::Value;
                true
            }
            Cs::Value => {
                // a value starts: a string, an object, an array, or a bare scalar
                self.call_skip_ws();
                let Some(&c0) = self.tb.as_bytes().first() else {
                    return false;
                };
                let in_array = self.cstack.last() == Some(&b'[');
                if in_array && c0 == b']' {
                    self.tb.drain(..1);
                    self.call_pop(d);
                    return true;
                }
                if in_array && c0 == b',' {
                    self.tb.drain(..1);
                    return true;
                }
                if maybe_delim(&self.tb) {
                    return false;
                }
                if in_array {
                    self.call_begin_value();
                }
                let sep = std::mem::take(&mut self.csep);
                if self.tb.starts_with(so.as_str()) {
                    self.tb.drain(..so.len());
                    self.vraw.clear();
                    // a top-level string for a parameter the tool's schema types otherwise is held
                    // and cast when it closes
                    let hold =
                        self.cstack.len() == 1 && !self.ptype.is_empty() && self.ptype != "string";
                    self.str_streamed = !hold;
                    let lead = sep + if hold { "" } else { "\"" };
                    self.tool_args(d, &lead);
                    self.cs = Cs::Str;
                    return true;
                }
                if c0 == b'{' || c0 == b'[' {
                    let lead = format!("{sep}{}", c0 as char);
                    self.tool_args(d, &lead);
                    self.cstack.push(c0);
                    self.cfilled.push(false);
                    self.cs = if c0 == b'{' { Cs::Key } else { Cs::Value };
                    self.tb.drain(..1);
                    return true;
                }
                if c0 == b'}' {
                    // "key:}" : no value
                    let lead = sep + "null";
                    self.tool_args(d, &lead);
                    self.cs = Cs::Next;
                    return true;
                }
                self.tool_args(d, &sep);
                self.vraw.clear();
                self.cs = Cs::Scalar;
                true
            }
            Cs::Str => {
                let e = self.tb.find(sc.as_str());
                let n =
                    e.unwrap_or_else(|| release_len(self.tb.len(), partial_suffix(&self.tb, &sc)));
                let head = self.tb[..n].to_string();
                if self.str_streamed {
                    let piece = jesc(&head);
                    self.tool_args(d, &piece);
                } else {
                    self.vraw.push_str(&head);
                }
                let Some(e) = e else {
                    self.tb.drain(..n);
                    return false;
                };
                self.tb.drain(..e + sc.len());
                if self.str_streamed {
                    self.tool_args(d, "\"");
                } else {
                    // the held string, cast by the parameter's type; what does not cast stays a string
                    let held = std::mem::take(&mut self.vraw);
                    let t = held.trim_matches([' ', '\n', '\r', '\t']);
                    let ty = self.ptype.as_str();
                    let cast = if ty == "boolean" {
                        match t.to_ascii_lowercase().as_str() {
                            "true" | "1" => Some("true".to_string()),
                            "false" | "0" => Some("false".to_string()),
                            _ => None,
                        }
                    } else {
                        match parse_json(t) {
                            Ok(JsonValue::Int(i)) if ty == "integer" || ty == "number" => {
                                Some(i.to_string())
                            }
                            Ok(JsonValue::Float(f))
                                if ty == "integer" && f.fract() == 0.0 && f.abs() < 9e15 =>
                            {
                                Some((f as i64).to_string())
                            }
                            Ok(j @ JsonValue::Float(_)) if ty == "number" => Some(j.dump()),
                            Ok(j @ JsonValue::Object(_)) if ty == "object" => Some(j.dump()),
                            Ok(j @ JsonValue::Array(_)) if ty == "array" => Some(j.dump()),
                            Ok(JsonValue::Null) if ty == "null" => Some("null".to_string()),
                            _ => None,
                        }
                    };
                    let text = cast.unwrap_or_else(|| jstr(&held));
                    self.tool_args(d, &text);
                }
                self.cs = Cs::Next;
                true
            }
            Cs::Scalar => {
                let Some(e) = self.tb.find([',', '}', ']']) else {
                    let more = std::mem::take(&mut self.tb);
                    self.vraw.push_str(&more);
                    return false;
                };
                let head = self.tb[..e].to_string();
                self.vraw.push_str(&head);
                self.tb.drain(..e);
                let text = self.call_scalar_text();
                self.tool_args(d, &text);
                self.cs = Cs::Next;
                true
            }
            Cs::Next => {
                // after a value: a comma, or the end of the container
                self.call_skip_ws();
                let Some(c) = self.tb.chars().next() else {
                    return false;
                };
                self.tb.drain(..c.len_utf8());
                match c {
                    ',' => {
                        self.cs = if self.cstack.last() == Some(&b'{') {
                            Cs::Key
                        } else {
                            Cs::Value
                        }
                    }
                    '}' | ']' => self.call_pop(d),
                    // anything else between members is dropped
                    _ => {}
                }
                true
            }
            Cs::Close => {
                // the arguments are complete: up to the closing tag is ignored
                let tool_close = self.f.tool_close.clone();
                let Some(p) = self.tb.find(tool_close.as_str()) else {
                    let cut = release_len(self.tb.len(), partial_suffix(&self.tb, &tool_close));
                    self.tb.drain(..cut);
                    return false;
                };
                let rest = self.tb[p + tool_close.len()..].to_string();
                self.end_block(d, rest);
                false
            }
        }
    }

    /// The JSON-schema type the request's tool list gives parameter `param` of function `func`
    /// (empty: none).
    fn param_type(&self, func: &str, param: &str) -> String {
        let Some(tools) = self.tools.as_ref().and_then(JsonValue::as_array) else {
            return String::new();
        };
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            if f.get("name").and_then(JsonValue::as_str) != Some(func) {
                continue;
            }
            let ty = f
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(|p| p.get(param))
                .and_then(|p| p.get("type"));
            return match ty {
                Some(JsonValue::Str(s)) => s.clone(),
                Some(JsonValue::Array(a)) => a
                    .iter()
                    .filter_map(JsonValue::as_str)
                    .find(|s| *s != "null")
                    .unwrap_or("")
                    .to_string(),
                _ => String::new(),
            };
        }
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Streamed {
        r: String,
        c: String,
        calls: Vec<ToolCall>,
        args: BTreeMap<usize, String>,
        names: BTreeMap<usize, String>,
        ids: BTreeMap<usize, String>,
        headers: BTreeMap<usize, usize>,
        order_ok: bool,
    }

    impl Streamed {
        fn add(&mut self, d: Delta) {
            self.r.push_str(&d.reasoning);
            self.c.push_str(&d.content);
            self.calls.extend(d.tool_calls);
            for x in d.tool_deltas {
                if x.header {
                    *self.headers.entry(x.index).or_default() += 1;
                    self.names.insert(x.index, x.name.clone());
                    self.ids.insert(x.index, x.id.clone());
                } else if !self.headers.contains_key(&x.index) {
                    self.order_ok = false; // a fragment before its header
                }
                self.args.entry(x.index).or_default().push_str(&x.arguments);
            }
        }

        /// Every call's streamed header and fragments agree with the finished call.
        fn consistent(&self) -> bool {
            self.order_ok
                && self.headers.len() == self.calls.len()
                && self.calls.iter().enumerate().all(|(i, c)| {
                    self.headers.get(&i) == Some(&1)
                        && self.args.get(&i) == Some(&c.arguments)
                        && self.names.get(&i) == Some(&c.name)
                        && self.ids.get(&i) == Some(&c.id)
                        && parse_json(&c.arguments).is_ok_and(|j| j.as_object().is_some())
                })
        }
    }

    /// Feeds `text` in pieces of `step` characters.
    fn stream_parse(p: &mut OutputParser, text: &str, step: usize) -> Streamed {
        let mut s = Streamed {
            order_ok: true,
            ..Streamed::default()
        };
        let chars: Vec<char> = text.chars().collect();
        for piece in chars.chunks(step.max(1)) {
            if p.stopped() {
                break;
            }
            s.add(p.feed(&piece.iter().collect::<String>()));
        }
        s.add(p.finish());
        s
    }

    fn tools() -> JsonValue {
        parse_json(
            r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{
                 "city":{"type":"string"},"days":{"type":"integer"},"opts":{"type":"object"},"ok":{"type":"boolean"}}}}},
                {"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{
                 "path":{"type":"string"},"content":{"type":"string"}}}}}]"#,
        )
        .unwrap()
    }

    fn params(reasoning_open: bool) -> OutputParser {
        OutputParser::new(
            reasoning_open,
            true,
            Some(tools()),
            Vec::new(),
            OutputFormat::default(),
        )
    }

    fn gemma() -> OutputFormat {
        let cfg = parse_json(
            r#"{"response_template": {"fields": {
                "thinking": {"close": "<channel|>", "content": "text", "open": "<|channel>thought\n"},
                "tool_calls": {"close": "<tool_call|>", "content": "json",
                  "content_args": {"string_delims": [["<|\"|>", "<|\"|>"]], "unquoted_keys": true},
                  "open_pattern": "<\\|tool_call>call:(?P<name>\\w+)", "repeats": true}}}}"#,
        )
        .unwrap();
        OutputFormat::detect(Some(&cfg), "").0
    }

    fn args_of(c: &ToolCall) -> JsonValue {
        parse_json(&c.arguments).unwrap()
    }

    #[test]
    fn reasoning_and_content_split_at_every_piece_size() {
        let text = "\n First I think. \n</think>\n\nThe answer is 4.\n";
        for step in 1..=text.len() {
            let mut p = OutputParser::new(true, true, None, Vec::new(), OutputFormat::default());
            let s = stream_parse(&mut p, text, step);
            assert_eq!(
                (s.r.as_str(), s.c.as_str()),
                ("First I think.", "The answer is 4.\n"),
                "step {step}"
            );
        }
        // the model opens its reasoning itself; a reply without reasoning is all content
        let mut p = OutputParser::new(false, true, None, Vec::new(), OutputFormat::default());
        let s = stream_parse(&mut p, "<think>\nhm\n</think>\n\nHi", 3);
        assert_eq!((s.r.as_str(), s.c.as_str()), ("hm", "Hi"));
        let mut p = OutputParser::new(false, true, None, Vec::new(), OutputFormat::default());
        assert_eq!(
            stream_parse(&mut p, "<thin air> is content", 2).c,
            "<thin air> is content"
        );
        // reasoning left in the content when it is not split out
        let mut p = OutputParser::new(true, false, None, Vec::new(), OutputFormat::default());
        assert_eq!(stream_parse(&mut p, "a</think>b", 2).c, "a</think>b");
        // generation cut inside the reasoning
        let mut p = OutputParser::new(true, true, None, Vec::new(), OutputFormat::default());
        let s = stream_parse(&mut p, "still thinking </thi", 4);
        assert_eq!((s.r.as_str(), s.c.as_str()), ("still thinking </thi", ""));
    }

    #[test]
    fn stop_strings_cut_the_text_and_are_never_emitted() {
        for step in 1..=12 {
            let mut p = OutputParser::new(
                false,
                true,
                None,
                vec!["STOP".into(), "\n\n##".into()],
                OutputFormat::default(),
            );
            let s = stream_parse(&mut p, "one STO two\n\n## three STOP four", step);
            assert_eq!(s.c, "one STO two", "step {step}");
            assert!(p.stopped());
            assert_eq!(p.matched_stop(), "\n\n##");
        }
        // text that only looked like a stop string comes out at the end
        let mut p = OutputParser::new(
            false,
            true,
            None,
            vec!["STOP".into()],
            OutputFormat::default(),
        );
        let s = stream_parse(&mut p, "almost STO", 1);
        assert_eq!(s.c, "almost STO");
        assert!(!p.stopped() && p.matched_stop().is_empty());
    }

    #[test]
    fn parameter_block_calls_stream_at_every_split_point() {
        // reasoning, content, three calls; a string value with quotes / backslashes / a tab /
        // non-ASCII / a fake tag / a control character, a value that is only newlines, typed
        // values, an untyped parameter (JSON when it parses)
        let body2 = "line \"one\"\\n\ttab\nü€😀 <parameter=fake> x</paramet\u{1}y\n\nend";
        let text = format!(
            "Thinking it over.\n</think>\n\nOk.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew \"York\"\n</parameter>\n<parameter=days>\n3\n</parameter>\n\
             <parameter=opts>\n{{\"u\": \"c\", \"n\": [1, 2]}}\n</parameter>\n<parameter=ok>\nTrue\n</parameter>\n<parameter=extra>\n[1,2]\n</parameter>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a b.txt\n</parameter>\n<parameter=content>\n{body2}\n</parameter>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=write_file>\n<parameter=path>\n\n\n</parameter>\n<parameter=content>\n\n</parameter>\n</function>\n</tool_call>"
        );
        for step in 1..=text.chars().count() {
            let mut p = params(true);
            let s = stream_parse(&mut p, &text, step);
            assert_eq!(
                (s.r.as_str(), s.c.as_str()),
                ("Thinking it over.", "Ok."),
                "step {step}"
            );
            assert_eq!(s.calls.len(), 3, "step {step}");
            assert!(s.consistent(), "step {step}");
            assert_eq!(
                args_of(&s.calls[0]),
                parse_json(r#"{"city":"New \"York\"","days":3,"opts":{"u":"c","n":[1,2]},"ok":true,"extra":[1,2]}"#).unwrap(),
                "step {step}"
            );
            let a1 = args_of(&s.calls[1]);
            assert_eq!(
                a1.get("path").and_then(JsonValue::as_str),
                Some("/tmp/a b.txt")
            );
            assert_eq!(
                a1.get("content").and_then(JsonValue::as_str),
                Some(body2),
                "step {step}"
            );
            assert_eq!(
                s.calls[1].arguments,
                a1.dump(),
                "the streamed text is compact JSON"
            );
            let a2 = args_of(&s.calls[2]);
            assert_eq!(a2.get("path").and_then(JsonValue::as_str), Some("\n"));
            assert_eq!(a2.get("content").and_then(JsonValue::as_str), Some(""));
            assert_eq!(p.tool_calls(), s.calls.as_slice());
        }
    }

    #[test]
    fn a_call_header_goes_out_first_and_strings_stream() {
        let mut p = params(false);
        let d = p.feed("<tool_call>\n<function=write_file>\n");
        assert_eq!(d.tool_deltas.len(), 1);
        assert!(d.tool_deltas[0].header && d.tool_deltas[0].name == "write_file");
        assert!(d.tool_deltas[0].id.starts_with("call_") && d.tool_deltas[0].id.len() == 29);
        let d = p.feed("<parameter=content>\nhello wor");
        assert_eq!(d.tool_deltas[0].arguments, "{\"content\":\"hello wor");
        // the trailing newline is held (it may be the template's)
        let d = p.feed("ld\n");
        assert_eq!(d.tool_deltas[0].arguments, "ld");
        p.feed("x\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(
            p.tool_calls()[0].arguments,
            "{\"content\":\"hello world\\nx\"}"
        );
        // two calls never share an id
        let mut q = params(false);
        stream_parse(&mut q, "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>", 7);
        assert_eq!(q.tool_calls().len(), 2);
        assert_ne!(q.tool_calls()[0].id, q.tool_calls()[1].id);
    }

    #[test]
    fn cut_malformed_and_limited_calls() {
        // a call cut by the end of generation is closed and kept
        let mut p = params(false);
        let s = stream_parse(
            &mut p,
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
            3,
        );
        assert_eq!(s.calls.len(), 1);
        assert_eq!(s.calls[0].arguments, "{\"city\":\"Par\"}");
        assert!(s.consistent());
        // a block without a function is content; so is one cut before its name is complete
        let mut p = params(false);
        let s = stream_parse(&mut p, "See <tool_call> the docs </tool_call> ok", 2);
        assert!(s.calls.is_empty());
        assert_eq!(s.c, "See<tool_call> the docs </tool_call> ok");
        let mut p = params(false);
        let s = stream_parse(&mut p, "text <tool_call>\n<function=x", 4);
        assert!(s.calls.is_empty());
        assert_eq!(s.c, "text<tool_call>\n<function=x");
        // two functions in one block are two calls
        let mut p = params(false);
        let s = stream_parse(&mut p, "<tool_call>\n<function=a>\n</function>\n<function=b>\n<parameter=k>\n1\n</parameter>\n</function>\n</tool_call>", 5);
        assert_eq!(
            s.calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(s.calls[1].arguments, "{\"k\":1}");
        assert!(s.consistent());
        // parallel_tool_calls false: the first call ends the generation
        let mut p = params(false);
        p.set_max_calls(1);
        let s = stream_parse(&mut p, "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>", 5);
        assert_eq!(s.calls.len(), 1);
        assert_eq!(
            (s.calls[0].name.as_str(), s.calls[0].arguments.as_str()),
            ("a", "{}")
        );
        assert!(p.stopped() && p.matched_stop().is_empty());
        // without tools in the request a tool block is plain content
        let mut p = OutputParser::new(false, true, None, Vec::new(), OutputFormat::default());
        let s = stream_parse(
            &mut p,
            "<tool_call>\n<function=a>\n</function>\n</tool_call>",
            5,
        );
        assert!(s.calls.is_empty() && s.c.starts_with("<tool_call>"));
    }

    #[test]
    fn json_object_calls_are_read_at_their_closing_tag() {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": 2}}\n</tool_call>\n\
                    <tool_call>\n{\"name\": \"write_file\", \"arguments\": \"{\\\"path\\\": \\\"a\\\"}\"}\n</tool_call>";
        for step in 1..=text.len() {
            let mut p = params(false);
            let s = stream_parse(&mut p, text, step);
            assert_eq!(s.c, "Let me check.", "step {step}");
            assert_eq!(s.calls.len(), 2, "step {step}");
            assert!(s.consistent(), "step {step}");
            assert_eq!(s.calls[0].arguments, "{\"city\":\"Oslo\",\"days\":2}");
            assert_eq!(s.calls[1].arguments, "{\"path\":\"a\"}");
        }
        // not an object with a name: content
        let mut p = params(false);
        let s = stream_parse(&mut p, "<tool_call>\n{\"nope\": 1}\n</tool_call> after", 4);
        assert!(s.calls.is_empty());
        assert_eq!(s.c, "<tool_call>\n{\"nope\": 1}\n</tool_call> after");
        // cut before the closing tag but complete: kept
        let mut p = params(false);
        let s = stream_parse(
            &mut p,
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}",
            6,
        );
        assert_eq!(s.calls.len(), 1);
    }

    #[test]
    fn call_form_is_transcribed_to_json_at_every_split_point() {
        let f = gemma();
        assert_eq!(f.style, CallStyle::Call);
        assert_eq!(
            (f.tool_open.as_str(), f.tool_close.as_str()),
            ("<|tool_call>call:", "<tool_call|>")
        );
        assert_eq!(
            (f.think_open.as_str(), f.think_close.as_str()),
            ("<|channel>thought\n", "<channel|>")
        );
        let text = "<|channel>thought\nNeed the weather.\n<channel|>Sure.\n<|tool_call>call:get_weather{city:<|\"|>New \"York\", NY<|\"|>,days:<|\"|>3<|\"|>,ok:true,\
                    opts:{u:<|\"|>c<|\"|>,n:[1,2.5,null,[<|\"|>x<|\"|>]],e:{}},extra:7,}<tool_call|>\
                    <|tool_call>call:write_file{path:7,content:<|\"|>a\n{b}<|\"|>}<tool_call|>";
        for step in 1..=text.chars().count() {
            let mut p = OutputParser::new(false, true, Some(tools()), Vec::new(), f.clone());
            let s = stream_parse(&mut p, text, step);
            assert_eq!(
                (s.r.as_str(), s.c.as_str()),
                ("Need the weather.", "Sure."),
                "step {step}"
            );
            assert_eq!(s.calls.len(), 2, "step {step}");
            assert!(s.consistent(), "step {step}: {:?}", s.calls);
            // a string for an integer parameter is cast; a bare number for a string one is a string
            assert_eq!(
                s.calls[0].arguments,
                r#"{"city":"New \"York\", NY","days":3,"ok":true,"opts":{"u":"c","n":[1,2.5,null,["x"]],"e":{}},"extra":7}"#,
                "step {step}"
            );
            assert_eq!(
                s.calls[1].arguments, "{\"path\":\"7\",\"content\":\"a\\n{b}\"}",
                "step {step}"
            );
        }
        // cut inside a string: closed and kept; no name: content
        let mut p = OutputParser::new(false, true, Some(tools()), Vec::new(), f.clone());
        let s = stream_parse(&mut p, "<|tool_call>call:write_file{path:<|\"|>/tm", 3);
        assert_eq!(s.calls[0].arguments, "{\"path\":\"/tm\"}");
        assert!(s.consistent());
        let mut p = OutputParser::new(false, true, Some(tools()), Vec::new(), f.clone());
        let s = stream_parse(&mut p, "x <|tool_call>call: nothing", 3);
        assert!(s.calls.is_empty());
        assert_eq!(s.c, "x <|tool_call>call: nothing");
        let mut p = OutputParser::new(false, true, Some(tools()), Vec::new(), f);
        let s = stream_parse(&mut p, "<|tool_call>call:noargs{}<tool_call|>", 2);
        assert_eq!(s.calls[0].arguments, "{}");
    }

    #[test]
    fn formats_are_detected_and_forced_prefixes_parse() {
        let d = OutputFormat::default();
        assert_eq!(
            OutputFormat::detect(None, "... <function=example> ..."),
            (d.clone(), None)
        );
        let json = OutputFormat::detect(
            None,
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}",
        )
        .0;
        assert_eq!(json.style, CallStyle::Json);
        let bad = parse_json(r#"{"response_template": {"fields": {"tool_calls": {"open_pattern": "a.*", "close": "b", "content": "json"}}}}"#).unwrap();
        let (fallback, why) = OutputFormat::detect(Some(&bad), "");
        assert_eq!(fallback, d);
        assert!(why.is_some_and(|w| w.contains("tool_calls")));

        assert_eq!(d.forced_prefix(""), "<tool_call>\n");
        assert_eq!(
            d.forced_prefix("get_weather"),
            "<tool_call>\n<function=get_weather>\n"
        );
        assert_eq!(
            json.forced_prefix("f"),
            "<tool_call>\n{\"name\": \"f\", \"arguments\": "
        );
        assert_eq!(gemma().forced_prefix("f"), "<|tool_call>call:f{");
        assert_eq!(gemma().forced_prefix(""), "<|tool_call>call:");
        assert!(gemma().is_marker_piece("<|\"|>") && gemma().is_marker_piece("<|channel>"));
        assert!(!gemma().is_marker_piece("<turn|>") && !d.is_marker_piece(""));

        // the forced opening is fed first; the model's text completes the call
        for (f, name, rest, want) in [
            (
                d.clone(),
                "get_weather",
                "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
                "{\"city\":\"Oslo\"}",
            ),
            (
                d.clone(),
                "",
                "<function=get_weather>\n</function>\n</tool_call>",
                "{}",
            ),
            (
                json.clone(),
                "get_weather",
                "{\"city\": \"Oslo\"}}\n</tool_call>",
                "{\"city\":\"Oslo\"}",
            ),
            (
                json.clone(),
                "",
                "get_weather\", \"arguments\": {}}\n</tool_call>",
                "{}",
            ),
            (
                gemma(),
                "get_weather",
                "city:<|\"|>Oslo<|\"|>}<tool_call|>",
                "{\"city\":\"Oslo\"}",
            ),
            (gemma(), "", "get_weather{}<tool_call|>", "{}"),
        ] {
            let mut p = OutputParser::new(false, true, Some(tools()), Vec::new(), f.clone());
            let mut s = Streamed {
                order_ok: true,
                ..Streamed::default()
            };
            s.add(p.feed(&f.forced_prefix(name)));
            s.add(p.feed(rest));
            s.add(p.finish());
            assert_eq!(s.calls.len(), 1, "{:?} {name}", f.style);
            assert_eq!(
                (s.calls[0].name.as_str(), s.calls[0].arguments.as_str()),
                ("get_weather", want)
            );
            assert!(s.consistent() && s.c.is_empty());
        }
    }

    fn blocks() -> OutputFormat {
        OutputFormat::detect(
            None,
            "{{ '<|message_user|><|content_text|>' ~ m ~ '<|end_message|><|message_model|>' }}",
        )
        .0
    }

    fn block_parser(tools_on: bool, split: bool) -> OutputParser {
        OutputParser::new(false, split, tools_on.then(tools), Vec::new(), blocks())
    }

    const H: &str = "<|message_model|>";
    const E: &str = "<|end_message|>";

    #[test]
    fn message_blocks_split_at_every_piece_size() {
        let reply = format!(
            "{H}<|content_thinking|>Let me think <about> it.{E}{H}<|content_text|>Hello <|world!{E}\
             {H}get_weather<|content_invoke_tool_json|>{{\"name\":\"get_weather\",\"args\":{{\"city\":\"Oslo\",\"days\":2}}}}{E}\
             {H}<|content_invoke_tool_json|>{{\"name\":\"write_file\",\"args\":{{\"path\":\"a\",\"content\":\"x<|y\"}}}}{E}\
             <|content_model_end_sampling|>trailing text is not read"
        );
        for step in 1..reply.chars().count() {
            let mut p = block_parser(true, true);
            let s = stream_parse(&mut p, &reply, step);
            assert_eq!(s.r, "Let me think <about> it.", "step {step}");
            assert_eq!(s.c, "Hello <|world!", "step {step}");
            assert!(s.consistent(), "step {step}");
            assert_eq!(s.calls.len(), 2, "step {step}");
            assert_eq!(s.calls[0].name, "get_weather");
            assert_eq!(args_of(&s.calls[0]).get("days").unwrap().as_i64(), Some(2));
            assert_eq!(s.calls[1].name, "write_file");
            assert_eq!(
                args_of(&s.calls[1]).get("content").unwrap().as_str(),
                Some("x<|y")
            );
            assert_eq!((p.content(), p.reasoning()), (s.c.as_str(), s.r.as_str()));
        }
    }

    #[test]
    fn message_blocks_without_split_tools_or_ends() {
        let reply = format!(
            "{H}<|content_thinking|>hmm{E}{H}x<|content_invoke_tool_json|>{{\"name\":\"x\",\"args\":{{}}}}{E}"
        );
        // reasoning stays in the content; without tools a call block is text
        let s = stream_parse(&mut block_parser(false, false), &reply, 3);
        assert_eq!(
            (s.r.as_str(), s.c.as_str()),
            ("", "hmm{\"name\":\"x\",\"args\":{}}")
        );
        assert!(s.calls.is_empty());
        // a call cut off by the end of generation: kept when whole, dropped (never shown) when not
        let whole = format!("{H}<|content_invoke_tool_json|>{{\"name\":\"get_weather\",\"args\":{{\"city\":\"Oslo\"}}}}");
        let s = stream_parse(&mut block_parser(true, true), &whole, 2);
        assert_eq!(s.calls.len(), 1);
        assert!(s.consistent());
        let cut = format!("{H}<|content_text|>Sure.{E}{H}get_weather<|content_invoke_tool_json|>{{\"name\":\"get_weather\",\"args\":{{\"ci");
        for step in 1..cut.len() {
            let s = stream_parse(&mut block_parser(true, true), &cut, step);
            assert_eq!((s.c.as_str(), s.calls.len()), ("Sure.", 0), "step {step}");
        }
        // a closed block that is no call is given back as content
        let bad = format!("{H}<|content_invoke_tool_json|>not json{E}{H}<|content_text|>!{E}");
        let s = stream_parse(&mut block_parser(true, true), &bad, 4);
        assert_eq!((s.c.as_str(), s.calls.len()), ("not json!", 0));
        // the name before the call marker stands in for a JSON without one
        let named =
            format!("{H}get_weather<|content_invoke_tool_json|>{{\"args\":{{\"days\":1}}}}{E}");
        let s = stream_parse(&mut block_parser(true, true), &named, 5);
        assert_eq!(s.calls[0].name, "get_weather");
        // another role's header ends the reply
        let over = format!("{H}<|content_text|>Done.{E}<|message_user|><|content_text|>more{E}");
        let s = stream_parse(&mut block_parser(true, true), &over, 1);
        assert_eq!(s.c, "Done.");
        // one call only: the reply ends after it
        let mut p = block_parser(true, true);
        p.set_max_calls(1);
        let two = format!(
            "{H}<|content_invoke_tool_json|>{{\"name\":\"a\",\"args\":{{}}}}{E}{H}<|content_invoke_tool_json|>{{\"name\":\"b\",\"args\":{{}}}}{E}"
        );
        let s = stream_parse(&mut p, &two, 7);
        assert!(p.stopped() && s.calls.len() == 1 && s.calls[0].name == "a");
        // a prompt that ends inside a reasoning block starts there
        let mut p = OutputParser::new(true, true, None, Vec::new(), blocks());
        let s = stream_parse(
            &mut p,
            &format!("still thinking{E}{H}<|content_text|>ok{E}"),
            2,
        );
        assert_eq!((s.r.as_str(), s.c.as_str()), ("still thinking", "ok"));
    }

    #[test]
    fn message_block_form_is_detected_and_forces_named_calls() {
        let f = blocks();
        assert_eq!(f.style, CallStyle::Blocks);
        assert_eq!(
            (f.think_open.as_str(), f.think_close.as_str()),
            ("<|content_thinking|>", E)
        );
        assert!(f.is_marker_piece(E) && f.is_marker_piece("<|content_model_end_sampling|>"));
        assert!(!f.is_marker_piece("<|endoftext|>") && !f.is_marker_piece(""));
        assert!(f.stop_markers().contains(&"<|message_user|>"));
        assert!(OutputFormat::default().stop_markers().is_empty());
        // a call to any tool cannot be opened (its name comes first); a named one can
        assert!(
            !f.can_force("") && f.can_force("get_weather") && OutputFormat::default().can_force("")
        );
        let prefix = f.forced_prefix("get_weather");
        assert_eq!(
            prefix,
            format!(
                "{H}get_weather<|content_invoke_tool_json|>{{\"name\":\"get_weather\",\"args\":"
            )
        );
        let mut p = block_parser(true, true);
        let opened = p.feed(&prefix);
        let rest = format!("{{\"city\":\"Oslo\"}}}}{E}<|content_model_end_sampling|>");
        let t = stream_parse(&mut p, &rest, 3);
        assert_eq!(t.calls.len(), 1);
        assert_eq!(args_of(&t.calls[0]).dump(), "{\"city\":\"Oslo\"}");
        assert!(t.c.is_empty() && opened.is_empty());
    }
}
