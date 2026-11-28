//! Constrained decoding: a reply grammar (JSON mode, a JSON schema, strict tool calls) compiled
//! to a byte-level pushdown automaton, applied per decode step as a mask over the model's
//! vocabulary.
//!
//! - [`build`]: the JSON Schema subset and the reply forms (a JSON document, tool calls in the
//!   model's call form, free text with strict calls) compiled to a [`Grammar`]. Request parsing:
//!   an unsupported keyword is refused with its name.
//! - this module: the automaton. A [`Cursor`] is the set of stacks the automaton follows (an
//!   ambiguous `anyOf` keeps several alive until the bytes decide); a byte maps each stack to
//!   zero or more successors.
//! - [`vocab`]: the model's token byte strings as a trie, walked once per automaton state to
//!   find the admitted tokens.
//! - [`constraint`]: one request's run: the reply phase, the token mask of each state (cached),
//!   and the commit of every emitted token.
//!
//! The decisions around the automaton are verified (`verified/grammar.bend`): how deep a stack
//! may grow (`gram_push_ok`), how many alternatives it may follow (`gram_alive_ok`), how much
//! work one call into it may do and how much memory a compiled grammar may take
//! (`gram_budget_ok`), which
//! tokens are admitted (`gram_admit`), when the reply may end (`gram_accepting`), the reply's
//! phase (`gram_phase_step`), that a refused token is never emitted (`gram_token_class` with
//! `pen_emit_ok`) and that an empty mask never picks (`gram_can_pick`). The JSON grammar itself
//! is RFC 8259 and the schema keywords are JSON Schema: formats defined outside the engine, kept
//! here with Rust tests (one differential against `crate::json::parse_json`).

pub mod build;
pub mod constraint;
pub mod vocab;

#[cfg(test)]
pub(crate) mod tests;

use crate::verified;

/// Frames a stack may hold.
pub const MAX_STACK: u64 = 160;
/// Alternatives the automaton may follow at once.
pub const MAX_ALIVE: u64 = 256;
/// Whitespace bytes allowed in one run between JSON tokens (a model asked for JSON that has not
/// been told so can otherwise emit whitespace until its token limit).
pub const WS_MAX: u32 = 96;
/// Digits allowed in a number's integer part, its fraction and its exponent (an integer of up to
/// 18 digits fits an i64).
pub const INT_DIGITS: u32 = 18;
pub const FRAC_DIGITS: u32 = 18;
/// Two exponent digits keep every admitted number finite as an `f64` (at most about 1e117), so a
/// tool argument read back through the JSON parser never turns into an infinity (written `null`).
pub const EXP_DIGITS: u32 = 2;
/// Work (frame operations, each charged by the frames it copies: [`step_cost`]) one call into
/// the automaton may do: a byte step, the start, a completion check, one token's commit, or the
/// whole of one mask. Past it the call stops with [`Work::exhausted`] set and the request ends
/// with an error. Measured on a 248K-token vocabulary: a mask of an ordinary schema takes 1-10
/// thousand, one that follows 256 alternatives about 240 thousand (7 ms on the dev host); the
/// cap is about 30 times that. One unit took 44-75 ns at 3 to 142 frames (release build, 199
/// alternatives alive), so the cap is under a second of one host core.
pub const MAX_WORK: u64 = 8_000_000;
/// Work every mask and commit of one request may take together: at the measured 44-75 ns a
/// unit, about 3 to 5 minutes of one host core at most.
pub const MAX_REQUEST_WORK: u64 = 4_000_000_000;
/// Frames one unit of work may copy: a frame operation copies the stack it works on, so an
/// operation on a stack of `n` frames is charged `1 + n / FRAMES_PER_UNIT` units (see
/// [`step_cost`]), and the caps above bound time at every depth.
pub const FRAMES_PER_UNIT: u64 = 8;

/// The work charged for one frame operation on a stack of `frames` frames.
pub fn step_cost(frames: usize) -> u64 {
    1 + frames as u64 / FRAMES_PER_UNIT
}

pub type NodeId = u32;

/// A byte pattern searched for in free text, with its prefix function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pat {
    pub bytes: Vec<u8>,
    fail: Vec<u32>,
}

impl Pat {
    pub fn new(bytes: &[u8]) -> Self {
        let mut fail = vec![0u32; bytes.len()];
        let mut k = 0usize;
        for i in 1..bytes.len() {
            while k > 0 && bytes[i] != bytes[k] {
                k = fail[k - 1] as usize;
            }
            if bytes[i] == bytes[k] {
                k += 1;
            }
            fail[i] = k as u32;
        }
        Self {
            bytes: bytes.to_vec(),
            fail,
        }
    }

    /// Matched prefix length after byte `b`, from `k` matched (`k < len`).
    pub fn next(&self, mut k: u32, b: u8) -> u32 {
        loop {
            if self.bytes.get(k as usize) == Some(&b) {
                return k + 1;
            }
            if k == 0 {
                return 0;
            }
            k = self.fail[k as usize - 1];
        }
    }

    pub fn len(&self) -> u32 {
        self.bytes.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// A node of a literal trie: a [`Node::Switch`] (literal alternatives, each with an optional
/// continuation) or an object's key set at one position.
#[derive(Debug, Clone, Default)]
pub struct TrieNode {
    pub children: Vec<(u8, u32)>,
    /// `NO_LEAF`: no literal ends here; `FINISH`: a literal ends and the node is complete; else a
    /// continuation node (Switch) or a property index (object keys).
    pub leaf: u32,
}

pub const NO_LEAF: u32 = u32::MAX;
pub const FINISH: u32 = u32::MAX - 1;

impl TrieNode {
    fn child(&self, b: u8) -> Option<u32> {
        self.children
            .binary_search_by_key(&b, |c| c.0)
            .ok()
            .map(|i| self.children[i].1)
    }
}

#[derive(Debug, Clone)]
pub struct PropDef {
    pub value: NodeId,
    pub required: bool,
}

/// An object with known properties. One trie holds every quoted key (its leaves are property
/// indices); at position `p` the keys that may come next are the properties `p ..= last[p]` (up
/// to and including the next required one), so a trie node may be entered only while its
/// subtree holds such an index.
#[derive(Debug, Clone)]
pub struct ObjDef {
    pub props: Vec<PropDef>,
    /// The key trie's root; its nodes are `keys .. keys + spans.len()`.
    pub keys: u32,
    /// Per trie node: the range of `idx` that lists, in increasing order, the property indices
    /// of its subtree.
    pub spans: Vec<(u32, u32)>,
    pub idx: Vec<u32>,
    /// Per position `p < props.len()`: the last property that may come next.
    pub last: Vec<u32>,
    /// Per position `p` (0 ..= props.len()): no required property remains, so the object may close.
    pub close_ok: Vec<bool>,
}

impl ObjDef {
    /// Trie node `t` leads to a key that may come at position `p`.
    fn key_alive(&self, t: u32, p: u32) -> bool {
        let (Some(&last), Some(&(a, b))) = (
            self.last.get(p as usize),
            self.spans.get(t.wrapping_sub(self.keys) as usize),
        ) else {
            return false;
        };
        let list = &self.idx[a as usize..b as usize];
        let i = list.partition_point(|&x| x < p);
        list.get(i).is_some_and(|&x| x <= last)
    }
}

#[derive(Debug, Clone)]
pub enum Node {
    /// Exactly these bytes.
    Lit(Vec<u8>),
    Seq(Vec<NodeId>),
    Alt(Vec<NodeId>),
    /// Zero or more of the child (which must not match the empty string).
    Rep(NodeId),
    /// Up to [`WS_MAX`] JSON whitespace bytes.
    Ws,
    /// A JSON string whose raw bytes hold none of `forbid` (at most 3 patterns, each shorter
    /// than 256 bytes and without a quote): the tags a reply parser splits the text at.
    Str { forbid: Vec<Pat> },
    /// A JSON number (`integer`: no fraction, no exponent).
    Num { integer: bool },
    /// Literal alternatives (a trie root): enum values, a const, a tool name.
    Switch(u32),
    /// An object with known properties, in the schema's order (index into `objs`).
    Object(u32),
    /// An object with free keys (each a `key` string node), every value matching `value`.
    Map { key: NodeId, value: NodeId },
    Array {
        items: NodeId,
        min: u32,
        max: u32,
    },
    /// Raw text that ends with `term` and holds none of `forbid` (at most 3 patterns, each
    /// shorter than 256 bytes).
    Raw { term: Pat, forbid: Vec<Pat> },
    /// Free text in which `open` starts a `call`, after which the text resumes.
    Text { open: Pat, call: NodeId },
}

/// One frame of a stack: a node and its local state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Frame {
    pub node: NodeId,
    pub ph: u8,
    /// Whitespace bytes in the current run (JSON containers).
    pub w: u8,
    pub x: u32,
    /// An object's position while its key is read; a string's matched lengths of its forbidden
    /// patterns (one byte each).
    pub y: u32,
}

impl Frame {
    fn new(node: NodeId, ph: u8, x: u32) -> Self {
        Self { node, ph, w: 0, x, y: 0 }
    }
}

/// The work one call into the automaton has done, against its cap. Whether a charge fits is the
/// verified `gram_budget_ok`, and the ledger after it is the verified `gram_budget_charge`: a
/// refused charge leaves the ledger past its cap, where every further charge is refused (laws
/// `gram_budget_refused_latches`, `gram_budget_over_stays_over`): the call stays cut short.
#[derive(Debug, Clone)]
pub struct Work {
    used: u64,
    cap: u64,
}

impl Work {
    /// `cap` below `u64::MAX`.
    pub fn new(cap: u64) -> Self {
        Self {
            used: 0,
            cap: cap.min(u64::MAX - 1),
        }
    }

    /// One frame operation on a stack of `frames` frames ([`step_cost`]); `false` (and
    /// exhausted from then on) once the cap is reached.
    fn spend(&mut self, frames: usize) -> bool {
        let cost = step_cost(frames);
        let ok = verified::gram_budget_ok(self.used, cost, self.cap);
        self.used = verified::gram_budget_charge(self.used, cost, self.cap);
        ok
    }

    /// The cap was reached: what the call returned is incomplete and must not be used.
    pub fn exhausted(&self) -> bool {
        !verified::gram_budget_ok(self.used, 0, self.cap)
    }

    pub fn used(&self) -> u64 {
        self.used.min(self.cap)
    }
}

pub type Stack = Vec<Frame>;

/// The set of stacks the automaton follows, sorted, without duplicates. An empty set is a dead
/// cursor (never kept: a step that leaves nothing returns `None`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cursor {
    pub stacks: Vec<Stack>,
}

// frame phases
const DONE: u8 = 255;
// Seq / Rep
const SEQ_RUN: u8 = 0;
const REP_READY: u8 = 0;
const REP_CHILD: u8 = 1;
// Str
const S_START: u8 = 0;
const S_NORMAL: u8 = 1;
const S_ESC: u8 = 2;
const S_HEX: u8 = 3;
// after a high-surrogate escape (`\uD800`-`\uDBFF`): its low half's `\`, `u` and four digits
const S_LOW_BS: u8 = 4;
const S_LOW_U: u8 = 5;
const S_LOW_HEX: u8 = 6;
// a `\u` escape's state in `x`: digits read (low bits), its first digit was `D`, it is a high
// surrogate (known after the second digit)
const HEX_COUNT: u32 = 0xFF;
const HEX_D: u32 = 1 << 8;
const HEX_HIGH: u32 = 1 << 9;
// Num
const N_START: u8 = 0;
const N_MINUS: u8 = 1;
const N_ZERO: u8 = 2;
const N_INT: u8 = 3;
const N_DOT: u8 = 4;
const N_FRAC: u8 = 5;
const N_E: u8 = 6;
const N_ESIGN: u8 = 7;
const N_EXP: u8 = 8;
// Object
const O_OPEN: u8 = 0;
const O_FIRST: u8 = 1;
const O_KEY: u8 = 2;
const O_COLON: u8 = 3;
const O_PRE: u8 = 4;
const O_VALUE: u8 = 5;
const O_AFTER: u8 = 6;
const O_COMMA: u8 = 7;
// Map
const M_OPEN: u8 = 0;
const M_FIRST: u8 = 1;
const M_KEY: u8 = 2;
const M_COLON: u8 = 3;
const M_PRE: u8 = 4;
const M_VALUE: u8 = 5;
const M_AFTER: u8 = 6;
const M_COMMA: u8 = 7;
// Array
const A_OPEN: u8 = 0;
const A_FIRST: u8 = 1;
const A_ITEM: u8 = 2;
const A_AFTER: u8 = 3;
const A_COMMA: u8 = 4;
// Text
const T_TEXT: u8 = 0;
const T_CALL: u8 = 1;
// Raw
const R_RUN: u8 = 0;

/// Epsilon steps one byte may take (entering and leaving frames); bounds a grammar with a
/// nullable loop.
const FUEL: u32 = 4 * MAX_STACK as u32;

pub fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}

/// A compiled reply grammar.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub nodes: Vec<Node>,
    pub tries: Vec<TrieNode>,
    pub objs: Vec<ObjDef>,
    pub root: NodeId,
}

impl Grammar {
    /// The cursor before the reply's first byte (`None`: the grammar matches nothing, or the
    /// work ran out).
    pub fn start(&self) -> Option<Cursor> {
        self.start_in(&mut Work::new(MAX_WORK))
    }

    pub fn start_in(&self, w: &mut Work) -> Option<Cursor> {
        let mut out = Vec::new();
        self.enter(Vec::new(), self.root, &mut out, FUEL, w);
        Self::cursor(out, w)
    }

    fn cursor(mut stacks: Vec<Stack>, w: &Work) -> Option<Cursor> {
        if w.exhausted() {
            return None;
        }
        stacks.sort_unstable();
        stacks.dedup();
        if stacks.is_empty() || !verified::gram_alive_ok(stacks.len() as u64, MAX_ALIVE) {
            return None;
        }
        Some(Cursor { stacks })
    }

    /// The cursor after byte `b` (`None`: the byte is refused).
    pub fn step(&self, c: &Cursor, b: u8) -> Option<Cursor> {
        self.step_in(c, b, &mut Work::new(MAX_WORK))
    }

    /// The cursor after byte `b`, charged to `w` (`None` with `w` exhausted: not decided).
    pub fn step_in(&self, c: &Cursor, b: u8, w: &mut Work) -> Option<Cursor> {
        let mut out = Vec::new();
        for st in &c.stacks {
            self.feed(st.clone(), b, &mut out, FUEL, w);
        }
        Self::cursor(out, w)
    }

    pub fn step_bytes(&self, c: &Cursor, bytes: &[u8]) -> Option<Cursor> {
        self.step_bytes_in(c, bytes, &mut Work::new(MAX_WORK))
    }

    pub fn step_bytes_in(&self, c: &Cursor, bytes: &[u8], w: &mut Work) -> Option<Cursor> {
        let mut cur = c.clone();
        for &b in bytes {
            cur = self.step_in(&cur, b, w)?;
        }
        Some(cur)
    }

    /// The reply may end here: some stack completes without another byte.
    pub fn complete(&self, c: &Cursor) -> bool {
        self.complete_in(c, &mut Work::new(MAX_WORK))
    }

    pub fn complete_in(&self, c: &Cursor, w: &mut Work) -> bool {
        let done = c.stacks.iter().any(|s| self.completes(s.clone(), FUEL, w));
        done && !w.exhausted()
    }

    fn completes(&self, mut st: Stack, fuel: u32, w: &mut Work) -> bool {
        let Some(&f) = st.last() else {
            return true;
        };
        if fuel == 0 || !w.spend(st.len()) || !self.can_finish(f) {
            return false;
        }
        st.pop();
        if st.is_empty() {
            return true;
        }
        let mut parents = Vec::new();
        self.child_done(st, &mut parents, fuel - 1, w);
        parents.into_iter().any(|p| self.completes(p, fuel - 1, w))
    }

    /// Pushes `id` (alternatives: one stack per branch) onto `st`.
    fn enter(&self, mut st: Stack, id: NodeId, out: &mut Vec<Stack>, fuel: u32, w: &mut Work) {
        // `fuel` bounds the recursion's depth, `w` its total work across every branch
        if fuel == 0 || !w.spend(st.len()) {
            return;
        }
        let node = &self.nodes[id as usize];
        if let Node::Alt(bs) = node {
            for &b in bs {
                self.enter(st.clone(), b, out, fuel - 1, w);
            }
            return;
        }
        if !verified::gram_push_ok(st.len() as u64, MAX_STACK) {
            return;
        }
        match node {
            Node::Seq(cs) if cs.is_empty() => {
                st.push(Frame::new(id, DONE, 0));
                out.push(st);
            }
            Node::Seq(cs) => {
                st.push(Frame::new(id, SEQ_RUN, 0));
                self.enter(st, cs[0], out, fuel - 1, w);
            }
            Node::Switch(root) => {
                st.push(Frame::new(id, 0, *root));
                out.push(st);
            }
            _ => {
                st.push(Frame::new(id, 0, 0));
                out.push(st);
            }
        }
    }

    fn can_finish(&self, f: Frame) -> bool {
        match &self.nodes[f.node as usize] {
            Node::Lit(b) => f.x as usize == b.len(),
            Node::Ws => true,
            Node::Rep(_) => f.ph == REP_READY,
            Node::Num { .. } => matches!(f.ph, N_ZERO | N_INT | N_FRAC | N_EXP),
            Node::Switch(_) => self.tries[f.x as usize].leaf == FINISH,
            Node::Text { .. } => f.ph == T_TEXT,
            _ => f.ph == DONE,
        }
    }

    /// The parent on top of `st` resumes after its child completed.
    fn child_done(&self, mut st: Stack, out: &mut Vec<Stack>, fuel: u32, w: &mut Work) {
        let Some(top) = st.last_mut() else {
            return;
        };
        let id = top.node;
        match &self.nodes[id as usize] {
            Node::Seq(cs) => {
                let next = top.x + 1;
                if next as usize >= cs.len() {
                    top.ph = DONE;
                    out.push(st);
                } else {
                    top.x = next;
                    let child = cs[next as usize];
                    self.enter(st, child, out, fuel, w);
                }
            }
            Node::Rep(_) => {
                top.ph = REP_READY;
                out.push(st);
            }
            Node::Object(_) if top.ph == O_VALUE => {
                top.ph = O_AFTER;
                top.w = 0;
                out.push(st);
            }
            Node::Map { .. } if top.ph == M_KEY => {
                top.ph = M_COLON;
                top.w = 0;
                out.push(st);
            }
            Node::Map { .. } if top.ph == M_VALUE => {
                top.ph = M_AFTER;
                top.w = 0;
                out.push(st);
            }
            Node::Array { min, max, .. } if top.ph == A_ITEM => {
                top.ph = A_AFTER;
                top.w = 0;
                if *min > 0 || *max != u32::MAX {
                    top.x += 1;
                }
                out.push(st);
            }
            Node::Text { .. } if top.ph == T_CALL => {
                top.ph = T_TEXT;
                top.x = 0;
                out.push(st);
            }
            // a child completed under a frame that runs none: nothing resumes
            _ => {}
        }
    }

    fn feed(&self, st: Stack, b: u8, out: &mut Vec<Stack>, fuel: u32, w: &mut Work) {
        if fuel == 0 || !w.spend(st.len()) {
            return;
        }
        let Some(&f) = st.last() else {
            return;
        };
        if self.can_finish(f) {
            let mut p = st.clone();
            p.pop();
            if !p.is_empty() {
                let mut parents = Vec::new();
                self.child_done(p, &mut parents, fuel - 1, w);
                for q in parents {
                    self.feed(q, b, out, fuel - 1, w);
                }
            }
        }
        self.consume(st, f, b, out, fuel, w);
    }

    /// Sets the top frame to `f` and enters `child` above it, then feeds `b` to the child.
    fn enter_feed(
        &self,
        mut st: Stack,
        f: Frame,
        child: NodeId,
        b: u8,
        out: &mut Vec<Stack>,
        fuel: u32,
        w: &mut Work,
    ) {
        *st.last_mut().expect("a frame") = f;
        let mut entered = Vec::new();
        self.enter(st, child, &mut entered, fuel - 1, w);
        for s in entered {
            self.feed(s, b, out, fuel - 1, w);
        }
    }

    fn set_top(mut st: Stack, f: Frame, out: &mut Vec<Stack>) {
        *st.last_mut().expect("a frame") = f;
        out.push(st);
    }

    /// Whitespace inside a JSON container: counted, up to [`WS_MAX`].
    fn ws(st: Stack, mut f: Frame, out: &mut Vec<Stack>) {
        if (f.w as u32) < WS_MAX {
            f.w += 1;
            Self::set_top(st, f, out);
        }
    }

    /// The matched lengths of `forbid` (packed one byte each) after byte `b`; `None`: a
    /// forbidden pattern is complete.
    fn forbid_next(forbid: &[Pat], packed: u32, b: u8) -> Option<u32> {
        let mut next = 0u32;
        for (i, p) in forbid.iter().enumerate() {
            let k = p.next((packed >> (8 * i)) & 0xFF, b);
            if k == p.len() {
                return None;
            }
            next |= k << (8 * i);
        }
        Some(next)
    }

    fn consume(&self, st: Stack, f: Frame, b: u8, out: &mut Vec<Stack>, fuel: u32, w: &mut Work) {
        let mut g = f;
        match &self.nodes[f.node as usize] {
            Node::Lit(bytes) => {
                if bytes.get(f.x as usize) == Some(&b) {
                    g.x += 1;
                    Self::set_top(st, g, out);
                }
            }
            Node::Ws => {
                if is_ws(b) && f.x < WS_MAX {
                    g.x += 1;
                    Self::set_top(st, g, out);
                }
            }
            Node::Rep(child) => {
                if f.ph == REP_READY {
                    g.ph = REP_CHILD;
                    self.enter_feed(st, g, *child, b, out, fuel, w);
                }
            }
            Node::Str { forbid } => {
                if f.ph != S_START {
                    // every raw byte of the body, escapes included, is what a parser searches
                    match Self::forbid_next(forbid, f.y, b) {
                        Some(y) => g.y = y,
                        None => return,
                    }
                }
                match f.ph {
                    S_START if b == b'"' => {
                        g.ph = S_NORMAL;
                        Self::set_top(st, g, out);
                    }
                    S_NORMAL => {
                        if b == b'"' {
                            g.ph = DONE;
                        } else if b == b'\\' {
                            g.ph = S_ESC;
                        } else if b < 0x20 {
                            return;
                        }
                        Self::set_top(st, g, out);
                    }
                    S_ESC => {
                        if b"\"\\/bfnrt".contains(&b) {
                            g.ph = S_NORMAL;
                        } else if b == b'u' {
                            g.ph = S_HEX;
                            g.x = 0;
                        } else {
                            return;
                        }
                        Self::set_top(st, g, out);
                    }
                    // RFC 8259 escapes a code point above U+FFFF as a surrogate pair: a high
                    // half `\uD800`-`\uDBFF` is followed at once by a low half
                    // `\uDC00`-`\uDFFF`, and a low half never stands alone. Each is refused at
                    // the digit that decides it, so every admitted prefix can still complete.
                    S_HEX | S_LOW_HEX if b.is_ascii_hexdigit() => {
                        let low = f.ph == S_LOW_HEX;
                        let v = (b as char).to_digit(16).unwrap_or(0);
                        let d = f.x & HEX_D != 0;
                        g.x = match f.x & HEX_COUNT {
                            0 if low && v != 0xD => return,
                            0 => 1 | if v == 0xD { HEX_D } else { 0 },
                            // a low half needs DC-DF; outside one, DC-DF is a lone low half
                            1 if low && v < 0xC => return,
                            1 if !low && d && v >= 0xC => return,
                            1 => 2 | if !low && d && v >= 0x8 { HEX_HIGH } else { 0 },
                            2 => 3 | (f.x & HEX_HIGH),
                            _ => {
                                g.ph = if f.x & HEX_HIGH != 0 { S_LOW_BS } else { S_NORMAL };
                                0
                            }
                        };
                        Self::set_top(st, g, out);
                    }
                    S_LOW_BS if b == b'\\' => {
                        g.ph = S_LOW_U;
                        Self::set_top(st, g, out);
                    }
                    S_LOW_U if b == b'u' => {
                        g.ph = S_LOW_HEX;
                        g.x = 0;
                        Self::set_top(st, g, out);
                    }
                    _ => {}
                }
            }
            Node::Num { integer } => {
                let digit = b.is_ascii_digit();
                let (ph, x) = match (f.ph, b) {
                    (N_START, b'-') => (N_MINUS, 0),
                    (N_START | N_MINUS, b'0') => (N_ZERO, 0),
                    (N_START | N_MINUS, _) if digit => (N_INT, 1),
                    (N_INT, _) if digit && f.x < INT_DIGITS => (N_INT, f.x + 1),
                    (N_ZERO | N_INT, b'.') if !integer => (N_DOT, 0),
                    (N_ZERO | N_INT | N_FRAC, b'e' | b'E') if !integer => (N_E, 0),
                    (N_DOT, _) if digit => (N_FRAC, 1),
                    (N_FRAC, _) if digit && f.x < FRAC_DIGITS => (N_FRAC, f.x + 1),
                    (N_E, b'+' | b'-') => (N_ESIGN, 0),
                    (N_E | N_ESIGN, _) if digit => (N_EXP, 1),
                    (N_EXP, _) if digit && f.x < EXP_DIGITS => (N_EXP, f.x + 1),
                    _ => return,
                };
                g.ph = ph;
                g.x = x;
                Self::set_top(st, g, out);
            }
            Node::Switch(_) => {
                let Some(next) = self.tries[f.x as usize].child(b) else {
                    return;
                };
                let t = &self.tries[next as usize];
                if !t.children.is_empty() || t.leaf == FINISH {
                    g.x = next;
                    Self::set_top(st.clone(), g, out);
                }
                if t.leaf != NO_LEAF && t.leaf != FINISH {
                    // the literal is complete: its continuation takes the frame's place
                    let mut p = st;
                    p.pop();
                    self.enter(p, t.leaf, out, fuel - 1, w);
                }
            }
            Node::Object(oi) => {
                let o = &self.objs[*oi as usize];
                // the key's opening quote, at position `pos`
                let key_from = |pos: u32| -> Option<Frame> {
                    let n = self.tries[o.keys as usize].child(b'"')?;
                    o.key_alive(n, pos).then(|| Frame {
                        y: pos,
                        ..Frame::new(f.node, O_KEY, n)
                    })
                };
                match f.ph {
                    O_OPEN if b == b'{' => {
                        g.ph = O_FIRST;
                        Self::set_top(st, g, out);
                    }
                    O_FIRST | O_COLON | O_PRE | O_AFTER | O_COMMA if is_ws(b) => {
                        Self::ws(st, f, out)
                    }
                    O_FIRST if b == b'}' && o.close_ok[0] => {
                        g.ph = DONE;
                        Self::set_top(st, g, out);
                    }
                    O_FIRST if b == b'"' => {
                        if let Some(g) = key_from(0) {
                            Self::set_top(st, g, out);
                        }
                    }
                    O_COMMA if b == b'"' => {
                        if let Some(g) = key_from(f.x) {
                            Self::set_top(st, g, out);
                        }
                    }
                    O_KEY => {
                        let Some(n) = self.tries[f.x as usize].child(b) else {
                            return;
                        };
                        if !o.key_alive(n, f.y) {
                            return;
                        }
                        // quoted keys are prefix-free: a leaf ends its key
                        let leaf = self.tries[n as usize].leaf;
                        g = if leaf != NO_LEAF && leaf != FINISH {
                            Frame::new(f.node, O_COLON, leaf)
                        } else {
                            Frame { y: f.y, ..Frame::new(f.node, O_KEY, n) }
                        };
                        Self::set_top(st, g, out);
                    }
                    O_COLON if b == b':' => {
                        g.ph = O_PRE;
                        g.w = 0;
                        Self::set_top(st, g, out);
                    }
                    O_PRE => {
                        let child = o.props[f.x as usize].value;
                        g.ph = O_VALUE;
                        g.w = 0;
                        self.enter_feed(st, g, child, b, out, fuel, w);
                    }
                    O_AFTER if b == b',' => {
                        let next = f.x + 1;
                        if (next as usize) < o.props.len() {
                            g = Frame::new(f.node, O_COMMA, next);
                            Self::set_top(st, g, out);
                        }
                    }
                    O_AFTER if b == b'}' => {
                        if o.close_ok.get(f.x as usize + 1).copied().unwrap_or(false) {
                            g.ph = DONE;
                            Self::set_top(st, g, out);
                        }
                    }
                    _ => {}
                }
            }
            Node::Map { key, value } => match f.ph {
                M_OPEN if b == b'{' => {
                    g.ph = M_FIRST;
                    Self::set_top(st, g, out);
                }
                M_FIRST | M_COLON | M_PRE | M_AFTER | M_COMMA if is_ws(b) => Self::ws(st, f, out),
                M_FIRST if b == b'}' => {
                    g.ph = DONE;
                    Self::set_top(st, g, out);
                }
                M_FIRST | M_COMMA if b == b'"' => {
                    g.ph = M_KEY;
                    g.w = 0;
                    self.enter_feed(st, g, *key, b, out, fuel, w);
                }
                M_COLON if b == b':' => {
                    g.ph = M_PRE;
                    g.w = 0;
                    Self::set_top(st, g, out);
                }
                M_PRE => {
                    g.ph = M_VALUE;
                    g.w = 0;
                    self.enter_feed(st, g, *value, b, out, fuel, w);
                }
                M_AFTER if b == b',' => {
                    g.ph = M_COMMA;
                    g.w = 0;
                    Self::set_top(st, g, out);
                }
                M_AFTER if b == b'}' => {
                    g.ph = DONE;
                    Self::set_top(st, g, out);
                }
                _ => {}
            },
            Node::Array { items, min, max } => match f.ph {
                A_OPEN if b == b'[' => {
                    g.ph = A_FIRST;
                    Self::set_top(st, g, out);
                }
                A_FIRST | A_AFTER | A_COMMA if is_ws(b) => Self::ws(st, f, out),
                A_FIRST if b == b']' && *min == 0 => {
                    g.ph = DONE;
                    Self::set_top(st, g, out);
                }
                A_FIRST | A_COMMA => {
                    if *max == 0 {
                        return;
                    }
                    g.ph = A_ITEM;
                    g.w = 0;
                    self.enter_feed(st, g, *items, b, out, fuel, w);
                }
                A_AFTER if b == b',' && f.x < *max => {
                    g.ph = A_COMMA;
                    g.w = 0;
                    Self::set_top(st, g, out);
                }
                A_AFTER if b == b']' && f.x >= *min => {
                    g.ph = DONE;
                    Self::set_top(st, g, out);
                }
                _ => {}
            },
            Node::Raw { term, forbid } => {
                if f.ph != R_RUN {
                    return;
                }
                let mut packed = 0u32;
                for (i, p) in forbid.iter().enumerate() {
                    let k = p.next((f.x >> (8 * (i + 1))) & 0xFF, b);
                    if k == p.len() {
                        return; // a forbidden string
                    }
                    packed |= k << (8 * (i + 1));
                }
                let k = term.next(f.x & 0xFF, b);
                if k == term.len() {
                    g.ph = DONE;
                    g.x = 0;
                } else {
                    g.x = packed | k;
                }
                Self::set_top(st, g, out);
            }
            Node::Text { open, call } => {
                if f.ph != T_TEXT {
                    return;
                }
                let k = open.next(f.x, b);
                if k == open.len() {
                    g.ph = T_CALL;
                    g.x = 0;
                    let mut st = st;
                    *st.last_mut().expect("a frame") = g;
                    self.enter(st, *call, out, fuel - 1, w);
                } else {
                    g.x = k;
                    Self::set_top(st, g, out);
                }
            }
            Node::Seq(_) | Node::Alt(_) => {}
        }
    }

    /// The bytes that may change the top frame of `st`, when every other byte leaves the whole
    /// stack as it is (a string body, raw text, free text with no partial tag); `None` when no
    /// such set is known.
    pub fn loop_bytes(&self, st: &Stack) -> Option<[u64; 4]> {
        let f = *st.last()?;
        let mut set = [0u64; 4];
        let mut add = |b: u8| set[(b >> 6) as usize] |= 1 << (b & 63);
        match &self.nodes[f.node as usize] {
            Node::Str { forbid } if f.ph == S_NORMAL && f.y == 0 => {
                add(b'"');
                add(b'\\');
                for b in 0..0x20u8 {
                    add(b);
                }
                for p in forbid {
                    add(p.bytes[0]);
                }
            }
            Node::Raw { term, forbid } if f.ph == R_RUN && f.x == 0 => {
                add(term.bytes[0]);
                for p in forbid {
                    add(p.bytes[0]);
                }
            }
            // a finishable frame would also offer the byte to its parent: only at the root
            Node::Text { open, .. } if f.ph == T_TEXT && f.x == 0 && st.len() == 1 => {
                add(open.bytes[0]);
            }
            _ => return None,
        }
        Some(set)
    }
}
