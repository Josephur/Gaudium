//! One constrained request's run: the reply phase, the token mask of every automaton state it
//! reaches (computed once, cached up to a byte cap), and the commit of each emitted token.
//!
//! What a request can make the host spend is bounded: the cache holds at most
//! [`MASK_CACHE_BYTES`] (verified `gram_cache_act` / `gram_cache_held`), one mask or one commit
//! does at most [`MAX_WORK`] frame operations and the whole request at most
//! [`MAX_REQUEST_WORK`] (verified `gram_budget_ok`); past either the request ends with an error.

use std::collections::HashMap;
use std::sync::Arc;

use super::vocab::{Kind, TokenTable, Trie};
use super::{Cursor, Grammar, Pat, Work, MAX_REQUEST_WORK, MAX_WORK};
use crate::verified::{self, GramCacheAct, GramPhase};

/// Bytes the masks one request keeps (and their keys) may hold.
pub const MASK_CACHE_BYTES: u64 = 64 << 20;

/// What a request's reply is held to: compiled once when the request is parsed.
pub struct Plan {
    pub grammar: Grammar,
    pub vocab: Arc<TokenTable>,
    /// The reasoning's closing tag, when the reply starts inside the model's reasoning.
    pub think_close: Option<Pat>,
    /// The cursor the answer starts from (after any forced prefix).
    pub start: Cursor,
    /// What the request asked for, for messages (`json_schema 'name'`, `strict tools`).
    pub label: String,
    /// Bytes the mask cache may hold ([`MASK_CACHE_BYTES`]), the frame operations one mask or
    /// one commit may take ([`MAX_WORK`]) and those the request may take in all
    /// ([`MAX_REQUEST_WORK`]).
    pub cache_cap: u64,
    pub work_cap: u64,
    pub request_work_cap: u64,
}

impl Plan {
    pub fn new(grammar: Grammar, vocab: Arc<TokenTable>, think_close: Option<Pat>, start: Cursor, label: String) -> Self {
        Self {
            grammar,
            vocab,
            think_close,
            start,
            label,
            cache_cap: MASK_CACHE_BYTES,
            work_cap: MAX_WORK,
            request_work_cap: MAX_REQUEST_WORK,
        }
    }
}

impl std::fmt::Debug for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Plan({}, {} grammar nodes)", self.label, self.grammar.nodes.len())
    }
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constraint({}, {:?})", self.plan.label, self.phase)
    }
}

/// The token ids a position admits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    bits: Vec<u64>,
    count: u64,
}

impl Mask {
    fn new(len: usize) -> Self {
        Self {
            bits: vec![0; len.div_ceil(64)],
            count: 0,
        }
    }

    fn set(&mut self, id: u32) {
        let (w, b) = ((id >> 6) as usize, id & 63);
        if self.bits[w] & (1 << b) == 0 {
            self.bits[w] |= 1 << b;
            self.count += 1;
        }
    }

    pub fn has(&self, id: u32) -> bool {
        self.bits
            .get((id >> 6) as usize)
            .is_some_and(|w| w & (1 << (id & 63)) != 0)
    }

    /// Admitted ids.
    pub fn count(&self) -> u64 {
        self.count
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Think(u32),
    Answer(Cursor),
}

impl Key {
    /// Bytes a cache entry of this key and a mask of `mask_words` words holds.
    fn entry_bytes(&self, mask_words: usize) -> u64 {
        let key = match self {
            Key::Think(_) => 0,
            Key::Answer(c) => c
                .stacks
                .iter()
                .map(|s| std::mem::size_of::<super::Stack>() + s.len() * std::mem::size_of::<super::Frame>())
                .sum::<usize>(),
        };
        (key + 8 * mask_words + 2 * std::mem::size_of::<(Key, Arc<Mask>)>()) as u64
    }
}

/// Counters of a run, for the request's report.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stats {
    pub masks_built: u64,
    pub mask_ms: f64,
    pub cache_hits: u64,
    /// Times the cache was full and was emptied.
    pub cache_clears: u64,
    /// The most frame operations one mask took.
    pub mask_work_max: u64,
}

/// A request's constraint while it decodes.
pub struct Constraint {
    plan: Arc<Plan>,
    phase: GramPhase,
    /// Matched length of the reasoning's closing tag.
    kmp: u32,
    cursor: Cursor,
    cache: HashMap<Key, Arc<Mask>>,
    /// Bytes the cache's entries hold (`gram_cache_held`).
    cache_bytes: u64,
    /// The request's work so far, against `request_work_cap` (`gram_budget_ok`, updated only
    /// through `gram_budget_charge`; past the cap once a charge was refused).
    spent: u64,
    pub stats: Stats,
}

impl Constraint {
    pub fn begin(plan: Arc<Plan>) -> Self {
        let phase = if plan.think_close.is_some() {
            GramPhase::GpThink
        } else {
            GramPhase::GpAnswer
        };
        Self {
            cursor: plan.start.clone(),
            plan,
            phase,
            kmp: 0,
            cache: HashMap::new(),
            cache_bytes: 0,
            spent: 0,
            stats: Stats::default(),
        }
    }

    /// Charges a call's work to the request; `Err` once the request's cap is passed.
    fn charge(&mut self, w: &Work) -> Result<(), String> {
        let cap = self.plan.request_work_cap.min(u64::MAX - 1);
        let ok = verified::gram_budget_ok(self.spent, w.used(), cap);
        self.spent = verified::gram_budget_charge(self.spent, w.used(), cap);
        if ok {
            return Ok(());
        }
        Err(format!(
            "{}: the reply's format took more than {} automaton steps in all",
            self.plan.label, self.plan.request_work_cap
        ))
    }

    fn work(&self) -> Work {
        Work::new(self.plan.work_cap)
    }

    fn out_of_work(&self) -> String {
        format!(
            "{}: the reply's format took more than {} automaton steps for one token",
            self.plan.label, self.plan.work_cap
        )
    }

    /// Bytes the mask cache holds now.
    pub fn cache_bytes(&self) -> u64 {
        self.cache_bytes
    }

    pub fn phase(&self) -> GramPhase {
        self.phase
    }

    fn accepting(&self, w: &mut Work) -> Result<bool, String> {
        let complete = self.plan.grammar.complete_in(&self.cursor, w);
        if w.exhausted() {
            return Err(self.out_of_work());
        }
        Ok(verified::gram_accepting(self.phase, complete))
    }

    fn key(&self) -> Key {
        match self.phase {
            GramPhase::GpThink => Key::Think(self.kmp),
            GramPhase::GpAnswer => Key::Answer(self.cursor.clone()),
        }
    }

    /// The ids the next position admits (`Err`: the mask took more than the work cap).
    pub fn mask(&mut self) -> Result<Arc<Mask>, String> {
        let key = self.key();
        if let Some(m) = self.cache.get(&key) {
            self.stats.cache_hits += 1;
            return Ok(Arc::clone(m));
        }
        let t0 = std::time::Instant::now();
        let mut w = self.work();
        let m = match self.phase {
            GramPhase::GpThink => self.think_mask(&mut w),
            GramPhase::GpAnswer => self.answer_mask(&mut w),
        };
        if w.exhausted() {
            return Err(self.out_of_work());
        }
        let m = Arc::new(m?);
        self.charge(&w)?;
        self.stats.masks_built += 1;
        self.stats.mask_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.stats.mask_work_max = self.stats.mask_work_max.max(w.used());
        let add = key.entry_bytes(m.bits.len());
        let act = verified::gram_cache_act(self.cache_bytes, add, self.plan.cache_cap);
        match act {
            GramCacheAct::GcKeep => {
                self.cache.insert(key, Arc::clone(&m));
            }
            GramCacheAct::GcClear => {
                self.cache.clear();
                self.stats.cache_clears += 1;
            }
        }
        self.cache_bytes = verified::gram_cache_held(act, self.cache_bytes, add);
        Ok(m)
    }

    /// In the reasoning every text and control token is free, except one whose bytes close the
    /// reasoning and continue with bytes the answer refuses; an end token waits for the answer.
    fn think_mask(&self, w: &mut Work) -> Result<Mask, String> {
        let v = &*self.plan.vocab;
        let close = self.plan.think_close.as_ref().expect("a reasoning phase");
        let accepting = self.accepting(w)?;
        let control = verified::gram_control_free(self.phase);
        let mut m = Mask::new(v.len());
        for id in 0..v.len() as u32 {
            let ok = match v.kind(id) {
                Kind::End => verified::gram_admit(true, accepting, false),
                Kind::Control => verified::gram_admit(false, accepting, control),
                Kind::Text => verified::gram_admit(false, accepting, self.think_steps(close, id, w)),
            };
            if ok {
                m.set(id);
            }
            if w.exhausted() {
                break;
            }
        }
        Ok(m)
    }

    /// Whether text token `id` keeps the reply alive from the reasoning phase.
    fn think_steps(&self, close: &Pat, id: u32, w: &mut Work) -> bool {
        let bytes = self.plan.vocab.bytes(id);
        let mut k = self.kmp;
        for (i, &b) in bytes.iter().enumerate() {
            k = close.next(k, b);
            if k == close.len() {
                return self
                    .plan
                    .grammar
                    .step_bytes_in(&self.cursor, &bytes[i + 1..], w)
                    .is_some();
            }
        }
        true
    }

    fn answer_mask(&self, w: &mut Work) -> Result<Mask, String> {
        let v = &*self.plan.vocab;
        let g = &self.plan.grammar;
        let accepting = self.accepting(w)?;
        let mut m = Mask::new(v.len());
        for id in v.ids_of(Kind::End) {
            if verified::gram_admit(true, accepting, false) {
                m.set(id);
            }
        }
        // a control token adds no text: free only where `gram_control_free` says so
        let control = verified::gram_control_free(self.phase);
        for id in v.ids_of(Kind::Control) {
            if verified::gram_admit(false, accepting, control) {
                m.set(id);
            }
        }
        let loop_set = self
            .cursor
            .stacks
            .iter()
            .map(|s| g.loop_bytes(s))
            .try_fold([0u64; 4], |acc, s| {
                s.map(|s| [acc[0] | s[0], acc[1] | s[1], acc[2] | s[2], acc[3] | s[3]])
            });
        match loop_set {
            Some(set) => {
                // every token without a byte of the set leaves the cursor as it is
                let split = v.split(set);
                for &id in &split.plain {
                    if verified::gram_admit(false, accepting, true) {
                        m.set(id);
                    }
                }
                Self::walk(g, &split.rest, 0, &self.cursor, accepting, &mut m, w);
            }
            None => Self::walk(g, v.trie(), 0, &self.cursor, accepting, &mut m, w),
        }
        Ok(m)
    }

    fn walk(g: &Grammar, t: &Trie, n: u32, c: &Cursor, accepting: bool, m: &mut Mask, w: &mut Work) {
        for &(b, child) in t.children(n) {
            let Some(next) = g.step_in(c, b, w) else {
                if w.exhausted() {
                    return;
                }
                continue;
            };
            for &id in t.tokens(child) {
                if verified::gram_admit(false, accepting, true) {
                    m.set(id);
                }
            }
            Self::walk(g, t, child, &next, accepting, m, w);
        }
    }

    /// Records the token the request continues with (picked, or put in place by teacher
    /// forcing): it must be one the position admits.
    pub fn commit(&mut self, tok: i32) -> Result<(), String> {
        let Ok(id) = u32::try_from(tok) else {
            return Err(format!("{}: token {tok} is not a token id", self.plan.label));
        };
        let v = Arc::clone(&self.plan.vocab);
        let mut w = self.work();
        let accepting = self.accepting(&mut w)?;
        let refused = || {
            format!(
                "{}: token {tok} does not continue the reply in its format",
                self.plan.label
            )
        };
        match v.kind(id) {
            Kind::End => {
                if !verified::gram_admit(true, accepting, false) {
                    return Err(refused());
                }
            }
            Kind::Control => {
                if !verified::gram_admit(false, accepting, verified::gram_control_free(self.phase)) {
                    return Err(refused());
                }
            }
            Kind::Text => match self.phase {
                GramPhase::GpThink => {
                    let close = self.plan.think_close.clone().expect("a reasoning phase");
                    let bytes = v.bytes(id);
                    let mut closed_at = None;
                    for (i, &b) in bytes.iter().enumerate() {
                        self.kmp = close.next(self.kmp, b);
                        if self.kmp == close.len() {
                            closed_at = Some(i + 1);
                            break;
                        }
                    }
                    self.phase = verified::gram_phase_step(self.phase, closed_at.is_some());
                    if let Some(i) = closed_at {
                        self.kmp = 0;
                        let next = self.plan.grammar.step_bytes_in(&self.cursor, &bytes[i..], &mut w);
                        if w.exhausted() {
                            return Err(self.out_of_work());
                        }
                        if !verified::gram_admit(false, accepting, next.is_some()) {
                            return Err(refused());
                        }
                        self.cursor = next.expect("admitted");
                    }
                }
                GramPhase::GpAnswer => {
                    let next = self.plan.grammar.step_bytes_in(&self.cursor, v.bytes(id), &mut w);
                    if w.exhausted() {
                        return Err(self.out_of_work());
                    }
                    if !verified::gram_admit(false, accepting, next.is_some()) {
                        return Err(refused());
                    }
                    self.cursor = next.expect("admitted");
                }
            },
        }
        self.charge(&w)
    }

    /// The reply as it stands could end here (`false` also when the check ran out of work).
    pub fn complete(&self) -> bool {
        self.accepting(&mut self.work()).unwrap_or(false)
    }

    pub fn label(&self) -> &str {
        &self.plan.label
    }
}
