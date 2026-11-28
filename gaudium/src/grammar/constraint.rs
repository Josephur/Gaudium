//! One constrained request's run: the reply phase, the token mask of every automaton state it
//! reaches (computed once, cached), and the commit of each emitted token.

use std::collections::HashMap;
use std::sync::Arc;

use super::vocab::{Kind, TokenTable, Trie};
use super::{Cursor, Grammar, Pat};
use crate::verified::{self, GramPhase};

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

/// Counters of a run, for the request's report.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stats {
    pub masks_built: u64,
    pub mask_ms: f64,
    pub cache_hits: u64,
}

/// A request's constraint while it decodes.
pub struct Constraint {
    plan: Arc<Plan>,
    phase: GramPhase,
    /// Matched length of the reasoning's closing tag.
    kmp: u32,
    cursor: Cursor,
    cache: HashMap<Key, Arc<Mask>>,
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
            stats: Stats::default(),
        }
    }

    pub fn phase(&self) -> GramPhase {
        self.phase
    }

    fn accepting(&self) -> bool {
        let complete =
            self.phase == GramPhase::GpAnswer && self.plan.grammar.complete(&self.cursor);
        verified::gram_accepting(self.phase, complete)
    }

    fn key(&self) -> Key {
        match self.phase {
            GramPhase::GpThink => Key::Think(self.kmp),
            GramPhase::GpAnswer => Key::Answer(self.cursor.clone()),
        }
    }

    /// The ids the next position admits.
    pub fn mask(&mut self) -> Arc<Mask> {
        let key = self.key();
        if let Some(m) = self.cache.get(&key) {
            self.stats.cache_hits += 1;
            return Arc::clone(m);
        }
        let t0 = std::time::Instant::now();
        let m = Arc::new(match self.phase {
            GramPhase::GpThink => self.think_mask(),
            GramPhase::GpAnswer => self.answer_mask(),
        });
        self.stats.masks_built += 1;
        self.stats.mask_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.cache.insert(key, Arc::clone(&m));
        m
    }

    /// In the reasoning every text and control token is free, except one whose bytes close the
    /// reasoning and continue with bytes the answer refuses; an end token waits for the answer.
    fn think_mask(&self) -> Mask {
        let v = &*self.plan.vocab;
        let close = self.plan.think_close.as_ref().expect("a reasoning phase");
        let accepting = self.accepting();
        let mut m = Mask::new(v.len());
        for id in 0..v.len() as u32 {
            let ok = match v.kind(id) {
                Kind::End => verified::gram_admit(true, accepting, false),
                Kind::Control => verified::gram_admit(false, accepting, true),
                Kind::Text => verified::gram_admit(false, accepting, self.think_steps(close, id)),
            };
            if ok {
                m.set(id);
            }
        }
        m
    }

    /// Whether text token `id` keeps the reply alive from the reasoning phase.
    fn think_steps(&self, close: &Pat, id: u32) -> bool {
        let bytes = self.plan.vocab.bytes(id);
        let mut k = self.kmp;
        for (i, &b) in bytes.iter().enumerate() {
            k = close.next(k, b);
            if k == close.len() {
                return self
                    .plan
                    .grammar
                    .step_bytes(&self.cursor, &bytes[i + 1..])
                    .is_some();
            }
        }
        true
    }

    fn answer_mask(&self) -> Mask {
        let v = &*self.plan.vocab;
        let g = &self.plan.grammar;
        let accepting = self.accepting();
        let mut m = Mask::new(v.len());
        for id in v.ids_of(Kind::End) {
            if verified::gram_admit(true, accepting, false) {
                m.set(id);
            }
        }
        // (control tokens add no text: `gram_admit(false, _, false)`, never admitted)
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
                Self::walk(g, &split.rest, 0, &self.cursor, accepting, &mut m);
            }
            None => Self::walk(g, v.trie(), 0, &self.cursor, accepting, &mut m),
        }
        m
    }

    fn walk(g: &Grammar, t: &Trie, n: u32, c: &Cursor, accepting: bool, m: &mut Mask) {
        for &(b, child) in t.children(n) {
            let Some(next) = g.step(c, b) else {
                continue;
            };
            for &id in t.tokens(child) {
                if verified::gram_admit(false, accepting, true) {
                    m.set(id);
                }
            }
            Self::walk(g, t, child, &next, accepting, m);
        }
    }

    /// Records the token the request continues with (picked, or put in place by teacher
    /// forcing): it must be one the position admits.
    pub fn commit(&mut self, tok: i32) -> Result<(), String> {
        let Ok(id) = u32::try_from(tok) else {
            return Err(format!("{}: token {tok} is not a token id", self.plan.label));
        };
        let v = Arc::clone(&self.plan.vocab);
        let accepting = self.accepting();
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
                let free = self.phase == GramPhase::GpThink;
                if !verified::gram_admit(false, accepting, free) {
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
                        let next = self.plan.grammar.step_bytes(&self.cursor, &bytes[i..]);
                        if !verified::gram_admit(false, accepting, next.is_some()) {
                            return Err(refused());
                        }
                        self.cursor = next.expect("admitted");
                    }
                }
                GramPhase::GpAnswer => {
                    let next = self.plan.grammar.step_bytes(&self.cursor, v.bytes(id));
                    if !verified::gram_admit(false, accepting, next.is_some()) {
                        return Err(refused());
                    }
                    self.cursor = next.expect("admitted");
                }
            },
        }
        Ok(())
    }

    /// The reply as it stands could end here.
    pub fn complete(&self) -> bool {
        self.accepting()
    }

    pub fn label(&self) -> &str {
        &self.plan.label
    }
}
