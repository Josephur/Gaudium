//! Per-request sampler penalties and token biases, applied on the host to a logits row the card
//! read out: `presence_penalty`, `frequency_penalty`, `repetition_penalty` and `logit_bias`.
//!
//! A penalised position's logits are a function of the row's own BF16 logits and of the counts
//! of the tokens generated before that position, nothing else. The token is then picked from the
//! adjusted row by the host twin of the on-card sampler (`crate::sampler::sample_row`: greedy, or
//! Gumbel-max with the position's own noise), so a penalised request is deterministic for a given
//! seed and its stream is its own reference.
//!
//! The decisions around the arithmetic are verified (`verified/sampler_penalty.bend`): which ids
//! a request may name and a count may index, whether its bans leave a token to emit, which decode
//! paths may take it (only one that applies the penalties at every position it picks), whether a
//! picked token may be emitted (never a banned one), and whether the count ledger holds exactly
//! the generated tokens before the position being picked.
//!
//! What is counted: every generated token, the reasoning span included (the engine does not see
//! where reasoning ends; the server splits it afterwards). The prompt is not counted.
//!
//! [`TokenMask`] is the per-request additive / ban mask over token ids. It is kept separate from
//! the count-driven penalties so that a later per-position mask (constrained decoding) can reuse
//! the same application and the same verified emission check.

use std::collections::HashMap;

use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};
use crate::sampler::{sample_row, DevSampleParams};
use crate::verified::{self, PenBias};

/// BF16 negative infinity: a banned token's logit (below every threshold the sampler tests).
pub const BF16_NEG_INF: u16 = 0xFF80;

/// What a mask does to one token's logit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bias {
    /// Added to the logit (after the count penalties).
    Add(f32),
    /// The token is never emitted.
    Ban,
}

/// A sparse mask over token ids: sorted by id, one entry per id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TokenMask {
    entries: Vec<(u32, Bias)>,
}

impl TokenMask {
    /// Builds a mask; a later entry for the same id replaces an earlier one.
    pub fn new(mut entries: Vec<(u32, Bias)>) -> Self {
        // stable sort, then keep the last entry of each id
        entries.sort_by_key(|e| e.0);
        let mut out: Vec<(u32, Bias)> = Vec::with_capacity(entries.len());
        for e in entries {
            match out.last_mut() {
                Some(last) if last.0 == e.0 => *last = e,
                _ => out.push(e),
            }
        }
        Self { entries: out }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, id: u32) -> Option<Bias> {
        self.entries
            .binary_search_by_key(&id, |e| e.0)
            .ok()
            .map(|i| self.entries[i].1)
    }

    /// The verified core's view of one token's entry.
    pub fn class(&self, id: u32) -> PenBias {
        match self.get(id) {
            None => PenBias::PbNone,
            Some(Bias::Add(_)) => PenBias::PbAdd,
            Some(Bias::Ban) => PenBias::PbBan,
        }
    }

    /// Distinct banned ids.
    pub fn banned(&self) -> u64 {
        self.entries
            .iter()
            .filter(|e| matches!(e.1, Bias::Ban))
            .count() as u64
    }

    pub fn entries(&self) -> &[(u32, Bias)] {
        &self.entries
    }
}

/// A request's penalties. The default changes nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct Penalties {
    /// Subtracted once from the logit of every token generated at least once.
    pub presence: f32,
    /// Subtracted per occurrence from the logit of every generated token.
    pub frequency: f32,
    /// A generated token's positive logit is divided by it, a negative one multiplied (1 = off).
    pub repetition: f32,
    pub bias: TokenMask,
}

impl Default for Penalties {
    fn default() -> Self {
        Self {
            presence: 0.0,
            frequency: 0.0,
            repetition: 1.0,
            bias: TokenMask::default(),
        }
    }
}

impl Penalties {
    /// The request carries something that changes its logits.
    pub fn active(&self) -> bool {
        self.presence != 0.0
            || self.frequency != 0.0
            || self.repetition != 1.0
            || !self.bias.is_empty()
    }

    fn counted(&self) -> bool {
        self.presence != 0.0 || self.frequency != 0.0 || self.repetition != 1.0
    }

    /// The adjusted logit of a token generated `count` times with mask entry `bias`.
    fn adjust(&self, l: f32, count: u32, bias: Option<Bias>) -> u16 {
        if matches!(bias, Some(Bias::Ban)) {
            return BF16_NEG_INF;
        }
        let mut x = l;
        if count > 0 {
            if self.repetition != 1.0 {
                x = if x > 0.0 {
                    x / self.repetition
                } else {
                    x * self.repetition
                };
            }
            x -= self.frequency * count as f32 + self.presence;
        }
        if let Some(Bias::Add(b)) = bias {
            x += b;
        }
        f32_to_bf16_rne(x)
    }
}

/// The count ledger of one request: how often each generated token occurred so far.
#[derive(Debug, Clone, Default)]
pub struct PenaltyState {
    counts: HashMap<u32, u32>,
    total: u64,
    prompt_len: u64,
}

impl PenaltyState {
    /// A request's ledger before its first generated token.
    pub fn begin(prompt_len: usize) -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
            prompt_len: prompt_len as u64,
        }
    }

    /// Records a generated token (the one the request continues with).
    pub fn commit(&mut self, tok: i32, vocab: u64) -> Result<(), String> {
        let id = u64::try_from(tok).unwrap_or(u64::MAX);
        if !verified::pen_token_in_vocab(id, vocab) {
            return Err(format!(
                "penalties: generated token {tok} is outside the vocabulary of {vocab}"
            ));
        }
        *self.counts.entry(id as u32).or_insert(0) += 1;
        self.total += 1;
        Ok(())
    }

    /// Tokens recorded so far.
    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn count(&self, id: u32) -> u32 {
        self.counts.get(&id).copied().unwrap_or(0)
    }
}

/// Applies `p` to one BF16 logits row in place (the row of the position whose ledger is `st`).
/// Only the tokens generated so far and the mask's tokens change; each is rounded once.
pub fn apply(p: &Penalties, st: &PenaltyState, logits: &mut [u16]) {
    if p.counted() {
        for (&id, &c) in &st.counts {
            // ids were checked against the vocabulary at commit; the row is the vocabulary
            if let Some(slot) = logits.get_mut(id as usize) {
                *slot = p.adjust(bf16_to_f32(*slot), c, p.bias.get(id));
            }
        }
    }
    for &(id, b) in p.bias.entries() {
        if p.counted() && st.count(id) > 0 {
            continue; // done above, with its count
        }
        if let Some(slot) = logits.get_mut(id as usize) {
            *slot = p.adjust(bf16_to_f32(*slot), 0, Some(b));
        }
    }
}

/// Picks the token at sequence index `seq_pos` from a penalised row: the penalties of `p` with
/// the ledger `st`, then the sampler's host twin with the request's device parameters `dev`.
pub fn pick(
    p: &Penalties,
    st: &PenaltyState,
    logits: &mut [u16],
    dev: &DevSampleParams,
    seq_pos: u32,
) -> Result<i32, String> {
    let vocab = logits.len() as u64;
    if !verified::pen_counts_match(st.total, seq_pos as u64, st.prompt_len) {
        return Err(format!(
            "internal: the penalty ledger holds {} tokens at sequence index {seq_pos} of a {}-token prompt",
            st.total, st.prompt_len
        ));
    }
    apply(p, st, logits);
    let tok = sample_row(logits, dev, 0, seq_pos, true).token;
    if !verified::pen_emit_ok(tok as u64, vocab, p.bias.class(tok as u32)) {
        return Err(format!(
            "the penalised sampler found no token to emit at sequence index {seq_pos} (token {tok} is banned)"
        ));
    }
    Ok(tok as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SampleParams;

    fn row(vals: &[f32]) -> Vec<u16> {
        vals.iter().map(|&v| f32_to_bf16_rne(v)).collect()
    }

    fn greedy() -> DevSampleParams {
        SampleParams::default().to_dev(256)
    }

    #[test]
    fn default_penalties_are_inactive_and_change_nothing() {
        let p = Penalties::default();
        assert!(!p.active());
        let mut st = PenaltyState::begin(3);
        st.commit(1, 8).unwrap();
        let mut l = row(&[1.0, 2.0, -3.0, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let before = l.clone();
        apply(&p, &st, &mut l);
        assert_eq!(l, before);
    }

    #[test]
    fn presence_frequency_and_repetition_follow_the_counts() {
        let p = Penalties {
            presence: 0.5,
            frequency: 0.25,
            repetition: 2.0,
            ..Penalties::default()
        };
        assert!(p.active());
        let mut st = PenaltyState::begin(1);
        for t in [0, 0, 1, 2] {
            st.commit(t, 4).unwrap();
        }
        let mut l = row(&[4.0, -2.0, 1.0, 3.0]);
        apply(&p, &st, &mut l);
        let f: Vec<f32> = l.iter().map(|&b| bf16_to_f32(b)).collect();
        // token 0: 4/2 - 2*0.25 - 0.5 = 1.0; token 1: -2*2 - 0.25 - 0.5 = -4.75
        // token 2: 1/2 - 0.25 - 0.5 = -0.25; token 3 never generated: unchanged
        assert_eq!(f, vec![1.0, -4.75, -0.25, 3.0]);
    }

    #[test]
    fn bias_adds_and_bans_and_rounds_once_with_the_counts() {
        let p = Penalties {
            frequency: 1.0,
            bias: TokenMask::new(vec![
                (0, Bias::Add(0.5)),
                (2, Bias::Ban),
                (3, Bias::Add(-1.0)),
            ]),
            ..Penalties::default()
        };
        let mut st = PenaltyState::begin(0);
        st.commit(0, 4).unwrap();
        st.commit(2, 4).unwrap();
        let mut l = row(&[2.0, 1.0, 9.0, 1.5]);
        apply(&p, &st, &mut l);
        assert_eq!(bf16_to_f32(l[0]), 1.5); // 2 - 1 + 0.5
        assert_eq!(bf16_to_f32(l[1]), 1.0);
        assert_eq!(l[2], BF16_NEG_INF);
        assert_eq!(bf16_to_f32(l[3]), 0.5);
    }

    #[test]
    fn a_banned_maximum_is_never_picked() {
        let p = Penalties {
            bias: TokenMask::new(vec![(5, Bias::Ban)]),
            ..Penalties::default()
        };
        let st = PenaltyState::begin(4);
        let mut vals = vec![-1.0f32; 256];
        vals[5] = 10.0;
        vals[9] = 3.0;
        let mut l = row(&vals);
        assert_eq!(pick(&p, &st, &mut l, &greedy(), 4).unwrap(), 9);
        // sampled, many positions: never the banned token
        let dev = SampleParams {
            temperature: 1.0,
            seed: 7,
            ..SampleParams::default()
        }
        .to_dev(256);
        let mut st = PenaltyState::begin(4);
        for n in 4..64u32 {
            let mut l = row(&vals);
            let t = pick(&p, &st, &mut l, &dev, n).unwrap();
            assert_ne!(t, 5);
            st.commit(t, 256).unwrap();
        }
    }

    #[test]
    fn every_token_banned_is_refused_not_emitted() {
        let p = Penalties {
            bias: TokenMask::new((0..256).map(|i| (i, Bias::Ban)).collect()),
            ..Penalties::default()
        };
        let st = PenaltyState::begin(1);
        let mut l = row(&[0.0; 256]);
        assert!(pick(&p, &st, &mut l, &greedy(), 1).is_err());
    }

    #[test]
    fn the_ledger_must_match_the_position() {
        let p = Penalties {
            presence: 1.0,
            ..Penalties::default()
        };
        let mut st = PenaltyState::begin(10);
        let mut l = row(&[0.0; 256]);
        assert!(pick(&p, &st, &mut l, &greedy(), 10).is_ok());
        assert!(pick(&p, &st, &mut l, &greedy(), 11).is_err()); // a token missing
        st.commit(3, 256).unwrap();
        assert!(pick(&p, &st, &mut l, &greedy(), 10).is_err()); // its own token counted
        assert!(pick(&p, &st, &mut l, &greedy(), 11).is_ok());
        assert!(st.commit(256, 256).is_err());
        assert!(st.commit(-1, 256).is_err());
    }

    #[test]
    fn a_repeated_greedy_maximum_gives_way_to_the_next_token() {
        let p = Penalties {
            frequency: 2.0,
            ..Penalties::default()
        };
        let mut vals = vec![-5.0f32; 256];
        vals[7] = 3.0;
        vals[8] = 2.0;
        let mut st = PenaltyState::begin(2);
        let mut out = Vec::new();
        for n in 2..6u32 {
            let mut l = row(&vals);
            let t = pick(&p, &st, &mut l, &greedy(), n).unwrap();
            st.commit(t, 256).unwrap();
            out.push(t);
        }
        // 7 (3.0), then 8 (2.0 > 1.0), then 7 (1.0 > 0.0) ... deterministic
        assert_eq!(out, vec![7, 8, 7, 8]);
    }

    #[test]
    fn the_mask_keeps_the_last_entry_of_an_id() {
        let m = TokenMask::new(vec![(4, Bias::Add(1.0)), (2, Bias::Ban), (4, Bias::Ban)]);
        assert_eq!(m.entries(), &[(2, Bias::Ban), (4, Bias::Ban)]);
        assert_eq!(m.banned(), 2);
        assert_eq!(m.class(3), PenBias::PbNone);
        assert_eq!(m.class(4), PenBias::PbBan);
    }
}
