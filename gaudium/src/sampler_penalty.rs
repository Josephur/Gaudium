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
//! picked token may be emitted (never a banned one, never a padding id of the logits row), how
//! the count ledger changes when a token is emitted (`pen_count_add`, `pen_total_add`), and
//! whether the ledger holds exactly the generated tokens before the position being picked.
//!
//! An adjusted row never reaches the sampler with a value it can not order: a row holding a NaN
//! or +inf after the penalties (an overflow, or a non-finite logit from the card) is refused, and
//! so is a pick whose adjusted logit is -inf (every candidate banned or pushed to -inf). Either
//! fails the request instead of emitting the sampler's fallback token.
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
    /// Token ids the request may name, ban and emit: the verified `pen_emit_vocab` of the logits
    /// row and the tokenizer (a padded row's padding ids are not tokens).
    pub vocab: u64,
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
            vocab: 0,
            presence: 0.0,
            frequency: 0.0,
            repetition: 1.0,
            bias: TokenMask::default(),
        }
    }
}

impl Penalties {
    /// Penalties that change nothing, for a model whose request may emit `vocab` token ids.
    pub fn for_vocab(vocab: u64) -> Self {
        Self {
            vocab,
            ..Self::default()
        }
    }

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
    fn adjust(&self, l: f32, count: u64, bias: Option<Bias>) -> u16 {
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

/// The count ledger of one request: how often each generated token occurred so far. It changes
/// only through [`PenaltyState::commit`], by the verified `pen_count_add` / `pen_total_add`.
#[derive(Debug, Clone, Default)]
pub struct PenaltyState {
    counts: HashMap<u32, u64>,
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

    /// Records the token the request continues with: the picked one, or one teacher forcing
    /// put in its place. It goes through the same verified emission check as a picked token
    /// (never banned, never outside the request's vocabulary) before it is counted.
    pub fn commit(&mut self, p: &Penalties, tok: i32) -> Result<(), String> {
        let id = u64::try_from(tok).unwrap_or(u64::MAX);
        let class = u32::try_from(tok).map_or(PenBias::PbNone, |t| p.bias.class(t));
        if !verified::pen_emit_ok(id, p.vocab, class) {
            return Err(format!(
                "penalties: token {tok} may not be emitted (banned by logit_bias, or outside the {} token ids of the vocabulary)",
                p.vocab
            ));
        }
        // Only `tok`'s entry changes: `pen_count_add` leaves every other id's count as it was
        // (laws `pen_count_lower_unchanged`, `pen_count_higher_unchanged`).
        let key = id as u32;
        let next = verified::pen_count_add(self.count(key), id, id);
        self.counts.insert(key, next);
        self.total = verified::pen_total_add(self.total);
        Ok(())
    }

    /// Tokens recorded so far.
    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn count(&self, id: u32) -> u64 {
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

/// A BF16 value the sampler can not order: a NaN of either sign, or +inf.
fn unorderable(b: u16) -> bool {
    (b & 0x7F80) == 0x7F80 && b != BF16_NEG_INF
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
    pick_masked(p, st, logits, dev, seq_pos, None)
}

/// [`pick`] with the position's grammar mask (`crate::grammar`): a token the mask does not admit
/// is set to -inf before the sampler sees the row, and is classed as banned for the emission
/// check (`gram_token_class`), so it is never emitted; a mask that admits nothing refuses the
/// pick (`gram_can_pick`) instead of letting the sampler fall back to a token.
pub fn pick_masked(
    p: &Penalties,
    st: &PenaltyState,
    logits: &mut [u16],
    dev: &DevSampleParams,
    seq_pos: u32,
    mask: Option<&crate::grammar::constraint::Mask>,
) -> Result<i32, String> {
    // the request's ids, bounded again by the row actually read back
    let vocab = verified::pen_emit_vocab(logits.len() as u64, p.vocab);
    if !verified::pen_counts_match(st.total, seq_pos as u64, st.prompt_len) {
        return Err(format!(
            "internal: the penalty ledger holds {} tokens at sequence index {seq_pos} of a {}-token prompt",
            st.total, st.prompt_len
        ));
    }
    apply(p, st, logits);
    if let Some(m) = mask {
        if !verified::gram_can_pick(m.count()) {
            return Err(format!(
                "the response format admits no token at sequence index {seq_pos}: the reply can not continue in its format"
            ));
        }
        for (id, slot) in logits.iter_mut().enumerate() {
            if !m.has(id as u32) {
                *slot = BF16_NEG_INF;
            }
        }
    }
    if let Some(i) = logits.iter().position(|&b| unorderable(b)) {
        return Err(format!(
            "the penalised logits row holds a non-finite value at token {i} (sequence index {seq_pos}): the penalties overflowed or the row was not finite"
        ));
    }
    let tok = sample_row(logits, dev, 0, seq_pos, true).token;
    let class = match logits.get(tok as usize) {
        // a -inf pick is the sampler's fallback over a row with nothing left to pick
        Some(&BF16_NEG_INF) | None => PenBias::PbBan,
        Some(_) => verified::gram_token_class(
            mask.is_none_or(|m| m.has(tok as u32)),
            p.bias.class(tok as u32),
        ),
    };
    if !verified::pen_emit_ok(tok as u64, vocab, class) {
        return Err(format!(
            "the penalised sampler found no token to emit at sequence index {seq_pos} (token {tok} is banned, refused by the response format, at -inf, or outside the {vocab} token ids of the vocabulary)"
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
        let p = Penalties::for_vocab(8);
        assert!(!p.active());
        let mut st = PenaltyState::begin(3);
        st.commit(&p, 1).unwrap();
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
            ..Penalties::for_vocab(4)
        };
        assert!(p.active());
        let mut st = PenaltyState::begin(1);
        for t in [0, 0, 1, 2] {
            st.commit(&p, t).unwrap();
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
            ..Penalties::for_vocab(4)
        };
        // the ledger as an unbanned run left it (a banned token is never committed)
        let plain = Penalties::for_vocab(4);
        let mut st = PenaltyState::begin(0);
        st.commit(&plain, 0).unwrap();
        st.commit(&plain, 2).unwrap();
        assert!(st.clone().commit(&p, 2).is_err());
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
            ..Penalties::for_vocab(256)
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
            st.commit(&p, t).unwrap();
        }
    }

    #[test]
    fn every_token_banned_is_refused_not_emitted() {
        let p = Penalties {
            bias: TokenMask::new((0..256).map(|i| (i, Bias::Ban)).collect()),
            ..Penalties::for_vocab(256)
        };
        let st = PenaltyState::begin(1);
        let mut l = row(&[0.0; 256]);
        assert!(pick(&p, &st, &mut l, &greedy(), 1).is_err());
    }

    #[test]
    fn the_ledger_must_match_the_position() {
        let p = Penalties {
            presence: 1.0,
            ..Penalties::for_vocab(256)
        };
        let mut st = PenaltyState::begin(10);
        let mut l = row(&[0.0; 256]);
        assert!(pick(&p, &st, &mut l, &greedy(), 10).is_ok());
        assert!(pick(&p, &st, &mut l, &greedy(), 11).is_err()); // a token missing
        st.commit(&p, 3).unwrap();
        assert!(pick(&p, &st, &mut l, &greedy(), 10).is_err()); // its own token counted
        assert!(pick(&p, &st, &mut l, &greedy(), 11).is_ok());
        assert!(st.commit(&p, 256).is_err());
        assert!(st.commit(&p, -1).is_err());
    }

    #[test]
    fn a_repeated_greedy_maximum_gives_way_to_the_next_token() {
        let p = Penalties {
            frequency: 2.0,
            ..Penalties::for_vocab(256)
        };
        let mut vals = vec![-5.0f32; 256];
        vals[7] = 3.0;
        vals[8] = 2.0;
        let mut st = PenaltyState::begin(2);
        let mut out = Vec::new();
        for n in 2..6u32 {
            let mut l = row(&vals);
            let t = pick(&p, &st, &mut l, &greedy(), n).unwrap();
            st.commit(&p, t).unwrap();
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
    // Codex P2 on #132: a tiny repetition penalty divided a positive logit to +inf, and the
    // sampler then returned its fallback token (0) in place of the maximum (7).
    #[test]
    fn an_overflowing_repetition_penalty_is_refused_not_a_fallback_token() {
        let p = Penalties {
            repetition: 1e-40,
            ..Penalties::for_vocab(256)
        };
        let mut st = PenaltyState::begin(1);
        st.commit(&p, 7).unwrap();
        let mut vals = vec![1.0f32; 256];
        vals[7] = 4.0;
        let mut l = row(&vals);
        let r = pick(&p, &st, &mut l, &greedy(), 2);
        assert!(r.is_err(), "picked {r:?}");
        // (the request itself is refused at parsing: `penalty_fields` tests)
    }

    #[test]
    fn an_accepted_penalty_that_overflows_a_huge_logit_is_refused() {
        // inside the accepted range, but the logit itself is near the BF16 maximum
        let p = Penalties {
            repetition: 0.01,
            ..Penalties::for_vocab(256)
        };
        let mut st = PenaltyState::begin(1);
        st.commit(&p, 7).unwrap();
        let mut vals = vec![1.0f32; 256];
        vals[7] = 3.0e38;
        let mut l = row(&vals);
        assert!(pick(&p, &st, &mut l, &greedy(), 2).is_err());
        // pushed down to -inf instead, the token is out of the running and the maximum of the
        // rest is picked
        let p = Penalties {
            repetition: 100.0,
            ..Penalties::for_vocab(256)
        };
        vals[7] = -3.0e38;
        vals[9] = 2.0;
        let mut l = row(&vals);
        assert_eq!(pick(&p, &st, &mut l, &greedy(), 2).unwrap(), 9);
        assert_eq!(l[7], BF16_NEG_INF);
    }

    #[test]
    fn a_non_finite_row_is_refused() {
        let p = Penalties {
            presence: 0.5,
            ..Penalties::for_vocab(256)
        };
        let st = PenaltyState::begin(1);
        for bad in [0x7FC0u16, 0xFFC0, 0x7F80] {
            let mut l = row(&[0.0; 256]);
            l[100] = bad;
            assert!(pick(&p, &st, &mut l, &greedy(), 1).is_err(), "{bad:#x}");
        }
        // -inf (a ban, or a logit pushed down) is ordered: the maximum is still picked
        let mut l = row(&[0.0; 256]);
        l[100] = BF16_NEG_INF;
        l[3] = f32_to_bf16_rne(2.0);
        assert_eq!(pick(&p, &st, &mut l, &greedy(), 1).unwrap(), 3);
    }

    #[test]
    fn a_padding_id_of_the_row_is_never_emitted() {
        // 250 tokenizer tokens in a row padded to 256; the padding holds the maximum
        let p = Penalties {
            presence: 0.5,
            ..Penalties::for_vocab(250)
        };
        let st = PenaltyState::begin(1);
        let mut vals = vec![-1.0f32; 256];
        vals[252] = 10.0;
        let mut l = row(&vals);
        assert!(pick(&p, &st, &mut l, &greedy(), 1).is_err());
        let mut st = PenaltyState::begin(1);
        assert!(st.commit(&p, 252).is_err());
        assert!(st.commit(&p, 249).is_ok());
    }

    #[test]
    fn the_sparse_ledger_is_the_verified_transition_over_every_id() {
        // the ledger touches only the emitted token's entry; the verified transition, applied
        // to every id of a small vocabulary, gives the same counts and total
        const V: u64 = 16;
        let p = Penalties::for_vocab(V);
        let mut st = PenaltyState::begin(5);
        let mut dense = vec![0u64; V as usize];
        let mut total = 0u64;
        for tok in [3u64, 3, 0, 15, 7, 3, 15, 1] {
            st.commit(&p, tok as i32).unwrap();
            for (id, c) in dense.iter_mut().enumerate() {
                *c = verified::pen_count_add(*c, id as u64, tok);
            }
            total = verified::pen_total_add(total);
            for id in 0..V {
                assert_eq!(
                    st.count(id as u32),
                    dense[id as usize],
                    "id {id} after {tok}"
                );
            }
            assert_eq!(st.total(), total);
        }
        assert_eq!(dense.iter().sum::<u64>(), total);
        assert_eq!((dense[3], dense[15], dense[2]), (3, 2, 0));
    }

    /// The log-probability readout of a penalised token reads the row `pick` left behind: the
    /// penalised distribution the token was drawn from, not the raw logits.
    #[test]
    fn the_readout_of_a_penalised_pick_is_the_penalised_distribution() {
        use crate::logprobs::row_logprobs_u16;
        let p = Penalties {
            presence: 1.5,
            bias: TokenMask::new(vec![(5, Bias::Ban), (7, Bias::Add(0.75))]),
            ..Penalties::for_vocab(256)
        };
        let mut vals: Vec<f32> = (0..256).map(|i| -2.0 + (i % 13) as f32 * 0.125).collect();
        vals[5] = 10.0;
        vals[9] = 3.0;
        vals[7] = 2.75;
        let raw = row(&vals);
        let sampled = SampleParams {
            temperature: 0.8,
            top_p: 0.9,
            seed: 11,
            ..SampleParams::default()
        }
        .to_dev(256);
        for dev in [greedy(), sampled] {
            let mut st = PenaltyState::begin(4);
            for n in 4..24u32 {
                let mut l = raw.clone();
                let tok = pick(&p, &st, &mut l, &dev, n).unwrap();
                let pen = row_logprobs_u16(&l, &dev, tok, 5);
                assert!(pen.logprob.is_finite(), "n={n} tok={tok}");
                // the banned token is never an alternative with a finite logprob
                assert!(pen.top.iter().all(|&(id, lp)| id != 5 || !lp.is_finite()));
                if dev.i[3] & crate::sampler::F_GREEDY != 0 {
                    // greedy: the picked token is the penalised row's top-1
                    assert_eq!(pen.top[0].0, tok, "n={n}");
                }
                // the raw row would report another distribution (its top-1 is the banned 5)
                let rawlp = row_logprobs_u16(&raw, &dev, tok, 5);
                assert_ne!(rawlp.top[0].0, pen.top[0].0, "n={n}");
                st.commit(&p, tok).unwrap();
            }
        }
    }
}
