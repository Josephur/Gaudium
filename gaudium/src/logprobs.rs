//! Per-token log-probabilities (`logprobs` / `top_logprobs`) from a logits row read back from the
//! card.
//!
//! The numbers are those of the distribution the emitted token was chosen from:
//!
//! - **Sampled rows** (`temperature > 0`): the on-card sampler draws by Gumbel-max over a support
//!   (every token whose BF16 logit is at least the support threshold) with weights
//!   `exp((l - max) / T)`. The threshold is the one the sampler kernels search (top-k, top-p,
//!   min-p and the fixed `20.5 T` logit window), recomputed with the bit-exact CPU twin
//!   ([`crate::sampler::row_support`]). A token's log-probability is
//!   `(l - max) / T - ln sum_support exp((l' - max) / T)`; tokens outside the support have none.
//! - **Greedy rows** (`temperature <= 0`): the token is the row's argmax; the numbers are the
//!   model's own distribution, the softmax of the whole row at temperature 1 with no truncation.
//!
//! The row is BF16, so every logit is one of 65 536 values. The sum is taken over a histogram of
//! the row's BF16 values, each distinct value's `exp` once, in `f64`: one pass over the row and no
//! rounding of the sum beyond `f64`'s. Two runs whose rows are bit-identical report bit-identical
//! numbers; rows that differ report numbers that differ by the change of the logits (scaled by
//! `1 / T`) and nothing more.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::sampler::{key_of_bits, row_support, val_of_key, DevSampleParams, F_GREEDY};
use crate::verified;

/// Most alternatives one token may report (the request range of `top_logprobs`).
pub fn max_top() -> u64 {
    verified::lp_top_cap()
}

/// Is a request's `top_logprobs` accepted?
pub fn top_ok(n: u64) -> bool {
    verified::lp_top_ok(n)
}

/// One emitted token's log-probability and its most likely alternatives (highest logit first,
/// the smaller token id first on equal logits).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token: i32,
    /// `f64::NEG_INFINITY`: the token is outside the distribution it was drawn from (never for a
    /// token the engine emitted itself).
    pub logprob: f64,
    pub top: Vec<(i32, f64)>,
}

/// A request's log-probability readout: how many alternatives each token reports, and the queue
/// the engine pushes one entry into for every emitted token, before it hands the token over.
#[derive(Clone)]
pub struct LogprobTap {
    pub top: u64,
    pub queue: Arc<Mutex<VecDeque<TokenLogprob>>>,
}

impl std::fmt::Debug for LogprobTap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogprobTap")
            .field("top", &self.top)
            .finish()
    }
}

impl LogprobTap {
    pub fn new(top: u64) -> Self {
        Self {
            top,
            queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn push(&self, e: TokenLogprob) {
        self.queue.lock().unwrap().push_back(e);
    }

    /// The entry of the oldest token not taken yet.
    pub fn pop(&self) -> Option<TokenLogprob> {
        self.queue.lock().unwrap().pop_front()
    }
}

const NBINS: usize = 1 << 16;

thread_local! {
    static COUNTS: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(vec![0u32; NBINS]);
}

fn is_nan_bits(b: u16) -> bool {
    (b & 0x7F80) == 0x7F80 && (b & 0x007F) != 0
}

/// The log-probability readout of one row of BF16 logits (`row`: the row's raw little-endian
/// bytes, `2 * vocab` of them) for the emitted `token`, under the request's device sampler
/// parameters `dev` (`samp_rt::dev_params`), with `top` alternatives asked for.
pub fn row_logprobs(row: &[u8], dev: &DevSampleParams, token: i32, top: u64) -> TokenLogprob {
    let n = row.len() / 2;
    let mut words = Vec::with_capacity(n);
    for c in row.as_chunks::<2>().0 {
        words.push(u16::from_le_bytes(*c));
    }
    row_logprobs_u16(&words, dev, token, top)
}

/// [`row_logprobs`] on the row's BF16 bit patterns.
pub fn row_logprobs_u16(
    words: &[u16],
    dev: &DevSampleParams,
    token: i32,
    top: u64,
) -> TokenLogprob {
    if words.is_empty() {
        return TokenLogprob {
            token,
            logprob: f64::NEG_INFINITY,
            top: Vec::new(),
        };
    }
    let greedy = (dev.i[3] & F_GREEDY) != 0;
    // (max logit, support threshold, 1 / T)
    let (m, thr, inv_t) = if greedy {
        let mut m = f32::NEG_INFINITY;
        for &b in words {
            if !is_nan_bits(b) {
                m = m.max(f32::from_bits((b as u32) << 16));
            }
        }
        (m, f32::NEG_INFINITY, 1.0f64)
    } else {
        let sup = row_support(words, dev, 0, true);
        (sup.max_logit, sup.thr, dev.f[0] as f64)
    };
    COUNTS.with(|c| {
        let mut counts = c.borrow_mut();
        counts.iter_mut().for_each(|x| *x = 0);
        for &b in words {
            counts[key_of_bits(b as u32) as usize] += 1;
        }
        // in the support: a non-NaN value at least the threshold (compared as values, so -0 and
        // +0 are one value, as in the sampler)
        let in_support = |k: usize| -> bool {
            let v = val_of_key(k as i32);
            !v.is_nan() && v >= thr
        };
        let md = m as f64;
        let mut sum = 0.0f64;
        let mut support: u64 = 0;
        for k in (0..NBINS).rev() {
            let cnt = counts[k];
            if cnt == 0 || !in_support(k) {
                continue;
            }
            support += cnt as u64;
            let x = (val_of_key(k as i32) as f64 - md) * inv_t;
            sum += cnt as f64 * x.exp();
        }
        let lse = sum.ln();
        let lp_of = |bits: u16| -> f64 {
            let k = key_of_bits(bits as u32) as usize;
            if !in_support(k) {
                return f64::NEG_INFINITY;
            }
            (val_of_key(k as i32) as f64 - md) * inv_t - lse
        };
        let logprob = match usize::try_from(token).ok().and_then(|i| words.get(i)) {
            Some(&b) => lp_of(b),
            None => f64::NEG_INFINITY,
        };
        // the alternatives: the highest support bins holding the first `want` tokens
        let want = verified::lp_top_count(top, support) as usize;
        let mut alts: Vec<(i32, f64)> = Vec::new();
        if want > 0 {
            let mut seen = 0usize;
            let mut kmin = NBINS - 1;
            for k in (0..NBINS).rev() {
                if counts[k] == 0 || !in_support(k) {
                    continue;
                }
                seen += counts[k] as usize;
                kmin = k;
                if seen >= want {
                    break;
                }
            }
            let floor = val_of_key(kmin as i32);
            let mut cand: Vec<(f32, i32)> = words
                .iter()
                .enumerate()
                .filter_map(|(i, &b)| {
                    let v = f32::from_bits((b as u32) << 16);
                    let k = key_of_bits(b as u32) as usize;
                    (!is_nan_bits(b) && v >= floor && in_support(k)).then_some((v, i as i32))
                })
                .collect();
            cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            cand.truncate(want);
            alts = cand
                .into_iter()
                .map(|(_, id)| (id, lp_of(words[id as usize])))
                .collect();
        }
        TokenLogprob {
            token,
            logprob,
            top: alts,
        }
    })
}

/// The readout of the first `used` rows of a launch's logits copy (`copy`: `t` rows of `rb`
/// bytes each, row `i` the token at `first_pos + i`), one entry per token of `tokens`.
pub fn step_logprobs(
    copy: &[u8],
    rb: usize,
    t: u64,
    tokens: &[i32],
    dev: &DevSampleParams,
    top: u64,
) -> Vec<TokenLogprob> {
    let rows = verified::lp_step_rows(tokens.len() as u64, t) as usize;
    let row = |i: usize| {
        let start = verified::lp_row_start(i as u64, rb as u64) as usize;
        row_logprobs(&copy[start..start + rb], dev, tokens[i], top)
    };
    if rows <= 1 {
        return (0..rows).map(row).collect();
    }
    // the rows of a speculative step are independent: one host thread each
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..rows).map(|i| sc.spawn(move || row(i))).collect();
        hs.into_iter()
            .map(|h| h.join().expect("logprob row thread"))
            .collect()
    })
}

/// The `token` string of a logprob entry: the token's text, or (for bytes that are not whole
/// UTF-8, e.g. one byte of a multi-byte character) `bytes:` followed by `\xNN` per byte. The
/// entry's `bytes` field carries the exact bytes either way.
pub fn token_text(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            let mut s = String::from("bytes:");
            for b in bytes {
                s.push_str(&format!("\\x{b:02x}"));
            }
            s
        }
    }
}

/// A log-probability as JSON: a token outside the draw's support reports `-9999.0`, the value
/// the OpenAI API uses for "very unlikely" (JSON has no infinity).
pub fn logprob_json(lp: f64) -> String {
    if lp.is_finite() {
        format!("{lp}")
    } else {
        "-9999.0".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};
    use crate::sampler::{sample_row, SampleParams};

    fn row(vals: &[f32]) -> Vec<u16> {
        vals.iter().map(|&v| f32_to_bf16_rne(v)).collect()
    }

    fn naive(words: &[u16], inv_t: f64, thr: f32) -> Vec<f64> {
        let m = words
            .iter()
            .map(|&b| bf16_to_f32(b))
            .fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = words
            .iter()
            .map(|&b| bf16_to_f32(b))
            .filter(|&l| l >= thr)
            .map(|l| ((l as f64 - m) * inv_t).exp())
            .sum();
        words
            .iter()
            .map(|&b| {
                let l = bf16_to_f32(b);
                if l >= thr {
                    (l as f64 - m) * inv_t - s.ln()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    fn wide_row(n: usize, seed: u32) -> Vec<u16> {
        let mut h = seed;
        (0..n)
            .map(|i| {
                h = crate::sampler::fmix32(h ^ i as u32);
                let u = (h >> 8) as f32 / (1u32 << 24) as f32;
                let mut v = -12.0 + 14.0 * u * u;
                if i % 997 == 3 {
                    v += 9.0;
                }
                f32_to_bf16_rne(v)
            })
            .collect()
    }

    #[test]
    fn greedy_rows_report_the_full_softmax_at_temperature_one() {
        let w = row(&[1.0, 3.0, -2.0, 3.0, 0.5]);
        let dev = SampleParams::default().to_dev(5);
        let r = row_logprobs_u16(&w, &dev, 1, 3);
        let want = naive(&w, 1.0, f32::NEG_INFINITY);
        assert!((r.logprob - want[1]).abs() < 1e-12);
        // ties: the smaller id first; then the next logit
        assert_eq!(r.top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3, 0]);
        for (id, lp) in &r.top {
            assert!((lp - want[*id as usize]).abs() < 1e-12);
        }
        let total: f64 = want.iter().map(|l| l.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sampled_rows_report_the_truncated_tempered_distribution() {
        let w = wide_row(4096, 7);
        for (t, k, p) in [
            (0.7f32, 0u64, 1.0f32),
            (1.0, 20, 1.0),
            (0.8, 0, 0.9),
            (1.3, 40, 0.95),
        ] {
            let sp = SampleParams {
                temperature: t,
                top_k: k,
                top_p: p,
                seed: 5,
                ..SampleParams::default()
            };
            let dev = sp.to_dev(4096);
            let res = sample_row(&w, &dev, 0, 17, true);
            let r = row_logprobs_u16(&w, &dev, res.token as i32, 20);
            let want = naive(&w, dev.f[0] as f64, res.thr);
            assert!(r.logprob.is_finite(), "the drawn token is in the support");
            assert!(
                (r.logprob - want[res.token]).abs() < 1e-9,
                "T={t} k={k} p={p}"
            );
            let support = want.iter().filter(|l| l.is_finite()).count();
            assert_eq!(r.top.len(), support.min(20));
            if k > 0 && p >= 1.0 {
                // top-k by value: fewer than k tokens lie strictly above the threshold (ties at
                // the threshold stay in, as in the sampler)
                let above = w.iter().filter(|&&b| bf16_to_f32(b) > res.thr).count();
                assert!(above < k as usize && support >= k as usize);
            }
            // the alternatives are the most likely tokens, in order
            let mut sorted: Vec<f64> = want.iter().copied().filter(|l| l.is_finite()).collect();
            sorted.sort_by(|a, b| b.total_cmp(a));
            for (j, (_, lp)) in r.top.iter().enumerate() {
                assert!((lp - sorted[j]).abs() < 1e-9);
            }
            let mass: f64 = want.iter().filter(|l| l.is_finite()).map(|l| l.exp()).sum();
            assert!((mass - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn a_small_support_reports_fewer_alternatives() {
        let w = row(&[5.0, 4.0, 3.0, -1.0]);
        let sp = SampleParams {
            temperature: 1.0,
            top_k: 2,
            ..SampleParams::default()
        };
        let dev = sp.to_dev(4);
        let r = row_logprobs_u16(&w, &dev, 0, 5);
        assert_eq!(r.top.len(), 2);
        let z = (1.0f64 + (-1.0f64).exp()).ln();
        assert!((r.logprob + z).abs() < 1e-12);
        let out = row_logprobs_u16(&w, &dev, 3, 0);
        assert_eq!(out.logprob, f64::NEG_INFINITY);
        assert!(out.top.is_empty());
        assert_eq!(logprob_json(out.logprob), "-9999.0");
    }

    #[test]
    fn step_rows_follow_the_row_layout() {
        let rb = 8usize; // 4 BF16 logits per row
        let rows = [
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ];
        let mut copy = Vec::new();
        for r in rows {
            for v in r {
                copy.extend_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
            }
        }
        let dev = SampleParams::default().to_dev(4);
        let lp = step_logprobs(&copy, rb, 3, &[0, 1], &dev, 1);
        assert_eq!(lp.len(), 2);
        assert_eq!(lp[0].top[0].0, 0);
        assert_eq!(lp[1].top[0].0, 1);
    }

    #[test]
    fn partial_utf8_tokens_are_spelled_by_their_bytes() {
        assert_eq!(token_text("é".as_bytes()), "é");
        assert_eq!(token_text(&[0xE2, 0x80]), "bytes:\\xe2\\x80");
        assert!(top_ok(0) && top_ok(20) && !top_ok(21));
        assert_eq!(max_top(), 20);
    }

    /// Host cost of one 248 320-token row (run with `--ignored --nocapture` in a release build).
    #[test]
    #[ignore]
    fn row_cost_on_a_full_vocabulary() {
        let w = wide_row(248_320, 3);
        let bytes: Vec<u8> = w.iter().flat_map(|b| b.to_le_bytes()).collect();
        for (name, sp) in [
            ("greedy", SampleParams::default()),
            (
                "T=0.7",
                SampleParams {
                    temperature: 0.7,
                    ..SampleParams::default()
                },
            ),
            (
                "T=0.7 top_p=0.95",
                SampleParams {
                    temperature: 0.7,
                    top_p: 0.95,
                    ..SampleParams::default()
                },
            ),
            (
                "T=0.7 top_k=20 top_p=0.95",
                SampleParams {
                    temperature: 0.7,
                    top_k: 20,
                    top_p: 0.95,
                    ..SampleParams::default()
                },
            ),
        ] {
            let dev = sp.to_dev(248_320);
            let t0 = std::time::Instant::now();
            let n = 20;
            for _ in 0..n {
                std::hint::black_box(row_logprobs(&bytes, &dev, 0, 20));
            }
            eprintln!(
                "{name}: {:.3} ms per row",
                t0.elapsed().as_secs_f64() * 1e3 / n as f64
            );
        }
    }

    /// Today's host ring (the laws bound it; the layout itself is pinned here): one slot per
    /// launch parity, as many rows as the widest launch, slot 1 right after slot 0.
    #[test]
    fn host_ring_layout_today() {
        let rb = 248_320 * 2; // one BF16 row of the Qwen3.8 vocabulary
        assert_eq!(verified::lp_ring_rows(0), 1, "no drafter: one-row steps");
        assert_eq!(verified::lp_ring_rows(16), 16);
        let cap = verified::lp_ring_rows(16);
        let ring = verified::lp_ring_bytes(cap, rb);
        assert_eq!(ring, 2 * 16 * rb);
        assert!(ring < 16 << 20, "about 15 MiB for a 16-row verify block");
        assert_eq!(verified::lp_slot_start(0, cap, rb), 0);
        assert_eq!(verified::lp_slot_start(1, cap, rb), 16 * rb);
        // a copy goes into either parity's slot when it fits, never into a third one
        assert!(verified::lp_slot_ok(0, 16, cap) && verified::lp_slot_ok(1, 1, cap));
        assert!(!verified::lp_slot_ok(2, 1, cap));
        assert!(!verified::lp_slot_ok(0, 17, cap));
        // the device side: the copy's bytes, not its rows, are compared with the buffer
        assert!(verified::lp_src_fits(16, rb, 16 * rb));
        assert!(!verified::lp_src_fits(16, rb, 16 * rb - 1));
        assert!(!verified::lp_src_fits(2, rb, rb));
    }
}
