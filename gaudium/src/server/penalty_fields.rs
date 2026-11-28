//! Request fields of the sampler penalties: `presence_penalty`, `frequency_penalty`,
//! `repetition_penalty` and `logit_bias`, parsed and range-checked into
//! `sampler_penalty::Penalties`. An out-of-range value is refused (the caller answers 400).
//! Whether the model's engine may take the result is the verified `pen_path_allowed`.
//!
//! Accepted ranges: `presence_penalty` and `frequency_penalty` -2..2, `repetition_penalty`
//! 0.01..100 (1 = off), `logit_bias` values -100..100 (-100 bans the token). Every one is finite,
//! so with the card's logits (tens, not near the BF16 maximum) no adjusted logit overflows; a
//! row that overflows anyway is refused by `sampler_penalty::pick`, never sampled.

use crate::json::JsonValue;
use crate::sampler_penalty::{Bias, Penalties, TokenMask};
use crate::verified;

/// `presence_penalty` and `frequency_penalty` accept `-LIMIT..=LIMIT`.
pub const PRESENCE_FREQUENCY_LIMIT: f64 = 2.0;
/// `logit_bias` values accept `-LIMIT..=LIMIT`; `-LIMIT` bans the token.
pub const BIAS_LIMIT: f64 = 100.0;
/// `repetition_penalty` accepts `MIN..=MAX`: a divisor far below 1 turns a logit into +inf.
pub const REPETITION_MIN: f64 = 0.01;
pub const REPETITION_MAX: f64 = 100.0;

/// A numeric field: absent or `null` -> `None`; any other non-number is refused.
fn number(val: &JsonValue, key: &str) -> Result<Option<f64>, String> {
    match val.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() => Ok(Some(x)),
            _ => Err(format!("'{key}' must be a number")),
        },
    }
}

/// Reads the penalty fields of a request for a model whose requests may emit `vocab` token ids
/// (the verified `pen_emit_vocab` of the logits row and the tokenizer).
pub fn parse(val: &JsonValue, vocab: u64) -> Result<Penalties, String> {
    let mut p = Penalties::for_vocab(vocab);
    for (key, slot) in [
        ("presence_penalty", &mut p.presence),
        ("frequency_penalty", &mut p.frequency),
    ] {
        if let Some(x) = number(val, key)? {
            if !(-PRESENCE_FREQUENCY_LIMIT..=PRESENCE_FREQUENCY_LIMIT).contains(&x) {
                return Err(format!(
                    "'{key}' must be between -{PRESENCE_FREQUENCY_LIMIT} and {PRESENCE_FREQUENCY_LIMIT}, got {x}"
                ));
            }
            *slot = x as f32;
        }
    }
    if let Some(x) = number(val, "repetition_penalty")? {
        if !(REPETITION_MIN..=REPETITION_MAX).contains(&x) {
            return Err(format!(
                "'repetition_penalty' must be between {REPETITION_MIN} and {REPETITION_MAX}, got {x}"
            ));
        }
        p.repetition = x as f32;
    }
    match val.get("logit_bias") {
        None | Some(JsonValue::Null) => {}
        Some(JsonValue::Object(entries)) => {
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                let id: u64 = k
                    .trim()
                    .parse()
                    .map_err(|_| format!("'logit_bias' keys must be token ids, got '{k}'"))?;
                if !verified::pen_token_in_vocab(id, vocab) {
                    return Err(format!(
                        "'logit_bias' names token {id}, outside the model's vocabulary of {vocab}"
                    ));
                }
                let b = v
                    .as_f64()
                    .filter(|b| b.is_finite())
                    .ok_or_else(|| format!("'logit_bias' value of token {id} must be a number"))?;
                if !(-BIAS_LIMIT..=BIAS_LIMIT).contains(&b) {
                    return Err(format!(
                        "'logit_bias' value of token {id} must be between -{BIAS_LIMIT} and {BIAS_LIMIT}, got {b}"
                    ));
                }
                out.push((
                    id as u32,
                    if b <= -BIAS_LIMIT {
                        Bias::Ban
                    } else {
                        Bias::Add(b as f32)
                    },
                ));
            }
            p.bias = TokenMask::new(out);
            if !verified::pen_bans_leave_tokens(p.bias.banned(), vocab) {
                return Err("'logit_bias' bans every token of the vocabulary".into());
            }
        }
        Some(_) => return Err("'logit_bias' must be an object of token id -> bias".into()),
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    fn p(s: &str) -> Result<Penalties, String> {
        parse(&parse_json(s).unwrap(), 1000)
    }

    #[test]
    fn absent_and_null_fields_are_inactive() {
        assert!(!p("{}").unwrap().active());
        assert!(
            !p(r#"{"presence_penalty":null,"logit_bias":null,"frequency_penalty":0}"#)
                .unwrap()
                .active()
        );
        assert!(!p(r#"{"repetition_penalty":1.0,"logit_bias":{}}"#)
            .unwrap()
            .active());
    }

    #[test]
    fn values_in_range_are_taken() {
        let q = p(
            r#"{"presence_penalty":-2,"frequency_penalty":2,"repetition_penalty":1.3,
                    "logit_bias":{"5":-100,"7":2.5,"999":100}}"#,
        )
        .unwrap();
        assert_eq!((q.presence, q.frequency, q.repetition), (-2.0, 2.0, 1.3));
        assert_eq!(q.bias.get(5), Some(Bias::Ban));
        assert_eq!(q.bias.get(7), Some(Bias::Add(2.5)));
        assert_eq!(q.bias.get(999), Some(Bias::Add(100.0)));
        assert!(q.active());
    }

    #[test]
    fn out_of_range_values_are_refused() {
        for bad in [
            r#"{"presence_penalty":2.01}"#,
            r#"{"frequency_penalty":-3}"#,
            r#"{"frequency_penalty":"1"}"#,
            r#"{"repetition_penalty":0}"#,
            r#"{"repetition_penalty":-1}"#,
            r#"{"repetition_penalty":1e300}"#,
            r#"{"repetition_penalty":1e-40}"#,
            r#"{"repetition_penalty":0.009}"#,
            r#"{"repetition_penalty":100.5}"#,
            r#"{"logit_bias":{"1000":1}}"#,
            r#"{"logit_bias":{"-1":1}}"#,
            r#"{"logit_bias":{"abc":1}}"#,
            r#"{"logit_bias":{"3":100.5}}"#,
            r#"{"logit_bias":{"3":"x"}}"#,
            r#"{"logit_bias":[1,2]}"#,
        ] {
            assert!(p(bad).is_err(), "{bad} was accepted");
        }
    }

    #[test]
    fn the_repetition_range_ends_are_accepted() {
        assert_eq!(
            p(r#"{"repetition_penalty":0.01}"#).unwrap().repetition,
            0.01
        );
        assert_eq!(
            p(r#"{"repetition_penalty":100}"#).unwrap().repetition,
            100.0
        );
        assert_eq!(p("{}").unwrap().vocab, 1000);
    }

    #[test]
    fn banning_every_token_of_a_padded_row_is_refused() {
        // 4 tokenizer tokens in a row padded to 8: the ids a request may emit are the 4 tokens
        let vocab = verified::pen_emit_vocab(8, 4);
        let all: Vec<String> = (0..4).map(|i| format!("\"{i}\":-100")).collect();
        let v = parse_json(&format!("{{\"logit_bias\":{{{}}}}}", all.join(","))).unwrap();
        assert!(parse(&v, vocab).is_err());
        // and a padding id is not a token a request may name
        let v = parse_json(r#"{"logit_bias":{"5":1}}"#).unwrap();
        assert!(parse(&v, vocab).is_err());
    }

    #[test]
    fn banning_the_whole_vocabulary_is_refused() {
        let all: Vec<String> = (0..4).map(|i| format!("\"{i}\":-100")).collect();
        let v = parse_json(&format!("{{\"logit_bias\":{{{}}}}}", all.join(","))).unwrap();
        assert!(parse(&v, 4).is_err());
        assert!(parse(&v, 5).is_ok());
    }
}
