//! Plain text completion requests: `POST /completion` (the native shape) and `POST /completions`,
//! `POST /v1/completions` (the OpenAI text-completion shape). The prompt is the client's own
//! text or token ids, run without a chat template; this file reads it and sorts the request's
//! other fields into applied, refused (a 400 naming the field) and accepted but not applied
//! (listed in the reply's `gaudium.ignored`).
//!
//! The run itself (admission, sampling, the card, the output) is the chat path's
//! (`super::run_prompt`); only the prompt and the reply's shape differ.

use crate::json::JsonValue;
use crate::verified::prompt_token_ok;

/// The prompt of a completion request, as the client gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CompletionPrompt {
    /// Text, tokenised with the model's tokenizer (special-token text is matched); no chat
    /// template and no beginning-of-sequence token are added.
    Text(String),
    /// Token ids, run as given (each checked against the model, [`prompt_ids`]).
    Ids(Vec<u64>),
}

/// What a completion request asks for beyond the fields the chat path reads itself.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CompletionRequest {
    pub prompt: CompletionPrompt,
    /// The generation limit (`max_tokens`, else the native `n_predict`); `None`: none named.
    pub max_tokens: Option<u64>,
    /// Fields accepted but not applied.
    pub ignored: Vec<String>,
}

/// Fields this path applies (read here or by the shared run), or that ask for nothing beyond
/// what it does anyway: never reported.
const APPLIED: &[&str] = &[
    "model",
    "prompt",
    "stream",
    // the final chunk always carries `usage`
    "stream_options",
    "max_tokens",
    "n_predict",
    "stop",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "cache_prompt",
    "timings_per_token",
    "return_progress",
    "coupled",
    "draft_trunc",
    "gaudium_method",
    "gaudium_return_tokens",
    // sampler penalties and the logit bias (the shared run parses them: `penalty_fields`)
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logit_bias",
];

/// Reads a completion request body (`native`: the native `/completion` shape). `Err`: a 400
/// with this message.
pub(super) fn parse(body: &JsonValue, native: bool) -> Result<CompletionRequest, String> {
    let Some(fields) = body.as_object() else {
        return Err("the request body must be a JSON object".into());
    };
    let prompt = parse_prompt(body.get("prompt"))?;
    refuse_unsupported(body)?;
    if !native {
        // (applied by the shared run, which reads it again)
        logprobs_field(body)?;
    }
    let max_tokens = match body.get("max_tokens") {
        // as on the chat path: 0 or absent names no limit
        Some(v) if !matches!(v, JsonValue::Null) => match v.as_u64() {
            Some(n) => Some(n).filter(|&n| n > 0),
            None => return Err("'max_tokens' must be a non-negative integer".into()),
        },
        _ => n_predict(body.get("n_predict"))?,
    };
    let mut ignored = Vec::new();
    for (k, v) in fields {
        if APPLIED.contains(&k.as_str()) || is_default(k, v) || (!native && k == "logprobs") {
            continue;
        }
        let name = match k.as_str() {
            "logprobs" | "n_probs" => format!(
                "{k} (token log-probabilities are not reported in the native shape yet: send logprobs to /v1/completions)"
            ),
            "repeat_penalty" => format!("{k} (not applied: send repetition_penalty)"),
            _ => k.clone(),
        };
        if !ignored.contains(&name) {
            ignored.push(name);
        }
    }
    Ok(CompletionRequest {
        prompt,
        max_tokens,
        ignored,
    })
}

/// The OpenAI shape's `logprobs`: `Some(n)` asks for every token's log-probability with its
/// `n` likeliest alternatives (`n = 0`: none), `None` (absent, `null` or `false`) for none. `n`
/// is an integer in the verified range (`crate::logprobs::top_ok`).
pub(super) fn logprobs_field(body: &JsonValue) -> Result<Option<u64>, String> {
    let max = crate::logprobs::max_top();
    match body.get("logprobs") {
        None | Some(JsonValue::Null) | Some(JsonValue::Bool(false)) => Ok(None),
        Some(v) => {
            let n = v
                .as_u64()
                .filter(|&n| v.as_f64() == Some(n as f64))
                .ok_or_else(|| format!("'logprobs' must be an integer from 0 to {max}"))?;
            if !crate::logprobs::top_ok(n) {
                return Err(format!(
                    "'logprobs' must be an integer from 0 to {max} (got {n})"
                ));
            }
            Ok(Some(n))
        }
    }
}

/// The native `n_predict`: a negative value (or none) names no limit; 0 (process the prompt
/// only) is refused, as no run here returns without a generated token.
fn n_predict(v: Option<&JsonValue>) -> Result<Option<u64>, String> {
    match v {
        None | Some(JsonValue::Null) => Ok(None),
        Some(v) => match v.as_i64() {
            Some(n) if n < 0 => Ok(None),
            Some(0) => Err(
                "'n_predict': 0 (prompt processing only) is not supported: ask for at least 1 token"
                    .into(),
            ),
            Some(n) => Ok(Some(n as u64)),
            None => Err("'n_predict' must be an integer".into()),
        },
    }
}

/// Fields whose silent drop would hand the client a different reply than it asked for, in a
/// way it could not tell from the reply.
fn refuse_unsupported(body: &JsonValue) -> Result<(), String> {
    if let Some(n) = body.get("n").filter(|v| !matches!(v, JsonValue::Null)) {
        if n.as_u64() != Some(1) {
            return Err("'n': only one choice per request is supported (n = 1)".into());
        }
    }
    if let Some(b) = body
        .get("best_of")
        .filter(|v| !matches!(v, JsonValue::Null))
    {
        if b.as_u64() != Some(1) {
            return Err("'best_of': only best_of = 1 is supported".into());
        }
    }
    if body.get("echo").and_then(JsonValue::as_bool) == Some(true) {
        return Err("'echo': returning the prompt with the completion is not supported".into());
    }
    match body.get("suffix") {
        None | Some(JsonValue::Null) => {}
        Some(JsonValue::Str(s)) if s.is_empty() => {}
        Some(_) => {
            return Err("'suffix': insertion (text after the completion) is not supported".into())
        }
    }
    Ok(())
}

/// A field present with the value that asks for nothing (not worth reporting).
fn is_default(key: &str, v: &JsonValue) -> bool {
    match (key, v) {
        (_, JsonValue::Null) => true,
        ("n" | "best_of", v) => v.as_u64() == Some(1),
        ("echo", JsonValue::Bool(false)) => true,
        ("suffix", JsonValue::Str(s)) => s.is_empty(),
        ("logprobs", JsonValue::Bool(false)) => true,
        // (the native shape reports none either way; the OpenAI shape applies `logprobs`)
        ("logprobs" | "n_probs", v) => v.as_u64() == Some(0),
        ("repeat_penalty", v) => v.as_f64() == Some(1.0),
        _ => false,
    }
}

/// `prompt`: a string, an array of token ids, or a one-element array of either. More than one
/// prompt in a request (more than one choice) is refused for now.
fn parse_prompt(v: Option<&JsonValue>) -> Result<CompletionPrompt, String> {
    match v {
        None | Some(JsonValue::Null) => Err("'prompt' is required".into()),
        Some(JsonValue::Str(s)) => Ok(CompletionPrompt::Text(s.clone())),
        Some(JsonValue::Array(a)) if a.is_empty() => Err("'prompt' is an empty array".into()),
        Some(JsonValue::Array(a)) => {
            let numbers = a
                .iter()
                .filter(|x| matches!(x, JsonValue::Int(_) | JsonValue::Float(_)))
                .count();
            if numbers == a.len() {
                return ids(a).map(CompletionPrompt::Ids);
            }
            if numbers > 0 {
                return Err(
                    "'prompt' mixes token ids and other values: give one string or one array of token ids"
                        .into(),
                );
            }
            if a.len() > 1 {
                return Err(format!(
                    "'prompt' holds {} prompts: one prompt per request is supported",
                    a.len()
                ));
            }
            match &a[0] {
                JsonValue::Str(s) => Ok(CompletionPrompt::Text(s.clone())),
                JsonValue::Array(inner) if !inner.is_empty() => {
                    ids(inner).map(CompletionPrompt::Ids)
                }
                _ => Err("'prompt' must be a string, an array of token ids, or a one-element array of either".into()),
            }
        }
        Some(_) => Err("'prompt' must be a string or an array of token ids".into()),
    }
}

fn ids(a: &[JsonValue]) -> Result<Vec<u64>, String> {
    a.iter()
        .map(|x| {
            x.as_u64()
                .ok_or_else(|| "'prompt' token ids must be non-negative integers".to_string())
        })
        .collect()
}

/// The engine's token ids of a prompt given as ids: every id must be a row of the model's
/// embedding (`rows`) and a token its tokenizer names (`names` entries), the verified
/// `prompt_token_ok`.
pub(super) fn prompt_ids(ids: &[u64], rows: u64, names: u64) -> Result<Vec<i32>, String> {
    ids.iter()
        .map(|&id| {
            if !prompt_token_ok(id, rows, names) {
                return Err(format!(
                    "prompt token id {id} is outside the model's vocabulary (0..{})",
                    rows.min(names)
                ));
            }
            i32::try_from(id).map_err(|_| format!("prompt token id {id} does not fit a token id"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    fn req(body: &str) -> Result<CompletionRequest, String> {
        parse(&parse_json(body).unwrap(), false)
    }

    fn native_req(body: &str) -> Result<CompletionRequest, String> {
        parse(&parse_json(body).unwrap(), true)
    }

    #[test]
    fn prompt_forms() {
        let p = |b: &str| req(b).map(|r| r.prompt);
        assert_eq!(
            p(r#"{"prompt":"Hello"}"#),
            Ok(CompletionPrompt::Text("Hello".into()))
        );
        assert_eq!(
            p(r#"{"prompt":["Hello"]}"#),
            Ok(CompletionPrompt::Text("Hello".into()))
        );
        assert_eq!(
            p(r#"{"prompt":[1, 2, 3]}"#),
            Ok(CompletionPrompt::Ids(vec![1, 2, 3]))
        );
        assert_eq!(
            p(r#"{"prompt":[[4, 5]]}"#),
            Ok(CompletionPrompt::Ids(vec![4, 5]))
        );
        for bad in [
            r#"{}"#,
            r#"{"prompt":null}"#,
            r#"{"prompt":[]}"#,
            r#"{"prompt":[[]]}"#,
            r#"{"prompt":["a", "b"]}"#,
            r#"{"prompt":[[1], [2]]}"#,
            r#"{"prompt":[1, "a"]}"#,
            r#"{"prompt":[1, -2]}"#,
            r#"{"prompt":[1.5]}"#,
            r#"{"prompt":7}"#,
            r#"{"prompt":{"text":"a"}}"#,
        ] {
            assert!(p(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn limits_from_max_tokens_or_n_predict() {
        let m = |b: &str| req(b).map(|r| r.max_tokens);
        assert_eq!(m(r#"{"prompt":"a"}"#), Ok(None));
        assert_eq!(m(r#"{"prompt":"a","max_tokens":12}"#), Ok(Some(12)));
        assert_eq!(m(r#"{"prompt":"a","max_tokens":0}"#), Ok(None));
        assert_eq!(m(r#"{"prompt":"a","n_predict":7}"#), Ok(Some(7)));
        assert_eq!(m(r#"{"prompt":"a","n_predict":-1}"#), Ok(None));
        // max_tokens wins over n_predict
        assert_eq!(
            m(r#"{"prompt":"a","max_tokens":3,"n_predict":9}"#),
            Ok(Some(3))
        );
        assert!(m(r#"{"prompt":"a","n_predict":0}"#).is_err());
        assert!(m(r#"{"prompt":"a","max_tokens":-4}"#).is_err());
        assert!(m(r#"{"prompt":"a","n_predict":"x"}"#).is_err());
    }

    #[test]
    fn fields_that_change_the_reply_are_refused() {
        for (bad, field) in [
            (r#"{"prompt":"a","n":2}"#, "'n'"),
            (r#"{"prompt":"a","best_of":3}"#, "'best_of'"),
            (r#"{"prompt":"a","echo":true}"#, "'echo'"),
            (r#"{"prompt":"a","suffix":"tail"}"#, "'suffix'"),
        ] {
            let e = req(bad).unwrap_err();
            assert!(e.contains(field), "{bad}: {e}");
        }
        // the values that ask for nothing run, unreported
        let r = req(
            r#"{"prompt":"a","n":1,"best_of":1,"echo":false,"suffix":"","logprobs":null,
                "presence_penalty":0,"frequency_penalty":0.0,"repeat_penalty":1.0,"logit_bias":{}}"#,
        )
        .unwrap();
        assert!(r.ignored.is_empty(), "{:?}", r.ignored);
    }

    #[test]
    fn fields_not_applied_are_reported() {
        let r = req(
            r#"{"prompt":"a","temperature":0,"logprobs":5,"presence_penalty":0.5,"repeat_penalty":1.2,"user":"u","future_field":1}"#,
        )
        .unwrap();
        // presence_penalty and logprobs are applied (the shared run reads them): not reported
        assert_eq!(r.ignored.len(), 3, "{:?}", r.ignored);
        assert!(r.ignored[0].starts_with("repeat_penalty (not applied: send repetition_penalty)"));
        assert_eq!(r.ignored[1], "user");
        assert_eq!(r.ignored[2], "future_field");
        // the native shape reports logprobs and n_probs as not applied
        let r = native_req(r#"{"prompt":"a","logprobs":5,"n_probs":3}"#).unwrap();
        assert_eq!(r.ignored.len(), 2, "{:?}", r.ignored);
        assert!(r.ignored[0].starts_with("logprobs (token log-probabilities are not reported"));
        assert!(r.ignored[1].starts_with("n_probs"));
    }

    #[test]
    fn openai_logprobs_field_is_an_integer_in_range() {
        let f = |b: &str| logprobs_field(&parse_json(b).unwrap());
        assert_eq!(f(r#"{}"#), Ok(None));
        assert_eq!(f(r#"{"logprobs":null}"#), Ok(None));
        assert_eq!(f(r#"{"logprobs":false}"#), Ok(None));
        assert_eq!(f(r#"{"logprobs":0}"#), Ok(Some(0)));
        assert_eq!(f(r#"{"logprobs":5}"#), Ok(Some(5)));
        assert_eq!(f(r#"{"logprobs":20}"#), Ok(Some(20)));
        for bad in [
            r#"{"logprobs":21}"#,
            r#"{"logprobs":-1}"#,
            r#"{"logprobs":2.5}"#,
            r#"{"logprobs":true}"#,
            r#"{"logprobs":"5"}"#,
        ] {
            assert!(f(bad).is_err(), "{bad}");
            // and the request is refused before it runs
            assert!(
                req(&bad.replace('{', r#"{"prompt":"a","#)).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn prompt_ids_inside_the_vocabulary_only() {
        assert_eq!(prompt_ids(&[0, 5, 9], 10, 12), Ok(vec![0, 5, 9]));
        // the embedding's rows and the tokenizer's names both bound an id
        assert!(prompt_ids(&[10], 10, 12).is_err());
        assert!(prompt_ids(&[10], 12, 10).is_err());
        assert!(prompt_ids(&[u64::MAX], 10, 12).is_err());
        assert!(prompt_ids(&[1], 0, 0).is_err());
    }
}
