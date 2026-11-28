//! The top-level request fields of `POST /v1/chat/completions`, each in one class:
//!
//! - **applied**: the chat path reads the field and acts on it;
//! - **ignored**: accepted but not applied; the reply names it in `gaudium.ignored`;
//! - **refused**: a 400 that names the field, because ignoring it would change what the client
//!   gets in a way the client cannot see (`logprobs: true` and no log probabilities back).
//!
//! A field outside the table is reported in `gaudium.ignored` as unknown, never refused, so a
//! client written for a newer API keeps working. An ignored or refused field whose value changes
//! nothing (`logprobs: false`, `stream_options: {"include_usage": true}`, or `null`) is neither
//! reported nor refused. As a feature lands, its field moves to `Applied`
//! here (and the code that reads it moves into the chat path).
//!
//! This is request parsing (AGENTS.md 3a, second list): it decides only whether the request is
//! well-formed for this server and what the reply reports.

use crate::json::JsonValue;

/// What the chat path does with a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Read and applied.
    Applied,
    /// Accepted, not applied; the reply reports it with this reason.
    Ignored(&'static str),
    /// Refused with a 400; the message adds this reason.
    Refused(&'static str),
}

/// One known field: its class, and which of its values change nothing (never reported or
/// refused). `null` changes nothing for every field.
struct Field {
    name: &'static str,
    class: Class,
    noop: fn(&JsonValue) -> bool,
}

const fn applied(name: &'static str) -> Field {
    Field {
        name,
        class: Class::Applied,
        noop: never,
    }
}

const fn ignored(name: &'static str, why: &'static str, noop: fn(&JsonValue) -> bool) -> Field {
    Field {
        name,
        class: Class::Ignored(why),
        noop,
    }
}

const fn refused(name: &'static str, why: &'static str, noop: fn(&JsonValue) -> bool) -> Field {
    Field {
        name,
        class: Class::Refused(why),
        noop,
    }
}

/// The table. Keep one entry per field; a later change moves a field between classes.
const FIELDS: &[Field] = &[
    // the API's own fields the chat path applies
    applied("messages"),
    applied("model"),
    applied("stream"),
    applied("temperature"),
    applied("top_p"),
    applied("max_tokens"),
    applied("max_completion_tokens"),
    applied("stop"),
    applied("seed"),
    applied("tools"),
    applied("tool_choice"),
    applied("parallel_tool_calls"),
    applied("reasoning_effort"),
    // extensions the chat path applies
    applied("top_k"),
    applied("min_p"),
    applied("reasoning_format"),
    applied("chat_template_kwargs"),
    applied("continue_final_message"),
    applied("add_generation_prompt"),
    applied("cache_prompt"),
    applied("timings_per_token"),
    applied("return_progress"),
    applied("coupled"),
    applied("draft_trunc"),
    applied("gaudium_method"),
    applied("gaudium_a2ui"),
    applied("gaudium_hidden_tools"),
    applied("gaudium_return_tokens"),
    // sampler penalties and the logit bias (`server/penalty_fields.rs` parses them)
    applied("presence_penalty"),
    applied("frequency_penalty"),
    applied("repetition_penalty"),
    applied("logit_bias"),
    // the reasoning span's token budget (the Messages API's `thinking.budget_tokens`)
    applied("thinking_budget_tokens"),
    // structured output (`server/structured.rs` parses it)
    applied("response_format"),
    // choices (`server/choices.rs` parses and bounds it, `--max-choices`)
    applied("n"),
    // refused: ignoring them changes the reply without a sign
    refused(
        "logprobs",
        "token log probabilities are not returned yet",
        is_false,
    ),
    refused(
        "top_logprobs",
        "token log probabilities are not returned yet",
        is_zero,
    ),
    // ignored and reported
    ignored(
        "stream_options",
        "not applied: the last chunk of a stream always carries usage",
        is_usage_only,
    ),
    ignored(
        "functions",
        "not applied: the legacy function-calling fields are not supported; use tools",
        is_empty_array,
    ),
    ignored(
        "function_call",
        "not applied: the legacy function-calling fields are not supported; use tool_choice",
        never,
    ),
    ignored(
        "modalities",
        "not applied: replies are text only",
        is_text_modalities,
    ),
    ignored("audio", "not applied: replies are text only", never),
    ignored(
        "prediction",
        "not applied: predicted outputs are not used",
        never,
    ),
    ignored(
        "web_search_options",
        "not applied: this server does not search the web",
        never,
    ),
    ignored("user", "not applied: end-user ids are not recorded", never),
    ignored(
        "safety_identifier",
        "not applied: end-user ids are not recorded",
        never,
    ),
    ignored("metadata", "not applied: completions are not stored", never),
    ignored("store", "not applied: completions are not stored", is_false),
    ignored("service_tier", "not applied: there is one tier", never),
    ignored(
        "prompt_cache_key",
        "not applied: prompt reuse is automatic",
        never,
    ),
    ignored("verbosity", "not applied", never),
    ignored(
        "sse_ping_interval",
        "not applied: the stream's keep-alive interval is fixed",
        never,
    ),
];

/// A refused field: its name and the error message that names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub field: String,
    pub message: String,
}

/// Sorts the top-level fields of a chat-completions body: `Ok` with what the reply reports in
/// `gaudium.ignored` (in body order), or the first refused field. A body that is not an object
/// gives `Ok` with nothing (the prompt renderer refuses it).
pub fn check(body: &JsonValue) -> Result<Vec<String>, Refusal> {
    let Some(entries) = body.as_object() else {
        return Ok(Vec::new());
    };
    let mut ignored = Vec::new();
    for (name, value) in entries {
        if matches!(value, JsonValue::Null) {
            continue;
        }
        let Some(field) = FIELDS.iter().find(|f| f.name == name) else {
            ignored.push(format!("{name} (unknown field)"));
            continue;
        };
        if (field.noop)(value) {
            continue;
        }
        match field.class {
            Class::Applied => {}
            Class::Ignored(why) => ignored.push(format!("{name} ({why})")),
            Class::Refused(why) => {
                return Err(Refusal {
                    field: name.clone(),
                    message: format!("{name}={}: {why}", short(value)),
                })
            }
        }
    }
    Ok(ignored)
}

/// The value as the error shows it, cut to a readable length.
fn short(v: &JsonValue) -> String {
    let s = v.dump();
    match s.char_indices().nth(80) {
        Some((i, _)) => format!("{}...", &s[..i]),
        None => s,
    }
}

fn never(_: &JsonValue) -> bool {
    false
}

fn is_false(v: &JsonValue) -> bool {
    matches!(v, JsonValue::Bool(false))
}

fn is_zero(v: &JsonValue) -> bool {
    v.as_f64() == Some(0.0)
}

fn is_empty_array(v: &JsonValue) -> bool {
    v.as_array().is_some_and(|a| a.is_empty())
}

/// `{"type": "text"}`: plain text, what the reply is anyway.
/// `["text"]`: text out, what the reply is anyway.
fn is_text_modalities(v: &JsonValue) -> bool {
    v.as_array()
        .is_some_and(|a| a.len() == 1 && a[0].as_str() == Some("text"))
}

/// `{"include_usage": true}` (or `{}`): usage in the last chunk, which a stream carries anyway.
fn is_usage_only(v: &JsonValue) -> bool {
    v.as_object().is_some_and(|o| {
        o.iter()
            .all(|(k, val)| k == "include_usage" && matches!(val, JsonValue::Bool(true)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    fn run(body: &str) -> Result<Vec<String>, Refusal> {
        check(&parse_json(body).unwrap())
    }

    #[test]
    fn the_table_names_each_field_once() {
        for (i, f) in FIELDS.iter().enumerate() {
            assert!(
                FIELDS[i + 1..].iter().all(|g| g.name != f.name),
                "{} is in the table twice",
                f.name
            );
        }
    }

    #[test]
    fn applied_fields_are_not_reported() {
        let r = run(
            r#"{"messages":[],"model":"m","stream":true,"temperature":0.7,"top_p":0.9,
            "top_k":20,"min_p":0.05,"seed":3,"max_tokens":10,"stop":["x"],"tools":[],
            "tool_choice":"auto","parallel_tool_calls":false,"reasoning_effort":"low",
            "chat_template_kwargs":{},"cache_prompt":true,"gaudium_method":"default",
            "thinking_budget_tokens":256,"n":3}"#,
        );
        assert_eq!(r, Ok(Vec::new()));
    }

    #[test]
    fn ignored_fields_are_reported_with_a_reason() {
        let r = run(
            r#"{"messages":[],"presence_penalty":0.5,"logit_bias":{"12":-100},"user":"u1","service_tier":"auto"}"#,
        )
        .unwrap();
        // the penalties and the bias are applied now: only the other two are reported
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(r[0].starts_with("user (not applied"), "{r:?}");
        assert!(r[1].starts_with("service_tier (not applied"), "{r:?}");
    }

    #[test]
    fn values_that_change_nothing_are_neither_reported_nor_refused() {
        let r = run(r#"{"messages":[],"n":1,"logprobs":false,"top_logprobs":0,
            "response_format":{"type":"text"},"presence_penalty":0,"frequency_penalty":0.0,
            "repetition_penalty":1.0,"logit_bias":{},"stream_options":{"include_usage":true},
            "modalities":["text"],"store":false,"functions":[],"audio":null}"#);
        assert_eq!(r, Ok(Vec::new()));
    }

    #[test]
    fn unknown_fields_are_reported_never_refused() {
        let r = run(r#"{"messages":[],"some_future_field":{"a":1}}"#).unwrap();
        assert_eq!(r, vec!["some_future_field (unknown field)".to_string()]);
    }

    #[test]
    fn refused_fields_name_the_field() {
        for (body, field) in [
            (r#"{"logprobs":true}"#, "logprobs"),
            (r#"{"top_logprobs":5}"#, "top_logprobs"),
        ] {
            let e = run(body).unwrap_err();
            assert_eq!(e.field, field, "{body}");
            assert!(e.message.starts_with(&format!("{field}=")), "{}", e.message);
        }
    }

    #[test]
    fn a_long_value_is_cut_in_the_message() {
        let long = "x".repeat(500);
        let e = run(&format!(r#"{{"logprobs":{{"s":"{long}"}}}}"#)).unwrap_err();
        assert!(e.message.len() < 300, "{}", e.message);
        assert!(e.message.contains("..."));
    }
}
