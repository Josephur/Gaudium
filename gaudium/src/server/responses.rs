//! The OpenAI Responses API (`POST /v1/responses` and the stored-response endpoints) as a
//! translation layer onto the chat pipeline.
//!
//! A Responses request is rewritten into the chat-completions body the chat path takes (input
//! items -> chat messages, `instructions` -> the system turn, function tools, `tool_choice`,
//! `max_output_tokens`, `reasoning.effort`), so rendering, tokenisation, tool calling, reasoning,
//! sampling, budgets and lanes are the chat path's. The output side turns the output parser's
//! deltas into output items (`reasoning`, `message` with `output_text`, `function_call`): the
//! semantic server-sent events while streaming, the same items in the non-streaming response
//! (both come from one [`Builder`], so a stream accumulates to exactly the stored response).
//! Which item is open, when a new one starts and when its closing events are owed follow the
//! verified block lifecycle (`core::tool_calls::BlockTrack`), as on the Messages endpoint.
//!
//! Each top-level request field is in one class (the same rule as the chat endpoint's table):
//! applied, ignored and reported in `gaudium.ignored`, or refused with a 400 that names it when
//! ignoring it would change the reply without a sign (`background`, `conversation`, a JSON
//! `text.format`, built-in tools, log probabilities). A field outside the table is reported as
//! unknown, never refused. This is request parsing and an outside wire format (AGENTS.md 3a,
//! second list); the store's bound and accounting are verified (`response_store.rs`).

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::anthropic::{signature_reasoning, thinking_signature};
use super::outparse::Delta;
use super::readout::jq;
use super::response_store::StoredResponse;
use crate::core::tool_calls::{BlockKind, BlockTrack};
use crate::json::JsonValue;

// ============================================================================
// Request fields
// ============================================================================

/// What the translation does with a top-level field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Applied,
    Ignored(&'static str),
    Refused(&'static str),
}

struct Field {
    name: &'static str,
    class: Class,
    /// Values that change nothing (never reported or refused); `null` changes nothing for all.
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

const END_USER: &str = "not applied: end-user ids are not recorded";

/// The table. One entry per field; a later change moves a field between classes.
const FIELDS: &[Field] = &[
    // the API's own fields the translation applies (nested parts are checked below)
    applied("model"),
    applied("input"),
    applied("instructions"),
    applied("tools"),
    applied("tool_choice"),
    applied("parallel_tool_calls"),
    applied("max_output_tokens"),
    applied("temperature"),
    applied("top_p"),
    applied("stream"),
    applied("store"),
    applied("previous_response_id"),
    applied("reasoning"),
    applied("text"),
    applied("include"),
    applied("metadata"),
    // extensions, as on the chat endpoint
    applied("seed"),
    applied("top_k"),
    applied("min_p"),
    applied("stop"),
    applied("chat_template_kwargs"),
    applied("reasoning_format"),
    applied("cache_prompt"),
    applied("coupled"),
    applied("draft_trunc"),
    applied("gaudium_method"),
    applied("gaudium_a2ui"),
    applied("gaudium_return_tokens"),
    // refused: ignoring them changes the reply without a sign
    refused(
        "background",
        "background responses are not supported yet; send background: false and keep the connection open (stream: true sends events as they come)",
        is_false,
    ),
    refused(
        "conversation",
        "conversations are not kept by this server; chain responses with previous_response_id",
        never,
    ),
    refused(
        "prompt",
        "stored prompt templates are not available on this server; send instructions and input",
        never,
    ),
    refused(
        "top_logprobs",
        "token log probabilities are not returned yet",
        is_zero,
    ),
    // ignored and reported
    ignored(
        "truncation",
        "not applied: the input is never truncated; a prompt over the context is refused",
        is_disabled,
    ),
    ignored(
        "max_tool_calls",
        "not applied: no built-in tools run on this server",
        never,
    ),
    ignored("user", END_USER, never),
    ignored("safety_identifier", END_USER, never),
    ignored(
        "prompt_cache_key",
        "not applied: prompt reuse is automatic",
        never,
    ),
    ignored(
        "prompt_cache_retention",
        "not applied: prompt reuse is automatic",
        never,
    ),
    ignored(
        "prompt_cache_options",
        "not applied: prompt reuse is automatic",
        never,
    ),
    ignored("service_tier", "not applied: there is one tier", never),
    ignored(
        "stream_options",
        "not applied: stream events carry no obfuscation",
        never,
    ),
    ignored("context_management", "not applied", never),
    ignored("moderation", "not applied", never),
    ignored(
        "background_options",
        "not applied: background responses are not supported",
        never,
    ),
];

fn never(_: &JsonValue) -> bool {
    false
}

fn is_false(v: &JsonValue) -> bool {
    matches!(v, JsonValue::Bool(false))
}

fn is_zero(v: &JsonValue) -> bool {
    v.as_f64() == Some(0.0)
}

fn is_disabled(v: &JsonValue) -> bool {
    v.as_str() == Some("disabled")
}

/// The built-in tool types of the API. They run on the server; this one runs none yet, so a
/// request that offers one is refused with the type named (server-side tools are #118 / #119: a
/// served built-in tool is taken out of this list where `convert_tools` meets it).
const BUILTIN_TOOLS: &[&str] = &[
    "file_search",
    "web_search",
    "web_search_2025_08_26",
    "web_search_preview",
    "web_search_preview_2025_03_11",
    "computer",
    "computer_use",
    "computer_use_preview",
    "mcp",
    "code_interpreter",
    "image_generation",
    "local_shell",
    "shell",
    "apply_patch",
    "tool_search",
    "programmatic_tool_calling",
];

// ============================================================================
// Translation
// ============================================================================

/// A request this server does not serve: the status, the field it names and the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub status: u16,
    pub param: Option<String>,
    pub message: String,
}

type R<T> = Result<T, Refusal>;

fn refuse<T>(param: &str, message: impl Into<String>) -> R<T> {
    Err(Refusal {
        status: 400,
        param: (!param.is_empty()).then(|| param.to_string()),
        message: message.into(),
    })
}

/// The value as an error shows it, cut to a readable length.
fn short(v: &JsonValue) -> String {
    let s = v.dump();
    match s.char_indices().nth(80) {
        Some((i, _)) => format!("{}...", &s[..i]),
        None => s,
    }
}

fn s(v: &str) -> JsonValue {
    JsonValue::Str(v.to_string())
}

fn obj(pairs: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn kind(o: &JsonValue) -> &str {
    o.get("type").and_then(JsonValue::as_str).unwrap_or("")
}

fn present<'a>(o: &'a JsonValue, key: &str) -> Option<&'a JsonValue> {
    o.get(key).filter(|v| !matches!(v, JsonValue::Null))
}

fn text_field<'a>(o: &'a JsonValue, key: &str, path: &str) -> R<&'a str> {
    o.get(key).and_then(JsonValue::as_str).map_or_else(
        || {
            refuse(
                &format!("{path}.{key}"),
                format!("{path}.{key}: a string is required"),
            )
        },
        Ok,
    )
}

/// The request context a response carries from its request to its reply and into the store:
/// the request fields the response object echoes, and the conversation's input items.
#[derive(Debug, Clone)]
pub struct Head {
    pub store: bool,
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u64>,
    /// The request's tools as it sent them, as one JSON array text. The JSON fields a response
    /// echoes are held as text, like `input`, so that [`Head::size`] counts what they hold.
    pub tools: String,
    /// `tool_choice` as sent (JSON text).
    pub tool_choice: String,
    pub parallel_tool_calls: bool,
    pub reasoning_effort: Option<String>,
    pub reasoning_summary: Option<String>,
    pub text_verbosity: Option<String>,
    /// `metadata` as sent (JSON text).
    pub metadata: String,
    /// Echoed as sent (they are not applied), each value as JSON text.
    pub echo: Vec<(&'static str, String)>,
    /// The conversation's input items as one JSON array: those of the responses it continues,
    /// then its own (held as text while the response runs; the store keeps it so).
    pub input: String,
    /// Where its own input items begin in `input`.
    pub own_from: usize,
    /// Reasoning items carry `encrypted_content` (asked for in `include`, or `store: false`).
    pub encrypted: bool,
}

impl Default for Head {
    fn default() -> Self {
        Self {
            store: false,
            previous_response_id: None,
            instructions: None,
            max_output_tokens: None,
            tools: "[]".to_string(),
            tool_choice: "\"auto\"".to_string(),
            parallel_tool_calls: true,
            reasoning_effort: None,
            reasoning_summary: None,
            text_verbosity: None,
            metadata: "{}".to_string(),
            echo: Vec::new(),
            input: "[]".to_string(),
            own_from: 0,
            encrypted: false,
        }
    }
}

/// A Responses request as the chat pipeline takes it.
#[derive(Debug)]
pub struct Translated {
    /// The chat-completions body.
    pub chat: JsonValue,
    /// Accepted, not applied.
    pub ignored: Vec<String>,
    pub head: Head,
}

impl Head {
    /// The bytes this context holds, as the in-flight account counts them: the capacity of
    /// every text it holds (every field it keeps from the request is text, none a parsed tree),
    /// the slots of its echo list, and a fixed overhead for the struct itself.
    pub fn size(&self) -> u64 {
        let opt = |o: &Option<String>| o.as_ref().map_or(0, String::capacity);
        let texts = self.input.capacity()
            + self.tools.capacity()
            + self.tool_choice.capacity()
            + self.metadata.capacity()
            + self.echo.capacity() * std::mem::size_of::<(&'static str, String)>()
            + self.echo.iter().map(|(_, v)| v.capacity()).sum::<usize>()
            + opt(&self.previous_response_id)
            + opt(&self.instructions)
            + opt(&self.reasoning_effort)
            + opt(&self.reasoning_summary)
            + opt(&self.text_verbosity);
        (texts as u64).saturating_add(std::mem::size_of::<Self>() as u64 + 64)
    }

    /// The conversation's input items (parsed from the held text).
    #[cfg(test)]
    pub fn input_items(&self) -> Vec<JsonValue> {
        match crate::json::parse_json(&self.input) {
            Ok(JsonValue::Array(items)) => items,
            _ => Vec::new(),
        }
    }
}

/// Bytes of randomness in a response id (128 bits).
const ID_RANDOM_BYTES: usize = 16;

/// A new response id: `resp_` and 32 hex digits, 128 bits read from the operating system's
/// random source, so that a caller cannot guess the id of another caller's response (an id is
/// what reads, deletes, cancels and continues a stored conversation). The ids of its items
/// carry the same digits. `Err`: the random source cannot be read.
pub fn new_response_id() -> Result<String, String> {
    use std::io::Read;
    let mut b = [0u8; ID_RANDOM_BYTES];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| format!("cannot read /dev/urandom for a response id: {e}"))?;
    let mut id = String::with_capacity(5 + 2 * ID_RANDOM_BYTES);
    id.push_str("resp_");
    for x in b {
        id.push_str(&format!("{x:02x}"));
    }
    Ok(id)
}

/// The id suffix shared by a response's item ids (`msg_<suffix>_<n>`, ...).
fn suffix(id: &str) -> &str {
    id.strip_prefix("resp_").unwrap_or(id)
}

/// Responses request -> chat body. `id`: the new response's id; `prev`: the stored response of
/// an id, for `previous_response_id`; `store_on`: the server keeps responses at all.
pub fn translate(
    body: &JsonValue,
    id: &str,
    prev: &dyn Fn(&str) -> Option<Arc<StoredResponse>>,
    store_on: bool,
) -> R<Translated> {
    let Some(entries) = body.as_object() else {
        return refuse("", "the request body must be a JSON object");
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
            Class::Refused(why) => return refuse(name, format!("{name}={}: {why}", short(value))),
        }
    }
    if body
        .get("model")
        .is_some_and(|m| !matches!(m, JsonValue::Str(_) | JsonValue::Null))
    {
        return refuse("model", "model: must be a string");
    }

    let mut head = Head {
        store: match present(body, "store") {
            None => true,
            Some(JsonValue::Bool(b)) => *b,
            Some(_) => return refuse("store", "store: must be a boolean"),
        },
        ..Head::default()
    };
    for k in [
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "service_tier",
        "max_tool_calls",
    ] {
        if let Some(v) = present(body, k) {
            head.echo.push((k, v.dump()));
        }
    }
    if let Some(m) = present(body, "metadata") {
        let Some(pairs) = m.as_object() else {
            return refuse("metadata", "metadata: must be an object of string values");
        };
        if pairs.iter().any(|(_, v)| v.as_str().is_none()) {
            return refuse("metadata", "metadata: every value must be a string");
        }
        head.metadata = m.dump();
    }

    // text: plain text only (constrained decoding is not available yet, #107)
    if let Some(t) = present(body, "text") {
        if t.as_object().is_none() {
            return refuse("text", "text: must be an object");
        }
        if let Some(f) = present(t, "format") {
            match kind(f) {
                "text" => {}
                k @ ("json_object" | "json_schema") => {
                    return refuse(
                        "text.format",
                        format!("text.format={}: constrained output ({k}) is not available yet; only {{\"type\": \"text\"}} is accepted", short(f)),
                    )
                }
                _ => return refuse("text.format", "text.format.type: must be \"text\", \"json_object\" or \"json_schema\""),
            }
        }
        if let Some(v) = present(t, "verbosity") {
            head.text_verbosity = v.as_str().map(str::to_string);
            ignored.push(format!("text.verbosity={} (not applied)", v.dump()));
        }
    }

    // reasoning: the effort goes to the chat path; there are no summaries
    let mut effort = None;
    if let Some(r) = present(body, "reasoning") {
        if r.as_object().is_none() {
            return refuse("reasoning", "reasoning: must be an object");
        }
        if let Some(e) = present(r, "effort") {
            let Some(e) = e.as_str() else {
                return refuse("reasoning.effort", "reasoning.effort: must be a string");
            };
            effort = Some(e.to_string());
        }
        for k in ["summary", "generate_summary"] {
            if let Some(v) = present(r, k) {
                head.reasoning_summary = v.as_str().map(str::to_string);
                ignored.push(format!(
                    "reasoning.{k}={} (no summaries are written; the reasoning text is in the reasoning item's content)",
                    v.dump()
                ));
            }
        }
        for k in ["context", "mode"] {
            if let Some(v) = present(r, k) {
                ignored.push(format!("reasoning.{k}={} (not applied)", v.dump()));
            }
        }
    }
    head.reasoning_effort = effort.clone();

    // include: reasoning.encrypted_content is served; the rest name items this server never makes
    let mut encrypted = !head.store;
    if let Some(inc) = present(body, "include") {
        let Some(list) = inc.as_array() else {
            return refuse("include", "include: must be an array of strings");
        };
        for v in list {
            match v.as_str() {
                Some("reasoning.encrypted_content") => encrypted = true,
                Some("message.output_text.logprobs") => {
                    return refuse("include", "include=message.output_text.logprobs: token log probabilities are not returned yet")
                }
                Some(other) => ignored.push(format!("include={other} (no such items are made by this server)")),
                None => return refuse("include", "include: must be an array of strings"),
            }
        }
    }
    head.encrypted = encrypted;

    if let Some(i) = present(body, "instructions") {
        let Some(t) = i.as_str() else {
            return refuse("instructions", "instructions: must be a string");
        };
        head.instructions = Some(t.to_string());
    }

    // the conversation: the stored one it continues, then this request's input
    let mut items: Vec<JsonValue> = Vec::new();
    if let Some(p) = present(body, "previous_response_id") {
        let Some(pid) = p.as_str() else {
            return refuse(
                "previous_response_id",
                "previous_response_id: must be a string",
            );
        };
        let Some(prior) = prev(pid) else {
            let why = if store_on {
                ""
            } else {
                " (this server keeps no responses: it runs with --response-store 0)"
            };
            return refuse(
                "previous_response_id",
                format!("Previous response with id '{pid}' not found.{why}"),
            );
        };
        items.extend(prior.input_items());
        items.extend(prior.output_items());
        head.previous_response_id = Some(pid.to_string());
    }
    head.own_from = items.len();
    let own = own_items(body.get("input"), id)?;
    items.extend(own);

    // tools
    let (tools, strict_note) = convert_tools(body.get("tools"))?;
    if strict_note {
        ignored.push("tools[].strict (function arguments are not constrained to their schema while decoding)".into());
    }
    if let Some(t) = present(body, "tools") {
        if t.as_array().is_some() {
            head.tools = t.dump();
        }
    }
    let names: Vec<String> = tools
        .iter()
        .filter_map(|t| t.get("function")?.get("name")?.as_str().map(str::to_string))
        .collect();

    let mut chat: Vec<(&str, JsonValue)> = vec![("messages", JsonValue::Array(messages(&head, &items)?))];
    let mut offered = tools;
    if let Some(tc) = present(body, "tool_choice") {
        head.tool_choice = tc.dump();
        let (choice, allowed) = convert_tool_choice(tc, &names)?;
        if let Some(allowed) = allowed {
            offered.retain(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(JsonValue::as_str)
                    .is_some_and(|n| allowed.iter().any(|a| a == n))
            });
        }
        chat.push(("tool_choice", choice));
    }
    if !offered.is_empty() {
        chat.push(("tools", JsonValue::Array(offered)));
    }
    if let Some(p) = present(body, "parallel_tool_calls") {
        let Some(b) = p.as_bool() else {
            return refuse(
                "parallel_tool_calls",
                "parallel_tool_calls: must be a boolean",
            );
        };
        head.parallel_tool_calls = b;
        chat.push(("parallel_tool_calls", JsonValue::Bool(b)));
    }
    if let Some(mt) = present(body, "max_output_tokens") {
        match mt.as_u64().filter(|&n| n > 0) {
            Some(n) if !matches!(mt, JsonValue::Float(_)) => {
                head.max_output_tokens = Some(n);
                chat.push(("max_tokens", mt.clone()));
            }
            _ => {
                return refuse(
                    "max_output_tokens",
                    "max_output_tokens: must be a positive integer",
                )
            }
        }
    }
    for k in ["temperature", "top_p"] {
        if let Some(v) = present(body, k) {
            if v.as_f64().is_none() {
                return refuse(k, format!("{k}: must be a number"));
            }
            chat.push((k, v.clone()));
        }
    }
    if let Some(e) = effort {
        chat.push(("reasoning_effort", JsonValue::Str(e)));
    }
    if let Some(st) = present(body, "stream") {
        if st.as_bool().is_none() {
            return refuse("stream", "stream: must be a boolean");
        }
        chat.push(("stream", st.clone()));
    }
    for k in [
        "model",
        "seed",
        "top_k",
        "min_p",
        "stop",
        "chat_template_kwargs",
        "reasoning_format",
        "cache_prompt",
        "coupled",
        "draft_trunc",
        "gaudium_method",
        "gaudium_a2ui",
        "gaudium_return_tokens",
    ] {
        if let Some(v) = present(body, k) {
            chat.push((k, v.clone()));
        }
    }
    // the chat path names its reply with the response id
    chat.push(("gaudium_response_id", s(id)));
    head.input = JsonValue::Array(items).dump();
    Ok(Translated {
        chat: obj(chat),
        ignored,
        head,
    })
}

/// This request's input as items, each with an id: a string is one user message.
fn own_items(input: Option<&JsonValue>, id: &str) -> R<Vec<JsonValue>> {
    let sfx = suffix(id);
    let with_id = |item: &JsonValue, i: usize| -> JsonValue {
        let mut it = item.clone();
        if it.get("id").and_then(JsonValue::as_str).is_none() {
            let prefix = match item_kind(item) {
                "function_call" => "fc",
                "function_call_output" => "fco",
                "reasoning" => "rs",
                _ => "msg",
            };
            if let JsonValue::Object(e) = &mut it {
                e.retain(|(k, _)| k != "id");
                e.push(("id".into(), s(&format!("{prefix}_in{sfx}_{i}"))));
            }
        }
        it
    };
    match input {
        None | Some(JsonValue::Null) => Ok(Vec::new()),
        Some(JsonValue::Str(t)) => Ok(vec![with_id(
            &obj(vec![
                ("type", s("message")),
                ("role", s("user")),
                (
                    "content",
                    JsonValue::Array(vec![obj(vec![("type", s("input_text")), ("text", s(t))])]),
                ),
            ]),
            0,
        )]),
        Some(JsonValue::Array(items)) => items
            .iter()
            .enumerate()
            .map(|(i, it)| {
                if it.as_object().is_none() {
                    return refuse(
                        &format!("input[{i}]"),
                        format!("input[{i}]: must be an object"),
                    );
                }
                Ok(with_id(it, i))
            })
            .collect(),
        Some(_) => refuse(
            "input",
            "input: must be a string or an array of input items",
        ),
    }
}

/// An item's type: `type`, or `message` for an item that has only a role.
fn item_kind(item: &JsonValue) -> &str {
    match item.get("type").and_then(JsonValue::as_str) {
        Some(t) => t,
        None if item.get("role").is_some() => "message",
        None => "",
    }
}

/// The function tools of the request, in the chat shape; whether any asked for `strict`.
fn convert_tools(tools: Option<&JsonValue>) -> R<(Vec<JsonValue>, bool)> {
    let list = match tools {
        None | Some(JsonValue::Null) => return Ok((Vec::new(), false)),
        Some(JsonValue::Array(l)) => l,
        Some(_) => return refuse("tools", "tools: must be an array"),
    };
    let mut out = Vec::new();
    let mut strict = false;
    for (i, t) in list.iter().enumerate() {
        let p = format!("tools[{i}]");
        if t.as_object().is_none() {
            return refuse(&p, format!("{p}: must be an object"));
        }
        match kind(t) {
            "function" => {}
            ty if BUILTIN_TOOLS.contains(&ty) => {
                return refuse(
                    &format!("{p}.type"),
                    format!("{p}: '{ty}' is a built-in tool; this server runs no built-in tools yet. Define it as a function tool (name, description, parameters) and run it in the client"),
                )
            }
            "" => return refuse(&format!("{p}.type"), format!("{p}.type: a string is required")),
            ty => {
                return refuse(
                    &format!("{p}.type"),
                    format!("{p}: '{ty}' tools are not supported by this server; only function tools are"),
                )
            }
        }
        let mut f = vec![("name", s(text_field(t, "name", &p)?))];
        if let Some(d) = t.get("description").and_then(JsonValue::as_str) {
            f.push(("description", s(d)));
        }
        f.push((
            "parameters",
            match present(t, "parameters") {
                None => obj(vec![
                    ("type", s("object")),
                    ("properties", JsonValue::Object(Vec::new())),
                ]),
                Some(schema @ JsonValue::Object(_)) => schema.clone(),
                Some(_) => {
                    return refuse(
                        &format!("{p}.parameters"),
                        format!("{p}.parameters: must be a JSON schema object"),
                    )
                }
            },
        ));
        strict |= t.get("strict").and_then(JsonValue::as_bool) == Some(true);
        out.push(obj(vec![("type", s("function")), ("function", obj(f))]));
    }
    Ok((out, strict))
}

/// `tool_choice` in the chat shape, and the names an `allowed_tools` choice keeps offered.
fn convert_tool_choice(tc: &JsonValue, names: &[String]) -> R<(JsonValue, Option<Vec<String>>)> {
    let named = |name: &str, param: &str| -> R<()> {
        if names.iter().any(|n| n == name) {
            Ok(())
        } else {
            refuse(
                param,
                format!("{param}: the function '{name}' is not in 'tools'"),
            )
        }
    };
    match tc {
        JsonValue::Str(w) if matches!(w.as_str(), "auto" | "none" | "required") => {
            Ok((tc.clone(), None))
        }
        JsonValue::Str(_) => refuse("tool_choice", "tool_choice: must be \"auto\", \"none\", \"required\" or an object"),
        JsonValue::Object(_) => match kind(tc) {
            "function" => {
                let name = text_field(tc, "name", "tool_choice")?;
                named(name, "tool_choice.name")?;
                Ok((
                    obj(vec![("type", s("function")), ("function", obj(vec![("name", s(name))]))]),
                    None,
                ))
            }
            "allowed_tools" => {
                let mode = tc.get("mode").and_then(JsonValue::as_str).unwrap_or("auto");
                if !matches!(mode, "auto" | "required") {
                    return refuse("tool_choice.mode", "tool_choice.mode: must be \"auto\" or \"required\"");
                }
                let Some(list) = tc.get("tools").and_then(JsonValue::as_array) else {
                    return refuse("tool_choice.tools", "tool_choice.tools: an array of tools is required");
                };
                let mut keep = Vec::new();
                for (i, t) in list.iter().enumerate() {
                    let p = format!("tool_choice.tools[{i}]");
                    if kind(t) != "function" {
                        return refuse(&p, format!("{p}: only function tools can be allowed on this server"));
                    }
                    let name = text_field(t, "name", &p)?;
                    named(name, &p)?;
                    keep.push(name.to_string());
                }
                Ok((s(mode), Some(keep)))
            }
            ty if BUILTIN_TOOLS.contains(&ty) || ty == "custom" => refuse(
                "tool_choice.type",
                format!("tool_choice: '{ty}' names a tool this server does not run; choose a function tool"),
            ),
            _ => refuse("tool_choice.type", "tool_choice.type: must be \"function\" or \"allowed_tools\""),
        },
        _ => refuse("tool_choice", "tool_choice: must be a string or an object"),
    }
}

/// An assistant turn being put together from consecutive output items.
#[derive(Default)]
struct Turn {
    text: String,
    reasoning: String,
    calls: Vec<JsonValue>,
}

impl Turn {
    fn is_empty(&self) -> bool {
        self.text.is_empty() && self.reasoning.is_empty() && self.calls.is_empty()
    }

    fn flush(&mut self, msgs: &mut Vec<JsonValue>) {
        if self.is_empty() {
            return;
        }
        let t = std::mem::take(self);
        let mut m = vec![
            ("role", s("assistant")),
            ("content", JsonValue::Str(t.text)),
        ];
        if !t.reasoning.is_empty() {
            m.push(("reasoning_content", JsonValue::Str(t.reasoning)));
        }
        if !t.calls.is_empty() {
            m.push(("tool_calls", JsonValue::Array(t.calls)));
        }
        msgs.push(obj(m));
    }
}

/// A user, system or developer message's content in the chat shape: a string, or its parts.
fn user_content(c: &JsonValue, path: &str) -> R<JsonValue> {
    let parts = match c {
        JsonValue::Str(_) => return Ok(c.clone()),
        JsonValue::Array(p) => p,
        _ => {
            return refuse(
                path,
                format!("{path}: must be a string or an array of content parts"),
            )
        }
    };
    let mut out = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let pp = format!("{path}[{i}]");
        if p.as_object().is_none() {
            return refuse(&pp, format!("{pp}: must be an object"));
        }
        match kind(p) {
            "input_text" | "output_text" | "text" => {
                out.push(obj(vec![("type", s("text")), ("text", s(text_field(p, "text", &pp)?))]));
            }
            "input_image" => match present(p, "image_url").and_then(JsonValue::as_str) {
                Some(url) => out.push(obj(vec![("type", s("input_image")), ("image_url", s(url))])),
                None if present(p, "file_id").is_some() => {
                    return refuse(
                        &format!("{pp}.file_id"),
                        format!("{pp}: file ids are not supported (this server keeps no files); send image_url as a data: URL"),
                    )
                }
                None => return refuse(&format!("{pp}.image_url"), format!("{pp}.image_url: a string is required")),
            },
            "input_file" => {
                return refuse(
                    &pp,
                    format!("{pp}: input_file parts are not supported by this server; send the file's text as input_text"),
                )
            }
            "input_audio" => return refuse(&pp, format!("{pp}: audio input is not supported by this server")),
            k => return refuse(&format!("{pp}.type"), format!("{pp}.type: '{k}' content parts are not supported here")),
        }
    }
    Ok(JsonValue::Array(out))
}

/// An assistant message's text (its output_text and refusal parts).
fn assistant_text(c: &JsonValue, path: &str) -> R<String> {
    match c {
        JsonValue::Str(t) => Ok(t.clone()),
        JsonValue::Array(parts) => {
            let mut t = String::new();
            for (i, p) in parts.iter().enumerate() {
                let pp = format!("{path}[{i}]");
                match kind(p) {
                    "output_text" | "input_text" | "text" => {
                        t.push_str(text_field(p, "text", &pp)?)
                    }
                    "refusal" => t.push_str(text_field(p, "refusal", &pp)?),
                    k => {
                        return refuse(
                            &format!("{pp}.type"),
                            format!(
                                "{pp}.type: '{k}' parts are not supported in assistant messages"
                            ),
                        )
                    }
                }
            }
            Ok(t)
        }
        _ => refuse(
            path,
            format!("{path}: must be a string or an array of content parts"),
        ),
    }
}

/// The text of a system or developer message (text parts only).
fn system_text(c: &JsonValue, path: &str) -> R<String> {
    match user_content(c, path)? {
        JsonValue::Str(t) => Ok(t),
        JsonValue::Array(parts) => parts
            .iter()
            .enumerate()
            .map(|(i, p)| match p.get("text").and_then(JsonValue::as_str) {
                Some(t) => Ok(t.to_string()),
                None => refuse(
                    &format!("{path}[{i}]"),
                    format!("{path}[{i}]: system and developer messages take text only"),
                ),
            })
            .collect::<R<Vec<_>>>()
            .map(|v| v.concat()),
        _ => Ok(String::new()),
    }
}

/// The reasoning a reasoning item carries: its reasoning text, else what its opaque
/// `encrypted_content` holds (ours carries the text), else its summary.
fn reasoning_text(item: &JsonValue) -> String {
    let join = |key: &str| -> String {
        item.get(key)
            .and_then(JsonValue::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(JsonValue::as_str))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            })
            .unwrap_or_default()
    };
    let content = join("content");
    if !content.is_empty() {
        return content;
    }
    if let Some(r) = item
        .get("encrypted_content")
        .and_then(JsonValue::as_str)
        .and_then(signature_reasoning)
    {
        return r;
    }
    join("summary")
}

fn is_system_message(item: &JsonValue) -> bool {
    item_kind(item) == "message"
        && matches!(
            item.get("role").and_then(JsonValue::as_str),
            Some("system" | "developer")
        )
}

/// The chat messages of a conversation: the instructions and any leading system or developer
/// messages as the system turn, then the items in order (consecutive reasoning, assistant
/// message and function_call items are one assistant turn).
fn messages(head: &Head, items: &[JsonValue]) -> R<Vec<JsonValue>> {
    // paths name the request's own input; items of a stored response are named by their id
    let path = |i: usize| -> String {
        if i >= head.own_from {
            format!("input[{}]", i - head.own_from)
        } else {
            let id = items[i]
                .get("id")
                .and_then(JsonValue::as_str)
                .unwrap_or("?");
            format!("previous_response_id (item {id})")
        }
    };
    let mut msgs = Vec::new();
    let mut sys = head.instructions.clone().unwrap_or_default();
    let mut i = 0;
    while i < items.len() && is_system_message(&items[i]) {
        let t = system_text(
            items[i].get("content").unwrap_or(&JsonValue::Null),
            &format!("{}.content", path(i)),
        )?;
        if !sys.is_empty() && !t.is_empty() {
            sys.push_str("\n\n");
        }
        sys.push_str(&t);
        i += 1;
    }
    if !sys.is_empty() {
        msgs.push(obj(vec![
            ("role", s("system")),
            ("content", JsonValue::Str(sys)),
        ]));
    }
    let mut turn = Turn::default();
    for (i, item) in items.iter().enumerate().skip(i) {
        let p = path(i);
        match item_kind(item) {
            "message" => {
                let role = item.get("role").and_then(JsonValue::as_str).unwrap_or("");
                let Some(content) = item.get("content") else {
                    return refuse(&format!("{p}.content"), format!("{p}.content: Field required"));
                };
                let cp = format!("{p}.content");
                match role {
                    "assistant" => {
                        if !turn.text.is_empty() || !turn.calls.is_empty() {
                            turn.flush(&mut msgs);
                        }
                        turn.text.push_str(&assistant_text(content, &cp)?);
                    }
                    "user" => {
                        turn.flush(&mut msgs);
                        msgs.push(obj(vec![("role", s("user")), ("content", user_content(content, &cp)?)]));
                    }
                    "system" | "developer" => {
                        turn.flush(&mut msgs);
                        msgs.push(obj(vec![("role", s("system")), ("content", JsonValue::Str(system_text(content, &cp)?))]));
                    }
                    _ => {
                        return refuse(
                            &format!("{p}.role"),
                            format!("{p}.role: must be \"user\", \"assistant\", \"system\" or \"developer\""),
                        )
                    }
                }
            }
            "reasoning" => {
                if !turn.text.is_empty() || !turn.calls.is_empty() {
                    turn.flush(&mut msgs);
                }
                let r = reasoning_text(item);
                if !r.is_empty() {
                    if !turn.reasoning.is_empty() {
                        turn.reasoning.push_str("\n\n");
                    }
                    turn.reasoning.push_str(&r);
                }
            }
            "function_call" => {
                let args = match item.get("arguments") {
                    Some(JsonValue::Str(a)) => a.clone(),
                    None | Some(JsonValue::Null) => "{}".into(),
                    Some(_) => return refuse(&format!("{p}.arguments"), format!("{p}.arguments: must be a JSON string")),
                };
                turn.calls.push(obj(vec![
                    ("id", s(text_field(item, "call_id", &p)?)),
                    ("type", s("function")),
                    (
                        "function",
                        obj(vec![("name", s(text_field(item, "name", &p)?)), ("arguments", JsonValue::Str(args))]),
                    ),
                ]));
            }
            "function_call_output" => {
                turn.flush(&mut msgs);
                let content = match item.get("output") {
                    Some(o @ JsonValue::Array(_)) => user_content(o, &format!("{p}.output"))?,
                    Some(JsonValue::Str(t)) => JsonValue::Str(t.clone()),
                    _ => return refuse(&format!("{p}.output"), format!("{p}.output: a string or an array of parts is required")),
                };
                msgs.push(obj(vec![
                    ("role", s("tool")),
                    ("tool_call_id", s(text_field(item, "call_id", &p)?)),
                    ("content", content),
                ]));
            }
            "item_reference" => {
                return refuse(
                    &p,
                    format!("{p}: item references are not supported by this server; send the item itself, or continue with previous_response_id"),
                )
            }
            "" => return refuse(&format!("{p}.type"), format!("{p}.type: a string is required")),
            k => {
                return refuse(
                    &format!("{p}.type"),
                    format!("{p}.type: '{k}' items are not supported by this server"),
                )
            }
        }
    }
    turn.flush(&mut msgs);
    if msgs.is_empty() {
        return refuse(
            "input",
            "input: Field required (a string or an array of input items)",
        );
    }
    Ok(msgs)
}

// ============================================================================
// Output
// ============================================================================

/// `{"error":{"message","type","param","code"}}`.
pub fn error_json(message: &str, err_type: &str, param: Option<&str>) -> String {
    format!(
        "{{\"error\":{{\"message\":{},\"type\":{},\"param\":{},\"code\":null}}}}",
        jq(message),
        jq(err_type),
        param.map_or_else(|| "null".to_string(), jq)
    )
}

/// The `error` event of a stream that failed before its response began.
pub fn stream_error(message: &str) -> String {
    sse(
        "error",
        &format!(
            "{{\"type\":\"error\",\"code\":\"server_error\",\"message\":{},\"param\":null,\"sequence_number\":0}}",
            jq(message)
        ),
    )
}

fn sse(event: &str, data: &str) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// The usage object: `prompt_tokens` with `reused` of them taken over from the request before
/// (reported as cached), `output_tokens` generated, `reasoning_tokens` of them reasoning.
pub fn usage_json(
    prompt_tokens: u64,
    reused: u64,
    output_tokens: u64,
    reasoning_tokens: u64,
) -> String {
    format!(
        "{{\"input_tokens\":{prompt_tokens},\"input_tokens_details\":{{\"cached_tokens\":{}}},\
         \"output_tokens\":{output_tokens},\"output_tokens_details\":{{\"reasoning_tokens\":{}}},\
         \"total_tokens\":{}}}",
        crate::verified::pc_reused_rows(prompt_tokens, reused),
        reasoning_tokens.min(output_tokens),
        prompt_tokens.saturating_add(output_tokens),
    )
}

/// How a response ended, as its `status` tells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    Completed,
    /// The output limit (or the context) ended it.
    Incomplete,
    Cancelled,
    Failed,
}

impl Ending {
    fn status(self) -> &'static str {
        match self {
            Ending::Completed => "completed",
            Ending::Incomplete => "incomplete",
            Ending::Cancelled => "cancelled",
            Ending::Failed => "failed",
        }
    }

    /// The status of the item open at the end.
    fn item_status(self) -> &'static str {
        match self {
            Ending::Completed => "completed",
            _ => "incomplete",
        }
    }

    /// The terminal event's type.
    fn event(self) -> &'static str {
        match self {
            Ending::Completed => "response.completed",
            Ending::Incomplete => "response.incomplete",
            // the API has no event of its own for a cancelled response
            Ending::Cancelled => "response.incomplete",
            Ending::Failed => "response.failed",
        }
    }
}

enum Item {
    Reasoning {
        id: String,
        text: String,
    },
    Message {
        id: String,
        text: String,
    },
    Call {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

/// Output items from parser deltas, in arrival order (reasoning, message, function_call; a kind
/// change closes the open item), with the stream's semantic events. Each [`Builder::feed`]
/// returns the events it produced (written only when streaming).
pub struct Builder {
    head: Arc<Head>,
    id: String,
    model: String,
    created: u64,
    temperature: f32,
    top_p: f32,
    seq: u64,
    items: Vec<Item>,
    /// Each item's status.
    status: Vec<&'static str>,
    track: BlockTrack,
    /// The open item's text (its arguments for a call).
    buf: String,
}

impl Builder {
    pub fn new(
        head: Arc<Head>,
        id: &str,
        model: &str,
        created: u64,
        temperature: f32,
        top_p: f32,
    ) -> Self {
        Self {
            head,
            id: id.to_string(),
            model: model.to_string(),
            created,
            temperature,
            top_p,
            seq: 0,
            items: Vec::new(),
            status: Vec::new(),
            track: BlockTrack::new(),
            buf: String::new(),
        }
    }

    pub fn head(&self) -> &Head {
        &self.head
    }

    /// One event: `fields` are the members after `type` and `sequence_number`.
    fn ev(&mut self, ty: &str, fields: &str) -> String {
        let n = self.seq;
        self.seq += 1;
        let sep = if fields.is_empty() { "" } else { "," };
        sse(
            ty,
            &format!(
                "{{\"type\":{},\"sequence_number\":{n}{sep}{fields}}}",
                jq(ty)
            ),
        )
    }

    /// `response.created` and `response.in_progress`.
    pub fn start(&mut self) -> String {
        let r = self.response(Ending::Completed, None, "null", self.head.store, "", true);
        let mut out = self.ev("response.created", &format!("\"response\":{r}"));
        out.push_str(&self.ev("response.in_progress", &format!("\"response\":{r}")));
        out
    }

    fn index(&self) -> usize {
        self.items.len().saturating_sub(1)
    }

    fn item_json(&self, i: usize) -> String {
        let st = self.status[i];
        match &self.items[i] {
            Item::Reasoning { id, text } => {
                let enc = if self.head.encrypted {
                    format!(",\"encrypted_content\":{}", jq(&thinking_signature(text)))
                } else {
                    String::new()
                };
                let content = if text.is_empty() && st == "in_progress" {
                    "[]".to_string()
                } else {
                    format!("[{}]", reasoning_part(text))
                };
                format!(
                    "{{\"id\":{},\"type\":\"reasoning\",\"status\":\"{st}\",\"summary\":[],\"content\":{content}{enc}}}",
                    jq(id)
                )
            }
            Item::Message { id, text } => {
                let content = if st == "in_progress" {
                    "[]".to_string()
                } else {
                    format!("[{}]", text_part(text))
                };
                format!(
                    "{{\"id\":{},\"type\":\"message\",\"status\":\"{st}\",\"role\":\"assistant\",\"content\":{content}}}",
                    jq(id)
                )
            }
            Item::Call { id, call_id, name, arguments } => format!(
                "{{\"id\":{},\"type\":\"function_call\",\"status\":\"{st}\",\"call_id\":{},\"name\":{},\"arguments\":{}}}",
                jq(id),
                jq(call_id),
                jq(name),
                jq(arguments)
            ),
        }
    }

    fn open(&mut self, kind: BlockKind, item: Item) -> String {
        let mut out = self.close_item("completed");
        self.items.push(item);
        self.status.push("in_progress");
        self.track.opened(kind);
        self.buf.clear();
        let i = self.index();
        let item = self.item_json(i);
        out.push_str(&self.ev(
            "response.output_item.added",
            &format!("\"output_index\":{i},\"item\":{item}"),
        ));
        let part = match kind {
            BlockKind::BkThinking => Some(reasoning_part("")),
            BlockKind::BkText => Some(text_part("")),
            _ => None,
        };
        if let Some(part) = part {
            let item_id = jq(self.item_id(i));
            out.push_str(&self.ev(
                "response.content_part.added",
                &format!("\"item_id\":{item_id},\"output_index\":{i},\"content_index\":0,\"part\":{part}"),
            ));
        }
        out
    }

    fn item_id(&self, i: usize) -> &str {
        match &self.items[i] {
            Item::Reasoning { id, .. } | Item::Message { id, .. } | Item::Call { id, .. } => id,
        }
    }

    /// Closes the open item with `status` (its final text is recorded; its closing events are
    /// owed once).
    fn close_item(&mut self, status: &'static str) -> String {
        if !self.track.close() {
            return String::new();
        }
        let i = self.index();
        let buf = std::mem::take(&mut self.buf);
        let item_id = jq(self.item_id(i));
        let mut out = String::new();
        let head = format!("\"item_id\":{item_id},\"output_index\":{i}");
        match self.items.last_mut() {
            Some(Item::Reasoning { text, .. }) => {
                *text = buf.clone();
                out.push_str(&self.ev(
                    "response.reasoning_text.done",
                    &format!("{head},\"content_index\":0,\"text\":{}", jq(&buf)),
                ));
                out.push_str(&self.ev(
                    "response.content_part.done",
                    &format!(
                        "{head},\"content_index\":0,\"part\":{}",
                        reasoning_part(&buf)
                    ),
                ));
            }
            Some(Item::Message { text, .. }) => {
                *text = buf.clone();
                out.push_str(&self.ev(
                    "response.output_text.done",
                    &format!(
                        "{head},\"content_index\":0,\"text\":{},\"logprobs\":[]",
                        jq(&buf)
                    ),
                ));
                out.push_str(&self.ev(
                    "response.content_part.done",
                    &format!("{head},\"content_index\":0,\"part\":{}", text_part(&buf)),
                ));
            }
            Some(Item::Call { arguments, .. }) => {
                // the streamed text is the call's arguments: a JSON object, or nothing
                *arguments = if buf.trim().is_empty() {
                    "{}".into()
                } else {
                    buf
                };
                let args = jq(arguments);
                out.push_str(&self.ev(
                    "response.function_call_arguments.done",
                    &format!("{head},\"arguments\":{args}"),
                ));
            }
            None => return out,
        }
        self.status[i] = status;
        let item = self.item_json(i);
        out.push_str(&self.ev(
            "response.output_item.done",
            &format!("\"output_index\":{i},\"item\":{item}"),
        ));
        out
    }

    fn delta(&mut self, ty: &str, text: &str, content_index: bool) -> String {
        let i = self.index();
        let item_id = jq(self.item_id(i));
        let ci = if content_index {
            ",\"content_index\":0"
        } else {
            ""
        };
        let lp = if ty == "response.output_text.delta" {
            ",\"logprobs\":[]"
        } else {
            ""
        };
        self.ev(
            ty,
            &format!(
                "\"item_id\":{item_id},\"output_index\":{i}{ci},\"delta\":{}{lp}",
                jq(text)
            ),
        )
    }

    pub fn feed(&mut self, d: &Delta) -> String {
        let mut out = String::new();
        let sfx = suffix(&self.id).to_string();
        if !d.reasoning.is_empty() {
            if self.track.must_open(BlockKind::BkThinking) {
                let id = format!("rs_{sfx}_{}", self.items.len());
                out.push_str(&self.open(
                    BlockKind::BkThinking,
                    Item::Reasoning {
                        id,
                        text: String::new(),
                    },
                ));
            }
            self.buf.push_str(&d.reasoning);
            out.push_str(&self.delta("response.reasoning_text.delta", &d.reasoning, true));
        }
        if !d.content.is_empty() {
            if self.track.must_open(BlockKind::BkText) {
                let id = format!("msg_{sfx}_{}", self.items.len());
                out.push_str(&self.open(
                    BlockKind::BkText,
                    Item::Message {
                        id,
                        text: String::new(),
                    },
                ));
            }
            self.buf.push_str(&d.content);
            out.push_str(&self.delta("response.output_text.delta", &d.content, true));
        }
        for td in &d.tool_deltas {
            // a header starts the call's own item
            if td.header && self.track.must_open(BlockKind::BkTool) {
                let call_id = td.id.clone();
                let id = format!("fc_{}", call_id.strip_prefix("call_").unwrap_or(&call_id));
                out.push_str(&self.open(
                    BlockKind::BkTool,
                    Item::Call {
                        id,
                        call_id,
                        name: td.name.clone(),
                        arguments: String::new(),
                    },
                ));
            }
            if self.track.open() != BlockKind::BkTool || td.arguments.is_empty() {
                continue;
            }
            self.buf.push_str(&td.arguments);
            out.push_str(&self.delta(
                "response.function_call_arguments.delta",
                &td.arguments,
                false,
            ));
        }
        out
    }

    /// Closes the open item as the response ends (`ending` sets its status).
    pub fn close(&mut self, ending: Ending) -> String {
        self.close_item(ending.item_status())
    }

    /// The output items as JSON values (after [`Builder::close`]).
    #[cfg(test)]
    pub fn output_items(&self) -> Vec<JsonValue> {
        (0..self.items.len())
            .filter_map(|i| crate::json::parse_json(&self.item_json(i)).ok())
            .collect()
    }

    /// The response object. `usage`: its JSON (`null` while in progress); `extra`: further
    /// members as `,"key":value` text; `in_progress`: the object of `response.created`.
    pub fn response(
        &self,
        ending: Ending,
        error: Option<&str>,
        usage: &str,
        stored: bool,
        extra: &str,
        in_progress: bool,
    ) -> String {
        let h = &*self.head;
        let status = if in_progress {
            "in_progress"
        } else {
            ending.status()
        };
        let output: Vec<String> = if in_progress {
            Vec::new()
        } else {
            (0..self.items.len()).map(|i| self.item_json(i)).collect()
        };
        let completed_at = if in_progress || ending != Ending::Completed {
            "null".to_string()
        } else {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
                .to_string()
        };
        let error = match error {
            Some(m) if !in_progress => {
                format!("{{\"code\":\"server_error\",\"message\":{}}}", jq(m))
            }
            _ => "null".into(),
        };
        let incomplete = if !in_progress && ending == Ending::Incomplete {
            "{\"reason\":\"max_output_tokens\"}"
        } else {
            "null"
        };
        let opt = |v: &Option<String>| v.as_ref().map_or_else(|| "null".to_string(), |x| jq(x));
        let opt_n = |v: Option<u64>| v.map_or_else(|| "null".to_string(), |x| x.to_string());
        let echo: String = h
            .echo
            .iter()
            .map(|(k, v)| format!(",{}:{v}", jq(k)))
            .collect();
        let verbosity = h
            .text_verbosity
            .as_ref()
            .map_or_else(String::new, |v| format!(",\"verbosity\":{}", jq(v)));
        format!(
            "{{\"id\":{},\"object\":\"response\",\"created_at\":{},\"status\":\"{status}\",\"background\":false,\
             \"completed_at\":{completed_at},\"error\":{error},\"incomplete_details\":{incomplete},\
             \"instructions\":{},\"max_output_tokens\":{},\"model\":{},\"output\":[{}],\
             \"parallel_tool_calls\":{},\"previous_response_id\":{},\
             \"reasoning\":{{\"effort\":{},\"summary\":{}}},\"store\":{stored},\
             \"temperature\":{},\"text\":{{\"format\":{{\"type\":\"text\"}}{verbosity}}},\
             \"tool_choice\":{},\"tools\":{},\"top_logprobs\":0,\"top_p\":{},\
             \"truncation\":\"disabled\",\"usage\":{usage},\"metadata\":{}{echo}{extra}}}",
            jq(&self.id),
            self.created,
            opt(&h.instructions),
            opt_n(h.max_output_tokens),
            jq(&self.model),
            output.join(","),
            h.parallel_tool_calls,
            opt(&h.previous_response_id),
            opt(&h.reasoning_effort),
            opt(&h.reasoning_summary),
            num(self.temperature),
            h.tool_choice,
            h.tools,
            num(self.top_p),
            h.metadata,
        )
    }

    /// The terminal event (`response.completed`, `.incomplete`, `.cancelled` or `.failed`)
    /// carrying the response object `response`.
    pub fn finish(&mut self, ending: Ending, response: &str) -> String {
        self.ev(ending.event(), &format!("\"response\":{response}"))
    }
}

/// A sampling parameter as JSON (three decimals, as given within float precision).
fn num(v: f32) -> String {
    let r = (f64::from(v) * 1000.0).round() / 1000.0;
    JsonValue::Float(r).dump()
}

fn text_part(text: &str) -> String {
    format!(
        "{{\"type\":\"output_text\",\"text\":{},\"annotations\":[],\"logprobs\":[]}}",
        jq(text)
    )
}

fn reasoning_part(text: &str) -> String {
    format!("{{\"type\":\"reasoning_text\",\"text\":{}}}", jq(text))
}

/// `GET /v1/responses/{id}/input_items`: a page of the response's own input items. `order`
/// "asc" or "desc" (the default), `after` an item id, `limit` 1..=100 (default 20).
pub fn input_items_page(
    r: &StoredResponse,
    after: Option<&str>,
    limit: Option<&str>,
    order: Option<&str>,
) -> Result<String, String> {
    let limit = match limit {
        None => 20,
        Some(l) => match l.parse::<usize>() {
            Ok(n) if (1..=100).contains(&n) => n,
            _ => return Err("limit: must be an integer from 1 to 100".into()),
        },
    };
    let all = r.input_items();
    let mut items: Vec<&JsonValue> = all.iter().skip(r.own_from).collect();
    match order.unwrap_or("desc") {
        "desc" => items.reverse(),
        "asc" => {}
        _ => return Err("order: must be \"asc\" or \"desc\"".into()),
    }
    let id_of = |i: &JsonValue| {
        i.get("id")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_string()
    };
    let start = match after {
        None => 0,
        Some(a) => match items.iter().position(|i| id_of(i) == a) {
            Some(p) => p + 1,
            None => return Err(format!("after: no input item '{a}' in this response")),
        },
    };
    let page: Vec<&JsonValue> = items.iter().skip(start).take(limit).copied().collect();
    let has_more = start + page.len() < items.len();
    let first = page
        .first()
        .map_or_else(|| "null".to_string(), |i| jq(&id_of(i)));
    let last = page
        .last()
        .map_or_else(|| "null".to_string(), |i| jq(&id_of(i)));
    let data: Vec<String> = page.iter().map(|i| i.dump()).collect();
    Ok(format!(
        "{{\"object\":\"list\",\"data\":[{}],\"first_id\":{first},\"last_id\":{last},\"has_more\":{has_more}}}",
        data.join(",")
    ))
}

#[cfg(test)]
mod tests {
    use super::super::outparse::ToolDelta;
    use super::*;
    use crate::json::parse_json;

    #[test]
    fn response_ids_are_128_random_bits() {
        let ids: Vec<String> = (0..64).map(|_| new_response_id().unwrap()).collect();
        for id in &ids {
            let hex = id.strip_prefix("resp_").unwrap();
            assert_eq!(hex.len(), 32, "{id}");
            assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()), "{id}");
        }
        // ids made one after another share no clock or counter prefix: every pair differs early
        for w in ids.windows(2) {
            assert_ne!(w[0][..13], w[1][..13], "{} {}", w[0], w[1]);
        }
        // and the item ids carry the same digits
        assert_eq!(suffix(&ids[0]).len(), 32);
    }

    fn tr(body: &str) -> R<Translated> {
        translate(&parse_json(body).unwrap(), "resp_abc", &|_| None, true)
    }

    fn msgs(t: &Translated) -> String {
        t.chat.get("messages").unwrap().dump()
    }

    #[test]
    fn a_context_with_many_small_json_nodes_is_counted_as_held() {
        // A tool schema of many small properties, many metadata pairs and an echoed field: as
        // parsed trees these would hold far more than their serialised length (a node, a vector
        // slot and a string per value); the context holds them as text, and its size counts
        // every byte of that text.
        let props: Vec<String> = (0..4000)
            .map(|i| format!("\"p{i}\":{{\"type\":\"string\"}}"))
            .collect();
        let meta: Vec<String> = (0..200).map(|i| format!("\"k{i}\":\"v\"")).collect();
        let body = format!(
            r#"{{"model":"m","input":"Hi","user":"u",
            "tools":[{{"type":"function","name":"f","parameters":{{"type":"object","properties":{{{}}}}}}}],
            "tool_choice":{{"type":"function","name":"f"}},"metadata":{{{}}}}}"#,
            props.join(","),
            meta.join(",")
        );
        let req = parse_json(&body).unwrap();
        let h = tr(&body).unwrap().head;
        // the fields are the request's own JSON, as text
        assert_eq!(h.tools, req.get("tools").unwrap().dump());
        assert_eq!(h.tool_choice, req.get("tool_choice").unwrap().dump());
        assert_eq!(h.metadata, req.get("metadata").unwrap().dump());
        assert_eq!(h.echo, vec![("user", "\"u\"".to_string())]);
        // the account covers every text the context holds, and stays within a fixed overhead of it
        let texts = h.input.capacity()
            + h.tools.capacity()
            + h.tool_choice.capacity()
            + h.metadata.capacity()
            + h.echo.iter().map(|(_, v)| v.capacity()).sum::<usize>();
        let size = h.size();
        assert!(size >= texts as u64, "{size} < {texts}");
        assert!(size <= texts as u64 + 4096, "{size} > {texts} + 4096");
        assert!(h.tools.len() > 4000 * 20);
    }

    #[test]
    fn a_string_input_is_one_user_message_after_the_instructions() {
        let t = tr(
            r#"{"model":"m","instructions":"Be brief.","input":"Hi","max_output_tokens":64,
            "temperature":0.2,"reasoning":{"effort":"low"},"stream":true}"#,
        )
        .unwrap();
        assert_eq!(
            msgs(&t),
            r#"[{"role":"system","content":"Be brief."},{"role":"user","content":[{"type":"text","text":"Hi"}]}]"#
        );
        assert_eq!(t.chat.get("max_tokens").unwrap().as_u64(), Some(64));
        assert_eq!(
            t.chat.get("reasoning_effort").unwrap().as_str(),
            Some("low")
        );
        assert_eq!(t.chat.get("stream").unwrap().as_bool(), Some(true));
        assert_eq!(
            t.chat.get("gaudium_response_id").unwrap().as_str(),
            Some("resp_abc")
        );
        assert!(t.ignored.is_empty(), "{:?}", t.ignored);
        // the item kept for the store has an id
        assert_eq!(t.head.input_items().len(), 1);
        assert!(t.head.input_items()[0]
            .get("id")
            .unwrap()
            .as_str()
            .unwrap()
            .starts_with("msg_"));
        assert!(t.head.store);
    }

    #[test]
    fn a_function_call_round_trip_becomes_an_assistant_call_and_a_tool_message() {
        let t = tr(r#"{"input":[
            {"role":"developer","content":"Use tools."},
            {"role":"user","content":[{"type":"input_text","text":"Weather in Oslo?"}]},
            {"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"Need the tool."}]},
            {"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Oslo\"}"},
            {"type":"function_call_output","call_id":"call_1","output":"sunny"}],
            "tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}},"strict":true}],
            "tool_choice":{"type":"function","name":"get_weather"}}"#)
        .unwrap();
        assert_eq!(
            msgs(&t),
            concat!(
                r#"[{"role":"system","content":"Use tools."},{"role":"user","content":[{"type":"text","text":"Weather in Oslo?"}]},"#,
                r#"{"role":"assistant","content":"","reasoning_content":"Need the tool.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}}]},"#,
                r#"{"role":"tool","tool_call_id":"call_1","content":"sunny"}]"#
            )
        );
        assert_eq!(
            t.chat.get("tool_choice").unwrap().dump(),
            r#"{"type":"function","function":{"name":"get_weather"}}"#
        );
        assert_eq!(
            t.chat.get("tools").unwrap().dump(),
            r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#
        );
        assert_eq!(t.ignored.len(), 1, "{:?}", t.ignored);
        assert!(t.ignored[0].starts_with("tools[].strict"));
    }

    #[test]
    fn allowed_tools_narrow_the_offered_tools() {
        let t = tr(r#"{"input":"x","tools":[{"type":"function","name":"a"},{"type":"function","name":"b"}],
            "tool_choice":{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"b"}]}}"#)
        .unwrap();
        assert_eq!(
            t.chat.get("tool_choice").unwrap().as_str(),
            Some("required")
        );
        let tools = t.chat.get("tools").unwrap().dump();
        assert!(
            tools.contains("\"b\"") && !tools.contains("\"a\""),
            "{tools}"
        );
    }

    #[test]
    fn refusals_name_the_field() {
        for (body, param) in [
            (r#"{"input":"x","background":true}"#, "background"),
            (r#"{"input":"x","conversation":"conv_1"}"#, "conversation"),
            (r#"{"input":"x","top_logprobs":3}"#, "top_logprobs"),
            (
                r#"{"input":"x","text":{"format":{"type":"json_object"}}}"#,
                "text.format",
            ),
            (
                r#"{"input":"x","text":{"format":{"type":"json_schema","name":"n","schema":{}}}}"#,
                "text.format",
            ),
            (
                r#"{"input":"x","tools":[{"type":"web_search"}]}"#,
                "tools[0].type",
            ),
            (
                r#"{"input":"x","tools":[{"type":"code_interpreter","container":{"type":"auto"}}]}"#,
                "tools[0].type",
            ),
            (
                r#"{"input":"x","include":["message.output_text.logprobs"]}"#,
                "include",
            ),
            (
                r#"{"input":"x","previous_response_id":"resp_nope"}"#,
                "previous_response_id",
            ),
            (
                r#"{"input":[{"type":"item_reference","id":"msg_1"}]}"#,
                "input[0]",
            ),
            (
                r#"{"input":[{"role":"user","content":[{"type":"input_image","file_id":"file_1"}]}]}"#,
                "input[0].content[0].file_id",
            ),
            (
                r#"{"input":[{"role":"user","content":[{"type":"input_file","file_data":"x"}]}]}"#,
                "input[0].content[0]",
            ),
            (
                r#"{"input":"x","max_output_tokens":0}"#,
                "max_output_tokens",
            ),
            (
                r#"{"input":"x","tools":[{"type":"function","name":"a"}],"tool_choice":{"type":"function","name":"b"}}"#,
                "tool_choice.name",
            ),
        ] {
            let e = tr(body).unwrap_err();
            assert_eq!(e.param.as_deref(), Some(param), "{body}: {}", e.message);
            assert_eq!(e.status, 400);
        }
        // the built-in tool is named in the message
        let e = tr(r#"{"input":"x","tools":[{"type":"web_search"}]}"#).unwrap_err();
        assert!(
            e.message.contains("'web_search' is a built-in tool"),
            "{}",
            e.message
        );
    }

    #[test]
    fn values_that_change_nothing_pass_and_others_are_reported() {
        let t = tr(r#"{"input":"x","background":false,"top_logprobs":0,"truncation":"disabled",
            "text":{"format":{"type":"text"}},"store":false,"user":"u1","service_tier":"auto","future":1}"#)
        .unwrap();
        assert_eq!(t.ignored.len(), 3, "{:?}", t.ignored);
        assert!(t.ignored[0].starts_with("user (not applied"));
        assert!(t.ignored[1].starts_with("service_tier (not applied"));
        assert_eq!(t.ignored[2], "future (unknown field)");
        assert!(!t.head.store && t.head.encrypted);
    }

    #[test]
    fn a_previous_response_is_prepended_without_its_instructions() {
        let prior = Arc::new(StoredResponse {
            id: "resp_1".into(),
            response: r#"{"id":"resp_1","output":[{"type":"message","id":"msg_b","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello.","annotations":[]}]}]}"#.into(),
            input: r#"[{"type":"message","id":"msg_a","role":"user","content":"Hi"}]"#.into(),
            own_from: 0,
        });
        let t = translate(
            &parse_json(
                r#"{"previous_response_id":"resp_1","instructions":"New.","input":"And?"}"#,
            )
            .unwrap(),
            "resp_2",
            &|id| (id == "resp_1").then(|| Arc::clone(&prior)),
            true,
        )
        .unwrap();
        assert_eq!(
            msgs(&t),
            r#"[{"role":"system","content":"New."},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello."},{"role":"user","content":[{"type":"text","text":"And?"}]}]"#
        );
        assert_eq!(t.head.own_from, 2);
        assert_eq!(t.head.input_items().len(), 3);
    }

    #[test]
    fn our_encrypted_reasoning_comes_back_as_the_reasoning_text() {
        let enc = thinking_signature("hidden thought");
        let t = tr(&format!(
            r#"{{"input":[{{"role":"user","content":"q"}},{{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"{enc}"}},
            {{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"a"}}]}},{{"role":"user","content":"q2"}}]}}"#
        ))
        .unwrap();
        assert!(
            msgs(&t).contains(
                r#"{"role":"assistant","content":"a","reasoning_content":"hidden thought"}"#
            ),
            "{}",
            msgs(&t)
        );
    }

    fn header(id: &str, name: &str) -> ToolDelta {
        ToolDelta {
            index: 0,
            header: true,
            id: id.into(),
            name: name.into(),
            arguments: String::new(),
        }
    }

    fn events(s: &str) -> Vec<JsonValue> {
        s.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|d| parse_json(d).unwrap())
            .collect()
    }

    #[test]
    fn the_builder_streams_items_in_order_with_rising_sequence_numbers() {
        let mut b = Builder::new(Arc::new(Head::default()), "resp_x", "m", 7, 0.5, 1.0);
        let mut out = b.start();
        out.push_str(&b.feed(&Delta {
            reasoning: "think".into(),
            ..Delta::default()
        }));
        out.push_str(&b.feed(&Delta {
            content: "Hi".into(),
            ..Delta::default()
        }));
        out.push_str(&b.feed(&Delta {
            content: " there".into(),
            ..Delta::default()
        }));
        let mut d = Delta::default();
        d.tool_deltas.push(header("call_9", "f"));
        d.tool_deltas.push(ToolDelta {
            index: 0,
            header: false,
            id: String::new(),
            name: String::new(),
            arguments: "{\"a\":1}".into(),
        });
        out.push_str(&b.feed(&d));
        out.push_str(&b.close(Ending::Completed));
        let r = b.response(
            Ending::Completed,
            None,
            &usage_json(5, 0, 4, 1),
            true,
            "",
            false,
        );
        out.push_str(&b.finish(Ending::Completed, &r));
        let ev = events(&out);
        let kinds: Vec<&str> = ev
            .iter()
            .map(|e| e.get("type").unwrap().as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (i, e) in ev.iter().enumerate() {
            assert_eq!(e.get("sequence_number").unwrap().as_u64(), Some(i as u64));
        }
        // the event name of each frame is its type
        assert!(out.contains(
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\""
        ));
        let done = ev.last().unwrap().get("response").unwrap();
        let output = done.get("output").unwrap().as_array().unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(
            output[1].get("content").unwrap().dump(),
            r#"[{"type":"output_text","text":"Hi there","annotations":[],"logprobs":[]}]"#
        );
        assert_eq!(output[2].get("call_id").unwrap().as_str(), Some("call_9"));
        assert_eq!(output[2].get("id").unwrap().as_str(), Some("fc_9"));
        assert_eq!(
            output[2].get("arguments").unwrap().as_str(),
            Some("{\"a\":1}")
        );
        assert_eq!(done.get("status").unwrap().as_str(), Some("completed"));
        assert_eq!(
            done.get("usage")
                .unwrap()
                .get("total_tokens")
                .unwrap()
                .as_u64(),
            Some(9)
        );
        // the stored items are the response's output
        assert_eq!(b.output_items().len(), 3);
    }

    #[test]
    fn an_output_limit_leaves_the_open_item_incomplete() {
        let mut b = Builder::new(Arc::new(Head::default()), "resp_x", "m", 7, 0.5, 1.0);
        b.feed(&Delta {
            content: "Hal".into(),
            ..Delta::default()
        });
        b.close(Ending::Incomplete);
        let r =
            parse_json(&b.response(Ending::Incomplete, None, "null", false, "", false)).unwrap();
        assert_eq!(r.get("status").unwrap().as_str(), Some("incomplete"));
        assert_eq!(
            r.get("incomplete_details").unwrap().dump(),
            r#"{"reason":"max_output_tokens"}"#
        );
        assert_eq!(
            r.get("output").unwrap().as_array().unwrap()[0]
                .get("status")
                .unwrap()
                .as_str(),
            Some("incomplete")
        );
    }

    #[test]
    fn input_items_are_paged_newest_first() {
        let r = StoredResponse {
            id: "resp_1".into(),
            response: "{}".into(),
            input: format!(
                "[{}]",
                (0..5)
                    .map(|i| format!(r#"{{"type":"message","id":"m{i}","role":"user","content":"x"}}"#))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            own_from: 1,
        };
        let p = parse_json(&input_items_page(&r, None, Some("2"), None).unwrap()).unwrap();
        assert_eq!(p.get("first_id").unwrap().as_str(), Some("m4"));
        assert_eq!(p.get("last_id").unwrap().as_str(), Some("m3"));
        assert_eq!(p.get("has_more").unwrap().as_bool(), Some(true));
        let p = parse_json(&input_items_page(&r, Some("m3"), None, None).unwrap()).unwrap();
        assert_eq!(p.get("data").unwrap().as_array().unwrap().len(), 2);
        assert_eq!(p.get("has_more").unwrap().as_bool(), Some(false));
        let p = parse_json(&input_items_page(&r, None, None, Some("asc")).unwrap()).unwrap();
        assert_eq!(p.get("first_id").unwrap().as_str(), Some("m1"));
        assert!(input_items_page(&r, None, Some("0"), None).is_err());
    }
}
