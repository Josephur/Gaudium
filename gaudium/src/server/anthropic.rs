//! Messages API (`POST /v1/messages`, `/v1/messages/count_tokens`) as a translation layer onto
//! the chat pipeline.
//!
//! A Messages request is rewritten into the chat-completions body the chat path takes, so
//! rendering, tokenisation, tool calling, reasoning and sampling are the chat path's. The output
//! side turns the output parser's deltas into content blocks: server-sent events while
//! streaming, the same blocks in the non-streaming body (both come from one [`Builder`], so a
//! stream accumulates to exactly the non-streaming message).
//!
//! Which block is open, when a new one starts and when a stop event is owed follow the verified
//! block lifecycle (`core::tool_calls::BlockTrack`).

use super::outparse::Delta;
use super::readout::jq;
use crate::core::tool_calls::{BlockKind, BlockTrack};
use crate::json::{parse_json, JsonValue};

const SIG_PREFIX: &str = "Gdm1";
/// A per-request header block some clients put first in the system prompt; it changes every
/// request, so it is dropped.
const BILLING: &str = "x-anthropic-billing-header:";

type R<T> = Result<T, String>;

fn bad<T>(m: impl Into<String>) -> R<T> {
    Err(m.into())
}

fn at(path: &str, i: usize) -> String {
    format!("{path}.{i}")
}

fn s(v: &str) -> JsonValue {
    JsonValue::Str(v.to_string())
}

fn obj(pairs: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn text_of<'a>(o: &'a JsonValue, key: &str, path: &str) -> R<&'a str> {
    match o.get(key).and_then(JsonValue::as_str) {
        Some(v) => Ok(v),
        None => bad(format!("{path}.{key}: Field required (a string)")),
    }
}

fn kind(o: &JsonValue) -> &str {
    o.get("type").and_then(JsonValue::as_str).unwrap_or("")
}

/// The smallest `thinking.budget_tokens` the Messages API accepts.
const MIN_THINKING_BUDGET: u64 = 1024;

fn present(o: &JsonValue, key: &str) -> Option<JsonValue> {
    o.get(key)
        .filter(|v| !matches!(v, JsonValue::Null))
        .cloned()
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(input: &[u8]) -> String {
    let mut o = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let v = u32::from(chunk[0]) << 16
            | u32::from(*chunk.get(1).unwrap_or(&0)) << 8
            | u32::from(*chunk.get(2).unwrap_or(&0));
        o.push(B64[(v >> 18) as usize] as char);
        o.push(B64[(v >> 12 & 63) as usize] as char);
        o.push(if chunk.len() > 1 {
            B64[(v >> 6 & 63) as usize] as char
        } else {
            '='
        });
        o.push(if chunk.len() > 2 {
            B64[(v & 63) as usize] as char
        } else {
            '='
        });
    }
    o
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for c in input.bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = B64.iter().position(|x| *x == c)? as u32;
        acc = acc << 6 | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// The opaque thinking signature: it carries the reasoning text, so a thinking block sent back
/// with its text omitted still re-renders the turn exactly.
pub fn thinking_signature(reasoning: &str) -> String {
    format!("{SIG_PREFIX}{}", base64_encode(reasoning.as_bytes()))
}

/// The reasoning a signature of ours carries (`None`: not one of ours).
pub fn signature_reasoning(signature: &str) -> Option<String> {
    let body = signature.strip_prefix(SIG_PREFIX)?;
    String::from_utf8(base64_decode(body)?).ok()
}

/// `call_X` -> `toolu_X`.
pub fn tool_use_id(call_id: &str) -> String {
    format!("toolu_{}", call_id.strip_prefix("call_").unwrap_or(call_id))
}

/// The error type of an HTTP status.
pub fn error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        s if s >= 500 => "api_error",
        _ => "invalid_request_error",
    }
}

/// `{"type":"error","error":{"type","message"}}`.
pub fn error_json(status: u16, message: &str) -> String {
    format!(
        "{{\"type\":\"error\",\"error\":{{\"type\":{},\"message\":{}}}}}",
        jq(error_type(status)),
        jq(message)
    )
}

/// `event: <name>\ndata: <json>\n\n`.
pub fn sse(event: &str, data: &str) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Input / output token counts in the Messages shape.
pub fn usage_json(input_tokens: u64, output_tokens: u64) -> String {
    usage_json_cached(input_tokens, 0, output_tokens)
}

/// The usage of a request whose prompt of `prompt_tokens` took `reused` tokens over from the
/// request before it (prefix reuse): those are reported as read from the cache, the rest as
/// input.
pub fn usage_json_cached(prompt_tokens: u64, reused: u64, output_tokens: u64) -> String {
    format!(
        "{{\"input_tokens\":{},\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":{},\"output_tokens\":{output_tokens}}}",
        crate::verified::pc_prefill_rows(prompt_tokens, reused),
        crate::verified::pc_reused_rows(prompt_tokens, reused),
    )
}

/// A system prompt's text (a string or text blocks).
fn system_text(v: &JsonValue, path: &str, ignored: &mut Vec<String>) -> R<String> {
    match v {
        JsonValue::Null => Ok(String::new()),
        JsonValue::Str(t) => Ok(t.clone()),
        JsonValue::Array(blocks) => {
            let mut out = String::new();
            for (i, b) in blocks.iter().enumerate() {
                let p = at(path, i);
                if kind(b) != "text" {
                    return bad(format!("{p}: system content must be text blocks"));
                }
                let t = text_of(b, "text", &p)?;
                if t.starts_with(BILLING) {
                    ignored.push(format!(
                        "{p} billing header (dropped: it changes every request)"
                    ));
                    continue;
                }
                out.push_str(t);
            }
            Ok(out)
        }
        _ => bad(format!(
            "{path}: must be a string or an array of text blocks"
        )),
    }
}

/// A document block as text (plain-text sources only).
fn document_text(b: &JsonValue, path: &str) -> R<String> {
    let Some(src) = b.get("source").filter(|x| x.as_object().is_some()) else {
        return bad(format!("{path}.source: Field required"));
    };
    let sp = format!("{path}.source");
    let mut text = match kind(src) {
        "text" => text_of(src, "data", &sp)?.to_string(),
        "content" => match src.get("content") {
            Some(JsonValue::Str(t)) => t.clone(),
            Some(JsonValue::Array(parts)) => {
                let mut t = String::new();
                for (i, part) in parts.iter().enumerate() {
                    let pp = at(&format!("{sp}.content"), i);
                    if kind(part) != "text" {
                        return bad(format!("{pp}: only text blocks are supported in a content document"));
                    }
                    t.push_str(text_of(part, "text", &pp)?);
                }
                t
            }
            _ => return bad(format!("{sp}.content: Field required")),
        },
        "base64" => {
            let media = src.get("media_type").and_then(JsonValue::as_str).unwrap_or("");
            if !media.starts_with("text/") {
                let what = if media.is_empty() { "this" } else { media };
                return bad(format!(
                    "{path}: {what} documents are not supported by this server (binary documents need a renderer); send the text"
                ));
            }
            match base64_decode(text_of(src, "data", &sp)?).and_then(|b| String::from_utf8(b).ok()) {
                Some(t) => t,
                None => return bad(format!("{sp}.data: invalid base64 text")),
            }
        }
        k @ ("url" | "file") => {
            return bad(format!(
                "{sp}: document {k} sources are not supported by this server; send the text (source type \"text\")"
            ))
        }
        k => return bad(format!("{sp}.type: unknown document source type '{k}'")),
    };
    if let Some(title) = b
        .get("title")
        .and_then(JsonValue::as_str)
        .filter(|t| !t.is_empty())
    {
        text = format!("{title}\n\n{text}");
    }
    Ok(text)
}

/// Content under construction: text, and image blocks in their place. Without an image it is
/// one string (as a text-only request was always translated); with one it is a list of parts.
#[derive(Default)]
struct Parts {
    parts: Vec<JsonValue>,
    has_image: bool,
}

impl Parts {
    fn push_str(&mut self, t: &str) {
        match self.parts.last_mut() {
            Some(JsonValue::Object(e))
                if e.iter()
                    .any(|(k, v)| k == "type" && v.as_str() == Some("text")) =>
            {
                if let Some((_, JsonValue::Str(prev))) = e.iter_mut().find(|(k, _)| k == "text") {
                    prev.push_str(t);
                }
            }
            _ => self
                .parts
                .push(obj(vec![("type", s("text")), ("text", s(t))])),
        }
    }

    /// An `image` block (`source`: base64 or url), passed on for the chat path to resolve.
    fn push_image(&mut self, b: &JsonValue, path: &str) -> R<()> {
        let Some(src) = b.get("source").filter(|x| x.as_object().is_some()) else {
            return bad(format!(
                "{path}.source: an image block needs a source object"
            ));
        };
        self.parts
            .push(obj(vec![("type", s("image")), ("source", src.clone())]));
        self.has_image = true;
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    fn into_content(self) -> JsonValue {
        if self.has_image {
            return JsonValue::Array(self.parts);
        }
        let mut text = String::new();
        for p in &self.parts {
            if let Some(t) = p.get("text").and_then(JsonValue::as_str) {
                text.push_str(t);
            }
        }
        JsonValue::Str(text)
    }
}

/// The content of a `tool_result` block: its text, with its images in place.
fn tool_result_content(b: &JsonValue, path: &str, all_tools: &[JsonValue]) -> R<JsonValue> {
    let mut text = Parts::default();
    if b.get("is_error").and_then(JsonValue::as_bool) == Some(true) {
        text.push_str("Error: ");
    }
    match b.get("content") {
        None | Some(JsonValue::Null) => {}
        Some(JsonValue::Str(t)) => text.push_str(t),
        Some(JsonValue::Array(parts)) => {
            for (i, x) in parts.iter().enumerate() {
                let p = at(&format!("{path}.content"), i);
                if x.as_object().is_none() {
                    return bad(format!("{p}: must be an object"));
                }
                match kind(x) {
                    "text" => text.push_str(text_of(x, "text", &p)?),
                    "document" => text.push_str(&document_text(x, &p)?),
                    "image" => text.push_image(x, &p)?,
                    // a tool search result: the deferred tool's definition, inline
                    "tool_reference" => {
                        let name = text_of(x, "tool_name", &p)?;
                        let def = all_tools.iter().find(|t| {
                            t.get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(JsonValue::as_str)
                                == Some(name)
                        });
                        text.push_str(&match def {
                            Some(d) => format!(
                                "Tool {name} is now available:\n<tools>\n{}\n</tools>\n",
                                d.dump()
                            ),
                            None => format!("Tool {name} is not available."),
                        });
                    }
                    k => {
                        return bad(format!(
                        "{p}.type: '{k}' blocks are not supported in a tool_result by this server"
                    ))
                    }
                }
            }
        }
        Some(_) => {
            return bad(format!(
                "{path}.content: must be a string or an array of blocks"
            ))
        }
    }
    Ok(text.into_content())
}

fn convert_user(
    content: &JsonValue,
    path: &str,
    msgs: &mut Vec<JsonValue>,
    all_tools: &[JsonValue],
) -> R<()> {
    let user = |content: JsonValue| obj(vec![("role", s("user")), ("content", content)]);
    let blocks = match content {
        JsonValue::Str(t) => {
            msgs.push(user(JsonValue::Str(t.clone())));
            return Ok(());
        }
        JsonValue::Array(b) => b,
        _ => {
            return bad(format!(
                "{path}.content: must be a string or an array of content blocks"
            ))
        }
    };
    let mut text: Option<Parts> = None;
    let mut any = false;
    for (i, b) in blocks.iter().enumerate() {
        let p = at(&format!("{path}.content"), i);
        if b.as_object().is_none() {
            return bad(format!("{p}: must be an object"));
        }
        any = true;
        match kind(b) {
            "text" => text
                .get_or_insert_with(Parts::default)
                .push_str(text_of(b, "text", &p)?),
            "document" => text
                .get_or_insert_with(Parts::default)
                .push_str(&document_text(b, &p)?),
            "image" => text.get_or_insert_with(Parts::default).push_image(b, &p)?,
            "tool_result" => {
                if let Some(t) = text.take() {
                    msgs.push(user(t.into_content()));
                }
                msgs.push(obj(vec![
                    ("role", s("tool")),
                    ("tool_call_id", s(text_of(b, "tool_use_id", &p)?)),
                    ("content", tool_result_content(b, &p, all_tools)?),
                ]));
            }
            k @ ("thinking" | "redacted_thinking" | "tool_use") => {
                return bad(format!("{p}: '{k}' blocks belong in assistant messages"))
            }
            k => {
                return bad(format!(
                    "{p}.type: '{k}' content blocks are not supported by this server"
                ))
            }
        }
    }
    if !any {
        msgs.push(user(JsonValue::Str(String::new())));
    } else if let Some(t) = text.filter(|t| !t.is_empty()) {
        msgs.push(user(t.into_content()));
    }
    Ok(())
}

/// Adds the assistant message; returns its text (for the prefill check).
fn convert_assistant(
    content: &JsonValue,
    path: &str,
    msgs: &mut Vec<JsonValue>,
    ignored: &mut Vec<String>,
) -> R<String> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    match content {
        JsonValue::Str(t) => text.push_str(t),
        JsonValue::Array(blocks) => {
            for (i, b) in blocks.iter().enumerate() {
                let p = at(&format!("{path}.content"), i);
                if b.as_object().is_none() {
                    return bad(format!("{p}: must be an object"));
                }
                match kind(b) {
                    "text" => text.push_str(text_of(b, "text", &p)?),
                    "thinking" => {
                        let mut r = b.get("thinking").and_then(JsonValue::as_str).unwrap_or("").to_string();
                        if r.is_empty() {
                            r = b
                                .get("signature")
                                .and_then(JsonValue::as_str)
                                .and_then(signature_reasoning)
                                .unwrap_or_default();
                        }
                        if !r.is_empty() {
                            if !reasoning.is_empty() {
                                reasoning.push_str("\n\n");
                            }
                            reasoning.push_str(&r);
                        }
                    }
                    "redacted_thinking" => ignored.push(format!("{p} redacted_thinking (dropped)")),
                    "tool_use" => {
                        let input = match b.get("input") {
                            None | Some(JsonValue::Null) => JsonValue::Object(Vec::new()),
                            Some(JsonValue::Str(t)) => match parse_json(t) {
                                Ok(v) => v,
                                Err(_) => return bad(format!("{p}.input: not valid JSON")),
                            },
                            Some(v) => v.clone(),
                        };
                        if input.as_object().is_none() {
                            return bad(format!("{p}.input: must be an object"));
                        }
                        calls.push(obj(vec![
                            ("id", s(text_of(b, "id", &p)?)),
                            ("type", s("function")),
                            ("function", obj(vec![("name", s(text_of(b, "name", &p)?)), ("arguments", input)])),
                        ]));
                    }
                    k => return bad(format!("{p}.type: '{k}' blocks are not supported in assistant messages by this server")),
                }
            }
        }
        _ => {
            return bad(format!(
                "{path}.content: must be a string or an array of content blocks"
            ))
        }
    }
    let mut m = vec![
        ("role", s("assistant")),
        ("content", JsonValue::Str(text.clone())),
    ];
    if !reasoning.is_empty() {
        m.push(("reasoning_content", JsonValue::Str(reasoning)));
    }
    if !calls.is_empty() {
        m.push(("tool_calls", JsonValue::Array(calls)));
    }
    msgs.push(obj(m));
    Ok(text)
}

/// A Messages request as the chat pipeline takes it.
#[derive(Debug)]
pub struct Translated {
    /// The chat-completions body.
    pub chat: JsonValue,
    /// `thinking.display` "omitted": thinking blocks carry only the signature.
    pub show_thinking: bool,
    /// Accepted, not applied.
    pub ignored: Vec<String>,
}

/// Messages request -> chat body. `Err`: what this server cannot serve (unknown block types,
/// server tools, images, binary documents, ...).
pub fn translate(body: &JsonValue) -> R<Translated> {
    if body.as_object().is_none() {
        return bad("the request body must be a JSON object");
    }
    let mut ignored = Vec::new();
    let Some(messages) = body.get("messages").and_then(JsonValue::as_array) else {
        return bad("messages: Field required (an array of messages)");
    };
    if messages.is_empty() {
        return bad("messages: at least one message is required");
    }
    if body.get("model").is_some_and(|m| m.as_str().is_none()) {
        return bad("model: must be a string");
    }

    // tools first: tool_reference blocks in the history look their definitions up
    let (mut tools, mut hidden, mut all) = (Vec::new(), Vec::new(), Vec::new());
    match body.get("tools") {
        None | Some(JsonValue::Null) => {}
        Some(JsonValue::Array(list)) => {
            for (i, x) in list.iter().enumerate() {
                let p = at("tools", i);
                if x.as_object().is_none() {
                    return bad(format!("{p}: must be an object"));
                }
                let ty = x
                    .get("type")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("custom");
                if ty != "custom" {
                    let name = x
                        .get("name")
                        .and_then(JsonValue::as_str)
                        .map(|n| format!(" ({n})"))
                        .unwrap_or_default();
                    return bad(format!(
                        "{p}: '{ty}'{name} is a server tool; this server runs no tools itself. Define it as a client tool (name, description, input_schema) and run it in the client"
                    ));
                }
                let mut f = vec![("name", s(text_of(x, "name", &p)?))];
                if let Some(d) = x.get("description").and_then(JsonValue::as_str) {
                    f.push(("description", s(d)));
                }
                f.push((
                    "parameters",
                    match x.get("input_schema") {
                        None | Some(JsonValue::Null) => obj(vec![
                            ("type", s("object")),
                            ("properties", JsonValue::Object(Vec::new())),
                        ]),
                        Some(schema @ JsonValue::Object(_)) => schema.clone(),
                        Some(_) => {
                            return bad(format!("{p}.input_schema: must be a JSON schema object"))
                        }
                    },
                ));
                let tool = obj(vec![("type", s("function")), ("function", obj(f))]);
                all.push(tool.clone());
                if x.get("defer_loading").and_then(JsonValue::as_bool) == Some(true) {
                    hidden.push(tool);
                } else {
                    tools.push(tool);
                }
            }
        }
        Some(_) => return bad("tools: must be an array"),
    }

    // a leading system-role message joins the system prompt; a later one becomes a user turn
    let mut msgs = Vec::new();
    let mut sys = match body.get("system") {
        Some(v) => system_text(v, "system", &mut ignored)?,
        None => String::new(),
    };
    let role_of = |m: &JsonValue| {
        m.get("role")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_string()
    };
    let mut i = 0;
    while i < messages.len() && role_of(&messages[i]) == "system" {
        sys.push_str(&system_text(
            messages[i].get("content").unwrap_or(&JsonValue::Null),
            &format!("{}.content", at("messages", i)),
            &mut ignored,
        )?);
        i += 1;
    }
    if !sys.is_empty() {
        msgs.push(obj(vec![
            ("role", s("system")),
            ("content", JsonValue::Str(sys)),
        ]));
    }
    let mut last_text = String::new();
    let mut last_role = String::new();
    for (i, m) in messages.iter().enumerate().skip(i) {
        let p = at("messages", i);
        if m.as_object().is_none() {
            return bad(format!("{p}: must be an object"));
        }
        let Some(content) = m.get("content") else {
            return bad(format!("{p}.content: Field required"));
        };
        last_role = role_of(m);
        match last_role.as_str() {
            "user" => convert_user(content, &p, &mut msgs, &all)?,
            "assistant" => last_text = convert_assistant(content, &p, &mut msgs, &mut ignored)?,
            "system" => {
                let t = system_text(content, &format!("{p}.content"), &mut ignored)?;
                msgs.push(obj(vec![
                    ("role", s("user")),
                    (
                        "content",
                        JsonValue::Str(format!("<system-reminder>\n{t}\n</system-reminder>")),
                    ),
                ]));
            }
            _ => return bad(format!("{p}.role: must be \"user\" or \"assistant\"")),
        }
    }
    // a final assistant turn is a prefill: the model continues its text
    let mut prefill = false;
    if last_role == "assistant" {
        if !last_text.is_empty() {
            prefill = true;
        } else if msgs
            .last()
            .is_some_and(|m| m.get("tool_calls").is_none() && m.get("reasoning_content").is_none())
        {
            msgs.pop(); // empty: nothing to continue
        }
    }

    let mut chat: Vec<(&str, JsonValue)> = vec![("messages", JsonValue::Array(msgs))];
    if prefill {
        chat.push(("continue_final_message", JsonValue::Bool(true)));
        chat.push(("add_generation_prompt", JsonValue::Bool(false)));
    }
    let have_tools = !tools.is_empty();
    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(JsonValue::as_str)
                .map(str::to_string)
        })
        .collect();
    if have_tools {
        chat.push(("tools", JsonValue::Array(tools)));
    }
    if !hidden.is_empty() {
        chat.push(("gaudium_hidden_tools", JsonValue::Array(hidden)));
    }

    if let Some(tc) = present(body, "tool_choice") {
        if tc.as_object().is_none() {
            return bad("tool_choice: must be an object {\"type\": \"auto\" | \"any\" | \"tool\" | \"none\"}");
        }
        match kind(&tc) {
            "auto" => chat.push(("tool_choice", s("auto"))),
            "none" => chat.push(("tool_choice", s("none"))),
            ty @ ("any" | "tool") => {
                if !have_tools {
                    return bad(format!(
                        "tool_choice: type '{ty}' needs a non-empty 'tools' list"
                    ));
                }
                if ty == "any" {
                    chat.push(("tool_choice", s("required")));
                } else {
                    let name = text_of(&tc, "name", "tool_choice")?;
                    if !tool_names.iter().any(|n| n == name) {
                        return bad(format!("tool_choice: the tool '{name}' is not in 'tools'"));
                    }
                    chat.push((
                        "tool_choice",
                        obj(vec![
                            ("type", s("function")),
                            ("function", obj(vec![("name", s(name))])),
                        ]),
                    ));
                }
            }
            _ => return bad("tool_choice.type: must be \"auto\", \"any\", \"tool\" or \"none\""),
        }
        if let Some(off) = tc
            .get("disable_parallel_tool_use")
            .and_then(JsonValue::as_bool)
        {
            chat.push(("parallel_tool_calls", JsonValue::Bool(!off)));
        }
    }

    // thinking: enabled / adaptive -> on, disabled -> off, absent -> the template's default
    let mut show_thinking = true;
    let mut kwargs: Vec<(String, JsonValue)> = Vec::new();
    if let Some(th) = present(body, "thinking") {
        if th.as_object().is_none() {
            return bad(
                "thinking: must be an object {\"type\": \"enabled\" | \"adaptive\" | \"disabled\"}",
            );
        }
        match kind(&th) {
            "enabled" | "adaptive" => {
                kwargs.push(("enable_thinking".into(), JsonValue::Bool(true)))
            }
            "disabled" => kwargs.push(("enable_thinking".into(), JsonValue::Bool(false))),
            _ => return bad("thinking.type: must be \"enabled\", \"adaptive\" or \"disabled\""),
        }
        // `enabled` takes a budget of at least 1024 tokens, below `max_tokens`; the reasoning
        // is closed when it reaches the budget (`run_chat`). `adaptive` has no budget.
        if let Some(b) = present(&th, "budget_tokens") {
            if kind(&th) == "enabled" {
                let Some(n) = b.as_u64().filter(|_| !matches!(b, JsonValue::Float(_))) else {
                    return bad("thinking.budget_tokens: must be an integer");
                };
                if n < MIN_THINKING_BUDGET {
                    return bad(format!(
                        "thinking.budget_tokens: must be at least {MIN_THINKING_BUDGET}"
                    ));
                }
                if let Some(mt) = body.get("max_tokens").and_then(JsonValue::as_u64) {
                    if n >= mt {
                        return bad(format!(
                            "thinking.budget_tokens: must be less than max_tokens ({n} >= {mt})"
                        ));
                    }
                }
                chat.push(("thinking_budget_tokens", JsonValue::Int(n as i64)));
            } else {
                ignored.push(format!(
                    "thinking.budget_tokens (only thinking.type \"enabled\" takes a budget, not \"{}\")",
                    kind(&th)
                ));
            }
        }
        if th.get("display").and_then(JsonValue::as_str) == Some("omitted") {
            show_thinking = false;
        }
    }
    if let Some(oc) = body
        .get("output_config")
        .filter(|o| o.as_object().is_some())
    {
        if let Some(e) = oc.get("effort").and_then(JsonValue::as_str) {
            if !matches!(e, "low" | "medium" | "high" | "xhigh" | "max") {
                return bad("output_config.effort: must be low, medium, high, xhigh or max");
            }
            chat.push(("reasoning_effort", s(e)));
        }
        if present(oc, "format").is_some() {
            ignored.push("output_config.format (constrained decoding is not available)".into());
        }
    }
    // extension: explicit template arguments win
    if let Some(extra) = body
        .get("chat_template_kwargs")
        .and_then(JsonValue::as_object)
    {
        for (k, v) in extra {
            kwargs.retain(|(name, _)| name != k);
            kwargs.push((k.clone(), v.clone()));
        }
    }
    if !kwargs.is_empty() {
        chat.push(("chat_template_kwargs", JsonValue::Object(kwargs)));
    }

    if let Some(mt) = present(body, "max_tokens") {
        if mt.as_u64().is_none() || matches!(mt, JsonValue::Float(_)) {
            return bad("max_tokens: must be a non-negative integer");
        }
        chat.push(("max_tokens", mt));
    }
    for k in ["temperature", "top_p", "top_k"] {
        if let Some(v) = present(body, k) {
            if v.as_f64().is_none() {
                return bad(format!("{k}: must be a number"));
            }
            chat.push((k, v));
        }
    }
    if let Some(seqs) = present(body, "stop_sequences") {
        let Some(list) = seqs.as_array() else {
            return bad("stop_sequences: must be an array of strings");
        };
        if list.iter().any(|x| x.as_str().is_none()) {
            return bad("stop_sequences: must be an array of strings");
        }
        let kept: Vec<JsonValue> = list
            .iter()
            .filter(|x| x.as_str() != Some(""))
            .cloned()
            .collect();
        if !kept.is_empty() {
            chat.push(("stop", JsonValue::Array(kept)));
        }
    }
    if let Some(st) = body.get("stream") {
        if st.as_bool().is_none() {
            return bad("stream: must be a boolean");
        }
        chat.push(("stream", st.clone()));
    }
    if body
        .get("mcp_servers")
        .and_then(JsonValue::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        return bad("mcp_servers: server-side MCP is not supported by this server; run MCP tools in the client");
    }
    for k in [
        "service_tier",
        "context_management",
        "container",
        "inference_geo",
    ] {
        if present(body, k).is_some() {
            ignored.push(format!("{k} (ignored)"));
        }
    }
    // extensions (as on the chat endpoint)
    for k in [
        "model",
        "seed",
        "min_p",
        "gaudium_method",
        "gaudium_a2ui",
        // the generated token ids in the reply's details (exactness checks between runs)
        "gaudium_return_tokens",
        "reasoning_effort",
        "coupled",
        "draft_trunc",
        // false = a whole prefill: neither the device nor host memory is resumed from
        "cache_prompt",
    ] {
        if let Some(v) = present(body, k) {
            if !chat.iter().any(|(name, _)| *name == k) {
                chat.push((k, v));
            }
        }
    }
    Ok(Translated {
        chat: obj(chat),
        show_thinking,
        ignored,
    })
}

enum Block {
    Thinking {
        text: String,
        signature: String,
    },
    Text(String),
    Tool {
        id: String,
        name: String,
        input: String,
    },
}

impl Block {
    fn json(&self) -> String {
        match self {
            Block::Thinking { text, signature } => format!(
                "{{\"type\":\"thinking\",\"thinking\":{},\"signature\":{}}}",
                jq(text),
                jq(signature)
            ),
            Block::Text(t) => format!("{{\"type\":\"text\",\"text\":{}}}", jq(t)),
            Block::Tool { id, name, input } => format!(
                "{{\"type\":\"tool_use\",\"id\":{},\"name\":{},\"input\":{input}}}",
                jq(id),
                jq(name)
            ),
        }
    }
}

/// Content blocks from parser deltas, in arrival order: thinking, text, tool_use; a kind change
/// closes the open block. Each [`Builder::feed`] returns the events it produced (ignored when
/// not streaming).
pub struct Builder {
    show: bool,
    blocks: Vec<Block>,
    track: BlockTrack,
    /// The open thinking or text block's text; the open tool block's input JSON text.
    buf: String,
}

impl Builder {
    pub fn new(show_thinking: bool) -> Self {
        Self {
            show: show_thinking,
            blocks: Vec::new(),
            track: BlockTrack::new(),
            buf: String::new(),
        }
    }

    pub fn message_start(id: &str, model: &str, input_tokens: u64) -> String {
        sse(
            "message_start",
            &format!(
                "{{\"type\":\"message_start\",\"message\":{{\"id\":{},\"type\":\"message\",\"role\":\"assistant\",\"model\":{},\
                 \"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{}}}}}",
                jq(id),
                jq(model),
                usage_json(input_tokens, 0)
            ),
        )
    }

    fn index(&self) -> usize {
        self.blocks.len().saturating_sub(1)
    }

    fn open(&mut self, kind: BlockKind, block: Block) -> String {
        let mut out = self.close();
        let start = block.json();
        self.blocks.push(block);
        self.track.opened(kind);
        self.buf.clear();
        out.push_str(&sse(
            "content_block_start",
            &format!(
                "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{start}}}",
                self.index()
            ),
        ));
        out
    }

    fn delta(&self, body: &str) -> String {
        sse(
            "content_block_delta",
            &format!(
                "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{body}}}",
                self.index()
            ),
        )
    }

    /// Closes the open block (its final content is recorded; a stop event is owed once).
    pub fn close(&mut self) -> String {
        if !self.track.close() {
            return String::new();
        }
        let mut out = String::new();
        let idx = self.index();
        let buf = std::mem::take(&mut self.buf);
        match self.blocks.last_mut() {
            Some(Block::Thinking { text, signature }) => {
                *signature = thinking_signature(&buf);
                if self.show {
                    *text = buf;
                }
                let sig = format!(
                    "{{\"type\":\"signature_delta\",\"signature\":{}}}",
                    jq(signature)
                );
                out.push_str(&sse(
                    "content_block_delta",
                    &format!(
                        "{{\"type\":\"content_block_delta\",\"index\":{idx},\"delta\":{sig}}}"
                    ),
                ));
            }
            Some(Block::Text(text)) => *text = buf,
            Some(Block::Tool { input, .. }) => {
                // the streamed text is the call's arguments: a JSON object, or nothing
                *input = match parse_json(&buf) {
                    Ok(v @ JsonValue::Object(_)) => v.dump(),
                    _ => "{}".into(),
                };
            }
            None => {}
        }
        out.push_str(&sse(
            "content_block_stop",
            &format!("{{\"type\":\"content_block_stop\",\"index\":{idx}}}"),
        ));
        out
    }

    pub fn feed(&mut self, d: &Delta) -> String {
        let mut out = String::new();
        if !d.reasoning.is_empty() {
            if self.track.must_open(BlockKind::BkThinking) {
                out.push_str(&self.open(
                    BlockKind::BkThinking,
                    Block::Thinking {
                        text: String::new(),
                        signature: String::new(),
                    },
                ));
            }
            self.buf.push_str(&d.reasoning);
            if self.show {
                out.push_str(&self.delta(&format!(
                    "{{\"type\":\"thinking_delta\",\"thinking\":{}}}",
                    jq(&d.reasoning)
                )));
            }
        }
        if !d.content.is_empty() {
            if self.track.must_open(BlockKind::BkText) {
                out.push_str(&self.open(BlockKind::BkText, Block::Text(String::new())));
            }
            self.buf.push_str(&d.content);
            out.push_str(&self.delta(&format!(
                "{{\"type\":\"text_delta\",\"text\":{}}}",
                jq(&d.content)
            )));
        }
        for td in &d.tool_deltas {
            // a header starts the call's own block
            if td.header && self.track.must_open(BlockKind::BkTool) {
                out.push_str(&self.open(
                    BlockKind::BkTool,
                    Block::Tool {
                        id: tool_use_id(&td.id),
                        name: td.name.clone(),
                        input: "{}".into(),
                    },
                ));
            }
            if self.track.open() != BlockKind::BkTool || td.arguments.is_empty() {
                continue;
            }
            self.buf.push_str(&td.arguments);
            out.push_str(&self.delta(&format!(
                "{{\"type\":\"input_json_delta\",\"partial_json\":{}}}",
                jq(&td.arguments)
            )));
        }
        out
    }

    /// Closes the open block, then `message_delta` (stop reason, stop sequence, usage and
    /// `extra` fields as `,"key":value` text) and `message_stop`.
    pub fn finish(
        &mut self,
        stop_reason: &str,
        stop_sequence: &str,
        usage: &str,
        extra: &str,
    ) -> String {
        let mut out = self.close();
        let seq = if stop_sequence.is_empty() {
            "null".to_string()
        } else {
            jq(stop_sequence)
        };
        out.push_str(&sse(
            "message_delta",
            &format!(
                "{{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":{},\"stop_sequence\":{seq}}},\"usage\":{usage}{extra}}}",
                jq(stop_reason)
            ),
        ));
        out.push_str(&sse("message_stop", "{\"type\":\"message_stop\"}"));
        out
    }

    /// The whole message (after [`Builder::close`]).
    pub fn message(
        &self,
        id: &str,
        model: &str,
        stop_reason: &str,
        stop_sequence: &str,
        usage: &str,
        extra: &str,
    ) -> String {
        let seq = if stop_sequence.is_empty() {
            "null".to_string()
        } else {
            jq(stop_sequence)
        };
        let blocks: Vec<String> = self.blocks.iter().map(Block::json).collect();
        format!(
            "{{\"id\":{},\"type\":\"message\",\"role\":\"assistant\",\"model\":{},\"content\":[{}],\
             \"stop_reason\":{},\"stop_sequence\":{seq},\"usage\":{usage}{extra}}}",
            jq(id),
            jq(model),
            blocks.join(","),
            jq(stop_reason)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::outparse::ToolDelta;
    use super::*;

    fn translate_str(body: &str) -> R<Translated> {
        translate(&parse_json(body).unwrap())
    }

    #[test]
    fn a_messages_request_becomes_a_chat_request() {
        let t = translate_str(
            r#"{"model":"m","max_tokens":4096,"system":[{"type":"text","text":"x-anthropic-billing-header: abc"},{"type":"text","text":"Be brief."}],
                "tools":[{"name":"get_weather","description":"Weather.","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}},
                         {"name":"later","defer_loading":true}],
                "tool_choice":{"type":"tool","name":"get_weather","disable_parallel_tool_use":true},
                "thinking":{"type":"enabled","budget_tokens":2000,"display":"omitted"},
                "stop_sequences":["END",""],"temperature":0.5,"stream":true,
                "messages":[
                  {"role":"user","content":"Weather in Oslo?"},
                  {"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"Gdm1bmVlZCB0aGUgdG9vbA=="},
                     {"type":"text","text":"Checking."},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Oslo"}}]},
                  {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","is_error":true,"content":[{"type":"text","text":"timeout"}]},
                     {"type":"text","text":"Try again."}]}]}"#,
        )
        .unwrap();
        assert!(!t.show_thinking);
        assert_eq!(t.ignored.len(), 1, "{:?}", t.ignored);
        let c = &t.chat;
        let msgs = c.get("messages").and_then(JsonValue::as_array).unwrap();
        let roles: Vec<&str> = msgs
            .iter()
            .map(|m| m.get("role").and_then(JsonValue::as_str).unwrap())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "user"]);
        assert_eq!(
            msgs[0].get("content").and_then(JsonValue::as_str),
            Some("Be brief.")
        );
        assert_eq!(
            msgs[2].get("reasoning_content").and_then(JsonValue::as_str),
            Some("need the tool")
        );
        assert_eq!(
            msgs[2].get("tool_calls").unwrap().dump(),
            r#"[{"id":"toolu_1","type":"function","function":{"name":"get_weather","arguments":{"city":"Oslo"}}}]"#
        );
        assert_eq!(
            msgs[3].get("tool_call_id").and_then(JsonValue::as_str),
            Some("toolu_1")
        );
        assert_eq!(
            msgs[3].get("content").and_then(JsonValue::as_str),
            Some("Error: timeout")
        );
        assert_eq!(
            msgs[4].get("content").and_then(JsonValue::as_str),
            Some("Try again.")
        );
        assert_eq!(
            c.get("tools").and_then(JsonValue::as_array).unwrap().len(),
            1
        );
        assert_eq!(
            c.get("gaudium_hidden_tools")
                .and_then(JsonValue::as_array)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            c.get("tool_choice").unwrap().dump(),
            r#"{"type":"function","function":{"name":"get_weather"}}"#
        );
        assert_eq!(
            c.get("parallel_tool_calls").and_then(JsonValue::as_bool),
            Some(false)
        );
        assert_eq!(
            c.get("chat_template_kwargs").unwrap().dump(),
            r#"{"enable_thinking":true}"#
        );
        assert_eq!(c.get("stop").unwrap().dump(), r#"["END"]"#);
        assert_eq!(c.get("max_tokens").and_then(JsonValue::as_u64), Some(4096));
        assert_eq!(
            c.get("thinking_budget_tokens").and_then(JsonValue::as_u64),
            Some(2000)
        );
        assert_eq!(c.get("stream").and_then(JsonValue::as_bool), Some(true));
        assert_eq!(c.get("model").and_then(JsonValue::as_str), Some("m"));
    }

    #[test]
    fn prefill_any_choice_and_refusals() {
        let t = translate_str(r#"{"messages":[{"role":"user","content":"Count"},{"role":"assistant","content":"One, "}]}"#).unwrap();
        assert_eq!(
            t.chat
                .get("continue_final_message")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            t.chat
                .get("add_generation_prompt")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        // an empty final assistant turn is nothing to continue
        let t = translate_str(
            r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":""}]}"#,
        )
        .unwrap();
        assert_eq!(
            t.chat
                .get("messages")
                .and_then(JsonValue::as_array)
                .unwrap()
                .len(),
            1
        );
        assert!(t.chat.get("continue_final_message").is_none());
        let t = translate_str(r#"{"tools":[{"name":"f"}],"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"x"}]}"#).unwrap();
        assert_eq!(
            t.chat.get("tool_choice").and_then(JsonValue::as_str),
            Some("required")
        );
        for (body, want) in [
            (r#"{"messages":[]}"#, "at least one"),
            (r#"{"messages":[{"role":"robot","content":"x"}]}"#, "role"),
            (
                r#"{"tools":[{"type":"web_search_20250305","name":"web_search"}],"messages":[{"role":"user","content":"x"}]}"#,
                "server tool",
            ),
            (
                r#"{"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"x"}]}"#,
                "non-empty 'tools'",
            ),
            (
                r#"{"tools":[{"name":"f"}],"tool_choice":{"type":"tool","name":"g"},"messages":[{"role":"user","content":"x"}]}"#,
                "not in 'tools'",
            ),
            (
                r#"{"messages":[{"role":"user","content":[{"type":"image"}]}]}"#,
                "needs a source object",
            ),
            (
                r#"{"messages":[{"role":"user","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]}]}"#,
                "assistant messages",
            ),
            (
                r#"{"max_tokens":-1,"messages":[{"role":"user","content":"x"}]}"#,
                "max_tokens",
            ),
            (
                r#"{"mcp_servers":[{"url":"u"}],"messages":[{"role":"user","content":"x"}]}"#,
                "MCP",
            ),
            (
                r#"{"max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":1023},"messages":[{"role":"user","content":"x"}]}"#,
                "at least 1024",
            ),
            (
                r#"{"max_tokens":2048,"thinking":{"type":"enabled","budget_tokens":2048},"messages":[{"role":"user","content":"x"}]}"#,
                "less than max_tokens",
            ),
            (
                r#"{"max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":1500.5},"messages":[{"role":"user","content":"x"}]}"#,
                "must be an integer",
            ),
        ] {
            let err = translate_str(body).unwrap_err();
            assert!(err.contains(want), "{body}: {err}");
        }
    }

    #[test]
    fn only_enabled_thinking_takes_a_budget() {
        let t = translate_str(
            r#"{"model":"m","max_tokens":4096,"thinking":{"type":"adaptive","budget_tokens":2000},
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(t.chat.get("thinking_budget_tokens").is_none());
        assert!(
            t.ignored.iter().any(|i| i.contains("budget_tokens")),
            "{:?}",
            t.ignored
        );
        let t = translate_str(
            r#"{"model":"m","max_tokens":1025,"thinking":{"type":"enabled","budget_tokens":1024},
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(
            t.chat.get("thinking_budget_tokens").and_then(JsonValue::as_u64),
            Some(1024)
        );
        assert!(t.ignored.is_empty(), "{:?}", t.ignored);
    }

    #[test]
    fn the_engine_s_own_request_fields_reach_the_chat_request() {
        // `cache_prompt: false` asks for a whole prefill (no rows taken over from the device or
        // from host memory): a Messages client must be able to ask for it too
        let t = translate_str(
            r#"{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],
                "cache_prompt":false,"seed":7,"min_p":0.05,"coupled":false}"#,
        )
        .unwrap();
        assert_eq!(t.chat.get("cache_prompt").and_then(JsonValue::as_bool), Some(false));
        assert_eq!(t.chat.get("seed").and_then(JsonValue::as_u64), Some(7));
        assert_eq!(t.chat.get("coupled").and_then(JsonValue::as_bool), Some(false));
        assert!(t.ignored.is_empty(), "{:?}", t.ignored);
        // a request that does not name it leaves the chat endpoint's default (reuse allowed)
        let t = translate_str(
            r#"{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(t.chat.get("cache_prompt").is_none());
        let t = translate_str(
            r#"{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"cache_prompt":true}"#,
        )
        .unwrap();
        assert_eq!(t.chat.get("cache_prompt").and_then(JsonValue::as_bool), Some(true));
    }

    #[test]
    fn signatures_ids_and_errors() {
        for text in ["", "a", "ab", "abc", "need the tool ü😀\n"] {
            assert_eq!(
                signature_reasoning(&thinking_signature(text)).as_deref(),
                Some(text)
            );
        }
        assert_eq!(
            thinking_signature("need the tool"),
            "Gdm1bmVlZCB0aGUgdG9vbA=="
        );
        assert_eq!(signature_reasoning("EqQBCkgIA..."), None);
        assert_eq!(
            (tool_use_id("call_abc"), tool_use_id("xyz")),
            ("toolu_abc".to_string(), "toolu_xyz".to_string())
        );
        assert_eq!(
            error_json(503, "busy"),
            r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#
        );
        assert_eq!(error_type(500), "api_error");
    }

    /// The events of a stream as `(event name, data)`.
    fn events(stream: &str) -> Vec<(String, JsonValue)> {
        stream
            .split("\n\n")
            .filter(|e| !e.is_empty())
            .map(|e| {
                let (name, data) = e.split_once('\n').unwrap();
                (
                    name.strip_prefix("event: ").unwrap().to_string(),
                    parse_json(data.strip_prefix("data: ").unwrap()).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn a_stream_accumulates_to_the_non_streaming_message() {
        let header = |index, id: &str, name: &str| ToolDelta {
            index,
            header: true,
            id: id.into(),
            name: name.into(),
            arguments: String::new(),
        };
        let frag = |index, a: &str| ToolDelta {
            index,
            arguments: a.into(),
            ..ToolDelta::default()
        };
        let deltas = vec![
            Delta {
                reasoning: "I should ".into(),
                ..Delta::default()
            },
            Delta {
                reasoning: "check.".into(),
                content: "Let me ".into(),
                ..Delta::default()
            },
            Delta {
                content: "look.".into(),
                tool_deltas: vec![header(0, "call_A", "get_weather"), frag(0, "{\"city\":")],
                ..Delta::default()
            },
            Delta {
                tool_deltas: vec![frag(0, "\"Oslo\"}"), header(1, "call_B", "now")],
                ..Delta::default()
            },
        ];
        let mut b = Builder::new(true);
        let mut stream = Builder::message_start("msg_1", "m", 12);
        for d in &deltas {
            stream.push_str(&b.feed(d));
        }
        let usage = usage_json(12, 30);
        stream.push_str(&b.finish("tool_use", "", &usage, ",\"x\":1"));
        // closing again owes nothing
        assert_eq!(b.close(), "");

        let ev = events(&stream);
        let names: Vec<&str> = ev.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        // every block starts once and stops once, with growing indices
        let idx = |name: &str| -> Vec<u64> {
            ev.iter()
                .filter(|(n, _)| n == name)
                .map(|(_, d)| d.get("index").and_then(JsonValue::as_u64).unwrap())
                .collect()
        };
        assert_eq!(idx("content_block_start"), [0, 1, 2, 3]);
        assert_eq!(idx("content_block_stop"), [0, 1, 2, 3]);
        // the deltas of each block rebuild the final message's block
        let msg = parse_json(&b.message("msg_1", "m", "tool_use", "", &usage, "")).unwrap();
        let blocks = msg.get("content").and_then(JsonValue::as_array).unwrap();
        let joined = |i: u64, kind: &str, field: &str| -> String {
            ev.iter()
                .filter(|(n, d)| {
                    n == "content_block_delta"
                        && d.get("index").and_then(JsonValue::as_u64) == Some(i)
                })
                .filter_map(|(_, d)| {
                    d.get("delta")
                        .filter(|x| x.get("type").and_then(JsonValue::as_str) == Some(kind))
                })
                .map(|x| {
                    x.get(field)
                        .and_then(JsonValue::as_str)
                        .unwrap()
                        .to_string()
                })
                .collect()
        };
        assert_eq!(
            blocks[0].get("thinking").and_then(JsonValue::as_str),
            Some("I should check.")
        );
        assert_eq!(joined(0, "thinking_delta", "thinking"), "I should check.");
        assert_eq!(
            blocks[0]
                .get("signature")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            Some(joined(0, "signature_delta", "signature"))
        );
        assert_eq!(
            blocks[1].get("text").and_then(JsonValue::as_str),
            Some("Let me look.")
        );
        assert_eq!(joined(1, "text_delta", "text"), "Let me look.");
        assert_eq!(
            blocks[2].dump(),
            r#"{"type":"tool_use","id":"toolu_A","name":"get_weather","input":{"city":"Oslo"}}"#
        );
        assert_eq!(
            joined(2, "input_json_delta", "partial_json"),
            "{\"city\":\"Oslo\"}"
        );
        assert_eq!(
            blocks[3].dump(),
            r#"{"type":"tool_use","id":"toolu_B","name":"now","input":{}}"#
        );
        assert_eq!(
            msg.get("stop_reason").and_then(JsonValue::as_str),
            Some("tool_use")
        );
        assert_eq!(
            msg.get("usage")
                .and_then(|u| u.get("output_tokens"))
                .and_then(JsonValue::as_u64),
            Some(30)
        );
        let (_, md) = &ev[ev.len() - 2];
        assert_eq!(md.get("x").and_then(JsonValue::as_u64), Some(1));
        assert_eq!(
            md.get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(JsonValue::as_str),
            Some("tool_use")
        );

        // thinking not shown: the block carries only its signature
        let mut hidden = Builder::new(false);
        let out = hidden.feed(&Delta {
            reasoning: "secret".into(),
            ..Delta::default()
        }) + &hidden.finish("end_turn", "", &usage, "");
        assert!(!out.contains("thinking_delta") && out.contains("signature_delta"));
        let msg = parse_json(&hidden.message("m", "m", "end_turn", "END", &usage, "")).unwrap();
        let block = &msg.get("content").and_then(JsonValue::as_array).unwrap()[0];
        assert_eq!(block.get("thinking").and_then(JsonValue::as_str), Some(""));
        assert_eq!(
            block
                .get("signature")
                .and_then(JsonValue::as_str)
                .and_then(signature_reasoning)
                .as_deref(),
            Some("secret")
        );
        assert_eq!(
            msg.get("stop_sequence").and_then(JsonValue::as_str),
            Some("END")
        );
    }
}
