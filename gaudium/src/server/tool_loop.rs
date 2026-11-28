//! The server-side tool loop: a Messages request whose tools the server runs itself (the MCP
//! connector, `mcp_servers`) is answered over several generation rounds. After each round the
//! server runs the round's calls of its own tools, puts their results into the conversation and
//! generates again, until the model ends its turn, calls a client tool, or the rounds run out.
//!
//! The reply is one message: its content blocks go on across the rounds (`mcp_tool_use` where
//! the model called a server-side tool, then its `mcp_tool_result`), streamed in that order, and
//! its usage sums every round.
//!
//! Every decision of the loop is verified (`verified/server_tools.bend`): whether a request with
//! tool servers runs, where a call goes, whether a round's calls run and whether another round
//! follows, and the bounds on calls and result sizes. The runner of the tools is a trait
//! ([`ServerTools`]), so other tools the server runs itself use the same loop.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::anthropic::{self, Translated};
use super::conn::Conn;
use super::mcp_client::{Connector, BOUNDS};
use super::outparse::ToolCall;
use super::proxy::Caller;
use super::stream_out::{KeepAlive, SseSink};
use super::{run_chat, write_json, Api, AppState, CancelEntry};
use crate::core::tool_calls::{stop_reason_name, GenEnd};
use crate::json::{parse_json, JsonValue};
use crate::verified::{
    ge_model_ended, pc_prefill_rows, pc_reused_rows, sv_admit, sv_admit_is_none, sv_admit_runs,
    sv_budget_spent, sv_call_runs, sv_calls_take, sv_goes_on_live, sv_next, sv_route,
    sv_route_clash, sv_route_server, sv_runs, sv_tokens_left, sv_waits_client, SvAdmit,
};

/// Tools the server runs itself while it answers a request.
pub(super) trait ServerTools {
    /// `name` (as the model sees it) is one of these tools.
    fn owns(&self, name: &str) -> bool;
    /// Runs one call. A call that fails is an error result for the model, not an error of the
    /// request.
    fn run(&mut self, name: &str, arguments: &JsonValue) -> ToolOutput;
    /// The request is over (sessions are closed).
    fn finish(&mut self);
}

/// The result of one call: text parts, and whether the tool reported an error.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ToolOutput {
    pub is_error: bool,
    pub texts: Vec<String>,
}

impl ToolOutput {
    /// The result as the next round's prompt has it (and as a replayed history has it,
    /// [`anthropic::mcp_result_text`]).
    pub(super) fn prompt_text(&self) -> String {
        anthropic::mcp_result_text(self.is_error, &self.texts)
    }
}

/// What the rounds of one reply share (`run_chat` takes it for each round).
pub(super) struct Rounds {
    /// The reply's id: the one in its `message_start` (or its message), which every round goes by
    /// and the client cancels by.
    pub id: String,
    /// What a cancel by `id` reaches, registered for the whole reply: the reply, and the round
    /// that runs.
    pub cancel: Arc<CancelEntry>,
    /// The reply's stream, once its first round began it.
    pub sink: Option<SseSink>,
    /// The stream's keep-alive, from the first round to the end of the reply: it also runs while
    /// the server waits for its tools.
    pub keep_alive: Option<KeepAlive>,
    /// The reply's content blocks so far.
    pub builder: Option<anthropic::Builder>,
    /// How the last round ended; `None` after a round that answered the request itself (an
    /// error).
    pub end: Option<RoundEnd>,
}

/// The reply's entry in the server's cancel table, from before its first round to the end of its
/// last.
struct Registered<'a> {
    state: &'a AppState,
    id: String,
}

impl Drop for Registered<'_> {
    fn drop(&mut self) {
        // a poisoned lock (a panic elsewhere) must not abort the process while unwinding
        self.state
            .cancels
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
    }
}

/// One generation round's outcome.
pub(super) struct RoundEnd {
    /// The round's model.
    pub model: String,
    pub end: GenEnd,
    /// Every whole call the model made in the round.
    pub calls: Vec<ToolCall>,
    pub content: String,
    pub reasoning: String,
    pub stop_sequence: String,
    pub prompt_tokens: u64,
    pub reused: u64,
    pub output_tokens: u64,
    /// The last of `calls` was cut by the end of generation and closed by the parser: the model
    /// never finished it.
    pub repaired: bool,
    /// The round's `timings` and `gaudium` fields (`,"key":value` text).
    pub extra: String,
    /// Writing the stream to the client failed.
    pub write_failed: bool,
}

/// The tool names of a chat body (offered and held back).
fn chat_tool_names(chat: &JsonValue) -> Vec<String> {
    ["tools", "gaudium_hidden_tools"]
        .iter()
        .filter_map(|k| chat.get(k).and_then(JsonValue::as_array))
        .flatten()
        .filter_map(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(JsonValue::as_str)
                .map(str::to_string)
        })
        .collect()
}

/// Appends `items` to the array `key` of a chat body (made when absent).
fn extend_list(chat: &mut JsonValue, key: &str, items: Vec<JsonValue>) {
    if items.is_empty() {
        return;
    }
    let JsonValue::Object(fields) = chat else {
        return;
    };
    match fields.iter_mut().find(|(k, _)| k == key) {
        Some((_, JsonValue::Array(a))) => a.extend(items),
        Some((_, v)) => *v = JsonValue::Array(items),
        None => fields.push((key.to_string(), JsonValue::Array(items))),
    }
}

fn set_field(chat: &mut JsonValue, key: &str, value: Option<JsonValue>) {
    let JsonValue::Object(fields) = chat else {
        return;
    };
    fields.retain(|(k, _)| k != key);
    if let Some(v) = value {
        fields.push((key.to_string(), v));
    }
}

/// Why a request with server-side tools is refused before it runs: the HTTP status, the error
/// type and the message.
#[derive(Debug)]
pub(super) struct Refused {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
}

impl From<String> for Refused {
    fn from(message: String) -> Refused {
        Refused {
            status: 400,
            kind: "invalid_request_error",
            message,
        }
    }
}

/// Opens the request's MCP servers and adds their tools to its chat body (`None`: it names
/// none). `caller`: the verdicts of the HTTP request that carries it; `body`: the Messages body
/// `t` was translated from. The error is the reason the request is refused; nothing is sent to
/// any server before admission lets it run.
pub(super) fn open_server_tools<'a>(
    state: &'a AppState,
    caller: Caller,
    body: &JsonValue,
    t: &mut Translated,
) -> Result<Option<Connector<'a>>, Refused> {
    let asked = t.mcp.len() as u64;
    let admit = sv_admit(asked, state.proxy.enabled(), caller.site_ok, BOUNDS.servers);
    if sv_admit_is_none(admit) {
        return Ok(None);
    }
    if !sv_admit_runs(admit) {
        return Err(match admit {
            SvAdmit::SvProxyOff => "mcp_servers: this server was started without --mcp-proxy. \
                 The MCP connector's requests go out through the MCP proxy and its allow-list \
                 (--mcp-proxy, --mcp-proxy-allow, --mcp-proxy-allow-private); or run the MCP \
                 tools in the client"
                .to_string()
                .into(),
            SvAdmit::SvForeign => Refused {
                status: 403,
                kind: "permission_error",
                message: "mcp_servers: the request comes from a page of another site, not from \
                          this server's own page; a page of another site may not make this \
                          server reach MCP servers (run the MCP tools in the client)"
                    .into(),
            },
            SvAdmit::SvTooMany | SvAdmit::SvRun | SvAdmit::SvNone => format!(
                "mcp_servers: {asked} servers named; this server takes at most {} per request",
                BOUNDS.servers
            )
            .into(),
        });
    }
    let client = chat_tool_names(&t.chat);
    let c = Connector::open(&state.proxy, caller, &t.mcp)?;
    if let Some(name) = c
        .names()
        .find(|n| sv_route_clash(sv_route(true, client.iter().any(|x| x == n))))
    {
        return Err(format!(
            "tools: '{name}' is the name of a client tool and of a tool of the request's MCP servers"
        )
        .into());
    }
    // a tool search result in the history may name one of the servers' tools (a deferred one
    // above all): the history is translated again, now that their definitions are known
    if anthropic::has_tool_reference(body) {
        let known: Vec<JsonValue> = c.offered.iter().chain(&c.deferred).cloned().collect();
        *t = anthropic::translate_with(body, &known)?;
    }
    extend_list(&mut t.chat, "tools", c.offered.clone());
    extend_list(&mut t.chat, "gaudium_hidden_tools", c.deferred.clone());
    t.ignored.extend(c.notes.iter().cloned());
    Ok(Some(c))
}

/// `POST /v1/messages` once translated: a request without tool servers runs as one round, as
/// any chat request; one with them runs the tool loop. `caller`: the verdicts of the HTTP
/// request.
pub(super) fn run_messages(
    state: &Arc<AppState>,
    caller: Caller,
    body: &JsonValue,
    mut t: Translated,
    stream: &mut Conn,
) {
    let api = Api::Messages {
        show_thinking: t.show_thinking,
    };
    let client = chat_tool_names(&t.chat);
    let mut tools = match open_server_tools(state, caller, body, &mut t) {
        Ok(Some(c)) => c,
        Ok(None) => {
            run_chat(state, &t.chat, stream, api, t.ignored, None);
            return;
        }
        Err(e) => {
            api.error(stream, e.status, e.kind, &e.message);
            return;
        }
    };
    let labels = tools.labels();
    run_loop(state, t, stream, &mut tools, &client, labels);
    tools.finish();
}

/// The next round's conversation: the round's assistant turn with its server-side calls, and a
/// tool message with each call's result.
fn next_round_chat(
    chat: &mut JsonValue,
    end: &RoundEnd,
    calls: Vec<JsonValue>,
    results: Vec<JsonValue>,
) {
    let mut content = end.content.clone();
    let mut reasoning = end.reasoning.clone();
    let JsonValue::Object(fields) = chat else {
        return;
    };
    let prefilled = fields
        .iter()
        .any(|(k, v)| k == "continue_final_message" && v.as_bool() == Some(true));
    let Some((_, JsonValue::Array(msgs))) = fields.iter_mut().find(|(k, _)| k == "messages") else {
        return;
    };
    // a prefilled reply goes on from the prefill's text
    if prefilled {
        if let Some(last) = msgs.pop() {
            let text = |k: &str| {
                last.get(k)
                    .and_then(JsonValue::as_str)
                    .unwrap_or("")
                    .to_string()
            };
            content = text("content") + &content;
            reasoning = text("reasoning_content") + &reasoning;
        }
    }
    let mut m = vec![
        ("role".to_string(), JsonValue::Str("assistant".into())),
        ("content".to_string(), JsonValue::Str(content)),
    ];
    if !reasoning.is_empty() {
        m.push(("reasoning_content".to_string(), JsonValue::Str(reasoning)));
    }
    m.push(("tool_calls".to_string(), JsonValue::Array(calls)));
    msgs.push(JsonValue::Object(m));
    msgs.extend(results);
    // the next round is a new assistant turn after the results; a forced tool choice was the
    // first round's
    for k in [
        "continue_final_message",
        "add_generation_prompt",
        "tool_choice",
    ] {
        set_field(chat, k, None);
    }
}

/// The rounds of one reply.
fn run_loop(
    state: &Arc<AppState>,
    t: Translated,
    stream: &mut Conn,
    tools: &mut dyn ServerTools,
    client: &[String],
    labels: Vec<(String, String, String)>,
) {
    let Translated {
        mut chat,
        show_thinking,
        ignored,
        ..
    } = t;
    let stream_mode = chat.get("stream").and_then(JsonValue::as_bool) == Some(true);
    // the request's output limit holds for the whole reply, over every round (0 or none: the
    // chat path's own limit, the end of the context)
    let limit = chat
        .get("max_tokens")
        .and_then(JsonValue::as_u64)
        .filter(|&v| v > 0);
    // one id for the whole reply, registered for cancels until its last round ends
    let id = format!(
        "msg_gaudium_{}",
        state.next_req_id.fetch_add(1, Ordering::Relaxed)
    );
    let cancel = CancelEntry::rounds();
    state
        .cancels
        .lock()
        .unwrap()
        .insert(id.clone(), Arc::clone(&cancel));
    let _registered = Registered {
        state,
        id: id.clone(),
    };
    let mut rounds = Rounds {
        id: id.clone(),
        cancel: Arc::clone(&cancel),
        sink: None,
        keep_alive: None,
        builder: Some(anthropic::Builder::new(show_thinking).with_server_tools(labels)),
        end: None,
    };
    let api = Api::Messages { show_thinking };
    let (mut done, mut calls_run) = (0u64, 0u64);
    let (mut input, mut cached, mut output) = (0u64, 0u64, 0u64);
    let mut model: Option<String> = None;
    loop {
        run_chat(
            state,
            &chat,
            stream,
            api,
            ignored.clone(),
            Some(&mut rounds),
        );
        let Some(end) = rounds.end.take() else {
            return; // the round answered the request itself (an error)
        };
        done += 1;
        input += pc_prefill_rows(end.prompt_tokens, end.reused);
        cached += pc_reused_rows(end.prompt_tokens, end.reused);
        output += end.output_tokens;
        let left = sv_tokens_left(limit.unwrap_or(0), output);
        let model = model.get_or_insert_with(|| end.model.clone()).clone();
        // the call the parser closed at the end of generation, if any, is the round's last
        let cut = end.calls.len().checked_sub(1).filter(|_| end.repaired);
        let (server, client_calls): (Vec<_>, Vec<_>) =
            end.calls.iter().enumerate().partition(|(_, c)| {
                sv_route_server(sv_route(
                    tools.owns(&c.name),
                    client.iter().any(|n| *n == c.name),
                ))
            });
        let next = sv_next(
            ge_model_ended(end.end),
            server.len() as u64,
            client_calls.len() as u64,
            done,
            BOUNDS.rounds,
        );
        let sink = rounds.sink.clone();
        let Some(b) = rounds.builder.as_mut() else {
            return;
        };
        let mut write_failed = end.write_failed;
        let mut stop_sequence = String::new();
        let stop = if !sv_runs(next) {
            stop_sequence = end.stop_sequence.clone();
            stop_reason_name(end.end, !end.calls.is_empty())
        } else {
            // a call the parser closed at the end of generation never runs, and does not count
            // against the request's calls
            let whole = server.iter().filter(|(k, _)| Some(*k) != cut).count();
            let take = sv_calls_take(calls_run, whole as u64, BOUNDS.calls);
            calls_run += take;
            let (mut calls, mut results) = (Vec::new(), Vec::new());
            for (i, &(k, c)) in server.iter().enumerate() {
                if write_failed {
                    return; // the client is gone: no further call is sent anywhere
                }
                let block_id = b.call_block_id(&c.id, &c.name);
                let args = parse_json(&c.arguments)
                    .ok()
                    .filter(|v| v.as_object().is_some());
                let is_whole = Some(k) != cut;
                let cancelled = cancel.cancelled();
                let out = match (&args, sv_call_runs(is_whole, cancelled, i as u64, take)) {
                    (Some(a), true) => tools.run(&c.name, a),
                    (None, true) => ToolOutput::error("the call's arguments are not a JSON object"),
                    // why it did not run, for the model
                    (_, false) if !is_whole => ToolOutput::error(
                        "not run: the end of generation cut the call off before it was complete",
                    ),
                    (_, false) if cancelled => {
                        ToolOutput::error("not run: the request was cancelled")
                    }
                    (_, false) => ToolOutput::error(format!(
                        "not run: this request reached its limit of {} server-side tool calls",
                        BOUNDS.calls
                    )),
                };
                let events = b.push_result(&block_id, out.is_error, &out.texts);
                if let Some(s) = &sink {
                    if !write_failed && s.write(events.as_bytes()).is_err() {
                        write_failed = true;
                    }
                }
                calls.push(JsonValue::Object(vec![
                    ("id".to_string(), JsonValue::Str(block_id.clone())),
                    ("type".to_string(), JsonValue::Str("function".into())),
                    (
                        "function".to_string(),
                        JsonValue::Object(vec![
                            ("name".to_string(), JsonValue::Str(c.name.clone())),
                            (
                                "arguments".to_string(),
                                args.unwrap_or(JsonValue::Object(Vec::new())),
                            ),
                        ]),
                    ),
                ]));
                results.push(JsonValue::Object(vec![
                    ("role".to_string(), JsonValue::Str("tool".into())),
                    ("tool_call_id".to_string(), JsonValue::Str(block_id)),
                    ("content".to_string(), JsonValue::Str(out.prompt_text())),
                ]));
            }
            if write_failed {
                return; // the client is gone
            }
            let cancelled = cancel.cancelled();
            if sv_goes_on_live(cancelled, next, limit.is_some(), left) {
                next_round_chat(&mut chat, &end, calls, results);
                if limit.is_some() {
                    set_field(&mut chat, "max_tokens", Some(JsonValue::Int(left as i64)));
                }
                continue;
            }
            if cancelled {
                // cancelled between the round's calls or after them: the reply ends here
                stop_reason_name(GenEnd::GeCancelled, !client_calls.is_empty())
            } else if sv_budget_spent(next, limit.is_some(), left) {
                "max_tokens" // the request's output limit is spent over the rounds
            } else if sv_waits_client(next) {
                "tool_use" // the client's own calls wait for it
            } else {
                "pause_turn" // no round is left: the client may send the reply back to go on
            }
        };
        let usage = anthropic::usage_json_parts(input, cached, output);
        let extra = format!(
            "{},\"gaudium_tool_loop\":{{\"rounds\":{done},\"server_calls\":{calls_run}}}",
            end.extra
        );
        if let Some(s) = &sink {
            if !write_failed {
                let events = b.finish(stop, &stop_sequence, &usage, &extra);
                if s.write(events.as_bytes()).is_ok() {
                    s.end();
                }
            }
        } else if !stream_mode {
            b.close();
            write_json(
                stream,
                200,
                &b.message(&id, &model, stop, &stop_sequence, &usage, &extra),
            );
        }
        return;
    }
}
