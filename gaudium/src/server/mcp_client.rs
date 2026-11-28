//! An MCP client for the tools the server runs itself (the Messages MCP connector,
//! `mcp_servers`): initialize, `tools/list` and `tools/call` over Streamable HTTP, with answers
//! as JSON or as an event stream and the session the server assigns (`Mcp-Session-Id`).
//!
//! Every request goes out through the MCP proxy's rules (`McpProxy::fetch`): the same switch, the
//! same allow-lists, address checks, redirect and header rules, and the same size and time caps
//! as the Web UI's requests through `/cors-proxy`. A server's `authorization_token` goes to it as
//! a bearer header only; it is never logged, and no message of this module repeats it.
//!
//! How many pages of a tool list are read, how many calls a request makes, and how much of a
//! result goes back into the prompt are verified decisions (`verified/server_tools.bend`).

use super::proxy::{Caller, Fetched, McpProxy};
use super::tool_loop::{ServerTools, ToolOutput};
use crate::core::proxy::Method;
use crate::json::{parse_json, parse_json_bytes, JsonValue};
use crate::verified::{sv_list_more, sv_result_keep};

/// The protocol revision this client asks for first.
const PROTOCOL: &str = "2025-06-18";
/// Revisions a server may answer with that this client speaks (all of them carry Streamable
/// HTTP and the three methods it uses).
const PROTOCOLS: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];

/// The bounds of one request's server-side tools. The rules that use them are verified; the
/// values are tuning (pinned by the tests below).
#[derive(Debug, Clone, Copy)]
pub(super) struct Bounds {
    /// Tool servers one request may name.
    pub servers: u64,
    /// Generation rounds of one request.
    pub rounds: u64,
    /// Server-side calls of one request.
    pub calls: u64,
    /// Bytes of one tool result put back into the prompt.
    pub result_bytes: u64,
    /// Pages of one server's tool list.
    pub list_pages: u64,
}

pub(super) const BOUNDS: Bounds = Bounds {
    servers: 8,
    rounds: 10,
    calls: 64,
    result_bytes: 128 * 1024,
    list_pages: 16,
};

/// Which of a server's tools the model sees, as the request configures it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ToolFilter {
    /// Tools without a setting of their own are offered.
    pub enabled: bool,
    /// Tools without a setting of their own are offered on request only (tool search).
    pub defer: bool,
    /// Settings of single tools: `(name, enabled, defer_loading)`.
    pub configs: Vec<(String, Option<bool>, Option<bool>)>,
}

impl Default for ToolFilter {
    fn default() -> Self {
        Self {
            enabled: true,
            defer: false,
            configs: Vec::new(),
        }
    }
}

impl ToolFilter {
    fn config(&self, name: &str) -> Option<&(String, Option<bool>, Option<bool>)> {
        self.configs.iter().find(|c| c.0 == name)
    }

    /// The tool is offered at all (a tool's own setting wins over the default).
    pub fn offers(&self, name: &str) -> bool {
        self.config(name).and_then(|c| c.1).unwrap_or(self.enabled)
    }

    /// The tool's definition is held back until a tool search finds it.
    pub fn defers(&self, name: &str) -> bool {
        self.config(name).and_then(|c| c.2).unwrap_or(self.defer)
    }
}

/// One entry of a request's `mcp_servers`.
#[derive(Clone, PartialEq)]
pub(super) struct ServerSpec {
    pub name: String,
    pub url: String,
    pub token: Option<String>,
    pub filter: ToolFilter,
}

impl std::fmt::Debug for ServerSpec {
    // the token is never printed
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerSpec")
            .field("name", &self.name)
            .field("url", &self.url)
            .field("token", &self.token.as_ref().map(|_| "<set>"))
            .field("filter", &self.filter)
            .finish()
    }
}

/// A tool of an MCP server.
#[derive(Debug, Clone)]
pub(super) struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: JsonValue,
}

/// The name the model sees for a server's tool: `mcp__<server>__<tool>`, with every character
/// outside letters, digits, `_` and `-` written as `_` (chat templates and tool-call parsers
/// take such names).
pub(super) fn model_name(server: &str, tool: &str) -> String {
    let clean = |s: &str| -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    };
    format!("mcp__{}__{}", clean(server), clean(tool))
}

/// A session with one MCP server.
pub(super) struct Session<'a> {
    proxy: &'a McpProxy,
    /// The request the session is opened for: every exchange is decided on its verdicts.
    caller: Caller,
    url: String,
    token: Option<String>,
    session: Option<String>,
    protocol: Option<String>,
    next_id: u64,
}

/// What came back for one JSON-RPC request.
enum Reply {
    Result(JsonValue),
    /// A JSON-RPC error: its message.
    Error(String),
    /// The server answered with this HTTP status (not a success).
    Http(u16),
}

impl Reply {
    /// The result, or what went wrong as a message.
    fn result(self, method: &str) -> Result<JsonValue, String> {
        match self {
            Reply::Result(r) => Ok(r),
            Reply::Error(e) => Err(format!("{method}: {e}")),
            Reply::Http(s @ (401 | 403)) => Err(format!(
                "{method}: the server refused the request (HTTP {s}): check authorization_token"
            )),
            Reply::Http(s) => Err(format!("{method}: HTTP {s}")),
        }
    }
}

impl<'a> Session<'a> {
    /// Connects and runs the initialization handshake.
    pub fn open(
        proxy: &'a McpProxy,
        caller: Caller,
        url: &str,
        token: Option<&str>,
    ) -> Result<Self, String> {
        let mut s = Session {
            proxy,
            caller,
            url: url.to_string(),
            token: token.map(str::to_string),
            session: None,
            protocol: None,
            next_id: 1,
        };
        s.initialize()?;
        Ok(s)
    }

    fn initialize(&mut self) -> Result<(), String> {
        self.session = None;
        self.protocol = None;
        let params = format!(
            "{{\"protocolVersion\":\"{PROTOCOL}\",\"capabilities\":{{}},\
             \"clientInfo\":{{\"name\":\"gaudium\",\"version\":\"{}\"}}}}",
            env!("CARGO_PKG_VERSION")
        );
        let (reply, fetched) = self.request("initialize", &params)?;
        let result = reply.result("initialize")?;
        let version = result
            .get("protocolVersion")
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if !PROTOCOLS.contains(&version) {
            return Err(format!(
                "the server speaks MCP revision '{}', which this client does not ({} are)",
                version.chars().take(40).collect::<String>(),
                PROTOCOLS.join(", ")
            ));
        }
        self.protocol = Some(version.to_string());
        self.session = fetched
            .header("mcp-session-id")
            .filter(|v| !v.is_empty() && v.bytes().all(|b| (0x21..=0x7e).contains(&b)))
            .map(str::to_string);
        self.notify("notifications/initialized")
    }

    fn headers(&self) -> Vec<(String, String)> {
        let mut h = vec![
            (
                "accept".to_string(),
                "application/json, text/event-stream".to_string(),
            ),
            ("content-type".to_string(), "application/json".to_string()),
            ("user-agent".to_string(), "gaudium-mcp-client".to_string()),
        ];
        if let Some(v) = &self.protocol {
            h.push(("mcp-protocol-version".to_string(), v.clone()));
        }
        if let Some(v) = &self.session {
            h.push(("mcp-session-id".to_string(), v.clone()));
        }
        if let Some(t) = &self.token {
            h.push(("authorization".to_string(), format!("Bearer {t}")));
        }
        h
    }

    /// A notification (the server answers 202 with no body).
    fn notify(&mut self, method: &str) -> Result<(), String> {
        let body = format!("{{\"jsonrpc\":\"2.0\",\"method\":\"{method}\"}}");
        let fetched = self.proxy.fetch(
            self.caller,
            Method::PxPost,
            &self.url,
            &self.headers(),
            Some(body.as_bytes()),
            &mut |_, _| true,
        )?;
        if (200..300).contains(&fetched.status) {
            Ok(())
        } else {
            Err(format!("{method}: HTTP {}", fetched.status))
        }
    }

    /// One JSON-RPC request; its answer from a JSON body or from the event stream (other
    /// messages on the stream, requests and notifications of the server, are passed over).
    fn request(&mut self, method: &str, params: &str) -> Result<(Reply, Fetched), String> {
        let id = self.next_id;
        self.next_id += 1;
        let body = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"{method}\",\"params\":{params}}}"
        );
        let mut raw: Vec<u8> = Vec::new();
        let mut events = EventLines::default();
        let mut found: Option<JsonValue> = None;
        let fetched = self.proxy.fetch(
            self.caller,
            Method::PxPost,
            &self.url,
            &self.headers(),
            Some(body.as_bytes()),
            &mut |f, data| {
                if !is_event_stream(f) {
                    raw.extend_from_slice(data);
                    return true;
                }
                for msg in events.push(data) {
                    if let Some(v) = parse_json(&msg).ok().filter(|v| answers(v, id)) {
                        found = Some(v);
                        return false;
                    }
                }
                true
            },
        )?;
        if !(200..300).contains(&fetched.status) {
            return Ok((Reply::Http(fetched.status), fetched));
        }
        let msg = match found {
            Some(v) => v,
            None if !is_event_stream(&fetched) => {
                let v = parse_json_bytes(&raw)
                    .map_err(|_| format!("{method}: the answer is not JSON"))?;
                // a batch answer holds the response among others
                match v {
                    JsonValue::Array(items) => items
                        .into_iter()
                        .find(|x| answers(x, id))
                        .ok_or_else(|| format!("{method}: no answer to the request"))?,
                    v => v,
                }
            }
            None => {
                return Err(format!(
                    "{method}: the event stream ended without an answer"
                ))
            }
        };
        if !answers(&msg, id) {
            return Err(format!("{method}: the answer is for another request"));
        }
        if let Some(e) = msg.get("error") {
            let m = e
                .get("message")
                .and_then(JsonValue::as_str)
                .unwrap_or("error");
            return Ok((Reply::Error(m.chars().take(500).collect()), fetched));
        }
        let result = msg.get("result").cloned().unwrap_or(JsonValue::Null);
        Ok((Reply::Result(result), fetched))
    }

    /// A request in the session; a session the server ended (404) is opened again once.
    fn call_rpc(&mut self, method: &str, params: &str) -> Result<Reply, String> {
        let had_session = self.session.is_some();
        match self.request(method, params)?.0 {
            Reply::Http(404) if had_session => {
                self.initialize()?;
                Ok(self.request(method, params)?.0)
            }
            r => Ok(r),
        }
    }

    /// Every tool the server lists, up to the page bound; `true` with them when the bound cut
    /// the list.
    pub fn list_tools(&mut self, most_pages: u64) -> Result<(Vec<Tool>, bool), String> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0u64;
        loop {
            let params = match &cursor {
                Some(c) => format!("{{\"cursor\":{}}}", crate::json::quote_json(c)),
                None => "{}".to_string(),
            };
            let result = self.call_rpc("tools/list", &params)?.result("tools/list")?;
            pages += 1;
            for t in result
                .get("tools")
                .and_then(JsonValue::as_array)
                .unwrap_or(&[])
            {
                let Some(name) = t.get("name").and_then(JsonValue::as_str) else {
                    continue;
                };
                let schema = match t.get("inputSchema") {
                    Some(s @ JsonValue::Object(_)) => s.clone(),
                    _ => parse_json(r#"{"type":"object","properties":{}}"#).expect("valid"),
                };
                tools.push(Tool {
                    name: name.to_string(),
                    description: t
                        .get("description")
                        .and_then(JsonValue::as_str)
                        .map(str::to_string),
                    input_schema: schema,
                });
            }
            cursor = result
                .get("nextCursor")
                .and_then(JsonValue::as_str)
                .filter(|c| !c.is_empty())
                .map(str::to_string);
            if !sv_list_more(cursor.is_some(), pages, most_pages) {
                return Ok((tools, cursor.is_some()));
            }
        }
    }

    /// Runs one tool. A call the server answers with an error, or that fails on the way, is an
    /// error result for the model.
    pub fn call(&mut self, tool: &str, arguments: &JsonValue) -> ToolOutput {
        let params = format!(
            "{{\"name\":{},\"arguments\":{}}}",
            crate::json::quote_json(tool),
            arguments.dump()
        );
        match self
            .call_rpc("tools/call", &params)
            .and_then(|r| r.result("tools/call"))
        {
            Ok(r) => result_output(&r),
            Err(e) => ToolOutput::error(e),
        }
    }

    /// Ends the session (the server may not allow it; that is not an error).
    pub fn close(&mut self) {
        if self.session.is_none() {
            return;
        }
        let _ = self.proxy.fetch(
            self.caller,
            Method::PxDelete,
            &self.url,
            &self.headers(),
            None,
            &mut |_, _| true,
        );
        self.session = None;
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        self.close();
    }
}

fn is_event_stream(f: &Fetched) -> bool {
    f.header("content-type").is_some_and(|v| {
        v.trim()
            .to_ascii_lowercase()
            .starts_with("text/event-stream")
    })
}

/// The JSON-RPC message is the response to request `id`.
fn answers(v: &JsonValue, id: u64) -> bool {
    v.get("id").and_then(JsonValue::as_u64) == Some(id)
        && (v.get("result").is_some() || v.get("error").is_some())
}

/// Server-sent events cut into their data payloads.
#[derive(Default)]
struct EventLines {
    line: Vec<u8>,
    data: String,
}

impl EventLines {
    /// Takes the next bytes; returns the data of every event they complete.
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        for &b in bytes {
            if b != b'\n' {
                self.line.push(b);
                continue;
            }
            let line = String::from_utf8_lossy(&self.line)
                .trim_end_matches('\r')
                .to_string();
            self.line.clear();
            if line.is_empty() {
                if !self.data.is_empty() {
                    out.push(std::mem::take(&mut self.data));
                }
            } else if let Some(d) = line.strip_prefix("data:") {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(d.strip_prefix(' ').unwrap_or(d));
            }
        }
        out
    }
}

/// A `tools/call` result as text blocks for the reply and the prompt.
fn result_output(r: &JsonValue) -> ToolOutput {
    let mut texts: Vec<String> = Vec::new();
    for item in r
        .get("content")
        .and_then(JsonValue::as_array)
        .unwrap_or(&[])
    {
        let ty = item.get("type").and_then(JsonValue::as_str).unwrap_or("");
        let text = match ty {
            "text" => item
                .get("text")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string(),
            "resource" => match item.get("resource") {
                Some(res) => match res.get("text").and_then(JsonValue::as_str) {
                    Some(t) => t.to_string(),
                    None => format!(
                        "[resource {}]",
                        res.get("uri").and_then(JsonValue::as_str).unwrap_or("")
                    ),
                },
                None => continue,
            },
            "resource_link" => format!(
                "[resource {}]",
                item.get("uri").and_then(JsonValue::as_str).unwrap_or("")
            ),
            "image" | "audio" => format!(
                "[{ty} ({}) not shown: this server passes tool results on as text]",
                item.get("mimeType")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("unknown type")
            ),
            _ => continue,
        };
        texts.push(text);
    }
    if texts.is_empty() {
        if let Some(sc) = r.get("structuredContent") {
            texts.push(sc.dump());
        }
    }
    let is_error = r.get("isError").and_then(JsonValue::as_bool) == Some(true);
    ToolOutput { is_error, texts }.within(BOUNDS.result_bytes)
}

impl ToolOutput {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            is_error: true,
            texts: vec![message.into()],
        }
    }

    /// The result cut to `cap` bytes of the text it puts into the prompt (`sv_result_keep`),
    /// at a character boundary, with a note where it was cut. The length measured is that of
    /// the parts as the prompt joins them (`ToolOutput::prompt_text`), separators included, so
    /// many short or empty parts cannot pass the cap either. A result that is cut becomes one
    /// part (the joined text, cut) and the note.
    pub fn within(mut self, cap: u64) -> Self {
        let joined = self.texts.join("\n");
        let total = joined.len() as u64;
        let keep = sv_result_keep(total, cap);
        if keep == total {
            return self;
        }
        let mut end = (keep as usize).min(joined.len());
        while !joined.is_char_boundary(end) {
            end -= 1;
        }
        let mut kept = Vec::new();
        if end > 0 {
            kept.push(joined[..end].to_string());
        }
        kept.push(format!(
            "[the result was cut: at most {keep} of its {total} bytes are shown]"
        ));
        self.texts = kept;
        self
    }
}

/// The MCP servers of one request: their sessions and the tools the model sees.
pub(super) struct Connector<'a> {
    sessions: Vec<Session<'a>>,
    /// `(model name, server index, server name, tool name)`
    tools: Vec<(String, usize, String, String)>,
    /// The tool definitions offered (chat `tools` form), and those held back for a tool search.
    pub offered: Vec<JsonValue>,
    pub deferred: Vec<JsonValue>,
    /// Notes for the reply (`gaudium.ignored`).
    pub notes: Vec<String>,
}

impl<'a> Connector<'a> {
    /// Opens every server and lists its tools. The error names the server that failed.
    pub fn open(proxy: &'a McpProxy, caller: Caller, specs: &[ServerSpec]) -> Result<Self, String> {
        let mut c = Connector {
            sessions: Vec::new(),
            tools: Vec::new(),
            offered: Vec::new(),
            deferred: Vec::new(),
            notes: Vec::new(),
        };
        for (i, spec) in specs.iter().enumerate() {
            let fail = |e: String| format!("mcp_servers.{i} ({}): {e}", spec.name);
            let mut session =
                Session::open(proxy, caller, &spec.url, spec.token.as_deref()).map_err(fail)?;
            let (tools, cut) = session.list_tools(BOUNDS.list_pages).map_err(fail)?;
            if cut {
                c.notes.push(format!(
                    "mcp_servers.{i} ({}): only the first {} pages of its tool list were read",
                    spec.name, BOUNDS.list_pages
                ));
            }
            for t in tools {
                if !spec.filter.offers(&t.name) {
                    continue;
                }
                let name = model_name(&spec.name, &t.name);
                if c.tools.iter().any(|x| x.0 == name) {
                    return Err(fail(format!(
                        "two tools of the request's MCP servers are both called '{name}' as the model sees them; rename a server"
                    )));
                }
                let mut f = vec![("name".to_string(), JsonValue::Str(name.clone()))];
                if let Some(d) = &t.description {
                    f.push(("description".to_string(), JsonValue::Str(d.clone())));
                }
                f.push(("parameters".to_string(), t.input_schema.clone()));
                let def = JsonValue::Object(vec![
                    ("type".to_string(), JsonValue::Str("function".into())),
                    ("function".to_string(), JsonValue::Object(f)),
                ]);
                if spec.filter.defers(&t.name) {
                    c.deferred.push(def);
                } else {
                    c.offered.push(def);
                }
                c.tools
                    .push((name, c.sessions.len(), spec.name.clone(), t.name));
            }
            c.sessions.push(session);
        }
        Ok(c)
    }

    /// Model name -> `(server name, tool name)`, for the reply's `mcp_tool_use` blocks.
    pub fn labels(&self) -> Vec<(String, String, String)> {
        self.tools
            .iter()
            .map(|(m, _, s, t)| (m.clone(), s.clone(), t.clone()))
            .collect()
    }

    /// Every model name of these tools.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|t| t.0.as_str())
    }
}

impl ServerTools for Connector<'_> {
    fn owns(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.0 == name)
    }

    fn run(&mut self, name: &str, arguments: &JsonValue) -> ToolOutput {
        let Some((_, idx, _, tool)) = self.tools.iter().find(|t| t.0 == name).cloned() else {
            return ToolOutput::error(format!("no MCP tool is called '{name}'"));
        };
        self.sessions[idx].call(&tool, arguments)
    }

    fn finish(&mut self) {
        for s in &mut self.sessions {
            s.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_and_names() {
        // today's tuning (the rules that use them are verified)
        assert_eq!(
            (
                BOUNDS.servers,
                BOUNDS.rounds,
                BOUNDS.calls,
                BOUNDS.result_bytes,
                BOUNDS.list_pages
            ),
            (8, 10, 64, 128 * 1024, 16)
        );
        assert_eq!(model_name("docs", "search"), "mcp__docs__search");
        assert_eq!(model_name("my server", "a.b/c"), "mcp__my_server__a_b_c");
    }

    #[test]
    fn filters_follow_tool_settings_over_the_default() {
        let f = ToolFilter {
            enabled: false,
            defer: true,
            configs: vec![
                ("a".into(), Some(true), None),
                ("b".into(), None, Some(false)),
            ],
        };
        assert!(f.offers("a") && f.defers("a"));
        assert!(!f.offers("b") && !f.defers("b"));
        assert!(!f.offers("c") && f.defers("c"));
        assert!(ToolFilter::default().offers("x") && !ToolFilter::default().defers("x"));
    }

    #[test]
    fn event_lines_and_results() {
        let mut ev = EventLines::default();
        assert!(ev.push(b"event: message\r\ndata: {\"a\"").is_empty());
        assert_eq!(
            ev.push(b":1}\r\n\r\n: ping\n\ndata: x\ndata: y\n\n"),
            ["{\"a\":1}".to_string(), "x\ny".to_string()]
        );
        let r = parse_json(
            r#"{"content":[{"type":"text","text":"héllo"},{"type":"image","data":"AA","mimeType":"image/png"}],"isError":true}"#,
        )
        .unwrap();
        let out = result_output(&r);
        assert!(out.is_error);
        assert_eq!(out.texts[0], "héllo");
        assert!(out.texts[1].contains("image/png"));
        // structured content alone becomes text
        let r = parse_json(r#"{"structuredContent":{"t":22}}"#).unwrap();
        assert_eq!(result_output(&r).texts, ["{\"t\":22}"]);
        // a result past the cap is cut at a character boundary, with a note
        let cut = ToolOutput {
            is_error: false,
            texts: vec!["ab".into(), "héllo".into()],
        }
        .within(4);
        assert_eq!(cut.texts[0], "ab\nh");
        assert!(cut.texts[1].contains("4 of its 9 bytes"));
        // the separators the prompt puts between parts count: many empty parts are cut too
        let empties = ToolOutput {
            is_error: false,
            texts: vec![String::new(); 300],
        };
        assert_eq!(empties.prompt_text().len(), 299);
        let cut = empties.within(100);
        assert_eq!(cut.texts.len(), 2);
        assert_eq!(cut.texts[0], "\n".repeat(100));
        assert!(cut.texts[1].contains("100 of its 299 bytes"));
        // the cut text plus its note is all the prompt gets
        assert!(cut.prompt_text().len() <= 100 + 1 + cut.texts[1].len());
        // parts that fit with their separators stay as they are
        let fits = ToolOutput {
            is_error: false,
            texts: vec!["ab".into(), "cd".into()],
        };
        assert_eq!(fits.clone().within(5), fits);
        assert_eq!(fits.within(4).texts[0], "ab\nc");
        let whole = ToolOutput {
            is_error: false,
            texts: vec!["ab".into()],
        }
        .within(2);
        assert_eq!(whole.texts, ["ab"]);
    }

    #[test]
    fn the_token_is_not_printed() {
        let s = ServerSpec {
            name: "n".into(),
            url: "https://example.com/mcp".into(),
            token: Some("SECRET-TOKEN".into()),
            filter: ToolFilter::default(),
        };
        assert!(!format!("{s:?}").contains("SECRET"));
    }
}
