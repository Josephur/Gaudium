//! The Messages MCP connector over real sockets: a request goes through the server's HTTP
//! handler, the tool loop and the scripted engine, and its MCP calls go through the MCP proxy's
//! rules to a throwaway MCP server on a loopback port (Streamable HTTP, JSON and event-stream
//! answers, a session id). Nothing leaves the machine.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use super::*;

const TOKEN: &str = "tok-SECRET-123";

/// One request the MCP server received: method, headers (lower-case names) and body.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|h| h.0 == name)
            .map(|h| h.1.as_str())
    }
    fn rpc(&self) -> String {
        json::parse_json(&self.body)
            .ok()
            .and_then(|v| {
                v.get("method")
                    .and_then(JsonValue::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_default()
    }
}

fn read_request(s: &mut TcpStream) -> Option<Seen> {
    let mut buf = Vec::new();
    let end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let mut b = [0u8; 4096];
        match s.read(&mut b) {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&b[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let mut lines = head.lines();
    let method = lines.next()?.split(' ').next()?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let len: usize = headers
        .iter()
        .find(|h| h.0 == "content-length")
        .and_then(|h| h.1.parse().ok())
        .unwrap_or(0);
    let mut body = buf[end..].to_vec();
    while body.len() < len {
        let mut b = [0u8; 4096];
        match s.read(&mut b) {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&b[..n]),
        }
    }
    Some(Seen {
        method,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

/// A small MCP server: `initialize` (JSON, assigns a session), `tools/list` (JSON, two pages),
/// `tools/call` (an event stream with a notification before the answer).
struct McpServer {
    port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl McpServer {
    fn start() -> McpServer {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        thread::spawn(move || {
            for tcp in l.incoming() {
                let Ok(mut s) = tcp else { break };
                let Some(req) = read_request(&mut s) else {
                    continue;
                };
                log.lock().unwrap().push(req.clone());
                answer(&req, &mut s);
            }
        });
        McpServer { port, seen }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

fn answer(req: &Seen, s: &mut TcpStream) {
    let json_reply = |s: &mut TcpStream, extra: &str, body: &str| {
        let _ = write!(
            s,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    };
    if req.method == "DELETE" {
        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
        return;
    }
    let v = json::parse_json(&req.body).unwrap();
    let id = v.get("id").map(|i| i.dump()).unwrap_or_default();
    match req.rpc().as_str() {
        "initialize" => json_reply(
            s,
            "Mcp-Session-Id: sess-1\r\n",
            &format!(
                r#"{{"jsonrpc":"2.0","id":{id},"result":{{"protocolVersion":"2025-06-18","capabilities":{{"tools":{{}}}},"serverInfo":{{"name":"clock","version":"1"}}}}}}"#
            ),
        ),
        "notifications/initialized" => {
            let _ = write!(s, "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n");
        }
        "tools/list" => {
            let second = v.get("params").and_then(|p| p.get("cursor")).is_some();
            let body = if second {
                format!(
                    r#"{{"jsonrpc":"2.0","id":{id},"result":{{"tools":[{{"name":"hidden","description":"Not offered.","inputSchema":{{"type":"object","properties":{{}}}}}}]}}}}"#
                )
            } else {
                format!(
                    r#"{{"jsonrpc":"2.0","id":{id},"result":{{"tools":[{{"name":"get_time","description":"The time.","inputSchema":{{"type":"object","properties":{{"zone":{{"type":"string"}}}}}}}}],"nextCursor":"p2"}}}}"#
                )
            };
            json_reply(s, "", &body);
        }
        "tools/call" => {
            let zone = v
                .get("params")
                .and_then(|p| p.get("arguments"))
                .and_then(|a| a.get("zone"))
                .and_then(JsonValue::as_str)
                .unwrap_or("?")
                .to_string();
            // a slow tool: the answer comes after a while
            if zone == "SLOW" {
                thread::sleep(std::time::Duration::from_millis(200));
            }
            let body = format!(
                "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{{}}}}\n\n\
                 event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"12:00 in {zone}\"}}],\"isError\":false}}}}\n\n"
            );
            // the stream stays open after the answer: the client stops reading at the answer
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{body}\r\n",
                body.len()
            );
            let _ = s.flush();
            thread::sleep(std::time::Duration::from_millis(300));
            let _ = write!(s, "0\r\n\r\n");
        }
        _ => {
            let _ = write!(s, "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
        }
    }
}

/// The harness with an MCP proxy that may reach this machine's loopback address.
fn harness_with_proxy(allow_private: &str) -> Harness {
    let mut h = harness();
    let opts = ProxyOptions {
        enabled: true,
        allow_private: vec![allow_private.to_string()],
        ..ProxyOptions::default()
    };
    Arc::get_mut(&mut h.state)
        .expect("the harness's only handle")
        .proxy = proxy::McpProxy::new(&opts, false).unwrap();
    h
}

fn servers(url: &str) -> String {
    format!(
        r#""mcp_servers":[{{"type":"url","url":"{url}","name":"clock","authorization_token":"{TOKEN}"}}]"#
    )
}

const ROUND_CALL: &str = "Need the time.\n</think>\n\nChecking.\n\n<tool_call>\n<function=mcp__clock__get_time>\n<parameter=zone>\nUTC\n</parameter>\n</function>\n</tool_call>";
const ROUND_END: &str = "Got it.\n</think>\n\nIt is 12:00 in UTC.";

fn kinds(blocks: &[JsonValue]) -> Vec<String> {
    blocks.iter().map(|b| s(b, &["type"]).to_string()).collect()
}

#[test]
fn mcp_servers_run_server_side_and_the_reply_goes_on() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    h.script(ROUND_CALL);
    h.script(ROUND_END);
    let before = h.prompts.lock().unwrap().len();
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"model":"any","max_tokens":256,{},"tools":[{{"type":"mcp_toolset","mcp_server_name":"clock","configs":{{"hidden":{{"enabled":false}}}}}}],"messages":[{{"role":"user","content":"What time is it?"}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 200, "{text}");
    assert!(!text.contains(TOKEN), "{text}");
    let r = json::parse_json(&text).unwrap();
    let blocks = at(&r, &["content"]).as_array().unwrap();
    assert_eq!(
        kinds(blocks),
        [
            "thinking",
            "text",
            "mcp_tool_use",
            "mcp_tool_result",
            "thinking",
            "text"
        ]
    );
    let use_ = &blocks[2];
    assert_eq!(
        (s(use_, &["name"]), s(use_, &["server_name"])),
        ("get_time", "clock")
    );
    assert!(s(use_, &["id"]).starts_with("mcptoolu_"));
    assert_eq!(at(use_, &["input"]).dump(), r#"{"zone":"UTC"}"#);
    let res = &blocks[3];
    assert_eq!(s(res, &["tool_use_id"]), s(use_, &["id"]));
    assert_eq!(at(res, &["is_error"]).as_bool(), Some(false));
    assert_eq!(s(res, &["content", "0", "text"]), "12:00 in UTC");
    assert_eq!(s(&blocks[5], &["text"]), "It is 12:00 in UTC.");
    assert_eq!(s(&r, &["stop_reason"]), "end_turn");
    assert_eq!(at(&r, &["gaudium_tool_loop", "rounds"]).as_u64(), Some(2));

    // two rounds on the card; the second saw the tools and the result
    let prompts = h.prompts.lock().unwrap()[before..].to_vec();
    assert_eq!(prompts.len(), 2);
    assert!(
        prompts[0].contains("mcp__clock__get_time"),
        "{}",
        prompts[0]
    );
    assert!(!prompts[0].contains("mcp__clock__hidden"), "{}", prompts[0]);
    assert!(
        prompts[1].contains("<tool_response>\n12:00 in UTC\n</tool_response>"),
        "{}",
        prompts[1]
    );

    // the server saw the handshake, the session, the protocol version and the bearer token
    let seen = mcp.seen();
    let rpcs: Vec<String> = seen.iter().map(Seen::rpc).collect();
    assert_eq!(
        &rpcs[..5],
        [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/list",
            "tools/call"
        ]
    );
    for x in &seen {
        assert_eq!(x.header("authorization"), Some(&*format!("Bearer {TOKEN}")));
        assert!(x
            .header("accept")
            .is_some_and(|a| a.contains("text/event-stream")));
    }
    for x in &seen[1..] {
        assert_eq!(x.header("mcp-session-id"), Some("sess-1"));
        assert_eq!(x.header("mcp-protocol-version"), Some("2025-06-18"));
    }
    // the session is ended
    assert_eq!(seen.last().unwrap().method, "DELETE");

    // usage sums both rounds
    let out = at(&r, &["usage", "output_tokens"]).as_u64().unwrap();
    assert!(out > h.state.models[0].tokenizer.encode(ROUND_END, false).len() as u64);
}

#[test]
fn mcp_blocks_stream_in_order() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    h.script(ROUND_CALL);
    h.script(ROUND_END);
    let (status, body) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"stream":true,"max_tokens":256,{},"messages":[{{"role":"user","content":"Time?"}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 200);
    assert!(!body.contains(TOKEN));
    let events: Vec<(String, JsonValue)> = body
        .split("\n\n")
        .filter(|e| !e.is_empty())
        .map(|e| {
            let (name, data) = e.split_once('\n').unwrap();
            (
                name.strip_prefix("event: ").unwrap().to_string(),
                json::parse_json(data.strip_prefix("data: ").unwrap()).unwrap(),
            )
        })
        .collect();
    let count = |n: &str| events.iter().filter(|(e, _)| e == n).count();
    assert_eq!(
        (
            count("message_start"),
            count("message_delta"),
            count("message_stop")
        ),
        (1, 1, 1)
    );
    assert_eq!(events[0].0, "message_start");
    assert_eq!(events.last().unwrap().0, "message_stop");
    let starts: Vec<(u64, String)> = events
        .iter()
        .filter(|(e, _)| e == "content_block_start")
        .map(|(_, d)| {
            (
                at(d, &["index"]).as_u64().unwrap(),
                s(d, &["content_block", "type"]).to_string(),
            )
        })
        .collect();
    assert_eq!(
        starts,
        [
            (0, "thinking".to_string()),
            (1, "text".to_string()),
            (2, "mcp_tool_use".to_string()),
            (3, "mcp_tool_result".to_string()),
            (4, "thinking".to_string()),
            (5, "text".to_string()),
        ]
    );
    // every block stops once, in order
    let stops: Vec<u64> = events
        .iter()
        .filter(|(e, _)| e == "content_block_stop")
        .map(|(_, d)| at(d, &["index"]).as_u64().unwrap())
        .collect();
    assert_eq!(stops, [0, 1, 2, 3, 4, 5]);
    // the call's input arrives as its deltas; the result whole in its start event
    let input: String = events
        .iter()
        .filter(|(e, d)| e == "content_block_delta" && at(d, &["index"]).as_u64() == Some(2))
        .map(|(_, d)| s(d, &["delta", "partial_json"]).to_string())
        .collect();
    assert_eq!(
        json::parse_json(&input).unwrap().dump(),
        r#"{"zone":"UTC"}"#
    );
    let result = events
        .iter()
        .find(|(e, d)| e == "content_block_start" && at(d, &["index"]).as_u64() == Some(3))
        .unwrap();
    assert_eq!(
        s(&result.1, &["content_block", "content", "0", "text"]),
        "12:00 in UTC"
    );
    let (_, md) = &events[events.len() - 2];
    assert_eq!(s(md, &["delta", "stop_reason"]), "end_turn");
}

#[test]
fn the_round_bound_pauses_the_turn() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    let rounds = mcp_client::BOUNDS.rounds as usize;
    for _ in 0..rounds + 1 {
        h.script(ROUND_CALL);
    }
    let before = h.prompts.lock().unwrap().len();
    // the output limit holds over every round: it leaves room for all of them here
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":4096,{},"messages":[{{"role":"user","content":"Time?"}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(s(&r, &["stop_reason"]), "pause_turn");
    assert_eq!(h.prompts.lock().unwrap().len() - before, rounds);
    let blocks = at(&r, &["content"]).as_array().unwrap();
    let k = kinds(blocks);
    // every call of the last round has its result too
    assert_eq!(k.iter().filter(|x| *x == "mcp_tool_use").count(), rounds);
    assert_eq!(k.iter().filter(|x| *x == "mcp_tool_result").count(), rounds);
    assert_eq!(k.last().map(String::as_str), Some("mcp_tool_result"));
}

#[test]
fn a_client_call_stops_the_loop_after_the_server_calls_ran() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    h.script("Both.\n</think>\n\n<tool_call>\n<function=mcp__clock__get_time>\n<parameter=zone>\nCET\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>");
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":256,{},{},"messages":[{{"role":"user","content":"Time and weather?"}}]}}"#,
            servers(&mcp.url()),
            super::MESSAGES_TOOLS
        ),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(s(&r, &["stop_reason"]), "tool_use");
    let blocks = at(&r, &["content"]).as_array().unwrap();
    assert_eq!(
        kinds(blocks),
        ["thinking", "mcp_tool_use", "tool_use", "mcp_tool_result"]
    );
    assert!(s(&blocks[2], &["id"]).starts_with("toolu_"));
    assert_eq!(s(&blocks[3], &["content", "0", "text"]), "12:00 in CET");

    // the client sends the turn back with its own result: the history renders both calls
    h.script("</think>\n\nNoon, and sunny.");
    let assistant = at(&r, &["content"]).dump();
    let tool_id = s(&blocks[2], &["id"]).to_string();
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":256,{},{},"messages":[{{"role":"user","content":"Time and weather?"}},
                {{"role":"assistant","content":{assistant}}},
                {{"role":"user","content":[{{"type":"tool_result","tool_use_id":"{tool_id}","content":"sunny"}}]}}]}}"#,
            servers(&mcp.url()),
            super::MESSAGES_TOOLS
        ),
    );
    assert_eq!(status, 200, "{text}");
    let p = h.last_prompt();
    assert!(p.contains("<function=mcp__clock__get_time>"), "{p}");
    assert!(p.contains("<function=get_weather>"), "{p}");
    assert!(
        p.contains("<tool_response>\n12:00 in CET\n</tool_response>"),
        "{p}"
    );
    assert!(
        p.contains("<tool_response>\nsunny\n</tool_response>"),
        "{p}"
    );
}

#[test]
fn mcp_servers_are_refused_without_the_proxy_or_outside_its_list() {
    let mcp = McpServer::start();
    let body = format!(
        r#"{{"max_tokens":64,{},"messages":[{{"role":"user","content":"Time?"}}]}}"#,
        servers(&mcp.url())
    );
    // the server was started without the proxy
    let h = harness();
    let (status, text) = h.post("/v1/messages", &body);
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("--mcp-proxy"), "{text}");
    // the proxy is on, but its list does not hold the server's address
    let h = harness_with_proxy("192.0.2.7");
    let (status, text) = h.post("/v1/messages", &body);
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("mcp_servers.0 (clock)"), "{text}");
    assert!(text.contains("not in the allow-list"), "{text}");
    assert!(!text.contains(TOKEN), "{text}");
    assert!(mcp.seen().is_empty(), "the refused server was contacted");
    // too many servers
    let h = harness_with_proxy("127.0.0.1");
    let many: Vec<String> = (0..=mcp_client::BOUNDS.servers)
        .map(|i| format!(r#"{{"type":"url","url":"{}","name":"s{i}"}}"#, mcp.url()))
        .collect();
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":64,"mcp_servers":[{}],"messages":[{{"role":"user","content":"x"}}]}}"#,
            many.join(",")
        ),
    );
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("at most"), "{text}");
    assert!(mcp.seen().is_empty());
    // a client tool of the same name as a server's tool
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":64,{},"tools":[{{"name":"mcp__clock__get_time"}}],"messages":[{{"role":"user","content":"x"}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("client tool"), "{text}");
}

#[test]
fn token_counts_include_the_servers_tools() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    let plain = r#"{"messages":[{"role":"user","content":"Time?"}]}"#;
    let (_, a) = h.post("/v1/messages/count_tokens", plain);
    let (status, b) = h.post(
        "/v1/messages/count_tokens",
        &format!(
            r#"{{{},"messages":[{{"role":"user","content":"Time?"}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 200, "{b}");
    let n = |t: &str| {
        json::parse_json(t)
            .unwrap()
            .get("input_tokens")
            .and_then(JsonValue::as_u64)
            .unwrap()
    };
    assert!(n(&b) > n(&a), "{a} {b}");
}

/// The headers a page of another site sends, as a browser sets them (page script cannot).
const FOREIGN: [&str; 2] = [
    "Origin: https://other.example",
    "Sec-Fetch-Site: cross-site",
];

#[test]
fn a_page_of_another_site_cannot_drive_the_mcp_servers() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    let ask = format!(
        r#"{{"model":"any","max_tokens":64,{},"messages":[{{"role":"user","content":"Time?"}}]}}"#,
        servers(&mcp.url())
    );
    // the route refuses such a page, and so does each Messages route that would reach a server
    let (status, text) = h.exchange_with(
        "GET",
        &format!("/cors-proxy?url={}", mcp.url()),
        &FOREIGN,
        "",
    );
    assert_eq!(status, 403, "{text}");
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let (status, text) = h.exchange_with("POST", path, &FOREIGN, &ask);
        assert_eq!(status, 403, "{path}: {text}");
        assert!(text.contains("permission_error"), "{path}: {text}");
        assert!(text.contains("another site"), "{path}: {text}");
        assert!(!text.contains(TOKEN), "{path}: {text}");
        // an origin that is not the request's own host, without the browser's statement
        let (status, text) =
            h.exchange_with("POST", path, &["Origin: https://other.example"], &ask);
        assert_eq!(status, 403, "{path}: {text}");
    }
    assert!(
        mcp.seen().is_empty(),
        "a page of another site reached the MCP server: {:?}",
        mcp.seen()
    );
    // the server's own page still runs them, and a request without servers from anywhere runs
    // as before
    let own = ["Origin: http://test", "Sec-Fetch-Site: same-origin"];
    let (status, text) = h.exchange_with("POST", "/v1/messages/count_tokens", &own, &ask);
    assert_eq!(status, 200, "{text}");
    assert!(mcp.seen().iter().any(|x| x.rpc() == "tools/list"));
    let (status, text) = h.exchange_with(
        "POST",
        "/v1/messages/count_tokens",
        &FOREIGN,
        r#"{"messages":[{"role":"user","content":"Time?"}]}"#,
    );
    assert_eq!(status, 200, "{text}");
}

#[test]
fn the_servers_own_page_runs_the_mcp_servers() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    h.script(ROUND_CALL);
    h.script(ROUND_END);
    let (status, text) = h.exchange_with(
        "POST",
        "/v1/messages",
        &["Origin: http://test"],
        &format!(
            r#"{{"model":"any","max_tokens":256,{},"messages":[{{"role":"user","content":"What time is it?"}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    let blocks = at(&r, &["content"]).as_array().unwrap();
    assert!(
        kinds(blocks).contains(&"mcp_tool_result".to_string()),
        "{text}"
    );
    assert!(mcp.seen().iter().any(|x| x.rpc() == "tools/call"));
}

/// `max_tokens` with the tokens of the first round, plus `extra`.
fn first_round_plus(h: &Harness, extra: u64) -> u64 {
    h.state.models[0].tokenizer.encode(ROUND_CALL, true).len() as u64 + extra
}

#[test]
fn the_output_limit_holds_over_every_round() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    // the first round fits, and leaves 3 tokens for the second
    let max = first_round_plus(&h, 3);
    assert!(h.state.models[0].tokenizer.encode(ROUND_END, true).len() > 3);
    h.script(ROUND_CALL);
    h.script(ROUND_END);
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":{max},{},"messages":[{{"role":"user","content":"Time?"}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(at(&r, &["gaudium_tool_loop", "rounds"]).as_u64(), Some(2));
    // the second round had only what was left: the reply is cut there
    assert_eq!(s(&r, &["stop_reason"]), "max_tokens", "{text}");
    let out = at(&r, &["usage", "output_tokens"]).as_u64().unwrap();
    assert!(out <= max, "{out} output tokens over a limit of {max}");

    // a first round that spends the whole limit: its calls run, and no round follows
    let h = harness_with_proxy("127.0.0.1");
    let max = first_round_plus(&h, 0);
    h.script(ROUND_CALL);
    h.script(ROUND_END);
    let before = h.prompts.lock().unwrap().len();
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":{max},{},"messages":[{{"role":"user","content":"Time?"}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(h.prompts.lock().unwrap().len() - before, 1);
    assert_eq!(s(&r, &["stop_reason"]), "max_tokens", "{text}");
    let blocks = at(&r, &["content"]).as_array().unwrap();
    assert_eq!(kinds(blocks).last().map(String::as_str), Some("mcp_tool_result"));
    assert!(at(&r, &["usage", "output_tokens"]).as_u64().unwrap() <= max);
}

#[test]
fn no_call_is_sent_once_the_client_is_gone() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    let call = |zone: &str| {
        format!("<tool_call>\n<function=mcp__clock__get_time>\n<parameter=zone>\n{zone}\n</parameter>\n</function>\n</tool_call>")
    };
    h.script(&format!(
        "Three.\n</think>\n\n{}\n{}\n{}",
        call("SLOW"),
        call("UTC"),
        call("CET")
    ));
    h.script(ROUND_END);
    let body = format!(
        r#"{{"stream":true,"max_tokens":4096,{},"messages":[{{"role":"user","content":"Time?"}}]}}"#,
        servers(&mcp.url())
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::clone(&h.state);
    let server = thread::spawn(move || {
        let (s, _) = listener.accept().unwrap();
        handle_connection(state, s);
    });
    let mut c = TcpStream::connect(addr).unwrap();
    c.set_read_timeout(Some(std::time::Duration::from_secs(20)))
        .unwrap();
    write!(
        c,
        "POST /v1/messages HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    // the client reads the round up to its third call, and goes away while the first (slow) call
    // runs
    let mut seen = Vec::new();
    while String::from_utf8_lossy(&seen)
        .matches("\"type\":\"mcp_tool_use\"")
        .count()
        < 3
    {
        let mut b = [0u8; 4096];
        match c.read(&mut b) {
            Ok(0) | Err(_) => break,
            Ok(n) => seen.extend_from_slice(&b[..n]),
        }
    }
    thread::sleep(std::time::Duration::from_millis(50));
    drop(c);
    server.join().unwrap();
    let calls = mcp.seen().iter().filter(|x| x.rpc() == "tools/call").count();
    // the result that found the client gone is the last: the calls after it are never sent
    // (the write that finds the connection closed is the first or the second one after it)
    assert!(calls < 3, "{calls} calls sent after the client went away");
}

#[test]
fn a_tool_reference_finds_a_deferred_mcp_tool() {
    let mcp = McpServer::start();
    let h = harness_with_proxy("127.0.0.1");
    h.script("</think>\n\nOk.");
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":64,{},"tools":[{{"name":"search","input_schema":{{"type":"object"}}}},
                {{"type":"mcp_toolset","mcp_server_name":"clock","configs":{{"get_time":{{"defer_loading":true}}}}}}],
              "messages":[{{"role":"user","content":"Find a clock tool"}},
                {{"role":"assistant","content":[{{"type":"tool_use","id":"toolu_1","name":"search","input":{{}}}}]}},
                {{"role":"user","content":[{{"type":"tool_result","tool_use_id":"toolu_1","content":[{{"type":"tool_reference","tool_name":"mcp__clock__get_time"}}]}}]}}]}}"#,
            servers(&mcp.url())
        ),
    );
    assert_eq!(status, 200, "{text}");
    let p = h.last_prompt();
    assert!(
        p.contains("Tool mcp__clock__get_time is now available:"),
        "{p}"
    );
    assert!(p.contains("The time."), "{p}");
    assert!(!p.contains("is not available"), "{p}");
}
