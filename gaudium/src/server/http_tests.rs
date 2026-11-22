//! The chat endpoints over real sockets, with a scripted engine in place of the card: a request
//! goes through the HTTP parser, the model's chat template, the turn queue, the output parser
//! and the reply writer exactly as in service; only the token source is canned.

use std::collections::VecDeque;
use std::io::{Read, Write};

use super::*;
use crate::engine::{GenParams, GenReport};

const TOOLS: &str = r#""tools":[{"type":"function","function":{"name":"get_weather","description":"Weather.","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}},{"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{}}}}]"#;

/// Emits the next canned reply token by token and records the prompt it was given.
struct ScriptedEngine {
    spec: ModelSpec,
    tok: Arc<Tokenizer>,
    replies: Arc<Mutex<VecDeque<String>>>,
    prompts: Arc<Mutex<Vec<String>>>,
}

impl GenEngine for ScriptedEngine {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn max_ctx(&self) -> u64 {
        16384
    }

    fn hbm_bytes(&self) -> [u64; 5] {
        [0; 5]
    }

    fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        cancel: Option<&AtomicBool>,
        on_token: &mut dyn FnMut(i32),
    ) -> Result<GenReport, String> {
        self.prompts
            .lock()
            .unwrap()
            .push(self.tok.decode(prompt_ids, false));
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .ok_or("no scripted reply")?;
        let ids = self.tok.encode(&reply, true);
        let mut rep = GenReport {
            prompt_tokens: prompt_ids.len(),
            method: "scripted".into(),
            ..GenReport::default()
        };
        for id in ids {
            // as an engine: cancellation is seen between launches
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                rep.stopped_by_cancel = true;
                return Ok(rep);
            }
            if rep.generated_tokens.len() as u64 >= params.max_new {
                return Ok(rep);
            }
            // as an engine: an end-of-sequence id of the request ends the generation
            if params.extra_eos_ids.contains(&id) {
                rep.generated_tokens.push(id);
                break;
            }
            rep.generated_tokens.push(id);
            on_token(id);
        }
        rep.stopped_by_eos = true;
        Ok(rep)
    }
}

struct Harness {
    state: Arc<AppState>,
    replies: Arc<Mutex<VecDeque<String>>>,
    prompts: Arc<Mutex<Vec<String>>>,
}

/// A second model family for the tests: reasoning and tool calls between special tokens, the
/// call form of a `response_template`.
const CALL_FORM_TEMPLATE: &str = "{{- bos_token }}{% if tools %}<|turn>system\n{% for t in tools %}<|tool>{{ t.function.name }}<tool|>{% endfor %}<turn|>\n{% endif %}\
{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}<|turn>model\n";
const CALL_FORM_CONFIG: &str = r#"{"bos_token": "<bos>", "response_template": {"fields": {
    "thinking": {"open": "<|channel>thought\n", "close": "<channel|>"},
    "tool_calls": {"open_pattern": "<\\|tool_call>call:(?P<name>\\w+)", "close": "<tool_call|>", "content": "json",
                   "content_args": {"string_delims": [["<|\"|>", "<|\"|>"]]}}}}}"#;

fn harness() -> Harness {
    let spec = ModelSpec::from_config_json(
        r#"{"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text",
            "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320,
            "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
            "head_dim": 256, "full_attention_interval": 4,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_conv_kernel_dim": 4}}"#,
    )
    .unwrap();
    let tok = Arc::new(crate::tokenizer::test_tokenizer(&[
        ("<|im_start|>", true),
        ("<|im_end|>", true),
        ("<bos>", true),
        ("<|turn>", true),
        ("<turn|>", true),
        ("<|channel>", true),
        ("<channel|>", true),
        ("<|tool_call>", true),
        ("<tool_call|>", true),
        ("<|\"|>", true),
        ("<|tool_response>", true),
        ("<|tool>", true),
        ("<tool|>", true),
    ]));
    let model = |id: &str, template: &str, config: Option<&str>| {
        let config = config.map(|c| json::parse_json(c).unwrap());
        let (chat, note) = ChatTemplate::from_parts(template, config.as_ref()).unwrap();
        assert!(note.is_none(), "{note:?}");
        RegisteredModel {
            id: id.into(),
            dir: PathBuf::from("/nonexistent"),
            spec: spec.clone(),
            format: chat.format.clone(),
            chat: Some(chat),
            tokenizer: Arc::clone(&tok),
            extra_eos_ids: Vec::new(),
            max_ctx: 16384,
            gemma_closes_thought: false,
            gen_defaults: SamplerDefaults::FALLBACK,
        }
    };
    let replies = Arc::new(Mutex::new(VecDeque::new()));
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let engine = ScriptedEngine {
        spec: spec.clone(),
        tok: Arc::clone(&tok),
        replies: Arc::clone(&replies),
        prompts: Arc::clone(&prompts),
    };
    let state = Arc::new(AppState {
        cfg: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            models: Vec::new(),
            kernels_dir: PathBuf::new(),
            ui_dir: None,
            max_ctx: 16384,
            cache_dir: None,
            no_cache: true,
            lazy: true,
            idle_park_secs: 0,
            queue_config: None,
            api_key: None,
            a2ui: true,
            verbose: false,
            per_model: Vec::new(),
            draft_k: Vec::new(),
        },
        models: vec![model("tags", chat::TEST_TEMPLATE, None)],
        slot: Mutex::new(EngineSlot {
            engine: Some(Box::new(engine)),
            resident_idx: Some(0),
            lifecycle: Lifecycle::Ready(0),
            detail: String::new(),
            since: Instant::now(),
            last_activity: Instant::now(),
            admin_parked: false,
            hold: false,
            active_requests: 0,
            waiting: vec![0],
            switches: 0,
            last_switch_s: None,
            last_warmup_s: None,
            last_live_json: None,
            last_request_json: None,
            hbm_bytes: None,
            facts: None,
        }),
        card_op: Mutex::new(()),
        cancels: Mutex::new(HashMap::new()),
        next_req_id: AtomicU64::new(1),
        hub: EventHub::new(),
        dtypes: Mutex::new(HashMap::new()),
        turns: TurnQueue::default(),
    });
    let _ = model; // (one resident model per harness)
    Harness {
        state,
        replies,
        prompts,
    }
}

/// The same harness with the call-form family as its model.
fn call_form_harness() -> Harness {
    let h = harness();
    let m = &h.state.models[0];
    let config = json::parse_json(CALL_FORM_CONFIG).unwrap();
    let (chat, note) = ChatTemplate::from_parts(CALL_FORM_TEMPLATE, Some(&config)).unwrap();
    assert!(note.is_none(), "{note:?}");
    let model = RegisteredModel {
        id: "callform".into(),
        dir: m.dir.clone(),
        spec: m.spec.clone(),
        format: chat.format.clone(),
        chat: Some(chat),
        tokenizer: Arc::clone(&m.tokenizer),
        extra_eos_ids: vec![m.tokenizer.token_id("<|tool_response>").unwrap()],
        max_ctx: m.max_ctx,
        gemma_closes_thought: false,
        gen_defaults: m.gen_defaults,
    };
    let engine = h.state.slot.lock().unwrap().engine.take();
    let state = Arc::new(AppState {
        cfg: h.state.cfg.clone(),
        models: vec![model],
        slot: Mutex::new(EngineSlot {
            engine,
            resident_idx: Some(0),
            lifecycle: Lifecycle::Ready(0),
            detail: String::new(),
            since: Instant::now(),
            last_activity: Instant::now(),
            admin_parked: false,
            hold: false,
            active_requests: 0,
            waiting: vec![0],
            switches: 0,
            last_switch_s: None,
            last_warmup_s: None,
            last_live_json: None,
            last_request_json: None,
            hbm_bytes: None,
            facts: None,
        }),
        card_op: Mutex::new(()),
        cancels: Mutex::new(HashMap::new()),
        next_req_id: AtomicU64::new(1),
        hub: EventHub::new(),
        dtypes: Mutex::new(HashMap::new()),
        turns: TurnQueue::default(),
    });
    Harness {
        state,
        replies: h.replies,
        prompts: h.prompts,
    }
}

impl Harness {
    fn script(&self, reply: &str) {
        self.replies.lock().unwrap().push_back(reply.to_string());
    }

    fn last_prompt(&self) -> String {
        self.prompts
            .lock()
            .unwrap()
            .last()
            .cloned()
            .unwrap_or_default()
    }

    /// One HTTP exchange with the server's connection handler: `(status, body)`, the body with
    /// its transfer chunks joined.
    fn post(&self, path: &str, body: &str) -> (u16, String) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::clone(&self.state);
        let server = thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            handle_connection(state, s);
        });
        let mut c = TcpStream::connect(addr).unwrap();
        write!(
            c,
            "POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut raw = Vec::new();
        c.read_to_end(&mut raw).unwrap();
        server.join().unwrap();
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("a response head");
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut rest = &raw[split + 4..];
        if !head.contains("Transfer-Encoding: chunked") {
            return (status, String::from_utf8_lossy(rest).into_owned());
        }
        let mut out = Vec::new();
        loop {
            let eol = rest
                .windows(2)
                .position(|w| w == b"\r\n")
                .expect("a chunk size line");
            let n = usize::from_str_radix(std::str::from_utf8(&rest[..eol]).unwrap().trim(), 16)
                .unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&rest[eol + 2..eol + 2 + n]);
            rest = &rest[eol + 2 + n + 2..];
        }
        (status, String::from_utf8(out).unwrap())
    }

    fn chat(&self, extra: &str) -> (u16, JsonValue) {
        let body = format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"Weather in Oslo?\"}}]{extra}}}"
        );
        let (status, text) = self.post("/v1/chat/completions", &body);
        (
            status,
            json::parse_json(&text).unwrap_or_else(|e| panic!("{e}: {text}")),
        )
    }
}

fn s<'a>(v: &'a JsonValue, path: &[&str]) -> &'a str {
    at(v, path)
        .as_str()
        .unwrap_or_else(|| panic!("{path:?} is not a string in {}", v.dump()))
}

fn at<'a>(v: &'a JsonValue, path: &[&str]) -> &'a JsonValue {
    path.iter().fold(v, |cur, key| match key.parse::<usize>() {
        Ok(i) => &cur
            .as_array()
            .unwrap_or_else(|| panic!("{key} of {}", cur.dump()))[i],
        Err(_) => cur
            .get(key)
            .unwrap_or_else(|| panic!("no '{key}' in {}", cur.dump())),
    })
}

/// A chat-completion stream put back together.
#[derive(Default)]
struct ChatStream {
    reasoning: String,
    content: String,
    /// `(id, name, arguments)` per call index.
    calls: Vec<(String, String, String)>,
    finish: String,
    done: bool,
}

fn collect_chat_stream(body: &str) -> ChatStream {
    let mut st = ChatStream::default();
    for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            st.done = true;
            continue;
        }
        let chunk = json::parse_json(line).unwrap();
        let choice = at(&chunk, &["choices", "0"]);
        if let Some(f) = choice.get("finish_reason").and_then(JsonValue::as_str) {
            st.finish = f.to_string();
        }
        let delta = at(choice, &["delta"]);
        let text = |key: &str| delta.get(key).and_then(JsonValue::as_str).unwrap_or("");
        st.reasoning.push_str(text("reasoning_content"));
        st.content.push_str(text("content"));
        for tc in delta
            .get("tool_calls")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
        {
            let i = tc.get("index").and_then(JsonValue::as_u64).unwrap() as usize;
            if let Some(id) = tc.get("id").and_then(JsonValue::as_str) {
                // a call's first fragment: the next index, with its type and name
                assert_eq!(i, st.calls.len(), "a header for a new index");
                assert_eq!(s(tc, &["type"]), "function");
                let name = s(tc, &["function", "name"]).to_string();
                st.calls.push((id.to_string(), name, String::new()));
            }
            assert!(i < st.calls.len(), "a fragment before its header");
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(JsonValue::as_str)
                .unwrap_or("");
            st.calls[i].2.push_str(args);
        }
    }
    st
}

const CALL_REPLY: &str = "I should look it up.\n</think>\n\nLet me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo \"S\"\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>";

#[test]
fn chat_completion_returns_tool_calls() {
    let h = harness();
    h.script(CALL_REPLY);
    let (status, r) = h.chat(&format!(",{TOOLS}"));
    assert_eq!(status, 200, "{}", r.dump());
    // the tools reached the model through its template
    assert!(h
        .last_prompt()
        .contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\""));
    assert!(h
        .last_prompt()
        .ends_with("<|im_start|>assistant\n<think>\n"));
    let msg = at(&r, &["choices", "0", "message"]);
    assert_eq!(s(msg, &["reasoning_content"]), "I should look it up.");
    assert_eq!(s(msg, &["content"]), "Let me check.");
    let calls = at(msg, &["tool_calls"]).as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(s(&calls[0], &["type"]), "function");
    assert!(s(&calls[0], &["id"]).starts_with("call_"));
    assert_eq!(s(&calls[0], &["function", "name"]), "get_weather");
    assert_eq!(
        s(&calls[0], &["function", "arguments"]),
        "{\"city\":\"Oslo \\\"S\\\"\",\"days\":2}"
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "tool_calls");
    assert_eq!(at(&r, &["gaudium", "tool_calls"]).as_u64(), Some(1));

    // a reply that is only a call has no content; without tools the same text is plain content
    h.script("</think>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>");
    let (_, r) = h.chat(&format!(",{TOOLS}"));
    assert_eq!(
        at(&r, &["choices", "0", "message", "content"]),
        &JsonValue::Null
    );
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "arguments"
            ]
        ),
        "{}"
    );
    h.script("</think>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>");
    let (_, r) = h.chat("");
    assert!(s(&r, &["choices", "0", "message", "content"]).starts_with("<tool_call>"));
    assert!(at(&r, &["choices", "0", "message"])
        .get("tool_calls")
        .is_none());
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "stop");
}

#[test]
fn chat_completion_streams_tool_call_fragments() {
    let h = harness();
    h.script(&format!(
        "{CALL_REPLY}\n<tool_call>\n<function=now>\n</function>\n</tool_call>"
    ));
    let (status, body) = h.post(
        "/v1/chat/completions",
        &format!("{{\"stream\":true,\"messages\":[{{\"role\":\"user\",\"content\":\"Weather?\"}}],{TOOLS}}}"),
    );
    assert_eq!(status, 200);
    let st = collect_chat_stream(&body);
    assert_eq!(st.reasoning, "I should look it up.");
    assert_eq!(st.content, "Let me check.");
    assert_eq!(st.calls.len(), 2, "{:?}", st.calls);
    assert_eq!(
        (st.calls[0].1.as_str(), st.calls[0].2.as_str()),
        ("get_weather", "{\"city\":\"Oslo \\\"S\\\"\",\"days\":2}")
    );
    assert_eq!(
        (st.calls[1].1.as_str(), st.calls[1].2.as_str()),
        ("now", "{}")
    );
    assert_ne!(st.calls[0].0, st.calls[1].0);
    assert_eq!(st.finish, "tool_calls");
    assert!(st.done);
    // a string argument streams before the call is complete: more than one fragment chunk
    let fragments = body.matches("\"tool_calls\":[").count();
    assert!(
        fragments > 3,
        "{fragments} chunks carried tool-call fragments"
    );
}

#[test]
fn tool_results_go_back_through_the_template() {
    let h = harness();
    h.script("</think>\n\nIt is sunny in Oslo.");
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":"Weather in Oslo?"}},
            {{"role":"assistant","content":"","tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"get_weather","arguments":"{{\"city\":\"Oslo\",\"days\":2}}"}}}}]}},
            {{"role":"tool","tool_call_id":"call_1","content":"sunny, 4 C"}}],{TOOLS}}}"#
    );
    let (status, text) = h.post("/v1/chat/completions", &body);
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(
        s(&r, &["choices", "0", "message", "content"]),
        "It is sunny in Oslo."
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "stop");
    assert!(h.last_prompt().contains(
        "<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\nsunny, 4 C\n</tool_response><|im_end|>\n"
    ));
}

#[test]
fn tool_choice_parallel_calls_and_stop_strings() {
    let h = harness();
    // a named choice: the call's opening ends the prompt, the model writes the rest
    h.script("<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>");
    let (_, r) = h.chat(&format!(",{TOOLS},\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}"));
    assert!(h
        .last_prompt()
        .ends_with("<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n"));
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "name"
            ]
        ),
        "get_weather"
    );
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "arguments"
            ]
        ),
        "{\"city\":\"Oslo\"}"
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "tool_calls");
    // required: the model names the function
    h.script("<function=now>\n</function>\n</tool_call>");
    let (_, r) = h.chat(&format!(",{TOOLS},\"tool_choice\":\"required\""));
    assert!(h.last_prompt().ends_with("</think>\n\n<tool_call>\n"));
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "name"
            ]
        ),
        "now"
    );
    // none: the tools are not shown
    h.script("</think>\n\nNo tools here.");
    let (_, r) = h.chat(&format!(",{TOOLS},\"tool_choice\":\"none\""));
    assert!(!h.last_prompt().contains("<tools>"));
    assert_eq!(
        s(&r, &["choices", "0", "message", "content"]),
        "No tools here."
    );
    // refused before the card is asked
    let before = h.prompts.lock().unwrap().len();
    let (status, r) = h.chat(",\"tool_choice\":\"required\"");
    assert_eq!(status, 400);
    assert!(s(&r, &["error", "message"]).contains("non-empty 'tools'"));
    assert_eq!(h.prompts.lock().unwrap().len(), before);

    // parallel calls off: the first call ends the generation
    h.script("</think>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call> and more text");
    let (_, r) = h.chat(&format!(",{TOOLS},\"parallel_tool_calls\":false"));
    assert_eq!(
        at(&r, &["choices", "0", "message", "tool_calls"])
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "tool_calls");
    assert!(
        (at(&r, &["usage", "completion_tokens"]).as_u64().unwrap() as usize) < 80,
        "the generation was cut"
    );

    // a stop string ends the reply and is not part of it
    h.script("</think>\n\none two STOP three");
    let (_, r) = h.chat(",\"stop\":[\"STOP\"]");
    assert_eq!(s(&r, &["choices", "0", "message", "content"]), "one two ");
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "stop");
    // the token limit
    h.script("</think>\n\nabcdefghijklmnopqrstuvwxyz");
    let (_, r) = h.chat(",\"max_tokens\":16");
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "length");
}

#[test]
fn special_token_tags_and_the_call_form() {
    let h = call_form_harness();
    h.script("<|channel>thought\nNeed the tool.<channel|>On it.<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,days:<|\"|>2<|\"|>}<tool_call|><|tool_response>never shown");
    let (status, r) = h.chat(&format!(",{TOOLS}"));
    assert_eq!(status, 200, "{}", r.dump());
    assert!(h
        .last_prompt()
        .starts_with("<bos><|turn>system\n<|tool>get_weather<tool|>"));
    let msg = at(&r, &["choices", "0", "message"]);
    assert_eq!(s(msg, &["reasoning_content"]), "Need the tool.");
    assert_eq!(s(msg, &["content"]), "On it.");
    assert_eq!(
        s(msg, &["tool_calls", "0", "function", "name"]),
        "get_weather"
    );
    // the schema types `days` as an integer
    assert_eq!(
        s(msg, &["tool_calls", "0", "function", "arguments"]),
        "{\"city\":\"Oslo\",\"days\":2}"
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "tool_calls");
    // a forced call in this form
    h.script("city:<|\"|>Rome<|\"|>}<tool_call|>");
    let (_, r) = h.chat(&format!(",{TOOLS},\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}"));
    assert!(h
        .last_prompt()
        .ends_with("<|turn>model\n<|tool_call>call:get_weather{"));
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "arguments"
            ]
        ),
        "{\"city\":\"Rome\"}"
    );
}

const MESSAGES_TOOLS: &str = r#""tools":[{"name":"get_weather","description":"Weather.","input_schema":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}]"#;

#[test]
fn messages_endpoint_returns_content_blocks() {
    let h = harness();
    h.script(CALL_REPLY);
    let (status, text) = h.post(
        "/v1/messages",
        &format!("{{\"model\":\"any\",\"max_tokens\":512,\"system\":\"Be brief.\",{MESSAGES_TOOLS},\"messages\":[{{\"role\":\"user\",\"content\":\"Weather in Oslo?\"}}]}}"),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(
        (s(&r, &["type"]), s(&r, &["role"]), s(&r, &["stop_reason"])),
        ("message", "assistant", "tool_use")
    );
    assert!(s(&r, &["id"]).starts_with("msg_"));
    let blocks = at(&r, &["content"]).as_array().unwrap();
    let kinds: Vec<&str> = blocks.iter().map(|b| s(b, &["type"])).collect();
    assert_eq!(kinds, ["thinking", "text", "tool_use"]);
    assert_eq!(s(&blocks[0], &["thinking"]), "I should look it up.");
    assert_eq!(
        anthropic::signature_reasoning(s(&blocks[0], &["signature"])).as_deref(),
        Some("I should look it up.")
    );
    assert_eq!(s(&blocks[1], &["text"]), "Let me check.");
    assert!(s(&blocks[2], &["id"]).starts_with("toolu_"));
    assert_eq!(
        at(&blocks[2], &["input"]).dump(),
        "{\"city\":\"Oslo \\\"S\\\"\",\"days\":2}"
    );
    assert!(at(&r, &["usage", "input_tokens"]).as_u64().unwrap() > 0);
    assert!(at(&r, &["usage", "output_tokens"]).as_u64().unwrap() > 0);
    assert!(h
        .last_prompt()
        .contains("<|im_start|>system\nBe brief.<|im_end|>"));

    // the tool result goes back; a stop sequence ends the next reply
    h.script("</think>\n\nSunny. END of story");
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":512,"stop_sequences":["END"],{MESSAGES_TOOLS},"messages":[
                {{"role":"user","content":"Weather in Oslo?"}},
                {{"role":"assistant","content":[{{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{{"city":"Oslo"}}}}]}},
                {{"role":"user","content":[{{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}}]}}]}}"#
        ),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(
        (s(&r, &["stop_reason"]), s(&r, &["stop_sequence"])),
        ("stop_sequence", "END")
    );
    assert_eq!(s(&r, &["content", "0", "text"]), "Sunny. ");
    assert!(h
        .last_prompt()
        .contains("<tool_response>\nsunny\n</tool_response>"));

    // errors in this API's shape
    let (status, text) = h.post(
        "/v1/messages",
        r#"{"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"x"}]}"#,
    );
    let r = json::parse_json(&text).unwrap();
    assert_eq!(
        (status, s(&r, &["type"]), s(&r, &["error", "type"])),
        (400, "error", "invalid_request_error")
    );
}

#[test]
fn messages_endpoint_streams_blocks_that_rebuild_the_message() {
    let h = harness();
    h.script(CALL_REPLY);
    let (status, body) = h.post(
        "/v1/messages",
        &format!("{{\"stream\":true,\"max_tokens\":512,{MESSAGES_TOOLS},\"messages\":[{{\"role\":\"user\",\"content\":\"Weather?\"}}]}}"),
    );
    assert_eq!(status, 200);
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
    assert_eq!(events.first().unwrap().0, "message_start");
    assert_eq!(events.last().unwrap().0, "message_stop");
    let count = |name: &str| events.iter().filter(|(n, _)| n == name).count();
    assert_eq!(
        (count("content_block_start"), count("content_block_stop")),
        (3, 3)
    );
    // each block's deltas rebuild its text
    let joined = |index: u64, kind: &str, field: &str| -> String {
        events
            .iter()
            .filter(|(n, d)| {
                n == "content_block_delta"
                    && d.get("index").and_then(JsonValue::as_u64) == Some(index)
            })
            .filter_map(|(_, d)| {
                d.get("delta")
                    .filter(|x| x.get("type").and_then(JsonValue::as_str) == Some(kind))
            })
            .map(|x| s(x, &[field]).to_string())
            .collect()
    };
    assert_eq!(
        joined(0, "thinking_delta", "thinking"),
        "I should look it up."
    );
    assert_eq!(joined(1, "text_delta", "text"), "Let me check.");
    assert_eq!(
        joined(2, "input_json_delta", "partial_json"),
        "{\"city\":\"Oslo \\\"S\\\"\",\"days\":2}"
    );
    let start = &events
        .iter()
        .filter(|(n, _)| n == "content_block_start")
        .nth(2)
        .unwrap()
        .1;
    assert_eq!(s(start, &["content_block", "name"]), "get_weather");
    let delta = &events.iter().find(|(n, _)| n == "message_delta").unwrap().1;
    assert_eq!(s(delta, &["delta", "stop_reason"]), "tool_use");
    assert!(at(delta, &["usage", "output_tokens"]).as_u64().unwrap() > 0);
}

#[test]
fn template_and_token_count_endpoints_and_props() {
    let h = harness();
    let body = format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],{TOOLS}}}");
    let (status, text) = h.post("/apply-template", &body);
    assert_eq!(status, 200);
    let prompt = json::parse_json(&text).unwrap();
    assert!(
        s(&prompt, &["prompt"]).contains("<tools>")
            && s(&prompt, &["prompt"]).ends_with("<think>\n")
    );
    let (_, text) = h.post("/gaudium/count_tokens", &body);
    let with_tools = at(&json::parse_json(&text).unwrap(), &["n_prompt_tokens"])
        .as_u64()
        .unwrap();
    let (_, text) = h.post(
        "/gaudium/count_tokens",
        r#"{"messages":[{"role":"user","content":"Hi"}]}"#,
    );
    let without = at(&json::parse_json(&text).unwrap(), &["n_prompt_tokens"])
        .as_u64()
        .unwrap();
    assert!(
        with_tools > without + 100,
        "{with_tools} vs {without}: the tools count"
    );
    let (status, text) = h.post(
        "/v1/messages/count_tokens",
        &format!("{{{MESSAGES_TOOLS},\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}"),
    );
    assert_eq!(status, 200, "{text}");
    assert!(
        at(&json::parse_json(&text).unwrap(), &["input_tokens"])
            .as_u64()
            .unwrap()
            > without
    );

    let props = json::parse_json(&h.state.props_json(0, true)).unwrap();
    let caps = at(&props, &["chat_template_caps"]);
    for key in [
        "supports_tools",
        "supports_tool_calls",
        "supports_parallel_tool_calls",
    ] {
        assert_eq!(
            caps.get(key).and_then(JsonValue::as_bool),
            Some(true),
            "{key}"
        );
    }
}

#[test]
fn a2ui_declaration_adds_the_instructions_to_the_system_turn() {
    let h = harness();
    // a request that declares nothing renders as before
    h.script("Hello.");
    let (status, _) = h.chat("");
    assert_eq!(status, 200);
    assert!(!h.last_prompt().contains("a2ui"));

    // a declared renderer: the instructions open the conversation as a system turn
    h.script("A form helps.\n</think>\n\nHere.\n<a2ui-json>\n[]\n</a2ui-json>");
    let (status, reply) = h.chat(",\"gaudium_a2ui\":true");
    assert_eq!(status, 200, "{reply:?}");
    let prompt = h.last_prompt();
    assert!(
        prompt.starts_with("<|im_start|>system\n# Interactive interfaces (A2UI 1.0)"),
        "{prompt}"
    );
    assert!(prompt.contains("<|im_end|>\n<|im_start|>user\nWeather in Oslo?"));
    // the model's block reaches the client inside the ordinary content
    assert!(s(&reply, &["choices", "0", "message", "content"]).contains("<a2ui-json>"));

    // the same prompt is what the template and token-count endpoints see
    let body = r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],"gaudium_a2ui":{"1.0":{"supportedCatalogIds":["https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"]}}}"#;
    let (status, text) = h.post("/apply-template", body);
    assert_eq!(status, 200, "{text}");
    let rendered = json::parse_json(&text).unwrap();
    let rendered = s(&rendered, &["prompt"]);
    assert!(rendered.starts_with("<|im_start|>system\nBe brief.\n\n# Interactive interfaces"));
    assert_eq!(rendered.matches("<|im_start|>system").count(), 1);
    let (_, text) = h.post("/gaudium/count_tokens", body);
    let with = at(&json::parse_json(&text).unwrap(), &["n_prompt_tokens"])
        .as_u64()
        .unwrap();
    let (_, text) = h.post(
        "/gaudium/count_tokens",
        r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]}"#,
    );
    let without = at(&json::parse_json(&text).unwrap(), &["n_prompt_tokens"])
        .as_u64()
        .unwrap();
    assert!(with > without + 500, "{with} vs {without}");

    // a malformed declaration is refused; the Messages endpoint takes the field too
    let (status, _) = h.chat(",\"gaudium_a2ui\":\"yes\"");
    assert_eq!(status, 400);
    h.script("Hello.");
    let (status, text) = h.post(
        "/v1/messages",
        r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],"gaudium_a2ui":true}"#,
    );
    assert_eq!(status, 200, "{text}");
    assert!(h.last_prompt().contains("# Interactive interfaces (A2UI 1.0)"));

    let props = json::parse_json(&h.state.props_json(0, true)).unwrap();
    assert_eq!(at(&props, &["a2ui", "enabled"]).as_bool(), Some(true));
    assert_eq!(s(&props, &["a2ui", "request_field"]), "gaudium_a2ui");
    assert!(at(&props, &["a2ui"]).get("instructions").is_none());
}
