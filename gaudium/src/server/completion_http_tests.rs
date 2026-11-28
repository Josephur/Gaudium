//! `POST /completion`, `/completions` and `/v1/completions` over real sockets, with the scripted
//! engine of `http_tests`: the prompt reaches the engine as the client gave it, and the reply
//! has the shape of the endpoint.

use super::*;

fn parsed(text: &str) -> JsonValue {
    json::parse_json(text).unwrap_or_else(|e| panic!("{e}: {text}"))
}

/// The `data:` payloads of a stream, and whether it ended with `[DONE]`.
fn stream_payloads(body: &str) -> (Vec<JsonValue>, bool) {
    let mut done = false;
    let mut out = Vec::new();
    for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            done = true;
        } else {
            out.push(parsed(line));
        }
    }
    (out, done)
}

#[test]
fn text_prompt_runs_as_given_without_a_template() {
    let h = harness();
    // a completion's output is its text as generated: no reasoning split, no tool parsing
    h.script("<think>a</think> b <tool_call>c</tool_call>");
    let (status, text) = h.post(
        "/v1/completions",
        r#"{"model":"tags","prompt":"Once upon a time","temperature":0}"#,
    );
    assert_eq!(status, 200, "{text}");
    assert_eq!(h.last_prompt(), "Once upon a time");
    let v = parsed(&text);
    assert_eq!(s(&v, &["object"]), "text_completion");
    assert!(s(&v, &["id"]).starts_with("cmpl-"));
    assert_eq!(s(&v, &["model"]), "tags");
    let choice = at(&v, &["choices", "0"]);
    assert_eq!(at(choice, &["index"]).as_u64(), Some(0));
    assert_eq!(
        s(choice, &["text"]),
        "<think>a</think> b <tool_call>c</tool_call>"
    );
    assert_eq!(s(choice, &["finish_reason"]), "stop");
    assert!(matches!(at(choice, &["logprobs"]), JsonValue::Null));
    let n_prompt = h.state.models[0].tokenizer.encode("Once upon a time", true).len() as u64;
    assert_eq!(at(&v, &["usage", "prompt_tokens"]).as_u64(), Some(n_prompt));
    assert!(at(&v, &["usage", "completion_tokens"]).as_u64().unwrap() > 0);
    // the same path answers on the unversioned name, and a one-element array is that prompt
    h.script("x");
    let (status, text) = h.post("/completions", r#"{"prompt":["Hi there"]}"#);
    assert_eq!(status, 200, "{text}");
    assert_eq!(h.last_prompt(), "Hi there");
}

/// A completion given the token ids a chat request renders runs exactly that prompt, and its
/// generated ids are the chat's (the engine sees the same input and the same parameters).
#[test]
fn token_id_prompt_equals_the_chat_prompt() {
    let h = harness();
    let chat = r#"{"messages":[{"role":"user","content":"Weather in Oslo?"}],"temperature":0,"seed":7,"gaudium_return_tokens":true}"#;
    let (_, counted) = h.post(
        "/gaudium/count_tokens",
        &chat.replacen('{', r#"{"return_tokens":true,"#, 1),
    );
    let ids = at(&parsed(&counted), &["tokens"]).dump();
    h.script("Sunny.");
    let (status, chat_text) = h.post("/v1/chat/completions", chat);
    assert_eq!(status, 200, "{chat_text}");
    let chat_prompt = h.last_prompt();
    h.script("Sunny.");
    let (status, text) = h.post(
        "/v1/completions",
        &format!(r#"{{"prompt":{ids},"temperature":0,"seed":7,"gaudium_return_tokens":true}}"#),
    );
    assert_eq!(status, 200, "{text}");
    assert_eq!(h.last_prompt(), chat_prompt);
    let v = parsed(&text);
    let c = parsed(&chat_text);
    assert_eq!(
        at(&v, &["usage", "prompt_tokens"]).as_u64(),
        at(&c, &["usage", "prompt_tokens"]).as_u64()
    );
    assert_eq!(
        at(&v, &["gaudium", "tokens"]).dump(),
        at(&c, &["gaudium", "tokens"]).dump()
    );
    assert_eq!(s(&v, &["choices", "0", "text"]), "Sunny.");
    // the nested form of the same ids
    h.script("Sunny.");
    let (status, _) = h.post("/v1/completions", &format!(r#"{{"prompt":[{ids}]}}"#));
    assert_eq!(status, 200);
    assert_eq!(h.last_prompt(), chat_prompt);
}

#[test]
fn completion_streams_text_chunks() {
    let h = harness();
    h.script("one two three");
    let (status, body) = h.post(
        "/v1/completions",
        r#"{"prompt":"Count:","stream":true,"temperature":0}"#,
    );
    assert_eq!(status, 200, "{body}");
    let (chunks, done) = stream_payloads(&body);
    assert!(done, "{body}");
    assert!(chunks.len() >= 2, "{body}");
    let mut text = String::new();
    for c in &chunks {
        assert_eq!(s(c, &["object"]), "text_completion");
        assert!(c.get("choices").is_some(), "{}", c.dump());
        text.push_str(s(c, &["choices", "0", "text"]));
    }
    assert_eq!(text, "one two three");
    let last = chunks.last().unwrap();
    assert_eq!(s(last, &["choices", "0", "finish_reason"]), "stop");
    assert!(at(last, &["usage", "completion_tokens"]).as_u64().unwrap() > 0);
    // every chunk before the last is unfinished
    for c in &chunks[..chunks.len() - 1] {
        assert!(matches!(
            at(c, &["choices", "0", "finish_reason"]),
            JsonValue::Null
        ));
    }
}

#[test]
fn native_completion_reports_how_it_stopped() {
    let h = harness();
    h.script("alpha beta gamma");
    let (status, text) = h.post("/completion", r#"{"prompt":"Go","temperature":0}"#);
    assert_eq!(status, 200, "{text}");
    let v = parsed(&text);
    assert_eq!(s(&v, &["content"]), "alpha beta gamma");
    assert_eq!(at(&v, &["stop"]).as_bool(), Some(true));
    assert_eq!(s(&v, &["stop_type"]), "eos");
    assert!(at(&v, &["tokens_predicted"]).as_u64().unwrap() > 0);
    let n_prompt = h.state.models[0].tokenizer.encode("Go", true).len() as u64;
    assert_eq!(at(&v, &["tokens_evaluated"]).as_u64(), Some(n_prompt));

    // `n_predict` limits the generation
    h.script("alpha beta gamma");
    let (_, text) = h.post("/completion", r#"{"prompt":"Go","n_predict":1}"#);
    let v = parsed(&text);
    assert_eq!(s(&v, &["stop_type"]), "limit");
    assert_eq!(at(&v, &["tokens_predicted"]).as_u64(), Some(1));

    // a stop string ends it, and is named
    h.script("alpha beta gamma");
    let (_, text) = h.post("/completion", r#"{"prompt":"Go","stop":[" beta"]}"#);
    let v = parsed(&text);
    assert_eq!(s(&v, &["content"]), "alpha");
    assert_eq!(s(&v, &["stop_type"]), "word");
    assert_eq!(s(&v, &["stopping_word"]), " beta");
}

#[test]
fn native_completion_streams_content_chunks() {
    let h = harness();
    h.script("alpha beta");
    let (status, body) = h.post("/completion", r#"{"prompt":"Go","stream":true}"#);
    assert_eq!(status, 200, "{body}");
    let (chunks, done) = stream_payloads(&body);
    assert!(!done, "a native stream has no end marker: {body}");
    let mut text = String::new();
    for c in &chunks {
        text.push_str(s(c, &["content"]));
    }
    assert_eq!(text, "alpha beta");
    let last = chunks.last().unwrap();
    assert_eq!(at(last, &["stop"]).as_bool(), Some(true));
    assert_eq!(s(last, &["stop_type"]), "eos");
    for c in &chunks[..chunks.len() - 1] {
        assert_eq!(at(c, &["stop"]).as_bool(), Some(false));
    }
}

#[test]
fn completion_requests_are_refused_with_the_reason() {
    let h = harness();
    let rows = h.state.models[0].tokenizer.vocab_size();
    let long = vec!["0"; 16384].join(",");
    let cases = [
        (r#"{"prompt":"a","n":2}"#.to_string(), "'n'"),
        (r#"{"prompt":"a","echo":true}"#.to_string(), "'echo'"),
        (r#"{"prompt":["a","b"]}"#.to_string(), "2 prompts"),
        (r#"{"messages":[]}"#.to_string(), "'prompt' is required"),
        (r#"{"prompt":[1,-1]}"#.to_string(), "non-negative"),
        (format!(r#"{{"prompt":[1,{rows}]}}"#), "outside the model's vocabulary"),
        // an empty prompt and one past the served context fail the shared admission
        (r#"{"prompt":""}"#.to_string(), "exceeds context budget"),
        (format!(r#"{{"prompt":[{long}]}}"#), "exceeds context budget"),
        ("not json".to_string(), "invalid JSON"),
    ];
    for (body, want) in cases {
        for path in ["/v1/completions", "/completion"] {
            let (status, text) = h.post(path, &body);
            assert_eq!(status, 400, "{path} {body}: {text}");
            let v = parsed(&text);
            let msg = s(&v, &["error", "message"]);
            assert!(msg.contains(want), "{path} {body}: {msg}");
        }
    }
    // nothing reached the engine
    assert!(h.prompts.lock().unwrap().is_empty());
    for path in ["/completion", "/completions", "/v1/completions"] {
        let (status, text) = h.get(path);
        assert_eq!(status, 405, "{path}: {text}");
        assert!(text.contains("takes POST"), "{text}");
    }
}

#[test]
fn completion_fields_not_applied_are_reported() {
    let h = harness();
    h.script("ok");
    let (status, text) = h.post(
        "/v1/completions",
        r#"{"prompt":"a","logprobs":2,"frequency_penalty":0.4,"repeat_penalty":1.1,"user":"u","n":1,"echo":false}"#,
    );
    assert_eq!(status, 200, "{text}");
    let v = parsed(&text);
    let ignored: Vec<String> = at(&v, &["gaudium", "ignored"])
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    assert_eq!(ignored.len(), 3, "{ignored:?}");
    assert!(ignored[0].starts_with("logprobs"));
    // frequency_penalty is applied by the shared run: not reported; the native name is
    assert!(ignored[1].starts_with("repeat_penalty"), "{ignored:?}");
    assert_eq!(ignored[2], "user");
}
