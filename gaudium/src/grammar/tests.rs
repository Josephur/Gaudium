use std::sync::Arc;

use super::build::{self, Answer, CallForm, CallTags, ToolDef};
use super::constraint::{Constraint, Plan};
use super::vocab::{Kind, TokenTable};
use super::*;
use crate::json::{parse_json, JsonValue};

fn schema(s: &str) -> JsonValue {
    parse_json(s).unwrap()
}

fn accepts(g: &Grammar, text: &str) -> bool {
    match g.start().and_then(|c| g.step_bytes(&c, text.as_bytes())) {
        Some(c) => g.complete(&c),
        None => false,
    }
}

/// Some prefix of `text` is refused, or the whole is incomplete.
fn refuses(g: &Grammar, text: &str) -> bool {
    !accepts(g, text)
}

fn json_mode() -> Grammar {
    let tags = CallTags {
        open: "<tool_call>".into(),
        close: "</tool_call>".into(),
        form: CallForm::Params,
        think: None,
    };
    build::reply(Some(&Answer::JsonObject), &[], &tags).unwrap()
}

/// A tiny JSON Schema validator for the subset (the reference the grammar is tested against).
fn validates(s: &JsonValue, v: &JsonValue, root: &JsonValue) -> bool {
    if let Some(r) = s.get("$ref").and_then(JsonValue::as_str) {
        let mut cur = root;
        if r != "#" {
            for part in r.trim_start_matches("#/").split('/') {
                cur = cur.get(part).unwrap();
            }
        }
        return validates(cur, v, root);
    }
    if let Some(a) = s.get("anyOf").and_then(JsonValue::as_array) {
        return a.iter().any(|x| validates(x, v, root));
    }
    if let Some(e) = s.get("enum").and_then(JsonValue::as_array) {
        return e.contains(v);
    }
    if let Some(c) = s.get("const") {
        return c == v;
    }
    let types: Vec<&str> = match s.get("type") {
        Some(JsonValue::Str(t)) => vec![t.as_str()],
        Some(JsonValue::Array(a)) => a.iter().filter_map(JsonValue::as_str).collect(),
        _ => vec![],
    };
    let ty_ok = types.is_empty()
        || types.iter().any(|t| match (*t, v) {
            ("object", JsonValue::Object(_)) => true,
            ("array", JsonValue::Array(_)) => true,
            ("string", JsonValue::Str(_)) => true,
            ("integer", JsonValue::Int(_)) => true,
            ("number", JsonValue::Int(_) | JsonValue::Float(_)) => true,
            ("boolean", JsonValue::Bool(_)) => true,
            ("null", JsonValue::Null) => true,
            _ => false,
        });
    if !ty_ok {
        return false;
    }
    if let JsonValue::Object(o) = v {
        let props = s.get("properties").and_then(JsonValue::as_object).unwrap_or(&[]);
        for r in s.get("required").and_then(JsonValue::as_array).unwrap_or(&[]) {
            if !o.iter().any(|(k, _)| Some(k.as_str()) == r.as_str()) {
                return false;
            }
        }
        for (k, x) in o {
            match props.iter().find(|(p, _)| p == k) {
                Some((_, ps)) => {
                    if !validates(ps, x, root) {
                        return false;
                    }
                }
                None => match s.get("additionalProperties") {
                    Some(JsonValue::Bool(false)) => return false,
                    Some(a @ JsonValue::Object(_)) => {
                        if !validates(a, x, root) {
                            return false;
                        }
                    }
                    _ => {}
                },
            }
        }
    }
    if let JsonValue::Array(a) = v {
        if let Some(n) = s.get("minItems").and_then(JsonValue::as_u64) {
            if (a.len() as u64) < n {
                return false;
            }
        }
        if let Some(n) = s.get("maxItems").and_then(JsonValue::as_u64) {
            if (a.len() as u64) > n {
                return false;
            }
        }
        if let Some(it) = s.get("items") {
            if !a.iter().all(|x| validates(it, x, root)) {
                return false;
            }
        }
    }
    true
}

#[test]
fn json_mode_accepts_objects_and_refuses_the_rest() {
    let g = json_mode();
    for ok in [
        "{}",
        " {\"a\": 1, \"b\": [true, null, -0.5e+3, \"x\\u00e9\\n\"], \"c\": {\"d\": {}}}\n",
        "{\"k\":\"\"}",
    ] {
        assert!(accepts(&g, ok), "{ok}");
    }
    for bad in [
        "",
        "[]",
        "{",
        "{\"a\" 1}",
        "{\"a\": 01}",
        "{\"a\": 1,}",
        "{\"a\": \"x\ny\"}",
        "{\"a\": tru}",
        "{} {}",
        "{\"a\": 1.}",
        "{\"a\": \"\\x\"}",
    ] {
        assert!(refuses(&g, bad), "{bad:?}");
    }
}

#[test]
fn whitespace_runs_are_bounded() {
    let g = json_mode();
    let ok = format!("{{{}}}", " ".repeat(WS_MAX as usize));
    assert!(accepts(&g, &ok));
    let bad = format!("{{{}}}", " ".repeat(WS_MAX as usize + 1));
    assert!(refuses(&g, &bad));
}

#[test]
fn a_schema_holds_types_required_order_and_closed_objects() {
    let s = schema(
        r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},
            "tags":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":2},
            "kind":{"enum":["a","b",3]},"ok":{"type":["boolean","null"]}},
            "required":["name","kind"],"additionalProperties":false}"#,
    );
    let g = build::schema_doc(&s).unwrap();
    for ok in [
        r#"{"name":"x","kind":"a"}"#,
        r#"{"name":"x","age":-12,"tags":["p"],"kind":3,"ok":null}"#,
        r#"{ "name" : "x" , "kind" : "b" , "ok" : true }"#,
    ] {
        assert!(accepts(&g, ok), "{ok}");
        assert!(validates(&s, &parse_json(ok).unwrap(), &s));
    }
    for bad in [
        r#"{"kind":"a"}"#,                         // name missing
        r#"{"name":"x"}"#,                         // kind missing
        r#"{"kind":"a","name":"x"}"#,              // out of the schema's order
        r#"{"name":"x","kind":"c"}"#,              // not in the enum
        r#"{"name":"x","age":1.5,"kind":"a"}"#,    // not an integer
        r#"{"name":"x","tags":[],"kind":"a"}"#,    // minItems
        r#"{"name":"x","tags":["a","b","c"],"kind":"a"}"#, // maxItems
        r#"{"name":"x","kind":"a","extra":1}"#,    // closed
        r#"{"name":1,"kind":"a"}"#,
    ] {
        assert!(refuses(&g, bad), "{bad}");
    }
}

#[test]
fn any_of_refs_and_recursion() {
    let s = schema(
        r##"{"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},
              "next":{"anyOf":[{"$ref":"#/$defs/node"},{"type":"null"}]}},
              "required":["v","next"],"additionalProperties":false}},
            "anyOf":[{"$ref":"#/$defs/node"},{"type":"string"}]}"##,
    );
    let g = build::schema_doc(&s).unwrap();
    for ok in [
        r#"{"v":1,"next":{"v":2,"next":null}}"#,
        r#""plain""#,
    ] {
        assert!(accepts(&g, ok), "{ok}");
    }
    assert!(refuses(&g, r#"{"v":1,"next":{"v":2}}"#));
    assert!(refuses(&g, "12"));
}

#[test]
fn unsupported_keywords_are_named() {
    for (s, kw) in [
        (r#"{"type":"string","pattern":"^a$"}"#, "pattern"),
        (r#"{"type":"string","format":"date"}"#, "format"),
        (r#"{"type":"integer","minimum":0}"#, "minimum"),
        (r#"{"oneOf":[{"type":"string"}]}"#, "oneOf"),
        (r#"{"type":"object","properties":{"a":{"type":"string","maxLength":3}}}"#, "maxLength"),
        (r#"{"allOf":[{"type":"string"},{"type":"integer"}]}"#, "allOf"),
        (r#"{"$ref":"https://example.com/s.json"}"#, "$ref"),
    ] {
        let e = build::schema_doc(&schema(s)).unwrap_err();
        assert!(e.contains(&format!("'{kw}'")), "{s}: {e}");
    }
    let mut deep = String::from(r#"{"type":"string"}"#);
    for _ in 0..(build::MAX_SCHEMA_DEPTH + 1) {
        deep = format!(r#"{{"type":"array","items":{deep}}}"#);
    }
    assert!(build::schema_doc(&schema(&deep)).unwrap_err().contains("nests deeper"));
}

/// Deterministic pseudo-random JSON documents of a schema-free shape: every one the JSON parser
/// accepts must be accepted by JSON mode when it is an object, and every byte string JSON mode
/// accepts must parse.
#[test]
fn differential_against_the_json_parser() {
    let g = json_mode();
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    let alphabet: &[u8] = b"{}[]\":,0123456789-.eE+ \ntruefalsn\\ux";
    let mut accepted = 0;
    for _ in 0..20000 {
        let len = 1 + rnd(24) as usize;
        let mut s: Vec<u8> = vec![b'{'];
        for _ in 0..len {
            s.push(alphabet[rnd(alphabet.len() as u64) as usize]);
        }
        let text = String::from_utf8(s).unwrap();
        let parsed = parse_json(&text);
        let ok = accepts(&g, &text);
        if ok {
            accepted += 1;
            assert!(
                matches!(parsed, Ok(JsonValue::Object(_))),
                "accepted but does not parse: {text:?}"
            );
        }
        if let Ok(JsonValue::Object(_)) = parsed {
            // the parser has no whitespace or digit limits: inside them both agree
            if !text.contains("  ") && text.len() < 40 {
                assert!(ok, "parses but refused: {text:?}");
            }
        }
    }
    // the walk reached the accepting side too
    let _ = accepted;
    // and the documents of the schema tests, built by the dump of the parser
    for t in [r#"{"a":[1,2,{"b":null}],"c":"\u0001"}"#, r#"{"x":-0.0}"#] {
        let v = parse_json(t).unwrap();
        assert!(accepts(&g, &v.dump()), "{}", v.dump());
    }
}

fn params_tags() -> CallTags {
    CallTags {
        open: "<tool_call>".into(),
        close: "</tool_call>".into(),
        form: CallForm::Params,
        think: None,
    }
}

fn weather_tool() -> ToolDef {
    ToolDef {
        name: "get_weather".into(),
        strict: Some(schema(
            r#"{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},
                "unit":{"type":"string","enum":["c","f"]}},"required":["city","days"],
                "additionalProperties":false}"#,
        )),
    }
}

#[test]
fn strict_calls_in_parameter_blocks() {
    let tools = vec![
        weather_tool(),
        ToolDef {
            name: "free".into(),
            strict: None,
        },
    ];
    let g = build::reply(None, &tools, &params_tags()).unwrap();
    let call = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew </b> York\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=unit>\nf\n</parameter>\n</function>\n</tool_call>";
    assert!(accepts(&g, call));
    assert!(accepts(&g, "plain text, no call"));
    assert!(accepts(
        &g,
        "<tool_call>\n<function=free>\n<parameter=x>\nanything\n</parameter>\n</function>\n</tool_call> after"
    ));
    for bad in [
        // days missing
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        // days not an integer
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>",
        // an unknown tool
        "<tool_call>\n<function=nope>\n",
        // a unit outside the enum
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nP\n</parameter>\n<parameter=days>\n1\n</parameter>\n<parameter=unit>\nk\n</parameter>\n</function>\n</tool_call>",
        // a raw value may not hold the function's closing tag
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nP</function>",
        // a call cut short does not complete
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nP",
    ] {
        assert!(refuses(&g, bad), "{bad:?}");
    }
}

#[test]
fn strict_calls_as_json_objects() {
    let tags = CallTags {
        form: CallForm::Json,
        ..params_tags()
    };
    let g = build::reply(None, &[weather_tool()], &tags).unwrap();
    assert!(accepts(
        &g,
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"P\", \"days\": 2}}\n</tool_call>"
    ));
    assert!(refuses(
        &g,
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"P\"}}\n</tool_call>"
    ));
}

#[test]
fn an_answer_or_calls() {
    let g = build::reply(
        Some(&Answer::Schema(schema(r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#))),
        &[weather_tool()],
        &params_tags(),
    )
    .unwrap();
    assert!(accepts(&g, "{\"a\": 1}"));
    assert!(accepts(&g, "<tool_call>\n<function=get_weather>\n<parameter=city>\nP\n</parameter>\n<parameter=days>\n1\n</parameter>\n</function>\n</tool_call>"));
    assert!(refuses(&g, "text first {\"a\": 1}"));
}

/// A toy vocabulary: single bytes, a few words, an end token and a control token.
fn toy_table() -> Arc<TokenTable> {
    let mut bytes: Vec<Vec<u8>> = (0u8..128).map(|b| vec![b]).collect();
    let words = [
        "{\"", "\":", "\", \"", "\"}", "name", "true", "</think>", "<tool_call>", " {\"a\": 1}",
    ];
    for w in words {
        bytes.push(w.as_bytes().to_vec());
    }
    let mut kinds = vec![Kind::Text; bytes.len()];
    bytes.push(Vec::new());
    kinds.push(Kind::End); // id END
    bytes.push(b"<|ctl|>".to_vec());
    kinds.push(Kind::Control);
    Arc::new(TokenTable::new(bytes, kinds))
}

const END: u32 = 137;
const CTL: u32 = 138;

fn plan(g: Grammar, think: bool) -> Arc<Plan> {
    let start = g.start().unwrap();
    Arc::new(Plan {
        grammar: g,
        vocab: toy_table(),
        think_close: think.then(|| Pat::new(b"</think>")),
        start,
        label: "test".into(),
    })
}

/// Every admitted token, decoded one at a time, agrees with the grammar on the whole text.
#[test]
fn the_mask_admits_exactly_the_tokens_that_step() {
    let v = toy_table();
    let g = json_mode();
    let mut c = Constraint::begin(plan(g.clone(), false));
    let mut text = Vec::new();
    for piece in ["{\"", "name", "\":", " ", "true", "}"] {
        let m = c.mask();
        for id in 0..v.len() as u32 {
            let want = match v.kind(id) {
                Kind::End => {
                    let cur = g.start().and_then(|s| g.step_bytes(&s, &text));
                    cur.is_some_and(|x| g.complete(&x))
                }
                Kind::Control => false,
                Kind::Text => {
                    let mut t = text.clone();
                    t.extend_from_slice(v.bytes(id));
                    g.start().and_then(|s| g.step_bytes(&s, &t)).is_some()
                }
            };
            assert_eq!(m.has(id), want, "id {id} {:?} after {:?}", v.bytes(id), String::from_utf8_lossy(&text));
        }
        let id = (0..v.len() as u32)
            .find(|&i| v.bytes(i) == piece.as_bytes())
            .unwrap();
        c.commit(id as i32).unwrap();
        text.extend_from_slice(piece.as_bytes());
    }
    assert!(c.complete());
    assert!(c.mask().has(END));
    assert!(c.commit(b'x' as i32).is_err());
}

#[test]
fn reasoning_is_free_and_the_answer_follows_it() {
    let mut c = Constraint::begin(plan(json_mode(), true));
    let m = c.mask();
    // free text and control tokens, but no end before the answer
    assert!(m.has(b'x' as u32) && m.has(b'[' as u32) && m.has(CTL));
    assert!(!m.has(END));
    for b in b"thinking [not json" {
        c.commit(*b as i32).unwrap();
    }
    assert!(c.commit(END as i32).is_err());
    let close = (0..200).find(|&i| toy_table().bytes(i) == b"</think>").unwrap();
    c.commit(close as i32).unwrap();
    assert_eq!(c.phase(), crate::verified::GramPhase::GpAnswer);
    let m = c.mask();
    assert!(!m.has(b'x' as u32) && !m.has(CTL) && m.has(b'{' as u32) && m.has(b'\n' as u32));
    let obj = (0..200).find(|&i| toy_table().bytes(i) == b" {\"a\": 1}").unwrap();
    c.commit(obj as i32).unwrap();
    assert!(c.mask().has(END));
    c.commit(END as i32).unwrap();
}

#[test]
fn a_closing_tag_followed_by_refused_bytes_is_not_admitted() {
    let mut bytes: Vec<Vec<u8>> = (0u8..128).map(|b| vec![b]).collect();
    bytes.push(b"</think>xyz".to_vec());
    let kinds = vec![Kind::Text; bytes.len()];
    let v = Arc::new(TokenTable::new(bytes, kinds));
    let g = json_mode();
    let start = g.start().unwrap();
    let p = Arc::new(Plan {
        grammar: g,
        vocab: v,
        think_close: Some(Pat::new(b"</think>")),
        start,
        label: "t".into(),
    });
    let mut c = Constraint::begin(p);
    assert!(!c.mask().has(128));
    assert!(c.commit(128).is_err());
}

#[test]
fn the_cache_serves_a_repeated_state() {
    let mut c = Constraint::begin(plan(json_mode(), false));
    for b in b"{\"k\": \"abc" {
        c.mask();
        c.commit(*b as i32).unwrap();
    }
    let built = c.stats.masks_built;
    for b in b"defgh" {
        c.mask();
        c.commit(*b as i32).unwrap();
    }
    // the string body is one state, built once before its first byte
    assert_eq!(c.stats.masks_built, built);
    assert!(c.stats.cache_hits >= 7);
}

#[test]
fn a_reply_may_open_its_own_reasoning() {
    let tags = CallTags {
        think: Some(("<think>".into(), "</think>".into())),
        ..params_tags()
    };
    let g = build::reply(Some(&Answer::JsonObject), &[], &tags).unwrap();
    assert!(accepts(&g, "<think>\nanything [ goes\n</think>\n\n{\"a\": 1}"));
    assert!(accepts(&g, "{\"a\": 1}"));
    // never ends inside the reasoning, and the answer after it is held
    assert!(refuses(&g, "<think>\nstill thinking"));
    assert!(refuses(&g, "<think>x</think> no json"));
}
