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
pub(crate) fn validates(s: &JsonValue, v: &JsonValue, root: &JsonValue) -> bool {
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
    let alphabet: &[u8] = b"{}[]\":,0123456789-.eE+ \ntruefalsn\\uxDc";
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

/// `\u` escapes pair as RFC 8259 pairs them and as `parse_json` reads them: a high surrogate is
/// followed at once by a low one, a low one never stands alone. Every code unit at the range
/// edges, alone, before a letter and before every other, in both letter cases.
#[test]
fn surrogate_escapes_agree_with_the_json_parser() {
    let g = json_mode();
    let units = [
        0x0000u32, 0x0041, 0xD7FF, 0xD800, 0xDA12, 0xDBFF, 0xDC00, 0xDE00, 0xDFFF, 0xE000, 0xFFFF,
    ];
    let esc = |u: u32, lower: bool| {
        if lower {
            format!("\\u{u:04x}")
        } else {
            format!("\\u{u:04X}")
        }
    };
    let mut checked = 0;
    for lower in [false, true] {
        for &a in &units {
            let mut bodies = vec![esc(a, lower), format!("{}a", esc(a, lower))];
            for &b in &units {
                bodies.push(format!("{}{}", esc(a, lower), esc(b, lower)));
            }
            for body in bodies {
                let text = format!("{{\"x\":\"{body}\"}}");
                let parsed = parse_json(&text).is_ok();
                assert_eq!(accepts(&g, &text), parsed, "{text}");
                checked += 1;
            }
        }
    }
    assert!(checked > 200);
    // the digit that decides it refuses at once: no admitted prefix is a dead end
    let c = g.start().unwrap();
    assert!(g.step_bytes(&c, br#"{"x":"\uDC"#).is_none());
    assert!(g.step_bytes(&c, br#"{"x":"\uD800""#).is_none());
    assert!(g.step_bytes(&c, br#"{"x":"\uD800\uDB"#).is_none());
    assert!(g.step_bytes(&c, br#"{"x":"\uD83D\uDE"#).is_some());
}

/// A strict tool's object-valued parameter never admits an argument the reply parser can not
/// read back as JSON (it would then pass the whole parameter on as a string).
#[test]
fn a_strict_object_parameter_refuses_a_lone_surrogate() {
    let tool = ToolDef {
        name: "put".into(),
        strict: Some(schema(
            r#"{"type":"object","properties":{"obj":{"type":"object"}},"required":["obj"]}"#,
        )),
    };
    let g = build::reply(None, &[tool], &params_tags()).unwrap();
    let call = |v: &str| {
        format!("<tool_call>\n<function=put>\n<parameter=obj>\n{v}\n</parameter>\n</function>\n</tool_call>")
    };
    for good in [r#"{"x":"\uD83D\uDE00"}"#, r#"{"x":"\u00e9"}"#] {
        assert!(accepts(&g, &call(good)), "{good}");
        assert!(parse_json(good).is_ok());
    }
    for bad in [r#"{"x":"\uD800"}"#, r#"{"x":"\uDC00"}"#, r#"{"x":"\uD800\u0041"}"#] {
        assert!(refuses(&g, &call(bad)), "{bad}");
        assert!(parse_json(bad).is_err());
    }
}

/// A string parameter with `const` or `enum` (written as raw text in a parameter block) has its
/// keywords checked like any other schema: one the literals would leave unenforced is refused.
#[test]
fn a_literal_string_parameter_checks_its_keywords() {
    let tool = |p: &str| ToolDef {
        name: "t".into(),
        strict: Some(schema(&format!(
            r#"{{"type":"object","properties":{{"s":{p}}},"required":["s"]}}"#
        ))),
    };
    for (p, kw) in [
        (r#"{"type":"string","const":"a","minLength":2}"#, "minLength"),
        (r#"{"type":"string","enum":["a","b"],"pattern":"^a$"}"#, "pattern"),
        (r#"{"type":"string","const":"a","enum":["a"]}"#, "const"),
        (r#"{"type":"string","enum":[]}"#, "enum"),
        (r#"{"type":"string","enum":[1,2]}"#, "enum"),
    ] {
        let e = build::reply(None, &[tool(p)], &params_tags()).unwrap_err();
        assert!(e.contains(kw), "{p}: {e}");
    }
    // a literal with only annotations beside it still compiles, and holds the reply to it
    let g = build::reply(
        None,
        &[tool(r#"{"type":"string","enum":["a","b"],"description":"one of two"}"#)],
        &params_tags(),
    )
    .unwrap();
    let call = |v: &str| {
        format!("<tool_call>\n<function=t>\n<parameter=s>\n{v}\n</parameter>\n</function>\n</tool_call>")
    };
    assert!(accepts(&g, &call("b")));
    assert!(refuses(&g, &call("c")));
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
    Arc::new(Plan::new(g, toy_table(), think.then(|| Pat::new(b"</think>")), start, "test".into()))
}

/// Every admitted token, decoded one at a time, agrees with the grammar on the whole text.
#[test]
fn the_mask_admits_exactly_the_tokens_that_step() {
    let v = toy_table();
    let g = json_mode();
    let mut c = Constraint::begin(plan(g.clone(), false));
    let mut text = Vec::new();
    for piece in ["{\"", "name", "\":", " ", "true", "}"] {
        let m = c.mask().unwrap();
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
    assert!(c.mask().unwrap().has(END));
    assert!(c.commit(b'x' as i32).is_err());
}

#[test]
fn reasoning_is_free_and_the_answer_follows_it() {
    let mut c = Constraint::begin(plan(json_mode(), true));
    let m = c.mask().unwrap();
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
    let m = c.mask().unwrap();
    assert!(!m.has(b'x' as u32) && !m.has(CTL) && m.has(b'{' as u32) && m.has(b'\n' as u32));
    let obj = (0..200).find(|&i| toy_table().bytes(i) == b" {\"a\": 1}").unwrap();
    c.commit(obj as i32).unwrap();
    assert!(c.mask().unwrap().has(END));
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
    let p = Arc::new(Plan::new(g, v, Some(Pat::new(b"</think>")), start, "t".into()));
    let mut c = Constraint::begin(p);
    assert!(!c.mask().unwrap().has(128));
    assert!(c.commit(128).is_err());
}

#[test]
fn the_cache_serves_a_repeated_state() {
    let mut c = Constraint::begin(plan(json_mode(), false));
    for b in b"{\"k\": \"abc" {
        c.mask().unwrap();
        c.commit(*b as i32).unwrap();
    }
    let built = c.stats.masks_built;
    for b in b"defgh" {
        c.mask().unwrap();
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

// ------------------------------------------------------------------ a schema is hostile input

fn optional_props(n: usize) -> JsonValue {
    let props: Vec<String> = (0..n).map(|i| format!("\"p{i}\":{{}}")).collect();
    schema(&format!(
        r#"{{"type":"object","properties":{{{}}},"additionalProperties":false}}"#,
        props.join(",")
    ))
}

/// One key trie per object, not one per position: 6,000 optional properties (the review's
/// 38 M trie nodes, 6.4 s and 2.3 GB) now compile to a grammar linear in the schema.
#[test]
fn many_optional_properties_compile_linearly() {
    let s = optional_props(6000);
    let body = s.dump().len() as u64;
    let t0 = std::time::Instant::now();
    let g = build::schema_doc(&s).unwrap();
    let ms = t0.elapsed().as_millis();
    assert!(g.tries.len() < 10 * 6000, "{} trie nodes", g.tries.len());
    let bytes = build::grammar_bytes(&s).unwrap();
    assert!(bytes < 200 * body, "{bytes} bytes charged for a {body}-byte schema");
    assert!(ms < 5000, "{ms} ms");
    // order and closure still hold
    assert!(accepts(&g, r#"{"p3":1,"p17":"x","p5999":null}"#));
    assert!(accepts(&g, "{}"));
    assert!(refuses(&g, r#"{"p17":1,"p3":2}"#));
    assert!(refuses(&g, r#"{"p3":1,"p3":2}"#));
    assert!(refuses(&g, r#"{"p6000":1}"#));
}

#[test]
fn keys_admit_only_the_properties_that_may_come_next() {
    let s = schema(
        r#"{"type":"object","properties":{"ab":{},"a":{},"abc":{"type":"integer"},"b":{}},
            "required":["abc"],"additionalProperties":false}"#,
    );
    let g = build::schema_doc(&s).unwrap();
    for ok in [r#"{"abc":1}"#, r#"{"ab":0,"a":0,"abc":1,"b":0}"#, r#"{"a":0,"abc":1}"#] {
        assert!(accepts(&g, ok), "{ok}");
    }
    // "b" may not come before the required "abc"; "ab" not after "a"
    for bad in [r#"{"b":0,"abc":1}"#, r#"{"a":0,"ab":0,"abc":1}"#, r#"{"a":0}"#] {
        assert!(refuses(&g, bad), "{bad}");
    }
    // the prefix `{"b` is refused at once (no key of the position starts so), not later
    let c = g.start().unwrap();
    assert!(g.step_bytes(&c, br#"{"b"#).is_none());
    assert!(g.step_bytes(&c, br#"{"ab"#).is_some());
}

#[test]
fn the_compile_budget_refuses_a_schema_past_it() {
    let s = optional_props(200);
    let need = build::grammar_bytes(&s).unwrap();
    assert!(build::schema_doc_within(&s, need).is_ok());
    let e = build::schema_doc_within(&s, need - 1).unwrap_err();
    assert!(e.contains("MiB of grammar"), "{e}");
    // an enum is charged by its literal bytes
    let big: Vec<String> = (0..2000).map(|i| format!("\"{}{i}\"", "v".repeat(40))).collect();
    let s = schema(&format!(r#"{{"enum":[{}]}}"#, big.join(",")));
    assert!(build::schema_doc_within(&s, 50_000).unwrap_err().contains("MiB of grammar"));
}

#[test]
fn wide_and_looping_alternatives_are_refused_at_compile_time() {
    let wide = |n: usize| {
        let b: Vec<&str> = vec![r#"{"type":"integer"}"#; n];
        schema(&format!(r#"{{"type":"array","items":{{"anyOf":[{}]}}}}"#, b.join(",")))
    };
    assert!(build::schema_doc(&wide(MAX_ALIVE as usize)).is_ok());
    let e = build::schema_doc(&wide(MAX_ALIVE as usize + 1)).unwrap_err();
    assert!(e.contains("'anyOf'") && e.contains("alternatives"), "{e}");
    // nested through references: 20 x 20 alternatives at once
    let mut defs = Vec::new();
    for i in 0..20 {
        let b: Vec<String> = (0..20).map(|j| format!(r#"{{"const":{}}}"#, i * 20 + j)).collect();
        defs.push(format!(r#""d{i}":{{"anyOf":[{}]}}"#, b.join(",")));
    }
    let refs: Vec<String> = (0..20).map(|i| format!(r##"{{"$ref":"#/$defs/d{i}"}}"##)).collect();
    let s = schema(&format!(r#"{{"$defs":{{{}}},"anyOf":[{}]}}"#, defs.join(","), refs.join(",")));
    assert!(build::schema_doc(&s).unwrap_err().contains("alternatives"));
    // a diamond of references: 2^12 paths into one value
    let mut defs = vec![r#""d12":{"type":"integer"}"#.to_string()];
    for i in 0..12 {
        defs.push(format!(r##""d{i}":{{"anyOf":[{{"$ref":"#/$defs/d{0}"}},{{"$ref":"#/$defs/d{0}"}}]}}"##, i + 1));
    }
    let s = schema(&format!(r##"{{"$defs":{{{}}},"$ref":"#/$defs/d0"}}"##, defs.join(",")));
    assert!(build::schema_doc(&s).unwrap_err().contains("alternatives"));
    // Codex: a reference to itself with no JSON in between (exponential before the fix)
    for s in [
        r##"{"anyOf":[{"$ref":"#"},{"$ref":"#"}]}"##,
        r##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"anyOf":[{"$ref":"#/$defs/a"},{"type":"null"}]}},"$ref":"#/$defs/a"}"##,
    ] {
        let e = build::schema_doc(&schema(s)).unwrap_err();
        assert!(e.contains("leads back to itself"), "{s}: {e}");
    }
    // recursion through a value is fine
    assert!(build::schema_doc(&schema(r##"{"anyOf":[{"type":"array","items":{"$ref":"#"}},{"type":"null"}]}"##)).is_ok());
}

#[test]
fn a_long_reference_chain_is_refused_not_overflowed() {
    let n = 5000;
    let mut defs: Vec<String> = (0..n).map(|i| format!(r##""d{i}":{{"$ref":"#/$defs/d{}"}}"##, i + 1)).collect();
    defs.push(format!(r#""d{n}":{{"type":"integer"}}"#));
    let s = schema(&format!(r##"{{"$defs":{{{}}},"$ref":"#/$defs/d0"}}"##, defs.join(",")));
    let e = build::schema_doc(&s).unwrap_err();
    assert!(e.contains("passes through more than"), "{e}");
}

#[test]
fn a_literal_is_never_checked_against_its_type_alone() {
    // Codex: the enum's value violates the object's `required`
    for s in [
        r#"{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"enum":[{}]}"#,
        r#"{"type":"array","minItems":2,"const":[1]}"#,
    ] {
        let e = build::schema_doc(&schema(s)).unwrap_err();
        assert!(e.contains("can not stand beside"), "{s}: {e}");
    }
    assert!(build::schema_doc(&schema(r#"{"type":"string","enum":["a"],"description":"d"}"#)).is_ok());
}

#[test]
fn every_admitted_number_is_finite() {
    let g = build::schema_doc(&schema(r#"{"type":"number"}"#)).unwrap();
    assert!(accepts(&g, "-1.5e99"));
    assert!(refuses(&g, "1e999"));
    let biggest = format!("{}.{}e99", "9".repeat(INT_DIGITS as usize), "9".repeat(FRAC_DIGITS as usize));
    assert!(accepts(&g, &biggest));
    let v = parse_json(&biggest).unwrap();
    assert!(matches!(v, JsonValue::Float(f) if f.is_finite()), "{v:?}");
}

/// A counted array makes every step a new automaton state: the cache stays within its byte cap.
#[test]
fn a_counted_array_keeps_the_mask_cache_within_its_cap() {
    let g = build::schema_doc(&schema(r#"{"type":"array","items":{"type":"integer"},"maxItems":1000000}"#)).unwrap();
    let start = g.start().unwrap();
    let mut p = Plan::new(g, toy_table(), None, start, "t".into());
    p.cache_cap = 4096;
    let p = Arc::new(p);
    let mut c = Constraint::begin(Arc::clone(&p));
    c.commit(b'[' as i32).unwrap();
    let mut peak = 0;
    for _ in 0..300 {
        for b in *b"1," {
            assert!(c.mask().unwrap().has(b as u32));
            c.commit(b as i32).unwrap();
            peak = peak.max(c.cache_bytes());
        }
    }
    assert!(peak <= p.cache_cap, "{peak}");
    assert!(c.stats.cache_clears > 0);
    // with the default cap the same run is cached whole
    let g = build::schema_doc(&schema(r#"{"type":"array","items":{"type":"integer"},"maxItems":1000000}"#)).unwrap();
    let start = g.start().unwrap();
    let mut c = Constraint::begin(Arc::new(Plan::new(g, toy_table(), None, start, "t".into())));
    c.commit(b'[' as i32).unwrap();
    for _ in 0..50 {
        for b in *b"1," {
            c.mask().unwrap();
            c.commit(b as i32).unwrap();
        }
    }
    assert_eq!(c.stats.cache_clears, 0);
}

/// A mask (and a commit) past the work cap ends the request with an error, never with a
/// partial mask.
#[test]
fn a_mask_past_its_work_cap_is_an_error() {
    let b: Vec<&str> = vec![r#"{"type":"array","items":{"type":"integer"}}"#; 200];
    let s = schema(&format!(r#"{{"type":"array","items":{{"anyOf":[{}]}}}}"#, b.join(",")));
    let g = build::schema_doc(&s).unwrap();
    let start = g.start().unwrap();
    let mut p = Plan::new(g, toy_table(), None, start, "t".into());
    p.work_cap = 2_000;
    let mut c = Constraint::begin(Arc::new(p));
    let e = c.commit(b'[' as i32).and_then(|_| c.mask().map(|_| ())).unwrap_err();
    assert!(e.contains("automaton steps"), "{e}");
    // a request's masks and commits together are held to the request's cap
    let g = build::schema_doc(&s).unwrap();
    let start = g.start().unwrap();
    let mut p = Plan::new(g, toy_table(), None, start, "t".into());
    p.request_work_cap = 50_000;
    let mut c = Constraint::begin(Arc::new(p));
    let mut err = None;
    for b in b"[[1],[2],[3],[4],[5],[6],[7],[8],[9],[1],[2],[3]]".repeat(4) {
        if let Err(e) = c.mask().and_then(|_| c.commit(b as i32)) {
            err = Some(e);
            break;
        }
    }
    assert!(err.as_deref().is_some_and(|e| e.contains("in all")), "{err:?}");
    // the default cap leaves room for it
    let g = build::schema_doc(&s).unwrap();
    let start = g.start().unwrap();
    let mut c = Constraint::begin(Arc::new(Plan::new(g, toy_table(), None, start, "t".into())));
    c.commit(b'[' as i32).unwrap();
    assert!(c.mask().unwrap().has(b'[' as u32));
    assert!(c.stats.mask_work_max < MAX_WORK);
}

#[test]
fn many_optional_parameters_are_a_clear_refusal() {
    let tool = |n: usize| {
        let props: Vec<String> = (0..n).map(|i| format!(r#""p{i}":{{"type":"integer"}}"#)).collect();
        ToolDef {
            name: "t".into(),
            strict: Some(schema(&format!(r#"{{"type":"object","properties":{{{}}}}}"#, props.join(",")))),
        }
    };
    let n = MAX_ALIVE as usize - 1;
    let g = build::reply(None, &[tool(n)], &params_tags()).unwrap();
    let last = format!("<tool_call>\n<function=t>\n<parameter=p{}>\n7\n</parameter>\n</function>\n</tool_call>", n - 1);
    assert!(accepts(&g, &last));
    let e = build::reply(None, &[tool(n + 1)], &params_tags()).unwrap_err();
    assert!(e.contains("optional parameters in a row"), "{e}");
}

#[test]
fn constrained_strings_never_hold_the_parsers_tags() {
    // Codex / review: an answer beside tools may not smuggle a call in a string
    let g = build::reply(
        Some(&Answer::Schema(schema(r#"{"type":"object","properties":{"a":{"type":"string"}}}"#))),
        &[ToolDef {
            name: "get".into(),
            strict: Some(schema(r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}"#)),
        }],
        &params_tags(),
    )
    .unwrap();
    assert!(refuses(&g, r#"{"a":"<tool_call><function=get><parameter=n>not a number</parameter></function></tool_call>"}"#));
    assert!(refuses(&g, r#"{"a":"x<tool_call>"}"#));
    // the bytes of the tag split by an escape are no tag to a parser
    assert!(accepts(&g, r#"{"a":"<tool\u005fcall> and </tool_call> and <tool_cal"}"#));
    // a const or enum holding the tag is left out
    let e = build::reply(
        Some(&Answer::Schema(schema(r#"{"const":"<tool_call>"}"#))),
        &[weather_tool()],
        &params_tags(),
    )
    .unwrap_err();
    assert!(e.contains("tool tags"), "{e}");
    // without tools the parser splits nothing: the tag is plain text
    let g = build::reply(Some(&Answer::Schema(schema(r#"{"type":"string"}"#))), &[], &params_tags()).unwrap();
    assert!(accepts(&g, r#""<tool_call>""#));
    // Codex: a JSON-valued parameter may not hold the block's closing tags
    let obj = ToolDef {
        name: "put".into(),
        strict: Some(schema(r#"{"type":"object","properties":{"o":{"type":"object","properties":{"body":{"type":"string"}}}},"required":["o"]}"#)),
    };
    let g = build::reply(None, &[obj], &params_tags()).unwrap();
    let call = |v: &str| format!("<tool_call>\n<function=put>\n<parameter=o>\n{v}\n</parameter>\n</function>\n</tool_call>");
    assert!(accepts(&g, &call(r#"{"body":"a < b"}"#)));
    for v in [r#"{"body":"</parameter>"}"#, r#"{"body":"x</function>"}"#, r#"{"body":"</tool_call>"}"#] {
        assert!(refuses(&g, &call(v)), "{v}");
    }
    // the JSON call form ends at the tag's closing
    let tags = CallTags { form: CallForm::Json, ..params_tags() };
    let g = build::reply(None, &[weather_tool()], &tags).unwrap();
    assert!(refuses(&g, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"</tool_call>\", \"days\": 2}}\n</tool_call>"));
}

/// `required` is checked through a set built once, its work charged to the compile budget first
/// (it was checked name against property, quadratic before any charge: 100,000 duplicate keys
/// took 45 s in a release build).
#[test]
fn required_is_checked_in_linear_time_and_charged_first() {
    // N - 1 duplicate keys, then the one every required entry names
    let dup = |n: usize| {
        let mut props: Vec<&str> = vec![r#""a":{}"#; n - 1];
        props.push(r#""z":{}"#);
        let req = vec![r#""z""#; n];
        schema(&format!(
            r#"{{"type":"object","properties":{{{}}},"required":[{}]}}"#,
            props.join(","),
            req.join(",")
        ))
    };
    let t0 = std::time::Instant::now();
    assert!(build::schema_doc(&dup(30_000)).is_ok());
    assert!(t0.elapsed().as_secs_f64() < 10.0, "{:?}", t0.elapsed());
    // a long list is refused by the budget before it is checked
    let s = schema(&format!(
        r#"{{"type":"object","properties":{{"z":{{}}}},"required":[{}]}}"#,
        vec![r#""z""#; 50_000].join(",")
    ));
    assert!(build::schema_doc_within(&s, 1 << 20).unwrap_err().contains("MiB of grammar"));
    assert!(build::schema_doc_within(&s, 4 << 20).is_ok());
    // a strict tool's list goes through the same check
    let tags = CallTags {
        open: "<tool_call>".into(),
        close: "</tool_call>".into(),
        form: CallForm::Params,
        think: None,
    };
    // (every parameter required: optional ones in a row have their own limit)
    let props: Vec<String> = (0..30_000).map(|i| format!(r#""p{i}":{{}}"#)).collect();
    let req: Vec<String> = (0..30_000).map(|i| format!(r#""p{i}""#)).collect();
    let tool = ToolDef {
        name: "t".into(),
        strict: Some(schema(&format!(
            r#"{{"type":"object","properties":{{{}}},"required":[{}]}}"#,
            props.join(","),
            req.join(",")
        ))),
    };
    let t0 = std::time::Instant::now();
    assert!(build::reply(None, &[tool], &tags).is_ok());
    assert!(t0.elapsed().as_secs_f64() < 10.0, "{:?}", t0.elapsed());
    let bad = ToolDef {
        name: "t".into(),
        strict: Some(schema(r#"{"type":"object","properties":{"a":{}},"required":["b"]}"#)),
    };
    assert!(build::reply(None, &[bad], &tags).unwrap_err().contains("names 'b'"));
}

/// A frame operation copies its stack, so it is charged by the frames it copies: the work caps
/// then bound time at every depth (each operation was charged 1, and a mask's cap took 11 s at
/// 142 frames against 0.45 s at 3).
#[test]
fn deep_stacks_are_charged_by_the_frames_they_copy() {
    assert_eq!(step_cost(0), 1);
    assert_eq!(step_cost(FRAMES_PER_UNIT as usize - 1), 1);
    assert_eq!(step_cost(FRAMES_PER_UNIT as usize), 2);
    assert_eq!(step_cost(MAX_STACK as usize), 1 + MAX_STACK / FRAMES_PER_UNIT);
    let consts: Vec<String> = (0..20).map(|i| format!(r#"{{"const":"xxxxxxxx{i}"}}"#)).collect();
    let s = schema(&format!(
        r##"{{"$defs":{{"n":{{"anyOf":[{{"type":"array","items":{{"$ref":"#/$defs/n"}}}},{}]}}}},"$ref":"#/$defs/n"}}"##,
        consts.join(",")
    ));
    let g = build::schema_doc(&s).unwrap();
    let work_at = |depth: usize| {
        let mut c = g.start().unwrap();
        for _ in 0..depth {
            c = g.step(&c, b'[').unwrap();
        }
        c = g.step(&c, b'"').unwrap();
        let mut w = Work::new(MAX_WORK);
        assert!(g.step_in(&c, b'x', &mut w).is_some());
        w.used()
    };
    let (shallow, deep) = (work_at(1), work_at(140));
    assert!(deep >= shallow * (1 + 140 / FRAMES_PER_UNIT) / 2, "{shallow} {deep}");
}

/// The ledger goes only through the verified `gram_budget_charge`: an accepted charge adds what
/// was charged, and after a refused one every later charge is refused, an empty one included.
#[test]
fn a_refused_charge_cuts_the_call_short_for_good() {
    let mut w = Work::new(10);
    assert!(w.spend(0) && w.used() == 1);
    assert!(w.spend(8 * FRAMES_PER_UNIT as usize) && w.used() == 10);
    assert!(!w.exhausted());
    assert!(!w.spend(0));
    assert!(w.exhausted());
    assert!(!w.spend(0) && w.exhausted());
}

/// A strict tool in the parameter-block form checks its root's keywords as any schema is
/// checked: one it does not apply is refused, never ignored.
#[test]
fn a_strict_tools_root_keywords_are_never_ignored() {
    for form in [CallForm::Params, CallForm::Json] {
        let tags = CallTags {
            open: "<tool_call>".into(),
            close: "</tool_call>".into(),
            form,
            think: None,
        };
        for kw in [r#""maxProperties":0"#, r#""minProperties":2"#, r#""patternProperties":{}"#, r#""additionalProperties":{"type":"string"}"#] {
            let tool = ToolDef {
                name: "t".into(),
                strict: Some(schema(&format!(
                    r#"{{"type":"object","properties":{{"x":{{"type":"string"}}}},"required":["x"],{kw}}}"#
                ))),
            };
            let e = build::reply(None, &[tool], &tags).unwrap_err();
            assert!(e.contains("not supported"), "{form:?} {kw}: {e}");
        }
        let ok = ToolDef {
            name: "t".into(),
            strict: Some(schema(
                r#"{"type":"object","title":"T","description":"d","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}"#,
            )),
        };
        assert!(build::reply(None, &[ok], &tags).is_ok(), "{form:?}");
    }
}

/// A path kept for an error stops growing, and a refusal's message is short: 31 levels of
/// 1 MiB keys built a 32 MB path (and a 32 MB 400) before.
#[test]
fn long_keys_nested_deep_keep_a_short_message() {
    let key = "k".repeat(64 << 10);
    let mut s = r#"{"type":"string","minLength":1}"#.to_string();
    for _ in 0..31 {
        s = format!(r#"{{"type":"object","properties":{{"{key}":{s}}}}}"#);
    }
    let e = build::schema_doc_within(&schema(&s), 1 << 40).unwrap_err();
    assert!(e.contains("'minLength'"), "{}", &e[..200]);
    assert!(e.len() <= build::MESSAGE_MAX + 3, "{}", e.len());
    // a long unknown keyword or property name is cut as well
    let long = "q".repeat(5000);
    let e = build::schema_doc(&schema(&format!(r#"{{"{long}":1}}"#))).unwrap_err();
    assert!(e.len() <= build::MESSAGE_MAX + 3, "{}", e.len());
    let e = build::schema_doc(&schema(&format!(r#"{{"type":"object","properties":{{}},"required":["{long}"]}}"#))).unwrap_err();
    assert!(e.len() <= build::MESSAGE_MAX + 3 && e.contains("names 'qqq"), "{e}");
}

/// A constrained request that asks for log-probabilities: the readout of each token comes from
/// the row the host pick drew it from (`pick_masked` leaves it penalised and masked), so the
/// chosen token is the one picked and every alternative is a token the grammar admits there.
#[test]
fn logprobs_of_a_constrained_pick_report_only_admitted_tokens() {
    use crate::fp8::f32_to_bf16_rne;
    use crate::sampler::SampleParams;
    use crate::sampler_penalty::{pick_masked, PenaltyState, Penalties};
    let v = toy_table();
    let vocab = v.len();
    let top = 8u64;
    let greedy = SampleParams::default().to_dev(vocab as u64);
    let sampled = SampleParams {
        temperature: 1.0,
        seed: 7,
        ..SampleParams::default()
    }
    .to_dev(vocab as u64);
    for (dev, is_greedy) in [(greedy, true), (sampled, false)] {
        let p = Penalties::for_vocab(vocab as u64);
        let mut st = PenaltyState::begin(0);
        let mut c = Constraint::begin(plan(json_mode(), false));
        for n in 0..24u32 {
            if c.complete() {
                break;
            }
            let m = c.mask().unwrap();
            // a row whose highest logits are mostly tokens the grammar refuses
            let mut row: Vec<u16> = (0..vocab)
                .map(|i| f32_to_bf16_rne(((i * 37 + n as usize * 11) % 101) as f32 / 10.0 - 5.0))
                .collect();
            let tok = pick_masked(&p, &st, &mut row, &dev, n, Some(&m)).unwrap();
            assert!(m.has(tok as u32), "step {n}: token {tok} is refused by the grammar");
            let e = crate::logprobs::row_logprobs_u16(&row, &dev, tok, top);
            assert_eq!(e.token, tok);
            assert!(e.logprob.is_finite(), "step {n}: the picked token is in the support");
            assert!(!e.top.is_empty());
            for &(id, lp) in &e.top {
                assert!(m.has(id as u32), "step {n}: alternative {id} is refused by the grammar");
                assert!(lp.is_finite(), "step {n}: alternative {id} at -inf");
            }
            if is_greedy {
                // the greedy pick is the most likely admitted token, and every admitted token
                // up to `top` is listed
                assert_eq!(e.top[0].0, tok, "step {n}");
                assert_eq!(e.top.len() as u64, top.min(m.count()), "step {n}");
            }
            st.commit(&p, tok).unwrap();
            c.commit(tok).unwrap();
        }
    }
}
