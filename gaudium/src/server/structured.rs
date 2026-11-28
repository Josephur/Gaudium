//! Request fields of structured output, parsed into the reply grammar (`crate::grammar`):
//!
//! - chat `response_format`: `{"type": "text"}` (nothing to hold), `{"type": "json_object"}`
//!   (any JSON object), `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`;
//! - a function tool with `"strict": true`: the arguments of a call to it follow its
//!   `parameters` (Messages `strict` tools arrive here through `anthropic::translate`, and
//!   `output_config.format` as a `json_schema` response format).
//!
//! A schema outside the supported subset is refused with the keyword's name (the caller answers
//! 400). Whether the model's engine may take a constrained request is the verified
//! `pen_path_allowed` of `gram_host_pick`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::chat::ChatPrompt;
use super::outparse::{CallStyle, OutputFormat};
use crate::grammar::build::{self, Answer, CallForm, CallTags, ToolDef};
use crate::grammar::constraint::Plan;
use crate::grammar::vocab::TokenTable;
use crate::grammar::{Pat, Work, MAX_WORK};
use crate::json::JsonValue;
use crate::tokenizer::Tokenizer;

/// What a request asks its reply to hold to.
#[derive(Debug, Clone)]
pub struct Request {
    pub answer: Option<Answer>,
    /// The tools the reply may call (strict ones with their schema).
    pub tools: Vec<ToolDef>,
    pub label: String,
}

/// Reads `response_format` and the strict tools of a rendered request; `None`: the reply is
/// not constrained.
pub fn request(val: &JsonValue, prompt: &ChatPrompt) -> Result<Option<Request>, String> {
    let mut label = Vec::new();
    let answer = match val.get("response_format") {
        None | Some(JsonValue::Null) => None,
        Some(rf @ JsonValue::Object(_)) => match rf.get("type").and_then(JsonValue::as_str) {
            Some("text") => None,
            Some("json_object") => {
                label.push("response_format json_object".to_string());
                Some(Answer::JsonObject)
            }
            Some("json_schema") => {
                let Some(js) = rf.get("json_schema").filter(|j| j.as_object().is_some()) else {
                    return Err("response_format: 'json_schema' must be an object {\"name\", \"schema\", \"strict\"}".into());
                };
                let schema = match js.get("schema") {
                    Some(s @ (JsonValue::Object(_) | JsonValue::Bool(true))) => s.clone(),
                    None => JsonValue::Bool(true),
                    Some(_) => {
                        return Err("response_format.json_schema.schema must be a JSON schema object".into())
                    }
                };
                let name = js.get("name").and_then(JsonValue::as_str).unwrap_or("");
                label.push(if name.is_empty() {
                    "response_format json_schema".to_string()
                } else {
                    format!("response_format json_schema '{name}'")
                });
                Some(Answer::Schema(schema))
            }
            Some(other) => {
                return Err(format!(
                    "response_format.type '{other}' is not supported (text, json_object or json_schema)"
                ))
            }
            None => return Err("response_format: 'type' is required".into()),
        },
        Some(_) => return Err("response_format must be an object".into()),
    };
    let mut tools = Vec::new();
    let mut strict = 0;
    for t in prompt
        .tools
        .as_ref()
        .and_then(JsonValue::as_array)
        .unwrap_or(&[])
    {
        let f = t.get("function").unwrap_or(t);
        let Some(name) = f.get("name").and_then(JsonValue::as_str) else {
            continue;
        };
        let is_strict = f.get("strict").and_then(JsonValue::as_bool) == Some(true);
        let schema = is_strict.then(|| {
            f.get("parameters").cloned().unwrap_or_else(|| {
                JsonValue::Object(vec![("type".into(), JsonValue::Str("object".into()))])
            })
        });
        strict += usize::from(is_strict);
        if !tools.iter().any(|x: &ToolDef| x.name == name) {
            tools.push(ToolDef {
                name: name.to_string(),
                strict: schema,
            });
        }
    }
    if strict > 0 {
        label.push(format!("strict tools ({strict})"));
    }
    if answer.is_none() && strict == 0 {
        return Ok(None);
    }
    if prompt.continue_final {
        return Err(
            "structured output (response_format or strict tools) can not continue a prefilled assistant message"
                .into(),
        );
    }
    Ok(Some(Request {
        answer,
        tools,
        label: label.join(", "),
    }))
}

/// The grammar of `req` for a model whose replies have the form `format`, started where the
/// rendered prompt leaves the reply (in the reasoning, or after a forced call's opening).
pub fn plan(
    req: &Request,
    prompt: &ChatPrompt,
    format: &OutputFormat,
    vocab: Arc<TokenTable>,
) -> Result<Arc<Plan>, String> {
    let form = match format.style {
        CallStyle::Params => Some(CallForm::Params),
        CallStyle::Json => Some(CallForm::Json),
        CallStyle::Call | CallStyle::Blocks => None,
    };
    let form = match (form, req.tools.is_empty()) {
        (Some(f), _) => f,
        (None, true) => CallForm::Params,
        (None, false) => {
            return Err(format!(
                "{}: this model's tool-call form can not be held to a schema",
                req.label
            ))
        }
    };
    let tags = CallTags {
        open: format.tool_open.clone(),
        close: format.tool_close.clone(),
        form,
        think: offers_reasoning(prompt, format)
            .then(|| (format.think_open.clone(), format.think_close.clone())),
    };
    let grammar = build::reply(req.answer.as_ref(), &req.tools, &tags)?;
    let mut w = Work::new(MAX_WORK);
    let start = grammar
        .start_in(&mut w)
        .and_then(|c| grammar.step_bytes_in(&c, prompt.forced_prefix.as_bytes(), &mut w));
    if w.exhausted() {
        return Err(format!("{}: the reply's format takes too many automaton steps to start", req.label));
    }
    let start = start.ok_or_else(|| {
        format!(
            "{}: the forced tool call's opening does not fit the reply's format",
            req.label
        )
    })?;
    Ok(Arc::new(Plan::new(
        grammar,
        vocab,
        prompt
            .reasoning_open
            .then(|| Pat::new(format.think_close.as_bytes())),
        start,
        req.label.clone(),
    )))
}

/// The reply may open the model's reasoning itself: the prompt neither opened it (the reply
/// then starts inside it) nor closed it (reasoning turned off renders an empty block), and no
/// forced call's opening starts the reply.
fn offers_reasoning(prompt: &ChatPrompt, format: &OutputFormat) -> bool {
    let close = format.think_close.trim();
    !prompt.reasoning_open
        && prompt.forced_prefix.is_empty()
        && !close.is_empty()
        && !prompt.text.trim_end().ends_with(close)
}

/// The grammar's view of a model's vocabulary, built once per model (and end-token set).
pub fn table(
    model: &str,
    tok: &Tokenizer,
    vocab: u64,
    end_ids: &[i32],
    format: &OutputFormat,
) -> Arc<TokenTable> {
    type Tables = Mutex<HashMap<(String, u64, Vec<i32>), Arc<TokenTable>>>;
    static TABLES: OnceLock<Tables> = OnceLock::new();
    let key = (model.to_string(), vocab, end_ids.to_vec());
    let tables = TABLES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(t) = tables.lock().unwrap().get(&key) {
        return Arc::clone(t);
    }
    let t = Arc::new(TokenTable::from_tokenizer(tok, vocab, end_ids, |s| {
        format.is_marker_piece(s)
    }));
    tables.lock().unwrap().insert(key, Arc::clone(&t));
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    fn prompt(tools: Option<&str>) -> ChatPrompt {
        ChatPrompt {
            text: String::new(),
            reasoning_open: false,
            tools: tools.map(|t| parse_json(t).unwrap()),
            forced_prefix: String::new(),
            single_call: false,
            continue_final: false,
            ignored: Vec::new(),
            media: Vec::new(),
        }
    }

    fn req(body: &str, tools: Option<&str>) -> Result<Option<Request>, String> {
        request(&parse_json(body).unwrap(), &prompt(tools))
    }

    #[test]
    fn response_formats_are_read() {
        assert!(req(r#"{}"#, None).unwrap().is_none());
        assert!(req(r#"{"response_format":{"type":"text"}}"#, None).unwrap().is_none());
        let r = req(r#"{"response_format":{"type":"json_object"}}"#, None).unwrap().unwrap();
        assert!(matches!(r.answer, Some(Answer::JsonObject)));
        let r = req(
            r#"{"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object"}}}}"#,
            None,
        )
        .unwrap()
        .unwrap();
        assert!(matches!(r.answer, Some(Answer::Schema(_))));
        assert_eq!(r.label, "response_format json_schema 'x'");
        for bad in [
            r#"{"response_format":"json"}"#,
            r#"{"response_format":{"type":"grammar"}}"#,
            r#"{"response_format":{"type":"json_schema"}}"#,
            r#"{"response_format":{"type":"json_schema","json_schema":{"schema":3}}}"#,
        ] {
            assert!(req(bad, None).is_err(), "{bad}");
        }
    }

    #[test]
    fn strict_tools_are_read_and_plain_tools_join_them() {
        let tools = r#"[{"type":"function","function":{"name":"a","strict":true,"parameters":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}}},
                        {"type":"function","function":{"name":"b","parameters":{"type":"object"}}}]"#;
        let r = req("{}", Some(tools)).unwrap().unwrap();
        assert!(r.answer.is_none());
        assert_eq!(r.tools.len(), 2);
        assert!(r.tools[0].strict.is_some() && r.tools[1].strict.is_none());
        assert_eq!(r.label, "strict tools (1)");
        // no strict tool and no format: nothing to hold
        let plain = r#"[{"type":"function","function":{"name":"b"}}]"#;
        assert!(req("{}", Some(plain)).unwrap().is_none());
    }

    #[test]
    fn a_prefilled_reply_is_refused() {
        let mut p = prompt(None);
        p.continue_final = true;
        let body = parse_json(r#"{"response_format":{"type":"json_object"}}"#).unwrap();
        assert!(request(&body, &p).is_err());
    }

    #[test]
    fn a_forced_call_starts_the_grammar_inside_it() {
        use crate::grammar::constraint::Constraint;
        use crate::grammar::vocab::Kind;
        let tools = r#"[{"type":"function","function":{"name":"get","strict":true,"parameters":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}}}]"#;
        let mut p = prompt(Some(tools));
        let f = OutputFormat::default();
        p.forced_prefix = f.forced_prefix("get");
        let r = request(&parse_json("{}").unwrap(), &p).unwrap().unwrap();
        let bytes: Vec<Vec<u8>> = (0u8..128).map(|b| vec![b]).collect();
        let kinds = vec![Kind::Text; 128];
        let plan = plan(&r, &p, &f, Arc::new(TokenTable::new(bytes, kinds))).unwrap();
        let mut c = Constraint::begin(plan);
        for b in "<parameter=n>\n7\n</parameter>\n</function>\n</tool_call>".bytes() {
            assert!(c.mask().unwrap().has(b as u32), "{}", b as char);
            c.commit(b as i32).unwrap();
        }
        assert!(c.complete());
        // the call's name is already written: no other parameter opens
        let mut c = Constraint::begin(Arc::clone(&c_plan(&r, &p, &f)));
        for b in "<parameter=m".bytes() {
            if c.commit(b as i32).is_err() {
                return;
            }
        }
        panic!("an unknown parameter was admitted");
    }

    fn c_plan(r: &Request, p: &ChatPrompt, f: &OutputFormat) -> Arc<Plan> {
        use crate::grammar::vocab::Kind;
        let bytes: Vec<Vec<u8>> = (0u8..128).map(|b| vec![b]).collect();
        plan(r, p, f, Arc::new(TokenTable::new(bytes, vec![Kind::Text; 128]))).unwrap()
    }

    #[test]
    fn reasoning_is_offered_only_where_the_prompt_left_it_open() {
        let f = OutputFormat::default();
        let mut p = prompt(None);
        p.text = "<|im_start|>assistant\n".into();
        assert!(offers_reasoning(&p, &f));
        // reasoning turned off: the template closed an empty block
        p.text = "<|im_start|>assistant\n<think>\n\n</think>\n\n".into();
        assert!(!offers_reasoning(&p, &f));
        // the prompt opened it: the reply starts inside it (the verified reasoning phase)
        p.text = "<|im_start|>assistant\n<think>\n".into();
        p.reasoning_open = true;
        assert!(!offers_reasoning(&p, &f));
    }

    #[test]
    fn a_call_form_that_cannot_be_held_is_refused() {
        let tools = r#"[{"type":"function","function":{"name":"a","strict":true,"parameters":{"type":"object"}}}]"#;
        let p = prompt(Some(tools));
        let r = request(&parse_json("{}").unwrap(), &p).unwrap().unwrap();
        let f = OutputFormat {
            style: CallStyle::Call,
            ..OutputFormat::default()
        };
        let t = Arc::new(TokenTable::new(vec![b"a".to_vec()], vec![crate::grammar::vocab::Kind::Text]));
        assert!(plan(&r, &p, &f, t).unwrap_err().contains("can not be held"));
    }

    /// For replies the grammar accepts (random walks over adversarial pieces: tags inside
    /// strings, escapes, numbers at the exponent limit), what the reply parser extracts is valid:
    /// every call's arguments against its tool's schema, and an answer without calls against the
    /// answer's schema. Text is fed in random splits, as a stream would be.
    #[test]
    fn what_the_parser_extracts_from_an_accepted_reply_is_valid() {
        use super::super::outparse::OutputParser;
        use crate::grammar::tests::validates;
        let answer = r#"{"type":"object","properties":{"a":{"type":"string"},"k":{"type":"array","items":{"type":"number"},"maxItems":3},
            "m":{"type":"object","additionalProperties":{"type":"string"}}},"required":["a"],"additionalProperties":false}"#;
        let get = r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false}"#;
        let put = r#"{"type":"object","properties":{"o":{"type":"object","properties":{"body":{"type":"string"}},"additionalProperties":false},
            "s":{"type":"string"},"x":{"type":"number"}},"required":["o"],"additionalProperties":false}"#;
        let tools = format!(
            r#"[{{"type":"function","function":{{"name":"get","strict":true,"parameters":{get}}}}},
                {{"type":"function","function":{{"name":"put","strict":true,"parameters":{put}}}}}]"#
        );
        let p = prompt(Some(&tools));
        let body = parse_json(&format!(
            r#"{{"response_format":{{"type":"json_schema","json_schema":{{"name":"r","schema":{answer}}}}}}}"#
        ))
        .unwrap();
        let r = request(&body, &p).unwrap().unwrap();
        let f = OutputFormat::default();
        let plan = c_plan(&r, &p, &f);
        let g = &plan.grammar;
        let answer = parse_json(answer).unwrap();
        let schemas = [("get", parse_json(get).unwrap()), ("put", parse_json(put).unwrap())];
        let pieces: &[&str] = &[
            "{", "}", "[", "]", "\"", ",", ":", " ", "\n", "\\", "\\n", "\\u003c", "a", "b", "k", "m",
            "o", "n", "s", "x", "body", "0", "1", "7", "-", ".", "e", "9", "<", ">", "/", "_",
            "<tool_call>", "</tool_call>", "<tool_call>\n<function=get>\n", "<tool_call>\n<function=put>\n",
            "<function=get>", "<parameter=n>\n", "<parameter=o>\n", "<parameter=s>\n", "<parameter=x>\n",
            "\n</parameter>\n", "</parameter>", "</function>", "</function>\n</tool_call>", "<think>\n",
            "</think>\n\n", "tool_call", "parameter", "function", "\"a\"", "\"k\"", "\"m\"", "\"body\"",
            "true", "null", "1e99", "99",
        ];
        let closing: &[&str] = &[
            "\"", "}", "]", "\n</parameter>\n", "</function>\n</tool_call>", "</think>\n\n", "0", "\n",
        ];
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut rnd = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let (mut docs, mut with_calls) = (0, 0);
        for _ in 0..1500 {
            let mut cur = g.start().unwrap();
            let mut text = String::new();
            let done = loop {
                let late = text.len() > 60 + rnd(200) as usize;
                let pool = if late { closing } else { pieces };
                let ok: Vec<&str> = pool
                    .iter()
                    .copied()
                    .filter(|x| g.step_bytes(&cur, x.as_bytes()).is_some())
                    .collect();
                if g.complete(&cur) && (ok.is_empty() || rnd(6) == 0) {
                    break true;
                }
                if text.len() > 600 {
                    break false;
                }
                let pick = if ok.is_empty() {
                    // past the closing set: any admitted piece
                    match pieces.iter().find(|x| g.step_bytes(&cur, x.as_bytes()).is_some()) {
                        Some(x) => *x,
                        None => break false,
                    }
                } else {
                    ok[rnd(ok.len() as u64) as usize]
                };
                cur = g.step_bytes(&cur, pick.as_bytes()).unwrap();
                text.push_str(pick);
            };
            if !done {
                continue;
            }
            let mut parser = OutputParser::new(false, true, p.tools.clone(), Vec::new(), f.clone());
            let mut at = 0;
            while at < text.len() {
                let mut end = (at + 1 + rnd(12) as usize).min(text.len());
                while !text.is_char_boundary(end) {
                    end += 1;
                }
                parser.feed(&text[at..end]);
                at = end;
            }
            parser.finish();
            let calls = parser.tool_calls();
            if calls.is_empty() {
                let v = parse_json(parser.content().trim())
                    .unwrap_or_else(|e| panic!("accepted answer does not parse ({e}): {text:?}"));
                assert!(validates(&answer, &v, &answer), "answer off its schema: {text:?}");
                docs += 1;
            } else {
                assert!(parser.content().trim().is_empty(), "content beside calls: {text:?}");
                for c in calls {
                    let (_, s) = schemas
                        .iter()
                        .find(|(n, _)| *n == c.name)
                        .unwrap_or_else(|| panic!("call to '{}' extracted from {text:?}", c.name));
                    let args = parse_json(&c.arguments)
                        .unwrap_or_else(|e| panic!("arguments {:?} do not parse ({e}): {text:?}", c.arguments));
                    assert!(validates(s, &args, s), "arguments {} off the schema of '{}': {text:?}", c.arguments, c.name);
                }
                with_calls += 1;
            }
        }
        assert!(docs >= 30 && with_calls >= 30, "walks reached {docs} answers and {with_calls} call replies");
        // the review's reply, accepted before the fix
        let smuggled = r#"{"a":"<tool_call><function=get><parameter=n>not a number</parameter></function></tool_call>"}"#;
        assert!(g.start().and_then(|c| g.step_bytes(&c, smuggled.as_bytes())).is_none());
    }
}
