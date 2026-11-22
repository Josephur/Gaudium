//! Chat request -> prompt: the request's messages and tools become the context of the model's
//! own chat template (what the reference tokenizer library's `apply_chat_template` gets), which
//! is rendered to the prompt text.
//!
//! The tool-choice decisions (are the tools offered, is a call forced, may the model reason
//! first) come from the verified core (`core::tool_calls::ToolPolicy`).

use std::path::Path;

use super::jinja::{self, Template, Value};
use super::outparse::OutputFormat;
use crate::core::tool_calls::{ToolChoice, ToolPolicy};
use crate::json::{parse_json, JsonValue};

/// A model's chat template with what rendering and parsing its turns needs.
#[derive(Debug)]
pub struct ChatTemplate {
    template: Template,
    /// The special-token strings the template may name (`bos_token`, `eos_token`, ...).
    tokens: Vec<(String, String)>,
    /// How the family writes its reasoning and its tool calls.
    pub format: OutputFormat,
    /// The template renders a tool list.
    pub supports_tools: bool,
}

impl ChatTemplate {
    /// Builds from the template text and the parsed `tokenizer_config.json` (when there is one).
    /// The second value is a note for the log (an output form that was not understood).
    pub fn from_parts(
        source: &str,
        tokenizer_config: Option<&JsonValue>,
    ) -> Result<(Self, Option<String>), String> {
        let template = Template::parse(source).map_err(|e| e.to_string())?;
        let mut tokens = Vec::new();
        if let Some(entries) = tokenizer_config.and_then(JsonValue::as_object) {
            for (k, v) in entries {
                if k.len() < 7 || !k.ends_with("_token") {
                    continue;
                }
                // a token is a string, or an object with its `content`
                if let Some(s) = v
                    .as_str()
                    .or_else(|| v.get("content").and_then(JsonValue::as_str))
                {
                    tokens.push((k.clone(), s.to_string()));
                }
            }
        }
        if tokens.is_empty() {
            // no tokenizer configuration: the names a ChatML template may use
            tokens.push(("eos_token".into(), "<|im_end|>".into()));
            tokens.push(("pad_token".into(), "<|endoftext|>".into()));
        }
        let (format, note) = OutputFormat::detect(tokenizer_config, source);
        Ok((
            Self {
                template,
                tokens,
                format,
                supports_tools: source.contains("tools"),
            },
            note,
        ))
    }

    /// Reads a model directory: `chat_template.jinja`, else the `chat_template` of its
    /// `tokenizer_config.json`. `Ok(None)`: the model has no template.
    pub fn load(dir: &Path) -> Result<Option<(Self, Option<String>)>, String> {
        let config = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|t| parse_json(&t).ok());
        let source = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(s) => Some(s),
            Err(_) => config.as_ref().and_then(|c| match c.get("chat_template") {
                Some(JsonValue::Str(s)) => Some(s.clone()),
                // a list of named templates: the default one
                Some(JsonValue::Array(list)) => list
                    .iter()
                    .find(|t| t.get("name").and_then(JsonValue::as_str) == Some("default"))
                    .and_then(|t| t.get("template"))
                    .and_then(JsonValue::as_str)
                    .map(str::to_string),
                _ => None,
            }),
        };
        match source {
            Some(s) => Self::from_parts(&s, config.as_ref()).map(Some),
            None => Ok(None),
        }
    }

    /// The text of a special token the template names (`bos_token`, `eos_token`).
    pub fn token(&self, name: &str) -> &str {
        self.tokens
            .iter()
            .find(|(k, _)| k == name)
            .map_or("", |(_, v)| v.as_str())
    }
}

/// A rendered chat request.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatPrompt {
    pub text: String,
    /// The prompt ends inside the family's reasoning opening (the model starts in its reasoning).
    pub reasoning_open: bool,
    /// The tool list the reply is parsed against; `None`: tool blocks are plain content.
    pub tools: Option<JsonValue>,
    /// The opening of a forced tool call, already at the end of `text`; fed to the output parser
    /// before the generated text.
    pub forced_prefix: String,
    /// `parallel_tool_calls` false: the reply ends after its first call.
    pub single_call: bool,
    /// The reply continues the final message.
    pub continue_final: bool,
    /// Request fields that were accepted but not applied.
    pub ignored: Vec<String>,
}

fn object(pairs: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn set(obj: &mut JsonValue, key: &str, value: JsonValue) {
    if let JsonValue::Object(entries) = obj {
        match entries.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => entries.push((key.to_string(), value)),
        }
    }
}

fn flag(body: &JsonValue, key: &str) -> Option<bool> {
    body.get(key).and_then(JsonValue::as_bool)
}

/// A message's content parts as one text. This server takes text only.
fn parts_text(parts: &[JsonValue]) -> Result<String, String> {
    let mut text = String::new();
    for p in parts {
        if p.as_object().is_none() {
            return Err("content parts must be objects".into());
        }
        let kind = p.get("type").and_then(JsonValue::as_str).unwrap_or("");
        match kind {
            "text" | "" if p.get("text").is_some() => {
                match p.get("text").and_then(JsonValue::as_str) {
                    Some(t) => text.push_str(t),
                    None => return Err("a text content part needs a string 'text'".into()),
                }
            }
            "image_url" | "image" | "input_video" | "video" | "video_url" => {
                return Err("this model was started without vision support".into())
            }
            "input_audio" | "audio" | "audio_url" => {
                return Err("audio input is not supported by this model".into())
            }
            "file" => {
                return Err("file content parts are not supported (send the file's text)".into())
            }
            other => return Err(format!("unknown content part type '{other}'")),
        }
    }
    Ok(text)
}

/// One tool call of an assistant message, its `arguments` as an object (clients send JSON text).
fn convert_tool_call(call: &JsonValue) -> Result<JsonValue, String> {
    let mut c = call.clone();
    let nested = c.get("function").is_some_and(|f| f.as_object().is_some());
    let func = if nested {
        c.get("function").unwrap()
    } else {
        &c
    };
    if func.get("name").and_then(JsonValue::as_str).is_none() {
        return Err("a tool call needs function.name".into());
    }
    if let Some(JsonValue::Str(text)) = func.get("arguments") {
        let parsed = if text.is_empty() {
            JsonValue::Object(Vec::new())
        } else {
            match parse_json(text) {
                Ok(v @ JsonValue::Object(_)) => v,
                Ok(_) => return Err("tool call arguments must be a JSON object".into()),
                Err(_) => {
                    let head: String = text.chars().take(200).collect();
                    return Err(format!("tool call arguments are not valid JSON: {head}"));
                }
            }
        };
        if nested {
            if let JsonValue::Object(entries) = &mut c {
                if let Some((_, f)) = entries.iter_mut().find(|(k, _)| k == "function") {
                    set(f, "arguments", parsed);
                }
            }
        } else {
            set(&mut c, "arguments", parsed);
        }
    }
    Ok(c)
}

/// The request's messages as the template gets them.
fn convert_messages(msgs: &JsonValue) -> Result<Vec<JsonValue>, String> {
    let Some(list) = msgs.as_array() else {
        return Err("'messages' must be an array".into());
    };
    if list.is_empty() {
        return Err("'messages' must not be empty".into());
    }
    let mut out = Vec::with_capacity(list.len());
    for m in list {
        let Some(entries) = m.as_object() else {
            return Err("each message must be an object".into());
        };
        if m.get("role").and_then(JsonValue::as_str).is_none() {
            return Err("each message needs a string 'role'".into());
        }
        let mut t: Vec<(String, JsonValue)> = Vec::with_capacity(entries.len());
        for (k, v) in entries {
            let v = match (k.as_str(), v) {
                ("role", JsonValue::Str(r)) if r == "developer" => JsonValue::Str("system".into()),
                ("content", JsonValue::Str(_) | JsonValue::Null) => v.clone(),
                ("content", JsonValue::Array(parts)) => JsonValue::Str(parts_text(parts)?),
                ("content", _) => {
                    return Err(
                        "message content must be a string, an array of parts, or null".into(),
                    )
                }
                ("tool_calls", JsonValue::Null) => continue,
                ("tool_calls", JsonValue::Array(calls)) => JsonValue::Array(
                    calls
                        .iter()
                        .map(convert_tool_call)
                        .collect::<Result<_, _>>()?,
                ),
                ("tool_calls", _) => return Err("tool_calls must be an array".into()),
                _ => v.clone(),
            };
            t.push((k.clone(), v));
        }
        out.push(JsonValue::Object(t));
    }
    Ok(out)
}

fn tool_name(tool: &JsonValue) -> Option<&str> {
    tool.get("function")
        .unwrap_or(tool)
        .get("name")
        .and_then(JsonValue::as_str)
}

/// `tool_choice` of the request: the choice and, for a named function, its name.
fn parse_tool_choice(body: &JsonValue) -> Result<(ToolChoice, String), String> {
    match body.get("tool_choice") {
        None | Some(JsonValue::Null) => Ok((ToolChoice::TcAuto, String::new())),
        Some(JsonValue::Str(s)) => match s.as_str() {
            "auto" => Ok((ToolChoice::TcAuto, String::new())),
            "none" => Ok((ToolChoice::TcNone, String::new())),
            "required" => Ok((ToolChoice::TcRequired, String::new())),
            _ => Err("tool_choice must be \"auto\", \"none\", \"required\" or {\"type\": \"function\", \"function\": {\"name\": ...}}".into()),
        },
        Some(tc @ JsonValue::Object(_)) => match tool_name(tc).filter(|n| !n.is_empty()) {
            Some(name) => Ok((ToolChoice::TcNamed, name.to_string())),
            None => Err("tool_choice: a named tool choice needs function.name".into()),
        },
        Some(_) => Err("tool_choice must be a string or an object".into()),
    }
}

/// Renders a chat-completions request body with the model's template.
pub fn render_chat(t: &ChatTemplate, body: &JsonValue) -> Result<ChatPrompt, String> {
    if body.as_object().is_none() {
        return Err("the request body must be a JSON object".into());
    }
    let Some(raw_msgs) = body.get("messages") else {
        return Err("'messages' is required".into());
    };
    let msgs = convert_messages(raw_msgs)?;
    let mut ignored = Vec::new();

    let mut kwargs = match body.get("chat_template_kwargs") {
        None | Some(JsonValue::Null) => JsonValue::Object(Vec::new()),
        Some(k @ JsonValue::Object(_)) => k.clone(),
        Some(_) => return Err("chat_template_kwargs must be an object".into()),
    };
    if let Some(effort) = body.get("reasoning_effort").and_then(JsonValue::as_str) {
        if kwargs.get("reasoning_effort").is_none() {
            // an effort asks for reasoning ("none": for none); templates know low / medium / xhigh
            let thinks = effort != "none";
            if kwargs.get("enable_thinking").is_none() {
                set(&mut kwargs, "enable_thinking", JsonValue::Bool(thinks));
            }
            if thinks {
                let level = match effort {
                    "high" | "max" => "xhigh",
                    "minimal" => "low",
                    other => other,
                };
                set(
                    &mut kwargs,
                    "reasoning_effort",
                    JsonValue::Str(level.into()),
                );
            }
        }
    }

    let continue_final = flag(body, "continue_final_message").unwrap_or(false);
    let add_generation_prompt = flag(body, "add_generation_prompt").unwrap_or(!continue_final);
    if continue_final && add_generation_prompt {
        return Err("continue_final_message and add_generation_prompt cannot both be true".into());
    }

    let tools = body
        .get("tools")
        .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    let (choice, force_name) = parse_tool_choice(body)?;
    let Some(policy) = ToolPolicy::decide(choice, tools.is_some(), continue_final) else {
        let word = if choice == ToolChoice::TcNamed {
            "function"
        } else {
            "required"
        };
        return Err(format!(
            "tool_choice '{word}' needs a non-empty 'tools' list"
        ));
    };
    if choice == ToolChoice::TcNamed {
        let known = tools
            .and_then(JsonValue::as_array)
            .is_some_and(|a| a.iter().any(|x| tool_name(x) == Some(force_name.as_str())));
        if !known {
            return Err(format!(
                "tool_choice names the function '{force_name}', which is not in 'tools'"
            ));
        }
        if continue_final {
            ignored.push("tool_choice=function with continue_final_message (not forced)".into());
        }
    } else if choice == ToolChoice::TcRequired && continue_final {
        ignored.push("tool_choice=required with continue_final_message (not forced)".into());
    }
    // a forced call opens the reply, so the model cannot reason first
    let wants_thinking = kwargs.get("enable_thinking").and_then(JsonValue::as_bool);
    if policy.forces && !policy.thinking(wants_thinking.unwrap_or(true)) {
        if wants_thinking == Some(true) {
            ignored
                .push("enable_thinking with a forced tool call (the call opens the reply)".into());
        }
        set(&mut kwargs, "enable_thinking", JsonValue::Bool(false));
    }

    let mut ctx: Vec<(String, Value)> = vec![(
        "messages".into(),
        Value::from_json(&JsonValue::Array(msgs.clone())),
    )];
    if let Some(tools) = tools.filter(|_| policy.offers) {
        ctx.push(("tools".into(), Value::from_json(tools)));
    }
    ctx.push((
        "add_generation_prompt".into(),
        Value::Bool(add_generation_prompt),
    ));
    for (k, v) in &t.tokens {
        ctx.push((k.clone(), Value::str(v)));
    }
    for (k, v) in kwargs.as_object().unwrap_or(&[]) {
        ctx.retain(|(name, _)| name != k);
        ctx.push((k.clone(), Value::from_json(v)));
    }
    let mut text = t.template.render(ctx).map_err(|e| match e {
        jinja::Error::Raised(m) => format!("chat template: {m}"),
        other => format!("chat template error: {other}"),
    })?;

    if continue_final {
        // cut the rendered text right after the final message's content
        let final_text = msgs
            .last()
            .and_then(|m| m.get("content"))
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        let ws = |c: char| c.is_ascii_whitespace();
        let stripped = final_text.trim_matches(ws);
        let Some(loc) = text.rfind(stripped) else {
            return Err(
                "continue_final_message: the final message does not appear in the rendered chat"
                    .into(),
            );
        };
        let lstripped = final_text.trim_start_matches(ws);
        let keep = if text[loc..].starts_with(lstripped) {
            lstripped.len()
        } else {
            stripped.len()
        };
        text.truncate(loc + keep);
    }

    let reasoning_open = |text: &str| match (
        text.rfind(&t.format.think_open),
        text.rfind(&t.format.think_close),
    ) {
        (Some(open), Some(close)) => close < open,
        (Some(_), None) => true,
        _ => false,
    };
    let mut forced_prefix = String::new();
    if policy.forces {
        if reasoning_open(&text) {
            // the template opened the reasoning anyway: the call cannot be placed
            ignored.push("tool_choice with a template that always reasons (not forced)".into());
        } else {
            forced_prefix = t.format.forced_prefix(&force_name);
            text.push_str(&forced_prefix);
        }
    }

    // the reply is parsed against the offered tools, plus tools known but not rendered
    let mut parse_tools = tools.filter(|_| policy.offers).cloned();
    if let Some(hidden) = body
        .get("gaudium_hidden_tools")
        .and_then(JsonValue::as_array)
        .filter(|a| !a.is_empty() && policy.offers)
    {
        let mut all = parse_tools
            .as_ref()
            .and_then(JsonValue::as_array)
            .map(<[JsonValue]>::to_vec)
            .unwrap_or_default();
        all.extend(hidden.iter().cloned());
        parse_tools = Some(JsonValue::Array(all));
    }

    Ok(ChatPrompt {
        reasoning_open: reasoning_open(&text),
        text,
        tools: parse_tools,
        forced_prefix,
        single_call: flag(body, "parallel_tool_calls") == Some(false),
        continue_final,
        ignored,
    })
}

/// An assistant message of a conversation, as a client sends it back (`tool_calls` with JSON
/// text arguments).
pub fn tool_calls_json(calls: &[super::outparse::ToolCall]) -> JsonValue {
    JsonValue::Array(
        calls
            .iter()
            .map(|c| {
                object(vec![
                    ("id", JsonValue::Str(c.id.clone())),
                    ("type", JsonValue::Str("function".into())),
                    (
                        "function",
                        object(vec![
                            ("name", JsonValue::Str(c.name.clone())),
                            ("arguments", JsonValue::Str(c.arguments.clone())),
                        ]),
                    ),
                ])
            })
            .collect(),
    )
}

/// A small ChatML-style template with tools, tool calls and tool results (the tests' own).
#[cfg(test)]
pub(super) const TEST_TEMPLATE: &str = r#"{%- if tools %}
    {{- '<|im_start|>system\n# Tools\n<tools>' }}
    {%- for tool in tools %}
        {{- '\n' ~ (tool | tojson) }}
    {%- endfor %}
    {{- '\n</tools><|im_end|>\n' }}
{%- endif %}
{%- for m in messages %}
    {%- if m.role == 'tool' %}
        {{- '<|im_start|>user\n<tool_response>\n' ~ m.content ~ '\n</tool_response><|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>' ~ m.role ~ '\n' ~ (m.content or '') }}
        {%- for tc in m.tool_calls or [] %}
            {{- '<tool_call>\n<function=' ~ tc.function.name ~ '>\n' }}
            {%- for k, v in tc.function.arguments | items %}
                {{- '<parameter=' ~ k ~ '>\n' ~ (v if v is string else v | tojson) ~ '\n</parameter>\n' }}
            {%- endfor %}
            {{- '</function>\n</tool_call>' }}
        {%- endfor %}
        {%- if not (loop.last and continue_final_message is defined) %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"#;

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE: &str = TEST_TEMPLATE;

    const TOOLS: &str = r#""tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#;

    fn template() -> ChatTemplate {
        ChatTemplate::from_parts(TEMPLATE, None).unwrap().0
    }

    fn render(extra: &str) -> Result<ChatPrompt, String> {
        let body = format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"w?\"}}]{extra}}}");
        render_chat(&template(), &parse_json(&body).unwrap())
    }

    #[test]
    fn messages_tools_and_history_render_through_the_template() {
        let t = template();
        assert!(t.supports_tools);
        assert_eq!(
            (t.token("eos_token"), t.token("bos_token")),
            ("<|im_end|>", "")
        );
        let body = parse_json(&format!(
            r#"{{"messages":[
                {{"role":"developer","content":"Be brief."}},
                {{"role":"user","content":[{{"type":"text","text":"Weather "}},{{"type":"text","text":"in Oslo?"}}]}},
                {{"role":"assistant","content":null,"tool_calls":[{{"id":"c1","type":"function","function":{{"name":"get_weather","arguments":"{{\"city\":\"Oslo\",\"n\":2}}"}}}}]}},
                {{"role":"tool","tool_call_id":"c1","content":"sunny"}}],{TOOLS}}}"#
        ))
        .unwrap();
        let p = render_chat(&t, &body).unwrap();
        assert_eq!(
            p.text,
            "<|im_start|>system\n# Tools\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}}}\n</tools><|im_end|>\n\
             <|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWeather in Oslo?<|im_end|>\n\
             <|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert!(
            p.reasoning_open && p.tools.is_some() && p.forced_prefix.is_empty() && !p.single_call
        );
    }

    #[test]
    fn tool_choice_follows_the_verified_policy() {
        // none hides the tools from the prompt and from the parser
        let p = render(&format!(",{TOOLS},\"tool_choice\":\"none\"")).unwrap();
        assert!(p.tools.is_none() && !p.text.contains("<tools>"));
        // a named call: reasoning off, the call's opening ends the prompt
        let p = render(&format!(
            ",{TOOLS},\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}},\"parallel_tool_calls\":false"
        ))
        .unwrap();
        assert_eq!(p.forced_prefix, "<tool_call>\n<function=get_weather>\n");
        assert!(p
            .text
            .ends_with("<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n"));
        assert!(!p.reasoning_open && p.single_call && p.ignored.is_empty());
        // required: the same, without a name; asking for reasoning as well is noted
        let p = render(&format!(
            ",{TOOLS},\"tool_choice\":\"required\",\"chat_template_kwargs\":{{\"enable_thinking\":true}}"
        ))
        .unwrap();
        assert!(p.text.ends_with("</think>\n\n<tool_call>\n"));
        assert_eq!(p.ignored.len(), 1);
        // refused requests
        assert!(render(",\"tool_choice\":\"required\"")
            .unwrap_err()
            .contains("non-empty 'tools'"));
        assert!(render(&format!(",{TOOLS},\"tool_choice\":\"sometimes\"")).is_err());
        assert!(render(&format!(
            ",{TOOLS},\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"nope\"}}}}"
        ))
        .unwrap_err()
        .contains("not in 'tools'"));
        // tools the parser knows but the prompt does not show
        let p = render(&format!(",{TOOLS},\"gaudium_hidden_tools\":[{{\"type\":\"function\",\"function\":{{\"name\":\"later\"}}}}]")).unwrap();
        assert_eq!(p.tools.unwrap().as_array().unwrap().len(), 2);
        assert!(!p.text.contains("later"));
    }

    #[test]
    fn thinking_effort_continuation_and_errors() {
        let off = render(",\"chat_template_kwargs\":{\"enable_thinking\":false}").unwrap();
        assert!(off.text.ends_with("<think>\n\n</think>\n\n") && !off.reasoning_open);
        assert!(
            !render(",\"reasoning_effort\":\"none\"")
                .unwrap()
                .reasoning_open
        );
        assert!(
            render(",\"reasoning_effort\":\"high\"")
                .unwrap()
                .reasoning_open
        );

        let body = parse_json(
            r#"{"messages":[{"role":"user","content":"Count."},{"role":"assistant","content":" One, two, "}],
                "continue_final_message":true,"chat_template_kwargs":{"continue_final_message":true}}"#,
        )
        .unwrap();
        let p = render_chat(&template(), &body).unwrap();
        assert!(p.continue_final && p.text.ends_with("<|im_start|>assistant\n One, two, "));

        for (body, want) in [
            (r#"{"messages":[]}"#, "must not be empty"),
            (r#"{"messages":[{"content":"x"}]}"#, "string 'role'"),
            (
                r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:x"}}]}]}"#,
                "vision",
            ),
            (
                r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"{bad"}}]}]}"#,
                "not valid JSON",
            ),
            (
                r#"{"messages":[{"role":"user","content":"x"}],"continue_final_message":true,"add_generation_prompt":true}"#,
                "cannot both",
            ),
        ] {
            let err = render_chat(&template(), &parse_json(body).unwrap()).unwrap_err();
            assert!(err.contains(want), "{body}: {err}");
        }
        let raising = ChatTemplate::from_parts("{{ raise_exception('No user query.') }}", None)
            .unwrap()
            .0;
        assert_eq!(
            render_chat(
                &raising,
                &parse_json(r#"{"messages":[{"role":"user","content":"x"}]}"#).unwrap()
            )
            .unwrap_err(),
            "chat template: No user query."
        );
    }

    #[test]
    fn template_tokens_and_the_output_form_come_from_the_tokenizer_config() {
        let cfg = parse_json(
            r#"{"bos_token": "<bos>", "eos_token": {"content": "<eos>"}, "model_max_length": 5, "add_bos_token": null,
                "response_template": {"fields": {"thinking": {"open": "<|channel>thought\n", "close": "<channel|>"}}}}"#,
        )
        .unwrap();
        let (t, note) = ChatTemplate::from_parts(
            "{{ bos_token }}{{ messages[0].content }}{{ eos_token }}",
            Some(&cfg),
        )
        .unwrap();
        assert!(note.is_none() && !t.supports_tools);
        assert_eq!(t.format.think_close, "<channel|>");
        let p = render_chat(
            &t,
            &parse_json(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(p.text, "<bos>hi<eos>");
        assert!(ChatTemplate::from_parts("{% if %}", None).is_err());
    }
}
