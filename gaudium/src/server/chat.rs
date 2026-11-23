//! Chat request -> prompt: the request's messages and tools become the context of the model's
//! own chat template (what the reference tokenizer library's `apply_chat_template` gets), which
//! is rendered to the prompt text.
//!
//! The tool-choice decisions (are the tools offered, is a call forced, may the model reason
//! first) come from the verified core (`core::tool_calls::ToolPolicy`).

use std::path::Path;

use super::jinja::{self, Template, Value};
use super::outparse::{CallStyle, OutputFormat};
use crate::core::effort::{Effort, EffortForm, EffortLevels, NumberEffort, OFF};
use crate::core::tool_calls::{ForcePlace, ToolChoice, ToolPolicy};
use crate::json::{parse_json, JsonValue};
use crate::vision::media::MediaRef;

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
    /// The levels of the effort scale the template accepts for `reasoning_effort`.
    pub effort: EffortLevels,
    /// How the template takes reasoning off and numeric efforts.
    pub effort_form: EffortForm,
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
        let mut t = Self {
            template,
            tokens,
            format,
            supports_tools: source.contains("tools"),
            effort: EffortLevels::NONE,
            effort_form: EffortForm::default(),
        };
        if source.contains("reasoning_effort") {
            t.effort = t.detect_effort();
            t.effort_form = t.detect_effort_form();
        }
        Ok((t, note))
    }

    /// A one-message chat rendered with reasoning on and `reasoning_effort` set to `level`.
    fn probe_effort(&self, level: &str) -> Result<String, jinja::Error> {
        self.probe(Some(Value::str(level)), true)
    }

    /// A one-message chat rendered with the reasoning flag `thinking` and `reasoning_effort`
    /// set to `effort` (`None`: not set).
    fn probe(&self, effort: Option<Value>, thinking: bool) -> Result<String, jinja::Error> {
        let mut ctx: Vec<(String, Value)> = vec![
            (
                "messages".into(),
                Value::from_json(&parse_json(r#"[{"role":"user","content":"Hi"}]"#).unwrap()),
            ),
            ("add_generation_prompt".into(), Value::Bool(true)),
        ];
        for (k, v) in &self.tokens {
            ctx.push((k.clone(), Value::str(v)));
        }
        ctx.push(("enable_thinking".into(), Value::Bool(thinking)));
        if let Some(e) = effort {
            ctx.push(("reasoning_effort".into(), e));
        }
        self.template.render(ctx)
    }

    /// The off level, numbers and the flag, found like the levels: a value is accepted when the
    /// template renders with it and tells it apart from a value it cannot know.
    fn detect_effort_form(&self) -> EffortForm {
        let unknown = self.probe_effort("gaudium-unknown-effort").ok();
        let accepted = |v: Value| match (self.probe(Some(v), true), &unknown) {
            (Ok(text), Some(u)) => text != *u,
            (Ok(_), None) => true,
            (Err(_), _) => false,
        };
        EffortForm {
            off_level: accepted(Value::str(OFF)),
            numbers: accepted(Value::Float(0.5)),
            reads_flag: self.probe(None, true).ok() != self.probe(None, false).ok(),
        }
    }

    /// The template renders with this level (it does not raise).
    pub fn effort_renders(&self, level: &str) -> bool {
        self.probe_effort(level).is_ok()
    }

    /// The levels the template accepts, found by rendering it with each level of the scale: a
    /// level is accepted when the render succeeds and the template tells it apart from a value
    /// it cannot know (that value raises, or renders differently). A level is never reported
    /// unless the template rendered with it.
    fn detect_effort(&self) -> EffortLevels {
        let unknown = self.probe_effort("gaudium-unknown-effort").ok();
        EffortLevels::from_fn(|level| match (self.probe_effort(level), &unknown) {
            (Ok(text), Some(u)) => text != *u,
            (Ok(_), None) => true,
            (Err(_), _) => false,
        })
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
    /// The request's images and videos in the order of their placeholders in `text`.
    pub media: Vec<MediaRef>,
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

/// The URL of a media reference: a string, or an object holding it as `url` or `image_url`.
fn media_url(v: &JsonValue) -> Option<&str> {
    match v {
        JsonValue::Str(s) => Some(s.as_str()),
        JsonValue::Object(_) => v
            .get("url")
            .and_then(JsonValue::as_str)
            .or_else(|| v.get("image_url").and_then(media_url)),
        _ => None,
    }
}

/// An image part's reference: `image_url` (a string or `{url}`), `image` or `url` (a string),
/// or a `source` block (`{type: "base64", media_type, data}` or `{type: "url", url}`).
fn image_ref(p: &JsonValue) -> Result<MediaRef, String> {
    if let Some(src) = p.get("source").filter(|s| s.as_object().is_some()) {
        return match src.get("type").and_then(JsonValue::as_str) {
            Some("base64") => {
                let data = src
                    .get("data")
                    .and_then(JsonValue::as_str)
                    .ok_or("an image source of type base64 needs a string 'data'")?;
                let mt = src
                    .get("media_type")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("application/octet-stream");
                Ok(MediaRef::Image(format!("data:{mt};base64,{data}")))
            }
            Some("url") => src
                .get("url")
                .and_then(JsonValue::as_str)
                .map(|u| MediaRef::Image(u.to_string()))
                .ok_or_else(|| "an image source of type url needs a string 'url'".to_string()),
            _ => Err("an image source must be of type base64 or url".into()),
        };
    }
    ["image_url", "image", "url"]
        .iter()
        .find_map(|k| p.get(k).and_then(media_url))
        .map(|u| MediaRef::Image(u.to_string()))
        .ok_or_else(|| "an image content part needs image_url.url (a data: URL)".to_string())
}

/// A video part's reference. A video is its frames (the server decodes no video container):
/// `{"frames": [<data URL> | {"url": ..} ..], "fps": F | "timestamps": [seconds ..]}` (a bare
/// list is the frames), or one animated image as a data URL (a string or `{url}`), under the
/// part's `video`, `input_video` or `video_url` key.
fn video_ref(p: &JsonValue) -> Result<MediaRef, String> {
    let v = ["video", "input_video", "video_url"]
        .iter()
        .find_map(|k| p.get(k))
        .ok_or("a video content part needs 'video' (its frames) or 'video_url' (an animated image as a data: URL)")?;
    let list = match v {
        JsonValue::Array(a) => Some(a.as_slice()),
        _ => v.get("frames").and_then(JsonValue::as_array),
    };
    let Some(list) = list else {
        return media_url(v)
            .map(|u| MediaRef::Animation(u.to_string()))
            .ok_or_else(|| {
                "a video must be a list of frames, {\"frames\": [...]}, or a data: URL of an animated image".to_string()
            });
    };
    if list.is_empty() {
        return Err("a video needs at least one frame".into());
    }
    let frames = list
        .iter()
        .map(|f| {
            media_url(f).map(str::to_string).ok_or_else(|| {
                "a frame must be a data: URL (a string, or an object with 'url')".to_string()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fps = match v.get("fps") {
        None | Some(JsonValue::Null) => None,
        Some(f) => Some(f.as_f64().ok_or("a video's fps must be a number")?),
    };
    let times = match v.get("timestamps") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Array(ts)) => Some(
            ts.iter()
                .map(|t| {
                    t.as_f64()
                        .ok_or("a video's timestamps must be numbers (seconds)")
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Some(_) => return Err("a video's timestamps must be a list of seconds".into()),
    };
    Ok(MediaRef::Frames { frames, fps, times })
}

/// A message's content parts as the template gets them. Text-only parts become one string (as
/// before image input). With an image or a video the parts stay a list, each media part reduced
/// to its type (the template writes its placeholder; the bytes stay out of the context) and
/// appended to `media` in order.
fn convert_parts(
    parts: &[JsonValue],
    media: &mut Vec<MediaRef>,
    keep_list: bool,
) -> Result<JsonValue, String> {
    let mut out: Vec<JsonValue> = Vec::with_capacity(parts.len());
    let mut text = String::new();
    let mut has_media = false;
    for p in parts {
        if p.as_object().is_none() {
            return Err("content parts must be objects".into());
        }
        let kind = p.get("type").and_then(JsonValue::as_str).unwrap_or("");
        match kind {
            "text" | "input_text" | "" if p.get("text").is_some() => {
                match p.get("text").and_then(JsonValue::as_str) {
                    Some(t) => {
                        text.push_str(t);
                        out.push(object(vec![
                            ("type", JsonValue::Str("text".into())),
                            ("text", JsonValue::Str(t.to_string())),
                        ]));
                    }
                    None => return Err("a text content part needs a string 'text'".into()),
                }
            }
            "image_url" | "image" | "input_image" => {
                media.push(image_ref(p)?);
                has_media = true;
                out.push(object(vec![("type", JsonValue::Str("image".into()))]));
            }
            "input_video" | "video" | "video_url" => {
                media.push(video_ref(p)?);
                has_media = true;
                out.push(object(vec![("type", JsonValue::Str("video".into()))]));
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
    Ok(if has_media || keep_list {
        JsonValue::Array(out)
    } else {
        JsonValue::Str(text)
    })
}

/// Whether any message of a request body holds an image or a video part.
pub fn has_media_parts(body: &JsonValue) -> bool {
    let media = |p: &JsonValue| {
        matches!(
            p.get("type").and_then(JsonValue::as_str),
            Some("image_url" | "image" | "input_image" | "input_video" | "video" | "video_url")
        )
    };
    body.get("messages")
        .and_then(JsonValue::as_array)
        .is_some_and(|msgs| {
            msgs.iter().any(|m| {
                m.get("content")
                    .and_then(JsonValue::as_array)
                    .is_some_and(|parts| parts.iter().any(media))
            })
        })
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

/// The request's messages as the template gets them, and their images and videos in order.
/// `keep_lists`: text-only content parts stay a list (a template that renders each part as a
/// block of its own), else they are joined into one string.
fn convert_messages(
    msgs: &JsonValue,
    keep_lists: bool,
) -> Result<(Vec<JsonValue>, Vec<MediaRef>), String> {
    let Some(list) = msgs.as_array() else {
        return Err("'messages' must be an array".into());
    };
    if list.is_empty() {
        return Err("'messages' must not be empty".into());
    }
    let mut out = Vec::with_capacity(list.len());
    let mut media = Vec::new();
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
                ("content", JsonValue::Array(parts)) => {
                    convert_parts(parts, &mut media, keep_lists)?
                }
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
    Ok((out, media))
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
    let continue_final = flag(body, "continue_final_message").unwrap_or(false);
    // a block template renders each text part as its own block (a continued final message is
    // cut after its text, which must then be one string)
    let keep_lists = t.format.style == CallStyle::Blocks && !continue_final;
    let (msgs, media) = convert_messages(raw_msgs, keep_lists)?;
    let mut ignored = Vec::new();

    let mut kwargs = match body.get("chat_template_kwargs") {
        None | Some(JsonValue::Null) => JsonValue::Object(Vec::new()),
        Some(k @ JsonValue::Object(_)) => k.clone(),
        Some(_) => return Err("chat_template_kwargs must be an object".into()),
    };
    // the request's effort turned reasoning off
    let mut effort_off = false;
    if let Some(n @ (JsonValue::Int(_) | JsonValue::Float(_))) = body.get("reasoning_effort") {
        if kwargs.get("reasoning_effort").is_none() {
            match t.effort_form.number(n.as_f64() == Some(0.0)) {
                NumberEffort::Pass { thinks } => {
                    effort_off = !thinks;
                    if kwargs.get("enable_thinking").is_none() {
                        set(&mut kwargs, "enable_thinking", JsonValue::Bool(thinks));
                    }
                    set(&mut kwargs, "reasoning_effort", n.clone());
                }
                NumberEffort::Refuse => return Err(format!(
                    "reasoning_effort must be a level name for this model (its levels: none{}{})",
                    if t.effort.supported() { ", " } else { "" },
                    t.effort.advertised().join(", ")
                )),
            }
        }
    }
    if let Some(effort) = body.get("reasoning_effort").and_then(JsonValue::as_str) {
        if kwargs.get("reasoning_effort").is_none() {
            // the verified plan: reasoning off, on at one of the template's levels, or on alone
            let plan = t.effort.plan(effort);
            effort_off = plan == Effort::Off;
            let (thinks, level) = match plan {
                Effort::Off => (false, None),
                Effort::Level(l) => (true, Some(l)),
                Effort::ThinkOnly => (true, None),
                Effort::Refuse => {
                    return Err(format!(
                        "reasoning_effort '{effort}' is not a level of this model (its levels: none, {})",
                        t.effort.advertised().join(", ")
                    ))
                }
            };
            if kwargs.get("enable_thinking").is_none() {
                set(&mut kwargs, "enable_thinking", JsonValue::Bool(thinks));
            }
            if let Some(level) = level {
                set(
                    &mut kwargs,
                    "reasoning_effort",
                    JsonValue::Str(level.into()),
                );
            }
        }
    }

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
    // reasoning off reaches a template that ignores the flag through its off level
    let off =
        effort_off || kwargs.get("enable_thinking").and_then(JsonValue::as_bool) == Some(false);
    if t.effort_form
        .off_by_level(off, kwargs.get("reasoning_effort").is_some())
    {
        set(&mut kwargs, "reasoning_effort", JsonValue::Str(OFF.into()));
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
        let final_content = msgs.last().and_then(|m| m.get("content"));
        if matches!(final_content, Some(JsonValue::Array(_))) {
            return Err(
                "continue_final_message: the final message must be text (it holds an image or a video)".into(),
            );
        }
        let final_text = final_content.and_then(JsonValue::as_str).unwrap_or("");
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
    let in_reasoning = reasoning_open(&text);
    match policy.placement(in_reasoning, t.format.can_force(&force_name)) {
        ForcePlace::FpPlace => {
            forced_prefix = t.format.forced_prefix(&force_name);
            text.push_str(&forced_prefix);
        }
        // the template opened the reasoning anyway: the call cannot be placed
        ForcePlace::FpIgnored if in_reasoning => {
            ignored.push("tool_choice with a template that always reasons (not forced)".into())
        }
        // the model writes a call's name first, so a call to any tool cannot be opened for it
        ForcePlace::FpIgnored => ignored.push(
            "tool_choice=required with a model whose calls open with their name (not forced; name the function to force it)".into(),
        ),
        ForcePlace::FpNone => {}
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
        media,
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

/// A small template with an effort setting: three levels, the highest by default, anything
/// else raises (the tests' own).
#[cfg(test)]
pub(super) const EFFORT_TEMPLATE: &str = r#"{%- set thinking = enable_thinking is undefined or enable_thinking %}
{%- if thinking %}
    {%- set level = reasoning_effort | default('xhigh') %}
    {%- if level not in ('low', 'medium', 'xhigh') %}
        {{- raise_exception('No effort ' ~ level ~ ' here.') }}
    {%- endif %}
    {%- if level != 'medium' %}
        {{- '<|im_start|>system\nThink at ' ~ level ~ '.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for m in messages %}
    {{- '<|im_start|>' ~ m.role ~ '\n' ~ m.content ~ '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' ~ ('<think>\n' if thinking else '<think>\n\n</think>\n\n') }}
{%- endif %}"#;

/// A small template in the message-block form: typed blocks, an effort block whose level is a
/// name or a number (0 = no reasoning), and the reasoning flag ignored (the tests' own).
#[cfg(test)]
pub(super) const BLOCK_TEMPLATE: &str = r#"{%- set levels = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}
{%- macro effort() -%}
{%- set e = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
{%- set n = levels[e | trim] if e is string else e | float -%}
{%- if n is undefined or n < 0.0 or n > 0.99 -%}{{ raise_exception("Unknown effort: " ~ e) }}{%- endif -%}
<|message_system|><|content_text|>Effort: {{ n }}<|end_message|>
{%- endmacro -%}
{%- if tools -%}<|message_system|>tool_declare<|content_xml|>{{ tools | tojson(sort_keys=true, separators=(",", ":")) }}<|end_message|>{%- endif -%}
{{- effort() -}}
{%- for m in messages -%}
{%- set r = "<|message_model|>" if m.role == "assistant" else "<|message_" ~ m.role ~ "|>" -%}
{%- if m.content is string and m.content -%}{{ r }}<|content_text|>{{ m.content }}<|end_message|>
{%- elif m.content is sequence -%}{%- for p in m.content -%}{{ r }}{{ "<|content_image|><|unused_200054|>" if p.type == "image" else "<|content_text|>" ~ p.text }}<|end_message|>{%- endfor -%}
{%- endif -%}
{%- for tc in m.tool_calls or [] -%}<|message_model|>{{ tc.function.name }}<|content_invoke_tool_json|>{"name":{{ tc.function.name | tojson }},"args":{{ tc.function.arguments | tojson(sort_keys=true, separators=(",", ":")) }}}<|end_message|>{%- endfor -%}
{%- if m.role == "assistant" -%}<|content_model_end_sampling|>{%- endif -%}
{%- endfor -%}"#;

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE: &str = TEST_TEMPLATE;

    fn block_prompt(t: &ChatTemplate, extra: &str) -> Result<ChatPrompt, String> {
        let body = format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"w?\"}}]{extra}}}");
        render_chat(t, &parse_json(&body).unwrap())
    }

    #[test]
    fn block_templates_take_efforts_names_numbers_and_reasoning_off() {
        let t = ChatTemplate::from_parts(BLOCK_TEMPLATE, None).unwrap().0;
        assert_eq!(t.format.style, CallStyle::Blocks);
        assert_eq!(
            t.effort.advertised(),
            ["minimal", "low", "medium", "high", "xhigh", "max"]
        );
        assert_eq!(
            t.effort_form,
            EffortForm {
                off_level: true,
                numbers: true,
                reads_flag: false
            }
        );
        let effort = |extra: &str| {
            let p = block_prompt(&t, extra).unwrap();
            assert!(!p.reasoning_open && p.forced_prefix.is_empty());
            let at = p.text.find("Effort: ").unwrap() + 8;
            p.text[at..p.text[at..].find('<').unwrap() + at].to_string()
        };
        assert_eq!(effort(""), "0.9");
        assert_eq!(effort(",\"reasoning_effort\":\"low\""), "0.2");
        assert_eq!(effort(",\"reasoning_effort\":\"max\""), "0.99");
        assert_eq!(effort(",\"reasoning_effort\":0.5"), "0.5");
        // reasoning off: the effort's own word, the flag (the Web UI), a zero effort
        assert_eq!(effort(",\"reasoning_effort\":\"none\""), "0.0");
        assert_eq!(
            effort(",\"reasoning_effort\":\"none\",\"chat_template_kwargs\":{\"enable_thinking\":true}"),
            "0.0"
        );
        assert_eq!(
            effort(",\"chat_template_kwargs\":{\"enable_thinking\":false}"),
            "0.0"
        );
        assert_eq!(effort(",\"reasoning_effort\":0"), "0.0");
        // an effort the request names wins over the off level
        assert_eq!(
            effort(",\"chat_template_kwargs\":{\"enable_thinking\":false,\"reasoning_effort\":\"low\"}"),
            "0.2"
        );
        assert_eq!(
            block_prompt(&t, ",\"reasoning_effort\":\"turbo\"").unwrap_err(),
            "reasoning_effort 'turbo' is not a level of this model (its levels: none, minimal, low, medium, high, xhigh, max)"
        );
        // the template's own refusal is the request's error
        assert_eq!(
            block_prompt(&t, ",\"reasoning_effort\":1.5").unwrap_err(),
            "chat template: Unknown effort: 1.5"
        );
        // a template that takes no numbers refuses one (it would be ignored otherwise)
        let named = ChatTemplate::from_parts(EFFORT_TEMPLATE, None).unwrap().0;
        assert_eq!(
            named.effort_form,
            EffortForm {
                reads_flag: true,
                ..EffortForm::default()
            }
        );
        assert!(block_prompt(&named, ",\"reasoning_effort\":0.5")
            .unwrap_err()
            .starts_with(
                "reasoning_effort must be a level name for this model (its levels: none, low"
            ));
        // the flag reaches a template that reads it, without the off level
        let p = block_prompt(
            &named,
            ",\"chat_template_kwargs\":{\"enable_thinking\":false}",
        )
        .unwrap();
        assert!(p.text.starts_with("<|im_start|>user") && !p.reasoning_open);
    }

    #[test]
    fn block_templates_keep_text_parts_and_force_named_calls() {
        let t = ChatTemplate::from_parts(BLOCK_TEMPLATE, None).unwrap().0;
        let body = parse_json(&format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"a"}},{{"type":"text","text":"b"}}]}},
                {{"role":"assistant","content":null,"tool_calls":[{{"id":"c1","type":"function","function":{{"name":"get_weather","arguments":"{{\"days\":2,\"city\":\"Oslo\"}}"}}}}]}},
                {{"role":"tool","tool_call_id":"c1","content":"sunny"}}],{TOOLS}}}"#
        ))
        .unwrap();
        let p = render_chat(&t, &body).unwrap();
        assert!(p.text.contains(
            "<|message_user|><|content_text|>a<|end_message|><|message_user|><|content_text|>b<|end_message|>\
             <|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\",\"days\":2}}<|end_message|><|content_model_end_sampling|>\
             <|message_tool|><|content_text|>sunny<|end_message|>"
        ), "{}", p.text);
        // a named call opens the reply; reasoning is off through the off level
        let p = block_prompt(
            &t,
            &format!(",{TOOLS},\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}"),
        )
        .unwrap();
        assert_eq!(
            p.forced_prefix,
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":"
        );
        assert!(p.text.ends_with(&p.forced_prefix) && p.text.contains("Effort: 0.0"));
        assert!(p.ignored.is_empty());
        // a call to any tool cannot be opened in this form: noted, not forced
        let p = block_prompt(&t, &format!(",{TOOLS},\"tool_choice\":\"required\"")).unwrap();
        assert!(p.forced_prefix.is_empty() && p.ignored.len() == 1);
        assert!(p.ignored[0].contains("name the function"));
    }

    /// Set `GAUDIUM_INKLING2_DIR` to an Inkling 2 model directory.
    #[test]
    #[ignore = "needs GAUDIUM_INKLING2_DIR with the model's chat template and tokenizer configuration"]
    fn inkling2_requests_render_through_the_model_template() {
        let dir = std::env::var("GAUDIUM_INKLING2_DIR").expect("GAUDIUM_INKLING2_DIR");
        let (t, note) = ChatTemplate::load(Path::new(&dir)).unwrap().unwrap();
        assert!(note.is_none() && t.supports_tools);
        assert_eq!(t.format.style, CallStyle::Blocks);
        assert_eq!(t.token("eos_token"), "<|content_model_end_sampling|>");
        assert_eq!(
            t.effort.advertised(),
            ["minimal", "low", "medium", "high", "xhigh", "max"]
        );
        assert_eq!(
            t.effort_form,
            EffortForm {
                off_level: true,
                numbers: true,
                reads_flag: false
            }
        );
        let level = |extra: &str| -> Result<String, String> {
            let p = block_prompt(&t, extra)?;
            let at = p.text.find("Thinking effort level: ").unwrap() + 23;
            Ok(p.text[at..p.text[at..].find('<').unwrap() + at].to_string())
        };
        for (extra, want) in [
            ("", "0.9"),
            (",\"reasoning_effort\":\"none\"", "0"),
            (",\"chat_template_kwargs\":{\"enable_thinking\":false}", "0"),
            (",\"reasoning_effort\":\"minimal\"", "0.1"),
            (",\"reasoning_effort\":\"medium\"", "0.7"),
            (",\"reasoning_effort\":\"xhigh\"", "0.99"),
            (",\"reasoning_effort\":0.5", "0.5"),
            (",\"reasoning_effort\":0", "0"),
        ] {
            assert_eq!(level(extra).unwrap(), want, "{extra}");
        }
        assert_eq!(
            level(",\"reasoning_effort\":1").unwrap_err(),
            "chat template: reasoning_effort must be in [0.0, 0.99]"
        );
        // the prompt ends after the last message: the model opens its own block
        let p = block_prompt(&t, "").unwrap();
        assert!(p
            .text
            .ends_with("<|message_user|><|content_text|>w?<|end_message|>"));
        // a client's tool calls (JSON text arguments) and results, the result named by its call
        let body = parse_json(&format!(
            r#"{{"messages":[{{"role":"user","content":"w?"}},
                {{"role":"assistant","content":"","reasoning_content":"r","tool_calls":[{{"id":"c1","type":"function","function":{{"name":"get_weather","arguments":"{{\"days\":2,\"city\":\"Oslo\"}}"}}}}]}},
                {{"role":"tool","tool_call_id":"c1","content":"sunny"}},
                {{"role":"user","content":[{{"type":"text","text":"and this?"}},{{"type":"image_url","image_url":{{"url":"data:x"}}}}]}}],{TOOLS}}}"#
        ))
        .unwrap();
        let p = render_chat(&t, &body).unwrap();
        assert!(p.text.contains(
            "<|message_model|><|content_thinking|>r<|end_message|>\
             <|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\",\"days\":2}}<|end_message|><|content_model_end_sampling|>\
             <|message_tool|>get_weather<|content_text|>sunny<|end_message|>\
             <|message_user|><|content_text|>and this?<|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|>"
        ), "{}", p.text);
        assert_eq!(p.media.len(), 1);
        // the template's refusals are the request's error
        let err = render_chat(
            &t,
            &parse_json(r#"{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{}}]}]}"#).unwrap(),
        )
        .unwrap_err();
        assert!(err.contains("audio"), "{err}");
        let err = render_chat(
            &t,
            &parse_json(r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"data:x"}}]}]}"#).unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            "chat template: Image content is only supported for user and tool messages"
        );
    }

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
                r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{}}]}]}"#,
                "needs image_url.url",
            ),
            (
                r#"{"messages":[{"role":"user","content":[{"type":"video","video":{"frames":["data:x"],"fps":"fast"}}]}]}"#,
                "fps must be a number",
            ),
            (
                r#"{"messages":[{"role":"user","content":[{"type":"video","video":{"frames":[7]}}]}]}"#,
                "a frame must be a data: URL",
            ),
            (
                r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":[{"type":"image","image":"data:x"}]}],"continue_final_message":true}"#,
                "final message must be text",
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
        // media parts: collected in order, each part reduced to its type for the template
        let body = parse_json(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"a"},
                {"type":"image_url","image_url":{"url":"data:1"}},
                {"type":"input_image","image_url":"data:2"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"QQ=="}},
                {"type":"video_url","video_url":{"url":"data:3"}},
                {"type":"input_video","input_video":["data:4",{"url":"data:5"},{"type":"image_url","image_url":{"url":"data:6"}}]},
                {"type":"video","video":{"frames":["data:7"],"timestamps":[0.5]}}]}]}"#,
        )
        .unwrap();
        let (msgs, media) = convert_messages(body.get("messages").unwrap(), false).unwrap();
        assert_eq!(
            msgs[0].get("content").unwrap().dump(),
            r#"[{"type":"text","text":"a"},{"type":"image"},{"type":"image"},{"type":"image"},{"type":"video"},{"type":"video"},{"type":"video"}]"#
        );
        assert_eq!(
            media,
            vec![
                MediaRef::Image("data:1".into()),
                MediaRef::Image("data:2".into()),
                MediaRef::Image("data:image/png;base64,QQ==".into()),
                MediaRef::Animation("data:3".into()),
                MediaRef::Frames {
                    frames: vec!["data:4".into(), "data:5".into(), "data:6".into()],
                    fps: None,
                    times: None
                },
                MediaRef::Frames {
                    frames: vec!["data:7".into()],
                    fps: None,
                    times: Some(vec![0.5])
                },
            ]
        );
        assert!(has_media_parts(&body));
        // text parts alone stay one string, as before image input
        let body = parse_json(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#,
        )
        .unwrap();
        let (msgs, media) = convert_messages(body.get("messages").unwrap(), false).unwrap();
        assert_eq!(msgs[0].get("content").unwrap().as_str(), Some("ab"));
        assert!(media.is_empty() && !has_media_parts(&body));

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
    fn effort_levels_come_from_the_template_and_requests_map_onto_them() {
        let t = ChatTemplate::from_parts(EFFORT_TEMPLATE, None).unwrap().0;
        assert_eq!(t.effort.advertised(), ["low", "medium", "xhigh"]);
        let prompt = |extra: &str| {
            let body =
                format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"w?\"}}]{extra}}}");
            render_chat(&t, &parse_json(&body).unwrap())
        };
        // no effort: the template's own default, as before
        assert!(prompt("")
            .unwrap()
            .text
            .starts_with("<|im_start|>system\nThink at xhigh."));
        for (asked, system) in [
            ("low", Some("low")),
            ("minimal", Some("low")),
            ("medium", None),
            ("high", Some("xhigh")),
            ("max", Some("xhigh")),
        ] {
            let p = prompt(&format!(",\"reasoning_effort\":\"{asked}\"")).unwrap();
            match system {
                Some(l) => assert!(
                    p.text
                        .starts_with(&format!("<|im_start|>system\nThink at {l}.")),
                    "{asked}"
                ),
                None => assert!(p.text.starts_with("<|im_start|>user"), "{asked}"),
            }
            assert!(p.reasoning_open, "{asked}");
        }
        let off = prompt(",\"reasoning_effort\":\"none\"").unwrap();
        assert!(off.text.starts_with("<|im_start|>user") && !off.reasoning_open);
        // a name off the scale is refused with the model's levels
        assert_eq!(
            prompt(",\"reasoning_effort\":\"turbo\"").unwrap_err(),
            "reasoning_effort 'turbo' is not a level of this model (its levels: none, low, medium, xhigh)"
        );
        // a level set in the template's own arguments reaches it unchanged
        assert!(
            prompt(",\"chat_template_kwargs\":{\"reasoning_effort\":\"high\"}")
                .unwrap_err()
                .contains("No effort high here.")
        );

        // a template that reads no effort has none, and an effort only turns reasoning on
        assert_eq!(template().effort, EffortLevels::NONE);
        assert!(
            render(",\"reasoning_effort\":\"turbo\"")
                .unwrap()
                .reasoning_open
        );
        // a template that reads the effort but never tells two values apart has no levels either
        let blind = ChatTemplate::from_parts(
            "{{ reasoning_effort | default('x') | length > 0 }}{{ messages[0].content }}",
            None,
        )
        .unwrap()
        .0;
        assert!(!blind.effort.supported());
        // one that accepts any value: the levels it renders differently from an unknown one
        let open = ChatTemplate::from_parts(
            "{% if reasoning_effort == 'high' %}H{% elif reasoning_effort == 'low' %}L{% endif %}{{ messages[0].content }}",
            None,
        )
        .unwrap()
        .0;
        assert_eq!(open.effort.advertised(), ["low", "high"]);
        assert!(open.effort_renders("max"));
    }

    /// Set `GAUDIUM_INKLING2_DIR` to an Inkling 2 model directory (its `chat_template.jinja`).
    #[test]
    #[ignore = "needs GAUDIUM_INKLING2_DIR with the model's chat_template.jinja"]
    fn inkling2_template_renders_like_jinja2() {
        let dir = std::env::var("GAUDIUM_INKLING2_DIR").expect("GAUDIUM_INKLING2_DIR");
        let source = std::fs::read_to_string(Path::new(&dir).join("chat_template.jinja")).unwrap();
        let fixture =
            parse_json(include_str!("../../tests/data/chat/inkling2_renders.json")).unwrap();
        let bad = jinja::fixture_mismatches(&source, fixture.get("cases").unwrap());
        assert!(bad.is_empty(), "{}", bad.join("\n"));
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
