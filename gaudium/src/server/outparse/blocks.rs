//! The message-block form: a reply written as typed blocks the model opens itself,
//! `<|message_model|>[NAME]<|content_TYPE|>...<|end_message|>`, ended by
//! `<|content_model_end_sampling|>` (or by another role's header, if the model writes one).
//!
//! The text is cut into events at the markers; where each piece goes and when a call is read are
//! decided by the verified block reader (`verified/message_blocks.bend`: `mb_next`, `mb_act`,
//! `mb_call_event`). A call block holds `{"name": NAME, "args": {...}}` and is read whole when
//! its block ends: its header and arguments go out together, so a partial call never streams.
//! A marker cut across two pieces of text is held back until it is complete.

use super::{partial_suffix, CallStyle, Delta, OutputFormat, OutputParser};
use crate::core::tool_calls::{release_len, CallEvent};
use crate::json::{parse_json, JsonValue};
use crate::verified::{mb_act, mb_call_event, mb_next, MbAct, MbEvent, MbState};

/// A model block's header.
pub(super) const HEADER: &str = "<|message_model|>";
/// The end of a block.
pub(super) const END: &str = "<|end_message|>";
/// The reasoning content type.
pub(super) const THINK: &str = "<|content_thinking|>";
/// The tool-call content type.
pub(super) const CALL: &str = "<|content_invoke_tool_json|>";
/// The end of the reply (the model's end-of-sequence token).
pub(super) const REPLY_END: &str = "<|content_model_end_sampling|>";
/// The text content type.
const TEXT: &str = "<|content_text|>";

/// What a marker is to the reader (`Call`: the event depends on whether tools are offered).
#[derive(Clone, Copy)]
enum Marker {
    Event(MbEvent),
    Call,
}

/// Every marker the reader knows. Text content types: text, a tool error, a call written as
/// text; the declaration and image types are not the model's to write, and read as text too.
const MARKERS: [(&str, Marker); 13] = [
    (HEADER, Marker::Event(MbEvent::MeHeader)),
    (THINK, Marker::Event(MbEvent::MeOpenThink)),
    (TEXT, Marker::Event(MbEvent::MeOpenText)),
    ("<|content_tool_error|>", Marker::Event(MbEvent::MeOpenText)),
    (
        "<|content_invoke_tool_text|>",
        Marker::Event(MbEvent::MeOpenText),
    ),
    ("<|content_xml|>", Marker::Event(MbEvent::MeOpenText)),
    ("<|content_image|>", Marker::Event(MbEvent::MeOpenText)),
    (CALL, Marker::Call),
    (END, Marker::Event(MbEvent::MeEnd)),
    (REPLY_END, Marker::Event(MbEvent::MeStop)),
    ("<|message_user|>", Marker::Event(MbEvent::MeStop)),
    ("<|message_system|>", Marker::Event(MbEvent::MeStop)),
    ("<|message_tool|>", Marker::Event(MbEvent::MeStop)),
];

/// The tokens that end a reply of this form: its end-of-sequence token and the headers of the
/// other roles (the model starting a turn that is not its own).
pub(super) const STOP_MARKERS: [&str; 4] = [
    REPLY_END,
    "<|message_user|>",
    "<|message_system|>",
    "<|message_tool|>",
];

/// A chat template of this form: it writes typed message blocks.
pub(super) fn is_block_template(chat_template: &str) -> bool {
    chat_template.contains(HEADER) && chat_template.contains(TEXT) && chat_template.contains(END)
}

/// The form's tags as the shared fields name them: the reasoning opening and the end that
/// closes it (a prompt ends inside reasoning when its last reasoning opening is not closed).
pub(super) fn block_format() -> OutputFormat {
    OutputFormat {
        think_open: THINK.into(),
        think_close: END.into(),
        tool_open: CALL.into(),
        tool_close: END.into(),
        style: CallStyle::Blocks,
        str_open: String::new(),
        str_close: String::new(),
    }
}

pub(super) fn is_marker(text: &str) -> bool {
    !text.is_empty() && MARKERS.iter().any(|(m, _)| m.contains(text))
}

/// The opening of a forced call to `name`, as the template writes a call.
pub(super) fn forced_call_prefix(name: &str) -> String {
    let quoted = crate::json::quote_json(name);
    format!("{HEADER}{name}{CALL}{{\"name\":{quoted},\"args\":")
}

impl OutputParser {
    /// The reader's state when the prompt ends inside a reasoning block, or between blocks.
    pub(super) fn block_start(reasoning_open: bool) -> MbState {
        if reasoning_open {
            MbState::MsThink
        } else {
            MbState::MsOutside
        }
    }

    pub(super) fn route_blocks(&mut self, text: &str, d: &mut Delta, last: bool) {
        self.buf.push_str(text);
        loop {
            if self.stopped {
                // the call limit ended the generation
                self.buf.clear();
                return;
            }
            let hit = MARKERS
                .iter()
                .filter_map(|(m, k)| self.buf.find(m).map(|p| (p, m.len(), *k)))
                .min_by_key(|(p, _, _)| *p);
            let Some((p, len, marker)) = hit else {
                let hold = if last {
                    0
                } else {
                    MARKERS
                        .iter()
                        .map(|(m, _)| partial_suffix(&self.buf, m))
                        .max()
                        .unwrap_or(0)
                };
                let cut = release_len(self.buf.len(), hold);
                if cut > 0 {
                    let words: String = self.buf.drain(..cut).collect();
                    self.block_event(MbEvent::MeWords, &words, d);
                }
                if last {
                    self.block_event(MbEvent::MeFinish, "", d);
                }
                return;
            };
            if p > 0 {
                let words: String = self.buf.drain(..p).collect();
                self.block_event(MbEvent::MeWords, &words, d);
            }
            self.buf.drain(..len);
            let event = match marker {
                Marker::Event(e) => e,
                Marker::Call => mb_call_event(self.tools_on),
            };
            self.block_event(event, "", d);
        }
    }

    fn block_event(&mut self, e: MbEvent, words: &str, d: &mut Delta) {
        let act = mb_act(self.mb, e, self.split);
        self.mb = mb_next(self.mb, e);
        match act {
            MbAct::MaNone => {}
            MbAct::MaReasoning => {
                d.reasoning.push_str(words);
                self.reasoning.push_str(words);
            }
            MbAct::MaContent => {
                d.content.push_str(words);
                self.content.push_str(words);
            }
            MbAct::MaName => self.mb_name.push_str(words),
            MbAct::MaCallText => self.mb_call.push_str(words),
            MbAct::MaReadCall => self.read_block_call(d, true),
            MbAct::MaCutCall => self.read_block_call(d, false),
        }
        // a new block's own text starts empty
        if e == MbEvent::MeHeader && self.mb == MbState::MsHeader {
            self.mb_name.clear();
        }
        if e == MbEvent::MeOpenCall && self.mb == MbState::MsCall {
            self.mb_call.clear();
            self.stream.apply(CallEvent::CeOpen);
        }
    }

    /// The call block's text: `{"name": NAME, "args": {...}}` (the name before the call marker
    /// stands in when the JSON has none). `ended`: its end marker arrived, so text that is not a
    /// call is given back as content; a block cut off without its end is dropped.
    fn read_block_call(&mut self, d: &mut Delta, ended: bool) {
        let raw = std::mem::take(&mut self.mb_call);
        let header_name = self.mb_name.trim().to_string();
        let call = parse_json(raw.trim()).ok().and_then(|j| {
            let name = match j.get("name") {
                Some(JsonValue::Str(n)) if !n.trim().is_empty() => n.trim().to_string(),
                None if !header_name.is_empty() => header_name,
                _ => return None,
            };
            let args = match j.get("args").or_else(|| j.get("arguments")) {
                None | Some(JsonValue::Null) => "{}".to_string(),
                Some(a @ JsonValue::Object(_)) => a.dump(),
                // arguments written as a JSON string of an object
                Some(JsonValue::Str(s)) => match parse_json(s) {
                    Ok(a @ JsonValue::Object(_)) => a.dump(),
                    _ => return None,
                },
                Some(_) => return None,
            };
            Some((name, args))
        });
        match call {
            Some((name, args)) => {
                self.cur.name = name;
                self.send_header(d);
                self.tool_args(d, &args);
                if self.push_call(d) {
                    self.stopped = true;
                }
            }
            None => {
                self.stream.apply(CallEvent::CeAbandon);
                if ended {
                    d.content.push_str(&raw);
                    self.content.push_str(&raw);
                }
            }
        }
    }
}
