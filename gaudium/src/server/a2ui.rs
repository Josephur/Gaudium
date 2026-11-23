//! A2UI (agent-to-user-interface, protocol version 1.0) as a per-request option of the chat
//! endpoints.
//!
//! A client that can draw A2UI surfaces declares its component catalogs in the request field
//! `gaudium_a2ui`. When the server offers A2UI and shares a catalog with the client, the
//! instructions that teach the model the format are added to the conversation's system turn; the
//! model then writes `<a2ui-json>` blocks inside its ordinary content, which the client draws.
//! A client that declares nothing gets the conversation exactly as it sent it.
//!
//! The decisions (does the request get the instructions, is it refused, where do the
//! instructions go, how many catalog identifiers are compared) come from the verified core
//! (`core::a2ui`). This module reads the declaration and edits the message list.

use crate::core::a2ui::{
    ids_scanned, message_count, place, prompt_size_ok, A2uiPlace, A2uiPolicy, PromptPick,
};
use crate::json::{quote_json, JsonValue};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

/// The request field that carries the client's renderer capabilities.
pub const REQUEST_FIELD: &str = "gaudium_a2ui";
/// The protocol version the instructions teach.
pub const VERSION: &str = "1.0";
/// The one catalog the instructions describe (the protocol's basic catalog).
pub const BASIC_CATALOG_ID: &str =
    "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json";
/// Catalog identifiers of a request that are compared (the rest are not looked at).
pub const MAX_CATALOG_IDS: usize = 64;
/// What the model is told (added to the system turn) when no instructions file is given, or
/// while the given one has never passed its checks.
pub const INSTRUCTIONS: &str = include_str!("a2ui_prompt.md");
/// Largest instructions file that is read, in bytes (a larger one is refused).
pub const MAX_PROMPT_BYTES: u64 = 256 * 1024;

// ------------------------------------------------------------------------------------------------
// The instructions file (`--a2ui-prompt`): read at start and again whenever it changes on disk.
// Every request looks at the file's stamp (one `stat`); a changed file is read and checked, and
// whether it is served comes from the verified core (`PromptPick`): text that fails a check is
// never served, the text served before stays, and the reason is logged once per change of the
// file (not per request). An editor that saves by truncating and rewriting, or by renaming a new
// file over the old one, may be caught half-way: the half-written text either fails a check or
// is replaced at the next request, because the finished file has a new stamp.

/// What identifies one version of the file on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stamp {
    Gone,
    NotAFile,
    File {
        mtime: Option<SystemTime>,
        len: u64,
        inode: u64,
    },
}

fn stamp_of(path: &Path) -> Stamp {
    use std::os::unix::fs::MetadataExt;
    match std::fs::metadata(path) {
        Err(_) => Stamp::Gone,
        Ok(m) if !m.is_file() => Stamp::NotAFile,
        Ok(m) => Stamp::File {
            mtime: m.modified().ok(),
            len: m.len(),
            inode: m.ino(),
        },
    }
}

/// The file's text, or why it cannot be served.
fn read_checked(path: &Path, stamp: Stamp) -> Result<String, String> {
    let len = match stamp {
        Stamp::Gone => return Err("the file is missing or cannot be looked at".into()),
        Stamp::NotAFile => return Err("it is not a regular file".into()),
        Stamp::File { len, .. } => len,
    };
    let size_err = |n: u64| {
        if n == 0 {
            "the file is empty".to_string()
        } else {
            format!("the file is {n} bytes, more than the {MAX_PROMPT_BYTES} allowed")
        }
    };
    if !prompt_size_ok(len, MAX_PROMPT_BYTES) {
        return Err(size_err(len));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("it could not be read ({e})"))?;
    if !prompt_size_ok(bytes.len() as u64, MAX_PROMPT_BYTES) {
        return Err(size_err(bytes.len() as u64));
    }
    let text = String::from_utf8(bytes).map_err(|e| {
        format!("it is not UTF-8 text (bad byte at offset {})", e.utf8_error().valid_up_to())
    })?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    if text.contains('\0') {
        return Err("it holds NUL bytes (not a text file)".into());
    }
    if text.trim().is_empty() {
        return Err("it holds only whitespace".into());
    }
    if !text.contains("<a2ui-json>") || !text.contains("</a2ui-json>") {
        return Err(
            "it never names the <a2ui-json> and </a2ui-json> tags, the only blocks the Web UI draws"
                .into(),
        );
    }
    Ok(text.to_string())
}

struct Served {
    /// The stamp of the file when it was last looked at (`None`: never looked at).
    stamp: Option<Stamp>,
    text: Arc<str>,
    from_file: bool,
    /// Why the file on disk is not what is served (cleared when a good file loads).
    error: Option<String>,
    loads: u64,
}

/// Where the instructions come from: built in, or a file that is re-read when it changes.
pub struct PromptSource {
    path: Option<PathBuf>,
    served: Mutex<Served>,
}

impl PromptSource {
    pub fn built_in() -> Self {
        Self {
            path: None,
            served: Mutex::new(Served {
                stamp: None,
                text: Arc::from(INSTRUCTIONS),
                from_file: false,
                error: None,
                loads: 0,
            }),
        }
    }

    /// Reads `path` now (a file that fails its checks leaves the built-in text in place, and is
    /// read again once it changes).
    pub fn file(path: PathBuf) -> Self {
        let s = Self {
            path: Some(path),
            ..Self::built_in()
        };
        s.text();
        s
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Served> {
        self.served.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The instructions to serve now (looks at the file first, if there is one).
    pub fn text(&self) -> Arc<str> {
        let Some(path) = &self.path else {
            return self.lock().text.clone();
        };
        let stamp = stamp_of(path);
        let mut st = self.lock();
        let changed = st.stamp != Some(stamp);
        let read = if changed { Some(read_checked(path, stamp)) } else { None };
        let pick = PromptPick::decide(changed, matches!(read, Some(Ok(_))));
        st.stamp = Some(stamp);
        match read {
            Some(Ok(text)) if pick.loads() => {
                st.loads += 1;
                eprintln!(
                    "[server] a2ui instructions {} from {} ({} bytes)",
                    if st.loads == 1 { "read" } else { "reloaded" },
                    path.display(),
                    text.len()
                );
                st.text = Arc::from(text);
                st.from_file = true;
                st.error = None;
            }
            Some(Err(e)) if pick.warns() => {
                eprintln!(
                    "[server] a2ui instructions file {} not used: {e}; still serving {}",
                    path.display(),
                    if st.from_file { "the last good version of the file" } else { "the built-in instructions" }
                );
                st.error = Some(e);
            }
            _ => {}
        }
        st.text.clone()
    }

    /// `{"file", "serving", "loads", "error"}` for `GET /gaudium/a2ui`.
    pub fn status_json(&self) -> String {
        self.text();
        let st = self.lock();
        format!(
            "{{\"file\":{},\"serving\":\"{}\",\"loads\":{},\"error\":{}}}",
            self.path.as_ref().map_or("null".into(), |p| quote_json(&p.display().to_string())),
            if st.from_file { "file" } else { "built-in" },
            st.loads,
            st.error.as_deref().map_or("null".into(), quote_json),
        )
    }
}

static SOURCE: OnceLock<PromptSource> = OnceLock::new();

/// Serve the instructions from `path` from now on, re-read whenever it changes (call once, at
/// start; later calls are ignored).
pub fn use_prompt_file(path: PathBuf) {
    if SOURCE.get().is_none() {
        let _ = SOURCE.set(PromptSource::file(path));
    }
}

fn source() -> &'static PromptSource {
    SOURCE.get_or_init(PromptSource::built_in)
}

/// The instructions to add to a request now.
pub fn instructions() -> Arc<str> {
    source().text()
}

/// A request after its A2UI declaration was applied.
#[derive(Debug, Clone, PartialEq)]
pub enum Applied {
    /// The request runs as it came; `note`: its declaration was accepted but not applied.
    Unchanged { note: Option<String> },
    /// The request with the instructions in its system turn.
    Injected(JsonValue),
}

/// The client's catalog identifiers, or `None` when the request does not ask for A2UI.
/// Accepted forms: `true` (the basic catalog), `{"supportedCatalogIds": [...]}`, and the
/// protocol's capability object `{"1.0": {"supportedCatalogIds": [...]}}`.
fn declared_catalogs(body: &JsonValue) -> Result<Option<Vec<String>>, String> {
    let bad = || {
        format!(
            "{REQUEST_FIELD} must be true or an object with 'supportedCatalogIds' (a list of catalog id strings)"
        )
    };
    let field = match body.get(REQUEST_FIELD) {
        None | Some(JsonValue::Null | JsonValue::Bool(false)) => return Ok(None),
        Some(JsonValue::Bool(true)) => return Ok(Some(vec![BASIC_CATALOG_ID.to_string()])),
        Some(f @ JsonValue::Object(_)) => f,
        Some(_) => return Err(bad()),
    };
    let caps = match field.get(VERSION) {
        Some(v @ JsonValue::Object(_)) => v,
        Some(_) => return Err(bad()),
        None => field,
    };
    let Some(ids) = caps.get("supportedCatalogIds").and_then(JsonValue::as_array) else {
        return Err(bad());
    };
    let mut out = Vec::new();
    for id in &ids[..ids_scanned(ids.len(), MAX_CATALOG_IDS)] {
        match id.as_str() {
            Some(s) => out.push(s.to_string()),
            None => return Err(bad()),
        }
    }
    Ok(Some(out))
}

fn is_system(msg: &JsonValue) -> bool {
    matches!(
        msg.get("role").and_then(JsonValue::as_str),
        Some("system" | "developer")
    )
}

/// `content` of a system message with the instructions `ins` after it.
fn appended(content: Option<&JsonValue>, ins: &str) -> JsonValue {
    match content {
        Some(JsonValue::Str(s)) if !s.trim().is_empty() => JsonValue::Str(format!("{s}\n\n{ins}")),
        Some(JsonValue::Array(parts)) => {
            let mut parts = parts.clone();
            parts.push(JsonValue::Object(vec![
                ("type".into(), JsonValue::Str("text".into())),
                ("text".into(), JsonValue::Str(format!("\n\n{ins}"))),
            ]));
            JsonValue::Array(parts)
        }
        _ => JsonValue::Str(ins.to_string()),
    }
}

/// Applies the request's A2UI declaration. `Err`: the request is refused with this message.
pub fn apply(enabled: bool, body: &JsonValue) -> Result<Applied, String> {
    let declared = declared_catalogs(body)?;
    let common = declared
        .as_ref()
        .is_some_and(|ids| ids.iter().any(|id| id == BASIC_CATALOG_ID));
    let policy = A2uiPolicy::decide(declared.is_some(), enabled, common);
    if policy.refuses() {
        return Err(format!(
            "{REQUEST_FIELD}: this server was started without --a2ui"
        ));
    }
    if policy.notes() {
        return Ok(Applied::Unchanged {
            note: Some(format!(
                "{REQUEST_FIELD} (no catalog in common; this server writes for {BASIC_CATALOG_ID})"
            )),
        });
    }
    if !policy.injects() {
        return Ok(Applied::Unchanged { note: None });
    }
    // a request without a usable message list is left to the renderer, which reports it
    let Some(msgs) = body.get("messages").and_then(JsonValue::as_array) else {
        return Ok(Applied::Unchanged { note: None });
    };
    let ins = instructions();
    let spot = place(msgs.first().is_some_and(is_system));
    let mut out = Vec::with_capacity(message_count(msgs.len(), spot));
    match spot {
        A2uiPlace::A2Append => {
            let mut first = msgs[0].clone();
            let content = appended(first.get("content"), &ins);
            if let JsonValue::Object(entries) = &mut first {
                match entries.iter_mut().find(|(k, _)| k == "content") {
                    Some(slot) => slot.1 = content,
                    None => entries.push(("content".into(), content)),
                }
            }
            out.push(first);
            out.extend(msgs[1..].iter().cloned());
        }
        A2uiPlace::A2Insert => {
            out.push(JsonValue::Object(vec![
                ("role".into(), JsonValue::Str("system".into())),
                ("content".into(), JsonValue::Str(ins.to_string())),
            ]));
            out.extend(msgs.iter().cloned());
        }
    }
    debug_assert_eq!(out.len(), message_count(msgs.len(), spot));
    let mut new_body = body.clone();
    if let JsonValue::Object(entries) = &mut new_body {
        if let Some(slot) = entries.iter_mut().find(|(k, _)| k == "messages") {
            slot.1 = JsonValue::Array(out);
        }
    }
    Ok(Applied::Injected(new_body))
}

/// What the server offers (the protocol's agent capabilities, plus how to ask for them here), as
/// the `a2ui` object of `/props` and the body of `GET /gaudium/a2ui`.
pub fn capabilities_json(enabled: bool, with_instructions: bool) -> String {
    let mut out = format!(
        "{{\"enabled\":{enabled},\"version\":{},\"request_field\":{},\"block_tag\":\"a2ui-json\",\
         \"supportedCatalogIds\":[{}],\"acceptsInlineCatalogs\":false",
        quote_json(VERSION),
        quote_json(REQUEST_FIELD),
        quote_json(BASIC_CATALOG_ID),
    );
    if with_instructions {
        out.push_str(",\"instructions\":");
        out.push_str(&quote_json(&instructions()));
        out.push_str(",\"instructions_source\":");
        out.push_str(&source().status_json());
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    #[test]
    fn the_instructions_file_is_reloaded_and_a_bad_file_never_served() {
        let dir = std::env::temp_dir().join(format!("gaudium-a2ui-prompt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prompt.md");
        let good = |n: u32| format!("Version {n}.\n<a2ui-json>\n[]\n</a2ui-json>\n");
        let write = |body: &[u8]| {
            // a new file renamed over the old one: a new stamp even within one clock tick
            let tmp = dir.join("prompt.md.tmp");
            std::fs::write(&tmp, body).unwrap();
            std::fs::rename(&tmp, &path).unwrap();
        };
        let status = |s: &PromptSource| parse_json(&s.status_json()).unwrap();

        // missing at start: the built-in text, with the reason
        let src = PromptSource::file(path.clone());
        assert_eq!(&*src.text(), INSTRUCTIONS);
        let st = status(&src);
        assert_eq!(st.get("serving").unwrap().as_str(), Some("built-in"));
        assert!(st.get("error").unwrap().as_str().unwrap().contains("missing"));

        // appears: served; changes: served again
        write(good(1).as_bytes());
        assert_eq!(&*src.text(), good(1));
        write(good(2).as_bytes());
        assert_eq!(&*src.text(), good(2));
        assert_eq!(status(&src).get("loads").unwrap().as_u64(), Some(2));

        // every kind of bad file keeps version 2 and says why
        let big = vec![b'a'; MAX_PROMPT_BYTES as usize + 1];
        let cases: [(&[u8], &str); 6] = [
            (b"", "empty"),
            (b"  \n\t\n", "whitespace"),
            (b"\xff\xfe<a2ui-json></a2ui-json>", "UTF-8"),
            (b"<a2ui-json>\0</a2ui-json>", "NUL"),
            (b"Just some prose with no tags.", "tags"),
            (&big, "more than"),
        ];
        for (body, why) in cases {
            write(body);
            assert_eq!(&*src.text(), good(2), "{why}");
            let st = status(&src);
            assert_eq!(st.get("serving").unwrap().as_str(), Some("file"));
            assert!(st.get("error").unwrap().as_str().unwrap().contains(why), "{why}: {st:?}");
        }
        // a directory where the file was, then the file gone
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert_eq!(&*src.text(), good(2));
        assert!(status(&src).get("error").unwrap().as_str().unwrap().contains("regular file"));
        std::fs::remove_dir(&path).unwrap();
        assert_eq!(&*src.text(), good(2));

        // fixed: served, error cleared; a byte-order mark is dropped
        write(format!("\u{feff}{}", good(3)).as_bytes());
        assert_eq!(&*src.text(), good(3));
        assert!(matches!(status(&src).get("error"), Some(JsonValue::Null)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn body(extra: &str, msgs: &str) -> JsonValue {
        parse_json(&format!("{{\"messages\":{msgs}{extra}}}")).unwrap()
    }

    const USER: &str = r#"[{"role":"user","content":"hi"}]"#;
    const BASIC: &str = r#","gaudium_a2ui":{"supportedCatalogIds":["https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"]}"#;

    fn messages(a: &Applied) -> &[JsonValue] {
        match a {
            Applied::Injected(b) => b.get("messages").unwrap().as_array().unwrap(),
            Applied::Unchanged { .. } => panic!("not injected"),
        }
    }

    #[test]
    fn a_request_that_declares_nothing_is_untouched() {
        for extra in ["", ",\"gaudium_a2ui\":null", ",\"gaudium_a2ui\":false"] {
            for enabled in [true, false] {
                assert_eq!(
                    apply(enabled, &body(extra, USER)).unwrap(),
                    Applied::Unchanged { note: None }
                );
            }
        }
    }

    #[test]
    fn the_instructions_become_or_join_the_system_turn() {
        // no system message: one is put first
        let a = apply(true, &body(BASIC, USER)).unwrap();
        let m = messages(&a);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].get("role").unwrap().as_str(), Some("system"));
        assert_eq!(m[0].get("content").unwrap().as_str(), Some(INSTRUCTIONS));
        assert_eq!(m[1].get("content").unwrap().as_str(), Some("hi"));

        // a leading system message keeps its text, the instructions follow it
        let sys = r#"[{"role":"system","content":"Be brief."},{"role":"user","content":"hi"}]"#;
        let a = apply(true, &body(BASIC, sys)).unwrap();
        let m = messages(&a);
        assert_eq!(m.len(), 2);
        let text = m[0].get("content").unwrap().as_str().unwrap();
        assert!(text.starts_with("Be brief.\n\n# Interactive interfaces") && text.ends_with(INSTRUCTIONS));

        // content parts get one more text part; an empty system message is replaced
        let parts = r#"[{"role":"developer","content":[{"type":"text","text":"Be brief."}]},{"role":"user","content":"hi"}]"#;
        let a = apply(true, &body(BASIC, parts)).unwrap();
        let c = messages(&a)[0].get("content").unwrap().as_array().unwrap();
        assert_eq!(c.len(), 2);
        assert!(c[1].get("text").unwrap().as_str().unwrap().ends_with(INSTRUCTIONS));
        let empty = r#"[{"role":"system","content":""},{"role":"user","content":"hi"}]"#;
        let a = apply(true, &body(BASIC, empty)).unwrap();
        assert_eq!(messages(&a)[0].get("content").unwrap().as_str(), Some(INSTRUCTIONS));
    }

    #[test]
    fn every_accepted_form_of_the_declaration_is_read() {
        for extra in [
            ",\"gaudium_a2ui\":true".to_string(),
            BASIC.to_string(),
            format!(",\"gaudium_a2ui\":{{\"1.0\":{{\"supportedCatalogIds\":[\"other\",\"{BASIC_CATALOG_ID}\"]}}}}"),
        ] {
            assert!(matches!(
                apply(true, &body(&extra, USER)).unwrap(),
                Applied::Injected(_)
            ));
        }
        for extra in [
            ",\"gaudium_a2ui\":7",
            ",\"gaudium_a2ui\":{}",
            ",\"gaudium_a2ui\":{\"supportedCatalogIds\":\"basic\"}",
            ",\"gaudium_a2ui\":{\"supportedCatalogIds\":[1]}",
            ",\"gaudium_a2ui\":{\"1.0\":true}",
        ] {
            assert!(apply(true, &body(extra, USER))
                .unwrap_err()
                .contains("supportedCatalogIds"));
        }
    }

    #[test]
    fn refused_when_off_and_plain_text_without_a_common_catalog() {
        assert!(apply(false, &body(BASIC, USER))
            .unwrap_err()
            .contains("--a2ui"));
        let other = ",\"gaudium_a2ui\":{\"supportedCatalogIds\":[\"example.com:mine\"]}";
        match apply(true, &body(other, USER)).unwrap() {
            Applied::Unchanged { note: Some(n) } => assert!(n.contains("no catalog in common")),
            other => panic!("{other:?}"),
        }
        // only the first MAX_CATALOG_IDS identifiers are compared
        let many = format!(
            ",\"gaudium_a2ui\":{{\"supportedCatalogIds\":[{}\"{BASIC_CATALOG_ID}\"]}}",
            "\"x\",".repeat(MAX_CATALOG_IDS)
        );
        assert!(matches!(
            apply(true, &body(&many, USER)).unwrap(),
            Applied::Unchanged { note: Some(_) }
        ));
    }

    #[test]
    fn the_instructions_teach_what_the_capabilities_advertise() {
        assert!(INSTRUCTIONS.contains("<a2ui-json>") && INSTRUCTIONS.contains(BASIC_CATALOG_ID));
        assert!(INSTRUCTIONS.contains("\"version\":\"1.0\""));
        // the example block in the instructions is valid JSON
        let start = INSTRUCTIONS.rfind("<a2ui-json>").unwrap() + "<a2ui-json>".len();
        let end = INSTRUCTIONS.rfind("</a2ui-json>").unwrap();
        let example = parse_json(&INSTRUCTIONS[start..end]).unwrap();
        assert!(example.as_array().unwrap()[0].get("createSurface").is_some());

        let caps = parse_json(&capabilities_json(true, true)).unwrap();
        assert_eq!(caps.get("enabled").unwrap().as_bool(), Some(true));
        assert_eq!(caps.get("instructions").unwrap().as_str(), Some(INSTRUCTIONS));
        assert!(parse_json(&capabilities_json(false, false))
            .unwrap()
            .get("instructions")
            .is_none());
    }
}
