//! The Files API over the file store (`file_store.rs`), in the OpenAI shape and the Messages-API
//! shape, and `file_id` sources in chat and Messages requests.
//!
//! `POST /v1/files` (multipart `file`, and `purpose` in the OpenAI shape), `GET /v1/files`,
//! `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, `DELETE /v1/files/{id}`. Both APIs use
//! the same paths; a request with an `anthropic-version` or `anthropic-beta` header gets the
//! Messages-API shape (its objects, list paging and errors), any other the OpenAI shape. Files
//! are shared between the two: an id from one shape works in the other.
//!
//! Resolving `file_id` sources (before a request is translated, so the existing source handling
//! runs on the result):
//! - a Messages `image` block with `{"type":"file","file_id"}` becomes a base64 image source;
//! - a Messages `document` block's file source becomes a text source for a text file, else a
//!   base64 source of the file's media type (which the document path takes or refuses as it does
//!   inline data);
//! - a chat `{"type":"file","file":{"file_id"}}` part becomes an image part for an image, a text
//!   part for a text file, and else a `file_data` part (the path inline file data takes).

use std::io::{Read, Write};

use crate::json::JsonValue;

use super::conn::Conn;
use super::file_store::{is_media_type, FileError, FileMeta, FileStore, Owner};
use super::multipart::{boundary, BodyReader, Multipart};
use super::readout::jq;
use super::{anthropic, write_error, write_json, AppState, HttpRequest, MAX_REQUEST_BODY_BYTES};

/// The purposes an OpenAI-shape upload may name.
const PURPOSES: &[&str] = &[
    "assistants",
    "batch",
    "fine-tune",
    "vision",
    "user_data",
    "evals",
];
/// The longest value of a small form field (`purpose`).
const MAX_FIELD_BYTES: u64 = 1024;
/// How much of a refused upload's body is still read, so the client sees the answer instead of
/// a reset connection.
const DRAIN_BYTES: u64 = 64 << 20;
/// The legacy beta of the Messages-API Files API, whose list pages by `before_id` / `after_id`.
const LEGACY_BETA: &str = "files-api-2025-04-14";

/// Whether a request's body is left unread by the request parser for its handler to stream.
pub(super) fn streams_body(method: &str, path: &str) -> bool {
    method == "POST" && path == "/v1/files"
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    OpenAi,
    Messages { legacy: bool },
}

fn shape_of(req: &HttpRequest) -> Shape {
    let beta = req.headers.get("anthropic-beta");
    if beta.is_some() || req.headers.contains_key("anthropic-version") {
        Shape::Messages {
            legacy: beta.is_some_and(|b| b.split(',').any(|x| x.trim() == LEGACY_BETA)),
        }
    } else {
        Shape::OpenAi
    }
}

/// The owner a request acts as: the key it presented when the server has API keys.
pub(super) fn owner_of(state: &AppState, req: &HttpRequest) -> Owner {
    if state.cfg.api_key.is_none() {
        return Owner::everyone();
    }
    let bearer = req
        .headers
        .get("authorization")
        .and_then(|a| a.strip_prefix("Bearer "))
        .map(str::trim);
    let key = bearer
        .or_else(|| req.headers.get("x-api-key").map(|k| k.trim()))
        .unwrap_or("");
    Owner::of_key(key)
}

fn error(shape: Shape, stream: &mut Conn, status: u16, msg: &str) {
    match shape {
        Shape::OpenAi => {
            let t = match status {
                401 => "authentication_error",
                404 => "not_found_error",
                s if s >= 500 => "server_error",
                _ => "invalid_request_error",
            };
            write_error(stream, status, t, msg)
        }
        Shape::Messages { .. } => write_json(stream, status, &anthropic::error_json(status, msg)),
    }
}

fn file_error(shape: Shape, stream: &mut Conn, e: &FileError) {
    if let (Shape::Messages { .. }, FileError::OverQuota { .. }) = (shape, e) {
        let body = format!(
            "{{\"type\":\"error\",\"error\":{{\"type\":\"invalid_request_error\",\"message\":{},\"details\":{{\"error_code\":\"storage_quota_exceeded\"}}}}}}",
            jq(&e.message())
        );
        return write_json(stream, 400, &body);
    }
    error(shape, stream, e.status(), &e.message())
}

const FILES_OFF: &str = "the Files API is off on this server: start it with --files-dir <dir>";

fn store<'a>(state: &'a AppState, shape: Shape, stream: &mut Conn) -> Option<&'a FileStore> {
    let s = state.files.as_deref();
    if s.is_none() {
        error(shape, stream, 404, FILES_OFF);
    }
    s
}

/// Unix seconds as RFC 3339 (`2025-04-15T18:37:24Z`).
fn rfc3339(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // civil date from days since 1970-01-01 (proleptic Gregorian)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn object_json(shape: Shape, m: &FileMeta) -> String {
    match shape {
        Shape::OpenAi => format!(
            "{{\"id\":{},\"object\":\"file\",\"bytes\":{},\"created_at\":{},\"expires_at\":null,\"filename\":{},\"purpose\":{},\"status\":\"processed\",\"status_details\":null}}",
            jq(&m.id),
            m.bytes,
            m.created_at,
            jq(&m.filename),
            jq(&m.purpose)
        ),
        Shape::Messages { .. } => format!(
            "{{\"id\":{},\"type\":\"file\",\"filename\":{},\"mime_type\":{},\"size_bytes\":{},\"created_at\":{},\"downloadable\":{},\"expires_at\":null}}",
            jq(&m.id),
            jq(&m.filename),
            jq(&m.mime_type),
            m.bytes,
            jq(&rfc3339(m.created_at)),
            m.downloadable
        ),
    }
}

/// Reads what is left of a refused upload's body (up to [`DRAIN_BYTES`]).
fn drain(body: &mut dyn Read) {
    let _ = std::io::copy(&mut body.take(DRAIN_BYTES), &mut std::io::sink());
}

/// Refuses an upload before its body is read: the answer first, then (only when the client is
/// sending the body, not waiting for a `100 Continue`) what is left of the body.
fn refuse_unread(
    shape: Shape,
    stream: &mut Conn,
    body: &mut dyn Read,
    expect: bool,
    status: u16,
    msg: &str,
) {
    error(shape, stream, status, msg);
    if !expect {
        drain(body);
    }
}

/// `POST /v1/files`: one multipart `file` part, streamed to the store.
pub(super) fn handle_upload(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let shape = shape_of(req);
    let expect = req
        .headers
        .get("expect")
        .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"));
    let prefix = req.streamed.clone().unwrap_or_default();
    let Ok(conn) = stream.try_clone() else {
        return error(shape, stream, 500, "cannot read the upload");
    };
    let _ = conn.set_read_timeout(Some(std::time::Duration::from_secs(60)));
    let mut body = match BodyReader::new(prefix, conn, &req.headers) {
        Ok(b) => b,
        Err(e) => return error(shape, stream, 400, &e),
    };
    let Some(store) = state.files.as_deref() else {
        return refuse_unread(shape, stream, &mut body, expect, 404, FILES_OFF);
    };
    let ct = req.headers.get("content-type").map_or("", String::as_str);
    let b = match boundary(ct) {
        Ok(b) => b,
        Err(e) => return refuse_unread(shape, stream, &mut body, expect, 400, &e),
    };
    if expect {
        let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        let _ = stream.flush();
    }
    let owner = owner_of(state, req);
    // the file part is at most the body: the store reserves that much (or its cap) before reading
    let declared = body.declared_len();
    let parsed = read_upload(store, &mut body, &b, shape, declared);
    // what follows the closing delimiter, or the rest of a refused upload
    drain(&mut body);
    match parsed {
        Ok(up) => {
            let purpose = match (shape, up.purpose) {
                (Shape::OpenAi, None) => {
                    return error(shape, stream, 400, "'purpose' is required: one of assistants, batch, fine-tune, vision, user_data, evals")
                }
                (Shape::OpenAi, Some(p)) if !PURPOSES.contains(&p.as_str()) => {
                    return error(
                        shape,
                        stream,
                        400,
                        &format!("unknown purpose '{p}': one of assistants, batch, fine-tune, vision, user_data, evals"),
                    )
                }
                (_, Some(p)) => p,
                (Shape::Messages { .. }, None) => "user_data".to_string(),
            };
            match store.commit(
                up.staged,
                &owner,
                up.filename.as_deref().unwrap_or(""),
                up.content_type.as_deref(),
                &purpose,
                false,
            ) {
                Ok(m) => write_json(stream, 200, &object_json(shape, &m)),
                Err(e) => file_error(shape, stream, &e),
            }
        }
        Err(e) => file_error(shape, stream, &e),
    }
}

struct Upload<'a> {
    staged: super::file_store::Staged<'a>,
    filename: Option<String>,
    content_type: Option<String>,
    purpose: Option<String>,
}

fn read_upload<'a>(
    store: &'a FileStore,
    body: &mut dyn Read,
    boundary: &str,
    shape: Shape,
    declared: Option<u64>,
) -> Result<Upload<'a>, FileError> {
    let io = |e: std::io::Error| FileError::BadUpload(format!("invalid multipart body: {e}"));
    let bad = |m: String| FileError::BadUpload(m);
    let mut mp = Multipart::new(body, boundary);
    let mut file = None;
    let mut purpose = None;
    while let Some(head) = mp.next_part().map_err(io)? {
        match head.name.as_str() {
            "file" => {
                if file.is_some() {
                    return Err(bad("an upload holds one 'file' part".into()));
                }
                let staged = store.stage(declared, &mut mp.part())?;
                // a file the store will refuse is not read further
                store.precheck(&staged)?;
                file = Some((staged, head.filename, head.content_type));
            }
            "purpose" => {
                let mut v = String::new();
                mp.part()
                    .take(MAX_FIELD_BYTES)
                    .read_to_string(&mut v)
                    .map_err(|_| bad("'purpose' must be UTF-8 text".into()))?;
                purpose = Some(v.trim().to_string());
            }
            k if k.starts_with("expires_after") || k == "expires_in_seconds" => {
                return Err(bad(format!(
                    "'{k}': file expiry is not supported by this server (files are kept until deleted)"
                )))
            }
            other => {
                return Err(bad(format!(
                    "unknown form field '{other}' (an upload takes 'file'{})",
                    if shape == Shape::OpenAi { " and 'purpose'" } else { "" }
                )))
            }
        }
    }
    let (staged, filename, content_type) =
        file.ok_or_else(|| bad("an upload needs a 'file' part".into()))?;
    Ok(Upload {
        staged,
        filename,
        content_type,
        purpose,
    })
}

fn limit_param(req: &HttpRequest, default: usize, max: usize) -> Result<usize, String> {
    match req.query.get("limit") {
        None => Ok(default),
        Some(v) => v
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=max).contains(n))
            .ok_or_else(|| format!("'limit' must be 1 to {max}")),
    }
}

/// The position after the entry `id` in `files`, or an error naming the unknown cursor.
fn after(files: &[FileMeta], id: &str, key: &str) -> Result<usize, String> {
    files
        .iter()
        .position(|m| m.id == id)
        .map(|i| i + 1)
        .ok_or_else(|| format!("'{key}': no file '{id}'"))
}

fn page_json(shape: Shape, page: &[FileMeta]) -> String {
    page.iter()
        .map(|m| object_json(shape, m))
        .collect::<Vec<_>>()
        .join(",")
}

fn first_last(page: &[FileMeta]) -> (String, String) {
    let id = |m: Option<&FileMeta>| m.map_or("null".to_string(), |m| jq(&m.id));
    (id(page.first()), id(page.last()))
}

/// `GET /v1/files`.
pub(super) fn handle_list(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let shape = shape_of(req);
    let Some(store) = store(state, shape, stream) else {
        return;
    };
    let mut files = store.list(&owner_of(state, req));
    match list_json(shape, req, &mut files) {
        Ok(j) => write_json(stream, 200, &j),
        Err(e) => error(shape, stream, 400, &e),
    }
}

fn list_json(shape: Shape, req: &HttpRequest, files: &mut Vec<FileMeta>) -> Result<String, String> {
    let q = |k: &str| req.query.get(k).map(String::as_str);
    match shape {
        Shape::OpenAi => {
            if let Some(p) = q("purpose") {
                files.retain(|m| m.purpose == p);
            }
            match q("order").unwrap_or("desc") {
                "desc" => files.reverse(),
                "asc" => {}
                o => return Err(format!("'order' must be asc or desc (got '{o}')")),
            }
            let limit = limit_param(req, 10_000, 10_000)?;
            let start = match q("after") {
                Some(id) => after(files, id, "after")?,
                None => 0,
            };
            let rest = &files[start..];
            let page = &rest[..limit.min(rest.len())];
            let (first, last) = first_last(page);
            Ok(format!(
                "{{\"object\":\"list\",\"data\":[{}],\"first_id\":{first},\"last_id\":{last},\"has_more\":{}}}",
                page_json(shape, page),
                rest.len() > page.len()
            ))
        }
        Shape::Messages { legacy } => {
            files.reverse(); // newest first
            if let Some(ids) = q("ids") {
                if q("page").is_some() || q("limit").is_some() {
                    return Err("'ids' cannot be combined with 'page' or 'limit'".into());
                }
                let want: Vec<&str> = ids
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                if want.len() > 100 {
                    return Err("'ids' names at most 100 files".into());
                }
                files.retain(|m| want.contains(&m.id.as_str()));
                return Ok(format!(
                    "{{\"data\":[{}],\"next_page\":null}}",
                    page_json(shape, files)
                ));
            }
            let limit = limit_param(req, 20, 1000)?;
            if legacy {
                let lo = match q("after_id") {
                    Some(id) => after(files, id, "after_id")?,
                    None => 0,
                };
                let hi = match q("before_id") {
                    Some(id) => after(files, id, "before_id")? - 1,
                    None => files.len(),
                };
                let window = &files[lo.min(hi)..hi];
                // only a before_id cursor: the entries just before it
                let page = if q("before_id").is_some() && q("after_id").is_none() {
                    &window[window.len().saturating_sub(limit)..]
                } else {
                    &window[..limit.min(window.len())]
                };
                let (first, last) = first_last(page);
                return Ok(format!(
                    "{{\"data\":[{}],\"has_more\":{},\"first_id\":{first},\"last_id\":{last}}}",
                    page_json(shape, page),
                    window.len() > page.len()
                ));
            }
            let start = match q("page") {
                Some(p) => {
                    let id = p
                        .strip_prefix("page_")
                        .ok_or("'page' is a next_page value")?;
                    after(files, id, "page")?
                }
                None => 0,
            };
            let rest = &files[start..];
            let page = &rest[..limit.min(rest.len())];
            let next = match page.last() {
                Some(m) if rest.len() > page.len() => jq(&format!("page_{}", m.id)),
                _ => "null".into(),
            };
            Ok(format!(
                "{{\"data\":[{}],\"next_page\":{next}}}",
                page_json(shape, page)
            ))
        }
    }
}

/// `GET` / `DELETE /v1/files/{id}` and `GET /v1/files/{id}/content`.
pub(super) fn handle_file(state: &AppState, req: &HttpRequest, stream: &mut Conn) {
    let shape = shape_of(req);
    let Some(store) = store(state, shape, stream) else {
        return;
    };
    let owner = owner_of(state, req);
    let rest = &req.path["/v1/files/".len()..];
    let (id, content) = match rest.strip_suffix("/content") {
        Some(id) => (id, true),
        None => (rest, false),
    };
    match (req.method.as_str(), content) {
        ("GET", false) => match store.get(&owner, id) {
            Ok(m) => write_json(stream, 200, &object_json(shape, &m)),
            Err(e) => file_error(shape, stream, &e),
        },
        ("DELETE", false) => match store.delete(&owner, id) {
            Ok(m) => {
                let body = match shape {
                    Shape::OpenAi => {
                        format!(
                            "{{\"id\":{},\"object\":\"file\",\"deleted\":true}}",
                            jq(&m.id)
                        )
                    }
                    Shape::Messages { .. } => {
                        format!("{{\"id\":{},\"type\":\"file_deleted\"}}", jq(&m.id))
                    }
                };
                write_json(stream, 200, &body)
            }
            Err(e) => file_error(shape, stream, &e),
        },
        ("GET", true) => match store.open_content(&owner, id) {
            Ok((m, f)) => {
                if matches!(shape, Shape::Messages { .. }) && !m.downloadable {
                    let body = format!(
                        "{{\"type\":\"error\",\"error\":{{\"type\":\"invalid_request_error\",\"message\":{},\"details\":{{\"error_code\":\"file_not_downloadable\"}}}}}}",
                        jq(&format!("file '{id}' is not downloadable: uploaded files are not (the OpenAI shape of this endpoint serves them)"))
                    );
                    return write_json(stream, 400, &body);
                }
                send_content(stream, &m, f)
            }
            Err(e) => file_error(shape, stream, &e),
        },
        _ => error(
            shape,
            stream,
            404,
            &format!("unknown endpoint {} {}", req.method, req.path),
        ),
    }
}

fn send_content(stream: &mut Conn, m: &FileMeta, mut f: std::fs::File) {
    let name: String = m
        .filename
        .chars()
        .map(|c| {
            if c == '"' || c == '\\' || !c.is_ascii() {
                '_'
            } else {
                c
            }
        })
        .collect();
    // a type read back from a file written before types were checked is not sent as it is
    let mime = if is_media_type(&m.mime_type) {
        m.mime_type.as_str()
    } else {
        "application/octet-stream"
    };
    let head = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Content-Disposition: attachment; filename=\"{name}\"\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         {}\
         Connection: close\r\n\r\n",
        mime,
        m.bytes,
        stream.hsts_line()
    );
    if stream.write_all(head.as_bytes()).is_ok() {
        let _ = std::io::copy(&mut (&mut f).take(m.bytes), stream);
    }
    let _ = stream.flush();
}

// ------------------------------------------------------------------------------------------------
// file_id sources
// ------------------------------------------------------------------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let v = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        s.push(B64[(v >> 18) as usize] as char);
        s.push(B64[((v >> 12) & 63) as usize] as char);
        s.push(if c.len() > 1 {
            B64[((v >> 6) & 63) as usize] as char
        } else {
            '='
        });
        s.push(if c.len() > 2 {
            B64[(v & 63) as usize] as char
        } else {
            '='
        });
    }
    s
}

fn is_text(mime: &str) -> bool {
    mime.starts_with("text/")
        || matches!(
            mime,
            "application/json" | "application/jsonl" | "application/x-ndjson" | "application/xml"
        )
}

fn obj(pairs: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn s(v: &str) -> JsonValue {
    JsonValue::Str(v.to_string())
}

fn get_mut<'a>(v: &'a mut JsonValue, key: &str) -> Option<&'a mut JsonValue> {
    match v {
        JsonValue::Object(e) => e.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

/// Reads `id` for use in a request: its metadata and bytes. All the files one request names hold
/// at most the size an inline request body may have, together (`left`: what is left of it).
struct Resolver<'a> {
    store: Option<&'a FileStore>,
    owner: Owner,
    left: std::cell::Cell<u64>,
}

impl Resolver<'_> {
    fn read(&self, id: &str, path: &str) -> Result<(FileMeta, Vec<u8>), String> {
        let Some(store) = self.store else {
            return Err(format!(
                "{path}: file_id sources need the Files API, which is off on this server (start it with --files-dir <dir>)"
            ));
        };
        let left = self.left.get();
        match store.read_limited(&self.owner, id, left) {
            Ok(Ok(x)) => {
                self.left.set(left - x.1.len() as u64);
                Ok(x)
            }
            Ok(Err(size)) if size > MAX_REQUEST_BODY_BYTES as u64 => Err(format!(
                "{path}: file '{id}' has {size} bytes, more than a request may hold ({MAX_REQUEST_BODY_BYTES})"
            )),
            Ok(Err(size)) => Err(format!(
                "{path}: file '{id}' ({size} bytes) takes the files this request names past the {MAX_REQUEST_BODY_BYTES} bytes a request may hold together"
            )),
            Err(FileError::NotFound(_)) => Err(format!("{path}: file '{id}' not found")),
            Err(e) => Err(format!("{path}: {}", e.message())),
        }
    }

    fn text(&self, meta: &FileMeta, bytes: Vec<u8>, path: &str) -> Result<String, String> {
        String::from_utf8(bytes).map_err(|_| {
            format!(
                "{path}: file '{}' ({}) is not UTF-8 text",
                meta.id, meta.mime_type
            )
        })
    }

    /// A Messages `source` object: a `file` source rewritten into the inline source it stands
    /// for; any other source is left as it is.
    fn message_source(&self, src: &mut JsonValue, image: bool, path: &str) -> Result<(), String> {
        if src.get("type").and_then(JsonValue::as_str) != Some("file") {
            return Ok(());
        }
        let id = src
            .get("file_id")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| format!("{path}.file_id: Field required (a string)"))?
            .to_string();
        let (meta, bytes) = self.read(&id, path)?;
        *src = if image {
            if !meta.mime_type.starts_with("image/") {
                return Err(format!(
                    "{path}: file '{id}' is {}, not an image",
                    meta.mime_type
                ));
            }
            obj(vec![
                ("type", s("base64")),
                ("media_type", s(&meta.mime_type)),
                ("data", JsonValue::Str(base64(&bytes))),
            ])
        } else if is_text(&meta.mime_type) {
            let text = self.text(&meta, bytes, path)?;
            obj(vec![
                ("type", s("text")),
                ("media_type", s("text/plain")),
                ("data", JsonValue::Str(text)),
            ])
        } else {
            obj(vec![
                ("type", s("base64")),
                ("media_type", s(&meta.mime_type)),
                ("data", JsonValue::Str(base64(&bytes))),
            ])
        };
        Ok(())
    }

    fn message_blocks(&self, blocks: &mut [JsonValue], path: &str) -> Result<(), String> {
        for (i, b) in blocks.iter_mut().enumerate() {
            let p = format!("{path}.{i}");
            let kind = b
                .get("type")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string();
            match kind.as_str() {
                "image" | "document" => {
                    if let Some(src) = get_mut(b, "source") {
                        self.message_source(src, kind == "image", &format!("{p}.source"))?;
                    }
                }
                "tool_result" => {
                    if let Some(JsonValue::Array(inner)) = get_mut(b, "content") {
                        self.message_blocks(inner, &format!("{p}.content"))?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// A chat `file` part that names a `file_id`, rewritten into the part it stands for.
    fn chat_part(&self, part: &mut JsonValue, path: &str) -> Result<(), String> {
        let Some(id) = part
            .get("file")
            .and_then(|f| f.get("file_id"))
            .and_then(JsonValue::as_str)
            .map(str::to_string)
        else {
            return Ok(());
        };
        let (meta, bytes) = self.read(&id, path)?;
        *part = if meta.mime_type.starts_with("image/") {
            let url = format!("data:{};base64,{}", meta.mime_type, base64(&bytes));
            obj(vec![
                ("type", s("image_url")),
                ("image_url", obj(vec![("url", JsonValue::Str(url))])),
            ])
        } else if is_text(&meta.mime_type) {
            let text = self.text(&meta, bytes, path)?;
            obj(vec![("type", s("text")), ("text", JsonValue::Str(text))])
        } else {
            let data = format!("data:{};base64,{}", meta.mime_type, base64(&bytes));
            obj(vec![
                ("type", s("file")),
                (
                    "file",
                    obj(vec![
                        ("filename", s(&meta.filename)),
                        ("file_data", JsonValue::Str(data)),
                    ]),
                ),
            ])
        };
        Ok(())
    }
}

fn resolver<'a>(state: &'a AppState, req: &HttpRequest) -> Resolver<'a> {
    Resolver {
        store: state.files.as_deref(),
        owner: owner_of(state, req),
        left: std::cell::Cell::new(MAX_REQUEST_BODY_BYTES as u64),
    }
}

/// A Messages request body with its `file` sources resolved.
pub(super) fn resolve_messages(
    state: &AppState,
    req: &HttpRequest,
    body: &mut JsonValue,
) -> Result<(), String> {
    let r = resolver(state, req);
    let Some(JsonValue::Array(msgs)) = get_mut(body, "messages") else {
        return Ok(());
    };
    for (i, m) in msgs.iter_mut().enumerate() {
        if let Some(JsonValue::Array(blocks)) = get_mut(m, "content") {
            r.message_blocks(blocks, &format!("messages.{i}.content"))?;
        }
    }
    Ok(())
}

/// A chat-completions request body with its `file_id` parts resolved.
pub(super) fn resolve_chat(
    state: &AppState,
    req: &HttpRequest,
    body: &mut JsonValue,
) -> Result<(), String> {
    let r = resolver(state, req);
    let Some(JsonValue::Array(msgs)) = get_mut(body, "messages") else {
        return Ok(());
    };
    for (i, m) in msgs.iter_mut().enumerate() {
        if let Some(JsonValue::Array(parts)) = get_mut(m, "content") {
            for (j, p) in parts.iter_mut().enumerate() {
                if p.get("type").and_then(JsonValue::as_str) == Some("file") {
                    r.chat_part(p, &format!("messages.{i}.content.{j}"))?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_files_one_request_names_share_one_budget() {
        let d = std::env::temp_dir().join(format!("gaudium-files-budget-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let store = FileStore::open(&d, 100, 100).unwrap();
        let f = store
            .create(
                &Owner::everyone(),
                "a.txt",
                None,
                "user_data",
                false,
                b"0123456789",
            )
            .unwrap();
        let r = Resolver {
            store: Some(&store),
            owner: Owner::everyone(),
            left: std::cell::Cell::new(15),
        };
        assert_eq!(r.read(&f.id, "a").unwrap().1, b"0123456789");
        // the same file named again does not fit the 5 bytes left
        let e = r.read(&f.id, "b").unwrap_err();
        assert!(e.contains("together"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn dates_and_base64() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_744_742_244), "2025-04-15T18:37:24Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(base64(b"hello"), "aGVsbG8=");
        assert_eq!(base64(b"hi!"), "aGkh");
        assert_eq!(base64(b"h"), "aA==");
    }
}
