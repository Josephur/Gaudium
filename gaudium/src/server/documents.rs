//! Binary documents in requests: Messages `document` blocks with a PDF source (`base64` or
//! `url`) and chat-completions `file` parts become the text of the document before the request
//! is translated and rendered.
//!
//! A PDF is reduced to its text layer (`crate::pdf`), page by page; a PDF without one is
//! refused with a 400 that says so. Rendering pages to images for a vision model would plug in
//! here, beside the text: it is not done yet (it needs a font rasteriser).
//!
//! A `url` source is fetched only through the MCP proxy's outbound rules (`proxy.rs`, decided
//! by `verified/mcp_proxy.bend`), and only when the server was started with the proxy on;
//! otherwise it is refused with a 400 that says how to send the document instead. `file`
//! sources (uploaded files) are left to the Messages translation, which refuses them.
//!
//! The documents of one request are charged together to one [`pdf::Request`] budget (verified
//! like the budgets of one PDF): each is counted before it is decoded or fetched, its text
//! before it is kept, and the heap the expansion holds across all of them is checked at every
//! charge, so many documents that each fit cannot add up past the caps.

use crate::json::JsonValue;
use crate::pdf;

/// Fetches a document URL: `(url, max_bytes)` -> (body, content type). `None`: the server
/// fetches no URLs.
pub(super) type Fetch<'a> = &'a dyn Fn(&str, u64) -> Result<(Vec<u8>, Option<String>), String>;

/// What the expansion may use.
pub(super) struct Ctx<'a> {
    pub limits: pdf::Limits,
    pub fetch: Option<Fetch<'a>>,
}

type R<T> = Result<T, String>;

fn field_mut<'v>(v: &'v mut JsonValue, key: &str) -> Option<&'v mut JsonValue> {
    match v {
        JsonValue::Object(e) => e.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn kind(v: &JsonValue) -> &str {
    v.get("type").and_then(JsonValue::as_str).unwrap_or("")
}

fn obj(pairs: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn s(v: &str) -> JsonValue {
    JsonValue::Str(v.to_string())
}

/// Counts one more document of the request, before it is decoded or fetched.
fn admit(req: &mut pdf::Request, path: &str) -> R<()> {
    req.admit().map_err(|e| format!("{path}: {e}"))
}

/// The text of a document's bytes: a PDF's text layer, or UTF-8 text, charged to the
/// request's text budget before it is kept.
fn bytes_text(
    bytes: &[u8],
    media: &str,
    ctx: &Ctx,
    req: &mut pdf::Request,
    path: &str,
) -> R<String> {
    let media = media.trim().to_ascii_lowercase();
    let text = if media.starts_with("application/pdf") || pdf::is_pdf(bytes) {
        pdf::extract_text(bytes, &ctx.limits)
            .map(|e| e.text())
            .map_err(|e| format!("{path}: {e}"))?
    } else if let Some(t) = (media.starts_with("text/") || media.is_empty())
        .then(|| std::str::from_utf8(bytes).ok())
        .flatten()
    {
        t.to_string()
    } else {
        return Err(unsupported(&media, path));
    };
    req.keep(text.len() as u64)
        .map_err(|e| format!("{path}: {e}"))?;
    Ok(text)
}

fn unsupported(media: &str, path: &str) -> String {
    let what = if media.is_empty() { "this" } else { media };
    format!(
        "{path}: {what} documents are not supported by this server (PDF and plain-text documents are); send the text"
    )
}

fn decode_base64(data: &str, path: &str) -> R<Vec<u8>> {
    super::anthropic::base64_decode(data).ok_or_else(|| format!("{path}: invalid base64 data"))
}

/// Rewrites the Messages request's document blocks whose source is a PDF (`base64` with
/// `application/pdf`, or a `url`) into text sources. Returns the request fields accepted but
/// not applied (`citations`).
pub(super) fn expand_messages(body: &mut JsonValue, ctx: &Ctx) -> R<Vec<String>> {
    let mut ignored = Vec::new();
    let req = &mut pdf::Request::new(&ctx.limits);
    let Some(JsonValue::Array(messages)) = field_mut(body, "messages") else {
        return Ok(ignored);
    };
    for (i, m) in messages.iter_mut().enumerate() {
        let Some(JsonValue::Array(blocks)) = field_mut(m, "content") else {
            continue;
        };
        for (j, b) in blocks.iter_mut().enumerate() {
            let p = format!("messages.{i}.content.{j}");
            match kind(b) {
                "document" => document(b, &p, ctx, req, &mut ignored)?,
                "tool_result" => {
                    if let Some(JsonValue::Array(inner)) = field_mut(b, "content") {
                        for (k, x) in inner.iter_mut().enumerate() {
                            if kind(x) == "document" {
                                let p = format!("{p}.content.{k}");
                                document(x, &p, ctx, req, &mut ignored)?;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(ignored)
}

fn document(
    b: &mut JsonValue,
    path: &str,
    ctx: &Ctx,
    req: &mut pdf::Request,
    ignored: &mut Vec<String>,
) -> R<()> {
    if b.get("citations")
        .and_then(|c| c.get("enabled"))
        .and_then(JsonValue::as_bool)
        == Some(true)
    {
        ignored.push(format!("{path}.citations (citations are not served)"));
    }
    let Some(src) = b.get("source").filter(|x| x.as_object().is_some()) else {
        return Ok(()); // the translation reports the missing source
    };
    let sp = format!("{path}.source");
    let text = match kind(src) {
        "base64" => {
            let media = src
                .get("media_type")
                .and_then(JsonValue::as_str)
                .unwrap_or("");
            if !media.trim().eq_ignore_ascii_case("application/pdf") {
                return Ok(()); // text documents, and the refusal of other types, stay as they were
            }
            let Some(data) = src.get("data").and_then(JsonValue::as_str) else {
                return Err(format!("{sp}.data: Field required (a string)"));
            };
            admit(req, path)?;
            let bytes = decode_base64(data, &format!("{sp}.data"))?;
            bytes_text(&bytes, media, ctx, req, path)?
        }
        "url" => {
            let Some(url) = src.get("url").and_then(JsonValue::as_str) else {
                return Err(format!("{sp}.url: Field required (a string)"));
            };
            let Some(fetch) = ctx.fetch else {
                return Err(format!(
                    "{sp}: document url sources are fetched through the server's outbound proxy, which is off (start the server with --mcp-proxy and an allow-list that names the host); send the document as base64"
                ));
            };
            admit(req, path)?;
            let (bytes, ctype) =
                fetch(url, ctx.limits.max_bytes).map_err(|e| format!("{sp}: {e}"))?;
            bytes_text(&bytes, ctype.as_deref().unwrap_or(""), ctx, req, path)?
        }
        _ => return Ok(()),
    };
    if let Some(slot) = field_mut(b, "source") {
        *slot = obj(vec![
            ("type", s("text")),
            ("media_type", s("text/plain")),
            ("data", JsonValue::Str(text)),
        ]);
    }
    Ok(())
}

/// Rewrites the chat request's `file` content parts that carry their data
/// (`{"type":"file","file":{"file_data":"data:application/pdf;base64,...","filename":...}}`)
/// into text parts. A part that names an uploaded file (`file_id`) is left for the renderer,
/// which refuses it.
pub(super) fn expand_chat(body: &mut JsonValue, ctx: &Ctx) -> R<()> {
    let Some(JsonValue::Array(messages)) = field_mut(body, "messages") else {
        return Ok(());
    };
    let req = &mut pdf::Request::new(&ctx.limits);
    for (i, m) in messages.iter_mut().enumerate() {
        let Some(JsonValue::Array(parts)) = field_mut(m, "content") else {
            continue;
        };
        for (j, part) in parts.iter_mut().enumerate() {
            if kind(part) != "file" {
                continue;
            }
            let path = format!("messages[{i}].content[{j}].file");
            let Some(file) = part.get("file").filter(|f| f.as_object().is_some()) else {
                return Err(format!("{path}: a file content part needs a 'file' object"));
            };
            let Some(data) = file.get("file_data").and_then(JsonValue::as_str) else {
                continue;
            };
            let (media, b64) = match data.strip_prefix("data:") {
                Some(rest) => match rest.split_once(',') {
                    Some((head, b64)) if head.ends_with(";base64") => {
                        (head.trim_end_matches(";base64").to_string(), b64)
                    }
                    _ => {
                        return Err(format!(
                            "{path}.file_data: must be a base64 data URL (data:<type>;base64,...)"
                        ))
                    }
                },
                None => (String::new(), data),
            };
            admit(req, &path)?;
            let bytes = decode_base64(b64, &format!("{path}.file_data"))?;
            let mut text = bytes_text(&bytes, &media, ctx, req, &path)?;
            if let Some(name) = file
                .get("filename")
                .and_then(JsonValue::as_str)
                .filter(|n| !n.is_empty())
            {
                text = format!("{name}\n\n{text}");
            }
            *part = obj(vec![("type", s("text")), ("text", JsonValue::Str(text))]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    /// A small PDF without an xref table (the reader rebuilds it by a scan).
    fn pdf(text: &str) -> Vec<u8> {
        let content = if text.is_empty() {
            String::new()
        } else {
            format!("BT /F1 12 Tf 72 700 Td ({text}) Tj ET")
        };
        format!(
            "%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n\
             2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n\
             3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n\
             4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n\
             5 0 obj << /Length {} >>\nstream\n{content}\nendstream endobj\n\
             trailer << /Root 1 0 R >>\n%%EOF\n",
            content.len()
        )
        .into_bytes()
    }

    /// A small PDF whose page draws a form showing 1,000 letters `draws` times.
    fn pdf_drawn(draws: usize) -> Vec<u8> {
        let form = format!("BT /F1 12 Tf 0 0 Td ({}) Tj ET", "A".repeat(1000));
        let content = "q /X1 Do Q\n".repeat(draws);
        format!(
            "%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n\
             2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n\
             3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> /XObject << /X1 6 0 R >> >> /Contents 5 0 R >> endobj\n\
             4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n\
             5 0 obj << /Length {} >>\nstream\n{content}\nendstream endobj\n\
             6 0 obj << /Type /XObject /Subtype /Form /BBox [0 0 10 10] /Length {} >>\nstream\n{form}\nendstream endobj\n\
             trailer << /Root 1 0 R >>\n%%EOF\n",
            content.len(),
            form.len()
        )
        .into_bytes()
    }

    fn b64(b: &[u8]) -> String {
        const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut o = String::new();
        for c in b.chunks(3) {
            let v = u32::from(c[0]) << 16
                | u32::from(*c.get(1).unwrap_or(&0)) << 8
                | u32::from(*c.get(2).unwrap_or(&0));
            for k in 0..4 {
                if k <= c.len() {
                    o.push(A[(v >> (18 - 6 * k) & 63) as usize] as char);
                } else {
                    o.push('=');
                }
            }
        }
        o
    }

    fn ctx() -> Ctx<'static> {
        Ctx {
            limits: pdf::Limits::default(),
            fetch: None,
        }
    }

    fn messages_body(block: &str) -> JsonValue {
        parse_json(&format!(
            "{{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{{\"role\":\"user\",\"content\":[{block},{{\"type\":\"text\",\"text\":\"Summarise it.\"}}]}}]}}"
        ))
        .unwrap()
    }

    #[test]
    fn a_pdf_document_block_becomes_its_text() {
        let block = format!(
            "{{\"type\":\"document\",\"title\":\"Report\",\"context\":\"From the archive\",\"citations\":{{\"enabled\":true}},\"source\":{{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"{}\"}}}}",
            b64(&pdf("Hello PDF"))
        );
        let mut body = messages_body(&block);
        let ignored = expand_messages(&mut body, &ctx()).unwrap();
        assert_eq!(
            ignored,
            vec!["messages.0.content.0.citations (citations are not served)"]
        );
        let t = super::super::anthropic::translate(&body).unwrap();
        let chat = t.chat.dump();
        assert!(
            chat.contains(
                "Report\\n\\nFrom the archive\\n\\n--- Page 1 ---\\nHello PDF\\nSummarise it."
            ),
            "{chat}"
        );
    }

    #[test]
    fn documents_in_tool_results_and_urls() {
        let block = format!(
            "{{\"type\":\"tool_result\",\"tool_use_id\":\"t1\",\"content\":[{{\"type\":\"document\",\"source\":{{\"type\":\"url\",\"url\":\"https://docs.example/a.pdf\"}}}}]}}"
        );
        // no fetch: refused with what to do instead
        let mut body = messages_body(&block);
        let e = expand_messages(&mut body, &ctx()).unwrap_err();
        assert!(
            e.starts_with("messages.0.content.0.content.0.source:"),
            "{e}"
        );
        assert!(e.contains("--mcp-proxy") && e.contains("base64"), "{e}");
        // with a fetch: the fetched PDF's text, and the fetch gets the PDF size limit
        let bytes = pdf("Fetched");
        let fetch = |url: &str, max: u64| {
            assert_eq!(url, "https://docs.example/a.pdf");
            assert_eq!(max, pdf::Limits::default().max_bytes);
            Ok((bytes.clone(), Some("application/pdf".to_string())))
        };
        let c = Ctx {
            limits: pdf::Limits::default(),
            fetch: Some(&fetch),
        };
        let mut body = messages_body(&block);
        expand_messages(&mut body, &c).unwrap();
        assert!(
            body.dump().contains("--- Page 1 ---\\nFetched"),
            "{}",
            body.dump()
        );
        // a fetch that fails carries its reason
        let failing = |_: &str, _: u64| Err("the document url was refused: not listed".to_string());
        let c = Ctx {
            limits: pdf::Limits::default(),
            fetch: Some(&failing),
        };
        let e = expand_messages(&mut messages_body(&block), &c).unwrap_err();
        assert!(e.contains("not listed"), "{e}");
    }

    #[test]
    fn pdfs_without_text_and_other_binaries_are_refused() {
        let block = format!(
            "{{\"type\":\"document\",\"source\":{{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"{}\"}}}}",
            b64(&pdf(""))
        );
        let e = expand_messages(&mut messages_body(&block), &ctx()).unwrap_err();
        assert!(e.contains("text layer is empty"), "{e}");
        let small = pdf::Limits {
            max_bytes: 10,
            ..pdf::Limits::default()
        };
        let block = format!(
            "{{\"type\":\"document\",\"source\":{{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"{}\"}}}}",
            b64(&pdf("x"))
        );
        let c = Ctx {
            limits: small,
            fetch: None,
        };
        let e = expand_messages(&mut messages_body(&block), &c).unwrap_err();
        assert!(e.contains("--pdf-max-mb"), "{e}");
        // a type the server cannot read stays for the translation to refuse
        let block = "{\"type\":\"document\",\"source\":{\"type\":\"base64\",\"media_type\":\"application/msword\",\"data\":\"AAAA\"}}";
        let mut body = messages_body(block);
        expand_messages(&mut body, &ctx()).unwrap();
        let e = super::super::anthropic::translate(&body).unwrap_err();
        assert!(
            e.contains("application/msword documents are not supported"),
            "{e}"
        );
        let e = expand_messages(
            &mut messages_body(&block.replace("AAAA", "!!")).clone(),
            &ctx(),
        );
        assert!(e.is_ok()); // left alone: not a PDF
    }

    /// A chat request with `n` file parts, each the data URL `data`.
    fn chat_files(n: usize, data: &str) -> JsonValue {
        let part = format!("{{\"type\":\"file\",\"file\":{{\"file_data\":\"{data}\"}}}}");
        parse_json(&format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":[{}]}}]}}",
            vec![part; n].join(",")
        ))
        .unwrap()
    }

    #[test]
    fn the_documents_of_a_request_share_one_budget() {
        // Codex repro: every document had fresh budgets, so many PDFs that each fit added up
        // without a bound. Now they are counted together: the documents a request carries...
        let one = format!("data:application/pdf;base64,{}", b64(&pdf("Small PDF")));
        expand_chat(&mut chat_files(100, &one), &ctx()).unwrap();
        let e = expand_chat(&mut chat_files(101, &one), &ctx()).unwrap_err();
        assert_eq!(
            e,
            "messages[0].content[100].file: the request's documents are too large together: there are more than 100 documents (--pdf-max-docs)"
        );
        // ... the bytes of text they keep (each one under its own text cap) ...
        let words = "word ".repeat(400);
        let text = format!("--- Page 1 ---\n{}\n", words.trim()).len() as u64;
        let longer = format!("data:application/pdf;base64,{}", b64(&pdf(&words)));
        let c = Ctx {
            limits: pdf::Limits {
                max_decoded: 16_384,
                ..pdf::Limits::default()
            },
            fetch: None,
        };
        let fit = (16_384 / text) as usize;
        assert_eq!(fit, 8);
        expand_chat(&mut chat_files(fit, &longer), &c).unwrap();
        let e = expand_chat(&mut chat_files(fit + 1, &longer), &c).unwrap_err();
        assert!(
            e.starts_with(&format!("messages[0].content[{fit}].file: "))
                && e.ends_with("their text is more than 16384 bytes (--pdf-max-inflate-mb)"),
            "{e}"
        );
        // ... and the heap the expansion holds across them: small PDFs whose text is large (a
        // form drawn 200 times), each far under the heap cap on its own
        let c = Ctx {
            limits: pdf::Limits {
                max_heap: 1 << 20,
                ..pdf::Limits::default()
            },
            fetch: None,
        };
        let drawn = pdf_drawn(200);
        assert!(drawn.len() < 4096, "{}", drawn.len());
        let big = format!("data:application/pdf;base64,{}", b64(&drawn));
        let mut two = chat_files(2, &big);
        expand_chat(&mut two, &c).unwrap();
        assert!(two.dump().len() > 400_000);
        let e = expand_chat(&mut chat_files(8, &big), &c).unwrap_err();
        assert!(
            e.contains("more than 1048576 bytes of memory (--pdf-max-heap-mb)")
                && (e.contains("the request's documents") || e.contains("they hold")),
            "{e}"
        );
        // the Messages expansion is charged the same way
        let block = format!(
            "{{\"type\":\"document\",\"source\":{{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"{}\"}}}}",
            b64(&pdf("Small PDF"))
        );
        let body = |n: usize| {
            parse_json(&format!(
                "{{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{{\"role\":\"user\",\"content\":[{}]}}]}}",
                vec![block.clone(); n].join(",")
            ))
            .unwrap()
        };
        expand_messages(&mut body(100), &ctx()).unwrap();
        let e = expand_messages(&mut body(101), &ctx()).unwrap_err();
        assert!(
            e.starts_with("messages.0.content.100: ") && e.ends_with("(--pdf-max-docs)"),
            "{e}"
        );
        assert_eq!(crate::heap_meter::reading(), None);
    }

    #[test]
    fn releasing_the_request_body_does_not_hide_the_text_kept() {
        // Codex repro (#136): each file part's base64 string was allocated before the
        // expansion started counting; freeing it as the part became text took its size from
        // the count, so twenty 128 KiB text files kept 2,621,440 bytes under a 1 MiB cap
        let c = Ctx {
            limits: pdf::Limits {
                max_heap: 1 << 20,
                ..pdf::Limits::default()
            },
            fetch: None,
        };
        let text = "x".repeat(128 << 10);
        let one = format!("data:text/plain;base64,{}", b64(text.as_bytes()));
        let mut body = chat_files(20, &one);
        let e = expand_chat(&mut body, &c).unwrap_err();
        assert!(
            e.contains("they hold more than 1048576 bytes of memory (--pdf-max-heap-mb)"),
            "{e}"
        );
        // under the cap they are read
        let mut body = chat_files(4, &one);
        expand_chat(&mut body, &c).unwrap();
        assert!(body.dump().len() > 4 * (128 << 10));
        assert_eq!(crate::heap_meter::reading(), None);
    }

    #[test]
    fn chat_file_parts_become_text() {
        let body = |data: &str| {
            parse_json(&format!(
                "{{\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"file\",\"file\":{{\"filename\":\"notes.pdf\",\"file_data\":\"{data}\"}}}},{{\"type\":\"text\",\"text\":\"What does it say?\"}}]}}]}}"
            ))
            .unwrap()
        };
        let mut b = body(&format!(
            "data:application/pdf;base64,{}",
            b64(&pdf("Chat PDF"))
        ));
        expand_chat(&mut b, &ctx()).unwrap();
        let part = &b.get("messages").unwrap().as_array().unwrap()[0]
            .get("content")
            .unwrap()
            .as_array()
            .unwrap()[0];
        assert_eq!(part.get("type").and_then(JsonValue::as_str), Some("text"));
        assert_eq!(
            part.get("text").and_then(JsonValue::as_str),
            Some("notes.pdf\n\n--- Page 1 ---\nChat PDF\n")
        );
        // plain text files, and bare base64 PDFs
        let mut b = body(&format!("data:text/plain;base64,{}", b64(b"just text")));
        expand_chat(&mut b, &ctx()).unwrap();
        assert!(
            b.dump().contains("notes.pdf\\n\\njust text"),
            "{}",
            b.dump()
        );
        let mut b = body(&b64(&pdf("Bare")));
        expand_chat(&mut b, &ctx()).unwrap();
        assert!(b.dump().contains("Bare"), "{}", b.dump());
        // refusals name the part
        let e = expand_chat(&mut body("data:application/pdf,notbase64"), &ctx()).unwrap_err();
        assert!(
            e.starts_with("messages[0].content[0].file.file_data"),
            "{e}"
        );
        let e = expand_chat(&mut body("data:image/png;base64,AAAA"), &ctx()).unwrap_err();
        assert!(e.contains("image/png documents are not supported"), "{e}");
        // a part that names an uploaded file is left for the renderer
        let mut b = parse_json(
            "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"file\",\"file\":{\"file_id\":\"file-1\"}}]}]}",
        )
        .unwrap();
        let before = b.dump();
        expand_chat(&mut b, &ctx()).unwrap();
        assert_eq!(b.dump(), before);
    }
}
