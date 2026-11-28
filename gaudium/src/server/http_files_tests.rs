//! The Files API and `file_id` sources over real sockets (a child of `http_tests`, whose
//! harness it uses).

use super::*;

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("gaudium-http-files-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

/// `h` with a file store in `dir` (`quota` bytes, files up to `cap`) and the API key `key`.
fn with_files(h: Harness, dir: &Path, quota: u64, cap: u64, key: Option<&str>) -> Harness {
    let Harness {
        state,
        replies,
        prompts,
        images,
        tier,
    } = h;
    let mut st = Arc::try_unwrap(state)
        .ok()
        .expect("the harness owns its state");
    st.files = Some(Arc::new(
        file_store::FileStore::open(dir, quota, cap).unwrap(),
    ));
    st.cfg.api_key = key.map(str::to_string);
    Harness {
        state: Arc::new(st),
        replies,
        prompts,
        images,
        tier,
    }
}

const BOUNDARY: &str = "----gaudiumtest7d1";

/// A multipart body of `(field, file name, content type, bytes)` parts.
fn multipart(parts: &[(&str, Option<&str>, Option<&str>, &[u8])]) -> Vec<u8> {
    let mut b = Vec::new();
    for (name, filename, ct, data) in parts {
        b.extend_from_slice(
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"").as_bytes(),
        );
        if let Some(f) = filename {
            b.extend_from_slice(format!("; filename=\"{f}\"").as_bytes());
        }
        b.extend_from_slice(b"\r\n");
        if let Some(t) = ct {
            b.extend_from_slice(format!("Content-Type: {t}\r\n").as_bytes());
        }
        b.extend_from_slice(b"\r\n");
        b.extend_from_slice(data);
        b.extend_from_slice(b"\r\n");
    }
    b.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    b
}

/// One request with `headers` (lines without the final CRLF) and `body`: `(status, head, body)`.
fn send(
    state: &Arc<AppState>,
    method: &str,
    path: &str,
    headers: &str,
    body: &[u8],
) -> (u16, String, String) {
    let addr = serve(state, 1);
    let mut c = TcpStream::connect(addr).unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n");
    if !headers.is_empty() {
        req.push_str(headers);
        req.push_str("\r\n");
    }
    req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    c.write_all(req.as_bytes()).unwrap();
    c.write_all(body).unwrap();
    let mut raw = Vec::new();
    let _ = c.read_to_end(&mut raw);
    split_response(&raw)
}

fn upload(
    state: &Arc<AppState>,
    headers: &str,
    parts: &[(&str, Option<&str>, Option<&str>, &[u8])],
) -> (u16, JsonValue) {
    let ct = format!("Content-Type: multipart/form-data; boundary={BOUNDARY}");
    let h = if headers.is_empty() {
        ct
    } else {
        format!("{headers}\r\n{ct}")
    };
    let (status, _, body) = send(state, "POST", "/v1/files", &h, &multipart(parts));
    (
        status,
        json::parse_json(&body).unwrap_or_else(|e| panic!("{e}: {body}")),
    )
}

fn get_json(state: &Arc<AppState>, method: &str, path: &str, headers: &str) -> (u16, JsonValue) {
    let (status, _, body) = send(state, method, path, headers, b"");
    (
        status,
        json::parse_json(&body).unwrap_or_else(|e| panic!("{e}: {body}")),
    )
}

const ANTHROPIC: &str = "anthropic-version: 2023-06-01";

#[test]
fn files_api_in_the_openai_shape() {
    let dir = tmp_dir("openai");
    let h = with_files(harness(), &dir, 1 << 20, 1 << 16, None);
    let st = &h.state;
    let line = b"{\"custom_id\":\"a\"}\n";
    let (status, f) = upload(
        st,
        "",
        &[
            ("purpose", None, None, b"batch"),
            (
                "file",
                Some("in.jsonl"),
                Some("application/octet-stream"),
                line,
            ),
        ],
    );
    assert_eq!(status, 200, "{}", f.dump());
    let id = s(&f, &["id"]).to_string();
    assert!(id.starts_with("file-"));
    assert_eq!(
        (
            s(&f, &["object"]),
            s(&f, &["filename"]),
            s(&f, &["purpose"])
        ),
        ("file", "in.jsonl", "batch")
    );
    assert_eq!(at(&f, &["bytes"]).as_u64(), Some(line.len() as u64));
    assert!(at(&f, &["created_at"]).as_u64().unwrap() > 1_700_000_000);
    // a second file, purpose first or last: both work
    let (status, g) = upload(
        st,
        "",
        &[
            ("file", Some("notes.txt"), None, b"hello"),
            ("purpose", None, None, b"user_data"),
        ],
    );
    assert_eq!(status, 200, "{}", g.dump());
    let id2 = s(&g, &["id"]).to_string();

    // list: newest first by default, purpose filter, paging by `after`
    let (status, l) = get_json(st, "GET", "/v1/files", "");
    assert_eq!(status, 200);
    assert_eq!(s(&l, &["object"]), "list");
    let ids: Vec<&str> = at(&l, &["data"])
        .as_array()
        .unwrap()
        .iter()
        .map(|m| s(m, &["id"]))
        .collect();
    assert_eq!(ids, [id2.as_str(), id.as_str()]);
    assert_eq!(at(&l, &["has_more"]).as_bool(), Some(false));
    let (_, l) = get_json(st, "GET", "/v1/files?purpose=batch", "");
    assert_eq!(at(&l, &["data"]).as_array().unwrap().len(), 1);
    let (_, l) = get_json(st, "GET", "/v1/files?order=asc&limit=1", "");
    assert_eq!(
        (s(&l, &["first_id"]), s(&l, &["last_id"])),
        (id.as_str(), id.as_str())
    );
    assert_eq!(at(&l, &["has_more"]).as_bool(), Some(true));
    let (_, l) = get_json(
        st,
        "GET",
        &format!("/v1/files?order=asc&limit=1&after={id}"),
        "",
    );
    assert_eq!(s(&l, &["first_id"]), id2);
    assert_eq!(at(&l, &["has_more"]).as_bool(), Some(false));

    // retrieve, content, delete
    let (status, m) = get_json(st, "GET", &format!("/v1/files/{id}"), "");
    assert_eq!((status, s(&m, &["id"])), (200, id.as_str()));
    let (status, head, body) = send(st, "GET", &format!("/v1/files/{id}/content"), "", b"");
    assert_eq!(status, 200);
    assert!(head.contains("content-type: application/jsonl"), "{head}");
    assert_eq!(body.as_bytes(), line);
    let (status, d) = get_json(st, "DELETE", &format!("/v1/files/{id}"), "");
    assert_eq!(status, 200);
    assert_eq!(
        (
            s(&d, &["id"]),
            s(&d, &["object"]),
            at(&d, &["deleted"]).as_bool()
        ),
        (id.as_str(), "file", Some(true))
    );
    let (status, e) = get_json(st, "GET", &format!("/v1/files/{id}"), "");
    assert_eq!(status, 404, "{}", e.dump());
    let (status, _) = get_json(st, "DELETE", &format!("/v1/files/{id}"), "");
    assert_eq!(status, 404);

    // refusals: no purpose, an unknown one, an expiry, no file part, not multipart
    let (status, e) = upload(st, "", &[("file", Some("x"), None, b"x")]);
    assert_eq!(status, 400);
    assert!(s(&e, &["error", "message"]).contains("'purpose' is required"));
    let (status, e) = upload(
        st,
        "",
        &[
            ("purpose", None, None, b"nonsense"),
            ("file", Some("x"), None, b"x"),
        ],
    );
    assert_eq!(status, 400);
    assert!(s(&e, &["error", "message"]).contains("unknown purpose"));
    let (status, e) = upload(
        st,
        "",
        &[
            ("purpose", None, None, b"batch"),
            ("expires_after[seconds]", None, None, b"3600"),
            ("file", Some("x"), None, b"x"),
        ],
    );
    assert_eq!(status, 400);
    assert!(s(&e, &["error", "message"]).contains("expiry is not supported"));
    let (status, e) = upload(st, "", &[("purpose", None, None, b"batch")]);
    assert_eq!(status, 400);
    assert!(s(&e, &["error", "message"]).contains("needs a 'file' part"));
    let (status, _, body) = send(
        st,
        "POST",
        "/v1/files",
        "Content-Type: application/json",
        b"{}",
    );
    assert_eq!(status, 400);
    assert!(body.contains("multipart/form-data"), "{body}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn files_api_in_the_messages_shape() {
    let dir = tmp_dir("messages");
    let h = with_files(harness(), &dir, 1 << 20, 1 << 16, None);
    let st = &h.state;
    let mut ids = Vec::new();
    for (i, name) in ["a.txt", "b.txt", "c.txt"].iter().enumerate() {
        let (status, f) = upload(
            st,
            ANTHROPIC,
            &[(
                "file",
                Some(name),
                Some("text/plain"),
                format!("text {i}").as_bytes(),
            )],
        );
        assert_eq!(status, 200, "{}", f.dump());
        assert_eq!(
            (
                s(&f, &["type"]),
                s(&f, &["filename"]),
                s(&f, &["mime_type"])
            ),
            ("file", *name, "text/plain")
        );
        assert_eq!(at(&f, &["size_bytes"]).as_u64(), Some(6));
        assert_eq!(at(&f, &["downloadable"]).as_bool(), Some(false));
        let created = s(&f, &["created_at"]);
        assert!(
            created.len() == 20 && created.ends_with('Z') && created.as_bytes()[10] == b'T',
            "{created}"
        );
        ids.push(s(&f, &["id"]).to_string());
    }
    // the stable list: newest first, `limit` and `page`
    let (status, l) = get_json(st, "GET", "/v1/files?limit=2", ANTHROPIC);
    assert_eq!(status, 200);
    let got: Vec<&str> = at(&l, &["data"])
        .as_array()
        .unwrap()
        .iter()
        .map(|m| s(m, &["id"]))
        .collect();
    assert_eq!(got, [ids[2].as_str(), ids[1].as_str()]);
    let next = s(&l, &["next_page"]).to_string();
    assert_eq!(next, format!("page_{}", ids[1]));
    let (_, l) = get_json(
        st,
        "GET",
        &format!("/v1/files?limit=2&page={next}"),
        ANTHROPIC,
    );
    let got: Vec<&str> = at(&l, &["data"])
        .as_array()
        .unwrap()
        .iter()
        .map(|m| s(m, &["id"]))
        .collect();
    assert_eq!(got, [ids[0].as_str()]);
    assert_eq!(at(&l, &["next_page"]), &JsonValue::Null);
    let (_, l) = get_json(
        st,
        "GET",
        &format!("/v1/files?ids={},{}", ids[0], ids[2]),
        ANTHROPIC,
    );
    assert_eq!(at(&l, &["data"]).as_array().unwrap().len(), 2);
    // the legacy beta pages by ids and says has_more
    let legacy = "anthropic-beta: files-api-2025-04-14";
    let (_, l) = get_json(st, "GET", "/v1/files?limit=1", legacy);
    assert_eq!(
        (s(&l, &["first_id"]), at(&l, &["has_more"]).as_bool()),
        (ids[2].as_str(), Some(true))
    );
    let (_, l) = get_json(
        st,
        "GET",
        &format!("/v1/files?limit=1&before_id={}", ids[0]),
        legacy,
    );
    assert_eq!(s(&l, &["first_id"]), ids[1]);
    assert_eq!(at(&l, &["has_more"]).as_bool(), Some(true));
    // metadata; uploads are not downloadable in this shape, the server's own files are
    let (status, m) = get_json(st, "GET", &format!("/v1/files/{}", ids[0]), ANTHROPIC);
    assert_eq!((status, s(&m, &["id"])), (200, ids[0].as_str()));
    let (status, e) = get_json(
        st,
        "GET",
        &format!("/v1/files/{}/content", ids[0]),
        ANTHROPIC,
    );
    assert_eq!(status, 400);
    assert_eq!(
        s(&e, &["error", "details", "error_code"]),
        "file_not_downloadable"
    );
    let own = st
        .files
        .as_ref()
        .unwrap()
        .create(
            &file_store::Owner::everyone(),
            "out.jsonl",
            None,
            "batch_output",
            true,
            b"{}\n",
        )
        .unwrap();
    let (status, _, body) = send(
        st,
        "GET",
        &format!("/v1/files/{}/content", own.id),
        ANTHROPIC,
        b"",
    );
    assert_eq!((status, body.as_str()), (200, "{}\n"));
    // delete
    let (status, d) = get_json(st, "DELETE", &format!("/v1/files/{}", ids[0]), ANTHROPIC);
    assert_eq!(status, 200);
    assert_eq!(
        (s(&d, &["id"]), s(&d, &["type"])),
        (ids[0].as_str(), "file_deleted")
    );
    let (status, e) = get_json(st, "GET", &format!("/v1/files/{}", ids[0]), ANTHROPIC);
    assert_eq!(
        (status, s(&e, &["type"]), s(&e, &["error", "type"])),
        (404, "error", "not_found_error")
    );
    // one store for both shapes
    let (status, m) = get_json(st, "GET", &format!("/v1/files/{}", ids[1]), "");
    assert_eq!(
        (status, s(&m, &["object"]), s(&m, &["purpose"])),
        (200, "file", "user_data")
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn uploads_past_the_cap_or_the_quota_are_refused_with_their_numbers() {
    let dir = tmp_dir("quota");
    let h = with_files(harness(), &dir, 20, 15, None);
    let st = &h.state;
    let part = |data: &'static [u8]| {
        [
            ("purpose", None, None, &b"batch"[..]),
            ("file", Some("f"), None, data),
        ]
    };
    let (status, e) = upload(st, "", &part(&[b'x'; 16]));
    assert_eq!(status, 413, "{}", e.dump());
    assert!(s(&e, &["error", "message"]).contains("limit of 15 bytes per file"));
    let (status, _) = upload(st, "", &part(&[b'x'; 15]));
    assert_eq!(status, 200);
    let (status, e) = upload(st, "", &part(&[b'x'; 6]));
    assert_eq!(status, 400);
    let msg = s(&e, &["error", "message"]);
    assert!(
        msg.contains("more than 5 bytes")
            && msg.contains(
                "15 of 20 bytes are in use or reserved by uploads in progress, and 5 are free"
            ),
        "{msg}"
    );
    let (status, e) = upload(st, ANTHROPIC, &[("file", Some("f"), None, &[b'x'; 6])]);
    assert_eq!(status, 400);
    assert_eq!(
        s(&e, &["error", "details", "error_code"]),
        "storage_quota_exceeded"
    );
    // exactly the free bytes fit, sent as a chunked body with Expect: 100-continue
    let body = multipart(&part(&[b'y'; 5]));
    let addr = serve(st, 1);
    let mut c = TcpStream::connect(addr).unwrap();
    write!(
        c,
        "POST /v1/files HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary={BOUNDARY}\r\nExpect: 100-continue\r\nTransfer-Encoding: chunked\r\n\r\n"
    )
    .unwrap();
    for chunk in body.chunks(7) {
        write!(c, "{:x}\r\n", chunk.len()).unwrap();
        c.write_all(chunk).unwrap();
        c.write_all(b"\r\n").unwrap();
    }
    c.write_all(b"0\r\n\r\n").unwrap();
    let mut raw = Vec::new();
    let _ = c.read_to_end(&mut raw);
    let rest = raw
        .strip_prefix(b"HTTP/1.1 100 Continue\r\n\r\n".as_slice())
        .expect("a 100 Continue first");
    let (status, _, text) = split_response(rest);
    assert_eq!(status, 200, "{text}");
    assert_eq!(st.files.as_ref().unwrap().usage(), (20, 0, 20));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn files_belong_to_their_key_and_outlive_a_restart() {
    let dir = tmp_dir("owner");
    let h = with_files(harness(), &dir, 1 << 20, 1 << 16, Some("k1"));
    let st = &h.state;
    let key = "Authorization: Bearer k1";
    let (status, _) = upload(
        st,
        "",
        &[
            ("purpose", None, None, b"batch"),
            ("file", Some("f"), None, b"x"),
        ],
    );
    assert_eq!(status, 401);
    let (status, mine) = upload(
        st,
        key,
        &[
            ("purpose", None, None, b"batch"),
            ("file", Some("mine.txt"), None, b"mine"),
        ],
    );
    assert_eq!(status, 200);
    let mine = s(&mine, &["id"]).to_string();
    let theirs = st
        .files
        .as_ref()
        .unwrap()
        .create(
            &file_store::Owner::of_key("another key"),
            "theirs.txt",
            None,
            "batch",
            false,
            b"theirs",
        )
        .unwrap();
    for (m, p) in [
        ("GET", format!("/v1/files/{}", theirs.id)),
        ("GET", format!("/v1/files/{}/content", theirs.id)),
        ("DELETE", format!("/v1/files/{}", theirs.id)),
    ] {
        let (status, _, body) = send(st, m, &p, key, b"");
        assert_eq!(status, 404, "{m} {p}: {body}");
    }
    let (_, l) = get_json(st, "GET", "/v1/files", "x-api-key: k1");
    let ids: Vec<&str> = at(&l, &["data"])
        .as_array()
        .unwrap()
        .iter()
        .map(|m| s(m, &["id"]))
        .collect();
    assert_eq!(ids, [mine.as_str()]);
    // a restart: the store is read back from its directory
    drop(h);
    let h = with_files(harness(), &dir, 1 << 20, 1 << 16, Some("k1"));
    let (status, _, body) = send(
        &h.state,
        "GET",
        &format!("/v1/files/{mine}/content"),
        key,
        b"",
    );
    assert_eq!((status, body.as_str()), (200, "mine"));
    assert_eq!(h.state.files.as_ref().unwrap().usage().0, 4 + 6);
    let _ = fs::remove_dir_all(&dir);
}

/// The ids of the files a `GET /v1/files` with `headers` lists.
fn listed(st: &Arc<AppState>, headers: &str) -> Vec<String> {
    let (status, l) = get_json(st, "GET", "/v1/files", headers);
    assert_eq!(status, 200, "{headers}");
    at(&l, &["data"])
        .as_array()
        .unwrap()
        .iter()
        .map(|m| s(m, &["id"]).to_string())
        .collect()
}

#[test]
fn files_belong_to_the_key_that_let_the_request_in() {
    let dir = tmp_dir("credential");
    let h = with_files(harness(), &dir, 1 << 20, 1 << 16, Some("k1"));
    let st = &h.state;
    // the configured key as x-api-key, a Bearer token that is not it
    let both = "x-api-key: k1\r\nAuthorization: Bearer wrong";
    let (status, up) = upload(
        st,
        both,
        &[
            ("purpose", None, None, b"batch"),
            ("file", Some("mine.txt"), None, b"mine"),
        ],
    );
    assert_eq!(status, 200, "{up:?}");
    let mine = s(&up, &["id"]).to_string();
    // the key alone, in either header, sees, reads and deletes it
    for key in ["x-api-key: k1", "Authorization: Bearer k1", both] {
        assert_eq!(listed(st, key), [mine.as_str()], "{key}");
        let (status, _, body) = send(st, "GET", &format!("/v1/files/{mine}/content"), key, b"");
        assert_eq!((status, body.as_str()), (200, "mine"), "{key}");
    }
    let (status, _, body) = send(
        st,
        "DELETE",
        &format!("/v1/files/{mine}"),
        "x-api-key: k1",
        b"",
    );
    assert_eq!(status, 200, "{body}");
    assert!(listed(st, "x-api-key: k1").is_empty());

    // files under a key the server no longer has (rotated), and under no key
    let store = st.files.as_ref().unwrap();
    let old = store
        .create(
            &file_store::Owner::of_key("old"),
            "old.txt",
            None,
            "batch",
            false,
            b"old",
        )
        .unwrap();
    let open = store
        .create(
            &file_store::Owner::everyone(),
            "open.txt",
            None,
            "batch",
            false,
            b"open",
        )
        .unwrap();
    // a wrong key, alone or beside the configured one, never selects their namespace
    for wrong in [
        "Authorization: Bearer old",
        "x-api-key: old",
        "Authorization: Bearer ",
    ] {
        let (status, _) = upload(st, wrong, &[("file", Some("f"), None, b"x")]);
        assert_eq!(status, 401, "{wrong}");
        for id in [&old.id, &open.id] {
            for (m, p) in [
                ("GET", "/v1/files".to_string()),
                ("GET", format!("/v1/files/{id}")),
                ("GET", format!("/v1/files/{id}/content")),
                ("DELETE", format!("/v1/files/{id}")),
            ] {
                let (status, _, body) = send(st, m, &p, wrong, b"");
                assert_eq!(status, 401, "{wrong}: {m} {p}: {body}");
            }
        }
    }
    for mixed in [
        "x-api-key: k1\r\nAuthorization: Bearer old",
        "x-api-key: old\r\nAuthorization: Bearer k1",
    ] {
        assert!(listed(st, mixed).is_empty(), "{mixed}");
        for id in [&old.id, &open.id] {
            for (m, p) in [
                ("GET", format!("/v1/files/{id}")),
                ("GET", format!("/v1/files/{id}/content")),
                ("DELETE", format!("/v1/files/{id}")),
            ] {
                let (status, _, body) = send(st, m, &p, mixed, b"");
                assert_eq!(status, 404, "{mixed}: {m} {p}: {body}");
            }
        }
        let (status, up) = upload(
            st,
            mixed,
            &[
                ("purpose", None, None, b"batch"),
                ("file", Some("k1.txt"), None, b"k1"),
            ],
        );
        assert_eq!(status, 200, "{up:?}");
        let id = s(&up, &["id"]).to_string();
        assert_eq!(listed(st, "x-api-key: k1"), [id.as_str()], "{mixed}");
        let (status, _, _) = send(
            st,
            "DELETE",
            &format!("/v1/files/{id}"),
            "x-api-key: k1",
            b"",
        );
        assert_eq!(status, 200);
    }
    // the files under the old key and under none are still there, for their owners
    assert_eq!(
        store
            .get(&file_store::Owner::of_key("old"), &old.id)
            .unwrap()
            .id,
        old.id
    );
    assert_eq!(
        store
            .get(&file_store::Owner::everyone(), &open.id)
            .unwrap()
            .id,
        open.id
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_files_api_is_off_without_a_directory() {
    let h = harness();
    let (status, e) = upload(
        &h.state,
        "",
        &[
            ("purpose", None, None, b"batch"),
            ("file", Some("f"), None, b"x"),
        ],
    );
    assert_eq!(status, 404);
    assert!(s(&e, &["error", "message"]).contains("--files-dir"));
    let (status, _, body) = send(&h.state, "GET", "/v1/files", ANTHROPIC, b"");
    assert_eq!(status, 404);
    assert!(
        body.contains("--files-dir") && body.contains("\"type\":\"error\""),
        "{body}"
    );
    // a client waiting for 100 Continue gets the refusal without sending its body
    let addr = serve(&h.state, 1);
    let mut c = TcpStream::connect(addr).unwrap();
    c.set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    write!(
        c,
        "POST /v1/files HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary={BOUNDARY}\r\nExpect: 100-continue\r\nContent-Length: 100000\r\n\r\n"
    )
    .unwrap();
    let mut raw = Vec::new();
    let _ = c.read_to_end(&mut raw);
    let (status, _, text) = split_response(&raw);
    assert_eq!(status, 404, "{text}");
    let req = r#"{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":"file-000000000000000000000000"}}]}]}"#;
    let (status, text) = h.post("/v1/messages", req);
    assert_eq!(status, 400);
    assert!(
        text.contains("file_id sources need the Files API"),
        "{text}"
    );
}

/// The bytes of the synthetic image `image_url` makes.
fn image_bytes(w: usize, hgt: usize, k: usize) -> Vec<u8> {
    let url = image_url(w, hgt, k);
    crate::vision::media::base64_decode(url.split_once(',').unwrap().1).unwrap()
}

#[test]
fn file_ids_in_messages_image_and_document_blocks() {
    let dir = tmp_dir("msg-sources");
    let h = with_files(vision_harness(4096, 4096), &dir, 1 << 22, 1 << 21, None);
    let st = &h.state;
    let img = image_bytes(256, 256, 3);
    let (status, f) = upload(
        st,
        ANTHROPIC,
        &[(
            "file",
            Some("shot.ppm"),
            Some("image/x-portable-pixmap"),
            &img,
        )],
    );
    assert_eq!(status, 200, "{}", f.dump());
    let img_id = s(&f, &["id"]).to_string();
    let (_, f) = upload(
        st,
        ANTHROPIC,
        &[(
            "file",
            Some("notes.txt"),
            Some("text/plain"),
            b"The launch code is 1234.",
        )],
    );
    let doc_id = s(&f, &["id"]).to_string();
    let (_, f) = upload(
        st,
        ANTHROPIC,
        &[("file", Some("paper.pdf"), None, b"%PDF-1.7 not really")],
    );
    let pdf_id = s(&f, &["id"]).to_string();

    h.script("Seen.");
    let body = format!(
        r#"{{"model":"vl","max_tokens":64,"messages":[{{"role":"user","content":[{{"type":"document","title":"Notes","source":{{"type":"file","file_id":"{doc_id}"}}}},{{"type":"image","source":{{"type":"file","file_id":"{img_id}"}}}},{{"type":"text","text":"What is this?"}}]}}]}}"#
    );
    let (status, text) = h.post("/v1/messages", &body);
    assert_eq!(status, 200, "{text}");
    let prompt = h.last_prompt();
    assert!(
        prompt.contains("Notes\n\nThe launch code is 1234.<|vision_start|>"),
        "{prompt}"
    );
    assert_eq!(prompt.matches("<|image_pad|>").count(), 64);
    // the same tower input as the image sent inline
    assert_eq!(
        h.images.lock().unwrap().last().unwrap(),
        &vec![(16, 16, 256 * 1536)]
    );
    let (status, text) = h.post("/v1/messages/count_tokens", &body);
    assert_eq!(status, 200, "{text}");

    // a text file is not an image; a PDF takes the path of an inline PDF; unknown ids are refused
    let as_image = format!(
        r#"{{"model":"vl","max_tokens":8,"messages":[{{"role":"user","content":[{{"type":"image","source":{{"type":"file","file_id":"{doc_id}"}}}}]}}]}}"#
    );
    let (status, text) = h.post("/v1/messages", &as_image);
    assert_eq!(status, 400);
    assert!(text.contains("is text/plain, not an image"), "{text}");
    let pdf = format!(
        r#"{{"model":"vl","max_tokens":8,"messages":[{{"role":"user","content":[{{"type":"document","source":{{"type":"file","file_id":"{pdf_id}"}}}}]}}]}}"#
    );
    let inline_pdf = r#"{"model":"vl","max_tokens":8,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0xLjcgbm90IHJlYWxseQ=="}}]}]}"#;
    let (status, by_id) = h.post("/v1/messages", &pdf);
    let (status_inline, inline) = h.post("/v1/messages", inline_pdf);
    assert_eq!(status, status_inline, "{by_id} / {inline}");
    if status != 200 {
        assert_eq!(by_id, inline);
    }
    let missing = r#"{"model":"vl","max_tokens":8,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"file","file_id":"file-00000000ffff000000000000"}}]}]}"#;
    let (status, text) = h.post("/v1/messages", missing);
    assert_eq!(status, 400);
    assert!(text.contains("not found"), "{text}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn file_ids_in_chat_file_parts() {
    let dir = tmp_dir("chat-sources");
    let h = with_files(vision_harness(4096, 4096), &dir, 1 << 22, 1 << 21, None);
    let st = &h.state;
    let (_, f) = upload(
        st,
        "",
        &[
            ("purpose", None, None, b"user_data"),
            ("file", Some("notes.txt"), None, b"Remember the milk."),
        ],
    );
    let txt = s(&f, &["id"]).to_string();
    let img = image_bytes(256, 256, 1);
    let (_, f) = upload(
        st,
        "",
        &[
            ("purpose", None, None, b"vision"),
            ("file", Some("x.ppm"), Some("image/x-portable-pixmap"), &img),
        ],
    );
    let pic = s(&f, &["id"]).to_string();
    h.script("Noted.");
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"file","file":{{"file_id":"{txt}"}}}},{{"type":"file","file":{{"file_id":"{pic}"}}}},{{"type":"text","text":" What now?"}}]}}]}}"#
    );
    let (status, text) = h.post("/v1/chat/completions", &body);
    assert_eq!(status, 200, "{text}");
    let prompt = h.last_prompt();
    assert!(
        prompt.contains("Remember the milk.<|vision_start|>"),
        "{prompt}"
    );
    assert!(prompt.contains("<|vision_end|> What now?"), "{prompt}");
    assert_eq!(
        h.images.lock().unwrap().last().unwrap(),
        &vec![(16, 16, 256 * 1536)]
    );
    // an id of another shape's upload works too; an unknown one is refused
    let (_, f) = upload(
        st,
        ANTHROPIC,
        &[("file", Some("m.txt"), None, b"From the other shape.")],
    );
    let other = s(&f, &["id"]).to_string();
    h.script("Ok.");
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"file","file":{{"file_id":"{other}"}}}}]}}]}}"#
    );
    let (status, text) = h.post("/v1/chat/completions", &body);
    assert_eq!(status, 200, "{text}");
    assert!(h.last_prompt().contains("From the other shape."));
    let body = r#"{"messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"file-0000000fffff000000000000"}}]}]}"#;
    let (status, text) = h.post("/v1/chat/completions", body);
    assert_eq!(status, 400);
    assert!(text.contains("not found"), "{text}");
    let _ = fs::remove_dir_all(&dir);
}
