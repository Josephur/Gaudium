//! The MCP proxy over real sockets: a request goes through the server's HTTP parser and the
//! proxy's route handler to a throwaway target server on a loopback port (plain or TLS), and the
//! answer comes back through the proxy. Nothing leaves the machine: names resolve through a
//! resolver of the test, and addresses that stand for "public" ones are refused before any
//! connection is made.

use std::net::TcpListener;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use super::*;
use crate::tls::test_util::test_cert;
use crate::tls::TlsServer;

// ------------------------------------------------------------------------------------------------
// A throwaway target server
// ------------------------------------------------------------------------------------------------

trait Duplex: Read + Write {}
impl<T: Read + Write> Duplex for T {}

/// One request a target received.
#[derive(Debug, Clone, Default)]
struct Seen {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|h| h.0 == name)
            .map(|h| h.1.as_str())
    }
}

struct Target0 {
    port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Target0 {
    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }
    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

fn read_request(s: &mut dyn Duplex) -> Option<Seen> {
    let mut buf = Vec::new();
    let end = loop {
        if let Some(end) = find_head_end(&buf) {
            break end;
        }
        let mut b = [0u8; 4096];
        match s.read(&mut b) {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&b[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let mut lines = head.lines();
    let mut first = lines.next()?.split(' ');
    let mut seen = Seen {
        method: first.next()?.to_string(),
        path: first.next()?.to_string(),
        ..Seen::default()
    };
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            seen.headers
                .push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let len: usize = seen
        .header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[end..].to_vec();
    while body.len() < len {
        let mut b = [0u8; 4096];
        match s.read(&mut b) {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&b[..n]),
        }
    }
    seen.body = body;
    Some(seen)
}

/// A target on a loopback port. `answer` writes the response to each request it receives.
fn target<F>(tls: Option<TlsServer>, answer: F) -> Target0
where
    F: Fn(&Seen, &mut dyn Duplex) + Send + 'static,
{
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    thread::spawn(move || {
        for tcp in l.incoming() {
            let Ok(tcp) = tcp else { break };
            let mut plain;
            let mut secure;
            let s: &mut dyn Duplex = match &tls {
                None => {
                    plain = tcp;
                    &mut plain
                }
                Some(t) => match t.accept(tcp) {
                    Ok(x) => {
                        secure = x;
                        &mut secure
                    }
                    Err(_) => continue,
                },
            };
            if let Some(req) = read_request(s) {
                log.lock().unwrap().push(req.clone());
                answer(&req, s);
            }
        }
    });
    Target0 { port, seen }
}

fn ok_json(s: &mut dyn Duplex, extra_headers: &str, body: &str) {
    let _ = write!(
        s,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{extra_headers}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
}

// ------------------------------------------------------------------------------------------------
// The proxy under test and its client
// ------------------------------------------------------------------------------------------------

fn options(allow: &[&str], allow_private: &[&str]) -> ProxyOptions {
    ProxyOptions {
        enabled: true,
        allow: allow.iter().map(|s| s.to_string()).collect(),
        allow_private: allow_private.iter().map(|s| s.to_string()).collect(),
        ..ProxyOptions::default()
    }
}

/// A proxy with the given lists and an API key set on the server (so any list is accepted).
fn proxy(allow: &[&str], allow_private: &[&str]) -> McpProxy {
    McpProxy::new(&options(allow, allow_private), true).unwrap()
}

/// Names the test resolves itself; any other name fails to resolve.
fn resolve_as(p: &mut McpProxy, table: &[(&str, &[&str])]) {
    let table: Vec<(String, Vec<IpAddr>)> = table
        .iter()
        .map(|(n, a)| {
            (
                n.to_string(),
                a.iter().map(|x| x.parse().unwrap()).collect(),
            )
        })
        .collect();
    p.resolver = Box::new(move |host, _| {
        table
            .iter()
            .find(|(n, _)| n == host)
            .map(|(_, a)| a.clone())
            .ok_or_else(|| io::Error::other("no such name"))
    });
}

/// What came back from the proxy.
#[derive(Debug)]
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    /// The body ended as its framing says (a cut answer does not).
    complete: bool,
}

impl Reply {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|h| h.0 == name)
            .map(|h| h.1.as_str())
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    /// The message of the proxy's own JSON error.
    fn error(&self) -> String {
        let v = crate::json::parse_json(&self.text())
            .unwrap_or_else(|e| panic!("{e}: {}", self.text()));
        v.get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or_else(|| panic!("no error message in {}", self.text()))
            .to_string()
    }
}

fn parse_reply(raw: &[u8]) -> Reply {
    let end = find_head_end(raw).expect("a response head");
    let head = String::from_utf8_lossy(&raw[..end]).into_owned();
    let mut lines = head.lines();
    let status = lines
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|s| s.parse().ok())
        .expect("a status");
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let chunked = headers
        .iter()
        .any(|h| h.0 == "transfer-encoding" && h.1.contains("chunked"));
    let mut rest = &raw[end..];
    if !chunked {
        let len: usize = headers
            .iter()
            .find(|h| h.0 == "content-length")
            .and_then(|h| h.1.parse().ok())
            .unwrap_or(rest.len());
        return Reply {
            status,
            headers,
            complete: rest.len() >= len,
            body: rest.to_vec(),
        };
    }
    let mut body = Vec::new();
    let mut complete = false;
    while let Some(eol) = rest.windows(2).position(|w| w == b"\r\n") {
        let Ok(n) = usize::from_str_radix(String::from_utf8_lossy(&rest[..eol]).trim(), 16) else {
            break;
        };
        if n == 0 {
            complete = true;
            break;
        }
        if rest.len() < eol + 2 + n + 2 {
            body.extend_from_slice(&rest[(eol + 2).min(rest.len())..]);
            break;
        }
        body.extend_from_slice(&rest[eol + 2..eol + 2 + n]);
        rest = &rest[eol + 2 + n + 2..];
    }
    Reply {
        status,
        headers,
        body,
        complete,
    }
}

/// Sends `raw` (a whole HTTP request) to the proxy's route handler through the server's own
/// request parser and returns the client's end of the connection.
fn open(p: &Arc<McpProxy>, raw: String, key_ok: bool) -> TcpStream {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let p = Arc::clone(p);
    thread::spawn(move || {
        let (tcp, _) = l.accept().unwrap();
        let mut conn =
            Conn::open(tcp, None, super::super::DEFAULT_HSTS_MAX_AGE).expect("a plain connection");
        let req = super::super::read_http_request(&mut conn)
            .unwrap()
            .expect("a request");
        p.handle(&req, key_ok, &mut conn);
    });
    let mut c = TcpStream::connect(addr).unwrap();
    c.write_all(raw.as_bytes()).unwrap();
    c
}

fn query(url: &str) -> String {
    let mut out = String::new();
    for b in url.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// One request through the proxy: `method` to `url`, with `headers` (whole lines) and `body`.
fn send(p: &Arc<McpProxy>, method: &str, url: &str, headers: &[&str], body: &str) -> Reply {
    let mut raw = format!(
        "{method} /cors-proxy?url={} HTTP/1.1\r\nHost: localhost:8090\r\n",
        query(url)
    );
    for h in headers {
        raw.push_str(h);
        raw.push_str("\r\n");
    }
    raw.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
    let mut c = open(p, raw, true);
    let mut out = Vec::new();
    let _ = c.read_to_end(&mut out);
    parse_reply(&out)
}

fn get(p: &Arc<McpProxy>, url: &str) -> Reply {
    send(p, "GET", url, &[], "")
}

// ------------------------------------------------------------------------------------------------
// Text handling: URLs, entries, headers
// ------------------------------------------------------------------------------------------------

#[test]
fn target_urls_are_parsed_and_the_odd_ones_refused() {
    let ok = |u: &str| parse_target(u, Ok(())).unwrap_or_else(|e| panic!("{u}: {e}"));
    let no = |u: &str| {
        parse_target(u, Ok(()))
            .err()
            .unwrap_or_else(|| panic!("{u} passed"))
    };

    let t = ok("http://Example.COM./mcp?x=1#frag");
    assert_eq!(
        (t.host.as_str(), t.port, t.path.as_str()),
        ("example.com", 80, "/mcp?x=1")
    );
    assert_eq!(t.literal, None);
    assert_eq!(t.authority(), "example.com");
    assert_eq!(t.shown(), "http://example.com/mcp");
    let t = ok("https://example.com:8443");
    assert_eq!((t.port, t.path.as_str()), (8443, "/"));
    assert_eq!(t.origin(), "https://example.com:8443");
    let t = ok("https://example.com?q=1");
    assert_eq!((t.port, t.path.as_str()), (443, "/?q=1"));
    let t = ok("http://192.0.2.7:8931/mcp");
    assert_eq!(t.literal, Some("192.0.2.7".parse().unwrap()));
    let t = ok("http://[2001:db8::1]:8931/mcp");
    assert_eq!(t.literal, Some("2001:db8::1".parse().unwrap()));
    assert_eq!(t.authority(), "[2001:db8::1]:8931");
    let t = ok("http://[::1]/");
    assert_eq!((t.port, t.authority()), (80, "[::1]".to_string()));
    // bytes outside ASCII travel percent-encoded
    assert_eq!(ok("http://example.com/caf\u{e9}").path, "/caf%C3%A9");

    assert!(no("ftp://example.com/").contains("http:// or https://"));
    assert!(no("file:///etc/passwd").contains("http:// or https://"));
    assert!(no("example.com/mcp").contains("http:// or https://"));
    assert!(no("http://user:pw@example.com/").contains("user name or password"));
    assert!(no("http://example.com@192.0.2.1/").contains("user name or password"));
    assert!(no("http:///path").contains("host"));
    assert!(no("http://exa mple.com/").contains("not allowed"));
    assert!(no("http://example.com/a\r\nX-Injected: 1").contains("not allowed"));
    assert!(no("http://example.com/\u{0}").contains("not allowed"));
    assert!(no("http://exa$mple.com/").contains("host"));
    assert!(no("http://2001:db8::1/").contains("host"));
    assert!(no("http://[192.0.2.1]/").contains("host"));
    assert!(no("http://[::1]x/").contains("host"));
    for port in ["0", "65536", "99999", "123456", "80a", "-1"] {
        assert!(
            no(&format!("http://example.com:{port}/")).contains("port"),
            "{port}"
        );
    }

    // https without the TLS library: refused, and the message says what is missing
    let e = parse_target("https://example.com/", Err("libssl is not there"))
        .err()
        .unwrap();
    assert!(
        e.contains("https targets cannot be reached: libssl is not there"),
        "{e}"
    );
    assert!(parse_target("http://example.com/", Err("libssl is not there")).is_ok());
}

#[test]
fn redirect_locations_resolve_against_the_target() {
    let base = parse_target("https://example.com:8443/a/b?x=1", Ok(())).unwrap();
    for (loc, want) in [
        ("http://other.example.com/x", "http://other.example.com/x"),
        ("//other.example.com/x", "https://other.example.com/x"),
        ("/root", "https://example.com:8443/root"),
        ("sibling", "https://example.com:8443/a/sibling"),
        // the query-only reference keeps the path, the fragment-only one path and query
        ("?x=2", "https://example.com:8443/a/b?x=2"),
        ("#frag", "https://example.com:8443/a/b?x=1#frag"),
        ("", "https://example.com:8443/a/b?x=1"),
        ("  /trimmed  ", "https://example.com:8443/trimmed"),
        ("../up/./here", "https://example.com:8443/up/here"),
        // a URL inside a query is not a scheme of the reference
        (
            "/go?to=http://other.example.com/",
            "https://example.com:8443/go?to=http://other.example.com/",
        ),
    ] {
        assert_eq!(resolve_location(&base, loc), want, "{loc:?}");
    }
    // the fragment never travels: the request goes to the base's path and query
    let next = parse_target(&resolve_location(&base, "#frag"), Ok(())).unwrap();
    assert_eq!(next.path, "/a/b?x=1");
    let base6 = parse_target("http://[2001:db8::1]:8931/mcp", Ok(())).unwrap();
    assert_eq!(
        resolve_location(&base6, "?s=1"),
        "http://[2001:db8::1]:8931/mcp?s=1"
    );

    // the examples of RFC 3986, section 5.4 (normal and abnormal), base http://a/b/c/d;p?q
    let base = parse_target("http://a/b/c/d;p?q", Ok(())).unwrap();
    for (loc, want) in [
        ("g:h", "g:h"),
        ("g", "http://a/b/c/g"),
        ("./g", "http://a/b/c/g"),
        ("g/", "http://a/b/c/g/"),
        ("/g", "http://a/g"),
        ("//g", "http://g"),
        ("?y", "http://a/b/c/d;p?y"),
        ("g?y", "http://a/b/c/g?y"),
        ("#s", "http://a/b/c/d;p?q#s"),
        ("g#s", "http://a/b/c/g#s"),
        ("g?y#s", "http://a/b/c/g?y#s"),
        (";x", "http://a/b/c/;x"),
        ("g;x", "http://a/b/c/g;x"),
        ("g;x?y#s", "http://a/b/c/g;x?y#s"),
        ("", "http://a/b/c/d;p?q"),
        (".", "http://a/b/c/"),
        ("./", "http://a/b/c/"),
        ("..", "http://a/b/"),
        ("../", "http://a/b/"),
        ("../g", "http://a/b/g"),
        ("../..", "http://a/"),
        ("../../", "http://a/"),
        ("../../g", "http://a/g"),
        ("../../../g", "http://a/g"),
        ("../../../../g", "http://a/g"),
        ("/./g", "http://a/g"),
        ("/../g", "http://a/g"),
        ("g.", "http://a/b/c/g."),
        (".g", "http://a/b/c/.g"),
        ("g..", "http://a/b/c/g.."),
        ("..g", "http://a/b/c/..g"),
        ("./../g", "http://a/b/g"),
        ("./g/.", "http://a/b/c/g/"),
        ("g/./h", "http://a/b/c/g/h"),
        ("g/../h", "http://a/b/c/h"),
        ("g;x=1/./y", "http://a/b/c/g;x=1/y"),
        ("g;x=1/../y", "http://a/b/c/y"),
        ("g?y/./x", "http://a/b/c/g?y/./x"),
        ("g?y/../x", "http://a/b/c/g?y/../x"),
        ("g#s/./x", "http://a/b/c/g#s/./x"),
        ("g#s/../x", "http://a/b/c/g#s/../x"),
        ("http:g", "http:g"),
    ] {
        assert_eq!(resolve_location(&base, loc), want, "{loc:?}");
    }
    // a reference with a scheme but no `//` is no target (the strict reading of section 5.2.2)
    assert!(parse_target(&resolve_location(&base, "http:g"), Ok(())).is_err());
}

#[test]
fn shown_urls_hold_no_user_part_query_or_fragment() {
    for (url, want) in [
        ("http://example.com/mcp?token=t#f", "http://example.com/mcp"),
        ("https://user:pw@example.com/x", "https://example.com/x"),
        (
            "https://user:pw@example.com:8443",
            "https://example.com:8443",
        ),
        ("http://a@b:pw@example.com/", "http://example.com/"),
        ("http://user@example.com", "http://example.com"),
        // malformed: no scheme, one slash, backslashes, an open bracket, no slashes
        ("user:pw@example.com/x", "example.com/x"),
        ("http:/user:pw@example.com/x", "http:/example.com/x"),
        ("http:\\\\user:pw@example.com\\x", "http:\\\\example.com\\x"),
        ("http://user:pw@[::1", "http://[::1"),
        ("http:user:pw@example.com", "example.com"),
        ("//user:pw@example.com/", "//example.com/"),
        // an `@` in the path is no user part
        (
            "http://example.com/users/@me",
            "http://example.com/users/@me",
        ),
        ("http://user:pw@example.com/@me", "http://example.com/@me"),
        ("http://example.com?x=@user:pw", "http://example.com"),
    ] {
        assert_eq!(shown_url(url), want, "{url:?}");
    }
    // the scheme of a refused URL is shown only when it is a scheme
    let e = parse_target("user:secret-pw@example.com://x", Ok(()))
        .err()
        .unwrap();
    assert!(
        e.contains("not a scheme") && !e.contains("secret-pw"),
        "{e}"
    );
    let e = parse_target("gopher://example.com/", Ok(())).err().unwrap();
    assert!(e.contains("(scheme: gopher)"), "{e}");
}

#[test]
fn allow_list_entries_parse_and_match() {
    let e = |s: &str| parse_entry(s).unwrap_or_else(|e| panic!("{e}"));
    let t = |u: &str| parse_target(u, Ok(())).unwrap();
    assert_eq!(e("*").pat, Pattern::Any);
    assert_eq!(e("Example.COM.").pat, Pattern::Name("example.com".into()));
    assert_eq!(e("example.com:8931").port, Some(8931));
    assert_eq!(
        e("*.example.com").pat,
        Pattern::Suffix(".example.com".into())
    );
    assert_eq!(
        e("192.0.2.7").pat,
        Pattern::Net("192.0.2.7".parse().unwrap(), 32)
    );
    assert_eq!(e("192.0.2.7:8931").port, Some(8931));
    assert_eq!(
        e("192.0.2.0/24").pat,
        Pattern::Net("192.0.2.0".parse().unwrap(), 24)
    );
    assert_eq!(
        e("fd00::/8").pat,
        Pattern::Net("fd00::".parse().unwrap(), 8)
    );
    assert_eq!(e("::1").pat, Pattern::Net("::1".parse().unwrap(), 128));
    assert_eq!(e("[::1]:8931").port, Some(8931));
    for bad in [
        "192.0.2.0/33",
        "fd00::/129",
        "x/8",
        "exa mple",
        "host:port",
        "host:0",
        "[::1",
        "[::1]x",
        "*.",
        "a:b:c",
    ] {
        assert!(parse_entry(bad).is_err(), "{bad}");
    }
    for entry in [
        "*",
        "*.example.com",
        "example.com",
        "192.0.2.0/24",
        "[2001:db8::1]:8931",
        "fd00::/8",
    ] {
        assert_eq!(entry_text(&e(entry)), entry);
    }

    assert!(e("example.com").names(&t("http://example.com:99/")));
    assert!(!e("example.com").names(&t("http://sub.example.com/")));
    assert!(!e("example.com").names(&t("http://notexample.com/")));
    assert!(e("example.com:8931").names(&t("http://example.com:8931/")));
    assert!(!e("example.com:8931").names(&t("http://example.com/")));
    assert!(e("*.example.com").names(&t("https://a.b.example.com/")));
    assert!(!e("*.example.com").names(&t("https://example.com/")));
    assert!(!e("*.example.com").names(&t("https://evilexample.com/")));
    assert!(e("192.0.2.0/24").names(&t("http://192.0.2.9/")));
    assert!(!e("192.0.2.0/24").names(&t("http://192.0.3.9/")));
    // a name is never matched by an address entry as text
    assert!(!e("192.0.2.0/24").names(&t("http://192.0.2.9.example.com/")));
    assert!(e("*").is_open() && e("0.0.0.0/0").is_open() && e("::/0").is_open());
    assert!(!e("*.example.com").is_open() && !e("10.0.0.0/8").is_open());
}

#[test]
fn the_proxy_starts_only_with_named_targets_and_a_key_for_an_open_list() {
    let start = |o: &ProxyOptions, keyed: bool| McpProxy::new(o, keyed).map(|p| p.enabled());
    // off by default, whatever else is given
    assert_eq!(start(&ProxyOptions::default(), false), Ok(false));
    let mut off = options(&["*"], &[]);
    off.enabled = false;
    assert_eq!(start(&off, false), Ok(false));
    // on needs targets
    let e = start(&options(&[], &[]), true).err().unwrap();
    assert!(e.contains("--mcp-proxy-allow"), "{e}");
    // named targets: no key needed
    assert_eq!(start(&options(&["example.com"], &[]), false), Ok(true));
    assert_eq!(start(&options(&[], &["localhost:8931"]), false), Ok(true));
    // any target: only with a key
    for open in ["*", "0.0.0.0/0", "::/0"] {
        let e = start(&options(&[open], &[]), false).err().unwrap();
        assert!(e.contains("--api-key"), "{e}");
        assert_eq!(start(&options(&[open], &[]), true), Ok(true));
    }
    // never every private address at once, key or not
    for every in ["*", "0.0.0.0/0", "::/0"] {
        let e = start(&options(&["*"], &[every]), true).err().unwrap();
        assert!(e.contains("--mcp-proxy-allow-private"), "{e}");
    }
    // a mistyped entry stops the start instead of being skipped
    let e = start(&options(&["example.com, exa mple"], &[]), true)
        .err()
        .unwrap();
    assert!(e.contains("exa mple"), "{e}");
    // caps
    let mut o = options(&["example.com"], &[]);
    o.max_response_mb = 0;
    assert!(start(&o, true)
        .err()
        .unwrap()
        .contains("--mcp-proxy-max-mb"));
    o.max_response_mb = 64;
    o.idle_timeout_s = 0;
    assert!(start(&o, true)
        .err()
        .unwrap()
        .contains("--mcp-proxy-timeout"));
    // today's defaults
    let p = proxy(&["example.com"], &[]);
    assert_eq!(p.limits.max_response, 64 << 20);
    assert_eq!(p.limits.max_request, 8 << 20);
    assert_eq!((p.limits.idle_ms, p.limits.total_ms), (600_000, 3_600_000));
    assert_eq!(
        (
            p.limits.redirects,
            p.limits.keepalive_ms,
            p.limits.connect_ms
        ),
        (3, 20_000, 10_000)
    );
}

#[test]
fn chunked_bodies_decode_across_any_split() {
    let wire = b"5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\nTrailer: x\r\n\r\n";
    for split in 1..wire.len() {
        let mut framing = Framing::Chunked(ChunkState::Size);
        let mut buf = Vec::new();
        let mut out = Vec::new();
        let mut ended = false;
        for part in [&wire[..split], &wire[split..]] {
            buf.extend_from_slice(part);
            loop {
                match body_step(&mut framing, &mut buf) {
                    Step::Data(n) => {
                        out.extend_from_slice(&buf[..n]);
                        buf.drain(..n);
                    }
                    Step::More => break,
                    Step::End => {
                        ended = true;
                        break;
                    }
                    Step::Bad(why) => panic!("split {split}: {why}"),
                }
            }
        }
        assert!(ended, "split {split}");
        assert_eq!(out, b"hello world", "split {split}");
    }
    let mut framing = Framing::Chunked(ChunkState::Size);
    assert!(matches!(
        body_step(&mut framing, &mut b"zz\r\n".to_vec()),
        Step::Bad(_)
    ));
    let mut framing = Framing::Chunked(ChunkState::Size);
    let mut buf = b"2\r\nabXX\r\n".to_vec();
    assert!(matches!(body_step(&mut framing, &mut buf), Step::Data(2)));
    buf.drain(..2);
    assert!(matches!(body_step(&mut framing, &mut buf), Step::Bad(_)));
}

// ------------------------------------------------------------------------------------------------
// The gate
// ------------------------------------------------------------------------------------------------

#[test]
fn a_proxy_that_is_off_refuses_and_says_how_to_turn_it_on() {
    let p = Arc::new(McpProxy::disabled());
    assert!(!p.enabled());
    let r = get(&p, "http://example.com/");
    assert_eq!(r.status, 403);
    assert!(r.error().starts_with("mcp proxy: "), "{}", r.error());
    assert!(r.error().contains("--mcp-proxy"));
    assert!(r.text().contains("feature_disabled"));
}

#[test]
fn a_request_without_the_key_or_with_another_method_does_not_run() {
    let t = target(None, |_, s| ok_json(s, "", "{}"));
    let p = Arc::new(proxy(&[], &["127.0.0.1"]));
    let raw = format!(
        "GET /cors-proxy?url={} HTTP/1.1\r\nHost: localhost\r\n\r\n",
        query(&t.url("/"))
    );
    let mut c = open(&p, raw, false);
    let mut out = Vec::new();
    let _ = c.read_to_end(&mut out);
    let r = parse_reply(&out);
    assert_eq!(r.status, 401);
    assert!(r.text().contains("authentication_error"));

    for method in ["PUT", "PATCH", "HEAD"] {
        let r = send(&p, method, &t.url("/"), &[], "");
        assert_eq!(r.status, 405, "{method}");
    }
    let r = send(&p, "PUT", &t.url("/"), &[], "");
    assert!(
        r.error().starts_with("mcp proxy: method PUT"),
        "{}",
        r.error()
    );
    assert!(t.seen().is_empty(), "nothing reached the target");

    let r = send(&p, "GET", "", &[], "");
    assert_eq!(r.status, 400);
    assert!(r.error().contains("missing 'url'"));
}

#[test]
fn a_page_of_another_site_cannot_drive_the_proxy() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let p = Arc::new(proxy(&[], &["127.0.0.1"]));
    let url = t.url("/mcp");
    let hit = |headers: &[&str]| send(&p, "GET", &url, headers, "");

    // the browser says so itself
    for site in ["cross-site", "same-site", "weird"] {
        let r = hit(&[
            &format!("Sec-Fetch-Site: {site}"),
            "Origin: http://localhost:8090",
        ]);
        assert_eq!(r.status, 403, "{site}");
        assert!(r
            .error()
            .starts_with("mcp proxy: the request comes from a page of another origin"));
        assert!(r.header("access-control-allow-origin").is_none());
    }
    // no statement (a plain-http page): the origin must be the request's own host
    let r = hit(&["Origin: http://evil.example.com"]);
    assert_eq!(r.status, 403);
    assert!(r.error().contains("http://evil.example.com"));
    assert_eq!(hit(&["Origin: http://localhost:3000"]).status, 403);
    assert_eq!(hit(&["Origin: null"]).status, 403);
    assert_eq!(
        t.seen().len(),
        0,
        "a refused request never reaches the target"
    );

    // this server's own page, and clients that are not browsers
    assert_eq!(hit(&["Origin: http://localhost:8090"]).status, 200);
    assert_eq!(hit(&["Origin: http://LOCALHOST:8090"]).status, 200);
    assert_eq!(
        hit(&[
            "Sec-Fetch-Site: same-origin",
            "Origin: http://localhost:8090"
        ])
        .status,
        200
    );
    assert_eq!(hit(&["Sec-Fetch-Site: none"]).status, 200);
    assert_eq!(hit(&[]).status, 200);
    assert_eq!(t.seen().len(), 5);
}

/// One request to `/cors-proxy` with `Host: <host>` and the given header lines.
fn with_host(p: &Arc<McpProxy>, url: &str, host: &str, headers: &[&str]) -> Reply {
    let mut raw = format!(
        "POST /cors-proxy?url={} HTTP/1.1\r\nHost: {host}\r\n",
        query(url)
    );
    for h in headers {
        raw.push_str(h);
        raw.push_str("\r\n");
    }
    raw.push_str("Content-Length: 2\r\n\r\n{}");
    let mut c = open(p, raw, true);
    let mut out = Vec::new();
    let _ = c.read_to_end(&mut out);
    parse_reply(&out)
}

/// A page on any domain can point its own name at this server's address (DNS rebinding). The
/// browser then sends an origin and a `Host` that agree, and calls the request same-origin; the
/// `Host` still names the page's domain, which is not one of this server's names.
#[test]
fn a_page_that_points_its_own_name_at_this_server_is_refused() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let p = Arc::new(proxy(&[], &["127.0.0.1"]));
    let url = t.url("/mcp");
    for headers in [
        &["Origin: http://rebind.example.com:8090"][..],
        &[
            "Origin: http://rebind.example.com:8090",
            "Sec-Fetch-Site: same-origin",
        ],
        &["Sec-Fetch-Site: none"],
        &[],
    ] {
        let r = with_host(&p, &url, "rebind.example.com:8090", headers);
        assert_eq!(r.status, 403, "{headers:?}");
        let e = r.error();
        assert!(
            e.starts_with("mcp proxy: the request names the host rebind.example.com:8090, which is not one of this server's names"),
            "{e}"
        );
        assert!(e.contains("--server-name"), "{e}");
        assert!(r.header("access-control-allow-origin").is_none());
    }
    // a name that only starts or ends like one of the server's names is another name
    for host in ["localhost.example.com", "evil-localhost", "localhostx:8090"] {
        let origin = format!("Origin: http://{host}");
        assert_eq!(with_host(&p, &url, host, &[&origin]).status, 403, "{host}");
    }
    // no Host, or one that is not a host
    let mut c = open(
        &p,
        format!(
            "GET /cors-proxy?url={} HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            query(&url)
        ),
        true,
    );
    let mut out = Vec::new();
    let _ = c.read_to_end(&mut out);
    assert_eq!(parse_reply(&out).status, 403);
    for host in [
        "",
        "a b",
        "localhost:http",
        "[::1",
        "::1",
        "[::1]x",
        "localhost:0",
    ] {
        assert_eq!(with_host(&p, &url, host, &[]).status, 403, "{host:?}");
    }
    assert_eq!(
        t.seen().len(),
        0,
        "a refused request never reaches the target"
    );
}

/// The Web UI opened by loopback names and addresses keeps working, with and without a port and
/// whatever the case of the name; an IPv6 literal is bracketed.
#[test]
fn this_servers_own_names_and_addresses_pass() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let p = Arc::new(proxy(&[], &["127.0.0.1"]));
    let url = t.url("/mcp");
    let mut n = 0;
    for host in [
        "localhost",
        "localhost:8090",
        "LocalHost:8090",
        "localhost.:8090",
        "127.0.0.1",
        "127.0.0.1:8090",
        "[::1]",
        "[::1]:8090",
        "192.0.2.10:8090",
        "[2001:db8::10]:8090",
    ] {
        let origin = format!("Origin: http://{host}");
        assert_eq!(with_host(&p, &url, host, &[&origin]).status, 200, "{host}");
        assert_eq!(
            with_host(&p, &url, host, &["Sec-Fetch-Site: same-origin"]).status,
            200,
            "{host}"
        );
        n += 2;
    }
    assert_eq!(t.seen().len(), n);
}

/// A name given with --server-name, or one of the HTTPS certificate, passes like localhost.
#[test]
fn names_given_to_the_server_pass_and_others_do_not() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let mut o = options(&[], &["127.0.0.1"]);
    o.server_names = vec!["chat.example.com, Gaudium.Example.Org".into()];
    let mut p = McpProxy::new(&o, true).unwrap();
    p.add_server_names(&["cert.example.net".into(), "*.example.net".into()]);
    let p = Arc::new(p);
    let url = t.url("/mcp");
    for host in [
        "chat.example.com",
        "CHAT.EXAMPLE.COM:8090",
        "gaudium.example.org:443",
        "cert.example.net",
    ] {
        let origin = format!("Origin: https://{}", host.trim_end_matches(":443"));
        assert_eq!(with_host(&p, &url, host, &[&origin]).status, 200, "{host}");
    }
    for host in [
        "other.example.com",
        "chat.example.com.example.net",
        "x.example.net",
    ] {
        let origin = format!("Origin: https://{host}");
        assert_eq!(with_host(&p, &url, host, &[&origin]).status, 403, "{host}");
    }
    assert!(p.status_json().contains(
        "\"server_names\":[\"localhost\",\"chat.example.com\",\"gaudium.example.org\",\"cert.example.net\"]"
    ));
    // a malformed or wildcard name does not start the server
    for bad in ["*.example.com", "a b", "chat.example.com:8090"] {
        let mut o = options(&[], &["127.0.0.1"]);
        o.server_names = vec![bad.into()];
        let e = McpProxy::new(&o, true).err().expect(bad);
        assert!(e.starts_with("--server-name "), "{e}");
    }
}

/// Behind a front end that ends TLS: the request arrives as plain HTTP, the page's origin is the
/// public https name, and `Host` is whatever the front end sends. The public name is given with
/// --server-name; every host the request names (Host, X-Forwarded-Host, Forwarded host=) must be
/// one of the server's.
#[test]
fn behind_a_front_end_its_public_name_is_given_to_the_server() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let mut o = options(&[], &["127.0.0.1"]);
    o.server_names = vec!["chat.example.com".into()];
    let p = Arc::new(McpProxy::new(&o, true).unwrap());
    let bare = Arc::new(proxy(&[], &["127.0.0.1"]));
    let url = t.url("/mcp");
    // `send` writes `Host: localhost:8090`: a front end that passes its upstream's name on
    let hit = |p: &Arc<McpProxy>, headers: &[&str]| send(p, "POST", &url, headers, "{}").status;
    let proto = "X-Forwarded-Proto: https";

    // a front end that keeps the public name in Host
    let origin = "Origin: https://chat.example.com";
    assert_eq!(
        with_host(&p, &url, "chat.example.com", &[proto, origin]).status,
        200
    );
    assert_eq!(
        with_host(&p, &url, "chat.example.com:443", &[proto, origin]).status,
        200
    );
    assert_eq!(
        with_host(
            &p,
            &url,
            "chat.example.com",
            &[proto, "Origin: https://other.example.com"]
        )
        .status,
        403
    );
    // the same front end without the name given: refused, and the message says what to give
    let r = with_host(&bare, &url, "chat.example.com", &[proto, origin]);
    assert_eq!(r.status, 403);
    assert!(r.error().contains("--server-name"));

    // the front end rewrites Host and names the public host in a forwarding header
    let fwd = "X-Forwarded-Host: chat.example.com";
    assert_eq!(hit(&p, &[origin, fwd]), 200);
    assert_eq!(
        hit(
            &p,
            &[
                origin,
                "Forwarded: for=192.0.2.9;host=chat.example.com;proto=https"
            ]
        ),
        200
    );
    assert_eq!(
        hit(
            &p,
            &[
                origin,
                "Forwarded: for=192.0.2.9;host=\"chat.example.com:443\""
            ]
        ),
        200
    );
    let r = send(&bare, "POST", &url, &[origin, fwd], "{}");
    assert_eq!(r.status, 403);
    assert!(
        r.error().contains("a front-end header names a host"),
        "{}",
        r.error()
    );
    // a rebinding page behind the front end: the front end names the page's domain
    assert_eq!(
        hit(
            &p,
            &[
                "Origin: https://other.example.com",
                "X-Forwarded-Host: other.example.com"
            ]
        ),
        403
    );
    assert_eq!(
        hit(
            &p,
            &[
                origin,
                "X-Forwarded-Host: chat.example.com, other.example.com"
            ]
        ),
        403
    );
    // a chain of front ends, each adding its own line: every line counts, whichever comes first
    for (a, b) in [
        (
            "X-Forwarded-Host: evil.example",
            "X-Forwarded-Host: 127.0.0.1:8080",
        ),
        (
            "X-Forwarded-Host: 127.0.0.1:8080",
            "X-Forwarded-Host: evil.example",
        ),
        (
            "Forwarded: host=evil.example",
            "Forwarded: host=127.0.0.1:8080",
        ),
        (
            "Forwarded: host=127.0.0.1:8080",
            "Forwarded: host=evil.example",
        ),
        (
            "X-Forwarded-Host: evil.example",
            "X-Forwarded-Host: chat.example.com",
        ),
        (
            "Forwarded: host=evil.example",
            "Forwarded: for=192.0.2.9;host=localhost",
        ),
    ] {
        let r = send(
            &p,
            "POST",
            &url,
            &[origin, "Sec-Fetch-Site: same-origin", a, b],
            "{}",
        );
        assert_eq!(r.status, 403, "{a} / {b}");
        assert!(
            r.error().contains("a front-end header names a host"),
            "{}",
            r.error()
        );
    }
    // lines that all name this server pass
    assert_eq!(
        hit(
            &p,
            &[
                origin,
                "X-Forwarded-Host: chat.example.com",
                "X-Forwarded-Host: 127.0.0.1:8080"
            ]
        ),
        200
    );
    assert_eq!(
        hit(
            &p,
            &[
                origin,
                "Forwarded: host=chat.example.com",
                "Forwarded: for=192.0.2.9;host=localhost"
            ]
        ),
        200
    );
    // more hosts than the bound: refused before any is read
    let many = format!(
        "X-Forwarded-Host: {}",
        vec!["127.0.0.1"; names::MAX_FORWARDED_HOSTS as usize + 1].join(",")
    );
    let same = "Sec-Fetch-Site: same-origin";
    let r = send(&p, "POST", &url, &[origin, same, &many], "{}");
    assert_eq!(r.status, 403);
    assert!(r.error().contains("more than 16 hosts"), "{}", r.error());
    // the same bound across lines, and one host fewer passes
    let line = "X-Forwarded-Host: 127.0.0.1";
    let lines = vec![line; names::MAX_FORWARDED_HOSTS as usize + 1];
    let mut headers = vec![origin, same];
    headers.extend(lines.iter().copied());
    let r = send(&p, "POST", &url, &headers, "{}");
    assert_eq!(r.status, 403);
    assert!(r.error().contains("more than 16 hosts"), "{}", r.error());
    let lines = vec![line; names::MAX_FORWARDED_HOSTS as usize];
    let mut headers = vec![origin, same];
    headers.extend(lines.iter().copied());
    assert_eq!(hit(&p, &headers), 200);
    // the front end rewrites Host and says nothing: the browser's own statement decides
    // (every browser sends it to an https page's server)
    assert_eq!(hit(&p, &[origin, "Sec-Fetch-Site: same-origin"]), 200);
    assert_eq!(hit(&p, &[origin, "Sec-Fetch-Site: cross-site", fwd]), 403);
}

// ------------------------------------------------------------------------------------------------
// Targets and addresses
// ------------------------------------------------------------------------------------------------

#[test]
fn only_listed_targets_are_reached_and_private_ones_only_when_listed_as_private() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let port = t.port;

    // not in any list: refused before anything is looked up
    let mut p = proxy(&["example.com"], &[]);
    p.resolver = Box::new(|host, _| panic!("{host} was looked up"));
    let p = Arc::new(p);
    let r = get(&p, "http://other.example.com/");
    assert_eq!(r.status, 403);
    assert!(
        r.error()
            .starts_with("mcp proxy: target other.example.com:80 is not in the allow-list"),
        "{}",
        r.error()
    );
    let r = get(&p, &t.url("/"));
    assert_eq!(r.status, 403);
    assert!(r.error().contains("not in the allow-list"));

    // listed as a target, but its address is loopback and it is not listed as private
    let p = Arc::new(proxy(&["127.0.0.1", "*"], &[]));
    let r = get(&p, &t.url("/"));
    assert_eq!(r.status, 403);
    assert!(
        r.error().contains(
            "127.0.0.1 is a loopback address: not allowed without --mcp-proxy-allow-private"
        ),
        "{}",
        r.error()
    );
    // nor do the other spellings of the same address get through
    for url in [
        format!("http://[::ffff:127.0.0.1]:{port}/"),
        format!("http://[::ffff:7f00:1]:{port}/"),
        format!("http://[64:ff9b::7f00:1]:{port}/"),
        format!("http://[2002:7f00:1::]:{port}/"),
        format!("http://[::1]:{port}/"),
        format!("http://[::]:{port}/"),
        format!("http://0.0.0.0:{port}/"),
        format!("http://127.1.2.3:{port}/"),
        format!("http://169.254.169.254/latest/meta-data/"),
        format!("http://[fd00:ec2::254]/latest/meta-data/"),
        format!("http://10.0.0.1:{port}/"),
        format!("http://192.168.1.1:{port}/"),
    ] {
        let r = get(&p, &url);
        assert_eq!(r.status, 403, "{url}");
        assert!(r.error().starts_with("mcp proxy: "), "{url}: {}", r.error());
    }
    assert!(t.seen().is_empty());

    // listed as private: reached; the port of the entry is part of it
    let p = Arc::new(proxy(&[], &[&format!("127.0.0.1:{port}")]));
    let r = get(&p, &t.url("/mcp?x=1"));
    assert_eq!((r.status, r.text().as_str()), (200, "{\"ok\":true}"));
    assert_eq!(t.seen()[0].path, "/mcp?x=1");
    assert_eq!(
        t.seen()[0].header("host"),
        Some(format!("127.0.0.1:{port}").as_str())
    );
    let other = target(None, |_, s| ok_json(s, "", "{}"));
    assert_eq!(get(&p, &other.url("/")).status, 403);
    assert!(other.seen().is_empty());

    // an address that is never connected to stays refused even when listed
    let p = Arc::new(proxy(&[], &["0.0.0.0/8", "::/127", "224.0.0.0/4"]));
    for url in [
        format!("http://0.0.0.0:{port}/"),
        format!("http://[::]:{port}/"),
        "http://224.0.0.1/".to_string(),
    ] {
        let r = get(&p, &url);
        assert_eq!(r.status, 403, "{url}");
        assert!(r.error().contains("never connected to"), "{}", r.error());
    }
}

#[test]
fn a_name_is_judged_by_every_address_it_resolves_to() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let port = t.port;
    let names: &[(&str, &[&str])] = &[
        ("mixed.example.com", &["8.8.8.8", "127.0.0.1"]),
        ("inner.example.com", &["127.0.0.1"]),
        ("meta.example.com", &["169.254.169.254"]),
        ("mapped.example.com", &["::ffff:127.0.0.1"]),
        ("lan.example.com", &["127.0.0.1"]),
        ("stray.example.com", &["127.0.0.1", "192.0.2.77"]),
    ];

    // a public-looking name that points inside: refused, although the name is listed
    let mut p = proxy(&["*.example.com"], &[]);
    resolve_as(&mut p, names);
    let p = Arc::new(p);
    for (name, what) in [
        (
            "mixed.example.com",
            "resolves to 127.0.0.1, a loopback address",
        ),
        (
            "inner.example.com",
            "resolves to 127.0.0.1, a loopback address",
        ),
        (
            "meta.example.com",
            "resolves to 169.254.169.254, a link-local address",
        ),
        (
            "mapped.example.com",
            "resolves to ::ffff:127.0.0.1, a loopback address",
        ),
    ] {
        let r = get(&p, &format!("http://{name}:{port}/"));
        assert_eq!(r.status, 403, "{name}");
        assert!(r.error().contains(what), "{name}: {}", r.error());
        assert!(r.error().contains("--mcp-proxy-allow-private"));
    }
    assert!(t.seen().is_empty());
    // a name that does not resolve: the proxy tried and could not reach it
    let r = get(&p, "http://nowhere.example.com/");
    assert_eq!(r.status, 502);
    assert!(
        r.error()
            .starts_with("proxy error: cannot resolve nowhere.example.com"),
        "{}",
        r.error()
    );

    // the name listed as private: reached at the address that was checked, Host stays the name
    let mut p = proxy(&[], &["lan.example.com"]);
    resolve_as(&mut p, names);
    let p = Arc::new(p);
    assert_eq!(
        get(&p, &format!("http://lan.example.com:{port}/")).status,
        200
    );
    assert_eq!(
        t.seen()[0].header("host"),
        Some(format!("lan.example.com:{port}").as_str())
    );
    assert_eq!(
        get(&p, &format!("http://inner.example.com:{port}/")).status,
        403
    );

    // a name admitted through a range must resolve only inside the ranges
    let mut p = proxy(&[], &["127.0.0.0/8"]);
    resolve_as(&mut p, names);
    let p = Arc::new(p);
    assert_eq!(
        get(&p, &format!("http://inner.example.com:{port}/")).status,
        200
    );
    let r = get(&p, &format!("http://stray.example.com:{port}/"));
    assert_eq!(r.status, 403);
    assert!(
        r.error()
            .contains("resolves to 192.0.2.77, outside the address ranges"),
        "{}",
        r.error()
    );

    // the system's own resolver: localhost points at this machine and is refused unless listed
    let p = Arc::new(proxy(&["localhost"], &[]));
    let r = get(&p, &format!("http://localhost:{port}/"));
    assert_eq!(r.status, 403);
    assert!(r.error().contains("a loopback address"), "{}", r.error());
    let p = Arc::new(proxy(&[], &["localhost"]));
    assert_eq!(get(&p, &format!("http://localhost:{port}/")).status, 200);
}

// ------------------------------------------------------------------------------------------------
// Headers, methods, session
// ------------------------------------------------------------------------------------------------

#[test]
fn only_prefixed_headers_reach_the_target_and_the_servers_key_stays_here() {
    let t = target(None, |_, s| {
        ok_json(
            s,
            "Mcp-Session-Id: session-1\r\nSet-Cookie: a=b\r\nAccess-Control-Allow-Origin: *\r\nServer: target\r\nX-Custom: kept\r\nClear-Site-Data: \"*\"\r\n",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}",
        )
    });
    let p = Arc::new(proxy(&[], &["127.0.0.1"]));
    let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}";
    let r = send(
        &p,
        "POST",
        &t.url("/mcp"),
        &[
            "Authorization: Bearer server-key",
            "X-Api-Key: server-key",
            "Cookie: ui=1",
            "Content-Type: application/json",
            "x-gaudium-proxy-header-authorization: Bearer mcp-token",
            "x-gaudium-proxy-header-content-type: application/json",
            "x-gaudium-proxy-header-accept: application/json, text/event-stream",
            "x-gaudium-proxy-header-mcp-session-id: session-0",
            "x-gaudium-proxy-header-host: evil.example.com",
            "x-gaudium-proxy-header-content-length: 999",
            "x-gaudium-proxy-header-transfer-encoding: chunked",
            "x-gaudium-proxy-header-connection: keep-alive",
            "x-gaudium-proxy-header-: empty",
            "x-gaudium-proxy-header-bad name: x",
        ],
        body,
    );
    assert_eq!(r.status, 200);
    assert_eq!(r.text(), "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}");
    let seen = &t.seen()[0];
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.body, body.as_bytes());
    assert_eq!(seen.header("authorization"), Some("Bearer mcp-token"));
    assert_eq!(seen.header("content-type"), Some("application/json"));
    assert_eq!(
        seen.header("accept"),
        Some("application/json, text/event-stream")
    );
    assert_eq!(seen.header("mcp-session-id"), Some("session-0"));
    assert_eq!(
        seen.header("host"),
        Some(format!("127.0.0.1:{}", t.port).as_str())
    );
    assert_eq!(
        seen.header("content-length"),
        Some(body.len().to_string().as_str())
    );
    assert_eq!(seen.header("connection"), Some("close"));
    for absent in ["x-api-key", "cookie", "transfer-encoding", "bad name", ""] {
        assert!(seen.header(absent).is_none(), "{absent}");
    }
    assert!(
        !seen.headers.iter().any(|h| h.1.contains("server-key")),
        "{:?}",
        seen.headers
    );
    assert!(!seen.headers.iter().any(|h| h.0.starts_with(HEADER_PREFIX)));

    // back to the page: the session id and the target's own headers, nothing that would speak
    // for this server's origin
    assert_eq!(r.header("mcp-session-id"), Some("session-1"));
    assert_eq!(r.header("x-custom"), Some("kept"));
    assert_eq!(r.header("content-type"), Some("application/json"));
    for absent in [
        "set-cookie",
        "access-control-allow-origin",
        "server",
        "clear-site-data",
    ] {
        assert!(r.header(absent).is_none(), "{absent}");
    }
    assert_eq!(r.header("content-security-policy"), Some("sandbox"));
    assert_eq!(r.header("x-content-type-options"), Some("nosniff"));
    assert_eq!(r.header("cache-control"), Some("no-store"));
}

#[test]
fn delete_and_bodyless_answers_pass_through() {
    let t = target(None, |req, s| {
        let _ = match (req.method.as_str(), req.path.as_str()) {
            ("DELETE", "/mcp") => s.write_all(b"HTTP/1.1 204 No Content\r\n\r\n"),
            ("DELETE", _) => s.write_all(
                b"HTTP/1.1 405 Method Not Allowed\r\nAllow: GET, POST\r\nContent-Length: 2\r\n\r\nno",
            ),
            _ => s.write_all(b"HTTP/1.1 500 Oops\r\nContent-Length: 0\r\n\r\n"),
        };
    });
    let p = Arc::new(proxy(&[], &["127.0.0.1"]));
    let r = send(
        &p,
        "DELETE",
        &t.url("/mcp"),
        &["x-gaudium-proxy-header-mcp-session-id: s-9"],
        "",
    );
    assert_eq!((r.status, r.body.len(), r.complete), (204, 0, true));
    assert_eq!(r.header("content-length"), Some("0"));
    assert_eq!(t.seen()[0].method, "DELETE");
    assert_eq!(t.seen()[0].header("mcp-session-id"), Some("s-9"));
    assert!(t.seen()[0].header("content-length").is_none());

    // the target's own refusal is the target's answer, not the proxy's
    let r = send(&p, "DELETE", &t.url("/other"), &[], "");
    assert_eq!((r.status, r.text().as_str()), (405, "no"));
    assert_eq!(r.header("allow"), Some("GET, POST"));
    let r = get(&p, &t.url("/x"));
    assert_eq!((r.status, r.body.len()), (500, 0));
}

// ------------------------------------------------------------------------------------------------
// Redirects
// ------------------------------------------------------------------------------------------------

#[test]
fn redirects_are_followed_by_the_proxy_through_the_same_checks() {
    let last = target(None, |req, s| {
        ok_json(
            s,
            "",
            &format!(
                "{{\"method\":\"{}\",\"body\":{}}}",
                req.method,
                req.body.len()
            ),
        )
    });
    let last_port = last.port;
    // another origin that sends the request back to the first one
    let first_port = Arc::new(AtomicU64::new(0));
    let back = {
        let first_port = Arc::clone(&first_port);
        target(None, move |_, s| {
            let _ = write!(
                s,
                "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{}/landed\r\nContent-Length: 0\r\n\r\n",
                first_port.load(Ordering::Relaxed)
            );
        })
    };
    let back_port = back.port;
    let first = target(None, move |req, s| {
        let to = |status: u16, location: String| {
            format!(
                "HTTP/1.1 {status} Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\n\r\n"
            )
        };
        let out = match req.path.as_str() {
            "/same" => to(302, "/landed".into()),
            "/landed" => {
                return ok_json(
                    s,
                    "",
                    &format!(
                        "{{\"auth\":\"{}\"}}",
                        req.header("authorization").unwrap_or("-")
                    ),
                )
            }
            "/q/b?x=1" => to(302, "?x=2".into()),
            "/d/e/f" => to(302, "../g/./h?k=1".into()),
            "/q/b?x=2" | "/d/g/h?k=1" => {
                return ok_json(s, "", &format!("{{\"path\":\"{}\"}}", req.path))
            }
            "/away" => to(302, format!("http://127.0.0.1:{back_port}/")),
            "/moved" => to(302, format!("http://127.0.0.1:{last_port}/a")),
            "/kept" => to(307, format!("http://127.0.0.1:{last_port}/a")),
            "/see" => to(303, format!("http://127.0.0.1:{last_port}/a")),
            "/meta" => to(302, "http://169.254.169.254/latest/meta-data/".into()),
            "/unlisted" => to(301, "http://other.example.com/".into()),
            "/file" => to(302, "file:///etc/passwd".into()),
            "/loop" => to(302, "/loop".into()),
            "/multiple" => {
                "HTTP/1.1 300 Multiple Choices\r\nLocation: /x\r\nContent-Length: 2\r\n\r\nhi"
                    .into()
            }
            _ => to(302, "/same".into()),
        };
        let _ = s.write_all(out.as_bytes());
    });
    let mut p = proxy(
        &["*"],
        &[
            &format!("127.0.0.1:{}", first.port),
            &format!("127.0.0.1:{last_port}"),
            &format!("127.0.0.1:{back_port}"),
        ],
    );
    p.resolver = Box::new(|host, _| panic!("{host} was looked up"));
    first_port.store(u64::from(first.port), Ordering::Relaxed);
    let p = Arc::new(p);
    let auth = [
        "x-gaudium-proxy-header-authorization: Bearer mcp-token",
        "x-gaudium-proxy-header-accept: application/json",
    ];

    // within the origin the page named: the page's headers go along
    let r = send(&p, "GET", &first.url("/same"), &auth, "");
    assert_eq!(
        (r.status, r.text().as_str()),
        (200, "{\"auth\":\"Bearer mcp-token\"}")
    );

    // relative references resolve against the whole URL: the query-only one keeps the path,
    // dot segments go
    for (path, landed) in [("/q/b?x=1", "/q/b?x=2"), ("/d/e/f", "/d/g/h?k=1")] {
        let r = get(&p, &first.url(path));
        assert_eq!(
            (r.status, r.text()),
            (200, format!("{{\"path\":\"{landed}\"}}")),
            "{path}"
        );
    }

    // once a redirect left the origin, coming back to it does not bring the headers back
    let r = send(&p, "GET", &first.url("/away"), &auth, "");
    assert_eq!((r.status, r.text().as_str()), (200, "{\"auth\":\"-\"}"));

    // to another origin: a POST becomes a GET without its body on 302 and 303, stays a POST on
    // 307; headers that may carry a credential stay behind, content negotiation goes along
    for (path, method, body) in [
        ("/moved", "GET", 0),
        ("/see", "GET", 0),
        ("/kept", "POST", 7),
    ] {
        let before = last.seen().len();
        let r = send(&p, "POST", &first.url(path), &auth, "payload");
        assert_eq!(r.status, 200, "{path}");
        assert_eq!(
            r.text(),
            format!("{{\"method\":\"{method}\",\"body\":{body}}}"),
            "{path}"
        );
        let seen = &last.seen()[before];
        assert!(
            seen.header("authorization").is_none(),
            "{path}: {:?}",
            seen.headers
        );
        assert_eq!(seen.header("accept"), Some("application/json"), "{path}");
    }

    // to a private address, to a target that is not listed, to another scheme: refused, and the
    // refusal names the redirect
    let r = get(&p, &first.url("/meta"));
    assert_eq!(r.status, 403);
    assert!(r.error().starts_with("mcp proxy: redirect to http://169.254.169.254/latest/meta-data/: 169.254.169.254 is a link-local address"), "{}", r.error());
    let r = get(&p, &first.url("/file"));
    assert_eq!(r.status, 403);
    assert!(
        r.error().contains("redirect from") && r.error().contains("http:// or https://"),
        "{}",
        r.error()
    );
    let strict = Arc::new(proxy(&[], &[&format!("127.0.0.1:{}", first.port)]));
    let r = get(&strict, &first.url("/unlisted"));
    assert_eq!(r.status, 403);
    assert!(r.error().contains("redirect to http://other.example.com/: target other.example.com:80 is not in the allow-list"), "{}", r.error());
    let r = get(&strict, &first.url("/moved"));
    assert_eq!(
        r.status, 403,
        "another port of the same address is another target"
    );

    // bounded: the request and three redirects, then the proxy stops
    let before = first.seen().len();
    let r = get(&p, &first.url("/loop"));
    assert_eq!(r.status, 403);
    assert!(
        r.error().contains("too many redirects (more than 3)"),
        "{}",
        r.error()
    );
    assert_eq!(first.seen().len() - before, 4);

    // a 3xx the proxy does not follow goes to the page without its Location
    let r = get(&p, &first.url("/multiple"));
    assert_eq!((r.status, r.text().as_str()), (300, "hi"));
    assert!(r.header("location").is_none());
}

// ------------------------------------------------------------------------------------------------
// Streaming, caps, timeouts
// ------------------------------------------------------------------------------------------------

/// Reads from the proxy until `needle` has arrived (or five seconds passed).
fn read_until(c: &mut TcpStream, got: &mut Vec<u8>, needle: &[u8]) -> bool {
    c.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut b = [0u8; 4096];
    while !got.windows(needle.len()).any(|w| w == needle) {
        match c.read(&mut b) {
            Ok(0) | Err(_) => return false,
            Ok(n) => got.extend_from_slice(&b[..n]),
        }
    }
    true
}

#[test]
fn an_event_stream_passes_through_as_it_arrives() {
    // the target sends its second event only once the page has confirmed the first: a proxy
    // that held the first back would never see the second
    let (confirmed_tx, confirmed_rx) = mpsc::channel::<()>();
    let confirmed_rx = Mutex::new(confirmed_rx);
    let t = target(None, move |_, s| {
        let _ = s.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
        );
        let first = b"event: message\ndata: {\"n\":1}\n\n";
        let _ = write!(s, "{:x}\r\n", first.len());
        let _ = s.write_all(first);
        let _ = s.write_all(b"\r\n");
        let _ = s.flush();
        let ok = confirmed_rx
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .is_ok();
        let second: &[u8] = if ok {
            b"data: {\"n\":2}\n\n"
        } else {
            b"data: late\n\n"
        };
        let _ = write!(s, "{:x}\r\n", second.len());
        let _ = s.write_all(second);
        let _ = s.write_all(b"\r\n0\r\n\r\n");
    });
    let p = Arc::new(proxy(&[], &["127.0.0.1"]));
    let raw = format!(
        "GET /cors-proxy?url={} HTTP/1.1\r\nHost: localhost\r\nx-gaudium-proxy-header-accept: text/event-stream\r\n\r\n",
        query(&t.url("/mcp"))
    );
    let mut c = open(&p, raw, true);
    let mut got = Vec::new();
    assert!(
        read_until(&mut c, &mut got, b"data: {\"n\":1}\n\n"),
        "the first event did not arrive on its own"
    );
    assert!(!got.windows(5).any(|w| w == b"\"n\":2"));
    confirmed_tx.send(()).unwrap();
    let _ = c.read_to_end(&mut got);
    let r = parse_reply(&got);
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    assert_eq!(r.header("x-accel-buffering"), Some("no"));
    assert_eq!(
        r.text(),
        "event: message\ndata: {\"n\":1}\n\ndata: {\"n\":2}\n\n"
    );
    assert!(r.complete);
    assert_eq!(t.seen()[0].header("accept"), Some("text/event-stream"));
}

#[test]
fn an_idle_event_stream_gets_comment_lines_only_between_lines() {
    let t = target(None, |req, s| {
        let kind = if req.path == "/json" {
            "application/json"
        } else {
            "text/event-stream"
        };
        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\n\r\n");
        let _ = s.flush();
        thread::sleep(Duration::from_millis(180));
        // a line cut in two by a pause: nothing may be put inside it
        let _ = s.write_all(b"data: par");
        let _ = s.flush();
        thread::sleep(Duration::from_millis(180));
        let _ = s.write_all(b"tial\n");
        let _ = s.flush();
        thread::sleep(Duration::from_millis(180));
        let _ = s.write_all(b"\ndata: end\n\n");
    });
    let mut p = proxy(&[], &["127.0.0.1"]);
    p.limits.keepalive_ms = 40;
    let p = Arc::new(p);

    let r = get(&p, &t.url("/events"));
    assert_eq!(r.status, 200);
    let text = r.text();
    let alive = ": keep-alive\n";
    assert!(
        text.starts_with(alive),
        "a stream that starts idle is kept alive: {text:?}"
    );
    assert!(text.contains("data: partial\n"), "{text:?}");
    assert!(text.matches(alive).count() >= 4, "{text:?}");
    // every comment stands at the start of a line, and without them the stream is the target's
    assert_eq!(text.replace(alive, ""), "data: partial\n\ndata: end\n\n");
    assert!(
        text.contains(&format!("data: partial\n{alive}")),
        "inside an event, between two lines: {text:?}"
    );
    assert!(r.complete);

    // only an event stream has comments
    let r = get(&p, &t.url("/json"));
    assert_eq!(r.text(), "data: partial\n\ndata: end\n\n");
}

#[test]
fn bodies_over_the_caps_are_refused_or_cut() {
    let t = target(None, |req, s| {
        let big = vec![b'x'; 5000];
        match req.path.as_str() {
            "/declared" => {
                let _ = write!(
                    s,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                    big.len()
                );
                let _ = s.write_all(&big);
            }
            "/chunked" => {
                let _ = write!(
                    s,
                    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n",
                    big.len()
                );
                let _ = s.write_all(&big);
                let _ = s.write_all(b"\r\n0\r\n\r\n");
            }
            "/to-close" => {
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\n\r\n");
                let _ = s.write_all(&big);
            }
            _ => {
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n");
                let _ = s.write_all(&big[..1000]);
            }
        }
    });
    let mut p = proxy(&[], &["127.0.0.1"]);
    p.limits.max_response = 1000;
    p.limits.max_request = 100;
    let p = Arc::new(p);

    // a length the target declares: refused before anything is passed on
    let r = get(&p, &t.url("/declared"));
    assert_eq!(r.status, 502);
    assert!(
        r.error()
            .starts_with("proxy error: the response of 5000 bytes exceeds the limit of 1000 bytes"),
        "{}",
        r.error()
    );
    // a length only known as it arrives: cut at the cap, and the page sees a cut answer
    for path in ["/chunked", "/to-close"] {
        let r = get(&p, &t.url(path));
        assert_eq!(
            (r.status, r.body.len(), r.complete),
            (200, 1000, false),
            "{path}"
        );
    }
    // exactly the cap passes whole
    let r = get(&p, &t.url("/fits"));
    assert_eq!((r.status, r.body.len(), r.complete), (200, 1000, true));

    // the request body
    let before = t.seen().len();
    let r = send(&p, "POST", &t.url("/x"), &[], &"y".repeat(101));
    assert_eq!(r.status, 413);
    assert!(
        r.error()
            .contains("request body of 101 bytes exceeds the limit of 100 bytes"),
        "{}",
        r.error()
    );
    assert_eq!(t.seen().len(), before);
    assert_eq!(
        send(&p, "POST", &t.url("/x"), &[], &"y".repeat(100)).status,
        200
    );
}

#[test]
fn a_target_that_stalls_or_is_down_ends_the_request() {
    // one target per case: a stalled one answers nobody else meanwhile
    let stalling = || {
        target(None, |req, s| match req.path.as_str() {
            "/silent" => thread::sleep(Duration::from_millis(1500)),
            "/half" => {
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nhalf");
                let _ = s.flush();
                thread::sleep(Duration::from_millis(1500));
            }
            "/short" => {
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort");
            }
            _ => {
                let _ = s.write_all(b"not http at all\r\n\r\n");
            }
        })
    };
    let mut p = proxy(&[], &["127.0.0.1"]);
    p.limits.idle_ms = 150;
    let p = Arc::new(p);

    let t0 = Instant::now();
    let r = get(&p, &stalling().url("/silent"));
    assert_eq!(r.status, 502);
    assert!(
        r.error().starts_with("proxy error: ") && r.error().contains("no answer within"),
        "{}",
        r.error()
    );
    assert!(
        t0.elapsed() < Duration::from_millis(1200),
        "{:?}",
        t0.elapsed()
    );

    // silent in the middle of its body, or closing before the body is whole: the page gets what
    // arrived and sees that the answer was cut
    for (path, part) in [("/half", "half"), ("/short", "short")] {
        let r = get(&p, &stalling().url(path));
        assert_eq!(
            (r.status, r.text().as_str(), r.complete),
            (200, part, false),
            "{path}"
        );
    }

    let t = stalling();
    let r = get(&p, &t.url("/garbage"));
    assert_eq!(r.status, 502);
    assert!(
        r.error().contains("did not answer with HTTP/1.1"),
        "{}",
        r.error()
    );

    // nothing listens there
    let closed = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let r = get(&p, &format!("http://127.0.0.1:{closed}/"));
    assert_eq!(r.status, 502);
    assert!(
        r.error()
            .starts_with("proxy error: cannot connect to 127.0.0.1:"),
        "{}",
        r.error()
    );

    // the total limit ends a stream that never goes silent
    let chatty = target(None, |_, s| {
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
        for _ in 0..200 {
            if s.write_all(b"data: tick\n\n")
                .and_then(|()| s.flush())
                .is_err()
            {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
    });
    let mut p = proxy(&[], &["127.0.0.1"]);
    p.limits.total_ms = 200;
    let p = Arc::new(p);
    let t0 = Instant::now();
    let r = get(&p, &chatty.url("/"));
    assert_eq!((r.status, r.complete), (200, false));
    assert!(r.text().starts_with("data: tick\n\n"));
    assert!(
        t0.elapsed() < Duration::from_millis(2000),
        "{:?}",
        t0.elapsed()
    );
}

/// A page that sends its request and then never reads its answer. The target sends without
/// end; the exchange must still end within the proxy's limits, let go of the handler and close
/// the connection to the target.
fn a_page_that_stops_reading(tls: Option<(TlsServer, crate::tls::TlsClient)>) {
    let (closed_tx, closed_rx) = mpsc::channel::<()>();
    let t = target(None, move |_, s| {
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\r\n");
        let block = vec![b'x'; 64 * 1024];
        // until the proxy lets go of the connection (at most 256 MiB)
        for _ in 0..4096 {
            if s.write_all(&block).is_err() {
                break;
            }
        }
        let _ = closed_tx.send(());
    });
    let mut p = proxy(&[], &["127.0.0.1"]);
    p.limits.idle_ms = 300;
    p.limits.max_response = 1 << 30;
    let p = Arc::new(p);

    let (server, client) = match tls {
        Some((s, c)) => (Some(s), Some(c)),
        None => (None, None),
    };
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let handler = Arc::clone(&p);
    thread::spawn(move || {
        let (tcp, _) = l.accept().unwrap();
        let mut conn = Conn::open(tcp, server.as_ref(), super::super::DEFAULT_HSTS_MAX_AGE)
            .expect("a connection");
        let req = super::super::read_http_request(&mut conn)
            .unwrap()
            .expect("a request");
        handler.handle(&req, true, &mut conn);
        drop(conn);
        let _ = done_tx.send(());
    });
    let raw = format!(
        "GET /cors-proxy?url={} HTTP/1.1\r\nHost: localhost:8090\r\n\r\n",
        query(&t.url("/endless"))
    );
    let tcp = TcpStream::connect(addr).unwrap();
    // the page's end stays open until the checks below are done, and is never read
    let _page: Box<dyn Duplex> = match client {
        None => {
            let mut c = tcp;
            c.write_all(raw.as_bytes()).unwrap();
            Box::new(c)
        }
        Some(client) => {
            let mut c = client.connect(tcp, "localhost").unwrap();
            c.write_all(raw.as_bytes()).unwrap();
            Box::new(c)
        }
    };
    done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the handler let go of the exchange");
    closed_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the connection to the target was closed");
    let v = crate::json::parse_json(&p.status_json()).unwrap();
    let n = |k: &str| {
        v.get("counts")
            .and_then(|c| c.get(k))
            .and_then(|x| x.as_f64())
    };
    assert_eq!((n("proxied"), n("stalled")), (Some(1.0), Some(1.0)));
    let text = p.status_json();
    assert!(
        text.contains("\"result\":\"stalled\"") && text.contains("the page took nothing for"),
        "{text}"
    );
}

#[test]
fn a_page_that_stops_reading_ends_the_exchange() {
    a_page_that_stops_reading(None);
}

#[test]
fn a_page_that_stops_reading_over_https_ends_the_exchange() {
    let Some(tc) = test_cert(&["DNS:localhost"]) else {
        return;
    };
    let server = TlsServer::new(&tc.cert, &tc.key).unwrap();
    let client = crate::tls::TlsClient::new(Some(&tc.cert)).unwrap();
    a_page_that_stops_reading(Some((server, client)));
}

// ------------------------------------------------------------------------------------------------
// HTTPS targets
// ------------------------------------------------------------------------------------------------

#[test]
fn https_targets_are_verified_by_name_while_the_socket_goes_to_the_checked_address() {
    let Some(tc) = test_cert(&["DNS:mcp.example.com"]) else {
        return;
    };
    let t = target(
        Some(TlsServer::new(&tc.cert, &tc.key).unwrap()),
        |req, s| {
            ok_json(
                s,
                "Mcp-Session-Id: tls-1\r\n",
                &format!("{{\"host\":\"{}\"}}", req.header("host").unwrap_or("")),
            )
        },
    );
    let port = t.port;
    let names: &[(&str, &[&str])] = &[
        ("mcp.example.com", &["127.0.0.1"]),
        ("other.example.com", &["127.0.0.1"]),
    ];
    let mut opts = options(&[], &["mcp.example.com", "other.example.com", "127.0.0.1"]);
    opts.ca_file = Some(tc.cert.clone());
    let mut p = McpProxy::new(&opts, true).unwrap();
    resolve_as(&mut p, names);
    let p = Arc::new(p);

    let r = send(
        &p,
        "POST",
        &format!("https://mcp.example.com:{port}/mcp"),
        &["x-gaudium-proxy-header-content-type: application/json"],
        "{}",
    );
    assert_eq!(r.status, 200, "{}", r.text());
    assert_eq!(r.text(), format!("{{\"host\":\"mcp.example.com:{port}\"}}"));
    assert_eq!(r.header("mcp-session-id"), Some("tls-1"));

    // the same server under a name its certificate does not carry, or by its address
    for url in [
        format!("https://other.example.com:{port}/"),
        format!("https://127.0.0.1:{port}/"),
    ] {
        let r = get(&p, &url);
        assert_eq!(r.status, 502, "{url}");
        assert!(
            r.error().starts_with("proxy error: ") && r.error().contains("was not accepted"),
            "{url}: {}",
            r.error()
        );
    }

    // without the extra authority the certificate is unknown to the system's trust store
    let mut p = proxy(&[], &["mcp.example.com"]);
    resolve_as(&mut p, names);
    let p = Arc::new(p);
    let r = get(&p, &format!("https://mcp.example.com:{port}/"));
    assert_eq!(r.status, 502);
    assert!(
        r.error()
            .contains("certificate of mcp.example.com was not accepted"),
        "{}",
        r.error()
    );

    // a plain-http request to a TLS port is the target's failure, not a way in
    let r = get(&p, &format!("http://mcp.example.com:{port}/"));
    assert_eq!(r.status, 502);
}

#[test]
fn without_the_tls_library_https_targets_are_refused_and_http_ones_still_work() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let mut p = proxy(&["*"], &["127.0.0.1"]);
    p.tls = Err(crate::tls::missing_library_message());
    let p = Arc::new(p);
    assert!(p.enabled());
    assert!(p.status_json().contains("\"https_targets\":false"));

    let r = get(&p, "https://example.com/mcp");
    assert_eq!(r.status, 400);
    assert!(r.error().starts_with("mcp proxy: https targets cannot be reached: the system's OpenSSL library was not found"), "{}", r.error());
    assert_eq!(get(&p, &t.url("/")).status, 200);
    // a redirect to https is refused the same way: never fetched in the clear instead
    let hop = target(None, |_, s| {
        let _ = s.write_all(
            b"HTTP/1.1 302 Found\r\nLocation: https://example.com/x\r\nContent-Length: 0\r\n\r\n",
        );
    });
    let p2 = {
        let mut p = proxy(&["*"], &[&format!("127.0.0.1:{}", hop.port)]);
        p.tls = Err(crate::tls::missing_library_message());
        Arc::new(p)
    };
    let r = get(&p2, &hop.url("/"));
    assert_eq!(r.status, 403);
    assert!(
        r.error().contains("https targets cannot be reached"),
        "{}",
        r.error()
    );
}

#[test]
fn decisions_are_counted_and_listed_without_the_query_of_a_url() {
    let t = target(None, |_, s| ok_json(s, "", "{}"));
    let p = Arc::new(proxy(
        &["example.com:8931"],
        &[&format!("127.0.0.1:{}", t.port)],
    ));
    assert_eq!(get(&p, &t.url("/mcp?token=secret-in-query")).status, 200);
    assert_eq!(
        get(&p, "http://other.example.com/x?key=secret-in-query").status,
        403
    );
    let v = crate::json::parse_json(&p.status_json()).unwrap();
    let n = |k: &str| {
        v.get("counts")
            .and_then(|c| c.get(k))
            .and_then(|x| x.as_f64())
    };
    assert_eq!(
        (n("proxied"), n("refused"), n("failed")),
        (Some(1.0), Some(1.0), Some(0.0))
    );
    assert_eq!(v.get("enabled").and_then(|e| e.as_bool()), Some(true));
    let text = p.status_json();
    assert!(text.contains("\"allow\":[\"example.com:8931\"]"), "{text}");
    assert!(
        text.contains(&format!("\"allow_private\":[\"127.0.0.1:{}\"]", t.port)),
        "{text}"
    );
    assert!(
        text.contains("/mcp\"") && text.contains("other.example.com/x"),
        "{text}"
    );
    assert!(!text.contains("secret-in-query"), "{text}");

    // a user part is refused, and neither the log nor the answer nor the decisions hold it,
    // however the URL is written
    for url in [
        "http://user:secret-pw@example.com:8931/mcp",
        "https://user:secret-pw@example.com/",
        "user:secret-pw@example.com/x",
        "http:\\\\user:secret-pw@example.com/",
        "http://user:secret-pw@[::1",
        "user:secret-pw@example.com://x",
    ] {
        let r = get(&p, url);
        assert_eq!(r.status, 400, "{url}");
        assert!(!r.text().contains("secret-pw"), "{url}: {}", r.text());
    }
    let r = send(
        &p,
        "GET",
        &t.url("/"),
        &[
            "Sec-Fetch-Site: cross-site",
            "Origin: https://user:secret-pw@evil.example",
        ],
        "",
    );
    assert_eq!(r.status, 403);
    assert!(
        r.error().contains("(https://evil.example)"),
        "{}",
        r.error()
    );
    let text = p.status_json();
    assert!(!text.contains("secret-pw"), "{text}");
    assert!(
        text.contains("\"url\":\"http://example.com:8931/mcp\""),
        "{text}"
    );
}

#[test]
fn hosts_are_read_without_their_port_brackets_or_root_dot() {
    let n = |s: &str| NamedHost::Name(s.as_bytes().to_vec());
    assert_eq!(named_host("Example.COM"), n("Example.COM"));
    assert_eq!(named_host(" example.com:8090 "), n("example.com"));
    assert_eq!(named_host("example.com.:443"), n("example.com"));
    assert_eq!(named_host("\"example.com:443\""), n("example.com"));
    assert_eq!(named_host("192.0.2.7"), NamedHost::Addr);
    assert_eq!(named_host("192.0.2.7:8090"), NamedHost::Addr);
    assert_eq!(named_host("[2001:db8::7]"), NamedHost::Addr);
    assert_eq!(named_host("[2001:db8::7]:8090"), NamedHost::Addr);
    for bad in [
        "",
        "example.com:",
        "example.com:0",
        "example.com:65536",
        "example.com:x",
        "2001:db8::7",
        "[2001:db8::7",
        "[2001:db8::7]8090",
        "[example.com]",
        "exa mple.com",
        "example.com/x",
        "user@example.com",
    ] {
        assert_eq!(named_host(bad), NamedHost::Bad, "{bad:?}");
    }
}

#[test]
fn the_servers_own_requests_pass_the_same_gate_as_the_route() {
    let t = target(None, |_, s| ok_json(s, "", "{\"ok\":true}"));
    let p = proxy(&[], &["127.0.0.1"]);
    let url = t.url("/mcp");
    let fetch = |p: &McpProxy, key_ok: bool, site_ok: bool| {
        let mut body = Vec::new();
        p.fetch(
            Caller {
                key_ok,
                hosts_ours: true,
                site_ok,
            },
            Method::PxGet,
            &url,
            &[],
            None,
            &mut |_, d| {
                body.extend_from_slice(d);
                true
            },
        )
        .map(|f| (f.status, String::from_utf8_lossy(&body).into_owned()))
    };
    // made for a page of another site, for a request without the key, or with the proxy off:
    // refused before any connection
    let e = fetch(&p, true, false).unwrap_err();
    assert!(e.starts_with("mcp proxy: "), "{e}");
    assert!(e.contains("another site"), "{e}");
    let e = fetch(&p, false, true).unwrap_err();
    assert!(e.contains("API key"), "{e}");
    let e = fetch(&McpProxy::disabled(), true, true).unwrap_err();
    assert!(e.contains("--mcp-proxy"), "{e}");
    assert!(t.seen().is_empty(), "a refused fetch reached the target");
    // made for a request the route would serve
    assert_eq!(
        fetch(&p, true, true).unwrap(),
        (200, "{\"ok\":true}".to_string())
    );
    assert_eq!(t.seen().len(), 1);
}

#[test]
fn the_caller_verdict_is_the_routes_site_rule() {
    let req = |headers: &[(&str, &str)]| HttpRequest {
        method: "POST".into(),
        target: "/v1/messages".into(),
        path: "/v1/messages".into(),
        query: Default::default(),
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        body: Vec::new(),
    };
    let p = proxy(&[], &["127.0.0.1"]);
    let caller = |headers: &[(&str, &str)], key_ok: bool| p.caller(&req(headers), key_ok);
    let host = ("host", "localhost:8090");
    assert!(caller(&[host], true).site_ok);
    assert!(caller(&[host, ("origin", "http://localhost:8090")], true).site_ok);
    assert!(!caller(&[host, ("origin", "https://other.example")], true).site_ok);
    assert!(
        !caller(
            &[
                host,
                ("origin", "http://localhost:8090"),
                ("sec-fetch-site", "cross-site")
            ],
            true
        )
        .site_ok
    );
    assert!(!caller(&[host], false).key_ok);
    // a request sent to a name that is not this server's (DNS rebinding) fails the site rule
    // too, also with an origin of that name and from the same origin
    let theirs = [
        ("host", "rebound.example:8090"),
        ("origin", "http://rebound.example:8090"),
        ("sec-fetch-site", "same-origin"),
    ];
    let c = caller(&theirs, true);
    assert!(!c.hosts_ours && !c.site_ok, "{c:?}");
    let c = caller(&[host, ("x-forwarded-host", "rebound.example")], true);
    assert!(!c.hosts_ours && !c.site_ok, "{c:?}");
    assert!(caller(&[host], true).hosts_ours);
}
