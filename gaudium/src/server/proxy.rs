//! The MCP proxy: `GET | POST | DELETE /cors-proxy?url=<target>`.
//!
//! The Web UI's MCP client runs in the browser. A browser cannot read an MCP server that sends
//! no CORS headers, cannot reach one that only the server's machine reaches, and an HTTPS page
//! cannot call a plain-HTTP one at all. Through this route the server makes the request instead
//! and passes the answer back as it arrives (Streamable HTTP: JSON or an event stream).
//!
//! A route that fetches a URL named by the request is a forwarder inside the operator's
//! network, so every decision about a request is taken by the verified core
//! (`verified/mcp_proxy.bend`, through `core::proxy`): whether the proxy is offered, whether a
//! request runs, whether a target and each address it resolves to may be connected to,
//! redirects, headers, and the size and time caps. This file is the shell around those
//! decisions: text parsing, name lookup, sockets, TLS and the HTTP exchange with the target.
//!
//! The wire contract with the page (`ui/js/lib/mcp.js`):
//! - `url` is the absolute `http://` or `https://` URL of the target;
//! - only request headers named `x-gaudium-proxy-header-<name>` go to the target, as `<name>`;
//! - the target's status, headers (see `resp_header_kind`) and body come back unchanged;
//! - a request the proxy refuses is answered with a JSON error whose message starts with
//!   `mcp proxy: `; a target that could not be reached with one that starts with `proxy error: `.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::conn::Conn;
use super::{jq, HttpRequest};
use crate::core::proxy::{
    self as rule, AddrClass, AddrVerdict, Gate, ListMatch, Method, ReqHeader, RespHeader, Scheme,
    Site, Start, UrlVerdict,
};
use crate::tls::{TlsClient, TlsStream};

/// Request headers with this prefix are forwarded to the target without it.
pub(super) const HEADER_PREFIX: &str = "x-gaudium-proxy-header-";

const DEFAULT_MAX_RESPONSE_MB: u64 = 64;
const DEFAULT_IDLE_TIMEOUT_S: u64 = 600;
/// Largest `--mcp-proxy-max-mb` and `--mcp-proxy-timeout` values.
const MOST_RESPONSE_MB: u64 = 65_536;
const MOST_IDLE_TIMEOUT_S: u64 = 86_400;

/// Largest request body forwarded.
const MAX_REQUEST_BYTES: u64 = 8 << 20;
const CONNECT_TIMEOUT_MS: u64 = 10_000;
/// Longest one proxied exchange may last, however lively.
const TOTAL_TIMEOUT_MS: u64 = 3_600_000;
const MAX_REDIRECTS: u64 = 3;
/// An event stream that carries nothing gets a comment line this often, so that a front end
/// between the page and this server does not cut it as idle (common idle limits are 60 to 100 s).
const KEEPALIVE_MS: u64 = 20_000;
/// Largest response head of a target.
const MAX_HEAD_BYTES: usize = 64 * 1024;
/// Largest piece of the answer handed to one write call to the page (one TLS record), so that
/// each call is bounded by the time left of the exchange.
const PAGE_WRITE_PIECE: usize = 16 * 1024;
/// Decisions kept for `GET /gaudium/proxy`.
const RECENT_KEPT: usize = 50;

/// The proxy's settings as given on the command line.
#[derive(Debug, Clone)]
pub struct ProxyOptions {
    /// `--mcp-proxy`
    pub enabled: bool,
    /// `--mcp-proxy-allow`: targets that may be reached.
    pub allow: Vec<String>,
    /// `--mcp-proxy-allow-private`: targets that may be reached although their address is
    /// loopback, private or link-local.
    pub allow_private: Vec<String>,
    /// `--mcp-proxy-max-mb`: largest response forwarded.
    pub max_response_mb: u64,
    /// `--mcp-proxy-timeout`: longest silence of a target, in seconds.
    pub idle_timeout_s: u64,
    /// `--mcp-proxy-ca`: certificate authorities trusted for HTTPS targets besides the system's.
    pub ca_file: Option<PathBuf>,
}

impl Default for ProxyOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            allow: Vec::new(),
            allow_private: Vec::new(),
            max_response_mb: DEFAULT_MAX_RESPONSE_MB,
            idle_timeout_s: DEFAULT_IDLE_TIMEOUT_S,
            ca_file: None,
        }
    }
}

// ================================================================================================
// Allow-list entries
// ================================================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
enum Pattern {
    /// `*`: any target.
    Any,
    /// `*.example.com`: names under a domain (stored with its leading dot).
    Suffix(String),
    /// `mcp.example.com`
    Name(String),
    /// An address or an address range (`192.0.2.7`, `192.0.2.0/24`, `fd00::/8`).
    Net(IpAddr, u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    pat: Pattern,
    /// `Some`: the entry is for this port only.
    port: Option<u16>,
}

fn host_name_ok(h: &str) -> bool {
    !h.is_empty()
        && h.len() <= 253
        && h.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}

/// A host as it is compared: lower case, without the root dot (`example.com.` is `example.com`,
/// so the dotted form cannot slip past an entry).
fn normal_host(h: &str) -> String {
    h.trim_end_matches('.').to_ascii_lowercase()
}

fn parse_port(p: &str) -> Option<u16> {
    (!p.is_empty() && p.len() <= 5 && p.bytes().all(|b| b.is_ascii_digit()))
        .then(|| p.parse::<u16>().ok())
        .flatten()
        .filter(|p| *p != 0)
}

/// One entry of `--mcp-proxy-allow` / `--mcp-proxy-allow-private`: `*`, `*.domain`, `host`,
/// `host:port`, an address (`192.0.2.7`, `192.0.2.7:8931`, `[2001:db8::1]:8931`) or an address
/// range (`192.0.2.0/24`, `fd00::/8`).
fn parse_entry(raw: &str) -> Result<Entry, String> {
    let e = raw.trim();
    let bad = |why: &str| Err(format!("'{e}': {why}"));
    if e == "*" {
        return Ok(Entry {
            pat: Pattern::Any,
            port: None,
        });
    }
    if let Some((net, bits)) = e.split_once('/') {
        let Ok(net) = net.trim_matches(['[', ']']).parse::<IpAddr>() else {
            return bad("the part before '/' is not an address");
        };
        let Some(bits) = bits
            .parse::<u64>()
            .ok()
            .filter(|b| rule::net_bits_ok(net, *b))
        else {
            return bad("the part after '/' is not a prefix length of that address family");
        };
        return Ok(Entry {
            pat: Pattern::Net(net, bits),
            port: None,
        });
    }
    let full = |ip: IpAddr| Pattern::Net(ip, if ip.is_ipv4() { 32 } else { 128 });
    if let Some(rest) = e.strip_prefix('[') {
        let Some((addr, after)) = rest.split_once(']') else {
            return bad("'[' without ']'");
        };
        let Ok(ip) = addr.parse::<IpAddr>() else {
            return bad("not an address between the brackets");
        };
        let port = match after.strip_prefix(':') {
            None if after.is_empty() => None,
            Some(p) => match parse_port(p) {
                Some(p) => Some(p),
                None => return bad("not a port after ':'"),
            },
            None => return bad("unexpected text after ']'"),
        };
        return Ok(Entry {
            pat: full(ip),
            port,
        });
    }
    if let Ok(ip) = e.parse::<IpAddr>() {
        return Ok(Entry {
            pat: full(ip),
            port: None,
        });
    }
    let (host, port) = match e.rsplit_once(':') {
        Some((h, p)) => match parse_port(p) {
            Some(p) => (h, Some(p)),
            None => return bad("not a port after ':'"),
        },
        None => (e, None),
    };
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Ok(Entry {
            pat: full(ip),
            port,
        });
    }
    if let Some(domain) = host.strip_prefix("*.") {
        let domain = normal_host(domain);
        if !host_name_ok(&domain) {
            return bad("not a domain after '*.'");
        }
        return Ok(Entry {
            pat: Pattern::Suffix(format!(".{domain}")),
            port,
        });
    }
    let name = normal_host(host);
    if !host_name_ok(&name) {
        return bad("not a host name, an address or a range");
    }
    Ok(Entry {
        pat: Pattern::Name(name),
        port,
    })
}

fn parse_entries(flag: &str, raw: &[String]) -> Result<Vec<Entry>, String> {
    raw.iter()
        .flat_map(|s| s.split(','))
        .filter(|s| !s.trim().is_empty())
        .map(|s| parse_entry(s).map_err(|e| format!("{flag} {e}")))
        .collect()
}

impl Entry {
    fn port_fits(&self, port: u16) -> bool {
        self.port.is_none_or(|p| p == port)
    }

    /// The entry names the target as the URL gives it.
    fn names(&self, t: &Target) -> bool {
        self.port_fits(t.port)
            && match &self.pat {
                Pattern::Any => true,
                Pattern::Suffix(s) => t.literal.is_none() && t.host.ends_with(s.as_str()),
                Pattern::Name(n) => t.literal.is_none() && t.host == *n,
                Pattern::Net(net, bits) => t
                    .literal
                    .is_some_and(|ip| rule::net_contains(*net, *bits, ip)),
            }
    }

    /// The entry is a range that holds `ip`.
    fn holds(&self, ip: IpAddr, port: u16) -> bool {
        self.port_fits(port)
            && matches!(&self.pat, Pattern::Net(net, bits) if rule::net_contains(*net, *bits, ip))
    }

    /// The entry allows any target.
    fn is_open(&self) -> bool {
        match &self.pat {
            Pattern::Any => true,
            Pattern::Net(_, bits) => *bits == 0,
            _ => false,
        }
    }
}

// ================================================================================================
// The target URL
// ================================================================================================

#[derive(Debug, Clone)]
struct Target {
    scheme: Scheme,
    /// The host as compared: lower case, no brackets, no root dot.
    host: String,
    /// The host is an address literal.
    literal: Option<IpAddr>,
    port: u16,
    /// Path and query, starting with `/`.
    path: String,
}

impl Target {
    fn scheme_name(&self) -> &'static str {
        if self.scheme == Scheme::PxHttps {
            "https"
        } else {
            "http"
        }
    }

    fn default_port(&self) -> u16 {
        if self.scheme == Scheme::PxHttps {
            443
        } else {
            80
        }
    }

    /// `host` or `host:port` as the Host header carries it.
    fn authority(&self) -> String {
        let host = match self.literal {
            Some(IpAddr::V6(_)) => format!("[{}]", self.host),
            _ => self.host.clone(),
        };
        if self.port == self.default_port() {
            host
        } else {
            format!("{host}:{}", self.port)
        }
    }

    fn origin(&self) -> String {
        format!("{}://{}", self.scheme_name(), self.authority())
    }

    /// The URL for logs and messages: without its query, which may hold a credential.
    fn shown(&self) -> String {
        let path = self.path.split('?').next().unwrap_or("/");
        format!("{}{path}", self.origin())
    }
}

/// Parses a target URL. `tls`: `Ok` when HTTPS targets can be reached, else what is missing.
/// The error is the reason the URL is refused.
fn parse_target(url: &str, tls: Result<(), &str>) -> Result<Target, String> {
    // nothing that could end a line or a token of the request written to the target
    if let Some(c) = url
        .chars()
        .find(|c| c.is_control() || *c == ' ' || *c == '\u{fffd}')
    {
        return Err(format!(
            "the target URL holds a character that is not allowed (U+{:04X})",
            c as u32
        ));
    }
    let (scheme_text, rest) = url.split_once("://").unwrap_or(("", url));
    let scheme = match scheme_text.to_ascii_lowercase().as_str() {
        "http" => Scheme::PxHttp,
        "https" => Scheme::PxHttps,
        _ => Scheme::PxSchemeOther,
    };
    let rest = rest.split('#').next().unwrap_or("");
    let cut = rest.find(['/', '?']).unwrap_or(rest.len());
    let (authority, path) = rest.split_at(cut);
    let has_user = authority.contains('@');
    // `[address]` or `[address]:port`, else `host` or `host:port`
    let (host_text, port_text, bracketed) = match authority.strip_prefix('[') {
        Some(inner) => match inner.split_once(']') {
            Some((h, "")) => (h, None, true),
            Some((h, after)) if after.starts_with(':') => (h, Some(&after[1..]), true),
            _ => ("", None, true),
        },
        None => match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p), false),
            None => (authority, None, false),
        },
    };
    let host = normal_host(host_text);
    let literal = host.parse::<IpAddr>().ok();
    // an IPv6 literal is written in brackets, and only an IPv6 literal is
    let host_ok = match (bracketed, literal) {
        (true, Some(IpAddr::V6(_))) | (false, Some(IpAddr::V4(_))) => true,
        (false, None) => host_name_ok(&host),
        _ => false,
    };
    let default_port = if scheme == Scheme::PxHttps { 443 } else { 80 };
    // out of range reads as 0, which is not a port
    let port: u64 = match port_text {
        None | Some("") => default_port,
        Some(p) if p.len() <= 5 && p.bytes().all(|b| b.is_ascii_digit()) => p.parse().unwrap_or(0),
        Some(_) => 0,
    };
    // the text before `://` is shown only when it is a scheme: it could hold a user part
    let shown_scheme = if scheme_text.is_empty() {
        "none"
    } else if scheme_ok(scheme_text) {
        scheme_text
    } else {
        "not a scheme"
    };
    match rule::url_check(scheme, has_user, host_ok, port, tls.is_ok()) {
        UrlVerdict::PxUrlOk => {}
        UrlVerdict::PxUrlScheme => {
            return Err(format!(
                "the target URL must start with http:// or https:// (scheme: {shown_scheme})"
            ))
        }
        UrlVerdict::PxUrlUser => {
            return Err("a user name or password in the target URL is not supported".into())
        }
        UrlVerdict::PxUrlHost => return Err("the target URL has no valid host".into()),
        UrlVerdict::PxUrlPort => return Err("the target URL has no valid port".into()),
        UrlVerdict::PxUrlNoTls => {
            return Err(format!(
                "https targets cannot be reached: {}",
                tls.err().unwrap_or("TLS is not available")
            ))
        }
    }
    // bytes outside ASCII travel percent-encoded
    let mut clean = String::with_capacity(path.len() + 1);
    if !path.starts_with('/') {
        clean.push('/');
    }
    for b in path.bytes() {
        if b.is_ascii() {
            clean.push(b as char);
        } else {
            clean.push_str(&format!("%{b:02X}"));
        }
    }
    Ok(Target {
        scheme,
        host,
        literal,
        port: port as u16,
        path: clean,
    })
}

/// A URI scheme name (RFC 3986, section 3.1), of a length worth showing.
fn scheme_ok(s: &str) -> bool {
    s.len() <= 32
        && s.bytes().next().is_some_and(|b| b.is_ascii_alphabetic())
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
}

/// The URL a `Location` header points to, resolved against the whole URL of the request that
/// got it (RFC 3986, section 5.2): a reference with a scheme stands alone, `//host...` keeps the
/// scheme, `/path` the authority, a relative path is merged with the base's directory, `?query`
/// keeps the path, `#fragment` and the empty reference keep path and query; dot segments are
/// removed. The result is a URL to check like any other target (`parse_target`).
fn resolve_location(base: &Target, location: &str) -> String {
    // the parts of the reference (RFC 3986, appendix B)
    let r = location.trim();
    let (r, fragment) = match r.split_once('#') {
        Some((r, f)) => (r, Some(f)),
        None => (r, None),
    };
    let (r, query) = match r.split_once('?') {
        Some((r, q)) => (r, Some(q)),
        None => (r, None),
    };
    let (scheme, r) = match r.find(':') {
        Some(i) if i > 0 && !r[..i].contains('/') => (Some(&r[..i]), &r[i + 1..]),
        _ => (None, r),
    };
    let (authority, path) = match r.strip_prefix("//") {
        Some(rest) => {
            let cut = rest.find('/').unwrap_or(rest.len());
            (Some(&rest[..cut]), &rest[cut..])
        }
        None => (None, r),
    };
    let (base_path, base_query) = match base.path.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (base.path.as_str(), None),
    };

    // the target (section 5.2.2); the base always has an authority and a path that starts with `/`
    let (scheme, authority, path, query) = if let Some(s) = scheme {
        (
            s.to_string(),
            authority.map(str::to_string),
            remove_dot_segments(path),
            query,
        )
    } else if let Some(a) = authority {
        (
            base.scheme_name().to_string(),
            Some(a.to_string()),
            remove_dot_segments(path),
            query,
        )
    } else {
        let (path, query) = if path.is_empty() {
            (base_path.to_string(), query.or(base_query))
        } else if path.starts_with('/') {
            (remove_dot_segments(path), query)
        } else {
            // merged with the base's directory (section 5.2.3)
            let dir = &base_path[..base_path.rfind('/').map_or(0, |i| i + 1)];
            (remove_dot_segments(&format!("{dir}{path}")), query)
        };
        (
            base.scheme_name().to_string(),
            Some(base.authority()),
            path,
            query,
        )
    };

    // recomposed (section 5.3)
    let mut out = format!("{scheme}:");
    if let Some(a) = authority {
        out.push_str("//");
        out.push_str(&a);
    }
    out.push_str(&path);
    if let Some(q) = query {
        out.push('?');
        out.push_str(q);
    }
    if let Some(f) = fragment {
        out.push('#');
        out.push_str(f);
    }
    out
}

/// A path without its `.` and `..` segments (RFC 3986, section 5.2.4).
fn remove_dot_segments(path: &str) -> String {
    let mut input = path;
    // each segment with the `/` before it
    let mut out: Vec<&str> = Vec::new();
    loop {
        if let Some(rest) = input.strip_prefix("../") {
            input = rest;
        } else if let Some(rest) = input.strip_prefix("./") {
            input = rest;
        } else if input.starts_with("/./") {
            input = &input[2..];
        } else if input == "/." {
            input = "/";
        } else if input.starts_with("/../") {
            input = &input[3..];
            out.pop();
        } else if input == "/.." {
            input = "/";
            out.pop();
        } else if input == "." || input == ".." {
            input = "";
        } else if input.is_empty() {
            break;
        } else {
            let from = usize::from(input.starts_with('/'));
            let end = input[from..].find('/').map_or(input.len(), |i| i + from);
            out.push(&input[..end]);
            input = &input[end..];
        }
    }
    out.concat()
}

// ================================================================================================
// Headers
// ================================================================================================

/// A header name made of token characters only.
fn header_name_ok(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

fn header_value_ok(value: &str) -> bool {
    value.chars().all(|c| c == '\t' || !c.is_control())
}

/// The kind of a request header the page asks to forward (its name without the prefix, in lower
/// case).
fn req_header_kind(name: &str) -> ReqHeader {
    match name {
        "host"
        | "content-length"
        | "transfer-encoding"
        | "connection"
        | "keep-alive"
        | "te"
        | "trailer"
        | "upgrade"
        | "accept-encoding"
        | "expect"
        | "proxy-authorization"
        | "proxy-authenticate"
        | "proxy-connection" => ReqHeader::PxRhFraming,
        "accept"
        | "content-type"
        | "accept-language"
        | "mcp-protocol-version"
        | "user-agent"
        | "last-event-id" => ReqHeader::PxRhNeutral,
        _ => ReqHeader::PxRhBound,
    }
}

/// The kind of a response header of the target (name in lower case).
fn resp_header_kind(name: &str) -> RespHeader {
    match name {
        "transfer-encoding" | "content-length" | "connection" | "keep-alive" | "trailer"
        | "upgrade" | "proxy-authenticate" | "proxy-connection" => RespHeader::PxRpFraming,
        "set-cookie" | "set-cookie2" | "clear-site-data" => RespHeader::PxRpState,
        "server"
        | "location"
        | "refresh"
        | "content-security-policy"
        | "content-security-policy-report-only"
        | "strict-transport-security"
        | "alt-svc"
        | "service-worker-allowed"
        | "x-content-type-options"
        | "cache-control"
        | "x-accel-buffering" => RespHeader::PxRpPolicy,
        n if n.starts_with("access-control-") => RespHeader::PxRpCors,
        _ => RespHeader::PxRpPlain,
    }
}

/// The headers that go to the target on this hop (names in lower case, without the prefix).
fn forwarded_headers(req: &HttpRequest, same_origin: bool) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = req
        .headers
        .iter()
        .filter_map(|(k, v)| {
            let name = k.strip_prefix(HEADER_PREFIX);
            let prefixed = name.is_some_and(header_name_ok) && header_value_ok(v);
            let name = name.unwrap_or(k);
            rule::req_header_goes(prefixed, req_header_kind(name), same_origin)
                .then(|| (name.to_string(), v.clone()))
        })
        .collect();
    out.sort();
    out
}

/// What the request's own headers say about where it comes from.
fn request_site(req: &HttpRequest) -> (Site, bool, bool) {
    let site = match req.headers.get("sec-fetch-site").map(|s| s.trim()) {
        None => Site::PxSiteAbsent,
        Some(s) if s.eq_ignore_ascii_case("same-origin") => Site::PxSiteSameOrigin,
        Some(s) if s.eq_ignore_ascii_case("none") => Site::PxSiteNone,
        Some(_) => Site::PxSiteOther,
    };
    let origin = req.headers.get("origin").map(|s| s.trim());
    let origin_is_host = origin.is_some_and(|o| origin_is_request_host(o, req));
    (site, origin.is_some(), origin_is_host)
}

/// The origin names the host this request was sent to: the request's own `Host`, or the host a
/// front end says it received the request for (`X-Forwarded-Host`, `Forwarded: host=`). Only the
/// request is looked at; the server has no name of its own to compare with.
fn origin_is_request_host(origin: &str, req: &HttpRequest) -> bool {
    let Some((scheme, authority)) = origin.split_once("://") else {
        return false;
    };
    let default_port = match scheme.to_ascii_lowercase().as_str() {
        "http" => ":80",
        "https" => ":443",
        _ => "",
    };
    let normal = |a: &str| {
        let a = a.trim().trim_matches('"').to_ascii_lowercase();
        match a.strip_suffix(default_port) {
            Some(bare) if !default_port.is_empty() => bare.to_string(),
            _ => a,
        }
    };
    let want = normal(authority);
    if want.is_empty() {
        return false;
    }
    let mut named: Vec<String> = Vec::new();
    if let Some(h) = req.headers.get("host") {
        named.push(normal(h));
    }
    if let Some(h) = req.headers.get("x-forwarded-host") {
        named.extend(h.split(',').map(normal));
    }
    if let Some(f) = req.headers.get("forwarded") {
        for part in f.split([',', ';']) {
            if let Some((k, v)) = part.split_once('=') {
                if k.trim().eq_ignore_ascii_case("host") {
                    named.push(normal(v));
                }
            }
        }
    }
    named.iter().any(|n| *n == want)
}

// ================================================================================================
// The connection to the target
// ================================================================================================

enum Upstream {
    Plain(TcpStream),
    Tls(TlsStream),
}

impl Upstream {
    fn set_read_timeout(&self, ms: u64) {
        let t = Some(Duration::from_millis(ms.max(1)));
        let _ = match self {
            Upstream::Plain(s) => s.set_read_timeout(t),
            Upstream::Tls(s) => s.set_read_timeout(t),
        };
    }
}

impl Read for Upstream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Upstream::Plain(s) => s.read(buf),
            Upstream::Tls(s) => s.read(buf),
        }
    }
}

impl Write for Upstream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Upstream::Plain(s) => s.write(buf),
            Upstream::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Upstream::Plain(s) => s.flush(),
            Upstream::Tls(s) => s.flush(),
        }
    }
}

fn timed_out(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

/// The head of a target's response.
struct Head {
    status: u16,
    reason: String,
    /// `(name in lower case, name as sent, value)`
    headers: Vec<(String, String, String)>,
}

impl Head {
    fn get(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|h| h.0 == name)
            .map(|h| h.2.as_str())
    }
}

fn find_head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
        .or_else(|| buf.windows(2).position(|w| w == b"\n\n").map(|i| i + 2))
}

fn parse_head(text: &str) -> Result<Head, String> {
    let mut lines = text.lines();
    let status_line = lines.next().unwrap_or("");
    let mut parts = status_line.splitn(3, ' ');
    let version = parts.next().unwrap_or("");
    let status = parts
        .next()
        .filter(|s| s.len() == 3)
        .and_then(|s| s.parse::<u16>().ok())
        .filter(|s| (100..=599).contains(s));
    let (true, Some(status)) = (version.starts_with("HTTP/1."), status) else {
        return Err("the target did not answer with HTTP/1.1".into());
    };
    let reason: String = parts
        .next()
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_graphic() || *c == ' ')
        .take(64)
        .collect();
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let (name, value) = (name.trim(), value.trim());
        if header_name_ok(name) && header_value_ok(value) {
            headers.push((
                name.to_ascii_lowercase(),
                name.to_string(),
                value.to_string(),
            ));
        }
    }
    Ok(Head {
        status,
        reason,
        headers,
    })
}

/// How the body of a response is delimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Framing {
    /// No body (the status has none).
    Empty,
    /// This many bytes are left.
    Length(u64),
    Chunked(ChunkState),
    /// Until the target closes the connection.
    ToClose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkState {
    /// A chunk-size line is next.
    Size,
    /// This many bytes of the current chunk are left.
    Data(u64),
    /// The line end after a chunk's data is next.
    DataEnd,
    /// Trailer lines up to an empty line.
    Trailers,
}

/// One step of taking body bytes out of what was read so far.
enum Step {
    /// The first `usize` bytes of the buffer are body bytes.
    Data(usize),
    /// More must be read.
    More,
    End,
    Bad(&'static str),
}

/// Takes framing bytes off the front of `buf` and says what its front now is.
fn body_step(framing: &mut Framing, buf: &mut Vec<u8>) -> Step {
    loop {
        match framing {
            Framing::Empty => return Step::End,
            Framing::Length(0) => return Step::End,
            Framing::Length(left) => {
                if buf.is_empty() {
                    return Step::More;
                }
                let n = (*left).min(buf.len() as u64);
                *left -= n;
                return Step::Data(n as usize);
            }
            Framing::ToClose => {
                return if buf.is_empty() {
                    Step::More
                } else {
                    Step::Data(buf.len())
                };
            }
            Framing::Chunked(state) => match *state {
                ChunkState::Size => {
                    let Some(eol) = buf.iter().position(|b| *b == b'\n') else {
                        return if buf.len() > 256 {
                            Step::Bad("a chunk size line is too long")
                        } else {
                            Step::More
                        };
                    };
                    let line = String::from_utf8_lossy(&buf[..eol]).into_owned();
                    buf.drain(..=eol);
                    let hex = line.trim().split(';').next().unwrap_or("").trim();
                    let Ok(size) = u64::from_str_radix(hex, 16) else {
                        return Step::Bad("a chunk size is not a number");
                    };
                    *state = if size == 0 {
                        ChunkState::Trailers
                    } else {
                        ChunkState::Data(size)
                    };
                }
                ChunkState::Data(left) => {
                    if buf.is_empty() {
                        return Step::More;
                    }
                    let n = left.min(buf.len() as u64);
                    *state = if left == n {
                        ChunkState::DataEnd
                    } else {
                        ChunkState::Data(left - n)
                    };
                    return Step::Data(n as usize);
                }
                ChunkState::DataEnd => {
                    let Some(eol) = buf.iter().position(|b| *b == b'\n') else {
                        return if buf.len() > 2 {
                            Step::Bad("a chunk does not end where its size says")
                        } else {
                            Step::More
                        };
                    };
                    if eol > 1 {
                        return Step::Bad("a chunk does not end where its size says");
                    }
                    buf.drain(..=eol);
                    *state = ChunkState::Size;
                }
                ChunkState::Trailers => {
                    let Some(eol) = buf.iter().position(|b| *b == b'\n') else {
                        return if buf.len() > MAX_HEAD_BYTES {
                            Step::Bad("the trailers are too long")
                        } else {
                            Step::More
                        };
                    };
                    let blank = buf[..eol].iter().all(|b| *b == b'\r');
                    buf.drain(..=eol);
                    if blank {
                        return Step::End;
                    }
                }
            },
        }
    }
}

// ================================================================================================
// The proxy
// ================================================================================================

/// The limits of one proxied exchange (tests shorten them).
#[derive(Debug, Clone, Copy)]
struct Limits {
    max_request: u64,
    max_response: u64,
    connect_ms: u64,
    idle_ms: u64,
    total_ms: u64,
    keepalive_ms: u64,
    redirects: u64,
}

type Resolver = Box<dyn Fn(&str, u16) -> io::Result<Vec<IpAddr>> + Send + Sync>;

fn system_resolver(host: &str, port: u16) -> io::Result<Vec<IpAddr>> {
    let mut out: Vec<IpAddr> = Vec::new();
    for a in (host, port).to_socket_addrs()? {
        if !out.contains(&a.ip()) {
            out.push(a.ip());
        }
    }
    Ok(out)
}

/// Why a request got no answer from a target.
enum Failure {
    /// The proxy did not try (status, reason): `mcp proxy: <reason>`.
    Refused(u16, String),
    /// The target could not be reached or did not answer properly: `proxy error: <reason>`.
    Unreachable(String),
}

pub(super) struct McpProxy {
    start: Start,
    allow: Vec<Entry>,
    allow_private: Vec<Entry>,
    limits: Limits,
    /// The TLS client for HTTPS targets, or what is missing on this machine.
    tls: Result<TlsClient, String>,
    resolver: Resolver,
    proxied: AtomicU64,
    refused: AtomicU64,
    failed: AtomicU64,
    /// Answers ended because the page stopped reading them.
    stalled: AtomicU64,
    recent: Mutex<VecDeque<String>>,
}

impl McpProxy {
    /// A proxy that is off (a server started without `--mcp-proxy`).
    #[cfg(test)]
    pub fn disabled() -> McpProxy {
        McpProxy::new(&ProxyOptions::default(), false).expect("the default options are valid")
    }

    /// Builds the proxy from its options. `keyed`: the server has an API key. The error is the
    /// reason the server does not start with these options.
    pub fn new(opts: &ProxyOptions, keyed: bool) -> Result<McpProxy, String> {
        let allow = parse_entries("--mcp-proxy-allow", &opts.allow)?;
        let allow_private = parse_entries("--mcp-proxy-allow-private", &opts.allow_private)?;
        if allow_private.iter().any(Entry::is_open) {
            return Err(
                "--mcp-proxy-allow-private with '*' or a /0 range: name the private targets (hosts, addresses or ranges); every private address at once is not accepted"
                    .into(),
            );
        }
        if !rule::cap_ok(opts.max_response_mb, MOST_RESPONSE_MB) {
            return Err(format!(
                "--mcp-proxy-max-mb {}: must be between 1 and {MOST_RESPONSE_MB}",
                opts.max_response_mb
            ));
        }
        if !rule::cap_ok(opts.idle_timeout_s, MOST_IDLE_TIMEOUT_S) {
            return Err(format!(
                "--mcp-proxy-timeout {}: must be between 1 and {MOST_IDLE_TIMEOUT_S} seconds",
                opts.idle_timeout_s
            ));
        }
        let listed = !allow.is_empty() || !allow_private.is_empty();
        let open = allow.iter().chain(&allow_private).any(Entry::is_open);
        let start = rule::start(opts.enabled, listed, open, keyed);
        match start {
            Start::PxStNoList => {
                return Err(
                    "--mcp-proxy needs the targets it may reach: add --mcp-proxy-allow <hosts> (public MCP servers) or --mcp-proxy-allow-private <hosts> (MCP servers on this machine or the local network)"
                        .into(),
                )
            }
            Start::PxStNoKey => {
                return Err(
                    "--mcp-proxy with a list that allows any target ('*' or a /0 range) needs --api-key: without a key everyone who reaches the port could use the server as an open forwarder. Name the targets, or set a key"
                        .into(),
                )
            }
            Start::PxStOff | Start::PxStOn => {}
        }
        debug_assert!(!rule::start_refused(start));
        let tls = if rule::start_serves(start) {
            TlsClient::new(opts.ca_file.as_deref())
        } else {
            Err("the proxy is off".into())
        };
        if let (true, Some(ca), Err(e)) = (rule::start_serves(start), &opts.ca_file, &tls) {
            // a trust file that was asked for and cannot be used is a configuration error
            return Err(format!("--mcp-proxy-ca {}: {e}", ca.display()));
        }
        Ok(McpProxy {
            start,
            allow,
            allow_private,
            limits: Limits {
                max_request: MAX_REQUEST_BYTES,
                max_response: rule::cap_bytes(opts.max_response_mb),
                connect_ms: CONNECT_TIMEOUT_MS,
                idle_ms: opts.idle_timeout_s.saturating_mul(1000),
                total_ms: TOTAL_TIMEOUT_MS,
                keepalive_ms: KEEPALIVE_MS,
                redirects: MAX_REDIRECTS,
            },
            tls,
            resolver: Box::new(system_resolver),
            proxied: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            stalled: AtomicU64::new(0),
            recent: Mutex::new(VecDeque::new()),
        })
    }

    /// The proxy is offered (`/props` reports this).
    pub fn enabled(&self) -> bool {
        rule::start_serves(self.start)
    }

    /// One line for the server's start-up message; `None` when the proxy is off.
    pub fn summary(&self, keyed: bool) -> Option<String> {
        self.enabled().then(|| {
            format!(
                "MCP proxy on: {} target entr{}, {} of them private; https targets: {}{}",
                self.allow.len() + self.allow_private.len(),
                if self.allow.len() + self.allow_private.len() == 1 {
                    "y"
                } else {
                    "ies"
                },
                self.allow_private.len(),
                match &self.tls {
                    Ok(_) => "yes".to_string(),
                    Err(e) => format!("no ({e})"),
                },
                if keyed {
                    ""
                } else {
                    "; no API key is set, so everyone who reaches this port reaches those targets"
                }
            )
        })
    }

    fn note(&self, counter: &AtomicU64, line: String, entry: String) {
        counter.fetch_add(1, Ordering::Relaxed);
        eprintln!("[mcp-proxy] {line}");
        let at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut recent = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        recent.push_back(format!("{{\"time\":{at},{entry}}}"));
        while recent.len() > RECENT_KEPT {
            recent.pop_front();
        }
    }

    /// `GET /gaudium/proxy`: the configuration, counters and the last decisions.
    pub fn status_json(&self) -> String {
        let list = |entries: &[Entry]| {
            entries
                .iter()
                .map(|e| jq(&entry_text(e)))
                .collect::<Vec<_>>()
                .join(",")
        };
        let recent = self
            .recent
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"enabled\":{},\"allow\":[{}],\"allow_private\":[{}],\"https_targets\":{},\
             \"limits\":{{\"max_request_bytes\":{},\"max_response_bytes\":{},\"connect_timeout_ms\":{},\
             \"idle_timeout_ms\":{},\"total_timeout_ms\":{},\"keepalive_ms\":{},\"max_redirects\":{}}},\
             \"counts\":{{\"proxied\":{},\"refused\":{},\"failed\":{},\"stalled\":{}}},\"recent\":[{}]}}",
            self.enabled(),
            list(&self.allow),
            list(&self.allow_private),
            self.tls.is_ok(),
            self.limits.max_request,
            self.limits.max_response,
            self.limits.connect_ms,
            self.limits.idle_ms,
            self.limits.total_ms,
            self.limits.keepalive_ms,
            self.limits.redirects,
            self.proxied.load(Ordering::Relaxed),
            self.refused.load(Ordering::Relaxed),
            self.failed.load(Ordering::Relaxed),
            self.stalled.load(Ordering::Relaxed),
            recent,
        )
    }

    /// How the lists match a target as its URL gives it.
    fn list_match(&self, t: &Target) -> ListMatch {
        let mut entries = self.allow.iter().chain(&self.allow_private);
        if entries.clone().any(|e| e.names(t)) {
            ListMatch::PxMEntry
        } else if t.literal.is_none()
            && entries.any(|e| matches!(e.pat, Pattern::Net(..)) && e.port_fits(t.port))
        {
            // a name, and ranges that could admit it: decided on the addresses it resolves to
            ListMatch::PxMRange
        } else {
            ListMatch::PxMNone
        }
    }

    /// The operator listed this target, or a range holding this address, as one that may be
    /// non-public.
    fn private_listed(&self, t: &Target, ip: IpAddr) -> bool {
        self.allow_private.iter().any(|e| {
            e.holds(ip, t.port)
                || (e.port_fits(t.port)
                    && match &e.pat {
                        Pattern::Name(n) => t.literal.is_none() && t.host == *n,
                        Pattern::Suffix(s) => t.literal.is_none() && t.host.ends_with(s.as_str()),
                        _ => false,
                    })
        })
    }

    /// The addresses of a target that may be connected to: every address the name resolves to,
    /// each one checked. One refused address refuses the target.
    fn checked_addresses(&self, t: &Target) -> Result<Vec<IpAddr>, Failure> {
        let m = self.list_match(t);
        let host_port = format!("{}:{}", t.host, t.port);
        if !rule::match_resolves(m) {
            return Err(Failure::Refused(
                403,
                format!(
                    "target {host_port} is not in the allow-list (--mcp-proxy-allow, --mcp-proxy-allow-private)"
                ),
            ));
        }
        let addrs = match t.literal {
            Some(ip) => vec![ip],
            None => (self.resolver)(&t.host, t.port)
                .map_err(|e| Failure::Unreachable(format!("cannot resolve {}: {e}", t.host)))?,
        };
        let mut verdict: Option<AddrVerdict> = None;
        let mut culprit: Option<(IpAddr, AddrClass)> = None;
        for ip in &addrs {
            let class = rule::addr_class(*ip);
            let in_range = self
                .allow
                .iter()
                .chain(&self.allow_private)
                .any(|e| e.holds(*ip, t.port));
            let v = rule::addr_check(m, in_range, class, self.private_listed(t, *ip));
            if culprit.is_none() && !rule::addr_passes(v) {
                culprit = Some((*ip, class));
            }
            verdict = Some(match verdict {
                None => v,
                Some(acc) => rule::addr_join(acc, v),
            });
        }
        let Some(verdict) = verdict else {
            return Err(Failure::Unreachable(format!("{} has no address", t.host)));
        };
        if rule::addr_passes(verdict) {
            return Ok(addrs);
        }
        let (ip, class) = culprit.unwrap_or((addrs[0], AddrClass::PxAcNever));
        let what = match class {
            AddrClass::PxAcLoopback => "a loopback address",
            AddrClass::PxAcPrivate => "a private address",
            AddrClass::PxAcLinkLocal => {
                "a link-local address (where instance-metadata endpoints live)"
            }
            AddrClass::PxAcNever => "an address that is never connected to",
            AddrClass::PxAcPublic => "a public address",
        };
        let at = if t.literal.is_some() {
            format!("{ip} is {what}")
        } else {
            format!("{} resolves to {ip}, {what}", t.host)
        };
        let why = match verdict {
            AddrVerdict::PxAddrUnlisted => format!(
                "target {host_port} is not in the allow-list (--mcp-proxy-allow, --mcp-proxy-allow-private)"
            ),
            AddrVerdict::PxAddrOutside => format!(
                "{} resolves to {ip}, outside the address ranges of the allow-list (a name admitted through a range must resolve only inside the ranges)",
                t.host
            ),
            AddrVerdict::PxAddrPrivate => {
                format!("{at}: not allowed without --mcp-proxy-allow-private")
            }
            AddrVerdict::PxAddrNever | AddrVerdict::PxAddrOk => format!("{at}: not allowed"),
        };
        Err(Failure::Refused(403, why))
    }

    /// Connects to the first checked address that answers, and starts TLS for an https target
    /// (the name of the URL is sent and verified, whatever address the socket went to).
    fn connect(&self, t: &Target, addrs: &[IpAddr]) -> Result<Upstream, Failure> {
        let mut last = String::from("no address");
        for ip in addrs {
            let sa = SocketAddr::new(*ip, t.port);
            match TcpStream::connect_timeout(&sa, Duration::from_millis(self.limits.connect_ms)) {
                Ok(tcp) => {
                    let _ = tcp.set_nodelay(true);
                    let _ = tcp.set_write_timeout(Some(Duration::from_secs(60)));
                    let _ =
                        tcp.set_read_timeout(Some(Duration::from_millis(self.limits.connect_ms)));
                    if t.scheme != Scheme::PxHttps {
                        return Ok(Upstream::Plain(tcp));
                    }
                    let client = self.tls.as_ref().map_err(|e| {
                        Failure::Refused(403, format!("https targets cannot be reached: {e}"))
                    })?;
                    return client
                        .connect(tcp, &t.host)
                        .map(Upstream::Tls)
                        .map_err(|e| Failure::Unreachable(format!("{}: {e}", t.authority())));
                }
                Err(e) => last = format!("{sa}: {e}"),
            }
        }
        Err(Failure::Unreachable(format!(
            "cannot connect to {} ({last})",
            t.authority()
        )))
    }

    /// Sends one request and reads the head of the answer. What was read past the head is left
    /// in `buf`.
    #[allow(clippy::too_many_arguments)]
    fn exchange(
        &self,
        up: &mut Upstream,
        t: &Target,
        method: Method,
        headers: &[(String, String)],
        body: Option<&[u8]>,
        buf: &mut Vec<u8>,
        started: Instant,
    ) -> Result<Head, Failure> {
        let unreachable = |e: String| Failure::Unreachable(format!("{}: {e}", t.authority()));
        let mut out = format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\n",
            method_name(method),
            t.path,
            t.authority()
        );
        for (k, v) in headers {
            out.push_str(&format!("{k}: {v}\r\n"));
        }
        if !headers.iter().any(|h| h.0 == "user-agent") {
            out.push_str("User-Agent: gaudium-mcp-proxy\r\n");
        }
        if body.is_some() || method == Method::PxPost {
            out.push_str(&format!(
                "Content-Length: {}\r\n",
                body.map_or(0, <[u8]>::len)
            ));
        }
        out.push_str("Connection: close\r\n\r\n");
        up.write_all(out.as_bytes())
            .and_then(|()| up.write_all(body.unwrap_or(&[])))
            .and_then(|()| up.flush())
            .map_err(|e| unreachable(format!("sending the request failed: {e}")))?;

        let waited_from = Instant::now();
        loop {
            if let Some(end) = find_head_end(buf) {
                let head =
                    parse_head(&String::from_utf8_lossy(&buf[..end])).map_err(unreachable)?;
                buf.drain(..end);
                if (100..200).contains(&head.status) && head.status != 101 {
                    continue; // an interim answer: the real one follows
                }
                return Ok(head);
            }
            if buf.len() > MAX_HEAD_BYTES {
                return Err(unreachable("the response head is too long".into()));
            }
            let idle_left = rule::time_left(ms_since(waited_from), self.limits.idle_ms);
            let total_left = rule::time_left(ms_since(started), self.limits.total_ms);
            if rule::expired(ms_since(waited_from), self.limits.idle_ms)
                || rule::expired(ms_since(started), self.limits.total_ms)
            {
                return Err(unreachable(format!(
                    "no answer within {} s",
                    ms_since(waited_from) / 1000
                )));
            }
            up.set_read_timeout(rule::read_wait(idle_left, total_left, self.limits.idle_ms));
            let mut chunk = [0u8; 8192];
            match up.read(&mut chunk) {
                Ok(0) => return Err(unreachable("the connection closed before an answer".into())),
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(e) if timed_out(&e) || e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(unreachable(format!("reading the answer failed: {e}"))),
            }
        }
    }

    /// The route handler. `key_ok`: the request carries the server's API key, or there is none.
    pub fn handle(&self, req: &HttpRequest, key_ok: bool, stream: &mut Conn) {
        // No write to the page outlasts the exchange's limits, the proxy's own short answers
        // included (`write_page` bounds each write of a relayed answer more closely). The
        // connection ends with the exchange (`Connection: close`), so the bound is not undone.
        let _ = stream.set_write_timeout(Some(Duration::from_millis(
            rule::write_wait(self.limits.idle_ms, self.limits.total_ms).max(1),
        )));
        let method = match req.method.as_str() {
            "GET" => Some(Method::PxGet),
            "POST" => Some(Method::PxPost),
            "DELETE" => Some(Method::PxDelete),
            _ => None,
        };
        let (site, has_origin, origin_is_host) = request_site(req);
        let site_ok = rule::site_ok(site, has_origin, origin_is_host);
        let url = req.query.get("url").map(String::as_str).unwrap_or("");
        let refuse = |stream: &mut Conn, status: u16, kind: &str, why: String| {
            self.note(
                &self.refused,
                format!("refused {} {}: {why}", req.method, shown_url(url)),
                format!(
                    "\"result\":\"refused\",\"method\":{},\"url\":{},\"status\":{status},\"reason\":{}",
                    jq(&req.method),
                    jq(&shown_url(url)),
                    jq(&why)
                ),
            );
            write_proxy_error(stream, status, kind, &format!("mcp proxy: {why}"));
        };
        let method = match (
            rule::gate(key_ok, self.enabled(), site_ok, method.is_some()),
            method,
        ) {
            (Gate::PxGateRun, Some(m)) => m,
            (Gate::PxGateNoKey, _) => {
                super::write_error(
                    stream,
                    401,
                    "authentication_error",
                    "invalid or missing API key",
                );
                return;
            }
            (Gate::PxGateDisabled, _) => {
                write_proxy_error(
                    stream,
                    403,
                    "feature_disabled",
                    "mcp proxy: this server was started without --mcp-proxy",
                );
                return;
            }
            (Gate::PxGateForeign, _) => {
                let origin = req
                    .headers
                    .get("origin")
                    .map_or("another site", String::as_str);
                return refuse(
                    stream,
                    403,
                    "permission_error",
                    format!(
                        "the request comes from a page of another origin ({}), not from this server's own page",
                        shown_url(origin).chars().take(120).collect::<String>()
                    ),
                );
            }
            (Gate::PxGateMethod, _) | (Gate::PxGateRun, None) => {
                return refuse(
                    stream,
                    405,
                    "invalid_request_error",
                    format!(
                        "method {} is not forwarded (GET, POST and DELETE are)",
                        req.method
                    ),
                );
            }
        };
        if url.is_empty() {
            return refuse(
                stream,
                400,
                "invalid_request_error",
                "missing 'url' parameter".into(),
            );
        }
        let tls_state = self.tls.as_ref().map(|_| ()).map_err(String::as_str);
        let mut target = match parse_target(url, tls_state) {
            Ok(t) => t,
            Err(why) => return refuse(stream, 400, "invalid_request_error", why),
        };
        if !rule::body_fits(req.body.len() as u64, self.limits.max_request) {
            return refuse(
                stream,
                413,
                "invalid_request_error",
                format!(
                    "the request body of {} bytes exceeds the limit of {} bytes",
                    req.body.len(),
                    self.limits.max_request
                ),
            );
        }

        let named_origin = target.origin();
        let mut same_origin = true;
        let mut method = method;
        let mut body: Option<&[u8]> = (!req.body.is_empty()).then_some(req.body.as_slice());
        let mut redirects_left = self.limits.redirects;
        let mut hop = 0u32;
        let started = Instant::now();
        loop {
            let prefix = if hop == 0 {
                String::new()
            } else {
                format!("redirect to {}: ", target.shown())
            };
            let step = self.checked_addresses(&target).and_then(|addrs| {
                let mut up = self.connect(&target, &addrs)?;
                let mut buf = Vec::new();
                let headers = forwarded_headers(req, same_origin);
                let head =
                    self.exchange(&mut up, &target, method, &headers, body, &mut buf, started)?;
                Ok((up, buf, head))
            });
            let (up, buf, head) = match step {
                Ok(x) => x,
                Err(Failure::Refused(status, why)) => {
                    return refuse(stream, status, "permission_error", format!("{prefix}{why}"));
                }
                Err(Failure::Unreachable(why)) => {
                    self.note(
                        &self.failed,
                        format!("failed {} {}: {why}", method_name(method), target.shown()),
                        format!(
                            "\"result\":\"failed\",\"method\":{},\"url\":{},\"reason\":{}",
                            jq(method_name(method)),
                            jq(&target.shown()),
                            jq(&why)
                        ),
                    );
                    write_proxy_error(
                        stream,
                        502,
                        "upstream_error",
                        &format!("proxy error: {why}"),
                    );
                    return;
                }
            };

            let location = head.get("location").filter(|l| !l.trim().is_empty());
            let kind = rule::redirect_kind(head.status, location.is_some());
            if let (true, Some(location)) = (rule::redirect_follows(kind), location) {
                drop(up);
                if !rule::redirect_may(redirects_left) {
                    return refuse(
                        stream,
                        403,
                        "permission_error",
                        format!("too many redirects (more than {})", self.limits.redirects),
                    );
                }
                redirects_left = rule::redirect_left(redirects_left);
                let next = match parse_target(&resolve_location(&target, location), tls_state) {
                    Ok(t) => t,
                    Err(why) => {
                        return refuse(
                            stream,
                            403,
                            "permission_error",
                            format!("redirect from {}: {why}", target.shown()),
                        )
                    }
                };
                eprintln!(
                    "[mcp-proxy] redirect {} {} -> {}",
                    head.status,
                    target.shown(),
                    next.shown()
                );
                if !rule::redirect_keeps_body(method, kind) {
                    body = None;
                }
                method = rule::redirect_method(method, kind);
                // once a redirect left the origin the page named, its headers stay behind
                same_origin = rule::hop_same_origin(same_origin, next.origin() == named_origin);
                target = next;
                hop += 1;
                continue;
            }

            self.note(
                &self.proxied,
                format!(
                    "{} {} -> {}",
                    method_name(method),
                    target.shown(),
                    head.status
                ),
                format!(
                    "\"result\":\"proxied\",\"method\":{},\"url\":{},\"status\":{}",
                    jq(method_name(method)),
                    jq(&target.shown()),
                    head.status
                ),
            );
            self.relay(stream, up, buf, &head, &target, started);
            return;
        }
    }

    /// Passes a target's answer to the page as it arrives.
    fn relay(
        &self,
        stream: &mut Conn,
        mut up: Upstream,
        mut buf: Vec<u8>,
        head: &Head,
        target: &Target,
        started: Instant,
    ) {
        let chunked = head
            .get("transfer-encoding")
            .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
        let mut framing = if matches!(head.status, 100..=199 | 204 | 304) {
            Framing::Empty
        } else if chunked {
            Framing::Chunked(ChunkState::Size)
        } else {
            match head.get("content-length").map(|v| v.parse::<u64>()) {
                Some(Ok(n)) => Framing::Length(n),
                Some(Err(_)) => {
                    write_proxy_error(
                        stream,
                        502,
                        "upstream_error",
                        "proxy error: the target sent an invalid Content-Length",
                    );
                    return;
                }
                None => Framing::ToClose,
            }
        };
        if let Framing::Length(n) = framing {
            if rule::resp_cut(0, n, self.limits.max_response) {
                eprintln!(
                    "[mcp-proxy] {}: response of {n} bytes exceeds the limit of {} bytes",
                    target.shown(),
                    self.limits.max_response
                );
                write_proxy_error(
                    stream,
                    502,
                    "upstream_error",
                    &format!(
                        "proxy error: the response of {n} bytes exceeds the limit of {} bytes (--mcp-proxy-max-mb)",
                        self.limits.max_response
                    ),
                );
                return;
            }
        }
        // an encoded body is not text the proxy could add a line to
        let encoded = head
            .get("content-encoding")
            .is_some_and(|v| !v.trim().eq_ignore_ascii_case("identity"));
        let is_event_stream = !encoded
            && head.get("content-type").is_some_and(|v| {
                v.trim()
                    .to_ascii_lowercase()
                    .starts_with("text/event-stream")
            });

        // The answer's head. The page is of this server's origin, so nothing that would speak for
        // that origin passes (cookies, cross-origin grants, page policies), and a target's page
        // opened from this address runs without this origin's rights.
        let mut out = format!(
            "HTTP/1.1 {} {}\r\n",
            head.status,
            if head.reason.is_empty() {
                "-"
            } else {
                &head.reason
            }
        );
        for (lower, name, value) in &head.headers {
            if rule::resp_header_passes(resp_header_kind(lower)) {
                out.push_str(&format!("{name}: {value}\r\n"));
            }
        }
        out.push_str(
            "Cache-Control: no-store\r\n\
             X-Content-Type-Options: nosniff\r\n\
             Content-Security-Policy: sandbox\r\n\
             X-Accel-Buffering: no\r\n\
             Connection: close\r\n",
        );
        if framing == Framing::Empty || framing == Framing::Length(0) {
            out.push_str("Content-Length: 0\r\n\r\n");
            if let Err(e) = self.write_page(stream, out.as_bytes(), started) {
                self.page_ended(stream, target, e);
            }
            return;
        }
        out.push_str("Transfer-Encoding: chunked\r\n\r\n");
        if let Err(e) = self.write_page(stream, out.as_bytes(), started) {
            return self.page_ended(stream, target, e);
        }

        let mut sent: u64 = 0;
        let mut at_line_start = true;
        // the target's silence is counted from here: its last bytes, or the end of the last write
        // of its bytes to the page (time spent waiting on the page is not the target's silence)
        let mut last_from_target = Instant::now();
        let mut last_to_page = Instant::now();
        // `Ok(true)`: the body ended as its framing says; `Ok(false)`: it was cut
        let ended: Result<bool, String> = loop {
            match body_step(&mut framing, &mut buf) {
                Step::End => break Ok(true),
                Step::Bad(why) => break Err(why.to_string()),
                Step::Data(n) => {
                    let take = rule::resp_take(sent, n as u64, self.limits.max_response) as usize;
                    if take > 0 {
                        if let Err(e) = self.write_page(stream, &chunk(&buf[..take]), started) {
                            return self.page_ended(stream, target, e);
                        }
                        at_line_start = buf[take - 1] == b'\n';
                        last_to_page = Instant::now();
                    }
                    if rule::resp_cut(sent, n as u64, self.limits.max_response) {
                        break Err(format!(
                            "the response exceeds the limit of {} bytes (--mcp-proxy-max-mb)",
                            self.limits.max_response
                        ));
                    }
                    sent += take as u64;
                    buf.drain(..n);
                    last_from_target = Instant::now();
                }
                Step::More => {
                    let idle = ms_since(last_from_target);
                    let total = ms_since(started);
                    if rule::expired(idle, self.limits.idle_ms) {
                        break Err(format!("the target was silent for {} s", idle / 1000));
                    }
                    if rule::expired(total, self.limits.total_ms) {
                        break Err(format!("the exchange lasted {} s", total / 1000));
                    }
                    if rule::keepalive_due(
                        is_event_stream,
                        at_line_start,
                        ms_since(last_to_page),
                        self.limits.keepalive_ms,
                    ) {
                        if let Err(e) = self.write_page(stream, &chunk(b": keep-alive\n"), started)
                        {
                            return self.page_ended(stream, target, e);
                        }
                        last_to_page = Instant::now();
                    }
                    let tick = rule::keepalive_wait(
                        is_event_stream,
                        at_line_start,
                        ms_since(last_to_page),
                        self.limits.keepalive_ms,
                    );
                    up.set_read_timeout(rule::read_wait(
                        rule::time_left(idle, self.limits.idle_ms),
                        rule::time_left(total, self.limits.total_ms),
                        tick,
                    ));
                    let mut chunk = [0u8; 16 * 1024];
                    match up.read(&mut chunk) {
                        Ok(0) => break Ok(framing == Framing::ToClose),
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            last_from_target = Instant::now();
                        }
                        Err(e) if timed_out(&e) || e.kind() == io::ErrorKind::Interrupted => {}
                        Err(e) => break Err(format!("reading from the target failed: {e}")),
                    }
                }
            }
        };
        match ended {
            Ok(true) => {
                if let Err(e) = self.write_page(stream, b"0\r\n\r\n", started) {
                    self.page_ended(stream, target, e);
                }
            }
            // A body that did not end as its framing says is not ended cleanly towards the page
            // either: the page sees a cut answer, not a short one that looks whole.
            Ok(false) => eprintln!(
                "[mcp-proxy] {}: the target closed the connection in the middle of its answer",
                target.shown()
            ),
            Err(why) => eprintln!("[mcp-proxy] {}: answer cut: {why}", target.shown()),
        }
    }

    /// Writes `data` to the page. One write call waits no longer than what is left of the
    /// exchange's limits: the idle limit, counted from the page's last progress, and the total
    /// limit. When nothing is left, the write is given up.
    fn write_page(&self, stream: &mut Conn, data: &[u8], started: Instant) -> Result<(), PageEnd> {
        let mut off = 0;
        let mut progress = Instant::now();
        while off < data.len() {
            let (idle, total) = (ms_since(progress), ms_since(started));
            let idle_left = rule::time_left(idle, self.limits.idle_ms);
            let total_left = rule::time_left(total, self.limits.total_ms);
            if rule::write_gives_up(idle_left, total_left) {
                return Err(PageEnd::Stalled(
                    if rule::expired(total, self.limits.total_ms) {
                        format!("the exchange lasted {} s", total / 1000)
                    } else {
                        format!("the page took nothing for {} s", idle / 1000)
                    },
                ));
            }
            let wait = rule::write_wait(idle_left, total_left);
            let _ = stream.set_write_timeout(Some(Duration::from_millis(wait.max(1))));
            // a write that timed out is tried again with the same bytes (TLS asks for that)
            let end = data.len().min(off + PAGE_WRITE_PIECE);
            match stream.write(&data[off..end]) {
                Ok(0) => return Err(PageEnd::Gone),
                Ok(n) => {
                    off += n;
                    progress = Instant::now();
                }
                Err(e) if timed_out(&e) || e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => return Err(PageEnd::Gone),
            }
        }
        stream.flush().map_err(|_| PageEnd::Gone)
    }

    /// A write to the page did not complete. A page that stopped reading has its connection
    /// ended at once (no goodbye that could wait on it) and is counted; the connection to the
    /// target closes when the caller lets go of it.
    fn page_ended(&self, stream: &mut Conn, target: &Target, end: PageEnd) {
        let PageEnd::Stalled(why) = end else {
            return; // the page is gone
        };
        stream.abandon();
        self.note(
            &self.stalled,
            format!("stalled {}: {why}; the answer was cut", target.shown()),
            format!(
                "\"result\":\"stalled\",\"url\":{},\"reason\":{}",
                jq(&target.shown()),
                jq(&why)
            ),
        );
    }
}

/// How a write to the page ended without completing.
enum PageEnd {
    /// The page closed its connection, or writing to it failed.
    Gone,
    /// The page took nothing for the idle limit, or the exchange ran out of time.
    Stalled(String),
}

fn ms_since(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

fn method_name(m: Method) -> &'static str {
    match m {
        Method::PxGet => "GET",
        Method::PxPost => "POST",
        Method::PxDelete => "DELETE",
    }
}

fn entry_text(e: &Entry) -> String {
    let host = match &e.pat {
        Pattern::Any => "*".to_string(),
        Pattern::Suffix(s) => format!("*{s}"),
        Pattern::Name(n) => n.clone(),
        Pattern::Net(ip, bits) => {
            let full = if ip.is_ipv4() { 32 } else { 128 };
            match (*bits == full, ip, e.port) {
                (true, IpAddr::V6(_), Some(_)) => format!("[{ip}]"),
                (true, _, _) => ip.to_string(),
                (false, _, _) => format!("{ip}/{bits}"),
            }
        }
    };
    match e.port {
        Some(p) => format!("{host}:{p}"),
        None => host,
    }
}

/// A URL as the request gave it, for logs and messages: cut at its query and fragment (they may
/// hold a credential), without a user part, and shortened. Any text is accepted, also one that is
/// not a valid URL.
fn shown_url(url: &str) -> String {
    without_user(url.split(['?', '#']).next().unwrap_or(""))
        .chars()
        .filter(|c| !c.is_control())
        .take(200)
        .collect()
}

/// `url` without the user part of its authority (`user:password@`): everything up to the last
/// `@` of the authority is left out. The authority starts after the scheme and the slashes that
/// follow it, or at the start of the text when there is no `scheme:/` (so a malformed URL loses
/// its user part too), and ends at the first `/`, `\`, `?` or `#`.
fn without_user(url: &str) -> String {
    let after_scheme = match url.find(':') {
        Some(i) if scheme_ok(&url[..i]) => i + 1,
        _ => 0,
    };
    let rest = &url[after_scheme..];
    let slashes = rest.len() - rest.trim_start_matches(['/', '\\']).len();
    // `scheme:` without slashes may be no scheme at all (`user:password@host`)
    let start = if slashes > 0 || after_scheme == 0 {
        after_scheme + slashes
    } else {
        0
    };
    let len = url[start..]
        .find(['/', '\\', '?', '#'])
        .unwrap_or(url.len() - start);
    match url[start..start + len].rfind('@') {
        Some(at) => format!("{}{}", &url[..start], &url[start + at + 1..]),
        None => url.to_string(),
    }
}

/// One transfer chunk of the answer to the page.
fn chunk(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 16);
    out.extend_from_slice(format!("{:x}\r\n", data.len()).as_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(b"\r\n");
    out
}

/// An answer of the proxy itself. It carries no cross-origin grant: only this server's own page
/// reads the proxy.
fn write_proxy_error(stream: &mut Conn, status: u16, kind: &str, message: &str) {
    let reason = match status {
        400 => "Bad Request",
        403 => "Forbidden",
        405 => "Method Not Allowed",
        413 => "Content Too Large",
        502 => "Bad Gateway",
        _ => "Error",
    };
    let body = format!(
        "{{\"error\":{{\"message\":{},\"type\":{},\"code\":{status}}}}}",
        jq(message),
        jq(kind)
    );
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

#[cfg(test)]
#[path = "proxy_tests.rs"]
mod tests;
