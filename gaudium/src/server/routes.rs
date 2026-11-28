//! The route table: every route the router in `handle_request` serves, with its kind, and the API
//! key gate that answers or refuses a request by that kind.
//!
//! The decision is `route_admit` in `verified/route_auth.bend`; this file only maps a request's
//! method and path to the facts that decision takes (string matching). Closed by default: a
//! request the table does not list needs the key, unless it is a read outside the API's paths (a
//! Web UI file). A route added to the router without a line here fails `cargo test`
//! (`every_router_arm_is_in_the_route_table` below reads the router's `match`).

use crate::verified::route_admit;
pub use crate::verified::RouteKind;

/// How a route's path is matched.
#[derive(Debug, Clone, Copy)]
pub enum RoutePath {
    /// The path is exactly this.
    Exact(&'static str),
    /// The path starts with this.
    Prefix(&'static str),
}

impl RoutePath {
    fn matches(self, path: &str) -> bool {
        match self {
            RoutePath::Exact(p) => path == p,
            RoutePath::Prefix(p) => path.starts_with(p),
        }
    }
}

/// One route of the router.
#[derive(Debug, Clone, Copy)]
pub struct Route {
    /// The method, or `None` for a route that takes every method.
    pub method: Option<&'static str>,
    pub path: RoutePath,
    pub kind: RouteKind,
}

const fn route(method: &'static str, path: &'static str, kind: RouteKind) -> Route {
    Route {
        method: Some(method),
        path: RoutePath::Exact(path),
        kind,
    }
}

use RouteKind::{
    RouteAdmin as Admin, RouteInference as Inference, RouteLiveness, RouteStatus as Status,
};

/// Every route `handle_request` serves, apart from its fallbacks (Web UI files for `GET` and
/// `HEAD`, 404 for the rest).
pub const ROUTES: &[Route] = &[
    Route {
        method: None,
        path: RoutePath::Exact("/cors-proxy"),
        kind: Admin,
    },
    route("GET", "/gaudium/proxy", Status),
    route("GET", "/health", RouteLiveness),
    route("GET", "/v1/health", Status),
    route("GET", "/props", Status),
    route("GET", "/v1/models", Status),
    route("GET", "/models", Status),
    Route {
        method: Some("GET"),
        path: RoutePath::Prefix("/v1/models/"),
        kind: Status,
    },
    route("GET", "/models/sse", Status),
    route("POST", "/models/load", Admin),
    route("POST", "/models/unload", Admin),
    route("GET", "/gaudium/ui", Status),
    route("GET", "/gaudium/a2ui", Status),
    route("GET", "/gaudium/engine", Status),
    route("POST", "/gaudium/engine", Admin),
    route("GET", "/gaudium/live", Status),
    route("GET", "/gaudium/events", Status),
    route("GET", "/slots", Status),
    route("GET", "/gaudium/requests", Status),
    route("POST", "/v1/streams/lookup", Status),
    route("POST", "/tokenize", Inference),
    route("POST", "/detokenize", Inference),
    route("POST", "/gaudium/count_tokens", Inference),
    route("POST", "/v1/chat/completions/control", Admin),
    route("POST", "/v1/chat/completions", Inference),
    route("POST", "/chat/completions", Inference),
    route("POST", "/v1/messages", Inference),
    route("POST", "/v1/messages/count_tokens", Inference),
    // the Responses API: a response is made, read, cancelled and deleted by its own client
    route("POST", "/v1/responses", Inference),
    Route {
        method: Some("GET"),
        path: RoutePath::Prefix("/v1/responses/"),
        kind: Inference,
    },
    Route {
        method: Some("DELETE"),
        path: RoutePath::Prefix("/v1/responses/"),
        kind: Inference,
    },
    Route {
        method: Some("POST"),
        path: RoutePath::Prefix("/v1/responses/"),
        kind: Inference,
    },
    route("POST", "/apply-template", Inference),
    route("POST", "/completion", Inference),
    route("POST", "/completions", Inference),
    route("POST", "/v1/completions", Inference),
    // the completion paths answer GET with a 405; still behind the key like their POST
    route("GET", "/completion", Inference),
    route("GET", "/completions", Inference),
    route("GET", "/v1/completions", Inference),
    // the Files API: stored request content, owned by the key that uploaded it
    route("POST", "/v1/files", Inference),
    route("GET", "/v1/files", Inference),
    Route {
        method: Some("GET"),
        path: RoutePath::Prefix("/v1/files/"),
        kind: Inference,
    },
    Route {
        method: Some("DELETE"),
        path: RoutePath::Prefix("/v1/files/"),
        kind: Inference,
    },
];

/// Path prefixes the API owns: a request under one that the table does not list needs the key.
pub const API_PREFIXES: &[&str] = &["/v1/", "/gaudium/"];

/// The kind the table gives `method` and `path` (`RouteUnlisted` when it lists none).
pub fn listed_kind(method: &str, path: &str) -> RouteKind {
    ROUTES
        .iter()
        .find(|r| r.method.is_none_or(|m| m == method) && r.path.matches(path))
        .map_or(RouteKind::RouteUnlisted, |r| r.kind)
}

/// The path is one the API owns: under one of its prefixes, or one the table lists (under any
/// method).
pub fn api_path(path: &str) -> bool {
    API_PREFIXES.iter().any(|p| path.starts_with(p)) || ROUTES.iter().any(|r| r.path.matches(path))
}

/// The request is answered (`false`: it is refused with 401). `key_set`: the server was started
/// with an API key; `key_ok`: the request carries it (true when none is configured).
pub fn admits(method: &str, path: &str, key_set: bool, key_ok: bool) -> bool {
    route_admit(
        listed_kind(method, path),
        api_path(path),
        matches!(method, "GET" | "HEAD"),
        key_set,
        key_ok,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verified::{route_kind_of, route_kind_public};

    /// The kind a request reaches, as the gate sees it.
    fn kind(method: &str, path: &str) -> RouteKind {
        route_kind_of(
            listed_kind(method, path),
            api_path(path),
            matches!(method, "GET" | "HEAD"),
        )
    }

    /// The router's `match` in `handle_request`: each arm's method (`None` for `_`) and path
    /// pattern, read from the source. Every line at the arms' indentation that is not a comment
    /// or a closing bracket starts an arm, and an arm whose pattern this reader does not
    /// understand fails the test (a bound method such as `(m, "/x") if m == "GET"` included), so
    /// no route can be added in a form the table check would skip.
    fn router_arms() -> Vec<(Option<String>, RoutePathPat)> {
        let src = include_str!("mod.rs");
        let start = src
            .find("fn handle_request(")
            .expect("handle_request is in server/mod.rs");
        let body = &src[start..];
        let gate = body
            .find("routes::admits(")
            .expect("handle_request calls the route gate");
        let m = body
            .find("match (req.method.as_str(), req.path.as_str()) {")
            .expect("handle_request dispatches on (method, path)");
        assert!(gate < m, "the route gate runs before the router");
        // nothing between the gate's 401 block and the router: no route can answer before it
        let after_gate = &body[gate..m];
        let close = after_gate
            .find("\n        return;\n    }\n")
            .expect("the gate's refusal returns");
        let between = &after_gate[close + "\n        return;\n    }\n".len()..];
        assert!(
            between.trim().is_empty(),
            "code between the route gate and the router: {between:?}"
        );
        let body = &body[m..];
        // the match ends at the first line that closes it at the function's indentation
        let end = body.find("\n    }\n").expect("the router's match closes");
        let body = &body[..end];
        let mut heads = Vec::new();
        let mut open: Option<String> = None;
        for line in body.lines().skip(1) {
            let at_arm = line.starts_with("        ") && !line[8..].starts_with(' ');
            let t = line.trim();
            if let Some(h) = open.as_mut() {
                // an arm head that spans lines (`|` alternatives) until its `=>`
                h.push(' ');
                h.push_str(t);
                if t.contains("=>") {
                    heads.push(open.take().unwrap());
                }
                continue;
            }
            if !at_arm
                || t.is_empty()
                || t.starts_with("//")
                || t.starts_with('}')
                || t.starts_with(')')
            {
                continue;
            }
            if t.contains("=>") {
                heads.push(t.to_string());
            } else {
                open = Some(t.to_string());
            }
        }
        assert!(open.is_none(), "a router arm head never reached `=>`");
        let mut arms = Vec::new();
        let n = heads.len();
        for (i, head) in heads.iter().enumerate() {
            let head = head.split("=>").next().unwrap().trim();
            if head == "_" {
                // the 404 for everything else: only as the last arm
                assert_eq!(i + 1, n, "a catch-all router arm before the last one");
                continue;
            }
            for alt in head.split(" | ") {
                let alt = alt.trim().strip_prefix("| ").unwrap_or(alt.trim());
                let (pat, guard) = match alt.split_once(" if ") {
                    Some((p, g)) => (p.trim(), Some(g.trim())),
                    None => (alt, None),
                };
                let inner = pat
                    .strip_prefix('(')
                    .and_then(|p| p.strip_suffix(')'))
                    .unwrap_or_else(|| panic!("router arm pattern not understood: {alt}"));
                let (method, path) = inner
                    .split_once(',')
                    .unwrap_or_else(|| panic!("router arm pattern not understood: {alt}"));
                let method = match method.trim() {
                    "_" => None,
                    m if m.len() >= 2 && m.starts_with('"') && m.ends_with('"') => {
                        Some(m.trim_matches('"').to_string())
                    }
                    m => panic!("router arm method not understood (bind none; match a literal or `_`): {m} in {alt}"),
                };
                let path = path.trim();
                let pat = if path.starts_with('"') {
                    assert!(guard.is_none(), "a guard on a literal path: {alt}");
                    RoutePathPat::Exact(path.trim_matches('"').to_string())
                } else if path == "_" {
                    assert!(guard.is_none(), "a guard on a fallback arm: {alt}");
                    RoutePathPat::Fallback
                } else {
                    // a bound path with a `starts_with` guard
                    let g = guard.unwrap_or_else(|| panic!("a bound path without a guard: {alt}"));
                    let pre = g
                        .strip_prefix(&format!("{path}.starts_with(\""))
                        .and_then(|g| g.strip_suffix("\")"))
                        .unwrap_or_else(|| panic!("router guard not understood: {alt}"));
                    RoutePathPat::Prefix(pre.to_string())
                };
                arms.push((method, pat));
            }
        }
        arms
    }

    #[derive(Debug, PartialEq)]
    enum RoutePathPat {
        Exact(String),
        Prefix(String),
        Fallback,
    }

    #[test]
    fn every_router_arm_is_in_the_route_table() {
        let arms = router_arms();
        assert!(arms.len() > 20, "too few router arms found: {arms:?}");
        let mut fallbacks = Vec::new();
        for (method, pat) in &arms {
            let found = ROUTES.iter().any(|r| {
                let m_ok = r.method.map(str::to_string) == *method;
                let p_ok = match (pat, r.path) {
                    (RoutePathPat::Exact(a), RoutePath::Exact(b)) => a == b,
                    (RoutePathPat::Prefix(a), RoutePath::Prefix(b)) => a == b,
                    _ => false,
                };
                m_ok && p_ok
            });
            if *pat == RoutePathPat::Fallback {
                fallbacks.push(method.clone());
                continue;
            }
            assert!(
                found,
                "router arm {method:?} {pat:?} has no line in ROUTES (server/routes.rs): \
                 give it a kind"
            );
        }
        // the fallbacks are the Web UI files and the 404
        assert_eq!(
            fallbacks,
            vec![Some("GET".to_string()), Some("HEAD".to_string())],
            "router fallbacks changed"
        );
        // and the table lists nothing the router does not serve
        for r in ROUTES {
            let served = arms.iter().any(|(m, p)| {
                *m == r.method.map(str::to_string)
                    && match (p, r.path) {
                        (RoutePathPat::Exact(a), RoutePath::Exact(b)) => a == b,
                        (RoutePathPat::Prefix(a), RoutePath::Prefix(b)) => a == b,
                        _ => false,
                    }
            });
            assert!(
                served,
                "ROUTES lists {r:?}, which the router does not serve"
            );
        }
    }

    #[test]
    fn the_table_never_lists_a_route_as_unlisted_or_ui_asset() {
        for r in ROUTES {
            assert!(
                !matches!(r.kind, RouteKind::RouteUnlisted | RouteKind::RouteUiAsset),
                "{r:?}"
            );
        }
    }

    #[test]
    fn only_the_liveness_probe_is_a_public_route() {
        let public: Vec<_> = ROUTES
            .iter()
            .filter(|r| route_kind_public(r.kind))
            .map(|r| (r.method, format!("{:?}", r.path)))
            .collect();
        assert_eq!(
            public,
            vec![(Some("GET"), "Exact(\"/health\")".to_string())]
        );
    }

    #[test]
    fn inference_and_admin_routes_keep_their_kinds() {
        for (m, p) in [
            ("POST", "/v1/chat/completions"),
            ("POST", "/chat/completions"),
            ("POST", "/v1/messages"),
            ("POST", "/v1/messages/count_tokens"),
            ("POST", "/tokenize"),
            ("POST", "/detokenize"),
            ("POST", "/gaudium/count_tokens"),
            ("POST", "/apply-template"),
            ("POST", "/completion"),
            ("POST", "/completions"),
            ("POST", "/v1/completions"),
            ("POST", "/v1/responses"),
            ("GET", "/v1/responses/resp_1"),
            ("GET", "/v1/responses/resp_1/input_items"),
            ("DELETE", "/v1/responses/resp_1"),
            ("POST", "/v1/responses/resp_1/cancel"),
            ("POST", "/v1/files"),
            ("GET", "/v1/files"),
            ("GET", "/v1/files/file-x"),
            ("GET", "/v1/files/file-x/content"),
            ("DELETE", "/v1/files/file-x"),
        ] {
            assert_eq!(kind(m, p), RouteKind::RouteInference, "{m} {p}");
            assert!(!admits(m, p, true, false), "{m} {p} served without the key");
        }
        for (m, p) in [
            ("POST", "/models/load"),
            ("POST", "/models/unload"),
            ("POST", "/gaudium/engine"),
            ("POST", "/v1/chat/completions/control"),
            ("POST", "/cors-proxy"),
            ("GET", "/cors-proxy"),
            ("DELETE", "/cors-proxy"),
        ] {
            assert_eq!(kind(m, p), RouteKind::RouteAdmin, "{m} {p}");
        }
    }

    #[test]
    fn web_ui_files_and_health_are_public_and_the_api_is_not() {
        for (m, p) in [
            ("GET", "/"),
            ("GET", "/ui/"),
            ("HEAD", "/"),
            ("GET", "/index.html"),
            ("GET", "/js/lib/api.js"),
            ("GET", "/sw.js"),
            ("GET", "/favicon.ico"),
        ] {
            assert_eq!(kind(m, p), RouteKind::RouteUiAsset, "{m} {p}");
            assert!(admits(m, p, true, false), "{m} {p}");
        }
        assert_eq!(kind("GET", "/health"), RouteKind::RouteLiveness);
        assert!(admits("GET", "/health", true, false));
        // unlisted paths the API owns, and unlisted writes anywhere, are closed
        for (m, p) in [
            ("GET", "/v1/health"),
            ("GET", "/v1/unknown"),
            ("HEAD", "/v1/models"),
            ("GET", "/gaudium/unknown"),
            ("GET", "/tokenize"),
            ("GET", "/chat/completions"),
            ("POST", "/v1/models/x"),
            ("POST", "/unknown"),
            ("PUT", "/index.html"),
            ("POST", "/health"),
        ] {
            assert!(!admits(m, p, true, false), "{m} {p} served without the key");
            assert!(admits(m, p, true, true), "{m} {p} refused with the key");
            assert!(
                admits(m, p, false, true),
                "{m} {p} refused with no key configured"
            );
        }
        // every listed route but the liveness probe is closed without the key
        for r in ROUTES {
            let m = r.method.unwrap_or("POST");
            let p = match r.path {
                RoutePath::Exact(p) => p.to_string(),
                RoutePath::Prefix(p) => format!("{p}x"),
            };
            let open = admits(m, &p, true, false);
            assert_eq!(open, p == "/health", "{m} {p}");
            assert!(admits(m, &p, true, true), "{m} {p}");
            assert!(admits(m, &p, false, true), "{m} {p}");
        }
    }
}
