//! Rust Shell wrappers for the Host check of `verified/server_names.bend`.
//!
//! Whether a request names this server (every host it names is an address or one of the
//! server's own names, compared without regard to ASCII case), and whether the proxy's site
//! verdict counts, delegate to the verified core. What this module adds is the conversion of
//! names to the byte lists the core compares; reading hosts out of request headers is done by
//! the caller.

use crate::verified::{hn_request_ours, hn_site_ok, HnHost, List};

/// One host a request names, as read from a header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamedHost {
    /// An IPv4 or IPv6 address literal.
    Addr,
    /// A DNS name, its bytes as the request sent them (port and root dot taken off).
    Name(Vec<u8>),
    /// Neither (a missing or malformed value).
    Bad,
}

/// The server's own names, in the form the verified core compares.
#[derive(Debug, Clone)]
pub struct OwnNames(List<List<u64>>);

fn bytes(b: &[u8]) -> List<u64> {
    b.iter().rev().fold(List::Nil, |tail, &x| {
        List::Con(u64::from(x), Box::new(tail))
    })
}

impl OwnNames {
    pub fn new<S: AsRef<str>>(names: &[S]) -> OwnNames {
        OwnNames(names.iter().rev().fold(List::Nil, |tail, n| {
            List::Con(bytes(n.as_ref().as_bytes()), Box::new(tail))
        }))
    }
}

fn host(h: &NamedHost) -> HnHost {
    match h {
        NamedHost::Addr => HnHost::HnAddr,
        NamedHost::Name(n) => HnHost::HnName(bytes(n)),
        NamedHost::Bad => HnHost::HnBad,
    }
}

/// The request names this server: its Host (`host`) and every host a front end names for it
/// (`forwarded`) is an address or one of `names`.
pub fn request_ours(host_header: &NamedHost, forwarded: &[NamedHost], names: &OwnNames) -> bool {
    let fwd = forwarded
        .iter()
        .rev()
        .fold(List::Nil, |tail, h| List::Con(host(h), Box::new(tail)));
    hn_request_ours(host(host_header), fwd, names.0.clone())
}

/// The proxy's site verdict, counted only for a request that names this server.
pub fn site_ok(hosts_ours: bool, site_ok: bool) -> bool {
    hn_site_ok(hosts_ours, site_ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(s: &str) -> NamedHost {
        NamedHost::Name(s.as_bytes().to_vec())
    }

    #[test]
    fn names_are_compared_whole_and_without_case() {
        let own = OwnNames::new(&["localhost", "chat.example.com"]);
        assert!(request_ours(&name("localhost"), &[], &own));
        assert!(request_ours(&name("LocalHost"), &[], &own));
        assert!(request_ours(&name("CHAT.example.COM"), &[], &own));
        assert!(!request_ours(&name("chat.example.co"), &[], &own));
        assert!(!request_ours(
            &name("chat.example.com.evil.example"),
            &[],
            &own
        ));
        assert!(!request_ours(&name("evil.example"), &[], &own));
        assert!(!request_ours(&name(""), &[], &own));
        // only ASCII letters fold: '[' (91) is not '{' (123), '@' (64) is not '`' (96)
        let odd = OwnNames::new(&["a[b", "x@y"]);
        assert!(!request_ours(&name("a{b"), &[], &odd));
        assert!(!request_ours(&name("x`y"), &[], &odd));
        assert!(request_ours(&name("A[B"), &[], &odd));
    }

    #[test]
    fn addresses_pass_and_unreadable_hosts_do_not() {
        let own = OwnNames::new::<&str>(&[]);
        assert!(request_ours(&NamedHost::Addr, &[], &own));
        assert!(!request_ours(&NamedHost::Bad, &[], &own));
        assert!(!request_ours(&name("localhost"), &[], &own));
    }

    #[test]
    fn every_forwarded_host_must_be_ours() {
        let own = OwnNames::new(&["localhost", "chat.example.com"]);
        let fwd = [name("chat.example.com"), NamedHost::Addr];
        assert!(request_ours(&name("localhost"), &fwd, &own));
        let fwd = [name("chat.example.com"), name("evil.example")];
        assert!(!request_ours(&name("localhost"), &fwd, &own));
        assert!(!request_ours(&name("localhost"), &[NamedHost::Bad], &own));
        assert!(!request_ours(
            &name("evil.example"),
            &[name("localhost")],
            &own
        ));
    }

    #[test]
    fn the_site_verdict_counts_only_for_our_hosts() {
        assert!(site_ok(true, true));
        assert!(!site_ok(true, false));
        assert!(!site_ok(false, true));
        assert!(!site_ok(false, false));
    }
}
