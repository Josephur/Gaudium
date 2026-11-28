//! Rust Shell wrappers for the Host check of `verified/server_names.bend`.
//!
//! Whether a request names this server (every host it names is an address or one of the
//! server's own names, compared without regard to ASCII case), and whether the proxy's site
//! verdict counts, delegate to the verified core. What this module adds is the conversion of
//! names to the byte lists the core compares; reading hosts out of request headers is done by
//! the caller.
//!
//! The lists are built only for a request inside the bounds below (`hn_hosts_fit`): a request
//! can name any number of hosts, and a list of thousands would exhaust the stack of the
//! connection thread when the verified core walks or drops it.

use crate::verified::{hn_hosts_fit, hn_hosts_ours, hn_request_ours, hn_site_ok, HnHost, List};

/// Most hosts the front-end headers of one request may name (one or two per front end).
pub const MAX_FORWARDED_HOSTS: u64 = 16;
/// Most bytes the names of one request may hold between them, its Host included.
pub const MAX_NAME_BYTES: u64 = 4096;

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

fn name_bytes(h: &NamedHost) -> u64 {
    match h {
        NamedHost::Name(n) => n.len() as u64,
        NamedHost::Addr | NamedHost::Bad => 0,
    }
}

/// The hosts the request names are few enough to be read (`MAX_FORWARDED_HOSTS`,
/// `MAX_NAME_BYTES`).
pub fn hosts_fit(host_header: &NamedHost, forwarded: &[NamedHost]) -> bool {
    let bytes = forwarded.iter().fold(name_bytes(host_header), |sum, h| {
        sum.saturating_add(name_bytes(h))
    });
    hn_hosts_fit(
        forwarded.len() as u64,
        bytes,
        MAX_FORWARDED_HOSTS,
        MAX_NAME_BYTES,
    )
}

/// The request names this server: its hosts fit the bounds, and its Host (`host`) and every
/// host a front end names for it (`forwarded`) is an address or one of `names`.
pub fn request_ours(host_header: &NamedHost, forwarded: &[NamedHost], names: &OwnNames) -> bool {
    let fits = hosts_fit(host_header, forwarded);
    // the core refuses a request that does not fit whatever it names; its hosts are not read
    let (h, fwd) = match fits {
        true => (
            host(host_header),
            forwarded
                .iter()
                .rev()
                .fold(List::Nil, |tail, h| List::Con(host(h), Box::new(tail))),
        ),
        false => (HnHost::HnBad, List::Nil),
    };
    hn_hosts_ours(fits, hn_request_ours(h, fwd, names.0.clone()))
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
    fn a_request_naming_too_many_hosts_is_refused_unread() {
        let own = OwnNames::new(&["localhost"]);
        let max = MAX_FORWARDED_HOSTS as usize;
        let fwd = vec![NamedHost::Addr; max];
        assert!(request_ours(&name("localhost"), &fwd, &own));
        let fwd = vec![NamedHost::Addr; max + 1];
        assert!(!hosts_fit(&name("localhost"), &fwd));
        assert!(!request_ours(&name("localhost"), &fwd, &own));
        // far past the bound: refused without a list being built (no stack overflow)
        let fwd = vec![name("localhost"); 1_000_000];
        assert!(!request_ours(&name("localhost"), &fwd, &own));
        // the bytes of the names are bounded too, the Host's included
        let long = "a".repeat(MAX_NAME_BYTES as usize);
        let own = OwnNames::new(&[long.as_str()]);
        assert!(request_ours(&name(&long), &[], &own));
        assert!(!request_ours(&name(&long), &[name("a")], &own));
        let longer = "a".repeat(MAX_NAME_BYTES as usize + 1);
        assert!(!request_ours(
            &name(&longer),
            &[],
            &OwnNames::new(&[longer.as_str()])
        ));
    }

    #[test]
    fn the_bounds_are_today_16_hosts_and_4096_bytes() {
        assert_eq!(MAX_FORWARDED_HOSTS, 16);
        assert_eq!(MAX_NAME_BYTES, 4096);
    }

    #[test]
    fn the_site_verdict_counts_only_for_our_hosts() {
        assert!(site_ok(true, true));
        assert!(!site_ok(true, false));
        assert!(!site_ok(false, true));
        assert!(!site_ok(false, false));
    }
}
