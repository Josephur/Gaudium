//! Rust Shell wrappers for the MCP proxy decisions of `verified/mcp_proxy.bend`.
//!
//! Every allow / deny decision of the proxy, its redirect budget, its header policy and its
//! size and time caps delegate to the verified core. What this module adds is the conversion
//! from Rust's address types to the octets the core classifies.

use std::net::IpAddr;

use crate::verified::{
    px_addr_check, px_addr_join, px_addr_passes, px_body_fits, px_cap_bytes, px_cap_ok, px_expired,
    px_gate, px_keepalive_due, px_keepalive_wait, px_match_resolves, px_net_bits_at,
    px_net_bits_ok, px_net_byte_match, px_port_ok, px_read_wait, px_redirect_follows,
    px_redirect_keeps_body, px_redirect_kind, px_redirect_left, px_redirect_may,
    px_redirect_method, px_req_header_goes, px_resp_cut, px_resp_header_passes, px_resp_take,
    px_site_ok, px_start, px_start_refuses, px_start_serves, px_time_left, px_url_check, px_url_ok,
    px_v4_class, px_v6_class,
};
pub use crate::verified::{
    PxAddr as AddrVerdict, PxAddrClass as AddrClass, PxGate as Gate, PxMatch as ListMatch,
    PxMethod as Method, PxRedirect as Redirect, PxReqHeader as ReqHeader,
    PxRespHeader as RespHeader, PxScheme as Scheme, PxSite as Site, PxStart as Start,
    PxUrl as UrlVerdict,
};

/// What the proxy flags mean. `asked`: the proxy flag was given; `listed`: a target entry was
/// given; `open`: an entry allows any target; `keyed`: the server has an API key.
pub fn start(asked: bool, listed: bool, open: bool, keyed: bool) -> Start {
    px_start(asked, listed, open, keyed)
}

pub fn start_serves(s: Start) -> bool {
    px_start_serves(s)
}

pub fn start_refused(s: Start) -> bool {
    px_start_refuses(s)
}

/// May a request with this fetch-site statement and origin drive the proxy.
pub fn site_ok(site: Site, has_origin: bool, origin_is_host: bool) -> bool {
    px_site_ok(site, has_origin, origin_is_host)
}

/// What happens to one request to the proxy route.
pub fn gate(key_ok: bool, enabled: bool, site_ok: bool, method_ok: bool) -> Gate {
    px_gate(key_ok, enabled, site_ok, method_ok)
}

/// Whether a parsed target URL is one the proxy fetches.
pub fn url_check(
    scheme: Scheme,
    has_user: bool,
    host_ok: bool,
    port: u64,
    tls_ready: bool,
) -> UrlVerdict {
    px_url_check(scheme, has_user, host_ok, px_port_ok(port), tls_ready)
}

pub fn url_ok(v: UrlVerdict) -> bool {
    px_url_ok(v)
}

/// The class of an address.
pub fn addr_class(ip: IpAddr) -> AddrClass {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, c, d] = v4.octets().map(u64::from);
            px_v4_class(a, b, c, d)
        }
        IpAddr::V6(v6) => {
            let b = v6.octets().map(u64::from);
            px_v6_class(
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
                b[13], b[14], b[15],
            )
        }
    }
}

/// The target's name is looked up (a target no entry matches is refused before that).
pub fn match_resolves(m: ListMatch) -> bool {
    px_match_resolves(m)
}

/// A prefix length that fits the family of `net`.
pub fn net_bits_ok(net: IpAddr, bits: u64) -> bool {
    px_net_bits_ok(bits, if net.is_ipv4() { 32 } else { 128 })
}

/// `ip` lies in the range `net`/`bits` (never across address families).
pub fn net_contains(net: IpAddr, bits: u64, ip: IpAddr) -> bool {
    fn bytes_match(net: &[u8], ip: &[u8], bits: u64) -> bool {
        net.iter().zip(ip).enumerate().all(|(i, (n, a))| {
            px_net_byte_match(u64::from(*n), u64::from(*a), px_net_bits_at(bits, i as u64))
        })
    }
    match (net, ip) {
        (IpAddr::V4(n), IpAddr::V4(a)) => bytes_match(&n.octets(), &a.octets(), bits),
        (IpAddr::V6(n), IpAddr::V6(a)) => bytes_match(&n.octets(), &a.octets(), bits),
        _ => false,
    }
}

/// May one resolved address of a target be connected to.
pub fn addr_check(m: ListMatch, in_range: bool, class: AddrClass, listed: bool) -> AddrVerdict {
    px_addr_check(m, in_range, class, listed)
}

/// The verdict on a name from the verdicts on its addresses so far and the next one.
pub fn addr_join(acc: AddrVerdict, next: AddrVerdict) -> AddrVerdict {
    px_addr_join(acc, next)
}

pub fn addr_passes(v: AddrVerdict) -> bool {
    px_addr_passes(v)
}

/// The redirect a response asks for.
pub fn redirect_kind(status: u16, has_location: bool) -> Redirect {
    px_redirect_kind(u64::from(status), has_location)
}

/// The response is a redirect the proxy follows.
pub fn redirect_follows(k: Redirect) -> bool {
    px_redirect_follows(k)
}

/// With `left` redirects still allowed: may one more be followed.
pub fn redirect_may(left: u64) -> bool {
    px_redirect_may(left)
}

/// Redirects left after following one.
pub fn redirect_left(left: u64) -> u64 {
    px_redirect_left(left)
}

/// The method of the request that follows a redirect.
pub fn redirect_method(m: Method, k: Redirect) -> Method {
    px_redirect_method(m, k)
}

/// The request body goes along with the redirect.
pub fn redirect_keeps_body(m: Method, k: Redirect) -> bool {
    px_redirect_keeps_body(m, k)
}

/// A request header goes to the target.
pub fn req_header_goes(prefixed: bool, kind: ReqHeader, same_origin: bool) -> bool {
    px_req_header_goes(prefixed, kind, same_origin)
}

/// A response header of the target reaches the page.
pub fn resp_header_passes(kind: RespHeader) -> bool {
    px_resp_header_passes(kind)
}

/// A megabyte count as bytes.
pub fn cap_bytes(mb: u64) -> u64 {
    px_cap_bytes(mb)
}

/// A cap value the operator may set (`1..=most`).
pub fn cap_ok(value: u64, most: u64) -> bool {
    px_cap_ok(value, most)
}

pub fn body_fits(len: u64, cap: u64) -> bool {
    px_body_fits(len, cap)
}

/// Of `incoming` response bytes, with `sent` forwarded already: how many may go out.
pub fn resp_take(sent: u64, incoming: u64, cap: u64) -> u64 {
    px_resp_take(sent, incoming, cap)
}

/// The response is longer than the cap allows.
pub fn resp_cut(sent: u64, incoming: u64, cap: u64) -> bool {
    px_resp_cut(sent, incoming, cap)
}

pub fn time_left(elapsed: u64, limit: u64) -> u64 {
    px_time_left(elapsed, limit)
}

pub fn expired(elapsed: u64, limit: u64) -> bool {
    px_expired(elapsed, limit)
}

/// How long one read from the target may block.
pub fn read_wait(idle_left: u64, total_left: u64, tick: u64) -> u64 {
    px_read_wait(idle_left, total_left, tick)
}

/// A keep-alive comment line is due on an event stream.
pub fn keepalive_due(is_event_stream: bool, at_line_start: bool, quiet: u64, every: u64) -> bool {
    px_keepalive_due(is_event_stream, at_line_start, quiet, every)
}

/// How long to wait before looking again whether a keep-alive line is due.
pub fn keepalive_wait(is_event_stream: bool, at_line_start: bool, quiet: u64, every: u64) -> u64 {
    px_keepalive_wait(is_event_stream, at_line_start, quiet, every)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(s: &str) -> AddrClass {
        addr_class(s.parse().unwrap())
    }

    /// Today's address table (the laws fix the safety bounds; the full table is pinned here).
    #[test]
    fn address_classes_of_today() {
        use AddrClass::*;
        for (addr, want) in [
            ("93.184.216.34", PxAcPublic),
            ("8.8.8.8", PxAcPublic),
            ("172.15.255.255", PxAcPublic),
            ("172.32.0.1", PxAcPublic),
            ("100.63.255.255", PxAcPublic),
            ("100.128.0.1", PxAcPublic),
            ("223.255.255.255", PxAcPublic),
            ("127.0.0.1", PxAcLoopback),
            ("127.255.255.254", PxAcLoopback),
            ("10.0.0.5", PxAcPrivate),
            ("172.16.0.1", PxAcPrivate),
            ("172.31.255.255", PxAcPrivate),
            ("192.168.1.1", PxAcPrivate),
            ("100.64.0.1", PxAcPrivate),
            ("100.127.255.255", PxAcPrivate),
            ("192.0.0.8", PxAcPrivate),
            ("192.0.2.7", PxAcPrivate),
            ("198.18.0.1", PxAcPrivate),
            ("198.19.255.255", PxAcPrivate),
            ("198.51.100.9", PxAcPrivate),
            ("203.0.113.9", PxAcPrivate),
            ("169.254.169.254", PxAcLinkLocal),
            ("0.0.0.0", PxAcNever),
            ("0.1.2.3", PxAcNever),
            ("192.88.99.1", PxAcNever),
            ("224.0.0.1", PxAcNever),
            ("240.0.0.1", PxAcNever),
            ("255.255.255.255", PxAcNever),
            ("2606:4700:4700::1111", PxAcPublic),
            ("2a00:1450:4001:81b::200e", PxAcPublic),
            ("2001:4860:4860::8888", PxAcPublic),
            ("::1", PxAcLoopback),
            ("::", PxAcNever),
            ("::2", PxAcNever),
            ("::7f00:1", PxAcNever),
            ("::ffff:127.0.0.1", PxAcLoopback),
            ("::ffff:10.0.0.1", PxAcPrivate),
            ("::ffff:169.254.169.254", PxAcLinkLocal),
            ("::ffff:8.8.8.8", PxAcPublic),
            ("::ffff:0.0.0.0", PxAcNever),
            ("64:ff9b::7f00:1", PxAcLoopback),
            ("64:ff9b::a00:1", PxAcPrivate),
            ("64:ff9b::808:808", PxAcPublic),
            ("64:ff9b:1::1", PxAcPrivate),
            ("2002:7f00:1::1", PxAcLoopback),
            ("2002:c0a8:101::1", PxAcPrivate),
            ("2002:808:808::1", PxAcPublic),
            ("2001::1", PxAcNever),
            ("2001:10::1", PxAcNever),
            ("2001:db8::1", PxAcPrivate),
            ("100::1", PxAcNever),
            ("fc00::1", PxAcPrivate),
            ("fd00:ec2::254", PxAcPrivate),
            ("fe80::1", PxAcLinkLocal),
            ("febf::1", PxAcLinkLocal),
            ("fec0::1", PxAcPrivate),
            ("ff02::1", PxAcNever),
            ("4000::1", PxAcNever),
            ("::ffff:0:7f00:1", PxAcNever),
        ] {
            assert_eq!(class(addr), want, "{addr}");
        }
    }

    #[test]
    fn a_non_public_address_needs_the_listing_and_some_never_pass() {
        use AddrClass::*;
        use ListMatch::*;
        for c in [PxAcLoopback, PxAcPrivate, PxAcLinkLocal] {
            assert!(!addr_passes(addr_check(PxMEntry, true, c, false)), "{c:?}");
            assert!(addr_passes(addr_check(PxMEntry, true, c, true)), "{c:?}");
        }
        assert!(addr_passes(addr_check(PxMEntry, false, PxAcPublic, false)));
        assert!(!addr_passes(addr_check(PxMEntry, true, PxAcNever, true)));
        assert!(!addr_passes(addr_check(PxMNone, true, PxAcPublic, true)));
        assert!(!addr_passes(addr_check(PxMRange, false, PxAcPublic, true)));
        assert!(addr_passes(addr_check(PxMRange, true, PxAcPublic, false)));
        // one refused address refuses the name, in either order
        let ok = addr_check(PxMEntry, true, PxAcPublic, false);
        let no = addr_check(PxMEntry, true, PxAcLoopback, false);
        assert!(!addr_passes(addr_join(ok, no)));
        assert!(!addr_passes(addr_join(no, ok)));
        assert!(addr_passes(addr_join(ok, ok)));
    }

    #[test]
    fn ranges_compare_exactly_their_prefix_bits() {
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();
        assert!(net_contains(ip("10.0.0.0"), 8, ip("10.255.3.4")));
        assert!(!net_contains(ip("10.0.0.0"), 8, ip("11.0.0.1")));
        assert!(net_contains(ip("172.16.0.0"), 12, ip("172.31.255.255")));
        assert!(!net_contains(ip("172.16.0.0"), 12, ip("172.32.0.0")));
        assert!(net_contains(ip("192.0.2.128"), 25, ip("192.0.2.200")));
        assert!(!net_contains(ip("192.0.2.128"), 25, ip("192.0.2.127")));
        assert!(net_contains(ip("192.0.2.7"), 32, ip("192.0.2.7")));
        assert!(!net_contains(ip("192.0.2.7"), 32, ip("192.0.2.6")));
        assert!(net_contains(ip("192.0.2.6"), 31, ip("192.0.2.7")));
        assert!(net_contains(ip("0.0.0.0"), 0, ip("203.0.113.9")));
        assert!(net_contains(ip("fd00::"), 8, ip("fd12:3456::1")));
        assert!(!net_contains(ip("fd00::"), 8, ip("fc00::1")));
        assert!(net_contains(ip("fc00::"), 7, ip("fd00::1")));
        assert!(net_contains(ip("2001:db8::"), 33, ip("2001:db8:7fff::1")));
        assert!(!net_contains(ip("2001:db8::"), 33, ip("2001:db8:8000::1")));
        assert!(net_contains(ip("::1"), 128, ip("::1")));
        assert!(!net_contains(ip("::1"), 128, ip("::2")));
        // never across families, not even for the mapped form
        assert!(!net_contains(ip("10.0.0.0"), 8, ip("::ffff:10.0.0.1")));
        assert!(!net_contains(ip("::"), 0, ip("10.0.0.1")));
        assert!(net_bits_ok(ip("10.0.0.0"), 32) && !net_bits_ok(ip("10.0.0.0"), 33));
        assert!(net_bits_ok(ip("::"), 128) && !net_bits_ok(ip("::"), 129));
    }

    #[test]
    fn redirects_of_today() {
        use Method::*;
        use Redirect::*;
        assert_eq!(redirect_kind(301, true), PxRdMoved);
        assert_eq!(redirect_kind(302, true), PxRdMoved);
        assert_eq!(redirect_kind(303, true), PxRdSeeOther);
        assert_eq!(redirect_kind(307, true), PxRdKeep);
        assert_eq!(redirect_kind(308, true), PxRdKeep);
        for s in [200, 204, 300, 304, 305, 400, 500] {
            assert_eq!(redirect_kind(s, true), PxRdNone, "{s}");
        }
        assert_eq!(redirect_kind(302, false), PxRdNone);
        assert!(redirect_follows(PxRdKeep) && redirect_follows(PxRdMoved));
        assert!(redirect_follows(PxRdSeeOther) && !redirect_follows(PxRdNone));
        assert_eq!(redirect_method(PxPost, PxRdMoved), PxGet);
        assert_eq!(redirect_method(PxPost, PxRdSeeOther), PxGet);
        assert_eq!(redirect_method(PxPost, PxRdKeep), PxPost);
        assert_eq!(redirect_method(PxDelete, PxRdMoved), PxDelete);
        assert_eq!(redirect_method(PxDelete, PxRdSeeOther), PxGet);
        assert!(redirect_keeps_body(PxPost, PxRdKeep));
        assert!(!redirect_keeps_body(PxPost, PxRdMoved));
        // three allowed: three followed, the fourth refused
        let mut left = 3;
        let mut followed = 0;
        while redirect_may(left) {
            left = redirect_left(left);
            followed += 1;
        }
        assert_eq!(followed, 3);
    }

    #[test]
    fn caps_and_waits_of_today() {
        assert_eq!(cap_bytes(64), 64 << 20);
        assert!(cap_ok(1, 4096) && cap_ok(4096, 4096));
        assert!(!cap_ok(0, 4096) && !cap_ok(4097, 4096));
        assert!(body_fits(8, 8) && !body_fits(9, 8));
        assert_eq!(resp_take(90, 20, 100), 10);
        assert!(resp_cut(90, 20, 100));
        assert_eq!(resp_take(10, 20, 100), 20);
        assert!(!resp_cut(80, 20, 100));
        assert_eq!(resp_take(100, 5, 100), 0);
        assert_eq!(time_left(30, 100), 70);
        assert_eq!(time_left(130, 100), 0);
        assert!(expired(100, 100) && !expired(99, 100));
        assert_eq!(read_wait(500, 900, 200), 200);
        assert_eq!(read_wait(50, 900, 200), 50);
        assert_eq!(read_wait(500, 9, 200), 9);
        assert!(keepalive_due(true, true, 20, 20));
        assert!(!keepalive_due(true, true, 19, 20));
        assert!(!keepalive_due(false, true, 99, 20));
        assert!(!keepalive_due(true, false, 99, 20));
        assert!(!keepalive_due(true, true, 99, 0));
        assert_eq!(keepalive_wait(true, true, 5, 20), 15);
        assert_eq!(keepalive_wait(true, true, 25, 20), 0);
        assert_eq!(keepalive_wait(true, false, 25, 20), 20);
        assert_eq!(keepalive_wait(false, true, 5, 20), 20);
    }

    #[test]
    fn start_gate_site_and_url_of_today() {
        assert_eq!(start(false, true, true, true), Start::PxStOff);
        assert_eq!(start(true, false, false, true), Start::PxStNoList);
        assert_eq!(start(true, true, true, false), Start::PxStNoKey);
        // named targets, no key: offered (everyone who reaches the port reaches those targets)
        assert!(start_serves(start(true, true, false, false)));
        assert!(start_serves(start(true, true, true, true)));
        assert!(start_refused(start(true, false, false, false)));
        assert!(!start_refused(start(false, false, false, false)));

        // a client that is not a browser sends neither header
        assert!(site_ok(Site::PxSiteAbsent, false, false));
        assert!(site_ok(Site::PxSiteAbsent, true, true));
        assert!(!site_ok(Site::PxSiteAbsent, true, false));
        assert!(site_ok(Site::PxSiteSameOrigin, true, false));
        assert!(site_ok(Site::PxSiteNone, false, false));
        assert!(!site_ok(Site::PxSiteOther, true, true));

        assert_eq!(gate(false, true, true, true), Gate::PxGateNoKey);
        assert_eq!(gate(true, false, true, true), Gate::PxGateDisabled);
        assert_eq!(gate(true, true, false, true), Gate::PxGateForeign);
        assert_eq!(gate(true, true, true, false), Gate::PxGateMethod);
        assert_eq!(gate(true, true, true, true), Gate::PxGateRun);

        assert!(url_ok(url_check(Scheme::PxHttp, false, true, 80, false)));
        assert!(url_ok(url_check(Scheme::PxHttps, false, true, 65535, true)));
        assert_eq!(
            url_check(Scheme::PxHttps, false, true, 443, false),
            UrlVerdict::PxUrlNoTls
        );
        assert_eq!(
            url_check(Scheme::PxHttp, false, true, 65536, true),
            UrlVerdict::PxUrlPort
        );
        assert_eq!(
            url_check(Scheme::PxHttp, false, true, 0, true),
            UrlVerdict::PxUrlPort
        );
        assert_eq!(
            url_check(Scheme::PxHttp, true, true, 80, true),
            UrlVerdict::PxUrlUser
        );
        assert_eq!(
            url_check(Scheme::PxSchemeOther, false, true, 80, true),
            UrlVerdict::PxUrlScheme
        );
    }

    #[test]
    fn header_policy_of_today() {
        use ReqHeader::*;
        assert!(req_header_goes(true, PxRhNeutral, true));
        assert!(req_header_goes(true, PxRhNeutral, false));
        assert!(req_header_goes(true, PxRhBound, true));
        assert!(!req_header_goes(true, PxRhBound, false));
        assert!(!req_header_goes(true, PxRhFraming, true));
        assert!(!req_header_goes(false, PxRhNeutral, true));
        assert!(resp_header_passes(RespHeader::PxRpPlain));
        for k in [
            RespHeader::PxRpFraming,
            RespHeader::PxRpCors,
            RespHeader::PxRpState,
            RespHeader::PxRpPolicy,
        ] {
            assert!(!resp_header_passes(k), "{k:?}");
        }
    }
}
