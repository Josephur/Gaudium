# Slice 7 — Verified MCP Proxy Guard Core
# Formally specifies and proves the decisions of the server's MCP proxy (`/cors-proxy`), which
# fetches a URL named by the request on behalf of the Web UI:
#   - Whether the proxy may be started at all (asked for, targets named, a key for an open list)
#   - Whether one request runs (enabled, key, not driven by another site's page, method)
#   - Whether a target URL is one the proxy handles (scheme, no user part, port, TLS available)
#   - The class of an IPv4 / IPv6 address (public, loopback, private, link-local, never), with the
#     forms that carry an IPv4 address inside an IPv6 one
#   - Whether a resolved address may be connected to (allow-list match, range, explicit listing
#     of non-public targets), and that one refused address refuses the whole name
#   - Redirects: which are followed, with which method, and the bound on their number
#   - Which request headers go to the target and which response headers come back
#   - Size and time caps, and the keep-alive line of an idle event stream

import Base

# ============================================================================
# 1. Helper lemmas (erased before Rust codegen)
# ============================================================================

def px_le_refl(n: Nat) -> {Nat.is_le(n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      px_le_refl(p)

def px_le_succ(n: Nat) -> {Nat.is_le(n, 1n+n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      px_le_succ(p)

def px_lt_succ(n: Nat) -> {Nat.is_lt(n, 1n+n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      px_lt_succ(p)

def px_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      px_min_le_l(ap, bp)

def px_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      px_min_le_r(ap, bp)

# ============================================================================
# 2. Starting the Proxy
# ============================================================================
# The proxy is a forwarder inside the operator's network. It is offered only when asked for and
# when the operator named the targets; a list that allows any target needs an API key, because
# without one everyone who reaches the port could use the server as an open forwarder.

type PxStart is Data:
  PxStOff{}
  PxStNoList{}
  PxStNoKey{}
  PxStOn{}

# asked:  the proxy flag was given
# listed: at least one target entry was given
# open:   an entry allows any target
# keyed:  the server has an API key
def px_start(asked: Bool, listed: Bool, open: Bool, keyed: Bool) -> PxStart:
  match asked:
    case False{}:
      PxStOff{}
    case True{}:
      match listed:
        case False{}:
          PxStNoList{}
        case True{}:
          match open:
            case False{}:
              PxStOn{}
            case True{}:
              match keyed:
                case False{}:
                  PxStNoKey{}
                case True{}:
                  PxStOn{}

# The proxy is offered.
def px_start_serves(s: PxStart) -> Bool:
  match s:
    case PxStOn{}:
      True{}
    case _:
      False{}

# The configuration is refused (the server does not start with it).
def px_start_refuses(s: PxStart) -> Bool:
  match s:
    case PxStNoList{}:
      True{}
    case PxStNoKey{}:
      True{}
    case _:
      False{}

# LAW 2.1: Not asked for, the proxy is off whatever else is configured.
law px_start_off_unless_asked:
  for l: Bool
  for o: Bool
  for k: Bool
  {px_start(False{}, l, o, k) == PxStOff{} : PxStart}

def px_start_off_unless_asked(l, o, k):
  {==}

# LAW 2.2: Without a named target the proxy is never offered.
law px_start_needs_targets:
  for +a: Bool
  for o: Bool
  for k: Bool
  {px_start_serves(px_start(a, False{}, o, k)) == False{} : Bool}

def px_start_needs_targets(a, o, k):
  match a:
    case True{}: {==}
    case False{}: {==}

# LAW 2.3: A list that allows any target is never offered without an API key.
law px_start_open_needs_key:
  for +a: Bool
  for +l: Bool
  {px_start_serves(px_start(a, l, True{}, False{})) == False{} : Bool}

def px_start_open_needs_key(a, l):
  match a l:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# LAW 2.4: A configuration is never both offered and refused.
law px_start_serve_xor_refuse:
  for +s: PxStart
  {Bool.and(px_start_serves(s), px_start_refuses(s)) == False{} : Bool}

def px_start_serve_xor_refuse(s):
  match s:
    case PxStOff{}: {==}
    case PxStNoList{}: {==}
    case PxStNoKey{}: {==}
    case PxStOn{}: {==}

# ============================================================================
# 3. One Request
# ============================================================================
# A page of another site, open in the same browser, must not be able to drive the proxy. The
# browser says where a request comes from (its fetch-site statement, which page script cannot
# set); without that statement the request's own origin is compared with the host the request
# itself names. No configured name takes part, so the rule is the same behind any front end.

type PxSite is Data:
  PxSiteAbsent{}
  PxSiteSameOrigin{}
  PxSiteNone{}
  PxSiteOther{}

# site:           the browser's fetch-site statement (absent, same-origin, none = typed by the
#                 person, other = any other value)
# has_origin:     the request carries an origin
# origin_is_host: that origin names the host the request was sent to
def px_site_ok(site: PxSite, has_origin: Bool, origin_is_host: Bool) -> Bool:
  match site:
    case PxSiteSameOrigin{}:
      True{}
    case PxSiteNone{}:
      True{}
    case PxSiteOther{}:
      False{}
    case PxSiteAbsent{}:
      match has_origin:
        case False{}:
          True{}
        case True{}:
          origin_is_host

type PxGate is Data:
  PxGateNoKey{}
  PxGateDisabled{}
  PxGateForeign{}
  PxGateMethod{}
  PxGateRun{}

# key_ok:    the request carries the server's API key (or the server has none)
# enabled:   the proxy is offered
# site_ok:   px_site_ok of the request
# method_ok: a method the proxy forwards
def px_gate(key_ok: Bool, enabled: Bool, site_ok: Bool, method_ok: Bool) -> PxGate:
  match key_ok:
    case False{}:
      PxGateNoKey{}
    case True{}:
      match enabled:
        case False{}:
          PxGateDisabled{}
        case True{}:
          match site_ok:
            case False{}:
              PxGateForeign{}
            case True{}:
              match method_ok:
                case False{}:
                  PxGateMethod{}
                case True{}:
                  PxGateRun{}

def px_gate_runs(g: PxGate) -> Bool:
  match g:
    case PxGateRun{}:
      True{}
    case _:
      False{}

# LAW 3.1: A request another site's page sent is refused whatever origin it claims.
law px_site_other_refused:
  for o: Bool
  for h: Bool
  {px_site_ok(PxSiteOther{}, o, h) == False{} : Bool}

def px_site_other_refused(o, h):
  {==}

# LAW 3.2: Without the browser's statement, an origin that is not the request's own host is
# refused.
law px_site_foreign_origin_refused:
  {px_site_ok(PxSiteAbsent{}, True{}, False{}) == False{} : Bool}

def px_site_foreign_origin_refused():
  {==}

# LAW 3.3: A request without the API key never runs.
law px_gate_needs_key:
  for e: Bool
  for s: Bool
  for m: Bool
  {px_gate_runs(px_gate(False{}, e, s, m)) == False{} : Bool}

def px_gate_needs_key(e, s, m):
  {==}

# LAW 3.4: With the proxy off no request runs.
law px_gate_off_never_runs:
  for +k: Bool
  for s: Bool
  for m: Bool
  {px_gate_runs(px_gate(k, False{}, s, m)) == False{} : Bool}

def px_gate_off_never_runs(k, s, m):
  match k:
    case True{}: {==}
    case False{}: {==}

# LAW 3.5: A request that failed the site rule never runs.
law px_gate_foreign_never_runs:
  for +k: Bool
  for +e: Bool
  for m: Bool
  {px_gate_runs(px_gate(k, e, False{}, m)) == False{} : Bool}

def px_gate_foreign_never_runs(k, e, m):
  match k e:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# LAW 3.6: A method the proxy does not forward never runs.
law px_gate_method_never_runs:
  for +k: Bool
  for +e: Bool
  for +s: Bool
  {px_gate_runs(px_gate(k, e, s, False{})) == False{} : Bool}

def px_gate_method_never_runs(k, e, s):
  match k e s:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# ============================================================================
# 4. The Target URL
# ============================================================================

type PxScheme is Data:
  PxHttp{}
  PxHttps{}
  PxSchemeOther{}

type PxUrl is Data:
  PxUrlOk{}
  PxUrlScheme{}
  PxUrlUser{}
  PxUrlHost{}
  PxUrlPort{}
  PxUrlNoTls{}

# A TCP port a client may connect to.
def px_port_ok(+port: Nat) -> Bool:
  Bool.and(Nat.is_gt(port, 0n), Nat.is_le(port, 65535n))

def px_url_rest(user: Bool, host_ok: Bool, port_ok: Bool) -> PxUrl:
  match user:
    case True{}:
      PxUrlUser{}
    case False{}:
      match host_ok:
        case False{}:
          PxUrlHost{}
        case True{}:
          match port_ok:
            case False{}:
              PxUrlPort{}
            case True{}:
              PxUrlOk{}

# scheme:    the URL's scheme
# user:      the URL has a user part (name or password before the host)
# host_ok:   the host is a well-formed name or address literal
# port_ok:   px_port_ok of its port
# tls_ready: the TLS library is loaded
def px_url_check(scheme: PxScheme, user: Bool, host_ok: Bool, port_ok: Bool, tls_ready: Bool) -> PxUrl:
  match scheme:
    case PxSchemeOther{}:
      PxUrlScheme{}
    case PxHttp{}:
      px_url_rest(user, host_ok, port_ok)
    case PxHttps{}:
      match tls_ready:
        case False{}:
          PxUrlNoTls{}
        case True{}:
          px_url_rest(user, host_ok, port_ok)

def px_url_ok(u: PxUrl) -> Bool:
  match u:
    case PxUrlOk{}:
      True{}
    case _:
      False{}

# LAW 4.1: Port 0 is not a port.
law px_port_zero_refused:
  {px_port_ok(0n) == False{} : Bool}

def px_port_zero_refused():
  {==}

# LAW 4.2: Only http and https targets are fetched.
law px_url_other_scheme_refused:
  for u: Bool
  for h: Bool
  for p: Bool
  for t: Bool
  {px_url_ok(px_url_check(PxSchemeOther{}, u, h, p, t)) == False{} : Bool}

def px_url_other_scheme_refused(u, h, p, t):
  {==}

# LAW 4.3: An https target is never fetched without the TLS library (and so never in the clear).
law px_url_https_needs_tls:
  for u: Bool
  for h: Bool
  for p: Bool
  {px_url_ok(px_url_check(PxHttps{}, u, h, p, False{})) == False{} : Bool}

def px_url_https_needs_tls(u, h, p):
  {==}

# LAW 4.4: A URL with a user part is refused.
law px_url_user_refused:
  for +s: PxScheme
  for h: Bool
  for p: Bool
  for +t: Bool
  {px_url_ok(px_url_check(s, True{}, h, p, t)) == False{} : Bool}

def px_url_user_refused(s, h, p, t):
  match s t:
    case PxHttp{} True{}: {==}
    case PxHttp{} False{}: {==}
    case PxHttps{} True{}: {==}
    case PxHttps{} False{}: {==}
    case PxSchemeOther{} True{}: {==}
    case PxSchemeOther{} False{}: {==}

def px_url_rest_bad_port(+u: Bool, +h: Bool) -> {px_url_ok(px_url_rest(u, h, False{})) == False{} : Bool}:
  match u h:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# LAW 4.5: A URL with a port outside the valid range is refused.
law px_url_bad_port_refused:
  for +s: PxScheme
  for +u: Bool
  for +h: Bool
  for +t: Bool
  {px_url_ok(px_url_check(s, u, h, False{}, t)) == False{} : Bool}

def px_url_bad_port_refused(s, u, h, t):
  match s:
    case PxSchemeOther{}: {==}
    case PxHttp{}:
      px_url_rest_bad_port(u, h)
    case PxHttps{}:
      match t:
        case False{}: {==}
        case True{}:
          px_url_rest_bad_port(u, h)

# ============================================================================
# 5. Address Classes
# ============================================================================
# Every address a target resolves to is put in one class. Public addresses need nothing more;
# loopback, private and link-local ones (the last hold the instance-metadata endpoints) need the
# operator's explicit listing; the rest is never connected to.

type PxAddrClass is Data:
  PxAcPublic{}
  PxAcLoopback{}
  PxAcPrivate{}
  PxAcLinkLocal{}
  PxAcNever{}

def px_ac_public(c: PxAddrClass) -> Bool:
  match c:
    case PxAcPublic{}:
      True{}
    case _:
      False{}

def px_in(+x: Nat, lo: Nat, hi: Nat) -> Bool:
  Bool.and(Nat.is_ge(x, lo), Nat.is_le(x, hi))

def px_eq2(a: Nat, x: Nat, b: Nat, y: Nat) -> Bool:
  Bool.and(Nat.is_eq(a, x), Nat.is_eq(b, y))

def px_eq3(a: Nat, x: Nat, b: Nat, y: Nat, c: Nat, z: Nat) -> Bool:
  Bool.and(Nat.is_eq(a, x), Bool.and(Nat.is_eq(b, y), Nat.is_eq(c, z)))

# The class of the IPv4 address a.b.c.d (each an octet).
def px_v4_class(+a: Nat, +b: Nat, +c: Nat, d: Nat) -> PxAddrClass:
  Bool.pick(PxAddrClass, Nat.is_eq(a, 0n), PxAcNever{},
  Bool.pick(PxAddrClass, Nat.is_eq(a, 10n), PxAcPrivate{},
  Bool.pick(PxAddrClass, Bool.and(Nat.is_eq(a, 100n), px_in(b, 64n, 127n)), PxAcPrivate{},
  Bool.pick(PxAddrClass, Nat.is_eq(a, 127n), PxAcLoopback{},
  Bool.pick(PxAddrClass, px_eq2(a, 169n, b, 254n), PxAcLinkLocal{},
  Bool.pick(PxAddrClass, Bool.and(Nat.is_eq(a, 172n), px_in(b, 16n, 31n)), PxAcPrivate{},
  Bool.pick(PxAddrClass, px_eq3(a, 192n, b, 0n, c, 0n), PxAcPrivate{},
  Bool.pick(PxAddrClass, px_eq3(a, 192n, b, 0n, c, 2n), PxAcPrivate{},
  Bool.pick(PxAddrClass, px_eq3(a, 192n, b, 88n, c, 99n), PxAcNever{},
  Bool.pick(PxAddrClass, px_eq2(a, 192n, b, 168n), PxAcPrivate{},
  Bool.pick(PxAddrClass, Bool.and(Nat.is_eq(a, 198n), px_in(b, 18n, 19n)), PxAcPrivate{},
  Bool.pick(PxAddrClass, px_eq3(a, 198n, b, 51n, c, 100n), PxAcPrivate{},
  Bool.pick(PxAddrClass, px_eq3(a, 203n, b, 0n, c, 113n), PxAcPrivate{},
  Bool.pick(PxAddrClass, Nat.is_ge(a, 224n), PxAcNever{},
  PxAcPublic{}))))))))))))))

# The first eight of ten bytes are zero.
def px_zero4(a: Nat, b: Nat, c: Nat, d: Nat) -> Bool:
  Bool.and(Nat.is_eq(a, 0n), Bool.and(Nat.is_eq(b, 0n), Bool.and(Nat.is_eq(c, 0n), Nat.is_eq(d, 0n))))

# Addresses 00xx:… : the IPv4-mapped form, the two translation prefixes, the loopback address.
# mid_zero: bytes 4 to 7 are zero; tail_zero: bytes 8 and 9 are zero.
def px_v6_low(+b1: Nat, +b2: Nat, +b3: Nat, +mid_zero: Bool, b5: Nat, +tail_zero: Bool, +b10: Nat, +b11: Nat, +b12: Nat, +b13: Nat, +b14: Nat, +b15: Nat) -> PxAddrClass:
  Bool.pick(PxAddrClass, Bool.and(px_eq3(b1, 0n, b2, 0n, b3, 0n), Bool.and(mid_zero, tail_zero)),
    Bool.pick(PxAddrClass, px_eq2(b10, 255n, b11, 255n), px_v4_class(b12, b13, b14, b15),
    Bool.pick(PxAddrClass, Bool.and(px_eq2(b10, 0n, b11, 0n), Bool.and(px_eq3(b12, 0n, b13, 0n, b14, 0n), Nat.is_eq(b15, 1n))), PxAcLoopback{},
    PxAcNever{})),
  Bool.pick(PxAddrClass, px_eq3(b1, 100n, b2, 255n, b3, 155n),
    Bool.pick(PxAddrClass, Bool.and(mid_zero, Bool.and(tail_zero, px_eq2(b10, 0n, b11, 0n))), px_v4_class(b12, b13, b14, b15),
    PxAcPrivate{}),
  PxAcNever{}))

# Addresses whose first byte is not zero.
def px_v6_high(+b0: Nat, +b1: Nat, +b2: Nat, +b3: Nat, b4: Nat, b5: Nat) -> PxAddrClass:
  Bool.pick(PxAddrClass, px_eq2(b0, 32n, b1, 2n), px_v4_class(b2, b3, b4, b5),
  Bool.pick(PxAddrClass, Bool.and(px_eq2(b0, 32n, b1, 1n), Nat.is_le(b2, 1n)), PxAcNever{},
  Bool.pick(PxAddrClass, px_eq2(b0, 32n, b1, 1n), Bool.pick(PxAddrClass, px_eq2(b2, 13n, b3, 184n), PxAcPrivate{}, PxAcPublic{}),
  Bool.pick(PxAddrClass, px_in(b0, 32n, 63n), PxAcPublic{},
  Bool.pick(PxAddrClass, px_in(b0, 252n, 253n), PxAcPrivate{},
  Bool.pick(PxAddrClass, Bool.and(Nat.is_eq(b0, 254n), px_in(b1, 128n, 191n)), PxAcLinkLocal{},
  Bool.pick(PxAddrClass, Bool.and(Nat.is_eq(b0, 254n), Nat.is_ge(b1, 192n)), PxAcPrivate{},
  PxAcNever{})))))))

# The class of an IPv6 address given as its sixteen bytes.
def px_v6_class(+b0: Nat, +b1: Nat, +b2: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +b8: Nat, +b9: Nat, +b10: Nat, +b11: Nat, +b12: Nat, +b13: Nat, +b14: Nat, +b15: Nat) -> PxAddrClass:
  Bool.pick(PxAddrClass, Nat.is_eq(b0, 0n),
    px_v6_low(b1, b2, b3, px_zero4(b4, b5, b6, b7), b5, px_eq2(b8, 0n, b9, 0n), b10, b11, b12, b13, b14, b15),
    px_v6_high(b0, b1, b2, b3, b4, b5))

# class:  the address's class
# listed: the operator listed this target (or a range holding this address) as one that may be
#         non-public
def px_addr_ok(class: PxAddrClass, listed: Bool) -> Bool:
  match class:
    case PxAcPublic{}:
      True{}
    case PxAcNever{}:
      False{}
    case PxAcLoopback{}:
      listed
    case PxAcPrivate{}:
      listed
    case PxAcLinkLocal{}:
      listed

# Every class is public or it is not (used to carry a class through a form that embeds it).
def px_ac_public_or_not(c: PxAddrClass) -> {Bool.or(px_ac_public(c), Bool.not(px_ac_public(c))) == True{} : Bool}:
  match c:
    case PxAcPublic{}: {==}
    case PxAcLoopback{}: {==}
    case PxAcPrivate{}: {==}
    case PxAcLinkLocal{}: {==}
    case PxAcNever{}: {==}

# LAW 5.1: 127.x.x.x is never public.
law px_v4_loopback_not_public:
  for b: Nat
  for c: Nat
  for d: Nat
  {px_ac_public(px_v4_class(127n, b, c, d)) == False{} : Bool}

def px_v4_loopback_not_public(b, c, d):
  {==}

# LAW 5.2: 10.x.x.x is never public.
law px_v4_ten_not_public:
  for b: Nat
  for c: Nat
  for d: Nat
  {px_ac_public(px_v4_class(10n, b, c, d)) == False{} : Bool}

def px_v4_ten_not_public(b, c, d):
  {==}

# LAW 5.3: 192.168.x.x is never public.
law px_v4_192_168_not_public:
  for c: Nat
  for d: Nat
  {px_ac_public(px_v4_class(192n, 168n, c, d)) == False{} : Bool}

def px_v4_192_168_not_public(c, d):
  {==}

# LAW 5.4: 172.16.x.x and 172.31.x.x (the ends of the private block) are never public.
law px_v4_172_16_not_public:
  for c: Nat
  for d: Nat
  {px_ac_public(px_v4_class(172n, 16n, c, d)) == False{} : Bool}

def px_v4_172_16_not_public(c, d):
  {==}

law px_v4_172_31_not_public:
  for c: Nat
  for d: Nat
  {px_ac_public(px_v4_class(172n, 31n, c, d)) == False{} : Bool}

def px_v4_172_31_not_public(c, d):
  {==}

# LAW 5.5: 169.254.x.x (link-local, where the instance-metadata endpoints live) is never public.
law px_v4_link_local_not_public:
  for c: Nat
  for d: Nat
  {px_ac_public(px_v4_class(169n, 254n, c, d)) == False{} : Bool}

def px_v4_link_local_not_public(c, d):
  {==}

# LAW 5.6: 100.64.x.x (shared address space) is never public.
law px_v4_shared_not_public:
  for c: Nat
  for d: Nat
  {px_ac_public(px_v4_class(100n, 64n, c, d)) == False{} : Bool}

def px_v4_shared_not_public(c, d):
  {==}

# LAW 5.7: 0.x.x.x ("this host") is never connected to, listed or not.
law px_v4_zero_never:
  for b: Nat
  for c: Nat
  for d: Nat
  for l: Bool
  {px_addr_ok(px_v4_class(0n, b, c, d), l) == False{} : Bool}

def px_v4_zero_never(b, c, d, l):
  {==}

# LAW 5.8: The IPv6 loopback address is never public.
law px_v6_loopback_not_public:
  {px_ac_public(px_v6_class(0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 1n)) == False{} : Bool}

def px_v6_loopback_not_public():
  {==}

# LAW 5.9: The unspecified IPv6 address is never connected to, listed or not.
law px_v6_unspecified_never:
  for l: Bool
  {px_addr_ok(px_v6_class(0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n), l) == False{} : Bool}

def px_v6_unspecified_never(l):
  {==}

# LAW 5.10: An IPv4-mapped IPv6 address (::ffff:a.b.c.d) is public only if a.b.c.d is: the
# mapped form opens nothing its IPv4 address does not.
law px_v6_mapped_no_wider:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for +d: Nat
  {Bool.or(px_ac_public(px_v4_class(a, b, c, d)), Bool.not(px_ac_public(px_v6_class(0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 255n, 255n, a, b, c, d)))) == True{} : Bool}

def px_v6_mapped_no_wider(a, b, c, d):
  px_ac_public_or_not(px_v4_class(a, b, c, d))

# LAW 5.11: The same for the translation prefix 64:ff9b::a.b.c.d.
law px_v6_nat64_no_wider:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for +d: Nat
  {Bool.or(px_ac_public(px_v4_class(a, b, c, d)), Bool.not(px_ac_public(px_v6_class(0n, 100n, 255n, 155n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, a, b, c, d)))) == True{} : Bool}

def px_v6_nat64_no_wider(a, b, c, d):
  px_ac_public_or_not(px_v4_class(a, b, c, d))

# LAW 5.12: The same for the 2002:AABB:CCDD:: form, which carries a.b.c.d in bytes 2 to 5.
law px_v6_6to4_no_wider:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for +d: Nat
  for b6: Nat
  for b7: Nat
  for b8: Nat
  for b9: Nat
  for b10: Nat
  for b11: Nat
  for b12: Nat
  for b13: Nat
  for b14: Nat
  for b15: Nat
  {Bool.or(px_ac_public(px_v4_class(a, b, c, d)), Bool.not(px_ac_public(px_v6_class(32n, 2n, a, b, c, d, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15)))) == True{} : Bool}

def px_v6_6to4_no_wider(a, b, c, d, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15):
  px_ac_public_or_not(px_v4_class(a, b, c, d))

# LAW 5.13: Unique-local IPv6 addresses (fd00::/8, where one instance-metadata endpoint lives)
# are never public.
law px_v6_unique_local_not_public:
  for b1: Nat
  for b2: Nat
  for b3: Nat
  for b4: Nat
  for b5: Nat
  for b6: Nat
  for b7: Nat
  for b8: Nat
  for b9: Nat
  for b10: Nat
  for b11: Nat
  for b12: Nat
  for b13: Nat
  for b14: Nat
  for b15: Nat
  {px_ac_public(px_v6_class(253n, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15)) == False{} : Bool}

def px_v6_unique_local_not_public(b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15):
  {==}

# LAW 5.14: Link-local IPv6 addresses (fe80::/16 shown) are never public.
law px_v6_link_local_not_public:
  for b2: Nat
  for b3: Nat
  for b4: Nat
  for b5: Nat
  for b6: Nat
  for b7: Nat
  for b8: Nat
  for b9: Nat
  for b10: Nat
  for b11: Nat
  for b12: Nat
  for b13: Nat
  for b14: Nat
  for b15: Nat
  {px_ac_public(px_v6_class(254n, 128n, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15)) == False{} : Bool}

def px_v6_link_local_not_public(b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15):
  {==}

# ============================================================================
# 6. May This Address Be Connected To
# ============================================================================

# How the allow-list matched the target.
type PxMatch is Data:
  PxMNone{}
  PxMEntry{}
  PxMRange{}

# A target no entry matches is refused before its name is looked up.
def px_match_resolves(m: PxMatch) -> Bool:
  match m:
    case PxMNone{}:
      False{}
    case _:
      True{}

type PxAddr is Data:
  PxAddrOk{}
  PxAddrUnlisted{}
  PxAddrOutside{}
  PxAddrPrivate{}
  PxAddrNever{}

def px_addr_listed_check(listed: Bool) -> PxAddr:
  match listed:
    case True{}:
      PxAddrOk{}
    case False{}:
      PxAddrPrivate{}

def px_addr_class_check(class: PxAddrClass, listed: Bool) -> PxAddr:
  match class:
    case PxAcPublic{}:
      PxAddrOk{}
    case PxAcNever{}:
      PxAddrNever{}
    case PxAcLoopback{}:
      px_addr_listed_check(listed)
    case PxAcPrivate{}:
      px_addr_listed_check(listed)
    case PxAcLinkLocal{}:
      px_addr_listed_check(listed)

# m:        PxMNone = no entry; PxMEntry = an entry names the target (a name, a name pattern, the
#           any-target entry, or a range holding the address the URL gives); PxMRange = the target
#           is a name and only ranges could admit it, so each address it resolves to must lie in one
# in_range: this address lies in a range entry
# class:    the address's class
# listed:   as in px_addr_ok
def px_addr_check(m: PxMatch, in_range: Bool, class: PxAddrClass, listed: Bool) -> PxAddr:
  match m:
    case PxMNone{}:
      PxAddrUnlisted{}
    case PxMEntry{}:
      px_addr_class_check(class, listed)
    case PxMRange{}:
      match in_range:
        case False{}:
          PxAddrOutside{}
        case True{}:
          px_addr_class_check(class, listed)

def px_addr_passes(v: PxAddr) -> Bool:
  match v:
    case PxAddrOk{}:
      True{}
    case _:
      False{}

# A name may resolve to several addresses: the first refusal is the name's verdict.
def px_addr_join(acc: PxAddr, next: PxAddr) -> PxAddr:
  match acc:
    case PxAddrOk{}:
      next
    case PxAddrUnlisted{}:
      PxAddrUnlisted{}
    case PxAddrOutside{}:
      PxAddrOutside{}
    case PxAddrPrivate{}:
      PxAddrPrivate{}
    case PxAddrNever{}:
      PxAddrNever{}

# LAW 6.1: An address that is not public is never allowed without the operator's listing.
law px_nonpublic_needs_listing:
  for +c: PxAddrClass
  {Bool.or(px_ac_public(c), Bool.not(px_addr_ok(c, False{}))) == True{} : Bool}

def px_nonpublic_needs_listing(c):
  match c:
    case PxAcPublic{}: {==}
    case PxAcLoopback{}: {==}
    case PxAcPrivate{}: {==}
    case PxAcLinkLocal{}: {==}
    case PxAcNever{}: {==}

# LAW 6.2: An address of the never class is refused even when listed.
law px_never_class_refused:
  for l: Bool
  {px_addr_ok(PxAcNever{}, l) == False{} : Bool}

def px_never_class_refused(l):
  {==}

def px_or_true_r(b: Bool) -> {Bool.or(b, True{}) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def px_class_check_within(+c: PxAddrClass, +l: Bool) -> {Bool.or(px_addr_ok(c, l), Bool.not(px_addr_passes(px_addr_class_check(c, l)))) == True{} : Bool}:
  match c l:
    case PxAcPublic{} True{}: {==}
    case PxAcPublic{} False{}: {==}
    case PxAcLoopback{} True{}: {==}
    case PxAcLoopback{} False{}: {==}
    case PxAcPrivate{} True{}: {==}
    case PxAcPrivate{} False{}: {==}
    case PxAcLinkLocal{} True{}: {==}
    case PxAcLinkLocal{} False{}: {==}
    case PxAcNever{} True{}: {==}
    case PxAcNever{} False{}: {==}

# LAW 6.3: The full check allows an address only if px_addr_ok does (the allow-list can refuse
# more, never admit more).
law px_check_within_addr_ok:
  for +m: PxMatch
  for +r: Bool
  for +c: PxAddrClass
  for +l: Bool
  {Bool.or(px_addr_ok(c, l), Bool.not(px_addr_passes(px_addr_check(m, r, c, l)))) == True{} : Bool}

def px_check_within_addr_ok(m, r, c, l):
  match m:
    case PxMNone{}:
      px_or_true_r(px_addr_ok(c, l))
    case PxMEntry{}:
      px_class_check_within(c, l)
    case PxMRange{}:
      match r:
        case False{}:
          px_or_true_r(px_addr_ok(c, l))
        case True{}:
          px_class_check_within(c, l)

# LAW 6.4: A target no entry matches is refused whatever it resolves to.
law px_unlisted_target_refused:
  for r: Bool
  for c: PxAddrClass
  for l: Bool
  {px_addr_passes(px_addr_check(PxMNone{}, r, c, l)) == False{} : Bool}

def px_unlisted_target_refused(r, c, l):
  {==}

# LAW 6.5: A name admitted only through ranges is refused for an address outside them.
law px_range_name_resolves_inside:
  for c: PxAddrClass
  for l: Bool
  {px_addr_passes(px_addr_check(PxMRange{}, False{}, c, l)) == False{} : Bool}

def px_range_name_resolves_inside(c, l):
  {==}

# LAW 6.6: A name passes only if every one of its addresses passes: one refused address refuses
# the name.
law px_join_all_must_pass:
  for +a: PxAddr
  for +n: PxAddr
  {px_addr_passes(px_addr_join(a, n)) == Bool.and(px_addr_passes(a), px_addr_passes(n)) : Bool}

def px_join_all_must_pass(a, n):
  match a:
    case PxAddrOk{}:
      match n:
        case PxAddrOk{}: {==}
        case PxAddrUnlisted{}: {==}
        case PxAddrOutside{}: {==}
        case PxAddrPrivate{}: {==}
        case PxAddrNever{}: {==}
    case PxAddrUnlisted{}: {==}
    case PxAddrOutside{}: {==}
    case PxAddrPrivate{}: {==}
    case PxAddrNever{}: {==}

# ----------------------------------------------------------------------------
# Range entries (address/prefix-length)
# ----------------------------------------------------------------------------
# An address lies in a range when its leading `bits` bits equal the range's. The comparison goes
# byte by byte: byte number `index` is compared in its leading px_net_bits_at(bits, index) bits.

# A prefix length that fits the address family (32 or 128 bits).
def px_net_bits_ok(bits: Nat, family_bits: Nat) -> Bool:
  Nat.is_le(bits, family_bits)

# Of a prefix of `bits` bits, how many are left to compare from byte `index` on.
def px_net_bits_at(bits: Nat, index: Nat) -> Nat:
  Nat.sub(bits, Nat.mul(index, 8n))

# The weight of the lowest compared bit of a byte: 0 when no bit is compared, 1 for all eight.
def px_net_span(bits: Nat) -> Nat:
  match bits:
    case 0n: 0n
    case 1n: 128n
    case 2n: 64n
    case 3n: 32n
    case 4n: 16n
    case 5n: 8n
    case 6n: 4n
    case 7n: 2n
    case _: 1n

def px_net_div_eq(a: Nat, b: Nat, +d: Nat) -> Bool:
  Nat.is_eq(Nat.div(a, d), Nat.div(b, d))

def px_net_span_wide(a: Nat, b: Nat, below: Nat) -> Bool:
  match below:
    case 0n:
      Nat.is_eq(a, b)
    case 1n+q:
      px_net_div_eq(a, b, 1n+1n+q)

def px_net_span_match(a: Nat, b: Nat, span: Nat) -> Bool:
  match span:
    case 0n:
      True{}
    case 1n+below:
      px_net_span_wide(a, b, below)

# Bytes a and b agree in their leading `bits` bits (eight or more: in all of them).
def px_net_byte_match(a: Nat, b: Nat, bits: Nat) -> Bool:
  px_net_span_match(a, b, px_net_span(bits))

# LAW 6.7: With all eight bits compared, only equal bytes match: a full-length entry admits
# exactly its own address.
law px_net_full_byte_exact:
  for a: Nat
  for b: Nat
  {px_net_byte_match(a, b, 8n) == Nat.is_eq(a, b) : Bool}

def px_net_full_byte_exact(a, b):
  {==}

# LAW 6.8: Beyond the prefix nothing is compared.
law px_net_no_bits_any_byte:
  for a: Nat
  for b: Nat
  {px_net_byte_match(a, b, 0n) == True{} : Bool}

def px_net_no_bits_any_byte(a, b):
  {==}

# LAW 6.9: Past the last prefix byte no bits are left (the comparison ends with the prefix).
law px_net_bits_run_out:
  for i: Nat
  {px_net_bits_at(0n, i) == 0n : Nat}

def px_net_bits_run_out(i):
  {==}

# ============================================================================
# 7. Redirects
# ============================================================================
# The proxy follows a redirect itself, through every check above, a bounded number of times.

type PxRedirect is Data:
  PxRdNone{}
  PxRdKeep{}
  PxRdMoved{}
  PxRdSeeOther{}

# The redirect a response status asks for; none without a location to go to.
def px_redirect_kind(+status: Nat, has_location: Bool) -> PxRedirect:
  match has_location:
    case False{}:
      PxRdNone{}
    case True{}:
      Bool.pick(PxRedirect, Bool.or(Nat.is_eq(status, 307n), Nat.is_eq(status, 308n)), PxRdKeep{},
      Bool.pick(PxRedirect, Bool.or(Nat.is_eq(status, 301n), Nat.is_eq(status, 302n)), PxRdMoved{},
      Bool.pick(PxRedirect, Nat.is_eq(status, 303n), PxRdSeeOther{},
      PxRdNone{})))

def px_redirect_follows(k: PxRedirect) -> Bool:
  match k:
    case PxRdNone{}:
      False{}
    case _:
      True{}

type PxMethod is Data:
  PxGet{}
  PxPost{}
  PxDelete{}

def px_method_eq(a: PxMethod, b: PxMethod) -> Bool:
  match a b:
    case PxGet{} PxGet{}:
      True{}
    case PxPost{} PxPost{}:
      True{}
    case PxDelete{} PxDelete{}:
      True{}
    case _ _:
      False{}

# The method of the request that follows a redirect.
def px_redirect_method(m: PxMethod, k: PxRedirect) -> PxMethod:
  match m k:
    case PxPost{} PxRdSeeOther{}:
      PxGet{}
    case PxPost{} PxRdMoved{}:
      PxGet{}
    case PxPost{} _:
      PxPost{}
    case PxDelete{} PxRdSeeOther{}:
      PxGet{}
    case PxDelete{} _:
      PxDelete{}
    case PxGet{} _:
      PxGet{}

# The request body goes along only while the method stays what the page sent.
def px_redirect_keeps_body(+m: PxMethod, k: PxRedirect) -> Bool:
  px_method_eq(px_redirect_method(m, k), m)

# With `left` redirects still allowed, may one more be followed, and how many are left then.
def px_redirect_may(left: Nat) -> Bool:
  match left:
    case 0n:
      False{}
    case 1n+p:
      True{}

def px_redirect_left(left: Nat) -> Nat:
  match left:
    case 0n:
      0n
    case 1n+p:
      p

# LAW 7.1: A response without a location is never followed.
law px_redirect_needs_location:
  for s: Nat
  {px_redirect_follows(px_redirect_kind(s, False{})) == False{} : Bool}

def px_redirect_needs_location(s):
  {==}

# LAW 7.2: With no redirect left, none is followed.
law px_redirect_spent_refuses:
  {px_redirect_may(0n) == False{} : Bool}

def px_redirect_spent_refuses():
  {==}

# LAW 7.3: Every followed redirect uses up one of those left, so their number never exceeds the
# limit the request started with.
law px_redirect_budget_shrinks:
  for +p: Nat
  {Nat.is_lt(px_redirect_left(1n+p), 1n+p) == True{} : Bool}

def px_redirect_budget_shrinks(p):
  px_lt_succ(p)

# LAW 7.4: The number left never grows.
law px_redirect_budget_never_grows:
  for +l: Nat
  {Nat.is_le(px_redirect_left(l), l) == True{} : Bool}

def px_redirect_budget_never_grows(l):
  match l:
    case 0n: {==}
    case 1n+p:
      px_le_succ(p)

# LAW 7.5: A redirect never turns a request into anything but itself or a GET.
law px_redirect_same_or_get:
  for +m: PxMethod
  for +k: PxRedirect
  {Bool.or(px_method_eq(px_redirect_method(m, k), m), px_method_eq(px_redirect_method(m, k), PxGet{})) == True{} : Bool}

def px_redirect_same_or_get(m, k):
  match m k:
    case PxGet{} PxRdNone{}: {==}
    case PxGet{} PxRdKeep{}: {==}
    case PxGet{} PxRdMoved{}: {==}
    case PxGet{} PxRdSeeOther{}: {==}
    case PxPost{} PxRdNone{}: {==}
    case PxPost{} PxRdKeep{}: {==}
    case PxPost{} PxRdMoved{}: {==}
    case PxPost{} PxRdSeeOther{}: {==}
    case PxDelete{} PxRdNone{}: {==}
    case PxDelete{} PxRdKeep{}: {==}
    case PxDelete{} PxRdMoved{}: {==}
    case PxDelete{} PxRdSeeOther{}: {==}

# LAW 7.6: A GET stays a GET.
law px_redirect_get_stays_get:
  for +k: PxRedirect
  {px_redirect_method(PxGet{}, k) == PxGet{} : PxMethod}

def px_redirect_get_stays_get(k):
  match k:
    case PxRdNone{}: {==}
    case PxRdKeep{}: {==}
    case PxRdMoved{}: {==}
    case PxRdSeeOther{}: {==}

# ============================================================================
# 8. Headers
# ============================================================================
# The page marks each header it wants forwarded with a prefix. Nothing else of the request goes
# to the target, so the server's own API key and cookies stay here.

type PxReqHeader is Data:
  PxRhFraming{}
  PxRhNeutral{}
  PxRhBound{}

# prefixed:  the header carries the forwarding prefix and a valid name after it
# kind:      PxRhFraming = a header that frames the connection or the message, which the proxy
#            writes itself; PxRhNeutral = content negotiation; PxRhBound = any other header: it
#            may carry a credential for the origin the page named
# same_origin: the request goes to the origin the page named (false after a redirect to another)
def px_req_header_goes(prefixed: Bool, kind: PxReqHeader, same_origin: Bool) -> Bool:
  match prefixed:
    case False{}:
      False{}
    case True{}:
      match kind:
        case PxRhFraming{}:
          False{}
        case PxRhNeutral{}:
          True{}
        case PxRhBound{}:
          same_origin

type PxRespHeader is Data:
  PxRpPlain{}
  PxRpFraming{}
  PxRpCors{}
  PxRpState{}
  PxRpPolicy{}

# kind: PxRpFraming = connection and message framing; PxRpCors = cross-origin grants of the
# target (they speak for another origin); PxRpState = cookies (they would be set on this server's
# origin); PxRpPolicy = redirect target, server name and page policies of the target
def px_resp_header_passes(kind: PxRespHeader) -> Bool:
  match kind:
    case PxRpPlain{}:
      True{}
    case _:
      False{}

# LAW 8.1: A header without the prefix is never forwarded (the server's own key stays here).
law px_unprefixed_never_forwarded:
  for k: PxReqHeader
  for o: Bool
  {px_req_header_goes(False{}, k, o) == False{} : Bool}

def px_unprefixed_never_forwarded(k, o):
  {==}

# LAW 8.2: A framing header is never forwarded.
law px_framing_never_forwarded:
  for +p: Bool
  for o: Bool
  {px_req_header_goes(p, PxRhFraming{}, o) == False{} : Bool}

def px_framing_never_forwarded(p, o):
  match p:
    case True{}: {==}
    case False{}: {==}

# LAW 8.3: A header that may carry a credential never follows a redirect to another origin.
law px_bound_header_stays_with_origin:
  for +p: Bool
  {px_req_header_goes(p, PxRhBound{}, False{}) == False{} : Bool}

def px_bound_header_stays_with_origin(p):
  match p:
    case True{}: {==}
    case False{}: {==}

# LAW 8.4: The target's cookies never reach the page.
law px_resp_state_dropped:
  {px_resp_header_passes(PxRpState{}) == False{} : Bool}

def px_resp_state_dropped():
  {==}

# LAW 8.5: The target's cross-origin grants never reach the page.
law px_resp_cors_dropped:
  {px_resp_header_passes(PxRpCors{}) == False{} : Bool}

def px_resp_cors_dropped():
  {==}

# LAW 8.6: The target's framing headers never reach the page (the proxy frames its own answer).
law px_resp_framing_dropped:
  {px_resp_header_passes(PxRpFraming{}) == False{} : Bool}

def px_resp_framing_dropped():
  {==}

# ============================================================================
# 9. Size and Time Caps
# ============================================================================

# A megabyte count as bytes (saturating).
def px_cap_bytes(mb: Nat) -> Nat:
  Nat.mul(Nat.mul(mb, 1024n), 1024n)

# A cap the operator may set: at least one unit, at most `most`.
def px_cap_ok(+value: Nat, most: Nat) -> Bool:
  Bool.and(Nat.is_gt(value, 0n), Nat.is_le(value, most))

# A request body of `len` bytes is within `cap`.
def px_body_fits(len: Nat, cap: Nat) -> Bool:
  Nat.is_le(len, cap)

# Of `incoming` response bytes, with `sent` already forwarded, how many may still go out.
def px_resp_take(sent: Nat, incoming: Nat, cap: Nat) -> Nat:
  Nat.min(incoming, Nat.sub(cap, sent))

# The response is longer than the cap: it is cut after px_resp_take bytes.
def px_resp_cut(sent: Nat, incoming: Nat, cap: Nat) -> Bool:
  Nat.is_gt(incoming, Nat.sub(cap, sent))

# Time left of a limit (zero once it has passed).
def px_time_left(elapsed: Nat, limit: Nat) -> Nat:
  Nat.sub(limit, elapsed)

def px_expired(elapsed: Nat, limit: Nat) -> Bool:
  Nat.is_ge(elapsed, limit)

# How long one read from the target may block: no longer than what is left of the idle limit and
# of the total limit, and no longer than `tick` (the interval at which the proxy looks up).
def px_read_wait(idle_left: Nat, total_left: Nat, tick: Nat) -> Nat:
  Nat.min(tick, Nat.min(idle_left, total_left))

# An event stream that has been quiet for `quiet` gets a comment line every `every` (a front end
# between the page and this server may cut a connection that carries nothing). A comment may
# only start where a line starts, and only an event stream has comments.
def px_keepalive_due(is_event_stream: Bool, at_line_start: Bool, +quiet: Nat, +every: Nat) -> Bool:
  Bool.and(is_event_stream, Bool.and(at_line_start, Bool.and(Nat.is_gt(every, 0n), Nat.is_ge(quiet, every))))

# How long the proxy may wait before it looks whether a keep-alive line is due: what is left of
# the interval where a line could be written, the whole interval elsewhere.
def px_keepalive_wait(is_event_stream: Bool, at_line_start: Bool, quiet: Nat, +every: Nat) -> Nat:
  match is_event_stream at_line_start:
    case True{} True{}:
      Nat.min(every, Nat.sub(every, quiet))
    case _ _:
      every

# LAW 9.1: A cap of zero is not accepted (it would be read as "no limit" or as "nothing").
law px_cap_zero_refused:
  for m: Nat
  {px_cap_ok(0n, m) == False{} : Bool}

def px_cap_zero_refused(m):
  {==}

# LAW 9.2: Never more bytes are forwarded than arrived.
law px_resp_take_within_incoming:
  for +s: Nat
  for +n: Nat
  for +c: Nat
  {Nat.is_le(px_resp_take(s, n, c), n) == True{} : Bool}

def px_resp_take_within_incoming(s, n, c):
  px_min_le_l(n, Nat.sub(c, s))

# LAW 9.3: Never more bytes are forwarded than the cap leaves room for.
law px_resp_take_within_cap:
  for +s: Nat
  for +n: Nat
  for +c: Nat
  {Nat.is_le(px_resp_take(s, n, c), Nat.sub(c, s)) == True{} : Bool}

def px_resp_take_within_cap(s, n, c):
  px_min_le_r(n, Nat.sub(c, s))

# LAW 9.4: A read never waits longer than the tick.
law px_read_wait_within_tick:
  for +i: Nat
  for +t: Nat
  for +k: Nat
  {Nat.is_le(px_read_wait(i, t, k), k) == True{} : Bool}

def px_read_wait_within_tick(i, t, k):
  px_min_le_l(k, Nat.min(i, t))

# LAW 9.5: A read never waits past the nearer of the two limits.
law px_read_wait_within_limits:
  for +i: Nat
  for +t: Nat
  for +k: Nat
  {Nat.is_le(px_read_wait(i, t, k), Nat.min(i, t)) == True{} : Bool}

def px_read_wait_within_limits(i, t, k):
  px_min_le_r(k, Nat.min(i, t))

# LAW 9.6: Only an event stream gets a keep-alive line.
law px_keepalive_only_event_stream:
  for l: Bool
  for q: Nat
  for e: Nat
  {px_keepalive_due(False{}, l, q, e) == False{} : Bool}

def px_keepalive_only_event_stream(l, q, e):
  {==}

# LAW 9.7: A keep-alive line is never written in the middle of a line.
law px_keepalive_never_mid_line:
  for +s: Bool
  for q: Nat
  for e: Nat
  {px_keepalive_due(s, False{}, q, e) == False{} : Bool}

def px_keepalive_never_mid_line(s, q, e):
  match s:
    case True{}: {==}
    case False{}: {==}

# LAW 9.8: The wait for the next keep-alive check is never longer than the interval.
law px_keepalive_wait_within_interval:
  for +s: Bool
  for +l: Bool
  for +q: Nat
  for +e: Nat
  {Nat.is_le(px_keepalive_wait(s, l, q, e), e) == True{} : Bool}

def px_keepalive_wait_within_interval(s, l, q, e):
  match s l:
    case True{} True{}:
      px_min_le_l(e, Nat.sub(e, q))
    case True{} False{}:
      px_le_refl(e)
    case False{} True{}:
      px_le_refl(e)
    case False{} False{}:
      px_le_refl(e)
