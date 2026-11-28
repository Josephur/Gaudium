# Slice — Verified Server Names (the Host check of the proxy routes)
# Formally specifies and proves whether a request names this server:
#   - Each host a request names (its Host header, and the hosts a front end says it received the
#     request for) is an address, a DNS name, or something that is neither
#   - A name passes only when it is one of the server's own names, compared without regard to
#     the case of ASCII letters (DNS names are case-insensitive)
#   - A request passes only when every host it names passes
#   - The site verdict of the MCP proxy (`px_site_ok` in `mcp_proxy.bend`) counts only for a
#     request that names this server
#
# Why: a page on any domain can point its own name at this server's address (DNS rebinding).
# Its requests then carry an origin and a Host that agree with each other, and the browser calls
# them same-origin, so comparing the two says nothing. The Host names the page's domain, though,
# and that is not one of the server's names. An address literal cannot be rebound: a browser
# sends one only to the machine that holds it. Port numbers and the brackets of IPv6 literals
# are taken off by the Rust shell, which also reads the bytes of each name from the request.

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. Comparing Names
# ============================================================================

# One host a request names.
type HnHost is Data:
  HnAddr{}
  HnName{name: List<&2, Nat>}
  HnBad{}

# A byte of a name as it is compared: ASCII capitals as small letters, every other byte as it is.
def hn_fold(+b: Nat) -> Nat:
  Bool.pick(Nat, Bool.and(Nat.is_le(65n, b), Nat.is_le(b, 90n)), Nat.add(32n, b), b)

# Two names are the same name: as long as each other, and byte for byte the same once folded.
def hn_same(a: List<&2, Nat>, b: List<&2, Nat>) -> Bool:
  match a:
    case Nil{}:
      match b:
        case Nil{}:
          True{}
        case Con{y, ys}:
          False{}
    case Con{x, xs}:
      match b:
        case Nil{}:
          False{}
        case Con{y, ys}:
          Bool.and(Nat.is_eq(hn_fold(x), hn_fold(y)), hn_same(xs, ys))

# ns: the server's own names; n: a name the request carries
def hn_listed(ns: List<&2, List<&2, Nat>>, +n: List<&2, Nat>) -> Bool:
  match ns:
    case Nil{}:
      False{}
    case Con{m, rest}:
      Bool.or(hn_same(n, m), hn_listed(rest, n))

def hn_host_ours(h: HnHost, ns: List<&2, List<&2, Nat>>) -> Bool:
  match h:
    case HnAddr{}:
      True{}
    case HnName{name}:
      hn_listed(ns, name)
    case HnBad{}:
      False{}

def hn_all_ours(hs: List<&2, HnHost>, +ns: List<&2, List<&2, Nat>>) -> Bool:
  match hs:
    case Nil{}:
      True{}
    case Con{h, rest}:
      Bool.and(hn_host_ours(h, ns), hn_all_ours(rest, ns))

# host: the request's Host header (HnBad{} when it has none)
# fwd:  the hosts a front end names (X-Forwarded-Host, Forwarded host=)
# ns:   the server's own names
def hn_request_ours(host: HnHost, fwd: List<&2, HnHost>, +ns: List<&2, List<&2, Nat>>) -> Bool:
  Bool.and(hn_host_ours(host, ns), hn_all_ours(fwd, ns))

# hosts_ours: hn_request_ours of the request
# site_ok:    px_site_ok of the request
def hn_site_ok(hosts_ours: Bool, site_ok: Bool) -> Bool:
  Bool.and(hosts_ours, site_ok)

# ============================================================================
# 2. Proof Helpers (erased before Rust codegen)
# ============================================================================

def hn_lt26(+i: Nat, ok: {Nat.is_lt(i, 26n) == True{} : Bool}) -> {Nat.is_le(i, 25n) == True{} : Bool}:
  %L.nat_lt_succ_eq_le(i, 25n) : {_ == True{} : Bool}
  ok

def hn_same_refl(+a: List<&2, Nat>) -> {hn_same(a, a) == True{} : Bool}:
  match a:
    case Nil{}:
      {==}
    case Con{+x, +xs}:
      %Equal.sym(Bool, Nat.is_eq(hn_fold(x), hn_fold(x)), True{}, G.g_nat_eq_refl(hn_fold(x))) : {Bool.and(_, hn_same(xs, xs)) == True{} : Bool}
      %Equal.sym(Bool, hn_same(xs, xs), True{}, hn_same_refl(xs)) : {Bool.and(True{}, _) == True{} : Bool}
      {==}

def hn_and_mid(+a: Bool, c: Bool, +d: Bool, h: {Bool.and(a, d) == False{} : Bool}) -> {Bool.and(a, Bool.and(c, d)) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, d, False{}, h) : {Bool.and(c, _) == False{} : Bool}
      G.g_bool_and_false_r(c)

# ============================================================================
# 3. Laws: the Bytes of a Name
# ============================================================================

# promise: an ASCII capital letter compares as its small letter
# catches: the first capital left out (Nat.is_le(66n, b))
law hn_fold_capitals:
  for +i: Nat
  for ok: {Nat.is_lt(i, 26n) == True{} : Bool}
  {hn_fold(Nat.add(65n, i)) == Nat.add(97n, i) : Nat}

def hn_fold_capitals(i, ok):
  %Equal.sym(Bool, Nat.is_le(0n, i), True{}, L.nat_le_zero(i)) : {Bool.pick(Nat, Bool.and(_, Nat.is_le(i, 25n)), Nat.add(32n, Nat.add(65n, i)), Nat.add(65n, i)) == Nat.add(97n, i) : Nat}
  %Equal.sym(Bool, Nat.is_le(i, 25n), True{}, hn_lt26(i, ok)) : {Bool.pick(Nat, Bool.and(True{}, _), Nat.add(32n, Nat.add(65n, i)), Nat.add(65n, i)) == Nat.add(97n, i) : Nat}
  {==}

# promise: the last capital, Z, compares as z (the sampled capitals of hn_fold_capitals stop short of it)
# catches: the last capital left out (Nat.is_le(b, 89n))
law hn_fold_last_capital:
  {hn_fold(90n) == 122n : Nat}

def hn_fold_last_capital():
  {==}

# promise: a byte above the capitals is compared as it is
# catches: every byte from 65 up folded (the upper bound dropped)
law hn_fold_keeps_high:
  for b: Nat
  {hn_fold(Nat.add(91n, b)) == Nat.add(91n, b) : Nat}

def hn_fold_keeps_high(b):
  {==}

# promise: a byte below the capitals is compared as it is
# catches: the lower bound one too low (Nat.is_le(64n, b))
law hn_fold_keeps_low:
  for +b: Nat
  for ok: {Nat.is_lt(b, 65n) == True{} : Bool}
  {hn_fold(b) == b : Nat}

def hn_fold_keeps_low(b, ok):
  %Equal.sym(Bool, Nat.is_le(65n, b), False{}, L.nat_lt_le_false(b, 65n, ok)) : {Bool.pick(Nat, Bool.and(_, Nat.is_le(b, 90n)), Nat.add(32n, b), b) == b : Nat}
  {==}

# promise: a name never compares the same as a longer name
# catches: an empty name taken as a match for any name
law hn_same_shorter:
  for y: Nat
  for ys: List<&2, Nat>
  {hn_same(Nil{}, Con{y, ys}) == False{} : Bool}

def hn_same_shorter(y, ys):
  {==}

# promise: a name never compares the same as a shorter name
# catches: a longer name taken as matching a listed name it starts with (Con against Nil gives True)
law hn_same_longer:
  for x: Nat
  for xs: List<&2, Nat>
  {hn_same(Con{x, xs}, Nil{}) == False{} : Bool}

def hn_same_longer(x, xs):
  {==}

# promise: two names whose bytes differ, once case is set aside, are not the same name
# catches: the first bytes not compared (Bool.or in place of Bool.and)
law hn_same_byte_differs:
  for +x: Nat
  for +y: Nat
  for +xs: List<&2, Nat>
  for +ys: List<&2, Nat>
  for ok: {Nat.is_eq(hn_fold(x), hn_fold(y)) == False{} : Bool}
  {hn_same(Con{x, xs}, Con{y, ys}) == False{} : Bool}

def hn_same_byte_differs(x, y, xs, ys, ok):
  %Equal.sym(Bool, Nat.is_eq(hn_fold(x), hn_fold(y)), False{}, ok) : {Bool.and(_, hn_same(xs, ys)) == False{} : Bool}
  {==}

# promise: two names that differ after their first byte are not the same name
# catches: only the first bytes compared (the rest of the names ignored)
law hn_same_rest_differs:
  for +x: Nat
  for +y: Nat
  for +xs: List<&2, Nat>
  for +ys: List<&2, Nat>
  for ok: {hn_same(xs, ys) == False{} : Bool}
  {hn_same(Con{x, xs}, Con{y, ys}) == False{} : Bool}

def hn_same_rest_differs(x, y, xs, ys, ok):
  %Equal.sym(Bool, hn_same(xs, ys), False{}, ok) : {Bool.and(Nat.is_eq(hn_fold(x), hn_fold(y)), _) == False{} : Bool}
  G.g_bool_and_false_r(Nat.is_eq(hn_fold(x), hn_fold(y)))

# ============================================================================
# 4. Laws: One Request
# ============================================================================

# promise: a request whose Host is neither an address nor a name never names this server
# catches: an unreadable Host let through (HnBad{} gives True{})
law hn_bad_host_refused:
  for fs: List<&2, HnHost>
  for ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnBad{}, fs, ns) == False{} : Bool}

def hn_bad_host_refused(fs, ns):
  {==}

# promise: with no names of its own, no request that names a host by name names this server
# catches: an empty list of names taken as "any name"
law hn_no_names_no_name:
  for n: List<&2, Nat>
  for fs: List<&2, HnHost>
  {hn_request_ours(HnName{n}, fs, Nil{}) == False{} : Bool}

def hn_no_names_no_name(n, fs):
  {==}

# promise: a name of the server's that is not the requested name never lets the request through
# catches: any listed name taken as a match (hn_same dropped from hn_listed)
law hn_other_name_adds_nothing:
  for +n: List<&2, Nat>
  for +m: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  for ok: {hn_same(n, m) == False{} : Bool}
  {hn_request_ours(HnName{n}, Nil{}, Con{m, ns}) == hn_request_ours(HnName{n}, Nil{}, ns) : Bool}

def hn_other_name_adds_nothing(n, m, ns, ok):
  %Equal.sym(Bool, hn_same(n, m), False{}, ok) : {Bool.and(Bool.or(_, hn_listed(ns, n)), True{}) == hn_request_ours(HnName{n}, Nil{}, ns) : Bool}
  {==}

# promise: a request whose Host is one of the server's names, and that names no other host, names this server
# catches: a name required to match every listed name (Bool.and in place of Bool.or in hn_listed)
law hn_listed_name_passes:
  for +n: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnName{n}, Nil{}, Con{n, ns}) == True{} : Bool}

def hn_listed_name_passes(n, ns):
  %Equal.sym(Bool, hn_same(n, n), True{}, hn_same_refl(n)) : {Bool.and(Bool.or(_, hn_listed(ns, n)), True{}) == True{} : Bool}
  {==}

# promise: a name in capitals passes as the same name in small letters
# catches: names compared byte for byte without folding the case
law hn_name_case_ignored:
  for +i: Nat
  for ok: {Nat.is_lt(i, 26n) == True{} : Bool}
  for +xs: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnName{Con{Nat.add(65n, i), xs}}, Nil{}, Con{Con{Nat.add(97n, i), xs}, ns}) == True{} : Bool}

def hn_name_case_ignored(i, ok, xs, ns):
  %Equal.sym(Nat, hn_fold(Nat.add(65n, i)), Nat.add(97n, i), hn_fold_capitals(i, ok)) : {Bool.and(Bool.or(Bool.and(Nat.is_eq(_, hn_fold(Nat.add(97n, i))), hn_same(xs, xs)), hn_listed(ns, Con{Nat.add(65n, i), xs})), True{}) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_eq(i, i), True{}, G.g_nat_eq_refl(i)) : {Bool.and(Bool.or(Bool.and(_, hn_same(xs, xs)), hn_listed(ns, Con{Nat.add(65n, i), xs})), True{}) == True{} : Bool}
  %Equal.sym(Bool, hn_same(xs, xs), True{}, hn_same_refl(xs)) : {Bool.and(Bool.or(Bool.and(True{}, _), hn_listed(ns, Con{Nat.add(65n, i), xs})), True{}) == True{} : Bool}
  {==}

# promise: a request whose Host does not name this server is refused whatever the front-end headers name
# catches: the forwarded hosts alone decide (Bool.or in place of Bool.and in hn_request_ours)
law hn_foreign_host_refused:
  for +h: HnHost
  for +fs: List<&2, HnHost>
  for +ns: List<&2, List<&2, Nat>>
  for ok: {hn_host_ours(h, ns) == False{} : Bool}
  {hn_request_ours(h, fs, ns) == False{} : Bool}

def hn_foreign_host_refused(h, fs, ns, ok):
  %Equal.sym(Bool, hn_host_ours(h, ns), False{}, ok) : {Bool.and(_, hn_all_ours(fs, ns)) == False{} : Bool}
  {==}

# promise: a request that names, in a front-end header, a host that is not this server is refused
# catches: the forwarded hosts ignored (hn_all_ours always True{})
law hn_foreign_forward_refused:
  for +h: HnHost
  for +f: HnHost
  for +fs: List<&2, HnHost>
  for +ns: List<&2, List<&2, Nat>>
  for ok: {hn_host_ours(f, ns) == False{} : Bool}
  {hn_request_ours(h, Con{f, fs}, ns) == False{} : Bool}

def hn_foreign_forward_refused(h, f, fs, ns, ok):
  %Equal.sym(Bool, hn_host_ours(f, ns), False{}, ok) : {Bool.and(hn_host_ours(h, ns), Bool.and(_, hn_all_ours(fs, ns))) == False{} : Bool}
  G.g_bool_and_false_r(hn_host_ours(h, ns))

# promise: naming one more host never lets a refused request through
# catches: only the first forwarded host checked (the rest of the list ignored)
law hn_more_hosts_never_admit:
  for +h: HnHost
  for +f: HnHost
  for +fs: List<&2, HnHost>
  for +ns: List<&2, List<&2, Nat>>
  for ok: {hn_request_ours(h, fs, ns) == False{} : Bool}
  {hn_request_ours(h, Con{f, fs}, ns) == False{} : Bool}

def hn_more_hosts_never_admit(h, f, fs, ns, ok):
  hn_and_mid(hn_host_ours(h, ns), hn_host_ours(f, ns), hn_all_ours(fs, ns), ok)

# ============================================================================
# 5. Laws: the Site Verdict
# ============================================================================

# promise: a request that does not name this server never passes the proxy's site rule
# catches: the Host check dropped (hn_site_ok returns site_ok)
law hn_site_foreign_host_refused:
  for s: Bool
  {hn_site_ok(False{}, s) == False{} : Bool}

def hn_site_foreign_host_refused(s):
  {==}

# promise: naming this server never overturns a refusal of the site rule
# catches: the site verdict dropped (hn_site_ok returns hosts_ours)
law hn_site_never_widens:
  for +h: Bool
  {hn_site_ok(h, False{}) == False{} : Bool}

def hn_site_never_widens(h):
  G.g_bool_and_false_r(h)

# promise: a request that names this server and passes the site rule passes
# catches: every request refused (hn_site_ok always False{})
law hn_site_own_host_passes:
  {hn_site_ok(True{}, True{}) == True{} : Bool}

def hn_site_own_host_passes():
  {==}
