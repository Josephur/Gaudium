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

# The hosts a request names are few enough to be read: the front-end headers name at most
# max_hosts hosts, and the names the request carries (its Host included) hold at most max_bytes
# bytes between them. The Rust shell builds the lists hn_request_ours reads only for a request
# that fits, so a request naming thousands of hosts is refused before any is read.
# hosts: the number of hosts the front-end headers name
# bytes: the bytes of every name the request carries
def hn_hosts_fit(hosts: Nat, bytes: Nat, max_hosts: Nat, max_bytes: Nat) -> Bool:
  Bool.and(Nat.is_le(hosts, max_hosts), Nat.is_le(bytes, max_bytes))

# fits: hn_hosts_fit of the request
# ours: hn_request_ours of the request (of nothing it names, when it does not fit)
def hn_hosts_ours(fits: Bool, ours: Bool) -> Bool:
  Bool.and(fits, ours)

# Two names joined. No decision uses it: the laws of section 7 build a longer name, or two names
# with the same start, with it.
def hn_name_cat(a: List<&2, Nat>, b: List<&2, Nat>) -> List<&2, Nat>:
  match a:
    case Nil{}:
      b
    case Con{x, xs}:
      Con{x, hn_name_cat(xs, b)}

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

def hn_and_true_l(+a: Bool, b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      L.bool_absurd({False{} == True{} : Bool}, h)

def hn_and_true_r(+a: Bool, b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case True{}:
      h
    case False{}:
      L.bool_absurd({b == True{} : Bool}, h)

def hn_ne_of_lt(+a: Nat, +b: Nat, h: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_eq(a, b) == False{} : Bool}:
  match a b:
    case 0n 0n:
      L.bool_absurd({Nat.is_eq(0n, 0n) == False{} : Bool}, h)
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      L.bool_absurd({Nat.is_eq(1n+ap, 0n) == False{} : Bool}, h)
    case 1n+ap 1n+bp:
      hn_ne_of_lt(ap, bp, h)

def hn_ne_of_gt(+a: Nat, +b: Nat, h: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_eq(b, a) == False{} : Bool}:
  match a b:
    case 0n 0n:
      L.bool_absurd({Nat.is_eq(0n, 0n) == False{} : Bool}, h)
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      L.bool_absurd({Nat.is_eq(0n, 1n+ap) == False{} : Bool}, h)
    case 1n+ap 1n+bp:
      hn_ne_of_gt(ap, bp, h)

# A name never compares the same as itself with more bytes after it.
def hn_same_prefix(+n: List<&2, Nat>, +y: Nat, +ys: List<&2, Nat>) -> {hn_same(n, hn_name_cat(n, Con{y, ys})) == False{} : Bool}:
  match n:
    case Nil{}:
      {==}
    case Con{+x, +xs}:
      %Equal.sym(Bool, Nat.is_eq(hn_fold(x), hn_fold(x)), True{}, G.g_nat_eq_refl(hn_fold(x))) : {Bool.and(_, hn_same(xs, hn_name_cat(xs, Con{y, ys}))) == False{} : Bool}
      hn_same_prefix(xs, y, ys)

# A name with more bytes after it never compares the same as the name alone.
def hn_same_extended(+m: List<&2, Nat>, +y: Nat, +ys: List<&2, Nat>) -> {hn_same(hn_name_cat(m, Con{y, ys}), m) == False{} : Bool}:
  match m:
    case Nil{}:
      {==}
    case Con{+x, +xs}:
      %Equal.sym(Bool, Nat.is_eq(hn_fold(x), hn_fold(x)), True{}, G.g_nat_eq_refl(hn_fold(x))) : {Bool.and(_, hn_same(hn_name_cat(xs, Con{y, ys}), xs)) == False{} : Bool}
      hn_same_extended(xs, y, ys)

# Two names with the same start never compare the same when the next bytes differ once folded.
def hn_same_split(
  +p: List<&2, Nat>,
  +a: Nat,
  +b: Nat,
  +xs: List<&2, Nat>,
  +ys: List<&2, Nat>,
  ok: {Nat.is_eq(hn_fold(a), hn_fold(b)) == False{} : Bool}
) -> {hn_same(hn_name_cat(p, Con{a, xs}), hn_name_cat(p, Con{b, ys})) == False{} : Bool}:
  match p:
    case Nil{}:
      %Equal.sym(Bool, Nat.is_eq(hn_fold(a), hn_fold(b)), False{}, ok) : {Bool.and(_, hn_same(xs, ys)) == False{} : Bool}
      {==}
    case Con{+x, +ps}:
      %Equal.sym(Bool, Nat.is_eq(hn_fold(x), hn_fold(x)), True{}, G.g_nat_eq_refl(hn_fold(x))) : {Bool.and(_, hn_same(hn_name_cat(ps, Con{a, xs}), hn_name_cat(ps, Con{b, ys}))) == False{} : Bool}
      hn_same_split(ps, a, b, xs, ys, ok)

# Two names that compare the same never compare the same once each has a next byte and those
# bytes differ once folded.
def hn_same_split_same(
  +p: List<&2, Nat>,
  +q: List<&2, Nat>,
  +a: Nat,
  +b: Nat,
  +xs: List<&2, Nat>,
  +ys: List<&2, Nat>,
  +okp: {hn_same(p, q) == True{} : Bool},
  ok: {Nat.is_eq(hn_fold(a), hn_fold(b)) == False{} : Bool}
) -> {hn_same(hn_name_cat(p, Con{a, xs}), hn_name_cat(q, Con{b, ys})) == False{} : Bool}:
  match p q:
    case Nil{} Nil{}:
      %Equal.sym(Bool, Nat.is_eq(hn_fold(a), hn_fold(b)), False{}, ok) : {Bool.and(_, hn_same(xs, ys)) == False{} : Bool}
      {==}
    case Nil{} Con{+y, +qs}:
      L.bool_absurd({hn_same(Con{a, xs}, Con{y, hn_name_cat(qs, Con{b, ys})}) == False{} : Bool}, okp)
    case Con{+x, +ps} Nil{}:
      L.bool_absurd({hn_same(Con{x, hn_name_cat(ps, Con{a, xs})}, Con{b, ys}) == False{} : Bool}, okp)
    case Con{+x, +ps} Con{+y, +qs}:
      +hx = hn_and_true_l(Nat.is_eq(hn_fold(x), hn_fold(y)), hn_same(ps, qs), okp)
      +hr = hn_and_true_r(Nat.is_eq(hn_fold(x), hn_fold(y)), hn_same(ps, qs), okp)
      %Equal.sym(Bool, Nat.is_eq(hn_fold(x), hn_fold(y)), True{}, hx) : {Bool.and(_, hn_same(hn_name_cat(ps, Con{a, xs}), hn_name_cat(qs, Con{b, ys}))) == False{} : Bool}
      hn_same_split_same(ps, qs, a, b, xs, ys, hr, ok)

# The verdict on a request whose Host is n, with m as the only name, is hn_same(n, m).
def hn_one_name_true(+c: Bool, h: {Bool.and(Bool.or(c, False{}), True{}) == True{} : Bool}) -> {c == True{} : Bool}:
  match c:
    case True{}:
      {==}
    case False{}:
      L.bool_absurd({False{} == True{} : Bool}, h)

def hn_one_name_false(+c: Bool, h: {Bool.and(Bool.or(c, False{}), True{}) == False{} : Bool}) -> {c == False{} : Bool}:
  match c:
    case True{}:
      L.bool_absurd_tf({True{} == False{} : Bool}, h)
    case False{}:
      {==}

def hn_low_ne_aux(
  +x: Nat,
  +y: Nat,
  c: Bool,
  e: {c == Bool.and(Nat.is_le(65n, y), Nat.is_le(y, 90n)) : Bool},
  okx: {Nat.is_lt(x, 65n) == True{} : Bool},
  oky: {Nat.is_eq(x, y) == False{} : Bool}
) -> {Nat.is_eq(x, Bool.pick(Nat, c, Nat.add(32n, y), y)) == False{} : Bool}:
  match c:
    case True{}:
      +hy = hn_and_true_l(Nat.is_le(65n, y), Nat.is_le(y, 90n), Equal.sym(Bool, True{}, Bool.and(Nat.is_le(65n, y), Nat.is_le(y, 90n)), e))
      +hxy = L.nat_lt_le_trans(x, 65n, y, okx, hy)
      hn_ne_of_lt(x, Nat.add(32n, y), L.nat_lt_le_trans(x, y, Nat.add(32n, y), hxy, L.nat_add_le_l(32n, y)))
    case False{}:
      oky

# A byte below the capitals compares the same only as itself.
def hn_fold_low_ne(
  +x: Nat,
  +y: Nat,
  +okx: {Nat.is_lt(x, 65n) == True{} : Bool},
  oky: {Nat.is_eq(x, y) == False{} : Bool}
) -> {Nat.is_eq(hn_fold(x), hn_fold(y)) == False{} : Bool}:
  %Equal.sym(Bool, Nat.is_le(65n, x), False{}, L.nat_lt_le_false(x, 65n, okx)) : {Nat.is_eq(Bool.pick(Nat, Bool.and(_, Nat.is_le(x, 90n)), Nat.add(32n, x), x), hn_fold(y)) == False{} : Bool}
  hn_low_ne_aux(x, y, Bool.and(Nat.is_le(65n, y), Nat.is_le(y, 90n)), {==}, okx, oky)

def hn_high_ne_aux(
  +b: Nat,
  +y: Nat,
  c: Bool,
  e: {c == Bool.and(Nat.is_le(65n, y), Nat.is_le(y, 90n)) : Bool},
  oky: {Nat.is_eq(Nat.add(123n, b), y) == False{} : Bool}
) -> {Nat.is_eq(Nat.add(123n, b), Bool.pick(Nat, c, Nat.add(32n, y), y)) == False{} : Bool}:
  match c:
    case True{}:
      +hy = hn_and_true_r(Nat.is_le(65n, y), Nat.is_le(y, 90n), Equal.sym(Bool, True{}, Bool.and(Nat.is_le(65n, y), Nat.is_le(y, 90n)), e))
      +h91 = Equal.sym(Bool, Nat.is_lt(y, 91n), Nat.is_le(y, 90n), L.nat_lt_succ_eq_le(y, 90n))
      +hlt = L.nat_lt_le_trans(y, 91n, Nat.add(91n, b), Equal.trans(Bool, Nat.is_lt(y, 91n), Nat.is_le(y, 90n), True{}, L.nat_lt_succ_eq_le(y, 90n), hy), L.nat_add_le_r(91n, b))
      hn_ne_of_gt(y, Nat.add(91n, b), hlt)
    case False{}:
      oky

# A byte above the small letters (every non-ASCII byte among them) compares the same only as itself.
def hn_fold_high_ne(
  +b: Nat,
  +y: Nat,
  oky: {Nat.is_eq(Nat.add(123n, b), y) == False{} : Bool}
) -> {Nat.is_eq(hn_fold(Nat.add(123n, b)), hn_fold(y)) == False{} : Bool}:
  hn_high_ne_aux(b, y, Bool.and(Nat.is_le(65n, y), Nat.is_le(y, 90n)), {==}, oky)

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

# ============================================================================
# 6. Laws: How Many Hosts a Request May Name
# ============================================================================

# promise: a request whose front-end headers name more hosts than the bound never fits
# catches: the count of hosts not compared (only the bytes bounded)
law hn_too_many_hosts_refused:
  for +m: Nat
  for +d: Nat
  for b: Nat
  for mb: Nat
  {hn_hosts_fit(Nat.add(m, 1n+d), b, m, mb) == False{} : Bool}

def hn_too_many_hosts_refused(m, d, b, mb):
  %Equal.sym(Bool, Nat.is_le(Nat.add(m, 1n+d), m), False{}, G.g_nat_le_add_succ_false(m, d)) : {Bool.and(_, Nat.is_le(b, mb)) == False{} : Bool}
  {==}

# promise: a request whose names hold more bytes than the bound never fits
# catches: the bytes not compared (only the count of hosts bounded)
law hn_too_many_bytes_refused:
  for +h: Nat
  for +mh: Nat
  for +m: Nat
  for +d: Nat
  {hn_hosts_fit(h, Nat.add(m, 1n+d), mh, m) == False{} : Bool}

def hn_too_many_bytes_refused(h, mh, m, d):
  %Equal.sym(Bool, Nat.is_le(Nat.add(m, 1n+d), m), False{}, G.g_nat_le_add_succ_false(m, d)) : {Bool.and(Nat.is_le(h, mh), _) == False{} : Bool}
  G.g_bool_and_false_r(Nat.is_le(h, mh))

# promise: a request inside both bounds fits
# catches: the bound on the hosts made strict (Nat.is_lt in place of Nat.is_le)
law hn_within_bounds_fits:
  for +h: Nat
  for +mh: Nat
  for +b: Nat
  for +mb: Nat
  for okh: {Nat.is_le(h, mh) == True{} : Bool}
  for okb: {Nat.is_le(b, mb) == True{} : Bool}
  {hn_hosts_fit(h, b, mh, mb) == True{} : Bool}

def hn_within_bounds_fits(h, mh, b, mb, okh, okb):
  %Equal.sym(Bool, Nat.is_le(h, mh), True{}, okh) : {Bool.and(_, Nat.is_le(b, mb)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_le(b, mb), True{}, okb) : {Bool.and(True{}, _) == True{} : Bool}
  {==}

# promise: a request that does not fit the bounds never names this server
# catches: the bound ignored (hn_hosts_ours returns ours)
law hn_unfit_refused:
  for o: Bool
  {hn_hosts_ours(False{}, o) == False{} : Bool}

def hn_unfit_refused(o):
  {==}

# promise: fitting the bounds never lets through a request that does not name this server
# catches: the host check ignored (hn_hosts_ours returns fits)
law hn_fit_never_widens:
  for +f: Bool
  {hn_hosts_ours(f, False{}) == False{} : Bool}

def hn_fit_never_widens(f):
  G.g_bool_and_false_r(f)

# promise: a request that fits the bounds and names this server names this server
# catches: every request refused (hn_hosts_ours always False{})
law hn_fit_own_hosts_pass:
  {hn_hosts_ours(True{}, True{}) == True{} : Bool}

def hn_fit_own_hosts_pass():
  {==}

# ============================================================================
# 7. Laws: Names Compared Through the Request Verdict
# ============================================================================
# The laws of section 3 constrain hn_fold and hn_same, which the Rust shell reaches only through
# hn_request_ours. The laws below state the same promises on hn_request_ours itself. Each says
# that listing a name that is not the requested name, by a length, a byte or a case rule, adds
# nothing: the request passes or not as it would without that name, whatever else is listed.
#
#   section 3 law                     on hn_request_ours
#   hn_fold_capitals                  hn_name_case_ignored (section 4, sealed: every capital A..Z)
#   hn_fold_last_capital              hn_last_capital_passes
#   hn_fold_keeps_low                 hn_low_byte_adds_nothing, hn_at_sign_adds_nothing
#   hn_fold_keeps_high                hn_high_byte_adds_nothing, hn_left_brace_adds_nothing,
#                                     hn_non_ascii_adds_nothing
#   hn_same_shorter                   hn_longer_listed_name_adds_nothing
#   hn_same_longer                    hn_longer_host_name_adds_nothing
#   hn_same_byte_differs              hn_differing_byte_adds_nothing (after starts that compare the same)
#   hn_same_rest_differs              hn_differing_byte_adds_nothing (starts that compare the same,
#                                     a capital against its small letter among them),
#                                     hn_case_start_rest_adds_nothing (a capital, then any rest)

# promise: a listed name that is the requested name with more bytes after it never lets the request through
# catches: an empty name taken as a match for any name (hn_same of Nil and Con gives True{})
law hn_longer_listed_name_adds_nothing:
  for +n: List<&2, Nat>
  for +y: Nat
  for +ys: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnName{n}, Nil{}, Con{hn_name_cat(n, Con{y, ys}), ns}) == hn_request_ours(HnName{n}, Nil{}, ns) : Bool}

def hn_longer_listed_name_adds_nothing(n, y, ys, ns):
  hn_other_name_adds_nothing(n, hn_name_cat(n, Con{y, ys}), ns, hn_same_prefix(n, y, ys))

# promise: a requested name that is a listed name with more bytes after it (one of its names as the start of another domain) never passes by that name
# catches: a longer name taken as matching a listed name it starts with (hn_same of Con and Nil gives True{})
law hn_longer_host_name_adds_nothing:
  for +m: List<&2, Nat>
  for +y: Nat
  for +ys: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnName{hn_name_cat(m, Con{y, ys})}, Nil{}, Con{m, ns}) == hn_request_ours(HnName{hn_name_cat(m, Con{y, ys})}, Nil{}, ns) : Bool}

def hn_longer_host_name_adds_nothing(m, y, ys, ns):
  hn_other_name_adds_nothing(hn_name_cat(m, Con{y, ys}), m, ns, hn_same_extended(m, y, ys))

# promise: a listed name whose start compares the same as the requested name's start, but whose next byte differs once case is set aside, never lets the request through, whatever follows
# catches: only the first bytes compared (the rest of the names ignored in hn_same)
law hn_differing_byte_adds_nothing:
  for +p: List<&2, Nat>
  for +q: List<&2, Nat>
  for +a: Nat
  for +b: Nat
  for +xs: List<&2, Nat>
  for +ys: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  for okp: {hn_request_ours(HnName{p}, Nil{}, Con{q, Nil{}}) == True{} : Bool}
  for ok: {Nat.is_eq(hn_fold(a), hn_fold(b)) == False{} : Bool}
  {hn_request_ours(HnName{hn_name_cat(p, Con{a, xs})}, Nil{}, Con{hn_name_cat(q, Con{b, ys}), ns}) == hn_request_ours(HnName{hn_name_cat(p, Con{a, xs})}, Nil{}, ns) : Bool}

def hn_differing_byte_adds_nothing(p, q, a, b, xs, ys, ns, okp, ok):
  hn_other_name_adds_nothing(hn_name_cat(p, Con{a, xs}), hn_name_cat(q, Con{b, ys}), ns, hn_same_split_same(p, q, a, b, xs, ys, hn_one_name_true(hn_same(p, q), okp), ok))

# promise: a listed name whose first byte is the small letter of the requested name's capital, but whose rest differs from the requested name's rest, never lets the request through
# catches: a first byte that differs only in case taken as a match without comparing the rest (hn_same compares the rest only after identical bytes)
law hn_case_start_rest_adds_nothing:
  for +i: Nat
  for ok: {Nat.is_lt(i, 26n) == True{} : Bool}
  for +xs: List<&2, Nat>
  for +ys: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  for okr: {hn_request_ours(HnName{xs}, Nil{}, Con{ys, Nil{}}) == False{} : Bool}
  {hn_request_ours(HnName{Con{Nat.add(65n, i), xs}}, Nil{}, Con{Con{Nat.add(97n, i), ys}, ns}) == hn_request_ours(HnName{Con{Nat.add(65n, i), xs}}, Nil{}, ns) : Bool}

def hn_case_start_rest_adds_nothing(i, ok, xs, ys, ns, okr):
  hn_other_name_adds_nothing(Con{Nat.add(65n, i), xs}, Con{Nat.add(97n, i), ys}, ns, hn_same_rest_differs(Nat.add(65n, i), Nat.add(97n, i), xs, ys, hn_one_name_false(hn_same(xs, ys), okr)))

# promise: a byte below the capitals in the requested name is matched only by the same byte in a listed name
# catches: the first bytes not compared (Bool.or in place of Bool.and in hn_same)
law hn_low_byte_adds_nothing:
  for +p: List<&2, Nat>
  for +x: Nat
  for +y: Nat
  for +xs: List<&2, Nat>
  for +ys: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  for okx: {Nat.is_lt(x, 65n) == True{} : Bool}
  for oky: {Nat.is_eq(x, y) == False{} : Bool}
  {hn_request_ours(HnName{hn_name_cat(p, Con{x, xs})}, Nil{}, Con{hn_name_cat(p, Con{y, ys}), ns}) == hn_request_ours(HnName{hn_name_cat(p, Con{x, xs})}, Nil{}, ns) : Bool}

def hn_low_byte_adds_nothing(p, x, y, xs, ys, ns, okx, oky):
  hn_differing_byte_adds_nothing(p, p, x, y, xs, ys, ns, hn_listed_name_passes(p, Nil{}), hn_fold_low_ne(x, y, okx, oky))

# promise: a byte above the small letters in the requested name, every non-ASCII byte among them, is never folded into a match with another byte
# catches: the top bit of a byte dropped before folding (Nat.mod(b, 128n)), so a non-ASCII byte compares as an ASCII one
law hn_high_byte_adds_nothing:
  for +p: List<&2, Nat>
  for +b: Nat
  for +y: Nat
  for +xs: List<&2, Nat>
  for +ys: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  for oky: {Nat.is_eq(Nat.add(123n, b), y) == False{} : Bool}
  {hn_request_ours(HnName{hn_name_cat(p, Con{Nat.add(123n, b), xs})}, Nil{}, Con{hn_name_cat(p, Con{y, ys}), ns}) == hn_request_ours(HnName{hn_name_cat(p, Con{Nat.add(123n, b), xs})}, Nil{}, ns) : Bool}

def hn_high_byte_adds_nothing(p, b, y, xs, ys, ns, oky):
  hn_differing_byte_adds_nothing(p, p, Nat.add(123n, b), y, xs, ys, ns, hn_listed_name_passes(p, Nil{}), hn_fold_high_ne(b, y, oky))

# promise: @, the byte just below the capitals, never matches the byte 32 above it (the sampled bytes of hn_low_byte_adds_nothing stop short of it)
# catches: the lower bound one too low (Nat.is_le(64n, b) in hn_fold)
law hn_at_sign_adds_nothing:
  for +xs: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnName{Con{64n, xs}}, Nil{}, Con{Con{96n, xs}, ns}) == hn_request_ours(HnName{Con{64n, xs}}, Nil{}, ns) : Bool}

def hn_at_sign_adds_nothing(xs, ns):
  {==}

# promise: {, the first byte above the small letters, never matches [, the byte 32 below it (the sampled bytes of hn_high_byte_adds_nothing stop short of it)
# catches: the upper bound one too high (Nat.is_le(b, 91n) in hn_fold)
law hn_left_brace_adds_nothing:
  for +xs: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnName{Con{123n, xs}}, Nil{}, Con{Con{91n, xs}, ns}) == hn_request_ours(HnName{Con{123n, xs}}, Nil{}, ns) : Bool}

def hn_left_brace_adds_nothing(xs, ns):
  {==}

# promise: a non-ASCII byte is never folded: byte 192 never matches byte 224, 32 above it (the sampled bytes of hn_high_byte_adds_nothing stop short of them)
# catches: non-ASCII capitals folded as well (bytes 192..222 also get 32 added in hn_fold)
law hn_non_ascii_adds_nothing:
  for +xs: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnName{Con{192n, xs}}, Nil{}, Con{Con{224n, xs}, ns}) == hn_request_ours(HnName{Con{192n, xs}}, Nil{}, ns) : Bool}

def hn_non_ascii_adds_nothing(xs, ns):
  {==}

# promise: a requested name starting with Z passes as the listed name starting with z (the sampled capitals of hn_name_case_ignored stop short of it)
# catches: the last capital left out (Nat.is_le(b, 89n) in hn_fold)
law hn_last_capital_passes:
  for +xs: List<&2, Nat>
  for +ns: List<&2, List<&2, Nat>>
  {hn_request_ours(HnName{Con{90n, xs}}, Nil{}, Con{Con{122n, xs}, ns}) == True{} : Bool}

def hn_last_capital_passes(xs, ns):
  %Equal.sym(Bool, hn_same(xs, xs), True{}, hn_same_refl(xs)) : {Bool.and(Bool.or(_, hn_listed(ns, Con{90n, xs})), True{}) == True{} : Bool}
  {==}
