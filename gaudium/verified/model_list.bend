# Slice — Verified Model-List Page
# Formally specifies and proves which entries of the served-model list one page of
# `GET /v1/models` holds, and whether the page says that more follow. The list has `n` entries
# in a fixed order. A request may name an `after_id` cursor (the page starts after that entry)
# and a `before_id` cursor (the page ends before that entry), and a `limit` (the most entries a
# page holds). The Rust shell (`server::models_list_json`) turns the cursors into positions:
#   lo = the position after the `after_id` entry (0 without one)
#   hi = the position of the `before_id` entry (n without one)
# and asks this file, for each position, whether it is on the page. A page with a `before_id`
# cursor and no `after_id` one is read from its end (`back`): it holds the entries just before
# the cursor, as the Messages API pages backwards. Parsing the parameters and looking up the
# cursor ids stay in the shell.

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. The page
# ============================================================================

# Position `i` lies between the two cursors and inside the list.
def ml_in_window(+i: Nat, +n: Nat, +lo: Nat, +hi: Nat) -> Bool:
  Bool.and(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n)))

# Position `i` is close enough to the end the page is read from: within `limit` entries of the
# start (forward) or of the end (`back`) of the window.
def ml_in_reach(+i: Nat, +n: Nat, +lo: Nat, +hi: Nat, +limit: Nat, back: Bool) -> Bool:
  match back:
    case False{}:
      Nat.is_lt(Nat.sub(i, lo), limit)
    case True{}:
      Nat.is_le(Nat.sub(Nat.min(hi, n), i), limit)

# Whether position `i` of the list is on the page.
def ml_on_page(+i: Nat, +n: Nat, +lo: Nat, +hi: Nat, +limit: Nat, back: Bool) -> Bool:
  Bool.and(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, limit, back))

# Whether the window holds more entries than the page (`has_more`). An empty page never claims
# more, so a client that pages on from it always stops.
def ml_more(+n: Nat, +lo: Nat, +hi: Nat, +limit: Nat) -> Bool:
  Bool.and(Nat.is_lt(0n, limit), Nat.is_lt(limit, Nat.sub(Nat.min(hi, n), lo)))

# ============================================================================
# 2. Proof helpers (erased)
# ============================================================================

def ml_and_l(+a: Bool, +b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      L.bool_absurd({False{} == True{} : Bool}, h)

def ml_and_r(+a: Bool, +b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case True{}:
      h
    case False{}:
      L.bool_absurd({b == True{} : Bool}, h)

def ml_and_intro(+a: Bool, +b: Bool, ha: {a == True{} : Bool}, hb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  match a:
    case True{}:
      hb
    case False{}:
      L.bool_absurd({Bool.and(False{}, b) == True{} : Bool}, ha)

# `x < m - lo` gives `lo < m`.
def ml_lt_sub_imp_lt(+m: Nat, +lo: Nat, +x: Nat, h: {Nat.is_lt(x, Nat.sub(m, lo)) == True{} : Bool}) -> {Nat.is_lt(lo, m) == True{} : Bool}:
  match m lo:
    case 0n _:
      G.g_nat_lt_zero_absurd({Nat.is_lt(lo, 0n) == True{} : Bool}, x, h)
    case 1n+mp 0n:
      {==}
    case 1n+mp 1n+lp:
      ml_lt_sub_imp_lt(mp, lp, x, h)

# `(m + 1) - m == 1`.
def ml_sub_succ_self(+m: Nat) -> {Nat.sub(1n+m, m) == 1n : Nat}:
  match m:
    case 0n:
      {==}
    case 1n+mp:
      ml_sub_succ_self(mp)

# `0 < k` gives `1 <= k`.
def ml_pos_le_one(+k: Nat, h: {Nat.is_lt(0n, k) == True{} : Bool}) -> {Nat.is_le(1n, k) == True{} : Bool}:
  match k:
    case 0n:
      L.bool_absurd({Nat.is_le(1n, 0n) == True{} : Bool}, h)
    case 1n+kp:
      L.nat_le_zero(kp)

# `a <= b` gives `c - b <= c - a`.
def ml_sub_le_sub(+c: Nat, +a: Nat, +b: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.is_le(Nat.sub(c, b), Nat.sub(c, a)) == True{} : Bool}:
  match c a b:
    case 0n _ _:
      {==}
    case 1n+cp 0n 0n:
      L.nat_le_refl(1n+cp)
    case 1n+cp 0n 1n+bp:
      L.nat_le_succ_r(Nat.sub(cp, bp), cp, G.g_nat_sub_le_self(cp, bp))
    case 1n+cp 1n+ap 0n:
      L.bool_absurd({Nat.is_le(Nat.sub(1n+cp, 0n), Nat.sub(1n+cp, 1n+ap)) == True{} : Bool}, h)
    case 1n+cp 1n+ap 1n+bp:
      ml_sub_le_sub(cp, ap, bp, h)

# `a <= b` gives `a - c <= b - c`.
def ml_sub_le_sub_r(+a: Nat, +b: Nat, +c: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.is_le(Nat.sub(a, c), Nat.sub(b, c)) == True{} : Bool}:
  match a b c:
    case 0n _ _:
      L.nat_le_zero(Nat.sub(b, c))
    case 1n+ap 0n _:
      L.bool_absurd({Nat.is_le(Nat.sub(1n+ap, c), Nat.sub(0n, c)) == True{} : Bool}, h)
    case 1n+ap 1n+bp 0n:
      h
    case 1n+ap 1n+bp 1n+cp:
      ml_sub_le_sub_r(ap, bp, cp, h)

# `i < m` gives `i - lo < m - lo`, for `lo <= i`.
def ml_sub_lt_sub(+lo: Nat, +i: Nat, +m: Nat, hle: {Nat.is_le(lo, i) == True{} : Bool}, hlt: {Nat.is_lt(i, m) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(i, lo), Nat.sub(m, lo)) == True{} : Bool}:
  match lo i m:
    case 0n _ _:
      %Equal.sym(Nat, Nat.sub(i, 0n), i, L.nat_sub_zero_r(i)) : {Nat.is_lt(_, Nat.sub(m, 0n)) == True{} : Bool}
      %Equal.sym(Nat, Nat.sub(m, 0n), m, L.nat_sub_zero_r(m)) : {Nat.is_lt(i, _) == True{} : Bool}
      hlt
    case 1n+lp 0n _:
      L.bool_absurd({Nat.is_lt(Nat.sub(0n, 1n+lp), Nat.sub(m, 1n+lp)) == True{} : Bool}, hle)
    case 1n+lp 1n+ip 0n:
      L.bool_absurd({Nat.is_lt(Nat.sub(1n+ip, 1n+lp), Nat.sub(0n, 1n+lp)) == True{} : Bool}, hlt)
    case 1n+lp 1n+ip 1n+mp:
      ml_sub_lt_sub(lp, ip, mp, hle, hlt)

# `lo < mp + 1` gives `lo <= mp`.
def ml_lt_succ_le(+lo: Nat, +mp: Nat, h: {Nat.is_lt(lo, 1n+mp) == True{} : Bool}) -> {Nat.is_le(lo, mp) == True{} : Bool}:
  %L.nat_lt_succ_eq_le(lo, mp) : {_ == True{} : Bool}
  h

# The last position of a window ending at `mp + 1` is one entry from its end.
def ml_reach_last(+mp: Nat, +limit: Nat, h: {Nat.is_le(1n, limit) == True{} : Bool}) -> {Nat.is_le(Nat.sub(1n+mp, mp), limit) == True{} : Bool}:
  %Equal.sym(Nat, Nat.sub(1n+mp, mp), 1n, ml_sub_succ_self(mp)) : {Nat.is_le(_, limit) == True{} : Bool}
  h

# The last position of a window `[lo, m)` that is not empty is on a page read from its end,
# for any `limit` of at least one.
def ml_back_last(+m: Nat, +lo: Nat, +limit: Nat, hlt: {Nat.is_lt(lo, m) == True{} : Bool}, hlim: {Nat.is_lt(0n, limit) == True{} : Bool}) -> {Bool.and(Bool.and(Nat.is_le(lo, Nat.sub(m, 1n)), Nat.is_lt(Nat.sub(m, 1n), m)), Nat.is_le(Nat.sub(m, Nat.sub(m, 1n)), limit)) == True{} : Bool}:
  match m:
    case 0n:
      G.g_nat_lt_zero_absurd({Bool.and(Bool.and(Nat.is_le(lo, Nat.sub(0n, 1n)), Nat.is_lt(Nat.sub(0n, 1n), 0n)), Nat.is_le(Nat.sub(0n, Nat.sub(0n, 1n)), limit)) == True{} : Bool}, lo, hlt)
    case 1n+mp:
      %Equal.sym(Nat, Nat.sub(mp, 0n), mp, L.nat_sub_zero_r(mp)) : {Bool.and(Bool.and(Nat.is_le(lo, _), Nat.is_lt(_, 1n+mp)), Nat.is_le(Nat.sub(1n+mp, _), limit)) == True{} : Bool}
      ml_and_intro(Bool.and(Nat.is_le(lo, mp), Nat.is_lt(mp, 1n+mp)), Nat.is_le(Nat.sub(1n+mp, mp), limit), ml_and_intro(Nat.is_le(lo, mp), Nat.is_lt(mp, 1n+mp), ml_lt_succ_le(lo, mp, hlt), L.nat_lt_succ_self(mp)), ml_reach_last(mp, limit, ml_pos_le_one(limit, hlim)))

# `x < m - lo` gives `lo + x < m`.
def ml_lt_sub_add(+m: Nat, +lo: Nat, +x: Nat, h: {Nat.is_lt(x, Nat.sub(m, lo)) == True{} : Bool}) -> {Nat.is_lt(Nat.add(lo, x), m) == True{} : Bool}:
  match m lo:
    case 0n _:
      G.g_nat_lt_zero_absurd({Nat.is_lt(Nat.add(lo, x), 0n) == True{} : Bool}, x, h)
    case 1n+mp 0n:
      h
    case 1n+mp 1n+lp:
      ml_lt_sub_add(mp, lp, x, h)

# A window of at most `limit` entries is in reach of either end, for every position in it.
def ml_reach_all(+i: Nat, +n: Nat, +lo: Nat, +hi: Nat, +limit: Nat, back: Bool, hle: {Nat.is_le(lo, i) == True{} : Bool}, hlt: {Nat.is_lt(i, Nat.min(hi, n)) == True{} : Bool}, hcap: {Nat.is_le(Nat.sub(Nat.min(hi, n), lo), limit) == True{} : Bool}) -> {ml_in_reach(i, n, lo, hi, limit, back) == True{} : Bool}:
  match back:
    case False{}:
      L.nat_lt_le_trans(Nat.sub(i, lo), Nat.sub(Nat.min(hi, n), lo), limit, ml_sub_lt_sub(lo, i, Nat.min(hi, n), hle, hlt), hcap)
    case True{}:
      L.nat_le_trans(Nat.sub(Nat.min(hi, n), i), Nat.sub(Nat.min(hi, n), lo), limit, ml_sub_le_sub(Nat.min(hi, n), lo, i, hle), hcap)

# ============================================================================
# 3. Laws
# ============================================================================

# promise: every position on a page is a position of the model list
# harm: the shell would index past the end of the list
# catches: the window's end taken from the before_id cursor alone (`Nat.min(hi, n)` replaced by `hi`)
law ml_page_in_list:
  for +i: Nat
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +limit: Nat
  for +back: Bool
  for ok: {ml_on_page(i, n, lo, hi, limit, back) == True{} : Bool}
  {Nat.is_lt(i, n) == True{} : Bool}

def ml_page_in_list(i, n, lo, hi, limit, back, ok):
  +hw = ml_and_l(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, limit, back), ok)
  +hm = ml_and_r(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n)), hw)
  L.nat_lt_le_trans(i, Nat.min(hi, n), n, hm, L.nat_min_le_r(hi, n))

# promise: a page never holds the after_id entry or any entry before it
# catches: the after_id bound dropped from the window (`Nat.is_le(lo, i)` replaced by `True{}`)
law ml_page_after_cursor:
  for +i: Nat
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +limit: Nat
  for +back: Bool
  for ok: {ml_on_page(i, n, lo, hi, limit, back) == True{} : Bool}
  {Nat.is_le(lo, i) == True{} : Bool}

def ml_page_after_cursor(i, n, lo, hi, limit, back, ok):
  +hw = ml_and_l(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, limit, back), ok)
  ml_and_l(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n)), hw)

# promise: a page never holds the before_id entry or any entry after it
# catches: the window's end taken from the list length alone (`Nat.min(hi, n)` replaced by `n`)
law ml_page_before_cursor:
  for +i: Nat
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +limit: Nat
  for +back: Bool
  for ok: {ml_on_page(i, n, lo, hi, limit, back) == True{} : Bool}
  {Nat.is_lt(i, hi) == True{} : Bool}

def ml_page_before_cursor(i, n, lo, hi, limit, back, ok):
  +hw = ml_and_l(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, limit, back), ok)
  +hm = ml_and_r(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n)), hw)
  L.nat_lt_le_trans(i, Nat.min(hi, n), hi, hm, L.nat_min_le_l(hi, n))

# promise: a page read forward lies within `limit` positions of the after_id cursor, so it holds at most `limit` entries
# catches: one entry too many (`Nat.is_lt(Nat.sub(i, lo), limit)` replaced by `Nat.is_le`)
law ml_page_forward_within_limit:
  for +i: Nat
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +limit: Nat
  for ok: {ml_on_page(i, n, lo, hi, limit, False{}) == True{} : Bool}
  {Nat.is_lt(Nat.sub(i, lo), limit) == True{} : Bool}

def ml_page_forward_within_limit(i, n, lo, hi, limit, ok):
  ml_and_r(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, limit, False{}), ok)

# promise: a page read from its end lies within `limit` positions of the window's end, so it holds at most `limit` entries
# catches: a page read from its end measured from the after_id cursor (the forward rule `Nat.is_lt(Nat.sub(i, lo), limit)`)
law ml_page_back_within_limit:
  for +i: Nat
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +limit: Nat
  for ok: {ml_on_page(i, n, lo, hi, limit, True{}) == True{} : Bool}
  {Nat.is_le(Nat.sub(Nat.min(hi, n), i), limit) == True{} : Bool}

def ml_page_back_within_limit(i, n, lo, hi, limit, ok):
  ml_and_r(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, limit, True{}), ok)

# promise: a page read forward has no holes: every position from the after_id cursor up to a position on the page is on the page
# harm: a client paging on with the last id would never see an entry the page skipped
# catches: a page with more to follow keeping only its first and last positions (`ml_in_reach` forward also requiring `i == lo` or `i - lo == limit - 1` when `ml_more` holds)
law ml_page_forward_prefix:
  for +lo: Nat
  for +i: Nat
  for +j: Nat
  for +limit: Nat
  for +n: Nat
  for +hi: Nat
  for +hlo: {Nat.is_le(lo, i) == True{} : Bool}
  for +hij: {Nat.is_le(i, j) == True{} : Bool}
  for +ok: {ml_on_page(j, n, lo, hi, limit, False{}) == True{} : Bool}
  {ml_on_page(i, n, lo, hi, limit, False{}) == True{} : Bool}

def ml_page_forward_prefix(lo, i, j, limit, n, hi, hlo, hij, ok):
  +hw = ml_and_l(ml_in_window(j, n, lo, hi), ml_in_reach(j, n, lo, hi, limit, False{}), ok)
  +hr = ml_and_r(ml_in_window(j, n, lo, hi), ml_in_reach(j, n, lo, hi, limit, False{}), ok)
  +hjm = ml_and_r(Nat.is_le(lo, j), Nat.is_lt(j, Nat.min(hi, n)), hw)
  +him = L.nat_le_lt_trans(i, j, Nat.min(hi, n), hij, hjm)
  +hir = L.nat_le_lt_trans(Nat.sub(i, lo), Nat.sub(j, lo), limit, ml_sub_le_sub_r(i, j, lo, hij), hr)
  ml_and_intro(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, limit, False{}), ml_and_intro(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n)), hlo, him), hir)

# promise: a page read from its end has no holes: every position from a position on the page up to the window's end is on the page
# harm: a client paging back with the first id would never see an entry the page skipped
# catches: a page with more to follow keeping only its first and last positions (`ml_in_reach` back also requiring `m - i == 1` or `m - i == limit` when `ml_more` holds, `m` the window's end)
law ml_page_back_suffix:
  for +lo: Nat
  for +j: Nat
  for +i: Nat
  for +limit: Nat
  for +n: Nat
  for +hi: Nat
  for +hji: {Nat.is_le(j, i) == True{} : Bool}
  for +hend: {Nat.is_lt(i, Nat.min(hi, n)) == True{} : Bool}
  for +ok: {ml_on_page(j, n, lo, hi, limit, True{}) == True{} : Bool}
  {ml_on_page(i, n, lo, hi, limit, True{}) == True{} : Bool}

def ml_page_back_suffix(lo, j, i, limit, n, hi, hji, hend, ok):
  +hw = ml_and_l(ml_in_window(j, n, lo, hi), ml_in_reach(j, n, lo, hi, limit, True{}), ok)
  +hr = ml_and_r(ml_in_window(j, n, lo, hi), ml_in_reach(j, n, lo, hi, limit, True{}), ok)
  +hlj = ml_and_l(Nat.is_le(lo, j), Nat.is_lt(j, Nat.min(hi, n)), hw)
  +hli = L.nat_le_trans(lo, j, i, hlj, hji)
  +hir = L.nat_le_trans(Nat.sub(Nat.min(hi, n), i), Nat.sub(Nat.min(hi, n), j), limit, ml_sub_le_sub(Nat.min(hi, n), j, i, hji), hr)
  ml_and_intro(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, limit, True{}), ml_and_intro(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n)), hli, hend), hir)

# promise: a forward page that says more entries follow is not empty, so a client paging on with its last id always moves
# harm: a client would ask for the same empty page forever
# catches: has_more measured to the list's end (`Nat.min(hi, n)` replaced by `n` in `ml_more`)
law ml_more_forward_nonempty:
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +limit: Nat
  for +ok: {ml_more(n, lo, hi, limit) == True{} : Bool}
  {ml_on_page(lo, n, lo, hi, limit, False{}) == True{} : Bool}

def ml_more_forward_nonempty(n, lo, hi, limit, ok):
  +hpos = ml_and_l(Nat.is_lt(0n, limit), Nat.is_lt(limit, Nat.sub(Nat.min(hi, n), lo)), ok)
  +hwin = ml_and_r(Nat.is_lt(0n, limit), Nat.is_lt(limit, Nat.sub(Nat.min(hi, n), lo)), ok)
  +hlt = ml_lt_sub_imp_lt(Nat.min(hi, n), lo, limit, hwin)
  +hreach = L.nat_lt_subst_l(0n, Nat.sub(lo, lo), limit, Equal.sym(Nat, Nat.sub(lo, lo), 0n, L.nat_sub_self(lo)), hpos)
  ml_and_intro(ml_in_window(lo, n, lo, hi), ml_in_reach(lo, n, lo, hi, limit, False{}), ml_and_intro(Nat.is_le(lo, lo), Nat.is_lt(lo, Nat.min(hi, n)), L.nat_le_refl(lo), hlt), hreach)

# promise: a page read from its end that says more entries follow is not empty, so a client paging back with its first id always moves
# harm: a client would ask for the same empty page forever
# catches: the empty-page test dropped (`Nat.is_lt(0n, limit)` removed from `ml_more`)
law ml_more_back_nonempty:
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +limit: Nat
  for +ok: {ml_more(n, lo, hi, limit) == True{} : Bool}
  {ml_on_page(Nat.sub(Nat.min(hi, n), 1n), n, lo, hi, limit, True{}) == True{} : Bool}

def ml_more_back_nonempty(n, lo, hi, limit, ok):
  +hpos = ml_and_l(Nat.is_lt(0n, limit), Nat.is_lt(limit, Nat.sub(Nat.min(hi, n), lo)), ok)
  +hwin = ml_and_r(Nat.is_lt(0n, limit), Nat.is_lt(limit, Nat.sub(Nat.min(hi, n), lo)), ok)
  +hlt = ml_lt_sub_imp_lt(Nat.min(hi, n), lo, limit, hwin)
  ml_back_last(Nat.min(hi, n), lo, limit, hlt, hpos)

# promise: has_more is said only when the window holds an entry past the page's `limit` entries
# harm: a client would page on to an empty page
# catches: has_more said for a page that is exactly full (`Nat.is_lt(limit, ...)` replaced by `Nat.is_le`)
law ml_more_only_past_limit:
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +limit: Nat
  for ok: {ml_more(n, lo, hi, limit) == True{} : Bool}
  {Nat.is_lt(Nat.add(lo, limit), Nat.min(hi, n)) == True{} : Bool}

def ml_more_only_past_limit(n, lo, hi, limit, ok):
  +hwin = ml_and_r(Nat.is_lt(0n, limit), Nat.is_lt(limit, Nat.sub(Nat.min(hi, n), lo)), ok)
  ml_lt_sub_add(Nat.min(hi, n), lo, limit, hwin)

# promise: a page of at least one entry that does not say more entries follow holds every entry between the cursors
# harm: a client would stop paging with entries it never saw
# catches: has_more never said (`ml_more` returns `False{}`)
law ml_page_complete_without_more:
  for +i: Nat
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +p: Nat
  for +back: Bool
  for nm: {ml_more(n, lo, hi, 1n+p) == False{} : Bool}
  for +hw: {Bool.and(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n))) == True{} : Bool}
  {ml_on_page(i, n, lo, hi, 1n+p, back) == True{} : Bool}

def ml_page_complete_without_more(i, n, lo, hi, p, back, nm, hw):
  +hle = ml_and_l(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n)), hw)
  +hlt = ml_and_r(Nat.is_le(lo, i), Nat.is_lt(i, Nat.min(hi, n)), hw)
  +hcap = L.nat_not_lt_imp_le(1n+p, Nat.sub(Nat.min(hi, n), lo), nm)
  ml_and_intro(ml_in_window(i, n, lo, hi), ml_in_reach(i, n, lo, hi, 1n+p, back), hw, ml_reach_all(i, n, lo, hi, 1n+p, back, hle, hlt, hcap))

# A window that is not empty ends at an entry inside it.
def ml_last_or(+n: Nat, +lo: Nat, +hi: Nat, +p: Nat, b: Bool, eb: {b == Nat.is_le(Nat.min(hi, n), lo) : Bool}, nm: {ml_more(n, lo, hi, 1n+p) == False{} : Bool}) -> {Bool.or(b, ml_on_page(Nat.sub(Nat.min(hi, n), 1n), n, lo, hi, 1n+p, False{})) == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      +hlt = G.g_nat_not_le_imp_lt(Nat.min(hi, n), lo, Equal.sym(Bool, False{}, Nat.is_le(Nat.min(hi, n), lo), eb))
      +hlast = ml_back_last(Nat.min(hi, n), lo, 1n+p, hlt, {==})
      +hwin = ml_and_l(Bool.and(Nat.is_le(lo, Nat.sub(Nat.min(hi, n), 1n)), Nat.is_lt(Nat.sub(Nat.min(hi, n), 1n), Nat.min(hi, n))), Nat.is_le(Nat.sub(Nat.min(hi, n), Nat.sub(Nat.min(hi, n), 1n)), 1n+p), hlast)
      ml_page_complete_without_more(Nat.sub(Nat.min(hi, n), 1n), n, lo, hi, p, False{}, nm, hwin)

# promise: a forward page of at least one entry that does not say more entries follow ends at the last entry before the before_id cursor or the list's end
# harm: a client would stop paging one entry short
# catches: has_more off by one (`Nat.is_lt(limit, ...)` replaced by `Nat.is_lt(Nat.add(limit, 1n), ...)`)
law ml_last_page_reaches_end:
  for +n: Nat
  for +lo: Nat
  for +hi: Nat
  for +p: Nat
  for nm: {ml_more(n, lo, hi, 1n+p) == False{} : Bool}
  {Bool.or(Nat.is_le(Nat.min(hi, n), lo), ml_on_page(Nat.sub(Nat.min(hi, n), 1n), n, lo, hi, 1n+p, False{})) == True{} : Bool}

def ml_last_page_reaches_end(n, lo, hi, p, nm):
  ml_last_or(n, lo, hi, p, Nat.is_le(Nat.min(hi, n), lo), {==}, nm)
