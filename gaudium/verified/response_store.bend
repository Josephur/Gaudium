import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# The in-memory store of past responses (`POST /v1/responses` with `store`, read back with
# `GET /v1/responses/{id}`, chained with `previous_response_id`).
#
# The store holds at most `max_n` entries and at most `max_bytes` bytes of them (both set on the
# command line). The shell keeps the entries oldest first with the size it measured for each; this
# file holds the account of what the store holds (entries and bytes) and the decision for a new
# entry: take it now, evict the oldest entry and ask again, or refuse it. What an entry holds, how
# its size is measured and which entry is the oldest are the shell's; the rules below are what
# every store keeps: an entry is taken only when it leaves the store within both bounds, an
# eviction is asked for only while an entry is held, an entry that fits beside what is held is
# taken without evicting anything, an entry that can never fit is refused without evicting
# anything, and taking and then removing an entry leaves the account as it was.
#
# Two more decisions sit beside the store: whether a response that has just ended is offered to
# it at all (`rs_keep`: never when its request said `store: false` or a DELETE reached it while
# it ran), and whether the request context of a response about to be generated is admitted into
# the account of contexts in flight (`rs_admit`: only within that account's byte bound; those
# contexts cannot be evicted). Which entry leaves first (today the oldest) is the shell's, pinned
# by Rust tests: the laws leave a store that refuses when full instead of evicting open.

# ============================================================================
# 1. Rules
# ============================================================================

# Entries and bytes the store holds.
type RsAcct is Data:
  RsAcct{n: Nat, bytes: Nat}

# What becomes of a new entry.
type RsPut is Data:
  # the store takes it now
  RsPutStore{}
  # the oldest entry leaves first; then the store is asked again
  RsPutEvict{}
  # it is not kept
  RsPutRefuse{}

# The empty store.
def rs_empty() -> RsAcct:
  RsAcct{0n, 0n}

def rs_count(+a: RsAcct) -> Nat:
  match a:
    case RsAcct{n, b}:
      n

def rs_bytes(+a: RsAcct) -> Nat:
  match a:
    case RsAcct{n, b}:
      b

# The account once the store took an entry of `size` bytes.
def rs_add(+a: RsAcct, +size: Nat) -> RsAcct:
  match a:
    case RsAcct{n, b}:
      RsAcct{1n+n, Nat.add(b, size)}

def rs_remove_pick(ok: Bool, +n: Nat, +b: Nat, +size: Nat) -> RsAcct:
  match ok:
    case True{}:
      RsAcct{Nat.sub(n, 1n), Nat.sub(b, size)}
    case False{}:
      RsAcct{n, b}

# The account once an entry of `size` bytes left the store. An entry the account cannot hold (no
# entry held, or more bytes than held) changes nothing.
def rs_remove(+a: RsAcct, +size: Nat) -> RsAcct:
  match a:
    case RsAcct{n, b}:
      rs_remove_pick(Bool.and(Nat.is_lt(0n, n), Nat.is_le(size, b)), n, b, size)

# Does an entry of `size` bytes fit beside what `a` holds? (The byte check subtracts, so it
# cannot overflow on a size a client chose.)
def rs_fits(+a: RsAcct, +max_n: Nat, +max_bytes: Nat, +size: Nat) -> Bool:
  match a:
    case RsAcct{n, b}:
      Bool.and(Nat.is_lt(n, max_n), Bool.and(Nat.is_le(b, max_bytes), Nat.is_le(size, Nat.sub(max_bytes, b))))

def rs_put_pick(possible: Bool, fits: Bool, any: Bool) -> RsPut:
  match possible:
    case False{}:
      RsPutRefuse{}
    case True{}:
      match fits:
        case True{}:
          RsPutStore{}
        case False{}:
          match any:
            case True{}:
              RsPutEvict{}
            case False{}:
              RsPutRefuse{}

# What becomes of a new entry of `size` bytes in a store holding `a`, with bounds of `max_n`
# entries and `max_bytes` bytes: an entry that fits is taken; one that could fit an empty store
# waits for the oldest entry to leave; one that never fits (the store is off, or the entry alone
# is larger than the byte bound) is refused.
def rs_put(+a: RsAcct, +max_n: Nat, +max_bytes: Nat, +size: Nat) -> RsPut:
  rs_put_pick(Bool.and(Nat.is_lt(0n, max_n), Nat.is_le(size, max_bytes)), rs_fits(a, max_n, max_bytes, size), Nat.is_lt(0n, rs_count(a)))

def rs_put_stores(+p: RsPut) -> Bool:
  match p:
    case RsPutStore{}:
      True{}
    case RsPutEvict{}:
      False{}
    case RsPutRefuse{}:
      False{}

def rs_put_refuses(+p: RsPut) -> Bool:
  match p:
    case RsPutStore{}:
      False{}
    case RsPutEvict{}:
      False{}
    case RsPutRefuse{}:
      True{}

def rs_put_evicts(+p: RsPut) -> Bool:
  match p:
    case RsPutStore{}:
      False{}
    case RsPutEvict{}:
      True{}
    case RsPutRefuse{}:
      False{}

# Is a response that has just ended offered to the store? `want`: its request asked for it to be
# kept (`store`); `deleted`: a `DELETE /v1/responses/{id}` reached it while it was being
# generated. A response asked not to be kept, or deleted while it ran, is never offered.
def rs_keep(want: Bool, deleted: Bool) -> Bool:
  Bool.and(want, Bool.not(deleted))

# The request contexts of the responses being generated are held in an account of their own
# (`RsAcct`, changed with `rs_add` / `rs_remove`); none of them can be evicted. Is a new one of
# `size` bytes admitted beside what `a` holds, within `max_bytes`? (The check subtracts, so it
# cannot overflow on a size a client chose.)
def rs_admit(+a: RsAcct, +max_bytes: Nat, +size: Nat) -> Bool:
  match a:
    case RsAcct{n, b}:
      Bool.and(Nat.is_le(b, max_bytes), Nat.is_le(size, Nat.sub(max_bytes, b)))

# ============================================================================
# 2. Lemmas
# ============================================================================

def rs_lt_succ_le(+n: Nat, +m: Nat, h: {Nat.is_lt(n, m) == True{} : Bool}) -> {Nat.is_le(1n+n, m) == True{} : Bool}:
  match n m:
    case 0n 0n:
      L.bool_absurd({Nat.is_le(1n, 0n) == True{} : Bool}, h)
    case 0n 1n+mp:
      L.nat_le_zero(mp)
    case 1n+np 0n:
      L.bool_absurd({Nat.is_le(1n+1n+np, 0n) == True{} : Bool}, h)
    case 1n+np 1n+mp:
      rs_lt_succ_le(np, mp, h)

def rs_add_within(+b: Nat, +mb: Nat, s: Nat, h1: {Nat.is_le(b, mb) == True{} : Bool}, h2: {Nat.is_le(s, Nat.sub(mb, b)) == True{} : Bool}) -> {Nat.is_le(Nat.add(b, s), mb) == True{} : Bool}:
  match b mb:
    case 0n _:
      L.nat_le_subst_r(s, Nat.sub(mb, 0n), mb, L.nat_sub_zero_r(mb), h2)
    case 1n+bp 0n:
      L.bool_absurd({Nat.is_le(Nat.add(1n+bp, s), 0n) == True{} : Bool}, h1)
    case 1n+bp 1n+mp:
      rs_add_within(bp, mp, s, h1, h2)

# A pick that is told the entry does not fit never takes it.
def rs_pick_unfit_no_store(p: Bool, a: Bool, x: Bool) -> {Bool.or(Bool.not(rs_put_stores(rs_put_pick(p, False{}, a))), x) == True{} : Bool}:
  match p:
    case False{}:
      {==}
    case True{}:
      match a:
        case True{}:
          {==}
        case False{}:
          {==}

# A pick told that no entry is held never asks for an eviction.
def rs_pick_none_no_evict(p: Bool, f: Bool) -> {Bool.or(Bool.not(rs_put_evicts(rs_put_pick(p, f, False{}))), False{}) == True{} : Bool}:
  match p:
    case False{}:
      {==}
    case True{}:
      match f:
        case True{}:
          {==}
        case False{}:
          {==}

def rs_store_count_go(f1: Bool, f2: Bool, f3: Bool, p: Bool, a: Bool, +n: Nat, +mn: Nat, e1: {Nat.is_lt(n, mn) == f1 : Bool}) -> {Bool.or(Bool.not(rs_put_stores(rs_put_pick(p, Bool.and(f1, Bool.and(f2, f3)), a))), Nat.is_le(1n+n, mn)) == True{} : Bool}:
  match f1:
    case False{}:
      rs_pick_unfit_no_store(p, a, Nat.is_le(1n+n, mn))
    case True{}:
      match f2:
        case False{}:
          rs_pick_unfit_no_store(p, a, Nat.is_le(1n+n, mn))
        case True{}:
          match f3:
            case False{}:
              rs_pick_unfit_no_store(p, a, Nat.is_le(1n+n, mn))
            case True{}:
              match p:
                case False{}:
                  {==}
                case True{}:
                  rs_lt_succ_le(n, mn, e1)

def rs_store_bytes_go(f1: Bool, f2: Bool, f3: Bool, p: Bool, a: Bool, +b: Nat, +mb: Nat, +s: Nat, e2: {Nat.is_le(b, mb) == f2 : Bool}, e3: {Nat.is_le(s, Nat.sub(mb, b)) == f3 : Bool}) -> {Bool.or(Bool.not(rs_put_stores(rs_put_pick(p, Bool.and(f1, Bool.and(f2, f3)), a))), Nat.is_le(Nat.add(b, s), mb)) == True{} : Bool}:
  match f1:
    case False{}:
      rs_pick_unfit_no_store(p, a, Nat.is_le(Nat.add(b, s), mb))
    case True{}:
      match f2:
        case False{}:
          rs_pick_unfit_no_store(p, a, Nat.is_le(Nat.add(b, s), mb))
        case True{}:
          match f3:
            case False{}:
              rs_pick_unfit_no_store(p, a, Nat.is_le(Nat.add(b, s), mb))
            case True{}:
              match p:
                case False{}:
                  {==}
                case True{}:
                  rs_add_within(b, mb, s, e2, e3)

def rs_admit_go(f2: Bool, f3: Bool, +b: Nat, +mb: Nat, +s: Nat, e2: {Nat.is_le(b, mb) == f2 : Bool}, e3: {Nat.is_le(s, Nat.sub(mb, b)) == f3 : Bool}) -> {Bool.or(Bool.not(Bool.and(f2, f3)), Nat.is_le(Nat.add(b, s), mb)) == True{} : Bool}:
  match f2:
    case False{}:
      {==}
    case True{}:
      match f3:
        case False{}:
          {==}
        case True{}:
          rs_add_within(b, mb, s, e2, e3)

# ============================================================================
# 3. Laws
# ============================================================================

# promise: an entry the store takes leaves it holding no more entries than its bound.
# harm: the store would grow one entry past its bound on every full put and its memory with it
# catches: rs_fits compares Nat.is_le(n, max_n) instead of Nat.is_lt(n, max_n)
law rs_put_store_within_count:
  for +a: RsAcct
  for +mn: Nat
  for +mb: Nat
  for +s: Nat
  {Bool.or(Bool.not(rs_put_stores(rs_put(a, mn, mb, s))), Nat.is_le(rs_count(rs_add(a, s)), mn)) == True{} : Bool}

def rs_put_store_within_count(a, mn, mb, s):
  match a:
    case RsAcct{n, b}:
      rs_store_count_go(Nat.is_lt(n, mn), Nat.is_le(b, mb), Nat.is_le(s, Nat.sub(mb, b)), Bool.and(Nat.is_lt(0n, mn), Nat.is_le(s, mb)), Nat.is_lt(0n, n), n, mn, {==})

# promise: an entry the store takes leaves it holding no more bytes than its bound.
# harm: large responses could pile up past the byte bound and exhaust host memory
# catches: rs_fits compares the entry with max_bytes alone, ignoring the bytes already held
law rs_put_store_within_bytes:
  for +a: RsAcct
  for +mn: Nat
  for +mb: Nat
  for +s: Nat
  {Bool.or(Bool.not(rs_put_stores(rs_put(a, mn, mb, s))), Nat.is_le(rs_bytes(rs_add(a, s)), mb)) == True{} : Bool}

def rs_put_store_within_bytes(a, mn, mb, s):
  match a:
    case RsAcct{n, b}:
      rs_store_bytes_go(Nat.is_lt(n, mn), Nat.is_le(b, mb), Nat.is_le(s, Nat.sub(mb, b)), Bool.and(Nat.is_lt(0n, mn), Nat.is_le(s, mb)), Nat.is_lt(0n, n), b, mb, s, {==}, {==})

# promise: an eviction is asked for only while the store holds an entry, so a put ends after at most as many evictions as there are entries.
# harm: the shell would be told to evict from an empty store and loop or give up on an entry that fits
# catches: rs_put_pick returns RsPutEvict{} when the entry does not fit, whether or not any entry is held
law rs_put_evict_has_entry:
  for +a: RsAcct
  for +mn: Nat
  for +mb: Nat
  for +s: Nat
  {Bool.or(Bool.not(rs_put_evicts(rs_put(a, mn, mb, s))), Nat.is_lt(0n, rs_count(a))) == True{} : Bool}

def rs_put_evict_has_entry(a, mn, mb, s):
  match a:
    case RsAcct{n, b}:
      match n:
        case 0n:
          rs_pick_none_no_evict(Bool.and(Nat.is_lt(0n, mn), Nat.is_le(s, mb)), rs_fits(RsAcct{0n, b}, mn, mb, s))
        case 1n+np:
          G.g_bool_or_true_r(Bool.not(rs_put_evicts(rs_put(RsAcct{1n+np, b}, mn, mb, s))))

# promise: an entry larger than the byte bound is refused whatever the store holds, so nothing is evicted for it.
# harm: one oversized response would empty the store of every other response and still not be kept
# catches: rs_put drops the Nat.is_le(size, max_bytes) check from its first argument
law rs_put_oversize_refused:
  for +a: RsAcct
  for +mn: Nat
  for +mb: Nat
  for +x: Nat
  {rs_put(a, mn, mb, Nat.add(mb, 1n+x)) == RsPutRefuse{} : RsPut}

def rs_put_oversize_refused(a, mn, mb, x):
  match a:
    case RsAcct{n, b}:
      %Equal.sym(Bool, Nat.is_le(Nat.add(mb, 1n+x), mb), False{}, G.g_nat_le_add_succ_false(mb, x)) : {rs_put_pick(Bool.and(Nat.is_lt(0n, mn), _), Bool.and(Nat.is_lt(n, mn), Bool.and(Nat.is_le(b, mb), Nat.is_le(Nat.add(mb, 1n+x), Nat.sub(mb, b)))), Nat.is_lt(0n, n)) == RsPutRefuse{} : RsPut}
      %Equal.sym(Bool, Bool.and(Nat.is_lt(0n, mn), False{}), False{}, G.g_bool_and_false_r(Nat.is_lt(0n, mn))) : {rs_put_pick(_, Bool.and(Nat.is_lt(n, mn), Bool.and(Nat.is_le(b, mb), Nat.is_le(Nat.add(mb, 1n+x), Nat.sub(mb, b)))), Nat.is_lt(0n, n)) == RsPutRefuse{} : RsPut}
      {==}

# promise: a store whose entry bound is zero (the store is off) keeps nothing.
# harm: `--response-store 0` would still keep responses
# catches: rs_put drops the Nat.is_lt(0n, max_n) check from its first argument
law rs_put_off_refuses:
  for +a: RsAcct
  for +mb: Nat
  for +s: Nat
  {rs_put(a, 0n, mb, s) == RsPutRefuse{} : RsPut}

def rs_put_off_refuses(a, mb, s):
  match a:
    case RsAcct{n, b}:
      {==}

# promise: an empty store with room for one entry takes an entry within its byte bound.
# harm: a store that refuses everything would pass every other law and make previous_response_id useless
# catches: rs_put returns RsPutRefuse{} for every entry
law rs_put_empty_takes:
  for +k: Nat
  for +s: Nat
  for +x: Nat
  {rs_put(rs_empty(), 1n+k, Nat.add(s, x), s) == RsPutStore{} : RsPut}

def rs_put_empty_takes(k, s, x):
  %Equal.sym(Nat, Nat.sub(Nat.add(s, x), 0n), Nat.add(s, x), L.nat_sub_zero_r(Nat.add(s, x))) : {rs_put_pick(Nat.is_le(s, Nat.add(s, x)), Bool.and(Nat.is_le(0n, Nat.add(s, x)), Nat.is_le(s, _)), False{}) == RsPutStore{} : RsPut}
  %Equal.sym(Bool, Nat.is_le(0n, Nat.add(s, x)), True{}, L.nat_le_zero(Nat.add(s, x))) : {rs_put_pick(Nat.is_le(s, Nat.add(s, x)), Bool.and(_, Nat.is_le(s, Nat.add(s, x))), False{}) == RsPutStore{} : RsPut}
  %Equal.sym(Bool, Nat.is_le(s, Nat.add(s, x)), True{}, L.nat_add_le_r(s, x)) : {rs_put_pick(_, Bool.and(True{}, _), False{}) == RsPutStore{} : RsPut}
  {==}

# promise: taking an entry and then removing it leaves the account exactly as it was: entries and bytes are conserved.
# harm: the account would drift from what the store holds and the bounds would stop meaning anything
# catches: rs_remove takes the bytes off but leaves the entry count as it was
law rs_remove_add_inverse:
  for +a: RsAcct
  for +s: Nat
  {rs_remove(rs_add(a, s), s) == a : RsAcct}

def rs_remove_add_inverse(a, s):
  match a:
    case RsAcct{n, b}:
      %Equal.sym(Bool, Nat.is_le(s, Nat.add(b, s)), True{}, L.nat_add_le_l(b, s)) : {rs_remove_pick(_, 1n+n, Nat.add(b, s), s) == RsAcct{n, b} : RsAcct}
      %Equal.sym(Nat, Nat.sub(n, 0n), n, L.nat_sub_zero_r(n)) : {RsAcct{_, Nat.sub(Nat.add(b, s), s)} == RsAcct{n, b} : RsAcct}
      %Equal.sym(Nat, Nat.sub(Nat.add(b, s), s), b, L.nat_add_sub_cancel_r(b, s)) : {RsAcct{n, _} == RsAcct{n, b} : RsAcct}
      {==}

# promise: removing from a store that holds no entry changes nothing, so its bytes never go below what entries hold.
# harm: a stray removal would lower the byte count and let the store take more than its bound
# catches: rs_remove drops the Nat.is_lt(0n, n) check and takes the bytes off anyway
law rs_remove_empty_unchanged:
  for +b: Nat
  for +s: Nat
  {rs_remove(RsAcct{0n, b}, s) == RsAcct{0n, b} : RsAcct}

def rs_remove_empty_unchanged(b, s):
  {==}

# promise: removing more bytes than the store holds changes nothing, so the byte count never drops below what the remaining entries hold.
# harm: a removal with a wrong size would zero the byte count and let the store take far more than its bound
# catches: rs_remove drops the Nat.is_le(size, b) check and takes the entry off anyway
law rs_remove_oversize_unchanged:
  for +n: Nat
  for +b: Nat
  for +x: Nat
  {rs_remove(RsAcct{1n+n, b}, Nat.add(b, 1n+x)) == RsAcct{1n+n, b} : RsAcct}

def rs_remove_oversize_unchanged(n, b, x):
  %Equal.sym(Bool, Nat.is_le(Nat.add(b, 1n+x), b), False{}, G.g_nat_le_add_succ_false(b, x)) : {rs_remove_pick(_, 1n+n, b, Nat.add(b, 1n+x)) == RsAcct{1n+n, b} : RsAcct}
  {==}

# promise: an entry that fits beside what the store holds is taken now, so no entry is evicted that need not leave.
# harm: the store would forget older responses on every put and previous_response_id would work only for the latest one
# catches: rs_put_pick evicts whenever an entry is held, even when the new entry fits
law rs_put_fitting_taken:
  for +n: Nat
  for +b: Nat
  for +k: Nat
  for +s: Nat
  for +x: Nat
  {rs_put(RsAcct{n, b}, Nat.add(n, 1n+k), Nat.add(b, Nat.add(s, x)), s) == RsPutStore{} : RsPut}

def rs_put_fitting_taken(n, b, k, s, x):
  %Equal.sym(Nat, Nat.add(n, 1n+k), 1n+Nat.add(n, k), L.nat_add_succ_r(n, k)) : {rs_put_pick(Bool.and(Nat.is_lt(0n, _), Nat.is_le(s, Nat.add(b, Nat.add(s, x)))), Bool.and(Nat.is_lt(n, _), Bool.and(Nat.is_le(b, Nat.add(b, Nat.add(s, x))), Nat.is_le(s, Nat.sub(Nat.add(b, Nat.add(s, x)), b)))), Nat.is_lt(0n, n)) == RsPutStore{} : RsPut}
  %Equal.sym(Bool, Nat.is_lt(n, 1n+Nat.add(n, k)), Nat.is_le(n, Nat.add(n, k)), L.nat_lt_succ_eq_le(n, Nat.add(n, k))) : {rs_put_pick(Bool.and(Nat.is_lt(0n, 1n+Nat.add(n, k)), Nat.is_le(s, Nat.add(b, Nat.add(s, x)))), Bool.and(_, Bool.and(Nat.is_le(b, Nat.add(b, Nat.add(s, x))), Nat.is_le(s, Nat.sub(Nat.add(b, Nat.add(s, x)), b)))), Nat.is_lt(0n, n)) == RsPutStore{} : RsPut}
  %Equal.sym(Bool, Nat.is_le(n, Nat.add(n, k)), True{}, L.nat_add_le_r(n, k)) : {rs_put_pick(Bool.and(Nat.is_lt(0n, 1n+Nat.add(n, k)), Nat.is_le(s, Nat.add(b, Nat.add(s, x)))), Bool.and(_, Bool.and(Nat.is_le(b, Nat.add(b, Nat.add(s, x))), Nat.is_le(s, Nat.sub(Nat.add(b, Nat.add(s, x)), b)))), Nat.is_lt(0n, n)) == RsPutStore{} : RsPut}
  %Equal.sym(Bool, Nat.is_le(b, Nat.add(b, Nat.add(s, x))), True{}, L.nat_add_le_r(b, Nat.add(s, x))) : {rs_put_pick(Bool.and(Nat.is_lt(0n, 1n+Nat.add(n, k)), Nat.is_le(s, Nat.add(b, Nat.add(s, x)))), Bool.and(True{}, Bool.and(_, Nat.is_le(s, Nat.sub(Nat.add(b, Nat.add(s, x)), b)))), Nat.is_lt(0n, n)) == RsPutStore{} : RsPut}
  %Equal.sym(Nat, Nat.sub(Nat.add(b, Nat.add(s, x)), b), Nat.add(s, x), L.nat_add_sub_cancel_l(b, Nat.add(s, x))) : {rs_put_pick(Bool.and(Nat.is_lt(0n, 1n+Nat.add(n, k)), Nat.is_le(s, Nat.add(b, Nat.add(s, x)))), Bool.and(True{}, Bool.and(True{}, Nat.is_le(s, _))), Nat.is_lt(0n, n)) == RsPutStore{} : RsPut}
  %Equal.sym(Bool, Nat.is_le(s, Nat.add(s, x)), True{}, L.nat_add_le_r(s, x)) : {rs_put_pick(Bool.and(Nat.is_lt(0n, 1n+Nat.add(n, k)), Nat.is_le(s, Nat.add(b, Nat.add(s, x)))), Bool.and(True{}, Bool.and(True{}, _)), Nat.is_lt(0n, n)) == RsPutStore{} : RsPut}
  %Equal.sym(Bool, Nat.is_le(s, Nat.add(b, Nat.add(s, x))), True{}, L.nat_le_trans(s, Nat.add(s, x), Nat.add(b, Nat.add(s, x)), L.nat_add_le_r(s, x), L.nat_add_le_l(b, Nat.add(s, x)))) : {rs_put_pick(Bool.and(Nat.is_lt(0n, 1n+Nat.add(n, k)), _), Bool.and(True{}, Bool.and(True{}, True{})), Nat.is_lt(0n, n)) == RsPutStore{} : RsPut}
  {==}

# promise: rs_count reads the number of entries the account holds.
# harm: rs_put would decide on a wrong count: never evict from a full store, or evict from an empty one
# catches: rs_count returns 0n whatever the account holds
law rs_count_reads:
  for +n: Nat
  for +b: Nat
  {rs_count(RsAcct{n, b}) == n : Nat}

def rs_count_reads(n, b):
  {==}

# promise: rs_bytes reads the number of bytes the account holds.
# harm: the byte bound law would be stated about a number that is not what the store holds
# catches: rs_bytes returns 0n whatever the account holds
law rs_bytes_reads:
  for +n: Nat
  for +b: Nat
  {rs_bytes(RsAcct{n, b}) == b : Nat}

def rs_bytes_reads(n, b):
  {==}

# promise: a response whose request asked for it not to be kept (store: false) is never offered to the store.
# harm: a client that sent store: false would find its conversation readable by its id afterwards
# catches: rs_keep ignores want and returns Bool.not(deleted)
law rs_keep_needs_want:
  for d: Bool
  {rs_keep(False{}, d) == False{} : Bool}

def rs_keep_needs_want(d):
  {==}

# promise: a response deleted while it was being generated is never offered to the store.
# harm: a DELETE answered with deleted: true would be undone when the response finished, and the conversation kept
# catches: rs_keep ignores deleted and returns want
law rs_keep_deleted_dropped:
  for +w: Bool
  {rs_keep(w, True{}) == False{} : Bool}

def rs_keep_deleted_dropped(w):
  G.g_bool_and_false_r(w)

# promise: a response that was not deleted while it ran is offered to the store exactly when its request asked for it.
# harm: a store that is never offered anything would pass the other keep laws and make previous_response_id useless
# catches: rs_keep returns False{} for every response
law rs_keep_undeleted_as_asked:
  for +w: Bool
  {rs_keep(w, False{}) == w : Bool}

def rs_keep_undeleted_as_asked(w):
  match w:
    case True{}:
      {==}
    case False{}:
      {==}

# promise: a request context the in-flight account admits leaves it holding no more bytes than its bound.
# harm: concurrent requests chaining a large stored conversation would each hold a copy of it with no bound on their sum
# catches: rs_admit compares the new context with max_bytes alone, ignoring the bytes already held
law rs_admit_within_bytes:
  for +a: RsAcct
  for +mb: Nat
  for +s: Nat
  {Bool.or(Bool.not(rs_admit(a, mb, s)), Nat.is_le(rs_bytes(rs_add(a, s)), mb)) == True{} : Bool}

def rs_admit_within_bytes(a, mb, s):
  match a:
    case RsAcct{n, b}:
      rs_admit_go(Nat.is_le(b, mb), Nat.is_le(s, Nat.sub(mb, b)), b, mb, s, {==}, {==})

# promise: a request context that fits beside what the in-flight account holds is admitted.
# harm: an account that admits nothing would pass the bound law and refuse every Responses request
# catches: rs_admit uses Nat.is_lt(size, max_bytes - b) and refuses a context that fills the bound exactly
law rs_admit_fitting:
  for +n: Nat
  for +b: Nat
  for +s: Nat
  for +x: Nat
  {rs_admit(RsAcct{n, b}, Nat.add(b, Nat.add(s, x)), s) == True{} : Bool}

def rs_admit_fitting(n, b, s, x):
  %Equal.sym(Bool, Nat.is_le(b, Nat.add(b, Nat.add(s, x))), True{}, L.nat_add_le_r(b, Nat.add(s, x))) : {Bool.and(_, Nat.is_le(s, Nat.sub(Nat.add(b, Nat.add(s, x)), b))) == True{} : Bool}
  %Equal.sym(Nat, Nat.sub(Nat.add(b, Nat.add(s, x)), b), Nat.add(s, x), L.nat_add_sub_cancel_l(b, Nat.add(s, x))) : {Bool.and(True{}, Nat.is_le(s, _)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_le(s, Nat.add(s, x)), True{}, L.nat_add_le_r(s, x)) : {Bool.and(True{}, _) == True{} : Bool}
  {==}
