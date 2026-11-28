# Heap meter ownership: which count a released block credits, and what a reallocated block is
# charged (`src/heap_meter.rs`).
#
# The PDF reader checks the heap a request and each of its documents hold against their caps
# (`pdf_within`, `pdf_budget_fits` in `pdf_bounds.bend`). The allocator counts that heap per
# thread while a metered task runs, in frames that nest (a request's expansion, and one
# document read inside it). Each block allocated while a frame is on is recorded with the
# serial of the frame on top when it was allocated, and with the bytes it was charged:
#   - a release credits a frame only for a block that frame owns: allocated while it was the
#     top frame, or by a frame started inside it that has ended (serials grow with each start,
#     so a frame owns the serials from its own start up to the start of the frame on top of
#     it). A block allocated before the frame started (a request body parsed before the
#     expansion, an enclosing frame's block) never lowers that frame's count;
#   - a reallocation keeps the block's owner, charges every byte it adds, credits no more than
#     the bytes it removes, and never leaves the block charged for more than its size. A block
#     allocated before any frame started is charged only the bytes added under the meter.
#
# The count itself moves only through these functions too: a charge adds a block's bytes, a
# credit takes a released block's charge off (never below zero), a reallocation rebills the block
# from its old charge to its new one, and a frame that ends folds its count into the enclosing
# one. The PDF reader admits or refuses work on that count (`pdf_within`, `pdf_budget_fits`), so
# the count must hold every byte the frame's blocks were charged. The functions that grow it are
# total: on a count past 64 bits the shell keeps the largest count (`u64::MAX`), which every cap
# refuses, so a hostile document can never panic the allocator.
#
# The record itself (the table of blocks, its growth, the thread-local frames) and the peak (only
# reported) are allocator code in Rust; the decisions it makes on each release and reallocation
# are these functions.

import Base
import lemmas.bend as G
import nat_lemmas.bend as L

# ============================================================================
# 1. Decisions
# ============================================================================

# Whether the frame that started with serial `start` owns a block recorded with serial
# `serial`. `later`: a frame started inside this one is still on, with the serial `next`.
def meter_owns(+start: Nat, +serial: Nat, later: Bool, +next: Nat) -> Bool:
  Bool.and(Nat.is_le(start, serial), Bool.or(Bool.not(later), Nat.is_lt(serial, next)))

# The charge of a block of `old` bytes charged `charge` once it is reallocated to `new` bytes:
# what it was charged that the new size still holds, and every byte added.
def meter_regrow(+charge: Nat, +old: Nat, +new: Nat) -> Nat:
  Nat.add(Nat.min(charge, new), Nat.sub(new, old))

# A frame's count `live` once `n` more bytes are charged to it. `None` past 64 bits.
# rusty-bends: total
def meter_charge(+live: Nat, +n: Nat) -> Nat:
  Nat.add(live, n)

# A frame's count `live` once a released block it was charged `n` for is credited: never below
# zero.
def meter_credit(+live: Nat, +n: Nat) -> Nat:
  Nat.sub(live, n)

# A frame's count `live` once one of its blocks, charged `was`, is charged `now` instead (a
# reallocation, `meter_regrow`): the growth charged, the shrinkage credited. `None` past 64 bits.
# rusty-bends: total
def meter_rebill(+live: Nat, +was: Nat, +now: Nat) -> Nat:
  meter_credit(meter_charge(live, now), was)

# The enclosing frame's count `outer` once the frame inside it ends with the count `inner`: the
# blocks the inner frame kept pass to it. `None` past 64 bits.
# rusty-bends: total
def meter_fold(+outer: Nat, +inner: Nat) -> Nat:
  meter_charge(outer, inner)

# ============================================================================
# 2. Helper lemmas (erased before Rust codegen)
# ============================================================================

def meter_and_false_l(a: Bool, -b: Bool, h: {a == False{} : Bool}) -> {Bool.and(a, b) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      L.bool_absurd_tf({Bool.and(True{}, b) == False{} : Bool}, h)

def meter_and_false_r(a: Bool, -b: Bool, h: {b == False{} : Bool}) -> {Bool.and(a, b) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      h

# `a <= b` and `b < a` never hold together.
def meter_le_not_lt(+a: Nat, +b: Nat, x: Bool, e: {Nat.is_le(a, b) == x : Bool}) -> {Bool.and(x, Nat.is_lt(b, a)) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(b, a, e)

# `y + 1 + c < y` never holds.
def meter_succ_add_not_lt(+y: Nat, +c: Nat) -> {Nat.is_lt(1n+Nat.add(y, c), y) == False{} : Bool}:
  match y:
    case 0n:
      {==}
    case 1n+yp:
      meter_succ_add_not_lt(yp, c)

# `min(c, n) + y` is never more than `c + y`.
def meter_min_add_not_over(+c: Nat, +n: Nat, +y: Nat) -> {Nat.is_lt(Nat.add(c, y), Nat.add(Nat.min(c, n), y)) == False{} : Bool}:
  match c n:
    case 0n _:
      G.g_nat_lt_irrefl(y)
    case 1n+cp 0n:
      %L.nat_add_comm(y, cp) : {Nat.is_lt(1n+_, y) == False{} : Bool}
      meter_succ_add_not_lt(y, cp)
    case 1n+cp 1n+np:
      meter_min_add_not_over(cp, np, y)

# `n + (l - n)` is never less than `l`.
def meter_credit_floor(+l: Nat, +n: Nat) -> {Nat.is_lt(Nat.add(n, Nat.sub(l, n)), l) == False{} : Bool}:
  match l n:
    case 0n 0n:
      {==}
    case 0n 1n+np:
      {==}
    case 1n+lp 0n:
      G.g_nat_lt_irrefl(lp)
    case 1n+lp 1n+np:
      meter_credit_floor(lp, np)

# `(w + l + n) - w` is `l + n`.
def meter_rebill_cancel(+w: Nat, +l: Nat, +n: Nat) -> {Nat.sub(Nat.add(Nat.add(w, l), n), w) == Nat.add(l, n) : Nat}:
  match w:
    case 0n:
      L.nat_sub_zero_r(Nat.add(l, n))
    case 1n+wp:
      meter_rebill_cancel(wp, l, n)

# ============================================================================
# 3. Laws
# ============================================================================

# promise: a frame never owns a block recorded before it started, so releasing memory allocated before the meter (or by an enclosing frame) never lowers its count.
# harm: a request could free its own body's buffers to cancel the heap its documents hold and keep more than its cap.
# catches: the start bound dropped (Nat.is_le(start, serial) replaced by True{})
law meter_owns_earlier_refused:
  for +s: Nat
  for +d: Nat
  for later: Bool
  for +next: Nat
  {meter_owns(Nat.add(s, 1n+d), s, later, next) == False{} : Bool}

def meter_owns_earlier_refused(s, d, later, next):
  meter_and_false_l(Nat.is_le(Nat.add(s, 1n+d), s), Bool.or(Bool.not(later), Nat.is_lt(s, next)), G.g_nat_le_add_succ_false(s, d))

# promise: while a frame started inside another is on, the outer frame never owns a block recorded by the inner one.
# harm: a release inside a document read could credit the request's count for the read's own block and let the two counts drift apart.
# catches: the later frame ignored (Bool.or(Bool.not(later), ...) replaced by True{})
law meter_owns_later_refused:
  for +start: Nat
  for +next: Nat
  for +d: Nat
  {meter_owns(start, Nat.add(next, d), True{}, next) == False{} : Bool}

def meter_owns_later_refused(start, next, d):
  meter_and_false_r(Nat.is_le(start, Nat.add(next, d)), Nat.is_lt(Nat.add(next, d), next), G.g_nat_lt_add_r_false(next, d))

# promise: the top frame owns every block recorded since it started.
# catches: ownership refused for every block (the body replaced by False{})
law meter_owns_own_block:
  for +s: Nat
  for +d: Nat
  for +next: Nat
  {meter_owns(s, Nat.add(s, d), False{}, next) == True{} : Bool}

def meter_owns_own_block(s, d, next):
  G.g_bool_and_tt(Nat.is_le(s, Nat.add(s, d)), True{}, L.nat_add_le_r(s, d), {==})

# promise: a frame with another frame on top of it owns the blocks recorded between its own start and that frame's start.
# catches: the comparison with the next frame's start reversed (Nat.is_lt(serial, next) replaced by Nat.is_lt(next, serial))
law meter_owns_before_next:
  for +s: Nat
  for +d: Nat
  for +e: Nat
  {meter_owns(s, Nat.add(s, d), True{}, Nat.add(Nat.add(s, d), 1n+e)) == True{} : Bool}

def meter_owns_before_next(s, d, e):
  G.g_bool_and_tt(Nat.is_le(s, Nat.add(s, d)), Nat.is_lt(Nat.add(s, d), Nat.add(Nat.add(s, d), 1n+e)), L.nat_add_le_r(s, d), G.g_nat_lt_add_succ(Nat.add(s, d), e))

# promise: a reallocated block is never charged for more bytes than its new size, when it was charged no more than its old size.
# harm: a charge above the block's size would later be credited against blocks the frame still holds.
# catches: growth charged at the new size (Nat.sub(new, old) replaced by new)
law meter_regrow_inside:
  for +c: Nat
  for +o: Nat
  for +n: Nat
  {Bool.and(Nat.is_le(c, o), Nat.is_lt(n, meter_regrow(c, o, n))) == False{} : Bool}

def meter_regrow_inside(c, o, n):
  match c o n:
    case 0n _ _:
      meter_and_false_r(Nat.is_le(0n, o), Nat.is_lt(n, Nat.sub(n, o)), L.nat_le_lt_false(n, Nat.sub(n, o), G.g_nat_sub_le_self(n, o)))
    case 1n+cp 0n _:
      {==}
    case 1n+cp 1n+op 0n:
      G.g_bool_and_false_r(Nat.is_le(cp, op))
    case 1n+cp 1n+op 1n+np:
      meter_regrow_inside(cp, op, np)

# promise: a reallocation charges every byte it adds and credits no more than the bytes it removes (new charge plus bytes removed is at least old charge plus bytes added).
# harm: growing a block under the meter without charging the growth, or shrinking it to credit more than it gave back, would let a document hold heap its count does not show.
# catches: growth not charged (Nat.add(Nat.min(charge, new), Nat.sub(new, old)) replaced by Nat.min(charge, new))
law meter_regrow_charges:
  for +c: Nat
  for +o: Nat
  for +n: Nat
  {Bool.and(Nat.is_le(c, o), Nat.is_lt(Nat.add(meter_regrow(c, o, n), Nat.sub(o, n)), Nat.add(c, Nat.sub(n, o)))) == False{} : Bool}

def meter_regrow_charges(c, o, n):
  match c o n:
    case 0n _ _:
      meter_and_false_r(Nat.is_le(0n, o), Nat.is_lt(Nat.add(Nat.sub(n, o), Nat.sub(o, n)), Nat.sub(n, o)), G.g_nat_lt_add_r_false(Nat.sub(n, o), Nat.sub(o, n)))
    case 1n+cp 0n _:
      {==}
    case 1n+cp 1n+op 0n:
      %Equal.sym(Nat, Nat.add(cp, 0n), cp, L.nat_add_zero_r(cp)) : {Bool.and(Nat.is_le(cp, op), Nat.is_lt(op, _)) == False{} : Bool}
      meter_le_not_lt(cp, op, Nat.is_le(cp, op), {==})
    case 1n+cp 1n+op 1n+np:
      meter_regrow_charges(cp, op, np)

# promise: a reallocation never charges more than the block was charged plus the bytes it adds.
# harm: a block allocated before the meter would be charged its whole size when it grows, and a request refused for memory it never allocated.
# catches: the block charged its whole new size (the body replaced by new)
law meter_regrow_no_overcharge:
  for +c: Nat
  for +o: Nat
  for +n: Nat
  {Nat.is_lt(Nat.add(c, Nat.sub(n, o)), meter_regrow(c, o, n)) == False{} : Bool}

def meter_regrow_no_overcharge(c, o, n):
  meter_min_add_not_over(c, n, Nat.sub(n, o))

# promise: a charge followed by the credit of the same bytes leaves a frame's count as it was.
# harm: a count that drifts on every allocate-and-release pair would refuse a document for heap it gave back, or admit one past its cap.
# catches: the credit subtracts twice (meter_credit returning Nat.sub(Nat.sub(live, n), n))
law meter_charge_credit_conserved:
  for +l: Nat
  for +n: Nat
  {meter_credit(meter_charge(l, n), n) == l : Nat}

def meter_charge_credit_conserved(l, n):
  L.nat_add_sub_cancel_r(l, n)

# promise: a charge never leaves a frame's count below what it held plus the bytes charged.
# harm: an undercounted allocation lets a document hold heap past its cap without being refused.
# catches: add replaced by max (meter_charge returning Nat.max(live, n))
law meter_charge_counts_all:
  for +l: Nat
  for +n: Nat
  {Nat.is_lt(meter_charge(l, n), Nat.add(l, n)) == False{} : Bool}

def meter_charge_counts_all(l, n):
  G.g_nat_lt_irrefl(Nat.add(l, n))

# promise: a credit never raises a frame's count.
# catches: the credit adds (Nat.sub(live, n) replaced by Nat.add(live, n))
law meter_credit_never_raises:
  for +l: Nat
  for +n: Nat
  {Nat.is_lt(l, meter_credit(l, n)) == False{} : Bool}

def meter_credit_never_raises(l, n):
  L.nat_le_lt_false(l, Nat.sub(l, n), G.g_nat_sub_le_self(l, n))

# promise: a credit takes no more than the released block's charge off a frame's count.
# harm: a release that credits more than the block was charged would cancel heap the frame still holds and admit work past its cap.
# catches: the credit subtracts twice (meter_credit returning Nat.sub(Nat.sub(live, n), n))
law meter_credit_at_most_charge:
  for +l: Nat
  for +n: Nat
  {Nat.is_lt(Nat.add(n, meter_credit(l, n)), l) == False{} : Bool}

def meter_credit_at_most_charge(l, n):
  meter_credit_floor(l, n)

# promise: a frame holding a block charged w plus l other bytes holds l plus the block's new charge once the block is rebilled.
# harm: a reallocation that does not charge its growth, or credits more than it shrank, lets a document hold heap its count does not show.
# catches: the shrinkage not credited (meter_rebill returning meter_charge(live, Nat.sub(now, was)))
law meter_rebill_conserved:
  for +w: Nat
  for +l: Nat
  for +n: Nat
  {meter_rebill(Nat.add(w, l), w, n) == Nat.add(l, n) : Nat}

def meter_rebill_conserved(w, l, n):
  meter_rebill_cancel(w, l, n)

# promise: when a frame ends, the enclosing frame's count gains the whole count of the frame that ended.
# harm: the blocks a document read kept would leave the request's count, and the request could hold heap past its cap.
# catches: the fold drops the inner count (meter_fold returning outer)
law meter_fold_keeps_sum:
  for +o: Nat
  for +i: Nat
  {meter_fold(o, i) == Nat.add(o, i) : Nat}

def meter_fold_keeps_sum(o, i):
  {==}
