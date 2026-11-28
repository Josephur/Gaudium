# Slice — Verified Token Log-Probability Readout
# The decisions of the per-token log-probability readout (`logprobs` / `top_logprobs`):
#   - Whether a request's `top_logprobs` is accepted, and how many alternatives one token
#     reports (asked for, at most the readout's cap, at most the tokens the token was drawn from)
#   - Where row `i` of a readout lies in a copy of a launch's logits rows, and whether a copy of
#     `t` rows fits a host slot of `cap` rows
#   - The host ring: its rows (sized at load from the widest launch), its bytes, where each
#     launch parity's slot starts, and whether a copy fits its slot and its device source buffer
#   - How many rows of a speculative step are read (one per emitted token, never past the step)
#   - Whether a request keeps the decode path it would take without the readout (a path with no
#     readout is never taken by a request that asks for log-probabilities)

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. Helper lemmas (erased before Rust codegen)
# ============================================================================

def lp_min_of_le(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.min(a, b) == a : Nat}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      L.bool_absurd({Nat.min(1n+ap, 0n) == 1n+ap : Nat}, h)
    case 1n+ap 1n+bp:
      %Equal.sym(Nat, Nat.min(ap, bp), ap, lp_min_of_le(ap, bp, h)) : {1n+_ == 1n+ap : Nat}
      {==}

def lp_and_l(+a: Bool, b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      L.bool_absurd({False{} == True{} : Bool}, h)

def lp_and_r(+a: Bool, b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case True{}:
      h
    case False{}:
      L.bool_absurd({b == True{} : Bool}, h)

def lp_eq_refl(+a: Nat) -> {Nat.is_eq(a, a) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      lp_eq_refl(p)

# ============================================================================
# 2. Alternatives per token
# ============================================================================

# Most alternatives one token may report (the OpenAI `top_logprobs` range is 0..20).
def lp_top_cap() -> Nat:
  20n

# A request's `top_logprobs` is accepted.
def lp_top_ok(n: Nat) -> Bool:
  Nat.is_le(n, lp_top_cap())

# Alternatives reported for one token: `req` asked for, `support` tokens in the distribution the
# token was drawn from.
def lp_top_count(req: Nat, support: Nat) -> Nat:
  Nat.min(Nat.min(req, lp_top_cap()), support)

# promise: a token never reports more alternatives than the request asked for
# harm: a client reads entries it never asked for
# catches: count taken from the support alone (`support` in place of the min)
law lp_top_count_le_req:
  for r: Nat
  for s: Nat
  {Nat.is_le(lp_top_count(r, s), r) == True{} : Bool}

def lp_top_count_le_req(r, s):
  +r2 = {r : Nat}
  +s2 = {s : Nat}
  L.nat_le_trans(Nat.min(Nat.min(r2, 20n), s2), Nat.min(r2, 20n), r2, L.nat_min_le_l(Nat.min(r2, 20n), s2), L.nat_min_le_l(r2, 20n))

# promise: a token never reports more alternatives than the readout's cap, whatever was asked
# harm: the readout's fixed-size top list is indexed past its end
# catches: the cap dropped (`Nat.min(req, support)`)
law lp_top_count_le_cap:
  for r: Nat
  for s: Nat
  {Nat.is_le(lp_top_count(r, s), lp_top_cap()) == True{} : Bool}

def lp_top_count_le_cap(r, s):
  +r2 = {r : Nat}
  +s2 = {s : Nat}
  L.nat_le_trans(Nat.min(Nat.min(r2, 20n), s2), Nat.min(r2, 20n), 20n, L.nat_min_le_l(Nat.min(r2, 20n), s2), L.nat_min_le_r(r2, 20n))

# promise: a token never reports more alternatives than the distribution it was drawn from holds
# harm: tokens outside the sampled support are reported as alternatives of the draw
# catches: the support bound dropped (`Nat.min(req, cap)`)
law lp_top_count_le_support:
  for r: Nat
  for s: Nat
  {Nat.is_le(lp_top_count(r, s), s) == True{} : Bool}

def lp_top_count_le_support(r, s):
  L.nat_min_le_r(Nat.min(r, 20n), s)

def lp_covers_go(
  +r: Nat,
  +s: Nat,
  b: Bool,
  +eb: {b == lp_top_ok(r) : Bool}
) -> {Bool.or(Bool.not(lp_top_ok(r)), Nat.is_le(Nat.min(r, s), lp_top_count(r, s))) == True{} : Bool}:
  match b:
    case False{}:
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.min(r, s), lp_top_count(r, s))) == True{} : Bool}
      {==}
    case True{}:
      +h = Equal.sym(Bool, True{}, lp_top_ok(r), eb)
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.min(r, s), lp_top_count(r, s))) == True{} : Bool}
      %Equal.sym(Nat, Nat.min(r, 20n), r, lp_min_of_le(r, 20n, h)) : {Nat.is_le(Nat.min(r, s), Nat.min(_, s)) == True{} : Bool}
      L.nat_le_refl(Nat.min(r, s))

# promise: an accepted request gets every alternative it asked for, or every token of the draw when the draw holds fewer
# harm: a client asking for alternatives silently gets fewer than the distribution holds
# catches: no alternatives when the support is smaller than the request (`Bool.pick(Nat, Nat.is_le(req, support), Nat.min(req, lp_top_cap()), 0n)`)
law lp_top_count_covers:
  for r: Nat
  for s: Nat
  {Bool.or(Bool.not(lp_top_ok(r)), Nat.is_le(Nat.min(r, s), lp_top_count(r, s))) == True{} : Bool}

def lp_top_count_covers(r, s):
  +r2 = {r : Nat}
  +s2 = {s : Nat}
  lp_covers_go(r2, s2, lp_top_ok(r2), {==})

# ============================================================================
# 3. Rows of a readout copy
# ============================================================================

# First byte of row `i` in a copy of a launch's logits rows (`rb` bytes per row).
def lp_row_start(i: Nat, rb: Nat) -> Nat:
  Nat.mul(i, rb)

# Bytes of a copy of `t` logits rows.
def lp_copy_bytes(t: Nat, rb: Nat) -> Nat:
  Nat.mul(t, rb)

# Bytes of a host slot that holds `cap` logits rows.
def lp_slot_bytes(cap: Nat, rb: Nat) -> Nat:
  Nat.mul(cap, rb)

# A copy of `t` rows may go into a slot of `cap` rows.
def lp_copy_fits(t: Nat, cap: Nat) -> Bool:
  Nat.is_le(t, cap)

# promise: every row of a t-row readout copy, i < t, ends inside the copy
# harm: a token's log-probabilities are read from the next row or past the copied bytes
# catches: rows counted from one (`Nat.mul(1n+i, rb)`)
law lp_row_in_copy:
  for i: Nat
  for d: Nat
  for rb: Nat
  {Nat.is_le(Nat.add(lp_row_start(i, rb), rb), lp_copy_bytes(Nat.add(1n+i, d), rb)) == True{} : Bool}

def lp_row_in_copy(i, d, rb):
  +i2 = {i : Nat}
  +d2 = {d : Nat}
  +rb2 = {rb : Nat}
  %L.nat_add_comm(rb2, Nat.mul(i2, rb2)) : {Nat.is_le(_, Nat.add(rb2, Nat.mul(Nat.add(i2, d2), rb2))) == True{} : Bool}
  L.nat_add_le_mono(rb2, rb2, Nat.mul(i2, rb2), Nat.mul(Nat.add(i2, d2), rb2), L.nat_le_refl(rb2), L.nat_mul_le_mono_r(i2, Nat.add(i2, d2), rb2, L.nat_add_le_r(i2, d2)))

# promise: row i+1 of a readout copy starts where row i ends or later, so no two rows share a byte
# harm: every token reads its log-probabilities from the same row
# catches: a constant row start (`0n`)
law lp_rows_disjoint:
  for i: Nat
  for rb: Nat
  {Nat.is_le(Nat.add(lp_row_start(i, rb), rb), lp_row_start(1n+i, rb)) == True{} : Bool}

def lp_rows_disjoint(i, rb):
  +i2 = {i : Nat}
  +rb2 = {rb : Nat}
  %L.nat_add_comm(rb2, Nat.mul(i2, rb2)) : {Nat.is_le(_, Nat.add(rb2, Nat.mul(i2, rb2))) == True{} : Bool}
  L.nat_le_refl(Nat.add(rb2, Nat.mul(i2, rb2)))

def lp_fit_go(
  +t: Nat,
  +cap: Nat,
  +rb: Nat,
  b: Bool,
  +eb: {b == Nat.is_le(t, cap) : Bool}
) -> {Bool.or(Bool.not(lp_copy_fits(t, cap)), Nat.is_le(lp_copy_bytes(t, rb), lp_slot_bytes(cap, rb))) == True{} : Bool}:
  match b:
    case False{}:
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.mul(t, rb), Nat.mul(cap, rb))) == True{} : Bool}
      {==}
    case True{}:
      +h = Equal.sym(Bool, True{}, Nat.is_le(t, cap), eb)
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.mul(t, rb), Nat.mul(cap, rb))) == True{} : Bool}
      L.nat_mul_le_mono_r(t, cap, rb, h)

# promise: a copy the readout accepts never writes past the host slot it goes into
# harm: the device writes logits rows over host memory that is not the slot
# catches: the fit test loosened to `Nat.is_le(t, 1n+cap)`
law lp_copy_in_slot:
  for t: Nat
  for cap: Nat
  for rb: Nat
  {Bool.or(Bool.not(lp_copy_fits(t, cap)), Nat.is_le(lp_copy_bytes(t, rb), lp_slot_bytes(cap, rb))) == True{} : Bool}

def lp_copy_in_slot(t, cap, rb):
  +t2 = {t : Nat}
  +cap2 = {cap : Nat}
  +rb2 = {rb : Nat}
  lp_fit_go(t2, cap2, rb2, Nat.is_le(t2, cap2), {==})

# promise: a slot takes every copy no wider than the rows it was sized for
# harm: logprob requests on narrower verify widths are refused
# catches: the fit test narrowed to equality (`Nat.is_eq(t, cap)`)
law lp_copy_fits_narrower:
  for t: Nat
  for d: Nat
  {lp_copy_fits(t, Nat.add(t, d)) == True{} : Bool}

def lp_copy_fits_narrower(t, d):
  L.nat_add_le_r(t, d)

# ============================================================================
# 3b. The host ring and the device side of a copy
# ============================================================================

# Rows of each slot of the host ring: the widest verify launch (`blk` rows, 0 without a drafter)
# and the one-row steps of the non-speculative loop. Sized once, when the model loads.
def lp_ring_rows(blk: Nat) -> Nat:
  Nat.max(1n, blk)

# Bytes of the host ring: one slot of `cap` rows per launch parity.
def lp_ring_bytes(+cap: Nat, +rb: Nat) -> Nat:
  Nat.add(lp_slot_bytes(cap, rb), lp_slot_bytes(cap, rb))

# First byte of slot `p` (the launch parity) in the host ring.
def lp_slot_start(p: Nat, cap: Nat, rb: Nat) -> Nat:
  Nat.mul(p, lp_slot_bytes(cap, rb))

# A copy of `t` rows may go into slot `p` of a ring of `cap`-row slots.
def lp_slot_ok(p: Nat, t: Nat, cap: Nat) -> Bool:
  Bool.and(Nat.is_lt(p, 2n), lp_copy_fits(t, cap))

# A copy of `t` rows (`rb` bytes each) may be read from a device logits buffer of `src` bytes.
def lp_src_fits(t: Nat, rb: Nat, src: Nat) -> Bool:
  Nat.is_le(lp_copy_bytes(t, rb), src)

def lp_and_intro(+a: Bool, b: Bool, ha: {a == True{} : Bool}, hb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  match a:
    case True{}:
      hb
    case False{}:
      L.bool_absurd({Bool.and(False{}, b) == True{} : Bool}, ha)

# promise: every slot of the ring takes a one-row step and the widest verify launch it was sized for
# harm: a logprob request on a loaded engine is refused, or its copy overruns the slot
# catches: the one-row floor dropped (`blk` alone)
law lp_ring_rows_cover:
  for b: Nat
  {Bool.and(lp_copy_fits(1n, lp_ring_rows(b)), lp_copy_fits(b, lp_ring_rows(b))) == True{} : Bool}

def lp_ring_rows_cover(b):
  +b2 = {b : Nat}
  lp_and_intro(Nat.is_le(1n, Nat.max(1n, b2)), Nat.is_le(b2, Nat.max(1n, b2)), L.nat_max_ge_l(1n, b2), L.nat_max_ge_r(1n, b2))

def lp_ring_at(
  +p: Nat,
  +t: Nat,
  +cap: Nat,
  +rb: Nat,
  hp: {Nat.is_lt(p, 2n) == True{} : Bool},
  hf: {Nat.is_le(t, cap) == True{} : Bool}
) -> {Nat.is_le(Nat.add(lp_slot_start(p, cap, rb), lp_copy_bytes(t, rb)), lp_ring_bytes(cap, rb)) == True{} : Bool}:
  match p:
    case 0n:
      L.nat_le_trans(Nat.mul(t, rb), Nat.mul(cap, rb), Nat.add(Nat.mul(cap, rb), Nat.mul(cap, rb)), L.nat_mul_le_mono_r(t, cap, rb, hf), L.nat_add_le_r(Nat.mul(cap, rb), Nat.mul(cap, rb)))
    case 1n+pp:
      match pp:
        case 0n:
          %Equal.sym(Nat, Nat.add(Nat.mul(cap, rb), 0n), Nat.mul(cap, rb), L.nat_add_zero_r(Nat.mul(cap, rb))) : {Nat.is_le(Nat.add(_, Nat.mul(t, rb)), Nat.add(Nat.mul(cap, rb), Nat.mul(cap, rb))) == True{} : Bool}
          L.nat_add_le_mono(Nat.mul(cap, rb), Nat.mul(cap, rb), Nat.mul(t, rb), Nat.mul(cap, rb), L.nat_le_refl(Nat.mul(cap, rb)), L.nat_mul_le_mono_r(t, cap, rb, hf))
        case 1n+q:
          G.g_nat_lt_zero_absurd({Nat.is_le(Nat.add(lp_slot_start(2n+q, cap, rb), lp_copy_bytes(t, rb)), lp_ring_bytes(cap, rb)) == True{} : Bool}, q, hp)

def lp_ring_go(
  +p: Nat,
  +t: Nat,
  +cap: Nat,
  +rb: Nat,
  b: Bool,
  +eb: {b == Bool.and(Nat.is_lt(p, 2n), Nat.is_le(t, cap)) : Bool}
) -> {Bool.or(Bool.not(lp_slot_ok(p, t, cap)), Nat.is_le(Nat.add(lp_slot_start(p, cap, rb), lp_copy_bytes(t, rb)), lp_ring_bytes(cap, rb))) == True{} : Bool}:
  match b:
    case False{}:
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.add(lp_slot_start(p, cap, rb), lp_copy_bytes(t, rb)), lp_ring_bytes(cap, rb))) == True{} : Bool}
      {==}
    case True{}:
      +h = Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(p, 2n), Nat.is_le(t, cap)), eb)
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.add(lp_slot_start(p, cap, rb), lp_copy_bytes(t, rb)), lp_ring_bytes(cap, rb))) == True{} : Bool}
      lp_ring_at(p, t, cap, rb, lp_and_l(Nat.is_lt(p, 2n), Nat.is_le(t, cap), h), lp_and_r(Nat.is_lt(p, 2n), Nat.is_le(t, cap), h))

# promise: a copy the readout accepts for a slot never writes past the end of the host ring
# harm: the device writes logits rows over host memory that is not the ring
# catches: a ring of one slot (`lp_slot_bytes(cap, rb)`)
law lp_copy_in_ring:
  for p: Nat
  for t: Nat
  for cap: Nat
  for rb: Nat
  {Bool.or(Bool.not(lp_slot_ok(p, t, cap)), Nat.is_le(Nat.add(lp_slot_start(p, cap, rb), lp_copy_bytes(t, rb)), lp_ring_bytes(cap, rb))) == True{} : Bool}

def lp_copy_in_ring(p, t, cap, rb):
  +p2 = {p : Nat}
  +t2 = {t : Nat}
  +cap2 = {cap : Nat}
  +rb2 = {rb : Nat}
  lp_ring_go(p2, t2, cap2, rb2, Bool.and(Nat.is_lt(p2, 2n), Nat.is_le(t2, cap2)), {==})

def lp_slots_go(
  +t: Nat,
  +cap: Nat,
  +rb: Nat,
  b: Bool,
  +eb: {b == Nat.is_le(t, cap) : Bool}
) -> {Bool.or(Bool.not(lp_slot_ok(0n, t, cap)), Nat.is_le(Nat.add(lp_slot_start(0n, cap, rb), lp_copy_bytes(t, rb)), lp_slot_start(1n, cap, rb))) == True{} : Bool}:
  match b:
    case False{}:
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.mul(t, rb), Nat.add(Nat.mul(cap, rb), 0n))) == True{} : Bool}
      {==}
    case True{}:
      +h = Equal.sym(Bool, True{}, Nat.is_le(t, cap), eb)
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.mul(t, rb), Nat.add(Nat.mul(cap, rb), 0n))) == True{} : Bool}
      L.nat_le_subst_r(Nat.mul(t, rb), Nat.mul(cap, rb), Nat.add(Nat.mul(cap, rb), 0n), Equal.sym(Nat, Nat.add(Nat.mul(cap, rb), 0n), Nat.mul(cap, rb), L.nat_add_zero_r(Nat.mul(cap, rb))), L.nat_mul_le_mono_r(t, cap, rb, h))

# promise: a copy the readout accepts for slot 0 ends before slot 1 starts, so the two launches in flight never share a host byte
# harm: the launch queued behind overwrites the rows the host is still reading
# catches: both parities at the ring's start (`0n` in place of the slot start)
law lp_slots_disjoint:
  for t: Nat
  for cap: Nat
  for rb: Nat
  {Bool.or(Bool.not(lp_slot_ok(0n, t, cap)), Nat.is_le(Nat.add(lp_slot_start(0n, cap, rb), lp_copy_bytes(t, rb)), lp_slot_start(1n, cap, rb))) == True{} : Bool}

def lp_slots_disjoint(t, cap, rb):
  +t2 = {t : Nat}
  +cap2 = {cap : Nat}
  +rb2 = {rb : Nat}
  lp_slots_go(t2, cap2, rb2, Nat.is_le(t2, cap2), {==})

# promise: both launch parities take every copy no wider than the slots
# harm: logprob requests are refused on one parity, so every other step fails
# catches: parity 1 refused (`Nat.is_lt(p, 1n)`)
law lp_slot_ok_parities:
  for t: Nat
  for d: Nat
  {Bool.and(lp_slot_ok(0n, t, Nat.add(t, d)), lp_slot_ok(1n, t, Nat.add(t, d))) == True{} : Bool}

def lp_slot_ok_parities(t, d):
  +t2 = {t : Nat}
  +d2 = {d : Nat}
  lp_and_intro(Nat.is_le(t2, Nat.add(t2, d2)), Nat.is_le(t2, Nat.add(t2, d2)), L.nat_add_le_r(t2, d2), L.nat_add_le_r(t2, d2))

def lp_src_go(
  +t: Nat,
  +rb: Nat,
  +s: Nat,
  b: Bool,
  +eb: {b == Nat.is_le(Nat.mul(t, rb), s) : Bool}
) -> {Bool.or(Bool.not(lp_src_fits(t, rb, s)), Nat.is_le(lp_copy_bytes(t, rb), s)) == True{} : Bool}:
  match b:
    case False{}:
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.mul(t, rb), s)) == True{} : Bool}
      {==}
    case True{}:
      +h = Equal.sym(Bool, True{}, Nat.is_le(Nat.mul(t, rb), s), eb)
      %eb : {Bool.or(Bool.not(_), Nat.is_le(Nat.mul(t, rb), s)) == True{} : Bool}
      h

# promise: a copy the readout accepts never reads past the end of the device logits buffer it copies from
# harm: rows of another tensor, or past the buffer, are read back as a token's logits
# catches: rows compared with bytes (`Nat.is_le(t, src)`)
law lp_src_in_buffer:
  for t: Nat
  for rb: Nat
  for s: Nat
  {Bool.or(Bool.not(lp_src_fits(t, rb, s)), Nat.is_le(lp_copy_bytes(t, rb), s)) == True{} : Bool}

def lp_src_in_buffer(t, rb, s):
  +t2 = {t : Nat}
  +rb2 = {rb : Nat}
  +s2 = {s : Nat}
  lp_src_go(t2, rb2, s2, Nat.is_le(Nat.mul(t2, rb2), s2), {==})

# promise: a buffer takes every copy no larger than it
# harm: logprob requests are refused although the launch's rows are all in the buffer
# catches: the fit test made strict (`Nat.is_lt` in place of `Nat.is_le`)
law lp_src_fits_narrower:
  for t: Nat
  for rb: Nat
  for d: Nat
  {lp_src_fits(t, rb, Nat.add(lp_copy_bytes(t, rb), d)) == True{} : Bool}

def lp_src_fits_narrower(t, rb, d):
  L.nat_add_le_r(Nat.mul(t, rb), d)

# ============================================================================
# 4. Rows read per step
# ============================================================================

# Rows of a step read for its `used` emitted tokens (`t` rows computed).
def lp_step_rows(used: Nat, t: Nat) -> Nat:
  Nat.min(used, t)

# promise: a step never has more of its rows read than it computed
# harm: tokens get log-probabilities from rows the step never wrote
# catches: the emitted count taken as is (`used`)
law lp_step_rows_le_rows:
  for u: Nat
  for t: Nat
  {Nat.is_le(lp_step_rows(u, t), t) == True{} : Bool}

def lp_step_rows_le_rows(u, t):
  L.nat_min_le_r(u, t)

def lp_rows_go(
  +u: Nat,
  +t: Nat,
  b: Bool,
  +eb: {b == Nat.is_le(u, t) : Bool}
) -> {Bool.or(Bool.not(Nat.is_le(u, t)), Nat.is_eq(lp_step_rows(u, t), u)) == True{} : Bool}:
  match b:
    case False{}:
      %eb : {Bool.or(Bool.not(_), Nat.is_eq(Nat.min(u, t), u)) == True{} : Bool}
      {==}
    case True{}:
      +h = Equal.sym(Bool, True{}, Nat.is_le(u, t), eb)
      %eb : {Bool.or(Bool.not(_), Nat.is_eq(Nat.min(u, t), u)) == True{} : Bool}
      %Equal.sym(Nat, Nat.min(u, t), u, lp_min_of_le(u, t, h)) : {Nat.is_eq(_, u) == True{} : Bool}
      lp_eq_refl(u)

# promise: every emitted token of a step gets its own row read when the step computed it
# harm: a token is reported without its log-probability
# catches: one row fewer (`Nat.pred(Nat.min(used, t))`)
law lp_step_rows_cover:
  for u: Nat
  for t: Nat
  {Bool.or(Bool.not(Nat.is_le(u, t)), Nat.is_eq(lp_step_rows(u, t), u)) == True{} : Bool}

def lp_step_rows_cover(u, t):
  +u2 = {u : Nat}
  +t2 = {t : Nat}
  lp_rows_go(u2, t2, Nat.is_le(u2, t2), {==})

# ============================================================================
# 5. Decode path of a request
# ============================================================================

# The request keeps the decode path it would take without the readout (`False`: it runs the
# non-speculative path, whose tokens are the reference every path equals). `want`: the request
# asks for log-probabilities; `readout`: the path reads its rows out.
def lp_path_keeps(want: Bool, readout: Bool) -> Bool:
  Bool.or(Bool.not(want), readout)

# promise: a request that asks for log-probabilities never runs a path that cannot read them out
# harm: tokens are emitted with no log-probabilities although the client asked for them
# catches: the path kept unconditionally (`True{}`)
law lp_path_needs_readout:
  for w: Bool
  {lp_path_keeps(True{}, False{}) == False{} : Bool}

def lp_path_needs_readout(w):
  {==}

# promise: a request that does not ask for log-probabilities keeps its decode path, so it pays nothing for the readout
# harm: requests without logprobs are moved off the speculative path and slow down
# catches: the readout required for every request (`readout` alone)
law lp_path_unasked_keeps:
  for r: Bool
  {lp_path_keeps(False{}, r) == True{} : Bool}

def lp_path_unasked_keeps(r):
  {==}

# promise: a request that asks for log-probabilities keeps a path that reads them out
# harm: every logprob request falls back to the slow path
# catches: the path refused to every request that asks (`Bool.not(want)`)
law lp_path_readout_keeps:
  for w: Bool
  {lp_path_keeps(True{}, True{}) == True{} : Bool}

def lp_path_readout_keeps(w):
  {==}
