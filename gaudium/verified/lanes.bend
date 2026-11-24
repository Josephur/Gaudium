# Lanes: several requests served at once by one loaded engine (`--parallel N`).
# Formally specifies and proves:
#   - The lifecycle of one lane (free, prefill, decode, drain): a lane holds at most one request,
#     is freed only after its request ended, and only a decoding lane takes part in a step
#   - Where a request goes: the free lane that already holds the most of its prompt
#   - The device plan of N lanes beside the shared buffers, and its exact shortfall
#   - The slot table of a batched step: every slot's rows lie inside the chunks the step covers
#     and inside the slot's own lane
#
# Admission itself (how many lanes may be busy, an image request alone) is `lane_can_admit` in
# `lifecycle.bend`.

import Base
import nat_lemmas.bend as L

# ============================================================================
# 1. Lane lifecycle
# ============================================================================

type LaneState is Data:
  LnFree{}
  LnPrefill{}
  LnDecode{}
  LnDrain{}

type LaneEvent is Data:
  EvLnAdmit{}
  EvLnPrefilled{}
  EvLnFinish{}
  EvLnFail{}
  EvLnRelease{}

# LnFree     no request (the lane may still hold the K/V rows and state of its last prompt)
# LnPrefill  its request's prompt is being prefilled
# LnDecode   its request decodes, one or more tokens per step
# LnDrain    its request ended (stop, length, cancel or an error); the report is not delivered yet
def ln_step(s: LaneState, ev: LaneEvent) -> LaneState:
  match s ev:
    case LnFree{} EvLnAdmit{}:
      LnPrefill{}
    case LnPrefill{} EvLnPrefilled{}:
      LnDecode{}
    case LnPrefill{} EvLnFinish{}:
      LnDrain{}
    case LnPrefill{} EvLnFail{}:
      LnDrain{}
    case LnDecode{} EvLnFinish{}:
      LnDrain{}
    case LnDecode{} EvLnFail{}:
      LnDrain{}
    case LnDrain{} EvLnRelease{}:
      LnFree{}
    case st _:
      st

def ln_is_free(s: LaneState) -> Bool:
  match s:
    case LnFree{}:
      True{}
    case _:
      False{}

def ln_is_drain(s: LaneState) -> Bool:
  match s:
    case LnDrain{}:
      True{}
    case _:
      False{}

# Lanes that run in a decode step.
def ln_in_batch(s: LaneState) -> Bool:
  match s:
    case LnDecode{}:
      True{}
    case _:
      False{}

# A free lane is where routing may send a new request.
def ln_route_eligible(s: LaneState) -> Bool:
  ln_is_free(s)

def ln_eq(a: LaneState, b: LaneState) -> Bool:
  match a b:
    case LnFree{} LnFree{}:
      True{}
    case LnPrefill{} LnPrefill{}:
      True{}
    case LnDecode{} LnDecode{}:
      True{}
    case LnDrain{} LnDrain{}:
      True{}
    case _ _:
      False{}

def ln_ev_is_admit(ev: LaneEvent) -> Bool:
  match ev:
    case EvLnAdmit{}:
      True{}
    case _:
      False{}

# ============================================================================
# 2. Routing a request to a lane
# ============================================================================

# Is free lane `a` (it would take over `ra` rows of the request's prompt, last used at `sa`) a
# better place for a request than free lane `b`? More rows taken over wins; between equals, the
# lane used longest ago, so the conversations used most recently keep their rows.
def ln_route_better(+ra: Nat, +sa: Nat, +rb: Nat, +sb: Nat) -> Bool:
  Bool.or(Nat.is_gt(ra, rb), Bool.and(Nat.is_eq(ra, rb), Nat.is_lt(sa, sb)))

# ============================================================================
# 3. The device plan of N lanes
# ============================================================================

# Bytes of `n` lanes of `lane_bytes` each beside `shared` bytes every lane uses.
def ln_plan_bytes(n: Nat, lane_bytes: Nat, shared: Nat) -> Nat:
  Nat.add(shared, Nat.mul(n, lane_bytes))

def ln_plan_fits(n: Nat, lane_bytes: Nat, shared: Nat, usable: Nat) -> Bool:
  Nat.is_le(ln_plan_bytes(n, lane_bytes, shared), usable)

# What a refused plan lacks (0 for a plan that fits).
def ln_plan_shortfall(n: Nat, lane_bytes: Nat, shared: Nat, usable: Nat) -> Nat:
  Nat.sub(ln_plan_bytes(n, lane_bytes, shared), usable)

# ============================================================================
# 4. The slot table of a batched step
# ============================================================================
#
# A step over `nch` chunks of `chunk` rows binds, for every slot, the first `nch` chunks of that
# slot's lane. A slot whose request writes rows up to `cursor + rows` (exclusive) needs them all
# inside those chunks, and the chunks inside its lane (`lane_chunks` chunks).

# Chunks that hold rows [0, cursor + rows): at least one.
def ln_slot_chunks(+cursor: Nat, +rows: Nat, +chunk: Nat) -> Nat:
  Nat.max(1n, Nat.div(Nat.add(Nat.add(cursor, rows), Nat.sub(chunk, 1n)), chunk))

# The chunks a step covers after one more slot is folded in (start from 0).
def ln_batch_chunks(+acc: Nat, +cursor: Nat, +rows: Nat, +chunk: Nat) -> Nat:
  Nat.max(acc, ln_slot_chunks(cursor, rows, chunk))

# May a slot run in a step over `nch` chunks? Its rows inside the chunks, the chunks inside its lane.
def ln_slot_fits(+cursor: Nat, +rows: Nat, +nch: Nat, +chunk: Nat, +lane_chunks: Nat) -> Bool:
  Bool.and(Nat.is_le(nch, lane_chunks), Nat.is_le(Nat.add(cursor, rows), Nat.mul(nch, chunk)))

# ============================================================================
# 4b. Which models are served with lanes
# ============================================================================

# How a model decodes: plainly, with a DSpine drafter, or with another drafter.
type LaneDrafter is Data:
  LdNone{}
  LdDSpine{}
  LdOther{}

# Which kinds of decode have batched steps for several lanes.
def ln_drafter_ok(d: LaneDrafter) -> Bool:
  match d:
    case LdNone{}:
      True{}
    case _:
      False{}

# Lanes a model is served with: the lanes asked for (at least one) when its engine and its
# drafter have batched steps, otherwise one.
def ln_served_lanes(+requested: Nat, ok: Bool) -> Nat:
  match ok:
    case True{}:
      Nat.max(1n, requested)
    case False{}:
      1n

# ============================================================================
# 5. Helper lemmas (erased before Rust codegen)
# ============================================================================

def ln_gt_succ_add(+b: Nat, +e: Nat) -> {True{} == Nat.is_gt(1n+Nat.add(b, e), b) : Bool}:
  match b:
    case 0n:
      {==}
    case 1n+bp:
      ln_gt_succ_add(bp, e)

def ln_gt_refl_false(+n: Nat) -> {False{} == Nat.is_gt(n, n) : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      ln_gt_refl_false(p)

def ln_lt_refl_false(+n: Nat) -> {False{} == Nat.is_lt(n, n) : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      ln_lt_refl_false(p)

def ln_eq_refl_true(+n: Nat) -> {True{} == Nat.is_eq(n, n) : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      ln_eq_refl_true(p)

def ln_le_iff_sub_zero(+a: Nat, +b: Nat) -> {Nat.is_le(a, b) == Nat.is_eq(Nat.sub(a, b), 0n) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      ln_le_iff_sub_zero(ap, bp)

def ln_fit_within(
  +c: Nat,
  +r: Nat,
  +n: Nat,
  +ch: Nat,
  +lc: Nat,
  b1: Bool,
  e1: {Nat.is_le(n, lc) == b1 : Bool},
  b2: Bool,
  e2: {Nat.is_le(Nat.add(c, r), Nat.mul(n, ch)) == b2 : Bool}
) -> {Bool.and(Bool.and(b1, b2), Nat.is_lt(Nat.mul(lc, ch), Nat.add(c, r))) == False{} : Bool}:
  match b1:
    case False{}:
      {==}
    case True{}:
      match b2:
        case False{}:
          {==}
        case True{}:
          L.nat_le_lt_false(Nat.mul(lc, ch), Nat.add(c, r), L.nat_le_trans(Nat.add(c, r), Nat.mul(n, ch), Nat.mul(lc, ch), e2, L.nat_mul_le_mono_r(n, lc, ch, e1)))

def ln_fit_covered(
  +c: Nat,
  +r: Nat,
  +n: Nat,
  +ch: Nat,
  b1: Bool,
  b2: Bool,
  e2: {Nat.is_le(Nat.add(c, r), Nat.mul(n, ch)) == b2 : Bool}
) -> {Bool.and(Bool.and(b1, b2), Nat.is_lt(Nat.mul(n, ch), Nat.add(c, r))) == False{} : Bool}:
  match b1:
    case False{}:
      {==}
    case True{}:
      match b2:
        case False{}:
          {==}
        case True{}:
          L.nat_le_lt_false(Nat.mul(n, ch), Nat.add(c, r), e2)

# ============================================================================
# 6. Laws
# ============================================================================

# promise: a lane that holds a request is never given a second one: admission changes only a free lane.
law ln_busy_ignores_admit:
  for +s: LaneState
  {Bool.or(ln_is_free(s), ln_eq(ln_step(s, EvLnAdmit{}), s)) == True{} : Bool}

def ln_busy_ignores_admit(s):
  match s:
    case LnFree{}: {==}
    case LnPrefill{}: {==}
    case LnDecode{}: {==}
    case LnDrain{}: {==}

# promise: admitting a request to a free lane makes the lane busy.
law ln_admit_takes_lane:
  {ln_is_free(ln_step(LnFree{}, EvLnAdmit{})) == False{} : Bool}

def ln_admit_takes_lane():
  {==}

# promise: a free lane leaves the free state only when a request is admitted to it.
law ln_free_left_only_by_admit:
  for +ev: LaneEvent
  {Bool.or(ln_ev_is_admit(ev), ln_is_free(ln_step(LnFree{}, ev))) == True{} : Bool}

def ln_free_left_only_by_admit(ev):
  match ev:
    case EvLnAdmit{}: {==}
    case EvLnPrefilled{}: {==}
    case EvLnFinish{}: {==}
    case EvLnFail{}: {==}
    case EvLnRelease{}: {==}

# promise: a lane is freed only after its request ended, never while it prefills or decodes.
law ln_release_only_drained:
  for +s: LaneState
  {Bool.or(ln_is_drain(s), ln_eq(ln_step(s, EvLnRelease{}), s)) == True{} : Bool}

def ln_release_only_drained(s):
  match s:
    case LnFree{}: {==}
    case LnPrefill{}: {==}
    case LnDecode{}: {==}
    case LnDrain{}: {==}

# promise: a lane whose request finished takes part in no further step.
law ln_finish_leaves_batch:
  for s: LaneState
  {ln_in_batch(ln_step(s, EvLnFinish{})) == False{} : Bool}

def ln_finish_leaves_batch(s):
  match s:
    case LnFree{}: {==}
    case LnPrefill{}: {==}
    case LnDecode{}: {==}
    case LnDrain{}: {==}

# promise: a lane whose request failed takes part in no further step.
law ln_fail_leaves_batch:
  for s: LaneState
  {ln_in_batch(ln_step(s, EvLnFail{})) == False{} : Bool}

def ln_fail_leaves_batch(s):
  match s:
    case LnFree{}: {==}
    case LnPrefill{}: {==}
    case LnDecode{}: {==}
    case LnDrain{}: {==}

# promise: a free lane never takes part in a step.
law ln_batch_lane_busy:
  for +s: LaneState
  {Bool.and(ln_in_batch(s), ln_is_free(s)) == False{} : Bool}

def ln_batch_lane_busy(s):
  match s:
    case LnFree{}: {==}
    case LnPrefill{}: {==}
    case LnDecode{}: {==}
    case LnDrain{}: {==}

# promise: a request is routed only to a lane that admission then makes busy (never to a busy lane).
law ln_route_only_admittable:
  for +s: LaneState
  {Bool.and(ln_route_eligible(s), ln_is_free(ln_step(s, EvLnAdmit{}))) == False{} : Bool}

def ln_route_only_admittable(s):
  match s:
    case LnFree{}: {==}
    case LnPrefill{}: {==}
    case LnDecode{}: {==}
    case LnDrain{}: {==}

# promise: a free lane that would take over more of a request's prompt is always preferred to one that would take over less.
law ln_route_more_reuse_wins:
  for +rb: Nat
  for +e: Nat
  for +sa: Nat
  for +sb: Nat
  {ln_route_better(1n+Nat.add(rb, e), sa, rb, sb) == True{} : Bool}

def ln_route_more_reuse_wins(rb, e, sa, sb):
  %ln_gt_succ_add(rb, e) : {Bool.or(_, Bool.and(Nat.is_eq(1n+Nat.add(rb, e), rb), Nat.is_lt(sa, sb))) == True{} : Bool}
  {==}

# promise: no lane is a better place than itself, so the lane routing picks is well defined.
law ln_route_irreflexive:
  for +r: Nat
  for +s: Nat
  {ln_route_better(r, s, r, s) == False{} : Bool}

def ln_route_irreflexive(r, s):
  %ln_gt_refl_false(r) : {Bool.or(_, Bool.and(Nat.is_eq(r, r), Nat.is_lt(s, s))) == False{} : Bool}
  %ln_eq_refl_true(r) : {Bool.or(False{}, Bool.and(_, Nat.is_lt(s, s))) == False{} : Bool}
  %ln_lt_refl_false(s) : {Bool.or(False{}, Bool.and(True{}, _)) == False{} : Bool}
  {==}

# promise: a lane plan is refused exactly when it lacks device memory, and then it reports a positive shortfall.
law ln_plan_fits_iff_no_shortfall:
  for +n: Nat
  for +lb: Nat
  for +sh: Nat
  for +u: Nat
  {ln_plan_fits(n, lb, sh, u) == Nat.is_eq(ln_plan_shortfall(n, lb, sh, u), 0n) : Bool}

def ln_plan_fits_iff_no_shortfall(n, lb, sh, u):
  ln_le_iff_sub_zero(Nat.add(sh, Nat.mul(n, lb)), u)

# promise: folding one more slot into a batched step never shrinks the chunks it covers.
law ln_batch_chunks_keeps_acc:
  for +acc: Nat
  for +c: Nat
  for +r: Nat
  for +ch: Nat
  {Nat.is_le(acc, ln_batch_chunks(acc, c, r, ch)) == True{} : Bool}

def ln_batch_chunks_keeps_acc(acc, c, r, ch):
  L.nat_max_ge_l(acc, ln_slot_chunks(c, r, ch))

# promise: a batched step covers at least the chunks of every slot folded into it.
law ln_batch_chunks_covers_slot:
  for +acc: Nat
  for +c: Nat
  for +r: Nat
  for +ch: Nat
  {Nat.is_le(ln_slot_chunks(c, r, ch), ln_batch_chunks(acc, c, r, ch)) == True{} : Bool}

def ln_batch_chunks_covers_slot(acc, c, r, ch):
  L.nat_max_ge_r(acc, ln_slot_chunks(c, r, ch))

# promise: a slot the batch check accepts never writes a row past the chunks its lane owns.
law ln_slot_fits_within_lane:
  for +c: Nat
  for +r: Nat
  for +n: Nat
  for +ch: Nat
  for +lc: Nat
  {Bool.and(ln_slot_fits(c, r, n, ch, lc), Nat.is_lt(Nat.mul(lc, ch), Nat.add(c, r))) == False{} : Bool}

def ln_slot_fits_within_lane(c, r, n, ch, lc):
  ln_fit_within(c, r, n, ch, lc, Nat.is_le(n, lc), {==}, Nat.is_le(Nat.add(c, r), Nat.mul(n, ch)), {==})

# promise: a slot the batch check accepts never writes a row past the chunks the step covers.
law ln_slot_fits_covered:
  for +c: Nat
  for +r: Nat
  for +n: Nat
  for +ch: Nat
  for +lc: Nat
  {Bool.and(ln_slot_fits(c, r, n, ch, lc), Nat.is_lt(Nat.mul(n, ch), Nat.add(c, r))) == False{} : Bool}

def ln_slot_fits_covered(c, r, n, ch, lc):
  ln_fit_covered(c, r, n, ch, Nat.is_le(n, lc), Nat.is_le(Nat.add(c, r), Nat.mul(n, ch)), {==})

# promise: a model that decodes without a drafter can always be served with several lanes.
law ln_plain_decode_lanes_ok:
  {ln_drafter_ok(LdNone{}) == True{} : Bool}

def ln_plain_decode_lanes_ok():
  {==}

# promise: a model is always served with at least one lane.
law ln_served_lanes_at_least_one:
  for +r: Nat
  for +ok: Bool
  {Nat.is_le(1n, ln_served_lanes(r, ok)) == True{} : Bool}

def ln_served_lanes_at_least_one(r, ok):
  match ok:
    case True{}:
      L.nat_max_ge_l(1n, r)
    case False{}:
      {==}

# promise: a model is never served with more lanes than were asked for.
law ln_served_lanes_within_request:
  for +r: Nat
  for +ok: Bool
  {Nat.is_le(ln_served_lanes(1n+r, ok), 1n+r) == True{} : Bool}

def ln_served_lanes_within_request(r, ok):
  match r:
    case 0n:
      match ok:
        case True{}: {==}
        case False{}: {==}
    case 1n+rp:
      match ok:
        case True{}:
          L.nat_le_refl(1n+1n+rp)
        case False{}: {==}
