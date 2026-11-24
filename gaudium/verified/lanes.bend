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
import tensor_bounds.bend as TB

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

def ln_ev_is_release(ev: LaneEvent) -> Bool:
  match ev:
    case EvLnRelease{}:
      True{}
    case _:
      False{}

# Does the lifecycle allow `ev` in state `s`? Every allowed event moves the lane; one it does not
# allow leaves it where it was (the shell refuses it).
def ln_allows(s: LaneState, ev: LaneEvent) -> Bool:
  match s ev:
    case LnFree{} EvLnAdmit{}:
      True{}
    case LnPrefill{} EvLnPrefilled{}:
      True{}
    case LnPrefill{} EvLnFinish{}:
      True{}
    case LnPrefill{} EvLnFail{}:
      True{}
    case LnDecode{} EvLnFinish{}:
      True{}
    case LnDecode{} EvLnFail{}:
      True{}
    case LnDrain{} EvLnRelease{}:
      True{}
    case _ _:
      False{}

# Has a lane's request ended? It stopped at an end-of-sequence token, was cancelled, or has no
# token left in its budget.
def ln_run_done(eos: Bool, cancelled: Bool, +remaining: Nat) -> Bool:
  Bool.or(eos, Bool.or(cancelled, Nat.is_eq(remaining, 0n)))

# ============================================================================
# 1b. The lane session of a model (the server's queue)
# ============================================================================
#
# Requests for a model served with lanes wait in that model's queue; one session at a time hands
# them to the engine's lanes. Both decisions below are made under the queue's lock.

# A request arrives while a session is `active` or not: does it start one?
def ln_session_starts(active: Bool) -> Bool:
  Bool.not(active)

# Byte offset of slot `j`'s cursor in the host staging buffer of a step of `b` slots: the
# step's `b` token ids come first (4 bytes each), then one 64-byte block per slot.
def ln_stage_cursor(b: Nat, j: Nat) -> Nat:
  Nat.add(Nat.mul(b, 64n), Nat.mul(j, 64n))

# Does the session end, with no lane busy (`idle`) and a request waiting or not?
def ln_session_ends(idle: Bool, waiting: Bool) -> Bool:
  Bool.and(idle, Bool.not(waiting))

# Is a waiter for the card's turn due, so that no request may pass it over any more? It has waited
# `waited` (ms) against at most `max_wait`, and been passed over `skips` times against at most
# `max_skips`.
def turn_due(waited: Nat, max_wait: Nat, skips: Nat, max_skips: Nat) -> Bool:
  Bool.or(Nat.is_ge(waited, max_wait), Nat.is_ge(skips, max_skips))

# How a waiter for the card ranks when the card's next turn is given: a waiter whose turn is
# `due` first, then a waiter of the model `loaded` now (no reload), then the others. Between
# equal ranks the waiter that arrived first takes the turn (`turn_better` is strict).
def turn_rank(due: Bool, loaded: Bool) -> Nat:
  match due:
    case True{}:
      2n
    case False{}:
      match loaded:
        case True{}:
          1n
        case False{}:
          0n

# Does a waiter of rank `a`, later in arrival order, take the turn from the best so far (`b`)?
def turn_better(a: Nat, b: Nat) -> Bool:
  Nat.is_gt(a, b)

# A waiter's skip count after the card's turn went to a request that `passed` it over or not.
def ln_skips_after(skips: Nat, passed: Bool) -> Nat:
  match passed:
    case True{}:
      Nat.add(1n, skips)
    case False{}:
      skips

# The active requests a running lane session counts with `busy` lanes busy: at least one while it
# holds the card (so the card is not released under it).
def ln_session_active(busy: Nat) -> Nat:
  Nat.max(1n, busy)

# Does a request that arrived at `request` pass over a waiter for the card that arrived at
# `waiter` (both on one clock)? Only a waiter that arrived first is passed over.
def ln_passes_over(waiter: Nat, request: Nat) -> Bool:
  Nat.is_lt(waiter, request)

# Would admitting a session's next request (arrived at `request`) pass over the oldest waiter for
# the card (arrived at `waiter`), whose turn is `due`?
def ln_admission_blocked(due: Bool, waiter: Nat, request: Nat) -> Bool:
  Bool.and(due, ln_passes_over(waiter, request))

# Does a running session admit its next request, the admission being `blocked` or not?
def ln_session_admits(blocked: Bool) -> Bool:
  Bool.not(blocked)

# Does a session hand the card's turn back now, with no lane busy (`idle`), its next admission
# being `blocked`? It waits for its next turn then, keeping its requests queued.
def ln_session_yields(idle: Bool, blocked: Bool) -> Bool:
  Bool.and(idle, blocked)

# A lane's next row after a step that wrote `rows` rows at `cursor` and emitted `emitted`
# tokens: it moves past the rows the step accepted, never past the rows it wrote.
def ln_lane_cursor_next(cursor: Nat, emitted: Nat, rows: Nat) -> Nat:
  Nat.add(cursor, Nat.min(emitted, rows))

# The recurrent-state buffer (0 or 1) a lane's next step reads, after a step that read `p`: the
# one that step wrote.
def ln_lane_parity_next(p: Nat) -> Nat:
  Nat.sub(1n, p)

# What a lane session's driver does once the engine stopped serving: wait for the card's next
# turn (the session handed it back), fail the requests still queued (the engine stopped before its
# feed ended the session), or stop (the feed ended the session: nothing is queued).
type SessionAfter is Data:
  SaAgain{}
  SaFail{}
  SaDone{}

def ln_session_after_open(ok: Bool, yielded: Bool) -> SessionAfter:
  match ok:
    case True{}:
      match yielded:
        case True{}:
          SaAgain{}
        case False{}:
          SaFail{}
    case False{}:
      SaFail{}

# `ok`: the engine served without an error; `yielded`: the feed handed the card back; `ended`:
# the feed ended the session (it cleared the session's flag under the queue's lock, so a new
# arrival may already have started the next one: this driver only stops).
def ln_session_after(ok: Bool, yielded: Bool, ended: Bool) -> SessionAfter:
  match ended:
    case True{}:
      SaDone{}
    case False{}:
      ln_session_after_open(ok, yielded)

def ln_sa_done(a: SessionAfter) -> Bool:
  match a:
    case SaAgain{}: False{}
    case SaFail{}: False{}
    case SaDone{}: True{}

# What a running lane session does next.
type SessionMove is Data:
  SmEnd{}
  SmYield{}
  SmWait{}
  SmAdmit{}

def ln_session_move(ends: Bool, yields: Bool, admits: Bool, waiting: Bool) -> SessionMove:
  match ends:
    case True{}:
      SmEnd{}
    case False{}:
      match yields:
        case True{}:
          SmYield{}
        case False{}:
          match admits:
            case True{}:
              match waiting:
                case True{}:
                  SmAdmit{}
                case False{}:
                  SmWait{}
            case False{}:
              SmWait{}

# What a lane session does next, with no lane busy (`idle`) or not, a request `waiting` or not,
# and its next admission `blocked` (it would pass over a due waiter for the card) or not: end,
# hand the card's turn back, wait for its lanes, or admit the next request.
def ln_session_next(+idle: Bool, +waiting: Bool, +blocked: Bool) -> SessionMove:
  ln_session_move(
    ln_session_ends(idle, waiting),
    ln_session_yields(idle, blocked),
    ln_session_admits(blocked),
    waiting
  )

def ln_sm_ends(m: SessionMove) -> Bool:
  match m:
    case SmEnd{}: True{}
    case SmYield{}: False{}
    case SmWait{}: False{}
    case SmAdmit{}: False{}

def ln_sm_yields(m: SessionMove) -> Bool:
  match m:
    case SmEnd{}: False{}
    case SmYield{}: True{}
    case SmWait{}: False{}
    case SmAdmit{}: False{}

def ln_sm_waits(m: SessionMove) -> Bool:
  match m:
    case SmEnd{}: False{}
    case SmYield{}: False{}
    case SmWait{}: True{}
    case SmAdmit{}: False{}

def ln_sm_admits(m: SessionMove) -> Bool:
  match m:
    case SmEnd{}: False{}
    case SmYield{}: False{}
    case SmWait{}: False{}
    case SmAdmit{}: True{}

# ============================================================================
# 2. Routing a request to a lane
# ============================================================================

# Is free lane `a` a better place for a request than free lane `b`? `ra`: the prompt rows lane
# `a` would take over (resume); `la`: the rows of its held prompt the request would destroy
# (held but not shared: the conversation it holds loses them); `ca`: the rows its held prompt
# shares with the request; `sa`: when a request was last admitted to it. The larger net of rows taken over and
# rows kept wins (`ra + lb` against `rb + la`); between equals, more shared rows (a conversation
# stays in its lane); then the lane used longest ago.
def ln_route_better(+ra: Nat, +la: Nat, +ca: Nat, +sa: Nat, +rb: Nat, +lb: Nat, +cb: Nat, +sb: Nat) -> Bool:
  Bool.or(
    Nat.is_gt(Nat.add(lb, ra), Nat.add(la, rb)),
    Bool.and(
      Nat.is_eq(Nat.add(lb, ra), Nat.add(la, rb)),
      Bool.or(Nat.is_gt(ca, cb), Bool.and(Nat.is_eq(ca, cb), Nat.is_lt(sa, sb)))
    )
  )

# The routing clock after an admission at clock `c`: the admitted lane is stamped with it (when a
# request was last admitted to it, which routing reads: the lane used longest ago goes first).
def ln_admit_clock(c: Nat) -> Nat:
  Nat.add(1n, c)

# Rows of a lane's held prompt (`held`) that a request sharing `shared` of them destroys.
def ln_route_loss(held: Nat, shared: Nat) -> Nat:
  Nat.sub(held, shared)

# ============================================================================
# 3. The device plan of N lanes
# ============================================================================

# Bytes of `n` lanes of `lane_bytes` each beside `shared` bytes every lane uses.
def ln_plan_bytes(n: Nat, lane_bytes: Nat, shared: Nat) -> Nat:
  Nat.add(shared, Nat.mul(n, lane_bytes))

def ln_plan_fits(n: Nat, lane_bytes: Nat, shared: Nat, usable: Nat) -> Bool:
  Nat.is_le(ln_plan_bytes(n, lane_bytes, shared), usable)

# Bytes of the lanes' own caches in the device plan: `n` lanes of `kv` bytes (K/V caches and
# recurrent state), and `spec` bytes of speculative step buffers for every lane but the first
# (the first lane's are the speculative runtime's own, planned with it).
def ln_lane_arena_bytes(kv: Nat, spec: Nat, +n: Nat) -> Nat:
  Nat.add(Nat.mul(n, kv), Nat.mul(Nat.sub(n, 1n), spec))

# Bytes of the lanes' kept prefix positions: `n` lanes of `per_lane` positions of `slot` bytes.
def ln_lane_pfx_bytes(per_lane: Nat, slot: Nat, n: Nat) -> Nat:
  Nat.mul(n, Nat.mul(per_lane, slot))

# Kept prefix positions a load asks the plan for: `wanted` in each of `n` lanes.
def ln_lane_pfx_ask(wanted: Nat, n: Nat) -> Nat:
  Nat.mul(n, wanted)

# Bytes the activation arena gives one buffer of the batched steps (`bytes`): at least 256,
# rounded up to 256 (the arena's allocation unit).
def ln_lane_buf_bytes(bytes: Nat) -> Nat:
  TB.align_up(Nat.max(bytes, 256n), 256n)

# The total of the batched steps' buffers so far (`acc`), with one more buffer of `bytes`.
def ln_lane_bufs_add(acc: Nat, bytes: Nat) -> Nat:
  Nat.add(acc, ln_lane_buf_bytes(bytes))

# Bytes of the batched steps' io scalars: `count` buffers of `each` bytes, each as the arena
# allocates it.
def ln_lane_io_bytes(count: Nat, each: Nat) -> Nat:
  Nat.mul(count, ln_lane_buf_bytes(each))

# The device arena of the lanes' caches and state (`arena`) and their kept positions (`pfx`).
def ln_kv_arena_total(arena: Nat, pfx: Nat) -> Nat:
  Nat.add(arena, pfx)

# May an engine with `lanes` lanes have the host-RAM tier of prefix reuse? Only one lane, unless
# the tier keeps one prefix cache per lane (`per_lane`).
def ln_host_tier_ok(lanes: Nat, per_lane: Bool) -> Bool:
  Bool.or(Nat.is_le(lanes, 1n), per_lane)

# The shared part of a device plan of `total` bytes holding `n` lanes of `lane_bytes`: what the
# lane plan check is given beside the lanes.
def ln_plan_shared(total: Nat, +n: Nat, +lane_bytes: Nat) -> Nat:
  Nat.sub(total, Nat.mul(n, lane_bytes))

# Lanes of `lane_bytes` that fit beside `shared` bytes in `usable` bytes (the refusal says it).
def ln_lanes_fit(lane_bytes: Nat, shared: Nat, usable: Nat) -> Nat:
  Nat.div(Nat.sub(usable, shared), lane_bytes)

# Kept prefix positions per lane when `granted` fit for all `n` lanes: the same count each.
def ln_prefix_per_lane(granted: Nat, n: Nat) -> Nat:
  Nat.div(granted, n)

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

# The smallest power of two at least `n`, searched from `p` by doubling at most `fuel` times
# (`done`: `p` is already at least `n`).
def ln_pow2_up(fuel: Nat, done: Bool, +p: Nat, +n: Nat) -> Nat:
  match fuel:
    case 0n:
      p
    case 1n+f:
      match done:
        case True{}:
          p
        case False{}:
          ln_pow2_up(f, Nat.is_ge(Nat.add(p, p), n), Nat.add(p, p), n)

# The chunks a batched step covers when its slots need `need`: rounded up to a power of two, so
# one recipe serves every count up to it (a slot's chunks past its cursor are fully masked), and
# never past the lane's own `lane_chunks`.
def ln_step_chunks(+need: Nat, +lane_chunks: Nat) -> Nat:
  Nat.min(lane_chunks, Nat.max(need, ln_pow2_up(need, Nat.is_ge(1n, need), 1n, need)))

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

# Which kinds of decode have batched steps for several lanes: plain decode, and DSpine.
def ln_drafter_ok(d: LaneDrafter) -> Bool:
  match d:
    case LdNone{}:
      True{}
    case LdDSpine{}:
      True{}
    case LdOther{}:
      False{}

# Does a speculative lane whose next row is `cursor` take the batched step of `rows` rows, in a
# context of `max_ctx` rows? Otherwise it takes single-row steps, which draft nothing, to the
# end of its request (as a single stream does near the end of its context).
def ln_spec_batched(cursor: Nat, rows: Nat, max_ctx: Nat) -> Bool:
  Nat.is_le(Nat.add(cursor, rows), max_ctx)

# Of two draft widths asked for, the one lanes step at (every lane verifies the same rows in a
# launch, and the state rings are sized for it); folded over every width asked for.
def ln_lane_width_pick(+a: Nat, +b: Nat) -> Nat:
  Nat.min(a, b)

# The kind of drafter the lane rule judges, for a load with a drafter (`has`) or not, a DSpine
# drafter (`dspine`) or another, whose verify rows are in a form the batched speculative steps are
# built for (`batched_form`: today the checkpointed DeltaNet verify rows) or not.
def ln_lane_drafter_of(has: Bool, dspine: Bool, batched_form: Bool) -> LaneDrafter:
  match has:
    case False{}:
      LdNone{}
    case True{}:
      match dspine:
        case True{}:
          match batched_form:
            case True{}:
              LdDSpine{}
            case False{}:
              LdOther{}
        case False{}:
          LdOther{}

# Does a load get lanes? Its engine has batched steps (`engine_ok`), its drafter has, and it is
# not a calibration load (which runs one request at a time).
def ln_lanes_ok(engine_ok: Bool, d: LaneDrafter, calibrating: Bool) -> Bool:
  Bool.and(engine_ok, Bool.and(ln_drafter_ok(d), Bool.not(calibrating)))

# Lanes a model is served with: the lanes asked for (at least one, at most `max`, the most the
# batched step graphs are built for) when its engine and its drafter have batched steps,
# otherwise one.
def ln_served_lanes(+requested: Nat, +max: Nat, ok: Bool) -> Nat:
  match ok:
    case True{}:
      Nat.min(Nat.max(1n, max), Nat.max(1n, requested))
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

def ln_min_ge(+a: Nat, +b: Nat, +c: Nat, ha: {Nat.is_le(c, a) == True{} : Bool}, hb: {Nat.is_le(c, b) == True{} : Bool}) -> {Nat.is_le(c, Nat.min(a, b)) == True{} : Bool}:
  match a b:
    case 0n _:
      ha
    case 1n+ap 0n:
      hb
    case 1n+ap 1n+bp:
      match c:
        case 0n:
          {==}
        case 1n+cp:
          ln_min_ge(ap, bp, cp, ha, hb)

def ln_cover_aux(+need: Nat, +lc: Nat, b: Bool, e: {Nat.is_lt(lc, need) == b : Bool}) -> {Bool.or(b, Nat.is_le(need, ln_step_chunks(need, lc))) == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      ln_min_ge(lc, Nat.max(need, ln_pow2_up(need, Nat.is_ge(1n, need), 1n, need)), need, L.nat_not_lt_imp_le(lc, need, e), L.nat_max_ge_l(need, ln_pow2_up(need, Nat.is_ge(1n, need), 1n, need)))

def ln_div_mul_le(+a: Nat, +bp: Nat) -> {Nat.is_le(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), a) == True{} : Bool}:
  L.nat_le_subst_r(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.add(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp)), a, L.nat_div_mod_r(a, bp), L.nat_add_le_r(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp)))

def ln_fit_count_aux(+bp: Nat, +sh: Nat, +u: Nat, b: Bool, e: {Nat.is_lt(u, sh) == b : Bool}) -> {Bool.or(b, ln_plan_fits(ln_lanes_fit(1n+bp, sh, u), 1n+bp, sh, u)) == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      L.nat_le_subst_r(Nat.add(sh, Nat.mul(Nat.div(Nat.sub(u, sh), 1n+bp), 1n+bp)), Nat.add(Nat.sub(u, sh), sh), u, L.nat_sub_add_cancel(u, sh, L.nat_not_lt_imp_le(u, sh, e)), L.nat_le_subst_r(Nat.add(sh, Nat.mul(Nat.div(Nat.sub(u, sh), 1n+bp), 1n+bp)), Nat.add(sh, Nat.sub(u, sh)), Nat.add(Nat.sub(u, sh), sh), L.nat_add_comm(sh, Nat.sub(u, sh)), L.nat_add_le_mono(sh, sh, Nat.mul(Nat.div(Nat.sub(u, sh), 1n+bp), 1n+bp), Nat.sub(u, sh), L.nat_le_refl(sh), ln_div_mul_le(Nat.sub(u, sh), bp))))

def ln_min_ge1(+a: Nat, +b: Nat) -> {Nat.is_le(1n, Nat.min(Nat.max(1n, a), Nat.max(1n, b))) == True{} : Bool}:
  ln_min_ge(Nat.max(1n, a), Nat.max(1n, b), 1n, L.nat_max_ge_l(1n, a), L.nat_max_ge_l(1n, b))

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

# promise: a request is routed only to a free lane, never to a lane that holds a request.
law ln_route_only_free:
  for +s: LaneState
  {Bool.and(ln_route_eligible(s), Bool.not(ln_is_free(s))) == False{} : Bool}

def ln_route_only_free(s):
  match s:
    case LnFree{}: {==}
    case LnPrefill{}: {==}
    case LnDecode{}: {==}
    case LnDrain{}: {==}

# promise: no lane is a better place than itself: the routing order is strict.
law ln_route_irreflexive:
  for +r: Nat
  for +l: Nat
  for +c: Nat
  for +s: Nat
  {ln_route_better(r, l, c, s, r, l, c, s) == False{} : Bool}

def ln_route_irreflexive(r, l, c, s):
  %ln_gt_refl_false(Nat.add(l, r)) : {Bool.or(_, Bool.and(Nat.is_eq(Nat.add(l, r), Nat.add(l, r)), Bool.or(Nat.is_gt(c, c), Bool.and(Nat.is_eq(c, c), Nat.is_lt(s, s))))) == False{} : Bool}
  %ln_eq_refl_true(Nat.add(l, r)) : {Bool.or(False{}, Bool.and(_, Bool.or(Nat.is_gt(c, c), Bool.and(Nat.is_eq(c, c), Nat.is_lt(s, s))))) == False{} : Bool}
  %ln_gt_refl_false(c) : {Bool.or(False{}, Bool.and(True{}, Bool.or(_, Bool.and(Nat.is_eq(c, c), Nat.is_lt(s, s))))) == False{} : Bool}
  %ln_eq_refl_true(c) : {Bool.or(False{}, Bool.and(True{}, Bool.or(False{}, Bool.and(_, Nat.is_lt(s, s))))) == False{} : Bool}
  %ln_lt_refl_false(s) : {Bool.or(False{}, Bool.and(True{}, Bool.or(False{}, Bool.and(True{}, _)))) == False{} : Bool}
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

# promise: a model whose engine or drafter has no batched steps is served with one lane.
law ln_served_lanes_one_without_batching:
  for +r: Nat
  for +m: Nat
  {Nat.is_le(ln_served_lanes(r, m, False{}), 1n) == True{} : Bool}

def ln_served_lanes_one_without_batching(r, m):
  {==}

# promise: a model is always served with at least one lane.
law ln_served_lanes_at_least_one:
  for +r: Nat
  for +m: Nat
  for +ok: Bool
  {Nat.is_le(1n, ln_served_lanes(r, m, ok)) == True{} : Bool}

def ln_served_lanes_at_least_one(r, m, ok):
  match ok:
    case True{}:
      ln_min_ge1(m, r)
    case False{}:
      {==}

# promise: a model is never served with more lanes than were asked for.
law ln_served_lanes_within_request:
  for +r: Nat
  for +m: Nat
  for +ok: Bool
  {Nat.is_le(ln_served_lanes(1n+r, m, ok), 1n+r) == True{} : Bool}

def ln_served_lanes_within_request(r, m, ok):
  match r:
    case 0n:
      match ok:
        case True{}:
          L.nat_min_le_r(Nat.max(1n, m), Nat.max(1n, 1n))
        case False{}: {==}
    case 1n+rp:
      match ok:
        case True{}:
          L.nat_min_le_r(Nat.max(1n, m), Nat.max(1n, 1n+1n+rp))
        case False{}: {==}

# promise: a model is never served with more lanes than the batched step graphs are built for.
law ln_served_lanes_within_max:
  for +r: Nat
  for +m: Nat
  for +ok: Bool
  {Nat.is_le(ln_served_lanes(r, 1n+m, ok), 1n+m) == True{} : Bool}

def ln_served_lanes_within_max(r, m, ok):
  match m:
    case 0n:
      match ok:
        case True{}:
          L.nat_min_le_l(Nat.max(1n, 1n), Nat.max(1n, r))
        case False{}: {==}
    case 1n+mp:
      match ok:
        case True{}:
          L.nat_min_le_l(Nat.max(1n, 1n+1n+mp), Nat.max(1n, r))
        case False{}: {==}

# promise: a batched step never covers more chunks than a lane owns.
law ln_step_chunks_within_lane:
  for +need: Nat
  for +lc: Nat
  {Nat.is_le(ln_step_chunks(need, lc), lc) == True{} : Bool}

def ln_step_chunks_within_lane(need, lc):
  L.nat_min_le_l(lc, Nat.max(need, ln_pow2_up(need, Nat.is_ge(1n, need), 1n, need)))

# promise: a batched step covers every chunk its slots need whenever a lane can hold them.
law ln_step_chunks_cover_need:
  for +need: Nat
  for +lc: Nat
  {Bool.or(Nat.is_lt(lc, need), Nat.is_le(need, ln_step_chunks(need, lc))) == True{} : Bool}

def ln_step_chunks_cover_need(need, lc):
  ln_cover_aux(need, lc, Nat.is_lt(lc, need), {==})

# promise: a lane whose request ended leaves that state only when its report is delivered and it is released.
law ln_drain_left_only_by_release:
  for +ev: LaneEvent
  {Bool.or(ln_ev_is_release(ev), ln_is_drain(ln_step(LnDrain{}, ev))) == True{} : Bool}

def ln_drain_left_only_by_release(ev):
  match ev:
    case EvLnAdmit{}: {==}
    case EvLnPrefilled{}: {==}
    case EvLnFinish{}: {==}
    case EvLnFail{}: {==}
    case EvLnRelease{}: {==}

# promise: an event the lane lifecycle allows always moves the lane to another state, and one it refuses leaves the lane as it was.
law ln_allowed_moves:
  for +s: LaneState
  for +ev: LaneEvent
  {ln_allows(s, ev) == Bool.not(ln_eq(ln_step(s, ev), s)) : Bool}

def ln_allowed_moves(s, ev):
  match s ev:
    case LnFree{} EvLnAdmit{}: {==}
    case LnFree{} EvLnPrefilled{}: {==}
    case LnFree{} EvLnFinish{}: {==}
    case LnFree{} EvLnFail{}: {==}
    case LnFree{} EvLnRelease{}: {==}
    case LnPrefill{} EvLnAdmit{}: {==}
    case LnPrefill{} EvLnPrefilled{}: {==}
    case LnPrefill{} EvLnFinish{}: {==}
    case LnPrefill{} EvLnFail{}: {==}
    case LnPrefill{} EvLnRelease{}: {==}
    case LnDecode{} EvLnAdmit{}: {==}
    case LnDecode{} EvLnPrefilled{}: {==}
    case LnDecode{} EvLnFinish{}: {==}
    case LnDecode{} EvLnFail{}: {==}
    case LnDecode{} EvLnRelease{}: {==}
    case LnDrain{} EvLnAdmit{}: {==}
    case LnDrain{} EvLnPrefilled{}: {==}
    case LnDrain{} EvLnFinish{}: {==}
    case LnDrain{} EvLnFail{}: {==}
    case LnDrain{} EvLnRelease{}: {==}

# promise: a request with no token left in its budget has ended.
law ln_run_done_at_budget_end:
  for +e: Bool
  for +c: Bool
  {ln_run_done(e, c, 0n) == True{} : Bool}

def ln_run_done_at_budget_end(e, c):
  match e c:
    case True{} _: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a request that arrives while no session runs for its model starts one, so it is never left without one.
law ln_session_never_neither:
  for +a: Bool
  {Bool.or(a, ln_session_starts(a)) == True{} : Bool}

def ln_session_never_neither(a):
  match a:
    case True{}: {==}
    case False{}: {==}

# promise: a lane session never ends while a request waits.
law ln_session_waits_for_queue:
  for +i: Bool
  {ln_session_ends(i, True{}) == False{} : Bool}

def ln_session_waits_for_queue(i):
  match i:
    case True{}: {==}
    case False{}: {==}

# promise: a lane session never ends while a lane is busy.
law ln_session_waits_for_lanes:
  for +w: Bool
  {ln_session_ends(False{}, w) == False{} : Bool}

def ln_session_waits_for_lanes(w):
  {==}

# promise: the lanes of a plan are always counted in full in its bytes.
law ln_plan_counts_every_lane:
  for +n: Nat
  for +lb: Nat
  for +sh: Nat
  {Nat.is_le(Nat.mul(n, lb), ln_plan_bytes(n, lb, sh)) == True{} : Bool}

def ln_plan_counts_every_lane(n, lb, sh):
  L.nat_add_le_l(sh, Nat.mul(n, lb))

# promise: the shared part of a plan is always counted in full in its bytes.
law ln_plan_counts_shared:
  for +n: Nat
  for +lb: Nat
  for +sh: Nat
  {Nat.is_le(sh, ln_plan_bytes(n, lb, sh)) == True{} : Bool}

def ln_plan_counts_shared(n, lb, sh):
  L.nat_add_le_r(sh, Nat.mul(n, lb))

# promise: the lane count a refused plan reports as fitting does fit, whenever the shared part does.
law ln_lanes_fit_fits:
  for +bp: Nat
  for +sh: Nat
  for +u: Nat
  {Bool.or(Nat.is_lt(u, sh), ln_plan_fits(ln_lanes_fit(1n+bp, sh, u), 1n+bp, sh, u)) == True{} : Bool}

def ln_lanes_fit_fits(bp, sh, u):
  ln_fit_count_aux(bp, sh, u, Nat.is_lt(u, sh), {==})

# promise: the lanes together never get more kept prefix positions than were granted for them.
law ln_prefix_per_lane_within_grant:
  for +g: Nat
  for +np: Nat
  {Nat.is_le(Nat.mul(ln_prefix_per_lane(g, 1n+np), 1n+np), g) == True{} : Bool}

def ln_prefix_per_lane_within_grant(g, np):
  ln_div_mul_le(g, np)

# promise: a cancelled request has ended.
law ln_run_done_when_cancelled:
  for +e: Bool
  for +r: Nat
  {ln_run_done(e, True{}, r) == True{} : Bool}

def ln_run_done_when_cancelled(e, r):
  match e:
    case True{}: {==}
    case False{}: {==}

# promise: a request that produced an end-of-sequence token has ended.
law ln_run_done_at_eos:
  for +c: Bool
  for +r: Nat
  {ln_run_done(True{}, c, r) == True{} : Bool}

def ln_run_done_at_eos(c, r):
  {==}

# promise: a plan's bytes hold its shared part and all its lanes together.
law ln_plan_counts_all:
  for +n: Nat
  for +lb: Nat
  for +sh: Nat
  {Nat.is_le(Nat.add(sh, Nat.mul(n, lb)), ln_plan_bytes(n, lb, sh)) == True{} : Bool}

def ln_plan_counts_all(n, lb, sh):
  L.nat_le_refl(Nat.add(sh, Nat.mul(n, lb)))

# promise: every free lane is one routing may send a request to.
law ln_free_lane_routable:
  for +s: LaneState
  {Bool.or(Bool.not(ln_is_free(s)), ln_route_eligible(s)) == True{} : Bool}

def ln_free_lane_routable(s):
  match s:
    case LnFree{}: {==}
    case LnPrefill{}: {==}
    case LnDecode{}: {==}
    case LnDrain{}: {==}

# promise: a request that arrives while its model's session runs never starts a second one.
law ln_session_once:
  {ln_session_starts(True{}) == False{} : Bool}

def ln_session_once():
  {==}

# promise: a calibration load is never given lanes.
law ln_lanes_never_calibrating:
  for +e: Bool
  for +d: LaneDrafter
  {ln_lanes_ok(e, d, True{}) == False{} : Bool}

def ln_lanes_never_calibrating(e, d):
  match e d:
    case True{} LdNone{}: {==}
    case True{} LdDSpine{}: {==}
    case True{} LdOther{}: {==}
    case False{} _: {==}

# promise: an engine without batched steps is never given lanes, whatever its drafter.
law ln_lanes_need_engine:
  for +d: LaneDrafter
  for +c: Bool
  {ln_lanes_ok(False{}, d, c) == False{} : Bool}

def ln_lanes_need_engine(d, c):
  {==}

# promise: a load whose drafter has no batched steps is never given lanes.
law ln_lanes_need_drafter:
  for +e: Bool
  for +d: LaneDrafter
  for +c: Bool
  {Bool.and(ln_lanes_ok(e, d, c), Bool.not(ln_drafter_ok(d))) == False{} : Bool}

def ln_lanes_need_drafter(e, d, c):
  match e d c:
    case True{} LdNone{} True{}: {==}
    case True{} LdNone{} False{}: {==}
    case True{} LdDSpine{} True{}: {==}
    case True{} LdDSpine{} False{}: {==}
    case True{} LdOther{} True{}: {==}
    case True{} LdOther{} False{}: {==}
    case False{} LdNone{} _: {==}
    case False{} LdDSpine{} _: {==}
    case False{} LdOther{} _: {==}

# promise: a slot's cursor in the host staging buffer never lies inside the step's token ids.
law ln_stage_cursor_after_ids:
  for +b: Nat
  for +j: Nat
  {Nat.is_le(Nat.mul(b, 4n), ln_stage_cursor(b, j)) == True{} : Bool}

def ln_stage_cursor_after_ids(b, j):
  L.nat_le_trans(Nat.mul(b, 4n), Nat.mul(b, 64n), Nat.add(Nat.mul(b, 64n), Nat.mul(j, 64n)), L.nat_mul_le_mono_l(b, 4n, 64n, {==}), L.nat_add_le_r(Nat.mul(b, 64n), Nat.mul(j, 64n)))

# promise: two slots' cursors in the host staging buffer never overlap.
law ln_stage_cursors_apart:
  for +b: Nat
  for +j: Nat
  {Nat.is_le(Nat.add(ln_stage_cursor(b, j), 4n), ln_stage_cursor(b, 1n+j)) == True{} : Bool}

def ln_stage_cursors_apart(b, j):
  L.nat_le_trans(
    Nat.add(Nat.add(Nat.mul(b, 64n), Nat.mul(j, 64n)), 4n),
    Nat.add(Nat.mul(b, 64n), Nat.add(Nat.mul(j, 64n), 4n)),
    Nat.add(Nat.mul(b, 64n), Nat.add(64n, Nat.mul(j, 64n))),
    L.nat_le_subst_r(Nat.add(Nat.add(Nat.mul(b, 64n), Nat.mul(j, 64n)), 4n), Nat.add(Nat.add(Nat.mul(b, 64n), Nat.mul(j, 64n)), 4n), Nat.add(Nat.mul(b, 64n), Nat.add(Nat.mul(j, 64n), 4n)), L.nat_add_assoc(Nat.mul(b, 64n), Nat.mul(j, 64n), 4n), L.nat_le_refl(Nat.add(Nat.add(Nat.mul(b, 64n), Nat.mul(j, 64n)), 4n))),
    L.nat_add_le_mono(Nat.mul(b, 64n), Nat.mul(b, 64n), Nat.add(Nat.mul(j, 64n), 4n), Nat.add(64n, Nat.mul(j, 64n)), L.nat_le_refl(Nat.mul(b, 64n)), L.nat_le_trans(Nat.add(Nat.mul(j, 64n), 4n), Nat.add(4n, Nat.mul(j, 64n)), Nat.add(64n, Nat.mul(j, 64n)), L.nat_le_subst_r(Nat.add(Nat.mul(j, 64n), 4n), Nat.add(Nat.mul(j, 64n), 4n), Nat.add(4n, Nat.mul(j, 64n)), L.nat_add_comm(Nat.mul(j, 64n), 4n), L.nat_le_refl(Nat.add(Nat.mul(j, 64n), 4n))), L.nat_add_le_mono(4n, 64n, Nat.mul(j, 64n), Nat.mul(j, 64n), {==}, L.nat_le_refl(Nat.mul(j, 64n)))))
  )

def ln_ge_add_true(+w: Nat, +e: Nat) -> {True{} == Nat.is_ge(Nat.add(w, e), w) : Bool}:
  match w:
    case 0n:
      match e:
        case 0n:
          {==}
        case 1n+p:
          {==}
    case 1n+wp:
      ln_ge_add_true(wp, e)

def ln_or_true_r(b: Bool) -> {Bool.or(b, True{}) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def ln_or_not_self(b: Bool) -> {Bool.or(Bool.not(b), b) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# promise: a waiter that has waited its maximum wait is due: no request passes it over any more.
law turn_due_after_max_wait:
  for +w: Nat
  for +e: Nat
  for +s: Nat
  for +ms: Nat
  {turn_due(Nat.add(w, e), w, s, ms) == True{} : Bool}

def turn_due_after_max_wait(w, e, s, ms):
  %ln_ge_add_true(w, e) : {Bool.or(_, Nat.is_ge(s, ms)) == True{} : Bool}
  {==}

# promise: a waiter passed over its maximum number of times is due: no request passes it over any more.
law turn_due_after_max_skips:
  for +w: Nat
  for +mw: Nat
  for +ms: Nat
  for +e: Nat
  {turn_due(w, mw, Nat.add(ms, e), ms) == True{} : Bool}

def turn_due_after_max_skips(w, mw, ms, e):
  %ln_ge_add_true(ms, e) : {Bool.or(Nat.is_ge(w, mw), _) == True{} : Bool}
  ln_or_true_r(Nat.is_ge(w, mw))

# promise: a lane session never hands the card's turn back while one of its lanes is busy.
law ln_session_keeps_busy_lanes:
  for +d: Bool
  {ln_session_yields(False{}, d) == False{} : Bool}

def ln_session_keeps_busy_lanes(d):
  {==}

def ln_lt_add_false(+r: Nat, +e: Nat) -> {False{} == Nat.is_lt(Nat.add(r, e), r) : Bool}:
  match r:
    case 0n:
      match e:
        case 0n:
          {==}
        case 1n+p:
          {==}
    case 1n+rp:
      ln_lt_add_false(rp, e)

def ln_lt_succ_add(+w: Nat, +e: Nat) -> {True{} == Nat.is_lt(w, 1n+Nat.add(w, e)) : Bool}:
  match w:
    case 0n:
      {==}
    case 1n+wp:
      ln_lt_succ_add(wp, e)

def ln_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# promise: a request never counts as passing over a waiter for the card that arrived after it.
law ln_never_passes_younger:
  for +r: Nat
  for +e: Nat
  {ln_passes_over(Nat.add(r, e), r) == False{} : Bool}

def ln_never_passes_younger(r, e):
  %ln_lt_add_false(r, e) : {_ == False{} : Bool}
  {==}

# promise: a waiter whose turn is due and who arrived before a session's next request blocks its admission.
law ln_due_older_waiter_blocks:
  for +w: Nat
  for +e: Nat
  {ln_admission_blocked(True{}, w, Nat.add(1n+w, e)) == True{} : Bool}

def ln_due_older_waiter_blocks(w, e):
  %ln_lt_succ_add(w, e) : {Bool.and(True{}, _) == True{} : Bool}
  {==}

# promise: a waiter that arrived after a session's next request never blocks its admission.
law ln_younger_waiter_never_blocks:
  for +d: Bool
  for +r: Nat
  for +e: Nat
  {ln_admission_blocked(d, Nat.add(r, e), r) == False{} : Bool}

def ln_younger_waiter_never_blocks(d, r, e):
  %ln_lt_add_false(r, e) : {Bool.and(d, _) == False{} : Bool}
  ln_and_false_r(d)

# promise: a lane session never admits a request that would pass over a waiter whose turn on the card is due.
law ln_next_never_admits_blocked:
  for +i: Bool
  for +w: Bool
  {ln_sm_admits(ln_session_next(i, w, True{})) == False{} : Bool}

def ln_next_never_admits_blocked(i, w):
  match i w:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a lane session never hands the card's turn back while one of its lanes is busy.
law ln_next_never_yields_busy:
  for +w: Bool
  for +b: Bool
  {ln_sm_yields(ln_session_next(False{}, w, b)) == False{} : Bool}

def ln_next_never_yields_busy(w, b):
  match w b:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a lane session hands the card's turn back only when its next admission would pass over a due waiter.
law ln_next_never_yields_unblocked:
  for +i: Bool
  for +w: Bool
  {ln_sm_yields(ln_session_next(i, w, False{})) == False{} : Bool}

def ln_next_never_yields_unblocked(i, w):
  match i w:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a lane session never ends while a request of its model waits.
law ln_next_never_ends_waiting:
  for +i: Bool
  for +b: Bool
  {ln_sm_ends(ln_session_next(i, True{}, b)) == False{} : Bool}

def ln_next_never_ends_waiting(i, b):
  match i b:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a lane session with no lane busy and a request waiting admits it or hands the card's turn back; it never sits idle.
law ln_next_idle_moves:
  for +b: Bool
  {ln_sm_waits(ln_session_next(True{}, True{}, b)) == False{} : Bool}

def ln_next_idle_moves(b):
  match b:
    case True{}: {==}
    case False{}: {==}

# promise: a lane session admits a request only when one waits.
law ln_next_admits_only_waiting:
  for +i: Bool
  for +b: Bool
  {ln_sm_admits(ln_session_next(i, False{}, b)) == False{} : Bool}

def ln_next_admits_only_waiting(i, b):
  match i b:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a step moves a lane's cursor only over rows the step wrote.
law ln_cursor_never_past_rows:
  for +c: Nat
  for +e: Nat
  for +r: Nat
  {Nat.is_le(ln_lane_cursor_next(c, e, r), Nat.add(c, r)) == True{} : Bool}

def ln_cursor_never_past_rows(c, e, r):
  L.nat_add_le_mono(c, c, Nat.min(e, r), r, L.nat_le_refl(c), L.nat_min_le_r(e, r))

# promise: a step never moves a lane's cursor back.
law ln_cursor_never_back:
  for +c: Nat
  for +e: Nat
  for +r: Nat
  {Nat.is_le(c, ln_lane_cursor_next(c, e, r)) == True{} : Bool}

def ln_cursor_never_back(c, e, r):
  L.nat_add_le_r(c, Nat.min(e, r))

# promise: a lane's next step reads one of its two recurrent-state buffers.
law ln_parity_in_range:
  for +p: Nat
  {Nat.is_le(ln_lane_parity_next(p), 1n) == True{} : Bool}

def ln_parity_in_range(p):
  match p:
    case 0n:
      {==}
    case 1n+q:
      match q:
        case 0n:
          {==}
        case 1n+u:
          {==}

# promise: a lane's next step reads the recurrent-state buffer the step before it wrote, never the one it read.
law ln_parity_alternates:
  for +p: Nat
  {Bool.or(Nat.is_gt(p, 1n), Nat.is_ne(ln_lane_parity_next(p), p)) == True{} : Bool}

def ln_parity_alternates(p):
  match p:
    case 0n:
      {==}
    case 1n+q:
      match q:
        case 0n:
          {==}
        case 1n+u:
          {==}

def ln_lt_add_succ(+c: Nat, +m: Nat) -> {True{} == Nat.is_lt(c, Nat.add(c, 1n+m)) : Bool}:
  match c:
    case 0n:
      {==}
    case 1n+cp:
      ln_lt_add_succ(cp, m)

# promise: a waiter never takes the card's turn from an equally ranked waiter that arrived before it.
law turn_better_strict:
  for +r: Nat
  {turn_better(r, r) == False{} : Bool}

def turn_better_strict(r):
  %ln_gt_refl_false(r) : {_ == False{} : Bool}
  {==}

# promise: a lane session never ends while one of its lanes is busy.
law ln_next_never_ends_busy:
  for +w: Bool
  for +b: Bool
  {ln_sm_ends(ln_session_next(False{}, w, b)) == False{} : Bool}

def ln_next_never_ends_busy(w, b):
  match w b:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a lane session with no lane busy never sits holding the card: it ends, admits a request or hands the turn back.
law ln_next_idle_never_waits:
  for +w: Bool
  for +b: Bool
  {ln_sm_waits(ln_session_next(True{}, w, b)) == False{} : Bool}

def ln_next_idle_never_waits(w, b):
  match w b:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a step moves a lane's cursor at most over the rows it accepted.
law ln_cursor_never_past_emitted:
  for +c: Nat
  for +e: Nat
  for +r: Nat
  {Nat.is_le(ln_lane_cursor_next(c, e, r), Nat.add(c, e)) == True{} : Bool}

def ln_cursor_never_past_emitted(c, e, r):
  L.nat_add_le_mono(c, c, Nat.min(e, r), e, L.nat_le_refl(c), L.nat_min_le_l(e, r))

# promise: a step that wrote and accepted a row moves the lane's cursor forward.
law ln_cursor_moves_forward:
  for +c: Nat
  for +e: Nat
  for +r: Nat
  {Nat.is_lt(c, ln_lane_cursor_next(c, 1n+e, 1n+r)) == True{} : Bool}

def ln_cursor_moves_forward(c, e, r):
  %ln_lt_add_succ(c, Nat.min(e, r)) : {_ == True{} : Bool}
  {==}

def ln_gt_succ_or(+s: Nat, +c: Nat) -> {Bool.or(Nat.is_gt(s, c), Nat.is_gt(1n+c, s)) == True{} : Bool}:
  match s c:
    case 0n 0n:
      {==}
    case 0n 1n+cp:
      {==}
    case 1n+sp 0n:
      {==}
    case 1n+sp 1n+cp:
      ln_gt_succ_or(sp, cp)

# promise: a lane admitted now is stamped later than every lane stamped before, so routing reads it as the most recently used.
law ln_admit_stamp_latest:
  for +c: Nat
  for +s: Nat
  {Bool.or(Nat.is_gt(s, c), Nat.is_gt(ln_admit_clock(c), s)) == True{} : Bool}

def ln_admit_stamp_latest(c, s):
  ln_gt_succ_or(s, c)

# promise: a waiter whose turn on the card is due takes the turn from every earlier waiter whose turn is not due.
law turn_due_takes_turn:
  for +l1: Bool
  for +l2: Bool
  {turn_better(turn_rank(True{}, l1), turn_rank(False{}, l2)) == True{} : Bool}

def turn_due_takes_turn(l1, l2):
  match l1 l2:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: no later waiter takes the card's turn from a waiter whose turn is due.
law turn_due_keeps_turn:
  for +d: Bool
  for +l1: Bool
  for +l2: Bool
  {turn_better(turn_rank(d, l1), turn_rank(True{}, l2)) == False{} : Bool}

def turn_due_keeps_turn(d, l1, l2):
  match d l1 l2:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

def ln_gt_succ_self(+s: Nat) -> {True{} == Nat.is_gt(1n+s, s) : Bool}:
  match s:
    case 0n:
      {==}
    case 1n+p:
      ln_gt_succ_self(p)

# promise: every time a waiter for the card is passed over, it counts toward its limit.
law ln_skips_count_pass:
  for +s: Nat
  {Nat.is_gt(ln_skips_after(s, True{}), s) == True{} : Bool}

def ln_skips_count_pass(s):
  %ln_gt_succ_self(s) : {_ == True{} : Bool}
  {==}

# promise: a waiter's count of times passed over never goes down while it waits.
law ln_skips_never_drop:
  for +s: Nat
  for +p: Bool
  {Nat.is_le(s, ln_skips_after(s, p)) == True{} : Bool}

def ln_skips_never_drop(s, p):
  match p:
    case True{}:
      L.nat_add_le_l(1n, s)
    case False{}:
      L.nat_le_refl(s)

# promise: a waiter that was not passed over is not charged a skip.
law ln_skips_unpassed_kept:
  for +s: Nat
  {Nat.is_le(ln_skips_after(s, False{}), s) == True{} : Bool}

def ln_skips_unpassed_kept(s):
  L.nat_le_refl(s)

# promise: a running lane session counts as at least one active request, so the card is never released under it.
law ln_session_active_counts:
  for +b: Nat
  {Nat.is_le(1n, ln_session_active(b)) == True{} : Bool}

def ln_session_active_counts(b):
  L.nat_max_ge_l(1n, b)

# promise: a running lane session counts at least as many active requests as it has busy lanes.
law ln_session_active_covers_lanes:
  for +b: Nat
  {Nat.is_le(b, ln_session_active(b)) == True{} : Bool}

def ln_session_active_covers_lanes(b):
  L.nat_max_ge_r(1n, b)

def ln_sub_add_ge(+t: Nat, +m: Nat) -> {Nat.is_le(t, Nat.add(Nat.sub(t, m), m)) == True{} : Bool}:
  match t:
    case 0n:
      L.nat_le_zero(Nat.add(Nat.sub(0n, m), m))
    case 1n+tp:
      match m:
        case 0n:
          L.nat_add_le_r(1n+tp, 0n)
        case 1n+mp:
          %Equal.sym(Nat, Nat.add(Nat.sub(tp, mp), 1n+mp), 1n+Nat.add(Nat.sub(tp, mp), mp), L.nat_add_succ_r(Nat.sub(tp, mp), mp)) : {Nat.is_le(1n+tp, _) == True{} : Bool}
          ln_sub_add_ge(tp, mp)

# promise: the device plan counts the caches and recurrent state of every lane.
law ln_arena_counts_lane_caches:
  for +kv: Nat
  for +sp: Nat
  for +n: Nat
  {Nat.is_le(Nat.mul(n, kv), ln_lane_arena_bytes(kv, sp, n)) == True{} : Bool}

def ln_arena_counts_lane_caches(kv, sp, n):
  L.nat_add_le_r(Nat.mul(n, kv), Nat.mul(Nat.sub(n, 1n), sp))

# promise: the device plan counts the speculative step buffers of every lane but the first.
law ln_arena_counts_other_spec:
  for +kv: Nat
  for +sp: Nat
  for +n: Nat
  {Nat.is_le(Nat.mul(Nat.sub(n, 1n), sp), ln_lane_arena_bytes(kv, sp, n)) == True{} : Bool}

def ln_arena_counts_other_spec(kv, sp, n):
  L.nat_add_le_l(Nat.mul(n, kv), Nat.mul(Nat.sub(n, 1n), sp))

# promise: the lane plan check is never given less than the whole device plan.
law ln_plan_check_sees_whole_plan:
  for +t: Nat
  for +n: Nat
  for +lb: Nat
  {Nat.is_le(t, ln_plan_bytes(n, lb, ln_plan_shared(t, n, lb))) == True{} : Bool}

def ln_plan_check_sees_whole_plan(t, n, lb):
  ln_sub_add_ge(t, Nat.mul(n, lb))

# promise: the device plan counts every lane's caches and state and the speculative buffers of every lane but the first, together.
law ln_arena_counts_all:
  for +kv: Nat
  for +sp: Nat
  for +n: Nat
  {Nat.is_le(Nat.add(Nat.mul(n, kv), Nat.mul(Nat.sub(n, 1n), sp)), ln_lane_arena_bytes(kv, sp, n)) == True{} : Bool}

def ln_arena_counts_all(kv, sp, n):
  L.nat_le_refl(Nat.add(Nat.mul(n, kv), Nat.mul(Nat.sub(n, 1n), sp)))

# promise: the device plan counts the kept prefix positions of every lane.
law ln_pfx_counts_every_lane:
  for +p: Nat
  for +s: Nat
  for +n: Nat
  {Nat.is_le(Nat.mul(n, Nat.mul(p, s)), ln_lane_pfx_bytes(p, s, n)) == True{} : Bool}

def ln_pfx_counts_every_lane(p, s, n):
  L.nat_le_refl(Nat.mul(n, Nat.mul(p, s)))

# promise: a load asks the plan for the kept positions of every lane.
law ln_pfx_ask_counts_every_lane:
  for +w: Nat
  for +n: Nat
  {Nat.is_le(Nat.mul(n, w), ln_lane_pfx_ask(w, n)) == True{} : Bool}

def ln_pfx_ask_counts_every_lane(w, n):
  L.nat_le_refl(Nat.mul(n, w))

# promise: the device arena holds both the lanes' caches and state and their kept positions.
law ln_kv_arena_counts_both:
  for +a: Nat
  for +p: Nat
  {Nat.is_le(Nat.add(a, p), ln_kv_arena_total(a, p)) == True{} : Bool}

def ln_kv_arena_counts_both(a, p):
  L.nat_le_refl(Nat.add(a, p))

# promise: an engine with more than one lane never gets a host-RAM tier that is not built for several lanes.
law ln_host_tier_never_with_lanes:
  for +k: Nat
  {ln_host_tier_ok(Nat.add(2n, k), False{}) == False{} : Bool}

def ln_host_tier_never_with_lanes(k):
  {==}

# promise: a lane session's driver never stops while its feed has not ended the session, so no request is left queued with no session.
law ln_after_never_strands:
  for +o: Bool
  for +y: Bool
  {ln_sa_done(ln_session_after(o, y, False{})) == False{} : Bool}

def ln_after_never_strands(o, y):
  match o y:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a lane session that handed the card back waits for its next turn.
law ln_after_yield_returns:
  {ln_session_after(True{}, True{}, False{}) == SaAgain{} : SessionAfter}

def ln_after_yield_returns():
  {==}

# promise: a lane session whose feed ended it only stops: it never runs again beside the next session or fails that session's requests.
law ln_after_ended_done:
  for +o: Bool
  for +y: Bool
  {ln_sa_done(ln_session_after(o, y, True{})) == True{} : Bool}

def ln_after_ended_done(o, y):
  {==}

# promise: the activation arena gives every buffer of the batched steps at least the bytes it needs.
law ln_lane_buf_holds_it:
  for +b: Nat
  {Nat.is_le(b, ln_lane_buf_bytes(b)) == True{} : Bool}

def ln_lane_buf_holds_it(b):
  L.nat_le_trans(b, Nat.max(b, 256n), TB.align_up(Nat.max(b, 256n), 256n), L.nat_max_ge_l(b, 256n), TB.align_up_never_shrinks(Nat.max(b, 256n), 256n))


# promise: the plan counts every io scalar of the batched steps.
law ln_lane_io_counts_each:
  for +c: Nat
  for +e: Nat
  {Nat.is_le(Nat.mul(c, e), ln_lane_io_bytes(c, e)) == True{} : Bool}

def ln_lane_io_counts_each(c, e):
  L.nat_mul_le_mono_l(c, e, ln_lane_buf_bytes(e), ln_lane_buf_holds_it(e))

# promise: the batched steps' buffer total counts everything so far and the next buffer as the arena allocates it.
law ln_lane_bufs_add_rounded:
  for +a: Nat
  for +b: Nat
  {Nat.is_le(Nat.add(a, ln_lane_buf_bytes(b)), ln_lane_bufs_add(a, b)) == True{} : Bool}

def ln_lane_bufs_add_rounded(a, b):
  L.nat_le_refl(Nat.add(a, ln_lane_buf_bytes(b)))

# promise: the plan counts every io buffer of the batched steps as the arena allocates it.
law ln_lane_io_counts_rounded:
  for +c: Nat
  for +e: Nat
  {Nat.is_le(Nat.mul(c, ln_lane_buf_bytes(e)), ln_lane_io_bytes(c, e)) == True{} : Bool}

def ln_lane_io_counts_rounded(c, e):
  L.nat_le_refl(Nat.mul(c, ln_lane_buf_bytes(e)))

# promise: the stop outcome of a lane session's driver counts as stopping.
law ln_sa_done_of_done:
  {ln_sa_done(SaDone{}) == True{} : Bool}

def ln_sa_done_of_done():
  {==}

  L.nat_le_refl(Nat.mul(c, e))

# promise: a speculative lane takes a batched step only when all the step's rows fit in its context.
law ln_spec_batched_fits:
  for +c: Nat
  for +r: Nat
  for +m: Nat
  {Bool.or(Bool.not(ln_spec_batched(c, r, m)), Nat.is_le(Nat.add(c, r), m)) == True{} : Bool}

def ln_spec_batched_fits(c, r, m):
  ln_or_not_self(Nat.is_le(Nat.add(c, r), m))

# promise: a drafter without batched steps (any kind other than plain decode and DSpine with checkpointed verify rows) never gets lanes.
law ln_other_drafter_no_lanes:
  {ln_drafter_ok(LdOther{}) == False{} : Bool}

def ln_other_drafter_no_lanes():
  {==}

def ln_min_is_input(+a: Nat, +b: Nat) -> {Bool.or(Nat.is_eq(Nat.min(a, b), a), Nat.is_eq(Nat.min(a, b), b)) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      ln_min_is_input(ap, bp)

# promise: lanes step at a draft width that was asked for.
law ln_lane_width_is_asked:
  for +a: Nat
  for +b: Nat
  {Bool.or(Nat.is_eq(ln_lane_width_pick(a, b), a), Nat.is_eq(ln_lane_width_pick(a, b), b)) == True{} : Bool}

def ln_lane_width_is_asked(a, b):
  ln_min_is_input(a, b)

# promise: a load whose drafter verifies its rows in a form the batched speculative steps are not built for never gets lanes.
law ln_unbatched_verify_no_lanes:
  for +d: Bool
  {ln_drafter_ok(ln_lane_drafter_of(True{}, d, False{})) == False{} : Bool}

def ln_unbatched_verify_no_lanes(d):
  match d:
    case True{}: {==}
    case False{}: {==}
