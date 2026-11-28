# Verified Constrained-Decoding Core (structured output: JSON mode, JSON schema, strict tools)
# Formally specifies and proves the decisions the engine makes around a constrained request:
#   - Which requests need their tokens picked on the host (a penalised or a constrained one)
#   - Whether a token the grammar refuses may be emitted (never: it is classed as banned for the
#     verified emission check of the penalised pick)
#   - Whether a token is admitted at a position: an end-of-sequence token only when the reply is
#     complete, any other token only when its bytes keep the grammar alive
#   - Whether the pick may go ahead: never with no admitted token left (the request then ends
#     with an error, never with the sampler's fallback token)
#   - The reply's phase: free while the model reasons, constrained from the end of its reasoning
#     on, and never free again
#   - Whether the reply may end: only in the constrained phase, with the grammar complete
#   - The automaton's capacity bounds: how deep a stack may grow, and how many alternatives the
#     automaton follows at once
#   - The budgets a client's schema is held to: a charge to a budget (the bytes a grammar may
#     compile to, the work of one mask or one commit) never takes it past its cap
#   - The mask cache's byte ledger: what it holds never passes its cap
#   - Where a control token (one that adds no text) is free: in the reasoning only
# The grammar itself (JSON lexing, the schema subset, the token byte strings) is a format defined
# by an outside reference and stays in the Rust shell (`gaudium/src/grammar/`) with Rust tests,
# among them a differential test against the crate's JSON parser.

import Base
import nat_lemmas.bend as L
import sampler_penalty.bend as P

# ============================================================================
# 1. Host Pick and Emission
# ============================================================================

# penalised:   the request carries penalties or a logit_bias
# constrained: the request carries a grammar (response format or strict tools)
# The request's tokens are picked on the host, where both are applied.
def gram_host_pick(penalised: Bool, constrained: Bool) -> Bool:
  Bool.or(penalised, constrained)

# The class the emission check sees for a picked token: its logit_bias class when the grammar
# admits it, a ban when it does not.
def gram_token_class(admitted: Bool, bias: P.PenBias) -> P.PenBias:
  match admitted:
    case True{}:
      bias
    case False{}:
      P.PbBan{}

# is_end:    the token ends the sequence (an end-of-sequence id)
# accepting: the reply may end here (gram_accepting)
# steps:     the token's bytes keep the grammar alive
def gram_admit(is_end: Bool, accepting: Bool, steps: Bool) -> Bool:
  match is_end:
    case True{}:
      accepting
    case False{}:
      steps

# admitted: tokens the mask admits at this position
def gram_can_pick(admitted: Nat) -> Bool:
  Nat.is_lt(0n, admitted)

# ============================================================================
# 2. Reply Phase
# ============================================================================

# GpThink:  inside the model's reasoning (free text; the constraint waits for its end)
# GpAnswer: the answer (every token constrained)
type GramPhase is Data:
  GpThink{}
  GpAnswer{}

# closed: the token just committed completes the reasoning's closing tag
def gram_phase_step(ph: GramPhase, closed: Bool) -> GramPhase:
  match ph:
    case GpThink{}:
      match closed:
        case True{}:
          GpAnswer{}
        case False{}:
          GpThink{}
    case GpAnswer{}:
      GpAnswer{}

# complete: the grammar accepts the reply as it stands
def gram_accepting(ph: GramPhase, complete: Bool) -> Bool:
  match ph:
    case GpThink{}:
      False{}
    case GpAnswer{}:
      complete

# ============================================================================
# 3. Automaton Capacity
# ============================================================================

# depth: frames on a stack (or nesting levels of a schema); cap: the most it may hold
# A frame may be pushed only while the stack holds fewer than cap.
def gram_push_ok(depth: Nat, cap: Nat) -> Bool:
  Nat.is_lt(depth, cap)

# alive: alternatives the automaton follows after a byte; cap: the most it may follow
def gram_alive_ok(alive: Nat, cap: Nat) -> Bool:
  Nat.is_le(alive, cap)

# ============================================================================
# 4. Budgets and the Mask Cache
# ============================================================================

# used: what a budget has been charged so far; add: the next charge; cap: the budget
# A charge is accepted only when the ledger is within its cap and the charge fits in what is
# left (written without the sum, so no client-sized charge can overflow it).
def gram_budget_ok(+used: Nat, add: Nat, +cap: Nat) -> Bool:
  Bool.and(Nat.is_le(used, cap), Nat.is_le(add, Nat.sub(cap, used)))

# What the mask cache does with a new entry: keep it beside what it holds, or empty itself
# (the entry is then not kept).
type GramCacheAct is Data:
  GcKeep{}
  GcClear{}

def gram_cache_pick(fits: Bool) -> GramCacheAct:
  match fits:
    case True{}:
      GcKeep{}
    case False{}:
      GcClear{}

# held: bytes the cache holds; add: bytes of the new entry; cap: the most it may hold
def gram_cache_act(+held: Nat, +add: Nat, +cap: Nat) -> GramCacheAct:
  gram_cache_pick(gram_budget_ok(held, add, cap))

# The bytes the cache holds after the action.
def gram_cache_held(act: GramCacheAct, held: Nat, add: Nat) -> Nat:
  match act:
    case GcKeep{}:
      Nat.add(held, add)
    case GcClear{}:
      0n

# A control token adds no text: it is free while the model reasons, never in the answer.
def gram_control_free(ph: GramPhase) -> Bool:
  match ph:
    case GpThink{}:
      True{}
    case GpAnswer{}:
      False{}

# 1 + (r + e) <= r never holds
def gram_le_succ_add_false(+r: Nat, +e: Nat) -> {Nat.is_le(1n+Nat.add(r, e), r) == False{} : Bool}:
  match r:
    case 0n: {==}
    case 1n+p:
      gram_le_succ_add_false(p, e)

# a charge against a budget of h + r after h is a <= r
def gram_budget_shift(+h: Nat, +a: Nat, +r: Nat) -> {gram_budget_ok(h, a, Nat.add(h, r)) == Nat.is_le(a, r) : Bool}:
  match h:
    case 0n:
      match r:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      gram_budget_shift(p, a, r)

# h + a <= h + r is a <= r
def gram_le_add_shift(+h: Nat, +a: Nat, +r: Nat) -> {Nat.is_le(a, r) == Nat.is_le(Nat.add(h, a), Nat.add(h, r)) : Bool}:
  match h:
    case 0n: {==}
    case 1n+p:
      gram_le_add_shift(p, a, r)

def gram_cache_bound_go(+h: Nat, +a: Nat, +r: Nat, f: Bool, ef: {gram_budget_ok(h, a, Nat.add(h, r)) == f : Bool}) -> {Nat.is_le(gram_cache_held(gram_cache_pick(f), h, a), Nat.add(h, r)) == True{} : Bool}:
  match f:
    case True{}:
      %gram_le_add_shift(h, a, r) : {_ == True{} : Bool}
      %gram_budget_shift(h, a, r) : {_ == True{} : Bool}
      ef
    case False{}:
      L.nat_le_zero(Nat.add(h, r))

# ============================================================================
# 5. Laws
# ============================================================================

# promise: a constrained request always has its tokens picked on the host, where the mask applies
# harm: a constrained request on a card-picked path would emit tokens the grammar refuses
# catches: gram_host_pick returns penalised (the grammar ignored)
law gram_constrained_host_picked:
  for p: Bool
  {gram_host_pick(p, True{}) == True{} : Bool}

def gram_constrained_host_picked(p):
  match p:
    case True{}: {==}
    case False{}: {==}

# promise: a penalised request always has its tokens picked on the host
# catches: gram_host_pick returns constrained (the penalties ignored)
law gram_penalised_host_picked:
  for c: Bool
  {gram_host_pick(True{}, c) == True{} : Bool}

def gram_penalised_host_picked(c):
  {==}

# promise: a request with neither penalties nor a grammar keeps every decode path
# catches: gram_host_pick returns True{} (every request forced to the host path)
law gram_plain_not_host_picked:
  {gram_host_pick(False{}, False{}) == False{} : Bool}

def gram_plain_not_host_picked():
  {==}

# promise: a token the grammar refuses is never emitted, whatever its id, vocabulary or bias
# harm: the reply would leave its JSON schema or tool schema
# catches: gram_token_class returns bias whatever admitted says
law gram_refused_never_emitted:
  for t: Nat
  for v: Nat
  for b: P.PenBias
  {P.pen_emit_ok(t, v, gram_token_class(False{}, b)) == False{} : Bool}

def gram_refused_never_emitted(t, v, b):
  {==}

# promise: a token the grammar admits keeps its logit_bias class (a ban stays a ban, a plain token stays emittable)
# catches: gram_token_class returns P.PbBan{} for an admitted token too
law gram_admitted_keeps_bias:
  for b: P.PenBias
  {gram_token_class(True{}, b) == b : P.PenBias}

def gram_admitted_keeps_bias(b):
  {==}

# promise: an end-of-sequence token is never admitted while the reply is incomplete
# harm: the reply would stop with half a JSON value or half a tool call
# catches: gram_admit returns steps for an end token (its empty byte string always steps)
law gram_end_refused_incomplete:
  for s: Bool
  {gram_admit(True{}, False{}, s) == False{} : Bool}

def gram_end_refused_incomplete(s):
  {==}

# promise: an end-of-sequence token is admitted once the reply is complete
# harm: a complete reply could never stop and would run to its token limit
# catches: gram_admit returns False{} for every end token
law gram_end_admitted_complete:
  for s: Bool
  {gram_admit(True{}, True{}, s) == True{} : Bool}

def gram_end_admitted_complete(s):
  {==}

# promise: any other token is admitted exactly when its bytes keep the grammar alive
# catches: gram_admit returns Bool.or(accepting, steps) for a content token (anything admitted after a complete reply)
law gram_content_follows_steps:
  for a: Bool
  for s: Bool
  {gram_admit(False{}, a, s) == s : Bool}

def gram_content_follows_steps(a, s):
  {==}

# promise: a mask that admits no token never leads to a pick
# harm: the sampler would return its fallback token, which the grammar does not admit
# catches: gram_can_pick uses Nat.is_le(0n, admitted) (an empty mask picks)
law gram_empty_mask_refused:
  {gram_can_pick(0n) == False{} : Bool}

def gram_empty_mask_refused():
  {==}

# promise: a mask that admits at least one token leads to a pick
# catches: gram_can_pick returns False{} (every constrained request ends in an error)
law gram_nonempty_mask_picks:
  for n: Nat
  {gram_can_pick(1n+n) == True{} : Bool}

def gram_nonempty_mask_picks(n):
  {==}

# promise: once the answer has begun, the reply never returns to the free reasoning phase
# harm: tokens after a second closing tag would escape the constraint
# catches: the GpAnswer arm of gram_phase_step returns GpThink{} when closed (a second closing tag frees the reply)
law gram_answer_absorbing:
  for c: Bool
  {gram_phase_step(GpAnswer{}, c) == GpAnswer{} : GramPhase}

def gram_answer_absorbing(c):
  {==}

# promise: the reasoning's closing tag starts the constrained answer
# catches: gram_phase_step returns ph (the constraint never takes over)
law gram_think_closes_into_answer:
  {gram_phase_step(GpThink{}, True{}) == GpAnswer{} : GramPhase}

def gram_think_closes_into_answer():
  {==}

# promise: inside the reasoning, a token that does not close it leaves the reply free
# harm: the schema would be applied to the model's reasoning
# catches: gram_phase_step returns GpAnswer{} for every token (the constraint starts inside the reasoning)
law gram_think_stays_free:
  {gram_phase_step(GpThink{}, False{}) == GpThink{} : GramPhase}

def gram_think_stays_free():
  {==}

# promise: a reply never ends inside the model's reasoning
# harm: a constrained request would end with no answer at all
# catches: gram_accepting returns complete in the GpThink arm (the empty reasoning grammar is complete)
law gram_think_never_accepts:
  for c: Bool
  {gram_accepting(GpThink{}, c) == False{} : Bool}

def gram_think_never_accepts(c):
  {==}

# promise: in the answer, the reply may end exactly when the grammar accepts it
# catches: gram_accepting returns True{} in the GpAnswer arm (an incomplete answer may end)
law gram_answer_accepts_when_complete:
  for c: Bool
  {gram_accepting(GpAnswer{}, c) == c : Bool}

def gram_answer_accepts_when_complete(c):
  {==}

# promise: a frame is never pushed onto a stack that already holds the capacity or more
# harm: a hostile schema or a runaway recursion would grow the automaton's stack without bound
# catches: gram_push_ok uses Nat.is_le (one frame past the capacity pushed)
law gram_push_at_cap_refused:
  for +c: Nat
  for +e: Nat
  {gram_push_ok(Nat.add(c, e), c) == False{} : Bool}

def gram_push_at_cap_refused(c, e):
  match c:
    case 0n:
      match e:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+q:
      gram_push_at_cap_refused(q, e)

# promise: a frame may be pushed onto a stack below its capacity
# catches: gram_push_ok returns False{} (no frame ever pushed)
law gram_push_below_cap_ok:
  for +d: Nat
  for e: Nat
  {gram_push_ok(d, 1n+Nat.add(d, e)) == True{} : Bool}

def gram_push_below_cap_ok(d, e):
  match d:
    case 0n: {==}
    case 1n+q:
      gram_push_below_cap_ok(q, e)

# promise: the automaton never follows more alternatives than its capacity
# harm: an ambiguous schema would make every step's work grow without bound
# catches: gram_alive_ok returns True{} (any number of alternatives followed)
law gram_alive_past_cap_refused:
  for +c: Nat
  for e: Nat
  {gram_alive_ok(1n+Nat.add(c, e), c) == False{} : Bool}

def gram_alive_past_cap_refused(c, e):
  match c:
    case 0n: {==}
    case 1n+q:
      gram_alive_past_cap_refused(q, e)

# promise: the automaton may follow as many alternatives as its capacity
# catches: gram_alive_ok uses Nat.is_lt (a set exactly at the capacity refused)
law gram_alive_within_cap_ok:
  for +n: Nat
  for +e: Nat
  {gram_alive_ok(n, Nat.add(n, e)) == True{} : Bool}

def gram_alive_within_cap_ok(n, e):
  match n:
    case 0n:
      match e:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+q:
      gram_alive_within_cap_ok(q, e)

# promise: a charge that does not fit in what is left of a budget is refused
# harm: a hostile schema would make the compiler, or one mask, take memory or time without bound
# catches: gram_budget_ok uses Nat.is_le(add, cap) (what is already spent ignored)
law gram_budget_past_cap_refused:
  for +u: Nat
  for +r: Nat
  for +e: Nat
  {gram_budget_ok(u, 1n+Nat.add(r, e), Nat.add(u, r)) == False{} : Bool}

def gram_budget_past_cap_refused(u, r, e):
  %Equal.sym(Bool, gram_budget_ok(u, 1n+Nat.add(r, e), Nat.add(u, r)), Nat.is_le(1n+Nat.add(r, e), r), gram_budget_shift(u, 1n+Nat.add(r, e), r)) : {_ == False{} : Bool}
  gram_le_succ_add_false(r, e)

# promise: a charge that fits in what is left of a budget is accepted
# catches: gram_budget_ok uses Nat.is_lt(add, Nat.sub(cap, used)) (a charge that fills the budget exactly refused)
law gram_budget_within_cap_ok:
  for +u: Nat
  for +a: Nat
  for +e: Nat
  {gram_budget_ok(u, a, Nat.add(u, Nat.add(a, e))) == True{} : Bool}

def gram_budget_within_cap_ok(u, a, e):
  %Equal.sym(Bool, gram_budget_ok(u, a, Nat.add(u, Nat.add(a, e))), Nat.is_le(a, Nat.add(a, e)), gram_budget_shift(u, a, Nat.add(a, e))) : {_ == True{} : Bool}
  L.nat_add_le_r(a, e)

# promise: a ledger already past its budget accepts no charge, not even an empty one
# catches: gram_budget_ok drops the Nat.is_le(used, cap) term (a saturated Nat.sub leaves room 0, and a zero charge passes)
law gram_budget_over_refused:
  for +c: Nat
  for +e: Nat
  for +a: Nat
  {gram_budget_ok(1n+Nat.add(c, e), a, c) == False{} : Bool}

def gram_budget_over_refused(c, e, a):
  %Equal.sym(Bool, Nat.is_le(1n+Nat.add(c, e), c), False{}, gram_le_succ_add_false(c, e)) : {Bool.and(_, Nat.is_le(a, Nat.sub(c, 1n+Nat.add(c, e)))) == False{} : Bool}
  {==}

# promise: a mask cache within its cap stays within it after any new entry
# harm: a schema whose every step is a new state (a counted array) would fill the host's memory with masks
# catches: gram_cache_act returns GcKeep{} whatever the cap (every mask kept)
law gram_cache_within_cap:
  for +h: Nat
  for +r: Nat
  for +a: Nat
  {Nat.is_le(gram_cache_held(gram_cache_act(h, a, Nat.add(h, r)), h, a), Nat.add(h, r)) == True{} : Bool}

def gram_cache_within_cap(h, r, a):
  gram_cache_bound_go(h, a, r, gram_budget_ok(h, a, Nat.add(h, r)), {==})

# promise: an entry that fits beside what the cache holds is kept
# catches: gram_cache_act returns GcClear{} (nothing ever cached)
law gram_cache_keeps_what_fits:
  for +h: Nat
  for +a: Nat
  for +e: Nat
  {gram_cache_act(h, a, Nat.add(h, Nat.add(a, e))) == GcKeep{} : GramCacheAct}

def gram_cache_keeps_what_fits(h, a, e):
  %Equal.sym(Bool, gram_budget_ok(h, a, Nat.add(h, Nat.add(a, e))), True{}, gram_budget_within_cap_ok(h, a, e)) : {gram_cache_pick(_) == GcKeep{} : GramCacheAct}
  {==}

# promise: a kept entry's bytes are counted beside what the cache held
# harm: an undercounted ledger would let the cache grow past its cap
# catches: the GcKeep arm of gram_cache_held returns held (the new entry not counted)
law gram_cache_keep_counts:
  for h: Nat
  for a: Nat
  {gram_cache_held(GcKeep{}, h, a) == Nat.add(h, a) : Nat}

def gram_cache_keep_counts(h, a):
  {==}

# promise: an emptied cache counts no bytes
# catches: the GcClear arm of gram_cache_held returns held (the ledger keeps what was freed, and the cache stops caching)
law gram_cache_clear_empties:
  for h: Nat
  for a: Nat
  {gram_cache_held(GcClear{}, h, a) == 0n : Nat}

def gram_cache_clear_empties(h, a):
  {==}

# promise: a control token is never admitted in the constrained answer
# harm: a token that adds no text would pass the mask, and the model could end or switch modes outside the format
# catches: gram_control_free returns True{} in the GpAnswer arm
law gram_control_refused_in_answer:
  {gram_control_free(GpAnswer{}) == False{} : Bool}

def gram_control_refused_in_answer():
  {==}

# promise: while the model reasons, a control token stays free
# catches: gram_control_free returns False{} in the GpThink arm (the reasoning loses its control tokens)
law gram_control_free_in_think:
  {gram_control_free(GpThink{}) == True{} : Bool}

def gram_control_free_in_think():
  {==}
