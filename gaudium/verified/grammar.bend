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
# The grammar itself (JSON lexing, the schema subset, the token byte strings) is a format defined
# by an outside reference and stays in the Rust shell (`gaudium/src/grammar/`) with Rust tests,
# among them a differential test against the crate's JSON parser.

import Base
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
# 4. Laws
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
