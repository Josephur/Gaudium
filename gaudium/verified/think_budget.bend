# Slice — Verified Thinking Budget
# Formally specifies and proves how a request's thinking budget (`thinking.budget_tokens` of the
# Messages API) caps the reasoning span of a reply:
#   - The cap: how many reasoning tokens the reply may hold, inside the budget and leaving room
#     for the end-of-thinking marker in the request's token budget
#   - The step taken for each generated token: it goes on, or the reasoning is closed in its
#     place (the marker is written as if generated and the reply continues with the answer), or
#     the reply ends because the marker no longer fits
#   - What the reply may still generate after the forced close (`remaining_budget` of
#     `token_budget.bend`, given the tokens written so far and the marker)
#
# Counts are tokens of the reply. `n` counts the reasoning tokens kept so far, `room` the tokens
# the reply may still generate (`remaining_budget(sent, max_new)`), `close` the marker's length in
# tokens, `forced` whether the marker was already forced once.

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. Types
# ============================================================================

# What happens to a generated token.
type ThinkAct is Data:
  TaFeed{n: Nat}  # the token goes on; `n`: the reasoning tokens kept after it
  TaClose{}       # the token is dropped; the end-of-thinking marker takes its place
  TaEnd{}         # the token is dropped and the reply ends (the marker does not fit)

# ============================================================================
# 2. Helper lemmas (erased before Rust codegen)
# ============================================================================

def th_min_add_l(+b: Nat, k: Nat) -> {Nat.min(b, Nat.add(b, k)) == b : Nat}:
  match b:
    case 0n:
      {==}
    case 1n+p:
      %th_min_add_l(p, k) : {1n+Nat.min(p, Nat.add(p, k)) == 1n+_ : Nat}
      {==}

# ============================================================================
# 3. The cap and the step
# ============================================================================

# The reasoning tokens a reply may keep: the budget, cut so that the end-of-thinking marker of
# `close` tokens still fits in the reply's `max_new` after them.
def think_cap(budget: Nat, max_new: Nat, close: Nat) -> Nat:
  Nat.min(budget, Nat.sub(max_new, close))

def think_act_in(below: Bool, n: Nat, forced: Bool, fits: Bool) -> ThinkAct:
  match below:
    case True{}:
      TaFeed{1n+n}
    case False{}:
      match forced:
        case True{}:
          TaEnd{}
        case False{}:
          match fits:
            case True{}:
              TaClose{}
            case False{}:
              TaEnd{}

# The step for one generated token. `in_thought`: the reply is inside its reasoning span before
# this token.
def think_act(in_thought: Bool, +n: Nat, cap: Nat, close: Nat, room: Nat, forced: Bool) -> ThinkAct:
  match in_thought:
    case False{}:
      TaFeed{n}
    case True{}:
      think_act_in(Nat.is_lt(n, cap), n, forced, Nat.is_le(close, room))

def think_is_feed(a: ThinkAct) -> Bool:
  match a:
    case TaFeed{n}:
      True{}
    case _:
      False{}

def think_is_close(a: ThinkAct) -> Bool:
  match a:
    case TaClose{}:
      True{}
    case _:
      False{}

# The reasoning count a step leaves (`n` when the token is dropped).
def think_kept(a: ThinkAct, n: Nat) -> Nat:
  match a:
    case TaFeed{m}:
      m
    case _:
      n

# Helper lemmas over the step (erased before Rust codegen).

def th_in_not_below_not_feed(n: Nat, forced: Bool, fits: Bool) -> {think_is_feed(think_act_in(False{}, n, forced, fits)) == False{} : Bool}:
  match forced fits:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

def th_in_nofit_not_close(below: Bool, n: Nat, forced: Bool) -> {think_is_close(think_act_in(below, n, forced, False{})) == False{} : Bool}:
  match below forced:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

def th_in_forced_not_close(below: Bool, n: Nat, fits: Bool) -> {think_is_close(think_act_in(below, n, True{}, fits)) == False{} : Bool}:
  match below fits:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# ============================================================================
# 4. Laws
# ============================================================================

# promise: the reasoning cap never exceeds the budget the request gave
# harm: a reply would reason past the client's budget_tokens and spend its answer tokens on it
# catches: the budget dropped from the cap (the cap is max_new minus the marker alone)
law think_cap_within_budget:
  for +b: Nat
  for +m: Nat
  for +c: Nat
  {Nat.is_le(think_cap(b, m, c), b) == True{} : Bool}

def think_cap_within_budget(b, m, c):
  L.nat_min_le_l(b, Nat.sub(m, c))

# promise: when the budget, the marker and at least one answer token fit in max_new, the reply may reason for its whole budget
# catches: the cap replaced by a constant 0, or the marker subtracted twice
law think_cap_full_budget:
  for +b: Nat
  for +c: Nat
  for +e: Nat
  {think_cap(b, Nat.add(c, Nat.add(b, 1n+e)), c) == b : Nat}

def think_cap_full_budget(b, c, e):
  %Equal.sym(Nat, Nat.sub(Nat.add(c, Nat.add(b, 1n+e)), c), Nat.add(b, 1n+e), L.nat_add_sub_cancel_l(c, Nat.add(b, 1n+e))) : {Nat.min(b, _) == b : Nat}
  th_min_add_l(b, 1n+e)

# promise: the reasoning cap leaves room for the whole end-of-thinking marker inside max_new whenever the marker fits at all
# harm: a reply would reason up to a cap where the marker no longer fits and end without its answer
# catches: the marker not subtracted (the cap is the smaller of the budget and max_new)
law think_cap_leaves_close:
  for +b: Nat
  for +c: Nat
  for +e: Nat
  {Nat.is_le(Nat.add(c, think_cap(b, Nat.add(c, e), c)), Nat.add(c, e)) == True{} : Bool}

def think_cap_leaves_close(b, c, e):
  %Equal.sym(Nat, Nat.sub(Nat.add(c, e), c), e, L.nat_add_sub_cancel_l(c, e)) : {Nat.is_le(Nat.add(c, Nat.min(b, _)), Nat.add(c, e)) == True{} : Bool}
  L.nat_add_le_mono(c, c, Nat.min(b, e), e, L.nat_le_refl(c), L.nat_min_le_r(b, e))

# promise: a reasoning token at or past the cap is never kept
# harm: the reasoning span would outgrow the budget
# catches: the comparison n < cap replaced by n <= cap (one reasoning token past the budget)
law think_never_past_cap:
  for +cap: Nat
  for +e: Nat
  for +c: Nat
  for +room: Nat
  for +forced: Bool
  {think_is_feed(think_act(True{}, Nat.add(cap, e), cap, c, room, forced)) == False{} : Bool}

def think_never_past_cap(cap, e, c, room, forced):
  %Equal.sym(Bool, Nat.is_lt(Nat.add(cap, e), cap), False{}, G.g_nat_lt_add_r_false(cap, e)) : {think_is_feed(think_act_in(_, Nat.add(cap, e), forced, Nat.is_le(c, room))) == False{} : Bool}
  th_in_not_below_not_feed(Nat.add(cap, e), forced, Nat.is_le(c, room))

# promise: a reasoning token below the cap is kept and counted once
# catches: a kept reasoning token not counted (the count stays at n, so the cap is never reached)
law think_below_cap_counted:
  for +n: Nat
  for +d: Nat
  for +c: Nat
  for +room: Nat
  for +forced: Bool
  {think_kept(think_act(True{}, n, Nat.add(n, 1n+d), c, room, forced), n) == 1n+n : Nat}

def think_below_cap_counted(n, d, c, room, forced):
  %Equal.sym(Bool, Nat.is_lt(n, Nat.add(n, 1n+d)), True{}, G.g_nat_lt_add_succ(n, d)) : {think_kept(think_act_in(_, n, forced, Nat.is_le(c, room)), n) == 1n+n : Nat}
  {==}

# promise: a token of the answer is always kept by the thinking budget and leaves the reasoning count unchanged
# catches: in_thought ignored (answer tokens cut once the cap is reached)
law think_answer_kept:
  for +n: Nat
  for +cap: Nat
  for +c: Nat
  for +room: Nat
  for +forced: Bool
  {think_act(False{}, n, cap, c, room, forced) == TaFeed{n} : ThinkAct}

def think_answer_kept(n, cap, c, room, forced):
  {==}

# promise: the end-of-thinking marker is forced only where all of it fits in what the reply may still generate
# harm: the forced marker would push the reply past its max_new and the context it was admitted with
# catches: the fit check dropped (the marker forced whatever room is left)
law think_close_fits:
  for +t: Bool
  for +n: Nat
  for +cap: Nat
  for +room: Nat
  for +d: Nat
  for +forced: Bool
  {think_is_close(think_act(t, n, cap, Nat.add(room, 1n+d), room, forced)) == False{} : Bool}

def think_close_fits(t, n, cap, room, d, forced):
  match t:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, Nat.is_le(Nat.add(room, 1n+d), room), False{}, G.g_nat_le_add_succ_false(room, d)) : {think_is_close(think_act_in(Nat.is_lt(n, cap), n, forced, _)) == False{} : Bool}
      th_in_nofit_not_close(Nat.is_lt(n, cap), n, forced)

# promise: the end-of-thinking marker is forced at most once in a reply
# catches: the forced flag ignored (a reply that reasons again gets a second forced marker)
law think_close_once:
  for +t: Bool
  for +n: Nat
  for +cap: Nat
  for +c: Nat
  for +room: Nat
  {think_is_close(think_act(t, n, cap, c, room, True{})) == False{} : Bool}

def think_close_once(t, n, cap, c, room):
  match t:
    case False{}:
      {==}
    case True{}:
      th_in_forced_not_close(Nat.is_lt(n, cap), n, Nat.is_le(c, room))

# promise: a reply that reaches the cap inside its reasoning, with room for the marker and not yet forced, gets the marker
# catches: the close never chosen (the reply ends at the cap instead of going on to its answer)
law think_close_at_cap:
  for +n: Nat
  for +c: Nat
  for +e: Nat
  {think_is_close(think_act(True{}, n, n, c, Nat.add(c, e), False{})) == True{} : Bool}

def think_close_at_cap(n, c, e):
  %Equal.sym(Bool, Nat.is_lt(n, n), False{}, G.g_nat_lt_irrefl(n)) : {think_is_close(think_act_in(_, n, False{}, Nat.is_le(c, Nat.add(c, e)))) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_le(c, Nat.add(c, e)), True{}, L.nat_add_le_r(c, e)) : {think_is_close(think_act_in(False{}, n, False{}, _)) == True{} : Bool}
  {==}
