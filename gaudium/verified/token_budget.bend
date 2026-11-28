# Slice 2C — Verified Token / Context Budget Arithmetic & Sampler Compatibility Lattice
# Formally specifies and proves:
#   - Engine context reserve (128 tokens), served_ctx, prompt admission & max_new clamping
#   - Monotone non-underflowing remaining sequence budget & KV high-water cap
#   - Mixed-batch sampler family lattice & speculative method sampling compatibility

import Base

# ============================================================================
# 1. Helper Nat arithmetic lemmas (erased before Rust codegen)
# ============================================================================

def tb_le_refl(n: Nat) -> {True{} == Nat.is_le(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      tb_le_refl(p)

def tb_le_step_r_zero(xp: Nat, h: {False{} == True{} : Bool}) -> {Nat.is_le(xp, 0n) == True{} : Bool}:
  match xp:
    case 0n: {==}
    case 1n+_: h

def tb_le_step_r(x: Nat, y: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(x, 1n+y) == True{} : Bool}:
  match x y:
    case 0n _:
      {==}
    case 1n+xp 0n:
      tb_le_step_r_zero(xp, h)
    case 1n+xp 1n+yp:
      tb_le_step_r(xp, yp, h)

def tb_sub_le_self(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      Equal.sym(Bool, True{}, Nat.is_le(ap, ap), tb_le_refl(ap))
    case 1n+ap 1n+bp:
      tb_le_step_r(Nat.sub(ap, bp), ap, tb_sub_le_self(ap, bp))

def tb_sub_add_cancel(a: Nat, b: Nat) -> {Nat.sub(Nat.add(a, b), a) == b : Nat}:
  match a:
    case 0n:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      tb_sub_add_cancel(p, b)

def tb_sub_ge_zero(m: Nat, over: Nat) -> {Nat.sub(m, Nat.add(m, over)) == 0n : Nat}:
  match m:
    case 0n: {==}
    case 1n+p:
      tb_sub_ge_zero(p, over)

def tb_sub_succ_le(m: Nat, sent: Nat) -> {Nat.is_le(Nat.sub(m, 1n+sent), Nat.sub(m, sent)) == True{} : Bool}:
  match m sent:
    case 0n _:
      {==}
    case 1n+mp 0n:
      tb_sub_le_self(1n+mp, 1n)
    case 1n+mp 1n+sp:
      tb_sub_succ_le(mp, sp)

def tb_lt_add_r_false(a: Nat, extra: Nat) -> {False{} == Nat.is_lt(Nat.add(a, extra), a) : Bool}:
  match a:
    case 0n:
      match extra:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      tb_lt_add_r_false(p, extra)

def tb_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      tb_min_le_l(ap, bp)

def tb_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      tb_min_le_r(ap, bp)

def tb_bool_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def tb_lt_succ_add(+p: Nat, e: Nat) -> {Nat.is_lt(p, 1n+Nat.add(p, e)) == True{} : Bool}:
  match p:
    case 0n: {==}
    case 1n+q:
      tb_lt_succ_add(q, e)

# ============================================================================
# 2. Context & Token Budget Arithmetic (serve_proto.hpp, gaudium_engine.cpp)
# ============================================================================

def ctx_reserve() -> Nat:
  128n

# Advertised served context: Lmax - ctx_reserve(), saturating safely at 0 (no underflow).
def served_ctx(lmax: Nat) -> Nat:
  Nat.sub(lmax, ctx_reserve())

# Prompt is admitted iff 1 <= prompt_len < served_ctx(lmax) (leaves >= 1 token room).
def prompt_admitted(+prompt_len: Nat, lmax: Nat) -> Bool:
  Bool.and(Nat.is_gt(prompt_len, 0n), Nat.is_lt(prompt_len, served_ctx(lmax)))

# Context capacity a load sizes for its longest prompt of `prompt_len` tokens and one generated
# token: the prompt, that token and the engine's reserve.
def ctx_for_prompt(prompt_len: Nat) -> Nat:
  Nat.add(ctx_reserve(), 1n+prompt_len)

def prompt_room(prompt_len: Nat, lmax: Nat) -> Nat:
  Nat.sub(served_ctx(lmax), prompt_len)

# Clamps requested max_new to the available context room (unbounded => full room).
def effective_max_new(req_unbounded: Bool, req_max_new: Nat, prompt_len: Nat, lmax: Nat) -> Nat:
  match req_unbounded:
    case True{}:
      prompt_room(prompt_len, lmax)
    case False{}:
      Nat.min(req_max_new, prompt_room(prompt_len, lmax))

# Remaining sequence token budget (strictly non-underflowing).
def remaining_budget(sent: Nat, max_new: Nat) -> Nat:
  Nat.sub(max_new, sent)

# ============================================================================
# 3. Sampler Admission & Top-K Clamping
# ============================================================================

# Policy gate: greedy requests are always allowed; sampled requests require method capability.
def sampling_request_admitted(method_can_sample: Bool, wants_sampling: Bool) -> Bool:
  Bool.or(Bool.not(wants_sampling), method_can_sample)

def clamp_top_k(top_k: Nat, vocab: Nat) -> Nat:
  Nat.min(top_k, vocab)

# ============================================================================
# 4. Verified Token Budget & Sampler Laws
# ============================================================================

# LAW TB.1: served_ctx(lmax) NEVER exceeds lmax (no underflow).
law served_ctx_le_lmax:
  for +lmax: Nat
  {Nat.is_le(served_ctx(lmax), lmax) == True{} : Bool}

def served_ctx_le_lmax(lmax):
  tb_sub_le_self(lmax, ctx_reserve())

# LAW TB.2: For lmax = ctx_reserve() + extra, served_ctx recovers extra exactly.
# promise: served_ctx recovers extra exactly when lmax is ctx_reserve() + extra
# catches: served_ctx ignores ctx_reserve() and returns lmax unchanged
law served_ctx_above_reserve:
  for +extra: Nat
  {served_ctx(Nat.add(ctx_reserve(), extra)) == extra : Nat}

def served_ctx_above_reserve(extra):
  tb_sub_add_cancel(ctx_reserve(), extra)

# LAW TB.3: Empty prompt (len 0) is ALWAYS rejected.
law empty_prompt_rejected:
  for lmax: Nat
  {prompt_admitted(0n, lmax) == False{} : Bool}

def empty_prompt_rejected(lmax):
  {==}

# LAW TB.4: Any prompt >= served_ctx(lmax) is ALWAYS rejected (never silently truncated).
law oversized_prompt_rejected:
  for +lmax: Nat
  for +extra: Nat
  {prompt_admitted(Nat.add(served_ctx(lmax), extra), lmax) == False{} : Bool}

def oversized_prompt_rejected(lmax, extra):
  %tb_lt_add_r_false(served_ctx(lmax), extra) : {Bool.and(Nat.is_gt(Nat.add(served_ctx(lmax), extra), 0n), _) == False{} : Bool}
  tb_bool_and_false_r(Nat.is_gt(Nat.add(served_ctx(lmax), extra), 0n))

# LAW TB.5: Effective max_new NEVER exceeds available context room.
law effective_max_new_le_room:
  for unb: Bool
  for +req: Nat
  for +p: Nat
  for +lmax: Nat
  {Nat.is_le(effective_max_new(unb, req, p, lmax), prompt_room(p, lmax)) == True{} : Bool}

def effective_max_new_le_room(unb, req, p, lmax):
  match unb:
    case True{}:
      Equal.sym(Bool, True{}, Nat.is_le(prompt_room(p, lmax), prompt_room(p, lmax)), tb_le_refl(prompt_room(p, lmax)))
    case False{}:
      tb_min_le_r(req, prompt_room(p, lmax))

# LAW TB.6: Remaining budget reaches 0 at sent >= max_new (never underflows).
law remaining_zero_at_limit:
  for +m: Nat
  for +over: Nat
  {remaining_budget(Nat.add(m, over), m) == 0n : Nat}

def remaining_zero_at_limit(m, over):
  tb_sub_ge_zero(m, over)

# LAW TB.7: Emitting a token monotonically decreases (or keeps at 0) remaining budget.
law remaining_monotone_step:
  for +sent: Nat
  for +m: Nat
  {Nat.is_le(remaining_budget(1n+sent, m), remaining_budget(sent, m)) == True{} : Bool}

def remaining_monotone_step(sent, m):
  tb_sub_succ_le(m, sent)

# LAW TB.10: Any method without sampling capability refuses sampled requests; greedy is always admitted.
law unsupported_method_refuses_sampling:
  {sampling_request_admitted(False{}, True{}) == False{} : Bool}

def unsupported_method_refuses_sampling():
  {==}

law greedy_request_always_admitted:
  for cap: Bool
  {sampling_request_admitted(cap, False{}) == True{} : Bool}

def greedy_request_always_admitted(cap):
  match cap:
    case True{}: {==}
    case False{}: {==}

# LAW TB.11: Clamped top_k NEVER exceeds vocabulary size.
law top_k_clamped_to_vocab:
  for +k: Nat
  for +v: Nat
  {Nat.is_le(clamp_top_k(k, v), v) == True{} : Bool}

def top_k_clamped_to_vocab(k, v):
  tb_min_le_r(k, v)

# promise: a capacity of at least ctx_for_prompt(p) admits a non-empty prompt of p tokens, also after it is rounded up to whole cache chunks
# harm: a calibration load would refuse its own longest prompt after the engine had loaded
law ctx_for_prompt_admits:
  for +p: Nat
  for +extra: Nat
  {prompt_admitted(1n+p, Nat.add(ctx_for_prompt(1n+p), extra)) == True{} : Bool}

def ctx_for_prompt_admits(p, extra):
  tb_lt_succ_add(p, extra)

# LAW TB.12: Bounded requested max_new is never exceeded by effective_max_new.
# promise: effective_max_new with req_unbounded=False never exceeds req_max_new
# catches: effective_max_new returns prompt_room ignoring req_max_new
law effective_max_new_respects_req:
  for +req: Nat
  for +p: Nat
  for +lmax: Nat
  {Nat.is_le(effective_max_new(False{}, req, p, lmax), req) == True{} : Bool}

def effective_max_new_respects_req(req, p, lmax):
  tb_min_le_l(req, prompt_room(p, lmax))

# LAW TB.13: Clamped top_k NEVER exceeds the caller's requested top_k.
# promise: clamp_top_k never exceeds the input top_k
# catches: clamp_top_k returns vocab without clamping to top_k
law clamp_top_k_within_input:
  for +k: Nat
  for +v: Nat
  {Nat.is_le(clamp_top_k(k, v), k) == True{} : Bool}

def clamp_top_k_within_input(k, v):
  tb_min_le_l(k, v)

# LAW TB.14: A method supporting sampling always admits both greedy and sampled requests.
# promise: sampling_request_admitted returns True whenever method_can_sample is True
# catches: sampling_request_admitted returns False when method_can_sample and wants_sampling are True
law sampling_admitted_when_allowed:
  for +wants: Bool
  {sampling_request_admitted(True{}, wants) == True{} : Bool}

def sampling_admitted_when_allowed(wants):
  match wants:
    case True{}: {==}
    case False{}: {==}

# promise: after sent tokens of a max_new of sent + e, the remaining budget is exactly e (all that is left, never more)
# harm: an answer that resumes after a forced end of thinking would run past max_tokens, or lose tokens it was owed
# catches: an off-by-one in the remaining budget (max_new minus sent minus one)
law remaining_budget_exact:
  for +s: Nat
  for +e: Nat
  {remaining_budget(s, Nat.add(s, e)) == e : Nat}

def remaining_budget_exact(s, e):
  tb_sub_add_cancel(s, e)
