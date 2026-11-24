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

# Advertised served context: Lmax - 128, saturating safely at 0 (no underflow).
def served_ctx(lmax: Nat) -> Nat:
  Nat.sub(lmax, 128n)

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

# Speculative step safety headroom: verify + draft rows stay within Lmax (3*Kmax + 8 margin).
def step_headroom_fits(cursor: Nat, kmax: Nat, lmax: Nat) -> Bool:
  Nat.is_le(Nat.add(cursor, Nat.add(Nat.mul(3n, kmax), 8n)), lmax)

# KV high-water mark for post-request prefix clean: clamped to <= lmax, div-by-zero safe.
def kv_hw_round_if(+e: Nat, +kv_blk: Nat, is_zero: Bool) -> Nat:
  match is_zero:
    case True{}:
      e
    case False{}:
      Nat.add(Nat.mul(Nat.div(Nat.add(e, Nat.sub(kv_blk, 1n)), kv_blk), kv_blk), kv_blk)

def kv_high_water(prompt_len: Nat, cursor: Nat, kmax: Nat, +kv_blk: Nat, +lmax: Nat) -> Nat:
  Nat.min(
    lmax,
    kv_hw_round_if(
      Nat.add(Nat.max(prompt_len, cursor), Nat.add(Nat.mul(3n, kmax), 8n)),
      kv_blk,
      Nat.is_eq(kv_blk, 0n)
    )
  )

# ============================================================================
# 3. Sampler Family Lattice & Method Compatibility (backend.hpp, gaudium_engine.cpp)
# ============================================================================

type SamplerFamily is Data:
  SampGreedy{}
  SampNoSearch{}
  SampSearch{}

type SpecMethod is Data:
  MethodMtpChain{}
  MethodMtpAdaptive{}
  MethodDSpineChain{}
  MethodDSpineAdaptive{}
  MethodDFlash2{}
  MethodCertified{}
  MethodPromptLookup{}
  MethodMtpTree{}
  MethodDSpineFixedTree{}

def method_allows_sampling(m: SpecMethod) -> Bool:
  match m:
    case MethodMtpChain{}:
      True{}
    case MethodMtpAdaptive{}:
      True{}
    case MethodDSpineChain{}:
      True{}
    case MethodDSpineAdaptive{}:
      True{}
    case _:
      False{}

# Policy gate: greedy requests are always allowed; sampled requests require method capability.
def sampling_request_admitted(method_can_sample: Bool, wants_sampling: Bool) -> Bool:
  Bool.or(Bool.not(wants_sampling), method_can_sample)

# Mixed-batch sampler lattice join: Search > NoSearch > Greedy (GAUDIUM_BATCH_MIX).
def sampler_family_join(a: SamplerFamily, b: SamplerFamily) -> SamplerFamily:
  match a:
    case SampSearch{}:
      SampSearch{}
    case SampNoSearch{}:
      match b:
        case SampSearch{}:
          SampSearch{}
        case SampNoSearch{}:
          SampNoSearch{}
        case SampGreedy{}:
          SampNoSearch{}
    case SampGreedy{}:
      b

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
  tb_sub_le_self(lmax, 128n)

# LAW TB.2: For lmax = 128 + extra, served_ctx recovers extra exactly.
law served_ctx_above_reserve:
  for +extra: Nat
  {served_ctx(Nat.add(128n, extra)) == extra : Nat}

def served_ctx_above_reserve(extra):
  tb_sub_add_cancel(128n, extra)

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

# LAW TB.8: KV high-water mark is ALWAYS clamped to <= lmax.
law kv_hw_le_lmax:
  for +p: Nat
  for +c: Nat
  for +k: Nat
  for +b: Nat
  for +lmax: Nat
  {Nat.is_le(kv_high_water(p, c, k, b, lmax), lmax) == True{} : Bool}

def kv_hw_le_lmax(p, c, k, b, lmax):
  tb_min_le_l(
    lmax,
    kv_hw_round_if(
      Nat.add(Nat.max(p, c), Nat.add(Nat.mul(3n, k), 8n)),
      b,
      Nat.is_eq(b, 0n)
    )
  )

# LAW TB.9: Sampler mixed-batch family join is commutative, idempotent, and associative.
law sampler_join_comm:
  for a: SamplerFamily
  for b: SamplerFamily
  {sampler_family_join(a, b) == sampler_family_join(b, a) : SamplerFamily}

def sampler_join_comm(a, b):
  match a b:
    case SampGreedy{} SampGreedy{}: {==}
    case SampGreedy{} SampNoSearch{}: {==}
    case SampGreedy{} SampSearch{}: {==}
    case SampNoSearch{} SampGreedy{}: {==}
    case SampNoSearch{} SampNoSearch{}: {==}
    case SampNoSearch{} SampSearch{}: {==}
    case SampSearch{} SampGreedy{}: {==}
    case SampSearch{} SampNoSearch{}: {==}
    case SampSearch{} SampSearch{}: {==}

law sampler_join_idem:
  for a: SamplerFamily
  {sampler_family_join(a, a) == a : SamplerFamily}

def sampler_join_idem(a):
  match a:
    case SampGreedy{}: {==}
    case SampNoSearch{}: {==}
    case SampSearch{}: {==}

law sampler_join_assoc:
  for a: SamplerFamily
  for b: SamplerFamily
  for c: SamplerFamily
  {sampler_family_join(sampler_family_join(a, b), c) == sampler_family_join(a, sampler_family_join(b, c)) : SamplerFamily}

def sampler_join_assoc(a, b, c):
  match a:
    case SampSearch{}:
      {==}
    case SampGreedy{}:
      {==}
    case SampNoSearch{}:
      match b:
        case SampSearch{}:
          {==}
        case SampNoSearch{}:
          match c:
            case SampGreedy{}: {==}
            case SampNoSearch{}: {==}
            case SampSearch{}: {==}
        case SampGreedy{}:
          match c:
            case SampGreedy{}: {==}
            case SampNoSearch{}: {==}
            case SampSearch{}: {==}

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
