# Inkling 2 block drafter (the checkpoint's mtp.safetensors): the decisions the engine's drafter makes on the host.
# Specifies:
#   - which context positions a block anchored at position p reads (a windowed layer the last `window` before the anchor,
#     the global layer every one, never the anchor's own position or a later one)
#   - the slots of a windowed layer's context ring and when a ring is safe beside the rows a context pass writes ahead
#   - how many drafts one block yields for a verify pass
#   - which row of the last context pass holds the recurrent head's context
#   - how many chunks of the global layer's cache a block reads

import Base
import nat_lemmas.bend as L
import inkling2.bend as K

# ============================================================================
# 1. Context a block reads
# ============================================================================

# Context positions a block anchored at position `p` reads, counting back from p - 1: a windowed layer the last `window` of
# them, the global layer every one. Positions p - seen .. p - 1.
def mtp_ctx_seen(+p: Nat, window: Nat, windowed: Bool) -> Nat:
  match windowed:
    case True{}:
      Nat.min(p, window)
    case False{}:
      p

# promise: a block never reads more context positions than lie before its anchor, so it never reads the anchor's own position or a later one.
# harm: the drafter would attend rows of rejected drafts or stale rows of an earlier request.
law mtp_ctx_seen_before_anchor:
  for +p: Nat
  for +w: Nat
  for +b: Bool
  {Nat.is_le(mtp_ctx_seen(p, w, b), p) == True{} : Bool}

def mtp_ctx_seen_before_anchor(p, w, b):
  match b:
    case True{}:
      L.nat_min_le_l(p, w)
    case False{}:
      L.nat_le_refl(p)

# promise: a windowed layer never reads more context positions than its window.
law mtp_ctx_seen_in_window:
  for +p: Nat
  for +w: Nat
  {Nat.is_le(mtp_ctx_seen(p, w, True{}), w) == True{} : Bool}

def mtp_ctx_seen_in_window(p, w):
  L.nat_min_le_r(p, w)

# Chunks of the global layer's cache a block anchored at `p` reads: the chunks that hold positions 0 .. p - 1, at least one.
def mtp_ctx_chunks(+p: Nat, +chunk: Nat) -> Nat:
  Nat.max(1n, Nat.div(Nat.add(p, Nat.sub(chunk, 1n)), chunk))

# promise: a block always reads at least one chunk of the global layer's cache.
law mtp_ctx_chunks_some:
  for +p: Nat
  for +c: Nat
  {Nat.is_le(1n, mtp_ctx_chunks(p, c)) == True{} : Bool}

def mtp_ctx_chunks_some(p, c):
  L.nat_max_ge_l(1n, Nat.div(Nat.add(p, Nat.sub(c, 1n)), c))

# ============================================================================
# 2. Context rings
# ============================================================================

# Slots of a windowed layer's context ring: a power of two (of at least 128) that holds the window beside the rows a context
# pass writes ahead of an anchor (a prefill tile or a verify pass writes up to `ahead` positions at or beyond the next anchor).
def mtp_ring_slots(window: Nat, ahead: Nat) -> Nat:
  K.ik_pow2_rows(Nat.add(window, ahead))

# A ring of `ring` slots is safe: a position the window reads and a row written ahead never share a slot.
def mtp_ring_ok(+ring: Nat, +window: Nat, +ahead: Nat) -> Bool:
  Nat.is_le(Nat.add(window, ahead), ring)

def mtp_le_lt_false(+n: Nat, +r: Nat) -> {Bool.and(Nat.is_le(n, r), Nat.is_lt(r, n)) == False{} : Bool}:
  match n r:
    case 0n 0n:
      {==}
    case 0n 1n+rp:
      {==}
    case 1n+np 0n:
      {==}
    case 1n+np 1n+rp:
      mtp_le_lt_false(np, rp)

# promise: a ring the check accepts always holds the window beside every row a context pass writes ahead of the anchor.
# harm: a row written ahead would overwrite a context position the block still reads.
law mtp_ring_ok_holds:
  for +r: Nat
  for +w: Nat
  for +a: Nat
  {Bool.and(mtp_ring_ok(r, w, a), Nat.is_lt(r, Nat.add(w, a))) == False{} : Bool}

def mtp_ring_ok_holds(r, w, a):
  mtp_le_lt_false(Nat.add(w, a), r)

# ============================================================================
# 3. Drafts of a block
# ============================================================================

# Drafts one block yields for a verify pass of `k` drafts: the block's mask rows, at most `k`.
def mtp_draft_count(+block: Nat, +k: Nat) -> Nat:
  Nat.min(k, Nat.sub(block, 1n))

# promise: a block never yields more drafts than it has mask rows.
law mtp_draft_count_in_block:
  for +b: Nat
  for +k: Nat
  {Nat.is_le(mtp_draft_count(b, k), Nat.sub(b, 1n)) == True{} : Bool}

def mtp_draft_count_in_block(b, k):
  L.nat_min_le_r(k, Nat.sub(b, 1n))

# promise: a block never yields more drafts than the verify pass has rows for.
law mtp_draft_count_in_pass:
  for +b: Nat
  for +k: Nat
  {Nat.is_le(mtp_draft_count(b, k), k) == True{} : Bool}

def mtp_draft_count_in_pass(b, k):
  L.nat_min_le_l(k, Nat.sub(b, 1n))

# ============================================================================
# 4. The recurrent head's context row
# ============================================================================

# Row of the last context pass (of `rows` rows) that holds the recurrent head's context of the next block: the row of the last
# committed position before the next anchor, `row`, kept inside the pass.
def mtp_frnn_row(row: Nat, +rows: Nat) -> Nat:
  Nat.min(row, Nat.sub(rows, 1n))

# promise: the recurrent head's context row is always a row the last context pass wrote.
law mtp_frnn_row_in_pass:
  for +r: Nat
  for +n: Nat
  {Nat.is_lt(mtp_frnn_row(r, 1n+n), 1n+n) == True{} : Bool}

def mtp_frnn_row_in_pass(r, n):
  %Equal.sym(Nat, Nat.sub(n, 0n), n, L.nat_sub_zero_r(n)) : {Nat.is_lt(Nat.min(r, _), 1n+n) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(Nat.min(r, n), 1n+n), Nat.is_le(Nat.min(r, n), n), L.nat_lt_succ_eq_le(Nat.min(r, n), n)) : {_ == True{} : Bool}
  L.nat_min_le_r(r, n)

def im_not_le_imp_lt(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == False{} : Bool}) -> {Nat.is_lt(b, a) == True{} : Bool}:
  match a b:
    case 0n 0n:
      L.bool_absurd_tf({Nat.is_lt(0n, 0n) == True{} : Bool}, h)
    case 0n 1n+bp:
      L.bool_absurd_tf({Nat.is_lt(1n+bp, 0n) == True{} : Bool}, h)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      im_not_le_imp_lt(ap, bp, h)

def im_succ_le_inner(+b: Nat, +a: Nat, h: {Nat.is_lt(b, a) == True{} : Bool}) -> {Nat.is_le(Nat.add(b, 1n), a) == True{} : Bool}:
  match b a:
    case 0n 0n:
      L.bool_absurd({Nat.is_le(1n, 0n) == True{} : Bool}, h)
    case 0n 1n+ap:
      L.nat_le_zero(ap)
    case 1n+bp 0n:
      L.bool_absurd({Nat.is_le(Nat.add(1n+bp, 1n), 0n) == True{} : Bool}, h)
    case 1n+bp 1n+ap:
      im_succ_le_inner(bp, ap, h)

def im_ceil_le_r(
  +p: Nat,
  +cm1: Nat,
  +q: Nat,
  eqq: {q == Nat.div(Nat.add(p, cm1), 1n+cm1) : Nat},
  hadd1: {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+cm1), 1n), cm1), Nat.add(p, cm1)) == True{} : Bool}
) -> {Nat.is_le(Nat.add(Nat.mul(q, 1n+cm1), 1n+cm1), Nat.add(Nat.mul(q, 1n+cm1), Nat.mod(Nat.add(p, cm1), 1n+cm1))) == True{} : Bool}:
  %L.nat_add_assoc(Nat.mul(q, 1n+cm1), 1n, cm1) : {Nat.is_le(_, Nat.add(Nat.mul(q, 1n+cm1), Nat.mod(Nat.add(p, cm1), 1n+cm1))) == True{} : Bool}
  %Equal.sym(Nat, q, Nat.div(Nat.add(p, cm1), 1n+cm1), eqq) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+cm1), 1n), cm1), Nat.add(Nat.mul(_, 1n+cm1), Nat.mod(Nat.add(p, cm1), 1n+cm1))) == True{} : Bool}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.add(p, cm1), 1n+cm1), 1n+cm1), Nat.mod(Nat.add(p, cm1), 1n+cm1)), Nat.add(p, cm1), L.nat_div_mod_r(Nat.add(p, cm1), cm1)) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+cm1), 1n), cm1), _) == True{} : Bool}
  hadd1

def im_ceil_le_aux(
  +p: Nat,
  +cm1: Nat,
  +q: Nat,
  b: Bool,
  eb: {b == Nat.is_le(p, Nat.mul(q, 1n+cm1)) : Bool},
  eqq: {q == Nat.div(Nat.add(p, cm1), 1n+cm1) : Nat}
) -> {Nat.is_le(p, Nat.mul(q, 1n+cm1)) == True{} : Bool}:
  match b:
    case True{}:
      Equal.sym(Bool, True{}, Nat.is_le(p, Nat.mul(q, 1n+cm1)), eb)
    case False{}:
      +x = Nat.mul(q, 1n+cm1)
      +hlt = im_not_le_imp_lt(p, x, Equal.sym(Bool, False{}, Nat.is_le(p, x), eb))
      +hsle = im_succ_le_inner(x, p, hlt)
      +hadd1 = L.nat_add_le_mono(Nat.add(x, 1n), p, cm1, cm1, hsle, L.nat_le_refl(cm1))
      +hle_bad = im_ceil_le_r(p, cm1, q, eqq, hadd1)
      +hlt_bad = L.nat_add_lt_l(x, Nat.mod(Nat.add(p, cm1), 1n+cm1), 1n+cm1, L.nat_mod_lt(Nat.add(p, cm1), cm1))
      Empty.absurd({Nat.is_le(p, x) == True{} : Bool}, L.nat_lt_not_le(Nat.add(x, Nat.mod(Nat.add(p, cm1), 1n+cm1)), Nat.add(x, 1n+cm1), hlt_bad, hle_bad))

# promise: chunk count multiplied by chunk size always covers every position before anchor p
# catches: mtp_ctx_chunks omits the Nat.sub(chunk, 1n) ceiling term
law mtp_ctx_chunks_covers_prompt:
  for +p: Nat
  for +cm1: Nat
  {Nat.is_le(p, Nat.mul(mtp_ctx_chunks(p, 1n+cm1), 1n+cm1)) == True{} : Bool}

def mtp_ctx_chunks_covers_prompt(p, cm1):
  %Equal.sym(Nat, Nat.sub(cm1, 0n), cm1, L.nat_sub_zero_r(cm1)) : {Nat.is_le(p, Nat.mul(Nat.max(1n, Nat.div(Nat.add(p, _), 1n+cm1)), 1n+cm1)) == True{} : Bool}
  +q = Nat.div(Nat.add(p, cm1), 1n+cm1)
  L.nat_le_trans(p, Nat.mul(q, 1n+cm1), Nat.mul(Nat.max(1n, q), 1n+cm1), im_ceil_le_aux(p, cm1, q, Nat.is_le(p, Nat.mul(q, 1n+cm1)), {==}, {==}), L.nat_mul_le_mono_r(q, Nat.max(1n, q), 1n+cm1, L.nat_max_ge_r(1n, q)))

