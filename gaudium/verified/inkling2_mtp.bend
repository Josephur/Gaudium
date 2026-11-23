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
