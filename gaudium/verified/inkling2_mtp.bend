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
import lemmas.bend as G
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
# 0 when no ring holds them; refused on overflow (the window is read from the drafter's file, so a hostile file may give one
# near the top of u64, and the sum is never built past it).
# rusty-bends: total
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

# A result no smaller than `n` passes a check that `n` fits in it (erased before Rust codegen).
def mtp_and_not_le(c: Bool, +r: Nat, +n: Nat, h: {Bool.and(c, Nat.is_lt(r, n)) == False{} : Bool}) -> {Bool.and(c, Bool.not(Nat.is_le(n, r))) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      K.ikp_not_t(Nat.is_le(n, r), L.nat_not_lt_imp_le(r, n, h))

# promise: a ring the sizing returns (non-zero) always passes the ring check, so the engine never builds a ring in which a row written ahead shares a slot with a position the window reads.
# harm: a row written ahead would overwrite a context position the block still reads.
# catches: the ring sized for the window alone (the rows a pass writes ahead left out)
law mtp_ring_slots_pass:
  for +w: Nat
  for +a: Nat
  {Bool.and(Nat.is_ne(mtp_ring_slots(w, a), 0n), Bool.not(mtp_ring_ok(mtp_ring_slots(w, a), w, a))) == False{} : Bool}

def mtp_ring_slots_pass(w, a):
  mtp_and_not_le(Nat.is_ne(mtp_ring_slots(w, a), 0n), mtp_ring_slots(w, a), Nat.add(w, a), K.ikp_pow2_holds(40n, Nat.is_ge(128n, Nat.add(w, a)), 128n, Nat.add(w, a), K.ikp_ge_lt_excl(128n, Nat.add(w, a))))

# promise: a ring the sizing returns (non-zero) has a power-of-two number of slots, so a position's slot is the position modulo the ring.
# harm: the context kernel's slot mask would send two live positions to one slot.
# catches: the ring sized to the need itself, not rounded up to a power of two
law mtp_ring_slots_pow2:
  for +w: Nat
  for +a: Nat
  {Bool.and(Nat.is_ne(mtp_ring_slots(w, a), 0n), Bool.not(K.ik_is_pow2(mtp_ring_slots(w, a)))) == False{} : Bool}

def mtp_ring_slots_pow2(w, a):
  K.ikp_pow2_rows_pow2(Nat.add(w, a))

# Lemmas of the doubling search behind `K.ik_pow2_rows` (all erased before Rust codegen).

# Two answers of one Bool cannot both hold.
def mtp_clash(-x: Bool, hf: {x == False{} : Bool}, ht: {x == True{} : Bool}) -> {False{} == True{} : Bool}:
  %hf : {_ == True{} : Bool}
  ht

def mtp_not_self(x: Bool) -> {Bool.and(Bool.not(x), x) == False{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      {==}

# A search that has not stopped at `p` has a need above `p`.
def mtp_ge_le_excl(+p: Nat, +need: Nat) -> {Bool.and(Bool.not(Nat.is_ge(p, need)), Nat.is_le(need, p)) == False{} : Bool}:
  %Equal.sym(Bool, Nat.is_ge(p, need), Nat.is_le(need, p), K.ikp_ge_le(p, need)) : {Bool.and(Bool.not(_), Nat.is_le(need, p)) == False{} : Bool}
  mtp_not_self(Nat.is_le(need, p))

# Doubling a non-zero size keeps it non-zero.
def mtp_double_nz(+p: Nat, hp: {Nat.is_ne(p, 0n) == True{} : Bool}) -> {Nat.is_ne(Nat.mul(p, 2n), 0n) == True{} : Bool}:
  match p:
    case 0n:
      L.bool_absurd({Nat.is_ne(Nat.mul(0n, 2n), 0n) == True{} : Bool}, hp)
    case 1n+q:
      {==}

# A need of at most `cap` stays within `p` times 2^fuel when `cap` is that bound written another way.
def mtp_cap_bridge(-need: Nat, -p: Nat, -x: Nat, -y: Nat, e: {x == y : Nat}, h: {Nat.is_le(need, Nat.mul(p, x)) == True{} : Bool}) -> {Nat.is_le(need, Nat.mul(p, y)) == True{} : Bool}:
  %e : {Nat.is_le(need, Nat.mul(p, _)) == True{} : Bool}
  h

# The search from a non-zero `p` finds a ring for every need of at most p * 2^fuel.
def mtp_pow2_found(fuel: Nat, done: Bool, +p: Nat, +need: Nat, hp: {Nat.is_ne(p, 0n) == True{} : Bool}, hd: {Bool.and(Bool.not(done), Nat.is_le(need, p)) == False{} : Bool}, h: {Nat.is_le(need, Nat.mul(p, Nat.pow(2n, fuel))) == True{} : Bool}) -> {Nat.is_eq(K.ik_pow2_from(fuel, done, p, need), 0n) == False{} : Bool}:
  match fuel:
    case 0n:
      match done:
        case True{}:
          K.ikp_not_tf(Nat.is_eq(p, 0n), hp)
        case False{}:
          L.bool_absurd({Nat.is_eq(K.ik_pow2_from(0n, False{}, p, need), 0n) == False{} : Bool}, mtp_clash(Nat.is_le(need, p), hd, L.nat_le_subst_r(need, Nat.mul(p, 1n), p, L.nat_mul_one_r(p), h)))
    case 1n+f:
      match done:
        case True{}:
          K.ikp_not_tf(Nat.is_eq(p, 0n), hp)
        case False{}:
          +g = {f : Nat}
          mtp_pow2_found(g, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need, mtp_double_nz(p, hp), mtp_ge_le_excl(Nat.mul(p, 2n), need), L.nat_le_subst_r(need, Nat.mul(p, Nat.mul(2n, Nat.pow(2n, g))), Nat.mul(Nat.mul(p, 2n), Nat.pow(2n, g)), Equal.sym(Nat, Nat.mul(Nat.mul(p, 2n), Nat.pow(2n, g)), Nat.mul(p, Nat.mul(2n, Nat.pow(2n, g))), L.nat_mul_assoc(p, 2n, Nat.pow(2n, g))), h))

# A ring the search finds with some fuel it still finds with more.
def mtp_pow2_more_le(f: Nat, f2: Nat, done: Bool, +p: Nat, +need: Nat, hf: {Nat.is_le(f, f2) == True{} : Bool}, h: {Nat.is_eq(K.ik_pow2_from(f, done, p, need), 0n) == False{} : Bool}) -> {Nat.is_eq(K.ik_pow2_from(f2, done, p, need), 0n) == False{} : Bool}:
  match f:
    case 0n:
      match f2:
        case 0n:
          h
        case 1n+g2:
          match done:
            case True{}:
              h
            case False{}:
              L.bool_absurd_tf({Nat.is_eq(K.ik_pow2_from(1n+g2, False{}, p, need), 0n) == False{} : Bool}, h)
    case 1n+g:
      match f2:
        case 0n:
          L.bool_absurd({Nat.is_eq(K.ik_pow2_from(0n, done, p, need), 0n) == False{} : Bool}, hf)
        case 1n+g2:
          match done:
            case True{}:
              h
            case False{}:
              mtp_pow2_more_le(g, g2, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need, hf, h)

# A window and a pass within the cap get a ring: the search from 128 holds 128 * 2^9 rows within 9 of its 40 doublings.
def mtp_cap_ring(c: Bool, +w: Nat, +a: Nat, e: {Nat.is_le(Nat.add(w, a), Nat.mul(128n, 512n)) == c : Bool}) -> {Bool.and(c, Nat.is_eq(mtp_ring_slots(w, a), 0n)) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      +n = {Nat.add(w, a) : Nat}
      mtp_pow2_more_le(9n, 40n, Nat.is_ge(128n, n), 128n, n, {==}, mtp_pow2_found(9n, Nat.is_ge(128n, n), 128n, n, {==}, mtp_ge_le_excl(128n, n), mtp_cap_bridge(n, 128n, 512n, Nat.pow(2n, 9n), {==}, e)))

# promise: a window and a pass that together need at most 65536 slots (128 x 512, 64 times the 1024-row windows of today's drafters) always get a ring, so the two laws above never hold only because the sizing gave none.
# harm: the drafter would refuse a checkpoint whose window and passes fit a ring the engine can build.
# catches: the search cut to too few doublings (no ring past 128 slots, the 40 doublings of `K.ik_pow2_rows` cut to 0)
law mtp_ring_slots_found:
  for +w: Nat
  for +a: Nat
  {Bool.and(Nat.is_le(Nat.add(w, a), Nat.mul(128n, 512n)), Nat.is_eq(mtp_ring_slots(w, a), 0n)) == False{} : Bool}

def mtp_ring_slots_found(w, a):
  mtp_cap_ring(Nat.is_le(Nat.add(w, a), Nat.mul(128n, 512n)), w, a, {==})

# A need above a size stays above it when both are doubled.
def mtp_le_double(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == False{} : Bool}) -> {Nat.is_le(Nat.mul(a, 2n), Nat.mul(b, 2n)) == False{} : Bool}:
  match a b:
    case 0n 0n:
      h
    case 0n 1n+bp:
      h
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      mtp_le_double(ap, bp, h)

# The doubling search answers at most 128, or less than twice `need`, when the size it starts from does and it stops as soon
# as a size holds `need` (erased before Rust codegen, as are the three below).
def mtp_pow2_hi(fuel: Nat, done: Bool, +p: Nat, +need: Nat, hp: {Bool.and(Nat.is_lt(128n, p), Nat.is_le(Nat.mul(need, 2n), p)) == False{} : Bool}, hd: {Bool.and(Bool.not(done), Nat.is_le(need, p)) == False{} : Bool}) -> {Bool.and(Nat.is_lt(128n, K.ik_pow2_from(fuel, done, p, need)), Nat.is_le(Nat.mul(need, 2n), K.ik_pow2_from(fuel, done, p, need))) == False{} : Bool}:
  match fuel:
    case 0n:
      match done:
        case True{}:
          hp
        case False{}:
          {==}
    case 1n+f:
      match done:
        case True{}:
          hp
        case False{}:
          +g = {f : Nat}
          mtp_pow2_hi(g, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need, K.ikp_and_fr(Nat.is_lt(128n, Nat.mul(p, 2n)), Nat.is_le(Nat.mul(need, 2n), Nat.mul(p, 2n)), mtp_le_double(need, p, hd)), mtp_ge_le_excl(Nat.mul(p, 2n), need))

# A ring below twice `n` is below twice any `h` that holds `n`.
def mtp_bound_go(c: Bool, +n: Nat, +h: Nat, +r: Nat, e: {Nat.is_le(Nat.mul(h, 2n), r) == c : Bool}, hh: {Nat.is_le(n, h) == True{} : Bool}, hb: {Nat.is_le(Nat.mul(n, 2n), r) == False{} : Bool}) -> {c == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      L.bool_absurd({True{} == False{} : Bool}, mtp_clash(Nat.is_le(Nat.mul(n, 2n), r), hb, L.nat_le_trans(Nat.mul(n, 2n), Nat.mul(h, 2n), r, L.nat_mul_le_mono_r(n, h, 2n, hh), e)))

# A ring below twice `n` is below twice `h` whenever `h` holds `n`, read against the check's answer.
def mtp_bound_ok(b: Bool, +n: Nat, +h: Nat, +r: Nat, e: {Nat.is_le(n, h) == b : Bool}, hb: {Nat.is_le(Nat.mul(n, 2n), r) == False{} : Bool}) -> {Bool.and(b, Nat.is_le(Nat.mul(h, 2n), r)) == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      mtp_bound_go(Nat.is_le(Nat.mul(h, 2n), r), n, h, r, {==}, e, hb)

# A bound below twice `n`, read against every `h` that holds `n`.
def mtp_bound_all(x: Bool, +n: Nat, +h: Nat, +r: Nat, hb: {Bool.and(x, Nat.is_le(Nat.mul(n, 2n), r)) == False{} : Bool}) -> {Bool.and(x, Bool.and(Nat.is_le(n, h), Nat.is_le(Nat.mul(h, 2n), r))) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      mtp_bound_ok(Nat.is_le(n, h), n, h, r, {==}, hb)

# promise: a ring the sizing returns with more than 128 slots always has fewer than twice the slots of any ring the check accepts for the same window and rows written ahead.
# harm: the drafter's persistent K / V rings could take twice the HBM the window and the rows written ahead need, or more, and a load whose rings would fit could be refused.
# catches: the ring doubled after rounding (twice the power of two that holds the window and the pass)
law mtp_ring_slots_bounded:
  for +w: Nat
  for +a: Nat
  for +h: Nat
  {Bool.and(Nat.is_lt(128n, mtp_ring_slots(w, a)), Bool.and(mtp_ring_ok(h, w, a), Nat.is_le(Nat.mul(h, 2n), mtp_ring_slots(w, a)))) == False{} : Bool}

def mtp_ring_slots_bounded(w, a, h):
  +n = {Nat.add(w, a) : Nat}
  mtp_bound_all(Nat.is_lt(128n, mtp_ring_slots(w, a)), n, h, mtp_ring_slots(w, a), mtp_pow2_hi(40n, Nat.is_ge(128n, n), 128n, n, {==}, mtp_ge_le_excl(128n, n)))

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

# promise: chunk count multiplied by chunk size always covers every position before anchor p
# catches: mtp_ctx_chunks omits the Nat.sub(chunk, 1n) ceiling term
law mtp_ctx_chunks_covers_prompt:
  for +p: Nat
  for +cm1: Nat
  {Nat.is_le(p, Nat.mul(mtp_ctx_chunks(p, 1n+cm1), 1n+cm1)) == True{} : Bool}

def mtp_ctx_chunks_covers_prompt(p, cm1):
  %Equal.sym(Nat, Nat.sub(cm1, 0n), cm1, L.nat_sub_zero_r(cm1)) : {Nat.is_le(p, Nat.mul(Nat.max(1n, Nat.div(Nat.add(p, _), 1n+cm1)), 1n+cm1)) == True{} : Bool}
  +q = Nat.div(Nat.add(p, cm1), 1n+cm1)
  L.nat_le_trans(p, Nat.mul(q, 1n+cm1), Nat.mul(Nat.max(1n, q), 1n+cm1), G.g_nat_ceil_div_covers(p, cm1), L.nat_mul_le_mono_r(q, Nat.max(1n, q), 1n+cm1, L.nat_max_ge_r(1n, q)))

