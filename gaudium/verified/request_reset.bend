# Per-request device state reset (`engine/request_reset.rs`):
#   - Resets DeltaNet start state (`slot 0` of `ring 0` normally; full allocated rings on `Full`)
#     using separate `alloc_slots` for recurrent state (`s`) and conv state (`cs`) so a 1-slot
#     recurrent replay ring is never zeroed past its allocation.
#   - Computes overflow-free dirty byte / row spans for ChunkMajor and HeadMajor caches.
#   - Tracks per-request written high-water rows and transitions `ResetState` across request begin
#     and successful completion (so a failed request automatically zeroes written rows next turn).

import Base
import nat_lemmas.bend as L

# ============================================================================
# 1. Helper Nat / Bool lemmas (erased before Rust codegen)
# ============================================================================

def rr_le_add_sub_self(+floor: Nat, +r: Nat, h_le: {Nat.is_le(floor, r) == True{} : Bool}) -> {Nat.add(floor, Nat.sub(r, floor)) == r : Nat}:
  match floor r:
    case 0n 0n:
      {==}
    case 0n 1n+rp:
      {==}
    case 1n+fp 0n:
      L.bool_absurd({Nat.add(1n+fp, Nat.sub(0n, 1n+fp)) == 0n : Nat}, h_le)
    case 1n+fp 1n+rp:
      %rr_le_add_sub_self(fp, rp, h_le) : {1n+Nat.add(fp, Nat.sub(rp, fp)) == 1n+_ : Nat}
      {==}

def rr_ge_one_max1(x: Nat) -> {Nat.is_le(1n, Nat.max(1n, x)) == True{} : Bool}:
  L.nat_max_ge_l(1n, x)

def rr_min_ge_one(+a: Nat, +b: Nat, ha: {Nat.is_le(1n, a) == True{} : Bool}, hb: {Nat.is_le(1n, b) == True{} : Bool}) -> {Nat.is_le(1n, Nat.min(a, b)) == True{} : Bool}:
  match a b:
    case 0n _:
      L.bool_absurd({Nat.is_le(1n, Nat.min(0n, b)) == True{} : Bool}, ha)
    case 1n+ap 0n:
      L.bool_absurd({Nat.is_le(1n, Nat.min(1n+ap, 0n)) == True{} : Bool}, hb)
    case 1n+ap 1n+bp:
      L.nat_le_zero(Nat.min(ap, bp))

# ============================================================================
# 2. Reset mode, scope, and DeltaNet ring zero sizes
# ============================================================================

type RrMode is Data:
  RrModeAuto{}
  RrModeDirty{}
  RrModeFull{}

type RrScope is Data:
  RrScopeState{}
  RrScopeDirty{}
  RrScopeFull{}

def rr_scope(+mode: RrMode, +unclean: Bool) -> RrScope:
  match mode:
    case RrModeFull{}:
      RrScopeFull{}
    case RrModeDirty{}:
      RrScopeDirty{}
    case RrModeAuto{}:
      match unclean:
        case True{}:
          RrScopeDirty{}
        case False{}:
          RrScopeState{}

def rr_scope_is_state(+scope: RrScope) -> Bool:
  match scope:
    case RrScopeState{}:
      True{}
    case _:
      False{}

def rr_is_state_only(+mode: RrMode, +unclean: Bool) -> Bool:
  rr_scope_is_state(rr_scope(mode, unclean))

def rr_kept_hw(+scope: RrScope, dirty_rows: Nat) -> Nat:
  match scope:
    case RrScopeState{}:
      dirty_rows
    case _:
      0n

def rr_zero_both_rings(+scope: RrScope) -> Bool:
  match scope:
    case RrScopeFull{}:
      True{}
    case _:
      False{}

# Bytes of one DeltaNet tensor ring (`s[p]` or `cs[p]`) to zero for a request.
# `alloc_slots` is that tensor's own ring slot count (`gdn_state_slots()` for `s`,
# `gdn_slots()` for `cs`). In State/Dirty modes only slot 0 (`one_bytes`) is zeroed.
def rr_gdn_zero_bytes(+scope: RrScope, one_bytes: Nat, alloc_slots: Nat) -> Nat:
  match scope:
    case RrScopeFull{}:
      Nat.mul(one_bytes, Nat.max(1n, alloc_slots))
    case RrScopeDirty{}:
      one_bytes
    case RrScopeState{}:
      one_bytes

# ============================================================================
# 3. Overflow-free dirty range sizing for ChunkMajor & HeadMajor KV caches
# ============================================================================

def rr_div_ceil_pick(q: Nat, +rem_gt_zero: Bool) -> Nat:
  match rem_gt_zero:
    case True{}:
      Nat.add(q, 1n)
    case False{}:
      q

def rr_div_ceil(+rows: Nat, +block: Nat) -> Nat:
  match rows:
    case 0n:
      0n
    case 1n+rp:
      match block:
        case 0n:
          1n+rp
        case 1n+bm1:
          rr_div_ceil_pick(Nat.div(1n+rp, 1n+bm1), Nat.is_gt(Nat.mod(1n+rp, 1n+bm1), 0n))

def rr_chunk_dirty_nonzero(+cache_bytes: Nat, +c: Nat, +dirty_rows: Nat, +chunk_rows: Nat) -> Nat:
  match cache_bytes:
    case 0n:
      0n
    case 1n+cp:
      Nat.mul(Nat.div(1n+cp, c), Nat.min(rr_div_ceil(dirty_rows, chunk_rows), c))

# Prefix bytes to zero in a ChunkMajor KV cache (`0n` when `dirty_rows == 0n` or `cache_bytes == 0n`).
def rr_chunk_dirty_bytes(+cache_bytes: Nat, +chunks: Nat, +dirty_rows: Nat, +chunk_rows: Nat) -> Nat:
  match dirty_rows:
    case 0n:
      0n
    case 1n+rp:
      rr_chunk_dirty_nonzero(cache_bytes, Nat.max(1n, chunks), 1n+rp, chunk_rows)

def rr_head_dirty_pick(r: Nat, cand: Nat, +fits: Bool) -> Nat:
  match fits:
    case True{}:
      cand
    case False{}:
      r

def rr_head_dirty_align_nz(+r: Nat, dirty: Nat, +block: Nat, +rem_zero: Bool, +floor: Nat) -> Nat:
  match rem_zero:
    case True{}:
      dirty
    case False{}:
      rr_head_dirty_pick(r, Nat.add(floor, block), Nat.is_le(block, Nat.sub(r, floor)))

def rr_head_dirty_block(+r: Nat, +dirty: Nat, +block: Nat) -> Nat:
  match block:
    case 0n:
      dirty
    case 1n+bm1:
      rr_head_dirty_align_nz(
        r,
        dirty,
        1n+bm1,
        Nat.is_eq(Nat.mod(dirty, 1n+bm1), 0n),
        Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1)
      )

def rr_head_dirty_lt(+r: Nat, +dirty: Nat, +block: Nat, +within: Bool) -> Nat:
  match within:
    case True{}:
      rr_head_dirty_block(r, dirty, block)
    case False{}:
      r

# Rows per head to zero in a HeadMajor cache, rounded up to `row_block` and clamped at `max(1, rows)`.
def rr_head_dirty_rows(+rows: Nat, +dirty_rows: Nat, +row_block: Nat) -> Nat:
  match dirty_rows:
    case 0n:
      0n
    case 1n+rp:
      rr_head_dirty_lt(Nat.max(1n, rows), 1n+rp, row_block, Nat.is_lt(1n+rp, Nat.max(1n, rows)))

def rr_head_full_span(+rows: Nat, +n: Nat) -> Bool:
  Nat.is_eq(n, Nat.max(1n, rows))

def rr_head_span_bytes(+cache_bytes: Nat, +heads: Nat, +rows: Nat, +n: Nat) -> Nat:
  Nat.mul(
    Nat.min(n, Nat.max(1n, rows)),
    Nat.div(Nat.div(cache_bytes, Nat.max(1n, heads)), Nat.max(1n, rows))
  )

# ============================================================================
# 4. Overflow-free high-water mark & ResetState lifecycle
# ============================================================================

def rr_hw_step_pick(+cap: Nat, +base: Nat, +add: Nat, +fits: Bool) -> Nat:
  match fits:
    case True{}:
      Nat.add(base, add)
    case False{}:
      cap

def rr_hw_add_capped(+cap: Nat, +base: Nat, +add: Nat) -> Nat:
  rr_hw_step_pick(cap, base, add, Nat.is_le(add, Nat.sub(cap, base)))

def rr_hw_from_prompt(+max_ctx: Nat, +prompt: Nat, +new_tokens: Nat, +slack: Nat, +p_fits: Bool) -> Nat:
  match p_fits:
    case True{}:
      rr_hw_add_capped(max_ctx, rr_hw_add_capped(max_ctx, prompt, new_tokens), slack)
    case False{}:
      max_ctx

# Row high-water mark (`min(prompt + new_tokens + slack, max_ctx)` without overflow).
def rr_high_water(+prompt: Nat, +new_tokens: Nat, +slack: Nat, +max_ctx: Nat) -> Nat:
  rr_hw_from_prompt(max_ctx, prompt, new_tokens, slack, Nat.is_le(prompt, max_ctx))

type RrState is Data:
  RrState{mode: RrMode, kv_hw: Nat, held_hw: Nat, unclean: Bool}

def rr_state_hw(st: RrState) -> Nat:
  match st:
    case RrState{_, hw, _, _}:
      hw

def rr_state_held(st: RrState) -> Nat:
  match st:
    case RrState{_, _, held, _}:
      held

def rr_state_unclean(st: RrState) -> Bool:
  match st:
    case RrState{_, _, _, u}:
      u

def rr_state_mode(st: RrState) -> RrMode:
  match st:
    case RrState{m, _, _, _}:
      m

def rr_begin_kept(+m: RrMode, +kept: Nat, prompt: Nat, max_new: Nat, slack: Nat, max_ctx: Nat) -> RrState:
  RrState{m, Nat.max(kept, rr_high_water(prompt, max_new, slack, max_ctx)), kept, True{}}

def rr_begin_fields(+m: RrMode, hw: Nat, u: Bool, prompt: Nat, max_new: Nat, slack: Nat, max_ctx: Nat) -> RrState:
  rr_begin_kept(m, rr_kept_hw(rr_scope(m, u), hw), prompt, max_new, slack, max_ctx)

def rr_begin(st: RrState, prompt: Nat, max_new: Nat, slack: Nat, max_ctx: Nat) -> RrState:
  match st:
    case RrState{m, hw, _, u}:
      rr_begin_fields(m, hw, u, prompt, max_new, slack, max_ctx)

def rr_end_ok_fields(m: RrMode, +held: Nat, prompt: Nat, generated: Nat, slack: Nat, max_ctx: Nat) -> RrState:
  RrState{m, Nat.max(held, rr_high_water(prompt, generated, slack, max_ctx)), held, False{}}

def rr_end_ok(st: RrState, prompt: Nat, generated: Nat, slack: Nat, max_ctx: Nat) -> RrState:
  match st:
    case RrState{m, _, held, _}:
      rr_end_ok_fields(m, held, prompt, generated, slack, max_ctx)

# ============================================================================
# 5. Proofs & Laws
# ============================================================================

# LAW RR.1: Full reset mode always selects Full scope regardless of unclean history.
# promise: full reset mode always produces full scope
# catches: unclean checked before Full mode so Full falls back to Dirty
law rr_full_always_full:
  for +u: Bool
  {rr_scope(RrModeFull{}, u) == RrScopeFull{} : RrScope}

def rr_full_always_full(u):
  {==}

# LAW RR.2: An Auto request after a failed request escalates to Dirty scope.
# promise: an unclean Auto state always zeroes dirty KV rows on the next reset
# catches: Auto mode ignores unclean and stays in StateOnly
law rr_auto_unclean_is_dirty:
  {rr_scope(RrModeAuto{}, True{}) == RrScopeDirty{} : RrScope}

def rr_auto_unclean_is_dirty():
  {==}

# LAW RR.3: Clean Auto mode uses StateOnly and never zeroes both rings.
# promise: clean Auto mode selects StateOnly and never zeroes ring 1
# catches: rr_zero_both_rings returns True for StateOnly
law rr_auto_clean_is_state_only:
  {Bool.and(rr_is_state_only(RrModeAuto{}, False{}), Bool.not(rr_zero_both_rings(rr_scope(RrModeAuto{}, False{})))) == True{} : Bool}

def rr_auto_clean_is_state_only():
  {==}

# LAW RR.4: Zeroing bytes for a DeltaNet ring NEVER exceeds `one_bytes * alloc_slots` when alloc_slots >= 1.
# promise: DeltaNet reset bytes never exceed the tensor's own allocated ring bytes (`one_bytes * alloc_slots`)
# catches: Full reset returns one_bytes * (alloc_slots + 1) or ignores alloc_slots in favor of a larger constant
law rr_gdn_zero_within_alloc:
  for +sc: RrScope
  for +one_bytes: Nat
  for +slots_m1: Nat
  {Nat.is_le(rr_gdn_zero_bytes(sc, one_bytes, 1n+slots_m1), Nat.mul(one_bytes, 1n+slots_m1)) == True{} : Bool}

def rr_gdn_zero_within_alloc(sc, one_bytes, slots_m1):
  match sc:
    case RrScopeFull{}:
      L.nat_le_refl(Nat.mul(one_bytes, 1n+slots_m1))
    case RrScopeDirty{}:
      %L.nat_mul_one_r(one_bytes) : {Nat.is_le(_, Nat.mul(one_bytes, 1n+slots_m1)) == True{} : Bool}
      L.nat_mul_le_mono_l(one_bytes, 1n, 1n+slots_m1, L.nat_le_zero(slots_m1))
    case RrScopeState{}:
      %L.nat_mul_one_r(one_bytes) : {Nat.is_le(_, Nat.mul(one_bytes, 1n+slots_m1)) == True{} : Bool}
      L.nat_mul_le_mono_l(one_bytes, 1n, 1n+slots_m1, L.nat_le_zero(slots_m1))

# LAW RR.5: Every reset scope zeroes at least slot 0 (`one_bytes`).
# promise: every reset scope covers at least slot 0 of ring 0 (`one_bytes`)
# catches: StateOnly/Dirty returns 0n instead of one_bytes
law rr_gdn_zero_covers_slot0:
  for +sc: RrScope
  for +one_bytes: Nat
  for +alloc_slots: Nat
  {Nat.is_le(one_bytes, rr_gdn_zero_bytes(sc, one_bytes, alloc_slots)) == True{} : Bool}

def rr_gdn_zero_covers_slot0_full(+one_bytes: Nat, +max_slots: Nat, h: {Nat.is_le(1n, max_slots) == True{} : Bool}) -> {Nat.is_le(one_bytes, Nat.mul(one_bytes, max_slots)) == True{} : Bool}:
  %L.nat_mul_one_r(one_bytes) : {Nat.is_le(_, Nat.mul(one_bytes, max_slots)) == True{} : Bool}
  L.nat_mul_le_mono_l(one_bytes, 1n, max_slots, h)

def rr_gdn_zero_covers_slot0(sc, one_bytes, alloc_slots):
  match sc:
    case RrScopeFull{}:
      rr_gdn_zero_covers_slot0_full(one_bytes, Nat.max(1n, alloc_slots), rr_ge_one_max1(alloc_slots))
    case RrScopeDirty{}:
      L.nat_le_refl(one_bytes)
    case RrScopeState{}:
      L.nat_le_refl(one_bytes)

# LAW RR.6: Full reset covers ALL allocated slots of a DeltaNet ring (`one_bytes * alloc_slots`).
# promise: Full scope covers at least every allocated slot of the DeltaNet ring
# catches: Full arm replaced by StateOnly (`one_bytes`)
law rr_gdn_full_covers_all_slots:
  for +one_bytes: Nat
  for +slots_m1: Nat
  {Nat.is_le(Nat.mul(one_bytes, 1n+slots_m1), rr_gdn_zero_bytes(RrScopeFull{}, one_bytes, 1n+slots_m1)) == True{} : Bool}

def rr_gdn_full_covers_all_slots(one_bytes, slots_m1):
  L.nat_le_refl(Nat.mul(one_bytes, 1n+slots_m1))

# LAW RR.7: ChunkMajor dirty bytes NEVER exceed `cache_bytes`.
# promise: ChunkMajor dirty zero fill never exceeds cache_bytes
# catches: Nat.min(c) dropped from chunk count in rr_chunk_dirty_nonzero
law rr_chunk_dirty_within_cache:
  for +cb: Nat
  for +chunks: Nat
  for +rows: Nat
  for +cr: Nat
  {Nat.is_le(rr_chunk_dirty_bytes(cb, chunks, rows, cr), cb) == True{} : Bool}

def rr_chunk_dirty_within_cache_nz(+cb: Nat, +c_m1: Nat, +rows: Nat, +cr: Nat) -> {Nat.is_le(Nat.mul(Nat.div(cb, 1n+c_m1), Nat.min(rr_div_ceil(rows, cr), 1n+c_m1)), cb) == True{} : Bool}:
  L.nat_le_trans(
    Nat.mul(Nat.div(cb, 1n+c_m1), Nat.min(rr_div_ceil(rows, cr), 1n+c_m1)),
    Nat.mul(Nat.div(cb, 1n+c_m1), 1n+c_m1),
    cb,
    L.nat_mul_le_mono_l(Nat.div(cb, 1n+c_m1), Nat.min(rr_div_ceil(rows, cr), 1n+c_m1), 1n+c_m1, L.nat_min_le_r(rr_div_ceil(rows, cr), 1n+c_m1)),
    L.nat_align_down_le(cb, c_m1)
  )

def rr_chunk_dirty_within_rows_nz(+cb: Nat, +chunks: Nat, +rows: Nat, +cr: Nat) -> {Nat.is_le(rr_chunk_dirty_nonzero(cb, Nat.max(1n, chunks), rows, cr), cb) == True{} : Bool}:
  match cb:
    case 0n:
      {==}
    case 1n+cp:
      match chunks:
        case 0n:
          rr_chunk_dirty_within_cache_nz(1n+cp, 0n, rows, cr)
        case 1n+cm1:
          rr_chunk_dirty_within_cache_nz(1n+cp, cm1, rows, cr)

def rr_chunk_dirty_within_cache(cb, chunks, rows, cr):
  match rows:
    case 0n:
      L.nat_le_zero(cb)
    case 1n+rp:
      rr_chunk_dirty_within_rows_nz(cb, chunks, 1n+rp, cr)

# LAW RR.8: Zero dirty rows produces zero bytes for ChunkMajor and zero rows for HeadMajor.
# promise: zero dirty rows requires zero KV bytes to clear
# catches: rr_chunk_dirty_bytes or rr_head_dirty_rows returns non-zero when dirty_rows is 0
law rr_zero_dirty_clears_nothing:
  for +cb: Nat
  for +chunks: Nat
  for +rows: Nat
  for +cr: Nat
  for +rb: Nat
  {Bool.and(Nat.is_eq(rr_chunk_dirty_bytes(cb, chunks, 0n, cr), 0n), Nat.is_eq(rr_head_dirty_rows(rows, 0n, rb), 0n)) == True{} : Bool}

def rr_zero_dirty_clears_nothing(cb, chunks, rows, cr, rb):
  {==}

# LAW RR.9: Non-zero dirty rows in a ChunkMajor cache covers at least one chunk.
# promise: any positive dirty row count clears at least one full chunk
# catches: rr_div_ceil replaced with floor division so small row counts clear 0 bytes
law rr_chunk_dirty_covers_first_chunk:
  for +cb: Nat
  for +chunks: Nat
  for +r_m1: Nat
  for +cr: Nat
  {Nat.is_le(Nat.div(cb, Nat.max(1n, chunks)), rr_chunk_dirty_bytes(cb, chunks, 1n+r_m1, cr)) == True{} : Bool}

def rr_nat_eq_refl(+n: Nat) -> {Nat.is_eq(n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np: rr_nat_eq_refl(np)

def rr_zero_ne_succ_bool(+r_m1: Nat, h: {0n == 1n+r_m1 : Nat}) -> {False{} == True{} : Bool}:
  %Equal.sym(Nat, 0n, 1n+r_m1, h) : {Nat.is_eq(_, 1n+r_m1) == True{} : Bool}
  rr_nat_eq_refl(1n+r_m1)

def rr_zero_eq_succ_from_div(+r_m1: Nat, +bm1: Nat, hq: {Nat.div(1n+r_m1, 1n+bm1) == 0n : Nat}, hm: {Nat.mod(1n+r_m1, 1n+bm1) == 0n : Nat}) -> {0n == 1n+r_m1 : Nat}:
  %hm : {Nat.add(_, Nat.mul(0n, 1n+bm1)) == 1n+r_m1 : Nat}
  %hq : {Nat.add(Nat.mod(1n+r_m1, 1n+bm1), Nat.mul(_, 1n+bm1)) == 1n+r_m1 : Nat}
  L.nat_div_mod(1n+r_m1, bm1)

def rr_div_ceil_zero_q(+q: Nat, +r_m1: Nat, +bm1: Nat, hq: {Nat.div(1n+r_m1, 1n+bm1) == q : Nat}, hm: {Nat.mod(1n+r_m1, 1n+bm1) == 0n : Nat}) -> {Nat.is_le(1n, q) == True{} : Bool}:
  match q:
    case 1n+qp:
      L.nat_le_zero(qp)
    case 0n:
      L.bool_absurd({Nat.is_le(1n, 0n) == True{} : Bool}, rr_zero_ne_succ_bool(r_m1, rr_zero_eq_succ_from_div(r_m1, bm1, hq, hm)))

def rr_div_ceil_nz_step(+q: Nat, +m: Nat, +r_m1: Nat, +bm1: Nat, hq: {Nat.div(1n+r_m1, 1n+bm1) == q : Nat}, hm: {Nat.mod(1n+r_m1, 1n+bm1) == m : Nat}) -> {Nat.is_le(1n, rr_div_ceil_pick(q, Nat.is_gt(m, 0n))) == True{} : Bool}:
  match m:
    case 1n+mp:
      %L.nat_add_comm(1n, q) : {Nat.is_le(1n, _) == True{} : Bool}
      L.nat_le_zero(q)
    case 0n:
      rr_div_ceil_zero_q(q, r_m1, bm1, hq, hm)

def rr_div_ceil_pos_nz(+r_m1: Nat, +block: Nat) -> {Nat.is_le(1n, rr_div_ceil(1n+r_m1, block)) == True{} : Bool}:
  match block:
    case 0n:
      L.nat_le_zero(r_m1)
    case 1n+bm1:
      rr_div_ceil_nz_step(Nat.div(1n+r_m1, 1n+bm1), Nat.mod(1n+r_m1, 1n+bm1), r_m1, bm1, {==}, {==})

def rr_chunk_dirty_covers_step(+per_chunk: Nat, +c: Nat, +n_raw: Nat, ha: {Nat.is_le(1n, n_raw) == True{} : Bool}, hb: {Nat.is_le(1n, c) == True{} : Bool}) -> {Nat.is_le(per_chunk, Nat.mul(per_chunk, Nat.min(n_raw, c))) == True{} : Bool}:
  %L.nat_mul_one_r(per_chunk) : {Nat.is_le(_, Nat.mul(per_chunk, Nat.min(n_raw, c))) == True{} : Bool}
  L.nat_mul_le_mono_l(per_chunk, 1n, Nat.min(n_raw, c), rr_min_ge_one(n_raw, c, ha, hb))

def rr_chunk_dirty_covers_first_chunk(cb, chunks, r_m1, cr):
  match cb:
    case 0n:
      match chunks:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+cp:
      rr_chunk_dirty_covers_step(
        Nat.div(1n+cp, Nat.max(1n, chunks)),
        Nat.max(1n, chunks),
        rr_div_ceil(1n+r_m1, cr),
        rr_div_ceil_pos_nz(r_m1, cr),
        rr_ge_one_max1(chunks)
      )

# LAW RR.10: HeadMajor dirty rows NEVER exceed `max(1, rows)`.
# promise: HeadMajor dirty rows stay within [0, max(1, rows)]
# catches: rr_head_dirty_pick ignores rows clamp when floor + block > rows
law rr_head_dirty_within_rows:
  for +rows: Nat
  for +dirty: Nat
  for +rb: Nat
  {Nat.is_le(rr_head_dirty_rows(rows, dirty, rb), Nat.max(1n, rows)) == True{} : Bool}

def rr_head_dirty_pick_le(+r: Nat, +floor: Nat, +block: Nat, +fits: Bool, hfl: {Nat.is_le(floor, r) == True{} : Bool}, hfits: {Nat.is_le(block, Nat.sub(r, floor)) == fits : Bool}) -> {Nat.is_le(rr_head_dirty_pick(r, Nat.add(floor, block), fits), r) == True{} : Bool}:
  match fits:
    case True{}:
      %rr_le_add_sub_self(floor, r, hfl) : {Nat.is_le(Nat.add(floor, block), _) == True{} : Bool}
      L.nat_add_le_mono(floor, floor, block, Nat.sub(r, floor), L.nat_le_refl(floor), hfits)
    case False{}:
      L.nat_le_refl(r)

def rr_head_dirty_align_le(+r: Nat, +dirty: Nat, +bm1: Nat, +rz: Bool, hle: {Nat.is_le(dirty, r) == True{} : Bool}) -> {Nat.is_le(rr_head_dirty_align_nz(r, dirty, 1n+bm1, rz, Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1)), r) == True{} : Bool}:
  match rz:
    case True{}:
      hle
    case False{}:
      rr_head_dirty_pick_le(
        r,
        Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1),
        1n+bm1,
        Nat.is_le(1n+bm1, Nat.sub(r, Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1))),
        L.nat_le_trans(Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1), dirty, r, L.nat_align_down_le(dirty, bm1), hle),
        {==}
      )

def rr_head_dirty_lt_true_le(+r: Nat, +dirty: Nat, +rb: Nat, hle: {Nat.is_le(dirty, r) == True{} : Bool}) -> {Nat.is_le(rr_head_dirty_block(r, dirty, rb), r) == True{} : Bool}:
  match rb:
    case 0n:
      hle
    case 1n+bm1:
      rr_head_dirty_align_le(r, dirty, bm1, Nat.is_eq(Nat.mod(dirty, 1n+bm1), 0n), hle)

def rr_head_dirty_lt_proof(+r: Nat, +dirty: Nat, +rb: Nat, +w: Bool, hw: {Nat.is_lt(dirty, r) == w : Bool}) -> {Nat.is_le(rr_head_dirty_lt(r, dirty, rb, w), r) == True{} : Bool}:
  match w:
    case False{}:
      L.nat_le_refl(r)
    case True{}:
      rr_head_dirty_lt_true_le(r, dirty, rb, L.nat_lt_imp_le(dirty, r, hw))

def rr_head_dirty_within_rows(rows, dirty, rb):
  match dirty:
    case 0n:
      L.nat_le_zero(Nat.max(1n, rows))
    case 1n+rp:
      rr_head_dirty_lt_proof(Nat.max(1n, rows), 1n+rp, rb, Nat.is_lt(1n+rp, Nat.max(1n, rows)), {==})

# LAW RR.11: When dirty <= rows + extra (and so <= max(1, rows)), HeadMajor dirty rows cover all dirty rows.
# promise: HeadMajor dirty rows always cover all dirty rows whenever dirty <= max(1, rows)
# catches: rr_head_dirty_block truncates down to floor instead of aligning up
law rr_head_dirty_covers_input:
  for +dirty: Nat
  for +extra: Nat
  for +rb: Nat
  {Nat.is_le(dirty, rr_head_dirty_rows(Nat.add(dirty, extra), dirty, rb)) == True{} : Bool}

def rr_head_dirty_pick_ge(+dirty: Nat, +r: Nat, +floor: Nat, +block: Nat, +fits: Bool, hr: {Nat.is_le(dirty, r) == True{} : Bool}, hcov: {Nat.is_le(dirty, Nat.add(floor, block)) == True{} : Bool}) -> {Nat.is_le(dirty, rr_head_dirty_pick(r, Nat.add(floor, block), fits)) == True{} : Bool}:
  match fits:
    case True{}:
      hcov
    case False{}:
      hr

def rr_head_dirty_cov_from_eq(+dirty: Nat, +floor: Nat, +m: Nat, +bm1: Nat, eq_d: {Nat.add(floor, m) == dirty : Nat}, hm_le: {Nat.is_le(m, 1n+bm1) == True{} : Bool}) -> {Nat.is_le(dirty, Nat.add(floor, 1n+bm1)) == True{} : Bool}:
  %eq_d : {Nat.is_le(_, Nat.add(floor, 1n+bm1)) == True{} : Bool}
  L.nat_add_le_mono(floor, floor, m, 1n+bm1, L.nat_le_refl(floor), hm_le)

def rr_head_dirty_align_ge(+dirty: Nat, +r: Nat, +bm1: Nat, +rz: Bool, hr: {Nat.is_le(dirty, r) == True{} : Bool}) -> {Nat.is_le(dirty, rr_head_dirty_align_nz(r, dirty, 1n+bm1, rz, Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1))) == True{} : Bool}:
  match rz:
    case True{}:
      L.nat_le_refl(dirty)
    case False{}:
      rr_head_dirty_pick_ge(
        dirty,
        r,
        Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1),
        1n+bm1,
        Nat.is_le(1n+bm1, Nat.sub(r, Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1))),
        hr,
        rr_head_dirty_cov_from_eq(
          dirty,
          Nat.mul(Nat.div(dirty, 1n+bm1), 1n+bm1),
          Nat.mod(dirty, 1n+bm1),
          bm1,
          L.nat_div_mod_r(dirty, bm1),
          L.nat_lt_imp_le(Nat.mod(dirty, 1n+bm1), 1n+bm1, L.nat_mod_lt(dirty, bm1))
        )
      )

def rr_head_dirty_covers_true_lt(+r: Nat, +dirty: Nat, +rb: Nat, hr: {Nat.is_le(dirty, r) == True{} : Bool}) -> {Nat.is_le(dirty, rr_head_dirty_block(r, dirty, rb)) == True{} : Bool}:
  match rb:
    case 0n:
      L.nat_le_refl(dirty)
    case 1n+bm1:
      rr_head_dirty_align_ge(dirty, r, bm1, Nat.is_eq(Nat.mod(dirty, 1n+bm1), 0n), hr)

def rr_head_dirty_covers_lt(+r: Nat, +dirty: Nat, +rb: Nat, +w: Bool, hr: {Nat.is_le(dirty, r) == True{} : Bool}) -> {Nat.is_le(dirty, rr_head_dirty_lt(r, dirty, rb, w)) == True{} : Bool}:
  match w:
    case False{}:
      hr
    case True{}:
      rr_head_dirty_covers_true_lt(r, dirty, rb, hr)

def rr_head_dirty_covers_input(dirty, extra, rb):
  match dirty:
    case 0n:
      {==}
    case 1n+rp:
      rr_head_dirty_covers_lt(
        Nat.max(1n, Nat.add(1n+rp, extra)),
        1n+rp,
        rb,
        Nat.is_lt(1n+rp, Nat.max(1n, Nat.add(1n+rp, extra))),
        L.nat_le_trans(
          1n+rp,
          Nat.add(1n+rp, extra),
          Nat.max(1n, Nat.add(1n+rp, extra)),
          L.nat_add_le_r(1n+rp, extra),
          L.nat_max_ge_r(1n, Nat.add(1n+rp, extra))
        )
      )

# LAW RR.12: HeadMajor span bytes for any n rows NEVER exceed one head's byte stride.
# promise: a HeadMajor span byte length never exceeds per_head bytes
# catches: Nat.min(n, rows) omitted in rr_head_span_bytes
law rr_head_span_within_per_head:
  for +cb: Nat
  for +heads: Nat
  for +rows: Nat
  for +n: Nat
  {Nat.is_le(rr_head_span_bytes(cb, heads, rows, n), Nat.div(cb, Nat.max(1n, heads))) == True{} : Bool}

def rr_align_down_le_l(+ph: Nat, +rm1: Nat) -> {Nat.is_le(Nat.mul(1n+rm1, Nat.div(ph, 1n+rm1)), ph) == True{} : Bool}:
  %L.nat_mul_comm(Nat.div(ph, 1n+rm1), 1n+rm1) : {Nat.is_le(_, ph) == True{} : Bool}
  L.nat_align_down_le(ph, rm1)

def rr_head_span_within_per_head_aux(+ph: Nat, +rm1: Nat, +n: Nat) -> {Nat.is_le(Nat.mul(Nat.min(n, 1n+rm1), Nat.div(ph, 1n+rm1)), ph) == True{} : Bool}:
  L.nat_le_trans(
    Nat.mul(Nat.min(n, 1n+rm1), Nat.div(ph, 1n+rm1)),
    Nat.mul(1n+rm1, Nat.div(ph, 1n+rm1)),
    ph,
    L.nat_mul_le_mono_r(Nat.min(n, 1n+rm1), 1n+rm1, Nat.div(ph, 1n+rm1), L.nat_min_le_r(n, 1n+rm1)),
    rr_align_down_le_l(ph, rm1)
  )

def rr_head_span_within_per_head(cb, heads, rows, n):
  match rows:
    case 0n:
      rr_head_span_within_per_head_aux(Nat.div(cb, Nat.max(1n, heads)), 0n, n)
    case 1n+rm1:
      rr_head_span_within_per_head_aux(Nat.div(cb, Nat.max(1n, heads)), rm1, n)

# LAW RR.13: Request high-water mark NEVER exceeds `max_ctx`.
# promise: request high-water row mark never exceeds max_ctx
# catches: max_ctx cap omitted from rr_high_water
law rr_hw_le_max_ctx:
  for +prompt: Nat
  for new_tokens: Nat
  for slack: Nat
  for +max_ctx: Nat
  {Nat.is_le(rr_high_water(prompt, new_tokens, slack, max_ctx), max_ctx) == True{} : Bool}

def rr_hw_step_pick_le(+cap: Nat, +base: Nat, +add: Nat, +fits: Bool, hb: {Nat.is_le(base, cap) == True{} : Bool}, hfits: {Nat.is_le(add, Nat.sub(cap, base)) == fits : Bool}) -> {Nat.is_le(rr_hw_step_pick(cap, base, add, fits), cap) == True{} : Bool}:
  match fits:
    case True{}:
      %rr_le_add_sub_self(base, cap, hb) : {Nat.is_le(Nat.add(base, add), _) == True{} : Bool}
      L.nat_add_le_mono(base, base, add, Nat.sub(cap, base), L.nat_le_refl(base), hfits)
    case False{}:
      L.nat_le_refl(cap)

def rr_hw_from_prompt_le(+max_ctx: Nat, +prompt: Nat, +new_tokens: Nat, +slack: Nat, +p_fits: Bool, hp: {Nat.is_le(prompt, max_ctx) == p_fits : Bool}) -> {Nat.is_le(rr_hw_from_prompt(max_ctx, prompt, new_tokens, slack, p_fits), max_ctx) == True{} : Bool}:
  match p_fits:
    case False{}:
      L.nat_le_refl(max_ctx)
    case True{}:
      rr_hw_step_pick_le(
        max_ctx,
        rr_hw_add_capped(max_ctx, prompt, new_tokens),
        slack,
        Nat.is_le(slack, Nat.sub(max_ctx, rr_hw_add_capped(max_ctx, prompt, new_tokens))),
        rr_hw_step_pick_le(max_ctx, prompt, new_tokens, Nat.is_le(new_tokens, Nat.sub(max_ctx, prompt)), hp, {==}),
        {==}
      )

def rr_hw_le_max_ctx(prompt, new_tokens, slack, max_ctx):
  rr_hw_from_prompt_le(max_ctx, prompt, new_tokens, slack, Nat.is_le(prompt, max_ctx), {==})

# LAW RR.14: When `prompt + new_tokens + slack <= max_ctx`, high-water covers at least `prompt + new_tokens`.
# promise: high-water covers prompt + new_tokens when context has headroom
# catches: new_tokens or prompt omitted from rr_high_water
law rr_hw_covers_prompt_plus_tokens:
  for +prompt: Nat
  for +new_tokens: Nat
  for +slack: Nat
  for +extra: Nat
  {Nat.is_le(Nat.add(prompt, new_tokens), rr_high_water(prompt, new_tokens, slack, Nat.add(Nat.add(prompt, new_tokens), Nat.add(slack, extra)))) == True{} : Bool}

def rr_hw_step_pick_ge(+target: Nat, +cap: Nat, +base: Nat, +add: Nat, +fits: Bool, hcap: {Nat.is_le(target, cap) == True{} : Bool}, hsum: {Nat.is_le(target, Nat.add(base, add)) == True{} : Bool}) -> {Nat.is_le(target, rr_hw_step_pick(cap, base, add, fits)) == True{} : Bool}:
  match fits:
    case True{}:
      hsum
    case False{}:
      hcap

def rr_hw_from_prompt_ge(+target: Nat, +max_ctx: Nat, +prompt: Nat, +new_tokens: Nat, +slack: Nat, +p_fits: Bool, hcap: {Nat.is_le(target, max_ctx) == True{} : Bool}, hp: {Nat.is_le(target, rr_hw_add_capped(max_ctx, prompt, new_tokens)) == True{} : Bool}) -> {Nat.is_le(target, rr_hw_from_prompt(max_ctx, prompt, new_tokens, slack, p_fits)) == True{} : Bool}:
  match p_fits:
    case False{}:
      hcap
    case True{}:
      rr_hw_step_pick_ge(
        target,
        max_ctx,
        rr_hw_add_capped(max_ctx, prompt, new_tokens),
        slack,
        Nat.is_le(slack, Nat.sub(max_ctx, rr_hw_add_capped(max_ctx, prompt, new_tokens))),
        hcap,
        L.nat_le_trans(target, rr_hw_add_capped(max_ctx, prompt, new_tokens), Nat.add(rr_hw_add_capped(max_ctx, prompt, new_tokens), slack), hp, L.nat_add_le_r(rr_hw_add_capped(max_ctx, prompt, new_tokens), slack))
      )

def rr_hw_covers_prompt_plus_tokens_aux(+target: Nat, +prompt: Nat, +new_tokens: Nat, +slack: Nat, +max_ctx: Nat, +hcap: {Nat.is_le(target, max_ctx) == True{} : Bool}, hsum: {Nat.is_le(target, Nat.add(prompt, new_tokens)) == True{} : Bool}) -> {Nat.is_le(target, rr_high_water(prompt, new_tokens, slack, max_ctx)) == True{} : Bool}:
  rr_hw_from_prompt_ge(
    target,
    max_ctx,
    prompt,
    new_tokens,
    slack,
    Nat.is_le(prompt, max_ctx),
    hcap,
    rr_hw_step_pick_ge(target, max_ctx, prompt, new_tokens, Nat.is_le(new_tokens, Nat.sub(max_ctx, prompt)), hcap, hsum)
  )

def rr_hw_covers_prompt_plus_tokens(prompt, new_tokens, slack, extra):
  rr_hw_covers_prompt_plus_tokens_aux(
    Nat.add(prompt, new_tokens),
    prompt,
    new_tokens,
    slack,
    Nat.add(Nat.add(prompt, new_tokens), Nat.add(slack, extra)),
    L.nat_add_le_r(Nat.add(prompt, new_tokens), Nat.add(slack, extra)),
    L.nat_le_refl(Nat.add(prompt, new_tokens))
  )

# LAW RR.15: Beginning a request always sets `unclean = True`, and a failed Auto request forces Dirty on the next reset.
# promise: starting a request marks ResetState unclean so a failure without end_ok forces Dirty on the next Auto request
# catches: rr_begin fails to set unclean = True
law rr_begin_sets_unclean_and_dirties_auto:
  for +hw: Nat
  for +held: Nat
  for +u: Bool
  for +prompt: Nat
  for +max_new: Nat
  for +slack: Nat
  for +max_ctx: Nat
  {Bool.and(rr_state_unclean(rr_begin(RrState{RrModeAuto{}, hw, held, u}, prompt, max_new, slack, max_ctx)), Bool.not(rr_is_state_only(RrModeAuto{}, rr_state_unclean(rr_begin(RrState{RrModeAuto{}, hw, held, u}, prompt, max_new, slack, max_ctx))))) == True{} : Bool}

def rr_begin_sets_unclean_and_dirties_auto(hw, held, u, prompt, max_new, slack, max_ctx):
  match u:
    case True{}: {==}
    case False{}: {==}

# LAW RR.16: Zeroing scopes (`Dirty`/`Full`) clear `held_hw` to 0 on begin.
# promise: when begin runs with Dirty or Full scope, held_hw is reset to 0n
# catches: rr_kept_hw returns old kv_hw for Dirty or Full scopes
law rr_begin_zeroing_resets_held:
  for hw: Nat
  for held: Nat
  for prompt: Nat
  for new_t: Nat
  for slack: Nat
  for max_ctx: Nat
  {Nat.is_eq(rr_state_held(rr_begin(RrState{RrModeDirty{}, hw, held, False{}}, prompt, new_t, slack, max_ctx)), 0n) == True{} : Bool}

def rr_begin_zeroing_resets_held(hw, held, prompt, new_t, slack, max_ctx):
  {==}

# LAW RR.17: Completing a request (`rr_end_ok`) clears `unclean` and preserves `held_hw <= kv_hw`.
# promise: rr_end_ok clears unclean and preserves kept dirty rows from before the request
# catches: rr_end_ok leaves unclean True or drops held_hw below max
law rr_end_ok_clears_unclean_and_keeps_held:
  for +m: RrMode
  for +hw: Nat
  for +held: Nat
  for +u: Bool
  for +prompt: Nat
  for +gen: Nat
  for +slack: Nat
  for +max_ctx: Nat
  {Bool.and(Bool.not(rr_state_unclean(rr_end_ok(RrState{m, hw, held, u}, prompt, gen, slack, max_ctx))), Nat.is_le(held, rr_state_hw(rr_end_ok(RrState{m, hw, held, u}, prompt, gen, slack, max_ctx)))) == True{} : Bool}

def rr_end_ok_clears_unclean_and_keeps_held(m, hw, held, u, prompt, gen, slack, max_ctx):
  L.nat_max_ge_l(held, rr_high_water(prompt, gen, slack, max_ctx))

def rr_eq_refl(a: Nat) -> {True{} == Nat.is_eq(a, a) : Bool}:
  match a:
    case 0n: {==}
    case 1n+ap:
      rr_eq_refl(ap)

# LAW RR.18: rr_head_full_span holds at max(1, rows) and fails at 0 (so partial spans never trigger full-cache zeroing).
# promise: rr_head_full_span holds at the full head-row capacity and fails at zero
# catches: rr_head_full_span checks n <= max(1, rows) instead of equality
law rr_head_full_span_strict:
  for +rows: Nat
  {Bool.and(rr_head_full_span(rows, Nat.max(1n, rows)), Bool.not(rr_head_full_span(rows, 0n))) == True{} : Bool}

def rr_head_full_span_strict(rows):
  match rows:
    case 0n: {==}
    case 1n+rp:
      %rr_eq_refl(rp) : {Bool.and(_, True{}) == True{} : Bool}
      {==}

def rr_mode_refl(m: RrMode) -> {m == m : RrMode}:
  {==}

# LAW RR.19: Beginning or completing a request never alters the configured reset mode in RrState.
# promise: rr_begin and rr_end_ok preserve the reset mode in RrState
# catches: rr_begin resets mode to RrModeAuto instead of keeping st.mode
law rr_state_mode_preserved:
  for st: RrState
  for +p: Nat
  for +g: Nat
  for +s: Nat
  for +mc: Nat
  {rr_state_mode(rr_end_ok(rr_begin(st, p, g, s, mc), p, g, s, mc)) == rr_state_mode(st) : RrMode}

def rr_state_mode_preserved(st, p, g, s, mc):
  match st:
    case RrState{m, hw, held, u}:
      rr_mode_refl(m)
