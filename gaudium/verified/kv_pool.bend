# Slice 2A — Verified KV-Cache / Host-RAM Block Pool & Prefix Reuse Accounting
# Formally specifies and proves:
#   - Host-RAM block pool conservation (2 MiB blocks in 1 GiB slabs, OOM & double-free safety)
#   - Prefix-cache tile-boundary reuse rules (no underflow when lcp == 0, last-tile guard)

import Base

# ============================================================================
# 1. Helper Nat arithmetic lemmas (erased before Rust codegen)
# ============================================================================

def kv_add_zero(a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %kv_add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def kv_add_succ_r(a: Nat, b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %kv_add_succ_r(p, b) : {1n+Nat.add(p, 1n+b) == 1n+_ : Nat}
      {==}

def kv_le_add_r(a: Nat, b: Nat) -> {True{} == Nat.is_le(a, Nat.add(a, b)) : Bool}:
  match a:
    case 0n:
      match b:
        case 0n:
          {==}
        case 1n+_:
          {==}
    case 1n+p:
      kv_le_add_r(p, b)

def kv_le_add_gt_false(a: Nat, extra: Nat) -> {False{} == Nat.is_le(Nat.add(a, 1n+extra), a) : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      kv_le_add_gt_false(p, extra)

def kv_lt_add_r_false(cap: Nat, extra: Nat) -> {False{} == Nat.is_lt(Nat.add(cap, extra), cap) : Bool}:
  match cap:
    case 0n:
      match extra:
        case 0n:
          {==}
        case 1n+_:
          {==}
    case 1n+p:
      kv_lt_add_r_false(p, extra)

def kv_lt_refl_false(n: Nat) -> {False{} == Nat.is_lt(n, n) : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      kv_lt_refl_false(p)

def kv_sub_add_cancel(a: Nat, b: Nat) -> {Nat.sub(Nat.add(a, b), a) == b : Nat}:
  match a:
    case 0n:
      match b:
        case 0n:
          {==}
        case 1n+_:
          {==}
    case 1n+p:
      kv_sub_add_cancel(p, b)

def kv_divmod_zero_num(m: Nat, d: Nat, r: Nat) -> {Nat.divmod.go(0n, m, d, r) == (d, r) : Nat & Nat}:
  {==}

# ============================================================================
# 2. Block / Page Pool Accounting (allocated + free == capacity)
# ============================================================================

type PoolAcct is Data:
  PoolAcct{allocated: Nat, free: Nat}

def pool_init(cap: Nat) -> PoolAcct:
  PoolAcct{0n, cap}

def pool_total(p: PoolAcct) -> Nat:
  match p:
    case PoolAcct{a, f}:
      Nat.add(a, f)

def pool_can_alloc(p: PoolAcct, req: Nat) -> Bool:
  match p:
    case PoolAcct{_, f}:
      Nat.is_le(req, f)

def pool_alloc_if(a: Nat, f: Nat, +req: Nat, ok: Bool) -> PoolAcct:
  match ok:
    case True{}:
      PoolAcct{Nat.add(a, req), Nat.sub(f, req)}
    case False{}:
      PoolAcct{a, f}

def pool_alloc(p: PoolAcct, +req: Nat) -> PoolAcct:
  match p:
    case PoolAcct{a, +f}:
      pool_alloc_if(a, f, req, Nat.is_le(req, f))

def pool_free_if(a: Nat, f: Nat, +rel: Nat, ok: Bool) -> PoolAcct:
  match ok:
    case True{}:
      PoolAcct{Nat.sub(a, rel), Nat.add(f, rel)}
    case False{}:
      PoolAcct{a, f}

def pool_free(p: PoolAcct, +rel: Nat) -> PoolAcct:
  match p:
    case PoolAcct{+a, f}:
      pool_free_if(a, f, rel, Nat.is_le(rel, a))

# Pinned (in active DMA) or busy (being written) entries may NEVER be evicted.
def pool_can_evict(pins: Nat, busy: Bool) -> Bool:
  Bool.and(Nat.is_eq(pins, 0n), Bool.not(busy))

def pool_block_in_bounds(block_id: Nat, nblocks: Nat) -> Bool:
  Nat.is_lt(block_id, nblocks)

# 2 MB block count for `bytes` (ceil(bytes / block_bytes); safe 0 when block_bytes == 0).
def pool_blocks_for_nz(+bytes: Nat, +block_bytes: Nat) -> Nat:
  Nat.add(Nat.div(bytes, block_bytes), Bool.pick(Nat, Nat.is_eq(Nat.mod(bytes, block_bytes), 0n), 0n, 1n))

def pool_blocks_for_if(+bytes: Nat, +block_bytes: Nat, is_zero: Bool) -> Nat:
  match is_zero:
    case True{}:
      0n
    case False{}:
      pool_blocks_for_nz(bytes, block_bytes)

def pool_blocks_for(+bytes: Nat, +block_bytes: Nat) -> Nat:
  pool_blocks_for_if(bytes, block_bytes, Nat.is_eq(block_bytes, 0n))

# Slab coordinates (slab_idx, block_offset_in_slab) for 1 GB slabs (bps = 512 blocks/slab).
def pool_slab_coords(block_id: Nat, bps: Nat) -> Nat & Nat:
  Nat.divmod(block_id, bps)

# Two blocks merge into one DMA extent iff contiguous AND not crossing a slab boundary.
def pool_same_slab_adjacent(prev_block: Nat, +cur_block: Nat, bps: Nat) -> Bool:
  Bool.and(Nat.is_eq(Nat.add(prev_block, 1n), cur_block), Bool.not(Nat.is_eq(Nat.mod(cur_block, bps), 0n)))

# ============================================================================
# 3. Prefix-Cache Tile-Boundary Reuse Rule (prefix_cache.hpp)
# ============================================================================
# Strict < on lcp and prompt_len expresses `snap_pos <= lcp - 1` and `snap_pos <= P - 1`
# with ZERO unsigned underflow when lcp == 0 or prompt_len == 0.
def pfx_reuse_valid(+snap_pos: Nat, lcp: Nat, prompt_len: Nat, +tile: Nat, need_mtp: Bool, mtp_upto: Nat, need_df: Bool, df_upto: Nat) -> Bool:
  Bool.and(
    Bool.and(Nat.is_gt(snap_pos, 0n), Bool.and(Nat.is_gt(tile, 0n), Nat.is_eq(Nat.mod(snap_pos, tile), 0n))),
    Bool.and(
      Bool.and(Nat.is_lt(snap_pos, lcp), Nat.is_lt(snap_pos, prompt_len)),
      Bool.and(
        Bool.or(Bool.not(need_mtp), Nat.is_le(snap_pos, mtp_upto)),
        Bool.or(Bool.not(need_df), Nat.is_le(snap_pos, df_upto))
      )
    )
  )

# ============================================================================
# 4. Verified Conservation & Safety Laws
# ============================================================================

def kv_alloc_step_eq(+a: Nat, +fp: Nat, +rp: Nat, ok: Bool) -> {pool_total(pool_alloc_if(a, 1n+fp, 1n+rp, ok)) == pool_total(pool_alloc_if(1n+a, fp, rp, ok)) : Nat}:
  match ok:
    case True{}:
      %kv_add_succ_r(a, rp) : {Nat.add(Nat.add(a, 1n+rp), Nat.sub(fp, rp)) == Nat.add(_, Nat.sub(fp, rp)) : Nat}
      {==}
    case False{}:
      kv_add_succ_r(a, fp)

def kv_alloc_conserved_aux(+f: Nat, +req: Nat, +a: Nat) -> {pool_total(pool_alloc_if(a, f, req, Nat.is_le(req, f))) == Nat.add(a, f) : Nat}:
  match f req:
    case 0n 0n:
      kv_add_zero(Nat.add(a, 0n))
    case 0n 1n+rp:
      {==}
    case 1n+fp 0n:
      %kv_add_zero(a) : {Nat.add(Nat.add(a, 0n), 1n+fp) == Nat.add(_, 1n+fp) : Nat}
      {==}
    case 1n+fp 1n+rp:
      Equal.trans(
        Nat,
        pool_total(pool_alloc_if(a, 1n+fp, 1n+rp, Nat.is_le(rp, fp))),
        pool_total(pool_alloc_if(1n+a, fp, rp, Nat.is_le(rp, fp))),
        Nat.add(a, 1n+fp),
        kv_alloc_step_eq(a, fp, rp, Nat.is_le(rp, fp)),
        Equal.trans(
          Nat,
          pool_total(pool_alloc_if(1n+a, fp, rp, Nat.is_le(rp, fp))),
          1n+Nat.add(a, fp),
          Nat.add(a, 1n+fp),
          kv_alloc_conserved_aux(fp, rp, 1n+a),
          Equal.sym(Nat, Nat.add(a, 1n+fp), 1n+Nat.add(a, fp), kv_add_succ_r(a, fp))
        )
      )

def kv_free_step_eq(+ap: Nat, +f: Nat, +rp: Nat, ok: Bool) -> {pool_total(pool_free_if(1n+ap, f, 1n+rp, ok)) == pool_total(pool_free_if(ap, 1n+f, rp, ok)) : Nat}:
  match ok:
    case True{}:
      %kv_add_succ_r(f, rp) : {Nat.add(Nat.sub(ap, rp), Nat.add(f, 1n+rp)) == Nat.add(Nat.sub(ap, rp), _) : Nat}
      {==}
    case False{}:
      Equal.sym(Nat, Nat.add(ap, 1n+f), 1n+Nat.add(ap, f), kv_add_succ_r(ap, f))

def kv_free_conserved_aux(+a: Nat, +f: Nat, +rel: Nat) -> {pool_total(pool_free_if(a, f, rel, Nat.is_le(rel, a))) == Nat.add(a, f) : Nat}:
  match a rel:
    case 0n 0n:
      kv_add_zero(f)
    case 0n 1n+rp:
      {==}
    case 1n+ap 0n:
      %kv_add_zero(f) : {1n+Nat.add(ap, Nat.add(f, 0n)) == 1n+Nat.add(ap, _) : Nat}
      {==}
    case 1n+ap 1n+rp:
      Equal.trans(
        Nat,
        pool_total(pool_free_if(1n+ap, f, 1n+rp, Nat.is_le(rp, ap))),
        pool_total(pool_free_if(ap, 1n+f, rp, Nat.is_le(rp, ap))),
        1n+Nat.add(ap, f),
        kv_free_step_eq(ap, f, rp, Nat.is_le(rp, ap)),
        Equal.trans(
          Nat,
          pool_total(pool_free_if(ap, 1n+f, rp, Nat.is_le(rp, ap))),
          Nat.add(ap, 1n+f),
          1n+Nat.add(ap, f),
          kv_free_conserved_aux(ap, 1n+f, rp),
          kv_add_succ_r(ap, f)
        )
      )

# LAW KV.1: Initializing a pool preserves total == capacity.
law pool_init_conserved:
  for cap: Nat
  {pool_total(pool_init(cap)) == cap : Nat}

def pool_init_conserved(cap):
  {==}

# LAW KV.2: Every pool_alloc preserves allocated + free == capacity (on both hit and OOM).
law pool_alloc_conserved:
  for +a: Nat
  for +f: Nat
  for +req: Nat
  {pool_total(pool_alloc(PoolAcct{a, f}, req)) == Nat.add(a, f) : Nat}

def pool_alloc_conserved(a, f, req):
  kv_alloc_conserved_aux(f, req, a)

# LAW KV.3: Every pool_free preserves allocated + free == capacity (including rejected double-free).
law pool_free_conserved:
  for +a: Nat
  for +f: Nat
  for +rel: Nat
  {pool_total(pool_free(PoolAcct{a, f}, rel)) == Nat.add(a, f) : Nat}

def pool_free_conserved(a, f, rel):
  kv_free_conserved_aux(a, f, rel)

# LAW KV.4: Attempting to free more than allocated (double-free / underflow) is a strict no-op.
law pool_double_free_noop:
  for +a: Nat
  for +f: Nat
  for +extra: Nat
  {pool_free(PoolAcct{a, f}, Nat.add(a, 1n+extra)) == PoolAcct{a, f} : PoolAcct}

def pool_double_free_noop(a, f, extra):
  %kv_le_add_gt_false(a, extra) : {pool_free_if(a, f, Nat.add(a, 1n+extra), _) == PoolAcct{a, f} : PoolAcct}
  {==}

# LAW KV.5: Attempting to allocate more than free (OOM) is a strict no-op.
law pool_oom_alloc_noop:
  for +a: Nat
  for +f: Nat
  for +extra: Nat
  {pool_alloc(PoolAcct{a, f}, Nat.add(f, 1n+extra)) == PoolAcct{a, f} : PoolAcct}

def pool_oom_alloc_noop(a, f, extra):
  %kv_le_add_gt_false(f, extra) : {pool_alloc_if(a, f, Nat.add(f, 1n+extra), _) == PoolAcct{a, f} : PoolAcct}
  {==}

# LAW KV.6: A pinned entry (pins >= 1) can NEVER be evicted.
law pool_pinned_never_evicted:
  for p: Nat
  for busy: Bool
  {pool_can_evict(1n+p, busy) == False{} : Bool}

def pool_pinned_never_evicted(p, busy):
  {==}

# LAW KV.7: A busy entry (being written) can NEVER be evicted.
law pool_busy_never_evicted:
  for pins: Nat
  {pool_can_evict(pins, True{}) == False{} : Bool}

def pool_busy_never_evicted(pins):
  match pins:
    case 0n: {==}
    case 1n+_: {==}

# LAW KV.8: Out-of-bounds block IDs (>= capacity) are ALWAYS rejected.
law pool_oob_block_rejected:
  for +cap: Nat
  for +extra: Nat
  {pool_block_in_bounds(Nat.add(cap, extra), cap) == False{} : Bool}

def pool_oob_block_rejected(cap, extra):
  Equal.sym(Bool, False{}, Nat.is_lt(Nat.add(cap, extra), cap), kv_lt_add_r_false(cap, extra))

# LAW KV.9: Zero bytes always requires zero blocks (for any block size).
law pool_blocks_for_zero_bytes:
  for +b: Nat
  {pool_blocks_for(0n, b) == 0n : Nat}

def pool_blocks_for_zero_bytes(b):
  match b:
    case 0n: {==}
    case 1n+_: {==}

# LAW KV.10: Zero block size never divides by zero (returns 0 safely).
law pool_blocks_for_zero_blocksize_safe:
  for bytes: Nat
  {pool_blocks_for(bytes, 0n) == 0n : Nat}

def pool_blocks_for_zero_blocksize_safe(bytes):
  {==}

def kv_bool_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# LAW KV.11: Prefix reuse is ALWAYS refused when common prefix length is 0 (no underflow).
law pfx_reuse_zero_lcp_false:
  for +pos: Nat
  for p: Nat
  for +tile: Nat
  for nm: Bool
  for mu: Nat
  for nd: Bool
  for du: Nat
  {pfx_reuse_valid(pos, 0n, p, tile, nm, mu, nd, du) == False{} : Bool}

def pfx_reuse_zero_lcp_false(pos, p, tile, nm, mu, nd, du):
  match pos:
    case 0n: {==}
    case 1n+pp:
      match tile:
        case 0n: {==}
        case 1n+tp:
          kv_bool_and_false_r(Nat.is_eq(Nat.mod(1n+pp, 1n+tp), 0n))

# LAW KV.12: Prefix reuse is ALWAYS refused at snap_pos == prompt_len (final tile must always run).
law pfx_reuse_at_prompt_end_false:
  for +p: Nat
  for lcp: Nat
  for +tile: Nat
  for nm: Bool
  for mu: Nat
  for nd: Bool
  for du: Nat
  {pfx_reuse_valid(p, lcp, p, tile, nm, mu, nd, du) == False{} : Bool}

def pfx_reuse_at_prompt_end_false(p, lcp, tile, nm, mu, nd, du):
  match p:
    case 0n: {==}
    case 1n+pp:
      match tile:
        case 0n: {==}
        case 1n+tp:
          %kv_lt_refl_false(1n+pp) : {Bool.and(Nat.is_eq(Nat.mod(1n+pp, 1n+tp), 0n), Bool.and(Bool.and(Nat.is_lt(1n+pp, lcp), _), Bool.and(Bool.or(Bool.not(nm), Nat.is_le(1n+pp, mu)), Bool.or(Bool.not(nd), Nat.is_le(1n+pp, du))))) == False{} : Bool}
          %Equal.sym(Bool, Bool.and(Nat.is_lt(1n+pp, lcp), False{}), False{}, kv_bool_and_false_r(Nat.is_lt(1n+pp, lcp))) : {Bool.and(Nat.is_eq(Nat.mod(1n+pp, 1n+tp), 0n), Bool.and(_, Bool.and(Bool.or(Bool.not(nm), Nat.is_le(1n+pp, mu)), Bool.or(Bool.not(nd), Nat.is_le(1n+pp, du))))) == False{} : Bool}
          kv_bool_and_false_r(Nat.is_eq(Nat.mod(1n+pp, 1n+tp), 0n))
