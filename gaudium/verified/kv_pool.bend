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

# ============================================================================
# 3. Prefix-Cache Tile-Boundary Reuse Rule (prefix_cache.hpp)
# ============================================================================
# Strict < on lcp and prompt_len expresses `snap_pos <= lcp - 1` and `snap_pos <= P - 1`
# with ZERO unsigned underflow when lcp == 0 or prompt_len == 0.
def pfx_reuse_valid(
  +snap_pos: Nat,
  lcp: Nat,
  prompt_len: Nat,
  +tile: Nat,
  need_mtp: Bool,
  mtp_upto: Nat,
  need_df: Bool,
  df_upto: Nat
) -> Bool:
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

# ============================================================================
# 5. Bounds of the prefix reuse rule (what every resume point satisfies)
# ============================================================================

# a < b and b <= a never hold together
def kv_lt_le_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_lt(a, b), Nat.is_le(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      kv_lt_le_excl(ap, bp)

def kv_and_not_self(b: Bool) -> {Bool.and(b, Bool.not(b)) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# (a && ((x && b) && c)) && y == False when x && y == False
def kv_reuse_excl_x(a: Bool, x: Bool, b: Bool, c: Bool, y: Bool, h: {Bool.and(x, y) == False{} : Bool}) -> {Bool.and(Bool.and(a, Bool.and(Bool.and(x, b), c)), y) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}:
      match x:
        case False{}: {==}
        case True{}:
          match b:
            case False{}: {==}
            case True{}:
              match c:
                case False{}: {==}
                case True{}: h

# (a && ((x && b) && c)) && y == False when b && y == False
def kv_reuse_excl_b(a: Bool, x: Bool, b: Bool, c: Bool, y: Bool, h: {Bool.and(b, y) == False{} : Bool}) -> {Bool.and(Bool.and(a, Bool.and(Bool.and(x, b), c)), y) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}:
      match x:
        case False{}: {==}
        case True{}:
          match b:
            case False{}: {==}
            case True{}:
              match c:
                case False{}: {==}
                case True{}: h

# ((g && (t && m)) && r) && !m == False
def kv_reuse_excl_m(g: Bool, t: Bool, m: Bool, r: Bool) -> {Bool.and(Bool.and(Bool.and(g, Bool.and(t, m)), r), Bool.not(m)) == False{} : Bool}:
  match g:
    case False{}: {==}
    case True{}:
      match t:
        case False{}: {==}
        case True{}:
          match m:
            case False{}: {==}
            case True{}:
              match r:
                case False{}: {==}
                case True{}: {==}

# (a && (b && (c && d))) && !d == False
def kv_reuse_excl_d(a: Bool, b: Bool, c: Bool, d: Bool) -> {Bool.and(Bool.and(a, Bool.and(b, Bool.and(c, d))), Bool.not(d)) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}:
      match b:
        case False{}: {==}
        case True{}:
          match c:
            case False{}: {==}
            case True{}:
              kv_and_not_self(d)

# LAW KV.13: a resume point never lies at or beyond the rows the new prompt shares with the
# held one.
law pfx_reuse_within_common_prefix:
  for +pos: Nat
  for +lcp: Nat
  for +p: Nat
  for +tile: Nat
  for +nm: Bool
  for +mu: Nat
  for +nd: Bool
  for +du: Nat
  {Bool.and(pfx_reuse_valid(pos, lcp, p, tile, nm, mu, nd, du), Nat.is_le(lcp, pos)) == False{} : Bool}

def pfx_reuse_within_common_prefix(pos, lcp, p, tile, nm, mu, nd, du):
  kv_reuse_excl_x(
    Bool.and(Nat.is_gt(pos, 0n), Bool.and(Nat.is_gt(tile, 0n), Nat.is_eq(Nat.mod(pos, tile), 0n))),
    Nat.is_lt(pos, lcp),
    Nat.is_lt(pos, p),
    Bool.and(Bool.or(Bool.not(nm), Nat.is_le(pos, mu)), Bool.or(Bool.not(nd), Nat.is_le(pos, du))),
    Nat.is_le(lcp, pos),
    kv_lt_le_excl(pos, lcp)
  )

# LAW KV.14: a resume point never lies at or beyond the end of the prompt: at least one row is
# always prefilled, so the readout of the last prompt row is always this request's own.
law pfx_reuse_before_prompt_end:
  for +pos: Nat
  for +lcp: Nat
  for +p: Nat
  for +tile: Nat
  for +nm: Bool
  for +mu: Nat
  for +nd: Bool
  for +du: Nat
  {Bool.and(pfx_reuse_valid(pos, lcp, p, tile, nm, mu, nd, du), Nat.is_le(p, pos)) == False{} : Bool}

def pfx_reuse_before_prompt_end(pos, lcp, p, tile, nm, mu, nd, du):
  kv_reuse_excl_b(
    Bool.and(Nat.is_gt(pos, 0n), Bool.and(Nat.is_gt(tile, 0n), Nat.is_eq(Nat.mod(pos, tile), 0n))),
    Nat.is_lt(pos, lcp),
    Nat.is_lt(pos, p),
    Bool.and(Bool.or(Bool.not(nm), Nat.is_le(pos, mu)), Bool.or(Bool.not(nd), Nat.is_le(pos, du))),
    Nat.is_le(p, pos),
    kv_lt_le_excl(pos, p)
  )

# LAW KV.15: a resume point is always a whole number of prefill tiles.
law pfx_reuse_on_tile_boundary:
  for +pos: Nat
  for +lcp: Nat
  for +p: Nat
  for +tile: Nat
  for +nm: Bool
  for +mu: Nat
  for +nd: Bool
  for +du: Nat
  {Bool.and(pfx_reuse_valid(pos, lcp, p, tile, nm, mu, nd, du), Bool.not(Nat.is_eq(Nat.mod(pos, tile), 0n))) == False{} : Bool}

def pfx_reuse_on_tile_boundary(pos, lcp, p, tile, nm, mu, nd, du):
  kv_reuse_excl_m(
    Nat.is_gt(pos, 0n),
    Nat.is_gt(tile, 0n),
    Nat.is_eq(Nat.mod(pos, tile), 0n),
    Bool.and(
      Bool.and(Nat.is_lt(pos, lcp), Nat.is_lt(pos, p)),
      Bool.and(Bool.or(Bool.not(nm), Nat.is_le(pos, mu)), Bool.or(Bool.not(nd), Nat.is_le(pos, du)))
    )
  )

# LAW KV.16: with a drafter that keeps context rows, a resume point never lies beyond the rows
# the drafter still holds.
law pfx_reuse_within_drafter_rows:
  for +pos: Nat
  for +lcp: Nat
  for +p: Nat
  for +tile: Nat
  for +nm: Bool
  for +mu: Nat
  for +du: Nat
  {Bool.and(pfx_reuse_valid(pos, lcp, p, tile, nm, mu, True{}, du), Bool.not(Nat.is_le(pos, du))) == False{} : Bool}

def pfx_reuse_within_drafter_rows(pos, lcp, p, tile, nm, mu, du):
  kv_reuse_excl_d(
    Bool.and(Nat.is_gt(pos, 0n), Bool.and(Nat.is_gt(tile, 0n), Nat.is_eq(Nat.mod(pos, tile), 0n))),
    Bool.and(Nat.is_lt(pos, lcp), Nat.is_lt(pos, p)),
    Bool.or(Bool.not(nm), Nat.is_le(pos, mu)),
    Nat.is_le(pos, du)
  )

# ============================================================================
# 6. Host-RAM tier of prefix reuse: the rules that build on the ones above
# ============================================================================
#
# The tier itself (entry life, budget, what a save and a restore decide) is `host_cache.bend`.

# Is `pos` a valid place to resume from an entry kept in host memory? The same rule as for a
# position kept on the device (`pfx_reuse_valid`, with `cap` the rows the new prompt shares
# with the entry's prompt, cut at the first image that differs), and the entry must hold the
# rows before it: `saved_rows` rows are complete in host memory.
def hc_resume_valid(+pos: Nat, cap: Nat, prompt_len: Nat, +tile: Nat, saved_rows: Nat) -> Bool:
  Bool.and(pfx_reuse_valid(pos, cap, prompt_len, tile, False{}, 0n, False{}, 0n), Nat.is_le(pos, saved_rows))

# May an entry leave to make room? Never one a copy reads or writes (`pool_can_evict`), and
# never the entry the room is for.
def hc_can_evict(pins: Nat, busy: Bool, is_target: Bool) -> Bool:
  Bool.and(pool_can_evict(pins, busy), Bool.not(is_target))

# Bytes (or blocks) a ledger has given out.
def pool_allocated(p: PoolAcct) -> Nat:
  match p:
    case PoolAcct{a, f}:
      a

# (x && w) && y == False when x && y == False
def kv_and3_excl_l(x: Bool, w: Bool, y: Bool, h: {Bool.and(x, y) == False{} : Bool}) -> {Bool.and(Bool.and(x, w), y) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      match w:
        case False{}:
          {==}
        case True{}:
          h

# (x && w) && y == False when w && y == False
def kv_and3_excl_r(x: Bool, w: Bool, y: Bool, h: {Bool.and(w, y) == False{} : Bool}) -> {Bool.and(Bool.and(x, w), y) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      h

# a <= b and b < a never hold together
def kv_le_lt_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_le(a, b), Nat.is_lt(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      kv_le_lt_excl(ap, bp)

def kv_and_not_true_r(b: Bool) -> {Bool.and(b, Bool.not(True{})) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# LAW KV.17: a resume point of an entry never lies at or beyond the rows the new prompt shares
# with the entry's prompt (and so never across an image that differs: the shared rows stop
# before it).
law hc_resume_within_shared_rows:
  for +pos: Nat
  for +cap: Nat
  for +p: Nat
  for +tile: Nat
  for +saved: Nat
  {Bool.and(hc_resume_valid(pos, cap, p, tile, saved), Nat.is_le(cap, pos)) == False{} : Bool}

def hc_resume_within_shared_rows(pos, cap, p, tile, saved):
  kv_and3_excl_l(
    pfx_reuse_valid(pos, cap, p, tile, False{}, 0n, False{}, 0n),
    Nat.is_le(pos, saved),
    Nat.is_le(cap, pos),
    pfx_reuse_within_common_prefix(pos, cap, p, tile, False{}, 0n, False{}, 0n)
  )

# LAW KV.18: nor beyond the rows the entry holds complete.
law hc_resume_within_saved_rows:
  for +pos: Nat
  for +cap: Nat
  for +p: Nat
  for +tile: Nat
  for +saved: Nat
  {Bool.and(hc_resume_valid(pos, cap, p, tile, saved), Nat.is_lt(saved, pos)) == False{} : Bool}

def hc_resume_within_saved_rows(pos, cap, p, tile, saved):
  kv_and3_excl_r(
    pfx_reuse_valid(pos, cap, p, tile, False{}, 0n, False{}, 0n),
    Nat.is_le(pos, saved),
    Nat.is_lt(saved, pos),
    kv_le_lt_excl(pos, saved)
  )

# LAW KV.19: nor at or beyond the end of the prompt: the last rows are always prefilled.
law hc_resume_before_prompt_end:
  for +pos: Nat
  for +cap: Nat
  for +p: Nat
  for +tile: Nat
  for +saved: Nat
  {Bool.and(hc_resume_valid(pos, cap, p, tile, saved), Nat.is_le(p, pos)) == False{} : Bool}

def hc_resume_before_prompt_end(pos, cap, p, tile, saved):
  kv_and3_excl_l(
    pfx_reuse_valid(pos, cap, p, tile, False{}, 0n, False{}, 0n),
    Nat.is_le(pos, saved),
    Nat.is_le(p, pos),
    pfx_reuse_before_prompt_end(pos, cap, p, tile, False{}, 0n, False{}, 0n)
  )

# LAW KV.20: and it is always a whole number of prefill tiles.
law hc_resume_on_tile_boundary:
  for +pos: Nat
  for +cap: Nat
  for +p: Nat
  for +tile: Nat
  for +saved: Nat
  {Bool.and(hc_resume_valid(pos, cap, p, tile, saved), Bool.not(Nat.is_eq(Nat.mod(pos, tile), 0n))) == False{} : Bool}

def hc_resume_on_tile_boundary(pos, cap, p, tile, saved):
  kv_and3_excl_l(
    pfx_reuse_valid(pos, cap, p, tile, False{}, 0n, False{}, 0n),
    Nat.is_le(pos, saved),
    Bool.not(Nat.is_eq(Nat.mod(pos, tile), 0n)),
    pfx_reuse_on_tile_boundary(pos, cap, p, tile, False{}, 0n, False{}, 0n)
  )

# LAW KV.21: the entry a save makes room for is never evicted for it.
law hc_target_never_evicted:
  for +pins: Nat
  for +busy: Bool
  {hc_can_evict(pins, busy, True{}) == False{} : Bool}

def hc_target_never_evicted(pins, busy):
  kv_and_not_true_r(pool_can_evict(pins, busy))

# LAW KV.22: an entry a copy reads is never evicted.
law hc_pinned_never_evicted:
  for +p: Nat
  for +busy: Bool
  for +t: Bool
  {hc_can_evict(1n+p, busy, t) == False{} : Bool}

def hc_pinned_never_evicted(p, busy, t):
  {==}

# LAW KV.23: an entry a copy writes is never evicted.
law hc_busy_never_evicted:
  for +pins: Nat
  for +t: Bool
  {hc_can_evict(pins, True{}, t) == False{} : Bool}

def hc_busy_never_evicted(pins, t):
  match pins:
    case 0n: {==}
    case 1n+_: {==}

def kv_alloc_le_total(p: PoolAcct) -> {True{} == Nat.is_le(pool_allocated(p), pool_total(p)) : Bool}:
  match p:
    case PoolAcct{a, f}:
      kv_le_add_r(a, f)

# LAW KV.24: what a ledger has given out never exceeds its capacity, whatever is asked of it:
# the bytes the host tier holds stay within its budget.
law pool_alloc_within_capacity:
  for +a: Nat
  for +f: Nat
  for +req: Nat
  {Nat.is_le(pool_allocated(pool_alloc(PoolAcct{a, f}, req)), Nat.add(a, f)) == True{} : Bool}

def pool_alloc_within_capacity(a, f, req):
  %pool_alloc_conserved(a, f, req) : {Nat.is_le(pool_allocated(pool_alloc(PoolAcct{a, f}, req)), _) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_le(pool_allocated(pool_alloc(PoolAcct{a, f}, req)), pool_total(pool_alloc(PoolAcct{a, f}, req))), kv_alloc_le_total(pool_alloc(PoolAcct{a, f}, req)))

# LAW KV.25: nor after bytes are given back.
law pool_free_within_capacity:
  for +a: Nat
  for +f: Nat
  for +rel: Nat
  {Nat.is_le(pool_allocated(pool_free(PoolAcct{a, f}, rel)), Nat.add(a, f)) == True{} : Bool}

def pool_free_within_capacity(a, f, rel):
  %pool_free_conserved(a, f, rel) : {Nat.is_le(pool_allocated(pool_free(PoolAcct{a, f}, rel)), _) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_le(pool_allocated(pool_free(PoolAcct{a, f}, rel)), pool_total(pool_free(PoolAcct{a, f}, rel))), kv_alloc_le_total(pool_free(PoolAcct{a, f}, rel)))

def kv_bool_and_true_r(b: Bool) -> {Bool.and(b, True{}) == b : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def kv_reuse_excl_mu(g: Bool, b: Bool, +pos: Nat, +mu: Nat, d: Bool) -> {Bool.and(Bool.and(g, Bool.and(b, Bool.and(Nat.is_le(pos, mu), d))), Nat.is_lt(mu, pos)) == False{} : Bool}:
  match g:
    case False{}: {==}
    case True{}:
      match b:
        case False{}: {==}
        case True{}:
          match d:
            case False{}:
              %Equal.sym(Bool, Bool.and(Nat.is_le(pos, mu), False{}), False{}, kv_bool_and_false_r(Nat.is_le(pos, mu))) : {Bool.and(_, Nat.is_lt(mu, pos)) == False{} : Bool}
              {==}
            case True{}:
              %Equal.sym(Bool, Bool.and(Nat.is_le(pos, mu), True{}), Nat.is_le(pos, mu), kv_bool_and_true_r(Nat.is_le(pos, mu))) : {Bool.and(_, Nat.is_lt(mu, pos)) == False{} : Bool}
              kv_le_lt_excl(pos, mu)

# promise: with an MTP head that tracks kept rows, a resume point never lies beyond the rows the MTP head holds
# catches: pfx_reuse_valid ignores need_mtp / mtp_upto when need_mtp is True
law pfx_reuse_within_mtp_rows:
  for +pos: Nat
  for +lcp: Nat
  for +p: Nat
  for +tile: Nat
  for +mu: Nat
  for +nd: Bool
  for +du: Nat
  {Bool.and(pfx_reuse_valid(pos, lcp, p, tile, True{}, mu, nd, du), Nat.is_lt(mu, pos)) == False{} : Bool}

def pfx_reuse_within_mtp_rows(pos, lcp, p, tile, mu, nd, du):
  kv_reuse_excl_mu(
    Bool.and(Nat.is_gt(pos, 0n), Bool.and(Nat.is_gt(tile, 0n), Nat.is_eq(Nat.mod(pos, tile), 0n))),
    Bool.and(Nat.is_lt(pos, lcp), Nat.is_lt(pos, p)),
    pos,
    mu,
    Bool.or(Bool.not(nd), Nat.is_le(pos, du))
  )
