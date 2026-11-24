# Gemma 4 image input: the resized size of an image, its soft-token count against the budget the
# tower is planned for, and the prefill units that let an image block attend both ways in the
# sliding layers.
# Formally specifies and proves:
#   - An accepted resized size never holds more soft tokens than the budget
#   - A prefill unit (a tile and the later tiles its image blocks reach into) joins a tile only
#     while it still fits the sliding ring beside the window
#   - A unit that fits keeps every row's window and its block's extension inside the ring
#   - A prefill row of an image block never attends a key the unit has not written, and attends
#     the whole of its block that the unit has written
#   - A unit takes a later tile only while one exists, always when a block runs into it, and a
#     unit of several tiles always runs the two-way form
#   - The bytes planned for a buffer hold it, at least one arena step, and whole steps only, so
#     the plan covers what the arena's sub-allocation consumes for any step

import Base
import nat_lemmas.bend as L
import vision_budget.bend as VB
import tensor_bounds.bend as TB
import inkling2.bend as IK

# ============================================================================
# 1. Helper lemmas (erased before Rust codegen)
# ============================================================================

# (a and b) and c is false when b and c is.
def g4v_and3_false(a: Bool, b: Bool, c: Bool, e: {Bool.and(b, c) == False{} : Bool}) -> {Bool.and(Bool.and(a, b), c) == False{} : Bool}:
  match a:
    case True{}:
      e
    case False{}:
      {==}

# b and not b is false.
def g4v_and_not_self(b: Bool) -> {Bool.and(b, Bool.not(b)) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# ============================================================================
# 2. Resized size of an image
# ============================================================================

# Side of the resized height from the floored candidates `th0` / `tw0` (whole squares of `side`
# pixels at the budget's scale): a height that floored to zero takes one square, a width that did
# takes the height of the aspect ratio `qh` (source height over width) in squares, at most
# `max_side`; both zero is no size.
def g4v_pick_h(hz: Bool, wz: Bool, th0: Nat, qh: Nat, side: Nat, max_side: Nat) -> Nat:
  match hz:
    case True{}:
      match wz:
        case True{}:
          0n
        case False{}:
          side
    case False{}:
      match wz:
        case True{}:
          Nat.min(Nat.mul(qh, side), max_side)
        case False{}:
          th0

# The same for the width (`qw`: source width over height).
def g4v_pick_w(hz: Bool, wz: Bool, tw0: Nat, qw: Nat, side: Nat, max_side: Nat) -> Nat:
  match hz:
    case True{}:
      match wz:
        case True{}:
          0n
        case False{}:
          Nat.min(Nat.mul(qw, side), max_side)
    case False{}:
      match wz:
        case True{}:
          side
        case False{}:
          tw0

def g4v_target_h(+th0: Nat, +tw0: Nat, qh: Nat, side: Nat, max_side: Nat) -> Nat:
  g4v_pick_h(Nat.is_eq(th0, 0n), Nat.is_eq(tw0, 0n), th0, qh, side, max_side)

def g4v_target_w(+th0: Nat, +tw0: Nat, qw: Nat, side: Nat, max_side: Nat) -> Nat:
  g4v_pick_w(Nat.is_eq(th0, 0n), Nat.is_eq(tw0, 0n), tw0, qw, side, max_side)

# Soft tokens of a resized image of `th x tw` pixels: the tokens of its grid of `patch`-pixel
# patches under a `merge x merge` pooling, counted as every image of the request is counted.
def g4v_tokens(th: Nat, tw: Nat, +patch: Nat, +merge: Nat) -> Nat:
  VB.vis_grid_tokens(Nat.div(th, patch), Nat.div(tw, patch), merge)

# Whether a resized size is taken: both sides whole squares of `patch * merge` pixels, neither
# empty, and at most `max_soft` tokens (the overflow-free canvas test of the media check, on the
# token grid the count uses).
def g4v_size_ok(+th: Nat, +tw: Nat, +patch: Nat, +merge: Nat, max_soft: Nat) -> Bool:
  Bool.and(Bool.and(Nat.is_eq(Nat.mod(th, Nat.mul(patch, merge)), 0n), Nat.is_eq(Nat.mod(tw, Nat.mul(patch, merge)), 0n)), VB.media_canvas_ok(Nat.div(Nat.div(tw, patch), merge), Nat.div(Nat.div(th, patch), merge), max_soft))

# ============================================================================
# 1b. Geometry the image path's graphs take
# ============================================================================

# A width the family's row kernels take (`gaudium_ga_rmsnorm`, `gaudium_ga_lnorm`): whole blocks
# of 128 channels, not empty.
def g4v_width_ok(+c: Nat) -> Bool:
  Bool.and(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 128n), 0n))

# The processor's and the position table's sizes: a patch, a pooling, a table row per axis.
def g4v_grid_ok(+patch: Nat, +pool: Nat, +npos: Nat) -> Bool:
  Bool.and(Nat.is_gt(patch, 0n), Bool.and(Nat.is_gt(pool, 0n), Nat.is_gt(npos, 0n)))

# A tower has layers and an MLP.
def g4v_tower_ok(+layers: Nat, +inter: Nat) -> Bool:
  Bool.and(Nat.is_gt(layers, 0n), Nat.is_gt(inter, 0n))

# ============================================================================
# 2b. Capacity of the image path (sizes the up-front plan)
# ============================================================================

# Soft tokens a pixel budget holds: one per `side x side` square.
def g4v_soft_budget(max_pixels: Nat, +side: Nat) -> Nat:
  Nat.div(max_pixels, Nat.mul(side, side))

# Whether the host processor takes a vision config: one frame per patch, 3 channels, and a pixel
# budget that holds at least one soft token (a square of `patch * merge` pixels a side; a zero
# patch or pooling size holds none). Refused on overflow: the sizes come from a checkpoint's
# config.
# rusty-bends: total
def g4v_processor_ok(+patch: Nat, +merge: Nat, +frames: Nat, +channels: Nat, max_pixels: Nat) -> Bool:
  Bool.and(Bool.and(Nat.is_eq(frames, 1n), Nat.is_eq(channels, 3n)), Nat.is_gt(g4v_soft_budget(max_pixels, Nat.mul(patch, merge)), 0n))

# Soft tokens one image may hold: the checkpoint's own budget, lowered to the tokens of `ppt`
# patches that the tower planned for `planned_patches` patches holds.
def g4v_image_budget(model_tokens: Nat, planned_patches: Nat, +ppt: Nat) -> Nat:
  Nat.min(model_tokens, Nat.div(planned_patches, ppt))

# Image tokens one request may hold: the configured count, never below one image of the budget;
# the budget when no count is configured (0).
def g4v_request_tokens(configured: Nat, +budget: Nat) -> Nat:
  match configured:
    case 0n:
      budget
    case 1n+c:
      Nat.max(1n+c, budget)

# Rows of the image path's recipe for an image of the budget: its patches (the tower) or its
# tokens (the unified embedder's one recipe of fixed rows).
def g4v_recipe_rows(unified: Bool, +budget: Nat, ppt: Nat) -> Nat:
  match unified:
    case True{}:
      budget
    case False{}:
      Nat.mul(budget, ppt)

# Feature columns behind a request's tokens that a recipe may write past its image: the unified
# recipe writes all its fixed rows, the tower exactly its image's tokens.
def g4v_pad_cols(unified: Bool, budget: Nat) -> Nat:
  match unified:
    case True{}:
      budget
    case False{}:
      0n

# Patches of the whole tokens `p` patches hold (tokens of `ppt` patches each).
def g4v_whole_token_patches(p: Nat, +ppt: Nat) -> Nat:
  Nat.mul(Nat.div(p, ppt), ppt)

# Whether a grid of `gh x gw` (patches or merged patches) has a position-table row on each axis.
def g4v_table_holds(+gh: Nat, +gw: Nat, +npos: Nat) -> Bool:
  Bool.and(Nat.is_le(gh, npos), Nat.is_le(gw, npos))

# The step of the Gemma 4 engine's sealed arenas: every buffer starts a whole number of steps from
# the arena's start and takes at least one step (the Rust sub-allocation, `arena_take`, takes this
# value for both).
def g4v_arena_step() -> Nat:
  256n

# Pooling order of a patch grid with `nbx` pooling blocks of `pool x pool` patches per row (the
# tower's input rows): row `i` holds a patch of the block of soft token `i / pool^2`, the blocks in
# raster order (the order of the soft tokens), at offset `i % pool^2` inside its block. The grid
# row of that patch:
def g4v_pool_y(+i: Nat, +nbx: Nat, +pool: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(Nat.div(i, Nat.mul(pool, pool)), nbx), pool), Nat.div(Nat.mod(i, Nat.mul(pool, pool)), pool))

# ... and its grid column.
def g4v_pool_x(+i: Nat, +nbx: Nat, +pool: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.mod(Nat.div(i, Nat.mul(pool, pool)), nbx), pool), Nat.mod(Nat.mod(i, Nat.mul(pool, pool)), pool))

# Rows of the pooling order of `nby x nbx` whole blocks.
def g4v_pool_count(nby: Nat, nbx: Nat, +pool: Nat) -> Nat:
  Nat.mul(Nat.mul(nby, nbx), Nat.mul(pool, pool))

# The pooling-order row of the patch at grid row `y`, column `x` (the inverse of the two above).
def g4v_pool_row(+y: Nat, +x: Nat, nbx: Nat, +pool: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(y, pool), nbx), Nat.div(x, pool)), Nat.mul(pool, pool)), Nat.add(Nat.mul(Nat.mod(y, pool), pool), Nat.mod(x, pool)))

# Device bytes one buffer of `bytes` takes in a sealed arena: at least one step, a whole number
# of steps (what the arena's sub-allocation consumes for it).
def g4v_part_bytes(+bytes: Nat) -> Nat:
  TB.align_up(Nat.max(bytes, g4v_arena_step()), g4v_arena_step())

# A plan's running total after one more buffer of `bytes`.
def g4v_add_part(total: Nat, +bytes: Nat) -> Nat:
  Nat.add(total, g4v_part_bytes(bytes))

# ============================================================================
# 3. Prefill units of the two-way image blocks
# ============================================================================

# Whether `rows` consecutive prefill rows fit a sliding layer's ring of `ring` slots beside its
# window of `w` positions: the first row's window and the last row both stay in the ring
# (`rows + w - 1 <= ring`, written so that nothing is formed past the operands).
def g4u_fits(+ring: Nat, +w: Nat, +rows: Nat) -> Bool:
  match w:
    case 0n:
      Nat.is_le(rows, ring)
    case 1n+wp:
      Bool.and(Nat.is_lt(wp, ring), Nat.is_le(rows, Nat.sub(ring, wp)))

# Whether the next tile joins a unit of `tiles` tiles: an image block runs from the unit's last
# tile into it (`cross`), and the unit has room for one more tile of the `max_tiles` the plan
# holds buffers for (the load checks that a unit of `max_tiles` full tiles fits the ring).
def g4u_join(cross: Bool, +tiles: Nat, max_tiles: Nat) -> Bool:
  Bool.and(cross, Nat.is_lt(tiles, max_tiles))

# Whether a tile of a unit is final when the image blocks over it end at `end` (the largest
# end) and the unit's keys end at `u`: a two-way unit's tile once its rows attended the whole of
# their blocks; a causal unit's tile always (its rows attend nothing after their own).
def g4u_final(two_way: Bool, +end: Nat, u: Nat) -> Bool:
  match two_way:
    case True{}:
      Nat.is_le(end, u)
    case False{}:
      True{}

# Tiles a prefill unit holds buffers for, when its tiles have `t` rows and an image block has at
# most `b`: the tiles a block can reach from any row of a tile, its own tile included (at least
# one). A shorter count only makes units end sooner; the tile that starts a unit is final
# either way or the prefill refuses (`g4u_final`).
def g4u_unit_tiles(+t: Nat, +b: Nat) -> Nat:
  Nat.max(1n, Nat.div(Nat.add(Nat.add(Nat.sub(t, 1n), b), Nat.sub(t, 1n)), t))

# Whether a unit takes the tile after its last tile (`next`, of the prompt's `n` tiles): in a size
# whose image blocks attend both ways, an image block over the unit's last row runs past the
# unit's rows, which end at `end` (`be`: the largest end of the blocks over that row, 0 without
# one).
def g4u_cross(bidir: Bool, +next: Nat, +n: Nat, +be: Nat, +end: Nat) -> Bool:
  Bool.and(bidir, Bool.and(Nat.is_lt(next, n), Nat.is_lt(end, be)))

# Whether a unit runs the two-way form: a size whose image blocks attend both ways, with an image
# block over one of the unit's tiles (`be`: the largest end of the blocks over its tiles, 0
# without one).
def g4u_two_way(bidir: Bool, +be: Nat) -> Bool:
  Bool.and(bidir, Nat.is_lt(0n, be))

# Whether row `p` lies in the image block of `n` rows from row `s`.
def g4u_in_block(+s: Nat, +n: Nat, +p: Nat) -> Bool:
  Bool.and(Nat.is_le(s, p), Nat.is_lt(p, Nat.add(s, n)))

# Whether the image block of `n` rows from row `s` has a row in the rows `[a, b)`.
def g4u_overlaps(+s: Nat, +n: Nat, +a: Nat, +b: Nat) -> Bool:
  Bool.and(Nat.is_lt(s, b), Nat.is_lt(a, Nat.add(s, n)))

# Positions after its own that the prefill row at position `p` attends in a sliding layer: a row
# of an image block the rest of its block (which ends before `e`), cut at `u`, the end of the keys
# the unit has written; a text row none.
def g4u_ext(in_block: Bool, p: Nat, e: Nat, u: Nat) -> Nat:
  match in_block:
    case True{}:
      Nat.sub(Nat.min(e, u), 1n+p)
    case False{}:
      0n

# ============================================================================
# 4. Laws
# ============================================================================

# and(b, le(m, n)) is false when b says n < m.
def g4u_lt_le_false(+n: Nat, +m: Nat, b: Bool, e: {Nat.is_lt(n, m) == b : Bool}) -> {Bool.and(b, Nat.is_le(m, n)) == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      L.nat_lt_le_false(n, m, e)

# 1 + u + k <= u never holds.
def g4u_past_not_le(+u: Nat, +k: Nat) -> {Nat.is_le(1n+Nat.add(u, k), u) == False{} : Bool}:
  match u:
    case 0n:
      {==}
    case 1n+up:
      g4u_past_not_le(up, k)

# p + 0 < 1 + p + k.
def g4u_lt_self_add(+p: Nat, +k: Nat) -> {Nat.is_lt(Nat.add(p, 0n), 1n+Nat.add(p, k)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(p, 0n), p, L.nat_add_zero_r(p)) : {Nat.is_lt(_, 1n+Nat.add(p, k)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(p, 1n+Nat.add(p, k)), Nat.is_le(p, Nat.add(p, k)), L.nat_lt_succ_eq_le(p, Nat.add(p, k))) : {_ == True{} : Bool}
  L.nat_add_le_r(p, k)

# x <= 1 + p + k gives p + (x - (1 + p)) < 1 + p + k.
def g4u_ext_lt(+p: Nat, +x: Nat, +k: Nat, h: {Nat.is_le(x, 1n+Nat.add(p, k)) == True{} : Bool}) -> {Nat.is_lt(Nat.add(p, Nat.sub(x, 1n+p)), 1n+Nat.add(p, k)) == True{} : Bool}:
  match p:
    case 0n:
      match x:
        case 0n:
          {==}
        case 1n+xp:
          %Equal.sym(Nat, Nat.sub(xp, 0n), xp, L.nat_sub_zero_r(xp)) : {Nat.is_lt(_, 1n+k) == True{} : Bool}
          %Equal.sym(Bool, Nat.is_lt(xp, 1n+k), Nat.is_le(xp, k), L.nat_lt_succ_eq_le(xp, k)) : {_ == True{} : Bool}
          h
    case 1n+pp:
      match x:
        case 0n:
          g4u_lt_self_add(pp, k)
        case 1n+xp:
          g4u_ext_lt(pp, xp, k, h)

# w = 0: an accepted unit (rows <= ring) keeps rows + 0 within ring + 1.
def g4u_fits_zero(+r: Nat, +n: Nat, b: Bool, e: {Nat.is_le(n, r) == b : Bool}) -> {Bool.and(b, Nat.is_lt(1n+r, Nat.add(n, 0n))) == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Nat, Nat.add(n, 0n), n, L.nat_add_zero_r(n)) : {Nat.is_lt(1n+r, _) == False{} : Bool}
      L.nat_le_lt_false(1n+r, n, L.nat_le_succ_r(n, r, e))

# w = 1 + wp: wp < ring and rows <= ring - wp keep rows + 1 + wp within ring + 1.
def g4u_fits_succ(+r: Nat, +wp: Nat, +n: Nat, b1: Bool, e1: {Nat.is_lt(wp, r) == b1 : Bool}, b2: Bool, e2: {Nat.is_le(n, Nat.sub(r, wp)) == b2 : Bool}) -> {Bool.and(Bool.and(b1, b2), Nat.is_lt(1n+r, Nat.add(n, 1n+wp))) == False{} : Bool}:
  match b1:
    case False{}:
      {==}
    case True{}:
      match b2:
        case False{}:
          {==}
        case True{}:
          %Equal.sym(Nat, Nat.add(n, 1n+wp), 1n+Nat.add(n, wp), L.nat_add_succ_r(n, wp)) : {Nat.is_lt(1n+r, _) == False{} : Bool}
          L.nat_le_lt_false(r, Nat.add(n, wp), L.nat_le_subst_r(Nat.add(n, wp), Nat.add(Nat.sub(r, wp), wp), r, L.nat_sub_add_cancel(r, wp, L.nat_lt_imp_le(wp, r, e1)), L.nat_add_le_mono(n, Nat.sub(r, wp), wp, wp, e2, L.nat_le_refl(wp))))

# LAW G4V.1
# promise: an image size the processor accepts never holds more soft tokens than the budget the tower is planned for.
# harm: an image past the budget would write soft tokens beyond the planned buffers.
law g4v_size_within_budget:
  for +th: Nat
  for +tw: Nat
  for +patch: Nat
  for +merge: Nat
  for +m: Nat
  {Bool.and(g4v_size_ok(th, tw, patch, merge, m), Nat.is_lt(m, g4v_tokens(th, tw, patch, merge))) == False{} : Bool}

def g4v_size_within_budget(th, tw, patch, merge, m):
  %Equal.sym(Nat, Nat.mul(Nat.div(Nat.div(th, patch), merge), Nat.div(Nat.div(tw, patch), merge)), Nat.mul(Nat.div(Nat.div(tw, patch), merge), Nat.div(Nat.div(th, patch), merge)), L.nat_mul_comm(Nat.div(Nat.div(th, patch), merge), Nat.div(Nat.div(tw, patch), merge))) : {Bool.and(Bool.and(Bool.and(Nat.is_eq(Nat.mod(th, Nat.mul(patch, merge)), 0n), Nat.is_eq(Nat.mod(tw, Nat.mul(patch, merge)), 0n)), VB.media_canvas_ok(Nat.div(Nat.div(tw, patch), merge), Nat.div(Nat.div(th, patch), merge), m)), Nat.is_lt(m, _)) == False{} : Bool}
  g4v_and3_false(Bool.and(Nat.is_eq(Nat.mod(th, Nat.mul(patch, merge)), 0n), Nat.is_eq(Nat.mod(tw, Nat.mul(patch, merge)), 0n)), VB.media_canvas_ok(Nat.div(Nat.div(tw, patch), merge), Nat.div(Nat.div(th, patch), merge), m), Nat.is_lt(m, Nat.mul(Nat.div(Nat.div(tw, patch), merge), Nat.div(Nat.div(th, patch), merge))), VB.media_canvas_within_limit(Nat.div(Nat.div(tw, patch), merge), Nat.div(Nat.div(th, patch), merge), m))

# LAW G4U.1
# promise: a prefill unit the ring is said to fit keeps its rows and the window of its first row inside the ring (rows + window <= ring + 1).
# harm: a later tile's keys would overwrite keys an earlier tile's rows still attend.
law g4u_fits_within_ring:
  for +r: Nat
  for +w: Nat
  for +n: Nat
  {Bool.and(g4u_fits(r, w, n), Nat.is_lt(1n+r, Nat.add(n, w))) == False{} : Bool}

def g4u_fits_within_ring(r, w, n):
  match w:
    case 0n:
      g4u_fits_zero(r, n, Nat.is_le(n, r), {==})
    case 1n+wp:
      g4u_fits_succ(r, wp, n, Nat.is_lt(wp, r), {==}, Nat.is_le(n, Nat.sub(r, wp)), {==})

# LAW G4U.2
# promise: a unit never takes a tile beyond the tile buffers the plan holds for it.
law g4u_join_within_buffers:
  for c: Bool
  for +n: Nat
  for +m: Nat
  {Bool.and(g4u_join(c, n, m), Nat.is_le(m, n)) == False{} : Bool}

def g4u_join_within_buffers(c, n, m):
  match c:
    case True{}:
      g4u_lt_le_false(n, m, Nat.is_lt(n, m), {==})
    case False{}:
      {==}

# LAW G4U.3
# promise: a tile whose image block runs past the keys its unit wrote is never taken as final.
# harm: its block rows would keep an attention that missed the rest of their image.
law g4u_final_refuses_past_end:
  for +u: Nat
  for +k: Nat
  {g4u_final(True{}, 1n+Nat.add(u, k), u) == False{} : Bool}

def g4u_final_refuses_past_end(u, k):
  g4u_past_not_le(u, k)

# LAW G4U.4
# promise: a prefill row in a sliding layer never attends a position past the end of the keys its unit has written.
# harm: it would read a ring slot that holds a stale or foreign key.
law g4u_ext_within_unit:
  for b: Bool
  for +p: Nat
  for +e: Nat
  for +k: Nat
  {Nat.is_lt(Nat.add(p, g4u_ext(b, p, e, 1n+Nat.add(p, k))), 1n+Nat.add(p, k)) == True{} : Bool}

def g4u_ext_within_unit(b, p, e, k):
  match b:
    case True{}:
      g4u_ext_lt(p, Nat.min(e, 1n+Nat.add(p, k)), k, L.nat_min_le_r(e, 1n+Nat.add(p, k)))
    case False{}:
      g4u_lt_self_add(p, k)

# LAW G4U.5
# promise: a tile of a causal unit (no two-way image block) is always final, so the prefill always moves past it.
# harm: the small sizes' image prompts would be refused whenever an image crosses a tile boundary.
law g4u_causal_tile_final:
  for +e: Nat
  for u: Nat
  {g4u_final(False{}, e, u) == True{} : Bool}

def g4u_causal_tile_final(e, u):
  {==}

# LAW G4U.6
# promise: a prefill unit always holds at least the tile it starts with.
law g4u_unit_tiles_at_least_one:
  for +t: Nat
  for +b: Nat
  {Nat.is_le(1n, g4u_unit_tiles(t, b)) == True{} : Bool}

def g4u_unit_tiles_at_least_one(t, b):
  L.nat_max_ge_l(1n, Nat.div(Nat.add(Nat.add(Nat.sub(t, 1n), b), Nat.sub(t, 1n)), t))

# LAW G4U.7
# promise: a text row attends nothing after its own position.
law g4u_text_row_causal:
  for +p: Nat
  for e: Nat
  for u: Nat
  {g4u_ext(False{}, p, e, u) == 0n : Nat}

def g4u_text_row_causal(p, e, u):
  {==}

# LAW G4U.8
# promise: a row of an image block never attends a position past the end of its own block.
# harm: it would attend text or another image that the reference keeps causal.
law g4u_ext_within_block:
  for +p: Nat
  for +k: Nat
  for +u: Nat
  {Nat.is_lt(Nat.add(p, g4u_ext(True{}, p, 1n+Nat.add(p, k), u)), 1n+Nat.add(p, k)) == True{} : Bool}

def g4u_ext_within_block(p, k, u):
  g4u_ext_lt(p, Nat.min(1n+Nat.add(p, k), u), k, L.nat_min_le_l(1n+Nat.add(p, k), u))

# s + n < s + n never holds.
def g4u_lt_self_false(+x: Nat) -> {Nat.is_lt(x, x) == False{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+xp:
      g4u_lt_self_false(xp)

# a and false is false.
def g4u_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# LAW G4U.9
# promise: the row right after an image block is not in it.
law g4u_block_end_row_outside:
  for +s: Nat
  for +n: Nat
  {g4u_in_block(s, n, Nat.add(s, n)) == False{} : Bool}

def g4u_block_end_row_outside(s, n):
  %Equal.sym(Bool, Nat.is_lt(Nat.add(s, n), Nat.add(s, n)), False{}, g4u_lt_self_false(Nat.add(s, n))) : {Bool.and(Nat.is_le(s, Nat.add(s, n)), _) == False{} : Bool}
  g4u_and_false_r(Nat.is_le(s, Nat.add(s, n)))

# x < x + k never holds the other way: x + k < x is false.
def g4u_add_lt_false(+x: Nat, +k: Nat) -> {Nat.is_lt(Nat.add(x, k), x) == False{} : Bool}:
  match x:
    case 0n:
      match k:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+xp:
      g4u_add_lt_false(xp, k)

# LAW G4U.10
# promise: an image block that ends at or before the first of some rows has no row among them.
law g4u_block_before_rows_outside:
  for +s: Nat
  for +n: Nat
  for +k: Nat
  for +b: Nat
  {g4u_overlaps(s, n, Nat.add(Nat.add(s, n), k), b) == False{} : Bool}

def g4u_block_before_rows_outside(s, n, k, b):
  %Equal.sym(Bool, Nat.is_lt(Nat.add(Nat.add(s, n), k), Nat.add(s, n)), False{}, g4u_add_lt_false(Nat.add(s, n), k)) : {Bool.and(Nat.is_lt(s, b), _) == False{} : Bool}
  g4u_and_false_r(Nat.is_lt(s, b))

# (a / (1 + bp)) (1 + bp) <= a.
def g4v_div_mul_le(+a: Nat, +bp: Nat) -> {Nat.is_le(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), a) == True{} : Bool}:
  L.nat_le_subst_r(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.add(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp)), a, L.nat_div_mod_r(a, bp), L.nat_add_le_r(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp)))

# LAW G4V.2
# promise: an image's soft-token budget never exceeds the checkpoint's own budget.
law g4v_image_budget_within_model:
  for +m: Nat
  for +p: Nat
  for +k: Nat
  {Nat.is_le(g4v_image_budget(m, p, k), m) == True{} : Bool}

def g4v_image_budget_within_model(m, p, k):
  L.nat_min_le_l(m, Nat.div(p, k))

# LAW G4V.3
# promise: the patches of an image of the budget never exceed the patches the tower is planned for.
# harm: the tower's planned buffers would be too small for an image the processor accepts.
law g4v_image_budget_within_tower:
  for +m: Nat
  for +p: Nat
  for +kp: Nat
  {Nat.is_le(Nat.mul(g4v_image_budget(m, p, 1n+kp), 1n+kp), p) == True{} : Bool}

def g4v_image_budget_within_tower(m, p, kp):
  L.nat_le_trans(Nat.mul(Nat.min(m, Nat.div(p, 1n+kp)), 1n+kp), Nat.mul(Nat.div(p, 1n+kp), 1n+kp), p, L.nat_mul_le_mono_r(Nat.min(m, Nat.div(p, 1n+kp)), Nat.div(p, 1n+kp), 1n+kp, L.nat_min_le_r(m, Nat.div(p, 1n+kp))), g4v_div_mul_le(p, kp))

# LAW G4V.4
# promise: the tokens a request may hold always take at least one image of the budget.
law g4v_request_holds_an_image:
  for c: Nat
  for +b: Nat
  {Nat.is_le(b, g4v_request_tokens(c, b)) == True{} : Bool}

def g4v_request_holds_an_image(c, b):
  match c:
    case 0n:
      L.nat_le_refl(b)
    case 1n+cp:
      L.nat_max_ge_r(1n+cp, b)

# LAW G4V.5
# promise: the feature buffer's padding holds every row the unified embedder's recipe writes past an image.
# harm: the last image's recipe would write past the feature buffer.
law g4v_unified_pad_holds_its_rows:
  for +b: Nat
  for k: Nat
  {Nat.is_le(g4v_recipe_rows(True{}, b, k), g4v_pad_cols(True{}, b)) == True{} : Bool}

def g4v_unified_pad_holds_its_rows(b, k):
  L.nat_le_refl(b)

# LAW G4V.6
# promise: the soft tokens a pixel budget is said to hold never take more pixels than the budget.
law g4v_soft_budget_within_pixels:
  for +px: Nat
  for +sp: Nat
  {Nat.is_le(Nat.mul(g4v_soft_budget(px, 1n+sp), Nat.mul(1n+sp, 1n+sp)), px) == True{} : Bool}

def g4v_soft_budget_within_pixels(px, sp):
  g4v_div_mul_le(px, Nat.add(sp, Nat.mul(sp, 1n+sp)))

# LAW G4V.7a
# promise: the bytes planned for a buffer always hold the buffer.
# harm: a sealed arena would run out before the last buffer of the plan.
law g4v_part_holds_buffer:
  for +b: Nat
  {Nat.is_le(b, g4v_part_bytes(b)) == True{} : Bool}

def g4v_part_holds_buffer(b):
  L.nat_le_trans(b, Nat.max(b, g4v_arena_step()), g4v_part_bytes(b), L.nat_max_ge_l(b, g4v_arena_step()), TB.align_up_never_shrinks(Nat.max(b, g4v_arena_step()), g4v_arena_step()))

# LAW G4V.7b
# promise: the bytes planned for a buffer hold at least one step of the arena, an empty buffer's included.
# harm: the arena takes a step for every buffer, so a plan of small buffers would count less than it consumes.
law g4v_part_holds_a_step:
  for +b: Nat
  {Nat.is_le(g4v_arena_step(), g4v_part_bytes(b)) == True{} : Bool}

def g4v_part_holds_a_step(b):
  L.nat_le_trans(g4v_arena_step(), Nat.max(b, g4v_arena_step()), g4v_part_bytes(b), L.nat_max_ge_r(b, g4v_arena_step()), TB.align_up_never_shrinks(Nat.max(b, g4v_arena_step()), g4v_arena_step()))

# (1 + rp + m) + (sp - rp) == 1 + sp + m when rp <= sp.
def g4v_add_rest(+rp: Nat, +sp: Nat, +m: Nat, h: {Nat.is_le(rp, sp) == True{} : Bool}) -> {Nat.add(Nat.add(1n+rp, m), Nat.sub(sp, rp)) == Nat.add(1n+sp, m) : Nat}:
  %Equal.sym(Nat, Nat.add(Nat.add(rp, m), Nat.sub(sp, rp)), Nat.add(rp, Nat.add(m, Nat.sub(sp, rp))), L.nat_add_assoc(rp, m, Nat.sub(sp, rp))) : {1n+_ == Nat.add(1n+sp, m) : Nat}
  %L.nat_add_comm(Nat.sub(sp, rp), m) : {1n+Nat.add(rp, _) == Nat.add(1n+sp, m) : Nat}
  %L.nat_add_assoc(rp, Nat.sub(sp, rp), m) : {1n+_ == Nat.add(1n+sp, m) : Nat}
  %L.nat_add_comm(Nat.sub(sp, rp), rp) : {1n+Nat.add(_, m) == Nat.add(1n+sp, m) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.sub(sp, rp), rp), sp, L.nat_sub_add_cancel(sp, rp, h)) : {1n+Nat.add(_, m) == Nat.add(1n+sp, m) : Nat}
  {==}

# x % (1 + sp) == r: rounding x up past its remainder r leaves whole multiples of 1 + sp.
def g4v_align_rem_whole(+x: Nat, +sp: Nat, +r: Nat, e: {Nat.mod(x, 1n+sp) == r : Nat}, lt: {Nat.is_lt(r, 1n+sp) == True{} : Bool}) -> {Nat.mod(TB.align_up_rem_if(x, 1n+sp, r, Nat.is_eq(r, 0n)), 1n+sp) == 0n : Nat}:
  match r:
    case 0n:
      e
    case 1n+rp:
      %L.nat_div_mod(x, sp) : {Nat.mod(Nat.add(_, Nat.sub(sp, rp)), 1n+sp) == 0n : Nat}
      %Equal.sym(Nat, Nat.mod(x, 1n+sp), 1n+rp, e) : {Nat.mod(Nat.add(Nat.add(_, Nat.mul(Nat.div(x, 1n+sp), 1n+sp)), Nat.sub(sp, rp)), 1n+sp) == 0n : Nat}
      %Equal.sym(Nat, Nat.add(Nat.add(1n+rp, Nat.mul(Nat.div(x, 1n+sp), 1n+sp)), Nat.sub(sp, rp)), Nat.add(1n+sp, Nat.mul(Nat.div(x, 1n+sp), 1n+sp)), g4v_add_rest(rp, sp, Nat.mul(Nat.div(x, 1n+sp), 1n+sp), L.nat_lt_imp_le(rp, sp, lt))) : {Nat.mod(_, 1n+sp) == 0n : Nat}
      IK.ikp_mod_mul(1n+Nat.div(x, 1n+sp), sp)

# Rounding x up to a positive step 1 + sp leaves a whole number of steps.
def g4v_align_up_whole(+x: Nat, +sp: Nat) -> {Nat.mod(TB.align_up(x, 1n+sp), 1n+sp) == 0n : Nat}:
  g4v_align_rem_whole(x, sp, Nat.mod(x, 1n+sp), {==}, L.nat_mod_lt(x, sp))

# LAW G4V.7c
# promise: the bytes planned for a buffer are whole steps of the arena, so the next buffer starts where the plan counts it.
# harm: the arena would move the next buffer to a step boundary past the plan's count, and run out before the plan's last buffer.
law g4v_part_whole_steps:
  for +b: Nat
  {Nat.mod(g4v_part_bytes(b), g4v_arena_step()) == 0n : Nat}

# The step, 256, is 1 + 255: the proof holds for every positive step written there.
def g4v_part_whole_steps(b):
  g4v_align_up_whole(Nat.max(b, g4v_arena_step()), 255n)

# LAW G4V.8
# promise: adding a buffer to a plan grows its total by at least the buffer's bytes.
law g4v_add_part_covers:
  for +t: Nat
  for +b: Nat
  {Nat.is_le(Nat.add(t, b), g4v_add_part(t, b)) == True{} : Bool}

def g4v_add_part_covers(t, b):
  L.nat_add_le_mono(t, t, b, g4v_part_bytes(b), L.nat_le_refl(t), L.nat_le_trans(b, Nat.max(b, g4v_arena_step()), g4v_part_bytes(b), L.nat_max_ge_l(b, g4v_arena_step()), TB.align_up_never_shrinks(Nat.max(b, g4v_arena_step()), g4v_arena_step())))

# LAW G4V.9
# promise: an empty width is never taken by the row kernels.
law g4v_empty_width_refused:
  {g4v_width_ok(0n) == False{} : Bool}

def g4v_empty_width_refused():
  {==}

# LAW G4V.10
# promise: an image path without a patch size is refused.
law g4v_zero_patch_refused:
  for +q: Nat
  for +n: Nat
  {g4v_grid_ok(0n, q, n) == False{} : Bool}

def g4v_zero_patch_refused(q, n):
  {==}

# LAW G4V.11
# promise: a tower without layers is refused.
law g4v_layerless_tower_refused:
  for +i: Nat
  {g4v_tower_ok(0n, i) == False{} : Bool}

def g4v_layerless_tower_refused(i):
  {==}

# LAW G4V.12
# promise: flooring a patch count to whole tokens never raises it.
law g4v_whole_token_patches_within:
  for +p: Nat
  for +kp: Nat
  {Nat.is_le(g4v_whole_token_patches(p, 1n+kp), p) == True{} : Bool}

def g4v_whole_token_patches_within(p, kp):
  g4v_div_mul_le(p, kp)

# 1 + n + k <= n never holds.
def g4v_past_not_le(+n: Nat, +k: Nat) -> {Nat.is_le(1n+Nat.add(n, k), n) == False{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+np:
      g4v_past_not_le(np, k)

# LAW G4V.13
# promise: a grid taller than the position table is refused.
# harm: the host would read position rows past the end of the table.
law g4v_table_refuses_tall_grid:
  for +n: Nat
  for +k: Nat
  for +w: Nat
  {g4v_table_holds(1n+Nat.add(n, k), w, n) == False{} : Bool}

def g4v_table_refuses_tall_grid(n, k, w):
  %Equal.sym(Bool, Nat.is_le(1n+Nat.add(n, k), n), False{}, g4v_past_not_le(n, k)) : {Bool.and(_, Nat.is_le(w, n)) == False{} : Bool}
  {==}

# LAW G4V.14
# promise: a grid wider than the position table is refused.
# harm: the host would read a column's position row from the table's rows axis: wrong positions and no error.
law g4v_table_refuses_wide_grid:
  for +h: Nat
  for +n: Nat
  for +k: Nat
  {g4v_table_holds(h, 1n+Nat.add(n, k), n) == False{} : Bool}

def g4v_table_refuses_wide_grid(h, n, k):
  %Equal.sym(Bool, Nat.is_le(1n+Nat.add(n, k), n), False{}, g4v_past_not_le(n, k)) : {Bool.and(Nat.is_le(h, n), _) == False{} : Bool}
  g4u_and_false_r(Nat.is_le(h, n))

# LAW G4V.15
# promise: flooring a patch count to whole tokens leaves a whole number of tokens.
# harm: the last token's pooling block would take patches of no token.
law g4v_whole_token_patches_whole:
  for +p: Nat
  for +kp: Nat
  {Nat.mod(g4v_whole_token_patches(p, 1n+kp), 1n+kp) == 0n : Nat}

def g4v_whole_token_patches_whole(p, kp):
  IK.ikp_mod_mul(Nat.div(p, 1n+kp), kp)

# LAW G4U.11
# promise: a row before an image block is not in it.
# harm: a text row before an image would attend into the image.
law g4u_block_row_before_outside:
  for +p: Nat
  for +k: Nat
  for +n: Nat
  {g4u_in_block(1n+Nat.add(p, k), n, p) == False{} : Bool}

def g4u_block_row_before_outside(p, k, n):
  %Equal.sym(Bool, Nat.is_le(1n+Nat.add(p, k), p), False{}, g4u_past_not_le(p, k)) : {Bool.and(_, Nat.is_lt(p, Nat.add(1n+Nat.add(p, k), n))) == False{} : Bool}
  {==}

# LAW G4U.12
# promise: an image block that starts at or after the end of some rows has no row among them.
# harm: a unit would count a later block as its own, and join or refuse tiles for no reason.
law g4u_block_after_rows_outside:
  for +b: Nat
  for +k: Nat
  for +n: Nat
  for +a: Nat
  {g4u_overlaps(Nat.add(b, k), n, a, b) == False{} : Bool}

def g4u_block_after_rows_outside(b, k, n, a):
  %Equal.sym(Bool, Nat.is_lt(Nat.add(b, k), b), False{}, g4u_add_lt_false(b, k)) : {Bool.and(_, Nat.is_lt(a, Nat.add(Nat.add(b, k), n))) == False{} : Bool}
  {==}

# x <= p + (x - p).
def g4u_le_add_sub(+p: Nat, +x: Nat) -> {Nat.is_le(x, Nat.add(p, Nat.sub(x, p))) == True{} : Bool}:
  match p:
    case 0n:
      match x:
        case 0n:
          {==}
        case 1n+xp:
          L.nat_le_refl(1n+xp)
    case 1n+pp:
      match x:
        case 0n:
          {==}
        case 1n+xp:
          g4u_le_add_sub(pp, xp)

# x <= 1 + p + (x - (1 + p)).
def g4u_le_succ_add_sub(+p: Nat, +x: Nat) -> {Nat.is_le(x, 1n+Nat.add(p, Nat.sub(x, 1n+p))) == True{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+xp:
      g4u_le_add_sub(p, xp)

# LAW G4U.13
# promise: a row of an image block attends at least to the end of its block, or to the end of the unit's keys when they end sooner.
# harm: the row would miss part of its image that the reference attends.
law g4u_ext_reaches_block_end:
  for +p: Nat
  for +e: Nat
  for +u: Nat
  {Nat.is_le(Nat.min(e, u), 1n+Nat.add(p, g4u_ext(True{}, p, e, u))) == True{} : Bool}

def g4u_ext_reaches_block_end(p, e, u):
  g4u_le_succ_add_sub(p, Nat.min(e, u))

# LAW G4U.14
# promise: a unit never takes a tile past the prompt's last tile.
# harm: the prefill would index a tile that does not exist.
law g4u_cross_within_tiles:
  for b: Bool
  for +n: Nat
  for +k: Nat
  for +be: Nat
  for +e: Nat
  {g4u_cross(b, Nat.add(n, k), n, be, e) == False{} : Bool}

def g4u_cross_within_tiles(b, n, k, be, e):
  match b:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, Nat.is_lt(Nat.add(n, k), n), False{}, g4u_add_lt_false(n, k)) : {Bool.and(_, Nat.is_lt(e, be)) == False{} : Bool}
      {==}

# x < 1 + x + k.
def g4u_lt_succ_add(+x: Nat, +k: Nat) -> {Nat.is_lt(x, 1n+Nat.add(x, k)) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_lt(x, 1n+Nat.add(x, k)), Nat.is_le(x, Nat.add(x, k)), True{}, L.nat_lt_succ_eq_le(x, Nat.add(x, k)), L.nat_add_le_r(x, k))

# LAW G4U.15
# promise: in a size whose image blocks attend both ways, a block that runs past a unit's last tile into a later tile always takes that tile into the unit.
# harm: an image over a tile boundary would be refused although the plan holds the tiles it needs.
law g4u_cross_into_block:
  for +i: Nat
  for +k: Nat
  for +e: Nat
  for +j: Nat
  {g4u_cross(True{}, i, 1n+Nat.add(i, k), 1n+Nat.add(e, j), e) == True{} : Bool}

def g4u_cross_into_block(i, k, e, j):
  %Equal.sym(Bool, Nat.is_lt(i, 1n+Nat.add(i, k)), True{}, g4u_lt_succ_add(i, k)) : {Bool.and(_, Nat.is_lt(e, 1n+Nat.add(e, j))) == True{} : Bool}
  g4u_lt_succ_add(e, j)

# x < 0 never holds.
def g4u_lt_zero_false(+x: Nat) -> {Nat.is_lt(x, 0n) == False{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+_:
      {==}

# LAW G4U.16
# promise: a unit that takes a later tile always runs the two-way form, whatever further blocks its other tiles hold.
# harm: the causal path prefills only a unit's first tile and moves past the rest, which would leave their rows unwritten.
law g4u_joined_unit_two_way:
  for b: Bool
  for +i: Nat
  for +n: Nat
  for +be: Nat
  for +e: Nat
  for +k: Nat
  {Bool.and(g4u_cross(b, i, n, be, e), Bool.not(g4u_two_way(b, Nat.add(be, k)))) == False{} : Bool}

def g4u_joined_unit_two_way(b, i, n, be, e, k):
  match b:
    case False{}:
      {==}
    case True{}:
      match be:
        case 0n:
          %Equal.sym(Bool, Nat.is_lt(e, 0n), False{}, g4u_lt_zero_false(e)) : {Bool.and(Bool.and(Nat.is_lt(i, n), _), Bool.not(Nat.is_lt(0n, k))) == False{} : Bool}
          %Equal.sym(Bool, Bool.and(Nat.is_lt(i, n), False{}), False{}, g4u_and_false_r(Nat.is_lt(i, n))) : {Bool.and(_, Bool.not(Nat.is_lt(0n, k))) == False{} : Bool}
          {==}
        case 1n+bp:
          g4u_and_false_r(Bool.and(Nat.is_lt(i, n), Nat.is_lt(e, 1n+bp)))

# LAW G4U.17
# promise: a size whose reference keeps image rows causal never runs a unit in the two-way form.
# harm: its image rows would attend positions after their own, unlike the reference.
law g4u_two_way_needs_bidir:
  for +be: Nat
  {g4u_two_way(False{}, be) == False{} : Bool}

def g4u_two_way_needs_bidir(be):
  {==}

# LAW G4U.18
# promise: in a size whose image blocks attend both ways, a unit with an image block always runs the two-way form.
# harm: its image rows would attend causally, unlike the reference.
law g4u_two_way_with_block:
  for +be: Nat
  {g4u_two_way(True{}, 1n+be) == True{} : Bool}

def g4u_two_way_with_block(be):
  {==}

# a and (n > 0), and n == 0, is false.
def g4v_processor_budget_go(a: Bool, +n: Nat) -> {Bool.and(Bool.and(a, Nat.is_gt(n, 0n)), Nat.is_eq(n, 0n)) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      match n:
        case 0n:
          {==}
        case 1n+_:
          {==}

# LAW G4V.16
# promise: a vision config the host processor takes has a pixel budget of at least one soft token (so a positive patch and pooling size).
# harm: every image would resize to no tokens or be refused, with a tower planned for none.
law g4v_processor_budget_holds_a_token:
  for +p: Nat
  for +m: Nat
  for +f: Nat
  for +c: Nat
  for +px: Nat
  {Bool.and(g4v_processor_ok(p, m, f, c, px), Nat.is_eq(g4v_soft_budget(px, Nat.mul(p, m)), 0n)) == False{} : Bool}

def g4v_processor_budget_holds_a_token(p, m, f, c, px):
  g4v_processor_budget_go(Bool.and(Nat.is_eq(f, 1n), Nat.is_eq(c, 3n)), g4v_soft_budget(px, Nat.mul(p, m)))

# ----------------------------------------------------------------------------
# Pooling order of the tower's patches
# ----------------------------------------------------------------------------

# go(q (1 + bp) + n, bp, d, 0) == go(n, bp, q + d, 0): whole multiples of the divisor only count.
def g4v_dm_mul_add(+q: Nat, +bp: Nat, +n: Nat, +d: Nat) -> {Nat.divmod.go(Nat.add(Nat.mul(q, 1n+bp), n), bp, d, 0n) == Nat.divmod.go(n, bp, Nat.add(q, d), 0n) : Nat & Nat}:
  match q:
    case 0n:
      {==}
    case 1n+qp:
      %Equal.sym(Nat, Nat.add(Nat.add(bp, Nat.mul(qp, 1n+bp)), n), Nat.add(bp, Nat.add(Nat.mul(qp, 1n+bp), n)), L.nat_add_assoc(bp, Nat.mul(qp, 1n+bp), n)) : {Nat.divmod.go(1n+_, bp, d, 0n) == Nat.divmod.go(n, bp, 1n+Nat.add(qp, d), 0n) : Nat & Nat}
      %L.nat_add_succ_r(bp, Nat.add(Nat.mul(qp, 1n+bp), n)) : {Nat.divmod.go(_, bp, d, 0n) == Nat.divmod.go(n, bp, 1n+Nat.add(qp, d), 0n) : Nat & Nat}
      %L.nat_add_zero_r(bp) : {Nat.divmod.go(Nat.add(bp, 1n+Nat.add(Nat.mul(qp, 1n+bp), n)), _, d, 0n) == Nat.divmod.go(n, bp, 1n+Nat.add(qp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(bp, 1n+Nat.add(Nat.mul(qp, 1n+bp), n)), Nat.add(bp, 0n), d, 0n), Nat.divmod.go(1n+Nat.add(Nat.mul(qp, 1n+bp), n), 0n, d, Nat.add(bp, 0n)), IK.ikp_dm_skip(bp, 1n+Nat.add(Nat.mul(qp, 1n+bp), n), 0n, d, 0n)) : {_ == Nat.divmod.go(n, bp, 1n+Nat.add(qp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat, Nat.add(bp, 0n), bp, L.nat_add_zero_r(bp)) : {Nat.divmod.go(Nat.add(Nat.mul(qp, 1n+bp), n), _, 1n+d, 0n) == Nat.divmod.go(n, bp, 1n+Nat.add(qp, d), 0n) : Nat & Nat}
      %L.nat_add_succ_r(qp, d) : {Nat.divmod.go(Nat.add(Nat.mul(qp, 1n+bp), n), bp, 1n+d, 0n) == Nat.divmod.go(n, bp, _, 0n) : Nat & Nat}
      g4v_dm_mul_add(qp, bp, n, 1n+d)

# go(r, bp, d, 0) == (d, r) when r <= bp: a remainder below the divisor stays one.
def g4v_dm_small(+r: Nat, +bp: Nat, +d: Nat, h: {Nat.is_le(r, bp) == True{} : Bool}) -> {Nat.divmod.go(r, bp, d, 0n) == (d, r) : Nat & Nat}:
  %L.nat_add_zero_r(r) : {Nat.divmod.go(_, bp, d, 0n) == (d, r) : Nat & Nat}
  %Equal.trans(Nat, Nat.add(r, Nat.sub(bp, r)), Nat.add(Nat.sub(bp, r), r), bp, L.nat_add_comm(r, Nat.sub(bp, r)), L.nat_sub_add_cancel(bp, r, h)) : {Nat.divmod.go(Nat.add(r, 0n), _, d, 0n) == (d, r) : Nat & Nat}
  %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(r, 0n), Nat.add(r, Nat.sub(bp, r)), d, 0n), Nat.divmod.go(0n, Nat.sub(bp, r), d, Nat.add(r, 0n)), IK.ikp_dm_skip(r, 0n, Nat.sub(bp, r), d, 0n)) : {_ == (d, r) : Nat & Nat}
  %Equal.sym(Nat, Nat.add(r, 0n), r, L.nat_add_zero_r(r)) : {(d, _) == (d, r) : Nat & Nat}
  {==}

# divmod(q (1 + bp) + r, 1 + bp) == (q, r) when r <= bp.
def g4v_divmod_qr(+q: Nat, +bp: Nat, +r: Nat, h: {Nat.is_le(r, bp) == True{} : Bool}) -> {Nat.divmod(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp) == (q, r) : Nat & Nat}:
  %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(Nat.mul(q, 1n+bp), r), bp, 0n, 0n), Nat.divmod.go(r, bp, Nat.add(q, 0n), 0n), g4v_dm_mul_add(q, bp, r, 0n)) : {_ == (q, r) : Nat & Nat}
  %Equal.sym(Nat, Nat.add(q, 0n), q, L.nat_add_zero_r(q)) : {Nat.divmod.go(r, bp, _, 0n) == (q, r) : Nat & Nat}
  g4v_dm_small(r, bp, q, h)

# (q (1 + bp) + r) / (1 + bp) == q when r <= bp.
def g4v_div_qr(+q: Nat, +bp: Nat, +r: Nat, h: {Nat.is_le(r, bp) == True{} : Bool}) -> {Nat.div(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp) == q : Nat}:
  %Equal.sym(Nat & Nat, Nat.divmod(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp), (q, r), g4v_divmod_qr(q, bp, r, h)) : {Pair.fst(Nat, Nat, _) == q : Nat}
  {==}

# (q (1 + bp) + r) % (1 + bp) == r when r <= bp.
def g4v_mod_qr(+q: Nat, +bp: Nat, +r: Nat, h: {Nat.is_le(r, bp) == True{} : Bool}) -> {Nat.mod(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp) == r : Nat}:
  %Equal.sym(Nat & Nat, Nat.divmod(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp), (q, r), g4v_divmod_qr(q, bp, r, h)) : {Pair.snd(Nat, Nat, _) == r : Nat}
  {==}

# a < 1 + b gives a <= b.
def g4v_lt_succ_le(+a: Nat, +b: Nat, h: {Nat.is_lt(a, 1n+b) == True{} : Bool}) -> {Nat.is_le(a, b) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_le(a, b), Nat.is_lt(a, 1n+b), True{}, Equal.sym(Bool, Nat.is_lt(a, 1n+b), Nat.is_le(a, b), L.nat_lt_succ_eq_le(a, b)), h)

# The offset inside a block, a % (1 + pp)^2, splits into a row below 1 + pp ...
def g4v_pool_dy_le(+a: Nat, +pp: Nat) -> {Nat.is_le(Nat.div(Nat.mod(a, Nat.mul(1n+pp, 1n+pp)), 1n+pp), pp) == True{} : Bool}:
  g4v_lt_succ_le(Nat.div(Nat.mod(a, Nat.mul(1n+pp, 1n+pp)), 1n+pp), pp, L.nat_div_lt_of_lt_mul(Nat.mod(a, Nat.mul(1n+pp, 1n+pp)), 1n+pp, pp, L.nat_mod_lt(a, Nat.add(pp, Nat.mul(pp, 1n+pp)))))

# ... and a column below 1 + pp.
def g4v_mod_le(+a: Nat, +pp: Nat) -> {Nat.is_le(Nat.mod(a, 1n+pp), pp) == True{} : Bool}:
  g4v_lt_succ_le(Nat.mod(a, 1n+pp), pp, L.nat_mod_lt(a, pp))

# p + x < p + y when x < y.
def g4v_lt_add_l(+p: Nat, +x: Nat, +y: Nat, h: {Nat.is_lt(x, y) == True{} : Bool}) -> {Nat.is_lt(Nat.add(p, x), Nat.add(p, y)) == True{} : Bool}:
  match p:
    case 0n:
      h
    case 1n+pp:
      g4v_lt_add_l(pp, x, y, h)

# a < n and b < p give a p + b < n p.
def g4v_lt_mul_add(+a: Nat, +b: Nat, +n: Nat, +p: Nat, ha: {Nat.is_lt(a, n) == True{} : Bool}, hb: {Nat.is_lt(b, p) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(a, p), b), Nat.mul(n, p)) == True{} : Bool}:
  match a n:
    case 0n 0n:
      L.bool_absurd({Nat.is_lt(Nat.add(Nat.mul(0n, p), b), Nat.mul(0n, p)) == True{} : Bool}, ha)
    case 0n 1n+np:
      L.nat_lt_le_trans(b, p, Nat.add(p, Nat.mul(np, p)), hb, L.nat_add_le_r(p, Nat.mul(np, p)))
    case 1n+ap 0n:
      L.bool_absurd({Nat.is_lt(Nat.add(Nat.mul(1n+ap, p), b), Nat.mul(0n, p)) == True{} : Bool}, ha)
    case 1n+ap 1n+np:
      %Equal.sym(Nat, Nat.add(Nat.add(p, Nat.mul(ap, p)), b), Nat.add(p, Nat.add(Nat.mul(ap, p), b)), L.nat_add_assoc(p, Nat.mul(ap, p), b)) : {Nat.is_lt(_, Nat.add(p, Nat.mul(np, p))) == True{} : Bool}
      g4v_lt_add_l(p, Nat.add(Nat.mul(ap, p), b), Nat.mul(np, p), g4v_lt_mul_add(ap, b, np, p, ha, hb))

# LAW G4V.17
# promise: every row of the pooling order holds a patch of a column inside the grid's whole blocks.
# harm: the host would read a patch's pixels and position row from outside the image.
law g4v_pool_x_within:
  for +i: Nat
  for +nbxp: Nat
  for +pp: Nat
  {Nat.is_lt(g4v_pool_x(i, 1n+nbxp, 1n+pp), Nat.mul(1n+nbxp, 1n+pp)) == True{} : Bool}

def g4v_pool_x_within(i, nbxp, pp):
  g4v_lt_mul_add(Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), Nat.mod(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), 1n+nbxp, 1n+pp, L.nat_mod_lt(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), nbxp), L.nat_mod_lt(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), pp))

# The y bound once the row index is below the count (b says it is).
def g4v_pool_y_within_go(+i: Nat, +nby: Nat, +nbxp: Nat, +pp: Nat, b: Bool, e: {Nat.is_lt(i, g4v_pool_count(nby, 1n+nbxp, 1n+pp)) == b : Bool}) -> {Bool.and(b, Nat.is_le(Nat.mul(nby, 1n+pp), g4v_pool_y(i, 1n+nbxp, 1n+pp))) == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      L.nat_lt_le_false(g4v_pool_y(i, 1n+nbxp, 1n+pp), Nat.mul(nby, 1n+pp), g4v_lt_mul_add(Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), Nat.div(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), nby, 1n+pp, L.nat_div_lt_of_lt_mul(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), nby, nbxp, L.nat_div_lt_of_lt_mul(i, Nat.mul(nby, 1n+nbxp), Nat.add(pp, Nat.mul(pp, 1n+pp)), e)), L.nat_div_lt_of_lt_mul(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp, pp, L.nat_mod_lt(i, Nat.add(pp, Nat.mul(pp, 1n+pp))))))

# LAW G4V.18
# promise: every row of the pooling order of whole blocks holds a patch of a row inside the grid's whole blocks.
# harm: the host would read a patch's pixels and position row from outside the image.
law g4v_pool_y_within:
  for +i: Nat
  for +nby: Nat
  for +nbxp: Nat
  for +pp: Nat
  {Bool.and(Nat.is_lt(i, g4v_pool_count(nby, 1n+nbxp, 1n+pp)), Nat.is_le(Nat.mul(nby, 1n+pp), g4v_pool_y(i, 1n+nbxp, 1n+pp))) == False{} : Bool}

def g4v_pool_y_within(i, nby, nbxp, pp):
  g4v_pool_y_within_go(i, nby, nbxp, pp, Nat.is_lt(i, g4v_pool_count(nby, 1n+nbxp, 1n+pp)), {==})

# LAW G4V.19
# promise: the rows of a soft token in the pooling order all lie in the token's own pooling block, the blocks in raster order (the reference's order of the soft tokens): its block column.
# harm: a token would pool patches of another block, or the tokens would leave the tower out of order.
law g4v_pool_x_in_token_block:
  for +i: Nat
  for +nbxp: Nat
  for +pp: Nat
  {Nat.div(g4v_pool_x(i, 1n+nbxp, 1n+pp), 1n+pp) == Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp) : Nat}

def g4v_pool_x_in_token_block(i, nbxp, pp):
  g4v_div_qr(Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), pp, Nat.mod(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), g4v_mod_le(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), pp))

# LAW G4V.20
# promise: the rows of a soft token in the pooling order all lie in the token's own pooling block, the blocks in raster order (the reference's order of the soft tokens): its block row.
# harm: a token would pool patches of another block, or the tokens would leave the tower out of order.
law g4v_pool_y_in_token_block:
  for +i: Nat
  for +nbx: Nat
  for +pp: Nat
  {Nat.div(g4v_pool_y(i, nbx, 1n+pp), 1n+pp) == Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), nbx) : Nat}

def g4v_pool_y_in_token_block(i, nbx, pp):
  g4v_div_qr(Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), nbx), pp, Nat.div(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), g4v_pool_dy_le(i, pp))

# LAW G4V.21
# promise: the pooling order holds no patch twice: the row a patch is read from is recovered from its grid position.
# harm: a patch would enter its token's mean twice and another patch not at all.
law g4v_pool_rows_distinct:
  for +i: Nat
  for +nbxp: Nat
  for +pp: Nat
  {g4v_pool_row(g4v_pool_y(i, 1n+nbxp, 1n+pp), g4v_pool_x(i, 1n+nbxp, 1n+pp), 1n+nbxp, 1n+pp) == i : Nat}

def g4v_pool_rows_distinct(i, nbxp, pp):
  %Equal.sym(Nat, Nat.div(g4v_pool_y(i, 1n+nbxp, 1n+pp), 1n+pp), Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), g4v_div_qr(Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), pp, Nat.div(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), g4v_pool_dy_le(i, pp))) : {Nat.add(Nat.mul(Nat.add(Nat.mul(_, 1n+nbxp), Nat.div(g4v_pool_x(i, 1n+nbxp, 1n+pp), 1n+pp)), Nat.mul(1n+pp, 1n+pp)), Nat.add(Nat.mul(Nat.mod(g4v_pool_y(i, 1n+nbxp, 1n+pp), 1n+pp), 1n+pp), Nat.mod(g4v_pool_x(i, 1n+nbxp, 1n+pp), 1n+pp))) == i : Nat}
  %Equal.sym(Nat, Nat.div(g4v_pool_x(i, 1n+nbxp, 1n+pp), 1n+pp), Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), g4v_div_qr(Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), pp, Nat.mod(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), g4v_mod_le(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), pp))) : {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), 1n+nbxp), _), Nat.mul(1n+pp, 1n+pp)), Nat.add(Nat.mul(Nat.mod(g4v_pool_y(i, 1n+nbxp, 1n+pp), 1n+pp), 1n+pp), Nat.mod(g4v_pool_x(i, 1n+nbxp, 1n+pp), 1n+pp))) == i : Nat}
  %Equal.sym(Nat, Nat.mod(g4v_pool_y(i, 1n+nbxp, 1n+pp), 1n+pp), Nat.div(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), g4v_mod_qr(Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), pp, Nat.div(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), g4v_pool_dy_le(i, pp))) : {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), 1n+nbxp), Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp)), Nat.mul(1n+pp, 1n+pp)), Nat.add(Nat.mul(_, 1n+pp), Nat.mod(g4v_pool_x(i, 1n+nbxp, 1n+pp), 1n+pp))) == i : Nat}
  %Equal.sym(Nat, Nat.mod(g4v_pool_x(i, 1n+nbxp, 1n+pp), 1n+pp), Nat.mod(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), g4v_mod_qr(Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), pp, Nat.mod(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), g4v_mod_le(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), pp))) : {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), 1n+nbxp), Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp)), Nat.mul(1n+pp, 1n+pp)), Nat.add(Nat.mul(Nat.div(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), 1n+pp), _)) == i : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp), 1n+nbxp), Nat.mod(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), 1n+nbxp)), Nat.div(i, Nat.mul(1n+pp, 1n+pp)), L.nat_div_mod_r(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), nbxp)) : {Nat.add(Nat.mul(_, Nat.mul(1n+pp, 1n+pp)), Nat.add(Nat.mul(Nat.div(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), 1n+pp), Nat.mod(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp))) == i : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp), 1n+pp), Nat.mod(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), 1n+pp)), Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), L.nat_div_mod_r(Nat.mod(i, Nat.mul(1n+pp, 1n+pp)), pp)) : {Nat.add(Nat.mul(Nat.div(i, Nat.mul(1n+pp, 1n+pp)), Nat.mul(1n+pp, 1n+pp)), _) == i : Nat}
  L.nat_div_mod_r(i, Nat.add(pp, Nat.mul(pp, 1n+pp)))
