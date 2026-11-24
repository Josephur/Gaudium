# Gemma 4 image input: the resized size of an image, its soft-token count against the budget the
# tower is planned for, and the prefill units that let an image block attend both ways in the
# sliding layers.
# Formally specifies and proves:
#   - An accepted resized size never holds more soft tokens than the budget
#   - A prefill unit (a tile and the later tiles its image blocks reach into) joins a tile only
#     while it still fits the sliding ring beside the window
#   - A unit that fits keeps every row's window and its block's extension inside the ring
#   - A prefill row of an image block never attends a key the unit has not written

import Base
import nat_lemmas.bend as L
import vision_budget.bend as VB

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
# 2b. Capacity of the image path (sizes the up-front plan)
# ============================================================================

# Soft tokens a pixel budget holds: one per `side x side` square.
def g4v_soft_budget(max_pixels: Nat, +side: Nat) -> Nat:
  Nat.div(max_pixels, Nat.mul(side, side))

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
