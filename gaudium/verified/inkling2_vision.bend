# Inkling 2 image input: tile grid, tower sizes, feature buffer bounds and the row selection of
# the text prefill's embedding stage.
# Formally specifies and proves:
#   - The tile grid of an image covers every pixel and holds no tile that is only padding
#   - An image's token count is checked against a limit without overflow
#   - The tower's bucket (one per image size class) never leaves the plan and never shrinks
#     below the image it holds; the patches sit behind the class row
#   - Which keys of a bucketed sequence attend (the class token always, padding never)
#   - Feature rows written by the merger stay inside the planned feature buffer
#   - A prompt row reads either the zero row (text) or its own feature row (image), never past
#     the buffer, and a request whose placeholders do not match its features is refused

import Base
import nat_lemmas.bend as L
import vision_budget.bend as VB

# ============================================================================
# 1. Helper lemmas (erased before Rust codegen)
# ============================================================================

# x + e < x never holds.
def ikv_add_not_lt(+x: Nat, +e: Nat) -> {Nat.is_lt(Nat.add(x, e), x) == False{} : Bool}:
  L.nat_le_lt_false(Nat.add(x, e), x, L.nat_add_le_r(x, e))

# (1 + f + e) == f never holds.
def ikv_over_ne(+f: Nat, +e: Nat) -> {Nat.is_eq(Nat.add(1n+f, e), f) == False{} : Bool}:
  match f:
    case 0n: {==}
    case 1n+p:
      ikv_over_ne(p, e)

# f == (1 + f + e) never holds.
def ikv_under_ne(+f: Nat, +e: Nat) -> {Nat.is_eq(f, Nat.add(1n+f, e)) == False{} : Bool}:
  match f:
    case 0n: {==}
    case 1n+p:
      ikv_under_ne(p, e)

# a >= n and b >= n give min(a, b) >= n.
def ikv_min_ge(+n: Nat, +a: Nat, +b: Nat, ha: {Nat.is_le(n, a) == True{} : Bool}, hb: {Nat.is_le(n, b) == True{} : Bool}) -> {Nat.is_le(n, Nat.min(a, b)) == True{} : Bool}:
  match n a b:
    case 0n _ _:
      L.nat_le_zero(Nat.min(a, b))
    case 1n+np 0n _:
      L.bool_absurd({Nat.is_le(1n+np, Nat.min(0n, b)) == True{} : Bool}, ha)
    case 1n+np 1n+ap 0n:
      L.bool_absurd({Nat.is_le(1n+np, Nat.min(1n+ap, 0n)) == True{} : Bool}, hb)
    case 1n+np 1n+ap 1n+bp:
      ikv_min_ge(np, ap, bp, ha, hb)

# ============================================================================
# 2. Tile grid of an image
# ============================================================================

# Tiles along one axis from the quotient and remainder of the pixel count by the tile side: one
# more tile for a remainder (the padded part of the last tile).
def ikv_side_of(q: Nat, r: Nat) -> Nat:
  match r:
    case 0n:
      q
    case 1n+_:
      1n+q

# Tiles along an axis of `px` pixels with tiles of `tile` pixels: the axis rounded up to whole
# tiles (the image is padded at its bottom / right, never resized).
def ikv_side(+px: Nat, +tile: Nat) -> Nat:
  ikv_side_of(Nat.div(px, tile), Nat.mod(px, tile))

# An image of `rows x cols` tiles (one token per tile) against a limit of `max_tokens`. The area
# test is the overflow-free canvas test of the media check (rows <= max / cols, cols > 0).
def ikv_tokens_fit(+rows: Nat, +cols: Nat, max_tokens: Nat) -> Bool:
  VB.media_canvas_ok(cols, rows, max_tokens)

# Image tokens one image may hold under a tower planned for `max_patches` patches of which
# `ppt` make one token.
def ikv_token_cap(max_patches: Nat, ppt: Nat) -> Nat:
  Nat.div(max_patches, ppt)

# ============================================================================
# 3. Tower sequence of one image
# ============================================================================

# Bucket ladder from `b` upwards: b, 1.5 b, 2 b, 3 b, 4 b, ... (`fuel` steps at most; the
# picks are lazy, so the ladder stops at the first rung that holds the image).
def ikv_round_go(fuel: Nat, +need: Nat, +b: Nat) -> Nat:
  match fuel:
    case 0n:
      need
    case 1n+f:
      Bool.pick(Nat, Nat.is_le(need, b), b, Bool.pick(Nat, Nat.is_le(need, Nat.add(b, Nat.div(b, 2n))), Nat.add(b, Nat.div(b, 2n)), ikv_round_go(f, need, Nat.mul(b, 2n))))

# The ladder rung of an image of `need` tokens: the tower compiles one set of recipes per rung,
# so images of nearby sizes share them (padding rows are masked out of attention). A need near the
# top of u64 (only an operator flag can ask for one) is refused rather than doubled past it.
# rusty-bends: total
def ikv_round(+need: Nat) -> Nat:
  ikv_round_go(64n, need, 16n)

# Tokens the tower's recipes are built for when an image has `need` tokens, its ladder rung is
# `rung` (`ikv_round(need)`) and the tower holds at most `cap`: the rung, never below the image
# and never above the plan, whatever the ladder says.
def ikv_bucket(+need: Nat, rung: Nat, cap: Nat) -> Nat:
  Nat.min(Nat.max(need, rung), cap)

# Rows of a tower sequence of `tokens` image tokens of `ppt` patches each: the class row first,
# then the patches (refused on overflow: the plan's sizes come from operator flags).
# rusty-bends: total
def ikv_seq_rows(tokens: Nat, ppt: Nat) -> Nat:
  1n+Nat.mul(tokens, ppt)

# Whether key row `j` of a bucketed sequence takes part in attention when the image has `tokens`
# tokens: the class row and the image's patches do, the padding rows behind them do not.
def ikv_key_live(j: Nat, tokens: Nat, ppt: Nat) -> Bool:
  Nat.is_lt(j, ikv_seq_rows(tokens, ppt))

# ============================================================================
# 4. Feature buffer and the prefill's row selection
# ============================================================================

# Columns of the feature buffer: column 0 stays zero (what text rows read), then one column per
# image token of a request, then room for the padding columns the merger of the last image's
# bucket writes behind its tokens (refused on overflow: both counts come from operator flags).
# rusty-bends: total
def ikv_feat_cols(max_tokens: Nat, cap: Nat) -> Nat:
  Nat.add(1n+max_tokens, cap)

# The merger of a bucket of `b` tokens writes columns `off .. off + b` of a buffer of `cols`.
def ikv_feat_write_ok(off: Nat, b: Nat, cols: Nat) -> Bool:
  Nat.is_le(Nat.add(off, b), cols)

# Feature column a prompt row reads: an image row (the `rank`-th placeholder of the prompt) its
# own column, a text row the zero column.
def ikv_feat_row(is_img: Bool, rank: Nat) -> Nat:
  match is_img:
    case True{}:
      1n+rank
    case False{}:
      0n

# Column the first image's features start at: behind the text rows' column.
def ikv_feat_start() -> Nat:
  1n

# Column the next image's features start at when this image's start at `off` and it has `tokens`
# tokens (the padding columns of its bucket behind them are overwritten by the next image).
def ikv_feat_next(off: Nat, tokens: Nat) -> Nat:
  Nat.add(off, tokens)

# The check the shell runs before a merger writes: inside the buffer, and starting behind the
# column the text rows read (so a write can never cover it). Overflow-free: `off + b` is not formed.
def ikv_feat_write_safe(+off: Nat, +b: Nat, +cols: Nat) -> Bool:
  Bool.and(Nat.is_lt(ikv_feat_row(False{}, 0n), off), Bool.and(Nat.is_le(off, cols), Nat.is_le(b, Nat.sub(cols, off))))

# A request's prompt placeholders against its images' features: one row each, no more, no less.
def ikv_rows_match(placeholders: Nat, features: Nat) -> Bool:
  Nat.is_eq(placeholders, features)

# ============================================================================
# 5. Laws
# ============================================================================

# Coverage: with `q * t + r == px` and `r < t`, the tiles of `ikv_side_of(q, r)` hold px.
def ikv_cover(+q: Nat, +r: Nat, +t: Nat, +px: Nat, e: {Nat.add(Nat.mul(q, t), r) == px : Nat}, lt: {Nat.is_lt(r, t) == True{} : Bool}) -> {Nat.is_le(px, Nat.mul(ikv_side_of(q, r), t)) == True{} : Bool}:
  match r:
    case 0n:
      %e : {Nat.is_le(_, Nat.mul(q, t)) == True{} : Bool}
      %Equal.sym(Nat, Nat.add(Nat.mul(q, t), 0n), Nat.mul(q, t), L.nat_add_zero_r(Nat.mul(q, t))) : {Nat.is_le(_, Nat.mul(q, t)) == True{} : Bool}
      L.nat_le_refl(Nat.mul(q, t))
    case 1n+rp:
      %e : {Nat.is_le(_, Nat.add(t, Nat.mul(q, t))) == True{} : Bool}
      %L.nat_add_comm(Nat.mul(q, t), t) : {Nat.is_le(Nat.add(Nat.mul(q, t), 1n+rp), _) == True{} : Bool}
      L.nat_add_le_mono(Nat.mul(q, t), Nat.mul(q, t), 1n+rp, t, L.nat_le_refl(Nat.mul(q, t)), L.nat_lt_imp_le(1n+rp, t, lt))

# x < x + 1 + rp.
def ikv_lt_add_succ(+x: Nat, +rp: Nat) -> {Nat.is_lt(x, Nat.add(x, 1n+rp)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(x, 1n+rp), 1n+Nat.add(x, rp), L.nat_add_succ_r(x, rp)) : {Nat.is_lt(x, _) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(x, 1n+Nat.add(x, rp)), Nat.is_le(x, Nat.add(x, rp)), L.nat_lt_succ_eq_le(x, Nat.add(x, rp))) : {_ == True{} : Bool}
  L.nat_add_le_r(x, rp)

# No padding-only tile: with `q * t + r == px` and `r < t`, the tiles of `ikv_side_of(q, r)`
# reach less than one tile past px.
def ikv_tight(+q: Nat, +r: Nat, +t: Nat, +px: Nat, e: {Nat.add(Nat.mul(q, t), r) == px : Nat}, lt: {Nat.is_lt(r, t) == True{} : Bool}) -> {Nat.is_lt(Nat.mul(ikv_side_of(q, r), t), Nat.add(px, t)) == True{} : Bool}:
  match r:
    case 0n:
      %e : {Nat.is_lt(Nat.mul(q, t), Nat.add(_, t)) == True{} : Bool}
      %Equal.sym(Nat, Nat.add(Nat.mul(q, t), 0n), Nat.mul(q, t), L.nat_add_zero_r(Nat.mul(q, t))) : {Nat.is_lt(Nat.mul(q, t), Nat.add(_, t)) == True{} : Bool}
      %L.nat_add_comm(t, Nat.mul(q, t)) : {Nat.is_lt(Nat.mul(q, t), _) == True{} : Bool}
      L.nat_add_lt_le_mono(0n, t, Nat.mul(q, t), Nat.mul(q, t), lt, L.nat_le_refl(Nat.mul(q, t)))
    case 1n+rp:
      %e : {Nat.is_lt(Nat.add(t, Nat.mul(q, t)), Nat.add(_, t)) == True{} : Bool}
      %L.nat_add_comm(Nat.mul(q, t), t) : {Nat.is_lt(_, Nat.add(Nat.add(Nat.mul(q, t), 1n+rp), t)) == True{} : Bool}
      L.nat_add_lt_le_mono(Nat.mul(q, t), Nat.add(Nat.mul(q, t), 1n+rp), t, t, ikv_lt_add_succ(Nat.mul(q, t), rp), L.nat_le_refl(t))

# LAW IKV.1
# promise: the tiles along an image axis hold every pixel of that axis.
law ikv_side_covers:
  for +px: Nat
  for +tp: Nat
  {Nat.is_le(px, Nat.mul(ikv_side(px, 1n+tp), 1n+tp)) == True{} : Bool}

def ikv_side_covers(px, tp):
  ikv_cover(Nat.div(px, 1n+tp), Nat.mod(px, 1n+tp), 1n+tp, px, L.nat_div_mod_r(px, tp), L.nat_mod_lt(px, tp))

# LAW IKV.2
# promise: the tiles along an image axis reach less than one tile past its last pixel, so no token is padding only.
law ikv_side_tight:
  for +px: Nat
  for +tp: Nat
  {Nat.is_lt(Nat.mul(ikv_side(px, 1n+tp), 1n+tp), Nat.add(px, 1n+tp)) == True{} : Bool}

def ikv_side_tight(px, tp):
  ikv_tight(Nat.div(px, 1n+tp), Nat.mod(px, 1n+tp), 1n+tp, px, L.nat_div_mod_r(px, tp), L.nat_mod_lt(px, tp))

# LAW IKV.3
# promise: an image whose tile grid the token check accepts never has more tokens than the limit.
law ikv_tokens_within_limit:
  for +rows: Nat
  for +cols: Nat
  for +m: Nat
  {Bool.and(ikv_tokens_fit(rows, cols, m), Nat.is_lt(m, Nat.mul(cols, rows))) == False{} : Bool}

def ikv_tokens_within_limit(rows, cols, m):
  VB.media_canvas_within_limit(cols, rows, m)

# LAW IKV.4
# promise: an image with no tile column is refused.
law ikv_tokens_zero_cols_refused:
  for +rows: Nat
  for m: Nat
  {ikv_tokens_fit(rows, 0n, m) == False{} : Bool}

def ikv_tokens_zero_cols_refused(rows, m):
  {==}

# LAW IKV.5
# promise: the per-image token cap never needs more patches than the tower was planned for.
law ikv_token_cap_within_patches:
  for +mp: Nat
  for +pp: Nat
  {Nat.is_le(Nat.mul(ikv_token_cap(mp, 1n+pp), 1n+pp), mp) == True{} : Bool}

def ikv_token_cap_within_patches(mp, pp):
  L.nat_le_subst_r(Nat.mul(Nat.div(mp, 1n+pp), 1n+pp), Nat.add(Nat.mul(Nat.div(mp, 1n+pp), 1n+pp), Nat.mod(mp, 1n+pp)), mp, L.nat_div_mod_r(mp, pp), L.nat_add_le_r(Nat.mul(Nat.div(mp, 1n+pp), 1n+pp), Nat.mod(mp, 1n+pp)))

# LAW IKV.6
# promise: the tower's bucket for an image never exceeds the planned largest image.
law ikv_bucket_le_cap:
  for +need: Nat
  for +rung: Nat
  for +cap: Nat
  {Nat.is_le(ikv_bucket(need, rung, cap), cap) == True{} : Bool}

def ikv_bucket_le_cap(need, rung, cap):
  VB.vb_min_le_r(Nat.max(need, rung), cap)

# need <= cap gives need <= bucket(need, cap).
def ikv_bucket_ge(+need: Nat, +rung: Nat, +cap: Nat, x: Bool, ex: {Nat.is_le(need, cap) == x : Bool}) -> {Bool.and(x, Nat.is_lt(ikv_bucket(need, rung, cap), need)) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(ikv_bucket(need, rung, cap), need, ikv_min_ge(need, Nat.max(need, rung), cap, L.nat_max_ge_l(need, rung), ex))

# LAW IKV.7
# promise: an image the tower was planned for gets a bucket that holds all of its tokens.
law ikv_bucket_holds_image:
  for +need: Nat
  for +rung: Nat
  for +cap: Nat
  {Bool.and(Nat.is_le(need, cap), Nat.is_lt(ikv_bucket(need, rung, cap), need)) == False{} : Bool}

def ikv_bucket_holds_image(need, rung, cap):
  ikv_bucket_ge(need, rung, cap, Nat.is_le(need, cap), {==})

# LAW IKV.8
# promise: the patches of an image fit behind the class row of its sequence.
law ikv_patches_behind_class:
  for +t: Nat
  for +p: Nat
  {Nat.is_lt(Nat.mul(t, p), ikv_seq_rows(t, p)) == True{} : Bool}

def ikv_patches_behind_class(t, p):
  L.nat_lt_succ_self(Nat.mul(t, p))

# LAW IKV.10
# promise: every query attends the class row of its image.
law ikv_class_key_live:
  for t: Nat
  for p: Nat
  {ikv_key_live(0n, t, p) == True{} : Bool}

def ikv_class_key_live(t, p):
  {==}

# LAW IKV.11
# promise: no query attends a padding row behind the image's patches.
law ikv_padding_key_dead:
  for +t: Nat
  for +p: Nat
  for +e: Nat
  {ikv_key_live(Nat.add(ikv_seq_rows(t, p), e), t, p) == False{} : Bool}

def ikv_padding_key_dead(t, p, e):
  ikv_add_not_lt(Nat.mul(t, p), e)

# off <= 1 + mt and b <= cap give off + b <= feat_cols(mt, cap).
def ikv_write_in(+off: Nat, +b: Nat, +mt: Nat, +cap: Nat, x: Bool, ex: {Nat.is_le(off, 1n+mt) == x : Bool}, y: Bool, ey: {Nat.is_le(b, cap) == y : Bool}) -> {Bool.and(Bool.and(x, y), Bool.not(ikv_feat_write_ok(off, b, ikv_feat_cols(mt, cap)))) == False{} : Bool}:
  match x y:
    case False{} _:
      {==}
    case True{} False{}:
      {==}
    case True{} True{}:
      %Equal.sym(Bool, Nat.is_le(Nat.add(off, b), Nat.add(1n+mt, cap)), True{}, L.nat_add_le_mono(off, 1n+mt, b, cap, ex, ey)) : {Bool.and(True{}, Bool.not(_)) == False{} : Bool}
      {==}

# LAW IKV.12
# promise: the merger of an image that starts inside the request's columns, with a bucket the plan admits, never writes past the feature buffer.
law ikv_feat_write_in_buffer:
  for +off: Nat
  for +b: Nat
  for +mt: Nat
  for +cap: Nat
  {Bool.and(Bool.and(Nat.is_le(off, 1n+mt), Nat.is_le(b, cap)), Bool.not(ikv_feat_write_ok(off, b, ikv_feat_cols(mt, cap)))) == False{} : Bool}

def ikv_feat_write_in_buffer(off, b, mt, cap):
  ikv_write_in(off, b, mt, cap, Nat.is_le(off, 1n+mt), {==}, Nat.is_le(b, cap), {==})

# LAW IKV.13
# promise: a text row never reads a column an image row reads.
# harm: a text row would take an image's features as its embedding
law ikv_text_image_rows_apart:
  for +r1: Nat
  for +r2: Nat
  {Nat.is_eq(ikv_feat_row(True{}, r1), ikv_feat_row(False{}, r2)) == False{} : Bool}

def ikv_text_image_rows_apart(r1, r2):
  {==}

def ikv_row_in(+r: Nat, +mt: Nat, +cap: Nat, x: Bool, ex: {Nat.is_lt(r, mt) == x : Bool}) -> {Bool.and(x, Bool.not(Nat.is_lt(ikv_feat_row(True{}, r), ikv_feat_cols(mt, cap)))) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, Nat.is_lt(1n+r, Nat.add(1n+mt, cap)), True{}, L.nat_lt_le_trans(1n+r, 1n+mt, Nat.add(1n+mt, cap), ex, L.nat_add_le_r(1n+mt, cap))) : {Bool.and(True{}, Bool.not(_)) == False{} : Bool}
      {==}

# LAW IKV.14
# promise: an image row whose rank is inside the request's tokens reads a column inside the feature buffer.
# harm: the prefill's gather would read past the feature buffer
law ikv_image_row_in_range:
  for +r: Nat
  for +mt: Nat
  for +cap: Nat
  {Bool.and(Nat.is_lt(r, mt), Bool.not(Nat.is_lt(ikv_feat_row(True{}, r), ikv_feat_cols(mt, cap)))) == False{} : Bool}

def ikv_image_row_in_range(r, mt, cap):
  ikv_row_in(r, mt, cap, Nat.is_lt(r, mt), {==})

def ikv_and_l(a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      h

# A write that starts behind column 0 does not cover column 0.
def ikv_skip_zero(+off: Nat, +b: Nat, h: {Nat.is_lt(0n, off) == True{} : Bool}) -> {Bool.and(Nat.is_le(off, 0n), Nat.is_lt(0n, Nat.add(off, b))) == False{} : Bool}:
  match off:
    case 0n:
      L.bool_absurd({Bool.and(Nat.is_le(0n, 0n), Nat.is_lt(0n, Nat.add(0n, b))) == False{} : Bool}, h)
    case 1n+op:
      {==}

def ikv_write_skip(+off: Nat, +b: Nat, +cols: Nat, +r: Nat, v: Bool, hv: {ikv_feat_write_safe(off, b, cols) == v : Bool}) -> {Bool.and(v, Bool.and(Nat.is_le(off, ikv_feat_row(False{}, r)), Nat.is_lt(ikv_feat_row(False{}, r), Nat.add(off, b)))) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      ikv_skip_zero(off, b, ikv_and_l(Nat.is_lt(ikv_feat_row(False{}, 0n), off), Bool.and(Nat.is_le(off, cols), Nat.is_le(b, Nat.sub(cols, off))), hv))

# LAW IKV.15
# promise: a merger write the shell's check accepts never covers the column text rows read.
# harm: an image's features would overwrite the embedding every text row reads
law ikv_merge_skips_text_col:
  for +off: Nat
  for +b: Nat
  for +cols: Nat
  for +r: Nat
  {Bool.and(ikv_feat_write_safe(off, b, cols), Bool.and(Nat.is_le(off, ikv_feat_row(False{}, r)), Nat.is_lt(ikv_feat_row(False{}, r), Nat.add(off, b)))) == False{} : Bool}

def ikv_merge_skips_text_col(off, b, cols, r):
  ikv_write_skip(off, b, cols, r, ikv_feat_write_safe(off, b, cols), {==})

def ikv_and_r(a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case True{}:
      h
    case False{}:
      L.bool_absurd({b == True{} : Bool}, h)

# off <= cols gives off + (cols - off) == cols.
def ikv_off_rest(+off: Nat, +cols: Nat, h: {Nat.is_le(off, cols) == True{} : Bool}) -> {Nat.add(off, Nat.sub(cols, off)) == cols : Nat}:
  %L.nat_add_comm(Nat.sub(cols, off), off) : {_ == cols : Nat}
  L.nat_sub_add_cancel(cols, off, h)

# off <= cols and b <= cols - off give off + b <= cols.
def ikv_safe_in(+off: Nat, +b: Nat, +cols: Nat, +h: {Bool.and(Nat.is_le(off, cols), Nat.is_le(b, Nat.sub(cols, off))) == True{} : Bool}) -> {Nat.is_lt(cols, Nat.add(off, b)) == False{} : Bool}:
  L.nat_le_lt_false(cols, Nat.add(off, b), L.nat_le_subst_r(Nat.add(off, b), Nat.add(off, Nat.sub(cols, off)), cols, ikv_off_rest(off, cols, ikv_and_l(Nat.is_le(off, cols), Nat.is_le(b, Nat.sub(cols, off)), h)), L.nat_add_le_mono(off, off, b, Nat.sub(cols, off), L.nat_le_refl(off), ikv_and_r(Nat.is_le(off, cols), Nat.is_le(b, Nat.sub(cols, off)), h))))

def ikv_write_in_safe(+off: Nat, +b: Nat, +cols: Nat, v: Bool, hv: {ikv_feat_write_safe(off, b, cols) == v : Bool}) -> {Bool.and(v, Nat.is_lt(cols, Nat.add(off, b))) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      ikv_safe_in(off, b, cols, ikv_and_r(Nat.is_lt(ikv_feat_row(False{}, 0n), off), Bool.and(Nat.is_le(off, cols), Nat.is_le(b, Nat.sub(cols, off))), hv))

# LAW IKV.15a
# promise: a merger write the shell's check accepts ends inside the feature buffer.
# harm: the merger would write past the feature buffer into another tensor
law ikv_merge_write_in_buffer:
  for +off: Nat
  for +b: Nat
  for +cols: Nat
  {Bool.and(ikv_feat_write_safe(off, b, cols), Nat.is_lt(cols, Nat.add(off, b))) == False{} : Bool}

def ikv_merge_write_in_buffer(off, b, cols):
  ikv_write_in_safe(off, b, cols, ikv_feat_write_safe(off, b, cols), {==})

# LAW IKV.15b
# promise: the first image's features start behind the column text rows read.
# harm: the first merger write would be refused, or cover the text rows' column
law ikv_feat_start_past_text:
  for +r: Nat
  {Nat.is_lt(ikv_feat_row(False{}, r), ikv_feat_start()) == True{} : Bool}

def ikv_feat_start_past_text(r):
  {==}

# LAW IKV.15c
# promise: the next image's features start behind every column of the current image's tokens, so two images' tokens never share a column.
# harm: an image's features would overwrite the tokens of the image before it
law ikv_feat_next_past:
  for +off: Nat
  for +t: Nat
  {Nat.is_le(Nat.add(off, t), ikv_feat_next(off, t)) == True{} : Bool}

def ikv_feat_next_past(off, t):
  L.nat_le_refl(Nat.add(off, t))

# LAW IKV.16
# promise: a prompt with more image placeholders than its images have tokens is refused.
law ikv_extra_placeholders_refused:
  for +f: Nat
  for +e: Nat
  {ikv_rows_match(Nat.add(1n+f, e), f) == False{} : Bool}

def ikv_extra_placeholders_refused(f, e):
  ikv_over_ne(f, e)

# LAW IKV.17
# promise: a prompt with fewer image placeholders than its images have tokens is refused.
law ikv_missing_placeholders_refused:
  for +f: Nat
  for +e: Nat
  {ikv_rows_match(f, Nat.add(1n+f, e)) == False{} : Bool}

def ikv_missing_placeholders_refused(f, e):
  ikv_under_ne(f, e)
