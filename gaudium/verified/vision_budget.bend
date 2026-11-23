# Vision input admission, budgets and frame arithmetic (images, animations and video).
# Formally specifies and proves:
#   - Where a request's media may come from (inline bytes always, a remote URL only when allowed)
#   - Byte / canvas limits checked from a file's header, before anything is decoded
#   - The pixel budget of an image and of a video's frames, capped by the planned vision tower
#   - Embedding-row accounting of one request (rows used never exceed the planned buffer)
#   - Video frame sampling counts, sampled frame indices and temporal grouping (never out of range)
#   - Mirrored row / column indices of an orientation change (never out of range)
#   - Which speculative drafters a model may run beside its vision tower

import Base

# ============================================================================
# 1. Helper Nat / Bool lemmas (erased before Rust codegen)
# ============================================================================

def vb_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      vb_min_le_l(ap, bp)

def vb_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      vb_min_le_r(ap, bp)

# min(a, min(b, c)) <= b
def vb_min3_le_mid(+a: Nat, +b: Nat, +c: Nat) -> {Nat.is_le(Nat.min(a, Nat.min(b, c)), b) == True{} : Bool}:
  match a b c:
    case 0n 0n _: {==}
    case 0n 1n+_ 0n: {==}
    case 0n 1n+_ 1n+_: {==}
    case 1n+_ 0n _: {==}
    case 1n+_ 1n+_ 0n: {==}
    case 1n+ap 1n+bp 1n+cp:
      vb_min3_le_mid(ap, bp, cp)

def vb_le_refl(n: Nat) -> {True{} == Nat.is_le(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      vb_le_refl(p)

def vb_le_step_r_zero(xp: Nat, h: {False{} == True{} : Bool}) -> {Nat.is_le(xp, 0n) == True{} : Bool}:
  match xp:
    case 0n: {==}
    case 1n+_: h

def vb_le_step_r(x: Nat, y: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(x, 1n+y) == True{} : Bool}:
  match x y:
    case 0n _:
      {==}
    case 1n+xp 0n:
      vb_le_step_r_zero(xp, h)
    case 1n+xp 1n+yp:
      vb_le_step_r(xp, yp, h)

def vb_sub_le_self(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      Equal.sym(Bool, True{}, Nat.is_le(ap, ap), vb_le_refl(ap))
    case 1n+ap 1n+bp:
      vb_le_step_r(Nat.sub(ap, bp), ap, vb_sub_le_self(ap, bp))

# 1 + m + extra <= m never holds
def vb_over_not_le(m: Nat, extra: Nat) -> {Nat.is_le(Nat.add(1n+m, extra), m) == False{} : Bool}:
  match m:
    case 0n: {==}
    case 1n+p:
      vb_over_not_le(p, extra)

# ============================================================================
# 2. Media sources
# ============================================================================

# Where the bytes of an image, a frame or an animation come from.
type MediaSource is Data:
  MsInline{}
  MsRemote{}

# Inline bytes (base64, a data: URL) are always taken. Anything the server would have to fetch
# is taken only when the operator allowed it.
def media_source_allowed(s: MediaSource, allow_remote: Bool) -> Bool:
  match s:
    case MsInline{}:
      True{}
    case MsRemote{}:
      allow_remote

# The drafter a served model decodes with.
type VisDrafter is Data:
  VdNone{}
  VdDSpine{}
  VdBlock{}
  VdMtp{}

# Whether a model with this drafter may be loaded with its vision tower. Plain decode is the
# exactness reference and always may. A drafter may once it reads vision embeddings (never
# image-pad embeddings) at image positions AND has passed the vision regression on the card:
# today that is the DSpine drafter. The others are served text only until they have.
def vis_drafter_ok(d: VisDrafter) -> Bool:
  match d:
    case VdNone{}:
      True{}
    case VdDSpine{}:
      True{}
    case VdBlock{}:
      False{}
    case VdMtp{}:
      False{}

# ============================================================================
# 3. Limits checked before decoding
# ============================================================================

# An encoded file: not empty and within the byte limit.
def media_bytes_ok(+bytes: Nat, max_bytes: Nat) -> Bool:
  Bool.and(Nat.is_gt(bytes, 0n), Nat.is_le(bytes, max_bytes))

# A canvas read from a file header: both sides positive, the area within the pixel limit.
def media_canvas_ok(+w: Nat, +h: Nat, max_pixels: Nat) -> Bool:
  Bool.and(Bool.and(Nat.is_gt(w, 0n), Nat.is_gt(h, 0n)), Nat.is_le(Nat.mul(w, h), max_pixels))

# Media items (images + videos) of one request against the request limit.
def media_count_ok(count: Nat, max_items: Nat) -> Bool:
  Nat.is_le(count, max_items)

# ============================================================================
# 4. Pixel budgets under the planned vision tower
# ============================================================================

# Pixel cap of one image: the model's own cap, lowered to the largest image the tower holds.
def vis_pixel_cap(model_max_pixels: Nat, tower_patches: Nat, patch_pixels: Nat) -> Nat:
  Nat.min(model_max_pixels, Nat.mul(tower_patches, patch_pixels))

# Pixel budget of a whole video (all kept frames together): the model's own budget, lowered so
# that no single frame exceeds the per-frame cap.
def vid_pixel_budget(model_max_pixels: Nat, frames: Nat, frame_cap: Nat) -> Nat:
  Nat.min(model_max_pixels, Nat.mul(frames, frame_cap))

# Pixel budget of a whole video that keeps its tokens inside `rows_left` embedding rows:
# `kept` frames in `groups` temporal patches, one token per `cell_pixels` pixels of a patch
# (0 for a video without temporal patches).
def vid_rows_pixel_budget(rows_left: Nat, kept: Nat, groups: Nat, cell_pixels: Nat) -> Nat:
  Nat.div(Nat.mul(Nat.mul(rows_left, cell_pixels), kept), groups)

# A resized patch grid against the tower's largest image.
def vis_grid_fits(grid_h: Nat, grid_w: Nat, tower_patches: Nat) -> Bool:
  Nat.is_le(Nat.mul(grid_h, grid_w), tower_patches)

# Decoder tokens of a patch grid under an m x m merger (0 for a zero merge size).
def vis_grid_tokens(grid_h: Nat, grid_w: Nat, +merge: Nat) -> Nat:
  Nat.mul(Nat.div(grid_h, merge), Nat.div(grid_w, merge))

# ============================================================================
# 5. Embedding rows of one request
# ============================================================================

# Image tokens the server plans for per request: never more than the context can hold.
def vis_token_cap(configured: Nat, ctx_tokens: Nat) -> Nat:
  Nat.min(configured, ctx_tokens)

# `add` more rows fit behind `used` rows of a buffer of `cap` rows.
def vis_rows_fit(used: Nat, add: Nat, cap: Nat) -> Bool:
  Nat.is_le(Nat.add(used, add), cap)

# Rows in use after taking `add` (the caller refuses the request when they do not fit; the
# count itself never leaves the buffer).
def vis_rows_take(used: Nat, add: Nat, cap: Nat) -> Nat:
  Nat.min(Nat.add(used, add), cap)

# Rows still free.
def vis_rows_left(used: Nat, cap: Nat) -> Nat:
  Nat.sub(cap, used)

# ============================================================================
# 6. Video frames
# ============================================================================

# Frames kept from a video of `total` frames: the count its duration asks for (`by_rate`),
# raised to the model's minimum, then held to the model's maximum and to the frames there are.
def vid_keep_count(by_rate: Nat, min_frames: Nat, max_frames: Nat, total: Nat) -> Nat:
  Nat.min(max_frames, Nat.min(total, Nat.max(by_rate, min_frames)))

# A sampled frame index held inside the video.
def vid_frame_index(raw: Nat, total: Nat) -> Nat:
  Nat.min(raw, Nat.sub(total, 1n))

# Temporal patches of `frames` frames, `temporal` frames each (the last one padded).
def vid_groups(frames: Nat, +temporal: Nat) -> Nat:
  Nat.div(Nat.add(frames, Nat.sub(temporal, 1n)), temporal)

# The frame shown in place `slot` of temporal patch `group`: past the end, the last frame again.
def vid_group_frame(group: Nat, slot: Nat, temporal: Nat, frames: Nat) -> Nat:
  Nat.min(Nat.add(Nat.mul(group, temporal), slot), Nat.sub(frames, 1n))

# ============================================================================
# 7. Orientation
# ============================================================================

# Index `i` of an axis of `n` entries, mirrored.
def orient_mirror(i: Nat, n: Nat) -> Nat:
  Nat.sub(Nat.sub(n, 1n), i)

# ============================================================================
# 8. Laws
# ============================================================================

# LAW VB.1: inline media is taken whatever the remote setting is.
law media_inline_always_allowed:
  for a: Bool
  {media_source_allowed(MsInline{}, a) == True{} : Bool}

def media_inline_always_allowed(a):
  {==}

# LAW VB.2: remote media is exactly as allowed as the operator's setting.
law media_remote_follows_setting:
  for a: Bool
  {media_source_allowed(MsRemote{}, a) == a : Bool}

def media_remote_follows_setting(a):
  {==}

# LAW VB.23: plain decode (no drafter) always runs beside the vision tower.
law vis_plain_decode_always_ok:
  {vis_drafter_ok(VdNone{}) == True{} : Bool}

def vis_plain_decode_always_ok():
  {==}

# LAW VB.3: an empty file is refused.
law media_empty_refused:
  for m: Nat
  {media_bytes_ok(0n, m) == False{} : Bool}

def media_empty_refused(m):
  {==}

# LAW VB.4: a file above the byte limit is refused.
law media_oversized_refused:
  for +m: Nat
  for +extra: Nat
  {media_bytes_ok(Nat.add(1n+m, extra), m) == False{} : Bool}

def media_oversized_refused(m, extra):
  vb_over_not_le(m, extra)

# LAW VB.5: a canvas without width is refused.
law media_zero_width_refused:
  for h: Nat
  for m: Nat
  {media_canvas_ok(0n, h, m) == False{} : Bool}

def media_zero_width_refused(h, m):
  {==}

# LAW VB.6: a canvas without height is refused.
law media_zero_height_refused:
  for +w: Nat
  for m: Nat
  {media_canvas_ok(w, 0n, m) == False{} : Bool}

def media_zero_height_refused(w, m):
  match w:
    case 0n: {==}
    case 1n+_: {==}

# LAW VB.7: more items than the limit are refused.
law media_count_over_refused:
  for +m: Nat
  for +extra: Nat
  {media_count_ok(Nat.add(1n+m, extra), m) == False{} : Bool}

def media_count_over_refused(m, extra):
  vb_over_not_le(m, extra)

# LAW VB.8: an image's pixel cap never exceeds the model's own.
law vis_pixel_cap_le_model:
  for +mx: Nat
  for +tp: Nat
  for +ps: Nat
  {Nat.is_le(vis_pixel_cap(mx, tp, ps), mx) == True{} : Bool}

def vis_pixel_cap_le_model(mx, tp, ps):
  vb_min_le_l(mx, Nat.mul(tp, ps))

# LAW VB.9: an image's pixel cap never exceeds what the tower was planned for.
law vis_pixel_cap_le_tower:
  for +mx: Nat
  for +tp: Nat
  for +ps: Nat
  {Nat.is_le(vis_pixel_cap(mx, tp, ps), Nat.mul(tp, ps)) == True{} : Bool}

def vis_pixel_cap_le_tower(mx, tp, ps):
  vb_min_le_r(mx, Nat.mul(tp, ps))

# LAW VB.10: a video's pixel budget never exceeds the model's own.
law vid_pixel_budget_le_model:
  for +mx: Nat
  for +f: Nat
  for +cap: Nat
  {Nat.is_le(vid_pixel_budget(mx, f, cap), mx) == True{} : Bool}

def vid_pixel_budget_le_model(mx, f, cap):
  vb_min_le_l(mx, Nat.mul(f, cap))

# LAW VB.11: a video's pixel budget never exceeds its frames' caps together.
law vid_pixel_budget_le_frames:
  for +mx: Nat
  for +f: Nat
  for +cap: Nat
  {Nat.is_le(vid_pixel_budget(mx, f, cap), Nat.mul(f, cap)) == True{} : Bool}

def vid_pixel_budget_le_frames(mx, f, cap):
  vb_min_le_r(mx, Nat.mul(f, cap))

# LAW VB.22: a video without temporal patches gets no pixel budget (no division by zero).
law vid_rows_budget_zero_groups:
  for r: Nat
  for k: Nat
  for c: Nat
  {vid_rows_pixel_budget(r, k, 0n, c) == 0n : Nat}

def vid_rows_budget_zero_groups(r, k, c):
  {==}

# LAW VB.12: the planned image tokens of a request never exceed the context.
law vis_token_cap_le_ctx:
  for +c: Nat
  for +ctx: Nat
  {Nat.is_le(vis_token_cap(c, ctx), ctx) == True{} : Bool}

def vis_token_cap_le_ctx(c, ctx):
  vb_min_le_r(c, ctx)

# LAW VB.13: the rows in use never leave the buffer.
law vis_rows_take_le_cap:
  for +used: Nat
  for +add: Nat
  for +cap: Nat
  {Nat.is_le(vis_rows_take(used, add, cap), cap) == True{} : Bool}

def vis_rows_take_le_cap(used, add, cap):
  vb_min_le_r(Nat.add(used, add), cap)

# LAW VB.14: a full buffer takes no further row.
law vis_rows_full_refuses:
  for +cap: Nat
  for +extra: Nat
  {vis_rows_fit(cap, 1n+extra, cap) == False{} : Bool}

def vis_rows_full_refuses(cap, extra):
  match cap:
    case 0n: {==}
    case 1n+p:
      vis_rows_full_refuses(p, extra)

# LAW VB.15: the free rows never exceed the buffer.
law vis_rows_left_le_cap:
  for +used: Nat
  for +cap: Nat
  {Nat.is_le(vis_rows_left(used, cap), cap) == True{} : Bool}

def vis_rows_left_le_cap(used, cap):
  vb_sub_le_self(cap, used)

# LAW VB.16: never more frames are kept than the video has.
law vid_keep_le_total:
  for +r: Nat
  for +mn: Nat
  for +mx: Nat
  for +total: Nat
  {Nat.is_le(vid_keep_count(r, mn, mx, total), total) == True{} : Bool}

def vid_keep_le_total(r, mn, mx, total):
  vb_min3_le_mid(mx, total, Nat.max(r, mn))

# LAW VB.17: never more frames are kept than the model's maximum.
law vid_keep_le_max:
  for +r: Nat
  for +mn: Nat
  for +mx: Nat
  for +total: Nat
  {Nat.is_le(vid_keep_count(r, mn, mx, total), mx) == True{} : Bool}

def vid_keep_le_max(r, mn, mx, total):
  vb_min_le_l(mx, Nat.min(total, Nat.max(r, mn)))

# LAW VB.18: a sampled index stays inside a video that has frames.
law vid_frame_index_in_range:
  for +raw: Nat
  for +p: Nat
  {Nat.is_le(vid_frame_index(raw, 1n+p), p) == True{} : Bool}

def vid_frame_index_in_range(raw, p):
  match p:
    case 0n:
      vb_min_le_r(raw, 0n)
    case 1n+q:
      vb_min_le_r(raw, 1n+q)

# LAW VB.19: every place of every temporal patch shows a frame of the video.
law vid_group_frame_in_range:
  for +g: Nat
  for +s: Nat
  for +t: Nat
  for +p: Nat
  {Nat.is_le(vid_group_frame(g, s, t, 1n+p), p) == True{} : Bool}

def vid_group_frame_in_range(g, s, t, p):
  match p:
    case 0n:
      vb_min_le_r(Nat.add(Nat.mul(g, t), s), 0n)
    case 1n+q:
      vb_min_le_r(Nat.add(Nat.mul(g, t), s), 1n+q)

# LAW VB.20: a zero temporal size yields no temporal patch (no division by zero).
law vid_groups_zero_temporal:
  for f: Nat
  {vid_groups(f, 0n) == 0n : Nat}

def vid_groups_zero_temporal(f):
  {==}

# LAW VB.21: a mirrored index stays on its axis.
law orient_mirror_in_range:
  for +i: Nat
  for +p: Nat
  {Nat.is_le(orient_mirror(i, 1n+p), p) == True{} : Bool}

def orient_mirror_in_range(i, p):
  match p:
    case 0n:
      vb_sub_le_self(0n, i)
    case 1n+q:
      vb_sub_le_self(1n+q, i)
