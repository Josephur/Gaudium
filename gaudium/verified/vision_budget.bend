# Vision input admission, budgets and frame arithmetic (images, animations and video).
# Formally specifies and proves:
#   - Where a request's media may come from (inline bytes always, a remote URL only when allowed)
#   - Byte / canvas limits checked from a file's header, before anything is decoded
#   - The pixel budget of an image and of a video's frames, capped by the planned vision tower
#   - Embedding-row accounting of one request (rows used never exceed the planned buffer)
#   - Video frame sampling counts, sampled frame indices and temporal grouping (never out of range)
#   - Mirrored row / column indices of an orientation change (never out of range)
#   - Which speculative drafters a model may run beside its vision tower
#   - The image rows a prefill tile takes (never past the tile's rows, never past the image's)

import Base
import nat_lemmas.bend as L

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

# min(f, t) + (t - f) == t
def vb_min_add_sub(+f: Nat, +t: Nat) -> {Nat.add(Nat.min(f, t), Nat.sub(t, f)) == t : Nat}:
  match f t:
    case 0n 0n: {==}
    case 0n 1n+tp: {==}
    case 1n+fp 0n: {==}
    case 1n+fp 1n+tp:
      %vb_min_add_sub(fp, tp) : {1n+Nat.add(Nat.min(fp, tp), Nat.sub(tp, fp)) == 1n+_ : Nat}
      {==}

# c <= f: (min(f, t) - c) + (t - f) == t - c
def vb_min_sub_add_sub(+c: Nat, +f: Nat, +t: Nat, h: {Nat.is_le(c, f) == True{} : Bool}) -> {Nat.add(Nat.sub(Nat.min(f, t), c), Nat.sub(t, f)) == Nat.sub(t, c) : Nat}:
  match c f t:
    case 0n 0n _: {==}
    case 0n 1n+fp 0n: {==}
    case 0n 1n+fp 1n+tp:
      %vb_min_add_sub(fp, tp) : {1n+Nat.add(Nat.min(fp, tp), Nat.sub(tp, fp)) == 1n+_ : Nat}
      {==}
    case 1n+cp 0n _:
      L.bool_absurd({Nat.add(Nat.sub(Nat.min(0n, t), 1n+cp), Nat.sub(t, 0n)) == Nat.sub(t, 1n+cp) : Nat}, h)
    case 1n+cp 1n+fp 0n: {==}
    case 1n+cp 1n+fp 1n+tp:
      vb_min_sub_add_sub(cp, fp, tp, h)

# x <= y: x - c <= y - c
def vb_sub_le_mono(+x: Nat, +y: Nat, +c: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(Nat.sub(x, c), Nat.sub(y, c)) == True{} : Bool}:
  match x y c:
    case 0n _ _:
      L.nat_le_zero(Nat.sub(y, c))
    case 1n+xp 0n _:
      L.bool_absurd({Nat.is_le(Nat.sub(1n+xp, c), Nat.sub(0n, c)) == True{} : Bool}, h)
    case 1n+xp 1n+yp 0n:
      h
    case 1n+xp 1n+yp 1n+cp:
      vb_sub_le_mono(xp, yp, cp, h)

# (s + a) - (s + b) == a - b
def vb_add_sub_add_l(s: Nat, -a: Nat, -b: Nat) -> {Nat.sub(Nat.add(s, a), Nat.add(s, b)) == Nat.sub(a, b) : Nat}:
  match s:
    case 0n: {==}
    case 1n+sp:
      vb_add_sub_add_l(sp, a, b)

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
#   VdNone       plain decode (no drafter)
#   VdDSpine     a DSpine block drafter
#   VdBlock      a DFlash2 block drafter of a dense Qwen3.5 / 3.8 model
#   VdMtp        the own MTP head of a dense Qwen3.5 / 3.8 model
#   VdMtpMoe     the own MTP head of a mixture-of-experts Qwen3.5 / 3.6 model
#   VdStates     a block drafter that takes its context from the target model's hidden states
#                (Inkling 2's own drafter)
#   VdAssistant  a Gemma 4 model's own `-assistant` drafter checkpoint
#   VdOther      any other pairing of a drafter and a model family
type VisDrafter is Data:
  VdNone{}
  VdDSpine{}
  VdBlock{}
  VdMtp{}
  VdStates{}
  VdMtpMoe{}
  VdAssistant{}
  VdOther{}

# Whether a model with this drafter may be loaded with its vision tower. Plain decode is the
# exactness reference and always may. A drafter may once it reads vision embeddings (never
# image-pad embeddings) at image positions AND has passed the vision regression suite on the card
# (`gaudium --vision-suite`): today the DSpine and DFlash2 block drafters and the MTP head of a
# dense model, whose context is the target model's hidden states or which embeds the image's own
# rows (`engine/mtp.rs`), Inkling 2's own drafter (`VdStates`), and Gemma 4's `-assistant` drafter
# (`VdAssistant`: it embeds no prompt row, it reads the target's hidden state and keys only; token-
# identical to K=0 with images on the card, its suite rows are in the suite file). The
# mixture-of-experts MTP head is served text only until the suite has passed for it; a pairing this
# table does not know (a Gemma 4 block drafter among them) always is.
def vis_drafter_ok(d: VisDrafter) -> Bool:
  match d:
    case VdNone{}:
      True{}
    case VdDSpine{}:
      True{}
    case VdBlock{}:
      True{}
    case VdMtp{}:
      True{}
    case VdStates{}:
      True{}
    case VdMtpMoe{}:
      False{}
    case VdAssistant{}:
      True{}
    case VdOther{}:
      False{}

# ============================================================================
# 3. Limits checked before decoding
# ============================================================================

# An encoded file: not empty and within the byte limit.
def media_bytes_ok(+bytes: Nat, max_bytes: Nat) -> Bool:
  Bool.and(Nat.is_gt(bytes, 0n), Nat.is_le(bytes, max_bytes))

# A canvas read from a file header: both sides positive, the area within the pixel limit.
# The area test is h <= max_pixels / w (w > 0 here), the same as w * h <= max_pixels for whole
# numbers, written so that no header size can overflow it.
def media_canvas_ok(+w: Nat, +h: Nat, max_pixels: Nat) -> Bool:
  Bool.and(Bool.and(Nat.is_gt(w, 0n), Nat.is_gt(h, 0n)), Nat.is_le(h, Nat.div(max_pixels, w)))

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
# 8. Image rows of a prefill tile
# ============================================================================

# A prefill tile holds the prompt rows [lo, hi), and its row r embeds the token at slot
# r + shift: shift 0 for the decoder itself, shift 1 for a next-token head (the MTP fusion's
# row r embeds token r + 1). An image holds the slots [start, start + len) and one embedding row
# per slot. The tokens the tile embeds from the image form one run of slots [from, to), maybe
# empty; the rows below are where the run lands in the tile and in the image's embeddings.

# First slot of the run.
def vis_run_from(start: Nat, lo: Nat, shift: Nat) -> Nat:
  Nat.max(start, Nat.add(shift, lo))

# End of the run (one past its last slot).
def vis_run_to(start: Nat, len: Nat, hi: Nat, shift: Nat) -> Nat:
  Nat.min(Nat.add(start, len), Nat.add(shift, hi))

# Rows of the run (0: the tile takes nothing from this image).
def vis_run_rows(+start: Nat, len: Nat, lo: Nat, hi: Nat, +shift: Nat) -> Nat:
  Nat.sub(vis_run_to(start, len, hi, shift), vis_run_from(start, lo, shift))

# The run's first slot; an empty run sits at its end, so it never points past the tile.
def vis_run_first(+start: Nat, len: Nat, lo: Nat, hi: Nat, +shift: Nat) -> Nat:
  Nat.min(vis_run_from(start, lo, shift), vis_run_to(start, len, hi, shift))

# Tile row (0 = row lo) that embeds the run's first slot.
def vis_run_tile_row(+start: Nat, len: Nat, +lo: Nat, hi: Nat, +shift: Nat) -> Nat:
  Nat.sub(vis_run_first(start, len, lo, hi, shift), Nat.add(shift, lo))

# The run's first slot as a row of the image's own embeddings (0 = the image's first token).
def vis_run_emb_off(+start: Nat, len: Nat, lo: Nat, hi: Nat, shift: Nat) -> Nat:
  Nat.sub(vis_run_first(start, len, lo, hi, shift), start)

# ============================================================================
# 8a. Tower plan of a run's image requests
# ============================================================================

# Image tokens of one request: the tokens counted so far plus one more image's.
def vis_request_tokens(counted: Nat, image: Nat) -> Nat:
  Nat.add(counted, image)

# The tower of a run with several image requests is planned for its largest image and for the
# request holding the most image tokens: the plan so far, raised to one more request's or
# image's need.
def vis_plan_raise(planned: Nat, need: Nat) -> Nat:
  Nat.max(planned, need)

# ============================================================================
# 9. Laws
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

# An accepted height h <= m / (1 + wp) keeps h * (1 + wp) within m: it is at most
# (m / (1 + wp)) * (1 + wp), which with the remainder added is m itself.
def media_fit_within(+wp: Nat, +h: Nat, +m: Nat, b: Bool, e: {Nat.is_le(h, Nat.div(m, 1n+wp)) == b : Bool}) -> {Bool.and(b, Nat.is_lt(m, Nat.mul(1n+wp, h))) == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      %L.nat_mul_comm(h, 1n+wp) : {Bool.and(True{}, Nat.is_lt(m, _)) == False{} : Bool}
      L.nat_le_lt_false(m, Nat.mul(h, 1n+wp), L.nat_le_trans(Nat.mul(h, 1n+wp), Nat.mul(Nat.div(m, 1n+wp), 1n+wp), m, L.nat_mul_le_mono_r(h, Nat.div(m, 1n+wp), 1n+wp, e), L.nat_le_subst_r(Nat.mul(Nat.div(m, 1n+wp), 1n+wp), Nat.add(Nat.mul(Nat.div(m, 1n+wp), 1n+wp), Nat.mod(m, 1n+wp)), m, L.nat_div_mod_r(m, wp), L.nat_add_le_r(Nat.mul(Nat.div(m, 1n+wp), 1n+wp), Nat.mod(m, 1n+wp)))))

# LAW VB.6a: an accepted canvas never has more pixels than the limit.
# promise: a canvas the media check accepts never has more pixels (width times height) than the limit.
law media_canvas_within_limit:
  for +w: Nat
  for +h: Nat
  for +m: Nat
  {Bool.and(media_canvas_ok(w, h, m), Nat.is_lt(m, Nat.mul(w, h))) == False{} : Bool}

def media_canvas_within_limit(w, h, m):
  match w h:
    case 0n _:
      {==}
    case 1n+wp 0n:
      {==}
    case 1n+wp 1n+hp:
      media_fit_within(wp, 1n+hp, m, Nat.is_le(1n+hp, Nat.div(m, 1n+wp)), {==})

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

# LAW VB.24: a drafter pairing the table does not know never runs beside the vision tower.
# promise: a model whose drafter is not one the vision table knows is always served text only.
# harm: an unknown drafter could read image-pad embeddings and be served with images unchecked.
law vis_unknown_drafter_text_only:
  {vis_drafter_ok(VdOther{}) == False{} : Bool}

def vis_unknown_drafter_text_only():
  {==}

# LAW VB.25: the rows a tile takes from an image lie inside the tile.
# promise: the image rows a prefill tile takes never reach past the tile's own rows.
# harm: a device copy past the tile would overwrite rows of another buffer.
law vis_run_in_tile:
  for +start: Nat
  for +len: Nat
  for +lo: Nat
  for +hi: Nat
  for +shift: Nat
  {Nat.is_le(Nat.add(vis_run_tile_row(start, len, lo, hi, shift), vis_run_rows(start, len, lo, hi, shift)), Nat.sub(hi, lo)) == True{} : Bool}

def vis_run_in_tile(start, len, lo, hi, shift):
  +f = Nat.max(start, Nat.add(shift, lo))
  +t = Nat.min(Nat.add(start, len), Nat.add(shift, hi))
  +c = Nat.add(shift, lo)
  %Equal.sym(Nat, Nat.add(Nat.sub(Nat.min(f, t), c), Nat.sub(t, f)), Nat.sub(t, c), vb_min_sub_add_sub(c, f, t, L.nat_max_ge_r(start, c))) : {Nat.is_le(_, Nat.sub(hi, lo)) == True{} : Bool}
  L.nat_le_subst_r(Nat.sub(t, c), Nat.sub(Nat.add(shift, hi), c), Nat.sub(hi, lo), vb_add_sub_add_l(shift, hi, lo), vb_sub_le_mono(t, Nat.add(shift, hi), c, vb_min_le_r(Nat.add(start, len), Nat.add(shift, hi))))

# LAW VB.26: the rows a tile takes from an image lie inside the image's embeddings.
# promise: the image rows a prefill tile takes never reach past that image's own embedding rows.
# harm: a device copy past the image would read another image's (or stale) embedding rows.
law vis_run_in_image:
  for +start: Nat
  for +len: Nat
  for +lo: Nat
  for +hi: Nat
  for +shift: Nat
  {Nat.is_le(Nat.add(vis_run_emb_off(start, len, lo, hi, shift), vis_run_rows(start, len, lo, hi, shift)), len) == True{} : Bool}

def vis_run_in_image(start, len, lo, hi, shift):
  +f = Nat.max(start, Nat.add(shift, lo))
  +t = Nat.min(Nat.add(start, len), Nat.add(shift, hi))
  %Equal.sym(Nat, Nat.add(Nat.sub(Nat.min(f, t), start), Nat.sub(t, f)), Nat.sub(t, start), vb_min_sub_add_sub(start, f, t, L.nat_max_ge_l(start, Nat.add(shift, lo)))) : {Nat.is_le(_, len) == True{} : Bool}
  L.nat_le_subst_r(Nat.sub(t, start), Nat.sub(Nat.add(start, len), start), len, L.nat_add_sub_cancel_l(start, len), vb_sub_le_mono(t, Nat.add(start, len), start, vb_min_le_l(Nat.add(start, len), Nat.add(shift, hi))))

# LAW VB.27: a request's image tokens cover all of its images together, not only the largest.
# promise: a request's image-token count covers the tokens counted before plus the added image's, so it covers all of the request's images.
# harm: the embedding buffer could be planned for fewer rows than a multi-image request holds.
law vis_request_counts_all:
  for +c: Nat
  for +i: Nat
  {Nat.is_le(Nat.add(c, i), vis_request_tokens(c, i)) == True{} : Bool}

def vis_request_counts_all(c, i):
  L.nat_le_refl(Nat.add(c, i))

# LAW VB.28: adding an image never lowers a request's count.
# promise: counting one more image never lowers the image tokens already counted for the request.
law vis_request_count_grows:
  for +c: Nat
  for +i: Nat
  {Nat.is_le(c, vis_request_tokens(c, i)) == True{} : Bool}

def vis_request_count_grows(c, i):
  L.nat_add_le_r(c, i)

# LAW VB.29: the raised plan covers the request (or image) it was raised for.
# promise: the tower plan raised for a request's or an image's need always covers that need.
# harm: the tower or its embedding buffer could be planned too small for an image request of the run.
law vis_plan_covers_need:
  for +p: Nat
  for +n: Nat
  {Nat.is_le(n, vis_plan_raise(p, n)) == True{} : Bool}

def vis_plan_covers_need(p, n):
  L.nat_max_ge_r(p, n)

# LAW VB.30: raising the plan never shrinks what earlier requests were planned for.
# promise: raising the tower plan for one more need never shrinks what it covered before.
law vis_plan_keeps_earlier:
  for +p: Nat
  for +n: Nat
  {Nat.is_le(p, vis_plan_raise(p, n)) == True{} : Bool}

def vis_plan_keeps_earlier(p, n):
  L.nat_max_ge_l(p, n)

def vb_le_add_succ_false(+m: Nat, +d: Nat) -> {Nat.is_le(Nat.add(m, 1n+d), m) == False{} : Bool}:
  match m:
    case 0n: {==}
    case 1n+mp:
      vb_le_add_succ_false(mp, d)

# LAW VB.31: Embedding-row admission rejects any request that pushes used + add past capacity.
# promise: vis_rows_fit returns False whenever add exceeds cap - used
# catches: vis_rows_fit ignores add and only checks used <= cap
law vis_rows_fit_rejects_over:
  for +u: Nat
  for +a: Nat
  for +d: Nat
  {vis_rows_fit(u, Nat.add(a, 1n+d), Nat.add(u, a)) == False{} : Bool}

def vis_rows_fit_rejects_over(u, a, d):
  %L.nat_add_assoc(u, a, 1n+d) : {Nat.is_le(_, Nat.add(u, a)) == False{} : Bool}
  vb_le_add_succ_false(Nat.add(u, a), d)

def vb_not_le_imp_lt(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == False{} : Bool}) -> {Nat.is_lt(b, a) == True{} : Bool}:
  match a b:
    case 0n 0n:
      L.bool_absurd_tf({Nat.is_lt(0n, 0n) == True{} : Bool}, h)
    case 0n 1n+bp:
      L.bool_absurd_tf({Nat.is_lt(1n+bp, 0n) == True{} : Bool}, h)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      vb_not_le_imp_lt(ap, bp, h)

def vb_lt_succ_le_add1(+b: Nat, +a: Nat, h: {Nat.is_lt(b, a) == True{} : Bool}) -> {Nat.is_le(Nat.add(b, 1n), a) == True{} : Bool}:
  match b a:
    case 0n 0n:
      L.bool_absurd({Nat.is_le(1n, 0n) == True{} : Bool}, h)
    case 0n 1n+ap:
      L.nat_le_zero(ap)
    case 1n+bp 0n:
      L.bool_absurd({Nat.is_le(Nat.add(1n+bp, 1n), 0n) == True{} : Bool}, h)
    case 1n+bp 1n+ap:
      vb_lt_succ_le_add1(bp, ap, h)

def vb_ceil_le_r(
  +f: Nat,
  +tp: Nat,
  +q: Nat,
  eqq: {q == Nat.div(Nat.add(f, tp), 1n+tp) : Nat},
  hadd1: {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+tp), 1n), tp), Nat.add(f, tp)) == True{} : Bool}
) -> {Nat.is_le(Nat.add(Nat.mul(q, 1n+tp), 1n+tp), Nat.add(Nat.mul(q, 1n+tp), Nat.mod(Nat.add(f, tp), 1n+tp))) == True{} : Bool}:
  %L.nat_add_assoc(Nat.mul(q, 1n+tp), 1n, tp) : {Nat.is_le(_, Nat.add(Nat.mul(q, 1n+tp), Nat.mod(Nat.add(f, tp), 1n+tp))) == True{} : Bool}
  %Equal.sym(Nat, q, Nat.div(Nat.add(f, tp), 1n+tp), eqq) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+tp), 1n), tp), Nat.add(Nat.mul(_, 1n+tp), Nat.mod(Nat.add(f, tp), 1n+tp))) == True{} : Bool}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.add(f, tp), 1n+tp), 1n+tp), Nat.mod(Nat.add(f, tp), 1n+tp)), Nat.add(f, tp), L.nat_div_mod_r(Nat.add(f, tp), tp)) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+tp), 1n), tp), _) == True{} : Bool}
  hadd1

def vb_ceil_le_aux(
  +f: Nat,
  +tp: Nat,
  +q: Nat,
  b: Bool,
  eb: {b == Nat.is_le(f, Nat.mul(q, 1n+tp)) : Bool},
  eqq: {q == Nat.div(Nat.add(f, tp), 1n+tp) : Nat}
) -> {Nat.is_le(f, Nat.mul(q, 1n+tp)) == True{} : Bool}:
  match b:
    case True{}:
      Equal.sym(Bool, True{}, Nat.is_le(f, Nat.mul(q, 1n+tp)), eb)
    case False{}:
      +x = Nat.mul(q, 1n+tp)
      +hlt = vb_not_le_imp_lt(f, x, Equal.sym(Bool, False{}, Nat.is_le(f, x), eb))
      +hsle = vb_lt_succ_le_add1(x, f, hlt)
      +hadd1 = L.nat_add_le_mono(Nat.add(x, 1n), f, tp, tp, hsle, L.nat_le_refl(tp))
      +hle_bad = vb_ceil_le_r(f, tp, q, eqq, hadd1)
      +hlt_bad = L.nat_add_lt_l(x, Nat.mod(Nat.add(f, tp), 1n+tp), 1n+tp, L.nat_mod_lt(Nat.add(f, tp), tp))
      Empty.absurd({Nat.is_le(f, x) == True{} : Bool}, L.nat_lt_not_le(Nat.add(x, Nat.mod(Nat.add(f, tp), 1n+tp)), Nat.add(x, 1n+tp), hlt_bad, hle_bad))

# LAW VB.32: Temporal patch count multiplied by temporal stride always covers every frame.
# promise: vid_groups(f, 1+tp) * (1+tp) is always at least f
# catches: vid_groups floors frames / temporal without adding Nat.sub(temporal, 1n)
law vid_groups_covers_frames:
  for +f: Nat
  for +tp: Nat
  {Nat.is_le(f, Nat.mul(vid_groups(f, 1n+tp), 1n+tp)) == True{} : Bool}

def vid_groups_covers_frames(f, tp):
  %Equal.sym(Nat, Nat.sub(tp, 0n), tp, L.nat_sub_zero_r(tp)) : {Nat.is_le(f, Nat.mul(Nat.div(Nat.add(f, _), 1n+tp), 1n+tp)) == True{} : Bool}
  +q = Nat.div(Nat.add(f, tp), 1n+tp)
  vb_ceil_le_aux(f, tp, q, Nat.is_le(f, Nat.mul(q, 1n+tp)), {==}, {==})

# LAW VB.33: Mirroring an in-range axis index twice recovers the original index identically.
# promise: orient_mirror applied twice recovers any in-range index i < n
# catches: orient_mirror drops - i and returns Nat.sub(n, 1n)
law orient_mirror_involution:
  for +i: Nat
  for +d: Nat
  {orient_mirror(orient_mirror(i, Nat.add(1n+i, d)), Nat.add(1n+i, d)) == i : Nat}

def orient_mirror_involution(i, d):
  %Equal.sym(Nat, Nat.sub(Nat.add(i, d), 0n), Nat.add(i, d), L.nat_sub_zero_r(Nat.add(i, d))) : {Nat.sub(_, Nat.sub(_, i)) == i : Nat}
  %Equal.sym(Nat, Nat.sub(Nat.add(i, d), i), d, L.nat_add_sub_cancel_l(i, d)) : {Nat.sub(Nat.add(i, d), _) == i : Nat}
  L.nat_add_sub_cancel_r(i, d)
