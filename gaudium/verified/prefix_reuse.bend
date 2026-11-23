# Prefix reuse across requests: a request whose prompt starts with tokens the engine still holds
# state for prefills only the rest.
#
# What the engine holds after a request: the positional K/V rows its prefill wrote, and, in a
# small number of device slots, the recurrent state of the model after some whole prefill tiles
# of that prompt (a "kept position"). A later request resumes from a kept position inside the
# prefix it shares with the held prompt.
#
# Formally specifies and proves:
#   - How far two prompts count as the same (token prefix, cut at the first image that differs)
#   - The life of a state slot (empty, being written, held) and what a new prompt keeps of it
#   - Which positions a running request may write into a slot
#   - Which slot takes a new position (a free one before any held one)
#   - Slot memory beside the rest of the device plan
#   - The first valid row of a drafter's context ring after a resumed request
#   - The split of a prompt into reused and prefilled rows
#
# The rule that says whether one kept position is a valid place to resume from is
# `pfx_reuse_valid` in `kv_pool.bend`. Positions are counted in whole prefill tiles here, so a
# kept position is a tile boundary by construction.

import Base

# ============================================================================
# 1. Helper Nat / Bool lemmas (erased before Rust codegen)
# ============================================================================

def pc_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      pc_min_le_l(ap, bp)

def pc_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      pc_min_le_r(ap, bp)

def pc_le_refl(n: Nat) -> {Nat.is_le(n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      pc_le_refl(p)

def pc_lt_refl(n: Nat) -> {Nat.is_lt(n, n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      pc_lt_refl(p)

def pc_gt_refl(n: Nat) -> {Nat.is_gt(n, n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      pc_gt_refl(p)

def pc_zero_le(n: Nat) -> {Nat.is_le(0n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+_: {==}

# a <= max(a, b)
def pc_le_max_l(+a: Nat, +b: Nat) -> {Nat.is_le(a, Nat.max(a, b)) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+ap 0n:
      pc_le_refl(ap)
    case 1n+ap 1n+bp:
      pc_le_max_l(ap, bp)

# b <= max(a, b)
def pc_le_max_r(+a: Nat, +b: Nat) -> {Nat.is_le(b, Nat.max(a, b)) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+bp:
      pc_le_refl(bp)
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      pc_le_max_r(ap, bp)

# a < b and b <= a never hold together
def pc_lt_le_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_lt(a, b), Nat.is_le(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      pc_lt_le_excl(ap, bp)

# a <= b and b < a never hold together
def pc_le_lt_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_le(a, b), Nat.is_lt(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      pc_le_lt_excl(ap, bp)

def pc_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def pc_or_true_r(b: Bool) -> {Bool.or(b, True{}) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def pc_le_step_r_zero(xp: Nat, h: {False{} == True{} : Bool}) -> {Nat.is_le(xp, 0n) == True{} : Bool}:
  match xp:
    case 0n: {==}
    case 1n+_: h

def pc_le_step_r(x: Nat, y: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(x, 1n+y) == True{} : Bool}:
  match x y:
    case 0n _:
      {==}
    case 1n+xp 0n:
      pc_le_step_r_zero(xp, h)
    case 1n+xp 1n+yp:
      pc_le_step_r(xp, yp, h)

def pc_sub_le_self(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      pc_le_refl(ap)
    case 1n+ap 1n+bp:
      pc_le_step_r(Nat.sub(ap, bp), ap, pc_sub_le_self(ap, bp))

# min(min(a, b), c) <= a
def pc_min3_le_a(+a: Nat, +b: Nat, +c: Nat) -> {Nat.is_le(Nat.min(Nat.min(a, b), c), a) == True{} : Bool}:
  match a b c:
    case 0n _ _: {==}
    case 1n+_ 0n _: {==}
    case 1n+_ 1n+_ 0n: {==}
    case 1n+ap 1n+bp 1n+cp:
      pc_min3_le_a(ap, bp, cp)

# min(min(a, b), c) <= b
def pc_min3_le_b(+a: Nat, +b: Nat, +c: Nat) -> {Nat.is_le(Nat.min(Nat.min(a, b), c), b) == True{} : Bool}:
  match a b c:
    case 0n 0n _: {==}
    case 0n 1n+_ _: {==}
    case 1n+_ 0n _: {==}
    case 1n+_ 1n+_ 0n: {==}
    case 1n+ap 1n+bp 1n+cp:
      pc_min3_le_b(ap, bp, cp)

# (a && x) && y == False when x && y == False
def pc_and3_excl(a: Bool, x: Bool, y: Bool, h: {Bool.and(x, y) == False{} : Bool}) -> {Bool.and(Bool.and(a, x), y) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      h

# (x && a) && y == False when x && y == False
def pc_and3_excl_l(x: Bool, a: Bool, y: Bool, h: {Bool.and(x, y) == False{} : Bool}) -> {Bool.and(Bool.and(x, a), y) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      match a:
        case False{}:
          {==}
        case True{}:
          h

# min(r, p) + (p - r) == p
def pc_split_sum(+p: Nat, +r: Nat) -> {Nat.add(Nat.min(r, p), Nat.sub(p, r)) == p : Nat}:
  match p r:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+pp 1n+rp:
      %pc_split_sum(pp, rp) : {1n+Nat.add(Nat.min(rp, pp), Nat.sub(pp, rp)) == 1n+_ : Nat}
      {==}

# ============================================================================
# 2. How far two prompts are the same
# ============================================================================

# Two image runs at the same place of two prompts hold the same picture: they start at the same
# slot, have the same token grid, and the pixels the vision tower reads are the same.
def pc_image_same(same_start: Bool, same_grid: Bool, same_pixels: Bool) -> Bool:
  Bool.and(same_start, Bool.and(same_grid, same_pixels))

# Slot limit of reuse after looking at one pair of image runs (the k-th of the held prompt and
# the k-th of the new one), given `later`, the limit the pairs after it leave: a pair that
# differs stops reuse before the earlier of its two runs.
def pc_image_limit(same: Bool, start_held: Nat, start_new: Nat, later: Nat) -> Nat:
  match same:
    case True{}:
      later
    case False{}:
      Nat.min(Nat.min(start_held, start_new), later)

# Rows of the new prompt that count as the same as the held one: the common token prefix, cut
# at the image limit (the image-pad tokens of two different pictures are the same token ids).
def pc_cap(lcp_tokens: Nat, image_limit: Nat) -> Nat:
  Nat.min(lcp_tokens, image_limit)

# May this request resume from held state at all? Not when it asked for a cold prefill, not
# when the K/V rows were zeroed since they were written (a reset after a failed request, or a
# reset mode that zeroes them every time), not on an engine that cannot.
def pc_request_may_reuse(wanted: Bool, rows_zeroed: Bool, supported: Bool) -> Bool:
  Bool.and(wanted, Bool.and(Bool.not(rows_zeroed), supported))

# ============================================================================
# 3. State slots
# ============================================================================

# One device slot for the recurrent state of the model after `tiles` whole prefill tiles.
type PcSlot is Data:
  PcEmpty{}
  PcWriting{tiles: Nat}
  PcHeld{tiles: Nat}

# The running request starts to write the state after `tiles` tiles into the slot.
def pc_begin_write(s: PcSlot, tiles: Nat) -> PcSlot:
  PcWriting{tiles}

# The tile that wrote the slot has finished on the device.
def pc_commit(s: PcSlot) -> PcSlot:
  match s:
    case PcEmpty{}:
      PcEmpty{}
    case PcWriting{t}:
      PcHeld{t}
    case PcHeld{t}:
      PcHeld{t}

# The request ended before the tile that writes the slot finished.
def pc_abort(s: PcSlot) -> PcSlot:
  match s:
    case PcEmpty{}:
      PcEmpty{}
    case PcWriting{t}:
      PcEmpty{}
    case PcHeld{t}:
      PcHeld{t}

# A request may resume from the slot.
def pc_readable(s: PcSlot) -> Bool:
  match s:
    case PcEmpty{}:
      False{}
    case PcWriting{t}:
      False{}
    case PcHeld{t}:
      True{}

# The slot holds nothing a new position would replace.
def pc_is_free(s: PcSlot) -> Bool:
  match s:
    case PcEmpty{}:
      True{}
    case PcWriting{t}:
      False{}
    case PcHeld{t}:
      False{}

# Whole tiles the slot's state stands after (0 for a slot that holds none).
def pc_slot_tiles(s: PcSlot) -> Nat:
  match s:
    case PcEmpty{}:
      0n
    case PcWriting{t}:
      t
    case PcHeld{t}:
      t

# Row position after `tiles` whole tiles of `tile` rows.
def pc_boundary(tiles: Nat, tile: Nat) -> Nat:
  Nat.mul(tiles, tile)

def pc_retain_if(t: Nat, ok: Bool) -> PcSlot:
  match ok:
    case True{}:
      PcHeld{t}
    case False{}:
      PcEmpty{}

# What a slot is worth for a new prompt that shares `cap` rows with the held one: a held
# position inside those rows stays, everything else is dropped (the rows behind it are about to
# be overwritten). A slot left in `PcWriting` by a request that never finished it is dropped.
def pc_retain(s: PcSlot, cap: Nat, tile: Nat) -> PcSlot:
  match s:
    case PcEmpty{}:
      PcEmpty{}
    case PcWriting{t}:
      PcEmpty{}
    case PcHeld{+t}:
      pc_retain_if(t, Nat.is_le(pc_boundary(t, tile), cap))

# ============================================================================
# 4. Which positions a request writes
# ============================================================================

# May a request that resumes after `resume_tiles` tiles of a prompt of `prompt_len` rows keep
# the state after `tiles` tiles? Only a state the request computes itself (past its resume
# point) at the end of a whole tile that is not the end of the prompt: the state at the end of
# the prompt is the one decode continues from, and it is not a prefill tile's state when the
# prompt ends inside a tile.
def pc_capture_ok(+tiles: Nat, resume_tiles: Nat, prompt_len: Nat, tile: Nat) -> Bool:
  Bool.and(Nat.is_lt(resume_tiles, tiles), Nat.is_lt(pc_boundary(tiles, tile), prompt_len))

# Rows at the end of a prompt that the next turn's prompt may render differently (the opening
# of the answer): the position a request keeps first lies before them. Measured with the chat
# template of the served model: the next turn shares the whole prompt, or all but its last row
# when the client does not send the reasoning back.
def pc_tail_guard() -> Nat:
  8n

# Tiles of the first position a request wants to keep: the last tile boundary an appended
# turn can still share.
def pc_anchor_tiles(prompt_len: Nat, +tile: Nat) -> Nat:
  Nat.div(Nat.sub(prompt_len, Nat.add(pc_tail_guard(), 1n)), tile)

# Tiles the `rank`-th wanted position lies before the anchor: 0, 1, 3, 7, ... A request that
# prefills a long prompt keeps a thinning trail behind it, so that a later prompt that leaves
# the held one anywhere finds a kept position at most about as far back again.
def pc_ladder_back(rank: Nat) -> Nat:
  match rank:
    case 0n:
      0n
    case 1n+p:
      1n+Nat.mul(2n, pc_ladder_back(p))

# Tiles of the `rank`-th wanted position.
def pc_ladder_tiles(anchor: Nat, rank: Nat) -> Nat:
  Nat.sub(anchor, pc_ladder_back(rank))

# ============================================================================
# 5. Which slot takes a new position
# ============================================================================

# How little a kept position is worth: its rank on the ladder of the request that wrote it,
# plus the requests since it was written or last resumed from.
def pc_cost(rank: Nat, seq_now: Nat, seq_used: Nat) -> Nat:
  Nat.add(rank, Nat.sub(seq_now, seq_used))

# Victim class of a slot for a new position: 0 = free, 1 = holds a position, 2 = holds the
# position the running request resumes from, 3 = not available (being written, or the resume
# slot when the tile that would write it is the tile that reads it).
def pc_victim_class_held(is_resume: Bool, same_tile: Bool) -> Nat:
  match is_resume:
    case False{}:
      1n
    case True{}:
      match same_tile:
        case True{}:
          3n
        case False{}:
          2n

def pc_victim_class(s: PcSlot, is_resume: Bool, same_tile: Bool) -> Nat:
  match s:
    case PcEmpty{}:
      0n
    case PcWriting{t}:
      3n
    case PcHeld{t}:
      pc_victim_class_held(is_resume, same_tile)

# May a slot of this class be written at all?
def pc_class_usable(class: Nat) -> Bool:
  Nat.is_lt(class, 3n)

# Is slot a a better place for a new position than slot b? A lower class first, and among
# slots of one class the one whose position is worth less.
def pc_victim_better(+a_class: Nat, +a_cost: Nat, +b_class: Nat, +b_cost: Nat) -> Bool:
  Bool.or(Nat.is_lt(a_class, b_class), Bool.and(Nat.is_eq(a_class, b_class), Nat.is_gt(a_cost, b_cost)))

# May a new position worth `new_cost` take a slot of this class and cost? A free slot always;
# a held position only when the new one is worth more.
def pc_may_take(new_cost: Nat, class: Nat, cost: Nat) -> Bool:
  match class:
    case 0n:
      True{}
    case 1n+c:
      Bool.and(pc_class_usable(1n+c), Nat.is_lt(new_cost, cost))

# ============================================================================
# 6. Slot memory beside the rest of the device plan
# ============================================================================

# Do `slots` slots of `slot_bytes` fit beside `base` planned bytes in `usable` bytes?
def pc_plan_fits(slots: Nat, slot_bytes: Nat, base: Nat, usable: Nat) -> Bool:
  Nat.is_le(Nat.add(base, Nat.mul(slots, slot_bytes)), usable)

def pc_grant_pick(slots: Nat, ok: Bool) -> Nat:
  match ok:
    case True{}:
      slots
    case False{}:
      0n

# The slots a load allocates, given the count the budget rule granted: that count when it fits
# beside the rest of the plan, none otherwise. The context is planned first; slots only ever
# take what is left.
def pc_grant_checked(+slots: Nat, +slot_bytes: Nat, +base: Nat, +usable: Nat) -> Nat:
  pc_grant_pick(slots, pc_plan_fits(slots, slot_bytes, base, usable))

# ============================================================================
# 7. A drafter's context ring
# ============================================================================
#
# A drafter that looks back a bounded window keeps its context rows in a ring (slot = position
# mod ring rows). A row is valid while the ring slot still holds that position's row.

# First position a resumed request's drafter may read: rows from `prev_from` on were valid
# before the request, rows from `resume` on are written again by its prefill.
def pc_ring_flo(prev_from: Nat, resume: Nat) -> Nat:
  Nat.min(prev_from, resume)

# First valid position after a request that resumed at `resume` and wrote rows up to
# `written_end` (exclusive) into a ring of `ring_rows` rows: writing position p overwrites
# position p - ring_rows.
def pc_ring_from(prev_from: Nat, resume: Nat, written_end: Nat, ring_rows: Nat) -> Nat:
  Nat.max(pc_ring_flo(prev_from, resume), Nat.sub(written_end, ring_rows))

# ============================================================================
# 8. Reused and prefilled rows of a prompt
# ============================================================================

def pc_reused_rows(prompt_len: Nat, resume: Nat) -> Nat:
  Nat.min(resume, prompt_len)

def pc_prefill_rows(prompt_len: Nat, resume: Nat) -> Nat:
  Nat.sub(prompt_len, resume)

# ============================================================================
# 9. Laws
# ============================================================================

# LAW PC.1: the rows two prompts share never exceed their common token prefix.
law pc_cap_within_tokens:
  for +l: Nat
  for +i: Nat
  {Nat.is_le(pc_cap(l, i), l) == True{} : Bool}

def pc_cap_within_tokens(l, i):
  pc_min_le_l(l, i)

# LAW PC.2: nor the image limit: reuse never reaches into or past an image that differs.
law pc_cap_within_images:
  for +l: Nat
  for +i: Nat
  {Nat.is_le(pc_cap(l, i), i) == True{} : Bool}

def pc_cap_within_images(l, i):
  pc_min_le_r(l, i)

# LAW PC.3: two image runs with different pixels are never the same picture.
law pc_image_other_pixels_differ:
  for +s: Bool
  for +g: Bool
  {pc_image_same(s, g, False{}) == False{} : Bool}

def pc_image_other_pixels_differ(s, g):
  match s:
    case False{}: {==}
    case True{}:
      pc_and_false_r(g)

# LAW PC.4: a pair of image runs that differ stops reuse at or before the held run ...
law pc_image_limit_before_held:
  for +a: Nat
  for +b: Nat
  for +later: Nat
  {Nat.is_le(pc_image_limit(False{}, a, b, later), a) == True{} : Bool}

def pc_image_limit_before_held(a, b, later):
  pc_min3_le_a(a, b, later)

# LAW PC.5: ... and at or before the new run.
law pc_image_limit_before_new:
  for +a: Nat
  for +b: Nat
  for +later: Nat
  {Nat.is_le(pc_image_limit(False{}, a, b, later), b) == True{} : Bool}

def pc_image_limit_before_new(a, b, later):
  pc_min3_le_b(a, b, later)

# LAW PC.6: looking at one more pair never moves the limit past what the later pairs left.
law pc_image_limit_within_later:
  for +s: Bool
  for +a: Nat
  for +b: Nat
  for +later: Nat
  {Nat.is_le(pc_image_limit(s, a, b, later), later) == True{} : Bool}

def pc_image_limit_within_later(s, a, b, later):
  match s:
    case True{}:
      pc_le_refl(later)
    case False{}:
      pc_min_le_r(Nat.min(a, b), later)

# LAW PC.7: a request that asked for a cold prefill never resumes from held state.
law pc_cold_request_never_reuses:
  for +z: Bool
  for +s: Bool
  {pc_request_may_reuse(False{}, z, s) == False{} : Bool}

def pc_cold_request_never_reuses(z, s):
  {==}

# LAW PC.8: no request resumes over K/V rows that were zeroed since they were written.
law pc_zeroed_rows_never_reused:
  for +w: Bool
  for +s: Bool
  {pc_request_may_reuse(w, True{}, s) == False{} : Bool}

def pc_zeroed_rows_never_reused(w, s):
  match w:
    case False{}: {==}
    case True{}: {==}

# LAW PC.9: an engine that cannot resume a prefill never does.
law pc_unsupported_never_reuses:
  for +w: Bool
  for +z: Bool
  {pc_request_may_reuse(w, z, False{}) == False{} : Bool}

def pc_unsupported_never_reuses(w, z):
  match w:
    case False{}: {==}
    case True{}:
      match z:
        case False{}: {==}
        case True{}: {==}

# LAW PC.10: a slot that is being written is never one a request may resume from.
law pc_writing_not_readable:
  for +s: PcSlot
  for +t: Nat
  {pc_readable(pc_begin_write(s, t)) == False{} : Bool}

def pc_writing_not_readable(s, t):
  {==}

# LAW PC.11: a write that did not finish leaves the slot empty, whatever it held before.
law pc_abort_leaves_nothing:
  for +s: PcSlot
  for +t: Nat
  {pc_abort(pc_begin_write(s, t)) == PcEmpty{} : PcSlot}

def pc_abort_leaves_nothing(s, t):
  {==}

# LAW PC.12: a finished write holds the position that was written, and no other.
law pc_commit_holds_what_was_written:
  for +s: PcSlot
  for +t: Nat
  {pc_slot_tiles(pc_commit(pc_begin_write(s, t))) == t : Nat}

def pc_commit_holds_what_was_written(s, t):
  {==}

# LAW PC.13: committing never makes an empty slot readable.
law pc_commit_empty_stays_empty:
  for u: Unit
  {pc_readable(pc_commit(PcEmpty{})) == False{} : Bool}

def pc_commit_empty_stays_empty(u):
  {==}

def pc_retain_aux(t: Nat, tile: Nat, cap: Nat, b: Bool, h: {Bool.and(b, Nat.is_lt(cap, pc_boundary(t, tile))) == False{} : Bool}) -> {Bool.and(pc_readable(pc_retain_if(t, b)), Nat.is_lt(cap, pc_boundary(pc_slot_tiles(pc_retain_if(t, b)), tile))) == False{} : Bool}:
  match b:
    case True{}:
      h
    case False{}:
      {==}

# LAW PC.14: for a new prompt, no slot stays readable whose position lies past the rows the
# prompt shares with the held one.
law pc_retain_within_cap:
  for +s: PcSlot
  for +cap: Nat
  for +tile: Nat
  {Bool.and(pc_readable(pc_retain(s, cap, tile)), Nat.is_lt(cap, pc_boundary(pc_slot_tiles(pc_retain(s, cap, tile)), tile))) == False{} : Bool}

def pc_retain_within_cap(s, cap, tile):
  match s:
    case PcEmpty{}:
      {==}
    case PcWriting{t}:
      {==}
    case PcHeld{t}:
      pc_retain_aux(t, tile, cap, Nat.is_le(pc_boundary(t, tile), cap), pc_le_lt_excl(pc_boundary(t, tile), cap))

# LAW PC.15: a slot left in the middle of a write is never kept for a new prompt.
law pc_retain_drops_unfinished:
  for +t: Nat
  for +cap: Nat
  for +tile: Nat
  {pc_retain(PcWriting{t}, cap, tile) == PcEmpty{} : PcSlot}

def pc_retain_drops_unfinished(t, cap, tile):
  {==}

# LAW PC.16: a request never keeps a state at or past the end of its prompt.
law pc_capture_before_prompt_end:
  for +k: Nat
  for +r: Nat
  for +p: Nat
  for +tile: Nat
  {Bool.and(pc_capture_ok(k, r, p, tile), Nat.is_le(p, pc_boundary(k, tile))) == False{} : Bool}

def pc_capture_before_prompt_end(k, r, p, tile):
  pc_and3_excl(Nat.is_lt(r, k), Nat.is_lt(pc_boundary(k, tile), p), Nat.is_le(p, pc_boundary(k, tile)), pc_lt_le_excl(pc_boundary(k, tile), p))

# LAW PC.17: a request never keeps a state at or before its resume point: every state it
# writes into a slot is one its own tiles computed.
law pc_capture_past_resume:
  for +k: Nat
  for +r: Nat
  for +p: Nat
  for +tile: Nat
  {Bool.and(pc_capture_ok(k, r, p, tile), Nat.is_le(k, r)) == False{} : Bool}

def pc_capture_past_resume(k, r, p, tile):
  pc_and3_excl_l(Nat.is_lt(r, k), Nat.is_lt(pc_boundary(k, tile), p), Nat.is_le(k, r), pc_lt_le_excl(r, k))

# LAW PC.18: a wanted position never lies past the anchor.
law pc_ladder_within_anchor:
  for +a: Nat
  for +rank: Nat
  {Nat.is_le(pc_ladder_tiles(a, rank), a) == True{} : Bool}

def pc_ladder_within_anchor(a, rank):
  pc_sub_le_self(a, pc_ladder_back(rank))

# LAW PC.19: a slot that is being written is never given to another position.
law pc_writing_slot_not_usable:
  for +t: Nat
  for +r: Bool
  for +m: Bool
  {pc_class_usable(pc_victim_class(PcWriting{t}, r, m)) == False{} : Bool}

def pc_writing_slot_not_usable(t, r, m):
  {==}

# LAW PC.20: the slot a request resumes from is never written by the tile that reads it.
law pc_resume_slot_not_same_tile:
  for +t: Nat
  {pc_class_usable(pc_victim_class(PcHeld{t}, True{}, True{})) == False{} : Bool}

def pc_resume_slot_not_same_tile(t):
  {==}

def pc_held_not_before_free(+r: Bool, +m: Bool, +ca: Nat, +cb: Nat) -> {pc_victim_better(pc_victim_class_held(r, m), ca, 0n, cb) == False{} : Bool}:
  match r:
    case False{}: {==}
    case True{}:
      match m:
        case True{}: {==}
        case False{}: {==}

# LAW PC.21: a slot that holds a position is never preferred to an empty one: nothing is
# dropped while a slot is free.
law pc_free_slot_first:
  for +t: Nat
  for +r: Bool
  for +m: Bool
  for +ca: Nat
  for +r2: Bool
  for +m2: Bool
  for +cb: Nat
  {pc_victim_better(pc_victim_class(PcHeld{t}, r, m), ca, pc_victim_class(PcEmpty{}, r2, m2), cb) == False{} : Bool}

def pc_free_slot_first(t, r, m, ca, r2, m2, cb):
  pc_held_not_before_free(r, m, ca, cb)

# LAW PC.22: no slot is better than itself, so the choice among slots is stable.
law pc_victim_better_irreflexive:
  for +c: Nat
  for +k: Nat
  {pc_victim_better(c, k, c, k) == False{} : Bool}

def pc_victim_better_irreflexive(c, k):
  %Equal.sym(Bool, Nat.is_lt(c, c), False{}, pc_lt_refl(c)) : {Bool.or(_, Bool.and(Nat.is_eq(c, c), Nat.is_gt(k, k))) == False{} : Bool}
  %Equal.sym(Bool, Nat.is_gt(k, k), False{}, pc_gt_refl(k)) : {Bool.and(Nat.is_eq(c, c), _) == False{} : Bool}
  pc_and_false_r(Nat.is_eq(c, c))

# LAW PC.23: a slot that is being written is never taken for another position, whatever the
# positions are worth.
law pc_writing_slot_never_taken:
  for +n: Nat
  for +t: Nat
  for +r: Bool
  for +m: Bool
  for +k: Nat
  {pc_may_take(n, pc_victim_class(PcWriting{t}, r, m), k) == False{} : Bool}

def pc_writing_slot_never_taken(n, t, r, m, k):
  {==}

# LAW PC.24: the slot a request resumes from is never taken by the tile that reads it,
# whatever the positions are worth.
law pc_resume_slot_never_taken_by_its_reader:
  for +n: Nat
  for +t: Nat
  for +k: Nat
  {pc_may_take(n, pc_victim_class(PcHeld{t}, True{}, True{}), k) == False{} : Bool}

def pc_resume_slot_never_taken_by_its_reader(n, t, k):
  {==}

def pc_grant_none_aux(g: Nat, b: Bool) -> {Bool.or(b, Nat.is_eq(pc_grant_pick(g, b), 0n)) == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

# LAW PC.25: zero slots are granted whenever the granted count does not fit beside the plan.
law pc_grant_none_when_over:
  for +g: Nat
  for +sb: Nat
  for +bp: Nat
  for +u: Nat
  {Bool.or(pc_plan_fits(g, sb, bp, u), Nat.is_eq(pc_grant_checked(g, sb, bp, u), 0n)) == True{} : Bool}

def pc_grant_none_when_over(g, sb, bp, u):
  pc_grant_none_aux(g, pc_plan_fits(g, sb, bp, u))

def pc_grant_le_aux(g: Nat, b: Bool) -> {Nat.is_le(pc_grant_pick(g, b), g) == True{} : Bool}:
  match b:
    case True{}:
      pc_le_refl(g)
    case False{}:
      pc_zero_le(g)

# LAW PC.26: the granted slots never exceed the count asked for.
law pc_grant_within_request:
  for +g: Nat
  for +sb: Nat
  for +bp: Nat
  for +u: Nat
  {Nat.is_le(pc_grant_checked(g, sb, bp, u), g) == True{} : Bool}

def pc_grant_within_request(g, sb, bp, u):
  pc_grant_le_aux(g, pc_plan_fits(g, sb, bp, u))

# LAW PC.27: a resumed request's drafter never reads a ring row below both the rows that were
# valid before it and the rows its own prefill writes again.
law pc_ring_flo_bound:
  for +f: Nat
  for +r: Nat
  {Nat.is_le(Nat.min(f, r), pc_ring_flo(f, r)) == True{} : Bool}

def pc_ring_flo_bound(f, r):
  pc_le_refl(Nat.min(f, r))

# LAW PC.28: the rows a request's own prefill writes are always readable by its drafter.
law pc_ring_flo_within_resume:
  for +f: Nat
  for +r: Nat
  {Nat.is_le(pc_ring_flo(f, r), r) == True{} : Bool}

def pc_ring_flo_within_resume(f, r):
  pc_min_le_r(f, r)

# LAW PC.29: after a request, no position that its writes overwrote in the ring counts as valid.
law pc_ring_from_past_wrap:
  for +f: Nat
  for +r: Nat
  for +e: Nat
  for +g: Nat
  {Nat.is_le(Nat.sub(e, g), pc_ring_from(f, r, e, g)) == True{} : Bool}

def pc_ring_from_past_wrap(f, r, e, g):
  pc_le_max_r(pc_ring_flo(f, r), Nat.sub(e, g))

# LAW PC.30: nor any position that was not valid for the request itself.
law pc_ring_from_past_flo:
  for +f: Nat
  for +r: Nat
  for +e: Nat
  for +g: Nat
  {Nat.is_le(pc_ring_flo(f, r), pc_ring_from(f, r, e, g)) == True{} : Bool}

def pc_ring_from_past_flo(f, r, e, g):
  pc_le_max_l(pc_ring_flo(f, r), Nat.sub(e, g))

# LAW PC.31: reused rows and prefilled rows of a prompt add up to the prompt.
law pc_rows_conserved:
  for +p: Nat
  for +r: Nat
  {Nat.add(pc_reused_rows(p, r), pc_prefill_rows(p, r)) == p : Nat}

def pc_rows_conserved(p, r):
  pc_split_sum(p, r)

# LAW PC.32: the reused rows never exceed the prompt.
law pc_reused_within_prompt:
  for +p: Nat
  for +r: Nat
  {Nat.is_le(pc_reused_rows(p, r), p) == True{} : Bool}

def pc_reused_within_prompt(p, r):
  pc_min_le_r(r, p)
