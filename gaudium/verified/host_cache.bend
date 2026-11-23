# Host-RAM tier of prefix reuse: a conversation's reusable state kept in host memory, so that it
# outlives the two cases in which the card stops holding it (another conversation in between,
# and the release of the card).
#
# An entry is a copy of what a later request can resume from: the K/V rows of whole prefill
# tiles of a prompt, the recurrent state at some of their ends, and the prompt they belong to.
# Rows are counted in whole prefill tiles here, as in `prefix_reuse.bend`.
#
# Formally specifies and proves:
#   - What the host can give, the default budget and the check of a budget at start
#   - Whether a request may restore from an entry at all (engine identity, entry state)
#   - The life of an entry (vacant, held, being written, being read) with a bounded commit
#   - Whether a copy is worth it against prefilling the same rows
#   - Whether a prompt continues an entry or starts its own
#   - Which entry leaves first, and what a save does when the budget is full
#   - Which device rows still equal an entry's rows after a request wrote some
#   - When an unsaved prefix is copied out before a request overwrites it
#
# The rule for one resume point of an entry (`hc_resume_valid`), the eviction guard
# (`hc_can_evict`) and the byte ledger are in `kv_pool.bend`, beside the rules they build on.

import Base

# ============================================================================
# 1. Helper Nat / Bool lemmas (erased before Rust codegen)
# ============================================================================

def hc_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      hc_min_le_l(ap, bp)

def hc_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      hc_min_le_r(ap, bp)

def hc_le_refl(n: Nat) -> {Nat.is_le(n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      hc_le_refl(p)

def hc_lt_refl(n: Nat) -> {Nat.is_lt(n, n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      hc_lt_refl(p)

def hc_zero_le(n: Nat) -> {Nat.is_le(0n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+_: {==}

def hc_lt_zero(n: Nat) -> {Nat.is_lt(n, 0n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+_: {==}

# a <= max(a, b)
def hc_le_max_l(+a: Nat, +b: Nat) -> {Nat.is_le(a, Nat.max(a, b)) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+ap 0n:
      hc_le_refl(ap)
    case 1n+ap 1n+bp:
      hc_le_max_l(ap, bp)

# b <= max(a, b)
def hc_le_max_r(+a: Nat, +b: Nat) -> {Nat.is_le(b, Nat.max(a, b)) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+bp:
      hc_le_refl(bp)
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      hc_le_max_r(ap, bp)

# max(a, min(b, c)) <= max(a, b)
def hc_max_min_le_l(+a: Nat, +b: Nat, +c: Nat) -> {Nat.is_le(Nat.max(a, Nat.min(b, c)), Nat.max(a, b)) == True{} : Bool}:
  match a b c:
    case 0n 0n _: {==}
    case 0n 1n+bp 0n: {==}
    case 0n 1n+bp 1n+cp:
      hc_min_le_l(bp, cp)
    case 1n+ap 0n _:
      hc_le_refl(ap)
    case 1n+ap 1n+bp 0n:
      hc_le_max_l(ap, bp)
    case 1n+ap 1n+bp 1n+cp:
      hc_max_min_le_l(ap, bp, cp)

# max(a, min(b, c)) <= max(a, c)
def hc_max_min_le_r(+a: Nat, +b: Nat, +c: Nat) -> {Nat.is_le(Nat.max(a, Nat.min(b, c)), Nat.max(a, c)) == True{} : Bool}:
  match a b c:
    case 0n 0n 0n: {==}
    case 0n 0n 1n+_: {==}
    case 0n 1n+bp 0n: {==}
    case 0n 1n+bp 1n+cp:
      hc_min_le_r(bp, cp)
    case 1n+ap 0n 0n:
      hc_le_refl(ap)
    case 1n+ap 0n 1n+cp:
      hc_le_max_l(ap, cp)
    case 1n+ap 1n+bp 0n:
      hc_le_refl(ap)
    case 1n+ap 1n+bp 1n+cp:
      hc_max_min_le_r(ap, bp, cp)

# a < b and b <= a never hold together
def hc_lt_le_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_lt(a, b), Nat.is_le(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      hc_lt_le_excl(ap, bp)

# a <= b and b < a never hold together
def hc_le_lt_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_le(a, b), Nat.is_lt(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      hc_le_lt_excl(ap, bp)

def hc_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def hc_or_true_r(b: Bool) -> {Bool.or(b, True{}) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# (a && x) && y == False when x && y == False
def hc_and3_excl(a: Bool, x: Bool, y: Bool, h: {Bool.and(x, y) == False{} : Bool}) -> {Bool.and(Bool.and(a, x), y) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      h

# a <= a + b
def hc_le_add_r(a: Nat, b: Nat) -> {True{} == Nat.is_le(a, Nat.add(a, b)) : Bool}:
  match a:
    case 0n:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      hc_le_add_r(p, b)

# a <= (a + b) + c
def hc_le_add_add(a: Nat, b: Nat, c: Nat) -> {True{} == Nat.is_le(a, Nat.add(Nat.add(a, b), c)) : Bool}:
  match a:
    case 0n:
      Equal.sym(Bool, Nat.is_le(0n, Nat.add(Nat.add(0n, b), c)), True{}, hc_zero_le(Nat.add(Nat.add(0n, b), c)))
    case 1n+p:
      hc_le_add_add(p, b, c)

# a + (1 + b) <= a never holds
def hc_le_add_gt_false(a: Nat, extra: Nat) -> {False{} == Nat.is_le(Nat.add(a, 1n+extra), a) : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      hc_le_add_gt_false(p, extra)

# a + b < a never holds
def hc_lt_add_r_false(a: Nat, b: Nat) -> {False{} == Nat.is_lt(Nat.add(a, b), a) : Bool}:
  match a:
    case 0n:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      hc_lt_add_r_false(p, b)

def hc_sub_self(n: Nat) -> {Nat.sub(n, n) == 0n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      hc_sub_self(p)

# ============================================================================
# 2. What the host can give, and the budget
# ============================================================================

# Bytes this process can still take: what the host has available, and no more than its control
# group has left when the group has a limit.
def hc_available(+host_available: Nat, has_limit: Bool, limit: Nat, in_use: Nat) -> Nat:
  match has_limit:
    case True{}:
      Nat.min(host_available, Nat.sub(limit, in_use))
    case False{}:
      host_available

# The budget of a server that names none is this share of what is available when it starts
# (one part in ...). The cache takes memory only as prefixes are saved, and gives it back when
# they leave.
def hc_default_share() -> Nat:
  4n

def hc_default_budget(+available: Nat) -> Nat:
  Nat.min(available, Nat.div(available, hc_default_share()))

# What is available beside the pinned transfer buffers.
def hc_beside_pinned(available: Nat, pinned: Nat) -> Nat:
  Nat.sub(available, pinned)

# Can a budget be honoured? It and the pinned transfer buffers must fit in what is available.
def hc_budget_fits(budget: Nat, pinned: Nat, available: Nat) -> Bool:
  Nat.is_le(Nat.add(budget, pinned), available)

# ============================================================================
# 3. May a request restore from an entry at all?
# ============================================================================

# Not when it asked for a cold prefill, not on an engine that cannot resume a prefill, not from
# an entry made by an engine of another identity (model, precision, context, tile, drafter,
# kernels), and not from an entry that is not readable now.
def hc_may_restore(wanted: Bool, supported: Bool, same_engine: Bool, readable: Bool) -> Bool:
  Bool.and(wanted, Bool.and(supported, Bool.and(same_engine, readable)))

# ============================================================================
# 4. The life of an entry
# ============================================================================

# `tiles`: whole tiles whose rows are complete in host memory. `upto`: the tiles the write in
# flight is to reach.
type HcEntry is Data:
  HcVacant{}
  HcHeld{tiles: Nat}
  HcWriting{tiles: Nat, upto: Nat}
  HcReading{tiles: Nat}

# An entry with no rows is no entry.
def hc_of_tiles(tiles: Nat) -> HcEntry:
  match tiles:
    case 0n:
      HcVacant{}
    case 1n+p:
      HcHeld{1n+p}

# Whole tiles the entry holds complete.
def hc_tiles(e: HcEntry) -> Nat:
  match e:
    case HcVacant{}:
      0n
    case HcHeld{t}:
      t
    case HcWriting{t, u}:
      t
    case HcReading{t}:
      t

# A request may restore from the entry.
def hc_readable(e: HcEntry) -> Bool:
  match e:
    case HcVacant{}:
      False{}
    case HcHeld{t}:
      True{}
    case HcWriting{t, u}:
      False{}
    case HcReading{t}:
      False{}

# A save starts to extend the entry to `upto` tiles. An entry that is being written or read is
# left as it is.
def hc_begin_write(e: HcEntry, upto: Nat) -> HcEntry:
  match e:
    case HcVacant{}:
      HcWriting{0n, upto}
    case HcHeld{t}:
      HcWriting{t, upto}
    case HcWriting{t, u}:
      HcWriting{t, u}
    case HcReading{t}:
      HcReading{t}

# The save stopped with the rows of `done` whole tiles complete in host memory (all it set out
# to copy, or fewer when a request took the card first): the entry holds what it held before
# and what was finished, never more than the save's target.
def hc_end_write(e: HcEntry, +done: Nat) -> HcEntry:
  match e:
    case HcVacant{}:
      HcVacant{}
    case HcHeld{t}:
      HcHeld{t}
    case HcWriting{+t, +u}:
      hc_of_tiles(Nat.max(t, Nat.min(done, u)))
    case HcReading{t}:
      HcReading{t}

# A restore starts to read the entry.
def hc_begin_read(e: HcEntry) -> HcEntry:
  match e:
    case HcVacant{}:
      HcVacant{}
    case HcHeld{t}:
      HcReading{t}
    case HcWriting{t, u}:
      HcWriting{t, u}
    case HcReading{t}:
      HcReading{t}

# The restore finished.
def hc_end_read(e: HcEntry) -> HcEntry:
  match e:
    case HcVacant{}:
      HcVacant{}
    case HcHeld{t}:
      HcHeld{t}
    case HcWriting{t, u}:
      HcWriting{t, u}
    case HcReading{t}:
      HcHeld{t}

# A copy of the entry failed, in either direction: nothing of it may be read later.
def hc_abort(e: HcEntry) -> HcEntry:
  match e:
    case HcVacant{}:
      HcVacant{}
    case HcHeld{t}:
      HcHeld{t}
    case HcWriting{t, u}:
      HcVacant{}
    case HcReading{t}:
      HcVacant{}

# The prompt the entry is continued by shares only `keep` of its tiles: the rest is dropped.
def hc_truncate(e: HcEntry, +keep: Nat) -> HcEntry:
  match e:
    case HcVacant{}:
      HcVacant{}
    case HcHeld{+t}:
      hc_of_tiles(Nat.min(t, keep))
    case HcWriting{t, u}:
      HcWriting{t, u}
    case HcReading{t}:
      HcReading{t}

# Copies reading the entry now, and whether it is being written: what the eviction guard
# (`hc_can_evict` in `kv_pool.bend`) is asked with.
def hc_pins(e: HcEntry) -> Nat:
  match e:
    case HcVacant{}:
      0n
    case HcHeld{t}:
      0n
    case HcWriting{t, u}:
      0n
    case HcReading{t}:
      1n

def hc_busy(e: HcEntry) -> Bool:
  match e:
    case HcVacant{}:
      False{}
    case HcHeld{t}:
      False{}
    case HcWriting{t, u}:
      True{}
    case HcReading{t}:
      False{}

# May the rows of `tiles` whole tiles be copied out of the device as rows of the held prompt?
# Only rows a prefill wrote are: `held_rows` is the count of them.
def hc_rows_saveable(tiles: Nat, tile: Nat, held_rows: Nat) -> Bool:
  Nat.is_le(Nat.mul(tiles, tile), held_rows)

# ============================================================================
# 5. Is a copy worth it?
# ============================================================================

# Copying `copy_bytes` at `copy_rate` bytes per second against prefilling `rows` rows at
# `prefill_rate` rows per second: the copy wins when it takes less time. Both rates are
# measured by the engine that asks.
def hc_copy_wins(copy_bytes: Nat, copy_rate: Nat, rows: Nat, prefill_rate: Nat) -> Bool:
  Nat.is_lt(Nat.mul(copy_bytes, prefill_rate), Nat.mul(rows, copy_rate))

# ============================================================================
# 6. Does a prompt continue an entry?
# ============================================================================

# The prompt now held on the device shares `common` whole tiles with an entry of `tiles`. It
# continues the entry (which is then cut to the shared tiles and extended) when it shares at
# least as much as it would drop; otherwise it gets an entry of its own and the other stays.
def hc_continues(+common: Nat, tiles: Nat) -> Bool:
  Bool.and(Nat.is_gt(common, 0n), Nat.is_le(Nat.sub(tiles, common), common))

# Whole tiles of an entry of `entry_tiles` that a prompt sharing `shared_rows` rows with the
# entry's prompt can keep.
def hc_shared_tiles(shared_rows: Nat, tile: Nat, +entry_tiles: Nat) -> Nat:
  Nat.min(entry_tiles, Nat.div(shared_rows, tile))

# An entry keeps the recurrent state at no more positions than the engine keeps on the device:
# is one over? When it is, the position nearest the start of the prompt leaves first.
def hc_states_over(count: Nat, cap: Nat) -> Bool:
  Nat.is_lt(cap, count)

def hc_state_leaves_first(a_tiles: Nat, b_tiles: Nat) -> Bool:
  Nat.is_lt(a_tiles, b_tiles)

# ============================================================================
# 7. Room for a save
# ============================================================================

# Does entry a leave before entry b? The one used less recently.
def hc_evict_before(a_used: Nat, b_used: Nat) -> Bool:
  Nat.is_lt(a_used, b_used)

# What a save that needs `need` bytes does with `free` bytes free and `evictable` bytes in
# entries that may leave.
type HcRoom is Data:
  HcFits{}
  HcEvict{}
  HcRefuse{}

def hc_room_pick(can: Bool, fits: Bool) -> HcRoom:
  match can:
    case False{}:
      HcRefuse{}
    case True{}:
      match fits:
        case True{}:
          HcFits{}
        case False{}:
          HcEvict{}

def hc_room(+need: Nat, +free: Nat, evictable: Nat) -> HcRoom:
  hc_room_pick(Nat.is_le(need, Nat.add(free, evictable)), Nat.is_le(need, free))

# ============================================================================
# 8. Device rows that still equal an entry's rows
# ============================================================================
#
# After a save or a restore the device rows of some tiles are byte for byte the rows of an
# entry: a later restore need not copy them again. They are kept as one interval [lo, hi) of
# tiles. A request that wrote the tiles [w0, w1) leaves the larger part of the interval that
# lies wholly below or wholly above what it wrote.

def hc_clean_keep_upper(+lo: Nat, +hi: Nat, +w0: Nat, +w1: Nat) -> Bool:
  Nat.is_lt(Nat.sub(Nat.min(hi, w0), lo), Nat.sub(hi, Nat.max(lo, w1)))

def hc_clean_lo_if(upper: Bool, +lo: Nat, +w1: Nat) -> Nat:
  match upper:
    case True{}:
      Nat.max(lo, w1)
    case False{}:
      lo

def hc_clean_hi_if(upper: Bool, +hi: Nat, +w0: Nat) -> Nat:
  match upper:
    case True{}:
      hi
    case False{}:
      Nat.min(hi, w0)

def hc_clean_lo(+lo: Nat, +hi: Nat, +w0: Nat, +w1: Nat) -> Nat:
  hc_clean_lo_if(hc_clean_keep_upper(lo, hi, w0, w1), lo, w1)

def hc_clean_hi(+lo: Nat, +hi: Nat, +w0: Nat, +w1: Nat) -> Nat:
  hc_clean_hi_if(hc_clean_keep_upper(lo, hi, w0, w1), hi, w0)

# The tiles a request wrote, from the rows it wrote: `[resume, written_end)` rounded outward to
# whole tiles.
def hc_written_lo(resume_rows: Nat, tile: Nat) -> Nat:
  Nat.div(resume_rows, tile)

def hc_written_hi(written_end: Nat, +tile: Nat) -> Nat:
  Nat.div(Nat.add(written_end, Nat.sub(tile, 1n)), tile)

# A copy just made the device tiles [a, b) equal an entry's (a save read them, a restore wrote
# them). The interval [lo, hi) the entry had is joined with them when the two touch or overlap;
# otherwise only the new tiles count (one interval is kept, and a gap is never bridged).
def hc_clean_touch(+lo: Nat, +hi: Nat, +a: Nat, +b: Nat) -> Bool:
  Bool.and(Nat.is_le(a, hi), Nat.is_le(lo, b))

def hc_clean_join_lo_if(touch: Bool, +lo: Nat, +a: Nat) -> Nat:
  match touch:
    case True{}:
      Nat.min(lo, a)
    case False{}:
      a

def hc_clean_join_hi_if(touch: Bool, +hi: Nat, +b: Nat) -> Nat:
  match touch:
    case True{}:
      Nat.max(hi, b)
    case False{}:
      b

def hc_clean_join_lo(+lo: Nat, +hi: Nat, +a: Nat, +b: Nat) -> Nat:
  hc_clean_join_lo_if(hc_clean_touch(lo, hi, a, b), lo, a)

def hc_clean_join_hi(+lo: Nat, +hi: Nat, +a: Nat, +b: Nat) -> Nat:
  hc_clean_join_hi_if(hc_clean_touch(lo, hi, a, b), hi, b)

# Are the device rows of tile `t` still the entry's? Only inside the interval, and only while
# the device is linked to that entry at all.
def hc_tile_clean(linked: Bool, +t: Nat, lo: Nat, hi: Nat) -> Bool:
  Bool.and(linked, Bool.and(Nat.is_le(lo, t), Nat.is_lt(t, hi)))

# ============================================================================
# 9. When a save runs
# ============================================================================

# The save after a response runs while no request waits for the card.
def hc_idle_may_run(waiting: Nat) -> Bool:
  Nat.is_eq(waiting, 0n)

# Does a save copy the drafter's rings again? An entry's image stays good for every position
# the entry had when the image was taken, so only an entry without one, or one that this save
# gave new rows or positions, needs it.
def hc_ring_stale(has_image: Bool, grew: Bool) -> Bool:
  Bool.or(Bool.not(has_image), grew)

# May a ring image taken when its entry held `image_tiles` tiles serve a resume after
# `resume_tiles`? Only rows of positions before the resume point matter, and the image holds the
# prompt's rows only up to the tiles the entry had then.
def hc_ring_serves(image_tiles: Nat, resume_tiles: Nat) -> Bool:
  Nat.is_le(resume_tiles, image_tiles)

# The tiles a ring image still stands for after its entry was cut to `keep` tiles.
def hc_ring_cover_after_cut(+image_tiles: Nat, +keep: Nat) -> Nat:
  Nat.min(image_tiles, keep)

# Must a request first copy out what the device holds and host memory does not? Only when
# something is unsaved and the request shares fewer rows with the held prompt than the last
# kept position needs: the positions behind the shared rows are about to be dropped.
def hc_must_flush(unsaved: Bool, cap: Nat, kept_rows: Nat) -> Bool:
  Bool.and(unsaved, Nat.is_lt(cap, kept_rows))

# ============================================================================
# 10. Where a request's reused rows came from
# ============================================================================

type HcTier is Data:
  HcPrefilled{}
  HcFromDevice{}
  HcFromHost{}

# What a request reports: nothing reused means everything was prefilled, whatever was copied;
# rows reused after a restore came from host memory, any others were still on the device.
def hc_tier(restored: Bool, reused_rows: Nat) -> HcTier:
  match restored reused_rows:
    case True{} 0n:
      HcPrefilled{}
    case True{} 1n+p:
      HcFromHost{}
    case False{} 0n:
      HcPrefilled{}
    case False{} 1n+p:
      HcFromDevice{}

# ============================================================================
# 11. Laws
# ============================================================================

# LAW HC.1: what a process may take never exceeds what the host has available.
law hc_available_within_host:
  for +h: Nat
  for +l: Bool
  for +lim: Nat
  for +used: Nat
  {Nat.is_le(hc_available(h, l, lim, used), h) == True{} : Bool}

def hc_available_within_host(h, l, lim, used):
  match l:
    case True{}:
      hc_min_le_l(h, Nat.sub(lim, used))
    case False{}:
      hc_le_refl(h)

# LAW HC.2: nor what its control group has left, when the group has a limit.
law hc_available_within_group:
  for +h: Nat
  for +lim: Nat
  for +used: Nat
  {Nat.is_le(hc_available(h, True{}, lim, used), Nat.sub(lim, used)) == True{} : Bool}

def hc_available_within_group(h, lim, used):
  hc_min_le_r(h, Nat.sub(lim, used))

# LAW HC.3: the default budget never exceeds what is available.
law hc_default_within_available:
  for +a: Nat
  {Nat.is_le(hc_default_budget(a), a) == True{} : Bool}

def hc_default_within_available(a):
  hc_min_le_l(a, Nat.div(a, hc_default_share()))

# LAW HC.4: a budget that does not fit beside the pinned buffers in what is available is never
# accepted.
law hc_budget_over_refused:
  for +b: Nat
  for +p: Nat
  for +a: Nat
  {Bool.and(hc_budget_fits(b, p, a), Nat.is_lt(a, Nat.add(b, p))) == False{} : Bool}

def hc_budget_over_refused(b, p, a):
  hc_le_lt_excl(Nat.add(b, p), a)

# LAW HC.5: a request that asked for a cold prefill never restores from host memory.
law hc_cold_request_never_restores:
  for +s: Bool
  for +i: Bool
  for +r: Bool
  {hc_may_restore(False{}, s, i, r) == False{} : Bool}

def hc_cold_request_never_restores(s, i, r):
  {==}

# LAW HC.6: an engine that cannot resume a prefill never restores.
law hc_unsupported_never_restores:
  for +w: Bool
  for +i: Bool
  for +r: Bool
  {hc_may_restore(w, False{}, i, r) == False{} : Bool}

def hc_unsupported_never_restores(w, i, r):
  match w:
    case False{}: {==}
    case True{}: {==}

# LAW HC.7: an entry made by an engine of another identity is never restored.
law hc_other_engine_never_restores:
  for +w: Bool
  for +s: Bool
  for +r: Bool
  {hc_may_restore(w, s, False{}, r) == False{} : Bool}

def hc_other_engine_never_restores(w, s, r):
  match w:
    case False{}: {==}
    case True{}:
      match s:
        case False{}: {==}
        case True{}: {==}

# LAW HC.8: an entry that is not readable is never restored.
law hc_unreadable_never_restores:
  for +w: Bool
  for +s: Bool
  for +i: Bool
  {hc_may_restore(w, s, i, False{}) == False{} : Bool}

def hc_unreadable_never_restores(w, s, i):
  match w:
    case False{}: {==}
    case True{}:
      match s:
        case False{}: {==}
        case True{}:
          match i:
            case False{}: {==}
            case True{}: {==}

# LAW HC.9: an entry that is being written is never readable, whatever it was before.
law hc_writing_not_readable:
  for +e: HcEntry
  for +u: Nat
  {hc_readable(hc_begin_write(e, u)) == False{} : Bool}

def hc_writing_not_readable(e, u):
  match e:
    case HcVacant{}: {==}
    case HcHeld{t}: {==}
    case HcWriting{t, w}: {==}
    case HcReading{t}: {==}

# LAW HC.10: a vacant entry is never readable.
law hc_vacant_not_readable:
  for u: Unit
  {hc_readable(HcVacant{}) == False{} : Bool}

def hc_vacant_not_readable(u):
  {==}

def hc_of_tiles_tiles(n: Nat) -> {hc_tiles(hc_of_tiles(n)) == n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p: {==}

# LAW HC.11: after a save, an entry never holds rows that it neither held before nor the save
# finished: a save that stopped early leaves nothing half done to read.
law hc_end_write_within_done:
  for +t: Nat
  for +u: Nat
  for +d: Nat
  {Nat.is_le(hc_tiles(hc_end_write(HcWriting{t, u}, d)), Nat.max(t, d)) == True{} : Bool}

def hc_end_write_within_done(t, u, d):
  %Equal.sym(Nat, hc_tiles(hc_of_tiles(Nat.max(t, Nat.min(d, u)))), Nat.max(t, Nat.min(d, u)), hc_of_tiles_tiles(Nat.max(t, Nat.min(d, u)))) : {Nat.is_le(_, Nat.max(t, d)) == True{} : Bool}
  hc_max_min_le_l(t, d, u)

# LAW HC.12: nor more than the save set out to reach.
law hc_end_write_within_target:
  for +t: Nat
  for +u: Nat
  for +d: Nat
  {Nat.is_le(hc_tiles(hc_end_write(HcWriting{t, u}, d)), Nat.max(t, u)) == True{} : Bool}

def hc_end_write_within_target(t, u, d):
  %Equal.sym(Nat, hc_tiles(hc_of_tiles(Nat.max(t, Nat.min(d, u)))), Nat.max(t, Nat.min(d, u)), hc_of_tiles_tiles(Nat.max(t, Nat.min(d, u)))) : {Nat.is_le(_, Nat.max(t, u)) == True{} : Bool}
  hc_max_min_le_r(t, d, u)

# LAW HC.13: the end of a save changes no entry that was not being written.
law hc_end_write_keeps_held:
  for +t: Nat
  for +d: Nat
  {hc_end_write(HcHeld{t}, d) == HcHeld{t} : HcEntry}

def hc_end_write_keeps_held(t, d):
  {==}

# LAW HC.14: ... and never makes a vacant entry readable.
law hc_end_write_keeps_vacant:
  for +d: Nat
  {hc_end_write(HcVacant{}, d) == HcVacant{} : HcEntry}

def hc_end_write_keeps_vacant(d):
  {==}

# LAW HC.15: a save that failed leaves nothing, whatever the entry held before.
law hc_failed_write_leaves_nothing:
  for +e: HcEntry
  for +u: Nat
  {hc_abort(hc_begin_write(e, u)) == HcVacant{} : HcEntry}

def hc_failed_write_leaves_nothing(e, u):
  match e:
    case HcVacant{}: {==}
    case HcHeld{t}: {==}
    case HcWriting{t, w}: {==}
    case HcReading{t}: {==}

# LAW HC.16: a restore that failed leaves nothing either.
law hc_failed_read_leaves_nothing:
  for +e: HcEntry
  {hc_abort(hc_begin_read(e)) == HcVacant{} : HcEntry}

def hc_failed_read_leaves_nothing(e):
  match e:
    case HcVacant{}: {==}
    case HcHeld{t}: {==}
    case HcWriting{t, w}: {==}
    case HcReading{t}: {==}

# LAW HC.17: a restore that finished leaves the entry as it was.
law hc_read_returns_entry:
  for +t: Nat
  {hc_end_read(hc_begin_read(HcHeld{t})) == HcHeld{t} : HcEntry}

def hc_read_returns_entry(t):
  {==}

# LAW HC.18: an entry a restore reads counts as pinned (the eviction guard then refuses it).
law hc_reading_is_pinned:
  for +t: Nat
  {hc_pins(hc_begin_read(HcHeld{t})) == 1n : Nat}

def hc_reading_is_pinned(t):
  {==}

# LAW HC.19: an entry a save writes counts as busy (the eviction guard then refuses it).
law hc_writing_is_busy:
  for +t: Nat
  for +u: Nat
  {hc_busy(hc_begin_write(HcHeld{t}, u)) == True{} : Bool}

def hc_writing_is_busy(t, u):
  {==}

# LAW HC.20: a new entry that is being written counts as busy from its first byte.
law hc_new_entry_is_busy:
  for +u: Nat
  {hc_busy(hc_begin_write(HcVacant{}, u)) == True{} : Bool}

def hc_new_entry_is_busy(u):
  {==}

# LAW HC.21: cutting an entry never leaves it more tiles than the prompt shares.
law hc_truncate_within_keep:
  for +t: Nat
  for +k: Nat
  {Nat.is_le(hc_tiles(hc_truncate(HcHeld{t}, k)), k) == True{} : Bool}

def hc_truncate_within_keep(t, k):
  %Equal.sym(Nat, hc_tiles(hc_of_tiles(Nat.min(t, k))), Nat.min(t, k), hc_of_tiles_tiles(Nat.min(t, k))) : {Nat.is_le(_, k) == True{} : Bool}
  hc_min_le_r(t, k)

# LAW HC.22: nor more than it held.
law hc_truncate_never_grows:
  for +t: Nat
  for +k: Nat
  {Nat.is_le(hc_tiles(hc_truncate(HcHeld{t}, k)), t) == True{} : Bool}

def hc_truncate_never_grows(t, k):
  %Equal.sym(Nat, hc_tiles(hc_of_tiles(Nat.min(t, k))), Nat.min(t, k), hc_of_tiles_tiles(Nat.min(t, k))) : {Nat.is_le(_, t) == True{} : Bool}
  hc_min_le_l(t, k)

# LAW HC.23: rows past the ones a prefill wrote are never copied out as rows of its prompt.
law hc_only_prefilled_rows_saved:
  for +k: Nat
  for +tile: Nat
  for +h: Nat
  {Bool.and(hc_rows_saveable(k, tile, h), Nat.is_lt(h, Nat.mul(k, tile))) == False{} : Bool}

def hc_only_prefilled_rows_saved(k, tile, h):
  hc_le_lt_excl(Nat.mul(k, tile), h)

# LAW HC.24: a copy that saves no row of prefill is never chosen.
law hc_copy_never_wins_for_no_rows:
  for +b: Nat
  for +c: Nat
  for +p: Nat
  {hc_copy_wins(b, c, 0n, p) == False{} : Bool}

def hc_copy_never_wins_for_no_rows(b, c, p):
  hc_lt_zero(Nat.mul(b, p))

# LAW HC.25: a prompt that shares no whole tile with an entry never continues it: no entry is
# emptied to hold another conversation.
law hc_no_shared_tile_never_continues:
  for +t: Nat
  {hc_continues(0n, t) == False{} : Bool}

def hc_no_shared_tile_never_continues(t):
  {==}

# LAW HC.26: a prompt that shares every tile of an entry continues it: the same rows are never
# held twice.
law hc_whole_entry_shared_continues:
  for +t: Nat
  {hc_continues(1n+t, 1n+t) == True{} : Bool}

def hc_whole_entry_shared_continues(t):
  %Equal.sym(Nat, Nat.sub(t, t), 0n, hc_sub_self(t)) : {Nat.is_le(_, 1n+t) == True{} : Bool}
  {==}

# LAW HC.27: no entry leaves before itself, so the choice among entries is stable.
law hc_evict_before_irreflexive:
  for +a: Nat
  {hc_evict_before(a, a) == False{} : Bool}

def hc_evict_before_irreflexive(a):
  hc_lt_refl(a)

# LAW HC.28: a save that fits in the free bytes evicts nothing.
law hc_room_fits_evicts_nothing:
  for +n: Nat
  for +extra: Nat
  for +ev: Nat
  {hc_room(n, Nat.add(n, extra), ev) == HcFits{} : HcRoom}

def hc_room_fits_evicts_nothing(n, extra, ev):
  %hc_le_add_add(n, extra, ev) : {hc_room_pick(_, Nat.is_le(n, Nat.add(n, extra))) == HcFits{} : HcRoom}
  %hc_le_add_r(n, extra) : {hc_room_pick(True{}, _) == HcFits{} : HcRoom}
  {==}

# LAW HC.29: a save that cannot fit even with every entry that may leave gone is refused, and
# evicts nothing for it.
law hc_room_refuses_what_cannot_fit:
  for +f: Nat
  for +ev: Nat
  for +x: Nat
  {hc_room(Nat.add(Nat.add(f, ev), 1n+x), f, ev) == HcRefuse{} : HcRoom}

def hc_room_refuses_what_cannot_fit(f, ev, x):
  %hc_le_add_gt_false(Nat.add(f, ev), x) : {hc_room_pick(_, Nat.is_le(Nat.add(Nat.add(f, ev), 1n+x), f)) == HcRefuse{} : HcRoom}
  {==}

def hc_clean_apart_aux(k: Bool, +lo: Nat, +hi: Nat, +w0: Nat, +w1: Nat) -> {Bool.or(Nat.is_le(w1, hc_clean_lo_if(k, lo, w1)), Nat.is_le(hc_clean_hi_if(k, hi, w0), w0)) == True{} : Bool}:
  match k:
    case True{}:
      %Equal.sym(Bool, Nat.is_le(w1, Nat.max(lo, w1)), True{}, hc_le_max_r(lo, w1)) : {Bool.or(_, Nat.is_le(hi, w0)) == True{} : Bool}
      {==}
    case False{}:
      %Equal.sym(Bool, Nat.is_le(Nat.min(hi, w0), w0), True{}, hc_min_le_r(hi, w0)) : {Bool.or(Nat.is_le(w1, lo), _) == True{} : Bool}
      hc_or_true_r(Nat.is_le(w1, lo))

# LAW HC.30: after a request, the tiles that still count as an entry's lie wholly above or
# wholly below the tiles the request wrote: no written tile is ever taken for the entry's.
law hc_clean_apart_from_written:
  for +lo: Nat
  for +hi: Nat
  for +w0: Nat
  for +w1: Nat
  {Bool.or(Nat.is_le(w1, hc_clean_lo(lo, hi, w0, w1)), Nat.is_le(hc_clean_hi(lo, hi, w0, w1), w0)) == True{} : Bool}

def hc_clean_apart_from_written(lo, hi, w0, w1):
  hc_clean_apart_aux(hc_clean_keep_upper(lo, hi, w0, w1), lo, hi, w0, w1)

def hc_clean_lo_aux(k: Bool, +lo: Nat, +w1: Nat) -> {Nat.is_le(lo, hc_clean_lo_if(k, lo, w1)) == True{} : Bool}:
  match k:
    case True{}:
      hc_le_max_l(lo, w1)
    case False{}:
      hc_le_refl(lo)

# LAW HC.31: they never start below where they started ...
law hc_clean_never_grows_down:
  for +lo: Nat
  for +hi: Nat
  for +w0: Nat
  for +w1: Nat
  {Nat.is_le(lo, hc_clean_lo(lo, hi, w0, w1)) == True{} : Bool}

def hc_clean_never_grows_down(lo, hi, w0, w1):
  hc_clean_lo_aux(hc_clean_keep_upper(lo, hi, w0, w1), lo, w1)

def hc_clean_hi_aux(k: Bool, +hi: Nat, +w0: Nat) -> {Nat.is_le(hc_clean_hi_if(k, hi, w0), hi) == True{} : Bool}:
  match k:
    case True{}:
      hc_le_refl(hi)
    case False{}:
      hc_min_le_l(hi, w0)

# LAW HC.32: ... nor end above where they ended.
law hc_clean_never_grows_up:
  for +lo: Nat
  for +hi: Nat
  for +w0: Nat
  for +w1: Nat
  {Nat.is_le(hc_clean_hi(lo, hi, w0, w1), hi) == True{} : Bool}

def hc_clean_never_grows_up(lo, hi, w0, w1):
  hc_clean_hi_aux(hc_clean_keep_upper(lo, hi, w0, w1), hi, w0)

# LAW HC.33: without a link to the entry no device tile counts as the entry's.
law hc_unlinked_never_clean:
  for +t: Nat
  for +lo: Nat
  for +hi: Nat
  {hc_tile_clean(False{}, t, lo, hi) == False{} : Bool}

def hc_unlinked_never_clean(t, lo, hi):
  {==}

# LAW HC.34: a tile at or past the end of the interval never counts as the entry's.
law hc_tile_past_interval_not_clean:
  for +l: Bool
  for +t: Nat
  for +lo: Nat
  for +hi: Nat
  {Bool.and(hc_tile_clean(l, t, lo, hi), Nat.is_le(hi, t)) == False{} : Bool}

def hc_tile_past_interval_not_clean(l, t, lo, hi):
  hc_and3_excl(l, Bool.and(Nat.is_le(lo, t), Nat.is_lt(t, hi)), Nat.is_le(hi, t), hc_and3_excl(Nat.is_le(lo, t), Nat.is_lt(t, hi), Nat.is_le(hi, t), hc_lt_le_excl(t, hi)))

# LAW HC.35: nor a tile below its start.
law hc_tile_below_interval_not_clean:
  for +l: Bool
  for +t: Nat
  for +lo: Nat
  for +hi: Nat
  {Bool.and(hc_tile_clean(l, t, lo, hi), Nat.is_lt(t, lo)) == False{} : Bool}

def hc_tile_below_and(a: Bool, b: Bool, y: Bool, h: {Bool.and(a, y) == False{} : Bool}) -> {Bool.and(Bool.and(a, b), y) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      match b:
        case False{}:
          {==}
        case True{}:
          h

def hc_tile_below_interval_not_clean(l, t, lo, hi):
  hc_and3_excl(l, Bool.and(Nat.is_le(lo, t), Nat.is_lt(t, hi)), Nat.is_lt(t, lo), hc_tile_below_and(Nat.is_le(lo, t), Nat.is_lt(t, hi), Nat.is_lt(t, lo), hc_le_lt_excl(lo, t)))

# LAW HC.36: the save after a response never runs while a request waits for the card.
law hc_idle_save_yields:
  for +w: Nat
  {hc_idle_may_run(1n+w) == False{} : Bool}

def hc_idle_save_yields(w):
  {==}

# LAW HC.37: a request pays for no copy when nothing is unsaved.
law hc_nothing_unsaved_never_flushes:
  for +c: Nat
  for +k: Nat
  {hc_must_flush(False{}, c, k) == False{} : Bool}

def hc_nothing_unsaved_never_flushes(c, k):
  {==}

# LAW HC.38: nor when it keeps every kept position (it shares at least their rows).
law hc_shared_request_never_flushes:
  for +u: Bool
  for +k: Nat
  for +x: Nat
  {hc_must_flush(u, Nat.add(k, x), k) == False{} : Bool}

def hc_shared_request_never_flushes(u, k, x):
  %hc_lt_add_r_false(k, x) : {Bool.and(u, _) == False{} : Bool}
  hc_and_false_r(u)

# LAW HC.39: a prompt never keeps more tiles of an entry than the entry holds.
law hc_shared_tiles_within_entry:
  for +r: Nat
  for +tile: Nat
  for +e: Nat
  {Nat.is_le(hc_shared_tiles(r, tile, e), e) == True{} : Bool}

def hc_shared_tiles_within_entry(r, tile, e):
  hc_min_le_l(e, Nat.div(r, tile))

# LAW HC.40: an entry with no more kept positions than the engine has slots is never over.
law hc_states_within_cap_not_over:
  for +n: Nat
  for +x: Nat
  {hc_states_over(n, Nat.add(n, x)) == False{} : Bool}

def hc_states_within_cap_not_over(n, x):
  Equal.sym(Bool, False{}, Nat.is_lt(Nat.add(n, x), n), hc_lt_add_r_false(n, x))

# LAW HC.41: no kept position leaves before itself.
law hc_state_leaves_first_irreflexive:
  for +a: Nat
  {hc_state_leaves_first(a, a) == False{} : Bool}

def hc_state_leaves_first_irreflexive(a):
  hc_lt_refl(a)

def hc_le_step_r_zero(xp: Nat, h: {False{} == True{} : Bool}) -> {Nat.is_le(xp, 0n) == True{} : Bool}:
  match xp:
    case 0n: {==}
    case 1n+_: h

def hc_le_step_r(x: Nat, y: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(x, 1n+y) == True{} : Bool}:
  match x y:
    case 0n _:
      {==}
    case 1n+xp 0n:
      hc_le_step_r_zero(xp, h)
    case 1n+xp 1n+yp:
      hc_le_step_r(xp, yp, h)

def hc_sub_le_self(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      hc_le_refl(ap)
    case 1n+ap 1n+bp:
      hc_le_step_r(Nat.sub(ap, bp), ap, hc_sub_le_self(ap, bp))

# LAW HC.42: what is left beside the pinned buffers never exceeds what is available.
law hc_beside_pinned_within_available:
  for +a: Nat
  for +p: Nat
  {Nat.is_le(hc_beside_pinned(a, p), a) == True{} : Bool}

def hc_beside_pinned_within_available(a, p):
  hc_sub_le_self(a, p)

def hc_join_lo_aux(k: Bool, +lo: Nat, +a: Nat) -> {Nat.is_le(hc_clean_join_lo_if(k, lo, a), a) == True{} : Bool}:
  match k:
    case True{}:
      hc_min_le_r(lo, a)
    case False{}:
      hc_le_refl(a)

# LAW HC.43: the tiles a copy just made equal always count: the joined interval starts at or
# before them ...
law hc_clean_join_covers_start:
  for +lo: Nat
  for +hi: Nat
  for +a: Nat
  for +b: Nat
  {Nat.is_le(hc_clean_join_lo(lo, hi, a, b), a) == True{} : Bool}

def hc_clean_join_covers_start(lo, hi, a, b):
  hc_join_lo_aux(hc_clean_touch(lo, hi, a, b), lo, a)

def hc_join_hi_aux(k: Bool, +hi: Nat, +b: Nat) -> {Nat.is_le(b, hc_clean_join_hi_if(k, hi, b)) == True{} : Bool}:
  match k:
    case True{}:
      hc_le_max_r(hi, b)
    case False{}:
      hc_le_refl(b)

# LAW HC.44: ... and ends at or after them.
law hc_clean_join_covers_end:
  for +lo: Nat
  for +hi: Nat
  for +a: Nat
  for +b: Nat
  {Nat.is_le(b, hc_clean_join_hi(lo, hi, a, b)) == True{} : Bool}

def hc_clean_join_covers_end(lo, hi, a, b):
  hc_join_hi_aux(hc_clean_touch(lo, hi, a, b), hi, b)

# LAW HC.45: tiles that start past a gap after the old interval are never joined with it: the
# tiles in the gap never count as the entry's.
law hc_clean_join_never_bridges_a_gap:
  for +lo: Nat
  for +hi: Nat
  for +g: Nat
  for +b: Nat
  {hc_clean_join_lo(lo, hi, Nat.add(hi, 1n+g), b) == Nat.add(hi, 1n+g) : Nat}

def hc_clean_join_never_bridges_a_gap(lo, hi, g, b):
  %hc_le_add_gt_false(hi, g) : {hc_clean_join_lo_if(Bool.and(_, Nat.is_le(lo, b)), lo, Nat.add(hi, 1n+g)) == Nat.add(hi, 1n+g) : Nat}
  {==}

# LAW HC.46: nor tiles that end before a gap below it.
law hc_clean_join_never_bridges_a_gap_below:
  for +hi: Nat
  for +a: Nat
  for +b: Nat
  for +g: Nat
  {hc_clean_join_hi(Nat.add(b, 1n+g), hi, a, b) == b : Nat}

def hc_clean_join_never_bridges_a_gap_below(hi, a, b, g):
  %hc_le_add_gt_false(b, g) : {hc_clean_join_hi_if(Bool.and(Nat.is_le(a, hi), _), hi, b) == b : Nat}
  %Equal.sym(Bool, Bool.and(Nat.is_le(a, hi), False{}), False{}, hc_and_false_r(Nat.is_le(a, hi))) : {hc_clean_join_hi_if(_, hi, b) == b : Nat}
  {==}

# LAW HC.47: a request that reused no row never reports a reuse, restored or not.
law hc_tier_nothing_reused_is_prefilled:
  for +r: Bool
  {hc_tier(r, 0n) == HcPrefilled{} : HcTier}

def hc_tier_nothing_reused_is_prefilled(r):
  match r:
    case True{}: {==}
    case False{}: {==}

# LAW HC.48: rows reused without a restore are never reported as rows from host memory.
law hc_tier_no_restore_is_never_host:
  for +n: Nat
  {hc_tier(False{}, 1n+n) == HcFromDevice{} : HcTier}

def hc_tier_no_restore_is_never_host(n):
  {==}

# LAW HC.49: an entry without an image of the drafter's rings always gets one.
law hc_ring_missing_is_stale:
  for +g: Bool
  {hc_ring_stale(False{}, g) == True{} : Bool}

def hc_ring_missing_is_stale(g):
  {==}

# LAW HC.50: an entry that gained rows or positions never keeps the image taken before them.
law hc_ring_grown_is_stale:
  for +h: Bool
  {hc_ring_stale(h, True{}) == True{} : Bool}

def hc_ring_grown_is_stale(h):
  hc_or_true_r(Bool.not(h))

# LAW HC.51: a ring image never serves a resume point past the tiles its entry held when it was
# taken: rows written since are never passed off as that prompt's.
law hc_ring_never_serves_past_its_tiles:
  for +i: Nat
  for +r: Nat
  {Bool.and(hc_ring_serves(i, r), Nat.is_lt(i, r)) == False{} : Bool}

def hc_ring_never_serves_past_its_tiles(i, r):
  hc_le_lt_excl(r, i)

# LAW HC.52: after its entry is cut, a ring image stands for no tile the entry lost.
law hc_ring_cover_within_cut:
  for +i: Nat
  for +k: Nat
  {Nat.is_le(hc_ring_cover_after_cut(i, k), k) == True{} : Bool}

def hc_ring_cover_within_cut(i, k):
  hc_min_le_r(i, k)

# LAW HC.53: nor for more than it did.
law hc_ring_cover_never_grows:
  for +i: Nat
  for +k: Nat
  {Nat.is_le(hc_ring_cover_after_cut(i, k), i) == True{} : Bool}

def hc_ring_cover_never_grows(i, k):
  hc_min_le_l(i, k)
