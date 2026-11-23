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

# Bytes a save may take without evicting: what the budget has free, and never more than the
# host can give at that moment (the budget was checked when the server started; other
# processes have taken and given memory since).
def hc_free_now(+budget_free: Nat, +host_available: Nat) -> Nat:
  Nat.min(budget_free, host_available)

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

# Is a request about to drop kept positions that host memory lacks? Only when something is
# unsaved and the request shares fewer rows with the held prompt than the last kept position
# needs: the positions behind the shared rows are dropped.
def hc_loses_unsaved(unsaved: Bool, cap: Nat, kept_rows: Nat) -> Bool:
  Bool.and(unsaved, Nat.is_lt(cap, kept_rows))

# Must that request first copy them out, on its own time? Only for a conversation that has
# shown it goes on: the request that left the state took rows over itself (`reused_rows`). A
# prompt that was only ever prefilled once (a batch of unrelated requests looks like that) is
# saved in idle time or not at all, so no request pays for a prefix nobody comes back to.
def hc_must_flush(unsaved: Bool, reused_rows: Nat, cap: Nat, kept_rows: Nat) -> Bool:
  Bool.and(hc_loses_unsaved(unsaved, cap, kept_rows), Nat.is_gt(reused_rows, 0n))

# Can anything resume from an entry with this many kept positions? Not from one with none: its
# rows have no state to continue from.
def hc_resumable(states: Nat) -> Bool:
  Nat.is_gt(states, 0n)

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

# LAW HC.17: a restore never changes how many rows an entry holds, whatever state it was in.
law hc_read_keeps_rows:
  for +e: HcEntry
  {hc_tiles(hc_end_read(hc_begin_read(e))) == hc_tiles(e) : Nat}

def hc_read_keeps_rows(e):
  match e:
    case HcVacant{}: {==}
    case HcHeld{t}: {==}
    case HcWriting{t, w}: {==}
    case HcReading{t}: {==}

# LAW HC.18: an entry a restore reads counts as pinned (the eviction guard then refuses it).
law hc_reading_is_pinned:
  for +t: Nat
  {Nat.is_gt(hc_pins(hc_begin_read(HcHeld{t})), 0n) == True{} : Bool}

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
  for +r: Nat
  for +c: Nat
  for +k: Nat
  {hc_must_flush(False{}, r, c, k) == False{} : Bool}

def hc_nothing_unsaved_never_flushes(r, c, k):
  {==}

def hc_and_false_l_and(u: Bool, x: Bool, y: Bool, h: {x == False{} : Bool}) -> {Bool.and(Bool.and(u, x), y) == False{} : Bool}:
  match u:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, x, False{}, h) : {Bool.and(_, y) == False{} : Bool}
      {==}

# LAW HC.38: nor when it keeps every kept position (it shares at least their rows).
law hc_shared_request_never_flushes:
  for +u: Bool
  for +r: Nat
  for +k: Nat
  for +x: Nat
  {hc_must_flush(u, r, Nat.add(k, x), k) == False{} : Bool}

def hc_shared_request_never_flushes(u, r, k, x):
  hc_and_false_l_and(u, Nat.is_lt(Nat.add(k, x), k), Nat.is_gt(r, 0n), Equal.sym(Bool, False{}, Nat.is_lt(Nat.add(k, x), k), hc_lt_add_r_false(k, x)))

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

# (That the drafter's rings are not copied again for an entry that has an image and gained
# nothing is today's saving, pinned by a Rust test and by no law: a later version may copy them
# more often, never less safely.)

# LAW HC.50: a ring image never serves a resume point past the tiles its entry held when it was
# taken: rows written since are never passed off as that prompt's.
law hc_ring_never_serves_past_its_tiles:
  for +i: Nat
  for +r: Nat
  {Bool.and(hc_ring_serves(i, r), Nat.is_lt(i, r)) == False{} : Bool}

def hc_ring_never_serves_past_its_tiles(i, r):
  hc_le_lt_excl(r, i)

# LAW HC.51: after its entry is cut, a ring image stands for no tile the entry lost.
law hc_ring_cover_within_cut:
  for +i: Nat
  for +k: Nat
  {Nat.is_le(hc_ring_cover_after_cut(i, k), k) == True{} : Bool}

def hc_ring_cover_within_cut(i, k):
  hc_min_le_r(i, k)

# LAW HC.52: nor for more than it did.
law hc_ring_cover_never_grows:
  for +i: Nat
  for +k: Nat
  {Nat.is_le(hc_ring_cover_after_cut(i, k), i) == True{} : Bool}

def hc_ring_cover_never_grows(i, k):
  hc_min_le_l(i, k)

# LAW HC.53: a save never counts on more free bytes than its budget has left ...
law hc_free_now_within_budget:
  for +f: Nat
  for +a: Nat
  {Nat.is_le(hc_free_now(f, a), f) == True{} : Bool}

def hc_free_now_within_budget(f, a):
  hc_min_le_l(f, a)

# LAW HC.54: ... nor on more than the host has available now: the cache gives its own entries
# back before it takes memory the host does not have.
law hc_free_now_within_host:
  for +f: Nat
  for +a: Nat
  {Nat.is_le(hc_free_now(f, a), a) == True{} : Bool}

def hc_free_now_within_host(f, a):
  hc_min_le_r(f, a)

# LAW HC.55: an entry with no kept position is never one to resume from.
law hc_no_position_not_resumable:
  for u: Unit
  {hc_resumable(0n) == False{} : Bool}

def hc_no_position_not_resumable(u):
  {==}

# LAW HC.56: a request drops nothing unsaved when nothing is unsaved.
law hc_nothing_unsaved_nothing_lost:
  for +c: Nat
  for +k: Nat
  {hc_loses_unsaved(False{}, c, k) == False{} : Bool}

def hc_nothing_unsaved_nothing_lost(c, k):
  {==}

# LAW HC.57: a request never pays for a copy unless it would drop kept positions that host
# memory lacks.
law hc_flush_only_for_a_loss:
  for +u: Bool
  for +r: Nat
  for +c: Nat
  for +k: Nat
  {Bool.and(hc_must_flush(u, r, c, k), Bool.not(hc_loses_unsaved(u, c, k))) == False{} : Bool}

def hc_flush_only_for_a_loss_aux(a: Bool, b: Bool) -> {Bool.and(Bool.and(a, b), Bool.not(a)) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      hc_and_false_r(b)

def hc_flush_only_for_a_loss(u, r, c, k):
  hc_flush_only_for_a_loss_aux(hc_loses_unsaved(u, c, k), Nat.is_gt(r, 0n))

# ============================================================================
# 12. Choices among candidates, guards of the steps above, and what a save adds
# ============================================================================
#
# The shell loops over entries, positions and segments; which one is taken, whether a step may
# run and how many bytes it is charged is decided here.

# ---- helper lemmas of this section (erased before Rust codegen) ----

def hc_absurd(b: Bool, h: {False{} == True{} : Bool}) -> {b == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: h

# c <= d  ->  c <= b + d
def hc_le_add_l(+c: Nat, +d: Nat, +b: Nat, +h: {Nat.is_le(c, d) == True{} : Bool}) -> {Nat.is_le(c, Nat.add(b, d)) == True{} : Bool}:
  match b:
    case 0n:
      h
    case 1n+bp:
      hc_le_step_r(c, Nat.add(bp, d), hc_le_add_l(c, d, bp, h))

# a <= b and c <= d  ->  a + c <= b + d
def hc_le_add_mono(+a: Nat, +b: Nat, +c: Nat, +d: Nat, +h1: {Nat.is_le(a, b) == True{} : Bool}, +h2: {Nat.is_le(c, d) == True{} : Bool}) -> {Nat.is_le(Nat.add(a, c), Nat.add(b, d)) == True{} : Bool}:
  match a b:
    case 0n 0n:
      h2
    case 0n 1n+bp:
      hc_le_add_l(c, d, 1n+bp, h2)
    case 1n+ap 0n:
      hc_absurd(Nat.is_le(Nat.add(1n+ap, c), Nat.add(0n, d)), h1)
    case 1n+ap 1n+bp:
      hc_le_add_mono(ap, bp, c, d, h1, h2)

# a <= b  ->  a * c <= b * c
def hc_le_mul_mono(+a: Nat, +b: Nat, +c: Nat, +h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.is_le(Nat.mul(a, c), Nat.mul(b, c)) == True{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      hc_zero_le(Nat.mul(1n+bp, c))
    case 1n+ap 0n:
      hc_absurd(Nat.is_le(Nat.mul(1n+ap, c), Nat.mul(0n, c)), h)
    case 1n+ap 1n+bp:
      hc_le_add_mono(c, c, Nat.mul(ap, c), Nat.mul(bp, c), hc_le_refl(c), hc_le_mul_mono(ap, bp, c, h))

def hc_gt_refl(n: Nat) -> {Nat.is_gt(n, n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      hc_gt_refl(p)

def hc_eq_refl(n: Nat) -> {Nat.is_eq(n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      hc_eq_refl(p)

# a > b and b > a never hold together
def hc_gt_gt_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_gt(a, b), Nat.is_gt(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      hc_gt_gt_excl(ap, bp)

# a > b and a <= b never hold together
def hc_gt_le_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_gt(a, b), Nat.is_le(a, b)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      hc_gt_le_excl(ap, bp)

# n < n + (1 + x)
def hc_lt_add_succ(n: Nat, x: Nat) -> {True{} == Nat.is_lt(n, Nat.add(n, 1n+x)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      hc_lt_add_succ(p, x)

# n - (n + x) == 0
def hc_sub_add_zero(n: Nat, x: Nat) -> {Nat.sub(n, Nat.add(n, x)) == 0n : Nat}:
  match n:
    case 0n:
      match x:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      hc_sub_add_zero(p, x)

# ---- 12.1 which place a request resumes from ----

# Is candidate a (a resume point at row `a_pos`, in an entry last used at tick `a_used`) a
# better place to resume from than candidate b? The further point, and among equal points the
# entry used more recently.
def hc_resume_better(+a_pos: Nat, a_used: Nat, +b_pos: Nat, b_used: Nat) -> Bool:
  Bool.or(Nat.is_gt(a_pos, b_pos), Bool.and(Nat.is_eq(a_pos, b_pos), Nat.is_gt(a_used, b_used)))

# ---- 12.2 which entry a save goes into ----

# Does the save of the held prompt go into this entry instead of the best one found so far
# (`best_common`: the tiles the prompt shares with that one, 0 while there is none)? Only an
# entry of this engine that can be read, that the prompt continues (`hc_continues`), and that
# shares more whole tiles with the prompt.
def hc_continue_pick(same_engine: Bool, readable: Bool, +common: Nat, tiles: Nat, best_common: Nat) -> Bool:
  Bool.and(Bool.and(same_engine, readable), Bool.and(hc_continues(common, tiles), Nat.is_gt(common, best_common)))

# Does the kept position after `state_tiles` whole tiles stay when its entry is cut to `keep`?
def hc_state_survives_cut(state_tiles: Nat, keep: Nat) -> Bool:
  Nat.is_le(state_tiles, keep)

# ---- 12.3 guards of a restore ----

# May a restore read the entry up to the state after `tiles` whole tiles? Only an entry that
# is readable, and only tiles it holds.
def hc_may_read(+e: HcEntry, tiles: Nat) -> Bool:
  Bool.and(hc_readable(e), Nat.is_le(tiles, hc_tiles(e)))

# The restore ended. `ok`: every byte reached the device.
def hc_read_done(e: HcEntry, ok: Bool) -> HcEntry:
  match ok:
    case True{}:
      hc_end_read(e)
    case False{}:
      hc_abort(e)

# Does a restore copy the entry's ring image to the device? Only on an engine that has rings
# (`ring_bytes`), only an image that serves the resume point (`hc_ring_serves`), and not when
# the device's rings are that image already.
def hc_ring_restore(ring_bytes: Nat, serves: Bool, on_device: Bool) -> Bool:
  Bool.and(Nat.is_gt(ring_bytes, 0n), Bool.and(serves, Bool.not(on_device)))

# First position whose drafter row counts as valid after a restore that resumes at
# `resume_pos`: the first one valid in the entry's image when the engine has rings and the
# image serves the resume point, else the resume point itself (the rings then hold rows of
# whatever ran before).
def hc_ring_floor(ring_bytes: Nat, has_image: Bool, image_from: Nat, resume_pos: Nat) -> Nat:
  match ring_bytes has_image:
    case 0n True{}:
      resume_pos
    case 0n False{}:
      resume_pos
    case 1n+p True{}:
      image_from
    case 1n+p False{}:
      resume_pos

# ---- 12.4 guards of a save ----

# May a save copy the rows of `upto` whole tiles? Only rows a prefill wrote
# (`hc_rows_saveable`), and only tiles the K/V caches hold.
def hc_save_ok(+upto: Nat, tile: Nat, held_rows: Nat, cache_tiles: Nat) -> Bool:
  Bool.and(hc_rows_saveable(upto, tile, held_rows), Nat.is_le(upto, cache_tiles))

# Are the rows of tile `index` of a batch (0: its first) complete in host memory, with `done`
# of the batch's pieces copied and `per_tile` pieces to a tile? Only when every piece of that
# tile and of the tiles before it is.
def hc_tile_copied(index: Nat, per_tile: Nat, done: Nat) -> Bool:
  Nat.is_le(Nat.mul(1n+index, per_tile), done)

# Did a copy leave its entry more whole tiles than it had?
def hc_rows_added(before: Nat, after: Nat) -> Bool:
  Nat.is_lt(before, after)

# Does the entry hold every tile a batch of rows set out to reach?
def hc_rows_complete(tiles_now: Nat, target: Nat) -> Bool:
  Nat.is_le(target, tiles_now)

# Is a copy of `all` pieces whole, with `done` of them finished?
def hc_copy_whole(done: Nat, all: Nat) -> Bool:
  Nat.is_le(all, done)

# Does a save copy the state at a kept position? Only one the entry lacks, at a position its
# rows reach.
def hc_state_wanted(already: Bool, state_tiles: Nat, entry_tiles: Nat) -> Bool:
  Bool.and(Bool.not(already), Nat.is_le(state_tiles, entry_tiles))

# Is the state a save just copied kept by its entry? Only a whole copy, at a position inside
# the rows the entry holds.
def hc_state_kept(done: Nat, all: Nat, state_tiles: Nat, entry_tiles: Nat) -> Bool:
  Bool.and(hc_copy_whole(done, all), Nat.is_le(state_tiles, entry_tiles))

# May the kept position after `state_tiles` tiles leave an entry that has one too many? Never
# the one the save just reached (`last_tiles`): the next turn resumes there.
def hc_state_may_leave(state_tiles: Nat, last_tiles: Nat) -> Bool:
  Bool.not(Nat.is_eq(state_tiles, last_tiles))

# Does a save copy the drafter's rings? Only on an engine that has rings, only when everything
# before them was copied whole (a save that was cut short gives the card up at once), and only
# when the entry's image is stale (`hc_ring_stale`).
def hc_ring_save(ring_bytes: Nat, complete: Bool, has_image: Bool, grew: Bool) -> Bool:
  Bool.and(Nat.is_gt(ring_bytes, 0n), Bool.and(complete, hc_ring_stale(has_image, grew)))

# May the server's idle work take the engine? Only a ready engine that no request is using.
def hc_idle_may_take(active: Nat, ready: Bool) -> Bool:
  Bool.and(ready, Nat.is_eq(active, 0n))

# May the K/V caches of an engine be copied tile by tile, and if not, why not?
type HcLayout is Data:
  HcLayoutOk{}
  HcLayoutEmpty{}
  HcLayoutTile{}
  HcLayoutSize{}

def hc_layout_pick(some: Bool, tiled: Bool, sized: Bool) -> HcLayout:
  match some:
    case False{}:
      HcLayoutEmpty{}
    case True{}:
      match tiled:
        case False{}:
          HcLayoutTile{}
        case True{}:
          match sized:
            case True{}:
              HcLayoutOk{}
            case False{}:
              HcLayoutSize{}

# `caches` caches of `cache_bytes` each, `chunks` chunks of `chunk_rows` rows, `heads` heads
# of rows of `row_bytes`; rows are written in blocks of `block` rows. A tile's rows are one
# device range per cache and head only when the tile divides a chunk and is whole blocks, and
# the ranges are where the copies expect them only when a cache is exactly chunks x heads x
# rows x row bytes.
def hc_layout(caches: Nat, +chunks: Nat, +chunk_rows: Nat, +heads: Nat, +row_bytes: Nat, +tile: Nat, block: Nat, cache_bytes: Nat) -> HcLayout:
  hc_layout_pick(
    Bool.and(Nat.is_gt(caches, 0n), Bool.and(Nat.is_gt(tile, 0n), Bool.and(Nat.is_gt(chunks, 0n), Bool.and(Nat.is_gt(heads, 0n), Nat.is_gt(row_bytes, 0n))))),
    Bool.and(Nat.is_eq(Nat.mod(chunk_rows, tile), 0n), Nat.is_eq(Nat.mod(tile, block), 0n)),
    Nat.is_eq(cache_bytes, Nat.mul(Nat.mul(Nat.mul(chunks, heads), chunk_rows), row_bytes))
  )

def hc_layout_ok(l: HcLayout) -> Bool:
  match l:
    case HcLayoutOk{}:
      True{}
    case HcLayoutEmpty{}:
      False{}
    case HcLayoutTile{}:
      False{}
    case HcLayoutSize{}:
      False{}

# ---- 12.5 what a save adds to its entry ----

# Tiles a save to `upto` whole tiles adds to an entry that holds `have` of them.
def hc_new_tiles(upto: Nat, have: Nat) -> Nat:
  Nat.sub(upto, have)

def hc_ring_part(ring_copy: Bool, ring_charge: Nat) -> Nat:
  match ring_copy:
    case True{}:
      ring_charge
    case False{}:
      0n

# Bytes a save adds to the entry it writes into: the rows of the tiles the entry lacks, the
# state at the kept positions it lacks (`held_states` of the `kept` positions are in the entry
# already, and in the ledger), and a ring image when one is copied.
def hc_save_need(upto: Nat, have: Nat, tile_charge: Nat, kept: Nat, held_states: Nat, state_charge: Nat, ring_copy: Bool, ring_charge: Nat) -> Nat:
  Nat.add(Nat.add(Nat.mul(hc_new_tiles(upto, have), tile_charge), Nat.mul(Nat.sub(kept, held_states), state_charge)), hc_ring_part(ring_copy, ring_charge))

# ---- 12.6 spare segments and the host ----
#
# A segment an entry gave up during a save stays mapped until the save ends, for the save to
# take again. It is taken again only by an allocation of exactly its size.

# Bytes of the spare segments of one size that an allocation of `wanted_segments` of that
# size takes again.
def hc_spare_credit(+spare_segments: Nat, +wanted_segments: Nat, seg_charge: Nat) -> Nat:
  Nat.mul(Nat.min(spare_segments, wanted_segments), seg_charge)

# Bytes a save may take now without evicting: the budget's free bytes, and no more than the
# host has available plus what the save gave back and takes again.
def hc_take_now(+budget_free: Nat, +host_available: Nat, reusable: Nat) -> Nat:
  hc_free_now(budget_free, Nat.add(host_available, reusable))

# Do the spare segments an allocation cannot take again go back to the system first? When the
# host cannot give what the allocation needs beyond what it takes again, and there are some.
def hc_spares_return(need: Nat, host_available: Nat, reusable: Nat, unusable: Nat) -> Bool:
  Bool.and(Nat.is_gt(unusable, 0n), Nat.is_lt(Nat.add(host_available, reusable), need))

# ---- 12.7 laws ----

def hc_resume_better_same(+p: Nat, au: Nat, bu: Nat) -> {hc_resume_better(p, au, p, bu) == Nat.is_gt(au, bu) : Bool}:
  match p:
    case 0n: {==}
    case 1n+pp:
      hc_resume_better_same(pp, au, bu)

# LAW HC.58: no candidate is a better place to resume from than itself.
law hc_resume_better_irreflexive:
  for +p: Nat
  for +u: Nat
  {hc_resume_better(p, u, p, u) == False{} : Bool}

def hc_resume_better_irreflexive(p, u):
  %Equal.sym(Bool, hc_resume_better(p, u, p, u), Nat.is_gt(u, u), hc_resume_better_same(p, u, u)) : {_ == False{} : Bool}
  hc_gt_refl(u)

# (That the further resume point wins is today's choice, pinned by a Rust test and by no law:
# a later version may prefer a nearer point that costs less to copy.)

# (That equal resume points are told apart by the use order alone is today's choice too.)

# LAW HC.61: two candidates are never each preferred to the other.
law hc_resume_better_asymmetric:
  for +ap: Nat
  for +au: Nat
  for +bp: Nat
  for +bu: Nat
  {Bool.and(hc_resume_better(ap, au, bp, bu), hc_resume_better(bp, bu, ap, au)) == False{} : Bool}

def hc_resume_better_asymmetric(ap, au, bp, bu):
  match ap bp:
    case 0n 0n:
      hc_gt_gt_excl(au, bu)
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+app 1n+bpp:
      hc_resume_better_asymmetric(app, au, bpp, bu)

# LAW HC.62: a save never goes into an entry made by an engine of another identity.
law hc_other_engine_never_continued:
  for +r: Bool
  for +c: Nat
  for +t: Nat
  for +b: Nat
  {hc_continue_pick(False{}, r, c, t, b) == False{} : Bool}

def hc_other_engine_never_continued(r, c, t, b):
  {==}

# LAW HC.63: nor into an entry that is being written or read.
law hc_unreadable_never_continued:
  for +s: Bool
  for +c: Nat
  for +t: Nat
  for +b: Nat
  {hc_continue_pick(s, False{}, c, t, b) == False{} : Bool}

def hc_unreadable_never_continued(s, c, t, b):
  match s:
    case True{}: {==}
    case False{}: {==}

def hc_pick_needs_aux(g: Bool, x: Bool, y: Bool) -> {Bool.and(Bool.and(g, Bool.and(x, y)), Bool.not(x)) == False{} : Bool}:
  match g:
    case False{}:
      {==}
    case True{}:
      match x:
        case False{}:
          {==}
        case True{}:
          hc_and_false_r(y)

# LAW HC.64: nor into an entry the prompt does not continue.
law hc_pick_only_what_continues:
  for +s: Bool
  for +r: Bool
  for +c: Nat
  for +t: Nat
  for +b: Nat
  {Bool.and(hc_continue_pick(s, r, c, t, b), Bool.not(hc_continues(c, t))) == False{} : Bool}

def hc_pick_only_what_continues(s, r, c, t, b):
  hc_pick_needs_aux(Bool.and(s, r), hc_continues(c, t), Nat.is_gt(c, b))

def hc_pick_more_aux(g: Bool, x: Bool, y: Bool, z: Bool, h: {Bool.and(y, z) == False{} : Bool}) -> {Bool.and(Bool.and(g, Bool.and(x, y)), z) == False{} : Bool}:
  match g:
    case False{}:
      {==}
    case True{}:
      match x:
        case False{}:
          {==}
        case True{}:
          h

# (That the entry sharing the most whole tiles is the one continued is today's choice, pinned
# by a Rust test and by no law.)

# LAW HC.66: a kept position past the tiles its entry is cut to never stays.
law hc_state_past_cut_never_survives:
  for +k: Nat
  for +c: Nat
  {Bool.and(hc_state_survives_cut(k, c), Nat.is_lt(c, k)) == False{} : Bool}

def hc_state_past_cut_never_survives(k, c):
  hc_le_lt_excl(k, c)

# LAW HC.67: a vacant entry is never read.
law hc_vacant_never_read:
  for +t: Nat
  {hc_may_read(HcVacant{}, t) == False{} : Bool}

def hc_vacant_never_read(t):
  {==}

# LAW HC.68: an entry that is being written is never read, whatever it was before.
law hc_written_entry_never_read:
  for +e: HcEntry
  for +u: Nat
  for +t: Nat
  {hc_may_read(hc_begin_write(e, u), t) == False{} : Bool}

def hc_written_entry_never_read(e, u, t):
  match e:
    case HcVacant{}: {==}
    case HcHeld{h}: {==}
    case HcWriting{h, w}: {==}
    case HcReading{h}: {==}

# LAW HC.69: a restore never reads tiles past the ones its entry holds.
law hc_read_within_held_tiles:
  for +e: HcEntry
  for +t: Nat
  {Bool.and(hc_may_read(e, t), Nat.is_lt(hc_tiles(e), t)) == False{} : Bool}

def hc_read_within_held_tiles(e, t):
  hc_and3_excl(hc_readable(e), Nat.is_le(t, hc_tiles(e)), Nat.is_lt(hc_tiles(e), t), hc_le_lt_excl(t, hc_tiles(e)))

# LAW HC.70: after a restore that failed, the entry is never readable.
law hc_failed_restore_never_readable:
  for +e: HcEntry
  {hc_readable(hc_read_done(hc_begin_read(e), False{})) == False{} : Bool}

def hc_failed_restore_never_readable(e):
  match e:
    case HcVacant{}: {==}
    case HcHeld{h}: {==}
    case HcWriting{h, w}: {==}
    case HcReading{h}: {==}

# LAW HC.71: the end of a restore never leaves an entry more rows than it had.
law hc_read_done_never_adds_rows:
  for +e: HcEntry
  for +ok: Bool
  {Nat.is_le(hc_tiles(hc_read_done(e, ok)), hc_tiles(e)) == True{} : Bool}

def hc_read_done_never_adds_rows(e, ok):
  match e:
    case HcVacant{}:
      match ok:
        case True{}: {==}
        case False{}: {==}
    case HcHeld{h}:
      match ok:
        case True{}:
          hc_le_refl(h)
        case False{}:
          hc_le_refl(h)
    case HcWriting{h, w}:
      match ok:
        case True{}:
          hc_le_refl(h)
        case False{}:
          hc_zero_le(h)
    case HcReading{h}:
      match ok:
        case True{}:
          hc_le_refl(h)
        case False{}:
          hc_zero_le(h)

# LAW HC.72: an engine without rings never copies a ring image to the device.
law hc_no_rings_nothing_restored:
  for +s: Bool
  for +o: Bool
  {hc_ring_restore(0n, s, o) == False{} : Bool}

def hc_no_rings_nothing_restored(s, o):
  {==}

# LAW HC.73: a ring image that does not serve the resume point is never copied to the device.
law hc_unserving_ring_never_restored:
  for +b: Nat
  for +o: Bool
  {hc_ring_restore(b, False{}, o) == False{} : Bool}

def hc_unserving_ring_never_restored(b, o):
  hc_and_false_r(Nat.is_gt(b, 0n))

# LAW HC.74: without an image that serves the resume point, no drafter row before the resume
# point ever counts as valid ...
law hc_ring_floor_without_image:
  for +b: Nat
  for +f: Nat
  for +r: Nat
  {Nat.is_le(r, hc_ring_floor(b, False{}, f, r)) == True{} : Bool}

def hc_ring_floor_without_image(b, f, r):
  match b:
    case 0n:
      hc_le_refl(r)
    case 1n+_:
      hc_le_refl(r)

# LAW HC.75: ... nor on an engine without rings.
law hc_ring_floor_without_rings:
  for +i: Bool
  for +f: Nat
  for +r: Nat
  {Nat.is_le(r, hc_ring_floor(0n, i, f, r)) == True{} : Bool}

def hc_ring_floor_without_rings(i, f, r):
  match i:
    case True{}:
      hc_le_refl(r)
    case False{}:
      hc_le_refl(r)

# LAW HC.76: with an image, no row before the first one valid in it ever counts as valid.
law hc_ring_floor_within_image:
  for +b: Nat
  for +f: Nat
  for +r: Nat
  {Nat.is_le(f, hc_ring_floor(1n+b, True{}, f, r)) == True{} : Bool}

def hc_ring_floor_within_image(b, f, r):
  hc_le_refl(f)

# LAW HC.77: a save never copies rows past the ones a prefill wrote ...
law hc_save_only_prefilled_rows:
  for +u: Nat
  for +tile: Nat
  for +h: Nat
  for +c: Nat
  {Bool.and(hc_save_ok(u, tile, h, c), Nat.is_lt(h, Nat.mul(u, tile))) == False{} : Bool}

def hc_save_only_prefilled_rows(u, tile, h, c):
  hc_tile_below_and(hc_rows_saveable(u, tile, h), Nat.is_le(u, c), Nat.is_lt(h, Nat.mul(u, tile)), hc_le_lt_excl(Nat.mul(u, tile), h))

# LAW HC.78: ... nor tiles the K/V caches do not hold.
law hc_save_within_caches:
  for +u: Nat
  for +tile: Nat
  for +h: Nat
  for +c: Nat
  {Bool.and(hc_save_ok(u, tile, h, c), Nat.is_lt(c, u)) == False{} : Bool}

def hc_save_within_caches(u, tile, h, c):
  hc_and3_excl(hc_rows_saveable(u, tile, h), Nat.is_le(u, c), Nat.is_lt(c, u), hc_le_lt_excl(u, c))

# LAW HC.79: a tile never counts as copied while a piece of it, or of a tile before it in the
# batch, is missing.
law hc_tile_short_of_pieces_never_copied:
  for +i: Nat
  for +p: Nat
  for +d: Nat
  {Bool.and(hc_tile_copied(i, p, d), Nat.is_lt(d, Nat.mul(1n+i, p))) == False{} : Bool}

def hc_tile_short_of_pieces_never_copied(i, p, d):
  hc_le_lt_excl(Nat.mul(1n+i, p), d)

# LAW HC.80: an entry that holds no more tiles than before never counts as grown: no device
# tile is taken for its copy on the strength of a copy that added none.
law hc_no_new_rows_never_added:
  for +b: Nat
  for +a: Nat
  {Bool.and(hc_rows_added(b, a), Nat.is_le(a, b)) == False{} : Bool}

def hc_no_new_rows_never_added(b, a):
  hc_lt_le_excl(b, a)

# LAW HC.81: rows that stopped short of their target are never taken for complete.
law hc_short_rows_never_complete:
  for +n: Nat
  for +t: Nat
  {Bool.and(hc_rows_complete(n, t), Nat.is_lt(n, t)) == False{} : Bool}

def hc_short_rows_never_complete(n, t):
  hc_le_lt_excl(t, n)

# LAW HC.82: a copy with a piece missing is never whole.
law hc_part_copy_never_whole:
  for +d: Nat
  for +a: Nat
  {Bool.and(hc_copy_whole(d, a), Nat.is_lt(d, a)) == False{} : Bool}

def hc_part_copy_never_whole(d, a):
  hc_le_lt_excl(a, d)

# (That a state the entry already holds is not copied again is today's saving, pinned by Rust
# tests and by no law: `end_state` replaces a state it is given again. What a law states is
# that such a state is never charged, HC.92.)

# LAW HC.83: the state at a position past the rows the entry holds is never copied.
law hc_state_past_rows_never_wanted:
  for +a: Bool
  for +k: Nat
  for +t: Nat
  {Bool.and(hc_state_wanted(a, k, t), Nat.is_lt(t, k)) == False{} : Bool}

def hc_state_past_rows_never_wanted(a, k, t):
  hc_and3_excl(Bool.not(a), Nat.is_le(k, t), Nat.is_lt(t, k), hc_le_lt_excl(k, t))

# LAW HC.84: a state whose copy has a piece missing is never kept ...
law hc_part_state_never_kept:
  for +d: Nat
  for +a: Nat
  for +k: Nat
  for +t: Nat
  {Bool.and(hc_state_kept(d, a, k, t), Nat.is_lt(d, a)) == False{} : Bool}

def hc_part_state_never_kept(d, a, k, t):
  hc_tile_below_and(hc_copy_whole(d, a), Nat.is_le(k, t), Nat.is_lt(d, a), hc_le_lt_excl(a, d))

# LAW HC.85: ... nor one at a position past the rows the entry holds.
law hc_state_past_rows_never_kept:
  for +d: Nat
  for +a: Nat
  for +k: Nat
  for +t: Nat
  {Bool.and(hc_state_kept(d, a, k, t), Nat.is_lt(t, k)) == False{} : Bool}

def hc_state_past_rows_never_kept(d, a, k, t):
  hc_and3_excl(hc_copy_whole(d, a), Nat.is_le(k, t), Nat.is_lt(t, k), hc_le_lt_excl(k, t))

# LAW HC.86: the position a save just reached never leaves its entry to make room for others.
law hc_last_position_never_leaves:
  for +k: Nat
  {hc_state_may_leave(k, k) == False{} : Bool}

def hc_last_position_never_leaves(k):
  %Equal.sym(Bool, Nat.is_eq(k, k), True{}, hc_eq_refl(k)) : {Bool.not(_) == False{} : Bool}
  {==}

# LAW HC.87: an engine without rings never copies a ring image out.
law hc_no_rings_nothing_saved:
  for +c: Bool
  for +i: Bool
  for +g: Bool
  {hc_ring_save(0n, c, i, g) == False{} : Bool}

def hc_no_rings_nothing_saved(c, i, g):
  {==}

# LAW HC.88: a save that was cut short never goes on to copy the rings: it gives the card up.
law hc_cut_short_save_copies_no_ring:
  for +b: Nat
  for +i: Bool
  for +g: Bool
  {hc_ring_save(b, False{}, i, g) == False{} : Bool}

def hc_cut_short_save_copies_no_ring(b, i, g):
  hc_and_false_r(Nat.is_gt(b, 0n))

# LAW HC.89: idle work never takes the engine while a request is using it ...
law hc_idle_never_takes_a_busy_engine:
  for +a: Nat
  for +r: Bool
  {hc_idle_may_take(1n+a, r) == False{} : Bool}

def hc_idle_never_takes_a_busy_engine(a, r):
  match r:
    case True{}: {==}
    case False{}: {==}

# LAW HC.90: ... nor an engine that is not ready.
law hc_idle_never_takes_an_unready_engine:
  for +a: Nat
  {hc_idle_may_take(a, False{}) == False{} : Bool}

def hc_idle_never_takes_an_unready_engine(a):
  {==}

def hc_ring_part_le(c: Bool, r: Nat) -> {Nat.is_le(hc_ring_part(c, r), r) == True{} : Bool}:
  match c:
    case True{}:
      hc_le_refl(r)
    case False{}:
      hc_zero_le(r)

# LAW HC.91: what a save is charged never exceeds the full charge of what it leaves in the
# entry: every tile up to its target, the state at every kept position, and a ring image.
law hc_save_need_within_full_charge:
  for +u: Nat
  for +h: Nat
  for +tc: Nat
  for +k: Nat
  for +a: Nat
  for +sc: Nat
  for +f: Bool
  for +rc: Nat
  {Nat.is_le(hc_save_need(u, h, tc, k, a, sc, f, rc), Nat.add(Nat.add(Nat.mul(u, tc), Nat.mul(k, sc)), rc)) == True{} : Bool}

def hc_save_need_within_full_charge(u, h, tc, k, a, sc, f, rc):
  hc_le_add_mono(
    Nat.add(Nat.mul(Nat.sub(u, h), tc), Nat.mul(Nat.sub(k, a), sc)),
    Nat.add(Nat.mul(u, tc), Nat.mul(k, sc)),
    hc_ring_part(f, rc),
    rc,
    hc_le_add_mono(
      Nat.mul(Nat.sub(u, h), tc),
      Nat.mul(u, tc),
      Nat.mul(Nat.sub(k, a), sc),
      Nat.mul(k, sc),
      hc_le_mul_mono(Nat.sub(u, h), u, tc, hc_sub_le_self(u, h)),
      hc_le_mul_mono(Nat.sub(k, a), k, sc, hc_sub_le_self(k, a))
    ),
    hc_ring_part_le(f, rc)
  )

# LAW HC.92: the states an entry already holds add nothing to what a save is charged: with
# every kept position held, the charge is that of a save with no kept position at all.
law hc_held_states_never_charged:
  for +u: Nat
  for +h: Nat
  for +tc: Nat
  for +k: Nat
  for +x: Nat
  for +sc: Nat
  for +f: Bool
  for +rc: Nat
  {hc_save_need(u, h, tc, k, Nat.add(k, x), sc, f, rc) == hc_save_need(u, h, tc, 0n, 0n, sc, f, rc) : Nat}

def hc_held_states_never_charged(u, h, tc, k, x, sc, f, rc):
  %Equal.sym(Nat, Nat.sub(k, Nat.add(k, x)), 0n, hc_sub_add_zero(k, x)) : {Nat.add(Nat.add(Nat.mul(Nat.sub(u, h), tc), Nat.mul(_, sc)), hc_ring_part(f, rc)) == Nat.add(Nat.add(Nat.mul(Nat.sub(u, h), tc), Nat.mul(0n, sc)), hc_ring_part(f, rc)) : Nat}
  {==}

def hc_add_zero_le(n: Nat) -> {Nat.is_le(Nat.add(n, 0n), n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      hc_add_zero_le(p)

# LAW HC.93: the tiles a save adds never take its entry past its target, nor past what the
# entry held: no row beyond the ones the save was allowed to copy is ever asked for.
law hc_new_tiles_within_target:
  for +h: Nat
  for +u: Nat
  {Nat.is_le(Nat.add(h, hc_new_tiles(u, h)), Nat.max(h, u)) == True{} : Bool}

def hc_new_tiles_within_target(h, u):
  match h u:
    case 0n 0n: {==}
    case 0n 1n+up:
      hc_le_refl(up)
    case 1n+hp 0n:
      hc_add_zero_le(hp)
    case 1n+hp 1n+up:
      hc_new_tiles_within_target(hp, up)

# LAW HC.94: a save never takes more spare bytes again than the spare segments of that size
# hold ...
law hc_spare_credit_within_spares:
  for +s: Nat
  for +w: Nat
  for +c: Nat
  {Nat.is_le(hc_spare_credit(s, w, c), Nat.mul(s, c)) == True{} : Bool}

def hc_spare_credit_within_spares(s, w, c):
  hc_le_mul_mono(Nat.min(s, w), s, c, hc_min_le_l(s, w))

# LAW HC.95: ... nor more than the allocation asks for: a spare segment it cannot take again
# is never counted as memory it has.
law hc_spare_credit_within_wanted:
  for +s: Nat
  for +w: Nat
  for +c: Nat
  {Nat.is_le(hc_spare_credit(s, w, c), Nat.mul(w, c)) == True{} : Bool}

def hc_spare_credit_within_wanted(s, w, c):
  hc_le_mul_mono(Nat.min(s, w), w, c, hc_min_le_r(s, w))

# LAW HC.96: what a save may take never exceeds the budget's free bytes ...
law hc_take_now_within_budget:
  for +f: Nat
  for +a: Nat
  for +r: Nat
  {Nat.is_le(hc_take_now(f, a, r), f) == True{} : Bool}

def hc_take_now_within_budget(f, a, r):
  hc_min_le_l(f, Nat.add(a, r))

# LAW HC.97: ... nor what the host has available plus what the save itself gave back and
# takes again.
law hc_take_now_within_host_and_reuse:
  for +f: Nat
  for +a: Nat
  for +r: Nat
  {Nat.is_le(hc_take_now(f, a, r), Nat.add(a, r)) == True{} : Bool}

def hc_take_now_within_host_and_reuse(f, a, r):
  hc_min_le_r(f, Nat.add(a, r))

# LAW HC.98: when the host cannot give what an allocation needs, the spare segments the
# allocation cannot take again always go back to the system first: no entry is evicted, no save
# refused and no fresh memory mapped while the store sits on memory it cannot use.
law hc_unusable_spares_go_first:
  for +a: Nat
  for +r: Nat
  for +x: Nat
  for +u: Nat
  {hc_spares_return(Nat.add(Nat.add(a, r), 1n+x), a, r, 1n+u) == True{} : Bool}

def hc_unusable_spares_go_first(a, r, x, u):
  %hc_lt_add_succ(Nat.add(a, r), x) : {Bool.and(True{}, _) == True{} : Bool}
  {==}

# LAW HC.99: nothing is returned when there is nothing the allocation cannot use.
law hc_no_unusable_spares_nothing_returned:
  for +n: Nat
  for +a: Nat
  for +r: Nat
  {hc_spares_return(n, a, r, 0n) == False{} : Bool}

def hc_no_unusable_spares_nothing_returned(n, a, r):
  {==}

def hc_layout_tile_aux(some: Bool, t1: Bool, t2: Bool, sized: Bool) -> {Bool.and(hc_layout_ok(hc_layout_pick(some, Bool.and(t1, t2), sized)), Bool.not(t1)) == False{} : Bool}:
  match some:
    case False{}:
      {==}
    case True{}:
      match t1:
        case False{}:
          {==}
        case True{}:
          hc_and_false_r(hc_layout_ok(hc_layout_pick(True{}, t2, sized)))

# LAW HC.100: caches whose chunks the prefill tile does not divide are never copied tile by
# tile: a tile's rows would not be one range per head.
law hc_layout_tile_divides_a_chunk:
  for +c: Nat
  for +ch: Nat
  for +cr: Nat
  for +h: Nat
  for +rb: Nat
  for +t: Nat
  for +b: Nat
  for +cb: Nat
  {Bool.and(hc_layout_ok(hc_layout(c, ch, cr, h, rb, t, b, cb)), Bool.not(Nat.is_eq(Nat.mod(cr, t), 0n))) == False{} : Bool}

def hc_layout_tile_divides_a_chunk(c, ch, cr, h, rb, t, b, cb):
  hc_layout_tile_aux(
    Bool.and(Nat.is_gt(c, 0n), Bool.and(Nat.is_gt(t, 0n), Bool.and(Nat.is_gt(ch, 0n), Bool.and(Nat.is_gt(h, 0n), Nat.is_gt(rb, 0n))))),
    Nat.is_eq(Nat.mod(cr, t), 0n),
    Nat.is_eq(Nat.mod(t, b), 0n),
    Nat.is_eq(cb, Nat.mul(Nat.mul(Nat.mul(ch, h), cr), rb))
  )

def hc_layout_size_aux(some: Bool, tiled: Bool, sized: Bool) -> {Bool.and(hc_layout_ok(hc_layout_pick(some, tiled, sized)), Bool.not(sized)) == False{} : Bool}:
  match some:
    case False{}:
      {==}
    case True{}:
      match tiled:
        case False{}:
          {==}
        case True{}:
          match sized:
            case True{}:
              {==}
            case False{}:
              {==}

# LAW HC.101: nor caches of any other size than chunks x heads x rows x row bytes: no copy
# ever addresses a range the layout did not account for.
law hc_layout_cache_size_matches:
  for +c: Nat
  for +ch: Nat
  for +cr: Nat
  for +h: Nat
  for +rb: Nat
  for +t: Nat
  for +b: Nat
  for +cb: Nat
  {Bool.and(hc_layout_ok(hc_layout(c, ch, cr, h, rb, t, b, cb)), Bool.not(Nat.is_eq(cb, Nat.mul(Nat.mul(Nat.mul(ch, h), cr), rb)))) == False{} : Bool}

def hc_layout_cache_size_matches(c, ch, cr, h, rb, t, b, cb):
  hc_layout_size_aux(
    Bool.and(Nat.is_gt(c, 0n), Bool.and(Nat.is_gt(t, 0n), Bool.and(Nat.is_gt(ch, 0n), Bool.and(Nat.is_gt(h, 0n), Nat.is_gt(rb, 0n))))),
    Bool.and(Nat.is_eq(Nat.mod(cr, t), 0n), Nat.is_eq(Nat.mod(t, b), 0n)),
    Nat.is_eq(cb, Nat.mul(Nat.mul(Nat.mul(ch, h), cr), rb))
  )

# LAW HC.102: an engine without K/V caches has nothing to copy.
law hc_layout_no_caches_never_ok:
  for +ch: Nat
  for +cr: Nat
  for +h: Nat
  for +rb: Nat
  for +t: Nat
  for +b: Nat
  for +cb: Nat
  {hc_layout_ok(hc_layout(0n, ch, cr, h, rb, t, b, cb)) == False{} : Bool}

def hc_layout_no_caches_never_ok(ch, cr, h, rb, t, b, cb):
  {==}

# ============================================================================
# 13. More helper lemmas (erased before Rust codegen)
# ============================================================================

def hc_add_zero_r(n: Nat) -> {Nat.add(n, 0n) == n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.add(p, 0n), p, hc_add_zero_r(p)) : {1n+_ == 1n+p : Nat}
      {==}

def hc_add_succ_r(+a: Nat, +b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.add(p, 1n+b), 1n+Nat.add(p, b), hc_add_succ_r(p, b)) : {1n+_ == 1n+(1n+Nat.add(p, b)) : Nat}
      {==}

def hc_add_comm(+a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  match a:
    case 0n:
      Equal.sym(Nat, Nat.add(b, 0n), b, hc_add_zero_r(b))
    case 1n+p:
      %Equal.sym(Nat, Nat.add(b, 1n+p), 1n+Nat.add(b, p), hc_add_succ_r(b, p)) : {1n+Nat.add(p, b) == _ : Nat}
      %hc_add_comm(p, b) : {1n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

# a + (b + c) == (a + b) + c
def hc_add_assoc_r(+a: Nat, +b: Nat, +c: Nat) -> {Nat.add(a, Nat.add(b, c)) == Nat.add(Nat.add(a, b), c) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %hc_add_assoc_r(p, b, c) : {1n+Nat.add(p, Nat.add(b, c)) == 1n+_ : Nat}
      {==}

# ((a + b) + c) + p == ((p + a) + b) + c
def hc_add_shuffle3(+a: Nat, +b: Nat, +c: Nat, +p: Nat) -> {Nat.add(Nat.add(Nat.add(a, b), c), p) == Nat.add(Nat.add(Nat.add(p, a), b), c) : Nat}:
  %hc_add_assoc_r(p, a, b) : {Nat.add(Nat.add(Nat.add(a, b), c), p) == Nat.add(_, c) : Nat}
  %hc_add_assoc_r(p, Nat.add(a, b), c) : {Nat.add(Nat.add(Nat.add(a, b), c), p) == _ : Nat}
  hc_add_comm(Nat.add(Nat.add(a, b), c), p)

# (a + b) + p == (p + a) + b
def hc_add_shuffle2(+a: Nat, +b: Nat, +p: Nat) -> {Nat.add(Nat.add(a, b), p) == Nat.add(Nat.add(p, a), b) : Nat}:
  %hc_add_assoc_r(p, a, b) : {Nat.add(Nat.add(a, b), p) == _ : Nat}
  hc_add_comm(Nat.add(a, b), p)

# (a + b) - a == b
def hc_add_sub_cancel(a: Nat, b: Nat) -> {Nat.sub(Nat.add(a, b), a) == b : Nat}:
  match a:
    case 0n:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      hc_add_sub_cancel(p, b)

def hc_le_trans(+a: Nat, +b: Nat, +c: Nat, +h1: {Nat.is_le(a, b) == True{} : Bool}, +h2: {Nat.is_le(b, c) == True{} : Bool}) -> {Nat.is_le(a, c) == True{} : Bool}:
  match a b c:
    case 0n 0n 0n: {==}
    case 0n 0n 1n+_: {==}
    case 0n 1n+_ 0n: {==}
    case 0n 1n+_ 1n+_: {==}
    case 1n+ap 0n 0n:
      h1
    case 1n+ap 0n 1n+cp:
      hc_absurd(Nat.is_le(1n+ap, 1n+cp), h1)
    case 1n+ap 1n+bp 0n:
      h2
    case 1n+ap 1n+bp 1n+cp:
      hc_le_trans(ap, bp, cp, h1, h2)

# 1 + i <= i + (1 + e)
def hc_succ_le_add(i: Nat, e: Nat) -> {Nat.is_le(1n+i, Nat.add(i, 1n+e)) == True{} : Bool}:
  match i:
    case 0n:
      hc_zero_le(e)
    case 1n+p:
      hc_succ_le_add(p, e)

# An extent x that fits one unit of size s, placed in unit i of i + 1 + e units, ends inside
# them: x <= s  ->  x + i * s <= (i + 1 + e) * s
def hc_nest(+x: Nat, +s: Nat, +i: Nat, +e: Nat, +h: {Nat.is_le(x, s) == True{} : Bool}) -> {Nat.is_le(Nat.add(x, Nat.mul(i, s)), Nat.mul(Nat.add(i, 1n+e), s)) == True{} : Bool}:
  hc_le_trans(
    Nat.add(x, Nat.mul(i, s)),
    Nat.mul(1n+i, s),
    Nat.mul(Nat.add(i, 1n+e), s),
    hc_le_add_mono(x, s, Nat.mul(i, s), Nat.mul(i, s), h, hc_le_refl(Nat.mul(i, s))),
    hc_le_mul_mono(1n+i, Nat.add(i, 1n+e), s, hc_succ_le_add(i, e))
  )

# (a + b) * c == a * c + b * c
def hc_mul_add_distrib(+a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(Nat.add(a, b), c) == Nat.add(Nat.mul(a, c), Nat.mul(b, c)) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(Nat.add(p, b), c), Nat.add(Nat.mul(p, c), Nat.mul(b, c)), hc_mul_add_distrib(p, b, c)) : {Nat.add(c, _) == Nat.add(Nat.add(c, Nat.mul(p, c)), Nat.mul(b, c)) : Nat}
      hc_add_assoc_r(c, Nat.mul(p, c), Nat.mul(b, c))

# a * (b * c) == (a * b) * c
def hc_mul_assoc_r(+a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(a, Nat.mul(b, c)) == Nat.mul(Nat.mul(a, b), c) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(p, Nat.mul(b, c)), Nat.mul(Nat.mul(p, b), c), hc_mul_assoc_r(p, b, c)) : {Nat.add(Nat.mul(b, c), _) == Nat.mul(Nat.add(b, Nat.mul(p, b)), c) : Nat}
      Equal.sym(Nat, Nat.mul(Nat.add(b, Nat.mul(p, b)), c), Nat.add(Nat.mul(b, c), Nat.mul(Nat.mul(p, b), c)), hc_mul_add_distrib(b, Nat.mul(p, b), c))

def hc_and_not_l(x: Bool, y: Bool) -> {Bool.and(Bool.and(x, y), Bool.not(x)) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      hc_and_false_r(y)

def hc_and_not_r(x: Bool, y: Bool) -> {Bool.and(Bool.and(x, y), Bool.not(y)) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      match y:
        case False{}: {==}
        case True{}: {==}

# ============================================================================
# 14. The control groups a process is in
# ============================================================================
#
# A limit on any group above the process's own bounds it too. The shell reads the groups it can
# see, its own first and then each ancestor, and takes them into account one at a time with
# this step, starting from what the host has available.

# What is still available after one more group of the chain: no more than before it, and no
# more than that group has left when it has a limit.
def hc_chain_step(+so_far: Nat, has_limit: Bool, limit: Nat, in_use: Nat) -> Nat:
  hc_available(so_far, has_limit, limit, in_use)

# ============================================================================
# 15. Does the ring image cover what the entry keeps?
# ============================================================================
#
# A ring image serves a resume point up to the tiles its entry held when it was taken
# (`hc_ring_serves`). Whether a save has to take a new one is read from what the entry keeps,
# so a save that was cut short before the image is taken again at the next save.

# Does the image fall short of the entry's last kept position (`last_kept_tiles`: the whole
# tiles of it, 0 for an entry with none)? An entry without an image is always short.
def hc_ring_short(has_image: Bool, image_tiles: Nat, last_kept_tiles: Nat) -> Bool:
  Bool.or(Bool.not(has_image), Nat.is_lt(image_tiles, last_kept_tiles))

# Does a save copy the drafter's rings now? On an engine that has rings, when everything before
# them was copied whole (a save that was cut short gives the card up at once), and the entry's
# image is short of what the entry keeps.
def hc_ring_due(ring_bytes: Nat, complete: Bool, has_image: Bool, image_tiles: Nat, last_kept_tiles: Nat) -> Bool:
  Bool.and(Nat.is_gt(ring_bytes, 0n), Bool.and(complete, hc_ring_short(has_image, image_tiles, last_kept_tiles)))

# Is everything the device holds for its prompt in host memory after a save? Only when no copy
# of it was cut short, and (on an engine with rings) the entry's image covers its last kept
# position.
def hc_saved_whole(complete: Bool, ring_bytes: Nat, has_image: Bool, image_tiles: Nat, last_kept_tiles: Nat) -> Bool:
  Bool.and(complete, Bool.not(Bool.and(Nat.is_gt(ring_bytes, 0n), hc_ring_short(has_image, image_tiles, last_kept_tiles))))

# Does a kept position stay in an entry that keeps at most `cap` of them, with `above` positions
# (of the entry and of the save together) lying further into the prompt? The positions nearest
# the start leave first (`hc_state_leaves_first`), so one stays while fewer than `cap` lie above
# it.
def hc_state_stays(above: Nat, cap: Nat) -> Bool:
  Nat.is_lt(above, cap)

# Does a save copy, and is it charged for, the state at a kept position? One the entry lacks at
# a position its rows reach (`hc_state_wanted`), and that the entry will keep: a state that would
# leave again at the end of the same save is not copied.
def hc_state_due(already: Bool, state_tiles: Nat, entry_tiles: Nat, above: Nat, cap: Nat) -> Bool:
  Bool.and(hc_state_wanted(already, state_tiles, entry_tiles), hc_state_stays(above, cap))

# ============================================================================
# 16. Where the bytes of a tile lie
# ============================================================================
#
# A K/V cache is `chunks` chunks; a chunk holds, for each of `heads` heads, `tiles_per_chunk`
# tiles of `tile` rows of `row_bytes`. A tile's rows of one head in one cache are one piece. On
# the host the pieces of a tile follow each other, cache by cache and head by head, in the
# tile's segment. Every offset below is a count of bytes from the start of its cache or of its
# segment; the shell adds it to an address and does nothing else with it.

# Bytes of one piece.
def hc_piece_len(tile: Nat, row_bytes: Nat) -> Nat:
  Nat.mul(tile, row_bytes)

# Bytes of one head's rows in one chunk.
def hc_head_bytes(tiles_per_chunk: Nat, tile: Nat, row_bytes: Nat) -> Nat:
  Nat.mul(tiles_per_chunk, hc_piece_len(tile, row_bytes))

# Bytes of one chunk.
def hc_chunk_bytes(heads: Nat, tiles_per_chunk: Nat, tile: Nat, row_bytes: Nat) -> Nat:
  Nat.mul(heads, hc_head_bytes(tiles_per_chunk, tile, row_bytes))

# Bytes of one cache.
def hc_cache_span(chunks: Nat, heads: Nat, tiles_per_chunk: Nat, tile: Nat, row_bytes: Nat) -> Nat:
  Nat.mul(chunks, hc_chunk_bytes(heads, tiles_per_chunk, tile, row_bytes))

# Tiles of one chunk, and whether they fill it exactly (a tile then never straddles two chunks).
def hc_tiles_per_chunk(chunk_rows: Nat, tile: Nat) -> Nat:
  Nat.div(chunk_rows, tile)

def hc_tiles_fill_chunk(tiles_per_chunk: Nat, tile: Nat, chunk_rows: Nat) -> Bool:
  Nat.is_eq(Nat.mul(tiles_per_chunk, tile), chunk_rows)

# Tiles the caches hold.
def hc_cache_tiles(chunks: Nat, tiles_per_chunk: Nat) -> Nat:
  Nat.mul(chunks, tiles_per_chunk)

# The chunk a tile lies in, its place among the tiles of that chunk, and the tile that lies at
# a place.
def hc_tile_chunk(t: Nat, tiles_per_chunk: Nat) -> Nat:
  Nat.div(t, tiles_per_chunk)

def hc_tile_slot(t: Nat, tiles_per_chunk: Nat) -> Nat:
  Nat.mod(t, tiles_per_chunk)

def hc_tile_at(chunk: Nat, slot: Nat, tiles_per_chunk: Nat) -> Nat:
  Nat.add(slot, Nat.mul(chunk, tiles_per_chunk))

# Is this the piece of tile `t` for this cache and head, at a place the caches have? The shell
# asks before it uses the offsets below: the place must be the tile's (`hc_tile_at`), and the
# chunk, the place in it, the head and the cache must each be one the layout has.
def hc_piece_inside(t: Nat, +chunk: Nat, chunks: Nat, +slot: Nat, +tiles_per_chunk: Nat, head: Nat, heads: Nat, cache: Nat, caches: Nat) -> Bool:
  Bool.and(Bool.and(Bool.and(Bool.and(Nat.is_eq(hc_tile_at(chunk, slot, tiles_per_chunk), t), Nat.is_lt(chunk, chunks)), Nat.is_lt(slot, tiles_per_chunk)), Nat.is_lt(head, heads)), Nat.is_lt(cache, caches))

# Where a piece starts in its cache.
def hc_piece_dev_off(chunk: Nat, slot: Nat, head: Nat, heads: Nat, +tiles_per_chunk: Nat, +tile: Nat, +row_bytes: Nat) -> Nat:
  Nat.add(Nat.add(Nat.mul(slot, hc_piece_len(tile, row_bytes)), Nat.mul(head, hc_head_bytes(tiles_per_chunk, tile, row_bytes))), Nat.mul(chunk, hc_chunk_bytes(heads, tiles_per_chunk, tile, row_bytes)))

# Bytes of a tile's segment, and where a piece starts in it.
def hc_tile_bytes(caches: Nat, heads: Nat, tile: Nat, row_bytes: Nat) -> Nat:
  Nat.mul(caches, Nat.mul(heads, hc_piece_len(tile, row_bytes)))

def hc_piece_host_off(cache: Nat, head: Nat, heads: Nat, +tile: Nat, +row_bytes: Nat) -> Nat:
  Nat.add(Nat.mul(head, hc_piece_len(tile, row_bytes)), Nat.mul(cache, Nat.mul(heads, hc_piece_len(tile, row_bytes))))

# Device pieces of one tile.
def hc_pieces_per_tile(caches: Nat, heads: Nat) -> Nat:
  Nat.mul(caches, heads)

# Bytes the first copies through the ring are measured with: no more than a cache holds.
def hc_calibrate_bytes(wanted: Nat, cache_bytes: Nat) -> Nat:
  Nat.min(wanted, cache_bytes)

# Bytes a restore copies to the device, as the copy is weighed against the prefill it replaces
# (`hc_copy_wins`): the tiles the device no longer holds, the state at the resume point, and the
# rings when they are copied.
def hc_restore_bytes(tiles: Nat, tile_bytes: Nat, state_bytes: Nat, ring_copy: Bool, ring_bytes: Nat) -> Nat:
  Nat.add(Nat.add(Nat.mul(tiles, tile_bytes), state_bytes), hc_ring_part(ring_copy, ring_bytes))

# Device ranges of other sizes (a kept position's state, the drafter's rings) follow each other
# in a segment: where the next one starts, and whether one lies inside the segment.
def hc_range_end(off: Nat, bytes: Nat) -> Nat:
  Nat.add(off, bytes)

def hc_range_inside(off: Nat, bytes: Nat, seg_bytes: Nat) -> Bool:
  Nat.is_le(Nat.add(off, bytes), seg_bytes)

# ============================================================================
# 17. Packing a copy into the transfer ring
# ============================================================================
#
# A copy walks its pieces and the ring's buffers together: `off` bytes of a piece of `len` are
# placed, `fill` bytes of a buffer of `cap` are taken.

# Bytes placed next: what is left of the piece, and no more than the buffer has room for.
def xf_take(len: Nat, off: Nat, cap: Nat, fill: Nat) -> Nat:
  Nat.min(Nat.sub(len, off), Nat.sub(cap, fill))

# Is the buffer handed to the link before more is placed? When it has no room left.
def xf_buf_full(fill: Nat, cap: Nat) -> Bool:
  Nat.is_le(cap, fill)

def xf_next_if(inside: Bool, next: Nat) -> Nat:
  match inside:
    case True{}:
      next
    case False{}:
      0n

# The buffer after buffer `cur` of `bufs`, around the ring.
def xf_next_buf(+cur: Nat, bufs: Nat) -> Nat:
  xf_next_if(Nat.is_lt(1n+cur, bufs), 1n+cur)

# Do the bytes placed now go into the command before them? Only when they follow it both on
# the device and in the ring buffer.
def xf_cmd_joins(prev_dev: Nat, +prev_len: Nat, dev_addr: Nat, prev_ring: Nat, fill: Nat) -> Bool:
  Bool.and(Nat.is_eq(Nat.add(prev_dev, prev_len), dev_addr), Nat.is_eq(Nat.add(prev_ring, prev_len), fill))

# The host copy of one buffer is cut into a share for each copy thread. Bytes of a job given to
# the thread now: what is left of the job, and for every thread but the last no more than its
# share has room for. Any share is correct; it only spreads the work.
def xf_part_take(len: Nat, off: Nat, share: Nat, used: Nat, last: Bool) -> Nat:
  match last:
    case True{}:
      Nat.sub(len, off)
    case False{}:
      xf_take(len, off, share, used)

# Does the work go on with the next thread? When this one has its share, and is not the last.
def xf_part_full(used: Nat, share: Nat, last: Bool) -> Bool:
  Bool.and(Nat.is_le(share, used), Bool.not(last))

# The probe moves slices of one size: where slice `index` starts, and whether it lies inside
# `total` bytes.
def xf_slice_off(index: Nat, size: Nat) -> Nat:
  Nat.mul(index, size)

def xf_slice_fits(index: Nat, +size: Nat, total: Nat) -> Bool:
  Nat.is_le(Nat.add(size, xf_slice_off(index, size)), total)

# Where the probe's command `k` of buffer `b` reads or writes the device (`per_buf` commands to
# a buffer), and whether the whole round lies inside the device bytes it was given.
def xf_probe_dev_off(b: Nat, per_buf: Nat, k: Nat, cmd: Nat) -> Nat:
  xf_slice_off(Nat.add(k, Nat.mul(b, per_buf)), cmd)

def xf_probe_span(bufs: Nat, per_buf: Nat, cmd: Nat) -> Nat:
  Nat.mul(Nat.mul(bufs, per_buf), cmd)

def xf_probe_span_fits(bufs: Nat, per_buf: Nat, cmd: Nat, dev_bytes: Nat) -> Bool:
  Nat.is_le(xf_probe_span(bufs, per_buf, cmd), dev_bytes)

# ============================================================================
# 18. Laws of sections 14 to 17
# ============================================================================

# LAW HC.103: taking one more control group into account never makes more available.
law hc_chain_step_never_grows:
  for +a: Nat
  for +l: Bool
  for +lim: Nat
  for +u: Nat
  {Nat.is_le(hc_chain_step(a, l, lim, u), a) == True{} : Bool}

def hc_chain_step_never_grows(a, l, lim, u):
  hc_available_within_host(a, l, lim, u)

# LAW HC.104: what is available never exceeds what a group with a limit has left.
law hc_chain_step_within_group:
  for +a: Nat
  for +lim: Nat
  for +u: Nat
  {Nat.is_le(hc_chain_step(a, True{}, lim, u), Nat.sub(lim, u)) == True{} : Bool}

def hc_chain_step_within_group(a, lim, u):
  hc_available_within_group(a, lim, u)

# LAW HC.105: a bound that held before a group is taken into account holds after it: no group
# further up the chain gives back what a group before it took.
law hc_chain_step_keeps_earlier_bound:
  for +a: Nat
  for +b: Nat
  for +l: Bool
  for +lim: Nat
  for +u: Nat
  {Nat.is_le(hc_chain_step(Nat.min(a, b), l, lim, u), b) == True{} : Bool}

def hc_chain_step_keeps_earlier_bound(a, b, l, lim, u):
  hc_le_trans(
    hc_chain_step(Nat.min(a, b), l, lim, u),
    Nat.min(a, b),
    b,
    hc_available_within_host(Nat.min(a, b), l, lim, u),
    hc_min_le_r(a, b)
  )

# LAW HC.106: a group without a limit changes nothing.
law hc_chain_step_no_limit_changes_nothing:
  for +a: Nat
  for +lim: Nat
  for +u: Nat
  {hc_chain_step(a, False{}, lim, u) == a : Nat}

def hc_chain_step_no_limit_changes_nothing(a, lim, u):
  {==}

# LAW HC.107: on an engine with rings, an entry whose image does not cover its last kept
# position is never reported as completely saved.
law hc_short_image_never_whole:
  for +c: Bool
  for +b: Nat
  for +i: Nat
  for +x: Nat
  {hc_saved_whole(c, 1n+b, True{}, i, Nat.add(i, 1n+x)) == False{} : Bool}

def hc_short_image_never_whole(c, b, i, x):
  %hc_lt_add_succ(i, x) : {Bool.and(c, Bool.not(Bool.and(True{}, Bool.or(False{}, _)))) == False{} : Bool}
  hc_and_false_r(c)

# LAW HC.108: nor one that has no image at all.
law hc_no_image_never_whole:
  for +c: Bool
  for +b: Nat
  for +i: Nat
  for +k: Nat
  {hc_saved_whole(c, 1n+b, False{}, i, k) == False{} : Bool}

def hc_no_image_never_whole(c, b, i, k):
  hc_and_false_r(c)

# LAW HC.109: a save of which a copy was cut short is never reported as complete.
law hc_cut_short_never_whole:
  for +b: Nat
  for +h: Bool
  for +i: Nat
  for +k: Nat
  {hc_saved_whole(False{}, b, h, i, k) == False{} : Bool}

def hc_cut_short_never_whole(b, h, i, k):
  {==}

def hc_due_whole_aux(g: Bool, c: Bool, s: Bool) -> {Bool.and(Bool.and(g, Bool.and(c, s)), Bool.and(c, Bool.not(Bool.and(g, s)))) == False{} : Bool}:
  match g:
    case False{}:
      {==}
    case True{}:
      match c:
        case False{}:
          {==}
        case True{}:
          match s:
            case False{}: {==}
            case True{}: {==}

# LAW HC.110: a save that still owes a ring image is never complete.
law hc_ring_due_never_whole:
  for +b: Nat
  for +c: Bool
  for +h: Bool
  for +i: Nat
  for +k: Nat
  {Bool.and(hc_ring_due(b, c, h, i, k), hc_saved_whole(c, b, h, i, k)) == False{} : Bool}

def hc_ring_due_never_whole(b, c, h, i, k):
  hc_due_whole_aux(Nat.is_gt(b, 0n), c, hc_ring_short(h, i, k))

# LAW HC.111: an engine without rings never copies a ring image out.
law hc_ring_due_needs_rings:
  for +c: Bool
  for +h: Bool
  for +i: Nat
  for +k: Nat
  {hc_ring_due(0n, c, h, i, k) == False{} : Bool}

def hc_ring_due_needs_rings(c, h, i, k):
  {==}

# LAW HC.112: a save that was cut short never goes on to copy the rings: it gives the card up.
law hc_ring_due_not_after_a_cut:
  for +b: Nat
  for +h: Bool
  for +i: Nat
  for +k: Nat
  {hc_ring_due(b, False{}, h, i, k) == False{} : Bool}

def hc_ring_due_not_after_a_cut(b, h, i, k):
  hc_and_false_r(Nat.is_gt(b, 0n))

def hc_piece_end_in_cache(+c: Nat, +e: Nat, +k: Nat, +f: Nat, +h: Nat, +g: Nat, +t: Nat, +rb: Nat) -> {Nat.is_le(Nat.add(Nat.add(Nat.add(hc_piece_len(t, rb), Nat.mul(k, hc_piece_len(t, rb))), Nat.mul(h, hc_head_bytes(Nat.add(k, 1n+f), t, rb))), Nat.mul(c, hc_chunk_bytes(Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb))), hc_cache_span(Nat.add(c, 1n+e), Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb)) == True{} : Bool}:
  hc_nest(
    Nat.add(Nat.add(hc_piece_len(t, rb), Nat.mul(k, hc_piece_len(t, rb))), Nat.mul(h, hc_head_bytes(Nat.add(k, 1n+f), t, rb))),
    hc_chunk_bytes(Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb),
    c,
    e,
    hc_nest(
      Nat.add(hc_piece_len(t, rb), Nat.mul(k, hc_piece_len(t, rb))),
      hc_head_bytes(Nat.add(k, 1n+f), t, rb),
      h,
      g,
      hc_nest(hc_piece_len(t, rb), hc_piece_len(t, rb), k, f, hc_le_refl(hc_piece_len(t, rb)))
    )
  )

# LAW HC.113: a piece of a chunk, a place and a head the layout has ends inside its cache: the
# device offset plus the piece's length never exceeds the cache's bytes.
law hc_piece_within_cache:
  for +c: Nat
  for +e: Nat
  for +k: Nat
  for +f: Nat
  for +h: Nat
  for +g: Nat
  for +t: Nat
  for +rb: Nat
  {Nat.is_le(Nat.add(hc_piece_dev_off(c, k, h, Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb), hc_piece_len(t, rb)), hc_cache_span(Nat.add(c, 1n+e), Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb)) == True{} : Bool}

def hc_piece_within_cache(c, e, k, f, h, g, t, rb):
  %Equal.sym(Nat, Nat.add(Nat.add(Nat.add(Nat.mul(k, hc_piece_len(t, rb)), Nat.mul(h, hc_head_bytes(Nat.add(k, 1n+f), t, rb))), Nat.mul(c, hc_chunk_bytes(Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb))), hc_piece_len(t, rb)), Nat.add(Nat.add(Nat.add(hc_piece_len(t, rb), Nat.mul(k, hc_piece_len(t, rb))), Nat.mul(h, hc_head_bytes(Nat.add(k, 1n+f), t, rb))), Nat.mul(c, hc_chunk_bytes(Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb))), hc_add_shuffle3(Nat.mul(k, hc_piece_len(t, rb)), Nat.mul(h, hc_head_bytes(Nat.add(k, 1n+f), t, rb)), Nat.mul(c, hc_chunk_bytes(Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb)), hc_piece_len(t, rb))) : {Nat.is_le(_, hc_cache_span(Nat.add(c, 1n+e), Nat.add(h, 1n+g), Nat.add(k, 1n+f), t, rb)) == True{} : Bool}
  hc_piece_end_in_cache(c, e, k, f, h, g, t, rb)

# LAW HC.114: the bytes of a cache as the pieces count them are the bytes `hc_layout` checks a
# cache against, for chunks whose tiles fill them: a layout it accepted has room for every
# piece.
law hc_cache_span_is_the_checked_size:
  for +n: Nat
  for +h: Nat
  for +k: Nat
  for +t: Nat
  for +rb: Nat
  {hc_cache_span(n, h, k, t, rb) == Nat.mul(Nat.mul(Nat.mul(n, h), Nat.mul(k, t)), rb) : Nat}

def hc_cache_span_is_the_checked_size(n, h, k, t, rb):
  %hc_mul_assoc_r(Nat.mul(n, h), Nat.mul(k, t), rb) : {Nat.mul(n, Nat.mul(h, Nat.mul(k, Nat.mul(t, rb)))) == _ : Nat}
  %hc_mul_assoc_r(k, t, rb) : {Nat.mul(n, Nat.mul(h, Nat.mul(k, Nat.mul(t, rb)))) == Nat.mul(Nat.mul(n, h), _) : Nat}
  %hc_mul_assoc_r(n, h, Nat.mul(k, Nat.mul(t, rb))) : {Nat.mul(n, Nat.mul(h, Nat.mul(k, Nat.mul(t, rb)))) == _ : Nat}
  {==}

def hc_piece_end_in_segment(+c: Nat, +e: Nat, +h: Nat, +g: Nat, +t: Nat, +rb: Nat) -> {Nat.is_le(Nat.add(Nat.add(hc_piece_len(t, rb), Nat.mul(h, hc_piece_len(t, rb))), Nat.mul(c, Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb)))), Nat.mul(Nat.add(c, 1n+e), Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb)))) == True{} : Bool}:
  hc_nest(
    Nat.add(hc_piece_len(t, rb), Nat.mul(h, hc_piece_len(t, rb))),
    Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb)),
    c,
    e,
    hc_nest(hc_piece_len(t, rb), hc_piece_len(t, rb), h, g, hc_le_refl(hc_piece_len(t, rb)))
  )

# LAW HC.115: a piece of a cache and a head the layout has ends inside the tile's segment: the
# host offset plus the piece's length never exceeds the tile's bytes.
law hc_piece_within_segment:
  for +c: Nat
  for +e: Nat
  for +h: Nat
  for +g: Nat
  for +t: Nat
  for +rb: Nat
  {Nat.is_le(Nat.add(hc_piece_host_off(c, h, Nat.add(h, 1n+g), t, rb), hc_piece_len(t, rb)), hc_tile_bytes(Nat.add(c, 1n+e), Nat.add(h, 1n+g), t, rb)) == True{} : Bool}

def hc_piece_within_segment(c, e, h, g, t, rb):
  %Equal.sym(Nat, Nat.add(Nat.add(Nat.mul(h, hc_piece_len(t, rb)), Nat.mul(c, Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb)))), hc_piece_len(t, rb)), Nat.add(Nat.add(hc_piece_len(t, rb), Nat.mul(h, hc_piece_len(t, rb))), Nat.mul(c, Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb)))), hc_add_shuffle2(Nat.mul(h, hc_piece_len(t, rb)), Nat.mul(c, Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb))), hc_piece_len(t, rb))) : {Nat.is_le(_, Nat.mul(Nat.add(c, 1n+e), Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb)))) == True{} : Bool}
  hc_piece_end_in_segment(c, e, h, g, t, rb)

# LAW HC.116: two pieces of one tile in one cache never overlap in host memory: the piece of a
# later head starts at or after the end of the piece of an earlier one.
law hc_pieces_apart_in_a_cache:
  for +c: Nat
  for +h: Nat
  for +d: Nat
  for +hs: Nat
  for +t: Nat
  for +rb: Nat
  {Nat.is_le(Nat.add(hc_piece_host_off(c, h, hs, t, rb), hc_piece_len(t, rb)), hc_piece_host_off(c, Nat.add(h, 1n+d), hs, t, rb)) == True{} : Bool}

def hc_pieces_apart_in_a_cache(c, h, d, hs, t, rb):
  %Equal.sym(Nat, Nat.add(Nat.add(Nat.mul(h, hc_piece_len(t, rb)), Nat.mul(c, Nat.mul(hs, hc_piece_len(t, rb)))), hc_piece_len(t, rb)), Nat.add(Nat.add(hc_piece_len(t, rb), Nat.mul(h, hc_piece_len(t, rb))), Nat.mul(c, Nat.mul(hs, hc_piece_len(t, rb)))), hc_add_shuffle2(Nat.mul(h, hc_piece_len(t, rb)), Nat.mul(c, Nat.mul(hs, hc_piece_len(t, rb))), hc_piece_len(t, rb))) : {Nat.is_le(_, Nat.add(Nat.mul(Nat.add(h, 1n+d), hc_piece_len(t, rb)), Nat.mul(c, Nat.mul(hs, hc_piece_len(t, rb))))) == True{} : Bool}
  hc_le_add_mono(
    Nat.mul(1n+h, hc_piece_len(t, rb)),
    Nat.mul(Nat.add(h, 1n+d), hc_piece_len(t, rb)),
    Nat.mul(c, Nat.mul(hs, hc_piece_len(t, rb))),
    Nat.mul(c, Nat.mul(hs, hc_piece_len(t, rb))),
    hc_le_mul_mono(1n+h, Nat.add(h, 1n+d), hc_piece_len(t, rb), hc_succ_le_add(h, d)),
    hc_le_refl(Nat.mul(c, Nat.mul(hs, hc_piece_len(t, rb))))
  )

# LAW HC.117: nor two pieces of one tile in two caches: every piece of a later cache starts at
# or after the end of every piece of an earlier one.
law hc_pieces_apart_across_caches:
  for +c: Nat
  for +d: Nat
  for +h: Nat
  for +g: Nat
  for +h2: Nat
  for +t: Nat
  for +rb: Nat
  {Nat.is_le(Nat.add(hc_piece_host_off(c, h, Nat.add(h, 1n+g), t, rb), hc_piece_len(t, rb)), hc_piece_host_off(Nat.add(c, 1n+d), h2, Nat.add(h, 1n+g), t, rb)) == True{} : Bool}

def hc_pieces_apart_across_caches(c, d, h, g, h2, t, rb):
  hc_le_trans(
    Nat.add(hc_piece_host_off(c, h, Nat.add(h, 1n+g), t, rb), hc_piece_len(t, rb)),
    Nat.mul(Nat.add(c, 1n+d), Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb))),
    hc_piece_host_off(Nat.add(c, 1n+d), h2, Nat.add(h, 1n+g), t, rb),
    hc_piece_within_segment(c, d, h, g, t, rb),
    hc_le_add_l(
      Nat.mul(Nat.add(c, 1n+d), Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb))),
      Nat.mul(Nat.add(c, 1n+d), Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb))),
      Nat.mul(h2, hc_piece_len(t, rb)),
      hc_le_refl(Nat.mul(Nat.add(c, 1n+d), Nat.mul(Nat.add(h, 1n+g), hc_piece_len(t, rb))))
    )
  )

# LAW HC.118: every place the layout has is a tile the caches count: its tile index lies below
# `hc_cache_tiles`.
law hc_tile_at_within_cache_tiles:
  for +c: Nat
  for +e: Nat
  for +k: Nat
  for +f: Nat
  {Nat.is_le(1n+hc_tile_at(c, k, Nat.add(k, 1n+f)), hc_cache_tiles(Nat.add(c, 1n+e), Nat.add(k, 1n+f))) == True{} : Bool}

def hc_tile_at_within_cache_tiles(c, e, k, f):
  hc_nest(1n+k, Nat.add(k, 1n+f), c, e, hc_succ_le_add(k, f))

# LAW HC.119: a piece is never taken for a tile's when the place is not that tile's ...
law hc_piece_inside_is_the_tile:
  for +t: Nat
  for +c: Nat
  for +cs: Nat
  for +k: Nat
  for +tp: Nat
  for +h: Nat
  for +hs: Nat
  for +a: Nat
  for +as: Nat
  {Bool.and(hc_piece_inside(t, c, cs, k, tp, h, hs, a, as), Bool.not(Nat.is_eq(hc_tile_at(c, k, tp), t))) == False{} : Bool}

def hc_piece_inside_is_the_tile(t, c, cs, k, tp, h, hs, a, as):
  hc_tile_below_and(
    Bool.and(Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)), Nat.is_lt(h, hs)),
    Nat.is_lt(a, as),
    Bool.not(Nat.is_eq(hc_tile_at(c, k, tp), t)),
    hc_tile_below_and(
      Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)),
      Nat.is_lt(h, hs),
      Bool.not(Nat.is_eq(hc_tile_at(c, k, tp), t)),
      hc_tile_below_and(
        Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)),
        Nat.is_lt(k, tp),
        Bool.not(Nat.is_eq(hc_tile_at(c, k, tp), t)),
        hc_and_not_l(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs))
      )
    )
  )

# LAW HC.120: ... nor in a chunk the caches do not have ...
law hc_piece_inside_a_chunk:
  for +t: Nat
  for +c: Nat
  for +cs: Nat
  for +k: Nat
  for +tp: Nat
  for +h: Nat
  for +hs: Nat
  for +a: Nat
  for +as: Nat
  {Bool.and(hc_piece_inside(t, c, cs, k, tp, h, hs, a, as), Nat.is_le(cs, c)) == False{} : Bool}

def hc_piece_inside_a_chunk(t, c, cs, k, tp, h, hs, a, as):
  hc_tile_below_and(
    Bool.and(Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)), Nat.is_lt(h, hs)),
    Nat.is_lt(a, as),
    Nat.is_le(cs, c),
    hc_tile_below_and(
      Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)),
      Nat.is_lt(h, hs),
      Nat.is_le(cs, c),
      hc_tile_below_and(
        Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)),
        Nat.is_lt(k, tp),
        Nat.is_le(cs, c),
        hc_and3_excl(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs), Nat.is_le(cs, c), hc_lt_le_excl(c, cs))
      )
    )
  )

# LAW HC.121: ... at a place past the tiles of a chunk ...
law hc_piece_inside_a_place:
  for +t: Nat
  for +c: Nat
  for +cs: Nat
  for +k: Nat
  for +tp: Nat
  for +h: Nat
  for +hs: Nat
  for +a: Nat
  for +as: Nat
  {Bool.and(hc_piece_inside(t, c, cs, k, tp, h, hs, a, as), Nat.is_le(tp, k)) == False{} : Bool}

def hc_piece_inside_a_place(t, c, cs, k, tp, h, hs, a, as):
  hc_tile_below_and(
    Bool.and(Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)), Nat.is_lt(h, hs)),
    Nat.is_lt(a, as),
    Nat.is_le(tp, k),
    hc_tile_below_and(
      Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)),
      Nat.is_lt(h, hs),
      Nat.is_le(tp, k),
      hc_and3_excl(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp), Nat.is_le(tp, k), hc_lt_le_excl(k, tp))
    )
  )

# LAW HC.122: ... for a head the caches do not have ...
law hc_piece_inside_a_head:
  for +t: Nat
  for +c: Nat
  for +cs: Nat
  for +k: Nat
  for +tp: Nat
  for +h: Nat
  for +hs: Nat
  for +a: Nat
  for +as: Nat
  {Bool.and(hc_piece_inside(t, c, cs, k, tp, h, hs, a, as), Nat.is_le(hs, h)) == False{} : Bool}

def hc_piece_inside_a_head(t, c, cs, k, tp, h, hs, a, as):
  hc_tile_below_and(
    Bool.and(Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)), Nat.is_lt(h, hs)),
    Nat.is_lt(a, as),
    Nat.is_le(hs, h),
    hc_and3_excl(Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)), Nat.is_lt(h, hs), Nat.is_le(hs, h), hc_lt_le_excl(h, hs))
  )

# LAW HC.123: ... or for a cache the engine does not have.
law hc_piece_inside_a_cache:
  for +t: Nat
  for +c: Nat
  for +cs: Nat
  for +k: Nat
  for +tp: Nat
  for +h: Nat
  for +hs: Nat
  for +a: Nat
  for +as: Nat
  {Bool.and(hc_piece_inside(t, c, cs, k, tp, h, hs, a, as), Nat.is_le(as, a)) == False{} : Bool}

def hc_piece_inside_a_cache(t, c, cs, k, tp, h, hs, a, as):
  hc_and3_excl(Bool.and(Bool.and(Bool.and(Nat.is_eq(hc_tile_at(c, k, tp), t), Nat.is_lt(c, cs)), Nat.is_lt(k, tp)), Nat.is_lt(h, hs)), Nat.is_lt(a, as), Nat.is_le(as, a), hc_lt_le_excl(a, as))

# LAW HC.124: the first copies are never measured with more bytes than a cache holds.
law hc_calibrate_within_cache:
  for +w: Nat
  for +c: Nat
  {Nat.is_le(hc_calibrate_bytes(w, c), c) == True{} : Bool}

def hc_calibrate_within_cache(w, c):
  hc_min_le_r(w, c)

# LAW HC.125: the range after a range never starts before that range ends: two ranges of one
# segment never overlap in host memory.
law hc_range_end_after_the_range:
  for +o: Nat
  for +b: Nat
  {Nat.is_le(Nat.add(o, b), hc_range_end(o, b)) == True{} : Bool}

def hc_range_end_after_the_range(o, b):
  hc_le_refl(Nat.add(o, b))

# LAW HC.126: a range that ends past its segment is never taken for one inside it.
law hc_range_past_segment_never_inside:
  for +o: Nat
  for +b: Nat
  for +s: Nat
  {Bool.and(hc_range_inside(o, b, s), Nat.is_lt(s, Nat.add(o, b))) == False{} : Bool}

def hc_range_past_segment_never_inside(o, b, s):
  hc_le_lt_excl(Nat.add(o, b), s)

# LAW HC.127: the bytes placed next never reach past the end of the piece ...
law xf_take_within_piece:
  for +off: Nat
  for +left: Nat
  for +cap: Nat
  for +fill: Nat
  {Nat.is_le(Nat.add(off, xf_take(Nat.add(off, left), off, cap, fill)), Nat.add(off, left)) == True{} : Bool}

def xf_take_within_piece(off, left, cap, fill):
  %Equal.sym(Nat, Nat.sub(Nat.add(off, left), off), left, hc_add_sub_cancel(off, left)) : {Nat.is_le(Nat.add(off, Nat.min(_, Nat.sub(cap, fill))), Nat.add(off, left)) == True{} : Bool}
  hc_le_add_mono(off, off, Nat.min(left, Nat.sub(cap, fill)), left, hc_le_refl(off), hc_min_le_l(left, Nat.sub(cap, fill)))

# LAW HC.128: ... nor past the end of the ring buffer.
law xf_take_within_buffer:
  for +len: Nat
  for +off: Nat
  for +fill: Nat
  for +room: Nat
  {Nat.is_le(Nat.add(fill, xf_take(len, off, Nat.add(fill, room), fill)), Nat.add(fill, room)) == True{} : Bool}

def xf_take_within_buffer(len, off, fill, room):
  %Equal.sym(Nat, Nat.sub(Nat.add(fill, room), fill), room, hc_add_sub_cancel(fill, room)) : {Nat.is_le(Nat.add(fill, Nat.min(Nat.sub(len, off), _)), Nat.add(fill, room)) == True{} : Bool}
  hc_le_add_mono(fill, fill, Nat.min(Nat.sub(len, off), room), room, hc_le_refl(fill), hc_min_le_r(Nat.sub(len, off), room))

# LAW HC.129: a buffer with no room left is always handed to the link: nothing is ever placed
# past its end, and a copy never stands still on a full buffer.
law xf_full_buffer_is_handed_over:
  for +c: Nat
  for +x: Nat
  {xf_buf_full(Nat.add(c, x), c) == True{} : Bool}

def xf_full_buffer_is_handed_over(c, x):
  Equal.sym(Bool, True{}, Nat.is_le(c, Nat.add(c, x)), hc_le_add_r(c, x))

def xf_next_aux(b: Bool, nx: Nat, m: Nat, h: {Nat.is_lt(nx, 1n+m) == b : Bool}) -> {Nat.is_lt(xf_next_if(b, nx), 1n+m) == True{} : Bool}:
  match b:
    case True{}:
      h
    case False{}:
      {==}

# LAW HC.130: the next buffer is always one the ring has.
law xf_next_buf_is_in_the_ring:
  for +c: Nat
  for +m: Nat
  {Nat.is_lt(xf_next_buf(c, 1n+m), 1n+m) == True{} : Bool}

def xf_next_buf_is_in_the_ring(c, m):
  xf_next_aux(Nat.is_lt(1n+c, 1n+m), 1n+c, m, {==})

# LAW HC.131: bytes are never added to a command they do not follow on the device ...
law xf_cmd_joins_on_the_device:
  for +pd: Nat
  for +pl: Nat
  for +da: Nat
  for +pr: Nat
  for +f: Nat
  {Bool.and(xf_cmd_joins(pd, pl, da, pr, f), Bool.not(Nat.is_eq(Nat.add(pd, pl), da))) == False{} : Bool}

def xf_cmd_joins_on_the_device(pd, pl, da, pr, f):
  hc_and_not_l(Nat.is_eq(Nat.add(pd, pl), da), Nat.is_eq(Nat.add(pr, pl), f))

# LAW HC.132: ... or in the ring buffer.
law xf_cmd_joins_in_the_ring:
  for +pd: Nat
  for +pl: Nat
  for +da: Nat
  for +pr: Nat
  for +f: Nat
  {Bool.and(xf_cmd_joins(pd, pl, da, pr, f), Bool.not(Nat.is_eq(Nat.add(pr, pl), f))) == False{} : Bool}

def xf_cmd_joins_in_the_ring(pd, pl, da, pr, f):
  hc_and_not_r(Nat.is_eq(Nat.add(pd, pl), da), Nat.is_eq(Nat.add(pr, pl), f))

def xf_part_take_aux(l: Bool, +off: Nat, +left: Nat, +s: Nat, +u: Nat) -> {Nat.is_le(Nat.add(off, xf_part_take(Nat.add(off, left), off, s, u, l)), Nat.add(off, left)) == True{} : Bool}:
  match l:
    case True{}:
      %Equal.sym(Nat, Nat.sub(Nat.add(off, left), off), left, hc_add_sub_cancel(off, left)) : {Nat.is_le(Nat.add(off, _), Nat.add(off, left)) == True{} : Bool}
      hc_le_refl(Nat.add(off, left))
    case False{}:
      xf_take_within_piece(off, left, s, u)

# LAW HC.133: a copy thread is never given bytes past the end of a job.
law xf_part_take_within_job:
  for +l: Bool
  for +off: Nat
  for +left: Nat
  for +s: Nat
  for +u: Nat
  {Nat.is_le(Nat.add(off, xf_part_take(Nat.add(off, left), off, s, u, l)), Nat.add(off, left)) == True{} : Bool}

def xf_part_take_within_job(l, off, left, s, u):
  xf_part_take_aux(l, off, left, s, u)

# LAW HC.134: the work never goes on past the last copy thread.
law xf_last_thread_takes_the_rest:
  for +u: Nat
  for +s: Nat
  {xf_part_full(u, s, True{}) == False{} : Bool}

def xf_last_thread_takes_the_rest(u, s):
  hc_and_false_r(Nat.is_le(s, u))

# LAW HC.135: a slice that ends past the bytes it is cut from is never taken for one inside.
law xf_slice_past_end_never_fits:
  for +i: Nat
  for +z: Nat
  for +t: Nat
  {Bool.and(xf_slice_fits(i, z, t), Nat.is_lt(t, Nat.add(z, xf_slice_off(i, z)))) == False{} : Bool}

def xf_slice_past_end_never_fits(i, z, t):
  hc_le_lt_excl(Nat.add(z, xf_slice_off(i, z)), t)

# LAW HC.136: every command of a probe round ends inside the round's span on the device.
law xf_probe_cmd_within_span:
  for +b: Nat
  for +e: Nat
  for +k: Nat
  for +f: Nat
  for +cmd: Nat
  {Nat.is_le(Nat.add(cmd, xf_probe_dev_off(b, Nat.add(k, 1n+f), k, cmd)), xf_probe_span(Nat.add(b, 1n+e), Nat.add(k, 1n+f), cmd)) == True{} : Bool}

def xf_probe_cmd_within_span(b, e, k, f, cmd):
  hc_le_mul_mono(
    1n+Nat.add(k, Nat.mul(b, Nat.add(k, 1n+f))),
    Nat.mul(Nat.add(b, 1n+e), Nat.add(k, 1n+f)),
    cmd,
    hc_nest(1n+k, Nat.add(k, 1n+f), b, e, hc_succ_le_add(k, f))
  )

# LAW HC.137: a probe round whose span exceeds the device bytes it was given never runs.
law xf_probe_span_over_refused:
  for +n: Nat
  for +p: Nat
  for +c: Nat
  for +d: Nat
  {Bool.and(xf_probe_span_fits(n, p, c, d), Nat.is_lt(d, xf_probe_span(n, p, c))) == False{} : Bool}

def xf_probe_span_over_refused(n, p, c, d):
  hc_le_lt_excl(xf_probe_span(n, p, c), d)

# LAW HC.138: a restore is never weighed with fewer bytes than the rows it copies: a copy never
# wins against a prefill by leaving its tiles out of the count.
law hc_restore_bytes_cover_the_tiles:
  for +n: Nat
  for +tb: Nat
  for +sb: Nat
  for +c: Bool
  for +rb: Nat
  {Nat.is_le(Nat.mul(n, tb), hc_restore_bytes(n, tb, sb, c, rb)) == True{} : Bool}

def hc_restore_bytes_cover_the_tiles(n, tb, sb, c, rb):
  Equal.sym(Bool, True{}, Nat.is_le(Nat.mul(n, tb), Nat.add(Nat.add(Nat.mul(n, tb), sb), hc_ring_part(c, rb))), hc_le_add_add(Nat.mul(n, tb), sb, hc_ring_part(c, rb)))

# LAW HC.139: the state at a position past the rows the entry holds is never copied.
law hc_state_due_never_past_rows:
  for +a: Bool
  for +k: Nat
  for +t: Nat
  for +ab: Nat
  for +c: Nat
  {Bool.and(hc_state_due(a, k, t, ab, c), Nat.is_lt(t, k)) == False{} : Bool}

def hc_state_due_never_past_rows(a, k, t, ab, c):
  hc_tile_below_and(hc_state_wanted(a, k, t), hc_state_stays(ab, c), Nat.is_lt(t, k), hc_state_past_rows_never_wanted(a, k, t))

# LAW HC.140: a state with as many positions above it as the entry may keep is never copied: a
# save never gives an entry a position beyond its cap.
law hc_state_due_within_cap:
  for +a: Bool
  for +k: Nat
  for +t: Nat
  for +ab: Nat
  for +c: Nat
  {Bool.and(hc_state_due(a, k, t, ab, c), Nat.is_le(c, ab)) == False{} : Bool}

def hc_state_due_within_cap(a, k, t, ab, c):
  hc_and3_excl(hc_state_wanted(a, k, t), Nat.is_lt(ab, c), Nat.is_le(c, ab), hc_lt_le_excl(ab, c))

# LAW HC.141: the state at the last kept position is always copied when the entry lacks it, holds
# its rows and may keep any position at all: the point the next turn resumes from is never left
# out of a save.
law hc_last_position_is_due:
  for +k: Nat
  for +x: Nat
  for +c: Nat
  {hc_state_due(False{}, k, Nat.add(k, x), 0n, 1n+c) == True{} : Bool}

def hc_last_position_is_due(k, x, c):
  %hc_le_add_r(k, x) : {Bool.and(Bool.and(True{}, _), True{}) == True{} : Bool}
  {==}

# ============================================================================
# 19. Pending idle work, and the host memory of the copy probe
# ============================================================================

# A chat request leaves idle work pending (the save of what the card holds). Any connection
# that ends looks whether it can start it. Is it started now? Only when some is pending and the
# engine may be taken (`hc_idle_may_take`).
def hc_idle_starts(due: Bool, active: Nat, ready: Bool) -> Bool:
  Bool.and(due, hc_idle_may_take(active, ready))

# Is idle work still pending after a caller looked? Unless that caller started it.
def hc_idle_left(due: Bool, started: Bool) -> Bool:
  Bool.and(due, Bool.not(started))

# Ordinary host memory the copy probe holds at its peak: one buffer as large as the device
# bytes it moves, or the two buffers of its host-copy test.
def xf_probe_host_need(+device_bytes: Nat, +copy_bytes: Nat) -> Nat:
  Nat.max(device_bytes, Nat.add(copy_bytes, copy_bytes))

# LAW HC.142: idle work is never started on an engine a request is using ...
law hc_idle_never_starts_on_a_busy_engine:
  for +d: Bool
  for +a: Nat
  for +r: Bool
  {hc_idle_starts(d, 1n+a, r) == False{} : Bool}

def hc_idle_never_starts_on_a_busy_engine(d, a, r):
  %Equal.sym(Bool, hc_idle_may_take(1n+a, r), False{}, hc_idle_never_takes_a_busy_engine(a, r)) : {Bool.and(d, _) == False{} : Bool}
  hc_and_false_r(d)

# LAW HC.143: ... nor on one that is not ready ...
law hc_idle_never_starts_on_an_unready_engine:
  for +d: Bool
  for +a: Nat
  {hc_idle_starts(d, a, False{}) == False{} : Bool}

def hc_idle_never_starts_on_an_unready_engine(d, a):
  hc_and_false_r(d)

# LAW HC.144: ... nor when none is pending.
law hc_idle_never_starts_without_work:
  for +a: Nat
  for +r: Bool
  {hc_idle_starts(False{}, a, r) == False{} : Bool}

def hc_idle_never_starts_without_work(a, r):
  {==}

# LAW HC.145: work that a caller could not start is never marked done: it stays as pending as
# the caller found it.
law hc_idle_unstarted_work_stays:
  for +d: Bool
  {hc_idle_left(d, False{}) == d : Bool}

def hc_idle_unstarted_work_stays(d):
  match d:
    case True{}: {==}
    case False{}: {==}

# LAW HC.146: so pending work is still pending after a caller that found the engine in use.
law hc_idle_busy_engine_keeps_work_pending:
  for +a: Nat
  for +r: Bool
  {hc_idle_left(True{}, hc_idle_starts(True{}, 1n+a, r)) == True{} : Bool}

def hc_idle_busy_engine_keeps_work_pending(a, r):
  %Equal.sym(Bool, hc_idle_may_take(1n+a, r), False{}, hc_idle_never_takes_a_busy_engine(a, r)) : {Bool.and(True{}, Bool.not(Bool.and(True{}, _))) == True{} : Bool}
  {==}

# LAW HC.147: the probe's host memory is never counted below the buffer it copies the device
# bytes into ...
law xf_probe_need_covers_the_buffer:
  for +b: Nat
  for +c: Nat
  {Nat.is_le(b, xf_probe_host_need(b, c)) == True{} : Bool}

def xf_probe_need_covers_the_buffer(b, c):
  hc_le_max_l(b, Nat.add(c, c))

# LAW HC.148: ... nor below the two buffers of its host-copy test.
law xf_probe_need_covers_the_copy:
  for +b: Nat
  for +c: Nat
  {Nat.is_le(Nat.add(c, c), xf_probe_host_need(b, c)) == True{} : Bool}

def xf_probe_need_covers_the_copy(b, c):
  hc_le_max_r(b, Nat.add(c, c))

# ============================================================================
# 20. One admission for every host allocation, and limits that cannot be read
# ============================================================================
#
# Every mapping and every pinning of host memory this feature makes (a segment of the store, an
# engine's transfer ring, the buffer of its first copies, the buffers of the copy probe) is
# taken out of a leave: so many bytes, asked for against what the host and every control group
# above the process have left at that moment (`hc_available` and `hc_chain_step` over the
# chain). The shell cannot map or pin without one.

# May `need` bytes of host memory be taken now, with `available` bytes available now?
def hm_admit(need: Nat, available: Nat) -> Bool:
  Nat.is_le(need, available)

# Fresh host memory an allocation of `need` bytes takes: what it needs beyond the mapped
# segments it takes again (`reused` bytes of them, memory the process already has).
def hm_fresh(need: Nat, reused: Nat) -> Nat:
  Nat.sub(need, reused)

# Does a leave with `left` bytes cover one more mapping of `bytes`?
def hm_covers(left: Nat, bytes: Nat) -> Bool:
  Nat.is_le(bytes, left)

# What a leave has left once a mapping of `bytes` was taken out of it.
def hm_left(left: Nat, bytes: Nat) -> Nat:
  Nat.sub(left, bytes)

# The budget of a server that names none is taken from what is available (`hc_default_budget`).
# May it be, when the limits of the process are established (`limits_known`: what the host has
# available was read, and so was the limit of every memory-controlled group the process is
# in)? Not otherwise: a share of a figure that leaves out a limit can be more than that limit
# allows, and the kernel stops a process that takes it. The operator who names a budget knows
# the limit; the server does not guess one.
def hc_default_allowed(limits_known: Bool) -> Bool:
  limits_known

# Host memory the store may count on at a reading: the bytes read as available, or the bytes
# its entries would give back to the host. All of them when the limits of the process are
# established, or when the operator named the budget (it bounds the store then, and what
# could be read still does). None otherwise: a limit can stop being readable while the server
# runs, and a figure that leaves a limit out is no ground to take memory on. The store then
# goes on with the memory it has (a save takes the segments of the entries that leave for it).
def hc_host_counted(limits_known: Bool, budget_named: Bool, +bytes: Nat) -> Nat:
  match limits_known:
    case True{}:
      bytes
    case False{}:
      match budget_named:
        case True{}:
          bytes
        case False{}:
          0n

# LAW HC.149: nothing is admitted beyond what is available at the moment it is asked for.
law hm_never_admits_beyond_available:
  for +n: Nat
  for +a: Nat
  {Bool.and(hm_admit(n, a), Nat.is_lt(a, n)) == False{} : Bool}

def hm_never_admits_beyond_available(n, a):
  hc_le_lt_excl(n, a)

# LAW HC.150: a mapping larger than what its leave has left is never covered ...
law hm_never_covers_beyond_the_leave:
  for +l: Nat
  for +b: Nat
  {Bool.and(hm_covers(l, b), Nat.is_lt(l, b)) == False{} : Bool}

def hm_never_covers_beyond_the_leave(l, b):
  hc_le_lt_excl(b, l)

# LAW HC.151: ... a leave never grows by being used ...
law hm_left_never_grows:
  for +l: Nat
  for +b: Nat
  {Nat.is_le(hm_left(l, b), l) == True{} : Bool}

def hm_left_never_grows(l, b):
  hc_sub_le_self(l, b)

# LAW HC.152: ... and what a covered mapping took and what is left never add up to more than
# the leave held: all the mappings made under one leave stay within what was admitted.
law hm_taken_and_left_within_the_leave:
  for +b: Nat
  for +r: Nat
  {Nat.is_le(Nat.add(b, hm_left(Nat.add(b, r), b)), Nat.add(b, r)) == True{} : Bool}

def hm_taken_and_left_within_the_leave(b, r):
  %Equal.sym(Nat, Nat.sub(Nat.add(b, r), b), r, hc_add_sub_cancel(b, r)) : {Nat.is_le(Nat.add(b, _), Nat.add(b, r)) == True{} : Bool}
  hc_le_refl(Nat.add(b, r))

# LAW HC.153: the fresh memory of an allocation is never counted above what it needs.
law hm_fresh_within_need:
  for +n: Nat
  for +r: Nat
  {Nat.is_le(hm_fresh(n, r), n) == True{} : Bool}

def hm_fresh_within_need(n, r):
  hc_sub_le_self(n, r)

# n <= (n - r) + r
def hm_le_sub_add(+n: Nat, +r: Nat) -> {Nat.is_le(n, Nat.add(Nat.sub(n, r), r)) == True{} : Bool}:
  match n r:
    case 0n _:
      hc_zero_le(r)
    case 1n+np 0n:
      Equal.sym(Bool, True{}, Nat.is_le(np, Nat.add(np, 0n)), hc_le_add_r(np, 0n))
    case 1n+np 1n+rp:
      %Equal.sym(Nat, Nat.add(Nat.sub(np, rp), 1n+rp), 1n+Nat.add(Nat.sub(np, rp), rp), hc_add_succ_r(Nat.sub(np, rp), rp)) : {Nat.is_le(1n+np, _) == True{} : Bool}
      hm_le_sub_add(np, rp)

# LAW HC.154: ... nor below what it maps beyond the segments it takes again: the leave an
# allocation is given covers every fresh mapping it makes.
law hm_fresh_and_reused_cover_the_need:
  for +n: Nat
  for +r: Nat
  {Nat.is_le(n, Nat.add(hm_fresh(n, r), r)) == True{} : Bool}

def hm_fresh_and_reused_cover_the_need(n, r):
  hm_le_sub_add(n, r)

# (That a server whose limits could not be read takes no default budget at all is today's
# decision, pinned by Rust tests and by no law: a later version may take a smaller one that is
# safe on other grounds.)

# LAW HC.156: the store never counts more host memory than was read ...
law hc_host_counted_within_read:
  for +k: Bool
  for +n: Bool
  for +b: Nat
  {Nat.is_le(hc_host_counted(k, n, b), b) == True{} : Bool}

def hc_host_counted_within_read(k, n, b):
  match k:
    case True{}:
      hc_le_refl(b)
    case False{}:
      match n:
        case True{}:
          hc_le_refl(b)
        case False{}:
          hc_zero_le(b)

# (That the store counts no host memory at all under an unread limit and an unnamed budget is
# the same decision, and is left to the Rust tests for the same reason.)
