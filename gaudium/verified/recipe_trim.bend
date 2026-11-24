import Base
import nat_lemmas.bend as L

# Trimming the on-disk compiled-recipe cache.
#
# An entry of the cache is one compiled recipe: its `.recipe` file and its `.sha256` sidecar,
# removed together. The shell scans the directory, measures each entry (bytes on disk, the last
# time it was used, whether this process holds it) and walks the entries in the order
# `rt_evict_before` gives, asking `rt_should_evict` for each one and keeping its count of the
# bytes still held with `rt_after`. Which files form an entry, how a time is read from the
# filesystem and how a budget is given on the command line are the shell's; the rules below are
# what every trim keeps: an entry this process holds, or one used within the keep window, is
# never removed, nothing is removed while the cache is within its target, and the byte count only
# goes down by what was removed.
#
# Times are whole seconds on one clock (the shell's: seconds since the Unix epoch).

# ============================================================================
# 1. Rules
# ============================================================================

# The last time an entry was used: the later of its two files' times (a missing file counts 0).
def rt_last_used(+recipe_time: Nat, +meta_time: Nat) -> Nat:
  Nat.max(recipe_time, meta_time)

# Is an entry last used at `used` kept whatever the budget? When this process holds it, or when
# it was used less than `keep` seconds before `now`. A time ahead of `now` (another clock, a copied
# file) counts as just used.
def rt_protected(+used: Nat, +now: Nat, keep: Nat, in_use: Bool) -> Bool:
  Bool.or(in_use, Nat.is_lt(Nat.sub(now, used), keep))

# Does an entry last used at `a_used` leave before one last used at `b_used`? The one used less
# recently.
def rt_evict_before(a_used: Nat, b_used: Nat) -> Bool:
  Nat.is_lt(a_used, b_used)

# With `held` bytes of entries still on disk and a target of `target` bytes, is the next entry in
# eviction order removed? Only when it is not protected and the cache is over the target.
def rt_should_evict(+held: Nat, target: Nat, protected: Bool) -> Bool:
  Bool.and(Bool.not(protected), Nat.is_lt(target, held))

# Bytes of an entry: its recipe file and its sidecar (a missing file counts 0); `None` past 64
# bits.
# rusty-bends: total
def rt_entry_bytes(recipe_bytes: Nat, meta_bytes: Nat) -> Nat:
  Nat.add(recipe_bytes, meta_bytes)

# The count of held bytes (or of freed bytes) with one more entry of `bytes` bytes. `None` when
# the count would not fit 64 bits: the shell then refuses to trim rather than guess.
# rusty-bends: total
def rt_count_add(held: Nat, bytes: Nat) -> Nat:
  Nat.add(held, bytes)

# Bytes still held after an entry of `bytes` bytes was (`removed`) or was not removed.
def rt_after(+held: Nat, bytes: Nat, removed: Bool) -> Nat:
  match removed:
    case True{}:
      Nat.sub(held, bytes)
    case False{}:
      held

# ============================================================================
# 2. Lemmas
# ============================================================================

def rt_lt_add_succ(d: Nat, x: Nat) -> {Nat.is_lt(d, Nat.add(d, 1n+x)) == True{} : Bool}:
  match d:
    case 0n:
      {==}
    case 1n+p:
      rt_lt_add_succ(p, x)

def rt_add_lt_false(h: Nat, x: Nat) -> {Nat.is_lt(Nat.add(h, x), h) == False{} : Bool}:
  match h:
    case 0n:
      match x:
        case 0n:
          {==}
        case 1n+_:
          {==}
    case 1n+q:
      rt_add_lt_false(q, x)

def rt_sub_ahead_zero(n: Nat, e: Nat) -> {Nat.sub(n, Nat.add(n, e)) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      rt_sub_ahead_zero(p, e)

def rt_or_true_r(b: Bool) -> {Bool.or(b, True{}) == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

def rt_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

def rt_sub_le(+h: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(h, b), h) == True{} : Bool}:
  match h b:
    case 0n _:
      {==}
    case 1n+p 0n:
      L.nat_le_refl(p)
    case 1n+p 1n+q:
      L.nat_le_succ_r(Nat.sub(p, q), p, rt_sub_le(p, q))

# ============================================================================
# 3. Laws
# ============================================================================

# promise: an entry's last use is never earlier than the time of its recipe file.
# harm: an entry read a moment ago through a fresh sidecar would look old and could be removed
law rt_last_used_ge_recipe:
  for +r: Nat
  for +m: Nat
  {Nat.is_le(r, rt_last_used(r, m)) == True{} : Bool}

def rt_last_used_ge_recipe(r, m):
  L.nat_max_ge_l(r, m)

# promise: an entry's last use is never earlier than the time of its sidecar, which every cache hit refreshes.
# harm: the recency a cache hit records would be ignored and a recipe in daily use could be removed
law rt_last_used_ge_meta:
  for +r: Nat
  for +m: Nat
  {Nat.is_le(m, rt_last_used(r, m)) == True{} : Bool}

def rt_last_used_ge_meta(r, m):
  L.nat_max_ge_r(r, m)

# promise: an entry this process holds is always protected, whatever its age and the keep window.
# harm: a running server could lose a recipe it loads again later and have to compile it anew
law rt_in_use_protected:
  for +u: Nat
  for +n: Nat
  for +k: Nat
  {rt_protected(u, n, k, True{}) == True{} : Bool}

def rt_in_use_protected(u, n, k):
  {==}

# promise: an entry used less than the keep window ago is protected.
# harm: a trim could remove recipes another process is using right now
law rt_recent_protected:
  for +u: Nat
  for +d: Nat
  for +x: Nat
  for +f: Bool
  {rt_protected(u, Nat.add(u, d), Nat.add(d, 1n+x), f) == True{} : Bool}

def rt_recent_protected(u, d, x, f):
  %Equal.sym(Nat, Nat.sub(Nat.add(u, d), u), d, L.nat_add_sub_cancel_l(u, d)) : {Bool.or(f, Nat.is_lt(_, Nat.add(d, 1n+x))) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(d, Nat.add(d, 1n+x)), True{}, rt_lt_add_succ(d, x)) : {Bool.or(f, _) == True{} : Bool}
  rt_or_true_r(f)

# promise: an entry whose time lies ahead of the clock is protected by any keep window.
# harm: a clock step back or a copied file could make fresh recipes look ancient and remove them
law rt_ahead_protected:
  for +n: Nat
  for +e: Nat
  for +k: Nat
  for +f: Bool
  {rt_protected(Nat.add(n, e), n, 1n+k, f) == True{} : Bool}

def rt_ahead_protected(n, e, k, f):
  %Equal.sym(Nat, Nat.sub(n, Nat.add(n, e)), 0n, rt_sub_ahead_zero(n, e)) : {Bool.or(f, Nat.is_lt(_, 1n+k)) == True{} : Bool}
  rt_or_true_r(f)

# promise: a protected entry is never removed, however far the cache is over its target.
# harm: the keep window and the entries a server holds would be ignored by the trim
law rt_protected_never_evicted:
  for +h: Nat
  for +t: Nat
  {rt_should_evict(h, t, True{}) == False{} : Bool}

def rt_protected_never_evicted(h, t):
  {==}

# promise: nothing is removed while the cache is within its target.
# harm: a trim could empty a cache that already fits and force every recipe to compile again
law rt_within_target_keeps:
  for +h: Nat
  for +x: Nat
  for +p: Bool
  {rt_should_evict(h, Nat.add(h, x), p) == False{} : Bool}

def rt_within_target_keeps(h, x, p):
  %Equal.sym(Bool, Nat.is_lt(Nat.add(h, x), h), False{}, rt_add_lt_false(h, x)) : {Bool.and(Bool.not(p), _) == False{} : Bool}
  rt_and_false_r(Bool.not(p))

# promise: an unprotected entry is removed while the cache is over its target.
# harm: a trim could stop early and leave the disk over the budget it was given
law rt_over_target_evicts:
  for +t: Nat
  for +x: Nat
  {rt_should_evict(Nat.add(t, 1n+x), t, False{}) == True{} : Bool}

def rt_over_target_evicts(t, x):
  rt_lt_add_succ(t, x)

# promise: no entry leaves before itself, so the eviction order is a strict order.
law rt_evict_before_irreflexive:
  for +a: Nat
  {rt_evict_before(a, a) == False{} : Bool}

def rt_evict_before_irreflexive(a):
  %L.nat_add_zero_r(a) : {Nat.is_lt(_, a) == False{} : Bool}
  rt_add_lt_false(a, 0n)

# promise: an entry that was not removed leaves the count of held bytes unchanged.
# harm: a failed removal would be counted as freed space and the trim would stop over budget
law rt_kept_holds:
  for +h: Nat
  for +b: Nat
  {rt_after(h, b, False{}) == h : Nat}

def rt_kept_holds(h, b):
  {==}

# promise: removing an entry takes exactly its bytes off the count of held bytes.
# harm: the trim's count would drift from the disk and remove too much or too little
law rt_removed_frees_its_bytes:
  for +h: Nat
  for +b: Nat
  {rt_after(Nat.add(h, b), b, True{}) == h : Nat}

def rt_removed_frees_its_bytes(h, b):
  L.nat_add_sub_cancel_r(h, b)

# promise: an entry's size covers its recipe file.
law rt_entry_bytes_covers_recipe:
  for +r: Nat
  for +m: Nat
  {Nat.is_le(r, rt_entry_bytes(r, m)) == True{} : Bool}

def rt_entry_bytes_covers_recipe(r, m):
  L.nat_add_le_r(r, m)

# promise: an entry's size covers its sidecar.
law rt_entry_bytes_covers_meta:
  for +r: Nat
  for +m: Nat
  {Nat.is_le(m, rt_entry_bytes(r, m)) == True{} : Bool}

def rt_entry_bytes_covers_meta(r, m):
  L.nat_add_le_l(r, m)

# promise: counting an entry in and then removing it gives back the count before it, so the count a trim starts from is the sum the removals take apart.
# harm: the starting count could disagree with what the removals free and the trim would stop over or under budget
law rt_count_then_remove_returns:
  for +h: Nat
  for +b: Nat
  {rt_after(rt_count_add(h, b), b, True{}) == h : Nat}

def rt_count_then_remove_returns(h, b):
  L.nat_add_sub_cancel_r(h, b)

# promise: counting an entry in never lowers the count.
law rt_count_never_shrinks:
  for +h: Nat
  for +b: Nat
  {Nat.is_le(h, rt_count_add(h, b)) == True{} : Bool}

def rt_count_never_shrinks(h, b):
  L.nat_add_le_r(h, b)

# promise: the count of held bytes never grows during a trim.
law rt_after_never_grows:
  for +h: Nat
  for +b: Nat
  for +r: Bool
  {Nat.is_le(rt_after(h, b, r), h) == True{} : Bool}

def rt_after_never_grows(h, b, r):
  match r:
    case True{}:
      rt_sub_le(h, b)
    case False{}:
      L.nat_le_refl(h)
