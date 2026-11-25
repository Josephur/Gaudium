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

# Bytes of one file of an entry that is gone from the disk: all of them when it is gone, none when
# it is still there.
def rt_if_gone(bytes: Nat, gone: Bool) -> Nat:
  match gone:
    case True{}:
      bytes
    case False{}:
      0n

# Bytes an entry's removal takes off the count of held bytes: those of each of its files that is
# gone afterwards (removed now, or already removed by someone else). `None` past 64 bits.
# rusty-bends: total
def rt_gone_bytes(recipe_bytes: Nat, recipe_gone: Bool, meta_bytes: Nat, meta_gone: Bool) -> Nat:
  Nat.add(rt_if_gone(recipe_bytes, recipe_gone), rt_if_gone(meta_bytes, meta_gone))

# The sidecar of an entry is removed first. May its recipe file go next? Only when the entry has
# no sidecar on disk, or the sidecar is gone: a sidecar never stays behind without the check
# reading it, and a recipe is never left without being an entry anyone can still trim.
def rt_may_remove_recipe(meta_present: Bool, meta_gone: Bool) -> Bool:
  Bool.or(Bool.not(meta_present), meta_gone)

# Bytes still held after `bytes` bytes of entries left the disk.
def rt_after(+held: Nat, bytes: Nat) -> Nat:
  Nat.sub(held, bytes)

# What the walk makes of an entry once it has read the entry's files again, just before it would
# remove them.
type RtRecheck is Data:
  # neither file is on disk any more: someone else removed the entry since the scan
  RtRecheckGone{}
  # it stays: protected, or the cache is within its target
  RtRecheckKeep{}
  # it is removed now
  RtRecheckRemove{}

def rt_recheck_pick(on_disk: Bool, evict: Bool) -> RtRecheck:
  match on_disk:
    case False{}:
      RtRecheckGone{}
    case True{}:
      match evict:
        case True{}:
          RtRecheckRemove{}
        case False{}:
          RtRecheckKeep{}

# With `held` bytes still counted and a target of `target` bytes, what becomes of an entry whose
# recipe file is (`recipe_on_disk`) or is not on disk, and whose sidecar is (`meta_on_disk`) or is
# not, as the second look found them; `protected` is `rt_protected` of the times that look read.
def rt_recheck(+held: Nat, target: Nat, recipe_on_disk: Bool, meta_on_disk: Bool, protected: Bool) -> RtRecheck:
  rt_recheck_pick(Bool.or(recipe_on_disk, meta_on_disk), rt_should_evict(held, target, protected))

# Is the entry removed now?
def rt_recheck_removes(+c: RtRecheck) -> Bool:
  match c:
    case RtRecheckGone{}:
      False{}
    case RtRecheckKeep{}:
      False{}
    case RtRecheckRemove{}:
      True{}

# Is the entry counted as gone from the disk (its bytes off the count, freed by someone else)?
def rt_recheck_gone(+c: RtRecheck) -> Bool:
  match c:
    case RtRecheckGone{}:
      True{}
    case RtRecheckKeep{}:
      False{}
    case RtRecheckRemove{}:
      False{}

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

def rt_lt_zero_false(b: Nat) -> {Nat.is_lt(b, 0n) == False{} : Bool}:
  match b:
    case 0n:
      {==}
    case 1n+_:
      {==}

def rt_before_trans(+c: Nat, +a: Nat, +b: Nat) -> {Bool.and(Bool.and(rt_evict_before(a, b), rt_evict_before(b, c)), Bool.not(rt_evict_before(a, c))) == False{} : Bool}:
  match c:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(b, 0n), False{}, rt_lt_zero_false(b)) : {Bool.and(Bool.and(Nat.is_lt(a, b), _), Bool.not(Nat.is_lt(a, 0n))) == False{} : Bool}
      %Equal.sym(Bool, Bool.and(Nat.is_lt(a, b), False{}), False{}, rt_and_false_r(Nat.is_lt(a, b))) : {Bool.and(_, Bool.not(Nat.is_lt(a, 0n))) == False{} : Bool}
      {==}
    case 1n+cp:
      match a:
        case 0n:
          rt_and_false_r(Bool.and(Nat.is_lt(0n, b), Nat.is_lt(b, 1n+cp)))
        case 1n+ap:
          match b:
            case 0n:
              {==}
            case 1n+bp:
              rt_before_trans(cp, ap, bp)

def rt_lt_irrefl(a: Nat) -> {Nat.is_lt(a, a) == False{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      rt_lt_irrefl(p)

def rt_or_false_r(b: Bool) -> {Bool.or(b, False{}) == b : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

def rt_before_negtrans(+c: Nat, +a: Nat, +b: Nat) -> {Bool.and(Bool.and(Bool.not(rt_evict_before(a, b)), Bool.not(rt_evict_before(b, c))), rt_evict_before(a, c)) == False{} : Bool}:
  match c:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(a, 0n), False{}, rt_lt_zero_false(a)) : {Bool.and(Bool.and(Bool.not(Nat.is_lt(a, b)), Bool.not(Nat.is_lt(b, 0n))), _) == False{} : Bool}
      rt_and_false_r(Bool.and(Bool.not(Nat.is_lt(a, b)), Bool.not(Nat.is_lt(b, 0n))))
    case 1n+cp:
      match a:
        case 0n:
          match b:
            case 0n:
              {==}
            case 1n+_:
              {==}
        case 1n+ap:
          match b:
            case 0n:
              {==}
            case 1n+bp:
              rt_before_negtrans(cp, ap, bp)

def rt_before_asym(+a: Nat, +b: Nat) -> {Bool.and(rt_evict_before(a, b), rt_evict_before(b, a)) == False{} : Bool}:
  match a:
    case 0n:
      match b:
        case 0n:
          {==}
        case 1n+_:
          {==}
    case 1n+ap:
      match b:
        case 0n:
          {==}
        case 1n+bp:
          rt_before_asym(ap, bp)

def rt_pick_on_disk_not_gone(e: Bool) -> {rt_recheck_gone(rt_recheck_pick(True{}, e)) == False{} : Bool}:
  match e:
    case True{}:
      {==}
    case False{}:
      {==}

def rt_pick_kept_not_removed(o: Bool) -> {rt_recheck_removes(rt_recheck_pick(o, False{})) == False{} : Bool}:
  match o:
    case True{}:
      {==}
    case False{}:
      {==}

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

# promise: an entry dated ahead of the clock by less than the keep window is protected.
# harm: a clock step back or a copied file could make fresh recipes look ancient and remove them
law rt_ahead_within_window_protected:
  for +n: Nat
  for +e: Nat
  for +k: Nat
  for +f: Bool
  {rt_protected(Nat.add(n, e), n, 1n+Nat.add(e, k), f) == True{} : Bool}

def rt_ahead_within_window_protected(n, e, k, f):
  %Equal.sym(Nat, Nat.sub(n, Nat.add(n, e)), 0n, rt_sub_ahead_zero(n, e)) : {Bool.or(f, Nat.is_lt(_, 1n+Nat.add(e, k))) == True{} : Bool}
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

# promise: of two entries, at most one leaves before the other.
law rt_evict_before_asymmetric:
  for +a: Nat
  for +b: Nat
  {Bool.and(rt_evict_before(a, b), rt_evict_before(b, a)) == False{} : Bool}

def rt_evict_before_asymmetric(a, b):
  rt_before_asym(a, b)

# promise: no entry leaves before itself, and an entry that leaves before a second one, which leaves before a third, leaves before the third.
# harm: an inconsistent order could make the sort misplace entries and remove recent recipes before old ones
law rt_evict_before_strict_order:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  {Bool.or(Bool.and(Bool.and(rt_evict_before(a, b), rt_evict_before(b, c)), Bool.not(rt_evict_before(a, c))), rt_evict_before(a, a)) == False{} : Bool}

def rt_evict_before_strict_order(a, b, c):
  %Equal.sym(Bool, Nat.is_lt(a, a), False{}, rt_lt_irrefl(a)) : {Bool.or(Bool.and(Bool.and(Nat.is_lt(a, b), Nat.is_lt(b, c)), Bool.not(Nat.is_lt(a, c))), _) == False{} : Bool}
  %Equal.sym(Bool, Bool.or(Bool.and(Bool.and(Nat.is_lt(a, b), Nat.is_lt(b, c)), Bool.not(Nat.is_lt(a, c))), False{}), Bool.and(Bool.and(Nat.is_lt(a, b), Nat.is_lt(b, c)), Bool.not(Nat.is_lt(a, c))), rt_or_false_r(Bool.and(Bool.and(Nat.is_lt(a, b), Nat.is_lt(b, c)), Bool.not(Nat.is_lt(a, c))))) : {_ == False{} : Bool}
  rt_before_trans(c, a, b)

# promise: no entry leaves before itself, and when neither of two entries leaves before the other, and neither of a second pair, the first leaves no earlier than the third: entries that tie form groups, as sorting with a tie-break needs.
# harm: an order that treats near times as ties could make the sort misorder entries or stop
law rt_evict_before_ties_transitive:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  {Bool.or(Bool.and(Bool.and(Bool.not(rt_evict_before(a, b)), Bool.not(rt_evict_before(b, c))), rt_evict_before(a, c)), rt_evict_before(a, a)) == False{} : Bool}

def rt_evict_before_ties_transitive(a, b, c):
  %Equal.sym(Bool, Nat.is_lt(a, a), False{}, rt_lt_irrefl(a)) : {Bool.or(Bool.and(Bool.and(Bool.not(Nat.is_lt(a, b)), Bool.not(Nat.is_lt(b, c))), Nat.is_lt(a, c)), _) == False{} : Bool}
  %Equal.sym(Bool, Bool.or(Bool.and(Bool.and(Bool.not(Nat.is_lt(a, b)), Bool.not(Nat.is_lt(b, c))), Nat.is_lt(a, c)), False{}), Bool.and(Bool.and(Bool.not(Nat.is_lt(a, b)), Bool.not(Nat.is_lt(b, c))), Nat.is_lt(a, c)), rt_or_false_r(Bool.and(Bool.and(Bool.not(Nat.is_lt(a, b)), Bool.not(Nat.is_lt(b, c))), Nat.is_lt(a, c)))) : {_ == False{} : Bool}
  rt_before_negtrans(c, a, b)

# promise: bytes that left the disk come off the count of held bytes exactly.
# harm: the trim's count would drift from the disk and remove too much or too little
law rt_removed_frees_its_bytes:
  for +h: Nat
  for +b: Nat
  {rt_after(Nat.add(h, b), b) == h : Nat}

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

# promise: counting an entry's bytes in and then taking the same bytes off gives back the count before it.
# harm: the starting count could disagree with what the removals free and the trim would stop over or under budget
law rt_count_then_remove_returns:
  for +h: Nat
  for +b: Nat
  {rt_after(rt_count_add(h, b), b) == h : Nat}

def rt_count_then_remove_returns(h, b):
  L.nat_add_sub_cancel_r(h, b)

# promise: an entry whose two files are both gone takes exactly its counted bytes off the count.
# harm: the bytes counted in and the bytes taken off could differ and the trim would stop over or under budget
law rt_both_gone_frees_the_entry:
  for +r: Nat
  for +m: Nat
  {rt_gone_bytes(r, True{}, m, True{}) == rt_entry_bytes(r, m) : Nat}

def rt_both_gone_frees_the_entry(r, m):
  {==}

# promise: a removal never takes more off the count than the entry's counted bytes, whichever of its files are gone.
law rt_gone_bytes_within_entry:
  for +r: Nat
  for +rg: Bool
  for +m: Nat
  for +mg: Bool
  {Nat.is_le(rt_gone_bytes(r, rg, m, mg), rt_entry_bytes(r, m)) == True{} : Bool}

def rt_gone_bytes_within_entry(r, rg, m, mg):
  match rg mg:
    case True{} True{}:
      L.nat_le_refl(Nat.add(r, m))
    case True{} False{}:
      %Equal.sym(Nat, Nat.add(r, 0n), r, L.nat_add_zero_r(r)) : {Nat.is_le(_, Nat.add(r, m)) == True{} : Bool}
      L.nat_add_le_r(r, m)
    case False{} True{}:
      L.nat_add_le_l(r, m)
    case False{} False{}:
      L.nat_le_zero(Nat.add(r, m))

# promise: the recipe file of an entry whose sidecar is still on disk is never removed.
# harm: a failed sidecar removal followed by the recipe's would leave a sidecar that no load can ever match
law rt_recipe_stays_with_its_sidecar:
  {rt_may_remove_recipe(True{}, False{}) == False{} : Bool}

def rt_recipe_stays_with_its_sidecar():
  {==}

# promise: the recipe file of an entry whose sidecar is gone may be removed, so a trim can free recipe bytes.
# harm: a trim could remove only sidecars and never bring the cache under its budget
law rt_recipe_follows_a_gone_sidecar:
  for +p: Bool
  {rt_may_remove_recipe(p, True{}) == True{} : Bool}

def rt_recipe_follows_a_gone_sidecar(p):
  match p:
    case True{}:
      {==}
    case False{}:
      {==}

# promise: the recipe file of an entry that has no sidecar may be removed.
# harm: a recipe left behind by an interrupted compile could never be trimmed
law rt_recipe_without_sidecar_may_go:
  for +g: Bool
  {rt_may_remove_recipe(False{}, g) == True{} : Bool}

def rt_recipe_without_sidecar_may_go(g):
  {==}

# promise: a recipe file still on disk is never counted as gone: what a removal takes off is at most the sidecar's bytes.
# harm: a failed removal would be counted as freed space and the trim would stop over budget
law rt_recipe_on_disk_not_counted:
  for +r: Nat
  for +m: Nat
  for +mg: Bool
  {Nat.is_le(rt_gone_bytes(r, False{}, m, mg), m) == True{} : Bool}

def rt_recipe_on_disk_not_counted(r, m, mg):
  match mg:
    case True{}:
      L.nat_le_refl(m)
    case False{}:
      L.nat_le_zero(m)

# promise: a sidecar still on disk is never counted as gone: what a removal takes off is at most the recipe's bytes.
# harm: a failed removal would be counted as freed space and the trim would stop over budget
law rt_sidecar_on_disk_not_counted:
  for +r: Nat
  for +rg: Bool
  for +m: Nat
  {Nat.is_le(rt_gone_bytes(r, rg, m, False{}), r) == True{} : Bool}

def rt_sidecar_on_disk_not_counted(r, rg, m):
  match rg:
    case True{}:
      %Equal.sym(Nat, Nat.add(r, 0n), r, L.nat_add_zero_r(r)) : {Nat.is_le(_, r) == True{} : Bool}
      L.nat_le_refl(r)
    case False{}:
      L.nat_le_zero(r)

# promise: the count of held bytes never grows during a trim.
law rt_after_never_grows:
  for +h: Nat
  for +b: Nat
  {Nat.is_le(rt_after(h, b), h) == True{} : Bool}

def rt_after_never_grows(h, b):
  rt_sub_le(h, b)

# promise: when a removal leaves both files of an entry on disk, nothing comes off the count of held bytes.
# harm: a failed removal would be counted as freed space and the trim would stop over budget
law rt_nothing_gone_frees_nothing:
  for +r: Nat
  for +m: Nat
  {rt_gone_bytes(r, False{}, m, False{}) == 0n : Nat}

def rt_nothing_gone_frees_nothing(r, m):
  {==}

# promise: when an entry's sidecar is gone and its recipe file stays on disk, exactly the sidecar's bytes come off the count.
# harm: the count would stay above the disk by up to a sidecar per failed recipe removal and the trim would remove more than its budget asks
law rt_sidecar_gone_frees_the_sidecar:
  for +r: Nat
  for +m: Nat
  {rt_gone_bytes(r, False{}, m, True{}) == m : Nat}

def rt_sidecar_gone_frees_the_sidecar(r, m):
  {==}

# promise: when an entry's recipe file is gone and its sidecar stays on disk, exactly the recipe's bytes come off the count.
# harm: the count would stay above the disk by up to a recipe and the trim would remove more than its budget asks
law rt_recipe_gone_frees_the_recipe:
  for +r: Nat
  for +m: Nat
  {rt_gone_bytes(r, True{}, m, False{}) == r : Nat}

def rt_recipe_gone_frees_the_recipe(r, m):
  L.nat_add_zero_r(r)

# promise: an entry whose recipe file is still on disk at the second look is never counted as gone.
# harm: its bytes would come off the count while still on disk and the trim would stop over budget
law rt_recheck_recipe_on_disk_not_gone:
  for +h: Nat
  for +t: Nat
  for +m: Bool
  for +p: Bool
  {rt_recheck_gone(rt_recheck(h, t, True{}, m, p)) == False{} : Bool}

def rt_recheck_recipe_on_disk_not_gone(h, t, m, p):
  rt_pick_on_disk_not_gone(rt_should_evict(h, t, p))

# promise: an entry whose sidecar is still on disk at the second look is never counted as gone.
# harm: its bytes would come off the count while still on disk and the trim would stop over budget
law rt_recheck_sidecar_on_disk_not_gone:
  for +h: Nat
  for +t: Nat
  for +r: Bool
  for +p: Bool
  {rt_recheck_gone(rt_recheck(h, t, r, True{}, p)) == False{} : Bool}

def rt_recheck_sidecar_on_disk_not_gone(h, t, r, p):
  %Equal.sym(Bool, Bool.or(r, True{}), True{}, rt_or_true_r(r)) : {rt_recheck_gone(rt_recheck_pick(_, rt_should_evict(h, t, p))) == False{} : Bool}
  rt_pick_on_disk_not_gone(rt_should_evict(h, t, p))

# promise: an entry neither of whose files is on disk at the second look is counted as gone.
# harm: a kept entry that is gone would stay counted as held and the trim would remove more than its budget asks
law rt_recheck_nothing_on_disk_gone:
  for +h: Nat
  for +t: Nat
  for +p: Bool
  {rt_recheck_gone(rt_recheck(h, t, False{}, False{}, p)) == True{} : Bool}

def rt_recheck_nothing_on_disk_gone(h, t, p):
  {==}

# promise: a protected entry is never removed, whatever the second look found.
# harm: the keep window and the entries a server holds would be ignored by the trim
law rt_recheck_protected_stays:
  for +h: Nat
  for +t: Nat
  for +r: Bool
  for +m: Bool
  {rt_recheck_removes(rt_recheck(h, t, r, m, True{})) == False{} : Bool}

def rt_recheck_protected_stays(h, t, r, m):
  rt_pick_kept_not_removed(Bool.or(r, m))

# promise: no entry is removed while the cache is within its target, whatever the second look found.
# harm: a trim could empty a cache that already fits and force every recipe to compile again
law rt_recheck_within_target_stays:
  for +h: Nat
  for +x: Nat
  for +r: Bool
  for +m: Bool
  for +p: Bool
  {rt_recheck_removes(rt_recheck(h, Nat.add(h, x), r, m, p)) == False{} : Bool}

def rt_recheck_within_target_stays(h, x, r, m, p):
  %Equal.sym(Bool, Nat.is_lt(Nat.add(h, x), h), False{}, rt_add_lt_false(h, x)) : {rt_recheck_removes(rt_recheck_pick(Bool.or(r, m), Bool.and(Bool.not(p), _))) == False{} : Bool}
  %Equal.sym(Bool, Bool.and(Bool.not(p), False{}), False{}, rt_and_false_r(Bool.not(p))) : {rt_recheck_removes(rt_recheck_pick(Bool.or(r, m), _)) == False{} : Bool}
  rt_pick_kept_not_removed(Bool.or(r, m))

# promise: an unprotected entry whose recipe file is on disk at the second look is removed while the cache is over its target.
# harm: a trim could stop early and leave the disk over the budget it was given
law rt_recheck_over_target_removes:
  for +t: Nat
  for +x: Nat
  for +m: Bool
  {rt_recheck_removes(rt_recheck(Nat.add(t, 1n+x), t, True{}, m, False{})) == True{} : Bool}

def rt_recheck_over_target_removes(t, x, m):
  %Equal.sym(Bool, Nat.is_lt(t, Nat.add(t, 1n+x)), True{}, rt_lt_add_succ(t, x)) : {rt_recheck_removes(rt_recheck_pick(True{}, Bool.and(True{}, _))) == True{} : Bool}
  {==}
