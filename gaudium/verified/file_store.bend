# Slice — Verified File Store Ledger (Files API)
# Formally specifies and proves the byte and id accounting of the server's file store
# (`/v1/files`, `server/file_store.rs`):
#   - used + free stays equal to the quota across every upload, stored or refused, and every delete
#   - an upload larger than the per-file cap, or than the free bytes, is never stored; one that
#     fits both is
#   - a deleted file frees its bytes once: deleting it again changes nothing
#   - a file id is never handed out twice: every stored id is at or above the ledger's counter and
#     below the counter after it; deletes keep the counter; a restart restores it past every id on
#     disk and never below the counter it saved; no id at or past the id limit is handed out
#   - a store starts only when the files found on disk fit its quota, and then used + free is the
#     quota
#   - an upload reserves its room before its bytes are read: the reservation is counted as used,
#     never takes more than the free bytes, and is given back once; an upload larger than its
#     reservation is never stored
#   - a file is shown only to its owner, and only while it is live
# The ledger holds `used` and `free` (their sum is the quota) and `next`, the lowest id not yet
# handed out. `used` counts the stored files and the reservations of uploads in progress.
# Streaming the bytes to disk, the metadata files, the owner digests and the HTTP shapes stay in
# the shell.

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. The ledger
# ============================================================================

type FsLedger is Data:
  FsLedger{used: Nat, free: Nat, next: Nat}

# One stored file as the ledger sees it: its size, and whether it is still counted.
type FsEntry is Data:
  FsEntry{size: Nat, live: Bool}

# What an upload of `size` bytes comes to: stored under `id` (and the ledger after it), or
# refused because it is larger than the per-file cap, or than the `free` bytes of the quota.
type FsUpload is Data:
  FsStored{id: Nat, after: FsLedger}
  FsTooLarge{}
  FsOverQuota{free: Nat}
  FsNoIds{}

# An empty store of `quota` bytes whose next id is `next`.
def fs_open(+quota: Nat, +next: Nat) -> FsLedger:
  FsLedger{0n, quota, next}

def fs_used(l: FsLedger) -> Nat:
  match l:
    case FsLedger{u, _, _}:
      u

def fs_free(l: FsLedger) -> Nat:
  match l:
    case FsLedger{_, f, _}:
      f

def fs_next(l: FsLedger) -> Nat:
  match l:
    case FsLedger{_, _, n}:
      n

# The quota the ledger accounts for: used + free.
def fs_quota(l: FsLedger) -> Nat:
  match l:
    case FsLedger{u, f, _}:
      Nat.add(u, f)

def fs_store_if(+u: Nat, +f: Nat, +n: Nat, +size: Nat, fits: Bool) -> FsUpload:
  match fits:
    case True{}:
      FsStored{n, FsLedger{Nat.add(u, size), Nat.sub(f, size), 1n+n}}
    case False{}:
      FsOverQuota{f}

def fs_upload_go(+u: Nat, +f: Nat, +n: Nat, +size: Nat, in_cap: Bool) -> FsUpload:
  match in_cap:
    case True{}:
      fs_store_if(u, f, n, size, Nat.is_le(size, f))
    case False{}:
      FsTooLarge{}

# An upload of `size` bytes against the per-file cap `cap` and the free bytes.
def fs_upload(l: FsLedger, +cap: Nat, +size: Nat) -> FsUpload:
  match l:
    case FsLedger{+u, +f, +n}:
      fs_upload_go(u, f, n, size, Nat.is_le(size, cap))

# The ledger after an upload decided against `l`: the stored ledger, or `l` itself.
def fs_upload_ledger(+l: FsLedger, r: FsUpload) -> FsLedger:
  match r:
    case FsStored{_, a}:
      a
    case FsTooLarge{}:
      l
    case FsOverQuota{_}:
      l
    case FsNoIds{}:
      l

# Whether an upload was stored.
def fs_is_stored(r: FsUpload) -> Bool:
  match r:
    case FsStored{_, _}:
      True{}
    case FsTooLarge{}:
      False{}
    case FsOverQuota{_}:
      False{}
    case FsNoIds{}:
      False{}

# A stored id is at or above the counter `n` it was decided against, and below `m` (the counter of
# the ledger after it).
def fs_id_fresh(+n: Nat, r: FsUpload, +m: Nat) -> Bool:
  match r:
    case FsStored{+id, _}:
      Bool.and(Nat.is_le(n, id), Nat.is_lt(id, m))
    case FsTooLarge{}:
      True{}
    case FsOverQuota{_}:
      True{}
    case FsNoIds{}:
      True{}

# The bytes an upload of `size` bytes takes: its size when stored, none when refused.
def fs_taken(r: FsUpload, +size: Nat) -> Nat:
  match r:
    case FsStored{_, _}:
      size
    case FsTooLarge{}:
      0n
    case FsOverQuota{_}:
      0n
    case FsNoIds{}:
      0n

# Whether each refusal of an upload of `size` bytes is true: too large only when it is larger than
# `cap`, over the quota only when it is larger than `room`. Other refusals are left to other laws.
def fs_refused_rightly(+size: Nat, +cap: Nat, +room: Nat, r: FsUpload) -> Bool:
  match r:
    case FsStored{_, _}:
      True{}
    case FsTooLarge{}:
      Bool.not(Nat.is_le(size, cap))
    case FsOverQuota{_}:
      Bool.not(Nat.is_le(size, room))
    case FsNoIds{}:
      True{}

# Whether an upload's id decision is true against the counter `n` and the id limit `lim`: a stored
# id is below the limit, and the ids run out only once the counter has reached it.
def fs_ids_rightly(+n: Nat, +lim: Nat, r: FsUpload) -> Bool:
  match r:
    case FsStored{+id, _}:
      Nat.is_lt(id, lim)
    case FsTooLarge{}:
      True{}
    case FsOverQuota{_}:
      True{}
    case FsNoIds{}:
      Bool.not(Nat.is_lt(n, lim))

# `u` when `r` stores the upload, `r` itself when it refuses it.
def fs_when_stored(r: FsUpload, +u: FsUpload) -> FsUpload:
  match r:
    case FsStored{_, _}:
      u
    case FsTooLarge{}:
      FsTooLarge{}
    case FsOverQuota{+x}:
      FsOverQuota{x}
    case FsNoIds{}:
      FsNoIds{}

# `a` when `r` stores the upload, `l` when it refuses it (whatever the refusal).
def fs_kept_if_refused(r: FsUpload, +a: FsLedger, +l: FsLedger) -> FsLedger:
  match r:
    case FsStored{_, _}:
      a
    case FsTooLarge{}:
      l
    case FsOverQuota{_}:
      l
    case FsNoIds{}:
      l

def fs_release_if(+u: Nat, +f: Nat, +n: Nat, +size: Nat, counted: Bool) -> FsLedger:
  match counted:
    case True{}:
      FsLedger{Nat.sub(u, size), Nat.add(f, size), n}
    case False{}:
      FsLedger{u, f, n}

def fs_release(l: FsLedger, +size: Nat) -> FsLedger:
  match l:
    case FsLedger{+u, +f, +n}:
      fs_release_if(u, f, n, size, Nat.is_le(size, u))

def fs_delete_go(+l: FsLedger, +size: Nat, live: Bool) -> FsLedger:
  match live:
    case True{}:
      fs_release(l, size)
    case False{}:
      l

# Deleting a file: a live entry gives its bytes back (when the ledger counts them); an entry
# already deleted changes nothing.
def fs_delete(+l: FsLedger, e: FsEntry) -> FsLedger:
  match e:
    case FsEntry{+size, live}:
      fs_delete_go(l, size, live)

# Whether the ledger still counts the entry's bytes (a deleted file is not served).
def fs_live(e: FsEntry) -> Bool:
  match e:
    case FsEntry{_, l}:
      l

# The entry after its delete: no longer counted.
def fs_kill(e: FsEntry) -> FsEntry:
  match e:
    case FsEntry{size, _}:
      FsEntry{size, False{}}

def fs_size(e: FsEntry) -> Nat:
  match e:
    case FsEntry{s, _}:
      s

# Whether a stored file is shown to a caller: it is live, the caller is its owner (`same_owner`:
# the caller's key digest is the one that stored it, or neither has a key), and the caller named it
# (`same_id`; a list names every file).
def fs_visible(e: FsEntry, same_owner: Bool, same_id: Bool) -> Bool:
  Bool.and(fs_live(e), Bool.and(same_owner, same_id))

# Counting a file found on disk at start (`id`, `size` bytes): its bytes are used whether or not
# they fit a quota lowered since, and the counter moves past its id.
def fs_restore(l: FsLedger, +id: Nat, +size: Nat) -> FsLedger:
  match l:
    case FsLedger{+u, +f, +n}:
      FsLedger{Nat.add(u, size), Nat.sub(f, size), Nat.max(n, 1n+id)}

# What a start comes to, once the files on disk are counted: the ledger to serve with, or a refusal
# because the files already hold more than the configured quota.
type FsStart is Data:
  FsReady{ledger: FsLedger}
  FsOverfull{}

def fs_start_go(+u: Nat, +n: Nat, +quota: Nat, fits: Bool) -> FsStart:
  match fits:
    case True{}:
      FsReady{FsLedger{u, Nat.sub(quota, u), n}}
    case False{}:
      FsOverfull{}

# Starting the store against the configured `quota`, from the ledger `l` that counted the files
# found on disk: ready with every byte the files do not use free, or refused when they use more
# than the quota (a quota lowered since they were stored).
def fs_start(l: FsLedger, +quota: Nat) -> FsStart:
  match l:
    case FsLedger{+u, _, +n}:
      fs_start_go(u, n, quota, Nat.is_le(u, quota))

def fs_is_ready(r: FsStart) -> Bool:
  match r:
    case FsReady{_}:
      True{}
    case FsOverfull{}:
      False{}

# The ledger a ready start serves with (an empty ledger for a refused one).
def fs_start_ledger(r: FsStart) -> FsLedger:
  match r:
    case FsReady{l}:
      l
    case FsOverfull{}:
      FsLedger{0n, 0n, 0n}

# A reservation: the ticket (its bytes, live until given back) and the ledger that counts it.
type FsHold is Data:
  FsHeld{ticket: FsEntry, after: FsLedger}

def fs_hold_go(+u: Nat, +f: Nat, +n: Nat, +g: Nat) -> FsHold:
  FsHeld{FsEntry{g, True{}}, FsLedger{Nat.add(u, g), Nat.sub(f, g), n}}

# Reserving room for an upload before its bytes are read: `want` bytes (its declared size, or the
# cap when it has none), at most the per-file cap and the free bytes. The reserved bytes count as
# used until the ticket is given back (`fs_delete`), once, when the upload is committed or dropped.
def fs_hold(l: FsLedger, +cap: Nat, +want: Nat) -> FsHold:
  match l:
    case FsLedger{+u, +f, +n}:
      fs_hold_go(u, f, n, Nat.min(Nat.min(want, cap), f))

def fs_hold_ticket(h: FsHold) -> FsEntry:
  match h:
    case FsHeld{t, _}:
      t

def fs_hold_ledger(h: FsHold) -> FsLedger:
  match h:
    case FsHeld{_, a}:
      a

def fs_commit_ids(+l: FsLedger, +cap: Nat, +size: Nat, ids_left: Bool) -> FsUpload:
  match ids_left:
    case True{}:
      fs_upload(l, cap, size)
    case False{}:
      FsNoIds{}

# An upload that ran past its reservation (`held` bytes): its bytes were not all read.
def fs_commit_past(+held: Nat, +cap: Nat, +size: Nat, in_cap: Bool) -> FsUpload:
  match in_cap:
    case True{}:
      FsOverQuota{held}
    case False{}:
      FsTooLarge{}

def fs_commit_go(+l: FsLedger, +cap: Nat, +lim: Nat, +held: Nat, +size: Nat, in_held: Bool) -> FsUpload:
  match in_held:
    case True{}:
      fs_commit_ids(l, cap, size, Nat.is_lt(fs_next(l), lim))
    case False{}:
      fs_commit_past(held, cap, size, Nat.is_le(size, cap))

# The decision on a staged upload of `size` bytes whose reservation is the ticket `t`, against the
# ledger `l` the ticket was already given back to, the per-file cap and the id limit `lim` (no id
# at or past it is handed out): refused when the upload is larger than its reservation (it was not
# read in full) or the ids have run out, else as `fs_upload` decides.
def fs_commit(+l: FsLedger, +cap: Nat, +lim: Nat, t: FsEntry, +size: Nat) -> FsUpload:
  match t:
    case FsEntry{+held, _}:
      fs_commit_go(l, cap, lim, held, size, Nat.is_le(size, held))

# ============================================================================
# 2. Proof helpers (erased)
# ============================================================================

def fs_store_step_eq(+u: Nat, +fp: Nat, +n: Nat, +sp: Nat, ok: Bool) -> {fs_quota(fs_upload_ledger(FsLedger{u, 1n+fp, n}, fs_store_if(u, 1n+fp, n, 1n+sp, ok))) == fs_quota(fs_upload_ledger(FsLedger{1n+u, fp, n}, fs_store_if(1n+u, fp, n, sp, ok))) : Nat}:
  match ok:
    case True{}:
      %L.nat_add_succ_r(u, sp) : {Nat.add(Nat.add(u, 1n+sp), Nat.sub(fp, sp)) == Nat.add(_, Nat.sub(fp, sp)) : Nat}
      {==}
    case False{}:
      L.nat_add_succ_r(u, fp)

def fs_store_conserved(+f: Nat, +s: Nat, +u: Nat, +n: Nat) -> {fs_quota(fs_upload_ledger(FsLedger{u, f, n}, fs_store_if(u, f, n, s, Nat.is_le(s, f)))) == Nat.add(u, f) : Nat}:
  match f s:
    case 0n 0n:
      L.nat_add_zero_r(Nat.add(u, 0n))
    case 0n 1n+sp:
      {==}
    case 1n+fp 0n:
      %L.nat_add_zero_r(u) : {Nat.add(Nat.add(u, 0n), 1n+fp) == Nat.add(_, 1n+fp) : Nat}
      {==}
    case 1n+fp 1n+sp:
      Equal.trans(
        Nat,
        fs_quota(fs_upload_ledger(FsLedger{u, 1n+fp, n}, fs_store_if(u, 1n+fp, n, 1n+sp, Nat.is_le(sp, fp)))),
        fs_quota(fs_upload_ledger(FsLedger{1n+u, fp, n}, fs_store_if(1n+u, fp, n, sp, Nat.is_le(sp, fp)))),
        Nat.add(u, 1n+fp),
        fs_store_step_eq(u, fp, n, sp, Nat.is_le(sp, fp)),
        Equal.trans(
          Nat,
          fs_quota(fs_upload_ledger(FsLedger{1n+u, fp, n}, fs_store_if(1n+u, fp, n, sp, Nat.is_le(sp, fp)))),
          1n+Nat.add(u, fp),
          Nat.add(u, 1n+fp),
          fs_store_conserved(fp, sp, 1n+u, n),
          Equal.sym(Nat, Nat.add(u, 1n+fp), 1n+Nat.add(u, fp), L.nat_add_succ_r(u, fp))
        )
      )

def fs_go_conserved(+u: Nat, +f: Nat, +n: Nat, +s: Nat, b: Bool) -> {fs_quota(fs_upload_ledger(FsLedger{u, f, n}, fs_upload_go(u, f, n, s, b))) == Nat.add(u, f) : Nat}:
  match b:
    case True{}:
      fs_store_conserved(f, s, u, n)
    case False{}:
      {==}

def fs_go_over_free(+u: Nat, +f: Nat, +n: Nat, +x: Nat, b: Bool) -> {fs_is_stored(fs_upload_go(u, f, n, Nat.add(f, 1n+x), b)) == False{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Bool, Nat.is_le(Nat.add(f, 1n+x), f), False{}, G.g_nat_le_add_succ_false(f, x)) : {fs_is_stored(fs_store_if(u, f, n, Nat.add(f, 1n+x), _)) == False{} : Bool}
      {==}
    case False{}:
      {==}

def fs_store_fresh(+u: Nat, +f: Nat, +n: Nat, +s: Nat, b: Bool) -> {fs_id_fresh(n, fs_store_if(u, f, n, s, b), fs_next(fs_upload_ledger(FsLedger{u, f, n}, fs_store_if(u, f, n, s, b)))) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Bool, Nat.is_le(n, n), True{}, L.nat_le_refl(n)) : {Bool.and(_, Nat.is_lt(n, 1n+n)) == True{} : Bool}
      L.nat_lt_succ_self(n)
    case False{}:
      {==}

def fs_go_fresh(+u: Nat, +f: Nat, +n: Nat, +s: Nat, b: Bool) -> {fs_id_fresh(n, fs_upload_go(u, f, n, s, b), fs_next(fs_upload_ledger(FsLedger{u, f, n}, fs_upload_go(u, f, n, s, b)))) == True{} : Bool}:
  match b:
    case True{}:
      fs_store_fresh(u, f, n, s, Nat.is_le(s, f))
    case False{}:
      {==}

def fs_store_takes(+u: Nat, +f: Nat, +n: Nat, +s: Nat, b: Bool) -> {fs_used(fs_upload_ledger(FsLedger{u, f, n}, fs_store_if(u, f, n, s, b))) == Nat.add(u, fs_taken(fs_store_if(u, f, n, s, b), s)) : Nat}:
  match b:
    case True{}:
      {==}
    case False{}:
      Equal.sym(Nat, Nat.add(u, 0n), u, L.nat_add_zero_r(u))

def fs_go_takes(+u: Nat, +f: Nat, +n: Nat, +s: Nat, b: Bool) -> {fs_used(fs_upload_ledger(FsLedger{u, f, n}, fs_upload_go(u, f, n, s, b))) == Nat.add(u, fs_taken(fs_upload_go(u, f, n, s, b), s)) : Nat}:
  match b:
    case True{}:
      fs_store_takes(u, f, n, s, Nat.is_le(s, f))
    case False{}:
      Equal.sym(Nat, Nat.add(u, 0n), u, L.nat_add_zero_r(u))

# `s <= a` is false when `s <= b` is and `a <= b`.
def fs_not_le_mono_go(+s: Nat, +a: Nat, +b: Nat, k: Bool, ek: {k == Nat.is_le(s, a) : Bool}, hab: {Nat.is_le(a, b) == True{} : Bool}, h: {Nat.is_le(s, b) == False{} : Bool}) -> {Nat.is_le(s, a) == False{} : Bool}:
  match k:
    case True{}:
      +t = L.nat_le_trans(s, a, b, Equal.sym(Bool, True{}, Nat.is_le(s, a), ek), hab)
      L.bool_absurd_tf({Nat.is_le(s, a) == False{} : Bool}, Equal.trans(Bool, True{}, Nat.is_le(s, b), False{}, Equal.sym(Bool, Nat.is_le(s, b), True{}, t), h))
    case False{}:
      Equal.sym(Bool, False{}, Nat.is_le(s, a), ek)

def fs_not_le_mono(+s: Nat, +a: Nat, +b: Nat, hab: {Nat.is_le(a, b) == True{} : Bool}, h: {Nat.is_le(s, b) == False{} : Bool}) -> {Nat.is_le(s, a) == False{} : Bool}:
  fs_not_le_mono_go(s, a, b, Nat.is_le(s, a), {==}, hab, h)

def fs_store_rightly(+u: Nat, +f: Nat, +n: Nat, +s: Nat, +c: Nat, +room: Nat, k: Bool, ek: {k == Nat.is_le(s, f) : Bool}, hr: {Nat.is_le(room, f) == True{} : Bool}) -> {fs_refused_rightly(s, c, room, fs_store_if(u, f, n, s, k)) == True{} : Bool}:
  match k:
    case True{}:
      {==}
    case False{}:
      %Equal.sym(Bool, Nat.is_le(s, room), False{}, fs_not_le_mono(s, room, f, hr, Equal.sym(Bool, False{}, Nat.is_le(s, f), ek))) : {Bool.not(_) == True{} : Bool}
      {==}

def fs_go_rightly(+u: Nat, +f: Nat, +n: Nat, +s: Nat, +c: Nat, +room: Nat, k: Bool, ek: {k == Nat.is_le(s, c) : Bool}, hr: {Nat.is_le(room, f) == True{} : Bool}) -> {fs_refused_rightly(s, c, room, fs_upload_go(u, f, n, s, k)) == True{} : Bool}:
  match k:
    case True{}:
      fs_store_rightly(u, f, n, s, c, room, Nat.is_le(s, f), {==}, hr)
    case False{}:
      %ek : {Bool.not(_) == True{} : Bool}
      {==}

def fs_start_go_fits(+u: Nat, +n: Nat, +q: Nat, k: Bool, ek: {k == Nat.is_le(u, q) : Bool}) -> {Bool.and(fs_is_ready(fs_start_go(u, n, q, k)), Nat.is_lt(q, u)) == False{} : Bool}:
  match k:
    case True{}:
      L.nat_le_lt_false(q, u, Equal.sym(Bool, True{}, Nat.is_le(u, q), ek))
    case False{}:
      {==}

def fs_take_conserved(+u: Nat, +f: Nat, +g: Nat, h: {Nat.is_le(g, f) == True{} : Bool}) -> {Nat.add(Nat.add(u, g), Nat.sub(f, g)) == Nat.add(u, f) : Nat}:
  %Equal.sym(Nat, Nat.add(Nat.add(u, g), Nat.sub(f, g)), Nat.add(u, Nat.add(g, Nat.sub(f, g))), L.nat_add_assoc(u, g, Nat.sub(f, g))) : {_ == Nat.add(u, f) : Nat}
  %Equal.sym(Nat, Nat.add(g, Nat.sub(f, g)), Nat.add(Nat.sub(f, g), g), L.nat_add_comm(g, Nat.sub(f, g))) : {Nat.add(u, _) == Nat.add(u, f) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.sub(f, g), g), f, L.nat_sub_add_cancel(f, g, h)) : {Nat.add(u, _) == Nat.add(u, f) : Nat}
  {==}

def fs_hold_back(+u: Nat, +f: Nat, +n: Nat, +g: Nat, h: {Nat.is_le(g, f) == True{} : Bool}) -> {fs_delete(fs_hold_ledger(fs_hold_go(u, f, n, g)), fs_hold_ticket(fs_hold_go(u, f, n, g))) == FsLedger{u, f, n} : FsLedger}:
  %Equal.sym(Bool, Nat.is_le(g, Nat.add(u, g)), True{}, L.nat_add_le_l(u, g)) : {fs_release_if(Nat.add(u, g), Nat.sub(f, g), n, g, _) == FsLedger{u, f, n} : FsLedger}
  %Equal.sym(Nat, Nat.sub(Nat.add(u, g), g), u, L.nat_add_sub_cancel_r(u, g)) : {FsLedger{_, Nat.add(Nat.sub(f, g), g), n} == FsLedger{u, f, n} : FsLedger}
  %Equal.sym(Nat, Nat.add(Nat.sub(f, g), g), f, L.nat_sub_add_cancel(f, g, h)) : {FsLedger{u, _, n} == FsLedger{u, f, n} : FsLedger}
  {==}

def fs_when_self(r: FsUpload) -> {fs_when_stored(r, r) == r : FsUpload}:
  match r:
    case FsStored{+i, +a}:
      {==}
    case FsTooLarge{}:
      {==}
    case FsOverQuota{+x}:
      {==}
    case FsNoIds{}:
      {==}

def fs_ids_is(+l: FsLedger, +c: Nat, +s: Nat, b: Bool) -> {fs_when_stored(fs_commit_ids(l, c, s, b), fs_upload(l, c, s)) == fs_commit_ids(l, c, s, b) : FsUpload}:
  match b:
    case True{}:
      fs_when_self(fs_upload(l, c, s))
    case False{}:
      {==}

def fs_past_is(+h: Nat, +c: Nat, +s: Nat, +r: FsUpload, b: Bool) -> {fs_when_stored(fs_commit_past(h, c, s, b), r) == fs_commit_past(h, c, s, b) : FsUpload}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

def fs_commit_go_is(+l: FsLedger, +c: Nat, +m: Nat, +h: Nat, +s: Nat, b: Bool) -> {fs_when_stored(fs_commit_go(l, c, m, h, s, b), fs_upload(l, c, s)) == fs_commit_go(l, c, m, h, s, b) : FsUpload}:
  match b:
    case True{}:
      fs_ids_is(l, c, s, Nat.is_lt(fs_next(l), m))
    case False{}:
      fs_past_is(h, c, s, fs_upload(l, c, s), Nat.is_le(s, c))

def fs_past_unstored(+h: Nat, +c: Nat, +s: Nat, b: Bool) -> {fs_is_stored(fs_commit_past(h, c, s, b)) == False{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

def fs_store_ids(+u: Nat, +f: Nat, +n: Nat, +s: Nat, +m: Nat, hn: {Nat.is_lt(n, m) == True{} : Bool}, b: Bool) -> {fs_ids_rightly(n, m, fs_store_if(u, f, n, s, b)) == True{} : Bool}:
  match b:
    case True{}:
      hn
    case False{}:
      {==}

def fs_upgo_ids(+u: Nat, +f: Nat, +n: Nat, +s: Nat, +m: Nat, hn: {Nat.is_lt(n, m) == True{} : Bool}, b: Bool) -> {fs_ids_rightly(n, m, fs_upload_go(u, f, n, s, b)) == True{} : Bool}:
  match b:
    case True{}:
      fs_store_ids(u, f, n, s, m, hn, Nat.is_le(s, f))
    case False{}:
      {==}

def fs_cids_ids(+u: Nat, +f: Nat, +n: Nat, +c: Nat, +s: Nat, +m: Nat, k: Bool, ek: {k == Nat.is_lt(n, m) : Bool}) -> {fs_ids_rightly(n, m, fs_commit_ids(FsLedger{u, f, n}, c, s, k)) == True{} : Bool}:
  match k:
    case True{}:
      fs_upgo_ids(u, f, n, s, m, Equal.sym(Bool, True{}, Nat.is_lt(n, m), ek), Nat.is_le(s, c))
    case False{}:
      %ek : {Bool.not(_) == True{} : Bool}
      {==}

def fs_past_ids(+n: Nat, +m: Nat, +h: Nat, +c: Nat, +s: Nat, b: Bool) -> {fs_ids_rightly(n, m, fs_commit_past(h, c, s, b)) == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

def fs_cgo_ids(+u: Nat, +f: Nat, +n: Nat, +c: Nat, +m: Nat, +h: Nat, +s: Nat, b: Bool) -> {fs_ids_rightly(n, m, fs_commit_go(FsLedger{u, f, n}, c, m, h, s, b)) == True{} : Bool}:
  match b:
    case True{}:
      fs_cids_ids(u, f, n, c, s, m, Nat.is_lt(n, m), {==})
    case False{}:
      fs_past_ids(n, m, h, c, s, Nat.is_le(s, c))

def fs_cids_rightly(+u: Nat, +f: Nat, +n: Nat, +c: Nat, +s: Nat, +room: Nat, hr: {Nat.is_le(room, f) == True{} : Bool}, b: Bool) -> {fs_refused_rightly(s, c, room, fs_commit_ids(FsLedger{u, f, n}, c, s, b)) == True{} : Bool}:
  match b:
    case True{}:
      fs_go_rightly(u, f, n, s, c, room, Nat.is_le(s, c), {==}, hr)
    case False{}:
      {==}

def fs_past_rightly(+h: Nat, +c: Nat, +s: Nat, +room: Nat, hroom: {Nat.is_le(s, room) == False{} : Bool}, k: Bool, ek: {k == Nat.is_le(s, c) : Bool}) -> {fs_refused_rightly(s, c, room, fs_commit_past(h, c, s, k)) == True{} : Bool}:
  match k:
    case True{}:
      %Equal.sym(Bool, Nat.is_le(s, room), False{}, hroom) : {Bool.not(_) == True{} : Bool}
      {==}
    case False{}:
      %ek : {Bool.not(_) == True{} : Bool}
      {==}

def fs_cgo_rightly(+u: Nat, +f: Nat, +n: Nat, +c: Nat, +m: Nat, +h: Nat, +s: Nat, k: Bool, ek: {k == Nat.is_le(s, h) : Bool}) -> {fs_refused_rightly(s, c, Nat.min(f, h), fs_commit_go(FsLedger{u, f, n}, c, m, h, s, k)) == True{} : Bool}:
  match k:
    case True{}:
      fs_cids_rightly(u, f, n, c, s, Nat.min(f, h), L.nat_min_le_l(f, h), Nat.is_lt(n, m))
    case False{}:
      fs_past_rightly(h, c, s, Nat.min(f, h), fs_not_le_mono(s, Nat.min(f, h), h, L.nat_min_le_r(f, h), Equal.sym(Bool, False{}, Nat.is_le(s, h), ek)), Nat.is_le(s, c), {==})

def fs_lt_add_succ(+n: Nat, +z: Nat) -> {Nat.is_lt(n, Nat.add(n, 1n+z)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(n, 1n+z), 1n+Nat.add(n, z), L.nat_add_succ_r(n, z)) : {Nat.is_lt(n, _) == True{} : Bool}
  Equal.trans(Bool, Nat.is_lt(n, 1n+Nat.add(n, z)), Nat.is_le(n, Nat.add(n, z)), True{}, L.nat_lt_succ_eq_le(n, Nat.add(n, z)), L.nat_add_le_r(n, z))

def fs_release_step_eq(+up: Nat, +f: Nat, +n: Nat, +sp: Nat, ok: Bool) -> {fs_quota(fs_release_if(1n+up, f, n, 1n+sp, ok)) == fs_quota(fs_release_if(up, 1n+f, n, sp, ok)) : Nat}:
  match ok:
    case True{}:
      %L.nat_add_succ_r(f, sp) : {Nat.add(Nat.sub(up, sp), Nat.add(f, 1n+sp)) == Nat.add(Nat.sub(up, sp), _) : Nat}
      {==}
    case False{}:
      Equal.sym(Nat, Nat.add(up, 1n+f), 1n+Nat.add(up, f), L.nat_add_succ_r(up, f))

def fs_release_conserved(+u: Nat, +f: Nat, +n: Nat, +s: Nat) -> {fs_quota(fs_release_if(u, f, n, s, Nat.is_le(s, u))) == Nat.add(u, f) : Nat}:
  match u s:
    case 0n 0n:
      L.nat_add_zero_r(f)
    case 0n 1n+sp:
      {==}
    case 1n+up 0n:
      %L.nat_add_zero_r(f) : {1n+Nat.add(up, Nat.add(f, 0n)) == 1n+Nat.add(up, _) : Nat}
      {==}
    case 1n+up 1n+sp:
      Equal.trans(
        Nat,
        fs_quota(fs_release_if(1n+up, f, n, 1n+sp, Nat.is_le(sp, up))),
        fs_quota(fs_release_if(up, 1n+f, n, sp, Nat.is_le(sp, up))),
        1n+Nat.add(up, f),
        fs_release_step_eq(up, f, n, sp, Nat.is_le(sp, up)),
        Equal.trans(
          Nat,
          fs_quota(fs_release_if(up, 1n+f, n, sp, Nat.is_le(sp, up))),
          Nat.add(up, 1n+f),
          1n+Nat.add(up, f),
          fs_release_conserved(up, 1n+f, n, sp),
          L.nat_add_succ_r(up, f)
        )
      )

def fs_release_next(+u: Nat, +f: Nat, +n: Nat, +s: Nat, b: Bool) -> {fs_next(fs_release_if(u, f, n, s, b)) == n : Nat}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

def fs_release_l_conserved(l: FsLedger, +s: Nat) -> {fs_quota(fs_release(l, s)) == fs_quota(l) : Nat}:
  match l:
    case FsLedger{+u, +f, +n}:
      fs_release_conserved(u, f, n, s)

def fs_delete_go_conserved(+l: FsLedger, +s: Nat, live: Bool) -> {fs_quota(fs_delete_go(l, s, live)) == fs_quota(l) : Nat}:
  match live:
    case True{}:
      fs_release_l_conserved(l, s)
    case False{}:
      {==}

def fs_release_l_next(l: FsLedger, +s: Nat) -> {fs_next(fs_release(l, s)) == fs_next(l) : Nat}:
  match l:
    case FsLedger{+u, +f, +n}:
      fs_release_next(u, f, n, s, Nat.is_le(s, u))

def fs_delete_go_next(+l: FsLedger, +s: Nat, live: Bool) -> {fs_next(fs_delete_go(l, s, live)) == fs_next(l) : Nat}:
  match live:
    case True{}:
      fs_release_l_next(l, s)
    case False{}:
      {==}

def fs_lt_of_succ_le(+i: Nat, +m: Nat, h: {Nat.is_le(1n+i, m) == True{} : Bool}) -> {Nat.is_lt(i, m) == True{} : Bool}:
  match i m:
    case 0n 0n:
      L.bool_absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, h)
    case 0n 1n+mp:
      {==}
    case 1n+ip 0n:
      L.bool_absurd({Nat.is_lt(1n+ip, 0n) == True{} : Bool}, h)
    case 1n+ip 1n+mp:
      fs_lt_of_succ_le(ip, mp, h)

def fs_ledger_kept(+l: FsLedger, r: FsUpload) -> {fs_kept_if_refused(r, fs_upload_ledger(l, r), l) == fs_upload_ledger(l, r) : FsLedger}:
  match r:
    case FsStored{+i, +a}:
      {==}
    case FsTooLarge{}:
      {==}
    case FsOverQuota{+x}:
      {==}
    case FsNoIds{}:
      {==}

def fs_min_add_self(+a: Nat, +b: Nat) -> {Nat.min(a, Nat.add(a, b)) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+ap:
      %Equal.sym(Nat, Nat.min(ap, Nat.add(ap, b)), ap, fs_min_add_self(ap, b)) : {1n+_ == 1n+ap : Nat}
      {==}

# ============================================================================
# 3. Laws
# ============================================================================

# promise: an open store has every byte of its quota free and none used.
# catches: the open ledger starts with its free bytes at zero
law fs_open_all_free:
  for +q: Nat
  for +n: Nat
  {fs_free(fs_open(q, n)) == q : Nat}

def fs_open_all_free(q, n):
  {==}

# promise: an open store counts no bytes as used.
# catches: the open ledger starts with the quota counted as used
law fs_open_none_used:
  for +q: Nat
  for +n: Nat
  {fs_used(fs_open(q, n)) == 0n : Nat}

def fs_open_none_used(q, n):
  {==}

# promise: an upload, stored or refused, leaves used + free equal to the quota it started from.
# harm: the store could hold more than its quota, or lose free bytes it never handed out
# catches: the free bytes not reduced by the size of a stored upload
law fs_upload_conserved:
  for +u: Nat
  for +f: Nat
  for +n: Nat
  for +c: Nat
  for +s: Nat
  {fs_quota(fs_upload_ledger(FsLedger{u, f, n}, fs_upload(FsLedger{u, f, n}, c, s))) == Nat.add(u, f) : Nat}

def fs_upload_conserved(u, f, n, c, s):
  fs_go_conserved(u, f, n, s, Nat.is_le(s, c))

# promise: an upload larger than the per-file cap is never stored.
# harm: one client could fill the disk with a single file past the configured limit
# catches: the per-file cap test dropped (only the free bytes are compared)
law fs_upload_within_cap:
  for +l: FsLedger
  for +c: Nat
  for +x: Nat
  {fs_is_stored(fs_upload(l, c, Nat.add(c, 1n+x))) == False{} : Bool}

def fs_upload_within_cap(l, c, x):
  match l:
    case FsLedger{+u, +f, +n}:
      %Equal.sym(Bool, Nat.is_le(Nat.add(c, 1n+x), c), False{}, G.g_nat_le_add_succ_false(c, x)) : {fs_is_stored(fs_upload_go(u, f, n, Nat.add(c, 1n+x), _)) == False{} : Bool}
      {==}

# promise: an upload larger than the free bytes of the quota is never stored, whatever the cap.
# harm: uploads could pass the quota and fill the disk the store was given a share of
# catches: the size compared with the whole quota (used + free) instead of the free bytes
law fs_upload_within_free:
  for +u: Nat
  for +f: Nat
  for +n: Nat
  for +c: Nat
  for +x: Nat
  {fs_is_stored(fs_upload(FsLedger{u, f, n}, c, Nat.add(f, 1n+x))) == False{} : Bool}

def fs_upload_within_free(u, f, n, c, x):
  fs_go_over_free(u, f, n, x, Nat.is_le(Nat.add(f, 1n+x), c))

# promise: an upload is refused as too large only when it is larger than the per-file cap, and as over the quota only when it is larger than the free bytes.
# harm: the store could refuse files it has room for, or give the client the wrong reason
# catches: the free-bytes test made strict, refusing an upload that fills the quota exactly
# catches: the too-large and over-quota refusals swapped
law fs_upload_fits_stored:
  for +l: FsLedger
  for +c: Nat
  for +s: Nat
  {fs_refused_rightly(s, c, fs_free(l), fs_upload(l, c, s)) == True{} : Bool}

def fs_upload_fits_stored(l, c, s):
  match l:
    case FsLedger{+u, +f, +n}:
      fs_go_rightly(u, f, n, s, c, f, Nat.is_le(s, c), {==}, L.nat_le_refl(f))

# promise: a stored upload's id is at or above the ledger's counter and below the counter of the ledger the store keeps after it, so no id is handed out twice.
# harm: two files could share an id, and a client could read or delete another client's file
# catches: fs_upload_ledger returning the old ledger for a stored upload, so the counter never advances
law fs_upload_fresh_id:
  for +l: FsLedger
  for +c: Nat
  for +s: Nat
  {fs_id_fresh(fs_next(l), fs_upload(l, c, s), fs_next(fs_upload_ledger(l, fs_upload(l, c, s)))) == True{} : Bool}

def fs_upload_fresh_id(l, c, s):
  match l:
    case FsLedger{+u, +f, +n}:
      fs_go_fresh(u, f, n, s, Nat.is_le(s, c))

# promise: a stored upload counts exactly its bytes as used in the ledger the store keeps after it, and a refused one counts none.
# harm: the store could hold files past its quota, or lose room it never handed out
# catches: fs_upload_ledger returning the old ledger for a stored upload, so stored bytes are never counted
law fs_upload_takes_bytes:
  for +l: FsLedger
  for +c: Nat
  for +s: Nat
  {fs_used(fs_upload_ledger(l, fs_upload(l, c, s))) == Nat.add(fs_used(l), fs_taken(fs_upload(l, c, s), s)) : Nat}

def fs_upload_takes_bytes(l, c, s):
  match l:
    case FsLedger{+u, +f, +n}:
      fs_go_takes(u, f, n, s, Nat.is_le(s, c))

# promise: a delete leaves used + free equal to the quota.
# catches: a delete that gives the bytes back to free without taking them from used
law fs_delete_conserved:
  for +l: FsLedger
  for +e: FsEntry
  {fs_quota(fs_delete(l, e)) == fs_quota(l) : Nat}

def fs_delete_conserved(l, e):
  match e:
    case FsEntry{+s, live}:
      fs_delete_go_conserved(l, s, live)

# promise: deleting a file a second time changes nothing: its bytes are freed once.
# harm: a repeated delete could free bytes another file still uses and let the store pass its quota
# catches: the deleted entry left live, so a second delete frees its bytes again
law fs_delete_once:
  for +l: FsLedger
  for +e: FsEntry
  {fs_delete(fs_delete(l, e), fs_kill(e)) == fs_delete(l, e) : FsLedger}

def fs_delete_once(l, e):
  match e:
    case FsEntry{+s, live}:
      {==}

# promise: a deleted entry is no longer live, so the file is not served and not freed again.
# catches: liveness read as always true, so a deleted file is still served
law fs_kill_retires:
  for +e: FsEntry
  {fs_live(fs_kill(e)) == False{} : Bool}

def fs_kill_retires(e):
  match e:
    case FsEntry{+s, +l}:
      {==}

# promise: deleting a live file the ledger counts frees exactly its bytes.
# harm: deleted files would keep holding the quota until the store refuses every upload
# catches: a delete that frees nothing
law fs_delete_frees:
  for +u: Nat
  for +s: Nat
  for +f: Nat
  for +n: Nat
  {fs_used(fs_delete(FsLedger{Nat.add(u, s), f, n}, FsEntry{s, True{}})) == u : Nat}

def fs_delete_frees(u, s, f, n):
  %Equal.sym(Bool, Nat.is_le(s, Nat.add(u, s)), True{}, L.nat_add_le_l(u, s)) : {fs_used(fs_release_if(Nat.add(u, s), f, n, s, _)) == u : Nat}
  L.nat_add_sub_cancel_r(u, s)

# promise: a delete never moves the id counter, so a deleted file's id is not handed out again.
# catches: a delete that resets the counter
law fs_delete_keeps_next:
  for +l: FsLedger
  for +e: FsEntry
  {fs_next(fs_delete(l, e)) == fs_next(l) : Nat}

def fs_delete_keeps_next(l, e):
  match e:
    case FsEntry{+s, live}:
      fs_delete_go_next(l, s, live)

# promise: a file found on disk at start counts its bytes as used.
# harm: after a restart the store could take uploads past its quota
# catches: the restored file's bytes not counted
law fs_restore_counts:
  for +u: Nat
  for +f: Nat
  for +n: Nat
  for +i: Nat
  for +s: Nat
  {fs_used(fs_restore(FsLedger{u, f, n}, i, s)) == Nat.add(u, s) : Nat}

def fs_restore_counts(u, f, n, i, s):
  {==}

# promise: after counting a file found on disk, the id counter is past that file's id.
# harm: a new upload after a restart could get the id of a file still on disk
# catches: the counter moved to the file's id instead of past it
law fs_restore_passes_id:
  for +l: FsLedger
  for +i: Nat
  for +s: Nat
  {Nat.is_lt(i, fs_next(fs_restore(l, i, s))) == True{} : Bool}

def fs_restore_passes_id(l, i, s):
  match l:
    case FsLedger{+u, +f, +n}:
      fs_lt_of_succ_le(i, Nat.max(n, 1n+i), L.nat_max_ge_r(n, 1n+i))

# promise: counting a file found on disk never moves the id counter back, so the ids of deleted files stay retired.
# harm: after a restart a new upload could get the id of a deleted file a client still refers to
# catches: the counter replaced by the restored id + 1 (the saved counter ignored)
law fs_restore_keeps_next:
  for +u: Nat
  for +f: Nat
  for +n: Nat
  for +i: Nat
  for +s: Nat
  {Nat.is_le(n, fs_next(fs_restore(FsLedger{u, f, n}, i, s))) == True{} : Bool}

def fs_restore_keeps_next(u, f, n, i, s):
  L.nat_max_ge_l(n, 1n+i)

# promise: counting a file found on disk that fits the free bytes takes its bytes from them: used + free stays the quota.
# harm: the files kept across a restart would leave the whole quota free for new uploads
# catches: fs_restore leaving the free bytes unchanged
law fs_restore_conserved:
  for +l: FsLedger
  for +i: Nat
  for +x: Nat
  {fs_quota(fs_restore(l, i, Nat.sub(fs_free(l), x))) == fs_quota(l) : Nat}

def fs_restore_conserved(l, i, x):
  match l:
    case FsLedger{+u, +f, +n}:
      fs_take_conserved(u, f, Nat.sub(f, x), G.g_nat_sub_le_self(f, x))

# promise: counting a file found on disk never adds free bytes.
# harm: after a restart the store could take uploads past its quota
# catches: fs_restore adding the restored size to the free bytes
law fs_restore_no_free_gain:
  for +l: FsLedger
  for +i: Nat
  for +s: Nat
  {Nat.is_le(fs_free(fs_restore(l, i, s)), fs_free(l)) == True{} : Bool}

def fs_restore_no_free_gain(l, i, s):
  match l:
    case FsLedger{+u, +f, +n}:
      G.g_nat_sub_le_self(f, s)

# promise: an open store keeps the id counter it was given, so ids handed out before a restart stay retired.
# harm: after a restart a new upload could get the id of a deleted file a client still refers to
# catches: fs_open dropping the saved counter (starting it at zero)
law fs_open_keeps_next:
  for +q: Nat
  for +n: Nat
  {Nat.is_le(n, fs_next(fs_open(q, n))) == True{} : Bool}

def fs_open_keeps_next(q, n):
  L.nat_le_refl(n)

# promise: a store whose files on disk hold more than its quota never starts.
# harm: a quota lowered below the files kept would not be enforced: deletes would free bytes past it
# catches: fs_start ready without comparing the used bytes with the quota
law fs_start_refuses_overfull:
  for +l: FsLedger
  for +q: Nat
  {Bool.and(fs_is_ready(fs_start(l, q)), Nat.is_lt(q, fs_used(l))) == False{} : Bool}

def fs_start_refuses_overfull(l, q):
  match l:
    case FsLedger{+u, +f, +n}:
      fs_start_go_fits(u, n, q, Nat.is_le(u, q), {==})

# promise: a store whose files fit its quota starts, with used + free equal to the configured quota.
# harm: a store started from an over-full ledger keeps a quota larger than configured
# catches: the ready ledger with the whole quota free (the used bytes not subtracted)
law fs_start_quota:
  for +l: FsLedger
  for +x: Nat
  {fs_quota(fs_start_ledger(fs_start(l, Nat.add(fs_used(l), x)))) == Nat.add(fs_used(l), x) : Nat}

def fs_start_quota(l, x):
  match l:
    case FsLedger{+u, +f, +n}:
      %Equal.sym(Bool, Nat.is_le(u, Nat.add(u, x)), True{}, L.nat_add_le_r(u, x)) : {fs_quota(fs_start_ledger(fs_start_go(u, n, Nat.add(u, x), _))) == Nat.add(u, x) : Nat}
      %Equal.sym(Nat, Nat.sub(Nat.add(u, x), u), x, L.nat_add_sub_cancel_l(u, x)) : {Nat.add(u, _) == Nat.add(u, x) : Nat}
      {==}

# promise: a started store counts every byte of the files found on disk as used.
# harm: files kept across a restart would not count against the quota
# catches: the ready ledger starting with no bytes used and the whole quota free
law fs_start_keeps_used:
  for +l: FsLedger
  for +x: Nat
  {fs_used(fs_start_ledger(fs_start(l, Nat.add(fs_used(l), x)))) == fs_used(l) : Nat}

def fs_start_keeps_used(l, x):
  match l:
    case FsLedger{+u, +f, +n}:
      %Equal.sym(Bool, Nat.is_le(u, Nat.add(u, x)), True{}, L.nat_add_le_r(u, x)) : {fs_used(fs_start_ledger(fs_start_go(u, n, Nat.add(u, x), _))) == u : Nat}
      {==}

# promise: a started store never moves the id counter back.
# harm: after a restart a new upload could get the id of a file deleted before it
# catches: the ready ledger starting the counter at zero
law fs_start_keeps_next:
  for +l: FsLedger
  for +x: Nat
  {Nat.is_le(fs_next(l), fs_next(fs_start_ledger(fs_start(l, Nat.add(fs_used(l), x))))) == True{} : Bool}

def fs_start_keeps_next(l, x):
  match l:
    case FsLedger{+u, +f, +n}:
      %Equal.sym(Bool, Nat.is_le(u, Nat.add(u, x)), True{}, L.nat_add_le_r(u, x)) : {Nat.is_le(n, fs_next(fs_start_ledger(fs_start_go(u, n, Nat.add(u, x), _)))) == True{} : Bool}
      L.nat_le_refl(n)

# promise: a reservation leaves used + free equal to the quota: its bytes move from free to used.
# harm: uploads in progress could stage more bytes on disk than the quota has room for
# catches: the reserved bytes added to used without being taken from free
law fs_hold_conserved:
  for +l: FsLedger
  for +c: Nat
  for +w: Nat
  {fs_quota(fs_hold_ledger(fs_hold(l, c, w))) == fs_quota(l) : Nat}

def fs_hold_conserved(l, c, w):
  match l:
    case FsLedger{+u, +f, +n}:
      fs_take_conserved(u, f, Nat.min(Nat.min(w, c), f), L.nat_min_le_r(Nat.min(w, c), f))

# promise: a reservation never takes more than the free bytes.
# harm: concurrent uploads could together stage more than the quota
# catches: the reservation not clamped to the free bytes
law fs_hold_within_free:
  for +l: FsLedger
  for +c: Nat
  for +w: Nat
  {Nat.is_le(fs_size(fs_hold_ticket(fs_hold(l, c, w))), fs_free(l)) == True{} : Bool}

def fs_hold_within_free(l, c, w):
  match l:
    case FsLedger{+u, +f, +n}:
      L.nat_min_le_r(Nat.min(w, c), f)

# promise: an upload that asks for no more than the per-file cap and the free bytes is reserved all it asked for.
# harm: uploads that fit the cap and the quota could be refused as over the quota
# catches: fs_hold reserving 0 bytes
# catches: fs_hold reserving half of what it may
law fs_hold_covers_ask:
  for +u: Nat
  for +w: Nat
  for +x: Nat
  for +y: Nat
  for +n: Nat
  {Nat.is_le(w, fs_size(fs_hold_ticket(fs_hold(FsLedger{u, Nat.add(w, x), n}, Nat.add(w, y), w)))) == True{} : Bool}

def fs_hold_covers_ask(u, w, x, y, n):
  %Equal.sym(Nat, Nat.min(w, Nat.add(w, y)), w, fs_min_add_self(w, y)) : {Nat.is_le(w, Nat.min(_, Nat.add(w, x))) == True{} : Bool}
  %Equal.sym(Nat, Nat.min(w, Nat.add(w, x)), w, fs_min_add_self(w, x)) : {Nat.is_le(w, _) == True{} : Bool}
  L.nat_le_refl(w)

# promise: a reservation never takes more than the upload asked for, nor more than the per-file cap.
# harm: one upload could hold the room other uploads need
# catches: the reservation taking every free byte whatever the upload asked for
law fs_hold_within_ask:
  for +l: FsLedger
  for +c: Nat
  for +w: Nat
  {Nat.is_le(fs_size(fs_hold_ticket(fs_hold(l, c, w))), Nat.min(w, c)) == True{} : Bool}

def fs_hold_within_ask(l, c, w):
  match l:
    case FsLedger{+u, +f, +n}:
      L.nat_min_le_l(Nat.min(w, c), f)

# promise: giving a reservation's ticket back returns the ledger to what it was before the reservation.
# harm: a dropped upload could keep its reserved bytes, or free more than it held
# catches: the ticket handed out already given back (not live), so the reservation is never released
law fs_hold_released:
  for +l: FsLedger
  for +c: Nat
  for +w: Nat
  {fs_delete(fs_hold_ledger(fs_hold(l, c, w)), fs_hold_ticket(fs_hold(l, c, w))) == l : FsLedger}

def fs_hold_released(l, c, w):
  match l:
    case FsLedger{+u, +f, +n}:
      fs_hold_back(u, f, n, Nat.min(Nat.min(w, c), f), L.nat_min_le_r(Nat.min(w, c), f))

# promise: giving back an entry already given back changes nothing, whatever the ledger is by then, so a reservation or a file is released once.
# harm: a ticket given back at commit and again when the upload is dropped would free bytes twice
# catches: fs_kill leaving the entry live
law fs_delete_killed:
  for +l: FsLedger
  for +e: FsEntry
  {fs_delete(l, fs_kill(e)) == l : FsLedger}

def fs_delete_killed(l, e):
  match e:
    case FsEntry{+s, +v}:
      {==}

# promise: a commit that stores an upload stores exactly what fs_upload decides for it, so the cap, quota, id and byte laws of fs_upload hold for every stored commit.
# harm: a commit could store a file the ledger never admitted, or count other bytes than the file's
# catches: the commit deciding with the reservation's size instead of the staged size
law fs_commit_is_upload:
  for +l: FsLedger
  for +c: Nat
  for +m: Nat
  for +t: FsEntry
  for +s: Nat
  {fs_when_stored(fs_commit(l, c, m, t, s), fs_upload(l, c, s)) == fs_commit(l, c, m, t, s) : FsUpload}

def fs_commit_is_upload(l, c, m, t, s):
  match t:
    case FsEntry{+h, +v}:
      fs_commit_go_is(l, c, m, h, s, Nat.is_le(s, h))

# promise: a refused commit, whatever the refusal, leaves the ledger the store keeps after it equal to the ledger it was decided against.
# harm: a refusal could zero the used bytes and the id counter: the quota would be lost and ids handed out again
# catches: fs_upload_ledger returning a zeroed ledger for FsNoIds
law fs_commit_refused_keeps:
  for +l: FsLedger
  for +c: Nat
  for +m: Nat
  for +t: FsEntry
  for +s: Nat
  {fs_kept_if_refused(fs_commit(l, c, m, t, s), fs_upload_ledger(l, fs_commit(l, c, m, t, s)), l) == fs_upload_ledger(l, fs_commit(l, c, m, t, s)) : FsLedger}

def fs_commit_refused_keeps(l, c, m, t, s):
  fs_ledger_kept(l, fs_commit(l, c, m, t, s))

# promise: an upload larger than its reservation is never stored: its bytes past the reservation were not read.
# harm: a truncated file could be stored as if complete
# catches: the reservation test dropped
law fs_commit_within_hold:
  for +l: FsLedger
  for +c: Nat
  for +m: Nat
  for +t: FsEntry
  for +x: Nat
  {fs_is_stored(fs_commit(l, c, m, t, Nat.add(fs_size(t), 1n+x))) == False{} : Bool}

def fs_commit_within_hold(l, c, m, t, x):
  match t:
    case FsEntry{+h, +v}:
      %Equal.sym(Bool, Nat.is_le(Nat.add(h, 1n+x), h), False{}, G.g_nat_le_add_succ_false(h, x)) : {fs_is_stored(fs_commit_go(l, c, m, h, Nat.add(h, 1n+x), _)) == False{} : Bool}
      fs_past_unstored(h, c, Nat.add(h, 1n+x), Nat.is_le(Nat.add(h, 1n+x), c))

# promise: a commit never hands out an id at or past the id limit, and refuses for want of ids only once the counter has reached it.
# harm: ids past the limit would not fit their 12 hex digits and could repeat
# catches: the id-limit test dropped
law fs_commit_ids_rightly:
  for +l: FsLedger
  for +c: Nat
  for +m: Nat
  for +t: FsEntry
  for +s: Nat
  {fs_ids_rightly(fs_next(l), m, fs_commit(l, c, m, t, s)) == True{} : Bool}

def fs_commit_ids_rightly(l, c, m, t, s):
  match l:
    case FsLedger{+u, +f, +n}:
      match t:
        case FsEntry{+h, +v}:
          fs_cgo_ids(u, f, n, c, m, h, s, Nat.is_le(s, h))

# promise: a commit is refused as too large only when the upload is larger than the per-file cap, and as over the quota only when it is larger than its reservation or the free bytes.
# harm: the store could refuse uploads it has room for, or give the client the wrong reason
# catches: the too-large and over-quota refusals swapped for an upload past its reservation
law fs_commit_refused_rightly:
  for +l: FsLedger
  for +c: Nat
  for +m: Nat
  for +t: FsEntry
  for +s: Nat
  {fs_refused_rightly(s, c, Nat.min(fs_free(l), fs_size(t)), fs_commit(l, c, m, t, s)) == True{} : Bool}

def fs_commit_refused_rightly(l, c, m, t, s):
  match l:
    case FsLedger{+u, +f, +n}:
      match t:
        case FsEntry{+h, +v}:
          fs_cgo_rightly(u, f, n, c, m, h, s, Nat.is_le(s, h), {==})

# promise: an upload that fills its reservation exactly, within the cap and the free bytes, with ids left, is stored.
# harm: the store could refuse every upload
# catches: the reservation test made strict, refusing an upload that fills its reservation
law fs_commit_fits_stored:
  for +g: Nat
  for +x: Nat
  for +y: Nat
  for +n: Nat
  for +z: Nat
  {fs_is_stored(fs_commit(fs_open(Nat.add(g, x), n), Nat.add(g, y), Nat.add(n, 1n+z), FsEntry{g, True{}}, g)) == True{} : Bool}

def fs_commit_fits_stored(g, x, y, n, z):
  %Equal.sym(Bool, Nat.is_le(g, g), True{}, L.nat_le_refl(g)) : {fs_is_stored(fs_commit_go(FsLedger{0n, Nat.add(g, x), n}, Nat.add(g, y), Nat.add(n, 1n+z), g, g, _)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(n, Nat.add(n, 1n+z)), True{}, fs_lt_add_succ(n, z)) : {fs_is_stored(fs_commit_ids(FsLedger{0n, Nat.add(g, x), n}, Nat.add(g, y), g, _)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_le(g, Nat.add(g, y)), True{}, L.nat_add_le_r(g, y)) : {fs_is_stored(fs_upload_go(0n, Nat.add(g, x), n, g, _)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_le(g, Nat.add(g, x)), True{}, L.nat_add_le_r(g, x)) : {fs_is_stored(fs_store_if(0n, Nat.add(g, x), n, g, _)) == True{} : Bool}
  {==}

# promise: a file is never shown to a caller who is not its owner.
# harm: one API key could read or delete another key's files
# catches: the owner test dropped
law fs_visible_owner_only:
  for +e: FsEntry
  for +i: Bool
  {fs_visible(e, False{}, i) == False{} : Bool}

def fs_visible_owner_only(e, i):
  G.g_bool_and_false_r(fs_live(e))

# promise: a deleted file is never shown.
# harm: a deleted file could still be listed or read
# catches: liveness ignored
law fs_visible_live_only:
  for +e: FsEntry
  for +o: Bool
  for +i: Bool
  {fs_visible(fs_kill(e), o, i) == False{} : Bool}

def fs_visible_live_only(e, o, i):
  match e:
    case FsEntry{+s, +v}:
      {==}

# promise: a request for one file never gets another one.
# harm: a caller naming one id could be given another file of the same owner
# catches: the id test dropped
law fs_visible_id_only:
  for +e: FsEntry
  for +o: Bool
  {fs_visible(e, o, False{}) == False{} : Bool}

def fs_visible_id_only(e, o):
  %Equal.sym(Bool, Bool.and(o, False{}), False{}, G.g_bool_and_false_r(o)) : {Bool.and(fs_live(e), _) == False{} : Bool}
  G.g_bool_and_false_r(fs_live(e))

# promise: the owner who names a live file sees it.
# harm: owners could lose access to their own files
# catches: a file never shown
law fs_visible_own_file:
  for +s: Nat
  {fs_visible(FsEntry{s, True{}}, True{}, True{}) == True{} : Bool}

def fs_visible_own_file(s):
  {==}
