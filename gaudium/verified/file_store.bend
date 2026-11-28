# Slice — Verified File Store Ledger (Files API)
# Formally specifies and proves the byte and id accounting of the server's file store
# (`/v1/files`, `server/file_store.rs`):
#   - used + free stays equal to the quota across every upload, stored or refused, and every delete
#   - an upload larger than the per-file cap, or than the free bytes, is never stored; one that
#     fits both is
#   - a deleted file frees its bytes once: deleting it again changes nothing
#   - a file id is never handed out twice: every stored id is at or above the ledger's counter and
#     below the counter after it; deletes keep the counter; a restart restores it past every id on
#     disk and never below the counter it saved
# The ledger holds `used` and `free` (their sum is the quota) and `next`, the lowest id not yet
# handed out. Streaming the bytes to disk, the metadata files, owners and the HTTP shapes stay in
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

# Whether an upload was stored.
def fs_is_stored(r: FsUpload) -> Bool:
  match r:
    case FsStored{_, _}:
      True{}
    case FsTooLarge{}:
      False{}
    case FsOverQuota{_}:
      False{}

# A stored id is at or above the counter `n` it was decided against, and below the counter after.
def fs_id_fresh(+n: Nat, r: FsUpload) -> Bool:
  match r:
    case FsStored{+id, a}:
      Bool.and(Nat.is_le(n, id), Nat.is_lt(id, fs_next(a)))
    case FsTooLarge{}:
      True{}
    case FsOverQuota{_}:
      True{}

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

# Counting a file found on disk at start (`id`, `size` bytes): its bytes are used whether or not
# they fit a quota lowered since, and the counter moves past its id.
def fs_restore(l: FsLedger, +id: Nat, +size: Nat) -> FsLedger:
  match l:
    case FsLedger{+u, +f, +n}:
      FsLedger{Nat.add(u, size), Nat.sub(f, size), Nat.max(n, 1n+id)}

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

def fs_store_fresh(+u: Nat, +f: Nat, +n: Nat, +s: Nat, b: Bool) -> {fs_id_fresh(n, fs_store_if(u, f, n, s, b)) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Bool, Nat.is_le(n, n), True{}, L.nat_le_refl(n)) : {Bool.and(_, Nat.is_lt(n, 1n+n)) == True{} : Bool}
      L.nat_lt_succ_self(n)
    case False{}:
      {==}

def fs_go_fresh(+u: Nat, +f: Nat, +n: Nat, +s: Nat, b: Bool) -> {fs_id_fresh(n, fs_upload_go(u, f, n, s, b)) == True{} : Bool}:
  match b:
    case True{}:
      fs_store_fresh(u, f, n, s, Nat.is_le(s, f))
    case False{}:
      {==}

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

# promise: an upload within both the per-file cap and the free bytes is stored.
# harm: the store could refuse files it has room for
# catches: the free-bytes test made strict, refusing an upload that fills the quota exactly
law fs_upload_fits_stored:
  for +u: Nat
  for +s: Nat
  for +x: Nat
  for +y: Nat
  for +n: Nat
  {fs_is_stored(fs_upload(FsLedger{u, Nat.add(s, x), n}, Nat.add(s, y), s)) == True{} : Bool}

def fs_upload_fits_stored(u, s, x, y, n):
  %Equal.sym(Bool, Nat.is_le(s, Nat.add(s, y)), True{}, L.nat_add_le_r(s, y)) : {fs_is_stored(fs_upload_go(u, Nat.add(s, x), n, s, _)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_le(s, Nat.add(s, x)), True{}, L.nat_add_le_r(s, x)) : {fs_is_stored(fs_store_if(u, Nat.add(s, x), n, s, _)) == True{} : Bool}
  {==}

# promise: a stored upload's id is at or above the ledger's counter and below the counter after it, so no id is handed out twice.
# harm: two files could share an id, and a client could read or delete another client's file
# catches: the counter not advanced past the id it hands out
law fs_upload_fresh_id:
  for +u: Nat
  for +f: Nat
  for +n: Nat
  for +c: Nat
  for +s: Nat
  {fs_id_fresh(n, fs_upload(FsLedger{u, f, n}, c, s)) == True{} : Bool}

def fs_upload_fresh_id(u, f, n, c, s):
  fs_go_fresh(u, f, n, s, Nat.is_le(s, c))

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
