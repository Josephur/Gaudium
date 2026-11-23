# Inkling 2 image tower of a GGUF model: which projector file beside the model is used.
# The Rust shell walks the candidate files in name order and folds each through `mmp_step`
# (starting from `MmpNone{}`) with whether it matches the model (`mmp_matches`), its position
# and its preference rank (`mmp_rank`).
# Formally specifies and proves:
#   - A file that does not match the model never becomes the choice and never displaces one,
#     so no file is chosen when none matches
#   - A matching file always leaves a choice behind
#   - The choice is always the file just seen or the earlier choice (one of the candidates)
#   - The choice's rank never gets worse, and is never worse than any matching file's
#   - A file of another projector type, or whose projection width differs from the model's
#     hidden size, never matches

import Base
import nat_lemmas.bend as L
import inkling2_vision.bend as IV

# ============================================================================
# 1. Types
# ============================================================================

# The choice so far: none, or the candidate at `index` with preference rank `rank`.
type MmpPick is Data:
  MmpNone{}
  MmpAt{index: Nat, rank: Nat}

# The precision of a projector file's tower weights.
type MmpPrecision is Data:
  MmpBf16{}
  MmpF32{}
  MmpF16{}
  MmpOther{}

# ============================================================================
# 2. The rule
# ============================================================================

# Preference rank of a projector file by the precision of its tower weights, lower first: BF16
# is what the tower runs in (bytes taken as stored), F32 rounds exactly back to a BF16 source,
# F16 and block formats are converted with loss.
def mmp_rank(p: MmpPrecision) -> Nat:
  match p:
    case MmpBf16{}:
      0n
    case MmpF32{}:
      1n
    case MmpF16{}:
      2n
    case MmpOther{}:
      3n

# Whether a projector file serves the model: it is of the family's projector type (`family`)
# and its projection width `proj` is the model's hidden size `hidden`.
def mmp_matches(family: Bool, +proj: Nat, +hidden: Nat) -> Bool:
  Bool.and(family, Nat.is_eq(proj, hidden))

# A matching file `i` of rank `rank` against an earlier choice `bi` of rank `br`: the new file
# only when its rank is strictly better (a tie keeps the earlier file, the first in name order).
def mmp_better(+i: Nat, +rank: Nat, +bi: Nat, +br: Nat) -> MmpPick:
  Bool.pick(MmpPick, Nat.is_lt(rank, br), MmpAt{i, rank}, MmpAt{bi, br})

# One candidate folded into the choice: file `i`, whether it matches (`ok`), its rank.
def mmp_step(ok: Bool, best: MmpPick, +i: Nat, +rank: Nat) -> MmpPick:
  match ok:
    case False{}:
      best
    case True{}:
      match best:
        case MmpNone{}:
          MmpAt{i, rank}
        case MmpAt{bi, br}:
          mmp_better(i, rank, bi, br)

# ============================================================================
# 3. Observations of a choice (used by the laws and the shell)
# ============================================================================

# Whether a file is chosen.
def mmp_some(p: MmpPick) -> Bool:
  match p:
    case MmpNone{}:
      False{}
    case MmpAt{pi, pr}:
      True{}

# Whether a file is chosen and its rank is at most `r`.
def mmp_rank_le(p: MmpPick, +r: Nat) -> Bool:
  match p:
    case MmpNone{}:
      False{}
    case MmpAt{pi, pr}:
      Nat.is_le(pr, r)

# Whether `new` is no worse than `old`: nothing was chosen before, or the rank did not grow.
def mmp_no_worse(new: MmpPick, old: MmpPick) -> Bool:
  match old:
    case MmpNone{}:
      True{}
    case MmpAt{oi, orank}:
      mmp_rank_le(new, orank)

# Whether the chosen file is file `k`.
def mmp_has_index(p: MmpPick, +k: Nat) -> Bool:
  match p:
    case MmpNone{}:
      False{}
    case MmpAt{pi, pr}:
      Nat.is_eq(pi, k)

# Whether file `ni` is file `i` or the earlier choice `old`.
def mmp_at_from(+ni: Nat, old: MmpPick, +i: Nat) -> Bool:
  Bool.or(Nat.is_eq(ni, i), mmp_has_index(old, ni))

# Whether `new` (after seeing file `i`) is file `i` or the earlier choice `old` (none only when
# there was none).
def mmp_index_from(new: MmpPick, old: MmpPick, +i: Nat) -> Bool:
  match new:
    case MmpNone{}:
      Bool.not(mmp_some(old))
    case MmpAt{ni, nr}:
      mmp_at_from(ni, old, i)

# ============================================================================
# 4. Helper lemmas (erased before Rust codegen)
# ============================================================================

def mmp_eq_refl(n: Nat) -> {Nat.is_eq(n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      mmp_eq_refl(p)

def mmp_or_true_r(b: Bool) -> {Bool.or(b, True{}) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# The rank bound of a pick between two files is the same after both ranks and the bound grow
# by one.
def mmp_rank_le_succ(c: Bool, +i: Nat, +r: Nat, +bi: Nat, +b: Nat, +x: Nat) -> {mmp_rank_le(Bool.pick(MmpPick, c, MmpAt{i, r}, MmpAt{bi, b}), x) == mmp_rank_le(Bool.pick(MmpPick, c, MmpAt{i, 1n+r}, MmpAt{bi, 1n+b}), 1n+x) : Bool}:
  match c:
    case True{}: {==}
    case False{}: {==}

# The better of two files has a rank at most the new file's.
def mmp_better_le_new(+i: Nat, +r: Nat, +bi: Nat, +b: Nat) -> {mmp_rank_le(mmp_better(i, r, bi, b), r) == True{} : Bool}:
  match r b:
    case 0n 0n: {==}
    case 0n 1n+bp: {==}
    case 1n+rp 0n: {==}
    case 1n+rp 1n+bp:
      %mmp_rank_le_succ(Nat.is_lt(rp, bp), i, rp, bi, bp, rp) : {_ == True{} : Bool}
      mmp_better_le_new(i, rp, bi, bp)

# The better of two files has a rank at most the earlier choice's.
def mmp_better_le_old(+i: Nat, +r: Nat, +bi: Nat, +b: Nat) -> {mmp_rank_le(mmp_better(i, r, bi, b), b) == True{} : Bool}:
  match r b:
    case 0n 0n: {==}
    case 0n 1n+bp: {==}
    case 1n+rp 0n: {==}
    case 1n+rp 1n+bp:
      %mmp_rank_le_succ(Nat.is_lt(rp, bp), i, rp, bi, bp, bp) : {_ == True{} : Bool}
      mmp_better_le_old(i, rp, bi, bp)

# A pick between two files always chooses one.
def mmp_pick_some(c: Bool, +i: Nat, +r: Nat, +bi: Nat, +b: Nat) -> {mmp_some(Bool.pick(MmpPick, c, MmpAt{i, r}, MmpAt{bi, b})) == True{} : Bool}:
  match c:
    case True{}: {==}
    case False{}: {==}

# The better of two files is a choice.
def mmp_better_some(+i: Nat, +r: Nat, +bi: Nat, +b: Nat) -> {mmp_some(mmp_better(i, r, bi, b)) == True{} : Bool}:
  mmp_pick_some(Nat.is_lt(r, b), i, r, bi, b)

# An earlier choice `bi` is its own index.
def mmp_kept_from(+bi: Nat, +i: Nat) -> {Bool.or(Nat.is_eq(bi, i), Nat.is_eq(bi, bi)) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_eq(bi, bi), True{}, mmp_eq_refl(bi)) : {Bool.or(Nat.is_eq(bi, i), _) == True{} : Bool}
  mmp_or_true_r(Nat.is_eq(bi, i))

# A pick between file `i` and an earlier choice `bi` is one of the two.
def mmp_pick_from(c: Bool, +i: Nat, +r: Nat, +bi: Nat, +b: Nat) -> {mmp_index_from(Bool.pick(MmpPick, c, MmpAt{i, r}, MmpAt{bi, b}), MmpAt{bi, b}, i) == True{} : Bool}:
  match c:
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(i, i), True{}, mmp_eq_refl(i)) : {Bool.or(_, Nat.is_eq(bi, i)) == True{} : Bool}
      {==}
    case False{}:
      mmp_kept_from(bi, i)

# The better of file `i` and an earlier choice `bi` is one of the two.
def mmp_better_from(+i: Nat, +r: Nat, +bi: Nat, +b: Nat) -> {mmp_index_from(mmp_better(i, r, bi, b), MmpAt{bi, b}, i) == True{} : Bool}:
  mmp_pick_from(Nat.is_lt(r, b), i, r, bi, b)

# ============================================================================
# 5. Laws
# ============================================================================

# promise: a projector file that does not match the model leaves the choice exactly as it was, so it is never chosen and nothing is chosen when no file matches
# harm: a tower of another model (wrong width or type) would be loaded and fed to the text model
law mmp_unmatched_keeps:
  for best: MmpPick
  for +i: Nat
  for +rank: Nat
  {mmp_step(False{}, best, i, rank) == best : MmpPick}

def mmp_unmatched_keeps(best, i, rank):
  {==}

# promise: after a matching projector file has been seen, some file is always chosen
# harm: a GGUF model with a usable projector beside it would be served text only
law mmp_match_chooses:
  for best: MmpPick
  for +i: Nat
  for +rank: Nat
  {mmp_some(mmp_step(True{}, best, i, rank)) == True{} : Bool}

def mmp_match_chooses(best, i, rank):
  match best:
    case MmpNone{}:
      {==}
    case MmpAt{bi, br}:
      mmp_better_some(i, rank, bi, br)

# promise: after a matching projector file has been seen, the chosen file's rank is never worse than that file's
# harm: a lossy projector file could be preferred over an exact one beside it
law mmp_match_rank_bound:
  for best: MmpPick
  for +i: Nat
  for +rank: Nat
  {mmp_rank_le(mmp_step(True{}, best, i, rank), rank) == True{} : Bool}

def mmp_match_rank_bound(best, i, rank):
  match best:
    case MmpNone{}:
      L.nat_le_refl(rank)
    case MmpAt{bi, br}:
      mmp_better_le_new(i, rank, bi, br)

# promise: folding in another projector file never makes the chosen file's rank worse
# harm: a later, worse file in name order could displace a better earlier choice
law mmp_never_worse:
  for ok: Bool
  for best: MmpPick
  for +i: Nat
  for +rank: Nat
  {mmp_no_worse(mmp_step(ok, best, i, rank), best) == True{} : Bool}

def mmp_never_worse(ok, best, i, rank):
  match ok:
    case False{}:
      match best:
        case MmpNone{}:
          {==}
        case MmpAt{bi, br}:
          L.nat_le_refl(br)
    case True{}:
      match best:
        case MmpNone{}:
          {==}
        case MmpAt{bi, br}:
          mmp_better_le_old(i, rank, bi, br)

# promise: the chosen projector file is always the file just seen or the earlier choice, so it is always one of the candidates
# harm: an index past the candidate list would name no file or another one
law mmp_choice_is_candidate:
  for ok: Bool
  for best: MmpPick
  for +i: Nat
  for +rank: Nat
  {mmp_index_from(mmp_step(ok, best, i, rank), best, i) == True{} : Bool}

def mmp_choice_is_candidate(ok, best, i, rank):
  match ok:
    case False{}:
      match best:
        case MmpNone{}:
          {==}
        case MmpAt{bi, br}:
          mmp_kept_from(bi, i)
    case True{}:
      match best:
        case MmpNone{}:
          %Equal.sym(Bool, Nat.is_eq(i, i), True{}, mmp_eq_refl(i)) : {Bool.or(_, False{}) == True{} : Bool}
          {==}
        case MmpAt{bi, br}:
          mmp_better_from(i, rank, bi, br)

# promise: a projector file of another projector type never matches the model
# harm: a tower of another architecture would be run on this model's images
law mmp_other_type_never_matches:
  for +proj: Nat
  for +hidden: Nat
  {mmp_matches(False{}, proj, hidden) == False{} : Bool}

def mmp_other_type_never_matches(proj, hidden):
  {==}

# promise: a projector file whose projection is wider than the model's hidden size never matches
# harm: image features of the wrong width would be written into the prompt rows
law mmp_wider_never_matches:
  for family: Bool
  for +hidden: Nat
  for +e: Nat
  {mmp_matches(family, Nat.add(1n+hidden, e), hidden) == False{} : Bool}

def mmp_wider_never_matches(family, hidden, e):
  match family:
    case False{}:
      {==}
    case True{}:
      IV.ikv_over_ne(hidden, e)

# promise: a projector file whose projection is narrower than the model's hidden size never matches
# harm: image features of the wrong width would be written into the prompt rows
law mmp_narrower_never_matches:
  for family: Bool
  for +hidden: Nat
  for +e: Nat
  {mmp_matches(family, hidden, Nat.add(1n+hidden, e)) == False{} : Bool}

def mmp_narrower_never_matches(family, hidden, e):
  match family:
    case False{}:
      {==}
    case True{}:
      IV.ikv_under_ne(hidden, e)
