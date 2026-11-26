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
#   - The reason a file is refused for (`mmp_fit`) agrees with the rule: a file is reported as
#     serving the model exactly when it matches the rule, and as of another type exactly when it is
#   - What the model gets from the search (`mmp_outcome`): the chosen file when there is one,
#     text only when no candidate lies beside it, and a refusal when candidates do but the search
#     chose none

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

# Whether a candidate projector file serves the model, and if not, which of the rule's two facts
# it fails: the reason both the search and the tower check report.
type MmpFit is Data:
  MmpServes{}     # of the family's projector type, with the model's hidden size as its width
  MmpOtherType{}  # of another projector type
  MmpWidth{}      # its projection width is not the model's hidden size

# What the model gets from the search over its candidate files.
type MmpOutcome is Data:
  MmpUse{index: Nat}  # the candidate at `index` is the model's projector
  MmpTextOnly{}       # no candidate lies beside the model: it runs on text alone
  MmpNoMatch{}        # candidates lie beside it, but none serves it: refused with their reasons

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

# The fit of a candidate from the rule's two facts: of the family's projector type (`family`),
# and with the model's hidden size as its width (`same_width`). The type is read first (another
# type's width is not a projection of this family's).
def mmp_fit_of(family: Bool, same_width: Bool) -> MmpFit:
  match family:
    case False{}:
      MmpOtherType{}
    case True{}:
      match same_width:
        case False{}:
          MmpWidth{}
        case True{}:
          MmpServes{}

# Why a candidate projector file does not serve the model, or that it does, from the facts
# `mmp_matches` reads.
def mmp_fit(family: Bool, +proj: Nat, +hidden: Nat) -> MmpFit:
  mmp_fit_of(family, Nat.is_eq(proj, hidden))

# What the model gets from the search: the choice `best` after all `files` candidates were folded.
def mmp_outcome(+files: Nat, best: MmpPick) -> MmpOutcome:
  match best:
    case MmpAt{bi, br}:
      MmpUse{bi}
    case MmpNone{}:
      Bool.pick(MmpOutcome, Nat.is_eq(files, 0n), MmpTextOnly{}, MmpNoMatch{})

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

# The fit is a file that serves the model (the observers the fit laws are stated with).
def mmp_fit_serves(f: MmpFit) -> Bool:
  match f:
    case MmpServes{}:
      True{}
    case _:
      False{}

# The fit is a file refused as of another projector type.
def mmp_fit_other_type(f: MmpFit) -> Bool:
  match f:
    case MmpOtherType{}:
      True{}
    case _:
      False{}

# ============================================================================
# 4. Helper lemmas (erased before Rust codegen)
# ============================================================================

# The fit serves exactly when both facts hold.
def mmp_fit_of_serves(a: Bool, b: Bool) -> {mmp_fit_serves(mmp_fit_of(a, b)) == Bool.and(a, b) : Bool}:
  match a:
    case False{}: {==}
    case True{}:
      match b:
        case False{}: {==}
        case True{}: {==}

# The fit refuses as of another type exactly when the file is of another type.
def mmp_fit_of_other_type(a: Bool, b: Bool) -> {mmp_fit_other_type(mmp_fit_of(a, b)) == Bool.not(a) : Bool}:
  match a:
    case False{}: {==}
    case True{}:
      match b:
        case False{}: {==}
        case True{}: {==}

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

# promise: a candidate projector file is reported as serving the model exactly when it matches the selection rule (`mmp_matches`), so every file the rule refuses has a reason and a file it can choose has none
# harm: the file the search chose would be refused when its tower is read, or a file it passed over would be listed with no reason
# catches: the width left out of the reason (a projector of the family but another width reported as serving)
law mmp_fit_serves_exactly:
  for family: Bool
  for +proj: Nat
  for +hidden: Nat
  {mmp_fit_serves(mmp_fit(family, proj, hidden)) == mmp_matches(family, proj, hidden) : Bool}

def mmp_fit_serves_exactly(family, proj, hidden):
  mmp_fit_of_serves(family, Nat.is_eq(proj, hidden))

# promise: a projector file is refused as of another projector type exactly when it is of another type, whatever width it states
# harm: a tower of another architecture would be reported by a width it isn't held to, or a family projector of the wrong width as another type
# catches: the width checked before the type (a projector of another type reported as a width mismatch)
law mmp_fit_other_type_exactly:
  for family: Bool
  for +proj: Nat
  for +hidden: Nat
  {mmp_fit_other_type(mmp_fit(family, proj, hidden)) == Bool.not(family) : Bool}

def mmp_fit_other_type_exactly(family, proj, hidden):
  mmp_fit_of_other_type(family, Nat.is_eq(proj, hidden))

# promise: the file the search chose is always the model's projector
# harm: a model with a matching projector beside it would run without images, or read its tower from another file
# catches: the choice dropped (a chosen file read as text only)
law mmp_outcome_uses_choice:
  for +files: Nat
  for +i: Nat
  for +rank: Nat
  {mmp_outcome(files, MmpAt{i, rank}) == MmpUse{i} : MmpOutcome}

def mmp_outcome_uses_choice(files, i, rank):
  {==}

# promise: the search over no candidate projector files gives the model text only, never a refusal
# harm: every GGUF model of the family shipped without a projector would refuse to load
# catches: a model without a projector refused (no candidate taken as no match)
law mmp_outcome_no_files_text_only:
  {mmp_outcome(0n, MmpNone{}) == MmpTextOnly{} : MmpOutcome}

def mmp_outcome_no_files_text_only():
  {==}

# promise: a search that chose nothing from one or more candidate projector files refuses the model, never runs it on text alone and never gives it a file
# harm: a model whose projector file doesn't fit would silently lose its images, or read a tower that doesn't serve it
# catches: no match taken as text only (the candidates' refusals dropped)
law mmp_outcome_no_match_refused:
  for +files: Nat
  {mmp_outcome(1n+files, MmpNone{}) == MmpNoMatch{} : MmpOutcome}

def mmp_outcome_no_match_refused(files):
  {==}
