# Inkling 2 block drafter of a GGUF model: which drafter file beside the model is used, and
# whether a table a drafter file stores with fewer rows than the device tensor still holds every
# row the drafter reads.
# The Rust shell walks the candidate files (the model's directory, then its `dspark/` directory,
# each in name order) and folds each through `mtpf_step` (starting from `MmpNone{}`): whether
# the file is a block drafter of the family's layout, its hidden size and vocabulary beside the
# model's, its position and its preference rank (`mtpf_rank`). The fold itself is the projector
# file's first-best-rank rule (`mmp_step` of inkling2_mmproj.bend).
# Formally specifies and proves:
#   - A file that is not a drafter of the family's layout, or whose hidden size or vocabulary
#     differs from the model's, leaves the choice exactly as it was: it is never chosen, and no
#     file is chosen when no file matches
#   - A file that matches always leaves a choice behind
#   - The choice is always the file just seen or the earlier choice (one of the candidates)
#   - A stored table with fewer rows than the drafter reads, or more than the device tensor
#     holds, is never taken

import Base
import nat_lemmas.bend as L
import inkling2_vision.bend as IV
import inkling2_mmproj.bend as MP

# ============================================================================
# 1. Types
# ============================================================================

# The precision of a drafter file's matrices.
type MtpfPrecision is Data:
  MtpfBf16{}
  MtpfF32{}
  MtpfQ8{}
  MtpfOther{}

# ============================================================================
# 2. The rule
# ============================================================================

# Preference rank of a drafter file by the precision of its matrices, lower first: BF16 is what
# the drafter runs in (bytes taken as stored), F32 rounds back to a BF16 source, block formats
# are converted with loss.
def mtpf_rank(p: MtpfPrecision) -> Nat:
  match p:
    case MtpfBf16{}:
      0n
    case MtpfF32{}:
      1n
    case MtpfQ8{}:
      2n
    case MtpfOther{}:
      3n

# Whether a drafter file serves the model: it is a block drafter of the family's layout
# (`drafter`), its hidden size `hid` is the model's `model_hid` and its vocabulary `vocab` is the
# model's `model_vocab`.
def mtpf_matches(drafter: Bool, +hid: Nat, +model_hid: Nat, +vocab: Nat, +model_vocab: Nat) -> Bool:
  Bool.and(drafter, Bool.and(Nat.is_eq(hid, model_hid), Nat.is_eq(vocab, model_vocab)))

# One candidate file `i` of rank `rank` folded into the choice `best`.
def mtpf_step(drafter: Bool, +hid: Nat, +model_hid: Nat, +vocab: Nat, +model_vocab: Nat, best: MP.MmpPick, +i: Nat, +rank: Nat) -> MP.MmpPick:
  MP.mmp_step(mtpf_matches(drafter, hid, model_hid, vocab, model_vocab), best, i, rank)

# Whether a table a drafter file stores with `rows` rows serves a device tensor of `cap` rows
# of which the drafter reads the first `need`: the rows past the file's are never read.
def mtpf_rows_fit(+rows: Nat, +need: Nat, +cap: Nat) -> Bool:
  Bool.and(Nat.is_le(need, rows), Nat.is_le(rows, cap))

# ============================================================================
# 3. Helper lemmas (erased before Rust codegen)
# ============================================================================

def mtpf_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# 1 + r + e <= r never holds.
def mtpf_le_over(+r: Nat, +e: Nat) -> {Nat.is_le(Nat.add(1n+r, e), r) == False{} : Bool}:
  match r:
    case 0n: {==}
    case 1n+p:
      mtpf_le_over(p, e)

# ============================================================================
# 4. Laws
# ============================================================================

# promise: a file that is not a block drafter of the family's layout leaves the choice exactly as it was, so it is never chosen
# harm: a projector, a main model or another kind of drafter would be loaded as the block drafter's weights
law mtpf_other_layout_keeps:
  for +hid: Nat
  for +model_hid: Nat
  for +vocab: Nat
  for +model_vocab: Nat
  for best: MP.MmpPick
  for +i: Nat
  for +rank: Nat
  {mtpf_step(False{}, hid, model_hid, vocab, model_vocab, best, i, rank) == best : MP.MmpPick}

def mtpf_other_layout_keeps(hid, model_hid, vocab, model_vocab, best, i, rank):
  {==}

# promise: a drafter file wider than the model's hidden size leaves the choice exactly as it was, so it is never chosen
# harm: the drafter of another model would read the model's states with the wrong width
law mtpf_wider_keeps:
  for drafter: Bool
  for +model_hid: Nat
  for +e: Nat
  for +vocab: Nat
  for +model_vocab: Nat
  for best: MP.MmpPick
  for +i: Nat
  for +rank: Nat
  {mtpf_step(drafter, Nat.add(1n+model_hid, e), model_hid, vocab, model_vocab, best, i, rank) == best : MP.MmpPick}

def mtpf_wider_keeps(drafter, model_hid, e, vocab, model_vocab, best, i, rank):
  match drafter:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(Nat.add(1n+model_hid, e), model_hid), False{}, IV.ikv_over_ne(model_hid, e)) : {MP.mmp_step(Bool.and(_, Nat.is_eq(vocab, model_vocab)), best, i, rank) == best : MP.MmpPick}
      {==}

# promise: a drafter file narrower than the model's hidden size leaves the choice exactly as it was, so it is never chosen
# harm: the drafter of another model would read the model's states with the wrong width
law mtpf_narrower_keeps:
  for drafter: Bool
  for +hid: Nat
  for +e: Nat
  for +vocab: Nat
  for +model_vocab: Nat
  for best: MP.MmpPick
  for +i: Nat
  for +rank: Nat
  {mtpf_step(drafter, hid, Nat.add(1n+hid, e), vocab, model_vocab, best, i, rank) == best : MP.MmpPick}

def mtpf_narrower_keeps(drafter, hid, e, vocab, model_vocab, best, i, rank):
  match drafter:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(hid, Nat.add(1n+hid, e)), False{}, IV.ikv_under_ne(hid, e)) : {MP.mmp_step(Bool.and(_, Nat.is_eq(vocab, model_vocab)), best, i, rank) == best : MP.MmpPick}
      {==}

# promise: a drafter file with a larger vocabulary than the model's leaves the choice exactly as it was, so it is never chosen
# harm: the drafter of another tokenizer would propose ids that mean other text
law mtpf_more_vocab_keeps:
  for drafter: Bool
  for +hid: Nat
  for +model_hid: Nat
  for +model_vocab: Nat
  for +e: Nat
  for best: MP.MmpPick
  for +i: Nat
  for +rank: Nat
  {mtpf_step(drafter, hid, model_hid, Nat.add(1n+model_vocab, e), model_vocab, best, i, rank) == best : MP.MmpPick}

def mtpf_more_vocab_keeps(drafter, hid, model_hid, model_vocab, e, best, i, rank):
  match drafter:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(Nat.add(1n+model_vocab, e), model_vocab), False{}, IV.ikv_over_ne(model_vocab, e)) : {MP.mmp_step(Bool.and(Nat.is_eq(hid, model_hid), _), best, i, rank) == best : MP.MmpPick}
      %Equal.sym(Bool, Bool.and(Nat.is_eq(hid, model_hid), False{}), False{}, mtpf_and_false_r(Nat.is_eq(hid, model_hid))) : {MP.mmp_step(_, best, i, rank) == best : MP.MmpPick}
      {==}

# promise: a drafter file with a smaller vocabulary than the model's leaves the choice exactly as it was, so it is never chosen
# harm: the drafter of another tokenizer would propose ids that mean other text
law mtpf_less_vocab_keeps:
  for drafter: Bool
  for +hid: Nat
  for +model_hid: Nat
  for +vocab: Nat
  for +e: Nat
  for best: MP.MmpPick
  for +i: Nat
  for +rank: Nat
  {mtpf_step(drafter, hid, model_hid, vocab, Nat.add(1n+vocab, e), best, i, rank) == best : MP.MmpPick}

def mtpf_less_vocab_keeps(drafter, hid, model_hid, vocab, e, best, i, rank):
  match drafter:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(vocab, Nat.add(1n+vocab, e)), False{}, IV.ikv_under_ne(vocab, e)) : {MP.mmp_step(Bool.and(Nat.is_eq(hid, model_hid), _), best, i, rank) == best : MP.MmpPick}
      %Equal.sym(Bool, Bool.and(Nat.is_eq(hid, model_hid), False{}), False{}, mtpf_and_false_r(Nat.is_eq(hid, model_hid))) : {MP.mmp_step(_, best, i, rank) == best : MP.MmpPick}
      {==}

# promise: after a drafter file that matches the model has been seen, some file is always chosen
# harm: a GGUF model with a usable drafter beside it would refuse speculative decoding
law mtpf_match_chooses:
  for +hid: Nat
  for +vocab: Nat
  for best: MP.MmpPick
  for +i: Nat
  for +rank: Nat
  {MP.mmp_some(mtpf_step(True{}, hid, hid, vocab, vocab, best, i, rank)) == True{} : Bool}

def mtpf_match_chooses(hid, vocab, best, i, rank):
  %Equal.sym(Bool, Nat.is_eq(hid, hid), True{}, MP.mmp_eq_refl(hid)) : {MP.mmp_some(MP.mmp_step(Bool.and(_, Nat.is_eq(vocab, vocab)), best, i, rank)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_eq(vocab, vocab), True{}, MP.mmp_eq_refl(vocab)) : {MP.mmp_some(MP.mmp_step(_, best, i, rank)) == True{} : Bool}
  MP.mmp_match_chooses(best, i, rank)

# promise: the chosen drafter file is always the file just seen or the earlier choice, so it is always one of the candidates
# harm: an index past the candidate list would name no file or another one
law mtpf_choice_is_candidate:
  for drafter: Bool
  for +hid: Nat
  for +model_hid: Nat
  for +vocab: Nat
  for +model_vocab: Nat
  for best: MP.MmpPick
  for +i: Nat
  for +rank: Nat
  {MP.mmp_index_from(mtpf_step(drafter, hid, model_hid, vocab, model_vocab, best, i, rank), best, i) == True{} : Bool}

def mtpf_choice_is_candidate(drafter, hid, model_hid, vocab, model_vocab, best, i, rank):
  MP.mmp_choice_is_candidate(mtpf_matches(drafter, hid, model_hid, vocab, model_vocab), best, i, rank)

# promise: a stored table with fewer rows than the drafter reads is never taken
# harm: the recurrent head would read token rows past the end of the stored table
law mtpf_short_never_fits:
  for +rows: Nat
  for +e: Nat
  for +cap: Nat
  {mtpf_rows_fit(rows, Nat.add(1n+rows, e), cap) == False{} : Bool}

def mtpf_short_never_fits(rows, e, cap):
  %Equal.sym(Bool, Nat.is_le(Nat.add(1n+rows, e), rows), False{}, mtpf_le_over(rows, e)) : {Bool.and(_, Nat.is_le(rows, cap)) == False{} : Bool}
  {==}

# promise: a stored table with more rows than the device tensor holds is never taken
# harm: the upload would write past the device tensor into the next one
law mtpf_long_never_fits:
  for +need: Nat
  for +cap: Nat
  for +e: Nat
  {mtpf_rows_fit(Nat.add(1n+cap, e), need, cap) == False{} : Bool}

def mtpf_long_never_fits(need, cap, e):
  %Equal.sym(Bool, Nat.is_le(Nat.add(1n+cap, e), cap), False{}, mtpf_le_over(cap, e)) : {Bool.and(Nat.is_le(need, Nat.add(1n+cap, e)), _) == False{} : Bool}
  mtpf_and_false_r(Nat.is_le(need, Nat.add(1n+cap, e)))
