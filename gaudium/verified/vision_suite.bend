# The verdict of the vision regression suite (`gaudium --vision-suite`): whether one model and
# speculative method passed with image input. The suite's result is what lets a drafter run
# beside the vision tower (`vis_drafter_ok` in vision_budget.bend), so its verdict states the
# exactness rule of AGENTS.md section 7:
#   - a default method must be token-identical to the engine's own K=0 stream, greedy and
#     sampled with the same seed;
#   - an opt-in method may fork from K=0 only at a traced numerical tie;
#   - a row passes only when every run's identity, every semantic answer check and the text
#     request after the images (no state left behind) passed.
# Formally specifies and proves that a fork never passes a default method, that a fork without a
# trace or a missing report never passes at all, that a diverged run passes an opt-in method only
# when its replay completed, traced the first divergence and found only ties, and that no failed
# check is outweighed by the others.

import Base

# How one run of a suite row ended against the engine's own K=0 stream (classified by the shell
# from the run's report: the token streams and, at a fork, its margin trace).
#   VsSame       identical tokens
#   VsTracedTie  forked at a position whose margin trace shows an exact tie of the BF16 logits
#                (within one unit in the last place)
#   VsFork       forked anywhere else, without a trace, or the lengths differ
#   VsMissing    no identity report (the run did not complete)
type VsIdentity is Data:
  VsSame{}
  VsTracedTie{}
  VsFork{}
  VsMissing{}

# The outcome of a run whose tokens diverged from K=0, from the facts the shell read off its
# margin trace: `replayed` (the replay of the run's whole stream on the K=0 loop completed),
# `first_traced` (the first divergence is among the traced positions) and `all_ties` (every
# traced position where K=0 would have chosen otherwise is a tie of the BF16 logits). Only with
# all three is the fork a traced tie; otherwise it is a fork.
def vs_fork_outcome(replayed: Bool, first_traced: Bool, all_ties: Bool) -> VsIdentity:
  match replayed:
    case True{}:
      match first_traced:
        case True{}:
          match all_ties:
            case True{}:
              VsTracedTie{}
            case False{}:
              VsFork{}
        case False{}:
          VsFork{}
    case False{}:
      VsFork{}

# Whether a run's identity outcome passes: identical always, a traced tie only for an opt-in
# method, anything else never.
def vs_identity_ok(o: VsIdentity, opt_in: Bool) -> Bool:
  match o:
    case VsSame{}:
      True{}
    case VsTracedTie{}:
      opt_in
    case VsFork{}:
      False{}
    case VsMissing{}:
      False{}

# A row passes when every run's identity passed, every semantic answer check passed, and the
# text request after the images gave the tokens of a fresh engine.
def vs_row_ok(identity_ok: Bool, semantic_ok: Bool, after_ok: Bool) -> Bool:
  Bool.and(identity_ok, Bool.and(semantic_ok, after_ok))

# ============================================================================
# Laws
# ============================================================================

# LAW VS.1: a default method never passes with a fork, not even at a traced tie (with VS.2 and
# VS.3: a default method passes a run only with identical tokens).
# promise: a default speculative method passes only with tokens identical to the engine's own K=0 stream.
# harm: a default mode that forks from K=0 would change what users get once images are on.
law vs_default_method_is_exact:
  {vs_identity_ok(VsTracedTie{}, False{}) == False{} : Bool}

def vs_default_method_is_exact():
  {==}

# LAW VS.2: a fork that is not a traced tie never passes, whatever the method.
# promise: a run that forked from K=0 anywhere but at a traced tie never passes the suite.
# harm: an opt-in mode could fork at real differences and still be reported as passing.
law vs_untraced_fork_fails:
  for b: Bool
  {vs_identity_ok(VsFork{}, b) == False{} : Bool}

def vs_untraced_fork_fails(b):
  {==}

# LAW VS.3: a run without an identity report never passes.
# promise: a run whose identity against K=0 was never reported never passes the suite.
# harm: a crashed or cut-off run would count as a pass.
law vs_missing_run_fails:
  for b: Bool
  {vs_identity_ok(VsMissing{}, b) == False{} : Bool}

def vs_missing_run_fails(b):
  {==}

# LAW VS.4: a row with a failed identity check never passes.
# promise: a row passes only when every run's identity against K=0 passed.
law vs_row_needs_identity:
  for s: Bool
  for a: Bool
  {vs_row_ok(False{}, s, a) == False{} : Bool}

def vs_row_needs_identity(s, a):
  {==}

# LAW VS.5: a row with a wrong answer never passes.
# promise: a row passes only when every semantic answer check passed.
law vs_row_needs_semantics:
  for i: Bool
  for a: Bool
  {vs_row_ok(i, False{}, a) == False{} : Bool}

def vs_row_needs_semantics(i, a):
  match i:
    case True{}: {==}
    case False{}: {==}

# LAW VS.6: a row whose text request after the images differs from a fresh engine never passes.
# promise: a row passes only when the text request after the image requests gave a fresh engine's tokens.
# harm: an image request that leaves state behind would go unnoticed.
law vs_row_needs_clean_after:
  for i: Bool
  for s: Bool
  {vs_row_ok(i, s, False{}) == False{} : Bool}

def vs_row_needs_clean_after(i, s):
  match i s:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} _: {==}

# LAW VS.7: a diverged run whose replay did not complete never passes.
# promise: a run that diverged from K=0 never passes the suite unless the replay of its whole stream on the K=0 loop completed.
# harm: a fork later in a cut-off replay would go untraced and the run would still pass.
law vs_fork_needs_replay:
  for +f: Bool
  for +t: Bool
  for +b: Bool
  {vs_identity_ok(vs_fork_outcome(False{}, f, t), b) == False{} : Bool}

def vs_fork_needs_replay(f, t, b):
  {==}

# LAW VS.8: a diverged run whose first divergence was not traced never passes.
# promise: a run that diverged from K=0 never passes the suite unless its first divergence is among the traced positions.
# harm: the fork that changed the output could be one no trace looked at.
law vs_fork_needs_first_trace:
  for +r: Bool
  for +t: Bool
  for +b: Bool
  {vs_identity_ok(vs_fork_outcome(r, False{}, t), b) == False{} : Bool}

def vs_fork_needs_first_trace(r, t, b):
  match r:
    case True{}: {==}
    case False{}: {==}

# LAW VS.9: a diverged run with a traced disagreement that is not a tie never passes.
# promise: a run that diverged from K=0 never passes the suite when any traced disagreement is not a tie.
# harm: an opt-in mode could fork at a real difference after a first tie and still pass.
law vs_fork_needs_all_ties:
  for +r: Bool
  for +f: Bool
  for +b: Bool
  {vs_identity_ok(vs_fork_outcome(r, f, False{}), b) == False{} : Bool}

def vs_fork_needs_all_ties(r, f, b):
  match r f:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} _: {==}

# LAW VS.10: a diverged run never passes a default method, whatever its trace shows.
# promise: a run that diverged from K=0 never passes a default speculative method, whatever its margin trace shows.
# harm: a default mode could fork from K=0 at a traced tie and still be reported as exact.
law vs_fork_fails_default:
  for +r: Bool
  for +f: Bool
  for +t: Bool
  {vs_identity_ok(vs_fork_outcome(r, f, t), False{}) == False{} : Bool}

def vs_fork_fails_default(r, f, t):
  match r f t:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} _: {==}
    case False{} _ _: {==}
