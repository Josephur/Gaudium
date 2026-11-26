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
# when its replay completed, traced the first divergence and found only ties, that a method is
# opt-in only when the suite file marks it and its options select an opt-in form, that no failed
# check is outweighed by the others, and that a suite passes only when every selected row ran,
# exited cleanly and passed.

import Base
import lemmas.bend as G

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

# Whether a method is an opt-in mode that may fork from K=0 at a traced tie (AGENTS.md section
# 7.2): only when the suite file marks it so (`marked`) AND its own options select an opt-in form
# (`opt_in_form`, read off the options by the shell: a draft tree, or the DeltaNet verify on the
# MME). A default mode marked by mistake, or an opt-in form nobody marked, stays held to exact
# identity.
def vs_method_opt_in(marked: Bool, opt_in_form: Bool) -> Bool:
  Bool.and(marked, opt_in_form)

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

# A row's final verdict: its checks passed (`vs_row_ok`) and the row's process exited cleanly.
def vs_row_verdict(checks_ok: Bool, exited_ok: Bool) -> Bool:
  Bool.and(checks_ok, exited_ok)

# The suite's verdict: every selected row ran (none was cut off), at least one did, and every
# row that ran passed.
def vs_suite_ok(selected: Nat, +ran: Nat, all_passed: Bool) -> Bool:
  Bool.and(Nat.is_eq(ran, selected), Bool.and(Nat.is_gt(ran, 0n), all_passed))

# ============================================================================
# Helper lemmas (erased before Rust codegen)
# ============================================================================

def vs_and_false_r(+x: Bool) -> {Bool.and(x, False{}) == False{} : Bool}:
  match x:
    case True{}: {==}
    case False{}: {==}

# r never equals r + 1 + k
def vs_eq_short_false(+r: Nat, k: Nat) -> {Nat.is_eq(r, Nat.add(r, 1n+k)) == False{} : Bool}:
  match r:
    case 0n: {==}
    case 1n+p:
      vs_eq_short_false(p, k)

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

# LAW VS.11: a method the suite file does not mark opt-in is held to exact identity.
# promise: a method the suite file does not mark as opt-in is never exempt from token identity to K=0, whatever its options select.
# harm: a speculative method could fork from K=0 without anyone having marked it as an opt-in mode.
law vs_unmarked_method_is_exact:
  for +f: Bool
  {vs_method_opt_in(False{}, f) == False{} : Bool}

def vs_unmarked_method_is_exact(f):
  {==}

# LAW VS.12: a method whose options select no opt-in form is held to exact identity, even if marked.
# promise: a method whose own options select no opt-in form is never exempt from token identity to K=0, even when the suite file marks it opt-in.
# harm: a default mode labelled opt-in by mistake would be allowed to fork from K=0.
law vs_default_form_is_exact:
  for +m: Bool
  {vs_method_opt_in(m, False{}) == False{} : Bool}

def vs_default_form_is_exact(m):
  match m:
    case True{}: {==}
    case False{}: {==}

# LAW VS.13: a row whose process did not exit cleanly never passes.
# promise: a row whose process ended with an error or a signal never passes, whatever its reports show.
# harm: a crash after the last report, with launches still queued, would go unnoticed.
law vs_row_needs_clean_exit:
  for +c: Bool
  {vs_row_verdict(c, False{}) == False{} : Bool}

def vs_row_needs_clean_exit(c):
  match c:
    case True{}: {==}
    case False{}: {==}

# LAW VS.14: a row whose checks failed never passes, however its process ended.
# promise: a row whose identity, answer or after-request checks failed never passes, even when its process exited cleanly.
law vs_row_verdict_needs_checks:
  for +e: Bool
  {vs_row_verdict(False{}, e) == False{} : Bool}

def vs_row_verdict_needs_checks(e):
  {==}

# LAW VS.15: a suite that ran no row never passes.
# promise: a suite run that ran no row at all never passes.
# harm: an empty selection (a mistyped --only) would report success.
law vs_suite_needs_a_row:
  for +s: Nat
  for +a: Bool
  {vs_suite_ok(s, 0n, a) == False{} : Bool}

def vs_suite_needs_a_row(s, a):
  match s:
    case 0n: {==}
    case 1n+_: {==}

# LAW VS.16: a suite cut off before every selected row ran never passes.
# promise: a suite run that stopped before every selected row ran never passes, even when every row that ran passed.
# harm: an interrupted run would report success for rows it never checked.
law vs_suite_needs_every_row:
  for +r: Nat
  for +k: Nat
  for +a: Bool
  {vs_suite_ok(Nat.add(r, 1n+k), r, a) == False{} : Bool}

def vs_suite_needs_every_row(r, k, a):
  %Equal.sym(Bool, Nat.is_eq(r, Nat.add(r, 1n+k)), False{}, vs_eq_short_false(r, k)) : {Bool.and(_, Bool.and(Nat.is_gt(r, 0n), a)) == False{} : Bool}
  {==}

# LAW VS.17: a suite with a failed row never passes.
# promise: a suite run with any failed row never passes.
law vs_suite_needs_every_pass:
  for +s: Nat
  for +r: Nat
  {vs_suite_ok(s, r, False{}) == False{} : Bool}

def vs_suite_needs_every_pass(s, r):
  %Equal.sym(Bool, Bool.and(Nat.is_gt(r, 0n), False{}), False{}, vs_and_false_r(Nat.is_gt(r, 0n))) : {Bool.and(Nat.is_eq(r, s), _) == False{} : Bool}
  vs_and_false_r(Nat.is_eq(r, s))

# LAW VS.18: identical tokens always pass identity, whether opt-in or default.
# promise: an identical token stream always passes the identity check, whether the method is opt-in or default.
# catches: vs_identity_ok returns opt_in for VsSame{} instead of True{}
law vs_same_passes:
  for +opt_in: Bool
  {vs_identity_ok(VsSame{}, opt_in) == True{} : Bool}

def vs_same_passes(opt_in):
  {==}

# LAW VS.19: a properly traced BF16 tie on a properly marked opt-in method passes identity.
# promise: a fork whose replay traced every divergence as a tie passes the identity check for a marked opt-in method.
# catches: vs_fork_outcome returns VsFork{} or vs_method_opt_in returns False{} when all inputs are True{}
law vs_traced_tie_opt_in_passes:
  {vs_identity_ok(vs_fork_outcome(True{}, True{}, True{}), vs_method_opt_in(True{}, True{})) == True{} : Bool}

def vs_traced_tie_opt_in_passes():
  {==}

# LAW VS.20: a row whose identity, semantics, after-check, and exit all pass yields a passing row verdict.
# promise: a row whose identity, semantic, after-image, and process-exit checks all passed receives a passing verdict.
# catches: vs_row_verdict or vs_row_ok always returns False{} even when all checks pass
law vs_valid_row_passes:
  {vs_row_verdict(vs_row_ok(True{}, True{}, True{}), True{}) == True{} : Bool}

def vs_valid_row_passes():
  {==}

# LAW VS.21: any non-empty suite where every selected row ran and passed yields a passing suite verdict.
# promise: a non-empty suite in which every selected row ran and passed receives a passing verdict.
# catches: vs_suite_ok rejects valid runs when selected == ran >= 1
law vs_valid_suite_passes:
  for +n_m1: Nat
  {vs_suite_ok(1n+n_m1, 1n+n_m1, True{}) == True{} : Bool}

def vs_valid_suite_passes(n_m1):
  %Equal.sym(Bool, Nat.is_eq(1n+n_m1, 1n+n_m1), True{}, G.g_nat_eq_refl(1n+n_m1)) : {Bool.and(_, Bool.and(Nat.is_gt(1n+n_m1, 0n), True{})) == True{} : Bool}
  {==}
