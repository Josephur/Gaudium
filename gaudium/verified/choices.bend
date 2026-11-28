# Chat completions with `n` choices (`n` on `POST /v1/chat/completions`).
# Formally specifies and proves:
#   - How many choices a request may ask for (at least one, at most the server's bound), and the
#     same rule for the bound itself on the command line
#   - How many generations the choices take: every sampled choice is a generation of its own, a
#     greedy request runs at least once and never more than once per choice
#   - Which generation each choice reads: one that runs, and its own when every choice has one
#   - The seed each choice runs with: choice 0 keeps the request's seed, no two choices of one
#     request share a seed, and every seed stays in the seed range (it wraps past the top)
#   - When a request with several generations is over: only after every one of them ended, so
#     its answer never goes out while one still runs in a lane; cancelling all of them is
#     idempotent and ends none of them by itself
#   - How the request ends: a generation that failed is never answered as a success
#
# Where the generations run (lanes sharing the prompt's prefill) is the lane fork rule in
# `lanes.bend`, section 16.

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. The number of choices
# ============================================================================

# May a request ask for `n` choices, with at most `max` allowed? (The command line's bound
# `--max-choices` is checked by the same rule against the largest bound the server takes.)
def ch_n_ok(+n: Nat, +max: Nat) -> Bool:
  Bool.and(Nat.is_gt(n, 0n), Nat.is_le(n, max))

# ============================================================================
# 2. Generations for the choices
# ============================================================================

# How many generations `n` choices take. A greedy request's choices are one stream (the engine
# is token-identical to its own greedy run whatever runs beside it), so it runs once and every
# choice is a copy of it; every sampled choice is a generation of its own, with a seed of its own.
def ch_runs(+n: Nat, +greedy: Bool) -> Nat:
  match greedy:
    case True{}:
      Nat.min(n, 1n)
    case False{}:
      n

# The generation choice `c` reads, of the `runs` the request's choices take: choice `c` reads
# its own when there is one, and the generations take turns among the choices past them (a greedy
# request's one generation is every choice's).
def ch_run_of(+c: Nat, +runs: Nat) -> Nat:
  Nat.mod(c, runs)

# The largest seed a request may send: seeds are 64-bit (2^64 - 1, built from literals Bend
# takes; a Rust test pins it to `u64::MAX`).
def ch_seed_top() -> Nat:
  Nat.add(Nat.mul(4294967295n, 4294967295n), Nat.mul(2n, 4294967295n))

# The seed of choice `i` when `room` is how far the request's `seed` may count up before it passes
# the top seed and `fits` says whether `i` is within it: past the top it wraps to 0. It never adds
# past the top, so the emitted Rust cannot overflow.
def ch_seed_from(+seed: Nat, +i: Nat, +room: Nat, +fits: Bool) -> Nat:
  match fits:
    case True{}:
      Nat.add(seed, i)
    case False{}:
      Nat.sub(Nat.sub(i, room), 1n)

# The seed choice `i` of a request whose seed is `seed` runs with, seeds running from 0 to `top`:
# the request's seed counted up by `i`, wrapping past `top` to 0 (`seed + i` modulo `top + 1`).
def ch_choice_seed(+seed: Nat, +i: Nat, +top: Nat) -> Nat:
  ch_seed_from(seed, i, Nat.sub(top, seed), Nat.is_le(i, Nat.sub(top, seed)))

# ============================================================================
# 3. The generations of one request
# ============================================================================

type ChGroup is Data:
  ChGroup{runs: Nat, ended: Nat, cancelled: Bool}

type ChEvent is Data:
  ChRunEnded{}
  ChCancelAll{}

# The group of a request that started `runs` generations.
def ch_new(+runs: Nat) -> ChGroup:
  ChGroup{runs, 0n, False{}}

# ChRunEnded   one generation delivered its end (an end past the group's count is ignored)
# ChCancelAll  the request was cancelled: every generation is told to stop
def ch_step(+g: ChGroup, +ev: ChEvent) -> ChGroup:
  match g:
    case ChGroup{r, e, c}:
      match ev:
        case ChRunEnded{}:
          ChGroup{r, Nat.min(1n+e, r), c}
        case ChCancelAll{}:
          ChGroup{r, e, True{}}

# The request is over: every generation it started delivered its end.
def ch_done(+g: ChGroup) -> Bool:
  match g:
    case ChGroup{r, e, c}:
      Nat.is_le(r, e)

def ch_ended(+g: ChGroup) -> Nat:
  match g:
    case ChGroup{r, e, c}:
      e

def ch_cancelled(+g: ChGroup) -> Bool:
  match g:
    case ChGroup{r, e, c}:
      c

# ============================================================================
# 4. How the request ends
# ============================================================================

type ChEnd is Data:
  ChEndOk{}
  ChEndFailed{}

# `failed`: generations that ended with an error.
def ch_end(+failed: Nat) -> ChEnd:
  match failed:
    case 0n:
      ChEndOk{}
    case 1n+f:
      ChEndFailed{}

def ch_end_ok(+e: ChEnd) -> Bool:
  match e:
    case ChEndOk{}:
      True{}
    case ChEndFailed{}:
      False{}

# Whether a request with `n` choices may carry a thinking budget. A budgeted reply whose
# reasoning reaches the budget is cut and resumed as a second generation (the prompt, the kept
# reasoning and the end-of-thinking marker); unless that resume runs for each choice on its own
# (`per_choice`), a request with more than one choice and a budget is refused. `budgeted`: the
# request's budget is applied.
def ch_think_ok(+n: Nat, +budgeted: Bool, +per_choice: Bool) -> Bool:
  Bool.not(Bool.and(Bool.and(budgeted, Nat.is_gt(n, 1n)), Bool.not(per_choice)))

# ============================================================================
# 5. Helper lemmas (erased before Rust codegen)
# ============================================================================

def ch_le_add_succ_false(+m: Nat, +k: Nat) -> {False{} == Nat.is_le(Nat.add(m, 1n+k), m) : Bool}:
  match m:
    case 0n:
      {==}
    case 1n+mp:
      ch_le_add_succ_false(mp, k)

def ch_le_min_self(+n: Nat) -> {Nat.is_le(n, Nat.min(n, n)) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      ch_le_min_self(p)

def ch_le_refl(+n: Nat) -> {Nat.is_le(n, n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      ch_le_refl(p)

# e > r, or e <= r.
def ch_gt_or_le(+e: Nat, +r: Nat) -> {Bool.or(Nat.is_gt(e, r), Nat.is_le(e, r)) == True{} : Bool}:
  match e r:
    case 0n 0n:
      {==}
    case 0n 1n+rp:
      {==}
    case 1n+ep 0n:
      {==}
    case 1n+ep 1n+rp:
      ch_gt_or_le(ep, rp)

# min(a, b) <= b (stated with True on the left, the side a rewrite puts in place).
def ch_min_le_r(+a: Nat, +b: Nat) -> {True{} == Nat.is_le(Nat.min(a, b), b) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      ch_min_le_r(ap, bp)

# d < 1 + (d + k).
def ch_lt_succ_add(+d: Nat, +k: Nat) -> {Nat.is_lt(d, 1n+Nat.add(d, k)) == True{} : Bool}:
  match d:
    case 0n:
      {==}
    case 1n+dp:
      ch_lt_succ_add(dp, k)

# min(1 + e, e + 1 + k) >= 1 + e.
def ch_min_succ_ge(+e: Nat, +k: Nat) -> {Nat.is_ge(Nat.min(1n+e, Nat.add(e, 1n+k)), 1n+e) == True{} : Bool}:
  match e:
    case 0n:
      {==}
    case 1n+ep:
      ch_min_succ_ge(ep, k)

# x - room - 1 < s when room < x <= room + s: a wrapped seed is below the request's seed.
def ch_wrap_lt(+x: Nat, +room: Nat, +s: Nat, hlt: {Nat.is_lt(room, x) == True{} : Bool}, hx: {Nat.is_le(x, Nat.add(room, s)) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(Nat.sub(x, room), 1n), s) == True{} : Bool}:
  match x room:
    case 0n 0n:
      L.bool_absurd({Nat.is_lt(Nat.sub(Nat.sub(0n, 0n), 1n), s) == True{} : Bool}, hlt)
    case 0n 1n+rp:
      L.bool_absurd({Nat.is_lt(Nat.sub(Nat.sub(0n, 1n+rp), 1n), s) == True{} : Bool}, hlt)
    case 1n+xp 0n:
      match s:
        case 0n:
          L.bool_absurd({Nat.is_lt(Nat.sub(xp, 0n), 0n) == True{} : Bool}, hx)
        case 1n+sp:
          %Equal.sym(Nat, Nat.sub(xp, 0n), xp, L.nat_sub_zero_r(xp)) : {Nat.is_lt(_, 1n+sp) == True{} : Bool}
          %Equal.sym(Bool, Nat.is_lt(xp, 1n+sp), Nat.is_le(xp, sp), L.nat_lt_succ_eq_le(xp, sp)) : {_ == True{} : Bool}
          hx
    case 1n+xp 1n+rp:
      ch_wrap_lt(xp, rp, s, hlt, hx)

# x - room - 1 is one-to-one over the x past room.
def ch_wrap_inj(+room: Nat, +x: Nat, +y: Nat, hx: {Nat.is_lt(room, x) == True{} : Bool}, hy: {Nat.is_lt(room, y) == True{} : Bool}, e: {Nat.sub(Nat.sub(x, room), 1n) == Nat.sub(Nat.sub(y, room), 1n) : Nat}) -> {x == y : Nat}:
  match room x y:
    case 0n 0n _:
      L.bool_absurd({0n == y : Nat}, hx)
    case 0n 1n+xp 0n:
      L.bool_absurd({1n+xp == 0n : Nat}, hy)
    case 0n 1n+xp 1n+yp:
      %L.nat_sub_zero_r(xp) : {1n+_ == 1n+yp : Nat}
      %L.nat_sub_zero_r(yp) : {1n+Nat.sub(xp, 0n) == 1n+_ : Nat}
      %e : {1n+Nat.sub(xp, 0n) == 1n+_ : Nat}
      {==}
    case 1n+rp 0n _:
      L.bool_absurd({0n == y : Nat}, hx)
    case 1n+rp 1n+xp 0n:
      L.bool_absurd({1n+xp == 0n : Nat}, hy)
    case 1n+rp 1n+xp 1n+yp:
      %ch_wrap_inj(rp, xp, yp, hx, hy, e) : {1n+xp == 1n+_ : Nat}
      {==}

# Two choices with the same seed are the same choice, the seed form taken as given.
def ch_seed_inj(+s: Nat, +room: Nat, +a: Nat, +b: Nat, fa: Bool, fb: Bool, efa: {fa == Nat.is_le(a, room) : Bool}, efb: {fb == Nat.is_le(b, room) : Bool}, ha: {Nat.is_le(a, Nat.add(room, s)) == True{} : Bool}, hb: {Nat.is_le(b, Nat.add(room, s)) == True{} : Bool}, e: {ch_seed_from(s, a, room, fa) == ch_seed_from(s, b, room, fb) : Nat}) -> {a == b : Nat}:
  match fa fb:
    case True{} True{}:
      L.nat_add_cancel_l(s, a, b, e)
    case False{} False{}:
      ch_wrap_inj(room, a, b, G.g_nat_not_le_imp_lt(a, room, Equal.sym(Bool, False{}, Nat.is_le(a, room), efa)), G.g_nat_not_le_imp_lt(b, room, Equal.sym(Bool, False{}, Nat.is_le(b, room), efb)), e)
    case True{} False{}:
      Empty.absurd({a == b : Nat}, L.nat_lt_not_le(Nat.sub(Nat.sub(b, room), 1n), s, ch_wrap_lt(b, room, s, G.g_nat_not_le_imp_lt(b, room, Equal.sym(Bool, False{}, Nat.is_le(b, room), efb)), hb), L.nat_le_subst_r(s, Nat.add(s, a), Nat.sub(Nat.sub(b, room), 1n), e, L.nat_add_le_r(s, a))))
    case False{} True{}:
      Empty.absurd({a == b : Nat}, L.nat_lt_not_le(Nat.sub(Nat.sub(a, room), 1n), s, ch_wrap_lt(a, room, s, G.g_nat_not_le_imp_lt(a, room, Equal.sym(Bool, False{}, Nat.is_le(a, room), efa)), ha), L.nat_le_subst_r(s, Nat.add(s, b), Nat.sub(Nat.sub(a, room), 1n), Equal.sym(Nat, Nat.sub(Nat.sub(a, room), 1n), Nat.add(s, b), e), L.nat_add_le_r(s, b))))

# A choice's seed is at most room + s, the top.
def ch_seed_within_from(+s: Nat, +room: Nat, +i: Nat, fi: Bool, efi: {fi == Nat.is_le(i, room) : Bool}, hi: {Nat.is_le(i, Nat.add(room, s)) == True{} : Bool}) -> {Nat.is_le(ch_seed_from(s, i, room, fi), Nat.add(room, s)) == True{} : Bool}:
  match fi:
    case True{}:
      %L.nat_add_comm(s, room) : {Nat.is_le(Nat.add(s, i), _) == True{} : Bool}
      L.nat_add_le_mono(s, s, i, room, L.nat_le_refl(s), Equal.sym(Bool, True{}, Nat.is_le(i, room), efi))
    case False{}:
      L.nat_le_trans(Nat.sub(Nat.sub(i, room), 1n), i, Nat.add(room, s), L.nat_le_trans(Nat.sub(Nat.sub(i, room), 1n), Nat.sub(i, room), i, G.g_nat_sub_le_self(Nat.sub(i, room), 1n), G.g_nat_sub_le_self(i, room)), hi)

# ============================================================================
# 6. Laws
# ============================================================================

# promise: a request never gets zero choices.
# catches: ch_n_ok returns Nat.is_le(n, max), letting n = 0 through as an empty choices array
law ch_n_never_zero:
  for +m: Nat
  {ch_n_ok(0n, m) == False{} : Bool}

def ch_n_never_zero(m):
  {==}

# promise: a request never gets more choices than the server's bound.
# catches: ch_n_ok returns Nat.is_gt(n, 0n), ignoring the bound
law ch_n_within_max:
  for +m: Nat
  for +k: Nat
  {ch_n_ok(Nat.add(m, 1n+k), m) == False{} : Bool}

def ch_n_within_max(m, k):
  match m:
    case 0n:
      {==}
    case 1n+mp:
      %ch_le_add_succ_false(mp, k) : {_ == False{} : Bool}
      {==}

# promise: one choice is always allowed when the bound allows any.
# catches: ch_n_ok compares Nat.is_lt(n, max), refusing n = 1 at a bound of 1
law ch_n_one_ok:
  for +m: Nat
  {ch_n_ok(1n, 1n+m) == True{} : Bool}

def ch_n_one_ok(m):
  match m:
    case 0n:
      {==}
    case 1n+mp:
      {==}

# promise: every sampled choice is a generation of its own, never a copy of another choice.
# catches: ch_runs returns Nat.min(n, 1n) for a sampled request, copying one sampled stream into every choice
law ch_runs_sampled_each:
  for +n: Nat
  {ch_runs(n, False{}) == n : Nat}

def ch_runs_sampled_each(n):
  {==}

# promise: the choices never take more generations than there are choices.
# catches: ch_runs returns Nat.max(n, 1n) for a greedy request, running one generation for zero choices
law ch_runs_within_n:
  for +n: Nat
  for +g: Bool
  {Nat.is_le(ch_runs(n, g), n) == True{} : Bool}

def ch_runs_within_n(n, g):
  match g:
    case True{}:
      L.nat_min_le_l(n, 1n)
    case False{}:
      ch_le_refl(n)

# promise: a request with at least one choice runs at least one generation.
# catches: ch_runs returns Nat.min(n, 0n) for a greedy request, answering with nothing generated
law ch_runs_some:
  for +n: Nat
  for +g: Bool
  {Nat.is_ge(ch_runs(1n+n, g), 1n) == True{} : Bool}

def ch_runs_some(n, g):
  match n g:
    case 0n True{}:
      {==}
    case 0n False{}:
      {==}
    case 1n+np True{}:
      {==}
    case 1n+np False{}:
      {==}

# promise: a request is never over while one of its generations has not delivered its end.
# catches: ch_done returns c, answering a cancelled request while its lanes still run
law ch_never_done_early:
  for +r: Nat
  for +e: Nat
  for +c: Bool
  {Bool.and(ch_done(ChGroup{r, e, c}), Nat.is_lt(e, r)) == False{} : Bool}

def ch_never_done_early(r, e, c):
  match r e:
    case 0n 0n:
      {==}
    case 0n 1n+ep:
      {==}
    case 1n+rp 0n:
      {==}
    case 1n+rp 1n+ep:
      ch_never_done_early(rp, ep, c)

# promise: the end of the last generation still running ends the request.
# catches: ch_step on ChRunEnded keeps e unchanged, so the request waits forever
law ch_last_end_done:
  for +e: Nat
  for +c: Bool
  {ch_done(ch_step(ChGroup{1n+e, e, c}, ChRunEnded{})) == True{} : Bool}

def ch_last_end_done(e, c):
  ch_le_min_self(1n+e)

# promise: no step counts more ended generations than the request started.
# catches: ch_step on ChRunEnded returns ChGroup{r, 1n+e, c}, counting an end the group never started
law ch_ended_within_runs:
  for +r: Nat
  for +e: Nat
  for +c: Bool
  for +ev: ChEvent
  {Bool.or(Nat.is_gt(e, r), Nat.is_le(ch_ended(ch_step(ChGroup{r, e, c}, ev)), r)) == True{} : Bool}

def ch_ended_within_runs(r, e, c, ev):
  match ev:
    case ChRunEnded{}:
      %ch_min_le_r(1n+e, r) : {Bool.or(Nat.is_gt(e, r), _) == True{} : Bool}
      G.g_bool_or_true_r(Nat.is_gt(e, r))
    case ChCancelAll{}:
      ch_gt_or_le(e, r)

# promise: cancelling a request twice leaves it as cancelling it once: cancelled, with the generations it started and the ends it counted unchanged.
# catches: ch_step on ChCancelAll returns ChGroup{r, e, Bool.not(c)}, so a second cancel resumes the request
law ch_cancel_idempotent:
  for +r: Nat
  for +e: Nat
  for +c: Bool
  {ch_step(ch_step(ChGroup{r, e, c}, ChCancelAll{}), ChCancelAll{}) == ChGroup{r, e, True{}} : ChGroup}

def ch_cancel_idempotent(r, e, c):
  {==}

# promise: a cancel never ends a request by itself: it still waits for the end of every generation it started.
# catches: ch_step on ChCancelAll returns ChGroup{r, r, True{}}, answering before the lanes stopped
law ch_cancel_keeps_waiting:
  for +g: ChGroup
  {ch_done(ch_step(g, ChCancelAll{})) == ch_done(g) : Bool}

def ch_cancel_keeps_waiting(g):
  match g:
    case ChGroup{r, e, c}:
      {==}

# promise: a cancel always marks the request cancelled.
# catches: ch_step on ChCancelAll returns g unchanged, so no generation is told to stop
law ch_cancel_marks:
  for +g: ChGroup
  {ch_cancelled(ch_step(g, ChCancelAll{})) == True{} : Bool}

def ch_cancel_marks(g):
  match g:
    case ChGroup{r, e, c}:
      {==}

# promise: a request in which a generation failed is never answered as a success.
# catches: ch_end returns ChEndOk{} for every count, answering the other choices as if nothing failed
law ch_failure_never_ok:
  for +f: Nat
  {ch_end_ok(ch_end(1n+f)) == False{} : Bool}

def ch_failure_never_ok(f):
  {==}

# promise: a request whose generations all ended without an error is answered as a success.
# catches: ch_end returns ChEndFailed{} for every count
law ch_no_failure_ok:
  {ch_end_ok(ch_end(0n)) == True{} : Bool}

def ch_no_failure_ok():
  {==}

# promise: a request that started a generation is not over before that generation delivered its end.
# catches: ch_new returns ChGroup{runs, runs, False{}}, answering a request before anything ran
law ch_new_not_done:
  for +r: Nat
  {ch_done(ch_new(1n+r)) == False{} : Bool}

def ch_new_not_done(r):
  {==}

# promise: the end of one generation counts as one end, never more.
# catches: ch_step on ChRunEnded returns ChGroup{r, Nat.min(2n+e, r), c}, answering while a generation still runs
law ch_end_counts_one:
  for +r: Nat
  for +e: Nat
  for +c: Bool
  {Nat.is_le(ch_ended(ch_step(ChGroup{r, e, c}, ChRunEnded{})), 1n+e) == True{} : Bool}

def ch_end_counts_one(r, e, c):
  L.nat_min_le_l(1n+e, r)

# promise: a request is not cancelled before anything cancels it.
# catches: ch_new returns ChGroup{runs, 0n, True{}}, so every generation of a request is stopped at once
law ch_new_not_cancelled:
  for +r: Nat
  {ch_cancelled(ch_new(r)) == False{} : Bool}

def ch_new_not_cancelled(r):
  {==}

# promise: every choice reads a generation the request runs.
# catches: ch_run_of returns c, so a choice past the generations reads one that never ran
law ch_run_of_within:
  for +c: Nat
  for +rp: Nat
  {Nat.is_lt(ch_run_of(c, 1n+rp), 1n+rp) == True{} : Bool}

def ch_run_of_within(c, rp):
  L.nat_mod_lt(c, rp)

# promise: a choice that has a generation of its own reads that one, so choices with a generation each never share one.
# catches: ch_run_of returns 0n, interleaving every generation's stream into choice 0's answer
law ch_run_of_own:
  for +d: Nat
  for +k: Nat
  {ch_run_of(d, 1n+Nat.add(d, k)) == d : Nat}

def ch_run_of_own(d, k):
  L.nat_mod_small(d, Nat.add(d, k), ch_lt_succ_add(d, k))

# promise: every end of a generation before the last one counts, so a request whose generations all ended is answered.
# catches: ch_step on ChRunEnded counts an end only when it is the last one (ChGroup{r, Bool.pick(Nat, Nat.is_eq(1n+e, r), r, e), c}), so a request with two generations never answers
law ch_end_counts_each:
  for +e: Nat
  for +k: Nat
  for +c: Bool
  {Nat.is_ge(ch_ended(ch_step(ChGroup{Nat.add(e, 1n+k), e, c}, ChRunEnded{})), 1n+e) == True{} : Bool}

def ch_end_counts_each(e, k, c):
  ch_min_succ_ge(e, k)

# promise: choice 0 runs with the request's own seed, so a request with one choice is the request it always was.
# catches: ch_seed_from counts from the next seed (Nat.add(seed, 1n+i)), so n = 1 no longer runs with the seed it asked for
law ch_choice_seed_first:
  for +s: Nat
  for +t: Nat
  {ch_choice_seed(s, 0n, t) == s : Nat}

def ch_choice_seed_first(s, t):
  %Equal.sym(Bool, Nat.is_le(0n, Nat.sub(t, s)), True{}, L.nat_le_zero(Nat.sub(t, s))) : {ch_seed_from(s, 0n, Nat.sub(t, s), _) == s : Nat}
  L.nat_add_zero_r(s)

# promise: every choice's seed is a seed the request could have sent: never past the largest one.
# catches: ch_choice_seed returns Nat.add(seed, i) without the wrap, so a seed near the top runs past it
law ch_choice_seed_within:
  for +s: Nat
  for +i: Nat
  for +t: Nat
  for +hs: {Nat.is_le(s, t) == True{} : Bool}
  for hi: {Nat.is_le(i, t) == True{} : Bool}
  {Nat.is_le(ch_choice_seed(s, i, t), t) == True{} : Bool}

def ch_choice_seed_within(s, i, t, hs, hi):
  %L.nat_sub_add_cancel(t, s, hs) : {Nat.is_le(ch_choice_seed(s, i, t), _) == True{} : Bool}
  ch_seed_within_from(s, Nat.sub(t, s), i, Nat.is_le(i, Nat.sub(t, s)), {==}, L.nat_le_subst_r(i, t, Nat.add(Nat.sub(t, s), s), Equal.sym(Nat, Nat.add(Nat.sub(t, s), s), t, L.nat_sub_add_cancel(t, s, hs)), hi))

# promise: two choices of one request never run with the same seed, so each sampled choice is a sample of its own.
# catches: ch_choice_seed returns seed for every choice, so every sampled choice is a copy of choice 0
# catches: ch_seed_from wraps one seed late (Nat.sub(i, room)), so the last choice index before the top gets choice 0's seed again
law ch_choice_seeds_distinct:
  for +s: Nat
  for +a: Nat
  for +b: Nat
  for +t: Nat
  for +hs: {Nat.is_le(s, t) == True{} : Bool}
  for ha: {Nat.is_le(a, t) == True{} : Bool}
  for hb: {Nat.is_le(b, t) == True{} : Bool}
  for e: {ch_choice_seed(s, a, t) == ch_choice_seed(s, b, t) : Nat}
  {a == b : Nat}

def ch_choice_seeds_distinct(s, a, b, t, hs, ha, hb, e):
  ch_seed_inj(s, Nat.sub(t, s), a, b, Nat.is_le(a, Nat.sub(t, s)), Nat.is_le(b, Nat.sub(t, s)), {==}, {==}, L.nat_le_subst_r(a, t, Nat.add(Nat.sub(t, s), s), Equal.sym(Nat, Nat.add(Nat.sub(t, s), s), t, L.nat_sub_add_cancel(t, s, hs)), ha), L.nat_le_subst_r(b, t, Nat.add(Nat.sub(t, s), s), Equal.sym(Nat, Nat.add(Nat.sub(t, s), s), t, L.nat_sub_add_cancel(t, s, hs)), hb), e)

# promise: a request with more than one choice never runs with a thinking budget unless the resume after a forced end of thinking runs for each choice.
# harm: the choices would share one think gate and one resumed generation, so every choice past the first would be cut at the wrong token or answered with another choice's reply.
# catches: the choice count ignored (ch_think_ok returns True)
law ch_think_many_only_per_choice:
  for +k: Nat
  for +c: Bool
  {Bool.and(ch_think_ok(2n+k, True{}, c), Bool.not(c)) == False{} : Bool}

def ch_think_many_only_per_choice(k, c):
  match c:
    case True{}: {==}
    case False{}: {==}

# promise: a request with one choice, or without a thinking budget, is never refused by this rule.
# catches: the rule refuses every budgeted request (Bool.not(budgeted))
law ch_think_rest_ok:
  for +n: Nat
  for +b: Bool
  for +c: Bool
  {Bool.and(ch_think_ok(1n, b, c), ch_think_ok(n, False{}, c)) == True{} : Bool}

def ch_think_rest_ok(n, b, c):
  match b:
    case True{}:
      match c:
        case True{}: {==}
        case False{}: {==}
    case False{}:
      match c:
        case True{}: {==}
        case False{}: {==}
