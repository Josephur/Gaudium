import Base
import nat_lemmas.bend as L

# ============================================================================
# Inkling 2: taking over the rows of the previous request
# ============================================================================
#
# A request whose prompt starts like the rows the card holds resumes its prefill behind them.
# The global layers' caches keep every position, so they hold any prefix. The sliding layers'
# K / V rings and the convolutions' history rings keep only the latest rows: a pass that wrote
# up to position `hi` has overwritten the rows more than a ring behind `hi`. A resume at `r`
# reads the `window` rows (sliding attention) and the `taps` rows (convolutions) before `r`, so
# it is exact only while `hi - r` is within the slack the rings were sized with.
#
# Two states can be resumed: the rings as they are now ("live"), and a copy of them taken at
# the end of the last prompt ("snapshot"), which is unharmed by a long generation.

type IkResume is Data:
  IrFresh{}
  IrLive{}
  IrSnap{}

# Rows that may have been written past a resume point while the rows read from before it are
# still in the rings.
def ikr_slack(ring: Nat, window: Nat, hist: Nat, taps: Nat) -> Nat:
  Nat.min(Nat.sub(ring, window), Nat.sub(hist, taps))

# Whether rings written up to position `hi` (exclusive) still hold what a pass starting at `r`
# reads from before `r`.
def ikr_holds(hi: Nat, r: Nat, slack: Nat) -> Bool:
  Nat.is_le(Nat.sub(hi, r), slack)

# A resume point that saves rows and is exact.
def ikr_usable(+r: Nat, hi: Nat, slack: Nat) -> Bool:
  Bool.and(Nat.is_gt(r, 0n), ikr_holds(hi, r, slack))

def ikr_pick_if(live: Bool, snap: Bool) -> IkResume:
  match live:
    case True{}:
      IrLive{}
    case False{}:
      match snap:
        case True{}:
          IrSnap{}
        case False{}:
          IrFresh{}

# Rows of the prompt the snapshot can give: the snapshot's rows are in the global caches only as
# far as they equal the rows on the card, which `r_live` bounds.
def ikr_snap_rows(r_snap: Nat, r_live: Nat) -> Nat:
  Nat.min(r_snap, r_live)

# The state a request resumes: the live rings when they hold what a resume at `r_live` reads
# (no copy), else the snapshot when it holds what a resume at its rows reads, else none (a
# prefill from position 0). `r_live` / `r_snap`: rows of the prompt equal to the rows on the
# card / to the snapshot's rows; `hi_live` / `hi_snap`: how far each state was written.
def ikr_pick(+r_live: Nat, hi_live: Nat, r_snap: Nat, hi_snap: Nat, +slack: Nat) -> IkResume:
  ikr_pick_if(
    ikr_usable(r_live, hi_live, slack),
    ikr_usable(ikr_snap_rows(r_snap, r_live), hi_snap, slack)
  )

# The position the prefill starts at.
def ikr_start(p: IkResume, +r_live: Nat, +r_snap: Nat) -> Nat:
  match p:
    case IrFresh{}:
      0n
    case IrLive{}:
      r_live
    case IrSnap{}:
      ikr_snap_rows(r_snap, r_live)

# Rows of a prompt of `rows` rows that may be taken over when `common` of them equal the rows
# held: the last row always runs, its output is the request's first token.
def ikr_cap(common: Nat, rows: Nat) -> Nat:
  Nat.min(common, Nat.sub(rows, 1n))

# Which state a pick names (the laws below speak of the state a request resumes).
def ikr_is_live(p: IkResume) -> Bool:
  match p:
    case IrFresh{}:
      False{}
    case IrLive{}:
      True{}
    case IrSnap{}:
      False{}

def ikr_is_snap(p: IkResume) -> Bool:
  match p:
    case IrFresh{}:
      False{}
    case IrLive{}:
      False{}
    case IrSnap{}:
      True{}

# ============================================================================
# Laws
# ============================================================================

# Proof helpers (erased): the pick over the usable flags, one flag at a time.
def ikr_live_if(+g: Bool, +k: Bool, +v: Bool) -> {Bool.or(Bool.not(ikr_is_live(ikr_pick_if(Bool.and(g, k), v))), k) == True{} : Bool}:
  match g k v:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

def ikr_snap_if(+a: Bool, +g: Bool, +k: Bool) -> {Bool.or(Bool.not(ikr_is_snap(ikr_pick_if(a, Bool.and(g, k)))), k) == True{} : Bool}:
  match a g k:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# promise: a resume of the live rings starts only where they still hold every row the pass reads from before its start
# harm: a resumed prefill reads overwritten ring rows and its tokens differ from a prefill from position 0
law ikr_live_holds:
  for +r_live: Nat
  for +hi_live: Nat
  for +r_snap: Nat
  for +hi_snap: Nat
  for +slack: Nat
  {Bool.or(Bool.not(ikr_is_live(ikr_pick(r_live, hi_live, r_snap, hi_snap, slack))), ikr_holds(hi_live, ikr_start(IrLive{}, r_live, r_snap), slack)) == True{} : Bool}

def ikr_live_holds(r_live, hi_live, r_snap, hi_snap, slack):
  ikr_live_if(Nat.is_gt(r_live, 0n), ikr_holds(hi_live, r_live, slack), ikr_usable(ikr_snap_rows(r_snap, r_live), hi_snap, slack))

# promise: a resume of the ring snapshot starts only where the snapshot still holds every row the pass reads from before its start
# harm: a resumed prefill reads overwritten snapshot rows and its tokens differ from a prefill from position 0
law ikr_snap_holds:
  for +r_live: Nat
  for +hi_live: Nat
  for +r_snap: Nat
  for +hi_snap: Nat
  for +slack: Nat
  {Bool.or(Bool.not(ikr_is_snap(ikr_pick(r_live, hi_live, r_snap, hi_snap, slack))), ikr_holds(hi_snap, ikr_start(IrSnap{}, r_live, r_snap), slack)) == True{} : Bool}

def ikr_snap_holds(r_live, hi_live, r_snap, hi_snap, slack):
  ikr_snap_if(ikr_usable(r_live, hi_live, slack), Nat.is_gt(ikr_snap_rows(r_snap, r_live), 0n), ikr_holds(hi_snap, ikr_snap_rows(r_snap, r_live), slack))

def ikr_or_not_self(+b: Bool) -> {Bool.or(Bool.not(b), b) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def ikr_or_r(+b: Bool, c: Bool, h: {c == True{} : Bool}) -> {Bool.or(b, c) == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      h

# promise: rings that hold what a resume at `r` reads were written no further than `r` plus the slack
# harm: a resume is taken over rows a ring has already overwritten
law ikr_holds_within:
  for +hi: Nat
  for +r: Nat
  for +slack: Nat
  {Bool.or(Bool.not(ikr_holds(hi, r, slack)), Nat.is_le(hi, Nat.add(r, slack))) == True{} : Bool}

def ikr_holds_within(hi, r, slack):
  match hi r:
    case 0n _:
      ikr_or_r(Bool.not(ikr_holds(0n, r, slack)), Nat.is_le(0n, Nat.add(r, slack)), L.nat_le_zero(Nat.add(r, slack)))
    case 1n+h 0n:
      ikr_or_not_self(Nat.is_le(1n+h, slack))
    case 1n+h 1n+q:
      ikr_holds_within(h, q, slack)

# promise: rings written exactly up to a resume point still hold everything a resume there reads
# harm: a state that is intact is never resumed and every request prefills from position 0
law ikr_holds_at:
  for +r: Nat
  for +slack: Nat
  {ikr_holds(r, r, slack) == True{} : Bool}

def ikr_holds_at(r, slack):
  match r:
    case 0n:
      L.nat_le_zero(slack)
    case 1n+p:
      ikr_holds_at(p, slack)

# promise: a resume of the live rings never starts past the rows of the prompt that equal the rows on the card
# harm: a resume skips rows the global caches do not hold for this prompt
law ikr_live_start_le_live:
  for +r_live: Nat
  for +r_snap: Nat
  {Nat.is_le(ikr_start(IrLive{}, r_live, r_snap), r_live) == True{} : Bool}

def ikr_live_start_le_live(r_live, r_snap):
  L.nat_le_refl(r_live)

# promise: a resume of the ring snapshot never starts past the rows of the prompt that equal the rows on the card
# harm: a resume skips rows the global caches do not hold for this prompt
law ikr_snap_start_le_live:
  for +r_live: Nat
  for +r_snap: Nat
  {Nat.is_le(ikr_start(IrSnap{}, r_live, r_snap), r_live) == True{} : Bool}

def ikr_snap_start_le_live(r_live, r_snap):
  L.nat_min_le_r(r_snap, r_live)

# promise: a resume of the snapshot never starts past the rows of the prompt that equal the snapshot's rows
# harm: a resume skips rows the snapshot's rings do not hold for this prompt
law ikr_start_le_snap:
  for +r_live: Nat
  for +r_snap: Nat
  {Nat.is_le(ikr_start(IrSnap{}, r_live, r_snap), r_snap) == True{} : Bool}

def ikr_start_le_snap(r_live, r_snap):
  L.nat_min_le_l(r_snap, r_live)

# promise: a prefill that resumes no state starts at position 0
# harm: a request takes over rows without a state that holds them
law ikr_start_fresh:
  for +r_live: Nat
  for +r_snap: Nat
  {ikr_start(IrFresh{}, r_live, r_snap) == 0n : Nat}

def ikr_start_fresh(r_live, r_snap):
  {==}

# promise: a prompt with no row equal to the rows on the card resumes no state
# harm: a state is resumed for a prompt that shares none of its rows
law ikr_pick_none_fresh:
  for +hi_live: Nat
  for +r_snap: Nat
  for +hi_snap: Nat
  for +slack: Nat
  {ikr_pick(0n, hi_live, r_snap, hi_snap, slack) == IrFresh{} : IkResume}

def ikr_pick_none_fresh(hi_live, r_snap, hi_snap, slack):
  match r_snap:
    case 0n:
      {==}
    case 1n+p:
      {==}

# promise: the rows taken over from a prompt of at least one row leave its last row to run
# harm: a request has no row left to compute and no first token
law ikr_cap_last_runs:
  for +common: Nat
  for +n: Nat
  {Nat.is_lt(ikr_cap(common, 1n+n), 1n+n) == True{} : Bool}

def ikr_cap_last_runs(common, n):
  %Equal.sym(Nat, Nat.sub(n, 0n), n, L.nat_sub_zero_r(n)) : {Nat.is_lt(Nat.min(common, _), 1n+n) == True{} : Bool}
  Equal.trans(Bool, Nat.is_lt(Nat.min(common, n), 1n+n), Nat.is_le(Nat.min(common, n), n), True{}, L.nat_lt_succ_eq_le(Nat.min(common, n), n), L.nat_min_le_r(common, n))

# promise: the rows taken over never exceed the rows that equal the rows held
# harm: a resume takes over rows the prompt does not share with the state
law ikr_cap_le_common:
  for +common: Nat
  for +rows: Nat
  {Nat.is_le(ikr_cap(common, rows), common) == True{} : Bool}

def ikr_cap_le_common(common, rows):
  L.nat_min_le_l(common, Nat.sub(rows, 1n))

# promise: the slack never exceeds the slots of a sliding ring beyond the window a pass reads
# harm: a resume is allowed after the sliding rings overwrote rows a pass reads
law ikr_slack_le_ring:
  for +ring: Nat
  for +window: Nat
  for +hist: Nat
  for +taps: Nat
  {Nat.is_le(ikr_slack(ring, window, hist, taps), Nat.sub(ring, window)) == True{} : Bool}

def ikr_slack_le_ring(ring, window, hist, taps):
  L.nat_min_le_l(Nat.sub(ring, window), Nat.sub(hist, taps))

# promise: the slack never exceeds the rows of a history ring beyond the taps a pass reads
# harm: a resume is allowed after the history rings overwrote rows a convolution reads
law ikr_slack_le_hist:
  for +ring: Nat
  for +window: Nat
  for +hist: Nat
  for +taps: Nat
  {Nat.is_le(ikr_slack(ring, window, hist, taps), Nat.sub(hist, taps)) == True{} : Bool}

def ikr_slack_le_hist(ring, window, hist, taps):
  L.nat_min_le_r(Nat.sub(ring, window), Nat.sub(hist, taps))

# The slack of an engine whose drafter keeps context rings of its own (the keys and values of a
# window of positions per windowed layer): no more than the main model's slack, and no more
# than the drafter's rings have beyond their window.
def ikr_slack_with(slack: Nat, ring: Nat, window: Nat) -> Nat:
  Nat.min(slack, Nat.sub(ring, window))

# promise: a drafter never widens the slack of the main model's rings
# harm: a resume is allowed after the main model's rings overwrote rows a pass reads
law ikr_slack_with_le_main:
  for +slack: Nat
  for +ring: Nat
  for +window: Nat
  {Nat.is_le(ikr_slack_with(slack, ring, window), slack) == True{} : Bool}

def ikr_slack_with_le_main(slack, ring, window):
  L.nat_min_le_l(slack, Nat.sub(ring, window))

# promise: with a drafter the slack never exceeds the slots of its context rings beyond their window
# harm: a resume is allowed after the drafter's rings overwrote context a block reads
law ikr_slack_with_le_ring:
  for +slack: Nat
  for +ring: Nat
  for +window: Nat
  {Nat.is_le(ikr_slack_with(slack, ring, window), Nat.sub(ring, window)) == True{} : Bool}

def ikr_slack_with_le_ring(slack, ring, window):
  L.nat_min_le_r(slack, Nat.sub(ring, window))
