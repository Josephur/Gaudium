import Base

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
