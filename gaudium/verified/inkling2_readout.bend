import Base
import nat_lemmas.bend as L

# ============================================================================
# Inkling 2: launches the live readout charges to one prefill tile
# ============================================================================
#
# The readout reports how many launches a prefill tile ran beside its modelled bytes. A tile runs
# the embedding and every layer application; with a block drafter also the tile's final mix and
# the drafter's context pass; the last tile of a prompt also the readout (the final mix, unless
# the drafter's pass already ran it, and the head).

# Which launches a tile runs beyond the embedding and its layer applications: the tile's final mix
# (the drafter reads the tile's final states, the last tile's readout its last row), the drafter's
# context pass over the tile's rows, the head (the request's first token). The launch count and the
# readout's byte model both follow this plan.
def ik_tile_runs_final(drafter: Bool, last: Bool) -> Bool:
  Bool.or(drafter, last)

def ik_tile_runs_ctx(drafter: Bool) -> Bool:
  drafter

def ik_tile_runs_head(last: Bool) -> Bool:
  last

def ik_b2n(b: Bool) -> Nat:
  match b:
    case True{}:
      1n
    case False{}:
      0n

# Launches a tile runs beyond the embedding and the layer applications.
def ik_tile_extra(+drafter: Bool, +last: Bool) -> Nat:
  Nat.add(
    ik_b2n(ik_tile_runs_final(drafter, last)),
    Nat.add(ik_b2n(ik_tile_runs_ctx(drafter)), ik_b2n(ik_tile_runs_head(last)))
  )

# promise: the last tile of a prompt always runs the head that gives the request's first token
# harm: a prompt ends without its first token's readout, or the readout reports none
law ik_tile_last_runs_head:
  {ik_tile_runs_head(True{}) == True{} : Bool}

def ik_tile_last_runs_head():
  {==}

# promise: a tile run with a drafter always runs the drafter's context pass over its rows
# harm: the drafter drafts without the context of the tile's rows
law ik_tile_drafter_runs_ctx:
  {ik_tile_runs_ctx(True{}) == True{} : Bool}

def ik_tile_drafter_runs_ctx():
  {==}

# Launches of one prefill tile of a model of `apps` layer applications.
def ik_tile_launches(apps: Nat, drafter: Bool, last: Bool) -> Nat:
  Nat.add(1n+apps, ik_tile_extra(drafter, last))

# promise: a prefill tile always runs at least one launch
# harm: the readout reports a tile that ran nothing on the card
law ik_tile_launches_some:
  for apps: Nat
  for drafter: Bool
  for last: Bool
  {Nat.is_lt(0n, ik_tile_launches(apps, drafter, last)) == True{} : Bool}

def ik_tile_launches_some(apps, drafter, last):
  {==}

# ============================================================================
# Positions an attention state tensor holds
# ============================================================================
#
# A state tensor of shape [lanes, slots, cols] with `nkv` key / value heads holds `slots` positions
# per chunk and cols / nkv chunks: a sliding ring is one chunk of its slots, a global cache its
# chunk rows times its chunks. The readout charges a pass's write to the tensor for the pass's
# rows out of these positions. The shape is read in the shell; on overflow (`total`: `None`) the
# readout uses the rule-wide count, which the tensor's own byte size makes unreachable.

# rusty-bends: total
def ik_state_positions(slots: Nat, cols: Nat, nkv: Nat) -> Nat:
  Nat.mul(Nat.max(1n, Nat.div(cols, nkv)), slots)

def ik_state_some(+m: Nat, +s: Nat, h: {Nat.is_le(1n, m) == True{} : Bool}) -> {Nat.is_lt(0n, Nat.mul(m, 1n+s)) == True{} : Bool}:
  match m:
    case 0n:
      L.bool_absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, h)
    case 1n+p:
      {==}

# promise: a state tensor with at least one slot row holds at least one position
# harm: the readout charges a whole cache to every pass that writes one row of it
law ik_state_positions_some:
  for +slots: Nat
  for +cols: Nat
  for +nkv: Nat
  {Nat.is_lt(0n, ik_state_positions(1n+slots, cols, nkv)) == True{} : Bool}

def ik_state_positions_some(slots, cols, nkv):
  ik_state_some(Nat.max(1n, Nat.div(cols, nkv)), slots, L.nat_max_ge_l(1n, Nat.div(cols, nkv)))

# ============================================================================
# Whether a request's decode runs the drafter
# ============================================================================

# The decode of a request runs the loaded drafter unless the request asks for the plain loop (the
# K = 0 reference); the decode dispatch and the live record's label both ask this.
def ik_decode_runs_drafter(has_drafter: Bool, plain_asked: Bool) -> Bool:
  Bool.and(has_drafter, Bool.not(plain_asked))

# promise: a request that asks for the plain decode loop never runs the drafter
# harm: the K = 0 reference of a comparison is itself speculative
law ik_decode_plain_no_drafter:
  for has_drafter: Bool
  {ik_decode_runs_drafter(has_drafter, True{}) == False{} : Bool}

def ik_decode_plain_no_drafter(has_drafter):
  match has_drafter:
    case True{}:
      {==}
    case False{}:
      {==}

# promise: an engine without a drafter never runs one
# harm: a request is labelled or run as speculative with no drafter loaded
law ik_decode_no_drafter:
  for plain_asked: Bool
  {ik_decode_runs_drafter(False{}, plain_asked) == False{} : Bool}

def ik_decode_no_drafter(plain_asked):
  {==}

# ============================================================================
# Which phase a launch's bytes are charged to
# ============================================================================
#
# A launch charged as phase `to` moves the bytes of every other phase into `to`, except a phase
# kept apart (`keep`, when `has_keep`): the verify pass keeps its readout apart.

type IkPhase is Data:
  PhVerify{}
  PhDraft{}
  PhReadout{}
  PhOther{}

def ik_phase_same(a: IkPhase, b: IkPhase) -> Bool:
  match a b:
    case PhVerify{} PhVerify{}:
      True{}
    case PhVerify{} PhDraft{}:
      False{}
    case PhVerify{} PhReadout{}:
      False{}
    case PhVerify{} PhOther{}:
      False{}
    case PhDraft{} PhVerify{}:
      False{}
    case PhDraft{} PhDraft{}:
      True{}
    case PhDraft{} PhReadout{}:
      False{}
    case PhDraft{} PhOther{}:
      False{}
    case PhReadout{} PhVerify{}:
      False{}
    case PhReadout{} PhDraft{}:
      False{}
    case PhReadout{} PhReadout{}:
      True{}
    case PhReadout{} PhOther{}:
      False{}
    case PhOther{} PhVerify{}:
      False{}
    case PhOther{} PhDraft{}:
      False{}
    case PhOther{} PhReadout{}:
      False{}
    case PhOther{} PhOther{}:
      True{}

# Whether the bytes of phase `p` move into `to`.
def ik_phase_moves(+p: IkPhase, to: IkPhase, keep: IkPhase, has_keep: Bool) -> Bool:
  Bool.and(Bool.not(ik_phase_same(p, to)), Bool.not(Bool.and(has_keep, ik_phase_same(p, keep))))

# promise: the phase a launch is charged as always keeps its own bytes
# harm: the readout moves a phase's bytes out of the phase the launch is charged as
law ik_phase_target_stays:
  for +to: IkPhase
  for keep: IkPhase
  for has_keep: Bool
  {ik_phase_moves(to, to, keep, has_keep) == False{} : Bool}

def ik_phase_target_stays(to, keep, has_keep):
  match to:
    case PhVerify{}:
      {==}
    case PhDraft{}:
      {==}
    case PhReadout{}:
      {==}
    case PhOther{}:
      {==}

# promise: a phase kept apart keeps its own bytes
# harm: the readout charges the kept phase (the verify pass's readout) to the verify pass
law ik_phase_kept_stays:
  for +p: IkPhase
  for to: IkPhase
  {ik_phase_moves(p, to, p, True{}) == False{} : Bool}

def ik_phase_kept_stays(p, to):
  match p to:
    case PhVerify{} PhVerify{}:
      {==}
    case PhVerify{} PhDraft{}:
      {==}
    case PhVerify{} PhReadout{}:
      {==}
    case PhVerify{} PhOther{}:
      {==}
    case PhDraft{} PhVerify{}:
      {==}
    case PhDraft{} PhDraft{}:
      {==}
    case PhDraft{} PhReadout{}:
      {==}
    case PhDraft{} PhOther{}:
      {==}
    case PhReadout{} PhVerify{}:
      {==}
    case PhReadout{} PhDraft{}:
      {==}
    case PhReadout{} PhReadout{}:
      {==}
    case PhReadout{} PhOther{}:
      {==}
    case PhOther{} PhVerify{}:
      {==}
    case PhOther{} PhDraft{}:
      {==}
    case PhOther{} PhReadout{}:
      {==}
    case PhOther{} PhOther{}:
      {==}

