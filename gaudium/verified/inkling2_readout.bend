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

# Launches a tile runs beyond the embedding and the layer applications.
def ik_tile_extra(drafter: Bool, last: Bool) -> Nat:
  match drafter:
    case True{}:
      match last:
        case True{}:
          3n
        case False{}:
          2n
    case False{}:
      match last:
        case True{}:
          2n
        case False{}:
          0n

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
