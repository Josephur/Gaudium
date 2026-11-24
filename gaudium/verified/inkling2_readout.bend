import Base

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
