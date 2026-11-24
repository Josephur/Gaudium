import Base
import nat_lemmas.bend as L

# ============================================================================
# Inkling 2: launches the live readout charges to one prefill tile
# ============================================================================
#
# The readout spreads a prefill tile's modelled bytes over the launches the tile ran. A tile runs
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
# harm: the readout divides a tile's bytes by no launch
law ik_tile_launches_some:
  for apps: Nat
  for drafter: Bool
  for last: Bool
  {Nat.is_lt(0n, ik_tile_launches(apps, drafter, last)) == True{} : Bool}

def ik_tile_launches_some(apps, drafter, last):
  {==}

# promise: the last tile of a prompt never runs fewer launches than a tile before it
# harm: the readout charges the prompt's readout to no launch
law ik_tile_launches_last_ge:
  for +apps: Nat
  for drafter: Bool
  {Nat.is_le(ik_tile_launches(apps, drafter, False{}), ik_tile_launches(apps, drafter, True{})) == True{} : Bool}

def ik_tile_launches_last_ge(apps, drafter):
  match drafter:
    case True{}:
      L.nat_add_le_mono(1n+apps, 1n+apps, 2n, 3n, L.nat_le_refl(1n+apps), {==})
    case False{}:
      L.nat_add_le_mono(1n+apps, 1n+apps, 0n, 2n, L.nat_le_refl(1n+apps), {==})

# promise: a block drafter never lowers the launches a prefill tile runs
# harm: the readout charges the drafter's context pass to no launch
law ik_tile_launches_drafter_ge:
  for +apps: Nat
  for last: Bool
  {Nat.is_le(ik_tile_launches(apps, False{}, last), ik_tile_launches(apps, True{}, last)) == True{} : Bool}

def ik_tile_launches_drafter_ge(apps, last):
  match last:
    case True{}:
      L.nat_add_le_mono(1n+apps, 1n+apps, 2n, 3n, L.nat_le_refl(1n+apps), {==})
    case False{}:
      L.nat_add_le_mono(1n+apps, 1n+apps, 0n, 2n, L.nat_le_refl(1n+apps), {==})
