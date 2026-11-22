import Base

# Prefill wavefront: two prompt tiles in flight.
#
# A tile walks the layers as a chain of half-steps (first half of layer 0, second half of layer 0,
# first half of layer 1, ...). Two consecutive tiles form a pair: the second ("lag") tile runs each
# half-step some launches after the first ("lead") tile, so one launch can hold a half-step of each
# and the card overlaps one tile's MME work with the other's TPC work. Which tiles pair up, how far
# the lag tile trails and how pairs follow each other are choices of the function bodies below;
# the laws only state what every such schedule must keep: no tile is invented, and a tile never
# runs a half-step before the tile in front of it ran the same half-step (recurrent state, KV rows
# and conv state of a layer pass from tile to tile in prompt order).

# Tiles of a prompt that run in pairs: today every whole pair of full tiles.
def wave_even(n: Nat) -> Nat:
  match n:
    case 0n:
      0n
    case 1n+p:
      match p:
        case 0n:
          0n
        case 1n+q:
          1n+1n+wave_even(q)

def wave_tiles_if(full_tiles: Nat, on: Bool) -> Nat:
  match on:
    case True{}:
      wave_even(full_tiles)
    case False{}:
      0n

# `enabled`: the model's tile forms have two-tile recipes and the request can use them.
def wave_tiles(full_tiles: Nat, enabled: Bool) -> Nat:
  wave_tiles_if(full_tiles, enabled)

# Launches the lag tile of a pair trails its lead tile by.
def wave_lag(lag: Bool) -> Nat:
  match lag:
    case True{}:
      1n
    case False{}:
      0n

# Launch index of half-step `half` of the lead tile of pair `pair`; `period` = half-steps of a tile.
def wave_base(pair: Nat, half: Nat, +period: Nat) -> Nat:
  match pair:
    case 0n:
      half
    case 1n+m:
      Nat.add(wave_base(m, half, period), period)

# Launch index of half-step `half` of a tile of pair `pair`.
def wave_step(pair: Nat, lag: Bool, half: Nat, period: Nat) -> Nat:
  Nat.add(wave_lag(lag), wave_base(pair, half, period))

# Buffer set of a tile (input rows, hand-over rows, ids, cursor, drafter taps): two tiles in
# flight never share one.
def wave_set(lag: Bool) -> Nat:
  wave_lag(lag)

# ---- lemmas ----

def pw_zero_le(n: Nat) -> {Nat.is_le(0n, n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+_:
      {==}

def pw_even_le(n: Nat) -> {Nat.is_le(wave_even(n), n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match p:
        case 0n:
          {==}
        case 1n+q:
          pw_even_le(q)

def pw_lt_succ(x: Nat) -> {Nat.is_lt(x, 1n+x) == True{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      pw_lt_succ(p)

# 1 + b < b + (2 + q)
def pw_lt_add(b: Nat, q: Nat) -> {Nat.is_lt(1n+b, Nat.add(b, 1n+1n+q)) == True{} : Bool}:
  match b:
    case 0n:
      {==}
    case 1n+c:
      pw_lt_add(c, q)

# ---- laws ----

# LAW PW1: the wavefront never takes more tiles than the prompt has full tiles: every tile it
# runs exists, and the rest (an odd full tile, the partial last tile) stay with the single-tile
# loop.
law wave_tiles_never_exceed_full:
  for n: Nat
  for e: Bool
  {Nat.is_le(wave_tiles(n, e), n) == True{} : Bool}

def wave_tiles_never_exceed_full(n, e):
  match e:
    case True{}:
      pw_even_le(n)
    case False{}:
      pw_zero_le(n)

# LAW PW2: a request that cannot use the two-tile recipes runs no tile in them.
law wave_off_runs_no_tile:
  for n: Nat
  {wave_tiles(n, False{}) == 0n : Nat}

def wave_off_runs_no_tile(n):
  {==}

# LAW PW3: inside a pair the lag tile runs every half-step strictly after the lead tile ran it,
# so a layer's state and KV rows are written in prompt order.
law wave_lag_follows_lead:
  for m: Nat
  for h: Nat
  for +p: Nat
  {Nat.is_lt(wave_step(m, False{}, h, p), wave_step(m, True{}, h, p)) == True{} : Bool}

def wave_lag_follows_lead(m, h, p):
  pw_lt_succ(wave_base(m, h, p))

# LAW PW4: the lead tile of the next pair runs every half-step strictly after the lag tile of
# this pair ran it, for any tile of at least two half-steps (one layer): prompt order holds
# across pairs too.
law wave_next_pair_follows:
  for m: Nat
  for h: Nat
  for +q: Nat
  {Nat.is_lt(wave_step(m, True{}, h, 1n+1n+q), wave_step(1n+m, False{}, h, 1n+1n+q)) == True{} : Bool}

def wave_next_pair_follows(m, h, q):
  pw_lt_add(wave_base(m, h, 1n+1n+q), q)

# LAW PW5: the two tiles of a pair use different buffer sets.
law wave_sets_differ:
  for u: Unit
  {Nat.is_eq(wave_set(False{}), wave_set(True{})) == False{} : Bool}

def wave_sets_differ(u):
  {==}
