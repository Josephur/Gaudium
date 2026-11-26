import Base
import nat_lemmas.bend as L

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

# LAW PW6: Every tile's buffer-set index lies strictly inside the 2-buffer pair (< 2).
# promise: wave_set always returns a valid 2-buffer pair index (< 2)
# catches: wave_set returns 2n for lagged tiles
law wave_set_in_pair:
  for +lag: Bool
  {Nat.is_lt(wave_set(lag), 2n) == True{} : Bool}

def wave_set_in_pair(lag):
  match lag:
    case True{}: {==}
    case False{}: {==}

def pw_lt_add_succ(+h: Nat, +extra: Nat) -> {Nat.is_lt(h, Nat.add(h, 1n+extra)) == True{} : Bool}:
  match h:
    case 0n: {==}
    case 1n+hp:
      pw_lt_add_succ(hp, extra)

def pw_base_half_lt(+m: Nat, +h: Nat, +extra: Nat, +p: Nat) -> {Nat.is_lt(wave_base(m, h, p), wave_base(m, Nat.add(h, 1n+extra), p)) == True{} : Bool}:
  match m:
    case 0n:
      pw_lt_add_succ(h, extra)
    case 1n+mp:
      L.nat_add_lt_le_mono(wave_base(mp, h, p), wave_base(mp, Nat.add(h, 1n+extra), p), p, p, pw_base_half_lt(mp, h, extra, p), L.nat_le_refl(p))

# LAW PW7: Within any tile, later half-steps always receive strictly later launch indices.
# promise: wave_step strictly increases with half-step index for any pair, lag flag, and period
# catches: wave_step drops the half index and ignores half-step ordering
law wave_step_half_order:
  for +m: Nat
  for +lag: Bool
  for +h: Nat
  for +extra: Nat
  for +p: Nat
  {Nat.is_lt(wave_step(m, lag, h, p), wave_step(m, lag, Nat.add(h, 1n+extra), p)) == True{} : Bool}

def wave_step_half_order(m, lag, h, extra, p):
  L.nat_add_lt_l(wave_lag(lag), wave_base(m, h, p), wave_base(m, Nat.add(h, 1n+extra), p), pw_base_half_lt(m, h, extra, p))

def pw_divmod_snd_d(n: Nat, +m: Nat, d1: Nat, d2: Nat, +r: Nat) -> {Pair.snd(Nat, Nat, Nat.divmod.go(n, m, d1, r)) == Pair.snd(Nat, Nat, Nat.divmod.go(n, m, d2, r)) : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+np:
      match m:
        case 0n:
          pw_divmod_snd_d(np, r, 1n+d1, 1n+d2, 0n)
        case 1n+mp:
          pw_divmod_snd_d(np, mp, d1, d2, 1n+r)

def pw_even_mod_zero(+n: Nat) -> {Nat.mod(wave_even(n), 2n) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match p:
        case 0n:
          {==}
        case 1n+q:
          Equal.trans(
            Nat,
            Pair.snd(Nat, Nat, Nat.divmod.go(wave_even(q), 1n, 1n, 0n)),
            Nat.mod(wave_even(q), 2n),
            0n,
            pw_divmod_snd_d(wave_even(q), 1n, 1n, 0n, 0n),
            pw_even_mod_zero(q)
          )

# LAW PW8: wave_tiles always returns an even number of tiles (only complete pairs run).
# promise: wave_tiles always produces an even count divisible by 2
# catches: wave_tiles returns full_tiles without rounding down to complete pairs
law wave_tiles_even:
  for +n: Nat
  for +e: Bool
  {Nat.is_eq(Nat.mod(wave_tiles(n, e), 2n), 0n) == True{} : Bool}

def wave_tiles_even(n, e):
  match e:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Nat, Nat.mod(wave_even(n), 2n), 0n, pw_even_mod_zero(n)) : {Nat.is_eq(_, 0n) == True{} : Bool}
      {==}
