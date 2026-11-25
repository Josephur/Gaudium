# Gaudium's shared Nat lemmas: Euclidean division, remainders, splits q * b + r, alignment.
#
# `nat_lemmas.bend` is the library RustyBends ships (build.rs rewrites it on every build); this
# file adds what real laws here need beyond it. Import it with `import lemmas.bend as G`.
# Every definition is a proof helper (an equality or `Empty` type), erased before Rust codegen.
# A divisor b > 0 is written 1n+bp, as in `nat_lemmas.bend`.

import Base
import nat_lemmas.bend as L

# ==============================================================================
# Order and addition
# ==============================================================================

# b <= (b + x) + y
def lem_le_add_add(+b: Nat, +x: Nat, +y: Nat) -> {Nat.is_le(b, Nat.add(Nat.add(b, x), y)) == True{} : Bool}:
  L.nat_le_trans(b, Nat.add(b, x), Nat.add(Nat.add(b, x), y), L.nat_add_le_r(b, x), L.nat_add_le_r(Nat.add(b, x), y))

# b + x is never below b.
def lem_add_not_lt_l(+b: Nat, +x: Nat, h: {Nat.is_lt(Nat.add(b, x), b) == True{} : Bool}) -> Empty:
  L.nat_lt_not_le(Nat.add(b, x), b, h, L.nat_add_le_r(b, x))

# a < b  ->  k + a < k + b
def lem_add_lt_l(+k: Nat, +a: Nat, +b: Nat, h: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_lt(Nat.add(k, a), Nat.add(k, b)) == True{} : Bool}:
  match k:
    case 0n:
      h
    case 1n+kp:
      lem_add_lt_l(kp, a, b, h)

# k + x == k + y  ->  x == y
def lem_add_cancel_l(+k: Nat, +x: Nat, +y: Nat, e: {Nat.add(k, x) == Nat.add(k, y) : Nat}) -> {x == y : Nat}:
  %L.nat_add_sub_cancel_l(k, x) : {_ == y : Nat}
  %Equal.sym(Nat, Nat.add(k, x), Nat.add(k, y), e) : {Nat.sub(_, k) == y : Nat}
  L.nat_add_sub_cancel_l(k, y)

# a == b, a < c  ->  b < c
def lem_lt_subst_l(-a: Nat, -b: Nat, -c: Nat, e: {a == b : Nat}, h: {Nat.is_lt(a, c) == True{} : Bool}) -> {Nat.is_lt(b, c) == True{} : Bool}:
  %e : {Nat.is_lt(_, c) == True{} : Bool}
  h

# j < k + n, 0 < n  ->  j - k < n
def lem_sub_lt(+j: Nat, +k: Nat, +n: Nat, hn: {Nat.is_lt(0n, n) == True{} : Bool}, h: {Nat.is_lt(j, Nat.add(k, n)) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(j, k), n) == True{} : Bool}:
  match j k:
    case 0n _:
      hn
    case 1n+jp 0n:
      h
    case 1n+jp 1n+kp:
      lem_sub_lt(jp, kp, n, hn, h)

# ==============================================================================
# Splits q * b + r with r < b
# ==============================================================================

# r1 < b is refuted when r1 == (b + m) + r2.
def lem_split_low_refuted(+bp: Nat, +m: Nat, +r1: Nat, +r2: Nat, h1: {Nat.is_lt(r1, 1n+bp) == True{} : Bool}, e: {r1 == Nat.add(Nat.add(1n+bp, m), r2) : Nat}) -> Empty:
  L.nat_lt_not_le(Nat.add(Nat.add(1n+bp, m), r2), 1n+bp, lem_lt_subst_l(r1, Nat.add(Nat.add(1n+bp, m), r2), 1n+bp, e, h1), lem_le_add_add(1n+bp, m, r2))

# Two splits whose remainders are below b are the same split: q1 * b + r1 == q2 * b + r2,
# r1 < b, r2 < b  ->  q1 == q2.
def lem_split_unique_q(
  +q1: Nat,
  +r1: Nat,
  +q2: Nat,
  +r2: Nat,
  +bp: Nat,
  h1: {Nat.is_lt(r1, 1n+bp) == True{} : Bool},
  h2: {Nat.is_lt(r2, 1n+bp) == True{} : Bool},
  e: {Nat.add(Nat.mul(q1, 1n+bp), r1) == Nat.add(Nat.mul(q2, 1n+bp), r2) : Nat}
) -> {q1 == q2 : Nat}:
  match q1 q2:
    case 0n 0n:
      {==}
    case 0n 1n+p2:
      Empty.absurd({0n == 1n+p2 : Nat}, lem_split_low_refuted(bp, Nat.mul(p2, 1n+bp), r1, r2, h1, e))
    case 1n+p1 0n:
      Empty.absurd({1n+p1 == 0n : Nat}, lem_split_low_refuted(bp, Nat.mul(p1, 1n+bp), r2, r1, h2, Equal.sym(Nat, Nat.add(Nat.add(1n+bp, Nat.mul(p1, 1n+bp)), r1), r2, e)))
    case 1n+p1 1n+p2:
      e1 = Equal.trans(Nat,
        Nat.add(1n+bp, Nat.add(Nat.mul(p1, 1n+bp), r1)),
        Nat.add(Nat.add(1n+bp, Nat.mul(p1, 1n+bp)), r1),
        Nat.add(Nat.add(1n+bp, Nat.mul(p2, 1n+bp)), r2),
        Equal.sym(Nat, Nat.add(Nat.add(1n+bp, Nat.mul(p1, 1n+bp)), r1), Nat.add(1n+bp, Nat.add(Nat.mul(p1, 1n+bp), r1)), L.nat_add_assoc(1n+bp, Nat.mul(p1, 1n+bp), r1)),
        e)
      e2 = Equal.trans(Nat,
        Nat.add(1n+bp, Nat.add(Nat.mul(p1, 1n+bp), r1)),
        Nat.add(Nat.add(1n+bp, Nat.mul(p2, 1n+bp)), r2),
        Nat.add(1n+bp, Nat.add(Nat.mul(p2, 1n+bp), r2)),
        e1,
        L.nat_add_assoc(1n+bp, Nat.mul(p2, 1n+bp), r2))
      %lem_split_unique_q(p1, r1, p2, r2, bp, h1, h2, lem_add_cancel_l(1n+bp, Nat.add(Nat.mul(p1, 1n+bp), r1), Nat.add(Nat.mul(p2, 1n+bp), r2), e2)) : {1n+p1 == 1n+_ : Nat}
      {==}

# q1 == q2 rewrites q2 to q1 in a split equation.
def lem_split_subst_q(-q1: Nat, -q2: Nat, -r1: Nat, -r2: Nat, -b: Nat, eq: {q1 == q2 : Nat}, e: {Nat.add(Nat.mul(q1, b), r1) == Nat.add(Nat.mul(q2, b), r2) : Nat}) -> {Nat.add(Nat.mul(q1, b), r1) == Nat.add(Nat.mul(q1, b), r2) : Nat}:
  %Equal.sym(Nat, q1, q2, eq) : {Nat.add(Nat.mul(q1, b), r1) == Nat.add(Nat.mul(_, b), r2) : Nat}
  e

# The same two splits have the same remainder: r1 == r2.
def lem_split_unique_r(
  +q1: Nat,
  +r1: Nat,
  +q2: Nat,
  +r2: Nat,
  +bp: Nat,
  h1: {Nat.is_lt(r1, 1n+bp) == True{} : Bool},
  h2: {Nat.is_lt(r2, 1n+bp) == True{} : Bool},
  +e: {Nat.add(Nat.mul(q1, 1n+bp), r1) == Nat.add(Nat.mul(q2, 1n+bp), r2) : Nat}
) -> {r1 == r2 : Nat}:
  lem_add_cancel_l(Nat.mul(q1, 1n+bp), r1, r2, lem_split_subst_q(q1, q2, r1, r2, 1n+bp, lem_split_unique_q(q1, r1, q2, r2, bp, h1, h2, e), e))

# ==============================================================================
# Euclidean division: the quotient and remainder of q * b + r, r < b
# ==============================================================================

# (q * b + r) / b == q
def lem_div_split(+q: Nat, +r: Nat, +bp: Nat, h: {Nat.is_lt(r, 1n+bp) == True{} : Bool}) -> {Nat.div(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp) == q : Nat}:
  lem_split_unique_q(
    Nat.div(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp),
    Nat.mod(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp),
    q,
    r,
    bp,
    L.nat_mod_lt(Nat.add(Nat.mul(q, 1n+bp), r), bp),
    h,
    L.nat_div_mod_r(Nat.add(Nat.mul(q, 1n+bp), r), bp))

# (q * b + r) % b == r
def lem_mod_split(+q: Nat, +r: Nat, +bp: Nat, h: {Nat.is_lt(r, 1n+bp) == True{} : Bool}) -> {Nat.mod(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp) == r : Nat}:
  lem_split_unique_r(
    Nat.div(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp),
    Nat.mod(Nat.add(Nat.mul(q, 1n+bp), r), 1n+bp),
    q,
    r,
    bp,
    L.nat_mod_lt(Nat.add(Nat.mul(q, 1n+bp), r), bp),
    h,
    L.nat_div_mod_r(Nat.add(Nat.mul(q, 1n+bp), r), bp))

# a < b  ->  a % b == a
def lem_mod_small(+a: Nat, +bp: Nat, h: {Nat.is_lt(a, 1n+bp) == True{} : Bool}) -> {Nat.mod(a, 1n+bp) == a : Nat}:
  lem_mod_split(0n, a, bp, h)

# a < b  ->  a / b == 0
def lem_div_small(+a: Nat, +bp: Nat, h: {Nat.is_lt(a, 1n+bp) == True{} : Bool}) -> {Nat.div(a, 1n+bp) == 0n : Nat}:
  lem_div_split(0n, a, bp, h)

# Two numbers with the same quotient and the same remainder by b are equal.
def lem_eq_of_div_mod(+a: Nat, +c: Nat, +bp: Nat, hq: {Nat.div(a, 1n+bp) == Nat.div(c, 1n+bp) : Nat}, hr: {Nat.mod(a, 1n+bp) == Nat.mod(c, 1n+bp) : Nat}) -> {a == c : Nat}:
  %L.nat_div_mod_r(a, bp) : {_ == c : Nat}
  %L.nat_div_mod_r(c, bp) : {Nat.add(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp)) == _ : Nat}
  %hq : {Nat.add(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp)) == Nat.add(Nat.mul(_, 1n+bp), Nat.mod(c, 1n+bp)) : Nat}
  %hr : {Nat.add(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp)) == Nat.add(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), _) : Nat}
  {==}

# A split with q < n stays below n * b: q < n, r < b  ->  q * b + r < n * b
def lem_split_lt(+q: Nat, +r: Nat, +n: Nat, +bp: Nat, hq: {Nat.is_lt(q, n) == True{} : Bool}, hr: {Nat.is_lt(r, 1n+bp) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(q, 1n+bp), r), Nat.mul(n, 1n+bp)) == True{} : Bool}:
  match q n:
    case 0n 0n:
      L.bool_absurd({Nat.is_lt(r, 0n) == True{} : Bool}, hq)
    case 0n 1n+np:
      L.nat_lt_le_trans(r, 1n+bp, Nat.add(1n+bp, Nat.mul(np, 1n+bp)), hr, L.nat_add_le_r(1n+bp, Nat.mul(np, 1n+bp)))
    case 1n+qp 0n:
      L.bool_absurd({Nat.is_lt(Nat.add(Nat.add(1n+bp, Nat.mul(qp, 1n+bp)), r), 0n) == True{} : Bool}, hq)
    case 1n+qp 1n+np:
      %Equal.sym(Nat, Nat.add(Nat.add(1n+bp, Nat.mul(qp, 1n+bp)), r), Nat.add(1n+bp, Nat.add(Nat.mul(qp, 1n+bp), r)), L.nat_add_assoc(1n+bp, Nat.mul(qp, 1n+bp), r)) : {Nat.is_lt(_, Nat.add(1n+bp, Nat.mul(np, 1n+bp))) == True{} : Bool}
      lem_add_lt_l(1n+bp, Nat.add(Nat.mul(qp, 1n+bp), r), Nat.mul(np, 1n+bp), lem_split_lt(qp, r, np, bp, hq, hr))

# ==============================================================================
# Alignment: multiples of b
# ==============================================================================

# (q * b) % b == 0: a multiple of b is aligned to b.
def lem_mod_mul(+q: Nat, +bp: Nat) -> {Nat.mod(Nat.mul(q, 1n+bp), 1n+bp) == 0n : Nat}:
  %L.nat_add_zero_r(Nat.mul(q, 1n+bp)) : {Nat.mod(_, 1n+bp) == 0n : Nat}
  lem_mod_split(q, 0n, bp, {==})

# (q * b) / b == q
def lem_div_mul(+q: Nat, +bp: Nat) -> {Nat.div(Nat.mul(q, 1n+bp), 1n+bp) == q : Nat}:
  %L.nat_add_zero_r(Nat.mul(q, 1n+bp)) : {Nat.div(_, 1n+bp) == q : Nat}
  lem_div_split(q, 0n, bp, {==})

# The aligned-down part of a never exceeds a: (a / b) * b <= a
def lem_align_down_le(+a: Nat, +bp: Nat) -> {Nat.is_le(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), a) == True{} : Bool}:
  %L.nat_div_mod_r(a, bp) : {Nat.is_le(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), _) == True{} : Bool}
  L.nat_add_le_r(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp))
