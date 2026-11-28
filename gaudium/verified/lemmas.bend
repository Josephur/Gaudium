# Shared project proof helpers beyond `nat_lemmas.bend` (AGENTS.md §4 item 6).
# Every definition here returns an equality type (`q=0`) and is erased before Rust codegen.

import Base
import nat_lemmas.bend as L

def g_nat_not_le_imp_lt(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == False{} : Bool}) -> {Nat.is_lt(b, a) == True{} : Bool}:
  match a b:
    case 0n 0n:
      L.bool_absurd_tf({Nat.is_lt(0n, 0n) == True{} : Bool}, h)
    case 0n 1n+bp:
      L.bool_absurd_tf({Nat.is_lt(1n+bp, 0n) == True{} : Bool}, h)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      g_nat_not_le_imp_lt(ap, bp, h)

def g_nat_lt_imp_succ_le(+b: Nat, +a: Nat, h: {Nat.is_lt(b, a) == True{} : Bool}) -> {Nat.is_le(Nat.add(b, 1n), a) == True{} : Bool}:
  match b a:
    case 0n 0n:
      L.bool_absurd({Nat.is_le(1n, 0n) == True{} : Bool}, h)
    case 0n 1n+ap:
      L.nat_le_zero(ap)
    case 1n+bp 0n:
      L.bool_absurd({Nat.is_le(Nat.add(1n+bp, 1n), 0n) == True{} : Bool}, h)
    case 1n+bp 1n+ap:
      g_nat_lt_imp_succ_le(bp, ap, h)

def g_nat_ceil_le_r(
  +n: Nat,
  +dp: Nat,
  +q: Nat,
  eqq: {q == Nat.div(Nat.add(n, dp), 1n+dp) : Nat},
  hadd1: {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+dp), 1n), dp), Nat.add(n, dp)) == True{} : Bool}
) -> {Nat.is_le(Nat.add(Nat.mul(q, 1n+dp), 1n+dp), Nat.add(Nat.mul(q, 1n+dp), Nat.mod(Nat.add(n, dp), 1n+dp))) == True{} : Bool}:
  %L.nat_add_assoc(Nat.mul(q, 1n+dp), 1n, dp) : {Nat.is_le(_, Nat.add(Nat.mul(q, 1n+dp), Nat.mod(Nat.add(n, dp), 1n+dp))) == True{} : Bool}
  %Equal.sym(Nat, q, Nat.div(Nat.add(n, dp), 1n+dp), eqq) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+dp), 1n), dp), Nat.add(Nat.mul(_, 1n+dp), Nat.mod(Nat.add(n, dp), 1n+dp))) == True{} : Bool}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.add(n, dp), 1n+dp), 1n+dp), Nat.mod(Nat.add(n, dp), 1n+dp)), Nat.add(n, dp), L.nat_div_mod_r(Nat.add(n, dp), dp)) : {Nat.is_le(Nat.add(Nat.add(Nat.mul(q, 1n+dp), 1n), dp), _) == True{} : Bool}
  hadd1

def g_nat_ceil_le_aux(
  +n: Nat,
  +dp: Nat,
  +q: Nat,
  b: Bool,
  eb: {b == Nat.is_le(n, Nat.mul(q, 1n+dp)) : Bool},
  eqq: {q == Nat.div(Nat.add(n, dp), 1n+dp) : Nat}
) -> {Nat.is_le(n, Nat.mul(q, 1n+dp)) == True{} : Bool}:
  match b:
    case True{}:
      Equal.sym(Bool, True{}, Nat.is_le(n, Nat.mul(q, 1n+dp)), eb)
    case False{}:
      +x = Nat.mul(q, 1n+dp)
      +hlt = g_nat_not_le_imp_lt(n, x, Equal.sym(Bool, False{}, Nat.is_le(n, x), eb))
      +hsle = g_nat_lt_imp_succ_le(x, n, hlt)
      +hadd1 = L.nat_add_le_mono(Nat.add(x, 1n), n, dp, dp, hsle, L.nat_le_refl(dp))
      +hle_bad = g_nat_ceil_le_r(n, dp, q, eqq, hadd1)
      +hlt_bad = L.nat_add_lt_l(x, Nat.mod(Nat.add(n, dp), 1n+dp), 1n+dp, L.nat_mod_lt(Nat.add(n, dp), dp))
      Empty.absurd({Nat.is_le(n, x) == True{} : Bool}, L.nat_lt_not_le(Nat.add(x, Nat.mod(Nat.add(n, dp), 1n+dp)), Nat.add(x, 1n+dp), hlt_bad, hle_bad))

def g_nat_ceil_div_covers(+n: Nat, +dp: Nat) -> {Nat.is_le(n, Nat.mul(Nat.div(Nat.add(n, dp), 1n+dp), 1n+dp)) == True{} : Bool}:
  +q = Nat.div(Nat.add(n, dp), 1n+dp)
  g_nat_ceil_le_aux(n, dp, q, Nat.is_le(n, Nat.mul(q, 1n+dp)), {==}, {==})

def g_nat_le_add_succ_false(+m: Nat, +d: Nat) -> {Nat.is_le(Nat.add(m, 1n+d), m) == False{} : Bool}:
  match m:
    case 0n: {==}
    case 1n+mp:
      g_nat_le_add_succ_false(mp, d)

def g_nat_sub_le_self(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n:
      L.nat_le_refl(1n+ap)
    case 1n+ap 1n+bp:
      L.nat_le_succ_r(Nat.sub(ap, bp), ap, g_nat_sub_le_self(ap, bp))

def g_nat_eq_refl(n: Nat) -> {Nat.is_eq(n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      g_nat_eq_refl(np)

def g_nat_eq_add_succ_false(+n: Nat, +d: Nat) -> {Nat.is_eq(Nat.add(n, 1n+d), n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      g_nat_eq_add_succ_false(np, d)

def g_nat_eq_succ_add_false(+n: Nat, +d: Nat) -> {Nat.is_eq(n, Nat.add(n, 1n+d)) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      g_nat_eq_succ_add_false(np, d)

def g_bool_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def g_bool_or_true_r(b: Bool) -> {Bool.or(b, True{}) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# `Nat.max(a, b)` is at most any `c` that both `a` and `b` are at most (least upper bound).
def g_nat_max_le(
  a: Nat,
  b: Nat,
  c: Nat,
  ha: {Nat.is_le(a, c) == True{} : Bool},
  hb: {Nat.is_le(b, c) == True{} : Bool}
) -> {Nat.is_le(Nat.max(a, b), c) == True{} : Bool}:
  match a:
    case 0n:
      hb
    case 1n+ap:
      match b:
        case 0n:
          ha
        case 1n+bp:
          match c:
            case 0n:
              L.bool_absurd({Nat.is_le(Nat.max(1n+ap, 1n+bp), 0n) == True{} : Bool}, ha)
            case 1n+cp:
              g_nat_max_le(ap, bp, cp, ha, hb)

def g_nat_align_up_nz_sum(+x: Nat, +ap: Nat, +rem: Nat, erem: {rem == Nat.mod(x, 1n+ap) : Nat}, h_lt: {Nat.is_lt(rem, 1n+ap) == True{} : Bool}) -> {Nat.add(x, Nat.sub(1n+ap, rem)) == Nat.mul(1n+Nat.div(x, 1n+ap), 1n+ap) : Nat}:
  +q = Nat.div(x, 1n+ap)
  %L.nat_div_mod_r(x, ap) : {Nat.add(_, Nat.sub(1n+ap, rem)) == Nat.mul(1n+q, 1n+ap) : Nat}
  %erem : {Nat.add(Nat.add(Nat.mul(q, 1n+ap), _), Nat.sub(1n+ap, rem)) == Nat.mul(1n+q, 1n+ap) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(Nat.mul(q, 1n+ap), rem), Nat.sub(1n+ap, rem)), Nat.add(Nat.mul(q, 1n+ap), Nat.add(rem, Nat.sub(1n+ap, rem))), L.nat_add_assoc(Nat.mul(q, 1n+ap), rem, Nat.sub(1n+ap, rem))) : {_ == Nat.mul(1n+q, 1n+ap) : Nat}
  %L.nat_add_comm(Nat.sub(1n+ap, rem), rem) : {Nat.add(Nat.mul(q, 1n+ap), _) == Nat.mul(1n+q, 1n+ap) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.sub(1n+ap, rem), rem), 1n+ap, L.nat_sub_add_cancel(1n+ap, rem, L.nat_lt_imp_le(rem, 1n+ap, h_lt))) : {Nat.add(Nat.mul(q, 1n+ap), _) == Nat.mul(1n+q, 1n+ap) : Nat}
  L.nat_add_comm(Nat.mul(q, 1n+ap), 1n+ap)

def g_nat_max_one_succ(+x: Nat) -> {Nat.max(1n, 1n+x) == 1n+x : Nat}:
  match x:
    case 0n: {==}
    case 1n+_: {==}

# A proof of `m < 0` proves anything.
def g_nat_lt_zero_absurd(-A: Type, +m: Nat, +h: {Nat.is_lt(m, 0n) == True{} : Bool}) -> A:
  match m:
    case 0n:
      L.bool_absurd(A, h)
    case 1n+mp:
      L.bool_absurd(A, h)

# `Nat.is_eq(a, b)` is True only when `a == b`.
def g_nat_eq_of_is_eq(+a: Nat, +b: Nat, +h: {Nat.is_eq(a, b) == True{} : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      L.bool_absurd({0n == 1n+bp : Nat}, h)
    case 1n+ap 0n:
      L.bool_absurd({1n+ap == 0n : Nat}, h)
    case 1n+ap 1n+bp:
      %g_nat_eq_of_is_eq(ap, bp, h) : {1n+ap == 1n+_ : Nat}
      {==}

# `a + e < a` never holds.
def g_nat_lt_add_r_false(+a: Nat, e: Nat) -> {Nat.is_lt(Nat.add(a, e), a) == False{} : Bool}:
  match a:
    case 0n:
      match e:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap:
      g_nat_lt_add_r_false(ap, e)

# `n < n + 1 + d` always holds.
def g_nat_lt_add_succ(+n: Nat, d: Nat) -> {Nat.is_lt(n, Nat.add(n, 1n+d)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      g_nat_lt_add_succ(np, d)

# `n < n` never holds.
def g_nat_lt_irrefl(+n: Nat) -> {Nat.is_lt(n, n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      g_nat_lt_irrefl(np)

# `min(b, k) <= min(b, k + j)`: the smaller of b and k grows with k.
def g_nat_min_le_mono_r(+b: Nat, +k: Nat, j: Nat) -> {Nat.is_le(Nat.min(b, k), Nat.min(b, Nat.add(k, j))) == True{} : Bool}:
  match b k:
    case 0n _:
      L.nat_le_zero(0n)
    case 1n+bp 0n:
      L.nat_le_zero(Nat.min(1n+bp, j))
    case 1n+bp 1n+kp:
      g_nat_min_le_mono_r(bp, kp, j)

# `off <= size` and `len <= size - off` keep `off + len` within `size` (a span check).
def g_span_fit(+o: Nat, +l: Nat, +s: Nat, b1: Bool, e1: {Nat.is_le(o, s) == b1 : Bool}, b2: Bool, e2: {Nat.is_le(l, Nat.sub(s, o)) == b2 : Bool}) -> {Bool.and(Bool.and(b1, b2), Nat.is_lt(s, Nat.add(o, l))) == False{} : Bool}:
  match b1:
    case False{}:
      {==}
    case True{}:
      match b2:
        case False{}:
          {==}
        case True{}:
          L.nat_le_lt_false(s, Nat.add(o, l), L.nat_le_subst_r(Nat.add(o, l), Nat.add(o, Nat.sub(s, o)), s, Equal.trans(Nat, Nat.add(o, Nat.sub(s, o)), Nat.add(Nat.sub(s, o), o), s, L.nat_add_comm(o, Nat.sub(s, o)), L.nat_sub_add_cancel(s, o, e1)), L.nat_add_le_mono(o, o, l, Nat.sub(s, o), L.nat_le_refl(o), e2)))

# `n <= a / (1 + ep)` keeps `n * (1 + ep)` within `a` (a count check against a quotient).
def g_fit_within(+n: Nat, +ep: Nat, +a: Nat, b: Bool, e: {Nat.is_le(n, Nat.div(a, 1n+ep)) == b : Bool}) -> {Bool.and(b, Nat.is_lt(a, Nat.mul(n, 1n+ep))) == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(a, Nat.mul(n, 1n+ep), L.nat_le_trans(Nat.mul(n, 1n+ep), Nat.mul(Nat.div(a, 1n+ep), 1n+ep), a, L.nat_mul_le_mono_r(n, Nat.div(a, 1n+ep), 1n+ep, e), L.nat_le_subst_r(Nat.mul(Nat.div(a, 1n+ep), 1n+ep), Nat.add(Nat.mul(Nat.div(a, 1n+ep), 1n+ep), Nat.mod(a, 1n+ep)), a, L.nat_div_mod_r(a, ep), L.nat_add_le_r(Nat.mul(Nat.div(a, 1n+ep), 1n+ep), Nat.mod(a, 1n+ep)))))

# Two true conditions make a true conjunction.
def g_bool_and_tt(a: Bool, -b: Bool, ea: {a == True{} : Bool}, eb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  match a:
    case True{}:
      eb
    case False{}:
      L.bool_absurd({Bool.and(False{}, b) == True{} : Bool}, ea)

# `k < n` gives `k + 1 <= n`, with the successor written as a constructor.
def g_nat_lt_succ_le(+k: Nat, +n: Nat, h: {Nat.is_lt(k, n) == True{} : Bool}) -> {Nat.is_le(1n+k, n) == True{} : Bool}:
  match k n:
    case 0n 0n:
      L.bool_absurd({Nat.is_le(1n, 0n) == True{} : Bool}, h)
    case 0n 1n+np:
      L.nat_le_zero(np)
    case 1n+kp 0n:
      L.bool_absurd({Nat.is_le(1n+1n+kp, 0n) == True{} : Bool}, h)
    case 1n+kp 1n+np:
      g_nat_lt_succ_le(kp, np, h)
