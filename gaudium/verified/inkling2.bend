# Inkling 2 (looped MoE decoder): the arithmetic every graph, plan and weight packer of the family
# shares. Specifies:
#   - the schedule of layer applications (a middle span of layers runs several times with shared weights)
#   - how many block states a layer's residual mixes read, and which layers open a block
#   - ring sizes of the sliding layers' K / V state and of the short convolutions' row history
#   - the geometry of the reversed, padded relative-position bias a softmax block loads
#   - the channel layout of the query-side projection and the byte geometry of 4-bit expert rows
#   - the shape guard, prefill tiles and the reuse of a prefill tile's row buffers
# and proves, for every version of these functions, the bounds the graphs, the packer and the
# device kernels rely on (sections marked "Laws").

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 0. Proof helpers (Bool and Nat facts; erased before Rust codegen)
# ============================================================================

def ikp_and_l(a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      h

def ikp_and_r(a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case True{}:
      h
    case False{}:
      L.bool_absurd({b == True{} : Bool}, h)

def ikp_and_tt(a: Bool, -b: Bool, ha: {a == True{} : Bool}, hb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  match a:
    case True{}:
      hb
    case False{}:
      ha

def ikp_and_fl(a: Bool, -b: Bool, h: {a == False{} : Bool}) -> {Bool.and(a, b) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      L.bool_absurd_tf({Bool.and(True{}, b) == False{} : Bool}, h)

def ikp_and_fr(a: Bool, -b: Bool, h: {b == False{} : Bool}) -> {Bool.and(a, b) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      h

def ikp_not_t(x: Bool, h: {x == True{} : Bool}) -> {Bool.not(x) == False{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      L.bool_absurd({True{} == False{} : Bool}, h)

def ikp_not_tf(x: Bool, h: {Bool.not(x) == True{} : Bool}) -> {x == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      L.bool_absurd({True{} == False{} : Bool}, h)

def ikp_ge_le(+a: Nat, +b: Nat) -> {Nat.is_ge(a, b) == Nat.is_le(b, a) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      ikp_ge_le(ap, bp)

def ikp_ge_to_le(+a: Nat, +b: Nat, h: {Nat.is_ge(a, b) == True{} : Bool}) -> {Nat.is_le(b, a) == True{} : Bool}:
  %ikp_ge_le(a, b) : {_ == True{} : Bool}
  h

def ikp_le_to_ge(+a: Nat, +b: Nat, h: {Nat.is_le(b, a) == True{} : Bool}) -> {Nat.is_ge(a, b) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_ge(a, b), Nat.is_le(b, a), ikp_ge_le(a, b)) : {_ == True{} : Bool}
  h

def ikp_ge_not_lt(+a: Nat, +b: Nat, h: {Nat.is_ge(a, b) == True{} : Bool}) -> {False{} == Nat.is_lt(a, b) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      L.bool_absurd({False{} == Nat.is_lt(0n, 1n+bp) : Bool}, h)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      ikp_ge_not_lt(ap, bp, h)

def ikp_ge_lt_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_ge(a, b), Nat.is_lt(a, b)) == False{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      ikp_ge_lt_excl(ap, bp)

def ikp_le_lt_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_le(a, b), Nat.is_lt(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      ikp_le_lt_excl(ap, bp)

def ikp_lt_not_ge(+a: Nat, +b: Nat, h: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_ge(a, b) == False{} : Bool}:
  match a b:
    case 0n 0n:
      L.bool_absurd({Nat.is_ge(0n, 0n) == False{} : Bool}, h)
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      L.bool_absurd({Nat.is_ge(1n+ap, 0n) == False{} : Bool}, h)
    case 1n+ap 1n+bp:
      ikp_lt_not_ge(ap, bp, h)

def ikp_eq_refl(+a: Nat) -> {Nat.is_eq(a, a) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+ap:
      ikp_eq_refl(ap)

def ikp_lt_irrefl(+a: Nat) -> {Nat.is_lt(a, a) == False{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+ap:
      ikp_lt_irrefl(ap)

def ikp_eq_of(+a: Nat, +b: Nat, h: {Nat.is_eq(a, b) == True{} : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      L.bool_absurd({0n == 1n+bp : Nat}, h)
    case 1n+ap 0n:
      L.bool_absurd({1n+ap == 0n : Nat}, h)
    case 1n+ap 1n+bp:
      %ikp_eq_of(ap, bp, h) : {1n+ap == 1n+_ : Nat}
      {==}

def ikp_ne_of_eq(+a: Nat, -b: Nat, e: {a == b : Nat}) -> {Nat.is_ne(a, b) == False{} : Bool}:
  %e : {Nat.is_ne(a, _) == False{} : Bool}
  %Equal.sym(Bool, Nat.is_eq(a, a), True{}, ikp_eq_refl(a)) : {Bool.not(_) == False{} : Bool}
  {==}

def ikp_ne_of_is_eq(-a: Nat, -b: Nat, h: {Nat.is_eq(a, b) == True{} : Bool}) -> {Nat.is_ne(a, b) == False{} : Bool}:
  %Equal.sym(Bool, Nat.is_eq(a, b), True{}, h) : {Bool.not(_) == False{} : Bool}
  {==}

def ikp_le_double(+p: Nat) -> {Nat.is_le(p, Nat.mul(p, 2n)) == True{} : Bool}:
  match p:
    case 0n:
      {==}
    case 1n+q:
      L.nat_le_succ_r(q, Nat.mul(q, 2n), ikp_le_double(q))

def ikp_sub_le(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      L.nat_le_refl(1n+ap)
    case 1n+ap 1n+bp:
      L.nat_le_succ_r(Nat.sub(ap, bp), ap, ikp_sub_le(ap, bp))

def ikp_sub_sub(+a: Nat, +b: Nat, h: {Nat.is_le(b, a) == True{} : Bool}) -> {Nat.sub(a, Nat.sub(a, b)) == b : Nat}:
  %L.nat_sub_add_cancel(a, b, h) : {Nat.sub(_, Nat.sub(a, b)) == b : Nat}
  L.nat_add_sub_cancel_l(Nat.sub(a, b), b)

def ikp_add_one(+y: Nat) -> {Nat.add(y, 1n) == 1n+y : Nat}:
  match y:
    case 0n:
      {==}
    case 1n+yp:
      %ikp_add_one(yp) : {1n+Nat.add(yp, 1n) == 1n+_ : Nat}
      {==}

def ikp_add_le_cancel(+k: Nat, -x: Nat, -y: Nat, h: {Nat.is_le(Nat.add(k, x), Nat.add(k, y)) == True{} : Bool}) -> {Nat.is_le(x, y) == True{} : Bool}:
  match k:
    case 0n:
      h
    case 1n+kp:
      ikp_add_le_cancel(kp, x, y, h)

def ikp_add_lt_cancel(+k: Nat, -x: Nat, -y: Nat, h: {Nat.is_lt(Nat.add(k, x), Nat.add(k, y)) == True{} : Bool}) -> {Nat.is_lt(x, y) == True{} : Bool}:
  match k:
    case 0n:
      h
    case 1n+kp:
      ikp_add_lt_cancel(kp, x, y, h)

def ikp_add_lt_mono(+k: Nat, -x: Nat, -y: Nat, h: {Nat.is_lt(x, y) == True{} : Bool}) -> {Nat.is_lt(Nat.add(k, x), Nat.add(k, y)) == True{} : Bool}:
  match k:
    case 0n:
      h
    case 1n+kp:
      ikp_add_lt_mono(kp, x, y, h)

# k + e <= d  ->  e <= d - k
def ikp_le_sub(+k: Nat, +e: Nat, +d: Nat, h: {Nat.is_le(Nat.add(k, e), d) == True{} : Bool}) -> {Nat.is_le(e, Nat.sub(d, k)) == True{} : Bool}:
  match k d:
    case 0n _:
      %Equal.sym(Nat, Nat.sub(d, 0n), d, L.nat_sub_zero_r(d)) : {Nat.is_le(e, _) == True{} : Bool}
      h
    case 1n+kp 0n:
      L.bool_absurd({Nat.is_le(e, 0n) == True{} : Bool}, h)
    case 1n+kp 1n+dp:
      ikp_le_sub(kp, e, dp, h)

# m < le - ls  ->  ls + m < le
def ikp_lt_sub_add(+ls: Nat, +le: Nat, +m: Nat, h: {Nat.is_lt(m, Nat.sub(le, ls)) == True{} : Bool}) -> {Nat.is_lt(Nat.add(ls, m), le) == True{} : Bool}:
  match ls le:
    case 0n _:
      %L.nat_sub_zero_r(le) : {Nat.is_lt(m, _) == True{} : Bool}
      h
    case 1n+lp 0n:
      G.g_nat_lt_zero_absurd({Nat.is_lt(Nat.add(1n+lp, m), 0n) == True{} : Bool}, m, h)
    case 1n+lp 1n+ep:
      ikp_lt_sub_add(lp, ep, m, h)

# ls <= a < ls + M  ->  a - ls < M
def ikp_sub_lt(+ls: Nat, +a: Nat, +M: Nat, hge: {Nat.is_le(ls, a) == True{} : Bool}, hlt: {Nat.is_lt(a, Nat.add(ls, M)) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(a, ls), M) == True{} : Bool}:
  match ls a:
    case 0n _:
      %Equal.sym(Nat, Nat.sub(a, 0n), a, L.nat_sub_zero_r(a)) : {Nat.is_lt(_, M) == True{} : Bool}
      hlt
    case 1n+lp 0n:
      L.bool_absurd({Nat.is_lt(0n, M) == True{} : Bool}, hge)
    case 1n+lp 1n+ap:
      ikp_sub_lt(lp, ap, M, hge, hlt)

# k <= a < n + k  ->  a - k < n
def ikp_sub_lt_of_h(+a: Nat, +k: Nat, +n: Nat, hk: {Nat.is_le(k, a) == True{} : Bool}, h: {Nat.is_lt(a, Nat.add(n, k)) == True{} : Bool}) -> {Nat.is_lt(Nat.add(k, Nat.sub(a, k)), Nat.add(k, n)) == True{} : Bool}:
  %L.nat_add_comm(Nat.sub(a, k), k) : {Nat.is_lt(_, Nat.add(k, n)) == True{} : Bool}
  %Equal.sym(Nat, Nat.add(Nat.sub(a, k), k), a, L.nat_sub_add_cancel(a, k, hk)) : {Nat.is_lt(_, Nat.add(k, n)) == True{} : Bool}
  %L.nat_add_comm(n, k) : {Nat.is_lt(a, _) == True{} : Bool}
  h

def ikp_sub_lt_of(+a: Nat, +k: Nat, +n: Nat, hk: {Nat.is_le(k, a) == True{} : Bool}, h: {Nat.is_lt(a, Nat.add(n, k)) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(a, k), n) == True{} : Bool}:
  ikp_add_lt_cancel(k, Nat.sub(a, k), n, ikp_sub_lt_of_h(a, k, n, hk, h))

# ls <= le  ->  ls + (le - ls) == le
def ikp_span_end(+ls: Nat, +le: Nat, h: {Nat.is_le(ls, le) == True{} : Bool}) -> {Nat.add(ls, Nat.sub(le, ls)) == le : Nat}:
  Equal.trans(Nat, Nat.add(ls, Nat.sub(le, ls)), Nat.add(Nat.sub(le, ls), ls), le, L.nat_add_comm(ls, Nat.sub(le, ls)), L.nat_sub_add_cancel(le, ls, h))

# l < n  ->  l <= n - 1
def ikp_lt_le_pred(+l: Nat, +n: Nat, h: {Nat.is_lt(l, n) == True{} : Bool}) -> {Nat.is_le(l, Nat.sub(n, 1n)) == True{} : Bool}:
  match n:
    case 0n:
      G.g_nat_lt_zero_absurd({Nat.is_le(l, Nat.sub(0n, 1n)) == True{} : Bool}, l, h)
    case 1n+np:
      %Equal.sym(Nat, Nat.sub(np, 0n), np, L.nat_sub_zero_r(np)) : {Nat.is_le(l, _) == True{} : Bool}
      Equal.trans(Bool, Nat.is_le(l, np), Nat.is_lt(l, 1n+np), True{}, Equal.sym(Bool, Nat.is_lt(l, 1n+np), Nat.is_le(l, np), L.nat_lt_succ_eq_le(l, np)), h)

# Division by a positive divisor 1 + bp, one unit of the dividend at a time.

# go(j + n, j + m, d, r) == go(n, m, d, j + r)
def ikp_dm_skip(+j: Nat, -n: Nat, -m: Nat, -d: Nat, +r: Nat) -> {Nat.divmod.go(Nat.add(j, n), Nat.add(j, m), d, r) == Nat.divmod.go(n, m, d, Nat.add(j, r)) : Nat & Nat}:
  match j:
    case 0n:
      {==}
    case 1n+jp:
      %L.nat_add_succ_r(jp, r) : {Nat.divmod.go(Nat.add(1n+jp, n), Nat.add(1n+jp, m), d, r) == Nat.divmod.go(n, m, d, _) : Nat & Nat}
      ikp_dm_skip(jp, n, m, d, 1n+r)

# go(q (1 + bp), bp, d, 0) == (q + d, 0)
def ikp_dm_mul(+q: Nat, +bp: Nat, +d: Nat) -> {Nat.divmod.go(Nat.mul(q, 1n+bp), bp, d, 0n) == (Nat.add(q, d), 0n) : Nat & Nat}:
  match q:
    case 0n:
      {==}
    case 1n+qp:
      %L.nat_add_succ_r(bp, Nat.mul(qp, 1n+bp)) : {Nat.divmod.go(_, bp, d, 0n) == (Nat.add(1n+qp, d), 0n) : Nat & Nat}
      %L.nat_add_zero_r(bp) : {Nat.divmod.go(Nat.add(bp, 1n+Nat.mul(qp, 1n+bp)), _, d, 0n) == (Nat.add(1n+qp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(bp, 1n+Nat.mul(qp, 1n+bp)), Nat.add(bp, 0n), d, 0n), Nat.divmod.go(1n+Nat.mul(qp, 1n+bp), 0n, d, Nat.add(bp, 0n)), ikp_dm_skip(bp, 1n+Nat.mul(qp, 1n+bp), 0n, d, 0n)) : {_ == (Nat.add(1n+qp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat, Nat.add(bp, 0n), bp, L.nat_add_zero_r(bp)) : {Nat.divmod.go(Nat.mul(qp, 1n+bp), _, 1n+d, 0n) == (Nat.add(1n+qp, d), 0n) : Nat & Nat}
      %L.nat_add_succ_r(qp, d) : {Nat.divmod.go(Nat.mul(qp, 1n+bp), bp, 1n+d, 0n) == (_, 0n) : Nat & Nat}
      ikp_dm_mul(qp, bp, 1n+d)

def ikp_mod_mul(+q: Nat, +bp: Nat) -> {Nat.mod(Nat.mul(q, 1n+bp), 1n+bp) == 0n : Nat}:
  %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.mul(q, 1n+bp), bp, 0n, 0n), (Nat.add(q, 0n), 0n), ikp_dm_mul(q, bp, 0n)) : {Pair.snd(Nat, Nat, _) == 0n : Nat}
  {==}

def ikp_div_mul(+q: Nat, +bp: Nat) -> {Nat.div(Nat.mul(q, 1n+bp), 1n+bp) == q : Nat}:
  %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.mul(q, 1n+bp), bp, 0n, 0n), (Nat.add(q, 0n), 0n), ikp_dm_mul(q, bp, 0n)) : {Pair.fst(Nat, Nat, _) == q : Nat}
  L.nat_add_zero_r(q)

def ikp_eq_succ_add(+r: Nat, +mp: Nat) -> {False{} == Nat.is_eq(r, 1n+Nat.add(mp, r)) : Bool}:
  match r:
    case 0n:
      {==}
    case 1n+rp:
      %Equal.sym(Nat, Nat.add(mp, 1n+rp), 1n+Nat.add(mp, rp), L.nat_add_succ_r(mp, rp)) : {False{} == Nat.is_eq(1n+rp, 1n+_) : Bool}
      ikp_eq_succ_add(rp, mp)

# One more unit of the dividend: the quotient steps up when the remainder was the largest one,
# otherwise the remainder does (m + r == bp is the division's invariant).
def ikp_dm_step(+n: Nat, +m: Nat, +d: Nat, +r: Nat, +bp: Nat, e: {Nat.add(m, r) == bp : Nat}) -> {Nat.divmod.go(1n+n, m, d, r) == Bool.pick(Nat & Nat, Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(n, m, d, r)), bp), (1n+Pair.fst(Nat, Nat, Nat.divmod.go(n, m, d, r)), 0n), (Pair.fst(Nat, Nat, Nat.divmod.go(n, m, d, r)), 1n+Pair.snd(Nat, Nat, Nat.divmod.go(n, m, d, r)))) : Nat & Nat}:
  match n:
    case 0n:
      match m:
        case 0n:
          %e : {(1n+d, 0n) == Bool.pick(Nat & Nat, Nat.is_eq(r, _), (1n+d, 0n), (d, 1n+r)) : Nat & Nat}
          %Equal.sym(Bool, Nat.is_eq(r, r), True{}, ikp_eq_refl(r)) : {(1n+d, 0n) == Bool.pick(Nat & Nat, _, (1n+d, 0n), (d, 1n+r)) : Nat & Nat}
          {==}
        case 1n+mp:
          %e : {(d, 1n+r) == Bool.pick(Nat & Nat, Nat.is_eq(r, _), (1n+d, 0n), (d, 1n+r)) : Nat & Nat}
          %ikp_eq_succ_add(r, mp) : {(d, 1n+r) == Bool.pick(Nat & Nat, _, (1n+d, 0n), (d, 1n+r)) : Nat & Nat}
          {==}
    case 1n+np:
      match m:
        case 0n:
          ikp_dm_step(np, r, 1n+d, 0n, bp, Equal.trans(Nat, Nat.add(r, 0n), r, bp, L.nat_add_zero_r(r), e))
        case 1n+mp:
          ikp_dm_step(np, mp, d, 1n+r, bp, Equal.trans(Nat, Nat.add(mp, 1n+r), 1n+Nat.add(mp, r), bp, L.nat_add_succ_r(mp, r), e))

def ikp_div_succ_case(+q: Nat, +r: Nat, c: Bool) -> {Nat.is_le(q, Pair.fst(Nat, Nat, Bool.pick(Nat & Nat, c, (1n+q, 0n), (q, 1n+r)))) == True{} : Bool}:
  match c:
    case True{}:
      L.nat_le_succ_r(q, q, L.nat_le_refl(q))
    case False{}:
      L.nat_le_refl(q)

# x / b <= (1 + x) / b
def ikp_div_succ_ge(+x: Nat, +bp: Nat) -> {Nat.is_le(Nat.div(x, 1n+bp), Nat.div(1n+x, 1n+bp)) == True{} : Bool}:
  %Equal.sym(Nat & Nat, Nat.divmod.go(1n+x, bp, 0n, 0n), Bool.pick(Nat & Nat, Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(x, bp, 0n, 0n)), bp), (1n+Pair.fst(Nat, Nat, Nat.divmod.go(x, bp, 0n, 0n)), 0n), (Pair.fst(Nat, Nat, Nat.divmod.go(x, bp, 0n, 0n)), 1n+Pair.snd(Nat, Nat, Nat.divmod.go(x, bp, 0n, 0n)))), ikp_dm_step(x, bp, 0n, 0n, bp, L.nat_add_zero_r(bp))) : {Nat.is_le(Nat.div(x, 1n+bp), Pair.fst(Nat, Nat, _)) == True{} : Bool}
  ikp_div_succ_case(Pair.fst(Nat, Nat, Nat.divmod.go(x, bp, 0n, 0n)), Pair.snd(Nat, Nat, Nat.divmod.go(x, bp, 0n, 0n)), Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(x, bp, 0n, 0n)), bp))

# x / b <= (k + x) / b
def ikp_div_mono_add(+k: Nat, +x: Nat, +bp: Nat) -> {Nat.is_le(Nat.div(x, 1n+bp), Nat.div(Nat.add(k, x), 1n+bp)) == True{} : Bool}:
  match k:
    case 0n:
      L.nat_le_refl(Nat.div(x, 1n+bp))
    case 1n+kp:
      L.nat_le_trans(Nat.div(x, 1n+bp), Nat.div(Nat.add(kp, x), 1n+bp), Nat.div(1n+Nat.add(kp, x), 1n+bp), ikp_div_mono_add(kp, x, bp), ikp_div_succ_ge(Nat.add(kp, x), bp))

def ikp_div_mono(+x: Nat, +y: Nat, +bp: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(Nat.div(x, 1n+bp), Nat.div(y, 1n+bp)) == True{} : Bool}:
  %L.nat_sub_add_cancel(y, x, h) : {Nat.is_le(Nat.div(x, 1n+bp), Nat.div(_, 1n+bp)) == True{} : Bool}
  ikp_div_mono_add(Nat.sub(y, x), x, bp)

# ============================================================================
# 1. Layer applications
# ============================================================================

# Layers of the looped span.
def ik_span(loop_start: Nat, loop_end: Nat) -> Nat:
  Nat.sub(loop_end, loop_start)

# Layer applications of one token: every layer once, the looped span `passes` times.
def ik_apps(n_layers: Nat, loop_start: Nat, loop_end: Nat, +passes: Nat) -> Nat:
  Nat.add(n_layers, Nat.mul(Nat.sub(passes, 1n), ik_span(loop_start, loop_end)))

def ik_app_layer_in_loop(+a: Nat, +loop_start: Nat, +span: Nat, +passes: Nat, in_loop: Bool) -> Nat:
  match in_loop:
    case True{}:
      Nat.add(loop_start, Nat.mod(Nat.sub(a, loop_start), span))
    case False{}:
      Nat.sub(a, Nat.mul(Nat.sub(passes, 1n), span))

def ik_app_layer_if(+a: Nat, +loop_start: Nat, +span: Nat, +passes: Nat, before: Bool) -> Nat:
  match before:
    case True{}:
      a
    case False{}:
      ik_app_layer_in_loop(a, loop_start, span, passes, Nat.is_lt(a, Nat.add(loop_start, Nat.mul(passes, span))))

# The layer whose weights application `a` runs: the layers before the span, the span `passes` times,
# the layers after it.
def ik_app_layer(+a: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat) -> Nat:
  ik_app_layer_if(a, loop_start, ik_span(loop_start, loop_end), passes, Nat.is_lt(a, loop_start))

def ik_app_pass_if(+a: Nat, +loop_start: Nat, +span: Nat, +passes: Nat, in_loop: Bool) -> Nat:
  match in_loop:
    case True{}:
      Nat.div(Nat.sub(a, loop_start), span)
    case False{}:
      0n

# The pass over the looped span application `a` belongs to (0: the first pass or a layer outside the span).
def ik_app_pass(+a: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat) -> Nat:
  ik_app_pass_if(
    a,
    loop_start,
    ik_span(loop_start, loop_end),
    passes,
    Bool.and(
      Nat.is_ge(a, loop_start),
      Nat.is_lt(a, Nat.add(loop_start, Nat.mul(passes, ik_span(loop_start, loop_end))))
    )
  )

# ----------------------------------------------------------------------------
# 1a. Span roles of an application (decisions the Rust schedule reads)
# ----------------------------------------------------------------------------

def ik_eff_loop_if(+bound: Nat, +block: Nat, looped: Bool) -> Nat:
  match looped:
    case True{}:
      bound
    case False{}:
      block

# The looped span the schedule runs: the configured span for a model of several passes; for a
# model of one pass, an empty span behind the first block (the guard and the counts stay defined).
def ik_eff_loop_start(+passes: Nat, +loop_start: Nat, +block: Nat) -> Nat:
  ik_eff_loop_if(loop_start, block, Nat.is_gt(passes, 1n))

def ik_eff_loop_end(+passes: Nat, +loop_end: Nat, +block: Nat) -> Nat:
  ik_eff_loop_if(loop_end, block, Nat.is_gt(passes, 1n))

# The layer is part of the looped span (its weights run once per pass, it has message gates).
def ik_layer_looped(+layer: Nat, +loop_start: Nat, +loop_end: Nat) -> Bool:
  Bool.and(Nat.is_ge(layer, loop_start), Nat.is_lt(layer, loop_end))

# Position of a looped layer inside the span (the index of its message-gate tables).
def ik_loop_index(layer: Nat, loop_start: Nat) -> Nat:
  Nat.sub(layer, loop_start)

# Application `a` receives messages of the previous pass: it belongs to a later pass.
def ik_app_msgs(+a: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat) -> Bool:
  Nat.is_gt(ik_app_pass(a, loop_start, loop_end, passes), 0n)

# ============================================================================
# 2. Block states
# ============================================================================

# A layer opens a block when its index is a multiple of the block length: its input becomes a block state.
def ik_opens_block(layer: Nat, +block: Nat) -> Bool:
  Bool.and(Nat.is_gt(block, 0n), Nat.is_eq(Nat.mod(layer, block), 0n))

def ik_states_attn_if(+layer: Nat, +block: Nat, first: Bool) -> Nat:
  match first:
    case True{}:
      0n
    case False{}:
      Nat.add(Nat.div(Nat.sub(layer, 1n), block), 1n)

# Block states the attention-input mix of a layer reads (the first layer has no mix).
def ik_states_attn(+layer: Nat, +block: Nat) -> Nat:
  ik_states_attn_if(layer, block, Nat.is_eq(layer, 0n))

def ik_states_mlp_if(n: Nat, opens: Bool) -> Nat:
  match opens:
    case True{}:
      Nat.add(n, 1n)
    case False{}:
      n

# Block states the MLP-input mix of a layer reads: a layer that opens a block has added its input.
def ik_states_mlp(+layer: Nat, +block: Nat) -> Nat:
  ik_states_mlp_if(ik_states_attn(layer, block), ik_opens_block(layer, block))

# Block states the final mix (behind the last layer) reads.
def ik_states_final(+n_layers: Nat, +block: Nat) -> Nat:
  ik_states_mlp(Nat.sub(n_layers, 1n), block)

# Block states a message of the previous pass is mixed from: the blocks the span opened, less its first.
def ik_msg_states(loop_start: Nat, loop_end: Nat, block: Nat) -> Nat:
  Nat.sub(Nat.div(ik_span(loop_start, loop_end), block), 1n)

# The geometry the kernels cover: at most `max_states` block states in any mix, a span that starts and
# ends on block boundaries behind the first layer, and at least one pass.
def ik_schedule_valid(+n_layers: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat, +block: Nat, +max_states: Nat) -> Bool:
  Bool.and(
    Bool.and(Nat.is_gt(n_layers, 0n), Nat.is_gt(block, 0n)),
    Bool.and(
      Bool.and(Nat.is_ge(passes, 1n), Nat.is_le(ik_states_mlp(Nat.sub(n_layers, 1n), block), max_states)),
      Bool.and(
        Bool.and(Nat.is_gt(loop_start, 0n), Nat.is_le(loop_start, loop_end)),
        Bool.and(
          Nat.is_le(loop_end, n_layers),
          Bool.and(
            Bool.and(Nat.is_eq(Nat.mod(loop_start, block), 0n), Nat.is_eq(Nat.mod(loop_end, block), 0n)),
            Nat.is_le(Nat.add(ik_msg_states(loop_start, loop_end, block), 1n), max_states)
          )
        )
      )
    )
  )
# ----------------------------------------------------------------------------
# 2a. Laws of the schedule (after section 2: they read the schedule guard)
# ----------------------------------------------------------------------------

# Parts of an accepted schedule.
def ikp_v_rest1(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, h: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Bool.and(Bool.and(Nat.is_ge(p, 1n), Nat.is_le(ik_states_mlp(Nat.sub(n, 1n), b), mx)), Bool.and(Bool.and(Nat.is_gt(ls, 0n), Nat.is_le(ls, le)), Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx))))) == True{} : Bool}:
  ikp_and_r(Bool.and(Nat.is_gt(n, 0n), Nat.is_gt(b, 0n)), Bool.and(Bool.and(Nat.is_ge(p, 1n), Nat.is_le(ik_states_mlp(Nat.sub(n, 1n), b), mx)), Bool.and(Bool.and(Nat.is_gt(ls, 0n), Nat.is_le(ls, le)), Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx))))), h)

def ikp_v_rest2(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, h: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Bool.and(Bool.and(Nat.is_gt(ls, 0n), Nat.is_le(ls, le)), Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx)))) == True{} : Bool}:
  ikp_and_r(Bool.and(Nat.is_ge(p, 1n), Nat.is_le(ik_states_mlp(Nat.sub(n, 1n), b), mx)), Bool.and(Bool.and(Nat.is_gt(ls, 0n), Nat.is_le(ls, le)), Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx)))), ikp_v_rest1(n, ls, le, p, b, mx, h))

def ikp_v_rest3(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, h: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx))) == True{} : Bool}:
  ikp_and_r(Bool.and(Nat.is_gt(ls, 0n), Nat.is_le(ls, le)), Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx))), ikp_v_rest2(n, ls, le, p, b, mx, h))

def ikp_v_block(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, h: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Nat.is_gt(b, 0n) == True{} : Bool}:
  ikp_and_r(Nat.is_gt(n, 0n), Nat.is_gt(b, 0n), ikp_and_l(Bool.and(Nat.is_gt(n, 0n), Nat.is_gt(b, 0n)), Bool.and(Bool.and(Nat.is_ge(p, 1n), Nat.is_le(ik_states_mlp(Nat.sub(n, 1n), b), mx)), Bool.and(Bool.and(Nat.is_gt(ls, 0n), Nat.is_le(ls, le)), Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx))))), h))

def ikp_v_mlp(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, h: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Nat.is_le(ik_states_mlp(Nat.sub(n, 1n), b), mx) == True{} : Bool}:
  ikp_and_r(Nat.is_ge(p, 1n), Nat.is_le(ik_states_mlp(Nat.sub(n, 1n), b), mx), ikp_and_l(Bool.and(Nat.is_ge(p, 1n), Nat.is_le(ik_states_mlp(Nat.sub(n, 1n), b), mx)), Bool.and(Bool.and(Nat.is_gt(ls, 0n), Nat.is_le(ls, le)), Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx)))), ikp_v_rest1(n, ls, le, p, b, mx, h)))

def ikp_v_lsle(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, h: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Nat.is_le(ls, le) == True{} : Bool}:
  ikp_and_r(Nat.is_gt(ls, 0n), Nat.is_le(ls, le), ikp_and_l(Bool.and(Nat.is_gt(ls, 0n), Nat.is_le(ls, le)), Bool.and(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx))), ikp_v_rest2(n, ls, le, p, b, mx, h)))

def ikp_v_len(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, h: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Nat.is_le(le, n) == True{} : Bool}:
  ikp_and_l(Nat.is_le(le, n), Bool.and(Bool.and(Nat.is_eq(Nat.mod(ls, b), 0n), Nat.is_eq(Nat.mod(le, b), 0n)), Nat.is_le(Nat.add(ik_msg_states(ls, le, b), 1n), mx)), ikp_v_rest3(n, ls, le, p, b, mx, h))

# the span of an accepted schedule ends inside the model
def ikp_v_hsn(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, +h: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Nat.is_le(Nat.add(ls, Nat.sub(le, ls)), n) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(ls, Nat.sub(le, ls)), le, ikp_span_end(ls, le, ikp_v_lsle(n, ls, le, p, b, mx, h))) : {Nat.is_le(_, n) == True{} : Bool}
  ikp_v_len(n, ls, le, p, b, mx, h)

# The layer an application runs is below n when the span ends inside the model and a is an
# application of the schedule.
def ikp_layer_after(+a: Nat, +ls: Nat, +span: Nat, +p: Nat, +n: Nat, hlt: {Nat.is_lt(a, Nat.add(n, Nat.mul(Nat.sub(p, 1n), span))) == True{} : Bool}, hil: {Nat.is_lt(a, Nat.add(ls, Nat.mul(p, span))) == False{} : Bool}) -> {Nat.is_lt(Nat.sub(a, Nat.mul(Nat.sub(p, 1n), span)), n) == True{} : Bool}:
  ikp_sub_lt_of(a, Nat.mul(Nat.sub(p, 1n), span), n, L.nat_le_trans(Nat.mul(Nat.sub(p, 1n), span), Nat.add(ls, Nat.mul(p, span)), a, L.nat_le_trans(Nat.mul(Nat.sub(p, 1n), span), Nat.mul(p, span), Nat.add(ls, Nat.mul(p, span)), L.nat_mul_le_mono_r(Nat.sub(p, 1n), p, span, ikp_sub_le(p, 1n)), L.nat_add_le_l(ls, Nat.mul(p, span))), L.nat_not_lt_imp_le(a, Nat.add(ls, Nat.mul(p, span)), hil)), hlt)

def ikp_add_mul0(+ls: Nat, +p: Nat) -> {Nat.add(ls, Nat.mul(p, 0n)) == ls : Nat}:
  %Equal.sym(Nat, Nat.mul(p, 0n), 0n, L.nat_mul_zero_r(p)) : {Nat.add(ls, _) == ls : Nat}
  L.nat_add_zero_r(ls)

def ikp_hil0(+a: Nat, +ls: Nat, +p: Nat, hil: {Nat.is_lt(a, Nat.add(ls, Nat.mul(p, 0n))) == True{} : Bool}) -> {True{} == Nat.is_lt(a, ls) : Bool}:
  %ikp_add_mul0(ls, p) : {True{} == Nat.is_lt(a, _) : Bool}
  Equal.sym(Bool, Nat.is_lt(a, Nat.add(ls, Nat.mul(p, 0n))), True{}, hil)

def ikp_layer_lt_in(+a: Nat, +ls: Nat, +span: Nat, +p: Nat, +n: Nat, hsn: {Nat.is_le(Nat.add(ls, span), n) == True{} : Bool}, hlt: {Nat.is_lt(a, Nat.add(n, Nat.mul(Nat.sub(p, 1n), span))) == True{} : Bool}, hbf: {Nat.is_lt(a, ls) == False{} : Bool}, il: Bool, hil: {Nat.is_lt(a, Nat.add(ls, Nat.mul(p, span))) == il : Bool}) -> {Nat.is_lt(ik_app_layer_in_loop(a, ls, span, p, il), n) == True{} : Bool}:
  match span:
    case 0n:
      match il:
        case True{}:
          L.bool_absurd_tf({Nat.is_lt(ik_app_layer_in_loop(a, ls, 0n, p, True{}), n) == True{} : Bool}, Equal.trans(Bool, True{}, Nat.is_lt(a, ls), False{}, ikp_hil0(a, ls, p, hil), hbf))
        case False{}:
          ikp_layer_after(a, ls, 0n, p, n, hlt, hil)
    case 1n+sp:
      match il:
        case True{}:
          L.nat_lt_le_trans(Nat.add(ls, Nat.mod(Nat.sub(a, ls), 1n+sp)), Nat.add(ls, 1n+sp), n, ikp_add_lt_mono(ls, Nat.mod(Nat.sub(a, ls), 1n+sp), 1n+sp, L.nat_mod_lt(Nat.sub(a, ls), sp)), hsn)
        case False{}:
          ikp_layer_after(a, ls, 1n+sp, p, n, hlt, hil)

def ikp_layer_lt(+a: Nat, +ls: Nat, +span: Nat, +p: Nat, +n: Nat, hsn: {Nat.is_le(Nat.add(ls, span), n) == True{} : Bool}, hlt: {Nat.is_lt(a, Nat.add(n, Nat.mul(Nat.sub(p, 1n), span))) == True{} : Bool}, bf: Bool, hbf: {Nat.is_lt(a, ls) == bf : Bool}) -> {Nat.is_lt(ik_app_layer_if(a, ls, span, p, bf), n) == True{} : Bool}:
  match bf:
    case True{}:
      L.nat_lt_le_trans(a, ls, n, hbf, L.nat_le_trans(ls, Nat.add(ls, span), n, L.nat_add_le_r(ls, span), hsn))
    case False{}:
      ikp_layer_lt_in(a, ls, span, p, n, hsn, hlt, hbf, Nat.is_lt(a, Nat.add(ls, Nat.mul(p, span))), {==})

def ikp_s1_c(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +a: Nat, hsn: {Nat.is_le(Nat.add(ls, Nat.sub(le, ls)), n) == True{} : Bool}, c: Bool, hc: {Nat.is_lt(a, ik_apps(n, ls, le, p)) == c : Bool}) -> {Bool.and(c, Nat.is_le(n, ik_app_layer(a, ls, le, p))) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      L.nat_lt_le_false(ik_app_layer(a, ls, le, p), n, ikp_layer_lt(a, ls, Nat.sub(le, ls), p, n, hsn, hc, Nat.is_lt(a, ls), {==}))

def ikp_s1(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, +a: Nat, v: Bool, hv: {ik_schedule_valid(n, ls, le, p, b, mx) == v : Bool}) -> {Bool.and(v, Bool.and(Nat.is_lt(a, ik_apps(n, ls, le, p)), Nat.is_le(n, ik_app_layer(a, ls, le, p)))) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      ikp_s1_c(n, ls, le, p, a, ikp_v_hsn(n, ls, le, p, b, mx, hv), Nat.is_lt(a, ik_apps(n, ls, le, p)), {==})

# LAW IK.1
# promise: on a schedule the guard accepts, every layer application runs the weights of a layer the model has.
law ik_app_layer_in_model:
  for +n: Nat
  for +ls: Nat
  for +le: Nat
  for +p: Nat
  for +b: Nat
  for +mx: Nat
  for +a: Nat
  {Bool.and(ik_schedule_valid(n, ls, le, p, b, mx), Bool.and(Nat.is_lt(a, ik_apps(n, ls, le, p)), Nat.is_le(n, ik_app_layer(a, ls, le, p)))) == False{} : Bool}

def ik_app_layer_in_model(n, ls, le, p, b, mx, a):
  ikp_s1(n, ls, le, p, b, mx, a, ik_schedule_valid(n, ls, le, p, b, mx), {==})

def ikp_pass_lt(+a: Nat, +ls: Nat, +span: Nat, +pp: Nat, b: Bool, +hb: {Bool.and(Nat.is_ge(a, ls), Nat.is_lt(a, Nat.add(ls, Nat.mul(1n+pp, span)))) == b : Bool}) -> {Nat.is_lt(ik_app_pass_if(a, ls, span, 1n+pp, b), 1n+pp) == True{} : Bool}:
  match span:
    case 0n:
      match b:
        case False{}:
          {==}
        case True{}:
          {==}
    case 1n+sp:
      match b:
        case False{}:
          {==}
        case True{}:
          L.nat_div_lt_of_lt_mul(Nat.sub(a, ls), 1n+pp, sp, ikp_sub_lt(ls, a, Nat.mul(1n+pp, 1n+sp), ikp_ge_to_le(a, ls, ikp_and_l(Nat.is_ge(a, ls), Nat.is_lt(a, Nat.add(ls, Nat.mul(1n+pp, 1n+sp))), hb)), ikp_and_r(Nat.is_ge(a, ls), Nat.is_lt(a, Nat.add(ls, Nat.mul(1n+pp, 1n+sp))), hb)))

# LAW IK.2
# promise: every layer application belongs to one of the passes the model runs (its pass is below the pass count).
law ik_app_pass_below_passes:
  for +a: Nat
  for +ls: Nat
  for +le: Nat
  for +pp: Nat
  {Nat.is_lt(ik_app_pass(a, ls, le, 1n+pp), 1n+pp) == True{} : Bool}

def ik_app_pass_below_passes(a, ls, le, pp):
  ikp_pass_lt(a, ls, ik_span(ls, le), pp, Bool.and(Nat.is_ge(a, ls), Nat.is_lt(a, Nat.add(ls, Nat.mul(1n+pp, ik_span(ls, le))))), {==})

def ikp_mod_lt_span(+x: Nat, +ls: Nat, +le: Nat, +sp: Nat, hs: {Nat.sub(le, ls) == 1n+sp : Nat}) -> {Nat.is_lt(Nat.mod(x, 1n+sp), Nat.sub(le, ls)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.sub(le, ls), 1n+sp, hs) : {Nat.is_lt(Nat.mod(x, 1n+sp), _) == True{} : Bool}
  L.nat_mod_lt(x, sp)

def ikp_looped_in(+ls: Nat, +le: Nat, +m: Nat, h: {Nat.is_lt(m, Nat.sub(le, ls)) == True{} : Bool}) -> {ik_layer_looped(Nat.add(ls, m), ls, le) == True{} : Bool}:
  ikp_and_tt(Nat.is_ge(Nat.add(ls, m), ls), Nat.is_lt(Nat.add(ls, m), le), ikp_le_to_ge(Nat.add(ls, m), ls, L.nat_add_le_r(ls, m)), ikp_lt_sub_add(ls, le, m, h))

def ikp_s2(+a: Nat, +ls: Nat, +le: Nat, +span: Nat, +p: Nat, hs: {Nat.sub(le, ls) == span : Nat}, b: Bool, +hb: {Bool.and(Nat.is_ge(a, ls), Nat.is_lt(a, Nat.add(ls, Nat.mul(p, span)))) == b : Bool}) -> {Bool.and(Nat.is_gt(ik_app_pass_if(a, ls, span, p, b), 0n), Bool.not(ik_layer_looped(ik_app_layer_if(a, ls, span, p, Nat.is_lt(a, ls)), ls, le))) == False{} : Bool}:
  match span:
    case 0n:
      match b:
        case False{}:
          {==}
        case True{}:
          {==}
    case 1n+sp:
      match b:
        case False{}:
          {==}
        case True{}:
          %ikp_ge_not_lt(a, ls, ikp_and_l(Nat.is_ge(a, ls), Nat.is_lt(a, Nat.add(ls, Nat.mul(p, 1n+sp))), hb)) : {Bool.and(Nat.is_gt(Nat.div(Nat.sub(a, ls), 1n+sp), 0n), Bool.not(ik_layer_looped(ik_app_layer_if(a, ls, 1n+sp, p, _), ls, le))) == False{} : Bool}
          %Equal.sym(Bool, Nat.is_lt(a, Nat.add(ls, Nat.mul(p, 1n+sp))), True{}, ikp_and_r(Nat.is_ge(a, ls), Nat.is_lt(a, Nat.add(ls, Nat.mul(p, 1n+sp))), hb)) : {Bool.and(Nat.is_gt(Nat.div(Nat.sub(a, ls), 1n+sp), 0n), Bool.not(ik_layer_looped(ik_app_layer_in_loop(a, ls, 1n+sp, p, _), ls, le))) == False{} : Bool}
          ikp_and_fr(Nat.is_gt(Nat.div(Nat.sub(a, ls), 1n+sp), 0n), Bool.not(ik_layer_looped(Nat.add(ls, Nat.mod(Nat.sub(a, ls), 1n+sp)), ls, le)), ikp_not_t(ik_layer_looped(Nat.add(ls, Nat.mod(Nat.sub(a, ls), 1n+sp)), ls, le), ikp_looped_in(ls, le, Nat.mod(Nat.sub(a, ls), 1n+sp), ikp_mod_lt_span(Nat.sub(a, ls), ls, le, sp, hs))))

# LAW IK.3
# promise: an application that receives messages of a previous pass runs a layer of the looped span.
law ik_app_msgs_looped:
  for +a: Nat
  for +ls: Nat
  for +le: Nat
  for +p: Nat
  {Bool.and(ik_app_msgs(a, ls, le, p), Bool.not(ik_layer_looped(ik_app_layer(a, ls, le, p), ls, le))) == False{} : Bool}

def ik_app_msgs_looped(a, ls, le, p):
  ikp_s2(a, ls, le, ik_span(ls, le), p, {==}, Bool.and(Nat.is_ge(a, ls), Nat.is_lt(a, Nat.add(ls, Nat.mul(p, ik_span(ls, le))))), {==})

def ikp_s4_lt(+layer: Nat, +ls: Nat, +le: Nat, hge: {Nat.is_le(ls, layer) == True{} : Bool}, +hlt: {Nat.is_lt(layer, le) == True{} : Bool}) -> {Nat.is_lt(layer, Nat.add(ls, Nat.sub(le, ls))) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(ls, Nat.sub(le, ls)), le, ikp_span_end(ls, le, L.nat_le_trans(ls, layer, le, hge, L.nat_lt_imp_le(layer, le, hlt)))) : {Nat.is_lt(layer, _) == True{} : Bool}
  hlt

def ikp_s4(+layer: Nat, +ls: Nat, +le: Nat, v: Bool, +hv: {ik_layer_looped(layer, ls, le) == v : Bool}) -> {Bool.and(v, Nat.is_le(ik_span(ls, le), ik_loop_index(layer, ls))) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      L.nat_lt_le_false(Nat.sub(layer, ls), Nat.sub(le, ls), ikp_sub_lt(ls, layer, Nat.sub(le, ls), ikp_ge_to_le(layer, ls, ikp_and_l(Nat.is_ge(layer, ls), Nat.is_lt(layer, le), hv)), ikp_s4_lt(layer, ls, le, ikp_ge_to_le(layer, ls, ikp_and_l(Nat.is_ge(layer, ls), Nat.is_lt(layer, le), hv)), ikp_and_r(Nat.is_ge(layer, ls), Nat.is_lt(layer, le), hv))))

# LAW IK.4
# promise: a looped layer's position inside the span is below the span's length (its message-gate tables have a row for it).
law ik_loop_index_in_span:
  for +layer: Nat
  for +ls: Nat
  for +le: Nat
  {Bool.and(ik_layer_looped(layer, ls, le), Nat.is_le(ik_span(ls, le), ik_loop_index(layer, ls))) == False{} : Bool}

def ik_loop_index_in_span(layer, ls, le):
  ikp_s4(layer, ls, le, ik_layer_looped(layer, ls, le), {==})

def ikp_msg_empty(+b: Nat) -> {Nat.is_ne(ik_msg_states(b, b, b), 0n) == False{} : Bool}:
  match b:
    case 0n:
      {==}
    case 1n+bp:
      %Equal.sym(Nat, Nat.sub(bp, bp), 0n, L.nat_sub_self(bp)) : {Nat.is_ne(Nat.sub(Nat.div(_, 1n+bp), 1n), 0n) == False{} : Bool}
      {==}

def ikp_s5(+p: Nat, +ls: Nat, +le: Nat, +b: Nat) -> {Bool.and(Nat.is_le(p, 1n), Nat.is_ne(ik_msg_states(ik_eff_loop_start(p, ls, b), ik_eff_loop_end(p, le, b), b), 0n)) == False{} : Bool}:
  match p:
    case 0n:
      ikp_msg_empty(b)
    case 1n+pp:
      match pp:
        case 0n:
          ikp_msg_empty(b)
        case 1n+q:
          {==}

# LAW IK.5
# promise: a model of one pass has no messages (its schedule's span is empty).
law ik_one_pass_no_messages:
  for +p: Nat
  for +ls: Nat
  for +le: Nat
  for +b: Nat
  {Bool.and(Nat.is_le(p, 1n), Nat.is_ne(ik_msg_states(ik_eff_loop_start(p, ls, b), ik_eff_loop_end(p, le, b), b), 0n)) == False{} : Bool}

def ik_one_pass_no_messages(p, ls, le, b):
  ikp_s5(p, ls, le, b)

# ----------------------------------------------------------------------------
# 2b. Laws of the block states
# ----------------------------------------------------------------------------

def ikp_mlp_split(+n: Nat, o: Bool) -> {ik_states_mlp_if(n, o) == Nat.add(n, Bool.pick(Nat, o, 1n, 0n)) : Nat}:
  match o:
    case True{}:
      {==}
    case False{}:
      Equal.sym(Nat, Nat.add(n, 0n), n, L.nat_add_zero_r(n))

# LAW IK.6a
# promise: a layer's MLP-input mix reads every block state its attention-input mix reads.
# harm: a state the attention mix used would be missing from the MLP mix of the same layer
law ik_states_mlp_covers_attn:
  for +l: Nat
  for +b: Nat
  {Nat.is_le(ik_states_attn(l, b), ik_states_mlp(l, b)) == True{} : Bool}

def ik_states_mlp_covers_attn(l, b):
  %Equal.sym(Nat, ik_states_mlp(l, b), Nat.add(ik_states_attn(l, b), Bool.pick(Nat, ik_opens_block(l, b), 1n, 0n)), ikp_mlp_split(ik_states_attn(l, b), ik_opens_block(l, b))) : {Nat.is_le(ik_states_attn(l, b), _) == True{} : Bool}
  L.nat_add_le_r(ik_states_attn(l, b), Bool.pick(Nat, ik_opens_block(l, b), 1n, 0n))

# LAW IK.6b
# promise: a layer's MLP-input mix reads at most one block state more than its attention-input mix, and that one only when the layer opens a block.
# harm: a mix would read a block state no layer has added yet
law ik_states_mlp_within_opened:
  for +l: Nat
  for +b: Nat
  {Nat.is_le(ik_states_mlp(l, b), Nat.add(ik_states_attn(l, b), Bool.pick(Nat, ik_opens_block(l, b), 1n, 0n))) == True{} : Bool}

def ik_states_mlp_within_opened(l, b):
  %Equal.sym(Nat, ik_states_mlp(l, b), Nat.add(ik_states_attn(l, b), Bool.pick(Nat, ik_opens_block(l, b), 1n, 0n)), ikp_mlp_split(ik_states_attn(l, b), ik_opens_block(l, b))) : {Nat.is_le(_, Nat.add(ik_states_attn(l, b), Bool.pick(Nat, ik_opens_block(l, b), 1n, 0n))) == True{} : Bool}
  L.nat_le_refl(Nat.add(ik_states_attn(l, b), Bool.pick(Nat, ik_opens_block(l, b), 1n, 0n)))

def ikp_mlp_case(+q: Nat, +r: Nat, c: Bool) -> {ik_states_mlp_if(Nat.add(q, 1n), Nat.is_eq(Pair.snd(Nat, Nat, Bool.pick(Nat & Nat, c, (1n+q, 0n), (q, 1n+r))), 0n)) == Nat.add(Pair.fst(Nat, Nat, Bool.pick(Nat & Nat, c, (1n+q, 0n), (q, 1n+r))), 1n) : Nat}:
  match c:
    case True{}:
      ikp_add_one(Nat.add(q, 1n))
    case False{}:
      {==}

# With blocks of 1 + bp layers, the MLP mix of layer l reads l / (1 + bp) + 1 block states.
def ikp_mlp_closed(+l: Nat, +bp: Nat) -> {ik_states_mlp(l, 1n+bp) == Nat.add(Nat.div(l, 1n+bp), 1n) : Nat}:
  match l:
    case 0n:
      {==}
    case 1n+lp:
      %Equal.sym(Nat & Nat, Nat.divmod.go(1n+lp, bp, 0n, 0n), Bool.pick(Nat & Nat, Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), bp), (1n+Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 0n), (Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 1n+Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)))), ikp_dm_step(lp, bp, 0n, 0n, bp, L.nat_add_zero_r(bp))) : {ik_states_mlp_if(Nat.add(Nat.div(Nat.sub(lp, 0n), 1n+bp), 1n), Nat.is_eq(Pair.snd(Nat, Nat, _), 0n)) == Nat.add(Pair.fst(Nat, Nat, Nat.divmod.go(1n+lp, bp, 0n, 0n)), 1n) : Nat}
      %Equal.sym(Nat & Nat, Nat.divmod.go(1n+lp, bp, 0n, 0n), Bool.pick(Nat & Nat, Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), bp), (1n+Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 0n), (Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 1n+Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)))), ikp_dm_step(lp, bp, 0n, 0n, bp, L.nat_add_zero_r(bp))) : {ik_states_mlp_if(Nat.add(Nat.div(Nat.sub(lp, 0n), 1n+bp), 1n), Nat.is_eq(Pair.snd(Nat, Nat, Bool.pick(Nat & Nat, Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), bp), (1n+Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 0n), (Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 1n+Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n))))), 0n)) == Nat.add(Pair.fst(Nat, Nat, _), 1n) : Nat}
      %Equal.sym(Nat, Nat.sub(lp, 0n), lp, L.nat_sub_zero_r(lp)) : {ik_states_mlp_if(Nat.add(Nat.div(_, 1n+bp), 1n), Nat.is_eq(Pair.snd(Nat, Nat, Bool.pick(Nat & Nat, Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), bp), (1n+Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 0n), (Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 1n+Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n))))), 0n)) == Nat.add(Pair.fst(Nat, Nat, Bool.pick(Nat & Nat, Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), bp), (1n+Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 0n), (Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), 1n+Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n))))), 1n) : Nat}
      ikp_mlp_case(Pair.fst(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), Nat.is_eq(Pair.snd(Nat, Nat, Nat.divmod.go(lp, bp, 0n, 0n)), bp))

# LAW IK.7
# promise: the block states a layer's attention-input mix reads are exactly the ones the previous layer's MLP-input mix read.
law ik_states_carry:
  for +l: Nat
  for +bp: Nat
  {ik_states_attn(1n+l, 1n+bp) == ik_states_mlp(l, 1n+bp) : Nat}

def ik_states_carry(l, bp):
  %Equal.sym(Nat, ik_states_mlp(l, 1n+bp), Nat.add(Nat.div(l, 1n+bp), 1n), ikp_mlp_closed(l, bp)) : {ik_states_attn(1n+l, 1n+bp) == _ : Nat}
  %Equal.sym(Nat, Nat.sub(l, 0n), l, L.nat_sub_zero_r(l)) : {Nat.add(Nat.div(_, 1n+bp), 1n) == Nat.add(Nat.div(l, 1n+bp), 1n) : Nat}
  {==}

def ikp_mlp_mono(+x: Nat, +y: Nat, +bp: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(ik_states_mlp(x, 1n+bp), ik_states_mlp(y, 1n+bp)) == True{} : Bool}:
  %Equal.sym(Nat, ik_states_mlp(x, 1n+bp), Nat.add(Nat.div(x, 1n+bp), 1n), ikp_mlp_closed(x, bp)) : {Nat.is_le(_, ik_states_mlp(y, 1n+bp)) == True{} : Bool}
  %Equal.sym(Nat, ik_states_mlp(y, 1n+bp), Nat.add(Nat.div(y, 1n+bp), 1n), ikp_mlp_closed(y, bp)) : {Nat.is_le(Nat.add(Nat.div(x, 1n+bp), 1n), _) == True{} : Bool}
  L.nat_add_le_mono(Nat.div(x, 1n+bp), Nat.div(y, 1n+bp), 1n, 1n, ikp_div_mono(x, y, bp, h), L.nat_le_refl(1n))

def ikp_s6_c(+n: Nat, +bp: Nat, +mx: Nat, +l: Nat, hm: {Nat.is_le(ik_states_mlp(Nat.sub(n, 1n), 1n+bp), mx) == True{} : Bool}, c: Bool, hc: {Nat.is_lt(l, n) == c : Bool}) -> {Bool.and(c, Nat.is_lt(mx, ik_states_mlp(l, 1n+bp))) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(mx, ik_states_mlp(l, 1n+bp), L.nat_le_trans(ik_states_mlp(l, 1n+bp), ik_states_mlp(Nat.sub(n, 1n), 1n+bp), mx, ikp_mlp_mono(l, Nat.sub(n, 1n), bp, ikp_lt_le_pred(l, n, hc)), hm))

def ikp_s6_b(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, +l: Nat, +hv: {ik_schedule_valid(n, ls, le, p, b, mx) == True{} : Bool}) -> {Bool.and(Nat.is_lt(l, n), Nat.is_lt(mx, ik_states_mlp(l, b))) == False{} : Bool}:
  match b:
    case 0n:
      L.bool_absurd({Bool.and(Nat.is_lt(l, n), Nat.is_lt(mx, ik_states_mlp(l, 0n))) == False{} : Bool}, ikp_v_block(n, ls, le, p, 0n, mx, hv))
    case 1n+bp:
      ikp_s6_c(n, bp, mx, l, ikp_v_mlp(n, ls, le, p, 1n+bp, mx, hv), Nat.is_lt(l, n), {==})

def ikp_s6(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +b: Nat, +mx: Nat, +l: Nat, v: Bool, hv: {ik_schedule_valid(n, ls, le, p, b, mx) == v : Bool}) -> {Bool.and(v, Bool.and(Nat.is_lt(l, n), Nat.is_lt(mx, ik_states_mlp(l, b)))) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      ikp_s6_b(n, ls, le, p, b, mx, l, hv)

# LAW IK.8
# promise: on a schedule the guard accepts, no layer's residual mix reads more block states than the kernels hold.
law ik_states_within_max:
  for +n: Nat
  for +ls: Nat
  for +le: Nat
  for +p: Nat
  for +b: Nat
  for +mx: Nat
  for +l: Nat
  {Bool.and(ik_schedule_valid(n, ls, le, p, b, mx), Bool.and(Nat.is_lt(l, n), Nat.is_lt(mx, ik_states_mlp(l, b)))) == False{} : Bool}

def ik_states_within_max(n, ls, le, p, b, mx, l):
  ikp_s6(n, ls, le, p, b, mx, l, ik_schedule_valid(n, ls, le, p, b, mx), {==})

# ============================================================================
# 3. Rings
# ============================================================================

def ik_pow2_stop(done: Bool, p: Nat) -> Nat:
  match done:
    case True{}:
      p
    case False{}:
      0n

def ik_pow2_from(fuel: Nat, done: Bool, +p: Nat, +need: Nat) -> Nat:
  match fuel:
    case 0n:
      ik_pow2_stop(done, p)
    case 1n+f:
      match done:
        case True{}:
          p
        case False{}:
          ik_pow2_from(f, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need)

# The smallest power of two of at least 128 that holds `need` rows; 0 when `need` is beyond 2^47 rows.
def ik_pow2_rows(+need: Nat) -> Nat:
  ik_pow2_from(40n, Nat.is_ge(128n, need), 128n, need)

# Slots of a sliding layer's K / V ring: the window beside the widest pass (a row must not see the rows
# the pass wrote ahead of it) and beside one vector of wrapped slots.
def ik_ring_need(window: Nat, rows: Nat) -> Nat:
  Nat.add(window, Nat.max(Nat.sub(rows, 1n), 64n))

def ik_ring_slots(window: Nat, rows: Nat) -> Nat:
  ik_pow2_rows(ik_ring_need(window, rows))

# Rows of a short convolution's history ring: the rows of the widest pass and the taps before its first row.
def ik_hist_rows(rows: Nat, taps: Nat) -> Nat:
  ik_pow2_rows(Nat.sub(Nat.add(rows, taps), 1n))

# A K / V ring the softmax kernel can stage as one chunk of at most `max_slots` keys: the ring
# itself, or 0 when it needs more slots (or none holds the pass).
def ik_ring_within_if(+slots: Nat, fits: Bool) -> Nat:
  match fits:
    case True{}:
      slots
    case False{}:
      0n

def ik_ring_within(+slots: Nat, +max_slots: Nat) -> Nat:
  ik_ring_within_if(slots, Nat.is_le(slots, max_slots))

def ik_ring_slots_within(window: Nat, rows: Nat, +max_slots: Nat) -> Nat:
  ik_ring_within(ik_ring_slots(window, rows), max_slots)

# A power of two: `x` halves down to 1 within `fuel` steps.
def ik_is_pow2_go(fuel: Nat, one: Bool, +x: Nat) -> Bool:
  match fuel:
    case 0n:
      one
    case 1n+f:
      match one:
        case True{}:
          True{}
        case False{}:
          Bool.and(Nat.is_eq(Nat.mod(x, 2n), 0n), ik_is_pow2_go(f, Nat.is_eq(Nat.div(x, 2n), 1n), Nat.div(x, 2n)))

def ik_is_pow2(+x: Nat) -> Bool:
  ik_is_pow2_go(64n, Nat.is_eq(x, 1n), x)

# ----------------------------------------------------------------------------
# 3a. Laws of the rings
# ----------------------------------------------------------------------------

# A non-zero answer of the doubling search holds `need` ...
def ikp_pow2_holds(fuel: Nat, done: Bool, +p: Nat, +need: Nat, hd: {Bool.and(done, Nat.is_lt(p, need)) == False{} : Bool}) -> {Bool.and(Nat.is_ne(ik_pow2_from(fuel, done, p, need), 0n), Nat.is_lt(ik_pow2_from(fuel, done, p, need), need)) == False{} : Bool}:
  match fuel:
    case 0n:
      match done:
        case True{}:
          ikp_and_fr(Nat.is_ne(p, 0n), Nat.is_lt(p, need), hd)
        case False{}:
          {==}
    case 1n+f:
      match done:
        case True{}:
          ikp_and_fr(Nat.is_ne(p, 0n), Nat.is_lt(p, need), hd)
        case False{}:
          ikp_pow2_holds(f, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need, ikp_ge_lt_excl(Nat.mul(p, 2n), need))

# ... and is at least where the search started.
def ikp_pow2_lo(fuel: Nat, done: Bool, +p: Nat, +need: Nat, +lo: Nat, h: {Nat.is_le(lo, p) == True{} : Bool}) -> {Bool.and(Nat.is_ne(ik_pow2_from(fuel, done, p, need), 0n), Nat.is_lt(ik_pow2_from(fuel, done, p, need), lo)) == False{} : Bool}:
  match fuel:
    case 0n:
      match done:
        case True{}:
          ikp_and_fr(Nat.is_ne(p, 0n), Nat.is_lt(p, lo), L.nat_le_lt_false(p, lo, h))
        case False{}:
          {==}
    case 1n+f:
      match done:
        case True{}:
          ikp_and_fr(Nat.is_ne(p, 0n), Nat.is_lt(p, lo), L.nat_le_lt_false(p, lo, h))
        case False{}:
          ikp_pow2_lo(f, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need, lo, L.nat_le_trans(lo, p, Nat.mul(p, 2n), h, ikp_le_double(p)))

def ikp_lt_mono_excl(c: Bool, +r: Nat, +x: Nat, +y: Nat, h: {Bool.and(c, Nat.is_lt(r, y)) == False{} : Bool}, hxy: {Nat.is_le(x, y) == True{} : Bool}) -> {Bool.and(c, Nat.is_lt(r, x)) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(r, x, L.nat_le_trans(x, y, r, hxy, L.nat_not_lt_imp_le(r, y, h)))

def ikp_ring_holds(+w: Nat, +rows: Nat, +x: Nat, hx: {Nat.is_le(x, ik_ring_need(w, rows)) == True{} : Bool}) -> {Bool.and(Nat.is_ne(ik_ring_slots(w, rows), 0n), Nat.is_lt(ik_ring_slots(w, rows), x)) == False{} : Bool}:
  ikp_lt_mono_excl(Nat.is_ne(ik_ring_slots(w, rows), 0n), ik_ring_slots(w, rows), x, ik_ring_need(w, rows), ikp_pow2_holds(40n, Nat.is_ge(128n, ik_ring_need(w, rows)), 128n, ik_ring_need(w, rows), ikp_ge_lt_excl(128n, ik_ring_need(w, rows))), hx)

# LAW IK.9
# promise: a K / V ring the sizing returns (non-zero) holds the window beside the rows of the widest pass, so no row sees the slots the rows ahead of it wrote.
law ik_ring_holds_pass:
  for +w: Nat
  for +rows: Nat
  {Bool.and(Nat.is_ne(ik_ring_slots(w, rows), 0n), Nat.is_lt(ik_ring_slots(w, rows), Nat.add(w, Nat.sub(rows, 1n)))) == False{} : Bool}

def ik_ring_holds_pass(w, rows):
  ikp_ring_holds(w, rows, Nat.add(w, Nat.sub(rows, 1n)), L.nat_add_le_mono(w, w, Nat.sub(rows, 1n), Nat.max(Nat.sub(rows, 1n), 64n), L.nat_le_refl(w), L.nat_max_ge_l(Nat.sub(rows, 1n), 64n)))

# LAW IK.10
# promise: a K / V ring the sizing returns (non-zero) holds the window beside one vector of 64 slots, so a vector's wrapped lanes stay outside a row's window.
law ik_ring_holds_vector:
  for +w: Nat
  for +rows: Nat
  {Bool.and(Nat.is_ne(ik_ring_slots(w, rows), 0n), Nat.is_lt(ik_ring_slots(w, rows), Nat.add(w, 64n))) == False{} : Bool}

def ik_ring_holds_vector(w, rows):
  ikp_ring_holds(w, rows, Nat.add(w, 64n), L.nat_add_le_mono(w, w, 64n, Nat.max(Nat.sub(rows, 1n), 64n), L.nat_le_refl(w), L.nat_max_ge_r(Nat.sub(rows, 1n), 64n)))

# LAW IK.11
# promise: a K / V ring the sizing returns (non-zero) has at least 128 slots (whole blocks of the softmax kernel's keys).
law ik_ring_min_slots:
  for +w: Nat
  for +rows: Nat
  {Bool.and(Nat.is_ne(ik_ring_slots(w, rows), 0n), Nat.is_lt(ik_ring_slots(w, rows), 128n)) == False{} : Bool}

def ik_ring_min_slots(w, rows):
  ikp_pow2_lo(40n, Nat.is_ge(128n, ik_ring_need(w, rows)), 128n, ik_ring_need(w, rows), 128n, L.nat_le_refl(128n))

# LAW IK.12
# promise: a convolution history ring the sizing returns (non-zero) holds the rows of the widest pass and the taps before its first row.
law ik_hist_holds_pass:
  for +rows: Nat
  for +taps: Nat
  {Bool.and(Nat.is_ne(ik_hist_rows(rows, taps), 0n), Nat.is_lt(ik_hist_rows(rows, taps), Nat.sub(Nat.add(rows, taps), 1n))) == False{} : Bool}

def ik_hist_holds_pass(rows, taps):
  ikp_pow2_holds(40n, Nat.is_ge(128n, Nat.sub(Nat.add(rows, taps), 1n)), 128n, Nat.sub(Nat.add(rows, taps), 1n), ikp_ge_lt_excl(128n, Nat.sub(Nat.add(rows, taps), 1n)))

def ikp_within_le(+slots: Nat, +mx: Nat, f: Bool, hf: {Nat.is_le(slots, mx) == f : Bool}) -> {Nat.is_le(ik_ring_within_if(slots, f), mx) == True{} : Bool}:
  match f:
    case True{}:
      hf
    case False{}:
      L.nat_le_zero(mx)

# LAW IK.13
# promise: a K / V ring the kernel-bounded sizing returns never has more slots than the chunk the softmax kernel stages.
law ik_ring_within_bounded:
  for +w: Nat
  for +rows: Nat
  for +mx: Nat
  {Nat.is_le(ik_ring_slots_within(w, rows, mx), mx) == True{} : Bool}

def ik_ring_within_bounded(w, rows, mx):
  ikp_within_le(ik_ring_slots(w, rows), mx, Nat.is_le(ik_ring_slots(w, rows), mx), {==})

def ikp_within_is(+slots: Nat, f: Bool) -> {Bool.and(Nat.is_ne(ik_ring_within_if(slots, f), 0n), Nat.is_ne(ik_ring_within_if(slots, f), slots)) == False{} : Bool}:
  match f:
    case True{}:
      ikp_and_fr(Nat.is_ne(slots, 0n), Nat.is_ne(slots, slots), ikp_ne_of_eq(slots, slots, {==}))
    case False{}:
      {==}

# LAW IK.14
# promise: the kernel-bounded sizing returns the K / V ring itself or no ring (0), never a different, smaller one.
law ik_ring_within_is_ring:
  for +w: Nat
  for +rows: Nat
  for +mx: Nat
  {Bool.and(Nat.is_ne(ik_ring_slots_within(w, rows, mx), 0n), Nat.is_ne(ik_ring_slots_within(w, rows, mx), ik_ring_slots(w, rows))) == False{} : Bool}

def ik_ring_within_is_ring(w, rows, mx):
  ikp_within_is(ik_ring_slots(w, rows), Nat.is_le(ik_ring_slots(w, rows), mx))

def ikp_not_ft(x: Bool, h: {Bool.not(x) == False{} : Bool}) -> {x == True{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      L.bool_absurd_tf({False{} == True{} : Bool}, h)

def ikp_pow2_double_m(+q: Nat) -> {Nat.is_eq(Nat.mod(Nat.mul(1n+q, 2n), 2n), 0n) == True{} : Bool}:
  %Equal.sym(Nat, Nat.mod(Nat.mul(1n+q, 2n), 2n), 0n, ikp_mod_mul(1n+q, 1n)) : {Nat.is_eq(_, 0n) == True{} : Bool}
  {==}

def ikp_pow2_double_d(f: Nat, +q: Nat, h: {ik_is_pow2_go(f, Nat.is_eq(1n+q, 1n), 1n+q) == True{} : Bool}) -> {ik_is_pow2_go(f, Nat.is_eq(Nat.div(Nat.mul(1n+q, 2n), 2n), 1n), Nat.div(Nat.mul(1n+q, 2n), 2n)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.div(Nat.mul(1n+q, 2n), 2n), 1n+q, ikp_div_mul(1n+q, 1n)) : {ik_is_pow2_go(f, Nat.is_eq(_, 1n), Nat.div(Nat.mul(1n+q, 2n), 2n)) == True{} : Bool}
  %Equal.sym(Nat, Nat.div(Nat.mul(1n+q, 2n), 2n), 1n+q, ikp_div_mul(1n+q, 1n)) : {ik_is_pow2_go(f, Nat.is_eq(1n+q, 1n), _) == True{} : Bool}
  h

# doubling a power of two takes one more halving
def ikp_pow2_double(f: Nat, +p: Nat, hp: {Nat.is_le(1n, p) == True{} : Bool}, h: {ik_is_pow2_go(f, Nat.is_eq(p, 1n), p) == True{} : Bool}) -> {ik_is_pow2_go(1n+f, Nat.is_eq(Nat.mul(p, 2n), 1n), Nat.mul(p, 2n)) == True{} : Bool}:
  match p:
    case 0n:
      L.bool_absurd({ik_is_pow2_go(1n+f, Nat.is_eq(Nat.mul(0n, 2n), 1n), Nat.mul(0n, 2n)) == True{} : Bool}, hp)
    case 1n+q:
      ikp_and_tt(Nat.is_eq(Nat.mod(Nat.mul(1n+q, 2n), 2n), 0n), ik_is_pow2_go(f, Nat.is_eq(Nat.div(Nat.mul(1n+q, 2n), 2n), 1n), Nat.div(Nat.mul(1n+q, 2n), 2n)), ikp_pow2_double_m(q), ikp_pow2_double_d(f, q, h))

# more halvings allowed: still a power of two
def ikp_pow2_fuel(f: Nat, o: Bool, +x: Nat, +h: {ik_is_pow2_go(f, o, x) == True{} : Bool}) -> {ik_is_pow2_go(1n+f, o, x) == True{} : Bool}:
  match f:
    case 0n:
      match o:
        case True{}:
          {==}
        case False{}:
          L.bool_absurd({ik_is_pow2_go(1n, False{}, x) == True{} : Bool}, h)
    case 1n+g:
      match o:
        case True{}:
          {==}
        case False{}:
          ikp_and_tt(Nat.is_eq(Nat.mod(x, 2n), 0n), ik_is_pow2_go(1n+g, Nat.is_eq(Nat.div(x, 2n), 1n), Nat.div(x, 2n)), ikp_and_l(Nat.is_eq(Nat.mod(x, 2n), 0n), ik_is_pow2_go(g, Nat.is_eq(Nat.div(x, 2n), 1n), Nat.div(x, 2n)), h), ikp_pow2_fuel(g, Nat.is_eq(Nat.div(x, 2n), 1n), Nat.div(x, 2n), ikp_and_r(Nat.is_eq(Nat.mod(x, 2n), 0n), ik_is_pow2_go(g, Nat.is_eq(Nat.div(x, 2n), 1n), Nat.div(x, 2n)), h)))

def ikp_pow2_lift(+k: Nat, +f: Nat, +o: Bool, +x: Nat, h: {ik_is_pow2_go(f, o, x) == True{} : Bool}) -> {ik_is_pow2_go(Nat.add(k, f), o, x) == True{} : Bool}:
  match k:
    case 0n:
      h
    case 1n+kp:
      ikp_pow2_fuel(Nat.add(kp, f), o, x, ikp_pow2_lift(kp, f, o, x, h))

# every answer of the doubling search is a power of two
def ikp_pow2_search(+fuel: Nat, done: Bool, +p: Nat, +need: Nat, +F: Nat, +hp: {Nat.is_le(1n, p) == True{} : Bool}, +h: {ik_is_pow2_go(F, Nat.is_eq(p, 1n), p) == True{} : Bool}) -> {Bool.and(Nat.is_ne(ik_pow2_from(fuel, done, p, need), 0n), Bool.not(ik_is_pow2_go(Nat.add(fuel, F), Nat.is_eq(ik_pow2_from(fuel, done, p, need), 1n), ik_pow2_from(fuel, done, p, need)))) == False{} : Bool}:
  match fuel:
    case 0n:
      match done:
        case True{}:
          ikp_and_fr(Nat.is_ne(p, 0n), Bool.not(ik_is_pow2_go(F, Nat.is_eq(p, 1n), p)), ikp_not_t(ik_is_pow2_go(F, Nat.is_eq(p, 1n), p), h))
        case False{}:
          {==}
    case 1n+f:
      match done:
        case True{}:
          ikp_and_fr(Nat.is_ne(p, 0n), Bool.not(ik_is_pow2_go(Nat.add(1n+f, F), Nat.is_eq(p, 1n), p)), ikp_not_t(ik_is_pow2_go(Nat.add(1n+f, F), Nat.is_eq(p, 1n), p), ikp_pow2_lift(1n+f, F, Nat.is_eq(p, 1n), p, h)))
        case False{}:
          %L.nat_add_succ_r(f, F) : {Bool.and(Nat.is_ne(ik_pow2_from(f, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need), 0n), Bool.not(ik_is_pow2_go(_, Nat.is_eq(ik_pow2_from(f, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need), 1n), ik_pow2_from(f, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need)))) == False{} : Bool}
          ikp_pow2_search(f, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need, 1n+F, L.nat_le_trans(1n, p, Nat.mul(p, 2n), hp, ikp_le_double(p)), ikp_pow2_double(F, p, hp, h))

def ikp_pow2_more(c: Bool, +k: Nat, +f: Nat, +o: Bool, +x: Nat, h: {Bool.and(c, Bool.not(ik_is_pow2_go(f, o, x))) == False{} : Bool}) -> {Bool.and(c, Bool.not(ik_is_pow2_go(Nat.add(k, f), o, x))) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      ikp_not_t(ik_is_pow2_go(Nat.add(k, f), o, x), ikp_pow2_lift(k, f, o, x, ikp_not_ft(ik_is_pow2_go(f, o, x), h)))

def ikp_pow2_rows_pow2(+need: Nat) -> {Bool.and(Nat.is_ne(ik_pow2_rows(need), 0n), Bool.not(ik_is_pow2(ik_pow2_rows(need)))) == False{} : Bool}:
  ikp_pow2_more(Nat.is_ne(ik_pow2_rows(need), 0n), 17n, Nat.add(40n, 7n), Nat.is_eq(ik_pow2_rows(need), 1n), ik_pow2_rows(need), ikp_pow2_search(40n, Nat.is_ge(128n, need), 128n, need, 7n, {==}, {==}))

# LAW IK.15
# promise: a K / V ring the sizing returns (non-zero) has a power-of-two number of slots, so the kernels' slot mask is the position modulo the ring.
law ik_ring_pow2:
  for +w: Nat
  for +rows: Nat
  {Bool.and(Nat.is_ne(ik_ring_slots(w, rows), 0n), Bool.not(ik_is_pow2(ik_ring_slots(w, rows)))) == False{} : Bool}

def ik_ring_pow2(w, rows):
  ikp_pow2_rows_pow2(ik_ring_need(w, rows))

# LAW IK.16
# promise: a convolution history ring the sizing returns (non-zero) has a power-of-two number of rows, so the kernels' row mask is the position modulo the ring.
law ik_hist_pow2:
  for +rows: Nat
  for +taps: Nat
  {Bool.and(Nat.is_ne(ik_hist_rows(rows, taps), 0n), Bool.not(ik_is_pow2(ik_hist_rows(rows, taps)))) == False{} : Bool}

def ik_hist_pow2(rows, taps):
  ikp_pow2_rows_pow2(Nat.sub(Nat.add(rows, taps), 1n))

# ============================================================================
# 4. Relative-position bias table
# ============================================================================

# Channel of the reversed table that holds distance 0: a block of 64 keys whose first key is at distance
# `dd <= off` loads the 64 channels from `off - dd`; with `off = extent + 63` every key of a block that is
# further away is beyond the extent.
def ik_bias_off(extent: Nat) -> Nat:
  Nat.add(extent, 63n)

# Channels of one head's reversed, padded table.
def ik_bias_width(extent: Nat) -> Nat:
  Nat.add(ik_bias_off(extent), 65n)

def ik_bias_src_if(off: Nat, extent: Nat, +x: Nat, in_table: Bool) -> Nat:
  match in_table:
    case True{}:
      Nat.sub(off, x)
    case False{}:
      extent

# The distance whose bias channel `x` of the reversed table holds; `extent` (no bias: zero) for the padding.
def ik_bias_src(+extent: Nat, +x: Nat) -> Nat:
  ik_bias_src_if(
    ik_bias_off(extent),
    extent,
    x,
    Bool.and(Nat.is_le(x, ik_bias_off(extent)), Nat.is_lt(Nat.sub(ik_bias_off(extent), x), extent))
  )

# ----------------------------------------------------------------------------
# 4a. Laws of the bias table
# ----------------------------------------------------------------------------

# LAW IK.17
# promise: the 64 bias channels a softmax block loads for any first-key distance up to the table's offset stay inside one head's table.
law ik_bias_block_in_table:
  for +e: Nat
  for +dd: Nat
  {Nat.is_le(Nat.add(Nat.sub(ik_bias_off(e), dd), 64n), ik_bias_width(e)) == True{} : Bool}

def ik_bias_block_in_table(e, dd):
  L.nat_le_trans(Nat.add(Nat.sub(ik_bias_off(e), dd), 64n), Nat.add(ik_bias_off(e), 64n), ik_bias_width(e), L.nat_add_le_mono(Nat.sub(ik_bias_off(e), dd), ik_bias_off(e), 64n, 64n, ikp_sub_le(ik_bias_off(e), dd), L.nat_le_refl(64n)), L.nat_add_le_mono(ik_bias_off(e), ik_bias_off(e), 64n, 65n, L.nat_le_refl(ik_bias_off(e)), {==}))

def ikp_far_h(+e: Nat, +dd: Nat, hc: {Nat.is_lt(Nat.add(e, 63n), dd) == True{} : Bool}) -> {Nat.is_le(Nat.add(63n, e), dd) == True{} : Bool}:
  %L.nat_add_comm(e, 63n) : {Nat.is_le(_, dd) == True{} : Bool}
  L.nat_lt_imp_le(Nat.add(e, 63n), dd, hc)

def ikp_far(+e: Nat, +dd: Nat, c: Bool, hc: {Nat.is_lt(ik_bias_off(e), dd) == c : Bool}) -> {Bool.and(c, Nat.is_lt(Nat.sub(dd, 63n), e)) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(Nat.sub(dd, 63n), e, ikp_le_sub(63n, e, dd, ikp_far_h(e, dd, hc)))

# LAW IK.18
# promise: a softmax block whose first key is further than the table's offset holds no key within the bias extent, so skipping its bias load drops no bias.
law ik_bias_far_block_unbiased:
  for +e: Nat
  for +dd: Nat
  {Bool.and(Nat.is_lt(ik_bias_off(e), dd), Nat.is_lt(Nat.sub(dd, 63n), e)) == False{} : Bool}

def ik_bias_far_block_unbiased(e, dd):
  ikp_far(e, dd, Nat.is_lt(ik_bias_off(e), dd), {==})

def ikp_src_le(+e: Nat, +x: Nat, b: Bool, hb: {Bool.and(Nat.is_le(x, ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), x), e)) == b : Bool}) -> {Nat.is_le(ik_bias_src_if(ik_bias_off(e), e, x, b), e) == True{} : Bool}:
  match b:
    case True{}:
      L.nat_lt_imp_le(Nat.sub(ik_bias_off(e), x), e, ikp_and_r(Nat.is_le(x, ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), x), e), hb))
    case False{}:
      L.nat_le_refl(e)

# LAW IK.19
# promise: a bias table channel holds a distance below the extent or the no-bias value (the extent), never a distance beyond it.
law ik_bias_src_bounded:
  for +e: Nat
  for +x: Nat
  {Nat.is_le(ik_bias_src(e, x), e) == True{} : Bool}

def ik_bias_src_bounded(e, x):
  ikp_src_le(e, x, Bool.and(Nat.is_le(x, ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), x), e)), {==})

def ikp_src_channel(+e: Nat, +x: Nat, b: Bool, +hb: {Bool.and(Nat.is_le(x, ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), x), e)) == b : Bool}) -> {Bool.and(Nat.is_lt(ik_bias_src_if(ik_bias_off(e), e, x, b), e), Nat.is_ne(Nat.add(ik_bias_src_if(ik_bias_off(e), e, x, b), x), ik_bias_off(e))) == False{} : Bool}:
  match b:
    case True{}:
      ikp_and_fr(Nat.is_lt(Nat.sub(ik_bias_off(e), x), e), Nat.is_ne(Nat.add(Nat.sub(ik_bias_off(e), x), x), ik_bias_off(e)), ikp_ne_of_eq(Nat.add(Nat.sub(ik_bias_off(e), x), x), ik_bias_off(e), L.nat_sub_add_cancel(ik_bias_off(e), x, ikp_and_l(Nat.is_le(x, ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), x), e), hb))))
    case False{}:
      ikp_and_fl(Nat.is_lt(e, e), Nat.is_ne(Nat.add(e, x), ik_bias_off(e)), ikp_lt_irrefl(e))

# LAW IK.20
# promise: a channel that holds a distance below the extent is the channel the softmax kernel reads for that distance (the table's offset less the distance).
law ik_bias_src_channel:
  for +e: Nat
  for +x: Nat
  {Bool.and(Nat.is_lt(ik_bias_src(e, x), e), Nat.is_ne(Nat.add(ik_bias_src(e, x), x), ik_bias_off(e))) == False{} : Bool}

def ik_bias_src_channel(e, x):
  ikp_src_channel(e, x, Bool.and(Nat.is_le(x, ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), x), e)), {==})

def ikp_d_le_off(+e: Nat, +d: Nat, hd: {Nat.is_lt(d, e) == True{} : Bool}) -> {Nat.is_le(d, ik_bias_off(e)) == True{} : Bool}:
  L.nat_le_trans(d, e, Nat.add(e, 63n), L.nat_lt_imp_le(d, e, hd), L.nat_add_le_r(e, 63n))

def ikp_dist_hb(+e: Nat, +d: Nat, +hd: {Nat.is_lt(d, e) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(ik_bias_off(e), Nat.sub(ik_bias_off(e), d)), e) == True{} : Bool}:
  %Equal.sym(Nat, Nat.sub(ik_bias_off(e), Nat.sub(ik_bias_off(e), d)), d, ikp_sub_sub(ik_bias_off(e), d, ikp_d_le_off(e, d, hd))) : {Nat.is_lt(_, e) == True{} : Bool}
  hd

def ikp_dist_hit(+e: Nat, +d: Nat, +hd: {Nat.is_lt(d, e) == True{} : Bool}, b: Bool, hb: {Bool.and(Nat.is_le(Nat.sub(ik_bias_off(e), d), ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), Nat.sub(ik_bias_off(e), d)), e)) == b : Bool}) -> {Nat.is_ne(ik_bias_src_if(ik_bias_off(e), e, Nat.sub(ik_bias_off(e), d), b), d) == False{} : Bool}:
  match b:
    case True{}:
      ikp_ne_of_eq(Nat.sub(ik_bias_off(e), Nat.sub(ik_bias_off(e), d)), d, ikp_sub_sub(ik_bias_off(e), d, ikp_d_le_off(e, d, hd)))
    case False{}:
      L.bool_absurd_tf({Nat.is_ne(e, d) == False{} : Bool}, Equal.trans(Bool, True{}, Bool.and(Nat.is_le(Nat.sub(ik_bias_off(e), d), ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), Nat.sub(ik_bias_off(e), d)), e)), False{}, Equal.sym(Bool, Bool.and(Nat.is_le(Nat.sub(ik_bias_off(e), d), ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), Nat.sub(ik_bias_off(e), d)), e)), True{}, ikp_and_tt(Nat.is_le(Nat.sub(ik_bias_off(e), d), ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), Nat.sub(ik_bias_off(e), d)), e), ikp_sub_le(ik_bias_off(e), d), ikp_dist_hb(e, d, hd))), hb))

def ikp_dist_ch(+e: Nat, +d: Nat, c: Bool, hc: {Nat.is_lt(d, e) == c : Bool}) -> {Bool.and(c, Nat.is_ne(ik_bias_src(e, Nat.sub(ik_bias_off(e), d)), d)) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      ikp_dist_hit(e, d, hc, Bool.and(Nat.is_le(Nat.sub(ik_bias_off(e), d), ik_bias_off(e)), Nat.is_lt(Nat.sub(ik_bias_off(e), Nat.sub(ik_bias_off(e), d)), e)), {==})

# LAW IK.21
# promise: every distance below the bias extent has its bias in the channel the softmax kernel reads for it (the table's offset less the distance).
law ik_bias_distance_channel:
  for +e: Nat
  for +d: Nat
  {Bool.and(Nat.is_lt(d, e), Nat.is_ne(ik_bias_src(e, Nat.sub(ik_bias_off(e), d)), d)) == False{} : Bool}

def ik_bias_distance_channel(e, d):
  ikp_dist_ch(e, d, Nat.is_lt(d, e), {==})

# ============================================================================
# 5. Projection layout and 4-bit expert rows
# ============================================================================

# Whole blocks of 128 channels that hold `x` channels.
def ik_align128(x: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.add(x, 127n), 128n), 128n)

# Channels of the query heads (128 per head): the head part of the query-side projection and the
# width of the attention output the output projection reads.
def ik_heads_width(n_heads: Nat) -> Nat:
  Nat.mul(128n, n_heads)

# Channels of the query-side projection: the query heads, the output gate, the relative states (16 per
# head), padded to whole blocks of 128.
def ik_gate_off(n_heads: Nat) -> Nat:
  Nat.mul(128n, n_heads)

def ik_rel_off(n_heads: Nat) -> Nat:
  Nat.mul(256n, n_heads)

def ik_qx_width(+n_heads: Nat, +d_rel: Nat) -> Nat:
  ik_align128(Nat.add(ik_rel_off(n_heads), Nat.mul(d_rel, n_heads)))

# Channels of the key / value projection: the heads, then (global layers) the per-row convolution taps.
def ik_kx_width(+n_kv: Nat, taps: Nat) -> Nat:
  ik_align128(Nat.add(Nat.mul(128n, n_kv), Nat.mul(taps, n_kv)))

# A row of `cols` 4-bit codes with one scale per 16: the row's byte length, 0 when `cols` is not a
# whole number of scale blocks.
def ik_nib_row_bytes_if(+cols: Nat, whole: Bool) -> Nat:
  match whole:
    case True{}:
      Nat.div(cols, 2n)
    case False{}:
      0n

def ik_nib_row_bytes(+cols: Nat) -> Nat:
  ik_nib_row_bytes_if(cols, Bool.and(Nat.is_gt(cols, 0n), Nat.is_eq(Nat.mod(cols, 16n), 0n)))

# Scales of such a row.
def ik_nib_row_scales(cols: Nat) -> Nat:
  Nat.div(cols, 16n)

# ----------------------------------------------------------------------------
# 5a. Laws of the projection layout and the 4-bit rows
# ----------------------------------------------------------------------------

def ikp_mod128_le(+x: Nat) -> {Nat.is_le(Nat.mod(Nat.add(x, 127n), 128n), 127n) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_le(Nat.mod(Nat.add(x, 127n), 128n), 127n), Nat.is_lt(Nat.mod(Nat.add(x, 127n), 128n), 128n), True{}, Equal.sym(Bool, Nat.is_lt(Nat.mod(Nat.add(x, 127n), 128n), 128n), Nat.is_le(Nat.mod(Nat.add(x, 127n), 128n), 127n), L.nat_lt_succ_eq_le(Nat.mod(Nat.add(x, 127n), 128n), 127n)), L.nat_mod_lt(Nat.add(x, 127n), 127n))

def ikp_align_ge_h(+x: Nat) -> {Nat.is_le(Nat.add(127n, x), Nat.add(127n, ik_align128(x))) == True{} : Bool}:
  %L.nat_add_comm(x, 127n) : {Nat.is_le(_, Nat.add(127n, ik_align128(x))) == True{} : Bool}
  %L.nat_div_mod(Nat.add(x, 127n), 127n) : {Nat.is_le(_, Nat.add(127n, ik_align128(x))) == True{} : Bool}
  L.nat_add_le_mono(Nat.mod(Nat.add(x, 127n), 128n), 127n, ik_align128(x), ik_align128(x), ikp_mod128_le(x), L.nat_le_refl(ik_align128(x)))

# whole blocks of 128 hold what they were sized for
def ikp_align_ge(+x: Nat) -> {Nat.is_le(x, ik_align128(x)) == True{} : Bool}:
  ikp_add_le_cancel(127n, x, ik_align128(x), ikp_align_ge_h(x))

def ikp_align_whole(+x: Nat) -> {Nat.is_eq(Nat.mod(ik_align128(x), 128n), 0n) == True{} : Bool}:
  %Equal.sym(Nat, Nat.mod(ik_align128(x), 128n), 0n, ikp_mod_mul(Nat.div(Nat.add(x, 127n), 128n), 127n)) : {Nat.is_eq(_, 0n) == True{} : Bool}
  {==}

def ikp_gate_rel(+nh: Nat) -> {Nat.is_le(Nat.add(ik_gate_off(nh), Nat.mul(128n, nh)), ik_rel_off(nh)) == True{} : Bool}:
  %L.nat_mul_add_dist_r(128n, 128n, nh) : {Nat.is_le(_, ik_rel_off(nh)) == True{} : Bool}
  L.nat_le_refl(Nat.mul(256n, nh))

# LAW IK.22
# promise: the query-side projection places the query heads before the output gate and the gate before the relative states, so no two parts share a channel.
law ik_qx_parts_ordered:
  for +nh: Nat
  {Bool.and(Nat.is_le(ik_heads_width(nh), ik_gate_off(nh)), Nat.is_le(Nat.add(ik_gate_off(nh), ik_heads_width(nh)), ik_rel_off(nh))) == True{} : Bool}

def ik_qx_parts_ordered(nh):
  ikp_and_tt(Nat.is_le(Nat.mul(128n, nh), ik_gate_off(nh)), Nat.is_le(Nat.add(ik_gate_off(nh), Nat.mul(128n, nh)), ik_rel_off(nh)), L.nat_le_refl(Nat.mul(128n, nh)), ikp_gate_rel(nh))

# LAW IK.23
# promise: the relative states of every head end inside the query-side projection's width.
law ik_qx_holds_rel:
  for +nh: Nat
  for +dr: Nat
  {Nat.is_le(Nat.add(ik_rel_off(nh), Nat.mul(dr, nh)), ik_qx_width(nh, dr)) == True{} : Bool}

def ik_qx_holds_rel(nh, dr):
  ikp_align_ge(Nat.add(ik_rel_off(nh), Nat.mul(dr, nh)))

# LAW IK.24
# promise: the key / value projection holds every key / value head and every head's convolution taps.
law ik_kx_holds_heads:
  for +nkv: Nat
  for +t: Nat
  {Nat.is_le(Nat.add(Nat.mul(128n, nkv), Nat.mul(t, nkv)), ik_kx_width(nkv, t)) == True{} : Bool}

def ik_kx_holds_heads(nkv, t):
  ikp_align_ge(Nat.add(Nat.mul(128n, nkv), Nat.mul(t, nkv)))

# LAW IK.25
# promise: the query-side and key / value projections are whole blocks of 128 channels.
law ik_proj_whole_blocks:
  for +nh: Nat
  for +dr: Nat
  for +nkv: Nat
  for +t: Nat
  {Bool.and(Nat.is_eq(Nat.mod(ik_qx_width(nh, dr), 128n), 0n), Nat.is_eq(Nat.mod(ik_kx_width(nkv, t), 128n), 0n)) == True{} : Bool}

def ik_proj_whole_blocks(nh, dr, nkv, t):
  ikp_and_tt(Nat.is_eq(Nat.mod(ik_qx_width(nh, dr), 128n), 0n), Nat.is_eq(Nat.mod(ik_kx_width(nkv, t), 128n), 0n), ikp_align_whole(Nat.add(ik_rel_off(nh), Nat.mul(dr, nh))), ikp_align_whole(Nat.add(Nat.mul(128n, nkv), Nat.mul(t, nkv))))

# a whole number of scale blocks: c == (c / 16) 16
def ikp_whole16(+c: Nat, hm: {Nat.mod(c, 16n) == 0n : Nat}) -> {Nat.mul(Nat.div(c, 16n), 16n) == c : Nat}:
  %hm : {Nat.add(_, Nat.mul(Nat.div(c, 16n), 16n)) == c : Nat}
  L.nat_div_mod(c, 15n)

def ikp_nib16(+c: Nat, +hm: {Nat.mod(c, 16n) == 0n : Nat}) -> {Nat.is_eq(Nat.mul(16n, Nat.div(c, 16n)), c) == True{} : Bool}:
  %L.nat_mul_comm(Nat.div(c, 16n), 16n) : {Nat.is_eq(_, c) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(Nat.div(c, 16n), 16n), c, ikp_whole16(c, hm)) : {Nat.is_eq(_, c) == True{} : Bool}
  ikp_eq_refl(c)

def ikp_c2(+c: Nat, +hm: {Nat.mod(c, 16n) == 0n : Nat}) -> {Nat.mul(Nat.mul(Nat.div(c, 16n), 8n), 2n) == c : Nat}:
  Equal.trans(Nat, Nat.mul(Nat.mul(Nat.div(c, 16n), 8n), 2n), Nat.mul(Nat.div(c, 16n), 16n), c, L.nat_mul_assoc(Nat.div(c, 16n), 8n, 2n), ikp_whole16(c, hm))

def ikp_nib2(+c: Nat, +hm: {Nat.mod(c, 16n) == 0n : Nat}) -> {Nat.is_eq(Nat.mul(2n, Nat.div(c, 2n)), c) == True{} : Bool}:
  %ikp_c2(c, hm) : {Nat.is_eq(Nat.mul(2n, Nat.div(_, 2n)), c) == True{} : Bool}
  %Equal.sym(Nat, Nat.div(Nat.mul(Nat.mul(Nat.div(c, 16n), 8n), 2n), 2n), Nat.mul(Nat.div(c, 16n), 8n), ikp_div_mul(Nat.mul(Nat.div(c, 16n), 8n), 1n)) : {Nat.is_eq(Nat.mul(2n, _), c) == True{} : Bool}
  %L.nat_mul_comm(Nat.mul(Nat.div(c, 16n), 8n), 2n) : {Nat.is_eq(_, c) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(Nat.mul(Nat.div(c, 16n), 8n), 2n), c, ikp_c2(c, hm)) : {Nat.is_eq(_, c) == True{} : Bool}
  ikp_eq_refl(c)

def ikp_nib_ok(+c: Nat, w: Bool, +hw: {Bool.and(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 16n), 0n)) == w : Bool}) -> {Bool.and(Nat.is_ne(ik_nib_row_bytes_if(c, w), 0n), Bool.not(Bool.and(Nat.is_eq(Nat.mul(2n, ik_nib_row_bytes_if(c, w)), c), Nat.is_eq(Nat.mul(16n, ik_nib_row_scales(c)), c)))) == False{} : Bool}:
  match w:
    case False{}:
      {==}
    case True{}:
      ikp_and_fr(Nat.is_ne(Nat.div(c, 2n), 0n), Bool.not(Bool.and(Nat.is_eq(Nat.mul(2n, Nat.div(c, 2n)), c), Nat.is_eq(Nat.mul(16n, Nat.div(c, 16n)), c))), ikp_not_t(Bool.and(Nat.is_eq(Nat.mul(2n, Nat.div(c, 2n)), c), Nat.is_eq(Nat.mul(16n, Nat.div(c, 16n)), c)), ikp_and_tt(Nat.is_eq(Nat.mul(2n, Nat.div(c, 2n)), c), Nat.is_eq(Nat.mul(16n, Nat.div(c, 16n)), c), ikp_nib2(c, ikp_eq_of(Nat.mod(c, 16n), 0n, ikp_and_r(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 16n), 0n), hw))), ikp_nib16(c, ikp_eq_of(Nat.mod(c, 16n), 0n, ikp_and_r(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 16n), 0n), hw))))))

# LAW IK.26
# promise: a 4-bit row the geometry accepts (non-zero bytes) has two codes per byte and one scale per 16 codes, covering exactly its columns.
law ik_nib_row_consistent:
  for +c: Nat
  {Bool.and(Nat.is_ne(ik_nib_row_bytes(c), 0n), Bool.not(Bool.and(Nat.is_eq(Nat.mul(2n, ik_nib_row_bytes(c)), c), Nat.is_eq(Nat.mul(16n, ik_nib_row_scales(c)), c)))) == False{} : Bool}

def ik_nib_row_consistent(c):
  ikp_nib_ok(c, Bool.and(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 16n), 0n)), {==})

def ikp_nib_partial(+c: Nat, w: Bool, hw: {Bool.and(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 16n), 0n)) == w : Bool}) -> {Bool.and(Nat.is_ne(Nat.mod(c, 16n), 0n), Nat.is_ne(ik_nib_row_bytes_if(c, w), 0n)) == False{} : Bool}:
  match w:
    case True{}:
      ikp_and_fl(Nat.is_ne(Nat.mod(c, 16n), 0n), Nat.is_ne(Nat.div(c, 2n), 0n), ikp_ne_of_is_eq(Nat.mod(c, 16n), 0n, ikp_and_r(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 16n), 0n), hw)))
    case False{}:
      ikp_and_fr(Nat.is_ne(Nat.mod(c, 16n), 0n), Nat.is_ne(0n, 0n), {==})

# LAW IK.27
# promise: a column count that is not a whole number of 16-code scale blocks gets no 4-bit row geometry (0 bytes).
law ik_nib_partial_refused:
  for +c: Nat
  {Bool.and(Nat.is_ne(Nat.mod(c, 16n), 0n), Nat.is_ne(ik_nib_row_bytes(c), 0n)) == False{} : Bool}

def ik_nib_partial_refused(c):
  ikp_nib_partial(c, Bool.and(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 16n), 0n)), {==})

# A row of `cols` 8-bit floats with one scale per 32 values (the second stored expert form): usable when
# `cols` is a non-zero whole number of 32-value scale blocks; the row's bytes are its values.
def ik_f8_row_ok(+cols: Nat) -> Bool:
  Bool.and(Nat.is_gt(cols, 0n), Nat.is_eq(Nat.mod(cols, 32n), 0n))

def ik_f8_row_scales(cols: Nat) -> Nat:
  Nat.div(cols, 32n)

def ikp_whole32(+c: Nat, hm: {Nat.mod(c, 32n) == 0n : Nat}) -> {Nat.mul(Nat.div(c, 32n), 32n) == c : Nat}:
  %hm : {Nat.add(_, Nat.mul(Nat.div(c, 32n), 32n)) == c : Nat}
  L.nat_div_mod(c, 31n)

def ikp_f8_32(+c: Nat, +hm: {Nat.mod(c, 32n) == 0n : Nat}) -> {Nat.is_eq(Nat.mul(32n, Nat.div(c, 32n)), c) == True{} : Bool}:
  %L.nat_mul_comm(Nat.div(c, 32n), 32n) : {Nat.is_eq(_, c) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(Nat.div(c, 32n), 32n), c, ikp_whole32(c, hm)) : {Nat.is_eq(_, c) == True{} : Bool}
  ikp_eq_refl(c)

def ikp_f8_ok(+c: Nat, w: Bool, +hw: {Bool.and(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 32n), 0n)) == w : Bool}) -> {Bool.and(w, Bool.not(Nat.is_eq(Nat.mul(32n, ik_f8_row_scales(c)), c))) == False{} : Bool}:
  match w:
    case False{}:
      {==}
    case True{}:
      ikp_not_t(Nat.is_eq(Nat.mul(32n, Nat.div(c, 32n)), c), ikp_f8_32(c, ikp_eq_of(Nat.mod(c, 32n), 0n, ikp_and_r(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 32n), 0n), hw))))

# LAW IK.27a
# promise: an 8-bit-float expert row the geometry accepts has one scale per 32 values, covering exactly its columns.
# harm: the packer would read a row's scales past its last value or leave values without a scale
law ik_f8_row_consistent:
  for +c: Nat
  {Bool.and(ik_f8_row_ok(c), Bool.not(Nat.is_eq(Nat.mul(32n, ik_f8_row_scales(c)), c))) == False{} : Bool}

def ik_f8_row_consistent(c):
  ikp_f8_ok(c, Bool.and(Nat.is_gt(c, 0n), Nat.is_eq(Nat.mod(c, 32n), 0n)), {==})

# ============================================================================
# 6. Shape guard
# ============================================================================

# A width of whole blocks of 128 channels (the hidden size, an expert's width).
def ik_width_whole(+x: Nat) -> Bool:
  Bool.and(Nat.is_gt(x, 0n), Nat.is_eq(Nat.mod(x, 128n), 0n))

# Query heads in whole groups per key / value head, and in groups of eight (one block of 128
# relative-state channels).
def ik_head_groups_valid(+n_heads: Nat, +n_kv: Nat) -> Bool:
  Bool.and(
    Bool.and(Nat.is_gt(n_kv, 0n), Nat.is_gt(n_heads, 0n)),
    Bool.and(Nat.is_eq(Nat.mod(n_heads, n_kv), 0n), Nat.is_eq(Nat.mod(n_heads, 8n), 0n))
  )

# The router's selection: 1 to 12 experts per row, never more than the model has.
def ik_top_k_valid(+top_k: Nat, +n_experts: Nat) -> Bool:
  Bool.and(Nat.is_ge(top_k, 1n), Bool.and(Nat.is_le(top_k, 12n), Nat.is_le(top_k, n_experts)))

# The short convolutions' taps: 1 to 16.
def ik_taps_valid(+taps: Nat) -> Bool:
  Bool.and(Nat.is_ge(taps, 1n), Nat.is_le(taps, 16n))

# The shapes the family's kernels cover: whole blocks of 128 hidden channels, heads of 128 channels in
# groups of query heads per key / value head, eight heads' relative states of 16 per block, experts in
# blocks of 128 with at most 12 per row and a width of whole blocks, 1 to 16 convolution taps.
def ik_dims_valid(+hidden: Nat, +n_heads: Nat, +n_kv: Nat, head_dim: Nat, d_rel: Nat, +n_experts: Nat, +top_k: Nat, +expert_ff: Nat, +taps: Nat) -> Bool:
  Bool.and(
    ik_width_whole(hidden),
    Bool.and(
      Bool.and(Nat.is_eq(head_dim, 128n), Nat.is_eq(d_rel, 16n)),
      Bool.and(
        ik_head_groups_valid(n_heads, n_kv),
        Bool.and(
          Bool.and(Nat.is_gt(n_experts, 0n), Bool.and(Nat.is_eq(Nat.mod(n_experts, 128n), 0n), Nat.is_le(n_experts, 512n))),
          Bool.and(
            ik_top_k_valid(top_k, n_experts),
            Bool.and(ik_width_whole(expert_ff), ik_taps_valid(taps))
          )
        )
      )
    )
  )

# ----------------------------------------------------------------------------
# 6a. Attention geometry and vocabulary guards
# ----------------------------------------------------------------------------

# The sliding window and the bias extent the kernels cover: both non-zero, the extent within the
# window (a biased distance is then a real distance inside the ring).
def ik_attn_geom_valid(+window: Nat, +extent: Nat) -> Bool:
  Bool.and(Nat.is_gt(window, 0n), Bool.and(Nat.is_gt(extent, 0n), Nat.is_le(extent, window)))

# The emitted vocabulary: at least one token, and every emitted token a row of the embedding.
def ik_vocab_valid(+unpadded: Nat, +vocab: Nat) -> Bool:
  Bool.and(Nat.is_gt(unpadded, 0n), Nat.is_le(unpadded, vocab))

# Layer applications of one token fit `max_apps` (the plans hold one entry per application). The
# check divides instead of forming passes x span, so a pass count from a file cannot overflow it.
def ik_apps_fit_if(+span: Nat, +passes: Nat, +room: Nat) -> Bool:
  match span:
    case 0n:
      True{}
    case 1n+sp:
      Nat.is_le(Nat.sub(passes, 1n), Nat.div(room, 1n+sp))

def ik_apps_fit(+n_layers: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat, +max_apps: Nat) -> Bool:
  Bool.and(Nat.is_le(n_layers, max_apps), ik_apps_fit_if(ik_span(loop_start, loop_end), passes, Nat.sub(max_apps, n_layers)))

# The counts a model file sets that the sizing arithmetic multiplies or adds to: the layer
# applications of one token, the sliding window and the query heads, each within a limit the shell
# passes. Checked before any schedule, ring or projection width is computed from them.
def ik_counts_valid(+n_layers: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat, +window: Nat, +n_heads: Nat, +max_apps: Nat, +max_window: Nat, +max_heads: Nat) -> Bool:
  Bool.and(
    ik_apps_fit(n_layers, loop_start, loop_end, passes, max_apps),
    Bool.and(Nat.is_le(window, max_window), Nat.is_le(n_heads, max_heads))
  )

# ----------------------------------------------------------------------------
# 6b. Laws of the guards
# ----------------------------------------------------------------------------

def ikp_and_not_self(y: Bool) -> {Bool.and(y, Bool.not(y)) == False{} : Bool}:
  match y:
    case True{}:
      {==}
    case False{}:
      {==}

# LAW IK.28
# promise: a width the shape guard accepts is a non-zero whole number of 128-channel blocks.
law ik_width_whole_blocks:
  for +x: Nat
  {Bool.and(ik_width_whole(x), Bool.not(Bool.and(Nat.is_gt(x, 0n), Nat.is_eq(Nat.mod(x, 128n), 0n)))) == False{} : Bool}

def ik_width_whole_blocks(x):
  ikp_and_not_self(Bool.and(Nat.is_gt(x, 0n), Nat.is_eq(Nat.mod(x, 128n), 0n)))

# LAW IK.29
# promise: the shape guard refuses a model without query heads.
law ik_heads_nonzero:
  for +nh: Nat
  for +kv: Nat
  {Bool.and(ik_head_groups_valid(nh, kv), Nat.is_eq(nh, 0n)) == False{} : Bool}

def ik_heads_nonzero(nh, kv):
  match nh:
    case 0n:
      match kv:
        case 0n:
          {==}
        case 1n+kp:
          {==}
    case 1n+np:
      ikp_and_fr(ik_head_groups_valid(1n+np, kv), Nat.is_eq(1n+np, 0n), {==})

def ikp_heads_grouped(+nh: Nat, +kv: Nat, v: Bool, hv: {ik_head_groups_valid(nh, kv) == v : Bool}) -> {Bool.and(v, Nat.is_ne(Nat.mod(nh, kv), 0n)) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      ikp_ne_of_is_eq(Nat.mod(nh, kv), 0n, ikp_and_l(Nat.is_eq(Nat.mod(nh, kv), 0n), Nat.is_eq(Nat.mod(nh, 8n), 0n), ikp_and_r(Bool.and(Nat.is_gt(kv, 0n), Nat.is_gt(nh, 0n)), Bool.and(Nat.is_eq(Nat.mod(nh, kv), 0n), Nat.is_eq(Nat.mod(nh, 8n), 0n)), hv)))

# LAW IK.30
# promise: heads the shape guard accepts come in whole groups of query heads per key / value head (so a model without key / value heads is refused).
law ik_heads_grouped:
  for +nh: Nat
  for +kv: Nat
  {Bool.and(ik_head_groups_valid(nh, kv), Nat.is_ne(Nat.mod(nh, kv), 0n)) == False{} : Bool}

def ik_heads_grouped(nh, kv):
  ikp_heads_grouped(nh, kv, ik_head_groups_valid(nh, kv), {==})

def ikp_top_k(+k: Nat, +ne: Nat, v: Bool, +hv: {ik_top_k_valid(k, ne) == v : Bool}) -> {Bool.and(v, Bool.not(Bool.and(Nat.is_ge(k, 1n), Nat.is_le(k, ne)))) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      ikp_not_t(Bool.and(Nat.is_ge(k, 1n), Nat.is_le(k, ne)), ikp_and_tt(Nat.is_ge(k, 1n), Nat.is_le(k, ne), ikp_and_l(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, 12n), Nat.is_le(k, ne)), hv), ikp_and_r(Nat.is_le(k, 12n), Nat.is_le(k, ne), ikp_and_r(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, 12n), Nat.is_le(k, ne)), hv))))

# LAW IK.31
# promise: a router selection the shape guard accepts takes at least one expert and never more experts than the model has.
law ik_top_k_within_experts:
  for +k: Nat
  for +ne: Nat
  {Bool.and(ik_top_k_valid(k, ne), Bool.not(Bool.and(Nat.is_ge(k, 1n), Nat.is_le(k, ne)))) == False{} : Bool}

def ik_top_k_within_experts(k, ne):
  ikp_top_k(k, ne, ik_top_k_valid(k, ne), {==})

# LAW IK.32
# promise: the shape guard refuses short convolutions without taps.
law ik_taps_nonzero:
  for +t: Nat
  {Bool.and(ik_taps_valid(t), Nat.is_eq(t, 0n)) == False{} : Bool}

def ik_taps_nonzero(t):
  match t:
    case 0n:
      {==}
    case 1n+tp:
      ikp_and_fr(ik_taps_valid(1n+tp), Nat.is_eq(1n+tp, 0n), {==})

# The parts of the shape guard, as Bool facts: an accepted conjunction a & (b & (c & (e & (f & (g & t)))))
# holds each part.
def ikp_conj7_a(a: Bool, b: Bool, c: Bool, e: Bool, f: Bool, g: Bool, t: Bool) -> {Bool.and(Bool.and(a, Bool.and(b, Bool.and(c, Bool.and(e, Bool.and(f, Bool.and(g, t)))))), Bool.not(a)) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      match b:
        case False{}:
          {==}
        case True{}:
          match c:
            case False{}:
              {==}
            case True{}:
              match e:
                case False{}:
                  {==}
                case True{}:
                  match f:
                    case False{}:
                      {==}
                    case True{}:
                      match g:
                        case False{}:
                          {==}
                        case True{}:
                          match t:
                            case False{}:
                              {==}
                            case True{}:
                              {==}

def ikp_conj7_g(a: Bool, b: Bool, c: Bool, e: Bool, f: Bool, g: Bool, t: Bool) -> {Bool.and(Bool.and(a, Bool.and(b, Bool.and(c, Bool.and(e, Bool.and(f, Bool.and(g, t)))))), Bool.not(g)) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      match b:
        case False{}:
          {==}
        case True{}:
          match c:
            case False{}:
              {==}
            case True{}:
              match e:
                case False{}:
                  {==}
                case True{}:
                  match f:
                    case False{}:
                      {==}
                    case True{}:
                      match g:
                        case False{}:
                          {==}
                        case True{}:
                          match t:
                            case False{}:
                              {==}
                            case True{}:
                              {==}

def ikp_conj7_cft(a: Bool, b: Bool, c: Bool, e: Bool, f: Bool, g: Bool, t: Bool) -> {Bool.and(Bool.and(a, Bool.and(b, Bool.and(c, Bool.and(e, Bool.and(f, Bool.and(g, t)))))), Bool.not(Bool.and(c, Bool.and(f, t)))) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      match b:
        case False{}:
          {==}
        case True{}:
          match c:
            case False{}:
              {==}
            case True{}:
              match e:
                case False{}:
                  {==}
                case True{}:
                  match f:
                    case False{}:
                      {==}
                    case True{}:
                      match g:
                        case False{}:
                          {==}
                        case True{}:
                          match t:
                            case False{}:
                              {==}
                            case True{}:
                              {==}

# The three laws below tie the guard the shell calls (`ik_dims_valid`) to its part guards, so a
# later guard cannot drop a part. Each leaves the arguments of the parts it names free and fixes
# only the arguments of the other conjuncts to accepted values (head dim 128, 16 relative states,
# whole-block widths and expert counts), so that the sampled law tests reach accepted shapes.

# LAW IK.32a
# promise: a shape the guard the shell calls accepts always has a hidden size of whole 128-channel blocks.
# harm: a later guard could drop the hidden-width check and hand the kernels a partial block
law ik_dims_guard_hidden:
  for +h: Nat
  for +nh: Nat
  for +kv: Nat
  for +e: Nat
  for +k: Nat
  for +f: Nat
  for +t: Nat
  {Bool.and(ik_dims_valid(h, nh, kv, 128n, 16n, Nat.mul(128n, 1n+e), k, Nat.mul(128n, 1n+f), t), Bool.not(ik_width_whole(h))) == False{} : Bool}

def ik_dims_guard_hidden(h, nh, kv, e, k, f, t):
  ikp_conj7_a(ik_width_whole(h), Bool.and(Nat.is_eq(128n, 128n), Nat.is_eq(16n, 16n)), ik_head_groups_valid(nh, kv), Bool.and(Nat.is_gt(Nat.mul(128n, 1n+e), 0n), Bool.and(Nat.is_eq(Nat.mod(Nat.mul(128n, 1n+e), 128n), 0n), Nat.is_le(Nat.mul(128n, 1n+e), 512n))), ik_top_k_valid(k, Nat.mul(128n, 1n+e)), ik_width_whole(Nat.mul(128n, 1n+f)), ik_taps_valid(t))

# LAW IK.32b
# promise: a shape the guard the shell calls accepts always has an expert width of whole 128-channel blocks.
# harm: a later guard could drop the expert-width check and hand the expert kernels a partial block
law ik_dims_guard_expert_width:
  for +g: Nat
  for +nh: Nat
  for +kv: Nat
  for +e: Nat
  for +k: Nat
  for +ff: Nat
  for +t: Nat
  {Bool.and(ik_dims_valid(Nat.mul(128n, 1n+g), nh, kv, 128n, 16n, Nat.mul(128n, 1n+e), k, ff, t), Bool.not(ik_width_whole(ff))) == False{} : Bool}

def ik_dims_guard_expert_width(g, nh, kv, e, k, ff, t):
  ikp_conj7_g(ik_width_whole(Nat.mul(128n, 1n+g)), Bool.and(Nat.is_eq(128n, 128n), Nat.is_eq(16n, 16n)), ik_head_groups_valid(nh, kv), Bool.and(Nat.is_gt(Nat.mul(128n, 1n+e), 0n), Bool.and(Nat.is_eq(Nat.mod(Nat.mul(128n, 1n+e), 128n), 0n), Nat.is_le(Nat.mul(128n, 1n+e), 512n))), ik_top_k_valid(k, Nat.mul(128n, 1n+e)), ik_width_whole(ff), ik_taps_valid(t))

# LAW IK.32c
# promise: a shape the guard the shell calls accepts always has grouped heads, a router selection within the experts and non-zero convolution taps.
# harm: a later guard could drop one of these checks and hand the kernels heads, experts or taps they do not cover
law ik_dims_guard_heads_router_taps:
  for +g: Nat
  for +nh: Nat
  for +kv: Nat
  for +e: Nat
  for +k: Nat
  for +f: Nat
  for +t: Nat
  {Bool.and(ik_dims_valid(Nat.mul(128n, 1n+g), nh, kv, 128n, 16n, Nat.mul(128n, 1n+e), k, Nat.mul(128n, 1n+f), t), Bool.not(Bool.and(ik_head_groups_valid(nh, kv), Bool.and(ik_top_k_valid(k, Nat.mul(128n, 1n+e)), ik_taps_valid(t))))) == False{} : Bool}

def ik_dims_guard_heads_router_taps(g, nh, kv, e, k, f, t):
  ikp_conj7_cft(ik_width_whole(Nat.mul(128n, 1n+g)), Bool.and(Nat.is_eq(128n, 128n), Nat.is_eq(16n, 16n)), ik_head_groups_valid(nh, kv), Bool.and(Nat.is_gt(Nat.mul(128n, 1n+e), 0n), Bool.and(Nat.is_eq(Nat.mod(Nat.mul(128n, 1n+e), 128n), 0n), Nat.is_le(Nat.mul(128n, 1n+e), 512n))), ik_top_k_valid(k, Nat.mul(128n, 1n+e)), ik_width_whole(Nat.mul(128n, 1n+f)), ik_taps_valid(t))

# (q / (1 + sp)) * (1 + sp) never exceeds q.
def ikp_divmul_le(+x: Nat, +sp: Nat) -> {Nat.is_le(Nat.mul(Nat.div(x, 1n+sp), 1n+sp), x) == True{} : Bool}:
  %L.nat_div_mod_r(x, sp) : {Nat.is_le(Nat.mul(Nat.div(x, 1n+sp), 1n+sp), _) == True{} : Bool}
  L.nat_add_le_r(Nat.mul(Nat.div(x, 1n+sp), 1n+sp), Nat.mod(x, 1n+sp))

def ikp_fit_le_s(+n: Nat, +sp: Nat, +p: Nat, +mx: Nat, hn: {Nat.is_le(n, mx) == True{} : Bool}, hq: {Nat.is_le(Nat.sub(p, 1n), Nat.div(Nat.sub(mx, n), 1n+sp)) == True{} : Bool}) -> {Nat.is_le(Nat.add(n, Nat.mul(Nat.sub(p, 1n), 1n+sp)), mx) == True{} : Bool}:
  %L.nat_sub_add_cancel(mx, n, hn) : {Nat.is_le(Nat.add(n, Nat.mul(Nat.sub(p, 1n), 1n+sp)), _) == True{} : Bool}
  %L.nat_add_comm(n, Nat.sub(mx, n)) : {Nat.is_le(Nat.add(n, Nat.mul(Nat.sub(p, 1n), 1n+sp)), _) == True{} : Bool}
  L.nat_add_le_mono(n, n, Nat.mul(Nat.sub(p, 1n), 1n+sp), Nat.sub(mx, n), L.nat_le_refl(n), L.nat_le_trans(Nat.mul(Nat.sub(p, 1n), 1n+sp), Nat.mul(Nat.div(Nat.sub(mx, n), 1n+sp), 1n+sp), Nat.sub(mx, n), L.nat_mul_le_mono_r(Nat.sub(p, 1n), Nat.div(Nat.sub(mx, n), 1n+sp), 1n+sp, hq), ikp_divmul_le(Nat.sub(mx, n), sp)))

# An accepted application count is n + (p - 1) * s <= mx.
def ikp_fit_le(+n: Nat, +s: Nat, +p: Nat, +mx: Nat, +h: {Bool.and(Nat.is_le(n, mx), ik_apps_fit_if(s, p, Nat.sub(mx, n))) == True{} : Bool}) -> {Nat.is_le(Nat.add(n, Nat.mul(Nat.sub(p, 1n), s)), mx) == True{} : Bool}:
  match s:
    case 0n:
      %Equal.sym(Nat, Nat.mul(Nat.sub(p, 1n), 0n), 0n, L.nat_mul_zero_r(Nat.sub(p, 1n))) : {Nat.is_le(Nat.add(n, _), mx) == True{} : Bool}
      %Equal.sym(Nat, Nat.add(n, 0n), n, L.nat_add_zero_r(n)) : {Nat.is_le(_, mx) == True{} : Bool}
      ikp_and_l(Nat.is_le(n, mx), True{}, h)
    case 1n+sp:
      ikp_fit_le_s(n, sp, p, mx, ikp_and_l(Nat.is_le(n, mx), Nat.is_le(Nat.sub(p, 1n), Nat.div(Nat.sub(mx, n), 1n+sp)), h), ikp_and_r(Nat.is_le(n, mx), Nat.is_le(Nat.sub(p, 1n), Nat.div(Nat.sub(mx, n), 1n+sp)), h))

def ikp_counts_apps(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +w: Nat, +nh: Nat, +ma: Nat, +mw: Nat, +mh: Nat, v: Bool, hv: {ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh) == v : Bool}) -> {Bool.and(v, Nat.is_lt(ma, ik_apps(n, ls, le, p))) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(ma, ik_apps(n, ls, le, p), ikp_fit_le(n, ik_span(ls, le), p, ma, ikp_and_l(ik_apps_fit(n, ls, le, p, ma), Bool.and(Nat.is_le(w, mw), Nat.is_le(nh, mh)), hv)))

# LAW IK.32b
# promise: a model the counts guard accepts runs no more layer applications per token than the limit the shell passes.
# harm: a pass count from a file could make the schedule overflow or allocate without bound
law ik_counts_apps_within:
  for +n: Nat
  for +ls: Nat
  for +le: Nat
  for +p: Nat
  for +w: Nat
  for +nh: Nat
  for +ma: Nat
  for +mw: Nat
  for +mh: Nat
  {Bool.and(ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh), Nat.is_lt(ma, ik_apps(n, ls, le, p))) == False{} : Bool}

def ik_counts_apps_within(n, ls, le, p, w, nh, ma, mw, mh):
  ikp_counts_apps(n, ls, le, p, w, nh, ma, mw, mh, ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh), {==})

def ikp_counts_window(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +w: Nat, +nh: Nat, +ma: Nat, +mw: Nat, +mh: Nat, v: Bool, hv: {ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh) == v : Bool}) -> {Bool.and(v, Nat.is_lt(mw, w)) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(mw, w, ikp_and_l(Nat.is_le(w, mw), Nat.is_le(nh, mh), ikp_and_r(ik_apps_fit(n, ls, le, p, ma), Bool.and(Nat.is_le(w, mw), Nat.is_le(nh, mh)), hv)))

# LAW IK.32c
# promise: a sliding window the counts guard accepts is within the limit the shell passes.
# harm: a window from a file could overflow the ring sizing arithmetic
law ik_counts_window_within:
  for +n: Nat
  for +ls: Nat
  for +le: Nat
  for +p: Nat
  for +w: Nat
  for +nh: Nat
  for +ma: Nat
  for +mw: Nat
  for +mh: Nat
  {Bool.and(ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh), Nat.is_lt(mw, w)) == False{} : Bool}

def ik_counts_window_within(n, ls, le, p, w, nh, ma, mw, mh):
  ikp_counts_window(n, ls, le, p, w, nh, ma, mw, mh, ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh), {==})

def ikp_counts_heads(+n: Nat, +ls: Nat, +le: Nat, +p: Nat, +w: Nat, +nh: Nat, +ma: Nat, +mw: Nat, +mh: Nat, v: Bool, hv: {ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh) == v : Bool}) -> {Bool.and(v, Nat.is_lt(mh, nh)) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(mh, nh, ikp_and_r(Nat.is_le(w, mw), Nat.is_le(nh, mh), ikp_and_r(ik_apps_fit(n, ls, le, p, ma), Bool.and(Nat.is_le(w, mw), Nat.is_le(nh, mh)), hv)))

# LAW IK.32d
# promise: a query-head count the counts guard accepts is within the limit the shell passes.
# harm: a head count from a file could overflow the projection width arithmetic
law ik_counts_heads_within:
  for +n: Nat
  for +ls: Nat
  for +le: Nat
  for +p: Nat
  for +w: Nat
  for +nh: Nat
  for +ma: Nat
  for +mw: Nat
  for +mh: Nat
  {Bool.and(ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh), Nat.is_lt(mh, nh)) == False{} : Bool}

def ik_counts_heads_within(n, ls, le, p, w, nh, ma, mw, mh):
  ikp_counts_heads(n, ls, le, p, w, nh, ma, mw, mh, ik_counts_valid(n, ls, le, p, w, nh, ma, mw, mh), {==})

def ikp_geom_vec(+w: Nat, +rows: Nat, hc: {Nat.is_ne(ik_ring_slots(w, rows), 0n) == True{} : Bool}) -> {Nat.is_lt(ik_ring_slots(w, rows), Nat.add(w, 64n)) == False{} : Bool}:
  %hc : {Bool.and(_, Nat.is_lt(ik_ring_slots(w, rows), Nat.add(w, 64n))) == False{} : Bool}
  ik_ring_holds_vector(w, rows)

def ikp_geom_c(+w: Nat, +e: Nat, +rows: Nat, he: {Nat.is_le(e, w) == True{} : Bool}, c: Bool, hc: {Nat.is_ne(ik_ring_slots(w, rows), 0n) == c : Bool}) -> {Bool.and(c, Nat.is_le(ik_ring_slots(w, rows), ik_bias_off(e))) == False{} : Bool}:
  match c:
    case False{}:
      {==}
    case True{}:
      L.nat_lt_le_false(ik_bias_off(e), ik_ring_slots(w, rows), L.nat_lt_le_trans(Nat.add(e, 63n), Nat.add(w, 64n), ik_ring_slots(w, rows), L.nat_le_lt_trans(Nat.add(e, 63n), Nat.add(w, 63n), Nat.add(w, 64n), L.nat_add_le_mono(e, w, 63n, 63n, he, L.nat_le_refl(63n)), ikp_add_lt_mono(w, 63n, 64n, {==})), L.nat_not_lt_imp_le(ik_ring_slots(w, rows), Nat.add(w, 64n), ikp_geom_vec(w, rows, hc))))

def ikp_geom(+w: Nat, +e: Nat, +rows: Nat, v: Bool, hv: {ik_attn_geom_valid(w, e) == v : Bool}) -> {Bool.and(v, Bool.and(Nat.is_ne(ik_ring_slots(w, rows), 0n), Nat.is_le(ik_ring_slots(w, rows), ik_bias_off(e)))) == False{} : Bool}:
  match v:
    case False{}:
      {==}
    case True{}:
      ikp_geom_c(w, e, rows, ikp_and_r(Nat.is_gt(e, 0n), Nat.is_le(e, w), ikp_and_r(Nat.is_gt(w, 0n), Bool.and(Nat.is_gt(e, 0n), Nat.is_le(e, w)), hv)), Nat.is_ne(ik_ring_slots(w, rows), 0n), {==})

# LAW IK.33
# promise: with a window and extent the guard accepts, every first-key distance the softmax kernel biases lies inside the K / V ring, so a wrapped slot is never biased as a near key.
law ik_bias_inside_ring:
  for +w: Nat
  for +e: Nat
  for +rows: Nat
  {Bool.and(ik_attn_geom_valid(w, e), Bool.and(Nat.is_ne(ik_ring_slots(w, rows), 0n), Nat.is_le(ik_ring_slots(w, rows), ik_bias_off(e)))) == False{} : Bool}

def ik_bias_inside_ring(w, e, rows):
  ikp_geom(w, e, rows, ik_attn_geom_valid(w, e), {==})

def ikp_vocab(g: Bool, +u: Nat, +v: Nat) -> {Bool.and(Bool.and(g, Nat.is_le(u, v)), Nat.is_lt(v, u)) == False{} : Bool}:
  match g:
    case False{}:
      {==}
    case True{}:
      ikp_le_lt_excl(u, v)

# LAW IK.34
# promise: a vocabulary the guard accepts emits only tokens that are rows of the embedding.
law ik_vocab_within_embedding:
  for +u: Nat
  for +v: Nat
  {Bool.and(ik_vocab_valid(u, v), Nat.is_lt(v, u)) == False{} : Bool}

def ik_vocab_within_embedding(u, v):
  ikp_vocab(Nat.is_gt(u, 0n), u, v)

# ============================================================================
# 7. Prefill tiles
# ============================================================================

def ik_tile_rows_fit(+rows: Nat, +padded: Nat, fits: Bool) -> Nat:
  match fits:
    case True{}:
      padded
    case False{}:
      rows

# Rows a prefill tile of `rows` prompt rows runs with: a power of two of at least 8, capped at the
# planned tile, so that a new prompt length compiles no new recipe (the rows past the prompt are
# computed and never read); the rows themselves when the padded tile would not fit the `room`
# of positions left in the context or when no such count holds them.
def ik_tile_rows(+rows: Nat, +tile: Nat, +room: Nat) -> Nat:
  ik_tile_rows_fit(
    rows,
    Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile),
    Bool.and(
      Nat.is_ge(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), rows),
      Nat.is_le(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), room)
    )
  )

# Prompt rows the next prefill tile carries: the rows left, at most the planned tile.
def ik_tile_take(+left: Nat, +tile: Nat) -> Nat:
  Nat.min(left, tile)

# The row of a tile whose final-norm output the readout reads: the tile's last prompt row.
def ik_tile_last_row(rows: Nat) -> Nat:
  Nat.sub(rows, 1n)

# ----------------------------------------------------------------------------
# 7a. Laws of the prefill tiles
# ----------------------------------------------------------------------------

def ikp_tile_holds(+rows: Nat, +pd: Nat, -room: Nat, f: Bool, hf: {Bool.and(Nat.is_ge(pd, rows), Nat.is_le(pd, room)) == f : Bool}) -> {Nat.is_le(rows, ik_tile_rows_fit(rows, pd, f)) == True{} : Bool}:
  match f:
    case True{}:
      ikp_ge_to_le(pd, rows, ikp_and_l(Nat.is_ge(pd, rows), Nat.is_le(pd, room), hf))
    case False{}:
      L.nat_le_refl(rows)

# LAW IK.35
# promise: a prefill tile never has fewer rows than the prompt rows it carries.
law ik_tile_holds_rows:
  for +rows: Nat
  for +tile: Nat
  for +room: Nat
  {Nat.is_le(rows, ik_tile_rows(rows, tile, room)) == True{} : Bool}

def ik_tile_holds_rows(rows, tile, room):
  ikp_tile_holds(rows, Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), room, Bool.and(Nat.is_ge(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), rows), Nat.is_le(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), room)), {==})

def ikp_tile_room(+rows: Nat, +pd: Nat, +room: Nat, f: Bool, hf: {Bool.and(Nat.is_ge(pd, rows), Nat.is_le(pd, room)) == f : Bool}) -> {Bool.and(Nat.is_le(rows, room), Nat.is_lt(room, ik_tile_rows_fit(rows, pd, f))) == False{} : Bool}:
  match f:
    case True{}:
      ikp_and_fr(Nat.is_le(rows, room), Nat.is_lt(room, pd), L.nat_le_lt_false(room, pd, ikp_and_r(Nat.is_ge(pd, rows), Nat.is_le(pd, room), hf)))
    case False{}:
      ikp_le_lt_excl(rows, room)

# LAW IK.36
# promise: a prefill tile of prompt rows that fit the room left in the context never has more rows than that room.
law ik_tile_within_room:
  for +rows: Nat
  for +tile: Nat
  for +room: Nat
  {Bool.and(Nat.is_le(rows, room), Nat.is_lt(room, ik_tile_rows(rows, tile, room))) == False{} : Bool}

def ik_tile_within_room(rows, tile, room):
  ikp_tile_room(rows, Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), room, Bool.and(Nat.is_ge(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), rows), Nat.is_le(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), room)), {==})

def ikp_tile_cap(+rows: Nat, +pd: Nat, -room: Nat, +tile: Nat, hp: {Nat.is_le(pd, tile) == True{} : Bool}, f: Bool) -> {Bool.and(Nat.is_lt(tile, ik_tile_rows_fit(rows, pd, f)), Nat.is_ne(ik_tile_rows_fit(rows, pd, f), rows)) == False{} : Bool}:
  match f:
    case True{}:
      ikp_and_fl(Nat.is_lt(tile, pd), Nat.is_ne(pd, rows), L.nat_le_lt_false(tile, pd, hp))
    case False{}:
      ikp_and_fr(Nat.is_lt(tile, rows), Nat.is_ne(rows, rows), ikp_ne_of_eq(rows, rows, {==}))

# LAW IK.37
# promise: a prefill tile has more rows than the planned tile only when it is exactly the prompt rows it carries (no padding past the plan).
law ik_tile_within_plan:
  for +rows: Nat
  for +tile: Nat
  for +room: Nat
  {Bool.and(Nat.is_lt(tile, ik_tile_rows(rows, tile, room)), Nat.is_ne(ik_tile_rows(rows, tile, room), rows)) == False{} : Bool}

def ik_tile_within_plan(rows, tile, room):
  ikp_tile_cap(rows, Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), room, tile, L.nat_min_le_r(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), Bool.and(Nat.is_ge(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), rows), Nat.is_le(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), room)))

# LAW IK.38
# promise: the row the prefill readout reads is a row of the last tile (its last prompt row).
law ik_tile_readout_inside:
  for +rp: Nat
  for +tile: Nat
  for +room: Nat
  {Nat.is_lt(ik_tile_last_row(1n+rp), ik_tile_rows(1n+rp, tile, room)) == True{} : Bool}

def ik_tile_readout_inside(rp, tile, room):
  %Equal.sym(Nat, Nat.sub(rp, 0n), rp, L.nat_sub_zero_r(rp)) : {Nat.is_lt(_, ik_tile_rows(1n+rp, tile, room)) == True{} : Bool}
  L.nat_lt_le_trans(rp, 1n+rp, ik_tile_rows(1n+rp, tile, room), L.nat_lt_succ_self(rp), ik_tile_holds_rows(1n+rp, tile, room))

# LAW IK.39
# promise: the next prefill tile carries no more prompt rows than are left and than the planned tile holds.
law ik_tile_take_bounded:
  for +left: Nat
  for +tile: Nat
  {Bool.and(Nat.is_le(ik_tile_take(left, tile), left), Nat.is_le(ik_tile_take(left, tile), tile)) == True{} : Bool}

def ik_tile_take_bounded(left, tile):
  ikp_and_tt(Nat.is_le(Nat.min(left, tile), left), Nat.is_le(Nat.min(left, tile), tile), L.nat_min_le_l(left, tile), L.nat_min_le_r(left, tile))

# LAW IK.40
# promise: while prompt rows are left and the planned tile is not empty, the next prefill tile carries at least one row (prefill makes progress).
law ik_tile_take_progress:
  for +lp: Nat
  for +tp: Nat
  {Nat.is_gt(ik_tile_take(1n+lp, 1n+tp), 0n) == True{} : Bool}

def ik_tile_take_progress(lp, tp):
  {==}

# ----------------------------------------------------------------------------
# 8. Row buffers of a prefill tile
# ----------------------------------------------------------------------------

# A row buffer may take application `a`'s output when the source it holds is last read before `a`.
def ik_buf_reusable(last_use: Nat, a: Nat) -> Bool:
  Nat.is_lt(last_use, a)

# Application `a` would overwrite a source that is still read: source `s`, already written
# (`s <= a`), lives in the buffer `a` writes and is read by `a` or later (its last read is not before `a`).
def ik_buf_clobbers(buf_s: Nat, buf_out: Nat, +s: Nat, +a: Nat, last_use: Nat) -> Bool:
  Bool.and(Nat.is_eq(buf_s, buf_out), Bool.and(Nat.is_le(s, a), Nat.is_ge(last_use, a)))

def ikp_buf(+lu: Nat, +a: Nat, +bs: Nat, +bo: Nat, +s: Nat, r: Bool, hr: {Nat.is_lt(lu, a) == r : Bool}) -> {Bool.and(r, ik_buf_clobbers(bs, bo, s, a, lu)) == False{} : Bool}:
  match r:
    case False{}:
      {==}
    case True{}:
      ikp_and_fr(Nat.is_eq(bs, bo), Bool.and(Nat.is_le(s, a), Nat.is_ge(lu, a)), ikp_and_fr(Nat.is_le(s, a), Nat.is_ge(lu, a), ikp_lt_not_ge(lu, a, hr)))

# LAW IK.41
# promise: a buffer the reuse rule hands to an application never holds a source that application or a later one still reads.
law ik_buf_reuse_safe:
  for +lu: Nat
  for +a: Nat
  for +bs: Nat
  for +bo: Nat
  for +s: Nat
  {Bool.and(ik_buf_reusable(lu, a), ik_buf_clobbers(bs, bo, s, a, lu)) == False{} : Bool}

def ik_buf_reuse_safe(lu, a, bs, bo, s):
  ikp_buf(lu, a, bs, bo, s, Nat.is_lt(lu, a), {==})

# ============================================================================
# 9. Whether a request's decode runs the drafter
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
