# Slice 2B — Verified HBM Pre-Allocation Planning & 5-Category Ledger Accounting
# Formally specifies and proves:
#   - Usable HBM capacity clamping, exact fit/shortfall partitioning, context-first prefix-slot priority
#   - 5-category byte-exact HBM allocation ledger conservation

import Base

# ============================================================================
# 1. Helper Nat arithmetic lemmas (erased before Rust codegen)
# ============================================================================

def hb_add_zero(a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %hb_add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def hb_add_succ_r(a: Nat, b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %hb_add_succ_r(p, b) : {1n+Nat.add(p, 1n+b) == 1n+_ : Nat}
      {==}

def hb_add_assoc(a: Nat, b: Nat, c: Nat) -> {Nat.add(Nat.add(a, b), c) == Nat.add(a, Nat.add(b, c)) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %hb_add_assoc(p, b, c) : {1n+Nat.add(Nat.add(p, b), c) == 1n+_ : Nat}
      {==}

def hb_add_comm(+a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  match a:
    case 0n:
      Equal.sym(Nat, Nat.add(b, 0n), b, hb_add_zero(b))
    case 1n+p:
      %Equal.sym(Nat, Nat.add(b, 1n+p), 1n+Nat.add(b, p), hb_add_succ_r(b, p)) : {1n+Nat.add(p, b) == _ : Nat}
      %hb_add_comm(p, b) : {1n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

def hb_sub_zero(a: Nat) -> {Nat.sub(a, 0n) == a : Nat}:
  match a:
    case 0n: {==}
    case 1n+_: {==}

def hb_sub_ge_zero(t: Nat, extra: Nat) -> {Nat.sub(t, Nat.add(t, extra)) == 0n : Nat}:
  match t:
    case 0n: {==}
    case 1n+p:
      hb_sub_ge_zero(p, extra)

def hb_le_refl(n: Nat) -> {True{} == Nat.is_le(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      hb_le_refl(p)

def hb_le_step_r_zero(xp: Nat, h: {False{} == True{} : Bool}) -> {Nat.is_le(xp, 0n) == True{} : Bool}:
  match xp:
    case 0n: {==}
    case 1n+_: h

def hb_le_step_r(x: Nat, y: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(x, 1n+y) == True{} : Bool}:
  match x y:
    case 0n _:
      {==}
    case 1n+xp 0n:
      hb_le_step_r_zero(xp, h)
    case 1n+xp 1n+yp:
      hb_le_step_r(xp, yp, h)

def hb_sub_le_self(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      Equal.sym(Bool, True{}, Nat.is_le(ap, ap), hb_le_refl(ap))
    case 1n+ap 1n+bp:
      hb_le_step_r(Nat.sub(ap, bp), ap, hb_sub_le_self(ap, bp))

def hb_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      hb_min_le_l(ap, bp)

def hb_gt_add_succ_true(u: Nat, extra: Nat) -> {True{} == Nat.is_gt(Nat.add(u, 1n+extra), u) : Bool}:
  match u:
    case 0n: {==}
    case 1n+p:
      hb_gt_add_succ_true(p, extra)

# ============================================================================
# 2. HBM Capacity, Fit / Shortfall & 5-Category Ledger
# ============================================================================

type HbmCat is Data:
  CatWeights{}
  CatKv{}
  CatCheckpoints{}
  CatScratch{}
  CatOther{}

type HbmLedger is Data:
  HbmLedger{weights: Nat, kv: Nat, checkpoints: Nat, scratch: Nat, other: Nat}

type HbmFit is Data:
  HbmFits{margin: Nat}
  HbmShort{shortfall: Nat}

# Usable HBM: total - (driver_base + spare), saturating safely at 0 (no underflow).
def hbm_usable(total: Nat, driver_base: Nat, spare: Nat) -> Nat:
  Nat.sub(total, Nat.add(driver_base, spare))

def hbm_check_fit_if(+planned: Nat, +usable: Nat, fits: Bool) -> HbmFit:
  match fits:
    case True{}:
      HbmFits{Nat.sub(usable, planned)}
    case False{}:
      HbmShort{Nat.sub(planned, usable)}

def hbm_check_fit(+planned: Nat, +usable: Nat) -> HbmFit:
  hbm_check_fit_if(planned, usable, Nat.is_le(planned, usable))

def hbm_fit_ok(f: HbmFit) -> Bool:
  match f:
    case HbmFits{_}:
      True{}
    case HbmShort{_}:
      False{}

# Reconstruct exact conservation across fit/shortfall:
#   HbmFits{m}  => planned + m == usable
#   HbmShort{s} => usable + s == planned
def hbm_fit_reconstruct_lhs(f: HbmFit, planned: Nat, usable: Nat) -> Nat:
  match f:
    case HbmFits{m}:
      Nat.add(planned, m)
    case HbmShort{s}:
      Nat.add(usable, s)

def hbm_fit_reconstruct_rhs(f: HbmFit, planned: Nat, usable: Nat) -> Nat:
  match f:
    case HbmFits{_}:
      usable
    case HbmShort{_}:
      planned

def hbm_ledger_empty() -> HbmLedger:
  HbmLedger{0n, 0n, 0n, 0n, 0n}

def hbm_ledger_add(l: HbmLedger, cat: HbmCat, bytes: Nat) -> HbmLedger:
  match l:
    case HbmLedger{w, kv, ck, sc, ot}:
      match cat:
        case CatWeights{}:
          HbmLedger{Nat.add(w, bytes), kv, ck, sc, ot}
        case CatKv{}:
          HbmLedger{w, Nat.add(kv, bytes), ck, sc, ot}
        case CatCheckpoints{}:
          HbmLedger{w, kv, Nat.add(ck, bytes), sc, ot}
        case CatScratch{}:
          HbmLedger{w, kv, ck, Nat.add(sc, bytes), ot}
        case CatOther{}:
          HbmLedger{w, kv, ck, sc, Nat.add(ot, bytes)}

def hbm_ledger_total(l: HbmLedger) -> Nat:
  match l:
    case HbmLedger{w, kv, ck, sc, ot}:
      Nat.add(w, Nat.add(kv, Nat.add(ck, Nat.add(sc, ot))))

def hbm_ledger_matches(l: HbmLedger, w: Nat, kv: Nat, ck: Nat, sc: Nat, ot: Nat) -> Bool:
  match l:
    case HbmLedger{lw, lkv, lck, lsc, lot}:
      Bool.and(
        Nat.is_eq(lw, w),
        Bool.and(
          Nat.is_eq(lkv, kv),
          Bool.and(
            Nat.is_eq(lck, ck),
            Bool.and(Nat.is_eq(lsc, sc), Nat.is_eq(lot, ot))
          )
        )
      )

# Context ALWAYS comes before prefix-reuse snapshot slots (hbm_plan.hpp).
def hbm_grant_prefix_if(+req_slots: Nat, slot_bytes: Nat, base_planned: Nat, usable: Nat, refuse: Bool) -> Nat:
  match refuse:
    case True{}:
      0n
    case False{}:
      Nat.min(req_slots, Nat.div(Nat.sub(usable, base_planned), slot_bytes))

def hbm_grant_prefix_slots(+req_slots: Nat, +slot_bytes: Nat, +base_planned: Nat, +usable: Nat) -> Nat:
  hbm_grant_prefix_if(
    req_slots,
    slot_bytes,
    base_planned,
    usable,
    Bool.or(Nat.is_gt(base_planned, usable), Nat.is_eq(slot_bytes, 0n))
  )

# ============================================================================
# 3. Verified HBM Budget & Ledger Laws
# ============================================================================

# LAW HB.1: When driver overhead alone >= total HBM, usable HBM is 0 (no underflow).
law hbm_usable_zero_when_exceeded:
  for +t: Nat
  for +extra: Nat
  for +spare: Nat
  {hbm_usable(t, Nat.add(t, extra), spare) == 0n : Nat}

def hbm_usable_zero_when_exceeded(t, extra, spare):
  %Equal.sym(Nat, Nat.add(Nat.add(t, extra), spare), Nat.add(t, Nat.add(extra, spare)), hb_add_assoc(t, extra, spare)) : {Nat.sub(t, _) == 0n : Nat}
  hb_sub_ge_zero(t, Nat.add(extra, spare))

# LAW HB.2: Usable HBM NEVER exceeds total HBM.
law hbm_usable_le_total:
  for +t: Nat
  for +b: Nat
  for +s: Nat
  {Nat.is_le(hbm_usable(t, b, s), t) == True{} : Bool}

def hbm_usable_le_total(t, b, s):
  hb_sub_le_self(t, Nat.add(b, s))

# Helper proof for fit/shortfall exact partition:
def hb_fit_step_eq(
  +pp: Nat,
  +up: Nat,
  fits: Bool,
  ih: {hbm_fit_reconstruct_lhs(hbm_check_fit_if(pp, up, fits), pp, up) == hbm_fit_reconstruct_rhs(hbm_check_fit_if(pp, up, fits), pp, up) : Nat}
) -> {hbm_fit_reconstruct_lhs(hbm_check_fit_if(1n+pp, 1n+up, fits), 1n+pp, 1n+up) == hbm_fit_reconstruct_rhs(hbm_check_fit_if(1n+pp, 1n+up, fits), 1n+pp, 1n+up) : Nat}:
  match fits:
    case True{}:
      %ih : {1n+Nat.add(pp, Nat.sub(up, pp)) == 1n+_ : Nat}
      {==}
    case False{}:
      %ih : {1n+Nat.add(up, Nat.sub(pp, up)) == 1n+_ : Nat}
      {==}

def hb_fit_partition_aux(+p: Nat, +u: Nat) -> {hbm_fit_reconstruct_lhs(hbm_check_fit_if(p, u, Nat.is_le(p, u)), p, u) == hbm_fit_reconstruct_rhs(hbm_check_fit_if(p, u, Nat.is_le(p, u)), p, u) : Nat}:
  match p u:
    case 0n 0n:
      {==}
    case 0n 1n+up:
      {==}
    case 1n+pp 0n:
      {==}
    case 1n+pp 1n+up:
      hb_fit_step_eq(pp, up, Nat.is_le(pp, up), hb_fit_partition_aux(pp, up))

# LAW HB.3: Fit/shortfall strictly partitions planned vs usable bytes with zero rounding loss.
law hbm_fit_partition_exact:
  for +p: Nat
  for +u: Nat
  {hbm_fit_reconstruct_lhs(hbm_check_fit(p, u), p, u) == hbm_fit_reconstruct_rhs(hbm_check_fit(p, u), p, u) : Nat}

def hbm_fit_partition_exact(p, u):
  hb_fit_partition_aux(p, u)

# Helper: pull +b from inside any nested Nat.add to the right
def hb_pull_r(+x: Nat, +y: Nat, +b: Nat) -> {Nat.add(Nat.add(x, b), y) == Nat.add(Nat.add(x, y), b) : Nat}:
  %Equal.sym(Nat, Nat.add(Nat.add(x, b), y), Nat.add(x, Nat.add(b, y)), hb_add_assoc(x, b, y)) : {_ == Nat.add(Nat.add(x, y), b) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(x, y), b), Nat.add(x, Nat.add(y, b)), hb_add_assoc(x, y, b)) : {Nat.add(x, Nat.add(b, y)) == _ : Nat}
  %hb_add_comm(b, y) : {Nat.add(x, Nat.add(b, y)) == Nat.add(x, _) : Nat}
  {==}

def hb_lift_add_r(+k: Nat, +s: Nat, +t: Nat, +b: Nat, e: {s == Nat.add(t, b) : Nat}) -> {Nat.add(k, s) == Nat.add(Nat.add(k, t), b) : Nat}:
  %Equal.sym(Nat, Nat.add(Nat.add(k, t), b), Nat.add(k, Nat.add(t, b)), hb_add_assoc(k, t, b)) : {Nat.add(k, s) == _ : Nat}
  %e : {Nat.add(k, s) == Nat.add(k, _) : Nat}
  {==}

# LAW HB.4: Adding `bytes` to ANY of the 5 HBM ledger categories increases total by EXACTLY `bytes`.
law hbm_ledger_add_conserved:
  for +w: Nat
  for +kv: Nat
  for +ck: Nat
  for +sc: Nat
  for +ot: Nat
  for cat: HbmCat
  for +b: Nat
  {hbm_ledger_total(hbm_ledger_add(HbmLedger{w, kv, ck, sc, ot}, cat, b)) == Nat.add(hbm_ledger_total(HbmLedger{w, kv, ck, sc, ot}), b) : Nat}

def hbm_ledger_add_conserved(w, kv, ck, sc, ot, cat, b):
  match cat:
    case CatWeights{}:
      hb_pull_r(w, Nat.add(kv, Nat.add(ck, Nat.add(sc, ot))), b)
    case CatKv{}:
      hb_lift_add_r(
        w,
        Nat.add(Nat.add(kv, b), Nat.add(ck, Nat.add(sc, ot))),
        Nat.add(kv, Nat.add(ck, Nat.add(sc, ot))),
        b,
        hb_pull_r(kv, Nat.add(ck, Nat.add(sc, ot)), b)
      )
    case CatCheckpoints{}:
      hb_lift_add_r(
        w,
        Nat.add(kv, Nat.add(Nat.add(ck, b), Nat.add(sc, ot))),
        Nat.add(kv, Nat.add(ck, Nat.add(sc, ot))),
        b,
        hb_lift_add_r(
          kv,
          Nat.add(Nat.add(ck, b), Nat.add(sc, ot)),
          Nat.add(ck, Nat.add(sc, ot)),
          b,
          hb_pull_r(ck, Nat.add(sc, ot), b)
        )
      )
    case CatScratch{}:
      hb_lift_add_r(
        w,
        Nat.add(kv, Nat.add(ck, Nat.add(Nat.add(sc, b), ot))),
        Nat.add(kv, Nat.add(ck, Nat.add(sc, ot))),
        b,
        hb_lift_add_r(
          kv,
          Nat.add(ck, Nat.add(Nat.add(sc, b), ot)),
          Nat.add(ck, Nat.add(sc, ot)),
          b,
          hb_lift_add_r(
            ck,
            Nat.add(Nat.add(sc, b), ot),
            Nat.add(sc, ot),
            b,
            hb_pull_r(sc, ot, b)
          )
        )
      )
    case CatOther{}:
      hb_lift_add_r(
        w,
        Nat.add(kv, Nat.add(ck, Nat.add(sc, Nat.add(ot, b)))),
        Nat.add(kv, Nat.add(ck, Nat.add(sc, ot))),
        b,
        hb_lift_add_r(
          kv,
          Nat.add(ck, Nat.add(sc, Nat.add(ot, b))),
          Nat.add(ck, Nat.add(sc, ot)),
          b,
          hb_lift_add_r(
            ck,
            Nat.add(sc, Nat.add(ot, b)),
            Nat.add(sc, ot),
            b,
            Equal.sym(Nat, Nat.add(Nat.add(sc, ot), b), Nat.add(sc, Nat.add(ot, b)), hb_add_assoc(sc, ot, b))
          )
        )
      )

def hb_zero_le(n: Nat) -> {Nat.is_le(0n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+_: {==}

# Helper for hbm_grant_prefix_if <= req_slots
def hb_grant_le_req_if(+req: Nat, sb: Nat, bp: Nat, u: Nat, refuse: Bool) -> {Nat.is_le(hbm_grant_prefix_if(req, sb, bp, u, refuse), req) == True{} : Bool}:
  match refuse:
    case True{}:
      hb_zero_le(req)
    case False{}:
      hb_min_le_l(req, Nat.div(Nat.sub(u, bp), sb))

# LAW HB.5: Granted prefix-reuse slots NEVER exceed requested slots.
law hbm_prefix_slots_bounded_by_req:
  for +req: Nat
  for +sb: Nat
  for +bp: Nat
  for +u: Nat
  {Nat.is_le(hbm_grant_prefix_slots(req, sb, bp, u), req) == True{} : Bool}

def hbm_prefix_slots_bounded_by_req(req, sb, bp, u):
  hb_grant_le_req_if(req, sb, bp, u, Bool.or(Nat.is_gt(bp, u), Nat.is_eq(sb, 0n)))

# LAW HB.6: If base context exceeds usable HBM, prefix-reuse slots granted == 0 (context always wins).
law hbm_prefix_slots_zero_when_context_over:
  for +req: Nat
  for +sb: Nat
  for +u: Nat
  for +extra: Nat
  {hbm_grant_prefix_slots(req, sb, Nat.add(u, 1n+extra), u) == 0n : Nat}

def hbm_prefix_slots_zero_when_context_over(req, sb, u, extra):
  %hb_gt_add_succ_true(u, extra) : {hbm_grant_prefix_if(req, sb, Nat.add(u, 1n+extra), u, Bool.or(_, Nat.is_eq(sb, 0n))) == 0n : Nat}
  {==}

def hb_or_true_r_zero(+req: Nat, bp: Nat, u: Nat, b: Bool) -> {hbm_grant_prefix_if(req, 0n, bp, u, Bool.or(b, True{})) == 0n : Nat}:
  match b:
    case True{}: {==}
    case False{}: {==}

# LAW HB.7: Zero slot_bytes never divides by zero (grants 0 slots safely).
law hbm_prefix_slots_zero_bytes_safe:
  for +req: Nat
  for +bp: Nat
  for +u: Nat
  {hbm_grant_prefix_slots(req, 0n, bp, u) == 0n : Nat}

def hbm_prefix_slots_zero_bytes_safe(req, bp, u):
  hb_or_true_r_zero(req, bp, u, Nat.is_gt(bp, u))
