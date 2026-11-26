# Launch binding invariants (`synapse/recipe_cache.rs::verify_launch_bindings`):
#   - Any tensor with address 0 or `addr < offset_in_section` is invalid (`lb_addr_valid`).
#   - A tensor's section base is `addr - offset_in_section` (`lb_section_base`), valid without
#     underflow only when `lb_addr_valid(addr, offset)` holds.
#   - Persistent sections and non-zero section bases are in 1-to-1 correspondence (`lb_pair_ok`):
#     no two distinct sections may alias one HBM base (`synLaunch` status 30), and two tensors
#     in the same persistent section must agree on the exact same base address.
#   - A recipe with `ws_size > 0` requires a non-zero workspace address (`lb_ws_ok`).

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

def lb_ws_ok(+ws_size: Nat, +ws_addr: Nat) -> Bool:
  Bool.or(Nat.is_eq(ws_size, 0n), Nat.is_gt(ws_addr, 0n))

def lb_addr_valid(+addr: Nat, +offset: Nat) -> Bool:
  Bool.and(Nat.is_gt(addr, 0n), Nat.is_le(offset, addr))

def lb_section_base(addr: Nat, offset: Nat) -> Nat:
  Nat.sub(addr, offset)

def lb_bool_eq(+a: Bool, +b: Bool) -> Bool:
  match a b:
    case True{} True{}:
      True{}
    case False{} False{}:
      True{}
    case _ _:
      False{}

def lb_pair_ok(+sec_a: Nat, +base_a: Nat, +sec_b: Nat, +base_b: Nat) -> Bool:
  lb_bool_eq(Nat.is_eq(sec_a, sec_b), Nat.is_eq(base_a, base_b))

def lb_eq_refl(+a: Nat) -> {True{} == Nat.is_eq(a, a) : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+ap:
      lb_eq_refl(ap)

def lb_eq_add_gt_false(+a: Nat, extra: Nat) -> {False{} == Nat.is_eq(a, Nat.add(a, 1n+extra)) : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+ap:
      lb_eq_add_gt_false(ap, extra)

# LAW LB.1: A zero HBM address is never accepted for any tensor binding.
# promise: a zero HBM address is always rejected
# catches: Nat.is_gt(addr, 0n) check removed from lb_addr_valid
law lb_zero_addr_invalid:
  for +off: Nat
  {lb_addr_valid(0n, off) == False{} : Bool}

def lb_zero_addr_invalid(off):
  {==}

# LAW LB.2: Binding an address below `offset_in_section` is always rejected rather than underflowing `sec_base`.
# promise: any binding with addr < offset_in_section is always rejected
# catches: Nat.is_le(offset, addr) check dropped from lb_addr_valid
law lb_underflow_addr_invalid:
  for +addr: Nat
  for +extra: Nat
  {lb_addr_valid(addr, Nat.add(addr, 1n+extra)) == False{} : Bool}

def lb_underflow_addr_invalid(addr, extra):
  match addr:
    case 0n:
      {==}
    case 1n+ap:
      G.g_nat_le_add_succ_false(1n+ap, extra)

# LAW LB.3: Any non-zero HBM address at or above `offset_in_section` (including `offset == 0` and `addr == offset_in_section`) is valid.
# promise: any positive address >= offset_in_section (including offset == 0 and offset == addr) satisfies lb_addr_valid
# catches: lb_addr_valid checks offset > 0 instead of addr > 0, or checks strict offset < addr
law lb_valid_addr_accepted:
  for +a_m1: Nat
  for +off: Nat
  {lb_addr_valid(1n+a_m1, Nat.min(off, 1n+a_m1)) == True{} : Bool}

def lb_valid_addr_accepted(a_m1, off):
  L.nat_min_le_r(off, 1n+a_m1)

# LAW LB.4: Two distinct persistent sections pointing to the same base address are always rejected.
# promise: distinct sections sharing the same base address are always rejected
# catches: lb_pair_ok only checks section equality when bases differ, missing cross-section aliasing
law lb_diff_section_same_base_rejected:
  for +sec_a: Nat
  for +extra: Nat
  for +base: Nat
  {lb_pair_ok(sec_a, base, Nat.add(sec_a, 1n+extra), base) == False{} : Bool}

def lb_diff_section_same_base_rejected(sec_a, extra, base):
  %lb_eq_add_gt_false(sec_a, extra) : {lb_bool_eq(_, Nat.is_eq(base, base)) == False{} : Bool}
  %lb_eq_refl(base) : {lb_bool_eq(False{}, _) == False{} : Bool}
  {==}

# LAW LB.5: Two tensors in the same persistent section with different base addresses are always rejected.
# promise: two tensors in the same persistent section with different computed base addresses are rejected
# catches: same-section base mismatch ignored (only section_by_base checked, not base_by_section)
law lb_same_section_diff_base_rejected:
  for +sec: Nat
  for +base_a: Nat
  for +extra: Nat
  {lb_pair_ok(sec, base_a, sec, Nat.add(base_a, 1n+extra)) == False{} : Bool}

def lb_same_section_diff_base_rejected(sec, base_a, extra):
  %lb_eq_refl(sec) : {lb_bool_eq(_, Nat.is_eq(base_a, Nat.add(base_a, 1n+extra))) == False{} : Bool}
  %lb_eq_add_gt_false(base_a, extra) : {lb_bool_eq(True{}, _) == False{} : Bool}
  {==}

# LAW LB.6: Two tensors in the same persistent section sharing the same base address are accepted.
# promise: two tensors sharing both section id and base address are accepted
# catches: lb_pair_ok rejects all pairs unconditionally or forbids shared section slices
law lb_same_section_same_base_ok:
  for +sec: Nat
  for +base: Nat
  {lb_pair_ok(sec, base, sec, base) == True{} : Bool}

def lb_same_section_same_base_ok(sec, base):
  %lb_eq_refl(sec) : {lb_bool_eq(_, Nat.is_eq(base, base)) == True{} : Bool}
  %lb_eq_refl(base) : {lb_bool_eq(True{}, _) == True{} : Bool}
  {==}

# LAW LB.7: Non-zero workspace size with a zero workspace address is always rejected.
# promise: non-zero workspace size requires a non-zero workspace address
# catches: lb_ws_ok ignores ws_addr or checks ws_size == 0 wrong way around
law lb_nonzero_ws_needs_addr:
  for +ws_m1: Nat
  {lb_ws_ok(1n+ws_m1, 0n) == False{} : Bool}

def lb_nonzero_ws_needs_addr(ws_m1):
  {==}

# LAW LB.8: For any binding at or above offset_in_section, lb_section_base recovers the exact base address without underflow.
# promise: lb_section_base recovers the exact section base whenever addr >= offset
# catches: lb_section_base adds addr and offset instead of subtracting offset
law lb_section_base_exact:
  for +base: Nat
  for +off: Nat
  {lb_section_base(Nat.add(base, off), off) == base : Nat}

def lb_section_base_exact(base, off):
  L.nat_add_sub_cancel_r(base, off)

# LAW LB.9: Distinct sections sharing one base are rejected whichever section comes first.
# `verify_launch_bindings` passes the section seen first, and recipe tensors are not sorted by section.
# promise: distinct sections sharing the same base address are rejected in either argument order
# catches: section equality in lb_pair_ok replaced by Nat.is_le(sec_b, sec_a), which LB.4 to LB.6 still accept
law lb_diff_section_same_base_rejected_rev:
  for +sec_b: Nat
  for +extra: Nat
  for +base: Nat
  {lb_pair_ok(Nat.add(sec_b, 1n+extra), base, sec_b, base) == False{} : Bool}

def lb_diff_section_same_base_rejected_rev(sec_b, extra, base):
  %Equal.sym(Bool, Nat.is_eq(Nat.add(sec_b, 1n+extra), sec_b), False{}, G.g_nat_eq_add_succ_false(sec_b, extra)) : {lb_bool_eq(_, Nat.is_eq(base, base)) == False{} : Bool}
  %lb_eq_refl(base) : {lb_bool_eq(False{}, _) == False{} : Bool}
  {==}

# LAW LB.10: Two tensors of one section with different bases are rejected whichever base comes first.
# promise: two tensors in the same persistent section with different base addresses are rejected in either order
# catches: base equality in lb_pair_ok replaced by Nat.is_le(base_b, base_a), which LB.4 to LB.6 still accept
law lb_same_section_diff_base_rejected_rev:
  for +sec: Nat
  for +base_b: Nat
  for +extra: Nat
  {lb_pair_ok(sec, Nat.add(base_b, 1n+extra), sec, base_b) == False{} : Bool}

def lb_same_section_diff_base_rejected_rev(sec, base_b, extra):
  %lb_eq_refl(sec) : {lb_bool_eq(_, Nat.is_eq(Nat.add(base_b, 1n+extra), base_b)) == False{} : Bool}
  %Equal.sym(Bool, Nat.is_eq(Nat.add(base_b, 1n+extra), base_b), False{}, G.g_nat_eq_add_succ_false(base_b, extra)) : {lb_bool_eq(True{}, _) == False{} : Bool}
  {==}
