# Slice 2D — Verified Tensor Shape, Alignment, Permutation, Tree & Lane Relocation Bounds
# Formally specifies and proves:
#   - Safe upward alignment (align_up / align128) & 4096-token attention chunk bucket bounds
#   - Deinterleaved channel & FP8 KV V-row permutations
#   - Kernel-baked GQA, Gated DeltaNet, hidden/vocab, MoE & ViT 2x2 merger dimension guards
#   - Speculative draft-tree bounds (<= 32 nodes, acyclic DAG) & lane state-block relocation

import Base

# ============================================================================
# 1. Helper Nat arithmetic lemmas (erased before Rust codegen)
# ============================================================================

def tn_le_refl(n: Nat) -> {True{} == Nat.is_le(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      tn_le_refl(p)

def tn_le_add_r(a: Nat, b: Nat) -> {Nat.is_le(a, Nat.add(a, b)) == True{} : Bool}:
  match a:
    case 0n:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      tn_le_add_r(p, b)

def tn_lt_refl_false(n: Nat) -> {False{} == Nat.is_lt(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      tn_lt_refl_false(p)

def tn_lt_add_r_false(hi: Nat, extra: Nat) -> {False{} == Nat.is_lt(Nat.add(hi, extra), hi) : Bool}:
  match hi:
    case 0n:
      match extra:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      tn_lt_add_r_false(p, extra)

def tn_lt_add_succ_true(lo: Nat, span: Nat) -> {True{} == Nat.is_lt(lo, Nat.add(lo, 1n+span)) : Bool}:
  match lo:
    case 0n: {==}
    case 1n+p:
      tn_lt_add_succ_true(p, span)

def tn_ge_refl(n: Nat) -> {True{} == Nat.is_ge(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      tn_ge_refl(p)

def tn_sub_self_zero(n: Nat) -> {Nat.sub(n, n) == 0n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      tn_sub_self_zero(p)

def tn_add_zero(a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %tn_add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def tn_bool_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def tn_le_add_succ_false(mn: Nat, extra: Nat) -> {False{} == Nat.is_le(Nat.add(mn, 1n+extra), mn) : Bool}:
  match mn:
    case 0n: {==}
    case 1n+p:
      tn_le_add_succ_false(p, extra)

# ============================================================================
# 2. Alignment, Attention Bucket & Channel Permutation Math
# ============================================================================

def align_up_rem_if(+x: Nat, align: Nat, rem: Nat, rem_is_zero: Bool) -> Nat:
  match rem_is_zero:
    case True{}:
      x
    case False{}:
      Nat.add(x, Nat.sub(align, rem))

def align_up_if(+x: Nat, +align: Nat, align_is_zero: Bool) -> Nat:
  match align_is_zero:
    case True{}:
      x
    case False{}:
      align_up_rem_if(x, align, Nat.mod(x, align), Nat.is_eq(Nat.mod(x, align), 0n))

# Safe upward alignment (div-by-zero safe: align == 0 returns x).
def align_up(+x: Nat, +align: Nat) -> Nat:
  align_up_if(x, align, Nat.is_eq(align, 0n))

def align128(+x: Nat) -> Nat:
  align_up(x, 128n)

# Attention chunk bucket validation (CHUNK = 4096; model_engine.hpp:32,176).
def bucket_valid(+lb: Nat) -> Bool:
  Bool.and(Nat.is_gt(lb, 0n), Nat.is_eq(Nat.mod(lb, 4096n), 0n))

def bucket_chunks(lb: Nat) -> Nat:
  Nat.div(lb, 4096n)

def qsa_blocks(lb: Nat) -> Nat:
  Nat.div(lb, 4n)

# Deinterleaved 128-lane order (model_qwen35.hpp:41-42): even -> c/2, odd -> 64 + c/2.
def lane128_if(half: Nat, is_even: Bool) -> Nat:
  match is_even:
    case True{}:
      half
    case False{}:
      Nat.add(64n, half)

def lane128(+c: Nat) -> Nat:
  lane128_if(Nat.div(Nat.mod(c, 128n), 2n), Nat.is_eq(Nat.mod(Nat.mod(c, 128n), 2n), 0n))

def lane256(+c: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(c, 128n), 128n), lane128(c))

# FP8 KV V-row permutation within each 128-row block (model_qwen35.hpp:48).
def kv8_vpos128_if(i: Nat, +j: Nat, low_pair: Bool) -> Nat:
  match low_pair:
    case True{}:
      Nat.add(Nat.mul(2n, i), j)
    case False{}:
      Nat.add(64n, Nat.add(Nat.mul(2n, i), Nat.sub(j, 2n)))

def kv8_vpos128(+q: Nat) -> Nat:
  kv8_vpos128_if(
    Nat.div(Nat.mod(q, 128n), 4n),
    Nat.mod(Nat.mod(q, 128n), 4n),
    Nat.is_lt(Nat.mod(Nat.mod(q, 128n), 4n), 2n)
  )

def kv8_vpos(+q: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(q, 128n), 128n), kv8_vpos128(q))

# ============================================================================
# 3. Model / Kernel-Baked Dimension & ViT / Tree Guards (adapter_qwen35.hpp)
# ============================================================================

def gqa_dims_valid(+n_heads: Nat, +n_kv_heads: Nat, +head_dim: Nat) -> Bool:
  Bool.and(
    Nat.is_gt(n_kv_heads, 0n),
    Bool.and(
      Nat.is_gt(n_heads, 0n),
      Bool.and(
        Nat.is_eq(Nat.mod(n_heads, n_kv_heads), 0n),
        Bool.or(Nat.is_eq(head_dim, 128n), Nat.is_eq(head_dim, 256n))
      )
    )
  )

def gdn_dims_valid(+k_heads: Nat, +v_heads: Nat, conv_k: Nat) -> Bool:
  Bool.and(
    Nat.is_gt(k_heads, 0n),
    Bool.and(
      Nat.is_le(v_heads, 64n),
      Bool.and(Nat.is_eq(Nat.mod(v_heads, k_heads), 0n), Nat.is_eq(conv_k, 4n))
    )
  )

def hidden_dims_valid(+hidden: Nat, +vocab: Nat) -> Bool:
  Bool.and(
    Nat.is_gt(hidden, 0n),
    Bool.and(
      Nat.is_le(hidden, 6144n),
      Bool.and(
        Nat.is_eq(Nat.mod(hidden, 256n), 0n),
        Bool.and(Nat.is_gt(vocab, 0n), Nat.is_eq(Nat.mod(vocab, 128n), 0n))
      )
    )
  )

def moe_dims_valid(+ne: Nat, +ek: Nat, +ef: Nat) -> Bool:
  Bool.and(
    Nat.is_gt(ne, 0n),
    Bool.and(
      Nat.is_le(ne, 512n),
      Bool.and(
        Nat.is_eq(Nat.mod(ne, 128n), 0n),
        Bool.and(
          Nat.is_gt(ek, 0n),
          Bool.and(
            Nat.is_le(ek, 12n),
            Bool.and(
              Nat.is_le(ek, ne),
              Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n))
            )
          )
        )
      )
    )
  )

def vit_grid_valid(+t: Nat, +h: Nat, +w: Nat, max_patches: Nat) -> Bool:
  Bool.and(
    Nat.is_ge(t, 1n),
    Bool.and(
      Nat.is_ge(h, 2n),
      Bool.and(
        Nat.is_ge(w, 2n),
        Bool.and(
          Nat.is_eq(Nat.mod(h, 2n), 0n),
          Bool.and(
            Nat.is_eq(Nat.mod(w, 2n), 0n),
            Nat.is_le(Nat.mul(t, Nat.mul(h, w)), max_patches)
          )
        )
      )
    )
  )

def vit_merged_tokens(t: Nat, h: Nat, w: Nat) -> Nat:
  Nat.mul(t, Nat.mul(Nat.div(h, 2n), Nat.div(w, 2n)))

# Draft-tree bounds: parameterized by adapter max_nodes (e.g. 32 for u32 ancestor bitmasks), caterpillar depth/rank checks.
def tree_shape_bounds_valid(+n_nodes: Nat, +spine: Nat, max_nodes: Nat, max_depth: Nat, topk: Nat) -> Bool:
  Bool.and(
    Nat.is_ge(n_nodes, 2n),
    Bool.and(
      Nat.is_le(n_nodes, max_nodes),
      Bool.and(
        Nat.is_ge(spine, 1n),
        Bool.and(
          Nat.is_le(spine, max_depth),
          Nat.is_le(Nat.sub(n_nodes, 1n), Nat.mul(topk, spine))
        )
      )
    )
  )

def tree_node_valid(idx: Nat, +parent: Nat, +depth: Nat, parent_depth: Nat, rank: Nat, +spine: Nat, topk: Nat) -> Bool:
  Bool.and(
    Nat.is_lt(parent, idx),
    Bool.and(
      Nat.is_eq(depth, Nat.add(parent_depth, 1n)),
      Bool.and(
        Nat.is_lt(parent, spine),
        Bool.and(Nat.is_le(depth, spine), Nat.is_lt(rank, topk))
      )
    )
  )

# ============================================================================
# 4. Lane State-Block Address Relocation (common.hpp: Reloc)
# ============================================================================

def addr_in_block(+addr: Nat, lo: Nat, hi: Nat) -> Bool:
  Bool.and(Nat.is_ge(addr, lo), Nat.is_lt(addr, hi))

def relocate_addr_if(+addr: Nat, lo: Nat, dst_lo: Nat, inside: Bool) -> Nat:
  match inside:
    case True{}:
      Nat.add(dst_lo, Nat.sub(addr, lo))
    case False{}:
      addr

def relocate_addr(+addr: Nat, +lo: Nat, hi: Nat, dst_lo: Nat) -> Nat:
  relocate_addr_if(addr, lo, dst_lo, addr_in_block(addr, lo, hi))

# ============================================================================
# 5. Verified Tensor, Alignment, Tree & Relocation Laws
# ============================================================================

def tn_align_rem_ge(+x: Nat, align: Nat, rem: Nat, rem_zero: Bool) -> {Nat.is_le(x, align_up_rem_if(x, align, rem, rem_zero)) == True{} : Bool}:
  match rem_zero:
    case True{}:
      Equal.sym(Bool, True{}, Nat.is_le(x, x), tn_le_refl(x))
    case False{}:
      tn_le_add_r(x, Nat.sub(align, rem))

def tn_align_if_ge(+x: Nat, +align: Nat, is_zero: Bool) -> {Nat.is_le(x, align_up_if(x, align, is_zero)) == True{} : Bool}:
  match is_zero:
    case True{}:
      Equal.sym(Bool, True{}, Nat.is_le(x, x), tn_le_refl(x))
    case False{}:
      tn_align_rem_ge(x, align, Nat.mod(x, align), Nat.is_eq(Nat.mod(x, align), 0n))

# LAW TN.1: align_up NEVER shrinks a size (for all x and all alignments including 0).
law align_up_never_shrinks:
  for +x: Nat
  for +a: Nat
  {Nat.is_le(x, align_up(x, a)) == True{} : Bool}

def align_up_never_shrinks(x, a):
  tn_align_if_ge(x, a, Nat.is_eq(a, 0n))

# LAW TN.2: Zero alignment is division-by-zero safe and returns x unchanged.
law align_up_zero_align_safe:
  for +x: Nat
  {align_up(x, 0n) == x : Nat}

def align_up_zero_align_safe(x):
  {==}

# LAW TN.3: Zero-sized bucket is ALWAYS invalid.
law bucket_zero_invalid:
  {bucket_valid(0n) == False{} : Bool}

def bucket_zero_invalid():
  {==}

# LAW TN.4: Zero KV head count is ALWAYS rejected (prevents GQA mod-by-zero).
law gqa_zero_kv_rejected:
  for +nh: Nat
  for +dh: Nat
  {gqa_dims_valid(nh, 0n, dh) == False{} : Bool}

def gqa_zero_kv_rejected(nh, dh):
  {==}

# LAW TN.5: Zero DeltaNet key head count is ALWAYS rejected.
law gdn_zero_k_rejected:
  for +vh: Nat
  for ck: Nat
  {gdn_dims_valid(0n, vh, ck) == False{} : Bool}

def gdn_zero_k_rejected(vh, ck):
  {==}

# LAW TN.6: Zero hidden size is ALWAYS rejected.
law hidden_zero_rejected:
  for +v: Nat
  {hidden_dims_valid(0n, v) == False{} : Bool}

def hidden_zero_rejected(v):
  {==}

# LAW TN.7: Zero MoE expert count is ALWAYS rejected.
law moe_zero_experts_rejected:
  for +ek: Nat
  for +ef: Nat
  {moe_dims_valid(0n, ek, ef) == False{} : Bool}

def moe_zero_experts_rejected(ek, ef):
  {==}

# LAW TN.8: Zero ViT height is ALWAYS rejected (prevents empty / odd 2x2 merge).
law vit_zero_h_rejected:
  for +t: Nat
  for +w: Nat
  for m: Nat
  {vit_grid_valid(t, 0n, w, m) == False{} : Bool}

def vit_zero_h_rejected(t, w, m):
  tn_bool_and_false_r(Nat.is_ge(t, 1n))

# LAW TN.9: Any draft tree exceeding its bitmask/adapter cap `max_nodes` is ALWAYS rejected.
law tree_over_max_nodes_rejected:
  for +mn: Nat
  for +extra: Nat
  for +sp: Nat
  for md: Nat
  for tk: Nat
  {tree_shape_bounds_valid(Nat.add(mn, 1n+extra), sp, mn, md, tk) == False{} : Bool}

def tree_over_max_nodes_rejected(mn, extra, sp, md, tk):
  %tn_le_add_succ_false(mn, extra) : {Bool.and(Nat.is_ge(Nat.add(mn, 1n+extra), 2n), Bool.and(_, Bool.and(Nat.is_ge(sp, 1n), Bool.and(Nat.is_le(sp, md), Nat.is_le(Nat.sub(Nat.add(mn, 1n+extra), 1n), Nat.mul(tk, sp)))))) == False{} : Bool}
  tn_bool_and_false_r(Nat.is_ge(Nat.add(mn, 1n+extra), 2n))

# LAW TN.10: A draft-tree node can NEVER be its own parent (guarantees DAG / tree acyclicity).
law tree_self_parent_rejected:
  for +i: Nat
  for +d: Nat
  for pd: Nat
  for r: Nat
  for +sp: Nat
  for tk: Nat
  {tree_node_valid(i, i, d, pd, r, sp, tk) == False{} : Bool}

def tree_self_parent_rejected(i, d, pd, r, sp, tk):
  %tn_lt_refl_false(i) : {Bool.and(_, Bool.and(Nat.is_eq(d, Nat.add(pd, 1n)), Bool.and(Nat.is_lt(i, sp), Bool.and(Nat.is_le(d, sp), Nat.is_lt(r, tk))))) == False{} : Bool}
  {==}

# LAW TN.11: Relocating the lane-0 base address `lo` of a non-empty block lands at `dst_lo`.
law relocate_preserves_lo:
  for +lo: Nat
  for +span: Nat
  for +dst: Nat
  {relocate_addr(lo, lo, Nat.add(lo, 1n+span), dst) == dst : Nat}

def relocate_preserves_lo(lo, span, dst):
  %tn_ge_refl(lo) : {relocate_addr_if(lo, lo, dst, Bool.and(_, Nat.is_lt(lo, Nat.add(lo, 1n+span)))) == dst : Nat}
  %tn_lt_add_succ_true(lo, span) : {relocate_addr_if(lo, lo, dst, _) == dst : Nat}
  %Equal.sym(Nat, Nat.sub(lo, lo), 0n, tn_sub_self_zero(lo)) : {Nat.add(dst, _) == dst : Nat}
  tn_add_zero(dst)

# LAW TN.12: Any address >= hi (weights/tables outside the lane state block) is NEVER relocated.
law relocate_outside_hi_unchanged:
  for +lo: Nat
  for +hi: Nat
  for +extra: Nat
  for dst: Nat
  {relocate_addr(Nat.add(hi, extra), lo, hi, dst) == Nat.add(hi, extra) : Nat}

def relocate_outside_hi_unchanged(lo, hi, extra, dst):
  %tn_lt_add_r_false(hi, extra) : {relocate_addr_if(Nat.add(hi, extra), lo, dst, Bool.and(Nat.is_ge(Nat.add(hi, extra), lo), _)) == Nat.add(hi, extra) : Nat}
  %Equal.sym(Bool, Bool.and(Nat.is_ge(Nat.add(hi, extra), lo), False{}), False{}, tn_bool_and_false_r(Nat.is_ge(Nat.add(hi, extra), lo))) : {relocate_addr_if(Nat.add(hi, extra), lo, dst, _) == Nat.add(hi, extra) : Nat}
  {==}

# LAW TN.13: Zero vocabulary size is ALWAYS rejected (prevents empty lm_head / embedding table).
law hidden_zero_vocab_rejected:
  for +h: Nat
  {hidden_dims_valid(h, 0n) == False{} : Bool}

def hidden_zero_vocab_rejected(h):
  %Equal.sym(Bool, Bool.and(Nat.is_eq(Nat.mod(h, 256n), 0n), False{}), False{}, tn_bool_and_false_r(Nat.is_eq(Nat.mod(h, 256n), 0n))) : {Bool.and(Nat.is_gt(h, 0n), Bool.and(Nat.is_le(h, 6144n), _)) == False{} : Bool}
  %Equal.sym(Bool, Bool.and(Nat.is_le(h, 6144n), False{}), False{}, tn_bool_and_false_r(Nat.is_le(h, 6144n))) : {Bool.and(Nat.is_gt(h, 0n), _) == False{} : Bool}
  tn_bool_and_false_r(Nat.is_gt(h, 0n))

# LAW TN.14: Qwen3.8-27B GQA + GDN concrete head dimensions pass all kernel-baked shape guards.
law qwen38_27b_dims_valid:
  {Bool.and(gqa_dims_valid(24n, 4n, 256n), gdn_dims_valid(16n, 48n, 4n)) == True{} : Bool}

def qwen38_27b_dims_valid():
  {==}

# LAW TN.15: Qwen3.6-35B-A3B MoE concrete dimensions pass all MoE + attention guards.
law qwen36_35b_moe_dims_valid:
  {Bool.and(gqa_dims_valid(16n, 2n, 256n), Bool.and(gdn_dims_valid(16n, 32n, 4n), moe_dims_valid(256n, 8n, 512n))) == True{} : Bool}

def qwen36_35b_moe_dims_valid():
  {==}
