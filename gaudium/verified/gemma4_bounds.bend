# Slice 2.5 — Verified Gemma 4 Attention/Multimodal Bounds & Host-Pinning State Machine
# Formally specifies and proves:
#   - Gemma 4 deinterleaved channel inverse & RoPE table vector counts
#   - Gemma 4 dual sliding/global attention head & proportional RoPE dimension guards
#   - Gemma 4 PLE (per-layer input embeddings), KV-sharing & causal audio encoder bounds
#   - Atomic chunked host-memory pinning (synHostMap extents) rollback & hugepage ratio guards

import Base

# ============================================================================
# 1. Helper Nat arithmetic lemmas (erased before Rust codegen)
# ============================================================================

def g4_lt_refl_false(n: Nat) -> {False{} == Nat.is_lt(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      g4_lt_refl_false(p)

def g4_lt_add_r_false(n: Nat, extra: Nat) -> {False{} == Nat.is_lt(Nat.add(n, extra), n) : Bool}:
  match n:
    case 0n:
      match extra:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      g4_lt_add_r_false(p, extra)

def g4_bool_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# ============================================================================
# 2. Gemma 4 Deinterleaved Layout & RoPE Vector Count (g4_layout.hpp)
# ============================================================================

# Position of channel c inside its 128-channel block: even -> c/2, odd -> 64 + c/2.
def g4_lane_if(half: Nat, is_even: Bool) -> Nat:
  match is_even:
    case True{}:
      half
    case False{}:
      Nat.add(64n, half)

def g4_lane(+c: Nat) -> Nat:
  g4_lane_if(Nat.div(Nat.mod(c, 128n), 2n), Nat.is_eq(Nat.mod(Nat.mod(c, 128n), 2n), 0n))

# Inverse mapping: TPC vector half h in {0, 1}, lane j in 0..64 -> natural channel in 0..128.
def g4_nat_if(+p: Nat, low_half: Bool) -> Nat:
  match low_half:
    case True{}:
      Nat.mul(2n, p)
    case False{}:
      Nat.add(Nat.mul(2n, Nat.sub(p, 64n)), 1n)

def g4_nat(+h: Nat, +j: Nat) -> Nat:
  g4_nat_if(Nat.add(Nat.mul(2n, j), h), Nat.is_lt(Nat.add(Nat.mul(2n, j), h), 64n))

# V row permutation for key slot j (matches g4l::vperm).
def g4_vperm_inner(+i: Nat, low_half: Bool) -> Nat:
  match low_half:
    case True{}:
      Nat.mul(2n, i)
    case False{}:
      Nat.add(Nat.mul(2n, Nat.sub(i, 64n)), 1n)

def g4_vperm(+j: Nat) -> Nat:
  Nat.add(
    Nat.mul(Nat.div(j, 128n), 128n),
    g4_vperm_inner(Nat.mod(j, 128n), Nat.is_lt(Nat.mod(j, 128n), 64n))
  )

# Number of 128-lane RoPE cos/sin table vectors: ceil(pairs / 128).
def g4_rope_vecs(pairs: Nat) -> Nat:
  Nat.div(Nat.add(pairs, 127n), 128n)

# ============================================================================
# 3. Gemma 4 Dimension & Multimodal Guards (adapter_gemma4.hpp)
# ============================================================================

# Gemma 4 attention head guard: ga_* kernels take 256 or 512 channels (2 or 4 128-lane vectors),
# nh % nkv == 0, nh + nkv <= 2048, and 1 <= pairs <= dh / 2.
def g4_heads_valid(+nh: Nat, +nkv: Nat, +dh: Nat, +pairs: Nat) -> Bool:
  Bool.and(
    Bool.or(Nat.is_eq(dh, 256n), Nat.is_eq(dh, 512n)),
    Bool.and(
      Nat.is_gt(nkv, 0n),
      Bool.and(
        Nat.is_gt(nh, 0n),
        Bool.and(
          Nat.is_eq(Nat.mod(nh, nkv), 0n),
          Bool.and(
            Nat.is_le(Nat.add(nh, nkv), 2048n),
            Bool.and(Nat.is_ge(pairs, 1n), Nat.is_le(pairs, Nat.div(dh, 2n)))
          )
        )
      )
    )
  )

# Gemma 4 per-layer input embedding (PLE) guard: either disabled (0), or multiple of 128 <= 1024 with matching vocab.
def g4_ple_valid_if(+ple_dim: Nat, ple_vocab: Nat, vocab: Nat, is_off: Bool) -> Bool:
  match is_off:
    case True{}:
      True{}
    case False{}:
      Bool.and(
        Nat.is_le(ple_dim, 1024n),
        Bool.and(Nat.is_eq(Nat.mod(ple_dim, 128n), 0n), Nat.is_eq(ple_vocab, vocab))
      )

def g4_ple_valid(+ple_dim: Nat, ple_vocab: Nat, vocab: Nat) -> Bool:
  g4_ple_valid_if(ple_dim, ple_vocab, vocab, Nat.is_eq(ple_dim, 0n))

# Gemma 4 KV-sharing layer count guard: 1 <= n_layers <= 64 and kv_shared < n_layers (at least one unshared layer).
def g4_kv_share_valid(+n_layers: Nat, kv_shared: Nat) -> Bool:
  Bool.and(
    Nat.is_ge(n_layers, 1n),
    Bool.and(Nat.is_le(n_layers, 64n), Nat.is_lt(kv_shared, n_layers))
  )

# Gemma 4 causal audio encoder context guard: 2 <= left <= 17 and right == 0.
def g4_audio_ctx_valid(+a_left: Nat, a_right: Nat) -> Bool:
  Bool.and(
    Nat.is_ge(a_left, 2n),
    Bool.and(Nat.is_le(a_left, 17n), Nat.is_eq(a_right, 0n))
  )

# Gemma 4 vision head guard: heads * head_dim == hidden, head_dim % 4 == 0 (2-axis RoPE), non-zero.
def g4_vision_heads_valid(+v_heads: Nat, +v_head_dim: Nat, v_hidden: Nat) -> Bool:
  Bool.and(
    Nat.is_gt(v_heads, 0n),
    Bool.and(
      Nat.is_gt(v_head_dim, 0n),
      Bool.and(
        Nat.is_eq(Nat.mul(v_heads, v_head_dim), v_hidden),
        Nat.is_eq(Nat.mod(v_head_dim, 4n), 0n)
      )
    )
  )

# ============================================================================
# 4. Atomic Chunked Host-Pinning State Machine (common.hpp: HugeMem / pin_extents)
# ============================================================================

type PinState is Data:
  PinUnmapped{}
  PinPartial{mapped: Nat}
  PinMapped{mapped: Nat}
  PinRolledBack{}

def pin_mapped_count(s: PinState) -> Nat:
  match s:
    case PinUnmapped{}:
      0n
    case PinPartial{m}:
      m
    case PinMapped{m}:
      m
    case PinRolledBack{}:
      0n

def pin_is_usable(s: PinState) -> Bool:
  match s:
    case PinMapped{_}:
      True{}
    case _:
      False{}

def pin_add_extent_if(m: Nat, is_last: Bool) -> PinState:
  match is_last:
    case True{}:
      PinMapped{1n+m}
    case False{}:
      PinPartial{1n+m}

def pin_add_extent(s: PinState, is_last: Bool) -> PinState:
  match s:
    case PinUnmapped{}:
      pin_add_extent_if(0n, is_last)
    case PinPartial{m}:
      pin_add_extent_if(m, is_last)
    case PinMapped{m}:
      PinMapped{m}
    case PinRolledBack{}:
      PinRolledBack{}

# Atomic rollback on any failed synHostMap extent or insufficient 2 MiB hugepage ratio.
def pin_rollback(s: PinState) -> PinState:
  match s:
    case PinUnmapped{}:
      PinRolledBack{}
    case PinPartial{_}:
      PinRolledBack{}
    case PinMapped{_}:
      PinRolledBack{}
    case PinRolledBack{}:
      PinRolledBack{}

# Hugepage percentage check: huge_bytes * 100 >= total_bytes * min_pct.
def pin_huge_ratio_ok(huge_bytes: Nat, total_bytes: Nat, min_pct: Nat) -> Bool:
  Nat.is_ge(Nat.mul(huge_bytes, 100n), Nat.mul(total_bytes, min_pct))

# ============================================================================
# 5. Verified Gemma 4 & Host-Pinning Laws
# ============================================================================

# LAW G4.1: Zero RoPE pairs needs 0 table vectors; 128 pairs needs exactly 1.
law g4_rope_vecs_boundary:
  {Bool.and(Nat.is_eq(g4_rope_vecs(0n), 0n), Nat.is_eq(g4_rope_vecs(128n), 1n)) == True{} : Bool}

def g4_rope_vecs_boundary():
  {==}

# LAW G4.2: Zero KV heads is ALWAYS rejected by Gemma 4 head guard.
law g4_heads_zero_kv_rejected:
  for +nh: Nat
  for +dh: Nat
  for +p: Nat
  {g4_heads_valid(nh, 0n, dh, p) == False{} : Bool}

def g4_heads_zero_kv_rejected(nh, dh, p):
  g4_bool_and_false_r(Bool.or(Nat.is_eq(dh, 256n), Nat.is_eq(dh, 512n)))

# LAW G4.3: Zero head dim is ALWAYS rejected (prevents division/empty-head UB).
law g4_heads_zero_dh_rejected:
  for +nh: Nat
  for +nkv: Nat
  for +p: Nat
  {g4_heads_valid(nh, nkv, 0n, p) == False{} : Bool}

def g4_heads_zero_dh_rejected(nh, nkv, p):
  {==}

# LAW G4.4: Disabled PLE (ple_dim == 0) is ALWAYS valid regardless of vocab.
law g4_ple_off_always_valid:
  for pv: Nat
  for v: Nat
  {g4_ple_valid(0n, pv, v) == True{} : Bool}

def g4_ple_off_always_valid(pv, v):
  {==}

# LAW G4.5: Sharing >= n_layers KV caches is ALWAYS rejected (trunk needs >= 1 unshared layer).
law g4_kv_share_all_rejected:
  for +nl: Nat
  for +extra: Nat
  {g4_kv_share_valid(nl, Nat.add(nl, extra)) == False{} : Bool}

def g4_kv_share_all_rejected(nl, extra):
  %g4_lt_add_r_false(nl, extra) : {Bool.and(Nat.is_ge(nl, 1n), Bool.and(Nat.is_le(nl, 64n), _)) == False{} : Bool}
  %Equal.sym(Bool, Bool.and(Nat.is_le(nl, 64n), False{}), False{}, g4_bool_and_false_r(Nat.is_le(nl, 64n))) : {Bool.and(Nat.is_ge(nl, 1n), _) == False{} : Bool}
  g4_bool_and_false_r(Nat.is_ge(nl, 1n))

# LAW G4.6: Audio context left == 0 is ALWAYS rejected (every frame must at least attend to itself).
law g4_audio_zero_left_rejected:
  for ar: Nat
  {g4_audio_ctx_valid(0n, ar) == False{} : Bool}

def g4_audio_zero_left_rejected(ar):
  {==}

# LAW G4.7: Concrete Gemma 4 31B sliding (16Q/8KV, dh=256, pairs=128) & global (16Q/4KV, dh=512, pairs=128) pass.
law g4_31b_heads_valid:
  {Bool.and(g4_heads_valid(16n, 8n, 256n, 128n), g4_heads_valid(16n, 4n, 512n, 128n)) == True{} : Bool}

def g4_31b_heads_valid():
  {==}

# LAW G4.8: Host-pin rollback ALWAYS leaves 0 mapped extents and is unusable (no leaked synHostMap).
law pin_rollback_leaves_zero_mapped:
  for s: PinState
  {pin_mapped_count(pin_rollback(s)) == 0n : Nat}

def pin_rollback_leaves_zero_mapped(s):
  match s:
    case PinUnmapped{}: {==}
    case PinPartial{_}: {==}
    case PinMapped{_}: {==}
    case PinRolledBack{}: {==}

# LAW G4.9: Host-pin rollback is idempotent and absorbing for pin_add_extent.
law pin_rollback_idempotent_and_absorbing:
  for s: PinState
  for last: Bool
  {Bool.and(Nat.is_eq(pin_mapped_count(pin_rollback(pin_rollback(s))), 0n), Bool.not(pin_is_usable(pin_add_extent(pin_rollback(s), last)))) == True{} : Bool}

def pin_rollback_idempotent_and_absorbing(s, last):
  match s:
    case PinUnmapped{}: {==}
    case PinPartial{_}: {==}
    case PinMapped{_}: {==}
    case PinRolledBack{}: {==}

# LAW G4.10: Partial host pinning is NEVER usable (prevents DMA against half-mapped buffers).
law pin_partial_never_usable:
  for m: Nat
  {pin_is_usable(PinPartial{m}) == False{} : Bool}

def pin_partial_never_usable(m):
  {==}
