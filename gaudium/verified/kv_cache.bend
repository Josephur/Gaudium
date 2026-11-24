# K/V cache element type (BF16 or FP8) and the FP8 scale rules of the trunk's attention.
#
# Formally specifies and proves:
#   - The bytes of one cache element and of one position of one layer's cache, per dtype
#   - Which caches take the configured dtype (those whose graphs have the FP8 attention kernels
#     and a scale table: the trunk's) and which stay BF16
#   - The block the V rows of a cache are permuted in, per dtype
#   - Which head shapes the FP8 attention kernels take (one FP8 vector per cache row, the scale
#     table's group slots)
#   - The power-of-two multiplier that maps a tensor into the FP8 range: its scaled maximum stays
#     below the format's top binade (nothing is clipped), and it never exceeds its cap
#   - The one V multiplier of a layer: never above what any of its heads allows
#   - Which decode forms may run on FP8 K/V (the non-speculative K=0 form always may)
#   - The row order of the V projection per dtype (the FP8 V rows follow the softmax's P bytes)
#
# Multipliers are powers of two, written as f32 exponent fields: a field e stands for 2^(e - 127).
# A tensor whose largest magnitude has exponent field `a`, scaled by 2^(e - 127), has its largest
# magnitude at exponent field a + e - 127; the FP8 format's largest finite value has exponent field
# f (Gaudi 2 E4M3: 240 = 1.875 * 2^7, f = 134). With top = f + 127, a + e <= top - margin and
# margin >= 1 puts the scaled maximum below 2^(f - 127), which the format holds without saturation
# (still rounded to its 3 mantissa bits). f is a parameter of the rules, not a constant of them:
# 134 holds for the default exponent bias of the format.

import Base
import nat_lemmas.bend as L
import tensor_bounds.bend as T

# ============================================================================
# 1. Helper lemmas (erased before Rust codegen)
# ============================================================================

def kc_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      kc_min_le_l(ap, bp)

def kc_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      kc_min_le_r(ap, bp)

# a <= b and b < a never hold together
def kc_le_lt_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_le(a, b), Nat.is_lt(b, a)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      kc_le_lt_excl(ap, bp)

# Bool.and(x, False) is False for every x
def kc_and_false_r(+x: Bool) -> {Bool.and(x, False{}) == False{} : Bool}:
  match x:
    case True{}: {==}
    case False{}: {==}

# x and (y and (z and w)) with z and w exclusive is False
def kc_and3_excl(+x: Bool, +y: Bool, +z: Bool, +w: Bool, h: {Bool.and(z, w) == False{} : Bool}) -> {Bool.and(Bool.and(x, Bool.and(y, z)), w) == False{} : Bool}:
  match x y:
    case False{} _: {==}
    case True{} False{}: {==}
    case True{} True{}: h

# four true conditions make a true conjunction
def kc_and4(+w: Bool, +x: Bool, +y: Bool, +z: Bool, hw: {w == True{} : Bool}, hx: {x == True{} : Bool}, hy: {y == True{} : Bool}, hz: {z == True{} : Bool}) -> {Bool.and(Bool.and(w, x), Bool.and(y, z)) == True{} : Bool}:
  match w:
    case False{}:
      L.bool_absurd({Bool.and(Bool.and(False{}, x), Bool.and(y, z)) == True{} : Bool}, hw)
    case True{}:
      match x:
        case False{}:
          L.bool_absurd({Bool.and(Bool.and(True{}, False{}), Bool.and(y, z)) == True{} : Bool}, hx)
        case True{}:
          match y:
            case False{}:
              L.bool_absurd({Bool.and(Bool.and(True{}, True{}), Bool.and(False{}, z)) == True{} : Bool}, hy)
            case True{}:
              hz

# a + min(c, u - a) <= u whenever a <= u
def kc_fit_le(+a: Nat, +u: Nat, +c: Nat) -> {Bool.or(Nat.is_lt(u, a), Nat.is_le(Nat.add(a, Nat.min(c, Nat.sub(u, a))), u)) == True{} : Bool}:
  match a u:
    case 0n 0n:
      match c:
        case 0n: {==}
        case 1n+_: {==}
    case 0n 1n+up:
      kc_min_le_r(c, 1n+up)
    case 1n+_ 0n:
      {==}
    case 1n+ap 1n+up:
      kc_fit_le(ap, up, c)

# ============================================================================
# 2. Element type, cache bytes and which caches take it
# ============================================================================

type KvDtype is Data:
  KvBf16{}
  KvFp8{}

# Bytes of one cache element.
def kv_elem_bytes(d: KvDtype) -> Nat:
  match d:
    case KvBf16{}:
      2n
    case KvFp8{}:
      1n

def kv_is_fp8(d: KvDtype) -> Bool:
  match d:
    case KvBf16{}:
      False{}
    case KvFp8{}:
      True{}

# Bytes of one position of one layer's K (or V) cache: `heads` rows of `width` elements.
def kv_row_bytes(+heads: Nat, +width: Nat, d: KvDtype) -> Nat:
  Nat.mul(Nat.mul(heads, width), kv_elem_bytes(d))

# The dtype of one attention cache. Only a cache whose graphs have the FP8 attention kernels and
# a scale table (`fp8_reader`: today the trunk's) takes the configured dtype; every other one (an
# MTP head's own cache) is read by BF16 graphs and stays BF16.
def kv_layer_dtype(fp8_reader: Bool, cfg: KvDtype) -> KvDtype:
  match fp8_reader:
    case True{}:
      cfg
    case False{}:
      KvBf16{}

# Rows of the blocks a cache's V rows are permuted in: the BF16 caches inside 128-row blocks (the
# softmax's interleaving BF16 conversion of P), the FP8 caches inside 256-row blocks (one FP8
# vector of P per 256 positions). A copy of whole rows ranges (the host-RAM tier) moves whole
# blocks, or V rows of neighbouring ranges mix.
def kv_vperm_rows(d: KvDtype) -> Nat:
  match d:
    case KvBf16{}:
      128n
    case KvFp8{}:
      256n

# ============================================================================
# 3. The FP8 attention kernels' shape rule
# ============================================================================

# One cache row of a head is one FP8 vector (`head_dim == row`, the kernels' vector width), and
# the scale table keeps `slots` entries per kind, one per kv group: 1 <= nkv <= slots.
def kv8_shape_ok(+head_dim: Nat, +nkv: Nat, +row: Nat, +slots: Nat) -> Bool:
  Bool.and(Nat.is_eq(head_dim, row), Bool.and(Nat.is_gt(nkv, 0n), Nat.is_le(nkv, slots)))

# ============================================================================
# 4. Power-of-two multipliers
# ============================================================================

# The multiplier exponent for a tensor whose largest magnitude has exponent field `a`: the largest
# e with a + e <= top - margin, and at most `cap` (a tensor measured as all zero, or tiny, is not
# blown up past it).
def kv8_fit(+a: Nat, +top: Nat, +margin: Nat, +cap: Nat) -> Nat:
  Nat.min(cap, Nat.sub(Nat.sub(top, margin), a))

# The one V multiplier of a layer from its heads' own: every group's P keeps the same multiplier
# because the kernels' softmax statistics carry one P x V scale per layer.
def kv8_min4(+a: Nat, +b: Nat, +c: Nat, +d: Nat) -> Nat:
  Nat.min(Nat.min(a, b), Nat.min(c, d))

# ============================================================================
# 5. Decode forms on FP8 K/V
# ============================================================================

# What a decode step reads the trunk's cache with: the non-speculative step (`KdPlain`), the
# chain verify of a drafter's drafts (`KdChain`), a draft-tree verify (`KdTree`), or a form that
# has not run on FP8 K/V (`KdOther`).
type KvDecode is Data:
  KdPlain{}
  KdChain{}
  KdTree{}
  KdOther{}

# Whether a decode form may run on FP8 K/V. The non-speculative form is the exactness reference
# of every speculative one and always may. A speculative form may once it has run token-identical
# to that reference on the card (text and images): today the chain verify. The rest are refused
# with FP8 K/V until they have.
def kv8_decode_ok(f: KvDecode) -> Bool:
  match f:
    case KdPlain{}:
      True{}
    case KdChain{}:
      True{}
    case KdTree{}:
      False{}
    case KdOther{}:
      False{}

# ============================================================================
# 6. Row order of the V projection
# ============================================================================

# The device row of channel c of a head's V projection. BF16 caches store a V row as the
# projection wrote it, in the lane order of the deinterleaving loads (`lane128` / `lane256`). The
# FP8 attention prep converts V through f32 and stores byte q of a row from channel
# kv8_vpos(q) of the row it read, so the FP8 order is kv8_vpos after the lane order: the PV
# GEMM then gives the attention output in the BF16 cache's channel order.
def kv_wv_row_if(+c: Nat, narrow: Bool, d: KvDtype) -> Nat:
  match narrow d:
    case True{} _:
      T.lane128(c)
    case False{} KvBf16{}:
      T.lane256(c)
    case False{} KvFp8{}:
      T.kv8_vpos(T.lane256(c))

def kv_wv_row(+c: Nat, +head_dim: Nat, d: KvDtype) -> Nat:
  kv_wv_row_if(c, Nat.is_eq(head_dim, 128n), d)

# ============================================================================
# 7. Laws
# ============================================================================

# promise: a cache element always takes at least one byte, so a cache row is never sized zero
# harm: the host-RAM tier would lay out rows of zero bytes and copy nothing it needs
law kv_elem_nonzero:
  for d: KvDtype
  {Nat.is_le(1n, kv_elem_bytes(d)) == True{} : Bool}

def kv_elem_nonzero(d):
  match d:
    case KvBf16{}: {==}
    case KvFp8{}: {==}

# promise: an FP8 cache position never takes more bytes than the same position in BF16
# harm: the host-RAM tier would hold FP8 entries larger than the BF16 ones they replace
law kv_fp8_row_not_larger:
  for +h: Nat
  for +w: Nat
  {Nat.is_le(kv_row_bytes(h, w, KvFp8{}), kv_row_bytes(h, w, KvBf16{})) == True{} : Bool}

def kv_fp8_row_not_larger(h, w):
  L.nat_mul_le_mono_l(Nat.mul(h, w), 1n, 2n, {==})

# promise: a cache whose graphs have no FP8 attention kernels and scale table is always BF16
# harm: a BF16 graph would read FP8 bytes as BF16 values
law kv_side_cache_bf16:
  for d: KvDtype
  {kv_layer_dtype(False{}, d) == KvBf16{} : KvDtype}

def kv_side_cache_bf16(d):
  {==}

# Not sealed: the reviewer's corrected form of an earlier law that pinned behaviour, waiting for a
# second review before it enters rusty-bends.lock.
# promise: a load configured for BF16 K/V never gets an FP8 cache
# harm: a cache would be FP8 without a scale table made for it
law kv_bf16_load_stays_bf16:
  for t: Bool
  {kv_layer_dtype(t, KvBf16{}) == KvBf16{} : KvDtype}

def kv_bf16_load_stays_bf16(t):
  match t:
    case True{}: {==}
    case False{}: {==}

# promise: a head shape the FP8 kernels accept never has more kv groups than the scale table has slots
# harm: a group past the table would read another kind's multiplier
law kv8_shape_within_slots:
  for +w: Nat
  for +n: Nat
  for +r: Nat
  for +s: Nat
  {Bool.and(kv8_shape_ok(w, n, r, s), Nat.is_lt(s, n)) == False{} : Bool}

def kv8_shape_within_slots(w, n, r, s):
  kc_and3_excl(Nat.is_eq(w, r), Nat.is_gt(n, 0n), Nat.is_le(n, s), Nat.is_lt(s, n), kc_le_lt_excl(n, s))

# promise: a model without kv heads is never accepted by the FP8 kernels
law kv8_shape_needs_groups:
  for +w: Nat
  for +r: Nat
  for +s: Nat
  {kv8_shape_ok(w, 0n, r, s) == False{} : Bool}

def kv8_shape_needs_groups(w, r, s):
  kc_and_false_r(Nat.is_eq(w, r))

# promise: whenever a multiplier can fit, the scaled maximum stays at or below top - margin, so with a margin of one or more nothing is clipped
# harm: the largest K / V values would saturate at the format's maximum
law kv8_fit_never_clips:
  for +a: Nat
  for +top: Nat
  for +margin: Nat
  for +cap: Nat
  {Bool.or(Nat.is_lt(Nat.sub(top, margin), a), Nat.is_le(Nat.add(a, kv8_fit(a, top, margin, cap)), Nat.sub(top, margin))) == True{} : Bool}

def kv8_fit_never_clips(a, top, margin, cap):
  kc_fit_le(a, Nat.sub(top, margin), cap)

# promise: a multiplier never exceeds its cap
# harm: a head measured as all zero would get a multiplier that blows any later value up to the format's maximum
law kv8_fit_capped:
  for +a: Nat
  for +top: Nat
  for +margin: Nat
  for +cap: Nat
  {Nat.is_le(kv8_fit(a, top, margin, cap), cap) == True{} : Bool}

def kv8_fit_capped(a, top, margin, cap):
  kc_min_le_l(cap, Nat.sub(Nat.sub(top, margin), a))

# promise: a layer's one V multiplier is never larger than any of its heads' own
# harm: the heads with the largest V values would be clipped
law kv8_min4_within_each:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for +d: Nat
  {Bool.and(Bool.and(Nat.is_le(kv8_min4(a, b, c, d), a), Nat.is_le(kv8_min4(a, b, c, d), b)), Bool.and(Nat.is_le(kv8_min4(a, b, c, d), c), Nat.is_le(kv8_min4(a, b, c, d), d))) == True{} : Bool}

def kv8_min4_within_each(a, b, c, d):
  kc_and4(
    Nat.is_le(kv8_min4(a, b, c, d), a),
    Nat.is_le(kv8_min4(a, b, c, d), b),
    Nat.is_le(kv8_min4(a, b, c, d), c),
    Nat.is_le(kv8_min4(a, b, c, d), d),
    L.nat_le_trans(kv8_min4(a, b, c, d), Nat.min(a, b), a, kc_min_le_l(Nat.min(a, b), Nat.min(c, d)), kc_min_le_l(a, b)),
    L.nat_le_trans(kv8_min4(a, b, c, d), Nat.min(a, b), b, kc_min_le_l(Nat.min(a, b), Nat.min(c, d)), kc_min_le_r(a, b)),
    L.nat_le_trans(kv8_min4(a, b, c, d), Nat.min(c, d), c, kc_min_le_r(Nat.min(a, b), Nat.min(c, d)), kc_min_le_l(c, d)),
    L.nat_le_trans(kv8_min4(a, b, c, d), Nat.min(c, d), d, kc_min_le_r(Nat.min(a, b), Nat.min(c, d)), kc_min_le_r(c, d))
  )

# promise: the non-speculative decode always runs on FP8 K/V, so every speculative form has its exactness reference
# harm: an FP8 K/V load could offer a drafter with nothing to check its tokens against
law kv8_plain_decode_ok:
  {kv8_decode_ok(KdPlain{}) == True{} : Bool}

def kv8_plain_decode_ok():
  {==}
