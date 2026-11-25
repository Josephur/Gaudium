# K/V cache element type (BF16 or FP8) and the FP8 scale rules of the trunk's attention.
#
# Formally specifies and proves:
#   - The bytes of one cache element and of one position of one layer's cache, per dtype
#   - Which caches take the configured dtype (those whose graphs have the FP8 attention kernels
#     and a scale table: the trunk's) and which stay BF16
#   - Which engines take an FP8 setting; which prefill cores read which caches, and the widest
#     prefill tile an FP8 cache takes (never a tile of zero rows)
#   - The block the V rows of a cache are permuted in, per dtype
#   - Which head shapes the FP8 attention kernels take (one FP8 vector per cache row, the scale
#     table's group slots)
#   - The power-of-two multiplier that maps a tensor into the FP8 range: its scaled maximum stays
#     below the format's top binade (nothing is clipped), and it never exceeds its cap
#   - The one V multiplier of a layer: never above what any of its heads allows
#   - Which decode forms may run on FP8 K/V (the non-speculative K=0 form always may)
#   - Which loads take FP8 K/V: never a calibration load, and only with a family, a head shape and
#     a decode form the FP8 attention takes; a scale table covers every kv head of the model
#   - The row order of the V projection per dtype (the FP8 V rows follow the softmax's P bytes):
#     every channel of a head lands on a row of that head, and no two on the same row
#   - The calibration's read-back of a cache: every range it copies lies inside the cache, each
#     copy fits the staging buffer and moves at least one byte while bytes are left, and the
#     batches of two different (chunk, kv head) pairs never overlap
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

# a == b and a < b never hold together
def kc_eq_lt_excl(+a: Nat, +b: Nat) -> {Bool.and(Nat.is_eq(a, b), Nat.is_lt(a, b)) == False{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      kc_eq_lt_excl(ap, bp)

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

# Whether an engine takes a load's K/V setting: BF16 always, FP8 only an engine whose graphs have
# the FP8 attention kernels and scale tables (`fp8_attention`: today the Qwen3.5 / 3.8 engine's).
# Any other engine refuses an FP8 setting instead of running BF16 caches under it.
def kv_setting_taken(fp8_attention: Bool, cfg: KvDtype) -> Bool:
  match fp8_attention:
    case True{}:
      True{}
    case False{}:
      Bool.not(kv_is_fp8(cfg))

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

# Whether a prefill tile whose form asks for the half-width core (`asked`: BF16 scores and the
# core's own V relayout of a BF16 cache) builds it on a cache of dtype `d`. The core reads BF16
# caches only, so a tile on an FP8 cache builds the full-width core instead.
def kv_half_core_taken(asked: Bool, d: KvDtype) -> Bool:
  match d:
    case KvBf16{}:
      asked
    case KvFp8{}:
      False{}

# The rows a prefill tile asked for with `rows` rows (an explicit width or a candidate) takes on a
# trunk cache of dtype `d`: a BF16 cache as asked, an FP8 cache at most `fp8_max` rows (the widest
# FP8 attention tile measured correct on the card, a parameter of the rule).
def kv_tile_rows(+rows: Nat, d: KvDtype, +fp8_max: Nat) -> Nat:
  match d:
    case KvBf16{}:
      rows
    case KvFp8{}:
      Nat.min(rows, fp8_max)

# ============================================================================
# 3. The FP8 attention kernels' shape rule
# ============================================================================

# One cache row of a head is one FP8 vector (`head_dim == row`, the kernels' vector width), and
# the scale table keeps `slots` entries per kind, one per kv group: 1 <= nkv <= slots.
def kv8_shape_ok(+head_dim: Nat, +nkv: Nat, +row: Nat, +slots: Nat) -> Bool:
  Bool.and(Nat.is_eq(head_dim, row), Bool.and(Nat.is_gt(nkv, 0n), Nat.is_le(nkv, slots)))

# A calibration lists the V maxima of `listed` kv heads for a layer whose model has `nkv`: the
# table is the model's when they are the same count.
def kv8_table_heads_ok(+listed: Nat, +nkv: Nat) -> Bool:
  Nat.is_eq(listed, nkv)

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
# 5b. Which loads take FP8 K/V
# ============================================================================

# Whether a model's family has run FP8 K/V on the card: the dense hybrids (DeltaNet layers
# between the full-attention ones, no routed experts). A policy, like `kv8_decode_ok`: a family
# joins once its FP8-K/V K=0 and its drafters have run on the card.
def kv8_family_ok(moe: Bool, +n_linear: Nat) -> Bool:
  Bool.and(Bool.not(moe), Nat.is_gt(n_linear, 0n))

# Why a load configured for FP8 K/V is refused, or `KrNone{}` when it takes it.
type Kv8Refusal is Data:
  KrNone{}
  KrCalibration{}
  KrFamily{}
  KrShape{}
  KrDecode{}

def kv8_refusal(calibrating: Bool, family: Bool, shape: Bool, decode: Bool) -> Kv8Refusal:
  match calibrating:
    case True{}:
      KrCalibration{}
    case False{}:
      match family:
        case False{}:
          KrFamily{}
        case True{}:
          match shape:
            case False{}:
              KrShape{}
            case True{}:
              match decode:
                case False{}:
                  KrDecode{}
                case True{}:
                  KrNone{}

# A load configured for FP8 K/V. A calibration load (`kv_cal`: the K/V maxima, `act_cal`: the
# activation scales) measures the BF16 graphs its table is defined on; otherwise the model's
# family, its heads (`kv8_shape_ok` with the kernels' `row` and the table's `slots`) and the
# decode form it serves (`kv8_decode_ok`) decide.
def kv8_load_check(kv_cal: Bool, act_cal: Bool, moe: Bool, +n_linear: Nat, +head_dim: Nat, +nkv: Nat, +row: Nat, +slots: Nat, f: KvDecode) -> Kv8Refusal:
  kv8_refusal(Bool.or(kv_cal, act_cal), kv8_family_ok(moe, n_linear), kv8_shape_ok(head_dim, nkv, row, slots), kv8_decode_ok(f))

# The load takes FP8 K/V.
def kv8_load_taken(r: Kv8Refusal) -> Bool:
  match r:
    case KrNone{}:
      True{}
    case KrCalibration{}:
      False{}
    case KrFamily{}:
      False{}
    case KrShape{}:
      False{}
    case KrDecode{}:
      False{}

# What a refusal rules out, one lemma per condition (erased before Rust codegen).
def kc_gate_calibrating(+c: Bool, +fa: Bool, +sh: Bool, +dc: Bool) -> {Bool.and(kv8_load_taken(kv8_refusal(c, fa, sh, dc)), c) == False{} : Bool}:
  match c:
    case True{}:
      {==}
    case False{}:
      kc_and_false_r(kv8_load_taken(kv8_refusal(False{}, fa, sh, dc)))

def kc_gate_family(+c: Bool, +fa: Bool, +sh: Bool, +dc: Bool) -> {Bool.and(kv8_load_taken(kv8_refusal(c, fa, sh, dc)), Bool.not(fa)) == False{} : Bool}:
  match c:
    case True{}:
      {==}
    case False{}:
      match fa:
        case False{}:
          {==}
        case True{}:
          kc_and_false_r(kv8_load_taken(kv8_refusal(False{}, True{}, sh, dc)))

def kc_gate_shape(+c: Bool, +fa: Bool, +sh: Bool, +dc: Bool) -> {Bool.and(kv8_load_taken(kv8_refusal(c, fa, sh, dc)), Bool.not(sh)) == False{} : Bool}:
  match c:
    case True{}:
      {==}
    case False{}:
      match fa:
        case False{}:
          {==}
        case True{}:
          match sh:
            case False{}:
              {==}
            case True{}:
              kc_and_false_r(kv8_load_taken(kv8_refusal(False{}, True{}, True{}, dc)))

def kc_gate_decode(+c: Bool, +fa: Bool, +sh: Bool, +dc: Bool) -> {Bool.and(kv8_load_taken(kv8_refusal(c, fa, sh, dc)), Bool.not(dc)) == False{} : Bool}:
  match c:
    case True{}:
      {==}
    case False{}:
      match fa:
        case False{}:
          {==}
        case True{}:
          match sh:
            case False{}:
              {==}
            case True{}:
              match dc:
                case False{}:
                  {==}
                case True{}:
                  {==}

# The refusal r as a type: `t` for KrNone{}, `f` for every refusal (erased before Rust codegen).
def kc_none_type(r: Kv8Refusal, t: Type, f: Type) -> Type:
  match r:
    case KrNone{}:
      t
    case KrCalibration{}:
      f
    case KrFamily{}:
      f
    case KrShape{}:
      f
    case KrDecode{}:
      f

# What an accepted result (`KrNone{}`) of `kv8_refusal` rules out, one lemma per condition: a
# refusal is never `KrNone{}` (erased before Rust codegen).
def kc_none_not_calibrating(+c: Bool, +fa: Bool, +sh: Bool, +dc: Bool, ok: {kv8_refusal(c, fa, sh, dc) == KrNone{} : Kv8Refusal}) -> {c == False{} : Bool}:
  match c:
    case True{}:
      %ok : kc_none_type(_, {True{} == False{} : Bool}, Unit)
      Unit{}
    case False{}:
      {==}

def kc_none_family(+c: Bool, +fa: Bool, +sh: Bool, +dc: Bool, ok: {kv8_refusal(c, fa, sh, dc) == KrNone{} : Kv8Refusal}) -> {fa == True{} : Bool}:
  match c:
    case True{}:
      %ok : kc_none_type(_, {fa == True{} : Bool}, Unit)
      Unit{}
    case False{}:
      match fa:
        case False{}:
          %ok : kc_none_type(_, {False{} == True{} : Bool}, Unit)
          Unit{}
        case True{}:
          {==}

def kc_none_shape(+c: Bool, +fa: Bool, +sh: Bool, +dc: Bool, ok: {kv8_refusal(c, fa, sh, dc) == KrNone{} : Kv8Refusal}) -> {sh == True{} : Bool}:
  match c:
    case True{}:
      %ok : kc_none_type(_, {sh == True{} : Bool}, Unit)
      Unit{}
    case False{}:
      match fa:
        case False{}:
          %ok : kc_none_type(_, {sh == True{} : Bool}, Unit)
          Unit{}
        case True{}:
          match sh:
            case False{}:
              %ok : kc_none_type(_, {False{} == True{} : Bool}, Unit)
              Unit{}
            case True{}:
              {==}

def kc_none_decode(+c: Bool, +fa: Bool, +sh: Bool, +dc: Bool, ok: {kv8_refusal(c, fa, sh, dc) == KrNone{} : Kv8Refusal}) -> {dc == True{} : Bool}:
  match c:
    case True{}:
      %ok : kc_none_type(_, {dc == True{} : Bool}, Unit)
      Unit{}
    case False{}:
      match fa:
        case False{}:
          %ok : kc_none_type(_, {dc == True{} : Bool}, Unit)
          Unit{}
        case True{}:
          match sh:
            case False{}:
              %ok : kc_none_type(_, {dc == True{} : Bool}, Unit)
              Unit{}
            case True{}:
              match dc:
                case False{}:
                  %ok : kc_none_type(_, {False{} == True{} : Bool}, Unit)
                  Unit{}
                case True{}:
                  {==}

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

# promise: an engine without the FP8 attention kernels never takes an FP8 K/V setting
# harm: a load asked for FP8 K/V would run, and report, BF16 caches without saying so
law kv_fp8_refused_without_fp8_attention:
  {kv_setting_taken(False{}, KvFp8{}) == False{} : Bool}

def kv_fp8_refused_without_fp8_attention():
  {==}

# promise: the half-width prefill core, which reads BF16 caches only, is never built on an FP8 cache
# harm: its BF16 scores and V relayout would read FP8 bytes as BF16 values, and the prefill would write wrong K/V and hidden states
law kv_half_core_never_fp8:
  for a: Bool
  {kv_half_core_taken(a, KvFp8{}) == False{} : Bool}

def kv_half_core_never_fp8(a):
  {==}

# promise: a prefill tile builds the half-width core only when its form asks for it
# harm: a load whose form turned the core off would still build it
law kv_half_core_only_asked:
  for d: KvDtype
  {kv_half_core_taken(False{}, d) == False{} : Bool}

def kv_half_core_only_asked(d):
  match d:
    case KvBf16{}: {==}
    case KvFp8{}: {==}

# promise: a prefill tile on an FP8 cache never has more rows than the widest FP8 tile measured correct
# harm: a wider FP8 attention tile, measured to give NaN rows past one cache chunk, would prefill wrong K/V and hidden states
law kv_tile_rows_fp8_within:
  for +t: Nat
  for +m: Nat
  {Nat.is_le(kv_tile_rows(t, KvFp8{}, m), m) == True{} : Bool}

def kv_tile_rows_fp8_within(t, m):
  kc_min_le_r(t, m)

# promise: the cache dtype only ever narrows a prefill tile, never makes it wider than the tile asked for
# harm: a tile wider than every width asked for (an explicit width or a candidate, at most the engine's widest) would overrun buffers sized for those widths
law kv_tile_rows_not_wider:
  for +t: Nat
  for d: KvDtype
  for +m: Nat
  {Nat.is_le(kv_tile_rows(t, d, m), t) == True{} : Bool}

def kv_tile_rows_not_wider(t, d, m):
  match d:
    case KvBf16{}:
      L.nat_le_refl(t)
    case KvFp8{}:
      kc_min_le_l(t, m)

# promise: a prefill tile asked for with at least one row keeps at least one row on either cache dtype, whenever the FP8 bound allows one
# harm: a tile of zero rows would never advance through the prompt, and its prefill would hang or compile an empty graph
law kv_tile_rows_nonzero:
  for +t: Nat
  for d: KvDtype
  for +m: Nat
  {Nat.is_lt(0n, kv_tile_rows(1n+t, d, 1n+m)) == True{} : Bool}

def kv_tile_rows_nonzero(t, d, m):
  match d:
    case KvBf16{}: {==}
    case KvFp8{}: {==}

# promise: a calibration load, of the K/V maxima or of the activation scales, never takes FP8 K/V
# harm: a calibration would measure through the FP8 attention, the K/V maxima already clipped by the multipliers they are meant to choose
law kv8_load_never_calibrating:
  for +kc: Bool
  for +ac: Bool
  for +m: Bool
  for +n: Nat
  for +w: Nat
  for +h: Nat
  for +r: Nat
  for +s: Nat
  for +f: KvDecode
  {Bool.and(kv8_load_taken(kv8_load_check(kc, ac, m, n, w, h, r, s, f)), Bool.or(kc, ac)) == False{} : Bool}

def kv8_load_never_calibrating(kc, ac, m, n, w, h, r, s, f):
  kc_gate_calibrating(Bool.or(kc, ac), kv8_family_ok(m, n), kv8_shape_ok(w, h, r, s), kv8_decode_ok(f))

# promise: a model whose family has not run FP8 K/V on the card never takes it
# harm: graphs nothing has validated on FP8 K/V would serve tokens without a checked reference
law kv8_load_needs_family:
  for +kc: Bool
  for +ac: Bool
  for +m: Bool
  for +n: Nat
  for +w: Nat
  for +h: Nat
  for +r: Nat
  for +s: Nat
  for +f: KvDecode
  {Bool.and(kv8_load_taken(kv8_load_check(kc, ac, m, n, w, h, r, s, f)), Bool.not(kv8_family_ok(m, n))) == False{} : Bool}

def kv8_load_needs_family(kc, ac, m, n, w, h, r, s, f):
  kc_gate_family(Bool.or(kc, ac), kv8_family_ok(m, n), kv8_shape_ok(w, h, r, s), kv8_decode_ok(f))

# promise: a load that takes FP8 K/V always has a head shape the FP8 attention kernels accept
# harm: the kernels would read cache rows or scale-table slots its heads do not have
law kv8_load_needs_shape:
  for +kc: Bool
  for +ac: Bool
  for +m: Bool
  for +n: Nat
  for +w: Nat
  for +h: Nat
  for +r: Nat
  for +s: Nat
  for +f: KvDecode
  {Bool.and(kv8_load_taken(kv8_load_check(kc, ac, m, n, w, h, r, s, f)), Bool.not(kv8_shape_ok(w, h, r, s))) == False{} : Bool}

def kv8_load_needs_shape(kc, ac, m, n, w, h, r, s, f):
  kc_gate_shape(Bool.or(kc, ac), kv8_family_ok(m, n), kv8_shape_ok(w, h, r, s), kv8_decode_ok(f))

# promise: a load that takes FP8 K/V always serves a decode form that may run on FP8 K/V
# harm: a drafter never checked against FP8-K/V K=0 would serve its tokens as exact
law kv8_load_needs_decode:
  for +kc: Bool
  for +ac: Bool
  for +m: Bool
  for +n: Nat
  for +w: Nat
  for +h: Nat
  for +r: Nat
  for +s: Nat
  for +f: KvDecode
  {Bool.and(kv8_load_taken(kv8_load_check(kc, ac, m, n, w, h, r, s, f)), Bool.not(kv8_decode_ok(f))) == False{} : Bool}

def kv8_load_needs_decode(kc, ac, m, n, w, h, r, s, f):
  kc_gate_decode(Bool.or(kc, ac), kv8_family_ok(m, n), kv8_shape_ok(w, h, r, s), kv8_decode_ok(f))

# promise: a scale table that lists fewer kv heads for a layer than the model has is refused
# harm: a head without its own measured maximum would take another head's V multiplier, and its largest values could clip
law kv8_table_covers_heads:
  for +a: Nat
  for +b: Nat
  {Bool.and(kv8_table_heads_ok(a, b), Nat.is_lt(a, b)) == False{} : Bool}

def kv8_table_covers_heads(a, b):
  kc_eq_lt_excl(a, b)

# The four laws below state the load gate's guarantees on the result Rust matches on: a load is
# accepted exactly when `kv8_load_check` returns `KrNone{}` (`core/kv_cache.rs` `check_fp8_load`).

# promise: a load the FP8 K/V check accepts is never a calibration load, of the K/V maxima or of the activation scales
# harm: a calibration would measure through the FP8 attention, the K/V maxima already clipped by the multipliers they are meant to choose
law kv8_accepted_not_calibrating:
  for +kc: Bool
  for +ac: Bool
  for +m: Bool
  for +n: Nat
  for +w: Nat
  for +k: Nat
  for +r: Nat
  for +s: Nat
  for +f: KvDecode
  for ok: {kv8_load_check(kc, ac, m, n, w, k, r, s, f) == KrNone{} : Kv8Refusal}
  {Bool.or(kc, ac) == False{} : Bool}

def kv8_accepted_not_calibrating(kc, ac, m, n, w, k, r, s, f, ok):
  kc_none_not_calibrating(Bool.or(kc, ac), kv8_family_ok(m, n), kv8_shape_ok(w, k, r, s), kv8_decode_ok(f), ok)

# promise: a load the FP8 K/V check accepts always has a family that has run FP8 K/V on the card
# harm: graphs nothing has validated on FP8 K/V would serve tokens without a checked reference
law kv8_accepted_family:
  for +kc: Bool
  for +ac: Bool
  for +m: Bool
  for +n: Nat
  for +w: Nat
  for +k: Nat
  for +r: Nat
  for +s: Nat
  for +f: KvDecode
  for ok: {kv8_load_check(kc, ac, m, n, w, k, r, s, f) == KrNone{} : Kv8Refusal}
  {kv8_family_ok(m, n) == True{} : Bool}

def kv8_accepted_family(kc, ac, m, n, w, k, r, s, f, ok):
  kc_none_family(Bool.or(kc, ac), kv8_family_ok(m, n), kv8_shape_ok(w, k, r, s), kv8_decode_ok(f), ok)

# promise: a load the FP8 K/V check accepts always has a head shape the FP8 attention kernels accept
# harm: the kernels would read cache rows or scale-table slots its heads do not have
law kv8_accepted_shape:
  for +kc: Bool
  for +ac: Bool
  for +m: Bool
  for +n: Nat
  for +w: Nat
  for +k: Nat
  for +r: Nat
  for +s: Nat
  for +f: KvDecode
  for ok: {kv8_load_check(kc, ac, m, n, w, k, r, s, f) == KrNone{} : Kv8Refusal}
  {kv8_shape_ok(w, k, r, s) == True{} : Bool}

def kv8_accepted_shape(kc, ac, m, n, w, k, r, s, f, ok):
  kc_none_shape(Bool.or(kc, ac), kv8_family_ok(m, n), kv8_shape_ok(w, k, r, s), kv8_decode_ok(f), ok)

# promise: a load the FP8 K/V check accepts always serves a decode form that may run on FP8 K/V
# harm: a drafter never checked against FP8-K/V K=0 would serve its tokens as exact
law kv8_accepted_decode:
  for +kc: Bool
  for +ac: Bool
  for +m: Bool
  for +n: Nat
  for +w: Nat
  for +k: Nat
  for +r: Nat
  for +s: Nat
  for +f: KvDecode
  for ok: {kv8_load_check(kc, ac, m, n, w, k, r, s, f) == KrNone{} : Kv8Refusal}
  {kv8_decode_ok(f) == True{} : Bool}

def kv8_accepted_decode(kc, ac, m, n, w, k, r, s, f, ok):
  kc_none_decode(Bool.or(kc, ac), kv8_family_ok(m, n), kv8_shape_ok(w, k, r, s), kv8_decode_ok(f), ok)

# ============================================================================
# 8. The V row order is a permutation of each head's rows
# ============================================================================

# The proofs below derive the bounds and the injectivity of the V row order from Euclidean
# division (the split lemmas of `nat_lemmas.bend`): lane128 and kv8_vpos128 place a residue
# mod 128 one-to-one inside a 128-row block, and lane256 / kv8_vpos keep the block of a row.

# A residue's half and quarter: (c % 128) / 2 < 64, (c % 128) / 4 < 32.
def kc_half_lt(+c: Nat) -> {Nat.is_lt(Nat.div(Nat.mod(c, 128n), 2n), 64n) == True{} : Bool}:
  L.nat_div_lt_of_lt_mul(Nat.mod(c, 128n), 64n, 1n, L.nat_mod_lt(c, 127n))

def kc_quarter_lt(+c: Nat) -> {Nat.is_lt(Nat.div(Nat.mod(c, 128n), 4n), 32n) == True{} : Bool}:
  L.nat_div_lt_of_lt_mul(Nat.mod(c, 128n), 32n, 3n, L.nat_mod_lt(c, 127n))

def kc_eq_of_is_eq(+a: Nat, +b: Nat, h: {Nat.is_eq(a, b) == True{} : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      L.bool_absurd({0n == 1n+bp : Nat}, h)
    case 1n+ap 0n:
      L.bool_absurd({1n+ap == 0n : Nat}, h)
    case 1n+ap 1n+bp:
      %kc_eq_of_is_eq(ap, bp, h) : {1n+ap == 1n+_ : Nat}
      {==}

# A remainder mod 2 that is not zero is one.
def kc_odd_is_one(+p: Nat, ne: {False{} == Nat.is_eq(p, 0n) : Bool}, lt: {Nat.is_lt(p, 2n) == True{} : Bool}) -> {p == 1n : Nat}:
  match p:
    case 0n:
      L.bool_absurd({0n == 1n : Nat}, ne)
    case 1n+pp:
      match pp:
        case 0n:
          {==}
        case 1n+ppp:
          match ppp:
            case 0n:
              L.bool_absurd({2n == 1n : Nat}, lt)
            case 1n+q:
              L.bool_absurd({1n+1n+1n+q == 1n : Nat}, lt)

# 2 * i written as i * 2, in a bound and in an equation.
def kc_two_lt(+i: Nat, +r: Nat, +n: Nat, h: {Nat.is_lt(Nat.add(Nat.mul(i, 2n), r), n) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(2n, i), r), n) == True{} : Bool}:
  %L.nat_mul_comm(i, 2n) : {Nat.is_lt(Nat.add(_, r), n) == True{} : Bool}
  h

def kc_two_swap(+i: Nat, +k: Nat, +r: Nat, +s: Nat, e: {Nat.add(Nat.mul(2n, i), r) == Nat.add(Nat.mul(2n, k), s) : Nat}) -> {Nat.add(Nat.mul(i, 2n), r) == Nat.add(Nat.mul(k, 2n), s) : Nat}:
  %L.nat_mul_comm(2n, i) : {Nat.add(_, r) == Nat.add(Nat.mul(k, 2n), s) : Nat}
  %L.nat_mul_comm(2n, k) : {Nat.add(Nat.mul(2n, i), r) == Nat.add(_, s) : Nat}
  e

# --- lane128: even residues to the low half of the block, odd ones to the high half ---

def kc_lane128_if_lt(+h: Nat, +even: Bool, lt: {Nat.is_lt(h, 64n) == True{} : Bool}) -> {Nat.is_lt(T.lane128_if(h, even), 128n) == True{} : Bool}:
  match even:
    case True{}:
      L.nat_lt_trans(h, 64n, 128n, lt, {==})
    case False{}:
      L.nat_add_lt_l(64n, h, 64n, lt)

def kc_lane128_lt(+c: Nat) -> {Nat.is_lt(T.lane128(c), 128n) == True{} : Bool}:
  kc_lane128_if_lt(Nat.div(Nat.mod(c, 128n), 2n), Nat.is_eq(Nat.mod(Nat.mod(c, 128n), 2n), 0n), kc_half_lt(c))

# Two residues m = 2 * h + p placed on the same lane have the same half and parity.
def kc_lane128_if_inj(
  +hx: Nat,
  +px: Nat,
  +hy: Nat,
  +py: Nat,
  +ex: Bool,
  +ey: Bool,
  eqx: {ex == Nat.is_eq(px, 0n) : Bool},
  eqy: {ey == Nat.is_eq(py, 0n) : Bool},
  ltx: {Nat.is_lt(hx, 64n) == True{} : Bool},
  lty: {Nat.is_lt(hy, 64n) == True{} : Bool},
  px2: {Nat.is_lt(px, 2n) == True{} : Bool},
  py2: {Nat.is_lt(py, 2n) == True{} : Bool},
  e: {T.lane128_if(hx, ex) == T.lane128_if(hy, ey) : Nat}
) -> {Nat.add(Nat.mul(hx, 2n), px) == Nat.add(Nat.mul(hy, 2n), py) : Nat}:
  match ex ey:
    case True{} True{}:
      %Equal.sym(Nat, px, 0n, kc_eq_of_is_eq(px, 0n, Equal.sym(Bool, True{}, Nat.is_eq(px, 0n), eqx))) : {Nat.add(Nat.mul(hx, 2n), _) == Nat.add(Nat.mul(hy, 2n), py) : Nat}
      %Equal.sym(Nat, py, 0n, kc_eq_of_is_eq(py, 0n, Equal.sym(Bool, True{}, Nat.is_eq(py, 0n), eqy))) : {Nat.add(Nat.mul(hx, 2n), 0n) == Nat.add(Nat.mul(hy, 2n), _) : Nat}
      %e : {Nat.add(Nat.mul(hx, 2n), 0n) == Nat.add(Nat.mul(_, 2n), 0n) : Nat}
      {==}
    case False{} False{}:
      %Equal.sym(Nat, px, 1n, kc_odd_is_one(px, eqx, px2)) : {Nat.add(Nat.mul(hx, 2n), _) == Nat.add(Nat.mul(hy, 2n), py) : Nat}
      %Equal.sym(Nat, py, 1n, kc_odd_is_one(py, eqy, py2)) : {Nat.add(Nat.mul(hx, 2n), 1n) == Nat.add(Nat.mul(hy, 2n), _) : Nat}
      %L.nat_add_cancel_l(64n, hx, hy, e) : {Nat.add(Nat.mul(hx, 2n), 1n) == Nat.add(Nat.mul(_, 2n), 1n) : Nat}
      {==}
    case True{} False{}:
      Empty.absurd({Nat.add(Nat.mul(hx, 2n), px) == Nat.add(Nat.mul(hy, 2n), py) : Nat}, L.nat_add_not_lt_l(64n, hy, L.nat_lt_subst_l(hx, Nat.add(64n, hy), 64n, e, ltx)))
    case False{} True{}:
      Empty.absurd({Nat.add(Nat.mul(hx, 2n), px) == Nat.add(Nat.mul(hy, 2n), py) : Nat}, L.nat_add_not_lt_l(64n, hx, L.nat_lt_subst_l(hy, Nat.add(64n, hx), 64n, Equal.sym(Nat, Nat.add(64n, hx), hy, e), lty)))

# Two channels on the same lane have the same residue mod 128.
def kc_lane128_inj(+x: Nat, +y: Nat, e: {T.lane128(x) == T.lane128(y) : Nat}) -> {Nat.mod(x, 128n) == Nat.mod(y, 128n) : Nat}:
  %L.nat_div_mod_r(Nat.mod(x, 128n), 1n) : {_ == Nat.mod(y, 128n) : Nat}
  %L.nat_div_mod_r(Nat.mod(y, 128n), 1n) : {Nat.add(Nat.mul(Nat.div(Nat.mod(x, 128n), 2n), 2n), Nat.mod(Nat.mod(x, 128n), 2n)) == _ : Nat}
  kc_lane128_if_inj(
    Nat.div(Nat.mod(x, 128n), 2n),
    Nat.mod(Nat.mod(x, 128n), 2n),
    Nat.div(Nat.mod(y, 128n), 2n),
    Nat.mod(Nat.mod(y, 128n), 2n),
    Nat.is_eq(Nat.mod(Nat.mod(x, 128n), 2n), 0n),
    Nat.is_eq(Nat.mod(Nat.mod(y, 128n), 2n), 0n),
    {==},
    {==},
    kc_half_lt(x),
    kc_half_lt(y),
    L.nat_mod_lt(Nat.mod(x, 128n), 1n),
    L.nat_mod_lt(Nat.mod(y, 128n), 1n),
    e)

# Two channels of a 128-channel head on the same lane are the same channel.
def kc_lane128_distinct(+a: Nat, +b: Nat, ha: {Nat.is_lt(a, 128n) == True{} : Bool}, hb: {Nat.is_lt(b, 128n) == True{} : Bool}, e: {T.lane128(a) == T.lane128(b) : Nat}) -> {a == b : Nat}:
  %L.nat_mod_small(a, 127n, ha) : {_ == b : Nat}
  %L.nat_mod_small(b, 127n, hb) : {Nat.mod(a, 128n) == _ : Nat}
  kc_lane128_inj(a, b, e)

# --- lane256: the block of c, then lane128 inside it ---

def kc_lane256_lt(+c: Nat, ok: {Nat.is_lt(c, 256n) == True{} : Bool}) -> {Nat.is_lt(T.lane256(c), 256n) == True{} : Bool}:
  L.nat_split_lt(Nat.div(c, 128n), T.lane128(c), 2n, 127n, L.nat_div_lt_of_lt_mul(c, 2n, 127n, ok), kc_lane128_lt(c))

def kc_lane256_inj(+a: Nat, +b: Nat, +e: {T.lane256(a) == T.lane256(b) : Nat}) -> {a == b : Nat}:
  L.nat_eq_of_div_mod(a, b, 127n,
    L.nat_split_unique_q(Nat.div(a, 128n), T.lane128(a), Nat.div(b, 128n), T.lane128(b), 127n, kc_lane128_lt(a), kc_lane128_lt(b), e),
    kc_lane128_inj(a, b, L.nat_split_unique_r(Nat.div(a, 128n), T.lane128(a), Nat.div(b, 128n), T.lane128(b), 127n, kc_lane128_lt(a), kc_lane128_lt(b), e)))

# --- kv8_vpos128: residue m = 4 * i + j; j < 2 to the low half, the others to the high half ---

def kc_vpos128_if_lt(
  +i: Nat,
  +j: Nat,
  +low: Bool,
  eql: {low == Nat.is_lt(j, 2n) : Bool},
  li: {Nat.is_lt(i, 32n) == True{} : Bool},
  lj: {Nat.is_lt(j, 4n) == True{} : Bool}
) -> {Nat.is_lt(T.kv8_vpos128_if(i, j, low), 128n) == True{} : Bool}:
  match low:
    case True{}:
      L.nat_lt_trans(Nat.add(Nat.mul(2n, i), j), 64n, 128n, kc_two_lt(i, j, 64n, L.nat_split_lt(i, j, 32n, 1n, li, Equal.sym(Bool, True{}, Nat.is_lt(j, 2n), eql))), {==})
    case False{}:
      L.nat_add_lt_l(64n, Nat.add(Nat.mul(2n, i), Nat.sub(j, 2n)), 64n, kc_two_lt(i, Nat.sub(j, 2n), 64n, L.nat_split_lt(i, Nat.sub(j, 2n), 32n, 1n, li, L.nat_sub_lt(j, 2n, 2n, {==}, lj))))

def kc_vpos128_lt(+q: Nat) -> {Nat.is_lt(T.kv8_vpos128(q), 128n) == True{} : Bool}:
  kc_vpos128_if_lt(Nat.div(Nat.mod(q, 128n), 4n), Nat.mod(Nat.mod(q, 128n), 4n), Nat.is_lt(Nat.mod(Nat.mod(q, 128n), 4n), 2n), {==}, kc_quarter_lt(q), L.nat_mod_lt(Nat.mod(q, 128n), 3n))

# j >= 2 is recovered from j - 2.
def kc_sub2_eq(+jx: Nat, +jy: Nat, gx: {Nat.is_le(2n, jx) == True{} : Bool}, gy: {Nat.is_le(2n, jy) == True{} : Bool}, e: {Nat.sub(jx, 2n) == Nat.sub(jy, 2n) : Nat}) -> {jx == jy : Nat}:
  %L.nat_sub_add_cancel(jx, 2n, gx) : {_ == jy : Nat}
  %L.nat_sub_add_cancel(jy, 2n, gy) : {Nat.add(Nat.sub(jx, 2n), 2n) == _ : Nat}
  %e : {Nat.add(Nat.sub(jx, 2n), 2n) == Nat.add(_, 2n) : Nat}
  {==}

# Two residues 4 * i + j placed on the same row of a block are the same residue.
def kc_vpos128_if_inj(
  +ix: Nat,
  +jx: Nat,
  +iy: Nat,
  +jy: Nat,
  +lx: Bool,
  +ly: Bool,
  eqx: {lx == Nat.is_lt(jx, 2n) : Bool},
  eqy: {ly == Nat.is_lt(jy, 2n) : Bool},
  lix: {Nat.is_lt(ix, 32n) == True{} : Bool},
  liy: {Nat.is_lt(iy, 32n) == True{} : Bool},
  ljx: {Nat.is_lt(jx, 4n) == True{} : Bool},
  ljy: {Nat.is_lt(jy, 4n) == True{} : Bool},
  e: {T.kv8_vpos128_if(ix, jx, lx) == T.kv8_vpos128_if(iy, jy, ly) : Nat}
) -> {Nat.add(Nat.mul(ix, 4n), jx) == Nat.add(Nat.mul(iy, 4n), jy) : Nat}:
  match lx ly:
    case True{} True{}:
      +hx = Equal.sym(Bool, True{}, Nat.is_lt(jx, 2n), eqx)
      +hy = Equal.sym(Bool, True{}, Nat.is_lt(jy, 2n), eqy)
      +e2 = kc_two_swap(ix, iy, jx, jy, e)
      %L.nat_split_unique_q(ix, jx, iy, jy, 1n, hx, hy, e2) : {Nat.add(Nat.mul(ix, 4n), jx) == Nat.add(Nat.mul(_, 4n), jy) : Nat}
      %L.nat_split_unique_r(ix, jx, iy, jy, 1n, hx, hy, e2) : {Nat.add(Nat.mul(ix, 4n), jx) == Nat.add(Nat.mul(ix, 4n), _) : Nat}
      {==}
    case False{} False{}:
      +hx = L.nat_sub_lt(jx, 2n, 2n, {==}, ljx)
      +hy = L.nat_sub_lt(jy, 2n, 2n, {==}, ljy)
      +e2 = kc_two_swap(ix, iy, Nat.sub(jx, 2n), Nat.sub(jy, 2n), L.nat_add_cancel_l(64n, Nat.add(Nat.mul(2n, ix), Nat.sub(jx, 2n)), Nat.add(Nat.mul(2n, iy), Nat.sub(jy, 2n)), e))
      %L.nat_split_unique_q(ix, Nat.sub(jx, 2n), iy, Nat.sub(jy, 2n), 1n, hx, hy, e2) : {Nat.add(Nat.mul(ix, 4n), jx) == Nat.add(Nat.mul(_, 4n), jy) : Nat}
      %kc_sub2_eq(jx, jy, L.nat_not_lt_imp_le(jx, 2n, Equal.sym(Bool, False{}, Nat.is_lt(jx, 2n), eqx)), L.nat_not_lt_imp_le(jy, 2n, Equal.sym(Bool, False{}, Nat.is_lt(jy, 2n), eqy)), L.nat_split_unique_r(ix, Nat.sub(jx, 2n), iy, Nat.sub(jy, 2n), 1n, hx, hy, e2)) : {Nat.add(Nat.mul(ix, 4n), jx) == Nat.add(Nat.mul(ix, 4n), _) : Nat}
      {==}
    case True{} False{}:
      Empty.absurd({Nat.add(Nat.mul(ix, 4n), jx) == Nat.add(Nat.mul(iy, 4n), jy) : Nat}, L.nat_add_not_lt_l(64n, Nat.add(Nat.mul(2n, iy), Nat.sub(jy, 2n)), L.nat_lt_subst_l(Nat.add(Nat.mul(2n, ix), jx), Nat.add(64n, Nat.add(Nat.mul(2n, iy), Nat.sub(jy, 2n))), 64n, e, kc_two_lt(ix, jx, 64n, L.nat_split_lt(ix, jx, 32n, 1n, lix, Equal.sym(Bool, True{}, Nat.is_lt(jx, 2n), eqx))))))
    case False{} True{}:
      Empty.absurd({Nat.add(Nat.mul(ix, 4n), jx) == Nat.add(Nat.mul(iy, 4n), jy) : Nat}, L.nat_add_not_lt_l(64n, Nat.add(Nat.mul(2n, ix), Nat.sub(jx, 2n)), L.nat_lt_subst_l(Nat.add(Nat.mul(2n, iy), jy), Nat.add(64n, Nat.add(Nat.mul(2n, ix), Nat.sub(jx, 2n))), 64n, Equal.sym(Nat, Nat.add(64n, Nat.add(Nat.mul(2n, ix), Nat.sub(jx, 2n))), Nat.add(Nat.mul(2n, iy), jy), e), kc_two_lt(iy, jy, 64n, L.nat_split_lt(iy, jy, 32n, 1n, liy, Equal.sym(Bool, True{}, Nat.is_lt(jy, 2n), eqy))))))

def kc_vpos128_inj(+x: Nat, +y: Nat, e: {T.kv8_vpos128(x) == T.kv8_vpos128(y) : Nat}) -> {Nat.mod(x, 128n) == Nat.mod(y, 128n) : Nat}:
  %L.nat_div_mod_r(Nat.mod(x, 128n), 3n) : {_ == Nat.mod(y, 128n) : Nat}
  %L.nat_div_mod_r(Nat.mod(y, 128n), 3n) : {Nat.add(Nat.mul(Nat.div(Nat.mod(x, 128n), 4n), 4n), Nat.mod(Nat.mod(x, 128n), 4n)) == _ : Nat}
  kc_vpos128_if_inj(
    Nat.div(Nat.mod(x, 128n), 4n),
    Nat.mod(Nat.mod(x, 128n), 4n),
    Nat.div(Nat.mod(y, 128n), 4n),
    Nat.mod(Nat.mod(y, 128n), 4n),
    Nat.is_lt(Nat.mod(Nat.mod(x, 128n), 4n), 2n),
    Nat.is_lt(Nat.mod(Nat.mod(y, 128n), 4n), 2n),
    {==},
    {==},
    kc_quarter_lt(x),
    kc_quarter_lt(y),
    L.nat_mod_lt(Nat.mod(x, 128n), 3n),
    L.nat_mod_lt(Nat.mod(y, 128n), 3n),
    e)

# --- kv8_vpos: the block of q, then kv8_vpos128 inside it ---

def kc_vpos_lt(+q: Nat, ok: {Nat.is_lt(q, 256n) == True{} : Bool}) -> {Nat.is_lt(T.kv8_vpos(q), 256n) == True{} : Bool}:
  L.nat_split_lt(Nat.div(q, 128n), T.kv8_vpos128(q), 2n, 127n, L.nat_div_lt_of_lt_mul(q, 2n, 127n, ok), kc_vpos128_lt(q))

def kc_vpos_inj(+a: Nat, +b: Nat, +e: {T.kv8_vpos(a) == T.kv8_vpos(b) : Nat}) -> {a == b : Nat}:
  L.nat_eq_of_div_mod(a, b, 127n,
    L.nat_split_unique_q(Nat.div(a, 128n), T.kv8_vpos128(a), Nat.div(b, 128n), T.kv8_vpos128(b), 127n, kc_vpos128_lt(a), kc_vpos128_lt(b), e),
    kc_vpos128_inj(a, b, L.nat_split_unique_r(Nat.div(a, 128n), T.kv8_vpos128(a), Nat.div(b, 128n), T.kv8_vpos128(b), 127n, kc_vpos128_lt(a), kc_vpos128_lt(b), e)))

# promise: every channel of a 256-channel head lands on a row of that head in its V projection
# harm: a V channel would be packed outside its head, over another head's rows or past the weight
law kv_wv_row_in_head_256:
  for +c: Nat
  for +d: KvDtype
  for ok: {Nat.is_lt(c, 256n) == True{} : Bool}
  {Nat.is_lt(kv_wv_row(c, 256n, d), 256n) == True{} : Bool}

def kv_wv_row_in_head_256(c, d, ok):
  match d:
    case KvBf16{}:
      kc_lane256_lt(c, ok)
    case KvFp8{}:
      kc_vpos_lt(T.lane256(c), kc_lane256_lt(c, ok))

# promise: every channel of a 128-channel head lands on a row of that head in its V projection
# harm: a V channel would be packed outside its head, over another head's rows or past the weight
law kv_wv_row_in_head_128:
  for +c: Nat
  for +d: KvDtype
  for ok: {Nat.is_lt(c, 128n) == True{} : Bool}
  {Nat.is_lt(kv_wv_row(c, 128n, d), 128n) == True{} : Bool}

def kv_wv_row_in_head_128(c, d, ok):
  match d:
    case KvBf16{}:
      kc_lane128_lt(c)
    case KvFp8{}:
      kc_lane128_lt(c)

# promise: two channels of a 256-channel head never land on the same row of its V projection
# harm: one V channel would overwrite another in the packed weight, and the attention output would lose it
law kv_wv_rows_distinct_256:
  for +a: Nat
  for +b: Nat
  for +d: KvDtype
  for ha: {Nat.is_lt(a, 256n) == True{} : Bool}
  for hb: {Nat.is_lt(b, 256n) == True{} : Bool}
  for e: {kv_wv_row(a, 256n, d) == kv_wv_row(b, 256n, d) : Nat}
  {a == b : Nat}

def kv_wv_rows_distinct_256(a, b, d, ha, hb, e):
  match d:
    case KvBf16{}:
      kc_lane256_inj(a, b, e)
    case KvFp8{}:
      kc_lane256_inj(a, b, kc_vpos_inj(T.lane256(a), T.lane256(b), e))

# promise: two channels of a 128-channel head never land on the same row of its V projection
# harm: one V channel would overwrite another in the packed weight, and the attention output would lose it
law kv_wv_rows_distinct_128:
  for +a: Nat
  for +b: Nat
  for +d: KvDtype
  for ha: {Nat.is_lt(a, 128n) == True{} : Bool}
  for hb: {Nat.is_lt(b, 128n) == True{} : Bool}
  for e: {kv_wv_row(a, 128n, d) == kv_wv_row(b, 128n, d) : Nat}
  {a == b : Nat}

def kv_wv_rows_distinct_128(a, b, d, ha, hb, e):
  match d:
    case KvBf16{}:
      kc_lane128_distinct(a, b, ha, hb, e)
    case KvFp8{}:
      kc_lane128_distinct(a, b, ha, hb, e)

# ============================================================================
# 9. The calibration's read-back of a cache
# ============================================================================

# Byte offset of kv head g's batch in chunk ch of a cache laid out chunk by chunk, `nkv` batches
# of `head` bytes per chunk.
def kv_batch_off(+ch: Nat, +g: Nat, +nkv: Nat, +head: Nat) -> Nat:
  Nat.mul(Nat.add(Nat.mul(ch, nkv), g), head)

# A read of `len` bytes at offset `off` lies inside an allocation of `bytes` bytes (written so
# that it cannot overflow).
def kv_range_inside(+off: Nat, +len: Nat, +bytes: Nat) -> Bool:
  Bool.and(Nat.is_le(off, bytes), Nat.is_le(len, Nat.sub(bytes, off)))

# The bytes of the next copy of a read of `len` bytes with `done` of them copied, through a
# staging buffer of `piece` bytes.
def kv_piece_len(+done: Nat, +len: Nat, +piece: Nat) -> Nat:
  Nat.min(piece, Nat.sub(len, done))

def kc_inside(+o: Nat, +n: Nat, +b: Nat, ok: {kv_range_inside(o, n, b) == True{} : Bool}) -> {Nat.is_le(Nat.add(o, n), b) == True{} : Bool}:
  match o b:
    case 0n 0n:
      ok
    case 0n 1n+bp:
      ok
    case 1n+op 0n:
      L.bool_absurd({Nat.is_le(1n+Nat.add(op, n), 0n) == True{} : Bool}, ok)
    case 1n+op 1n+bp:
      kc_inside(op, n, bp, ok)

# promise: a range the read-back admits ends inside the cache allocation it reads
# harm: a device-to-host copy would read past the cache, from another buffer or an unmapped address
law kv_range_inside_ends_inside:
  for +o: Nat
  for +n: Nat
  for +b: Nat
  for ok: {kv_range_inside(o, n, b) == True{} : Bool}
  {Nat.is_le(Nat.add(o, n), b) == True{} : Bool}

def kv_range_inside_ends_inside(o, n, b, ok):
  kc_inside(o, n, b, ok)

# promise: each copy of a read ends inside the range being read, whatever the staging buffer's size
# harm: the last copy of a head's batch would read past the batch, and past the cache for the last one
law kv_piece_within:
  for +done: Nat
  for +len: Nat
  for +piece: Nat
  {Bool.or(Nat.is_lt(len, done), Nat.is_le(Nat.add(done, kv_piece_len(done, len, piece)), len)) == True{} : Bool}

def kv_piece_within(done, len, piece):
  kc_fit_le(done, len, piece)

# promise: a copy of the read-back never takes more bytes than the staging buffer holds
# harm: the device-to-host copy would overrun the pinned staging buffer, or be refused and fail the calibration
law kv_piece_fits_stage:
  for +done: Nat
  for +len: Nat
  for +piece: Nat
  {Nat.is_le(kv_piece_len(done, len, piece), piece) == True{} : Bool}

def kv_piece_fits_stage(done, len, piece):
  kc_min_le_l(piece, Nat.sub(len, done))

# promise: while bytes of a read are left and the staging buffer holds at least one byte, each copy moves at least one byte
# harm: a copy of zero bytes would leave the read where it was, and the calibration's read-back would never finish
law kv_piece_progress:
  for +done: Nat
  for +len: Nat
  for +p: Nat
  {Bool.or(Nat.is_le(len, done), Nat.is_lt(0n, kv_piece_len(done, len, 1n+p))) == True{} : Bool}

def kv_piece_progress(done, len, p):
  match done len:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+dp 1n+lp:
      kv_piece_progress(dp, lp, p)

# Helper lemmas of the batch law below (erased before Rust codegen).

def kc_or_l(+x: Bool, +y: Bool, h: {x == True{} : Bool}) -> {Bool.or(x, y) == True{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      L.bool_absurd({Bool.or(False{}, y) == True{} : Bool}, h)

def kc_or_r(+x: Bool, +y: Bool, h: {y == True{} : Bool}) -> {Bool.or(x, y) == True{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      h

# a < b gives a + 1 <= b
def kc_lt_succ_le(+a: Nat, +b: Nat, h: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_le(1n+a, b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      L.bool_absurd({Nat.is_le(1n, 0n) == True{} : Bool}, h)
    case 0n 1n+bp:
      L.nat_le_zero(bp)
    case 1n+ap 0n:
      L.bool_absurd({Nat.is_le(1n+1n+ap, 0n) == True{} : Bool}, h)
    case 1n+ap 1n+bp:
      kc_lt_succ_le(ap, bp, h)

# two numbers are equal, or one is below the other
def kc_tri(+a: Nat, +b: Nat) -> {Bool.or(Nat.is_eq(a, b), Bool.or(Nat.is_lt(a, b), Nat.is_lt(b, a))) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+_: {==}
    case 1n+_ 0n: {==}
    case 1n+ap 1n+bp:
      kc_tri(ap, bp)

# adding n to both sides leaves a comparison as it was
def kc_lt_shift(+n: Nat, +a: Nat, +b: Nat, +c: Nat, +d: Nat) -> {Nat.is_lt(Nat.add(a, b), Nat.add(c, d)) == Nat.is_lt(Nat.add(Nat.add(n, a), b), Nat.add(Nat.add(n, c), d)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      kc_lt_shift(np, a, b, c, d)

# Two (chunk, kv head) pairs with heads below n are the same pair, or the index c * n + g of one
# lies below the other's.
def kc_idx_tri(+c1: Nat, +g1: Nat, +c2: Nat, +g2: Nat, +n: Nat, h1: {Nat.is_lt(g1, n) == True{} : Bool}, h2: {Nat.is_lt(g2, n) == True{} : Bool}) -> {Bool.or(Bool.and(Nat.is_eq(c1, c2), Nat.is_eq(g1, g2)), Bool.or(Nat.is_lt(Nat.add(Nat.mul(c1, n), g1), Nat.add(Nat.mul(c2, n), g2)), Nat.is_lt(Nat.add(Nat.mul(c2, n), g2), Nat.add(Nat.mul(c1, n), g1)))) == True{} : Bool}:
  match c1 c2:
    case 0n 0n:
      kc_tri(g1, g2)
    case 0n 1n+c2p:
      kc_or_l(
        Nat.is_lt(g1, Nat.add(Nat.add(n, Nat.mul(c2p, n)), g2)),
        Nat.is_lt(Nat.add(Nat.add(n, Nat.mul(c2p, n)), g2), g1),
        L.nat_lt_le_trans(g1, n, Nat.add(Nat.add(n, Nat.mul(c2p, n)), g2), h1,
          L.nat_le_trans(n, Nat.add(n, Nat.mul(c2p, n)), Nat.add(Nat.add(n, Nat.mul(c2p, n)), g2),
            L.nat_add_le_r(n, Nat.mul(c2p, n)), L.nat_add_le_r(Nat.add(n, Nat.mul(c2p, n)), g2))))
    case 1n+c1p 0n:
      kc_or_r(
        Nat.is_lt(Nat.add(Nat.add(n, Nat.mul(c1p, n)), g1), g2),
        Nat.is_lt(g2, Nat.add(Nat.add(n, Nat.mul(c1p, n)), g1)),
        L.nat_lt_le_trans(g2, n, Nat.add(Nat.add(n, Nat.mul(c1p, n)), g1), h2,
          L.nat_le_trans(n, Nat.add(n, Nat.mul(c1p, n)), Nat.add(Nat.add(n, Nat.mul(c1p, n)), g1),
            L.nat_add_le_r(n, Nat.mul(c1p, n)), L.nat_add_le_r(Nat.add(n, Nat.mul(c1p, n)), g1))))
    case 1n+c1p 1n+c2p:
      %kc_lt_shift(n, Nat.mul(c1p, n), g1, Nat.mul(c2p, n), g2) : {Bool.or(Bool.and(Nat.is_eq(c1p, c2p), Nat.is_eq(g1, g2)), Bool.or(_, Nat.is_lt(Nat.add(Nat.add(n, Nat.mul(c2p, n)), g2), Nat.add(Nat.add(n, Nat.mul(c1p, n)), g1)))) == True{} : Bool}
      %kc_lt_shift(n, Nat.mul(c2p, n), g2, Nat.mul(c1p, n), g1) : {Bool.or(Bool.and(Nat.is_eq(c1p, c2p), Nat.is_eq(g1, g2)), Bool.or(Nat.is_lt(Nat.add(Nat.mul(c1p, n), g1), Nat.add(Nat.mul(c2p, n), g2)), _)) == True{} : Bool}
      kc_idx_tri(c1p, g1, c2p, g2, n, h1, h2)

# i < j: the h bytes from i * h end at or before j * h.
def kc_off_apart(+i: Nat, +j: Nat, +h: Nat, lt: {Nat.is_lt(i, j) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(i, h), h), Nat.mul(j, h)) == True{} : Bool}:
  %L.nat_add_comm(h, Nat.mul(i, h)) : {Nat.is_le(_, Nat.mul(j, h)) == True{} : Bool}
  L.nat_mul_le_mono_r(1n+i, j, h, kc_lt_succ_le(i, j, lt))

# The same pair, or two indices one below the other: the same pair, or two ranges of h bytes
# apart.
def kc_apart_by(+e: Bool, +x: Bool, +y: Bool, +i: Nat, +j: Nat, +h: Nat, he: {Bool.or(e, Bool.or(x, y)) == True{} : Bool}, ex: {Nat.is_lt(i, j) == x : Bool}, ey: {Nat.is_lt(j, i) == y : Bool}) -> {Bool.or(e, Bool.or(Nat.is_le(Nat.add(Nat.mul(i, h), h), Nat.mul(j, h)), Nat.is_le(Nat.add(Nat.mul(j, h), h), Nat.mul(i, h)))) == True{} : Bool}:
  match e:
    case True{}:
      {==}
    case False{}:
      match x:
        case True{}:
          kc_or_l(Nat.is_le(Nat.add(Nat.mul(i, h), h), Nat.mul(j, h)), Nat.is_le(Nat.add(Nat.mul(j, h), h), Nat.mul(i, h)), kc_off_apart(i, j, h, ex))
        case False{}:
          match y:
            case True{}:
              kc_or_r(Nat.is_le(Nat.add(Nat.mul(i, h), h), Nat.mul(j, h)), Nat.is_le(Nat.add(Nat.mul(j, h), h), Nat.mul(i, h)), kc_off_apart(j, i, h, ey))
            case False{}:
              L.bool_absurd({Bool.or(False{}, Bool.or(Nat.is_le(Nat.add(Nat.mul(i, h), h), Nat.mul(j, h)), Nat.is_le(Nat.add(Nat.mul(j, h), h), Nat.mul(i, h)))) == True{} : Bool}, he)

# promise: the batches of two different (chunk, kv head) pairs of a cache never overlap, whatever order the layout puts them in
# harm: the read-back would fold one head's or one chunk's bytes into another's maxima, measure the wrong batch without a sign, and a V multiplier could clip
law kv_batches_apart:
  for +c1: Nat
  for +g1: Nat
  for +c2: Nat
  for +g2: Nat
  for +n: Nat
  for +h: Nat
  for h1: {Nat.is_lt(g1, n) == True{} : Bool}
  for h2: {Nat.is_lt(g2, n) == True{} : Bool}
  {Bool.or(Bool.and(Nat.is_eq(c1, c2), Nat.is_eq(g1, g2)), Bool.or(Nat.is_le(Nat.add(kv_batch_off(c1, g1, n, h), h), kv_batch_off(c2, g2, n, h)), Nat.is_le(Nat.add(kv_batch_off(c2, g2, n, h), h), kv_batch_off(c1, g1, n, h)))) == True{} : Bool}

def kv_batches_apart(c1, g1, c2, g2, n, h, h1, h2):
  kc_apart_by(
    Bool.and(Nat.is_eq(c1, c2), Nat.is_eq(g1, g2)),
    Nat.is_lt(Nat.add(Nat.mul(c1, n), g1), Nat.add(Nat.mul(c2, n), g2)),
    Nat.is_lt(Nat.add(Nat.mul(c2, n), g2), Nat.add(Nat.mul(c1, n), g1)),
    Nat.add(Nat.mul(c1, n), g1),
    Nat.add(Nat.mul(c2, n), g2),
    h,
    kc_idx_tri(c1, g1, c2, g2, n, h1, h2),
    {==},
    {==})
