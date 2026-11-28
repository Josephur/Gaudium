# Inkling 2 routed MoE execution, packed 4-bit/8-bit slot addressing, on-card softcap/sampler
# geometry, and routed MTP draft-width selection.
# Every public definition here is called by `engine::inkling2`, `engine::inkling2::packer`, `main.rs`,
# or `server/mod.rs`; every law is proved in Bend 2 + Lean (`--verdict`) and erased before Rust emission.

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. MoE execution mode
# ============================================================================

# Operator setting for mixture-of-experts execution (`--moe-form`):
#   Auto   - family default (routed for Inkling 2 when geometry fits; dense otherwise)
#   Dense  - force dense-masked wide BF16 experts
#   Routed - request routed experts (refused unless geometry is supported)
type IkMoeForm is Data:
  IkMoeAuto{}
  IkMoeDense{}
  IkMoeRouted{}

# Decision of the Inkling 2 MoE selector: run dense BF16, run routed W4/W8, or refuse an invalid request.
type IkMoeDecision is Data:
  IkDecDense{}
  IkDecRouted{}
  IkDecRefused{}

# Full three-way MoE selection rule used by the loader: auto falls back to dense, while explicit routed refuses invalid geometry.
def ikr_decide_moe(is_inkling2: Bool, form: IkMoeForm, dims_ok: Bool) -> IkMoeDecision:
  match form:
    case IkMoeAuto{}:
      Bool.pick(IkMoeDecision, Bool.and(is_inkling2, dims_ok), IkDecRouted{}, IkDecDense{})
    case IkMoeDense{}:
      IkDecDense{}
    case IkMoeRouted{}:
      Bool.pick(IkMoeDecision, dims_ok, IkDecRouted{}, IkDecRefused{})

def ikr_dec_is_routed(dec: IkMoeDecision) -> Bool:
  match dec:
    case IkDecRouted{}:
      True{}
    case IkDecDense{}:
      False{}
    case IkDecRefused{}:
      False{}

# Whether the Inkling 2 engine executes its routed expert path (projection of `ikr_decide_moe`).
def ikr_use_routed(is_inkling2: Bool, form: IkMoeForm, dims_ok: Bool) -> Bool:
  ikr_dec_is_routed(ikr_decide_moe(is_inkling2, form, dims_ok))

# ============================================================================
# 2. Routed geometry guard & tile counts
# ============================================================================

# Shapes covered by the Inkling 2 routed-expert kernels (`gaudium_ik_w4_*` / `gaudium_ik_w8_*`):
# hidden width in whole 256-channel down tiles, expert intermediate in whole 128-channel tiles,
# experts in non-zero multiples of 128 up to `max_ne`, and top-k selection in `1..min(max_k, n_experts)`.
def ikr_dims_valid(+hid: Nat, +ef: Nat, +ne: Nat, +top_k: Nat, +max_ne: Nat, +max_k: Nat) -> Bool:
  Bool.and(
    Bool.and(Nat.is_gt(hid, 0n), Nat.is_eq(Nat.mod(hid, 256n), 0n)),
    Bool.and(
      Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n)),
      Bool.and(
        Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))),
        Bool.and(Nat.is_ge(top_k, 1n), Bool.and(Nat.is_le(top_k, max_k), Nat.is_le(top_k, ne)))
      )
    )
  )

# Gate+up tiles per expert (128 gate + 128 up channels = 256 lanes per tile).
def ikr_gu_tiles(ef: Nat) -> Nat:
  Nat.div(ef, 128n)

# Down projection tiles per expert (256 output channels per tile).
def ikr_dn_tiles(hid: Nat) -> Nat:
  Nat.div(hid, 256n)

# ============================================================================
# 3. Packed 4-bit (W4 / NVFP4) and 8-bit (W8) expert pool byte sizes & offsets
# ============================================================================

# Total bytes of a packed 4-bit expert code pool `{256, in_cols / 2, tiles, ne}` (two nibbles/byte,
# 256 byte lanes per column pair). Refused on u64 overflow.
# rusty-bends: total
def ikr_w4_code_bytes(+in_cols: Nat, +tiles: Nat, +ne: Nat) -> Nat:
  Nat.mul(Nat.mul(Nat.mul(ne, tiles), Nat.div(in_cols, 2n)), 256n)

# Total bytes of a packed 8-bit expert code pool `{256, in_cols, tiles, ne}`. Refused on u64 overflow.
# rusty-bends: total
def ikr_w8_code_bytes(+in_cols: Nat, +tiles: Nat, +ne: Nat) -> Nat:
  Nat.mul(Nat.mul(Nat.mul(ne, tiles), in_cols), 256n)

# Total bytes of a packed fp16 expert scale pool `{256, in_cols / blk, tiles, ne}`. Refused on u64 overflow.
# rusty-bends: total
def ikr_scale_bytes(+in_cols: Nat, +blk: Nat, +tiles: Nat, +ne: Nat) -> Nat:
  Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), Nat.div(in_cols, blk)), 256n), 2n)

# Byte offset of the 128-lane vector for `(expert e, tile tt, input column col)` inside a W4 code pool,
# matching `SW_UNPACK | SW_UNPCK_4_TO_8`: `cp = col / 2`, `half = col % 2` selects `[0..128)` vs `[128..256)`.
# rusty-bends: total
def ikr_w4_col_off(+e: Nat, +tiles: Nat, +cp_cnt: Nat, +tt: Nat, +col: Nat) -> Nat:
  +cp = Nat.div(col, 2n)
  +half = Nat.mod(col, 2n)
  Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), cp_cnt), cp), 256n), Nat.mul(128n, half))

# Byte offset of the 256-lane vector for `(expert e, tile tt, input column col)` inside a W8 code pool.
# rusty-bends: total
def ikr_w8_col_off(+e: Nat, +tiles: Nat, +in_cols: Nat, +tt: Nat, +col: Nat) -> Nat:
  Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), in_cols), col), 256n)

# Half-index offset (in u16 elements) of the 256-half scale vector `(e, tt, blk_idx)` in `{256, nblk, tiles, ne}`.
# rusty-bends: total
def ikr_scale_vec_off(+e: Nat, +tiles: Nat, +nblk: Nat, +tt: Nat, +b: Nat) -> Nat:
  Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), nblk), b), 256n)

# Position `p in 0..256` inside a scale vector for byte lane `i = 4*l + j` so `v_convert_f16_to_f32_all`
# unpacks accumulator vector pairs (0, 1) in `0..128` and (2, 3) in `128..256`.
def ikr_scale_lane_pos(+i: Nat) -> Nat:
  +j = Nat.mod(i, 4n)
  +l = Nat.div(i, 4n)
  +hi = Bool.pick(Nat, Nat.is_lt(j, 2n), 0n, 128n)
  Nat.add(hi, Nat.add(Nat.mul(2n, l), Nat.mod(j, 2n)))

# Source weight matrix row read for `(tt, lane)` during expert packing: GU interleaves 128-channel
# gate+up pairs with `(j & 1) * 64 + l`; down projection unpacks 256-row tiles directly.
def ikr_src_row(is_gu: Bool, +tt: Nat, +lane: Nat) -> Nat:
  match is_gu:
    case True{}:
      +l = Nat.div(lane, 4n)
      +j = Nat.mod(lane, 4n)
      +ch = Nat.add(Nat.mul(128n, tt), Nat.add(Nat.mul(Nat.mod(j, 2n), 64n), l))
      Nat.add(Nat.mul(2n, ch), Nat.div(j, 2n))
    case False{}:
      +l = Nat.div(lane, 4n)
      +j = Nat.mod(lane, 4n)
      Nat.add(Nat.mul(256n, tt), Nat.add(Nat.mul(Nat.div(j, 2n), 128n), Nat.add(Nat.mul(2n, l), Nat.mod(j, 2n))))

# ============================================================================
# 4. Dispatch plan, on-card sampler vocab guard, and default MTP verify width
# ============================================================================

# Elements in `plan i32 {1 + 5 * K * T}` produced by `gaudium_ik_w4_plan`.
# rusty-bends: total
def ikr_plan_len(+k: Nat, +t: Nat) -> Nat:
  Nat.add(1n, Nat.mul(5n, Nat.mul(k, t)))

# Whether `(vocab_out, vocab_pad)` satisfies the on-card softcap + Gumbel-max sampler contract:
# at least one emitted token, all emitted ids inside `vocab_pad`, and `vocab_pad` aligned to 128 lanes.
def ikr_sampler_vocab_ok(+vocab_out: Nat, +vocab_pad: Nat) -> Bool:
  Bool.and(
    Nat.is_gt(vocab_out, 0n),
    Bool.and(Nat.is_le(vocab_out, vocab_pad), Nat.is_eq(Nat.mod(vocab_pad, 128n), 0n))
  )

# Default speculative draft width K per verify step for a model's own MTP block drafter.
# (Dense execution runs at full block - 1; routed verify scales with active expert union.)
def ikr_mtp_width(routed: Bool, +block: Nat) -> Nat:
  match routed:
    case False{}:
      Nat.sub(block, 1n)
    case True{}:
      match block:
        case 0n:
          0n
        case 1n+bp:
          match bp:
            case 0n:
              0n
            case 1n+bpp:
              Nat.min(1n, 1n+bpp)

# ============================================================================
# 5. Proof helpers (erased before Rust codegen)
# ============================================================================

def ikrp_and_l(a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}: {==}
    case False{}: h

def ikrp_and_r(a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case True{}: h
    case False{}: L.bool_absurd({b == True{} : Bool}, h)

def ikrp_eq_refl(+a: Nat) -> {Nat.is_eq(a, a) == True{} : Bool}:
  match a:
    case 0n: {==}
    case 1n+ap: ikrp_eq_refl(ap)

def ikrp_ne_of_eq(+a: Nat, +b: Nat, e: {a == b : Nat}) -> {Nat.is_ne(a, b) == False{} : Bool}:
  %e : {Nat.is_ne(a, _) == False{} : Bool}
  %Equal.sym(Bool, Nat.is_eq(a, a), True{}, ikrp_eq_refl(a)) : {Bool.not(_) == False{} : Bool}
  {==}

def ikrp_whole(+c: Nat, +bp: Nat, hm: {Nat.mod(c, 1n+bp) == 0n : Nat}) -> {Nat.mul(Nat.div(c, 1n+bp), 1n+bp) == c : Nat}:
  %hm : {Nat.add(_, Nat.mul(Nat.div(c, 1n+bp), 1n+bp)) == c : Nat}
  L.nat_div_mod(c, bp)

def ikrp_div_eq_succ(+a: Nat, +bp: Nat, +hb: {Nat.is_eq(Nat.mod(a, 1n+bp), 0n) == True{} : Bool}) -> {Nat.is_ne(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), a) == False{} : Bool}:
  ikrp_ne_of_eq(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), a, ikrp_whole(a, bp, G.g_nat_eq_of_is_eq(Nat.mod(a, 1n+bp), 0n, hb)))

def ikrp_dims_c1(a: Bool, b: Bool, c: Bool, d: Bool) -> {Bool.and(Bool.and(a, Bool.and(b, Bool.and(c, d))), Bool.not(a)) == False{} : Bool}:
  match a b c d:
    case False{} _ _ _: {==}
    case True{} False{} _ _: {==}
    case True{} True{} False{} _: {==}
    case True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{}: {==}

def ikrp_dims_c2(a: Bool, b: Bool, c: Bool, d: Bool) -> {Bool.and(Bool.and(a, Bool.and(b, Bool.and(c, d))), Bool.not(b)) == False{} : Bool}:
  match a b c d:
    case False{} _ _ _: {==}
    case True{} False{} _ _: {==}
    case True{} True{} False{} _: {==}
    case True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{}: {==}

def ikrp_dims_c3(a: Bool, b: Bool, c: Bool, d: Bool) -> {Bool.and(Bool.and(a, Bool.and(b, Bool.and(c, d))), Bool.not(c)) == False{} : Bool}:
  match a b c d:
    case False{} _ _ _: {==}
    case True{} False{} _ _: {==}
    case True{} True{} False{} _: {==}
    case True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{}: {==}

def ikrp_dims_c4(a: Bool, b: Bool, c: Bool, d: Bool) -> {Bool.and(Bool.and(a, Bool.and(b, Bool.and(c, d))), Bool.not(d)) == False{} : Bool}:
  match a b c d:
    case False{} _ _ _: {==}
    case True{} False{} _ _: {==}
    case True{} True{} False{} _: {==}
    case True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{}: {==}

def ikrp_mul128_half_le(+h: Nat, hh: {Nat.is_lt(h, 2n) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(128n, h), 128n), 256n) == True{} : Bool}:
  match h:
    case 0n: {==}
    case 1n+hp:
      match hp:
        case 0n: {==}
        case 1n+hpp:
          G.g_nat_lt_zero_absurd({Nat.is_le(Nat.add(Nat.mul(128n, 2n+hpp), 128n), 256n) == True{} : Bool}, hpp, hh)

def ikrp_succ_mul256(+idx: Nat, +n: Nat, hlt: {Nat.is_lt(idx, n) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(idx, 256n), 256n), Nat.mul(n, 256n)) == True{} : Bool}:
  +hs = G.g_nat_lt_imp_succ_le(idx, n, hlt)
  %L.nat_mul_add_dist_r(idx, 1n, 256n) : {Nat.is_le(_, Nat.mul(n, 256n)) == True{} : Bool}
  L.nat_mul_le_mono_r(Nat.add(idx, 1n), n, 256n, hs)

def ikrp_idx3_le256(
  +e: Nat,
  +ne: Nat,
  +tt: Nat,
  +tp: Nat,
  +c: Nat,
  +cp: Nat,
  he: {Nat.is_lt(e, ne) == True{} : Bool},
  htt: {Nat.is_lt(tt, 1n+tp) == True{} : Bool},
  hc: {Nat.is_lt(c, 1n+cp) == True{} : Bool}
) -> {Nat.is_le(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, 1n+tp), tt), 1n+cp), c), 256n), 256n), Nat.mul(Nat.mul(Nat.mul(ne, 1n+tp), 1n+cp), 256n)) == True{} : Bool}:
  +idx0 = Nat.add(Nat.mul(e, 1n+tp), tt)
  +h0 = L.nat_split_lt(e, tt, ne, tp, he, htt)
  +idx1 = Nat.add(Nat.mul(idx0, 1n+cp), c)
  +h1 = L.nat_split_lt(idx0, c, Nat.mul(ne, 1n+tp), cp, h0, hc)
  ikrp_succ_mul256(idx1, Nat.mul(Nat.mul(ne, 1n+tp), 1n+cp), h1)

def ikrp_2l_lt(+i: Nat, hi: {Nat.is_lt(i, 256n) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n)), 128n) == True{} : Bool}:
  +j = Nat.mod(i, 4n)
  +l = Nat.div(i, 4n)
  +m = Nat.mod(j, 2n)
  +hl = L.nat_div_lt_of_lt_mul(i, 64n, 3n, hi)
  +hm = L.nat_mod_lt(j, 1n)
  %L.nat_mul_comm(l, 2n) : {Nat.is_lt(Nat.add(_, m), 128n) == True{} : Bool}
  L.nat_split_lt(l, m, 64n, 1n, hl, hm)

def ikrp_4_plus_1_eq_5(+p: Nat) -> {Nat.add(Nat.mul(4n, p), p) == Nat.mul(5n, p) : Nat}:
  %L.nat_add_zero_r(p) : {Nat.add(Nat.mul(4n, p), _) == Nat.mul(5n, p) : Nat}
  Equal.sym(Nat, Nat.mul(5n, p), Nat.add(Nat.mul(4n, p), Nat.mul(1n, p)), L.nat_mul_add_dist_r(4n, 1n, p))

def ikrp_w4_in_go(
  +e: Nat,
  +ne: Nat,
  +tt: Nat,
  +tp: Nat,
  +c: Nat,
  +cp: Nat,
  he: {Nat.is_lt(e, ne) == True{} : Bool},
  htt: {Nat.is_lt(tt, 1n+tp) == True{} : Bool},
  hc: {Nat.is_lt(c, Nat.mul(1n+cp, 2n)) == True{} : Bool}
) -> {Nat.is_le(Nat.add(ikr_w4_col_off(e, 1n+tp, 1n+cp, tt, c), 128n), ikr_w4_code_bytes(Nat.mul(1n+cp, 2n), 1n+tp, ne)) == True{} : Bool}:
  +col_pair = Nat.div(c, 2n)
  +half = Nat.mod(c, 2n)
  +hcp = L.nat_div_lt_of_lt_mul(c, 1n+cp, 1n, hc)
  +hhalf = L.nat_mod_lt(c, 1n)
  +idx0 = Nat.add(Nat.mul(e, 1n+tp), tt)
  +h0 = L.nat_split_lt(e, tt, ne, tp, he, htt)
  +idx1 = Nat.add(Nat.mul(idx0, 1n+cp), col_pair)
  +h1 = L.nat_split_lt(idx0, col_pair, Nat.mul(ne, 1n+tp), cp, h0, hcp)
  +h_tail = L.nat_le_trans(
    Nat.add(Nat.mul(idx1, 256n), Nat.add(Nat.mul(128n, half), 128n)),
    Nat.add(Nat.mul(idx1, 256n), 256n),
    Nat.mul(Nat.mul(Nat.mul(ne, 1n+tp), 1n+cp), 256n),
    L.nat_add_le_mono(
      Nat.mul(idx1, 256n),
      Nat.mul(idx1, 256n),
      Nat.add(Nat.mul(128n, half), 128n),
      256n,
      L.nat_le_refl(Nat.mul(idx1, 256n)),
      ikrp_mul128_half_le(half, hhalf)
    ),
    ikrp_succ_mul256(idx1, Nat.mul(Nat.mul(ne, 1n+tp), 1n+cp), h1)
  )
  %Equal.sym(Nat, Nat.add(Nat.add(Nat.mul(idx1, 256n), Nat.mul(128n, half)), 128n), Nat.add(Nat.mul(idx1, 256n), Nat.add(Nat.mul(128n, half), 128n)), L.nat_add_assoc(Nat.mul(idx1, 256n), Nat.mul(128n, half), 128n)) : {Nat.is_le(_, ikr_w4_code_bytes(Nat.mul(1n+cp, 2n), 1n+tp, ne)) == True{} : Bool}
  %Equal.sym(Nat, Nat.div(Nat.mul(1n+cp, 2n), 2n), 1n+cp, L.nat_div_mul(1n+cp, 1n)) : {Nat.is_le(Nat.add(Nat.mul(idx1, 256n), Nat.add(Nat.mul(128n, half), 128n)), Nat.mul(Nat.mul(Nat.mul(ne, 1n+tp), _), 256n)) == True{} : Bool}
  h_tail

def ikrp_w4_col_go(
  b1: Bool,
  b2: Bool,
  b3: Bool,
  +e: Nat,
  +ne: Nat,
  +tt: Nat,
  +tiles: Nat,
  +c: Nat,
  +cp_cnt: Nat,
  +h1: {Nat.is_lt(e, ne) == b1 : Bool},
  +h2: {Nat.is_lt(tt, tiles) == b2 : Bool},
  +h3: {Nat.is_lt(c, Nat.mul(cp_cnt, 2n)) == b3 : Bool}
) -> {Bool.and(Bool.and(b1, Bool.and(b2, b3)), Nat.is_lt(ikr_w4_code_bytes(Nat.mul(cp_cnt, 2n), tiles, ne), Nat.add(ikr_w4_col_off(e, tiles, cp_cnt, tt, c), 128n))) == False{} : Bool}:
  match b1 b2 b3:
    case False{} _ _: {==}
    case True{} False{} _: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}:
      match tiles cp_cnt:
        case 0n _:
          G.g_nat_lt_zero_absurd(
            {Nat.is_lt(ikr_w4_code_bytes(Nat.mul(cp_cnt, 2n), 0n, ne), Nat.add(ikr_w4_col_off(e, 0n, cp_cnt, tt, c), 128n)) == False{} : Bool},
            tt,
            h2
          )
        case 1n+tp 0n:
          G.g_nat_lt_zero_absurd(
            {Nat.is_lt(ikr_w4_code_bytes(Nat.mul(0n, 2n), 1n+tp, ne), Nat.add(ikr_w4_col_off(e, 1n+tp, 0n, tt, c), 128n)) == False{} : Bool},
            c,
            h3
          )
        case 1n+tp 1n+cp:
          L.nat_le_lt_false(
            ikr_w4_code_bytes(Nat.mul(1n+cp, 2n), 1n+tp, ne),
            Nat.add(ikr_w4_col_off(e, 1n+tp, 1n+cp, tt, c), 128n),
            ikrp_w4_in_go(e, ne, tt, tp, c, cp, h1, h2, h3)
          )

def ikrp_w8_col_go(
  b1: Bool,
  b2: Bool,
  b3: Bool,
  +e: Nat,
  +ne: Nat,
  +tt: Nat,
  +tiles: Nat,
  +c: Nat,
  +in_cols: Nat,
  +h1: {Nat.is_lt(e, ne) == b1 : Bool},
  +h2: {Nat.is_lt(tt, tiles) == b2 : Bool},
  +h3: {Nat.is_lt(c, in_cols) == b3 : Bool}
) -> {Bool.and(Bool.and(b1, Bool.and(b2, b3)), Nat.is_lt(ikr_w8_code_bytes(in_cols, tiles, ne), Nat.add(ikr_w8_col_off(e, tiles, in_cols, tt, c), 256n))) == False{} : Bool}:
  match b1 b2 b3:
    case False{} _ _: {==}
    case True{} False{} _: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}:
      match tiles in_cols:
        case 0n _:
          G.g_nat_lt_zero_absurd(
            {Nat.is_lt(ikr_w8_code_bytes(in_cols, 0n, ne), Nat.add(ikr_w8_col_off(e, 0n, in_cols, tt, c), 256n)) == False{} : Bool},
            tt,
            h2
          )
        case 1n+tp 0n:
          G.g_nat_lt_zero_absurd(
            {Nat.is_lt(ikr_w8_code_bytes(0n, 1n+tp, ne), Nat.add(ikr_w8_col_off(e, 1n+tp, 0n, tt, c), 256n)) == False{} : Bool},
            c,
            h3
          )
        case 1n+tp 1n+ic:
          L.nat_le_lt_false(
            ikr_w8_code_bytes(1n+ic, 1n+tp, ne),
            Nat.add(ikr_w8_col_off(e, 1n+tp, 1n+ic, tt, c), 256n),
            ikrp_idx3_le256(e, ne, tt, tp, c, ic, h1, h2, h3)
          )

def ikrp_scale_in_go(
  +e: Nat,
  +ne: Nat,
  +tt: Nat,
  +tp: Nat,
  +b: Nat,
  +nbp: Nat,
  +bp: Nat,
  he: {Nat.is_lt(e, ne) == True{} : Bool},
  htt: {Nat.is_lt(tt, 1n+tp) == True{} : Bool},
  hb: {Nat.is_lt(b, 1n+nbp) == True{} : Bool}
) -> {Nat.is_le(Nat.mul(Nat.add(ikr_scale_vec_off(e, 1n+tp, 1n+nbp, tt, b), 256n), 2n), ikr_scale_bytes(Nat.mul(1n+nbp, 1n+bp), 1n+bp, 1n+tp, ne)) == True{} : Bool}:
  +hle = L.nat_mul_le_mono_r(
    Nat.add(ikr_scale_vec_off(e, 1n+tp, 1n+nbp, tt, b), 256n),
    Nat.mul(Nat.mul(Nat.mul(ne, 1n+tp), 1n+nbp), 256n),
    2n,
    ikrp_idx3_le256(e, ne, tt, tp, b, nbp, he, htt, hb)
  )
  %Equal.sym(Nat, Nat.div(Nat.mul(1n+nbp, 1n+bp), 1n+bp), 1n+nbp, L.nat_div_mul(1n+nbp, bp)) : {Nat.is_le(Nat.mul(Nat.add(ikr_scale_vec_off(e, 1n+tp, 1n+nbp, tt, b), 256n), 2n), Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, 1n+tp), _), 256n), 2n)) == True{} : Bool}
  hle

def ikrp_scale_off_go(
  b1: Bool,
  b2: Bool,
  b3: Bool,
  +e: Nat,
  +ne: Nat,
  +tt: Nat,
  +tiles: Nat,
  +b: Nat,
  +nblk: Nat,
  +bp: Nat,
  +h1: {Nat.is_lt(e, ne) == b1 : Bool},
  +h2: {Nat.is_lt(tt, tiles) == b2 : Bool},
  +h3: {Nat.is_lt(b, nblk) == b3 : Bool}
) -> {Bool.and(Bool.and(b1, Bool.and(b2, b3)), Nat.is_lt(ikr_scale_bytes(Nat.mul(nblk, 1n+bp), 1n+bp, tiles, ne), Nat.mul(Nat.add(ikr_scale_vec_off(e, tiles, nblk, tt, b), 256n), 2n))) == False{} : Bool}:
  match b1 b2 b3:
    case False{} _ _: {==}
    case True{} False{} _: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}:
      match tiles nblk:
        case 0n _:
          G.g_nat_lt_zero_absurd(
            {Nat.is_lt(ikr_scale_bytes(Nat.mul(nblk, 1n+bp), 1n+bp, 0n, ne), Nat.mul(Nat.add(ikr_scale_vec_off(e, 0n, nblk, tt, b), 256n), 2n)) == False{} : Bool},
            tt,
            h2
          )
        case 1n+tp 0n:
          G.g_nat_lt_zero_absurd(
            {Nat.is_lt(ikr_scale_bytes(Nat.mul(0n, 1n+bp), 1n+bp, 1n+tp, ne), Nat.mul(Nat.add(ikr_scale_vec_off(e, 1n+tp, 0n, tt, b), 256n), 2n)) == False{} : Bool},
            b,
            h3
          )
        case 1n+tp 1n+nbp:
          L.nat_le_lt_false(
            ikr_scale_bytes(Nat.mul(1n+nbp, 1n+bp), 1n+bp, 1n+tp, ne),
            Nat.mul(Nat.add(ikr_scale_vec_off(e, 1n+tp, 1n+nbp, tt, b), 256n), 2n),
            ikrp_scale_in_go(e, ne, tt, tp, b, nbp, bp, h1, h2, h3)
          )

def ikrp_lane_vec_go(
  +i: Nat,
  v: Bool,
  c: Bool,
  +hv: {Nat.is_lt(i, 256n) == v : Bool},
  +hc: {Nat.is_lt(Nat.mod(i, 4n), 2n) == c : Bool}
) -> {Bool.and(v, Bool.not(Nat.is_lt(ikr_scale_lane_pos(i), 256n))) == False{} : Bool}:
  match v c:
    case False{} _: {==}
    case True{} True{}:
      +low = Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n))
      +hlt256 = L.nat_lt_trans(low, 128n, 256n, ikrp_2l_lt(i, hv), {==})
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(i, 4n), 2n), True{}, hc) : {Bool.not(Nat.is_lt(Nat.add(Bool.pick(Nat, _, 0n, 128n), low), 256n)) == False{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(low, 256n), True{}, hlt256) : {Bool.not(_) == False{} : Bool}
      {==}
    case True{} False{}:
      +low = Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n))
      +hlt256 = L.nat_add_lt_l(128n, low, 128n, ikrp_2l_lt(i, hv))
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(i, 4n), 2n), False{}, hc) : {Bool.not(Nat.is_lt(Nat.add(Bool.pick(Nat, _, 0n, 128n), low), 256n)) == False{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(Nat.add(128n, low), 256n), True{}, hlt256) : {Bool.not(_) == False{} : Bool}
      {==}

def ikrp_lane_half_go(
  +i: Nat,
  v: Bool,
  c: Bool,
  +hv: {Nat.is_lt(i, 256n) == v : Bool},
  +hc: {Nat.is_lt(Nat.mod(i, 4n), 2n) == c : Bool}
) -> {Bool.and(v, Bool.xor(Nat.is_lt(Nat.mod(i, 4n), 2n), Nat.is_lt(ikr_scale_lane_pos(i), 128n))) == False{} : Bool}:
  match v c:
    case False{} _: {==}
    case True{} True{}:
      +low = Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n))
      +hlow = ikrp_2l_lt(i, hv)
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(i, 4n), 2n), True{}, hc) : {Bool.xor(_, Nat.is_lt(Nat.add(Bool.pick(Nat, _, 0n, 128n), low), 128n)) == False{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(low, 128n), True{}, hlow) : {Bool.xor(True{}, _) == False{} : Bool}
      {==}
    case True{} False{}:
      +low = Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n))
      +hge = L.nat_le_lt_false(Nat.add(128n, low), 128n, L.nat_add_le_r(128n, low))
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(i, 4n), 2n), False{}, hc) : {Bool.xor(_, Nat.is_lt(Nat.add(Bool.pick(Nat, _, 0n, 128n), low), 128n)) == False{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(Nat.add(128n, low), 128n), False{}, hge) : {Bool.xor(False{}, _) == False{} : Bool}
      {==}

def ikrp_plan_go(
  +k: Nat,
  +t: Nat,
  +u: Nat,
  +rs: Nat,
  b1: Bool,
  b2: Bool,
  +h1: {Nat.is_le(u, Nat.mul(k, t)) == b1 : Bool},
  +h2: {Nat.is_lt(rs, Nat.mul(k, t)) == b2 : Bool}
) -> {Bool.and(Bool.and(b1, b2), Nat.is_le(ikr_plan_len(k, t), Nat.add(Nat.add(1n, Nat.mul(4n, u)), rs))) == False{} : Bool}:
  match b1 b2:
    case False{} _: {==}
    case True{} False{}: {==}
    case True{} True{}:
      +p = Nat.mul(k, t)
      +hsum0 = L.nat_add_lt_le_mono(rs, p, Nat.mul(4n, u), Nat.mul(4n, p), h2, L.nat_mul_le_mono_l(4n, u, p, h1))
      +hsum1 = L.nat_lt_subst_l(Nat.add(rs, Nat.mul(4n, u)), Nat.add(Nat.mul(4n, u), rs), Nat.add(p, Nat.mul(4n, p)), L.nat_add_comm(rs, Nat.mul(4n, u)), hsum0)
      +hsum2 = L.nat_lt_subst_r(Nat.add(Nat.mul(4n, u), rs), Nat.add(p, Nat.mul(4n, p)), Nat.add(Nat.mul(4n, p), p), L.nat_add_comm(p, Nat.mul(4n, p)), hsum1)
      +hsum3 = L.nat_lt_subst_r(Nat.add(Nat.mul(4n, u), rs), Nat.add(Nat.mul(4n, p), p), Nat.mul(5n, p), ikrp_4_plus_1_eq_5(p), hsum2)
      +hlt = L.nat_lt_subst_l(
        Nat.add(1n, Nat.add(Nat.mul(4n, u), rs)),
        Nat.add(Nat.add(1n, Nat.mul(4n, u)), rs),
        Nat.add(1n, Nat.mul(5n, p)),
        Equal.sym(Nat, Nat.add(Nat.add(1n, Nat.mul(4n, u)), rs), Nat.add(1n, Nat.add(Nat.mul(4n, u), rs)), L.nat_add_assoc(1n, Nat.mul(4n, u), rs)),
        L.nat_add_lt_l(1n, Nat.add(Nat.mul(4n, u), rs), Nat.mul(5n, p), hsum3)
      )
      L.nat_lt_le_false(Nat.add(Nat.add(1n, Nat.mul(4n, u)), rs), ikr_plan_len(k, t), hlt)

def ikrp_svocab_grid_go(
  +vo: Nat,
  +vp: Nat,
  ok: Bool,
  +hok: {ikr_sampler_vocab_ok(vo, vp) == ok : Bool}
) -> {Bool.and(ok, Nat.is_ne(Nat.mul(Nat.div(vp, 128n), 128n), vp)) == False{} : Bool}:
  match ok:
    case False{}: {==}
    case True{}:
      +hb = ikrp_and_r(Nat.is_gt(vo, 0n), Bool.and(Nat.is_le(vo, vp), Nat.is_eq(Nat.mod(vp, 128n), 0n)), hok)
      +hm = ikrp_and_r(Nat.is_le(vo, vp), Nat.is_eq(Nat.mod(vp, 128n), 0n), hb)
      ikrp_div_eq_succ(vp, 127n, hm)

def ikrp_svocab_rng_go(
  +vo: Nat,
  +vp: Nat,
  ok: Bool,
  +hok: {ikr_sampler_vocab_ok(vo, vp) == ok : Bool}
) -> {Bool.and(ok, Bool.not(Bool.and(Nat.is_gt(vo, 0n), Nat.is_le(vo, vp)))) == False{} : Bool}:
  match ok:
    case False{}: {==}
    case True{}:
      +ha = ikrp_and_l(Nat.is_gt(vo, 0n), Bool.and(Nat.is_le(vo, vp), Nat.is_eq(Nat.mod(vp, 128n), 0n)), hok)
      +hb = ikrp_and_r(Nat.is_gt(vo, 0n), Bool.and(Nat.is_le(vo, vp), Nat.is_eq(Nat.mod(vp, 128n), 0n)), hok)
      +hc = ikrp_and_l(Nat.is_le(vo, vp), Nat.is_eq(Nat.mod(vp, 128n), 0n), hb)
      %Equal.sym(Bool, Nat.is_gt(vo, 0n), True{}, ha) : {Bool.not(Bool.and(_, Nat.is_le(vo, vp))) == False{} : Bool}
      %Equal.sym(Bool, Nat.is_le(vo, vp), True{}, hc) : {Bool.not(Bool.and(True{}, _)) == False{} : Bool}
      {==}

def ikrp_is_eq(+a: Nat, +b: Nat, e: {a == b : Nat}) -> {Nat.is_eq(a, b) == True{} : Bool}:
  %e : {Nat.is_eq(a, _) == True{} : Bool}
  ikrp_eq_refl(a)

def ikrp_and_true(a: Bool, b: Bool, ha: {a == True{} : Bool}, hb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  match a b:
    case True{} True{}: {==}
    case False{} _: L.bool_absurd({Bool.and(False{}, b) == True{} : Bool}, ha)
    case True{} False{}: L.bool_absurd({Bool.and(True{}, False{}) == True{} : Bool}, hb)

def ikrp_not_true(b: Bool, hb: {b == True{} : Bool}) -> {Bool.not(b) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: L.bool_absurd({Bool.not(False{}) == False{} : Bool}, hb)

def ikrp_mod_mul_zero(+q: Nat, +bp: Nat) -> {Nat.is_eq(Nat.mod(Nat.mul(q, 1n+bp), 1n+bp), 0n) == True{} : Bool}:
  ikrp_is_eq(Nat.mod(Nat.mul(q, 1n+bp), 1n+bp), 0n, L.nat_mod_mul(q, bp))

def ikrp_ge_one(+kp: Nat) -> {Nat.is_ge(1n+kp, 1n) == True{} : Bool}:
  match kp:
    case 0n: {==}
    case 1n+_: {==}

def ikrp_dims_acc_abi(
  +hp: Nat,
  +efp: Nat,
  +nep: Nat,
  +kp: Nat,
  +extra_ne: Nat,
  +extra_k: Nat,
  b_abi: Bool,
  b_k: Bool,
  hb: {Nat.is_le(1n+kp, Nat.mul(1n+nep, 128n)) == b_k : Bool}
) -> {Bool.and(Bool.and(b_abi, b_k), Bool.not(ikr_dims_valid(Nat.mul(1n+hp, 256n), Nat.mul(1n+efp, 128n), Nat.mul(1n+nep, 128n), 1n+kp, Nat.add(Nat.mul(1n+nep, 128n), extra_ne), Nat.add(1n+kp, extra_k)))) == False{} : Bool}:
  match b_abi b_k:
    case False{} _: {==}
    case True{} False{}: {==}
    case True{} True{}:
      +h = Nat.mul(1n+hp, 256n)
      +ef = Nat.mul(1n+efp, 128n)
      +ne = Nat.mul(1n+nep, 128n)
      +k = {1n+kp : Nat}
      +max_ne = Nat.add(ne, extra_ne)
      +max_k = Nat.add(k, extra_k)
      +ca = ikrp_and_true(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n), {==}, ikrp_mod_mul_zero(1n+hp, 255n))
      +cb = ikrp_and_true(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n), {==}, ikrp_mod_mul_zero(1n+efp, 127n))
      +cc = ikrp_and_true(
        Nat.is_gt(ne, 0n),
        Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne)),
        {==},
        ikrp_and_true(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne), ikrp_mod_mul_zero(1n+nep, 127n), L.nat_add_le_r(ne, extra_ne))
      )
      +cd = ikrp_and_true(
        Nat.is_ge(k, 1n),
        Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)),
        ikrp_ge_one(kp),
        ikrp_and_true(Nat.is_le(k, max_k), Nat.is_le(k, ne), L.nat_add_le_r(k, extra_k), hb)
      )
      +hall = ikrp_and_true(
        Bool.and(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n)),
        Bool.and(
          Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n)),
          Bool.and(
            Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))),
            Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)))
          )
        ),
        ca,
        ikrp_and_true(
          Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n)),
          Bool.and(
            Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))),
            Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)))
          ),
          cb,
          ikrp_and_true(
            Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))),
            Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne))),
            cc,
            cd
          )
        )
      )
      ikrp_not_true(ikr_dims_valid(h, ef, ne, k, max_ne, max_k), hall)

def ikrp_add_succ_gt0(+a: Nat, +r: Nat) -> {Nat.is_gt(Nat.add(a, 1n+r), 0n) == True{} : Bool}:
  %L.nat_add_comm(1n+r, a) : {Nat.is_gt(_, 0n) == True{} : Bool}
  {==}

def ikrp_svocab_le_canon(+vp_blk: Nat, +r: Nat, hr: {Nat.is_lt(r, 128n) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(vp_blk, 128n), 1n+r), Nat.mul(1n+vp_blk, 128n)) == True{} : Bool}:
  +hle_r = Equal.trans(Bool, Nat.is_le(r, 127n), Nat.is_lt(r, 128n), True{}, Equal.sym(Bool, Nat.is_lt(r, 128n), Nat.is_le(r, 127n), L.nat_lt_succ_eq_le(r, 127n)), hr)
  +hmono = L.nat_add_le_mono(Nat.mul(vp_blk, 128n), Nat.mul(vp_blk, 128n), 1n+r, 128n, L.nat_le_refl(Nat.mul(vp_blk, 128n)), hle_r)
  L.nat_le_subst_r(
    Nat.add(Nat.mul(vp_blk, 128n), 1n+r),
    Nat.add(Nat.mul(vp_blk, 128n), 128n),
    Nat.add(128n, Nat.mul(vp_blk, 128n)),
    L.nat_add_comm(Nat.mul(vp_blk, 128n), 128n),
    hmono
  )

def ikrp_svocab_acc_go(
  +vp_blk: Nat,
  +r: Nat,
  b_blk: Bool,
  b_r: Bool,
  hr: {Nat.is_lt(r, 128n) == b_r : Bool}
) -> {Bool.and(Bool.and(b_blk, b_r), Bool.not(ikr_sampler_vocab_ok(Nat.add(Nat.mul(vp_blk, 128n), 1n+r), Nat.mul(1n+vp_blk, 128n)))) == False{} : Bool}:
  match b_blk b_r:
    case False{} _: {==}
    case True{} False{}: {==}
    case True{} True{}:
      +vo = Nat.add(Nat.mul(vp_blk, 128n), 1n+r)
      +vp = Nat.mul(1n+vp_blk, 128n)
      +hall = ikrp_and_true(
        Nat.is_gt(vo, 0n),
        Bool.and(Nat.is_le(vo, vp), Nat.is_eq(Nat.mod(vp, 128n), 0n)),
        ikrp_add_succ_gt0(Nat.mul(vp_blk, 128n), r),
        ikrp_and_true(Nat.is_le(vo, vp), Nat.is_eq(Nat.mod(vp, 128n), 0n), ikrp_svocab_le_canon(vp_blk, r, hr), ikrp_mod_mul_zero(1n+vp_blk, 127n))
      )
      ikrp_not_true(ikr_sampler_vocab_ok(vo, vp), hall)

def ikrp_mod_subst(+a: Nat, +b: Nat, +d: Nat, +target: Nat, eq_ab: {a == b : Nat}, hm: {Nat.mod(b, d) == target : Nat}) -> {Nat.mod(a, d) == target : Nat}:
  %Equal.sym(Nat, a, b, eq_ab) : {Nat.mod(_, d) == target : Nat}
  hm

def ikrp_div_subst(+a: Nat, +b: Nat, +d: Nat, +target: Nat, eq_ab: {a == b : Nat}, hq: {Nat.div(b, d) == target : Nat}) -> {Nat.div(a, d) == target : Nat}:
  %Equal.sym(Nat, a, b, eq_ab) : {Nat.div(_, d) == target : Nat}
  hq

def ikrp_div128_subst(+x: Nat, +m: Nat, +t: Nat, hx: {x == m : Nat}, ht: {Nat.div(m, 128n) == t : Nat}) -> {Nat.div(x, 128n) == t : Nat}:
  %Equal.sym(Nat, x, m, hx) : {Nat.div(_, 128n) == t : Nat}
  ht

def ikrp_w4_col_rebuild(+mc: Nat, +col: Nat, +hm: Nat, +half: Nat, emc: {mc == Nat.div(col, 2n) : Nat}, ehm: {hm == half : Nat}, eh: {Nat.mod(col, 2n) == half : Nat}) -> {Nat.add(Nat.mul(mc, 2n), hm) == col : Nat}:
  %Equal.sym(Nat, mc, Nat.div(col, 2n), emc) : {Nat.add(Nat.mul(_, 2n), hm) == col : Nat}
  %Equal.sym(Nat, hm, half, ehm) : {Nat.add(Nat.mul(Nat.div(col, 2n), 2n), _) == col : Nat}
  %eh : {Nat.add(Nat.mul(Nat.div(col, 2n), 2n), _) == col : Nat}
  L.nat_div_mod_r(col, 1n)

def ikrp_w4_off0_eq(+e: Nat, +tiles: Nat, +cp_cnt: Nat, +tt: Nat, +col: Nat, eh: {Nat.mod(col, 2n) == 0n : Nat}) -> {ikr_w4_col_off(e, tiles, cp_cnt, tt, col) == Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), cp_cnt), Nat.div(col, 2n)), 256n) : Nat}:
  +idx1 = Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), cp_cnt), Nat.div(col, 2n))
  %Equal.sym(Nat, Nat.mod(col, 2n), 0n, eh) : {Nat.add(Nat.mul(idx1, 256n), Nat.mul(128n, _)) == Nat.mul(idx1, 256n) : Nat}
  L.nat_add_zero_r(Nat.mul(idx1, 256n))

def ikrp_w4_off0_mod128(+idx1: Nat) -> {Nat.mod(Nat.mul(idx1, 256n), 128n) == 0n : Nat}:
  %L.nat_mul_assoc(idx1, 2n, 128n) : {Nat.mod(_, 128n) == 0n : Nat}
  L.nat_mod_mul(Nat.mul(idx1, 2n), 127n)

def ikrp_w4_off1_eq(+e: Nat, +tiles: Nat, +cp_cnt: Nat, +tt: Nat, +col: Nat, eh: {Nat.mod(col, 2n) == 1n : Nat}) -> {ikr_w4_col_off(e, tiles, cp_cnt, tt, col) == Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), cp_cnt), Nat.div(col, 2n)), 256n), 128n) : Nat}:
  +idx1 = Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), cp_cnt), Nat.div(col, 2n))
  %Equal.sym(Nat, Nat.mod(col, 2n), 1n, eh) : {Nat.add(Nat.mul(idx1, 256n), Nat.mul(128n, _)) == Nat.add(Nat.mul(idx1, 256n), 128n) : Nat}
  {==}

def ikrp_w4_off1_mul(+idx1: Nat) -> {Nat.add(Nat.mul(idx1, 256n), 128n) == Nat.mul(1n+Nat.mul(idx1, 2n), 128n) : Nat}:
  %L.nat_mul_assoc(idx1, 2n, 128n) : {Nat.add(_, 128n) == Nat.mul(1n+Nat.mul(idx1, 2n), 128n) : Nat}
  L.nat_add_comm(Nat.mul(Nat.mul(idx1, 2n), 128n), 128n)

def ikrp_w4_dec_tail(
  +e: Nat,
  +tp: Nat,
  +cpp: Nat,
  +tt: Nat,
  +col: Nat,
  +off: Nat,
  +half: Nat,
  +htt: {Nat.is_lt(tt, 1n+tp) == True{} : Bool},
  +hcp: {Nat.is_lt(Nat.div(col, 2n), 1n+cpp) == True{} : Bool},
  +h128: {Nat.mod(off, 128n) == 0n : Nat},
  +hq256: {Nat.div(off, 256n) == Nat.add(Nat.mul(Nat.add(Nat.mul(e, 1n+tp), tt), 1n+cpp), Nat.div(col, 2n)) : Nat},
  +hm256: {Nat.div(Nat.mod(off, 256n), 128n) == half : Nat},
  +eh: {Nat.mod(col, 2n) == half : Nat}
) -> {Bool.not(Bool.and(Nat.is_eq(Nat.mod(off, 128n), 0n), Bool.and(Nat.is_eq(Nat.div(Nat.div(Nat.div(off, 256n), 1n+cpp), 1n+tp), e), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.div(off, 256n), 1n+cpp), 1n+tp), tt), Nat.is_eq(Nat.add(Nat.mul(Nat.mod(Nat.div(off, 256n), 1n+cpp), 2n), Nat.div(Nat.mod(off, 256n), 128n)), col))))) == False{} : Bool}:
  +tiles = {1n+tp : Nat}
  +cp_cnt = {1n+cpp : Nat}
  +cp = Nat.div(col, 2n)
  +idx0 = Nat.add(Nat.mul(e, tiles), tt)
  +idx1 = Nat.add(Nat.mul(idx0, cp_cnt), cp)
  +q2 = Nat.div(off, 256n)
  +qc = Nat.div(q2, cp_cnt)
  +mc = Nat.mod(q2, cp_cnt)
  +eq_mc = ikrp_mod_subst(q2, idx1, cp_cnt, cp, hq256, L.nat_mod_split(idx0, cp, cpp, hcp))
  +eq_qc = ikrp_div_subst(q2, idx1, cp_cnt, idx0, hq256, L.nat_div_split(idx0, cp, cpp, hcp))
  +eq_tt = ikrp_mod_subst(qc, idx0, tiles, tt, eq_qc, L.nat_mod_split(e, tt, tp, htt))
  +eq_e = ikrp_div_subst(qc, idx0, tiles, e, eq_qc, L.nat_div_split(e, tt, tp, htt))
  +eq_col = ikrp_w4_col_rebuild(mc, col, Nat.div(Nat.mod(off, 256n), 128n), half, eq_mc, hm256, eh)
  +c1 = ikrp_is_eq(Nat.mod(off, 128n), 0n, h128)
  +c2 = ikrp_is_eq(Nat.div(qc, tiles), e, eq_e)
  +c3 = ikrp_is_eq(Nat.mod(qc, tiles), tt, eq_tt)
  +c4 = ikrp_is_eq(Nat.add(Nat.mul(mc, 2n), Nat.div(Nat.mod(off, 256n), 128n)), col, eq_col)
  ikrp_not_true(
    Bool.and(Nat.is_eq(Nat.mod(off, 128n), 0n), Bool.and(Nat.is_eq(Nat.div(qc, tiles), e), Bool.and(Nat.is_eq(Nat.mod(qc, tiles), tt), Nat.is_eq(Nat.add(Nat.mul(mc, 2n), Nat.div(Nat.mod(off, 256n), 128n)), col)))),
    ikrp_and_true(
      Nat.is_eq(Nat.mod(off, 128n), 0n),
      Bool.and(Nat.is_eq(Nat.div(qc, tiles), e), Bool.and(Nat.is_eq(Nat.mod(qc, tiles), tt), Nat.is_eq(Nat.add(Nat.mul(mc, 2n), Nat.div(Nat.mod(off, 256n), 128n)), col))),
      c1,
      ikrp_and_true(
        Nat.is_eq(Nat.div(qc, tiles), e),
        Bool.and(Nat.is_eq(Nat.mod(qc, tiles), tt), Nat.is_eq(Nat.add(Nat.mul(mc, 2n), Nat.div(Nat.mod(off, 256n), 128n)), col)),
        c2,
        ikrp_and_true(Nat.is_eq(Nat.mod(qc, tiles), tt), Nat.is_eq(Nat.add(Nat.mul(mc, 2n), Nat.div(Nat.mod(off, 256n), 128n)), col), c3, c4)
      )
    )
  )

def ikrp_w4_dec_half(
  +e: Nat,
  +tp: Nat,
  +cpp: Nat,
  +tt: Nat,
  +col: Nat,
  +half: Nat,
  +htt: {Nat.is_lt(tt, 1n+tp) == True{} : Bool},
  +hcp: {Nat.is_lt(Nat.div(col, 2n), 1n+cpp) == True{} : Bool},
  +hh: {Nat.is_lt(half, 2n) == True{} : Bool},
  +eh: {Nat.mod(col, 2n) == half : Nat}
) -> {Bool.not(Bool.and(Nat.is_eq(Nat.mod(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 128n), 0n), Bool.and(Nat.is_eq(Nat.div(Nat.div(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 1n+tp), e), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 1n+tp), tt), Nat.is_eq(Nat.add(Nat.mul(Nat.mod(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 2n), Nat.div(Nat.mod(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 128n)), col))))) == False{} : Bool}:
  match half:
    case 0n:
      +tiles = {1n+tp : Nat}
      +cp_cnt = {1n+cpp : Nat}
      +idx1 = Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), cp_cnt), Nat.div(col, 2n))
      +off = ikr_w4_col_off(e, tiles, cp_cnt, tt, col)
      +off0 = Nat.mul(idx1, 256n)
      +eq_off0 = ikrp_w4_off0_eq(e, tiles, cp_cnt, tt, col, eh)
      +h128 = ikrp_mod_subst(off, off0, 128n, 0n, eq_off0, ikrp_w4_off0_mod128(idx1))
      +hq256 = ikrp_div_subst(off, off0, 256n, idx1, eq_off0, L.nat_div_mul(idx1, 255n))
      +hm_raw = ikrp_mod_subst(off, off0, 256n, 0n, eq_off0, L.nat_mod_mul(idx1, 255n))
      +hm256 = ikrp_div128_subst(Nat.mod(off, 256n), 0n, 0n, hm_raw, {==})
      ikrp_w4_dec_tail(e, tp, cpp, tt, col, off, 0n, htt, hcp, h128, hq256, hm256, eh)
    case 1n+hp:
      match hp:
        case 0n:
          +tiles = {1n+tp : Nat}
          +cp_cnt = {1n+cpp : Nat}
          +idx1 = Nat.add(Nat.mul(Nat.add(Nat.mul(e, tiles), tt), cp_cnt), Nat.div(col, 2n))
          +off = ikr_w4_col_off(e, tiles, cp_cnt, tt, col)
          +off1 = Nat.add(Nat.mul(idx1, 256n), 128n)
          +eq_off1 = ikrp_w4_off1_eq(e, tiles, cp_cnt, tt, col, eh)
          +eq_off1_mul = Equal.trans(Nat, off, off1, Nat.mul(1n+Nat.mul(idx1, 2n), 128n), eq_off1, ikrp_w4_off1_mul(idx1))
          +h128 = ikrp_mod_subst(off, Nat.mul(1n+Nat.mul(idx1, 2n), 128n), 128n, 0n, eq_off1_mul, L.nat_mod_mul(1n+Nat.mul(idx1, 2n), 127n))
          +hq256 = ikrp_div_subst(off, off1, 256n, idx1, eq_off1, L.nat_div_split(idx1, 128n, 255n, {==}))
          +hm_raw = ikrp_mod_subst(off, off1, 256n, 128n, eq_off1, L.nat_mod_split(idx1, 128n, 255n, {==}))
          +hm256 = ikrp_div128_subst(Nat.mod(off, 256n), 128n, 1n, hm_raw, {==})
          ikrp_w4_dec_tail(e, tp, cpp, tt, col, off, 1n, htt, hcp, h128, hq256, hm256, eh)
        case 1n+hpp:
          +tiles = {1n+tp : Nat}
          +cp_cnt = {1n+cpp : Nat}
          +off = ikr_w4_col_off(e, tiles, cp_cnt, tt, col)
          G.g_nat_lt_zero_absurd(
            {Bool.not(Bool.and(Nat.is_eq(Nat.mod(off, 128n), 0n), Bool.and(Nat.is_eq(Nat.div(Nat.div(Nat.div(off, 256n), cp_cnt), tiles), e), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.div(off, 256n), cp_cnt), tiles), tt), Nat.is_eq(Nat.add(Nat.mul(Nat.mod(Nat.div(off, 256n), cp_cnt), 2n), Nat.div(Nat.mod(off, 256n), 128n)), col))))) == False{} : Bool},
            hpp,
            hh
          )

def ikrp_w4_dec_go(
  b1: Bool,
  b2: Bool,
  +e: Nat,
  +tp: Nat,
  +cpp: Nat,
  +tt: Nat,
  +col: Nat,
  +h1: {Nat.is_lt(tt, 1n+tp) == b1 : Bool},
  +h2: {Nat.is_lt(col, Nat.mul(1n+cpp, 2n)) == b2 : Bool}
) -> {Bool.and(Bool.and(b1, b2), Bool.not(Bool.and(Nat.is_eq(Nat.mod(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 128n), 0n), Bool.and(Nat.is_eq(Nat.div(Nat.div(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 1n+tp), e), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 1n+tp), tt), Nat.is_eq(Nat.add(Nat.mul(Nat.mod(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 2n), Nat.div(Nat.mod(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 128n)), col)))))) == False{} : Bool}:
  match b1 b2:
    case False{} _: {==}
    case True{} False{}: {==}
    case True{} True{}:
      ikrp_w4_dec_half(
        e, tp, cpp, tt, col, Nat.mod(col, 2n),
        h1, L.nat_div_lt_of_lt_mul(col, 1n+cpp, 1n, h2), L.nat_mod_lt(col, 1n), {==}
      )

def ikrp_idx3_dec_go(
  b1: Bool,
  b2: Bool,
  +e: Nat,
  +tp: Nat,
  +np: Nat,
  +tt: Nat,
  +x: Nat,
  +h1: {Nat.is_lt(tt, 1n+tp) == b1 : Bool},
  +h2: {Nat.is_lt(x, 1n+np) == b2 : Bool}
) -> {Bool.and(Bool.and(b1, b2), Bool.not(Bool.and(Nat.is_eq(Nat.mod(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, 1n+tp), tt), 1n+np), x), 256n), 256n), 0n), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, 1n+tp), tt), 1n+np), x), 256n), 256n), 1n+np), x), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.div(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, 1n+tp), tt), 1n+np), x), 256n), 256n), 1n+np), 1n+tp), tt), Nat.is_eq(Nat.div(Nat.div(Nat.div(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(e, 1n+tp), tt), 1n+np), x), 256n), 256n), 1n+np), 1n+tp), e)))))) == False{} : Bool}:
  match b1 b2:
    case False{} _: {==}
    case True{} False{}: {==}
    case True{} True{}:
      +tiles = {1n+tp : Nat}
      +n = {1n+np : Nat}
      +idx0 = Nat.add(Nat.mul(e, tiles), tt)
      +idx1 = Nat.add(Nat.mul(idx0, n), x)
      +off = Nat.mul(idx1, 256n)
      +q256 = Nat.div(off, 256n)
      +qn = Nat.div(q256, n)
      +h256 = L.nat_mod_mul(idx1, 255n)
      +hq = L.nat_div_mul(idx1, 255n)
      +mx = ikrp_mod_subst(q256, idx1, n, x, hq, L.nat_mod_split(idx0, x, np, h2))
      +q0 = ikrp_div_subst(q256, idx1, n, idx0, hq, L.nat_div_split(idx0, x, np, h2))
      +mtt = ikrp_mod_subst(qn, idx0, tiles, tt, q0, L.nat_mod_split(e, tt, tp, h1))
      +qe = ikrp_div_subst(qn, idx0, tiles, e, q0, L.nat_div_split(e, tt, tp, h1))
      +c1 = ikrp_is_eq(Nat.mod(off, 256n), 0n, h256)
      +c2 = ikrp_is_eq(Nat.mod(q256, n), x, mx)
      +c3 = ikrp_is_eq(Nat.mod(qn, tiles), tt, mtt)
      +c4 = ikrp_is_eq(Nat.div(qn, tiles), e, qe)
      ikrp_not_true(
        Bool.and(Nat.is_eq(Nat.mod(off, 256n), 0n), Bool.and(Nat.is_eq(Nat.mod(q256, n), x), Bool.and(Nat.is_eq(Nat.mod(qn, tiles), tt), Nat.is_eq(Nat.div(qn, tiles), e)))),
        ikrp_and_true(
          Nat.is_eq(Nat.mod(off, 256n), 0n),
          Bool.and(Nat.is_eq(Nat.mod(q256, n), x), Bool.and(Nat.is_eq(Nat.mod(qn, tiles), tt), Nat.is_eq(Nat.div(qn, tiles), e))),
          c1,
          ikrp_and_true(
            Nat.is_eq(Nat.mod(q256, n), x),
            Bool.and(Nat.is_eq(Nat.mod(qn, tiles), tt), Nat.is_eq(Nat.div(qn, tiles), e)),
            c2,
            ikrp_and_true(Nat.is_eq(Nat.mod(qn, tiles), tt), Nat.is_eq(Nat.div(qn, tiles), e), c3, c4)
          )
        )
      )

def ikrp_mul2_add_eq(+l: Nat, +rem: Nat) -> {Nat.add(Nat.mul(2n, l), rem) == Nat.add(Nat.mul(l, 2n), rem) : Nat}:
  %L.nat_mul_comm(l, 2n) : {Nat.add(_, rem) == Nat.add(Nat.mul(l, 2n), rem) : Nat}
  {==}

def ikrp_hlow_subst(+i: Nat, +j: Nat, hi: {Nat.is_lt(i, 256n) == True{} : Bool}, ej: {Nat.mod(i, 4n) == j : Nat}) -> {Nat.is_lt(Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(j, 2n)), 128n) == True{} : Bool}:
  %ej : {Nat.is_lt(Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(_, 2n)), 128n) == True{} : Bool}
  ikrp_2l_lt(i, hi)

def ikrp_lane_p_low(+i: Nat, +j: Nat, +low: Nat, ej: {Nat.mod(i, 4n) == j : Nat}, elow: {Nat.add(Bool.pick(Nat, Nat.is_lt(j, 2n), 0n, 128n), Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(j, 2n))) == low : Nat}) -> {ikr_scale_lane_pos(i) == low : Nat}:
  +l = Nat.div(i, 4n)
  %Equal.sym(Nat, Nat.mod(i, 4n), j, ej) : {Nat.add(Bool.pick(Nat, Nat.is_lt(_, 2n), 0n, 128n), Nat.add(Nat.mul(2n, l), Nat.mod(_, 2n))) == low : Nat}
  elow

def ikrp_add128_mul1(+x: Nat) -> {Nat.add(128n, x) == Nat.add(Nat.mul(1n, 128n), x) : Nat}:
  %Equal.sym(Nat, Nat.add(128n, 0n), 128n, L.nat_add_zero_r(128n)) : {Nat.add(128n, x) == Nat.add(_, x) : Nat}
  {==}

def ikrp_lane_inv_tail(
  +i: Nat,
  +p: Nat,
  +q128: Nat,
  +m128: Nat,
  +j: Nat,
  +ep128: {Nat.div(p, 128n) == q128 : Nat},
  +em128: {Nat.mod(p, 128n) == m128 : Nat},
  +hlm: {Nat.is_lt(Nat.mod(j, 2n), 2n) == True{} : Bool},
  +em_eq: {m128 == Nat.add(Nat.mul(Nat.div(i, 4n), 2n), Nat.mod(j, 2n)) : Nat},
  +ej_val: {Nat.add(Nat.mul(2n, q128), Nat.mod(j, 2n)) == j : Nat},
  +ej: {Nat.mod(i, 4n) == j : Nat}
) -> {Nat.add(Nat.mul(4n, Nat.div(Nat.mod(p, 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(p, 128n)), Nat.mod(Nat.mod(p, 128n), 2n))) == i : Nat}:
  +l = Nat.div(i, 4n)
  +m = Nat.mod(j, 2n)
  +eq_mp = Equal.trans(Nat, Nat.mod(p, 128n), m128, Nat.add(Nat.mul(l, 2n), m), em128, em_eq)
  +hq2 = ikrp_div_subst(Nat.mod(p, 128n), Nat.add(Nat.mul(l, 2n), m), 2n, l, eq_mp, L.nat_div_split(l, m, 1n, hlm))
  +hm2 = ikrp_mod_subst(Nat.mod(p, 128n), Nat.add(Nat.mul(l, 2n), m), 2n, m, eq_mp, L.nat_mod_split(l, m, 1n, hlm))
  %Equal.sym(Nat, Nat.div(Nat.mod(p, 128n), 2n), l, hq2) : {Nat.add(Nat.mul(4n, _), Nat.add(Nat.mul(2n, Nat.div(p, 128n)), Nat.mod(Nat.mod(p, 128n), 2n))) == i : Nat}
  %Equal.sym(Nat, Nat.div(p, 128n), q128, ep128) : {Nat.add(Nat.mul(4n, l), Nat.add(Nat.mul(2n, _), Nat.mod(Nat.mod(p, 128n), 2n))) == i : Nat}
  %Equal.sym(Nat, Nat.mod(Nat.mod(p, 128n), 2n), m, hm2) : {Nat.add(Nat.mul(4n, l), Nat.add(Nat.mul(2n, q128), _)) == i : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(2n, q128), m), j, ej_val) : {Nat.add(Nat.mul(4n, l), _) == i : Nat}
  %L.nat_mul_comm(l, 4n) : {Nat.add(_, j) == i : Nat}
  %ej : {Nat.add(Nat.mul(l, 4n), _) == i : Nat}
  L.nat_div_mod_r(i, 3n)

def ikrp_lane_lt_256(
  +i: Nat,
  +hi: {Nat.is_lt(i, 256n) == True{} : Bool},
  c: Bool,
  +hc: {Nat.is_lt(Nat.mod(i, 4n), 2n) == c : Bool}
) -> {Nat.is_lt(ikr_scale_lane_pos(i), 256n) == True{} : Bool}:
  match c:
    case True{}:
      +low = Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n))
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(i, 4n), 2n), True{}, hc) : {Nat.is_lt(Nat.add(Bool.pick(Nat, _, 0n, 128n), low), 256n) == True{} : Bool}
      L.nat_lt_trans(low, 128n, 256n, ikrp_2l_lt(i, hi), {==})
    case False{}:
      +low = Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n))
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(i, 4n), 2n), False{}, hc) : {Nat.is_lt(Nat.add(Bool.pick(Nat, _, 0n, 128n), low), 256n) == True{} : Bool}
      L.nat_add_lt_l(128n, low, 128n, ikrp_2l_lt(i, hi))

def ikrp_lane_half_eq(
  +i: Nat,
  +hi: {Nat.is_lt(i, 256n) == True{} : Bool},
  c: Bool,
  +hc: {Nat.is_lt(Nat.mod(i, 4n), 2n) == c : Bool}
) -> {Bool.not(Bool.xor(Nat.is_lt(Nat.mod(i, 4n), 2n), Nat.is_lt(ikr_scale_lane_pos(i), 128n))) == True{} : Bool}:
  match c:
    case True{}:
      +low = Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n))
      +hlow = ikrp_2l_lt(i, hi)
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(i, 4n), 2n), True{}, hc) : {Bool.not(Bool.xor(_, Nat.is_lt(Nat.add(Bool.pick(Nat, _, 0n, 128n), low), 128n))) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(low, 128n), True{}, hlow) : {Bool.not(Bool.xor(True{}, _)) == True{} : Bool}
      {==}
    case False{}:
      +low = Nat.add(Nat.mul(2n, Nat.div(i, 4n)), Nat.mod(Nat.mod(i, 4n), 2n))
      +hge = L.nat_le_lt_false(Nat.add(128n, low), 128n, L.nat_add_le_r(128n, low))
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(i, 4n), 2n), False{}, hc) : {Bool.not(Bool.xor(_, Nat.is_lt(Nat.add(Bool.pick(Nat, _, 0n, 128n), low), 128n))) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(Nat.add(128n, low), 128n), False{}, hge) : {Bool.not(Bool.xor(False{}, _)) == True{} : Bool}
      {==}

def ikrp_lane_half2_go(
  +i: Nat,
  v: Bool,
  +hv: {Nat.is_lt(i, 256n) == v : Bool}
) -> {Bool.and(v, Bool.not(Bool.and(Nat.is_lt(ikr_scale_lane_pos(i), 256n), Bool.not(Bool.xor(Nat.is_lt(Nat.mod(i, 4n), 2n), Nat.is_lt(ikr_scale_lane_pos(i), 128n)))))) == False{} : Bool}:
  match v:
    case False{}: {==}
    case True{}:
      ikrp_not_true(
        Bool.and(Nat.is_lt(ikr_scale_lane_pos(i), 256n), Bool.not(Bool.xor(Nat.is_lt(Nat.mod(i, 4n), 2n), Nat.is_lt(ikr_scale_lane_pos(i), 128n)))),
        ikrp_and_true(
          Nat.is_lt(ikr_scale_lane_pos(i), 256n),
          Bool.not(Bool.xor(Nat.is_lt(Nat.mod(i, 4n), 2n), Nat.is_lt(ikr_scale_lane_pos(i), 128n))),
          ikrp_lane_lt_256(i, hv, Nat.is_lt(Nat.mod(i, 4n), 2n), {==}),
          ikrp_lane_half_eq(i, hv, Nat.is_lt(Nat.mod(i, 4n), 2n), {==})
        )
      )

def ikrp_lane_inv_j(
  +i: Nat,
  +j: Nat,
  +hi: {Nat.is_lt(i, 256n) == True{} : Bool},
  +hj: {Nat.is_lt(j, 4n) == True{} : Bool},
  +ej: {Nat.mod(i, 4n) == j : Nat}
) -> {Nat.is_eq(Nat.add(Nat.mul(4n, Nat.div(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(ikr_scale_lane_pos(i), 128n)), Nat.mod(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n))), i) == True{} : Bool}:
  match j:
    case 0n:
      +l = Nat.div(i, 4n)
      +p = ikr_scale_lane_pos(i)
      +p0 = Nat.add(Nat.mul(2n, l), 0n)
      +ep = ikrp_lane_p_low(i, 0n, p0, ej, {==})
      +hlow0 = ikrp_hlow_subst(i, 0n, hi, ej)
      +ep128 = ikrp_div_subst(p, p0, 128n, 0n, ep, L.nat_div_small(p0, 127n, hlow0))
      +em128 = ikrp_mod_subst(p, p0, 128n, p0, ep, L.nat_mod_small(p0, 127n, hlow0))
      +em_eq = ikrp_mul2_add_eq(l, 0n)
      ikrp_is_eq(
        Nat.add(Nat.mul(4n, Nat.div(Nat.mod(p, 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(p, 128n)), Nat.mod(Nat.mod(p, 128n), 2n))),
        i,
        ikrp_lane_inv_tail(i, p, 0n, p0, 0n, ep128, em128, {==}, em_eq, {==}, ej)
      )
    case 1n+j1:
      match j1:
        case 0n:
          +l = Nat.div(i, 4n)
          +p = ikr_scale_lane_pos(i)
          +p1 = Nat.add(Nat.mul(2n, l), 1n)
          +ep = ikrp_lane_p_low(i, 1n, p1, ej, {==})
          +hlow1 = ikrp_hlow_subst(i, 1n, hi, ej)
          +ep128 = ikrp_div_subst(p, p1, 128n, 0n, ep, L.nat_div_small(p1, 127n, hlow1))
          +em128 = ikrp_mod_subst(p, p1, 128n, p1, ep, L.nat_mod_small(p1, 127n, hlow1))
          +em_eq = ikrp_mul2_add_eq(l, 1n)
          ikrp_is_eq(
            Nat.add(Nat.mul(4n, Nat.div(Nat.mod(p, 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(p, 128n)), Nat.mod(Nat.mod(p, 128n), 2n))),
            i,
            ikrp_lane_inv_tail(i, p, 0n, p1, 1n, ep128, em128, {==}, em_eq, {==}, ej)
          )
        case 1n+j2:
          match j2:
            case 0n:
              +l = Nat.div(i, 4n)
              +p = ikr_scale_lane_pos(i)
              +p2l = Nat.add(Nat.mul(2n, l), 0n)
              +hlow2 = ikrp_hlow_subst(i, 2n, hi, ej)
              +ep = Equal.trans(Nat, p, Nat.add(128n, p2l), Nat.add(Nat.mul(1n, 128n), p2l), ikrp_lane_p_low(i, 2n, Nat.add(128n, p2l), ej, {==}), ikrp_add128_mul1(p2l))
              +ep128 = ikrp_div_subst(p, Nat.add(Nat.mul(1n, 128n), p2l), 128n, 1n, ep, L.nat_div_split(1n, p2l, 127n, hlow2))
              +em128 = ikrp_mod_subst(p, Nat.add(Nat.mul(1n, 128n), p2l), 128n, p2l, ep, L.nat_mod_split(1n, p2l, 127n, hlow2))
              +em_eq = ikrp_mul2_add_eq(l, 0n)
              ikrp_is_eq(
                Nat.add(Nat.mul(4n, Nat.div(Nat.mod(p, 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(p, 128n)), Nat.mod(Nat.mod(p, 128n), 2n))),
                i,
                ikrp_lane_inv_tail(i, p, 1n, p2l, 2n, ep128, em128, {==}, em_eq, {==}, ej)
              )
            case 1n+j3:
              match j3:
                case 0n:
                  +l = Nat.div(i, 4n)
                  +p = ikr_scale_lane_pos(i)
                  +p3l = Nat.add(Nat.mul(2n, l), 1n)
                  +hlow3 = ikrp_hlow_subst(i, 3n, hi, ej)
                  +ep = Equal.trans(Nat, p, Nat.add(128n, p3l), Nat.add(Nat.mul(1n, 128n), p3l), ikrp_lane_p_low(i, 3n, Nat.add(128n, p3l), ej, {==}), ikrp_add128_mul1(p3l))
                  +ep128 = ikrp_div_subst(p, Nat.add(Nat.mul(1n, 128n), p3l), 128n, 1n, ep, L.nat_div_split(1n, p3l, 127n, hlow3))
                  +em128 = ikrp_mod_subst(p, Nat.add(Nat.mul(1n, 128n), p3l), 128n, p3l, ep, L.nat_mod_split(1n, p3l, 127n, hlow3))
                  +em_eq = ikrp_mul2_add_eq(l, 1n)
                  ikrp_is_eq(
                    Nat.add(Nat.mul(4n, Nat.div(Nat.mod(p, 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(p, 128n)), Nat.mod(Nat.mod(p, 128n), 2n))),
                    i,
                    ikrp_lane_inv_tail(i, p, 1n, p3l, 3n, ep128, em128, {==}, em_eq, {==}, ej)
                  )
                case 1n+j4:
                  +p = ikr_scale_lane_pos(i)
                  G.g_nat_lt_zero_absurd(
                    {Nat.is_eq(Nat.add(Nat.mul(4n, Nat.div(Nat.mod(p, 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(p, 128n)), Nat.mod(Nat.mod(p, 128n), 2n))), i) == True{} : Bool},
                    j4,
                    hj
                  )

def ikrp_lane_inv_go(
  +i: Nat,
  v: Bool,
  +hv: {Nat.is_lt(i, 256n) == v : Bool}
) -> {Bool.and(v, Bool.not(Bool.and(Nat.is_lt(ikr_scale_lane_pos(i), 256n), Nat.is_eq(Nat.add(Nat.mul(4n, Nat.div(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(ikr_scale_lane_pos(i), 128n)), Nat.mod(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n))), i)))) == False{} : Bool}:
  match v:
    case False{}: {==}
    case True{}:
      ikrp_not_true(
        Bool.and(Nat.is_lt(ikr_scale_lane_pos(i), 256n), Nat.is_eq(Nat.add(Nat.mul(4n, Nat.div(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(ikr_scale_lane_pos(i), 128n)), Nat.mod(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n))), i)),
        ikrp_and_true(
          Nat.is_lt(ikr_scale_lane_pos(i), 256n),
          Nat.is_eq(Nat.add(Nat.mul(4n, Nat.div(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(ikr_scale_lane_pos(i), 128n)), Nat.mod(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n))), i),
          ikrp_lane_lt_256(i, hv, Nat.is_lt(Nat.mod(i, 4n), 2n), {==}),
          ikrp_lane_inv_j(i, Nat.mod(i, 4n), hv, L.nat_mod_lt(i, 3n), {==})
        )
      )

def ikrp_src_lt128(+l: Nat, +m: Nat, hl: {Nat.is_lt(l, 64n) == True{} : Bool}, hm: {Nat.is_lt(m, 2n) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(2n, l), m), 128n) == True{} : Bool}:
  %L.nat_mul_comm(l, 2n) : {Nat.is_lt(Nat.add(_, m), 128n) == True{} : Bool}
  L.nat_split_lt(l, m, 64n, 1n, hl, hm)

def ikrp_src_gu_base(+tt: Nat, +x: Nat, +d: Nat) -> {Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), x)), d) == Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, x), d)) : Nat}:
  %Equal.sym(Nat, Nat.mul(2n, Nat.add(Nat.mul(128n, tt), x)), Nat.add(Nat.mul(2n, Nat.mul(128n, tt)), Nat.mul(2n, x)), L.nat_mul_add_dist_l(2n, Nat.mul(128n, tt), x)) : {Nat.add(_, d) == Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, x), d)) : Nat}
  %L.nat_mul_assoc(2n, 128n, tt) : {Nat.add(Nat.add(_, Nat.mul(2n, x)), d) == Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, x), d)) : Nat}
  %L.nat_mul_comm(tt, 256n) : {Nat.add(Nat.add(_, Nat.mul(2n, x)), d) == Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, x), d)) : Nat}
  L.nat_add_assoc(Nat.mul(tt, 256n), Nat.mul(2n, x), d)

def ikrp_src_gu_hi128(+tt: Nat, +l: Nat, +d: Nat) -> {Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), d)) == Nat.add(Nat.mul(tt, 256n), Nat.add(128n, Nat.add(Nat.mul(2n, l), d))) : Nat}:
  %Equal.sym(Nat, Nat.add(64n, 0n), 64n, L.nat_add_zero_r(64n)) : {Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(_, l)), d)) == Nat.add(Nat.mul(tt, 256n), Nat.add(128n, Nat.add(Nat.mul(2n, l), d))) : Nat}
  %Equal.sym(Nat, Nat.mul(2n, Nat.add(64n, l)), Nat.add(128n, Nat.mul(2n, l)), L.nat_mul_add_dist_l(2n, 64n, l)) : {Nat.add(Nat.mul(tt, 256n), Nat.add(_, d)) == Nat.add(Nat.mul(tt, 256n), Nat.add(128n, Nat.add(Nat.mul(2n, l), d))) : Nat}
  %L.nat_add_assoc(128n, Nat.mul(2n, l), d) : {Nat.add(Nat.mul(tt, 256n), _) == Nat.add(Nat.mul(tt, 256n), Nat.add(128n, Nat.add(Nat.mul(2n, l), d))) : Nat}
  {==}

def ikrp_src_gu_raw(+tt: Nat, +lane: Nat, +j: Nat, ej: {Nat.mod(lane, 4n) == j : Nat}) -> {ikr_src_row(True{}, tt, lane) == Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), Nat.add(Nat.mul(Nat.mod(j, 2n), 64n), Nat.div(lane, 4n)))), Nat.div(j, 2n)) : Nat}:
  +l = Nat.div(lane, 4n)
  %Equal.sym(Nat, Nat.mod(lane, 4n), j, ej) : {Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), Nat.add(Nat.mul(Nat.mod(_, 2n), 64n), l))), Nat.div(_, 2n)) == Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), Nat.add(Nat.mul(Nat.mod(j, 2n), 64n), l))), Nat.div(j, 2n)) : Nat}
  {==}

def ikrp_src_dn_raw(+tt: Nat, +lane: Nat, +j: Nat, ej: {Nat.mod(lane, 4n) == j : Nat}) -> {ikr_src_row(False{}, tt, lane) == Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(Nat.div(j, 2n), 128n), Nat.add(Nat.mul(2n, Nat.div(lane, 4n)), Nat.mod(j, 2n)))) : Nat}:
  +l = Nat.div(lane, 4n)
  %Equal.sym(Nat, Nat.mod(lane, 4n), j, ej) : {Nat.add(Nat.mul(256n, tt), Nat.add(Nat.mul(Nat.div(_, 2n), 128n), Nat.add(Nat.mul(2n, l), Nat.mod(_, 2n)))) == Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(Nat.div(j, 2n), 128n), Nat.add(Nat.mul(2n, l), Nat.mod(j, 2n)))) : Nat}
  %L.nat_mul_comm(tt, 256n) : {Nat.add(_, Nat.add(Nat.mul(Nat.div(j, 2n), 128n), Nat.add(Nat.mul(2n, l), Nat.mod(j, 2n)))) == Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(Nat.div(j, 2n), 128n), Nat.add(Nat.mul(2n, l), Nat.mod(j, 2n)))) : Nat}
  {==}

def ikrp_src_dn_hi128(+tt: Nat, +l: Nat, +d: Nat) -> {Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(1n, 128n), Nat.add(Nat.mul(2n, l), d))) == Nat.add(Nat.mul(tt, 256n), Nat.add(128n, Nat.add(Nat.mul(2n, l), d))) : Nat}:
  %Equal.sym(Nat, Nat.add(128n, 0n), 128n, L.nat_add_zero_r(128n)) : {Nat.add(Nat.mul(tt, 256n), Nat.add(_, Nat.add(Nat.mul(2n, l), d))) == Nat.add(Nat.mul(tt, 256n), Nat.add(128n, Nat.add(Nat.mul(2n, l), d))) : Nat}
  {==}

def ikrp_src_lb(+r: Nat, +tt: Nat, +off: Nat, eq_r: {r == Nat.add(Nat.mul(tt, 256n), off) : Nat}) -> {Nat.is_le(Nat.mul(256n, tt), r) == True{} : Bool}:
  %Equal.sym(Nat, r, Nat.add(Nat.mul(tt, 256n), off), eq_r) : {Nat.is_le(Nat.mul(256n, tt), _) == True{} : Bool}
  %L.nat_mul_comm(tt, 256n) : {Nat.is_le(_, Nat.add(Nat.mul(tt, 256n), off)) == True{} : Bool}
  L.nat_add_le_r(Nat.mul(tt, 256n), off)

def ikrp_src_ub(+r: Nat, +tt: Nat, +off: Nat, eq_r: {r == Nat.add(Nat.mul(tt, 256n), off) : Nat}, hoff: {Nat.is_lt(off, 256n) == True{} : Bool}) -> {Nat.is_lt(r, Nat.mul(256n, 1n+tt)) == True{} : Bool}:
  %Equal.sym(Nat, r, Nat.add(Nat.mul(tt, 256n), off), eq_r) : {Nat.is_lt(_, Nat.mul(256n, 1n+tt)) == True{} : Bool}
  %L.nat_mul_comm(1n+tt, 256n) : {Nat.is_lt(Nat.add(Nat.mul(tt, 256n), off), _) == True{} : Bool}
  L.nat_split_lt(tt, off, 1n+tt, 255n, L.nat_lt_succ_self(tt), hoff)

def ikrp_src_tile_tail(
  +is_gu: Bool,
  +tt: Nat,
  +lane: Nat,
  +off: Nat,
  +eq_r: {ikr_src_row(is_gu, tt, lane) == Nat.add(Nat.mul(tt, 256n), off) : Nat},
  +hoff: {Nat.is_lt(off, 256n) == True{} : Bool}
) -> {Bool.not(Bool.and(Nat.is_le(Nat.mul(256n, tt), ikr_src_row(is_gu, tt, lane)), Nat.is_lt(ikr_src_row(is_gu, tt, lane), Nat.mul(256n, 1n+tt)))) == False{} : Bool}:
  +r = ikr_src_row(is_gu, tt, lane)
  ikrp_not_true(
    Bool.and(Nat.is_le(Nat.mul(256n, tt), r), Nat.is_lt(r, Nat.mul(256n, 1n+tt))),
    ikrp_and_true(Nat.is_le(Nat.mul(256n, tt), r), Nat.is_lt(r, Nat.mul(256n, 1n+tt)), ikrp_src_lb(r, tt, off, eq_r), ikrp_src_ub(r, tt, off, eq_r, hoff))
  )

def ikrp_src_dec_combine(
  +is_gu: Bool,
  +l: Nat,
  +lo128: Nat,
  +hi128: Nat,
  +q128: Nat,
  +m: Nat,
  +j: Nat,
  eq_l: {Nat.div(lo128, 2n) == l : Nat},
  eq_m: {Nat.mod(lo128, 2n) == m : Nat},
  eq_hi: {hi128 == q128 : Nat},
  ej_val: {Bool.pick(Nat, is_gu, Nat.add(Nat.mul(2n, m), q128), Nat.add(Nat.mul(2n, q128), m)) == j : Nat}
) -> {Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, Nat.mod(lo128, 2n)), hi128)), Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, hi128), Nat.mod(lo128, 2n)))) == Nat.add(Nat.mul(4n, l), j) : Nat}:
  match is_gu:
    case True{}:
      %Equal.sym(Nat, Nat.div(lo128, 2n), l, eq_l) : {Nat.add(Nat.mul(4n, _), Nat.add(Nat.mul(2n, Nat.mod(lo128, 2n)), hi128)) == Nat.add(Nat.mul(4n, l), j) : Nat}
      %Equal.sym(Nat, Nat.mod(lo128, 2n), m, eq_m) : {Nat.add(Nat.mul(4n, l), Nat.add(Nat.mul(2n, _), hi128)) == Nat.add(Nat.mul(4n, l), j) : Nat}
      %Equal.sym(Nat, hi128, q128, eq_hi) : {Nat.add(Nat.mul(4n, l), Nat.add(Nat.mul(2n, m), _)) == Nat.add(Nat.mul(4n, l), j) : Nat}
      %Equal.sym(Nat, Nat.add(Nat.mul(2n, m), q128), j, ej_val) : {Nat.add(Nat.mul(4n, l), _) == Nat.add(Nat.mul(4n, l), j) : Nat}
      {==}
    case False{}:
      %Equal.sym(Nat, Nat.div(lo128, 2n), l, eq_l) : {Nat.add(Nat.mul(4n, _), Nat.add(Nat.mul(2n, hi128), Nat.mod(lo128, 2n))) == Nat.add(Nat.mul(4n, l), j) : Nat}
      %Equal.sym(Nat, hi128, q128, eq_hi) : {Nat.add(Nat.mul(4n, l), Nat.add(Nat.mul(2n, _), Nat.mod(lo128, 2n))) == Nat.add(Nat.mul(4n, l), j) : Nat}
      %Equal.sym(Nat, Nat.mod(lo128, 2n), m, eq_m) : {Nat.add(Nat.mul(4n, l), Nat.add(Nat.mul(2n, q128), _)) == Nat.add(Nat.mul(4n, l), j) : Nat}
      %Equal.sym(Nat, Nat.add(Nat.mul(2n, q128), m), j, ej_val) : {Nat.add(Nat.mul(4n, l), _) == Nat.add(Nat.mul(4n, l), j) : Nat}
      {==}

def ikrp_src_finish_lane(+l: Nat, +j: Nat, +lane: Nat, ej: {Nat.mod(lane, 4n) == j : Nat}, eq_l: {l == Nat.div(lane, 4n) : Nat}) -> {Nat.add(Nat.mul(4n, l), j) == lane : Nat}:
  %Equal.sym(Nat, l, Nat.div(lane, 4n), eq_l) : {Nat.add(Nat.mul(4n, _), j) == lane : Nat}
  %L.nat_mul_comm(Nat.div(lane, 4n), 4n) : {Nat.add(_, j) == lane : Nat}
  %ej : {Nat.add(Nat.mul(Nat.div(lane, 4n), 4n), _) == lane : Nat}
  L.nat_div_mod_r(lane, 3n)

def ikrp_src_dec_tail(
  +is_gu: Bool,
  +tt: Nat,
  +lane: Nat,
  +off: Nat,
  +q128: Nat,
  +m128: Nat,
  +m: Nat,
  +j: Nat,
  +eq_r: {ikr_src_row(is_gu, tt, lane) == Nat.add(Nat.mul(tt, 256n), off) : Nat},
  +hoff: {Nat.is_lt(off, 256n) == True{} : Bool},
  +eq128: {Nat.div(off, 128n) == q128 : Nat},
  +em128: {Nat.mod(off, 128n) == m128 : Nat},
  +hlm: {Nat.is_lt(m, 2n) == True{} : Bool},
  +em_eq: {m128 == Nat.add(Nat.mul(Nat.div(lane, 4n), 2n), m) : Nat},
  +ej_val: {Bool.pick(Nat, is_gu, Nat.add(Nat.mul(2n, m), q128), Nat.add(Nat.mul(2n, q128), m)) == j : Nat},
  +ej: {Nat.mod(lane, 4n) == j : Nat}
) -> {Bool.not(Bool.and(Nat.is_eq(Nat.div(ikr_src_row(is_gu, tt, lane), 256n), tt), Nat.is_eq(Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.mod(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.div(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n))), Nat.add(Nat.mul(4n, Nat.div(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n)), Nat.mod(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)))), lane))) == False{} : Bool}:
  +r = ikr_src_row(is_gu, tt, lane)
  +rem = Nat.mod(r, 256n)
  +lo128 = Nat.mod(rem, 128n)
  +hi128 = Nat.div(rem, 128n)
  +l = Nat.div(lane, 4n)
  +eq_tile = ikrp_div_subst(r, Nat.add(Nat.mul(tt, 256n), off), 256n, tt, eq_r, L.nat_div_split(tt, off, 255n, hoff))
  +eq_rem = ikrp_mod_subst(r, Nat.add(Nat.mul(tt, 256n), off), 256n, off, eq_r, L.nat_mod_split(tt, off, 255n, hoff))
  +eq_hi = ikrp_div_subst(rem, off, 128n, q128, eq_rem, eq128)
  +eq_lo = Equal.trans(Nat, lo128, m128, Nat.add(Nat.mul(l, 2n), m), ikrp_mod_subst(rem, off, 128n, m128, eq_rem, em128), em_eq)
  +eq_l = ikrp_div_subst(lo128, Nat.add(Nat.mul(l, 2n), m), 2n, l, eq_lo, L.nat_div_split(l, m, 1n, hlm))
  +eq_m = ikrp_mod_subst(lo128, Nat.add(Nat.mul(l, 2n), m), 2n, m, eq_lo, L.nat_mod_split(l, m, 1n, hlm))
  +eq_dec = Equal.trans(
    Nat,
    Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, Nat.mod(lo128, 2n)), hi128)), Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, hi128), Nat.mod(lo128, 2n)))),
    Nat.add(Nat.mul(4n, l), j),
    lane,
    ikrp_src_dec_combine(is_gu, l, lo128, hi128, q128, m, j, eq_l, eq_m, eq_hi, ej_val),
    ikrp_src_finish_lane(l, j, lane, ej, {==})
  )
  +c_tile = ikrp_is_eq(Nat.div(r, 256n), tt, eq_tile)
  +c_dec = ikrp_is_eq(
    Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, Nat.mod(lo128, 2n)), hi128)), Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, hi128), Nat.mod(lo128, 2n)))),
    lane,
    eq_dec
  )
  ikrp_not_true(
    Bool.and(Nat.is_eq(Nat.div(r, 256n), tt), Nat.is_eq(Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, Nat.mod(lo128, 2n)), hi128)), Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, hi128), Nat.mod(lo128, 2n)))), lane)),
    ikrp_and_true(
      Nat.is_eq(Nat.div(r, 256n), tt),
      Nat.is_eq(Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, Nat.mod(lo128, 2n)), hi128)), Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, hi128), Nat.mod(lo128, 2n)))), lane),
      c_tile,
      c_dec
    )
  )

def ikrp_src_tile_j(
  +is_gu: Bool,
  +tt: Nat,
  +lane: Nat,
  +j: Nat,
  +hlane: {Nat.is_lt(lane, 256n) == True{} : Bool},
  +hj: {Nat.is_lt(j, 4n) == True{} : Bool},
  +ej: {Nat.mod(lane, 4n) == j : Nat}
) -> {Bool.not(Bool.and(Nat.is_le(Nat.mul(256n, tt), ikr_src_row(is_gu, tt, lane)), Nat.is_lt(ikr_src_row(is_gu, tt, lane), Nat.mul(256n, 1n+tt)))) == False{} : Bool}:
  match is_gu j:
    case True{} 0n:
      +l = Nat.div(lane, 4n)
      +off = Nat.add(Nat.mul(2n, l), 0n)
      +eq_r = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), l)), 0n), Nat.add(Nat.mul(tt, 256n), off), ikrp_src_gu_raw(tt, lane, 0n, ej), ikrp_src_gu_base(tt, l, 0n))
      +hoff = L.nat_lt_trans(off, 128n, 256n, ikrp_src_lt128(l, 0n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==}), {==})
      ikrp_src_tile_tail(True{}, tt, lane, off, eq_r, hoff)
    case True{} 1n+j1:
      match j1:
        case 0n:
          +l = Nat.div(lane, 4n)
          +off = Nat.add(128n, Nat.add(Nat.mul(2n, l), 0n))
          +eq_r1 = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), Nat.add(Nat.mul(1n, 64n), l))), 0n), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), 0n)), ikrp_src_gu_raw(tt, lane, 1n, ej), ikrp_src_gu_base(tt, Nat.add(Nat.mul(1n, 64n), l), 0n))
          +eq_r = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), 0n)), Nat.add(Nat.mul(tt, 256n), off), eq_r1, ikrp_src_gu_hi128(tt, l, 0n))
          +hoff = L.nat_add_lt_l(128n, Nat.add(Nat.mul(2n, l), 0n), 128n, ikrp_src_lt128(l, 0n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==}))
          ikrp_src_tile_tail(True{}, tt, lane, off, eq_r, hoff)
        case 1n+j2:
          match j2:
            case 0n:
              +l = Nat.div(lane, 4n)
              +off = Nat.add(Nat.mul(2n, l), 1n)
              +eq_r = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), l)), 1n), Nat.add(Nat.mul(tt, 256n), off), ikrp_src_gu_raw(tt, lane, 2n, ej), ikrp_src_gu_base(tt, l, 1n))
              +hoff = L.nat_lt_trans(off, 128n, 256n, ikrp_src_lt128(l, 1n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==}), {==})
              ikrp_src_tile_tail(True{}, tt, lane, off, eq_r, hoff)
            case 1n+j3:
              match j3:
                case 0n:
                  +l = Nat.div(lane, 4n)
                  +off = Nat.add(128n, Nat.add(Nat.mul(2n, l), 1n))
                  +eq_r1 = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), Nat.add(Nat.mul(1n, 64n), l))), 1n), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), 1n)), ikrp_src_gu_raw(tt, lane, 3n, ej), ikrp_src_gu_base(tt, Nat.add(Nat.mul(1n, 64n), l), 1n))
                  +eq_r = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), 1n)), Nat.add(Nat.mul(tt, 256n), off), eq_r1, ikrp_src_gu_hi128(tt, l, 1n))
                  +hoff = L.nat_add_lt_l(128n, Nat.add(Nat.mul(2n, l), 1n), 128n, ikrp_src_lt128(l, 1n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==}))
                  ikrp_src_tile_tail(True{}, tt, lane, off, eq_r, hoff)
                case 1n+j4:
                  G.g_nat_lt_zero_absurd({Bool.not(Bool.and(Nat.is_le(Nat.mul(256n, tt), ikr_src_row(True{}, tt, lane)), Nat.is_lt(ikr_src_row(True{}, tt, lane), Nat.mul(256n, 1n+tt)))) == False{} : Bool}, j4, hj)
    case False{} 0n:
      +l = Nat.div(lane, 4n)
      +off = Nat.add(Nat.mul(2n, l), 0n)
      +eq_r = ikrp_src_dn_raw(tt, lane, 0n, ej)
      +hoff = L.nat_lt_trans(off, 128n, 256n, ikrp_src_lt128(l, 0n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==}), {==})
      ikrp_src_tile_tail(False{}, tt, lane, off, eq_r, hoff)
    case False{} 1n+j1:
      match j1:
        case 0n:
          +l = Nat.div(lane, 4n)
          +off = Nat.add(Nat.mul(2n, l), 1n)
          +eq_r = ikrp_src_dn_raw(tt, lane, 1n, ej)
          +hoff = L.nat_lt_trans(off, 128n, 256n, ikrp_src_lt128(l, 1n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==}), {==})
          ikrp_src_tile_tail(False{}, tt, lane, off, eq_r, hoff)
        case 1n+j2:
          match j2:
            case 0n:
              +l = Nat.div(lane, 4n)
              +off = Nat.add(128n, Nat.add(Nat.mul(2n, l), 0n))
              +eq_r = Equal.trans(Nat, ikr_src_row(False{}, tt, lane), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(1n, 128n), Nat.add(Nat.mul(2n, l), 0n))), Nat.add(Nat.mul(tt, 256n), off), ikrp_src_dn_raw(tt, lane, 2n, ej), ikrp_src_dn_hi128(tt, l, 0n))
              +hoff = L.nat_add_lt_l(128n, Nat.add(Nat.mul(2n, l), 0n), 128n, ikrp_src_lt128(l, 0n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==}))
              ikrp_src_tile_tail(False{}, tt, lane, off, eq_r, hoff)
            case 1n+j3:
              match j3:
                case 0n:
                  +l = Nat.div(lane, 4n)
                  +off = Nat.add(128n, Nat.add(Nat.mul(2n, l), 1n))
                  +eq_r = Equal.trans(Nat, ikr_src_row(False{}, tt, lane), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(1n, 128n), Nat.add(Nat.mul(2n, l), 1n))), Nat.add(Nat.mul(tt, 256n), off), ikrp_src_dn_raw(tt, lane, 3n, ej), ikrp_src_dn_hi128(tt, l, 1n))
                  +hoff = L.nat_add_lt_l(128n, Nat.add(Nat.mul(2n, l), 1n), 128n, ikrp_src_lt128(l, 1n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==}))
                  ikrp_src_tile_tail(False{}, tt, lane, off, eq_r, hoff)
                case 1n+j4:
                  G.g_nat_lt_zero_absurd({Bool.not(Bool.and(Nat.is_le(Nat.mul(256n, tt), ikr_src_row(False{}, tt, lane)), Nat.is_lt(ikr_src_row(False{}, tt, lane), Nat.mul(256n, 1n+tt)))) == False{} : Bool}, j4, hj)

def ikrp_src_tile_go(+is_gu: Bool, +tt: Nat, +lane: Nat, v: Bool, +hv: {Nat.is_lt(lane, 256n) == v : Bool}) -> {Bool.and(v, Bool.not(Bool.and(Nat.is_le(Nat.mul(256n, tt), ikr_src_row(is_gu, tt, lane)), Nat.is_lt(ikr_src_row(is_gu, tt, lane), Nat.mul(256n, 1n+tt))))) == False{} : Bool}:
  match v:
    case False{}: {==}
    case True{}:
      ikrp_src_tile_j(is_gu, tt, lane, Nat.mod(lane, 4n), hv, L.nat_mod_lt(lane, 3n), {==})

def ikrp_src_dec_j(
  +is_gu: Bool,
  +tt: Nat,
  +lane: Nat,
  +j: Nat,
  +hlane: {Nat.is_lt(lane, 256n) == True{} : Bool},
  +hj: {Nat.is_lt(j, 4n) == True{} : Bool},
  +ej: {Nat.mod(lane, 4n) == j : Nat}
) -> {Bool.not(Bool.and(Nat.is_eq(Nat.div(ikr_src_row(is_gu, tt, lane), 256n), tt), Nat.is_eq(Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.mod(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.div(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n))), Nat.add(Nat.mul(4n, Nat.div(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n)), Nat.mod(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)))), lane))) == False{} : Bool}:
  match is_gu j:
    case True{} 0n:
      +l = Nat.div(lane, 4n)
      +p0 = Nat.add(Nat.mul(2n, l), 0n)
      +hlo = ikrp_src_lt128(l, 0n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==})
      +eq_r = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), l)), 0n), Nat.add(Nat.mul(tt, 256n), p0), ikrp_src_gu_raw(tt, lane, 0n, ej), ikrp_src_gu_base(tt, l, 0n))
      ikrp_src_dec_tail(True{}, tt, lane, p0, 0n, p0, 0n, 0n, eq_r, L.nat_lt_trans(p0, 128n, 256n, hlo, {==}), L.nat_div_small(p0, 127n, hlo), L.nat_mod_small(p0, 127n, hlo), {==}, ikrp_mul2_add_eq(l, 0n), {==}, ej)
    case True{} 1n+j1:
      match j1:
        case 0n:
          +l = Nat.div(lane, 4n)
          +p0 = Nat.add(Nat.mul(2n, l), 0n)
          +off = Nat.add(128n, p0)
          +hlo = ikrp_src_lt128(l, 0n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==})
          +eq_r1 = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), Nat.add(Nat.mul(1n, 64n), l))), 0n), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), 0n)), ikrp_src_gu_raw(tt, lane, 1n, ej), ikrp_src_gu_base(tt, Nat.add(Nat.mul(1n, 64n), l), 0n))
          +eq_r = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), 0n)), Nat.add(Nat.mul(tt, 256n), off), eq_r1, ikrp_src_gu_hi128(tt, l, 0n))
          +eq128 = ikrp_div_subst(off, Nat.add(Nat.mul(1n, 128n), p0), 128n, 1n, ikrp_add128_mul1(p0), L.nat_div_split(1n, p0, 127n, hlo))
          +em128 = ikrp_mod_subst(off, Nat.add(Nat.mul(1n, 128n), p0), 128n, p0, ikrp_add128_mul1(p0), L.nat_mod_split(1n, p0, 127n, hlo))
          ikrp_src_dec_tail(True{}, tt, lane, off, 1n, p0, 0n, 1n, eq_r, L.nat_add_lt_l(128n, p0, 128n, hlo), eq128, em128, {==}, ikrp_mul2_add_eq(l, 0n), {==}, ej)
        case 1n+j2:
          match j2:
            case 0n:
              +l = Nat.div(lane, 4n)
              +p1 = Nat.add(Nat.mul(2n, l), 1n)
              +hlo = ikrp_src_lt128(l, 1n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==})
              +eq_r = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), l)), 1n), Nat.add(Nat.mul(tt, 256n), p1), ikrp_src_gu_raw(tt, lane, 2n, ej), ikrp_src_gu_base(tt, l, 1n))
              ikrp_src_dec_tail(True{}, tt, lane, p1, 0n, p1, 1n, 2n, eq_r, L.nat_lt_trans(p1, 128n, 256n, hlo, {==}), L.nat_div_small(p1, 127n, hlo), L.nat_mod_small(p1, 127n, hlo), {==}, ikrp_mul2_add_eq(l, 1n), {==}, ej)
            case 1n+j3:
              match j3:
                case 0n:
                  +l = Nat.div(lane, 4n)
                  +p1 = Nat.add(Nat.mul(2n, l), 1n)
                  +off = Nat.add(128n, p1)
                  +hlo = ikrp_src_lt128(l, 1n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==})
                  +eq_r1 = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(128n, tt), Nat.add(Nat.mul(1n, 64n), l))), 1n), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), 1n)), ikrp_src_gu_raw(tt, lane, 3n, ej), ikrp_src_gu_base(tt, Nat.add(Nat.mul(1n, 64n), l), 1n))
                  +eq_r = Equal.trans(Nat, ikr_src_row(True{}, tt, lane), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(2n, Nat.add(Nat.mul(1n, 64n), l)), 1n)), Nat.add(Nat.mul(tt, 256n), off), eq_r1, ikrp_src_gu_hi128(tt, l, 1n))
                  +eq128 = ikrp_div_subst(off, Nat.add(Nat.mul(1n, 128n), p1), 128n, 1n, ikrp_add128_mul1(p1), L.nat_div_split(1n, p1, 127n, hlo))
                  +em128 = ikrp_mod_subst(off, Nat.add(Nat.mul(1n, 128n), p1), 128n, p1, ikrp_add128_mul1(p1), L.nat_mod_split(1n, p1, 127n, hlo))
                  ikrp_src_dec_tail(True{}, tt, lane, off, 1n, p1, 1n, 3n, eq_r, L.nat_add_lt_l(128n, p1, 128n, hlo), eq128, em128, {==}, ikrp_mul2_add_eq(l, 1n), {==}, ej)
                case 1n+j4:
                  +r = ikr_src_row(True{}, tt, lane)
                  +rem = Nat.mod(r, 256n)
                  +lo128 = Nat.mod(rem, 128n)
                  +hi128 = Nat.div(rem, 128n)
                  G.g_nat_lt_zero_absurd({Bool.not(Bool.and(Nat.is_eq(Nat.div(r, 256n), tt), Nat.is_eq(Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, Nat.mod(lo128, 2n)), hi128)), lane))) == False{} : Bool}, j4, hj)
    case False{} 0n:
      +l = Nat.div(lane, 4n)
      +p0 = Nat.add(Nat.mul(2n, l), 0n)
      +hlo = ikrp_src_lt128(l, 0n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==})
      +eq_r = ikrp_src_dn_raw(tt, lane, 0n, ej)
      ikrp_src_dec_tail(False{}, tt, lane, p0, 0n, p0, 0n, 0n, eq_r, L.nat_lt_trans(p0, 128n, 256n, hlo, {==}), L.nat_div_small(p0, 127n, hlo), L.nat_mod_small(p0, 127n, hlo), {==}, ikrp_mul2_add_eq(l, 0n), {==}, ej)
    case False{} 1n+j1:
      match j1:
        case 0n:
          +l = Nat.div(lane, 4n)
          +p1 = Nat.add(Nat.mul(2n, l), 1n)
          +hlo = ikrp_src_lt128(l, 1n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==})
          +eq_r = ikrp_src_dn_raw(tt, lane, 1n, ej)
          ikrp_src_dec_tail(False{}, tt, lane, p1, 0n, p1, 1n, 1n, eq_r, L.nat_lt_trans(p1, 128n, 256n, hlo, {==}), L.nat_div_small(p1, 127n, hlo), L.nat_mod_small(p1, 127n, hlo), {==}, ikrp_mul2_add_eq(l, 1n), {==}, ej)
        case 1n+j2:
          match j2:
            case 0n:
              +l = Nat.div(lane, 4n)
              +p0 = Nat.add(Nat.mul(2n, l), 0n)
              +off = Nat.add(128n, p0)
              +hlo = ikrp_src_lt128(l, 0n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==})
              +eq_r = Equal.trans(Nat, ikr_src_row(False{}, tt, lane), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(1n, 128n), p0)), Nat.add(Nat.mul(tt, 256n), off), ikrp_src_dn_raw(tt, lane, 2n, ej), ikrp_src_dn_hi128(tt, l, 0n))
              +eq128 = ikrp_div_subst(off, Nat.add(Nat.mul(1n, 128n), p0), 128n, 1n, ikrp_add128_mul1(p0), L.nat_div_split(1n, p0, 127n, hlo))
              +em128 = ikrp_mod_subst(off, Nat.add(Nat.mul(1n, 128n), p0), 128n, p0, ikrp_add128_mul1(p0), L.nat_mod_split(1n, p0, 127n, hlo))
              ikrp_src_dec_tail(False{}, tt, lane, off, 1n, p0, 0n, 2n, eq_r, L.nat_add_lt_l(128n, p0, 128n, hlo), eq128, em128, {==}, ikrp_mul2_add_eq(l, 0n), {==}, ej)
            case 1n+j3:
              match j3:
                case 0n:
                  +l = Nat.div(lane, 4n)
                  +p1 = Nat.add(Nat.mul(2n, l), 1n)
                  +off = Nat.add(128n, p1)
                  +hlo = ikrp_src_lt128(l, 1n, L.nat_div_lt_of_lt_mul(lane, 64n, 3n, hlane), {==})
                  +eq_r = Equal.trans(Nat, ikr_src_row(False{}, tt, lane), Nat.add(Nat.mul(tt, 256n), Nat.add(Nat.mul(1n, 128n), p1)), Nat.add(Nat.mul(tt, 256n), off), ikrp_src_dn_raw(tt, lane, 3n, ej), ikrp_src_dn_hi128(tt, l, 1n))
                  +eq128 = ikrp_div_subst(off, Nat.add(Nat.mul(1n, 128n), p1), 128n, 1n, ikrp_add128_mul1(p1), L.nat_div_split(1n, p1, 127n, hlo))
                  +em128 = ikrp_mod_subst(off, Nat.add(Nat.mul(1n, 128n), p1), 128n, p1, ikrp_add128_mul1(p1), L.nat_mod_split(1n, p1, 127n, hlo))
                  ikrp_src_dec_tail(False{}, tt, lane, off, 1n, p1, 1n, 3n, eq_r, L.nat_add_lt_l(128n, p1, 128n, hlo), eq128, em128, {==}, ikrp_mul2_add_eq(l, 1n), {==}, ej)
                case 1n+j4:
                  +r = ikr_src_row(False{}, tt, lane)
                  +rem = Nat.mod(r, 256n)
                  +lo128 = Nat.mod(rem, 128n)
                  +hi128 = Nat.div(rem, 128n)
                  G.g_nat_lt_zero_absurd({Bool.not(Bool.and(Nat.is_eq(Nat.div(r, 256n), tt), Nat.is_eq(Nat.add(Nat.mul(4n, Nat.div(lo128, 2n)), Nat.add(Nat.mul(2n, hi128), Nat.mod(lo128, 2n))), lane))) == False{} : Bool}, j4, hj)

def ikrp_src_dec_go(
  +is_gu: Bool,
  +tt: Nat,
  +lane: Nat,
  v: Bool,
  +hv: {Nat.is_lt(lane, 256n) == v : Bool}
) -> {Bool.and(v, Bool.not(Bool.and(Nat.is_eq(Nat.div(ikr_src_row(is_gu, tt, lane), 256n), tt), Nat.is_eq(Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.mod(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.div(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n))), Nat.add(Nat.mul(4n, Nat.div(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n)), Nat.mod(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)))), lane)))) == False{} : Bool}:
  match v:
    case False{}: {==}
    case True{}:
      ikrp_src_dec_j(is_gu, tt, lane, Nat.mod(lane, 4n), hv, L.nat_mod_lt(lane, 3n), {==})

# ============================================================================
# 6. Laws
# ============================================================================

# --- 6.1 Mode selection laws ---

# promise: asking for dense MoE execution never enables the routed expert path.
# harm: `--moe-form dense` would silently run quantized routed experts instead of the BF16 reference.
# catches: IkMoeDense arm returning dims_ok in ikr_use_routed
law ikr_dense_never_routed:
  for is_inkling2: Bool
  for dims_ok: Bool
  {ikr_use_routed(is_inkling2, IkMoeDense{}, dims_ok) == False{} : Bool}

def ikr_dense_never_routed(is_inkling2, dims_ok):
  {==}

# promise: routed MoE execution is never enabled when the model geometry fails the routed guard.
# harm: routed TPC kernels would read misaligned tiles or exceed router/expert table bounds.
# catches: IkMoeRouted arm returning True without checking dims_ok
law ikr_routed_requires_dims:
  for is_inkling2: Bool
  for form: IkMoeForm
  {ikr_use_routed(is_inkling2, form, False{}) == False{} : Bool}

def ikr_routed_requires_dims(is_inkling2, form):
  match is_inkling2:
    case False{}:
      match form:
        case IkMoeAuto{}: {==}
        case IkMoeDense{}: {==}
        case IkMoeRouted{}: {==}
    case True{}:
      match form:
        case IkMoeAuto{}: {==}
        case IkMoeDense{}: {==}
        case IkMoeRouted{}: {==}

# promise: explicit routed MoE mode enables the routed expert path whenever geometry satisfies the guard.
# harm: `--moe-form routed` would silently stay on dense BF16 even for valid Inkling 2 geometry.
# catches: IkMoeRouted arm returning False in ikr_use_routed
law ikr_routed_when_asked:
  for is_inkling2: Bool
  {ikr_use_routed(is_inkling2, IkMoeRouted{}, True{}) == True{} : Bool}

def ikr_routed_when_asked(is_inkling2):
  {==}

# promise: automatic MoE selection never enables Inkling 2 routed execution for a non-Inkling 2 model.
# harm: `--moe-form auto` on another MoE model family would try to dispatch Inkling 2 routed kernels.
# catches: IkMoeAuto arm ignoring is_inkling2 and returning dims_ok in ikr_use_routed
law ikr_auto_non_inkling2_dense:
  for dims_ok: Bool
  {ikr_use_routed(False{}, IkMoeAuto{}, dims_ok) == False{} : Bool}

def ikr_auto_non_inkling2_dense(dims_ok):
  {==}

# promise: explicit routed mode is refused whenever geometry fails the routed guard.
# harm: invalid model dimensions under `--moe-form routed` would either panic later or silently fall back to dense BF16.
# catches: IkMoeRouted arm returning IkDecDense{} when dims_ok is False{}
law ikr_decide_moe_refuses_explicit_invalid:
  for is_inkling2: Bool
  {ikr_decide_moe(is_inkling2, IkMoeRouted{}, False{}) == IkDecRefused{} : IkMoeDecision}

def ikr_decide_moe_refuses_explicit_invalid(is_inkling2):
  {==}

# promise: explicit dense mode always selects dense execution and is never refused or routed.
# harm: `--moe-form dense` could refuse an unsupported shape instead of running the BF16 reference.
# catches: IkMoeDense arm returning IkDecRefused{} in ikr_decide_moe
law ikr_decide_moe_dense_always_dense:
  for is_inkling2: Bool
  for dims_ok: Bool
  {ikr_decide_moe(is_inkling2, IkMoeDense{}, dims_ok) == IkDecDense{} : IkMoeDecision}

def ikr_decide_moe_dense_always_dense(is_inkling2, dims_ok):
  {==}

# promise: automatic MoE selection falls back to dense execution instead of refusing when geometry fails the routed guard.
# harm: `--moe-form auto` would fail to load a non-routable Inkling-family configuration instead of falling back to dense BF16.
# catches: IkMoeAuto arm returning IkDecRefused{} when dims_ok is False{}
law ikr_decide_moe_auto_invalid_falls_back:
  for is_inkling2: Bool
  {ikr_decide_moe(is_inkling2, IkMoeAuto{}, False{}) == IkDecDense{} : IkMoeDecision}

def ikr_decide_moe_auto_invalid_falls_back(is_inkling2):
  match is_inkling2:
    case False{}: {==}
    case True{}: {==}

# --- 6.2 Routed geometry & tile count laws ---

# promise: every hidden size accepted by the routed guard is non-zero and divisible by 256 (whole down tiles).
# harm: routed down-projection kernels would read past the hidden width or skip tail channels.
# catches: hid modulo check weakened from 256n to 128n
law ikr_dims_hidden_aligned:
  for +h: Nat
  for +ef: Nat
  for +ne: Nat
  for +k: Nat
  for +max_ne: Nat
  for +max_k: Nat
  {Bool.and(ikr_dims_valid(h, ef, ne, k, max_ne, max_k), Bool.not(Bool.and(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n)))) == False{} : Bool}

def ikr_dims_hidden_aligned(h, ef, ne, k, max_ne, max_k):
  ikrp_dims_c1(
    Bool.and(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n)),
    Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n)),
    Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))),
    Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)))
  )

# promise: every expert intermediate width accepted by the routed guard is non-zero and divisible by 128 (whole GU tiles).
# harm: routed gate+up kernels would read/write partial 128-channel tiles.
# catches: ef modulo check dropped in ikr_dims_valid
law ikr_dims_ef_aligned:
  for +h: Nat
  for +ef: Nat
  for +ne: Nat
  for +k: Nat
  for +max_ne: Nat
  for +max_k: Nat
  {Bool.and(ikr_dims_valid(h, ef, ne, k, max_ne, max_k), Bool.not(Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n)))) == False{} : Bool}

def ikr_dims_ef_aligned(h, ef, ne, k, max_ne, max_k):
  ikrp_dims_c2(
    Bool.and(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n)),
    Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n)),
    Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))),
    Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)))
  )

# promise: every expert count accepted by the routed guard is non-zero, a multiple of 128, and within `max_ne`.
# harm: `gaudium_ik_w4_plan` and `gaudium_ik_route_r` would overflow SLM tables on an out-of-range model.
# catches: ne <= max_ne upper bound dropped in ikr_dims_valid
law ikr_dims_ne_valid:
  for +h: Nat
  for +ef: Nat
  for +ne: Nat
  for +k: Nat
  for +max_ne: Nat
  for +max_k: Nat
  {Bool.and(ikr_dims_valid(h, ef, ne, k, max_ne, max_k), Bool.not(Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))))) == False{} : Bool}

def ikr_dims_ne_valid(h, ef, ne, k, max_ne, max_k):
  ikrp_dims_c3(
    Bool.and(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n)),
    Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n)),
    Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))),
    Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)))
  )

# promise: every router selection count accepted by the routed guard lies in `1..min(max_k, n_experts)`.
# harm: `gaudium_ik_route_r` or `gaudium_ik_w4_plan` would access past the kernel register cap or expert count.
# catches: top_k <= max_k bound dropped in ikr_dims_valid
law ikr_dims_topk_valid:
  for +h: Nat
  for +ef: Nat
  for +ne: Nat
  for +k: Nat
  for +max_ne: Nat
  for +max_k: Nat
  {Bool.and(ikr_dims_valid(h, ef, ne, k, max_ne, max_k), Bool.not(Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne))))) == False{} : Bool}

def ikr_dims_topk_valid(h, ef, ne, k, max_ne, max_k):
  ikrp_dims_c4(
    Bool.and(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n)),
    Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n)),
    Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne))),
    Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)))
  )

# promise: every positive aligned geometry with `h <= 16384`, `ef <= 8192` (the law's range) and `ne <= max_ne <= 512`, `1 <= k <= max_k <= 12` (the router and plan kernel caps) is accepted by `ikr_dims_valid`.
# harm: a valid Inkling 2 shape would be falsely rejected from routed execution.
# catches: ikr_dims_valid returning constant False{}
law ikr_dims_valid_accepts_multiples:
  for +hp: Nat
  for +efp: Nat
  for +nep: Nat
  for +kp: Nat
  for +extra_ne: Nat
  for +extra_k: Nat
  {Bool.and(Bool.and(Bool.and(Nat.is_lt(hp, 64n), Bool.and(Nat.is_lt(efp, 64n), Bool.and(Nat.is_le(Nat.add(Nat.mul(1n+nep, 128n), extra_ne), 512n), Nat.is_le(Nat.add(1n+kp, extra_k), 12n)))), Nat.is_le(1n+kp, Nat.mul(1n+nep, 128n))), Bool.not(ikr_dims_valid(Nat.mul(1n+hp, 256n), Nat.mul(1n+efp, 128n), Nat.mul(1n+nep, 128n), 1n+kp, Nat.add(Nat.mul(1n+nep, 128n), extra_ne), Nat.add(1n+kp, extra_k)))) == False{} : Bool}

def ikr_dims_valid_accepts_multiples(hp, efp, nep, kp, extra_ne, extra_k):
  ikrp_dims_acc_abi(
    hp, efp, nep, kp, extra_ne, extra_k,
    Bool.and(Nat.is_lt(hp, 64n), Bool.and(Nat.is_lt(efp, 64n), Bool.and(Nat.is_le(Nat.add(Nat.mul(1n+nep, 128n), extra_ne), 512n), Nat.is_le(Nat.add(1n+kp, extra_k), 12n)))),
    Nat.is_le(1n+kp, Nat.mul(1n+nep, 128n)),
    {==}
  )

def ikrp_gu_tiles_go(
  +h: Nat,
  +ef: Nat,
  +ne: Nat,
  +k: Nat,
  +max_ne: Nat,
  +max_k: Nat,
  v: Bool,
  +hv: {ikr_dims_valid(h, ef, ne, k, max_ne, max_k) == v : Bool}
) -> {Bool.and(v, Nat.is_ne(Nat.mul(ikr_gu_tiles(ef), 128n), ef)) == False{} : Bool}:
  match v:
    case False{}: {==}
    case True{}:
      +ha = Bool.and(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n))
      +hb = Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n))
      +hc = Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne)))
      +hd = Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)))
      +hr = ikrp_and_r(ha, Bool.and(hb, Bool.and(hc, hd)), hv)
      +hbe = ikrp_and_l(hb, Bool.and(hc, hd), hr)
      +hm = ikrp_and_r(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n), hbe)
      ikrp_div_eq_succ(ef, 127n, hm)

def ikrp_dn_tiles_go(
  +h: Nat,
  +ef: Nat,
  +ne: Nat,
  +k: Nat,
  +max_ne: Nat,
  +max_k: Nat,
  v: Bool,
  +hv: {ikr_dims_valid(h, ef, ne, k, max_ne, max_k) == v : Bool}
) -> {Bool.and(v, Nat.is_ne(Nat.mul(ikr_dn_tiles(h), 256n), h)) == False{} : Bool}:
  match v:
    case False{}: {==}
    case True{}:
      +ha = Bool.and(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n))
      +hb = Bool.and(Nat.is_gt(ef, 0n), Nat.is_eq(Nat.mod(ef, 128n), 0n))
      +hc = Bool.and(Nat.is_gt(ne, 0n), Bool.and(Nat.is_eq(Nat.mod(ne, 128n), 0n), Nat.is_le(ne, max_ne)))
      +hd = Bool.and(Nat.is_ge(k, 1n), Bool.and(Nat.is_le(k, max_k), Nat.is_le(k, ne)))
      +hae = ikrp_and_l(ha, Bool.and(hb, Bool.and(hc, hd)), hv)
      +hm = ikrp_and_r(Nat.is_gt(h, 0n), Nat.is_eq(Nat.mod(h, 256n), 0n), hae)
      ikrp_div_eq_succ(h, 255n, hm)

# promise: gate+up tile count times 128 covers the exact expert intermediate width when accepted.
# harm: the packer or HBM planner would allocate too few tiles and drop expert intermediate channels.
# catches: ikr_gu_tiles dividing by 256n instead of 128n
law ikr_gu_tiles_exact:
  for +h: Nat
  for +ef: Nat
  for +ne: Nat
  for +k: Nat
  for +max_ne: Nat
  for +max_k: Nat
  {Bool.and(ikr_dims_valid(h, ef, ne, k, max_ne, max_k), Nat.is_ne(Nat.mul(ikr_gu_tiles(ef), 128n), ef)) == False{} : Bool}

def ikr_gu_tiles_exact(h, ef, ne, k, max_ne, max_k):
  ikrp_gu_tiles_go(h, ef, ne, k, max_ne, max_k, ikr_dims_valid(h, ef, ne, k, max_ne, max_k), {==})

# promise: down-projection tile count times 256 covers the exact hidden width when accepted.
# harm: the packer or HBM planner would allocate too few tiles and corrupt the residual dimension.
# catches: ikr_dn_tiles dividing by 128n instead of 256n
law ikr_dn_tiles_exact:
  for +h: Nat
  for +ef: Nat
  for +ne: Nat
  for +k: Nat
  for +max_ne: Nat
  for +max_k: Nat
  {Bool.and(ikr_dims_valid(h, ef, ne, k, max_ne, max_k), Nat.is_ne(Nat.mul(ikr_dn_tiles(h), 256n), h)) == False{} : Bool}

def ikr_dn_tiles_exact(h, ef, ne, k, max_ne, max_k):
  ikrp_dn_tiles_go(h, ef, ne, k, max_ne, max_k, ikr_dims_valid(h, ef, ne, k, max_ne, max_k), {==})

# --- 6.3 Packed pool byte sizes & offset laws ---

# promise: a W4 code pool byte count holds at least 256 bytes per column pair per tile per expert.
# harm: the W4 pool allocation would be shorter than the packed tensor data uploaded into it.
# catches: factor 256n omitted or halved in ikr_w4_code_bytes
law ikr_w4_code_bytes_holds:
  for +cp: Nat
  for +tiles: Nat
  for +ne: Nat
  {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(ne, tiles), cp), 256n), ikr_w4_code_bytes(Nat.mul(cp, 2n), tiles, ne)) == True{} : Bool}

def ikr_w4_code_bytes_holds(cp, tiles, ne):
  %Equal.sym(Nat, Nat.div(Nat.mul(cp, 2n), 2n), cp, L.nat_div_mul(cp, 1n)) : {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(ne, tiles), cp), 256n), Nat.mul(Nat.mul(Nat.mul(ne, tiles), _), 256n)) == True{} : Bool}
  L.nat_le_refl(Nat.mul(Nat.mul(Nat.mul(ne, tiles), cp), 256n))

# promise: a W4 code pool byte count never exceeds 256 bytes per column pair per tile per expert.
# harm: oversized W4 pool sizing would waste HBM or fail the size check against the step graph tensor.
# catches: in_cols not divided by 2n in ikr_w4_code_bytes
law ikr_w4_code_bytes_within:
  for +cp: Nat
  for +tiles: Nat
  for +ne: Nat
  {Nat.is_le(ikr_w4_code_bytes(Nat.mul(cp, 2n), tiles, ne), Nat.mul(Nat.mul(Nat.mul(ne, tiles), cp), 256n)) == True{} : Bool}

def ikr_w4_code_bytes_within(cp, tiles, ne):
  %Equal.sym(Nat, Nat.div(Nat.mul(cp, 2n), 2n), cp, L.nat_div_mul(cp, 1n)) : {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(ne, tiles), _), 256n), Nat.mul(Nat.mul(Nat.mul(ne, tiles), cp), 256n)) == True{} : Bool}
  L.nat_le_refl(Nat.mul(Nat.mul(Nat.mul(ne, tiles), cp), 256n))

# promise: a W8 code pool byte count holds at least 256 bytes per column per tile per expert.
# harm: the W8 pool allocation would truncate expert codes.
# catches: factor 256n omitted or halved in ikr_w8_code_bytes
law ikr_w8_code_bytes_holds:
  for +in_cols: Nat
  for +tiles: Nat
  for +ne: Nat
  {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(ne, tiles), in_cols), 256n), ikr_w8_code_bytes(in_cols, tiles, ne)) == True{} : Bool}

def ikr_w8_code_bytes_holds(in_cols, tiles, ne):
  L.nat_le_refl(ikr_w8_code_bytes(in_cols, tiles, ne))

# promise: a W8 code pool byte count never exceeds 256 bytes per column per tile per expert.
# harm: oversized W8 pool sizing would waste HBM or reject a valid pack buffer.
# catches: W8 pool doubled instead of 1 byte per int8 code in ikr_w8_code_bytes
law ikr_w8_code_bytes_within:
  for +in_cols: Nat
  for +tiles: Nat
  for +ne: Nat
  {Nat.is_le(ikr_w8_code_bytes(in_cols, tiles, ne), Nat.mul(Nat.mul(Nat.mul(ne, tiles), in_cols), 256n)) == True{} : Bool}

def ikr_w8_code_bytes_within(in_cols, tiles, ne):
  L.nat_le_refl(ikr_w8_code_bytes(in_cols, tiles, ne))

# promise: an expert fp16 scale pool holds at least two bytes per lane of every 256-lane scale vector (512 bytes per vector) across every block, tile, and expert.
# harm: packed scales would overwrite adjacent buffers or truncate high-tile scales.
# catches: 2n byte-per-fp16 multiplier omitted in ikr_scale_bytes
law ikr_scale_bytes_holds:
  for +nb: Nat
  for +bp: Nat
  for +tiles: Nat
  for +ne: Nat
  {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), nb), 256n), 2n), ikr_scale_bytes(Nat.mul(nb, 1n+bp), 1n+bp, tiles, ne)) == True{} : Bool}

def ikr_scale_bytes_holds(nb, bp, tiles, ne):
  %Equal.sym(Nat, Nat.div(Nat.mul(nb, 1n+bp), 1n+bp), nb, L.nat_div_mul(nb, bp)) : {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), nb), 256n), 2n), Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), _), 256n), 2n)) == True{} : Bool}
  L.nat_le_refl(Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), nb), 256n), 2n))

# promise: an expert fp16 scale pool never exceeds two bytes per lane of every 256-lane scale vector (512 bytes per vector) across every block, tile, and expert.
# harm: oversized scale pool allocation would waste HBM or disagree with the step graph's f16 scale tensor.
# catches: block count not divided by blk in ikr_scale_bytes
law ikr_scale_bytes_within:
  for +nb: Nat
  for +bp: Nat
  for +tiles: Nat
  for +ne: Nat
  {Nat.is_le(ikr_scale_bytes(Nat.mul(nb, 1n+bp), 1n+bp, tiles, ne), Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), nb), 256n), 2n)) == True{} : Bool}

def ikr_scale_bytes_within(nb, bp, tiles, ne):
  %Equal.sym(Nat, Nat.div(Nat.mul(nb, 1n+bp), 1n+bp), nb, L.nat_div_mul(nb, bp)) : {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), _), 256n), 2n), Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), nb), 256n), 2n)) == True{} : Bool}
  L.nat_le_refl(Nat.mul(Nat.mul(Nat.mul(Nat.mul(ne, tiles), nb), 256n), 2n))

# promise: every 128-byte column slice written by the W4 packer ends inside the W4 code pool.
# harm: packing a nibble column would write out of bounds into another expert or past the code buffer.
# catches: col not divided by 2n in the base index of ikr_w4_col_off
law ikr_w4_col_off_in_pool:
  for +e: Nat
  for +ne: Nat
  for +tt: Nat
  for +tiles: Nat
  for +c: Nat
  for +cp_cnt: Nat
  {Bool.and(Bool.and(Nat.is_lt(e, ne), Bool.and(Nat.is_lt(tt, tiles), Nat.is_lt(c, Nat.mul(cp_cnt, 2n)))), Nat.is_lt(ikr_w4_code_bytes(Nat.mul(cp_cnt, 2n), tiles, ne), Nat.add(ikr_w4_col_off(e, tiles, cp_cnt, tt, c), 128n))) == False{} : Bool}

def ikr_w4_col_off_in_pool(e, ne, tt, tiles, c, cp_cnt):
  ikrp_w4_col_go(
    Nat.is_lt(e, ne), Nat.is_lt(tt, tiles), Nat.is_lt(c, Nat.mul(cp_cnt, 2n)),
    e, ne, tt, tiles, c, cp_cnt,
    {==}, {==}, {==}
  )

# promise: every 256-byte column vector written by the W8 packer ends inside the W8 code pool.
# harm: packing an 8-bit column would write out of bounds past the W8 code buffer.
# catches: e multiplied by in_cols instead of tiles in ikr_w8_col_off
law ikr_w8_col_off_in_pool:
  for +e: Nat
  for +ne: Nat
  for +tt: Nat
  for +tiles: Nat
  for +c: Nat
  for +in_cols: Nat
  {Bool.and(Bool.and(Nat.is_lt(e, ne), Bool.and(Nat.is_lt(tt, tiles), Nat.is_lt(c, in_cols))), Nat.is_lt(ikr_w8_code_bytes(in_cols, tiles, ne), Nat.add(ikr_w8_col_off(e, tiles, in_cols, tt, c), 256n))) == False{} : Bool}

def ikr_w8_col_off_in_pool(e, ne, tt, tiles, c, in_cols):
  ikrp_w8_col_go(
    Nat.is_lt(e, ne), Nat.is_lt(tt, tiles), Nat.is_lt(c, in_cols),
    e, ne, tt, tiles, c, in_cols,
    {==}, {==}, {==}
  )

# promise: every 256-half scale vector offset stays within the byte extent of the fp16 scale pool.
# harm: writing a block scale vector would overrun the allocated fp16 scale buffer.
# catches: scale vector stride 256n changed to 257n in ikr_scale_vec_off
law ikr_scale_vec_off_in_pool:
  for +e: Nat
  for +ne: Nat
  for +tt: Nat
  for +tiles: Nat
  for +b: Nat
  for +nblk: Nat
  for +bp: Nat
  {Bool.and(Bool.and(Nat.is_lt(e, ne), Bool.and(Nat.is_lt(tt, tiles), Nat.is_lt(b, nblk))), Nat.is_lt(ikr_scale_bytes(Nat.mul(nblk, 1n+bp), 1n+bp, tiles, ne), Nat.mul(Nat.add(ikr_scale_vec_off(e, tiles, nblk, tt, b), 256n), 2n))) == False{} : Bool}

def ikr_scale_vec_off_in_pool(e, ne, tt, tiles, b, nblk, bp):
  ikrp_scale_off_go(
    Nat.is_lt(e, ne), Nat.is_lt(tt, tiles), Nat.is_lt(b, nblk),
    e, ne, tt, tiles, b, nblk, bp,
    {==}, {==}, {==}
  )

# promise: for every byte lane in 0..256, `ikr_scale_lane_pos` lands inside the 256-half scale vector.
# harm: scale packing would index past the 256-entry scale vector or panic on slice write.
# catches: hi offset doubled from 128n to 256n in ikr_scale_lane_pos
law ikr_scale_lane_pos_in_vec:
  for +i: Nat
  {Bool.and(Nat.is_lt(i, 256n), Bool.not(Nat.is_lt(ikr_scale_lane_pos(i), 256n))) == False{} : Bool}

def ikr_scale_lane_pos_in_vec(i):
  ikrp_lane_vec_go(i, Nat.is_lt(i, 256n), Nat.is_lt(Nat.mod(i, 4n), 2n), {==}, {==})

# promise: `ikr_scale_lane_pos` lands within `< 256` and maps lanes `(4*l + 0, 1)` to `[0..128)` and `(4*l + 2, 3)` to `[128..256)` to match `v_convert_f16_to_f32_all`.
# harm: dequantizing W4/W8 columns on card would pair channels with the wrong accumulators and corrupt expert outputs.
# catches: Bool.pick returning 0n for all lanes in ikr_scale_lane_pos
law ikr_scale_lane_pos_half:
  for +i: Nat
  {Bool.and(Nat.is_lt(i, 256n), Bool.not(Bool.and(Nat.is_lt(ikr_scale_lane_pos(i), 256n), Bool.not(Bool.xor(Nat.is_lt(Nat.mod(i, 4n), 2n), Nat.is_lt(ikr_scale_lane_pos(i), 128n)))))) == False{} : Bool}

def ikr_scale_lane_pos_half(i):
  ikrp_lane_half2_go(i, Nat.is_lt(i, 256n), {==})

# promise: every valid `(expert, tile, col)` W4 offset is 128-byte aligned and uniquely decodes back to `(expert, tile, col)`, so distinct slices never overlap.
# harm: packing 4-bit expert slices could overwrite another expert, tile, or column nibble half.
# catches: ikr_w4_col_off returning 0n
law ikr_w4_col_off_decodes:
  for +e: Nat
  for +tp: Nat
  for +cpp: Nat
  for +tt: Nat
  for +col: Nat
  {Bool.and(Bool.and(Nat.is_lt(tt, 1n+tp), Nat.is_lt(col, Nat.mul(1n+cpp, 2n))), Bool.not(Bool.and(Nat.is_eq(Nat.mod(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 128n), 0n), Bool.and(Nat.is_eq(Nat.div(Nat.div(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 1n+tp), e), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 1n+tp), tt), Nat.is_eq(Nat.add(Nat.mul(Nat.mod(Nat.div(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 1n+cpp), 2n), Nat.div(Nat.mod(ikr_w4_col_off(e, 1n+tp, 1n+cpp, tt, col), 256n), 128n)), col)))))) == False{} : Bool}

def ikr_w4_col_off_decodes(e, tp, cpp, tt, col):
  ikrp_w4_dec_go(Nat.is_lt(tt, 1n+tp), Nat.is_lt(col, Nat.mul(1n+cpp, 2n)), e, tp, cpp, tt, col, {==}, {==})

# promise: every valid `(expert, tile, col)` W8 offset is 256-byte aligned and uniquely decodes back to `(expert, tile, col)`, so distinct vectors never overlap.
# harm: packing 8-bit expert columns could alias across experts, tiles, or columns.
# catches: ikr_w8_col_off returning 0n
law ikr_w8_col_off_decodes:
  for +e: Nat
  for +tp: Nat
  for +cp: Nat
  for +tt: Nat
  for +col: Nat
  {Bool.and(Bool.and(Nat.is_lt(tt, 1n+tp), Nat.is_lt(col, 1n+cp)), Bool.not(Bool.and(Nat.is_eq(Nat.mod(ikr_w8_col_off(e, 1n+tp, 1n+cp, tt, col), 256n), 0n), Bool.and(Nat.is_eq(Nat.mod(Nat.div(ikr_w8_col_off(e, 1n+tp, 1n+cp, tt, col), 256n), 1n+cp), col), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.div(ikr_w8_col_off(e, 1n+tp, 1n+cp, tt, col), 256n), 1n+cp), 1n+tp), tt), Nat.is_eq(Nat.div(Nat.div(Nat.div(ikr_w8_col_off(e, 1n+tp, 1n+cp, tt, col), 256n), 1n+cp), 1n+tp), e)))))) == False{} : Bool}

def ikr_w8_col_off_decodes(e, tp, cp, tt, col):
  ikrp_idx3_dec_go(Nat.is_lt(tt, 1n+tp), Nat.is_lt(col, 1n+cp), e, tp, cp, tt, col, {==}, {==})

# promise: every valid `(expert, tile, block)` fp16 scale vector offset is 256-half aligned and uniquely decodes back to `(expert, tile, block)`, so distinct scale vectors never overlap.
# harm: packing expert fp16 scale vectors could alias across experts, tiles, or scale blocks.
# catches: ikr_scale_vec_off returning 0n
law ikr_scale_vec_off_decodes:
  for +e: Nat
  for +tp: Nat
  for +nbp: Nat
  for +tt: Nat
  for +b: Nat
  {Bool.and(Bool.and(Nat.is_lt(tt, 1n+tp), Nat.is_lt(b, 1n+nbp)), Bool.not(Bool.and(Nat.is_eq(Nat.mod(ikr_scale_vec_off(e, 1n+tp, 1n+nbp, tt, b), 256n), 0n), Bool.and(Nat.is_eq(Nat.mod(Nat.div(ikr_scale_vec_off(e, 1n+tp, 1n+nbp, tt, b), 256n), 1n+nbp), b), Bool.and(Nat.is_eq(Nat.mod(Nat.div(Nat.div(ikr_scale_vec_off(e, 1n+tp, 1n+nbp, tt, b), 256n), 1n+nbp), 1n+tp), tt), Nat.is_eq(Nat.div(Nat.div(Nat.div(ikr_scale_vec_off(e, 1n+tp, 1n+nbp, tt, b), 256n), 1n+nbp), 1n+tp), e)))))) == False{} : Bool}

def ikr_scale_vec_off_decodes(e, tp, nbp, tt, b):
  ikrp_idx3_dec_go(Nat.is_lt(tt, 1n+tp), Nat.is_lt(b, 1n+nbp), e, tp, nbp, tt, b, {==}, {==})

# promise: packing lane `i < 256` with `p = ikr_scale_lane_pos(i)` stays within `< 256` and `v_convert_f16_to_f32_all` unpacks `p` back to the exact byte lane `i` (proving bijection and hardware lane order).
# harm: dequantized scale lanes would collide or pair with the wrong accumulator channels.
# catches: 2n multiplier on l replaced by 0n in ikr_scale_lane_pos
law ikr_scale_lane_pos_inverse:
  for +i: Nat
  {Bool.and(Nat.is_lt(i, 256n), Bool.not(Bool.and(Nat.is_lt(ikr_scale_lane_pos(i), 256n), Nat.is_eq(Nat.add(Nat.mul(4n, Nat.div(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(ikr_scale_lane_pos(i), 128n)), Nat.mod(Nat.mod(ikr_scale_lane_pos(i), 128n), 2n))), i)))) == False{} : Bool}

def ikr_scale_lane_pos_inverse(i):
  ikrp_lane_inv_go(i, Nat.is_lt(i, 256n), {==})

# promise: for every packed lane `lane < 256` in tile `tt`, `ikr_src_row` indexes a source channel strictly inside tile `tt`'s `[256*tt .. 256*(tt+1))` window.
# harm: packing or unpacking routed expert weights would read or scatter rows from an adjacent tile or past the row bound.
# catches: 128n tile stride omitted in ikr_src_row
law ikr_src_row_in_tile:
  for is_gu: Bool
  for +tt: Nat
  for +lane: Nat
  {Bool.and(Nat.is_lt(lane, 256n), Bool.not(Bool.and(Nat.is_le(Nat.mul(256n, tt), ikr_src_row(is_gu, tt, lane)), Nat.is_lt(ikr_src_row(is_gu, tt, lane), Nat.mul(256n, 1n+tt))))) == False{} : Bool}

def ikr_src_row_in_tile(is_gu, tt, lane):
  ikrp_src_tile_go(is_gu, tt, lane, Nat.is_lt(lane, 256n), {==})

# promise: every source row `ikr_src_row(is_gu, tt, lane)` for `lane < 256` uniquely decodes back to `(tt, lane)` for both gate+up (pairing `A[0]/A[2]` and `A[1]/A[3]` for SwiGLU) and down (storing 64-lane even/odd channel groups that `gaudium_ik_w4_comb` reinterleaves to BF16), proving bijection onto tile `tt`'s 256 source rows.
# harm: two packed lanes would read the same source row and drop weight channels during expert packing.
# catches: swap between div(lane, 4) and mod(lane, 4) or dropped +64/+128 half offset in ikr_src_row
law ikr_src_row_decodes:
  for is_gu: Bool
  for +tt: Nat
  for +lane: Nat
  {Bool.and(Nat.is_lt(lane, 256n), Bool.not(Bool.and(Nat.is_eq(Nat.div(ikr_src_row(is_gu, tt, lane), 256n), tt), Nat.is_eq(Bool.pick(Nat, is_gu, Nat.add(Nat.mul(4n, Nat.div(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.mod(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.div(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n))), Nat.add(Nat.mul(4n, Nat.div(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)), Nat.add(Nat.mul(2n, Nat.div(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n)), Nat.mod(Nat.mod(Nat.mod(ikr_src_row(is_gu, tt, lane), 256n), 128n), 2n)))), lane)))) == False{} : Bool}

def ikr_src_row_decodes(is_gu, tt, lane):
  ikrp_src_dec_go(is_gu, tt, lane, Nat.is_lt(lane, 256n), {==})

# --- 6.4 Dispatch plan, sampler vocab, and MTP width laws ---

# promise: the dispatch plan element count covers every header and row-slot entry written by `gaudium_ik_w4_plan`.
# harm: `gaudium_ik_w4_plan` would write row-slot pairs past the end of the `plan` tensor on card.
# catches: multiplier 5n replaced by 4n in ikr_plan_len
law ikr_plan_len_covers:
  for +k: Nat
  for +t: Nat
  for +u: Nat
  for +rs: Nat
  {Bool.and(Bool.and(Nat.is_le(u, Nat.mul(k, t)), Nat.is_lt(rs, Nat.mul(k, t))), Nat.is_le(ikr_plan_len(k, t), Nat.add(Nat.add(1n, Nat.mul(4n, u)), rs))) == False{} : Bool}

def ikr_plan_len_covers(k, t, u, rs):
  ikrp_plan_go(k, t, u, rs, Nat.is_le(u, Nat.mul(k, t)), Nat.is_lt(rs, Nat.mul(k, t)), {==}, {==})

# promise: the dispatch plan element count never exceeds `1 + 5 * k * t`.
# harm: step graph allocation would over-allocate `plan` vs what `gaudium_ik_w4_plan` expects.
# catches: multiplier 5n replaced by 6n in ikr_plan_len
law ikr_plan_len_within:
  for +k: Nat
  for +t: Nat
  {Nat.is_le(ikr_plan_len(k, t), Nat.add(1n, Nat.mul(5n, Nat.mul(k, t)))) == True{} : Bool}

def ikr_plan_len_within(k, t):
  L.nat_le_refl(ikr_plan_len(k, t))

# promise: every padded vocabulary size accepted by `ikr_sampler_vocab_ok` is divisible by 128 (matches `ik_softcap`/`ik_mcap` 128-lane grid).
# harm: launching `gaudium_ik_softcap` or `gaudium_ik_mcap` with misaligned `vocab_pad` would overrun logits or drop tail tokens.
# catches: modulo 128n check dropped in ikr_sampler_vocab_ok
law ikr_sampler_vocab_grid:
  for +vo: Nat
  for +vp: Nat
  {Bool.and(ikr_sampler_vocab_ok(vo, vp), Nat.is_ne(Nat.mul(Nat.div(vp, 128n), 128n), vp)) == False{} : Bool}

def ikr_sampler_vocab_grid(vo, vp):
  ikrp_svocab_grid_go(vo, vp, ikr_sampler_vocab_ok(vo, vp), {==})

# promise: every vocabulary pair accepted by `ikr_sampler_vocab_ok` has `1 <= vocab_out <= vocab_pad`.
# harm: softcap masking at `vocab_out..vocab_pad` or sampling over `vocab_pad` would mask live tokens or read out of range.
# catches: vocab_out <= vocab_pad bound dropped in ikr_sampler_vocab_ok
law ikr_sampler_vocab_range:
  for +vo: Nat
  for +vp: Nat
  {Bool.and(ikr_sampler_vocab_ok(vo, vp), Bool.not(Bool.and(Nat.is_gt(vo, 0n), Nat.is_le(vo, vp)))) == False{} : Bool}

def ikr_sampler_vocab_range(vo, vp):
  ikrp_svocab_rng_go(vo, vp, ikr_sampler_vocab_ok(vo, vp), {==})

# promise: every canonical 128-aligned vocabulary pair (`vp = align_up(vo, 128) <= 262144`) is accepted by `ikr_sampler_vocab_ok`.
# harm: valid Inkling 2 vocabularies would needlessly lose on-card sampling.
# catches: ikr_sampler_vocab_ok returning constant False{}
law ikr_sampler_vocab_accepts_valid:
  for +vp_blk: Nat
  for +r: Nat
  {Bool.and(Bool.and(Nat.is_lt(vp_blk, 2048n), Nat.is_lt(r, 128n)), Bool.not(ikr_sampler_vocab_ok(Nat.add(Nat.mul(vp_blk, 128n), 1n+r), Nat.mul(1n+vp_blk, 128n)))) == False{} : Bool}

def ikr_sampler_vocab_accepts_valid(vp_blk, r):
  ikrp_svocab_acc_go(vp_blk, r, Nat.is_lt(vp_blk, 2048n), Nat.is_lt(r, 128n), {==})

# promise: the chosen MTP draft width never exceeds `block - 1` when a verify block is configured (`block >= 1`).
# harm: the verify step width `1 + K` would exceed the model's MTP block width or draft past the block head.
# catches: Nat.min replaced by Nat.add in ikr_mtp_width
law ikr_mtp_width_in_block:
  for routed: Bool
  for +bp: Nat
  {Nat.is_le(ikr_mtp_width(routed, 1n+bp), bp) == True{} : Bool}

def ikr_mtp_width_in_block(routed, bp):
  match routed:
    case False{}:
      match bp:
        case 0n: {==}
        case 1n+bpp:
          L.nat_le_refl(1n+bpp)
    case True{}:
      match bp:
        case 0n: {==}
        case 1n+bpp:
          L.nat_min_le_r(1n, 1n+bpp)

# promise: whenever the MTP block allows speculation (`block >= 2`), at least one draft token is kept.
# harm: switching between dense and routed MoE could silently disable MTP speculation.
# catches: routed True arm returning 0n in ikr_mtp_width
law ikr_mtp_width_progress:
  for routed: Bool
  for +bpp: Nat
  {Nat.is_ge(ikr_mtp_width(routed, 2n+bpp), 1n) == True{} : Bool}

def ikr_mtp_width_progress(routed, bpp):
  match routed:
    case False{}:
      match bpp:
        case 0n: {==}
        case 1n+_: {==}
    case True{}:
      match bpp:
        case 0n: {==}
        case 1n+_: {==}
