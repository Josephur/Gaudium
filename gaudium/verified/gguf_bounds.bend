# GGUF container bounds: every length, count, size and offset the GGUF reader takes from a file.
#
# The Rust shell (`src/gguf/`) reads the bytes; every decision about whether a value read from
# the file may be used goes through a function here:
#   - a span of bytes lies inside the mapping (strings, scalars, arrays, tensor data);
#   - n elements of e bytes fit in the bytes left (array bodies, entry counts, products of dims,
#     block counts times block bytes), compared against a quotient so that nothing overflows;
#   - a tensor's byte size from its type's block geometry;
#   - the general.alignment value and the padding to the data section;
#   - a tensor's place in the data section (aligned, inside);
#   - a run of rows of a tensor, and the row partition of a bulk decode;
#   - small limits (array nesting, tensor rank) and the shard order of a split model;
#   - the value-head reorder of a gated DeltaNet layer, which maps a tensor's entries one-to-one
#     onto themselves when their number is the leading entries plus the value heads.
# The type table (block weights and block bytes per ggml type) and the value-type sizes are
# format facts of an outside specification; they live in Rust with tests.

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. Helper lemmas (erased before Rust codegen)
# ============================================================================

def gg_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# When `B` together with `C` is refused, so is `A` and `B` together with `C`.
def gg_and_l_false(a: Bool, -b: Bool, -c: Bool, h: {Bool.and(b, c) == False{} : Bool}) -> {Bool.and(Bool.and(a, b), c) == False{} : Bool}:
  match a:
    case True{}:
      h
    case False{}:
      {==}

# n <= a / (1 + ep) keeps n * (1 + ep) within a.
def gg_fit_within(+n: Nat, +ep: Nat, +a: Nat, b: Bool, e: {Nat.is_le(n, Nat.div(a, 1n+ep)) == b : Bool}) -> {Bool.and(b, Nat.is_lt(a, Nat.mul(n, 1n+ep))) == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      L.nat_le_lt_false(a, Nat.mul(n, 1n+ep), L.nat_le_trans(Nat.mul(n, 1n+ep), Nat.mul(Nat.div(a, 1n+ep), 1n+ep), a, L.nat_mul_le_mono_r(n, Nat.div(a, 1n+ep), 1n+ep, e), L.nat_le_subst_r(Nat.mul(Nat.div(a, 1n+ep), 1n+ep), Nat.add(Nat.mul(Nat.div(a, 1n+ep), 1n+ep), Nat.mod(a, 1n+ep)), a, L.nat_div_mod_r(a, ep), L.nat_add_le_r(Nat.mul(Nat.div(a, 1n+ep), 1n+ep), Nat.mod(a, 1n+ep)))))

# off <= size and len <= size - off keep off + len within size.
def gg_span_fit(+o: Nat, +l: Nat, +s: Nat, b1: Bool, e1: {Nat.is_le(o, s) == b1 : Bool}, b2: Bool, e2: {Nat.is_le(l, Nat.sub(s, o)) == b2 : Bool}) -> {Bool.and(Bool.and(b1, b2), Nat.is_lt(s, Nat.add(o, l))) == False{} : Bool}:
  match b1:
    case False{}:
      {==}
    case True{}:
      match b2:
        case False{}:
          {==}
        case True{}:
          L.nat_le_lt_false(s, Nat.add(o, l), L.nat_le_subst_r(Nat.add(o, l), Nat.add(o, Nat.sub(s, o)), s, Equal.trans(Nat, Nat.add(o, Nat.sub(s, o)), Nat.add(Nat.sub(s, o), o), s, L.nat_add_comm(o, Nat.sub(s, o)), L.nat_sub_add_cancel(s, o, e1)), L.nat_add_le_mono(o, o, l, Nat.sub(s, o), L.nat_le_refl(o), e2)))

# ============================================================================
# 2. Spans and counts
# ============================================================================

# `len` bytes starting at `off` lie inside a buffer of `size` bytes.
def gg_span_ok(+off: Nat, +len: Nat, +size: Nat) -> Bool:
  Bool.and(Nat.is_le(off, size), Nat.is_le(len, Nat.sub(size, off)))

# `n` elements of `elem` bytes each fit in `avail` bytes (a zero element size is refused).
# Compared against a quotient, so a huge count read from a file cannot overflow.
def gg_count_fits(+n: Nat, +elem: Nat, +avail: Nat) -> Bool:
  Bool.and(Nat.is_gt(elem, 0n), Nat.is_le(n, Nat.div(avail, elem)))

# Byte offset of element `k` of an array of `elem`-byte elements (callers first check that the
# whole array fits, so this never overflows for an index inside it).
def gg_elem_offset(k: Nat, elem: Nat) -> Nat:
  Nat.mul(k, elem)

# A dimension `d` may extend an element count `acc` that stays within `limit`.
def gg_dim_ok(+acc: Nat, +d: Nat, +limit: Nat) -> Bool:
  gg_count_fits(acc, d, limit)

# The element count after dimension `d`.
def gg_dim_step(acc: Nat, d: Nat) -> Nat:
  Nat.mul(acc, d)

# A sum of counts (tensors over the shards of a split model); `None` past 64 bits.
# rusty-bends: total
def gg_count_add(acc: Nat, n: Nat) -> Nat:
  Nat.add(acc, n)

# ============================================================================
# 3. Tensor byte sizes from block geometry
# ============================================================================

# A row of `row_len` weights is a whole number of blocks of `bw` weights.
def gg_blocks_ok(+row_len: Nat, +bw: Nat) -> Bool:
  Bool.and(Nat.is_gt(bw, 0n), Nat.is_eq(Nat.mod(row_len, bw), 0n))

# Bytes of `n` weights stored in blocks of `bw` weights and `bb` bytes.
def gg_nbytes(n: Nat, +bw: Nat, bb: Nat) -> Nat:
  Nat.mul(Nat.div(n, bw), bb)

# A tensor of `n` weights in rows of `row_len` may be stored in blocks of `bw` weights and
# `bb` bytes, its size staying within `limit`.
def gg_nbytes_ok(+row_len: Nat, +n: Nat, +bw: Nat, +bb: Nat, +limit: Nat) -> Bool:
  Bool.and(gg_blocks_ok(row_len, bw), gg_count_fits(Nat.div(n, bw), bb, limit))

# ============================================================================
# 4. Alignment and placement
# ============================================================================

# `general.alignment`: nonzero, at most `cap`, and a divisor of `cap` (with `cap` a power of
# two, exactly the powers of two up to `cap`).
def gg_alignment_ok(+a: Nat, +cap: Nat) -> Bool:
  Bool.and(Nat.is_gt(a, 0n), Bool.and(Nat.is_le(a, cap), Nat.is_eq(Nat.mod(cap, a), 0n)))

# Padding from position `x` to the next multiple of `a`.
def gg_pad(+x: Nat, +a: Nat) -> Nat:
  Nat.mod(Nat.sub(a, Nat.mod(x, a)), a)

# A position plus its padding (the caller checks the span first).
def gg_pos_add(x: Nat, d: Nat) -> Nat:
  Nat.add(x, d)

# A tensor of `len` bytes at offset `off` of a data section of `avail` bytes with alignment `a`.
def gg_place_ok(+off: Nat, +len: Nat, +a: Nat, +avail: Nat) -> Bool:
  Bool.and(Nat.is_eq(Nat.mod(off, a), 0n), gg_span_ok(off, len, avail))

# ============================================================================
# 5. Rows, partitions, limits, shards
# ============================================================================

# Rows [r0, r1) of rows of `row_bytes` bytes each lie inside `nbytes` bytes.
def gg_rows_ok(+r0: Nat, +r1: Nat, +row_bytes: Nat, +nbytes: Nat) -> Bool:
  Bool.and(Nat.is_le(r0, r1), gg_count_fits(r1, row_bytes, nbytes))

# Rows per part when `rows` rows are split over `parts` workers (0 parts: 0).
def gg_part_chunk(+rows: Nat, +parts: Nat) -> Nat:
  Nat.div(Nat.add(rows, Nat.sub(parts, 1n)), parts)

# First row of part `i` (each part covers [lo(i), lo(i + 1))), never past the end.
def gg_part_lo(i: Nat, +rows: Nat, chunk: Nat) -> Nat:
  Nat.min(rows, Nat.mul(i, chunk))

# A count `x` read from the file stays within `max` (array nesting, tensor rank).
def gg_limit_ok(+x: Nat, +max: Nat) -> Bool:
  Nat.is_le(x, max)

# Shard `k` of a split model of `count` shards says it is shard `no`.
def gg_shard_ok(+no: Nat, +k: Nat, +count: Nat) -> Bool:
  Bool.and(Nat.is_eq(no, k), Nat.is_lt(k, count))

# Shard `k` of a split model of `count` shards (the first shard's `split.count`) is in the set: its own
# `split.no` is `no` and its own `split.count` is `own`.
def gg_shard_in_set(+no: Nat, +own: Nat, +k: Nat, +count: Nat) -> Bool:
  Bool.and(gg_shard_ok(no, k, count), Nat.is_eq(own, count))

# The shards hold the tensor count the first shard declares (`split.tensors.count`).
def gg_tensor_total_ok(+declared: Nat, +total: Nat) -> Bool:
  Nat.is_eq(declared, total)

# `present` shards make up a split model of `count` shards.
def gg_split_complete(+present: Nat, +count: Nat) -> Bool:
  Bool.and(Nat.is_gt(count, 0n), Nat.is_eq(present, count))

# ============================================================================
# 5b. Value-head order of a gated DeltaNet layer
# ============================================================================

# A converter may store the value heads of a linear-attention layer tiled, (r, k), where the
# checkpoint groups them, (k, r): `hk` key heads, `r` value heads per key head, `d` values per
# head, after `base` leading entries that keep their place. Checkpoint index `i` reads stored
# index base + rr * (hk * d) + k * d + dd, with i - base = k * (r * d) + rr * d + dd.
# The value-head part: stored offset rr * (hk * d) + (k * d + dd) of checkpoint offset j.
def gg_vhead_sum(+j: Nat, +hk: Nat, +r: Nat, +d: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(Nat.mod(j, Nat.mul(r, d)), d), Nat.mul(hk, d)), Nat.add(Nat.mul(Nat.div(j, Nat.mul(r, d)), d), Nat.mod(Nat.mod(j, Nat.mul(r, d)), d)))

def gg_vhead_tail(+j: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat) -> Nat:
  Nat.add(base, gg_vhead_sum(j, hk, r, d))

def gg_vhead_pick(+i: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, lead: Bool) -> Nat:
  match lead:
    case True{}:
      i
    case False{}:
      gg_vhead_tail(Nat.sub(i, base), base, hk, r, d)

# rusty-bends: total
def gg_vhead_src(+i: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat) -> Nat:
  gg_vhead_pick(i, base, hk, r, d, Nat.is_lt(i, base))

# The `base` leading entries and the value heads: the number of entries over which the reorder
# stays in range and sends no two entries to one place (`gg_vhead_src_in_range`,
# `gg_vhead_src_one_to_one`).
def gg_vhead_len(+base: Nat, +hk: Nat, +r: Nat, +d: Nat) -> Nat:
  Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))

# Whether a tensor of `n` entries has that number (`gg_vhead_fits_only_len`).
# rusty-bends: total
def gg_vhead_fits(+n: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat) -> Bool:
  Nat.is_eq(n, gg_vhead_len(base, hk, r, d))

# ============================================================================
# 6. Laws
# ============================================================================

# promise: a span the reader accepts never ends past the end of the buffer.
# harm: an accepted span could read outside the file mapping.
law gg_span_within:
  for +o: Nat
  for +l: Nat
  for +s: Nat
  {Bool.and(gg_span_ok(o, l, s), Nat.is_lt(s, Nat.add(o, l))) == False{} : Bool}

def gg_span_within(o, l, s):
  gg_span_fit(o, l, s, Nat.is_le(o, s), {==}, Nat.is_le(l, Nat.sub(s, o)), {==})

# promise: n elements of e bytes that the count check accepts never need more than the available bytes.
# harm: a hostile count could make the reader walk past the end of the file.
law gg_count_within:
  for +n: Nat
  for +e: Nat
  for +a: Nat
  {Bool.and(gg_count_fits(n, e, a), Nat.is_lt(a, Nat.mul(n, e))) == False{} : Bool}

def gg_count_within(n, e, a):
  match e:
    case 0n:
      {==}
    case 1n+ep:
      gg_fit_within(n, ep, a, Nat.is_le(n, Nat.div(a, 1n+ep)), {==})

# promise: a dimension the reader accepts never makes the element count exceed its limit.
law gg_dim_within:
  for +acc: Nat
  for +d: Nat
  for +lim: Nat
  {Bool.and(gg_dim_ok(acc, d, lim), Nat.is_lt(lim, gg_dim_step(acc, d))) == False{} : Bool}

def gg_dim_within(acc, d, lim):
  gg_count_within(acc, d, lim)

# promise: a tensor size the reader accepts never exceeds the limit it was checked against.
# harm: a tensor could claim more bytes than the file holds.
law gg_nbytes_within:
  for +r: Nat
  for +n: Nat
  for +bw: Nat
  for +bb: Nat
  for +lim: Nat
  {Bool.and(gg_nbytes_ok(r, n, bw, bb, lim), Nat.is_lt(lim, gg_nbytes(n, bw, bb))) == False{} : Bool}

def gg_nbytes_within(r, n, bw, bb, lim):
  gg_and_l_false(gg_blocks_ok(r, bw), gg_count_fits(Nat.div(n, bw), bb, lim), Nat.is_lt(lim, Nat.mul(Nat.div(n, bw), bb)), gg_count_within(Nat.div(n, bw), bb, lim))

# promise: a type whose block holds no weights is always refused, so no size is ever divided by zero.
law gg_nbytes_zero_block_refused:
  for +r: Nat
  for +n: Nat
  for +bb: Nat
  for +lim: Nat
  {gg_nbytes_ok(r, n, 0n, bb, lim) == False{} : Bool}

def gg_nbytes_zero_block_refused(r, n, bb, lim):
  {==}

# promise: an alignment the reader accepts is never zero.
# harm: a zero alignment would divide by zero when the data section is placed.
law gg_alignment_nonzero:
  for +a: Nat
  for +c: Nat
  {Bool.and(gg_alignment_ok(a, c), Nat.is_eq(a, 0n)) == False{} : Bool}

def gg_alignment_nonzero(a, c):
  match a:
    case 0n:
      {==}
    case 1n+p:
      gg_and_false_r(gg_alignment_ok(1n+p, c))

# promise: the padding to the next multiple of a nonzero alignment is always smaller than the alignment.
law gg_pad_below_alignment:
  for +x: Nat
  for +ap: Nat
  {Nat.is_lt(gg_pad(x, 1n+ap), 1n+ap) == True{} : Bool}

def gg_pad_below_alignment(x, ap):
  L.nat_mod_lt(Nat.sub(1n+ap, Nat.mod(x, 1n+ap)), ap)

# promise: a tensor the reader places in the data section never ends past the end of that section.
# harm: tensor bytes could be read from outside the file mapping.
law gg_place_within:
  for +o: Nat
  for +l: Nat
  for +a: Nat
  for +v: Nat
  {Bool.and(gg_place_ok(o, l, a, v), Nat.is_lt(v, Nat.add(o, l))) == False{} : Bool}

def gg_place_within(o, l, a, v):
  gg_and_l_false(Nat.is_eq(Nat.mod(o, a), 0n), gg_span_ok(o, l, v), Nat.is_lt(v, Nat.add(o, l)), gg_span_within(o, l, v))

# promise: a run of rows the reader accepts never ends past the end of the tensor's bytes.
# harm: a row decode could read the next tensor or past the mapping.
law gg_rows_within:
  for +r0: Nat
  for +r1: Nat
  for +rb: Nat
  for +nb: Nat
  {Bool.and(gg_rows_ok(r0, r1, rb, nb), Nat.is_lt(nb, Nat.mul(r1, rb))) == False{} : Bool}

def gg_rows_within(r0, r1, rb, nb):
  gg_and_l_false(Nat.is_le(r0, r1), gg_count_fits(r1, rb, nb), Nat.is_lt(nb, Nat.mul(r1, rb)), gg_count_within(r1, rb, nb))

# promise: the first row of a decode part never lies past the last row.
law gg_part_lo_within:
  for +i: Nat
  for +rows: Nat
  for +c: Nat
  {Nat.is_le(gg_part_lo(i, rows, c), rows) == True{} : Bool}

def gg_part_lo_within(i, rows, c):
  L.nat_min_le_l(rows, Nat.mul(i, c))

# promise: a count the limit check accepts never exceeds the limit.
law gg_limit_within:
  for +x: Nat
  for +m: Nat
  {Bool.and(gg_limit_ok(x, m), Nat.is_lt(m, x)) == False{} : Bool}

def gg_limit_within(x, m):
  match x m:
    case 0n 0n:
      {==}
    case 0n 1n+mp:
      {==}
    case 1n+xp 0n:
      {==}
    case 1n+xp 1n+mp:
      gg_limit_within(xp, mp)

def gg_shard_fit(a: Bool, +k: Nat, +c: Nat, b: Bool, e: {Nat.is_lt(k, c) == b : Bool}) -> {Bool.and(Bool.and(a, b), Nat.is_le(c, k)) == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      match b:
        case False{}:
          {==}
        case True{}:
          L.nat_lt_le_false(k, c, e)

# promise: a shard the split check accepts always has an index below the shard count.
law gg_shard_index_within:
  for +no: Nat
  for +k: Nat
  for +c: Nat
  {Bool.and(gg_shard_ok(no, k, c), Nat.is_le(c, k)) == False{} : Bool}

def gg_shard_index_within(no, k, c):
  gg_shard_fit(Nat.is_eq(no, k), k, c, Nat.is_lt(k, c), {==})

def gg_and_not_r(a: Bool, e: Bool) -> {Bool.and(Bool.and(a, e), Bool.not(e)) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}:
      match e:
        case True{}: {==}
        case False{}: {==}

def gg_and_mid(a: Bool, e: Bool, -c: Bool, h: {Bool.and(a, c) == False{} : Bool}) -> {Bool.and(Bool.and(a, e), c) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}:
      match e:
        case False{}: {==}
        case True{}: h

def gg_and_not_self(x: Bool) -> {Bool.and(x, Bool.not(x)) == False{} : Bool}:
  match x:
    case True{}: {==}
    case False{}: {==}

# promise: every shard the reader accepts into a split model declares the same shard count as the first one.
# harm: shards of two different splits could be stitched into one model
law gg_shard_in_set_count:
  for +no: Nat
  for +own: Nat
  for +k: Nat
  for +c: Nat
  {Bool.and(gg_shard_in_set(no, own, k, c), Nat.is_ne(own, c)) == False{} : Bool}

def gg_shard_in_set_count(no, own, k, c):
  gg_and_not_r(gg_shard_ok(no, k, c), Nat.is_eq(own, c))

# promise: a shard the reader accepts into a split model has an index below the shard count.
law gg_shard_in_set_index:
  for +no: Nat
  for +own: Nat
  for +k: Nat
  for +c: Nat
  {Bool.and(gg_shard_in_set(no, own, k, c), Nat.is_le(c, k)) == False{} : Bool}

def gg_shard_in_set_index(no, own, k, c):
  gg_and_mid(gg_shard_ok(no, k, c), Nat.is_eq(own, c), Nat.is_le(c, k), gg_shard_fit(Nat.is_eq(no, k), k, c, Nat.is_lt(k, c), {==}))

# promise: a split model is accepted only when its shards hold the tensor count its first shard declares.
# harm: a missing or duplicated shard's tensors would go unnoticed
law gg_tensor_total_matches:
  for +d: Nat
  for +t: Nat
  {Bool.and(gg_tensor_total_ok(d, t), Nat.is_ne(d, t)) == False{} : Bool}

def gg_tensor_total_matches(d, t):
  gg_and_not_self(Nat.is_eq(d, t))

# promise: a model with no shards at all is never taken as a complete split model.
law gg_split_empty_refused:
  for +c: Nat
  {gg_split_complete(0n, c) == False{} : Bool}

def gg_split_empty_refused(c):
  match c:
    case 0n:
      {==}
    case 1n+cp:
      {==}

def gg_lt_add_succ(i: Nat, p: Nat) -> {True{} == Nat.is_lt(i, Nat.add(i, 1n+p)) : Bool}:
  match i:
    case 0n:
      {==}
    case 1n+q:
      gg_lt_add_succ(q, p)

# promise: the leading entries before the value heads keep their place in any head reordering.
law gg_vhead_lead_fixed:
  for +i: Nat
  for +p: Nat
  for +hk: Nat
  for +r: Nat
  for +d: Nat
  {gg_vhead_src(i, Nat.add(i, 1n+p), hk, r, d) == i : Nat}

def gg_vhead_lead_fixed(i, p, hk, r, d):
  %gg_lt_add_succ(i, p) : {gg_vhead_pick(i, Nat.add(i, 1n+p), hk, r, d, _) == i : Nat}
  {==}

# --- The value-head reorder maps a tensor that fits onto itself, one-to-one (helpers) ---------
# A value offset j < hk * (r * d) splits as j = k * (r * d) + q with q = rr * d + dd; its stored
# offset rr * (hk * d) + (k * d + dd) is below hk * (r * d) and gives back rr, k and dd, hence j.

def gg_lt_zero_absurd(-A: Type, +j: Nat, +h: {Nat.is_lt(j, 0n) == True{} : Bool}) -> A:
  match j:
    case 0n:
      L.bool_absurd(A, h)
    case 1n+jp:
      L.bool_absurd(A, h)

def gg_eq_of_is_eq(+a: Nat, +b: Nat, +h: {Nat.is_eq(a, b) == True{} : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      L.bool_absurd({0n == 1n+bp : Nat}, h)
    case 1n+ap 0n:
      L.bool_absurd({1n+ap == 0n : Nat}, h)
    case 1n+ap 1n+bp:
      %gg_eq_of_is_eq(ap, bp, h) : {1n+ap == 1n+_ : Nat}
      {==}

def gg_is_ne_self(+n: Nat) -> {Nat.is_ne(n, n) == False{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      gg_is_ne_self(p)

def gg_le_false_go(+a: Nat, +b: Nat, +h: {Nat.is_lt(a, b) == True{} : Bool}, +v: Bool, +e: {Nat.is_le(b, a) == v : Bool}) -> {v == False{} : Bool}:
  match v:
    case True{}:
      Empty.absurd({True{} == False{} : Bool}, L.nat_lt_not_le(a, b, h, e))
    case False{}:
      {==}

def gg_le_false_of_lt(+a: Nat, +b: Nat, +h: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_le(b, a) == False{} : Bool}:
  gg_le_false_go(a, b, h, Nat.is_le(b, a), {==})

# (a * b) * c == b * (a * c)
def gg_mul_swap_l(+a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(Nat.mul(a, b), c) == Nat.mul(b, Nat.mul(a, c)) : Nat}:
  +a2 = {a : Nat}
  +b2 = {b : Nat}
  +c2 = {c : Nat}
  %L.nat_mul_comm(b2, a2) : {Nat.mul(_, c2) == Nat.mul(b2, Nat.mul(a2, c2)) : Nat}
  L.nat_mul_assoc(b2, a2, c2)

# a * (b * c) == b * (a * c)
def gg_mul_swap(+a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(a, Nat.mul(b, c)) == Nat.mul(b, Nat.mul(a, c)) : Nat}:
  +a2 = {a : Nat}
  +b2 = {b : Nat}
  +c2 = {c : Nat}
  %L.nat_mul_assoc(a2, b2, c2) : {_ == Nat.mul(b2, Nat.mul(a2, c2)) : Nat}
  gg_mul_swap_l(a2, b2, c2)

# a * (b * 0) == 0
def gg_mul_mul_zero(+a: Nat, +b: Nat) -> {Nat.mul(a, Nat.mul(b, 0n)) == 0n : Nat}:
  +a2 = {a : Nat}
  +b2 = {b : Nat}
  %Equal.sym(Nat, Nat.mul(b2, 0n), 0n, L.nat_mul_zero_r(b2)) : {Nat.mul(a2, _) == 0n : Nat}
  L.nat_mul_zero_r(a2)

# With hk, r and d all positive: the stored offset of j < hk * (r * d) is below hk * (r * d).
def gg_vt_lt(+j: Nat, +hp: Nat, +rp: Nat, +dp: Nat, +h: {Nat.is_lt(j, Nat.mul(1n+hp, Nat.mul(1n+rp, 1n+dp))) == True{} : Bool}) -> {Nat.is_lt(gg_vhead_sum(j, 1n+hp, 1n+rp, 1n+dp), Nat.mul(1n+hp, Nat.mul(1n+rp, 1n+dp))) == True{} : Bool}:
  +j2 = {j : Nat}
  +hp2 = {hp : Nat}
  +rp2 = {rp : Nat}
  +dp2 = {dp : Nat}
  +kl = L.nat_div_lt_of_lt_mul(j2, 1n+hp2, Nat.add(dp2, Nat.mul(rp2, 1n+dp2)), h)
  +ql = L.nat_mod_lt(j2, Nat.add(dp2, Nat.mul(rp2, 1n+dp2)))
  +rrl = L.nat_div_lt_of_lt_mul(Nat.mod(j2, Nat.mul(1n+rp2, 1n+dp2)), 1n+rp2, dp2, ql)
  +ddl = L.nat_mod_lt(Nat.mod(j2, Nat.mul(1n+rp2, 1n+dp2)), dp2)
  +tl = L.nat_split_lt(Nat.div(j2, Nat.mul(1n+rp2, 1n+dp2)), Nat.mod(Nat.mod(j2, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), 1n+hp2, dp2, kl, ddl)
  +ol = L.nat_split_lt(Nat.div(Nat.mod(j2, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.add(Nat.mul(Nat.div(j2, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.mod(Nat.mod(j2, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2)), 1n+rp2, Nat.add(dp2, Nat.mul(hp2, 1n+dp2)), rrl, tl)
  L.nat_lt_subst_r(gg_vhead_sum(j2, 1n+hp2, 1n+rp2, 1n+dp2), Nat.mul(1n+rp2, Nat.mul(1n+hp2, 1n+dp2)), Nat.mul(1n+hp2, Nat.mul(1n+rp2, 1n+dp2)), gg_mul_swap(1n+rp2, 1n+hp2, 1n+dp2), ol)

# With hk, r and d all positive: two offsets below hk * (r * d) with one stored offset are one.
def gg_vt_inj(+j1: Nat, +j2: Nat, +hp: Nat, +rp: Nat, +dp: Nat, +h1: {Nat.is_lt(j1, Nat.mul(1n+hp, Nat.mul(1n+rp, 1n+dp))) == True{} : Bool}, +h2: {Nat.is_lt(j2, Nat.mul(1n+hp, Nat.mul(1n+rp, 1n+dp))) == True{} : Bool}, +e: {gg_vhead_sum(j1, 1n+hp, 1n+rp, 1n+dp) == gg_vhead_sum(j2, 1n+hp, 1n+rp, 1n+dp) : Nat}) -> {j1 == j2 : Nat}:
  +a = {j1 : Nat}
  +b = {j2 : Nat}
  +hp2 = {hp : Nat}
  +rp2 = {rp : Nat}
  +dp2 = {dp : Nat}
  +kl1 = L.nat_div_lt_of_lt_mul(a, 1n+hp2, Nat.add(dp2, Nat.mul(rp2, 1n+dp2)), h1)
  +kl2 = L.nat_div_lt_of_lt_mul(b, 1n+hp2, Nat.add(dp2, Nat.mul(rp2, 1n+dp2)), h2)
  +ql1 = L.nat_mod_lt(a, Nat.add(dp2, Nat.mul(rp2, 1n+dp2)))
  +ql2 = L.nat_mod_lt(b, Nat.add(dp2, Nat.mul(rp2, 1n+dp2)))
  +rrl1 = L.nat_div_lt_of_lt_mul(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+rp2, dp2, ql1)
  +rrl2 = L.nat_div_lt_of_lt_mul(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+rp2, dp2, ql2)
  +ddl1 = L.nat_mod_lt(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), dp2)
  +ddl2 = L.nat_mod_lt(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), dp2)
  +tl1 = L.nat_split_lt(Nat.div(a, Nat.mul(1n+rp2, 1n+dp2)), Nat.mod(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), 1n+hp2, dp2, kl1, ddl1)
  +tl2 = L.nat_split_lt(Nat.div(b, Nat.mul(1n+rp2, 1n+dp2)), Nat.mod(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), 1n+hp2, dp2, kl2, ddl2)
  +e_t = L.nat_split_unique_r(Nat.div(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.add(Nat.mul(Nat.div(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.mod(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2)), Nat.div(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.add(Nat.mul(Nat.div(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.mod(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2)), Nat.add(dp2, Nat.mul(hp2, 1n+dp2)), tl1, tl2, e)
  +e_rr = L.nat_split_unique_q(Nat.div(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.add(Nat.mul(Nat.div(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.mod(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2)), Nat.div(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.add(Nat.mul(Nat.div(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.mod(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2)), Nat.add(dp2, Nat.mul(hp2, 1n+dp2)), tl1, tl2, e)
  +e_k = L.nat_split_unique_q(Nat.div(a, Nat.mul(1n+rp2, 1n+dp2)), Nat.mod(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.div(b, Nat.mul(1n+rp2, 1n+dp2)), Nat.mod(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), dp2, ddl1, ddl2, e_t)
  +e_dd = L.nat_split_unique_r(Nat.div(a, Nat.mul(1n+rp2, 1n+dp2)), Nat.mod(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), Nat.div(b, Nat.mul(1n+rp2, 1n+dp2)), Nat.mod(Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), 1n+dp2), dp2, ddl1, ddl2, e_t)
  +e_q = L.nat_eq_of_div_mod(Nat.mod(a, Nat.mul(1n+rp2, 1n+dp2)), Nat.mod(b, Nat.mul(1n+rp2, 1n+dp2)), dp2, e_rr, e_dd)
  L.nat_eq_of_div_mod(a, b, Nat.add(dp2, Nat.mul(rp2, 1n+dp2)), e_k, e_q)

# Any hk, r, d: an offset below hk * (r * d) exists only when all three are positive.
def gg_vt_lt_any(+j: Nat, +hk: Nat, +r: Nat, +d: Nat, +h: {Nat.is_lt(j, Nat.mul(hk, Nat.mul(r, d))) == True{} : Bool}) -> {Nat.is_lt(gg_vhead_sum(j, hk, r, d), Nat.mul(hk, Nat.mul(r, d))) == True{} : Bool}:
  match hk:
    case 0n:
      gg_lt_zero_absurd({Nat.is_lt(gg_vhead_sum(j, 0n, r, d), Nat.mul(0n, Nat.mul(r, d))) == True{} : Bool}, j, h)
    case 1n+hp:
      match r:
        case 0n:
          gg_lt_zero_absurd({Nat.is_lt(gg_vhead_sum(j, 1n+hp, 0n, d), Nat.mul(1n+hp, Nat.mul(0n, d))) == True{} : Bool}, j, L.nat_lt_subst_r(j, Nat.mul(1n+hp, 0n), 0n, L.nat_mul_zero_r(1n+hp), h))
        case 1n+rp:
          match d:
            case 0n:
              gg_lt_zero_absurd({Nat.is_lt(gg_vhead_sum(j, 1n+hp, 1n+rp, 0n), Nat.mul(1n+hp, Nat.mul(1n+rp, 0n))) == True{} : Bool}, j, L.nat_lt_subst_r(j, Nat.mul(1n+hp, Nat.mul(1n+rp, 0n)), 0n, gg_mul_mul_zero(1n+hp, 1n+rp), h))
            case 1n+dp:
              gg_vt_lt(j, hp, rp, dp, h)

def gg_vt_inj_any(+j1: Nat, +j2: Nat, +hk: Nat, +r: Nat, +d: Nat, +h1: {Nat.is_lt(j1, Nat.mul(hk, Nat.mul(r, d))) == True{} : Bool}, +h2: {Nat.is_lt(j2, Nat.mul(hk, Nat.mul(r, d))) == True{} : Bool}, +e: {gg_vhead_sum(j1, hk, r, d) == gg_vhead_sum(j2, hk, r, d) : Nat}) -> {j1 == j2 : Nat}:
  match hk:
    case 0n:
      gg_lt_zero_absurd({j1 == j2 : Nat}, j1, h1)
    case 1n+hp:
      match r:
        case 0n:
          gg_lt_zero_absurd({j1 == j2 : Nat}, j1, L.nat_lt_subst_r(j1, Nat.mul(1n+hp, 0n), 0n, L.nat_mul_zero_r(1n+hp), h1))
        case 1n+rp:
          match d:
            case 0n:
              gg_lt_zero_absurd({j1 == j2 : Nat}, j1, L.nat_lt_subst_r(j1, Nat.mul(1n+hp, Nat.mul(1n+rp, 0n)), 0n, gg_mul_mul_zero(1n+hp, 1n+rp), h1))
            case 1n+dp:
              gg_vt_inj(j1, j2, hp, rp, dp, h1, h2, e)

# An entry at or past the leading ones of a tensor of base + x entries means x is positive.
def gg_vhead_pos(+i: Nat, +base: Nat, +x: Nat, +hlt: {Nat.is_lt(i, Nat.add(base, x)) == True{} : Bool}, +hge: {Nat.is_lt(i, base) == False{} : Bool}) -> {Nat.is_lt(0n, x) == True{} : Bool}:
  match x:
    case 0n:
      +a = L.nat_lt_subst_r(i, Nat.add(base, 0n), base, L.nat_add_zero_r(base), hlt)
      L.bool_absurd_tf({Nat.is_lt(0n, 0n) == True{} : Bool}, Equal.trans(Bool, True{}, Nat.is_lt(i, base), False{}, Equal.sym(Bool, Nat.is_lt(i, base), True{}, a), hge))
    case 1n+xp:
      {==}

def gg_vsrc_lt_go(+i: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, +h: {Nat.is_lt(i, Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))) == True{} : Bool}, +b: Bool, +eb: {Nat.is_lt(i, base) == b : Bool}) -> {Nat.is_lt(gg_vhead_pick(i, base, hk, r, d, b), Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))) == True{} : Bool}:
  match b:
    case True{}:
      h
    case False{}:
      +hn = gg_vhead_pos(i, base, Nat.mul(hk, Nat.mul(r, d)), h, eb)
      +hj = L.nat_sub_lt(i, base, Nat.mul(hk, Nat.mul(r, d)), hn, h)
      L.nat_add_lt_l(base, gg_vhead_sum(Nat.sub(i, base), hk, r, d), Nat.mul(hk, Nat.mul(r, d)), gg_vt_lt_any(Nat.sub(i, base), hk, r, d, hj))

def gg_vsrc_lt(+i: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, +h: {Nat.is_lt(i, Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))) == True{} : Bool}) -> {Nat.is_lt(gg_vhead_src(i, base, hk, r, d), Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))) == True{} : Bool}:
  gg_vsrc_lt_go(i, base, hk, r, d, h, Nat.is_lt(i, base), {==})

def gg_vsrc_inj_go(+i: Nat, +j: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, +hi: {Nat.is_lt(i, Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))) == True{} : Bool}, +hj: {Nat.is_lt(j, Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))) == True{} : Bool}, +bi: Bool, +ebi: {Nat.is_lt(i, base) == bi : Bool}, +bj: Bool, +ebj: {Nat.is_lt(j, base) == bj : Bool}, +e: {gg_vhead_pick(i, base, hk, r, d, bi) == gg_vhead_pick(j, base, hk, r, d, bj) : Nat}) -> {i == j : Nat}:
  match bi:
    case True{}:
      match bj:
        case True{}:
          e
        case False{}:
          Empty.absurd({i == j : Nat}, L.nat_add_not_lt_l(base, gg_vhead_sum(Nat.sub(j, base), hk, r, d), L.nat_lt_subst_l(i, Nat.add(base, gg_vhead_sum(Nat.sub(j, base), hk, r, d)), base, e, ebi)))
    case False{}:
      match bj:
        case True{}:
          Empty.absurd({i == j : Nat}, L.nat_add_not_lt_l(base, gg_vhead_sum(Nat.sub(i, base), hk, r, d), L.nat_lt_subst_l(j, Nat.add(base, gg_vhead_sum(Nat.sub(i, base), hk, r, d)), base, Equal.sym(Nat, Nat.add(base, gg_vhead_sum(Nat.sub(i, base), hk, r, d)), j, e), ebj)))
        case False{}:
          +es = L.nat_add_cancel_l(base, gg_vhead_sum(Nat.sub(i, base), hk, r, d), gg_vhead_sum(Nat.sub(j, base), hk, r, d), e)
          +hji = L.nat_sub_lt(i, base, Nat.mul(hk, Nat.mul(r, d)), gg_vhead_pos(i, base, Nat.mul(hk, Nat.mul(r, d)), hi, ebi), hi)
          +hjj = L.nat_sub_lt(j, base, Nat.mul(hk, Nat.mul(r, d)), gg_vhead_pos(j, base, Nat.mul(hk, Nat.mul(r, d)), hj, ebj), hj)
          +ejj = gg_vt_inj_any(Nat.sub(i, base), Nat.sub(j, base), hk, r, d, hji, hjj, es)
          %L.nat_sub_add_cancel(i, base, L.nat_not_lt_imp_le(i, base, ebi)) : {_ == j : Nat}
          %L.nat_sub_add_cancel(j, base, L.nat_not_lt_imp_le(j, base, ebj)) : {Nat.add(Nat.sub(i, base), base) == _ : Nat}
          %ejj : {Nat.add(Nat.sub(i, base), base) == Nat.add(_, base) : Nat}
          {==}

def gg_vsrc_inj(+i: Nat, +j: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, +hi: {Nat.is_lt(i, Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))) == True{} : Bool}, +hj: {Nat.is_lt(j, Nat.add(base, Nat.mul(hk, Nat.mul(r, d)))) == True{} : Bool}, +e: {gg_vhead_src(i, base, hk, r, d) == gg_vhead_src(j, base, hk, r, d) : Nat}) -> {i == j : Nat}:
  gg_vsrc_inj_go(i, j, base, hk, r, d, hi, hj, Nat.is_lt(i, base), {==}, Nat.is_lt(j, base), {==}, e)

def gg_vf_go(+n: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, +f: Bool, +ef: {gg_vhead_fits(n, base, hk, r, d) == f : Bool}) -> {Bool.and(f, Nat.is_ne(n, gg_vhead_len(base, hk, r, d))) == False{} : Bool}:
  match f:
    case False{}:
      {==}
    case True{}:
      +en = gg_eq_of_is_eq(n, gg_vhead_len(base, hk, r, d), ef)
      %Equal.sym(Nat, n, gg_vhead_len(base, hk, r, d), en) : {Nat.is_ne(_, gg_vhead_len(base, hk, r, d)) == False{} : Bool}
      gg_is_ne_self(gg_vhead_len(base, hk, r, d))

def gg_vr_go(+i: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, +li: Bool, +eli: {Nat.is_lt(i, gg_vhead_len(base, hk, r, d)) == li : Bool}) -> {Bool.and(li, Nat.is_le(gg_vhead_len(base, hk, r, d), gg_vhead_src(i, base, hk, r, d))) == False{} : Bool}:
  match li:
    case False{}:
      {==}
    case True{}:
      gg_le_false_of_lt(gg_vhead_src(i, base, hk, r, d), gg_vhead_len(base, hk, r, d), gg_vsrc_lt(i, base, hk, r, d, eli))

def gg_vi_go(+i: Nat, +j: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, +li: Bool, +eli: {Nat.is_lt(i, gg_vhead_len(base, hk, r, d)) == li : Bool}, +lj: Bool, +elj: {Nat.is_lt(j, gg_vhead_len(base, hk, r, d)) == lj : Bool}, +s: Bool, +es: {Nat.is_eq(gg_vhead_src(i, base, hk, r, d), gg_vhead_src(j, base, hk, r, d)) == s : Bool}) -> {Bool.and(Bool.and(li, lj), Bool.and(s, Nat.is_ne(i, j))) == False{} : Bool}:
  match li:
    case False{}:
      {==}
    case True{}:
      match lj:
        case False{}:
          {==}
        case True{}:
          match s:
            case False{}:
              {==}
            case True{}:
              +eij = gg_vsrc_inj(i, j, base, hk, r, d, eli, elj, gg_eq_of_is_eq(gg_vhead_src(i, base, hk, r, d), gg_vhead_src(j, base, hk, r, d), es))
              %eij : {Nat.is_ne(i, _) == False{} : Bool}
              gg_is_ne_self(i)

# promise: the value-head length check admits a tensor only when its number of entries is exactly its leading entries plus its value heads (`gg_vhead_len`).
# harm: a tensor of another length would be reordered by a map not shown to stay inside it or to keep its entries apart
# catches: a shorter tensor admitted (`Nat.is_le` in place of `Nat.is_eq`), which the reorder can read past the end of
law gg_vhead_fits_only_len:
  for +n: Nat
  for +base: Nat
  for +hk: Nat
  for +r: Nat
  for +d: Nat
  {Bool.and(gg_vhead_fits(n, base, hk, r, d), Nat.is_ne(n, gg_vhead_len(base, hk, r, d))) == False{} : Bool}

def gg_vhead_fits_only_len(n, base, hk, r, d):
  gg_vf_go(n, base, hk, r, d, gg_vhead_fits(n, base, hk, r, d), {==})

# promise: over a tensor of exactly its leading entries and its value heads (`gg_vhead_len`), every entry is read from a place inside the tensor.
# harm: a reordered row or column would be read from outside its tensor
# catches: the value heads laid out with the wrong stride (rr * (r * d) in place of rr * (hk * d)), which reads past the end when a key head has more value heads than there are key heads
law gg_vhead_src_in_range:
  for +i: Nat
  for +base: Nat
  for +hk: Nat
  for +r: Nat
  for +d: Nat
  {Bool.and(Nat.is_lt(i, gg_vhead_len(base, hk, r, d)), Nat.is_le(gg_vhead_len(base, hk, r, d), gg_vhead_src(i, base, hk, r, d))) == False{} : Bool}

def gg_vhead_src_in_range(i, base, hk, r, d):
  gg_vr_go(i, base, hk, r, d, Nat.is_lt(i, gg_vhead_len(base, hk, r, d)), {==})

# promise: over a tensor of exactly its leading entries and its value heads (`gg_vhead_len`), no two entries are read from the same place.
# harm: two rows or columns of the checkpoint would carry the same stored values and one would be lost
# catches: the key head left out of the stored place (every key head's values read from the first key head's place)
law gg_vhead_src_one_to_one:
  for +i: Nat
  for +j: Nat
  for +base: Nat
  for +hk: Nat
  for +r: Nat
  for +d: Nat
  {Bool.and(Bool.and(Nat.is_lt(i, gg_vhead_len(base, hk, r, d)), Nat.is_lt(j, gg_vhead_len(base, hk, r, d))), Bool.and(Nat.is_eq(gg_vhead_src(i, base, hk, r, d), gg_vhead_src(j, base, hk, r, d)), Nat.is_ne(i, j))) == False{} : Bool}

def gg_vhead_src_one_to_one(i, j, base, hk, r, d):
  gg_vi_go(i, j, base, hk, r, d, Nat.is_lt(i, gg_vhead_len(base, hk, r, d)), {==}, Nat.is_lt(j, gg_vhead_len(base, hk, r, d)), {==}, Nat.is_eq(gg_vhead_src(i, base, hk, r, d), gg_vhead_src(j, base, hk, r, d)), {==})

def gg_pad_aligns_aux(+x: Nat, +ap: Nat, +rem: Nat, erem1: {rem == Nat.mod(x, 1n+ap) : Nat}, erem2: {rem == Nat.mod(x, 1n+ap) : Nat}) -> {Nat.mod(Nat.add(x, Nat.mod(Nat.sub(1n+ap, rem), 1n+ap)), 1n+ap) == 0n : Nat}:
  match rem:
    case 0n:
      %Equal.sym(Nat, Nat.sub(1n+ap, 0n), 1n+ap, L.nat_sub_zero_r(1n+ap)) : {Nat.mod(Nat.add(x, Nat.mod(_, 1n+ap)), 1n+ap) == 0n : Nat}
      %L.nat_add_zero_r(1n+ap) : {Nat.mod(Nat.add(x, Nat.mod(_, 1n+ap)), 1n+ap) == 0n : Nat}
      %Equal.sym(Nat, Nat.mod(Nat.mul(1n, 1n+ap), 1n+ap), 0n, L.nat_mod_mul(1n, ap)) : {Nat.mod(Nat.add(x, _), 1n+ap) == 0n : Nat}
      %Equal.sym(Nat, Nat.add(x, 0n), x, L.nat_add_zero_r(x)) : {Nat.mod(_, 1n+ap) == 0n : Nat}
      Equal.sym(Nat, 0n, Nat.mod(x, 1n+ap), erem1)
    case 1n+rp:
      +h_lt = L.nat_lt_subst_l(Nat.mod(x, 1n+ap), 1n+rp, 1n+ap, Equal.sym(Nat, 1n+rp, Nat.mod(x, 1n+ap), erem1), L.nat_mod_lt(x, ap))
      +h_sub_lt = L.nat_le_lt_trans(Nat.sub(ap, rp), ap, 1n+ap, G.g_nat_sub_le_self(ap, rp), L.nat_lt_succ_self(ap))
      %Equal.sym(Nat, Nat.mod(Nat.sub(ap, rp), 1n+ap), Nat.sub(ap, rp), L.nat_mod_small(Nat.sub(ap, rp), ap, h_sub_lt)) : {Nat.mod(Nat.add(x, _), 1n+ap) == 0n : Nat}
      +sum_eq = G.g_nat_align_up_nz_sum(x, ap, 1n+rp, erem2, h_lt)
      %Equal.sym(Nat, Nat.add(x, Nat.sub(1n+ap, 1n+rp)), Nat.mul(1n+Nat.div(x, 1n+ap), 1n+ap), sum_eq) : {Nat.mod(_, 1n+ap) == 0n : Nat}
      L.nat_mod_mul(1n+Nat.div(x, 1n+ap), ap)

# promise: adding gg_pad(x, 1+ap) to x always lands on an exact multiple of 1+ap
# catches: gg_pad returns 1+ap unconditionally instead of alignment padding
law gg_pad_aligns:
  for +x: Nat
  for +ap: Nat
  {Nat.mod(Nat.add(x, gg_pad(x, 1n+ap)), 1n+ap) == 0n : Nat}

def gg_pad_aligns(x, ap):
  gg_pad_aligns_aux(x, ap, Nat.mod(x, 1n+ap), {==}, {==})

# promise: placing a tensor at a misaligned offset is always rejected by gg_place_ok
# catches: gg_place_ok drops the offset alignment check Nat.mod(off, a) == 0n
law gg_place_requires_aligned:
  for +q: Nat
  for +ap: Nat
  for len: Nat
  for avail: Nat
  {gg_place_ok(Nat.add(Nat.mul(q, 1n+1n+ap), 1n), len, 1n+1n+ap, avail) == False{} : Bool}

def gg_place_requires_aligned(q, ap, len, avail):
  %Equal.sym(Nat, Nat.mod(Nat.add(Nat.mul(q, 1n+1n+ap), 1n), 1n+1n+ap), 1n, L.nat_mod_split(q, 1n, 1n+ap, {==})) : {Bool.and(Nat.is_eq(_, 0n), gg_span_ok(Nat.add(Nat.mul(q, 1n+1n+ap), 1n), len, avail)) == False{} : Bool}
  {==}

# promise: a reversed row range (r0 > r1) is always rejected by gg_rows_ok
# catches: gg_rows_ok drops the Nat.is_le(r0, r1) ordering check
law gg_rows_requires_ordered:
  for +r1: Nat
  for +d: Nat
  for rb: Nat
  for nb: Nat
  {gg_rows_ok(Nat.add(r1, 1n+d), r1, rb, nb) == False{} : Bool}

def gg_rows_requires_ordered(r1, d, rb, nb):
  %Equal.sym(Bool, Nat.is_le(Nat.add(r1, 1n+d), r1), False{}, G.g_nat_le_add_succ_false(r1, d)) : {Bool.and(_, gg_count_fits(r1, rb, nb)) == False{} : Bool}
  {==}
