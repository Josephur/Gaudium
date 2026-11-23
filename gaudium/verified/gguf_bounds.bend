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
#   - small limits (array nesting, tensor rank) and the shard order of a split model.
# The type table (block weights and block bytes per ggml type) and the value-type sizes are
# format facts of an outside specification; they live in Rust with tests.

import Base
import nat_lemmas.bend as L

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

# A sum of counts (tensors over the shards of a split model).
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
def gg_vhead_tail(+j: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat) -> Nat:
  Nat.add(base, Nat.add(Nat.add(Nat.mul(Nat.div(Nat.mod(j, Nat.mul(r, d)), d), Nat.mul(hk, d)), Nat.mul(Nat.div(j, Nat.mul(r, d)), d)), Nat.mod(Nat.mod(j, Nat.mul(r, d)), d)))

def gg_vhead_pick(+i: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat, lead: Bool) -> Nat:
  match lead:
    case True{}:
      i
    case False{}:
      gg_vhead_tail(Nat.sub(i, base), base, hk, r, d)

# rusty-bends: total
def gg_vhead_src(+i: Nat, +base: Nat, +hk: Nat, +r: Nat, +d: Nat) -> Nat:
  gg_vhead_pick(i, base, hk, r, d, Nat.is_lt(i, base))

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
