# PDF reader bounds: every offset, length, count, depth and byte budget the PDF text reader
# (`src/pdf/`) takes from a document. A PDF arrives in a request, so every number in it is
# client data.
#
# The Rust shell reads the bytes and the PDF syntax (a format defined by an outside
# specification); every decision about whether a number read from the file may be used to
# index, slice, allocate or recurse goes through a function here:
#   - a span of bytes lies inside a buffer (a stream's data from its /Length, the decoded
#     data of an object stream from its /First);
#   - an offset points inside a buffer (an xref offset, the startxref position);
#   - n entries of e bytes fit in the bytes left (xref table subsections, xref stream rows,
#     the offset pairs of an object stream), compared against a quotient so nothing overflows;
#   - the byte offset of row k of an xref stream;
#   - the place of an object inside an object stream (/First plus its offset);
#   - the field widths of an xref stream (/W);
#   - the decompressed-byte budget of a document (the room one more inflate may use, whether
#     a result fits, and the running total), so that a zip bomb stops at the cap;
#   - depth and count limits (nesting of arrays and dictionaries, chains of references, the
#     page tree, form XObjects, pages) and the document size limit.
# The limits themselves (bytes, pages, decompressed bytes, depths) are configuration and Rust
# constants, pinned by Rust tests.

import Base
import nat_lemmas.bend as L
import lemmas.bend as G

# ============================================================================
# 1. Spans, offsets and counts
# ============================================================================

# `len` bytes starting at `off` lie inside a buffer of `size` bytes.
def pdf_span_ok(+off: Nat, +len: Nat, +size: Nat) -> Bool:
  Bool.and(Nat.is_le(off, size), Nat.is_le(len, Nat.sub(size, off)))

# `n` entries of `elem` bytes each fit in `avail` bytes (a zero entry size is refused).
# Compared against a quotient, so a huge count read from a file cannot overflow.
def pdf_count_fits(+n: Nat, +elem: Nat, +avail: Nat) -> Bool:
  Bool.and(Nat.is_gt(elem, 0n), Nat.is_le(n, Nat.div(avail, elem)))

# Byte offset of row `k` of rows of `w` bytes (callers first check that all rows fit, so this
# never overflows for a row inside them).
def pdf_row_offset(k: Nat, w: Nat) -> Nat:
  Nat.mul(k, w)

# An object at offset `off` after the header (`first` bytes) of an object stream whose decoded
# data is `dlen` bytes long starts inside that data.
def pdf_objstm_ok(+first: Nat, +off: Nat, +dlen: Nat) -> Bool:
  Bool.and(Nat.is_le(first, dlen), Nat.is_lt(off, Nat.sub(dlen, first)))

# Where that object starts in the decoded data.
def pdf_objstm_pos(first: Nat, off: Nat) -> Nat:
  Nat.add(first, off)

# The three field widths of an xref stream (/W) are each at most `max` bytes.
def pdf_widths_ok(+a: Nat, +b: Nat, +c: Nat, +max: Nat) -> Bool:
  Bool.and(Bool.and(Nat.is_le(a, max), Nat.is_le(b, max)), Nat.is_le(c, max))

# ============================================================================
# 2. Decompressed-byte budget
# ============================================================================

# The bytes one more decode may produce when `used` of `max` are spent.
def pdf_budget_room(+used: Nat, +max: Nat) -> Nat:
  Nat.sub(max, used)

# A decode of `n` bytes fits when `used` of `max` are spent.
def pdf_budget_fits(+used: Nat, +n: Nat, +max: Nat) -> Bool:
  pdf_span_ok(used, n, max)

# The bytes spent after a decode of `n` bytes.
def pdf_budget_take(used: Nat, n: Nat) -> Nat:
  Nat.add(used, n)

# ============================================================================
# 3. Limits
# ============================================================================

# A depth or a count `x` may grow by one more level or entry under the limit `max`
# (nesting, reference chains, the page tree, form XObjects, pages, an offset into a buffer).
def pdf_below(+x: Nat, +max: Nat) -> Bool:
  Nat.is_lt(x, max)

# A size `x` is within the limit `max` (the document's bytes).
def pdf_within(+x: Nat, +max: Nat) -> Bool:
  Nat.is_le(x, max)

# ============================================================================
# 4. Helper lemmas (erased before Rust codegen)
# ============================================================================

def pdf_row_fit(+n: Nat, +wp: Nat, +a: Nat, +k: Nat, b1: Bool, e1: {Nat.is_le(n, Nat.div(a, 1n+wp)) == b1 : Bool}, b2: Bool, e2: {Nat.is_lt(k, n) == b2 : Bool}) -> {Bool.and(Bool.and(b1, b2), Nat.is_lt(a, Nat.add(1n+wp, Nat.mul(k, 1n+wp)))) == False{} : Bool}:
  match b1:
    case False{}:
      {==}
    case True{}:
      match b2:
        case False{}:
          {==}
        case True{}:
          +hn = L.nat_le_trans(Nat.mul(n, 1n+wp), Nat.mul(Nat.div(a, 1n+wp), 1n+wp), a, L.nat_mul_le_mono_r(n, Nat.div(a, 1n+wp), 1n+wp, e1), L.nat_align_down_le(a, wp))
          +hk = L.nat_mul_le_mono_r(1n+k, n, 1n+wp, G.g_nat_lt_succ_le(k, n, e2))
          L.nat_le_lt_false(a, Nat.mul(1n+k, 1n+wp), L.nat_le_trans(Nat.mul(1n+k, 1n+wp), Nat.mul(n, 1n+wp), a, hk, hn))

def pdf_objstm_fit(+f: Nat, +o: Nat, +d: Nat, b1: Bool, e1: {Nat.is_le(f, d) == b1 : Bool}, b2: Bool, e2: {Nat.is_lt(o, Nat.sub(d, f)) == b2 : Bool}) -> {Bool.and(Bool.and(b1, b2), Nat.is_le(d, Nat.add(f, o))) == False{} : Bool}:
  match b1:
    case False{}:
      {==}
    case True{}:
      match b2:
        case False{}:
          {==}
        case True{}:
          +eq = Equal.trans(Nat, Nat.add(f, Nat.sub(d, f)), Nat.add(Nat.sub(d, f), f), d, L.nat_add_comm(f, Nat.sub(d, f)), L.nat_sub_add_cancel(d, f, e1))
          L.nat_lt_le_false(Nat.add(f, o), d, L.nat_lt_subst_r(Nat.add(f, o), Nat.add(f, Nat.sub(d, f)), d, eq, L.nat_add_lt_l(f, o, Nat.sub(d, f), e2)))

def pdf_widths_fit(+a: Nat, +b: Nat, +c: Nat, +m: Nat, x: Bool, ex: {Nat.is_le(a, m) == x : Bool}, y: Bool, ey: {Nat.is_le(b, m) == y : Bool}, z: Bool, ez: {Nat.is_le(c, m) == z : Bool}) -> {Bool.and(Bool.and(Bool.and(x, y), z), Bool.or(Bool.or(Nat.is_lt(m, a), Nat.is_lt(m, b)), Nat.is_lt(m, c))) == False{} : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      match y:
        case False{}:
          {==}
        case True{}:
          match z:
            case False{}:
              {==}
            case True{}:
              %Equal.sym(Bool, Nat.is_lt(m, a), False{}, L.nat_le_lt_false(m, a, ex)) : {Bool.or(Bool.or(_, Nat.is_lt(m, b)), Nat.is_lt(m, c)) == False{} : Bool}
              %Equal.sym(Bool, Nat.is_lt(m, b), False{}, L.nat_le_lt_false(m, b, ey)) : {Bool.or(Bool.or(False{}, _), Nat.is_lt(m, c)) == False{} : Bool}
              L.nat_le_lt_false(m, c, ez)

# ============================================================================
# 5. Laws
# ============================================================================

# promise: a span of a buffer the reader accepts never ends past the end of that buffer.
# harm: a stream /Length or an object-stream header from the file could make the reader slice past the buffer.
# catches: the offset ignored in the length check (Nat.sub(size, off) replaced by size)
law pdf_span_within:
  for +o: Nat
  for +l: Nat
  for +s: Nat
  {Bool.and(pdf_span_ok(o, l, s), Nat.is_lt(s, Nat.add(o, l))) == False{} : Bool}

def pdf_span_within(o, l, s):
  G.g_span_fit(o, l, s, Nat.is_le(o, s), {==}, Nat.is_le(l, Nat.sub(s, o)), {==})

# promise: every span that ends inside a buffer is accepted.
# catches: the length compared with a strict less-than (Nat.is_le replaced by Nat.is_lt), refusing a stream that ends at the last byte
law pdf_span_inside_accepted:
  for +o: Nat
  for +l: Nat
  for +r: Nat
  {pdf_span_ok(o, l, Nat.add(Nat.add(o, l), r)) == True{} : Bool}

def pdf_span_inside_accepted(o, l, r):
  %Equal.sym(Nat, Nat.add(Nat.add(o, l), r), Nat.add(o, Nat.add(l, r)), L.nat_add_assoc(o, l, r)) : {pdf_span_ok(o, l, _) == True{} : Bool}
  %Equal.sym(Nat, Nat.sub(Nat.add(o, Nat.add(l, r)), o), Nat.add(l, r), L.nat_add_sub_cancel_l(o, Nat.add(l, r))) : {Bool.and(Nat.is_le(o, Nat.add(o, Nat.add(l, r))), Nat.is_le(l, _)) == True{} : Bool}
  G.g_bool_and_tt(Nat.is_le(o, Nat.add(o, Nat.add(l, r))), Nat.is_le(l, Nat.add(l, r)), L.nat_add_le_r(o, Nat.add(l, r)), L.nat_add_le_r(l, r))

# promise: n entries of e bytes that the count check accepts never need more than the available bytes.
# harm: a hostile entry count could make the reader allocate or walk past the decoded data.
# catches: the entry size ignored (Nat.is_le(n, Nat.div(avail, elem)) replaced by Nat.is_le(n, avail))
law pdf_count_within:
  for +n: Nat
  for +e: Nat
  for +a: Nat
  {Bool.and(pdf_count_fits(n, e, a), Nat.is_lt(a, Nat.mul(n, e))) == False{} : Bool}

def pdf_count_within(n, e, a):
  match e:
    case 0n:
      {==}
    case 1n+ep:
      G.g_fit_within(n, ep, a, Nat.is_le(n, Nat.div(a, 1n+ep)), {==})

# promise: no entries at all always fit, whatever their size.
# catches: a constant refusal (pdf_count_fits returning False)
law pdf_count_none_fits:
  for +ep: Nat
  for +a: Nat
  {pdf_count_fits(0n, 1n+ep, a) == True{} : Bool}

def pdf_count_none_fits(ep, a):
  L.nat_le_zero(Nat.div(a, 1n+ep))

# promise: every row below the accepted row count ends inside the available bytes.
# harm: an xref stream row could be read past the end of the decoded stream.
# catches: the row offset off by one row (Nat.mul(k, w) replaced by Nat.mul(1n+k, w))
law pdf_row_within:
  for +n: Nat
  for +w: Nat
  for +a: Nat
  for +k: Nat
  {Bool.and(Bool.and(pdf_count_fits(n, w, a), Nat.is_lt(k, n)), Nat.is_lt(a, Nat.add(w, pdf_row_offset(k, w)))) == False{} : Bool}

def pdf_row_within(n, w, a, k):
  match w:
    case 0n:
      {==}
    case 1n+wp:
      pdf_row_fit(n, wp, a, k, Nat.is_le(n, Nat.div(a, 1n+wp)), {==}, Nat.is_lt(k, n), {==})

# promise: an object the reader takes from an object stream always starts inside the stream's decoded data.
# harm: a hostile /First or object offset could index past the decoded data.
# catches: the header length ignored (Nat.sub(dlen, first) replaced by dlen)
law pdf_objstm_within:
  for +f: Nat
  for +o: Nat
  for +d: Nat
  {Bool.and(pdf_objstm_ok(f, o, d), Nat.is_le(d, pdf_objstm_pos(f, o))) == False{} : Bool}

def pdf_objstm_within(f, o, d):
  pdf_objstm_fit(f, o, d, Nat.is_le(f, d), {==}, Nat.is_lt(o, Nat.sub(d, f)), {==})

# promise: the first object right after an empty header of a nonempty object stream is accepted.
# catches: a constant refusal (pdf_objstm_ok returning False)
law pdf_objstm_first_accepted:
  for +d: Nat
  {pdf_objstm_ok(0n, 0n, 1n+d) == True{} : Bool}

def pdf_objstm_first_accepted(d):
  {==}

# promise: xref field widths the reader accepts are never wider than the limit they were checked against.
# harm: a wider field would overflow the integer the reader assembles it into.
# catches: one width left unchecked (Nat.is_le(b, max) dropped)
law pdf_widths_within:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for +m: Nat
  {Bool.and(pdf_widths_ok(a, b, c, m), Bool.or(Bool.or(Nat.is_lt(m, a), Nat.is_lt(m, b)), Nat.is_lt(m, c))) == False{} : Bool}

def pdf_widths_within(a, b, c, m):
  pdf_widths_fit(a, b, c, m, Nat.is_le(a, m), {==}, Nat.is_le(b, m), {==}, Nat.is_le(c, m), {==})

# promise: fields of zero width (absent fields, which take their defaults) are always accepted.
# catches: a width of zero refused (Nat.is_le(a, max) replaced by Nat.is_lt(0n, a))
law pdf_widths_zero_accepted:
  for +m: Nat
  {pdf_widths_ok(0n, 0n, 0n, m) == True{} : Bool}

def pdf_widths_zero_accepted(m):
  %Equal.sym(Bool, Nat.is_le(0n, m), True{}, L.nat_le_zero(m)) : {Bool.and(Bool.and(_, _), _) == True{} : Bool}
  {==}

# promise: a decode the budget accepts never takes the bytes spent past the budget.
# harm: a zip bomb could make the reader hold more decompressed bytes than configured.
# catches: the bytes already spent ignored (pdf_budget_fits checking n against max alone)
law pdf_budget_within:
  for +u: Nat
  for +n: Nat
  for +m: Nat
  {Bool.and(pdf_budget_fits(u, n, m), Nat.is_lt(m, pdf_budget_take(u, n))) == False{} : Bool}

def pdf_budget_within(u, n, m):
  pdf_span_within(u, n, m)

# promise: a decode that fills exactly the room the budget leaves is accepted.
# catches: the room computed one byte too large (Nat.sub(max, used) replaced by Nat.sub(1n+max, used))
law pdf_budget_room_fits:
  for +u: Nat
  for +d: Nat
  {pdf_budget_fits(u, pdf_budget_room(u, Nat.add(u, d)), Nat.add(u, d)) == True{} : Bool}

def pdf_budget_room_fits(u, d):
  G.g_bool_and_tt(Nat.is_le(u, Nat.add(u, d)), Nat.is_le(Nat.sub(Nat.add(u, d), u), Nat.sub(Nat.add(u, d), u)), L.nat_add_le_r(u, d), L.nat_le_refl(Nat.sub(Nat.add(u, d), u)))

# promise: while the budget is not spent, the next decode gets some room.
# catches: a constant zero room (pdf_budget_room returning 0n)
law pdf_budget_room_positive:
  for +u: Nat
  for +d: Nat
  {Nat.is_gt(pdf_budget_room(u, Nat.add(u, 1n+d)), 0n) == True{} : Bool}

def pdf_budget_room_positive(u, d):
  %Equal.sym(Nat, Nat.sub(Nat.add(u, 1n+d), u), 1n+d, L.nat_add_sub_cancel_l(u, 1n+d)) : {Nat.is_gt(_, 0n) == True{} : Bool}
  {==}

# promise: a depth or count the limit check lets grow is always below the limit.
# harm: hostile nesting could recurse past the stack, or a page tree could yield more pages than configured.
# catches: the limit itself allowed (Nat.is_lt replaced by Nat.is_le)
law pdf_below_within:
  for +x: Nat
  for +m: Nat
  {Bool.and(pdf_below(x, m), Nat.is_le(m, x)) == False{} : Bool}

def pdf_below_within(x, m):
  match x m:
    case 0n 0n:
      {==}
    case 0n 1n+mp:
      {==}
    case 1n+xp 0n:
      {==}
    case 1n+xp 1n+mp:
      pdf_below_within(xp, mp)

# promise: the first level or entry is allowed under any positive limit.
# catches: a constant refusal (pdf_below returning False)
law pdf_below_zero_accepted:
  for +m: Nat
  {pdf_below(0n, 1n+m) == True{} : Bool}

def pdf_below_zero_accepted(m):
  {==}

# promise: a size the limit check accepts never exceeds the limit.
# harm: a document larger than configured could be read into memory.
# catches: the comparison turned around (Nat.is_le(x, max) replaced by Nat.is_ge(x, max))
law pdf_within_limit:
  for +x: Nat
  for +m: Nat
  {Bool.and(pdf_within(x, m), Nat.is_lt(m, x)) == False{} : Bool}

def pdf_within_limit(x, m):
  match x m:
    case 0n 0n:
      {==}
    case 0n 1n+mp:
      {==}
    case 1n+xp 0n:
      {==}
    case 1n+xp 1n+mp:
      pdf_within_limit(xp, mp)

# promise: a size equal to the limit is accepted.
# catches: the limit itself refused (Nat.is_le replaced by Nat.is_lt)
law pdf_within_at_limit:
  for +m: Nat
  {pdf_within(m, m) == True{} : Bool}

def pdf_within_at_limit(m):
  L.nat_le_refl(m)

# promise: a decode adds exactly its bytes to the bytes spent, no fewer.
# harm: an undercounted spend would let many streams together decode past the budget (a zip bomb split over streams).
# catches: the spend not counted (pdf_budget_take returning used)
law pdf_budget_take_counts_all:
  for +u: Nat
  for +n: Nat
  {Nat.sub(pdf_budget_take(u, n), u) == n : Nat}

def pdf_budget_take_counts_all(u, n):
  L.nat_add_sub_cancel_l(u, n)

# promise: the first row of an xref stream starts at its first byte (the file format's layout).
# catches: rows counted from one (Nat.mul(k, w) replaced by Nat.mul(1n+k, w))
law pdf_row_first_at_start:
  for +w: Nat
  {pdf_row_offset(0n, w) == 0n : Nat}

def pdf_row_first_at_start(w):
  {==}

# promise: each row of an xref stream starts right after the row before it (the file format's layout).
# catches: rows one byte apart (pdf_row_offset returning k)
law pdf_row_next_adjacent:
  for +k: Nat
  for +w: Nat
  {pdf_row_offset(1n+k, w) == Nat.add(w, pdf_row_offset(k, w)) : Nat}

def pdf_row_next_adjacent(k, w):
  {==}

# promise: the object at offset zero of an object stream starts right after its header (the file format's layout).
# catches: the header length ignored (pdf_objstm_pos returning off)
law pdf_objstm_pos_first:
  for +f: Nat
  {pdf_objstm_pos(f, 0n) == f : Nat}

def pdf_objstm_pos_first(f):
  L.nat_add_zero_r(f)

# promise: an object's place in an object stream moves byte for byte with its offset (the file format's layout).
# catches: the offset ignored (pdf_objstm_pos returning first)
law pdf_objstm_pos_step:
  for +f: Nat
  for +o: Nat
  {pdf_objstm_pos(f, 1n+o) == 1n+pdf_objstm_pos(f, o) : Nat}

def pdf_objstm_pos_step(f, o):
  L.nat_add_succ_r(f, o)
