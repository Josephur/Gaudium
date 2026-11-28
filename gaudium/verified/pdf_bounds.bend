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
#   - the budgets of a document (the room one more decode may use, whether a charge fits, the
#     spend after a charge, counted or refused, and what a lexer spent since it started): the
#     bytes its streams decode to (filtered or not), the memory its parsed objects, CMaps and
#     width tables take, the text it yields, the form XObjects it draws, and the steps of work
#     the reader takes over its structure (every node, object, read, stage, element, entry and
#     operator it reaches, each time it reaches it), so that a zip bomb, overlapping objects, a
#     form drawing itself or structure shared many times over stop at a cap;
#   - the rows a PNG predictor walks, and whether that work fits the budget;
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

# The rows a PNG predictor walks over `len` bytes of data, in rows of a tag byte and `row`
# bytes (the last row may be short).
def pdf_predict_rows(+len: Nat, +row: Nat) -> Nat:
  Nat.div(Nat.add(len, row), 1n+row)

# The predictor's work over `len` bytes (its rows times `row` bytes each) fits in `room`.
# Checked before any work, so the work a stream asks for is tied to the budget, not to the
# row width it declares.
def pdf_predict_fits(+len: Nat, +row: Nat, +room: Nat) -> Bool:
  pdf_count_fits(pdf_predict_rows(len, row), row, room)

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

# The spend after a charge of `n` that fits (`fits`) or is refused.
def pdf_budget_settle(fits: Bool, +used: Nat, +n: Nat) -> Nat:
  match fits:
    case True{}: pdf_budget_take(used, n)
    case False{}: used

# What a budget has spent after a charge of `n` when `used` of `max` are spent: the charge is
# counted when it fits, and refused (the spend stays) when it does not. Every budget of a
# document is charged through this: its decoded bytes, its object memory, its text, its forms
# and its work (the steps the reader takes over the file's structure).
def pdf_budget_charge(+used: Nat, +n: Nat, +max: Nat) -> Nat:
  pdf_budget_settle(pdf_budget_fits(used, n, max), used, n)

# What a budget has spent between two of its spends, `from` (when a lexer started) and `used`.
def pdf_budget_spent(+from: Nat, +used: Nat) -> Nat:
  Nat.sub(used, from)

# ============================================================================
# 3. Limits
# ============================================================================

# A depth or a count `x` may grow by one more level or entry under the limit `max`
# (nesting, reference chains, the page tree, form XObjects, pages, an offset into a buffer).
def pdf_below(+x: Nat, +max: Nat) -> Bool:
  Nat.is_lt(x, max)

# A size `x` is within the limit `max` (the document's bytes, the heap its read holds).
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

# promise: n entries of a nonzero size are accepted whenever the available bytes hold them all.
# catches: only an empty count accepted (Nat.is_le(n, Nat.div(avail, elem)) replaced by Nat.is_eq(n, 0n))
law pdf_count_inside_accepted:
  for +n: Nat
  for +ep: Nat
  for +r: Nat
  {pdf_count_fits(n, 1n+ep, Nat.add(Nat.mul(n, 1n+ep), r)) == True{} : Bool}

def pdf_count_inside_accepted(n, ep, r):
  G.g_div_mul_add_ge(n, ep, r)

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

# promise: every object that starts inside the decoded data after the header is accepted.
# catches: only the first object accepted (Nat.is_lt(off, ...) replaced by Nat.is_eq(off, 0n))
law pdf_objstm_inside_accepted:
  for +f: Nat
  for +o: Nat
  for +r: Nat
  {pdf_objstm_ok(f, o, Nat.add(Nat.add(f, o), 1n+r)) == True{} : Bool}

def pdf_objstm_inside_accepted(f, o, r):
  %Equal.sym(Nat, Nat.add(Nat.add(f, o), 1n+r), Nat.add(f, Nat.add(o, 1n+r)), L.nat_add_assoc(f, o, 1n+r)) : {pdf_objstm_ok(f, o, _) == True{} : Bool}
  %Equal.sym(Nat, Nat.sub(Nat.add(f, Nat.add(o, 1n+r)), f), Nat.add(o, 1n+r), L.nat_add_sub_cancel_l(f, Nat.add(o, 1n+r))) : {Bool.and(Nat.is_le(f, Nat.add(f, Nat.add(o, 1n+r))), Nat.is_lt(o, _)) == True{} : Bool}
  %Equal.sym(Nat, Nat.add(o, 1n+r), 1n+Nat.add(o, r), L.nat_add_succ_r(o, r)) : {Bool.and(Nat.is_le(f, Nat.add(f, Nat.add(o, 1n+r))), Nat.is_lt(o, _)) == True{} : Bool}
  G.g_bool_and_tt(Nat.is_le(f, Nat.add(f, Nat.add(o, 1n+r))), Nat.is_lt(o, 1n+Nat.add(o, r)), L.nat_add_le_r(f, Nat.add(o, 1n+r)), Equal.trans(Bool, Nat.is_lt(o, 1n+Nat.add(o, r)), Nat.is_le(o, Nat.add(o, r)), True{}, L.nat_lt_succ_eq_le(o, Nat.add(o, r)), L.nat_add_le_r(o, r)))

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

# promise: field widths that are each within the limit are always accepted.
# catches: only zero widths accepted (Nat.is_le(a, max) replaced by Nat.is_eq(a, 0n))
law pdf_widths_inside_accepted:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for +r: Nat
  {pdf_widths_ok(a, b, c, Nat.add(Nat.max(a, Nat.max(b, c)), r)) == True{} : Bool}

def pdf_widths_inside_accepted(a, b, c, r):
  +m = Nat.add(Nat.max(a, Nat.max(b, c)), r)
  +top = L.nat_add_le_r(Nat.max(a, Nat.max(b, c)), r)
  +bc = L.nat_max_ge_r(a, Nat.max(b, c))
  +ha = L.nat_le_trans(a, Nat.max(a, Nat.max(b, c)), m, L.nat_max_ge_l(a, Nat.max(b, c)), top)
  +hb = L.nat_le_trans(b, Nat.max(b, c), m, L.nat_max_ge_l(b, c), L.nat_le_trans(Nat.max(b, c), Nat.max(a, Nat.max(b, c)), m, bc, top))
  +hc = L.nat_le_trans(c, Nat.max(b, c), m, L.nat_max_ge_r(b, c), L.nat_le_trans(Nat.max(b, c), Nat.max(a, Nat.max(b, c)), m, bc, top))
  G.g_bool_and_tt(Bool.and(Nat.is_le(a, m), Nat.is_le(b, m)), Nat.is_le(c, m), G.g_bool_and_tt(Nat.is_le(a, m), Nat.is_le(b, m), ha, hb), hc)

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

# promise: while the budget is not overspent, every decode no larger than the room it leaves is accepted.
# catches: the room computed one byte too large (Nat.sub(max, used) replaced by Nat.sub(1n+max, used))
law pdf_budget_room_fits:
  for +u: Nat
  for +d: Nat
  for +r: Nat
  {pdf_budget_fits(u, Nat.sub(pdf_budget_room(u, Nat.add(u, d)), r), Nat.add(u, d)) == True{} : Bool}

def pdf_budget_room_fits(u, d, r):
  %Equal.sym(Nat, Nat.sub(Nat.add(u, d), u), d, L.nat_add_sub_cancel_l(u, d)) : {pdf_budget_fits(u, Nat.sub(_, r), Nat.add(u, d)) == True{} : Bool}
  %Equal.sym(Nat, Nat.sub(Nat.add(u, d), u), d, L.nat_add_sub_cancel_l(u, d)) : {Bool.and(Nat.is_le(u, Nat.add(u, d)), Nat.is_le(Nat.sub(d, r), _)) == True{} : Bool}
  G.g_bool_and_tt(Nat.is_le(u, Nat.add(u, d)), Nat.is_le(Nat.sub(d, r), d), L.nat_add_le_r(u, d), G.g_nat_sub_le_self(d, r))

# promise: every charge that fits in the bytes the budget has left is accepted.
# catches: only a charge that fills the budget accepted (pdf_budget_fits replaced by Nat.is_eq(n, Nat.sub(max, used)))
law pdf_budget_inside_accepted:
  for +u: Nat
  for +n: Nat
  for +r: Nat
  {pdf_budget_fits(u, n, Nat.add(Nat.add(u, n), r)) == True{} : Bool}

def pdf_budget_inside_accepted(u, n, r):
  pdf_span_inside_accepted(u, n, r)

# promise: the room a budget leaves covers everything it has left.
# harm: a room smaller than what is left would refuse decodes the budget still holds (every compressed PDF, with a room of one byte).
# catches: a room of at most one byte (Nat.sub(max, used) replaced by Nat.min(1n, Nat.sub(max, used)))
law pdf_budget_room_covers:
  for +u: Nat
  for +d: Nat
  {Nat.is_le(d, pdf_budget_room(u, Nat.add(u, d))) == True{} : Bool}

def pdf_budget_room_covers(u, d):
  %Equal.sym(Nat, Nat.sub(Nat.add(u, d), u), d, L.nat_add_sub_cancel_l(u, d)) : {Nat.is_le(d, _) == True{} : Bool}
  L.nat_le_refl(d)

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

# promise: a depth or count below the limit may always grow by one more.
# catches: only the first level allowed (pdf_below replaced by Bool.and(Nat.is_eq(x, 0n), Nat.is_lt(x, max)))
law pdf_below_inside_accepted:
  for +x: Nat
  for +r: Nat
  {pdf_below(x, Nat.add(x, 1n+r)) == True{} : Bool}

def pdf_below_inside_accepted(x, r):
  %Equal.sym(Nat, Nat.add(x, 1n+r), 1n+Nat.add(x, r), L.nat_add_succ_r(x, r)) : {pdf_below(x, _) == True{} : Bool}
  Equal.trans(Bool, Nat.is_lt(x, 1n+Nat.add(x, r)), Nat.is_le(x, Nat.add(x, r)), True{}, L.nat_lt_succ_eq_le(x, Nat.add(x, r)), L.nat_add_le_r(x, r))

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

# promise: every size up to the limit is accepted.
# catches: only the limit itself accepted (Nat.is_le(x, max) replaced by Nat.is_eq(x, max))
law pdf_within_inside_accepted:
  for +x: Nat
  for +r: Nat
  {pdf_within(x, Nat.add(x, r)) == True{} : Bool}

def pdf_within_inside_accepted(x, r):
  L.nat_add_le_r(x, r)

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

# promise: the rows a PNG predictor walks cover every byte of its data.
# harm: bytes past the last row would be dropped from a decoded xref stream or content stream.
# catches: the rows rounded down (Nat.div(Nat.add(len, row), 1n+row) replaced by Nat.div(len, 1n+row))
law pdf_predict_rows_cover:
  for +l: Nat
  for +r: Nat
  {Nat.is_le(l, Nat.mul(pdf_predict_rows(l, r), 1n+r)) == True{} : Bool}

def pdf_predict_rows_cover(l, r):
  G.g_nat_ceil_div_covers(l, r)

# promise: a PNG predictor walks no row that starts past the end of its data.
# harm: an extra row would be walked (and charged) for bytes the stream does not have.
# catches: one row too many (pdf_predict_rows returning one more)
law pdf_predict_rows_tight:
  for +l: Nat
  for +r: Nat
  {Nat.is_le(Nat.mul(pdf_predict_rows(l, r), 1n+r), Nat.add(l, r)) == True{} : Bool}

def pdf_predict_rows_tight(l, r):
  L.nat_align_down_le(Nat.add(l, r), r)

# promise: the work a PNG predictor is allowed (its rows times the row width) never exceeds the room it was checked against.
# harm: a tiny stream declaring huge rows could make the reader walk megabytes per decode.
# catches: the row width ignored (pdf_count_fits(rows, row, room) replaced by pdf_count_fits(rows, 1n, room))
law pdf_predict_within:
  for +l: Nat
  for +r: Nat
  for +m: Nat
  {Bool.and(pdf_predict_fits(l, r, m), Nat.is_lt(m, Nat.mul(pdf_predict_rows(l, r), r))) == False{} : Bool}

def pdf_predict_within(l, r, m):
  pdf_count_within(pdf_predict_rows(l, r), r, m)

# promise: a predictor over rows of a nonzero width is accepted whenever the room holds its work.
# catches: the tag byte charged as row work (pdf_count_fits(rows, row, room) replaced by pdf_count_fits(rows, 1n+row, room))
law pdf_predict_inside_accepted:
  for +l: Nat
  for +rp: Nat
  for +x: Nat
  {pdf_predict_fits(l, 1n+rp, Nat.add(Nat.mul(pdf_predict_rows(l, 1n+rp), 1n+rp), x)) == True{} : Bool}

def pdf_predict_inside_accepted(l, rp, x):
  G.g_div_mul_add_ge(pdf_predict_rows(l, 1n+rp), rp, x)

def pdf_fits_not_over(+u: Nat, +n: Nat, +m: Nat, e: {pdf_budget_fits(u, n, m) == True{} : Bool}) -> {Nat.is_lt(m, pdf_budget_take(u, n)) == False{} : Bool}:
  %e : {Bool.and(_, Nat.is_lt(m, pdf_budget_take(u, n))) == False{} : Bool}
  pdf_budget_within(u, n, m)

def pdf_settle_le(+u: Nat, +d: Nat, +n: Nat, b: Bool, e: {pdf_budget_fits(u, n, Nat.add(u, d)) == b : Bool}) -> {Nat.is_le(pdf_budget_settle(b, u, n), Nat.add(u, d)) == True{} : Bool}:
  match b:
    case False{}:
      L.nat_add_le_r(u, d)
    case True{}:
      L.nat_not_lt_imp_le(Nat.add(u, d), pdf_budget_take(u, n), pdf_fits_not_over(u, n, Nat.add(u, d), e))

def pdf_settle_keep(+u: Nat, +n: Nat, b: Bool) -> {Bool.pick(Nat, b, u, pdf_budget_settle(b, u, n)) == u : Nat}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

def pdf_settle_take(+u: Nat, +n: Nat, b: Bool, e: {b == True{} : Bool}) -> {pdf_budget_settle(b, u, n) == pdf_budget_take(u, n) : Nat}:
  match b:
    case True{}:
      {==}
    case False{}:
      L.bool_absurd_tf({pdf_budget_settle(False{}, u, n) == pdf_budget_take(u, n) : Nat}, Equal.sym(Bool, False{}, True{}, e))

# promise: while a budget is not overspent, no charge, counted or refused, takes its spend past the budget.
# harm: hostile input could make the reader take more steps, or hold more memory, than configured.
# catches: a refused charge counted anyway (the refused arm of pdf_budget_settle returning pdf_budget_take(used, n))
law pdf_budget_charge_within:
  for +u: Nat
  for +d: Nat
  for +n: Nat
  {Nat.is_le(pdf_budget_charge(u, n, Nat.add(u, d)), Nat.add(u, d)) == True{} : Bool}

def pdf_budget_charge_within(u, d, n):
  pdf_settle_le(u, d, n, pdf_budget_fits(u, n, Nat.add(u, d)), {==})

# promise: a charge the budget refuses leaves its spend as it was.
# harm: a refused charge that filled or reset the budget would refuse the rest of a good document, or let a later charge past the cap.
# catches: a refused charge that fills the budget (pdf_budget_charge returning max when the charge does not fit)
law pdf_budget_charge_refused_keeps:
  for +u: Nat
  for +n: Nat
  for +m: Nat
  {Bool.pick(Nat, pdf_budget_fits(u, n, m), u, pdf_budget_charge(u, n, m)) == u : Nat}

def pdf_budget_charge_refused_keeps(u, n, m):
  pdf_settle_keep(u, n, pdf_budget_fits(u, n, m))

# promise: a charge that fits is counted in full.
# harm: a charge that fits but is not counted would let many small steps run past the cap.
# catches: an accepted charge not counted (the accepted arm of pdf_budget_settle returning used)
law pdf_budget_charge_counts_accepted:
  for +u: Nat
  for +n: Nat
  for +r: Nat
  {pdf_budget_charge(u, n, Nat.add(Nat.add(u, n), r)) == pdf_budget_take(u, n) : Nat}

def pdf_budget_charge_counts_accepted(u, n, r):
  pdf_settle_take(u, n, pdf_budget_fits(u, n, Nat.add(Nat.add(u, n), r)), pdf_budget_inside_accepted(u, n, r))

# promise: a charge never lowers what a budget has spent, a charge of nothing included.
# harm: a zero charge (an empty stream, an empty stage) that reset the spend would reopen a bomb split over streams.
# catches: a zero charge resetting the spend (pdf_budget_take returning 0n when n is 0)
law pdf_budget_take_keeps_spent:
  for +u: Nat
  for +n: Nat
  {Nat.is_le(u, pdf_budget_take(u, n)) == True{} : Bool}

def pdf_budget_take_keeps_spent(u, n):
  L.nat_add_le_r(u, n)

# promise: what a lexer charged since it started is exactly what its budget spent since then.
# harm: an undercount would let the objects of a parse go uncharged; an overcount would charge the earlier spend again and refuse good files.
# catches: the earlier spend counted again (Nat.sub(used, from) replaced by used)
law pdf_budget_spent_counts:
  for +f: Nat
  for +d: Nat
  {pdf_budget_spent(f, Nat.add(f, d)) == d : Nat}

def pdf_budget_spent_counts(f, d):
  L.nat_add_sub_cancel_l(f, d)
