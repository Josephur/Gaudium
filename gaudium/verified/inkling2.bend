# Inkling 2 (looped MoE decoder): the arithmetic every graph, plan and weight packer of the family
# shares. Specifies:
#   - the schedule of layer applications (a middle span of layers runs several times with shared weights)
#   - how many block states a layer's residual mixes read, and which layers open a block
#   - ring sizes of the sliding layers' K / V state and of the short convolutions' row history
#   - the geometry of the reversed, padded relative-position bias a softmax block loads
#   - the channel layout of the query-side projection and the byte geometry of 4-bit expert rows

import Base

# ============================================================================
# 1. Layer applications
# ============================================================================

# Layers of the looped span.
def ik_span(loop_start: Nat, loop_end: Nat) -> Nat:
  Nat.sub(loop_end, loop_start)

# Layer applications of one token: every layer once, the looped span `passes` times.
def ik_apps(n_layers: Nat, loop_start: Nat, loop_end: Nat, +passes: Nat) -> Nat:
  Nat.add(n_layers, Nat.mul(Nat.sub(passes, 1n), ik_span(loop_start, loop_end)))

def ik_app_layer_in_loop(+a: Nat, +loop_start: Nat, +span: Nat, +passes: Nat, in_loop: Bool) -> Nat:
  match in_loop:
    case True{}:
      Nat.add(loop_start, Nat.mod(Nat.sub(a, loop_start), span))
    case False{}:
      Nat.sub(a, Nat.mul(Nat.sub(passes, 1n), span))

def ik_app_layer_if(+a: Nat, +loop_start: Nat, +span: Nat, +passes: Nat, before: Bool) -> Nat:
  match before:
    case True{}:
      a
    case False{}:
      ik_app_layer_in_loop(a, loop_start, span, passes, Nat.is_lt(a, Nat.add(loop_start, Nat.mul(passes, span))))

# The layer whose weights application `a` runs: the layers before the span, the span `passes` times,
# the layers after it.
def ik_app_layer(+a: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat) -> Nat:
  ik_app_layer_if(a, loop_start, ik_span(loop_start, loop_end), passes, Nat.is_lt(a, loop_start))

def ik_app_pass_if(+a: Nat, +loop_start: Nat, +span: Nat, +passes: Nat, in_loop: Bool) -> Nat:
  match in_loop:
    case True{}:
      Nat.div(Nat.sub(a, loop_start), span)
    case False{}:
      0n

# The pass over the looped span application `a` belongs to (0: the first pass or a layer outside the span).
def ik_app_pass(+a: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat) -> Nat:
  ik_app_pass_if(
    a,
    loop_start,
    ik_span(loop_start, loop_end),
    passes,
    Bool.and(
      Nat.is_ge(a, loop_start),
      Nat.is_lt(a, Nat.add(loop_start, Nat.mul(passes, ik_span(loop_start, loop_end))))
    )
  )

# ============================================================================
# 2. Block states
# ============================================================================

# A layer opens a block when its index is a multiple of the block length: its input becomes a block state.
def ik_opens_block(layer: Nat, +block: Nat) -> Bool:
  Bool.and(Nat.is_gt(block, 0n), Nat.is_eq(Nat.mod(layer, block), 0n))

def ik_states_attn_if(+layer: Nat, +block: Nat, first: Bool) -> Nat:
  match first:
    case True{}:
      0n
    case False{}:
      Nat.add(Nat.div(Nat.sub(layer, 1n), block), 1n)

# Block states the attention-input mix of a layer reads (the first layer has no mix).
def ik_states_attn(+layer: Nat, +block: Nat) -> Nat:
  ik_states_attn_if(layer, block, Nat.is_eq(layer, 0n))

def ik_states_mlp_if(n: Nat, opens: Bool) -> Nat:
  match opens:
    case True{}:
      Nat.add(n, 1n)
    case False{}:
      n

# Block states the MLP-input mix of a layer reads: a layer that opens a block has added its input.
def ik_states_mlp(+layer: Nat, +block: Nat) -> Nat:
  ik_states_mlp_if(ik_states_attn(layer, block), ik_opens_block(layer, block))

# Block states a message of the previous pass is mixed from: the blocks the span opened, less its first.
def ik_msg_states(loop_start: Nat, loop_end: Nat, block: Nat) -> Nat:
  Nat.sub(Nat.div(ik_span(loop_start, loop_end), block), 1n)

# The geometry the kernels cover: at most `max_states` block states in any mix, a span that starts and
# ends on block boundaries behind the first layer, and at least one pass.
def ik_schedule_valid(+n_layers: Nat, +loop_start: Nat, +loop_end: Nat, +passes: Nat, +block: Nat, +max_states: Nat) -> Bool:
  Bool.and(
    Bool.and(Nat.is_gt(n_layers, 0n), Nat.is_gt(block, 0n)),
    Bool.and(
      Bool.and(Nat.is_ge(passes, 1n), Nat.is_le(ik_states_mlp(Nat.sub(n_layers, 1n), block), max_states)),
      Bool.and(
        Bool.and(Nat.is_gt(loop_start, 0n), Nat.is_le(loop_start, loop_end)),
        Bool.and(
          Nat.is_le(loop_end, n_layers),
          Bool.and(
            Bool.and(Nat.is_eq(Nat.mod(loop_start, block), 0n), Nat.is_eq(Nat.mod(loop_end, block), 0n)),
            Nat.is_le(Nat.add(ik_msg_states(loop_start, loop_end, block), 1n), max_states)
          )
        )
      )
    )
  )

# ============================================================================
# 3. Rings
# ============================================================================

def ik_pow2_stop(done: Bool, p: Nat) -> Nat:
  match done:
    case True{}:
      p
    case False{}:
      0n

def ik_pow2_from(fuel: Nat, done: Bool, +p: Nat, +need: Nat) -> Nat:
  match fuel:
    case 0n:
      ik_pow2_stop(done, p)
    case 1n+f:
      match done:
        case True{}:
          p
        case False{}:
          ik_pow2_from(f, Nat.is_ge(Nat.mul(p, 2n), need), Nat.mul(p, 2n), need)

# The smallest power of two of at least 128 that holds `need` rows; 0 when `need` is beyond 2^47 rows.
def ik_pow2_rows(+need: Nat) -> Nat:
  ik_pow2_from(40n, Nat.is_ge(128n, need), 128n, need)

# Slots of a sliding layer's K / V ring: the window beside the widest pass (a row must not see the rows
# the pass wrote ahead of it) and beside one vector of wrapped slots.
def ik_ring_need(window: Nat, rows: Nat) -> Nat:
  Nat.add(window, Nat.max(Nat.sub(rows, 1n), 64n))

def ik_ring_slots(window: Nat, rows: Nat) -> Nat:
  ik_pow2_rows(ik_ring_need(window, rows))

# Rows of a short convolution's history ring: the rows of the widest pass and the taps before its first row.
def ik_hist_rows(rows: Nat, taps: Nat) -> Nat:
  ik_pow2_rows(Nat.sub(Nat.add(rows, taps), 1n))

# ============================================================================
# 4. Relative-position bias table
# ============================================================================

# Channel of the reversed table that holds distance 0: a block of 64 keys whose first key is at distance
# `dd <= off` loads the 64 channels from `off - dd`; with `off = extent + 63` every key of a block that is
# further away is beyond the extent.
def ik_bias_off(extent: Nat) -> Nat:
  Nat.add(extent, 63n)

# Channels of one head's reversed, padded table.
def ik_bias_width(extent: Nat) -> Nat:
  Nat.add(ik_bias_off(extent), 65n)

def ik_bias_src_if(off: Nat, extent: Nat, +x: Nat, in_table: Bool) -> Nat:
  match in_table:
    case True{}:
      Nat.sub(off, x)
    case False{}:
      extent

# The distance whose bias channel `x` of the reversed table holds; `extent` (no bias: zero) for the padding.
def ik_bias_src(+extent: Nat, +x: Nat) -> Nat:
  ik_bias_src_if(
    ik_bias_off(extent),
    extent,
    x,
    Bool.and(Nat.is_le(x, ik_bias_off(extent)), Nat.is_lt(Nat.sub(ik_bias_off(extent), x), extent))
  )

# ============================================================================
# 5. Projection layout and 4-bit expert rows
# ============================================================================

# Whole blocks of 128 channels that hold `x` channels.
def ik_align128(x: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.add(x, 127n), 128n), 128n)

# Channels of the query-side projection: the query heads, the output gate, the relative states (16 per
# head), padded to whole blocks of 128.
def ik_gate_off(n_heads: Nat) -> Nat:
  Nat.mul(128n, n_heads)

def ik_rel_off(n_heads: Nat) -> Nat:
  Nat.mul(256n, n_heads)

def ik_qx_width(+n_heads: Nat, +d_rel: Nat) -> Nat:
  ik_align128(Nat.add(ik_rel_off(n_heads), Nat.mul(d_rel, n_heads)))

# Channels of the key / value projection: the heads, then (global layers) the per-row convolution taps.
def ik_kx_width(+n_kv: Nat, taps: Nat) -> Nat:
  ik_align128(Nat.add(Nat.mul(128n, n_kv), Nat.mul(taps, n_kv)))

# A row of `cols` 4-bit codes with one scale per 16: the row's byte length, 0 when `cols` is not a
# whole number of scale blocks.
def ik_nib_row_bytes_if(+cols: Nat, whole: Bool) -> Nat:
  match whole:
    case True{}:
      Nat.div(cols, 2n)
    case False{}:
      0n

def ik_nib_row_bytes(+cols: Nat) -> Nat:
  ik_nib_row_bytes_if(cols, Bool.and(Nat.is_gt(cols, 0n), Nat.is_eq(Nat.mod(cols, 16n), 0n)))

# Scales of such a row.
def ik_nib_row_scales(cols: Nat) -> Nat:
  Nat.div(cols, 16n)

# ============================================================================
# 6. Shape guard
# ============================================================================

# The shapes the family's kernels cover: whole blocks of 128 hidden channels, heads of 128 channels in
# groups of query heads per key / value head, eight heads' relative states of 16 per block, experts in
# blocks of 128 with at most 12 per row and a width of whole blocks, 1 to 16 convolution taps.
def ik_dims_valid(+hidden: Nat, +n_heads: Nat, +n_kv: Nat, head_dim: Nat, d_rel: Nat, +n_experts: Nat, +top_k: Nat, +expert_ff: Nat, +taps: Nat) -> Bool:
  Bool.and(
    Bool.and(Nat.is_gt(hidden, 0n), Nat.is_eq(Nat.mod(hidden, 128n), 0n)),
    Bool.and(
      Bool.and(Nat.is_eq(head_dim, 128n), Nat.is_eq(d_rel, 16n)),
      Bool.and(
        Bool.and(Nat.is_gt(n_kv, 0n), Nat.is_gt(n_heads, 0n)),
        Bool.and(
          Bool.and(Nat.is_eq(Nat.mod(n_heads, n_kv), 0n), Nat.is_eq(Nat.mod(n_heads, 8n), 0n)),
          Bool.and(
            Bool.and(Nat.is_gt(n_experts, 0n), Bool.and(Nat.is_eq(Nat.mod(n_experts, 128n), 0n), Nat.is_le(n_experts, 512n))),
            Bool.and(
              Bool.and(Nat.is_ge(top_k, 1n), Bool.and(Nat.is_le(top_k, 12n), Nat.is_le(top_k, n_experts))),
              Bool.and(
                Bool.and(Nat.is_gt(expert_ff, 0n), Nat.is_eq(Nat.mod(expert_ff, 128n), 0n)),
                Bool.and(Nat.is_ge(taps, 1n), Nat.is_le(taps, 16n))
              )
            )
          )
        )
      )
    )
  )

# ============================================================================
# 7. Prefill tiles
# ============================================================================

def ik_tile_rows_fit(+rows: Nat, +padded: Nat, fits: Bool) -> Nat:
  match fits:
    case True{}:
      padded
    case False{}:
      rows

# Rows a prefill tile of `rows` prompt rows runs with: a power of two of at least 8, capped at the
# planned tile, so that a new prompt length compiles no new recipe (the rows past the prompt are
# computed and never read); the rows themselves when the padded tile would not fit the `room`
# of positions left in the context or when no such count holds them.
def ik_tile_rows(+rows: Nat, +tile: Nat, +room: Nat) -> Nat:
  ik_tile_rows_fit(
    rows,
    Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile),
    Bool.and(
      Nat.is_ge(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), rows),
      Nat.is_le(Nat.min(ik_pow2_from(40n, Nat.is_ge(8n, rows), 8n, rows), tile), room)
    )
  )
