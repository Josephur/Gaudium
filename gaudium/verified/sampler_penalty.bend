# Verified Sampler Penalty Core (presence / frequency / repetition penalties, logit_bias)
# Formally specifies and proves the decisions the engine makes around a penalised request:
#   - Which token ids a request may name, and which ids a penalty may index (inside the vocabulary)
#   - Whether a request's bans leave the model anything to emit
#   - Whether a decode path may take a penalised request (only a path that applies the penalties
#     at every position: a path that picks tokens without them would leave the request's own
#     reference stream)
#   - Whether a picked token may be emitted (never a banned one, never one outside the vocabulary)
#   - Whether the per-token counts a position is penalised with are the counts of exactly the
#     generated tokens before that position (none of the prompt, none of the position itself, none
#     dropped)
#   - How the ledger changes when a token is emitted (that token's count up by one, every other
#     token's count unchanged, the total up by one)
#   - How many token ids a request may emit (inside the logits row, and inside the tokenizer:
#     a padded row's padding ids are not tokens)
# The penalty arithmetic itself (BF16 logits, f32 math, rounding) is float work and stays in the
# Rust shell (`gaudium/src/sampler_penalty.rs`) with Rust tests.

import Base
import nat_lemmas.bend as L

# ============================================================================
# 1. Helper lemmas (erased before Rust codegen)
# ============================================================================

def pen_lt_add_false(a: Nat, extra: Nat) -> {Nat.is_lt(Nat.add(a, extra), a) == False{} : Bool}:
  match a:
    case 0n:
      match extra:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+p:
      pen_lt_add_false(p, extra)

def pen_lt_succ_add(+p: Nat, e: Nat) -> {Nat.is_lt(p, 1n+Nat.add(p, e)) == True{} : Bool}:
  match p:
    case 0n: {==}
    case 1n+q:
      pen_lt_succ_add(q, e)

def pen_eq_refl(+t: Nat) -> {Nat.is_eq(t, t) == True{} : Bool}:
  match t:
    case 0n: {==}
    case 1n+q:
      pen_eq_refl(q)

def pen_eq_succ_add_false(+t: Nat, d: Nat) -> {Nat.is_eq(1n+Nat.add(t, d), t) == False{} : Bool}:
  match t:
    case 0n: {==}
    case 1n+q:
      pen_eq_succ_add_false(q, d)

def pen_eq_add_succ_false(+t: Nat, d: Nat) -> {Nat.is_eq(t, 1n+Nat.add(t, d)) == False{} : Bool}:
  match t:
    case 0n: {==}
    case 1n+q:
      pen_eq_add_succ_false(q, d)

# ============================================================================
# 2. Token Ids and Bans
# ============================================================================

# What a request's logit_bias says about one token: nothing, an additive bias, or a ban.
type PenBias is Data:
  PbNone{}
  PbAdd{}
  PbBan{}

# A token id names a row of the vocabulary (a logit_bias key, a generated token a count indexes).
def pen_token_in_vocab(id: Nat, vocab: Nat) -> Bool:
  Nat.is_lt(id, vocab)

# A request's bans leave the model at least one token to emit (`banned` = distinct banned ids,
# every one inside the vocabulary).
def pen_bans_leave_tokens(banned: Nat, vocab: Nat) -> Bool:
  Nat.is_lt(banned, vocab)

# ============================================================================
# 3. Decode Paths and Emission
# ============================================================================

# penalised:    the request carries penalties or a logit_bias
# path_applies: the decode path applies them at every position it picks a token for
def pen_path_allowed(penalised: Bool, path_applies: Bool) -> Bool:
  Bool.or(Bool.not(penalised), path_applies)

# A token the penalised pick returned may be emitted.
def pen_emit_ok(tok: Nat, vocab: Nat, bias: PenBias) -> Bool:
  match bias:
    case PbNone{}:
      pen_token_in_vocab(tok, vocab)
    case PbAdd{}:
      pen_token_in_vocab(tok, vocab)
    case PbBan{}:
      False{}

# ============================================================================
# 4. Count Ledger
# ============================================================================

# total:      tokens the request's count ledger holds
# seq_index:  sequence index of the token about to be picked
# prompt_len: prompt tokens (the first generated token has sequence index prompt_len)
# The ledger may penalise the position only when it holds exactly the generated tokens before it.
def pen_counts_match(total: Nat, +seq_index: Nat, +prompt_len: Nat) -> Bool:
  Bool.and(Nat.is_le(prompt_len, seq_index), Nat.is_eq(total, Nat.sub(seq_index, prompt_len)))

# One token's count after an emission: raised when the emitted token is that token.
def pen_count_bump(count: Nat, hit: Bool) -> Nat:
  match hit:
    case True{}:
      1n+count
    case False{}:
      count

# The count token `id` holds after token `tok` is emitted, where `id` held `count` before.
# The shell's ledger stores only the tokens emitted so far; for every other id the result is its
# old count (the laws below), which is why the shell updates only the entry of `tok`.
def pen_count_add(count: Nat, +id: Nat, +tok: Nat) -> Nat:
  pen_count_bump(count, Nat.is_eq(id, tok))

# The ledger's total after one more emitted token.
def pen_total_add(total: Nat) -> Nat:
  1n+total

# row:    width of the logits row the engine reads out (it may be padded)
# tokens: token ids the tokenizer defines
# The token ids a request may name, ban and emit.
def pen_emit_vocab(row: Nat, tokens: Nat) -> Nat:
  Nat.min(row, tokens)

# ============================================================================
# 5. Laws
# ============================================================================

# promise: a token id at or past the vocabulary size is never accepted as a token id
# harm: a logit_bias key or a count past the vocabulary would index outside the logits row
# catches: pen_token_in_vocab uses Nat.is_le (the id equal to the vocabulary size accepted)
law pen_token_past_vocab_refused:
  for v: Nat
  for e: Nat
  {pen_token_in_vocab(Nat.add(v, e), v) == False{} : Bool}

def pen_token_past_vocab_refused(v, e):
  pen_lt_add_false(v, e)

# promise: every token id below the vocabulary size is accepted
# catches: pen_token_in_vocab returns False{} (no logit_bias key is ever accepted)
law pen_token_in_vocab_accepted:
  for +i: Nat
  for e: Nat
  {pen_token_in_vocab(i, 1n+Nat.add(i, e)) == True{} : Bool}

def pen_token_in_vocab_accepted(i, e):
  pen_lt_succ_add(i, e)

# promise: bans that cover the whole vocabulary are refused
# harm: the request could only ever emit a banned token
# catches: pen_bans_leave_tokens uses Nat.is_le (banning every token accepted)
law pen_bans_covering_vocab_refused:
  for v: Nat
  for e: Nat
  {pen_bans_leave_tokens(Nat.add(v, e), v) == False{} : Bool}

def pen_bans_covering_vocab_refused(v, e):
  pen_lt_add_false(v, e)

# promise: a request that bans fewer tokens than the vocabulary holds is accepted
# catches: pen_bans_leave_tokens returns False{} (every request with a ban refused)
law pen_bans_leaving_tokens_accepted:
  for +b: Nat
  for e: Nat
  {pen_bans_leave_tokens(b, 1n+Nat.add(b, e)) == True{} : Bool}

def pen_bans_leaving_tokens_accepted(b, e):
  pen_lt_succ_add(b, e)

# promise: a penalised request never takes a decode path that picks tokens without its penalties
# harm: the request's tokens would leave its own reference stream (AGENTS.md section 7)
# catches: pen_path_allowed returns Bool.or(penalised, path_applies) (the Bool.not dropped)
law pen_unapplied_path_refused:
  {pen_path_allowed(True{}, False{}) == False{} : Bool}

def pen_unapplied_path_refused():
  {==}

# promise: a request without penalties may take every decode path
# catches: pen_path_allowed returns Bool.and(Bool.not(penalised), path_applies)
law pen_unpenalised_path_allowed:
  for a: Bool
  {pen_path_allowed(False{}, a) == True{} : Bool}

def pen_unpenalised_path_allowed(a):
  match a:
    case True{}: {==}
    case False{}: {==}

# promise: a penalised request may take a decode path that applies its penalties
# catches: pen_path_allowed returns Bool.not(penalised) (penalised requests refused everywhere)
law pen_applied_path_allowed:
  {pen_path_allowed(True{}, True{}) == True{} : Bool}

def pen_applied_path_allowed():
  {==}

# promise: a token the request bans is never emitted, whatever its id
# harm: a logit_bias of -100 would not ban the token
# catches: the PbBan arm of pen_emit_ok returns pen_token_in_vocab(tok, vocab)
law pen_banned_never_emitted:
  for t: Nat
  for v: Nat
  {pen_emit_ok(t, v, PbBan{}) == False{} : Bool}

def pen_banned_never_emitted(t, v):
  {==}

# promise: a token id at or past the vocabulary size is never emitted, biased or not
# catches: the PbAdd arm of pen_emit_ok returns True{}
law pen_emit_past_vocab_refused:
  for v: Nat
  for e: Nat
  for b: PenBias
  {pen_emit_ok(Nat.add(v, e), v, b) == False{} : Bool}

def pen_emit_past_vocab_refused(v, e, b):
  match b:
    case PbNone{}:
      pen_lt_add_false(v, e)
    case PbAdd{}:
      pen_lt_add_false(v, e)
    case PbBan{}:
      {==}

# promise: an unbanned token inside the vocabulary is emitted, with or without a bias
# catches: the PbAdd arm of pen_emit_ok returns False{} (a biased token never emitted)
law pen_unbanned_token_emitted:
  for +t: Nat
  for +e: Nat
  {Bool.and(pen_emit_ok(t, 1n+Nat.add(t, e), PbNone{}), pen_emit_ok(t, 1n+Nat.add(t, e), PbAdd{})) == True{} : Bool}

def pen_unbanned_token_emitted(t, e):
  match t:
    case 0n: {==}
    case 1n+q:
      pen_unbanned_token_emitted(q, e)

# promise: a ledger holding exactly the generated tokens before a position may penalise it
# catches: pen_counts_match compares total with seq_index (the prompt counted as generated)
law pen_counts_match_exact:
  for +t: Nat
  for +p: Nat
  {pen_counts_match(t, Nat.add(p, t), p) == True{} : Bool}

def pen_counts_match_exact(t, p):
  match t p:
    case 0n 0n: {==}
    case 1n+r 0n:
      pen_eq_refl(r)
    case 0n 1n+q:
      pen_counts_match_exact(0n, q)
    case 1n+r 1n+q:
      pen_counts_match_exact(1n+r, q)

# promise: a ledger holding more tokens than were generated before a position never penalises it
# harm: the position would be penalised for a token it has not seen (its own, or a rejected one)
# catches: pen_counts_match compares with Nat.is_ge in place of Nat.is_eq (an over-full ledger passes)
law pen_counts_extra_refused:
  for +t: Nat
  for +d: Nat
  for +p: Nat
  {pen_counts_match(1n+Nat.add(t, d), Nat.add(p, t), p) == False{} : Bool}

def pen_counts_extra_refused(t, d, p):
  match t p:
    case 0n 0n: {==}
    case 1n+r 0n:
      pen_eq_succ_add_false(r, d)
    case 0n 1n+q:
      pen_counts_extra_refused(0n, d, q)
    case 1n+r 1n+q:
      pen_counts_extra_refused(1n+r, d, q)

# promise: a ledger holding fewer tokens than were generated before a position never penalises it
# harm: a generated token would escape its penalty
# catches: pen_counts_match compares with Nat.is_le in place of Nat.is_eq (a short ledger passes)
law pen_counts_short_refused:
  for +t: Nat
  for +d: Nat
  for +p: Nat
  {pen_counts_match(t, 1n+Nat.add(Nat.add(p, t), d), p) == False{} : Bool}

def pen_counts_short_refused(t, d, p):
  match t p:
    case 0n 0n: {==}
    case 1n+r 0n:
      pen_eq_add_succ_false(r, d)
    case 0n 1n+q:
      pen_counts_short_refused(0n, d, q)
    case 1n+r 1n+q:
      pen_counts_short_refused(1n+r, d, q)

# promise: a position inside the prompt is never penalised as a generated one
# catches: pen_counts_match drops its Nat.is_le(prompt_len, seq_index) guard (an empty ledger passes)
law pen_counts_prompt_row_refused:
  for t: Nat
  for +p: Nat
  for d: Nat
  {pen_counts_match(t, p, 1n+Nat.add(p, d)) == False{} : Bool}

def pen_counts_prompt_row_refused(t, p, d):
  match p:
    case 0n: {==}
    case 1n+q:
      pen_counts_prompt_row_refused(t, q, d)

# promise: emitting a token raises that token's count by exactly one
# harm: a dropped or doubled count would penalise the token too little or too much
# catches: the True{} arm of pen_count_bump returns count (the increment dropped)
law pen_count_emitted_up_one:
  for c: Nat
  for +t: Nat
  {pen_count_add(c, t, t) == 1n+c : Nat}

def pen_count_emitted_up_one(c, t):
  match t:
    case 0n: {==}
    case 1n+q:
      pen_count_emitted_up_one(c, q)

# promise: emitting a token leaves the count of every smaller token id unchanged
# harm: a token that was not emitted would be penalised
# catches: the False{} arm of pen_count_bump returns 1n+count (the wrong token incremented)
law pen_count_lower_unchanged:
  for c: Nat
  for +t: Nat
  for d: Nat
  {pen_count_add(c, t, 1n+Nat.add(t, d)) == c : Nat}

def pen_count_lower_unchanged(c, t, d):
  match t:
    case 0n: {==}
    case 1n+q:
      pen_count_lower_unchanged(c, q, d)

# promise: emitting a token leaves the count of every larger token id unchanged
# harm: a token that was not emitted would be penalised
# catches: pen_count_add compares with Nat.is_ge in place of Nat.is_eq (larger ids incremented too)
law pen_count_higher_unchanged:
  for c: Nat
  for +t: Nat
  for d: Nat
  {pen_count_add(c, 1n+Nat.add(t, d), t) == c : Nat}

def pen_count_higher_unchanged(c, t, d):
  match t:
    case 0n: {==}
    case 1n+q:
      pen_count_higher_unchanged(c, q, d)

# promise: emitting a token raises the ledger's total by exactly one
# harm: the ledger check against the sequence position would refuse or pass the wrong ledger
# catches: pen_total_add returns total (the increment dropped)
law pen_total_up_one:
  for t: Nat
  {pen_total_add(t) == 1n+t : Nat}

def pen_total_up_one(t):
  {==}

# promise: the ids a request may emit stay inside the logits row
# harm: a token id past the row would index outside the logits
# catches: pen_emit_vocab returns Nat.max(row, tokens) (a tokenizer larger than the row wins)
law pen_emit_vocab_within_row:
  for r: Nat
  for k: Nat
  {Nat.is_le(pen_emit_vocab(r, k), r) == True{} : Bool}

def pen_emit_vocab_within_row(r, k):
  L.nat_min_le_l(r, k)

# promise: the ids a request may emit stay inside the tokenizer's tokens (never a padding row)
# harm: banning every real token would be accepted and a padding id could be emitted
# catches: pen_emit_vocab returns row (the padding of the logits row counted as tokens)
law pen_emit_vocab_within_tokens:
  for r: Nat
  for k: Nat
  {Nat.is_le(pen_emit_vocab(r, k), k) == True{} : Bool}

def pen_emit_vocab_within_tokens(r, k):
  L.nat_min_le_r(r, k)

# promise: every id both the logits row and the tokenizer hold may be emitted
# catches: pen_emit_vocab returns 0n (no token may ever be named or emitted)
law pen_emit_vocab_covers_shared:
  for +v: Nat
  for e: Nat
  for f: Nat
  {Nat.is_le(v, pen_emit_vocab(Nat.add(v, e), Nat.add(v, f))) == True{} : Bool}

def pen_emit_vocab_covers_shared(v, e, f):
  match v:
    case 0n:
      L.nat_le_zero(Nat.min(e, f))
    case 1n+q:
      pen_emit_vocab_covers_shared(q, e, f)
