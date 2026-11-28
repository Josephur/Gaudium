# Prompt token ids a client sends: a plain completion request may give its prompt as token ids
# instead of text, and those ids go to the engine's embedding lookup as they are.
#
# The Rust shell (`src/server/completion.rs`) parses the ids from the request body (refusing
# values that are not non-negative integers) and asks this file whether each one may run.

import Base
import nat_lemmas.bend as L

# ============================================================================
# 1. Helper lemmas (erased before Rust codegen)
# ============================================================================

# id < rows refutes rows <= id, whatever else is checked beside it.
def pt_rows_refute(+id: Nat, +rows: Nat, a: Bool, e: {Nat.is_lt(id, rows) == a : Bool}, c: Bool) -> {Bool.and(Bool.and(a, c), Nat.is_le(rows, id)) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}:
      match c:
        case False{}: {==}
        case True{}: L.nat_lt_le_false(id, rows, e)

# ============================================================================
# 2. Decision
# ============================================================================

# A client's prompt token id may run: it is a row of the model's embedding (`rows`) and a token
# the model's tokenizer names (`names` entries).
def prompt_token_ok(+id: Nat, +rows: Nat, +names: Nat) -> Bool:
  Bool.and(Nat.is_lt(id, rows), Nat.is_lt(id, names))

# ============================================================================
# 3. Laws
# ============================================================================

# promise: a prompt token id at or past the embedding's last row is never accepted.
# harm: the engine would read the embedding table past its end for that prompt row.
# catches: the embedding bound dropped (only the tokenizer's size checked)
law prompt_token_within_rows:
  for +id: Nat
  for +rows: Nat
  for +names: Nat
  {Bool.and(prompt_token_ok(id, rows, names), Nat.is_le(rows, id)) == False{} : Bool}

def prompt_token_within_rows(id, rows, names):
  pt_rows_refute(id, rows, Nat.is_lt(id, rows), {==}, Nat.is_lt(id, names))

# promise: the first token of a model with any embedding rows and tokens is accepted.
# catches: every token id refused (a constant False)
law prompt_token_first_ok:
  for +r: Nat
  for +n: Nat
  {prompt_token_ok(0n, 1n+r, 1n+n) == True{} : Bool}

def prompt_token_first_ok(r, n):
  {==}
