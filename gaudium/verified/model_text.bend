# Model text resources: where a model's tokenizer, chat template, special-token names, sampling
# defaults and context limit come from, the bounds on a vocabulary read from GGUF metadata, what
# each kind of vocabulary entry may do, and the chat form the command line renders a prompt with.
#
# The Rust shell (`src/model_text.rs`, `src/tokenizer/gguf.rs`, `src/main.rs`) reads the files
# and the metadata; every choice it makes about them is a function here:
#   - the source of one resource: the model file's own GGUF metadata, a file of the directory
#     (the checkpoint directory, or the one that holds the GGUF file), or none;
#   - a vocabulary read from a file is usable (tokens, one type per token, a size limit);
#   - a token id read from metadata names a token of the vocabulary;
#   - what an entry of each kind does when text is encoded and decoded;
#   - how `--generate --chat` formats a prompt: a family's built-in form or the model's template.
# The numbering of the GGUF token types and the names of the metadata keys are format facts of
# an outside specification; they live in Rust with tests.

import Base
import nat_lemmas.bend as L

# ============================================================================
# 1. Types
# ============================================================================

# Where one text resource of a model comes from.
type MtSource is Data:
  MtFromFile{}
  MtFromDir{}
  MtAbsent{}

# What a vocabulary entry is (from the token type a GGUF file gives it).
type MtTokKind is Data:
  MtTokNormal{}
  MtTokControl{}
  MtTokUser{}
  MtTokUnused{}

# How the command line formats a `--chat` prompt.
type MtChatForm is Data:
  MtChatGemma{}
  MtChatChatml{}
  MtChatTemplate{}
  MtChatNone{}

# ============================================================================
# 2. Helper lemmas (erased before Rust codegen)
# ============================================================================

def mt_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# n <= lim refutes lim < n, whatever is checked beside it.
def mt_le_lt_false(a: Bool, +n: Nat, +lim: Nat, b: Bool, e: {Nat.is_le(n, lim) == b : Bool}) -> {Bool.and(Bool.and(a, b), Nat.is_lt(lim, n)) == False{} : Bool}:
  match a:
    case True{}:
      match b:
        case False{}: {==}
        case True{}: L.nat_le_lt_false(lim, n, e)
    case False{}: {==}

# id < n refutes n <= id.
def mt_lt_le_false(+id: Nat, +n: Nat, b: Bool, e: {Nat.is_lt(id, n) == b : Bool}) -> {Bool.and(b, Nat.is_le(n, id)) == False{} : Bool}:
  match b:
    case False{}: {==}
    case True{}: L.nat_lt_le_false(id, n, e)

# ============================================================================
# 3. Sources
# ============================================================================

def mt_dir_or_absent(+in_dir: Bool) -> MtSource:
  match in_dir:
    case True{}: MtFromDir{}
    case False{}: MtAbsent{}

# The source of one resource. `gguf`: the model is given as a GGUF file (or a directory of its
# shards); `in_file`: the file's metadata carries the resource; `in_dir`: the directory has it.
# The file's own metadata wins (it was written with the weights it describes); a file of the
# directory fills in only what the metadata lacks.
def mt_source(+gguf: Bool, +in_file: Bool, +in_dir: Bool) -> MtSource:
  match gguf:
    case True{}:
      match in_file:
        case True{}: MtFromFile{}
        case False{}: mt_dir_or_absent(in_dir)
    case False{}:
      mt_dir_or_absent(in_dir)

def mt_is_file(+s: MtSource) -> Bool:
  match s:
    case MtFromFile{}: True{}
    case MtFromDir{}: False{}
    case MtAbsent{}: False{}

def mt_is_dir(+s: MtSource) -> Bool:
  match s:
    case MtFromFile{}: False{}
    case MtFromDir{}: True{}
    case MtAbsent{}: False{}

# ============================================================================
# 4. Vocabulary bounds
# ============================================================================

# A vocabulary of `n` tokens with `t` token types read from a file is usable: at least one
# token, one type per token, and at most `lim` tokens. `lim` is the shell's limit (ids are
# 32-bit signed integers).
def mt_vocab_ok(+n: Nat, +t: Nat, +lim: Nat) -> Bool:
  Bool.and(Bool.and(Nat.is_gt(n, 0n), Nat.is_eq(t, n)), Nat.is_le(n, lim))

# A token id read from metadata names a token of a vocabulary of `n` tokens.
def mt_id_ok(+id: Nat, +n: Nat) -> Bool:
  Nat.is_lt(id, n)

# ============================================================================
# 5. Token kinds
# ============================================================================

# Matched as a whole in the text before the rest is split and merged.
def mt_tok_added(+k: MtTokKind) -> Bool:
  match k:
    case MtTokNormal{}: False{}
    case MtTokControl{}: True{}
    case MtTokUser{}: True{}
    case MtTokUnused{}: False{}

# Left out of a decode that skips special tokens.
def mt_tok_special(+k: MtTokKind) -> Bool:
  match k:
    case MtTokNormal{}: False{}
    case MtTokControl{}: True{}
    case MtTokUser{}: False{}
    case MtTokUnused{}: False{}

# Encoding may produce it (it is in the vocabulary map; an unused entry only decodes).
def mt_tok_encodable(+k: MtTokKind) -> Bool:
  match k:
    case MtTokNormal{}: True{}
    case MtTokControl{}: True{}
    case MtTokUser{}: True{}
    case MtTokUnused{}: False{}

# ============================================================================
# 6. Command-line chat form
# ============================================================================

def mt_builtin_form(+gemma_turns: Bool) -> MtChatForm:
  match gemma_turns:
    case True{}: MtChatGemma{}
    case False{}: MtChatChatml{}

def mt_template_or_none(+template: Bool) -> MtChatForm:
  match template:
    case True{}: MtChatTemplate{}
    case False{}: MtChatNone{}

# `gemma_turns`: the vocabulary has the Gemma 4 turn marker; `builtin`: the family has a built-in
# chat form; `template`: the model has a chat template that parses. A family with a built-in form
# keeps it (its token streams are the ones checked on the card); another renders the model's own
# template, and without one there is no chat form.
def mt_cli_chat(+gemma_turns: Bool, +builtin: Bool, +template: Bool) -> MtChatForm:
  match builtin:
    case True{}: mt_builtin_form(gemma_turns)
    case False{}: mt_template_or_none(template)

# Whether the command line loads the model's chat template: `--chat` is on (`chat`) and the family
# has no built-in chat form that `mt_cli_chat` would prefer (`builtin`).
def mt_cli_wants_template(+chat: Bool, +builtin: Bool) -> Bool:
  Bool.and(chat, Bool.not(builtin))

def mt_chat_is_template(+f: MtChatForm) -> Bool:
  match f:
    case MtChatGemma{}: False{}
    case MtChatChatml{}: False{}
    case MtChatTemplate{}: True{}
    case MtChatNone{}: False{}

# ============================================================================
# 7. Laws
# ============================================================================

# promise: a checkpoint directory never takes a resource from GGUF metadata.
law mt_checkpoint_never_file:
  for +f: Bool
  for +d: Bool
  {mt_is_file(mt_source(False{}, f, d)) == False{} : Bool}

def mt_checkpoint_never_file(f, d):
  match d:
    case True{}: {==}
    case False{}: {==}

# promise: a resource is taken from the file's metadata only when the metadata carries it.
law mt_file_needs_key:
  for +g: Bool
  for +d: Bool
  {mt_is_file(mt_source(g, False{}, d)) == False{} : Bool}

def mt_file_needs_key(g, d):
  match g:
    case True{}:
      match d:
        case True{}: {==}
        case False{}: {==}
    case False{}:
      match d:
        case True{}: {==}
        case False{}: {==}

# promise: a resource is taken from the directory only when the directory has it.
law mt_dir_needs_file:
  for +g: Bool
  for +f: Bool
  {mt_is_dir(mt_source(g, f, False{})) == False{} : Bool}

def mt_dir_needs_file(g, f):
  match g:
    case True{}:
      match f:
        case True{}: {==}
        case False{}: {==}
    case False{}: {==}

# promise: a vocabulary the reader accepts never has more tokens than the limit it was checked against.
# harm: a hostile token count could overflow the 32-bit token ids or exhaust host memory.
law mt_vocab_within:
  for +n: Nat
  for +t: Nat
  for +lim: Nat
  {Bool.and(mt_vocab_ok(n, t, lim), Nat.is_lt(lim, n)) == False{} : Bool}

def mt_vocab_within(n, t, lim):
  mt_le_lt_false(Bool.and(Nat.is_gt(n, 0n), Nat.is_eq(t, n)), n, lim, Nat.is_le(n, lim), {==})

# promise: an empty vocabulary is never accepted.
law mt_vocab_nonempty:
  for +t: Nat
  for +lim: Nat
  {mt_vocab_ok(0n, t, lim) == False{} : Bool}

def mt_vocab_nonempty(t, lim):
  {==}

# promise: a token id the reader accepts always names a token inside the vocabulary.
# harm: an end-of-sequence or padding id past the vocabulary would index outside the token tables.
law mt_id_within:
  for +id: Nat
  for +n: Nat
  {Bool.and(mt_id_ok(id, n), Nat.is_le(n, id)) == False{} : Bool}

def mt_id_within(id, n):
  mt_lt_le_false(id, n, Nat.is_lt(id, n), {==})

# promise: a special token is always an added token, matched whole and never built by merges.
law mt_special_is_added:
  for +k: MtTokKind
  {Bool.and(mt_tok_special(k), Bool.not(mt_tok_added(k))) == False{} : Bool}

def mt_special_is_added(k):
  match k:
    case MtTokNormal{}: {==}
    case MtTokControl{}: {==}
    case MtTokUser{}: {==}
    case MtTokUnused{}: {==}

# promise: an ordinary vocabulary entry is never matched as an added token.
law mt_normal_not_added:
  {mt_tok_added(MtTokNormal{}) == False{} : Bool}

def mt_normal_not_added():
  {==}

# promise: the command line never renders a chat prompt through a template the model does not have.
law mt_chat_template_needs_one:
  for +g: Bool
  for +b: Bool
  {mt_chat_is_template(mt_cli_chat(g, b, False{})) == False{} : Bool}

def mt_chat_template_needs_one(g, b):
  match g:
    case True{}:
      match b:
        case True{}: {==}
        case False{}: {==}
    case False{}:
      match b:
        case True{}: {==}
        case False{}: {==}

# promise: with --chat on, a template the command-line chat form would render through is always loaded first.
# harm: the chat form would pick the model's template and find none loaded, or render without it
law mt_cli_loads_used_template:
  for +g: Bool
  for +b: Bool
  {Bool.and(mt_chat_is_template(mt_cli_chat(g, b, True{})), Bool.not(mt_cli_wants_template(True{}, b))) == False{} : Bool}

def mt_cli_loads_used_template(g, b):
  match g:
    case True{}:
      match b:
        case True{}: {==}
        case False{}: {==}
    case False{}:
      match b:
        case True{}: {==}
        case False{}: {==}

# promise: the command line never loads a chat template when --chat is off.
law mt_cli_no_chat_no_template:
  for +b: Bool
  {mt_cli_wants_template(False{}, b) == False{} : Bool}

def mt_cli_no_chat_no_template(b):
  {==}

def mt_eq_succ_add_false(+n: Nat, +d: Nat) -> {Nat.is_eq(Nat.add(n, 1n+d), n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      mt_eq_succ_add_false(np, d)

# promise: vocabulary validation rejects files with more token types than tokens
# catches: mt_vocab_ok checks t >= n instead of t == n
law mt_vocab_rejects_extra_types:
  for +n: Nat
  for +d: Nat
  for lim: Nat
  {mt_vocab_ok(n, Nat.add(n, 1n+d), lim) == False{} : Bool}

def mt_vocab_rejects_extra_types(n, d, lim):
  %Equal.sym(Bool, Nat.is_eq(Nat.add(n, 1n+d), n), False{}, mt_eq_succ_add_false(n, d)) : {Bool.and(Bool.and(Nat.is_gt(n, 0n), _), Nat.is_le(n, lim)) == False{} : Bool}
  %Equal.sym(Bool, Bool.and(Nat.is_gt(n, 0n), False{}), False{}, mt_and_false_r(Nat.is_gt(n, 0n))) : {Bool.and(_, Nat.is_le(n, lim)) == False{} : Bool}
  {==}

def mt_eq_add_succ_false(+t: Nat, +d: Nat) -> {Nat.is_eq(t, Nat.add(t, 1n+d)) == False{} : Bool}:
  match t:
    case 0n: {==}
    case 1n+tp:
      mt_eq_add_succ_false(tp, d)

# promise: vocabulary validation rejects files with fewer token types than tokens
# catches: mt_vocab_ok checks t <= n instead of t == n
law mt_vocab_rejects_missing_types:
  for +t: Nat
  for +d: Nat
  for lim: Nat
  {mt_vocab_ok(Nat.add(t, 1n+d), t, lim) == False{} : Bool}

def mt_vocab_rejects_missing_types(t, d, lim):
  %Equal.sym(Bool, Nat.is_eq(t, Nat.add(t, 1n+d)), False{}, mt_eq_add_succ_false(t, d)) : {Bool.and(Bool.and(Nat.is_gt(Nat.add(t, 1n+d), 0n), _), Nat.is_le(Nat.add(t, 1n+d), lim)) == False{} : Bool}
  %Equal.sym(Bool, Bool.and(Nat.is_gt(Nat.add(t, 1n+d), 0n), False{}), False{}, mt_and_false_r(Nat.is_gt(Nat.add(t, 1n+d), 0n))) : {Bool.and(_, Nat.is_le(Nat.add(t, 1n+d), lim)) == False{} : Bool}
  {==}
