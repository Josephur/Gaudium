# Slice 6 — Verified A2UI Negotiation Core
# Formally specifies and proves the decisions of the chat server's A2UI (agent-to-user-interface)
# opt-in:
#   - Whether a request gets the A2UI instructions (server switch, request opt-in, a catalog in
#     common), is refused, or is answered as plain text
#   - Where the instructions go in the conversation (appended to the leading system message, or a
#     new leading system message) and what that does to the message count
#   - How many of a request's catalog identifiers are examined

import Base

# ============================================================================
# 1. Helper Nat lemmas (erased before Rust codegen)
# ============================================================================

def a2_le_refl(n: Nat) -> {True{} == Nat.is_le(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      a2_le_refl(p)

def a2_le_succ(n: Nat) -> {Nat.is_le(n, 1n+n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      a2_le_succ(p)

def a2_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      a2_min_le_l(ap, bp)

def a2_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      a2_min_le_r(ap, bp)

# ============================================================================
# 2. Opt-in Decision
# ============================================================================

type A2uiDecision is Data:
  A2Off{}
  A2Refused{}
  A2NoCatalog{}
  A2On{}

# requested: the request declares renderer capabilities
# enabled:   the server was started with A2UI on
# common:    one of the request's catalogs is one this server writes for
def a2ui_decide(requested: Bool, enabled: Bool, common: Bool) -> A2uiDecision:
  match requested:
    case False{}:
      A2Off{}
    case True{}:
      match enabled:
        case False{}:
          A2Refused{}
        case True{}:
          match common:
            case False{}:
              A2NoCatalog{}
            case True{}:
              A2On{}

# The A2UI instructions are added to the conversation.
def a2ui_injects(d: A2uiDecision) -> Bool:
  match d:
    case A2On{}:
      True{}
    case _:
      False{}

# The request is refused (it asked for what this server does not offer).
def a2ui_refuses(d: A2uiDecision) -> Bool:
  match d:
    case A2Refused{}:
      True{}
    case _:
      False{}

# The request runs, and the reply says its A2UI declaration was not applied.
def a2ui_notes(d: A2uiDecision) -> Bool:
  match d:
    case A2NoCatalog{}:
      True{}
    case _:
      False{}

# LAW 2.1: A request that does not ask for A2UI is never touched, whatever the server offers.
law a2ui_unrequested_is_off:
  for e: Bool
  for c: Bool
  {a2ui_decide(False{}, e, c) == A2Off{} : A2uiDecision}

def a2ui_unrequested_is_off(e, c):
  {==}

# LAW 2.2: The instructions are added exactly when the server offers A2UI, the request asked for
# it, and they share a catalog (a renderer never gets components it did not declare).
law a2ui_injects_iff_all:
  for +e: Bool
  for +r: Bool
  for +c: Bool
  {a2ui_injects(a2ui_decide(r, e, c)) == Bool.and(r, Bool.and(e, c)) : Bool}

def a2ui_injects_iff_all(e, r, c):
  match e r c:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# LAW 2.3: A request is refused exactly when it asks for A2UI on a server that has it off.
law a2ui_refuses_iff_disabled:
  for +e: Bool
  for +r: Bool
  for +c: Bool
  {a2ui_refuses(a2ui_decide(r, e, c)) == Bool.and(r, Bool.not(e)) : Bool}

def a2ui_refuses_iff_disabled(e, r, c):
  match e r c:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# LAW 2.4: A refused request never also gets the instructions.
law a2ui_refusal_never_injects:
  for +d: A2uiDecision
  {Bool.and(a2ui_refuses(d), a2ui_injects(d)) == False{} : Bool}

def a2ui_refusal_never_injects(d):
  match d:
    case A2Off{}: {==}
    case A2Refused{}: {==}
    case A2NoCatalog{}: {==}
    case A2On{}: {==}

# LAW 2.5: With A2UI on and asked for, the request always runs: with the instructions, or as
# plain text with a note (never silently, never refused).
law a2ui_enabled_request_runs:
  for +c: Bool
  {Bool.xor(a2ui_injects(a2ui_decide(True{}, True{}, c)), a2ui_notes(a2ui_decide(True{}, True{}, c))) == True{} : Bool}

def a2ui_enabled_request_runs(c):
  match c:
    case True{}: {==}
    case False{}: {==}

# ============================================================================
# 3. Placement of the Instructions
# ============================================================================

type A2uiPlace is Data:
  A2Append{}
  A2Insert{}

# A conversation that opens with a system message keeps one system message (the instructions are
# appended to it); otherwise a system message is put first.
def a2ui_place(first_is_system: Bool) -> A2uiPlace:
  match first_is_system:
    case True{}:
      A2Append{}
    case False{}:
      A2Insert{}

# Messages in the conversation after the instructions went in.
def a2ui_msg_count(n: Nat, p: A2uiPlace) -> Nat:
  match p:
    case A2Append{}:
      n
    case A2Insert{}:
      1n+n

# LAW 3.1: Appending keeps the message count.
law a2ui_append_keeps_count:
  for n: Nat
  {a2ui_msg_count(n, a2ui_place(True{})) == n : Nat}

def a2ui_append_keeps_count(n):
  {==}

# LAW 3.2: Inserting adds exactly one message.
law a2ui_insert_adds_one:
  for n: Nat
  {a2ui_msg_count(n, a2ui_place(False{})) == 1n+n : Nat}

def a2ui_insert_adds_one(n):
  {==}

# LAW 3.3: No message of the request is dropped.
law a2ui_place_keeps_messages:
  for +n: Nat
  for +p: A2uiPlace
  {Nat.is_le(n, a2ui_msg_count(n, p)) == True{} : Bool}

def a2ui_place_keeps_messages(n, p):
  match p:
    case A2Append{}:
      Equal.sym(Bool, True{}, Nat.is_le(n, n), a2_le_refl(n))
    case A2Insert{}:
      a2_le_succ(n)

# LAW 3.4: At most one message is added.
law a2ui_place_adds_at_most_one:
  for +n: Nat
  for +p: A2uiPlace
  {Nat.is_le(a2ui_msg_count(n, p), 1n+n) == True{} : Bool}

def a2ui_place_adds_at_most_one(n, p):
  match p:
    case A2Append{}:
      a2_le_succ(n)
    case A2Insert{}:
      Equal.sym(Bool, True{}, Nat.is_le(n, n), a2_le_refl(n))

# ============================================================================
# 4. Catalog Identifiers Examined
# ============================================================================

# A request may list any number of catalog identifiers; only the first `limit` are compared.
def a2ui_ids_scanned(listed: Nat, limit: Nat) -> Nat:
  Nat.min(listed, limit)

# LAW 4.1: Never more than were listed (the scan cannot run past the list).
law a2ui_scan_within_list:
  for +l: Nat
  for +m: Nat
  {Nat.is_le(a2ui_ids_scanned(l, m), l) == True{} : Bool}

def a2ui_scan_within_list(l, m):
  a2_min_le_l(l, m)

# LAW 4.2: Never more than the limit.
law a2ui_scan_within_limit:
  for +l: Nat
  for +m: Nat
  {Nat.is_le(a2ui_ids_scanned(l, m), m) == True{} : Bool}

def a2ui_scan_within_limit(l, m):
  a2_min_le_r(l, m)

# LAW 4.3: An empty list examines nothing.
law a2ui_scan_empty_list:
  for m: Nat
  {a2ui_ids_scanned(0n, m) == 0n : Nat}

def a2ui_scan_empty_list(m):
  {==}

# ============================================================================
# 5. Instructions Read From a File (hot reload)
# ============================================================================
# The server may read the instructions from a file and re-read it whenever it changes on disk.
# Each request looks at the file's stamp (modification time, size, or that it is gone). A changed
# file is loaded only when every check passes; otherwise the text served so far stays (the last
# good file text, or the built-in instructions when no file text was ever good) and the server
# says why. An unchanged stamp does nothing, so a bad file is reported once, not per request.

type A2PromptPick is Data:
  A2PkKeep{}
  A2PkLoad{}
  A2PkReject{}

# changed: the stamp differs from the one seen last; valid: the new text passed every check
def a2ui_prompt_pick(changed: Bool, valid: Bool) -> A2PromptPick:
  match changed:
    case False{}:
      A2PkKeep{}
    case True{}:
      match valid:
        case True{}:
          A2PkLoad{}
        case False{}:
          A2PkReject{}

# The new text replaces the served one.
def a2ui_prompt_loads(p: A2PromptPick) -> Bool:
  match p:
    case A2PkLoad{}:
      True{}
    case _:
      False{}

# The server reports that the file was not taken.
def a2ui_prompt_warns(p: A2PromptPick) -> Bool:
  match p:
    case A2PkReject{}:
      True{}
    case _:
      False{}

# The size check: a file with something in it, and no larger than the cap.
def a2ui_prompt_size_ok(+len: Nat, cap: Nat) -> Bool:
  Bool.and(Nat.is_gt(len, 0n), Nat.is_le(len, cap))

# LAW 5.1: Text that failed a check is never served, changed or not.
law a2ui_prompt_invalid_never_loads:
  for +c: Bool
  {a2ui_prompt_loads(a2ui_prompt_pick(c, False{})) == False{} : Bool}

def a2ui_prompt_invalid_never_loads(c):
  match c:
    case True{}: {==}
    case False{}: {==}

# LAW 5.2: A rejected file is always reported (a bad edit is never dropped silently).
law a2ui_prompt_reject_warns:
  {a2ui_prompt_warns(a2ui_prompt_pick(True{}, False{})) == True{} : Bool}

def a2ui_prompt_reject_warns():
  {==}

# LAW 5.3: An unchanged file neither reloads nor reports again.
law a2ui_prompt_unchanged_quiet:
  for +v: Bool
  {Bool.or(a2ui_prompt_loads(a2ui_prompt_pick(False{}, v)), a2ui_prompt_warns(a2ui_prompt_pick(False{}, v))) == False{} : Bool}

def a2ui_prompt_unchanged_quiet(v):
  match v:
    case True{}: {==}
    case False{}: {==}

# LAW 5.4: Loading and reporting never happen together.
law a2ui_prompt_load_xor_warn:
  for +p: A2PromptPick
  {Bool.and(a2ui_prompt_loads(p), a2ui_prompt_warns(p)) == False{} : Bool}

def a2ui_prompt_load_xor_warn(p):
  match p:
    case A2PkKeep{}: {==}
    case A2PkLoad{}: {==}
    case A2PkReject{}: {==}

# LAW 5.5: An empty file never passes the size check.
law a2ui_prompt_empty_rejected:
  for c: Nat
  {a2ui_prompt_size_ok(0n, c) == False{} : Bool}

def a2ui_prompt_empty_rejected(c):
  {==}

def a2_le_add_succ_false(+cap: Nat, +d: Nat) -> {Nat.is_le(Nat.add(cap, 1n+d), cap) == False{} : Bool}:
  match cap:
    case 0n: {==}
    case 1n+cp:
      a2_le_add_succ_false(cp, d)

def a2_and_false_r(b: Bool) -> {Bool.and(b, False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# LAW 5.6: A prompt file strictly larger than cap is always rejected by a2ui_prompt_size_ok.
# promise: a2ui_prompt_size_ok rejects any prompt size strictly exceeding cap
# catches: a2ui_prompt_size_ok drops Nat.is_le(len, cap) and accepts any non-empty prompt
law a2ui_prompt_over_cap_rejected:
  for +cap: Nat
  for +d: Nat
  {a2ui_prompt_size_ok(Nat.add(cap, 1n+d), cap) == False{} : Bool}

def a2ui_prompt_over_cap_rejected(cap, d):
  %Equal.sym(Bool, Nat.is_le(Nat.add(cap, 1n+d), cap), False{}, a2_le_add_succ_false(cap, d)) : {Bool.and(Nat.is_gt(Nat.add(cap, 1n+d), 0n), _) == False{} : Bool}
  a2_and_false_r(Nat.is_gt(Nat.add(cap, 1n+d), 0n))
