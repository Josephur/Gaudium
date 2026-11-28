# Slice 5 — Verified Tool-Calling Core
# Formally specifies and proves the decisions of the chat server's tool-calling path:
#   - Tool-choice policy (auto / none / required / named) and forced calls
#   - Tool-call stream lifecycle (header before fragments, no reopen, call limit)
#   - How a generation ended, and the finish / stop reason a client is told
#   - Hold-back arithmetic of the streaming text parser (partial tags, stop strings)
#   - Content-block lifecycle of a block-structured message stream
#   - Slice bounds of the chat-template renderer

import Base

# ============================================================================
# 1. Helper Nat / Bool lemmas (erased before Rust codegen)
# ============================================================================

def tc_le_refl(n: Nat) -> {True{} == Nat.is_le(n, n) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      tc_le_refl(p)

def tc_le_step_r_zero(xp: Nat, h: {False{} == True{} : Bool}) -> {Nat.is_le(xp, 0n) == True{} : Bool}:
  match xp:
    case 0n: {==}
    case 1n+_: h

def tc_le_step_r(x: Nat, y: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) -> {Nat.is_le(x, 1n+y) == True{} : Bool}:
  match x y:
    case 0n _:
      {==}
    case 1n+xp 0n:
      tc_le_step_r_zero(xp, h)
    case 1n+xp 1n+yp:
      tc_le_step_r(xp, yp, h)

def tc_sub_le_self(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      Equal.sym(Bool, True{}, Nat.is_le(ap, ap), tc_le_refl(ap))
    case 1n+ap 1n+bp:
      tc_le_step_r(Nat.sub(ap, bp), ap, tc_sub_le_self(ap, bp))

def tc_zero_le(b: Nat) -> {Nat.is_le(0n, b) == True{} : Bool}:
  match b:
    case 0n: {==}
    case 1n+_: {==}

def tc_sub_zero(b: Nat) -> {Nat.sub(b, 0n) == b : Nat}:
  match b:
    case 0n: {==}
    case 1n+_: {==}

def tc_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      tc_min_le_l(ap, bp)

def tc_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      tc_min_le_r(ap, bp)

def tc_min_lt_succ(+a: Nat, +b: Nat) -> {Nat.is_lt(Nat.min(a, b), 1n+b) == True{} : Bool}:
  match a b:
    case 0n _: {==}
    case 1n+ap 0n: {==}
    case 1n+ap 1n+bp:
      tc_min_lt_succ(ap, bp)

def tc_min_sub_conserved(+b: Nat, +x: Nat) -> {Nat.add(Nat.min(b, x), Nat.sub(b, Nat.min(b, x))) == b : Nat}:
  match b x:
    case 0n _: {==}
    case 1n+bp 0n: {==}
    case 1n+bp 1n+xp:
      %tc_min_sub_conserved(bp, xp) : {1n+Nat.add(Nat.min(bp, xp), Nat.sub(bp, Nat.min(bp, xp))) == 1n+_ : Nat}
      {==}

# ============================================================================
# 2. Tool-Choice Policy
# ============================================================================

type ToolChoice is Data:
  TcAuto{}
  TcNone{}
  TcRequired{}
  TcNamed{}

# The request's tools are shown to the model and its tool calls are parsed.
def tool_choice_offers(c: ToolChoice, tools_given: Bool) -> Bool:
  match c:
    case TcNone{}:
      False{}
    case _:
      tools_given

# The reply must be a tool call.
def tool_choice_forces(c: ToolChoice, tools_given: Bool) -> Bool:
  match c:
    case TcRequired{}:
      tools_given
    case TcNamed{}:
      tools_given
    case _:
      False{}

# `required` and a named function need a non-empty tool list (else the request is refused).
def tool_choice_valid(c: ToolChoice, tools_given: Bool) -> Bool:
  match c:
    case TcRequired{}:
      tools_given
    case TcNamed{}:
      tools_given
    case _:
      True{}

# A reply that continues the final message is never forced into a call.
def tool_force_applies(forces: Bool, continue_final: Bool) -> Bool:
  Bool.and(forces, Bool.not(continue_final))

# A forced call opens the reply, so the model cannot reason first.
def tool_thinking_allowed(wants_thinking: Bool, force_applies: Bool) -> Bool:
  Bool.and(wants_thinking, Bool.not(force_applies))

# LAW 2.1: A forced call implies the tools are offered (never force a call the model was not shown).
law tool_force_implies_offer:
  for +c: ToolChoice
  for +g: Bool
  {Bool.or(Bool.not(tool_choice_forces(c, g)), tool_choice_offers(c, g)) == True{} : Bool}

def tool_force_implies_offer(c, g):
  match c g:
    case TcAuto{} True{}: {==}
    case TcAuto{} False{}: {==}
    case TcNone{} True{}: {==}
    case TcNone{} False{}: {==}
    case TcRequired{} True{}: {==}
    case TcRequired{} False{}: {==}
    case TcNamed{} True{}: {==}
    case TcNamed{} False{}: {==}

# LAW 2.2: `none` neither offers nor forces, whatever the tool list.
law tool_none_offers_nothing:
  for +g: Bool
  {Bool.or(tool_choice_offers(TcNone{}, g), tool_choice_forces(TcNone{}, g)) == False{} : Bool}

def tool_none_offers_nothing(g):
  {==}

# LAW 2.3: Without tools nothing is offered or forced, whatever the choice.
law tool_no_tools_offers_nothing:
  for +c: ToolChoice
  {Bool.or(tool_choice_offers(c, False{}), tool_choice_forces(c, False{})) == False{} : Bool}

def tool_no_tools_offers_nothing(c):
  match c:
    case TcAuto{}: {==}
    case TcNone{}: {==}
    case TcRequired{}: {==}
    case TcNamed{}: {==}

# LAW 2.4: A request that is refused as invalid would not have forced a call.
law tool_invalid_choice_never_forces:
  for +c: ToolChoice
  for +g: Bool
  {Bool.or(tool_choice_valid(c, g), Bool.not(tool_choice_forces(c, g))) == True{} : Bool}

def tool_invalid_choice_never_forces(c, g):
  match c g:
    case TcAuto{} True{}: {==}
    case TcAuto{} False{}: {==}
    case TcNone{} True{}: {==}
    case TcNone{} False{}: {==}
    case TcRequired{} True{}: {==}
    case TcRequired{} False{}: {==}
    case TcNamed{} True{}: {==}
    case TcNamed{} False{}: {==}

# LAW 2.5: `auto` never forces.
law tool_auto_never_forces:
  for g: Bool
  {tool_choice_forces(TcAuto{}, g) == False{} : Bool}

def tool_auto_never_forces(g):
  {==}

# LAW 2.6: A continued final message is never forced.
law tool_continue_never_forces:
  for f: Bool
  {tool_force_applies(f, True{}) == False{} : Bool}

def tool_continue_never_forces(f):
  match f:
    case True{}: {==}
    case False{}: {==}

# LAW 2.7: A forced call and reasoning never go together.
law tool_forced_call_disables_thinking:
  for w: Bool
  {tool_thinking_allowed(w, True{}) == False{} : Bool}

def tool_forced_call_disables_thinking(w):
  match w:
    case True{}: {==}
    case False{}: {==}

# ============================================================================
# 3. Tool-Call Stream Lifecycle
# ============================================================================

type CallPhase is Data:
  CpText{}
  CpBlock{}
  CpArgs{}
  CpStopped{}

type CallEvent is Data:
  CeOpen{}
  CeHeader{}
  CeFragment{}
  CeFinish{}
  CeAbandon{}

# CpText: outside a call. CpBlock: inside a tool block whose name is not complete (it may still
# turn out to be plain text). CpArgs: the call's header is out and its arguments stream.
# CpStopped: the call limit ended the stream.
def call_accepts(p: CallPhase, ev: CallEvent) -> Bool:
  match p ev:
    case CpText{} CeOpen{}:
      True{}
    case CpBlock{} CeHeader{}:
      True{}
    case CpBlock{} CeAbandon{}:
      True{}
    case CpArgs{} CeFragment{}:
      True{}
    case CpArgs{} CeFinish{}:
      True{}
    case _ _:
      False{}

def call_step(p: CallPhase, ev: CallEvent) -> CallPhase:
  match p ev:
    case CpText{} CeOpen{}:
      CpBlock{}
    case CpBlock{} CeHeader{}:
      CpArgs{}
    case CpBlock{} CeAbandon{}:
      CpText{}
    case CpArgs{} CeFinish{}:
      CpText{}
    case st _:
      st

def call_is_streaming(p: CallPhase) -> Bool:
  match p:
    case CpArgs{}:
      True{}
    case _:
      False{}

def call_phase_eq(a: CallPhase, b: CallPhase) -> Bool:
  match a b:
    case CpText{} CpText{}:
      True{}
    case CpBlock{} CpBlock{}:
      True{}
    case CpArgs{} CpArgs{}:
      True{}
    case CpStopped{} CpStopped{}:
      True{}
    case _ _:
      False{}

def call_limit_phase(reached: Bool) -> CallPhase:
  match reached:
    case True{}:
      CpStopped{}
    case False{}:
      CpText{}

# The phase after a call finished, `done` calls having finished before it.
def call_after_finish(done: Nat, has_limit: Bool, limit: Nat) -> CallPhase:
  match has_limit:
    case False{}:
      CpText{}
    case True{}:
      call_limit_phase(Nat.is_le(limit, Nat.add(1n, done)))

# LAW 3.1: Argument fragments are accepted exactly while a call's header is out.
law call_fragment_needs_header:
  for +p: CallPhase
  {call_accepts(p, CeFragment{}) == call_is_streaming(p) : Bool}

def call_fragment_needs_header(p):
  match p:
    case CpText{}: {==}
    case CpBlock{}: {==}
    case CpArgs{}: {==}
    case CpStopped{}: {==}

# LAW 3.2: A call whose header went out is never taken back as plain text.
law call_started_never_abandoned:
  {call_accepts(CpArgs{}, CeAbandon{}) == False{} : Bool}

def call_started_never_abandoned():
  {==}

# LAW 3.3: A header is accepted only inside an opened block (never twice for one call).
law call_header_needs_block:
  for +p: CallPhase
  {call_accepts(p, CeHeader{}) == call_phase_eq(p, CpBlock{}) : Bool}

def call_header_needs_block(p):
  match p:
    case CpText{}: {==}
    case CpBlock{}: {==}
    case CpArgs{}: {==}
    case CpStopped{}: {==}

# LAW 3.4: A stopped stream is absorbing and accepts nothing.
law call_stopped_absorbing:
  for ev: CallEvent
  {call_step(CpStopped{}, ev) == CpStopped{} : CallPhase}

def call_stopped_absorbing(ev):
  match ev:
    case CeOpen{}: {==}
    case CeHeader{}: {==}
    case CeFragment{}: {==}
    case CeFinish{}: {==}
    case CeAbandon{}: {==}

law call_stopped_accepts_nothing:
  for ev: CallEvent
  {call_accepts(CpStopped{}, ev) == False{} : Bool}

def call_stopped_accepts_nothing(ev):
  match ev:
    case CeOpen{}: {==}
    case CeHeader{}: {==}
    case CeFragment{}: {==}
    case CeFinish{}: {==}
    case CeAbandon{}: {==}

# LAW 3.5: An event that is not accepted leaves the phase unchanged.
law call_rejected_keeps_phase:
  for +p: CallPhase
  for +ev: CallEvent
  {Bool.or(call_accepts(p, ev), call_phase_eq(call_step(p, ev), p)) == True{} : Bool}

def call_rejected_keeps_phase(p, ev):
  match p ev:
    case CpText{} CeOpen{}: {==}
    case CpText{} CeHeader{}: {==}
    case CpText{} CeFragment{}: {==}
    case CpText{} CeFinish{}: {==}
    case CpText{} CeAbandon{}: {==}
    case CpBlock{} CeOpen{}: {==}
    case CpBlock{} CeHeader{}: {==}
    case CpBlock{} CeFragment{}: {==}
    case CpBlock{} CeFinish{}: {==}
    case CpBlock{} CeAbandon{}: {==}
    case CpArgs{} CeOpen{}: {==}
    case CpArgs{} CeHeader{}: {==}
    case CpArgs{} CeFragment{}: {==}
    case CpArgs{} CeFinish{}: {==}
    case CpArgs{} CeAbandon{}: {==}
    case CpStopped{} CeOpen{}: {==}
    case CpStopped{} CeHeader{}: {==}
    case CpStopped{} CeFragment{}: {==}
    case CpStopped{} CeFinish{}: {==}
    case CpStopped{} CeAbandon{}: {==}

# LAW 3.6: Without a call limit the stream never stops on a finished call.
law call_no_limit_never_stops:
  for done: Nat
  for limit: Nat
  {call_after_finish(done, False{}, limit) == CpText{} : CallPhase}

def call_no_limit_never_stops(done, limit):
  {==}

# LAW 3.7: A limit of one call (parallel calls off) stops the stream at the first finished call.
law call_limit_one_stops:
  for done: Nat
  {call_after_finish(done, True{}, 1n) == CpStopped{} : CallPhase}

def call_limit_one_stops(done):
  match done:
    case 0n: {==}
    case 1n+_: {==}

# LAW 3.8: The call that reaches the limit stops the stream.
law call_limit_reached_stops:
  for +done: Nat
  {call_after_finish(done, True{}, 1n+done) == CpStopped{} : CallPhase}

def call_limit_reached_stops(done):
  %tc_le_refl(done) : {call_limit_phase(_) == CpStopped{} : CallPhase}
  {==}

# ============================================================================
# 4. Generation End, Finish Reason and Stop Reason
# ============================================================================

type GenEnd is Data:
  GeEos{}
  GeStopString{}
  GeCallLimit{}
  GeLength{}
  GeCancelled{}

# A stop string and the call limit end a generation by cancelling it, so they are read before the
# cancel flag; an end-of-sequence token that arrived anyway is a normal end.
def gen_end(stop_string: Bool, call_limit: Bool, eos: Bool, cancelled: Bool) -> GenEnd:
  match stop_string:
    case True{}:
      GeStopString{}
    case False{}:
      match call_limit:
        case True{}:
          GeCallLimit{}
        case False{}:
          match eos:
            case True{}:
              GeEos{}
            case False{}:
              match cancelled:
                case True{}:
                  GeCancelled{}
                case False{}:
                  GeLength{}

def ge_is_cancelled(e: GenEnd) -> Bool:
  match e:
    case GeCancelled{}:
      True{}
    case _:
      False{}

def ge_is_stop_string(e: GenEnd) -> Bool:
  match e:
    case GeStopString{}:
      True{}
    case _:
      False{}

# The generation ended where the model ended it: an end-of-sequence token, or the call limit
# after a whole call. A generation cut by the length limit, a stop string or a cancel did not.
# (The server-side tool loop runs a round's calls only after such an end.)
def ge_model_ended(e: GenEnd) -> Bool:
  match e:
    case GeEos{}:
      True{}
    case GeCallLimit{}:
      True{}
    case _:
      False{}

# promise: a generation cut by the length limit is never one the model ended
# harm: the half-written call at the cut would be run as a whole one
# catches: ge_model_ended answers True{} for every end
law ge_length_not_model_end:
  {ge_model_ended(GeLength{}) == False{} : Bool}

def ge_length_not_model_end():
  {==}

# promise: a cancelled generation is never one the model ended
# catches: ge_model_ended answers True{} for GeCancelled (a cancelled request runs its calls)
law ge_cancel_not_model_end:
  {ge_model_ended(GeCancelled{}) == False{} : Bool}

def ge_cancel_not_model_end():
  {==}

# promise: a generation ended by one of the request's stop strings is never one the model ended
# catches: ge_model_ended answers True{} for GeStopString (the client's stop sequence overridden)
law ge_stop_string_not_model_end:
  {ge_model_ended(GeStopString{}) == False{} : Bool}

def ge_stop_string_not_model_end():
  {==}

# promise: a generation that ended with an end-of-sequence token is one the model ended
# catches: ge_model_ended answers False{} always (no server-side call ever runs)
law ge_eos_is_model_end:
  {ge_model_ended(GeEos{}) == True{} : Bool}

def ge_eos_is_model_end():
  {==}

# promise: a generation the call limit ended after a whole call is one the model ended
# catches: ge_model_ended answers False{} for GeCallLimit (the round's whole calls are never run)
law ge_call_limit_is_model_end:
  {ge_model_ended(GeCallLimit{}) == True{} : Bool}

def ge_call_limit_is_model_end():
  {==}

# How a generation ended, from the engine's flags and the parser's. stopped: the parser stopped the
# generation (a stop string or the call limit); repaired: the parser closed the last call itself
# when the generation ended, so the model never finished that call. The call limit counts only
# when a call the model finished reached it: a generation whose last call was repaired ends as the
# engine ended it (a cancel, the length limit, an end-of-sequence token).
def gen_end_closed(stop_string: Bool, stopped: Bool, repaired: Bool, eos: Bool, cancelled: Bool) -> GenEnd:
  gen_end(stop_string, Bool.and(stopped, Bool.not(repaired)), eos, cancelled)

# promise: a generation cut by the length limit or a cancel inside a call the parser then closed is never one the model ended
# harm: the half-written call would be run as a whole one, on a remote tool server with arguments the model never finished
# catches: the repaired flag ignored (gen_end(stop_string, stopped, eos, cancelled)): the repair reaches the call limit and the round counts as ended
law ge_repaired_cut_not_model_end:
  for +ss: Bool
  for +st: Bool
  for +c: Bool
  {ge_model_ended(gen_end_closed(ss, st, True{}, False{}, c)) == False{} : Bool}

def ge_repaired_cut_not_model_end(ss, st, c):
  match ss st c:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# promise: a generation the call limit stopped after a call the model finished still ends as the call limit
# catches: the call-limit flag dropped (gen_end(stop_string, False{}, eos, cancelled)): a whole call under the one-call limit is reported as a length cut and never runs
law ge_whole_call_limit_stays:
  for +eos: Bool
  for +c: Bool
  {gen_end_closed(False{}, True{}, False{}, eos, c) == GeCallLimit{} : GenEnd}

def ge_whole_call_limit_stays(eos, c):
  match eos c:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# promise: a generation the parser did not stop (no stop string, no call limit) ends as the engine ended it, whatever the parser closed at the end
# harm: a plain length cut or cancel would count as the call limit, and its calls would run
# catches: the stopped flag dropped (gen_end(stop_string, Bool.not(repaired), eos, cancelled)): every generation without a repaired call counts as the call limit
law ge_unstopped_ends_as_engine:
  for +r: Bool
  for +eos: Bool
  for +c: Bool
  {gen_end_closed(False{}, False{}, r, eos, c) == gen_end(False{}, False{}, eos, c) : GenEnd}

def ge_unstopped_ends_as_engine(r, eos, c):
  match r eos c:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# promise: a generation one of the request's stop strings ended ends as the stop string, whatever the parser stopped or closed
# catches: the stop-string flag dropped (gen_end(False{}, Bool.and(stopped, Bool.not(repaired)), eos, cancelled)): a round a stop string ended counts as the call limit and runs its calls
law ge_stop_string_closed_wins:
  for +st: Bool
  for +r: Bool
  for +eos: Bool
  for +c: Bool
  {gen_end_closed(True{}, st, r, eos, c) == GeStopString{} : GenEnd}

def ge_stop_string_closed_wins(st, r, eos, c):
  {==}

type FinishReason is Data:
  FrStop{}
  FrLength{}
  FrToolCalls{}
  FrCancelled{}

def finish_pick(has_calls: Bool, otherwise: FinishReason) -> FinishReason:
  match has_calls:
    case True{}:
      FrToolCalls{}
    case False{}:
      otherwise

# `finish_reason` of a chat completion.
def finish_reason(e: GenEnd, has_calls: Bool) -> FinishReason:
  match e:
    case GeCancelled{}:
      FrCancelled{}
    case GeLength{}:
      finish_pick(has_calls, FrLength{})
    case _:
      finish_pick(has_calls, FrStop{})

def fr_is_tool_calls(f: FinishReason) -> Bool:
  match f:
    case FrToolCalls{}:
      True{}
    case _:
      False{}

type StopReason is Data:
  SrEndTurn{}
  SrMaxTokens{}
  SrStopSequence{}
  SrToolUse{}

def stop_pick(has_calls: Bool, otherwise: StopReason) -> StopReason:
  match has_calls:
    case True{}:
      SrToolUse{}
    case False{}:
      otherwise

# `stop_reason` of a block-structured message: a call cut by the token limit reports the limit.
def stop_reason(e: GenEnd, has_calls: Bool) -> StopReason:
  match e:
    case GeLength{}:
      SrMaxTokens{}
    case GeStopString{}:
      stop_pick(has_calls, SrStopSequence{})
    case _:
      stop_pick(has_calls, SrEndTurn{})

def sr_is_tool_use(s: StopReason) -> Bool:
  match s:
    case SrToolUse{}:
      True{}
    case _:
      False{}

def sr_is_stop_sequence(s: StopReason) -> Bool:
  match s:
    case SrStopSequence{}:
      True{}
    case _:
      False{}

# LAW 4.1: A matched stop string is the end, whatever else is set.
law gen_end_stop_string_wins:
  for limit: Bool
  for eos: Bool
  for cancelled: Bool
  {gen_end(True{}, limit, eos, cancelled) == GeStopString{} : GenEnd}

def gen_end_stop_string_wins(limit, eos, cancelled):
  {==}

# LAW 4.2: The call limit is never reported as a client cancellation.
law gen_end_call_limit_not_cancel:
  for eos: Bool
  for cancelled: Bool
  {gen_end(False{}, True{}, eos, cancelled) == GeCallLimit{} : GenEnd}

def gen_end_call_limit_not_cancel(eos, cancelled):
  {==}

# LAW 4.3: With no flag set the generation ran out of tokens.
law gen_end_no_flag_is_length:
  {gen_end(False{}, False{}, False{}, False{}) == GeLength{} : GenEnd}

def gen_end_no_flag_is_length():
  {==}

# LAW 4.4: A cancelled request reports cancelled, with or without calls.
law finish_cancel_absorbs:
  for c: Bool
  {finish_reason(GeCancelled{}, c) == FrCancelled{} : FinishReason}

def finish_cancel_absorbs(c):
  {==}

# LAW 4.5: `tool_calls` is never reported without a call.
law finish_tool_calls_need_calls:
  for e: GenEnd
  {fr_is_tool_calls(finish_reason(e, False{})) == False{} : Bool}

def finish_tool_calls_need_calls(e):
  match e:
    case GeEos{}: {==}
    case GeStopString{}: {==}
    case GeCallLimit{}: {==}
    case GeLength{}: {==}
    case GeCancelled{}: {==}

# LAW 4.6: A reply with calls reports `tool_calls` exactly when it was not cancelled.
law finish_calls_reported_unless_cancelled:
  for +e: GenEnd
  {Bool.xor(fr_is_tool_calls(finish_reason(e, True{})), ge_is_cancelled(e)) == True{} : Bool}

def finish_calls_reported_unless_cancelled(e):
  match e:
    case GeEos{}: {==}
    case GeStopString{}: {==}
    case GeCallLimit{}: {==}
    case GeLength{}: {==}
    case GeCancelled{}: {==}

# LAW 4.7: Running out of tokens is always `max_tokens`.
law stop_length_is_max_tokens:
  for c: Bool
  {stop_reason(GeLength{}, c) == SrMaxTokens{} : StopReason}

def stop_length_is_max_tokens(c):
  {==}

# LAW 4.8: `tool_use` is never reported without a call.
law stop_tool_use_needs_calls:
  for e: GenEnd
  {sr_is_tool_use(stop_reason(e, False{})) == False{} : Bool}

def stop_tool_use_needs_calls(e):
  match e:
    case GeEos{}: {==}
    case GeStopString{}: {==}
    case GeCallLimit{}: {==}
    case GeLength{}: {==}
    case GeCancelled{}: {==}

# LAW 4.9: `stop_sequence` is reported only when a stop string matched.
law stop_sequence_needs_stop_string:
  for +e: GenEnd
  for c: Bool
  {Bool.or(Bool.not(sr_is_stop_sequence(stop_reason(e, c))), ge_is_stop_string(e)) == True{} : Bool}

def stop_sequence_needs_stop_string(e, c):
  match e c:
    case GeEos{} True{}: {==}
    case GeEos{} False{}: {==}
    case GeStopString{} True{}: {==}
    case GeStopString{} False{}: {==}
    case GeCallLimit{} True{}: {==}
    case GeCallLimit{} False{}: {==}
    case GeLength{} True{}: {==}
    case GeLength{} False{}: {==}
    case GeCancelled{} True{}: {==}
    case GeCancelled{} False{}: {==}

# ============================================================================
# 5. Hold-Back Arithmetic (partial tags and stop strings)
# ============================================================================

# The longest tail of a `buf_len`-byte buffer that may still be the start of a `tag_len`-byte tag:
# a proper prefix of the tag, and no longer than the buffer.
def hold_max(buf_len: Nat, tag_len: Nat) -> Nat:
  match tag_len:
    case 0n:
      0n
    case 1n+t:
      Nat.min(buf_len, t)

# Bytes released when `held` bytes stay back.
def hold_release(buf_len: Nat, held: Nat) -> Nat:
  Nat.sub(buf_len, held)

# LAW 5.1: The held tail never exceeds the buffer (the cut index cannot underflow).
law hold_max_le_buf:
  for +b: Nat
  for t: Nat
  {Nat.is_le(hold_max(b, t), b) == True{} : Bool}

def hold_max_le_buf(b, t):
  match t:
    case 0n:
      tc_zero_le(b)
    case 1n+tp:
      tc_min_le_l(b, tp)

# LAW 5.2: The held tail is shorter than the tag (a complete tag is never held back).
law hold_max_lt_tag:
  for b: Nat
  for +t: Nat
  {Nat.is_lt(hold_max(b, 1n+t), 1n+t) == True{} : Bool}

def hold_max_lt_tag(b, t):
  tc_min_lt_succ(b, t)

# LAW 5.3: An empty tag holds nothing back.
law hold_empty_tag_holds_nothing:
  for b: Nat
  {hold_max(b, 0n) == 0n : Nat}

def hold_empty_tag_holds_nothing(b):
  {==}

# LAW 5.4: Held plus released is the whole buffer (no byte is lost or written twice).
law hold_conserved:
  for +b: Nat
  for +t: Nat
  {Nat.add(hold_max(b, t), hold_release(b, hold_max(b, t))) == b : Nat}

def hold_conserved(b, t):
  match t:
    case 0n:
      tc_sub_zero(b)
    case 1n+tp:
      tc_min_sub_conserved(b, tp)

# ============================================================================
# 6. Content-Block Lifecycle (block-structured message stream)
# ============================================================================

type BlockKind is Data:
  BkNone{}
  BkThinking{}
  BkText{}
  BkTool{}

# Stop events owed when the open block ends.
def block_close_events(open: BlockKind) -> Nat:
  match open:
    case BkNone{}:
      0n
    case _:
      1n

def block_close(open: BlockKind) -> BlockKind:
  match open:
    case _:
      BkNone{}

def block_is_tool(k: BlockKind) -> Bool:
  match k:
    case BkTool{}:
      True{}
    case _:
      False{}

# Text of kind `want` arrives while `open` is the open block: must a new block start? A call's
# header always starts its own block (`want` is BkTool for a header only).
def block_must_open(want: BlockKind, open: BlockKind) -> Bool:
  match want open:
    case BkNone{} _:
      False{}
    case BkTool{} _:
      True{}
    case BkThinking{} BkThinking{}:
      False{}
    case BkText{} BkText{}:
      False{}
    case _ _:
      True{}

# LAW 6.1: Closing is idempotent: a closed block owes no second stop event.
law block_close_idempotent:
  for k: BlockKind
  {block_close_events(block_close(k)) == 0n : Nat}

def block_close_idempotent(k):
  match k:
    case BkNone{}: {==}
    case BkThinking{}: {==}
    case BkText{}: {==}
    case BkTool{}: {==}

# LAW 6.2: More text of the open block's kind continues it (no new block).
law block_same_kind_continues:
  for +k: BlockKind
  {Bool.and(block_must_open(k, k), Bool.not(block_is_tool(k))) == False{} : Bool}

def block_same_kind_continues(k):
  match k:
    case BkNone{}: {==}
    case BkThinking{}: {==}
    case BkText{}: {==}
    case BkTool{}: {==}

# LAW 6.3: Every call header starts a block of its own.
law block_tool_always_opens:
  for k: BlockKind
  {block_must_open(BkTool{}, k) == True{} : Bool}

def block_tool_always_opens(k):
  match k:
    case BkNone{}: {==}
    case BkThinking{}: {==}
    case BkText{}: {==}
    case BkTool{}: {==}

# LAW 6.4: Nothing opens a block for no content.
law block_none_never_opens:
  for k: BlockKind
  {block_must_open(BkNone{}, k) == False{} : Bool}

def block_none_never_opens(k):
  match k:
    case BkNone{}: {==}
    case BkThinking{}: {==}
    case BkText{}: {==}
    case BkTool{}: {==}

# ============================================================================
# 7. Slice Bounds (chat-template renderer)
# ============================================================================

# A slice bound of a forward slice over `len` items: `mag` counted from the end when `from_end`,
# else from the start; clamped into the sequence.
def slice_bound(from_end: Bool, mag: Nat, len: Nat) -> Nat:
  match from_end:
    case True{}:
      Nat.sub(len, mag)
    case False{}:
      Nat.min(mag, len)

# LAW 7.1: A slice bound never passes the end of the sequence.
law slice_bound_in_range:
  for from_end: Bool
  for +mag: Nat
  for +len: Nat
  {Nat.is_le(slice_bound(from_end, mag, len), len) == True{} : Bool}

def slice_bound_in_range(from_end, mag, len):
  match from_end:
    case True{}:
      tc_sub_le_self(len, mag)
    case False{}:
      tc_min_le_r(mag, len)


# ============================================================================
# 8. Placing a forced call
# ============================================================================

# Where a forced call's opening goes.
type ForcePlace is Data:
  FpNone{}      # nothing is forced
  FpPlace{}     # the call's opening ends the prompt
  FpInReasoning{}  # forced, but the prompt ends inside the reasoning: noted as not applied
  FpCannotWrite{}  # forced, but the format cannot write the opening: noted as not applied

# A forced call's opening ends the prompt, unless the prompt ends inside the model's reasoning or
# the model's format cannot write the opening (`can_place` false, e.g. a call without a name).
def tool_force_place(forces: Bool, reasoning_open: Bool, can_place: Bool) -> ForcePlace:
  match forces:
    case False{}:
      FpNone{}
    case True{}:
      match reasoning_open:
        case True{}:
          FpInReasoning{}
        case False{}:
          match can_place:
            case True{}:
              FpPlace{}
            case False{}:
              FpCannotWrite{}

def force_is_placed(p: ForcePlace) -> Bool:
  match p:
    case FpPlace{}:
      True{}
    case _:
      False{}

def force_is_none(p: ForcePlace) -> Bool:
  match p:
    case FpNone{}:
      True{}
    case _:
      False{}

# promise: a call's opening is put into the prompt only when the call is forced, the prompt is not inside reasoning, and the format can write it
# harm: an opening placed inside reasoning or in a form the model cannot continue would garble the reply
law force_placed_only_when_possible:
  for +f: Bool
  for +r: Bool
  for +c: Bool
  {Bool.or(Bool.not(force_is_placed(tool_force_place(f, r, c))), Bool.and(f, Bool.and(Bool.not(r), c))) == True{} : Bool}

def force_placed_only_when_possible(f, r, c):
  match f r c:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# promise: a forced call is either placed or reported as not applied, never silently dropped
law force_never_silently_dropped:
  for +f: Bool
  for r: Bool
  for c: Bool
  {Bool.or(Bool.not(f), Bool.not(force_is_none(tool_force_place(f, r, c)))) == True{} : Bool}

def force_never_silently_dropped(f, r, c):
  match f r c:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# promise: nothing is placed or noted when no call is forced
law force_none_when_not_forced:
  for r: Bool
  for c: Bool
  {tool_force_place(False{}, r, c) == FpNone{} : ForcePlace}

def force_none_when_not_forced(r, c):
  {==}

# promise: requiring a tool call when no tools are provided is always refused
# catches: tool_choice_valid returns True for TcRequired when tools_given is False
law tool_choice_required_needs_tools:
  {tool_choice_valid(TcRequired{}, False{}) == False{} : Bool}

def tool_choice_required_needs_tools():
  {==}

# promise: forcing a named tool when no tools are provided is always refused
# catches: tool_choice_valid returns True for TcNamed when tools_given is False
law tool_choice_named_needs_tools:
  {tool_choice_valid(TcNamed{}, False{}) == False{} : Bool}

def tool_choice_named_needs_tools():
  {==}

# LAW 5.5: hold_max preserves at least min(buf_len, tag_len - 1) bytes for partial tag matching.
# promise: hold_max keeps at least min(b, t) bytes when the tag has length 1+t
# catches: hold_max always returns 0n instead of holding partial tag prefixes
law hold_max_keeps_partial_tag:
  for +b: Nat
  for +t: Nat
  {Nat.is_le(Nat.min(b, t), hold_max(b, 1n+t)) == True{} : Bool}

def hold_max_keeps_partial_tag(b, t):
  Equal.sym(Bool, True{}, Nat.is_le(Nat.min(b, t), Nat.min(b, t)), tc_le_refl(Nat.min(b, t)))
