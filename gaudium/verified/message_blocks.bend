# Slice — Verified Message-Block Reader
# Formally specifies and proves how a reply written as typed message blocks is read while it
# streams. Such a model opens each block itself: a header marker (with an optional name), a
# content-type marker, the block's text, and an end marker; a final marker ends the reply:
#   <header>[name]<thinking> ... <end> <header>[name]<text> ... <end>
#   <header>NAME<call>{"name": ..., "args": {...}} <end>  ...  <reply end>
# The marker texts and the call's JSON stay in the Rust shell (`server::blocks`), which turns
# the generated text into the events below and carries out the actions; this file decides where
# every piece of text goes and when a call is read.

import Base

# ============================================================================
# 1. Types
# ============================================================================

# Where the reader is.
type MbState is Data:
  MsOutside{}   # between blocks
  MsHeader{}    # after a header, before its content type (the text is the block's name)
  MsThink{}     # inside a reasoning block
  MsText{}      # inside a text block
  MsCall{}      # inside a tool-call block
  MsDone{}      # the reply is over

# What arrives next.
type MbEvent is Data:
  MeHeader{}      # a model block's header
  MeOpenThink{}   # the reasoning content type
  MeOpenText{}    # a text content type (text, a tool error, a call written as text)
  MeOpenCall{}    # the tool-call content type (offered tools only; else it is text)
  MeEnd{}         # the end of a block
  MeWords{}       # ordinary text
  MeStop{}        # the end of the reply (its final marker, or another role's header)
  MeFinish{}      # the end of generation

# What becomes of it.
type MbAct is Data:
  MaNone{}        # nothing
  MaReasoning{}   # the text is reasoning
  MaContent{}     # the text is visible content
  MaName{}        # the text is the block's name
  MaCallText{}    # the text belongs to the open call
  MaReadCall{}    # the call block ended: read as one call (given back as content if it is none)
  MaCutCall{}     # the call block was left open: read if it is one whole call, else dropped

# ============================================================================
# 2. The reader
# ============================================================================

def mb_next(s: MbState, e: MbEvent) -> MbState:
  match s e:
    case MsDone{} _:
      MsDone{}
    case _ MeStop{}:
      MsDone{}
    case _ MeFinish{}:
      MsDone{}
    case _ MeHeader{}:
      MsHeader{}
    case _ MeOpenThink{}:
      MsThink{}
    case _ MeOpenText{}:
      MsText{}
    case _ MeOpenCall{}:
      MsCall{}
    case _ MeEnd{}:
      MsOutside{}
    case st MeWords{}:
      st

# Reasoning text is reasoning when the request splits it out, else visible content.
def mb_think_act(split: Bool) -> MbAct:
  match split:
    case True{}:
      MaReasoning{}
    case False{}:
      MaContent{}

def mb_act(s: MbState, e: MbEvent, split: Bool) -> MbAct:
  match s e:
    case MsDone{} _:
      MaNone{}
    case MsCall{} MeWords{}:
      MaCallText{}
    case MsCall{} MeEnd{}:
      MaReadCall{}
    case MsCall{} _:
      MaCutCall{}
    case MsThink{} MeWords{}:
      mb_think_act(split)
    case MsText{} MeWords{}:
      MaContent{}
    case MsOutside{} MeWords{}:
      MaContent{}
    case MsHeader{} MeWords{}:
      MaName{}
    case _ _:
      MaNone{}

# ============================================================================
# 3. Observers
# ============================================================================

def mb_is_call(s: MbState) -> Bool:
  match s:
    case MsCall{}:
      True{}
    case _:
      False{}

def mb_is_words(e: MbEvent) -> Bool:
  match e:
    case MeWords{}:
      True{}
    case _:
      False{}

def mb_is_open_call(e: MbEvent) -> Bool:
  match e:
    case MeOpenCall{}:
      True{}
    case _:
      False{}

# The action reads (or drops) the open call.
def mb_closes_call(a: MbAct) -> Bool:
  match a:
    case MaReadCall{}:
      True{}
    case MaCutCall{}:
      True{}
    case _:
      False{}

# The action shows text to the client (as content, reasoning or a block name).
def mb_shows(a: MbAct) -> Bool:
  match a:
    case MaReasoning{}:
      True{}
    case MaContent{}:
      True{}
    case MaName{}:
      True{}
    case _:
      False{}

def mb_is_reasoning(a: MbAct) -> Bool:
  match a:
    case MaReasoning{}:
      True{}
    case _:
      False{}

def mb_is_none(a: MbAct) -> Bool:
  match a:
    case MaNone{}:
      True{}
    case _:
      False{}

# ============================================================================
# 3a. Start state and fresh buffers
# ============================================================================

# The reader's state when the prompt ends inside a reasoning block (`reasoning_open`), or between blocks.
def mb_start(reasoning_open: Bool) -> MbState:
  match reasoning_open:
    case True{}:
      MsThink{}
    case False{}:
      MsOutside{}

# What a transition starts empty: nothing, the block's name, or the call's text (and a new call
# in the reply's call lifecycle).
type MbFresh is Data:
  MfNone{}
  MfName{}
  MfCall{}

def mb_fresh_of(e: MbEvent, next: MbState) -> MbFresh:
  match e next:
    case MeHeader{} MsHeader{}:
      MfName{}
    case MeOpenCall{} MsCall{}:
      MfCall{}
    case _ _:
      MfNone{}

# What the event `e` read in state `s` starts empty.
def mb_fresh(+s: MbState, +e: MbEvent) -> MbFresh:
  mb_fresh_of(e, mb_next(s, e))

def mb_fresh_is_call(f: MbFresh) -> Bool:
  match f:
    case MfCall{}:
      True{}
    case _:
      False{}

def mb_fresh_is_name(f: MbFresh) -> Bool:
  match f:
    case MfName{}:
      True{}
    case _:
      False{}

def mb_is_header(s: MbState) -> Bool:
  match s:
    case MsHeader{}:
      True{}
    case _:
      False{}

def mb_is_done(s: MbState) -> Bool:
  match s:
    case MsDone{}:
      True{}
    case _:
      False{}

def mb_is_content(a: MbAct) -> Bool:
  match a:
    case MaContent{}:
      True{}
    case _:
      False{}

# ============================================================================
# 4. Laws
# ============================================================================

# promise: text inside an open tool-call block is never shown to the client as content, reasoning or a name
# harm: a partial call's JSON would leak into the visible reply
law mb_call_text_hidden:
  for +s: MbState
  for +e: MbEvent
  for +split: Bool
  {Bool.and(mb_is_call(s), mb_shows(mb_act(s, e, split))) == False{} : Bool}

def mb_call_text_hidden_think(split: Bool) -> {Bool.and(mb_is_call(MsThink{}), mb_shows(mb_act(MsThink{}, MeWords{}, split))) == False{} : Bool}:
  match split:
    case True{}: {==}
    case False{}: {==}

def mb_call_text_hidden(s, e, split):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}:
          mb_call_text_hidden_think(split)
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: every marker that arrives inside a tool-call block closes that call (it is read or dropped, never left open)
# harm: a call would stay open across a new block and swallow its text
law mb_call_closes_at_marker:
  for +s: MbState
  for +e: MbEvent
  for +split: Bool
  {Bool.or(Bool.not(mb_is_call(s)), Bool.or(mb_is_words(e), mb_closes_call(mb_act(s, e, split)))) == True{} : Bool}

def mb_call_closes_at_marker_think(split: Bool) -> {Bool.or(Bool.not(mb_is_call(MsThink{})), Bool.or(mb_is_words(MeWords{}), mb_closes_call(mb_act(MsThink{}, MeWords{}, split)))) == True{} : Bool}:
  match split:
    case True{}: {==}
    case False{}: {==}

def mb_call_closes_at_marker(s, e, split):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}:
          mb_call_closes_at_marker_think(split)
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: ordinary text never closes a tool call (a call is read only at a marker or at the end)
law mb_words_never_close_call:
  for +s: MbState
  for +e: MbEvent
  for +split: Bool
  {Bool.and(mb_is_words(e), mb_closes_call(mb_act(s, e, split))) == False{} : Bool}

def mb_words_never_close_call_think(split: Bool) -> {Bool.and(mb_is_words(MeWords{}), mb_closes_call(mb_act(MsThink{}, MeWords{}, split))) == False{} : Bool}:
  match split:
    case True{}: {==}
    case False{}: {==}

def mb_words_never_close_call(s, e, split):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}:
          mb_words_never_close_call_think(split)
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: a tool-call block is entered only through its own content-type marker, never by text
# harm: plain text would be parsed as a tool call
law mb_call_opens_by_marker:
  for +s: MbState
  for +e: MbEvent
  {Bool.or(mb_is_call(s), Bool.or(Bool.not(mb_is_call(mb_next(s, e))), mb_is_open_call(e))) == True{} : Bool}

def mb_call_opens_by_marker(s, e):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: reasoning is reported as reasoning only when the request splits reasoning out
law mb_reasoning_needs_split:
  for +s: MbState
  for +e: MbEvent
  for +split: Bool
  {Bool.or(Bool.not(mb_is_reasoning(mb_act(s, e, split))), split) == True{} : Bool}

def mb_reasoning_needs_split_think(split: Bool) -> {Bool.or(Bool.not(mb_is_reasoning(mb_act(MsThink{}, MeWords{}, split))), split) == True{} : Bool}:
  match split:
    case True{}: {==}
    case False{}: {==}

def mb_reasoning_needs_split(s, e, split):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}:
          mb_reasoning_needs_split_think(split)
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: once the reply is over nothing more is read from it, whatever arrives
law mb_done_absorbs:
  for e: MbEvent
  {mb_next(MsDone{}, e) == MsDone{} : MbState}

def mb_done_absorbs(e):
  match e:
    case MeHeader{}: {==}
    case MeOpenThink{}: {==}
    case MeOpenText{}: {==}
    case MeOpenCall{}: {==}
    case MeEnd{}: {==}
    case MeWords{}: {==}
    case MeStop{}: {==}
    case MeFinish{}: {==}

# promise: once the reply is over no text is routed anywhere
law mb_done_silent:
  for e: MbEvent
  for split: Bool
  {mb_act(MsDone{}, e, split) == MaNone{} : MbAct}

def mb_done_silent(e, split):
  match e:
    case MeHeader{}: {==}
    case MeOpenThink{}: {==}
    case MeOpenText{}: {==}
    case MeOpenCall{}: {==}
    case MeEnd{}: {==}
    case MeWords{}: {==}
    case MeStop{}: {==}
    case MeFinish{}: {==}

# promise: the end of generation always ends the reply
law mb_finish_ends:
  for s: MbState
  {mb_next(s, MeFinish{}) == MsDone{} : MbState}

def mb_finish_ends(s):
  match s:
    case MsOutside{}: {==}
    case MsHeader{}: {==}
    case MsThink{}: {==}
    case MsText{}: {==}
    case MsCall{}: {==}
    case MsDone{}: {==}

# promise: the end of the reply (its final marker or another role's header) always ends the reply
law mb_stop_ends:
  for s: MbState
  {mb_next(s, MeStop{}) == MsDone{} : MbState}

def mb_stop_ends(s):
  match s:
    case MsOutside{}: {==}
    case MsHeader{}: {==}
    case MsThink{}: {==}
    case MsText{}: {==}
    case MsCall{}: {==}
    case MsDone{}: {==}

# promise: finishing twice is the same as finishing once: the second finish routes nothing
law mb_finish_idempotent:
  for +s: MbState
  for split: Bool
  {mb_is_none(mb_act(mb_next(s, MeFinish{}), MeFinish{}, split)) == True{} : Bool}

def mb_finish_idempotent(s, split):
  match s:
    case MsOutside{}: {==}
    case MsHeader{}: {==}
    case MsThink{}: {==}
    case MsText{}: {==}
    case MsCall{}: {==}
    case MsDone{}: {==}

# ============================================================================
# 5. The call marker
# ============================================================================

# The tool-call content type opens a call only when the request offers tools; without tools the
# block is read as text (its JSON is plain content).
def mb_call_event(tools_on: Bool) -> MbEvent:
  match tools_on:
    case True{}:
      MeOpenCall{}
    case False{}:
      MeOpenText{}

# promise: without offered tools no block is ever read as a tool call
# harm: a reply to a request without tools would lose text into a call nobody asked for
law mb_no_tools_no_call:
  {mb_is_open_call(mb_call_event(False{})) == False{} : Bool}

def mb_no_tools_no_call():
  {==}

# promise: with offered tools the call marker always opens a tool call
law mb_tools_call_opens:
  {mb_call_event(True{}) == MeOpenCall{} : MbEvent}

def mb_tools_call_opens():
  {==}

# ----------------------------------------------------------------------------
# Laws of the start state and the fresh buffers
# ----------------------------------------------------------------------------

# promise: a new call (its text started empty, a call opened in the reply's lifecycle) begins only at the call marker, and the reader is then inside the call.
# harm: plain text or another block would open a tool call in the client's stream
law mb_call_fresh_only_on_open:
  for +s: MbState
  for +e: MbEvent
  {Bool.and(mb_fresh_is_call(mb_fresh(s, e)), Bool.not(Bool.and(mb_is_open_call(e), mb_is_call(mb_next(s, e))))) == False{} : Bool}

def mb_call_fresh_only_on_open(s, e):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: a tool-call block the reader enters always starts with empty call text and a new call.
# harm: a call would carry text of an earlier block, or arrive without its opening in the stream
law mb_call_entered_fresh:
  for +s: MbState
  for +e: MbEvent
  {Bool.and(Bool.and(Bool.not(mb_is_call(s)), mb_is_call(mb_next(s, e))), Bool.not(mb_fresh_is_call(mb_fresh(s, e)))) == False{} : Bool}

def mb_call_entered_fresh(s, e):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: a block header the reader enters always starts with an empty name.
# harm: a call would be named with the name of an earlier block
law mb_name_entered_fresh:
  for +s: MbState
  for +e: MbEvent
  {Bool.and(Bool.and(Bool.not(mb_is_header(s)), mb_is_header(mb_next(s, e))), Bool.not(mb_fresh_is_name(mb_fresh(s, e)))) == False{} : Bool}

def mb_name_entered_fresh(s, e):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: a block's name is started empty only when the reader enters a header.
# harm: the name of the open block would be lost in the middle of the block
law mb_name_fresh_only_on_header:
  for +s: MbState
  for +e: MbEvent
  {Bool.and(mb_fresh_is_name(mb_fresh(s, e)), Bool.not(mb_is_header(mb_next(s, e)))) == False{} : Bool}

def mb_name_fresh_only_on_header(s, e):
  match s:
    case MsOutside{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsHeader{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsThink{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsText{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsCall{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}
    case MsDone{}:
      match e:
        case MeHeader{}: {==}
        case MeOpenThink{}: {==}
        case MeOpenText{}: {==}
        case MeOpenCall{}: {==}
        case MeEnd{}: {==}
        case MeWords{}: {==}
        case MeStop{}: {==}
        case MeFinish{}: {==}

# promise: the reader never starts inside a tool call or after the end of the reply.
# harm: the first words of a reply would be read as a call, or dropped as if the reply were over
law mb_start_open:
  for +r: Bool
  {Bool.or(mb_is_call(mb_start(r)), mb_is_done(mb_start(r))) == False{} : Bool}

def mb_start_open(r):
  match r:
    case True{}: {==}
    case False{}: {==}

# promise: when the prompt ends inside reasoning and the request splits reasoning out, the reply's first words are never shown as content.
# harm: hidden reasoning would leak into the visible reply
law mb_start_reasoning_not_content:
  for +r: Bool
  {Bool.and(r, mb_is_content(mb_act(mb_start(r), MeWords{}, True{}))) == False{} : Bool}

def mb_start_reasoning_not_content(r):
  match r:
    case True{}: {==}
    case False{}: {==}
