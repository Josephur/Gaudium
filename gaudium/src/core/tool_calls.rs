//! Rust Shell wrappers for the tool-calling decisions of `verified/tool_calls.bend`.
//!
//! The chat server's tool-choice policy, the tool-call stream lifecycle, the end / finish / stop
//! reason selection, the text parser's hold-back arithmetic, the content-block lifecycle and the
//! template renderer's slice bounds all delegate to the verified core; nothing here decides.

use crate::verified::{
    block_close, block_close_events, block_must_open, call_accepts, call_after_finish,
    call_is_streaming, call_step, finish_reason, gen_end, hold_max, hold_release, slice_bound,
    stop_reason, tool_choice_forces, tool_choice_offers, tool_choice_valid, tool_force_applies,
    tool_force_place, tool_thinking_allowed,
};
pub use crate::verified::{
    BlockKind, CallEvent, CallPhase, FinishReason, ForcePlace, GenEnd, StopReason, ToolChoice,
};

/// What a request's `tool_choice` and tool list mean for its prompt and its reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolPolicy {
    /// The tools are shown to the model and its tool calls are parsed.
    pub offers: bool,
    /// The reply must be a tool call (its opening is put into the prompt).
    pub forces: bool,
}

impl ToolPolicy {
    /// `None`: the choice needs a non-empty tool list (the request is refused).
    pub fn decide(choice: ToolChoice, tools_given: bool, continue_final: bool) -> Option<Self> {
        tool_choice_valid(choice, tools_given).then(|| Self {
            offers: tool_choice_offers(choice, tools_given),
            forces: tool_force_applies(tool_choice_forces(choice, tools_given), continue_final),
        })
    }

    /// Whether the model may reason before it answers.
    pub fn thinking(&self, wants_thinking: bool) -> bool {
        tool_thinking_allowed(wants_thinking, self.forces)
    }

    /// Where a forced call's opening goes: the prompt ends inside the model's reasoning
    /// (`reasoning_open`), or the model's format can write the opening (`can_place`).
    pub fn placement(&self, reasoning_open: bool, can_place: bool) -> ForcePlace {
        tool_force_place(self.forces, reasoning_open, can_place)
    }
}

/// Checked lifecycle of the tool calls of one reply: a call's header precedes its argument
/// fragments, a call whose header went out is finished and never taken back, and the call limit
/// stops the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallStream {
    phase: CallPhase,
    done: u64,
    limit: Option<u64>,
}

impl Default for CallStream {
    fn default() -> Self {
        Self::new()
    }
}

impl CallStream {
    pub const fn new() -> Self {
        Self {
            phase: CallPhase::CpText,
            done: 0,
            limit: None,
        }
    }

    /// At most `limit` calls (`parallel_tool_calls` false: one).
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = Some(limit);
    }

    /// Applies `event` when the lifecycle accepts it; `false` leaves the phase as it was.
    pub fn apply(&mut self, event: CallEvent) -> bool {
        let ok = call_accepts(self.phase, event);
        if ok {
            self.phase = call_step(self.phase, event);
            if event == CallEvent::CeFinish {
                self.phase =
                    call_after_finish(self.done, self.limit.is_some(), self.limit.unwrap_or(0));
                self.done = self.done.saturating_add(1);
            }
        }
        ok
    }

    /// Whether `event` would be accepted now.
    pub fn accepts(&self, event: CallEvent) -> bool {
        call_accepts(self.phase, event)
    }

    /// A call's header is out and its arguments stream.
    pub fn is_streaming(&self) -> bool {
        call_is_streaming(self.phase)
    }

    /// The call limit ended the stream.
    pub fn stopped(&self) -> bool {
        self.phase == CallPhase::CpStopped
    }

    /// Finished calls: the index of the call being read.
    pub fn index(&self) -> usize {
        self.done as usize
    }
}

/// How a generation ended, from what the engine and the parser report.
pub fn generation_end(stop_string: bool, call_limit: bool, eos: bool, cancelled: bool) -> GenEnd {
    gen_end(stop_string, call_limit, eos, cancelled)
}

/// `finish_reason` of a chat completion.
pub fn finish_reason_name(end: GenEnd, has_calls: bool) -> &'static str {
    match finish_reason(end, has_calls) {
        FinishReason::FrStop => "stop",
        FinishReason::FrLength => "length",
        FinishReason::FrToolCalls => "tool_calls",
        FinishReason::FrCancelled => "cancelled",
    }
}

/// `stop_reason` of a block-structured message.
pub fn stop_reason_name(end: GenEnd, has_calls: bool) -> &'static str {
    match stop_reason(end, has_calls) {
        StopReason::SrEndTurn => "end_turn",
        StopReason::SrMaxTokens => "max_tokens",
        StopReason::SrStopSequence => "stop_sequence",
        StopReason::SrToolUse => "tool_use",
    }
}

/// The longest tail of a `buf_len`-byte buffer that may still be held back as the start of a
/// `tag_len`-byte tag.
pub fn hold_back_limit(buf_len: usize, tag_len: usize) -> usize {
    hold_max(buf_len as u64, tag_len as u64) as usize
}

/// Bytes of a `buf_len`-byte buffer released when `held` stay back.
pub fn release_len(buf_len: usize, held: usize) -> usize {
    hold_release(buf_len as u64, held as u64) as usize
}

/// A bound of a forward slice over `len` items: `index` as written (negative: from the end),
/// clamped into the sequence.
pub fn forward_slice_bound(index: i64, len: usize) -> usize {
    slice_bound(index < 0, index.unsigned_abs(), len as u64) as usize
}

/// The open content block of a block-structured message stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTrack {
    open: BlockKind,
}

impl Default for BlockTrack {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockTrack {
    pub const fn new() -> Self {
        Self {
            open: BlockKind::BkNone,
        }
    }

    pub fn open(&self) -> BlockKind {
        self.open
    }

    /// Text of kind `want` arrives (a tool kind: a call's header): whether a new block starts.
    pub fn must_open(&self, want: BlockKind) -> bool {
        block_must_open(want, self.open)
    }

    /// Records that a block of `kind` is now open (the caller closed the previous one).
    pub fn opened(&mut self, kind: BlockKind) {
        self.open = kind;
    }

    /// Closes the open block; `true` when a stop event is owed for it.
    pub fn close(&mut self) -> bool {
        let owed = block_close_events(self.open) > 0;
        self.open = block_close(self.open);
        owed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrappers_follow_the_verified_core() {
        // policy: named choice without tools is refused; a forced call turns reasoning off
        assert_eq!(ToolPolicy::decide(ToolChoice::TcNamed, false, false), None);
        let forced = ToolPolicy::decide(ToolChoice::TcRequired, true, false).unwrap();
        assert!(forced.offers && forced.forces && !forced.thinking(true));
        let auto = ToolPolicy::decide(ToolChoice::TcAuto, true, false).unwrap();
        assert!(auto.offers && !auto.forces && auto.thinking(true));
        let none = ToolPolicy::decide(ToolChoice::TcNone, true, false).unwrap();
        assert!(!none.offers && !none.forces);
        assert!(
            !ToolPolicy::decide(ToolChoice::TcRequired, true, true)
                .unwrap()
                .forces
        );

        // stream: fragments need a header; the limit stops it
        let mut s = CallStream::new();
        s.set_limit(1);
        assert!(!s.apply(CallEvent::CeFragment) && !s.apply(CallEvent::CeHeader));
        assert!(s.apply(CallEvent::CeOpen) && !s.apply(CallEvent::CeFragment));
        assert!(s.apply(CallEvent::CeHeader) && s.is_streaming() && !s.apply(CallEvent::CeAbandon));
        assert!(s.apply(CallEvent::CeFragment) && s.apply(CallEvent::CeFinish));
        assert!(s.stopped() && s.index() == 1 && !s.apply(CallEvent::CeOpen));
        let mut free = CallStream::new();
        for _ in 0..3 {
            assert!(
                free.apply(CallEvent::CeOpen)
                    && free.apply(CallEvent::CeHeader)
                    && free.apply(CallEvent::CeFinish)
            );
        }
        assert!(!free.stopped() && free.index() == 3);

        // reasons
        let end = generation_end(false, true, false, true);
        assert_eq!(finish_reason_name(end, true), "tool_calls");
        assert_eq!(
            stop_reason_name(generation_end(true, false, false, true), false),
            "stop_sequence"
        );
        assert_eq!(
            finish_reason_name(generation_end(false, false, false, true), true),
            "cancelled"
        );
        assert_eq!(
            stop_reason_name(generation_end(false, false, false, false), true),
            "max_tokens"
        );

        // arithmetic
        assert_eq!(
            (
                hold_back_limit(10, 4),
                hold_back_limit(2, 4),
                hold_back_limit(5, 0)
            ),
            (3, 2, 0)
        );
        assert_eq!(release_len(10, 3), 7);
        assert_eq!(
            (
                forward_slice_bound(-2, 5),
                forward_slice_bound(-9, 5),
                forward_slice_bound(9, 5)
            ),
            (3, 0, 5)
        );

        // blocks
        let mut b = BlockTrack::new();
        assert!(!b.close() && b.must_open(BlockKind::BkText));
        b.opened(BlockKind::BkText);
        assert!(!b.must_open(BlockKind::BkText) && b.must_open(BlockKind::BkTool));
        assert!(b.close() && !b.close());
    }
}
