//! Rust Shell wrappers for the A2UI negotiation decisions of `verified/a2ui.bend`.
//!
//! Whether a request gets the A2UI instructions, where they go in the conversation and how many
//! catalog identifiers are compared all delegate to the verified core; nothing here decides.

pub use crate::verified::{A2uiDecision, A2uiPlace};
use crate::verified::{
    a2ui_decide, a2ui_ids_scanned, a2ui_injects, a2ui_msg_count, a2ui_notes, a2ui_place,
    a2ui_prompt_loads, a2ui_prompt_pick, a2ui_prompt_size_ok, a2ui_prompt_warns, a2ui_refuses,
    A2PromptPick,
};

/// What a request's A2UI declaration means on this server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct A2uiPolicy {
    decision: A2uiDecision,
}

impl A2uiPolicy {
    /// `requested`: the request declares renderer capabilities; `enabled`: the server offers
    /// A2UI; `common`: one of the request's catalogs is one the server writes for.
    pub fn decide(requested: bool, enabled: bool, common: bool) -> Self {
        Self {
            decision: a2ui_decide(requested, enabled, common),
        }
    }

    /// The instructions are added to the conversation.
    pub fn injects(&self) -> bool {
        a2ui_injects(self.decision)
    }

    /// The request is refused.
    pub fn refuses(&self) -> bool {
        a2ui_refuses(self.decision)
    }

    /// The request runs as plain text and the reply notes the declaration was not applied.
    pub fn notes(&self) -> bool {
        a2ui_notes(self.decision)
    }
}

/// Where the instructions go: appended to a leading system message, or a new leading one.
pub fn place(first_is_system: bool) -> A2uiPlace {
    a2ui_place(first_is_system)
}

/// Messages in the conversation once the instructions are in.
pub fn message_count(messages: usize, place: A2uiPlace) -> usize {
    a2ui_msg_count(messages as u64, place) as usize
}

/// How many of `listed` catalog identifiers are compared.
pub fn ids_scanned(listed: usize, limit: usize) -> usize {
    a2ui_ids_scanned(listed as u64, limit as u64) as usize
}

/// What a look at the instructions file does (`verified/a2ui.bend` section 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptPick(A2PromptPick);

impl PromptPick {
    /// `changed`: the file's stamp differs from the last one seen; `valid`: its text passed every
    /// check.
    pub fn decide(changed: bool, valid: bool) -> Self {
        Self(a2ui_prompt_pick(changed, valid))
    }
    /// The new text replaces the served one.
    pub fn loads(self) -> bool {
        a2ui_prompt_loads(self.0)
    }
    /// The file was not taken and the server says why.
    pub fn warns(self) -> bool {
        a2ui_prompt_warns(self.0)
    }
}

/// The instructions file has something in it and is no larger than `cap` bytes.
pub fn prompt_size_ok(len: u64, cap: u64) -> bool {
    a2ui_prompt_size_ok(len, cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_pick_wrappers_report_the_verified_decision() {
        assert!(PromptPick::decide(true, true).loads());
        let bad = PromptPick::decide(true, false);
        assert!(bad.warns() && !bad.loads());
        for v in [true, false] {
            let same = PromptPick::decide(false, v);
            assert!(!same.loads() && !same.warns());
        }
        assert!(!prompt_size_ok(0, 10) && prompt_size_ok(10, 10) && !prompt_size_ok(11, 10));
    }

    #[test]
    fn the_wrappers_report_the_verified_decision() {
        let on = A2uiPolicy::decide(true, true, true);
        assert!(on.injects() && !on.refuses() && !on.notes());
        let refused = A2uiPolicy::decide(true, false, true);
        assert!(refused.refuses() && !refused.injects());
        let plain = A2uiPolicy::decide(true, true, false);
        assert!(plain.notes() && !plain.injects() && !plain.refuses());
        let untouched = A2uiPolicy::decide(false, true, true);
        assert!(!untouched.injects() && !untouched.refuses() && !untouched.notes());

        assert_eq!(message_count(3, place(true)), 3);
        assert_eq!(message_count(3, place(false)), 4);
        assert_eq!(ids_scanned(100, 64), 64);
        assert_eq!(ids_scanned(2, 64), 2);
    }
}
