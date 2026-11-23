//! Rust Shell wrapper for the reasoning-effort decisions of `verified/effort.bend`.
//!
//! A level name is text, which the verified core cannot hold: here a name becomes its rank on
//! the effort scale and a rank becomes its name again. What a request's effort turns into, and
//! which levels a model advertises, are decided by the verified core; nothing here decides.

use crate::verified::{effort_advertised, effort_any, effort_plan};
use crate::verified::{EffortAsk, EffortPlan, EffortSet};

/// The effort scale, low to high: a level's rank is its index.
pub const SCALE: [&str; 6] = ["minimal", "low", "medium", "high", "xhigh", "max"];

/// The request's word for no reasoning at all.
pub const OFF: &str = "none";

/// The levels of the scale a model's chat template accepts for `reasoning_effort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffortLevels([bool; 6]);

/// What a request's `reasoning_effort` does to the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    /// Reasoning off (`enable_thinking: false`), no level.
    Off,
    /// Reasoning on; the template is given this level.
    Level(&'static str),
    /// Reasoning on; no level (the template's own default).
    ThinkOnly,
    /// Refused: a name that is not one of the model's levels.
    Refuse,
}

impl EffortLevels {
    /// No effort setting.
    pub const NONE: Self = Self([false; 6]);

    /// The levels a predicate accepts, asked for each level of the scale.
    pub fn from_fn(mut accepts: impl FnMut(&'static str) -> bool) -> Self {
        let mut l = [false; 6];
        for (slot, name) in l.iter_mut().zip(SCALE) {
            *slot = accepts(name);
        }
        Self(l)
    }

    /// A comma-separated list of level names (`none` or an empty list: no effort setting).
    pub fn parse(list: &str) -> Result<Self, String> {
        let mut l = [false; 6];
        for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            if name == OFF {
                continue;
            }
            let Some(r) = rank(name) else {
                return Err(format!(
                    "unknown effort level '{name}' (the levels are {}, or none)",
                    SCALE.join(", ")
                ));
            };
            l[r] = true;
        }
        Ok(Self(l))
    }

    fn set(self) -> EffortSet {
        let [a, b, c, d, e, f] = self.0;
        EffortSet::EfSet(a, b, c, d, e, f)
    }

    /// The template has an effort setting.
    pub fn supported(self) -> bool {
        effort_any(self.set())
    }

    /// The level names the model advertises, low to high.
    pub fn advertised(self) -> Vec<&'static str> {
        (0..SCALE.len())
            .filter(|&r| effort_advertised(self.set(), r as u64))
            .map(|r| SCALE[r])
            .collect()
    }

    /// What a request's `reasoning_effort` asks of this model.
    pub fn plan(self, requested: &str) -> Effort {
        let ask = if requested == OFF {
            EffortAsk::EaOff
        } else {
            match rank(requested) {
                Some(r) => EffortAsk::EaLevel(r as u64),
                None => EffortAsk::EaOther,
            }
        };
        match effort_plan(self.set(), ask) {
            EffortPlan::EpThinkOff => Effort::Off,
            // an offered level (effort_plan_offered), so a rank of the scale
            EffortPlan::EpLevel(r) => Effort::Level(SCALE[r as usize]),
            EffortPlan::EpThinkOnly => Effort::ThinkOnly,
            EffortPlan::EpRefuse => Effort::Refuse,
        }
    }
}

fn rank(name: &str) -> Option<usize> {
    SCALE.iter().position(|&n| n == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_map_onto_the_offered_levels() {
        // the three levels of the Qwen3.8 template
        let q = EffortLevels::parse("low, medium,xhigh").unwrap();
        assert!(q.supported());
        assert_eq!(q.advertised(), ["low", "medium", "xhigh"]);
        // today's choices: the nearest offered level, the higher one on a tie
        for (asked, want) in [
            ("none", Effort::Off),
            ("minimal", Effort::Level("low")),
            ("low", Effort::Level("low")),
            ("medium", Effort::Level("medium")),
            ("high", Effort::Level("xhigh")),
            ("xhigh", Effort::Level("xhigh")),
            ("max", Effort::Level("xhigh")),
            ("turbo", Effort::Refuse),
            ("", Effort::Refuse),
        ] {
            assert_eq!(q.plan(asked), want, "{asked}");
        }
        // one level: everything on the scale goes to it
        let one = EffortLevels::parse("high").unwrap();
        assert!(SCALE.iter().all(|n| one.plan(n) == Effort::Level("high")));
        // no effort setting: reasoning on or off, never a level, never refused
        let none = EffortLevels::parse("none").unwrap();
        assert_eq!(none, EffortLevels::NONE);
        assert!(!none.supported() && none.advertised().is_empty());
        assert_eq!(none.plan("high"), Effort::ThinkOnly);
        assert_eq!(none.plan("turbo"), Effort::ThinkOnly);
        assert_eq!(none.plan("none"), Effort::Off);
        assert!(EffortLevels::parse("low,huge").is_err());
        assert_eq!(
            EffortLevels::from_fn(|n| n.ends_with("high")).advertised(),
            ["high", "xhigh"]
        );
    }
}
