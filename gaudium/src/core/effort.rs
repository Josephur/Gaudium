//! Rust Shell wrapper for the reasoning-effort decisions of `verified/effort.bend`.
//!
//! A level name is text, which the verified core cannot hold: here a name becomes its rank on
//! the effort scale and a rank becomes its name again. What a request's effort turns into, and
//! which levels a model advertises, are decided by the verified core; nothing here decides.

use crate::verified::{
    effort_advertised, effort_any, effort_number_plan, effort_off_by_level, effort_plan,
};
use crate::verified::{EffortAsk, EffortNumber, EffortPlan, EffortSet};

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

/// How a model's chat template takes reasoning off and efforts given as numbers (found by
/// rendering it, like [`EffortLevels`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffortForm {
    /// The template accepts the off level (`reasoning_effort` "none") and tells it apart.
    pub off_level: bool,
    /// The template accepts a number as `reasoning_effort` and tells it apart.
    pub numbers: bool,
    /// The template reads the reasoning flag (`enable_thinking`): it renders differently with it
    /// on and off.
    pub reads_flag: bool,
}

/// What a `reasoning_effort` given as a number does to the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberEffort {
    /// The template is given the number; reasoning on unless it is zero.
    Pass { thinks: bool },
    /// Refused: the template takes no number.
    Refuse,
}

impl EffortForm {
    /// Whether a request whose reasoning is off (`off`) also gives the template its off level
    /// (`effort_set`: the template already has an effort from the request).
    pub fn off_by_level(self, off: bool, effort_set: bool) -> bool {
        effort_off_by_level(off, effort_set, self.off_level, self.reads_flag)
    }

    /// What a numeric effort does (`zero`: the number is 0).
    pub fn number(self, zero: bool) -> NumberEffort {
        match effort_number_plan(self.numbers, zero) {
            EffortNumber::EnPass(thinks) => NumberEffort::Pass { thinks },
            EffortNumber::EnRefuse => NumberEffort::Refuse,
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

    #[test]
    fn reasoning_off_and_numbers_follow_the_template_form() {
        // a template that only an effort level can turn off (it ignores the flag)
        let levels_only = EffortForm {
            off_level: true,
            numbers: true,
            reads_flag: false,
        };
        assert!(levels_only.off_by_level(true, false));
        assert!(!levels_only.off_by_level(true, true) && !levels_only.off_by_level(false, false));
        // one that reads the flag keeps using it
        let flag = EffortForm {
            reads_flag: true,
            ..levels_only
        };
        assert!(!flag.off_by_level(true, false));
        assert!(!EffortForm::default().off_by_level(true, false));
        assert_eq!(levels_only.number(false), NumberEffort::Pass { thinks: true });
        assert_eq!(levels_only.number(true), NumberEffort::Pass { thinks: false });
        assert_eq!(EffortForm::default().number(false), NumberEffort::Refuse);
    }
}
