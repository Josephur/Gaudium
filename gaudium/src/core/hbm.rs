//! Rust Shell wrappers for HBM pre-allocation planning & 5-category byte-exact ledger accounting.
//!
//! All usable-capacity clamping, fit/shortfall partitioning, ledger conservation, and
//! context-over-prefix-slot priority rules delegate to `verified/hbm_budget.bend`.

use crate::verified::{
    hbm_check_fit, hbm_fit_ok, hbm_fit_reconstruct_lhs, hbm_fit_reconstruct_rhs,
    hbm_grant_prefix_slots, hbm_ledger_add, hbm_ledger_empty, hbm_ledger_matches, hbm_ledger_total,
    hbm_usable, HbmCat, HbmFit, HbmLedger,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HbmError {
    PlanOverflow {
        planned: u64,
        usable: u64,
        shortfall: u64,
    },
}

/// Computes usable HBM bytes (`total - (driver_base + spare)`), saturating safely at 0.
pub fn usable_hbm_bytes(total: u64, driver_base: u64, spare: u64) -> u64 {
    hbm_usable(total, driver_base, spare)
}

/// Checks whether `planned` bytes fit within `usable` HBM bytes, returning margin or exact shortfall.
pub fn check_hbm_fit(planned: u64, usable: u64) -> Result<u64, HbmError> {
    let fit = hbm_check_fit(planned, usable);
    match fit {
        HbmFit::HbmFits(margin) => Ok(margin),
        HbmFit::HbmShort(shortfall) => Err(HbmError::PlanOverflow {
            planned,
            usable,
            shortfall,
        }),
    }
}

/// Grants prefix-reuse HBM snapshot slots after reserving base context (`hbm_plan.hpp`).
/// Context ALWAYS wins: returns 0 if `base_planned > usable` or `slot_bytes == 0`.
pub fn grant_prefix_slots(
    requested_slots: u64,
    slot_bytes: u64,
    base_planned: u64,
    usable: u64,
) -> u64 {
    hbm_grant_prefix_slots(requested_slots, slot_bytes, base_planned, usable)
}

/// Checked 5-category HBM allocation ledger (mirrors `hbm_ledger.hpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HbmLedgerGuard {
    ledger: HbmLedger,
}

impl Default for HbmLedgerGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl HbmLedgerGuard {
    pub fn new() -> Self {
        Self {
            ledger: hbm_ledger_empty(),
        }
    }

    pub fn record(&mut self, cat: HbmCat, bytes: u64) {
        self.ledger = hbm_ledger_add(self.ledger, cat, bytes);
    }

    pub fn total_bytes(&self) -> u64 {
        hbm_ledger_total(self.ledger)
    }

    pub fn breakdown(&self) -> (u64, u64, u64, u64, u64) {
        let HbmLedger::HbmLedger(w, kv, ck, sc, ot) = self.ledger;
        (w, kv, ck, sc, ot)
    }

    pub fn matches(
        &self,
        weights: u64,
        kv: u64,
        checkpoints: u64,
        scratch: u64,
        other: u64,
    ) -> bool {
        hbm_ledger_matches(self.ledger, weights, kv, checkpoints, scratch, other)
    }
}

/// Exposes raw verified `HbmFit` status check.
pub fn fit_is_ok(fit: HbmFit) -> bool {
    hbm_fit_ok(fit)
}

/// Exposes raw verified fit/shortfall conservation sides (`lhs == rhs` proven in Bend 2).
pub fn fit_conservation_sides(fit: HbmFit, planned: u64, usable: u64) -> (u64, u64) {
    (
        hbm_fit_reconstruct_lhs(fit, planned, usable),
        hbm_fit_reconstruct_rhs(fit, planned, usable),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    #[test]
    fn hl225_96gb_usable_and_fit_partition() {
        let total = 96 * GIB;
        let driver = 4 * GIB;
        let spare = GIB;
        let usable = usable_hbm_bytes(total, driver, spare);
        assert_eq!(usable, 91 * GIB);

        // Underflow protection when driver + spare > total
        assert_eq!(usable_hbm_bytes(10 * GIB, 12 * GIB, GIB), 0);

        // Exact fit with margin
        let planned_ok = 82 * GIB;
        let margin = check_hbm_fit(planned_ok, usable).unwrap();
        assert_eq!(margin, 9 * GIB);
        let fit = hbm_check_fit(planned_ok, usable);
        assert!(fit_is_ok(fit));
        let (lhs, rhs) = fit_conservation_sides(fit, planned_ok, usable);
        assert_eq!(lhs, rhs);
        assert_eq!(lhs, usable);

        // Exact shortfall on overflow
        let planned_over = 95 * GIB;
        assert_eq!(
            check_hbm_fit(planned_over, usable),
            Err(HbmError::PlanOverflow {
                planned: planned_over,
                usable,
                shortfall: 4 * GIB,
            })
        );
        let short_fit = hbm_check_fit(planned_over, usable);
        assert!(!fit_is_ok(short_fit));
        let (slhs, srhs) = fit_conservation_sides(short_fit, planned_over, usable);
        assert_eq!(slhs, srhs);
        assert_eq!(slhs, planned_over);
    }

    #[test]
    fn five_category_ledger_conserves_exact_bytes() {
        let mut ledger = HbmLedgerGuard::new();
        assert_eq!(ledger.total_bytes(), 0);

        ledger.record(HbmCat::CatWeights, 54 * GIB);
        ledger.record(HbmCat::CatKv, 20 * GIB);
        ledger.record(HbmCat::CatCheckpoints, 4 * GIB);
        ledger.record(HbmCat::CatScratch, 640 * MIB);
        ledger.record(HbmCat::CatOther, 128 * MIB);
        ledger.record(HbmCat::CatKv, 2 * GIB);

        let expected_total = 54 * GIB + 22 * GIB + 4 * GIB + 640 * MIB + 128 * MIB;
        assert_eq!(ledger.total_bytes(), expected_total);
        assert!(ledger.matches(54 * GIB, 22 * GIB, 4 * GIB, 640 * MIB, 128 * MIB));
        assert_eq!(
            ledger.breakdown(),
            (54 * GIB, 22 * GIB, 4 * GIB, 640 * MIB, 128 * MIB)
        );
    }

    #[test]
    fn prefix_slots_yield_to_context_and_zero_slot_bytes() {
        let usable = 91 * GIB;
        let slot_bytes = 512 * MIB;

        // 2 GiB headroom => at most 4 slots even if 8 requested
        assert_eq!(grant_prefix_slots(8, slot_bytes, 89 * GIB, usable), 4);
        // 10 GiB headroom => capped at requested 8 slots
        assert_eq!(grant_prefix_slots(8, slot_bytes, 81 * GIB, usable), 8);
        // Base context exceeds usable => 0 prefix slots (context always wins)
        assert_eq!(grant_prefix_slots(8, slot_bytes, 92 * GIB, usable), 0);
        // Zero slot bytes => safe 0 (no division by zero)
        assert_eq!(grant_prefix_slots(8, 0, 50 * GIB, usable), 0);
    }
}
