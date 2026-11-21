//! Rust Shell wrappers for HPU device, Synapse recipe, MoE expert slot, request,
//! multi-lane gate, and single-resident model router lifecycles.
//!
//! Every transition delegates directly to the Lean-checked Bend 2 core in `verified`.

use crate::verified::{
    be_is_card_released, dev_can_alloc, dev_can_dma, dev_can_launch, dev_drain, dev_step,
    lane_can_admit, recipe_bind_ws, recipe_can_launch, recipe_step, req_cancel, req_is_terminal,
    req_step, router_can_switch, slot_begin_fill, slot_begin_retire, slot_is_readable,
    slot_is_writable, slot_mark_loaded, slot_publish, slot_quiesce, slot_rollback_retire,
    slot_set_retire_epoch, BackendState, DevEvent, DevState, RecipeEvent, RecipeState, ReqEvent,
    ReqState, SlotState,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleError {
    InvalidDeviceTransition { from: DevState, event: DevEvent },
    DeviceAllocDisallowed(DevState),
    DeviceDmaDisallowed(DevState),
    DeviceLaunchDisallowed(DevState),
    RecipeWorkspaceFailed { ws_need: u64, pool_size: u64 },
    RecipeLaunchDisallowed(RecipeState),
    InvalidSlotTransition { from: SlotState, op: &'static str },
    InvalidRequestTransition { from: ReqState, event: ReqEvent },
}

/// Checked HPU device lifecycle guard (mirrors `common.hpp` `Dev` + `AllocGuard`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceGuard {
    state: DevState,
}

impl Default for DeviceGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceGuard {
    pub const fn new() -> Self {
        Self {
            state: DevState::DevUninit,
        }
    }

    pub const fn state(&self) -> DevState {
        self.state
    }

    pub fn transition(&mut self, event: DevEvent) -> Result<DevState, LifecycleError> {
        let next = dev_step(self.state, event);
        if next == self.state && self.state != DevState::DevClosed {
            return Err(LifecycleError::InvalidDeviceTransition {
                from: self.state,
                event,
            });
        }
        self.state = next;
        Ok(next)
    }

    pub fn emergency_drain(&mut self) -> DevState {
        self.state = dev_drain(self.state);
        self.state
    }

    pub fn check_alloc(&self) -> Result<(), LifecycleError> {
        if dev_can_alloc(self.state) {
            Ok(())
        } else {
            Err(LifecycleError::DeviceAllocDisallowed(self.state))
        }
    }

    pub fn check_dma(&self) -> Result<(), LifecycleError> {
        if dev_can_dma(self.state) {
            Ok(())
        } else {
            Err(LifecycleError::DeviceDmaDisallowed(self.state))
        }
    }

    pub fn check_launch(&self) -> Result<(), LifecycleError> {
        if dev_can_launch(self.state) {
            Ok(())
        } else {
            Err(LifecycleError::DeviceLaunchDisallowed(self.state))
        }
    }
}

/// Checked Synapse recipe lifecycle guard (mirrors `common.hpp` `Recipe` + `bind_workspace`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecipeGuard {
    state: RecipeState,
}

impl Default for RecipeGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl RecipeGuard {
    pub const fn new() -> Self {
        Self {
            state: RecipeState::RecBuilding,
        }
    }

    pub const fn state(&self) -> RecipeState {
        self.state
    }

    pub fn compile(&mut self) -> RecipeState {
        self.state = recipe_step(self.state, RecipeEvent::EvRecCompile);
        self.state
    }

    pub fn bind_workspace(
        &mut self,
        ws_need: u64,
        pool_size: u64,
        alloc_sealed: bool,
    ) -> Result<RecipeState, LifecycleError> {
        let fits_shared = ws_need <= pool_size;
        let next = recipe_bind_ws(self.state, fits_shared, alloc_sealed);
        self.state = next;
        if next == RecipeState::RecWsFailed {
            Err(LifecycleError::RecipeWorkspaceFailed { ws_need, pool_size })
        } else {
            Ok(next)
        }
    }

    pub fn destroy(&mut self) -> RecipeState {
        self.state = recipe_step(self.state, RecipeEvent::EvRecDestroy);
        self.state
    }

    pub fn check_launch(&self) -> Result<(), LifecycleError> {
        if recipe_can_launch(self.state) {
            Ok(())
        } else {
            Err(LifecycleError::RecipeLaunchDisallowed(self.state))
        }
    }
}

/// Checked MoE hybrid expert HBM cache slot (mirrors `moe_cache.hpp` `mc::Slot`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlotGuard {
    state: SlotState,
}

impl Default for ExpertSlotGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertSlotGuard {
    pub const fn new() -> Self {
        Self {
            state: SlotState::SlotFree,
        }
    }

    pub const fn state(&self) -> SlotState {
        self.state
    }

    pub fn is_readable(&self) -> bool {
        slot_is_readable(self.state)
    }

    pub fn is_writable(&self) -> bool {
        slot_is_writable(self.state)
    }

    pub fn begin_fill(&mut self) -> Result<SlotState, LifecycleError> {
        let next = slot_begin_fill(self.state);
        if next == self.state {
            return Err(LifecycleError::InvalidSlotTransition {
                from: self.state,
                op: "begin_fill",
            });
        }
        self.state = next;
        Ok(next)
    }

    pub fn mark_loaded(&mut self) -> Result<SlotState, LifecycleError> {
        let next = slot_mark_loaded(self.state);
        if next == self.state {
            return Err(LifecycleError::InvalidSlotTransition {
                from: self.state,
                op: "mark_loaded",
            });
        }
        self.state = next;
        Ok(next)
    }

    pub fn publish(&mut self) -> Result<SlotState, LifecycleError> {
        let next = slot_publish(self.state);
        if next == self.state {
            return Err(LifecycleError::InvalidSlotTransition {
                from: self.state,
                op: "publish",
            });
        }
        self.state = next;
        Ok(next)
    }

    pub fn begin_retire(&mut self) -> Result<SlotState, LifecycleError> {
        let next = slot_begin_retire(self.state);
        if next == self.state {
            return Err(LifecycleError::InvalidSlotTransition {
                from: self.state,
                op: "begin_retire",
            });
        }
        self.state = next;
        Ok(next)
    }

    pub fn rollback_retire(&mut self) -> Result<SlotState, LifecycleError> {
        let next = slot_rollback_retire(self.state);
        if next == self.state {
            return Err(LifecycleError::InvalidSlotTransition {
                from: self.state,
                op: "rollback_retire",
            });
        }
        self.state = next;
        Ok(next)
    }

    pub fn set_retire_epoch(&mut self, epoch: u64) -> Result<SlotState, LifecycleError> {
        let next = slot_set_retire_epoch(self.state, epoch);
        if next == self.state {
            return Err(LifecycleError::InvalidSlotTransition {
                from: self.state,
                op: "set_retire_epoch",
            });
        }
        self.state = next;
        Ok(next)
    }

    pub fn quiesce(&mut self, completed_epoch: u64) -> bool {
        let prev = self.state;
        self.state = slot_quiesce(self.state, completed_epoch);
        prev != SlotState::SlotFree && self.state == SlotState::SlotFree
    }
}

/// Checked inference request lifecycle guard (mirrors `serve.hpp` / `gaudium_engine.cpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestGuard {
    state: ReqState,
}

impl Default for RequestGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestGuard {
    pub const fn new() -> Self {
        Self {
            state: ReqState::ReqQueued,
        }
    }

    pub const fn state(&self) -> ReqState {
        self.state
    }

    pub fn is_terminal(&self) -> bool {
        req_is_terminal(self.state)
    }

    pub fn apply(&mut self, event: ReqEvent) -> Result<ReqState, LifecycleError> {
        let next = req_step(self.state, event);
        if next == self.state && !req_is_terminal(self.state) {
            return Err(LifecycleError::InvalidRequestTransition {
                from: self.state,
                event,
            });
        }
        self.state = next;
        Ok(next)
    }

    /// Idempotent cancellation: transitions any active state to `ReqCancelled`, leaves terminal states intact.
    pub fn cancel(&mut self) -> ReqState {
        self.state = req_cancel(self.state);
        self.state
    }
}

/// Checked admission gate for multi-lane decode + exclusive vision requests (`LaneGate`).
pub fn can_admit_lane(
    busy: u64,
    max_lanes: u64,
    exclusive_active: bool,
    want_exclusive: bool,
) -> bool {
    lane_can_admit(busy, max_lanes, exclusive_active, want_exclusive)
}

/// Checked single-resident model router switch guard (`RouterBackend`).
pub fn can_switch_resident_model(outgoing_active: u64, outgoing_state: BackendState) -> bool {
    router_can_switch(outgoing_active, outgoing_state)
}

pub fn is_backend_card_released(state: BackendState) -> bool {
    be_is_card_released(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_lifecycle_enforces_seal_and_drain() {
        let mut dev = DeviceGuard::new();
        assert!(dev.check_alloc().is_err());
        assert!(dev.check_dma().is_err());
        assert!(dev.check_launch().is_err());

        dev.transition(DevEvent::EvSynInit).unwrap();
        assert!(dev.check_alloc().is_err());

        dev.transition(DevEvent::EvDevAcquire).unwrap();
        assert!(dev.check_alloc().is_ok());
        assert!(dev.check_launch().is_err());

        dev.transition(DevEvent::EvStreamsCreate).unwrap();
        assert!(dev.check_alloc().is_ok());
        assert!(dev.check_dma().is_ok());
        assert!(dev.check_launch().is_err());

        dev.transition(DevEvent::EvSealAlloc).unwrap();
        assert!(dev.check_alloc().is_err());
        assert!(dev.check_dma().is_ok());
        assert!(dev.check_launch().is_ok());

        assert_eq!(dev.emergency_drain(), DevState::DevDraining);
        assert!(dev.check_alloc().is_err());
        assert!(dev.check_dma().is_err());
        assert!(dev.check_launch().is_err());

        dev.transition(DevEvent::EvDevClose).unwrap();
        assert_eq!(dev.state(), DevState::DevClosed);
        // Closed absorbs further events
        assert_eq!(
            dev.transition(DevEvent::EvSynInit).unwrap(),
            DevState::DevClosed
        );
    }

    #[test]
    fn recipe_lifecycle_blocks_post_seal_ws_overflow() {
        let mut rec = RecipeGuard::new();
        assert!(rec.check_launch().is_err());
        rec.compile();
        assert!(rec.bind_workspace(700_000_000, 640_000_000, true).is_err());
        assert_eq!(rec.state(), RecipeState::RecWsFailed);
        assert!(rec.check_launch().is_err());

        let mut rec_ok = RecipeGuard::new();
        rec_ok.compile();
        rec_ok
            .bind_workspace(500_000_000, 640_000_000, true)
            .unwrap();
        assert!(rec_ok.check_launch().is_ok());
        rec_ok.destroy();
        assert!(rec_ok.check_launch().is_err());
    }

    #[test]
    fn moe_slot_lifecycle_requires_epoch_quiescence() {
        let mut slot = ExpertSlotGuard::new();
        assert!(!slot.is_readable() && !slot.is_writable());

        slot.begin_fill().unwrap();
        assert!(slot.is_writable() && !slot.is_readable());
        slot.mark_loaded().unwrap();
        slot.publish().unwrap();
        assert!(slot.is_readable() && !slot.is_writable());

        // Cannot fill while ready
        assert!(slot.begin_fill().is_err());

        // Retire -> rollback roundtrip
        slot.begin_retire().unwrap();
        slot.rollback_retire().unwrap();
        assert_eq!(slot.state(), SlotState::SlotReady);

        // Retire with epoch 10: cannot free before epoch 10 completes
        slot.begin_retire().unwrap();
        assert!(!slot.quiesce(999)); // unbound epoch never frees
        slot.set_retire_epoch(10).unwrap();
        assert!(!slot.quiesce(9));
        assert!(slot.is_readable());
        assert!(slot.quiesce(10));
        assert_eq!(slot.state(), SlotState::SlotFree);
    }

    #[test]
    fn request_lane_and_router_invariants() {
        let mut req = RequestGuard::new();
        req.apply(ReqEvent::EvReqStartPrefill).unwrap();
        req.apply(ReqEvent::EvReqPrefillDone).unwrap();
        req.apply(ReqEvent::EvReqPreempt).unwrap();
        req.apply(ReqEvent::EvReqResume).unwrap();
        req.apply(ReqEvent::EvReqPrefillDone).unwrap();
        req.apply(ReqEvent::EvReqFinishStop).unwrap();
        assert!(req.is_terminal());
        // Cancel after terminal finish leaves DoneStop intact
        assert_eq!(req.cancel(), ReqState::ReqDoneStop);

        let mut cancelled = RequestGuard::new();
        assert_eq!(cancelled.cancel(), ReqState::ReqCancelled);
        assert_eq!(cancelled.cancel(), ReqState::ReqCancelled);

        // LaneGate exclusive vision vs shared decode
        assert!(can_admit_lane(0, 4, false, true));
        assert!(!can_admit_lane(1, 4, false, true));
        assert!(!can_admit_lane(0, 4, true, false));
        assert!(can_admit_lane(3, 4, false, false));
        assert!(!can_admit_lane(4, 4, false, false));

        // Router single-resident swap
        assert!(can_switch_resident_model(0, BackendState::BeParked));
        assert!(can_switch_resident_model(0, BackendState::BeFailed));
        assert!(!can_switch_resident_model(1, BackendState::BeParked));
        assert!(!can_switch_resident_model(0, BackendState::BeStopping));
        assert!(!can_switch_resident_model(0, BackendState::BeReady));
    }
}
