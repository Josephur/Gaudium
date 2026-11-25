# Slice 1 — Verified Lifecycle State Machines for Gaudium (Intel Gaudi 2 HPU)
# Formally specifies and proves core engine runtime invariants:
#   - HPU device, AllocGuard, Synapse recipe & workspace pool lifecycles
#   - MoE hybrid expert HBM cache slot lifecycle & epoch quiescence
#   - Inference request lifecycle & idempotent cancellation
#   - Multi-lane admission gate (exclusive vision vs shared decode)
#   - Single-resident card exclusivity for model router switching

import Base

# ============================================================================
# 1. HPU Device & AllocGuard Lifecycle (common.hpp)
# ============================================================================

type DevState is Data:
  DevUninit{}
  DevInit{}
  DevAcquired{}
  DevStreamsReady{}
  DevSealed{}
  DevDraining{}
  DevClosed{}

type DevEvent is Data:
  EvSynInit{}
  EvDevAcquire{}
  EvStreamsCreate{}
  EvSealAlloc{}
  EvBeginDrain{}
  EvDevClose{}

def dev_step(s: DevState, ev: DevEvent) -> DevState:
  match s ev:
    case DevUninit{} EvSynInit{}:
      DevInit{}
    case DevInit{} EvDevAcquire{}:
      DevAcquired{}
    case DevInit{} EvDevClose{}:
      DevClosed{}
    case DevAcquired{} EvStreamsCreate{}:
      DevStreamsReady{}
    case DevAcquired{} EvBeginDrain{}:
      DevDraining{}
    case DevStreamsReady{} EvSealAlloc{}:
      DevSealed{}
    case DevStreamsReady{} EvBeginDrain{}:
      DevDraining{}
    case DevSealed{} EvBeginDrain{}:
      DevDraining{}
    case DevDraining{} EvDevClose{}:
      DevClosed{}
    case DevClosed{} _:
      DevClosed{}
    case st _:
      st

def dev_drain(s: DevState) -> DevState:
  match s:
    case DevUninit{}:
      DevClosed{}
    case DevInit{}:
      DevClosed{}
    case DevAcquired{}:
      DevDraining{}
    case DevStreamsReady{}:
      DevDraining{}
    case DevSealed{}:
      DevDraining{}
    case DevDraining{}:
      DevDraining{}
    case DevClosed{}:
      DevClosed{}

# Start-up device allocation (dmalloc) is permitted ONLY after acquire and BEFORE seal.
def dev_can_alloc(s: DevState) -> Bool:
  match s:
    case DevAcquired{}:
      True{}
    case DevStreamsReady{}:
      True{}
    case _:
      False{}

# Host<->HBM DMA is permitted once streams exist and before drain/close.
def dev_can_dma(s: DevState) -> Bool:
  match s:
    case DevStreamsReady{}:
      True{}
    case DevSealed{}:
      True{}
    case _:
      False{}

# Inference recipe launches are permitted ONLY after the HBM plan is sealed.
def dev_can_launch(s: DevState) -> Bool:
  match s:
    case DevSealed{}:
      True{}
    case _:
      False{}

# LAW 1.1: Device allocation phase and inference launch phase NEVER overlap.
law dev_alloc_launch_disjoint:
  for +s: DevState
  {Bool.and(dev_can_alloc(s), dev_can_launch(s)) == False{} : Bool}

def dev_alloc_launch_disjoint(s):
  match s:
    case DevUninit{}: {==}
    case DevInit{}: {==}
    case DevAcquired{}: {==}
    case DevStreamsReady{}: {==}
    case DevSealed{}: {==}
    case DevDraining{}: {==}
    case DevClosed{}: {==}

# LAW 1.2: Closed device is strictly absorbing across all events.
law dev_closed_absorbing:
  for ev: DevEvent
  {dev_step(DevClosed{}, ev) == DevClosed{} : DevState}

def dev_closed_absorbing(ev):
  match ev:
    case EvSynInit{}: {==}
    case EvDevAcquire{}: {==}
    case EvStreamsCreate{}: {==}
    case EvSealAlloc{}: {==}
    case EvBeginDrain{}: {==}
    case EvDevClose{}: {==}

# LAW 1.3: Emergency drain is idempotent across every device state.
law dev_drain_idempotent:
  for s: DevState
  {dev_drain(dev_drain(s)) == dev_drain(s) : DevState}

def dev_drain_idempotent(s):
  match s:
    case DevUninit{}: {==}
    case DevInit{}: {==}
    case DevAcquired{}: {==}
    case DevStreamsReady{}: {==}
    case DevSealed{}: {==}
    case DevDraining{}: {==}
    case DevClosed{}: {==}

# LAW 1.4: After draining, NO allocation, DMA, or launch is permitted.
law dev_drain_blocks_all:
  for +s: DevState
  {Bool.or(dev_can_alloc(dev_drain(s)), Bool.or(dev_can_dma(dev_drain(s)), dev_can_launch(dev_drain(s)))) == False{} : Bool}

def dev_drain_blocks_all(s):
  match s:
    case DevUninit{}: {==}
    case DevInit{}: {==}
    case DevAcquired{}: {==}
    case DevStreamsReady{}: {==}
    case DevSealed{}: {==}
    case DevDraining{}: {==}
    case DevClosed{}: {==}

# ============================================================================
# 2. SynapseAI Graph & Recipe Lifecycle (common.hpp: Recipe, bind_workspace)
# ============================================================================

type RecipeState is Data:
  RecBuilding{}
  RecCompiled{}
  RecWsBound{}
  RecWsFailed{}
  RecDestroyed{}

type RecipeEvent is Data:
  EvRecCompile{}
  EvRecDestroy{}

def recipe_step(s: RecipeState, ev: RecipeEvent) -> RecipeState:
  match s ev:
    case RecBuilding{} EvRecCompile{}:
      RecCompiled{}
    case RecDestroyed{} _:
      RecDestroyed{}
    case _ EvRecDestroy{}:
      RecDestroyed{}
    case st _:
      st

def recipe_bind_ws(s: RecipeState, fits_shared: Bool, alloc_sealed: Bool) -> RecipeState:
  match s:
    case RecCompiled{}:
      match fits_shared:
        case True{}:
          RecWsBound{}
        case False{}:
          match alloc_sealed:
            case True{}:
              RecWsFailed{}
            case False{}:
              RecWsBound{}
    case other:
      other

def recipe_bind_shared_ws_if(ws_fits: Bool) -> RecipeState:
  match ws_fits:
    case True{}:
      RecWsBound{}
    case False{}:
      RecWsFailed{}

def recipe_bind_shared_ws(s: RecipeState, ws_need: Nat, pool_size: Nat) -> RecipeState:
  match s:
    case RecCompiled{}:
      recipe_bind_shared_ws_if(Nat.is_le(ws_need, pool_size))
    case other:
      other

def rec_le_add_gt_false(+p: Nat, extra: Nat) -> {False{} == Nat.is_le(Nat.add(p, 1n+extra), p) : Bool}:
  match p:
    case 0n:
      {==}
    case 1n+pp:
      rec_le_add_gt_false(pp, extra)

def rec_le_add_r(+a: Nat, b: Nat) -> {True{} == Nat.is_le(a, Nat.add(a, b)) : Bool}:
  match a:
    case 0n:
      match b:
        case 0n: {==}
        case 1n+_: {==}
    case 1n+ap:
      rec_le_add_r(ap, b)

def recipe_can_launch(s: RecipeState) -> Bool:
  match s:
    case RecWsBound{}:
      True{}
    case _:
      False{}

# LAW 2.1: A recipe that failed workspace binding can NEVER be launched.
law recipe_ws_fail_never_launches:
  {recipe_can_launch(RecWsFailed{}) == False{} : Bool}

def recipe_ws_fail_never_launches():
  {==}

# LAW 2.4: Binding a compiled recipe to an undersized shared workspace fails and cannot launch.
# promise: binding a compiled recipe whose workspace exceeds the shared pool transitions to RecWsFailed
# catches: oversize check ignored so pre-seal shared bind accepts any workspace size
law recipe_shared_ws_oversize_fails:
  for +pool: Nat
  for +extra: Nat
  {recipe_bind_shared_ws(RecCompiled{}, Nat.add(pool, 1n+extra), pool) == RecWsFailed{} : RecipeState}

def recipe_shared_ws_oversize_fails(pool, extra):
  %rec_le_add_gt_false(pool, extra) : {recipe_bind_shared_ws_if(_) == RecWsFailed{} : RecipeState}
  {==}

# LAW 2.5: Binding a compiled recipe whose workspace fits in the shared pool binds for launch.
# promise: a compiled recipe whose workspace requirement fits in the shared pool becomes launch-ready
# catches: strict < used instead of <= so exact-fit shared workspace is rejected
law recipe_shared_ws_fit_binds:
  for +need: Nat
  for +slack: Nat
  {recipe_bind_shared_ws(RecCompiled{}, need, Nat.add(need, slack)) == RecWsBound{} : RecipeState}

def recipe_shared_ws_fit_binds(need, slack):
  %rec_le_add_r(need, slack) : {recipe_bind_shared_ws_if(_) == RecWsBound{} : RecipeState}
  {==}

# LAW 2.6: Binding shared workspace outside RecCompiled never makes uncompiled, failed, or destroyed recipes launchable.
# promise: binding shared workspace on RecBuilding, RecWsFailed, or RecDestroyed never enables launch
# catches: match arm on s dropped so uncompiled recipes transition to RecWsBound
law recipe_shared_ws_uncompiled_cannot_launch:
  for +ws_need: Nat
  for +pool_size: Nat
  {Bool.and(Bool.not(recipe_can_launch(recipe_bind_shared_ws(RecBuilding{}, ws_need, pool_size))), Bool.and(Bool.not(recipe_can_launch(recipe_bind_shared_ws(RecWsFailed{}, ws_need, pool_size))), Bool.not(recipe_can_launch(recipe_bind_shared_ws(RecDestroyed{}, ws_need, pool_size))))) == True{} : Bool}

def recipe_shared_ws_uncompiled_cannot_launch(ws_need, pool_size):
  {==}

# LAW 2.2: A destroyed recipe is absorbing across all events.
law recipe_destroyed_absorbing:
  for ev: RecipeEvent
  {recipe_step(RecDestroyed{}, ev) == RecDestroyed{} : RecipeState}

def recipe_destroyed_absorbing(ev):
  match ev:
    case EvRecCompile{}: {==}
    case EvRecDestroy{}: {==}

# LAW 2.3: Destroying a recipe is idempotent across every state.
law recipe_destroy_idempotent:
  for s: RecipeState
  {recipe_step(recipe_step(s, EvRecDestroy{}), EvRecDestroy{}) == recipe_step(s, EvRecDestroy{}) : RecipeState}

def recipe_destroy_idempotent(s):
  match s:
    case RecBuilding{}: {==}
    case RecCompiled{}: {==}
    case RecWsBound{}: {==}
    case RecWsFailed{}: {==}
    case RecDestroyed{}: {==}

# LAW 2.4: After start-up seal, a compiled recipe exceeding the shared workspace MUST fail binding.
law recipe_sealed_ws_overflow_fails:
  {recipe_bind_ws(RecCompiled{}, False{}, True{}) == RecWsFailed{} : RecipeState}

def recipe_sealed_ws_overflow_fails():
  {==}

# LAW 2.5: A compiled recipe that fits the shared workspace always binds cleanly.
law recipe_shared_fit_binds:
  for sealed: Bool
  {recipe_bind_ws(RecCompiled{}, True{}, sealed) == RecWsBound{} : RecipeState}

def recipe_shared_fit_binds(sealed):
  match sealed:
    case True{}: {==}
    case False{}: {==}

# ============================================================================
# 3. MoE Hybrid Expert HBM Slot Lifecycle (moe_cache.hpp: mc::State + epoch)
# ============================================================================

type SlotState is Data:
  SlotFree{}
  SlotFilling{}
  SlotLoaded{}
  SlotReady{}
  SlotRetiringUnbound{}
  SlotRetiringEpoch{epoch: Nat}

def slot_begin_fill(s: SlotState) -> SlotState:
  match s:
    case SlotFree{}:
      SlotFilling{}
    case other:
      other

def slot_mark_loaded(s: SlotState) -> SlotState:
  match s:
    case SlotFilling{}:
      SlotLoaded{}
    case other:
      other

def slot_publish(s: SlotState) -> SlotState:
  match s:
    case SlotLoaded{}:
      SlotReady{}
    case other:
      other

def slot_begin_retire(s: SlotState) -> SlotState:
  match s:
    case SlotReady{}:
      SlotRetiringUnbound{}
    case other:
      other

def slot_rollback_retire(s: SlotState) -> SlotState:
  match s:
    case SlotRetiringUnbound{}:
      SlotReady{}
    case other:
      other

def slot_set_retire_epoch(s: SlotState, epoch: Nat) -> SlotState:
  match s:
    case SlotRetiringUnbound{}:
      SlotRetiringEpoch{epoch}
    case other:
      other

def slot_quiesce_if(ep: Nat, done: Bool) -> SlotState:
  match done:
    case True{}:
      SlotFree{}
    case False{}:
      SlotRetiringEpoch{ep}

def slot_quiesce(s: SlotState, completed: Nat) -> SlotState:
  match s:
    case SlotRetiringEpoch{+ep}:
      slot_quiesce_if(ep, Nat.is_le(ep, completed))
    case other:
      other

# Device launches may read a slot while Ready or Retiring (until quiesced).
def slot_is_readable(s: SlotState) -> Bool:
  match s:
    case SlotReady{}:
      True{}
    case SlotRetiringUnbound{}:
      True{}
    case SlotRetiringEpoch{_}:
      True{}
    case _:
      False{}

# Host DMA fill may write a slot ONLY while Filling.
def slot_is_writable(s: SlotState) -> Bool:
  match s:
    case SlotFilling{}:
      True{}
    case _:
      False{}

# Helper proof lemmas for Nat.is_le (erased before Rust codegen)
def nat_is_le_refl(n: Nat) -> {True{} == Nat.is_le(n, n) : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      nat_is_le_refl(p)

def nat_is_le_succ_false(n: Nat) -> {False{} == Nat.is_le(1n+n, n) : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      nat_is_le_succ_false(p)

# LAW 3.1: No slot is ever simultaneously readable by device kernels and writable by fill DMA.
law slot_rw_disjoint:
  for +s: SlotState
  {Bool.and(slot_is_readable(s), slot_is_writable(s)) == False{} : Bool}

def slot_rw_disjoint(s):
  match s:
    case SlotFree{}: {==}
    case SlotFilling{}: {==}
    case SlotLoaded{}: {==}
    case SlotReady{}: {==}
    case SlotRetiringUnbound{}: {==}
    case SlotRetiringEpoch{_}: {==}

# LAW 3.2: Starting a fill on any readable (Ready or Retiring) slot NEVER makes it writable.
law slot_fill_never_overwrites_readable:
  for +s: SlotState
  {Bool.and(slot_is_readable(s), slot_is_writable(slot_begin_fill(s))) == False{} : Bool}

def slot_fill_never_overwrites_readable(s):
  match s:
    case SlotFree{}: {==}
    case SlotFilling{}: {==}
    case SlotLoaded{}: {==}
    case SlotReady{}: {==}
    case SlotRetiringUnbound{}: {==}
    case SlotRetiringEpoch{_}: {==}

# LAW 3.3: A retiring slot whose table unpublish epoch has not been recorded NEVER quiesces.
law slot_unbound_never_quiesces:
  for completed: Nat
  {slot_quiesce(SlotRetiringUnbound{}, completed) == SlotRetiringUnbound{} : SlotState}

def slot_unbound_never_quiesces(completed):
  {==}

# LAW 3.4: Once completed reaches a retiring slot's recorded epoch, quiesce frees the slot.
law slot_quiesce_at_exact_epoch:
  for +ep: Nat
  {slot_quiesce(SlotRetiringEpoch{ep}, ep) == SlotFree{} : SlotState}

def slot_quiesce_at_exact_epoch(ep):
  %nat_is_le_refl(ep) : {slot_quiesce_if(ep, _) == SlotFree{} : SlotState}
  {==}

# LAW 3.5: If a launch issued before unpublish has NOT completed (epoch = completed + 1), slot stays retiring.
law slot_quiesce_before_epoch_stays_retiring:
  for +c: Nat
  {slot_quiesce(SlotRetiringEpoch{1n+c}, c) == SlotRetiringEpoch{1n+c} : SlotState}

def slot_quiesce_before_epoch_stays_retiring(c):
  %nat_is_le_succ_false(c) : {slot_quiesce_if(1n+c, _) == SlotRetiringEpoch{1n+c} : SlotState}
  {==}

# LAW 3.6: Rolling back a batched eviction before table write restores SlotReady cleanly.
law slot_rollback_roundtrip:
  {slot_rollback_retire(slot_begin_retire(SlotReady{})) == SlotReady{} : SlotState}

def slot_rollback_roundtrip():
  {==}

# ============================================================================
# 4. Request Lifecycle, LaneGate & Multi-Model Router (serve.hpp, server/)
# ============================================================================

type ReqState is Data:
  ReqQueued{}
  ReqPrefilling{}
  ReqDecoding{}
  ReqPreempted{}
  ReqDoneStop{}
  ReqDoneLength{}
  ReqCancelled{}
  ReqFailed{}

type ReqEvent is Data:
  EvReqStartPrefill{}
  EvReqPrefillDone{}
  EvReqPreempt{}
  EvReqResume{}
  EvReqFinishStop{}
  EvReqFinishLength{}
  EvReqCancel{}
  EvReqFail{}

def req_is_terminal(s: ReqState) -> Bool:
  match s:
    case ReqDoneStop{}:
      True{}
    case ReqDoneLength{}:
      True{}
    case ReqCancelled{}:
      True{}
    case ReqFailed{}:
      True{}
    case _:
      False{}

def req_eq(a: ReqState, b: ReqState) -> Bool:
  match a b:
    case ReqQueued{} ReqQueued{}:
      True{}
    case ReqPrefilling{} ReqPrefilling{}:
      True{}
    case ReqDecoding{} ReqDecoding{}:
      True{}
    case ReqPreempted{} ReqPreempted{}:
      True{}
    case ReqDoneStop{} ReqDoneStop{}:
      True{}
    case ReqDoneLength{} ReqDoneLength{}:
      True{}
    case ReqCancelled{} ReqCancelled{}:
      True{}
    case ReqFailed{} ReqFailed{}:
      True{}
    case _ _:
      False{}

def req_step(s: ReqState, ev: ReqEvent) -> ReqState:
  match s ev:
    case ReqQueued{} EvReqStartPrefill{}:
      ReqPrefilling{}
    case ReqQueued{} EvReqCancel{}:
      ReqCancelled{}
    case ReqQueued{} EvReqFail{}:
      ReqFailed{}
    case ReqPrefilling{} EvReqPrefillDone{}:
      ReqDecoding{}
    case ReqPrefilling{} EvReqFinishLength{}:
      ReqDoneLength{}
    case ReqPrefilling{} EvReqCancel{}:
      ReqCancelled{}
    case ReqPrefilling{} EvReqFail{}:
      ReqFailed{}
    case ReqDecoding{} EvReqPreempt{}:
      ReqPreempted{}
    case ReqDecoding{} EvReqFinishStop{}:
      ReqDoneStop{}
    case ReqDecoding{} EvReqFinishLength{}:
      ReqDoneLength{}
    case ReqDecoding{} EvReqCancel{}:
      ReqCancelled{}
    case ReqDecoding{} EvReqFail{}:
      ReqFailed{}
    case ReqPreempted{} EvReqResume{}:
      ReqPrefilling{}
    case ReqPreempted{} EvReqCancel{}:
      ReqCancelled{}
    case ReqPreempted{} EvReqFail{}:
      ReqFailed{}
    case st _:
      st

def req_cancel(s: ReqState) -> ReqState:
  req_step(s, EvReqCancel{})

# LAW 4.1: Cancelling a request is idempotent across every state.
law req_cancel_idempotent:
  for s: ReqState
  {req_cancel(req_cancel(s)) == req_cancel(s) : ReqState}

def req_cancel_idempotent(s):
  match s:
    case ReqQueued{}: {==}
    case ReqPrefilling{}: {==}
    case ReqDecoding{}: {==}
    case ReqPreempted{}: {==}
    case ReqDoneStop{}: {==}
    case ReqDoneLength{}: {==}
    case ReqCancelled{}: {==}
    case ReqFailed{}: {==}

# LAW 4.2: Every terminal request state is strictly absorbing under every event.
law req_terminal_absorbing:
  for +s: ReqState
  for ev: ReqEvent
  {Bool.or(Bool.not(req_is_terminal(s)), req_eq(req_step(s, ev), s)) == True{} : Bool}

def req_terminal_absorbing(s, ev):
  match s ev:
    case ReqQueued{} _: {==}
    case ReqPrefilling{} _: {==}
    case ReqDecoding{} _: {==}
    case ReqPreempted{} _: {==}
    case ReqDoneStop{} EvReqStartPrefill{}: {==}
    case ReqDoneStop{} EvReqPrefillDone{}: {==}
    case ReqDoneStop{} EvReqPreempt{}: {==}
    case ReqDoneStop{} EvReqResume{}: {==}
    case ReqDoneStop{} EvReqFinishStop{}: {==}
    case ReqDoneStop{} EvReqFinishLength{}: {==}
    case ReqDoneStop{} EvReqCancel{}: {==}
    case ReqDoneStop{} EvReqFail{}: {==}
    case ReqDoneLength{} EvReqStartPrefill{}: {==}
    case ReqDoneLength{} EvReqPrefillDone{}: {==}
    case ReqDoneLength{} EvReqPreempt{}: {==}
    case ReqDoneLength{} EvReqResume{}: {==}
    case ReqDoneLength{} EvReqFinishStop{}: {==}
    case ReqDoneLength{} EvReqFinishLength{}: {==}
    case ReqDoneLength{} EvReqCancel{}: {==}
    case ReqDoneLength{} EvReqFail{}: {==}
    case ReqCancelled{} EvReqStartPrefill{}: {==}
    case ReqCancelled{} EvReqPrefillDone{}: {==}
    case ReqCancelled{} EvReqPreempt{}: {==}
    case ReqCancelled{} EvReqResume{}: {==}
    case ReqCancelled{} EvReqFinishStop{}: {==}
    case ReqCancelled{} EvReqFinishLength{}: {==}
    case ReqCancelled{} EvReqCancel{}: {==}
    case ReqCancelled{} EvReqFail{}: {==}
    case ReqFailed{} EvReqStartPrefill{}: {==}
    case ReqFailed{} EvReqPrefillDone{}: {==}
    case ReqFailed{} EvReqPreempt{}: {==}
    case ReqFailed{} EvReqResume{}: {==}
    case ReqFailed{} EvReqFinishStop{}: {==}
    case ReqFailed{} EvReqFinishLength{}: {==}
    case ReqFailed{} EvReqCancel{}: {==}
    case ReqFailed{} EvReqFail{}: {==}

# Multi-lane LaneGate admission (gaudium_engine.cpp: exclusive vision vs shared decode).
def lane_can_admit(busy: Nat, max_lanes: Nat, exclusive_active: Bool, want_exclusive: Bool) -> Bool:
  match exclusive_active:
    case True{}:
      False{}
    case False{}:
      match want_exclusive:
        case True{}:
          Bool.and(Nat.is_eq(busy, 0n), Nat.is_gt(max_lanes, 0n))
        case False{}:
          Nat.is_lt(busy, max_lanes)

def nat_is_lt_refl(n: Nat) -> {Nat.is_lt(n, n) == False{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      nat_is_lt_refl(p)

# LAW 4.3: An active exclusive (vision) request blocks ALL concurrent lane admissions.
law lane_exclusive_blocks_all:
  for busy: Nat
  for max_lanes: Nat
  for want_excl: Bool
  {lane_can_admit(busy, max_lanes, True{}, want_excl) == False{} : Bool}

def lane_exclusive_blocks_all(busy, max_lanes, want_excl):
  {==}

# LAW 4.4: An exclusive request cannot enter while any shared lane is busy (busy >= 1).
law lane_busy_blocks_exclusive:
  for b: Nat
  for max_lanes: Nat
  {lane_can_admit(1n+b, max_lanes, False{}, True{}) == False{} : Bool}

def lane_busy_blocks_exclusive(b, max_lanes):
  {==}

# LAW 4.5: A full lane pool (busy == max_lanes) never admits another shared lane.
law lane_full_blocks_shared:
  for +cap: Nat
  for excl: Bool
  {lane_can_admit(cap, cap, excl, False{}) == False{} : Bool}

def lane_full_blocks_shared(cap, excl):
  match excl:
    case True{}:
      {==}
    case False{}:
      nat_is_lt_refl(cap)

# EngineBackend process lifecycle & RouterBackend single-resident card exclusivity.
type BackendState is Data:
  BeParked{}
  BeQueued{}
  BeLoading{}
  BeReady{}
  BeStopping{}
  BeFailed{}

def be_is_card_released(s: BackendState) -> Bool:
  match s:
    case BeParked{}:
      True{}
    case BeFailed{}:
      True{}
    case _:
      False{}

def router_can_switch(outgoing_active: Nat, outgoing_state: BackendState) -> Bool:
  Bool.and(Nat.is_eq(outgoing_active, 0n), be_is_card_released(outgoing_state))

# LAW 4.6: Router NEVER switches models while outgoing requests are still active.
law router_no_switch_while_active:
  for act: Nat
  for st: BackendState
  {router_can_switch(1n+act, st) == False{} : Bool}

def router_no_switch_while_active(act, st):
  {==}

# LAW 4.7: Router NEVER switches models while outgoing engine still holds or is taking the card.
law router_no_switch_while_ready:
  for act: Nat
  {router_can_switch(act, BeReady{}) == False{} : Bool}

def router_no_switch_while_ready(act):
  match act:
    case 0n: {==}
    case 1n+_: {==}

law router_no_switch_while_loading:
  for act: Nat
  {router_can_switch(act, BeLoading{}) == False{} : Bool}

def router_no_switch_while_loading(act):
  match act:
    case 0n: {==}
    case 1n+_: {==}

law router_no_switch_while_stopping:
  for act: Nat
  {router_can_switch(act, BeStopping{}) == False{} : Bool}

def router_no_switch_while_stopping(act):
  match act:
    case 0n: {==}
    case 1n+_: {==}

# ============================================================================
# 5. Live Status Record Writes (live_report.rs)
# ============================================================================
#
# The engine publishes its status as small files (the live record, the card-lock holder). A file
# write can block in the filesystem for far longer than a launch takes, so once the reporter has
# its own writer thread, the thread that reported the change (a request's thread) hands the write
# over and never touches the filesystem itself.

type LiveWrite is Data:
  LwInline{}
  LwWake{}
  LwDefer{}

# Who writes the record after a status change:
#   LwInline  the caller writes it (a reporter without a writer thread)
#   LwWake    the writer thread is woken and writes it
#   LwDefer   the writer thread's next periodic write covers it
def live_write_route(has_writer: Bool, force: Bool) -> LiveWrite:
  match has_writer:
    case True{}:
      match force:
        case True{}:
          LwWake{}
        case False{}:
          LwDefer{}
    case False{}:
      LwInline{}

def lw_is_inline(r: LiveWrite) -> Bool:
  match r:
    case LwInline{}:
      True{}
    case _:
      False{}

def lw_is_defer(r: LiveWrite) -> Bool:
  match r:
    case LwDefer{}:
      True{}
    case _:
      False{}

# LAW 5.1: With a writer thread, the reporting thread never writes the record itself.
law live_write_off_the_caller:
  for force: Bool
  {lw_is_inline(live_write_route(True{}, force)) == False{} : Bool}

def live_write_off_the_caller(force):
  match force:
    case True{}:
      {==}
    case False{}:
      {==}

# LAW 5.2: A forced write is never left to the periodic write.
law live_write_forced_not_deferred:
  for has_writer: Bool
  {lw_is_defer(live_write_route(has_writer, True{})) == False{} : Bool}

def live_write_forced_not_deferred(has_writer):
  match has_writer:
    case True{}:
      {==}
    case False{}:
      {==}

# LAW 5.3: Nothing is left to a periodic write that does not exist (no writer thread).
law live_write_no_writer_no_defer:
  for force: Bool
  {lw_is_defer(live_write_route(False{}, force)) == False{} : Bool}

def live_write_no_writer_no_defer(force):
  match force:
    case True{}:
      {==}
    case False{}:
      {==}
