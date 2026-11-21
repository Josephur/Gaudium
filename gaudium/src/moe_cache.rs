//! MoE hybrid expert HBM cache book (`moe_cache.hpp`).
//!
//! Manages per-layer resident HBM expert slots, LRU/promoter eviction, CPU-share carry,
//! bounded miss-request queues, upload-now vs host-twin decisions, and host-side work-plan
//! construction (`plan_build`). Every slot transition is guarded by the build-time verified
//! Bend 2 `ExpertSlotGuard` (`core::lifecycle`).

use std::collections::VecDeque;

use crate::core::lifecycle::ExpertSlotGuard;
use crate::verified::SlotState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoePolicy {
    Lru,
    Promoter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotEntry {
    pub guard: ExpertSlotGuard,
    pub expert: i32,
}

impl Default for SlotEntry {
    fn default() -> Self {
        Self {
            guard: ExpertSlotGuard::new(),
            expert: -1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillJob {
    pub layer: usize,
    pub expert: usize,
    pub slot: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictJob {
    pub layer: usize,
    pub slot: usize,
    pub expert: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissAction {
    /// Host twin computes it and no fill is scheduled (CPU share or already resident/in-flight).
    TwinOnly,
    /// Host twin computes it and a background fill was requested.
    TwinAndRequest,
    /// Claimed a FREE slot for immediate upload so the card computes it in this step.
    UploadNow { slot: usize },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoeCounters {
    pub requests: u64,
    pub fills: u64,
    pub evictions: u64,
    pub dropped: u64,
    pub observed: u64,
    pub observed_off: u64,
    pub cpu_only: u64,
    pub uploads_now: u64,
}

#[derive(Debug, Clone)]
pub struct LayerBook {
    pub n_experts: usize,
    pub n_slots: usize,
    pub slots: Vec<SlotEntry>,
    pub slot_of: Vec<i32>,
    pub where_filling: Vec<i32>,
    pub waiting: Vec<bool>,
    pub used_tick: Vec<u64>,
    pub score: Vec<f64>,
    pub tick: u64,
    pub wait_queue: VecDeque<usize>,
    pub n_free: usize,
    pub n_retiring: usize,
    pub carry: f64,
}

/// Pure host-side MoE expert cache bookkeeping (no device, no threads).
#[derive(Debug, Clone)]
pub struct MoeBook {
    pub layers: Vec<LayerBook>,
    pub policy: MoePolicy,
    pub reserve: usize,
    pub wait_max: usize,
    pub retire_max: usize,
    pub decay: f64,
    pub margin: f64,
    pub cpu_share: f64,
    pub counters: MoeCounters,
}

impl Default for MoeBook {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            policy: MoePolicy::Lru,
            reserve: 8,
            wait_max: 64,
            retire_max: 8,
            decay: 0.98,
            margin: 2.0,
            cpu_share: 0.0,
            counters: MoeCounters::default(),
        }
    }
}

impl MoeBook {
    /// Registers one hybrid MoE layer where `initial_experts[s]` was preloaded into slot `s`.
    /// The final `reserve` slots start `SlotFree` so early misses have immediate slots without eviction.
    pub fn add_layer(&mut self, n_experts: usize, initial_experts: &[usize]) {
        let n_slots = initial_experts.len();
        let keep = n_slots.saturating_sub(self.reserve.min(n_slots));
        let mut slots = vec![SlotEntry::default(); n_slots];
        let mut slot_of = vec![-1i32; n_experts];
        let mut n_free = 0usize;

        for (s, &exp) in initial_experts.iter().enumerate() {
            if s < keep {
                let mut guard = ExpertSlotGuard::new();
                guard.begin_fill().expect("Free -> Filling");
                guard.mark_loaded().expect("Filling -> Loaded");
                guard.publish().expect("Loaded -> Ready");
                slots[s] = SlotEntry {
                    guard,
                    expert: exp as i32,
                };
                slot_of[exp] = s as i32;
            } else {
                n_free += 1;
            }
        }
        self.layers.push(LayerBook {
            n_experts,
            n_slots,
            slots,
            slot_of,
            where_filling: vec![-1; n_experts],
            waiting: vec![false; n_experts],
            used_tick: vec![0; n_experts],
            score: vec![0.0; n_experts],
            tick: 0,
            wait_queue: VecDeque::new(),
            n_free,
            n_retiring: 0,
            carry: 0.0,
        });
    }

    pub fn is_resident(&self, h: usize, expert: usize) -> bool {
        self.layers[h].slot_of[expert] >= 0
    }

    pub fn observe(&mut self, h: usize, ids: &[i32], on_device: Option<&[i32]>) {
        self.layers[h].tick += 1;
        let tick = self.layers[h].tick;
        let n_experts = self.layers[h].n_experts;
        for (i, &id) in ids.iter().enumerate() {
            if id < 0 || (id as usize) >= n_experts {
                continue;
            }
            let e = id as usize;
            self.layers[h].used_tick[e] = tick;
            self.layers[h].score[e] += 1.0;
            self.counters.observed += 1;
            if let Some(dev) = on_device {
                if dev[i] == 0 {
                    self.counters.observed_off += 1;
                    let y = &mut self.layers[h];
                    if self.policy != MoePolicy::Lru
                        || y.slot_of[e] >= 0
                        || y.where_filling[e] >= 0
                        || y.waiting[e]
                    {
                        continue;
                    }
                    y.carry += self.cpu_share;
                    if y.carry >= 1.0 {
                        y.carry -= 1.0;
                        self.counters.cpu_only += 1;
                        continue;
                    }
                    self.request(h, e);
                }
            }
        }
    }

    pub fn decay_scores(&mut self, h: usize) {
        let decay = self.decay;
        for s in &mut self.layers[h].score {
            *s *= decay;
        }
    }

    pub fn promote(&mut self, h: usize) -> bool {
        let y = &self.layers[h];
        let mut hot: Option<usize> = None;
        let mut cold: Option<usize> = None;
        for e in 0..y.n_experts {
            if y.slot_of[e] >= 0 {
                if cold.is_none_or(|c| y.score[e] < y.score[c]) {
                    cold = Some(e);
                }
            } else if y.where_filling[e] < 0
                && !y.waiting[e]
                && hot.is_none_or(|hi| y.score[e] > y.score[hi])
            {
                hot = Some(e);
            }
        }
        let (Some(hot_e), Some(cold_e)) = (hot, cold) else {
            return false;
        };
        if y.score[hot_e] <= self.margin * y.score[cold_e].max(0.5) {
            return false;
        }
        self.request(h, hot_e)
    }

    pub fn request(&mut self, h: usize, e: usize) -> bool {
        let y = &mut self.layers[h];
        if e >= y.n_experts || y.slot_of[e] >= 0 || y.where_filling[e] >= 0 || y.waiting[e] {
            return false;
        }
        if y.wait_queue.len() >= self.wait_max {
            if let Some(oldest) = y.wait_queue.pop_front() {
                y.waiting[oldest] = false;
                self.counters.dropped += 1;
            }
        }
        y.wait_queue.push_back(e);
        y.waiting[e] = true;
        self.counters.requests += 1;
        true
    }

    /// Claims up to `max_jobs` `SlotFree -> SlotFilling` jobs (newest requests first).
    pub fn take_jobs(&mut self, max_jobs: usize) -> Vec<FillJob> {
        let mut out = Vec::new();
        for h in 0..self.layers.len() {
            if out.len() >= max_jobs {
                break;
            }
            let y = &mut self.layers[h];
            for s in 0..y.n_slots {
                if y.wait_queue.is_empty() || y.n_free == 0 || out.len() >= max_jobs {
                    break;
                }
                if y.slots[s].guard.state() != SlotState::SlotFree {
                    continue;
                }
                let e = y.wait_queue.pop_back().unwrap();
                y.waiting[e] = false;
                y.slots[s].guard.begin_fill().expect("Free -> Filling");
                y.slots[s].expert = e as i32;
                y.where_filling[e] = s as i32;
                y.n_free -= 1;
                self.counters.fills += 1;
                out.push(FillJob {
                    layer: h,
                    expert: e,
                    slot: s,
                });
            }
        }
        out
    }

    /// Handles a miss while the card is paused at a layer's miss gate.
    pub fn miss_now(&mut self, h: usize, e: usize, may_upload: bool) -> MissAction {
        let y = &mut self.layers[h];
        if e >= y.n_experts || y.slot_of[e] >= 0 || y.where_filling[e] >= 0 {
            return MissAction::TwinOnly;
        }
        if self.policy != MoePolicy::Lru {
            return MissAction::TwinOnly;
        }
        y.carry += self.cpu_share;
        if y.carry >= 1.0 {
            y.carry -= 1.0;
            self.counters.cpu_only += 1;
            return MissAction::TwinOnly;
        }
        if may_upload && y.n_free > 0 {
            for s in 0..y.n_slots {
                if y.slots[s].guard.state() != SlotState::SlotFree {
                    continue;
                }
                if y.waiting[e] {
                    y.waiting[e] = false;
                    if let Some(pos) = y.wait_queue.iter().position(|&x| x == e) {
                        y.wait_queue.remove(pos);
                    }
                }
                y.slots[s].guard.begin_fill().expect("Free -> Filling");
                y.slots[s].expert = e as i32;
                y.where_filling[e] = s as i32;
                y.n_free -= 1;
                self.counters.fills += 1;
                self.counters.uploads_now += 1;
                return MissAction::UploadNow { slot: s };
            }
        }
        self.request(h, e);
        MissAction::TwinAndRequest
    }

    pub fn loaded(&mut self, job: FillJob) {
        let slot = &mut self.layers[job.layer].slots[job.slot];
        if slot.guard.state() == SlotState::SlotFilling {
            slot.guard.mark_loaded().expect("Filling -> Loaded");
        }
    }

    pub fn published(&mut self, job: FillJob) {
        let y = &mut self.layers[job.layer];
        let slot = &mut y.slots[job.slot];
        if slot.guard.state() == SlotState::SlotFilling {
            slot.guard.mark_loaded().expect("Filling -> Loaded");
        }
        slot.guard.publish().expect("Loaded -> Ready");
        y.where_filling[job.expert] = -1;
        y.slot_of[job.expert] = job.slot as i32;
    }

    fn victim(&self, h: usize) -> Option<usize> {
        let y = &self.layers[h];
        let mut best: Option<usize> = None;
        for s in 0..y.n_slots {
            let q = &y.slots[s];
            if q.guard.state() != SlotState::SlotReady {
                continue;
            }
            let e = q.expert as usize;
            if y.tick > 0 && y.used_tick[e] == y.tick {
                continue;
            }
            best = match best {
                None => Some(s),
                Some(bs) => {
                    let be = y.slots[bs].expert as usize;
                    let better = match self.policy {
                        MoePolicy::Lru => y.used_tick[e] < y.used_tick[be],
                        MoePolicy::Promoter => y.score[e] < y.score[be],
                    };
                    if better {
                        Some(s)
                    } else {
                        Some(bs)
                    }
                }
            };
        }
        best
    }

    /// Selects victim `SlotReady -> SlotRetiringUnbound` slots to maintain `reserve + wait_queue.len()`.
    pub fn take_evictions(&mut self) -> Vec<EvictJob> {
        let mut out = Vec::new();
        let max_ret = self.reserve.max(self.retire_max);
        for h in 0..self.layers.len() {
            let want = self.layers[h]
                .n_slots
                .min(self.reserve + self.layers[h].wait_queue.len());
            while self.layers[h].n_free + self.layers[h].n_retiring < want
                && self.layers[h].n_retiring < max_ret
            {
                let Some(s) = self.victim(h) else {
                    break;
                };
                let y = &mut self.layers[h];
                let exp = y.slots[s].expert as usize;
                y.slot_of[exp] = -1;
                y.slots[s]
                    .guard
                    .begin_retire()
                    .expect("Ready -> RetiringUnbound");
                y.n_retiring += 1;
                self.counters.evictions += 1;
                out.push(EvictJob {
                    layer: h,
                    slot: s,
                    expert: exp,
                });
            }
        }
        out
    }

    /// Records the launch epoch at which `ev`'s `-1` table write completed (`RetiringUnbound -> RetiringEpoch`).
    pub fn unpublished(&mut self, ev: EvictJob, epoch: u64) {
        self.layers[ev.layer].slots[ev.slot]
            .guard
            .set_retire_epoch(epoch)
            .expect("RetiringUnbound -> RetiringEpoch");
    }

    /// Frees every retiring slot whose bound epoch is `<= completed_epoch`.
    pub fn quiesce(&mut self, completed_epoch: u64) -> usize {
        let mut freed = 0usize;
        for y in &mut self.layers {
            if y.n_retiring == 0 {
                continue;
            }
            for q in &mut y.slots {
                if q.guard.quiesce(completed_epoch) {
                    q.expert = -1;
                    y.n_retiring -= 1;
                    y.n_free += 1;
                    freed += 1;
                }
            }
        }
        freed
    }
}

/// Builds the exact device work plan written by `gaudium_moe_plan` (`mc::plan_build`).
/// Returns `(plan, dev_flags)` where `plan.len() == 1 + 5 * k * t` and `dev_flags.len() == k * t`.
pub fn plan_build(ids: &[i32], slots: &[i32], k: usize, t: usize) -> (Vec<i32>, Vec<i32>) {
    let kt = k * t;
    assert_eq!(ids.len(), kt);
    assert_eq!(slots.len(), kt);
    let mut dev = vec![0i32; kt];
    let mut on_card: Vec<(i32, usize)> = Vec::new();
    for i in 0..kt {
        if slots[i] >= 0 {
            dev[i] = 1;
            on_card.push((ids[i], i));
        }
    }
    on_card.sort_unstable();
    let mut plan = vec![0i32; 1 + 5 * kt];
    let rbase = 1 + 4 * kt;
    let mut u = 0usize;
    let mut rs = 0usize;
    let mut i = 0usize;
    while i < on_card.len() {
        let exp = on_card[i].0;
        let mut j = i;
        while j < on_card.len() && on_card[j].0 == exp {
            j += 1;
        }
        let b = 1 + 4 * u;
        plan[b] = slots[on_card[i].1];
        plan[b + 1] = (j - i) as i32;
        plan[b + 2] = rs as i32;
        plan[b + 3] = exp;
        for &(_, assign) in &on_card[i..j] {
            plan[rbase + rs] = ((assign / k) * 16 + (assign % k)) as i32;
            rs += 1;
        }
        u += 1;
        i = j;
    }
    plan[0] = u as i32;
    (plan, dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleRng(u64);
    impl SimpleRng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn usize(&mut self, bound: usize) -> usize {
            (self.next_u64() as usize) % bound
        }
        fn f64(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
        }
    }

    #[test]
    fn mirrors_moe_cache_test_basics() {
        let mut b = MoeBook {
            reserve: 2,
            ..MoeBook::default()
        };
        b.add_layer(16, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b.layers[0].n_free, 2);
        assert!(b.is_resident(0, 5) && !b.is_resident(0, 6) && !b.is_resident(0, 9));

        b.observe(0, &[9, 3], Some(&[0, 1]));
        assert_eq!(b.counters.requests, 1);
        assert!(!b.request(0, 9) && !b.request(0, 3));

        let jobs = b.take_jobs(4);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].expert, 9);
        assert_eq!(
            b.layers[0].slots[jobs[0].slot].guard.state(),
            SlotState::SlotFilling
        );
        assert!(!b.request(0, 9) && !b.is_resident(0, 9));

        b.loaded(jobs[0]);
        assert!(!b.is_resident(0, 9));
        b.published(jobs[0]);
        assert!(b.is_resident(0, 9));

        let ev = b.take_evictions();
        assert_eq!(ev.len(), 1);
        assert!(ev[0].expert != 9 && ev[0].expert != 3);
        assert!(!b.is_resident(0, ev[0].expert));
        assert_eq!(b.quiesce(100), 0); // unbound epoch never frees
        b.unpublished(ev[0], 7);
        assert_eq!(b.quiesce(6), 0);
        assert_eq!(b.quiesce(7), 1);
        assert_eq!(b.layers[0].n_free, 2);

        // CPU share carry
        for sh in [0.5f64, 1.0, 0.25] {
            let mut d = MoeBook {
                reserve: 1,
                cpu_share: sh,
                ..MoeBook::default()
            };
            d.add_layer(64, &[0, 1]);
            let ids: Vec<i32> = (10..18).collect();
            let dev = vec![0; 8];
            d.observe(0, &ids, Some(&dev));
            let expected_req = (8.0 * (1.0 - sh) + 0.5) as u64;
            assert_eq!(d.counters.requests, expected_req);
            assert_eq!(d.counters.cpu_only, 8 - expected_req);
        }

        // miss_now upload-now vs background vs twin-only
        let mut u = MoeBook {
            reserve: 2,
            ..MoeBook::default()
        };
        u.add_layer(32, &[0, 1, 2, 3]);
        let MissAction::UploadNow { slot: s1 } = u.miss_now(0, 20, true) else {
            panic!("expected UploadNow");
        };
        let MissAction::UploadNow { slot: s2 } = u.miss_now(0, 21, true) else {
            panic!("expected UploadNow");
        };
        assert_ne!(s1, s2);
        assert_eq!(u.layers[0].n_free, 0);
        assert_eq!(u.miss_now(0, 22, true), MissAction::TwinAndRequest);
        assert_eq!(u.miss_now(0, 20, true), MissAction::TwinOnly);
        assert_eq!(u.miss_now(0, 1, true), MissAction::TwinOnly);
        u.published(FillJob {
            layer: 0,
            expert: 20,
            slot: s1,
        });
        assert!(u.is_resident(0, 20) && u.counters.uploads_now == 2);

        // plan_build exact match to moe_cache_test.cpp
        let ids = [1, 5, 7, 9, 1, 2, 7, 30, 5, 7, 8, 9];
        let slots = [11, -1, 3, 6, 11, -1, 3, -1, -1, 3, 0, 6];
        let (plan, dev) = plan_build(&ids, &slots, 4, 3);
        assert_eq!(plan[0], 4);
        assert_eq!(
            &plan[1..17],
            &[11, 2, 0, 1, 3, 3, 2, 7, 0, 1, 5, 8, 6, 2, 6, 9]
        );
        assert_eq!(&plan[1 + 48..1 + 48 + 8], &[0, 16, 2, 18, 33, 34, 3, 35]);
        for i in 0..12 {
            assert_eq!(dev[i], i32::from(slots[i] >= 0));
        }

        // wait queue cap drops oldest, serves newest first
        let mut c = MoeBook {
            reserve: 1,
            wait_max: 3,
            ..MoeBook::default()
        };
        c.add_layer(32, &[0, 1]);
        for e in 10..15 {
            c.request(0, e);
        }
        assert_eq!((c.layers[0].wait_queue.len(), c.counters.dropped), (3, 2));
        assert!(!c.layers[0].waiting[10] && c.layers[0].waiting[14]);
        let jj = c.take_jobs(4);
        assert_eq!(jj.len(), 1);
        assert_eq!(jj[0].expert, 14);
    }

    #[test]
    fn mirrors_randomized_multi_launch_stress_invariants() {
        // Interleaves launches, completions, evictions, fills, and publishes while verifying
        // NO slot is ever overwritten while a launch that planned it is still in flight.
        let layers = 4usize;
        let n_experts = 64usize;
        let n_slots = 32usize;
        let reserve = 2usize;

        let mut book = MoeBook {
            policy: MoePolicy::Lru,
            reserve,
            ..MoeBook::default()
        };
        let mut table = vec![vec![-1i32; n_experts]; layers];
        let mut content = vec![vec![-1i32; n_slots]; layers];
        let init: Vec<usize> = (0..n_slots).collect();
        let pub_n = n_slots - reserve;
        for h in 0..layers {
            book.add_layer(n_experts, &init);
            for s in 0..n_slots {
                content[h][s] = s as i32;
                if s < pub_n {
                    table[h][s] = s as i32;
                }
            }
        }

        type ActiveLaunch = (u64, Vec<(usize, usize, i32)>);
        let mut active: VecDeque<ActiveLaunch> = VecDeque::new();
        let mut launched = 0u64;
        let mut filling: Vec<FillJob> = Vec::new();
        let mut rng = SimpleRng(0x9E37_79B9_7F4A_7C15);
        let mut hits = 0u64;
        let mut events = 0u64;

        for step in 0..1500usize {
            // Issue a launch reading a skewed working set of experts across all layers
            launched += 1;
            let mut reads = Vec::new();
            for h in 0..layers {
                let mut ids = Vec::with_capacity(4);
                let mut dev = Vec::with_capacity(4);
                for _ in 0..4 {
                    let e = if rng.f64() < 0.92 {
                        rng.usize(24) // hot working set fits inside 30 resident slots
                    } else {
                        rng.usize(n_experts)
                    };
                    ids.push(e as i32);
                    let s = table[h][e];
                    events += 1;
                    if s >= 0 {
                        assert_eq!(
                            content[h][s as usize], e as i32,
                            "stale slot content at read!"
                        );
                        reads.push((h, s as usize, e as i32));
                        dev.push(1);
                        hits += 1;
                    } else {
                        dev.push(0);
                    }
                }
                book.observe(h, &ids, Some(&dev));
            }
            active.push_back((launched, reads));

            // Evict & bind current launch epoch
            for ev in book.take_evictions() {
                assert_eq!(table[ev.layer][ev.expert], ev.slot as i32);
                table[ev.layer][ev.expert] = -1;
                book.unpublished(ev, launched);
            }

            // Start fills: assert NO in-flight launch planned this slot and table does not name it!
            for job in book.take_jobs(4) {
                for (_, rlist) in &active {
                    for &(rh, rs, _) in rlist {
                        assert!(
                            !(rh == job.layer && rs == job.slot),
                            "RACE: slot overwritten while active launch reads it!"
                        );
                    }
                }
                for &slot_val in &table[job.layer][..n_experts] {
                    assert_ne!(slot_val, job.slot as i32);
                }
                content[job.layer][job.slot] = -2; // writing
                filling.push(job);
            }

            // Complete oldest launch periodically
            if active.len() >= 3 || (step % 2 == 0 && !active.is_empty()) {
                let (seq, rlist) = active.pop_front().unwrap();
                for (rh, rs, re) in rlist {
                    assert_eq!(content[rh][rs], re, "slot mutated before launch finished!");
                }
                book.quiesce(seq);
            }

            // Finish in-flight fills
            if !filling.is_empty() && step % 2 == 1 {
                for job in filling.drain(..) {
                    content[job.layer][job.slot] = job.expert as i32;
                    table[job.layer][job.expert] = job.slot as i32;
                    book.loaded(job);
                    book.published(job);
                }
            }
        }

        // Drain all active launches
        while let Some((seq, rlist)) = active.pop_front() {
            for (rh, rs, re) in rlist {
                assert_eq!(content[rh][rs], re);
            }
            book.quiesce(seq);
        }
        let hit_pct = (hits as f64) * 100.0 / (events as f64);
        assert!(
            hit_pct >= 90.0,
            "expected >= 90% LRU hit rate on hot set, got {hit_pct:.2}%"
        );
    }
}
