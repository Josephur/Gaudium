//! Prefix reuse across requests of the dense Qwen3.5 / Qwen3.8 engine: a request whose prompt
//! starts with rows the engine still holds state for prefills only the rest.
//!
//! This is a child module of `engine::qwen3` (declared there with `#[path]`).
//!
//! What a later request can take over, and why (measured on the card and in the kernel sources):
//! - **K/V rows of the attention layers** are indexed by position and stay where the prefill
//!   wrote them. Nothing has to be kept for them: with one lane the previous request's rows are
//!   still on the card when the next one arrives.
//! - **The recurrent and conv state of the Gated DeltaNet layers** is not positional: prefill and
//!   decode overwrite it as they go. To resume at a position the engine needs the state of every
//!   such layer at exactly that position. A prefill tile already writes that state when it ends;
//!   here a tile whose end is worth keeping writes it straight into a planned device slot
//!   instead of the ping-pong buffer, and the tile after it reads it from there. Keeping a
//!   position costs no copy and no launch, and resuming from it costs none either: the first
//!   tile of a resumed prefill reads the slot as its input state.
//! - **Rows written by decode are never reused.** A decode step and a prefill tile compute the
//!   same rows with different node shapes (single-row recurrence against the chunked one, f32
//!   attention scores against the tile's half-width core), so their bits differ. The reusable
//!   prefix is therefore the previous *prompt*, and only at ends of whole prefill tiles counted
//!   from position 0: the tiles a cold prefill of the new prompt would run are then the same
//!   recipes on the same rows, and the state they start from is the one the cold run would have
//!   carried there.
//! - **The drafter's context rows** are positional too. A drafter with a sliding window keeps
//!   them in a ring; a long generation overwrites the oldest rows of the next request's window,
//!   so the request tells the block attention the first position whose ring row is still that
//!   position's (`pc_ring_flo`), which the ring softmax kernel takes as its lower bound.
//!
//! Every decision is the verified core's (`verified/prefix_reuse.bend`, and `pfx_reuse_valid` in
//! `verified/kv_pool.bend`): how far two prompts are the same, what a slot is worth for a new
//! prompt, which positions a request may write, which slot takes one, how many slots fit beside
//! the rest of the device plan. This module holds the table and walks it.

use super::*;
use crate::core::kv_pool::{is_prefix_reuse_valid, PrefixReuseQuery};
use crate::verified::{
    pc_abort, pc_anchor_tiles, pc_begin_write, pc_boundary, pc_cap, pc_capture_ok, pc_class_usable,
    pc_commit_done, pc_cost, pc_image_limit, pc_image_same, pc_is_free, pc_ladder_tiles,
    pc_may_take, pc_prefill_rows, pc_readable, pc_request_may_reuse, pc_retain, pc_reused_rows,
    pc_ring_flo, pc_ring_from, pc_slot_tiles, pc_victim_better, pc_victim_class, PcSlot,
};

/// Positions an engine keeps when the configuration names no count.
pub const DEFAULT_SLOTS: u64 = 8;

/// One image run of a prompt: where it is, its token grid and the pixels the tower reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageId {
    pub start: u64,
    pub len: u64,
    pub rows: u64,
    pub cols: u64,
    pub pixels: [u64; 2],
}

/// 128-bit digest of a tower input (not cryptographic: it tells two pictures of one
/// conversation apart, it does not resist someone constructing a collision on purpose).
pub fn pixel_digest(pixels: &[u16], grid: (usize, usize)) -> [u64; 2] {
    const K0: u64 = 0x9E37_79B9_7F4A_7C15;
    const K1: u64 = 0xC2B2_AE3D_27D4_EB4F;
    let mut a = K0 ^ (grid.0 as u64) ^ ((pixels.len() as u64) << 20);
    let mut b = K1 ^ (grid.1 as u64) ^ ((pixels.len() as u64) << 24);
    let (words, rest) = pixels.as_chunks::<4>();
    for w in words {
        let v = u64::from(w[0])
            | (u64::from(w[1]) << 16)
            | (u64::from(w[2]) << 32)
            | (u64::from(w[3]) << 48);
        a = (a ^ v).wrapping_mul(K0).rotate_left(29);
        b = (b.rotate_left(31) ^ v).wrapping_mul(K1);
    }
    for (i, &p) in rest.iter().enumerate() {
        let v = u64::from(p) << (16 * i);
        a = (a ^ v).wrapping_mul(K0).rotate_left(29);
        b = (b.rotate_left(31) ^ v).wrapping_mul(K1);
    }
    // final avalanche
    let fin = |mut x: u64| {
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        x ^ (x >> 33)
    };
    [fin(a ^ b.rotate_left(17)), fin(b ^ a.rotate_left(41))]
}

/// One state slot of the table.
#[derive(Debug, Clone, Copy)]
struct Slot {
    state: PcSlot,
    /// Rank on the ladder of the request that wrote it (0 = the position nearest its prompt end).
    rank: u64,
    /// Request that wrote it or last resumed from it.
    seq: u64,
}

/// What one request does with the held state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    /// Rows taken over: the prefill starts at this position (0 = a cold prefill).
    pub resume: u64,
    /// Slot the first tile reads its state from (`None` for a cold prefill).
    pub resume_slot: Option<usize>,
    /// `(tiles, slot)`: the state after `tiles` whole tiles is written into `slot`.
    pub captures: Vec<(u64, usize)>,
    /// Common token prefix with the held prompt, and what is left of it at the first image that
    /// differs.
    pub lcp: u64,
    pub cap: u64,
    /// First position whose drafter context row is valid for this request.
    pub ring_flo: u64,
}

impl Plan {
    /// Slot the tile that ends after `tiles` whole tiles writes its state into, if any.
    pub fn capture_at(&self, tiles: u64) -> Option<usize> {
        self.captures
            .iter()
            .find(|&&(t, _)| t == tiles)
            .map(|&(_, s)| s)
    }
}

/// Counters of the prefix cache since the engine was loaded, and its last request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixStats {
    pub slots: u64,
    pub slot_bytes: u64,
    pub held_positions: u64,
    pub held_rows: u64,
    pub requests: u64,
    pub hits: u64,
    pub reused_rows: u64,
    pub last_prompt_rows: u64,
    pub last_reused_rows: u64,
    pub last_lcp: u64,
}

/// The table of kept positions of one loaded engine, and the prompt they belong to.
#[derive(Debug)]
pub struct PrefixCache {
    tile: u64,
    slots: Vec<Slot>,
    /// Prompt whose prefill wrote the K/V rows now on the device.
    held: Vec<i32>,
    held_images: Vec<ImageId>,
    seq: u64,
    /// The engine can resume a prefill (model family, drafter, reset mode).
    supported: bool,
    /// Rows of the drafter's context caches (a ring's rows, or the context size of linear
    /// caches); 0 without a drafter that keeps context rows.
    ctx_rows: u64,
    /// First position whose drafter context row is valid.
    ring_from: u64,
    /// The running request's resume point (for the ring bound at its end).
    cur_resume: u64,
    stats: PrefixStats,
}

fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Slot limit of reuse from the image runs of a held prompt and of a new one: the first pair
/// that is not the same picture at the same place stops reuse before it. A prompt with more
/// runs than the other stops it before its first extra run.
pub fn image_limit(held_images: &[ImageId], images: &[ImageId]) -> u64 {
    let n = held_images.len().max(images.len());
    let mut limit = u64::MAX;
    for k in (0..n).rev() {
        let (h, w) = (held_images.get(k), images.get(k));
        let same = match (h, w) {
            (Some(h), Some(w)) => pc_image_same(
                h.start == w.start,
                h.len == w.len && h.rows == w.rows && h.cols == w.cols,
                h.pixels == w.pixels,
            ),
            _ => false,
        };
        limit = pc_image_limit(
            same,
            h.map_or(u64::MAX, |x| x.start),
            w.map_or(u64::MAX, |x| x.start),
            limit,
        );
    }
    limit
}

/// How far a new prompt is a held one: `(common token prefix, rows that count as the same)`,
/// the second cut at the first image that differs (the verified `pc_cap`). The one rule for a
/// prompt held on the device and for one kept in host memory.
pub fn shared_rows(
    prompt: &[i32],
    images: &[ImageId],
    held: &[i32],
    held_images: &[ImageId],
) -> (u64, u64) {
    let lcp = common_prefix(prompt, held) as u64;
    (lcp, pc_cap(lcp, image_limit(held_images, images)))
}

/// What the device alone gives a request, without changing the table ([`PrefixCache::peek`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Peek {
    /// Rows the prompt shares with the held one.
    pub cap: u64,
    /// Position a plan would resume at (0 = a cold prefill).
    pub resume: u64,
}

impl PrefixCache {
    pub fn new(tile: u64, slots: u64, slot_bytes: u64, supported: bool, ctx_rows: u64) -> Self {
        let n = if supported && tile > 0 { slots } else { 0 };
        Self {
            tile,
            slots: vec![
                Slot {
                    state: PcSlot::PcEmpty,
                    rank: 0,
                    seq: 0,
                };
                n as usize
            ],
            held: Vec::new(),
            held_images: Vec::new(),
            seq: 0,
            supported: supported && n > 0,
            ctx_rows,
            ring_from: 0,
            cur_resume: 0,
            stats: PrefixStats {
                slots: n,
                slot_bytes,
                ..PrefixStats::default()
            },
        }
    }

    /// An engine without prefix reuse.
    pub fn off() -> Self {
        Self::new(0, 0, 0, false, 0)
    }

    pub fn enabled(&self) -> bool {
        self.supported
    }

    pub fn stats(&self) -> PrefixStats {
        let mut s = self.stats;
        s.held_positions = self
            .slots
            .iter()
            .filter(|x| pc_readable(x.state))
            .count() as u64;
        s.held_rows = self.held.len() as u64;
        s
    }

    /// Kept positions in rows, ascending (tests and the log).
    pub fn held_boundaries(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self
            .slots
            .iter()
            .filter(|x| pc_readable(x.state))
            .map(|x| pc_boundary(pc_slot_tiles(x.state), self.tile))
            .collect();
        v.sort_unstable();
        v
    }

    /// Forgets everything (a request failed, or the K/V rows were zeroed).
    pub fn clear(&mut self) {
        for s in &mut self.slots {
            s.state = PcSlot::PcEmpty;
        }
        self.held.clear();
        self.held_images.clear();
        self.ring_from = 0;
        self.cur_resume = 0;
    }

    /// The kept position a request resumes from: the furthest one the verified rule allows
    /// among `states` (the slots as the request's prompt leaves them), `(position, slot)`.
    fn best_resume(
        &self,
        states: &[PcSlot],
        cap: u64,
        prompt_len: u64,
        wanted: bool,
        rows_zeroed: bool,
    ) -> (u64, Option<usize>) {
        let mut resume = 0u64;
        let mut resume_slot = None;
        if pc_request_may_reuse(wanted, rows_zeroed, self.supported) {
            for (i, &state) in states.iter().enumerate() {
                if !pc_readable(state) {
                    continue;
                }
                let pos = pc_boundary(pc_slot_tiles(state), self.tile);
                let ok = is_prefix_reuse_valid(PrefixReuseQuery {
                    snap_pos: pos,
                    lcp: cap,
                    prompt_len,
                    tile: self.tile,
                    need_mtp: false,
                    mtp_upto: 0,
                    need_dflash: self.ctx_rows > 0,
                    dflash_upto: self.held.len() as u64,
                });
                if ok && pos > resume {
                    resume = pos;
                    resume_slot = Some(i);
                }
            }
        }
        (resume, resume_slot)
    }

    /// What [`Self::plan`] would do for this prompt with the state the device holds now, without
    /// changing the table: the rows the prompt shares with the held one and the position its
    /// prefill would resume at. The host-RAM tier asks this before it decides to restore.
    pub fn peek(
        &self,
        prompt: &[i32],
        images: &[ImageId],
        wanted: bool,
        rows_zeroed: bool,
    ) -> Peek {
        if rows_zeroed || !self.supported {
            return Peek::default();
        }
        let (_, cap) = shared_rows(prompt, images, &self.held, &self.held_images);
        let states: Vec<PcSlot> = self
            .slots
            .iter()
            .map(|s| pc_retain(s.state, cap, self.tile))
            .collect();
        let (resume, _) = self.best_resume(&states, cap, prompt.len() as u64, wanted, rows_zeroed);
        Peek { cap, resume }
    }

    /// Plans a request: where its prefill resumes, which slots it reads and writes. `wanted`:
    /// the request allows reuse; `rows_zeroed`: the K/V rows were zeroed since the held prompt
    /// was prefilled. After this call the table describes the running request; `tiles_done`,
    /// `prefill_end` and `request_end` close it.
    pub fn plan(
        &mut self,
        prompt: &[i32],
        images: &[ImageId],
        wanted: bool,
        rows_zeroed: bool,
    ) -> Plan {
        self.plan_from(prompt, images, wanted, rows_zeroed, None)
    }

    /// [`Self::plan`] for a request whose rows were just restored from a prompt kept in host
    /// memory ([`Self::adopt_restored`]): `restored` = `(common token prefix, shared rows)` of
    /// the new prompt and that whole saved prompt. The table holds only the restored part of
    /// it, so the rows the two prompts share are taken from the comparison the restore made.
    pub fn plan_from(
        &mut self,
        prompt: &[i32],
        images: &[ImageId],
        wanted: bool,
        rows_zeroed: bool,
        restored: Option<(u64, u64)>,
    ) -> Plan {
        self.seq += 1;
        self.stats.requests += 1;
        let p = prompt.len() as u64;
        let tile = self.tile;
        if rows_zeroed || !self.supported {
            self.clear();
        }
        if !self.supported {
            self.stats.last_prompt_rows = p;
            self.stats.last_reused_rows = 0;
            self.stats.last_lcp = 0;
            return Plan::default();
        }
        // how far the new prompt is the held one (a cold request still leaves those rows as
        // they are: it writes the same rows again)
        let (lcp, cap) = match restored.filter(|_| !rows_zeroed) {
            Some(shared) => shared,
            None => shared_rows(prompt, images, &self.held, &self.held_images),
        };
        for s in &mut self.slots {
            s.state = pc_retain(s.state, cap, tile);
        }

        // the kept position to resume from: the furthest one the verified rule allows
        let states: Vec<PcSlot> = self.slots.iter().map(|s| s.state).collect();
        let (resume, resume_slot) = self.best_resume(&states, cap, p, wanted, rows_zeroed);
        let resume_tiles = resume / tile.max(1);

        // the positions this request keeps: the ladder behind its anchor, each into the slot
        // the verified order picks
        let anchor = pc_anchor_tiles(p, tile);
        let mut captures: Vec<(u64, usize)> = Vec::new();
        for rank in 0..self.slots.len() as u64 {
            let k = pc_ladder_tiles(anchor, rank);
            if !pc_capture_ok(k, resume_tiles, p, tile) {
                break;
            }
            if captures.iter().any(|&(t, _)| t == k) {
                continue;
            }
            // a position that is still held needs no second slot
            if let Some(s) = self
                .slots
                .iter_mut()
                .find(|s| pc_readable(s.state) && pc_slot_tiles(s.state) == k)
            {
                s.seq = self.seq;
                s.rank = s.rank.min(rank);
                continue;
            }
            // the tile that writes position k is tile k - 1; the resume slot is read by tile
            // `resume_tiles`
            let same_tile = k == resume_tiles + 1;
            let mut best: Option<(usize, u64, u64)> = None;
            for (i, s) in self.slots.iter().enumerate() {
                let class = pc_victim_class(s.state, resume_slot == Some(i), same_tile);
                let cost = pc_cost(s.rank, self.seq, s.seq);
                let better = match best {
                    None => true,
                    Some((_, bc, bk)) => pc_victim_better(class, cost, bc, bk),
                };
                if better {
                    best = Some((i, class, cost));
                }
            }
            let Some((i, class, cost)) = best else { break };
            if !pc_class_usable(class) || !pc_may_take(pc_cost(rank, self.seq, self.seq), class, cost)
            {
                // nothing this request keeps from here on is worth more than what is held
                break;
            }
            debug_assert!(pc_is_free(self.slots[i].state) || pc_readable(self.slots[i].state));
            self.slots[i] = Slot {
                state: pc_begin_write(self.slots[i].state, k),
                rank,
                seq: self.seq,
            };
            captures.push((k, i));
        }

        // a position that was resumed from counts as used now (after the choice above, which
        // weighs it by its age before this request)
        if let Some(s) = resume_slot.map(|i| &mut self.slots[i]) {
            if pc_readable(s.state) {
                s.seq = self.seq;
            }
        }

        // the rows before the resume point stay; the rest is this request's to write
        self.held.truncate(resume as usize);
        self.held_images = images
            .iter()
            .copied()
            .filter(|im| im.start < resume)
            .collect();
        let ring_flo = pc_ring_flo(self.ring_from, resume);
        self.cur_resume = resume;

        let reused = pc_reused_rows(p, resume);
        debug_assert_eq!(reused + pc_prefill_rows(p, resume), p);
        self.stats.last_prompt_rows = p;
        self.stats.last_reused_rows = reused;
        self.stats.last_lcp = lcp;
        if reused > 0 {
            self.stats.hits += 1;
            self.stats.reused_rows += reused;
        }
        Plan {
            resume,
            resume_slot,
            captures,
            lcp,
            cap,
            ring_flo,
        }
    }

    /// The prefill has finished (synchronised on the device) every tile up to `tiles` whole
    /// tiles of `prompt`: the positions written by those tiles are held from now on, and the
    /// rows up to there are the prompt's.
    pub fn tiles_done(&mut self, prompt: &[i32], images: &[ImageId], tiles: u64) {
        if !self.supported {
            return;
        }
        // which writes are finished is the verified function's decision (`pc_commit_done`)
        for s in &mut self.slots {
            s.state = pc_commit_done(s.state, tiles);
        }
        self.extend_held(prompt, images, pc_boundary(tiles, self.tile));
    }

    fn extend_held(&mut self, prompt: &[i32], images: &[ImageId], rows: u64) {
        let rows = (rows as usize).min(prompt.len());
        if rows > self.held.len() {
            let from = self.held.len();
            self.held.extend_from_slice(&prompt[from..rows]);
            self.held_images = images
                .iter()
                .copied()
                .filter(|im| im.start < rows as u64)
                .collect();
        }
    }

    /// The prefill ended with `rows` rows of `prompt` written (all of it, or the whole tiles a
    /// cancelled prefill finished): positions whose tile did not finish are dropped.
    pub fn prefill_end(&mut self, prompt: &[i32], images: &[ImageId], rows: u64) {
        if !self.supported {
            return;
        }
        for s in &mut self.slots {
            s.state = pc_abort(s.state);
        }
        self.extend_held(prompt, images, rows);
    }

    /// The request ended. `ok`: without an error. `written_end`: one past the last row it may
    /// have written (prompt, generated tokens and the step slack).
    pub fn request_end(&mut self, ok: bool, written_end: u64) {
        if !self.supported {
            return;
        }
        if !ok {
            self.clear();
            return;
        }
        for s in &mut self.slots {
            s.state = pc_abort(s.state);
        }
        if self.ctx_rows > 0 {
            self.ring_from =
                pc_ring_from(self.ring_from, self.cur_resume, written_end, self.ctx_rows);
        }
    }

    // ---- what the host-RAM tier reads and writes (`engine/host_tier.rs`) ----

    pub fn tile(&self) -> u64 {
        self.tile
    }

    /// The prompt whose prefill wrote the K/V rows now on the device, and its image runs.
    pub fn held(&self) -> (&[i32], &[ImageId]) {
        (&self.held, &self.held_images)
    }

    /// Kept positions as `(whole tiles, slot)`, ascending.
    pub fn held_slots(&self) -> Vec<(u64, usize)> {
        let mut v: Vec<(u64, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| pc_readable(s.state))
            .map(|(i, s)| (pc_slot_tiles(s.state), i))
            .collect();
        v.sort_unstable();
        v
    }

    /// Rows up to the last kept position (0: none is kept).
    pub fn kept_rows(&self) -> u64 {
        self.held_slots()
            .last()
            .map_or(0, |&(t, _)| pc_boundary(t, self.tile))
    }

    pub fn slot_count(&self) -> u64 {
        self.slots.len() as u64
    }

    /// First position whose drafter context row is valid.
    pub fn ring_from(&self) -> u64 {
        self.ring_from
    }

    /// Rows of the drafter's context caches (0: no drafter that keeps context rows).
    pub fn ctx_rows(&self) -> u64 {
        self.ctx_rows
    }

    /// The device rows `[0, tokens.len())` are about to be replaced by a copy of the rows of
    /// `tokens` (a prompt kept in host memory): the table now describes that prompt. Kept
    /// positions stay where the old held prompt and the restored one are the same up to them
    /// (their state is the state of the same rows); every other one is dropped. Returns the
    /// slot the state after `tiles` whole tiles is to be copied into, in the middle of a write
    /// until [`Self::restore_done`] (`None`: a slot already holds that state, or the table
    /// has no slot for it).
    pub fn adopt_restored(
        &mut self,
        tokens: &[i32],
        images: &[ImageId],
        tiles: u64,
        ring_from: u64,
    ) -> Option<usize> {
        if !self.supported {
            return None;
        }
        let (_, cap) = shared_rows(tokens, images, &self.held, &self.held_images);
        for s in &mut self.slots {
            s.state = pc_retain(s.state, cap, self.tile);
        }
        self.held = tokens.to_vec();
        self.held_images = images.to_vec();
        self.ring_from = ring_from;
        self.cur_resume = 0;
        if self
            .slots
            .iter()
            .any(|s| pc_readable(s.state) && pc_slot_tiles(s.state) == tiles)
        {
            return None;
        }
        // the slot for the restored state: the verified order of the device tier (a free slot
        // first, then the held position worth least); no slot is read by a tile now
        let mut best: Option<(usize, u64, u64)> = None;
        for (i, s) in self.slots.iter().enumerate() {
            let class = pc_victim_class(s.state, false, false);
            let cost = pc_cost(s.rank, self.seq, s.seq);
            let better = match best {
                None => true,
                Some((_, bc, bk)) => pc_victim_better(class, cost, bc, bk),
            };
            if better {
                best = Some((i, class, cost));
            }
        }
        let (i, class, _) = best?;
        if !pc_class_usable(class) {
            return None;
        }
        self.slots[i] = Slot {
            state: pc_begin_write(self.slots[i].state, tiles),
            rank: 0,
            seq: self.seq,
        };
        Some(i)
    }

    /// The copy into the slots [`Self::adopt_restored`] began has ended. `done_tiles`: the whole
    /// tiles whose rows and state are complete on the device (0 after a copy that failed). A
    /// slot whose state is not complete holds nothing.
    pub fn restore_done(&mut self, done_tiles: u64) {
        for s in &mut self.slots {
            s.state = pc_abort(pc_commit_done(s.state, done_tiles));
        }
    }
}

/// Device addresses of one slot: `(recurrent state, conv state)` per Gated DeltaNet layer.
pub(super) type SlotAddrs = BTreeMap<usize, (u64, u64)>;

/// State buffers of one prefill tile of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StateIo {
    pub s_in: u64,
    pub s_out: u64,
    pub cs_in: u64,
    pub cs_out: u64,
}

/// The state buffers of layer `l` for the tile that starts after `tile_idx` whole tiles, whose
/// ping-pong buffers are `st.s[parity]` (in) and `st.s[1 - parity]` (out): a tile that starts at
/// a kept position reads the slot, a tile that ends at one writes the slot.
pub(super) fn state_io(
    st: &GdnLayerState,
    slots: &[SlotAddrs],
    plan: &Plan,
    tile: u64,
    l: usize,
    tile_idx: u64,
    parity: usize,
) -> Result<StateIo, String> {
    let mut io = StateIo {
        s_in: st.s[parity],
        s_out: st.s[1 - parity],
        cs_in: st.cs[parity],
        cs_out: st.cs[1 - parity],
    };
    let of = |slot: usize| -> Result<(u64, u64), String> {
        slots
            .get(slot)
            .and_then(|m| m.get(&l))
            .copied()
            .ok_or_else(|| format!("prefix cache: slot {slot} has no state buffer for layer {l}"))
    };
    let resume_tiles = plan.resume / tile.max(1);
    let src = if tile_idx == resume_tiles && plan.resume > 0 {
        plan.resume_slot
    } else {
        plan.capture_at(tile_idx)
    };
    if let Some(slot) = src {
        (io.s_in, io.cs_in) = of(slot)?;
    }
    if let Some(slot) = plan.capture_at(tile_idx + 1) {
        (io.s_out, io.cs_out) = of(slot)?;
    }
    if io.s_in == io.s_out || io.cs_in == io.cs_out {
        return Err(format!(
            "prefix cache: tile {tile_idx} of layer {l} would read and write one state buffer"
        ));
    }
    Ok(io)
}

/// The image runs of a request as the table compares them.
pub(super) fn image_ids(
    images: &[crate::vision::PreparedImage],
    layout: &crate::vision::PromptLayout,
) -> Vec<ImageId> {
    images
        .iter()
        .zip(&layout.spans)
        .map(|(img, span)| ImageId {
            start: span.start as u64,
            len: span.len as u64,
            rows: span.rows as u64,
            cols: span.cols as u64,
            pixels: pixel_digest(&img.pixels, (img.grid_h, img.grid_w)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: u64 = 512;

    fn cache(slots: u64) -> PrefixCache {
        PrefixCache::new(T, slots, 1, true, 0)
    }

    fn prompt(n: usize, salt: i32) -> Vec<i32> {
        (0..n as i32).map(|i| i.wrapping_mul(31) ^ salt).collect()
    }

    /// Runs a whole request on the table: plan, every tile, the end.
    fn run(c: &mut PrefixCache, p: &[i32], images: &[ImageId], wanted: bool) -> Plan {
        let plan = c.plan(p, images, wanted, false);
        c.tiles_done(p, images, p.len() as u64 / T);
        c.prefill_end(p, images, p.len() as u64);
        c.request_end(true, p.len() as u64 + 600);
        plan
    }

    #[test]
    fn a_first_request_is_cold_and_keeps_a_ladder_behind_its_prompt_end() {
        let mut c = cache(8);
        let p = prompt(8 * 512 + 100, 0);
        let plan = run(&mut c, &p, &[], true);
        assert_eq!(plan.resume, 0);
        assert_eq!(plan.resume_slot, None);
        // anchor: the last tile end the guard leaves (8 tiles), then 1, 3 and 7 tiles back
        assert_eq!(c.held_boundaries(), vec![512, 2560, 3584, 4096]);
        assert!(plan.captures.iter().all(|&(k, _)| k * T < p.len() as u64));
    }

    #[test]
    fn an_appended_turn_resumes_at_the_last_kept_position() {
        let mut c = cache(8);
        let p1 = prompt(3000, 0);
        run(&mut c, &p1, &[], true);
        // the next turn drops the last few rows of the old prompt and appends
        let mut p2 = p1[..2990].to_vec();
        p2.extend(prompt(700, 7));
        let plan = run(&mut c, &p2, &[], true);
        assert_eq!(plan.lcp, 2990);
        assert_eq!(plan.resume, 2560);
        assert!(plan.resume < plan.cap);
        assert_eq!(c.stats().last_reused_rows, 2560);
        assert_eq!(c.stats().hits, 1);
        // the new prompt's own anchor is kept too
        assert!(c.held_boundaries().contains(&3584));
    }

    #[test]
    fn the_same_prompt_again_still_prefills_its_last_rows() {
        let mut c = cache(4);
        let p = prompt(2048, 3);
        run(&mut c, &p, &[], true);
        let plan = run(&mut c, &p, &[], true);
        // never the prompt end, never past the guard
        assert_eq!(plan.resume, 1536);
        assert!(plan.resume < p.len() as u64);
    }

    #[test]
    fn a_prompt_that_differs_early_drops_the_positions_behind_the_difference() {
        let mut c = cache(8);
        let p1 = prompt(6000, 0);
        run(&mut c, &p1, &[], true);
        assert!(c.held_boundaries().contains(&5632));
        let mut p2 = p1[..1500].to_vec();
        p2.extend(prompt(3000, 9));
        let plan = c.plan(&p2, &[], true, false);
        assert_eq!(plan.cap, 1500);
        // 5632 - 7 * 512 = 2048 is past the difference; nothing is held below it
        assert_eq!(plan.resume, 0);
        assert!(c.held_boundaries().iter().all(|&b| b <= 1500));
    }

    #[test]
    fn a_cold_request_does_not_resume_but_keeps_what_it_shares() {
        let mut c = cache(8);
        let p = prompt(3000, 0);
        run(&mut c, &p, &[], true);
        let before = c.held_boundaries();
        let plan = run(&mut c, &p, &[], false);
        assert_eq!(plan.resume, 0);
        assert_eq!(c.held_boundaries(), before);
        assert_eq!(c.stats().hits, 0);
    }

    #[test]
    fn zeroed_rows_and_failed_requests_forget_everything() {
        let mut c = cache(8);
        let p = prompt(3000, 0);
        run(&mut c, &p, &[], true);
        let plan = c.plan(&p, &[], true, true);
        assert_eq!(plan.resume, 0);
        c.tiles_done(&p, &[], 5);
        c.prefill_end(&p, &[], 3000);
        c.request_end(false, 3600);
        assert!(c.held_boundaries().is_empty());
        assert_eq!(c.plan(&p, &[], true, false).resume, 0);
    }

    #[test]
    fn a_cancelled_prefill_keeps_only_the_positions_its_finished_tiles_wrote() {
        let mut c = cache(8);
        let p = prompt(10 * 512 + 50, 0);
        let plan = c.plan(&p, &[], true, false);
        assert!(plan.captures.len() >= 3);
        // cancelled after 4 whole tiles
        c.tiles_done(&p, &[], 4);
        c.prefill_end(&p, &[], 4 * 512);
        c.request_end(true, 4 * 512 + 600);
        assert!(c.held_boundaries().iter().all(|&b| b <= 4 * 512));
        assert_eq!(c.stats().held_rows, 4 * 512);
        // the same prompt again resumes from what was finished
        let again = c.plan(&p, &[], true, false);
        assert!(again.resume > 0 && again.resume <= 4 * 512);
    }

    #[test]
    fn another_picture_at_the_same_place_stops_reuse_before_it() {
        let mut c = cache(8);
        let p = prompt(4000, 0);
        let img = |px: u64| ImageId {
            start: 1100,
            len: 256,
            rows: 16,
            cols: 16,
            pixels: [px, 1],
        };
        run(&mut c, &p, &[img(1)], true);
        // same tokens, same picture: reuse past the image
        let same = c.plan(&p, &[img(1)], true, false);
        assert_eq!(same.cap, 4000);
        assert!(same.resume > 1100 + 256);
        c.tiles_done(&p, &[img(1)], 7);
        c.prefill_end(&p, &[img(1)], 4000);
        c.request_end(true, 4600);
        // same tokens, another picture: reuse stops before its run
        let other = c.plan(&p, &[img(2)], true, false);
        assert_eq!(other.lcp, 4000);
        assert_eq!(other.cap, 1100);
        assert!(other.resume <= 1100);
        // a prompt that lost the picture (text there instead) cannot pass its start either
        let mut c2 = cache(8);
        run(&mut c2, &p, &[img(1)], true);
        assert_eq!(c2.plan(&p, &[], true, false).cap, 1100);
    }

    #[test]
    fn slots_go_to_free_places_first_and_then_to_the_least_valuable_position() {
        let mut c = cache(2);
        let p1 = prompt(5 * 512 + 100, 0);
        run(&mut c, &p1, &[], true);
        // two slots: the anchor (5 tiles) and one tile behind it
        assert_eq!(c.held_boundaries(), vec![2048, 2560]);
        // an appended turn resumes at 2560 and keeps its own anchor in the other slot
        let mut p2 = p1.clone();
        p2.extend(prompt(1500, 5));
        let plan = run(&mut c, &p2, &[], true);
        assert_eq!(plan.resume, 2560);
        assert_eq!(c.held_boundaries(), vec![2560, 4096]);
    }

    #[test]
    fn one_slot_moves_forward_but_never_in_the_tile_that_reads_it() {
        let mut c = cache(1);
        let p1 = prompt(2 * 512 + 100, 0);
        run(&mut c, &p1, &[], true);
        assert_eq!(c.held_boundaries(), vec![1024]);
        // one tile further: the tile that would write the slot is the one that reads it
        let mut p2 = p1.clone();
        p2.extend(prompt(512, 5));
        let plan = run(&mut c, &p2, &[], true);
        assert_eq!(plan.resume, 1024);
        assert!(plan.captures.is_empty());
        assert_eq!(c.held_boundaries(), vec![1024]);
        // two tiles further: a later tile writes it
        let mut p3 = p2.clone();
        p3.extend(prompt(1024, 6));
        let plan = run(&mut c, &p3, &[], true);
        assert_eq!(plan.resume, 1024);
        assert_eq!(plan.captures.len(), 1);
        assert_eq!(c.held_boundaries(), vec![2560]);
    }

    #[test]
    fn state_buffers_follow_the_plan() {
        let st = GdnLayerState {
            s: [100, 200],
            s_bytes: 1,
            cs: [300, 400],
            cs_bytes: 1,
        };
        let slots: Vec<SlotAddrs> = (0..3u64)
            .map(|i| BTreeMap::from([(0usize, (1000 + i, 2000 + i))]))
            .collect();
        let plan = Plan {
            resume: 2 * T,
            resume_slot: Some(1),
            captures: vec![(4, 0), (5, 2)],
            ..Plan::default()
        };
        let io = |tile_idx: u64, parity: usize| state_io(&st, &slots, &plan, T, 0, tile_idx, parity).unwrap();
        // the first tile reads the resume slot and writes the ping-pong buffer
        assert_eq!(io(2, 0), StateIo { s_in: 1001, s_out: 200, cs_in: 2001, cs_out: 400 });
        // the tile that ends at a kept position writes its slot
        assert_eq!(io(3, 1), StateIo { s_in: 200, s_out: 1000, cs_in: 400, cs_out: 2000 });
        // the next one reads that slot and writes the next kept position
        assert_eq!(io(4, 0), StateIo { s_in: 1000, s_out: 1002, cs_in: 2000, cs_out: 2002 });
        // and the one after it is back on the ping-pong buffers
        assert_eq!(io(5, 1), StateIo { s_in: 1002, s_out: 100, cs_in: 2002, cs_out: 300 });
        assert_eq!(io(6, 0), StateIo { s_in: 100, s_out: 200, cs_in: 300, cs_out: 400 });
        // a cold plan binds what the loop always bound
        let cold = Plan::default();
        assert_eq!(
            state_io(&st, &slots, &cold, T, 0, 0, 0).unwrap(),
            StateIo { s_in: 100, s_out: 200, cs_in: 300, cs_out: 400 }
        );
    }

    #[test]
    fn the_ring_bound_follows_what_a_request_overwrote() {
        // a ring of 8960 rows (window 8192)
        let mut c = PrefixCache::new(T, 8, 1, true, 8960);
        let p1 = prompt(20_000, 0);
        let plan = c.plan(&p1, &[], true, false);
        assert_eq!(plan.ring_flo, 0);
        c.tiles_done(&p1, &[], 39);
        c.prefill_end(&p1, &[], 20_000);
        // 3000 generated tokens: rows up to 23_000 + slack were written
        c.request_end(true, 23_512);
        // an appended turn: everything below 23_512 - 8960 was overwritten
        let mut p2 = p1.clone();
        p2.extend(prompt(400, 4));
        let plan = c.plan(&p2, &[], true, false);
        assert!(plan.resume > 0);
        assert_eq!(plan.ring_flo, 23_512 - 8960);
        // a cold request reads from 0 again: its prefill writes every row
        let cold = c.plan(&prompt(9000, 8), &[], true, false);
        assert_eq!((cold.resume, cold.ring_flo), (0, 0));
    }

    #[test]
    fn pixel_digests_tell_pictures_and_grids_apart() {
        let a: Vec<u16> = (0..4096u32).map(|i| (i * 7) as u16).collect();
        let mut b = a.clone();
        b[4095] ^= 1;
        assert_eq!(pixel_digest(&a, (2, 2)), pixel_digest(&a, (2, 2)));
        assert_ne!(pixel_digest(&a, (2, 2)), pixel_digest(&b, (2, 2)));
        assert_ne!(pixel_digest(&a, (2, 2)), pixel_digest(&a, (1, 4)));
        assert_ne!(pixel_digest(&a[..4093], (2, 2)), pixel_digest(&a[..4094], (2, 2)));
    }

    #[test]
    fn today_s_tuning_values() {
        // pinned here, not in a law: the guard and the ladder may change with measurements
        assert_eq!(crate::verified::pc_tail_guard(), 8);
        assert_eq!(pc_anchor_tiles(8 * 512 + 100, 512), 8);
        assert_eq!(pc_anchor_tiles(8 * 512 + 9, 512), 8);
        assert_eq!(pc_anchor_tiles(8 * 512 + 8, 512), 7);
        let back: Vec<u64> = (0..5).map(crate::verified::pc_ladder_back).collect();
        assert_eq!(back, vec![0, 1, 3, 7, 15]);
        assert_eq!(DEFAULT_SLOTS, 8);
    }
}
