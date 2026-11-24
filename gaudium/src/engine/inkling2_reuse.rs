//! Prefix reuse of the Inkling 2 engine: a request whose prompt starts like the rows the card
//! already holds resumes its prefill behind them.
//!
//! What the card holds, by kind of state:
//! - **Global layers' caches** keep every position: any prefix of the rows written is there.
//! - **Sliding layers' K / V rings and the convolutions' history rings** keep the latest rows
//!   only. A pass that wrote up to position `hi` has overwritten the rows more than a ring behind
//!   `hi`, so a resume at `r` is exact only while `hi - r` is within the slack the rings were
//!   sized with (`ikr_slack`). A long generation uses that slack up; therefore the rings are
//!   copied once per request, when the prompt is in (one device-to-device copy of the ring
//!   arena), and a later request may resume that copy instead of the live rings.
//!
//! The decisions (which state, which position) are the verified `ikr_*` rules; this file keeps
//! the identities of the rows on the card and does the copies. A resumed prefill gives the same
//! rows as a prefill from position 0: every row is computed from the same inputs, and the
//! engine's rows do not depend on the shape of the pass that computes them.

use std::hash::{Hash, Hasher};

use super::Inkling2Engine;
use crate::engine::qwen3::prefix_cache::PrefixStats;
use crate::verified::{ikr_cap, ikr_pick, ikr_slack, ikr_slack_with, ikr_start, IkResume};

/// What the card holds for the next request, and the copy of the rings taken at the end of
/// the last prompt.
#[derive(Debug, Default)]
pub(super) struct IkReuse {
    enabled: bool,
    slack: u64,
    /// What a copy of the rings is made of: `(rings, bytes, their copy)` (the main model's
    /// ring arena, and each context ring of a drafter).
    copies: Vec<(u64, u64, u64)>,
    /// Position from which the rows of the request's own tokens do not count as held (a part
    /// of the state was not written for them); `None`: all of them count.
    held_cap: Option<usize>,
    /// Identity of the rows at positions `0..keys.len()` of the caches.
    keys: Vec<u64>,
    /// End (exclusive) of the highest position written since the last prefill from 0, rows
    /// that were only computed past the cursor (tile padding, refused drafts) included.
    hi: u64,
    snap_keys: Vec<u64>,
    snap_hi: u64,
    snap_valid: bool,
    /// The identities of the request being served.
    pending: Vec<u64>,
    stats: PrefixStats,
}

impl IkReuse {
    /// `ring` / `hist`: slots of a sliding ring / rows of a history ring; `window` / `taps`:
    /// the rows a pass reads from before its first row.
    pub(super) fn new(enabled: bool, ring: u64, window: u64, hist: u64, taps: u64) -> Self {
        Self {
            enabled,
            slack: ikr_slack(ring, window, hist, taps),
            stats: PrefixStats {
                slots: enabled as u64,
                ..PrefixStats::default()
            },
            ..Self::default()
        }
    }

    /// A part of the bounded state and its copy (both `bytes` long): the main model's ring
    /// arena, a drafter's context ring.
    pub(super) fn bind(&mut self, rings: u64, bytes: u64, copy: u64) {
        self.copies.push((rings, bytes, copy));
    }

    /// A drafter keeps context rings of `ring` slots for a window of `window` positions.
    pub(super) fn with_drafter(&mut self, ring: u64, window: u64) {
        self.slack = ikr_slack_with(self.slack, ring, window);
    }

    /// The rows of this request's tokens from position `pos` on are not held (the engine
    /// wrote only a part of its state for them).
    pub(super) fn hold_before(&mut self, pos: usize) {
        self.held_cap = Some(self.held_cap.map_or(pos, |c| c.min(pos)));
    }

    /// A pass is about to write the rows up to position `end` (exclusive).
    pub(super) fn will_write(&mut self, end: u64) {
        self.hi = self.hi.max(end);
    }

    /// The request's rows up to position `end` (exclusive) are in the caches.
    pub(super) fn rows_in(&mut self, end: usize) {
        let from = self.keys.len();
        if end > from && end <= self.pending.len() {
            self.keys.extend_from_slice(&self.pending[from..end]);
        }
    }

    pub(super) fn stats(&self) -> Option<PrefixStats> {
        self.enabled.then_some(self.stats)
    }
}

/// Identity of a text row: its token.
fn text_key(id: i32) -> u64 {
    id as u32 as u64
}

/// Identity of row `k` of an image: the image's content and the row's place in it (the top bit
/// keeps it apart from every text row).
fn image_key(fingerprint: u64, k: u64) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (fingerprint, k).hash(&mut h);
    h.finish() | (1 << 63)
}

/// Identities of a prompt's rows. `images`: `(fingerprint, rows)` of the request's images in
/// prompt order, whose rows are the prompt's `image_token` rows in order (none: every row is
/// a text row).
pub(super) fn row_keys(prompt_ids: &[i32], image_token: i32, images: &[(u64, u64)]) -> Vec<u64> {
    let mut img = images.iter().copied();
    let mut cur = img.next();
    let mut k = 0u64;
    prompt_ids
        .iter()
        .map(|&id| {
            if id != image_token {
                return text_key(id);
            }
            while let Some((_, rows)) = cur {
                if k < rows {
                    break;
                }
                cur = img.next();
                k = 0;
            }
            match cur {
                Some((fp, _)) => {
                    k += 1;
                    image_key(fp, k - 1)
                }
                None => text_key(id),
            }
        })
        .collect()
}

fn common(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl Inkling2Engine {
    /// Counters of prefix reuse since the engine was loaded (`None`: it is off).
    pub fn prefix_stats(&self) -> Option<PrefixStats> {
        self.reuse.stats()
    }

    /// The position this request's prefill starts at: 0, or behind the rows it takes over from
    /// the card (the live rings, or the copy taken at the end of the last prompt, which is
    /// given back first). `wanted` false (a request with `cache_prompt: false`): no row counts
    /// as held, the whole prompt is prefilled. Returns `(start, rows equal to the rows held)`.
    pub(super) fn reuse_begin(
        &mut self,
        prompt_ids: &[i32],
        wanted: bool,
    ) -> Result<(usize, usize), String> {
        if !self.reuse.enabled {
            return Ok((0, 0));
        }
        let (image_token, images) = self.ikv_request_keys();
        let keys = row_keys(prompt_ids, image_token, &images);
        let rows = keys.len() as u64;
        let ru = &mut self.reuse;
        let lcp = if wanted { common(&keys, &ru.keys) } else { 0 };
        let r_live = ikr_cap(lcp as u64, rows);
        let r_snap = if ru.snap_valid && wanted {
            ikr_cap(common(&keys, &ru.snap_keys) as u64, rows)
        } else {
            0
        };
        let pick = ikr_pick(r_live, ru.hi, r_snap, ru.snap_hi, ru.slack);
        let start = ikr_start(pick, r_live, r_snap) as usize;
        ru.pending = keys;
        ru.held_cap = None;
        ru.stats.requests += 1;
        ru.stats.last_prompt_rows = rows;
        ru.stats.last_lcp = lcp as u64;
        ru.stats.last_reused_rows = start as u64;
        if start > 0 {
            ru.stats.hits += 1;
            ru.stats.reused_rows += start as u64;
        }
        let t_back = std::time::Instant::now();
        match pick {
            IkResume::IrFresh => {
                ru.keys.clear();
                ru.hi = 0;
            }
            IkResume::IrLive => ru.keys.truncate(start),
            IkResume::IrSnap => {
                // the rings as they were when the last prompt was in
                let (copies, hi) = (ru.copies.clone(), ru.snap_hi);
                // the live rings are neither state until the copy is known to be in
                ru.keys.clear();
                for (rings, bytes, copy) in copies {
                    self.dev
                        .copy_d2d_compute(copy, bytes, rings)
                        .map_err(|e| format!("give the ring copy back: {e}"))?;
                }
                self.sync_compute("ring copy back")?;
                let ru = &mut self.reuse;
                ru.hi = hi;
                ru.keys.extend_from_slice(&ru.pending[..start]);
            }
        }
        if self.verbose {
            eprintln!(
                "[gaudium] inkling2 prefix: {lcp} of {rows} prompt rows equal the rows held; prefill starts at {start} ({}, {:.2} ms)",
                match pick {
                    IkResume::IrFresh => "nothing taken over",
                    IkResume::IrLive => "live rings",
                    IkResume::IrSnap => "ring copy of the last prompt given back",
                },
                t_back.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok((start, lcp))
    }

    /// The prompt is in: copies the rings, so that a later request can resume here however long
    /// this one generates.
    pub(super) fn reuse_prompt_in(&mut self) -> Result<(), String> {
        let ru = &mut self.reuse;
        if !ru.enabled {
            return Ok(());
        }
        ru.snap_valid = false;
        for (rings, bytes, copy) in ru.copies.clone() {
            self.dev
                .copy_d2d_compute(rings, bytes, copy)
                .map_err(|e| format!("copy the rings: {e}"))?;
        }
        self.sync_compute("ring copy")?;
        let ru = &mut self.reuse;
        ru.snap_keys.clone_from(&ru.keys);
        ru.snap_hi = ru.hi;
        ru.snap_valid = true;
        Ok(())
    }

    /// The request ended after emitting `generated`: every token but the last was fed, so its
    /// row is in the caches behind the prompt's.
    pub(super) fn reuse_request_done(&mut self, prompt_rows: usize, generated: &[i32]) {
        let ru = &mut self.reuse;
        if !ru.enabled || ru.keys.len() != prompt_rows {
            return;
        }
        let mut fed = generated.len().saturating_sub(1);
        if let Some(cap) = ru.held_cap {
            fed = fed.min(cap.saturating_sub(prompt_rows));
        }
        ru.keys
            .extend(generated[..fed].iter().map(|&t| text_key(t)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_rows_are_their_tokens_and_image_rows_their_image() {
        let img = 9;
        let a = row_keys(&[1, 9, 9, 2, 9, 3], img, &[(77, 2), (88, 1)]);
        let b = row_keys(&[1, 9, 9, 2, 9, 3], img, &[(77, 2), (99, 1)]);
        assert_eq!(a[0], 1);
        assert_eq!(a[3], 2);
        assert!(a[1] >> 63 == 1 && a[2] >> 63 == 1 && a[1] != a[2]);
        // the same first image: the rows up to the second image are equal
        assert_eq!(common(&a, &b), 4);
        // without images a placeholder row is a text row
        assert_eq!(row_keys(&[1, 9, 2], img, &[]), vec![1, 9, 2]);
        // more placeholder rows than image rows: the rest are text rows
        assert_eq!(row_keys(&[9, 9], img, &[(5, 1)])[1], 9);
    }

    #[test]
    fn rows_enter_in_order_and_only_from_the_request() {
        let mut ru = IkReuse::new(true, 2048, 1024, 1024, 4);
        assert_eq!(ru.slack, 1020);
        ru.pending = vec![1, 2, 3, 4];
        ru.rows_in(2);
        ru.rows_in(2);
        ru.rows_in(9);
        assert_eq!(ru.keys, vec![1, 2]);
        ru.rows_in(4);
        assert_eq!(ru.keys, vec![1, 2, 3, 4]);
        ru.will_write(8);
        ru.will_write(5);
        assert_eq!(ru.hi, 8);
    }

    // Today's choices of the `ikr_*` rules that their laws deliberately leave open (the laws
    // only bound them): the order of preference, the slack formula, what a one-row prompt takes.

    #[test]
    fn live_is_taken_before_the_snapshot() {
        // both states hold what a resume at their rows reads
        let p = ikr_pick(100, 120, 100, 100, 508);
        assert_eq!(p, IkResume::IrLive);
        assert_eq!(ikr_start(p, 100, 100), 100);
        // the snapshot gives more rows, live is still taken
        let p = ikr_pick(50, 60, 80, 80, 508);
        assert_eq!(p, IkResume::IrLive);
        assert_eq!(ikr_start(p, 50, 80), 50);
    }

    #[test]
    fn the_snapshot_is_taken_when_live_is_not_usable() {
        // a long generation wrote the live rings past the slack
        let p = ikr_pick(100, 100 + 509, 100, 100, 508);
        assert_eq!(p, IkResume::IrSnap);
        assert_eq!(ikr_start(p, 100, 100), 100);
        // the snapshot's rows equal the prompt only as far as the rows on the card do
        let p = ikr_pick(60, 1000, 90, 90, 508);
        assert_eq!(p, IkResume::IrSnap);
        assert_eq!(ikr_start(p, 60, 90), 60);
        // neither state holds what a resume reads
        assert_eq!(ikr_pick(100, 1000, 100, 1000, 508), IkResume::IrFresh);
    }

    #[test]
    fn slack_is_the_smaller_spare_of_the_two_rings() {
        assert_eq!(ikr_slack(2048, 1024, 512, 4), 508);
        assert_eq!(ikr_slack(2048, 1024, 1024, 4), 1020);
        assert_eq!(ikr_slack(1024, 1024, 512, 4), 0);
    }

    #[test]
    fn a_prompt_of_one_row_takes_over_nothing() {
        assert_eq!(ikr_cap(1, 1), 0);
        assert_eq!(ikr_cap(5, 1), 0);
        assert_eq!(ikr_pick(ikr_cap(1, 1), 1, 0, 0, 508), IkResume::IrFresh);
        // an empty prompt takes over nothing either
        assert_eq!(ikr_cap(3, 0), 0);
    }
}
