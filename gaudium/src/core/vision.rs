//! Rust shell wrappers for vision input admission, budgets and frame arithmetic.
//!
//! Every decision here delegates to `verified/vision_budget.bend`: which media sources a request
//! may use, the byte / canvas limits checked before decoding, the pixel budgets under the
//! planned vision tower, the embedding-row accounting of one request, video frame counts and
//! indices, and mirrored indices of an orientation change.

use crate::model_spec::ModelFamily;
use crate::verified::{
    media_bytes_ok, media_canvas_ok, media_count_ok, media_source_allowed, orient_mirror,
    vid_frame_index, vid_group_frame, vid_groups, vid_keep_count, vid_pixel_budget,
    vid_rows_pixel_budget, vis_drafter_ok, vis_grid_fits, vis_grid_tokens, vis_pixel_cap,
    vis_rows_fit, vis_rows_left, vis_rows_take, vis_token_cap, MediaSource, VisDrafter,
};

/// Limits on the media of one request, checked before anything is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaLimits {
    /// Largest encoded file (an image, one frame, or an animation).
    pub max_bytes: u64,
    /// Largest canvas (width x height) that is decoded.
    pub max_pixels: u64,
    /// Most media items (images + videos) of one request.
    pub max_items: u64,
    /// Media the server would have to fetch (anything but inline bytes) is taken.
    pub allow_remote: bool,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            max_bytes: 64 << 20,
            max_pixels: 64_000_000,
            max_items: 256,
            allow_remote: false,
        }
    }
}

impl MediaLimits {
    /// Inline bytes (`remote` false) are always taken; anything else only when allowed.
    pub fn source_allowed(&self, remote: bool) -> bool {
        let src = if remote {
            MediaSource::MsRemote
        } else {
            MediaSource::MsInline
        };
        media_source_allowed(src, self.allow_remote)
    }

    pub fn check_bytes(&self, bytes: usize) -> Result<(), String> {
        if media_bytes_ok(bytes as u64, self.max_bytes) {
            Ok(())
        } else if bytes == 0 {
            Err("the file is empty".into())
        } else {
            Err(format!(
                "the file has {bytes} bytes; the limit is {} bytes",
                self.max_bytes
            ))
        }
    }

    pub fn check_count(&self, count: usize) -> Result<(), String> {
        if media_count_ok(count as u64, self.max_items) {
            Ok(())
        } else {
            Err(format!(
                "the request holds {count} images and videos; the limit is {}",
                self.max_items
            ))
        }
    }
}

/// A canvas read from a file header against the decode limit.
pub fn check_canvas(width: usize, height: usize, max_pixels: u64) -> Result<(), String> {
    if media_canvas_ok(width as u64, height as u64, max_pixels) {
        Ok(())
    } else if width == 0 || height == 0 {
        Err("image has zero size".into())
    } else {
        Err(format!(
            "image of {width}x{height} pixels is larger than the limit of {max_pixels} pixels"
        ))
    }
}

/// Whether a model of `family` decoded with the drafter `kind` (`None`: plain decode; else the
/// server's drafter kinds `dspine`, `dflash2`, `mtp`) may be loaded with its vision tower. The
/// kind and the family name the drafter (a Qwen model's `mtp` is its own MTP head, a Gemma 4
/// model's is its `-assistant` checkpoint, Inkling 2's is its block drafter over the target's
/// hidden states); the verified `vis_drafter_ok` decides. A pairing not listed here is
/// [`VisDrafter::VdOther`], which never runs with images.
pub fn drafter_runs_with_vision(kind: Option<&str>, family: ModelFamily) -> bool {
    use ModelFamily as F;
    vis_drafter_ok(match (kind, family) {
        (None, _) => VisDrafter::VdNone,
        (Some("dspine"), _) => VisDrafter::VdDSpine,
        (Some("dflash2"), F::Qwen35Dense) => VisDrafter::VdBlock,
        (Some("mtp"), F::Qwen35Dense) => VisDrafter::VdMtp,
        (Some("mtp"), F::Qwen35Moe) => VisDrafter::VdMtpMoe,
        (Some("mtp"), F::Inkling2) => VisDrafter::VdStates,
        (Some("mtp"), F::Gemma4) => VisDrafter::VdAssistant,
        _ => VisDrafter::VdOther,
    })
}

/// Pixel cap of one image: the model's own, lowered to the largest image the tower holds.
pub fn image_pixel_cap(model_max_pixels: u64, tower_patches: u64, patch_side: u64) -> u64 {
    vis_pixel_cap(
        model_max_pixels,
        tower_patches,
        patch_side.saturating_mul(patch_side),
    )
}

/// Pixel budget of a whole video of `frames` kept frames whose single frames may take
/// `frame_cap` pixels at most.
pub fn video_pixel_budget(model_max_pixels: u64, frames: u64, frame_cap: u64) -> u64 {
    vid_pixel_budget(model_max_pixels, frames, frame_cap)
}

/// Pixel budget of a whole video that keeps its tokens inside `rows_left` embedding rows:
/// `kept` frames in `groups` temporal patches, one token per `cell_side x cell_side` pixels.
pub fn video_rows_pixel_budget(rows_left: u64, kept: usize, groups: usize, cell_side: u64) -> u64 {
    vid_rows_pixel_budget(
        rows_left,
        kept as u64,
        groups as u64,
        cell_side.saturating_mul(cell_side),
    )
}

/// A resized patch grid against the tower's largest image.
pub fn grid_fits(grid_h: usize, grid_w: usize, tower_patches: u64) -> bool {
    vis_grid_fits(grid_h as u64, grid_w as u64, tower_patches)
}

/// Decoder tokens of a patch grid under an `merge x merge` merger.
pub fn grid_tokens(grid_h: usize, grid_w: usize, merge: u64) -> u64 {
    vis_grid_tokens(grid_h as u64, grid_w as u64, merge)
}

/// Image tokens the server plans per request: the configured count, never above the context.
pub fn planned_image_tokens(configured: u64, ctx_tokens: u64) -> u64 {
    vis_token_cap(configured, ctx_tokens)
}

/// Embedding rows of one request against the planned buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbRows {
    used: u64,
    cap: u64,
}

impl EmbRows {
    pub fn new(cap: u64) -> Self {
        Self { used: 0, cap }
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn left(&self) -> u64 {
        vis_rows_left(self.used, self.cap)
    }

    /// Takes `add` rows; `Err((needed, cap))` when they do not fit (nothing is taken).
    pub fn take(&mut self, add: u64) -> Result<(), (u64, u64)> {
        if !vis_rows_fit(self.used, add, self.cap) {
            return Err((self.used.saturating_add(add), self.cap));
        }
        self.used = vis_rows_take(self.used, add, self.cap);
        Ok(())
    }
}

/// Frames kept from a video of `total` frames when its duration asks for `by_rate`.
pub fn video_keep_count(by_rate: u64, min_frames: u64, max_frames: u64, total: usize) -> usize {
    vid_keep_count(by_rate, min_frames, max_frames, total as u64) as usize
}

/// A sampled frame index held inside a video of `total` frames.
pub fn video_frame_index(raw: u64, total: usize) -> usize {
    vid_frame_index(raw, total as u64) as usize
}

/// Temporal patches of `frames` frames, `temporal` frames each.
pub fn video_groups(frames: usize, temporal: u64) -> usize {
    vid_groups(frames as u64, temporal) as usize
}

/// The frame in place `slot` of temporal patch `group` (the last frame again past the end).
pub fn video_group_frame(group: usize, slot: usize, temporal: u64, frames: usize) -> usize {
    vid_group_frame(group as u64, slot as u64, temporal, frames as u64) as usize
}

/// Index `i` of an axis of `n` entries, mirrored.
pub fn mirror(i: usize, n: usize) -> usize {
    orient_mirror(i as u64, n as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_refuse_empty_oversized_and_remote() {
        let l = MediaLimits {
            max_bytes: 10,
            max_pixels: 100,
            max_items: 2,
            allow_remote: false,
        };
        assert!(l.check_bytes(0).is_err());
        assert!(l.check_bytes(10).is_ok());
        assert!(l.check_bytes(11).unwrap_err().contains("limit is 10"));
        assert!(l.check_count(2).is_ok());
        assert!(l.check_count(3).is_err());
        assert!(l.source_allowed(false));
        assert!(!l.source_allowed(true));
        let open = MediaLimits {
            allow_remote: true,
            ..l
        };
        assert!(open.source_allowed(true));
        assert!(check_canvas(10, 10, 100).is_ok());
        assert!(check_canvas(10, 11, 100).is_err());
        assert!(check_canvas(0, 5, 100).is_err());
        // a canvas whose area overflows 64 bits is refused, not wrapped
        assert!(check_canvas(usize::MAX, usize::MAX, u64::MAX - 1).is_err());
    }

    #[test]
    fn canvas_check_matches_exact_area() {
        // the division form agrees with the exact (128-bit) area test, overflowing sizes included
        let sides = [0u64, 1, 2, 3, 7, 100, 4096, 65_535, 1 << 31, u64::MAX / 2, u64::MAX];
        let limits = [0u64, 1, 99, 100, 101, 1 << 24, 1 << 62, u64::MAX - 1, u64::MAX];
        for &w in &sides {
            for &h in &sides {
                for &m in &limits {
                    let exact = w > 0 && h > 0 && (w as u128) * (h as u128) <= m as u128;
                    assert_eq!(media_canvas_ok(w, h, m), exact, "w={w} h={h} m={m}");
                }
            }
        }
    }

    #[test]
    fn only_validated_drafters_run_with_the_tower() {
        use ModelFamily as F;
        // today's validated set (the vision suite passed on the card): plain decode, DSpine, a
        // dense model's DFlash2 drafter and own MTP head, Inkling 2's own drafter
        for f in [F::Qwen3, F::Qwen35Dense, F::Qwen35Moe, F::Gemma4, F::Inkling2] {
            assert!(drafter_runs_with_vision(None, f));
            assert!(drafter_runs_with_vision(Some("dspine"), f));
            assert!(!drafter_runs_with_vision(Some("something new"), f));
        }
        assert!(drafter_runs_with_vision(Some("dflash2"), F::Qwen35Dense));
        assert!(drafter_runs_with_vision(Some("mtp"), F::Qwen35Dense));
        assert!(drafter_runs_with_vision(Some("mtp"), F::Inkling2));
        // not yet through the suite: the mixture-of-experts MTP head, Gemma 4's drafters
        assert!(!drafter_runs_with_vision(Some("mtp"), F::Qwen35Moe));
        assert!(!drafter_runs_with_vision(Some("mtp"), F::Gemma4));
        assert!(!drafter_runs_with_vision(Some("dflash2"), F::Gemma4));
        assert!(!drafter_runs_with_vision(Some("dflash2"), F::Qwen35Moe));
        assert!(!drafter_runs_with_vision(Some("dflash2"), F::Inkling2));
    }

    #[test]
    fn pixel_budgets_follow_the_tower() {
        // the tower holds 4096 patches of 16x16: 1,048,576 pixels
        assert_eq!(image_pixel_cap(16_777_216, 4096, 16), 1_048_576);
        assert_eq!(image_pixel_cap(500_000, 4096, 16), 500_000);
        assert_eq!(video_pixel_budget(25_165_824, 8, 1_048_576), 8_388_608);
        assert_eq!(video_pixel_budget(25_165_824, 64, 1_048_576), 25_165_824);
        // 100 rows for 7 frames in 4 temporal patches of 32x32-pixel tokens
        assert_eq!(video_rows_pixel_budget(100, 7, 4, 32), 100 * 1024 * 7 / 4);
        assert_eq!(video_rows_pixel_budget(100, 7, 0, 32), 0);
        assert!(grid_fits(64, 64, 4096));
        assert!(!grid_fits(64, 66, 4096));
        assert_eq!(grid_tokens(16, 28, 2), 112);
        assert_eq!(grid_tokens(16, 28, 0), 0);
        assert_eq!(planned_image_tokens(16_384, 4_000), 4_000);
    }

    #[test]
    fn embedding_rows_never_leave_the_buffer() {
        let mut r = EmbRows::new(100);
        assert!(r.take(60).is_ok());
        assert_eq!(r.take(41), Err((101, 100)));
        assert_eq!((r.used(), r.left()), (60, 40));
        assert!(r.take(40).is_ok());
        assert_eq!(r.left(), 0);
        assert!(r.take(1).is_err());
        assert!(r.take(0).is_ok());
    }

    #[test]
    fn video_frames_stay_in_range() {
        // today's model settings: at least 4 frames, at most 768
        assert_eq!(video_keep_count(1, 4, 768, 7), 4);
        assert_eq!(video_keep_count(1, 4, 768, 3), 3);
        assert_eq!(video_keep_count(5000, 4, 768, 4000), 768);
        assert_eq!(video_keep_count(7, 4, 768, 7), 7);
        assert_eq!(video_frame_index(9, 7), 6);
        assert_eq!(video_frame_index(3, 7), 3);
        assert_eq!(video_groups(7, 2), 4);
        assert_eq!(video_groups(8, 2), 4);
        assert_eq!(video_groups(1, 2), 1);
        assert_eq!(video_groups(5, 0), 0);
        // the last temporal patch of 7 frames shows frame 6 twice
        assert_eq!(video_group_frame(3, 0, 2, 7), 6);
        assert_eq!(video_group_frame(3, 1, 2, 7), 6);
        assert_eq!(video_group_frame(1, 1, 2, 7), 3);
        assert_eq!((mirror(0, 5), mirror(4, 5), mirror(9, 5)), (4, 0, 0));
    }
}
