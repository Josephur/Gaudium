//! Prompt layout and interleaved M-RoPE for image and video prompts.
//!
//! Text tokens count positions up by one. An image-pad run starting at position `s` puts its
//! merged grid at `(t, h, w) = (s, s + row, s + col)`, and the next text token continues at
//! `s + max(rows, cols)`. So after an image the RoPE position of a token is no longer its KV
//! slot: inside the prompt it is `pos3[slot]`, after it `slot + delta`.
//!
//! A video is its temporal patches, each a run of video-pad tokens between its own vision
//! start / end markers with the patch's time written as text in front. Time is carried by that
//! text, not by the `t` position: every temporal patch is laid out exactly like an image.
//!
//! The decoder's RoPE tables stay **indexed by KV slot**: for a vision request the rows from the
//! first image slot on are rewritten with the cos / sin of each slot's M-RoPE position (every
//! frequency taking its own section), so prefill tiles, decode steps and any drafter that reads
//! the table by slot are right without a graph or kernel change. The causal order stays the
//! slot order. Anything that computes rotary angles itself must use `PromptLayout::position`.

use crate::verified::{vis_run_emb_off, vis_run_rows, vis_run_tile_row};
use crate::vision::VisionSpec;

/// One image's run of image-pad tokens in the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    /// First KV slot (prompt index) of the run.
    pub start: usize,
    /// Image-pad tokens = merged grid rows x columns.
    pub len: usize,
    /// Merged grid (rows, columns).
    pub rows: usize,
    pub cols: usize,
    /// Row of this image's first token in the request's embedding buffer.
    pub emb_row: usize,
}

impl ImageSpan {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

/// Positions and image spans of one prompt. This is what speculative methods need for vision:
/// the per-row position table, the span list (lookup must not match inside a span and drafters
/// take vision embeddings there) and, with the tower, the embedding buffer address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptLayout {
    /// `(t, h, w)` M-RoPE position of every prompt slot.
    pub pos3: Vec<[i32; 3]>,
    pub spans: Vec<ImageSpan>,
    /// `position(slot) = slot + delta` for every slot at or after the prompt end (<= 0).
    pub delta: i64,
}

impl PromptLayout {
    pub fn prompt_len(&self) -> usize {
        self.pos3.len()
    }

    pub fn image_tokens(&self) -> usize {
        self.spans.iter().map(|s| s.len).sum()
    }

    /// First slot whose position can differ from a text-only prompt's (the first image slot).
    pub fn first_image_slot(&self) -> Option<usize> {
        self.spans.first().map(|s| s.start)
    }

    /// M-RoPE position of `slot` in section `sec` (0 = t, 1 = h, 2 = w), for any slot.
    pub fn position(&self, slot: usize, sec: usize) -> i64 {
        match self.pos3.get(slot) {
            Some(p) => i64::from(p[sec]),
            None => slot as i64 + self.delta,
        }
    }

    /// The span holding `slot`, if it is an image-pad slot.
    pub fn span_at(&self, slot: usize) -> Option<&ImageSpan> {
        self.spans
            .iter()
            .find(|s| slot >= s.start && slot < s.end())
    }

    /// Overlaps of the image spans with the slot window `[lo, hi)`:
    /// `(embedding row, first slot, slot count)` per overlap, in prompt order.
    pub fn overlaps(
        &self,
        lo: usize,
        hi: usize,
    ) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.overlaps_shifted(lo, hi, 0)
    }

    /// Like `overlaps`, for a tile whose row `r` embeds the token at slot `r + shift` (a
    /// next-token head such as the MTP fusion: `shift` 1). Each overlap is `(embedding row,
    /// first tile row as a prompt row, rows)`: the rows `[first, first + rows)` take the
    /// embeddings `[row, row + rows)`. Rows whose token would lie before slot 0 do not exist,
    /// so a span at slot 0 loses its first `shift` rows. Where each run lands is the verified
    /// `vis_run_*` arithmetic (`verified/vision_budget.bend`: inside the tile, inside the image).
    pub fn overlaps_shifted(
        &self,
        lo: usize,
        hi: usize,
        shift: usize,
    ) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let (lo64, hi64, sh) = (lo as u64, hi as u64, shift as u64);
        self.spans.iter().filter_map(move |s| {
            let (start, len) = (s.start as u64, s.len as u64);
            let rows = vis_run_rows(start, len, lo64, hi64, sh);
            (rows > 0).then(|| {
                let row = vis_run_tile_row(start, len, lo64, hi64, sh);
                let off = vis_run_emb_off(start, len, lo64, hi64, sh);
                (s.emb_row + off as usize, lo + row as usize, rows as usize)
            })
        })
    }

    /// Token history for prompt-lookup drafting: every image-pad slot holds `sentinel` (a value
    /// that is never a token id), so no n-gram can match inside an image span and a proposed
    /// continuation ends at the first sentinel instead of drafting image pads.
    pub fn lookup_history(&self, prompt_ids: &[i32], sentinel: i32) -> Vec<i32> {
        let mut h = prompt_ids.to_vec();
        for s in &self.spans {
            let end = s.end().min(h.len());
            for t in &mut h[s.start.min(end)..end] {
                *t = sentinel;
            }
        }
        h
    }
}

/// Expands every single image-pad placeholder of `ids` to its image's token count. A run that
/// already has the right length is kept (prompts tokenised by the reference processor).
pub fn expand_image_pads(
    ids: &[i32],
    image_tokens: &[usize],
    image_token_id: i32,
) -> Result<Vec<i32>, String> {
    expand_vision_pads(ids, image_tokens, image_token_id, -1)
}

/// Like `expand_image_pads` for prompts that also hold video: `tokens[k]` is the token count
/// of the `k`-th pad run in prompt order, a run being image pads (an image) or video pads (one
/// temporal patch of a video). `video_token_id` < 0: the model has no video pad.
pub fn expand_vision_pads(
    ids: &[i32],
    image_tokens: &[usize],
    image_token_id: i32,
    video_token_id: i32,
) -> Result<Vec<i32>, String> {
    let is_pad = |t: i32| t == image_token_id || (video_token_id >= 0 && t == video_token_id);
    let mut out = Vec::with_capacity(ids.len() + image_tokens.iter().sum::<usize>());
    let mut k = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if !is_pad(ids[i]) {
            out.push(ids[i]);
            i += 1;
            continue;
        }
        let pad = ids[i];
        let mut j = i;
        while j < ids.len() && ids[j] == pad {
            j += 1;
        }
        let run = j - i;
        // images never adjoin (vision end / start tokens sit between them): a run is one
        // image, either its single placeholder or its already expanded pads
        let Some(&n) = image_tokens.get(k) else {
            return Err(format!(
                "the prompt has more image placeholders than images ({})",
                image_tokens.len()
            ));
        };
        if run != 1 && run != n {
            return Err(format!(
                "image {k}: a run of {run} image-pad tokens is neither one placeholder nor its {n} tokens"
            ));
        }
        out.extend(std::iter::repeat_n(pad, n));
        k += 1;
        i = j;
    }
    if k != image_tokens.len() {
        return Err(format!(
            "{} images but the prompt has {k} image placeholders",
            image_tokens.len()
        ));
    }
    Ok(out)
}

/// Builds the layout of an (expanded) prompt. `grids` are the patch grids `(rows, columns)` of
/// the prompt's pad runs in prompt order (an image, or one temporal patch of a video); every
/// run must match its grid exactly.
pub fn layout_prompt(
    ids: &[i32],
    grids: &[(usize, usize)],
    vs: &VisionSpec,
) -> Result<PromptLayout, String> {
    let m = vs.merge as usize;
    let mut pos3 = vec![[0i32; 3]; ids.len()];
    let mut spans = Vec::with_capacity(grids.len());
    let mut cur = 0i64;
    let mut max_pos = -1i64;
    let mut emb_row = 0usize;
    let mut i = 0usize;
    let is_pad =
        |t: i32| t == vs.image_token_id || (vs.video_token_id >= 0 && t == vs.video_token_id);
    while i < ids.len() {
        if !is_pad(ids[i]) {
            pos3[i] = [cur as i32; 3];
            max_pos = max_pos.max(cur);
            cur += 1;
            i += 1;
            continue;
        }
        let k = spans.len();
        let Some(&(gh, gw)) = grids.get(k) else {
            return Err(format!("image-pad run {k} has no image"));
        };
        if gh == 0 || gw == 0 || gh % m != 0 || gw % m != 0 {
            return Err(format!(
                "image {k}: patch grid {gh}x{gw} is not a multiple of {m}"
            ));
        }
        let (rows, cols) = (gh / m, gw / m);
        let n = rows * cols;
        let pad = ids[i];
        let mut j = i;
        while j < ids.len() && ids[j] == pad {
            j += 1;
        }
        if j - i != n {
            return Err(format!(
                "image {k}: {} pad tokens, its {rows}x{cols} grid gives {n}",
                j - i
            ));
        }
        for q in 0..n {
            let p = [cur, cur + (q / cols) as i64, cur + (q % cols) as i64];
            pos3[i + q] = [p[0] as i32, p[1] as i32, p[2] as i32];
            max_pos = max_pos.max(p[1]).max(p[2]);
        }
        spans.push(ImageSpan {
            start: i,
            len: n,
            rows,
            cols,
            emb_row,
        });
        emb_row += n;
        cur += rows.max(cols) as i64;
        i = j;
    }
    if spans.len() != grids.len() {
        return Err(format!(
            "{} images but {} image-pad runs",
            grids.len(),
            spans.len()
        ));
    }
    if cur > i64::from(i32::MAX) {
        return Err("prompt positions overflow".into());
    }
    Ok(PromptLayout {
        pos3,
        spans,
        delta: max_pos + 1 - ids.len() as i64,
    })
}

/// Section (0 = t, 1 = h, 2 = w) of rotary frequency `i` under interleaved M-RoPE.
#[inline]
pub fn mrope_section_of(i: usize, sections: [u64; 3]) -> usize {
    if i % 3 == 1 && (i as u64) < 3 * sections[1] {
        1
    } else if i % 3 == 2 && (i as u64) < 3 * sections[2] {
        2
    } else {
        0
    }
}

/// Decoder RoPE table rows `[lo, hi)` (`{128, rows}` F32 each for cos and sin, the layout of the
/// attention front end's tables) for per-slot positions `pos(slot, section)`. With all three
/// sections equal to the slot number the rows are bit-identical to the text table.
pub fn mrope_rows(
    theta: f64,
    rotary_dim: usize,
    sections: [u64; 3],
    lo: usize,
    hi: usize,
    pos: impl Fn(usize, usize) -> i64,
) -> (Vec<u8>, Vec<u8>) {
    let rot = rotary_dim;
    let np = rot / 4;
    let n = hi.saturating_sub(lo);
    let mut c = vec![1.0f32; 128 * n];
    let mut s = vec![0.0f32; 128 * n];
    // per-frequency constants (even and odd index of each lane pair)
    let freqs: Vec<(f64, f64, usize, usize)> = (0..np)
        .map(|j| {
            let fe = theta.powf(-2.0 * (2 * j) as f64 / rot as f64);
            let fo = theta.powf(-2.0 * (2 * j + 1) as f64 / rot as f64);
            (
                fe,
                fo,
                mrope_section_of(2 * j, sections),
                mrope_section_of(2 * j + 1, sections),
            )
        })
        .collect();
    for slot in lo..hi {
        let base = (slot - lo) * 128;
        let p3 = [
            pos(slot, 0) as f64,
            pos(slot, 1) as f64,
            pos(slot, 2) as f64,
        ];
        for (j, &(fe, fo, se, so)) in freqs.iter().enumerate() {
            let (pe, po) = (p3[se], p3[so]);
            let ce = (pe * fe).cos() as f32;
            let sne = (pe * fe).sin() as f32;
            let co = (po * fo).cos() as f32;
            let sno = (po * fo).sin() as f32;

            c[base + j] = ce;
            c[base + np + j] = ce;
            s[base + j] = -sne;
            s[base + np + j] = sne;

            c[base + 64 + j] = co;
            c[base + 64 + np + j] = co;
            s[base + 64 + j] = -sno;
            s[base + 64 + np + j] = sno;
        }
    }
    (f32s_to_bytes(&c), f32s_to_bytes(&s))
}

pub(crate) fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::packer::WeightPacker;
    use crate::model_spec::ModelSpec;
    use crate::safetensors::SafetensorsStore;
    use crate::vision::test_util::{qwen38_vision_spec, QWEN38_27B_CONFIG};

    const PAD: i32 = 248_056;

    fn two_image_prompt() -> (Vec<i32>, Vec<(usize, usize)>) {
        // 3 text, start, 64 pads (16x16 patches), end, start, 140 pads (20x28 patches), end, text
        let mut ids = vec![248_045, 846, 198, 248_053];
        ids.extend(std::iter::repeat_n(PAD, 64));
        ids.extend([248_054, 248_053]);
        ids.extend(std::iter::repeat_n(PAD, 140));
        ids.extend([248_054, 760, 1118]);
        (ids, vec![(16, 16), (20, 28)])
    }

    #[test]
    fn layout_matches_reference_positions_of_a_two_image_prompt() {
        let vs = qwen38_vision_spec();
        let (ids, grids) = two_image_prompt();
        let l = layout_prompt(&ids, &grids, &vs).unwrap();
        assert_eq!(l.spans.len(), 2);
        assert_eq!(
            (l.spans[0].start, l.spans[0].len, l.spans[0].emb_row),
            (4, 64, 0)
        );
        assert_eq!(
            (l.spans[1].start, l.spans[1].len, l.spans[1].emb_row),
            (70, 140, 64)
        );
        assert_eq!((l.spans[1].rows, l.spans[1].cols), (10, 14));
        // text 0..3, image 1 at 4: t = 4, h = 4 + row, w = 4 + col
        assert_eq!(l.pos3[3], [3, 3, 3]);
        assert_eq!(l.pos3[4], [4, 4, 4]);
        assert_eq!(l.pos3[4 + 9], [4, 5, 5]); // token 9: row 1, col 1
        assert_eq!(l.pos3[4 + 63], [4, 11, 11]);
        // next text position = 4 + max(8, 8) = 12
        assert_eq!(l.pos3[68], [12, 12, 12]);
        assert_eq!(l.pos3[69], [13, 13, 13]);
        // image 2 at 14: 10 rows x 14 cols
        assert_eq!(l.pos3[70], [14, 14, 14]);
        assert_eq!(l.pos3[70 + 139], [14, 23, 27]);
        // then 14 + 14 = 28
        assert_eq!(l.pos3[210], [28, 28, 28]);
        assert_eq!(l.pos3[212], [30, 30, 30]);
        // delta: max position 30 + 1 - 213 slots
        assert_eq!(l.delta, 31 - 213);
        assert_eq!(l.position(213, 0), 31);
        assert_eq!(l.position(300, 2), 300 - 182);
        assert_eq!(l.image_tokens(), 204);
        assert_eq!(l.first_image_slot(), Some(4));
        assert!(l.span_at(3).is_none() && l.span_at(68).is_none());
        assert_eq!(l.span_at(70).map(|s| s.emb_row), Some(64));
    }

    #[test]
    fn overlaps_cover_tiles_that_straddle_spans() {
        let vs = qwen38_vision_spec();
        let (ids, grids) = two_image_prompt();
        let l = layout_prompt(&ids, &grids, &vs).unwrap();
        // one tile over everything
        assert_eq!(
            l.overlaps(0, 256).collect::<Vec<_>>(),
            vec![(0, 4, 64), (64, 70, 140)]
        );
        // tiles of 100: second image straddles the boundary
        assert_eq!(
            l.overlaps(0, 100).collect::<Vec<_>>(),
            vec![(0, 4, 64), (64, 70, 30)]
        );
        assert_eq!(
            l.overlaps(100, 200).collect::<Vec<_>>(),
            vec![(94, 100, 100)]
        );
        assert_eq!(
            l.overlaps(200, 213).collect::<Vec<_>>(),
            vec![(194, 200, 10)]
        );
        assert_eq!(l.overlaps(210, 213).count(), 0);
    }

    #[test]
    fn shifted_overlaps_and_lookup_history_follow_the_drafter_rules() {
        let vs = qwen38_vision_spec();
        let (ids, grids) = two_image_prompt();
        let l = layout_prompt(&ids, &grids, &vs).unwrap();
        // a next-token head's row r embeds token r + 1: spans move one slot down
        assert_eq!(
            l.overlaps_shifted(0, 256, 1).collect::<Vec<_>>(),
            vec![(0, 3, 64), (64, 69, 140)]
        );
        assert_eq!(
            l.overlaps_shifted(0, 4, 1).collect::<Vec<_>>(),
            vec![(0, 3, 1)]
        );
        // a span at slot 0 shifted by one loses its first row
        let l0 = layout_prompt(&[PAD; 64], &[(16, 16)], &vs).unwrap();
        assert_eq!(
            l0.overlaps_shifted(0, 64, 1).collect::<Vec<_>>(),
            vec![(1, 0, 63)]
        );

        let h = l.lookup_history(&ids, -1);
        assert_eq!(h.len(), ids.len());
        assert_eq!(h.iter().filter(|&&t| t == -1).count(), 204);
        assert!(!h.contains(&PAD));
        assert_eq!((h[3], h[4], h[67], h[68]), (248_053, -1, -1, 248_054));
    }

    #[test]
    fn image_runs_take_exactly_the_tiles_image_tokens() {
        // every small span x tile x shift: a run's tile row and embedding row name the same
        // prompt slot, and the run covers every slot both the tile's tokens and the image hold
        use crate::verified::{vis_run_emb_off, vis_run_rows, vis_run_tile_row};
        for start in 0..9u64 {
            for len in 0..9u64 {
                for lo in 0..9u64 {
                    for hi in lo..12u64 {
                        for shift in 0..3u64 {
                            let n = vis_run_rows(start, len, lo, hi, shift);
                            let row = vis_run_tile_row(start, len, lo, hi, shift);
                            let off = vis_run_emb_off(start, len, lo, hi, shift);
                            let want: Vec<u64> = (lo + shift..hi + shift)
                                .filter(|t| *t >= start && *t < start + len)
                                .collect();
                            assert_eq!(n as usize, want.len(), "{start} {len} {lo} {hi} {shift}");
                            if n > 0 {
                                assert_eq!(lo + shift + row, start + off);
                                assert_eq!(want[0], start + off);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn layout_refuses_mismatched_runs() {
        let vs = qwen38_vision_spec();
        let (ids, grids) = two_image_prompt();
        assert!(layout_prompt(&ids, &grids[..1], &vs).is_err());
        assert!(layout_prompt(&ids, &[(16, 16), (20, 30)], &vs).is_err());
        assert!(layout_prompt(&ids, &[(16, 16), (20, 28), (16, 16)], &vs).is_err());
        // a video pad without a grid is refused like an image pad without one
        assert!(layout_prompt(&[1, 248_057, 2], &[], &vs).is_err());
        // text-only prompt: positions = slots, delta 0
        let l = layout_prompt(&[5, 6, 7], &[], &vs).unwrap();
        assert_eq!(l.delta, 0);
        assert_eq!(l.pos3[2], [2, 2, 2]);
        assert!(l.first_image_slot().is_none());
    }

    /// A prompt with one video of four temporal patches (7 frames of 96x64 at 2 fps), tokenised
    /// by the reference processor, and the positions the reference model computes for it.
    #[test]
    fn video_layout_matches_reference_positions() {
        let vs = qwen38_vision_spec();
        let vp = 248_057;
        let mut ids = vec![248_045, 846, 198, 10_071, 15_294];
        // "<0.2 seconds>", "<1.2 seconds>", "<2.2 seconds>", "<3.0 seconds>"
        for (a, b) in [(15, 17), (16, 17), (17, 17), (18, 15)] {
            ids.extend([a, 13, b, 6283, 29, 248_053]);
            ids.extend(std::iter::repeat_n(vp, 6));
            ids.extend([248_054, 27]);
        }
        ids.pop();
        ids.extend([
            3710, 8311, 30, 248_046, 198, 248_045, 74_455, 198, 248_068, 271, 248_069, 271,
        ]);
        assert_eq!(ids.len(), 72);
        let l = layout_prompt(&ids, &[(4, 6); 4], &vs).unwrap();
        let want_t: [i32; 72] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 22, 22, 22, 22, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33, 33, 33, 33, 33, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 44, 44, 44, 44, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59,
        ];
        let want_h: [i32; 72] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 12, 12, 12, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 22, 22, 23, 23, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33, 33, 34, 34, 34, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 44, 44, 45, 45, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59,
        ];
        let want_w: [i32; 72] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59,
        ];
        for (slot, p) in l.pos3.iter().enumerate() {
            assert_eq!(
                *p,
                [want_t[slot], want_h[slot], want_w[slot]],
                "slot {slot}"
            );
        }
        assert_eq!(l.delta, -12);
        assert_eq!(l.spans.len(), 4);
        assert_eq!(
            (l.spans[3].start, l.spans[3].len, l.spans[3].emb_row),
            (53, 6, 18)
        );
        // single placeholders expand to the same prompt
        let mut collapsed: Vec<i32> = Vec::new();
        for &t in &ids {
            if t != vp || collapsed.last() != Some(&t) {
                collapsed.push(t);
            }
        }
        assert_eq!(
            expand_vision_pads(&collapsed, &[6; 4], vs.image_token_id, vp).unwrap(),
            ids
        );
        // without a video pad id the tokens are plain text to the layout
        assert!(expand_image_pads(&collapsed, &[6; 4], vs.image_token_id).is_err());
    }

    #[test]
    fn expand_image_pads_handles_placeholders_and_expanded_runs() {
        // single placeholders, adjacent images
        let ids = [1, 248_053, PAD, 248_054, 248_053, PAD, 248_054, 9];
        let out = expand_image_pads(&ids, &[3, 2], PAD).unwrap();
        assert_eq!(
            out,
            vec![1, 248_053, PAD, PAD, PAD, 248_054, 248_053, PAD, PAD, 248_054, 9]
        );
        // already expanded prompt is kept
        assert_eq!(expand_image_pads(&out, &[3, 2], PAD).unwrap(), out);
        assert!(expand_image_pads(&ids, &[3], PAD).is_err());
        assert!(expand_image_pads(&ids, &[3, 2, 4], PAD).is_err());
        assert_eq!(expand_image_pads(&[1, 2], &[], PAD).unwrap(), vec![1, 2]);
    }

    #[test]
    fn frequency_sections_interleave_t_h_w() {
        let s = [11, 11, 10];
        let secs: Vec<usize> = (0..32).map(|i| mrope_section_of(i, s)).collect();
        for (i, &sec) in secs.iter().enumerate() {
            assert_eq!(sec, i % 3, "frequency {i}");
        }
        // a shorter w section falls back to t past its end
        assert_eq!(mrope_section_of(29, [12, 11, 9]), 0);
    }

    #[test]
    fn text_rows_are_bit_identical_to_the_decoder_table() {
        let spec = ModelSpec::from_config_json(QWEN38_27B_CONFIG).unwrap();
        let store = SafetensorsStore::empty();
        let (rc, rs) = WeightPacker::new(&store, &spec).build_rope_tables(300);
        let (c, s) = mrope_rows(
            spec.rope_theta,
            spec.rotary_dim as usize,
            [11, 11, 10],
            0,
            300,
            |slot, _| slot as i64,
        );
        assert_eq!(c, rc);
        assert_eq!(s, rs);
        // a shifted window with a delta equals the table rows at slot + delta
        let (c2, s2) = mrope_rows(
            spec.rope_theta,
            spec.rotary_dim as usize,
            [11, 11, 10],
            250,
            290,
            |slot, _| slot as i64 - 182,
        );
        assert_eq!(c2[..], rc[(250 - 182) * 512..(290 - 182) * 512]);
        assert_eq!(s2[..], rs[(250 - 182) * 512..(290 - 182) * 512]);
    }

    #[test]
    fn image_rows_take_each_frequency_from_its_own_section() {
        let spec = ModelSpec::from_config_json(QWEN38_27B_CONFIG).unwrap();
        let (theta, rot) = (spec.rope_theta, spec.rotary_dim as usize);
        // one slot at (t, h, w) = (4, 9, 17)
        let (c, _s) = mrope_rows(theta, rot, [11, 11, 10], 0, 1, |_, sec| [4, 9, 17][sec]);
        let f = |i: usize| -> f32 {
            let v = &c[i * 4..i * 4 + 4];
            f32::from_le_bytes([v[0], v[1], v[2], v[3]])
        };
        let np = rot / 4;
        let freq = |i: usize| theta.powf(-2.0 * i as f64 / rot as f64);
        // lane j holds frequency 2j (section 2j % 3), lane 64 + j frequency 2j + 1
        for j in 0..np {
            let pe = [4.0, 9.0, 17.0][(2 * j) % 3];
            let po = [4.0, 9.0, 17.0][(2 * j + 1) % 3];
            assert_eq!(f(j), ((pe * freq(2 * j)).cos()) as f32);
            assert_eq!(f(np + j), f(j));
            assert_eq!(f(64 + j), ((po * freq(2 * j + 1)).cos()) as f32);
        }
        // lanes outside the rotary part stay cos = 1
        assert_eq!(f(2 * np), 1.0);
        assert_eq!(f(127), 1.0);
    }
}
