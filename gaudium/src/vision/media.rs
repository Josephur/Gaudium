//! The media of one request: images, animations and videos from what the API carries to what
//! the vision tower takes.
//!
//! - An **image** is one file. It is decoded (`vision::decode`), turned by its EXIF
//!   orientation, and prepared by the image processor (`preprocess::prepare_image`).
//! - A **video** is its frames: an animated image file (GIF, WebP) or a list of image files
//!   with the frames' rate or times. The server decodes no video container. The frames the
//!   model is shown are sampled the way the checkpoint's video processor samples them (its
//!   frames per second, between its fewest and most frames), resized under one pixel budget for
//!   the whole video, and grouped into temporal patches. In the prompt each temporal patch is a
//!   pad run of its own with its time written in front (`<1.5 seconds>`), so a video is a
//!   sequence of tower inputs that are laid out like images.
//!
//! Decisions (limits, budgets, frame counts and indices) come from the verified core
//! (`core::vision`). What stays here is byte and text handling, and the float formulas of the
//! reference processors, which the tests pin to reference results.

use std::time::Instant;

use crate::core::vision::{
    grid_fits, grid_tokens, image_pixel_cap, video_frame_index, video_group_frame, video_groups,
    video_keep_count, video_pixel_budget, video_rows_pixel_budget, EmbRows, MediaLimits,
};
use crate::vision::decode::{self, DecodeLimits};
use crate::vision::mrope::expand_vision_pads;
use crate::vision::preprocess::{
    image_target_size, prepare_image, prepare_video, resize_rgb, smart_resize_video, PreparedImage,
};
use crate::vision::{VideoSpec, VisionSpec};

/// The template's block for one image / one video; the pad is expanded later.
pub const IMAGE_BLOCK: &str = "<|vision_start|><|image_pad|><|vision_end|>";
pub const VIDEO_BLOCK: &str = "<|vision_start|><|video_pad|><|vision_end|>";

/// Frame rate assumed for frames that come with neither a rate nor times (the reference video
/// processor's own assumption).
pub const DEFAULT_FRAME_RATE: f64 = 24.0;

/// One media part of a request as the API gave it. The strings are `data:` URLs.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaRef {
    Image(String),
    /// A video as one animated image file.
    Animation(String),
    /// A video as its frames (image files), with their rate (frames per second of video time)
    /// or the time of each frame in seconds.
    Frames {
        frames: Vec<String>,
        fps: Option<f64>,
        times: Option<Vec<f64>>,
    },
}

impl MediaRef {
    pub fn is_video(&self) -> bool {
        !matches!(self, Self::Image(_))
    }
}

/// What a served model's vision tower was planned for, and the request limits.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionServe {
    /// The model's vision settings with `max_pixels` already lowered to the tower's largest image.
    pub spec: VisionSpec,
    /// Largest single tower input (an image, or one temporal patch of a video), in patches.
    pub max_patches: u64,
    /// Embedding rows (image tokens) one request may fill.
    pub max_tokens: u64,
    pub limits: MediaLimits,
}

impl VisionServe {
    /// Lowers the model's image pixel cap to a tower of `max_patches`. Refuses a tower too small
    /// for the smallest image the model takes.
    pub fn new(
        mut spec: VisionSpec,
        max_patches: u64,
        max_tokens: u64,
        limits: MediaLimits,
    ) -> Result<Self, String> {
        let cap = image_pixel_cap(spec.max_pixels, max_patches, spec.patch);
        if cap < spec.min_pixels {
            return Err(format!(
                "a vision tower of {max_patches} patches is below the smallest image this model takes ({} patches)",
                spec.min_pixels.div_ceil(spec.patch * spec.patch)
            ));
        }
        spec.max_pixels = cap;
        Ok(Self {
            spec,
            max_patches,
            max_tokens,
            limits,
        })
    }

    fn decode_limits(&self) -> DecodeLimits {
        DecodeLimits {
            max_pixels: self.limits.max_pixels,
        }
    }
}

/// Picture sizes `(width, height)` that screens, cameras and phones produce. The tower compiles
/// its recipes per patch count; `VisionServe::common_patch_counts` turns these into the counts
/// worth compiling ahead (`--build-cache`), so the first image of a usual shape waits for no
/// compile. Any other size still works: it compiles on first use.
const COMMON_SIZES: &[(usize, usize)] = &[
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
    (1280, 720),
    (1366, 768),
    (1440, 900),
    (1536, 864),
    (1600, 900),
    (1280, 800),
    (1680, 1050),
    (1920, 1200),
    (2560, 1600),
    (2880, 1800),
    (3024, 1964),
    (3456, 2234),
    (1024, 768),
    (800, 600),
    (640, 480),
    (2048, 1536),
    (4032, 3024),
    (4000, 3000),
    (6000, 4000),
    (3000, 2000),
    (5472, 3648),
    (1024, 1024),
    (768, 768),
    (512, 512),
    (256, 256),
    (1080, 1080),
    (1080, 1350),
    (1200, 630),
    (1200, 675),
    (1170, 2532),
    (1179, 2556),
    (1290, 2796),
    (1080, 2400),
    (1080, 2340),
    (1440, 3120),
    (720, 1280),
    (640, 360),
    (854, 480),
    (320, 240),
];

impl VisionServe {
    /// Patch counts of the common picture sizes (both orientations) under this tower, sorted.
    pub fn common_patch_counts(&self) -> Vec<u64> {
        let mut out: Vec<u64> = Vec::new();
        for &(w, h) in COMMON_SIZES {
            for (w, h) in [(w, h), (h, w)] {
                if let Ok((rh, rw)) = image_target_size(w, h, &self.spec) {
                    let p = self.spec.patch as usize;
                    let (gh, gw) = (rh / p, rw / p);
                    if grid_fits(gh, gw, self.max_patches) {
                        out.push((gh * gw) as u64);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Decodes base64 (standard or URL-safe alphabet, padding optional, ASCII whitespace skipped).
pub fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3 + 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    let mut ended = false;
    for &c in text.as_bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => {
                ended = true;
                continue;
            }
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        if ended {
            return None;
        }
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    // a lone trailing sextet carries no byte
    (bits < 6).then_some(out)
}

/// The bytes of a media reference. Only inline data is taken (`data:<type>;base64,<payload>`):
/// the server fetches nothing unless the limits allow it, and has no fetcher.
pub fn inline_bytes(url: &str, limits: &MediaLimits) -> Result<Vec<u8>, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        if !limits.source_allowed(true) {
            return Err(
                "only data: URLs are accepted (base64 in the request; remote URLs are not fetched)"
                    .into(),
            );
        }
        return Err("fetching remote media is not available in this build".into());
    };
    if !limits.source_allowed(false) {
        return Err("inline media is not accepted".into());
    }
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err("malformed data URL (no comma)".into());
    };
    if !meta.split(';').any(|p| p.eq_ignore_ascii_case("base64")) {
        return Err("the data URL is not base64".into());
    }
    // the size is known from the text: refuse before decoding
    limits.check_bytes(payload.len() / 4 * 3)?;
    let bytes = base64_decode(payload).ok_or("the data URL's payload is not valid base64")?;
    limits.check_bytes(bytes.len())?;
    Ok(bytes)
}

/// A video ready for the tower: its temporal patches in order and each one's time.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedVideo {
    pub patches: Vec<PreparedImage>,
    /// Seconds of video time of each temporal patch (the mean of its first and last frame).
    pub times: Vec<f64>,
    /// Frames the source held and frames the model is shown.
    pub frames_total: usize,
    pub frames_kept: usize,
}

/// One prepared media part.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaItem {
    Image(PreparedImage),
    Video(PreparedVideo),
}

/// Sizes of a request's media, for the readout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaCounts {
    pub images: usize,
    pub videos: usize,
    /// Video frames the model is shown.
    pub frames: usize,
    /// Tower inputs (images + temporal patches).
    pub inputs: usize,
    pub patches: u64,
    pub tokens: u64,
}

/// The prepared media of one request, in prompt order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestMedia {
    pub items: Vec<MediaItem>,
    /// The template's text for one image (`VisionSpec::image_block`; empty: `IMAGE_BLOCK`).
    pub image_block: String,
    /// Host time spent decoding and preparing (ms).
    pub host_ms: f64,
}

/// The frames of a video the model is shown, as indices into its `total` frames: the reference
/// video processor's sampling. The duration asks for `fps` frames per second; the count is held
/// between the model's fewest and most frames and to the frames there are; the kept frames are
/// spread evenly from the first to the last one.
pub fn sample_frames(total: usize, src_fps: f64, v: &VideoSpec) -> Result<Vec<usize>, String> {
    if total == 0 {
        return Err("a video needs at least one frame".into());
    }
    // truncation towards zero, as the reference computes the count
    let by_rate = (total as f64 / src_fps * v.fps) as u64;
    let n = video_keep_count(by_rate, v.min_frames, v.max_frames, total);
    if n == 0 {
        return Err("the model's video settings keep no frame of this video".into());
    }
    let stop = (total - 1) as f64;
    let step = if n > 1 { stop / (n - 1) as f64 } else { 0.0 };
    Ok((0..n)
        .map(|i| {
            // evenly spaced, the last one exactly the last frame; ties round to even
            let y = if n > 1 && i == n - 1 {
                stop
            } else {
                i as f64 * step
            };
            video_frame_index(y.round_ties_even() as u64, total)
        })
        .collect())
}

/// Time of every temporal patch: the mean of the times of its first and last frame.
fn patch_times(kept_times: &[f64], temporal: u64) -> Vec<f64> {
    let n = kept_times.len();
    (0..video_groups(n, temporal))
        .map(|g| {
            let a = kept_times[video_group_frame(g, 0, temporal, n)];
            let last_place = temporal.saturating_sub(1) as usize;
            let b = kept_times[video_group_frame(g, last_place, temporal, n)];
            (a + b) / 2.0
        })
        .collect()
}

/// The time label the prompt carries in front of a temporal patch.
pub fn time_label(seconds: f64) -> String {
    format!("<{seconds:.1} seconds>")
}

fn check_tower_fit(img: &PreparedImage, serve: &VisionServe) -> Result<(), String> {
    if grid_fits(img.grid_h, img.grid_w, serve.max_patches) {
        Ok(())
    } else {
        Err(format!(
            "{}x{} pixels give {} patches; the vision tower was planned for {}",
            img.width,
            img.height,
            img.n_patches(),
            serve.max_patches
        ))
    }
}

fn prepare_one_image(
    bytes: &[u8],
    serve: &VisionServe,
    with_pixels: bool,
) -> Result<PreparedImage, String> {
    let img = if with_pixels {
        let rgb = decode::decode_image(bytes, &serve.decode_limits())?;
        prepare_image(&rgb, &serve.spec)?
    } else {
        let p = decode::probe(bytes)?;
        serve.decode_limits().check_canvas(p.width, p.height)?;
        let (rh, rw) = image_target_size(p.width, p.height, &serve.spec)?;
        PreparedImage::sized((p.width, p.height), rw, rh, &serve.spec)
    };
    check_tower_fit(&img, serve)?;
    Ok(img)
}

/// Where a video's frames come from, after the request's references are resolved.
enum VideoSource<'a> {
    Animation(Vec<u8>),
    Frames {
        frames: &'a [String],
        fps: Option<f64>,
        times: Option<&'a [f64]>,
    },
}

fn prepare_one_video(
    src: VideoSource,
    serve: &VisionServe,
    rows_left: u64,
    with_pixels: bool,
) -> Result<PreparedVideo, String> {
    let vs = &serve.spec;
    let Some(vsp) = vs.video else {
        return Err("this model takes no video input".into());
    };
    let limits = serve.decode_limits();

    // 1. the time of every source frame and the source's frame rate
    let (total, times, src_fps): (usize, Vec<f64>, f64) = match &src {
        VideoSource::Animation(bytes) => {
            let scan = decode::scan_frames(bytes)?;
            // times from whole milliseconds, so each one is the nearest f64 of its exact value
            // (a running f64 sum drifts, and a drifted mean can round to another time label)
            let mut t = Vec::with_capacity(scan.delays_ms.len());
            let mut at_ms = 0u64;
            for &d in &scan.delays_ms {
                t.push(at_ms as f64 / 1000.0);
                at_ms = at_ms.saturating_add(u64::from(d));
            }
            let n = t.len();
            let fps = if at_ms > 0 {
                n as f64 * 1000.0 / at_ms as f64
            } else {
                DEFAULT_FRAME_RATE
            };
            (n, t, fps)
        }
        VideoSource::Frames { frames, fps, times } => {
            let n = frames.len();
            match (times, fps) {
                (Some(ts), _) => {
                    if ts.len() != n {
                        return Err(format!(
                            "{} timestamps for {n} frames (one per frame is needed)",
                            ts.len()
                        ));
                    }
                    if ts.iter().any(|t| !t.is_finite() || *t < 0.0)
                        || ts.windows(2).any(|w| w[1] < w[0])
                    {
                        return Err(
                            "timestamps must be seconds, not negative and not decreasing".into(),
                        );
                    }
                    // the rate the times imply (n frames span n - 1 gaps)
                    let span =
                        ts.last().copied().unwrap_or(0.0) - ts.first().copied().unwrap_or(0.0);
                    let fps = if n > 1 && span > 0.0 {
                        (n - 1) as f64 / span
                    } else {
                        DEFAULT_FRAME_RATE
                    };
                    (n, ts.to_vec(), fps)
                }
                (None, Some(f)) => {
                    if !(f.is_finite() && *f > 0.0) {
                        return Err("fps must be a positive number".into());
                    }
                    (n, (0..n).map(|i| i as f64 / f).collect(), *f)
                }
                (None, None) => (
                    n,
                    (0..n).map(|i| i as f64 / DEFAULT_FRAME_RATE).collect(),
                    DEFAULT_FRAME_RATE,
                ),
            }
        }
    };

    // 2. the frames the model is shown
    let kept = sample_frames(total, src_fps, &vsp)?;
    let n = kept.len();
    let groups = video_groups(n, vs.temporal);
    let kept_times: Vec<f64> = kept.iter().map(|&k| times[k]).collect();

    // 3. one size for all frames, under the model's budget, the tower and the rows left
    let frame_bytes = |k: usize| -> Result<Vec<u8>, String> {
        match &src {
            VideoSource::Frames { frames, .. } => {
                inline_bytes(&frames[k], &serve.limits).map_err(|e| format!("frame {}: {e}", k + 1))
            }
            VideoSource::Animation(_) => unreachable!("an animation is one file"),
        }
    };
    let (src_w, src_h) = match &src {
        VideoSource::Animation(bytes) => {
            let scan = decode::scan_frames(bytes)?;
            (scan.width, scan.height)
        }
        VideoSource::Frames { .. } => {
            let p = decode::probe(&frame_bytes(kept[0])?)
                .map_err(|e| format!("frame {}: {e}", kept[0] + 1))?;
            (p.width, p.height)
        }
    };
    limits.check_canvas(src_w, src_h)?;
    if (groups as u64) > rows_left {
        return Err(format!(
            "the video's {groups} temporal patches need at least {groups} image tokens; {rows_left} are left in this request"
        ));
    }
    let frame_cap = image_pixel_cap(u64::MAX, serve.max_patches, vs.patch);
    let budget = video_pixel_budget(vsp.max_pixels, n as u64, frame_cap).min(
        video_rows_pixel_budget(rows_left, n, groups, vs.resize_factor()),
    );
    let (rh, rw) = smart_resize_video(
        n,
        src_h,
        src_w,
        vs.temporal as usize,
        vs.resize_factor() as usize,
        vsp.min_pixels.min(budget),
        budget,
    )?;
    let sized = PreparedImage::sized((src_w, src_h), rw, rh, vs);
    check_tower_fit(&sized, serve)?;

    // 4. the pixels: every kept frame is resized as soon as it is decoded
    let patches = if with_pixels {
        let mut resized: Vec<Vec<u8>> = Vec::with_capacity(n);
        match &src {
            VideoSource::Animation(bytes) => {
                decode::decode_frames(bytes, &limits, &kept, &mut |w, h, rgb| {
                    resized.push(resize_rgb(rgb, w, h, rw, rh));
                    Ok(())
                })?;
            }
            VideoSource::Frames { .. } => {
                for &k in &kept {
                    let img = decode::decode_image(&frame_bytes(k)?, &limits)
                        .map_err(|e| format!("frame {}: {e}", k + 1))?;
                    if (img.width, img.height) != (src_w, src_h) {
                        return Err(format!(
                            "frame {} is {}x{} pixels, frame {} {src_w}x{src_h}: the frames of a video must have one size",
                            k + 1,
                            img.width,
                            img.height,
                            kept[0] + 1
                        ));
                    }
                    resized.push(resize_rgb(&img.data, img.width, img.height, rw, rh));
                }
            }
        }
        prepare_video(
            &resized,
            rw,
            rh,
            (src_w, src_h),
            groups,
            |g, place| video_group_frame(g, place, vs.temporal, n),
            vs,
        )?
    } else {
        vec![sized; groups]
    };
    Ok(PreparedVideo {
        patches,
        times: patch_times(&kept_times, vs.temporal),
        frames_total: total,
        frames_kept: n,
    })
}

impl RequestMedia {
    /// Decodes and prepares the media of one request, in order. `with_pixels` false computes
    /// sizes only (token counts without decoding pixels). Limits are checked before decoding;
    /// the tokens of all items together must fit the planned embedding rows.
    pub fn prepare(
        refs: &[MediaRef],
        serve: &VisionServe,
        with_pixels: bool,
    ) -> Result<Self, String> {
        let t0 = Instant::now();
        serve.limits.check_count(refs.len())?;
        let mut rows = EmbRows::new(serve.max_tokens);
        let mut items = Vec::with_capacity(refs.len());
        let (mut n_img, mut n_vid) = (0usize, 0usize);
        for r in refs {
            let take = |rows: &mut EmbRows, tokens: u64| {
                rows.take(tokens).map_err(|(need, cap)| {
                    format!(
                        "the request's images and videos need {need} image tokens; this server takes {cap} per request"
                    )
                })
            };
            match r {
                MediaRef::Image(url) => {
                    n_img += 1;
                    let named = |e: String| format!("image {n_img}: {e}");
                    let bytes = inline_bytes(url, &serve.limits).map_err(named)?;
                    let img = prepare_one_image(&bytes, serve, with_pixels).map_err(named)?;
                    take(
                        &mut rows,
                        grid_tokens(img.grid_h, img.grid_w, serve.spec.merge),
                    )
                    .map_err(named)?;
                    items.push(MediaItem::Image(img));
                }
                MediaRef::Animation(_) | MediaRef::Frames { .. } => {
                    n_vid += 1;
                    let named = |e: String| format!("video {n_vid}: {e}");
                    let src = match r {
                        MediaRef::Animation(url) => {
                            VideoSource::Animation(inline_bytes(url, &serve.limits).map_err(named)?)
                        }
                        MediaRef::Frames { frames, fps, times } => VideoSource::Frames {
                            frames,
                            fps: *fps,
                            times: times.as_deref(),
                        },
                        MediaRef::Image(_) => unreachable!(),
                    };
                    let v =
                        prepare_one_video(src, serve, rows.left(), with_pixels).map_err(named)?;
                    let tokens: u64 = v
                        .patches
                        .iter()
                        .map(|p| grid_tokens(p.grid_h, p.grid_w, serve.spec.merge))
                        .sum();
                    take(&mut rows, tokens).map_err(named)?;
                    items.push(MediaItem::Video(v));
                }
            }
        }
        Ok(Self {
            items,
            image_block: serve.spec.image_block.clone(),
            host_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The tower inputs in prompt order: an image, or each temporal patch of a video.
    fn inputs(&self) -> impl Iterator<Item = (&PreparedImage, bool)> + '_ {
        self.items
            .iter()
            .flat_map(|it| -> Box<dyn Iterator<Item = (&PreparedImage, bool)>> {
                match it {
                    MediaItem::Image(i) => Box::new(std::iter::once((i, false))),
                    MediaItem::Video(v) => Box::new(v.patches.iter().map(|p| (p, true))),
                }
            })
    }

    pub fn counts(&self) -> MediaCounts {
        let mut c = MediaCounts::default();
        for it in &self.items {
            match it {
                MediaItem::Image(_) => c.images += 1,
                MediaItem::Video(v) => {
                    c.videos += 1;
                    c.frames += v.frames_kept;
                }
            }
        }
        for (p, _) in self.inputs() {
            c.inputs += 1;
            c.patches += p.n_patches() as u64;
            c.tokens += p.n_tokens() as u64;
        }
        c
    }

    /// Writes every video into the rendered prompt: the template's one video block becomes one
    /// block per temporal patch with its time in front. Image blocks stay as they are. The
    /// blocks in the text must be this request's media, in order.
    pub fn expand_text(&self, prompt: &str) -> Result<String, String> {
        let image_block = if self.image_block.is_empty() {
            IMAGE_BLOCK
        } else {
            self.image_block.as_str()
        };
        let mut out = String::with_capacity(prompt.len());
        let mut rest = prompt;
        for it in &self.items {
            let (block, what) = match it {
                MediaItem::Image(_) => (image_block, "image"),
                MediaItem::Video(_) => (VIDEO_BLOCK, "video"),
            };
            // the next block of either kind must be this item's
            let next = [image_block, VIDEO_BLOCK]
                .iter()
                .filter_map(|b| rest.find(b).map(|at| (at, *b)))
                .min();
            let Some((at, found)) = next else {
                return Err(format!(
                    "the conversation has more images and videos than the prompt has placeholders (no place for a {what})"
                ));
            };
            if found != block {
                return Err(
                    "the prompt's image and video placeholders are not in the order of the conversation's images and videos".into(),
                );
            }
            out.push_str(&rest[..at]);
            match it {
                MediaItem::Image(_) => out.push_str(image_block),
                MediaItem::Video(v) => {
                    for &t in &v.times {
                        out.push_str(&time_label(t));
                        out.push_str(VIDEO_BLOCK);
                    }
                }
            }
            rest = &rest[at + block.len()..];
        }
        if rest.contains(image_block) || rest.contains(VIDEO_BLOCK) {
            return Err(
                "the prompt has more image or video placeholders than the conversation has images and videos".into(),
            );
        }
        out.push_str(rest);
        Ok(out)
    }

    /// Expands every pad placeholder of the tokenised prompt to its tower input's tokens. The
    /// pad runs must be this request's inputs in order and in kind (image pad / video pad).
    pub fn expand_ids(&self, ids: &[i32], vs: &VisionSpec) -> Result<Vec<i32>, String> {
        let is_pad =
            |t: i32| t == vs.image_token_id || (vs.video_token_id >= 0 && t == vs.video_token_id);
        // kinds first: the k-th run's pad must be the k-th input's
        let mut k = 0usize;
        let mut prev: Option<i32> = None;
        let kinds: Vec<bool> = self.inputs().map(|(_, video)| video).collect();
        for &t in ids {
            if is_pad(t) && prev != Some(t) {
                let want_video = *kinds.get(k).ok_or_else(|| {
                    format!(
                        "the prompt has more image or video placeholders than the conversation has images and videos ({})",
                        kinds.len()
                    )
                })?;
                if (t == vs.video_token_id) != want_video {
                    return Err(
                        "the prompt's image and video placeholders are not in the order of the conversation's images and videos".into(),
                    );
                }
                k += 1;
            }
            prev = Some(t);
        }
        let tokens: Vec<usize> = self.inputs().map(|(p, _)| p.n_tokens()).collect();
        expand_vision_pads(ids, &tokens, vs.image_token_id, vs.video_token_id)
    }

    /// The inputs handed to the engine, in prompt order.
    pub fn into_tower_inputs(self) -> Vec<PreparedImage> {
        let mut out = Vec::new();
        for it in self.items {
            match it {
                MediaItem::Image(i) => out.push(i),
                MediaItem::Video(v) => out.extend(v.patches),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::test_util::qwen38_vision_spec;

    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub(crate) fn base64(data: &[u8]) -> String {
        let mut s = String::new();
        for c in data.chunks(3) {
            let v = (u32::from(c[0]) << 16)
                | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
                | u32::from(*c.get(2).unwrap_or(&0));
            s.push(B64[(v >> 18) as usize] as char);
            s.push(B64[((v >> 12) & 63) as usize] as char);
            s.push(if c.len() > 1 {
                B64[((v >> 6) & 63) as usize] as char
            } else {
                '='
            });
            s.push(if c.len() > 2 {
                B64[(v & 63) as usize] as char
            } else {
                '='
            });
        }
        s
    }

    /// Deterministic frame `k` of a synthetic video (the generator of the reference dumps).
    pub(crate) fn synthetic_frame(w: usize, h: usize, k: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let n = ((x as u32).wrapping_mul(73_856_093)
                    ^ (y as u32).wrapping_mul(19_349_663)
                    ^ (k as u32).wrapping_mul(83_492_791))
                    >> 7;
                data.push((((x * 255) / w.max(1) + k * 17) % 256) as u8);
                data.push(((y * 255) / h.max(1)) as u8);
                data.push((n & 0xFF) as u8);
            }
        }
        data
    }

    fn ppm_url(w: usize, h: usize, data: &[u8]) -> String {
        let mut f = format!("P6\n{w} {h}\n255\n").into_bytes();
        f.extend_from_slice(data);
        format!("data:image/x-portable-pixmap;base64,{}", base64(&f))
    }

    fn serve(max_patches: u64, max_tokens: u64) -> VisionServe {
        VisionServe::new(
            qwen38_vision_spec(),
            max_patches,
            max_tokens,
            MediaLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn common_sizes_give_patch_counts_inside_the_tower() {
        let sv = serve(4096, 4096);
        let n = sv.common_patch_counts();
        assert!(n.len() > 10 && n.len() < 80, "{}", n.len());
        assert!(n.windows(2).all(|w| w[0] < w[1]));
        assert!(n.iter().all(|&c| c <= 4096 && c % 4 == 0 && c >= 256));
        // 1024 x 1024 is the tower's largest image; 1920 x 1080 comes down to 48 x 84 patches
        assert!(n.contains(&4096) && n.contains(&4032));
        // a small tower holds every size at or under its cap
        assert!(serve(256, 4096)
            .common_patch_counts()
            .iter()
            .all(|&c| c <= 256));
    }

    #[test]
    fn base64_takes_both_alphabets_and_refuses_garbage() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVs\nbG8=\r\n").unwrap(), b"hello");
        assert_eq!(
            base64_decode("-_-_").unwrap(),
            base64_decode("+/+/").unwrap()
        );
        assert_eq!(base64_decode("").unwrap(), b"");
        assert!(base64_decode("a").is_none());
        assert!(base64_decode("aGV$").is_none());
        assert!(base64_decode("aG=Vs").is_none());
        for n in 0..40usize {
            let data: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(base64_decode(&base64(&data)).unwrap(), data);
        }
    }

    #[test]
    fn inline_bytes_takes_data_urls_only() {
        let l = MediaLimits::default();
        assert_eq!(
            inline_bytes("data:image/png;base64,aGVsbG8=", &l).unwrap(),
            b"hello"
        );
        assert_eq!(inline_bytes("data:;BASE64,aGVsbG8=", &l).unwrap(), b"hello");
        for remote in [
            "https://example.com/a.png",
            "http://x/y.jpg",
            "file:///etc/passwd",
            "a.png",
        ] {
            assert!(
                inline_bytes(remote, &l)
                    .unwrap_err()
                    .contains("not fetched"),
                "{remote}"
            );
        }
        assert!(inline_bytes("data:image/png,rawtext", &l)
            .unwrap_err()
            .contains("not base64"));
        assert!(inline_bytes("data:image/png;base64", &l)
            .unwrap_err()
            .contains("no comma"));
        assert!(inline_bytes("data:image/png;base64,@@@@", &l)
            .unwrap_err()
            .contains("valid base64"));
        assert!(inline_bytes("data:image/png;base64,", &l)
            .unwrap_err()
            .contains("empty"));
        let small = MediaLimits { max_bytes: 4, ..l };
        assert!(inline_bytes("data:image/png;base64,aGVsbG8=", &small)
            .unwrap_err()
            .contains("limit is 4"));
    }

    #[test]
    fn frame_sampling_matches_the_reference_processor() {
        let v = qwen38_vision_spec().video.unwrap();
        // (total frames, frames per second) -> indices the reference keeps
        assert_eq!(sample_frames(7, 24.0, &v).unwrap(), vec![0, 2, 4, 6]);
        assert_eq!(
            sample_frames(7, 2.0, &v).unwrap(),
            vec![0, 1, 2, 3, 4, 5, 6]
        );
        assert_eq!(sample_frames(7, 5.0, &v).unwrap(), vec![0, 2, 4, 6]);
        assert_eq!(sample_frames(1, 24.0, &v).unwrap(), vec![0]);
        assert_eq!(sample_frames(3, 30.0, &v).unwrap(), vec![0, 1, 2]);
        assert_eq!(sample_frames(240, 24.0, &v).unwrap(), REF_240_AT_24);
        assert_eq!(sample_frames(100, 10.0, &v).unwrap(), REF_100_AT_10);
        assert_eq!(
            sample_frames(31, 3.0, &v).unwrap(),
            vec![0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 17, 19, 21, 22, 24, 25, 27, 28, 30]
        );
        assert_eq!(
            sample_frames(50, 7.0, &v).unwrap(),
            vec![0, 4, 8, 11, 15, 19, 23, 26, 30, 34, 38, 41, 45, 49]
        );
        assert!(sample_frames(0, 24.0, &v).is_err());
        // more than the model's most frames: exactly that many, first and last kept
        let long = sample_frames(100_000, 30.0, &v).unwrap();
        assert_eq!((long.len(), long[0], long[767]), (768, 0, 99_999));
        assert_eq!(
            (long[1], long[2], long[383], long[766]),
            (130, 261, 49_934, 99_869)
        );
        assert!(long.windows(2).all(|w| w[0] < w[1]));
    }

    /// `np.linspace(0, total - 1, n).round()` of the reference for 240 frames at 24 fps
    /// (10 s -> 20 frames) and 100 frames at 10 fps (10 s -> 20 frames).
    const REF_240_AT_24: [usize; 20] = [
        0, 13, 25, 38, 50, 63, 75, 88, 101, 113, 126, 138, 151, 164, 176, 189, 201, 214, 226, 239,
    ];
    const REF_100_AT_10: [usize; 20] = [
        0, 5, 10, 16, 21, 26, 31, 36, 42, 47, 52, 57, 63, 68, 73, 78, 83, 89, 94, 99,
    ];

    #[test]
    fn time_labels_round_like_the_reference() {
        // (0 + 0.5) / 2 = 0.25 -> "0.2" (ties to even on the exact value), 1.25 -> "1.2"
        assert_eq!(time_label(0.25), "<0.2 seconds>");
        assert_eq!(time_label(1.25), "<1.2 seconds>");
        assert_eq!(time_label(2.25), "<2.2 seconds>");
        assert_eq!(time_label(3.0), "<3.0 seconds>");
        assert_eq!(time_label(0.35), "<0.3 seconds>");
        assert_eq!(time_label(119.75), "<119.8 seconds>");
        assert_eq!(time_label(0.05), "<0.1 seconds>");
        assert_eq!(time_label(0.15), "<0.1 seconds>");
        assert_eq!(time_label(10.45), "<10.4 seconds>");
        // 7 frames at 2 fps: pairs (0, 0.5) (1, 1.5) (2, 2.5) (3, 3)
        let t: Vec<f64> = (0..7).map(|i| i as f64 / 2.0).collect();
        assert_eq!(patch_times(&t, 2), vec![0.25, 1.25, 2.25, 3.0]);
        assert_eq!(patch_times(&[0.0], 2), vec![0.0]);
    }

    #[test]
    fn images_are_prepared_counted_and_limited() {
        let sv = serve(4096, 300);
        let a = ppm_url(256, 256, &synthetic_frame(256, 256, 0));
        let b = ppm_url(448, 320, &synthetic_frame(448, 320, 1));
        let refs = [MediaRef::Image(a.clone()), MediaRef::Image(b.clone())];
        let m = RequestMedia::prepare(&refs, &sv, true).unwrap();
        let c = m.counts();
        assert_eq!(
            (c.images, c.videos, c.inputs, c.patches, c.tokens),
            (2, 0, 2, 816, 204)
        );
        // sizes without pixels agree with the full preparation
        let plan = RequestMedia::prepare(&refs, &sv, false).unwrap();
        assert_eq!(plan.counts(), c);
        match (&plan.items[1], &m.items[1]) {
            (MediaItem::Image(p), MediaItem::Image(f)) => {
                assert!(p.pixels.is_empty());
                assert_eq!((p.grid_h, p.grid_w), (f.grid_h, f.grid_w));
                assert_eq!(f.pixels.len(), f.n_patches() * f.row_len);
            }
            _ => panic!("images expected"),
        }
        // the embedding rows of the request are a hard limit
        let tight = serve(4096, 203);
        let e = RequestMedia::prepare(&refs, &tight, false).unwrap_err();
        assert!(
            e.contains("image 2") && e.contains("204 image tokens") && e.contains("203"),
            "{e}"
        );
        // a tower smaller than the image resizes the image down instead of refusing it
        let small = serve(256, 300);
        let m2 = RequestMedia::prepare(&[MediaRef::Image(b)], &small, false).unwrap();
        assert!(m2.counts().patches <= 256);
        // a tower below the model's smallest image is a configuration error
        assert!(VisionServe::new(qwen38_vision_spec(), 128, 0, MediaLimits::default()).is_err());
        // errors name the item
        let bad = [
            MediaRef::Image(a),
            MediaRef::Image("https://x/y.png".into()),
        ];
        assert!(RequestMedia::prepare(&bad, &sv, false)
            .unwrap_err()
            .starts_with("image 2: "));
        let lim = VisionServe {
            limits: MediaLimits {
                max_items: 1,
                ..MediaLimits::default()
            },
            ..sv
        };
        assert!(RequestMedia::prepare(&bad, &lim, false)
            .unwrap_err()
            .contains("limit is 1"));
    }

    fn frames_ref(w: usize, h: usize, n: usize, fps: Option<f64>) -> MediaRef {
        MediaRef::Frames {
            frames: (0..n)
                .map(|k| ppm_url(w, h, &synthetic_frame(w, h, k)))
                .collect(),
            fps,
            times: None,
        }
    }

    #[test]
    fn a_frame_list_becomes_temporal_patches_with_times() {
        let sv = serve(4096, 4096);
        // the reference processor on the same 7 frames of 96x64 at 2 fps: grid (4, 4, 6)
        let m = RequestMedia::prepare(&[frames_ref(96, 64, 7, Some(2.0))], &sv, true).unwrap();
        let MediaItem::Video(v) = &m.items[0] else {
            panic!("video expected")
        };
        assert_eq!((v.frames_total, v.frames_kept, v.patches.len()), (7, 7, 4));
        assert_eq!(v.times, vec![0.25, 1.25, 2.25, 3.0]);
        assert!(v.patches.iter().all(|p| (p.grid_h, p.grid_w) == (4, 6)));
        let c = m.counts();
        assert_eq!(
            (c.videos, c.frames, c.inputs, c.patches, c.tokens),
            (1, 7, 4, 96, 24)
        );
        // without a rate: 24 frames per second assumed, 4 frames kept, grid (2, 4, 6)
        let m = RequestMedia::prepare(&[frames_ref(96, 64, 7, None)], &sv, false).unwrap();
        let MediaItem::Video(v) = &m.items[0] else {
            panic!("video expected")
        };
        assert_eq!((v.frames_kept, v.patches.len()), (4, 2));
        assert_eq!(
            v.times.iter().map(|&t| time_label(t)).collect::<Vec<_>>(),
            ["<0.0 seconds>", "<0.2 seconds>"]
        );
        // the last temporal patch of an odd frame count shows its last frame twice
        let m = RequestMedia::prepare(&[frames_ref(96, 64, 3, Some(2.0))], &sv, true).unwrap();
        let MediaItem::Video(v) = &m.items[0] else {
            panic!("video expected")
        };
        let last = &v.patches[1];
        let row = &last.pixels[..last.row_len];
        // a row is channel x 2 frames x 16 x 16: both frames of the padded patch are equal
        assert_eq!(row[..256], row[256..512]);
        let first = &v.patches[0];
        assert_ne!(first.pixels[..256], first.pixels[256..512]);
    }

    /// Reference fingerprints from the checkpoint's own video processor, run on the CPU on the
    /// same synthetic frames: `(width, height, frames, frames per second, grid (t, h, w),
    /// FNV-1a of its f32 patch rows)`. Equal fingerprints mean sampling, the shared resize,
    /// temporal grouping and patch order are bit-identical on that video.
    type FramesRef = (usize, usize, usize, Option<f64>, (usize, usize, usize), u64);
    const REFERENCE_VIDEOS: &[FramesRef] = &[
        (96, 64, 7, Some(2.0), (4, 4, 6), 0x4ee5_a01d_b9c1_916d), // odd count: last frame twice
        (96, 64, 7, None, (2, 4, 6), 0x3fe2_1c6b_4a82_2ebe),      // no rate: 4 of 7 frames
        (320, 200, 5, Some(2.0), (3, 12, 20), 0x18ab_9a76_0548_4580), // height up, width kept
        (70, 50, 4, Some(2.0), (2, 4, 4), 0x63b5_ade8_caf3_e34f), // both sides resized
        (1000, 700, 6, Some(2.0), (3, 44, 62), 0xced7_a0cd_a11b_0152),
        (640, 480, 9, Some(4.0), (2, 30, 40), 0x2287_54eb_bd4e_a24c), // 9 frames at 4 fps -> 4
    ];

    #[test]
    fn videos_match_reference_video_processor_fingerprints() {
        use crate::fp8::f32_to_bf16_rne;
        use crate::vision::preprocess::{fnv1a64_f32, video_rows_f32};
        let sv = serve(16_384, 65_536);
        let vs = &sv.spec;
        let vsp = vs.video.unwrap();
        for &(w, h, n, fps, grid, hash) in REFERENCE_VIDEOS {
            let m = RequestMedia::prepare(&[frames_ref(w, h, n, fps)], &sv, true).unwrap();
            let MediaItem::Video(v) = &m.items[0] else {
                panic!("video expected")
            };
            let p0 = &v.patches[0];
            assert_eq!(
                (v.patches.len(), p0.grid_h, p0.grid_w),
                grid,
                "grid of {w}x{h}x{n}"
            );
            // the same frames through the f32 form of the pipeline
            let kept = sample_frames(n, fps.unwrap_or(DEFAULT_FRAME_RATE), &vsp).unwrap();
            let resized: Vec<Vec<u8>> = kept
                .iter()
                .map(|&k| resize_rgb(&synthetic_frame(w, h, k), w, h, p0.width, p0.height))
                .collect();
            let nk = kept.len();
            let rows = video_rows_f32(
                &resized,
                p0.width,
                p0.height,
                grid.0,
                |g, place| video_group_frame(g, place, vs.temporal, nk),
                vs,
            );
            assert_eq!(fnv1a64_f32(&rows), hash, "pixel fingerprint of {w}x{h}x{n}");
            // and what the engine is handed is exactly those rows rounded to BF16
            let got: Vec<u16> = v
                .patches
                .iter()
                .flat_map(|p| p.pixels.iter().copied())
                .collect();
            let want: Vec<u16> = rows.iter().map(|&x| f32_to_bf16_rne(x)).collect();
            assert!(got == want, "BF16 rows of {w}x{h}x{n}");
        }
    }

    macro_rules! fixture {
        ($path:literal) => {
            (
                $path,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/data/vision/",
                    $path
                ))
                .as_slice(),
            )
        };
    }

    /// Image files through the whole host pipeline (the file's decoder, EXIF orientation, smart
    /// resize, patch rows): `(file, patch grid (rows, columns), FNV-1a of the f32 patch rows)`
    /// from the reference loader and the checkpoint's own image processor on the same file
    /// (`tests/data/vision/make_pipeline_refs.py --fingerprints`). Equal fingerprints mean the
    /// tower gets the rows the reference pipeline computes, bit for bit.
    type StillRef = ((&'static str, &'static [u8]), (usize, usize), u64);
    const REFERENCE_STILLS: &[StillRef] = &[
        (
            fixture!("jpeg/base_420_17x13.jpg"),
            (14, 20),
            0x34e7_00c5_5860_82a1,
        ),
        (
            fixture!("jpeg/prog_420_33x49.jpg"),
            (20, 14),
            0x9496_f076_f0ec_1bd1,
        ),
        (
            fixture!("jpeg/base_422_33x49.jpg"),
            (20, 14),
            0x1fc6_c866_7fbf_914d,
        ),
        (
            fixture!("jpeg/base_440_17x13.jpg"),
            (14, 20),
            0xf093_1b3d_3374_6fd9,
        ),
        (
            fixture!("jpeg/cmyk_17x13.jpg"),
            (14, 20),
            0x702d_e858_dc0a_df59,
        ),
        (
            fixture!("jpeg/gray_17x13.jpg"),
            (14, 20),
            0xc5de_2a48_1604_3125,
        ),
        (
            fixture!("jpeg/restart_420_33x49.jpg"),
            (20, 14),
            0x74de_79c8_bd2c_09c5,
        ),
        (fixture!("png/rgb8.png"), (14, 20), 0x2f7a_9f86_7bb0_9945),
        (fixture!("png/rgba8.png"), (14, 20), 0xf2b8_29d9_2f62_4c21),
        (
            fixture!("png/pal8_trns.png"),
            (14, 20),
            0x255a_aa84_1b5b_9095,
        ),
        (fixture!("png/grey16.png"), (14, 20), 0xc74b_4c59_d992_6b99),
        (fixture!("png/rgb16.png"), (14, 20), 0xf1a9_9f1f_324f_eded),
        (
            fixture!("png/rgb8_adam7.png"),
            (14, 20),
            0xeac3_d70c_d761_0055,
        ),
        (fixture!("png/grey1.png"), (14, 20), 0x62bf_7858_d037_c5c9),
        (fixture!("png/exif.png"), (20, 14), 0xa891_722f_bde7_63d5),
        (
            fixture!("png/apng_first_frame.png"),
            (14, 20),
            0xf18f_5b04_d0e2_4c89,
        ),
        (fixture!("bmp/rgb24.bmp"), (12, 22), 0x512d_edca_bec5_dd41),
        (fixture!("bmp/pal8.bmp"), (12, 22), 0xba97_9dca_b92e_ed29),
        (fixture!("bmp/rgb32.bmp"), (12, 22), 0x512d_edca_bec5_dd41),
        (
            fixture!("bmp/rgb565_masks.bmp"),
            (12, 22),
            0x8271_05df_79b1_8e31,
        ),
        (fixture!("bmp/rle8.bmp"), (12, 26), 0x630f_f199_3a08_e08d),
        (
            fixture!("bmp/pal4_top_down.bmp"),
            (12, 22),
            0x373b_f18c_787b_96fd,
        ),
        (
            fixture!("gif/still_256.gif"),
            (14, 20),
            0x09d5_abaa_be62_a169,
        ),
        (
            fixture!("gif/still_interlaced.gif"),
            (14, 20),
            0xcf5c_6f50_6fd0_ee11,
        ),
        (
            fixture!("gif/still_offset_transparent.gif"),
            (14, 20),
            0xab72_0247_82b0_bed1,
        ),
        // an animation sent as an image is its first frame
        (
            fixture!("gif/anim_dispose.gif"),
            (14, 20),
            0x57b4_35a7_0e4f_889d,
        ),
        // pictures stored turned: the EXIF orientation turns them back before the resize
        (
            fixture!("pipeline/exif_o3.jpg"),
            (14, 22),
            0x6ba0_6961_8eaa_db2d,
        ),
        (
            fixture!("pipeline/exif_o6.jpg"),
            (22, 14),
            0xfb0e_9053_3b31_10d1,
        ),
        (
            fixture!("pipeline/exif_o5.png"),
            (22, 14),
            0x6b90_67f5_4ecf_3d9d,
        ),
        (
            fixture!("pipeline/exif_o8.png"),
            (22, 14),
            0x0804_a19a_2d64_a83d,
        ),
        (
            fixture!("pipeline/exif_o7.webp"),
            (22, 14),
            0x872f_d950_f865_8d75,
        ),
        (
            fixture!("webp/lossy_exif_20x12.webp"),
            (22, 14),
            0x2ea2_293d_a7dc_14c1,
        ),
        (
            fixture!("webp/lossy_q75_33x47.webp"),
            (20, 14),
            0x00bf_c2bb_3797_e579,
        ),
        (
            fixture!("webp/lossy_simple_filter_64x48.webp"),
            (14, 20),
            0x462e_a560_4c11_ff8d,
        ),
        (
            fixture!("webp/lossy_filter_deltas_48x40.webp"),
            (16, 18),
            0xc357_d93b_fcf3_fdb5,
        ),
        (
            fixture!("webp/ll_photo_40x30.webp"),
            (14, 20),
            0x9250_c142_ff9d_5ded,
        ),
        (
            fixture!("webp/ll_pal16_33x9.webp"),
            (10, 32),
            0xc029_1bb4_f915_9f75,
        ),
        (
            fixture!("webp/ll_photo_alpha_31x16.webp"),
            (12, 24),
            0xb7d5_c7cb_e7c3_106d,
        ),
        (
            fixture!("webp/la_24x18.webp"),
            (14, 20),
            0x28e3_3b9d_b398_3cd1,
        ),
        (
            fixture!("webp/anim_mux_mixed.webp"),
            (14, 20),
            0xbab2_3251_2229_aa6d,
        ),
    ];

    #[test]
    fn image_files_match_the_reference_pipeline_fingerprints() {
        use crate::vision::preprocess::{fnv1a64_f32, patch_rows_f32};
        let vs = qwen38_vision_spec();
        let limits = DecodeLimits {
            max_pixels: 1 << 24,
        };
        for &((name, bytes), grid, hash) in REFERENCE_STILLS {
            let img =
                decode::decode_image(bytes, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (gh, gw, rows) = patch_rows_f32(&img, &vs).unwrap();
            assert_eq!((gh, gw), grid, "grid of {name}");
            assert_eq!(fnv1a64_f32(&rows), hash, "pixel fingerprint of {name}");
            // the header-only size (token counting) agrees with the decoded picture
            let p = decode::probe(bytes).unwrap();
            assert_eq!(
                (p.width, p.height),
                (img.width, img.height),
                "probe of {name}"
            );
        }
    }

    /// Animations through the video path: `(file, grid (temporal patches, rows, columns),
    /// FNV-1a of the f32 patch rows, time labels)` from the reference library's frames through
    /// the checkpoint's own video processor at the rate the file's delays give
    /// (`make_pipeline_refs.py --video --fingerprints`).
    type VideoRef = (
        (&'static str, &'static [u8]),
        (usize, usize, usize),
        u64,
        &'static [&'static str],
    );
    const REFERENCE_ANIMATIONS: &[VideoRef] = &[
        // 30 frames at 10 per second: 6 kept; "1.4" is a rounding tie of the mean of two times
        (
            fixture!("pipeline/anim_30f.gif"),
            (3, 2, 4),
            0xa399_2f74_730d_1974,
            &["0.3", "1.4", "2.6"],
        ),
        // 7 frames at 2 per second: all kept, the last frame twice in the last temporal patch
        (
            fixture!("pipeline/anim_7f.gif"),
            (4, 4, 4),
            0x8399_dc9a_516f_ae8d,
            &["0.2", "1.2", "2.2", "3.0"],
        ),
        // 9 frames at 4 per second (lossless and lossy frames): 4 kept
        (
            fixture!("pipeline/anim_9f.webp"),
            (2, 4, 4),
            0xfc4c_71e0_704c_dbda,
            &["0.4", "1.6"],
        ),
        (
            fixture!("pipeline/anim_9f_lossy.webp"),
            (2, 4, 4),
            0x9f75_ca07_3511_8c9f,
            &["0.4", "1.6"],
        ),
    ];

    #[test]
    fn animations_match_the_reference_video_pipeline_fingerprints() {
        use crate::fp8::f32_to_bf16_rne;
        use crate::vision::preprocess::{fnv1a64_f32, video_rows_f32};
        let sv = serve(16_384, 65_536);
        let vs = &sv.spec;
        let limits = DecodeLimits {
            max_pixels: 1 << 24,
        };
        for &((name, bytes), grid, hash, labels) in REFERENCE_ANIMATIONS {
            let url = format!("data:image/gif;base64,{}", base64(bytes));
            let m = RequestMedia::prepare(&[MediaRef::Animation(url)], &sv, true)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let MediaItem::Video(v) = &m.items[0] else {
                panic!("video expected")
            };
            let p0 = &v.patches[0];
            assert_eq!(
                (v.patches.len(), p0.grid_h, p0.grid_w),
                grid,
                "grid of {name}"
            );
            let got: Vec<String> = v.times.iter().map(|&t| format!("{t:.1}")).collect();
            assert_eq!(got, labels, "time labels of {name}");
            // the kept frames again, through the f32 form of the rows
            let scan = decode::scan_frames(bytes).unwrap();
            let total = scan.delays_ms.len();
            let ms: u64 = scan.delays_ms.iter().map(|&d| u64::from(d)).sum();
            let kept = sample_frames(total, total as f64 * 1000.0 / ms as f64, &vs.video.unwrap())
                .unwrap();
            assert_eq!(
                (v.frames_total, v.frames_kept),
                (total, kept.len()),
                "{name}"
            );
            let mut resized = Vec::new();
            decode::decode_frames(bytes, &limits, &kept, &mut |w, h, rgb| {
                resized.push(resize_rgb(rgb, w, h, p0.width, p0.height));
                Ok(())
            })
            .unwrap();
            let n = kept.len();
            let rows = video_rows_f32(
                &resized,
                p0.width,
                p0.height,
                grid.0,
                |g, place| video_group_frame(g, place, vs.temporal, n),
                vs,
            );
            assert_eq!(fnv1a64_f32(&rows), hash, "pixel fingerprint of {name}");
            let bf16: Vec<u16> = v
                .patches
                .iter()
                .flat_map(|p| p.pixels.iter().copied())
                .collect();
            assert!(
                bf16 == rows.iter().map(|&x| f32_to_bf16_rne(x)).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }

    /// Checks the whole host pipeline (file decoder, EXIF orientation, resize, patch rows)
    /// against records of the reference loader and image processor: set
    /// `GAUDIUM_IMAGE_PIPELINE_REF_DIR` to a directory written by
    /// `tests/data/vision/make_pipeline_refs.py` and run with `--ignored`. Every `<file>` there
    /// with a `<file>.f32` (the processor's patch rows) and a `<file>.json` (its grid and pixel
    /// cap) must give the same rows bit for bit.
    #[test]
    #[ignore = "needs GAUDIUM_IMAGE_PIPELINE_REF_DIR with reference pipeline records"]
    fn reference_pipeline_records_match_bit_for_bit() {
        use crate::json::parse_json;
        use crate::vision::preprocess::patch_rows_f32;
        let dir = std::path::PathBuf::from(
            std::env::var("GAUDIUM_IMAGE_PIPELINE_REF_DIR")
                .expect("GAUDIUM_IMAGE_PIPELINE_REF_DIR is not set"),
        );
        let limits = DecodeLimits {
            max_pixels: 256_000_000,
        };
        let (mut checked, mut bad) = (0usize, Vec::new());
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let (rows_path, meta_path) = (
                dir.join(format!("{name}.f32")),
                dir.join(format!("{name}.json")),
            );
            if name.ends_with(".f32") || name.ends_with(".json") {
                continue;
            }
            // an animation's record: the file as a video
            let video_rows = dir.join(format!("{name}.video.f32"));
            if video_rows.exists() {
                let meta = parse_json(
                    &std::fs::read_to_string(dir.join(format!("{name}.video.json"))).unwrap(),
                )
                .unwrap();
                let grid: Vec<usize> = meta
                    .get("grid")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();
                let labels: Vec<String> = meta
                    .get("labels")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect();
                let url = format!(
                    "data:application/octet-stream;base64,{}",
                    base64(&std::fs::read(&path).unwrap())
                );
                checked += 1;
                match RequestMedia::prepare(
                    &[MediaRef::Animation(url)],
                    &serve(65_536, 1 << 20),
                    true,
                ) {
                    Ok(m) => {
                        let MediaItem::Video(v) = &m.items[0] else {
                            panic!("video expected")
                        };
                        let got_labels: Vec<String> =
                            v.times.iter().map(|&t| time_label(t)).collect();
                        let p0 = &v.patches[0];
                        let want: Vec<u16> = std::fs::read(&video_rows)
                            .unwrap()
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| crate::fp8::f32_to_bf16_rne(f32::from_le_bytes(*b)))
                            .collect();
                        let got: Vec<u16> = v
                            .patches
                            .iter()
                            .flat_map(|p| p.pixels.iter().copied())
                            .collect();
                        if vec![v.patches.len(), p0.grid_h, p0.grid_w] != grid {
                            bad.push(format!(
                                "{name}: video grid {}x{}x{}, reference {grid:?}",
                                v.patches.len(),
                                p0.grid_h,
                                p0.grid_w
                            ));
                        } else if got_labels != labels {
                            bad.push(format!(
                                "{name}: time labels {got_labels:?}, reference {labels:?}"
                            ));
                        } else if got != want {
                            bad.push(format!("{name}: video patch rows differ"));
                        }
                    }
                    Err(e) => bad.push(format!("{name}: {e}")),
                }
                continue;
            }
            if !rows_path.exists() {
                continue;
            }
            let meta = parse_json(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
            let grid = meta.get("grid").unwrap().as_array().unwrap();
            let want_grid = (
                grid[0].as_u64().unwrap() as usize,
                grid[1].as_u64().unwrap() as usize,
            );
            let mut vs = qwen38_vision_spec();
            vs.max_pixels = meta.get("max_pixels").unwrap().as_u64().unwrap();
            let got = decode::decode_image(&std::fs::read(&path).unwrap(), &limits)
                .and_then(|img| patch_rows_f32(&img, &vs));
            checked += 1;
            match got {
                Ok((gh, gw, rows)) => {
                    let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
                    if (gh, gw) != want_grid {
                        bad.push(format!("{name}: grid {gh}x{gw}, reference {want_grid:?}"));
                    } else if bytes != std::fs::read(&rows_path).unwrap() {
                        bad.push(format!("{name}: patch rows differ"));
                    }
                }
                Err(e) => bad.push(format!("{name}: {e}")),
            }
        }
        assert!(checked > 0, "no reference records in {}", dir.display());
        assert!(
            bad.is_empty(),
            "{} of {checked} differ:\n{}",
            bad.len(),
            bad.join("\n")
        );
        eprintln!("{checked} files bit-identical to the reference pipeline");
    }

    #[test]
    fn video_errors_and_budgets() {
        let sv = serve(4096, 4096);
        let timed = |times: Vec<f64>| MediaRef::Frames {
            frames: (0..3)
                .map(|k| ppm_url(64, 64, &synthetic_frame(64, 64, k)))
                .collect(),
            fps: None,
            times: Some(times),
        };
        assert!(RequestMedia::prepare(&[timed(vec![0.0, 1.0])], &sv, false)
            .unwrap_err()
            .contains("2 timestamps for 3 frames"));
        assert!(
            RequestMedia::prepare(&[timed(vec![0.0, 2.0, 1.0])], &sv, false)
                .unwrap_err()
                .contains("not decreasing")
        );
        let m = RequestMedia::prepare(&[timed(vec![0.0, 1.0, 4.0])], &sv, false).unwrap();
        let MediaItem::Video(v) = &m.items[0] else {
            panic!("video expected")
        };
        assert_eq!(v.times, vec![0.5, 4.0]);
        // frames of different sizes are refused
        let mixed = MediaRef::Frames {
            frames: vec![
                ppm_url(64, 64, &synthetic_frame(64, 64, 0)),
                ppm_url(96, 64, &synthetic_frame(96, 64, 1)),
            ],
            fps: Some(2.0),
            times: None,
        };
        assert!(RequestMedia::prepare(&[mixed], &sv, true)
            .unwrap_err()
            .contains("must have one size"));
        // frames smaller than one token are refused like the reference refuses them
        assert!(
            RequestMedia::prepare(&[frames_ref(20, 20, 4, Some(2.0))], &sv, false)
                .unwrap_err()
                .contains("smaller than 32")
        );
        assert!(
            RequestMedia::prepare(&[frames_ref(64, 64, 4, Some(0.0))], &sv, false)
                .unwrap_err()
                .contains("fps")
        );
        // a model without video settings takes none
        let mut spec = qwen38_vision_spec();
        spec.video = None;
        let no_video = VisionServe::new(spec, 4096, 4096, MediaLimits::default()).unwrap();
        assert!(
            RequestMedia::prepare(&[frames_ref(64, 64, 4, Some(2.0))], &no_video, false)
                .unwrap_err()
                .contains("no video input")
        );
        // few embedding rows left: the video is resized down to fit them
        let tight = serve(4096, 16);
        let m =
            RequestMedia::prepare(&[frames_ref(640, 480, 8, Some(2.0))], &tight, false).unwrap();
        assert!(m.counts().tokens <= 16, "{:?}", m.counts());
        assert_eq!(m.counts().inputs, 4);
        let none = serve(4096, 3);
        assert!(
            RequestMedia::prepare(&[frames_ref(640, 480, 8, Some(2.0))], &none, false)
                .unwrap_err()
                .contains("temporal patches need at least 4")
        );
        // a small tower caps every frame
        let small = serve(256, 4096);
        let m =
            RequestMedia::prepare(&[frames_ref(640, 480, 8, Some(2.0))], &small, false).unwrap();
        let MediaItem::Video(v) = &m.items[0] else {
            panic!("video expected")
        };
        assert!(v.patches[0].n_patches() <= 256);
    }

    #[test]
    fn prompt_text_and_ids_are_expanded_in_order() {
        let sv = serve(4096, 4096);
        let vs = &sv.spec;
        let img = MediaRef::Image(ppm_url(256, 256, &synthetic_frame(256, 256, 0)));
        let vid = frames_ref(96, 64, 7, Some(2.0));
        let m = RequestMedia::prepare(&[img.clone(), vid.clone()], &sv, false).unwrap();
        let text = format!("A{IMAGE_BLOCK}B{VIDEO_BLOCK}C");
        let out = m.expand_text(&text).unwrap();
        assert_eq!(
            out,
            format!(
                "A{IMAGE_BLOCK}B<0.2 seconds>{VIDEO_BLOCK}<1.2 seconds>{VIDEO_BLOCK}<2.2 seconds>{VIDEO_BLOCK}<3.0 seconds>{VIDEO_BLOCK}C"
            )
        );
        // order and count are checked
        assert!(m
            .expand_text(&format!("{VIDEO_BLOCK}{IMAGE_BLOCK}"))
            .unwrap_err()
            .contains("not in the order"));
        assert!(m
            .expand_text(IMAGE_BLOCK)
            .unwrap_err()
            .contains("more images and videos than"));
        assert!(m
            .expand_text(&format!("{text}{IMAGE_BLOCK}"))
            .unwrap_err()
            .contains("more image or video placeholders"));

        // ids: start, pad, end per input; the video's four patches use the video pad
        let (s, e, ip, vp) = (
            vs.vision_start_id,
            vs.vision_end_id,
            vs.image_token_id,
            vs.video_token_id,
        );
        let mut ids = vec![1, s, ip, e, 2];
        for _ in 0..4 {
            ids.extend([7, s, vp, e]);
        }
        ids.push(3);
        let x = m.expand_ids(&ids, vs).unwrap();
        assert_eq!(x.iter().filter(|&&t| t == ip).count(), 64);
        assert_eq!(x.iter().filter(|&&t| t == vp).count(), 24);
        assert_eq!(x.len(), ids.len() - 5 + 64 + 24);
        // an image pad where a video pad belongs is refused
        let mut wrong = ids.clone();
        wrong[6] = ip;
        assert!(m
            .expand_ids(&wrong, vs)
            .unwrap_err()
            .contains("not in the order"));
        let mut extra = ids.clone();
        extra.extend([s, ip, e]);
        assert!(m
            .expand_ids(&extra, vs)
            .unwrap_err()
            .contains("more image or video placeholders"));
        // the engine gets five inputs: the image, then the temporal patches
        let full = RequestMedia::prepare(&[img, vid], &sv, true).unwrap();
        assert_eq!(full.into_tower_inputs().len(), 5);
    }
}
