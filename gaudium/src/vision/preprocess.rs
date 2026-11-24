//! Host image preparation for the vision tower.
//!
//! Input is uncompressed RGB8 (what `vision::decode` returns for an image file, a binary PPM
//! `P6`, or raw bytes with explicit width / height). The pipeline is the checkpoint's image
//! processor:
//!
//! 1. `smart_resize`: both sides to multiples of `patch * merge`, pixel count inside
//!    `[min_pixels, max_pixels]`, aspect ratio kept as closely as the grid allows.
//! 2. `resize_rgb`: antialiased bicubic resampling on uint8 with int16 fixed-point weights,
//!    horizontal pass first, each pass rounded and clamped to uint8.
//! 3. Normalise `(x - mean * 255) / (std * 255)` in f32 and round to BF16 (nearest even).
//! 4. Patch rows in merge order (2x2 blocks of patches stay adjacent), each row laid out
//!    channel, temporal frame, y, x. A still image is `temporal` identical frames.
//!
//! A video goes the same way with two differences (the checkpoint's video processor):
//! `smart_resize_video` shares one pixel budget among all frames, and a patch row holds
//! `temporal` consecutive frames instead of one frame repeated (`prepare_video`). Each temporal
//! patch is then one tower input of its own, exactly like an image.

use std::path::Path;

use crate::fp8::f32_to_bf16_rne;
use crate::vision::{ImageProcessor, VisionSpec};

/// Uncompressed 8-bit RGB image, rows top to bottom, 3 bytes per pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn from_raw(width: usize, height: usize, data: Vec<u8>) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("image has zero size".into());
        }
        let want = width
            .checked_mul(height)
            .and_then(|p| p.checked_mul(3))
            .ok_or("image dimensions overflow")?;
        if data.len() != want {
            return Err(format!(
                "raw RGB8 image {width}x{height} needs {want} bytes, got {}",
                data.len()
            ));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Parses a binary PPM (`P6`, maxval 255). Header comments (`# ...`) are accepted.
    pub fn parse_ppm(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 2 || &bytes[..2] != b"P6" {
            return Err(
                "not a binary PPM (P6) file; convert the image to PPM or pass raw RGB8".into(),
            );
        }
        let mut pos = 2usize;
        let mut fields = [0usize; 3];
        for f in &mut fields {
            // whitespace and comments before each header number
            loop {
                match bytes.get(pos) {
                    Some(b'#') => {
                        while pos < bytes.len() && bytes[pos] != b'\n' {
                            pos += 1;
                        }
                    }
                    Some(c) if c.is_ascii_whitespace() => pos += 1,
                    Some(_) => break,
                    None => return Err("truncated PPM header".into()),
                }
            }
            let start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            if start == pos || pos - start > 9 {
                return Err("malformed PPM header number".into());
            }
            *f = std::str::from_utf8(&bytes[start..pos])
                .ok()
                .and_then(|s| s.parse().ok())
                .ok_or("malformed PPM header number")?;
        }
        let [width, height, maxval] = fields;
        if maxval != 255 {
            return Err(format!(
                "PPM maxval {maxval} is not supported (8-bit, maxval 255 only)"
            ));
        }
        // exactly one whitespace byte separates the header from the pixel data
        if !bytes.get(pos).is_some_and(u8::is_ascii_whitespace) {
            return Err("malformed PPM header (no separator before pixel data)".into());
        }
        pos += 1;
        let want = width
            .checked_mul(height)
            .and_then(|p| p.checked_mul(3))
            .ok_or("PPM dimensions overflow")?;
        if bytes.len() - pos < want {
            return Err(format!(
                "PPM {width}x{height} needs {want} pixel bytes, file holds {}",
                bytes.len() - pos
            ));
        }
        Self::from_raw(width, height, bytes[pos..pos + want].to_vec())
    }

    pub fn load_ppm(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::parse_ppm(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Loads `WIDTHxHEIGHT:path` raw RGB8 (no header).
    pub fn load_raw_rgb(spec: &str) -> Result<Self, String> {
        let (dims, path) = spec
            .split_once(':')
            .ok_or("raw image spec must be WIDTHxHEIGHT:path")?;
        let (w, h) = dims
            .split_once(['x', 'X'])
            .ok_or("raw image spec must be WIDTHxHEIGHT:path")?;
        let width: usize = w
            .parse()
            .map_err(|_| format!("invalid raw image width '{w}'"))?;
        let height: usize = h
            .parse()
            .map_err(|_| format!("invalid raw image height '{h}'"))?;
        let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        Self::from_raw(width, height, data).map_err(|e| format!("{path}: {e}"))
    }
}

/// Target `(height, width)` of an image: multiples of `factor`, pixel count within
/// `[min_pixels, max_pixels]`. Rounding is half-to-even, as the reference processor's.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 || factor == 0 {
        return Err("image has zero size".into());
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err("absolute aspect ratio must be smaller than 200".into());
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Target `(height, width)` of every frame of a video of `num_frames` frames: multiples of
/// `factor`, and `frames x height x width` (frames rounded up to whole temporal patches) within
/// `[min_pixels, max_pixels]`. The reference video processor's rule, rounding included; like
/// it, a frame smaller than `factor` on a side is refused.
pub fn smart_resize_video(
    num_frames: usize,
    height: usize,
    width: usize,
    temporal: usize,
    factor: usize,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(usize, usize), String> {
    if num_frames == 0 || temporal == 0 || factor == 0 {
        return Err("video has no frames".into());
    }
    if height < factor || width < factor {
        return Err(format!(
            "video frames of {width}x{height} pixels are smaller than {factor} pixels on a side"
        ));
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err("absolute aspect ratio must be smaller than 200".into());
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    let t_bar = (num_frames.div_ceil(temporal) * temporal) as f64;
    let volume = (num_frames as f64) * h * w;
    if t_bar * h_bar * w_bar > max_pixels as f64 {
        let beta = (volume / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if t_bar * h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / volume).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

fn aa_cubic(x: f64) -> f64 {
    let a = -0.5f64;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

/// Fixed-point resampling taps of one axis: per output position a window `[lo, lo + n)` of the
/// input and `ksize` int16 weights at `precision` fractional bits.
struct AxisWeights {
    lo: Vec<usize>,
    n: Vec<usize>,
    w: Vec<i16>,
    ksize: usize,
    precision: u32,
}

fn make_weights(input: usize, output: usize) -> AxisWeights {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let ksize = (support.ceil() as usize) * 2 + 1;
    let mut lo = vec![0usize; output];
    let mut n = vec![0usize; output];
    let mut wd = vec![0.0f64; output * ksize];
    let mut wmax = 0.0f64;
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xsize = (((center + support + 0.5) as i64).min(input as i64) - xmin)
            .clamp(0, ksize as i64) as usize;
        let row = &mut wd[i * ksize..(i + 1) * ksize];
        let mut total = 0.0f64;
        for (j, slot) in row.iter_mut().enumerate().take(xsize) {
            let v = aa_cubic(((j as i64 + xmin) as f64 - center + 0.5) * invscale);
            *slot = v;
            total += v;
        }
        if total != 0.0 {
            for slot in row.iter_mut().take(xsize) {
                *slot /= total;
                wmax = wmax.max(*slot);
            }
        }
        lo[i] = xmin as usize;
        n[i] = xsize;
    }
    // the largest precision that keeps every weight below 2^15
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + wmax * f64::from(1u32 << (precision + 1))) as i32;
        if next >= (1 << 15) {
            break;
        }
        precision += 1;
    }
    let unit = f64::from(1u32 << precision);
    let w = wd
        .iter()
        .map(|&v| {
            let v = v * unit;
            (if v < 0.0 {
                (v - 0.5).trunc()
            } else {
                (v + 0.5).trunc()
            }) as i16
        })
        .collect();
    AxisWeights {
        lo,
        n,
        w,
        ksize,
        precision,
    }
}

#[inline]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Antialiased bicubic resize of RGB8 (`a = -0.5`, the kernel support widened by the downscale
/// factor). A dimension whose size does not change is not resampled.
pub fn resize_rgb(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), w * h * 3);
    let mut cur = src.to_vec();
    let mut cw = w;
    if ow != w {
        let ax = make_weights(w, ow);
        let init = 1i32 << (ax.precision - 1);
        let mut nx = vec![0u8; h * ow * 3];
        for y in 0..h {
            let row = &cur[y * cw * 3..(y + 1) * cw * 3];
            for x in 0..ow {
                let k = &ax.w[x * ax.ksize..x * ax.ksize + ax.n[x]];
                let base = ax.lo[x] * 3;
                for c in 0..3 {
                    let mut s = init;
                    for (j, &kj) in k.iter().enumerate() {
                        s += i32::from(row[base + j * 3 + c]) * i32::from(kj);
                    }
                    nx[(y * ow + x) * 3 + c] = clip8(s >> ax.precision);
                }
            }
        }
        cur = nx;
        cw = ow;
    }
    if oh != h {
        let ax = make_weights(h, oh);
        let init = 1i32 << (ax.precision - 1);
        let mut ny = vec![0u8; oh * cw * 3];
        for y in 0..oh {
            let k = &ax.w[y * ax.ksize..y * ax.ksize + ax.n[y]];
            for x in 0..cw {
                for c in 0..3 {
                    let mut s = init;
                    for (j, &kj) in k.iter().enumerate() {
                        s += i32::from(cur[((ax.lo[y] + j) * cw + x) * 3 + c]) * i32::from(kj);
                    }
                    ny[(y * cw + x) * 3 + c] = clip8(s >> ax.precision);
                }
            }
        }
        cur = ny;
    }
    cur
}

/// One image ready for the vision tower.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedImage {
    pub src_width: usize,
    pub src_height: usize,
    /// Resized pixel size.
    pub width: usize,
    pub height: usize,
    /// Patch grid of the resized image (rows, columns); both even.
    pub grid_h: usize,
    pub grid_w: usize,
    /// Values per patch row (`VisionSpec::patch_row`; a Gemma 4 unified row holds a whole
    /// token's `merge x merge` patches).
    pub row_len: usize,
    /// Patches along each axis that make one decoder token (`VisionSpec::merge`).
    pub merge: usize,
    /// BF16 patch rows `[n_patches][row_len]` in merge order (the tower's `{row_len, N}` input).
    pub pixels: Vec<u16>,
}

impl PreparedImage {
    /// The sizes of a tower input without its pixels (`pixels` is empty): enough to count
    /// tokens and lay a prompt out, not to encode.
    pub fn sized(src: (usize, usize), rw: usize, rh: usize, vs: &VisionSpec) -> Self {
        let p = vs.patch as usize;
        Self {
            src_width: src.0,
            src_height: src.1,
            width: rw,
            height: rh,
            grid_h: rh / p,
            grid_w: rw / p,
            row_len: crate::engine::gemma4::vision::row_len(vs),
            merge: vs.merge as usize,
            pixels: Vec::new(),
        }
    }

    pub fn n_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Decoder tokens (image-pad rows) this image occupies.
    pub fn n_tokens(&self) -> usize {
        crate::core::vision::grid_tokens(self.grid_h, self.grid_w, self.merge as u64) as usize
    }
}

/// f32 mean and std scaled to the 0..255 pixel range (rescale 1/255 folded in).
fn mean_std_255(vs: &VisionSpec) -> ([f32; 3], [f32; 3]) {
    let mut mean = [0f32; 3];
    let mut std = [0f32; 3];
    for c in 0..3 {
        mean[c] = (vs.image_mean[c] * (1.0 / (1.0 / 255.0))) as f32;
        std[c] = (vs.image_std[c] * (1.0 / (1.0 / 255.0))) as f32;
    }
    (mean, std)
}

/// Normalised f32 value of pixel byte `v` in channel `c`.
#[inline]
fn norm_pixel(v: u8, mean255: f32, std255: f32) -> f32 {
    (f32::from(v) - mean255) / std255
}

/// Walks one temporal patch in patch-row order, calling `put(channel, byte)` once per value:
/// 2x2 merge blocks row-major, the block's patches row-major, then channel, frame, y, x.
/// `frames` holds the temporal patch's resized frames (`VisionSpec::temporal` of them, all
/// `rw` pixels wide); a still image passes the same frame in every place.
fn for_each_patch_value(
    frames: &[&[u8]],
    rw: usize,
    grid_h: usize,
    grid_w: usize,
    vs: &VisionSpec,
    mut put: impl FnMut(usize, u8),
) {
    let (p, m) = (vs.patch as usize, vs.merge as usize);
    for bh in 0..grid_h / m {
        for bw in 0..grid_w / m {
            for mh in 0..m {
                for mw in 0..m {
                    let y0 = (bh * m + mh) * p;
                    let x0 = (bw * m + mw) * p;
                    for c in 0..3 {
                        for frame in frames {
                            for py in 0..p {
                                let row = ((y0 + py) * rw + x0) * 3 + c;
                                for px in 0..p {
                                    put(c, frame[row + px * 3]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The resized `(height, width)` of a `width x height` image under the model's image settings.
pub fn image_target_size(
    width: usize,
    height: usize,
    vs: &VisionSpec,
) -> Result<(usize, usize), String> {
    if vs.processor == ImageProcessor::Tiles {
        return crate::engine::inkling2::vision::padded_size(width, height, vs);
    }
    if let ImageProcessor::Gemma4 { .. } = vs.processor {
        return crate::engine::gemma4::vision::target_size(width, height, vs);
    }
    smart_resize(
        height,
        width,
        vs.resize_factor() as usize,
        vs.min_pixels,
        vs.max_pixels,
    )
}

fn resize_for_tower(img: &RgbImage, vs: &VisionSpec) -> Result<(usize, usize, Vec<u8>), String> {
    let (rh, rw) = image_target_size(img.width, img.height, vs)?;
    let resized = resize_rgb(&img.data, img.width, img.height, rw, rh);
    Ok((rh, rw, resized))
}

/// BF16 value of every pixel byte per channel: normalised and rounded once.
fn bf16_lut(vs: &VisionSpec) -> [[u16; 256]; 3] {
    let (mean, std) = mean_std_255(vs);
    let mut lut = [[0u16; 256]; 3];
    for c in 0..3 {
        for v in 0..=255u8 {
            lut[c][v as usize] = f32_to_bf16_rne(norm_pixel(v, mean[c], std[c]));
        }
    }
    lut
}

/// BF16 patch rows of one temporal patch from its resized frames.
fn patch_rows_bf16(
    frames: &[&[u8]],
    rw: usize,
    rh: usize,
    src: (usize, usize),
    lut: &[[u16; 256]; 3],
    vs: &VisionSpec,
) -> PreparedImage {
    let p = vs.patch as usize;
    let (grid_h, grid_w) = (rh / p, rw / p);
    let row_len = vs.patch_row() as usize;
    let mut pixels = Vec::with_capacity(grid_h * grid_w * row_len);
    for_each_patch_value(frames, rw, grid_h, grid_w, vs, |c, v| {
        pixels.push(lut[c][v as usize]);
    });
    debug_assert_eq!(pixels.len(), grid_h * grid_w * row_len);
    PreparedImage {
        merge: vs.merge as usize,
        src_width: src.0,
        src_height: src.1,
        width: rw,
        height: rh,
        grid_h,
        grid_w,
        row_len,
        pixels,
    }
}

/// Full host pipeline: resize, normalise, patch rows in merge order, BF16.
pub fn prepare_image(img: &RgbImage, vs: &VisionSpec) -> Result<PreparedImage, String> {
    if vs.processor == ImageProcessor::Tiles {
        return crate::engine::inkling2::vision::prepare_tiles(img, vs);
    }
    if let ImageProcessor::Gemma4 { .. } = vs.processor {
        return crate::engine::gemma4::vision::prepare_image(img, vs);
    }
    let (rh, rw, resized) = resize_for_tower(img, vs)?;
    let frames = vec![resized.as_slice(); vs.temporal as usize];
    Ok(patch_rows_bf16(
        &frames,
        rw,
        rh,
        (img.width, img.height),
        &bf16_lut(vs),
        vs,
    ))
}

/// The same patch rows as f32 (before BF16 rounding): what the reference processor returns as
/// `pixel_values`. Used to check the pipeline against stored reference dumps.
pub fn patch_rows_f32(img: &RgbImage, vs: &VisionSpec) -> Result<(usize, usize, Vec<f32>), String> {
    if vs.processor == ImageProcessor::Tiles {
        return crate::engine::inkling2::vision::tile_rows_f32(img, vs);
    }
    if let ImageProcessor::Gemma4 { .. } = vs.processor {
        return crate::engine::gemma4::vision::pixel_rows_f32(img, vs);
    }
    let (rh, rw, resized) = resize_for_tower(img, vs)?;
    let p = vs.patch as usize;
    let (grid_h, grid_w) = (rh / p, rw / p);
    let (mean, std) = mean_std_255(vs);
    let frames = vec![resized.as_slice(); vs.temporal as usize];
    let mut out = Vec::with_capacity(grid_h * grid_w * vs.patch_row() as usize);
    for_each_patch_value(&frames, rw, grid_h, grid_w, vs, |c, v| {
        out.push(norm_pixel(v, mean[c], std[c]));
    });
    Ok((grid_h, grid_w, out))
}

/// The temporal patches of a video from its resized frames (`rw x rh` pixels each): patch `g`
/// holds the frames `frame_of(g, place)` for `place` in `0..temporal` (the caller repeats the
/// last frame to fill the last patch). One `PreparedImage` per temporal patch.
pub fn prepare_video(
    resized: &[Vec<u8>],
    rw: usize,
    rh: usize,
    src: (usize, usize),
    groups: usize,
    frame_of: impl Fn(usize, usize) -> usize,
    vs: &VisionSpec,
) -> Result<Vec<PreparedImage>, String> {
    let lut = bf16_lut(vs);
    let t = vs.temporal as usize;
    let mut out = Vec::with_capacity(groups);
    for g in 0..groups {
        let mut frames: Vec<&[u8]> = Vec::with_capacity(t);
        for place in 0..t {
            let k = frame_of(g, place);
            let f = resized.get(k).ok_or_else(|| {
                format!("temporal patch {g} names frame {k} of {}", resized.len())
            })?;
            if f.len() != rw * rh * 3 {
                return Err("a resized video frame has the wrong size".into());
            }
            frames.push(f);
        }
        out.push(patch_rows_bf16(&frames, rw, rh, src, &lut, vs));
    }
    Ok(out)
}

/// f32 patch rows of a whole video (temporal patch after temporal patch), before BF16
/// rounding: the reference video processor's `pixel_values_videos`.
pub fn video_rows_f32(
    resized: &[Vec<u8>],
    rw: usize,
    rh: usize,
    groups: usize,
    frame_of: impl Fn(usize, usize) -> usize,
    vs: &VisionSpec,
) -> Vec<f32> {
    let p = vs.patch as usize;
    let (grid_h, grid_w) = (rh / p, rw / p);
    let (mean, std) = mean_std_255(vs);
    let t = vs.temporal as usize;
    let mut out = Vec::with_capacity(groups * grid_h * grid_w * vs.patch_row() as usize);
    for g in 0..groups {
        let frames: Vec<&[u8]> = (0..t).map(|k| resized[frame_of(g, k)].as_slice()).collect();
        for_each_patch_value(&frames, rw, grid_h, grid_w, vs, |c, v| {
            out.push(norm_pixel(v, mean[c], std[c]));
        });
    }
    out
}

/// FNV-1a 64 over the little-endian bytes of f32 values (compact reference fingerprints).
pub fn fnv1a64_f32(values: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::bf16_to_f32;
    use crate::vision::test_util::qwen38_vision_spec;

    /// Deterministic synthetic test image (no files): smooth gradients plus a hashed texture.
    pub(crate) fn synthetic_image(width: usize, height: usize) -> RgbImage {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let n = ((x as u32).wrapping_mul(73_856_093) ^ (y as u32).wrapping_mul(19_349_663))
                    >> 7;
                data.push(((x * 255) / width.max(1)) as u8);
                data.push(((y * 255) / height.max(1)) as u8);
                data.push((n & 0xFF) as u8);
            }
        }
        RgbImage::from_raw(width, height, data).unwrap()
    }

    #[test]
    fn ppm_parser_accepts_comments_and_rejects_bad_input() {
        let mut ppm = b"P6\n# a comment\n2 1\n255\n".to_vec();
        ppm.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let img = RgbImage::parse_ppm(&ppm).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.data, vec![1, 2, 3, 4, 5, 6]);

        assert!(RgbImage::parse_ppm(b"P3\n1 1\n255\n0 0 0").is_err());
        assert!(RgbImage::parse_ppm(b"P6\n2 2\n255\n\x00\x00\x00").is_err());
        assert!(RgbImage::parse_ppm(b"P6\n1 1\n65535\n\x00\x00\x00\x00\x00\x00").is_err());
        assert!(RgbImage::from_raw(2, 2, vec![0; 11]).is_err());
    }

    #[test]
    fn smart_resize_matches_reference_processor_cases() {
        // multiples of 32 inside the pixel range are kept
        assert_eq!(
            smart_resize(256, 256, 32, 65_536, 16_777_216).unwrap(),
            (256, 256)
        );
        assert_eq!(
            smart_resize(448, 672, 32, 65_536, 16_777_216).unwrap(),
            (448, 672)
        );
        // half-to-even: 48 / 32 = 1.5 -> 2, 80 / 32 = 2.5 -> 2 (then lifted to min_pixels)
        assert_eq!(
            smart_resize(700, 1000, 32, 65_536, 16_777_216).unwrap(),
            (704, 992)
        );
        // below min_pixels: scaled up by sqrt(min / area), ceil to the grid
        assert_eq!(
            smart_resize(50, 70, 32, 65_536, 16_777_216).unwrap(),
            (224, 320)
        );
        // above max_pixels: scaled down, floor to the grid
        assert_eq!(
            smart_resize(3000, 4000, 32, 65_536, 1_048_576).unwrap(),
            (864, 1152)
        );
        assert!(smart_resize(1, 300, 32, 65_536, 16_777_216).is_err());
        assert!(smart_resize(0, 10, 32, 65_536, 16_777_216).is_err());
    }

    #[test]
    fn solid_image_gives_uniform_normalised_patch_rows() {
        let vs = qwen38_vision_spec();
        let mut data = Vec::new();
        for _ in 0..256 * 256 {
            data.extend_from_slice(&[255, 0, 128]);
        }
        let img = RgbImage::from_raw(256, 256, data).unwrap();
        let p = prepare_image(&img, &vs).unwrap();
        assert_eq!(
            (p.grid_h, p.grid_w, p.n_patches(), p.n_tokens()),
            (16, 16, 256, 64)
        );
        assert_eq!(p.row_len, 1536);
        assert_eq!(p.pixels.len(), 256 * 1536);
        // row = [R x 512 | G x 512 | B x 512] (channel, 2 frames, 16, 16)
        let row = &p.pixels[..1536];
        assert!(row[..512].iter().all(|&v| bf16_to_f32(v) == 1.0));
        assert!(row[512..1024].iter().all(|&v| bf16_to_f32(v) == -1.0));
        let b = (128.0f32 - 127.5) / 127.5;
        assert!(row[1024..].iter().all(|&v| v == f32_to_bf16_rne(b)));
    }

    #[test]
    fn patch_rows_follow_merge_order_and_duplicate_the_frame() {
        let vs = qwen38_vision_spec();
        // 64 x 64 would be below min_pixels; use 256 x 256 with a per-patch constant red value
        let (w, h) = (256usize, 256usize);
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                data[(y * w + x) * 3] = ((y / 16) * 16 + (x / 16)) as u8; // patch index, row-major
            }
        }
        let img = RgbImage::from_raw(w, h, data).unwrap();
        let (gh, gw, rows) = patch_rows_f32(&img, &vs).unwrap();
        assert_eq!((gh, gw), (16, 16));
        let patch_of_row = |r: usize| (rows[r * 1536] * 127.5 + 127.5).round() as usize;
        // merge order: (0,0) (0,1) (1,0) (1,1) of block 0, then block 1 = columns 2..4
        assert_eq!(
            (0..8).map(patch_of_row).collect::<Vec<_>>(),
            vec![0, 1, 16, 17, 2, 3, 18, 19]
        );
        // both temporal frames of a channel hold the same pixels
        assert_eq!(rows[0..256], rows[256..512]);
    }

    #[test]
    fn resize_is_identity_when_the_size_is_unchanged_and_exact_on_flat_colour() {
        let img = synthetic_image(96, 64);
        assert_eq!(resize_rgb(&img.data, 96, 64, 96, 64), img.data);
        let flat = vec![200u8; 40 * 30 * 3];
        let up = resize_rgb(&flat, 40, 30, 64, 96);
        assert!(up.iter().all(|&v| v == 200));
        let down = resize_rgb(&flat, 40, 30, 16, 8);
        assert!(down.iter().all(|&v| v == 200));
    }

    /// Reference fingerprints from the checkpoint's own reference image processor, run on the
    /// CPU on the same synthetic images: patch grid and FNV-1a of its f32 patch rows. Equal
    /// fingerprints mean the whole host pipeline is bit-identical on that image.
    #[test]
    fn resize_paths_match_reference_processor_fingerprints() {
        let vs = qwen38_vision_spec();
        for &(w, h, grid, hash) in REFERENCE_FINGERPRINTS {
            let (gh, gw, rows) = patch_rows_f32(&synthetic_image(w, h), &vs).unwrap();
            assert_eq!((gh, gw), grid, "grid of {w}x{h}");
            assert_eq!(fnv1a64_f32(&rows), hash, "pixel fingerprint of {w}x{h}");
        }
    }

    /// `(width, height, (grid_h, grid_w), fnv1a64 of pixel_values)`.
    const REFERENCE_FINGERPRINTS: &[(usize, usize, (usize, usize), u64)] = &[
        (70, 50, (14, 20), 0x0eba_e011_0d31_3bdd), // both sides upscaled
        (1000, 700, (44, 62), 0x7a98_a840_6947_ccdd), // width down, height up
        (333, 517, (32, 20), 0xd2d5_450b_9b3c_0b05), // both sides downscaled
        (1300, 900, (56, 82), 0x2e6f_81d3_c46c_a0d1), // width up, height down
        (640, 256, (16, 40), 0x5934_669b_72ec_ec31), // unchanged size
    ];
}
