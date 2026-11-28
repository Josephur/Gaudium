//! Host decoders for the image files a request carries: JPEG, PNG, GIF, WebP, BMP and binary
//! PPM, each written for this crate (no dependencies) and bit-exact with the decoder of the
//! checkpoint's reference image processor, so the patch rows the tower gets are the rows the
//! reference pipeline computes from the same file.
//!
//! This module picks the decoder from the file's first bytes (never from a declared media
//! type), applies the EXIF orientation the way the reference loader does, and gives animations
//! (GIF, WebP) as frames with their display times. Size limits are checked from the header,
//! before a decoder allocates pixel buffers (`DecodeLimits::check_canvas`).

pub mod types;

mod bmp;
mod gif;
pub(crate) mod inflate;
mod jpeg;
mod png;
mod vp8;
mod vp8l;
mod webp;

pub use types::{DecodeLimits, Scan, Still};

use crate::core::vision::mirror;
use crate::vision::preprocess::RgbImage;

/// An image file format this crate decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Gif,
    Webp,
    Bmp,
    Ppm,
}

impl ImageFormat {
    pub fn name(self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG",
            Self::Png => "PNG",
            Self::Gif => "GIF",
            Self::Webp => "WebP",
            Self::Bmp => "BMP",
            Self::Ppm => "PPM",
        }
    }
}

/// The format of a file by its signature, or why it cannot be decoded.
pub fn sniff(bytes: &[u8]) -> Result<ImageFormat, String> {
    let at = |off: usize, sig: &[u8]| bytes.get(off..off + sig.len()) == Some(sig);
    if at(0, &[0xFF, 0xD8, 0xFF]) {
        return Ok(ImageFormat::Jpeg);
    }
    if at(0, &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Ok(ImageFormat::Png);
    }
    if at(0, b"GIF87a") || at(0, b"GIF89a") {
        return Ok(ImageFormat::Gif);
    }
    if at(0, b"RIFF") && at(8, b"WEBP") {
        return Ok(ImageFormat::Webp);
    }
    if at(0, b"BM") {
        return Ok(ImageFormat::Bmp);
    }
    if at(0, b"P6") {
        return Ok(ImageFormat::Ppm);
    }
    const TAKES: &str = "JPEG, PNG, GIF, WebP and BMP are decoded";
    // containers a client may send by mistake: name them, so the error says what to do
    let video = (at(4, b"ftyp")
        && !at(8, b"avif")
        && !at(8, b"avis")
        && !at(8, b"heic")
        && !at(8, b"heix")
        && !at(8, b"mif1"))
        || at(0, &[0x1A, 0x45, 0xDF, 0xA3])
        || (at(0, b"RIFF") && at(8, b"AVI "))
        || at(0, b"OggS");
    if video {
        return Err(
            "this is a video file, which the server does not decode: send the video as its frames or as an animated GIF / WebP".into(),
        );
    }
    if at(4, b"ftyp") {
        return Err(format!("AVIF / HEIC images are not decoded ({TAKES})"));
    }
    if at(0, b"II*\0") || at(0, b"MM\0*") {
        return Err(format!("TIFF images are not decoded ({TAKES})"));
    }
    if bytes.is_empty() {
        return Err("the image is empty".into());
    }
    Err(format!("not an image this server decodes ({TAKES})"))
}

/// Orientation (1..=8) stored in an EXIF block (a TIFF structure); 1 when there is none or the
/// block cannot be read.
pub fn exif_orientation(tiff: &[u8]) -> u8 {
    let le = match tiff.get(..4) {
        Some([b'I', b'I', 42, 0]) => true,
        Some([b'M', b'M', 0, 42]) => false,
        _ => return 1,
    };
    let u16_at = |off: usize| -> Option<u32> {
        let b = tiff.get(off..off.checked_add(2)?)?;
        Some(u32::from(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        }))
    };
    let u32_at = |off: usize| -> Option<u32> {
        let b = tiff.get(off..off.checked_add(4)?)?;
        Some(if le {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    };
    let read = || -> Option<u8> {
        let ifd = u32_at(4)? as usize;
        let n = u16_at(ifd)? as usize;
        for k in 0..n {
            let e = ifd.checked_add(2)?.checked_add(k.checked_mul(12)?)?;
            if u16_at(e)? != 0x0112 {
                continue;
            }
            // SHORT or LONG, one value, stored in the entry itself
            let v = match (u16_at(e + 2)?, u32_at(e + 4)?) {
                (3, 1) => u16_at(e + 8)?,
                (4, 1) => u32_at(e + 8)?,
                _ => return None,
            };
            return (1..=8).contains(&v).then_some(v as u8);
        }
        None
    };
    read().unwrap_or(1)
}

/// The image turned as its EXIF orientation asks (orientations 5..=8 swap the sides).
pub fn apply_orientation(img: RgbImage, orientation: u8) -> RgbImage {
    if !(2..=8).contains(&orientation) {
        return img;
    }
    let (w, h) = (img.width, img.height);
    let swap = orientation >= 5;
    let (ow, oh) = if swap { (h, w) } else { (w, h) };
    let mut out = vec![0u8; ow * oh * 3];
    for y in 0..oh {
        for x in 0..ow {
            // the source pixel shown at (x, y)
            let (sx, sy) = match orientation {
                2 => (mirror(x, w), y),
                3 => (mirror(x, w), mirror(y, h)),
                4 => (x, mirror(y, h)),
                5 => (y, x),
                6 => (y, mirror(x, h)),
                7 => (mirror(y, w), mirror(x, h)),
                _ => (mirror(y, w), x),
            };
            let (s, d) = ((sy * w + sx) * 3, (y * ow + x) * 3);
            out[d..d + 3].copy_from_slice(&img.data[s..s + 3]);
        }
    }
    RgbImage {
        width: ow,
        height: oh,
        data: out,
    }
}

/// The EXIF block of a JPEG file: the payload of its first APP1 `Exif` segment before the
/// first scan.
fn jpeg_exif(bytes: &[u8]) -> Option<&[u8]> {
    let mut pos = 2usize;
    loop {
        // a marker: 0xFF (fill bytes repeat it), then the code
        if *bytes.get(pos)? != 0xFF {
            return None;
        }
        while *bytes.get(pos + 1)? == 0xFF {
            pos += 1;
        }
        let code = bytes[pos + 1];
        pos += 2;
        match code {
            0xD8 | 0x01 | 0xD0..=0xD7 => continue,
            0xD9 | 0xDA => return None,
            _ => {}
        }
        let len = usize::from(u16::from_be_bytes([*bytes.get(pos)?, *bytes.get(pos + 1)?]));
        let seg = bytes.get(pos + 2..pos.checked_add(len.max(2))?)?;
        if code == 0xE1 && seg.starts_with(b"Exif\0\0") {
            return Some(&seg[6..]);
        }
        pos += len.max(2);
    }
}

/// The payload of a PNG file's `eXIf` chunk.
fn png_exif(bytes: &[u8]) -> Option<&[u8]> {
    let mut pos = 8usize;
    loop {
        let head = bytes.get(pos..pos.checked_add(8)?)?;
        let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
        let body = pos + 8;
        match &head[4..8] {
            b"eXIf" => return bytes.get(body..body.checked_add(len)?),
            b"IEND" => return None,
            _ => pos = body.checked_add(len)?.checked_add(4)?,
        }
    }
}

/// What a file's header tells: its format, its size as shown (EXIF orientation applied) and
/// its frame count (1 for a still image).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Probe {
    pub format: ImageFormat,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
}

/// Reads a file's header without decoding pixels.
pub fn probe(bytes: &[u8]) -> Result<Probe, String> {
    let format = sniff(bytes)?;
    let named = |e: String| format!("{}: {e}", format.name());
    let (width, height, frames, orientation) = match format {
        ImageFormat::Jpeg => {
            let (w, h) = jpeg::probe(bytes).map_err(named)?;
            (w, h, 1, jpeg_exif(bytes).map_or(1, exif_orientation))
        }
        ImageFormat::Png => {
            let (w, h) = png::probe(bytes).map_err(named)?;
            (w, h, 1, png_exif(bytes).map_or(1, exif_orientation))
        }
        ImageFormat::Bmp => {
            let (w, h) = bmp::probe(bytes).map_err(named)?;
            (w, h, 1, 1)
        }
        ImageFormat::Gif => {
            let s = gif::scan(bytes).map_err(named)?;
            (s.width, s.height, s.delays_ms.len().max(1), 1)
        }
        ImageFormat::Webp => {
            let s = webp::scan(bytes).map_err(named)?;
            let n = s.delays_ms.len().max(1);
            // an animation is shown as stored; a still image is turned
            let o = if n > 1 {
                1
            } else {
                s.exif.as_deref().map_or(1, exif_orientation)
            };
            (s.width, s.height, n, o)
        }
        ImageFormat::Ppm => {
            let img = RgbImage::parse_ppm(bytes)?;
            (img.width, img.height, 1, 1)
        }
    };
    let (width, height) = if orientation >= 5 {
        (height, width)
    } else {
        (width, height)
    };
    Ok(Probe {
        format,
        width,
        height,
        frames,
    })
}

fn still_to_image(s: Still) -> Result<RgbImage, String> {
    let orientation = s.exif.as_deref().map_or(1, exif_orientation);
    let img = RgbImage::from_raw(s.width, s.height, s.rgb)?;
    Ok(apply_orientation(img, orientation))
}

/// Decodes an image file (frame 0 of an animation) to RGB8, turned as its EXIF orientation
/// asks: what the reference loader hands to the image processor.
pub fn decode_image(bytes: &[u8], limits: &DecodeLimits) -> Result<RgbImage, String> {
    let format = sniff(bytes)?;
    let named = |e: String| format!("{}: {e}", format.name());
    match format {
        ImageFormat::Jpeg => jpeg::decode(bytes, limits).and_then(still_to_image),
        ImageFormat::Png => png::decode(bytes, limits).and_then(still_to_image),
        ImageFormat::Bmp => bmp::decode(bytes, limits).and_then(still_to_image),
        ImageFormat::Gif => gif::decode(bytes, limits).and_then(still_to_image),
        ImageFormat::Webp => webp::decode(bytes, limits).and_then(still_to_image),
        ImageFormat::Ppm => RgbImage::parse_ppm(bytes).and_then(|img| {
            limits.check_canvas(img.width, img.height)?;
            Ok(img)
        }),
    }
    .map_err(named)
}

/// Canvas size and the display time of every frame of an animation file (GIF, WebP), without
/// decoding pixels. A still image of those formats is one frame.
pub fn scan_frames(bytes: &[u8]) -> Result<Scan, String> {
    let format = sniff(bytes)?;
    match format {
        ImageFormat::Gif => gif::scan(bytes),
        ImageFormat::Webp => webp::scan(bytes),
        other => Err(format!(
            "a {} file holds no animation (animated GIF and WebP are taken)",
            other.name()
        )),
    }
    .map_err(|e| format!("{}: {e}", format.name()))
}

/// Takes one decoded frame: `(width, height, rgb)`.
pub type FrameSink<'a> = dyn FnMut(usize, usize, &[u8]) -> Result<(), String> + 'a;

/// Decodes the frames `wanted` (ascending frame indices; an index may repeat) of an animation
/// file, each composited as it is shown, and hands them to `sink(width, height, rgb)` in the
/// order of `wanted`. Frames after the last wanted one are not decoded. The canvas stays inside
/// the decoder: a caller that keeps a smaller copy of each frame never holds them all at full
/// size.
pub fn decode_frames(
    bytes: &[u8],
    limits: &DecodeLimits,
    wanted: &[usize],
    sink: &mut FrameSink<'_>,
) -> Result<(), String> {
    let format = sniff(bytes)?;
    let named = |e: String| format!("{}: {e}", format.name());
    if wanted.windows(2).any(|w| w[0] > w[1]) {
        return Err("frame indices must ascend".into());
    }
    let Some(&last) = wanted.last() else {
        return Ok(());
    };
    // the size is needed inside the callback, before the decoder returns it
    let scan = scan_frames(bytes)?;
    let (width, height) = (scan.width, scan.height);
    let mut next = 0usize;
    let mut failed: Option<String> = None;
    let mut keep = |index: usize, _delay: u32, rgb: &[u8]| -> bool {
        while next < wanted.len() && wanted[next] == index {
            if rgb.len() != width * height * 3 {
                failed = Some("a frame does not have the canvas size".into());
                return false;
            }
            if let Err(e) = sink(width, height, rgb) {
                failed = Some(e);
                return false;
            }
            next += 1;
        }
        index < last
    };
    match format {
        ImageFormat::Gif => gif::decode_frames(bytes, limits, &mut keep),
        ImageFormat::Webp => webp::decode_frames(bytes, limits, &mut keep),
        other => Err(format!(
            "a {} file holds no animation (animated GIF and WebP are taken)",
            other.name()
        )),
    }
    .map_err(named)?;
    if let Some(e) = failed {
        return Err(e);
    }
    if next != wanted.len() {
        return Err(named(format!(
            "the file holds fewer frames than asked for (frame {} is missing)",
            wanted[next]
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ppm(w: usize, h: usize, data: &[u8]) -> Vec<u8> {
        let mut f = format!("P6\n{w} {h}\n255\n").into_bytes();
        f.extend_from_slice(data);
        f
    }

    #[test]
    fn sniff_goes_by_signature_and_names_what_it_refuses() {
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0]).unwrap(), ImageFormat::Jpeg);
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\n....").unwrap(), ImageFormat::Png);
        assert_eq!(sniff(b"GIF89a....").unwrap(), ImageFormat::Gif);
        assert_eq!(sniff(b"RIFF\x10\0\0\0WEBPVP8 ").unwrap(), ImageFormat::Webp);
        assert_eq!(sniff(b"BM......").unwrap(), ImageFormat::Bmp);
        assert_eq!(sniff(b"P6\n1 1\n255\n\0\0\0").unwrap(), ImageFormat::Ppm);
        assert!(sniff(b"\0\0\0\x20ftypisom....")
            .unwrap_err()
            .contains("video file"));
        assert!(sniff(&[0x1A, 0x45, 0xDF, 0xA3, 0, 0])
            .unwrap_err()
            .contains("video file"));
        assert!(sniff(b"RIFF\x10\0\0\0AVI LIST")
            .unwrap_err()
            .contains("video file"));
        assert!(sniff(b"\0\0\0\x20ftypavif....")
            .unwrap_err()
            .contains("AVIF"));
        assert!(sniff(b"II*\0\x08\0\0\0").unwrap_err().contains("TIFF"));
        assert!(sniff(b"").unwrap_err().contains("empty"));
        assert!(sniff(b"<svg xmlns=").unwrap_err().contains("JPEG, PNG"));
    }

    /// A minimal EXIF block with one IFD entry: the orientation.
    fn exif(le: bool, value: u16) -> Vec<u8> {
        let mut t = Vec::new();
        let p16 =
            |t: &mut Vec<u8>, v: u16| t.extend(if le { v.to_le_bytes() } else { v.to_be_bytes() });
        let p32 =
            |t: &mut Vec<u8>, v: u32| t.extend(if le { v.to_le_bytes() } else { v.to_be_bytes() });
        t.extend(if le { *b"II" } else { *b"MM" });
        p16(&mut t, 42);
        p32(&mut t, 8);
        p16(&mut t, 1);
        p16(&mut t, 0x0112);
        p16(&mut t, 3);
        p32(&mut t, 1);
        p16(&mut t, value);
        p16(&mut t, 0);
        p32(&mut t, 0);
        t
    }

    #[test]
    fn exif_orientation_reads_both_byte_orders_and_survives_garbage() {
        for o in 1..=8u16 {
            assert_eq!(exif_orientation(&exif(true, o)), o as u8);
            assert_eq!(exif_orientation(&exif(false, o)), o as u8);
        }
        assert_eq!(exif_orientation(&exif(true, 0)), 1);
        assert_eq!(exif_orientation(&exif(true, 9)), 1);
        assert_eq!(exif_orientation(b""), 1);
        assert_eq!(exif_orientation(b"II*\0\xff\xff\xff\xff"), 1);
        let whole = exif(false, 6);
        for cut in 0..whole.len() {
            let _ = exif_orientation(&whole[..cut]);
        }
    }

    /// Reference results for a 3x2 image with pixel values 1..=6 (red channel): what the
    /// reference image library's EXIF transpose gives for each orientation, row-major.
    #[test]
    fn orientations_match_the_reference_transpose() {
        let img = || RgbImage {
            width: 3,
            height: 2,
            data: (1..=6u8).flat_map(|v| [v, 0, 0]).collect(),
        };
        let red = |i: RgbImage| -> (usize, usize, Vec<u8>) {
            (i.width, i.height, i.data.chunks(3).map(|p| p[0]).collect())
        };
        let want: [(usize, usize, [u8; 6]); 8] = [
            (3, 2, [1, 2, 3, 4, 5, 6]),
            (3, 2, [3, 2, 1, 6, 5, 4]),
            (3, 2, [6, 5, 4, 3, 2, 1]),
            (3, 2, [4, 5, 6, 1, 2, 3]),
            (2, 3, [1, 4, 2, 5, 3, 6]),
            (2, 3, [4, 1, 5, 2, 6, 3]),
            (2, 3, [6, 3, 5, 2, 4, 1]),
            (2, 3, [3, 6, 2, 5, 1, 4]),
        ];
        for (k, (w, h, px)) in want.iter().enumerate() {
            let (gw, gh, got) = red(apply_orientation(img(), k as u8 + 1));
            assert_eq!(
                (gw, gh, got.as_slice()),
                (*w, *h, px.as_slice()),
                "orientation {}",
                k + 1
            );
        }
        // values outside 1..=8 leave the image alone
        assert_eq!(apply_orientation(img(), 0), img());
        assert_eq!(apply_orientation(img(), 9), img());
    }

    #[test]
    fn ppm_goes_through_probe_and_decode_under_the_canvas_limit() {
        let f = ppm(2, 1, &[1, 2, 3, 4, 5, 6]);
        let p = probe(&f).unwrap();
        assert_eq!(
            (p.format, p.width, p.height, p.frames),
            (ImageFormat::Ppm, 2, 1, 1)
        );
        let lim = DecodeLimits { max_pixels: 2 };
        assert_eq!(decode_image(&f, &lim).unwrap().data, vec![1, 2, 3, 4, 5, 6]);
        let small = DecodeLimits { max_pixels: 1 };
        assert!(decode_image(&f, &small)
            .unwrap_err()
            .contains("larger than the limit"));
        assert!(scan_frames(&f).unwrap_err().contains("no animation"));
    }
}
