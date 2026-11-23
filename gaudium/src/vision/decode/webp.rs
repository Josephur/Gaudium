//! WebP files: the RIFF container with its simple (`VP8 ` / `VP8L`) and extended (`VP8X`)
//! layouts, alpha planes (`ALPH`), animations (`ANIM` / `ANMF`) and the `EXIF` chunk. The
//! bitstreams are decoded by `vp8` (lossy) and `vp8l` (lossless).
//!
//! The output is what the reference image library shows after conversion to RGB: the RGB
//! bytes of its RGBA canvas, alpha dropped. A fully transparent pixel therefore keeps the
//! colour stored in the file, and canvas areas no frame has drawn are black.
//!
//! Container rules, all taken from what the reference library accepts (differential tests):
//! - the RIFF size must fit in the file; bytes after it are ignored; the first chunk must be
//!   `VP8 `, `VP8L` or `VP8X`; every chunk, with its padding byte, must lie inside the RIFF
//!   chunk;
//! - the `VP8X` chunk is 10 bytes and its reserved flag bits are zero;
//! - a still image is one image chunk of the size of the canvas, in an extended file
//!   optionally preceded by `ALPH`; the alpha plane of a still is used only when the `VP8X`
//!   header announces alpha;
//! - an animation is `ANIM` followed by `ANMF` frames; a frame without any chunk of its own
//!   is dropped, a frame takes its size from its bitstream (not from the `ANMF` header) and
//!   must lie inside the canvas;
//! - `EXIF` is returned only when the `VP8X` header announces it; `ICCP`, `XMP ` and unknown
//!   chunks are skipped (no colour management).
//!
//! Animation frames are composited as the reference library does: the canvas starts
//! transparent black (the background colour of the file is not used), a frame is drawn over
//! the canvas left by the previous frame after its disposal, and a frame that asks for
//! blending is blended in non-premultiplied arithmetic (`blend_pixel`). Three details of that
//! library are reproduced because they change pixels: a frame that covers the canvas and
//! cannot show what is under it restarts from a cleared canvas; after a frame disposed to the
//! background, pixels inside its rectangle are not blended (they are copied as decoded); and
//! blending a partly transparent pixel onto a transparent one rescales its colour.
//!
//! Refuses: files the reference library refuses (truncated chunks, frames outside the canvas,
//! an animation without frames, ...), canvases over the limit, and bitstream errors.

use super::types::{DecodeLimits, Scan, Still};
use super::{vp8, vp8l};

/// Largest chunk payload the reference library accepts.
const MAX_CHUNK: u64 = 0xffff_fff6;
/// The canvas, and every frame, must have fewer pixels than this.
const MAX_AREA: u64 = 1 << 32;

const FLAG_ANIMATION: u8 = 0x02;
const FLAG_EXIF: u8 = 0x08;
const FLAG_ALPHA: u8 = 0x10;
/// Bits of the `VP8X` flags byte that have a meaning.
const KNOWN_FLAGS: u8 = 0x3e;

const NOT_WEBP: &str = "not a WebP file";
const TRUNCATED: &str = "WebP file is truncated";
const BAD_CHUNK: &str = "WebP chunk size is invalid";

/// The image chunk of a frame.
#[derive(Clone, Copy)]
struct ImageChunk {
    /// Offset of the chunk header.
    at: usize,
    /// The payload with its padding byte (the reference decoder reads the bitstream up to it).
    start: usize,
    end: usize,
    lossless: bool,
}

/// One frame of the file (a still image is one frame over the whole canvas).
#[derive(Clone, Copy, Default)]
struct Frame {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    duration: u32,
    dispose_to_background: bool,
    blend: bool,
    /// What the file says about transparency: an `ALPH` chunk, or the alpha bit of a lossless
    /// bitstream.
    has_alpha: bool,
    /// The `ALPH` chunk: offset of its header, start and length of its payload.
    alpha: Option<(usize, usize, usize)>,
    image: Option<ImageChunk>,
    /// True once the frame holds an alpha or an image chunk.
    present: bool,
}

/// What the container says, without any pixel decoded.
struct Layout<'a> {
    /// The file up to the end of the RIFF chunk.
    data: &'a [u8],
    width: usize,
    height: usize,
    frames: Vec<Frame>,
    /// Payload of the `EXIF` chunk: start and length.
    exif: Option<(usize, usize)>,
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    extended: bool,
    flags: u8,
    canvas: (usize, usize),
    frames: Vec<Frame>,
    exif: Option<(usize, usize)>,
}

fn le16(data: &[u8], at: usize) -> usize {
    data[at] as usize | (data[at + 1] as usize) << 8
}

fn le24(data: &[u8], at: usize) -> usize {
    le16(data, at) | (data[at + 2] as usize) << 16
}

fn le32(data: &[u8], at: usize) -> u64 {
    (le24(data, at) | (data[at + 3] as usize) << 24) as u64
}

impl<'a> Parser<'a> {
    fn left(&self) -> usize {
        self.data.len() - self.pos
    }

    /// The chunk header at the cursor (at least 8 bytes are left): its tag, the offset of its
    /// payload, the payload size and the size with padding. The whole padded payload is
    /// inside the file.
    fn chunk(&self) -> Result<([u8; 4], usize, usize, usize), String> {
        let at = self.pos;
        let tag = [
            self.data[at],
            self.data[at + 1],
            self.data[at + 2],
            self.data[at + 3],
        ];
        let size = le32(self.data, at + 4);
        if size > MAX_CHUNK {
            return Err(BAD_CHUNK.into());
        }
        let padded = size + (size & 1);
        if padded > (self.data.len() - at - 8) as u64 {
            return Err(TRUNCATED.into());
        }
        Ok((tag, at + 8, size as usize, padded as usize))
    }

    /// Collects the chunks of one frame at the cursor: at most one `ALPH` and one image chunk.
    /// Stops in front of the first chunk that does not belong to the frame.
    fn frame_chunks(&mut self, frame: &mut Frame) -> Result<(), String> {
        if self.left() < 8 {
            return Err(TRUNCATED.into());
        }
        loop {
            let at = self.pos;
            let (tag, start, size, padded) = self.chunk()?;
            match &tag {
                b"ALPH" if frame.alpha.is_none() => {
                    frame.alpha = Some((at, start, size));
                    frame.has_alpha = true;
                }
                b"VP8L" if frame.alpha.is_some() => {
                    return Err("WebP lossless image has an alpha chunk".into());
                }
                b"VP8 " | b"VP8L" if frame.image.is_none() => {
                    let payload = &self.data[start..start + padded];
                    let lossless = &tag == b"VP8L";
                    let size = if lossless {
                        vp8l::read_header(payload)
                    } else {
                        vp8::read_header(payload, size).map(|(w, h)| (w, h, false))
                    };
                    let (width, height, alpha) = size.ok_or("WebP image header is invalid")?;
                    frame.image = Some(ImageChunk {
                        at,
                        start,
                        end: start + padded,
                        lossless,
                    });
                    frame.width = width;
                    frame.height = height;
                    frame.has_alpha |= alpha;
                }
                _ => return Ok(()),
            }
            frame.present = true;
            self.pos = start + padded;
            if self.left() == 0 {
                return Ok(());
            }
            if self.left() < 8 {
                return Err(TRUNCATED.into());
            }
        }
    }

    /// A still image at the cursor.
    fn still(&mut self) -> Result<(), String> {
        if !self.frames.is_empty() {
            return Err("WebP file has more than one image".into());
        }
        let mut frame = Frame {
            blend: true,
            ..Frame::default()
        };
        self.frame_chunks(&mut frame)?;
        // the alpha plane of a still counts only when the file header announces alpha
        if self.flags & FLAG_ALPHA == 0 && frame.alpha.is_some() {
            frame.alpha = None;
            frame.has_alpha = false;
        }
        if !self.extended {
            self.canvas = (frame.width, frame.height);
        }
        self.frames.push(frame);
        Ok(())
    }

    /// An `ANMF` chunk whose payload of `padded` bytes starts at `start`.
    fn animation_frame(
        &mut self,
        start: usize,
        padded: usize,
        animated: bool,
    ) -> Result<(), String> {
        if padded < 16 {
            return Err(BAD_CHUNK.into());
        }
        let d = self.data;
        let bits = d[start + 15];
        let mut frame = Frame {
            x: 2 * le24(d, start),
            y: 2 * le24(d, start + 3),
            width: 1 + le24(d, start + 6),
            height: 1 + le24(d, start + 9),
            duration: le24(d, start + 12) as u32,
            dispose_to_background: bits & 1 != 0,
            blend: bits & 2 == 0,
            ..Frame::default()
        };
        if frame.width as u64 * frame.height as u64 >= MAX_AREA {
            return Err("WebP frame is too large".into());
        }
        self.pos = start + 16;
        self.frame_chunks(&mut frame)?;
        if self.pos > start + padded {
            return Err("WebP frame data runs past its chunk".into());
        }
        // a frame that holds neither alpha nor image is dropped
        if animated && frame.present {
            if self.frames.last().is_some_and(|f| f.image.is_none()) {
                return Err("WebP frame has no image".into());
            }
            self.frames.push(frame);
        }
        Ok(())
    }

    /// The chunks of an extended file, from the `VP8X` chunk at the cursor.
    fn extended_file(&mut self) -> Result<(), String> {
        self.extended = true;
        let d = self.data;
        // the header chunk holds exactly its 10 bytes: flags, 3 reserved bytes, canvas size
        if le32(d, self.pos + 4) != 10 {
            return Err(BAD_CHUNK.into());
        }
        self.pos += 8;
        if self.left() < 10 {
            return Err(TRUNCATED.into());
        }
        self.flags = d[self.pos];
        self.canvas = (1 + le24(d, self.pos + 4), 1 + le24(d, self.pos + 7));
        if self.canvas.0 as u64 * self.canvas.1 as u64 >= MAX_AREA {
            return Err("WebP canvas is too large".into());
        }
        self.pos += 10;
        if self.left() < 8 {
            return Err(TRUNCATED.into());
        }
        let animated = self.flags & FLAG_ANIMATION != 0;
        let mut in_animation = false;
        loop {
            let (tag, start, size, padded) = self.chunk()?;
            match &tag {
                b"VP8X" => return Err("WebP file has two headers".into()),
                b"ALPH" | b"VP8 " | b"VP8L" => {
                    if in_animation || animated {
                        return Err("WebP animation has an image outside its frames".into());
                    }
                    self.still()?;
                }
                b"ANIM" => {
                    if padded < 6 {
                        return Err(BAD_CHUNK.into());
                    }
                    in_animation = true;
                    self.pos = start + padded;
                }
                b"ANMF" => {
                    if !in_animation {
                        return Err("WebP animation frame comes before the animation header".into());
                    }
                    self.animation_frame(start, padded, animated)?;
                }
                b"EXIF" => {
                    if self.flags & FLAG_EXIF != 0 && self.exif.is_none() {
                        self.exif = Some((start, size));
                    }
                    self.pos = start + padded;
                }
                _ => self.pos = start + padded,
            }
            if self.left() == 0 {
                return Ok(());
            }
            if self.left() < 8 {
                return Err(TRUNCATED.into());
            }
        }
    }
}

/// Reads the container: canvas, frames and the EXIF chunk.
fn parse(bytes: &[u8]) -> Result<Layout<'_>, String> {
    if bytes.len() < 20 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return Err(NOT_WEBP.into());
    }
    let riff = le32(bytes, 4);
    if !(8..=MAX_CHUNK).contains(&riff) {
        return Err(BAD_CHUNK.into());
    }
    if (bytes.len() as u64) < riff + 8 {
        return Err(TRUNCATED.into());
    }
    let data = &bytes[..riff as usize + 8];
    if data.len() < 20 {
        // not even one chunk header after the file header
        return Err(TRUNCATED.into());
    }
    let mut parser = Parser {
        data,
        pos: 12,
        extended: false,
        flags: 0,
        canvas: (0, 0),
        frames: Vec::new(),
        exif: None,
    };
    match &data[12..16] {
        b"VP8 " | b"VP8L" => parser.still()?,
        b"VP8X" => parser.extended_file()?,
        _ => return Err(NOT_WEBP.into()),
    }

    let (width, height) = parser.canvas;
    let animated = parser.flags & FLAG_ANIMATION != 0;
    if parser.frames.is_empty() {
        return Err("WebP file has no image".into());
    }
    if parser.flags & !KNOWN_FLAGS != 0 {
        return Err("WebP header flags are invalid".into());
    }
    for frame in &parser.frames {
        let image = frame.image.ok_or("WebP frame has no image")?;
        if frame.alpha.is_some_and(|(at, _, _)| at > image.at) {
            return Err("WebP alpha chunk comes after its image".into());
        }
        let inside = if animated {
            frame.x + frame.width <= width && frame.y + frame.height <= height
        } else {
            frame.x == 0 && frame.y == 0 && frame.width == width && frame.height == height
        };
        if !inside {
            return Err("WebP frame does not fit the canvas".into());
        }
    }
    Ok(Layout {
        data,
        width,
        height,
        frames: parser.frames,
        exif: parser.exif,
    })
}

impl Layout<'_> {
    /// The EXIF block from its byte-order mark: an "Exif\0\0" prefix some writers add is cut.
    fn exif_block(&self) -> Option<Vec<u8>> {
        let (start, len) = self.exif?;
        let block = &self.data[start..start + len];
        let block = block.strip_prefix(b"Exif\0\0").unwrap_or(block);
        if block.is_empty() {
            None
        } else {
            Some(block.to_vec())
        }
    }
}

/// The pixels of one frame.
enum Pixels {
    Lossy {
        frame: vp8::Frame,
        alpha: Option<Vec<u8>>,
    },
    Lossless(Vec<u32>),
}

/// Decodes the alpha plane of a lossy frame from the payload of its `ALPH` chunk: one header
/// byte (compression, prediction filter, pre-processing), then the plane raw or as a lossless
/// stream, then the prediction filter undone.
fn decode_alpha(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    const BAD: &str = "WebP alpha plane is invalid";
    if data.len() <= 1 {
        return Err(BAD.into());
    }
    let (method, filter, pre_processing, reserved) = (
        data[0] & 3,
        (data[0] >> 2) & 3,
        (data[0] >> 4) & 3,
        data[0] >> 6,
    );
    if method > 1 || pre_processing > 1 || reserved != 0 {
        return Err(BAD.into());
    }
    let mut plane = if method == 0 {
        match data.get(1..1 + width * height) {
            Some(raw) => raw.to_vec(),
            None => return Err(BAD.into()),
        }
    } else {
        vp8l::decode_alpha_plane(&data[1..], width, height)?
    };
    if filter == 0 {
        return Ok(plane);
    }
    // the first row is predicted from the left (the first pixel from 0); below it the first
    // pixel of a row is predicted from the pixel above
    for x in 1..width {
        plane[x] = plane[x].wrapping_add(plane[x - 1]);
    }
    for y in 1..height {
        let (above, row) = plane[(y - 1) * width..(y + 1) * width].split_at_mut(width);
        match filter {
            1 => {
                row[0] = row[0].wrapping_add(above[0]);
                for x in 1..width {
                    row[x] = row[x].wrapping_add(row[x - 1]);
                }
            }
            2 => {
                for (p, &a) in row.iter_mut().zip(above.iter()) {
                    *p = p.wrapping_add(a);
                }
            }
            _ => {
                row[0] = row[0].wrapping_add(above[0]);
                for x in 1..width {
                    let gradient = row[x - 1] as i32 + above[x] as i32 - above[x - 1] as i32;
                    row[x] = row[x].wrapping_add(gradient.clamp(0, 255) as u8);
                }
            }
        }
    }
    Ok(plane)
}

/// Decodes the bitstreams of one frame.
fn decode_frame(data: &[u8], frame: &Frame, limits: &DecodeLimits) -> Result<Pixels, String> {
    limits.check_canvas(frame.width, frame.height)?;
    let image = frame.image.ok_or("WebP frame has no image")?;
    let payload = &data[image.start..image.end];
    if image.lossless {
        let (width, height, pixels) = vp8l::decode(payload, limits)?;
        if (width, height) != (frame.width, frame.height) {
            return Err("WebP image header is invalid".into());
        }
        return Ok(Pixels::Lossless(pixels));
    }
    let decoded = vp8::decode(payload, limits)?;
    if (decoded.width, decoded.height) != (frame.width, frame.height) {
        return Err("WebP image header is invalid".into());
    }
    let alpha = match frame.alpha {
        Some((_, start, len)) => Some(decode_alpha(
            &data[start..start + len],
            frame.width,
            frame.height,
        )?),
        None => None,
    };
    Ok(Pixels::Lossy {
        frame: decoded,
        alpha,
    })
}

impl Pixels {
    /// Writes the frame of `width` pixels per row to `out`: rows `stride` bytes apart, `step`
    /// bytes per pixel, R G B and, when `step` is 4, alpha.
    fn write(&self, width: usize, out: &mut [u8], stride: usize, step: usize) {
        match self {
            Pixels::Lossy { frame, alpha } => {
                frame.write_rgb(out, stride, step);
                if step == 4 {
                    for y in 0..frame.height {
                        let row = &mut out[y * stride..y * stride + 4 * width];
                        match alpha {
                            Some(plane) => {
                                let (pixels, _) = row.as_chunks_mut::<4>();
                                for (px, &a) in pixels.iter_mut().zip(&plane[y * width..]) {
                                    px[3] = a;
                                }
                            }
                            None => row
                                .as_chunks_mut::<4>()
                                .0
                                .iter_mut()
                                .for_each(|px| px[3] = 255),
                        }
                    }
                }
            }
            Pixels::Lossless(pixels) => {
                for (y, src) in pixels.chunks_exact(width).enumerate() {
                    let row = &mut out[y * stride..y * stride + step * width];
                    for (px, &p) in row.chunks_exact_mut(step).zip(src) {
                        px[0] = (p >> 16) as u8;
                        px[1] = (p >> 8) as u8;
                        px[2] = p as u8;
                        if step == 4 {
                            px[3] = (p >> 24) as u8;
                        }
                    }
                }
            }
        }
    }
}

/// Number of bytes of a canvas with `channels` bytes per pixel.
fn canvas_bytes(width: usize, height: usize, channels: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| "WebP canvas is too large".to_string())
}

/// Blends the RGBA pixel `src` over `dst` (neither premultiplied) into `dst`, with the
/// rounding of the reference library: the weight of `dst` is `dst_a * (256 - src_a) >> 8`,
/// and the weighted sum is divided by the new alpha as a multiplication by `2^24 / alpha`.
#[inline]
fn blend_pixel(src: &[u8; 4], dst: &mut [u8; 4]) {
    let src_a = src[3] as u32;
    if src_a == 255 {
        *dst = *src;
        return;
    }
    if src_a == 0 {
        return;
    }
    let dst_a = (dst[3] as u32 * (256 - src_a)) >> 8;
    let alpha = src_a + dst_a;
    let scale = (1u32 << 24) / alpha;
    for c in 0..3 {
        dst[c] = (((src[c] as u32 * src_a + dst[c] as u32 * dst_a) * scale) >> 24) as u8;
    }
    dst[3] = alpha as u8;
}

/// Canvas size, every frame's delay (one entry for a still image) and the EXIF chunk, without
/// decoding pixels.
pub fn scan(bytes: &[u8]) -> Result<Scan, String> {
    let layout = parse(bytes)?;
    Ok(Scan {
        width: layout.width,
        height: layout.height,
        delays_ms: layout.frames.iter().map(|f| f.duration).collect(),
        exif: layout.exif_block(),
    })
}

/// Frame 0 as RGB8 (for a still image: the image).
pub fn decode(bytes: &[u8], limits: &DecodeLimits) -> Result<Still, String> {
    let layout = parse(bytes)?;
    let (width, height) = (layout.width, layout.height);
    limits.check_canvas(width, height)?;
    let rgb = first_frame(&layout, limits)?;
    Ok(Still {
        width,
        height,
        rgb,
        exif: layout.exif_block(),
    })
}

/// The canvas after the first frame, as RGB. The first frame is drawn on a cleared canvas
/// without blending, so no alpha is needed.
fn first_frame(layout: &Layout, limits: &DecodeLimits) -> Result<Vec<u8>, String> {
    let (width, height) = (layout.width, layout.height);
    let frame = &layout.frames[0];
    let pixels = decode_frame(layout.data, frame, limits)?;
    let mut rgb = vec![0u8; canvas_bytes(width, height, 3)?];
    let at = (frame.y * width + frame.x) * 3;
    pixels.write(frame.width, &mut rgb[at..], width * 3, 3);
    Ok(rgb)
}

/// Decodes the frames in order, composited as the reference library shows each one, and hands
/// every frame to `keep(frame_index, delay_ms, rgb_canvas)` (`rgb_canvas`: width*height*3
/// bytes, valid only during the call). Decoding stops early when `keep` returns false. Returns
/// the canvas (width, height). A still image is one frame with delay 0.
pub fn decode_frames(
    bytes: &[u8],
    limits: &DecodeLimits,
    keep: &mut dyn FnMut(usize, u32, &[u8]) -> bool,
) -> Result<(usize, usize), String> {
    let layout = parse(bytes)?;
    let (width, height) = (layout.width, layout.height);
    limits.check_canvas(width, height)?;
    if layout.frames.len() == 1 {
        let rgb = first_frame(&layout, limits)?;
        keep(0, layout.frames[0].duration, &rgb);
        return Ok((width, height));
    }

    // The RGBA canvas being shown and its RGB copy for `keep`. Both are updated in place, one
    // frame rectangle at a time: the disposal of the previous frame, then the new frame.
    let mut canvas = vec![0u8; canvas_bytes(width, height, 4)?];
    let mut rgb = vec![0u8; canvas_bytes(width, height, 3)?];
    // the frame being drawn, as RGBA
    let mut drawn: Vec<u8> = Vec::new();
    let covers = |f: &Frame| f.width == width && f.height == height;
    let mut previous: Option<(&Frame, bool)> = None;
    for (index, frame) in layout.frames.iter().enumerate() {
        // A frame starts from a cleared canvas when nothing of the previous canvas can show:
        // it covers the canvas and is opaque or not blended, or the previous canvas is empty
        // after its disposal. Such a frame is not blended. (In both cases the canvas updated
        // in place is what a cleared canvas would be.)
        let key = match previous {
            None => true,
            Some((prev, prev_key)) => {
                ((!frame.has_alpha || !frame.blend) && covers(frame))
                    || (prev.dispose_to_background && (covers(prev) || prev_key))
            }
        };
        // the rectangle of the previous frame that its disposal cleared
        let mut cleared = None;
        if let Some((prev, _)) = previous {
            if prev.dispose_to_background {
                for y in prev.y..prev.y + prev.height {
                    let at = y * width + prev.x;
                    canvas[at * 4..(at + prev.width) * 4].fill(0);
                    rgb[at * 3..(at + prev.width) * 3].fill(0);
                }
                cleared = Some(prev);
            }
        }
        let pixels = decode_frame(layout.data, frame, limits)?;
        drawn.clear();
        drawn.resize(canvas_bytes(frame.width, frame.height, 4)?, 0);
        pixels.write(frame.width, &mut drawn, frame.width * 4, 4);
        drop(pixels);
        let blend = frame.blend && !key;
        for (row, src) in drawn.chunks_exact(frame.width * 4).enumerate() {
            let y = frame.y + row;
            let at = y * width + frame.x;
            let dst = &mut canvas[at * 4..(at + frame.width) * 4];
            if blend {
                // pixels inside the rectangle a disposal has just cleared are copied as decoded
                let unblended = match cleared {
                    Some(prev) if y >= prev.y && y < prev.y + prev.height => {
                        prev.x..prev.x + prev.width
                    }
                    _ => 0..0,
                };
                let pairs = src
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<4>().0);
                for (i, (s, d)) in pairs.enumerate() {
                    if unblended.contains(&(frame.x + i)) {
                        *d = *s;
                    } else {
                        blend_pixel(s, d);
                    }
                }
            } else {
                dst.copy_from_slice(src);
            }
            let out = &mut rgb[at * 3..(at + frame.width) * 3];
            for (o, d) in out
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(dst.as_chunks::<4>().0)
            {
                *o = [d[0], d[1], d[2]];
            }
        }
        if !keep(index, frame.duration, &rgb) {
            break;
        }
        previous = Some((frame, key));
    }
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        name: &'static str,
        data: &'static [u8],
        width: usize,
        height: usize,
        /// Delay and FNV-1a 64 hash of the reference RGB bytes of every frame.
        frames: &'static [(u32, u64)],
    }

    macro_rules! fixture {
        ($name:literal, $width:expr, $height:expr, [$(($delay:expr, $hash:expr)),+]) => {
            Fixture {
                name: $name,
                data: include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/data/vision/webp/",
                    $name
                )),
                width: $width,
                height: $height,
                frames: &[$(($delay, $hash)),+],
            }
        };
    }

    /// The tracked fixtures with what the reference image library returns for them; the table
    /// is printed by `tests/data/vision/webp/make_fixtures.py`.
    #[rustfmt::skip]
    const FIXTURES: &[Fixture] = &[
        fixture!("anim_alpha_ll_f0.webp", 12, 9, [(40, 0x1c73c25766d79d49), (60, 0x41520c372091b964)]),
        fixture!("anim_alpha_ll_f1.webp", 12, 9, [(40, 0x4acbd547314a2431), (60, 0xffda64e9b0147755)]),
        fixture!("anim_alpha_ll_f2.webp", 12, 9, [(40, 0x306f9cc8fb4b8a4b), (60, 0xa4417a182c4d12ba)]),
        fixture!("anim_alpha_ll_f3.webp", 12, 9, [(40, 0xf7aa6bed47a062af), (60, 0x69ba6ea90ac5982c)]),
        fixture!("anim_alpha_raw_f0.webp", 12, 9, [(40, 0x1c73c25766d79d49), (60, 0x41520c372091b964)]),
        fixture!("anim_alpha_raw_f1.webp", 12, 9, [(40, 0x4acbd547314a2431), (60, 0xffda64e9b0147755)]),
        fixture!("anim_alpha_raw_f2.webp", 12, 9, [(40, 0x306f9cc8fb4b8a4b), (60, 0xa4417a182c4d12ba)]),
        fixture!("anim_alpha_raw_f3.webp", 12, 9, [(40, 0xf7aa6bed47a062af), (60, 0x69ba6ea90ac5982c)]),
        fixture!("anim_first_small.webp", 20, 14, [(77, 0xd53337fc82a2c848), (5, 0x42b4af8e2a0cf9d5)]),
        fixture!("anim_ll_20x15.webp", 20, 15, [(30, 0xbf39f63a11498cbf), (0, 0x2751a023f081da09), (100, 0x64b3dc4c5623e9d5), (65535, 0x2e991ab958a9b799)]),
        fixture!("anim_lossy_20x15.webp", 20, 15, [(30, 0xaab737e1fbaa9bcb), (0, 0x9e561ed8b28c70f5), (100, 0x546d2f67da0c5159), (65535, 0x1e9c51a72580d884)]),
        fixture!("anim_mux_ll.webp", 24, 18, [(10, 0xf16ccf1e99df40db), (20, 0x980a4084f7c9a16c), (30, 0x386b8846ecf090af), (40, 0x5acb41187f626ecf), (50, 0xdee7e917ef642655), (60, 0xf043b6379282f971)]),
        fixture!("anim_mux_mixed.webp", 24, 18, [(10, 0xb27016e636ddc793), (20, 0x7a5ea0705bc690a1), (30, 0x2c41ddd9cd468050), (40, 0x61ca27c3dcae7cab), (50, 0x2574849aee2e4714), (60, 0x7545117716444b39)]),
        fixture!("la_24x18.webp", 24, 18, [(0, 0x38555a708e43fd41)]),
        fixture!("ll_1x1.webp", 1, 1, [(0, 0x2d4432189addadc3)]),
        fixture!("ll_pal16_33x9.webp", 33, 9, [(0, 0xd5e9c2981acc7450)]),
        fixture!("ll_pal2_33x9.webp", 33, 9, [(0, 0xb0992b465515a2ba)]),
        fixture!("ll_pal40_33x9.webp", 33, 9, [(0, 0xc9148f8c2fbdb7a3)]),
        fixture!("ll_pal4_33x9.webp", 33, 9, [(0, 0x3d44c1069b1afb12)]),
        fixture!("ll_photo_40x30.webp", 40, 30, [(0, 0x3c286988ced301a8)]),
        fixture!("ll_photo_alpha_31x16.webp", 31, 16, [(0, 0x65b6ad386b1957f8)]),
        fixture!("ll_synth_a.webp", 2, 23, [(0, 0x45f83599480cc6ca)]),
        fixture!("ll_synth_b.webp", 8, 3, [(0, 0x6b32e363e5cc0002)]),
        fixture!("ll_synth_c.webp", 9, 2, [(0, 0x8ec6a3daf0a9e1fc)]),
        fixture!("ll_tiles_45x35.webp", 45, 35, [(0, 0x68c7b32e1845ea16)]),
        fixture!("lossy_1x1.webp", 1, 1, [(0, 0xd2db0e1869094082)]),
        fixture!("lossy_1x16.webp", 1, 16, [(0, 0x1f6b0d9c4c1a4df7)]),
        fixture!("lossy_2x3.webp", 2, 3, [(0, 0x70c05110b37b0bca)]),
        fixture!("lossy_9x1.webp", 9, 1, [(0, 0x86f25ea484838cd3)]),
        fixture!("lossy_exif_20x12.webp", 20, 12, [(0, 0x1782da33a7808062)]),
        fixture!("lossy_filter_deltas_48x40.webp", 48, 40, [(0, 0xf6e1826adbab8bf9)]),
        fixture!("lossy_q100_edges_16x16.webp", 16, 16, [(0, 0xb2184895a558964a)]),
        fixture!("lossy_q20_17x17.webp", 17, 17, [(0, 0x481864c7ef842293)]),
        fixture!("lossy_q75_33x47.webp", 33, 47, [(0, 0xa54ed9f2207477ce)]),
        fixture!("lossy_sharp7_50x50.webp", 50, 50, [(0, 0x8bbcba3ab3e6833d)]),
        fixture!("lossy_simple_filter_64x48.webp", 64, 48, [(0, 0x1595389ed27bc2b9)]),
        fixture!("lossy_synth_a.webp", 7, 35, [(0, 0x1cbb668561dce8cf)]),
        fixture!("lossy_synth_b.webp", 40, 8, [(0, 0xcdc5b9592573ecfa)]),
        fixture!("still_unannounced_24x18.webp", 24, 18, [(0, 0x38555a708e43fd41)]),
    ];

    const LIMITS: DecodeLimits = DecodeLimits {
        max_pixels: 1 << 24,
    };

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
        })
    }

    fn fixture(name: &str) -> &'static Fixture {
        FIXTURES.iter().find(|f| f.name == name).expect("fixture")
    }

    /// Delay and hash of every frame `decode_frames` hands out.
    fn frames_of(data: &[u8]) -> Result<Vec<(u32, u64)>, String> {
        let mut frames = Vec::new();
        decode_frames(data, &LIMITS, &mut |index, delay, rgb| {
            assert_eq!(index, frames.len());
            frames.push((delay, fnv1a(rgb)));
            true
        })?;
        Ok(frames)
    }

    fn chunk(tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = tag.to_vec();
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    fn riff(body: &[u8]) -> Vec<u8> {
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32 + 4).to_le_bytes());
        out.extend_from_slice(b"WEBP");
        out.extend_from_slice(body);
        out
    }

    fn header(flags: u8, width: usize, height: usize) -> Vec<u8> {
        let (w, h) = ((width - 1) as u32, (height - 1) as u32);
        let mut payload = vec![flags, 0, 0, 0];
        payload.extend_from_slice(&w.to_le_bytes()[..3]);
        payload.extend_from_slice(&h.to_le_bytes()[..3]);
        chunk(b"VP8X", &payload)
    }

    /// An `ANMF` chunk holding `image` (the chunks of a still) at (x, y).
    fn frame_chunk(x: usize, y: usize, width: usize, height: usize, image: &[u8]) -> Vec<u8> {
        let mut payload = Vec::new();
        for v in [x / 2, y / 2, width - 1, height - 1, 10] {
            payload.extend_from_slice(&(v as u32).to_le_bytes()[..3]);
        }
        payload.push(0);
        payload.extend_from_slice(image);
        chunk(b"ANMF", &payload)
    }

    #[test]
    fn fixtures_decode_to_the_reference_pixels() {
        for f in FIXTURES {
            let still = decode(f.data, &LIMITS).unwrap_or_else(|e| panic!("{}: {e}", f.name));
            assert_eq!(
                (still.width, still.height),
                (f.width, f.height),
                "{}",
                f.name
            );
            assert_eq!(still.rgb.len(), f.width * f.height * 3, "{}", f.name);
            assert_eq!(fnv1a(&still.rgb), f.frames[0].1, "{} frame 0", f.name);
            let frames = frames_of(f.data).unwrap_or_else(|e| panic!("{}: {e}", f.name));
            assert_eq!(frames, f.frames, "{}", f.name);
        }
    }

    #[test]
    fn scan_reports_size_and_delays_without_pixels() {
        for f in FIXTURES {
            let scan = scan(f.data).unwrap_or_else(|e| panic!("{}: {e}", f.name));
            assert_eq!((scan.width, scan.height), (f.width, f.height), "{}", f.name);
            let delays: Vec<u32> = f.frames.iter().map(|&(delay, _)| delay).collect();
            assert_eq!(scan.delays_ms, delays, "{}", f.name);
        }
        // a header is enough: the bitstream after it is not read
        let f = fixture("lossy_q75_33x47.webp");
        let mut damaged = f.data.to_vec();
        for b in &mut damaged[40..] {
            *b = 0xa5;
        }
        assert_eq!(scan(&damaged).unwrap().delays_ms, [0]);
    }

    #[test]
    fn exif_is_returned_only_when_the_header_announces_it() {
        let f = fixture("lossy_exif_20x12.webp");
        let exif = decode(f.data, &LIMITS).unwrap().exif.expect("EXIF block");
        assert_eq!(&exif[..4], b"MM\0*");
        assert_eq!((exif.len(), fnv1a(&exif)), (26, 0x22c2_4257_a026_3425));
        assert_eq!(scan(f.data).unwrap().exif, Some(exif.clone()));
        // the same chunk in a file whose header does not announce it
        let f = fixture("still_unannounced_24x18.webp");
        assert!(f.data.windows(4).any(|w| w == b"EXIF"));
        assert_eq!(decode(f.data, &LIMITS).unwrap().exif, None);
        assert_eq!(scan(f.data).unwrap().exif, None);
        // a prefix some writers put before the byte-order mark is cut
        let image = &fixture("lossy_q20_17x17.webp").data[12..];
        let mut prefixed = b"Exif\0\0".to_vec();
        prefixed.extend_from_slice(&exif);
        let mut body = header(FLAG_EXIF, 17, 17);
        body.extend_from_slice(image);
        body.extend_from_slice(&chunk(b"EXIF", &prefixed));
        assert_eq!(decode(&riff(&body), &LIMITS).unwrap().exif, Some(exif));
    }

    #[test]
    fn alpha_plane_of_a_still_does_not_change_its_colours() {
        // the same lossy image with its alpha plane used and ignored
        let with = fixture("la_24x18.webp");
        let without = fixture("still_unannounced_24x18.webp");
        assert_eq!(with.frames[0].1, without.frames[0].1);
    }

    #[test]
    fn keep_stops_an_animation() {
        let f = fixture("anim_mux_mixed.webp");
        let mut seen = 0;
        let size = decode_frames(f.data, &LIMITS, &mut |index, _, rgb| {
            assert_eq!(fnv1a(rgb), f.frames[index].1);
            seen += 1;
            index < 2
        });
        assert_eq!(size, Ok((24, 18)));
        assert_eq!(seen, 3);
    }

    #[test]
    fn canvas_over_the_limit_is_refused() {
        for f in FIXTURES {
            let pixels = (f.width * f.height) as u64;
            let exact = DecodeLimits { max_pixels: pixels };
            assert!(decode(f.data, &exact).is_ok(), "{}", f.name);
            let tight = DecodeLimits {
                max_pixels: pixels - 1,
            };
            if pixels > 1 {
                // the refusal is the limit's own
                let refusal = tight.check_canvas(f.width, f.height).unwrap_err();
                assert_eq!(decode(f.data, &tight).unwrap_err(), refusal, "{}", f.name);
                assert!(decode_frames(f.data, &tight, &mut |_, _, _| true).is_err());
            }
            // the cheap pass has no limit to check
            assert!(scan(f.data).is_ok());
        }
        // a huge canvas with one tiny frame: refused from the header, nothing allocated
        let image = &fixture("lossy_1x1.webp").data[12..];
        let mut body = header(FLAG_ANIMATION, 60_000, 60_000);
        body.extend_from_slice(&chunk(b"ANIM", &[0; 6]));
        body.extend_from_slice(&frame_chunk(0, 0, 1, 1, image));
        body.extend_from_slice(&frame_chunk(2, 2, 1, 1, image));
        let file = riff(&body);
        assert_eq!(scan(&file).unwrap().delays_ms, [10, 10]);
        assert_eq!(
            decode(&file, &LIMITS).err(),
            LIMITS.check_canvas(60_000, 60_000).err()
        );
        assert!(decode_frames(&file, &LIMITS, &mut |_, _, _| true).is_err());
        // 2^32 pixels and more are not a valid canvas at all
        let mut body = header(0, 1 << 16, 1 << 16);
        body.extend_from_slice(image);
        assert!(scan(&riff(&body)).is_err());
    }

    #[test]
    fn broken_containers_are_refused() {
        let good = fixture("lossy_q20_17x17.webp").data;
        let image = &good[12..];
        assert!(scan(good).is_ok());
        assert!(scan(b"").is_err());
        assert!(scan(&good[..19]).is_err());
        let patched = |at: usize, with: &[u8]| {
            let mut d = good.to_vec();
            d[at..at + with.len()].copy_from_slice(with);
            d
        };
        assert!(scan(&patched(0, b"RIFX")).is_err());
        assert!(scan(&patched(8, b"WEBQ")).is_err());
        assert!(scan(&patched(12, b"JUNK")).is_err());
        // a RIFF size past the end of the file, and one that cuts the image chunk
        let size = (good.len() - 8) as u32;
        assert!(scan(&patched(4, &(size + 1).to_le_bytes())).is_err());
        assert!(scan(&patched(4, &(size - 2).to_le_bytes())).is_err());
        assert!(scan(&patched(4, &10u32.to_le_bytes())).is_err());
        // bytes after the RIFF chunk are ignored
        let mut longer = good.to_vec();
        longer.extend_from_slice(b"trailing bytes");
        assert_eq!(frames_of(&longer).unwrap(), frames_of(good).unwrap());
        // a bitstream header that is not a shown key frame
        assert!(scan(&patched(20, &[good[20] | 1])).is_err());
        assert!(scan(&patched(23, &[0x9c])).is_err());

        let extended = |flags: u8, width: usize, height: usize, rest: &[u8]| {
            let mut body = header(flags, width, height);
            body.extend_from_slice(rest);
            riff(&body)
        };
        assert!(scan(&extended(0, 17, 17, image)).is_ok());
        assert!(
            scan(&extended(0, 18, 17, image)).is_err(),
            "canvas and image differ"
        );
        assert!(
            scan(&extended(0x01, 17, 17, image)).is_err(),
            "reserved flag"
        );
        assert!(scan(&extended(0, 17, 17, &[])).is_err(), "no image");
        assert!(
            scan(&extended(0, 17, 17, &[image, image].concat())).is_err(),
            "two images"
        );
        let mut long_header = extended(0, 17, 17, image);
        long_header[16] = 12;
        assert!(scan(&long_header).is_err(), "header chunk of another size");

        let animation = |flags: u8, chunks: &[Vec<u8>]| extended(flags, 20, 20, &chunks.concat());
        let anim = chunk(b"ANIM", &[0; 6]);
        let inside = frame_chunk(2, 2, 17, 17, image);
        assert_eq!(
            scan(&animation(FLAG_ANIMATION, &[anim.clone(), inside.clone()]))
                .unwrap()
                .delays_ms,
            [10]
        );
        assert!(
            scan(&animation(FLAG_ANIMATION, std::slice::from_ref(&inside))).is_err(),
            "frame before ANIM"
        );
        assert!(
            scan(&animation(FLAG_ANIMATION, std::slice::from_ref(&anim))).is_err(),
            "no frames"
        );
        assert!(
            scan(&animation(0, &[anim.clone(), inside])).is_err(),
            "frames without the flag"
        );
        let outside = frame_chunk(4, 2, 17, 17, image);
        assert!(
            scan(&animation(FLAG_ANIMATION, &[anim.clone(), outside])).is_err(),
            "frame outside"
        );
        let top_level = animation(FLAG_ANIMATION, &[anim, image.to_vec()]);
        assert!(scan(&top_level).is_err(), "image outside a frame");
    }

    #[test]
    fn damaged_alpha_fails_only_the_frame_that_uses_it() {
        let f = fixture("anim_alpha_raw_f1.webp");
        let at = f.data.windows(4).position(|w| w == b"ALPH").unwrap();
        let mut damaged = f.data.to_vec();
        damaged[at + 8] = 0xff;
        // frame 0 has no alpha plane and still decodes
        assert_eq!(
            fnv1a(&decode(&damaged, &LIMITS).unwrap().rgb),
            f.frames[0].1
        );
        let mut delivered = 0;
        let result = decode_frames(&damaged, &LIMITS, &mut |_, _, _| {
            delivered += 1;
            true
        });
        assert!(result.is_err());
        assert_eq!(delivered, 1);
    }

    #[test]
    fn truncated_fixtures_are_refused() {
        for f in FIXTURES {
            for cut in 0..f.data.len() {
                let part = &f.data[..cut];
                assert!(scan(part).is_err(), "{} cut at {cut}", f.name);
                assert!(decode(part, &LIMITS).is_err(), "{} cut at {cut}", f.name);
                assert!(frames_of(part).is_err(), "{} cut at {cut}", f.name);
            }
        }
    }

    /// Cuts `cut` bytes off the last chunk of a file and repairs the chunk and RIFF sizes, so
    /// that the container stays valid and the bitstream is what ends early.
    fn shorten_last_chunk(data: &[u8], cut: usize) -> Vec<u8> {
        let mut at = 12;
        loop {
            let size = le32(data, at + 4) as usize;
            let next = at + 8 + size + (size & 1);
            if next + 8 > data.len() {
                let kept = size.saturating_sub(cut);
                let mut body = data[12..at].to_vec();
                body.extend_from_slice(&chunk(
                    &[data[at], data[at + 1], data[at + 2], data[at + 3]],
                    &data[at + 8..at + 8 + kept],
                ));
                return riff(&body);
            }
            at = next;
        }
    }

    #[test]
    fn truncated_bitstreams_never_panic() {
        for f in FIXTURES {
            let last = f.data.len().min(700);
            for cut in 1..last {
                let data = shorten_last_chunk(f.data, cut);
                let _ = scan(&data);
                let _ = decode(&data, &LIMITS);
                let _ = frames_of(&data);
            }
        }
    }

    #[test]
    fn corrupted_fixtures_never_panic() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize
        };
        let small = DecodeLimits {
            max_pixels: 1 << 16,
        };
        for f in FIXTURES {
            for round in 0..400 {
                let mut data = f.data.to_vec();
                for _ in 0..1 + round % 3 {
                    let at = next() % data.len();
                    if round % 2 == 0 {
                        data[at] ^= 1 << (next() % 8);
                    } else {
                        data[at] = next() as u8;
                    }
                }
                let _ = scan(&data);
                let _ = decode(&data, &small);
                let _ = decode_frames(&data, &small, &mut |index, _, _| index < 8);
            }
        }
    }

    #[test]
    fn blend_follows_the_reference_arithmetic() {
        let blend = |src: [u8; 4], dst: [u8; 4]| {
            let mut out = dst;
            blend_pixel(&src, &mut out);
            out
        };
        // opaque and transparent sources
        assert_eq!(blend([1, 2, 3, 255], [9, 9, 9, 9]), [1, 2, 3, 255]);
        assert_eq!(blend([1, 2, 3, 0], [9, 8, 7, 6]), [9, 8, 7, 6]);
        // half over opaque: the destination weighs 255 * 128 >> 8 = 127
        assert_eq!(
            blend([200, 100, 0, 128], [0, 100, 200, 255]),
            [100, 99, 99, 255]
        );
        // over a transparent pixel the colour is rescaled by 2^24 / alpha and loses one step
        assert_eq!(blend([200, 100, 50, 3], [0, 0, 0, 0]), [199, 99, 49, 3]);
        assert_eq!(
            blend([200, 100, 50, 128], [0, 0, 0, 0]),
            [200, 100, 50, 128]
        );
    }

    /// Compares every `<name>.webp` of the directory in `GAUDIUM_IMAGE_REF_DIR` with the raw
    /// RGB dumps beside it: `<name>.rgb` for a still, `<name>.k<frame>.rgb` for the frames of
    /// an animation. Every frame that has a dump must be delivered with the pixels of the dump;
    /// frames after the last dump are not checked (the reference library may fail on them). A
    /// file with a `<name>.err` beside it must be refused.
    #[test]
    #[ignore]
    fn reference_directory() {
        let Ok(dir) = std::env::var("GAUDIUM_IMAGE_REF_DIR") else {
            return;
        };
        let limits = DecodeLimits {
            max_pixels: 1 << 28,
        };
        let (mut files, mut frames_compared) = (0, 0);
        let mut failures = Vec::new();
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .expect("GAUDIUM_IMAGE_REF_DIR")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "webp"))
            .collect();
        paths.sort();
        for path in paths {
            let data = std::fs::read(&path).unwrap();
            let beside = |ext: &str| path.with_extension(ext);
            if beside("err").exists() {
                files += 1;
                if decode(&data, &limits).is_ok() {
                    failures.push(format!(
                        "{}: decoded, the reference refuses it",
                        path.display()
                    ));
                }
                continue;
            }
            if let Ok(want) = std::fs::read(beside("rgb")) {
                files += 1;
                frames_compared += 1;
                match decode(&data, &limits) {
                    Ok(still) if still.rgb == want => {}
                    Ok(_) => failures.push(format!("{}: pixels differ", path.display())),
                    Err(e) => failures.push(format!("{}: {e}", path.display())),
                }
                continue;
            }
            if !beside("k0.rgb").exists() {
                continue;
            }
            files += 1;
            let mut wrong = Vec::new();
            let mut delivered = 0;
            let result = decode_frames(&data, &limits, &mut |index, _, rgb| {
                delivered = index + 1;
                if let Ok(want) = std::fs::read(beside(&format!("k{index}.rgb"))) {
                    frames_compared += 1;
                    if want != rgb {
                        wrong.push(index);
                    }
                }
                true
            });
            if !wrong.is_empty() {
                failures.push(format!("{}: frames {wrong:?} differ", path.display()));
            } else if beside(&format!("k{delivered}.rgb")).exists() {
                let why = result.err().unwrap_or_default();
                failures.push(format!(
                    "{}: stopped before frame {delivered}: {why}",
                    path.display()
                ));
            }
        }
        println!(
            "{files} files, {frames_compared} frames compared, {} failures",
            failures.len()
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
