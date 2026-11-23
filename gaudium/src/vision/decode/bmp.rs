//! BMP decoder (device-independent bitmaps in a "BM" file).
//!
//! Decodes the 12-byte core header and the 40, 52, 56, 64, 108 and 124-byte info headers;
//! palette images of 1, 4 and 8 bits, 16-bit 5-5-5 and 5-6-5, 24-bit, and 32-bit in the byte
//! orders listed at `Samples::for_masks`; bottom-up and top-down rows; RLE8 and RLE4.
//!
//! Refuses: a file without the "BM" signature or with another header size; depths other than
//! 1, 4, 8, 16, 24 and 32 bits; embedded JPEG or PNG data and every compression code above 3;
//! channel masks other than the listed ones (and masks on a palette image); run-length data on
//! a 16, 24 or 32-bit image; a palette of more than 256 colours; pixel data that ends before
//! the image is complete; run-length data that does not fill the image.
//!
//! Rules that make the RGB output equal to the reference image library's `convert("RGB")`:
//! - A fourth byte per pixel (alpha or padding) is dropped, never blended.
//! - 5-bit samples scale as `v * 255 / 31` and 6-bit samples as `v * 255 / 63`, rounded down.
//! - A palette index beyond the palette is black. The palette is read from the bytes after
//!   the header, as many entries as the header announces (2^bits when it says 0) and the file
//!   holds whole, wherever the pixel data starts.
//! - A pixel offset of 0 means "right after the header, masks and palette"; a palette image
//!   whose offset points at the end of its header has its pixels after the palette.
//! - A height whose top byte is 0xFF is a top-down image; any other value is a row count.
//! - A palette of exactly black then white (or of black alone) on a 1-bit image shows index 1
//!   as white. A palette that is the grey ramp 0, 1, 2, ... on an 8-bit image shows every
//!   index as that grey level, also an index beyond the ramp's end. On run-length data of
//!   any depth the same two palettes apply to the decoded indices: every index but 0 is
//!   white, or every index is its grey level.
//! - Run-length data follows the reference's reading of it: a run is cut at the row's end, an
//!   end-of-line code on a complete row moves nowhere, a literal run continues into the next
//!   row, its padding byte is the one that makes the file position even, and a delta leaves
//!   index 0 behind.
//!
//! Not matched on purpose: with one of those two grey palettes on an uncompressed image of a
//! different depth (black-and-white palette on 4 or 8 bits, grey ramp on 1 or 4 bits) the
//! reference reads the rows at the wrong depth and shows noise. Such a file is refused here.
//!
//! Limits: the canvas is checked right after the header. An uncompressed image is allocated
//! only once the file is known to hold all its rows, so the output is at most 24 times the
//! file's size. Run-length data grows its index buffer as it is decoded, never past the
//! canvas, and the RGB buffer is allocated after that succeeded.

use super::types::{DecodeLimits, Still};

/// Size of the canvas, from the header.
pub fn probe(bytes: &[u8]) -> Result<(usize, usize), String> {
    let header = Header::parse(bytes)?;
    Ok((header.width, header.height))
}

/// The image as RGB8, rows top to bottom. A BMP file has no EXIF block.
pub fn decode(bytes: &[u8], limits: &DecodeLimits) -> Result<Still, String> {
    let header = Header::parse(bytes)?;
    limits.check_canvas(header.width, header.height)?;
    let pixels = header
        .width
        .checked_mul(header.height)
        .ok_or_else(too_large)?;
    let rgb_len = pixels.checked_mul(3).ok_or_else(too_large)?;

    let palette = match header.samples {
        Samples::Indexed => Some(Palette::read(bytes, &header)?),
        _ => None,
    };

    // Where the pixel data starts.
    let mut start = u64::from(header.data_offset);
    if start == header.end as u64 && header.bits <= 8 {
        start += header.palette_bytes();
    }
    if start == 0 {
        start = header.after_masks as u64 + palette.as_ref().map_or(0, |p| p.stored as u64);
    }
    let start = usize::try_from(start).map_err(|_| cut_short())?;

    let rgb = match (header.compression, &palette) {
        (Compression::Rle8 | Compression::Rle4, Some(palette)) => {
            let nibbles = header.compression == Compression::Rle4;
            let indices = run_length_indices(bytes, start, header.width, pixels, nibbles)?;
            let mut rgb = vec![0u8; rgb_len];
            for (y, out) in rgb.chunks_exact_mut(header.width * 3).enumerate() {
                let at = header.image_row(y) * header.width;
                let row = indices.get(at..at + header.width).unwrap_or(&[]);
                for (px, &index) in out.as_chunks_mut::<3>().0.iter_mut().zip(row) {
                    *px = palette.colors[usize::from(index)];
                }
            }
            rgb
        }
        _ => uncompressed(bytes, start, &header, palette.as_ref(), rgb_len)?,
    };

    Ok(Still {
        width: header.width,
        height: header.height,
        rgb,
        exif: None,
    })
}

fn too_large() -> String {
    "BMP image is too large".to_string()
}

fn cut_short() -> String {
    "BMP file is cut short (pixel data is missing)".to_string()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Compression {
    None,
    Rle8,
    Rle4,
}

/// How the bytes of an uncompressed row turn into pixels.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Samples {
    /// Palette indices of 1, 4 or 8 bits, the leftmost pixel in the most significant bits.
    Indexed,
    /// 16 bits, little endian: five bits each of red, green and blue from bit 14 down.
    Rgb555,
    /// 16 bits, little endian: five bits of red, six of green, five of blue from bit 15 down.
    Rgb565,
    /// `size` bytes per pixel with red, green and blue at the given byte positions.
    Bytes {
        size: usize,
        red: usize,
        green: usize,
        blue: usize,
    },
}

impl Samples {
    /// The layout for a bit depth without channel masks.
    fn for_depth(bits: u32) -> Samples {
        match bits {
            16 => Samples::Rgb555,
            24 => Samples::Bytes {
                size: 3,
                red: 2,
                green: 1,
                blue: 0,
            },
            32 => Samples::Bytes {
                size: 4,
                red: 2,
                green: 1,
                blue: 0,
            },
            _ => Samples::Indexed,
        }
    }

    /// The layout for a bit depth with channel masks (red, green, blue, alpha), or `None` for
    /// a combination the reference library does not read.
    ///
    /// 32 bits: each colour mask selects one whole byte; the accepted sets are blue-green-red
    /// with the fourth byte unused or alpha, red-green-blue with alpha, the three orders that
    /// put red in the top byte, and four zero masks (read as blue-green-red-alpha).
    fn for_masks(bits: u32, masks: [u32; 4]) -> Option<Samples> {
        const BYTE: [u32; 4] = [0xff, 0xff00, 0x00ff_0000, 0xff00_0000];
        match bits {
            16 => match [masks[0], masks[1], masks[2]] {
                [0xf800, 0x07e0, 0x001f] => Some(Samples::Rgb565),
                [0x7c00, 0x03e0, 0x001f] => Some(Samples::Rgb555),
                _ => None,
            },
            24 => ([masks[0], masks[1], masks[2]] == [BYTE[2], BYTE[1], BYTE[0]])
                .then_some(Samples::for_depth(24)),
            32 => {
                // Byte positions of red, green, blue and the alpha mask that goes with them.
                const ACCEPTED: [([usize; 3], u32); 7] = [
                    ([2, 1, 0], 0),
                    ([3, 2, 1], 0),
                    ([3, 1, 0], 0),
                    ([3, 2, 1], BYTE[0]),
                    ([0, 1, 2], BYTE[3]),
                    ([2, 1, 0], BYTE[3]),
                    ([3, 1, 0], BYTE[2]),
                ];
                if masks == [0; 4] {
                    return Some(Samples::for_depth(32));
                }
                ACCEPTED
                    .iter()
                    .find(|(at, alpha)| masks == [BYTE[at[0]], BYTE[at[1]], BYTE[at[2]], *alpha])
                    .map(|(at, _)| Samples::Bytes {
                        size: 4,
                        red: at[0],
                        green: at[1],
                        blue: at[2],
                    })
            }
            _ => None,
        }
    }
}

struct Header {
    width: usize,
    height: usize,
    /// True when the first row in the file is the top row of the image.
    top_down: bool,
    bits: u32,
    compression: Compression,
    samples: Samples,
    /// Palette entries the header announces (only meaningful for 1, 4 and 8 bits).
    colors: u32,
    /// Bytes per palette entry: 3 after a core header, 4 otherwise.
    entry_size: usize,
    /// File position after the header.
    end: usize,
    /// File position after the header and the channel masks that follow a 40-byte header.
    after_masks: usize,
    /// The pixel data position the file header gives.
    data_offset: u32,
}

fn le16(bytes: &[u8], at: usize) -> Option<u32> {
    let b = bytes.get(at..at.checked_add(2)?)?;
    Some(u32::from(u16::from_le_bytes([b[0], b[1]])))
}

fn le32(bytes: &[u8], at: usize) -> Option<u32> {
    let b = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

impl Header {
    fn parse(bytes: &[u8]) -> Result<Header, String> {
        if bytes.get(..2) != Some(b"BM") {
            return Err("not a BMP file".to_string());
        }
        let cut = || "BMP file is cut short (header is incomplete)".to_string();
        let data_offset = le32(bytes, 10).ok_or_else(cut)?;
        let header_size = le32(bytes, 14).ok_or_else(cut)?;
        if ![12, 40, 52, 56, 64, 108, 124].contains(&header_size) {
            return Err(format!("unsupported BMP header size ({header_size})"));
        }
        let end = 14 + header_size as usize;
        if bytes.len() < end {
            return Err(cut());
        }
        // The header is complete, so the fixed fields below are all present.
        let field16 = |at: usize| le16(bytes, at).ok_or_else(cut);
        let field32 = |at: usize| le32(bytes, at).ok_or_else(cut);

        let (width, height, top_down, bits, code, colors);
        if header_size == 12 {
            width = field16(18)?;
            height = field16(20)?;
            top_down = false;
            bits = field16(24)?;
            code = 0;
            colors = 0;
        } else {
            width = field32(18)?;
            let rows = field32(22)?;
            // The reference library takes a height whose top byte is 0xFF as the negative
            // count of a top-down image, and every other value as an unsigned row count.
            top_down = rows >> 24 == 0xff;
            height = if top_down { rows.wrapping_neg() } else { rows };
            bits = field16(28)?;
            code = field32(30)?;
            colors = field32(46)?;
        }
        if width == 0 || height == 0 {
            return Err("BMP image has zero size".to_string());
        }
        if ![1, 4, 8, 16, 24, 32].contains(&bits) {
            return Err(format!("unsupported BMP pixel depth ({bits} bits)"));
        }

        let mut after_masks = end;
        let (compression, samples) = match code {
            0 => (Compression::None, Samples::for_depth(bits)),
            1 | 2 => {
                if bits > 8 {
                    return Err(format!(
                        "unsupported BMP: run-length data on a {bits}-bit image"
                    ));
                }
                let kind = if code == 1 {
                    Compression::Rle8
                } else {
                    Compression::Rle4
                };
                (kind, Samples::Indexed)
            }
            3 => {
                // Headers of 52 bytes and more hold the masks (alpha from 56 bytes on); after
                // a 40-byte header the three colour masks follow it.
                let mut masks = [0u32; 4];
                let (at, count) = match header_size {
                    40 => (end, 3),
                    52 => (54, 3),
                    _ => (54, 4),
                };
                for (i, mask) in masks.iter_mut().take(count).enumerate() {
                    *mask = field32(at + i * 4)?;
                }
                if header_size == 40 {
                    after_masks = end + 12;
                }
                match Samples::for_masks(bits, masks) {
                    Some(samples) => (Compression::None, samples),
                    None => return Err("unsupported BMP channel masks".to_string()),
                }
            }
            4 | 5 => return Err("unsupported BMP: embedded JPEG or PNG data".to_string()),
            other => return Err(format!("unsupported BMP compression ({other})")),
        };

        let colors = if colors == 0 && bits <= 8 {
            1 << bits
        } else {
            colors
        };
        if bits <= 8 && colors > 65536 {
            return Err(format!("unsupported BMP palette size ({colors} colours)"));
        }

        Ok(Header {
            width: width as usize,
            height: height as usize,
            top_down,
            bits,
            compression,
            samples,
            colors,
            entry_size: if header_size == 12 { 3 } else { 4 },
            end,
            after_masks,
            data_offset,
        })
    }

    /// Bytes the announced palette takes in the file.
    fn palette_bytes(&self) -> u64 {
        self.entry_size as u64 * u64::from(self.colors)
    }

    /// The image row that the `file_row`-th row of pixel data holds.
    fn image_row(&self, file_row: usize) -> usize {
        if self.top_down {
            file_row
        } else {
            self.height.saturating_sub(1).saturating_sub(file_row)
        }
    }
}

/// The colour of every 8-bit index value.
struct Palette {
    colors: [[u8; 3]; 256],
    /// Palette bytes the file holds (the announced size, or less at the end of the file).
    stored: usize,
}

impl Palette {
    fn read(bytes: &[u8], header: &Header) -> Result<Palette, String> {
        let entry = header.entry_size;
        let announced = usize::try_from(header.palette_bytes()).unwrap_or(usize::MAX);
        let from = header.end.min(bytes.len());
        let to = from.saturating_add(announced).min(bytes.len());
        let stored = bytes.get(from..to).unwrap_or(&[]);
        let entries = stored.chunks_exact(entry);

        // The two palettes the reference library replaces by a fixed rule: black then white
        // (or black alone), and the grey ramp whose entry i is the level i modulo 256. Only
        // the three colour bytes of an entry are compared.
        let two = header.colors <= 2;
        let level = |i: usize| if two { (i * 255) as u8 } else { i as u8 };
        let fixed = (0..header.colors as usize)
            .all(|i| stored.get(i * entry..i * entry + 3) == Some(&[level(i); 3][..]));

        let mut colors = [[0u8; 3]; 256];
        let run_length = header.compression != Compression::None;
        if fixed && two {
            if !run_length && header.bits != 1 {
                return Err(format!(
                    "unsupported BMP: black-and-white palette on a {}-bit image",
                    header.bits
                ));
            }
            // Every index but 0 is white (a 1-bit row only holds 0 and 1).
            for color in colors.iter_mut().skip(1) {
                *color = [255; 3];
            }
        } else if fixed {
            if !run_length && header.bits != 8 {
                return Err(format!(
                    "unsupported BMP: grey-ramp palette on a {}-bit image",
                    header.bits
                ));
            }
            for (i, color) in colors.iter_mut().enumerate() {
                *color = [i as u8; 3];
            }
        } else {
            if entries.len() > 256 {
                return Err("unsupported BMP: palette of more than 256 colours".to_string());
            }
            // Entries are stored blue, green, red.
            for (color, e) in colors.iter_mut().zip(entries) {
                *color = [e[2], e[1], e[0]];
            }
        }
        Ok(Palette {
            colors,
            stored: stored.len(),
        })
    }
}

/// `v * 255 / 31` for every 5-bit sample.
const SCALE5: [u8; 32] = scale_table::<32>();
/// `v * 255 / 63` for every 6-bit sample.
const SCALE6: [u8; 64] = scale_table::<64>();

const fn scale_table<const N: usize>() -> [u8; N] {
    let mut table = [0u8; N];
    let mut v = 0;
    while v < N {
        table[v] = (v * 255 / (N - 1)) as u8;
        v += 1;
    }
    table
}

/// Decodes rows stored without compression: each row takes a multiple of four bytes.
fn uncompressed(
    bytes: &[u8],
    start: usize,
    header: &Header,
    palette: Option<&Palette>,
    rgb_len: usize,
) -> Result<Vec<u8>, String> {
    let row_bits = (header.width as u64)
        .checked_mul(u64::from(header.bits))
        .ok_or_else(too_large)?;
    let line = usize::try_from(row_bits.div_ceil(8)).map_err(|_| too_large())?;
    let stride = usize::try_from(row_bits.div_ceil(32) * 4).map_err(|_| too_large())?;
    // The last row in the file does not need its padding.
    let needed = (header.height - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(line))
        .and_then(|n| n.checked_add(start))
        .ok_or_else(cut_short)?;
    if needed > bytes.len() {
        return Err(cut_short());
    }

    let mut rgb = vec![0u8; rgb_len];
    let out_rows = rgb.chunks_exact_mut(header.width * 3);
    for (y, out) in out_rows.enumerate() {
        let file_row = header.image_row(y);
        let src = file_row
            .checked_mul(stride)
            .and_then(|n| n.checked_add(start))
            .and_then(|at| bytes.get(at..at.checked_add(line)?))
            .ok_or_else(cut_short)?;
        let pixels = out.as_chunks_mut::<3>().0.iter_mut();
        match (header.samples, palette) {
            (Samples::Indexed, Some(palette)) => {
                let table = &palette.colors;
                match header.bits {
                    8 => {
                        for (px, &index) in pixels.zip(src) {
                            *px = table[usize::from(index)];
                        }
                    }
                    4 => {
                        let indices = src.iter().flat_map(|&b| [b >> 4, b & 15]);
                        for (px, index) in pixels.zip(indices) {
                            *px = table[usize::from(index)];
                        }
                    }
                    _ => {
                        let indices = src
                            .iter()
                            .flat_map(|&b| (0..8).map(move |bit| (b >> (7 - bit)) & 1));
                        for (px, index) in pixels.zip(indices) {
                            *px = table[usize::from(index)];
                        }
                    }
                }
            }
            (Samples::Indexed, None) => return Err("BMP image has no palette".to_string()),
            (Samples::Rgb555, _) => {
                for (px, s) in pixels.zip(src.as_chunks::<2>().0) {
                    let v = usize::from(u16::from_le_bytes(*s));
                    *px = [
                        SCALE5[(v >> 10) & 31],
                        SCALE5[(v >> 5) & 31],
                        SCALE5[v & 31],
                    ];
                }
            }
            (Samples::Rgb565, _) => {
                for (px, s) in pixels.zip(src.as_chunks::<2>().0) {
                    let v = usize::from(u16::from_le_bytes(*s));
                    *px = [
                        SCALE5[(v >> 11) & 31],
                        SCALE6[(v >> 5) & 63],
                        SCALE5[v & 31],
                    ];
                }
            }
            (
                Samples::Bytes {
                    size,
                    red,
                    green,
                    blue,
                },
                _,
            ) => {
                for (px, s) in pixels.zip(src.chunks_exact(size)) {
                    *px = [s[red], s[green], s[blue]];
                }
            }
        }
    }
    Ok(rgb)
}

/// Decodes RLE8 or RLE4 data into one palette index per pixel, in file row order.
///
/// The data is a list of two-byte codes: (n, v) with n > 0 repeats v for n pixels (RLE4
/// alternates the two halves of v); (0, 0) ends the row; (0, 1) ends the image; (0, 2, dx,
/// dy) skips pixels; (0, n) with n > 2 is followed by n literal indices and padding. The edge
/// cases follow the reference library, as listed in the module header.
fn run_length_indices(
    bytes: &[u8],
    start: usize,
    width: usize,
    pixels: usize,
    nibbles: bool,
) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::new();
    // Position in the file, and the column the run codes have reached. A literal run that
    // continues into the next row moves `out` on while the column runs past the width, so the
    // two are tracked separately.
    let mut pos = start;
    let mut column = 0usize;
    while out.len() < pixels {
        let (count, value) = match bytes.get(pos..pos.saturating_add(2)) {
            Some(&[count, value]) => (usize::from(count), value),
            _ => break,
        };
        pos += 2;
        if count != 0 {
            let n = count.min(width.saturating_sub(column));
            if nibbles {
                out.extend((0..n).map(|i| {
                    if i.is_multiple_of(2) {
                        value >> 4
                    } else {
                        value & 15
                    }
                }));
            } else {
                out.extend(std::iter::repeat_n(value, n));
            }
            column += n;
            continue;
        }
        match value {
            0 => {
                let rest = (width - out.len() % width) % width;
                out.extend(std::iter::repeat_n(0, rest));
                column = 0;
            }
            1 => break,
            2 => {
                let (right, up) = match bytes.get(pos..pos.saturating_add(2)) {
                    Some(&[right, up]) => (usize::from(right), usize::from(up)),
                    _ => break,
                };
                pos += 2;
                let skip = up
                    .saturating_mul(width)
                    .saturating_add(right)
                    .min(pixels - out.len());
                out.extend(std::iter::repeat_n(0, skip));
                column = out.len() % width;
            }
            literal => {
                let announced = usize::from(literal);
                let take = announced.min(pixels - out.len());
                let stored = if nibbles { take.div_ceil(2) } else { take };
                let end = pos.saturating_add(stored).min(bytes.len());
                let src = bytes.get(pos..end).unwrap_or(&[]);
                if nibbles {
                    for (i, &b) in src.iter().enumerate() {
                        out.push(b >> 4);
                        if i * 2 + 1 < take {
                            out.push(b & 15);
                        }
                    }
                } else {
                    out.extend_from_slice(src);
                }
                pos += src.len();
                if src.len() < stored || out.len() >= pixels {
                    break;
                }
                column += announced;
                pos += pos % 2;
            }
        }
    }
    if out.len() < pixels {
        return Err("BMP run-length data ends before the image is complete".to_string());
    }
    out.truncate(pixels);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: DecodeLimits = DecodeLimits {
        max_pixels: 1 << 20,
    };

    macro_rules! fixtures {
        ($(($name:literal, $width:literal, $height:literal, $hash:literal),)*) => {
            &[$((
                $name,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/data/vision/bmp/",
                    $name
                )) as &[u8],
                $width,
                $height,
                $hash,
            ),)*]
        };
    }

    /// Name, file, width, height, and the FNV-1a 64 hash of the RGB bytes the reference image
    /// library returns for the file. `tests/data/vision/bmp/make_fixtures.py` writes the files
    /// and prints these lines.
    const FIXTURES: &[(&str, &[u8], usize, usize, u64)] = fixtures![
        ("pal1.bmp", 13, 7, 0x498f6bd21c25e63d),
        ("pal1_black_white.bmp", 13, 7, 0x3b957d8f96ca0a4f),
        ("pal1_white_black.bmp", 13, 7, 0x82a099790245b200),
        ("pal1_black_only.bmp", 13, 7, 0x5d69d954edbefdd0),
        ("pal4.bmp", 13, 7, 0xa42ae0f4702b8857),
        ("pal4_top_down.bmp", 13, 7, 0xf0d989137c73c439),
        ("pal8.bmp", 13, 7, 0x0fae373a2ae1e200),
        ("pal8_short.bmp", 13, 7, 0xb7040def076cf82b),
        ("pal8_grey_ramp.bmp", 13, 7, 0x5a93d7213f3d2112),
        ("pal8_grey_ramp_short.bmp", 13, 7, 0x6326766ae7956dfc),
        ("pal8_core_header.bmp", 13, 7, 0x974fa0c3fcf02663),
        ("pal4_core_header.bmp", 13, 7, 0x12718a9b34d4c252),
        ("pal8_offset_zero.bmp", 13, 7, 0x7ecb102162e50ee9),
        ("pal8_offset_header_end.bmp", 13, 7, 0x7ecb102162e50ee9),
        ("pal8_gap.bmp", 13, 7, 0x7ecb102162e50ee9),
        ("pal8_no_last_padding.bmp", 13, 7, 0x7ecb102162e50ee9),
        ("rgb555.bmp", 13, 7, 0xffe3e4312c4cb07e),
        ("rgb565_masks.bmp", 13, 7, 0xe57452af2988cde1),
        ("rgb555_masks_v4.bmp", 13, 7, 0xffe3e4312c4cb07e),
        ("rgb24.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb24_top_down.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb24_core_header.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb24_os2_header.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb24_masks.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb24_1x1.bmp", 1, 1, 0xd94d12186c0f2fb7),
        ("rgb32.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb32_bgrx_after_header.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb32_bgrx_v2.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb32_xbgr_v3.bmp", 13, 7, 0x4fcd099c307d987e),
        ("rgb32_bgxr_v4.bmp", 13, 7, 0xddfe29a2c1bbc036),
        ("rgb32_abgr_v4.bmp", 13, 7, 0x4fcd099c307d987e),
        ("rgb32_rgba_v5.bmp", 13, 7, 0xb28ad19ea464126a),
        ("rgb32_bgra_v5.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rgb32_bgar_v5.bmp", 13, 7, 0xddfe29a2c1bbc036),
        ("rgb32_zero_masks_v5.bmp", 13, 7, 0x95050198c8ecdf4a),
        ("rle8.bmp", 21, 9, 0xb530efc91c080332),
        ("rle8_top_down.bmp", 21, 9, 0xa371d4ebc0ab4ccf),
        ("rle8_odd_offset.bmp", 21, 9, 0xa0a168aa9972c33b),
        ("rle8_grey_ramp.bmp", 21, 9, 0x96625b1e99683ca1),
        ("rle8_black_white.bmp", 21, 9, 0xb1869b330fb281a8),
        ("rle4.bmp", 21, 9, 0xcfeda6e2279cfb66),
        ("rle4_short_palette.bmp", 21, 9, 0x7e30458cc7e084cf),
        ("rle8_edge_cases.bmp", 5, 5, 0x1b04a1cacaeb2446),
    ];

    fn fnv1a64(data: &[u8]) -> u64 {
        data.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    fn fixture(name: &str) -> &'static [u8] {
        FIXTURES
            .iter()
            .find(|f| f.0 == name)
            .unwrap_or_else(|| panic!("no fixture {name}"))
            .1
    }

    /// A file with a 40-byte header: `extra` (masks, palette) follows the header, then `data`.
    fn bmp_of(
        width: i32,
        height: i32,
        bits: u16,
        compression: u32,
        extra: &[u8],
        data: &[u8],
    ) -> Vec<u8> {
        let start = 54 + extra.len();
        let mut out = b"BM".to_vec();
        out.extend_from_slice(&((start + data.len()) as u32).to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&(start as u32).to_le_bytes());
        out.extend_from_slice(&40u32.to_le_bytes());
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(&compression.to_le_bytes());
        out.extend_from_slice(&[0; 20]);
        out.extend_from_slice(extra);
        out.extend_from_slice(data);
        out
    }

    /// A fixture with one header field (at `at`, `value.len()` bytes) replaced.
    fn patched(name: &str, at: usize, value: &[u8]) -> Vec<u8> {
        let mut file = fixture(name).to_vec();
        file[at..at + value.len()].copy_from_slice(value);
        file
    }

    #[test]
    fn fixtures_match_the_reference() {
        for &(name, bytes, width, height, hash) in FIXTURES {
            let still = decode(bytes, &LIMITS).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((still.width, still.height), (width, height), "{name}");
            assert_eq!(still.rgb.len(), width * height * 3, "{name}");
            assert_eq!(fnv1a64(&still.rgb), hash, "{name}");
            assert_eq!(still.exif, None, "{name}");
            assert_eq!(probe(bytes), Ok((width, height)), "{name}");
        }
    }

    #[test]
    fn sample_rules() {
        let rgb = |file: &[u8]| decode(file, &LIMITS).unwrap().rgb;
        // 5-5-5: v * 255 / 31 per channel, the top bit unused.
        let words = [0x01, 0x00, 0xe0, 0x03, 0x00, 0x7c, 0x21, 0x84];
        assert_eq!(
            rgb(&bmp_of(4, 1, 16, 0, &[], &words)),
            [0, 0, 8, 0, 255, 0, 255, 0, 0, 8, 8, 8]
        );
        // 5-6-5: green scales as v * 255 / 63.
        let masks = [0x00, 0xf8, 0, 0, 0xe0, 0x07, 0, 0, 0x1f, 0, 0, 0];
        let words = [0x20, 0x00, 0xe0, 0x07, 0x00, 0xf8, 0x1f, 0x00];
        assert_eq!(
            rgb(&bmp_of(4, 1, 16, 3, &masks, &words)),
            [0, 4, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255]
        );
        // 24 and 32 bits are blue, green, red; the fourth byte is dropped.
        assert_eq!(rgb(&bmp_of(1, 1, 24, 0, &[], &[1, 2, 3, 0])), [3, 2, 1]);
        assert_eq!(rgb(&bmp_of(1, 1, 32, 0, &[], &[1, 2, 3, 0])), [3, 2, 1]);
        // Rows are stored bottom-up unless the height is negative.
        let rows = [10, 10, 10, 0, 20, 20, 20, 0];
        assert_eq!(
            rgb(&bmp_of(1, 2, 24, 0, &[], &rows)),
            [20, 20, 20, 10, 10, 10]
        );
        assert_eq!(
            rgb(&bmp_of(1, -2, 24, 0, &[], &rows)),
            [10, 10, 10, 20, 20, 20]
        );
        // A palette index beyond the palette is black. The palette size is at offset 46.
        let palette = [1, 2, 3, 0, 4, 5, 6, 0];
        let mut file = bmp_of(3, 1, 8, 0, &palette, &[1, 0, 2, 0]);
        file[46] = 2;
        assert_eq!(rgb(&file), [6, 5, 4, 3, 2, 1, 0, 0, 0]);
    }

    #[test]
    fn run_length_rules() {
        let palette: Vec<u8> = (0..=255u8).flat_map(|i| [i, i, i, 0]).collect();
        let grey = |width: i32, height: i32, data: &[u8]| {
            decode(&bmp_of(width, height, 8, 1, &palette, data), &LIMITS).map(|s| {
                s.rgb
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|px| px[0])
                    .collect::<Vec<u8>>()
            })
        };
        // A run is cut at the end of its row, and an end-of-line code on a complete row (or
        // on an empty one) does not move to another row.
        assert_eq!(
            grey(2, -3, &[5, 7, 0, 0, 0, 0, 2, 8, 0, 0, 2, 9, 0, 1]),
            Ok(vec![7, 7, 8, 8, 9, 9])
        );
        // An end-of-line code fills the rest of its row with index 0.
        assert_eq!(grey(3, -2, &[1, 7, 0, 0, 3, 8]), Ok(vec![7, 0, 0, 8, 8, 8]));
        // A literal run with its padding byte, then a delta that leaves index 0 behind.
        assert_eq!(
            grey(4, -2, &[0, 3, 1, 2, 3, 0, 0, 2, 2, 0, 3, 9]),
            Ok(vec![1, 2, 3, 0, 0, 9, 9, 9])
        );
        // The end-of-image code before the image is full, and data that just stops.
        assert!(grey(2, 2, &[2, 7, 0, 0, 0, 1]).is_err());
        assert!(grey(2, 2, &[2, 7, 0, 0, 1]).is_err());
        assert!(grey(2, 2, &[]).is_err());
        // Data that goes on after the image is full is not looked at.
        assert_eq!(grey(2, -1, &[2, 7, 0, 3, 1]), Ok(vec![7, 7]));
        // RLE4: a run alternates the two halves of its byte.
        let file = bmp_of(5, 1, 4, 2, &palette[..64], &[5, 0x12, 0, 1]);
        let rgb = decode(&file, &LIMITS).unwrap().rgb;
        assert_eq!(rgb, [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1]);
    }

    #[test]
    fn unsupported_files_are_refused() {
        let refused = |file: Vec<u8>| {
            assert!(probe(&file).is_err());
            decode(&file, &LIMITS).unwrap_err()
        };
        assert_eq!(refused(b"MB".to_vec()), "not a BMP file");
        assert_eq!(refused(Vec::new()), "not a BMP file");
        // Header size (offset 14), depth (28), compression (30).
        refused(patched("rgb24.bmp", 14, &16u32.to_le_bytes()));
        refused(patched("rgb24.bmp", 28, &2u16.to_le_bytes()));
        refused(patched("rgb24.bmp", 28, &48u16.to_le_bytes()));
        for code in [1u32, 2, 4, 5, 6, 11] {
            refused(patched("rgb24.bmp", 30, &code.to_le_bytes()));
        }
        // Masks on a palette image, and masks that are not whole known bytes.
        refused(patched("pal8.bmp", 30, &3u32.to_le_bytes()));
        refused(patched("rgb565_masks.bmp", 54, &0x0f00u32.to_le_bytes()));
        refused(patched("rgb32_rgba_v5.bmp", 66, &0u32.to_le_bytes()));
        // Zero size (width at 18, height at 22).
        refused(patched("rgb24.bmp", 18, &0u32.to_le_bytes()));
        refused(patched("rgb24.bmp", 22, &0u32.to_le_bytes()));
        // A palette of more than 256 colours (count at offset 46).
        assert!(decode(&patched("pal8.bmp", 46, &300u32.to_le_bytes()), &LIMITS).is_err());
        refused(patched("pal8.bmp", 46, &70000u32.to_le_bytes()));
        // The grey palettes the reference library reads at the wrong depth.
        let black_white = [0, 0, 0, 0, 255, 255, 255, 0];
        let mut file = bmp_of(2, 1, 8, 0, &black_white, &[0, 1, 0, 0]);
        file[46] = 2;
        assert!(decode(&file, &LIMITS)
            .unwrap_err()
            .contains("black-and-white"));
        let ramp: Vec<u8> = (0..16u8).flat_map(|i| [i, i, i, 0]).collect();
        let file = bmp_of(2, 1, 4, 0, &ramp, &[0x12, 0, 0, 0]);
        assert!(decode(&file, &LIMITS).unwrap_err().contains("grey-ramp"));
    }

    #[test]
    fn missing_pixel_data_is_refused() {
        // 13 pixels of 24 bits: 39 bytes and one byte of padding per row.
        let bytes = fixture("rgb24.bmp");
        assert!(decode(&bytes[..bytes.len() - 1], &LIMITS).is_ok());
        for len in 0..bytes.len() - 1 {
            assert!(decode(&bytes[..len], &LIMITS).is_err(), "cut at {len}");
        }
        let bytes = fixture("rle8.bmp");
        for len in 0..bytes.len() - 60 {
            assert!(decode(&bytes[..len], &LIMITS).is_err(), "cut at {len}");
        }
        assert!(probe(&bytes[..53]).is_err());
        assert_eq!(probe(&bytes[..54]), Ok((21, 9)));
        // A pixel offset past the end of the file.
        let file = patched("rgb24.bmp", 10, &5000u32.to_le_bytes());
        assert!(decode(&file, &LIMITS).is_err());
        let file = patched("rle8.bmp", 10, &u32::MAX.to_le_bytes());
        assert!(decode(&file, &LIMITS).is_err());
    }

    #[test]
    fn canvas_limit_is_enforced() {
        let bytes = fixture("rgb24.bmp");
        let tight = DecodeLimits {
            max_pixels: 13 * 7 - 1,
        };
        assert!(decode(bytes, &tight).unwrap_err().contains("limit"));
        let exact = DecodeLimits { max_pixels: 13 * 7 };
        assert!(decode(bytes, &exact).is_ok());
        // Sizes far beyond the file: refused by the limit, and within the limit by the
        // missing rows, before any buffer of the announced size exists.
        let unlimited = DecodeLimits {
            max_pixels: u64::MAX,
        };
        for size in [0x7fff_ffffu32, 0x8000_0000, 0xfeff_ffff, 0xff00_0000] {
            let wide = patched("rgb24.bmp", 18, &size.to_le_bytes());
            assert!(decode(&wide, &LIMITS).is_err());
            assert!(decode(&wide, &unlimited).is_err());
            let tall = patched("pal1.bmp", 22, &size.to_le_bytes());
            assert!(decode(&tall, &LIMITS).is_err());
            assert!(decode(&tall, &unlimited).is_err());
        }
        // Run-length data can fill a large canvas from a few bytes; the limit bounds it.
        let wide = patched("rle8.bmp", 18, &0x0400_0000u32.to_le_bytes());
        assert!(decode(&wide, &LIMITS).unwrap_err().contains("limit"));
    }

    /// Flips bits and cuts the file at many places; the decoder may refuse, never panic.
    #[test]
    fn damaged_input_never_panics() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for &(_, bytes, _, _, _) in FIXTURES {
            for len in 0..bytes.len() {
                let _ = decode(&bytes[..len], &LIMITS);
                let _ = probe(&bytes[..len]);
            }
            for at in 0..bytes.len() {
                // The header and the start of the data are damaged more often.
                let rounds = if at < 160 { 12 } else { 2 };
                for _ in 0..rounds {
                    let mut bad = bytes.to_vec();
                    let r = next();
                    bad[at] = match r % 4 {
                        0 => bad[at] ^ (1 << ((r >> 8) % 8)),
                        1 => (r >> 8) as u8,
                        2 => [0, 1, 2, 3, 4, 8, 16, 127, 128, 255][(r >> 8) as usize % 10],
                        _ => bad[at].wrapping_add(1),
                    };
                    // Sometimes a second byte, so that two header fields change together.
                    if r % 8 == 5 {
                        let other = (r >> 20) as usize % bad.len().min(60);
                        bad[other] = (r >> 40) as u8;
                    }
                    let _ = decode(&bad, &LIMITS);
                    let _ = probe(&bad);
                }
            }
        }
    }

    /// Compares every `<name>.bmp` in the directory `GAUDIUM_IMAGE_REF_DIR` with `<name>.rgb`,
    /// the raw RGB bytes the reference image library returns for it.
    #[test]
    #[ignore = "needs GAUDIUM_IMAGE_REF_DIR"]
    fn reference_directory() {
        let dir = std::env::var("GAUDIUM_IMAGE_REF_DIR").expect("GAUDIUM_IMAGE_REF_DIR is not set");
        let limits = DecodeLimits {
            max_pixels: 1 << 28,
        };
        let mut compared = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("bmp") {
                continue;
            }
            let Ok(expected) = std::fs::read(path.with_extension("rgb")) else {
                continue;
            };
            let still = decode(&std::fs::read(&path).unwrap(), &limits)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(still.rgb == expected, "{} differs", path.display());
            compared += 1;
        }
        assert!(compared > 0, "no <name>.bmp with <name>.rgb in {dir}");
    }
}
