//! PNG decoder (PNG specification, third edition; zlib and DEFLATE in `inflate.rs`).
//!
//! Decodes every colour type and bit depth (greyscale 1/2/4/8/16, truecolour 8/16, palette
//! 1/2/4/8, greyscale with alpha 8/16, truecolour with alpha 8/16), all five row filters,
//! Adam7 interlace and image data split over any number of IDAT chunks. For an animated PNG it
//! returns the first frame the reference image library shows, which is the IDAT image.
//!
//! Refuses: a file that is not a complete PNG datastream (bad signature, a chunk cut short,
//! no IEND), a bad CRC in a critical chunk, chunks out of order (IHDR not first or repeated,
//! PLTE after IDAT or repeated, IDAT chunks that are not consecutive), an unknown critical
//! chunk, header values the specification does not define, a palette image without PLTE, a
//! filter type above 4, image data that is shorter than the image, and a zlib stream that is
//! damaged (see `inflate.rs`). An ancillary chunk with a bad CRC is skipped.
//!
//! This is stricter than the reference image library, which checks no CRC from the first IDAT
//! chunk on, needs no IEND, reads a palette image without PLTE as black, and shows the rows it
//! got when the image data ends early on a row boundary. A file refused here for one of those
//! reasons is damaged or not a PNG datastream; a file both decode gives the same pixels.
//!
//! Rules that make the RGB output equal to the reference library's `convert("RGB")`:
//! - Alpha is dropped, never blended: the colour samples of a transparent pixel come out as
//!   stored. tRNS (colour key or palette alpha) therefore changes nothing.
//! - gAMA, cHRM, sRGB and iCCP are not applied.
//! - 16-bit truecolour and 16-bit greyscale-with-alpha keep the high byte of each sample.
//! - 16-bit greyscale without alpha is clipped, not scaled: a sample above 255 becomes 255.
//! - Greyscale of 1, 2 and 4 bits is scaled by 255, 85 and 17.
//! - A palette index beyond the end of PLTE is black.
//! - A frame-control chunk (fcTL) before the image data makes the image data cover only that
//!   chunk's region; the rest of the canvas is zero (black, or palette entry 0). An
//!   interlaced region is placed at the canvas origin, whatever offset the chunk gives.
//! - Image data longer than the image is cut; nothing after the last row is looked at.
//!
//! Limits: the canvas is checked right after IHDR, and the zlib stream is inflated into a
//! buffer that never grows past the size the header implies (nor is set aside beyond what the
//! compressed bytes can expand to), so a file can not make the decoder allocate more than its
//! header announces, and a small file with a large header allocates little.

use super::inflate::zlib_decompress_prefix;
use super::types::{DecodeLimits, Still};

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// Size of the canvas, from IHDR.
pub fn probe(bytes: &[u8]) -> Result<(usize, usize), String> {
    let mut chunks = ChunkReader::new(bytes)?;
    let header = read_header(&mut chunks)?;
    Ok((header.width, header.height))
}

/// The image the reference library shows for the file (frame 0), as RGB8. `Still::exif` is
/// the payload of the eXIf chunk if the file has one.
pub fn decode(bytes: &[u8], limits: &DecodeLimits) -> Result<Still, String> {
    let mut chunks = ChunkReader::new(bytes)?;
    let header = read_header(&mut chunks)?;
    limits.check_canvas(header.width, header.height)?;

    let mut palette: Option<[[u8; 3]; 256]> = None;
    let mut exif: Option<&[u8]> = None;
    // Region the image data covers: (x, y, width, height).
    let mut region = (0usize, 0usize, header.width, header.height);
    let mut idat: Vec<&[u8]> = Vec::new();
    let mut idat_closed = false;
    loop {
        let chunk = match chunks.next()? {
            Some(c) => c,
            None => return Err("PNG file ends before its IEND chunk".to_string()),
        };
        if chunk.is_critical() && !chunk.crc_ok {
            return Err(format!(
                "PNG file is damaged (bad CRC in {} chunk)",
                chunk.name()
            ));
        }
        if &chunk.kind != b"IDAT" && !idat.is_empty() {
            idat_closed = true;
        }
        match &chunk.kind {
            b"IHDR" => return Err("PNG file has more than one IHDR chunk".to_string()),
            b"PLTE" => {
                if !idat.is_empty() || palette.is_some() {
                    return Err("PNG file has a misplaced PLTE chunk".to_string());
                }
                palette = Some(read_palette(chunk.data)?);
            }
            b"IDAT" => {
                if idat_closed {
                    return Err("PNG image data is split by other chunks".to_string());
                }
                idat.push(chunk.data);
            }
            b"IEND" => break,
            _ if chunk.is_critical() => {
                return Err(format!(
                    "PNG file needs an unknown chunk type ({})",
                    chunk.name()
                ));
            }
            _ if !chunk.crc_ok => {}
            b"eXIf" => exif = Some(chunk.data),
            b"fcTL" if idat.is_empty() => {
                if let Some(r) = frame_region(chunk.data, &header)? {
                    region = r;
                }
            }
            _ => {}
        }
    }
    if idat.is_empty() {
        return Err("PNG file has no image data".to_string());
    }
    let palette = match (header.color, palette) {
        (ColorType::Palette, None) => {
            return Err("PNG palette image has no PLTE chunk".to_string());
        }
        (_, p) => p.unwrap_or([[0; 3]; 256]),
    };

    let (mut region_x, mut region_y, region_w, region_h) = region;
    if header.interlaced {
        // The reference library places the passes of an interlaced region at the canvas origin.
        (region_x, region_y) = (0, 0);
    }
    let passes = passes(header.interlaced, region_w, region_h);
    let mut raw_len: usize = 0;
    for pass in &passes {
        let row = header
            .row_bytes(pass.width)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(pass.height))
            .and_then(|n| raw_len.checked_add(n));
        raw_len = row.ok_or("PNG image is too large")?;
    }

    let mut raw = match idat.as_slice() {
        [single] => zlib_decompress_prefix(single, raw_len),
        many => zlib_decompress_prefix(&many.concat(), raw_len),
    }
    .map_err(|e| format!("PNG image data: {e}"))?;
    if raw.len() < raw_len {
        return Err("PNG image data is shorter than the image".to_string());
    }

    let rgb_len = header
        .width
        .checked_mul(header.height)
        .and_then(|n| n.checked_mul(3))
        .ok_or("PNG image is too large")?;
    let mut rgb = vec![0u8; rgb_len];
    if region != (0, 0, header.width, header.height) && header.color == ColorType::Palette {
        rgb.as_chunks_mut::<3>().0.fill(palette[0]);
    }

    let filter_unit = header.filter_unit();
    let mut at = 0usize;
    for pass in &passes {
        let row_bytes = header
            .row_bytes(pass.width)
            .ok_or("PNG image is too large")?;
        let mut previous: Option<usize> = None;
        for row in 0..pass.height {
            let (before, rest) = raw.split_at_mut(at + 1);
            let filter = before[at];
            let current = &mut rest[..row_bytes];
            let above = previous.map(|start| &before[start..start + row_bytes]);
            unfilter(filter, filter_unit, current, above)?;

            let y = region_y + pass.y0 + row * pass.dy;
            let first = (y * header.width + region_x + pass.x0) * 3;
            let row_end = (y * header.width + region_x + region_w) * 3;
            expand_row(
                &header,
                current,
                pass.width,
                &palette,
                &mut rgb[first..row_end],
                pass.dx,
            );
            previous = Some(at + 1);
            at += 1 + row_bytes;
        }
    }

    Ok(Still {
        width: header.width,
        height: header.height,
        rgb,
        exif: exif.map(|e| e.to_vec()),
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorType {
    Grey,
    Rgb,
    Palette,
    GreyAlpha,
    RgbAlpha,
}

struct Header {
    width: usize,
    height: usize,
    depth: u8,
    color: ColorType,
    interlaced: bool,
}

impl Header {
    fn channels(&self) -> usize {
        match self.color {
            ColorType::Grey | ColorType::Palette => 1,
            ColorType::GreyAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::RgbAlpha => 4,
        }
    }

    fn bits_per_pixel(&self) -> usize {
        self.channels() * usize::from(self.depth)
    }

    /// Bytes of one row of `width` pixels, without the filter byte.
    fn row_bytes(&self, width: usize) -> Option<usize> {
        Some(width.checked_mul(self.bits_per_pixel())?.checked_add(7)? / 8)
    }

    /// Distance in bytes between a byte and the corresponding byte of the pixel to its left,
    /// as the filters define it.
    fn filter_unit(&self) -> usize {
        (self.bits_per_pixel() / 8).max(1)
    }
}

struct Chunk<'a> {
    kind: [u8; 4],
    data: &'a [u8],
    crc_ok: bool,
}

impl Chunk<'_> {
    /// A chunk a decoder must understand: the first letter of its type is upper case.
    fn is_critical(&self) -> bool {
        self.kind[0] & 0x20 == 0
    }

    fn name(&self) -> String {
        String::from_utf8_lossy(&self.kind).into_owned()
    }
}

struct ChunkReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ChunkReader<'a> {
    /// Checks the signature and stands before the first chunk.
    fn new(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.get(..8) != Some(&SIGNATURE[..]) {
            return Err("not a PNG file".to_string());
        }
        Ok(ChunkReader { bytes, pos: 8 })
    }

    /// The next chunk, or `None` at the end of the file.
    fn next(&mut self) -> Result<Option<Chunk<'a>>, String> {
        let cut = || "PNG file is cut short".to_string();
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let head = self.bytes.get(self.pos..self.pos + 8).ok_or_else(cut)?;
        let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
        let kind = [head[4], head[5], head[6], head[7]];
        if !kind.iter().all(u8::is_ascii_alphabetic) {
            return Err("PNG file is damaged (invalid chunk type)".to_string());
        }
        if len > 0x7fff_ffff {
            return Err("PNG file is damaged (invalid chunk length)".to_string());
        }
        let start = self.pos + 8;
        let end = start.checked_add(len as usize).ok_or_else(cut)?;
        let data = self.bytes.get(start..end).ok_or_else(cut)?;
        let crc = self.bytes.get(end..end + 4).ok_or_else(cut)?;
        let stored = u32::from_be_bytes([crc[0], crc[1], crc[2], crc[3]]);
        let computed = !crc32(crc32(!0, &kind), data);
        self.pos = end + 4;
        Ok(Some(Chunk {
            kind,
            data,
            crc_ok: stored == computed,
        }))
    }
}

/// Reads IHDR, which must be the first chunk.
fn read_header(chunks: &mut ChunkReader) -> Result<Header, String> {
    let chunk = match chunks.next()? {
        Some(c) if &c.kind == b"IHDR" => c,
        _ => return Err("PNG file does not start with an IHDR chunk".to_string()),
    };
    if !chunk.crc_ok {
        return Err("PNG file is damaged (bad CRC in IHDR chunk)".to_string());
    }
    let d = chunk.data;
    if d.len() != 13 {
        return Err("PNG file has an invalid IHDR chunk".to_string());
    }
    let width = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
    let height = u32::from_be_bytes([d[4], d[5], d[6], d[7]]);
    if width == 0 || height == 0 || width > 0x7fff_ffff || height > 0x7fff_ffff {
        return Err("PNG file has an invalid image size".to_string());
    }
    let depth = d[8];
    let (color, depths): (ColorType, &[u8]) = match d[9] {
        0 => (ColorType::Grey, &[1, 2, 4, 8, 16]),
        2 => (ColorType::Rgb, &[8, 16]),
        3 => (ColorType::Palette, &[1, 2, 4, 8]),
        4 => (ColorType::GreyAlpha, &[8, 16]),
        6 => (ColorType::RgbAlpha, &[8, 16]),
        other => return Err(format!("PNG file has an unknown colour type ({other})")),
    };
    if !depths.contains(&depth) {
        return Err(format!(
            "PNG file has a bit depth ({depth}) its colour type does not allow"
        ));
    }
    if d[10] != 0 || d[11] != 0 {
        return Err("PNG file uses an unknown compression or filter method".to_string());
    }
    if d[12] > 1 {
        return Err("PNG file uses an unknown interlace method".to_string());
    }
    Ok(Header {
        width: width as usize,
        height: height as usize,
        depth,
        color,
        interlaced: d[12] == 1,
    })
}

/// PLTE as a full table; entries the chunk does not define stay black.
fn read_palette(data: &[u8]) -> Result<[[u8; 3]; 256], String> {
    if data.is_empty() || !data.len().is_multiple_of(3) || data.len() > 768 {
        return Err("PNG file has an invalid PLTE chunk".to_string());
    }
    let mut table = [[0u8; 3]; 256];
    for (entry, rgb) in table.iter_mut().zip(data.as_chunks::<3>().0) {
        *entry = *rgb;
    }
    Ok(table)
}

/// The region a frame-control chunk gives to the image data that follows it.
fn frame_region(
    data: &[u8],
    header: &Header,
) -> Result<Option<(usize, usize, usize, usize)>, String> {
    if data.len() < 26 {
        return Ok(None);
    }
    let field = |at: usize| {
        u64::from(u32::from_be_bytes([
            data[at],
            data[at + 1],
            data[at + 2],
            data[at + 3],
        ]))
    };
    let (w, h, x, y) = (field(4), field(8), field(12), field(16));
    if w == 0 || h == 0 || x + w > header.width as u64 || y + h > header.height as u64 {
        return Err("PNG animation frame does not fit the image".to_string());
    }
    Ok(Some((x as usize, y as usize, w as usize, h as usize)))
}

/// One pass over the image: every `dx`-th pixel of every `dy`-th row, starting at (x0, y0).
struct Pass {
    x0: usize,
    y0: usize,
    dx: usize,
    dy: usize,
    width: usize,
    height: usize,
}

/// The passes that hold pixels: one for a sequential image, up to seven for Adam7.
fn passes(interlaced: bool, width: usize, height: usize) -> Vec<Pass> {
    const ADAM7: [(usize, usize, usize, usize); 7] = [
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ];
    let grid: &[(usize, usize, usize, usize)] = if interlaced { &ADAM7 } else { &[(0, 0, 1, 1)] };
    grid.iter()
        .filter(|&&(x0, y0, _, _)| x0 < width && y0 < height)
        .map(|&(x0, y0, dx, dy)| Pass {
            x0,
            y0,
            dx,
            dy,
            width: (width - x0).div_ceil(dx),
            height: (height - y0).div_ceil(dy),
        })
        .collect()
}

/// Undoes the row filter in place. `above` is the unfiltered previous row of the same pass;
/// the first row of a pass has none, and the filters then read zeros there.
fn unfilter(filter: u8, unit: usize, row: &mut [u8], above: Option<&[u8]>) -> Result<(), String> {
    match (filter, above) {
        (0, _) | (2, None) => {}
        // Without a row above, Paeth predicts the left byte, which is the Sub filter.
        (1, _) | (4, None) => {
            for i in unit..row.len() {
                row[i] = row[i].wrapping_add(row[i - unit]);
            }
        }
        (2, Some(above)) => {
            for (byte, up) in row.iter_mut().zip(above) {
                *byte = byte.wrapping_add(*up);
            }
        }
        (3, None) => {
            for i in unit..row.len() {
                row[i] = row[i].wrapping_add(row[i - unit] >> 1);
            }
        }
        (3, Some(above)) => {
            for i in 0..row.len().min(unit) {
                row[i] = row[i].wrapping_add(above[i] >> 1);
            }
            for i in unit..row.len() {
                let mean = (u16::from(row[i - unit]) + u16::from(above[i])) >> 1;
                row[i] = row[i].wrapping_add(mean as u8);
            }
        }
        (4, Some(above)) => {
            for i in 0..row.len().min(unit) {
                row[i] = row[i].wrapping_add(above[i]);
            }
            for i in unit..row.len() {
                let predicted = paeth(row[i - unit], above[i], above[i - unit]);
                row[i] = row[i].wrapping_add(predicted);
            }
        }
        _ => {
            return Err(format!(
                "PNG image data uses an unknown row filter ({filter})"
            ))
        }
    }
    Ok(())
}

/// The Paeth predictor: whichever of left, above and upper-left is closest to
/// `left + above - upper_left`, ties in that order.
#[inline(always)]
fn paeth(left: u8, above: u8, upper_left: u8) -> u8 {
    let (a, b, c) = (i16::from(left), i16::from(above), i16::from(upper_left));
    let pa = (b - c).abs();
    let pb = (a - c).abs();
    let pc = (a + b - 2 * c).abs();
    if pa <= pb && pa <= pc {
        left
    } else if pb <= pc {
        above
    } else {
        upper_left
    }
}

/// Converts `count` pixels of an unfiltered row to RGB. Pixel `j` goes to `out[j * step * 3..]`,
/// so a sequential row uses step 1 and an interlace pass its column distance.
fn expand_row(
    header: &Header,
    row: &[u8],
    count: usize,
    palette: &[[u8; 3]; 256],
    out: &mut [u8],
    step: usize,
) {
    let pixels = out
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .step_by(step)
        .take(count);
    let depth = usize::from(header.depth);
    // Sample `j` of a row of samples narrower than a byte, most significant bits first.
    let packed = |j: usize| {
        let bit = j * depth;
        (row[bit / 8] >> (8 - depth - bit % 8)) & ((1u16 << depth) - 1) as u8
    };
    match (header.color, header.depth) {
        (ColorType::Rgb, 8) if step == 1 => out[..count * 3].copy_from_slice(&row[..count * 3]),
        (ColorType::Rgb, 8) => {
            for (px, s) in pixels.zip(row.as_chunks::<3>().0) {
                *px = *s;
            }
        }
        (ColorType::RgbAlpha, 8) => {
            for (px, s) in pixels.zip(row.as_chunks::<4>().0) {
                *px = [s[0], s[1], s[2]];
            }
        }
        (ColorType::Rgb, _) => {
            for (px, s) in pixels.zip(row.as_chunks::<6>().0) {
                *px = [s[0], s[2], s[4]];
            }
        }
        (ColorType::RgbAlpha, _) => {
            for (px, s) in pixels.zip(row.as_chunks::<8>().0) {
                *px = [s[0], s[2], s[4]];
            }
        }
        (ColorType::Grey, 8) => {
            for (px, &v) in pixels.zip(row) {
                *px = [v; 3];
            }
        }
        (ColorType::Grey, 16) => {
            for (px, s) in pixels.zip(row.as_chunks::<2>().0) {
                *px = [if s[0] != 0 { 255 } else { s[1] }; 3];
            }
        }
        (ColorType::Grey, _) => {
            let scale = (255 / ((1u16 << depth) - 1)) as u8;
            for (j, px) in pixels.enumerate() {
                *px = [packed(j) * scale; 3];
            }
        }
        (ColorType::GreyAlpha, 8) => {
            for (px, s) in pixels.zip(row.as_chunks::<2>().0) {
                *px = [s[0]; 3];
            }
        }
        (ColorType::GreyAlpha, _) => {
            for (px, s) in pixels.zip(row.as_chunks::<4>().0) {
                *px = [s[0]; 3];
            }
        }
        (ColorType::Palette, 8) => {
            for (px, &v) in pixels.zip(row) {
                *px = palette[usize::from(v)];
            }
        }
        (ColorType::Palette, _) => {
            for (j, px) in pixels.enumerate() {
                *px = palette[usize::from(packed(j))];
            }
        }
    }
}

/// Lookup tables of the CRC-32 PNG uses (polynomial 0xEDB88320, reflected), for eight bytes
/// per step.
static CRC_TABLES: [[u32; 256]; 8] = crc_tables();

const fn crc_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            bit += 1;
        }
        tables[0][i] = c;
        i += 1;
    }
    let mut i = 0;
    while i < 256 {
        let mut t = 1;
        while t < 8 {
            let below = tables[t - 1][i];
            tables[t][i] = (below >> 8) ^ tables[0][(below & 0xff) as usize];
            t += 1;
        }
        i += 1;
    }
    tables
}

/// Continues a CRC-32 over `data`. The register starts at all ones and the result is
/// complemented by the caller.
fn crc32(mut crc: u32, data: &[u8]) -> u32 {
    let (words, rest) = data.as_chunks::<8>();
    for w in words {
        let low = crc ^ u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        crc = CRC_TABLES[7][(low & 0xff) as usize]
            ^ CRC_TABLES[6][((low >> 8) & 0xff) as usize]
            ^ CRC_TABLES[5][((low >> 16) & 0xff) as usize]
            ^ CRC_TABLES[4][(low >> 24) as usize]
            ^ CRC_TABLES[3][usize::from(w[4])]
            ^ CRC_TABLES[2][usize::from(w[5])]
            ^ CRC_TABLES[1][usize::from(w[6])]
            ^ CRC_TABLES[0][usize::from(w[7])];
    }
    for &byte in rest {
        crc = CRC_TABLES[0][((crc ^ u32::from(byte)) & 0xff) as usize] ^ (crc >> 8);
    }
    crc
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
                    "/tests/data/vision/png/",
                    $name
                )) as &[u8],
                $width,
                $height,
                $hash,
            ),)*]
        };
    }

    /// Name, file, width, height, and the FNV-1a 64 hash of the RGB bytes the reference image
    /// library returns for the file. `tests/data/vision/png/make_fixtures.py` writes the files
    /// and prints these lines.
    const FIXTURES: &[(&str, &[u8], usize, usize, u64)] = fixtures![
        ("grey1.png", 17, 13, 0xfbae8fa6eca8a7b3),
        ("grey2.png", 17, 13, 0xe1fa709febec013b),
        ("grey4.png", 17, 13, 0x527b5b0667f2eada),
        ("grey8.png", 17, 13, 0x0ae47574f73b4c37),
        ("grey16.png", 17, 13, 0x5387877ed7a35cd4),
        ("rgb8.png", 17, 13, 0x5b2b94c5f656b9f0),
        ("rgb16.png", 17, 13, 0x83d648ebd56d8c5c),
        ("greya8.png", 17, 13, 0xd755e87fe108ca37),
        ("greya16.png", 17, 13, 0x13f67bd7d35fdde3),
        ("rgba8.png", 17, 13, 0x66d2efefb34acc41),
        ("rgba16.png", 17, 13, 0x59a1bda4aa76aa8b),
        ("pal1.png", 17, 13, 0x74f39e26f854db57),
        ("pal2.png", 17, 13, 0x84d8c8d951d40835),
        ("pal4.png", 17, 13, 0x26a98d2d33f7ef2a),
        ("pal8.png", 17, 13, 0xe737253cd43002dd),
        ("grey1_adam7.png", 17, 13, 0xb8cbc739d54d61d9),
        ("grey16_adam7.png", 17, 13, 0x8bdfb055759476bf),
        ("rgb8_adam7.png", 17, 13, 0x2f396971f353eb93),
        ("pal2_adam7.png", 17, 13, 0x89ae03c597c4c495),
        ("pal4_adam7.png", 17, 13, 0xcfa19a543d222a6e),
        ("greya8_adam7.png", 17, 13, 0x77d169e76d7be798),
        ("rgba16_adam7.png", 17, 13, 0x6f700d8448373dc6),
        ("rgb8_adam7_1x7.png", 1, 7, 0x5d3e0ed2fc95ef19),
        ("rgb8_adam7_3x2.png", 3, 2, 0xc6ad75beca3a8150),
        ("grey8_1x1.png", 1, 1, 0x52dc9e18b0feba61),
        ("pal1_7x1.png", 7, 1, 0xf6dda29a13dfc40b),
        ("grey8_trns.png", 17, 13, 0x2af58cf3710839d6),
        ("rgb16_trns.png", 17, 13, 0x40f144bb72ee38b2),
        ("pal8_trns.png", 17, 13, 0xac0faa6604f83b32),
        ("pal8_short.png", 17, 13, 0x195a0d2eeb8655fb),
        ("pal4_short.png", 17, 13, 0x14d56e5c590acf53),
        ("grey16_clip.png", 79, 1, 0xc85e390ec87a115d),
        ("rgb8_idat_pieces.png", 17, 13, 0xd519e90606d37a0c),
        ("rgba8_stored.png", 17, 13, 0x08b51664b8e17b84),
        ("rgb8_fixed.png", 17, 13, 0x4e81b1f664237742),
        ("grey8_long_data.png", 17, 13, 0x6c064319b34bba38),
        ("rgb8_ancillary.png", 17, 13, 0xd68a3d24412a1e36),
        ("exif.png", 17, 13, 0x36c4af4f48c38c01),
        ("apng_first_frame.png", 17, 13, 0x9aa4f1aa408c3003),
        ("apng_default_image.png", 17, 13, 0xb17bfb17be787df9),
        ("apng_region_pal4.png", 17, 13, 0xc173d0109e7d8e44),
        ("apng_region_rgb8.png", 17, 13, 0x9bcc225367452de7),
        ("apng_region_adam7.png", 17, 13, 0x655527d306e4130e),
    ];

    /// The EXIF block `make_fixtures.py` puts into exif.png.
    const EXIF: &[u8] = b"MM\x00*\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00\x00\x00\x00\x00";

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

    /// The chunks of a file as (type, data).
    fn chunks_of(bytes: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
        let mut reader = ChunkReader::new(bytes).unwrap();
        let mut out = Vec::new();
        while let Some(chunk) = reader.next().unwrap() {
            out.push((chunk.kind, chunk.data.to_vec()));
        }
        out
    }

    /// A file made of the given chunks, each with a correct CRC.
    fn file_of(chunks: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut out = SIGNATURE.to_vec();
        for (kind, data) in chunks {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(kind);
            out.extend_from_slice(data);
            out.extend_from_slice(&(!crc32(crc32(!0, kind), data)).to_be_bytes());
        }
        out
    }

    /// A fixture with its chunk list edited and every CRC made right again.
    fn edited(name: &str, edit: impl FnOnce(&mut Vec<([u8; 4], Vec<u8>)>)) -> Vec<u8> {
        let mut chunks = chunks_of(fixture(name));
        edit(&mut chunks);
        file_of(&chunks)
    }

    /// `raw` as a zlib stream of stored blocks.
    fn zlib_stored(raw: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        let mut blocks = raw.chunks(65535).peekable();
        if blocks.peek().is_none() {
            out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
        }
        while let Some(block) = blocks.next() {
            out.push(u8::from(blocks.peek().is_none()));
            out.extend_from_slice(&(block.len() as u16).to_le_bytes());
            out.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
            out.extend_from_slice(block);
        }
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in raw {
            a = (a + u32::from(byte)) % 65521;
            b = (b + a) % 65521;
        }
        out.extend_from_slice(&((b << 16) | a).to_be_bytes());
        out
    }

    /// A file with the given header fields and filtered image data.
    fn png_of(width: u32, height: u32, depth: u8, color: u8, interlace: u8, raw: &[u8]) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(&width.to_be_bytes());
        header.extend_from_slice(&height.to_be_bytes());
        header.extend_from_slice(&[depth, color, 0, 0, interlace]);
        file_of(&[
            (*b"IHDR", header),
            (*b"IDAT", zlib_stored(raw)),
            (*b"IEND", Vec::new()),
        ])
    }

    #[test]
    fn fixtures_match_the_reference() {
        for &(name, bytes, width, height, hash) in FIXTURES {
            let still = decode(bytes, &LIMITS).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((still.width, still.height), (width, height), "{name}");
            assert_eq!(still.rgb.len(), width * height * 3, "{name}");
            assert_eq!(fnv1a64(&still.rgb), hash, "{name}");
            assert_eq!(probe(bytes), Ok((width, height)), "{name}");
            let exif = (name == "exif.png").then(|| EXIF.to_vec());
            assert_eq!(still.exif, exif, "{name}");
        }
    }

    #[test]
    fn sample_rules() {
        let rgb = |file: &[u8]| decode(file, &LIMITS).unwrap().rgb;
        // 16-bit greyscale is clipped at 255.
        let raw = [0, 0x00, 0x07, 0x00, 0xff, 0x01, 0x00, 0xff, 0xff];
        assert_eq!(
            rgb(&png_of(4, 1, 16, 0, 0, &raw)),
            [7, 7, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        );
        // 16-bit truecolour and 16-bit greyscale with alpha keep the high byte.
        let raw = [0, 0x12, 0x34, 0x00, 0xff, 0xfe, 0x01];
        assert_eq!(rgb(&png_of(1, 1, 16, 2, 0, &raw)), [0x12, 0x00, 0xfe]);
        let raw = [0, 0x01, 0xff, 0x00, 0x00, 0xfe, 0x00, 0xff, 0xff];
        assert_eq!(
            rgb(&png_of(2, 1, 16, 4, 0, &raw)),
            [1, 1, 1, 0xfe, 0xfe, 0xfe]
        );
        // Greyscale of 1, 2 and 4 bits scales to the full range.
        assert_eq!(
            rgb(&png_of(2, 1, 1, 0, 0, &[0, 0b0100_0000])),
            [0, 0, 0, 255, 255, 255]
        );
        assert_eq!(
            rgb(&png_of(4, 1, 2, 0, 0, &[0, 0b0001_1011])),
            [0, 0, 0, 85, 85, 85, 170, 170, 170, 255, 255, 255]
        );
        assert_eq!(
            rgb(&png_of(2, 1, 4, 0, 0, &[0, 0x1e])),
            [17, 17, 17, 238, 238, 238]
        );
        // Alpha is dropped: a fully transparent pixel keeps its colour.
        assert_eq!(
            rgb(&png_of(1, 1, 8, 6, 0, &[0, 10, 20, 30, 0])),
            [10, 20, 30]
        );
        assert_eq!(rgb(&png_of(1, 1, 8, 4, 0, &[0, 99, 0])), [99, 99, 99]);
    }

    #[test]
    fn truncated_files_are_refused() {
        for name in [
            "rgb8.png",
            "pal4_adam7.png",
            "rgb8_idat_pieces.png",
            "exif.png",
        ] {
            let bytes = fixture(name);
            for len in 0..bytes.len() {
                assert!(
                    decode(&bytes[..len], &LIMITS).is_err(),
                    "{name} cut at {len}"
                );
            }
        }
        assert!(probe(&fixture("rgb8.png")[..32]).is_err());
        assert_eq!(probe(&fixture("rgb8.png")[..33]), Ok((17, 13)));
    }

    #[test]
    fn damaged_chunks_are_refused() {
        let good = fixture("rgb8_ancillary.png");
        let pixels = decode(good, &LIMITS).unwrap().rgb;
        // Positions of each chunk's CRC.
        let mut crc_at = Vec::new();
        let mut reader = ChunkReader::new(good).unwrap();
        while let Some(chunk) = reader.next().unwrap() {
            crc_at.push((chunk.kind, reader.pos - 4));
        }
        for (kind, at) in crc_at {
            let mut bad = good.to_vec();
            bad[at] ^= 0x40;
            let result = decode(&bad, &LIMITS);
            if kind[0].is_ascii_uppercase() {
                let message = result.unwrap_err();
                assert!(message.contains("CRC"), "{message}");
            } else {
                // An ancillary chunk with a bad CRC is skipped.
                assert_eq!(result.unwrap().rgb, pixels);
            }
        }
        let mut bad = good.to_vec();
        bad[1] = b'p';
        assert_eq!(decode(&bad, &LIMITS).unwrap_err(), "not a PNG file");
        assert!(probe(&bad).is_err());
        assert!(decode(b"", &LIMITS).is_err());
        // A chunk length that reaches past the end of the file.
        let mut bad = good.to_vec();
        bad[33..37].copy_from_slice(&0x7fff_ffffu32.to_be_bytes());
        assert!(decode(&bad, &LIMITS).is_err());
        // An EXIF chunk with a bad CRC is not returned.
        let exif = fixture("exif.png");
        let at = exif.windows(4).position(|w| w == b"eXIf").unwrap();
        let mut bad = exif.to_vec();
        bad[at + 4] ^= 1;
        assert_eq!(decode(&bad, &LIMITS).unwrap().exif, None);
    }

    #[test]
    fn chunk_order_is_enforced() {
        let refused = |file: Vec<u8>| decode(&file, &LIMITS).unwrap_err();
        let ihdr = |c: &mut Vec<([u8; 4], Vec<u8>)>| c[0].clone();
        // IHDR missing, not first, or repeated.
        refused(edited("rgb8.png", |c| {
            c.remove(0);
        }));
        refused(edited("rgb8.png", |c| c.swap(0, 1)));
        refused(edited("rgb8.png", |c| {
            let again = ihdr(c);
            c.insert(1, again);
        }));
        // A palette image without PLTE, with two, or with PLTE after the image data.
        refused(edited("pal8.png", |c| {
            c.remove(1);
        }));
        refused(edited("pal8.png", |c| {
            let again = c[1].clone();
            c.insert(1, again);
        }));
        refused(edited("pal8.png", |c| c.swap(1, 2)));
        // PLTE of a wrong size.
        refused(edited("pal8.png", |c| {
            c[1].1.pop();
        }));
        refused(edited("pal8.png", |c| c[1].1.clear()));
        refused(edited("pal8.png", |c| c[1].1.resize(771, 0)));
        // No image data, and image data split by another chunk.
        refused(edited("rgb8.png", |c| {
            c.remove(1);
        }));
        refused(edited("rgb8_idat_pieces.png", |c| {
            c.insert(3, (*b"tEXt", b"a\0b".to_vec()));
        }));
        // No IEND, and a critical chunk the decoder does not know.
        refused(edited("rgb8.png", |c| {
            c.pop();
        }));
        refused(edited("rgb8.png", |c| {
            c.insert(1, (*b"ABCD", vec![1, 2, 3]));
        }));
        // An unknown ancillary chunk anywhere else changes nothing.
        let with_note = edited("rgb8.png", |c| c.insert(1, (*b"abCd", vec![1, 2, 3])));
        assert_eq!(
            decode(&with_note, &LIMITS).unwrap(),
            decode(fixture("rgb8.png"), &LIMITS).unwrap()
        );
        // The last EXIF chunk wins, also after the image data.
        let two = edited("exif.png", |c| {
            let at = c.len() - 1;
            c.insert(at, (*b"eXIf", b"II*\0".to_vec()));
        });
        assert_eq!(decode(&two, &LIMITS).unwrap().exif.unwrap(), b"II*\0");
    }

    #[test]
    fn header_values_are_checked() {
        let with_header = |at: usize, value: u8| edited("rgb8.png", |c| c[0].1[at] = value);
        for (at, value) in [
            (8, 3),
            (8, 4),
            (8, 0),
            (9, 1),
            (9, 5),
            (10, 1),
            (11, 1),
            (12, 2),
        ] {
            let file = with_header(at, value);
            assert!(decode(&file, &LIMITS).is_err(), "IHDR byte {at} = {value}");
            assert!(probe(&file).is_err(), "IHDR byte {at} = {value}");
        }
        // Zero and oversized dimensions, and an IHDR of the wrong length.
        for size in [[0u8, 0, 0, 0], [0x80, 0, 0, 0]] {
            let file = edited("rgb8.png", |c| c[0].1[..4].copy_from_slice(&size));
            assert!(decode(&file, &LIMITS).is_err());
            let file = edited("rgb8.png", |c| c[0].1[4..8].copy_from_slice(&size));
            assert!(decode(&file, &LIMITS).is_err());
        }
        assert!(decode(&edited("rgb8.png", |c| c[0].1.push(0)), &LIMITS).is_err());
    }

    #[test]
    fn canvas_limit_is_enforced() {
        let bytes = fixture("rgb8.png");
        let tight = DecodeLimits {
            max_pixels: 17 * 13 - 1,
        };
        assert!(decode(bytes, &tight).unwrap_err().contains("limit"));
        let exact = DecodeLimits {
            max_pixels: 17 * 13,
        };
        assert!(decode(bytes, &exact).is_ok());
        // A header that announces far more than the file holds is refused by the limit, and
        // within the limit by the missing data, without a buffer of the announced size.
        let huge = png_of(0x7fff_ffff, 0x7fff_ffff, 16, 6, 0, &[0; 9]);
        assert!(decode(&huge, &LIMITS).unwrap_err().contains("limit"));
        assert_eq!(probe(&huge), Ok((0x7fff_ffff, 0x7fff_ffff)));
        let unlimited = DecodeLimits {
            max_pixels: u64::MAX,
        };
        assert!(decode(&png_of(1 << 20, 1 << 20, 16, 6, 0, &[0; 9]), &unlimited).is_err());
        assert!(decode(&png_of(1 << 20, 1 << 20, 8, 2, 1, &[0; 9]), &unlimited).is_err());
    }

    #[test]
    fn image_data_must_cover_the_image() {
        // Two rows of two grey pixels.
        let raw = [0, 10, 20, 2, 5, 5];
        let rgb = decode(&png_of(2, 2, 8, 0, 0, &raw), &LIMITS).unwrap().rgb;
        assert_eq!(rgb, [10, 10, 10, 20, 20, 20, 15, 15, 15, 25, 25, 25]);
        // Shorter data is refused, also when it ends on a row boundary.
        for len in 0..raw.len() {
            assert!(decode(&png_of(2, 2, 8, 0, 0, &raw[..len]), &LIMITS).is_err());
        }
        // Longer data is cut.
        let long = [&raw[..], &[9; 40]].concat();
        assert_eq!(
            decode(&png_of(2, 2, 8, 0, 0, &long), &LIMITS).unwrap().rgb,
            rgb
        );
        // A row filter the format does not define.
        let bad_filter = [0, 10, 20, 5, 5, 5];
        let message = decode(&png_of(2, 2, 8, 0, 0, &bad_filter), &LIMITS).unwrap_err();
        assert!(message.contains("row filter"), "{message}");
        // A damaged zlib stream: the checksum of the image data is verified.
        let mut file = chunks_of(&png_of(2, 2, 8, 0, 0, &raw));
        let last = file[1].1.len() - 1;
        file[1].1[last] ^= 1;
        assert!(decode(&file_of(&file), &LIMITS).is_err());
    }

    #[test]
    fn frame_control_gives_the_region_of_the_image_data() {
        // A 4x3 canvas whose image data covers 2x1 at (1, 2).
        let mut control = vec![0u8; 26];
        for (at, value) in [(4, 2u32), (8, 1), (12, 1), (16, 2)] {
            control[at..at + 4].copy_from_slice(&value.to_be_bytes());
        }
        let mut chunks = chunks_of(&png_of(4, 3, 8, 0, 0, &[0, 50, 60]));
        chunks.insert(1, (*b"fcTL", control.clone()));
        let rgb = decode(&file_of(&chunks), &LIMITS).unwrap().rgb;
        let mut expected = [0u8; 36];
        expected[27..30].fill(50);
        expected[30..33].fill(60);
        assert_eq!(rgb, expected);
        // A region outside the canvas is refused.
        control[12..16].copy_from_slice(&3u32.to_be_bytes());
        chunks[1].1 = control;
        assert!(decode(&file_of(&chunks), &LIMITS).is_err());
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
            // Damage anywhere in the file; most of it is caught by a chunk's CRC.
            for at in 0..bytes.len() {
                let mut bad = bytes.to_vec();
                bad[at] ^= 1 << (next() % 8);
                let _ = decode(&bad, &LIMITS);
            }
            // Damage inside the chunks with the CRCs made right, to reach the code behind them.
            let chunks = chunks_of(bytes);
            for index in 0..chunks.len() {
                for at in 0..chunks[index].1.len() {
                    let mut bad = chunks.clone();
                    let r = next();
                    bad[index].1[at] = match r % 4 {
                        0 => bad[index].1[at] ^ (1 << ((r >> 8) % 8)),
                        1 => (r >> 8) as u8,
                        2 => [0, 1, 2, 3, 4, 8, 16, 127, 128, 255][(r >> 8) as usize % 10],
                        _ => bad[index].1[at].wrapping_add(1),
                    };
                    if r % 16 == 5 {
                        bad[index].1.truncate(at);
                    }
                    let file = file_of(&bad);
                    let _ = decode(&file, &LIMITS);
                    let _ = probe(&file);
                }
            }
        }
    }

    /// Compares every `<name>.png` in the directory `GAUDIUM_IMAGE_REF_DIR` with `<name>.rgb`,
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
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
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
        assert!(compared > 0, "no <name>.png with <name>.rgb in {dir}");
    }
}
