//! JPEG decoder (ITU T.81 with the JFIF and Adobe conventions), bit-exact with the reference
//! image library.
//!
//! Decodes: baseline and extended sequential Huffman frames (SOF0, SOF1) and progressive
//! Huffman frames (SOF2) with 8-bit samples; 1 component (greyscale), 3 components (YCbCr, or
//! RGB when the Adobe transform or the component ids say so) and 4 components (CMYK, YCCK);
//! every sampling layout whose factors divide the largest one; any scan order and
//! interleaving; restart intervals; 8- and 16-bit quantization tables; the standard Huffman
//! tables for sequential files that carry none. Only the first picture of a file is read (the
//! primary image of a multi-picture file); thumbnails live inside APPn segments and are
//! skipped with them.
//!
//! Refuses with an error: arithmetic coding, lossless and hierarchical frames, sample
//! precisions other than 8 bits, 2 or more than 4 components, sampling factors that do not
//! divide the largest factor, files without an end-of-image marker, entropy-coded data that is
//! corrupt or ends early, restart markers out of sequence, and progressive files whose
//! low-frequency coefficients were not all sent to full precision (the reference decoder
//! blurs those with inter-block smoothing, which is not reproduced).
//!
//! Rules that make the output equal the reference decoder's, byte for byte:
//!
//! * Inverse DCT: the accurate integer ("ISLOW") transform with 13-bit constants and two extra
//!   bits kept after the column pass, as the reference decoder's vectorized transform
//!   computes it. For the coefficients an encoder produces this is the plain 32-bit transform
//!   with a clamp to 0..=255. Beyond that range, dequantized values and the pair sums that
//!   feed the multipliers wrap to 16 bits, the column pass saturates to 16 bits (a block with
//!   only its first row present wraps instead), and the row pass saturates to the sample range.
//! * Chroma upsampling: the triangle ("fancy") filter for components at half horizontal
//!   resolution when they are more than 2 samples wide (3/4 near + 1/4 far, rounding bias
//!   alternating 1, 2), for half vertical resolution at any width (bias 1 above, 2 below), and
//!   the 9/3/3/1 filter for half resolution in both directions when more than 2 samples wide
//!   (bias 8, 7). Every other ratio, and the narrow cases, replicate samples. Rows and columns
//!   outside the component's real extent repeat the edge sample.
//! * Colour: 16-bit fixed-point YCbCr to RGB with the reference rounding; CMYK and YCCK go
//!   through the reference library's inverted-ink convention and its CMYK to RGB arithmetic.
//!   A 3-component file is YCbCr when it has a JFIF segment, else what its Adobe segment
//!   says, else RGB only for the component ids 'R', 'G', 'B'.
//! * A quantization table belongs to a component from the first scan that names it; later
//!   redefinitions do not change it. Sequential scans after the first one only overwrite the
//!   coefficients they code as nonzero. A zero run that passes the end of a block puts its
//!   coefficient in the last position.

use super::types::{DecodeLimits, Still};

/// Zigzag position to row-major coefficient index.
const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Largest width or height the reference decoder accepts.
const MAX_DIMENSION: usize = 65500;
/// Most blocks one interleaved MCU may hold.
const MAX_BLOCKS_IN_MCU: usize = 10;
/// Most scans read from one file. Real progressive files have about ten; the cap bounds the
/// work a small hostile file can ask for, since every scan visits every block.
const MAX_SCANS: usize = 512;

const TRUNCATED: &str = "JPEG file is truncated (no end-of-image marker)";
const BAD_ENTROPY: &str = "JPEG entropy-coded data is corrupt or ends early";

// Standard Huffman tables of ITU T.81 Annex K, used when a file defines none.
const STD_DC_LUMA_COUNTS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const STD_DC_CHROMA_COUNTS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const STD_DC_SYMBOLS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const STD_AC_LUMA_COUNTS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];
const STD_AC_LUMA_SYMBOLS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];
const STD_AC_CHROMA_COUNTS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119];
const STD_AC_CHROMA_SYMBOLS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Width and height from the frame header, without decoding.
pub fn probe(bytes: &[u8]) -> Result<(usize, usize), String> {
    let mut parser = Parser::new(bytes)?;
    loop {
        let marker = parser.next_marker()?;
        match marker {
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                let frame = parser.frame_header(marker)?;
                return Ok((frame.width, frame.height));
            }
            0xDA => return Err("JPEG has a scan before its frame header".into()),
            0xD9 => return Err("JPEG file holds no image".into()),
            _ => parser.segment(marker)?,
        }
    }
}

/// Decodes the image to RGB8. `Still::exif` is the payload of the first APP1 "Exif\0\0"
/// segment (the TIFF structure after those 6 bytes), not applied.
pub fn decode(bytes: &[u8], limits: &DecodeLimits) -> Result<Still, String> {
    let mut parser = Parser::new(bytes)?;
    let mut frame: Option<Frame> = None;
    loop {
        let marker = parser.next_marker()?;
        match marker {
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                if frame.is_some() {
                    return Err("JPEG has more than one frame header".into());
                }
                let header = parser.frame_header(marker)?;
                limits.check_canvas(header.width, header.height)?;
                frame = Some(header);
            }
            0xDA => {
                let Some(frame) = frame.as_mut() else {
                    return Err("JPEG has a scan before its frame header".into());
                };
                parser.scan(frame)?;
            }
            0xD9 => {
                let Some(frame) = frame else {
                    return Err("JPEG file holds no image".into());
                };
                if frame.scans == 0 {
                    return Err("JPEG file holds no image".into());
                }
                return finish(frame, parser.exif.take());
            }
            _ => parser.segment(marker)?,
        }
    }
}

/// Allocates `len` default elements, reporting exhaustion instead of aborting.
fn buffer<T: Clone + Default>(len: usize) -> Result<Vec<T>, String> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| "not enough memory to decode the JPEG image".to_string())?;
    v.resize(len, T::default());
    Ok(v)
}

/// How the components map to RGB.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    Gray,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
}

/// A Huffman table as a DHT segment gives it.
#[derive(Clone)]
struct HuffSpec {
    counts: [u8; 16],
    symbols: [u8; 256],
}

impl HuffSpec {
    fn standard(counts: &[u8; 16], symbols: &[u8]) -> Self {
        let mut spec = HuffSpec {
            counts: *counts,
            symbols: [0; 256],
        };
        for (dst, src) in spec.symbols.iter_mut().zip(symbols) {
            *dst = *src;
        }
        spec
    }
}

/// Decoding form of a Huffman table.
struct Huff {
    /// Indexed by the next 9 bits: `(length << 8) | symbol`, or 0 for codes longer than 9 bits.
    fast: [u16; 512],
    /// Per code length, one past its last code, left-aligned to 16 bits.
    bound: [u32; 17],
    /// Per code length, index of its first symbol minus its first code.
    offset: [i32; 17],
    symbols: [u8; 256],
}

impl Huff {
    /// Builds the canonical code. Refuses the tables the reference decoder refuses: a code
    /// space that is over-full or uses the all-ones code, and DC symbols above 15.
    fn build(spec: &HuffSpec, is_dc: bool) -> Result<Box<Huff>, String> {
        let bad = || "JPEG has an invalid Huffman table".to_string();
        let mut table = Box::new(Huff {
            fast: [0; 512],
            bound: [0; 17],
            offset: [0; 17],
            symbols: spec.symbols,
        });
        let mut code: u32 = 0;
        let mut next: usize = 0;
        for len in 1..=16usize {
            let count = spec.counts[len - 1] as u32;
            if count > 0 && code + count >= (1u32 << len) {
                return Err(bad());
            }
            table.offset[len] = next as i32 - code as i32;
            for _ in 0..count {
                let symbol = *spec.symbols.get(next).ok_or_else(bad)?;
                if is_dc && symbol > 15 {
                    return Err(bad());
                }
                if len <= 9 {
                    let shift = 9 - len;
                    let first = (code as usize) << shift;
                    let entry = ((len as u16) << 8) | symbol as u16;
                    if let Some(slots) = table.fast.get_mut(first..first + (1 << shift)) {
                        slots.fill(entry);
                    }
                }
                code += 1;
                next += 1;
            }
            table.bound[len] = code << (16 - len);
            code <<= 1;
        }
        Ok(table)
    }
}

/// Bit reader over entropy-coded data: removes the zero byte stuffed after 0xFF and stops at
/// the first marker.
struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte that is not in `buf` yet.
    pos: usize,
    /// Unread bits, left-aligned; the bits below them are zero.
    buf: u64,
    /// Count of unread bits. Negative once more bits were taken than the data holds.
    bits: i32,
    /// A marker or the end of the data was reached; nothing more is loaded.
    stopped: bool,
    /// A bit pattern that is no Huffman code, or a refinement symbol of a size other than 1.
    corrupt: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            buf: 0,
            bits: 0,
            stopped: false,
            corrupt: false,
        }
    }

    /// Makes at least 32 bits available unless the data ends first. One call covers a Huffman
    /// code (at most 16 bits) and the value bits after it (at most 16).
    #[inline(always)]
    fn ensure(&mut self) {
        if self.bits < 32 {
            self.refill();
        }
    }

    fn refill(&mut self) {
        if self.stopped || self.bits < 0 {
            return;
        }
        // Fast path: the next 8 bytes hold no 0xFF, so whole bytes load at once.
        if let Some(Ok(chunk)) = self
            .data
            .get(self.pos..self.pos.wrapping_add(8))
            .map(<[u8; 8]>::try_from)
        {
            let word = u64::from_be_bytes(chunk);
            let inverted = !word;
            let has_ff =
                inverted.wrapping_sub(0x0101_0101_0101_0101) & word & 0x8080_8080_8080_8080;
            if has_ff == 0 {
                let take = (64 - self.bits as u32) / 8;
                let word = if take >= 8 {
                    word
                } else {
                    word & !(u64::MAX >> (take * 8))
                };
                self.buf |= word >> self.bits as u32;
                self.bits += (take * 8) as i32;
                self.pos += take as usize;
                return;
            }
        }
        while self.bits <= 56 {
            let Some(&byte) = self.data.get(self.pos) else {
                self.stopped = true;
                return;
            };
            if byte == 0xFF {
                // Fill bytes may precede the byte that says whether this is data or a marker.
                let mut next = self.pos + 1;
                while self.data.get(next) == Some(&0xFF) {
                    next += 1;
                }
                if self.data.get(next) != Some(&0) {
                    self.stopped = true;
                    return;
                }
                self.pos = next + 1;
            } else {
                self.pos += 1;
            }
            self.buf |= (byte as u64) << (56 - self.bits as u32);
            self.bits += 8;
        }
    }

    #[inline(always)]
    fn consume(&mut self, count: u32) {
        self.buf <<= count;
        self.bits -= count as i32;
    }

    /// Decodes one Huffman symbol. An invalid code marks the stream corrupt and reads as 0.
    #[inline(always)]
    fn symbol(&mut self, table: &Huff) -> u32 {
        let entry = table.fast[(self.buf >> 55) as usize];
        if entry != 0 {
            self.consume((entry >> 8) as u32);
            return (entry & 0xFF) as u32;
        }
        self.long_symbol(table)
    }

    fn long_symbol(&mut self, table: &Huff) -> u32 {
        let window = (self.buf >> 48) as u32;
        for len in 10..=16usize {
            if window < table.bound[len] {
                let code = (window >> (16 - len)) as i32;
                let index = code.wrapping_add(table.offset[len]) as usize & 0xFF;
                self.consume(len as u32);
                return table.symbols[index] as u32;
            }
        }
        self.corrupt = true;
        0
    }

    /// Reads `count` (1..=16) bits as an unsigned number.
    #[inline(always)]
    fn take(&mut self, count: u32) -> u32 {
        let value = (self.buf >> (64 - count)) as u32;
        self.consume(count);
        value
    }

    /// Reads a `size`-bit (1..=15) signed value in the JPEG magnitude-category encoding.
    #[inline(always)]
    fn extend(&mut self, size: u32) -> i32 {
        let value = self.take(size) as i32;
        if value < (1 << (size - 1)) {
            value - (1 << size) + 1
        } else {
            value
        }
    }

    #[inline(always)]
    fn bit(&mut self) -> bool {
        self.take(1) != 0
    }

    /// Fails when the data read so far was invalid or more bits were taken than it holds.
    #[inline(always)]
    fn check(&self) -> Result<(), String> {
        if self.bits < 0 || self.corrupt {
            Err(BAD_ENTROPY.into())
        } else {
            Ok(())
        }
    }

    /// Drops the rest of the current interval and steps over the restart marker `index`.
    fn restart(&mut self, index: u8) -> Result<(), String> {
        let (marker, after) = find_marker(self.data, self.pos).ok_or(BAD_ENTROPY)?;
        if marker != 0xD0 + index {
            return Err("JPEG restart markers are out of sequence".into());
        }
        self.pos = after;
        self.buf = 0;
        self.bits = 0;
        self.stopped = false;
        Ok(())
    }
}

/// Finds the next marker at or after `pos`: skips bytes that are not 0xFF, fill bytes and
/// stuffed zeros. Returns the marker code and the position after it.
fn find_marker(data: &[u8], mut pos: usize) -> Option<(u8, usize)> {
    loop {
        let rest = data.get(pos..)?;
        pos += rest.iter().position(|&b| b == 0xFF)?;
        while data.get(pos) == Some(&0xFF) {
            pos += 1;
        }
        let code = *data.get(pos)?;
        pos += 1;
        if code != 0 {
            return Some((code, pos));
        }
    }
}

struct Component {
    /// Component id as the scan headers name it.
    id: u8,
    h: usize,
    v: usize,
    /// Quantization table slot.
    tq: usize,
    /// Output samples per component sample, horizontally and vertically.
    h_expand: usize,
    v_expand: usize,
    /// Sample columns and rows that carry image data.
    data_w: usize,
    data_h: usize,
    /// Blocks that cover `data_w` x `data_h`; a scan of this component alone codes these.
    blocks_w: usize,
    blocks_h: usize,
    /// Blocks per row and per column when the component is padded to whole MCUs.
    mcu_blocks_w: usize,
    mcu_blocks_h: usize,
    /// Quantization table in row-major order, taken at the component's first scan.
    quant: [u16; 64],
    quant_latched: bool,
    /// Progressive state per zigzag position: -1 before the coefficient is sent, else the
    /// number of low bits still missing.
    missing_bits: [i8; 64],
    /// Decoded samples, `blocks_w * 8` per row.
    plane: Vec<u8>,
    /// Coefficients in row-major order per block, `mcu_blocks_w` blocks per row. Used when
    /// the image takes more than one scan.
    coefs: Vec<i16>,
}

struct Frame {
    width: usize,
    height: usize,
    progressive: bool,
    mcus_x: usize,
    mcus_y: usize,
    comps: Vec<Component>,
    color: Color,
    scans: usize,
    /// The first scan carries every component of a sequential frame, so blocks are
    /// transformed as they are read and no further scan may follow.
    single_scan: bool,
}

/// One component's part in a scan.
struct ScanComp {
    index: usize,
    /// Positions in the scan's table lists.
    dc: usize,
    ac: usize,
    /// Blocks per MCU, horizontally and vertically.
    h: usize,
    v: usize,
    pred: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScanKind {
    Sequential,
    DcFirst,
    DcRefine,
    AcFirst,
    AcRefine,
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    quant: [Option<[u16; 64]>; 4],
    dc_specs: [Option<HuffSpec>; 4],
    ac_specs: [Option<HuffSpec>; 4],
    restart_interval: usize,
    jfif: bool,
    adobe_transform: Option<u8>,
    exif: Option<Vec<u8>>,
    /// A scan header was read. Before that the reference library's own marker walk applies
    /// too, which is stricter about a few segments.
    in_scans: bool,
}

impl<'a> Parser<'a> {
    fn new(data: &'a [u8]) -> Result<Self, String> {
        if !data.starts_with(&[0xFF, 0xD8]) {
            if data.starts_with(&[0xFF, 0x4F, 0xFF, 0x51])
                || data.starts_with(&[0, 0, 0, 0x0C, 0x6A, 0x50, 0x20, 0x20])
            {
                return Err("JPEG 2000 images are not supported".into());
            }
            if data.starts_with(&[0xFF, 0x0A])
                || data.starts_with(&[0, 0, 0, 0x0C, 0x4A, 0x58, 0x4C, 0x20])
            {
                return Err("JPEG XL images are not supported".into());
            }
            return Err("not a JPEG file (no start-of-image marker)".into());
        }
        // The reference library only takes files whose first marker follows directly.
        if data.get(2).is_some_and(|&b| b != 0xFF) {
            return Err("not a JPEG file (no marker after the start-of-image marker)".into());
        }
        Ok(Parser {
            data,
            pos: 2,
            quant: [None; 4],
            dc_specs: [None, None, None, None],
            ac_specs: [None, None, None, None],
            restart_interval: 0,
            jfif: false,
            adobe_transform: None,
            exif: None,
            in_scans: false,
        })
    }

    fn next_marker(&mut self) -> Result<u8, String> {
        let (marker, after) = find_marker(self.data, self.pos).ok_or(TRUNCATED)?;
        self.pos = after;
        Ok(marker)
    }

    /// Takes the body of a marker segment (after its 2-byte length).
    fn body(&mut self) -> Result<&'a [u8], String> {
        let len = self
            .data
            .get(self.pos..self.pos + 2)
            .map(|b| ((b[0] as usize) << 8) | b[1] as usize)
            .ok_or(TRUNCATED)?;
        if len < 2 {
            return Err("JPEG marker segment has an invalid length".into());
        }
        self.take_body(len)
    }

    /// Like `body`, for segments that are only skipped or inspected: a length below 2 reads
    /// as an empty segment, as the reference decoder treats it.
    fn loose_body(&mut self) -> Result<&'a [u8], String> {
        let len = self
            .data
            .get(self.pos..self.pos + 2)
            .map(|b| ((b[0] as usize) << 8) | b[1] as usize)
            .ok_or(TRUNCATED)?;
        self.take_body(len.max(2))
    }

    fn take_body(&mut self, len: usize) -> Result<&'a [u8], String> {
        let body = self
            .data
            .get(self.pos + 2..self.pos + len)
            .ok_or(TRUNCATED)?;
        self.pos += len;
        Ok(body)
    }

    /// Handles every marker that is not a frame header, a scan header or the end of image.
    fn segment(&mut self, marker: u8) -> Result<(), String> {
        match marker {
            // Restart markers outside a scan carry no data; neither does TEM, which the
            // reference library accepts only after the first scan header.
            0xD0..=0xD7 => Ok(()),
            0x01 if self.in_scans => Ok(()),
            0xD8 => Err("JPEG has a second start-of-image marker".into()),
            0xC4 => self.huffman_tables(),
            0xDB => self.quant_tables(),
            0xDD => {
                let body = self.body()?;
                if body.len() != 2 {
                    return Err("JPEG restart interval segment has an invalid length".into());
                }
                self.restart_interval = ((body[0] as usize) << 8) | body[1] as usize;
                Ok(())
            }
            0xE0 => {
                let body = self.loose_body()?;
                self.check_tag(body, b"JFIF")?;
                if body.len() >= 14 && body.starts_with(b"JFIF\0") {
                    self.jfif = true;
                }
                Ok(())
            }
            0xE1 => {
                let body = self.loose_body()?;
                if self.exif.is_none() && !self.in_scans {
                    if let Some(tiff) = body.strip_prefix(b"Exif\0\0") {
                        self.exif = Some(tiff.to_vec());
                    }
                }
                Ok(())
            }
            0xEE => {
                let body = self.loose_body()?;
                self.check_tag(body, b"Adobe")?;
                if body.len() >= 12 && body.starts_with(b"Adobe") {
                    self.adobe_transform = Some(body[11]);
                }
                Ok(())
            }
            // Other application segments, comments and DNL are skipped.
            0xE2..=0xED | 0xEF | 0xFE | 0xDC => self.loose_body().map(|_| ()),
            // Arithmetic conditioning values have no effect on a Huffman file, but the
            // reference decoder still checks them.
            0xCC => {
                let body = self.body()?;
                let (pairs, rest) = body.as_chunks::<2>();
                let valid = |&[slot, value]: &[u8; 2]| match slot {
                    0..=15 => value & 15 <= value >> 4,
                    _ => slot < 32,
                };
                if !rest.is_empty() || !pairs.iter().all(valid) {
                    return Err("JPEG has an invalid arithmetic conditioning segment".into());
                }
                Ok(())
            }
            0xDE | 0xDF => Err("hierarchical JPEG is not supported".into()),
            _ => Err(format!("JPEG has an unsupported marker 0xFF{marker:02X}")),
        }
    }

    /// Before the first scan the reference library reads a 16-bit field after a JFIF or
    /// Adobe tag and gives the file up when the segment is too short to hold it.
    fn check_tag(&self, body: &[u8], tag: &[u8]) -> Result<(), String> {
        if !self.in_scans && body.starts_with(tag) && body.len() < 7 {
            return Err("JPEG application segment is too short".into());
        }
        Ok(())
    }

    fn quant_tables(&mut self) -> Result<(), String> {
        let mut body = self.body()?;
        while let Some((&head, rest)) = body.split_first() {
            let wide = head >> 4 != 0;
            let slot = (head & 15) as usize;
            let size = if wide { 128 } else { 64 };
            let (Some(values), Some(table)) = (rest.get(..size), self.quant.get_mut(slot)) else {
                return Err("JPEG has an invalid quantization table segment".into());
            };
            let mut natural = [0u16; 64];
            for (k, &at) in ZIGZAG.iter().enumerate() {
                natural[at as usize] = if wide {
                    ((values[2 * k] as u16) << 8) | values[2 * k + 1] as u16
                } else {
                    values[k] as u16
                };
            }
            *table = Some(natural);
            body = &rest[size..];
        }
        Ok(())
    }

    fn huffman_tables(&mut self) -> Result<(), String> {
        let bad = || "JPEG has an invalid Huffman table segment".to_string();
        let mut body = self.body()?;
        while body.len() > 16 {
            let head = body[0];
            let mut spec = HuffSpec {
                counts: [0; 16],
                symbols: [0; 256],
            };
            spec.counts.copy_from_slice(&body[1..17]);
            let total: usize = spec.counts.iter().map(|&c| c as usize).sum();
            let symbols = body.get(17..17 + total).ok_or_else(bad)?;
            if total > 256 {
                return Err(bad());
            }
            spec.symbols[..total].copy_from_slice(symbols);
            let slot = (head & 15) as usize;
            let target = match head >> 4 {
                0 => self.dc_specs.get_mut(slot),
                1 => self.ac_specs.get_mut(slot),
                _ => None,
            };
            *target.ok_or_else(bad)? = Some(spec);
            body = &body[17 + total..];
        }
        if !body.is_empty() {
            return Err(bad());
        }
        Ok(())
    }

    /// Parses a frame header and derives the component geometry.
    fn frame_header(&mut self, marker: u8) -> Result<Frame, String> {
        let progressive = match marker {
            0xC0 | 0xC1 => false,
            0xC2 => true,
            0xC3 => return Err("lossless JPEG is not supported".into()),
            0xC5..=0xC7 | 0xCD..=0xCF => {
                return Err("hierarchical JPEG is not supported".into());
            }
            _ => return Err("JPEG arithmetic coding is not supported".into()),
        };
        let body = self.body()?;
        if body.len() < 6 {
            return Err("JPEG frame header is too short".into());
        }
        let precision = body[0];
        let height = ((body[1] as usize) << 8) | body[2] as usize;
        let width = ((body[3] as usize) << 8) | body[4] as usize;
        let count = body[5] as usize;
        if precision != 8 {
            return Err(format!(
                "JPEG with {precision}-bit samples is not supported (only 8-bit)"
            ));
        }
        if width == 0 || height == 0 {
            return Err("JPEG image has zero size".into());
        }
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(format!(
                "JPEG image of {width}x{height} pixels is larger than the format decoder allows"
            ));
        }
        if !matches!(count, 1 | 3 | 4) {
            return Err(format!("JPEG with {count} components is not supported"));
        }
        if body.len() != 6 + 3 * count {
            return Err("JPEG frame header has an invalid length".into());
        }
        let mut comps: Vec<Component> = Vec::with_capacity(count);
        for spec in body[6..].as_chunks::<3>().0 {
            let h = (spec[1] >> 4) as usize;
            let v = (spec[1] & 15) as usize;
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                return Err("JPEG has an invalid sampling factor".into());
            }
            comps.push(Component {
                id: spec[0],
                h,
                v,
                tq: spec[2] as usize,
                h_expand: 1,
                v_expand: 1,
                data_w: 0,
                data_h: 0,
                blocks_w: 0,
                blocks_h: 0,
                mcu_blocks_w: 0,
                mcu_blocks_h: 0,
                quant: [0; 64],
                quant_latched: false,
                missing_bits: [-1; 64],
                plane: Vec::new(),
                coefs: Vec::new(),
            });
        }
        let h_max = comps.iter().map(|c| c.h).max().unwrap_or(1);
        let v_max = comps.iter().map(|c| c.v).max().unwrap_or(1);
        let mcus_x = width.div_ceil(8 * h_max);
        let mcus_y = height.div_ceil(8 * v_max);
        for c in &mut comps {
            if h_max % c.h != 0 || v_max % c.v != 0 {
                return Err(
                    "JPEG sampling factors that do not divide the largest one are not supported"
                        .into(),
                );
            }
            c.h_expand = h_max / c.h;
            c.v_expand = v_max / c.v;
            c.data_w = (width * c.h).div_ceil(h_max);
            c.data_h = (height * c.v).div_ceil(v_max);
            c.blocks_w = c.data_w.div_ceil(8);
            c.blocks_h = c.data_h.div_ceil(8);
            c.mcu_blocks_w = mcus_x * c.h;
            c.mcu_blocks_h = mcus_y * c.v;
        }
        Ok(Frame {
            width,
            height,
            progressive,
            mcus_x,
            mcus_y,
            comps,
            color: Color::Gray,
            scans: 0,
            single_scan: false,
        })
    }

    /// What the first scan header settles: the colour model, the standard Huffman tables for
    /// slots a sequential file left empty, and the buffers.
    fn begin_image(&mut self, frame: &mut Frame, scan_comps: usize) -> Result<(), String> {
        let ids: Vec<u8> = frame.comps.iter().map(|c| c.id).collect();
        frame.color = match ids.len() {
            1 => Color::Gray,
            3 => {
                if self.jfif {
                    Color::YCbCr
                } else if let Some(transform) = self.adobe_transform {
                    if transform == 0 {
                        Color::Rgb
                    } else {
                        Color::YCbCr
                    }
                } else if ids == *b"RGB" {
                    Color::Rgb
                } else {
                    Color::YCbCr
                }
            }
            _ => match self.adobe_transform {
                Some(0) | None => Color::Cmyk,
                Some(_) => Color::Ycck,
            },
        };
        let defaults: [(&mut Option<HuffSpec>, &[u8; 16], &[u8]); 4] = {
            let [dc0, dc1, ..] = &mut self.dc_specs;
            let [ac0, ac1, ..] = &mut self.ac_specs;
            [
                (dc0, &STD_DC_LUMA_COUNTS, &STD_DC_SYMBOLS),
                (ac0, &STD_AC_LUMA_COUNTS, &STD_AC_LUMA_SYMBOLS),
                (dc1, &STD_DC_CHROMA_COUNTS, &STD_DC_SYMBOLS),
                (ac1, &STD_AC_CHROMA_COUNTS, &STD_AC_CHROMA_SYMBOLS),
            ]
        };
        for (slot, counts, symbols) in defaults {
            // The reference decoder supplies them for sequential frames only.
            if slot.is_none() && !frame.progressive {
                *slot = Some(HuffSpec::standard(counts, symbols));
            }
        }
        frame.single_scan = !frame.progressive && scan_comps == frame.comps.len();
        for c in &mut frame.comps {
            if frame.single_scan {
                c.plane = buffer(plane_len(c)?)?;
            } else {
                let blocks = c
                    .mcu_blocks_w
                    .checked_mul(c.mcu_blocks_h)
                    .and_then(|n| n.checked_mul(64))
                    .ok_or("JPEG image is too large")?;
                c.coefs = buffer(blocks)?;
            }
        }
        Ok(())
    }

    /// Parses a scan header and decodes the scan's entropy-coded data.
    fn scan(&mut self, frame: &mut Frame) -> Result<(), String> {
        let body = self.body()?;
        let count = *body.first().ok_or("JPEG scan header is too short")? as usize;
        if !(1..=4).contains(&count) || body.len() != 4 + 2 * count {
            return Err("JPEG scan header has an invalid length".into());
        }
        self.in_scans = true;
        if frame.scans == 0 {
            self.begin_image(frame, count)?;
        } else if frame.single_scan {
            return Err("JPEG has a second scan after one that already holds the image".into());
        }
        frame.scans += 1;
        if frame.scans > MAX_SCANS {
            return Err(format!("JPEG has more than {MAX_SCANS} scans"));
        }

        let mut comps: Vec<ScanComp> = Vec::with_capacity(count);
        let mut dc_slots = [0usize; 4];
        let mut ac_slots = [0usize; 4];
        for (i, spec) in body[1..1 + 2 * count].as_chunks::<2>().0.iter().enumerate() {
            // The reference decoder takes the first frame component with this id whose
            // position is not below the position in the scan, which also resolves frames
            // that repeat an id.
            let index = (i..frame.comps.len())
                .find(|&ci| frame.comps[ci].id == spec[0])
                .ok_or("JPEG scan names a component the frame does not have at that position")?;
            if comps.iter().any(|sc| sc.index == index) {
                return Err("JPEG scan names a component twice".into());
            }
            dc_slots[i] = (spec[1] >> 4) as usize;
            ac_slots[i] = (spec[1] & 15) as usize;
            let (h, v) = if count == 1 {
                (1, 1)
            } else {
                (frame.comps[index].h, frame.comps[index].v)
            };
            comps.push(ScanComp {
                index,
                dc: i,
                ac: i,
                h,
                v,
                pred: 0,
            });
        }
        let tail = &body[1 + 2 * count..];
        let (ss, se) = (tail[0] as usize, tail[1] as usize);
        let (ah, al) = ((tail[2] >> 4) as u32, (tail[2] & 15) as u32);

        if count > 1 && comps.iter().map(|sc| sc.h * sc.v).sum::<usize>() > MAX_BLOCKS_IN_MCU {
            return Err("JPEG scan has too many blocks per MCU".into());
        }

        let kind = if !frame.progressive {
            ScanKind::Sequential
        } else {
            let dc_band = ss == 0;
            let bad = (dc_band && se != 0)
                || (!dc_band && (ss > se || se > 63 || count != 1))
                || (ah != 0 && al != ah - 1)
                || al > 13;
            if bad {
                return Err("JPEG progressive scan has invalid parameters".into());
            }
            match (dc_band, ah == 0) {
                (true, true) => ScanKind::DcFirst,
                (true, false) => ScanKind::DcRefine,
                (false, true) => ScanKind::AcFirst,
                (false, false) => ScanKind::AcRefine,
            }
        };

        // The scan's components take their quantization tables now if they have none yet.
        for sc in &comps {
            let c = &mut frame.comps[sc.index];
            if !c.quant_latched {
                c.quant = self
                    .quant
                    .get(c.tq)
                    .copied()
                    .flatten()
                    .ok_or("JPEG uses a quantization table it does not define")?;
                c.quant_latched = true;
            }
            if frame.progressive {
                for bits in &mut c.missing_bits[ss..=se] {
                    *bits = al as i8;
                }
            }
        }

        let needs_dc = matches!(kind, ScanKind::Sequential | ScanKind::DcFirst);
        let needs_ac = matches!(
            kind,
            ScanKind::Sequential | ScanKind::AcFirst | ScanKind::AcRefine
        );
        let missing = || "JPEG uses a Huffman table it does not define".to_string();
        let mut dc_tables: Vec<Box<Huff>> = Vec::with_capacity(count);
        let mut ac_tables: Vec<Box<Huff>> = Vec::with_capacity(count);
        for i in 0..count {
            if needs_dc {
                let spec = self.dc_specs.get(dc_slots[i]).and_then(Option::as_ref);
                dc_tables.push(Huff::build(spec.ok_or_else(missing)?, true)?);
            }
            if needs_ac {
                let spec = self.ac_specs.get(ac_slots[i]).and_then(Option::as_ref);
                ac_tables.push(Huff::build(spec.ok_or_else(missing)?, false)?);
            }
        }

        let (mcus_x, mcus_y) = if count == 1 {
            let c = &frame.comps[comps[0].index];
            (c.blocks_w, c.blocks_h)
        } else {
            (frame.mcus_x, frame.mcus_y)
        };

        let mut reader = BitReader::new(self.data, self.pos);
        let interval = self.restart_interval;
        let mut until_restart = interval;
        let mut next_restart = 0u8;
        let mut eob_run = 0u32;
        let single_scan = frame.single_scan;
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                if interval != 0 {
                    if until_restart == 0 {
                        reader.restart(next_restart)?;
                        next_restart = (next_restart + 1) & 7;
                        until_restart = interval;
                        eob_run = 0;
                        for sc in &mut comps {
                            sc.pred = 0;
                        }
                    }
                    until_restart -= 1;
                }
                for sc in &mut comps {
                    let c = &mut frame.comps[sc.index];
                    for v in 0..sc.v {
                        let by = my * sc.v + v;
                        for h in 0..sc.h {
                            let bx = mx * sc.h + h;
                            if single_scan {
                                let mut block = [0i16; 64];
                                let ac_seen = sequential_block(
                                    &mut reader,
                                    &dc_tables[sc.dc],
                                    &ac_tables[sc.ac],
                                    &mut sc.pred,
                                    &mut block,
                                );
                                if bx < c.blocks_w && by < c.blocks_h {
                                    let stride = c.blocks_w * 8;
                                    let at = by * 8 * stride + bx * 8;
                                    let out = c.plane.get_mut(at..).ok_or(BAD_ENTROPY)?;
                                    if ac_seen {
                                        inverse_dct(&block, &c.quant, out, stride);
                                    } else {
                                        flat_block(block[0], c.quant[0], out, stride);
                                    }
                                }
                                continue;
                            }
                            let at = (by * c.mcu_blocks_w + bx) * 64;
                            let block: &mut [i16; 64] = c
                                .coefs
                                .get_mut(at..at + 64)
                                .and_then(|b| b.try_into().ok())
                                .ok_or(BAD_ENTROPY)?;
                            match kind {
                                ScanKind::Sequential => {
                                    sequential_block(
                                        &mut reader,
                                        &dc_tables[sc.dc],
                                        &ac_tables[sc.ac],
                                        &mut sc.pred,
                                        block,
                                    );
                                }
                                ScanKind::DcFirst => {
                                    reader.ensure();
                                    let size = reader.symbol(&dc_tables[sc.dc]);
                                    if size != 0 {
                                        sc.pred = sc.pred.wrapping_add(reader.extend(size));
                                    }
                                    block[0] = sc.pred.wrapping_shl(al) as i16;
                                }
                                ScanKind::DcRefine => {
                                    reader.ensure();
                                    if reader.bit() {
                                        block[0] |= 1 << al;
                                    }
                                }
                                ScanKind::AcFirst => ac_first_block(
                                    &mut reader,
                                    &ac_tables[sc.ac],
                                    block,
                                    (ss, se),
                                    al,
                                    &mut eob_run,
                                ),
                                ScanKind::AcRefine => ac_refine_block(
                                    &mut reader,
                                    &ac_tables[sc.ac],
                                    block,
                                    (ss, se),
                                    al,
                                    &mut eob_run,
                                ),
                            }
                        }
                    }
                }
                reader.check()?;
            }
        }
        self.pos = reader.pos;
        Ok(())
    }
}

fn plane_len(c: &Component) -> Result<usize, String> {
    (c.blocks_w * 8)
        .checked_mul(c.blocks_h * 8)
        .ok_or_else(|| "JPEG image is too large".to_string())
}

/// Decodes one block of a sequential scan into `block` (row-major). Coefficients the scan
/// codes as zero are left as they are. Returns whether any AC coefficient was written.
#[inline(always)]
fn sequential_block(
    reader: &mut BitReader,
    dc: &Huff,
    ac: &Huff,
    pred: &mut i32,
    block: &mut [i16; 64],
) -> bool {
    reader.ensure();
    let size = reader.symbol(dc);
    if size != 0 {
        *pred = pred.wrapping_add(reader.extend(size));
    }
    block[0] = *pred as i16;
    let mut ac_seen = false;
    let mut k = 1usize;
    while k < 64 {
        reader.ensure();
        let symbol = reader.symbol(ac);
        let run = (symbol >> 4) as usize;
        let size = symbol & 15;
        if size != 0 {
            k += run;
            let value = reader.extend(size);
            // A run past the end of the block lands on the last coefficient.
            block[ZIGZAG[k.min(63)] as usize] = value as i16;
            ac_seen = true;
            k += 1;
        } else if run == 15 {
            k += 16;
        } else {
            break;
        }
    }
    ac_seen
}

/// Decodes one block of a progressive AC scan that sends the band `ss..=se` for the first time.
fn ac_first_block(
    reader: &mut BitReader,
    table: &Huff,
    block: &mut [i16; 64],
    (ss, se): (usize, usize),
    al: u32,
    eob_run: &mut u32,
) {
    if *eob_run > 0 {
        *eob_run -= 1;
        return;
    }
    let mut k = ss;
    while k <= se {
        reader.ensure();
        let symbol = reader.symbol(table);
        let run = symbol >> 4;
        let size = symbol & 15;
        if size != 0 {
            k += run as usize;
            let value = reader.extend(size);
            block[ZIGZAG[k.min(63)] as usize] = value.wrapping_shl(al) as i16;
            k += 1;
        } else if run == 15 {
            k += 16;
        } else {
            // End of band for this block and the next `2^run + extra - 1` blocks.
            *eob_run = (1 << run) - 1;
            if run != 0 {
                *eob_run += reader.take(run);
            }
            break;
        }
    }
}

/// Decodes one block of a progressive AC scan that adds one bit of precision to `ss..=se`.
fn ac_refine_block(
    reader: &mut BitReader,
    table: &Huff,
    block: &mut [i16; 64],
    (ss, se): (usize, usize),
    al: u32,
    eob_run: &mut u32,
) {
    let plus = 1i16 << al;
    let minus = (-1i16) << al;
    // Reads the correction bit of a coefficient that is already nonzero.
    let correct = |reader: &mut BitReader, coef: &mut i16| {
        reader.ensure();
        if reader.bit() && *coef & plus == 0 {
            *coef = coef.wrapping_add(if *coef >= 0 { plus } else { minus });
        }
    };
    let mut k = ss;
    if *eob_run == 0 {
        while k <= se {
            reader.ensure();
            let symbol = reader.symbol(table);
            let mut run = (symbol >> 4) as i32;
            let size = symbol & 15;
            let mut fresh = 0i16;
            if size != 0 {
                // A coefficient that becomes nonzero in a refinement scan is +1 or -1.
                if size != 1 {
                    reader.corrupt = true;
                }
                fresh = if reader.bit() { plus } else { minus };
            } else if run != 15 {
                *eob_run = 1 << run;
                if run != 0 {
                    *eob_run += reader.take(run as u32);
                }
                break;
            }
            // Pass `run` zero coefficients, correcting the nonzero ones on the way.
            while k <= se {
                let coef = &mut block[ZIGZAG[k] as usize];
                if *coef != 0 {
                    correct(reader, coef);
                } else {
                    run -= 1;
                    if run < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if fresh != 0 {
                block[ZIGZAG[k.min(63)] as usize] = fresh;
            }
            k += 1;
        }
    }
    if *eob_run > 0 {
        // Blocks with no coefficient at all take no bits; passing them quickly keeps a scan
        // that is one long end-of-band run cheap.
        if block[1..].iter().any(|&v| v != 0) {
            while k <= se {
                let coef = &mut block[ZIGZAG[k] as usize];
                if *coef != 0 {
                    correct(reader, coef);
                }
                k += 1;
            }
        }
        *eob_run -= 1;
    }
}

/// One dequantized coefficient: the product wraps to 16 bits.
#[inline(always)]
fn dequantize(coef: i16, quant: u16) -> i32 {
    (coef as i32).wrapping_mul(quant as i32) as i16 as i32
}

/// The 8-point inverse DCT kernel shared by both passes. Inputs are 16-bit values; the sums
/// of input pairs wrap to 16 bits and everything after the multipliers wraps to 32 bits.
/// The outputs are scaled by 2^13 times the pass gain and not yet rounded.
#[inline(always)]
fn idct_kernel(x: [i32; 8]) -> [i32; 8] {
    let m = |a: i32, k: i32| a.wrapping_mul(k);
    let wrap16 = |a: i32| a as i16 as i32;

    // Even part.
    let e3 = m(x[2], 10703).wrapping_add(m(x[6], 4433));
    let e2 = m(x[2], 4433).wrapping_add(m(x[6], -10704));
    let e0 = wrap16(x[0].wrapping_add(x[4])) << 13;
    let e1 = wrap16(x[0].wrapping_sub(x[4])) << 13;
    let a0 = e0.wrapping_add(e3);
    let a3 = e0.wrapping_sub(e3);
    let a1 = e1.wrapping_add(e2);
    let a2 = e1.wrapping_sub(e2);

    // Odd part.
    let z3 = wrap16(x[7].wrapping_add(x[3]));
    let z4 = wrap16(x[5].wrapping_add(x[1]));
    let w3 = m(z3, -6436).wrapping_add(m(z4, 9633));
    let w4 = m(z3, 9633).wrapping_add(m(z4, 6437));
    let b0 = m(x[7], -4927).wrapping_add(m(x[1], -7373)).wrapping_add(w3);
    let b1 = m(x[5], -4176)
        .wrapping_add(m(x[3], -20995))
        .wrapping_add(w4);
    let b2 = m(x[5], -20995).wrapping_add(m(x[3], 4177)).wrapping_add(w3);
    let b3 = m(x[7], -7373).wrapping_add(m(x[1], 4926)).wrapping_add(w4);

    [
        a0.wrapping_add(b3),
        a1.wrapping_add(b2),
        a2.wrapping_add(b1),
        a3.wrapping_add(b0),
        a3.wrapping_sub(b0),
        a2.wrapping_sub(b1),
        a1.wrapping_sub(b2),
        a0.wrapping_sub(b3),
    ]
}

/// Accurate integer inverse DCT of one block into `out` (8 rows, `stride` apart).
fn inverse_dct(block: &[i16; 64], quant: &[u16; 64], out: &mut [u8], stride: usize) {
    let mut work = [0i32; 64];
    if block[8..].iter().all(|&v| v == 0) {
        // Only the first row is present: every column is its dequantized top value times 4,
        // wrapped to 16 bits (this case does not saturate).
        for col in 0..8 {
            let top = (dequantize(block[col], quant[col]) << 2) as i16 as i32;
            for row in 0..8 {
                work[row * 8 + col] = top;
            }
        }
    } else {
        // Columns: keep 2 extra bits, saturate to 16 bits.
        for col in 0..8 {
            let at = |row: usize| dequantize(block[row * 8 + col], quant[row * 8 + col]);
            let y = idct_kernel([at(0), at(1), at(2), at(3), at(4), at(5), at(6), at(7)]);
            for row in 0..8 {
                work[row * 8 + col] = (y[row].wrapping_add(1 << 10) >> 11).clamp(-32768, 32767);
            }
        }
    }
    // Rows: remove the scaling and the factor 8, saturate to the sample range.
    for (row, line) in work.as_chunks::<8>().0.iter().enumerate() {
        let Some(dst) = out.get_mut(row * stride..row * stride + 8) else {
            return;
        };
        if line[1..].iter().all(|&v| v == 0) {
            dst.fill((((line[0] + 16) >> 5).clamp(-128, 127) + 128) as u8);
            continue;
        }
        let y = idct_kernel(*line);
        for (d, v) in dst.iter_mut().zip(y) {
            *d = ((v.wrapping_add(1 << 17) >> 18).clamp(-128, 127) + 128) as u8;
        }
    }
}

/// Inverse DCT of a block that has only its DC coefficient.
fn flat_block(dc: i16, quant: u16, out: &mut [u8], stride: usize) {
    let column = (dequantize(dc, quant) << 2) as i16 as i32;
    let sample = (((column + 16) >> 5).clamp(-128, 127) + 128) as u8;
    for row in 0..8 {
        if let Some(dst) = out.get_mut(row * stride..row * stride + 8) {
            dst.fill(sample);
        }
    }
}

/// Fixed-point YCbCr to RGB tables: Cr to red, Cb to blue (rounded), and the two green terms
/// scaled by 2^16 with the rounding constant folded into the Cb one.
const fn color_tables() -> ([i32; 256], [i32; 256], [i32; 256], [i32; 256]) {
    let mut cr_r = [0i32; 256];
    let mut cb_b = [0i32; 256];
    let mut cr_g = [0i32; 256];
    let mut cb_g = [0i32; 256];
    let mut i = 0;
    while i < 256 {
        let x = i as i32 - 128;
        cr_r[i] = (91881 * x + 32768) >> 16;
        cb_b[i] = (116130 * x + 32768) >> 16;
        cr_g[i] = -46802 * x;
        cb_g[i] = -22554 * x + 32768;
        i += 1;
    }
    (cr_r, cb_b, cr_g, cb_g)
}

const COLOR_TABLES: ([i32; 256], [i32; 256], [i32; 256], [i32; 256]) = color_tables();

#[inline(always)]
fn ycc_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let (cr_r, cb_b, cr_g, cb_g) = &COLOR_TABLES;
    let y = y as i32;
    let r = y + cr_r[cr as usize];
    let g = y + ((cb_g[cb as usize] + cr_g[cr as usize]) >> 16);
    let b = y + cb_b[cb as usize];
    [
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    ]
}

/// One channel of the reference library's CMYK to RGB step. `ink` is the amount of ink
/// (0 = none) and `white` is 255 minus the black ink.
#[inline(always)]
fn apply_ink(ink: u8, white: u8) -> u8 {
    let t = ink as u32 * white as u32 + 128;
    (white as u32 - (((t >> 8) + t) >> 8)) as u8
}

/// Writes output row `y` of a component at full resolution into `out`.
fn upsample_row(c: &Component, y: usize, out: &mut [u8]) {
    let stride = c.blocks_w * 8;
    let line = |row: usize| -> &[u8] {
        c.plane
            .get(row * stride..row * stride + c.data_w)
            .unwrap_or(&[])
    };
    let wide = c.data_w > 2;
    match (c.h_expand, c.v_expand) {
        (1, 1) => {
            for (d, s) in out.iter_mut().zip(line(y)) {
                *d = *s;
            }
        }
        (2, 1) if wide => {
            let src = line(y);
            let get = |i: usize| src.get(i).copied().unwrap_or(0) as u32;
            let last = c.data_w - 1;
            let pairs = out.as_chunks_mut::<2>().0.iter_mut();
            for (i, pair) in pairs.take(c.data_w).enumerate() {
                let near = 3 * get(i);
                pair[0] = if i == 0 {
                    get(0) as u8
                } else {
                    ((near + get(i - 1) + 1) >> 2) as u8
                };
                pair[1] = if i == last {
                    get(last) as u8
                } else {
                    ((near + get(i + 1) + 2) >> 2) as u8
                };
            }
        }
        (1, 2) => {
            let row = y / 2;
            let (far, bias) = if y & 1 == 0 {
                (row.saturating_sub(1), 1)
            } else {
                ((row + 1).min(c.data_h - 1), 2)
            };
            for ((d, n), f) in out.iter_mut().zip(line(row)).zip(line(far)) {
                *d = ((3 * *n as u32 + *f as u32 + bias) >> 2) as u8;
            }
        }
        (2, 2) if wide => {
            let row = y / 2;
            let far = if y & 1 == 0 {
                row.saturating_sub(1)
            } else {
                (row + 1).min(c.data_h - 1)
            };
            let (near, far) = (line(row), line(far));
            let sum = |i: usize| {
                3 * near.get(i).copied().unwrap_or(0) as u32
                    + far.get(i).copied().unwrap_or(0) as u32
            };
            let last = c.data_w - 1;
            let mut prev = sum(0);
            let mut this = prev;
            let pairs = out.as_chunks_mut::<2>().0.iter_mut();
            for (i, pair) in pairs.take(c.data_w).enumerate() {
                let next = if i == last { this } else { sum(i + 1) };
                pair[0] = if i == 0 {
                    ((this * 4 + 8) >> 4) as u8
                } else {
                    ((this * 3 + prev + 8) >> 4) as u8
                };
                pair[1] = if i == last {
                    ((this * 4 + 7) >> 4) as u8
                } else {
                    ((this * 3 + next + 7) >> 4) as u8
                };
                prev = this;
                this = next;
            }
        }
        (hx, vx) => {
            let src = line(y / vx);
            for (group, s) in out.chunks_mut(hx).zip(src) {
                group.fill(*s);
            }
        }
    }
}

/// Turns the decoded coefficients or planes into the RGB image.
fn finish(mut frame: Frame, exif: Option<Vec<u8>>) -> Result<Still, String> {
    if !frame.single_scan {
        if frame.progressive {
            // The reference decoder smooths across blocks when every component has its DC and
            // some component lacks full precision in its first nine AC coefficients.
            let all_started = frame.comps.iter().all(|c| {
                c.quant_latched
                    && c.missing_bits[0] >= 0
                    && ZIGZAG[..10].iter().all(|&at| c.quant[at as usize] != 0)
            });
            let imprecise = frame
                .comps
                .iter()
                .any(|c| c.missing_bits[1..10].iter().any(|&bits| bits != 0));
            if all_started && imprecise {
                return Err(
                    "progressive JPEG is incomplete (not all coefficient scans are present)".into(),
                );
            }
        }
        for c in &mut frame.comps {
            let stride = c.blocks_w * 8;
            c.plane = buffer(plane_len(c)?)?;
            for by in 0..c.blocks_h {
                for bx in 0..c.blocks_w {
                    let at = (by * c.mcu_blocks_w + bx) * 64;
                    let Some(block) = c
                        .coefs
                        .get(at..at + 64)
                        .and_then(|b| <&[i16; 64]>::try_from(b).ok())
                    else {
                        continue;
                    };
                    let Some(out) = c.plane.get_mut(by * 8 * stride + bx * 8..) else {
                        continue;
                    };
                    if block[1..].iter().all(|&v| v == 0) {
                        flat_block(block[0], c.quant[0], out, stride);
                    } else {
                        inverse_dct(block, &c.quant, out, stride);
                    }
                }
            }
            c.coefs = Vec::new();
        }
    }

    let (width, height) = (frame.width, frame.height);
    let rgb_len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or("JPEG image is too large")?;
    let mut rgb: Vec<u8> = buffer(rgb_len)?;
    // One row per component at full resolution; the filters write an even number of samples.
    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(frame.comps.len());
    for _ in &frame.comps {
        rows.push(buffer(width + 1)?);
    }
    for (y, out) in rgb.chunks_exact_mut(width * 3).enumerate() {
        for (c, row) in frame.comps.iter().zip(rows.iter_mut()) {
            upsample_row(c, y, row);
        }
        let pixels = out.as_chunks_mut::<3>().0.iter_mut();
        match (frame.color, rows.as_slice()) {
            (Color::Gray, [g]) => {
                for (px, &v) in pixels.zip(g) {
                    px.fill(v);
                }
            }
            (Color::YCbCr, [a, b, c]) => {
                for (((px, &y), &cb), &cr) in pixels.zip(a).zip(b).zip(c) {
                    *px = ycc_to_rgb(y, cb, cr);
                }
            }
            (Color::Rgb, [a, b, c]) => {
                for (((px, &r), &g), &b) in pixels.zip(a).zip(b).zip(c) {
                    *px = [r, g, b];
                }
            }
            // The samples are inverted inks: 255 means no ink.
            (Color::Cmyk, [a, b, c, k]) => {
                for ((((px, &c0), &c1), &c2), &k) in pixels.zip(a).zip(b).zip(c).zip(k) {
                    *px = [
                        apply_ink(255 - c0, k),
                        apply_ink(255 - c1, k),
                        apply_ink(255 - c2, k),
                    ];
                }
            }
            // YCbCr codes the inverted CMY inks, so its RGB value is the ink amount.
            (Color::Ycck, [a, b, c, k]) => {
                for ((((px, &y), &cb), &cr), &k) in pixels.zip(a).zip(b).zip(c).zip(k) {
                    let [r, g, b] = ycc_to_rgb(y, cb, cr);
                    *px = [apply_ink(r, k), apply_ink(g, k), apply_ink(b, k)];
                }
            }
            _ => return Err("JPEG component count does not match its colour model".into()),
        }
    }
    Ok(Still {
        width,
        height,
        rgb,
        exif,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: DecodeLimits = DecodeLimits {
        max_pixels: 1 << 24,
    };

    macro_rules! fixture {
        ($name:literal) => {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/vision/jpeg/",
                $name
            )) as &[u8]
        };
    }

    macro_rules! fixtures {
        ($(($name:literal, $w:literal, $h:literal, $hash:literal),)*) => {
            &[$(($name, fixture!($name), $w, $h, $hash),)*]
        };
    }

    /// File, bytes, width, height and the FNV-1a 64 hash of the RGB bytes the reference
    /// library returns, as `tests/data/vision/jpeg/make_fixtures.py` prints them.
    const FIXTURES: &[(&str, &[u8], usize, usize, u64)] = fixtures![
        ("base_411_33x13.jpg", 33, 13, 0x6ef37706e571829f),
        ("base_420_17x13.jpg", 17, 13, 0x5b3a110a1483ff34),
        ("base_420_q1_33x49.jpg", 33, 49, 0xcda5a62bf03ba736),
        ("base_422_33x49.jpg", 33, 49, 0x2221efd8b1c7609b),
        ("base_440_17x13.jpg", 17, 13, 0xde6f31e0785b5ca5),
        ("base_444_q100_17x13.jpg", 17, 13, 0x11895abbfd70f75b),
        ("base_wide_quant_17x13.jpg", 17, 13, 0xcc6914d5b0400db8),
        ("cmyk_17x13.jpg", 17, 13, 0x908e8346c4b9c6fa),
        ("exif_thumbnail_17x13.jpg", 17, 13, 0x2288795aec9e5959),
        ("gray_17x13.jpg", 17, 13, 0xc5a629be208d76b2),
        ("gray_prog_33x49.jpg", 33, 49, 0x6873082605c58275),
        ("luma_small_17x13.jpg", 17, 13, 0x1ff26d70ed5790cb),
        ("mixed_scans_17x13.jpg", 17, 13, 0x80519b49847d3a2a),
        ("narrow_420_3x9.jpg", 3, 9, 0x8d8e36a46d81e76d),
        ("no_tables_17x13.jpg", 17, 13, 0x2288795aec9e5959),
        ("one_pixel.jpg", 1, 1, 0x4335f81b7d0a7bf2),
        ("prog_420_33x49.jpg", 33, 49, 0xfd2dc9ee444c1ab8),
        ("prog_444_restart_17x13.jpg", 17, 13, 0xa593d548b76f1986),
        ("restart_420_33x49.jpg", 33, 49, 0x023d72fe21fcdc22),
        ("rgb_17x13.jpg", 17, 13, 0xf9bbf19daa7337a8),
        ("ycck_420_17x13.jpg", 17, 13, 0x447a2ab5f9ee787b),
    ];

    const BASELINE: &[u8] = fixture!("base_420_17x13.jpg");
    const PROGRESSIVE: &[u8] = fixture!("prog_420_33x49.jpg");

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Offset of the first `0xFF marker` pair among the header segments.
    fn marker_at(data: &[u8], marker: u8) -> usize {
        let mut pos = 2;
        loop {
            assert_eq!(data[pos], 0xFF, "segment walk lost at {pos}");
            if data[pos + 1] == marker {
                return pos;
            }
            pos += 2 + (((data[pos + 2] as usize) << 8) | data[pos + 3] as usize);
        }
    }

    fn error_of(data: &[u8]) -> String {
        decode(data, &LIMITS).expect_err("the file must be refused")
    }

    #[test]
    fn fixtures_match_the_reference() {
        for &(name, bytes, width, height, hash) in FIXTURES {
            assert_eq!(probe(bytes), Ok((width, height)), "{name}");
            let still = decode(bytes, &LIMITS).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((still.width, still.height), (width, height), "{name}");
            assert_eq!(still.rgb.len(), width * height * 3, "{name}");
            assert_eq!(fnv1a(&still.rgb), hash, "{name}");
        }
    }

    #[test]
    fn exif_block_is_returned_unapplied() {
        let still = decode(fixture!("exif_thumbnail_17x13.jpg"), &LIMITS).unwrap();
        let exif = still.exif.expect("the fixture has an EXIF segment");
        assert_eq!(exif.len(), 666);
        assert!(exif.starts_with(b"MM\0*"));
        assert_eq!(fnv1a(&exif), 0x3b21561e8d3a0584);
        assert_eq!(decode(BASELINE, &LIMITS).unwrap().exif, None);
    }

    #[test]
    fn only_the_first_picture_is_decoded() {
        let alone = decode(BASELINE, &LIMITS).unwrap();
        let mut two = BASELINE.to_vec();
        two.extend_from_slice(fixture!("gray_prog_33x49.jpg"));
        assert_eq!(decode(&two, &LIMITS).unwrap(), alone);
        let mut padded = BASELINE.to_vec();
        padded.extend_from_slice(b"\0\0trailing bytes \xff\xd8\xff\xc0");
        assert_eq!(decode(&padded, &LIMITS).unwrap(), alone);
    }

    #[test]
    fn every_truncation_is_refused() {
        for &(name, bytes, ..) in FIXTURES {
            for len in 0..bytes.len() {
                assert!(
                    decode(&bytes[..len], &LIMITS).is_err(),
                    "{name} cut to {len}"
                );
            }
        }
        assert!(error_of(&BASELINE[..BASELINE.len() - 2]).contains("truncated"));
        assert!(error_of(&PROGRESSIVE[..PROGRESSIVE.len() / 2]).contains("JPEG"));
        // The frame header is enough for the size.
        let sof_end = marker_at(BASELINE, 0xC0) + 19;
        assert_eq!(probe(&BASELINE[..sof_end]), Ok((17, 13)));
        assert!(probe(&BASELINE[..sof_end - 1]).is_err());
    }

    #[test]
    fn damaged_files_give_ok_or_err() {
        for &(name, bytes, ..) in FIXTURES {
            let mut data = bytes.to_vec();
            let mut pos = 2;
            while pos < data.len() {
                let original = data[pos];
                for value in [original ^ (1 << (pos % 8)), 0xFF, 0x00] {
                    data[pos] = value;
                    let _ = probe(&data);
                    if let Ok(still) = decode(&data, &LIMITS) {
                        assert_eq!(still.rgb.len(), still.width * still.height * 3, "{name}");
                    }
                }
                data[pos] = original;
                // Every byte of the headers, a sample of the entropy-coded data.
                pos += if pos < 700 { 1 } else { 5 };
            }
        }
    }

    #[test]
    fn unsupported_kinds_are_refused() {
        let sof = marker_at(BASELINE, 0xC0);
        for (marker, word) in [
            (0xC3, "lossless"),
            (0xC5, "hierarchical"),
            (0xC9, "arithmetic"),
            (0xCA, "arithmetic"),
            (0xCB, "arithmetic"),
            (0xCD, "hierarchical"),
        ] {
            let mut data = BASELINE.to_vec();
            data[sof + 1] = marker;
            assert!(error_of(&data).contains(word), "marker {marker:02X}");
            assert!(probe(&data).is_err());
        }
        let mut data = BASELINE.to_vec();
        data[sof + 4] = 12;
        assert!(error_of(&data).contains("12-bit"));
        // Two components.
        let mut data = BASELINE.to_vec();
        data[sof + 9] = 2;
        assert!(error_of(&data).contains("components"));
        // A sampling factor that does not divide the largest one.
        let mut data = BASELINE.to_vec();
        data[sof + 11] = 0x32;
        data[sof + 14] = 0x21;
        assert!(error_of(&data).contains("sampling"));
        assert!(error_of(b"\xff\x4f\xff\x51\0\0").contains("JPEG 2000"));
        assert!(error_of(b"\0\0\0\x0cjP  \r\n\x87\n").contains("JPEG 2000"));
        assert!(error_of(b"\xff\x0a\0\0").contains("JPEG XL"));
        assert!(error_of(b"\0\0\0\x0cJXL \r\n\x87\n").contains("JPEG XL"));
        assert!(error_of(b"\x89PNG\r\n\x1a\n").contains("not a JPEG"));
        assert!(error_of(b"").contains("not a JPEG"));
        assert!(error_of(b"\xff\xd8\xff\xd9").contains("no image"));
    }

    #[test]
    fn canvas_limit_is_checked_before_decoding() {
        let tight = DecodeLimits { max_pixels: 220 };
        assert!(decode(BASELINE, &tight).is_err());
        let exact = DecodeLimits { max_pixels: 221 };
        assert!(decode(BASELINE, &exact).is_ok());
        // A header that claims the largest canvas is refused without allocating it.
        let sof = marker_at(BASELINE, 0xC0);
        let mut data = BASELINE.to_vec();
        data[sof + 5..sof + 9].copy_from_slice(&[0xFF, 0xDC, 0xFF, 0xDC]);
        assert_eq!(probe(&data), Ok((65500, 65500)));
        assert!(decode(&data, &LIMITS).is_err());
        data[sof + 5..sof + 9].copy_from_slice(&[0xFF, 0xFF, 0x00, 0x01]);
        assert!(error_of(&data).contains("larger"));
        data[sof + 5..sof + 9].copy_from_slice(&[0, 0, 0, 17]);
        assert!(error_of(&data).contains("zero size"));
    }

    #[test]
    fn incomplete_progressive_file_is_refused() {
        let error = error_of(fixture!("prog_incomplete_17x13.jpg"));
        assert!(error.contains("incomplete"), "{error}");
        // Without its Huffman tables a progressive file has none: the standard tables are
        // for sequential files only.
        let dht = marker_at(PROGRESSIVE, 0xC4);
        let mut data = PROGRESSIVE.to_vec();
        data[dht + 1] = 0xEF;
        assert!(error_of(&data).contains("Huffman table"));
    }

    #[test]
    fn broken_structure_is_refused() {
        let restart = fixture!("restart_420_33x49.jpg");
        let first = restart
            .windows(2)
            .position(|w| w == [0xFF, 0xD0])
            .expect("the fixture has restart markers");
        let mut data = restart.to_vec();
        data[first + 1] = 0xD1;
        assert!(error_of(&data).contains("restart"));
        // A second start-of-image marker and a second frame header.
        let dqt = marker_at(BASELINE, 0xDB);
        let mut data = BASELINE.to_vec();
        data.splice(dqt..dqt, [0xFF, 0xD8]);
        assert!(error_of(&data).contains("second start-of-image"));
        let sof = marker_at(BASELINE, 0xC0);
        let mut data = BASELINE.to_vec();
        data.splice(sof..sof, BASELINE[sof..sof + 19].iter().copied());
        assert!(error_of(&data).contains("more than one frame"));
        // A scan that names a component the frame does not have.
        let sos = marker_at(BASELINE, 0xDA);
        let mut data = BASELINE.to_vec();
        data[sos + 5] = 9;
        assert!(error_of(&data).contains("component"));
        // Entropy-coded data cut short by an early end-of-image marker.
        let mut data = BASELINE[..BASELINE.len() - 40].to_vec();
        data.extend_from_slice(&[0xFF, 0xD9]);
        assert!(error_of(&data).contains("corrupt or ends early"));
        // A quantization table slot that was never defined.
        let mut data = BASELINE.to_vec();
        data[sof + 12] = 3;
        assert!(error_of(&data).contains("quantization table"));
    }

    #[test]
    fn huffman_tables_are_validated() {
        let mut spec = HuffSpec {
            counts: [0; 16],
            symbols: [0; 256],
        };
        // Two codes of one bit use the all-ones code.
        spec.counts[0] = 2;
        assert!(Huff::build(&spec, false).is_err());
        spec.counts[0] = 1;
        assert!(Huff::build(&spec, false).is_ok());
        // A DC table may only hold sizes up to 15.
        spec.symbols[0] = 16;
        assert!(Huff::build(&spec, true).is_err());
        assert!(Huff::build(&spec, false).is_ok());
        for (counts, symbols, is_dc) in [
            (&STD_DC_LUMA_COUNTS, &STD_DC_SYMBOLS[..], true),
            (&STD_DC_CHROMA_COUNTS, &STD_DC_SYMBOLS[..], true),
            (&STD_AC_LUMA_COUNTS, &STD_AC_LUMA_SYMBOLS[..], false),
            (&STD_AC_CHROMA_COUNTS, &STD_AC_CHROMA_SYMBOLS[..], false),
        ] {
            let total: usize = counts.iter().map(|&c| c as usize).sum();
            assert_eq!(total, symbols.len());
            assert!(Huff::build(&HuffSpec::standard(counts, symbols), is_dc).is_ok());
        }
    }

    /// Sample values the reference decoder gives for blocks with only a DC coefficient far
    /// outside the range an encoder produces (quantizer, coefficient, sample).
    #[test]
    fn out_of_range_dc_wraps_like_the_reference() {
        for (quant, dc, sample) in [
            (1u16, 0i16, 128u8),
            (1, 8, 129),
            (1, -1024, 0),
            (1, 1016, 255),
            (1, -32767, 128),
            (1, -28655, 255),
            (1, -24543, 0),
            (1, -16319, 136),
            (3, -32767, 128),
            (3, -27627, 8),
            (3, -29940, 0),
            (255, -32767, 160),
        ] {
            let mut out = [0u8; 64];
            flat_block(dc, quant, &mut out, 8);
            assert_eq!(out, [sample; 64], "quantizer {quant}, coefficient {dc}");
            let mut block = [0i16; 64];
            block[0] = dc;
            let mut table = [1u16; 64];
            table[0] = quant;
            inverse_dct(&block, &table, &mut out, 8);
            assert_eq!(out, [sample; 64], "quantizer {quant}, coefficient {dc}");
        }
    }

    /// Compares every JPEG file in the directory `GAUDIUM_IMAGE_REF_DIR` with the raw RGB
    /// dump `<name>.rgb` that the reference library wrote for it.
    #[test]
    #[ignore = "needs GAUDIUM_IMAGE_REF_DIR with reference dumps"]
    fn files_in_reference_directory_match() {
        let dir = std::env::var_os("GAUDIUM_IMAGE_REF_DIR").expect("GAUDIUM_IMAGE_REF_DIR");
        let limits = DecodeLimits {
            max_pixels: 1 << 28,
        };
        let mut checked = 0usize;
        let mut failures = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("reference directory") {
            let path = entry.expect("directory entry").path();
            let is_jpeg = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                ["jpg", "jpeg", "jpe", "jfif", "mpo"].contains(&e.to_ascii_lowercase().as_str())
            });
            let Ok(expected) = std::fs::read(path.with_extension("rgb")) else {
                continue;
            };
            if !is_jpeg {
                continue;
            }
            let bytes = std::fs::read(&path).expect("image file");
            checked += 1;
            match decode(&bytes, &limits) {
                Ok(still) if still.rgb == expected => {}
                Ok(_) => failures.push(format!("{}: pixels differ", path.display())),
                Err(e) => failures.push(format!("{}: {e}", path.display())),
            }
        }
        assert!(
            checked > 0,
            "no JPEG file with a .rgb dump in the directory"
        );
        assert!(
            failures.is_empty(),
            "{} of {checked} files differ:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
