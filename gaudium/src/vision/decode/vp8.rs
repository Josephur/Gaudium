//! Lossy WebP bitstream (the `VP8 ` chunk): the key-frame decoder of RFC 6386, and the
//! conversion of its YUV 4:2:0 output to RGB as the reference decoder does it.
//!
//! Decodes: the boolean entropy decoder, segmentation, all 16x16 / 4x4 / chroma intra
//! prediction modes, coefficient tokens with their probability updates, dequantisation with
//! segment and delta quantisers, the inverse DCT and WHT, the simple and the normal loop
//! filter (sharpness, segment and mode level adjustments), several token partitions, and
//! frames whose size is not a multiple of 16. The scaling bits of the header are ignored, as
//! the reference decoder ignores them.
//!
//! Refuses: inter frames, frames marked not to be shown, a first partition or a token
//! partition that ends before its last macroblock is read, an empty token partition.
//!
//! Bit-exact rules: the reconstruction is the integer arithmetic of the RFC. The RGB output
//! uses "fancy upsampling": every chroma sample is interpolated from its four neighbours with
//! weights 9:3:3:1 as `(9a + 3b + 3c + d + 8) >> 4` (at the left / right edge of a row
//! `(3a + b + 2) >> 2`; the first row, and the last row of an image of even height, take both
//! chroma rows from the same row), then the 14-bit fixed-point conversion in `yuv_to_rgb`.
//!
//! Damaged and hostile files decode to the reference decoder's pixels too, or are refused where
//! it refuses. That takes four of its habits, none of which shows on a file an encoder wrote:
//! a partition is at its end as soon as a decision needs a byte it does not have (`BoolReader`),
//! the sign of a coefficient is read with its branch-free step (`BoolReader::get_sign`), a
//! dequantised coefficient is cut to 16 bits, and a block with coefficients past the third is
//! transformed in 16-bit arithmetic that wraps (`add_inverse_dct_full`).

use super::types::DecodeLimits;

// 4x4 prediction modes, in the order of the RFC.
const B_DC: u8 = 0;
const B_TM: u8 = 1;
const B_VE: u8 = 2;
const B_HE: u8 = 3;
const B_LD: u8 = 4;
const B_RD: u8 = 5;
const B_VR: u8 = 6;
const B_VL: u8 = 7;
const B_HD: u8 = 8;
const B_HU: u8 = 9;

// Whole-block prediction modes (16x16 luma, 8x8 chroma). Their values are those of the 4x4
// modes they stand for in the mode context of neighbouring blocks.
const DC_PRED: u8 = B_DC;
const TM_PRED: u8 = B_TM;
const V_PRED: u8 = B_VE;
const H_PRED: u8 = B_HE;
/// Luma mode of a macroblock predicted per 4x4 block.
const B_PRED: u8 = 4;

/// Coefficient position in the block for each position in scan order.
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// Probability band of each scan position.
const BAND: [usize; 16] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7];

/// Probabilities of the extra bits of the four largest coefficient categories.
const CATEGORY_3: [u8; 3] = [173, 148, 140];
const CATEGORY_4: [u8; 4] = [176, 155, 140, 135];
const CATEGORY_5: [u8; 5] = [180, 157, 141, 134, 130];
const CATEGORY_6: [u8; 11] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];

type CoeffProbs = [[[[u8; 11]; 3]; 8]; 4];

/// The boolean entropy decoder of RFC 6386 section 7. It keeps `range - 1` and a window of
/// input bits; `bits` is the number of window bits below the 8 the next decision needs.
struct BoolReader<'a> {
    data: &'a [u8],
    pos: usize,
    value: u64,
    range: u32,
    bits: i32,
    /// Set when a decision needed input after the last byte. The reference decoder treats
    /// that as the end of the partition; the caller checks the flag where it does.
    eof: bool,
}

impl<'a> BoolReader<'a> {
    /// Starts reading `data`. The first bytes are loaded at once, so an empty partition is at
    /// its end from the start, whether or not anything is read from it.
    fn new(data: &'a [u8]) -> Self {
        let mut reader = BoolReader {
            data,
            pos: 0,
            value: 0,
            range: 254,
            bits: -8,
            eof: false,
        };
        reader.load();
        reader
    }

    #[cold]
    fn load(&mut self) {
        if let Some(chunk) = self.data.get(self.pos..self.pos + 7) {
            if self.pos + 8 <= self.data.len() {
                let mut wide = [0u8; 8];
                wide[1..].copy_from_slice(chunk);
                self.value = (self.value << 56) | u64::from_be_bytes(wide);
                self.pos += 7;
                self.bits += 56;
                return;
            }
        }
        if let Some(&byte) = self.data.get(self.pos) {
            self.value = (self.value << 8) | byte as u64;
            self.pos += 1;
            self.bits += 8;
        } else if !self.eof {
            self.value <<= 8;
            self.bits += 8;
            self.eof = true;
        } else {
            self.bits = 0;
        }
    }

    /// Reads one decision whose probability of being 0 is `prob` / 256.
    #[inline]
    fn get(&mut self, prob: u8) -> bool {
        if self.bits < 0 {
            self.load();
        }
        let split = (self.range * prob as u32) >> 8;
        let window = (self.value >> self.bits) as u32;
        let range;
        let bit = window > split;
        if bit {
            range = self.range - split;
            self.value = self.value.wrapping_sub(((split + 1) as u64) << self.bits);
        } else {
            range = split + 1;
        }
        // `range` is 1..=255 here: normalise it to 128..=255
        let shift = (range.leading_zeros() as i32 - 24).max(0);
        self.bits -= shift;
        self.range = ((range << shift) - 1) & 0xff;
        bit
    }

    /// Reads the sign of a coefficient: a decision with probability one half, with the
    /// branch-free arithmetic of the reference decoder. For a valid stream it is `get(128)`.
    /// A stream whose partition starts with a byte the encoder cannot write (0xff) makes the
    /// window grow past 31 bits, and there the two disagree; this keeps the reference's result.
    #[inline]
    fn get_sign(&mut self) -> bool {
        if self.bits < 0 {
            self.load();
        }
        let split = self.range >> 1;
        let window = (self.value >> self.bits) as u32;
        // all ones when the decision is 1
        let mask = ((split.wrapping_sub(window) as i32) >> 31) as u32;
        self.bits -= 1;
        self.range = (self.range.wrapping_add(mask) | 1) & 0xff;
        self.value = self
            .value
            .wrapping_sub((((split + 1) & mask) as u64) << (self.bits + 1));
        mask != 0
    }

    /// An unsigned number of `n` bits, most significant first, each bit equally likely.
    fn literal(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.get(128) as u32;
        }
        v
    }

    /// A magnitude of `n` bits followed by a sign bit.
    fn signed(&mut self, n: u32) -> i32 {
        let v = self.literal(n) as i32;
        if self.get(128) {
            -v
        } else {
            v
        }
    }

    /// A signed number preceded by a flag saying whether it is present (0 otherwise).
    fn optional_signed(&mut self, n: u32) -> i32 {
        if self.get(128) {
            self.signed(n)
        } else {
            0
        }
    }
}

/// Dequantisation factors of one segment: (DC, AC) for luma, the luma DC block (Y2) and
/// chroma.
#[derive(Clone, Copy, Default)]
struct Quant {
    y1: [i32; 2],
    y2: [i32; 2],
    uv: [i32; 2],
}

/// Loop-filter parameters of one macroblock.
#[derive(Clone, Copy, Default)]
struct FilterInfo {
    /// Edge limit (0 = the macroblock is not filtered).
    limit: u8,
    interior: u8,
    hev_threshold: u8,
    /// Whether the inner 4x4 edges are filtered too.
    inner: bool,
}

/// What the first partition says about one macroblock.
#[derive(Clone, Copy)]
struct MbModes {
    luma: u8,
    sub: [u8; 16],
    chroma: u8,
    segment: u8,
    skip: bool,
}

/// A decoded key frame: full planes of whole macroblocks, of which `width` x `height` pixels
/// are the picture.
pub(super) struct Frame {
    pub(super) width: usize,
    pub(super) height: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
}

/// Picture size of a lossy bitstream from its 10-byte frame header, with the consistency
/// checks the reference decoder makes before it accepts a frame. `payload` is the size the
/// chunk header gives (the data may hold a padding byte more).
pub(super) fn read_header(data: &[u8], payload: usize) -> Option<(usize, usize)> {
    if data.len() < 10 || data[3..6] != [0x9d, 0x01, 0x2a] {
        return None;
    }
    let tag = data[0] as usize | (data[1] as usize) << 8 | (data[2] as usize) << 16;
    let key_frame = tag & 1 == 0;
    let profile = (tag >> 1) & 7;
    let shown = (tag >> 4) & 1 == 1;
    let first_partition = tag >> 5;
    let width = (data[6] as usize | (data[7] as usize) << 8) & 0x3fff;
    let height = (data[8] as usize | (data[9] as usize) << 8) & 0x3fff;
    if !key_frame
        || profile > 3
        || !shown
        || first_partition >= payload
        || width == 0
        || height == 0
    {
        return None;
    }
    Some((width, height))
}

fn truncated<T>() -> Result<T, String> {
    Err("WebP lossy data ends early".into())
}

/// Decodes a lossy bitstream (the payload of a `VP8 ` chunk) to its YUV planes.
pub(super) fn decode(data: &[u8], limits: &DecodeLimits) -> Result<Frame, String> {
    let (width, height) = match read_header(data, usize::MAX) {
        Some(size) => size,
        None => return Err("WebP lossy frame header is invalid".into()),
    };
    limits.check_canvas(width, height)?;
    let first_len = (data[0] as usize | (data[1] as usize) << 8 | (data[2] as usize) << 16) >> 5;
    let body = &data[10..];
    if first_len > body.len() {
        return truncated();
    }
    let (first, tokens) = body.split_at(first_len);
    let mut br = BoolReader::new(first);

    // colour space and clamping type: one bit each, not used
    br.literal(2);

    // segmentation: per-segment quantiser and filter level, and the tree probabilities of the
    // per-macroblock segment number
    let mut segments_on = false;
    let mut update_map = false;
    let mut absolute = true;
    let mut segment_quant = [0i32; 4];
    let mut segment_filter = [0i32; 4];
    let mut segment_probs = [255u8; 3];
    if br.get(128) {
        segments_on = true;
        update_map = br.get(128);
        if br.get(128) {
            absolute = br.get(128);
            for q in segment_quant.iter_mut() {
                *q = br.optional_signed(7);
            }
            for f in segment_filter.iter_mut() {
                *f = br.optional_signed(6);
            }
        }
        if update_map {
            for p in segment_probs.iter_mut() {
                *p = if br.get(128) {
                    br.literal(8) as u8
                } else {
                    255
                };
            }
        }
    }
    if br.eof {
        return truncated();
    }

    // loop filter
    let simple_filter = br.get(128);
    let filter_level = br.literal(6) as i32;
    let sharpness = br.literal(3) as i32;
    let mut ref_delta = [0i32; 4];
    let mut mode_delta = [0i32; 4];
    let use_deltas = br.get(128);
    if use_deltas && br.get(128) {
        for d in ref_delta.iter_mut() {
            if br.get(128) {
                *d = br.signed(6);
            }
        }
        for d in mode_delta.iter_mut() {
            if br.get(128) {
                *d = br.signed(6);
            }
        }
    }
    if br.eof {
        return truncated();
    }

    // token partitions: their sizes (3 bytes each, all but the last) follow the first
    // partition; a size larger than what is left is cut to it, and the last partition, which
    // takes the rest, must not be empty
    let partition_count = 1usize << br.literal(2);
    let sizes_len = 3 * (partition_count - 1);
    if tokens.len() < sizes_len {
        return truncated();
    }
    let (sizes, mut rest) = tokens.split_at(sizes_len);
    let mut partitions = Vec::with_capacity(partition_count);
    for s in sizes.as_chunks::<3>().0 {
        let size = (s[0] as usize | (s[1] as usize) << 8 | (s[2] as usize) << 16).min(rest.len());
        let (part, tail) = rest.split_at(size);
        partitions.push(BoolReader::new(part));
        rest = tail;
    }
    if rest.is_empty() {
        return truncated();
    }
    partitions.push(BoolReader::new(rest));

    // quantiser indices, then the factors of every segment
    let base_q = br.literal(7) as i32;
    let y1_dc_delta = br.optional_signed(4);
    let y2_dc_delta = br.optional_signed(4);
    let y2_ac_delta = br.optional_signed(4);
    let uv_dc_delta = br.optional_signed(4);
    let uv_ac_delta = br.optional_signed(4);
    let mut quant = [Quant::default(); 4];
    for (i, q) in quant.iter_mut().enumerate() {
        let index = if !segments_on {
            base_q
        } else if absolute {
            segment_quant[i]
        } else {
            segment_quant[i] + base_q
        };
        let dc = |delta: i32, max: i32| DC_QLOOKUP[(index + delta).clamp(0, max) as usize] as i32;
        let ac = |delta: i32| AC_QLOOKUP[(index + delta).clamp(0, 127) as usize] as i32;
        q.y1 = [dc(y1_dc_delta, 127), ac(0)];
        q.y2 = [
            dc(y2_dc_delta, 127) * 2,
            ((ac(y2_ac_delta) * 101_581) >> 16).max(8),
        ];
        q.uv = [dc(uv_dc_delta, 117), ac(uv_ac_delta)];
    }

    // a key frame starts from the default coefficient probabilities; the "keep the
    // probabilities" flag before the updates means nothing for a single frame
    br.get(128);
    let mut probs: Box<CoeffProbs> = Box::new([[[[0u8; 11]; 3]; 8]; 4]);
    let mut k = 0;
    for kind in probs.iter_mut() {
        for band in kind.iter_mut() {
            for context in band.iter_mut() {
                for p in context.iter_mut() {
                    *p = if br.get(COEFF_UPDATE_PROBS[k]) {
                        br.literal(8) as u8
                    } else {
                        DEFAULT_COEFF_PROBS[k]
                    };
                    k += 1;
                }
            }
        }
    }
    let skip_prob = if br.get(128) {
        Some(br.literal(8) as u8)
    } else {
        None
    };

    // loop-filter parameters per segment and per kind of luma prediction (whole / 4x4)
    let filter_kind = if filter_level == 0 {
        0
    } else if simple_filter {
        1
    } else {
        2
    };
    let mut strengths = [[FilterInfo::default(); 2]; 4];
    for (s, pair) in strengths.iter_mut().enumerate() {
        let base = if !segments_on {
            filter_level
        } else if absolute {
            segment_filter[s]
        } else {
            segment_filter[s] + filter_level
        };
        for (sub, info) in pair.iter_mut().enumerate() {
            let mut level = base;
            if use_deltas {
                level += ref_delta[0];
                if sub == 1 {
                    level += mode_delta[0];
                }
            }
            let level = level.clamp(0, 63);
            if level > 0 {
                let mut interior = level;
                if sharpness > 0 {
                    interior >>= if sharpness > 4 { 2 } else { 1 };
                    interior = interior.min(9 - sharpness);
                }
                let interior = interior.max(1);
                info.interior = interior as u8;
                info.limit = (2 * level + interior) as u8;
                info.hev_threshold = if level >= 40 {
                    2
                } else if level >= 15 {
                    1
                } else {
                    0
                };
            }
            info.inner = sub == 1;
        }
    }

    let mb_w = (width + 15) >> 4;
    let mb_h = (height + 15) >> 4;
    let mut frame = Frame {
        width,
        height,
        y: vec![0; mb_w * 16 * mb_h * 16],
        u: vec![0; mb_w * 8 * mb_h * 8],
        v: vec![0; mb_w * 8 * mb_h * 8],
        y_stride: mb_w * 16,
        uv_stride: mb_w * 8,
    };
    let mut filter_info =
        vec![FilterInfo::default(); if filter_kind > 0 { mb_w * mb_h } else { 0 }];

    // context carried from the row above: the 4x4 modes of its bottom blocks, and which of its
    // bottom blocks have coefficients (4 luma, 2 + 2 chroma, 1 for the luma DC block)
    let mut top_modes = vec![B_DC; 4 * mb_w];
    let mut top_nz = vec![[0u8; 9]; mb_w];
    let mut row = vec![
        MbModes {
            luma: DC_PRED,
            sub: [B_DC; 16],
            chroma: DC_PRED,
            segment: 0,
            skip: false,
        };
        mb_w
    ];
    let mut coeffs = [0i16; 384];

    for mb_y in 0..mb_h {
        // the modes of the whole row come first in the first partition
        let mut left_modes = [B_DC; 4];
        for (mb_x, mb) in row.iter_mut().enumerate() {
            let top = &mut top_modes[4 * mb_x..4 * mb_x + 4];
            mb.segment = if update_map {
                if !br.get(segment_probs[0]) {
                    br.get(segment_probs[1]) as u8
                } else {
                    2 + br.get(segment_probs[2]) as u8
                }
            } else {
                0
            };
            mb.skip = match skip_prob {
                Some(p) => br.get(p),
                None => false,
            };
            if br.get(145) {
                mb.luma = if br.get(156) {
                    if br.get(128) {
                        TM_PRED
                    } else {
                        H_PRED
                    }
                } else if br.get(163) {
                    V_PRED
                } else {
                    DC_PRED
                };
                top.fill(mb.luma);
                left_modes = [mb.luma; 4];
            } else {
                mb.luma = B_PRED;
                for (by, left) in left_modes.iter_mut().enumerate() {
                    let mut mode = *left;
                    for (bx, above) in top.iter_mut().enumerate() {
                        let at = (*above as usize * 10 + mode as usize) * 9;
                        mode = read_sub_mode(&mut br, &KF_BMODE_PROB[at..at + 9]);
                        *above = mode;
                        mb.sub[by * 4 + bx] = mode;
                    }
                    *left = mode;
                }
            }
            mb.chroma = if !br.get(142) {
                DC_PRED
            } else if !br.get(114) {
                V_PRED
            } else if br.get(183) {
                TM_PRED
            } else {
                H_PRED
            };
        }
        if br.eof {
            return truncated();
        }

        let tokens = &mut partitions[mb_y & (partition_count - 1)];
        let mut left_nz = [0u8; 9];
        for (mb_x, mb) in row.iter().enumerate() {
            let q = &quant[mb.segment as usize];
            let top = &mut top_nz[mb_x];
            let mut has_coeffs = false;
            // two bits per block (16 luma, 4 U, 4 V): the kind of residual it has
            let mut residual = 0u64;
            if !mb.skip {
                coeffs.fill(0);
                // the scan position the luma blocks start at, and their kind of probabilities
                let (first, kind) = if mb.luma != B_PRED {
                    // the DC coefficients of the 16 luma blocks are coded as a block of their
                    // own, transformed by the WHT
                    let mut dc = [0i16; 16];
                    let ctx = (top[8] + left_nz[8]) as usize;
                    let n = read_coeffs(tokens, &probs[1], ctx, q.y2, 0, &mut dc);
                    top[8] = (n > 0) as u8;
                    left_nz[8] = top[8];
                    inverse_wht(&dc, &mut coeffs);
                    (1, 0)
                } else {
                    (0, 3)
                };
                for (by, left) in left_nz.iter_mut().take(4).enumerate() {
                    for (bx, above) in top.iter_mut().take(4).enumerate() {
                        let block = by * 4 + bx;
                        let out = &mut coeffs[block * 16..block * 16 + 16];
                        let ctx = (*left + *above) as usize;
                        let n = read_coeffs(tokens, &probs[kind], ctx, q.y1, first, out);
                        let flag = (n > first) as u8;
                        *left = flag;
                        *above = flag;
                        residual |= residual_kind(n, out[0]) << (2 * block);
                    }
                }
                for plane in 0..2 {
                    for by in 0..2 {
                        for bx in 0..2 {
                            let block = 16 + plane * 4 + by * 2 + bx;
                            let out = &mut coeffs[block * 16..block * 16 + 16];
                            let (l, t) = (4 + plane * 2 + by, 4 + plane * 2 + bx);
                            let ctx = (left_nz[l] + top[t]) as usize;
                            let n = read_coeffs(tokens, &probs[2], ctx, q.uv, 0, out);
                            let flag = (n > 0) as u8;
                            left_nz[l] = flag;
                            top[t] = flag;
                            residual |= residual_kind(n, out[0]) << (2 * block);
                        }
                    }
                }
                has_coeffs = residual != 0;
            } else {
                // a skipped macroblock has no coefficients; the DC block context stays as it
                // is when the macroblock has no DC block
                top[..8].fill(0);
                left_nz[..8].fill(0);
                if mb.luma != B_PRED {
                    top[8] = 0;
                    left_nz[8] = 0;
                }
            }
            if tokens.eof {
                return truncated();
            }
            if filter_kind > 0 {
                let sub = (mb.luma == B_PRED) as usize;
                let mut info = strengths[mb.segment as usize][sub];
                info.inner |= has_coeffs;
                filter_info[mb_y * mb_w + mb_x] = info;
            }
            frame.reconstruct(mb, mb_x, mb_y, mb_w, &coeffs, residual);
        }
    }

    if filter_kind > 0 {
        frame.loop_filter(&filter_info, mb_w, mb_h, filter_kind == 1);
    }
    Ok(frame)
}

/// No residual in a block.
const RESIDUAL_NONE: u64 = 0;
/// Only the DC coefficient is set.
const RESIDUAL_DC: u64 = 1;
/// Only the first three coefficients in scan order can be set.
const RESIDUAL_FEW: u64 = 2;
/// Any coefficient can be set.
const RESIDUAL_FULL: u64 = 3;

/// The kind of residual of a block from the scan position after its last token and its DC
/// coefficient.
fn residual_kind(end: usize, dc: i16) -> u64 {
    if end > 3 {
        RESIDUAL_FULL
    } else if end > 1 {
        RESIDUAL_FEW
    } else if dc != 0 {
        RESIDUAL_DC
    } else {
        RESIDUAL_NONE
    }
}

/// Reads the mode of one 4x4 block with the probabilities its two neighbours select.
fn read_sub_mode(br: &mut BoolReader, p: &[u8]) -> u8 {
    if !br.get(p[0]) {
        B_DC
    } else if !br.get(p[1]) {
        B_TM
    } else if !br.get(p[2]) {
        B_VE
    } else if !br.get(p[3]) {
        if !br.get(p[4]) {
            B_HE
        } else if !br.get(p[5]) {
            B_RD
        } else {
            B_VR
        }
    } else if !br.get(p[6]) {
        B_LD
    } else if !br.get(p[7]) {
        B_VL
    } else if !br.get(p[8]) {
        B_HD
    } else {
        B_HU
    }
}

/// Reads the tokens of one block from scan position `first` and stores the dequantised
/// coefficients. Returns the scan position after the last token read (`first` when the block
/// is empty). A coefficient is kept in 16 bits, as the reference decoder keeps it.
fn read_coeffs(
    br: &mut BoolReader,
    probs: &[[[u8; 11]; 3]; 8],
    ctx: usize,
    dq: [i32; 2],
    first: usize,
    out: &mut [i16],
) -> usize {
    let mut n = first;
    let mut p = &probs[BAND[n]][ctx];
    while n < 16 {
        if !br.get(p[0]) {
            return n;
        }
        while !br.get(p[1]) {
            n += 1;
            if n == 16 {
                return 16;
            }
            p = &probs[BAND[n]][0];
        }
        let (magnitude, next_ctx) = if !br.get(p[2]) {
            (1, 1)
        } else {
            (read_large(br, p), 2)
        };
        let value = if br.get_sign() { -magnitude } else { magnitude };
        out[ZIGZAG[n]] = (value * dq[(n > 0) as usize]) as i16;
        n += 1;
        if n < 16 {
            p = &probs[BAND[n]][next_ctx];
        }
    }
    16
}

/// The magnitude of a coefficient larger than 1.
fn read_large(br: &mut BoolReader, p: &[u8; 11]) -> i32 {
    if !br.get(p[3]) {
        if !br.get(p[4]) {
            2
        } else {
            3 + br.get(p[5]) as i32
        }
    } else if !br.get(p[6]) {
        if !br.get(p[7]) {
            5 + br.get(159) as i32
        } else {
            let high = br.get(165) as i32;
            7 + 2 * high + br.get(145) as i32
        }
    } else {
        let wide = br.get(p[8]);
        let category = 2 * wide as usize + br.get(p[9 + wide as usize]) as usize;
        let extra: &[u8] = match category {
            0 => &CATEGORY_3,
            1 => &CATEGORY_4,
            2 => &CATEGORY_5,
            _ => &CATEGORY_6,
        };
        let mut v = 0;
        for &prob in extra {
            v = 2 * v + br.get(prob) as i32;
        }
        v + 3 + (8 << category)
    }
}

/// Inverse Walsh-Hadamard transform of the luma DC block: output `i` is the DC coefficient of
/// luma block `i`.
fn inverse_wht(dc: &[i16; 16], coeffs: &mut [i16; 384]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = dc[i] as i32 + dc[12 + i] as i32;
        let a1 = dc[4 + i] as i32 + dc[8 + i] as i32;
        let a2 = dc[4 + i] as i32 - dc[8 + i] as i32;
        let a3 = dc[i] as i32 - dc[12 + i] as i32;
        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for i in 0..4 {
        let rounded = tmp[4 * i] + 3;
        let a0 = rounded + tmp[4 * i + 3];
        let a1 = tmp[4 * i + 1] + tmp[4 * i + 2];
        let a2 = tmp[4 * i + 1] - tmp[4 * i + 2];
        let a3 = rounded - tmp[4 * i + 3];
        coeffs[(4 * i) * 16] = ((a0 + a1) >> 3) as i16;
        coeffs[(4 * i + 1) * 16] = ((a3 + a2) >> 3) as i16;
        coeffs[(4 * i + 2) * 16] = ((a0 - a1) >> 3) as i16;
        coeffs[(4 * i + 3) * 16] = ((a3 - a2) >> 3) as i16;
    }
}

/// `x * sqrt(2) * cos(pi / 8)` in the fixed point of the RFC.
#[inline(always)]
fn mul_cos(x: i32) -> i32 {
    ((x as i64 * 20091) >> 16) as i32 + x
}

/// `x * sqrt(2) * sin(pi / 8)` in the fixed point of the RFC.
#[inline(always)]
fn mul_sin(x: i32) -> i32 {
    ((x as i64 * 35468) >> 16) as i32
}

/// The high 16 bits of a 16 x 16 bit signed product.
#[inline(always)]
fn mul_high(x: i16, k: i16) -> i16 {
    ((x as i32 * k as i32) >> 16) as i16
}

/// One pass of the inverse DCT over four values, in 16-bit arithmetic that wraps. The two
/// rotations are `x * 20091 >> 16 + x` and `x * 35468 >> 16`, the second written with the
/// factor 35468 - 65536 so that it fits a signed 16-bit multiplication.
#[inline(always)]
fn dct_pass_16(c0: i16, c1: i16, c2: i16, c3: i16) -> [i16; 4] {
    const K1: i16 = 20091;
    const K2: i16 = -30068;
    let a = c0.wrapping_add(c2);
    let b = c0.wrapping_sub(c2);
    let c = mul_high(c1, K2)
        .wrapping_sub(mul_high(c3, K1))
        .wrapping_add(c1.wrapping_sub(c3));
    let d = mul_high(c1, K1)
        .wrapping_add(mul_high(c3, K2))
        .wrapping_add(c1.wrapping_add(c3));
    [
        a.wrapping_add(d),
        b.wrapping_add(c),
        b.wrapping_sub(c),
        a.wrapping_sub(d),
    ]
}

/// Inverse DCT of one 4x4 block with any coefficients, added to the prediction at `at` and
/// clamped. It works in 16 bits that wrap, like the reference decoder: for coefficients an
/// encoder can produce nothing wraps and the result is that of the RFC; for larger ones a
/// hostile file can hold, the wrapped result is the reference's.
fn add_inverse_dct_full(c: &[i16], plane: &mut [u8], at: usize, stride: usize) {
    let mut tmp = [0i16; 16];
    for i in 0..4 {
        let column = dct_pass_16(c[i], c[4 + i], c[8 + i], c[12 + i]);
        tmp[4 * i..4 * i + 4].copy_from_slice(&column);
    }
    for i in 0..4 {
        let row = dct_pass_16(tmp[i].wrapping_add(4), tmp[4 + i], tmp[8 + i], tmp[12 + i]);
        let px = &mut plane[at + i * stride..at + i * stride + 4];
        for (p, v) in px.iter_mut().zip(row) {
            *p = (*p as i16).wrapping_add(v >> 3).clamp(0, 255) as u8;
        }
    }
}

/// Adds the residual of one block according to its kind.
#[inline]
fn add_residual(kind: u64, c: &[i16], plane: &mut [u8], at: usize, stride: usize) {
    match kind {
        RESIDUAL_NONE => {}
        RESIDUAL_FULL => add_inverse_dct_full(c, plane, at, stride),
        _ => add_inverse_dct(c, plane, at, stride),
    }
}

/// Inverse DCT of one 4x4 block of which at most the first three coefficients in scan order
/// are set, added to the prediction at `at` and clamped. Exact integer arithmetic.
fn add_inverse_dct(c: &[i16], plane: &mut [u8], at: usize, stride: usize) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let (c0, c1, c2, c3) = (
            c[i] as i32,
            c[4 + i] as i32,
            c[8 + i] as i32,
            c[12 + i] as i32,
        );
        let a = c0 + c2;
        let b = c0 - c2;
        let c = mul_sin(c1) - mul_cos(c3);
        let d = mul_cos(c1) + mul_sin(c3);
        tmp[4 * i] = a + d;
        tmp[4 * i + 1] = b + c;
        tmp[4 * i + 2] = b - c;
        tmp[4 * i + 3] = a - d;
    }
    for i in 0..4 {
        let rounded = tmp[i] + 4;
        let a = rounded + tmp[8 + i];
        let b = rounded - tmp[8 + i];
        let c = mul_sin(tmp[4 + i]) - mul_cos(tmp[12 + i]);
        let d = mul_cos(tmp[4 + i]) + mul_sin(tmp[12 + i]);
        let px = &mut plane[at + i * stride..at + i * stride + 4];
        px[0] = (px[0] as i32 + ((a + d) >> 3)).clamp(0, 255) as u8;
        px[1] = (px[1] as i32 + ((b + c) >> 3)).clamp(0, 255) as u8;
        px[2] = (px[2] as i32 + ((b - c) >> 3)).clamp(0, 255) as u8;
        px[3] = (px[3] as i32 + ((a - d) >> 3)).clamp(0, 255) as u8;
    }
}

#[inline(always)]
fn avg2(a: u8, b: u8) -> u8 {
    ((a as u32 + b as u32 + 1) >> 1) as u8
}

#[inline(always)]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u32 + 2 * b as u32 + c as u32 + 2) >> 2) as u8
}

/// Predicts a whole square block (16x16 luma or 8x8 chroma) at (`x0`, `y0`). Outside the
/// frame the row above reads as 127 and the column to the left as 129; DC prediction uses
/// only the edges that exist (128 when there is none).
fn predict_block(plane: &mut [u8], stride: usize, x0: usize, y0: usize, size: usize, mode: u8) {
    let has_top = y0 > 0;
    let has_left = x0 > 0;
    let at = y0 * stride + x0;
    let mut above = [127u8; 16];
    let mut left = [129u8; 16];
    if has_top {
        above[..size].copy_from_slice(&plane[at - stride..at - stride + size]);
    }
    if has_left {
        for (j, l) in left.iter_mut().take(size).enumerate() {
            *l = plane[at + j * stride - 1];
        }
    }
    match mode {
        DC_PRED => {
            let mut sum = 0u32;
            let mut edges = 0;
            if has_top {
                sum += above[..size].iter().map(|&p| p as u32).sum::<u32>();
                edges += 1;
            }
            if has_left {
                sum += left[..size].iter().map(|&p| p as u32).sum::<u32>();
                edges += 1;
            }
            let log = size.trailing_zeros();
            let value = match edges {
                2 => (sum + size as u32) >> (log + 1),
                1 => (sum + size as u32 / 2) >> log,
                _ => 128,
            } as u8;
            for j in 0..size {
                plane[at + j * stride..at + j * stride + size].fill(value);
            }
        }
        V_PRED => {
            for j in 0..size {
                plane[at + j * stride..at + j * stride + size].copy_from_slice(&above[..size]);
            }
        }
        H_PRED => {
            for j in 0..size {
                plane[at + j * stride..at + j * stride + size].fill(left[j]);
            }
        }
        _ => {
            let corner = if !has_top {
                127
            } else if !has_left {
                129
            } else {
                plane[at - stride - 1] as i32
            };
            for j in 0..size {
                let base = left[j] as i32 - corner;
                let row = &mut plane[at + j * stride..at + j * stride + size];
                for (p, &a) in row.iter_mut().zip(&above[..size]) {
                    *p = (base + a as i32).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Predicts one 4x4 luma block from `a` (the 4 pixels above and the 4 above to the right),
/// `l` (the 4 pixels to the left) and `p` (the pixel above-left).
fn predict_sub_block(mode: u8, a: &[u8; 8], l: &[u8; 4], p: u8) -> [[u8; 4]; 4] {
    let mut b = [[0u8; 4]; 4];
    // the edge from bottom-left over the corner to top-right
    let e = [l[3], l[2], l[1], l[0], p, a[0], a[1], a[2], a[3]];
    match mode {
        B_DC => {
            let sum: u32 = a[..4].iter().chain(l.iter()).map(|&v| v as u32).sum();
            b = [[((sum + 4) >> 3) as u8; 4]; 4];
        }
        B_TM => {
            for r in 0..4 {
                for c in 0..4 {
                    b[r][c] = (l[r] as i32 + a[c] as i32 - p as i32).clamp(0, 255) as u8;
                }
            }
        }
        B_VE => {
            let row = [
                avg3(p, a[0], a[1]),
                avg3(a[0], a[1], a[2]),
                avg3(a[1], a[2], a[3]),
                avg3(a[2], a[3], a[4]),
            ];
            b = [row; 4];
        }
        B_HE => {
            b[0] = [avg3(p, l[0], l[1]); 4];
            b[1] = [avg3(l[0], l[1], l[2]); 4];
            b[2] = [avg3(l[1], l[2], l[3]); 4];
            b[3] = [avg3(l[2], l[3], l[3]); 4];
        }
        B_LD => {
            for (r, row) in b.iter_mut().enumerate() {
                for (c, px) in row.iter_mut().enumerate() {
                    let i = r + c;
                    *px = avg3(a[i], a[i + 1], a[(i + 2).min(7)]);
                }
            }
        }
        B_RD => {
            for (r, row) in b.iter_mut().enumerate() {
                for (c, px) in row.iter_mut().enumerate() {
                    let i = 4 + c - r;
                    *px = avg3(e[i - 1], e[i], e[i + 1]);
                }
            }
        }
        B_VR => {
            b[3][0] = avg3(e[1], e[2], e[3]);
            b[2][0] = avg3(e[2], e[3], e[4]);
            b[1][0] = avg3(e[3], e[4], e[5]);
            b[3][1] = b[1][0];
            b[0][0] = avg2(e[4], e[5]);
            b[2][1] = b[0][0];
            b[1][1] = avg3(e[4], e[5], e[6]);
            b[3][2] = b[1][1];
            b[0][1] = avg2(e[5], e[6]);
            b[2][2] = b[0][1];
            b[1][2] = avg3(e[5], e[6], e[7]);
            b[3][3] = b[1][2];
            b[0][2] = avg2(e[6], e[7]);
            b[2][3] = b[0][2];
            b[1][3] = avg3(e[6], e[7], e[8]);
            b[0][3] = avg2(e[7], e[8]);
        }
        B_VL => {
            b[0][0] = avg2(a[0], a[1]);
            b[1][0] = avg3(a[0], a[1], a[2]);
            b[0][1] = avg2(a[1], a[2]);
            b[2][0] = b[0][1];
            b[1][1] = avg3(a[1], a[2], a[3]);
            b[3][0] = b[1][1];
            b[0][2] = avg2(a[2], a[3]);
            b[2][1] = b[0][2];
            b[1][2] = avg3(a[2], a[3], a[4]);
            b[3][1] = b[1][2];
            b[0][3] = avg2(a[3], a[4]);
            b[2][2] = b[0][3];
            b[1][3] = avg3(a[3], a[4], a[5]);
            b[3][2] = b[1][3];
            b[2][3] = avg3(a[4], a[5], a[6]);
            b[3][3] = avg3(a[5], a[6], a[7]);
        }
        B_HD => {
            b[3][0] = avg2(e[0], e[1]);
            b[3][1] = avg3(e[0], e[1], e[2]);
            b[2][0] = avg2(e[1], e[2]);
            b[3][2] = b[2][0];
            b[2][1] = avg3(e[1], e[2], e[3]);
            b[3][3] = b[2][1];
            b[1][0] = avg2(e[2], e[3]);
            b[2][2] = b[1][0];
            b[1][1] = avg3(e[2], e[3], e[4]);
            b[2][3] = b[1][1];
            b[0][0] = avg2(e[3], e[4]);
            b[1][2] = b[0][0];
            b[0][1] = avg3(e[3], e[4], e[5]);
            b[1][3] = b[0][1];
            b[0][2] = avg3(e[4], e[5], e[6]);
            b[0][3] = avg3(e[5], e[6], e[7]);
        }
        _ => {
            b[0][0] = avg2(l[0], l[1]);
            b[0][1] = avg3(l[0], l[1], l[2]);
            b[0][2] = avg2(l[1], l[2]);
            b[1][0] = b[0][2];
            b[0][3] = avg3(l[1], l[2], l[3]);
            b[1][1] = b[0][3];
            b[1][2] = avg2(l[2], l[3]);
            b[2][0] = b[1][2];
            b[1][3] = avg3(l[2], l[3], l[3]);
            b[2][1] = b[1][3];
            b[2][2] = l[3];
            b[2][3] = l[3];
            b[3] = [l[3]; 4];
        }
    }
    b
}

impl Frame {
    /// Predicts one macroblock from the pixels decoded so far and adds its residual.
    fn reconstruct(
        &mut self,
        mb: &MbModes,
        mb_x: usize,
        mb_y: usize,
        mb_w: usize,
        coeffs: &[i16; 384],
        residual: u64,
    ) {
        let stride = self.y_stride;
        let (x0, y0) = (mb_x * 16, mb_y * 16);
        if mb.luma == B_PRED {
            // the 4 pixels above and to the right of the macroblock: they are taken from the
            // row above also for the blocks further down its right column
            let mut above_right = [127u8; 4];
            if mb_y > 0 {
                let at = (y0 - 1) * stride + x0 + 16;
                if mb_x + 1 < mb_w {
                    above_right.copy_from_slice(&self.y[at..at + 4]);
                } else {
                    above_right = [self.y[at - 1]; 4];
                }
            }
            for block in 0..16 {
                let (bx, by) = (block & 3, block >> 2);
                let (x, y) = (x0 + bx * 4, y0 + by * 4);
                let at = y * stride + x;
                let mut a = [127u8; 8];
                if y > 0 {
                    a[..4].copy_from_slice(&self.y[at - stride..at - stride + 4]);
                    if bx < 3 {
                        a[4..].copy_from_slice(&self.y[at - stride + 4..at - stride + 8]);
                    } else {
                        a[4..].copy_from_slice(&above_right);
                    }
                }
                let mut l = [129u8; 4];
                if x > 0 {
                    for (j, v) in l.iter_mut().enumerate() {
                        *v = self.y[at + j * stride - 1];
                    }
                }
                let p = if y == 0 {
                    127
                } else if x == 0 {
                    129
                } else {
                    self.y[at - stride - 1]
                };
                let predicted = predict_sub_block(mb.sub[block], &a, &l, p);
                for (j, row) in predicted.iter().enumerate() {
                    self.y[at + j * stride..at + j * stride + 4].copy_from_slice(row);
                }
                add_residual(
                    (residual >> (2 * block)) & 3,
                    &coeffs[block * 16..block * 16 + 16],
                    &mut self.y,
                    at,
                    stride,
                );
            }
        } else {
            predict_block(&mut self.y, stride, x0, y0, 16, mb.luma);
            for block in 0..16 {
                let at = (y0 + (block >> 2) * 4) * stride + x0 + (block & 3) * 4;
                add_residual(
                    (residual >> (2 * block)) & 3,
                    &coeffs[block * 16..block * 16 + 16],
                    &mut self.y,
                    at,
                    stride,
                );
            }
        }
        let stride = self.uv_stride;
        let (x0, y0) = (mb_x * 8, mb_y * 8);
        for (plane, first) in [(&mut self.u, 16), (&mut self.v, 20)] {
            predict_block(plane, stride, x0, y0, 8, mb.chroma);
            // a chroma plane with more than DC coefficients gets the full transform on all
            // four blocks
            let kinds = (residual >> (2 * first)) & 0xff;
            let any_ac = kinds & 0xaa != 0;
            for i in 0..4 {
                let block = first + i;
                let kind = if any_ac {
                    RESIDUAL_FULL
                } else {
                    (kinds >> (2 * i)) & 3
                };
                let at = (y0 + (i >> 1) * 4) * stride + x0 + (i & 1) * 4;
                add_residual(
                    kind,
                    &coeffs[block * 16..block * 16 + 16],
                    plane,
                    at,
                    stride,
                );
            }
        }
    }

    /// Applies the loop filter to the whole frame, macroblock by macroblock in raster order:
    /// the left macroblock edge, the inner vertical edges, the top macroblock edge, the inner
    /// horizontal edges. The simple filter touches luma only.
    fn loop_filter(&mut self, info: &[FilterInfo], mb_w: usize, mb_h: usize, simple: bool) {
        let ys = self.y_stride;
        let uvs = self.uv_stride;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let f = info[mb_y * mb_w + mb_x];
                if f.limit == 0 {
                    continue;
                }
                let limit = f.limit as i32;
                let interior = f.interior as i32;
                let hev = f.hev_threshold as i32;
                let y_at = mb_y * 16 * ys + mb_x * 16;
                let uv_at = mb_y * 8 * uvs + mb_x * 8;
                if simple {
                    if mb_x > 0 {
                        filter_simple(&mut self.y, y_at, 1, ys, 16, limit + 4);
                    }
                    if f.inner {
                        for k in 1..4 {
                            filter_simple(&mut self.y, y_at + 4 * k, 1, ys, 16, limit);
                        }
                    }
                    if mb_y > 0 {
                        filter_simple(&mut self.y, y_at, ys, 1, 16, limit + 4);
                    }
                    if f.inner {
                        for k in 1..4 {
                            filter_simple(&mut self.y, y_at + 4 * k * ys, ys, 1, 16, limit);
                        }
                    }
                    continue;
                }
                if mb_x > 0 {
                    filter_normal(&mut self.y, y_at, 1, ys, 16, limit + 4, interior, hev, true);
                    filter_normal(
                        &mut self.u,
                        uv_at,
                        1,
                        uvs,
                        8,
                        limit + 4,
                        interior,
                        hev,
                        true,
                    );
                    filter_normal(
                        &mut self.v,
                        uv_at,
                        1,
                        uvs,
                        8,
                        limit + 4,
                        interior,
                        hev,
                        true,
                    );
                }
                if f.inner {
                    for k in 1..4 {
                        filter_normal(
                            &mut self.y,
                            y_at + 4 * k,
                            1,
                            ys,
                            16,
                            limit,
                            interior,
                            hev,
                            false,
                        );
                    }
                    filter_normal(
                        &mut self.u,
                        uv_at + 4,
                        1,
                        uvs,
                        8,
                        limit,
                        interior,
                        hev,
                        false,
                    );
                    filter_normal(
                        &mut self.v,
                        uv_at + 4,
                        1,
                        uvs,
                        8,
                        limit,
                        interior,
                        hev,
                        false,
                    );
                }
                if mb_y > 0 {
                    filter_normal(&mut self.y, y_at, ys, 1, 16, limit + 4, interior, hev, true);
                    filter_normal(
                        &mut self.u,
                        uv_at,
                        uvs,
                        1,
                        8,
                        limit + 4,
                        interior,
                        hev,
                        true,
                    );
                    filter_normal(
                        &mut self.v,
                        uv_at,
                        uvs,
                        1,
                        8,
                        limit + 4,
                        interior,
                        hev,
                        true,
                    );
                }
                if f.inner {
                    for k in 1..4 {
                        filter_normal(
                            &mut self.y,
                            y_at + 4 * k * ys,
                            ys,
                            1,
                            16,
                            limit,
                            interior,
                            hev,
                            false,
                        );
                    }
                    filter_normal(
                        &mut self.u,
                        uv_at + 4 * uvs,
                        uvs,
                        1,
                        8,
                        limit,
                        interior,
                        hev,
                        false,
                    );
                    filter_normal(
                        &mut self.v,
                        uv_at + 4 * uvs,
                        uvs,
                        1,
                        8,
                        limit,
                        interior,
                        hev,
                        false,
                    );
                }
            }
        }
    }

    /// Writes the picture as RGB: `step` bytes per pixel (3 or 4, the first three are set),
    /// rows `stride` bytes apart, starting at `out[0]`.
    pub(super) fn write_rgb(&self, out: &mut [u8], stride: usize, step: usize) {
        let (w, h) = (self.width, self.height);
        let chroma_rows = h.div_ceil(2);
        for y in 0..h {
            // an odd row lies between its own chroma row (nearer) and the next one; an even
            // row between its own and the previous one; the frame edges repeat the edge row
            let near = y / 2;
            let far = if y & 1 == 1 {
                (near + 1).min(chroma_rows - 1)
            } else {
                near.saturating_sub(1)
            };
            let luma = &self.y[y * self.y_stride..y * self.y_stride + w];
            let row = &mut out[y * stride..];
            upsample_row(
                luma,
                &self.u[near * self.uv_stride..],
                &self.u[far * self.uv_stride..],
                &self.v[near * self.uv_stride..],
                &self.v[far * self.uv_stride..],
                row,
                step,
            );
        }
    }
}

#[inline(always)]
fn tap(buf: &[u8], at: usize) -> i32 {
    buf[at] as i32
}

/// The common edge test: the step across the edge is small enough to be a coding artefact.
#[inline(always)]
fn edge_is_flat(p1: i32, p0: i32, q0: i32, q1: i32, limit2: i32) -> bool {
    4 * (p0 - q0).abs() + (p1 - q1).abs() <= limit2
}

/// Adjusts the two pixels next to the edge, using the outer pair as well.
#[inline(always)]
fn adjust_inner_pair(buf: &mut [u8], at: usize, across: usize) {
    let (p1, p0, q0, q1) = (
        tap(buf, at - 2 * across),
        tap(buf, at - across),
        tap(buf, at),
        tap(buf, at + across),
    );
    let a = 3 * (q0 - p0) + (p1 - q1).clamp(-128, 127);
    let a1 = ((a + 4) >> 3).clamp(-16, 15);
    let a2 = ((a + 3) >> 3).clamp(-16, 15);
    buf[at - across] = (p0 + a2).clamp(0, 255) as u8;
    buf[at] = (q0 - a1).clamp(0, 255) as u8;
}

/// The simple loop filter on `count` pixels of one edge. `at` is the first pixel after the
/// edge, `across` the distance between pixels across the edge, `along` along it.
fn filter_simple(buf: &mut [u8], at: usize, across: usize, along: usize, count: usize, limit: i32) {
    let limit2 = 2 * limit + 1;
    for i in 0..count {
        let at = at + i * along;
        let (p1, p0, q0, q1) = (
            tap(buf, at - 2 * across),
            tap(buf, at - across),
            tap(buf, at),
            tap(buf, at + across),
        );
        if edge_is_flat(p1, p0, q0, q1, limit2) {
            adjust_inner_pair(buf, at, across);
        }
    }
}

/// The normal loop filter on `count` pixels of one edge: a macroblock edge (`mb_edge`, up to
/// three pixels changed on each side) or an inner edge (up to two).
#[allow(clippy::too_many_arguments)]
fn filter_normal(
    buf: &mut [u8],
    at: usize,
    across: usize,
    along: usize,
    count: usize,
    limit: i32,
    interior: i32,
    hev_threshold: i32,
    mb_edge: bool,
) {
    let limit2 = 2 * limit + 1;
    for i in 0..count {
        let at = at + i * along;
        let p3 = tap(buf, at - 4 * across);
        let p2 = tap(buf, at - 3 * across);
        let p1 = tap(buf, at - 2 * across);
        let p0 = tap(buf, at - across);
        let q0 = tap(buf, at);
        let q1 = tap(buf, at + across);
        let q2 = tap(buf, at + 2 * across);
        let q3 = tap(buf, at + 3 * across);
        if !edge_is_flat(p1, p0, q0, q1, limit2) {
            continue;
        }
        if (p3 - p2).abs() > interior
            || (p2 - p1).abs() > interior
            || (p1 - p0).abs() > interior
            || (q3 - q2).abs() > interior
            || (q2 - q1).abs() > interior
            || (q1 - q0).abs() > interior
        {
            continue;
        }
        let high_variance = (p1 - p0).abs() > hev_threshold || (q1 - q0).abs() > hev_threshold;
        if high_variance {
            adjust_inner_pair(buf, at, across);
        } else if mb_edge {
            let a = (3 * (q0 - p0) + (p1 - q1).clamp(-128, 127)).clamp(-128, 127);
            let a1 = (27 * a + 63) >> 7;
            let a2 = (18 * a + 63) >> 7;
            let a3 = (9 * a + 63) >> 7;
            buf[at - 3 * across] = (p2 + a3).clamp(0, 255) as u8;
            buf[at - 2 * across] = (p1 + a2).clamp(0, 255) as u8;
            buf[at - across] = (p0 + a1).clamp(0, 255) as u8;
            buf[at] = (q0 - a1).clamp(0, 255) as u8;
            buf[at + across] = (q1 - a2).clamp(0, 255) as u8;
            buf[at + 2 * across] = (q2 - a3).clamp(0, 255) as u8;
        } else {
            let a = 3 * (q0 - p0);
            let a1 = ((a + 4) >> 3).clamp(-16, 15);
            let a2 = ((a + 3) >> 3).clamp(-16, 15);
            let a3 = (a1 + 1) >> 1;
            buf[at - 2 * across] = (p1 + a3).clamp(0, 255) as u8;
            buf[at - across] = (p0 + a2).clamp(0, 255) as u8;
            buf[at] = (q0 - a1).clamp(0, 255) as u8;
            buf[at + across] = (q1 - a3).clamp(0, 255) as u8;
        }
    }
}

/// Converts one YUV sample to RGB in 14-bit fixed point: the three sums are kept with 6
/// fraction bits, cut off (not rounded) and clamped to 0..=255.
#[inline(always)]
fn yuv_to_rgb(y: u8, u: i32, v: i32, out: &mut [u8]) {
    let luma = (y as i32 * 19077) >> 8;
    let clip = |v: i32| (v >> 6).clamp(0, 255) as u8;
    out[0] = clip(luma + ((v * 26149) >> 8) - 14234);
    out[1] = clip(luma - ((u * 6419) >> 8) - ((v * 13320) >> 8) + 8708);
    out[2] = clip(luma + ((u * 33050) >> 8) - 17685);
}

/// Writes one row of RGB. `near_*` is the chroma row this luma row lies closest to, `far_*`
/// the chroma row on its other side.
fn upsample_row(
    luma: &[u8],
    near_u: &[u8],
    far_u: &[u8],
    near_v: &[u8],
    far_v: &[u8],
    out: &mut [u8],
    step: usize,
) {
    let w = luma.len();
    let edge = |near: &[u8], far: &[u8], c: usize| (3 * near[c] as i32 + far[c] as i32 + 2) >> 2;
    let inner = |near: &[u8], far: &[u8], c: usize, other: usize| {
        (9 * near[c] as i32 + 3 * near[other] as i32 + 3 * far[c] as i32 + far[other] as i32 + 8)
            >> 4
    };
    yuv_to_rgb(luma[0], edge(near_u, far_u, 0), edge(near_v, far_v, 0), out);
    // pixels 2c - 1 and 2c lie between chroma columns c - 1 and c
    for c in 1..=(w - 1) / 2 {
        let x = 2 * c - 1;
        yuv_to_rgb(
            luma[x],
            inner(near_u, far_u, c - 1, c),
            inner(near_v, far_v, c - 1, c),
            &mut out[x * step..],
        );
        yuv_to_rgb(
            luma[x + 1],
            inner(near_u, far_u, c, c - 1),
            inner(near_v, far_v, c, c - 1),
            &mut out[(x + 1) * step..],
        );
    }
    if w & 1 == 0 {
        let c = w / 2 - 1;
        yuv_to_rgb(
            luma[w - 1],
            edge(near_u, far_u, c),
            edge(near_v, far_v, c),
            &mut out[(w - 1) * step..],
        );
    }
}

/// Tree probabilities of a 4x4 mode in a key frame, indexed by the mode above, then the mode
/// to the left (9 per pair).
#[rustfmt::skip]
const KF_BMODE_PROB: [u8; 900] = [
    231, 120, 48, 89, 115, 113, 120, 152, 112,
    152, 179, 64, 126, 170, 118, 46, 70, 95,
    175, 69, 143, 80, 85, 82, 72, 155, 103,
    56, 58, 10, 171, 218, 189, 17, 13, 152,
    144, 71, 10, 38, 171, 213, 144, 34, 26,
    114, 26, 17, 163, 44, 195, 21, 10, 173,
    121, 24, 80, 195, 26, 62, 44, 64, 85,
    170, 46, 55, 19, 136, 160, 33, 206, 71,
    63, 20, 8, 114, 114, 208, 12, 9, 226,
    81, 40, 11, 96, 182, 84, 29, 16, 36,
    134, 183, 89, 137, 98, 101, 106, 165, 148,
    72, 187, 100, 130, 157, 111, 32, 75, 80,
    66, 102, 167, 99, 74, 62, 40, 234, 128,
    41, 53, 9, 178, 241, 141, 26, 8, 107,
    104, 79, 12, 27, 217, 255, 87, 17, 7,
    74, 43, 26, 146, 73, 166, 49, 23, 157,
    65, 38, 105, 160, 51, 52, 31, 115, 128,
    87, 68, 71, 44, 114, 51, 15, 186, 23,
    47, 41, 14, 110, 182, 183, 21, 17, 194,
    66, 45, 25, 102, 197, 189, 23, 18, 22,
    88, 88, 147, 150, 42, 46, 45, 196, 205,
    43, 97, 183, 117, 85, 38, 35, 179, 61,
    39, 53, 200, 87, 26, 21, 43, 232, 171,
    56, 34, 51, 104, 114, 102, 29, 93, 77,
    107, 54, 32, 26, 51, 1, 81, 43, 31,
    39, 28, 85, 171, 58, 165, 90, 98, 64,
    34, 22, 116, 206, 23, 34, 43, 166, 73,
    68, 25, 106, 22, 64, 171, 36, 225, 114,
    34, 19, 21, 102, 132, 188, 16, 76, 124,
    62, 18, 78, 95, 85, 57, 50, 48, 51,
    193, 101, 35, 159, 215, 111, 89, 46, 111,
    60, 148, 31, 172, 219, 228, 21, 18, 111,
    112, 113, 77, 85, 179, 255, 38, 120, 114,
    40, 42, 1, 196, 245, 209, 10, 25, 109,
    100, 80, 8, 43, 154, 1, 51, 26, 71,
    88, 43, 29, 140, 166, 213, 37, 43, 154,
    61, 63, 30, 155, 67, 45, 68, 1, 209,
    142, 78, 78, 16, 255, 128, 34, 197, 171,
    41, 40, 5, 102, 211, 183, 4, 1, 221,
    51, 50, 17, 168, 209, 192, 23, 25, 82,
    125, 98, 42, 88, 104, 85, 117, 175, 82,
    95, 84, 53, 89, 128, 100, 113, 101, 45,
    75, 79, 123, 47, 51, 128, 81, 171, 1,
    57, 17, 5, 71, 102, 57, 53, 41, 49,
    115, 21, 2, 10, 102, 255, 166, 23, 6,
    38, 33, 13, 121, 57, 73, 26, 1, 85,
    41, 10, 67, 138, 77, 110, 90, 47, 114,
    101, 29, 16, 10, 85, 128, 101, 196, 26,
    57, 18, 10, 102, 102, 213, 34, 20, 43,
    117, 20, 15, 36, 163, 128, 68, 1, 26,
    138, 31, 36, 171, 27, 166, 38, 44, 229,
    67, 87, 58, 169, 82, 115, 26, 59, 179,
    63, 59, 90, 180, 59, 166, 93, 73, 154,
    40, 40, 21, 116, 143, 209, 34, 39, 175,
    57, 46, 22, 24, 128, 1, 54, 17, 37,
    47, 15, 16, 183, 34, 223, 49, 45, 183,
    46, 17, 33, 183, 6, 98, 15, 32, 183,
    65, 32, 73, 115, 28, 128, 23, 128, 205,
    40, 3, 9, 115, 51, 192, 18, 6, 223,
    87, 37, 9, 115, 59, 77, 64, 21, 47,
    104, 55, 44, 218, 9, 54, 53, 130, 226,
    64, 90, 70, 205, 40, 41, 23, 26, 57,
    54, 57, 112, 184, 5, 41, 38, 166, 213,
    30, 34, 26, 133, 152, 116, 10, 32, 134,
    75, 32, 12, 51, 192, 255, 160, 43, 51,
    39, 19, 53, 221, 26, 114, 32, 73, 255,
    31, 9, 65, 234, 2, 15, 1, 118, 73,
    88, 31, 35, 67, 102, 85, 55, 186, 85,
    56, 21, 23, 111, 59, 205, 45, 37, 192,
    55, 38, 70, 124, 73, 102, 1, 34, 98,
    102, 61, 71, 37, 34, 53, 31, 243, 192,
    69, 60, 71, 38, 73, 119, 28, 222, 37,
    68, 45, 128, 34, 1, 47, 11, 245, 171,
    62, 17, 19, 70, 146, 85, 55, 62, 70,
    75, 15, 9, 9, 64, 255, 184, 119, 16,
    37, 43, 37, 154, 100, 163, 85, 160, 1,
    63, 9, 92, 136, 28, 64, 32, 201, 85,
    86, 6, 28, 5, 64, 255, 25, 248, 1,
    56, 8, 17, 132, 137, 255, 55, 116, 128,
    58, 15, 20, 82, 135, 57, 26, 121, 40,
    164, 50, 31, 137, 154, 133, 25, 35, 218,
    51, 103, 44, 131, 131, 123, 31, 6, 158,
    86, 40, 64, 135, 148, 224, 45, 183, 128,
    22, 26, 17, 131, 240, 154, 14, 1, 209,
    83, 12, 13, 54, 192, 255, 68, 47, 28,
    45, 16, 21, 91, 64, 222, 7, 1, 197,
    56, 21, 39, 155, 60, 138, 23, 102, 213,
    85, 26, 85, 85, 128, 128, 32, 146, 171,
    18, 11, 7, 63, 144, 171, 4, 4, 246,
    35, 27, 10, 146, 174, 171, 12, 26, 128,
    190, 80, 35, 99, 180, 80, 126, 54, 45,
    85, 126, 47, 87, 176, 51, 41, 20, 32,
    101, 75, 128, 139, 118, 146, 116, 128, 85,
    56, 41, 15, 176, 236, 85, 37, 9, 62,
    146, 36, 19, 30, 171, 255, 97, 27, 20,
    71, 30, 17, 119, 118, 255, 17, 18, 138,
    101, 38, 60, 138, 55, 70, 43, 26, 142,
    138, 45, 61, 62, 219, 1, 81, 188, 64,
    32, 41, 20, 117, 151, 142, 20, 21, 163,
    112, 19, 12, 61, 195, 128, 48, 4, 24,
];

/// Probability that a coefficient probability is replaced in the frame header, in the
/// order [kind][band][context][node].
#[rustfmt::skip]
const COEFF_UPDATE_PROBS: [u8; 1056] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255,
    249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255,
    234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255,
    250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255,
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255,
    234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255,
    255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255,
    255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255,
    250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255,
    234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255,
    251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255,
    255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255,
    255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255,
    248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255,
    255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255,
    246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255,
    252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255,
    255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255,
    248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255,
    253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255,
    255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255,
    252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255,
    250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

/// Coefficient probabilities at the start of a key frame, same order.
#[rustfmt::skip]
const DEFAULT_COEFF_PROBS: [u8; 1056] = [
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128,
    189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128,
    106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128,
    1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128,
    181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128,
    78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128,
    1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128,
    184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128,
    77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128,
    1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128,
    170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128,
    37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128,
    1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128,
    207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128,
    102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128,
    1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128,
    177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128,
    80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128,
    1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62,
    131, 45, 198, 221, 172, 176, 220, 157, 252, 221, 1,
    68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128,
    1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128,
    184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128,
    81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128,
    1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128,
    99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128,
    23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128,
    1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128,
    109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128,
    44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128,
    1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128,
    94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128,
    22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128,
    1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128,
    124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128,
    35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128,
    1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128,
    121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128,
    45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128,
    1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128,
    203, 1, 248, 255, 255, 128, 128, 128, 128, 128, 128,
    137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128,
    253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128,
    175, 13, 224, 243, 193, 185, 249, 198, 255, 255, 128,
    73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128,
    1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128,
    239, 90, 244, 250, 211, 209, 255, 255, 128, 128, 128,
    155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128,
    1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128,
    201, 51, 219, 255, 196, 186, 128, 128, 128, 128, 128,
    69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128,
    1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128,
    223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128,
    141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128,
    1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128,
    190, 36, 230, 255, 236, 255, 128, 128, 128, 128, 128,
    149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128,
    213, 62, 250, 255, 255, 128, 128, 128, 128, 128, 128,
    55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255,
    126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128,
    61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128,
    1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128,
    166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128,
    39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128,
    1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128,
    124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128,
    24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128,
    1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128,
    149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128,
    28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128,
    1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128,
    123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128,
    20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128,
    1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128,
    168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128,
    47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128,
    1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128,
    141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128,
    42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128,
    1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    244, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
    238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
];

/// DC dequantisation factor of each quantiser index.
#[rustfmt::skip]
const DC_QLOOKUP: [u16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17,
    18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    91, 93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
    122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

/// AC dequantisation factor of each quantiser index.
#[rustfmt::skip]
const AC_QLOOKUP: [u16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
    78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
    155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
    213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! fixture {
        ($name:literal) => {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/vision/webp/",
                $name
            ))
        };
    }

    /// Simple lossy fixtures with the FNV-1a 64 hash of the RGB bytes the reference image
    /// library returns (from `tests/data/vision/webp/make_fixtures.py`).
    const FIXTURES: &[(&str, &[u8], u64)] = &[
        (
            "lossy_1x1",
            fixture!("lossy_1x1.webp"),
            0xd2db_0e18_6909_4082,
        ),
        (
            "lossy_1x16",
            fixture!("lossy_1x16.webp"),
            0x1f6b_0d9c_4c1a_4df7,
        ),
        (
            "lossy_2x3",
            fixture!("lossy_2x3.webp"),
            0x70c0_5110_b37b_0bca,
        ),
        (
            "lossy_9x1",
            fixture!("lossy_9x1.webp"),
            0x86f2_5ea4_8483_8cd3,
        ),
        (
            "lossy_filter_deltas_48x40",
            fixture!("lossy_filter_deltas_48x40.webp"),
            0xf6e1_826a_dbab_8bf9,
        ),
        (
            "lossy_q100_edges_16x16",
            fixture!("lossy_q100_edges_16x16.webp"),
            0xb218_4895_a558_964a,
        ),
        (
            "lossy_q20_17x17",
            fixture!("lossy_q20_17x17.webp"),
            0x4818_64c7_ef84_2293,
        ),
        (
            "lossy_q75_33x47",
            fixture!("lossy_q75_33x47.webp"),
            0xa54e_d9f2_2074_77ce,
        ),
        (
            "lossy_sharp7_50x50",
            fixture!("lossy_sharp7_50x50.webp"),
            0x8bbc_ba3a_b3e6_833d,
        ),
        (
            "lossy_simple_filter_64x48",
            fixture!("lossy_simple_filter_64x48.webp"),
            0x1595_389e_d27b_c2b9,
        ),
        (
            "lossy_synth_a",
            fixture!("lossy_synth_a.webp"),
            0x1cbb_6685_61dc_e8cf,
        ),
        (
            "lossy_synth_b",
            fixture!("lossy_synth_b.webp"),
            0xcdc5_b959_2573_ecfa,
        ),
    ];

    const LIMITS: DecodeLimits = DecodeLimits {
        max_pixels: 1 << 24,
    };

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// The bitstream of a simple file: everything after the RIFF header and the chunk header.
    fn bitstream(file: &[u8]) -> &[u8] {
        assert_eq!(&file[12..16], b"VP8 ");
        &file[20..]
    }

    fn rgb(frame: &Frame) -> Vec<u8> {
        let mut out = vec![0; frame.width * frame.height * 3];
        frame.write_rgb(&mut out, frame.width * 3, 3);
        out
    }

    #[test]
    fn key_frames_decode_to_the_reference_pixels() {
        for &(name, file, hash) in FIXTURES {
            let frame = decode(bitstream(file), &LIMITS).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(fnv1a(&rgb(&frame)), hash, "{name}");
            // four bytes per pixel: the same colours, the fourth byte left alone
            let mut wide = vec![7u8; frame.width * frame.height * 4];
            frame.write_rgb(&mut wide, frame.width * 4, 4);
            let packed: Vec<u8> = wide
                .chunks(4)
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            assert_eq!(fnv1a(&packed), hash, "{name}");
            assert!(wide.chunks(4).all(|px| px[3] == 7), "{name}");
        }
    }

    #[test]
    fn frame_header_is_checked() {
        let good = bitstream(FIXTURES[7].1);
        assert_eq!(read_header(good, good.len()), Some((33, 47)));
        let patched = |at: usize, value: u8| {
            let mut d = good.to_vec();
            d[at] = value;
            d
        };
        assert_eq!(
            read_header(&good[..9], good.len()),
            None,
            "shorter than the header"
        );
        assert_eq!(
            read_header(&patched(0, good[0] | 1), good.len()),
            None,
            "inter frame"
        );
        assert_eq!(
            read_header(&patched(0, good[0] & !0x10), good.len()),
            None,
            "hidden frame"
        );
        assert_eq!(
            read_header(&patched(0, good[0] | 0x08), good.len()),
            None,
            "profile over 3"
        );
        assert_eq!(read_header(&patched(4, 0), good.len()), None, "start code");
        let first = (good[0] as usize | (good[1] as usize) << 8 | (good[2] as usize) << 16) >> 5;
        assert_eq!(
            read_header(good, first),
            None,
            "first partition fills the chunk"
        );
        let mut flat = patched(6, 0);
        flat[7] &= 0xc0;
        assert_eq!(read_header(&flat, good.len()), None, "zero width");
        // the two scaling bits above each dimension are not part of the size
        let scaled = patched(7, good[7] | 0xc0);
        assert_eq!(read_header(&scaled, good.len()), Some((33, 47)));
        assert_eq!(
            fnv1a(&rgb(&decode(&scaled, &LIMITS).unwrap())),
            FIXTURES[7].2
        );
    }

    #[test]
    fn frame_over_the_limit_is_refused() {
        let good = bitstream(FIXTURES[7].1);
        let tight = DecodeLimits {
            max_pixels: 33 * 47 - 1,
        };
        assert!(tight.check_canvas(33, 47).is_err());
        assert_eq!(decode(good, &tight).err(), tight.check_canvas(33, 47).err());
        // the largest size a header can give, with no data behind it
        let mut huge = good[..10].to_vec();
        huge[6..10].copy_from_slice(&[0xff, 0x3f, 0xff, 0x3f]);
        assert!(decode(&huge, &LIMITS).is_err());
    }

    #[test]
    fn truncated_frames_are_refused_or_decoded_without_panic() {
        for &(name, file, _) in FIXTURES {
            let data = bitstream(file);
            let first =
                (data[0] as usize | (data[1] as usize) << 8 | (data[2] as usize) << 16) >> 5;
            for cut in 0..data.len() {
                let result = decode(&data[..cut], &LIMITS);
                // without the whole first partition and a byte of tokens nothing decodes
                if cut <= 10 + first {
                    assert!(result.is_err(), "{name} cut at {cut}");
                }
            }
        }
    }

    #[test]
    fn corrupted_frames_never_panic() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize
        };
        let small = DecodeLimits {
            max_pixels: 1 << 16,
        };
        for &(_, file, _) in FIXTURES {
            for round in 0..600 {
                let mut data = bitstream(file).to_vec();
                for _ in 0..1 + round % 4 {
                    let at = next() % data.len();
                    data[at] = if round % 2 == 0 {
                        data[at] ^ (1 << (next() % 8))
                    } else {
                        next() as u8
                    };
                }
                if let Ok(frame) = decode(&data, &small) {
                    rgb(&frame);
                }
            }
        }
    }

    #[test]
    fn empty_partition_is_at_its_end_before_any_read() {
        assert!(BoolReader::new(&[]).eof);
        // a decision needs 8 bits in the window: one byte serves the first decision, and the
        // second one asks for a byte that is not there
        let mut one = BoolReader::new(&[0x80]);
        assert!(!one.eof);
        assert!(one.get(128));
        assert!(!one.eof);
        one.get(128);
        assert!(one.eof);
    }

    #[test]
    fn signs_read_like_even_decisions() {
        // for a valid stream the sign shortcut is a decision of probability one half (a sign is
        // never the first decision of a partition)
        let data: Vec<u8> = (0..64u32).map(|i| (i * 37 + 11) as u8 & 0x7f).collect();
        let (mut a, mut b) = (BoolReader::new(&data), BoolReader::new(&data));
        assert_eq!(a.get(200), b.get(200));
        for i in 0..300 {
            if i % 3 == 0 {
                assert_eq!(a.get(128), b.get_sign(), "decision {i}");
            } else {
                let prob = (i * 29 % 255 + 1) as u8;
                assert_eq!(a.get(prob), b.get(prob), "decision {i}");
            }
            assert_eq!((a.range, a.bits, a.value), (b.range, b.bits, b.value));
        }
    }

    /// The inverse DCT of the RFC in plain integer arithmetic.
    fn inverse_dct_exact(c: &[i16; 16]) -> [i32; 16] {
        let mul1 = |x: i32| ((x * 20091) >> 16) + x;
        let mul2 = |x: i32| (x * 35468) >> 16;
        let mut tmp = [0i32; 16];
        for i in 0..4 {
            let (c0, c1, c2, c3) = (
                c[i] as i32,
                c[4 + i] as i32,
                c[8 + i] as i32,
                c[12 + i] as i32,
            );
            let (a, b) = (c0 + c2, c0 - c2);
            let (cc, d) = (mul2(c1) - mul1(c3), mul1(c1) + mul2(c3));
            tmp[4 * i..4 * i + 4].copy_from_slice(&[a + d, b + cc, b - cc, a - d]);
        }
        let mut out = [0i32; 16];
        for i in 0..4 {
            let (a, b) = (tmp[i] + 4 + tmp[8 + i], tmp[i] + 4 - tmp[8 + i]);
            let (cc, d) = (
                mul2(tmp[4 + i]) - mul1(tmp[12 + i]),
                mul1(tmp[4 + i]) + mul2(tmp[12 + i]),
            );
            out[4 * i..4 * i + 4].copy_from_slice(&[
                (a + d) >> 3,
                (b + cc) >> 3,
                (b - cc) >> 3,
                (a - d) >> 3,
            ]);
        }
        out
    }

    #[test]
    fn both_inverse_transforms_follow_the_rfc_arithmetic() {
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..2000 {
            // coefficients an encoder can produce: a block whose magnitudes sum to under 2048
            let mut c = [0i16; 16];
            let mut budget = 2047i64;
            for v in c.iter_mut() {
                if next() % 3 == 0 {
                    let m = (next() % 600) as i64;
                    let m = m.min(budget);
                    budget -= m;
                    *v = if next() % 2 == 0 {
                        m as i16
                    } else {
                        -(m as i16)
                    };
                }
            }
            let base = (round * 7 % 256) as u8;
            let want: Vec<u8> = inverse_dct_exact(&c)
                .iter()
                .map(|&v| (base as i32 + v).clamp(0, 255) as u8)
                .collect();
            let mut wide = [base; 16];
            add_inverse_dct_full(&c, &mut wide, 0, 4);
            assert_eq!(&wide[..], &want[..], "16-bit transform, round {round}");
            let mut exact = [base; 16];
            add_inverse_dct(&c, &mut exact, 0, 4);
            assert_eq!(&exact[..], &want[..], "exact transform, round {round}");
        }
        // a DC coefficient alone moves every pixel by (dc + 4) >> 3
        let mut c = [0i16; 16];
        c[0] = 84;
        let mut px = [100u8; 16];
        add_residual(RESIDUAL_DC, &c, &mut px, 0, 4);
        assert_eq!(px, [111u8; 16]);
    }

    #[test]
    fn colour_conversion_matches_known_values() {
        let convert = |y: u8, u: i32, v: i32| {
            let mut out = [0u8; 3];
            yuv_to_rgb(y, u, v, &mut out);
            out
        };
        assert_eq!(convert(16, 128, 128), [0, 0, 0]);
        assert_eq!(convert(235, 128, 128), [255, 255, 255]);
        assert_eq!(convert(126, 128, 128), [128, 128, 128]);
        // saturated colours, and the corners of the input range, which clamp
        assert_eq!(convert(81, 90, 240), [254, 0, 0]);
        assert_eq!(convert(145, 54, 34), [0, 255, 1]);
        assert_eq!(convert(41, 240, 110), [0, 0, 255]);
        assert_eq!(convert(0, 0, 0), [0, 136, 0]);
        assert_eq!(convert(255, 255, 255), [255, 125, 255]);
    }

    /// Compares every simple lossy `<name>.webp` of the directory in `GAUDIUM_IMAGE_REF_DIR`
    /// with the raw RGB dump `<name>.rgb` beside it.
    #[test]
    #[ignore]
    fn reference_directory() {
        let Ok(dir) = std::env::var("GAUDIUM_IMAGE_REF_DIR") else {
            return;
        };
        let limits = DecodeLimits {
            max_pixels: 1 << 28,
        };
        let (mut compared, mut failures) = (0, Vec::new());
        for entry in std::fs::read_dir(&dir).expect("GAUDIUM_IMAGE_REF_DIR") {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "webp") {
                continue;
            }
            let (file, Ok(want)) = (
                std::fs::read(&path).unwrap(),
                std::fs::read(path.with_extension("rgb")),
            ) else {
                continue;
            };
            if file.len() < 30 || &file[12..16] != b"VP8 " {
                continue;
            }
            // the chunk with its padding byte, as the container hands it over
            let size = u32::from_le_bytes([file[16], file[17], file[18], file[19]]) as usize;
            let Some(data) = file.get(20..20 + size + (size & 1)) else {
                continue;
            };
            compared += 1;
            match decode(data, &limits) {
                Ok(frame) if rgb(&frame) == want => {}
                Ok(_) => failures.push(format!("{}: pixels differ", path.display())),
                Err(e) => failures.push(format!("{}: {e}", path.display())),
            }
        }
        println!(
            "{compared} lossy files compared, {} failures",
            failures.len()
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
