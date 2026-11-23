//! zlib streams (RFC 1950) and their DEFLATE payload (RFC 1951), for the PNG decoder.
//!
//! Decodes stored, fixed-Huffman and dynamic-Huffman blocks and checks the Adler-32 trailer.
//!
//! Refuses: a stream with a preset dictionary, a compression method other than DEFLATE, a
//! header whose check bits are wrong, an over-subscribed or incomplete Huffman code (the one
//! incomplete code RFC 1951 allows, a single code of length one, is accepted), a block without
//! an end-of-block code, a distance that reaches before the start of the output, a stream that
//! ends early, and a trailer that does not match the data.
//!
//! The output is exact by construction (DEFLATE is lossless), so nothing here needs to follow
//! another implementation's arithmetic.
//!
//! Every length comes from the stream and is checked against the caller's limit before the
//! output grows, so a small file can not make the decoder allocate more than `max_out` bytes.

/// Inflates a zlib stream. `max_out`: refuse (Err) as soon as the output would exceed this
/// many bytes.
///
/// The PNG decoder uses the prefix form below, so without another caller this strict form is
/// only reached from tests.
#[allow(dead_code)]
pub fn zlib_decompress(data: &[u8], max_out: usize) -> Result<Vec<u8>, String> {
    inflate_zlib(data, max_out, 0, AtLimit::Refuse)
}

/// Inflates a zlib stream whose output is expected to be exactly `want` bytes long.
///
/// A stream that holds more is cut at `want` bytes (its trailer is then never reached, so no
/// checksum is compared); a stream that holds less is returned whole, after its checksum is
/// verified. The caller compares the returned length with `want`.
pub(super) fn zlib_decompress_prefix(data: &[u8], want: usize) -> Result<Vec<u8>, String> {
    inflate_zlib(data, want, want, AtLimit::Cut)
}

/// What happens when the stream holds more than the output limit.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AtLimit {
    Refuse,
    Cut,
}

/// Longest Huffman code DEFLATE allows.
const MAX_CODE_LEN: usize = 15;
/// Index bits of the literal/length lookup table.
const LIT_TABLE_BITS: u32 = 10;
/// Index bits of the distance lookup table.
const DIST_TABLE_BITS: u32 = 8;

/// Base match length of length symbols 257..=285.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// Extra bits of length symbols 257..=285.
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Base distance of distance symbols 0..=29.
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// Extra bits of distance symbols 0..=29.
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// Order in which a dynamic block stores the lengths of its code-length code.
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Least-significant-bit-first reader over the compressed bytes.
///
/// Past the end of the input it supplies zero bits and counts them in `pad`, so the hot loop
/// needs no end test per read; `overrun` tells afterwards whether any padding was consumed.
struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to load into `buf`.
    pos: usize,
    buf: u64,
    /// Valid bits in `buf` (padding included).
    n: u32,
    /// Zero bits supplied past the end of the input.
    pad: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            buf: 0,
            n: 0,
            pad: 0,
        }
    }

    /// Tops the buffer up to at least 56 bits.
    #[inline(always)]
    fn refill(&mut self) {
        if let Some(chunk) = self.data.get(self.pos..self.pos + 8) {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            // Bits of a byte that only fits partly are loaded again by the next refill; the
            // position advances by whole bytes only.
            self.buf |= u64::from_le_bytes(word) << self.n;
            let bytes = (63 - self.n) >> 3;
            self.pos += bytes as usize;
            self.n += bytes * 8;
        } else {
            while self.n <= 56 {
                match self.data.get(self.pos) {
                    Some(&b) => {
                        self.buf |= u64::from(b) << self.n;
                        self.pos += 1;
                    }
                    None => self.pad += 8,
                }
                self.n += 8;
            }
        }
    }

    #[inline(always)]
    fn peek(&self, count: u32) -> u32 {
        (self.buf & ((1u64 << count) - 1)) as u32
    }

    /// Drops `count` bits; the caller refilled so that `count <= n`.
    #[inline(always)]
    fn consume(&mut self, count: u32) {
        self.buf >>= count;
        self.n = self.n.saturating_sub(count);
    }

    #[inline(always)]
    fn take(&mut self, count: u32) -> u32 {
        let v = self.peek(count);
        self.consume(count);
        v
    }

    /// True when bits past the end of the input were consumed.
    #[inline(always)]
    fn overrun(&self) -> bool {
        self.pad > self.n
    }

    /// Moves to the next byte boundary and returns the byte position there, leaving the bit
    /// buffer empty. Fails when the stream ended before that position.
    fn align_to_byte(&mut self) -> Result<usize, String> {
        if self.overrun() {
            return Err(truncated());
        }
        let real_bits = self.n - self.pad;
        let at = self.pos - (real_bits / 8) as usize;
        self.pos = at;
        self.buf = 0;
        self.n = 0;
        self.pad = 0;
        Ok(at)
    }
}

fn truncated() -> String {
    "compressed data ends early".to_string()
}

/// A canonical Huffman code: a direct lookup table for the short codes and the per-length
/// counts for the long ones.
struct Huffman {
    /// Indexed by the next `table_bits` input bits: `symbol << 4 | code length`, or 0 when the
    /// code there is longer than `table_bits` or not assigned.
    table: Vec<u16>,
    table_bits: u32,
    /// Number of codes of each length.
    count: [u16; MAX_CODE_LEN + 1],
    /// Symbols ordered by code length, then by value.
    sorted: Vec<u16>,
    /// Length of the longest code; 0 for a code without symbols.
    longest: usize,
    /// True when some bit patterns have no symbol.
    incomplete: bool,
}

impl Huffman {
    fn new(lengths: &[u8], table_bits: u32) -> Result<Huffman, String> {
        let mut count = [0u16; MAX_CODE_LEN + 1];
        for &len in lengths {
            let slot = count
                .get_mut(len as usize)
                .ok_or("Huffman code length out of range")?;
            *slot += 1;
        }
        count[0] = 0;
        let mut longest = 0;
        let mut left: i32 = 1;
        for (len, &c) in count.iter().enumerate().skip(1) {
            left = (left << 1) - i32::from(c);
            if left < 0 {
                return Err("over-subscribed Huffman code in compressed data".to_string());
            }
            if c != 0 {
                longest = len;
            }
        }

        // First index in `sorted` of each length.
        let mut offset = [0u16; MAX_CODE_LEN + 2];
        for len in 1..=MAX_CODE_LEN {
            offset[len + 1] = offset[len] + count[len];
        }
        let mut sorted = vec![0u16; lengths.len()];
        for (symbol, &len) in lengths.iter().enumerate() {
            if len != 0 {
                let at = &mut offset[len as usize];
                if let Some(slot) = sorted.get_mut(*at as usize) {
                    *slot = symbol as u16;
                }
                *at += 1;
            }
        }

        // Canonical codes are assigned in `sorted` order; the stream sends each code most
        // significant bit first, so the table is indexed by the bit-reversed code.
        let mut table = vec![0u16; 1usize << table_bits];
        let mut code: u32 = 0;
        let mut next = 0usize;
        for (len, &codes) in count.iter().enumerate().skip(1) {
            for _ in 0..codes {
                if len as u32 <= table_bits {
                    let symbol = sorted.get(next).copied().unwrap_or(0);
                    let entry = (symbol << 4) | len as u16;
                    let reversed = (code.reverse_bits() >> (32 - len as u32)) as usize;
                    let mut at = reversed;
                    while at < table.len() {
                        table[at] = entry;
                        at += 1usize << len;
                    }
                }
                code += 1;
                next += 1;
            }
            code <<= 1;
        }

        Ok(Huffman {
            table,
            table_bits,
            count,
            sorted,
            longest,
            incomplete: left > 0,
        })
    }

    /// True for the incomplete codes DEFLATE allows: no symbol at all, or one code of one bit.
    fn acceptable_when_incomplete(&self) -> bool {
        self.longest <= 1
    }

    /// Decodes one symbol. The caller refilled, so at least 15 bits are buffered.
    #[inline(always)]
    fn decode(&self, bits: &mut BitReader) -> Result<u32, String> {
        let entry = self.table[bits.peek(self.table_bits) as usize];
        if entry != 0 {
            bits.consume(u32::from(entry & 15));
            return Ok(u32::from(entry >> 4));
        }
        self.decode_long(bits)
    }

    /// Walks the code one bit at a time; used for codes longer than the lookup table.
    #[cold]
    fn decode_long(&self, bits: &mut BitReader) -> Result<u32, String> {
        let window = bits.peek(MAX_CODE_LEN as u32);
        let mut code: u32 = 0;
        let mut first: u32 = 0;
        let mut index: u32 = 0;
        for len in 1..=MAX_CODE_LEN {
            code |= (window >> (len - 1)) & 1;
            let count = u32::from(self.count[len]);
            if code < first + count {
                bits.consume(len as u32);
                let at = (index + (code - first)) as usize;
                return self
                    .sorted
                    .get(at)
                    .map(|&s| u32::from(s))
                    .ok_or_else(invalid_code);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(invalid_code())
    }
}

fn invalid_code() -> String {
    "invalid Huffman code in compressed data".to_string()
}

/// The decompressed bytes, grown on demand up to `max`.
struct Output {
    buf: Vec<u8>,
    /// Bytes written; `buf` beyond it is spare room.
    pos: usize,
    max: usize,
}

impl Output {
    /// Makes room for `count` more bytes. False when the limit does not allow them all.
    #[inline(always)]
    fn room(&mut self, count: usize) -> bool {
        let need = self.pos.saturating_add(count);
        if need <= self.buf.len() {
            return true;
        }
        self.grow(need)
    }

    #[cold]
    fn grow(&mut self, need: usize) -> bool {
        if need > self.max {
            return false;
        }
        let target = self
            .buf
            .len()
            .saturating_mul(2)
            .max(need)
            .max(4096)
            .min(self.max);
        self.buf.resize(target, 0);
        true
    }

    /// Bytes the limit still allows.
    fn remaining(&self) -> usize {
        self.max - self.pos
    }

    /// Appends `len` bytes that repeat the output `dist` bytes back. The caller made room and
    /// checked `dist <= pos`.
    #[inline(always)]
    fn copy_match(&mut self, dist: usize, len: usize) {
        let pos = self.pos;
        if dist >= len {
            self.buf.copy_within(pos - dist..pos - dist + len, pos);
        } else if dist == 1 {
            let byte = self.buf[pos - 1];
            self.buf[pos..pos + len].fill(byte);
        } else {
            // The source overlaps the destination: bytes written by this copy are read again.
            for i in pos..pos + len {
                self.buf[i] = self.buf[i - dist];
            }
        }
        self.pos = pos + len;
    }
}

/// How a block ended.
enum BlockEnd {
    /// The end-of-block code was read.
    Done,
    /// The output limit was reached (only in `AtLimit::Cut` mode).
    Full,
}

fn inflate_zlib(
    data: &[u8],
    max_out: usize,
    reserve: usize,
    at_limit: AtLimit,
) -> Result<Vec<u8>, String> {
    let (&cmf, &flg) = match (data.first(), data.get(1)) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(truncated()),
    };
    if (u32::from(cmf) * 256 + u32::from(flg)) % 31 != 0 {
        return Err("not a zlib stream (header check failed)".to_string());
    }
    if cmf & 15 != 8 || cmf >> 4 > 7 {
        return Err("unsupported zlib compression method".to_string());
    }
    if flg & 0x20 != 0 {
        return Err("zlib stream needs a preset dictionary".to_string());
    }

    let mut bits = BitReader::new(&data[2..]);
    // The room set aside up front is no more than the input can expand to: a DEFLATE symbol
    // of two bits yields at most 258 bytes, a ratio of 1032.
    let reserve = reserve.min(max_out).min(data.len().saturating_mul(1032));
    let mut out = Output {
        buf: vec![0u8; reserve],
        pos: 0,
        max: max_out,
    };
    let mut fixed: Option<(Huffman, Huffman)> = None;

    loop {
        bits.refill();
        let last = bits.take(1) == 1;
        let kind = bits.take(2);
        let end = match kind {
            0 => stored_block(&mut bits, &mut out, at_limit)?,
            1 => {
                if fixed.is_none() {
                    fixed = Some(fixed_codes()?);
                }
                match &fixed {
                    Some((lit, dist)) => huffman_block(&mut bits, &mut out, lit, dist, at_limit)?,
                    None => return Err(invalid_code()),
                }
            }
            2 => {
                let (lit, dist) = dynamic_codes(&mut bits)?;
                huffman_block(&mut bits, &mut out, &lit, &dist, at_limit)?
            }
            _ => return Err("invalid block type in compressed data".to_string()),
        };
        // A symbol read from the padding past the end of the input is not data, whether it
        // ended the block or filled the output.
        if bits.overrun() {
            return Err(truncated());
        }
        if let BlockEnd::Full = end {
            out.buf.truncate(out.pos);
            return Ok(out.buf);
        }
        if last {
            break;
        }
    }

    let at = bits.align_to_byte()?;
    let trailer = bits.data.get(at..at + 4).ok_or_else(truncated)?;
    let stored = u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    out.buf.truncate(out.pos);
    if adler32(&out.buf) != stored {
        return Err("compressed data is damaged (checksum mismatch)".to_string());
    }
    Ok(out.buf)
}

fn over_limit(max: usize) -> String {
    format!("compressed data expands to more than {max} bytes")
}

/// A stored block: a byte-aligned length, its complement, and that many literal bytes.
fn stored_block(
    bits: &mut BitReader,
    out: &mut Output,
    at_limit: AtLimit,
) -> Result<BlockEnd, String> {
    let at = bits.align_to_byte()?;
    let head = bits.data.get(at..at + 4).ok_or_else(truncated)?;
    let len = u16::from_le_bytes([head[0], head[1]]);
    let complement = u16::from_le_bytes([head[2], head[3]]);
    if len != !complement {
        return Err("damaged stored block in compressed data".to_string());
    }
    let len = usize::from(len);
    let body = bits.data.get(at + 4..at + 4 + len).ok_or_else(truncated)?;
    bits.pos = at + 4 + len;
    if out.room(len) {
        out.buf[out.pos..out.pos + len].copy_from_slice(body);
        out.pos += len;
        return Ok(BlockEnd::Done);
    }
    if at_limit == AtLimit::Refuse {
        return Err(over_limit(out.max));
    }
    let fits = out.remaining();
    if out.room(fits) {
        out.buf[out.pos..out.pos + fits].copy_from_slice(&body[..fits]);
        out.pos += fits;
    }
    Ok(BlockEnd::Full)
}

/// The fixed codes of block type 1.
fn fixed_codes() -> Result<(Huffman, Huffman), String> {
    let mut lengths = [0u8; 288];
    for (symbol, len) in lengths.iter_mut().enumerate() {
        *len = match symbol {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    Ok((
        Huffman::new(&lengths, LIT_TABLE_BITS)?,
        Huffman::new(&[5u8; 30], DIST_TABLE_BITS)?,
    ))
}

/// Reads the code descriptions at the start of a dynamic block.
fn dynamic_codes(bits: &mut BitReader) -> Result<(Huffman, Huffman), String> {
    bits.refill();
    let lit_count = bits.take(5) as usize + 257;
    let dist_count = bits.take(5) as usize + 1;
    let length_code_count = bits.take(4) as usize + 4;
    if lit_count > 286 || dist_count > 30 {
        return Err("too many Huffman symbols in compressed data".to_string());
    }

    let mut length_code_lengths = [0u8; 19];
    for &slot in CODE_LENGTH_ORDER.iter().take(length_code_count) {
        bits.refill();
        length_code_lengths[slot] = bits.take(3) as u8;
    }
    let length_code = Huffman::new(&length_code_lengths, 7)?;
    if length_code.incomplete {
        return Err("incomplete Huffman code in compressed data".to_string());
    }

    let total = lit_count + dist_count;
    let mut lengths = [0u8; 286 + 30];
    let mut filled = 0usize;
    while filled < total {
        bits.refill();
        if bits.overrun() {
            return Err(truncated());
        }
        let symbol = length_code.decode(bits)?;
        let (value, repeat) = match symbol {
            0..=15 => (symbol as u8, 1),
            16 => {
                let previous = match filled.checked_sub(1) {
                    Some(i) => lengths[i],
                    None => return Err("damaged Huffman code lengths".to_string()),
                };
                (previous, 3 + bits.take(2) as usize)
            }
            17 => (0, 3 + bits.take(3) as usize),
            _ => (0, 11 + bits.take(7) as usize),
        };
        if filled + repeat > total {
            return Err("damaged Huffman code lengths".to_string());
        }
        lengths[filled..filled + repeat].fill(value);
        filled += repeat;
    }
    if bits.overrun() {
        return Err(truncated());
    }
    if lengths[256] == 0 {
        return Err("compressed block has no end code".to_string());
    }

    let lit = Huffman::new(&lengths[..lit_count], LIT_TABLE_BITS)?;
    let dist = Huffman::new(&lengths[lit_count..total], DIST_TABLE_BITS)?;
    if (lit.incomplete && !lit.acceptable_when_incomplete())
        || (dist.incomplete && !dist.acceptable_when_incomplete())
    {
        return Err("incomplete Huffman code in compressed data".to_string());
    }
    Ok((lit, dist))
}

/// Decodes the symbols of one Huffman-coded block.
fn huffman_block(
    bits: &mut BitReader,
    out: &mut Output,
    lit: &Huffman,
    dist: &Huffman,
    at_limit: AtLimit,
) -> Result<BlockEnd, String> {
    loop {
        // One refill covers a whole length/distance pair: 15 + 5 + 15 + 13 bits.
        bits.refill();
        if bits.pad != 0 && bits.overrun() {
            return Err(truncated());
        }
        let symbol = lit.decode(bits)?;
        if symbol < 256 {
            if out.pos == out.buf.len() && !out.room(1) {
                return limit_reached(out, at_limit);
            }
            out.buf[out.pos] = symbol as u8;
            out.pos += 1;
            continue;
        }
        if symbol == 256 {
            return Ok(BlockEnd::Done);
        }
        let slot = (symbol - 257) as usize;
        let (&base, &extra) = match (LENGTH_BASE.get(slot), LENGTH_EXTRA.get(slot)) {
            (Some(b), Some(e)) => (b, e),
            _ => return Err(invalid_code()),
        };
        let len = usize::from(base) + bits.take(u32::from(extra)) as usize;

        let slot = dist.decode(bits)? as usize;
        let (&base, &extra) = match (DIST_BASE.get(slot), DIST_EXTRA.get(slot)) {
            (Some(b), Some(e)) => (b, e),
            _ => return Err(invalid_code()),
        };
        let distance = usize::from(base) + bits.take(u32::from(extra)) as usize;
        if distance > out.pos {
            return Err("compressed data refers to bytes before its start".to_string());
        }

        if out.room(len) {
            out.copy_match(distance, len);
            continue;
        }
        if at_limit == AtLimit::Cut {
            let fits = out.remaining();
            if out.room(fits) {
                out.copy_match(distance, fits);
            }
        }
        return limit_reached(out, at_limit);
    }
}

fn limit_reached(out: &Output, at_limit: AtLimit) -> Result<BlockEnd, String> {
    match at_limit {
        AtLimit::Refuse => Err(over_limit(out.max)),
        AtLimit::Cut => Ok(BlockEnd::Full),
    }
}

/// The Adler-32 checksum of RFC 1950.
fn adler32(data: &[u8]) -> u32 {
    const MODULUS: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    // 5552 is the longest run for which the sums can not overflow 32 bits before a reduction.
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += u32::from(byte);
            b += a;
        }
        a %= MODULUS;
        b %= MODULUS;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 600 bytes that compress with matches, literals and long runs.
    fn sample() -> Vec<u8> {
        (0..600u32)
            .map(|i| match i % 200 {
                0..=79 => (i % 7) as u8,
                80..=139 => 0x41,
                _ => (i.wrapping_mul(2654435761) >> 24) as u8,
            })
            .collect()
    }

    /// `sample()` as a zlib stream made of one stored block.
    fn stored_stream() -> Vec<u8> {
        let raw = sample();
        let mut z = vec![0x78, 0x01, 0x01];
        z.extend_from_slice(&(raw.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(raw.len() as u16)).to_le_bytes());
        z.extend_from_slice(&raw);
        z.extend_from_slice(&adler32(&raw).to_be_bytes());
        z
    }

    /// "hello hello hello hello\n" in one fixed-Huffman block (a literal run and one match).
    const FIXED: [u8; 17] = [
        0x78, 0xda, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0xb9, 0x00, 0x70, 0xbe,
        0x08, 0xbb,
    ];

    #[test]
    fn stored_block_round_trip() {
        let z = stored_stream();
        assert_eq!(zlib_decompress(&z, 600).unwrap(), sample());
        assert_eq!(zlib_decompress(&z, 1 << 20).unwrap(), sample());
    }

    #[test]
    fn limit_is_enforced() {
        let z = stored_stream();
        assert!(zlib_decompress(&z, 599).is_err());
        assert!(zlib_decompress(&z, 0).is_err());
        // The prefix form cuts instead of failing.
        let cut = zlib_decompress_prefix(&z, 100).unwrap();
        assert_eq!(cut, sample()[..100]);
        // A stream shorter than wanted comes back whole.
        assert_eq!(zlib_decompress_prefix(&z, 5000).unwrap(), sample());
    }

    #[test]
    fn damage_is_refused() {
        let z = stored_stream();
        // Checksum.
        let mut bad = z.clone();
        let n = bad.len();
        bad[n - 1] ^= 1;
        assert!(zlib_decompress(&bad, 1 << 20).is_err());
        // Stored length and its complement disagree.
        let mut bad = z.clone();
        bad[3] ^= 1;
        assert!(zlib_decompress(&bad, 1 << 20).is_err());
        // Header check bits, preset dictionary, unknown method.
        assert!(zlib_decompress(&[0x78, 0x02, 0x03, 0x00], 16).is_err());
        assert!(zlib_decompress(&[0x78, 0xbb, 0x03, 0x00], 16).is_err());
        assert!(zlib_decompress(&[0x79, 0x9b, 0x03, 0x00], 16).is_err());
        // Reserved block type.
        assert!(zlib_decompress(&[0x78, 0x9c, 0x07, 0x00], 16).is_err());
        // Every truncation fails cleanly.
        for cut in 0..z.len() {
            assert!(zlib_decompress(&z[..cut], 1 << 20).is_err(), "cut {cut}");
        }
        for cut in 0..FIXED.len() {
            assert!(
                zlib_decompress(&FIXED[..cut], 1 << 20).is_err(),
                "cut {cut}"
            );
        }
    }

    #[test]
    fn adler32_known_values() {
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"a"), 0x0062_0062);
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
    }

    /// A dynamic-Huffman stream: 600 letters with uneven frequencies.
    const DYNAMIC: [u8; 201] = [
        0x78, 0xda, 0x55, 0x51, 0x09, 0x12, 0xc3, 0x40, 0x08, 0x7a, 0xab, 0xc2, 0xff, 0xdf, 0x50,
        0x39, 0xb6, 0xd3, 0x26, 0x93, 0xec, 0x21, 0x22, 0x22, 0x76, 0x86, 0x8b, 0x9d, 0xe5, 0x8e,
        0x96, 0x7b, 0x67, 0xfd, 0xef, 0x43, 0x23, 0x0c, 0xc9, 0x25, 0xfc, 0xdd, 0x81, 0x09, 0x2b,
        0x33, 0xb1, 0xc4, 0x0f, 0x4a, 0x87, 0x36, 0x27, 0x7a, 0x6b, 0x00, 0x37, 0x04, 0x48, 0x06,
        0x58, 0xca, 0x43, 0x1d, 0xdb, 0x96, 0xef, 0x42, 0x25, 0xb7, 0xaa, 0xd2, 0x20, 0xa7, 0xd0,
        0x45, 0x26, 0x22, 0xd7, 0x9f, 0x14, 0xe1, 0x47, 0xb8, 0xe8, 0x1d, 0xc9, 0x05, 0x05, 0x18,
        0xb2, 0x12, 0x15, 0xbb, 0xea, 0x94, 0x12, 0xaf, 0x6e, 0xcb, 0xad, 0x9a, 0x1c, 0xc2, 0xa2,
        0xd4, 0x0c, 0xcd, 0x75, 0x12, 0x85, 0xed, 0xa7, 0xda, 0x2c, 0x58, 0xdc, 0x3a, 0x88, 0x04,
        0x3b, 0xf3, 0xef, 0xa1, 0x8a, 0xbd, 0x4e, 0x5d, 0x0e, 0xf5, 0x60, 0xdc, 0x8e, 0xd5, 0x5c,
        0xe2, 0x6d, 0x81, 0xa9, 0x06, 0xd5, 0x73, 0xe2, 0xd3, 0x6e, 0xcb, 0xae, 0xe8, 0xa8, 0xf0,
        0x95, 0xb3, 0xed, 0x96, 0x25, 0x1f, 0x84, 0x49, 0x5a, 0x5b, 0xa4, 0x89, 0xed, 0x01, 0x3b,
        0x1e, 0x7b, 0xa2, 0x34, 0x9e, 0x48, 0x16, 0xc4, 0xed, 0x74, 0x6c, 0x79, 0xfd, 0x0a, 0xfa,
        0x35, 0xf0, 0x06, 0x65, 0x51, 0x2a, 0x16, 0xa7, 0x7b, 0xd1, 0x44, 0x05, 0xb8, 0xdf, 0x81,
        0x7c, 0x00, 0x0b, 0x30, 0xe5, 0x30,
    ];

    #[test]
    fn dynamic_block_round_trip() {
        let mut state = 1u32;
        let expected: Vec<u8> = (0..600)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                b"aaaaabbbcd"[(state >> 16) as usize % 10]
            })
            .collect();
        assert_eq!(DYNAMIC[2] & 6, 4, "the sample is a dynamic block");
        assert_eq!(zlib_decompress(&DYNAMIC, expected.len()).unwrap(), expected);
        assert!(zlib_decompress(&DYNAMIC, expected.len() - 1).is_err());
        assert_eq!(
            zlib_decompress_prefix(&DYNAMIC, 100).unwrap(),
            expected[..100]
        );
        assert_eq!(
            zlib_decompress(&FIXED, 64).unwrap(),
            b"hello hello hello hello\n"
        );
        for cut in 0..DYNAMIC.len() {
            assert!(
                zlib_decompress(&DYNAMIC[..cut], 1 << 20).is_err(),
                "cut {cut}"
            );
        }
        // Damage never panics and never yields more than the limit.
        for at in 0..DYNAMIC.len() {
            for bit in 0..8 {
                let mut bad = DYNAMIC;
                bad[at] ^= 1 << bit;
                if let Ok(out) = zlib_decompress(&bad, 4096) {
                    assert!(out.len() <= 4096);
                }
                let _ = zlib_decompress_prefix(&bad, 300);
            }
        }
    }

    /// A stream that ends 32 bytes into its data. The zero bits past its end read as a match
    /// long enough to fill a 35-byte output; that must not pass for data.
    const CUT_SHORT: [u8; 36] = [
        0x78, 0x9c, 0x63, 0x94, 0x64, 0xa8, 0x61, 0xc8, 0xfe, 0xcf, 0xa8, 0xcb, 0x70, 0x9d, 0x61,
        0xf5, 0x7f, 0x46, 0xbf, 0xff, 0x9b, 0x18, 0x0e, 0xbf, 0x65, 0x74, 0xfd, 0xbf, 0x9b, 0xe1,
        0x8d, 0x34, 0x63, 0x5a, 0x7e, 0xd4,
    ];

    #[test]
    fn padding_past_the_end_is_not_data() {
        assert!(zlib_decompress_prefix(&CUT_SHORT, 35).is_err());
        assert!(zlib_decompress(&CUT_SHORT, 1 << 20).is_err());
        assert_eq!(zlib_decompress_prefix(&CUT_SHORT, 30).unwrap().len(), 30);
    }

    #[test]
    fn a_small_input_reserves_a_small_output() {
        // The limit is far above what the input can expand to; nothing of that size is set
        // aside, and the stream still decodes.
        let z = stored_stream();
        assert_eq!(zlib_decompress_prefix(&z, usize::MAX).unwrap(), sample());
        assert!(zlib_decompress_prefix(&[0x78, 0x9c], usize::MAX).is_err());
    }
}
