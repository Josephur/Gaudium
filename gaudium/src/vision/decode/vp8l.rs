//! Lossless WebP bitstream (the `VP8L` chunk, RFC 9649), and the same coding when it carries
//! the alpha plane of a lossy image.
//!
//! Decodes: canonical prefix codes (simple, normal and repeat-coded code lengths), LZ77 copies
//! with the 2-D short-distance codes, the colour cache, meta prefix codes (entropy image) and
//! the four transforms (predictor, cross-colour, subtract-green, colour indexing with pixel
//! bundling). The result is ARGB, one `u32` per pixel.
//!
//! Refuses: a version other than 0, a prefix code that is not complete (a single used symbol is
//! the one exception, it takes no bits), a transform used twice, a colour cache size outside
//! 1..=11 bits, a copy that starts before the image or runs past its end, a stream that ends
//! early, and more prefix-code tables than `MAX_TABLE_BYTES`.
//!
//! Bit-exact rules shared with the reference decoder: predictor modes 14 and 15 act as mode 0;
//! a palette index past the table gives transparent black; every decoded pixel enters the
//! colour cache, one read from the cache included; the stream may be read up to its last bit
//! but not beyond (a stream shorter than 8 bytes counts as 8 bytes, zero filled).

use super::types::DecodeLimits;

/// Upper bound of the memory the prefix-code tables of one image may take. An encoder writes
/// a few thousand code groups at most; a hostile file can ask for 65536.
const MAX_TABLE_BYTES: usize = 96 << 20;

/// Width of the first-level lookup of a prefix code, in bits.
const ROOT_BITS: u32 = 8;
/// Length nibble of a lookup entry whose code is longer than `ROOT_BITS`.
const LONG_CODE: u16 = 15;
/// Longest code length a symbol may have.
const MAX_LENGTH: usize = 15;

const GREEN: usize = 0;
const RED: usize = 1;
const BLUE: usize = 2;
const ALPHA: usize = 3;
const DIST: usize = 4;

/// Order in which the code lengths of the code-length code are stored.
const LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// The 120 short distance codes as (dx, dy): the copy source is `dy` rows up and `dx` columns
/// to the left.
const NEAR: [(i8, i8); 120] = [
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 2),
    (2, 0),
    (1, 2),
    (-1, 2),
    (2, 1),
    (-2, 1),
    (2, 2),
    (-2, 2),
    (0, 3),
    (3, 0),
    (1, 3),
    (-1, 3),
    (3, 1),
    (-3, 1),
    (2, 3),
    (-2, 3),
    (3, 2),
    (-3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (-1, 4),
    (4, 1),
    (-4, 1),
    (3, 3),
    (-3, 3),
    (2, 4),
    (-2, 4),
    (4, 2),
    (-4, 2),
    (0, 5),
    (3, 4),
    (-3, 4),
    (4, 3),
    (-4, 3),
    (5, 0),
    (1, 5),
    (-1, 5),
    (5, 1),
    (-5, 1),
    (2, 5),
    (-2, 5),
    (5, 2),
    (-5, 2),
    (4, 4),
    (-4, 4),
    (3, 5),
    (-3, 5),
    (5, 3),
    (-5, 3),
    (0, 6),
    (6, 0),
    (1, 6),
    (-1, 6),
    (6, 1),
    (-6, 1),
    (2, 6),
    (-2, 6),
    (6, 2),
    (-6, 2),
    (4, 5),
    (-4, 5),
    (5, 4),
    (-5, 4),
    (3, 6),
    (-3, 6),
    (6, 3),
    (-6, 3),
    (0, 7),
    (7, 0),
    (1, 7),
    (-1, 7),
    (5, 5),
    (-5, 5),
    (7, 1),
    (-7, 1),
    (4, 6),
    (-4, 6),
    (6, 4),
    (-6, 4),
    (2, 7),
    (-2, 7),
    (7, 2),
    (-7, 2),
    (3, 7),
    (-3, 7),
    (7, 3),
    (-7, 3),
    (5, 6),
    (-5, 6),
    (6, 5),
    (-6, 5),
    (8, 0),
    (4, 7),
    (-4, 7),
    (7, 4),
    (-7, 4),
    (8, 1),
    (8, 2),
    (6, 6),
    (-6, 6),
    (8, 3),
    (5, 7),
    (-5, 7),
    (7, 5),
    (-7, 5),
    (8, 4),
    (6, 7),
    (-6, 7),
    (7, 6),
    (-7, 6),
    (8, 5),
    (7, 7),
    (-7, 7),
    (8, 6),
    (8, 7),
];

const TRUNCATED: &str = "WebP lossless data ends early";

/// Reads bits from the least significant end of each byte. Past the end of the data it gives
/// zeros and remembers how far the stream was over-read.
struct Bits<'a> {
    data: &'a [u8],
    /// Index of the next byte to load; it counts the zero bytes loaded past the end too.
    next: usize,
    window: u64,
    /// Valid bits in `window`.
    avail: u32,
    /// Bits the stream may supply.
    limit: u64,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Bits {
            data,
            next: 0,
            window: 0,
            avail: 0,
            limit: 8 * data.len().max(8) as u64,
        }
    }

    /// Fills the window to at least 56 bits.
    #[inline]
    fn refill(&mut self) {
        if let Some(chunk) = self.data.get(self.next..self.next.wrapping_add(8)) {
            let mut eight = [0u8; 8];
            eight.copy_from_slice(chunk);
            // bits above `avail` + 8 * take are the head of the next load: they are set again,
            // to the same value, when that load happens
            self.window |= u64::from_le_bytes(eight) << self.avail;
            let take = (63 - self.avail) >> 3;
            self.next += take as usize;
            self.avail += take * 8;
        } else {
            while self.avail <= 56 {
                let byte = self.data.get(self.next).copied().unwrap_or(0);
                self.window |= (byte as u64) << self.avail;
                self.next = self.next.saturating_add(1);
                self.avail += 8;
            }
        }
    }

    /// Makes at least `n` (at most 56) bits available to `peek` / `skip`.
    #[inline]
    fn need(&mut self, n: u32) {
        if self.avail < n {
            self.refill();
        }
    }

    #[inline]
    fn peek(&self, n: u32) -> u32 {
        (self.window & ((1u64 << n) - 1)) as u32
    }

    #[inline]
    fn skip(&mut self, n: u32) {
        self.window >>= n;
        self.avail -= n;
    }

    /// Reads `n` bits (at most 32) as an unsigned number, least significant bit first.
    #[inline]
    fn read(&mut self, n: u32) -> u32 {
        self.need(n);
        let v = self.peek(n);
        self.skip(n);
        v
    }

    /// True once more bits were read than the stream holds.
    #[inline]
    fn overrun(&self) -> bool {
        (self.next as u64) * 8 - self.avail as u64 > self.limit
    }
}

/// One canonical prefix code. Codes of up to `ROOT_BITS` bits are found with one lookup; longer
/// ones by walking the canonical order one bit at a time, which keeps the tables proportional
/// to the number of symbols.
#[derive(Clone, Copy, Default)]
struct Code {
    /// Offset of the lookup table in the arena: `1 << root_bits` entries of
    /// `symbol << 4 | length`.
    root: u32,
    root_bits: u8,
    /// Offset in the arena of the symbols with codes longer than `ROOT_BITS`, in code order.
    long: u32,
    /// First code value and number of codes for each length 9..=15.
    first: [u16; 7],
    count: [u16; 7],
}

impl Code {
    /// True when the code has one symbol, which takes no bits.
    fn is_single(&self) -> bool {
        self.root_bits == 0
    }
}

/// Builds the canonical code of `lengths` (one length per symbol, 0 = unused) into `arena`.
fn build_code(lengths: &[u8], arena: &mut Vec<u16>) -> Result<Code, String> {
    let mut count = [0u32; MAX_LENGTH + 1];
    for &l in lengths {
        count[(l as usize).min(MAX_LENGTH)] += 1;
    }
    let used = lengths.len() as u32 - count[0];
    if used == 0 {
        return Err("WebP lossless prefix code has no symbols".into());
    }
    let mut code = Code {
        root: arena.len() as u32,
        ..Code::default()
    };
    if used == 1 {
        let symbol = lengths.iter().position(|&l| l != 0).unwrap_or(0);
        arena.push((symbol as u16) << 4);
        return Ok(code);
    }
    // a complete code fills the code space exactly
    let mut space = 0u32;
    let mut longest = 0;
    for (len, &n) in count.iter().enumerate().skip(1) {
        space += n << (MAX_LENGTH - len);
        if n != 0 {
            longest = len as u32;
        }
    }
    if space != 1 << MAX_LENGTH {
        return Err("WebP lossless prefix code is not complete".into());
    }
    let root_bits = longest.min(ROOT_BITS);
    code.root_bits = root_bits as u8;
    let root = arena.len();
    arena.resize(root + (1usize << root_bits), 0);
    let long_total: u32 = count[ROOT_BITS as usize + 1..].iter().sum();
    code.long = arena.len() as u32;
    let long = arena.len();
    arena.resize(long + long_total as usize, 0);

    // first code value of every length, and where the long symbols of that length start
    let mut next = [0u32; MAX_LENGTH + 1];
    let mut long_at = [0usize; MAX_LENGTH + 1];
    let mut value = 0u32;
    let mut at = long;
    for len in 1..=MAX_LENGTH {
        value = (value + if len > 1 { count[len - 1] } else { 0 }) << 1;
        next[len] = value;
        if len > ROOT_BITS as usize {
            code.first[len - ROOT_BITS as usize - 1] = value as u16;
            code.count[len - ROOT_BITS as usize - 1] = count[len] as u16;
            long_at[len] = at;
            at += count[len] as usize;
        }
    }
    for (symbol, &l) in lengths.iter().enumerate() {
        let len = l as u32;
        if len == 0 {
            continue;
        }
        let value = next[len as usize];
        next[len as usize] += 1;
        if len <= root_bits {
            // the stream gives a code most significant bit first, the lookup index holds the
            // first bit read in its lowest bit
            let index = (value.reverse_bits() >> (32 - len)) as usize;
            let entry = ((symbol as u16) << 4) | len as u16;
            let mut i = index;
            while i < 1usize << root_bits {
                arena[root + i] = entry;
                i += 1usize << len;
            }
        } else {
            let prefix = value >> (len - ROOT_BITS);
            let index = (prefix.reverse_bits() >> (32 - ROOT_BITS)) as usize;
            arena[root + index] = LONG_CODE;
            arena[long_at[len as usize]] = symbol as u16;
            long_at[len as usize] += 1;
        }
    }
    Ok(code)
}

/// Reads one symbol. The caller has made 15 bits available.
#[inline]
fn read_symbol(code: &Code, arena: &[u16], bits: &mut Bits) -> u32 {
    let entry = arena[code.root as usize + bits.peek(code.root_bits as u32) as usize];
    let len = entry & 15;
    if len != LONG_CODE {
        bits.skip(len as u32);
        (entry >> 4) as u32
    } else {
        read_long_symbol(code, arena, bits)
    }
}

/// Reads a symbol whose code is longer than the lookup index.
#[inline(never)]
fn read_long_symbol(code: &Code, arena: &[u16], bits: &mut Bits) -> u32 {
    let mut value = (bits.peek(ROOT_BITS) as u8).reverse_bits() as u32;
    bits.skip(ROOT_BITS);
    let mut at = code.long as usize;
    for i in 0..7 {
        value = (value << 1) | bits.read(1);
        let index = value.wrapping_sub(code.first[i] as u32);
        if index < code.count[i] as u32 {
            return arena.get(at + index as usize).copied().unwrap_or(0) as u32;
        }
        at += code.count[i] as usize;
    }
    0
}

/// The prefix codes of an image: five per group (green + length + cache, red, blue, alpha,
/// distance), all tables in one arena.
struct Groups {
    arena: Vec<u16>,
    codes: Vec<[Code; 5]>,
}

/// What the meta prefix codes of an image say: which group codes each block.
struct Meta {
    bits: u32,
    width: usize,
    /// Group of every block, as an index into `Groups::codes`.
    group: Vec<u32>,
}

enum Transform {
    Predictor {
        bits: u32,
        data: Vec<u32>,
        width: usize,
    },
    CrossColor {
        bits: u32,
        data: Vec<u32>,
        width: usize,
    },
    SubtractGreen,
    /// `width` is the image width after the indices are unpacked.
    Palette {
        bits: u32,
        table: Vec<u32>,
        width: usize,
    },
}

/// Size of a sub-sampled image: `size / 2^bits`, rounded up.
fn subsample(size: usize, bits: u32) -> usize {
    (size + (1usize << bits) - 1) >> bits
}

struct Decoder<'a> {
    bits: Bits<'a>,
    /// Scratch for the code lengths of the code being read.
    lengths: Vec<u8>,
    /// Scratch for the tables of the code-length code.
    length_arena: Vec<u16>,
    /// True when every code group of the main image has single-symbol red, blue and alpha
    /// codes.
    plain_literals: bool,
    /// Set for an image that may be read past its end while its last pixels are decoded: an
    /// alpha plane stored as palette indices (see `read_main`).
    lenient_tail: bool,
    /// Whether the main image met every condition of `lenient_tail`.
    tail_accepted: bool,
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Decoder {
            bits: Bits::new(data),
            lengths: vec![0; 256 + 24 + (1 << 11)],
            length_arena: Vec::new(),
            plain_literals: true,
            lenient_tail: false,
            tail_accepted: false,
        }
    }

    /// Reads the code lengths of one prefix code and builds it.
    fn read_code(&mut self, alphabet: usize, arena: &mut Vec<u16>) -> Result<Code, String> {
        let span = alphabet.max(256);
        self.lengths[..span].fill(0);
        if self.bits.read(1) == 1 {
            // simple code: one or two symbols, the first stored in 1 or 8 bits
            let two = self.bits.read(1) == 1;
            let wide = self.bits.read(1) == 1;
            let first = self.bits.read(if wide { 8 } else { 1 }) as usize;
            self.lengths[first] = 1;
            if two {
                let second = self.bits.read(8) as usize;
                self.lengths[second] = 1;
            }
        } else {
            let mut length_lengths = [0u8; 19];
            let stored = 4 + self.bits.read(4) as usize;
            for &slot in LENGTH_ORDER.iter().take(stored) {
                length_lengths[slot] = self.bits.read(3) as u8;
            }
            self.length_arena.clear();
            let length_code = build_code(&length_lengths, &mut self.length_arena)?;
            // the number of code-length symbols that follow may be given explicitly
            let mut tokens = alphabet;
            if self.bits.read(1) == 1 {
                let width = 2 + 2 * self.bits.read(3);
                tokens = 2 + self.bits.read(width) as usize;
                if tokens > alphabet {
                    return Err("WebP lossless code length count is too large".into());
                }
            }
            let mut symbol = 0;
            let mut previous = 8u8;
            while symbol < alphabet && tokens > 0 {
                tokens -= 1;
                self.bits.need(15);
                let token = read_symbol(&length_code, &self.length_arena, &mut self.bits);
                if token < 16 {
                    self.lengths[symbol] = token as u8;
                    symbol += 1;
                    if token != 0 {
                        previous = token as u8;
                    }
                } else {
                    let (extra, base) = match token {
                        16 => (2, 3),
                        17 => (3, 3),
                        _ => (7, 11),
                    };
                    let repeat = self.bits.read(extra) as usize + base;
                    if symbol + repeat > alphabet {
                        return Err("WebP lossless code lengths run past the alphabet".into());
                    }
                    let value = if token == 16 { previous } else { 0 };
                    self.lengths[symbol..symbol + repeat].fill(value);
                    symbol += repeat;
                }
            }
        }
        if self.bits.overrun() {
            return Err(TRUNCATED.into());
        }
        build_code(&self.lengths[..alphabet], arena)
    }

    /// Reads the five codes of one group.
    fn read_group(&mut self, cache_bits: u32, arena: &mut Vec<u16>) -> Result<[Code; 5], String> {
        let cache = if cache_bits > 0 {
            1usize << cache_bits
        } else {
            0
        };
        let mut group = [Code::default(); 5];
        for (kind, code) in group.iter_mut().enumerate() {
            let alphabet = match kind {
                GREEN => 256 + 24 + cache,
                DIST => 40,
                _ => 256,
            };
            *code = self.read_code(alphabet, arena)?;
        }
        Ok(group)
    }

    /// Decodes one entropy-coded image of `width` x `height` pixels: its colour cache size,
    /// its meta prefix codes (the main image only), its prefix codes and its pixels.
    fn read_image(&mut self, width: usize, height: usize, main: bool) -> Result<Vec<u32>, String> {
        let mut cache_bits = 0;
        if self.bits.read(1) == 1 {
            cache_bits = self.bits.read(4);
            if !(1..=11).contains(&cache_bits) {
                return Err("WebP lossless colour cache size is invalid".into());
            }
        }
        let mut meta = None;
        if main && self.bits.read(1) == 1 {
            let bits = self.bits.read(3) + 2;
            let meta_width = subsample(width, bits);
            let image = self.read_image(meta_width, subsample(height, bits), false)?;
            meta = Some(Meta {
                bits,
                width: meta_width,
                group: image.iter().map(|&p| (p >> 8) & 0xffff).collect(),
            });
        }
        let groups = self.read_groups(cache_bits, &mut meta, width, height, main)?;
        let lenient = main && self.lenient_tail && cache_bits == 0 && self.plain_literals;
        if main {
            self.tail_accepted = lenient;
        }
        self.read_pixels(width, height, cache_bits, &groups, meta.as_ref(), lenient)
    }

    /// Reads all code groups of an image. Groups no block refers to are checked and dropped;
    /// `meta.group` is renumbered to the groups that are kept.
    fn read_groups(
        &mut self,
        cache_bits: u32,
        meta: &mut Option<Meta>,
        width: usize,
        height: usize,
        main: bool,
    ) -> Result<Groups, String> {
        let mut groups = Groups {
            arena: Vec::new(),
            codes: Vec::new(),
        };
        let total = match meta {
            Some(m) => m.group.iter().copied().max().unwrap_or(0) as usize + 1,
            None => 1,
        };
        let mut slot = vec![u32::MAX; total];
        match meta {
            Some(m) => {
                for &g in &m.group {
                    slot[g as usize] = 0;
                }
            }
            None => slot[0] = 0,
        }
        // the reference decoder looks at unused groups too when their number is small
        let counts_unused = total <= 1000 && total as u64 <= width as u64 * height as u64;
        let mut spare = Vec::new();
        for entry in slot.iter_mut() {
            let used = *entry == 0;
            let plain;
            if used {
                let group = self.read_group(cache_bits, &mut groups.arena)?;
                plain =
                    group[RED].is_single() && group[BLUE].is_single() && group[ALPHA].is_single();
                *entry = groups.codes.len() as u32;
                groups.codes.push(group);
                let bytes =
                    groups.arena.len() * 2 + groups.codes.len() * std::mem::size_of::<[Code; 5]>();
                if bytes > MAX_TABLE_BYTES {
                    return Err("WebP lossless image has too many prefix codes".into());
                }
            } else {
                spare.clear();
                let group = self.read_group(cache_bits, &mut spare)?;
                plain = !counts_unused
                    || (group[RED].is_single()
                        && group[BLUE].is_single()
                        && group[ALPHA].is_single());
            }
            if main && !plain {
                self.plain_literals = false;
            }
        }
        if let Some(m) = meta {
            for g in m.group.iter_mut() {
                *g = slot[*g as usize];
            }
        }
        Ok(groups)
    }

    /// Decodes the pixels of one image: literals, colour-cache hits and backward copies.
    fn read_pixels(
        &mut self,
        width: usize,
        height: usize,
        cache_bits: u32,
        groups: &Groups,
        meta: Option<&Meta>,
        lenient: bool,
    ) -> Result<Vec<u32>, String> {
        if self.bits.overrun() {
            return Err(TRUNCATED.into());
        }
        let total = width * height;
        let mut out = vec![0u32; total];
        let mut cache = vec![0u32; if cache_bits > 0 { 1 << cache_bits } else { 0 }];
        let cache_shift = 32 - cache_bits;
        let arena = &groups.arena[..];
        let mask = meta.map_or(usize::MAX, |m| (1usize << m.bits) - 1);
        let group_at = |x: usize, y: usize| -> &[Code; 5] {
            match meta {
                Some(m) => {
                    let index = m.group[(y >> m.bits) * m.width + (x >> m.bits)];
                    &groups.codes[index as usize]
                }
                None => &groups.codes[0],
            }
        };
        let mut group = group_at(0, 0);
        let mut pos = 0usize;
        let mut x = 0usize;
        let mut y = 0usize;
        while pos < total {
            if x & mask == 0 {
                group = group_at(x, y);
            }
            self.bits.need(32);
            let green = read_symbol(&group[GREEN], arena, &mut self.bits);
            if green < 256 {
                let red = read_symbol(&group[RED], arena, &mut self.bits);
                self.bits.need(32);
                let blue = read_symbol(&group[BLUE], arena, &mut self.bits);
                let alpha = read_symbol(&group[ALPHA], arena, &mut self.bits);
                let pixel = (alpha << 24) | (red << 16) | (green << 8) | blue;
                out[pos] = pixel;
                if cache_bits > 0 {
                    cache[(pixel.wrapping_mul(0x1e35_a7bd) >> cache_shift) as usize] = pixel;
                }
                pos += 1;
                x += 1;
                if x == width {
                    x = 0;
                    y += 1;
                }
            } else if green < 256 + 24 {
                let length = self.read_extended(green - 256) as usize;
                self.bits.need(32);
                let symbol = read_symbol(&group[DIST], arena, &mut self.bits);
                let code = self.read_extended(symbol) as usize;
                let distance = if code > NEAR.len() {
                    code - NEAR.len()
                } else {
                    let (dx, dy) = NEAR[code - 1];
                    (dy as i64 * width as i64 + dx as i64).max(1) as usize
                };
                if !lenient && self.bits.overrun() {
                    return Err(TRUNCATED.into());
                }
                if distance > pos || length > total - pos {
                    return Err("WebP lossless copy is outside the image".into());
                }
                if distance >= length {
                    out.copy_within(pos - distance..pos - distance + length, pos);
                } else {
                    for i in pos..pos + length {
                        out[i] = out[i - distance];
                    }
                }
                if cache_bits > 0 {
                    for &pixel in &out[pos..pos + length] {
                        cache[(pixel.wrapping_mul(0x1e35_a7bd) >> cache_shift) as usize] = pixel;
                    }
                }
                pos += length;
                x += length;
                while x >= width {
                    x -= width;
                    y += 1;
                }
                if pos < total && x & mask != 0 {
                    group = group_at(x, y);
                }
            } else {
                // the alphabet has cache symbols only when the image has a cache
                let key = (green - 280) as usize;
                let pixel = cache.get(key).copied().unwrap_or(0);
                out[pos] = pixel;
                // every decoded pixel enters the cache: one read from a slot nothing was
                // stored in (zero) belongs to another slot
                if cache_bits > 0 {
                    cache[(pixel.wrapping_mul(0x1e35_a7bd) >> cache_shift) as usize] = pixel;
                }
                pos += 1;
                x += 1;
                if x == width {
                    x = 0;
                    y += 1;
                }
            }
            if pos < total && self.bits.overrun() {
                return Err(TRUNCATED.into());
            }
        }
        Ok(out)
    }

    /// The value of a length or distance prefix symbol with its extra bits.
    #[inline]
    fn read_extended(&mut self, symbol: u32) -> u32 {
        if symbol < 4 {
            return symbol + 1;
        }
        let extra = (symbol - 2) >> 1;
        let offset = (2 + (symbol & 1)) << extra;
        offset + self.bits.read(extra) + 1
    }

    /// Decodes a main image: its transforms, its pixels, and the inverse transforms.
    ///
    /// With `alpha_plane` set the image is the alpha plane of a lossy frame. The reference
    /// decoder reads such a plane with a byte-wide loop when it is nothing but palette indices
    /// (one colour-indexing transform, no colour cache, single-symbol red / blue / alpha
    /// codes), and that loop accepts a stream whose last symbol runs past the end of the data
    /// (the missing bits read as zeros). Every other image must end inside its data.
    fn read_main(
        &mut self,
        width: usize,
        height: usize,
        alpha_plane: bool,
    ) -> Result<Vec<u32>, String> {
        let mut transforms: Vec<Transform> = Vec::new();
        let mut seen = 0u32;
        let mut coded_width = width;
        while self.bits.read(1) == 1 {
            let kind = self.bits.read(2);
            if seen & (1 << kind) != 0 {
                return Err("WebP lossless transform is used twice".into());
            }
            seen |= 1 << kind;
            match kind {
                0 | 1 => {
                    let bits = self.bits.read(3) + 2;
                    let data = self.read_image(
                        subsample(coded_width, bits),
                        subsample(height, bits),
                        false,
                    )?;
                    transforms.push(if kind == 0 {
                        Transform::Predictor {
                            bits,
                            data,
                            width: coded_width,
                        }
                    } else {
                        Transform::CrossColor {
                            bits,
                            data,
                            width: coded_width,
                        }
                    });
                }
                2 => transforms.push(Transform::SubtractGreen),
                _ => {
                    let colors = self.bits.read(8) as usize + 1;
                    let bits = match colors {
                        0..=2 => 3,
                        3..=4 => 2,
                        5..=16 => 1,
                        _ => 0,
                    };
                    let mut table = self.read_image(colors, 1, false)?;
                    // the table is stored as differences between neighbouring entries
                    for i in 1..colors {
                        table[i] = add_pixels(table[i], table[i - 1]);
                    }
                    table.resize(1usize << (8 >> bits), 0);
                    transforms.push(Transform::Palette {
                        bits,
                        table,
                        width: coded_width,
                    });
                    coded_width = subsample(coded_width, bits);
                }
            }
            if self.bits.overrun() {
                return Err(TRUNCATED.into());
            }
        }
        self.lenient_tail = alpha_plane && matches!(transforms[..], [Transform::Palette { .. }]);
        let mut pixels = self.read_image(coded_width, height, true)?;
        if self.bits.overrun() && !self.tail_accepted {
            return Err(TRUNCATED.into());
        }
        for transform in transforms.iter().rev() {
            match transform {
                Transform::Predictor { bits, data, width } => {
                    undo_predictor(&mut pixels, *width, height, *bits, data)
                }
                Transform::CrossColor { bits, data, width } => {
                    undo_cross_color(&mut pixels, *width, *bits, data)
                }
                Transform::SubtractGreen => {
                    for p in pixels.iter_mut() {
                        let green = (*p >> 8) & 0xff;
                        let red_blue = (*p & 0x00ff_00ff).wrapping_add((green << 16) | green);
                        *p = (*p & 0xff00_ff00) | (red_blue & 0x00ff_00ff);
                    }
                }
                Transform::Palette { bits, table, width } => {
                    pixels = undo_palette(&pixels, *width, height, *bits, table);
                }
            }
        }
        Ok(pixels)
    }
}

/// Adds two pixels channel by channel, modulo 256.
#[inline(always)]
fn add_pixels(a: u32, b: u32) -> u32 {
    let alpha_green = (a & 0xff00_ff00).wrapping_add(b & 0xff00_ff00) & 0xff00_ff00;
    let red_blue = (a & 0x00ff_00ff).wrapping_add(b & 0x00ff_00ff) & 0x00ff_00ff;
    alpha_green | red_blue
}

/// Channel-wise average of two pixels, rounded down.
#[inline(always)]
fn average(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xfefe_fefe) >> 1).wrapping_add(a & b)
}

#[inline(always)]
fn channel(p: u32, shift: u32) -> i32 {
    ((p >> shift) & 0xff) as i32
}

/// The left or the top pixel, whichever is closer to `left + top - top_left`.
#[inline(always)]
fn select(left: u32, top: u32, top_left: u32) -> u32 {
    let mut to_left = 0;
    let mut to_top = 0;
    for shift in [0, 8, 16, 24] {
        to_left += (channel(top, shift) - channel(top_left, shift)).abs();
        to_top += (channel(left, shift) - channel(top_left, shift)).abs();
    }
    if to_left < to_top {
        left
    } else {
        top
    }
}

/// `left + top - top_left` per channel, clamped to 0..=255.
#[inline(always)]
fn clamped_gradient(left: u32, top: u32, top_left: u32) -> u32 {
    let mut out = 0;
    for shift in [0, 8, 16, 24] {
        let v = channel(left, shift) + channel(top, shift) - channel(top_left, shift);
        out |= (v.clamp(0, 255) as u32) << shift;
    }
    out
}

/// `a + (a - b) / 2` per channel (the division truncates towards zero), clamped to 0..=255.
#[inline(always)]
fn clamped_half_gradient(a: u32, b: u32) -> u32 {
    let mut out = 0;
    for shift in [0, 8, 16, 24] {
        let v = channel(a, shift) + (channel(a, shift) - channel(b, shift)) / 2;
        out |= (v.clamp(0, 255) as u32) << shift;
    }
    out
}

/// Adds the prediction of `predict(left, top, top_left, top_right)` to the pixels
/// `start..end` of one row (not the first row, not the first column).
#[inline(always)]
fn add_predicted(
    pixels: &mut [u32],
    width: usize,
    start: usize,
    end: usize,
    predict: impl Fn(u32, u32, u32, u32) -> u32,
) {
    for i in start..end {
        let prediction = predict(
            pixels[i - 1],
            pixels[i - width],
            pixels[i - width - 1],
            pixels[i - width + 1],
        );
        pixels[i] = add_pixels(pixels[i], prediction);
    }
}

/// Inverse predictor transform, in place. The top-left pixel is predicted as opaque black, the
/// rest of the first row from the left, the first column from above, and every other pixel by
/// the mode of its block. The top-right neighbour of the last pixel of a row is the first
/// pixel of that row (the next pixel in memory).
fn undo_predictor(pixels: &mut [u32], width: usize, height: usize, bits: u32, modes: &[u32]) {
    pixels[0] = add_pixels(pixels[0], 0xff00_0000);
    for i in 1..width {
        pixels[i] = add_pixels(pixels[i], pixels[i - 1]);
    }
    let blocks_per_row = subsample(width, bits);
    for y in 1..height {
        let row = y * width;
        pixels[row] = add_pixels(pixels[row], pixels[row - width]);
        let mode_row = (y >> bits) * blocks_per_row;
        let mut x = 1;
        while x < width {
            let mode = (modes[mode_row + (x >> bits)] >> 8) & 15;
            let end = (((x >> bits) + 1) << bits).min(width);
            let (from, to) = (row + x, row + end);
            match mode {
                1 => add_predicted(pixels, width, from, to, |l, _, _, _| l),
                2 => add_predicted(pixels, width, from, to, |_, t, _, _| t),
                3 => add_predicted(pixels, width, from, to, |_, _, _, tr| tr),
                4 => add_predicted(pixels, width, from, to, |_, _, tl, _| tl),
                5 => add_predicted(pixels, width, from, to, |l, t, _, tr| {
                    average(average(l, tr), t)
                }),
                6 => add_predicted(pixels, width, from, to, |l, _, tl, _| average(l, tl)),
                7 => add_predicted(pixels, width, from, to, |l, t, _, _| average(l, t)),
                8 => add_predicted(pixels, width, from, to, |_, t, tl, _| average(tl, t)),
                9 => add_predicted(pixels, width, from, to, |_, t, _, tr| average(t, tr)),
                10 => add_predicted(pixels, width, from, to, |l, t, tl, tr| {
                    average(average(l, tl), average(t, tr))
                }),
                11 => add_predicted(pixels, width, from, to, |l, t, tl, _| select(l, t, tl)),
                12 => add_predicted(pixels, width, from, to, |l, t, tl, _| {
                    clamped_gradient(l, t, tl)
                }),
                13 => add_predicted(pixels, width, from, to, |l, t, tl, _| {
                    clamped_half_gradient(average(l, t), tl)
                }),
                // 0, and the two unassigned modes the reference decoder treats alike
                _ => add_predicted(pixels, width, from, to, |_, _, _, _| 0xff00_0000),
            }
            x = end;
        }
    }
}

/// `(t * c) >> 5` of two signed bytes, the low byte of which is used.
#[inline(always)]
fn color_delta(t: u32, c: u32) -> u32 {
    ((t as u8 as i8 as i32 * c as u8 as i8 as i32) >> 5) as u32
}

/// Inverse cross-colour transform, in place: red gets a share of green, blue a share of green
/// and of the restored red.
fn undo_cross_color(pixels: &mut [u32], width: usize, bits: u32, data: &[u32]) {
    let blocks_per_row = subsample(width, bits);
    for (y, row) in pixels.chunks_exact_mut(width).enumerate() {
        let block_row = (y >> bits) * blocks_per_row;
        for (x, p) in row.iter_mut().enumerate() {
            let element = data[block_row + (x >> bits)];
            let (green_to_red, green_to_blue, red_to_blue) = (
                element & 0xff,
                (element >> 8) & 0xff,
                (element >> 16) & 0xff,
            );
            let green = (*p >> 8) & 0xff;
            let red = (((*p >> 16) & 0xff).wrapping_add(color_delta(green_to_red, green))) & 0xff;
            let blue = (*p & 0xff)
                .wrapping_add(color_delta(green_to_blue, green))
                .wrapping_add(color_delta(red_to_blue, red))
                & 0xff;
            *p = (*p & 0xff00_ff00) | (red << 16) | blue;
        }
    }
}

/// Inverse colour-indexing transform: every index in the green channel becomes its table
/// entry. With `bits` > 0 one coded pixel packs `1 << bits` indices, lowest bits first.
fn undo_palette(coded: &[u32], width: usize, height: usize, bits: u32, table: &[u32]) -> Vec<u32> {
    let lookup = |index: u32| table.get(index as usize).copied().unwrap_or(0);
    if bits == 0 {
        return coded.iter().map(|&p| lookup((p >> 8) & 0xff)).collect();
    }
    let coded_width = subsample(width, bits);
    let index_bits = 8 >> bits;
    let index_mask = (1u32 << index_bits) - 1;
    let per_pixel = 1usize << bits;
    let mut out = vec![0u32; width * height];
    for (src, dst) in coded
        .chunks_exact(coded_width)
        .zip(out.chunks_exact_mut(width))
    {
        for (&packed, group) in src.iter().zip(dst.chunks_mut(per_pixel)) {
            let mut indices = (packed >> 8) & 0xff;
            for p in group {
                *p = lookup(indices & index_mask);
                indices >>= index_bits;
            }
        }
    }
    out
}

/// Size and alpha hint of a lossless bitstream, from its 5-byte header; `None` when the
/// signature or the version is wrong.
pub(super) fn read_header(data: &[u8]) -> Option<(usize, usize, bool)> {
    if data.len() < 5 || data[0] != 0x2f || data[4] >> 5 != 0 {
        return None;
    }
    let bits = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
    let width = (bits & 0x3fff) as usize + 1;
    let height = ((bits >> 14) & 0x3fff) as usize + 1;
    Some((width, height, (bits >> 28) & 1 == 1))
}

/// Decodes a lossless bitstream (the payload of a `VP8L` chunk) to ARGB pixels, row by row.
pub(super) fn decode(
    data: &[u8],
    limits: &DecodeLimits,
) -> Result<(usize, usize, Vec<u32>), String> {
    let (width, height, _) = read_header(data).ok_or("WebP lossless header is invalid")?;
    limits.check_canvas(width, height)?;
    let mut decoder = Decoder::new(data);
    decoder.bits.read(32);
    decoder.bits.read(8);
    let pixels = decoder.read_main(width, height, false)?;
    Ok((width, height, pixels))
}

/// Decodes a losslessly coded alpha plane of `width` x `height` (already checked against the
/// limits by the caller): a lossless stream without its header, the green channel of which is
/// the alpha value.
pub(super) fn decode_alpha_plane(
    data: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, String> {
    let mut decoder = Decoder::new(data);
    let pixels = decoder.read_main(width, height, true)?;
    Ok(pixels.iter().map(|&p| (p >> 8) as u8).collect())
}

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

    /// Simple lossless fixtures with the FNV-1a 64 hash of the RGB bytes the reference image
    /// library returns (from `tests/data/vision/webp/make_fixtures.py`).
    const FIXTURES: &[(&str, &[u8], u64)] = &[
        ("ll_1x1", fixture!("ll_1x1.webp"), 0x2d44_3218_9add_adc3),
        (
            "ll_pal2_33x9",
            fixture!("ll_pal2_33x9.webp"),
            0xb099_2b46_5515_a2ba,
        ),
        (
            "ll_pal4_33x9",
            fixture!("ll_pal4_33x9.webp"),
            0x3d44_c106_9b1a_fb12,
        ),
        (
            "ll_pal16_33x9",
            fixture!("ll_pal16_33x9.webp"),
            0xd5e9_c298_1acc_7450,
        ),
        (
            "ll_pal40_33x9",
            fixture!("ll_pal40_33x9.webp"),
            0xc914_8f8c_2fbd_b7a3,
        ),
        (
            "ll_photo_40x30",
            fixture!("ll_photo_40x30.webp"),
            0x3c28_6988_ced3_01a8,
        ),
        (
            "ll_synth_a",
            fixture!("ll_synth_a.webp"),
            0x45f8_3599_480c_c6ca,
        ),
        (
            "ll_synth_b",
            fixture!("ll_synth_b.webp"),
            0x6b32_e363_e5cc_0002,
        ),
        (
            "ll_synth_c",
            fixture!("ll_synth_c.webp"),
            0x8ec6_a3da_f0a9_e1fc,
        ),
        (
            "ll_tiles_45x35",
            fixture!("ll_tiles_45x35.webp"),
            0x68c7_b32e_1845_ea16,
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
        assert_eq!(&file[12..16], b"VP8L");
        &file[20..]
    }

    fn rgb(pixels: &[u32]) -> Vec<u8> {
        pixels
            .iter()
            .flat_map(|&p| [(p >> 16) as u8, (p >> 8) as u8, p as u8])
            .collect()
    }

    #[test]
    fn images_decode_to_the_reference_pixels() {
        for &(name, file, hash) in FIXTURES {
            let (width, height, pixels) =
                decode(bitstream(file), &LIMITS).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(pixels.len(), width * height, "{name}");
            assert_eq!(fnv1a(&rgb(&pixels)), hash, "{name}");
        }
    }

    #[test]
    fn alpha_channel_is_decoded_with_the_colours() {
        let data = bitstream(fixture!("ll_photo_alpha_31x16.webp"));
        let (width, height, pixels) = decode(data, &LIMITS).unwrap();
        assert_eq!((width, height), (31, 16));
        assert_eq!(read_header(data), Some((31, 16, true)));
        assert_eq!(fnv1a(&rgb(&pixels)), 0x65b6_ad38_6b19_57f8);
        let alpha: Vec<u8> = pixels.iter().map(|&p| (p >> 24) as u8).collect();
        assert_eq!(fnv1a(&alpha), 0xef8c_a298_071c_1945);
        // fully transparent pixels keep the colour the file stores
        assert!(pixels.iter().any(|&p| p >> 24 == 0 && p & 0x00ff_ffff != 0));
    }

    #[test]
    fn header_is_checked() {
        let good = bitstream(FIXTURES[5].1);
        assert_eq!(read_header(good), Some((40, 30, false)));
        assert_eq!(read_header(&good[..4]), None);
        let mut bad = good.to_vec();
        bad[0] = 0x2e;
        assert_eq!(read_header(&bad), None, "signature");
        assert!(decode(&bad, &LIMITS).is_err());
        let mut versioned = good.to_vec();
        versioned[4] |= 0x20;
        assert_eq!(read_header(&versioned), None, "version");
        // the largest size a header can give
        assert_eq!(
            read_header(&[0x2f, 0xff, 0xff, 0xff, 0x0f]),
            Some((16384, 16384, false))
        );
        assert_eq!(
            decode(&[0x2f, 0xff, 0xff, 0xff, 0x0f, 0, 0, 0], &LIMITS).err(),
            LIMITS.check_canvas(16384, 16384).err()
        );
    }

    #[test]
    fn image_over_the_limit_is_refused() {
        let good = bitstream(FIXTURES[5].1);
        let exact = DecodeLimits {
            max_pixels: 40 * 30,
        };
        assert!(decode(good, &exact).is_ok());
        let tight = DecodeLimits {
            max_pixels: 40 * 30 - 1,
        };
        assert!(tight.check_canvas(40, 30).is_err());
        assert_eq!(decode(good, &tight).err(), tight.check_canvas(40, 30).err());
    }

    #[test]
    fn truncated_images_are_refused() {
        for &(name, file, _) in FIXTURES {
            let data = bitstream(file);
            // the chunk may end with a padding byte the stream does not use
            let used = data.len() - (file[16] & 1) as usize;
            for cut in 0..used {
                let result = decode(&data[..cut], &LIMITS);
                // a stream of under 8 bytes is read as 8 bytes, zero filled; every longer one
                // that lost its end runs out of bits
                if !(5..8).contains(&cut) {
                    assert!(result.is_err(), "{name} cut at {cut}");
                }
            }
        }
    }

    #[test]
    fn corrupted_images_never_panic() {
        let mut state = 0x7f4a_7c15_9e37_79b9u64;
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
                let _ = decode(&data, &small);
                let _ = decode_alpha_plane(&data[5..], 33, 9);
            }
        }
    }

    /// Reads symbols of a code built from `lengths` out of `bytes`.
    fn symbols(lengths: &[u8], bytes: &[u8], count: usize) -> Vec<u32> {
        let mut arena = Vec::new();
        let code = build_code(lengths, &mut arena).unwrap();
        let mut bits = Bits::new(bytes);
        (0..count)
            .map(|_| {
                bits.need(15);
                read_symbol(&code, &arena, &mut bits)
            })
            .collect()
    }

    #[test]
    fn prefix_codes_are_canonical_and_complete() {
        let mut arena = Vec::new();
        // lengths 1, 2, 3, 3: codes 0, 10, 110, 111, read first bit first
        assert_eq!(
            symbols(&[1, 2, 3, 3], &[0b1011_1010, 0b0000_0001], 5),
            [0, 1, 3, 0, 2]
        );
        // one used symbol takes no bits, whatever its length
        let single = build_code(&[0, 0, 7, 0], &mut arena).unwrap();
        assert!(single.is_single());
        assert_eq!(symbols(&[0, 0, 7, 0], &[0xff], 3), [2, 2, 2]);
        // not complete, over-subscribed, empty
        assert!(build_code(&[1, 2, 3], &mut arena).is_err());
        assert!(build_code(&[1, 1, 1], &mut arena).is_err());
        assert!(build_code(&[0, 0, 0], &mut arena).is_err());
        // codes longer than the lookup index: 1, 2, ..., 14, 15, 15
        let mut lengths: Vec<u8> = (1..=15).collect();
        lengths.push(15);
        // symbol k is k ones and a zero; the last two are fourteen ones and a zero, and fifteen
        // ones
        assert_eq!(symbols(&lengths, &[0xff, 0xbf, 0xff, 0x7f], 2), [14, 15]);
        assert_eq!(symbols(&lengths, &[0xff, 0x7f, 0x00], 2), [15, 0]);
        assert_eq!(symbols(&lengths, &[0b0000_0010], 2), [0, 1]);
    }

    #[test]
    fn bits_past_the_end_are_zero_and_counted() {
        let mut bits = Bits::new(&[0xff; 9]);
        assert_eq!(bits.read(32), u32::MAX);
        assert_eq!(bits.read(32), u32::MAX);
        assert_eq!(bits.read(8), 0xff);
        assert!(!bits.overrun());
        assert_eq!(bits.read(4), 0);
        assert!(bits.overrun());
        // a stream of under 8 bytes counts as 8
        let mut short = Bits::new(&[0xff]);
        assert_eq!(short.read(16), 0xff);
        assert_eq!(short.read(32), 0);
        assert_eq!(short.read(16), 0);
        assert!(!short.overrun());
        short.read(1);
        assert!(short.overrun());
    }

    #[test]
    fn inverse_transforms_follow_their_edge_rules() {
        // predictor: opaque black for the first pixel, left along the first row, above down the
        // first column, the block mode elsewhere (2 = above, 14 = black like 0)
        let mut px = vec![0x0102_0304u32; 6];
        undo_predictor(&mut px, 3, 2, 2, &[2 << 8]);
        assert_eq!(
            px,
            [
                0x0002_0304,
                0x0104_0608,
                0x0206_090c,
                0x0104_0608,
                0x0206_090c,
                0x0308_0c10
            ]
        );
        let mut px = vec![0x0102_0304u32; 4];
        undo_predictor(&mut px, 2, 2, 2, &[14 << 8]);
        assert_eq!(px[3], 0x0002_0304);
        // the half-gradient predictor divides towards zero
        assert_eq!(clamped_half_gradient(0x0000_0005, 0x0000_0008), 0x0000_0004);
        assert_eq!(clamped_half_gradient(0x00ff_0000, 0), 0x00ff_0000);
        assert_eq!(select(0x10, 0x20, 0x11), 0x20);
        // palette: 4 colours pack 4 indices per pixel, lowest bits first; a missing entry is
        // transparent black
        let table = [0xff00_0001, 0xff00_0002, 0xff00_0003, 0];
        let out = undo_palette(&[0b1110_0100 << 8, 0b0000_0001 << 8], 5, 1, 2, &table);
        assert_eq!(out, [0xff00_0001, 0xff00_0002, 0xff00_0003, 0, 0xff00_0002]);
        assert_eq!(undo_palette(&[200 << 8], 1, 1, 0, &table), [0]);
    }

    /// Compares every simple lossless `<name>.webp` of the directory in
    /// `GAUDIUM_IMAGE_REF_DIR` with the raw RGB dump `<name>.rgb` beside it.
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
            if file.len() < 25 || &file[12..16] != b"VP8L" {
                continue;
            }
            // the chunk with its padding byte, as the container hands it over
            let size = u32::from_le_bytes([file[16], file[17], file[18], file[19]]) as usize;
            let Some(data) = file.get(20..20 + size + (size & 1)) else {
                continue;
            };
            compared += 1;
            match decode(data, &limits) {
                Ok((_, _, pixels)) if rgb(&pixels) == want => {}
                Ok(_) => failures.push(format!("{}: pixels differ", path.display())),
                Err(e) => failures.push(format!("{}: {e}", path.display())),
            }
        }
        println!(
            "{compared} lossless files compared, {} failures",
            failures.len()
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
