//! GIF decoder: GIF87a and GIF89a, still images and animations, to RGB8.
//!
//! Decoded: global and local colour tables, interlaced frames, LZW pixel data of every code
//! size the reference takes (0 to 12 bits), graphic control extensions (delay, transparent
//! index, disposal), frames smaller than or offset in the logical screen. Comment, plain-text
//! and application extensions are skipped.
//!
//! Refused: a file without an image, a frame of zero size, a code size above 12, damaged or
//! incomplete pixel data, an animation whose later frame reaches outside the canvas of the
//! first one (the reference enlarges the image from that frame on, which one canvas size can
//! not express), and the one layout the reference fails on itself (`Animation::advance`).
//!
//! Every frame equals, byte for byte, what the reference image library shows for it after a
//! conversion to RGB. The rules that make it so:
//!
//! * The canvas is the logical screen, enlarged to hold the first frame.
//! * A colour table whose every entry `i` is the grey `(i, i, i)` counts as no table, and a
//!   frame without a table shows its indices as grey levels. An index past the end of a table
//!   is black.
//! * The first frame takes its local table, or the global one when it has no local table (a
//!   local grey ramp is none). It is painted whole, its transparent index being a colour like
//!   any other, on a canvas filled with the colour of that index, or of index 0 without one.
//!   The background index of the header is not used here.
//! * A later frame paints the pixels that differ from its transparent index with its local
//!   table, or with the global one when the file gives it no local table. As long as no frame
//!   had a table, though, the canvas is a picture of the first frame's colours: frames are
//!   painted with those, the first frame that has a table included, and only the frames after
//!   that one use their own.
//! * The disposal method is sticky: a frame that names none keeps the method of the frame
//!   before. "Restore to background" fills the frame's rectangle with the colour of the
//!   frame's transparent index, else of the header's background index, looked up in that
//!   frame's table (entry 0 for an index past its end). "Restore to previous" puts back what
//!   the rectangle showed before the frame; on the first frame it fills with the transparent
//!   index and does nothing without one.
//! * A frame's delay is the value of its last graphic control extension times 10 ms, and 0
//!   without one.
//! * Pixel data is read as one chain of sub-blocks that does not stop at an empty sub-block,
//!   needs every sub-block whole, and ends when the frame is full. An end code does not end
//!   it. The file is taken in pieces of `FEED` bytes, and an end code or an incomplete
//!   sub-block is an error exactly when no further piece is left.
//! * The block structure is walked as the reference walks it (see `next_frame`): a file cut
//!   inside the structure of a later frame ends the animation at the frame before.

use super::types::{DecodeLimits, Scan, Still};

/// The reference hands the pixel decoder the file in pieces of this many bytes, counted from
/// a frame's first data sub-block.
const FEED: usize = 65536;
/// Entries of the LZW string table (12-bit codes).
const TABLE: usize = 4096;
/// Largest LZW code size the reference takes from the file.
const MAX_CODE_BITS: u8 = 12;

const NO_IMAGE: &str = "the file holds no image (it ends before the first frame)";
const CUT: &str = "the file ends inside the pixel data";
const DAMAGED: &str = "the pixel data is damaged (invalid LZW code)";

/// A colour table as a frame names it.
#[derive(Clone, Copy)]
enum Local<'a> {
    Absent,
    /// Present, but the plain grey ramp: the frame is treated as having no table at all.
    Gray,
    Colours(&'a [u8]),
}

/// The logical screen descriptor.
struct Screen<'a> {
    width: usize,
    height: usize,
    /// Background index (0 without a global colour table).
    background: u8,
    /// Global colour table, unless it is absent or the grey ramp.
    global: Option<&'a [u8]>,
    /// Offset of the first block after the header.
    blocks: usize,
}

/// One image of the file with the settings of the extensions in front of it.
struct Frame<'a> {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    interlaced: bool,
    local: Local<'a>,
    transparent: Option<u8>,
    delay_ms: u32,
    /// Disposal method of the frame's control extensions, if one names a method other than 0.
    disposal: Option<u8>,
    code_bits: u8,
    /// Offset of the first pixel-data sub-block.
    data: usize,
}

/// `Some` for a table that gives colours, `None` for the grey ramp.
fn colours(table: &[u8]) -> Option<&[u8]> {
    let (entries, _) = table.as_chunks::<3>();
    let gray = entries
        .iter()
        .enumerate()
        .all(|(i, c)| c.iter().all(|&v| usize::from(v) == i));
    (!gray).then_some(table)
}

fn read_screen(bytes: &[u8]) -> Result<Screen<'_>, String> {
    if !bytes.starts_with(b"GIF87a") && !bytes.starts_with(b"GIF89a") {
        return Err("not a GIF file (no GIF87a / GIF89a signature)".into());
    }
    let head = bytes
        .get(..13)
        .ok_or("the file ends inside the GIF header")?;
    let flags = head[10];
    let mut screen = Screen {
        width: usize::from(u16::from_le_bytes([head[6], head[7]])),
        height: usize::from(u16::from_le_bytes([head[8], head[9]])),
        background: 0,
        global: None,
        blocks: 13,
    };
    if flags & 0x80 != 0 {
        let len = 3usize << ((flags & 7) + 1);
        let table = bytes
            .get(13..13 + len)
            .ok_or("the file ends inside the global colour table")?;
        screen.background = head[11];
        screen.global = colours(table);
        screen.blocks = 13 + len;
    }
    Ok(screen)
}

/// Reads one sub-block at `pos`: its bytes (`None` for the terminator or the end of the file;
/// cut short when the file ends inside it) and the offset after it.
fn sub_block(bytes: &[u8], pos: usize) -> (Option<&[u8]>, usize) {
    match bytes.get(pos) {
        None => (None, bytes.len()),
        Some(0) => (None, pos + 1),
        Some(&len) => {
            let end = (pos + 1 + usize::from(len)).min(bytes.len());
            (Some(&bytes[pos + 1..end]), end)
        }
    }
}

/// Skips sub-blocks up to and including their terminator (or to the end of the file).
fn skip_blocks(bytes: &[u8], mut pos: usize) -> usize {
    loop {
        match sub_block(bytes, pos) {
            (Some(block), after) if !block.is_empty() => pos = after,
            (_, after) => return after,
        }
    }
}

/// Walks the blocks from `pos` to the next image descriptor. `None` when the file has no
/// further image: at the trailer, at the end of the file, or where the file is cut inside a
/// block header.
///
/// The walk repeats the reference's reading of the structure, including two of its habits: an
/// extension other than a comment whose first sub-block is already the terminator makes it
/// read on as if more sub-blocks followed, and so does, in front of the first image only, a
/// looping extension that has no data sub-block. Bytes that start no block are stepped over.
fn next_frame(bytes: &[u8], mut pos: usize, first: bool) -> Option<Frame<'_>> {
    let mut transparent = None;
    let mut delay_ms = 0;
    let mut disposal = None;
    loop {
        let lead = *bytes.get(pos)?;
        pos += 1;
        match lead {
            b';' => return None,
            b'!' => {
                let label = *bytes.get(pos)?;
                let (block, after) = sub_block(bytes, pos + 1);
                pos = after;
                match (label, block) {
                    (0xF9, Some(control)) => {
                        let flags = *control.first()?;
                        if flags & 1 != 0 {
                            transparent = Some(*control.get(3)?);
                        }
                        let delay = control.get(1..3)?;
                        delay_ms = u32::from(u16::from_le_bytes([delay[0], delay[1]])) * 10;
                        let method = (flags >> 2) & 7;
                        if method != 0 {
                            disposal = Some(method);
                        }
                        pos = skip_blocks(bytes, pos);
                    }
                    (0xFE, text) => {
                        if text.is_some_and(|t| !t.is_empty()) {
                            pos = skip_blocks(bytes, pos);
                        }
                    }
                    (0xFF, Some(name)) if first => {
                        if name.starts_with(b"NETSCAPE2.0") {
                            pos = sub_block(bytes, pos).1;
                        }
                        pos = skip_blocks(bytes, pos);
                    }
                    _ => pos = skip_blocks(bytes, pos),
                }
            }
            b',' => {
                let d = bytes.get(pos..pos + 9)?;
                pos += 9;
                let word = |at: usize| usize::from(u16::from_le_bytes([d[at], d[at + 1]]));
                let flags = d[8];
                let mut local = Local::Absent;
                if flags & 0x80 != 0 {
                    let len = 3usize << ((flags & 7) + 1);
                    let table = bytes.get(pos..pos + len)?;
                    pos += len;
                    local = colours(table).map_or(Local::Gray, Local::Colours);
                }
                let code_bits = *bytes.get(pos)?;
                return Some(Frame {
                    x: word(0),
                    y: word(2),
                    width: word(4),
                    height: word(6),
                    interlaced: flags & 0x40 != 0,
                    local,
                    transparent,
                    delay_ms,
                    disposal,
                    code_bits,
                    data: pos + 1,
                });
            }
            _ => {}
        }
    }
}

/// The 256 colours a table gives: its entries, black past its end; grey levels without one.
fn palette(table: Option<&[u8]>) -> [[u8; 3]; 256] {
    let mut lut = [[0u8; 3]; 256];
    match table {
        Some(table) => {
            for (slot, c) in lut.iter_mut().zip(table.as_chunks::<3>().0) {
                *slot = *c;
            }
        }
        None => {
            for (i, slot) in lut.iter_mut().enumerate() {
                *slot = [i as u8; 3];
            }
        }
    }
    lut
}

/// The colour a "restore to background" fill takes for `index` once the canvas holds colours:
/// the table's entry, entry 0 for an index past its end, the grey level without a table.
fn fill_colour(table: Option<&[u8]>, index: u8) -> [u8; 3] {
    match table {
        Some(table) => {
            let at = usize::from(index) * 3;
            let at = if at + 3 > table.len() { 0 } else { at };
            [table[at], table[at + 1], table[at + 2]]
        }
        None => [index; 3],
    }
}

/// The LZW code source: bits of the sub-block chain that starts at `pos`, least significant
/// first.
struct Codes<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// End of the part of the file handed over so far.
    fed: usize,
    /// Bytes left in the open sub-block.
    block: usize,
    acc: u32,
    held: u32,
}

impl Codes<'_> {
    /// Hands over the next piece of the file; an error when none is left.
    fn feed(&mut self) -> Result<(), String> {
        if self.fed >= self.bytes.len() {
            return Err(CUT.into());
        }
        self.fed = self.fed.saturating_add(FEED).min(self.bytes.len());
        Ok(())
    }

    /// Opens the sub-block at `pos`, which has to lie whole inside the part handed over.
    fn open(&mut self) -> Result<(), String> {
        loop {
            if let Some(&len) = self.bytes[..self.fed].get(self.pos) {
                if self.pos + usize::from(len) < self.fed {
                    self.block = usize::from(len);
                    self.pos += 1;
                    return Ok(());
                }
            }
            self.feed()?;
        }
    }

    /// The next code of `width` bits (at most 13).
    #[inline]
    fn next(&mut self, width: u32) -> Result<u32, String> {
        while self.held < width {
            if self.block == 0 {
                self.open()?;
                continue;
            }
            // an open sub-block lies inside the file, so `pos` is in range
            self.acc |= u32::from(self.bytes[self.pos]) << self.held;
            self.pos += 1;
            self.block -= 1;
            self.held += 8;
        }
        let code = self.acc & ((1 << width) - 1);
        self.acc >>= width;
        self.held -= width;
        Ok(code)
    }
}

/// Decodes the LZW pixel data at `start` into `out`, one index per pixel in the order the file
/// stores them, until `out` is full.
///
/// The table follows the reference: a code is stored as its low 8 bits, the code size grows
/// when the entry just added has the number of the current code mask (so a 1-bit or 0-bit
/// code size never grows), and a full table stays as it is until a clear code.
fn unpack(bytes: &[u8], start: usize, code_bits: u8, out: &mut [u8]) -> Result<(), String> {
    if code_bits > MAX_CODE_BITS {
        return Err(format!("an LZW code size of {code_bits} bits is not valid"));
    }
    let clear = 1u32 << code_bits;
    let end = clear + 1;
    // entry `e` is the string of `prefix[e]` followed by `suffix[e]`; `span[e]` is its length
    let mut prefix = [0u16; TABLE];
    let mut suffix = [0u8; TABLE];
    let mut span = [0u16; TABLE];
    let mut codes = Codes {
        bytes,
        pos: start,
        fed: start.min(bytes.len()),
        block: 0,
        acc: 0,
        held: 0,
    };
    codes.feed()?;
    let first_width = u32::from(code_bits) + 1;
    let mut width = first_width;
    let mut next = clear + 2;
    // true until the first code after a clear code (or after the start) is taken
    let mut fresh = true;
    // the code before this one and the length of its string
    let (mut last, mut last_len) = (0u32, 0usize);
    let mut at = 0;
    while at < out.len() {
        let code = codes.next(width)?;
        if code == clear {
            next = clear + 2;
            width = first_width;
            fresh = true;
            continue;
        }
        if code == end {
            // the reference returns to its caller here, which goes on only with a new piece
            codes.feed()?;
            continue;
        }
        if fresh {
            if code > clear {
                return Err(DAMAGED.into());
            }
            out[at] = code as u8;
            at += 1;
            (last, last_len) = (code, 1);
            fresh = false;
            continue;
        }
        if code > next {
            return Err(DAMAGED.into());
        }
        // a code one past the table is the previous string plus its own first index
        let repeat = code == next;
        let source = if repeat { last } else { code };
        if source >= clear && source as usize >= TABLE {
            return Err(DAMAGED.into());
        }
        let len = if source < clear {
            1
        } else {
            usize::from(span[source as usize])
        };
        // the string goes to out[at..at + len], cut at the end of the frame; it is written
        // from its last index back to its first by following the prefixes
        let room = out.len() - at;
        let mut entry = source;
        let mut i = len;
        while entry >= clear {
            i -= 1;
            if i < room {
                out[at + i] = suffix[entry as usize];
            }
            entry = u32::from(prefix[entry as usize]);
        }
        let head = entry as u8;
        out[at] = head;
        if repeat && len < room {
            out[at + len] = head;
        }
        let total = len + usize::from(repeat);
        if (next as usize) < TABLE {
            prefix[next as usize] = last as u16;
            suffix[next as usize] = head;
            span[next as usize] = (last_len + 1) as u16;
            if next + 1 == 1 << width && width < u32::from(MAX_CODE_BITS) {
                width += 1;
            }
            next += 1;
        }
        (last, last_len) = (code, total);
        at += total.min(room);
    }
    Ok(())
}

/// Frame rows in the order the file stores them.
fn rows(height: usize, interlaced: bool) -> impl Iterator<Item = usize> {
    let passes = if interlaced {
        [(0, 8), (4, 8), (2, 4), (1, 2)]
    } else {
        [(0, 1), (height, 1), (height, 1), (height, 1)]
    };
    passes
        .into_iter()
        .flat_map(move |(first, step)| (first..height).step_by(step))
}

/// What happens to a frame's rectangle before the next frame is painted.
struct Undo {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    /// The colour to fill with; `None` puts back the pixels kept in `Animation::kept`.
    fill: Option<[u8; 3]>,
}

/// An animation in progress: the canvas after the frame decoded last.
struct Animation<'a> {
    bytes: &'a [u8],
    width: usize,
    height: usize,
    background: u8,
    global: Option<&'a [u8]>,
    /// The canvas, RGB8.
    rgb: Vec<u8>,
    /// Indices of the frame being decoded, in file order.
    indices: Vec<u8>,
    /// Canvas pixels saved for a "restore to previous" disposal.
    kept: Vec<u8>,
    undo: Option<Undo>,
    /// Disposal method in force (it carries over to frames that name none).
    disposal: u8,
    /// The first frame's colours.
    first_colours: [[u8; 3]; 256],
    /// Whether the first frame had a colour table (local or global).
    first_has_table: bool,
    /// The first frame's transparent index.
    first_transparent: Option<u8>,
    /// True while the canvas is a picture of the first frame's colours: frames are painted
    /// with `first_colours`, and a "restore to background" fill is one of them. It ends at
    /// the frame after the first one that has a colour table (`paletted`), which for most
    /// files is the second frame.
    indexed: bool,
    /// Whether the frame decoded last had a colour table (looked at while `indexed`).
    paletted: bool,
    /// Offset of the pixel data of the frame decoded last.
    data: usize,
    /// Number of the frame decoded last.
    frame: usize,
    delay_ms: u32,
}

impl<'a> Animation<'a> {
    /// Reads the header and decodes the first frame.
    fn start(bytes: &'a [u8], limits: &DecodeLimits) -> Result<Self, String> {
        let screen = read_screen(bytes)?;
        let frame = next_frame(bytes, screen.blocks, true).ok_or(NO_IMAGE)?;
        let width = screen.width.max(frame.x + frame.width);
        let height = screen.height.max(frame.y + frame.height);
        limits.check_canvas(width, height)?;
        if frame.width == 0 || frame.height == 0 {
            return Err("the first frame has zero size".into());
        }
        let len = width
            .checked_mul(height)
            .and_then(|px| px.checked_mul(3))
            .ok_or("the image is too large for this machine")?;
        let table = match frame.local {
            Local::Colours(table) => Some(table),
            _ => screen.global,
        };
        let first_colours = palette(table);
        // the canvas around the frame; skipped where the frame covers it or a grey is set
        let ground = first_colours[usize::from(frame.transparent.unwrap_or(0))];
        let mut rgb = vec![ground[0]; len];
        let covers = frame.width == width && frame.height == height;
        if !covers && (ground[1] != ground[0] || ground[2] != ground[0]) {
            rgb.as_chunks_mut::<3>().0.fill(ground);
        }
        let mut anim = Animation {
            bytes,
            width,
            height,
            background: screen.background,
            global: screen.global,
            rgb,
            indices: Vec::new(),
            kept: Vec::new(),
            undo: None,
            disposal: frame.disposal.unwrap_or(0),
            first_colours,
            first_has_table: table.is_some(),
            first_transparent: frame.transparent,
            indexed: true,
            paletted: match frame.local {
                Local::Colours(_) => true,
                Local::Gray => false,
                Local::Absent => screen.global.is_some(),
            },
            data: frame.data,
            frame: 0,
            delay_ms: frame.delay_ms,
        };
        if anim.disposal >= 2 {
            // the first frame has nothing before it to restore: both methods fill
            let index = if anim.disposal == 2 {
                Some(frame.transparent.unwrap_or(screen.background))
            } else {
                frame.transparent
            };
            anim.undo = index.map(|i| Undo {
                x: frame.x,
                y: frame.y,
                width: frame.width,
                height: frame.height,
                fill: Some(first_colours[usize::from(i)]),
            });
        }
        anim.paint(&frame, &first_colours, None)?;
        Ok(anim)
    }

    /// Decodes a frame's pixels and paints those that differ from `skip` through `colours`.
    fn paint(
        &mut self,
        frame: &Frame,
        colours: &[[u8; 3]; 256],
        skip: Option<u8>,
    ) -> Result<(), String> {
        // the frame lies inside the canvas, so this is never larger than the canvas
        let len = frame.width * frame.height;
        if self.indices.len() < len {
            self.indices.resize(len, 0);
        }
        unpack(
            self.bytes,
            frame.data,
            frame.code_bits,
            &mut self.indices[..len],
        )?;
        let stored = self.indices[..len].chunks_exact(frame.width);
        for (row, line) in rows(frame.height, frame.interlaced).zip(stored) {
            let at = ((frame.y + row) * self.width + frame.x) * 3;
            let (out, _) = self.rgb[at..at + frame.width * 3].as_chunks_mut::<3>();
            match skip {
                None => {
                    for (px, &index) in out.iter_mut().zip(line) {
                        *px = colours[usize::from(index)];
                    }
                }
                Some(skip) => {
                    for (px, &index) in out.iter_mut().zip(line) {
                        if index != skip {
                            *px = colours[usize::from(index)];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Applies the disposal of the frame decoded last.
    fn dispose(&mut self) {
        let Some(undo) = self.undo.take() else {
            return;
        };
        let line = undo.width * 3;
        for row in 0..undo.height {
            let at = ((undo.y + row) * self.width + undo.x) * 3;
            let out = &mut self.rgb[at..at + line];
            match undo.fill {
                Some(colour) => out.as_chunks_mut::<3>().0.fill(colour),
                None => out.copy_from_slice(&self.kept[row * line..(row + 1) * line]),
            }
        }
    }

    /// Decodes the next frame onto the canvas. `Ok(false)` at the end of the animation.
    fn advance(&mut self) -> Result<bool, String> {
        let after = skip_blocks(self.bytes, self.data);
        let Some(frame) = next_frame(self.bytes, after, false) else {
            return Ok(false);
        };
        let number = self.frame + 1;
        if frame.x + frame.width > self.width || frame.y + frame.height > self.height {
            return Err(format!(
                "frame {number} reaches outside the {}x{} canvas of the first frame",
                self.width, self.height
            ));
        }
        if frame.width == 0 || frame.height == 0 {
            return Err(format!("frame {number} has zero size"));
        }
        self.dispose();
        if let Some(method) = frame.disposal {
            self.disposal = method;
        }
        let table = match frame.local {
            Local::Colours(table) => Some(table),
            Local::Gray => None,
            Local::Absent => self.global,
        };
        if self.paletted {
            self.indexed = false;
        }
        if self.disposal >= 2 {
            let fill = if self.disposal > 2 {
                let line = frame.width * 3;
                self.kept.clear();
                for row in 0..frame.height {
                    let at = ((frame.y + row) * self.width + frame.x) * 3;
                    self.kept.extend_from_slice(&self.rgb[at..at + line]);
                }
                None
            } else if self.indexed {
                let index = self
                    .first_transparent
                    .or(frame.transparent)
                    .unwrap_or(self.background);
                Some(self.first_colours[usize::from(index)])
            } else {
                let index = frame.transparent.unwrap_or(self.background);
                Some(fill_colour(table, index))
            };
            self.undo = Some(Undo {
                x: frame.x,
                y: frame.y,
                width: frame.width,
                height: frame.height,
                fill,
            });
        }
        let colours = if self.indexed {
            self.first_colours
        } else {
            palette(table)
        };
        self.paint(&frame, &colours, frame.transparent)
            .map_err(|e| format!("frame {number}: {e}"))?;
        if self.indexed {
            self.paletted = table.is_some();
            if self.paletted && self.first_transparent.is_some() && !self.first_has_table {
                // a first frame with a transparent index and no table, then a frame with a
                // table: the reference fails to show that frame
                return Err(format!(
                    "frame {number} brings a colour table to an image that began without one"
                ));
            }
        }
        self.data = frame.data;
        self.frame = number;
        self.delay_ms = frame.delay_ms;
        Ok(true)
    }
}

/// Canvas size and every frame's delay, without decoding pixels (walks the block structure
/// only). A frame is counted when its image descriptor is whole; whether its pixel data
/// decodes is not looked at.
pub fn scan(bytes: &[u8]) -> Result<Scan, String> {
    let screen = read_screen(bytes)?;
    let first = next_frame(bytes, screen.blocks, true).ok_or(NO_IMAGE)?;
    let width = screen.width.max(first.x + first.width);
    let height = screen.height.max(first.y + first.height);
    if width == 0 || height == 0 {
        return Err("image has zero size".into());
    }
    if first.width == 0 || first.height == 0 {
        return Err("the first frame has zero size".into());
    }
    let mut delays_ms = vec![first.delay_ms];
    let mut data = first.data;
    while let Some(frame) = next_frame(bytes, skip_blocks(bytes, data), false) {
        delays_ms.push(frame.delay_ms);
        data = frame.data;
    }
    Ok(Scan {
        width,
        height,
        delays_ms,
        exif: None,
    })
}

/// Frame 0 as RGB8.
pub fn decode(bytes: &[u8], limits: &DecodeLimits) -> Result<Still, String> {
    let anim = Animation::start(bytes, limits)?;
    Ok(Still {
        width: anim.width,
        height: anim.height,
        rgb: anim.rgb,
        exif: None,
    })
}

/// Decodes the frames in order, composited as the reference library shows each one, and hands
/// every frame to `keep(frame_index, delay_ms, rgb_canvas)` (`rgb_canvas`: width*height*3
/// bytes, valid only during the call). Decoding stops early when `keep` returns false.
/// Returns the canvas (width, height).
pub fn decode_frames(
    bytes: &[u8],
    limits: &DecodeLimits,
    keep: &mut dyn FnMut(usize, u32, &[u8]) -> bool,
) -> Result<(usize, usize), String> {
    let mut anim = Animation::start(bytes, limits)?;
    while keep(anim.frame, anim.delay_ms, &anim.rgb) && anim.advance()? {}
    Ok((anim.width, anim.height))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: DecodeLimits = DecodeLimits {
        max_pixels: 1 << 24,
    };

    struct Fixture {
        name: &'static str,
        bytes: &'static [u8],
        width: usize,
        height: usize,
        /// (delay_ms, FNV-1a 64 of the RGB bytes) of every frame the reference shows.
        frames: &'static [(u32, u64)],
    }

    macro_rules! fixture {
        ($name:literal, $width:literal, $height:literal, [$($frame:expr),*]) => {
            Fixture {
                name: $name,
                bytes: include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/data/vision/gif/",
                    $name
                )),
                width: $width,
                height: $height,
                frames: &[$($frame),*],
            }
        };
    }

    /// The table `tests/data/vision/gif/make_fixtures.py` prints.
    const FIXTURES: &[Fixture] = &[
        fixture!(
            "anim_code_sizes.gif",
            17,
            11,
            [
                (0, 0x0D184850AF6BF863),
                (30, 0x88E1EB7B546A0EC1),
                (60, 0x18CB5B1E48722197),
                (90, 0x86701FA494702269),
                (120, 0x90C4274E384DF15B),
                (150, 0x54261C0E386C8A1C),
                (180, 0xEA87344B971FCDE3)
            ]
        ),
        fixture!(
            "anim_dispose.gif",
            28,
            20,
            [
                (50, 0x7E591D41B92EF584),
                (100, 0x7A1A77DF4C8724B1),
                (0, 0xFCDFDCA8A8837456),
                (70, 0xEE554FFFF330672D),
                (655350, 0x8BD0466B08C49A6C),
                (0, 0x23372C28512D1AA4),
                (30, 0x05E2E162C68F8DC6),
                (40, 0x233271B422782096)
            ]
        ),
        fixture!(
            "anim_first_previous.gif",
            14,
            10,
            [
                (10, 0xAB1680E65E921B88),
                (20, 0x0BB8061627EC4DDD),
                (30, 0x9ACF5D1D6D581AF4)
            ]
        ),
        fixture!(
            "anim_first_small.gif",
            20,
            16,
            [
                (20, 0x5D54D513B1CAF286),
                (40, 0xBDF362239719FF8C),
                (60, 0xDF7A560859D65590),
                (80, 0x983716E046076B50)
            ]
        ),
        fixture!(
            "anim_gray.gif",
            18,
            12,
            [
                (10, 0x65E097BC00E62775),
                (20, 0xB0A9D51A48C8B572),
                (30, 0xF2CF7B1DA986DF66),
                (40, 0xDF4893EAA9E98CC9),
                (50, 0x62125962E337F2C8),
                (60, 0x7C5AD9E5F12C0970),
                (70, 0x923AC836A0B4A1CC)
            ]
        ),
        fixture!(
            "anim_odd_code_sizes.gif",
            8,
            4,
            [
                (10, 0xB7F63090CC19B605),
                (20, 0x70B80F0A66330B25),
                (30, 0x5879B97B071224BD),
                (40, 0xBE102B7C5BE95E68)
            ]
        ),
        // the reference fails on frame 1 (AssertionError) of the next file
        fixture!(
            "anim_table_after_none.gif",
            6,
            4,
            [(10, 0xCB6FEC73FABB938A)]
        ),
        fixture!("still_1x1.gif", 1, 1, [(0, 0xF920341BE414D4AF)]),
        fixture!("still_1x9_interlaced.gif", 1, 9, [(0, 0x9ED9142A1751EBBE)]),
        fixture!("still_256.gif", 31, 23, [(0, 0xD1325EBDE6B0E6B5)]),
        fixture!("still_7x1.gif", 7, 1, [(0, 0xDF5DF97276F73DA3)]),
        fixture!("still_empty_blocks.gif", 20, 13, [(0, 0x2E702648E29B328D)]),
        fixture!("still_interlaced.gif", 29, 21, [(0, 0xD15DC20ABC0B556A)]),
        fixture!("still_local_gray.gif", 12, 8, [(0, 0xD58971F0D474D75F)]),
        fixture!("still_local_only.gif", 19, 14, [(0, 0xC1FC26BD7F709542)]),
        fixture!("still_no_table.gif", 16, 10, [(0, 0xF2DAB489DEBF0F09)]),
        fixture!(
            "still_offset_transparent.gif",
            24,
            16,
            [(0, 0xD329DE94219583AE)]
        ),
        fixture!(
            "still_outside_screen.gif",
            15,
            11,
            [(0, 0x6016F79D09E2EB08)]
        ),
        fixture!("still_sub_blocks.gif", 20, 13, [(0, 0xE75FDBC186C9885A)]),
        fixture!("still_table_full.gif", 66, 62, [(0, 0xA618C9C54960068B)]),
        fixture!("still_two_colours.gif", 40, 9, [(0, 0x1408D3C6F4DFB1A0)]),
    ];

    /// The fixture whose second frame the reference fails to show.
    const FAILS_LATE: &str = "anim_table_after_none.gif";

    fn fnv(data: &[u8]) -> u64 {
        data.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        })
    }

    /// The frames `decode_frames` hands over, as (delay, hash), with its result.
    type Seen = (Vec<(u32, u64)>, Result<(usize, usize), String>);

    fn frames(bytes: &[u8], limits: &DecodeLimits) -> Seen {
        let mut seen = Vec::new();
        let result = decode_frames(bytes, limits, &mut |index, delay, rgb| {
            assert_eq!(index, seen.len());
            seen.push((delay, fnv(rgb)));
            true
        });
        (seen, result)
    }

    fn fixture(name: &str) -> &'static Fixture {
        FIXTURES.iter().find(|f| f.name == name).unwrap()
    }

    /// `(value, bits)` codes packed least significant bit first.
    fn pack(codes: &[(u32, u32)]) -> Vec<u8> {
        let (mut acc, mut held, mut out) = (0u64, 0, Vec::new());
        for &(value, bits) in codes {
            acc |= u64::from(value) << held;
            held += bits;
            while held >= 8 {
                out.push(acc as u8);
                acc >>= 8;
                held -= 8;
            }
        }
        if held > 0 {
            out.push(acc as u8);
        }
        out
    }

    /// A header with a two-colour global table.
    fn head(width: u16, height: u16) -> Vec<u8> {
        let mut gif = b"GIF89a".to_vec();
        gif.extend(width.to_le_bytes());
        gif.extend(height.to_le_bytes());
        gif.extend([0x80, 0, 0, 10, 20, 30, 200, 100, 50]);
        gif
    }

    /// An image block whose pixel data is one sub-block.
    fn image(x: u16, y: u16, width: u16, height: u16, code_bits: u8, data: &[u8]) -> Vec<u8> {
        let mut block = vec![b','];
        for v in [x, y, width, height] {
            block.extend(v.to_le_bytes());
        }
        block.extend([0, code_bits, data.len() as u8]);
        block.extend(data);
        block.push(0);
        block
    }

    /// `count` pixels of index 1 as 2-bit codes (a 1-bit code size keeps its 2-bit codes).
    fn ones(count: usize) -> Vec<u8> {
        let mut codes = vec![(2, 2)];
        codes.resize(count + 1, (1, 2));
        pack(&codes)
    }

    #[test]
    fn fixtures_match_the_reference() {
        for f in FIXTURES {
            let still = decode(f.bytes, &LIMITS).unwrap_or_else(|e| panic!("{}: {e}", f.name));
            assert_eq!(
                (still.width, still.height),
                (f.width, f.height),
                "{}",
                f.name
            );
            assert_eq!(still.rgb.len(), f.width * f.height * 3, "{}", f.name);
            assert_eq!(fnv(&still.rgb), f.frames[0].1, "{}", f.name);
            assert_eq!(still.exif, None);

            let (seen, result) = frames(f.bytes, &LIMITS);
            assert_eq!(seen, f.frames, "{}", f.name);
            let scanned = scan(f.bytes).unwrap_or_else(|e| panic!("{}: {e}", f.name));
            assert_eq!(
                (scanned.width, scanned.height),
                (f.width, f.height),
                "{}",
                f.name
            );
            assert_eq!(scanned.exif, None);
            if f.name == FAILS_LATE {
                assert!(result.unwrap_err().contains("frame 1"));
                assert_eq!(scanned.delays_ms, [10, 20]);
            } else {
                assert_eq!(result, Ok((f.width, f.height)), "{}", f.name);
                let delays: Vec<u32> = f.frames.iter().map(|&(delay, _)| delay).collect();
                assert_eq!(scanned.delays_ms, delays, "{}", f.name);
            }
        }
    }

    #[test]
    fn decoding_stops_when_keep_says_so() {
        for name in ["anim_dispose.gif", FAILS_LATE] {
            let f = fixture(name);
            let mut calls = 0;
            let result = decode_frames(f.bytes, &LIMITS, &mut |index, delay, rgb| {
                assert_eq!((delay, fnv(rgb)), f.frames[index]);
                calls += 1;
                false
            });
            // the frame that fails is never reached
            assert_eq!(result, Ok((f.width, f.height)));
            assert_eq!(calls, 1);
        }
        let f = fixture("anim_dispose.gif");
        let mut calls = 0;
        let result = decode_frames(f.bytes, &LIMITS, &mut |index, _, _| {
            calls += 1;
            index < 4
        });
        assert_eq!((result, calls), (Ok((28, 20)), 5));
    }

    #[test]
    fn a_cut_file_gives_the_frames_before_the_cut_or_an_error() {
        for f in FIXTURES {
            for len in 0..f.bytes.len() {
                let cut = &f.bytes[..len];
                let (seen, result) = frames(cut, &LIMITS);
                assert!(seen.len() <= f.frames.len(), "{} cut at {len}", f.name);
                assert_eq!(seen, f.frames[..seen.len()], "{} cut at {len}", f.name);
                match decode(cut, &LIMITS) {
                    Ok(still) => assert_eq!(fnv(&still.rgb), f.frames[0].1),
                    Err(_) => assert!(seen.is_empty(), "{} cut at {len}", f.name),
                }
                if let (Ok(size), Ok(scanned)) = (result, scan(cut)) {
                    assert_eq!(size, (scanned.width, scanned.height));
                    let delays: Vec<u32> = seen.iter().map(|&(delay, _)| delay).collect();
                    assert_eq!(scanned.delays_ms, delays, "{} cut at {len}", f.name);
                }
            }
        }
        // cut inside the pixel data of the last frame: an error after the frames before it,
        // and the scan still counts the frame
        let f = fixture("anim_dispose.gif");
        let cut = &f.bytes[..f.bytes.len() - 40];
        let (seen, result) = frames(cut, &LIMITS);
        assert_eq!(seen, f.frames[..7]);
        assert!(result
            .unwrap_err()
            .contains("frame 7: the file ends inside the pixel data"));
        assert_eq!(scan(cut).unwrap().delays_ms.len(), 8);
        // cut inside the control extension of the last frame: the animation is one frame shorter
        let f = fixture("anim_first_previous.gif");
        let last = f.bytes.len() - 1 - image(0, 0, 0, 0, 0, &[0; 13]).len() - 4;
        let (seen, result) = frames(&f.bytes[..last], &LIMITS);
        assert_eq!((seen.len(), result), (2, Ok((14, 10))));
        assert_eq!(scan(&f.bytes[..last]).unwrap().delays_ms, [10, 20]);
    }

    #[test]
    fn changed_bytes_never_panic() {
        let limits = DecodeLimits {
            max_pixels: 1 << 18,
        };
        for f in FIXTURES {
            let mut bytes = f.bytes.to_vec();
            for at in 0..bytes.len() {
                // every bit near the front and in small files, one bit of every third byte
                // elsewhere
                let bits = if at < 120 || bytes.len() < 400 {
                    0..8
                } else if at % 3 == 0 {
                    at % 8..at % 8 + 1
                } else {
                    0..0
                };
                for bit in bits {
                    bytes[at] ^= 1 << bit;
                    let (seen, result) = frames(&bytes, &limits);
                    if let (Ok(size), Ok(scanned)) = (result, scan(&bytes)) {
                        assert_eq!(size, (scanned.width, scanned.height));
                        let delays: Vec<u32> = seen.iter().map(|&(delay, _)| delay).collect();
                        assert_eq!(scanned.delays_ms, delays, "{} byte {at} bit {bit}", f.name);
                    }
                    bytes[at] ^= 1 << bit;
                }
            }
        }
    }

    #[test]
    fn what_follows_the_trailer_is_ignored_and_the_trailer_may_be_missing() {
        let f = fixture("anim_dispose.gif");
        let mut longer = f.bytes.to_vec();
        longer.extend(b",junk after the trailer \x21\xf9\x04");
        assert_eq!(frames(&longer, &LIMITS).0, f.frames);
        assert_eq!(scan(&longer).unwrap().delays_ms.len(), 8);
        let shorter = &f.bytes[..f.bytes.len() - 1];
        assert_eq!(frames(shorter, &LIMITS), (f.frames.to_vec(), Ok((28, 20))));
    }

    #[test]
    fn bad_files_are_refused_with_a_reason() {
        assert!(scan(b"").unwrap_err().contains("not a GIF"));
        assert!(decode(b"GIF88a\x01\0\x01\0\0\0\0", &LIMITS)
            .unwrap_err()
            .contains("not a GIF"));
        assert!(scan(b"GIF89a\x01\0").unwrap_err().contains("header"));
        assert!(scan(&head(4, 4)[..17])
            .unwrap_err()
            .contains("colour table"));
        // a header and a trailer, but no image
        let mut empty = head(4, 4);
        empty.push(b';');
        assert!(scan(&empty).unwrap_err().contains("no image"));
        assert!(decode(&empty, &LIMITS).unwrap_err().contains("no image"));
        assert!(decode_frames(&empty, &LIMITS, &mut |_, _, _| true)
            .unwrap_err()
            .contains("no image"));
        // a first frame without pixels
        let mut flat = head(4, 4);
        flat.extend(image(0, 0, 4, 0, 1, &ones(0)));
        assert!(scan(&flat).unwrap_err().contains("zero size"));
        assert!(decode(&flat, &LIMITS).unwrap_err().contains("zero size"));
        // no screen and no frame area at all
        let mut nothing = head(0, 0);
        nothing.extend(image(0, 0, 0, 0, 1, &ones(0)));
        assert!(scan(&nothing).is_err());
        assert!(decode(&nothing, &LIMITS).is_err());
        // a code size the format does not have
        let mut wide = head(4, 4);
        wide.extend(image(0, 0, 4, 4, 13, &ones(16)));
        assert!(decode(&wide, &LIMITS).unwrap_err().contains("13 bits"));
        // a code past the table, straight after the clear code and later
        for codes in [[(4, 3), (6, 3), (0, 3)], [(4, 3), (1, 3), (7, 3)]] {
            let mut damaged = head(4, 4);
            damaged.extend(image(0, 0, 4, 4, 2, &pack(&codes)));
            assert!(decode(&damaged, &LIMITS).unwrap_err().contains("damaged"));
        }
        // pixel data that stops before the frame is full
        let mut short = head(4, 4);
        short.extend(image(0, 0, 4, 4, 1, &ones(11)));
        short.push(b';');
        assert!(decode(&short, &LIMITS)
            .unwrap_err()
            .contains("ends inside the pixel data"));
        assert_eq!(scan(&short).unwrap().delays_ms, [0]);
    }

    #[test]
    fn the_canvas_limit_is_checked_before_decoding() {
        let f = fixture("still_256.gif");
        let exact = DecodeLimits {
            max_pixels: 31 * 23,
        };
        assert!(decode(f.bytes, &exact).is_ok());
        let tight = DecodeLimits {
            max_pixels: 31 * 23 - 1,
        };
        assert!(decode(f.bytes, &tight).is_err());
        assert!(decode_frames(f.bytes, &tight, &mut |_, _, _| panic!("no frame is due")).is_err());
        // the canvas counts, not the screen: this frame reaches outside an 8x8 screen
        let grown = fixture("still_outside_screen.gif");
        let screen_only = DecodeLimits { max_pixels: 64 };
        assert!(decode(grown.bytes, &screen_only).is_err());
        // a huge screen with a tiny frame is refused before anything is allocated
        let mut huge = head(65535, 65535);
        huge.extend(image(0, 0, 1, 1, 1, &ones(1)));
        assert!(decode(&huge, &LIMITS).is_err());
        assert_eq!(scan(&huge).map(|s| (s.width, s.height)), Ok((65535, 65535)));
    }

    #[test]
    fn a_later_frame_outside_the_canvas_is_refused() {
        let mut gif = head(4, 4);
        gif.extend(image(0, 0, 4, 4, 1, &ones(16)));
        gif.extend(image(2, 2, 4, 4, 1, &ones(16)));
        gif.push(b';');
        // the first frame alone is an image
        assert_eq!(
            decode(&gif, &LIMITS).unwrap().rgb,
            [200, 100, 50].repeat(16)
        );
        assert_eq!(scan(&gif).unwrap().delays_ms, [0, 0]);
        let (seen, result) = frames(&gif, &LIMITS);
        assert_eq!(seen.len(), 1);
        assert!(result
            .unwrap_err()
            .contains("frame 1 reaches outside the 4x4 canvas"));
        // and so is one without pixels
        let mut gif = head(4, 4);
        gif.extend(image(0, 0, 4, 4, 1, &ones(16)));
        gif.extend(image(1, 1, 0, 2, 1, &ones(0)));
        let (seen, result) = frames(&gif, &LIMITS);
        assert_eq!(seen.len(), 1);
        assert!(result.unwrap_err().contains("frame 1 has zero size"));
    }

    #[test]
    fn extensions_are_read_as_the_reference_reads_them() {
        let frame = image(0, 0, 2, 2, 1, &ones(4));
        let with = |extension: &[u8]| {
            let mut gif = head(2, 2);
            gif.extend(extension);
            gif.extend(&frame);
            gif.extend(&frame);
            gif.push(b';');
            gif
        };
        // delay 0x0102 hundredths; of two control extensions the last one gives the delay
        assert_eq!(
            scan(&with(b"\x21\xf9\x04\x00\x02\x01\x00\x00"))
                .unwrap()
                .delays_ms,
            [2580, 0]
        );
        let two = b"\x21\xf9\x04\x00\x02\x01\x00\x00\x21\xf9\x04\x00\x07\x00\x00\x00";
        assert_eq!(scan(&with(two)).unwrap().delays_ms, [70, 0]);
        // an empty comment and a stray byte change nothing
        assert_eq!(
            scan(&with(b"\x21\xfe\x00\x00\x99")).unwrap().delays_ms,
            [0, 0]
        );
        // an empty control extension makes the reference read the image as sub-blocks
        assert!(scan(&with(b"\x21\xf9\x00")).is_err());
        assert!(decode(&with(b"\x21\xf9\x00"), &LIMITS).is_err());
        // a looping extension without data does the same in front of the first image only
        assert!(scan(&with(b"\x21\xff\x0bNETSCAPE2.0\x00")).is_err());
        let mut later = head(2, 2);
        later.extend(&frame);
        later.extend(b"\x21\xff\x0bNETSCAPE2.0\x00");
        later.extend(&frame);
        assert_eq!(scan(&later).unwrap().delays_ms, [0, 0]);
        assert_eq!(frames(&later, &LIMITS).0.len(), 2);
    }

    /// A 4x4 frame whose data stops at an end code after 8 pixels, followed by `tail`
    /// one-byte sub-blocks of zero bits (`feed_case` in `make_fixtures.py`).
    fn early_end(tail: usize) -> Vec<u8> {
        let mut codes = vec![(2, 2)];
        codes.extend([1, 0, 1, 1, 0, 1, 0, 0].map(|v| (v, 2)));
        codes.push((3, 2));
        let mut gif = head(4, 4);
        gif.extend(image(0, 0, 4, 4, 1, &pack(&codes)));
        gif.extend([1, 0].repeat(tail));
        gif
    }

    #[test]
    fn an_end_code_goes_on_only_with_another_piece_of_the_file() {
        // more than one piece: the reference reads on through the bytes after the image
        let long = decode(&early_end(40000), &LIMITS).unwrap();
        assert_eq!(fnv(&long.rgb), 0x9600_03E1_0E2D_5785);
        // one piece: the reference reports a cut file
        assert!(decode(&early_end(20000), &LIMITS)
            .unwrap_err()
            .contains("ends inside"));
    }

    /// Compares every `<name>.gif` in the directory `GAUDIUM_IMAGE_REF_DIR` with the
    /// reference's raw RGB dumps beside it: `<name>.rgb` for the first frame, and
    /// `<name>.k<frame>.rgb` for the frames of an animation.
    #[test]
    #[ignore = "needs GAUDIUM_IMAGE_REF_DIR"]
    fn reference_directory() {
        let dir = std::env::var("GAUDIUM_IMAGE_REF_DIR").expect("GAUDIUM_IMAGE_REF_DIR is not set");
        let limits = DecodeLimits {
            max_pixels: 1 << 28,
        };
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gif")))
            .collect();
        paths.sort();
        let (mut stills, mut animations, mut frame_count) = (0, 0, 0);
        let mut wrong = Vec::new();
        for path in paths {
            let bytes = std::fs::read(&path).unwrap();
            let name = path.display().to_string();
            if let Ok(want) = std::fs::read(path.with_extension("rgb")) {
                stills += 1;
                match decode(&bytes, &limits) {
                    Ok(still) if still.rgb == want => {}
                    Ok(_) => wrong.push(format!("{name}: the first frame differs")),
                    Err(e) => wrong.push(format!("{name}: {e}")),
                }
            }
            if !path.with_extension("k0.rgb").exists() {
                continue;
            }
            animations += 1;
            let mut checked = 0;
            let result = decode_frames(&bytes, &limits, &mut |index, _, rgb| match std::fs::read(
                path.with_extension(format!("k{index}.rgb")),
            ) {
                Ok(want) if want == rgb => {
                    checked += 1;
                    true
                }
                Ok(_) => {
                    wrong.push(format!("{name}: frame {index} differs"));
                    false
                }
                Err(_) => {
                    wrong.push(format!("{name}: frame {index} has no dump"));
                    false
                }
            });
            frame_count += checked;
            if let Err(e) = result {
                wrong.push(format!("{name}: {e}"));
            } else if path.with_extension(format!("k{checked}.rgb")).exists() {
                wrong.push(format!("{name}: frame {checked} is missing"));
            }
        }
        println!(
            "{stills} first frames and {frame_count} frames of {animations} animations compared"
        );
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
