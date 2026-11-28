//! Fonts as the text reader needs them (ISO 32000-1, section 9.5 to 9.10): how a string's
//! bytes split into character codes, the Unicode text of each code, and its advance width.
//!
//! The text of a code comes from the font's ToUnicode CMap when it maps the code, else from the
//! font's encoding: the base encoding (StandardEncoding, WinAnsiEncoding, MacRomanEncoding, or
//! the built-in encoding of the Symbol font) with its /Differences, whose glyph names are looked up in a table of common names and the
//! `uniXXXX` / `uXXXX` forms. Composite (Type0) fonts take their code lengths from the
//! codespace ranges of their CMap (two bytes for Identity-H / Identity-V); their text needs a
//! ToUnicode CMap, except for the predefined Unicode (UCS2 / UTF16) CMaps.
//! The encoding tables follow Annex D of the specification.

use std::collections::{BTreeMap, HashMap};

use super::lex::{Dict, Lexer, Obj, Token};

/// The Unicode text of codes 0x20..=0x7E in StandardEncoding: ASCII, except the two quotes.
fn standard(code: u8) -> Option<char> {
    let u: u32 = match code {
        0x27 => 0x2019,
        0x60 => 0x2018,
        0x20..=0x7E => u32::from(code),
        0xA1 => 0xA1,
        0xA2 => 0xA2,
        0xA3 => 0xA3,
        0xA4 => 0x2044,
        0xA5 => 0xA5,
        0xA6 => 0x192,
        0xA7 => 0xA7,
        0xA8 => 0xA4,
        0xA9 => 0x27,
        0xAA => 0x201C,
        0xAB => 0xAB,
        0xAC => 0x2039,
        0xAD => 0x203A,
        0xAE => 0xFB01,
        0xAF => 0xFB02,
        0xB1 => 0x2013,
        0xB2 => 0x2020,
        0xB3 => 0x2021,
        0xB4 => 0xB7,
        0xB6 => 0xB6,
        0xB7 => 0x2022,
        0xB8 => 0x201A,
        0xB9 => 0x201E,
        0xBA => 0x201D,
        0xBB => 0xBB,
        0xBC => 0x2026,
        0xBD => 0x2030,
        0xBF => 0xBF,
        0xC1 => 0x60,
        0xC2 => 0xB4,
        0xC3 => 0x2C6,
        0xC4 => 0x2DC,
        0xC5 => 0xAF,
        0xC6 => 0x2D8,
        0xC7 => 0x2D9,
        0xC8 => 0xA8,
        0xCA => 0x2DA,
        0xCB => 0xB8,
        0xCD => 0x2DD,
        0xCE => 0x2DB,
        0xCF => 0x2C7,
        0xD0 => 0x2014,
        0xE1 => 0xC6,
        0xE3 => 0xAA,
        0xE8 => 0x141,
        0xE9 => 0xD8,
        0xEA => 0x152,
        0xEB => 0xBA,
        0xF1 => 0xE6,
        0xF5 => 0x131,
        0xF8 => 0x142,
        0xF9 => 0xF8,
        0xFA => 0x153,
        0xFB => 0xDF,
        _ => return None,
    };
    char::from_u32(u)
}

/// WinAnsiEncoding: Latin-1 with the Windows characters in 0x80..=0x9F.
fn win_ansi(code: u8) -> Option<char> {
    const HIGH: [u16; 32] = [
        0x20AC, 0, 0x201A, 0x192, 0x201E, 0x2026, 0x2020, 0x2021, 0x2C6, 0x2030, 0x160, 0x2039,
        0x152, 0, 0x17D, 0, 0, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x2DC,
        0x2122, 0x161, 0x203A, 0x153, 0, 0x17E, 0x178,
    ];
    let u = match code {
        0x20..=0x7E | 0xA0..=0xFF => u32::from(code),
        0x80..=0x9F => u32::from(HIGH[usize::from(code - 0x80)]),
        _ => 0,
    };
    (u != 0).then(|| char::from_u32(u)).flatten()
}

/// MacRomanEncoding, codes 0x80..=0xFF.
/// The Unicode text of the codes of the Symbol font's built-in encoding (Annex D.5).
fn symbol(code: u8) -> Option<char> {
    // 0x20..=0x7E (0: no character; 0x60 is the radical's extender, not text)
    const LOW: [u16; 95] = [
        0x20, 0x21, 0x2200, 0x23, 0x2203, 0x25, 0x26, 0x220B, 0x28, 0x29, 0x2217, 0x2B, 0x2C,
        0x2212, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,
        0x3C, 0x3D, 0x3E, 0x3F, 0x2245, 0x391, 0x392, 0x3A7, 0x394, 0x395, 0x3A6, 0x393, 0x397,
        0x399, 0x3D1, 0x39A, 0x39B, 0x39C, 0x39D, 0x39F, 0x3A0, 0x398, 0x3A1, 0x3A3, 0x3A4, 0x3A5,
        0x3C2, 0x3A9, 0x39E, 0x3A8, 0x396, 0x5B, 0x2234, 0x5D, 0x22A5, 0x5F, 0, 0x3B1, 0x3B2,
        0x3C7, 0x3B4, 0x3B5, 0x3C6, 0x3B3, 0x3B7, 0x3B9, 0x3D5, 0x3BA, 0x3BB, 0x3BC, 0x3BD, 0x3BF,
        0x3C0, 0x3B8, 0x3C1, 0x3C3, 0x3C4, 0x3C5, 0x3D6, 0x3C9, 0x3BE, 0x3C8, 0x3B6, 0x7B, 0x7C,
        0x7D, 0x223C,
    ];
    // 0xA0..=0xFE (0: no character)
    const HIGH: [u16; 95] = [
        0x20AC, 0x3D2, 0x2032, 0x2264, 0x2044, 0x221E, 0x192, 0x2663, 0x2666, 0x2665, 0x2660,
        0x2194, 0x2190, 0x2191, 0x2192, 0x2193, 0xB0, 0xB1, 0x2033, 0x2265, 0xD7, 0x221D, 0x2202,
        0x2022, 0xF7, 0x2260, 0x2261, 0x2248, 0x2026, 0x23D0, 0x23AF, 0x21B5, 0x2135, 0x2111,
        0x211C, 0x2118, 0x2297, 0x2295, 0x2205, 0x2229, 0x222A, 0x2283, 0x2287, 0x2284, 0x2282,
        0x2286, 0x2208, 0x2209, 0x2220, 0x2207, 0xAE, 0xA9, 0x2122, 0x220F, 0x221A, 0x22C5, 0xAC,
        0x2227, 0x2228, 0x21D4, 0x21D0, 0x21D1, 0x21D2, 0x21D3, 0x25CA, 0x2329, 0xAE, 0xA9, 0x2122,
        0x2211, 0x239B, 0x239C, 0x239D, 0x23A1, 0x23A2, 0x23A3, 0x23A7, 0x23A8, 0x23A9, 0x23AA, 0,
        0x232A, 0x222B, 0x2320, 0x23AE, 0x2321, 0x239E, 0x239F, 0x23A0, 0x23A4, 0x23A5, 0x23A6,
        0x23AB, 0x23AC, 0x23AD,
    ];
    let u = match code {
        0x20..=0x7E => LOW[usize::from(code - 0x20)],
        0xA0..=0xFE => HIGH[usize::from(code - 0xA0)],
        _ => 0,
    };
    (u != 0).then(|| char::from_u32(u32::from(u))).flatten()
}

fn mac_roman(code: u8) -> Option<char> {
    const HIGH: [u16; 128] = [
        0xC4, 0xC5, 0xC7, 0xC9, 0xD1, 0xD6, 0xDC, 0xE1, 0xE0, 0xE2, 0xE4, 0xE3, 0xE5, 0xE7, 0xE9,
        0xE8, 0xEA, 0xEB, 0xED, 0xEC, 0xEE, 0xEF, 0xF1, 0xF3, 0xF2, 0xF4, 0xF6, 0xF5, 0xFA, 0xF9,
        0xFB, 0xFC, 0x2020, 0xB0, 0xA2, 0xA3, 0xA7, 0x2022, 0xB6, 0xDF, 0xAE, 0xA9, 0x2122, 0xB4,
        0xA8, 0x2260, 0xC6, 0xD8, 0x221E, 0xB1, 0x2264, 0x2265, 0xA5, 0xB5, 0x2202, 0x2211, 0x220F,
        0x3C0, 0x222B, 0xAA, 0xBA, 0x3A9, 0xE6, 0xF8, 0xBF, 0xA1, 0xAC, 0x221A, 0x192, 0x2248,
        0x2206, 0xAB, 0xBB, 0x2026, 0xA0, 0xC0, 0xC3, 0xD5, 0x152, 0x153, 0x2013, 0x2014, 0x201C,
        0x201D, 0x2018, 0x2019, 0xF7, 0x25CA, 0xFF, 0x178, 0x2044, 0xA4, 0x2039, 0x203A, 0xFB01,
        0xFB02, 0x2021, 0xB7, 0x201A, 0x201E, 0x2030, 0xC2, 0xCA, 0xC1, 0xCB, 0xC8, 0xCD, 0xCE,
        0xCF, 0xCC, 0xD3, 0xD4, 0, 0xD2, 0xDA, 0xDB, 0xD9, 0x131, 0x2C6, 0x2DC, 0xAF, 0x2D8, 0x2D9,
        0x2DA, 0xB8, 0x2DD, 0x2DB, 0x2C7,
    ];
    let u = match code {
        0x20..=0x7E => u32::from(code),
        0x80..=0xFF => u32::from(HIGH[usize::from(code - 0x80)]),
        _ => 0,
    };
    (u != 0).then(|| char::from_u32(u)).flatten()
}

/// Glyph names of printable ASCII, 0x20..=0x7E.
const ASCII_NAMES: [&str; 95] = [
    "space",
    "exclam",
    "quotedbl",
    "numbersign",
    "dollar",
    "percent",
    "ampersand",
    "quotesingle",
    "parenleft",
    "parenright",
    "asterisk",
    "plus",
    "comma",
    "hyphen",
    "period",
    "slash",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "colon",
    "semicolon",
    "less",
    "equal",
    "greater",
    "question",
    "at",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "bracketleft",
    "backslash",
    "bracketright",
    "asciicircum",
    "underscore",
    "grave",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "braceleft",
    "bar",
    "braceright",
    "asciitilde",
];

/// Glyph names of Latin-1, 0xA1..=0xFF.
const LATIN1_NAMES: [&str; 95] = [
    "exclamdown",
    "cent",
    "sterling",
    "currency",
    "yen",
    "brokenbar",
    "section",
    "dieresis",
    "copyright",
    "ordfeminine",
    "guillemotleft",
    "logicalnot",
    "sfthyphen",
    "registered",
    "macron",
    "degree",
    "plusminus",
    "twosuperior",
    "threesuperior",
    "acute",
    "mu",
    "paragraph",
    "periodcentered",
    "cedilla",
    "onesuperior",
    "ordmasculine",
    "guillemotright",
    "onequarter",
    "onehalf",
    "threequarters",
    "questiondown",
    "Agrave",
    "Aacute",
    "Acircumflex",
    "Atilde",
    "Adieresis",
    "Aring",
    "AE",
    "Ccedilla",
    "Egrave",
    "Eacute",
    "Ecircumflex",
    "Edieresis",
    "Igrave",
    "Iacute",
    "Icircumflex",
    "Idieresis",
    "Eth",
    "Ntilde",
    "Ograve",
    "Oacute",
    "Ocircumflex",
    "Otilde",
    "Odieresis",
    "multiply",
    "Oslash",
    "Ugrave",
    "Uacute",
    "Ucircumflex",
    "Udieresis",
    "Yacute",
    "Thorn",
    "germandbls",
    "agrave",
    "aacute",
    "acircumflex",
    "atilde",
    "adieresis",
    "aring",
    "ae",
    "ccedilla",
    "egrave",
    "eacute",
    "ecircumflex",
    "edieresis",
    "igrave",
    "iacute",
    "icircumflex",
    "idieresis",
    "eth",
    "ntilde",
    "ograve",
    "oacute",
    "ocircumflex",
    "otilde",
    "odieresis",
    "divide",
    "oslash",
    "ugrave",
    "uacute",
    "ucircumflex",
    "udieresis",
    "yacute",
    "thorn",
    "ydieresis",
];

/// Other common glyph names.
const OTHER_NAMES: [(&str, u32); 55] = [
    ("Euro", 0x20AC),
    ("quotesinglbase", 0x201A),
    ("florin", 0x192),
    ("quotedblbase", 0x201E),
    ("ellipsis", 0x2026),
    ("dagger", 0x2020),
    ("daggerdbl", 0x2021),
    ("circumflex", 0x2C6),
    ("perthousand", 0x2030),
    ("Scaron", 0x160),
    ("guilsinglleft", 0x2039),
    ("OE", 0x152),
    ("Zcaron", 0x17D),
    ("quoteleft", 0x2018),
    ("quoteright", 0x2019),
    ("quotedblleft", 0x201C),
    ("quotedblright", 0x201D),
    ("bullet", 0x2022),
    ("endash", 0x2013),
    ("emdash", 0x2014),
    ("tilde", 0x2DC),
    ("trademark", 0x2122),
    ("scaron", 0x161),
    ("guilsinglright", 0x203A),
    ("oe", 0x153),
    ("zcaron", 0x17E),
    ("Ydieresis", 0x178),
    ("fi", 0xFB01),
    ("fl", 0xFB02),
    ("ff", 0xFB00),
    ("ffi", 0xFB03),
    ("ffl", 0xFB04),
    ("dotlessi", 0x131),
    ("fraction", 0x2044),
    ("minus", 0x2212),
    ("Lslash", 0x141),
    ("lslash", 0x142),
    ("ring", 0x2DA),
    ("breve", 0x2D8),
    ("dotaccent", 0x2D9),
    ("hungarumlaut", 0x2DD),
    ("ogonek", 0x2DB),
    ("caron", 0x2C7),
    ("nbspace", 0xA0),
    ("nonbreakingspace", 0xA0),
    ("hyphensoft", 0xAD),
    ("Delta", 0x2206),
    ("Omega", 0x2126),
    ("pi", 0x3C0),
    ("notequal", 0x2260),
    ("infinity", 0x221E),
    ("lessequal", 0x2264),
    ("greaterequal", 0x2265),
    ("radical", 0x221A),
    ("approxequal", 0x2248),
];

fn name_table() -> &'static HashMap<&'static str, u32> {
    static TABLE: std::sync::OnceLock<HashMap<&'static str, u32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut m = HashMap::new();
        for (i, n) in ASCII_NAMES.iter().enumerate() {
            m.insert(*n, 0x20 + i as u32);
        }
        for (i, n) in LATIN1_NAMES.iter().enumerate() {
            m.insert(*n, 0xA1 + i as u32);
        }
        for (n, u) in OTHER_NAMES {
            m.insert(n, u);
        }
        m
    })
}

/// The text of a glyph name; `None` for a name that says nothing (`g12`, `cid42`, `.notdef`).
pub(crate) fn glyph_text(name: &[u8]) -> Option<String> {
    let name = std::str::from_utf8(name).ok()?;
    // a suffix after a period is a variant of the same character (`a.sc`, `one.oldstyle`)
    let base = name.split('.').next().unwrap_or("");
    if base.is_empty() {
        return None;
    }
    if base.contains('_') {
        let parts: Option<String> = base.split('_').map(|p| glyph_text(p.as_bytes())).collect();
        return parts;
    }
    if let Some(&u) = name_table().get(base) {
        return char::from_u32(u).map(String::from);
    }
    let hex = |s: &str| u32::from_str_radix(s, 16).ok().and_then(char::from_u32);
    if let Some(rest) = base.strip_prefix("uni") {
        if !rest.is_empty() && rest.len() % 4 == 0 && rest.bytes().all(|c| c.is_ascii_hexdigit()) {
            let units: Vec<u16> = rest
                .as_bytes()
                .chunks(4)
                .filter_map(|c| u16::from_str_radix(std::str::from_utf8(c).ok()?, 16).ok())
                .collect();
            return Some(String::from_utf16_lossy(&units));
        }
    }
    if let Some(rest) = base.strip_prefix('u') {
        if (4..=6).contains(&rest.len()) && rest.bytes().all(|c| c.is_ascii_hexdigit()) {
            return hex(rest).map(String::from);
        }
    }
    None
}

/// UTF-16BE bytes as text (unpaired surrogates become U+FFFD).
pub(crate) fn utf16be(b: &[u8]) -> String {
    let units: Vec<u16> = b
        .chunks(2)
        .map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]))
        .collect();
    String::from_utf16_lossy(&units)
}

/// Most codespace ranges kept from one CMap (the specification's own limit is 100).
const MAX_SPACE_RANGES: usize = 256;

/// Longest text one code may map to, in UTF-16 units (longer mappings are not used).
const MAX_DST_UNITS: usize = 64;

/// A CMap: code lengths (codespace ranges) and, for a ToUnicode CMap, the text of codes.
#[derive(Default, Debug)]
pub(crate) struct CMap {
    /// (bytes, low, high)
    space: Vec<(usize, u32, u32)>,
    single: HashMap<(usize, u32), String>,
    /// keyed by (bytes, low): (high, destination UTF-16 units)
    ranges: BTreeMap<(usize, u32), (u32, Vec<u16>)>,
}

fn code_of(b: &[u8]) -> Option<(usize, u32)> {
    if b.is_empty() || b.len() > 4 {
        return None;
    }
    Some((b.len(), b.iter().fold(0u32, |a, &x| a << 8 | u32::from(x))))
}

impl CMap {
    /// Parses a CMap stream's decoded bytes; the operands held at a time may take the room
    /// `mem` leaves. Unknown parts are skipped.
    pub fn parse(data: &[u8], mem: super::Budget) -> CMap {
        let mut m = CMap::default();
        let mut lx = Lexer::new(data, 0, mem);
        let mut args: Vec<Obj> = Vec::new();
        #[derive(PartialEq)]
        enum Mode {
            None,
            Space,
            Char,
            Range,
        }
        let mut mode = Mode::None;
        while let Ok(Some(t)) = lx.token(0, true) {
            match t {
                Token::Obj(o) => {
                    args.push(o);
                    match mode {
                        Mode::Space if args.len() == 2 => {
                            if let (Obj::Str(lo), Obj::Str(hi)) = (&args[0], &args[1]) {
                                if let (Some((n, l)), Some((n2, h))) = (code_of(lo), code_of(hi)) {
                                    if n == n2 && l <= h && m.space.len() < MAX_SPACE_RANGES {
                                        m.space.push((n, l, h));
                                    }
                                }
                            }
                            args.clear();
                            lx.restart();
                        }
                        Mode::Char if args.len() == 2 => {
                            if let (Obj::Str(src), dst) = (&args[0], &args[1]) {
                                if let Some(k) = code_of(src) {
                                    let text = match dst {
                                        Obj::Str(d) if d.len() <= 2 * MAX_DST_UNITS => {
                                            Some(utf16be(d))
                                        }
                                        Obj::Name(n) => glyph_text(n),
                                        _ => None,
                                    };
                                    if let Some(t) = text {
                                        m.single.insert(k, t);
                                    }
                                }
                            }
                            args.clear();
                            lx.restart();
                        }
                        Mode::Range if args.len() == 3 => {
                            m.add_range(&args[0], &args[1], &args[2]);
                            args.clear();
                            lx.restart();
                        }
                        _ => {
                            if mode == Mode::None && args.len() > 16 {
                                args.remove(0);
                            }
                        }
                    }
                }
                Token::Op(op) => {
                    args.clear();
                    lx.restart();
                    mode = match op.as_slice() {
                        b"begincodespacerange" => Mode::Space,
                        b"beginbfchar" => Mode::Char,
                        b"beginbfrange" => Mode::Range,
                        _ => Mode::None,
                    };
                }
            }
        }
        m
    }

    fn add_range(&mut self, lo: &Obj, hi: &Obj, dst: &Obj) {
        let (Obj::Str(lo), Obj::Str(hi)) = (lo, hi) else {
            return;
        };
        let (Some((n, l)), Some((n2, h))) = (code_of(lo), code_of(hi)) else {
            return;
        };
        if n != n2 || l > h {
            return;
        }
        match dst {
            Obj::Str(d) if !d.is_empty() && d.len() <= 2 * MAX_DST_UNITS => {
                let units: Vec<u16> = d
                    .chunks(2)
                    .map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]))
                    .collect();
                self.ranges.insert((n, l), (h, units));
            }
            Obj::Array(items) => {
                for (i, it) in items.iter().enumerate() {
                    let Some(code) = l.checked_add(i as u32).filter(|c| *c <= h) else {
                        break;
                    };
                    if let Obj::Str(d) = it {
                        if d.len() <= 2 * MAX_DST_UNITS {
                            self.single.insert((n, code), utf16be(d));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// The text of code `code` of `n` bytes.
    pub fn lookup(&self, n: usize, code: u32) -> Option<String> {
        if let Some(t) = self.single.get(&(n, code)) {
            return Some(t.clone());
        }
        let ((rn, lo), (hi, units)) = self.ranges.range(..=(n, code)).next_back()?;
        if *rn != n || code > *hi {
            return None;
        }
        let mut u = units.clone();
        if let Some(last) = u.last_mut() {
            *last = last.wrapping_add((code - lo) as u16);
        }
        Some(String::from_utf16_lossy(&u))
    }

    /// The length of the code at the start of `b` from the codespace ranges; `None` when the
    /// CMap has none.
    pub fn code_len(&self, b: &[u8]) -> Option<usize> {
        if self.space.is_empty() {
            return None;
        }
        for n in 1..=4usize.min(b.len()) {
            let (_, c) = code_of(&b[..n])?;
            if self
                .space
                .iter()
                .any(|&(sn, lo, hi)| sn == n && (lo..=hi).contains(&c))
            {
                return Some(n);
            }
        }
        // a code in no range: as long as the shortest range, per the specification
        self.space.iter().map(|s| s.0).min()
    }
}

/// How a composite font's codes become text when it has no ToUnicode CMap.
#[derive(Debug, PartialEq, Clone, Copy)]
enum CidText {
    /// The codes are Unicode (a predefined UCS2 / UTF16 CMap).
    Unicode,
    /// Not known.
    None,
}

/// A font, reduced to what the text reader needs.
#[derive(Debug)]
pub(crate) struct Font {
    composite: bool,
    to_unicode: Option<CMap>,
    /// Code lengths of a composite font (its encoding CMap), if not the ToUnicode one's.
    encoding_cmap: Option<CMap>,
    cid_text: CidText,
    /// Simple fonts: the text of each code.
    simple: Vec<Option<String>>,
    /// Simple fonts: first code with a width, and the widths.
    first_char: u32,
    widths: Vec<f64>,
    /// Composite fonts: width ranges (low, high, width) and the default width.
    cid_widths: Vec<(u32, u32, f64)>,
    default_width: f64,
    /// Glyph units to text space (1/1000 for all but Type 3 fonts).
    scale: f64,
}

/// One character code of a shown string.
pub(crate) struct Glyph {
    pub text: Option<String>,
    /// Advance in text space units per unit of font size.
    pub width: f64,
    /// A single-byte code 32 (word spacing applies to it).
    pub space: bool,
}

/// The objects a font dictionary refers to, resolved by the caller.
pub(crate) struct FontParts<'a> {
    pub dict: &'a Dict,
    /// The decoded ToUnicode stream.
    pub to_unicode: Option<Vec<u8>>,
    /// The resolved /Encoding (a name, or a dictionary with resolved /Differences), or the
    /// decoded bytes of an embedded CMap of a composite font.
    pub encoding: Option<Obj>,
    pub encoding_cmap: Option<Vec<u8>>,
    /// The resolved /Widths array (simple) or the descendant font's /W array (composite).
    pub widths: Option<std::rc::Rc<[Obj]>>,
    /// The descendant font's /DW, or the /MissingWidth of a simple font's descriptor.
    pub default_width: Option<f64>,
    /// The object budget the objects of its CMaps are charged to while they are read (they
    /// take the room it leaves; the CMaps themselves are small).
    pub mem: super::Budget,
}

impl Font {
    pub fn new(p: FontParts) -> Font {
        let subtype = p.dict.get(b"Subtype").and_then(Obj::as_name).unwrap_or(b"");
        let composite = subtype == b"Type0";
        let to_unicode = p.to_unicode.as_deref().map(|d| CMap::parse(d, p.mem));
        let scale = if subtype == b"Type3" {
            p.dict
                .get(b"FontMatrix")
                .and_then(Obj::as_array)
                .and_then(|m| m.first())
                .and_then(Obj::as_f64)
                .filter(|v| v.abs() > 0.0 && v.abs() < 10.0)
                .unwrap_or(0.001)
        } else {
            0.001
        };
        let mut f = Font {
            composite,
            to_unicode,
            encoding_cmap: None,
            cid_text: CidText::None,
            simple: Vec::new(),
            first_char: 0,
            widths: Vec::new(),
            cid_widths: Vec::new(),
            default_width: 0.0,
            scale,
        };
        if composite {
            if let Some(Obj::Name(n)) = &p.encoding {
                let n = String::from_utf8_lossy(n);
                if n.contains("UCS2") || n.contains("UTF16") {
                    f.cid_text = CidText::Unicode;
                }
            }
            f.encoding_cmap = p.encoding_cmap.as_deref().map(|d| CMap::parse(d, p.mem));
            f.default_width = p.default_width.unwrap_or(1000.0);
            if let Some(w) = &p.widths {
                f.cid_widths = cid_widths(w);
                f.cid_widths.sort_by_key(|r| r.0);
            }
        } else {
            let base_font = p
                .dict
                .get(b"BaseFont")
                .and_then(Obj::as_name)
                .unwrap_or(b"");
            f.simple = simple_encoding(subtype, base_font, p.encoding.as_ref());
            f.first_char = p
                .dict
                .get(b"FirstChar")
                .and_then(Obj::as_int)
                .and_then(|c| u32::try_from(c).ok())
                .unwrap_or(0);
            if let Some(w) = &p.widths {
                f.widths = w
                    .iter()
                    .take(256)
                    .map(|o| o.as_f64().unwrap_or(0.0))
                    .collect();
            }
            f.default_width =
                p.default_width
                    .unwrap_or(if f.widths.is_empty() { 500.0 } else { 0.0 });
        }
        f
    }

    /// Splits `s` into codes and calls `out` for each.
    pub fn glyphs(&self, s: &[u8], mut out: impl FnMut(Glyph)) {
        let mut i = 0usize;
        while i < s.len() {
            let rest = &s[i..];
            let n = if self.composite {
                self.encoding_cmap
                    .as_ref()
                    .and_then(|c| c.code_len(rest))
                    .or_else(|| self.to_unicode.as_ref().and_then(|c| c.code_len(rest)))
                    .unwrap_or(2)
            } else {
                1
            }
            .clamp(1, 4)
            .min(rest.len());
            let code = rest[..n].iter().fold(0u32, |a, &x| a << 8 | u32::from(x));
            i += n;
            let mut text = self.to_unicode.as_ref().and_then(|c| c.lookup(n, code));
            if text.is_none() {
                text = if self.composite {
                    match self.cid_text {
                        CidText::Unicode => Some(utf16be(&rest[..n])),
                        CidText::None => None,
                    }
                } else {
                    self.simple.get(code as usize).cloned().flatten()
                };
            }
            let w = if self.composite {
                let at = self.cid_widths.partition_point(|r| r.0 <= code);
                at.checked_sub(1)
                    .map(|k| self.cid_widths[k])
                    .filter(|r| code <= r.1)
                    .map_or(self.default_width, |r| r.2)
            } else {
                code.checked_sub(self.first_char)
                    .and_then(|k| self.widths.get(k as usize))
                    .copied()
                    .filter(|w| *w > 0.0)
                    .unwrap_or(self.default_width)
            };
            out(Glyph {
                text,
                width: w * self.scale,
                space: n == 1 && code == 32,
            });
        }
    }
}

/// Whether a /BaseFont names the Symbol font (with a subset tag or a style suffix).
fn is_symbol_font(base_font: &[u8]) -> bool {
    let name = match base_font.iter().position(|&b| b == b'+') {
        Some(i) if i == 6 => &base_font[7..],
        _ => base_font,
    };
    let stem = name
        .split(|&b| b == b',' || b == b'-')
        .next()
        .unwrap_or(b"");
    stem == b"Symbol" || stem == b"SymbolMT"
}

/// The text of each code of a simple font: its base encoding with its /Differences. A font
/// that names no base encoding uses its built-in one: the Symbol font's for the Symbol font.
fn simple_encoding(subtype: &[u8], base_font: &[u8], enc: Option<&Obj>) -> Vec<Option<String>> {
    let base_name: &[u8] = match enc {
        Some(Obj::Name(n)) => n,
        Some(Obj::Dict(d)) => d.get(b"BaseEncoding").and_then(Obj::as_name).unwrap_or(b""),
        _ => b"",
    };
    let base: fn(u8) -> Option<char> = match base_name {
        b"WinAnsiEncoding" => win_ansi,
        b"MacRomanEncoding" => mac_roman,
        b"StandardEncoding" => standard,
        _ if is_symbol_font(base_font) => symbol,
        // a TrueType font without an encoding is most often written in the Windows code page
        _ if subtype == b"TrueType" => win_ansi,
        _ => standard,
    };
    let mut table: Vec<Option<String>> = (0..=255u8).map(|c| base(c).map(String::from)).collect();
    if let Some(Obj::Dict(d)) = enc {
        if let Some(Obj::Array(diff)) = d.get(b"Differences") {
            let mut code: Option<usize> = None;
            for item in diff.iter() {
                match item {
                    Obj::Int(c) => code = usize::try_from(*c).ok().filter(|c| *c < 256),
                    Obj::Name(n) => {
                        if let Some(c) = code {
                            table[c] = glyph_text(n);
                            code = Some(c + 1).filter(|c| *c < 256);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    table
}

/// A composite font's /W array: `c [w1 w2 ...]` and `c1 c2 w` entries.
fn cid_widths(w: &[Obj]) -> Vec<(u32, u32, f64)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < w.len() {
        let Some(c) = w[i].as_int().and_then(|c| u32::try_from(c).ok()) else {
            break;
        };
        match w.get(i + 1) {
            Some(Obj::Array(list)) => {
                for (k, x) in list.iter().enumerate() {
                    if let (Some(code), Some(v)) = (c.checked_add(k as u32), x.as_f64()) {
                        out.push((code, code, v));
                    }
                }
                i += 2;
            }
            Some(o) => {
                let (Some(hi), Some(v)) = (
                    o.as_int().and_then(|h| u32::try_from(h).ok()),
                    w.get(i + 2).and_then(Obj::as_f64),
                ) else {
                    break;
                };
                out.push((c, hi, v));
                i += 3;
            }
            None => break,
        }
    }
    out
}
