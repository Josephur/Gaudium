//! Zero-dependency BPE tokenizer and streaming UTF-8 decoder.
//!
//! Loads Hugging Face `tokenizer.json` files for both:
//! - **ByteLevel BPE** (Qwen2 / Qwen3 / Qwen3.5 / Qwen3.8): GPT-2 byte-to-Unicode alphabet,
//!   exact hand-written Qwen2 / Qwen3.5 pre-tokenizer state machine, and O(n log n) min-heap BPE.
//! - **Character-level `byte_fallback` BPE** (Gemma 4): U+2581 space substitution, `<0xNN>`
//!   byte fallback tokens, and valid-UTF-8 byte-run coalescing on decode.
//! - **ByteLevel BPE with the letter-case pattern** (Inkling 2): a pre-tokenizer that splits
//!   words at case runs (an uppercase run may lead a lowercase one), numbers in groups of three,
//!   punctuation runs with their trailing newlines or slashes, and whitespace runs; its character
//!   classes come from a table generated from the reference tokenizer itself
//!   (`tokenizer/char_classes.rs`). With `ignore_merges` a piece that is a whole vocabulary entry
//!   is that entry, without merging.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use crate::json::parse_json;

mod char_classes;

#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Format(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "tokenizer I/O error: {e}"),
            Self::Format(s) => write!(f, "tokenizer format error: {s}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[derive(Debug, Clone)]
pub struct AddedToken {
    pub id: i32,
    pub content: String,
    pub special: bool,
}

/// ByteLevel GPT-2 `bytes_to_unicode` lookup tables.
fn byte_level_tables() -> ([u32; 256], HashMap<u32, u8>) {
    let mut b2cp = [0u32; 256];
    let mut cp2b = HashMap::with_capacity(256);
    let mut extra = 0u32;
    for (b, slot) in b2cp.iter_mut().enumerate() {
        let keep =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        let cp = if keep {
            b as u32
        } else {
            let v = 256 + extra;
            extra += 1;
            v
        };
        *slot = cp;
        cp2b.insert(cp, b as u8);
    }
    (b2cp, cp2b)
}

#[inline]
fn pair_key(a: i32, b: i32) -> u64 {
    ((a as u32 as u64) << 32) | (b as u32 as u64)
}

#[inline]
fn is_combining_mark(c: char) -> bool {
    let u = c as u32;
    (0x0300..=0x036F).contains(&u)
        || (0x1AB0..=0x1AFF).contains(&u)
        || (0x1DC0..=0x1DFF).contains(&u)
        || (0x20D0..=0x20FF).contains(&u)
        || (0xFE20..=0xFE2F).contains(&u)
}

const FLAG_L: u8 = 1;
const FLAG_M: u8 = 2;
const FLAG_N: u8 = 4;
const FLAG_WS: u8 = 8;

#[inline]
fn char_flags(c: char) -> u8 {
    let mut f = 0u8;
    if c.is_alphabetic() {
        f |= FLAG_L;
    }
    if is_combining_mark(c) {
        f |= FLAG_M;
    }
    if c.is_numeric() {
        f |= FLAG_N;
    }
    if c.is_whitespace() {
        f |= FLAG_WS;
    }
    f
}

/// Exact Qwen2 / Qwen3.5 regex pre-tokenizer state machine over Unicode code points.
/// Returns the end char-index `e > i` of the next isolated piece.
fn match_pretok_at(chars: &[char], i: usize, include_marks: bool) -> usize {
    let n = chars.len();
    let fl = |k: usize| -> u8 {
        if k < n {
            char_flags(chars[k])
        } else {
            0
        }
    };
    let lower_ascii = |k: usize| -> u32 {
        if k >= n {
            return 0;
        }
        let u = chars[k] as u32;
        if (b'A' as u32..=b'Z' as u32).contains(&u) {
            u + 32
        } else if u == 0x017F {
            b's' as u32
        } else {
            u
        }
    };

    // 1. Contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if chars[i] == '\'' && i + 1 < n {
        let a = lower_ascii(i + 1);
        if a == b's' as u32 || a == b't' as u32 || a == b'm' as u32 || a == b'd' as u32 {
            return i + 2;
        }
        let b = lower_ascii(i + 2);
        if (a == b'r' as u32 && b == b'e' as u32)
            || (a == b'v' as u32 && b == b'e' as u32)
            || (a == b'l' as u32 && b == b'l' as u32)
        {
            return i + 3;
        }
    }

    let f0 = fl(i);
    let lm = if include_marks {
        FLAG_L | FLAG_M
    } else {
        FLAG_L
    };
    let lm_run = |mut k: usize| -> usize {
        while k < n && (fl(k) & lm) != 0 {
            k += 1;
        }
        k
    };

    // 2. [^\r\n\p{L}\p{N}]?[\p{L}(\p{M})]+
    if (f0 & FLAG_L) != 0 {
        return lm_run(i);
    }
    if chars[i] != '\r' && chars[i] != '\n' && (f0 & FLAG_N) == 0 {
        if (fl(i + 1) & lm) != 0 {
            return lm_run(i + 1);
        }
        if (f0 & lm) != 0 {
            return lm_run(i);
        }
    }

    // 3. \p{N} (single digit per piece)
    if (f0 & FLAG_N) != 0 {
        return i + 1;
    }

    // 4.  ?[^\s\p{L}(\p{M})\p{N}]+[\r\n]*
    let is_other = |k: usize| -> bool { k < n && (fl(k) & (FLAG_WS | lm | FLAG_N)) == 0 };
    {
        let mut k = i;
        if chars[k] == ' ' && is_other(k + 1) {
            k += 1;
        }
        if is_other(k) {
            while is_other(k) {
                k += 1;
            }
            while k < n && (chars[k] == '\r' || chars[k] == '\n') {
                k += 1;
            }
            return k;
        }
    }

    // 5-7. Whitespace runs
    if (f0 & FLAG_WS) != 0 {
        let mut j = i;
        while j < n && (fl(j) & FLAG_WS) != 0 {
            j += 1;
        }
        // \s*[\r\n]+
        for k in (i + 1..=j).rev() {
            if chars[k - 1] == '\r' || chars[k - 1] == '\n' {
                return k;
            }
        }
        // \s+(?!\S)
        if j == n {
            return j;
        }
        if j - 1 > i {
            return j - 1;
        }
        // \s+
        return j;
    }

    i + 1
}

/// The `Split` pattern of the letter-case pre-tokenizer, exactly as `tokenizer.json` writes it.
const LETTER_CASE_PATTERN: &str = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// A character's class under the letter-case pattern (the ids of `char_classes.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharClass {
    /// Not a space, a letter, a mark or a number (punctuation, symbols, unassigned).
    Other,
    /// `Lu`, `Lt`: may only lead a word's uppercase run.
    Upper,
    /// `Ll`: lowercase only.
    Lower,
    /// `Lm`, `Lo`: a letter of both runs.
    Both,
    /// `M`: in both runs, but not a letter.
    Mark,
    /// `N`.
    Number,
    /// `\s`.
    Space,
}

fn char_class(c: char) -> CharClass {
    let key = ((c as u32) << 3) | 7;
    let runs = &char_classes::CLASS_RUNS;
    // the last run starting at or before `c`
    let k = runs.partition_point(|&r| r <= key);
    match runs[k.saturating_sub(1)] & 7 {
        1 => CharClass::Upper,
        2 => CharClass::Lower,
        3 => CharClass::Both,
        4 => CharClass::Mark,
        5 => CharClass::Number,
        6 => CharClass::Space,
        _ => CharClass::Other,
    }
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `i`: its end, or `i` when there is none.
fn contraction_end(chars: &[char], i: usize) -> usize {
    if chars.get(i) != Some(&'\'') {
        return i;
    }
    let fold = |k: usize| -> char {
        match chars.get(k) {
            // the one non-ASCII letter that case-folds onto these: U+017F LATIN SMALL LETTER LONG S
            Some('\u{17F}') => 's',
            Some(c) => c.to_ascii_lowercase(),
            None => '\0',
        }
    };
    match (fold(i + 1), fold(i + 2)) {
        ('s' | 't' | 'm' | 'd', _) => i + 2,
        ('r' | 'v', 'e') | ('l', 'l') => i + 3,
        _ => i,
    }
}

/// The letter-case pattern at `i`: the end `e > i` of the next isolated piece. The alternatives
/// are tried in the pattern's order with its backtracking, so the first one that matches wins.
fn match_letter_case_at(chars: &[char], cls: &[CharClass], i: usize) -> usize {
    use CharClass::*;
    let n = chars.len();
    let at = |k: usize| if k < n { Some(cls[k]) } else { None };
    // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and [\p{Ll}\p{Lm}\p{Lo}\p{M}]
    let upper = |k: usize| matches!(at(k), Some(Upper | Both | Mark));
    let lower = |k: usize| matches!(at(k), Some(Lower | Both | Mark));
    // [^\r\n\p{L}\p{N}]
    let prefix = |k: usize| {
        k < n
            && !matches!(cls[k], Upper | Lower | Both | Number)
            && !matches!(chars[k], '\r' | '\n')
    };
    // [^\s\p{L}\p{N}]
    let other = |k: usize| matches!(at(k), Some(Other | Mark));
    let run = |mut k: usize, f: &dyn Fn(usize) -> bool| {
        while f(k) {
            k += 1;
        }
        k
    };
    // 1. P? U* L+ C?
    for take in [true, false] {
        if take && !prefix(i) {
            continue;
        }
        let k0 = i + usize::from(take);
        let u_end = run(k0, &upper);
        // L+ starts at the last position the U* run can give back to it
        let mut p = u_end;
        loop {
            if lower(p) {
                return contraction_end(chars, run(p, &lower));
            }
            if p == k0 {
                break;
            }
            p -= 1;
        }
    }
    // 2. P? U+ L* C?
    for take in [true, false] {
        if take && !prefix(i) {
            continue;
        }
        let k0 = i + usize::from(take);
        let u_end = run(k0, &upper);
        if u_end > k0 {
            return contraction_end(chars, run(u_end, &lower));
        }
    }
    // 3. \p{N}{1,3}
    if at(i) == Some(Number) {
        let mut k = i + 1;
        while k < n && k < i + 3 && cls[k] == Number {
            k += 1;
        }
        return k;
    }
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
    let k = if chars[i] == ' ' && other(i + 1) {
        i + 1
    } else {
        i
    };
    if other(k) {
        let mut k = run(k, &other);
        while k < n && matches!(chars[k], '\r' | '\n' | '/') {
            k += 1;
        }
        return k;
    }
    // 5-7. whitespace runs
    if cls[i] == Space {
        let j = run(i, &|k: usize| at(k) == Some(Space));
        // \s*[\r\n]+
        if let Some(q) = (i..j).rev().find(|&q| matches!(chars[q], '\r' | '\n')) {
            return q + 1;
        }
        // \s+(?!\S)
        if j == n {
            return j;
        }
        if j - 1 > i {
            return j - 1;
        }
        // \s+
        return j;
    }
    i + 1
}

/// Parses `<0xNN>` uppercase hex fallback byte tokens (SentencePiece / Gemma 4).
fn parse_hex_byte_token(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 6 || b[0] != b'<' || b[1] != b'0' || b[2] != b'x' || b[5] != b'>' {
        return None;
    }
    let hex = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    Some((hex(b[3])? << 4) | hex(b[4])?)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct MergeCand {
    rank: i32,
    pos: usize,
    a: i32,
    b: i32,
}

impl Ord for MergeCand {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap order: lowest rank first, then leftmost pos
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for MergeCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

const FLAG_ADDED: u8 = 1;
const FLAG_SPECIAL: u8 = 2;
const FLAG_BYTE_FALLBACK: u8 = 4;

/// The pre-tokenizer a `tokenizer.json` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pretok {
    /// The Qwen2 / Qwen3.5 pattern (with or without `\p{M}` in the letter runs).
    Qwen { include_marks: bool },
    /// The letter-case pattern.
    LetterCase,
}

pub struct Tokenizer {
    byte_fallback: bool,
    pretok: Pretok,
    /// A piece that is a whole vocabulary entry is that entry (the model's `ignore_merges`).
    ignore_merges: bool,
    /// ByteLevel: the alphabet character of each byte.
    byte_char: [char; 256],
    vocab: HashMap<String, i32>,
    id_bytes: Vec<Vec<u8>>,
    id_piece: Vec<String>,
    id_flags: Vec<u8>,
    merges: HashMap<u64, (i32, i32)>,
    byte_id: [i32; 256],
    added: Vec<AddedToken>,
    added_first_byte: [bool; 256],
    piece_cache: Mutex<HashMap<String, Vec<i32>>>,
}

impl Tokenizer {
    /// Loads a Hugging Face `tokenizer.json` from disk.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json_str(&text)
    }

    /// Loads `tokenizer.json` from a model directory.
    pub fn from_model_dir(dir: &Path) -> Result<Self, TokenizerError> {
        Self::from_file(&dir.join("tokenizer.json"))
    }

    /// Parses a `tokenizer.json` string.
    pub fn from_json_str(text: &str) -> Result<Self, TokenizerError> {
        let root = parse_json(text).map_err(|e| TokenizerError::Format(format!("json: {e}")))?;
        let model = root
            .get("model")
            .ok_or_else(|| TokenizerError::Format("missing model object".into()))?;
        let mtype = model
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if mtype != "BPE" {
            return Err(TokenizerError::Format(format!(
                "unsupported tokenizer model type '{mtype}' (expected BPE)"
            )));
        }
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TokenizerError::Format("missing model.vocab".into()))?;

        let added_arr = root
            .get("added_tokens")
            .and_then(|v| v.as_array())
            .unwrap_or(&[]);

        let mut max_id: i32 = -1;
        for (_, v) in vocab_obj {
            let id = v
                .as_i64()
                .ok_or_else(|| TokenizerError::Format("non-integer vocab id".into()))?
                as i32;
            if id > max_id {
                max_id = id;
            }
        }
        for a in added_arr {
            if let Some(id) = a.get("id").and_then(|v| v.as_i64()) {
                let id = id as i32;
                if id > max_id {
                    max_id = id;
                }
            }
        }
        if max_id < 0 {
            return Err(TokenizerError::Format("empty vocabulary".into()));
        }

        let n_ids = (max_id as usize) + 1;
        let mut vocab = HashMap::with_capacity(vocab_obj.len() + added_arr.len());
        let mut id_bytes = vec![Vec::new(); n_ids];
        let mut id_piece = vec![String::new(); n_ids];
        let mut id_flags = vec![0u8; n_ids];
        let mut byte_id = [-1i32; 256];

        let byte_fallback = model
            .get("byte_fallback")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let (b2cp, cp2b) = byte_level_tables();
        let ignore_merges = model
            .get("ignore_merges")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        for (piece, v) in vocab_obj {
            let id = v.as_i64().unwrap() as i32;
            if id < 0 {
                continue;
            }
            let uid = id as usize;
            vocab.insert(piece.clone(), id);
            id_piece[uid] = piece.clone();
            if byte_fallback {
                if let Some(b) = parse_hex_byte_token(piece) {
                    id_bytes[uid] = vec![b];
                    byte_id[b as usize] = id;
                    id_flags[uid] |= FLAG_BYTE_FALLBACK;
                } else {
                    let decoded = piece.replace('\u{2581}', " ");
                    id_bytes[uid] = decoded.into_bytes();
                }
            } else {
                let mut raw = Vec::with_capacity(piece.len());
                for ch in piece.chars() {
                    if let Some(&b) = cp2b.get(&(ch as u32)) {
                        raw.push(b);
                    } else {
                        // Some checkpoints include raw Unicode added tokens in vocab
                        let mut buf = [0u8; 4];
                        raw.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
                if raw.len() == 1 && piece.chars().count() == 1 {
                    byte_id[raw[0] as usize] = id;
                }
                id_bytes[uid] = raw;
            }
        }

        // Parse added_tokens
        let mut added = Vec::with_capacity(added_arr.len());
        let mut added_first_byte = [false; 256];
        for a in added_arr {
            let Some(id) = a.get("id").and_then(|v| v.as_i64()).map(|x| x as i32) else {
                continue;
            };
            let Some(content) = a.get("content").and_then(|v| v.as_str()) else {
                continue;
            };
            let special = a.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
            if id >= 0 && (id as usize) < n_ids {
                let uid = id as usize;
                id_bytes[uid] = content.as_bytes().to_vec();
                id_piece[uid] = content.to_string();
                id_flags[uid] |= FLAG_ADDED;
                if special {
                    id_flags[uid] |= FLAG_SPECIAL;
                }
            }
            vocab.insert(content.to_string(), id);
            if let Some(&first) = content.as_bytes().first() {
                added_first_byte[first as usize] = true;
            }
            added.push(AddedToken {
                id,
                content: content.to_string(),
                special,
            });
        }
        // Sort longest first so leftmost-longest wins
        added.sort_by_key(|a| std::cmp::Reverse(a.content.len()));

        // Parse merges (supports both `"a b"` string and `["a", "b"]` pair array)
        let merges_arr = model
            .get("merges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TokenizerError::Format("missing model.merges".into()))?;
        let mut merges = HashMap::with_capacity(merges_arr.len());
        let mut joined = String::new();
        for (rank, entry) in merges_arr.iter().enumerate() {
            let (left, right): (&str, &str) = if let Some(s) = entry.as_str() {
                let Some(sp) = s.find(' ') else { continue };
                (&s[..sp], &s[sp + 1..])
            } else if let Some(arr) = entry.as_array() {
                if arr.len() != 2 {
                    continue;
                }
                let (Some(l), Some(r)) = (arr[0].as_str(), arr[1].as_str()) else {
                    continue;
                };
                (l, r)
            } else {
                continue;
            };
            let (Some(&a), Some(&b)) = (vocab.get(left), vocab.get(right)) else {
                continue;
            };
            joined.clear();
            joined.push_str(left);
            joined.push_str(right);
            if let Some(&ab) = vocab.get(&joined) {
                merges.insert(pair_key(a, b), (rank as i32, ab));
            }
        }

        // Detect whether pre-tokenizer regex includes Unicode marks \p{M} (Qwen3.5 vs Qwen2/Qwen3),
        // or is the letter-case pattern
        let mut include_marks = false;
        let mut letter_case = false;
        if let Some(pt) = root.get("pre_tokenizer") {
            if let Some(list) = pt.get("pretokenizers").and_then(|v| v.as_array()) {
                for p in list {
                    if p.get("type").and_then(|v| v.as_str()) == Some("Split") {
                        if let Some(re) = p
                            .get("pattern")
                            .and_then(|pat| pat.get("Regex"))
                            .and_then(|v| v.as_str())
                        {
                            if re.contains("\\p{M}") {
                                include_marks = true;
                            }
                            if re == LETTER_CASE_PATTERN {
                                letter_case = true;
                            }
                        }
                    }
                }
            }
        }

        if !byte_fallback {
            for (b, &id) in byte_id.iter().enumerate() {
                if id < 0 {
                    return Err(TokenizerError::Format(format!(
                        "ByteLevel vocab missing single-byte token for 0x{b:02X}"
                    )));
                }
            }
        }

        let mut byte_char = ['\0'; 256];
        for (slot, &cp) in byte_char.iter_mut().zip(b2cp.iter()) {
            *slot = char::from_u32(cp).unwrap_or('\0');
        }
        Ok(Self {
            byte_fallback,
            pretok: if letter_case {
                Pretok::LetterCase
            } else {
                Pretok::Qwen { include_marks }
            },
            ignore_merges,
            byte_char,
            vocab,
            id_bytes,
            id_piece,
            id_flags,
            merges,
            byte_id,
            added,
            added_first_byte,
            piece_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_bytes.len()
    }

    pub fn token_id(&self, token_text: &str) -> Option<i32> {
        self.vocab.get(token_text).copied()
    }

    pub fn is_special(&self, id: i32) -> bool {
        id >= 0
            && (id as usize) < self.id_flags.len()
            && (self.id_flags[id as usize] & FLAG_SPECIAL) != 0
    }

    pub fn is_added(&self, id: i32) -> bool {
        id >= 0
            && (id as usize) < self.id_flags.len()
            && (self.id_flags[id as usize] & FLAG_ADDED) != 0
    }

    pub fn token_bytes(&self, id: i32) -> &[u8] {
        if id >= 0 && (id as usize) < self.id_bytes.len() {
            &self.id_bytes[id as usize]
        } else {
            &[]
        }
    }

    pub fn token_piece(&self, id: i32) -> &str {
        if id >= 0 && (id as usize) < self.id_piece.len() {
            &self.id_piece[id as usize]
        } else {
            ""
        }
    }

    /// Encodes `text` into token IDs. When `parse_special` is true, added tokens (such as
    /// `<|im_start|>`, `<|im_end|>`) are matched greedily (leftmost-longest) before BPE.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<i32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        if !parse_special || self.added.is_empty() {
            self.encode_segment(text, &mut out);
            return out;
        }
        let bytes = text.as_bytes();
        let mut seg = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            if self.added_first_byte[bytes[i] as usize] {
                let mut hit: Option<&AddedToken> = None;
                for tok in &self.added {
                    if text[i..].starts_with(&tok.content) {
                        hit = Some(tok);
                        break;
                    }
                }
                if let Some(tok) = hit {
                    if i > seg {
                        self.encode_segment(&text[seg..i], &mut out);
                    }
                    out.push(tok.id);
                    i += tok.content.len();
                    seg = i;
                    continue;
                }
            }
            i += 1;
        }
        if seg < text.len() {
            self.encode_segment(&text[seg..], &mut out);
        }
        out
    }

    fn encode_segment(&self, seg: &str, out: &mut Vec<i32>) {
        if seg.is_empty() {
            return;
        }
        if self.byte_fallback {
            let mut ids = Vec::with_capacity(seg.len());
            let mut ch_buf = String::with_capacity(4);
            for ch in seg.chars() {
                ch_buf.clear();
                if ch == ' ' {
                    ch_buf.push('\u{2581}');
                } else {
                    ch_buf.push(ch);
                }
                if let Some(&id) = self.vocab.get(&ch_buf) {
                    if !self.is_added(id) {
                        ids.push(id);
                        continue;
                    }
                }
                let mut utf8 = [0u8; 4];
                for &b in ch.encode_utf8(&mut utf8).as_bytes() {
                    let bid = self.byte_id[b as usize];
                    if bid >= 0 {
                        ids.push(bid);
                    }
                }
            }
            self.merge_all(&mut ids);
            out.extend_from_slice(&ids);
            return;
        }

        // ByteLevel BPE pre-tokenization
        let mut chars: Vec<char> = Vec::with_capacity(seg.len());
        let mut offs: Vec<usize> = Vec::with_capacity(seg.len() + 1);
        for (byte_idx, ch) in seg.char_indices() {
            chars.push(ch);
            offs.push(byte_idx);
        }
        offs.push(seg.len());

        let mut i = 0usize;
        match self.pretok {
            Pretok::Qwen { include_marks } => {
                while i < chars.len() {
                    let e = match_pretok_at(&chars, i, include_marks);
                    self.bpe_piece(&seg[offs[i]..offs[e]], out);
                    i = e;
                }
            }
            Pretok::LetterCase => {
                let cls: Vec<CharClass> = chars.iter().map(|&c| char_class(c)).collect();
                while i < chars.len() {
                    let e = match_letter_case_at(&chars, &cls, i);
                    self.bpe_piece(&seg[offs[i]..offs[e]], out);
                    i = e;
                }
            }
        }
    }

    fn bpe_piece(&self, piece: &str, out: &mut Vec<i32>) {
        let raw = piece.as_bytes();
        if raw.len() == 1 {
            out.push(self.byte_id[raw[0] as usize]);
            return;
        }
        if self.ignore_merges {
            let word: String = raw.iter().map(|&b| self.byte_char[b as usize]).collect();
            if let Some(&id) = self.vocab.get(&word) {
                if !self.is_added(id) {
                    out.push(id);
                    return;
                }
            }
        }
        let cacheable = raw.len() <= 64;
        if cacheable {
            if let Ok(guard) = self.piece_cache.lock() {
                if let Some(cached) = guard.get(piece) {
                    out.extend_from_slice(cached);
                    return;
                }
            }
        }
        let mut ids: Vec<i32> = raw.iter().map(|&b| self.byte_id[b as usize]).collect();
        self.merge_all(&mut ids);
        out.extend_from_slice(&ids);
        if cacheable {
            if let Ok(mut guard) = self.piece_cache.lock() {
                if guard.len() > 200_000 {
                    guard.clear();
                }
                guard.insert(piece.to_string(), ids);
            }
        }
    }

    fn merge_all(&self, ids: &mut Vec<i32>) {
        let n = ids.len();
        if n < 2 {
            return;
        }
        let mut prev: Vec<isize> = (0..n).map(|k| k as isize - 1).collect();
        let mut next: Vec<isize> = (0..n)
            .map(|k| if k + 1 < n { (k + 1) as isize } else { -1 })
            .collect();
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::with_capacity(n);

        for k in 0..n - 1 {
            if let Some(&(rank, _)) = self.merges.get(&pair_key(ids[k], ids[k + 1])) {
                heap.push(MergeCand {
                    rank,
                    pos: k,
                    a: ids[k],
                    b: ids[k + 1],
                });
            }
        }

        while let Some(c) = heap.pop() {
            if !alive[c.pos] {
                continue;
            }
            let q = next[c.pos];
            if q < 0 {
                continue;
            }
            let qu = q as usize;
            if ids[c.pos] != c.a || ids[qu] != c.b {
                continue;
            }
            let Some(&(_, merged)) = self.merges.get(&pair_key(c.a, c.b)) else {
                continue;
            };
            ids[c.pos] = merged;
            alive[qu] = false;
            let nq = next[qu];
            next[c.pos] = nq;
            if nq >= 0 {
                prev[nq as usize] = c.pos as isize;
            }
            let p = prev[c.pos];
            if p >= 0 {
                let pu = p as usize;
                if let Some(&(rank, _)) = self.merges.get(&pair_key(ids[pu], ids[c.pos])) {
                    heap.push(MergeCand {
                        rank,
                        pos: pu,
                        a: ids[pu],
                        b: ids[c.pos],
                    });
                }
            }
            if nq >= 0 {
                let nqu = nq as usize;
                if let Some(&(rank, _)) = self.merges.get(&pair_key(ids[c.pos], ids[nqu])) {
                    heap.push(MergeCand {
                        rank,
                        pos: c.pos,
                        a: ids[c.pos],
                        b: ids[nqu],
                    });
                }
            }
        }

        let mut res = Vec::with_capacity(n);
        let mut cur = 0isize;
        while cur >= 0 {
            res.push(ids[cur as usize]);
            cur = next[cur as usize];
        }
        *ids = res;
    }

    /// Decodes token IDs back to UTF-8 (replacing invalid byte sequences with U+FFFD).
    pub fn decode(&self, ids: &[i32], skip_special: bool) -> String {
        if self.byte_fallback {
            let mut out = String::new();
            let mut run = Vec::new();
            let mut nrun = 0usize;
            let flush = |run: &mut Vec<u8>, nrun: &mut usize, out: &mut String| {
                if *nrun == 0 {
                    return;
                }
                match std::str::from_utf8(run) {
                    Ok(s) => out.push_str(s),
                    Err(_) => {
                        for _ in 0..*nrun {
                            out.push('\u{FFFD}');
                        }
                    }
                }
                run.clear();
                *nrun = 0;
            };
            for &id in ids {
                if skip_special && self.is_special(id) {
                    continue;
                }
                if id >= 0
                    && (id as usize) < self.id_flags.len()
                    && (self.id_flags[id as usize] & FLAG_BYTE_FALLBACK) != 0
                {
                    run.extend_from_slice(self.token_bytes(id));
                    nrun += 1;
                } else {
                    flush(&mut run, &mut nrun, &mut out);
                    out.push_str(&String::from_utf8_lossy(self.token_bytes(id)));
                }
            }
            flush(&mut run, &mut nrun, &mut out);
            return out;
        }

        let mut raw = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if skip_special && self.is_special(id) {
                continue;
            }
            raw.extend_from_slice(self.token_bytes(id));
        }
        String::from_utf8_lossy(&raw).into_owned()
    }
}

/// Returns the longest prefix byte length of `bytes` that ends on a complete UTF-8 sequence
/// (valid or invalid-but-terminated), holding back only a trailing incomplete multi-byte prefix.
pub fn utf8_complete_prefix(bytes: &[u8]) -> usize {
    let mut i = 0usize;
    while i < bytes.len() {
        match std::str::from_utf8(&bytes[i..]) {
            Ok(_) => return bytes.len(),
            Err(e) => {
                i += e.valid_up_to();
                if let Some(err_len) = e.error_len() {
                    i += err_len;
                } else {
                    // Incomplete multi-byte sequence at end of buffer — hold back!
                    return i;
                }
            }
        }
    }
    i
}

/// Streaming incremental UTF-8 detokenizer that never splits multi-byte code points across chunks.
pub struct StreamDecoder<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
}

impl<'a> StreamDecoder<'a> {
    pub fn new(tok: &'a Tokenizer) -> Self {
        Self {
            tok,
            pending: Vec::new(),
        }
    }

    pub fn push(&mut self, id: i32) -> String {
        self.pending.extend_from_slice(self.tok.token_bytes(id));
        let k = utf8_complete_prefix(&self.pending);
        if k == 0 {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending[..k]).into_owned();
        self.pending.drain(..k);
        out
    }

    pub fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Formats a single-turn user prompt with the Qwen3 ChatML template.
/// When `enable_thinking` is false, appends `<think>\n\n</think>\n\n` so Qwen3 answers directly.
pub fn format_qwen3_chat(
    user_prompt: &str,
    system_prompt: Option<&str>,
    enable_thinking: bool,
) -> String {
    let mut s = String::new();
    if let Some(sys) = system_prompt {
        if !sys.is_empty() {
            s.push_str("<|im_start|>system\n");
            s.push_str(sys);
            s.push_str("<|im_end|>\n");
        }
    }
    s.push_str("<|im_start|>user\n");
    s.push_str(user_prompt);
    s.push_str("<|im_end|>\n<|im_start|>assistant\n");
    if !enable_thinking {
        s.push_str("<think>\n\n</think>\n\n");
    }
    s
}

/// Formats a single-turn user prompt with the Gemma 4 turn template:
/// `<bos>[<|turn>system\n[<|think|>\n]{system}<turn|>\n]<|turn>user\n{user}<turn|>\n<|turn>model\n`.
///
/// `close_empty_thought`: the model's template ends a non-thinking generation prompt with an empty
/// thought channel (`<|channel>thought\n<channel|>`); see [`gemma4_template_closes_thought`].
pub fn format_gemma4_chat(
    user_prompt: &str,
    system_prompt: Option<&str>,
    enable_thinking: bool,
    close_empty_thought: bool,
) -> String {
    let sys = system_prompt.map(str::trim).filter(|s| !s.is_empty());
    let mut s = String::from("<bos>");
    if enable_thinking || sys.is_some() {
        s.push_str("<|turn>system\n");
        if enable_thinking {
            s.push_str("<|think|>\n");
        }
        if let Some(sys) = sys {
            s.push_str(sys);
        }
        s.push_str("<turn|>\n");
    }
    s.push_str("<|turn>user\n");
    s.push_str(user_prompt.trim());
    s.push_str("<turn|>\n<|turn>model\n");
    if !enable_thinking && close_empty_thought {
        s.push_str("<|channel>thought\n<channel|>");
    }
    s
}

/// True when a Gemma 4 `chat_template.jinja` closes an empty thought channel after the generation
/// prompt of a non-thinking request (the larger sizes' templates do, the small sizes' do not).
pub fn gemma4_template_closes_thought(chat_template: &str) -> bool {
    chat_template.contains("<|channel>thought\\n<channel|>")
}

/// A byte-level tokenizer for tests: one token per byte (ids 0..=255) and the given added
/// tokens `(content, special)` from id 256 on.
#[cfg(test)]
pub(crate) fn test_tokenizer(added: &[(&str, bool)]) -> Tokenizer {
    let (b2cp, _) = byte_level_tables();
    let vocab: Vec<String> = b2cp
        .iter()
        .enumerate()
        .map(|(b, &cp)| {
            let ch = char::from_u32(cp).unwrap();
            format!("{}: {b}", crate::json::quote_json(&ch.to_string()))
        })
        .collect();
    let added: Vec<String> = added
        .iter()
        .enumerate()
        .map(|(i, (content, special))| {
            format!(
                "{{\"id\": {}, \"content\": {}, \"special\": {special}}}",
                256 + i,
                crate::json::quote_json(content)
            )
        })
        .collect();
    let json = format!(
        r#"{{"model": {{"type": "BPE", "vocab": {{{}}}, "merges": []}}, "added_tokens": [{}],
            "pre_tokenizer": {{"type": "Sequence", "pretokenizers": [
              {{"type": "Split", "pattern": {{"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{{L}}\\p{{N}}]?\\p{{L}}+|\\p{{N}}| ?[^\\s\\p{{L}}\\p{{N}}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}}}}]}}}}"#,
        vocab.join(", "),
        added.join(", ")
    );
    Tokenizer::from_json_str(&json).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::JsonValue;

    fn synthetic_byte_level_tokenizer() -> Tokenizer {
        let (b2cp, _) = byte_level_tables();
        let mut vocab_entries = Vec::new();
        for (b, &cp) in b2cp.iter().enumerate() {
            let ch = char::from_u32(cp).unwrap();
            let esc = match ch {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                _ => ch.to_string(),
            };
            vocab_entries.push(format!("\"{esc}\": {b}"));
        }
        // Add merged tokens: "he"=256, "ll"=257, "he"+"ll"="hell"=258, "hell"+"o"="hello"=259
        vocab_entries.push("\"he\": 256".into());
        vocab_entries.push("\"ll\": 257".into());
        vocab_entries.push("\"hell\": 258".into());
        vocab_entries.push("\"hello\": 259".into());

        let json = format!(
            r#"{{
              "model": {{
                "type": "BPE",
                "vocab": {{{}}},
                "merges": [["h","e"], ["l","l"], ["he","ll"], ["hell","o"]]
              }},
              "added_tokens": [
                {{"id": 260, "content": "<|im_start|>", "special": true}},
                {{"id": 261, "content": "<|im_end|>", "special": true}}
              ],
              "pre_tokenizer": {{
                "type": "Sequence",
                "pretokenizers": [
                  {{"type": "Split", "pattern": {{"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{{L}}\\p{{N}}]?\\p{{L}}+|\\p{{N}}| ?[^\\s\\p{{L}}\\p{{N}}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}}}}
                ]
              }}
            }}"#,
            vocab_entries.join(", ")
        );
        Tokenizer::from_json_str(&json).unwrap()
    }

    #[test]
    fn bpe_merges_and_special_tokens_roundtrip() {
        let tok = synthetic_byte_level_tokenizer();
        let ids = tok.encode("<|im_start|>hello<|im_end|>", true);
        assert_eq!(ids, vec![260, 259, 261]);
        assert_eq!(tok.decode(&ids, false), "<|im_start|>hello<|im_end|>");
        assert_eq!(tok.decode(&ids, true), "hello");
    }

    #[test]
    fn gemma4_chat_template_forms() {
        assert_eq!(
            format_gemma4_chat(" Hi there \n", None, false, false),
            "<bos><|turn>user\nHi there<turn|>\n<|turn>model\n"
        );
        assert_eq!(
            format_gemma4_chat("Hi", None, false, true),
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
        assert_eq!(
            format_gemma4_chat("Hi", Some("Be brief."), true, true),
            "<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        );
        assert!(gemma4_template_closes_thought(
            "{{- '<|turn>model\\n' -}}{%- if not enable_thinking -%}{{- '<|channel>thought\\n<channel|>' -}}"
        ));
        assert!(!gemma4_template_closes_thought("{{- '<|turn>model\\n' -}}"));
    }

    /// The letter-case pre-tokenizer's split of `seg`.
    fn letter_case_split(seg: &str) -> Vec<&str> {
        let chars: Vec<char> = seg.chars().collect();
        let offs: Vec<usize> = seg
            .char_indices()
            .map(|(b, _)| b)
            .chain([seg.len()])
            .collect();
        let cls: Vec<CharClass> = chars.iter().map(|&c| char_class(c)).collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let e = match_letter_case_at(&chars, &cls, i);
            assert!(e > i);
            out.push(&seg[offs[i]..offs[e]]);
            i = e;
        }
        out
    }

    fn inkling2_cases() -> JsonValue {
        parse_json(include_str!("../tests/data/tokenizer/inkling2_cases.json")).unwrap()
    }

    #[test]
    fn letter_case_pieces_match_the_reference() {
        // the reference pre-tokenizer's pieces are byte-level text: map them back to bytes
        let (b2cp, cp2b) = byte_level_tables();
        let raw = |piece: &str| -> Vec<u8> { piece.chars().map(|c| cp2b[&(c as u32)]).collect() };
        let level = |bytes: &[u8]| -> String {
            bytes
                .iter()
                .map(|&b| char::from_u32(b2cp[b as usize]).unwrap())
                .collect()
        };
        let cases = inkling2_cases();
        let mut n = 0;
        for case in cases.get("encode").unwrap().as_array().unwrap() {
            for seg in case.get("pieces").unwrap().as_array().unwrap() {
                let want: Vec<&str> = seg
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p.as_str().unwrap())
                    .collect();
                let bytes: Vec<u8> = want.iter().flat_map(|p| raw(p)).collect();
                let text = String::from_utf8(bytes).unwrap();
                let got: Vec<String> = letter_case_split(&text)
                    .iter()
                    .map(|p| level(p.as_bytes()))
                    .collect();
                assert_eq!(got, want, "{text:?}");
                n += 1;
            }
        }
        assert!(n > 300, "{n} segments");
        // the classes behind the pattern, spot-checked
        use CharClass::*;
        for (c, k) in [
            ('a', Lower),
            ('A', Upper),
            ('\u{1C5}', Upper),
            ('\u{2B0}', Both),
            ('\u{4E00}', Both),
            ('\u{301}', Mark),
            ('7', Number),
            ('\u{2160}', Number),
            (' ', Space),
            ('\u{3000}', Space),
            ('\u{200B}', Other),
            ('!', Other),
            ('\u{10FFFF}', Other),
        ] {
            assert_eq!(char_class(c), k, "{c:?}");
        }
    }

    /// Set `GAUDIUM_INKLING2_DIR` to an Inkling 2 model directory (its `tokenizer.json`).
    #[test]
    #[ignore = "needs GAUDIUM_INKLING2_DIR with the model's tokenizer.json"]
    fn inkling2_ids_match_the_reference() {
        let dir = std::env::var("GAUDIUM_INKLING2_DIR").expect("GAUDIUM_INKLING2_DIR");
        let tok = Tokenizer::from_model_dir(Path::new(&dir)).unwrap();
        assert!(tok.ignore_merges && tok.pretok == Pretok::LetterCase);
        let ids_of = |v: &JsonValue| -> Vec<i32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect()
        };
        let cases = inkling2_cases();
        for case in cases.get("encode").unwrap().as_array().unwrap() {
            let text = case.get("text").unwrap().as_str().unwrap();
            let want = ids_of(case.get("ids").unwrap());
            assert_eq!(tok.encode(text, true), want, "{text:?}");
            assert_eq!(tok.decode(&want, false), text);
            // streaming: the pieces join to the text, never splitting a character
            let mut sd = StreamDecoder::new(&tok);
            let mut streamed = String::new();
            for &id in &want {
                streamed.push_str(&sd.push(id));
            }
            streamed.push_str(&sd.flush());
            assert_eq!(streamed, text);
        }
        for case in cases.get("decode").unwrap().as_array().unwrap() {
            let ids = ids_of(case.get("ids").unwrap());
            let skip = case.get("skip_special").unwrap().as_bool().unwrap();
            let want = case.get("text").unwrap().as_str().unwrap();
            assert_eq!(tok.decode(&ids, skip), want, "{ids:?}");
        }
    }

    #[test]
    fn stream_decoder_holds_incomplete_utf8() {
        let tok = synthetic_byte_level_tokenizer();
        // UTF-8 for U+1F600 (4 bytes: F0 9F 98 80)
        let bytes = "\u{1F600}".as_bytes();
        let ids: Vec<i32> = bytes.iter().map(|&b| b as i32).collect();
        let mut sd = StreamDecoder::new(&tok);
        assert_eq!(sd.push(ids[0]), "");
        assert_eq!(sd.push(ids[1]), "");
        assert_eq!(sd.push(ids[2]), "");
        assert_eq!(sd.push(ids[3]), "\u{1F600}");
        assert_eq!(sd.flush(), "");
    }
}
