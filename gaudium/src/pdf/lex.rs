//! PDF objects and their syntax (ISO 32000-1, section 7.2 and 7.3): the lexer and the object
//! parser shared by the file reader, content streams and CMaps.
//!
//! Every recursion is bounded through the verified core (`pdf_below`), and the parser never
//! indexes past its slice: a truncated or corrupt file ends in an `Err`, never in a panic.

use std::rc::Rc;

use super::{Fail, MAX_NEST};

/// A PDF object.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Obj {
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
    Name(Vec<u8>),
    Str(Vec<u8>),
    Array(Vec<Obj>),
    Dict(Dict),
    /// An indirect reference: object number and generation.
    Ref(u32, u16),
    /// A stream: its dictionary and its raw (still encoded) data.
    Stream(Rc<Stream>),
}

impl Default for Obj {
    fn default() -> Self {
        Obj::Null
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct Dict(pub Vec<(Vec<u8>, Obj)>);

#[derive(Debug, PartialEq)]
pub(crate) struct Stream {
    pub dict: Dict,
    pub data: Rc<[u8]>,
}

impl Dict {
    pub fn get(&self, key: &[u8]) -> Option<&Obj> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

impl Obj {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Obj::Int(i) => Some(*i),
            Obj::Real(r) if r.is_finite() && r.fract() == 0.0 && r.abs() < 9.0e15 => {
                Some(*r as i64)
            }
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Obj::Int(i) => Some(*i as f64),
            Obj::Real(r) if r.is_finite() => Some(*r),
            _ => None,
        }
    }

    pub fn as_name(&self) -> Option<&[u8]> {
        match self {
            Obj::Name(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&Dict> {
        match self {
            Obj::Dict(d) => Some(d),
            Obj::Stream(s) => Some(&s.dict),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Obj]> {
        match self {
            Obj::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// One token of a content stream or CMap: an operand or an operator.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Token {
    Obj(Obj),
    /// A bare keyword (an operator such as `Tj`, or `begincodespacerange`).
    Op(Vec<u8>),
}

pub(crate) fn is_white(b: u8) -> bool {
    matches!(b, 0 | 9 | 10 | 12 | 13 | 32)
}

pub(crate) fn is_delim(b: u8) -> bool {
    matches!(
        b,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

fn is_regular(b: u8) -> bool {
    !is_white(b) && !is_delim(b)
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// A cursor over PDF bytes.
pub(crate) struct Lexer<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(data: &'a [u8], pos: usize) -> Self {
        Lexer {
            data,
            pos: pos.min(data.len()),
        }
    }

    pub fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Skips white space and comments.
    pub fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if is_white(b) {
                self.pos += 1;
            } else if b == b'%' {
                while let Some(c) = self.peek() {
                    if c == b'\n' || c == b'\r' {
                        break;
                    }
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    /// The next run of regular characters (a keyword or a number), without consuming it.
    fn word_at(&self, pos: usize) -> &'a [u8] {
        let start = pos.min(self.data.len());
        let mut end = start;
        while end < self.data.len() && is_regular(self.data[end]) {
            end += 1;
        }
        &self.data[start..end]
    }

    /// Consumes `kw` if it is the next word.
    pub fn eat_keyword(&mut self, kw: &[u8]) -> bool {
        self.skip_ws();
        if self.word_at(self.pos) == kw {
            self.pos += kw.len();
            true
        } else {
            false
        }
    }

    /// An unsigned integer word at the cursor (used for `N G obj` headers and xref tables).
    pub fn read_uint(&mut self) -> Option<u64> {
        self.skip_ws();
        let w = self.word_at(self.pos);
        if w.is_empty() || w.len() > 19 || !w.iter().all(u8::is_ascii_digit) {
            return None;
        }
        let v = std::str::from_utf8(w).ok()?.parse().ok()?;
        self.pos += w.len();
        Some(v)
    }

    /// Parses one object. `depth`: the nesting it sits in.
    pub fn object(&mut self, depth: u64) -> Result<Obj, Fail> {
        match self.token(depth, false)? {
            Some(Token::Obj(o)) => Ok(o),
            Some(Token::Op(op)) => Err(Fail::damaged(format!(
                "unexpected keyword '{}'",
                String::from_utf8_lossy(&op)
            ))),
            None => Err(Fail::damaged("the file ends inside an object")),
        }
    }

    /// The next token. `ops`: bare keywords are operators (content streams, CMaps) instead of
    /// an error. `None` at the end of the data.
    pub fn token(&mut self, depth: u64, ops: bool) -> Result<Option<Token>, Fail> {
        if !crate::verified::pdf_below(depth, MAX_NEST) {
            return Err(Fail::damaged(format!(
                "objects nested more than {MAX_NEST} deep"
            )));
        }
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Ok(None);
        };
        let o = match b {
            b'[' => {
                self.pos += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_ws();
                    match self.peek() {
                        None => return Err(Fail::damaged("an array is not closed")),
                        Some(b']') => {
                            self.pos += 1;
                            break;
                        }
                        _ => match self.token(depth + 1, ops)? {
                            Some(Token::Obj(o)) => items.push(o),
                            // content streams may hold stray keywords inside arrays
                            Some(Token::Op(_)) if ops => {}
                            Some(Token::Op(op)) => {
                                return Err(Fail::damaged(format!(
                                    "unexpected keyword '{}' in an array",
                                    String::from_utf8_lossy(&op)
                                )))
                            }
                            None => return Err(Fail::damaged("an array is not closed")),
                        },
                    }
                    self.collapse_ref(&mut items);
                }
                Obj::Array(items)
            }
            b'<' if self.data.get(self.pos + 1) == Some(&b'<') => {
                self.pos += 2;
                Obj::Dict(self.dict_body(depth, ops)?)
            }
            b'<' => {
                self.pos += 1;
                Obj::Str(self.hex_string()?)
            }
            b'(' => {
                self.pos += 1;
                Obj::Str(self.literal_string()?)
            }
            b'/' => {
                self.pos += 1;
                Obj::Name(self.name())
            }
            b')' | b'>' | b']' | b'}' | b'{' => {
                self.pos += 1;
                if ops {
                    // PostScript procedures in CMaps and stray delimiters: not text
                    return Ok(Some(Token::Op(vec![b])));
                }
                return Err(Fail::damaged(format!("unexpected '{}'", b as char)));
            }
            _ => {
                let w = self.word_at(self.pos);
                if w.is_empty() {
                    // a lone delimiter not handled above (cannot happen) or a stray byte
                    self.pos += 1;
                    return Ok(Some(Token::Op(vec![b])));
                }
                self.pos += w.len();
                if let Some(n) = number(w) {
                    n
                } else {
                    match w {
                        b"true" => Obj::Bool(true),
                        b"false" => Obj::Bool(false),
                        b"null" => Obj::Null,
                        _ => return Ok(Some(Token::Op(w.to_vec()))),
                    }
                }
            }
        };
        Ok(Some(Token::Obj(o)))
    }

    /// `a b R` at the end of `items` becomes a reference (arrays and dictionaries).
    fn collapse_ref(&mut self, items: &mut Vec<Obj>) {
        // the `R` is seen as the next word: look ahead without consuming anything else
        let n = items.len();
        if n < 2 {
            return;
        }
        if let (Obj::Int(num), Obj::Int(gen)) = (&items[n - 2], &items[n - 1]) {
            let save = self.pos;
            self.skip_ws();
            if self.word_at(self.pos) == b"R" {
                if let (Ok(num), Ok(gen)) = (u32::try_from(*num), u16::try_from(*gen)) {
                    self.pos += 1;
                    items.truncate(n - 2);
                    items.push(Obj::Ref(num, gen));
                    return;
                }
            }
            self.pos = save;
        }
    }

    fn dict_body(&mut self, depth: u64, ops: bool) -> Result<Dict, Fail> {
        let mut entries: Vec<(Vec<u8>, Obj)> = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => return Err(Fail::damaged("a dictionary is not closed")),
                Some(b'>') if self.data.get(self.pos + 1) == Some(&b'>') => {
                    self.pos += 2;
                    break;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let key = self.name();
                    self.skip_ws();
                    if self.peek() == Some(b'>') && self.data.get(self.pos + 1) == Some(&b'>') {
                        entries.push((key, Obj::Null));
                        continue;
                    }
                    let mut v = match self.token(depth + 1, ops)? {
                        Some(Token::Obj(o)) => vec![o],
                        Some(Token::Op(_)) => vec![Obj::Null],
                        None => return Err(Fail::damaged("a dictionary is not closed")),
                    };
                    // a value `a b R`: read the generation and check for the `R`
                    if let Obj::Int(_) = v[0] {
                        let save = self.pos;
                        self.skip_ws();
                        let w = self.word_at(self.pos);
                        if !w.is_empty() && w.iter().all(u8::is_ascii_digit) {
                            if let Some(Token::Obj(g)) = self.token(depth + 1, ops)? {
                                v.push(g);
                                self.collapse_ref(&mut v);
                            }
                        }
                        if v.len() != 1 {
                            self.pos = save;
                            v.truncate(1);
                        }
                    }
                    let value = v.pop().unwrap_or(Obj::Null);
                    match entries.iter_mut().find(|(k, _)| *k == key) {
                        Some(e) => e.1 = value,
                        None => entries.push((key, value)),
                    }
                }
                Some(_) => {
                    if ops {
                        // a damaged dictionary in a content stream: skip a token
                        if self.token(depth + 1, ops)?.is_none() {
                            return Err(Fail::damaged("a dictionary is not closed"));
                        }
                    } else {
                        return Err(Fail::damaged("a dictionary key is not a name"));
                    }
                }
            }
        }
        Ok(Dict(entries))
    }

    fn name(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        while let Some(b) = self.peek() {
            if !is_regular(b) {
                break;
            }
            self.pos += 1;
            if b == b'#' {
                let h = self.data.get(self.pos).copied().and_then(hex_val);
                let l = self.data.get(self.pos + 1).copied().and_then(hex_val);
                if let (Some(h), Some(l)) = (h, l) {
                    out.push(h << 4 | l);
                    self.pos += 2;
                    continue;
                }
            }
            out.push(b);
        }
        out
    }

    fn hex_string(&mut self) -> Result<Vec<u8>, Fail> {
        let mut out = Vec::new();
        let mut hi: Option<u8> = None;
        loop {
            let Some(b) = self.peek() else {
                return Err(Fail::damaged("a hex string is not closed"));
            };
            self.pos += 1;
            if b == b'>' {
                break;
            }
            if let Some(v) = hex_val(b) {
                match hi.take() {
                    Some(h) => out.push(h << 4 | v),
                    None => hi = Some(v),
                }
            }
        }
        if let Some(h) = hi {
            out.push(h << 4);
        }
        Ok(out)
    }

    fn literal_string(&mut self) -> Result<Vec<u8>, Fail> {
        let mut out = Vec::new();
        let mut open = 1usize;
        loop {
            let Some(b) = self.peek() else {
                return Err(Fail::damaged("a string is not closed"));
            };
            self.pos += 1;
            match b {
                b'(' => {
                    open += 1;
                    out.push(b);
                }
                b')' => {
                    open -= 1;
                    if open == 0 {
                        break;
                    }
                    out.push(b);
                }
                b'\\' => {
                    let Some(e) = self.peek() else {
                        return Err(Fail::damaged("a string is not closed"));
                    };
                    self.pos += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'\r' => {
                            if self.peek() == Some(b'\n') {
                                self.pos += 1;
                            }
                        }
                        b'\n' => {}
                        b'0'..=b'7' => {
                            let mut v = u32::from(e - b'0');
                            for _ in 0..2 {
                                match self.peek() {
                                    Some(d @ b'0'..=b'7') => {
                                        v = v * 8 + u32::from(d - b'0');
                                        self.pos += 1;
                                    }
                                    _ => break,
                                }
                            }
                            out.push(v as u8);
                        }
                        other => out.push(other),
                    }
                }
                b'\r' => {
                    // an end of line in a string is a line feed, whatever its bytes
                    if self.peek() == Some(b'\n') {
                        self.pos += 1;
                    }
                    out.push(b'\n');
                }
                _ => out.push(b),
            }
        }
        Ok(out)
    }
}

/// A number word (`12`, `-3`, `.5`, `4.`, `+1.25`); `None` for anything else.
fn number(w: &[u8]) -> Option<Obj> {
    let first = *w.first()?;
    if !(first.is_ascii_digit() || matches!(first, b'+' | b'-' | b'.')) {
        return None;
    }
    let body = if matches!(first, b'+' | b'-') {
        &w[1..]
    } else {
        w
    };
    if body.is_empty() || !body.iter().all(|c| c.is_ascii_digit() || *c == b'.') {
        return None;
    }
    let dots = body.iter().filter(|c| **c == b'.').count();
    if dots > 1 || (dots == 1 && body.len() == 1) {
        return None;
    }
    let text = std::str::from_utf8(w).ok()?;
    if dots == 0 {
        match text.parse::<i64>() {
            Ok(i) => Some(Obj::Int(i)),
            // an integer too large for 64 bits: kept as a real
            Err(_) => text.parse::<f64>().ok().map(Obj::Real),
        }
    } else {
        let t = if text.ends_with('.') {
            format!("{text}0")
        } else if let Some(rest) = text.strip_prefix("-.") {
            format!("-0.{rest}")
        } else if let Some(rest) = text.strip_prefix("+.") {
            format!("0.{rest}")
        } else if text.starts_with('.') {
            format!("0{text}")
        } else {
            text.to_string()
        };
        t.parse::<f64>().ok().map(Obj::Real)
    }
}
