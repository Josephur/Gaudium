//! Minimal, dependency-free strict JSON parser preserving object key order.
//!
//! Used by `safetensors` header decoding and `model_spec` (`config.json`) parsing.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub offset: usize,
    pub message: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "json parse error at byte {}: {}",
            self.offset, self.message
        )
    }
}

impl std::error::Error for JsonError {}

impl JsonValue {
    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Int(n) => Some(*n),
            JsonValue::Float(f) if f.is_finite() && f.fract() == 0.0 => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_i64().and_then(|v| u64::try_from(v).ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Int(n) => Some(*n as f64),
            JsonValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

impl JsonValue {
    /// Compact JSON text (`,` and `:` without spaces, non-ASCII characters as they are).
    pub fn dump(&self) -> String {
        let mut o = String::new();
        self.dump_into(&mut o);
        o
    }

    pub fn dump_into(&self, o: &mut String) {
        match self {
            JsonValue::Null => o.push_str("null"),
            JsonValue::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
            JsonValue::Int(i) => o.push_str(&i.to_string()),
            // JSON has no NaN or infinity
            JsonValue::Float(f) if !f.is_finite() => o.push_str("null"),
            JsonValue::Float(f) => o.push_str(&format!("{f:?}")),
            JsonValue::Str(s) => {
                o.push('"');
                escape_json_into(o, s);
                o.push('"');
            }
            JsonValue::Array(items) => {
                o.push('[');
                for (i, x) in items.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    x.dump_into(o);
                }
                o.push(']');
            }
            JsonValue::Object(entries) => {
                o.push('{');
                for (i, (k, x)) in entries.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    o.push('"');
                    escape_json_into(o, k);
                    o.push_str("\":");
                    x.dump_into(o);
                }
                o.push('}');
            }
        }
    }
}

/// Appends the body of a JSON string: `"` `\\` and control characters escaped, the rest as is.
pub fn escape_json_into(o: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
}

/// A JSON string literal for `s`.
pub fn quote_json(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    escape_json_into(&mut o, s);
    o.push('"');
    o
}

pub fn parse_json(input: &str) -> Result<JsonValue, JsonError> {
    parse_json_bytes(input.as_bytes())
}

pub fn parse_json_bytes(bytes: &[u8]) -> Result<JsonValue, JsonError> {
    let mut p = Parser { b: bytes, pos: 0 };
    let v = p.parse_value(0)?;
    p.skip_ws();
    if p.pos != p.b.len() {
        return Err(p.err("trailing characters after JSON value"));
    }
    Ok(v)
}

const MAX_DEPTH: usize = 128;

struct Parser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: impl Into<String>) -> JsonError {
        JsonError {
            offset: self.pos,
            message: msg.into(),
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.b.len() {
            match self.b[self.pos] {
                b' ' | b'\n' | b'\r' | b'\t' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&mut self) -> Result<u8, JsonError> {
        self.skip_ws();
        self.b
            .get(self.pos)
            .copied()
            .ok_or_else(|| self.err("unexpected end of input"))
    }

    fn expect(&mut self, expected: u8) -> Result<(), JsonError> {
        let c = self.peek()?;
        if c != expected {
            return Err(self.err(format!(
                "expected '{}', found '{}'",
                expected as char, c as char
            )));
        }
        self.pos += 1;
        Ok(())
    }

    fn eat_literal(&mut self, lit: &[u8]) -> bool {
        if self.b[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, JsonError> {
        if self.pos + 4 > self.b.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let mut val = 0u16;
        for _ in 0..4 {
            let c = self.b[self.pos];
            self.pos += 1;
            val <<= 4;
            val |= match c {
                b'0'..=b'9' => (c - b'0') as u16,
                b'a'..=b'f' => (c - b'a' + 10) as u16,
                b'A'..=b'F' => (c - b'A' + 10) as u16,
                _ => return Err(self.err("invalid hex digit in \\u escape")),
            };
        }
        Ok(val)
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        while self.pos < self.b.len() {
            let c = self.b[self.pos];
            self.pos += 1;
            if c == b'"' {
                return Ok(out);
            }
            if c != b'\\' {
                if c < 0x80 {
                    out.push(c as char);
                } else {
                    // Decode UTF-8 multi-byte sequence from raw slice
                    let start = self.pos - 1;
                    while self.pos < self.b.len() && (self.b[self.pos] & 0xC0) == 0x80 {
                        self.pos += 1;
                    }
                    let s = std::str::from_utf8(&self.b[start..self.pos])
                        .map_err(|_| self.err("invalid UTF-8 in string"))?;
                    out.push_str(s);
                }
                continue;
            }
            if self.pos >= self.b.len() {
                return Err(self.err("unterminated escape sequence"));
            }
            let esc = self.b[self.pos];
            self.pos += 1;
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\x08'),
                b'f' => out.push('\x0C'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    let c1 = self.parse_hex4()?;
                    let cp = if (0xD800..=0xDBFF).contains(&c1) {
                        if !self.eat_literal(b"\\u") {
                            return Err(self.err("missing low surrogate after high surrogate"));
                        }
                        let c2 = self.parse_hex4()?;
                        if !(0xDC00..=0xDFFF).contains(&c2) {
                            return Err(self.err("invalid low surrogate"));
                        }
                        0x10000 + (((c1 as u32) - 0xD800) << 10) + ((c2 as u32) - 0xDC00)
                    } else {
                        c1 as u32
                    };
                    let ch = char::from_u32(cp)
                        .ok_or_else(|| self.err("invalid Unicode scalar value"))?;
                    out.push(ch);
                }
                _ => return Err(self.err(format!("invalid escape '\\{}'", esc as char))),
            }
        }
        Err(self.err("unterminated string"))
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        if self.pos < self.b.len() && self.b[self.pos] == b'-' {
            self.pos += 1;
        }
        let digits_start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == digits_start {
            return Err(self.err("expected digit in number"));
        }
        let mut is_float = false;
        if self.pos < self.b.len() && self.b[self.pos] == b'.' {
            is_float = true;
            self.pos += 1;
            let frac_start = self.pos;
            while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos == frac_start {
                return Err(self.err("expected digit after decimal point"));
            }
        }
        if self.pos < self.b.len() && matches!(self.b[self.pos], b'e' | b'E') {
            is_float = true;
            self.pos += 1;
            if self.pos < self.b.len() && matches!(self.b[self.pos], b'+' | b'-') {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err(self.err("expected digit in exponent"));
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.pos])
            .map_err(|_| self.err("invalid UTF-8 in number"))?;
        if !is_float {
            if let Ok(i) = s.parse::<i64>() {
                return Ok(JsonValue::Int(i));
            }
        }
        let f = s.parse::<f64>().map_err(|_| self.err("invalid float"))?;
        Ok(JsonValue::Float(f))
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        if depth > MAX_DEPTH {
            return Err(self.err("JSON nesting exceeds maximum depth"));
        }
        match self.peek()? {
            b'{' => {
                self.pos += 1;
                let mut entries = Vec::new();
                if self.peek()? == b'}' {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                loop {
                    let k = self.parse_string()?;
                    self.expect(b':')?;
                    let v = self.parse_value(depth + 1)?;
                    entries.push((k, v));
                    match self.peek()? {
                        b',' => self.pos += 1,
                        b'}' => {
                            self.pos += 1;
                            return Ok(JsonValue::Object(entries));
                        }
                        _ => return Err(self.err("expected ',' or '}' in object")),
                    }
                }
            }
            b'[' => {
                self.pos += 1;
                let mut items = Vec::new();
                if self.peek()? == b']' {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                loop {
                    items.push(self.parse_value(depth + 1)?);
                    match self.peek()? {
                        b',' => self.pos += 1,
                        b']' => {
                            self.pos += 1;
                            return Ok(JsonValue::Array(items));
                        }
                        _ => return Err(self.err("expected ',' or ']' in array")),
                    }
                }
            }
            b'"' => Ok(JsonValue::Str(self.parse_string()?)),
            b't' if self.eat_literal(b"true") => Ok(JsonValue::Bool(true)),
            b'f' if self.eat_literal(b"false") => Ok(JsonValue::Bool(false)),
            b'n' if self.eat_literal(b"null") => Ok(JsonValue::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            c => Err(self.err(format!("unexpected character '{}'", c as char))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_objects_numbers_escapes_and_surrogates() {
        let raw = r#"{"a": 42, "b": -3.5e2, "c": [true, false, null], "s": "hi\n\uD83D\uDE80"}"#;
        let v = parse_json(raw).unwrap();
        assert_eq!(v.get("a").and_then(JsonValue::as_i64), Some(42));
        assert_eq!(v.get("b").and_then(JsonValue::as_f64), Some(-350.0));
        assert_eq!(v.get("s").and_then(JsonValue::as_str), Some("hi\n🚀"));
        let arr = v.get("c").and_then(JsonValue::as_array).unwrap();
        assert_eq!(
            arr,
            &[
                JsonValue::Bool(true),
                JsonValue::Bool(false),
                JsonValue::Null
            ]
        );
    }

    #[test]
    fn dump_writes_compact_json_that_reads_back() {
        let raw = r#"{"a":[1,-2.5,1e-7,true,null],"s":"q\"\\\n\u0001\u00fc","o":{}}"#;
        let v = parse_json(raw).unwrap();
        assert_eq!(
            v.dump(),
            "{\"a\":[1,-2.5,1e-7,true,null],\"s\":\"q\\\"\\\\\\n\\u0001\u{fc}\",\"o\":{}}"
        );
        assert_eq!(parse_json(&v.dump()).unwrap(), v);
        assert_eq!(quote_json("a\tb"), "\"a\\tb\"");
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(parse_json("{1: 2}").is_err());
        assert!(parse_json("{\"a\": 1} trailing").is_err());
        assert!(parse_json("[1, 2").is_err());
    }
}
