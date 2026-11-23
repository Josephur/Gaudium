//! A compact Jinja2 interpreter for model chat templates.
//!
//! It renders a model's own `chat_template` the way the reference tokenizer library does:
//! `trim_blocks` and `lstrip_blocks` on, loop controls, a `tojson` that writes
//! `json.dumps(..., ensure_ascii=False)`, the globals `raise_exception`, `strftime_now`,
//! `namespace`, `range` and `dict`, and Python value semantics (`str()` printing, truthiness,
//! `==`, `in`, code-point string indexing and slicing, Python float text).
//!
//! The parsed [`Template`] is plain data (shared between request threads); values exist only
//! while one render runs.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use crate::json::JsonValue;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The template text is not valid (or uses a construct this interpreter does not have).
    Syntax(String),
    /// `raise_exception(...)` in the template: the message is the template author's.
    Raised(String),
    /// A render-time failure (an operation on values that do not support it).
    Runtime(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Syntax(m) => write!(f, "template syntax error: {m}"),
            Error::Raised(m) => write!(f, "{m}"),
            Error::Runtime(m) => write!(f, "template error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type R<T> = Result<T, Error>;

fn rt<T>(m: impl Into<String>) -> R<T> {
    Err(Error::Runtime(m.into()))
}

fn syn<T>(m: impl Into<String>) -> R<T> {
    Err(Error::Syntax(m.into()))
}

type Pairs = Vec<(String, Value)>;

#[derive(Clone)]
pub enum Value {
    Undefined,
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    List(Rc<Vec<Value>>),
    /// Insertion-ordered mapping with string keys.
    Dict(Rc<Pairs>),
    Namespace(Rc<RefCell<Pairs>>),
    Macro(Arc<MacroDef>),
    Global(&'static str),
}

impl Value {
    pub fn str(s: impl AsRef<str>) -> Value {
        Value::Str(Rc::from(s.as_ref()))
    }

    fn list(v: Vec<Value>) -> Value {
        Value::List(Rc::new(v))
    }

    pub fn from_json(j: &JsonValue) -> Value {
        match j {
            JsonValue::Null => Value::None,
            JsonValue::Bool(b) => Value::Bool(*b),
            JsonValue::Int(i) => Value::Int(*i),
            JsonValue::Float(f) => Value::Float(*f),
            JsonValue::Str(s) => Value::str(s),
            JsonValue::Array(a) => Value::list(a.iter().map(Value::from_json).collect()),
            JsonValue::Object(o) => Value::Dict(Rc::new(
                o.iter()
                    .map(|(k, v)| (k.clone(), Value::from_json(v)))
                    .collect(),
            )),
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Value::Undefined | Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Dict(d) => !d.is_empty(),
            Value::Namespace(_) | Value::Macro(_) | Value::Global(_) => true,
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Value::Undefined => "undefined",
            Value::None => "none",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Namespace(_) => "namespace",
            Value::Macro(_) | Value::Global(_) => "function",
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            Value::Bool(b) => Some(i64::from(*b)),
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Python `str()`; an undefined value prints as nothing.
    fn to_text(&self) -> String {
        match self {
            Value::Undefined => String::new(),
            Value::Str(s) => s.to_string(),
            other => other.repr(),
        }
    }

    /// Python `repr()`.
    fn repr(&self) -> String {
        match self {
            Value::Undefined => "Undefined".into(),
            Value::None => "None".into(),
            Value::Bool(true) => "True".into(),
            Value::Bool(false) => "False".into(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => python_float(*f),
            Value::Str(s) => python_str_repr(s),
            Value::List(l) => format!(
                "[{}]",
                l.iter().map(Value::repr).collect::<Vec<_>>().join(", ")
            ),
            Value::Dict(d) => format!(
                "{{{}}}",
                d.iter()
                    .map(|(k, v)| format!("{}: {}", python_str_repr(k), v.repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Value::Namespace(_) => "<Namespace>".into(),
            Value::Macro(m) => format!("<macro {}>", m.name),
            Value::Global(n) => format!("<function {n}>"),
        }
    }

    fn get_item(&self, key: &str) -> Option<Value> {
        match self {
            Value::Dict(d) => d.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone()),
            Value::Namespace(n) => n
                .borrow()
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone()),
            _ => None,
        }
    }

    /// The values a `for` loop (or a sequence filter) walks.
    fn iterate(&self) -> R<Vec<Value>> {
        Ok(match self {
            Value::Undefined => Vec::new(),
            Value::List(l) => l.as_ref().clone(),
            Value::Dict(d) => d.iter().map(|(k, _)| Value::str(k)).collect(),
            Value::Str(s) => s.chars().map(|c| Value::str(c.to_string())).collect(),
            other => return rt(format!("'{}' object is not iterable", other.type_name())),
        })
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::None, Value::None) => true,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| values_equal(p, q))
        }
        (Value::Dict(x), Value::Dict(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| {
                    y.iter()
                        .find(|(k2, _)| k2 == k)
                        .is_some_and(|(_, v2)| values_equal(v, v2))
                })
        }
        (Value::Namespace(x), Value::Namespace(y)) => Rc::ptr_eq(x, y),
        _ => match (a.as_int(), b.as_int()) {
            (Some(x), Some(y)) => x == y,
            _ => match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            },
        },
    }
}

fn compare(a: &Value, b: &Value) -> R<std::cmp::Ordering> {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => Ok(x.cmp(y)),
        (Value::List(x), Value::List(y)) => {
            for (p, q) in x.iter().zip(y.iter()) {
                let o = compare(p, q)?;
                if o != Ordering::Equal {
                    return Ok(o);
                }
            }
            Ok(x.len().cmp(&y.len()))
        }
        _ => match (a.as_int(), b.as_int()) {
            (Some(x), Some(y)) => Ok(x.cmp(&y)),
            _ => match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => Ok(x.partial_cmp(&y).unwrap_or(Ordering::Equal)),
                _ => rt(format!(
                    "'<' not supported between '{}' and '{}'",
                    a.type_name(),
                    b.type_name()
                )),
            },
        },
    }
}

/// Python's `repr(float)`: the shortest text that reads back, `1e+16` style exponents.
pub fn python_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{f:?}");
    match s.split_once('e') {
        None => s,
        Some((mant, exp)) => {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp),
            };
            format!("{mant}e{sign}{digits:0>2}")
        }
    }
}

fn python_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut o = String::with_capacity(s.len() + 2);
    o.push(quote);
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == quote => {
                o.push('\\');
                o.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                o.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => o.push(c),
        }
    }
    o.push(quote);
    o
}

/// The arguments of Python's `json.dumps` that `tojson` passes on.
#[derive(Clone)]
struct JsonOpts {
    indent: Option<usize>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
    ensure_ascii: bool,
}

impl JsonOpts {
    /// `json.dumps(v, ensure_ascii=False, indent=indent)`.
    fn new(indent: Option<usize>) -> Self {
        Self {
            item_sep: if indent.is_some() { "," } else { ", " }.into(),
            key_sep: ": ".into(),
            indent,
            sort_keys: false,
            ensure_ascii: false,
        }
    }
}

/// A JSON string literal as Python writes it (`ensure_ascii`: every non-ASCII character as
/// `\uXXXX`, UTF-16 surrogate pairs above the BMP).
fn json_str(o: &mut String, s: &str, ensure_ascii: bool) {
    o.push('"');
    if ensure_ascii && !s.is_ascii() {
        let mut ascii = String::new();
        for c in s.chars() {
            if c.is_ascii() {
                ascii.push(c);
                continue;
            }
            crate::json::escape_json_into(o, &ascii);
            ascii.clear();
            let mut units = [0u16; 2];
            for u in c.encode_utf16(&mut units) {
                o.push_str(&format!("\\u{u:04x}"));
            }
        }
        crate::json::escape_json_into(o, &ascii);
    } else {
        crate::json::escape_json_into(o, s);
    }
    o.push('"');
}

/// Python `json.dumps(v, ...)` with the options of [`JsonOpts`].
fn json_dumps(v: &Value, opts: &JsonOpts, level: usize, o: &mut String) -> R<()> {
    let indent = opts.indent;
    let newline = |o: &mut String, level: usize| {
        if let Some(n) = indent {
            o.push('\n');
            o.push_str(&" ".repeat(n * level));
        }
    };
    let item_sep = opts.item_sep.as_str();
    match v {
        Value::None => o.push_str("null"),
        Value::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => o.push_str(&i.to_string()),
        Value::Float(f) => o.push_str(&match *f {
            x if x.is_nan() => "NaN".to_string(),
            x if x.is_infinite() => if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string(),
            x => python_float(x),
        }),
        Value::Str(s) => json_str(o, s, opts.ensure_ascii),
        Value::List(l) => {
            if l.is_empty() {
                o.push_str("[]");
                return Ok(());
            }
            o.push('[');
            for (i, x) in l.iter().enumerate() {
                if i > 0 {
                    o.push_str(item_sep);
                }
                newline(o, level + 1);
                json_dumps(x, opts, level + 1, o)?;
            }
            newline(o, level);
            o.push(']');
        }
        Value::Dict(_) | Value::Namespace(_) => {
            let mut pairs: Pairs = match v {
                Value::Dict(d) => d.as_ref().clone(),
                Value::Namespace(n) => n.borrow().clone(),
                _ => unreachable!(),
            };
            if opts.sort_keys {
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
            }
            if pairs.is_empty() {
                o.push_str("{}");
                return Ok(());
            }
            o.push('{');
            for (i, (k, x)) in pairs.iter().enumerate() {
                if i > 0 {
                    o.push_str(item_sep);
                }
                newline(o, level + 1);
                json_str(o, k, opts.ensure_ascii);
                o.push_str(&opts.key_sep);
                json_dumps(x, opts, level + 1, o)?;
            }
            newline(o, level);
            o.push('}');
        }
        Value::Undefined => return rt("an undefined value is not JSON serializable"),
        Value::Macro(_) | Value::Global(_) => return rt("a function is not JSON serializable"),
    }
    Ok(())
}

// ------------------------------------------------------------------------------------- lexer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Name(String),
    Str(String),
    Int(i64),
    Float(f64),
    Op(&'static str),
}

enum Piece {
    Text(String),
    Var(Vec<Tok>),
    Block(Vec<Tok>),
}

const OPS: &[&str] = &[
    "**", "//", "==", "!=", "<=", ">=", "(", ")", "[", "]", "{", "}", ",", ":", ".", "|", "~", "+",
    "-", "*", "/", "%", "<", ">", "=",
];

/// Lexes the inside of a `{{ }}` / `{% %}` tag starting at `pos`; returns the tokens, the
/// position after the closing delimiter and whether the close asked to strip what follows.
fn lex_tag(src: &str, mut pos: usize, close: &str) -> R<(Vec<Tok>, usize, bool)> {
    let b = src.as_bytes();
    let mut toks = Vec::new();
    let mut depth = 0usize;
    loop {
        while pos < b.len() && b[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= b.len() {
            return syn(format!("unclosed tag (missing '{close}')"));
        }
        let rest = &src[pos..];
        if depth == 0 {
            if rest.starts_with(close) {
                return Ok((toks, pos + close.len(), false));
            }
            if (rest.starts_with('-') || rest.starts_with('+')) && rest[1..].starts_with(close) {
                return Ok((toks, pos + 1 + close.len(), rest.starts_with('-')));
            }
        }
        let c = b[pos];
        if c == b'\'' || c == b'"' {
            let mut s = String::new();
            let mut it = rest[1..].char_indices();
            let mut end = None;
            while let Some((i, ch)) = it.next() {
                if ch as u32 == u32::from(c) {
                    end = Some(i);
                    break;
                }
                if ch != '\\' {
                    s.push(ch);
                    continue;
                }
                let Some((_, e)) = it.next() else { break };
                match e {
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    '0' => s.push('\0'),
                    '\\' | '\'' | '"' => s.push(e),
                    'x' | 'u' => {
                        let n = if e == 'x' { 2 } else { 4 };
                        let hex: String = it.clone().take(n).map(|(_, h)| h).collect();
                        match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                            Some(ch) if hex.len() == n => {
                                s.push(ch);
                                for _ in 0..n {
                                    it.next();
                                }
                            }
                            _ => {
                                s.push('\\');
                                s.push(e);
                            }
                        }
                    }
                    other => {
                        s.push('\\');
                        s.push(other);
                    }
                }
            }
            let Some(end) = end else {
                return syn("unterminated string literal");
            };
            pos += 1 + end + 1;
            toks.push(Tok::Str(s));
        } else if c.is_ascii_digit() {
            let start = pos;
            while pos < b.len() && b[pos].is_ascii_digit() {
                pos += 1;
            }
            let mut float = false;
            if pos + 1 < b.len() && b[pos] == b'.' && b[pos + 1].is_ascii_digit() {
                float = true;
                pos += 1;
                while pos < b.len() && b[pos].is_ascii_digit() {
                    pos += 1;
                }
            }
            if pos < b.len() && (b[pos] == b'e' || b[pos] == b'E') {
                let mut q = pos + 1;
                if q < b.len() && (b[q] == b'+' || b[q] == b'-') {
                    q += 1;
                }
                if q < b.len() && b[q].is_ascii_digit() {
                    float = true;
                    while q < b.len() && b[q].is_ascii_digit() {
                        q += 1;
                    }
                    pos = q;
                }
            }
            let text = &src[start..pos];
            toks.push(if float {
                Tok::Float(
                    text.parse()
                        .map_err(|_| Error::Syntax(format!("bad number '{text}'")))?,
                )
            } else {
                Tok::Int(
                    text.parse()
                        .map_err(|_| Error::Syntax(format!("bad number '{text}'")))?,
                )
            });
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = pos;
            while pos < b.len() && (b[pos].is_ascii_alphanumeric() || b[pos] == b'_') {
                pos += 1;
            }
            toks.push(Tok::Name(src[start..pos].to_string()));
        } else {
            let Some(op) = OPS.iter().find(|op| rest.starts_with(**op)) else {
                let ch = rest.chars().next().unwrap_or(' ');
                return syn(format!("unexpected character '{ch}'"));
            };
            match *op {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" => depth = depth.saturating_sub(1),
                _ => {}
            }
            pos += op.len();
            toks.push(Tok::Op(op));
        }
    }
}

/// Splits a template into text and tags, applying whitespace control (`-`, `trim_blocks`,
/// `lstrip_blocks`).
fn lex(src: &str) -> R<Vec<Piece>> {
    let b = src.as_bytes();
    let mut pieces = Vec::new();
    let mut pos = 0usize;
    let mut text_start = 0usize;
    while pos < b.len() {
        let Some(off) = src[pos..].find('{') else {
            break;
        };
        let open = pos + off;
        let kind = match b.get(open + 1) {
            Some(b'{') => 0,
            Some(b'%') => 1,
            Some(b'#') => 2,
            _ => {
                pos = open + 1;
                continue;
            }
        };
        let mut inner = open + 2;
        let sign = b.get(inner).copied().filter(|c| *c == b'-' || *c == b'+');
        // `{{-1}}` style: a sign directly before a digit in an output tag is still a control mark,
        // as in Jinja's own lexer.
        if sign.is_some() {
            inner += 1;
        }
        let mut text = src[text_start..open].to_string();
        if sign == Some(b'-') {
            text.truncate(text.trim_end().len());
        } else if kind != 0 && sign.is_none() {
            // lstrip_blocks: spaces and tabs between a line start and a block tag or comment
            let line = text.rfind('\n').map_or(0, |i| i + 1);
            let at_line_start = line > 0 || text_start == 0 || b[text_start - 1] == b'\n';
            if at_line_start && text[line..].bytes().all(|c| c == b' ' || c == b'\t') {
                text.truncate(line);
            }
        }
        if !text.is_empty() {
            pieces.push(Piece::Text(text));
        }
        let (after, strip) = if kind == 2 {
            let Some(end) = src[inner..].find("#}") else {
                return syn("unclosed comment");
            };
            let end = inner + end;
            (end + 2, end > inner && b[end - 1] == b'-')
        } else {
            let (toks, after, strip) = lex_tag(src, inner, if kind == 0 { "}}" } else { "%}" })?;
            pieces.push(if kind == 0 {
                Piece::Var(toks)
            } else {
                Piece::Block(toks)
            });
            (after, strip)
        };
        pos = after;
        if strip {
            while pos < b.len() && b[pos].is_ascii_whitespace() {
                pos += 1;
            }
        } else if kind != 0 {
            // trim_blocks: the newline right after a block tag or comment
            if b.get(pos) == Some(&b'\n') {
                pos += 1;
            } else if b.get(pos) == Some(&b'\r') && b.get(pos + 1) == Some(&b'\n') {
                pos += 2;
            }
        }
        text_start = pos;
    }
    if text_start < b.len() {
        pieces.push(Piece::Text(src[text_start..].to_string()));
    }
    Ok(pieces)
}

// --------------------------------------------------------------------------------------- AST

#[derive(Debug, Clone)]
enum Lit {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

type Kwargs = Vec<(String, Expr)>;

#[derive(Debug, Clone)]
enum Expr {
    Lit(Lit),
    Name(String),
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, [Option<Box<Expr>>; 3]),
    Call(Box<Expr>, Vec<Expr>, Kwargs),
    Filter(Box<Expr>, String, Vec<Expr>, Kwargs),
    Test(Box<Expr>, String, Vec<Expr>, bool),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Binary(&'static str, Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
}

#[derive(Debug)]
enum Node {
    Text(String),
    Output(Expr),
    If(Vec<(Expr, Vec<Node>)>, Vec<Node>),
    For {
        targets: Vec<String>,
        iter: Expr,
        cond: Option<Expr>,
        body: Vec<Node>,
        orelse: Vec<Node>,
    },
    Set(Vec<String>, Expr),
    SetAttr(String, String, Expr),
    SetBlock(String, Vec<Node>),
    Macro(Arc<MacroDef>),
    Do(Expr),
    /// A run of nodes rendered in place (`{% generation %}`: the reference marks the assistant's
    /// part of the text with it and renders the body unchanged).
    Seq(Vec<Node>),
    Break,
    Continue,
}

#[derive(Debug)]
pub struct MacroDef {
    name: String,
    params: Vec<(String, Option<Expr>)>,
    body: Vec<Node>,
}

// ------------------------------------------------------------------------------------ parser

struct ExprParser<'a> {
    t: &'a [Tok],
    i: usize,
}

impl<'a> ExprParser<'a> {
    fn peek(&self) -> Option<&'a Tok> {
        self.t.get(self.i)
    }

    fn is_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if *o == op)
    }

    fn is_name(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Tok::Name(n)) if n == name)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        let hit = self.is_op(op);
        if hit {
            self.i += 1;
        }
        hit
    }

    fn eat_name(&mut self, name: &str) -> bool {
        let hit = self.is_name(name);
        if hit {
            self.i += 1;
        }
        hit
    }

    fn expect_op(&mut self, op: &str) -> R<()> {
        if self.eat_op(op) {
            Ok(())
        } else {
            syn(format!("expected '{op}', found {}", self.describe()))
        }
    }

    fn name(&mut self) -> R<String> {
        match self.peek() {
            Some(Tok::Name(n)) => {
                self.i += 1;
                Ok(n.clone())
            }
            _ => syn(format!("expected a name, found {}", self.describe())),
        }
    }

    fn describe(&self) -> String {
        match self.peek() {
            None => "the end of the tag".into(),
            Some(Tok::Name(n)) => format!("'{n}'"),
            Some(Tok::Str(s)) => format!("string '{s}'"),
            Some(Tok::Int(i)) => i.to_string(),
            Some(Tok::Float(f)) => f.to_string(),
            Some(Tok::Op(o)) => format!("'{o}'"),
        }
    }

    fn done(&self) -> R<()> {
        if self.i == self.t.len() {
            Ok(())
        } else {
            syn(format!("unexpected {}", self.describe()))
        }
    }

    fn expr(&mut self) -> R<Expr> {
        let mut e = self.or()?;
        while self.eat_name("if") {
            let test = self.or()?;
            let other = if self.eat_name("else") {
                Some(Box::new(self.expr()?))
            } else {
                None
            };
            e = Expr::Cond(Box::new(test), Box::new(e), other);
        }
        Ok(e)
    }

    fn or(&mut self) -> R<Expr> {
        let mut e = self.and()?;
        while self.eat_name("or") {
            e = Expr::Binary("or", Box::new(e), Box::new(self.and()?));
        }
        Ok(e)
    }

    fn and(&mut self) -> R<Expr> {
        let mut e = self.not()?;
        while self.eat_name("and") {
            e = Expr::Binary("and", Box::new(e), Box::new(self.not()?));
        }
        Ok(e)
    }

    fn not(&mut self) -> R<Expr> {
        if self.eat_name("not") {
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        self.compare()
    }

    fn compare(&mut self) -> R<Expr> {
        let mut e = self.math1()?;
        loop {
            let op = if let Some(op) = ["==", "!=", "<=", ">=", "<", ">"]
                .into_iter()
                .find(|op| self.is_op(op))
            {
                self.i += 1;
                op
            } else if self.eat_name("in") {
                "in"
            } else if self.is_name("not")
                && matches!(self.t.get(self.i + 1), Some(Tok::Name(n)) if n == "in")
            {
                self.i += 2;
                "notin"
            } else {
                return Ok(e);
            };
            e = Expr::Binary(op, Box::new(e), Box::new(self.math1()?));
        }
    }

    fn math1(&mut self) -> R<Expr> {
        let mut e = self.concat()?;
        while let Some(op) = ["+", "-"].into_iter().find(|op| self.is_op(op)) {
            self.i += 1;
            e = Expr::Binary(op, Box::new(e), Box::new(self.concat()?));
        }
        Ok(e)
    }

    fn concat(&mut self) -> R<Expr> {
        let mut e = self.math2()?;
        while self.eat_op("~") {
            e = Expr::Binary("~", Box::new(e), Box::new(self.math2()?));
        }
        Ok(e)
    }

    fn math2(&mut self) -> R<Expr> {
        let mut e = self.pow()?;
        while let Some(op) = ["*", "//", "/", "%"].into_iter().find(|op| self.is_op(op)) {
            self.i += 1;
            e = Expr::Binary(op, Box::new(e), Box::new(self.pow()?));
        }
        Ok(e)
    }

    fn pow(&mut self) -> R<Expr> {
        let mut e = self.unary(true)?;
        while self.eat_op("**") {
            e = Expr::Binary("**", Box::new(e), Box::new(self.unary(true)?));
        }
        Ok(e)
    }

    fn unary(&mut self, with_filter: bool) -> R<Expr> {
        let e = if self.eat_op("-") {
            Expr::Neg(Box::new(self.unary(false)?))
        } else if self.eat_op("+") {
            self.unary(false)?
        } else {
            self.primary()?
        };
        let e = self.postfix(e)?;
        if with_filter {
            self.filters(e)
        } else {
            Ok(e)
        }
    }

    fn primary(&mut self) -> R<Expr> {
        let Some(tok) = self.peek() else {
            return syn("unexpected end of an expression");
        };
        self.i += 1;
        Ok(match tok {
            Tok::Name(n) => match n.as_str() {
                "true" | "True" => Expr::Lit(Lit::Bool(true)),
                "false" | "False" => Expr::Lit(Lit::Bool(false)),
                "none" | "None" => Expr::Lit(Lit::None),
                _ => Expr::Name(n.clone()),
            },
            Tok::Str(s) => {
                // adjacent string literals join
                let mut s = s.clone();
                while let Some(Tok::Str(more)) = self.peek() {
                    s.push_str(more);
                    self.i += 1;
                }
                Expr::Lit(Lit::Str(s))
            }
            Tok::Int(i) => Expr::Lit(Lit::Int(*i)),
            Tok::Float(f) => Expr::Lit(Lit::Float(*f)),
            Tok::Op("(") => {
                if self.eat_op(")") {
                    return Ok(Expr::List(Vec::new()));
                }
                let first = self.expr()?;
                if self.is_op(",") {
                    let mut items = vec![first];
                    while self.eat_op(",") {
                        if self.is_op(")") {
                            break;
                        }
                        items.push(self.expr()?);
                    }
                    self.expect_op(")")?;
                    Expr::List(items)
                } else {
                    self.expect_op(")")?;
                    first
                }
            }
            Tok::Op("[") => {
                let mut items = Vec::new();
                while !self.is_op("]") {
                    items.push(self.expr()?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect_op("]")?;
                Expr::List(items)
            }
            Tok::Op("{") => {
                let mut items = Vec::new();
                while !self.is_op("}") {
                    let k = self.expr()?;
                    self.expect_op(":")?;
                    items.push((k, self.expr()?));
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect_op("}")?;
                Expr::Dict(items)
            }
            _ => {
                self.i -= 1;
                return syn(format!("unexpected {}", self.describe()));
            }
        })
    }

    fn call_args(&mut self) -> R<(Vec<Expr>, Kwargs)> {
        let mut args = Vec::new();
        let mut kwargs = Vec::new();
        while !self.is_op(")") {
            let is_kw = matches!(self.peek(), Some(Tok::Name(_)))
                && matches!(self.t.get(self.i + 1), Some(Tok::Op("=")));
            if is_kw {
                let k = self.name()?;
                self.i += 1;
                kwargs.push((k, self.expr()?));
            } else {
                args.push(self.expr()?);
            }
            if !self.eat_op(",") {
                break;
            }
        }
        self.expect_op(")")?;
        Ok((args, kwargs))
    }

    fn postfix(&mut self, mut e: Expr) -> R<Expr> {
        loop {
            if self.eat_op(".") {
                let attr = match self.peek() {
                    Some(Tok::Name(n)) => n.clone(),
                    Some(Tok::Int(i)) => {
                        let idx = Expr::Lit(Lit::Int(*i));
                        self.i += 1;
                        e = Expr::Index(Box::new(e), Box::new(idx));
                        continue;
                    }
                    _ => {
                        return syn(format!(
                            "expected an attribute name, found {}",
                            self.describe()
                        ))
                    }
                };
                self.i += 1;
                e = Expr::Attr(Box::new(e), attr);
            } else if self.eat_op("[") {
                let mut parts: [Option<Box<Expr>>; 3] = [None, None, None];
                let mut slice = false;
                let mut n = 0;
                loop {
                    if self.is_op("]") {
                        break;
                    }
                    if self.eat_op(":") {
                        slice = true;
                        n += 1;
                        if n > 2 {
                            return syn("too many ':' in a slice");
                        }
                        continue;
                    }
                    parts[n] = Some(Box::new(self.expr()?));
                }
                self.expect_op("]")?;
                e = if slice {
                    Expr::Slice(Box::new(e), parts)
                } else {
                    match parts[0].take() {
                        Some(idx) => Expr::Index(Box::new(e), idx),
                        None => return syn("empty subscript"),
                    }
                };
            } else if self.eat_op("(") {
                let (args, kwargs) = self.call_args()?;
                e = Expr::Call(Box::new(e), args, kwargs);
            } else {
                return Ok(e);
            }
        }
    }

    fn filters(&mut self, mut e: Expr) -> R<Expr> {
        loop {
            if self.eat_op("|") {
                let name = self.name()?;
                let (args, kwargs) = if self.eat_op("(") {
                    self.call_args()?
                } else {
                    (Vec::new(), Vec::new())
                };
                e = Expr::Filter(Box::new(e), name, args, kwargs);
            } else if self.is_name("is") {
                self.i += 1;
                let negated = self.eat_name("not");
                let name = self.name()?;
                let mut args = Vec::new();
                if self.eat_op("(") {
                    args = self.call_args()?.0;
                } else {
                    let takes_arg = match self.peek() {
                        Some(Tok::Name(n)) => !matches!(
                            n.as_str(),
                            "else" | "or" | "and" | "if" | "in" | "is" | "not" | "recursive"
                        ),
                        Some(Tok::Str(_) | Tok::Int(_) | Tok::Float(_)) => true,
                        Some(Tok::Op(o)) => matches!(*o, "[" | "{"),
                        None => false,
                    };
                    if takes_arg {
                        let a = self.primary()?;
                        args.push(self.postfix(a)?);
                    }
                }
                e = Expr::Test(Box::new(e), name, args, negated);
            } else if self.eat_op("(") {
                let (args, kwargs) = self.call_args()?;
                e = Expr::Call(Box::new(e), args, kwargs);
            } else {
                return Ok(e);
            }
        }
    }
}

/// The block tag that ended a run of nodes: its name and its remaining tokens.
type EndTag = Option<(String, Vec<Tok>)>;

struct TemplateParser {
    pieces: Vec<Piece>,
    i: usize,
}

impl TemplateParser {
    /// Parses nodes up to a block tag named in `ends` (returned with its remaining tokens) or
    /// the end of the template.
    fn until(&mut self, ends: &[&str]) -> R<(Vec<Node>, EndTag)> {
        let mut nodes = Vec::new();
        while self.i < self.pieces.len() {
            let piece = std::mem::replace(&mut self.pieces[self.i], Piece::Text(String::new()));
            self.i += 1;
            match piece {
                Piece::Text(t) => nodes.push(Node::Text(t)),
                Piece::Var(toks) => {
                    let mut p = ExprParser { t: &toks, i: 0 };
                    let e = p.expr()?;
                    p.done()?;
                    nodes.push(Node::Output(e));
                }
                Piece::Block(toks) => {
                    let Some(Tok::Name(kw)) = toks.first() else {
                        return syn("a block tag needs a statement name");
                    };
                    if ends.contains(&kw.as_str()) {
                        return Ok((nodes, Some((kw.clone(), toks[1..].to_vec()))));
                    }
                    nodes.push(self.statement(kw, &toks[1..])?);
                }
            }
        }
        if ends.is_empty() {
            Ok((nodes, None))
        } else {
            syn(format!("missing '{}'", ends[ends.len() - 1]))
        }
    }

    fn statement(&mut self, kw: &str, toks: &[Tok]) -> R<Node> {
        let mut p = ExprParser { t: toks, i: 0 };
        match kw {
            "if" => {
                let mut arms = Vec::new();
                let mut cond = p.expr()?;
                p.done()?;
                loop {
                    let (body, end) = self.until(&["elif", "else", "endif"])?;
                    arms.push((cond, body));
                    let (name, rest) = end.expect("an end tag");
                    match name.as_str() {
                        "elif" => {
                            let mut q = ExprParser { t: &rest, i: 0 };
                            cond = q.expr()?;
                            q.done()?;
                        }
                        "else" => {
                            let (orelse, _) = self.until(&["endif"])?;
                            return Ok(Node::If(arms, orelse));
                        }
                        _ => return Ok(Node::If(arms, Vec::new())),
                    }
                }
            }
            "for" => {
                let mut targets = vec![p.name()?];
                while p.eat_op(",") {
                    targets.push(p.name()?);
                }
                if !p.eat_name("in") {
                    return syn("expected 'in' in a for statement");
                }
                let iter = p.or()?;
                let cond = if p.eat_name("if") {
                    Some(p.expr()?)
                } else {
                    None
                };
                p.eat_name("recursive");
                p.done()?;
                let (body, end) = self.until(&["else", "endfor"])?;
                let orelse = if end.is_some_and(|(n, _)| n == "else") {
                    self.until(&["endfor"])?.0
                } else {
                    Vec::new()
                };
                Ok(Node::For {
                    targets,
                    iter,
                    cond,
                    body,
                    orelse,
                })
            }
            "set" => {
                let first = p.name()?;
                if p.eat_op(".") {
                    let attr = p.name()?;
                    p.expect_op("=")?;
                    let value = p.expr()?;
                    p.done()?;
                    return Ok(Node::SetAttr(first, attr, value));
                }
                let mut targets = vec![first];
                while p.eat_op(",") {
                    targets.push(p.name()?);
                }
                if p.eat_op("=") {
                    let mut value = p.expr()?;
                    if p.is_op(",") {
                        let mut items = vec![value];
                        while p.eat_op(",") {
                            items.push(p.expr()?);
                        }
                        value = Expr::List(items);
                    }
                    p.done()?;
                    return Ok(Node::Set(targets, value));
                }
                p.done()?;
                if targets.len() != 1 {
                    return syn("a block set takes one name");
                }
                let (body, _) = self.until(&["endset"])?;
                Ok(Node::SetBlock(targets.remove(0), body))
            }
            "macro" => {
                let name = p.name()?;
                p.expect_op("(")?;
                let mut params = Vec::new();
                while !p.is_op(")") {
                    let param = p.name()?;
                    let default = if p.eat_op("=") { Some(p.expr()?) } else { None };
                    params.push((param, default));
                    if !p.eat_op(",") {
                        break;
                    }
                }
                p.expect_op(")")?;
                p.done()?;
                let (body, _) = self.until(&["endmacro"])?;
                Ok(Node::Macro(Arc::new(MacroDef { name, params, body })))
            }
            "do" => {
                let e = p.expr()?;
                p.done()?;
                Ok(Node::Do(e))
            }
            "generation" => {
                p.done()?;
                let (body, _) = self.until(&["endgeneration"])?;
                Ok(Node::Seq(body))
            }
            "break" => Ok(Node::Break),
            "continue" => Ok(Node::Continue),
            other => syn(format!("unknown statement '{other}'")),
        }
    }
}

/// A parsed chat template.
#[derive(Debug)]
pub struct Template {
    nodes: Vec<Node>,
}

impl Template {
    pub fn parse(source: &str) -> Result<Template, Error> {
        let mut p = TemplateParser {
            pieces: lex(source)?,
            i: 0,
        };
        let (nodes, _) = p.until(&[])?;
        Ok(Template { nodes })
    }

    /// Renders with `context` as the template's variables.
    pub fn render(&self, context: Vec<(String, Value)>) -> Result<String, Error> {
        let mut it = Interp {
            scopes: vec![context],
            depth: 0,
        };
        let mut out = String::new();
        it.exec(&self.nodes, &mut out)?;
        Ok(out)
    }
}

// ------------------------------------------------------------------------------- interpreter

enum Flow {
    Normal,
    Break,
    Continue,
}

const GLOBALS: &[&str] = &[
    "namespace",
    "range",
    "raise_exception",
    "strftime_now",
    "dict",
];
const MAX_DEPTH: usize = 96;

struct Interp {
    scopes: Vec<Pairs>,
    depth: usize,
}

struct Args {
    pos: Vec<Value>,
    kw: Pairs,
}

impl Args {
    /// Positional `i`, else keyword `name`.
    fn get(&self, i: usize, name: &str) -> Option<Value> {
        self.pos.get(i).cloned().or_else(|| {
            self.kw
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        })
    }

    fn text(&self, i: usize, name: &str) -> Option<String> {
        self.get(i, name).map(|v| v.to_text())
    }
}

impl Interp {
    fn lookup(&self, name: &str) -> Value {
        for frame in self.scopes.iter().rev() {
            if let Some((_, v)) = frame.iter().rev().find(|(k, _)| k == name) {
                return v.clone();
            }
        }
        match GLOBALS.iter().find(|g| **g == name) {
            Some(g) => Value::Global(g),
            None => Value::Undefined,
        }
    }

    fn assign(&mut self, name: &str, v: Value) {
        let frame = self.scopes.last_mut().expect("a scope");
        match frame.iter_mut().find(|(k, _)| k == name) {
            Some(slot) => slot.1 = v,
            None => frame.push((name.to_string(), v)),
        }
    }

    fn assign_targets(&mut self, targets: &[String], v: Value) -> R<()> {
        if let [one] = targets {
            self.assign(one, v);
            return Ok(());
        }
        let items = v.iterate()?;
        if items.len() != targets.len() {
            return rt(format!(
                "cannot unpack {} values into {} names",
                items.len(),
                targets.len()
            ));
        }
        for (t, x) in targets.iter().zip(items) {
            self.assign(t, x);
        }
        Ok(())
    }

    fn exec(&mut self, nodes: &[Node], out: &mut String) -> R<Flow> {
        for node in nodes {
            match node {
                Node::Text(t) => out.push_str(t),
                Node::Output(e) => out.push_str(&self.eval(e)?.to_text()),
                Node::If(arms, orelse) => {
                    let mut taken = false;
                    for (cond, body) in arms {
                        if self.eval(cond)?.truthy() {
                            taken = true;
                            match self.exec(body, out)? {
                                Flow::Normal => {}
                                flow => return Ok(flow),
                            }
                            break;
                        }
                    }
                    if !taken {
                        match self.exec(orelse, out)? {
                            Flow::Normal => {}
                            flow => return Ok(flow),
                        }
                    }
                }
                Node::For {
                    targets,
                    iter,
                    cond,
                    body,
                    orelse,
                } => self.exec_for(targets, iter, cond.as_ref(), body, orelse, out)?,
                Node::Set(targets, e) => {
                    let v = self.eval(e)?;
                    self.assign_targets(targets, v)?;
                }
                Node::SetAttr(ns, attr, e) => {
                    let v = self.eval(e)?;
                    match self.lookup(ns) {
                        Value::Namespace(n) => {
                            let mut n = n.borrow_mut();
                            match n.iter_mut().find(|(k, _)| k == attr) {
                                Some(slot) => slot.1 = v,
                                None => n.push((attr.clone(), v)),
                            }
                        }
                        _ => {
                            return rt(format!(
                                "cannot assign an attribute of '{ns}': not a namespace"
                            ))
                        }
                    }
                }
                Node::SetBlock(name, body) => {
                    let mut buf = String::new();
                    self.scopes.push(Vec::new());
                    let r = self.exec(body, &mut buf);
                    self.scopes.pop();
                    r?;
                    self.assign(name, Value::str(buf));
                }
                Node::Macro(m) => self.assign(&m.name, Value::Macro(Arc::clone(m))),
                Node::Do(e) => {
                    self.eval(e)?;
                }
                Node::Seq(body) => match self.exec(body, out)? {
                    Flow::Normal => {}
                    flow => return Ok(flow),
                },
                Node::Break => return Ok(Flow::Break),
                Node::Continue => return Ok(Flow::Continue),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_for(
        &mut self,
        targets: &[String],
        iter: &Expr,
        cond: Option<&Expr>,
        body: &[Node],
        orelse: &[Node],
        out: &mut String,
    ) -> R<()> {
        let seq = self.eval(iter)?;
        let mut items = match &seq {
            // a mapping walked with two names yields its pairs only through `items`; plain keys here
            Value::None => return rt("'none' object is not iterable"),
            other => other.iterate()?,
        };
        if let Some(c) = cond {
            let mut kept = Vec::with_capacity(items.len());
            for item in items {
                self.scopes.push(Vec::new());
                let ok = self
                    .assign_targets(targets, item.clone())
                    .and_then(|()| self.eval(c));
                self.scopes.pop();
                if ok?.truthy() {
                    kept.push(item);
                }
            }
            items = kept;
        }
        if items.is_empty() {
            self.exec(orelse, out)?;
            return Ok(());
        }
        let n = items.len();
        for i in 0..n {
            let int = |v: usize| Value::Int(v as i64);
            let lp: Pairs = vec![
                ("index".into(), int(i + 1)),
                ("index0".into(), int(i)),
                ("first".into(), Value::Bool(i == 0)),
                ("last".into(), Value::Bool(i + 1 == n)),
                ("length".into(), int(n)),
                ("revindex".into(), int(n - i)),
                ("revindex0".into(), int(n - i - 1)),
                (
                    "previtem".into(),
                    if i > 0 {
                        items[i - 1].clone()
                    } else {
                        Value::Undefined
                    },
                ),
                (
                    "nextitem".into(),
                    items.get(i + 1).cloned().unwrap_or(Value::Undefined),
                ),
            ];
            self.scopes
                .push(vec![("loop".into(), Value::Dict(Rc::new(lp)))]);
            let flow = self
                .assign_targets(targets, items[i].clone())
                .and_then(|()| self.exec(body, out));
            self.scopes.pop();
            if let Flow::Break = flow? {
                break;
            }
        }
        Ok(())
    }

    fn eval_args(&mut self, args: &[Expr], kwargs: &Kwargs) -> R<Args> {
        let mut a = Args {
            pos: Vec::with_capacity(args.len()),
            kw: Vec::with_capacity(kwargs.len()),
        };
        for e in args {
            a.pos.push(self.eval(e)?);
        }
        for (k, e) in kwargs {
            a.kw.push((k.clone(), self.eval(e)?));
        }
        Ok(a)
    }

    fn eval(&mut self, e: &Expr) -> R<Value> {
        Ok(match e {
            Expr::Lit(l) => match l {
                Lit::None => Value::None,
                Lit::Bool(b) => Value::Bool(*b),
                Lit::Int(i) => Value::Int(*i),
                Lit::Float(f) => Value::Float(*f),
                Lit::Str(s) => Value::str(s),
            },
            Expr::Name(n) => self.lookup(n),
            Expr::List(items) => {
                let mut v = Vec::with_capacity(items.len());
                for x in items {
                    v.push(self.eval(x)?);
                }
                Value::list(v)
            }
            Expr::Dict(items) => {
                let mut d: Pairs = Vec::with_capacity(items.len());
                for (k, x) in items {
                    let k = self.eval(k)?.to_text();
                    let x = self.eval(x)?;
                    match d.iter_mut().find(|(k2, _)| *k2 == k) {
                        Some(slot) => slot.1 = x,
                        None => d.push((k, x)),
                    }
                }
                Value::Dict(Rc::new(d))
            }
            Expr::Attr(obj, name) => self.eval(obj)?.get_item(name).unwrap_or(Value::Undefined),
            Expr::Index(obj, idx) => {
                let o = self.eval(obj)?;
                let i = self.eval(idx)?;
                index_value(&o, &i)?
            }
            Expr::Slice(obj, parts) => {
                let o = self.eval(obj)?;
                let mut p = [None, None, None];
                for (slot, part) in p.iter_mut().zip(parts.iter()) {
                    if let Some(x) = part {
                        *slot = match self.eval(x)? {
                            Value::None => None,
                            v => Some(v.as_int().ok_or_else(|| {
                                Error::Runtime("slice indices must be integers".into())
                            })?),
                        };
                    }
                }
                slice_value(&o, p)?
            }
            Expr::Call(callee, args, kwargs) => {
                if let Expr::Attr(obj, method) = callee.as_ref() {
                    let o = self.eval(obj)?;
                    // a callable stored under that name (a macro in a namespace) wins over a method
                    return match o.get_item(method) {
                        Some(f @ (Value::Macro(_) | Value::Global(_))) => {
                            let a = self.eval_args(args, kwargs)?;
                            self.call(&f, a)
                        }
                        _ => {
                            let a = self.eval_args(args, kwargs)?;
                            call_method(&o, method, &a)
                        }
                    };
                }
                let f = self.eval(callee)?;
                let a = self.eval_args(args, kwargs)?;
                self.call(&f, a)?
            }
            Expr::Filter(obj, name, args, kwargs) => {
                let v = self.eval(obj)?;
                let a = self.eval_args(args, kwargs)?;
                self.filter(name, v, &a)?
            }
            Expr::Test(obj, name, args, negated) => {
                let v = self.eval(obj)?;
                let mut a = Vec::with_capacity(args.len());
                for x in args {
                    a.push(self.eval(x)?);
                }
                Value::Bool(test(name, &v, &a)? != *negated)
            }
            Expr::Not(x) => Value::Bool(!self.eval(x)?.truthy()),
            Expr::Neg(x) => match self.eval(x)? {
                Value::Float(f) => Value::Float(-f),
                v => match v.as_int() {
                    Some(i) => Value::Int(-i),
                    None => return rt(format!("bad operand for unary -: '{}'", v.type_name())),
                },
            },
            Expr::Binary("and", l, r) => {
                let a = self.eval(l)?;
                if a.truthy() {
                    self.eval(r)?
                } else {
                    a
                }
            }
            Expr::Binary("or", l, r) => {
                let a = self.eval(l)?;
                if a.truthy() {
                    a
                } else {
                    self.eval(r)?
                }
            }
            Expr::Binary(op, l, r) => {
                let a = self.eval(l)?;
                let b = self.eval(r)?;
                binary(op, &a, &b)?
            }
            Expr::Cond(test, then, other) => {
                if self.eval(test)?.truthy() {
                    self.eval(then)?
                } else {
                    match other {
                        Some(o) => self.eval(o)?,
                        None => Value::Undefined,
                    }
                }
            }
        })
    }

    fn call(&mut self, f: &Value, a: Args) -> R<Value> {
        match f {
            Value::Macro(m) => self.call_macro(m, a),
            Value::Global("namespace") | Value::Global("dict") => {
                let mut pairs: Pairs = Vec::new();
                if let Some(Value::Dict(d)) = a.pos.first() {
                    pairs.extend(d.iter().cloned());
                }
                for (k, v) in a.kw {
                    match pairs.iter_mut().find(|(k2, _)| *k2 == k) {
                        Some(slot) => slot.1 = v,
                        None => pairs.push((k, v)),
                    }
                }
                Ok(if matches!(f, Value::Global("dict")) {
                    Value::Dict(Rc::new(pairs))
                } else {
                    Value::Namespace(Rc::new(RefCell::new(pairs)))
                })
            }
            Value::Global("range") => {
                let ints: Option<Vec<i64>> = a.pos.iter().map(Value::as_int).collect();
                let Some(ints) = ints else {
                    return rt("range() takes integers");
                };
                let (start, stop, step) = match ints.as_slice() {
                    [stop] => (0, *stop, 1),
                    [start, stop] => (*start, *stop, 1),
                    [start, stop, step] if *step != 0 => (*start, *stop, *step),
                    _ => return rt("range() takes 1 to 3 arguments and a non-zero step"),
                };
                let mut v = Vec::new();
                let mut x = start;
                while (step > 0 && x < stop) || (step < 0 && x > stop) {
                    if v.len() >= 1_000_000 {
                        return rt("range() is too long");
                    }
                    v.push(Value::Int(x));
                    x += step;
                }
                Ok(Value::list(v))
            }
            Value::Global("raise_exception") => {
                Err(Error::Raised(a.text(0, "message").unwrap_or_default()))
            }
            Value::Global("strftime_now") => Ok(Value::str(strftime_now(
                &a.text(0, "format").unwrap_or_default(),
            ))),
            other => rt(format!("'{}' object is not callable", other.type_name())),
        }
    }

    fn call_macro(&mut self, m: &Arc<MacroDef>, a: Args) -> R<Value> {
        if self.depth >= MAX_DEPTH {
            return rt("macro recursion is too deep");
        }
        // a macro sees the template's top-level names and its own arguments
        let saved = self.scopes.split_off(1);
        self.scopes.push(Vec::new());
        self.depth += 1;
        let mut out = String::new();
        let r = (|| {
            for (i, (name, default)) in m.params.iter().enumerate() {
                let v = match a.get(i, name) {
                    Some(v) => v,
                    None => match default {
                        Some(d) => self.eval(d)?,
                        None => Value::Undefined,
                    },
                };
                self.assign(name, v);
            }
            self.exec(&m.body, &mut out)
        })();
        self.depth -= 1;
        self.scopes.truncate(1);
        self.scopes.extend(saved);
        r?;
        Ok(Value::str(out))
    }

    fn filter(&mut self, name: &str, v: Value, a: &Args) -> R<Value> {
        Ok(match name {
            "tojson" => {
                // the chat-template `tojson`: json.dumps(v, ensure_ascii=False, indent=None,
                // separators=None, sort_keys=False), each settable by keyword
                let indent = a
                    .get(1, "indent")
                    .and_then(|x| x.as_int())
                    .map(|n| n.max(0) as usize);
                let mut opts = JsonOpts::new(indent);
                if let Some(e) = a.get(0, "ensure_ascii") {
                    opts.ensure_ascii = e.truthy();
                }
                match a.get(2, "separators") {
                    None | Some(Value::None) => {}
                    Some(Value::List(l)) if l.len() == 2 => {
                        opts.item_sep = l[0].to_text();
                        opts.key_sep = l[1].to_text();
                    }
                    Some(_) => return rt("tojson: separators must be an (item, key) pair"),
                }
                if let Some(k) = a.get(3, "sort_keys") {
                    opts.sort_keys = k.truthy();
                }
                let mut o = String::new();
                json_dumps(&v, &opts, 0, &mut o)?;
                Value::str(o)
            }
            "string" => Value::str(v.to_text()),
            "safe" => v,
            "trim" => match a.text(0, "chars") {
                Some(chars) => Value::str(v.to_text().trim_matches(|c| chars.contains(c))),
                None => Value::str(v.to_text().trim()),
            },
            "upper" => Value::str(v.to_text().to_uppercase()),
            "lower" => Value::str(v.to_text().to_lowercase()),
            "capitalize" => Value::str(capitalize(&v.to_text())),
            "title" => Value::str(title(&v.to_text())),
            "default" | "d" => {
                let boolean = a.get(1, "boolean").is_some_and(|b| b.truthy());
                if matches!(v, Value::Undefined) || (boolean && !v.truthy()) {
                    a.get(0, "default_value").unwrap_or_else(|| Value::str(""))
                } else {
                    v
                }
            }
            "length" | "count" => Value::Int(match &v {
                Value::Str(s) => s.chars().count(),
                Value::List(l) => l.len(),
                Value::Dict(d) => d.len(),
                Value::Undefined => 0,
                other => {
                    return rt(format!(
                        "object of type '{}' has no length",
                        other.type_name()
                    ))
                }
            } as i64),
            "list" => Value::list(v.iterate()?),
            "items" => match &v {
                Value::Undefined => Value::list(Vec::new()),
                Value::Dict(_) | Value::Namespace(_) => dict_items(&v),
                _ => return rt("the items filter takes a mapping"),
            },
            "dictsort" => {
                let Value::List(pairs) = dict_items(&v) else {
                    unreachable!()
                };
                if !matches!(v, Value::Dict(_) | Value::Namespace(_)) {
                    return rt("the dictsort filter takes a mapping");
                }
                let case = a.get(0, "case_sensitive").is_some_and(|b| b.truthy());
                let by_value = a.text(1, "by").as_deref() == Some("value");
                let mut pairs = pairs.as_ref().clone();
                let mut err = None;
                pairs.sort_by(|x, y| {
                    let (Value::List(x), Value::List(y)) = (x, y) else {
                        unreachable!()
                    };
                    let i = usize::from(by_value);
                    let key = |v: &Value| match v {
                        Value::Str(s) if !case => Value::str(s.to_lowercase()),
                        other => other.clone(),
                    };
                    compare(&key(&x[i]), &key(&y[i])).unwrap_or_else(|e| {
                        err = Some(e);
                        std::cmp::Ordering::Equal
                    })
                });
                if let Some(e) = err {
                    return Err(e);
                }
                if a.get(2, "reverse").is_some_and(|b| b.truthy()) {
                    pairs.reverse();
                }
                Value::list(pairs)
            }
            "join" => {
                let sep = a.text(0, "d").unwrap_or_default();
                let attr = a.text(1, "attribute");
                let parts: Vec<String> = v
                    .iterate()?
                    .iter()
                    .map(|x| match &attr {
                        Some(k) => x.get_item(k).unwrap_or(Value::Undefined).to_text(),
                        None => x.to_text(),
                    })
                    .collect();
                Value::str(parts.join(&sep))
            }
            "first" => v.iterate()?.into_iter().next().unwrap_or(Value::Undefined),
            "last" => v.iterate()?.pop().unwrap_or(Value::Undefined),
            "reverse" => match &v {
                Value::Str(s) => Value::str(s.chars().rev().collect::<String>()),
                other => {
                    let mut l = other.iterate()?;
                    l.reverse();
                    Value::list(l)
                }
            },
            "unique" => {
                let mut seen: Vec<Value> = Vec::new();
                for x in v.iterate()? {
                    if !seen.iter().any(|s| values_equal(s, &x)) {
                        seen.push(x);
                    }
                }
                Value::list(seen)
            }
            "sort" => {
                let mut l = v.iterate()?;
                let attr = a.text(2, "attribute");
                let mut err = None;
                l.sort_by(|x, y| {
                    let key = |v: &Value| match &attr {
                        Some(k) => v.get_item(k).unwrap_or(Value::Undefined),
                        None => v.clone(),
                    };
                    compare(&key(x), &key(y)).unwrap_or_else(|e| {
                        err = Some(e);
                        std::cmp::Ordering::Equal
                    })
                });
                if let Some(e) = err {
                    return Err(e);
                }
                if a.get(0, "reverse").is_some_and(|b| b.truthy()) {
                    l.reverse();
                }
                Value::list(l)
            }
            "replace" => {
                let (Some(from), Some(to)) = (a.text(0, "old"), a.text(1, "new")) else {
                    return rt("replace takes two arguments");
                };
                let s = v.to_text();
                Value::str(match a.get(2, "count").and_then(|c| c.as_int()) {
                    Some(n) => s.replacen(&from, &to, n.max(0) as usize),
                    None => s.replace(&from, &to),
                })
            }
            "int" => match &v {
                Value::Float(f) => Value::Int(*f as i64),
                Value::Str(s) => match s.trim().parse::<i64>() {
                    Ok(i) => Value::Int(i),
                    Err(_) => match s.trim().parse::<f64>() {
                        Ok(f) => Value::Int(f as i64),
                        Err(_) => a.get(0, "default").unwrap_or(Value::Int(0)),
                    },
                },
                other => match other.as_int() {
                    Some(i) => Value::Int(i),
                    None => a.get(0, "default").unwrap_or(Value::Int(0)),
                },
            },
            "float" => match &v {
                Value::Str(s) => match s.trim().parse::<f64>() {
                    Ok(f) => Value::Float(f),
                    Err(_) => a.get(0, "default").unwrap_or(Value::Float(0.0)),
                },
                other => match other.as_f64() {
                    Some(f) => Value::Float(f),
                    None => a.get(0, "default").unwrap_or(Value::Float(0.0)),
                },
            },
            "abs" => match &v {
                Value::Float(f) => Value::Float(f.abs()),
                other => match other.as_int() {
                    Some(i) => Value::Int(i.abs()),
                    None => return rt("abs takes a number"),
                },
            },
            "min" | "max" => {
                let mut best: Option<Value> = None;
                for x in v.iterate()? {
                    let better = match &best {
                        None => true,
                        Some(b) => {
                            let o = compare(&x, b)?;
                            if name == "min" {
                                o.is_lt()
                            } else {
                                o.is_gt()
                            }
                        }
                    };
                    if better {
                        best = Some(x);
                    }
                }
                best.unwrap_or(Value::Undefined)
            }
            "sum" => {
                let mut acc = a.get(1, "start").unwrap_or(Value::Int(0));
                let attr = a.text(0, "attribute");
                for x in v.iterate()? {
                    let x = match &attr {
                        Some(k) => x.get_item(k).unwrap_or(Value::Undefined),
                        None => x,
                    };
                    acc = binary("+", &acc, &x)?;
                }
                acc
            }
            "map" => {
                let items = v.iterate()?;
                if let Some(attr) =
                    a.kw.iter()
                        .find(|(k, _)| k == "attribute")
                        .map(|(_, v)| v.to_text())
                {
                    let default =
                        a.kw.iter()
                            .find(|(k, _)| k == "default")
                            .map(|(_, v)| v.clone());
                    Value::list(
                        items
                            .iter()
                            .map(|x| {
                                attr.split('.')
                                    .try_fold(x.clone(), |cur, part| cur.get_item(part))
                                    .or_else(|| default.clone())
                                    .unwrap_or(Value::Undefined)
                            })
                            .collect(),
                    )
                } else {
                    let Some(fname) = a.pos.first().map(Value::to_text) else {
                        return rt("map needs a filter name or attribute=");
                    };
                    let rest = Args {
                        pos: a.pos[1..].to_vec(),
                        kw: a.kw.clone(),
                    };
                    let mut out = Vec::with_capacity(items.len());
                    for x in items {
                        out.push(self.filter(&fname, x, &rest)?);
                    }
                    Value::list(out)
                }
            }
            "select" | "reject" | "selectattr" | "rejectattr" => {
                let by_attr = name.ends_with("attr");
                let keep_on = name.starts_with("select");
                let mut pos = a.pos.iter();
                let attr = if by_attr {
                    match pos.next() {
                        Some(k) => Some(k.to_text()),
                        None => return rt(format!("{name} needs an attribute name")),
                    }
                } else {
                    None
                };
                let test_name = pos.next().map(Value::to_text);
                let test_args: Vec<Value> = pos.cloned().collect();
                let mut out = Vec::new();
                for x in v.iterate()? {
                    let subject = match &attr {
                        Some(k) => k
                            .split('.')
                            .try_fold(x.clone(), |cur, part| cur.get_item(part))
                            .unwrap_or(Value::Undefined),
                        None => x.clone(),
                    };
                    let hit = match &test_name {
                        Some(t) => test(t, &subject, &test_args)?,
                        None => subject.truthy(),
                    };
                    if hit == keep_on {
                        out.push(x);
                    }
                }
                Value::list(out)
            }
            "e" | "escape" => {
                let mut o = String::new();
                for c in v.to_text().chars() {
                    match c {
                        '&' => o.push_str("&amp;"),
                        '<' => o.push_str("&lt;"),
                        '>' => o.push_str("&gt;"),
                        '"' => o.push_str("&#34;"),
                        '\'' => o.push_str("&#39;"),
                        c => o.push(c),
                    }
                }
                Value::str(o)
            }
            "indent" => {
                let width = a
                    .get(0, "width")
                    .and_then(|w| w.as_int())
                    .unwrap_or(4)
                    .max(0) as usize;
                let first = a.get(1, "first").is_some_and(|b| b.truthy());
                let blank = a.get(2, "blank").is_some_and(|b| b.truthy());
                let pad = " ".repeat(width);
                let text = v.to_text();
                let mut o = String::new();
                for (i, line) in text.split('\n').enumerate() {
                    if i > 0 {
                        o.push('\n');
                    }
                    if (i > 0 || first) && (blank || !line.is_empty()) {
                        o.push_str(&pad);
                    }
                    o.push_str(line);
                }
                Value::str(o)
            }
            "round" => {
                let Some(f) = v.as_f64() else {
                    return rt("round takes a number");
                };
                let p = a.get(0, "precision").and_then(|p| p.as_int()).unwrap_or(0);
                let m = 10f64.powi(p as i32);
                Value::Float((f * m).round() / m)
            }
            other => return rt(format!("no filter named '{other}'")),
        })
    }
}

fn dict_items(v: &Value) -> Value {
    let pairs: Pairs = match v {
        Value::Dict(d) => d.as_ref().clone(),
        Value::Namespace(n) => n.borrow().clone(),
        _ => Vec::new(),
    };
    Value::list(
        pairs
            .into_iter()
            .map(|(k, x)| Value::list(vec![Value::str(k), x]))
            .collect(),
    )
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
        None => String::new(),
    }
}

fn title(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    let mut start = true;
    for c in s.chars() {
        if c.is_alphabetic() {
            if start {
                o.extend(c.to_uppercase());
            } else {
                o.extend(c.to_lowercase());
            }
            start = false;
        } else {
            o.push(c);
            start = true;
        }
    }
    o
}

fn norm_index(i: i64, len: usize) -> Option<usize> {
    let i = if i < 0 { i + len as i64 } else { i };
    (i >= 0 && (i as usize) < len).then_some(i as usize)
}

fn index_value(o: &Value, i: &Value) -> R<Value> {
    Ok(match (o, i) {
        (Value::Dict(_) | Value::Namespace(_), Value::Str(k)) => {
            o.get_item(k).unwrap_or(Value::Undefined)
        }
        (Value::List(l), _) if i.as_int().is_some() => {
            norm_index(i.as_int().unwrap(), l.len()).map_or(Value::Undefined, |k| l[k].clone())
        }
        (Value::Str(s), _) if i.as_int().is_some() => {
            let chars: Vec<char> = s.chars().collect();
            norm_index(i.as_int().unwrap(), chars.len())
                .map_or(Value::Undefined, |k| Value::str(chars[k].to_string()))
        }
        (Value::Undefined | Value::None, _) => {
            return rt(format!("'{}' object is not subscriptable", o.type_name()))
        }
        _ => Value::Undefined,
    })
}

fn slice_value(o: &Value, p: [Option<i64>; 3]) -> R<Value> {
    fn pick<T: Clone>(items: &[T], p: [Option<i64>; 3]) -> R<Vec<T>> {
        let n = items.len() as i64;
        let step = p[2].unwrap_or(1);
        if step == 0 {
            return rt("slice step cannot be zero");
        }
        let clamp = |v: i64, lo: i64, hi: i64| {
            let v = if v < 0 { v + n } else { v };
            v.clamp(lo, hi)
        };
        let mut out = Vec::new();
        if step > 0 {
            // forward bounds come from the verified core (never past the end)
            let bound =
                |v: i64| crate::core::tool_calls::forward_slice_bound(v, items.len()) as i64;
            let start = p[0].map_or(0, bound);
            let stop = p[1].map_or(n, bound);
            let mut i = start;
            while i < stop {
                out.push(items[i as usize].clone());
                i += step;
            }
        } else {
            let start = p[0].map_or(n - 1, |v| clamp(v, -1, n - 1));
            let stop = p[1].map_or(-1, |v| clamp(v, -1, n - 1));
            let mut i = start;
            while i > stop {
                out.push(items[i as usize].clone());
                i += step;
            }
        }
        Ok(out)
    }
    match o {
        Value::List(l) => Ok(Value::list(pick(l, p)?)),
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            Ok(Value::str(pick(&chars, p)?.into_iter().collect::<String>()))
        }
        other => rt(format!("'{}' object is not sliceable", other.type_name())),
    }
}

fn binary(op: &str, a: &Value, b: &Value) -> R<Value> {
    let bad = || {
        rt(format!(
            "unsupported operand types for {op}: '{}' and '{}'",
            a.type_name(),
            b.type_name()
        ))
    };
    Ok(match op {
        "==" => Value::Bool(values_equal(a, b)),
        "!=" => Value::Bool(!values_equal(a, b)),
        "<" => Value::Bool(compare(a, b)?.is_lt()),
        "<=" => Value::Bool(compare(a, b)?.is_le()),
        ">" => Value::Bool(compare(a, b)?.is_gt()),
        ">=" => Value::Bool(compare(a, b)?.is_ge()),
        "in" | "notin" => {
            let found = match (a, b) {
                (Value::Str(x), Value::Str(y)) => y.contains(x.as_ref()),
                (_, Value::List(l)) => l.iter().any(|x| values_equal(x, a)),
                (Value::Str(k), Value::Dict(_) | Value::Namespace(_)) => b.get_item(k).is_some(),
                (_, Value::Dict(_) | Value::Namespace(_) | Value::Undefined) => false,
                _ => return bad(),
            };
            Value::Bool(found == (op == "in"))
        }
        "~" => Value::str(a.to_text() + &b.to_text()),
        "+" => match (a, b) {
            (Value::Str(x), Value::Str(y)) => Value::str(format!("{x}{y}")),
            (Value::List(x), Value::List(y)) => {
                Value::list(x.iter().chain(y.iter()).cloned().collect())
            }
            _ => return arith(op, a, b).or_else(|_| bad()),
        },
        "*" => match (a, b) {
            (Value::Str(s), n) | (n, Value::Str(s)) if n.as_int().is_some() => {
                Value::str(s.repeat(n.as_int().unwrap().max(0) as usize))
            }
            _ => return arith(op, a, b).or_else(|_| bad()),
        },
        _ => return arith(op, a, b).or_else(|_| bad()),
    })
}

fn arith(op: &str, a: &Value, b: &Value) -> R<Value> {
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        let v = match op {
            "+" => x.checked_add(y),
            "-" => x.checked_sub(y),
            "*" => x.checked_mul(y),
            "//" if y != 0 => Some(x.div_euclid(y) - i64::from(y < 0 && x.rem_euclid(y) != 0)),
            "%" if y != 0 => {
                Some(x - y * (x.div_euclid(y) - i64::from(y < 0 && x.rem_euclid(y) != 0)))
            }
            "**" if y >= 0 => u32::try_from(y).ok().and_then(|e| x.checked_pow(e)),
            "/" if y != 0 => return Ok(Value::Float(x as f64 / y as f64)),
            "**" => return Ok(Value::Float((x as f64).powf(y as f64))),
            _ => return rt("division by zero"),
        };
        return v
            .map(Value::Int)
            .ok_or_else(|| Error::Runtime("integer overflow".into()));
    }
    let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) else {
        return rt("not numbers");
    };
    Ok(Value::Float(match op {
        "+" => x + y,
        "-" => x - y,
        "*" => x * y,
        "/" if y != 0.0 => x / y,
        "//" if y != 0.0 => (x / y).floor(),
        "%" if y != 0.0 => x - y * (x / y).floor(),
        "**" => x.powf(y),
        _ => return rt("division by zero"),
    }))
}

fn test(name: &str, v: &Value, args: &[Value]) -> R<bool> {
    let arg = |i: usize| args.get(i).cloned().unwrap_or(Value::Undefined);
    Ok(match name {
        "defined" => !matches!(v, Value::Undefined),
        "undefined" => matches!(v, Value::Undefined),
        "none" => matches!(v, Value::None),
        "string" => matches!(v, Value::Str(_)),
        "mapping" => matches!(v, Value::Dict(_)),
        "iterable" => matches!(
            v,
            Value::Str(_) | Value::List(_) | Value::Dict(_) | Value::Undefined
        ),
        "sequence" => matches!(v, Value::Str(_) | Value::List(_) | Value::Dict(_)),
        "number" => matches!(v, Value::Int(_) | Value::Float(_) | Value::Bool(_)),
        "integer" => matches!(v, Value::Int(_)),
        "float" => matches!(v, Value::Float(_)),
        "boolean" => matches!(v, Value::Bool(_)),
        "true" => matches!(v, Value::Bool(true)),
        "false" => matches!(v, Value::Bool(false)),
        "callable" => matches!(v, Value::Macro(_) | Value::Global(_)),
        "eq" | "equalto" | "sameas" | "==" => values_equal(v, &arg(0)),
        "ne" | "!=" => !values_equal(v, &arg(0)),
        "lt" | "lessthan" | "<" => compare(v, &arg(0))?.is_lt(),
        "le" | "<=" => compare(v, &arg(0))?.is_le(),
        "gt" | "greaterthan" | ">" => compare(v, &arg(0))?.is_gt(),
        "ge" | ">=" => compare(v, &arg(0))?.is_ge(),
        "in" => matches!(binary("in", v, &arg(0))?, Value::Bool(true)),
        "even" => v.as_int().is_some_and(|i| i % 2 == 0),
        "odd" => v.as_int().is_some_and(|i| i % 2 != 0),
        "divisibleby" => match (v.as_int(), arg(0).as_int()) {
            (Some(x), Some(y)) if y != 0 => x % y == 0,
            _ => return rt("divisibleby takes non-zero integers"),
        },
        "lower" => v.to_text().chars().all(|c| !c.is_uppercase()),
        "upper" => v.to_text().chars().all(|c| !c.is_lowercase()),
        other => return rt(format!("no test named '{other}'")),
    })
}

fn call_method(o: &Value, method: &str, a: &Args) -> R<Value> {
    if let Value::Dict(_) | Value::Namespace(_) = o {
        return Ok(match method {
            "get" => {
                let key = a.pos.first().map(Value::to_text).unwrap_or_default();
                o.get_item(&key)
                    .or_else(|| a.get(1, "default"))
                    .unwrap_or(Value::None)
            }
            "items" => dict_items(o),
            "keys" => Value::list(o.iterate().unwrap_or_default()),
            "values" => {
                let Value::List(pairs) = dict_items(o) else {
                    unreachable!()
                };
                Value::list(
                    pairs
                        .iter()
                        .map(|p| index_value(p, &Value::Int(1)).unwrap_or(Value::Undefined))
                        .collect(),
                )
            }
            _ => return rt(format!("a mapping has no method '{method}'")),
        });
    }
    let Value::Str(s) = o else {
        return rt(format!(
            "'{}' object has no method '{method}'",
            o.type_name()
        ));
    };
    let s: &str = s;
    // an argument that is one string or a tuple of strings (`startswith`)
    let any_of = |v: Option<Value>, f: &dyn Fn(&str) -> bool| match v {
        Some(Value::List(l)) => l.iter().any(|x| f(&x.to_text())),
        Some(x) => f(&x.to_text()),
        None => false,
    };
    let strip = |how: u8| -> String {
        match a.pos.first() {
            Some(Value::Str(chars)) => {
                let m = |c: char| chars.contains(c);
                match how {
                    0 => s.trim_matches(m),
                    1 => s.trim_start_matches(m),
                    _ => s.trim_end_matches(m),
                }
            }
            _ => match how {
                0 => s.trim(),
                1 => s.trim_start(),
                _ => s.trim_end(),
            },
        }
        .to_string()
    };
    Ok(match method {
        "startswith" => Value::Bool(any_of(a.pos.first().cloned(), &|p| s.starts_with(p))),
        "endswith" => Value::Bool(any_of(a.pos.first().cloned(), &|p| s.ends_with(p))),
        "strip" => Value::str(strip(0)),
        "lstrip" => Value::str(strip(1)),
        "rstrip" => Value::str(strip(2)),
        "upper" => Value::str(s.to_uppercase()),
        "lower" => Value::str(s.to_lowercase()),
        "title" => Value::str(title(s)),
        "capitalize" => Value::str(capitalize(s)),
        "split" => {
            let limit = a
                .get(1, "maxsplit")
                .and_then(|n| n.as_int())
                .filter(|n| *n >= 0);
            let parts: Vec<String> = match a.pos.first() {
                Some(Value::Str(sep)) if !sep.is_empty() => match limit {
                    Some(n) => s
                        .splitn(n as usize + 1, sep.as_ref())
                        .map(str::to_string)
                        .collect(),
                    None => s.split(sep.as_ref()).map(str::to_string).collect(),
                },
                Some(Value::Str(_)) => return rt("empty separator"),
                _ => s.split_whitespace().map(str::to_string).collect(),
            };
            Value::list(parts.into_iter().map(Value::str).collect())
        }
        "splitlines" => Value::list(s.lines().map(Value::str).collect()),
        "replace" => match (a.pos.first(), a.pos.get(1)) {
            (Some(x), Some(y)) => Value::str(s.replace(&x.to_text(), &y.to_text())),
            _ => return rt("replace takes two arguments"),
        },
        "find" => {
            let needle = a.pos.first().map(Value::to_text).unwrap_or_default();
            Value::Int(
                s.find(&needle)
                    .map_or(-1, |byte| s[..byte].chars().count() as i64),
            )
        }
        "join" => {
            let parts: Vec<String> = match a.pos.first() {
                Some(v) => v.iterate()?.iter().map(Value::to_text).collect(),
                None => Vec::new(),
            };
            Value::str(parts.join(s))
        }
        "isdigit" => Value::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())),
        "isalpha" => Value::Bool(!s.is_empty() && s.chars().all(char::is_alphabetic)),
        "isalnum" => Value::Bool(!s.is_empty() && s.chars().all(char::is_alphanumeric)),
        "isspace" => Value::Bool(!s.is_empty() && s.chars().all(char::is_whitespace)),
        _ => return rt(format!("a string has no method '{method}'")),
    })
}

/// `strftime` of the current time in UTC for the directives chat templates use.
fn strftime_now(format: &str) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    strftime_utc(format, secs)
}

fn strftime_utc(format: &str, secs: i64) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: [&str; 7] = [
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
    ];
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    // civil date from a day count (proleptic Gregorian calendar)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let weekday = DAYS[days.rem_euclid(7) as usize];
    let mut o = String::new();
    let mut it = format.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            o.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => o.push_str(&year.to_string()),
            Some('y') => o.push_str(&format!("{:02}", year.rem_euclid(100))),
            Some('m') => o.push_str(&format!("{month:02}")),
            Some('d') => o.push_str(&format!("{day:02}")),
            Some('e') => o.push_str(&format!("{day:2}")),
            Some('H') => o.push_str(&format!("{:02}", tod / 3600)),
            Some('M') => o.push_str(&format!("{:02}", tod / 60 % 60)),
            Some('S') => o.push_str(&format!("{:02}", tod % 60)),
            Some('B') => o.push_str(MONTHS[month as usize - 1]),
            Some('b') => o.push_str(&MONTHS[month as usize - 1][..3]),
            Some('A') => o.push_str(weekday),
            Some('a') => o.push_str(&weekday[..3]),
            Some('%') => o.push('%'),
            Some(other) => {
                o.push('%');
                o.push(other);
            }
            None => o.push('%'),
        }
    }
    o
}

/// Renders `source` with the context of each fixture case (`{"name", "context", "text" | "error"}`,
/// written by `tests/data/chat/make_template_fixtures.py`) and returns the cases whose rendering
/// (or `raise_exception` message) differs from the reference's.
#[cfg(test)]
pub(crate) fn fixture_mismatches(source: &str, cases: &JsonValue) -> Vec<String> {
    let template = Template::parse(source).expect("the template parses");
    let mut bad = Vec::new();
    for case in cases.as_array().expect("a list of cases") {
        let name = case.get("name").and_then(JsonValue::as_str).unwrap_or("?");
        let vars = case
            .get("context")
            .and_then(JsonValue::as_object)
            .expect("a context object")
            .iter()
            .map(|(k, v)| (k.clone(), Value::from_json(v)))
            .collect();
        let got = template.render(vars);
        let want_text = case.get("text").and_then(JsonValue::as_str);
        let want_err = case.get("error").and_then(JsonValue::as_str);
        let ok = match (&got, want_text, want_err) {
            (Ok(t), Some(w), _) => t == w,
            (Err(Error::Raised(m)), _, Some(w)) => m == w,
            _ => false,
        };
        if !ok {
            bad.push(format!(
                "{name}: got {got:?}, want {:?}",
                want_text.or(want_err)
            ));
        }
    }
    bad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    #[test]
    fn features_render_like_jinja2() {
        let fixture = parse_json(include_str!("../../tests/data/chat/jinja_features.json")).unwrap();
        let source = fixture.get("template").and_then(JsonValue::as_str).unwrap();
        let cases = fixture.get("cases").unwrap();
        assert!(cases.as_array().unwrap().len() >= 10);
        let bad = fixture_mismatches(source, cases);
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    fn render(src: &str, ctx: &str) -> String {
        try_render(src, ctx).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    fn try_render(src: &str, ctx: &str) -> Result<String, Error> {
        let ctx = parse_json(ctx).expect("context JSON");
        let vars = ctx
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), Value::from_json(v)))
            .collect();
        Template::parse(src)?.render(vars)
    }

    #[test]
    fn whitespace_control_follows_trim_and_lstrip_blocks() {
        // the newline after a block tag goes, and so does indentation before one
        let t = "a\n  {% if x %}\n    yes\n  {% endif %}\nb";
        assert_eq!(render(t, r#"{"x": true}"#), "a\n    yes\nb");
        assert_eq!(render(t, r#"{"x": false}"#), "a\nb");
        // an output tag keeps both; `-` strips everything on its side
        assert_eq!(render("a\n  {{ x }}\nb", r#"{"x": 1}"#), "a\n  1\nb");
        assert_eq!(render("a \n {{- x -}} \n b", r#"{"x": 1}"#), "a1b");
        // whitespace between two tags on one line is not at a line start
        assert_eq!(render("{% if x %}  {% endif %}|", r#"{"x": true}"#), "  |");
        assert_eq!(render("a {# note #}\nb{#- c -#}  d", "{}"), "a bd");
    }

    #[test]
    fn values_print_and_compare_as_python_does() {
        let ctx = r#"{"f": 1.0, "g": 1e-05, "h": 2.5e16, "n": null, "t": true,
                      "l": [1, "a'b", null, {"k": false}], "s": "héllo"}"#;
        assert_eq!(
            render(
                "{{ f }} {{ g }} {{ h }} {{ n }} {{ t }} {{ l }} {{ missing }}|",
                ctx
            ),
            "1.0 1e-05 2.5e+16 None True [1, \"a'b\", None, {'k': False}] |"
        );
        assert_eq!(
            render("{{ s[1] }}{{ s[-1] }}{{ s[1:3] }}{{ s[::-1] }}", ctx),
            "éoélolléh"
        );
        assert_eq!(
            render("{{ 1 == 1.0 }} {{ 'a' in 'cat' }} {{ 2 in l }} {{ 'k' in l[3] }} {{ 7 // 2 }} {{ -7 % 3 }} {{ 1 / 2 }}", ctx),
            "True True False True 3 2 0.5"
        );
        assert_eq!(
            render("{{ 'a' ~ 1 ~ none }} {{ 'x' + 'y' }} {{ [1] + [2] }}", ctx),
            "a1None xy [1, 2]"
        );
        assert_eq!(
            render(
                "{{ (n or 'd') }} {{ t and 'yes' }} {{ 'a' if n else 'b' }}{{ 'c' if n }}",
                ctx
            ),
            "d yes b"
        );
    }

    #[test]
    fn tojson_writes_python_json() {
        let ctx = r#"{"tool": {"name": "f", "n": 1.0, "s": "a\"b\nü\u0001", "l": [1, [], {}], "d": {"x": null, "y": true}}}"#;
        assert_eq!(
            render("{{ tool | tojson }}", ctx),
            "{\"name\": \"f\", \"n\": 1.0, \"s\": \"a\\\"b\\nü\\u0001\", \"l\": [1, [], {}], \"d\": {\"x\": null, \"y\": true}}"
        );
        assert_eq!(
            render("{{ tool.d | tojson(indent=2) }}", ctx),
            "{\n  \"x\": null,\n  \"y\": true\n}"
        );
        assert!(try_render("{{ nope | tojson }}", "{}").is_err());
    }

    #[test]
    fn loops_scopes_namespaces_and_macros() {
        let ctx = r#"{"items": [{"r": "a"}, {"r": "b"}, {"r": "c"}]}"#;
        // a plain set inside a loop does not leave it; a namespace attribute does
        let t = "{%- set last = 'none' %}{%- set ns = namespace(n=0, seen='') %}\
                 {%- for it in items %}{%- set last = it.r %}{%- set ns.n = ns.n + 1 %}\
                 {%- set ns.seen = ns.seen ~ loop.index0 ~ it.r ~ (loop.previtem.r if loop.previtem else '-') ~ (loop.nextitem.r if not loop.last else '-') %}\
                 {%- endfor %}{{ last }} {{ ns.n }} {{ ns.seen }}";
        assert_eq!(render(t, ctx), "none 3 0a-b1bac2cb-");
        assert_eq!(
            render("{% for it in items[::-1] if it.r != 'b' %}{{ it.r }}{{ loop.length }}{% else %}empty{% endfor %}", ctx),
            "c2a2"
        );
        assert_eq!(
            render("{% for it in [] %}x{% else %}empty{% endfor %}", ctx),
            "empty"
        );
        assert_eq!(
            render("{% for i in range(5) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 3 %}{% break %}{% endif %}{{ i }}{% endfor %}", ctx),
            "02"
        );
        // macros: defaults, keyword arguments, recursion, a later macro, top-level names
        let t = "{%- set sep = '/' %}\
                 {%- macro show(v, wrap=true) %}{%- if v is mapping %}{{ '{' }}{% for k, x in v | dictsort %}{{ k }}:{{ show(x, wrap=false) }}{{ sep if not loop.last }}{% endfor %}{{ '}' }}\
                 {%- elif v is string %}{{ quote(v) if wrap else v }}{%- else %}{{ v }}{%- endif %}{%- endmacro %}\
                 {%- macro quote(s) %}'{{ s }}'{% endmacro %}\
                 {{- show({'b': 1, 'A': {'z': 'q'}}) }} {{ show('w') }} {{ show('w', wrap=false) | upper }}";
        assert_eq!(render(t, "{}"), "{A:{z:q}/b:1} 'w' W");
        assert_eq!(
            render(
                "{% set body %}  x {{ 1 + 1 }} {% endset %}[{{ body | trim }}]",
                "{}"
            ),
            "[x 2]"
        );
    }

    #[test]
    fn filters_tests_and_methods() {
        let ctx = r#"{"m": {"role": "user", "content": " hi \n"}, "l": [{"a": 1, "b": "x"}, {"a": 2}], "u": ["b", "a", "b"]}"#;
        assert_eq!(
            render("{{ m.content | trim | length }} {{ m.missing | default('d') }} {{ '' | default('e', true) }} {{ l | map(attribute='a') | list }} {{ u | unique | join(',') }} {{ u | map('upper') | join }}", ctx),
            "2 d e [1, 2] b,a BAB"
        );
        assert_eq!(
            render("{{ l | selectattr('b', 'defined') | list | length }} {{ l | rejectattr('a', 'eq', 1) | map(attribute='a') | first }} {{ m | items | list | length }} {{ m.get('x') is none }} {{ m.get('x', 3) }}", ctx),
            "1 2 2 True 3"
        );
        assert_eq!(
            render("{{ m.content.strip().startswith('h') }} {{ 'a</t>b'.split('</t>')[-1] }} {{ 'xxaxx'.strip('x') }} {{ '\\nq'.lstrip('\\n') }} {{ 'a,b'.split(',') | last }}", ctx),
            "True b a q b"
        );
        assert_eq!(
            render("{{ m is mapping }} {{ m.content is string }} {{ l is iterable and l is not mapping }} {{ x is undefined }} {{ m.role is defined }} {{ 3 is number }} {{ true is boolean }} {{ l is sequence }}", ctx),
            "True True True True True True True True"
        );
        // a filter binds tighter than a comparison or an addition
        assert_eq!(
            render("{{ m.role | upper == 'USER' }} {{ u | length - 1 }}", ctx),
            "True 2"
        );
    }

    #[test]
    fn errors_are_reported() {
        assert_eq!(
            try_render("{{ raise_exception('no ' 'way') }}", "{}"),
            Err(Error::Raised("no way".into()))
        );
        assert!(matches!(
            Template::parse("{% if x %}a"),
            Err(Error::Syntax(_))
        ));
        assert!(matches!(
            Template::parse("{% frobnicate %}"),
            Err(Error::Syntax(_))
        ));
        assert!(matches!(
            try_render("{{ 'a' + 1 }}", "{}"),
            Err(Error::Runtime(_))
        ));
        assert!(matches!(
            try_render("{{ x | nosuch }}", "{}"),
            Err(Error::Runtime(_))
        ));
    }

    #[test]
    fn strftime_formats_a_known_instant() {
        // 2026-11-22 13:05:09 UTC, a Sunday
        assert_eq!(
            strftime_utc("%Y-%m-%d %H:%M:%S %A %d %b %B %y %%", 1_795_352_709),
            "2026-11-22 13:05:09 Sunday 22 Nov November 26 %"
        );
    }
}
