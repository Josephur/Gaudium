//! GGUF checkpoint reader (versions 2 and 3) over a read-only memory map.
//!
//! ```text
//! let g = Gguf::open("model.gguf" | "model-00001-of-00003.gguf" | "dir/")?;
//! g.arch()                                   general.architecture
//! g.kv("qwen35.block_count")                 typed metadata (arrays stay in the mapping)
//! g.tensor("blk.0.attn_q.weight")            name, dims (GGUF order), ggml type, byte range
//! dequant::rows_bf16(&g, t, r0, r1, &mut out) rows [r0, r1) decoded on the host
//! ```
//!
//! Format facts (GGUF specification and the reference reader, read for facts only):
//! - little-endian; header = magic `GGUF`, u32 version (2 or 3), u64 tensor count, u64
//!   key/value count; a string is a u64 length and that many UTF-8 bytes;
//! - a key/value is key, u32 value type, value; an array is u32 element type, u64 count, then
//!   the elements packed (string elements each with their own length);
//! - a tensor info is name, u32 rank, rank u64 dims (innermost first: a row-major `[out, in]`
//!   matrix is listed as `{in, out}`), u32 ggml type, u64 offset from the data section;
//! - the data section starts at the first multiple of `general.alignment` (default 32) after
//!   the tensor infos; every offset is a multiple of the alignment;
//! - a split model `<stem>-NNNNN-of-NNNNN.gguf` keeps the metadata in shard 0 and the keys
//!   `split.no` / `split.count` / `split.tensors.count` in every shard, each tensor in exactly
//!   one shard, with offsets relative to its own shard's data section.
//!
//! Every length, count, size and offset read from a file goes through the verified functions
//! of `verified/gguf_bounds.bend` before it is used, so a truncated or hostile file gives an
//! error naming the file and the key or tensor, never a panic or a read outside the mapping.

pub mod adapter;
pub mod dequant;
pub mod inspect;
pub mod qwen35;
pub mod types;

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::safetensors::MappedShard;
use crate::verified as v;

pub use types::{GgmlType, TypeInfo, ValueType};

/// Largest element count a tensor may have (the reference keeps counts in a signed 64-bit int).
pub const MAX_ELEMS: u64 = i64::MAX as u64;
/// Largest `general.alignment` accepted (a power of two; any power of two up to it is valid).
pub const MAX_ALIGNMENT: u64 = 1 << 30;
/// Default alignment of the data section.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// Largest tensor rank (the reference's limit).
pub const MAX_RANK: u64 = 4;
/// Deepest array-of-array nesting accepted.
pub const MAX_NESTING: u64 = 4;
/// Smallest tensor info record: name length, rank, type, offset.
const MIN_TENSOR_INFO: u64 = 8 + 4 + 4 + 8;
/// Smallest key/value record: key length, value type, a one-byte value.
const MIN_KV: u64 = 8 + 4 + 1;

/// A refused or unreadable GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufError {
    pub file: String,
    pub msg: String,
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.file.is_empty() {
            write!(f, "gguf: {}", self.msg)
        } else {
            write!(f, "gguf: {}: {}", self.file, self.msg)
        }
    }
}

impl std::error::Error for GgufError {}

fn err(file: &str, msg: impl Into<String>) -> GgufError {
    GgufError {
        file: file.to_owned(),
        msg: msg.into(),
    }
}

/// Where a value lies in its file's mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: usize,
    len: usize,
}

/// A metadata value as stored: scalars decoded, strings and arrays left in the mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Stored {
    U64(u64, ValueType),
    I64(i64, ValueType),
    F64(f64, ValueType),
    Bool(bool),
    Str(Span),
    Array {
        elem: ValueType,
        count: u64,
        body: Span,
    },
}

/// A metadata value, borrowed from the file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value<'a> {
    /// Unsigned integers (`u8`, `u16`, `u32`, `u64`); the type says which.
    UInt(u64, ValueType),
    /// Signed integers (`i8` .. `i64`).
    Int(i64, ValueType),
    /// `f32` (widened exactly) or `f64`.
    Float(f64, ValueType),
    Bool(bool),
    Str(&'a str),
    Array(ArrayView<'a>),
}

impl<'a> Value<'a> {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::UInt(_, t) | Value::Int(_, t) | Value::Float(_, t) => *t,
            Value::Bool(_) => ValueType::Bool,
            Value::Str(_) => ValueType::Str,
            Value::Array(_) => ValueType::Array,
        }
    }
    /// The value as an unsigned integer (integers that are not negative, and booleans).
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::UInt(u, _) => Some(u),
            Value::Int(i, _) => u64::try_from(i).ok(),
            Value::Bool(b) => Some(b as u64),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::UInt(u, _) => i64::try_from(u).ok(),
            Value::Int(i, _) => Some(i),
            Value::Bool(b) => Some(b as i64),
            _ => None,
        }
    }
    /// Any number as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::UInt(u, _) => Some(u as f64),
            Value::Int(i, _) => Some(i as f64),
            Value::Float(f, _) => Some(f),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<ArrayView<'a>> {
        match *self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// An array value left in the mapping: elements are decoded only when asked for, so a
/// 250 000-entry vocabulary costs nothing until it is read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrayView<'a> {
    elem: ValueType,
    count: u64,
    body: &'a [u8],
}

impl<'a> ArrayView<'a> {
    pub fn elem_type(&self) -> ValueType {
        self.elem
    }
    pub fn len(&self) -> u64 {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Element `k` of a numeric or boolean array.
    pub fn get(&self, k: u64) -> Option<Value<'a>> {
        let sz = self.elem.fixed_size()?;
        let len = self.body.len() as u64;
        if !v::gg_count_fits(k, sz, len) || !v::gg_span_ok(v::gg_elem_offset(k, sz), sz, len) {
            return None;
        }
        let at = v::gg_elem_offset(k, sz) as usize;
        Some(decode_scalar(self.elem, &self.body[at..at + sz as usize]))
    }
    /// Every element of a numeric array as `i64` (`None` for a float, string or nested array).
    pub fn to_i64(&self) -> Option<Vec<i64>> {
        if self.elem.is_float() || self.elem.fixed_size().is_none() {
            return None;
        }
        (0..self.count).map(|k| self.get(k)?.as_i64()).collect()
    }
    /// Every element of a numeric array as `f64`.
    pub fn to_f64(&self) -> Option<Vec<f64>> {
        self.elem.fixed_size()?;
        (0..self.count).map(|k| self.get(k)?.as_f64()).collect()
    }
    /// The elements of a string array, in order, borrowed from the mapping (lengths were checked
    /// when the file was opened; a string that is not UTF-8 is given as `None`).
    pub fn strings(&self) -> StrIter<'a> {
        StrIter {
            body: if self.elem == ValueType::Str {
                self.body
            } else {
                &[]
            },
            pos: 0,
            left: if self.elem == ValueType::Str {
                self.count
            } else {
                0
            },
        }
    }
    /// The raw bytes of each string element (tokens of a byte-level vocabulary need not be
    /// UTF-8).
    pub fn byte_strings(&self) -> ByteStrIter<'a> {
        ByteStrIter(self.strings())
    }
    /// Element `k` of a string array (walks the array: use [`Self::strings`] to read them all).
    pub fn string_at(&self, k: u64) -> Option<&'a [u8]> {
        self.byte_strings().nth(usize::try_from(k).ok()?)
    }
}

/// Iterator over a string array's elements as `&str` (`None` items for non-UTF-8 strings).
pub struct StrIter<'a> {
    body: &'a [u8],
    pos: u64,
    left: u64,
}

impl<'a> StrIter<'a> {
    fn next_bytes(&mut self) -> Option<&'a [u8]> {
        if self.left == 0 {
            return None;
        }
        let size = self.body.len() as u64;
        if !v::gg_span_ok(self.pos, 8, size) {
            self.left = 0;
            return None;
        }
        let p = self.pos as usize;
        let n = u64::from_le_bytes(self.body[p..p + 8].try_into().ok()?);
        let s = v::gg_pos_add(self.pos, 8);
        if !v::gg_span_ok(s, n, size) {
            self.left = 0;
            return None;
        }
        self.pos = v::gg_pos_add(s, n);
        self.left -= 1;
        Some(&self.body[s as usize..self.pos as usize])
    }
}

impl<'a> Iterator for StrIter<'a> {
    type Item = Option<&'a str>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_bytes().map(|b| std::str::from_utf8(b).ok())
    }
}

/// Iterator over a string array's elements as bytes.
pub struct ByteStrIter<'a>(StrIter<'a>);

impl<'a> Iterator for ByteStrIter<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<&'a [u8]> {
        self.0.next_bytes()
    }
}

fn decode_scalar(t: ValueType, b: &[u8]) -> Value<'_> {
    match t {
        ValueType::U8 => Value::UInt(b[0] as u64, t),
        ValueType::I8 => Value::Int(b[0] as i8 as i64, t),
        ValueType::U16 => Value::UInt(u16::from_le_bytes([b[0], b[1]]) as u64, t),
        ValueType::I16 => Value::Int(i16::from_le_bytes([b[0], b[1]]) as i64, t),
        ValueType::U32 => Value::UInt(u32::from_le_bytes(b[..4].try_into().unwrap()) as u64, t),
        ValueType::I32 => Value::Int(i32::from_le_bytes(b[..4].try_into().unwrap()) as i64, t),
        ValueType::F32 => Value::Float(f32::from_le_bytes(b[..4].try_into().unwrap()) as f64, t),
        ValueType::Bool => Value::Bool(b[0] != 0),
        ValueType::U64 => Value::UInt(u64::from_le_bytes(b[..8].try_into().unwrap()), t),
        ValueType::I64 => Value::Int(i64::from_le_bytes(b[..8].try_into().unwrap()), t),
        ValueType::F64 => Value::Float(f64::from_le_bytes(b[..8].try_into().unwrap()), t),
        // callers pass fixed-size types only
        ValueType::Str | ValueType::Array => Value::Bool(false),
    }
}

/// One tensor of the index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: `dims[0]` is the innermost (contiguous) axis.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset from the start of its shard's data section, as stored.
    pub offset: u64,
    /// Bytes of tensor data.
    pub nbytes: u64,
    /// Elements (product of the dims).
    pub n_elems: u64,
    /// Index of the shard (file) that holds it.
    pub shard: usize,
    /// Offset of its first byte in its shard's mapping.
    abs: u64,
}

impl TensorInfo {
    /// Elements per row (`dims[0]`).
    pub fn row_len(&self) -> u64 {
        self.dims[0]
    }
    /// Rows: every axis but the innermost, flattened.
    pub fn rows(&self) -> u64 {
        self.n_elems / self.dims[0]
    }
    /// Bytes per row.
    pub fn row_bytes(&self) -> u64 {
        let ti = self.ty.info();
        v::gg_nbytes(self.dims[0], ti.block_weights, ti.block_bytes)
    }
    /// The shape in row-major order (outermost first), as safetensors lists it: GGUF dims
    /// reversed. The bytes are the same; nothing is transposed.
    pub fn shape_row_major(&self) -> Vec<u64> {
        self.dims.iter().rev().copied().collect()
    }
}

/// One mapped GGUF file.
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    /// Offset of the data section in the file.
    pub data_start: u64,
    map: MappedShard,
    kvs: Vec<(String, Stored)>,
    tensors: Vec<TensorInfo>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: u64,
    file: &'a str,
}

impl<'a> Cursor<'a> {
    fn size(&self) -> u64 {
        self.b.len() as u64
    }
    fn left(&self) -> u64 {
        self.size().saturating_sub(self.p)
    }
    fn take(&mut self, n: u64, what: &dyn Fn() -> String) -> Result<&'a [u8], GgufError> {
        if !v::gg_span_ok(self.p, n, self.size()) {
            return Err(err(
                self.file,
                format!(
                    "truncated: {} needs {n} bytes at offset {}, the file has {}",
                    what(),
                    self.p,
                    self.size()
                ),
            ));
        }
        let s = self.p as usize;
        self.p = v::gg_pos_add(self.p, n);
        Ok(&self.b[s..self.p as usize])
    }
    fn u32(&mut self, what: &dyn Fn() -> String) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64(&mut self, what: &dyn Fn() -> String) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn str_span(&mut self, what: &dyn Fn() -> String) -> Result<Span, GgufError> {
        let n = self.u64(what)?;
        let start = self.p as usize;
        let b = self.take(n, what)?;
        Ok(Span {
            start,
            len: b.len(),
        })
    }
    fn string(&mut self, what: &dyn Fn() -> String) -> Result<String, GgufError> {
        let sp = self.str_span(what)?;
        std::str::from_utf8(&self.b[sp.start..sp.start + sp.len])
            .map(str::to_owned)
            .map_err(|_| err(self.file, format!("{} is not UTF-8", what())))
    }
    /// Advances over `n` elements of type `et`, checking every length.
    fn skip_array(
        &mut self,
        et: ValueType,
        n: u64,
        depth: u64,
        key: &str,
    ) -> Result<(), GgufError> {
        let what = || format!("array {key}");
        match et {
            ValueType::Str => {
                // every element carries at least its 8-byte length
                if !v::gg_count_fits(n, 8, self.left()) {
                    return Err(err(
                        self.file,
                        format!(
                            "array {key}: {n} strings cannot fit in the {} bytes left",
                            self.left()
                        ),
                    ));
                }
                for _ in 0..n {
                    let len = self.u64(&what)?;
                    self.take(len, &what)?;
                }
            }
            ValueType::Array => {
                if !v::gg_limit_ok(depth, MAX_NESTING) {
                    return Err(err(self.file, format!("array {key}: nested too deeply")));
                }
                // every element carries at least its type and count
                if !v::gg_count_fits(n, 12, self.left()) {
                    return Err(err(
                        self.file,
                        format!(
                            "array {key}: {n} nested arrays cannot fit in the {} bytes left",
                            self.left()
                        ),
                    ));
                }
                for _ in 0..n {
                    let et2 = self.u32(&what)?;
                    let et2 = ValueType::from_id(et2).ok_or_else(|| {
                        err(
                            self.file,
                            format!("array {key}: bad nested element type {et2}"),
                        )
                    })?;
                    let n2 = self.u64(&what)?;
                    self.skip_array(et2, n2, v::gg_count_add(depth, 1), key)?;
                }
            }
            _ => {
                let sz = et.fixed_size().unwrap_or(0);
                if !v::gg_count_fits(n, sz, self.left()) {
                    return Err(err(
                        self.file,
                        format!(
                            "truncated: array {key} of {n} {} elements does not fit in the {} bytes left",
                            et.name(),
                            self.left()
                        ),
                    ));
                }
                self.take(v::gg_elem_offset(n, sz), &what)?;
            }
        }
        Ok(())
    }
}

impl GgufFile {
    /// Maps and parses one GGUF file.
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let name = path.display().to_string();
        let map = MappedShard::open(path).map_err(|e| err(&name, e.to_string()))?;
        let (version, alignment, data_start, kvs, tensors) = parse(map.as_bytes(), &name)?;
        Ok(Self {
            path: path.to_path_buf(),
            version,
            alignment,
            data_start,
            map,
            kvs,
            tensors,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        self.map.as_bytes()
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Keys in file order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.kvs.iter().map(|(k, _)| k.as_str())
    }

    pub fn kv(&self, key: &str) -> Option<Value<'_>> {
        let (_, s) = self.kvs.iter().find(|(k, _)| k == key)?;
        Some(self.value(s))
    }

    fn value(&self, s: &Stored) -> Value<'_> {
        let b = self.bytes();
        match *s {
            Stored::U64(u, t) => Value::UInt(u, t),
            Stored::I64(i, t) => Value::Int(i, t),
            Stored::F64(f, t) => Value::Float(f, t),
            Stored::Bool(x) => Value::Bool(x),
            // checked as UTF-8 when the file was opened
            Stored::Str(sp) => {
                Value::Str(std::str::from_utf8(&b[sp.start..sp.start + sp.len]).unwrap_or(""))
            }
            Stored::Array { elem, count, body } => Value::Array(ArrayView {
                elem,
                count,
                body: &b[body.start..body.start + body.len],
            }),
        }
    }

    fn get_u64(&self, key: &str) -> Option<u64> {
        self.kv(key).and_then(|v| v.as_u64())
    }
}

type Parsed = (u32, u64, u64, Vec<(String, Stored)>, Vec<TensorInfo>);

fn parse(b: &[u8], file: &str) -> Result<Parsed, GgufError> {
    let mut c = Cursor { b, p: 0, file };
    let magic = c.take(4, &|| "magic".into())?;
    if magic != b"GGUF" {
        return Err(err(file, "not a GGUF file (magic)"));
    }
    let version = c.u32(&|| "version".into())?;
    if version != 2 && version != 3 {
        return Err(err(
            file,
            if version == 1 {
                "GGUF version 1 is not supported (32-bit lengths)".to_owned()
            } else if version & 0xFFFF == 0 {
                "big-endian GGUF is not supported".to_owned()
            } else {
                format!("unsupported GGUF version {version}")
            },
        ));
    }
    let nt = c.u64(&|| "tensor count".into())?;
    let nkv = c.u64(&|| "key count".into())?;
    if !v::gg_count_fits(nt, MIN_TENSOR_INFO, c.left()) || !v::gg_count_fits(nkv, MIN_KV, c.left())
    {
        return Err(err(
            file,
            format!(
                "{nt} tensors and {nkv} keys cannot fit in a file of {} bytes",
                c.size()
            ),
        ));
    }
    let mut kvs: Vec<(String, Stored)> = Vec::with_capacity(nkv as usize);
    for k in 0..nkv {
        let key = c.string(&|| format!("key {k}"))?;
        if key.is_empty() {
            return Err(err(file, format!("key {k} is empty")));
        }
        if kvs.iter().any(|(e, _)| *e == key) {
            return Err(err(file, format!("key {key} appears twice")));
        }
        let what = || format!("value of {key}");
        let t = c.u32(&what)?;
        let vt = ValueType::from_id(t)
            .ok_or_else(|| err(file, format!("bad value type {t} for {key}")))?;
        let stored = match vt {
            ValueType::Str => {
                let sp = c.str_span(&what)?;
                if std::str::from_utf8(&b[sp.start..sp.start + sp.len]).is_err() {
                    return Err(err(file, format!("value of {key} is not UTF-8")));
                }
                Stored::Str(sp)
            }
            ValueType::Array => {
                let et = c.u32(&what)?;
                let elem = ValueType::from_id(et)
                    .ok_or_else(|| err(file, format!("bad array element type {et} for {key}")))?;
                let count = c.u64(&what)?;
                let start = c.p as usize;
                c.skip_array(elem, count, 0, &key)?;
                Stored::Array {
                    elem,
                    count,
                    body: Span {
                        start,
                        len: c.p as usize - start,
                    },
                }
            }
            _ => {
                let sz = vt.fixed_size().unwrap_or(0);
                let raw = c.take(sz, &what)?;
                match decode_scalar(vt, raw) {
                    Value::UInt(u, t) => Stored::U64(u, t),
                    Value::Int(i, t) => Stored::I64(i, t),
                    Value::Float(f, t) => Stored::F64(f, t),
                    Value::Bool(x) => Stored::Bool(x),
                    _ => unreachable!("fixed-size types decode to scalars"),
                }
            }
        };
        kvs.push((key, stored));
    }

    let mut alignment = DEFAULT_ALIGNMENT;
    if let Some((_, s)) = kvs.iter().find(|(k, _)| k == "general.alignment") {
        let a = match *s {
            Stored::U64(u, _) => Some(u),
            Stored::I64(i, _) => u64::try_from(i).ok(),
            _ => None,
        };
        match a {
            Some(a) if v::gg_alignment_ok(a, MAX_ALIGNMENT) => alignment = a,
            _ => {
                return Err(err(
                    file,
                    format!("general.alignment {s:?} is not a power of two up to {MAX_ALIGNMENT}"),
                ))
            }
        }
    }

    let mut infos = Vec::with_capacity(nt as usize);
    for k in 0..nt {
        let name = c.string(&|| format!("name of tensor {k}"))?;
        let what = || format!("tensor {name}");
        let nd = c.u32(&what)? as u64;
        if !v::gg_limit_ok(nd, MAX_RANK) || !v::gg_limit_ok(1, nd) {
            return Err(err(
                file,
                format!("tensor {name}: rank {nd} (1 to {MAX_RANK} allowed)"),
            ));
        }
        let mut dims = Vec::with_capacity(nd as usize);
        let mut n: u64 = 1;
        for _ in 0..nd {
            let d = c.u64(&what)?;
            if !v::gg_dim_ok(n, d, MAX_ELEMS) {
                return Err(err(
                    file,
                    format!("tensor {name}: dimension {d} (dims so far {dims:?}) gives a zero or too large element count"),
                ));
            }
            n = v::gg_dim_step(n, d);
            dims.push(d);
        }
        let t = c.u32(&what)?;
        let offset = c.u64(&what)?;
        infos.push((name, dims, n, t, offset));
    }

    let pad = v::gg_pad(c.p, alignment);
    let data_start = if nt > 0 {
        if !v::gg_span_ok(c.p, pad, c.size()) {
            return Err(err(file, "no tensor data after the index"));
        }
        v::gg_pos_add(c.p, pad)
    } else {
        c.p
    };
    let avail = c.size().saturating_sub(data_start);

    let mut tensors = Vec::with_capacity(infos.len());
    for (name, dims, n, t, offset) in infos {
        let ty = GgmlType::from_id(t)
            .ok_or_else(|| err(file, format!("tensor {name}: unknown ggml type {t}")))?;
        let ti = ty.info();
        if !v::gg_blocks_ok(dims[0], ti.block_weights) {
            return Err(err(
                file,
                format!(
                    "tensor {name}: innermost dimension {} is not a multiple of the {} block ({} weights)",
                    dims[0], ti.name, ti.block_weights
                ),
            ));
        }
        if !v::gg_nbytes_ok(dims[0], n, ti.block_weights, ti.block_bytes, avail) {
            return Err(err(
                file,
                format!(
                    "tensor {name}: {n} {} weights do not fit in the {avail}-byte data section",
                    ti.name
                ),
            ));
        }
        let nbytes = v::gg_nbytes(n, ti.block_weights, ti.block_bytes);
        if !v::gg_place_ok(offset, nbytes, alignment, avail) {
            return Err(err(
                file,
                if v::gg_span_ok(offset, nbytes, avail) {
                    format!("tensor {name}: offset {offset} is not a multiple of the alignment {alignment}")
                } else {
                    format!(
                        "tensor {name}: data [{offset}, +{nbytes}) lies outside the {avail}-byte data section"
                    )
                },
            ));
        }
        tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
            nbytes,
            n_elems: n,
            shard: 0,
            abs: v::gg_pos_add(data_start, offset),
        });
    }
    Ok((version, alignment, data_start, kvs, tensors))
}

/// `<stem>-00002-of-00003.gguf` -> (stem, 2, 3).
fn split_name(path: &Path) -> Option<(String, u32, u32)> {
    let s = path.to_str()?;
    let tail = s.get(s.len().checked_sub(20)?..)?; // -NNNNN-of-NNNNN.gguf
    let b = tail.as_bytes();
    if b[0] != b'-' || &tail[6..10] != "-of-" || &tail[15..] != ".gguf" {
        return None;
    }
    let no: u32 = tail[1..6].parse().ok()?;
    let count: u32 = tail[10..15].parse().ok()?;
    Some((s[..s.len() - 20].to_owned(), no, count))
}

fn is_gguf_name(p: &Path) -> bool {
    p.extension().is_some_and(|e| e == "gguf")
}

/// True when `path` is a `.gguf` file, or a directory holding at least one and no
/// `config.json` (a checkpoint directory with a config is read as safetensors).
pub fn looks_like_gguf(path: &Path) -> bool {
    if path.is_dir() {
        if path.join("config.json").is_file() {
            return false;
        }
        list_gguf(path).map(|l| !l.is_empty()).unwrap_or(false)
    } else {
        is_gguf_name(path) && path.is_file()
    }
}

fn list_gguf(dir: &Path) -> Result<Vec<PathBuf>, GgufError> {
    let rd = fs::read_dir(dir).map_err(|e| err(&dir.display().to_string(), e.to_string()))?;
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_gguf_name(p))
        .collect();
    out.sort();
    Ok(out)
}

/// A model: one GGUF file or every shard of a split model.
pub struct Gguf {
    /// What was opened (a file or a directory).
    pub source: PathBuf,
    /// The model's shards in split order (one entry for an unsplit file).
    pub files: Vec<GgufFile>,
    /// Other GGUF files of a directory that are not part of the model: projector files
    /// (`general.architecture` `clip`, or `general.type` `mmproj`). Not opened.
    pub side_files: Vec<PathBuf>,
    by_name: BTreeMap<String, (usize, usize)>,
}

impl fmt::Debug for Gguf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gguf")
            .field("source", &self.source)
            .field("shards", &self.files.len())
            .field("tensors", &self.by_name.len())
            .finish()
    }
}

impl Gguf {
    /// Opens a `.gguf` file (a shard of a split finds its siblings) or a directory of them.
    /// In a directory the projector files are set aside; more than one other model is refused.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let where_ = path.display().to_string();
        let mut files = Vec::new();
        let mut side_files = Vec::new();
        if path.is_dir() {
            let cand = list_gguf(path)?;
            if cand.is_empty() {
                return Err(err(&where_, "no .gguf file in the directory"));
            }
            for c in cand {
                let f = GgufFile::open(&c)?;
                let arch = f
                    .kv("general.architecture")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let ty = f.kv("general.type").and_then(|v| v.as_str()).unwrap_or("");
                if arch == "clip" || ty == "mmproj" {
                    side_files.push(c);
                } else {
                    files.push(f);
                }
            }
            if files.is_empty() {
                return Err(err(
                    &where_,
                    "holds only projector (mmproj) files, no model",
                ));
            }
        } else if let Some((stem, _, count)) = split_name(path) {
            for k in 1..=count {
                files.push(GgufFile::open(Path::new(&format!(
                    "{stem}-{k:05}-of-{count:05}.gguf"
                )))?);
            }
        } else {
            files.push(GgufFile::open(path)?);
        }

        files.sort_by_key(|f| f.get_u64("split.no").unwrap_or(0));
        let count = files[0].get_u64("split.count").unwrap_or(0);
        if count == 0 {
            if files.len() > 1 {
                return Err(err(
                    &where_,
                    format!(
                        "holds {} unrelated models ({}, {}): name one file",
                        files.len(),
                        files[0].path.display(),
                        files[1].path.display()
                    ),
                ));
            }
        } else {
            if !v::gg_split_complete(files.len() as u64, count) {
                return Err(err(
                    &where_,
                    format!("{} of {count} shards present", files.len()),
                ));
            }
            for (k, f) in files.iter().enumerate() {
                let no = f.get_u64("split.no").unwrap_or(u64::MAX);
                let c = f.get_u64("split.count").unwrap_or(0);
                if !v::gg_shard_ok(no, k as u64, count) || c != count {
                    return Err(err(
                        &f.path.display().to_string(),
                        "split.no / split.count do not form one set with the other shards",
                    ));
                }
            }
        }
        let mut by_name = BTreeMap::new();
        let mut total: u64 = 0;
        for (fi, f) in files.iter_mut().enumerate() {
            for (ti, t) in f.tensors.iter_mut().enumerate() {
                t.shard = fi;
                if by_name.insert(t.name.clone(), (fi, ti)).is_some() {
                    return Err(err(
                        &f.path.display().to_string(),
                        format!("tensor {} appears twice", t.name),
                    ));
                }
            }
            total = v::gg_count_add(total, f.tensors.len() as u64);
        }
        if count > 0 {
            if let Some(want) = files[0].kv("split.tensors.count").and_then(|v| v.as_u64()) {
                if want != total {
                    return Err(err(
                        &where_,
                        format!("split.tensors.count is {want}, the shards hold {total}"),
                    ));
                }
            }
        }
        Ok(Self {
            source: path.to_path_buf(),
            files,
            side_files,
            by_name,
        })
    }

    /// Metadata of the model (shard 0).
    pub fn kv(&self, key: &str) -> Option<Value<'_>> {
        self.files[0].kv(key)
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.files[0].keys()
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.kv(key)?.as_str()
    }
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.kv(key)?.as_u64()
    }
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.kv(key)?.as_f64()
    }
    /// `general.architecture` (empty when missing).
    pub fn arch(&self) -> &str {
        self.get_str("general.architecture").unwrap_or("")
    }
    /// `<arch>.<suffix>`, the per-architecture hyper-parameter keys.
    pub fn arch_kv(&self, suffix: &str) -> Option<Value<'_>> {
        self.kv(&format!("{}.{suffix}", self.arch()))
    }

    /// All tensors, shard by shard in file order.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.files.iter().flat_map(|f| f.tensors.iter())
    }
    pub fn tensor_count(&self) -> usize {
        self.by_name.len()
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        let &(f, t) = self.by_name.get(name)?;
        Some(&self.files[f].tensors[t])
    }
    pub fn get(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        self.tensor(name).ok_or_else(|| {
            err(
                &self.source.display().to_string(),
                format!("tensor not found: {name}"),
            )
        })
    }
    /// The stored bytes of a tensor (its range was checked when the file was opened).
    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        let b = self.files[t.shard].bytes();
        &b[t.abs as usize..v::gg_pos_add(t.abs, t.nbytes) as usize]
    }
    /// A small F32 tensor (e.g. an NVFP4 per-expert `.scale`) as values.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let t = self.get(name)?;
        let mut out = vec![0f32; t.n_elems as usize];
        dequant::rows_f32(self, t, 0, t.rows(), &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
