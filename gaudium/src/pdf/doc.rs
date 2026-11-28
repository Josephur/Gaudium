//! The file structure (ISO 32000-1, section 7.5): the cross-reference data (xref tables, xref
//! streams, hybrid files, /Prev chains), indirect objects, object streams, and the trailer.
//!
//! A file whose cross-reference data is missing or damaged is read by scanning it for
//! `N G obj` headers instead. Every offset, length, count and depth taken from the file goes
//! through the verified core before it is used.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::lex::{Dict, Lexer, Obj, Stream};
use super::{Budget, Fail, MAX_REF_DEPTH, MAX_XREF_SECTIONS};
use crate::verified as v;

/// Widest field of an xref stream row, in bytes (a field is assembled into a `u64`).
const MAX_XREF_FIELD: u64 = 8;
/// Fewest bytes one xref table entry takes (`O G n` without its end of line).
const MIN_XREF_ENTRY: u64 = 5;
/// Fewest bytes one object-number / offset pair of an object stream header takes.
const MIN_OBJSTM_PAIR: u64 = 2;
/// Bytes from the end of the file searched for `startxref`.
const TAIL: usize = 4096;

#[derive(Clone, Copy, Debug)]
enum Loc {
    /// At a byte offset of the file.
    At(u64),
    /// Entry `index` of the object stream `stream`.
    InStream(u32, u64),
}

struct ObjStm {
    data: Rc<[u8]>,
    first: u64,
    /// The objects' offsets after `first`, in entry order.
    offsets: Vec<u64>,
}

pub(crate) struct Doc<'a> {
    data: &'a [u8],
    xref: HashMap<u32, Loc>,
    pub trailer: Dict,
    cache: HashMap<u32, Obj>,
    objstm: HashMap<u32, Rc<ObjStm>>,
    /// Offsets of every `endstream` keyword (found once, for streams whose /Length is wrong).
    endstreams: Option<Vec<usize>>,
    pub budget: Budget,
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= hay.len() {
        return None;
    }
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

fn rfind(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).rposition(|w| w == needle)
}

fn non_negative(o: Option<&Obj>) -> Option<u64> {
    o.and_then(Obj::as_int).and_then(|i| u64::try_from(i).ok())
}

impl<'a> Doc<'a> {
    pub fn open(data: &'a [u8], budget: Budget) -> Result<Doc<'a>, Fail> {
        let head = &data[..data.len().min(1024)];
        if find(head, b"%PDF-", 0).is_none() {
            return Err(Fail::NotPdf);
        }
        let mut doc = Doc {
            data,
            xref: HashMap::new(),
            trailer: Dict::default(),
            cache: HashMap::new(),
            objstm: HashMap::new(),
            endstreams: None,
            budget,
        };
        let read = doc.read_xref();
        if read.is_err() || doc.trailer.get(b"Root").is_none() {
            if let Err(e @ Fail::TooLarge(_)) = read {
                return Err(e);
            }
            doc.xref.clear();
            doc.cache.clear();
            doc.reconstruct()?;
        }
        if !matches!(doc.trailer.get(b"Encrypt"), None | Some(Obj::Null)) {
            return Err(Fail::Encrypted);
        }
        Ok(doc)
    }

    // ---------------------------------------------------------------------------------------
    // Cross-reference data
    // ---------------------------------------------------------------------------------------

    fn read_xref(&mut self) -> Result<(), Fail> {
        let len = self.data.len();
        let tail_from = len.saturating_sub(TAIL);
        let at = rfind(&self.data[tail_from..], b"startxref")
            .map(|p| p + tail_from)
            .ok_or_else(|| Fail::damaged("no startxref"))?;
        let mut lx = Lexer::new(self.data, at + b"startxref".len());
        let mut next = lx.read_uint();
        let mut seen: HashSet<u64> = HashSet::new();
        let mut sections = 0u64;
        while let Some(off) = next {
            if !seen.insert(off) {
                break;
            }
            if !v::pdf_below(sections, MAX_XREF_SECTIONS) {
                return Err(Fail::damaged(format!(
                    "more than {MAX_XREF_SECTIONS} cross-reference sections"
                )));
            }
            sections += 1;
            if !v::pdf_below(off, len as u64) {
                return Err(Fail::damaged(format!(
                    "a cross-reference offset ({off}) lies past the end of the file ({len} bytes)"
                )));
            }
            let mut lx = Lexer::new(self.data, off as usize);
            let trailer = if lx.eat_keyword(b"xref") {
                let t = self.xref_table(&mut lx)?;
                if let Some(stm) = non_negative(t.get(b"XRefStm")) {
                    if v::pdf_below(stm, len as u64) {
                        // a hybrid file: the stream's entries count before the older sections
                        let _ = self.xref_stream_at(stm);
                    }
                }
                t
            } else {
                self.xref_stream_at(off)?
            };
            next = non_negative(trailer.get(b"Prev"));
            for (k, val) in trailer.0 {
                if self.trailer.get(&k).is_none() && k != b"Prev" {
                    self.trailer.0.push((k, val));
                }
            }
        }
        Ok(())
    }

    fn insert(&mut self, num: u32, loc: Loc) {
        // newer sections are read first, and their entries win
        self.xref.entry(num).or_insert(loc);
    }

    /// A classic xref table at the cursor (after `xref`), and the trailer that follows it.
    fn xref_table(&mut self, lx: &mut Lexer) -> Result<Dict, Fail> {
        loop {
            let save = lx.pos;
            let (Some(start), Some(count)) = (lx.read_uint(), lx.read_uint()) else {
                lx.pos = save;
                break;
            };
            let left = (lx.data.len() - lx.pos.min(lx.data.len())) as u64;
            if !v::pdf_count_fits(count, MIN_XREF_ENTRY, left) {
                return Err(Fail::damaged(format!(
                    "an xref subsection of {count} entries does not fit in the {left} bytes left"
                )));
            }
            for k in 0..count {
                let (Some(off), Some(_gen)) = (lx.read_uint(), lx.read_uint()) else {
                    return Err(Fail::damaged("a damaged xref entry"));
                };
                let in_use = if lx.eat_keyword(b"n") {
                    true
                } else if lx.eat_keyword(b"f") {
                    false
                } else {
                    return Err(Fail::damaged("a damaged xref entry"));
                };
                let num = start.checked_add(k).and_then(|n| u32::try_from(n).ok());
                if let (true, Some(num)) = (in_use && off > 0, num) {
                    self.insert(num, Loc::At(off));
                }
            }
        }
        if !lx.eat_keyword(b"trailer") {
            return Err(Fail::damaged("no trailer after the xref table"));
        }
        match lx.object(0)? {
            Obj::Dict(d) => Ok(d),
            _ => Err(Fail::damaged("the trailer is not a dictionary")),
        }
    }

    /// The xref stream at file offset `off`; returns its dictionary (the trailer).
    fn xref_stream_at(&mut self, off: u64) -> Result<Dict, Fail> {
        let Some((_, obj)) = self.parse_at(off, 0)? else {
            return Err(Fail::damaged("no cross-reference data at startxref"));
        };
        let Obj::Stream(s) = obj else {
            return Err(Fail::damaged("no cross-reference data at startxref"));
        };
        if s.dict.get(b"Type").and_then(Obj::as_name) != Some(b"XRef") {
            return Err(Fail::damaged("no cross-reference data at startxref"));
        }
        self.xref_stream(&s)?;
        Ok(s.dict.clone())
    }

    fn xref_stream(&mut self, s: &Stream) -> Result<(), Fail> {
        let data = self.decode(s, 0)?;
        let w: Vec<u64> = match s.dict.get(b"W") {
            Some(Obj::Array(a)) if a.len() == 3 => a
                .iter()
                .map(|x| non_negative(Some(x)))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Fail::damaged("bad /W in an xref stream"))?,
            _ => return Err(Fail::damaged("bad /W in an xref stream")),
        };
        if !v::pdf_widths_ok(w[0], w[1], w[2], MAX_XREF_FIELD) {
            return Err(Fail::damaged("an xref stream field is wider than 8 bytes"));
        }
        let width = w[0] + w[1] + w[2];
        let size = non_negative(s.dict.get(b"Size")).unwrap_or(0);
        let index: Vec<u64> = match s.dict.get(b"Index") {
            Some(Obj::Array(a)) => a.iter().filter_map(|x| non_negative(Some(x))).collect(),
            _ => vec![0, size],
        };
        let mut rest: &[u8] = &data;
        for pair in index.chunks(2) {
            let [start, count] = pair else { break };
            if !v::pdf_count_fits(*count, width, rest.len() as u64) {
                return Err(Fail::damaged(format!(
                    "an xref stream subsection of {count} rows of {width} bytes does not fit in its {} bytes",
                    rest.len()
                )));
            }
            for k in 0..*count {
                let at = v::pdf_row_offset(k, width) as usize;
                let row = rest
                    .get(at..at + width as usize)
                    .ok_or_else(|| Fail::damaged("an xref stream row out of range"))?;
                let field = |from: usize, n: u64| {
                    row[from..from + n as usize]
                        .iter()
                        .fold(0u64, |a, &b| a << 8 | u64::from(b))
                };
                let (a, b) = (w[0] as usize, w[1] as usize);
                let kind = if w[0] == 0 { 1 } else { field(0, w[0]) };
                let f2 = field(a, w[1]);
                let f3 = field(a + b, w[2]);
                let Some(num) = start.checked_add(k).and_then(|n| u32::try_from(n).ok()) else {
                    continue;
                };
                match kind {
                    1 if f2 > 0 => self.insert(num, Loc::At(f2)),
                    2 => {
                        if let Ok(stm) = u32::try_from(f2) {
                            self.insert(num, Loc::InStream(stm, f3));
                        }
                    }
                    _ => {}
                }
            }
            let used = v::pdf_row_offset(*count, width) as usize;
            rest = rest.get(used..).unwrap_or(&[]);
        }
        Ok(())
    }

    /// Rebuilds the cross-reference data by scanning the file for `N G obj` headers.
    fn reconstruct(&mut self) -> Result<(), Fail> {
        let data = self.data;
        let mut i = 0usize;
        let mut trailer: Option<Dict> = None;
        while let Some(p) = find(data, b"obj", i) {
            i = p + 3;
            // `obj` as a whole word, after `N G `
            if data
                .get(p + 3)
                .is_some_and(|b| !super::lex::is_white(*b) && !super::lex::is_delim(*b))
            {
                continue;
            }
            let mut j = p;
            let digits = |j: &mut usize| {
                let end = *j;
                while *j > 0 && data[*j - 1].is_ascii_digit() {
                    *j -= 1;
                }
                *j < end
            };
            let ws = |j: &mut usize| {
                let end = *j;
                while *j > 0 && super::lex::is_white(data[*j - 1]) {
                    *j -= 1;
                }
                *j < end
            };
            if !(ws(&mut j) && digits(&mut j) && ws(&mut j) && digits(&mut j)) {
                continue;
            }
            let mut lx = Lexer::new(data, j);
            let (Some(num), Some(_)) = (lx.read_uint(), lx.read_uint()) else {
                continue;
            };
            if let Ok(num) = u32::try_from(num) {
                // a later definition of an object replaces an earlier one
                self.xref.insert(num, Loc::At(j as u64));
            }
        }
        // the trailer: the last `trailer` dictionary, else an xref stream's dictionary
        if let Some(t) = rfind(data, b"trailer") {
            let mut lx = Lexer::new(data, t + b"trailer".len());
            if let Ok(Obj::Dict(d)) = lx.object(0) {
                trailer = Some(d);
            }
        }
        if trailer.as_ref().is_none_or(|t| t.get(b"Root").is_none()) {
            let mut nums: Vec<u32> = self.xref.keys().copied().collect();
            nums.sort_unstable();
            for num in nums {
                let Ok(o) = self.resolve(num, 0) else {
                    continue;
                };
                if let Obj::Stream(s) = &o {
                    if s.dict.get(b"Type").and_then(Obj::as_name) == Some(b"XRef") {
                        // entries of objects inside object streams; scanned offsets stay
                        if let Err(e @ Fail::TooLarge(_)) = self.xref_stream(s) {
                            return Err(e);
                        }
                    }
                }
                if let Some(d) = o.as_dict() {
                    let ty = d.get(b"Type").and_then(Obj::as_name);
                    if ty == Some(b"XRef")
                        && d.get(b"Root").is_some()
                        && trailer.as_ref().is_none_or(|t| t.get(b"Root").is_none())
                    {
                        trailer = Some(d.clone());
                    }
                    if ty == Some(b"Catalog")
                        && trailer.as_ref().is_none_or(|t| t.get(b"Root").is_none())
                    {
                        trailer = Some(Dict(vec![(b"Root".to_vec(), Obj::Ref(num, 0))]));
                    }
                }
            }
        }
        match trailer {
            Some(t) if t.get(b"Root").is_some() => {
                self.trailer = t;
                Ok(())
            }
            _ => Err(Fail::damaged("no document catalog")),
        }
    }

    // ---------------------------------------------------------------------------------------
    // Objects
    // ---------------------------------------------------------------------------------------

    /// The indirect object at file offset `off`: its number and the object (`None` when no
    /// object header is there).
    fn parse_at(&mut self, off: u64, depth: u64) -> Result<Option<(u32, Obj)>, Fail> {
        let len = self.data.len() as u64;
        if !v::pdf_below(off, len) {
            return Ok(None);
        }
        let mut lx = Lexer::new(self.data, off as usize);
        let (Some(num), Some(_gen)) = (lx.read_uint(), lx.read_uint()) else {
            return Ok(None);
        };
        if !lx.eat_keyword(b"obj") {
            return Ok(None);
        }
        let Ok(num) = u32::try_from(num) else {
            return Ok(None);
        };
        let obj = lx.object(0)?;
        let Obj::Dict(dict) = obj else {
            return Ok(Some((num, obj)));
        };
        if !lx.eat_keyword(b"stream") {
            return Ok(Some((num, Obj::Dict(dict))));
        }
        // the data starts after the end of line that follows the keyword
        let mut start = lx.pos;
        if self.data.get(start) == Some(&b'\r') {
            start += 1;
        }
        if self.data.get(start) == Some(&b'\n') {
            start += 1;
        }
        let declared = match dict.get(b"Length") {
            Some(Obj::Ref(n, _)) if *n != num => {
                let n = *n;
                non_negative(Some(&self.resolve(n, depth + 1)?))
            }
            other => non_negative(other),
        };
        let span = declared.filter(|&l| {
            v::pdf_span_ok(start as u64, l, len) && {
                // the length is right when `endstream` follows it
                let mut after = Lexer::new(self.data, start + l as usize);
                after.eat_keyword(b"endstream")
            }
        });
        let end = match span {
            Some(l) => start + l as usize,
            None => {
                let ends = self.endstreams.get_or_insert_with(|| {
                    let mut v = Vec::new();
                    let mut i = 0;
                    while let Some(p) = find(self.data, b"endstream", i) {
                        v.push(p);
                        i = p + 1;
                    }
                    v
                });
                let k = ends.partition_point(|&p| p < start);
                let mut e = ends.get(k).copied().unwrap_or(self.data.len());
                // the end of line before `endstream` is not data
                if e > start && self.data[e - 1] == b'\n' {
                    e -= 1;
                }
                if e > start && self.data[e - 1] == b'\r' {
                    e -= 1;
                }
                e
            }
        };
        let data: Rc<[u8]> = Rc::from(self.data.get(start..end).unwrap_or(&[]));
        Ok(Some((num, Obj::Stream(Rc::new(Stream { dict, data })))))
    }

    /// Object `num` (null when the file has no such object).
    pub fn resolve(&mut self, num: u32, depth: u64) -> Result<Obj, Fail> {
        if !v::pdf_below(depth, MAX_REF_DEPTH) {
            return Err(Fail::damaged(format!(
                "references nested more than {MAX_REF_DEPTH} deep"
            )));
        }
        if let Some(o) = self.cache.get(&num) {
            return Ok(o.clone());
        }
        // a cycle back to this object sees null
        self.cache.insert(num, Obj::Null);
        let obj = match self.xref.get(&num).copied() {
            None => Obj::Null,
            Some(Loc::At(off)) => match self.parse_at(off, depth)? {
                Some((n, o)) if n == num => o,
                _ => Obj::Null,
            },
            Some(Loc::InStream(stm, index)) => self.from_objstm(stm, index, depth)?,
        };
        self.cache.insert(num, obj.clone());
        Ok(obj)
    }

    /// `o` with references followed.
    pub fn get(&mut self, o: &Obj) -> Result<Obj, Fail> {
        let mut cur = o.clone();
        let mut depth = 0u64;
        while let Obj::Ref(n, _) = cur {
            cur = self.resolve(n, depth)?;
            depth += 1;
        }
        Ok(cur)
    }

    /// A dictionary entry with references followed.
    pub fn entry(&mut self, d: &Dict, key: &[u8]) -> Result<Obj, Fail> {
        match d.get(key) {
            Some(o) => self.get(o),
            None => Ok(Obj::Null),
        }
    }

    fn from_objstm(&mut self, stm: u32, index: u64, depth: u64) -> Result<Obj, Fail> {
        let os = match self.objstm.get(&stm) {
            Some(os) => os.clone(),
            None => {
                // an object stream is itself never inside another one
                let Some(Loc::At(_)) = self.xref.get(&stm).copied() else {
                    return Ok(Obj::Null);
                };
                let Obj::Stream(s) = self.resolve(stm, depth + 1)? else {
                    return Ok(Obj::Null);
                };
                if s.dict.get(b"Type").and_then(Obj::as_name) != Some(b"ObjStm") {
                    return Ok(Obj::Null);
                }
                let data: Rc<[u8]> = Rc::from(self.decode(&s, depth + 1)?);
                let dlen = data.len() as u64;
                let n = non_negative(s.dict.get(b"N")).unwrap_or(0);
                let first = non_negative(s.dict.get(b"First")).unwrap_or(0);
                if !v::pdf_span_ok(0, first, dlen) {
                    return Err(Fail::damaged(format!(
                        "an object stream's header ({first} bytes) is longer than the stream ({dlen} bytes)"
                    )));
                }
                if !v::pdf_count_fits(n, MIN_OBJSTM_PAIR, first) {
                    return Err(Fail::damaged(format!(
                        "an object stream claims {n} objects in a header of {first} bytes"
                    )));
                }
                let mut lx = Lexer::new(&data[..first as usize], 0);
                let mut offsets = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    let (Some(_num), Some(off)) = (lx.read_uint(), lx.read_uint()) else {
                        break;
                    };
                    offsets.push(off);
                }
                let os = Rc::new(ObjStm {
                    data,
                    first,
                    offsets,
                });
                self.objstm.insert(stm, os.clone());
                os
            }
        };
        let Some(&off) = usize::try_from(index).ok().and_then(|i| os.offsets.get(i)) else {
            return Ok(Obj::Null);
        };
        if !v::pdf_objstm_ok(os.first, off, os.data.len() as u64) {
            return Err(Fail::damaged(format!(
                "an object at offset {off} lies outside its object stream"
            )));
        }
        let pos = v::pdf_objstm_pos(os.first, off) as usize;
        let mut lx = Lexer::new(&os.data, pos);
        lx.object(0)
    }

    /// The decoded data of a stream, charged to the decompression budget.
    pub fn decode(&mut self, s: &Stream, depth: u64) -> Result<Vec<u8>, Fail> {
        // filters and their parameters may be indirect
        let mut dict = Dict::default();
        for key in [&b"Filter"[..], b"DecodeParms", b"F", b"DP"] {
            if let Some(o) = s.dict.get(key) {
                let mut val = self.get_depth(o, depth)?;
                if let Obj::Array(items) = &mut val {
                    for it in items.iter_mut() {
                        if let Obj::Ref(..) = it {
                            *it = self.get_depth(it, depth)?;
                        }
                    }
                }
                dict.0.push((key.to_vec(), val));
            }
        }
        super::filter::decode(&dict, &s.data, &mut self.budget)
    }

    fn get_depth(&mut self, o: &Obj, depth: u64) -> Result<Obj, Fail> {
        match o {
            Obj::Ref(n, _) => self.resolve(*n, depth + 1),
            _ => Ok(o.clone()),
        }
    }
}
