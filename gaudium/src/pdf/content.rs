//! Content streams (ISO 32000-1, sections 8.4, 9.3, 9.4): the text-showing operators and the
//! state they depend on, interpreted to recover a page's text in reading order.
//!
//! Text is placed by its position: a move to another baseline starts a new line, and a gap on
//! the same baseline wider than a fraction of the font size becomes a space (words placed by
//! `Td`, `TJ` adjustments or separate strings). Form XObjects are followed; inline images are
//! skipped; graphics other than text are ignored.

use std::collections::HashMap;
use std::rc::Rc;

use super::doc::Doc;
use super::font::{Font, FontParts};
use super::lex::{Dict, Lexer, Obj, Token};
use super::{Budget, Fail, MAX_FORM_DEPTH, MAX_GSTATE_DEPTH};
use crate::verified as v;

/// Operands kept for one operator (more are dropped from the front).
const MAX_OPERANDS: usize = 64;
/// A gap on the baseline wider than this share of the font size is a space between words.
const SPACE_GAP: f64 = 0.15;
/// A move off the baseline larger than this share of the font size starts a new line.
const NEW_LINE: f64 = 0.5;
/// A move larger than this share of the font size leaves an empty line (a new paragraph).
const NEW_PARAGRAPH: f64 = 1.8;

#[derive(Clone, Copy, Debug, PartialEq)]
struct M {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

impl M {
    const ID: M = M {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    fn from(args: &[Obj]) -> Option<M> {
        let n: Vec<f64> = args.iter().map(Obj::as_f64).collect::<Option<_>>()?;
        let [a, b, c, d, e, f] = n.as_slice() else {
            return None;
        };
        Some(M {
            a: *a,
            b: *b,
            c: *c,
            d: *d,
            e: *e,
            f: *f,
        })
    }

    /// `self` then `o` (row-vector convention: a point p maps to p * self * o).
    fn then(self, o: M) -> M {
        M {
            a: self.a * o.a + self.b * o.c,
            b: self.a * o.b + self.b * o.d,
            c: self.c * o.a + self.d * o.c,
            d: self.c * o.b + self.d * o.d,
            e: self.e * o.a + self.f * o.c + o.e,
            f: self.e * o.b + self.f * o.d + o.f,
        }
    }

    fn translate(tx: f64, ty: f64) -> M {
        M {
            e: tx,
            f: ty,
            ..M::ID
        }
    }
}

#[derive(Clone)]
struct GState {
    ctm: M,
    font: Option<Rc<Font>>,
    size: f64,
    tc: f64,
    tw: f64,
    th: f64,
    tl: f64,
}

impl GState {
    fn new(ctm: M) -> Self {
        GState {
            ctm,
            font: None,
            size: 0.0,
            tc: 0.0,
            tw: 0.0,
            th: 1.0,
            tl: 0.0,
        }
    }
}

/// Fonts already read, by object number (shared by the pages of a document).
pub(crate) type FontCache = HashMap<u32, Rc<Font>>;

/// The text of one page under construction.
pub(crate) struct Page<'d, 'a> {
    pub doc: &'d mut Doc<'a>,
    pub fonts: &'d mut FontCache,
    /// Bytes of text the document may yield.
    pub text_budget: &'d mut Budget,
    pub out: String,
    /// Where the last text shown ended: x, y and the height of its font, in page space.
    last: Option<(f64, f64, f64)>,
}

impl<'d, 'a> Page<'d, 'a> {
    pub fn new(
        doc: &'d mut Doc<'a>,
        fonts: &'d mut FontCache,
        text_budget: &'d mut Budget,
    ) -> Self {
        Page {
            doc,
            fonts,
            text_budget,
            out: String::new(),
            last: None,
        }
    }

    /// Interprets a page's content (`depth` 0) or a form's (deeper). A damaged content stream
    /// ends its interpretation with the text read so far; only a broken limit is an error.
    pub fn run(
        &mut self,
        content: &[u8],
        resources: &Dict,
        ctm: M0,
        depth: u64,
    ) -> Result<(), Fail> {
        let mut gs = GState::new(ctm.0);
        let mut stack: Vec<GState> = Vec::new();
        let mut tm = M::ID;
        let mut tlm = M::ID;
        let mut args: Vec<Obj> = Vec::new();
        let mut lx = Lexer::new(content, 0);
        loop {
            let tok = match lx.token(0, true) {
                Ok(Some(t)) => t,
                Ok(None) => break,
                Err(_) => break,
            };
            let op = match tok {
                Token::Obj(o) => {
                    if args.len() >= MAX_OPERANDS {
                        args.remove(0);
                    }
                    args.push(o);
                    continue;
                }
                Token::Op(op) => op,
            };
            let num = |i: usize| args.get(i).and_then(Obj::as_f64);
            match op.as_slice() {
                b"q" => {
                    if v::pdf_below(stack.len() as u64, MAX_GSTATE_DEPTH) {
                        stack.push(gs.clone());
                    }
                }
                b"Q" => {
                    if let Some(g) = stack.pop() {
                        gs = g;
                    }
                }
                b"cm" => {
                    if let Some(m) = M::from(&args) {
                        gs.ctm = m.then(gs.ctm);
                    }
                }
                b"BT" => {
                    tm = M::ID;
                    tlm = M::ID;
                }
                b"Tf" => {
                    if let (Some(Obj::Name(name)), Some(size)) = (args.first(), num(1)) {
                        let name = name.clone();
                        gs.font = self.font(resources, &name)?;
                        gs.size = size;
                    }
                }
                b"Tc" => gs.tc = num(0).unwrap_or(gs.tc),
                b"Tw" => gs.tw = num(0).unwrap_or(gs.tw),
                b"Tz" => gs.th = num(0).map_or(gs.th, |z| z / 100.0),
                b"TL" => gs.tl = num(0).unwrap_or(gs.tl),
                b"Td" | b"TD" => {
                    if let (Some(tx), Some(ty)) = (num(0), num(1)) {
                        if op == b"TD" {
                            gs.tl = -ty;
                        }
                        tlm = M::translate(tx, ty).then(tlm);
                        tm = tlm;
                    }
                }
                b"Tm" => {
                    if let Some(m) = M::from(&args) {
                        tlm = m;
                        tm = m;
                    }
                }
                b"T*" => {
                    tlm = M::translate(0.0, -gs.tl).then(tlm);
                    tm = tlm;
                }
                b"Tj" => {
                    if let Some(Obj::Str(s)) = args.first() {
                        let s = s.clone();
                        self.show(&s, &gs, &mut tm)?;
                    }
                }
                b"'" | b"\"" => {
                    if op == b"\"" {
                        gs.tw = num(0).unwrap_or(gs.tw);
                        gs.tc = num(1).unwrap_or(gs.tc);
                    }
                    tlm = M::translate(0.0, -gs.tl).then(tlm);
                    tm = tlm;
                    if let Some(Obj::Str(s)) = args.last() {
                        let s = s.clone();
                        self.show(&s, &gs, &mut tm)?;
                    }
                }
                b"TJ" => {
                    if let Some(Obj::Array(items)) = args.first() {
                        let items = items.clone();
                        for it in &items {
                            match it {
                                Obj::Str(s) => self.show(s, &gs, &mut tm)?,
                                o => {
                                    if let Some(n) = o.as_f64() {
                                        let adv = -n / 1000.0 * gs.size * gs.th;
                                        tm = M::translate(adv, 0.0).then(tm);
                                    }
                                }
                            }
                        }
                    }
                }
                b"Do" => {
                    if let Some(Obj::Name(name)) = args.first() {
                        let name = name.clone();
                        self.form(resources, &name, gs.ctm, depth)?;
                    }
                }
                b"BI" => skip_inline_image(&mut lx),
                _ => {}
            }
            args.clear();
        }
        Ok(())
    }

    /// Shows a string: its text, placed after the text before it.
    fn show(&mut self, s: &[u8], gs: &GState, tm: &mut M) -> Result<(), Fail> {
        let Some(font) = gs.font.clone() else {
            return Ok(());
        };
        let trm = tm.then(gs.ctm);
        let height = gs.size.abs() * trm.c.hypot(trm.d);
        let em = gs.size.abs() * trm.a.hypot(trm.b);
        let mut text = String::new();
        font.glyphs(s, |g| {
            if let Some(t) = g.text {
                text.extend(t.chars().map(|c| if c.is_control() { ' ' } else { c }));
            }
            let adv = (g.width * gs.size + gs.tc + if g.space { gs.tw } else { 0.0 }) * gs.th;
            *tm = M::translate(adv, 0.0).then(*tm);
        });
        if text.is_empty() {
            return Ok(());
        }
        self.place(trm.e, trm.f, height, em);
        self.push(&text)?;
        let end = tm.then(gs.ctm);
        self.last = Some((end.e, end.f, height));
        Ok(())
    }

    /// Separates text starting at (x, y) from the text before it: a line break, an empty line,
    /// a space, or nothing.
    fn place(&mut self, x: f64, y: f64, height: f64, em: f64) {
        let Some((lx, ly, lh)) = self.last else {
            return;
        };
        let h = if height > 0.0 { height } else { lh };
        let dy = (y - ly).abs();
        if h > 0.0 && dy > NEW_LINE * h {
            while self.out.ends_with(' ') {
                self.out.pop();
            }
            if !self.out.ends_with('\n') {
                self.out.push('\n');
            }
            if dy > NEW_PARAGRAPH * h && !self.out.ends_with("\n\n") {
                self.out.push('\n');
            }
            return;
        }
        let gap = x - lx;
        let ends_white = self.out.chars().last().is_none_or(char::is_whitespace);
        if !ends_white && em > 0.0 && (gap > SPACE_GAP * em || gap < -em) {
            self.out.push(' ');
        }
    }

    fn push(&mut self, t: &str) -> Result<(), Fail> {
        self.text_budget.take(t.len() as u64)?;
        self.out.push_str(t);
        Ok(())
    }

    /// The font named `name` in the resources (`None`: no such font; its text is not shown).
    fn font(&mut self, resources: &Dict, name: &[u8]) -> Result<Option<Rc<Font>>, Fail> {
        let fonts = soft(self.doc.entry(resources, b"Font"))?;
        let Some(fonts) = fonts.as_dict() else {
            return Ok(None);
        };
        let Some(entry) = fonts.get(name).cloned() else {
            return Ok(None);
        };
        let key = match entry {
            Obj::Ref(n, _) => Some(n),
            _ => None,
        };
        if let Some(f) = key.and_then(|k| self.fonts.get(&k)) {
            return Ok(Some(f.clone()));
        }
        let obj = soft(self.doc.get(&entry))?;
        let Some(dict) = obj.as_dict().cloned() else {
            return Ok(None);
        };
        let font = Rc::new(self.read_font(&dict)?);
        if let Some(k) = key {
            self.fonts.insert(k, font.clone());
        }
        Ok(Some(font))
    }

    fn read_font(&mut self, dict: &Dict) -> Result<Font, Fail> {
        let to_unicode = match soft(self.doc.entry(dict, b"ToUnicode"))? {
            Obj::Stream(s) => soft(self.doc.decode(&s, 0).map(Some))?,
            _ => None,
        };
        let composite = dict.get(b"Subtype").and_then(Obj::as_name) == Some(b"Type0");
        let mut encoding = None;
        let mut encoding_cmap = None;
        match soft(self.doc.entry(dict, b"Encoding"))? {
            Obj::Name(n) => encoding = Some(Obj::Name(n)),
            Obj::Stream(s) if composite => {
                encoding_cmap = soft(self.doc.decode(&s, 0).map(Some))?;
            }
            Obj::Dict(d) => {
                let mut e = Dict::default();
                if let Some(b) = d.get(b"BaseEncoding") {
                    e.0.push((b"BaseEncoding".to_vec(), soft(self.doc.get(b))?));
                }
                let diff = soft(self.doc.entry(&d, b"Differences"))?;
                e.0.push((b"Differences".to_vec(), diff));
                encoding = Some(Obj::Dict(e));
            }
            _ => {}
        }
        let mut widths = None;
        let mut default_width = None;
        if composite {
            let desc = soft(self.doc.entry(dict, b"DescendantFonts"))?;
            let first = desc.as_array().and_then(|a| a.first()).cloned();
            if let Some(first) = first {
                if let Some(df) = soft(self.doc.get(&first))?.as_dict().cloned() {
                    if let Obj::Array(w) = soft(self.doc.entry(&df, b"W"))? {
                        let mut out = Vec::with_capacity(w.len());
                        for x in w {
                            out.push(soft(self.doc.get(&x))?);
                        }
                        widths = Some(out);
                    }
                    default_width = soft(self.doc.entry(&df, b"DW"))?.as_f64();
                }
            }
        } else {
            if let Obj::Array(w) = soft(self.doc.entry(dict, b"Widths"))? {
                widths = Some(w);
            }
            if let Some(fd) = soft(self.doc.entry(dict, b"FontDescriptor"))?
                .as_dict()
                .cloned()
            {
                default_width = soft(self.doc.entry(&fd, b"MissingWidth"))?
                    .as_f64()
                    .filter(|w| *w > 0.0);
            }
        }
        Ok(Font::new(FontParts {
            dict,
            to_unicode,
            encoding,
            encoding_cmap,
            widths,
            default_width,
        }))
    }

    /// Follows `Do` into a form XObject.
    fn form(&mut self, resources: &Dict, name: &[u8], ctm: M, depth: u64) -> Result<(), Fail> {
        if !v::pdf_below(depth + 1, MAX_FORM_DEPTH) {
            return Ok(());
        }
        let xobjects = soft(self.doc.entry(resources, b"XObject"))?;
        let Some(entry) = xobjects.as_dict().and_then(|d| d.get(name)).cloned() else {
            return Ok(());
        };
        let Obj::Stream(s) = soft(self.doc.get(&entry))? else {
            return Ok(());
        };
        if s.dict.get(b"Subtype").and_then(Obj::as_name) != Some(b"Form") {
            return Ok(());
        }
        // each use decodes the form again, so repeated forms are charged to the budget
        let Some(data) = soft(self.doc.decode(&s, 0).map(Some))? else {
            return Ok(());
        };
        let own = match soft(self.doc.entry(&s.dict, b"Resources"))? {
            Obj::Dict(d) => d,
            _ => resources.clone(),
        };
        let matrix = s
            .dict
            .get(b"Matrix")
            .and_then(Obj::as_array)
            .and_then(M::from)
            .unwrap_or(M::ID);
        self.run(&data, &own, M0(matrix.then(ctm)), depth + 1)
    }
}

/// The page space a content stream starts in.
#[derive(Clone, Copy)]
pub(crate) struct M0(M);

impl M0 {
    pub const ID: M0 = M0(M::ID);
}

/// Keeps a broken limit as the error and turns other failures into nothing: a damaged font or
/// form loses its text, not the document's.
fn soft<T: Default>(r: Result<T, Fail>) -> Result<T, Fail> {
    match r {
        Ok(x) => Ok(x),
        Err(e @ Fail::TooLarge(_)) => Err(e),
        Err(_) => Ok(T::default()),
    }
}

/// Skips an inline image after `BI`: its parameters, `ID`, the image bytes and `EI`.
fn skip_inline_image(lx: &mut Lexer) {
    loop {
        match lx.token(0, true) {
            Ok(Some(Token::Op(op))) if op == b"ID" => break,
            Ok(Some(_)) => {}
            _ => return,
        }
    }
    // one white-space byte, then data up to `EI` between white space
    lx.pos += 1;
    let d = lx.data;
    let mut i = lx.pos;
    while i + 1 < d.len() {
        if d[i] == b'E'
            && d[i + 1] == b'I'
            && i > 0
            && super::lex::is_white(d[i - 1])
            && d.get(i + 2).is_none_or(|b| super::lex::is_white(*b))
        {
            lx.pos = i + 2;
            return;
        }
        i += 1;
    }
    lx.pos = d.len();
}
