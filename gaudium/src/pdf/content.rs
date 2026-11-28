//! Content streams (ISO 32000-1, sections 8.4, 9.3, 9.4): the text-showing operators and the
//! state they depend on, interpreted to recover a page's text in reading order.
//!
//! Text is placed by its position: a move to another baseline starts a new line, and a gap on
//! the same baseline wider than a fraction of the font size becomes a space (words placed by
//! `Td`, `TJ` adjustments or separate strings). Both are measured along the baseline of the
//! text before and across it, so rotated text (a table heading set vertically) splits into
//! words and lines as it would unrotated. Form XObjects are followed; inline images are
//! skipped; graphics other than text are ignored. A form starts in the graphics state of the
//! `Do` that draws it (its font and text state included), and a font may also be selected by an
//! ExtGState's /Font entry (`gs`).
//!
//! What one document may make the interpreter do is bounded: every content stream it reads is a
//! fresh decode charged to the decode budget, so the bytes interpreted never exceed it; every
//! form XObject drawn is charged to the form budget (`--pdf-max-forms`), so a form drawing
//! itself many times stops at the cap; the operands held at any time are charged to the room
//! the object budget leaves; every operator costs a step of the work budget; the text of each
//! glyph is charged to the text budget before it is kept. Resources and objects are shared
//! (`Rc`), and looking up a font or a form by name does not copy the resource dictionary.

use std::collections::HashMap;
use std::rc::Rc;

use super::doc::Doc;
use super::font::{Font, FontParts};
use super::lex::{is_white, Dict, Lexer, Obj, Token};
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

/// Fonts already read (shared by the pages of a document): by object number, and direct font
/// dictionaries by their shared allocation (kept alive here, so the address is not reused).
#[derive(Default)]
pub(crate) struct FontCache {
    by_ref: HashMap<u32, Rc<Font>>,
    direct: HashMap<*const Dict, (Rc<Dict>, Rc<Font>)>,
}

/// The text of one page under construction.
pub(crate) struct Page<'d, 'a> {
    pub doc: &'d mut Doc<'a>,
    pub fonts: &'d mut FontCache,
    /// Bytes of text the document may yield.
    pub text_budget: &'d mut Budget,
    /// Form XObjects the document may still draw.
    pub forms: &'d mut Budget,
    pub out: String,
    /// Where the last text shown ended, in page space, the height of its font, and the
    /// direction of its baseline.
    last: Option<Last>,
}

#[derive(Clone, Copy)]
struct Last {
    x: f64,
    y: f64,
    height: f64,
    /// The baseline's direction in page space (a unit vector).
    ux: f64,
    uy: f64,
}

/// The unit vector along the baseline of text drawn with the matrix `m` (text space x in page
/// space); x when the matrix flattens it.
fn baseline(m: &M) -> (f64, f64) {
    let n = m.a.hypot(m.b);
    if n > 0.0 && n.is_finite() {
        (m.a / n, m.b / n)
    } else {
        (1.0, 0.0)
    }
}

impl<'d, 'a> Page<'d, 'a> {
    pub fn new(
        doc: &'d mut Doc<'a>,
        fonts: &'d mut FontCache,
        text_budget: &'d mut Budget,
        forms: &'d mut Budget,
    ) -> Self {
        Page {
            doc,
            fonts,
            text_budget,
            forms,
            out: String::new(),
            last: None,
        }
    }

    /// Interprets a page's content (`depth` 0) or a form's (deeper), from the graphics state
    /// `start`. A damaged content stream ends its interpretation with the text read so far;
    /// only a broken limit is an error.
    pub fn run(
        &mut self,
        content: &[u8],
        resources: &Rc<Dict>,
        start: Start,
        depth: u64,
    ) -> Result<(), Fail> {
        let mut gs = start.0;
        let mut stack: Vec<GState> = Vec::new();
        let mut tm = M::ID;
        let mut tlm = M::ID;
        let mut args: Vec<Obj> = Vec::new();
        // the operands held between two operators may take the room the object budget leaves
        let mut lx = Lexer::new(content, 0, self.doc.objects);
        loop {
            let tok = match lx.token(0, true) {
                Ok(Some(t)) => t,
                Ok(None) => break,
                Err(e @ Fail::TooLarge(_)) => return Err(e),
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
            // every operator is a step; its operands are dropped once it has run, and the next
            // operands may take only the room the document leaves after it (an operator that
            // reads from the document, a font or a form, drops them before it reads)
            self.doc.work.take(1)?;
            if op == b"Do" {
                let name = match args.first() {
                    Some(Obj::Name(n)) => Some(n.clone()),
                    _ => None,
                };
                args.clear();
                if let Some(name) = name {
                    self.form(resources, &name, &gs, depth)?;
                }
                lx.restart(self.doc.objects);
                continue;
            }
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
                        // the operands are dropped before the font is read: what the font
                        // keeps takes the room they took
                        args.clear();
                        gs.font = self.font(resources, &name)?;
                        gs.size = size;
                    }
                }
                b"gs" => {
                    if let Some(Obj::Name(name)) = args.first() {
                        let name = name.clone();
                        args.clear();
                        if let Some((font, size)) = self.state_font(resources, &name)? {
                            gs.font = font;
                            gs.size = size;
                        }
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
                        for it in items.iter() {
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
                b"BI" => skip_inline_image(&mut lx, &mut self.doc.work)?,
                _ => {}
            }
            args.clear();
            lx.restart(self.doc.objects);
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
        let budget = &mut *self.text_budget;
        font.glyphs(s, &mut self.doc.work, |g| {
            if let Some(t) = g.text {
                // charged before it is kept: one code may map to a long text
                budget.take(t.len() as u64)?;
                text.extend(t.chars().map(|c| if c.is_control() { ' ' } else { c }));
            }
            let adv = (g.width * gs.size + gs.tc + if g.space { gs.tw } else { 0.0 }) * gs.th;
            *tm = M::translate(adv, 0.0).then(*tm);
            Ok(())
        })?;
        if text.is_empty() {
            return Ok(());
        }
        self.place(trm.e, trm.f, height, em);
        self.out.push_str(&text);
        let end = tm.then(gs.ctm);
        let (ux, uy) = baseline(&trm);
        self.last = Some(Last {
            x: end.e,
            y: end.f,
            height,
            ux,
            uy,
        });
        Ok(())
    }

    /// Separates text starting at (x, y) from the text before it: a line break, an empty line,
    /// a space, or nothing. The move is measured along the baseline of the text before (the
    /// gap) and across it (the change of line).
    fn place(&mut self, x: f64, y: f64, height: f64, em: f64) {
        let Some(last) = self.last else {
            return;
        };
        let h = if height > 0.0 { height } else { last.height };
        let (mx, my) = (x - last.x, y - last.y);
        let gap = mx * last.ux + my * last.uy;
        let dy = (my * last.ux - mx * last.uy).abs();
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
        let ends_white = self.out.chars().last().is_none_or(char::is_whitespace);
        if !ends_white && em > 0.0 && (gap > SPACE_GAP * em || gap < -em) {
            self.out.push(' ');
        }
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
        self.font_at(entry)
    }

    /// The font and size an ExtGState named `name` selects with its /Font entry (`None`: it
    /// selects none).
    fn state_font(
        &mut self,
        resources: &Dict,
        name: &[u8],
    ) -> Result<Option<(Option<Rc<Font>>, f64)>, Fail> {
        let states = soft(self.doc.entry(resources, b"ExtGState"))?;
        let Some(entry) = states.as_dict().and_then(|d| d.get(name)).cloned() else {
            return Ok(None);
        };
        let state = soft(self.doc.get(&entry))?;
        let Some(state) = state.as_dict() else {
            return Ok(None);
        };
        let spec = soft(self.doc.entry(state, b"Font"))?;
        let Some([font, size]) = spec.as_array() else {
            return Ok(None);
        };
        let Some(size) = soft(self.doc.get(size))?.as_f64() else {
            return Ok(None);
        };
        Ok(Some((self.font_at(font.clone())?, size)))
    }

    /// The font a resource entry names (a reference or a direct dictionary), read once.
    fn font_at(&mut self, entry: Obj) -> Result<Option<Rc<Font>>, Fail> {
        let key = match entry {
            Obj::Ref(n, _) => Some(n),
            _ => None,
        };
        if let Some(f) = key.and_then(|k| self.fonts.by_ref.get(&k)) {
            return Ok(Some(f.clone()));
        }
        if let Obj::Dict(d) = &entry {
            if let Some((_, f)) = self.fonts.direct.get(&Rc::as_ptr(d)) {
                return Ok(Some(f.clone()));
            }
        }
        let obj = soft(self.doc.get(&entry))?;
        let Some(dict) = obj.as_dict() else {
            return Ok(None);
        };
        let font = Rc::new(self.read_font(dict)?);
        match (key, &entry) {
            (Some(k), _) => {
                self.fonts.by_ref.insert(k, font.clone());
            }
            (None, Obj::Dict(d)) => {
                self.fonts
                    .direct
                    .insert(Rc::as_ptr(d), (d.clone(), font.clone()));
            }
            _ => {}
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
                    e.set(b"BaseEncoding".to_vec(), soft(self.doc.get(b))?);
                }
                let diff = soft(self.doc.entry(&d, b"Differences"))?;
                e.set(b"Differences".to_vec(), diff);
                encoding = Some(Obj::Dict(Rc::new(e)));
            }
            _ => {}
        }
        let mut widths = None;
        let mut default_width = None;
        if composite {
            let desc = soft(self.doc.entry(dict, b"DescendantFonts"))?;
            let first = desc.as_array().and_then(|a| a.first()).cloned();
            if let Some(first) = first {
                let df = soft(self.doc.get(&first))?;
                if let Some(df) = df.as_dict() {
                    if let Obj::Array(w) = soft(self.doc.entry(df, b"W"))? {
                        self.doc.work.take(w.len() as u64)?;
                        let mut out = Vec::with_capacity(w.len());
                        for x in w.iter() {
                            out.push(soft(self.doc.get(x))?);
                        }
                        widths = Some(Rc::from(out));
                    }
                    default_width = soft(self.doc.entry(df, b"DW"))?.as_f64();
                }
            }
        } else {
            if let Obj::Array(w) = soft(self.doc.entry(dict, b"Widths"))? {
                widths = Some(w);
            }
            let fd = soft(self.doc.entry(dict, b"FontDescriptor"))?;
            if let Some(fd) = fd.as_dict() {
                default_width = soft(self.doc.entry(fd, b"MissingWidth"))?
                    .as_f64()
                    .filter(|w| *w > 0.0);
            }
        }
        Font::new(FontParts {
            dict,
            to_unicode,
            encoding,
            encoding_cmap,
            widths,
            default_width,
            mem: &mut self.doc.objects,
            work: &mut self.doc.work,
        })
    }

    /// Follows `Do` into a form XObject, which starts in the graphics state `gs` of the `Do`.
    /// Each form drawn is charged to the form budget, and each draw decodes the form again,
    /// charged to the decode budget.
    fn form(
        &mut self,
        resources: &Rc<Dict>,
        name: &[u8],
        gs: &GState,
        depth: u64,
    ) -> Result<(), Fail> {
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
        self.forms.take(1)?;
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
        let start = GState {
            ctm: matrix.then(gs.ctm),
            ..gs.clone()
        };
        self.run(&data, &own, Start(start), depth + 1)
    }
}

/// The graphics state a content stream starts in.
pub(crate) struct Start(GState);

impl Start {
    /// A page's: page space, no font selected.
    pub fn page() -> Start {
        Start(GState::new(M::ID))
    }
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
///
/// The image bytes may hold anything, `EI` between white space too, so where they end is taken
/// from the image's own framing when it has one ([`framed_end`]): its declared length (`/L`),
/// the end marker of an ASCII filter, or, unfiltered, its size (rows of its width times its
/// components times their bits). An end found that way counts only when `EI` follows it.
/// Otherwise (another filter, a colour space named in the resources, a size that does not fit)
/// the image ends at the first `EI` between white space that content-stream operators follow
/// ([`operators_follow`]); each candidate's check is charged to the work budget.
fn skip_inline_image(lx: &mut Lexer, work: &mut Budget) -> Result<(), Fail> {
    let mut params: Vec<Obj> = Vec::new();
    loop {
        match lx.token(0, true) {
            Ok(Some(Token::Op(op))) if op == b"ID" => break,
            Ok(Some(Token::Obj(o))) => {
                if params.len() < MAX_OPERANDS {
                    params.push(o);
                }
            }
            Ok(Some(_)) => {}
            Err(e @ Fail::TooLarge(_)) => return Err(e),
            _ => return Ok(()),
        }
    }
    // one white-space byte, then the image bytes
    let d = lx.data;
    let start = (lx.pos + 1).min(d.len());
    if let Some(after) = framed_end(&params, d, start).and_then(|end| ei_at(d, end)) {
        lx.pos = after;
        return Ok(());
    }
    let mut i = start;
    while i + 1 < d.len() {
        if d[i] == b'E'
            && d[i + 1] == b'I'
            && i > 0
            && is_white(d[i - 1])
            && d.get(i + 2).is_none_or(|b| is_white(*b))
        {
            work.take(1)?;
            if operators_follow(d, i + 2, work)? {
                lx.pos = i + 2;
                return Ok(());
            }
        }
        i += 1;
    }
    lx.pos = d.len();
    Ok(())
}

/// Where an inline image's bytes starting at `start` end by its own framing (`None`: it has
/// none the reader knows, or the end it declares lies past the data).
fn framed_end(params: &[Obj], d: &[u8], start: usize) -> Option<usize> {
    let get = |a: &[u8], b: &[u8]| {
        params.chunks_exact(2).find_map(|kv| match &kv[0] {
            Obj::Name(n) if &n[..] == a || &n[..] == b => Some(&kv[1]),
            _ => None,
        })
    };
    let avail = (d.len() - start) as u64;
    // a declared length (PDF 2.0)
    if let Some(l) = get(b"L", b"Length").and_then(Obj::as_int) {
        let l = u64::try_from(l).ok()?;
        return v::pdf_span_ok(start as u64, l, d.len() as u64).then(|| start + l as usize);
    }
    let filter = match get(b"F", b"Filter") {
        Some(Obj::Name(n)) => Some(n.clone()),
        Some(Obj::Array(a)) => match a.first() {
            Some(Obj::Name(n)) => Some(n.clone()),
            Some(_) => return None,
            None => None,
        },
        Some(_) => return None,
        None => None,
    };
    if let Some(f) = filter {
        let rest = &d[start..];
        return match &f[..] {
            b"AHx" | b"ASCIIHexDecode" => {
                rest.iter().position(|&b| b == b'>').map(|k| start + k + 1)
            }
            b"A85" | b"ASCII85Decode" => rest
                .windows(2)
                .position(|w| w == b"~>")
                .map(|k| start + k + 2),
            _ => None,
        };
    }
    // unfiltered: rows of the width's samples, each row padded to a whole byte
    let int = |a: &[u8], b: &[u8]| get(a, b).and_then(Obj::as_int);
    let mask = matches!(get(b"IM", b"ImageMask"), Some(Obj::Bool(true)));
    let (components, bpc): (u64, i64) = if mask {
        (1, 1)
    } else {
        let components = match get(b"CS", b"ColorSpace") {
            Some(Obj::Name(n)) => match &n[..] {
                b"G" | b"DeviceGray" | b"CalGray" | b"I" | b"Indexed" => 1,
                b"RGB" | b"DeviceRGB" | b"CalRGB" => 3,
                b"CMYK" | b"DeviceCMYK" => 4,
                _ => return None,
            },
            Some(Obj::Array(a)) => match a.first() {
                Some(Obj::Name(n)) if &n[..] == b"I" || &n[..] == b"Indexed" => 1,
                _ => return None,
            },
            _ => return None,
        };
        (components, int(b"BPC", b"BitsPerComponent")?)
    };
    let width = int(b"W", b"Width")?;
    let height = u64::try_from(int(b"H", b"Height")?).ok()?;
    if !matches!(bpc, 1 | 2 | 4 | 8 | 16) || !(1..=1 << 24).contains(&width) {
        return None;
    }
    // all three are small, so their product fits easily
    let row = (width as u64 * components * bpc as u64).div_ceil(8);
    v::pdf_count_fits(height, row, avail).then(|| start + v::pdf_row_offset(height, row) as usize)
}

/// Where the operator after `EI` starts, when `EI` (between optional white space before it and
/// white space, a delimiter or the end after it) is at `end`.
fn ei_at(d: &[u8], end: usize) -> Option<usize> {
    let mut j = end;
    while d.get(j).is_some_and(|b| is_white(*b)) {
        j += 1;
    }
    (d.get(j..j + 2) == Some(b"EI")
        && d.get(j + 2)
            .is_none_or(|b| is_white(*b) || super::lex::is_delim(*b)))
    .then_some(j + 2)
}

/// Bytes of operands an end-of-image check may read.
const CHECK_BYTES: u64 = 4096;
/// Operands an end-of-image check may read before its operator.
const CHECK_TOKENS: usize = 16;

/// Whether the bytes at `at` read as content: the end of the data, or operands and then a
/// content-stream operator. The bytes the check reads are charged to the work budget.
fn operators_follow(d: &[u8], at: usize, work: &mut Budget) -> Result<bool, Fail> {
    let mem = Budget::new(
        CHECK_BYTES,
        "an inline image's end check reads",
        "bytes",
        super::INFLATE_FLAG,
    );
    let mut lx = Lexer::new(d, at, mem);
    let mut ok = false;
    for _ in 0..CHECK_TOKENS {
        match lx.token(0, true) {
            Ok(None) => {
                ok = true;
                break;
            }
            Ok(Some(Token::Op(op))) => {
                ok = is_operator(&op);
                break;
            }
            Ok(Some(Token::Obj(_))) => {}
            Err(_) => break,
        }
    }
    work.take((lx.pos - at) as u64)?;
    Ok(ok)
}

/// The operators of content streams (ISO 32000-1, Annex A).
fn is_operator(op: &[u8]) -> bool {
    matches!(
        op,
        b"b" | b"B"
            | b"b*"
            | b"B*"
            | b"BDC"
            | b"BI"
            | b"BMC"
            | b"BT"
            | b"BX"
            | b"c"
            | b"cm"
            | b"CS"
            | b"cs"
            | b"d"
            | b"d0"
            | b"d1"
            | b"Do"
            | b"DP"
            | b"EMC"
            | b"ET"
            | b"EX"
            | b"f"
            | b"F"
            | b"f*"
            | b"G"
            | b"g"
            | b"gs"
            | b"h"
            | b"i"
            | b"j"
            | b"J"
            | b"K"
            | b"k"
            | b"l"
            | b"m"
            | b"M"
            | b"MP"
            | b"n"
            | b"q"
            | b"Q"
            | b"re"
            | b"RG"
            | b"rg"
            | b"ri"
            | b"s"
            | b"S"
            | b"SC"
            | b"sc"
            | b"SCN"
            | b"scn"
            | b"sh"
            | b"T*"
            | b"Tc"
            | b"Td"
            | b"TD"
            | b"Tf"
            | b"TJ"
            | b"Tj"
            | b"TL"
            | b"Tm"
            | b"Tr"
            | b"Ts"
            | b"Tw"
            | b"Tz"
            | b"v"
            | b"w"
            | b"W"
            | b"W*"
            | b"y"
            | b"'"
            | b"\""
    )
}
