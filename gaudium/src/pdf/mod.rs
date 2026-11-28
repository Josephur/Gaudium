//! PDF text extraction, written from the PDF specification (ISO 32000-1).
//!
//! A PDF sent with a request (a Messages `document` block, a chat `file` part) is reduced to the
//! text of its pages: the file structure (xref tables and streams, object streams, damaged
//! files rebuilt by a scan), the page tree, content streams through their filters, and the
//! text operators with each font's encoding and ToUnicode CMap. The result is the text layer,
//! page by page; a PDF whose pages hold no text (scanned pages) is refused with a clear error.
//!
//! Rendering pages to images for a vision model is not done here: it needs a font rasteriser.
//! The seam for it is [`extract_text`]'s caller, which could hand pages to the vision path
//! instead of (or beside) their text.
//!
//! A PDF is client data. Every offset, length, count, depth and byte budget taken from the file
//! is checked through the verified core (`verified/pdf_bounds.bend`) before it is used, and
//! every limit is refused with its numbers. What one document may make the server spend is
//! capped by budgets charged through the verified core: the bytes its streams decode to (every
//! read of a stream, filtered or not, and the work of a PNG predictor), the bytes its object
//! parses read and the memory their objects take (objects may overlap), the text it yields, and
//! the form XObjects it draws. So a zip bomb, a stream listed many times, overlapping objects
//! and a form drawing itself all stop at a cap.
//!
//! Beside those, one budget covers the reader's work on the file's structure, whatever kind of
//! work it is (`--pdf-max-work`): every step that touches something taken from the file costs at
//! least one unit (a page-tree node visited, an object resolved, a stream read, a filter stage,
//! an element of a /Contents, /Filter, /W or /Differences array, an entry a width list or a
//! CMap expands to, a content-stream operator). Structure the file shares many times over (one
//! array listed by many nodes, one stream read many times, one width list repeated) is charged
//! for every use, so a path that misses a specific budget is still bounded by this one.
//!
//! Behind every budget, the heap the reader actually holds is counted by the process allocator
//! ([`crate::heap_meter`], on only on the reading thread and only while a PDF is read) and
//! checked at every charge of every budget (`--pdf-max-heap-mb`): memory a path keeps without
//! charging it to a budget is still refused at the next charge.
//!
//! A request may carry many documents, so what they keep together is capped as well, through
//! the same verified budgets ([`Request`]): the number of documents (`--pdf-max-docs`), the
//! bytes of text they keep (the text cap of one document, `--pdf-max-inflate-mb`), and the heap
//! the expansion holds (`--pdf-max-heap-mb`), counted across all of them: each read checks its
//! own heap and, at every charge, the heap the request held before it plus its own.
//!
//! Encrypted PDFs are refused.

mod content;
mod doc;
mod filter;
mod font;
mod lex;
#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::rc::Rc;

use crate::verified as v;
use doc::Doc;
use lex::{Dict, Obj};

/// `--pdf-max-mb` default: largest PDF read, in MiB.
pub const DEFAULT_MAX_MB: u64 = 32;
/// `--pdf-max-pages` default: most pages read from one PDF.
pub const DEFAULT_MAX_PAGES: u64 = 500;
/// `--pdf-max-inflate-mb` default: most bytes one PDF's streams may decode to, in MiB.
pub const DEFAULT_MAX_INFLATE_MB: u64 = 256;
/// `--pdf-max-forms` default: most form XObjects one PDF may draw, all pages together.
pub const DEFAULT_MAX_FORMS: u64 = 100_000;
/// `--pdf-max-work` default: most steps of work the reader may take over one PDF's structure.
pub const DEFAULT_MAX_WORK: u64 = 20_000_000;
/// `--pdf-max-heap-mb` default: most heap the reader may hold while it reads one PDF, in MiB
/// (and the documents of one request together).
pub const DEFAULT_MAX_HEAP_MB: u64 = 1024;
/// `--pdf-max-docs` default: most documents (PDF or text files) one request may carry.
pub const DEFAULT_MAX_DOCS: u64 = 100;

/// Deepest nesting of arrays and dictionaries.
pub(crate) const MAX_NEST: u64 = 64;
/// Longest chain of references followed while reading one object.
pub(crate) const MAX_REF_DEPTH: u64 = 32;
/// Most cross-reference sections (`/Prev` chain) read.
pub(crate) const MAX_XREF_SECTIONS: u64 = 64;
/// Deepest page tree.
pub(crate) const MAX_TREE_DEPTH: u64 = 64;
/// Deepest nesting of form XObjects.
pub(crate) const MAX_FORM_DEPTH: u64 = 12;
/// Deepest graphics-state stack (`q`).
pub(crate) const MAX_GSTATE_DEPTH: u64 = 64;

/// What one PDF may cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Bytes of the file.
    pub max_bytes: u64,
    /// Pages.
    pub max_pages: u64,
    /// Bytes all its streams decode to together, counting every read of a stream (and,
    /// separately, the bytes its object parses read and their objects take, and the bytes of
    /// text it yields).
    pub max_decoded: u64,
    /// Form XObjects drawn, all pages together (a form drawn again counts again).
    pub max_forms: u64,
    /// Steps of work over the file's structure (see the module documentation).
    pub max_work: u64,
    /// Bytes of heap the reader may hold while it reads the PDF, as the allocator counts them
    /// (and the expansion of all the documents of a request, [`Request`]).
    pub max_heap: u64,
    /// Documents one request may carry ([`Request`]).
    pub max_docs: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_bytes: DEFAULT_MAX_MB << 20,
            max_pages: DEFAULT_MAX_PAGES,
            max_decoded: DEFAULT_MAX_INFLATE_MB << 20,
            max_forms: DEFAULT_MAX_FORMS,
            max_work: DEFAULT_MAX_WORK,
            max_heap: DEFAULT_MAX_HEAP_MB << 20,
            max_docs: DEFAULT_MAX_DOCS,
        }
    }
}

/// Why a PDF yields no text.
#[derive(Debug, Clone, PartialEq)]
pub enum Fail {
    NotPdf,
    Encrypted,
    /// A limit is exceeded (what, with its numbers).
    TooLarge(String),
    /// A limit of the request's documents together is exceeded ([`Request`]).
    RequestTooLarge(String),
    Damaged(String),
    /// A feature the reader does not have (a filter, a predictor).
    Unsupported(String),
    /// The pages hold no text.
    NoText {
        pages: usize,
    },
}

impl Fail {
    pub(crate) fn damaged(m: impl Into<String>) -> Fail {
        Fail::Damaged(m.into())
    }
}

impl std::fmt::Display for Fail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fail::NotPdf => write!(f, "not a PDF file (no %PDF- header)"),
            Fail::Encrypted => write!(
                f,
                "the PDF is encrypted, and this server reads only unencrypted PDFs; remove its password or encryption and send it again"
            ),
            Fail::TooLarge(m) => write!(f, "the PDF is too large: {m}"),
            Fail::RequestTooLarge(m) => {
                write!(f, "the request's documents are too large together: {m}")
            }
            Fail::Damaged(m) => write!(f, "the PDF is damaged or not supported: {m}"),
            Fail::Unsupported(m) => write!(f, "the PDF uses {m}, which this server cannot read"),
            Fail::NoText { pages } => write!(
                f,
                "the PDF's text layer is empty ({pages} page{} without extractable text): scanned or image-only pages need a page renderer, which this server does not have; send the text, or the pages as images",
                if *pages == 1 { "" } else { "s" }
            ),
        }
    }
}

/// The flag that sets the byte budgets of a document.
pub(crate) const INFLATE_FLAG: &str = "--pdf-max-inflate-mb";
/// The flag that sets the work budget of a document.
pub(crate) const WORK_FLAG: &str = "--pdf-max-work";
/// The flag that sets the heap the reader may hold.
pub(crate) const HEAP_FLAG: &str = "--pdf-max-heap-mb";

/// The flag that sets the documents a request may carry.
pub(crate) const DOCS_FLAG: &str = "--pdf-max-docs";

/// Whether the heap this read holds (as the allocator counts it) is within its cap, and, when
/// the read runs inside a request's expansion ([`Request`]), whether what the request held
/// before it plus what it holds is within the request's cap. Every charge of every budget
/// checks it first and is refused when it is not, so memory a path keeps without charging it
/// is refused at the next charge. Outside a read (the meter off) it holds.
#[inline]
pub(crate) fn heap_check() -> Result<(), Fail> {
    let Some(r) = crate::heap_meter::reading() else {
        return Ok(());
    };
    if !v::pdf_within(r.live, r.cap) {
        return Err(heap_over(r.cap));
    }
    match r.outer {
        Some(o) if !v::pdf_budget_fits(o.held, r.live, o.cap) => Err(request_heap_over(o.cap)),
        _ => Ok(()),
    }
}

/// The refusal of a read that holds more heap than it may.
#[cold]
#[inline(never)]
fn heap_over(cap: u64) -> Fail {
    Fail::TooLarge(format!(
        "reading it holds more than {cap} bytes of memory ({HEAP_FLAG})"
    ))
}

/// The refusal of a read that, with the request's documents before it, holds more heap than
/// the request may.
#[cold]
#[inline(never)]
fn request_heap_over(cap: u64) -> Fail {
    Fail::TooLarge(format!(
        "reading it, with the request's documents before it, holds more than {cap} bytes of memory ({HEAP_FLAG})"
    ))
}

/// A budget of one document: the bytes its streams may still decode to, the memory its parsed
/// objects may still take, the text it may still yield, the form XObjects it may still draw, or
/// the steps of work it may still cost. Every charge goes through the verified core
/// (`pdf_budget_charge`: counted when it fits, refused with the spend unchanged when not).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Budget {
    used: u64,
    max: u64,
    what: &'static str,
    unit: &'static str,
    flag: &'static str,
}

impl Budget {
    pub fn new(max: u64, what: &'static str, unit: &'static str, flag: &'static str) -> Self {
        Budget {
            used: 0,
            max,
            what,
            unit,
            flag,
        }
    }

    /// What one more charge may take.
    pub fn room(&self) -> u64 {
        v::pdf_budget_room(self.used, self.max)
    }

    /// What is spent.
    pub fn used(&self) -> u64 {
        self.used
    }

    /// Charges `n`; refused past the budget (the spend then stays as it was), and refused,
    /// whatever `n`, once the read holds more heap than it may ([`heap_check`]).
    pub fn take(&mut self, n: u64) -> Result<(), Fail> {
        heap_check()?;
        let fits = v::pdf_budget_fits(self.used, n, self.max);
        self.used = v::pdf_budget_charge(self.used, n, self.max);
        if fits {
            Ok(())
        } else {
            Err(self.over())
        }
    }

    #[cold]
    #[inline(never)]
    pub fn over(&self) -> Fail {
        Fail::TooLarge(format!(
            "{} more than {} {} ({})",
            self.what, self.max, self.unit, self.flag
        ))
    }
}

/// What all the documents of one request may cost together: the documents it carries, the
/// bytes of text they keep, and the heap its expansion holds, counted by the allocator from
/// [`Request::new`] until it is dropped (on the thread that expands the request). Every
/// document read while it lives checks, at every charge, the heap the request held before the
/// read plus its own ([`heap_check`]); every document is charged to it before it is read
/// ([`Request::admit`]) and its text before it is kept ([`Request::keep`]). All charges go
/// through the verified budget (`pdf_budget_charge`).
pub struct Request {
    docs: Budget,
    text: Budget,
    _heap: crate::heap_meter::Metered,
}

impl Request {
    /// The budgets of a request under `limits`; the heap is counted from here.
    pub fn new(limits: &Limits) -> Request {
        Request {
            docs: Budget::new(limits.max_docs, "there are", "documents", DOCS_FLAG),
            text: Budget::new(limits.max_decoded, "their text is", "bytes", INFLATE_FLAG),
            _heap: crate::heap_meter::Metered::start(limits.max_heap),
        }
    }

    /// Charges one more document, before it is read: refused past the request's documents, or
    /// once the request holds more heap than it may.
    pub fn admit(&mut self) -> Result<(), Fail> {
        self.heap()?;
        self.docs.take(1).map_err(request_over)
    }

    /// Charges `n` bytes of a document's text, before it is kept: refused past the request's
    /// text, or once the request holds more heap than it may.
    pub fn keep(&mut self, n: u64) -> Result<(), Fail> {
        self.heap()?;
        self.text.take(n).map_err(request_over)
    }

    fn heap(&self) -> Result<(), Fail> {
        match crate::heap_meter::reading() {
            Some(r) if !v::pdf_within(r.live, r.cap) => Err(Fail::RequestTooLarge(format!(
                "they hold more than {} bytes of memory ({HEAP_FLAG})",
                r.cap
            ))),
            _ => Ok(()),
        }
    }
}

/// A request budget's refusal, worded for the request rather than one PDF.
fn request_over(e: Fail) -> Fail {
    match e {
        Fail::TooLarge(m) => Fail::RequestTooLarge(m),
        e => e,
    }
}

/// The text of a PDF, page by page.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    pub pages: Vec<String>,
}

impl Extracted {
    /// The pages' text as the model reads it, each page under a `--- Page N ---` line.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for (i, p) in self.pages.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&format!("--- Page {} ---\n", i + 1));
            let t = p.trim();
            out.push_str(t);
            if !t.is_empty() {
                out.push('\n');
            }
        }
        out
    }
}

/// Whether `bytes` look like a PDF (its header near the start).
pub fn is_pdf(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(1024)];
    head.windows(5).any(|w| w == b"%PDF-")
}

/// The text layer of a PDF.
pub fn extract_text(bytes: &[u8], limits: &Limits) -> Result<Extracted, Fail> {
    if !v::pdf_within(bytes.len() as u64, limits.max_bytes) {
        return Err(Fail::TooLarge(format!(
            "{} bytes, more than the limit of {} bytes (--pdf-max-mb)",
            bytes.len(),
            limits.max_bytes
        )));
    }
    // the heap this read holds is counted from here, and checked at every charge
    let _heap = crate::heap_meter::Metered::start(limits.max_heap);
    let mut doc = Doc::open(
        bytes,
        Budget::new(
            limits.max_decoded,
            "its streams decode to",
            "bytes",
            INFLATE_FLAG,
        ),
        Budget::new(
            limits.max_decoded,
            "its objects take",
            "bytes",
            INFLATE_FLAG,
        ),
        Budget::new(limits.max_work, "it costs", "steps of work", WORK_FLAG),
    )?;
    let pages = page_list(&mut doc, limits.max_pages)?;
    let mut fonts = content::FontCache::default();
    let mut text_budget = Budget::new(limits.max_decoded, "its text is", "bytes", INFLATE_FLAG);
    let mut forms = Budget::new(
        limits.max_forms,
        "it draws",
        "form XObjects",
        "--pdf-max-forms",
    );
    let mut out = Vec::with_capacity(pages.len());
    for p in &pages {
        // the content streams of a page are one stream split at token boundaries
        let data = contents(&mut doc, &p.contents)?;
        let mut page = content::Page::new(&mut doc, &mut fonts, &mut text_budget, &mut forms);
        page.run(&data, &p.resources, content::Start::page(), 0)?;
        out.push(page.out);
    }
    if out.iter().all(|t| t.trim().is_empty()) {
        return Err(Fail::NoText { pages: out.len() });
    }
    Ok(Extracted { pages: out })
}

struct PageRef {
    contents: Obj,
    resources: Rc<Dict>,
}

/// The pages in order, with their inherited resources.
fn page_list(doc: &mut Doc, max_pages: u64) -> Result<Vec<PageRef>, Fail> {
    let trailer = doc.trailer.clone();
    let root = doc.entry(&trailer, b"Root")?;
    let Some(catalog) = root.as_dict() else {
        return Err(Fail::damaged("no document catalog"));
    };
    let pages = catalog.get(b"Pages").cloned().unwrap_or(Obj::Null);
    let mut out = Vec::new();
    let mut seen = Seen::default();
    walk(doc, &pages, None, 0, &mut seen, &mut out, max_pages)?;
    Ok(out)
}

/// The page-tree nodes already visited, by their shared allocation (a node reached again, by
/// a reference or through a Kids array listed twice, is read once). Each node is kept here, so
/// its address is not reused while the tree is walked.
#[derive(Default)]
struct Seen {
    at: HashSet<*const Dict>,
    keep: Vec<Obj>,
}

fn walk(
    doc: &mut Doc,
    node: &Obj,
    inherited: Option<&Rc<Dict>>,
    depth: u64,
    seen: &mut Seen,
    out: &mut Vec<PageRef>,
    max_pages: u64,
) -> Result<(), Fail> {
    // every node visited costs a step, a node seen before too
    doc.work.take(1)?;
    if !v::pdf_below(depth, MAX_TREE_DEPTH) {
        return Err(Fail::damaged(format!(
            "a page tree deeper than {MAX_TREE_DEPTH} levels"
        )));
    }
    let obj = doc.get(node)?;
    let Some(d) = obj.as_dict() else {
        return Ok(());
    };
    if !seen.at.insert(d as *const Dict) {
        return Ok(()); // a cycle, or a node listed twice
    }
    seen.keep.push(obj.clone());
    let resources = match doc.entry(d, b"Resources")? {
        Obj::Dict(r) => Some(r),
        _ => inherited.cloned(),
    };
    match doc.entry(d, b"Kids")? {
        Obj::Array(kids) => {
            for k in kids.iter() {
                walk(doc, k, resources.as_ref(), depth + 1, seen, out, max_pages)?;
            }
        }
        _ => {
            let is_page = d.get(b"Type").and_then(Obj::as_name) == Some(b"Page")
                || d.get(b"Contents").is_some();
            if is_page {
                if !v::pdf_below(out.len() as u64, max_pages) {
                    return Err(Fail::TooLarge(format!(
                        "it has more than {max_pages} pages (--pdf-max-pages)"
                    )));
                }
                out.push(PageRef {
                    contents: d.get(b"Contents").cloned().unwrap_or(Obj::Null),
                    resources: resources.unwrap_or_default(),
                });
            }
        }
    }
    Ok(())
}

/// The content of a page: its content streams decoded (each read charged to the decode budget)
/// and joined into one, with a white-space byte between parts. A stream the reader cannot
/// decode is left out; a broken limit is an error.
fn contents(doc: &mut Doc, c: &Obj) -> Result<Vec<u8>, Fail> {
    let parts: Rc<[Obj]> = match doc.get(c)? {
        Obj::Array(a) => a,
        o => Rc::from(vec![o]),
    };
    let mut out = Vec::new();
    for p in parts.iter() {
        doc.work.take(1)?;
        if let Obj::Stream(s) = doc.get(p)? {
            match doc.decode(&s, 0) {
                Ok(d) => {
                    out.extend_from_slice(&d);
                    out.push(b'\n');
                }
                Err(e @ Fail::TooLarge(_)) => return Err(e),
                Err(_) => {}
            }
        }
    }
    Ok(out)
}
