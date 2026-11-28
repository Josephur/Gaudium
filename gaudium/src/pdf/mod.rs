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
//! and a form drawing itself all stop at a cap. Encrypted PDFs are refused.

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
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_bytes: DEFAULT_MAX_MB << 20,
            max_pages: DEFAULT_MAX_PAGES,
            max_decoded: DEFAULT_MAX_INFLATE_MB << 20,
            max_forms: DEFAULT_MAX_FORMS,
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

/// A budget of one document: the bytes its streams may still decode to, the memory its parsed
/// objects may still take, the text it may still yield, or the form XObjects it may still draw.
/// Every charge goes through the verified core (`pdf_budget_fits`, `pdf_budget_take`).
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

    /// The same budget with `used` spent (no more than it has spent now).
    pub fn restarted_at(&self, used: u64) -> Self {
        Budget {
            used: used.min(self.used),
            ..*self
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

    /// Charges `n`; refused past the budget.
    pub fn take(&mut self, n: u64) -> Result<(), Fail> {
        if !v::pdf_budget_fits(self.used, n, self.max) {
            return Err(self.over());
        }
        self.used = v::pdf_budget_take(self.used, n);
        Ok(())
    }

    pub fn over(&self) -> Fail {
        Fail::TooLarge(format!(
            "{} more than {} {} ({})",
            self.what, self.max, self.unit, self.flag
        ))
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
        page.run(&data, &p.resources, content::M0::ID, 0)?;
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
    let mut seen = HashSet::new();
    walk(doc, &pages, None, 0, &mut seen, &mut out, max_pages)?;
    Ok(out)
}

fn walk(
    doc: &mut Doc,
    node: &Obj,
    inherited: Option<&Rc<Dict>>,
    depth: u64,
    seen: &mut HashSet<u32>,
    out: &mut Vec<PageRef>,
    max_pages: u64,
) -> Result<(), Fail> {
    if !v::pdf_below(depth, MAX_TREE_DEPTH) {
        return Err(Fail::damaged(format!(
            "a page tree deeper than {MAX_TREE_DEPTH} levels"
        )));
    }
    if let Obj::Ref(n, _) = node {
        if !seen.insert(*n) {
            return Ok(()); // a cycle, or a node listed twice
        }
    }
    let obj = doc.get(node)?;
    let Some(d) = obj.as_dict() else {
        return Ok(());
    };
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
