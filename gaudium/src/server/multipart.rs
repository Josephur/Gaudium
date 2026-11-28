//! Request bodies read as a stream, and `multipart/form-data` (RFC 7578) parts read from one.
//!
//! An upload (`POST /v1/files`) is not held in memory: the request parser leaves its body unread
//! (`HttpRequest::streamed` holds the bytes it had already buffered past the head), and the
//! handler reads it through a [`BodyReader`] (a `Content-Length` body or a chunked one) and a
//! [`Multipart`] reader that hands out one part at a time. Only a part's head and the small form
//! fields are buffered; a file part's bytes go straight to the reader's caller.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Cursor, Read};

/// The largest head of one part (its header lines).
const MAX_PART_HEAD: usize = 16 * 1024;
/// The largest chunk-size line of a chunked body.
const MAX_CHUNK_LINE: usize = 1024;
/// Bytes the reader asks its source for at a time.
const READ_STEP: usize = 64 * 1024;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A request body: `Content-Length` bytes, or a chunked body up to its last chunk.
pub(super) struct BodyReader<R: Read> {
    inner: BufReader<io::Chain<Cursor<Vec<u8>>, R>>,
    mode: Mode,
}

enum Mode {
    Length(u64),
    /// `left` bytes of the current chunk; `crlf`: the CRLF after a chunk is still owed.
    Chunked {
        left: u64,
        crlf: bool,
        done: bool,
    },
}

impl<R: Read> BodyReader<R> {
    /// `prefix`: the body bytes already read past the head; `src`: the rest of the connection.
    /// `None` when the headers give the body no length (neither `Content-Length` nor chunked).
    pub fn new(prefix: Vec<u8>, src: R, headers: &HashMap<String, String>) -> Result<Self, String> {
        let chunked = headers
            .get("transfer-encoding")
            .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
        let mode = if chunked {
            Mode::Chunked {
                left: 0,
                crlf: false,
                done: false,
            }
        } else {
            match headers.get("content-length") {
                Some(v) => Mode::Length(v.trim().parse().map_err(|_| "invalid Content-Length")?),
                None => return Err("an upload needs a Content-Length or a chunked body".into()),
            }
        };
        Ok(BodyReader {
            inner: BufReader::with_capacity(READ_STEP, Cursor::new(prefix).chain(src)),
            mode,
        })
    }

    /// The body's length when the headers declare it (`Content-Length`); asked before reading.
    pub fn declared_len(&self) -> Option<u64> {
        match self.mode {
            Mode::Length(n) => Some(n),
            Mode::Chunked { .. } => None,
        }
    }

    fn line(&mut self) -> io::Result<String> {
        let mut line = Vec::new();
        let n = (&mut self.inner)
            .take(MAX_CHUNK_LINE as u64)
            .read_until(b'\n', &mut line)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        if !line.ends_with(b"\n") {
            return Err(bad("chunk-size line too long"));
        }
        Ok(String::from_utf8_lossy(&line).trim().to_string())
    }
}

impl<R: Read> Read for BodyReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            match self.mode {
                Mode::Length(0) => return Ok(0),
                Mode::Length(left) => {
                    let want = buf.len().min(usize::try_from(left).unwrap_or(usize::MAX));
                    let n = self.inner.read(&mut buf[..want])?;
                    if n == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "the body ended before its Content-Length",
                        ));
                    }
                    self.mode = Mode::Length(left - n as u64);
                    return Ok(n);
                }
                Mode::Chunked { done: true, .. } => return Ok(0),
                Mode::Chunked {
                    left: 0,
                    crlf,
                    done: false,
                } => {
                    if crlf {
                        let l = self.line()?;
                        if !l.is_empty() {
                            return Err(bad("missing CRLF after a chunk"));
                        }
                    }
                    let l = self.line()?;
                    let hex = l.split(';').next().unwrap_or("").trim();
                    let size = u64::from_str_radix(hex, 16)
                        .map_err(|_| bad(format!("invalid chunk size '{hex}'")))?;
                    if size == 0 {
                        // trailer fields up to the empty line
                        while !self.line()?.is_empty() {}
                        self.mode = Mode::Chunked {
                            left: 0,
                            crlf: false,
                            done: true,
                        };
                        return Ok(0);
                    }
                    self.mode = Mode::Chunked {
                        left: size,
                        crlf: true,
                        done: false,
                    };
                }
                Mode::Chunked { left, .. } => {
                    let want = buf.len().min(usize::try_from(left).unwrap_or(usize::MAX));
                    let n = self.inner.read(&mut buf[..want])?;
                    if n == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "the body ended inside a chunk",
                        ));
                    }
                    self.mode = Mode::Chunked {
                        left: left - n as u64,
                        crlf: true,
                        done: false,
                    };
                    return Ok(n);
                }
            }
        }
    }
}

/// The boundary of a `multipart/form-data` content type, or why there is none.
pub(super) fn boundary(content_type: &str) -> Result<String, String> {
    let mut parts = content_type.split(';');
    let kind = parts.next().unwrap_or("").trim().to_ascii_lowercase();
    if kind != "multipart/form-data" {
        return Err(format!(
            "an upload is a multipart/form-data body (got Content-Type '{}')",
            if kind.is_empty() { "none" } else { &kind }
        ));
    }
    for p in parts {
        if let Some((k, v)) = p.split_once('=') {
            if k.trim().eq_ignore_ascii_case("boundary") {
                let b = v.trim().trim_matches('"');
                if b.is_empty() || b.len() > 70 {
                    return Err("the multipart boundary must be 1 to 70 characters".into());
                }
                return Ok(b.to_string());
            }
        }
    }
    Err("the multipart/form-data Content-Type has no boundary".into())
}

/// The head of one part: its form field name, file name and content type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PartHead {
    pub name: String,
    pub filename: Option<String>,
    pub content_type: Option<String>,
}

#[derive(PartialEq, Eq)]
enum At {
    /// Before the first delimiter (the preamble is skipped).
    Preamble,
    /// Inside a part's content.
    Part,
    /// At a delimiter: the current part (or the preamble) has ended.
    Delimiter,
    /// After the closing delimiter.
    Done,
}

/// The parts of a `multipart/form-data` body, one at a time.
pub(super) struct Multipart<R: Read> {
    src: R,
    /// `CRLF--boundary`: what ends a part.
    delim: Vec<u8>,
    buf: Vec<u8>,
    start: usize,
    eof: bool,
    at: At,
}

impl<R: Read> Multipart<R> {
    pub fn new(src: R, boundary: &str) -> Self {
        let mut delim = b"\r\n--".to_vec();
        delim.extend_from_slice(boundary.as_bytes());
        Multipart {
            src,
            delim,
            // a body opens with the delimiter itself, without the CRLF that ends a part
            buf: b"\r\n".to_vec(),
            start: 0,
            eof: false,
            at: At::Preamble,
        }
    }

    fn avail(&self) -> &[u8] {
        &self.buf[self.start..]
    }

    /// Reads more of the source into the buffer; `false` at its end.
    fn fill(&mut self) -> io::Result<bool> {
        if self.eof {
            return Ok(false);
        }
        if self.start > READ_STEP {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        let old = self.buf.len();
        self.buf.resize(old + READ_STEP, 0);
        let n = loop {
            match self.src.read(&mut self.buf[old..]) {
                Ok(n) => break n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.buf.truncate(old);
                    return Err(e);
                }
            }
        };
        self.buf.truncate(old + n);
        if n == 0 {
            self.eof = true;
        }
        Ok(n > 0)
    }

    /// Reads the current part's content; 0 at its end.
    fn read_content(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if !matches!(self.at, At::Part | At::Preamble) || out.is_empty() {
            return Ok(0);
        }
        loop {
            let avail = self.avail();
            if let Some(i) = find(avail, &self.delim) {
                let n = i.min(out.len());
                out[..n].copy_from_slice(&avail[..n]);
                self.start += n;
                if i == 0 {
                    self.at = At::Delimiter;
                }
                return Ok(n);
            }
            // keep what could be the start of a delimiter that the next read completes
            let safe = avail.len().saturating_sub(self.delim.len() - 1);
            if safe > 0 {
                let n = safe.min(out.len());
                out[..n].copy_from_slice(&avail[..n]);
                self.start += n;
                return Ok(n);
            }
            if !self.fill()? {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "the multipart body ended inside a part",
                ));
            }
        }
    }

    /// Makes sure `n` bytes are buffered; `false` if the source ends first.
    fn need(&mut self, n: usize) -> io::Result<bool> {
        while self.avail().len() < n {
            if !self.fill()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The next part's head (the rest of the current part is skipped); `None` after the last.
    pub fn next_part(&mut self) -> io::Result<Option<PartHead>> {
        let mut sink = [0u8; 8192];
        while matches!(self.at, At::Part | At::Preamble) {
            if self.read_content(&mut sink)? == 0 && self.at != At::Delimiter {
                return Err(bad("the multipart body ended inside a part"));
            }
        }
        if self.at == At::Done {
            return Ok(None);
        }
        // at the delimiter: `--` closes the body, else the delimiter line ends in CRLF
        let dl = self.delim.len();
        if !self.need(dl + 2)? {
            return Err(bad("the multipart body ended at a delimiter"));
        }
        self.start += dl;
        if self.avail().starts_with(b"--") {
            self.at = At::Done;
            return Ok(None);
        }
        // transport padding (spaces, tabs) may follow a delimiter
        loop {
            if !self.need(2)? {
                return Err(bad("the multipart body ended at a delimiter"));
            }
            match self.avail()[0] {
                b' ' | b'\t' => self.start += 1,
                _ => break,
            }
        }
        if !self.avail().starts_with(b"\r\n") {
            return Err(bad("a multipart delimiter must end its line"));
        }
        self.start += 2;
        let head_len = loop {
            if let Some(i) = find(self.avail(), b"\r\n\r\n") {
                break i;
            }
            if self.avail().starts_with(b"\r\n") {
                // a part with no header lines at all
                break 0;
            }
            if self.avail().len() > MAX_PART_HEAD {
                return Err(bad("a multipart part head is too long"));
            }
            if !self.fill()? {
                return Err(bad("the multipart body ended inside a part head"));
            }
        };
        let head = String::from_utf8_lossy(&self.avail()[..head_len]).into_owned();
        self.start += if head_len == 0 { 2 } else { head_len + 4 };
        self.at = At::Part;
        parse_head(&head).map(Some)
    }

    /// Reads the current part's content (0 at its end).
    pub fn part(&mut self) -> PartReader<'_, R> {
        PartReader(self)
    }
}

/// The content of the current part, as a reader.
pub(super) struct PartReader<'a, R: Read>(&'a mut Multipart<R>);

impl<R: Read> Read for PartReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read_content(buf)
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    let first = needle[0];
    let last_start = hay.len() - needle.len();
    let mut i = 0;
    while i <= last_start {
        match hay[i..=last_start].iter().position(|&b| b == first) {
            None => return None,
            Some(p) => {
                i += p;
                if &hay[i..i + needle.len()] == needle {
                    return Some(i);
                }
                i += 1;
            }
        }
    }
    None
}

/// A `Content-Disposition` parameter value: quoted (with `\"` escapes) or a token.
fn param<'a>(disp: &'a str, key: &str) -> Option<String> {
    let mut rest = disp;
    while let Some(semi) = rest.find(';') {
        rest = &rest[semi + 1..];
        let t = rest.trim_start();
        let Some(eq) = t.find('=') else { continue };
        let k = t[..eq].trim();
        let v = &t[eq + 1..];
        if !k.eq_ignore_ascii_case(key) {
            continue;
        }
        let v = v.trim_start();
        if let Some(q) = v.strip_prefix('"') {
            let mut out = String::new();
            let mut chars = q.chars();
            while let Some(c) = chars.next() {
                match c {
                    '\\' => out.extend(chars.next()),
                    '"' => break,
                    c => out.push(c),
                }
            }
            return Some(out);
        }
        let end = v.find(';').unwrap_or(v.len());
        return Some(v[..end].trim().to_string());
    }
    None
}

/// `filename*=UTF-8''<percent-encoded>` (RFC 5987), the form some clients send for non-ASCII names.
fn ext_filename(disp: &str) -> Option<String> {
    let v = param(disp, "filename*")?;
    let (charset, rest) = v.split_once('\'')?;
    let (_, enc) = rest.split_once('\'')?;
    if !charset.eq_ignore_ascii_case("utf-8") {
        return None;
    }
    let b = enc.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let h = std::str::from_utf8(&b[i + 1..i + 3]).ok()?;
            out.push(u8::from_str_radix(h, 16).ok()?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn parse_head(head: &str) -> io::Result<PartHead> {
    let mut disp: Option<&str> = None;
    let mut content_type = None;
    for line in head.split("\r\n") {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let k = k.trim();
        if k.eq_ignore_ascii_case("content-disposition") {
            disp = Some(v.trim());
        } else if k.eq_ignore_ascii_case("content-type") {
            content_type = Some(v.trim().to_string());
        }
    }
    let disp = disp.ok_or_else(|| bad("a multipart part has no Content-Disposition"))?;
    if !disp
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .eq_ignore_ascii_case("form-data")
    {
        return Err(bad("a multipart part must be form-data"));
    }
    let name = param(disp, "name").ok_or_else(|| bad("a multipart part has no name"))?;
    let filename = ext_filename(disp).or_else(|| param(disp, "filename"));
    Ok(PartHead {
        name,
        filename,
        content_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader that hands out at most `step` bytes per call (a slow socket).
    struct Trickle<'a>(&'a [u8], usize);

    impl Read for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.0.len().min(self.1).min(buf.len());
            buf[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }

    fn body() -> Vec<u8> {
        let mut b = b"preamble\r\n--XyZ\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--XyZ  \r\nContent-Disposition: form-data; name=\"file\"; filename=\"a\\\"b.txt\"\r\nContent-Type: text/plain\r\n\r\n".to_vec();
        // content that holds a near-delimiter, which must stay content
        b.extend_from_slice(b"line one\r\n--XyY\r\n--Xy");
        b.extend(std::iter::repeat(b'z').take(200_000));
        b.extend_from_slice(b"\r\n--XyZ--\r\nepilogue");
        b
    }

    fn parts(src: impl Read) -> Vec<(PartHead, Vec<u8>)> {
        let mut m = Multipart::new(src, "XyZ");
        let mut out = Vec::new();
        while let Some(h) = m.next_part().unwrap() {
            let mut v = Vec::new();
            m.part().read_to_end(&mut v).unwrap();
            out.push((h, v));
        }
        out
    }

    #[test]
    fn parts_come_out_whole_whatever_the_read_sizes() {
        let b = body();
        for step in [1, 3, 7, 64, 65_536, usize::MAX] {
            let p = parts(Trickle(&b, step));
            assert_eq!(p.len(), 2, "step {step}");
            assert_eq!(p[0].0.name, "purpose");
            assert_eq!(p[0].1, b"batch");
            assert_eq!(p[1].0.name, "file");
            assert_eq!(p[1].0.filename.as_deref(), Some("a\"b.txt"));
            assert_eq!(p[1].0.content_type.as_deref(), Some("text/plain"));
            assert_eq!(p[1].1.len(), 8 + 2 + 5 + 2 + 4 + 200_000, "step {step}");
            assert!(p[1].1.starts_with(b"line one\r\n--XyY\r\n--Xyzz"));
        }
    }

    #[test]
    fn a_skipped_part_is_drained_and_truncated_bodies_are_refused() {
        let b = body();
        let mut m = Multipart::new(&b[..], "XyZ");
        assert_eq!(m.next_part().unwrap().unwrap().name, "purpose");
        assert_eq!(m.next_part().unwrap().unwrap().name, "file");
        assert!(m.next_part().unwrap().is_none());
        let cut = &b[..b.len() - 30];
        let mut m = Multipart::new(cut, "XyZ");
        m.next_part().unwrap();
        m.next_part().unwrap();
        assert!(m.next_part().is_err());
    }

    #[test]
    fn content_types_and_boundaries() {
        assert_eq!(
            boundary("multipart/form-data; boundary=abc").unwrap(),
            "abc"
        );
        assert_eq!(
            boundary("Multipart/Form-Data; charset=x; boundary=\"a b\"").unwrap(),
            "a b"
        );
        assert!(boundary("application/json").is_err());
        assert!(boundary("multipart/form-data").is_err());
        assert_eq!(
            ext_filename("form-data; name=\"file\"; filename*=UTF-8''%C3%A9t%C3%A9.txt").as_deref(),
            Some("été.txt")
        );
    }

    #[test]
    fn bodies_by_length_and_in_chunks() {
        let mut h = HashMap::new();
        h.insert("content-length".to_string(), "11".to_string());
        let mut r = BodyReader::new(b"hello".to_vec(), &b" worldEXTRA"[..], &h).unwrap();
        let mut s = String::new();
        r.read_to_string(&mut s).unwrap();
        assert_eq!(s, "hello world");

        let mut h = HashMap::new();
        h.insert("transfer-encoding".to_string(), "chunked".to_string());
        let raw = b"5;x=y\r\nhello\r\n6\r\n world\r\n0\r\nTrailer: 1\r\n\r\nNEXT";
        let mut r = BodyReader::new(raw[..4].to_vec(), Trickle(&raw[4..], 3), &h).unwrap();
        let mut s = String::new();
        r.read_to_string(&mut s).unwrap();
        assert_eq!(s, "hello world");

        let mut h = HashMap::new();
        h.insert("content-length".to_string(), "20".to_string());
        let mut r = BodyReader::new(Vec::new(), &b"short"[..], &h).unwrap();
        assert!(r.read_to_end(&mut Vec::new()).is_err());
        assert!(BodyReader::new(Vec::new(), &b""[..], &HashMap::new()).is_err());
    }
}
