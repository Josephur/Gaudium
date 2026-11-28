//! Tests of the PDF text reader on small PDFs written here (no third-party files), and of its
//! refusals on hostile ones: every truncation and many corruptions of the test files must end
//! in an `Err` or in text, never in a panic.

use super::*;

/// A tiny PDF writer: objects numbered from 1, then an xref table or an xref stream.
struct Pdf {
    objs: Vec<Vec<u8>>,
}

fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut o = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
    o.extend_from_slice(data);
    o.extend_from_slice(b"\nendstream");
    o
}

fn adler32(d: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in d {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    b << 16 | a
}

/// A zlib stream of stored (uncompressed) DEFLATE blocks: valid FlateDecode data.
fn zlib_stored(d: &[u8]) -> Vec<u8> {
    let mut o = vec![0x78, 0x01];
    let chunks: Vec<&[u8]> = if d.is_empty() {
        vec![&[][..]]
    } else {
        d.chunks(65535).collect()
    };
    for (i, c) in chunks.iter().enumerate() {
        o.push(u8::from(i + 1 == chunks.len()));
        let n = c.len() as u16;
        o.extend_from_slice(&n.to_le_bytes());
        o.extend_from_slice(&(!n).to_le_bytes());
        o.extend_from_slice(c);
    }
    o.extend_from_slice(&adler32(d).to_be_bytes());
    o
}

/// A zlib stream that expands to `258 * repeats + 1` bytes of `a` (fixed Huffman codes).
fn zlib_bomb(repeats: usize) -> Vec<u8> {
    zlib_run(b'a', repeats)
}

/// A zlib stream that expands to `258 * repeats + 1` bytes of `byte` (below 144: fixed
/// Huffman codes).
fn zlib_run(byte: u8, repeats: usize) -> Vec<u8> {
    assert!(byte < 144);
    struct Bits {
        out: Vec<u8>,
        acc: u64,
        n: u32,
    }
    impl Bits {
        fn put(&mut self, v: u64, n: u32) {
            self.acc |= v << self.n;
            self.n += n;
            while self.n >= 8 {
                self.out.push(self.acc as u8);
                self.acc >>= 8;
                self.n -= 8;
            }
        }
        fn code(&mut self, code: u64, n: u32) {
            let mut r = 0u64;
            for i in 0..n {
                r |= ((code >> i) & 1) << (n - 1 - i);
            }
            self.put(r, n);
        }
    }
    let mut b = Bits {
        out: vec![0x78, 0x01],
        acc: 0,
        n: 0,
    };
    b.put(1, 1); // final block
    b.put(1, 2); // fixed Huffman codes
    b.code(0x30 + u64::from(byte), 8); // the literal
    for _ in 0..repeats {
        b.code(0xC0 + 5, 8); // length 258
        b.put(0, 5); // distance 1
    }
    b.code(0, 7); // end of block
    b.put(0, 7);
    // the checksum of what it expands to
    let (mut a, mut c) = (1u32, 0u32);
    for _ in 0..258 * repeats + 1 {
        a = (a + u32::from(byte)) % 65521;
        c = (c + a) % 65521;
    }
    b.out.extend_from_slice(&(c << 16 | a).to_be_bytes());
    b.out
}

impl Pdf {
    fn new() -> Self {
        Pdf { objs: Vec::new() }
    }

    fn add(&mut self, body: impl AsRef<[u8]>) -> u32 {
        self.objs.push(body.as_ref().to_vec());
        self.objs.len() as u32
    }

    fn reserve(&mut self) -> u32 {
        self.add(b"null")
    }

    fn set(&mut self, num: u32, body: impl AsRef<[u8]>) {
        self.objs[num as usize - 1] = body.as_ref().to_vec();
    }

    /// The file with a classic xref table.
    fn table(&self, root: u32, trailer_extra: &str) -> Vec<u8> {
        let mut out = b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n".to_vec();
        let mut offs = Vec::new();
        for (i, o) in self.objs.iter().enumerate() {
            offs.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(o);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f\r\n", offs.len() + 1).as_bytes(),
        );
        for o in &offs {
            out.extend_from_slice(format!("{o:010} 00000 n\r\n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root {root} 0 R {trailer_extra} >>\nstartxref\n{xref}\n%%EOF\n",
                offs.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    /// The file with an xref stream (PNG Up predictor) and the objects in `packed` inside one
    /// object stream.
    fn xref_stream(&self, root: u32, packed: &[u32]) -> Vec<u8> {
        let mut out = b"%PDF-1.7\n".to_vec();
        let n = self.objs.len() as u32;
        let stm_num = n + 1;
        let xref_num = n + 2;
        // the object stream
        let mut header = String::new();
        let mut body = Vec::new();
        for &p in packed {
            header.push_str(&format!("{p} {} ", body.len()));
            body.extend_from_slice(&self.objs[p as usize - 1]);
            body.push(b' ');
        }
        let mut stm_data = header.clone().into_bytes();
        stm_data.extend_from_slice(&body);
        let mut entries: Vec<(u8, u32, u16)> = vec![(0, 0, 65535)];
        for i in 1..=n {
            if let Some(k) = packed.iter().position(|&p| p == i) {
                entries.push((2, stm_num, k as u16));
                continue;
            }
            entries.push((1, out.len() as u32, 0));
            out.extend_from_slice(format!("{i} 0 obj\n").as_bytes());
            out.extend_from_slice(&self.objs[i as usize - 1]);
            out.extend_from_slice(b"\nendobj\n");
        }
        entries.push((1, out.len() as u32, 0));
        out.extend_from_slice(format!("{stm_num} 0 obj\n").as_bytes());
        out.extend_from_slice(&stream(
            &format!(
                "/Type /ObjStm /N {} /First {} /Filter /FlateDecode",
                packed.len(),
                header.len()
            ),
            &zlib_stored(&stm_data),
        ));
        out.extend_from_slice(b"\nendobj\n");
        let xref_at = out.len() as u32;
        entries.push((1, xref_at, 0));
        // rows of W [1 4 2], each after its PNG Up filter byte
        let mut rows = Vec::new();
        let mut prev = [0u8; 7];
        for (t, f2, f3) in &entries {
            let mut r = [0u8; 7];
            r[0] = *t;
            r[1..5].copy_from_slice(&f2.to_be_bytes());
            r[5..7].copy_from_slice(&f3.to_be_bytes());
            rows.push(2u8);
            for k in 0..7 {
                rows.push(r[k].wrapping_sub(prev[k]));
            }
            prev = r;
        }
        out.extend_from_slice(format!("{xref_num} 0 obj\n").as_bytes());
        out.extend_from_slice(&stream(
            &format!(
                "/Type /XRef /Size {} /W [1 4 2] /Root {root} 0 R /Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 7 >>",
                entries.len()
            ),
            &zlib_stored(&rows),
        ));
        out.extend_from_slice(format!("\nendobj\nstartxref\n{xref_at}\n%%EOF\n").as_bytes());
        out
    }
}

const HELVETICA: &str =
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";

/// A document whose pages have the given content streams (raw, or compressed when `flate`),
/// with font F1 = `font`. Returns the writer and the catalog's number.
fn doc(contents: &[&[u8]], font: &str, flate: bool) -> (Pdf, u32) {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(font);
    let mut kids = Vec::new();
    for c in contents {
        let data = if flate {
            stream("/Filter /FlateDecode", &zlib_stored(c))
        } else {
            stream("", c)
        };
        let content = p.add(data);
        kids.push(p.add(format!(
            "<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R >>"
        )));
    }
    let kid_refs: Vec<String> = kids.iter().map(|k| format!("{k} 0 R")).collect();
    p.set(
        pages,
        format!(
            "<< /Type /Pages /Kids [{}] /Count {} /Resources << /Font << /F1 {f1} 0 R >> >> >>",
            kid_refs.join(" "),
            kids.len()
        ),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    (p, catalog)
}

fn text_of(bytes: &[u8]) -> Result<Vec<String>, Fail> {
    extract_text(bytes, &Limits::default())
        .map(|e| e.pages.iter().map(|p| p.trim().to_string()).collect())
}

#[test]
fn reads_a_simple_page() {
    let (p, root) = doc(
        &[b"BT /F1 12 Tf 72 700 Td (Hello World) Tj ET"],
        HELVETICA,
        false,
    );
    assert_eq!(text_of(&p.table(root, "")).unwrap(), vec!["Hello World"]);
}

#[test]
fn reads_compressed_pages_with_separators() {
    let (p, root) = doc(
        &[
            b"BT /F1 12 Tf 72 700 Td (First page) Tj ET",
            b"BT /F1 12 Tf 72 700 Td (Second page) Tj ET",
        ],
        HELVETICA,
        true,
    );
    let e = extract_text(&p.table(root, ""), &Limits::default()).unwrap();
    assert_eq!(e.pages.len(), 2);
    assert_eq!(
        e.text(),
        "--- Page 1 ---\nFirst page\n\n--- Page 2 ---\nSecond page\n"
    );
}

#[test]
fn tj_adjustments_and_moves_become_spaces_and_lines() {
    let content: &[u8] = b"BT /F1 12 Tf 72 700 Td [(Hello) -300 (World)] TJ \
        [(, kern) -20 (ed)] TJ 0 -14 Td (Line two) Tj 60 0 Td (after) Tj \
        0 -40 Td (Para) Tj T* ET";
    assert_eq!(
        text_of(&doc(&[content], HELVETICA, false).0.table(1, "")).unwrap(),
        vec!["Hello World, kerned\nLine two after\n\nPara"]
    );
}

#[test]
fn quote_operators_and_leading() {
    let content: &[u8] = b"BT /F1 10 Tf 14 TL 72 700 Td (one) Tj (two) ' 2 0 (three) \" ET";
    assert_eq!(
        text_of(&doc(&[content], HELVETICA, false).0.table(1, "")).unwrap(),
        vec!["one\ntwo\nthree"]
    );
}

#[test]
fn win_ansi_and_differences() {
    let font =
        "<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding << /BaseEncoding /WinAnsiEncoding \
                /Differences [65 /uni00E9 /fi /g12 /f_l /quoteright] >> >>";
    let content: &[u8] = b"BT /F1 12 Tf 72 700 Td (\x93ABCDE\x94 caf\xe9 \x80) Tj ET";
    assert_eq!(
        text_of(&doc(&[content], font, false).0.table(1, "")).unwrap(),
        vec!["\u{201C}\u{e9}\u{fb01}fl\u{2019}\u{201D} caf\u{e9} \u{20ac}"]
    );
}

#[test]
fn standard_and_mac_roman_encodings() {
    let std_font = "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>";
    let content: &[u8] = b"BT /F1 12 Tf 72 700 Td (it\x27s \xaene\xba) Tj ET";
    assert_eq!(
        text_of(&doc(&[content], std_font, false).0.table(1, "")).unwrap(),
        vec!["it\u{2019}s \u{fb01}ne\u{201D}"]
    );
    let mac = "<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding /MacRomanEncoding >>";
    let content: &[u8] = b"BT /F1 12 Tf 72 700 Td (\x80\x8e\xa5) Tj ET";
    assert_eq!(
        text_of(&doc(&[content], mac, false).0.table(1, "")).unwrap(),
        vec!["\u{c4}\u{e9}\u{2022}"]
    );
}

#[test]
fn type0_identity_h_with_to_unicode() {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let cmap: &[u8] = b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n\
        1 begincodespacerange <0000> <FFFF> endcodespacerange\n\
        2 beginbfchar <0001> <0048> <0002> <00690021> endbfchar\n\
        1 beginbfrange <0010> <0012> <0410> endbfrange\n\
        1 beginbfrange <0020> <0021> [<D83DDE00> <0041>] endbfrange\n\
        endcmap CMapName currentdict /CMap defineresource pop end end";
    let tu = p.add(stream("/Filter /FlateDecode", &zlib_stored(cmap)));
    let desc = p.add(
        "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 1000 /W [1 [500 600] 16 18 700] >>",
    );
    let f1 = p.add(format!(
        "<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [{desc} 0 R] /ToUnicode {tu} 0 R >>"
    ));
    let content = p.add(stream(
        "",
        b"BT /F1 12 Tf 72 700 Td <00010002> Tj [<0010> -400 <00110012>] TJ <00200021> Tj ET",
    ));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    assert_eq!(
        text_of(&p.table(catalog, "")).unwrap(),
        vec!["Hi!\u{410} \u{411}\u{412}\u{1F600}A"]
    );
}

#[test]
fn xref_stream_object_stream_and_predictor() {
    let (p, root) = doc(
        &[b"BT /F1 12 Tf 72 700 Td (Packed objects) Tj ET"],
        HELVETICA,
        true,
    );
    // the catalog, the page tree and the font live in the object stream
    let bytes = p.xref_stream(root, &[1, 2, 3]);
    assert_eq!(text_of(&bytes).unwrap(), vec!["Packed objects"]);
}

#[test]
fn form_xobjects_and_inline_images() {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(HELVETICA);
    let form = p.add(stream(
        &format!("/Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /Font << /F1 {f1} 0 R >> >>"),
        b"BT /F1 12 Tf 0 0 Td (in the form) Tj ET",
    ));
    let content = p.add(stream(
        "",
        b"BI /W 2 /H 1 /BPC 8 /CS /G ID \x00(Tj) EI\nq 1 0 0 1 72 600 cm /X1 Do Q BT /F1 12 Tf 72 500 Td (after) Tj ET",
    ));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents [{content} 0 R] /Resources << /Font << /F1 {f1} 0 R >> /XObject << /X1 {form} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    assert_eq!(
        text_of(&p.table(catalog, "")).unwrap(),
        vec!["in the form\n\nafter"]
    );
}

#[test]
fn damaged_xref_is_rebuilt() {
    let (p, root) = doc(
        &[b"BT /F1 12 Tf 72 700 Td (Still here) Tj ET"],
        HELVETICA,
        false,
    );
    let mut bytes = p.table(root, "");
    // every xref offset off by 3, and the startxref offset wrong
    let s = String::from_utf8_lossy(&bytes).into_owned();
    let damaged = s
        .replace(" 00000 n", " 00003 n")
        .replace("startxref\n", "startxref\n9");
    bytes = damaged.into_bytes();
    assert_eq!(text_of(&bytes).unwrap(), vec!["Still here"]);
    // no xref at all
    let cut = String::from_utf8_lossy(&p.table(root, "")).into_owned();
    let cut = &cut[..cut.find("xref\n0").unwrap()];
    let mut b = cut.as_bytes().to_vec();
    b.extend_from_slice(format!("trailer\n<< /Root {root} 0 R >>\n%%EOF").as_bytes());
    assert_eq!(text_of(&b).unwrap(), vec!["Still here"]);
}

#[test]
fn wrong_stream_length_is_recovered() {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(HELVETICA);
    let content = p.add(
        b"<< /Length 99999999 >>\nstream\nBT /F1 12 Tf 72 700 Td (Long) Tj ET\nendstream".to_vec(),
    );
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    assert_eq!(text_of(&p.table(catalog, "")).unwrap(), vec!["Long"]);
}

#[test]
fn encrypted_and_textless_documents_are_refused() {
    let (p, root) = doc(
        &[b"BT /F1 12 Tf 72 700 Td (Secret) Tj ET"],
        HELVETICA,
        false,
    );
    let bytes = p.table(root, "/Encrypt << /Filter /Standard /V 2 /R 3 >>");
    assert_eq!(text_of(&bytes), Err(Fail::Encrypted));
    let (p, root) = doc(&[b"q 100 0 0 100 0 0 cm /Im1 Do Q", b""], HELVETICA, false);
    let e = text_of(&p.table(root, "")).unwrap_err();
    assert_eq!(e, Fail::NoText { pages: 2 });
    assert!(e.to_string().contains("text layer is empty"), "{e}");
    assert_eq!(text_of(b"hello"), Err(Fail::NotPdf));
}

#[test]
fn limits_are_refused_with_their_numbers() {
    let (p, root) = doc(
        &[
            b"BT /F1 12 Tf (a) Tj ET",
            b"BT /F1 12 Tf (b) Tj ET",
            b"BT /F1 12 Tf (c) Tj ET",
        ],
        HELVETICA,
        false,
    );
    let bytes = p.table(root, "");
    let lim = Limits {
        max_pages: 2,
        ..Limits::default()
    };
    let e = extract_text(&bytes, &lim).unwrap_err();
    assert!(
        e.to_string()
            .contains("more than 2 pages (--pdf-max-pages)"),
        "{e}"
    );
    let lim = Limits {
        max_bytes: 100,
        ..Limits::default()
    };
    let e = extract_text(&bytes, &lim).unwrap_err();
    assert!(
        e.to_string()
            .contains("more than the limit of 100 bytes (--pdf-max-mb)"),
        "{e}"
    );
    // exactly at the limits is accepted
    let lim = Limits {
        max_bytes: bytes.len() as u64,
        max_pages: 3,
        ..Limits::default()
    };
    assert_eq!(extract_text(&bytes, &lim).unwrap().pages.len(), 3);
}

#[test]
fn a_zip_bomb_stops_at_the_cap() {
    let bomb = zlib_bomb(20_000); // about 32 KiB expanding to 5.2 MB
    assert!(bomb.len() < 40_000);
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let content = p.add(stream("/Filter /FlateDecode", &bomb));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    let lim = Limits {
        max_decoded: 1 << 20,
        ..Limits::default()
    };
    let e = extract_text(&p.table(catalog, ""), &lim).unwrap_err();
    assert!(matches!(e, Fail::TooLarge(_)), "{e}");
    assert!(
        e.to_string()
            .contains("more than 1048576 bytes (--pdf-max-inflate-mb)"),
        "{e}"
    );
}

#[test]
fn the_budget_counts_every_stream_and_form_use() {
    // one small form drawn many times: each use is charged again
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let form = p.add(stream(
        "/Subtype /Form /Filter /FlateDecode",
        &zlib_stored(&[b' '; 4000]),
    ));
    let content = p.add(stream("", &b"/X Do ".repeat(1000)));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /XObject << /X {form} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    let lim = Limits {
        max_decoded: 1 << 20,
        ..Limits::default()
    };
    assert!(matches!(
        extract_text(&p.table(catalog, ""), &lim),
        Err(Fail::TooLarge(_))
    ));
}

/// Hostile structures: each must end in an error or in text, quickly and without a panic.
#[test]
fn hostile_structures_are_refused_without_panic() {
    let nested = format!("{}1{}", "[".repeat(100_000), "]".repeat(100_000));
    let mut cases: Vec<Vec<u8>> = Vec::new();
    // nesting far past the limit, in the catalog and in a content stream
    {
        let (mut p, root) = doc(&[b"BT /F1 12 Tf (x) Tj ET"], HELVETICA, false);
        p.set(
            root,
            format!("<< /Type /Catalog /Pages 2 0 R /Deep {nested} >>"),
        );
        cases.push(p.table(root, ""));
        let (p, root) = doc(&[nested.as_bytes()], HELVETICA, false);
        cases.push(p.table(root, ""));
    }
    // an object stream claiming a huge count, a huge /First, and an offset past its end
    for (n, first) in [("1000000000000", "5"), ("1", "99999999999"), ("1", "4")] {
        let mut p = Pdf::new();
        let root = p.reserve();
        let stm = p.add(stream(
            &format!("/Type /ObjStm /N {n} /First {first}"),
            b"1 999999 << /Type /Catalog >>",
        ));
        let _ = stm;
        let mut b = p.table(root, "");
        // point object 1 into the object stream through an xref stream appended at the end
        let s = String::from_utf8_lossy(&b).into_owned();
        b = s.replace("1 0 obj\nnull", "1 0 obj\n(gone)").into_bytes();
        cases.push(b);
        // the same object stream reached as a compressed object
        let mut q = Pdf::new();
        q.add("<< /Type /Catalog /Pages 3 0 R >>");
        q.add(stream(
            &format!("/Type /ObjStm /N {n} /First {first}"),
            b"3 0 << /Type /Pages /Kids [] >>",
        ));
        q.add("<< /Type /Pages /Kids [] /Count 0 >>");
        cases.push(q.xref_stream(1, &[3]));
    }
    // an xref stream with fields wider than 8 bytes, and one with rows past its end
    for w in ["[1 9 1]", "[1 4 2] /Index [0 1000000000]"] {
        let mut b = b"%PDF-1.7\n".to_vec();
        let at = b.len();
        b.extend_from_slice(b"1 0 obj\n");
        b.extend_from_slice(&stream(
            &format!("/Type /XRef /Size 3 /W {w} /Root 1 0 R"),
            &[1u8; 21],
        ));
        b.extend_from_slice(format!("\nendobj\nstartxref\n{at}\n%%EOF\n").as_bytes());
        cases.push(b);
    }
    // a /Prev chain that loops, a startxref past the end, a page tree that loops
    {
        let (p, root) = doc(&[b"BT /F1 12 Tf (x) Tj ET"], HELVETICA, false);
        let s = String::from_utf8_lossy(&p.table(root, "/Prev 9")).into_owned();
        let at = s.rfind("startxref\n").unwrap() + 10;
        let xref_at: String = s[at..].chars().take_while(char::is_ascii_digit).collect();
        cases.push(
            s.replace("/Prev 9", &format!("/Prev {xref_at}"))
                .into_bytes(),
        );
        cases.push(
            s.replace(
                &format!("startxref\n{xref_at}"),
                "startxref\n99999999999999",
            )
            .into_bytes(),
        );
        let mut p2 = Pdf::new();
        let c = p2.reserve();
        let pages = p2.reserve();
        p2.set(
            pages,
            format!("<< /Type /Pages /Kids [{pages} 0 R {pages} 0 R] >>"),
        );
        p2.set(c, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
        cases.push(p2.table(c, ""));
    }
    // references that chain into each other, and a stream whose /Length refers to itself
    {
        let mut p = Pdf::new();
        let c = p.add("<< /Type /Catalog /Pages 2 0 R >>");
        p.add("3 0 R");
        p.add("2 0 R");
        p.add(b"<< /Length 4 0 R >>\nstream\nabc\nendstream".to_vec());
        cases.push(p.table(c, ""));
    }
    // a broken ToUnicode CMap and a broken font
    {
        let font = "<< /Type /Font /Subtype /Type0 /Encoding /Identity-H /ToUnicode 99 0 R /DescendantFonts 5 >>";
        let (p, root) = doc(&[b"BT /F1 12 Tf <0001> Tj ET"], font, false);
        cases.push(p.table(root, ""));
    }
    for b in &cases {
        let _ = extract_text(b, &Limits::default());
    }
}

/// Truncations and byte flips of valid files never panic.
#[test]
fn truncated_and_corrupted_files_never_panic() {
    let (p, root) = doc(
        &[
            b"BT /F1 12 Tf 72 700 Td [(Hello) -300 (World)] TJ ET",
            b"BT /F1 12 Tf 72 700 Td (Page two) Tj ET",
        ],
        HELVETICA,
        true,
    );
    let mut files = vec![p.table(root, ""), p.xref_stream(root, &[1, 2, 3])];
    let mut q = Pdf::new();
    let cmap: &[u8] = b"1 begincodespacerange <00> <FF> endcodespacerange 1 beginbfrange <00> <FF> <0041> endbfrange";
    let tu = q.add(stream("", cmap));
    let c = q.add("<< /Type /Catalog /Pages 3 0 R >>");
    q.add(format!(
        "<< /Type /Pages /Kids [4 0 R] /Resources << /Font << /F1 << /Subtype /Type1 /ToUnicode {tu} 0 R >> >> >> >>"
    ));
    q.add("<< /Type /Page /Contents 5 0 R >>");
    q.add(stream(
        "/Filter [/ASCIIHexDecode /FlateDecode]",
        b"789c<not hex really>",
    ));
    files.push(q.table(c, ""));
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for f in &files {
        for cut in (0..f.len()).step_by(3) {
            let _ = extract_text(&f[..cut], &Limits::default());
        }
        for _ in 0..400 {
            let mut g = f.clone();
            for _ in 0..1 + rand() % 4 {
                let at = (rand() % g.len() as u64) as usize;
                g[at] = rand() as u8;
            }
            let _ = extract_text(&g, &Limits::default());
        }
    }
}

#[test]
fn filters_decode() {
    let mut b = Budget::new(1 << 20, "test", "bytes", "--test");
    let d =
        |f: &str| lex::Dict::from_iter([(b"Filter".to_vec(), lex::Obj::Name(f.as_bytes().into()))]);
    assert_eq!(
        filter::decode(&d("ASCIIHexDecode"), b"48 65 6c6C 6>", &mut b).unwrap(),
        b"Hell`"
    );
    assert_eq!(
        filter::decode(&d("ASCII85Decode"), b"<~87cURDZ~>", &mut b).unwrap(),
        b"Hello"
    );
    assert_eq!(
        filter::decode(
            &d("RunLengthDecode"),
            &[2, b'a', b'b', b'c', 254, b'x', 128],
            &mut b
        )
        .unwrap(),
        b"abcxxx"
    );
    // LZW: the classic example from the specification's LZW section
    assert_eq!(
        filter::decode(
            &d("LZWDecode"),
            &[0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01],
            &mut b
        )
        .unwrap(),
        b"-----A---B"
    );
    assert!(matches!(
        filter::decode(&d("DCTDecode"), b"", &mut b),
        Err(Fail::Unsupported(_))
    ));
}

// ---------------------------------------------------------------------------------------------
// The law review's hostile inputs (PR #136): each must now be refused, or finish quickly and
// within its budgets. The files are small and built here.
// ---------------------------------------------------------------------------------------------

/// One page whose content lists the same stream `times` times.
fn repeated_contents(stream_dict: &str, data: &[u8], times: usize) -> Vec<u8> {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(HELVETICA);
    let content = p.add(stream(stream_dict, data));
    let parts = vec![format!("{content} 0 R"); times].join(" ");
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents [{parts}] /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

/// A page drawing form X, which draws itself `fan` times (unfiltered).
fn self_drawing_form(fan: usize) -> Vec<u8> {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let form = p.reserve();
    let body = format!("{}BT /F1 1 Tf (x) Tj ET", "/X Do ".repeat(fan));
    let f1 = p.add(HELVETICA);
    p.set(
        form,
        stream(
            &format!(
                "/Type /XObject /Subtype /Form /Resources << /XObject << /X {form} 0 R >> /Font << /F1 {f1} 0 R >> >>"
            ),
            body.as_bytes(),
        ),
    );
    let content = p.add(stream("", b"/X Do"));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /XObject << /X {form} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

/// `n` objects that overlap: each object's header lies inside the one before it, as a stream
/// (all ending at one `endstream`, their /Length wrong) or as a string (nested parentheses), all
/// listed as page-tree kids, and no xref table (the file is rebuilt by a scan).
fn overlapping_objects(n: usize, as_strings: bool) -> Vec<u8> {
    let mut out = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n".to_vec();
    let kids: Vec<String> = (0..n).map(|k| format!("{} 0 R", k + 3)).collect();
    out.extend_from_slice(
        format!(
            "2 0 obj << /Type /Pages /Kids [{}] /Count {n} >> endobj\n",
            kids.join(" ")
        )
        .as_bytes(),
    );
    let filler = vec![b'x'; 900];
    for k in 0..n {
        if as_strings {
            out.extend_from_slice(format!("{} 0 obj (", k + 3).as_bytes());
        } else {
            out.extend_from_slice(
                format!("{} 0 obj << /Length 999999999 >> stream\n", k + 3).as_bytes(),
            );
        }
        out.extend_from_slice(&filler);
        out.push(b'\n');
    }
    if as_strings {
        out.extend_from_slice(&vec![b')'; n]);
        out.extend_from_slice(b"\nendobj\n");
    } else {
        out.extend_from_slice(b"\nendstream\nendobj\n");
    }
    out.extend_from_slice(b"trailer\n<< /Root 1 0 R >>\n%%EOF\n");
    out
}

/// A page whose font dictionary has `n` keys, each used once by `Tf`.
fn many_keys(n: usize) -> Vec<u8> {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(HELVETICA);
    let fonts: String = (0..n).map(|i| format!("/F{i} {f1} 0 R ")).collect();
    let fonts = p.add(format!("<< {fonts}>>"));
    let ops: String = (0..n).map(|i| format!("/F{i} 1 Tf ")).collect();
    let content = p.add(stream("", format!("BT {ops}(x) Tj ET").as_bytes()));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font {fonts} 0 R >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

/// A two-byte compressed stream whose PNG predictor declares rows of 64 MiB, listed `times`
/// times.
fn wide_predictor(times: usize) -> Vec<u8> {
    repeated_contents(
        "/Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 1048576 /Colors 32 /BitsPerComponent 16 >>",
        &zlib_stored(&[2, 0]),
        times,
    )
}

/// A page tree `levels` deep in which each level is one shared Kids array of `fan` direct
/// dictionaries, all pointing to the next level's array (`fan`^`levels` paths), with one page
/// at the bottom.
fn shared_kids(fan: usize, levels: usize) -> Vec<u8> {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(HELVETICA);
    let content = p.add(stream("", b"BT /F1 12 Tf (x) Tj ET"));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    let mut next = p.add(format!("[{page} 0 R]"));
    for _ in 0..levels {
        let items = vec![format!("<< /Kids {next} 0 R >>"); fan].join(" ");
        next = p.add(format!("[{items}]"));
    }
    p.set(
        pages,
        format!("<< /Type /Pages /Kids {next} 0 R /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

/// `pages` pages sharing one /Contents array of `times` references to one stream.
fn shared_contents(stream_dict: &str, data: &[u8], times: usize, pages: usize) -> Vec<u8> {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let root = p.reserve();
    let f1 = p.add(HELVETICA);
    let content = p.add(stream(stream_dict, data));
    let parts = p.add(format!(
        "[{}]",
        vec![format!("{content} 0 R"); times].join(" ")
    ));
    let kids: Vec<String> = (0..pages)
        .map(|_| {
            let k = p.add(format!(
                "<< /Type /Page /Parent {root} 0 R /Contents {parts} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
            ));
            format!("{k} 0 R")
        })
        .collect();
    p.set(
        root,
        format!(
            "<< /Type /Pages /Kids [{}] /Count {pages} >>",
            kids.join(" ")
        ),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {root} 0 R >>"));
    p.table(catalog, "")
}

/// An empty stream with `filters` ASCIIHex filters, listed `times` times by one page.
fn long_filter_chain(filters: usize, times: usize) -> Vec<u8> {
    let chain = vec!["/AHx"; filters].join(" ");
    repeated_contents(&format!("/Filter [{chain}]"), b"", times)
}

/// A Type0 font whose /W lists `entries` width lists, all one shared array of `widths` numbers.
fn shared_widths(entries: usize, widths: usize) -> Vec<u8> {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let list = p.add(format!("[{}]", vec!["500"; widths].join(" ")));
    let w = vec![format!("0 {list} 0 R"); entries].join(" ");
    let cid = p.add(format!(
        "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X /W [{w}] >>"
    ));
    let to_unicode = p.add(stream(
        "",
        b"1 begincodespacerange <0000> <FFFF> endcodespacerange 1 beginbfchar <0001> <0078> endbfchar",
    ));
    let f1 = p.add(format!(
        "<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [{cid} 0 R] /ToUnicode {to_unicode} 0 R >>"
    ));
    let content = p.add(stream("", b"BT /F1 12 Tf <0001> Tj ET"));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

/// A simple font whose ToUnicode CMap maps `entries` three-byte codes one by one (`bfchar`).
fn large_cmap(entries: usize) -> Vec<u8> {
    let mut cmap = b"1 begincodespacerange <00> <FF> endcodespacerange\n".to_vec();
    for chunk in (0..entries).collect::<Vec<_>>().chunks(100) {
        cmap.extend_from_slice(format!("{} beginbfchar\n", chunk.len()).as_bytes());
        for &k in chunk {
            cmap.extend_from_slice(format!("<{k:06X}> <{:04X}>\n", 0x4E00 + k % 0x5000).as_bytes());
        }
        cmap.extend_from_slice(b"endbfchar\n");
    }
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let to_unicode = p.add(stream("", &cmap));
    let f1 = p.add(format!(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode {to_unicode} 0 R >>"
    ));
    let content = p.add(stream("", b"BT /F1 12 Tf (x) Tj ET"));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

fn quick(what: &str, start: std::time::Instant) {
    let took = start.elapsed();
    assert!(took.as_secs() < 30, "{what} took {took:?}");
}

#[test]
fn unfiltered_streams_are_charged() {
    // review repro 1: the same unfiltered 64 KiB stream listed 200 times, 1 MiB budget
    let lim = Limits {
        max_decoded: 1 << 20,
        ..Limits::default()
    };
    let body = [b"BT /F1 12 Tf (a) Tj ET ".as_slice(), &[b' '; 65_536 - 23]].concat();
    let e = extract_text(&repeated_contents("", &body, 200), &lim).unwrap_err();
    assert!(
        e.to_string()
            .contains("its streams decode to more than 1048576 bytes (--pdf-max-inflate-mb)"),
        "{e}"
    );
    // an empty filter array copies too, and is charged the same way
    let e = extract_text(&repeated_contents("/Filter []", &body, 200), &lim).unwrap_err();
    assert!(matches!(e, Fail::TooLarge(_)), "{e}");
    // a single unfiltered stream larger than the budget (Codex's repro: 20 KB, budget 10 bytes)
    let lim = Limits {
        max_decoded: 10,
        ..Limits::default()
    };
    let e = extract_text(&repeated_contents("", &[b' '; 20_000], 1), &lim).unwrap_err();
    assert!(matches!(e, Fail::TooLarge(_)), "{e}");
    // within the budget, the same page is read
    let lim = Limits {
        max_decoded: 1 << 24,
        ..Limits::default()
    };
    let got = extract_text(&repeated_contents("", &body, 2), &lim).unwrap();
    assert_eq!(got.pages[0].split_whitespace().collect::<Vec<_>>(), ["aa"]);
}

#[test]
fn a_form_drawing_itself_stops_at_the_form_cap() {
    // review repro 1b: a form that draws itself 8 times, nested 12 deep (8^11 draws)
    let start = std::time::Instant::now();
    let e = extract_text(&self_drawing_form(8), &Limits::default()).unwrap_err();
    quick("the self-drawing form", start);
    assert!(
        e.to_string()
            .contains("it draws more than 100000 form XObjects (--pdf-max-forms)"),
        "{e}"
    );
    // a lower cap stops it sooner; forms within the cap are read
    let lim = Limits {
        max_forms: 3,
        ..Limits::default()
    };
    assert!(matches!(
        extract_text(&self_drawing_form(8), &lim),
        Err(Fail::TooLarge(_))
    ));
    let lim = Limits {
        max_forms: 11,
        ..Limits::default()
    };
    let got = extract_text(&self_drawing_form(0), &lim).unwrap();
    assert_eq!(got.pages[0].trim(), "x");
}

#[test]
fn overlapping_objects_are_charged_not_copied() {
    // review repro 2: 300 overlapping streams (about 276 KiB) rebuilt by a scan; stream data
    // is a span of the file, so the objects take little memory and the file is read
    let file = overlapping_objects(300, false);
    assert!(file.len() < 300 << 10, "{}", file.len());
    let lim = Limits {
        max_decoded: 1 << 20,
        ..Limits::default()
    };
    let start = std::time::Instant::now();
    let e = extract_text(&file, &lim).unwrap_err();
    quick("overlapping streams", start);
    assert!(matches!(e, Fail::NoText { .. }), "{e}");
    // the same with nested strings: each spans the rest of the file, so the parses are
    // charged to the object budget and stop at it
    let file = overlapping_objects(300, true);
    let start = std::time::Instant::now();
    let e = extract_text(&file, &lim).unwrap_err();
    quick("overlapping strings", start);
    assert!(
        e.to_string()
            .contains("its objects take more than 1048576 bytes (--pdf-max-inflate-mb)"),
        "{e}"
    );
}

#[test]
fn large_dictionaries_stay_linear() {
    // review repro 3: 80,000 keys in one dictionary, each looked up once by `Tf`
    let file = many_keys(80_000);
    let start = std::time::Instant::now();
    let got = extract_text(&file, &Limits::default()).unwrap();
    quick("80,000 keys", start);
    assert_eq!(got.pages[0].trim(), "x");
    // the last of a repeated key wins, in a small and in a large dictionary
    for n in [3usize, 50] {
        let keys: String = (0..n).map(|i| format!("/K{i} {i} ")).collect();
        let src = format!("<< {keys}/K1 /Last >>");
        let mut lx = lex::Lexer::new(
            src.as_bytes(),
            0,
            Budget::new(u64::MAX, "test", "bytes", "--test"),
        );
        let o = lx.object(0).unwrap();
        let d = o.as_dict().unwrap();
        assert_eq!(d.get(b"K1").and_then(lex::Obj::as_name), Some(&b"Last"[..]));
        assert_eq!(d.get(b"K2").and_then(lex::Obj::as_int), Some(2));
        assert_eq!(d.iter().count(), n);
    }
}

#[test]
fn predictor_work_is_charged_before_it_starts() {
    // review repro 4: a two-byte stream declaring 64 MiB rows, listed 20 times
    let start = std::time::Instant::now();
    let lim = Limits {
        max_decoded: 8 << 20,
        ..Limits::default()
    };
    let e = extract_text(&wide_predictor(20), &lim).unwrap_err();
    quick("the wide predictor", start);
    assert!(
        e.to_string()
            .contains("its streams decode to more than 8388608 bytes"),
        "{e}"
    );
}

#[test]
fn a_page_split_over_streams_keeps_its_state() {
    // Codex repro: the font is selected in the first stream and used in the second
    let (p, root) = doc(&[b"BT /F1 12 Tf 72 700 Td (First) Tj"], HELVETICA, false);
    let mut p = p;
    let second = p.add(stream("", b"0 -14 Td (Second) Tj ET"));
    // the page is the last object written by `doc`: give it both streams
    let page = p.objs.len() as u32 - 1;
    let first = page - 1;
    let body = String::from_utf8(p.objs[page as usize - 1].clone()).unwrap();
    let body = body.replace(
        &format!("/Contents {first} 0 R"),
        &format!("/Contents [{first} 0 R {second} 0 R]"),
    );
    p.set(page, body);
    let got = extract_text(&p.table(root, ""), &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "First\nSecond");
}

#[test]
fn the_symbol_font_uses_its_built_in_encoding() {
    // Codex repro: /BaseFont /Symbol without /Encoding or /ToUnicode
    let symbol = "<< /Type /Font /Subtype /Type1 /BaseFont /Symbol >>";
    let (p, root) = doc(&[b"BT /F1 12 Tf (abg) Tj ET"], symbol, false);
    let got = extract_text(&p.table(root, ""), &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "\u{3b1}\u{3b2}\u{3b3}");
    // a subset tag and a style suffix name the same font; /Differences still apply over it
    let tagged = "<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+Symbol,Bold /Encoding << /Differences [98 /x] >> >>";
    let (p, root) = doc(&[b"BT /F1 12 Tf (ab\\245) Tj ET"], tagged, false);
    let got = extract_text(&p.table(root, ""), &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "\u{3b1}x\u{221e}");
    // other fonts keep StandardEncoding
    let (p, root) = doc(&[b"BT /F1 12 Tf (abg) Tj ET"], HELVETICA, false);
    let got = extract_text(&p.table(root, ""), &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "abg");
}

#[test]
fn a_shared_page_tree_is_walked_once() {
    // second review, repro 1: each level one shared Kids array of 10 direct nodes, 20 levels
    // deep (10^20 paths in a 5 KB file); a node is read once, whichever way it is reached
    let file = shared_kids(10, 20);
    assert!(file.len() < 6_000, "{}", file.len());
    let start = std::time::Instant::now();
    let got = extract_text(&file, &Limits::default()).unwrap();
    quick("the shared page tree", start);
    assert_eq!(got.pages.len(), 1);
    assert_eq!(got.pages[0].trim(), "x");
    // every node visited is a step of work, so a small work budget stops the walk
    let lim = Limits {
        max_work: 50,
        ..Limits::default()
    };
    let e = extract_text(&file, &lim).unwrap_err();
    assert!(
        e.to_string()
            .contains("it costs more than 50 steps of work (--pdf-max-work)"),
        "{e}"
    );
}

#[test]
fn a_long_filter_chain_read_many_times_is_charged() {
    // second review, repro 2: an empty stream with 10,000 filters, read 10,000 times
    let file = long_filter_chain(10_000, 10_000);
    let start = std::time::Instant::now();
    let e = extract_text(&file, &Limits::default()).unwrap_err();
    quick("the long filter chain", start);
    assert!(e.to_string().contains("(--pdf-max-work)"), "{e}");
    // a short chain read a few times is read
    let e = extract_text(&long_filter_chain(3, 3), &Limits::default()).unwrap_err();
    assert!(matches!(e, Fail::NoText { pages: 1 }), "{e}");
}

#[test]
fn empty_reads_are_charged() {
    // second review, repro 3: 100,000 references to one empty stream, shared by 500 pages
    let file = shared_contents("", b"", 100_000, 500);
    let start = std::time::Instant::now();
    let e = extract_text(&file, &Limits::default()).unwrap_err();
    quick("the empty reads", start);
    assert!(e.to_string().contains("(--pdf-max-work)"), "{e}");
    // a shared /Contents array is still read for every page
    let got = extract_text(
        &shared_contents("", b"BT /F1 12 Tf (x) Tj ET", 2, 3),
        &Limits::default(),
    )
    .unwrap();
    assert_eq!(got.pages.len(), 3);
    assert!(got.pages.iter().all(|p| p.contains('x')), "{got:?}");
}

#[test]
fn a_shared_width_list_is_charged_by_its_expanded_size() {
    // second review, repro 4: /W lists 1,000 times one shared array of 100,000 widths
    let file = shared_widths(1_000, 100_000);
    let start = std::time::Instant::now();
    let e = extract_text(&file, &Limits::default()).unwrap_err();
    quick("the shared width list", start);
    assert!(matches!(e, Fail::TooLarge(_)), "{e}");
    // a few entries are read
    let got = extract_text(&shared_widths(2, 10), &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "x");
}

#[test]
fn cmap_entries_are_charged_to_the_object_budget() {
    // second review, repro 5: a ToUnicode CMap of 1,000,000 entries (16 MB decoded)
    let file = large_cmap(1_000_000);
    let got = extract_text(&file, &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "x");
    // the entries it keeps take more memory than 32 MiB: under that budget it is refused,
    // though its 16 MB decode fits
    let lim = Limits {
        max_decoded: 32 << 20,
        ..Limits::default()
    };
    let e = extract_text(&file, &lim).unwrap_err();
    assert!(
        e.to_string()
            .contains("its objects take more than 33554432 bytes"),
        "{e}"
    );
}

#[test]
fn glyph_text_is_charged_before_it_is_kept() {
    // Codex repro: one code mapped to 64 characters, shown 10,000 times, 20,000-byte budget
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let dst = "0041".repeat(64);
    let to_unicode = p.add(stream(
        "",
        format!("1 begincodespacerange <00> <FF> endcodespacerange 1 beginbfchar <61> <{dst}> endbfchar")
            .as_bytes(),
    ));
    let f1 = p.add(format!(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode {to_unicode} 0 R >>"
    ));
    let shown = "61".repeat(10_000);
    let content = p.add(stream(
        "/Filter /ASCIIHexDecode",
        format!(
            "{}>",
            hex(format!("BT /F1 12 Tf <{shown}> Tj ET").as_bytes())
        )
        .as_bytes(),
    ));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    let lim = Limits {
        max_decoded: 200_000,
        ..Limits::default()
    };
    let e = extract_text(&p.table(catalog, ""), &lim).unwrap_err();
    assert!(
        e.to_string().contains("its text is more than 200000 bytes"),
        "{e}"
    );
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn a_form_starts_in_the_state_of_its_do() {
    // Codex repro: the font is selected on the page, the form shows text with it
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(HELVETICA);
    let form = p.add(stream(
        "/Type /XObject /Subtype /Form",
        b"BT (Inside) Tj ET",
    ));
    let content = p.add(stream(
        "",
        b"BT /F1 12 Tf 72 700 Td (Page) Tj ET /F1 12 Tf /X Do",
    ));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> /XObject << /X {form} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    let got = extract_text(&p.table(catalog, ""), &Limits::default()).unwrap();
    assert!(got.pages[0].contains("Inside"), "{:?}", got.pages[0]);
    assert!(got.pages[0].contains("Page"), "{:?}", got.pages[0]);
}

#[test]
fn an_ext_g_state_selects_a_font() {
    // Codex repro: /GS1 << /Font [F 12] >> selects the font that `Tj` shows with
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f = p.add(HELVETICA);
    let content = p.add(stream("", b"/GS1 gs BT 72 700 Td (Hello GS) Tj ET"));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /ExtGState << /GS1 << /Font [{f} 0 R 12] >> >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    let got = extract_text(&p.table(catalog, ""), &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "Hello GS");
}

#[test]
fn budgets_keep_their_spend_on_a_refusal() {
    // a refused charge leaves the spend as it was, and the next charge that fits is counted
    let mut b = Budget::new(10, "test", "steps", "--test");
    b.take(7).unwrap();
    assert!(b.take(4).is_err());
    assert_eq!(b.used(), 7);
    assert_eq!(b.room(), 3);
    b.take(0).unwrap();
    assert_eq!(b.used(), 7);
    b.take(3).unwrap();
    assert_eq!(b.used(), 10);
    assert!(b.take(1).is_err());
    assert_eq!(b.used(), 10);
}

#[test]
fn the_default_work_budget_is_pinned() {
    assert_eq!(Limits::default().max_work, 20_000_000);
}

/// A page that selects `n` distinct simple fonts (each its own object) and shows a code in each.
fn many_fonts(n: usize) -> Vec<u8> {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let mut names = Vec::new();
    let mut content = String::from("BT ");
    for i in 0..n {
        let f = p.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
        names.push(format!("/F{i} {f} 0 R"));
        content.push_str(&format!("/F{i} 12 Tf (x) Tj "));
    }
    content.push_str("ET");
    let c = p.add(stream("", content.as_bytes()));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 612 792] /Contents {c} 0 R /Resources << /Font << {} >> >> >>",
        names.join(" ")
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

#[test]
fn simple_font_tables_are_charged_to_the_object_budget() {
    // Codex repro: 2,000 distinct simple fonts, each keeping a 256-entry code table
    let file = many_fonts(2_000);
    assert!(file.len() < 300_000, "{}", file.len());
    let lim = Limits {
        max_decoded: 4 << 20,
        ..Limits::default()
    };
    let e = extract_text(&file, &lim).unwrap_err();
    assert!(
        e.to_string()
            .contains("its objects take more than 4194304 bytes"),
        "{e}"
    );
    // under the default limits the same page is read, and a few fonts fit the small budget
    let got = extract_text(&file, &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "x".repeat(2_000));
    let got = extract_text(&many_fonts(20), &lim).unwrap();
    assert_eq!(got.pages[0].trim(), "x".repeat(20));
}

/// The offset the last `startxref` of a file names.
fn last_startxref(file: &[u8]) -> u64 {
    let at = file
        .windows(9)
        .rposition(|w| w == b"startxref")
        .expect("a startxref")
        + 9;
    std::str::from_utf8(&file[at..])
        .ok()
        .and_then(|t| t.split_whitespace().next())
        .and_then(|t| t.parse().ok())
        .expect("its offset")
}

/// `base` with an incremental update appended: an xref section (classic table or xref stream)
/// whose entries are `entries` (object number, `None` for free or `Some(offset)`), chained to
/// the base file's own section by /Prev.
fn updated(
    base: &[u8],
    root: u32,
    size: u32,
    entries: &[(u32, Option<u64>)],
    as_stream: bool,
) -> Vec<u8> {
    let prev = last_startxref(base);
    let mut out = base.to_vec();
    let at = out.len();
    if as_stream {
        // W [1 4 2], unfiltered, one row per entry
        let mut rows = Vec::new();
        let mut index = Vec::new();
        for (num, off) in entries {
            index.push(format!("{num} 1"));
            // type 1 at the offset, or type 0 (free, next generation 1)
            rows.push(u8::from(off.is_some()));
            rows.extend_from_slice(&(off.unwrap_or(0) as u32).to_be_bytes());
            rows.extend_from_slice(&[0, u8::from(off.is_none())]);
        }
        out.extend_from_slice(format!("{size} 0 obj\n").as_bytes());
        out.extend_from_slice(&stream(
            &format!(
                "/Type /XRef /Size {} /W [1 4 2] /Index [{}] /Root {root} 0 R /Prev {prev}",
                size + 1,
                index.join(" ")
            ),
            &rows,
        ));
        out.extend_from_slice(b"\nendobj\n");
    } else {
        out.extend_from_slice(b"xref\n");
        for (num, off) in entries {
            let line = match off {
                Some(o) => format!("{num} 1\n{o:010} 00000 n\r\n"),
                None => format!("{num} 1\n0000000000 00001 f\r\n"),
            };
            out.extend_from_slice(line.as_bytes());
        }
        out.extend_from_slice(
            format!("trailer\n<< /Size {size} /Root {root} 0 R /Prev {prev} >>\n").as_bytes(),
        );
    }
    out.extend_from_slice(format!("startxref\n{at}\n%%EOF\n").as_bytes());
    out
}

#[test]
fn a_freed_object_stays_deleted() {
    // Codex repro: an update frees the second page's content stream (object 6)
    let (p, root) = doc(
        &[
            b"BT /F1 12 Tf (Kept) Tj ET",
            b"BT /F1 12 Tf (Deleted secret) Tj ET",
        ],
        HELVETICA,
        false,
    );
    let base = p.table(root, "");
    assert_eq!(text_of(&base).unwrap(), vec!["Kept", "Deleted secret"]);
    let size = p.objs.len() as u32 + 1;
    // freed in a classic table
    let file = updated(&base, root, size, &[(6, None)], false);
    assert_eq!(text_of(&file).unwrap(), vec!["Kept", ""]);
    // freed by a type-0 entry of an xref stream
    let file = updated(&base, root, size, &[(6, None)], true);
    assert_eq!(text_of(&file).unwrap(), vec!["Kept", ""]);
    // an update that keeps the object (an in-use entry at its old offset) still reads it
    let at = (base.windows(8).position(|w| w == b"\n6 0 obj").unwrap() + 1) as u64;
    let file = updated(&base, root, size, &[(6, Some(at))], false);
    assert_eq!(text_of(&file).unwrap(), vec!["Kept", "Deleted secret"]);
}

#[test]
fn a_hybrid_tables_free_entries_yield_to_its_stream() {
    // a hybrid file: the table leaves the objects of the object stream free, and its /XRefStm
    // lists them in the stream
    let (p, root) = doc(
        &[b"BT /F1 12 Tf 72 700 Td (Packed objects) Tj ET"],
        HELVETICA,
        true,
    );
    let mut file = p.xref_stream(root, &[1, 2, 3]);
    let stm = last_startxref(&file);
    let at = file.len();
    file.extend_from_slice(
        format!(
            "xref\n0 4\n0000000000 65535 f\r\n0000000000 00000 f\r\n0000000000 00000 f\r\n0000000000 00000 f\r\ntrailer\n<< /Size {} /Root {root} 0 R /XRefStm {stm} >>\nstartxref\n{at}\n%%EOF\n",
            p.objs.len() + 3
        )
        .as_bytes(),
    );
    assert_eq!(text_of(&file).unwrap(), vec!["Packed objects"]);
}

/// The heap `f` holds at its peak, as the allocator counts it on this thread.
fn peak_heap<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let meter = crate::heap_meter::Metered::start(u64::MAX);
    let got = f();
    let peak = crate::heap_meter::reading().expect("the meter is on").peak;
    drop(meter);
    (got, peak)
}

/// A one-page document (read through a classic table) whose newer cross-reference section is
/// a compressed xref stream of `258 * repeats + 1` one-byte rows of `byte`, for objects
/// numbered from 1000. With `/W [0 1 0]` each row is an object at an offset, with `/W [1 0 0]`
/// each row is its type (0: a free entry; any other byte: an entry kept nowhere).
fn xref_rows(w: &str, byte: u8, repeats: usize) -> Vec<u8> {
    let rows = 258 * repeats + 1;
    let (p, root) = doc(&[b"BT /F1 12 Tf (x) Tj ET"], HELVETICA, false);
    let mut file = p.table(root, "");
    let prev = last_startxref(&file);
    let at = file.len();
    file.extend_from_slice(format!("{} 0 obj\n", p.objs.len() + 1).as_bytes());
    file.extend_from_slice(&stream(
        &format!(
            "/Type /XRef /Size {} /W {w} /Index [1000 {rows}] /Root {root} 0 R /Prev {prev} /Filter /FlateDecode",
            1000 + rows
        ),
        &zlib_run(byte, repeats),
    ));
    file.extend_from_slice(format!("\nendobj\nstartxref\n{at}\n%%EOF\n").as_bytes());
    file
}

#[test]
fn xref_stream_rows_are_charged() {
    // third review, repro 1: about 10.3M one-byte rows in a 63 KB file; every entry the
    // cross-reference map keeps is charged to the object budget (here 16 MiB)
    let lim = Limits {
        max_decoded: 16 << 20,
        ..Limits::default()
    };
    for (w, byte) in [("[0 1 0]", b'a'), ("[1 0 0]", 0u8)] {
        let file = xref_rows(w, byte, 40_000);
        assert!(file.len() < 70_000, "{}", file.len());
        let start = std::time::Instant::now();
        let (got, peak) = peak_heap(|| extract_text(&file, &lim));
        quick("the xref rows", start);
        let e = got.unwrap_err();
        assert!(
            e.to_string()
                .contains("its objects take more than 16777216 bytes"),
            "{w}: {e}"
        );
        assert!(peak < 64 << 20, "{w}: {peak}");
    }
    // rows that keep no entry still cost a step each
    let file = xref_rows("[1 0 0]", b'a', 40_000);
    let e = extract_text(
        &file,
        &Limits {
            max_work: 1_000_000,
            ..Limits::default()
        },
    )
    .unwrap_err();
    assert!(
        e.to_string()
            .contains("it costs more than 1000000 steps of work (--pdf-max-work)"),
        "{e}"
    );
    // a few rows: the document is read through its older section
    for (w, byte) in [("[0 1 0]", b'a'), ("[1 0 0]", 0u8), ("[1 0 0]", b'a')] {
        let got = extract_text(&xref_rows(w, byte, 4), &Limits::default()).unwrap();
        assert_eq!(got.pages[0].trim(), "x", "{w}");
    }
}

/// A page whose content is the first of `levels` streams, each holding an array of `items`
/// numbers in its dictionary and naming the next one as its /Length (a stream, so its data is
/// found by its `endstream`); reading the first stream reads every other one.
fn length_chain(levels: usize, items: usize) -> Vec<u8> {
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(HELVETICA);
    let first = p.objs.len() as u32 + 1;
    let array = vec!["0"; items].join(" ");
    for k in 0..levels as u32 {
        let length = if k + 1 < levels as u32 {
            format!("{} 0 R", first + k + 1)
        } else {
            "22".to_string()
        };
        let mut body = format!("<< /A [{array}] /Length {length} >>\nstream\n").into_bytes();
        body.extend_from_slice(b"BT /F1 12 Tf (x) Tj ET\nendstream");
        p.add(body);
    }
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {first} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

#[test]
fn a_length_chain_is_charged_level_by_level() {
    // third review, repro 2: 31 streams, each holding 20,000 numbers (about 0.8 MB of objects,
    // under the 1 MiB budget alone), each one's /Length the next one; the objects a parse holds
    // are charged before it reads its /Length, so the next level has only the room left
    let file = length_chain(31, 20_000);
    let lim = Limits {
        max_decoded: 1 << 20,
        ..Limits::default()
    };
    let (got, peak) = peak_heap(|| extract_text(&file, &lim));
    let e = got.unwrap_err();
    assert!(
        e.to_string()
            .contains("its objects take more than 1048576 bytes (--pdf-max-inflate-mb)"),
        "{e}"
    );
    assert!(peak < 6 << 20, "{peak}");
    // a short chain within the budget is read
    let got = extract_text(&length_chain(3, 10), &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "x");
}

/// A page that selects F1 (with `select`: `Tf`, or an ExtGState's /Font), a simple font whose
/// ToUnicode CMap keeps `entries` entries, then shows a string in an array of `items` numbers
/// that follows the selection right away.
fn font_then_operands(select: &str, entries: usize, items: usize) -> Vec<u8> {
    let mut cmap = b"1 begincodespacerange <00> <FF> endcodespacerange\n".to_vec();
    for chunk in (0..entries).collect::<Vec<_>>().chunks(100) {
        cmap.extend_from_slice(format!("{} beginbfchar\n", chunk.len()).as_bytes());
        for &k in chunk {
            cmap.extend_from_slice(format!("<{k:06X}> <0041>\n").as_bytes());
        }
        cmap.extend_from_slice(b"endbfchar\n");
    }
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let to_unicode = p.add(stream("", &cmap));
    let f1 = p.add(format!(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode {to_unicode} 0 R >>"
    ));
    let content = format!("BT {select} [(x) {}] TJ ET", vec!["0"; items].join(" "));
    let content = p.add(stream("", content.as_bytes()));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> /ExtGState << /G << /Font [{f1} 0 R 12] >> >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

#[test]
fn operands_after_a_font_have_the_room_it_leaves() {
    // third review, the content-stream case of repro 2: a font whose CMap keeps about 2.9 MB,
    // then an array of about 1.6 MB, each under the 4 MiB budget alone; the operands that
    // follow a font read have only the room the font leaves
    let lim = Limits {
        max_decoded: 4 << 20,
        ..Limits::default()
    };
    for select in ["/F1 12 Tf", "/G gs"] {
        let e = extract_text(&font_then_operands(select, 25_000, 40_000), &lim).unwrap_err();
        assert!(
            e.to_string()
                .contains("its objects take more than 4194304 bytes"),
            "{select}: {e}"
        );
        // each alone is read
        let got = extract_text(&font_then_operands(select, 25_000, 10), &lim).unwrap();
        assert_eq!(got.pages[0].trim(), "x", "{select}");
        let got = extract_text(&font_then_operands(select, 10, 40_000), &lim).unwrap();
        assert_eq!(got.pages[0].trim(), "x", "{select}");
    }
}

/// A page showing `258 * repeats + 1` one-byte codes in a composite font whose embedded CMap
/// has one one-byte codespace range and 255 four-byte ranges none of the codes is in. The
/// string is split over three content streams, the codes compressed.
fn composite_glyphs(repeats: usize) -> Vec<u8> {
    let mut space = String::from("256 begincodespacerange\n<00> <00>\n");
    for k in 0..255u32 {
        space.push_str(&format!("<{:08X}> <{:08X}>\n", 2 * k, 2 * k + 1));
    }
    space.push_str("endcodespacerange\n");
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let enc = p.add(stream("/Type /CMap", space.as_bytes()));
    let cid = p.add("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X >>");
    let f1 = p.add(format!(
        "<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding {enc} 0 R /DescendantFonts [{cid} 0 R] >>"
    ));
    let a = p.add(stream("", b"BT /F1 12 Tf ("));
    let b = p.add(stream("/Filter /FlateDecode", &zlib_run(b'a', repeats)));
    let c = p.add(stream("", b") Tj ET"));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents [{a} 0 R {b} 0 R {c} 0 R] /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    p.table(catalog, "")
}

#[test]
fn composite_glyphs_cost_a_step_each() {
    // third review, repro 3: 5.2M codes in a 39 KB file, each tried against 256 codespace
    // ranges; the ranges are indexed, and each code is a step of work
    let file = composite_glyphs(20_000);
    assert!(file.len() < 45_000, "{}", file.len());
    let start = std::time::Instant::now();
    let e = extract_text(&file, &Limits::default()).unwrap_err();
    quick("the composite glyphs", start);
    assert!(matches!(e, Fail::NoText { pages: 1 }), "{e}");
    let e = extract_text(
        &file,
        &Limits {
            max_work: 1_000_000,
            ..Limits::default()
        },
    )
    .unwrap_err();
    assert!(
        e.to_string()
            .contains("it costs more than 1000000 steps of work (--pdf-max-work)"),
        "{e}"
    );
}

#[test]
fn indexed_codespace_ranges_give_the_lengths_a_scan_gives() {
    // overlapping and nested ranges of each length, against a scan of every range
    let ranges: Vec<(usize, u32, u32)> = vec![
        (1, 0x00, 0x1F),
        (1, 0x10, 0x18),
        (1, 0x80, 0x9F),
        (2, 0x8140, 0x9FFC),
        (2, 0x8000, 0x8150),
        (2, 0xE040, 0xFCFC),
        (3, 0x00A0_0000, 0x00A0_FFFF),
        (4, 0x2000_0000, 0x2FFF_FFFF),
        (4, 0x2100_0000, 0x2100_00FF),
    ];
    let mut src = format!("{} begincodespacerange\n", ranges.len());
    for &(n, lo, hi) in &ranges {
        let w = 2 * n;
        src.push_str(&format!("<{lo:0w$X}> <{hi:0w$X}>\n"));
    }
    src.push_str("endcodespacerange\n");
    let mut mem = Budget::new(u64::MAX, "test", "bytes", "--test");
    let mut work = Budget::new(u64::MAX, "test", "steps", "--test");
    let m = font::CMap::parse(src.as_bytes(), &mut mem, &mut work).unwrap();
    let scan = |b: &[u8]| -> Option<usize> {
        for n in 1..=4usize.min(b.len()) {
            let c = b[..n].iter().fold(0u32, |a, &x| a << 8 | u32::from(x));
            if ranges
                .iter()
                .any(|&(sn, lo, hi)| sn == n && (lo..=hi).contains(&c))
            {
                return Some(n);
            }
        }
        Some(1)
    };
    let mut x = 0x2545_F491u32;
    for _ in 0..20_000 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let b = x.to_be_bytes();
        for len in 1..=4 {
            assert_eq!(m.code_len(&b[..len]), scan(&b[..len]), "{:02X?}", &b[..len]);
        }
    }
    for b in [
        &[0x81u8, 0x40][..],
        &[0x20, 0, 0, 0],
        &[0x9F],
        &[0xA0, 0x00, 0x05],
    ] {
        assert_eq!(m.code_len(b), scan(b), "{b:02X?}");
    }
    let none = font::CMap::parse(b"", &mut mem, &mut work).unwrap();
    assert_eq!(none.code_len(b"ab"), None);
}

#[test]
fn the_heap_a_read_holds_is_capped() {
    // the backstop: a read that holds more heap than --pdf-max-heap-mb is refused at the next
    // charge, whatever path holds it (here a 1M-entry CMap, about 115 MB, within its budgets)
    let file = large_cmap(1_000_000);
    let lim = Limits {
        max_heap: 32 << 20,
        ..Limits::default()
    };
    let (got, peak) = peak_heap(|| extract_text(&file, &lim));
    let e = got.unwrap_err();
    assert!(
        e.to_string()
            .contains("reading it holds more than 33554432 bytes of memory (--pdf-max-heap-mb)"),
        "{e}"
    );
    // it may pass the cap by what is allocated between two charges (here a hash table that
    // doubles); unchecked, this read holds about 200 MB
    assert!(peak < 64 << 20, "{peak}");
    // the meter is on only while a PDF is read
    assert_eq!(crate::heap_meter::reading(), None);
    let got = extract_text(&file, &Limits::default()).unwrap();
    assert_eq!(got.pages[0].trim(), "x");
    assert_eq!(crate::heap_meter::reading(), None);
}

#[test]
fn the_default_heap_cap_is_pinned() {
    assert_eq!(Limits::default().max_heap, 1024 << 20);
    // above what the byte budgets may hold together
    let l = Limits::default();
    assert!(l.max_heap >= 3 * l.max_decoded);
}

#[test]
fn a_shared_filter_name_is_not_copied_per_use() {
    // Codex repro: a 64 KiB name, one indirect object, listed 1,000 times in /Filter
    let mut p = Pdf::new();
    let catalog = p.reserve();
    let pages = p.reserve();
    let f1 = p.add(HELVETICA);
    let name = p.add(format!("/{}", "N".repeat(65_536)));
    let filters = vec![format!("{name} 0 R"); 1_000].join(" ");
    let content = p.add(stream(&format!("/Filter [{filters}]"), b"x"));
    let page = p.add(format!(
        "<< /Type /Page /Parent {pages} 0 R /Contents {content} 0 R /Resources << /Font << /F1 {f1} 0 R >> >> >>"
    ));
    p.set(
        pages,
        format!("<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>"),
    );
    p.set(catalog, format!("<< /Type /Catalog /Pages {pages} 0 R >>"));
    let file = p.table(catalog, "");
    assert!(file.len() < 80_000, "{}", file.len());
    let lim = Limits {
        max_decoded: 1 << 20,
        ..Limits::default()
    };
    let (got, peak) = peak_heap(|| extract_text(&file, &lim));
    // the unknown filter leaves the page without text; the names are never copied
    assert!(matches!(got, Err(Fail::NoText { pages: 1 })), "{got:?}");
    assert!(peak < 8 << 20, "{peak}");
}
