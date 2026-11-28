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
    b.code(0x30 + 97, 8); // literal 'a'
    for _ in 0..repeats {
        b.code(0xC0 + 5, 8); // length 258
        b.put(0, 5); // distance 1
    }
    b.code(0, 7); // end of block
    b.put(0, 7);
    b.out.extend_from_slice(&[0, 0, 0, 0]);
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
    let mut b = Budget::new(1 << 20, "test");
    let d = |f: &str| {
        lex::Dict(vec![(
            b"Filter".to_vec(),
            lex::Obj::Name(f.as_bytes().to_vec()),
        )])
    };
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
