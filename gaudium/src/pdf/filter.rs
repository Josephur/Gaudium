//! Stream filters (ISO 32000-1, section 7.4): FlateDecode (with PNG predictors), LZWDecode,
//! ASCIIHexDecode, ASCII85Decode and RunLengthDecode.
//!
//! Every read of a stream is charged to the document's decode budget, filtered or not: the
//! raw bytes read, then each stage's output. Every decoder gets the room the budget still leaves
//! (`pdf_budget_room`) and stops with an error as soon as its output would grow past it, so a
//! small stream that expands without end (a zip bomb) stops at the cap, and a stream read again
//! (a page listing it many times, a form drawn again) costs again. A PNG predictor's work is
//! checked against the room and charged before it starts (`pdf_predict_fits`). Image filters
//! (DCT, JPX, CCITT, JBIG2) hold no text and are refused.

use super::lex::{Dict, Obj};
use super::{Budget, Fail};
use crate::verified as v;

/// Decodes `data` through the filters named in `dict`. The raw bytes are charged first; each
/// stage may produce the bytes the budget still has room for, and its output is charged.
pub(crate) fn decode(dict: &Dict, data: &[u8], budget: &mut Budget) -> Result<Vec<u8>, Fail> {
    let filters: Vec<Vec<u8>> = match dict.get(b"Filter").or_else(|| dict.get(b"F")) {
        None | Some(Obj::Null) => Vec::new(),
        Some(Obj::Name(n)) => vec![n.to_vec()],
        Some(Obj::Array(a)) => a
            .iter()
            .filter_map(|o| o.as_name().map(<[u8]>::to_vec))
            .collect(),
        Some(_) => return Err(Fail::damaged("a stream's /Filter is not a name")),
    };
    let parms: Vec<Option<&Dict>> = match dict.get(b"DecodeParms").or_else(|| dict.get(b"DP")) {
        Some(Obj::Dict(d)) => vec![Some(d)],
        Some(Obj::Array(a)) => a.iter().map(Obj::as_dict).collect(),
        _ => Vec::new(),
    };
    // the bytes read are charged before they are copied or decoded
    budget.take(data.len() as u64)?;
    let mut cur: Option<Vec<u8>> = None;
    for (i, f) in filters.iter().enumerate() {
        let parm = parms.get(i).copied().flatten();
        let input: &[u8] = cur.as_deref().unwrap_or(data);
        let room = budget.room();
        let cap = usize::try_from(room).unwrap_or(usize::MAX);
        let (stage, predicted) = match f.as_slice() {
            b"FlateDecode" | b"Fl" => (
                crate::vision::decode::inflate::zlib_decompress(input, cap)
                    .map_err(|e| inflate_fail(&e, room)),
                true,
            ),
            b"LZWDecode" | b"LZW" => {
                let early = parm
                    .and_then(|p| p.get(b"EarlyChange"))
                    .and_then(Obj::as_int)
                    .unwrap_or(1);
                (lzw(input, cap, early != 0, room), true)
            }
            b"ASCIIHexDecode" | b"AHx" => (ascii_hex(input, cap, room), false),
            b"ASCII85Decode" | b"A85" => (ascii85(input, cap, room), false),
            b"RunLengthDecode" | b"RL" => (run_length(input, cap, room), false),
            other => {
                return Err(Fail::Unsupported(format!(
                    "stream filter {}",
                    String::from_utf8_lossy(other)
                )))
            }
        };
        let mut out = stage.map_err(|e| match e {
            Fail::TooLarge(_) => budget.over(),
            e => e,
        })?;
        budget.take(out.len() as u64)?;
        if predicted {
            out = predict(out, parm, budget)?;
        }
        cur = Some(out);
    }
    Ok(cur.unwrap_or_else(|| data.to_vec()))
}

fn over(room: u64) -> Fail {
    Fail::TooLarge(format!(
        "its compressed streams expand to more than the {room} bytes left of the decompression limit"
    ))
}

fn inflate_fail(e: &str, room: u64) -> Fail {
    if e.contains("expands to more than") {
        over(room)
    } else {
        Fail::damaged(format!("a compressed stream is damaged ({e})"))
    }
}

/// Undoes a PNG predictor (`/Predictor` 10..=15) or passes the data through (1 or absent). The
/// rows it walks are counted from the data (`pdf_predict_rows`), and their work (rows times the
/// row width) must fit in the budget's room and is charged before any of it is done.
fn predict(data: Vec<u8>, parm: Option<&Dict>, budget: &mut Budget) -> Result<Vec<u8>, Fail> {
    let Some(p) = parm else {
        return Ok(data);
    };
    let get = |k: &[u8], d: i64| p.get(k).and_then(Obj::as_int).unwrap_or(d);
    let predictor = get(b"Predictor", 1);
    if predictor <= 1 {
        return Ok(data);
    }
    if predictor == 2 {
        return Err(Fail::Unsupported("the TIFF predictor".into()));
    }
    if !(10..=15).contains(&predictor) {
        return Err(Fail::damaged(format!("unknown predictor {predictor}")));
    }
    let colors = get(b"Colors", 1);
    let bpc = get(b"BitsPerComponent", 8);
    let columns = get(b"Columns", 1);
    if !(1..=32).contains(&colors)
        || !matches!(bpc, 1 | 2 | 4 | 8 | 16)
        || !(1..=1 << 20).contains(&columns)
    {
        return Err(Fail::damaged("predictor parameters out of range"));
    }
    // all three are small, so their product fits easily
    let bits_per_pixel = (colors * bpc) as usize;
    let bpp = bits_per_pixel.div_ceil(8).max(1);
    let row = (columns as usize * bits_per_pixel).div_ceil(8);
    let (len, width) = (data.len() as u64, row as u64);
    if !v::pdf_predict_fits(len, width, budget.room()) {
        return Err(budget.over());
    }
    let rows = v::pdf_predict_rows(len, width);
    // the work fits in the room, so the product does not overflow
    budget.take(v::pdf_row_offset(rows, width))?;
    let mut out = Vec::with_capacity(data.len());
    if rows == 0 {
        return Ok(out);
    }
    let mut prev = vec![0u8; row];
    let mut cur = vec![0u8; row];
    for k in 0..rows {
        let at = v::pdf_row_offset(k, width + 1) as usize;
        let Some(chunk) = data.get(at..data.len().min(at + row + 1)) else {
            break;
        };
        let Some((&kind, body)) = chunk.split_first() else {
            break;
        };
        cur[..body.len()].copy_from_slice(body);
        for b in cur[body.len()..].iter_mut() {
            *b = 0;
        }
        for i in 0..row {
            let left = if i >= bpp { cur[i - bpp] } else { 0 };
            let up = prev[i];
            let up_left = if i >= bpp { prev[i - bpp] } else { 0 };
            cur[i] = match kind {
                0 => cur[i],
                1 => cur[i].wrapping_add(left),
                2 => cur[i].wrapping_add(up),
                3 => cur[i].wrapping_add(((u16::from(left) + u16::from(up)) / 2) as u8),
                4 => cur[i].wrapping_add(paeth(left, up, up_left)),
                _ => return Err(Fail::damaged("unknown PNG predictor row type")),
            };
        }
        out.extend_from_slice(&cur[..body.len()]);
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(out)
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (i16::from(a), i16::from(b), i16::from(c));
    let p = ia + ib - ic;
    let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn push_capped(out: &mut Vec<u8>, b: u8, cap: usize, room: u64) -> Result<(), Fail> {
    if !v::pdf_below(out.len() as u64, cap as u64) {
        return Err(over(room));
    }
    out.push(b);
    Ok(())
}

fn ascii_hex(data: &[u8], cap: usize, room: u64) -> Result<Vec<u8>, Fail> {
    let mut out = Vec::new();
    let mut hi: Option<u8> = None;
    for &b in data {
        if b == b'>' {
            break;
        }
        let v = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ if super::lex::is_white(b) => continue,
            _ => return Err(Fail::damaged("bad ASCIIHex data")),
        };
        match hi.take() {
            Some(h) => push_capped(&mut out, h << 4 | v, cap, room)?,
            None => hi = Some(v),
        }
    }
    if let Some(h) = hi {
        push_capped(&mut out, h << 4, cap, room)?;
    }
    Ok(out)
}

fn ascii85(data: &[u8], cap: usize, room: u64) -> Result<Vec<u8>, Fail> {
    let mut out = Vec::new();
    let mut group = [0u8; 5];
    let mut n = 0usize;
    let body = data.strip_prefix(b"<~").unwrap_or(data);
    for &b in body {
        if b == b'~' {
            break;
        }
        if super::lex::is_white(b) {
            continue;
        }
        if b == b'z' && n == 0 {
            for _ in 0..4 {
                push_capped(&mut out, 0, cap, room)?;
            }
            continue;
        }
        if !(b'!'..=b'u').contains(&b) {
            return Err(Fail::damaged("bad ASCII85 data"));
        }
        group[n] = b - b'!';
        n += 1;
        if n == 5 {
            let v = group.iter().fold(0u64, |a, &d| a * 85 + u64::from(d));
            if v > u64::from(u32::MAX) {
                return Err(Fail::damaged("bad ASCII85 group"));
            }
            for byte in (v as u32).to_be_bytes() {
                push_capped(&mut out, byte, cap, room)?;
            }
            n = 0;
        }
    }
    if n > 1 {
        for g in group.iter_mut().skip(n) {
            *g = 84;
        }
        let v = group.iter().fold(0u64, |a, &d| a * 85 + u64::from(d));
        let bytes = (v.min(u64::from(u32::MAX)) as u32).to_be_bytes();
        for &byte in &bytes[..n - 1] {
            push_capped(&mut out, byte, cap, room)?;
        }
    }
    Ok(out)
}

fn run_length(data: &[u8], cap: usize, room: u64) -> Result<Vec<u8>, Fail> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        let len = data[i];
        i += 1;
        match len {
            128 => break,
            0..=127 => {
                let n = usize::from(len) + 1;
                let Some(run) = data.get(i..i + n) else {
                    return Err(Fail::damaged("a RunLength run ends early"));
                };
                for &b in run {
                    push_capped(&mut out, b, cap, room)?;
                }
                i += n;
            }
            _ => {
                let Some(&b) = data.get(i) else {
                    return Err(Fail::damaged("a RunLength run ends early"));
                };
                for _ in 0..(257 - usize::from(len)) {
                    push_capped(&mut out, b, cap, room)?;
                }
                i += 1;
            }
        }
    }
    Ok(out)
}

/// LZW with variable code widths of 9 to 12 bits (`early`: the width grows one code early, the
/// PDF default).
fn lzw(data: &[u8], cap: usize, early: bool, room: u64) -> Result<Vec<u8>, Fail> {
    const CLEAR: usize = 256;
    const EOD: usize = 257;
    let mut out: Vec<u8> = Vec::new();
    // each entry: (prefix code, last byte, length); entries below 256 are single bytes
    let mut table: Vec<(usize, u8, usize)> = (0..256).map(|b| (usize::MAX, b as u8, 1)).collect();
    table.push((usize::MAX, 0, 0));
    table.push((usize::MAX, 0, 0));
    let mut width = 9u32;
    let mut acc = 0u32;
    let mut nbits = 0u32;
    let mut prev: Option<usize> = None;
    let mut entry = Vec::new();
    let mut bytes = data.iter();
    loop {
        while nbits < width {
            match bytes.next() {
                Some(&b) => {
                    acc = acc << 8 | u32::from(b);
                    nbits += 8;
                }
                None => return Ok(out),
            }
        }
        let code = ((acc >> (nbits - width)) & ((1 << width) - 1)) as usize;
        nbits -= width;
        acc &= (1u32 << nbits).wrapping_sub(1);
        if code == CLEAR {
            table.truncate(258);
            width = 9;
            prev = None;
            continue;
        }
        if code == EOD {
            return Ok(out);
        }
        let first_of = |table: &Vec<(usize, u8, usize)>, mut c: usize| {
            while table[c].0 != usize::MAX {
                c = table[c].0;
            }
            table[c].1
        };
        let known = code < table.len();
        let Some(p) = prev else {
            if !known || code > 255 {
                return Err(Fail::damaged("bad LZW data"));
            }
            push_capped(&mut out, code as u8, cap, room)?;
            prev = Some(code);
            continue;
        };
        let first = if known {
            first_of(&table, code)
        } else if code == table.len() {
            first_of(&table, p)
        } else {
            return Err(Fail::damaged("bad LZW data"));
        };
        if table.len() < 4096 {
            let len = table[p].2 + 1;
            table.push((p, first, len));
        }
        if code >= table.len() {
            return Err(Fail::damaged("bad LZW data"));
        }
        // write the entry's bytes
        entry.clear();
        let mut c = code;
        loop {
            entry.push(table[c].1);
            if table[c].0 == usize::MAX {
                break;
            }
            c = table[c].0;
        }
        for &b in entry.iter().rev() {
            push_capped(&mut out, b, cap, room)?;
        }
        prev = Some(code);
        let limit = if early { table.len() + 1 } else { table.len() };
        if limit >= (1 << width) && width < 12 {
            width += 1;
        }
    }
}
