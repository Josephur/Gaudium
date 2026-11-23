//! The tokenizer of a GGUF file's metadata: the same tables `tokenizer.json` gives, read straight
//! from the file's mapping (each token text is copied once into each table that keeps it; no
//! intermediate list or JSON text is made).
//!
//! Format facts (GGUF specification, read for facts only):
//! - `tokenizer.ggml.model`: the tokenizer model; `gpt2` is byte-level BPE (tokens are written in
//!   the byte-to-character alphabet, merges as `"left right"` strings in rank order);
//! - `tokenizer.ggml.pre`: the name of the pre-tokenizer the converter recognized from the
//!   original tokenizer: `inkling2` (the letter-case pattern, with whole-word vocabulary lookups
//!   before merging), `qwen2` (the Qwen2 pattern) and `qwen35` (the Qwen3.5 pattern, letters with
//!   their marks);
//! - `tokenizer.ggml.token_type`, one per token: 1 normal, 2 unknown, 3 control, 4 user defined,
//!   5 unused, 6 byte. Control tokens are the original's special added tokens, user-defined ones
//!   its other added tokens; their text is stored as written, not in the byte alphabet.
//!
//! A model or pre-tokenizer name not listed here, or a token type a byte-level vocabulary does
//! not use, is refused: the split of a text would otherwise be a guess.

use super::*;
use crate::gguf::Gguf;
use crate::verified as v;
use crate::verified::MtTokKind;

/// Largest vocabulary accepted from a file (ids are 32-bit; real vocabularies are far smaller).
pub const MAX_GGUF_VOCAB: u64 = 1 << 24;

/// The pre-tokenizer and whole-word lookup of a `tokenizer.ggml.pre` name.
fn pretok_of(pre: &str) -> Option<(Pretok, bool)> {
    match pre {
        "inkling2" => Some((Pretok::LetterCase, true)),
        "qwen2" => Some((
            Pretok::Qwen {
                include_marks: false,
            },
            false,
        )),
        "qwen35" => Some((
            Pretok::Qwen {
                include_marks: true,
            },
            false,
        )),
        _ => None,
    }
}

/// The kind of a GGUF token type (`None`: a type a byte-level vocabulary does not use).
fn kind_of(token_type: i64) -> Option<MtTokKind> {
    match token_type {
        1 => Some(MtTokKind::MtTokNormal),
        3 => Some(MtTokKind::MtTokControl),
        4 => Some(MtTokKind::MtTokUser),
        5 => Some(MtTokKind::MtTokUnused),
        _ => None,
    }
}

fn fmt_err(msg: impl Into<String>) -> TokenizerError {
    TokenizerError::Format(msg.into())
}

impl Tokenizer {
    /// Builds the tokenizer a GGUF file's `tokenizer.ggml.*` metadata describes.
    pub fn from_gguf(g: &Gguf) -> Result<Self, TokenizerError> {
        let model = g
            .get_str("tokenizer.ggml.model")
            .ok_or_else(|| fmt_err("missing tokenizer.ggml.model"))?;
        if model != "gpt2" {
            return Err(fmt_err(format!(
                "unsupported tokenizer.ggml.model '{model}' (only byte-level BPE, 'gpt2', is read)"
            )));
        }
        let pre = g.get_str("tokenizer.ggml.pre").unwrap_or("");
        let (pretok, ignore_merges) = pretok_of(pre).ok_or_else(|| {
            fmt_err(format!(
                "unsupported tokenizer.ggml.pre '{pre}' (known: inkling2, qwen2, qwen35)"
            ))
        })?;
        let tokens = g
            .kv("tokenizer.ggml.tokens")
            .and_then(|x| x.as_array())
            .filter(|a| a.elem_type() == crate::gguf::ValueType::Str)
            .ok_or_else(|| fmt_err("missing tokenizer.ggml.tokens (a list of strings)"))?;
        let types = g
            .kv("tokenizer.ggml.token_type")
            .and_then(|x| x.as_array())
            .ok_or_else(|| fmt_err("missing tokenizer.ggml.token_type"))?;
        if !v::mt_vocab_ok(tokens.len(), types.len(), MAX_GGUF_VOCAB) {
            return Err(fmt_err(format!(
                "{} tokens with {} token types (one per token, 1 to {MAX_GGUF_VOCAB} tokens)",
                tokens.len(),
                types.len()
            )));
        }
        let merges = g
            .kv("tokenizer.ggml.merges")
            .and_then(|x| x.as_array())
            .ok_or_else(|| fmt_err("missing tokenizer.ggml.merges"))?;

        let n_ids = tokens.len() as usize;
        let mut vocab: HashMap<String, i32> = HashMap::with_capacity(n_ids);
        let mut id_bytes = vec![Vec::new(); n_ids];
        let mut id_piece = vec![String::new(); n_ids];
        let mut id_flags = vec![0u8; n_ids];
        let mut byte_id = [-1i32; 256];
        let mut added = Vec::new();
        let mut added_first_byte = [false; 256];
        let (b2cp, cp2b) = byte_level_tables();

        for (k, raw) in tokens.byte_strings().enumerate() {
            let id = k as i32;
            let ty = types.get(k as u64).and_then(|x| x.as_i64()).unwrap_or(-1);
            let kind = kind_of(ty).ok_or_else(|| {
                fmt_err(format!(
                    "token {k} has type {ty}, which a byte-level vocabulary does not use"
                ))
            })?;
            let piece =
                std::str::from_utf8(raw).map_err(|_| fmt_err(format!("token {k} is not UTF-8")))?;
            if v::mt_tok_added(kind) {
                id_bytes[k] = raw.to_vec();
                id_flags[k] |= FLAG_ADDED;
                let special = v::mt_tok_special(kind);
                if special {
                    id_flags[k] |= FLAG_SPECIAL;
                }
                if let Some(&first) = raw.first() {
                    added_first_byte[first as usize] = true;
                }
                added.push(AddedToken {
                    id,
                    content: piece.to_string(),
                    special,
                });
            } else {
                let mut bytes = Vec::with_capacity(piece.len());
                for ch in piece.chars() {
                    match cp2b.get(&(ch as u32)) {
                        Some(&b) => bytes.push(b),
                        None => {
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
                if v::mt_tok_encodable(kind) && bytes.len() == 1 && piece.chars().count() == 1 {
                    byte_id[bytes[0] as usize] = id;
                }
                id_bytes[k] = bytes;
            }
            if v::mt_tok_encodable(kind) {
                vocab.insert(piece.to_string(), id);
            }
            id_piece[k] = piece.to_string();
        }
        // longest first, so the leftmost-longest added token wins
        added.sort_by_key(|a| std::cmp::Reverse(a.content.len()));

        let mut merge_map = HashMap::with_capacity(merges.len() as usize);
        let mut joined = String::new();
        for (rank, entry) in merges.strings().enumerate() {
            let Some(s) = entry else { continue };
            let Some(sp) = s.find(' ') else { continue };
            let (left, right) = (&s[..sp], &s[sp + 1..]);
            let (Some(&a), Some(&b)) = (vocab.get(left), vocab.get(right)) else {
                continue;
            };
            joined.clear();
            joined.push_str(left);
            joined.push_str(right);
            if let Some(&ab) = vocab.get(&joined) {
                merge_map.insert(pair_key(a, b), (rank as i32, ab));
            }
        }

        for (b, &id) in byte_id.iter().enumerate() {
            if id < 0 {
                return Err(fmt_err(format!(
                    "byte-level vocabulary has no single-byte token for 0x{b:02X}"
                )));
            }
        }
        let mut byte_char = ['\0'; 256];
        for (slot, &cp) in byte_char.iter_mut().zip(b2cp.iter()) {
            *slot = char::from_u32(cp).unwrap_or('\0');
        }
        Ok(Self {
            byte_fallback: false,
            pretok,
            ignore_merges,
            byte_char,
            vocab,
            id_bytes,
            id_piece,
            id_flags,
            merges: merge_map,
            byte_id,
            added,
            added_first_byte,
            piece_cache: Mutex::new(HashMap::new()),
        })
    }
}

/// Test support: a minimal GGUF metadata writer (no tensors) and a tiny byte-level vocabulary.
#[cfg(test)]
pub(crate) mod test_gguf {
    use std::path::Path;

    /// A metadata value.
    pub enum Kv {
        U32(u32),
        I32(i32),
        F32(f32),
        Str(String),
        StrArr(Vec<String>),
        I32Arr(Vec<i32>),
    }

    fn put_str(out: &mut Vec<u8>, s: &[u8]) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s);
    }

    /// Writes a GGUF v3 file holding `kvs` and no tensors.
    pub fn write_gguf(path: &Path, kvs: &[(String, Kv)]) {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, val) in kvs {
            put_str(&mut out, k.as_bytes());
            let (t, body): (u32, Vec<u8>) = match val {
                Kv::U32(x) => (4, x.to_le_bytes().to_vec()),
                Kv::I32(x) => (5, x.to_le_bytes().to_vec()),
                Kv::F32(x) => (6, x.to_le_bytes().to_vec()),
                Kv::Str(s) => {
                    let mut b = Vec::new();
                    put_str(&mut b, s.as_bytes());
                    (8, b)
                }
                Kv::StrArr(items) => {
                    let mut b = 8u32.to_le_bytes().to_vec();
                    b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for s in items {
                        put_str(&mut b, s.as_bytes());
                    }
                    (9, b)
                }
                Kv::I32Arr(items) => {
                    let mut b = 5u32.to_le_bytes().to_vec();
                    b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for x in items {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                    (9, b)
                }
            };
            out.extend_from_slice(&t.to_le_bytes());
            out.extend_from_slice(&body);
        }
        std::fs::write(path, out).unwrap();
    }

    /// The 256 byte tokens (ids 0..=255, in the byte alphabet), then `extra` normal tokens, then
    /// the `control` tokens: the `tokens` and `token_type` arrays.
    pub fn tiny_vocab(extra: &[&str], control: &[&str]) -> (Vec<String>, Vec<i32>) {
        let (b2cp, _) = super::byte_level_tables();
        let mut toks: Vec<String> = b2cp
            .iter()
            .map(|&cp| char::from_u32(cp).unwrap().to_string())
            .collect();
        toks.extend(extra.iter().map(|s| s.to_string()));
        let mut types = vec![1i32; toks.len()];
        toks.extend(control.iter().map(|s| s.to_string()));
        types.resize(toks.len(), 3);
        (toks, types)
    }
}

#[cfg(test)]
mod tests {
    use super::test_gguf::{tiny_vocab, write_gguf, Kv};
    use super::*;
    use std::path::PathBuf;

    fn file(tag: &str, kvs: &[(String, Kv)]) -> (PathBuf, Gguf) {
        let p =
            std::env::temp_dir().join(format!("gaudium-tokgguf-{tag}-{}.gguf", std::process::id()));
        write_gguf(&p, kvs);
        let g = Gguf::open(&p).unwrap();
        (p, g)
    }

    fn vocab_kvs(
        pre: &str,
        extra: &[&str],
        merges: &[&str],
        types: Option<Vec<i32>>,
    ) -> Vec<(String, Kv)> {
        let (toks, t) = tiny_vocab(extra, &["<|ctl|>"]);
        vec![
            ("tokenizer.ggml.model".to_string(), Kv::Str("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Kv::Str(pre.into())),
            ("tokenizer.ggml.tokens".into(), Kv::StrArr(toks)),
            (
                "tokenizer.ggml.token_type".into(),
                Kv::I32Arr(types.unwrap_or(t)),
            ),
            (
                "tokenizer.ggml.merges".into(),
                Kv::StrArr(merges.iter().map(|s| s.to_string()).collect()),
            ),
        ]
    }

    /// The tokenizer.json of the same vocabulary.
    fn json_of(extra: &[&str], merges: &[&str], pattern: &str, ignore_merges: bool) -> Tokenizer {
        let (toks, _) = tiny_vocab(extra, &[]);
        let vocab: Vec<String> = toks
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{}: {i}", crate::json::quote_json(t)))
            .collect();
        let merges: Vec<String> = merges.iter().map(|m| crate::json::quote_json(m)).collect();
        let json = format!(
            r#"{{"model": {{"type": "BPE", "ignore_merges": {ignore_merges}, "vocab": {{{}}}, "merges": [{}]}},
                "added_tokens": [{{"id": {}, "content": "<|ctl|>", "special": true}}],
                "pre_tokenizer": {{"type": "Sequence", "pretokenizers": [{{"type": "Split", "pattern": {{"Regex": {}}}}}]}}}}"#,
            vocab.join(", "),
            merges.join(", "),
            toks.len(),
            crate::json::quote_json(pattern)
        );
        Tokenizer::from_json_str(&json).unwrap()
    }

    #[test]
    fn gguf_and_json_forms_of_one_vocabulary_agree() {
        // "ab" and "abc" by merges; with whole-word lookups "abc" is one token without them
        let extra = ["ab", "abc", "\u{120}x"];
        let merges = ["a b", "ab c", "\u{120} x"];
        let texts = ["abc ab xabc", "<|ctl|>abc<|ctl|> x", "Abc\nabcABC 123"];
        for (pre, pattern, ignore) in [
            ("inkling2", LETTER_CASE_PATTERN, true),
            (
                "qwen2",
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                false,
            ),
        ] {
            let (p, g) = file(pre, &vocab_kvs(pre, &extra, &merges, None));
            let tg = Tokenizer::from_gguf(&g).unwrap();
            let tj = json_of(&extra, &merges, pattern, ignore);
            assert_eq!((tg.pretok, tg.ignore_merges), (tj.pretok, tj.ignore_merges), "{pre}");
            assert_eq!(tg.vocab_size(), tj.vocab_size());
            for t in texts {
                assert_eq!(tg.encode(t, true), tj.encode(t, true), "{pre} {t:?}");
                let ids = tg.encode(t, true);
                assert_eq!(tg.decode(&ids, false), t);
                assert_eq!(tg.decode(&ids, true), tj.decode(&ids, true));
            }
            let ctl = tg.token_id("<|ctl|>").unwrap();
            assert!(tg.is_special(ctl) && tg.is_added(ctl));
            assert!(!tg.is_added(256));
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn unused_entries_decode_but_never_encode() {
        let (toks, mut types) = tiny_vocab(&["ab"], &[]);
        types[256] = 5;
        let kvs = vec![
            ("tokenizer.ggml.model".to_string(), Kv::Str("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Kv::Str("qwen2".into())),
            ("tokenizer.ggml.tokens".into(), Kv::StrArr(toks)),
            ("tokenizer.ggml.token_type".into(), Kv::I32Arr(types)),
            (
                "tokenizer.ggml.merges".into(),
                Kv::StrArr(vec!["a b".into()]),
            ),
        ];
        let (p, g) = file("unused", &kvs);
        let t = Tokenizer::from_gguf(&g).unwrap();
        assert_eq!(t.encode("ab", false), vec![b'a' as i32, b'b' as i32]);
        assert_eq!(t.decode(&[256], false), "ab");
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn unknown_names_and_types_are_refused() {
        let check = |tag: &str, kvs: Vec<(String, Kv)>, want: &str| {
            let (p, g) = file(tag, &kvs);
            let e = Tokenizer::from_gguf(&g).err().expect("refused").to_string();
            assert!(e.contains(want), "{tag}: {e}");
            let _ = std::fs::remove_file(p);
        };
        let mut kvs = vocab_kvs("qwen2", &[], &[], None);
        kvs[0].1 = Kv::Str("unigram".into());
        check("model", kvs, "tokenizer.ggml.model 'unigram'");
        check(
            "pre",
            vocab_kvs("gpt-9", &[], &[], None),
            "tokenizer.ggml.pre 'gpt-9'",
        );
        let (_, mut types) = tiny_vocab(&[], &["<|ctl|>"]);
        types[3] = 6;
        check(
            "type",
            vocab_kvs("qwen2", &[], &[], Some(types)),
            "token 3 has type 6",
        );
        check(
            "count",
            vocab_kvs("qwen2", &[], &[], Some(vec![1; 10])),
            "257 tokens with 10 token types",
        );
        // every byte needs its own token
        let (mut toks, types) = tiny_vocab(&[], &["<|ctl|>"]);
        toks[0] = "zz".into();
        let mut kvs = vocab_kvs("qwen2", &[], &[], Some(types));
        kvs[2].1 = Kv::StrArr(toks);
        check("bytes", kvs, "0x00");
    }

    /// Set `GAUDIUM_INKLING2_GGUF` to an Inkling 2 GGUF file and `GAUDIUM_INKLING2_DIR` to the
    /// publisher's checkpoint directory of the same model (its `tokenizer.json`).
    #[test]
    #[ignore = "needs GAUDIUM_INKLING2_GGUF and GAUDIUM_INKLING2_DIR"]
    fn inkling2_gguf_tokenizer_matches_the_checkpoint() {
        use crate::json::{parse_json, JsonValue};
        let gguf = std::env::var("GAUDIUM_INKLING2_GGUF").expect("GAUDIUM_INKLING2_GGUF");
        let dir = std::env::var("GAUDIUM_INKLING2_DIR").expect("GAUDIUM_INKLING2_DIR");
        let t0 = std::time::Instant::now();
        let g = Gguf::open(&gguf).unwrap();
        let t1 = std::time::Instant::now();
        let tg = Tokenizer::from_gguf(&g).unwrap();
        let t2 = std::time::Instant::now();
        let tj = Tokenizer::from_model_dir(std::path::Path::new(&dir)).unwrap();
        let t3 = std::time::Instant::now();
        eprintln!(
            "open {:?}, tokenizer from GGUF {:?}, from tokenizer.json {:?}",
            t1 - t0,
            t2 - t1,
            t3 - t2
        );
        assert!(tg.ignore_merges && tg.pretok == Pretok::LetterCase);
        assert_eq!(tg.vocab_size(), tj.vocab_size());
        assert_eq!(tg.merges.len(), tj.merges.len());
        assert_eq!(tg.merges, tj.merges);
        assert_eq!(tg.byte_id, tj.byte_id);
        for id in 0..tg.vocab_size() as i32 {
            assert_eq!(tg.token_bytes(id), tj.token_bytes(id), "{id}");
            assert_eq!(tg.token_piece(id), tj.token_piece(id), "{id}");
            assert_eq!(tg.is_special(id), tj.is_special(id), "{id}");
            assert_eq!(tg.is_added(id), tj.is_added(id), "{id}");
        }
        let cases = parse_json(include_str!(
            "../../tests/data/tokenizer/inkling2_cases.json"
        ))
        .unwrap();
        let ids_of = |v: &JsonValue| -> Vec<i32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect()
        };
        for case in cases.get("encode").unwrap().as_array().unwrap() {
            let text = case.get("text").unwrap().as_str().unwrap();
            assert_eq!(
                tg.encode(text, true),
                ids_of(case.get("ids").unwrap()),
                "{text:?}"
            );
        }
        for case in cases.get("decode").unwrap().as_array().unwrap() {
            let ids = ids_of(case.get("ids").unwrap());
            let skip = case.get("skip_special").unwrap().as_bool().unwrap();
            let want = case.get("text").unwrap().as_str().unwrap();
            assert_eq!(tg.decode(&ids, skip), want, "{ids:?}");
        }
    }
}
