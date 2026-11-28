//! The model's vocabulary as the grammar sees it: each token id's byte string (what the server's
//! decoder turns it into), its kind, and a byte trie over the text tokens, walked once per
//! automaton state to find the admitted tokens.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// What a token id is to the grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Text: its bytes are fed to the automaton.
    Text,
    /// An end-of-sequence id: admitted only when the reply may end.
    End,
    /// A control token that adds no text (a special token outside the family's tags), or a
    /// token with no bytes: never admitted where the reply is constrained.
    Control,
}

/// A byte trie over a set of token ids.
#[derive(Debug, Default)]
pub struct Trie {
    /// Children of node `i`: `edges[first[i] .. first[i + 1]]`, sorted by byte.
    first: Vec<u32>,
    edges: Vec<(u8, u32)>,
    /// Tokens whose bytes end at node `i`: `toks[tfirst[i] .. tfirst[i + 1]]`.
    tfirst: Vec<u32>,
    toks: Vec<u32>,
}

impl Trie {
    fn build(table: &[Vec<u8>], ids: impl Iterator<Item = u32>) -> Self {
        // a pointer trie first, flattened breadth-first afterwards
        struct N {
            kids: Vec<(u8, usize)>,
            toks: Vec<u32>,
        }
        let mut ns = vec![N {
            kids: Vec::new(),
            toks: Vec::new(),
        }];
        for id in ids {
            let mut cur = 0usize;
            for &b in &table[id as usize] {
                cur = match ns[cur].kids.iter().find(|k| k.0 == b) {
                    Some(k) => k.1,
                    None => {
                        ns.push(N {
                            kids: Vec::new(),
                            toks: Vec::new(),
                        });
                        let n = ns.len() - 1;
                        ns[cur].kids.push((b, n));
                        n
                    }
                };
            }
            ns[cur].toks.push(id);
        }
        let mut order = vec![0usize];
        let mut index = vec![0u32; ns.len()];
        let mut i = 0;
        while i < order.len() {
            let n = order[i];
            ns[n].kids.sort_unstable_by_key(|k| k.0);
            for &(_, c) in &ns[n].kids {
                index[c] = order.len() as u32;
                order.push(c);
            }
            i += 1;
        }
        let mut t = Trie::default();
        for &n in &order {
            t.first.push(t.edges.len() as u32);
            t.tfirst.push(t.toks.len() as u32);
            for &(b, c) in &ns[n].kids {
                t.edges.push((b, index[c]));
            }
            t.toks.extend_from_slice(&ns[n].toks);
        }
        t.first.push(t.edges.len() as u32);
        t.tfirst.push(t.toks.len() as u32);
        t
    }

    pub fn children(&self, n: u32) -> &[(u8, u32)] {
        &self.edges[self.first[n as usize] as usize..self.first[n as usize + 1] as usize]
    }

    pub fn tokens(&self, n: u32) -> &[u32] {
        &self.toks[self.tfirst[n as usize] as usize..self.tfirst[n as usize + 1] as usize]
    }

    pub fn nodes(&self) -> usize {
        self.first.len().saturating_sub(1)
    }
}

/// The text tokens split by a byte set: those holding none of its bytes, and a trie over the
/// rest.
#[derive(Debug)]
pub struct Split {
    pub plain: Vec<u32>,
    pub rest: Trie,
}

/// One model's vocabulary for the grammar; built once per model and shared.
#[derive(Debug)]
pub struct TokenTable {
    bytes: Vec<Vec<u8>>,
    kinds: Vec<Kind>,
    trie: Trie,
    splits: Mutex<HashMap<[u64; 4], Arc<Split>>>,
}

impl TokenTable {
    /// `bytes[id]` and `kinds[id]` for every id the request may emit.
    pub fn new(bytes: Vec<Vec<u8>>, mut kinds: Vec<Kind>) -> Self {
        for (k, b) in kinds.iter_mut().zip(&bytes) {
            if *k == Kind::Text && b.is_empty() {
                *k = Kind::Control;
            }
        }
        let trie = Trie::build(
            &bytes,
            (0..bytes.len() as u32).filter(|&i| kinds[i as usize] == Kind::Text),
        );
        Self {
            bytes,
            kinds,
            trie,
            splits: Mutex::new(HashMap::new()),
        }
    }

    /// The table of a tokenizer: a special token is text only when `is_marker` says it is part
    /// of the family's tags (what the server's decoder does with it); `end_ids` end the sequence.
    pub fn from_tokenizer(
        tok: &crate::tokenizer::Tokenizer,
        vocab: u64,
        end_ids: &[i32],
        is_marker: impl Fn(&str) -> bool,
    ) -> Self {
        let mut bytes = Vec::with_capacity(vocab as usize);
        let mut kinds = Vec::with_capacity(vocab as usize);
        for id in 0..vocab as i32 {
            let b = tok.token_bytes(id).to_vec();
            let kind = if end_ids.contains(&id) {
                Kind::End
            } else if tok.is_special(id) && !is_marker(&String::from_utf8_lossy(&b)) {
                Kind::Control
            } else {
                Kind::Text
            };
            bytes.push(b);
            kinds.push(kind);
        }
        Self::new(bytes, kinds)
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn kind(&self, id: u32) -> Kind {
        self.kinds.get(id as usize).copied().unwrap_or(Kind::Control)
    }

    pub fn bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    pub fn trie(&self) -> &Trie {
        &self.trie
    }

    pub fn ids_of(&self, kind: Kind) -> impl Iterator<Item = u32> + '_ {
        (0..self.kinds.len() as u32).filter(move |&i| self.kinds[i as usize] == kind)
    }

    /// The text tokens split by the byte set `set` (cached per set).
    pub fn split(&self, set: [u64; 4]) -> Arc<Split> {
        let mut m = self.splits.lock().unwrap();
        if let Some(s) = m.get(&set) {
            return Arc::clone(s);
        }
        let has = |id: u32| {
            self.bytes[id as usize]
                .iter()
                .any(|&b| set[(b >> 6) as usize] & (1 << (b & 63)) != 0)
        };
        let text = || self.ids_of(Kind::Text);
        let plain: Vec<u32> = text().filter(|&i| !has(i)).collect();
        let rest = Trie::build(&self.bytes, text().filter(|&i| has(i)));
        let s = Arc::new(Split { plain, rest });
        m.insert(set, Arc::clone(&s));
        s
    }
}
