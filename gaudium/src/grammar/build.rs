//! Compiling a reply grammar: the JSON Schema subset, and the reply forms built on it.
//!
//! Supported schema keywords: `type` (a name or a list), `properties`, `required`,
//! `additionalProperties` (`false`; `true` or absent beside `properties`, which then holds the
//! reply to the listed properties; a schema for an object without `properties`), `items`,
//! `minItems`, `maxItems`, `enum`, `const`, `anyOf`, `allOf` with one schema, `$ref` to `#`,
//! `#/$defs/...` or `#/definitions/...` (recursion included), the boolean schema `true`, and
//! annotations (`title`, `description`, `default`, `examples`, ...). Any other keyword is refused
//! with its name and place. The reply follows the schema's property order; optional properties
//! may be left out.
//!
//! A schema is a client's input: everything it makes the compiler build is charged to one
//! budget of [`MAX_GRAMMAR_BYTES`] (the verified `gram_budget_ok`), and a schema past it, or one
//! whose alternatives could not all be followed (`gram_alive_ok`), is refused before any token.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::{Grammar, Node, NodeId, ObjDef, Pat, PropDef, TrieNode, FINISH, MAX_ALIVE, NO_LEAF};
use crate::json::{quote_json, JsonValue};
use crate::verified;

/// Schema nesting levels (objects and arrays inside each other) a request may declare.
pub const MAX_SCHEMA_DEPTH: u64 = 32;
/// Schemas the compiler may be inside at once (`anyOf`, `allOf`, `$ref`, containers): bounds
/// its recursion, so a chain of references can not exhaust the thread's stack.
pub const MAX_SCHEMA_LEVELS: u64 = 128;
/// Bytes of grammar a request may compile to: nodes, trie nodes, literal bytes and the key
/// tables of objects.
pub const MAX_GRAMMAR_BYTES: u64 = 32 << 20;
/// Bytes of a schema path, and of a name from the schema, kept for an error message (a client's
/// keys may be megabytes long, and a path grows with each level).
pub const PATH_MAX: usize = 200;
/// Bytes of an error message a schema's refusal carries.
pub const MESSAGE_MAX: usize = 1000;
/// The parameter-block form's tags: a value ends at the first of them.
pub const PARAM_CLOSE: &str = "</parameter>";
pub const FUNCTION_CLOSE: &str = "</function>";

/// Keywords that annotate and constrain nothing.
const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "$schema",
    "$id",
    "$comment",
    "$anchor",
    "readOnly",
    "writeOnly",
    "deprecated",
    "$defs",
    "definitions",
];

/// How a model writes a tool call between its tool tags (`outparse::CallStyle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallForm {
    /// `<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n`.
    Params,
    /// `{"name": NAME, "arguments": {...}}`.
    Json,
}

/// A tool the reply may call.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    /// `parameters` of a strict tool (the call's arguments follow it); `None`: free arguments.
    pub strict: Option<JsonValue>,
}

/// What the answer is held to.
#[derive(Debug, Clone)]
pub enum Answer {
    /// `json_object`: any JSON object.
    JsonObject,
    /// `json_schema`: a value of this schema.
    Schema(JsonValue),
}

/// The tool tags and call form of the model.
#[derive(Debug, Clone)]
pub struct CallTags {
    pub open: String,
    pub close: String,
    pub form: CallForm,
    /// The reasoning tags the reply may open with, when the prompt did not open the reasoning
    /// itself (a model that writes its opening tag): `(open, close)`. The reasoning between them
    /// is free text; the reply can not end inside it.
    pub think: Option<(String, String)>,
}

/// A reply's grammar: an answer form, the tools it may call, or both.
pub fn reply(answer: Option<&Answer>, tools: &[ToolDef], tags: &CallTags) -> Result<Grammar, String> {
    let mut b = Builder::new();
    let calls = (!tools.is_empty())
        .then(|| b.call_body(tools, tags))
        .transpose()?;
    let root = match (answer, calls) {
        (Some(a), calls) => {
            // beside tools, the reply parser splits the answer at the tool tag's opening, even
            // inside a JSON string: the answer's strings may not hold it
            b.forbid = if calls.is_some() {
                forbid_list(&[tags.open.as_str()])?
            } else {
                Vec::new()
            };
            let doc = b.answer_doc(a)?;
            b.forbid.clear();
            match calls {
                Some(body) => {
                    // the answer, or one or more calls
                    let open = b.push(Node::Lit(tags.open.as_bytes().to_vec()));
                    let block = b.push(Node::Seq(vec![open, body]));
                    let ws = b.push(Node::Ws);
                    let more = b.push(Node::Seq(vec![ws, block]));
                    let rep = b.push(Node::Rep(more));
                    let ws2 = b.push(Node::Ws);
                    let ws3 = b.push(Node::Ws);
                    let calls = b.push(Node::Seq(vec![ws2, block, rep, ws3]));
                    b.push(Node::Alt(vec![doc, calls]))
                }
                None => doc,
            }
        }
        (None, Some(body)) => b.push(Node::Text {
            open: Pat::new(tags.open.as_bytes()),
            call: body,
        }),
        (None, None) => return Err("internal: a reply grammar with nothing to hold".into()),
    };
    let root = match &tags.think {
        Some((open, close)) if !open.is_empty() && !close.is_empty() => {
            let o = b.lit(open);
            let free = b.push(Node::Raw {
                term: Pat::new(close.as_bytes()),
                forbid: Vec::new(),
            });
            let thought = b.seq(vec![o, free, root]);
            b.push(Node::Alt(vec![thought, root]))
        }
        _ => root,
    };
    b.finish(root)
}

/// The grammar of one JSON document of `schema` (tests and tools).
pub fn schema_doc(schema: &JsonValue) -> Result<Grammar, String> {
    schema_doc_within(schema, MAX_GRAMMAR_BYTES)
}

/// [`schema_doc`] with a compile budget of `cap` bytes.
pub fn schema_doc_within(schema: &JsonValue, cap: u64) -> Result<Grammar, String> {
    let mut b = Builder::new();
    b.cap = cap.min(u64::MAX - 1);
    let root = b.answer_doc(&Answer::Schema(schema.clone()))?;
    b.finish(root)
}

/// The bytes a grammar was charged to the compile budget.
pub fn grammar_bytes(schema: &JsonValue) -> Result<u64, String> {
    let mut b = Builder::new();
    let root = b.answer_doc(&Answer::Schema(schema.clone()))?;
    let used = b.used;
    b.finish(root)?;
    Ok(used)
}

/// Patterns a JSON string may not hold. Each must be non-empty, shorter than 256 bytes, and
/// hold no quote or backslash and some byte that JSON never writes outside a string: then every
/// occurrence in a JSON text lies inside one string's raw bytes, where `Node::Str` looks for it.
fn forbid_list(tags: &[&str]) -> Result<Vec<Pat>, String> {
    let outside = |c: u8| b"{}[],: \t\n\r0123456789-+.eEtrufalsn".contains(&c);
    let mut out: Vec<Pat> = Vec::new();
    for t in tags {
        let b = t.as_bytes();
        let ok = !b.is_empty()
            && b.len() < 256
            && !b.iter().any(|&c| c == b'"' || c == b'\\')
            && b.iter().any(|&c| !outside(c));
        if !ok {
            return Err(format!(
                "this model's tag '{t}' can not be kept out of the JSON strings of its replies"
            ));
        }
        if !out.iter().any(|p| p.bytes == b) {
            out.push(Pat::new(b));
        }
    }
    debug_assert!(out.len() <= 3);
    Ok(out)
}

/// The heap and inline bytes of a node, for the compile budget.
fn node_bytes(n: &Node) -> u64 {
    let heap = match n {
        Node::Lit(b) => b.len(),
        Node::Seq(c) | Node::Alt(c) => 4 * c.len(),
        Node::Raw { term, forbid } => 5 * (term.bytes.len() + forbid.iter().map(|p| p.bytes.len()).sum::<usize>()),
        Node::Str { forbid } => 5 * forbid.iter().map(|p| p.bytes.len()).sum::<usize>(),
        Node::Text { open, .. } => 5 * open.bytes.len(),
        _ => 0,
    };
    (std::mem::size_of::<Node>() + heap) as u64
}

struct Builder {
    nodes: Vec<Node>,
    tries: Vec<TrieNode>,
    objs: Vec<ObjDef>,
    /// Bytes charged to the compile budget (below `u64::MAX`), changed only through the verified
    /// `gram_budget_charge`. A refused charge leaves `used` past `cap`, where every further
    /// charge is refused (`gram_budget_refused_latches`): the builder then adds nothing more,
    /// returns node 0, and `finish` refuses the grammar.
    used: u64,
    cap: u64,
    /// Schemas the compiler is inside now.
    levels: u64,
    /// What the strings compiled now may not hold, and the string and any-value nodes compiled
    /// for each such list.
    forbid: Vec<Pat>,
    strs: HashMap<Vec<Vec<u8>>, NodeId>,
    anys: HashMap<Vec<Vec<u8>>, NodeId>,
    /// The schema `$ref`s resolve in, and the node of every pointer compiled so far.
    root_schema: Option<Rc<JsonValue>>,
    refs: HashMap<String, NodeId>,
}

fn kw_err(path: &str, kw: &str, why: &str) -> String {
    message(&format!("response schema: '{}' at {} {why}", short(kw), short(path))).into_owned()
}

/// `s` cut to `max` bytes (on a character boundary), `...` marking the cut.
fn cut(s: &str, max: usize) -> std::borrow::Cow<'_, str> {
    if s.len() <= max {
        return s.into();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end]).into()
}

/// A name or a path from a schema, cut to [`PATH_MAX`] bytes for a message.
pub fn short(s: &str) -> std::borrow::Cow<'_, str> {
    cut(s, PATH_MAX)
}

/// An error message cut to [`MESSAGE_MAX`] bytes (what a refusal's 400 carries).
pub fn message(s: &str) -> std::borrow::Cow<'_, str> {
    cut(s, MESSAGE_MAX)
}

/// `path` with `parts` appended, for error messages: it stops growing at [`PATH_MAX`] bytes, so
/// a deep schema with long keys does not build a long path at every level.
fn sub_path(path: &str, parts: &[&str]) -> String {
    if path.len() >= PATH_MAX {
        return path.to_string();
    }
    let mut p = String::with_capacity(2 * PATH_MAX + 8);
    p.push_str(path);
    for part in parts {
        if p.len() > PATH_MAX {
            break;
        }
        // at most PATH_MAX + 1 bytes of the part: enough for `short` to mark the cut
        let mut end = part.len().min(PATH_MAX + 1);
        while !part.is_char_boundary(end) {
            end -= 1;
        }
        p.push_str(&part[..end]);
    }
    short(&p).into_owned()
}

fn budget_err() -> String {
    format!(
        "response schema: compiles to more than the {} MiB of grammar a request may use",
        MAX_GRAMMAR_BYTES >> 20
    )
}
// (a test's smaller cap reports the same message)

impl Builder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            tries: Vec::new(),
            objs: Vec::new(),
            used: 0,
            cap: MAX_GRAMMAR_BYTES,
            levels: 0,
            forbid: Vec::new(),
            strs: HashMap::new(),
            anys: HashMap::new(),
            root_schema: None,
            refs: HashMap::new(),
        }
    }

    /// Charges `bytes` to the compile budget; `false` (and over from then on) past it.
    fn charge(&mut self, bytes: u64) -> bool {
        let ok = verified::gram_budget_ok(self.used, bytes, self.cap);
        self.used = verified::gram_budget_charge(self.used, bytes, self.cap);
        ok
    }

    /// A charge was refused.
    fn over(&self) -> bool {
        !verified::gram_budget_ok(self.used, 0, self.cap)
    }

    fn push(&mut self, n: Node) -> NodeId {
        if !self.charge(node_bytes(&n)) {
            return 0;
        }
        self.nodes.push(n);
        (self.nodes.len() - 1) as NodeId
    }

    fn finish(mut self, root: NodeId) -> Result<Grammar, String> {
        if self.over() || self.nodes.is_empty() {
            return Err(budget_err());
        }
        self.check_alternatives()?;
        self.nodes.shrink_to_fit();
        self.tries.shrink_to_fit();
        Ok(Grammar {
            nodes: self.nodes,
            tries: self.tries,
            objs: self.objs,
            root,
        })
    }

    fn check_size(&self) -> Result<(), String> {
        if self.over() {
            return Err(budget_err());
        }
        Ok(())
    }

    /// Every set of alternatives one step enters at once (an `anyOf`, with the `anyOf`s and
    /// references its branches lead to without a byte in between) is one the automaton can
    /// follow (`gram_alive_ok` with `MAX_ALIVE`), and no reference leads back to itself without
    /// a byte in between (that loop would be entered without end).
    fn check_alternatives(&self) -> Result<(), String> {
        let n = self.nodes.len();
        // 0: not seen, 1: on the path, 2: done; width saturates past the cap
        let mut state = vec![0u8; n];
        let mut width = vec![0u64; n];
        let cap = MAX_ALIVE + 1;
        for start in 0..n {
            if state[start] != 0 || !matches!(self.nodes[start], Node::Alt(_)) {
                continue;
            }
            let mut path: Vec<(usize, usize)> = vec![(start, 0)];
            state[start] = 1;
            while let Some(&mut (id, ref mut next)) = path.last_mut() {
                let Node::Alt(bs) = &self.nodes[id] else {
                    unreachable!("only alternatives are walked")
                };
                if let Some(&b) = bs.get(*next) {
                    *next += 1;
                    let b = b as usize;
                    if matches!(self.nodes[b], Node::Alt(_)) {
                        match state[b] {
                            0 => {
                                state[b] = 1;
                                path.push((b, 0));
                            }
                            1 => {
                                return Err("response schema: a reference leads back to itself without a JSON value in between".into())
                            }
                            _ => {}
                        }
                    }
                    continue;
                }
                let wd = bs
                    .iter()
                    .map(|&b| match self.nodes[b as usize] {
                        Node::Alt(_) => width[b as usize],
                        _ => 1,
                    })
                    .fold(0u64, |a, x| a.saturating_add(x).min(cap));
                if !verified::gram_alive_ok(wd, MAX_ALIVE) {
                    return Err(format!(
                        "response schema: an anyOf (with the anyOfs and references its branches lead to) offers more than the {MAX_ALIVE} alternatives a reply can follow at once"
                    ));
                }
                width[id] = wd;
                state[id] = 2;
                path.pop();
            }
        }
        Ok(())
    }

    /// A new trie node, charged to the budget (`None` past it).
    fn trie_node(&mut self) -> Option<u32> {
        if !self.charge(std::mem::size_of::<TrieNode>() as u64 + 8) {
            return None;
        }
        self.tries.push(TrieNode {
            children: Vec::new(),
            leaf: NO_LEAF,
        });
        Some((self.tries.len() - 1) as u32)
    }

    /// The trie node under `cur` for byte `c`, made when missing.
    fn trie_child(&mut self, cur: u32, c: u8) -> Option<u32> {
        if let Some(n) = self.tries[cur as usize].child(c) {
            return Some(n);
        }
        let n = self.trie_node()?;
        let kids = &mut self.tries[cur as usize].children;
        let at = kids.partition_point(|e| e.0 < c);
        kids.insert(at, (c, n));
        Some(n)
    }

    /// A trie over `lits`, each ending with its leaf value.
    fn trie(&mut self, lits: &[(Vec<u8>, u32)]) -> u32 {
        let Some(root) = self.trie_node() else {
            return 0;
        };
        for (bytes, leaf) in lits {
            let mut cur = root;
            for &c in bytes {
                let Some(n) = self.trie_child(cur, c) else {
                    return root;
                };
                cur = n;
            }
            // a literal listed twice keeps its first leaf
            if self.tries[cur as usize].leaf == NO_LEAF {
                self.tries[cur as usize].leaf = *leaf;
            }
        }
        root
    }

    /// `bytes` hold none of the strings' forbidden patterns.
    fn clean(&self, bytes: &[u8]) -> bool {
        !self
            .forbid
            .iter()
            .any(|p| bytes.windows(p.bytes.len()).any(|w| w == p.bytes.as_slice()))
    }

    fn forbid_key(&self) -> Vec<Vec<u8>> {
        self.forbid.iter().map(|p| p.bytes.clone()).collect()
    }

    /// A JSON string (holding none of the forbidden patterns).
    fn str_node(&mut self) -> NodeId {
        let key = self.forbid_key();
        if let Some(&n) = self.strs.get(&key) {
            return n;
        }
        let n = self.push(Node::Str {
            forbid: self.forbid.clone(),
        });
        self.strs.insert(key, n);
        n
    }

    fn lit(&mut self, s: &str) -> NodeId {
        self.push(Node::Lit(s.as_bytes().to_vec()))
    }

    fn seq(&mut self, parts: Vec<NodeId>) -> NodeId {
        self.push(Node::Seq(parts))
    }

    fn literals(&mut self, values: &[String]) -> NodeId {
        let lits: Vec<(Vec<u8>, u32)> = values
            .iter()
            .map(|v| (v.as_bytes().to_vec(), FINISH))
            .collect();
        let t = self.trie(&lits);
        self.push(Node::Switch(t))
    }

    /// An object with free keys, every value of node `value`.
    fn map(&mut self, value: NodeId) -> NodeId {
        let key = self.str_node();
        self.push(Node::Map { key, value })
    }

    /// Any JSON value (JSON mode).
    fn any(&mut self) -> NodeId {
        let key = self.forbid_key();
        if let Some(&a) = self.anys.get(&key) {
            return a;
        }
        let a = self.push(Node::Alt(Vec::new()));
        self.anys.insert(key, a);
        let map = self.map(a);
        let arr = self.push(Node::Array {
            items: a,
            min: 0,
            max: u32::MAX,
        });
        let num = self.push(Node::Num { integer: false });
        let words = self.literals(&["true".into(), "false".into(), "null".into()]);
        let s = self.str_node();
        if !self.over() {
            self.nodes[a as usize] = Node::Alt(vec![map, arr, s, num, words]);
        }
        a
    }

    /// Starts compiling the schema `s`, the root its `$ref`s resolve in.
    fn set_root(&mut self, s: &JsonValue) {
        self.root_schema = Some(Rc::new(s.clone()));
        self.refs.clear();
    }

    fn answer_doc(&mut self, a: &Answer) -> Result<NodeId, String> {
        let value = match a {
            Answer::JsonObject => {
                let any = self.any();
                self.map(any)
            }
            Answer::Schema(s) => {
                self.set_root(s);
                self.value(s, "#", 0)?
            }
        };
        let w1 = self.push(Node::Ws);
        let w2 = self.push(Node::Ws);
        Ok(self.seq(vec![w1, value, w2]))
    }

    /// The node of a JSON value of `schema` at `path`, `depth` containers deep.
    fn value(&mut self, schema: &JsonValue, path: &str, depth: u64) -> Result<NodeId, String> {
        self.check_size()?;
        if !verified::gram_push_ok(self.levels, MAX_SCHEMA_LEVELS) {
            return Err(format!(
                "response schema: passes through more than {MAX_SCHEMA_LEVELS} schemas (anyOf, allOf, $ref, objects and arrays) at {path}"
            ));
        }
        self.levels += 1;
        let r = self.value_in(schema, path, depth);
        self.levels -= 1;
        let n = r?;
        self.check_size()?;
        Ok(n)
    }

    fn value_in(&mut self, schema: &JsonValue, path: &str, depth: u64) -> Result<NodeId, String> {
        let obj = match schema {
            JsonValue::Bool(true) => return Ok(self.any()),
            JsonValue::Bool(false) => {
                return Err(format!("response schema: the schema at {path} is false and admits no value"))
            }
            JsonValue::Object(o) => o,
            _ => return Err(format!("response schema: the schema at {path} must be an object")),
        };
        let has = |k: &str| obj.iter().any(|(n, _)| n == k);
        let get = |k: &str| schema.get(k);
        for (k, _) in obj {
            let known = ANNOTATIONS.contains(&k.as_str())
                || matches!(
                    k.as_str(),
                    "type"
                        | "properties"
                        | "required"
                        | "additionalProperties"
                        | "items"
                        | "minItems"
                        | "maxItems"
                        | "enum"
                        | "const"
                        | "anyOf"
                        | "allOf"
                        | "$ref"
                );
            if !known {
                return Err(kw_err(path, k, "is not supported (supported: type, properties, required, additionalProperties, items, minItems, maxItems, enum, const, anyOf, allOf with one schema, $ref, $defs)"));
            }
        }
        let others = |allowed: &[&str]| {
            obj.iter()
                .map(|(k, _)| k.as_str())
                .find(|k| !ANNOTATIONS.contains(k) && !allowed.contains(k))
        };
        if let Some(r) = get("$ref") {
            if let Some(k) = others(&["$ref"]) {
                return Err(kw_err(path, k, "can not stand beside '$ref'"));
            }
            let Some(r) = r.as_str() else {
                return Err(kw_err(path, "$ref", "must be a string"));
            };
            return self.reference(r, path, depth);
        }
        if let Some(any_of) = get("anyOf") {
            if let Some(k) = others(&["anyOf"]) {
                return Err(kw_err(path, k, "can not stand beside 'anyOf'"));
            }
            let Some(list) = any_of.as_array().filter(|l| !l.is_empty()) else {
                return Err(kw_err(path, "anyOf", "must be a non-empty array"));
            };
            if !verified::gram_alive_ok(list.len() as u64, MAX_ALIVE) {
                return Err(kw_err(
                    path,
                    "anyOf",
                    &format!("lists more than the {MAX_ALIVE} alternatives a reply can follow at once"),
                ));
            }
            let mut alts = Vec::with_capacity(list.len());
            for (i, s) in list.iter().enumerate() {
                alts.push(self.value(s, &sub_path(path, &["/anyOf/", &i.to_string()]), depth)?);
            }
            return Ok(self.push(Node::Alt(alts)));
        }
        if let Some(all_of) = get("allOf") {
            if let Some(k) = others(&["allOf"]) {
                return Err(kw_err(path, k, "can not stand beside 'allOf'"));
            }
            return match all_of.as_array() {
                Some([one]) => self.value(one, &sub_path(path, &["/allOf/0"]), depth),
                _ => Err(kw_err(path, "allOf", "is supported with exactly one schema")),
            };
        }
        let types: Vec<String> = match get("type") {
            None => Vec::new(),
            Some(JsonValue::Str(t)) => vec![t.clone()],
            Some(JsonValue::Array(a)) if !a.is_empty() && a.iter().all(|t| t.as_str().is_some()) => {
                a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect()
            }
            Some(_) => return Err(kw_err(path, "type", "must be a type name or a list of them")),
        };
        for t in &types {
            if !matches!(
                t.as_str(),
                "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
            ) {
                return Err(kw_err(path, "type", &format!("names the unknown type '{t}'")));
            }
        }
        if has("enum") || has("const") {
            // a literal is checked against its type only: any other constraint beside it is
            // refused rather than left unchecked
            let lit = if has("enum") { "enum" } else { "const" };
            if let Some(k) = others(&["enum", "const", "type"]) {
                return Err(kw_err(path, k, &format!("can not stand beside '{lit}'")));
            }
            let values: Vec<JsonValue> = if let Some(c) = get("const") {
                if has("enum") {
                    return Err(kw_err(path, "const", "can not stand beside 'enum'"));
                }
                vec![c.clone()]
            } else {
                match get("enum").and_then(JsonValue::as_array) {
                    Some(l) if !l.is_empty() => l.to_vec(),
                    _ => return Err(kw_err(path, "enum", "must be a non-empty array")),
                }
            };
            let typed: Vec<String> = values
                .iter()
                .filter(|v| types.is_empty() || types.iter().any(|t| type_of_value(t, v)))
                .map(JsonValue::dump)
                .collect();
            if typed.is_empty() {
                return Err(kw_err(path, lit, "has no value of the schema's type"));
            }
            let kept: Vec<String> = typed.into_iter().filter(|v| self.clean(v.as_bytes())).collect();
            if kept.is_empty() {
                return Err(kw_err(path, lit, "has only values that hold this model's tool tags"));
            }
            for v in &kept {
                if !self.charge(v.len() as u64) {
                    return Err(budget_err());
                }
            }
            return Ok(self.literals(&kept));
        }
        let types = if types.is_empty() {
            // no type: what the other keywords describe, else any value
            if has("properties") || has("additionalProperties") || has("required") {
                vec!["object".to_string()]
            } else if has("items") || has("minItems") || has("maxItems") {
                vec!["array".to_string()]
            } else {
                return Ok(self.any());
            }
        } else {
            types
        };
        let mut alts = Vec::new();
        let mut numeric = false;
        for t in &types {
            let n = match t.as_str() {
                "object" => self.object(schema, path, depth)?,
                "array" => self.array(schema, path, depth)?,
                "string" => self.str_node(),
                // `number` admits every integer, so the pair is one number
                "number" | "integer" if numeric => continue,
                "number" | "integer" => {
                    numeric = true;
                    let integer = !types.iter().any(|t| t == "number");
                    self.push(Node::Num { integer })
                }
                "boolean" => self.literals(&["true".into(), "false".into()]),
                _ => self.lit("null"),
            };
            alts.push(n);
        }
        Ok(if alts.len() == 1 {
            alts[0]
        } else {
            self.push(Node::Alt(alts))
        })
    }

    fn nest(path: &str, depth: u64) -> Result<u64, String> {
        if !verified::gram_push_ok(depth, MAX_SCHEMA_DEPTH) {
            return Err(format!(
                "response schema: nests deeper than {MAX_SCHEMA_DEPTH} objects and arrays at {path}"
            ));
        }
        Ok(depth + 1)
    }

    fn object(&mut self, schema: &JsonValue, path: &str, depth: u64) -> Result<NodeId, String> {
        let depth = Self::nest(path, depth)?;
        let props: &[(String, JsonValue)] = match schema.get("properties") {
            None => &[],
            Some(JsonValue::Object(p)) => p,
            Some(_) => return Err(kw_err(path, "properties", "must be an object")),
        };
        let required = self.required_set(schema, props, path)?;
        match schema.get("additionalProperties") {
            None | Some(JsonValue::Bool(_)) => {}
            Some(extra @ JsonValue::Object(_)) => {
                if !props.is_empty() {
                    return Err(kw_err(
                        path,
                        "additionalProperties",
                        "with a schema is supported only for an object without 'properties'",
                    ));
                }
                let value = self.value(extra, &sub_path(path, &["/additionalProperties"]), depth)?;
                return Ok(self.map(value));
            }
            Some(_) => {
                return Err(kw_err(path, "additionalProperties", "must be a boolean or a schema"))
            }
        }
        let closed = schema.get("additionalProperties") == Some(&JsonValue::Bool(false));
        if props.is_empty() && !closed {
            // an object of free properties
            let any = self.any();
            return Ok(self.map(any));
        }
        let mut defs = Vec::with_capacity(props.len());
        for (k, s) in props {
            let value = self.value(s, &sub_path(path, &["/properties/", k]), depth)?;
            defs.push(PropDef {
                value,
                required: required.contains(k.as_str()),
            });
        }
        let n = defs.len();
        // per position: no required property from it on, and the last property that may come
        // next (the next required one, else the last)
        let mut close_ok = vec![true; n + 1];
        let mut last = vec![0u32; n];
        let mut next_req = None;
        for p in (0..n).rev() {
            if defs[p].required {
                next_req = Some(p as u32);
            }
            close_ok[p] = close_ok[p + 1] && !defs[p].required;
            last[p] = next_req.unwrap_or(n as u32 - 1);
        }
        let table = 4 * n as u64 + 4 * n as u64 + (n as u64 + 1) + (std::mem::size_of::<PropDef>() * n) as u64;
        if !self.charge(table) {
            return Err(budget_err());
        }
        // one trie of every quoted key, its leaves the property indices; each node lists the
        // indices below it, in increasing order (each key adds its index once per byte)
        let Some(root) = self.trie_node() else {
            return Err(budget_err());
        };
        let mut lists: Vec<Vec<u32>> = vec![Vec::new()];
        for (i, (k, _)) in props.iter().enumerate() {
            let q = quote_json(k);
            if !self.clean(q.as_bytes()) {
                return Err(kw_err(path, "properties", &format!("names '{k}', which holds this model's tool tags")));
            }
            if !self.charge(4 * (q.len() as u64 + 1)) {
                return Err(budget_err());
            }
            let mut cur = root;
            lists[0].push(i as u32);
            for &c in q.as_bytes() {
                let Some(next) = self.trie_child(cur, c) else {
                    return Err(budget_err());
                };
                cur = next;
                let at = (cur - root) as usize;
                if at >= lists.len() {
                    lists.resize(at + 1, Vec::new());
                }
                lists[at].push(i as u32);
            }
            if self.tries[cur as usize].leaf == NO_LEAF {
                self.tries[cur as usize].leaf = i as u32;
            }
        }
        let count = self.tries.len() - root as usize;
        lists.resize(count, Vec::new());
        let mut spans = Vec::with_capacity(count);
        let mut idx = Vec::new();
        for l in lists {
            let a = idx.len() as u32;
            idx.extend(l);
            spans.push((a, idx.len() as u32));
        }
        self.objs.push(ObjDef {
            props: defs,
            keys: root,
            spans,
            idx,
            last,
            close_ok,
        });
        Ok(self.push(Node::Object((self.objs.len() - 1) as u32)))
    }

    fn array(&mut self, schema: &JsonValue, path: &str, depth: u64) -> Result<NodeId, String> {
        let depth = Self::nest(path, depth)?;
        let count = |k: &str| -> Result<Option<u32>, String> {
            match schema.get(k) {
                None => Ok(None),
                Some(v) => v
                    .as_u64()
                    .filter(|_| !matches!(v, JsonValue::Float(_)))
                    .map(|n| Some(n.min(u32::MAX as u64 - 1) as u32))
                    .ok_or_else(|| kw_err(path, k, "must be a non-negative integer")),
            }
        };
        let min = count("minItems")?.unwrap_or(0);
        let max = count("maxItems")?.unwrap_or(u32::MAX);
        if min > max {
            return Err(kw_err(path, "minItems", "is larger than 'maxItems'"));
        }
        let items = match schema.get("items") {
            None => self.any(),
            Some(s) => self.value(s, &sub_path(path, &["/items"]), depth)?,
        };
        Ok(self.push(Node::Array { items, min, max }))
    }

    fn reference(&mut self, r: &str, path: &str, depth: u64) -> Result<NodeId, String> {
        if let Some(&n) = self.refs.get(r) {
            return Ok(n);
        }
        let Some(root) = self.root_schema.clone() else {
            return Err(kw_err(path, "$ref", "has no schema to resolve in"));
        };
        let mut target: &JsonValue = &root;
        if r != "#" {
            let Some(rest) = r.strip_prefix("#/") else {
                return Err(kw_err(path, "$ref", &format!("'{r}' is not a local reference ('#/$defs/...')")));
            };
            for part in rest.split('/') {
                let key = part.replace("~1", "/").replace("~0", "~");
                target = match target.get(&key) {
                    Some(c) => c,
                    None => return Err(kw_err(path, "$ref", &format!("'{r}' points at nothing"))),
                };
            }
        }
        // a placeholder first, so a recursive reference finds itself
        let slot = self.push(Node::Alt(Vec::new()));
        self.refs.insert(r.to_string(), slot);
        let n = self.value(target, &short(r), depth)?;
        self.nodes[slot as usize] = Node::Alt(vec![n]);
        Ok(slot)
    }

    /// The names an object schema at `path` lists in `required`, each checked to be one of
    /// `props`. Both lists come from the client, so the work (one hash set of each) is charged
    /// to the compile budget before it is done, and is linear in their sizes.
    fn required_set<'s>(
        &mut self,
        schema: &'s JsonValue,
        props: &'s [(String, JsonValue)],
        path: &str,
    ) -> Result<HashSet<&'s str>, String> {
        let required: &[JsonValue] = match schema.get("required") {
            None => &[],
            Some(JsonValue::Array(r)) if r.iter().all(|x| x.as_str().is_some()) => r,
            Some(_) => return Err(kw_err(path, "required", "must be an array of property names")),
        };
        let entry = (std::mem::size_of::<&str>() + 8) as u64;
        let names = props
            .iter()
            .map(|(k, _)| k.as_str())
            .chain(required.iter().filter_map(JsonValue::as_str));
        let cost = names.fold(0u64, |a, k| a.saturating_add(entry + k.len() as u64));
        if !self.charge(cost) {
            return Err(budget_err());
        }
        let known: HashSet<&str> = props.iter().map(|(k, _)| k.as_str()).collect();
        let mut set = HashSet::with_capacity(required.len().min(known.len()));
        for r in required.iter().filter_map(JsonValue::as_str) {
            if !known.contains(r) {
                return Err(kw_err(
                    path,
                    "required",
                    &format!("names '{}', which is not in 'properties'", short(r)),
                ));
            }
            set.insert(r);
        }
        Ok(set)
    }

    /// A strict tool's parameter list: `(name, schema, required)` in the schema's order. The
    /// root is checked here as `value_in` checks a schema: a keyword it does not apply is
    /// refused, never ignored.
    fn tool_props(&mut self, schema: &JsonValue, path: &str) -> Result<Vec<(String, JsonValue, bool)>, String> {
        let Some(obj) = schema.as_object() else {
            return Err(format!("{path}: must be a JSON schema object"));
        };
        match schema.get("type") {
            None => {}
            Some(JsonValue::Str(t)) if t == "object" => {}
            Some(_) => return Err(format!("{path}: a strict tool's parameters must be an object schema")),
        }
        if schema.get("additionalProperties").is_some_and(|a| a.as_object().is_some()) {
            return Err(kw_err(path, "additionalProperties", "with a schema is not supported for a strict tool"));
        }
        for k in ["anyOf", "allOf", "$ref", "enum", "const"] {
            if schema.get(k).is_some() {
                return Err(kw_err(path, k, "is not supported at the top of a strict tool's parameters"));
            }
        }
        for (k, v) in obj {
            let applied = ANNOTATIONS.contains(&k.as_str())
                || matches!(k.as_str(), "type" | "properties" | "required")
                || (k == "additionalProperties" && matches!(v, JsonValue::Bool(_)));
            if !applied {
                return Err(kw_err(
                    path,
                    k,
                    "is not supported at the top of a strict tool's parameters (supported: type, properties, required, additionalProperties)",
                ));
            }
        }
        let props = match schema.get("properties") {
            None => &[][..],
            Some(JsonValue::Object(p)) => p.as_slice(),
            Some(_) => return Err(kw_err(path, "properties", "must be an object")),
        };
        let required = self.required_set(schema, props, path)?;
        Ok(props
            .iter()
            .map(|(k, s)| (k.clone(), s.clone(), required.contains(k.as_str())))
            .collect())
    }

    /// What follows the tool tag's opening: one call to one of `tools` in the model's form,
    /// through the tag's closing.
    fn call_body(&mut self, tools: &[ToolDef], tags: &CallTags) -> Result<NodeId, String> {
        let mut lits = Vec::with_capacity(tools.len());
        for t in tools {
            // what the reply parser ends a call's name, and the strings of its arguments, at
            self.forbid = match tags.form {
                CallForm::Params => forbid_list(&[PARAM_CLOSE, FUNCTION_CLOSE, tags.close.as_str()])?,
                CallForm::Json => forbid_list(&[tags.close.as_str()])?,
            };
            let name = match tags.form {
                CallForm::Params => format!("{}>\n", t.name),
                CallForm::Json => quote_json(&t.name),
            };
            let bad_name = match tags.form {
                CallForm::Params => t.name.contains(['>', '\n']),
                CallForm::Json => false,
            };
            if bad_name || !self.clean(name.as_bytes()) || !self.charge(name.len() as u64) {
                self.forbid.clear();
                return Err(format!(
                    "tool '{}': its name can not be written in this model's call form",
                    short(&t.name)
                ));
            }
            let cont = match tags.form {
                CallForm::Params => self.params_call(t, tags),
                CallForm::Json => self.json_call(t, tags),
            };
            self.forbid.clear();
            lits.push((name.into_bytes(), cont?));
        }
        let sw = self.trie(&lits);
        let names = self.push(Node::Switch(sw));
        Ok(match tags.form {
            CallForm::Params => {
                let head = self.lit("\n<function=");
                self.seq(vec![head, names])
            }
            CallForm::Json => {
                let w0 = self.push(Node::Ws);
                let open = self.lit("{");
                let w1 = self.push(Node::Ws);
                let key = self.lit("\"name\"");
                let w2 = self.push(Node::Ws);
                let colon = self.lit(":");
                let w3 = self.push(Node::Ws);
                self.seq(vec![w0, open, w1, key, w2, colon, w3, names])
            }
        })
    }

    /// After `NAME>\n`: the parameters, `</function>` and the closing tag.
    fn params_call(&mut self, t: &ToolDef, tags: &CallTags) -> Result<NodeId, String> {
        let tail = self.lit(&format!("</function>\n{}", tags.close));
        let Some(schema) = &t.strict else {
            let raw = self.push(Node::Raw {
                term: Pat::new(b"</function>"),
                forbid: vec![Pat::new(tags.close.as_bytes())],
            });
            let close = self.lit(&format!("\n{}", tags.close));
            return Ok(self.seq(vec![raw, close]));
        };
        let path = format!("tool '{}' parameters", short(&t.name));
        let props = self.tool_props(schema, &path)?;
        // the optional parameters before the next required one (or the end) are all open at
        // once after a parameter, each one stack of the automaton, with the end beside them
        let mut run = 0u64;
        for (_, _, required) in &props {
            run = if *required { 0 } else { run + 1 };
            if !verified::gram_alive_ok(run + 1, MAX_ALIVE) {
                return Err(format!(
                    "{path}: more than {} optional parameters in a row can not be held in this model's call form (make some required, or group them in an object parameter)",
                    MAX_ALIVE - 1
                ));
            }
        }
        self.set_root(schema);
        let mut parts = Vec::with_capacity(props.len() + 1);
        for (k, s, required) in props {
            if k.contains('>') || k.contains('\n') || !self.clean(k.as_bytes()) {
                return Err(format!(
                    "{path}: the property name '{}' can not be written in this model's call form",
                    short(&k)
                ));
            }
            let open = self.lit(&format!("<parameter={k}>\n"));
            let value = if param_is_string(&s) {
                self.raw_param(&s, tags, &sub_path(&path, &["/properties/", &k]))?
            } else {
                let v = self.value(&s, &sub_path(&path, &["/properties/", &k]), 1)?;
                let close = self.lit("\n</parameter>\n");
                self.seq(vec![v, close])
            };
            let one = self.seq(vec![open, value]);
            parts.push(if required {
                one
            } else {
                let none = self.seq(Vec::new());
                self.push(Node::Alt(vec![one, none]))
            });
        }
        parts.push(tail);
        Ok(self.seq(parts))
    }

    /// A string parameter's raw text and its closing (`outparse` reads it to `</parameter>`,
    /// `</function>` or the tool tag's close, whichever comes first).
    fn raw_param(&mut self, s: &JsonValue, tags: &CallTags, path: &str) -> Result<NodeId, String> {
        let close = "\n</parameter>\n";
        // the schema's keywords are checked first, as anywhere else: a keyword the literals
        // below would leave unenforced (`minLength` beside a `const`, `const` beside `enum`) is
        // refused, not ignored
        self.value(s, path, 1)?;
        let values = s
            .get("const")
            .map(|c| vec![c.clone()])
            .or_else(|| s.get("enum").and_then(JsonValue::as_array).map(<[JsonValue]>::to_vec));
        if let Some(values) = values {
            let texts: Vec<String> = values
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|v| !v.contains("</"))
                .map(|v| format!("{v}{close}"))
                .collect();
            if !self.charge(texts.iter().map(|t| t.len() as u64).sum()) {
                return Err(budget_err());
            }
            if texts.is_empty() {
                return Err(format!("{path}: the enum has no string value this call form can write"));
            }
            return Ok(self.literals(&texts));
        }
        // the string schema itself admits every string: its keywords are checked above
        let raw = self.push(Node::Raw {
            term: Pat::new(PARAM_CLOSE.as_bytes()),
            forbid: vec![Pat::new(FUNCTION_CLOSE.as_bytes()), Pat::new(tags.close.as_bytes())],
        });
        let nl = self.lit("\n");
        Ok(self.seq(vec![raw, nl]))
    }

    /// After `"NAME"`: `, "arguments": ARGS }`, whitespace and the closing tag.
    fn json_call(&mut self, t: &ToolDef, tags: &CallTags) -> Result<NodeId, String> {
        let args = match &t.strict {
            Some(schema) => {
                let path = format!("tool '{}' parameters", short(&t.name));
                self.tool_props(schema, &path)?;
                self.set_root(schema);
                self.value(schema, &path, 0)?
            }
            None => {
                let any = self.any();
                self.map(any)
            }
        };
        let mut parts = Vec::new();
        for piece in [None, Some(","), None, Some("\"arguments\""), None, Some(":"), None] {
            parts.push(match piece {
                None => self.push(Node::Ws),
                Some(s) => self.lit(s),
            });
        }
        parts.push(args);
        let w = self.push(Node::Ws);
        let close_brace = self.lit("}");
        let w2 = self.push(Node::Ws);
        let close = self.lit(&tags.close);
        parts.extend([w, close_brace, w2, close]);
        Ok(self.seq(parts))
    }
}

/// The parameter is a string to `outparse` (its raw text is the value): `type` "string", or a
/// list whose first type other than "null" is "string".
pub fn param_is_string(s: &JsonValue) -> bool {
    match s.get("type") {
        Some(JsonValue::Str(t)) => t == "string",
        Some(JsonValue::Array(a)) => a.iter().filter_map(JsonValue::as_str).find(|t| *t != "null") == Some("string"),
        _ => false,
    }
}

/// `v` is a JSON value of type `t`.
fn type_of_value(t: &str, v: &JsonValue) -> bool {
    match (t, v) {
        ("null", JsonValue::Null) => true,
        ("boolean", JsonValue::Bool(_)) => true,
        ("integer", JsonValue::Int(_)) => true,
        ("integer", JsonValue::Float(f)) => f.fract() == 0.0,
        ("number", JsonValue::Int(_) | JsonValue::Float(_)) => true,
        ("string", JsonValue::Str(_)) => true,
        ("array", JsonValue::Array(_)) => true,
        ("object", JsonValue::Object(_)) => true,
        _ => false,
    }
}
