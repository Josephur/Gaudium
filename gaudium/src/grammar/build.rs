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

use std::collections::HashMap;

use super::{Grammar, Node, NodeId, ObjDef, Pat, PropDef, TrieNode, FINISH, NO_LEAF};
use crate::json::{quote_json, JsonValue};
use crate::verified;

/// Schema nesting levels (objects and arrays inside each other) a request may declare.
pub const MAX_SCHEMA_DEPTH: u64 = 32;
/// Grammar nodes a request may compile to.
pub const MAX_NODES: usize = 200_000;

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
}

/// A reply's grammar: an answer form, the tools it may call, or both.
pub fn reply(answer: Option<&Answer>, tools: &[ToolDef], tags: &CallTags) -> Result<Grammar, String> {
    let mut b = Builder::new();
    let calls = (!tools.is_empty())
        .then(|| b.call_body(tools, tags))
        .transpose()?;
    let root = match (answer, calls) {
        (Some(a), calls) => {
            let doc = b.answer_doc(a)?;
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
    b.finish(root)
}

/// The grammar of one JSON document of `schema` (tests and tools).
pub fn schema_doc(schema: &JsonValue) -> Result<Grammar, String> {
    let mut b = Builder::new();
    let root = b.answer_doc(&Answer::Schema(schema.clone()))?;
    b.finish(root)
}

struct Builder {
    nodes: Vec<Node>,
    tries: Vec<TrieNode>,
    objs: Vec<ObjDef>,
    any: Option<NodeId>,
    str_node: NodeId,
    /// The schema `$ref`s resolve in, and the node of every pointer compiled so far.
    root_schema: Option<JsonValue>,
    refs: HashMap<String, NodeId>,
}

fn kw_err(path: &str, kw: &str, why: &str) -> String {
    format!("response schema: '{kw}' at {path} {why}")
}

impl Builder {
    fn new() -> Self {
        let mut b = Self {
            nodes: Vec::new(),
            tries: Vec::new(),
            objs: Vec::new(),
            any: None,
            str_node: 0,
            root_schema: None,
            refs: HashMap::new(),
        };
        b.str_node = b.push(Node::Str);
        b
    }

    fn push(&mut self, n: Node) -> NodeId {
        self.nodes.push(n);
        (self.nodes.len() - 1) as NodeId
    }

    fn finish(self, root: NodeId) -> Result<Grammar, String> {
        if self.nodes.len() > MAX_NODES {
            return Err(format!(
                "response schema: compiles to {} grammar nodes, more than the {MAX_NODES} a request may use",
                self.nodes.len()
            ));
        }
        Ok(Grammar {
            nodes: self.nodes,
            tries: self.tries,
            objs: self.objs,
            root,
            str_node: self.str_node,
        })
    }

    fn check_size(&self) -> Result<(), String> {
        if self.nodes.len() > MAX_NODES {
            return Err(format!(
                "response schema: compiles to more than the {MAX_NODES} grammar nodes a request may use"
            ));
        }
        Ok(())
    }

    /// A trie over `lits`, each ending with its leaf value.
    fn trie(&mut self, lits: &[(Vec<u8>, u32)]) -> u32 {
        let root = self.tries.len() as u32;
        self.tries.push(TrieNode {
            children: Vec::new(),
            leaf: NO_LEAF,
        });
        for (bytes, leaf) in lits {
            let mut cur = root;
            for &c in bytes {
                let found = self.tries[cur as usize]
                    .children
                    .iter()
                    .find(|e| e.0 == c)
                    .map(|e| e.1);
                cur = match found {
                    Some(n) => n,
                    None => {
                        let n = self.tries.len() as u32;
                        self.tries.push(TrieNode {
                            children: Vec::new(),
                            leaf: NO_LEAF,
                        });
                        let kids = &mut self.tries[cur as usize].children;
                        let at = kids.partition_point(|e| e.0 < c);
                        kids.insert(at, (c, n));
                        n
                    }
                };
            }
            // a literal listed twice keeps its first leaf
            if self.tries[cur as usize].leaf == NO_LEAF {
                self.tries[cur as usize].leaf = *leaf;
            }
        }
        root
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

    /// Any JSON value (JSON mode).
    fn any(&mut self) -> NodeId {
        if let Some(a) = self.any {
            return a;
        }
        let a = self.push(Node::Alt(Vec::new()));
        self.any = Some(a);
        let map = self.push(Node::Map { value: a });
        let arr = self.push(Node::Array {
            items: a,
            min: 0,
            max: u32::MAX,
        });
        let num = self.push(Node::Num { integer: false });
        let words = self.literals(&["true".into(), "false".into(), "null".into()]);
        let s = self.str_node;
        self.nodes[a as usize] = Node::Alt(vec![map, arr, s, num, words]);
        a
    }

    fn answer_doc(&mut self, a: &Answer) -> Result<NodeId, String> {
        let value = match a {
            Answer::JsonObject => {
                let any = self.any();
                self.push(Node::Map { value: any })
            }
            Answer::Schema(s) => {
                self.root_schema = Some(s.clone());
                self.refs.clear();
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
            let mut alts = Vec::with_capacity(list.len());
            for (i, s) in list.iter().enumerate() {
                alts.push(self.value(s, &format!("{path}/anyOf/{i}"), depth)?);
            }
            return Ok(self.push(Node::Alt(alts)));
        }
        if let Some(all_of) = get("allOf") {
            if let Some(k) = others(&["allOf"]) {
                return Err(kw_err(path, k, "can not stand beside 'allOf'"));
            }
            return match all_of.as_array() {
                Some([one]) => self.value(one, &format!("{path}/allOf/0"), depth),
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
            let kept: Vec<String> = values
                .iter()
                .filter(|v| types.is_empty() || types.iter().any(|t| type_of_value(t, v)))
                .map(JsonValue::dump)
                .collect();
            if kept.is_empty() {
                return Err(kw_err(path, "enum", "has no value of the schema's type"));
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
                "string" => self.str_node,
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
        let required: Vec<&str> = match schema.get("required") {
            None => Vec::new(),
            Some(JsonValue::Array(r)) if r.iter().all(|x| x.as_str().is_some()) => {
                r.iter().filter_map(JsonValue::as_str).collect()
            }
            Some(_) => return Err(kw_err(path, "required", "must be an array of property names")),
        };
        if let Some(missing) = required.iter().find(|r| !props.iter().any(|(k, _)| k == *r)) {
            return Err(kw_err(
                path,
                "required",
                &format!("names '{missing}', which is not in 'properties'"),
            ));
        }
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
                let value = self.value(extra, &format!("{path}/additionalProperties"), depth)?;
                return Ok(self.push(Node::Map { value }));
            }
            Some(_) => {
                return Err(kw_err(path, "additionalProperties", "must be a boolean or a schema"))
            }
        }
        let closed = schema.get("additionalProperties") == Some(&JsonValue::Bool(false));
        if props.is_empty() && !closed {
            // an object of free properties
            let any = self.any();
            return Ok(self.push(Node::Map { value: any }));
        }
        let mut defs = Vec::with_capacity(props.len());
        for (k, s) in props {
            let value = self.value(s, &format!("{path}/properties/{k}"), depth)?;
            defs.push(PropDef {
                value,
                required: required.contains(&k.as_str()),
            });
        }
        let n = defs.len();
        let mut keys = Vec::with_capacity(n + 1);
        let mut close_ok = Vec::with_capacity(n + 1);
        for p in 0..=n {
            close_ok.push(defs[p..].iter().all(|d| !d.required));
            if p == n {
                keys.push(None);
                continue;
            }
            let last = (p..n).find(|&i| defs[i].required).unwrap_or(n - 1);
            let lits: Vec<(Vec<u8>, u32)> = (p..=last)
                .map(|i| (quote_json(&props[i].0).into_bytes(), i as u32))
                .collect();
            keys.push(Some(self.trie(&lits)));
        }
        self.objs.push(ObjDef {
            props: defs,
            keys,
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
            Some(s) => self.value(s, &format!("{path}/items"), depth)?,
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
        let target = if r == "#" {
            root.clone()
        } else {
            let Some(rest) = r.strip_prefix("#/") else {
                return Err(kw_err(path, "$ref", &format!("'{r}' is not a local reference ('#/$defs/...')")));
            };
            let mut cur = &root;
            for part in rest.split('/') {
                let key = part.replace("~1", "/").replace("~0", "~");
                cur = match cur.get(&key) {
                    Some(c) => c,
                    None => return Err(kw_err(path, "$ref", &format!("'{r}' points at nothing"))),
                };
            }
            cur.clone()
        };
        // a placeholder first, so a recursive reference finds itself
        let slot = self.push(Node::Alt(Vec::new()));
        self.refs.insert(r.to_string(), slot);
        let n = self.value(&target, r, depth)?;
        self.nodes[slot as usize] = Node::Alt(vec![n]);
        Ok(slot)
    }

    /// What follows the tool tag's opening: one call to one of `tools` in the model's form,
    /// through the tag's closing.
    fn call_body(&mut self, tools: &[ToolDef], tags: &CallTags) -> Result<NodeId, String> {
        let mut lits = Vec::with_capacity(tools.len());
        for t in tools {
            let cont = match tags.form {
                CallForm::Params => self.params_call(t, tags)?,
                CallForm::Json => self.json_call(t, tags)?,
            };
            let name = match tags.form {
                CallForm::Params => format!("{}>\n", t.name),
                CallForm::Json => quote_json(&t.name),
            };
            lits.push((name.into_bytes(), cont));
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
        let path = format!("tool '{}' parameters", t.name);
        let props = tool_props(schema, &path)?;
        self.root_schema = Some(schema.clone());
        self.refs.clear();
        let mut parts = Vec::with_capacity(props.len() + 1);
        for (k, s, required) in props {
            if k.contains('>') || k.contains('\n') {
                return Err(format!(
                    "{path}: the property name '{k}' can not be written in this model's call form"
                ));
            }
            let open = self.lit(&format!("<parameter={k}>\n"));
            let value = if param_is_string(&s) {
                self.raw_param(&s, tags, &format!("{path}/properties/{k}"))?
            } else {
                let v = self.value(&s, &format!("{path}/properties/{k}"), 1)?;
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
            if texts.is_empty() {
                return Err(format!("{path}: the enum has no string value this call form can write"));
            }
            return Ok(self.literals(&texts));
        }
        // the string schema itself admits every string: only its keywords are checked
        self.value(s, path, 1)?;
        let raw = self.push(Node::Raw {
            term: Pat::new(b"</parameter>"),
            forbid: vec![Pat::new(b"</function>"), Pat::new(tags.close.as_bytes())],
        });
        let nl = self.lit("\n");
        Ok(self.seq(vec![raw, nl]))
    }

    /// After `"NAME"`: `, "arguments": ARGS }`, whitespace and the closing tag.
    fn json_call(&mut self, t: &ToolDef, tags: &CallTags) -> Result<NodeId, String> {
        let args = match &t.strict {
            Some(schema) => {
                let path = format!("tool '{}' parameters", t.name);
                tool_props(schema, &path)?;
                self.root_schema = Some(schema.clone());
                self.refs.clear();
                self.value(schema, &path, 0)?
            }
            None => {
                let any = self.any();
                self.push(Node::Map { value: any })
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

/// A strict tool's parameter list: `(name, schema, required)` in the schema's order.
fn tool_props(schema: &JsonValue, path: &str) -> Result<Vec<(String, JsonValue, bool)>, String> {
    if schema.as_object().is_none() {
        return Err(format!("{path}: must be a JSON schema object"));
    }
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
    let props = match schema.get("properties") {
        None => &[][..],
        Some(JsonValue::Object(p)) => p.as_slice(),
        Some(_) => return Err(kw_err(path, "properties", "must be an object")),
    };
    let required: Vec<&str> = schema
        .get("required")
        .and_then(JsonValue::as_array)
        .map(|r| r.iter().filter_map(JsonValue::as_str).collect())
        .unwrap_or_default();
    if let Some(missing) = required.iter().find(|r| !props.iter().any(|(k, _)| k == *r)) {
        return Err(kw_err(path, "required", &format!("names '{missing}', which is not in 'properties'")));
    }
    Ok(props
        .iter()
        .map(|(k, s)| (k.clone(), s.clone(), required.contains(&k.as_str())))
        .collect())
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
