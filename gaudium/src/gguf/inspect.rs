//! `gaudium --inspect-model <file.gguf | dir>`: what a GGUF model holds, checked.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use super::{Gguf, GgufError, Value};
use crate::verified as v;

/// Tensors, weights and bytes of one ggml type.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TypeCount {
    pub tensors: u64,
    pub params: u64,
    pub bytes: u64,
}

/// Weights per ggml type name over every tensor of the model.
pub fn type_counts(g: &Gguf) -> BTreeMap<&'static str, TypeCount> {
    let mut m: BTreeMap<&'static str, TypeCount> = BTreeMap::new();
    for t in g.tensors() {
        let c = m.entry(t.ty.name()).or_default();
        c.tensors = v::gg_count_add(c.tensors, 1);
        c.params = v::gg_count_add(c.params, t.n_elems);
        c.bytes = v::gg_count_add(c.bytes, t.nbytes);
    }
    m
}

fn show(v: &Value<'_>) -> String {
    match v {
        Value::UInt(u, _) => u.to_string(),
        Value::Int(i, _) => i.to_string(),
        Value::Float(f, super::ValueType::F32) => format!("{}", *f as f32),
        Value::Float(f, _) => format!("{f}"),
        Value::Bool(b) => b.to_string(),
        Value::Str(s) if s.len() > 60 => format!(
            "{:?}... ({} bytes)",
            &s[..s.floor_char_boundary(60)],
            s.len()
        ),
        Value::Str(s) => format!("{s:?}"),
        Value::Array(a) => {
            let head: Vec<String> = match a.elem_type() {
                super::ValueType::Str => a
                    .strings()
                    .take(4)
                    .map(|s| format!("{:?}", s.unwrap_or("<bytes>")))
                    .collect(),
                _ => (0..a.len().min(8))
                    .filter_map(|k| a.get(k))
                    .map(|e| show(&e))
                    .collect(),
            };
            let more = if a.len() > head.len() as u64 {
                ", ..."
            } else {
                ""
            };
            format!(
                "[{}{more}] ({} x {})",
                head.join(", "),
                a.len(),
                a.elem_type().name()
            )
        }
    }
}

/// Opens the model (every tensor's byte range and every metadata length is checked on open)
/// and describes it. An error names what is malformed.
pub fn inspect(path: &Path) -> Result<String, GgufError> {
    let g = Gguf::open(path)?;
    let mut o = String::new();
    let f0 = &g.files[0];
    let _ = writeln!(
        o,
        "GGUF v{} ({} shard{}), alignment {}, {} keys, {} tensors",
        f0.version,
        g.files.len(),
        if g.files.len() == 1 { "" } else { "s" },
        f0.alignment,
        g.keys().count(),
        g.tensor_count()
    );
    for side in &g.side_files {
        let _ = writeln!(o, "  beside it (not part of the model): {}", side.display());
    }
    let arch = g.arch().to_owned();
    let _ = writeln!(o, "architecture: {arch:?}");
    for k in [
        "general.name",
        "general.type",
        "general.size_label",
        "general.file_type",
        "general.quantization_version",
    ] {
        if let Some(val) = g.kv(k) {
            let _ = writeln!(o, "  {k} = {}", show(&val));
        }
    }
    let _ = writeln!(o, "hyper-parameters:");
    let prefix = format!("{arch}.");
    for k in g.keys() {
        if k.starts_with(&prefix) {
            if let Some(val) = g.kv(k) {
                let _ = writeln!(o, "  {k} = {}", show(&val));
            }
        }
    }
    let toks = g
        .kv("tokenizer.ggml.tokens")
        .and_then(|t| t.as_array())
        .map(|a| a.len());
    if let Some(n) = toks {
        let _ = writeln!(
            o,
            "tokenizer: {} ({n} tokens), chat template: {}",
            g.get_str("tokenizer.ggml.model").unwrap_or("?"),
            if g.kv("tokenizer.chat_template").is_some() {
                "yes"
            } else {
                "no"
            }
        );
    }
    let counts = type_counts(&g);
    let mut total = TypeCount::default();
    let _ = writeln!(o, "tensors by type:");
    for (name, c) in &counts {
        let dec = super::GgmlType::from_name(name).is_some_and(|t| t.info().decodable);
        let _ = writeln!(
            o,
            "  {name:<8} {:>6} tensors {:>16} params {:>16} bytes{}",
            c.tensors,
            c.params,
            c.bytes,
            if dec { "" } else { "  (no host decoder yet)" }
        );
        total.tensors = v::gg_count_add(total.tensors, c.tensors);
        total.params = v::gg_count_add(total.params, c.params);
        total.bytes = v::gg_count_add(total.bytes, c.bytes);
    }
    let _ = writeln!(
        o,
        "  {:<8} {:>6} tensors {:>16} params {:>16} bytes",
        "total", total.tensors, total.params, total.bytes
    );
    let _ = write!(
        o,
        "OK: every tensor lies inside its file, aligned, and is a whole number of blocks"
    );
    Ok(o)
}
