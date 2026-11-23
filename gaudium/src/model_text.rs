//! A model's text resources, for a model given as a checkpoint directory or as a GGUF file (or
//! a directory holding one GGUF model): its tokenizer, chat template, the special-token names the
//! template may use, the model description (`ModelSpec`), its context limit and its default
//! sampling settings. The command line, the server and the engines ask here instead of reading
//! the checkpoint's files themselves.
//!
//! Where each resource comes from (`verified/model_text.bend`, `mt_source`):
//! - a checkpoint directory: its own files (`config.json`, `tokenizer.json`, `chat_template.jinja`
//!   or the `chat_template` of `tokenizer_config.json`, `generation_config.json`);
//! - a GGUF model: the file's own metadata whenever it carries the resource (it was written with
//!   the weights it describes, so its vocabulary is the one the embedding rows were trained on);
//!   a file of the directory that holds the GGUF file fills in only what the metadata lacks.
//!   A GGUF build published with the original text files beside it therefore uses the file's
//!   metadata for everything it carries, and the files beside it are not read for those.
//!
//! What a GGUF file gives in place of each checkpoint file:
//! - `config.json`: the family's description built from the `<arch>.*` keys (Inkling 2:
//!   `engine::inkling2::gguf_source::config_json`); the context limit is `<arch>.context_length`;
//! - `tokenizer.json`: `tokenizer.ggml.*` (`Tokenizer::from_gguf`);
//! - the chat template: `tokenizer.chat_template`;
//! - the special-token names of `tokenizer_config.json`: the token texts of
//!   `tokenizer.ggml.{bos,eos,padding,unknown}_token_id`. A `response_template` has no GGUF
//!   form: a template whose own markers give the output form (the Inkling 2 block form) does not
//!   need one; another template is parsed in the default form;
//! - `generation_config.json`: `general.sampling.{temp,top_k,top_p}`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gguf::Gguf;
use crate::json::{parse_json, quote_json, JsonValue};
use crate::model_spec::ModelSpec;
use crate::tokenizer::Tokenizer;
use crate::verified as v;
use crate::verified::MtSource;

/// The special-token names of a chat template and the GGUF keys of their ids.
const TOKEN_KEYS: [(&str, &str); 4] = [
    ("bos_token", "tokenizer.ggml.bos_token_id"),
    ("eos_token", "tokenizer.ggml.eos_token_id"),
    ("pad_token", "tokenizer.ggml.padding_token_id"),
    ("unk_token", "tokenizer.ggml.unknown_token_id"),
];

/// The sampling settings of `generation_config.json` and their GGUF keys.
const SAMPLING_KEYS: [(&str, &str); 3] = [
    ("temperature", "general.sampling.temp"),
    ("top_k", "general.sampling.top_k"),
    ("top_p", "general.sampling.top_p"),
];

/// The text resources of one model path.
pub struct ModelText {
    path: PathBuf,
    /// Where the side files are: the checkpoint directory, or the directory holding the GGUF.
    dir: PathBuf,
    gguf: Option<Arc<Gguf>>,
}

impl ModelText {
    /// A checkpoint directory, a GGUF file, or a directory holding one GGUF model (the GGUF
    /// metadata is read here; tensors are not touched).
    pub fn open(path: &Path) -> Result<Self, String> {
        if crate::gguf::looks_like_gguf(path) {
            let g = Gguf::open(path).map_err(|e| e.to_string())?;
            return Ok(Self::from_gguf(path, Arc::new(g)));
        }
        Ok(Self {
            path: path.to_path_buf(),
            dir: path.to_path_buf(),
            gguf: None,
        })
    }

    /// A GGUF model already opened from `path`.
    pub fn from_gguf(path: &Path, g: Arc<Gguf>) -> Self {
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().map(Path::to_path_buf).unwrap_or_default()
        };
        Self {
            path: path.to_path_buf(),
            dir,
            gguf: Some(g),
        }
    }

    /// The path as given.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The GGUF model, when the path is one.
    pub fn gguf(&self) -> Option<&Arc<Gguf>> {
        self.gguf.as_ref()
    }

    /// Where one resource comes from: `in_file`, the GGUF metadata carries it; `in_dir`, the
    /// directory has it.
    fn source(&self, in_file: bool, in_dir: bool) -> MtSource {
        v::mt_source(self.gguf.is_some(), self.gguf.is_some() && in_file, in_dir)
    }

    fn read_dir_file(&self, name: &str) -> Option<String> {
        std::fs::read_to_string(self.dir.join(name)).ok()
    }

    fn dir_has(&self, name: &str) -> bool {
        self.dir.join(name).is_file()
    }

    /// The `config.json` text of the model: the checkpoint's file, or the description built from
    /// a GGUF file's metadata.
    pub fn config_json(&self) -> Result<String, String> {
        let from_file = self.gguf.as_deref().map(gguf_config_json);
        let in_file = matches!(from_file, Some(Ok(_)));
        match self.source(in_file, self.dir_has("config.json")) {
            MtSource::MtFromFile => from_file.unwrap_or_else(|| Err(String::new())),
            MtSource::MtFromDir => std::fs::read_to_string(self.dir.join("config.json"))
                .map_err(|e| format!("{}: {e}", self.dir.join("config.json").display())),
            MtSource::MtAbsent => Err(match from_file {
                Some(Err(e)) => e,
                _ => format!("no config.json in {}", self.dir.display()),
            }),
        }
    }

    /// The model description.
    pub fn spec(&self) -> Result<ModelSpec, String> {
        let text = self.config_json()?;
        ModelSpec::from_config_json(&text).map_err(|e| e.to_string())
    }

    /// The model's own context limit (`max_position_embeddings` of the text config, else its
    /// `model_max_length`, or `<arch>.context_length` of a GGUF file); `None` when none is stated.
    pub fn max_positions(&self) -> Option<u64> {
        let from_file = self
            .gguf
            .as_deref()
            .and_then(|g| g.arch_kv("context_length"))
            .and_then(|x| x.as_u64())
            .filter(|&n| n > 0);
        match self.source(from_file.is_some(), self.dir_has("config.json")) {
            MtSource::MtFromFile => from_file,
            MtSource::MtFromDir => {
                let root = parse_json(&self.read_dir_file("config.json")?).ok()?;
                let tc = root.get("text_config").unwrap_or(&root);
                ["max_position_embeddings", "model_max_length"]
                    .iter()
                    .find_map(|k| tc.get(k).and_then(JsonValue::as_u64).filter(|&n| n > 0))
            }
            MtSource::MtAbsent => None,
        }
    }

    /// The tokenizer: `tokenizer.json`, or a GGUF file's `tokenizer.ggml.*` metadata.
    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        let in_file = self
            .gguf
            .as_deref()
            .is_some_and(|g| g.kv("tokenizer.ggml.model").is_some());
        match self.source(in_file, self.dir_has("tokenizer.json")) {
            MtSource::MtFromFile => {
                let g = self.gguf.as_deref().expect("source is the GGUF file");
                Tokenizer::from_gguf(g)
                    .map_err(|e| format!("tokenizer of {}: {e}", self.path.display()))
            }
            MtSource::MtFromDir => Tokenizer::from_model_dir(&self.dir)
                .map_err(|e| format!("{}: {e}", self.dir.join("tokenizer.json").display())),
            MtSource::MtAbsent => Err(format!(
                "no tokenizer: {} has no tokenizer.json{}",
                self.dir.display(),
                if self.gguf.is_some() {
                    " and the GGUF file no tokenizer.ggml metadata"
                } else {
                    ""
                }
            )),
        }
    }

    /// The parsed `tokenizer_config.json` of the directory.
    fn dir_tokenizer_config(&self) -> Option<JsonValue> {
        parse_json(&self.read_dir_file("tokenizer_config.json")?).ok()
    }

    /// The chat template's source text and the configuration it is read with (the special-token
    /// names, and for a checkpoint the rest of its `tokenizer_config.json`). `None` source: the
    /// model has no template.
    pub fn chat_template(&self) -> (Option<String>, Option<JsonValue>) {
        let dir_config = self.dir_tokenizer_config();
        let file_source = self
            .gguf
            .as_deref()
            .and_then(|g| g.get_str("tokenizer.chat_template"))
            .map(str::to_string);
        let dir_source = || -> Option<String> {
            match self.read_dir_file("chat_template.jinja") {
                Some(s) => Some(s),
                None => dir_config.as_ref().and_then(template_of_config),
            }
        };
        let in_dir = self.dir_has("chat_template.jinja")
            || dir_config.as_ref().and_then(template_of_config).is_some();
        let source = match self.source(file_source.is_some(), in_dir) {
            MtSource::MtFromFile => file_source,
            MtSource::MtFromDir => dir_source(),
            MtSource::MtAbsent => None,
        };
        let file_tokens = self.gguf.as_deref().and_then(gguf_token_names);
        let config = match self.source(file_tokens.is_some(), dir_config.is_some()) {
            MtSource::MtFromFile => file_tokens,
            MtSource::MtFromDir => dir_config,
            MtSource::MtAbsent => None,
        };
        (source, config)
    }

    /// The chat template's source text alone.
    pub fn chat_template_source(&self) -> Option<String> {
        self.chat_template().0
    }

    /// The model's default sampling settings as `generation_config.json` text (a GGUF file's
    /// `general.sampling.*` keys in that form); `None`: the model states none.
    pub fn generation_config(&self) -> Option<String> {
        let file = self.gguf.as_deref().and_then(gguf_sampling_json);
        match self.source(file.is_some(), self.dir_has("generation_config.json")) {
            MtSource::MtFromFile => file,
            MtSource::MtFromDir => self.read_dir_file("generation_config.json"),
            MtSource::MtAbsent => None,
        }
    }
}

/// The `chat_template` of a `tokenizer_config.json`: a string, or the default entry of a list of
/// named templates.
fn template_of_config(c: &JsonValue) -> Option<String> {
    match c.get("chat_template") {
        Some(JsonValue::Str(s)) => Some(s.clone()),
        Some(JsonValue::Array(list)) => list
            .iter()
            .find(|t| t.get("name").and_then(JsonValue::as_str) == Some("default"))
            .and_then(|t| t.get("template"))
            .and_then(JsonValue::as_str)
            .map(str::to_string),
        _ => None,
    }
}

/// The `config.json` text of a GGUF model of a family whose metadata Gaudium reads.
fn gguf_config_json(g: &Gguf) -> Result<String, String> {
    use crate::engine::inkling2::gguf_source as ink;
    if ink::is_family(g) {
        return ink::config_json(g);
    }
    Err(format!(
        "no model description for GGUF architecture '{}' yet",
        g.arch()
    ))
}

/// Number of tokens of a GGUF vocabulary (0 when there is none).
fn gguf_vocab_len(g: &Gguf) -> u64 {
    g.kv("tokenizer.ggml.tokens")
        .and_then(|x| x.as_array())
        .map_or(0, |a| a.len())
}

/// The text of token `id` of a GGUF vocabulary, when the id names one.
pub(crate) fn gguf_token_text(g: &Gguf, id: u64) -> Option<String> {
    let toks = g.kv("tokenizer.ggml.tokens")?.as_array()?;
    if !v::mt_id_ok(id, toks.len()) {
        return None;
    }
    std::str::from_utf8(toks.string_at(id)?)
        .ok()
        .map(str::to_string)
}

/// The special-token names of a GGUF file, as a `tokenizer_config.json` object holds them;
/// `None` when the file names none.
fn gguf_token_names(g: &Gguf) -> Option<JsonValue> {
    if gguf_vocab_len(g) == 0 {
        return None;
    }
    let mut entries = Vec::new();
    for (name, key) in TOKEN_KEYS {
        if let Some(text) = g.get_u64(key).and_then(|id| gguf_token_text(g, id)) {
            entries.push((name.to_string(), JsonValue::Str(text)));
        }
    }
    (!entries.is_empty()).then_some(JsonValue::Object(entries))
}

/// A GGUF file's `general.sampling.*` keys as `generation_config.json` text.
fn gguf_sampling_json(g: &Gguf) -> Option<String> {
    let mut fields = Vec::new();
    for (name, key) in SAMPLING_KEYS {
        let Some(x) = g.kv(key) else { continue };
        let value = if name == "top_k" {
            x.as_i64().map(|k| k.to_string())
        } else {
            x.as_f64()
                .filter(|f| f.is_finite())
                .map(|f| format!("{f:?}"))
        };
        if let Some(value) = value {
            fields.push(format!("{}:{value}", quote_json(name)));
        }
    }
    (!fields.is_empty()).then(|| format!("{{{}}}", fields.join(",")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::gguf::test_gguf::{tiny_vocab, write_gguf, Kv};

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gaudium-mt-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A tiny byte-level GGUF vocabulary: the 256 byte tokens, "ab", and two control tokens.
    fn tiny_vocab_kvs(template: Option<&str>) -> Vec<(String, Kv)> {
        let (toks, types) = tiny_vocab(&["ab"], &["<|end|>", "<|pad|>"]);
        let mut kvs = vec![
            ("general.architecture".to_string(), Kv::Str("tiny".into())),
            ("tokenizer.ggml.model".into(), Kv::Str("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Kv::Str("qwen2".into())),
            ("tokenizer.ggml.tokens".into(), Kv::StrArr(toks)),
            ("tokenizer.ggml.token_type".into(), Kv::I32Arr(types)),
            (
                "tokenizer.ggml.merges".into(),
                Kv::StrArr(vec!["a b".into()]),
            ),
            ("tokenizer.ggml.eos_token_id".into(), Kv::U32(257)),
            ("tokenizer.ggml.padding_token_id".into(), Kv::U32(258)),
            ("tokenizer.ggml.bos_token_id".into(), Kv::U32(9999)),
            ("tiny.context_length".into(), Kv::U32(8192)),
            ("general.sampling.temp".into(), Kv::F32(0.5)),
            ("general.sampling.top_k".into(), Kv::I32(0)),
        ];
        if let Some(t) = template {
            kvs.push(("tokenizer.chat_template".into(), Kv::Str(t.into())));
        }
        kvs
    }

    #[test]
    fn a_gguf_file_gives_its_own_text_resources() {
        let d = scratch("own");
        let f = d.join("m.gguf");
        write_gguf(
            &f,
            &tiny_vocab_kvs(Some("{{ messages[0].content }}{{ eos_token }}")),
        );
        // text files beside the file are not read for what the file carries
        std::fs::write(d.join("chat_template.jinja"), "BESIDE").unwrap();
        std::fs::write(d.join("generation_config.json"), r#"{"temperature":0.1}"#).unwrap();
        let t = ModelText::open(&f).unwrap();
        assert!(t.gguf().is_some());
        let tok = t.tokenizer().unwrap();
        assert_eq!(tok.encode("ab<|end|>", true), vec![256, 257]);
        assert!(tok.is_special(257) && !tok.is_special(256));
        let (source, config) = t.chat_template();
        assert_eq!(
            source.as_deref(),
            Some("{{ messages[0].content }}{{ eos_token }}")
        );
        let config = config.unwrap();
        assert_eq!(
            config.get("eos_token").and_then(JsonValue::as_str),
            Some("<|end|>")
        );
        assert_eq!(
            config.get("pad_token").and_then(JsonValue::as_str),
            Some("<|pad|>")
        );
        // an id past the vocabulary names no token
        assert!(config.get("bos_token").is_none());
        assert_eq!(t.max_positions(), Some(8192));
        assert_eq!(
            t.generation_config().as_deref(),
            Some(r#"{"temperature":0.5,"top_k":0}"#)
        );
        // the family has no description yet: the error says so
        assert!(t.config_json().unwrap_err().contains("'tiny'"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn files_beside_a_gguf_fill_in_what_it_lacks() {
        let d = scratch("beside");
        let f = d.join("m.gguf");
        let kvs: Vec<_> = tiny_vocab_kvs(None)
            .into_iter()
            .filter(|(k, _)| !k.starts_with("general.sampling"))
            .collect();
        write_gguf(&f, &kvs);
        std::fs::write(d.join("chat_template.jinja"), "BESIDE").unwrap();
        std::fs::write(d.join("generation_config.json"), r#"{"temperature":0.1}"#).unwrap();
        let t = ModelText::open(&f).unwrap();
        assert_eq!(t.chat_template_source().as_deref(), Some("BESIDE"));
        assert_eq!(
            t.generation_config().as_deref(),
            Some(r#"{"temperature":0.1}"#)
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_checkpoint_directory_reads_its_files() {
        let d = scratch("dir");
        std::fs::write(
            d.join("config.json"),
            r#"{"text_config":{"max_position_embeddings":40960}}"#,
        )
        .unwrap();
        std::fs::write(
            d.join("tokenizer_config.json"),
            r#"{"eos_token":"<e>","chat_template":"FROM-CONFIG"}"#,
        )
        .unwrap();
        let t = ModelText::open(&d).unwrap();
        assert!(t.gguf().is_none());
        assert_eq!(t.max_positions(), Some(40960));
        // a text config that states the limit as `model_max_length`
        std::fs::write(
            d.join("config.json"),
            r#"{"text_config":{"model_max_length":1048576}}"#,
        )
        .unwrap();
        assert_eq!(t.max_positions(), Some(1048576));
        let (source, config) = t.chat_template();
        assert_eq!(source.as_deref(), Some("FROM-CONFIG"));
        assert_eq!(
            config.unwrap().get("eos_token").and_then(JsonValue::as_str),
            Some("<e>")
        );
        // the .jinja file comes first
        std::fs::write(d.join("chat_template.jinja"), "FROM-JINJA").unwrap();
        assert_eq!(t.chat_template_source().as_deref(), Some("FROM-JINJA"));
        assert!(t.generation_config().is_none());
        assert!(t.tokenizer().err().unwrap().contains("tokenizer.json"));
        let _ = std::fs::remove_dir_all(&d);
    }
}
