//! Unified model checkpoint spec parser & validator (`model_spec.hpp`, `adapter_qwen35.hpp`,
//! `adapter_gemma4.hpp`).
//!
//! Parses Hugging Face `config.json` for Qwen3, Qwen3.5/3.6/3.8 (dense, MoE, Gated DeltaNet,
//! MTP, ViT) and Gemma 4 (E2B/E4B/12B/31B, sliding/global dual RoPE, K=V, PLE, KV-sharing,
//! vision, audio), and validates all kernel-baked dimensions through `core::shapes` and
//! `core::gemma4`.

use std::fmt;
use std::fs;
use std::path::Path;

use crate::core::gemma4::{
    validate_gemma4_audio_ctx, validate_gemma4_heads, validate_gemma4_kv_share,
    validate_gemma4_ple, validate_gemma4_vision_heads, Gemma4Rope,
};
use crate::core::shapes::{validate_gdn, validate_gqa, validate_hidden, validate_moe};
use crate::json::{parse_json, JsonError, JsonValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3,
    Qwen35Dense,
    Qwen35Moe,
    Gemma4,
    Gemma4Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Spec {
    pub unified: bool,
    pub sliding_window: u64,
    pub global_head_dim: u64,
    pub global_kv_heads: u64,
    pub k_eq_v: bool,
    pub softcap: f64,
    pub ple_dim: u64,
    pub ple_vocab: u64,
    pub kv_shared: u64,
    pub double_wide_mlp: bool,
    pub s_theta: f64,
    pub g_theta: f64,
    pub s_rope_type: String,
    pub g_rope_type: String,
    pub s_partial: f64,
    pub g_partial: f64,
    pub has_audio: bool,
    pub a_left: u64,
    pub a_right: u64,
    pub v_heads: u64,
    pub v_head_dim: u64,
    pub v_hidden: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub family: ModelFamily,
    pub architecture: String,
    pub model_type: String,
    pub hidden: u64,
    pub intermediate: u64,
    pub vocab: u64,
    pub n_layers: u64,
    pub n_heads: u64,
    pub n_kv_heads: u64,
    pub head_dim: u64,
    pub rms_eps: f64,
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<String>,
    // Qwen3.5 / 3.6 / 3.8 Gated DeltaNet fields (0 when absent)
    pub gdn_k_heads: u64,
    pub gdn_v_heads: u64,
    pub gdn_conv_k: u64,
    // MoE fields (0 when dense)
    pub n_experts: u64,
    pub experts_per_tok: u64,
    pub moe_intermediate: u64,
    // Multimodal / FP8 flags
    pub has_vision: bool,
    pub fp8_quantized: bool,
    pub gemma4: Option<Gemma4Spec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    Io(String),
    Json(JsonError),
    Unsupported(String),
    InvalidDims(String),
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(m) => write!(f, "model_spec io: {m}"),
            Self::Json(e) => write!(f, "model_spec {e}"),
            Self::Unsupported(m) => write!(f, "model_spec unsupported: {m}"),
            Self::InvalidDims(m) => write!(f, "model_spec invalid dims: {m}"),
        }
    }
}

impl std::error::Error for SpecError {}

impl From<JsonError> for SpecError {
    fn from(e: JsonError) -> Self {
        Self::Json(e)
    }
}

fn u64_field(v: &JsonValue, key: &str) -> u64 {
    v.get(key).and_then(JsonValue::as_u64).unwrap_or(0)
}

fn f64_field(v: &JsonValue, key: &str, default: f64) -> f64 {
    v.get(key).and_then(JsonValue::as_f64).unwrap_or(default)
}

fn bool_field(v: &JsonValue, key: &str, default: bool) -> bool {
    v.get(key).and_then(JsonValue::as_bool).unwrap_or(default)
}

fn str_field<'a>(v: &'a JsonValue, key: &str) -> &'a str {
    v.get(key).and_then(JsonValue::as_str).unwrap_or("")
}

impl ModelSpec {
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self, SpecError> {
        let txt = fs::read_to_string(path.as_ref())
            .map_err(|e| SpecError::Io(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_config_json(&txt)
    }

    pub fn from_config_json(text: &str) -> Result<Self, SpecError> {
        let root = parse_json(text)?;
        let tc = root.get("text_config").unwrap_or(&root);

        let architecture = root
            .get("architectures")
            .and_then(JsonValue::as_array)
            .and_then(|a| a.first())
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_owned();
        let model_type = str_field(&root, "model_type").to_owned();
        let text_model_type = str_field(tc, "model_type");

        let family = if architecture.contains("Gemma4Assistant") || model_type == "gemma4_assistant"
        {
            ModelFamily::Gemma4Assistant
        } else if architecture.contains("Gemma4")
            || model_type.starts_with("gemma4")
            || text_model_type.starts_with("gemma4")
        {
            ModelFamily::Gemma4
        } else if architecture.contains("Qwen3_5Moe")
            || text_model_type == "qwen3_5_moe_text"
            || u64_field(tc, "num_experts") > 0
        {
            ModelFamily::Qwen35Moe
        } else if architecture.contains("Qwen3_5")
            || text_model_type == "qwen3_5_text"
            || u64_field(tc, "linear_num_key_heads") > 0
        {
            ModelFamily::Qwen35Dense
        } else if architecture.contains("Qwen3") || model_type == "qwen3" {
            ModelFamily::Qwen3
        } else {
            return Err(SpecError::Unsupported(format!(
                "unrecognized architecture='{architecture}', model_type='{model_type}'"
            )));
        };

        let hidden = u64_field(tc, "hidden_size");
        let intermediate = u64_field(tc, "intermediate_size");
        let vocab = u64_field(tc, "vocab_size");
        let n_layers = u64_field(tc, "num_hidden_layers");
        let n_heads = u64_field(tc, "num_attention_heads");
        let n_kv_heads = u64_field(tc, "num_key_value_heads");
        let head_dim = {
            let d = u64_field(tc, "head_dim");
            if d > 0 {
                d
            } else {
                hidden.checked_div(n_heads).unwrap_or(0)
            }
        };
        let rms_eps = f64_field(tc, "rms_norm_eps", 1e-6);
        let tie_word_embeddings = bool_field(
            &root,
            "tie_word_embeddings",
            bool_field(tc, "tie_word_embeddings", false),
        );

        let layer_types: Vec<String> =
            if let Some(arr) = tc.get("layer_types").and_then(JsonValue::as_array) {
                arr.iter()
                    .map(|x| x.as_str().unwrap_or("").to_owned())
                    .collect()
            } else {
                let interval = u64_field(tc, "full_attention_interval");
                if interval > 0 {
                    (0..n_layers)
                        .map(|l| {
                            if (l + 1) % interval == 0 {
                                "full_attention".to_owned()
                            } else {
                                "linear_attention".to_owned()
                            }
                        })
                        .collect()
                } else {
                    vec!["full_attention".to_owned(); n_layers as usize]
                }
            };

        let gdn_k_heads = u64_field(tc, "linear_num_key_heads");
        let gdn_v_heads = u64_field(tc, "linear_num_value_heads");
        let gdn_conv_k = u64_field(tc, "linear_conv_kernel_dim");

        let n_experts = u64_field(tc, "num_experts");
        let experts_per_tok = u64_field(tc, "num_experts_per_tok");
        let moe_intermediate = u64_field(tc, "moe_intermediate_size");

        let has_vision = root
            .get("vision_config")
            .and_then(JsonValue::as_object)
            .is_some();
        let fp8_quantized = root
            .get("quantization_config")
            .and_then(JsonValue::as_object)
            .is_some();

        let gemma4 = if matches!(family, ModelFamily::Gemma4 | ModelFamily::Gemma4Assistant) {
            let rp = tc.get("rope_parameters");
            let srp = rp.and_then(|x| x.get("sliding_attention"));
            let grp = rp.and_then(|x| x.get("full_attention"));
            let g_dh = {
                let v = u64_field(tc, "global_head_dim");
                if v > 0 {
                    v
                } else {
                    head_dim
                }
            };
            let g_nkv = {
                let v = u64_field(tc, "num_global_key_value_heads");
                if v > 0 {
                    v
                } else {
                    n_kv_heads
                }
            };
            let vc = root.get("vision_config");
            let ac = root.get("audio_config").filter(|v| v.as_object().is_some());
            Some(Gemma4Spec {
                unified: architecture.contains("Unified"),
                sliding_window: u64_field(tc, "sliding_window"),
                global_head_dim: g_dh,
                global_kv_heads: g_nkv,
                k_eq_v: bool_field(tc, "attention_k_eq_v", false),
                softcap: f64_field(tc, "final_logit_softcapping", 0.0),
                ple_dim: u64_field(tc, "hidden_size_per_layer_input"),
                ple_vocab: u64_field(tc, "vocab_size_per_layer_input"),
                kv_shared: u64_field(tc, "num_kv_shared_layers"),
                double_wide_mlp: bool_field(tc, "use_double_wide_mlp", false),
                s_theta: srp
                    .map(|x| f64_field(x, "rope_theta", 10_000.0))
                    .unwrap_or(10_000.0),
                g_theta: grp
                    .map(|x| f64_field(x, "rope_theta", 1_000_000.0))
                    .unwrap_or(1_000_000.0),
                s_rope_type: srp
                    .map(|x| str_field(x, "rope_type"))
                    .unwrap_or("default")
                    .to_owned(),
                g_rope_type: grp
                    .map(|x| str_field(x, "rope_type"))
                    .unwrap_or("default")
                    .to_owned(),
                s_partial: srp
                    .map(|x| f64_field(x, "partial_rotary_factor", 1.0))
                    .unwrap_or(1.0),
                g_partial: grp
                    .map(|x| f64_field(x, "partial_rotary_factor", 1.0))
                    .unwrap_or(1.0),
                has_audio: ac.is_some(),
                a_left: ac
                    .map(|x| u64_field(x, "attention_context_left"))
                    .unwrap_or(0),
                a_right: ac
                    .map(|x| u64_field(x, "attention_context_right"))
                    .unwrap_or(0),
                v_heads: vc.map(|x| u64_field(x, "num_attention_heads")).unwrap_or(0),
                v_head_dim: vc.map(|x| u64_field(x, "head_dim")).unwrap_or(0),
                v_hidden: vc.map(|x| u64_field(x, "hidden_size")).unwrap_or(0),
            })
        } else {
            None
        };

        let spec = Self {
            family,
            architecture,
            model_type,
            hidden,
            intermediate,
            vocab,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
            tie_word_embeddings,
            layer_types,
            gdn_k_heads,
            gdn_v_heads,
            gdn_conv_k,
            n_experts,
            experts_per_tok,
            moe_intermediate,
            has_vision,
            fp8_quantized,
            gemma4,
        };
        Ok(spec)
    }

    /// Validates all kernel-baked dimensions against the verified Bend 2 guards (`core::shapes` & `core::gemma4`).
    pub fn validate(&self) -> Result<(), SpecError> {
        match self.family {
            ModelFamily::Qwen3 => {
                validate_hidden(self.hidden, self.vocab)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                validate_gqa(self.n_heads, self.n_kv_heads, self.head_dim)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
            }
            ModelFamily::Qwen35Dense => {
                validate_hidden(self.hidden, self.vocab)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                validate_gqa(self.n_heads, self.n_kv_heads, self.head_dim)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                if self.gdn_k_heads > 0 {
                    validate_gdn(self.gdn_k_heads, self.gdn_v_heads, self.gdn_conv_k)
                        .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                }
            }
            ModelFamily::Qwen35Moe => {
                validate_hidden(self.hidden, self.vocab)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                validate_gqa(self.n_heads, self.n_kv_heads, self.head_dim)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                if self.gdn_k_heads > 0 {
                    validate_gdn(self.gdn_k_heads, self.gdn_v_heads, self.gdn_conv_k)
                        .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                }
                validate_moe(self.n_experts, self.experts_per_tok, self.moe_intermediate)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
            }
            ModelFamily::Gemma4 => {
                let g4 = self
                    .gemma4
                    .as_ref()
                    .ok_or_else(|| SpecError::InvalidDims("missing gemma4 spec".into()))?;
                validate_hidden(self.hidden, self.vocab)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                let s_pairs =
                    Gemma4Rope::rotated_pairs(&g4.s_rope_type, g4.s_partial, self.head_dim);
                let g_pairs =
                    Gemma4Rope::rotated_pairs(&g4.g_rope_type, g4.g_partial, g4.global_head_dim);
                validate_gemma4_heads(
                    "sliding",
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    s_pairs,
                )
                .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                validate_gemma4_heads(
                    "global",
                    self.n_heads,
                    g4.global_kv_heads,
                    g4.global_head_dim,
                    g_pairs,
                )
                .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                validate_gemma4_ple(g4.ple_dim, g4.ple_vocab, self.vocab)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                validate_gemma4_kv_share(self.n_layers, g4.kv_shared)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                if g4.has_audio && !g4.unified && g4.a_left > 0 {
                    validate_gemma4_audio_ctx(g4.a_left, g4.a_right)
                        .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                }
                if self.has_vision && !g4.unified && g4.v_heads > 0 {
                    validate_gemma4_vision_heads(g4.v_heads, g4.v_head_dim, g4.v_hidden)
                        .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
                }
            }
            ModelFamily::Gemma4Assistant => {
                validate_hidden(self.hidden, self.vocab)
                    .map_err(|e| SpecError::InvalidDims(format!("{e:?}")))?;
            }
        }
        Ok(())
    }

    pub fn n_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| t.as_str() == "full_attention")
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_qwen38_dense_and_qwen36_moe_and_gemma4() {
        let q38 = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320,
                "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
                "head_dim": 256, "full_attention_interval": 4,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_conv_kernel_dim": 4
            },
            "vision_config": {"depth": 27}
        }"#;
        let s = ModelSpec::from_config_json(q38).unwrap();
        assert_eq!(s.family, ModelFamily::Qwen35Dense);
        assert_eq!(s.n_full_attention_layers(), 16);
        assert!(s.has_vision);
        s.validate().unwrap();

        let q36 = r#"{
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "quantization_config": {"quant_method": "fp8"},
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048, "intermediate_size": 512, "vocab_size": 248320,
                "num_hidden_layers": 40, "num_attention_heads": 16, "num_key_value_heads": 2,
                "head_dim": 256, "full_attention_interval": 4,
                "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_conv_kernel_dim": 4,
                "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512
            }
        }"#;
        let m = ModelSpec::from_config_json(q36).unwrap();
        assert_eq!(m.family, ModelFamily::Qwen35Moe);
        assert!(m.fp8_quantized);
        m.validate().unwrap();

        let g4 = r#"{
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 5376, "intermediate_size": 21504, "vocab_size": 262144,
                "num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 16,
                "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 4,
                "sliding_window": 1024, "attention_k_eq_v": true,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                    "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
                }
            }
        }"#;
        let g = ModelSpec::from_config_json(g4).unwrap();
        assert_eq!(g.family, ModelFamily::Gemma4);
        assert!(g.gemma4.as_ref().unwrap().k_eq_v);
        g.validate().unwrap();
    }
}
