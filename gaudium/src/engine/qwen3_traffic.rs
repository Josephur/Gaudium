//! Byte model of the Qwen3 / Qwen3.5 / Qwen3.8 recipes for the live record's `profile` block.
//!
//! Rebuilds the pure graph descriptions (no compile, no device) and charges their nodes with
//! `live_profile::Traffic`. The reporter caches each result, so a model is built once per step
//! recipe (chunk count) and once per prefill tile shape.

use crate::engine::qwen3::{
    build_decode_step_graph, build_embed_graph, build_prefill_attn_layer_graph,
    build_prefill_linear_layer_graph, is_generic_layer_weight, is_weight_or_table, spec, KV_CHUNK,
};
use crate::live_profile::{Traffic, TrafficRules};
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::synapse::ffi::SynDataType;
use crate::synapse::graph::GraphBuilder;

fn is_weight(name: &str) -> bool {
    is_weight_or_table(name) || is_generic_layer_weight(name)
}

/// Weights and tables of the trunk and of every device drafter a step recipe may carry.
fn is_step_weight(name: &str) -> bool {
    is_weight(name)
        || spec::dspine::is_drafter_weight(name)
        || spec::dflash2::is_drafter_weight(name)
        || name == spec::dflash2::DV_HEAD
        || name == spec::dflash2::DV_IDS
        || spec::mtp::is_mtp_weight(name)
}

fn rules(spec: &ModelSpec, rows: u64, nch: u64) -> TrafficRules {
    TrafficRules {
        rows,
        kv_positions: KV_CHUNK * nch,
        experts_per_row: if spec.moe_routed {
            spec.experts_per_tok
        } else {
            0
        },
    }
}

/// One launch of a speculative step recipe (`gb` = its graph description: `rows` verify rows
/// over `nch` cache chunks, plus the drafter section when the recipe carries one), or of a
/// drafter's own recipe (`rows` = its block).
pub(crate) fn spec_graph(spec: &ModelSpec, gb: &GraphBuilder, rows: u64, nch: u64) -> Traffic {
    Traffic::from_graph(gb, rules(spec, rows, nch), &is_step_weight)
}

/// Device data types and sizes the decode step graph of a model declares, by role: what a
/// description of the loaded model reports as its precision. Read off the graph description the
/// recipes are compiled from (no compile, no device).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepDtypes {
    /// The projection / MLP matrices (layer 0's gate matrix).
    pub matrix: Option<SynDataType>,
    /// The readout matrix (`lm_head`, or the tied embedding table).
    pub head: Option<SynDataType>,
    /// Attention key / value caches (`None`: the model has no full-attention layer).
    pub kv: Option<SynDataType>,
    /// Recurrent state of the linear-attention layers (`None`: the model has none).
    pub state: Option<SynDataType>,
    /// Bytes of the MLP matrices of every layer; for a mixture-of-experts model these are the
    /// expert matrices (routed experts and the shared expert).
    pub mlp_bytes: u64,
}

/// See [`StepDtypes`]. `None` for a family whose step graph is not built here.
pub fn step_dtypes(spec: &ModelSpec) -> Option<StepDtypes> {
    if !matches!(
        spec.family,
        ModelFamily::Qwen3 | ModelFamily::Qwen35Dense | ModelFamily::Qwen35Moe
    ) {
        return None;
    }
    let gb = build_decode_step_graph(spec, KV_CHUNK, 1, &[]);
    let specs = gb.persistent_tensor_specs();
    let find = |pred: &dyn Fn(&str) -> bool| specs.iter().find(|t| pred(&t.0)).map(|t| t.1);
    let lm = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    let is_mlp = |n: &str| {
        n.starts_with('L')
            && ["_W_gate", "_W_up", "_W_down", "_EW12", "_EW3"]
                .iter()
                .any(|s| n.ends_with(s))
    };
    Some(StepDtypes {
        matrix: find(&|n| n == "L0_W_gate"),
        head: find(&|n| n == lm),
        kv: find(&|n| n.ends_with("_kcache")),
        state: find(&|n| n.ends_with("_S_in")),
        mlp_bytes: specs.iter().filter(|t| is_mlp(&t.0)).map(|t| t.3).sum(),
    })
}

/// One launch of the monolithic decode step recipe reading `nch` cache chunks.
pub(crate) fn decode_step(spec: &ModelSpec, lmax: u64, nch: u64) -> Traffic {
    let gb = build_decode_step_graph(spec, lmax, nch, &[]);
    Traffic::from_graph(&gb, rules(spec, 1, nch), &is_weight)
}

/// All launches of one prefill tile of `t` rows: the embedding gather plus one single-layer
/// launch per layer (full-attention layers read `nch` cache chunks). The final readout launch of
/// a request is not included.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn prefill_tile(spec: &ModelSpec, lmax: u64, t: u64, nch: u64) -> Traffic {
    let n_lin = spec.n_linear_layers() as u64;
    let n_attn = spec.n_layers.saturating_sub(n_lin);
    let attn = (n_attn > 0).then(|| build_prefill_attn_layer_graph(spec, t, lmax, nch, &[]));
    let lin = (n_lin > 0).then(|| build_prefill_linear_layer_graph(spec, t, &[]));
    prefill_tile_of(spec, t, nch, attn.as_ref(), lin.as_ref())
}

/// `prefill_tile` for the tile graphs the engine actually launches (`engine/qwen3/prefill.rs`
/// builds them in the form its levers select).
pub(crate) fn prefill_tile_of(
    spec: &ModelSpec,
    t: u64,
    nch: u64,
    attn: Option<&crate::synapse::graph::GraphBuilder>,
    lin: Option<&crate::synapse::graph::GraphBuilder>,
) -> Traffic {
    let rules = rules(spec, t, nch);
    let mut total = Traffic::from_graph(&build_embed_graph(spec, t), rules, &is_weight);
    let n_lin = spec.n_linear_layers() as u64;
    let n_attn = spec.n_layers.saturating_sub(n_lin);
    if let Some(gb) = attn {
        total.add(&Traffic::from_graph(gb, rules, &is_weight), n_attn as f64);
    }
    if let Some(gb) = lin {
        total.add(&Traffic::from_graph(gb, rules, &is_weight), n_lin as f64);
    }
    // [moe] routed form: every layer's routed experts and tail run in the routed-experts recipe
    // of the tile, at its padded row count
    if spec.is_moe() && spec.moe_routed {
        let rows =
            crate::engine::moe::routed_prefill_rows(t, crate::engine::qwen3::MAX_PREFILL_TILE);
        let gb = crate::engine::moe::build_prefill_routed_graph(spec, rows);
        total.add(
            &Traffic::from_graph(&gb, self::rules(spec, rows, nch), &is_weight),
            spec.n_layers as f64,
        );
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_profile::{Kind, Phase};

    fn qwen3_8b_spec() -> ModelSpec {
        let cfg = r#"{
          "model_type": "qwen3",
          "hidden_size": 4096,
          "intermediate_size": 12288,
          "num_hidden_layers": 36,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "vocab_size": 151936,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "tie_word_embeddings": false
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    fn qwen38_27b_spec() -> ModelSpec {
        let cfg = r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
          }
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    #[test]
    fn decode_step_8b_reads_every_trunk_weight_once_and_the_cache_bucket() {
        let spec = qwen3_8b_spec();
        let tr = decode_step(&spec, 4096, 1);
        let (hid, inter, vocab) = (4096.0f64, 12288.0f64, 151936.0f64);
        // per layer: q, o (hid x hid), k, v (hid x 1024), gate, up, down (hid x inter), BF16
        let big = 2.0 * (2.0 * hid * hid + 2.0 * hid * 1024.0 + 3.0 * hid * inter);
        let trunk = 36.0 * big;
        let lm_head = 2.0 * hid * vocab;
        let w = tr.by_kind(Kind::Weight as usize);
        // small tensors (norm weights, RoPE rows, one embedding row) stay under 0.1 %
        assert!(
            w >= trunk + lm_head && w < (trunk + lm_head) * 1.001,
            "weights {w}"
        );
        assert!(
            (tr.bytes[Phase::Readout as usize][Kind::Weight as usize] - lm_head).abs() < 1.0,
            "the readout reads the lm_head once"
        );
        // qk reads the K bucket, pv the V bucket, prep writes one row of each: per layer
        let bucket = 128.0 * 4096.0 * 8.0 * 2.0;
        let kv = tr.by_kind(Kind::Kv as usize);
        let want = 36.0 * (2.0 * bucket + 2.0 * 128.0 * 8.0 * 2.0);
        assert!((kv - want).abs() < 1.0, "kv {kv} want {want}");
        assert_eq!(tr.by_kind(Kind::State as usize), 0.0);
        // a second chunk doubles the cache read, not the weights
        let tr2 = decode_step(&spec, 8192, 2);
        assert!((tr2.by_kind(Kind::Kv as usize) - 36.0 * (4.0 * bucket + 4096.0)).abs() < 1.0);
        assert!((tr2.by_kind(Kind::Weight as usize) - w).abs() < 1.0);
        assert!(tr.total_flops() > 0.0);
    }

    #[test]
    fn step_dtypes_follow_the_run_precision_and_the_layer_kinds() {
        let dense = step_dtypes(&qwen3_8b_spec()).unwrap();
        assert_eq!(dense.matrix, Some(SynDataType::Bf16));
        assert_eq!(dense.head, Some(SynDataType::Bf16));
        assert_eq!(dense.kv, Some(SynDataType::Bf16));
        assert_eq!(
            dense.state, None,
            "no linear-attention layer, no recurrent state"
        );
        // gate + up + down of 36 layers, BF16
        assert_eq!(dense.mlp_bytes, 36 * 3 * 4096 * 12288 * 2);

        let mut hybrid = qwen38_27b_spec();
        let bf = step_dtypes(&hybrid).unwrap();
        assert_eq!(bf.matrix, Some(SynDataType::Bf16));
        assert_eq!(bf.state, Some(SynDataType::Float));
        hybrid.fp8_weights = true;
        let f8 = step_dtypes(&hybrid).unwrap();
        assert_eq!(f8.matrix, Some(SynDataType::Fp8_143));
        assert_eq!(
            f8.head,
            Some(SynDataType::Bf16),
            "the readout matrix stays BF16"
        );
        assert_eq!(f8.kv, Some(SynDataType::Bf16));
        assert_eq!(f8.mlp_bytes * 2, bf.mlp_bytes);
    }

    #[test]
    fn hybrid_27b_models_recurrent_state_and_prefill_tiles() {
        let spec = qwen38_27b_spec();
        let step = decode_step(&spec, 4096, 1);
        assert!(
            step.by_kind(Kind::State as usize) > 0.0,
            "GDN layers move state"
        );
        assert!(
            step.by_kind(Kind::Kv as usize) > 0.0,
            "attention layers read caches"
        );
        let gb = step.total() / 1e9;
        assert!((40.0..80.0).contains(&gb), "27B step moves {gb:.1} GB");

        let tile = prefill_tile(&spec, 4096, 256, 1);
        // a tile reads the trunk weights once more or less, like a step, without the readout
        let w_step = step.by_kind(Kind::Weight as usize)
            - step.bytes[Phase::Readout as usize][Kind::Weight as usize];
        let w_tile = tile.by_kind(Kind::Weight as usize);
        assert!(
            (w_tile / w_step - 1.0).abs() < 0.01,
            "tile {w_tile} step {w_step}"
        );
        assert_eq!(tile.by_phase(Phase::Readout as usize), 0.0);
        // activations scale with the tile width
        assert!(tile.by_kind(Kind::Act as usize) > 100.0 * step.by_kind(Kind::Act as usize) / 2.0);
    }
}
