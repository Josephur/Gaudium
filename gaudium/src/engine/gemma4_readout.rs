//! Byte model of the Gemma 4 decode step for the live record's `profile` block.
//!
//! A child module of `engine::gemma4`, so it reads that module's private name rules. Host-only
//! bookkeeping: the step's graph description is rebuilt with the builder its recipe was compiled
//! from and is never compiled, launched or fingerprinted.

use super::*;
use crate::live_profile::{Traffic, TrafficRules};

/// One launch of the monolithic decode step recipe whose global caches hold `bucket` chunks.
pub(super) fn decode_step(d: &G4Dims, lmax: u64, bucket: u64) -> Traffic {
    let gb = build_step_graph(d, lmax, bucket, &[]);
    Traffic::from_graph(
        &gb,
        TrafficRules {
            rows: 1,
            kv_positions: G4_CHUNK * bucket,
            experts_per_row: 0,
        },
        &is_weight_name,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_profile::{Kind, Phase};

    fn b31() -> ModelSpec {
        let types: Vec<&str> = (0..60)
            .map(|l| {
                if (l + 1) % 6 == 0 {
                    "\"full_attention\""
                } else {
                    "\"sliding_attention\""
                }
            })
            .collect();
        ModelSpec::from_config_json(&format!(
            r#"{{
              "architectures": ["Gemma4ForConditionalGeneration"],
              "model_type": "gemma4",
              "tie_word_embeddings": true,
              "eos_token_id": [1, 106],
              "text_config": {{
                "model_type": "gemma4_text",
                "vocab_size": 262144, "num_hidden_layers": 60,
                "head_dim": 256, "global_head_dim": 512, "final_logit_softcapping": 30.0,
                "rms_norm_eps": 1e-06,
                "layer_types": [{}],
                "rope_parameters": {{
                    "sliding_attention": {{"rope_theta": 10000.0, "rope_type": "default"}},
                    "full_attention": {{"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}}
                }},
                "hidden_size": 5376, "intermediate_size": 21504, "num_attention_heads": 32,
                "num_key_value_heads": 16, "num_global_key_value_heads": 4, "sliding_window": 1024,
                "attention_k_eq_v": true, "hidden_size_per_layer_input": 0,
                "vocab_size_per_layer_input": 262144, "num_kv_shared_layers": 0,
                "use_double_wide_mlp": false
              }}
            }}"#,
            types.join(", ")
        ))
        .unwrap()
    }

    #[test]
    fn the_31b_step_streams_its_mlp_weights_and_reads_caches_and_rings() {
        let d = G4Dims::from_spec(&b31(), 256, G4NormMode::Tile).unwrap();
        let tr = decode_step(&d, 8192, 1);
        // gate + up + down of 60 layers, BF16, are the bulk of the weights a step streams
        let mlp = 60.0 * 3.0 * 5376.0 * 21504.0 * 2.0;
        let w = tr.by_kind(Kind::Weight as usize);
        assert!(w > mlp && w < 1.6 * mlp, "weights {w} vs MLP {mlp}");
        assert!(
            tr.by_kind(Kind::Kv as usize) > 0.0,
            "caches and rings are read"
        );
        assert_eq!(tr.by_kind(Kind::State as usize), 0.0);
        assert!(
            tr.by_phase(Phase::Readout as usize) > 0.0,
            "the readout reads the embedding table"
        );
        assert!(tr.by_phase(Phase::Verify as usize) > tr.by_phase(Phase::Readout as usize));
        assert_eq!(tr.by_phase(Phase::Draft as usize), 0.0);
        // a second chunk grows the global caches' read, not the weights
        let tr2 = decode_step(&d, 8192, 2);
        assert!(tr2.by_kind(Kind::Kv as usize) > tr.by_kind(Kind::Kv as usize));
        assert!((tr2.by_kind(Kind::Weight as usize) - w).abs() < 1.0);
    }
}
