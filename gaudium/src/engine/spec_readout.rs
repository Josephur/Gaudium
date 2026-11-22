//! What a readout needs to know about the speculative runtime of a loaded engine: what it is
//! ([`SpecInfo`], for model descriptions and method names) and what one of its step launches
//! moves over HBM (a [`LaunchModel`] for the live record's `profile` block).
//!
//! A child module of `engine::qwen3::spec`, so it reads the runtime's private configuration
//! without widening it. Everything here is host-only bookkeeping: graph descriptions are rebuilt
//! with the same builders the recipes were compiled from and never compiled, launched or
//! fingerprinted.

use super::*;
use crate::engine::qwen3_traffic;
use crate::live_profile::Traffic;
use crate::live_report::LaunchModel;

/// The speculative runtime of a loaded engine, as configured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecInfo {
    /// `SpecConfig::method_name`: `dspine`, `dflash2`, `mtp`, `mtp-tree`, `oracle`, `oracle-tree`.
    pub method: &'static str,
    /// Drafts verified per launch, widest first (several values: the adaptive policy's set).
    pub draft_k: Vec<u64>,
    /// Whether sampled requests run speculatively too (otherwise they take the `K = 0` loop).
    pub samples: bool,
    /// Shape name of a tree method.
    pub tree: Option<String>,
    /// Device data type of a block drafter's own projection matrices (`None`: the method has
    /// no drafter checkpoint of its own).
    pub drafter_matrix: Option<SynDataType>,
}

impl Qwen3Engine {
    /// The speculative runtime this engine was loaded with (`None`: a `K = 0` engine).
    pub fn spec_info(&self) -> Option<SpecInfo> {
        let rt = self.specdec.as_ref()?;
        Some(SpecInfo {
            method: rt.cfg.method_name(),
            draft_k: rt
                .widths
                .iter()
                .filter(|&&t| t > 1)
                .map(|t| t - 1)
                .collect(),
            samples: rt.handles_sampled(),
            tree: rt.tree.as_ref().map(|t| t.shape.name.clone()),
            drafter_matrix: drafter_matrix_dtype(
                &self.spec,
                rt.lmax,
                rt.drafter.as_ref().map(|d| &d.cfg),
                rt.dflash2.as_ref().map(|d| &d.cfg),
            ),
        })
    }
}

/// Data type of the drafter's first-layer query projection, read off its first-block graph.
fn drafter_matrix_dtype(
    spec: &ModelSpec,
    lmax: u64,
    dspine: Option<&dspine::DSpineCfg>,
    dflash2: Option<&dflash2::DFlash2Cfg>,
) -> Option<SynDataType> {
    let gb = match (dspine, dflash2) {
        (Some(dc), _) => dspine::build_block0_graph(spec, dc, lmax, &[], false),
        (None, Some(dc)) => dflash2::build_block0_graph(spec, dc, lmax, &[]),
        (None, None) => return None,
    };
    gb.persistent_tensor_specs()
        .iter()
        // [step] an FP8 drafter keeps its layer matrices as `<W>_f8` codes
        .find(|t| t.0 == "D0_W_q" || t.0 == "D0_W_q_f8" || t.0 == "D0_W_k")
        .map(|t| t.1)
}

/// Byte model of the chain step recipes of one request (see [`LaunchModel`]): the launch that
/// verified `K` drafts is the `K + 1`-row step graph over the chunks its rows reach.
struct SpecStepModel {
    spec: ModelSpec,
    lmax: u64,
    ns: u64,
    max_chunks: u64,
    dspine: Option<dspine::DSpineCfg>,
    dflash2: Option<dflash2::DFlash2Cfg>,
    /// Draft-readout rows of an MTP chain (`None`: not an MTP method).
    mtp_nvd: Option<u64>,
    /// A sampled request: its launches are the sampled step family.
    sampled: bool,
    /// [step] The kernels that select the DeltaNet verify form of the step families (replay /
    /// MME rows); empty: the checkpoint form. Only the names matter to the graph builders.
    form: Vec<(String, String)>,
}

impl SpecStepModel {
    fn step_graph(&self, t: u64, nch: u64) -> GraphBuilder {
        match self.mtp_nvd {
            Some(nvd) => mtp::build_chain_step_graph_as(
                &self.spec,
                self.lmax,
                nch,
                t,
                self.ns,
                nvd,
                &self.form,
                self.sampled,
            ),
            None => build_step_graph_for(
                &self.spec,
                self.lmax,
                nch,
                t,
                self.ns,
                self.dspine.as_ref(),
                self.dflash2.as_ref(),
                &self.form,
                self.sampled,
            ),
        }
    }
}

impl LaunchModel for SpecStepModel {
    fn key(&self, drafted: u64, pos: u64) -> Option<u64> {
        let t = drafted + 1;
        if t > SPEC_MAX_ROWS {
            return None;
        }
        let nch = (pos + t).div_ceil(KV_CHUNK).clamp(1, self.max_chunks);
        Some((u64::from(self.sampled) << 40) | (t << 16) | nch)
    }

    fn build(&self, key: u64) -> Traffic {
        let (t, nch) = ((key >> 16) & 0xffff, key & 0xffff);
        // A sampled request runs the sampled step family (sampler nodes inside the step recipe);
        // the drafter stays inside the step, as for greedy.
        qwen3_traffic::spec_graph(&self.spec, &self.step_graph(t, nch), t, nch)
    }
}

/// Start of a request: hands the live reporter the byte model of this request's step launches
/// (the speculative loops report each consumed launch but do not mark the launch calls). Tree
/// methods are not modelled (a tree step's rows are not `drafts + 1`), nor is a sampled request
/// of a runtime that sends those to the `K = 0` loop, which marks its own launches.
pub(crate) fn arm(eng: &Qwen3Engine, sampled: bool) {
    let model = eng
        .specdec
        .as_ref()
        .filter(|rt| rt.tree.is_none() && (!sampled || rt.handles_sampled()))
        .map(|rt| {
            Box::new(SpecStepModel {
                spec: eng.spec.clone(),
                lmax: rt.lmax,
                ns: rt.ns,
                max_chunks: eng.max_chunks,
                dspine: rt.drafter.as_ref().map(|d| d.cfg.clone()),
                dflash2: rt.dflash2.as_ref().map(|d| d.cfg.clone()),
                mtp_nvd: rt.mtp.as_ref().map(|m| m.nvd),
                sampled,
                form: rt
                    .elf_hashes
                    .iter()
                    .filter(|(k, _)| k == replay::GDNR_ELF || k == vmme::VREC_ELF)
                    .cloned()
                    .collect(),
            }) as Box<dyn LaunchModel>
        });
    eng.live.set_launch_model(model);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_profile::{Kind, Phase};

    const DSPINE_27B: &str = r#"{
      "architectures": ["DFlashDraftModel"], "block_size": 16,
      "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"},
      "dflash_config": {
        "draft_block_causal": true, "mask_token_id": 248070,
        "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-06, "kappa": 10.0, "rank": 512},
        "selected_prefix_decode_mode": "serial",
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_writes": true,
        "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
        "selected_prefix_terminal": false,
        "target_layer_ids": [1, 16, 31, 46, 61]
      },
      "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
      "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
      "sliding_window": 8192, "vocab_size": 248320
    }"#;

    fn target_27b(fp8: bool) -> ModelSpec {
        let mut spec = ModelSpec::from_config_json(
            r#"{"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": {"model_type": "qwen3_5_text", "hidden_size": 5120,
                "intermediate_size": 17408, "num_hidden_layers": 64, "num_attention_heads": 24,
                "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                "rms_norm_eps": 1e-6, "full_attention_interval": 4, "linear_num_key_heads": 16,
                "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}}}"#,
        )
        .unwrap();
        spec.fp8_weights = fp8;
        spec
    }

    fn model(fp8: bool, sampled: bool) -> SpecStepModel {
        SpecStepModel {
            spec: target_27b(fp8),
            lmax: 8192,
            ns: 16,
            max_chunks: 2,
            dspine: Some(dspine::DSpineCfg::from_json(DSPINE_27B).unwrap()),
            dflash2: None,
            mtp_nvd: None,
            sampled,
            form: Vec::new(),
        }
    }

    #[test]
    fn the_dspine_drafter_runs_its_own_matrices_in_bf16_whatever_the_trunk_runs() {
        let dc = dspine::DSpineCfg::from_json(DSPINE_27B).unwrap();
        for fp8 in [false, true] {
            assert_eq!(
                drafter_matrix_dtype(&target_27b(fp8), 8192, Some(&dc), None),
                Some(SynDataType::Bf16)
            );
        }
        assert_eq!(
            drafter_matrix_dtype(&target_27b(true), 8192, None, None),
            None
        );
    }

    #[test]
    fn keys_follow_the_rows_and_the_chunks_they_reach() {
        let m = model(false, false);
        let k = m.key(15, 100).unwrap();
        assert_eq!(((k >> 16) & 0xffff, k & 0xffff), (16, 1));
        // 16 rows starting at position 4090 reach into the second chunk
        assert_eq!(m.key(15, 4090).unwrap() & 0xffff, 2);
        assert_eq!(m.key(7, 4090).unwrap() & 0xffff, 2);
        assert_eq!(m.key(7, 4000).unwrap() & 0xffff, 1);
        // never more chunks than the engine has
        assert_eq!(m.key(15, 8190).unwrap() & 0xffff, 2);
        assert!(m.key(SPEC_MAX_ROWS, 0).is_none());
        assert_ne!(m.key(7, 0), model(false, true).key(7, 0));
        assert_ne!(m.key(7, 0), m.key(15, 0));
    }

    #[test]
    fn dspine_step_reads_the_trunk_once_and_charges_the_drafter_as_drafts() {
        let m = model(false, false);
        let plain = qwen3_traffic::decode_step(&m.spec, 8192, 1);
        let wide = m.build(m.key(15, 100).unwrap());
        let narrow = m.build(m.key(7, 100).unwrap());

        // the verify trunk reads every weight once whatever the row count
        let trunk = |tr: &Traffic| tr.bytes[Phase::Verify as usize][Kind::Weight as usize];
        assert!(
            (trunk(&wide) / trunk(&plain) - 1.0).abs() < 0.01,
            "16-row verify weights {} vs K=0 {}",
            trunk(&wide),
            trunk(&plain)
        );
        assert!((trunk(&narrow) / trunk(&wide) - 1.0).abs() < 0.01);
        // the drafter section exists and is charged to the drafts phase
        assert!(wide.by_phase(Phase::Draft as usize) > 0.0);
        assert_eq!(plain.by_phase(Phase::Draft as usize), 0.0);
        // more rows: more activations, more checkpoint slots, same weights
        assert!(wide.by_kind(Kind::Act as usize) > narrow.by_kind(Kind::Act as usize));
        assert!(wide.by_kind(Kind::State as usize) > narrow.by_kind(Kind::State as usize));
        assert!(wide.total() > plain.total());
        // a step never moves a multiple of the model: the drafter and 16 rows add well under 2x
        assert!(wide.total() < 2.0 * plain.total());
    }

    #[test]
    fn fp8_weights_halve_the_matrix_traffic_and_sampling_adds_the_block_launch() {
        let bf = model(false, false);
        let f8 = model(true, false);
        let (wb, wf) = (
            bf.build(bf.key(15, 100).unwrap()),
            f8.build(f8.key(15, 100).unwrap()),
        );
        let trunk = |tr: &Traffic| tr.bytes[Phase::Verify as usize][Kind::Weight as usize];
        let ratio = trunk(&wf) / trunk(&wb);
        assert!(
            (0.45..0.60).contains(&ratio),
            "FP8 trunk weights should be about half of BF16, got {ratio:.3}"
        );
        let sampled = model(true, true);
        let ws = sampled.build(sampled.key(15, 100).unwrap());
        assert!(ws.by_phase(Phase::Draft as usize) > wf.by_phase(Phase::Draft as usize));
        assert!((trunk(&ws) - trunk(&wf)).abs() < 1.0);
    }
}
