//! Builders of what a readout client (the Gaudium Web UI) reads from this server: the model
//! description, the method names, a request's timings, its accept sequence and its sections.
//!
//! Everything here is a pure function of facts the engine or the request measured. A value that
//! was not measured is left out of the JSON (or `null`), never filled with a default, so a
//! client shows it as unknown.

use crate::engine::qwen3::spec::readout::SpecInfo;
use crate::engine::qwen3_traffic::StepDtypes;
use crate::live_report::{LaunchRec, RequestSummary};
use crate::model_spec::ModelSpec;
use crate::synapse::ffi::SynDataType;

/// The drafter of a served model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DrafterFacts {
    /// `dspine`, `dflash2`, `mtp`, `mtp-tree`, or a test method's name.
    pub kind: String,
    /// Drafts verified per launch, widest first (several: the adaptive policy's set).
    pub draft_k: Vec<u64>,
    /// Whether sampled requests run speculatively too.
    pub samples: bool,
    pub tree: Option<String>,
    /// Data type of the drafter's own matrices (known once the engine is loaded).
    pub matrix: Option<SynDataType>,
    /// Directory name of the drafter checkpoint.
    pub checkpoint: Option<String>,
}

impl DrafterFacts {
    pub fn from_engine(info: &SpecInfo, checkpoint: Option<String>) -> Self {
        Self {
            kind: info.method.to_string(),
            draft_k: info.draft_k.clone(),
            samples: info.samples,
            tree: info.tree.clone(),
            matrix: info.drafter_matrix,
            checkpoint,
        }
    }

    /// The drafter's name in a model description's `drafters` list (`None`: a test method
    /// that has no drafter of its own).
    pub fn desc_name(&self) -> Option<&'static str> {
        match self.kind.as_str() {
            "dspine" => Some("dspine"),
            "dflash2" => Some("dflash2"),
            "mtp" | "mtp-tree" => Some("mtp"),
            _ => None,
        }
    }
}

/// How a served model runs on the card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RunFacts {
    /// `true`: read from the loaded engine. `false`: what the server's configuration will load
    /// (the model is not resident), reported as `planned`.
    pub from_engine: bool,
    /// The projection / MLP matrices run as FP8.
    pub fp8_weights: bool,
    pub moe_routed: bool,
    pub drafter: Option<DrafterFacts>,
    /// The engine was loaded with image input.
    pub vision: bool,
    /// Exact device-memory ledger `[weights, kv, state, scratch, workspace]` of the loaded engine.
    pub hbm_bytes: Option<[u64; 5]>,
}

/// Run label of the method a request takes, in the readout's label grammar: `0` (no drafts),
/// `ds<K>p` / `ds-ap` (DSpine, fixed / adaptive width), `df<K>p` / `df-ap` (DFlash2), `<K>` /
/// `adaptive` (MTP chain), `T<shape>` (MTP tree).
pub(super) fn method_label(drafter: Option<&DrafterFacts>, sampled: bool) -> String {
    let Some(d) = drafter.filter(|d| (!sampled || d.samples) && !d.draft_k.is_empty()) else {
        return "0".to_string();
    };
    let ks = &d.draft_k;
    let block = |p: &str| {
        if ks.len() > 1 {
            format!("{p}-ap")
        } else {
            format!("{p}{}p", ks[0])
        }
    };
    match d.kind.as_str() {
        "dspine" => block("ds"),
        "dflash2" => block("df"),
        "mtp" if ks.len() > 1 => "adaptive".to_string(),
        "mtp" => ks[0].to_string(),
        "mtp-tree" => format!("T{}", d.tree.as_deref().unwrap_or("")),
        other => {
            let k: Vec<String> = ks.iter().map(u64::to_string).collect();
            format!("{other}:{}", k.join(","))
        }
    }
}

/// `(greedy, sampled, loaded, greedy_only)` labels of a model's requests.
pub(super) fn method_set(drafter: Option<&DrafterFacts>) -> (String, String, String, Vec<String>) {
    let g = method_label(drafter, false);
    let s = method_label(drafter, true);
    if g == s {
        (g.clone(), s, g, Vec::new())
    } else {
        let loaded = format!("{g},{s}");
        (g.clone(), s, loaded, vec![g])
    }
}

/// Accept sequence of a request: one base-32 digit per decode launch, the tokens the launch
/// committed minus one (so `digit + 1` is the launch's bar in a tokens-per-step chart; 31 and
/// more share the last digit). It equals the drafts the launch accepted, except on a launch cut
/// short by the stop token or the token budget.
pub(super) fn aseq(log: &[LaunchRec]) -> String {
    const DIGITS: &[u8; 32] = b"0123456789abcdefghijklmnopqrstuv";
    log.iter()
        .map(|l| DIGITS[usize::from(l.emitted.saturating_sub(1)).min(31)] as char)
        .collect()
}

pub(super) fn dtype_label(d: SynDataType) -> &'static str {
    match d {
        SynDataType::Bf16 => "BF16",
        SynDataType::Fp8_143 => "FP8",
        SynDataType::Fp8_152 => "FP8 (fp8_152)",
        SynDataType::Float => "F32",
        SynDataType::Fp16 => "FP16",
        SynDataType::Int8 => "INT8",
        SynDataType::Uint8 => "UINT8",
        SynDataType::Int32 => "INT32",
        SynDataType::Uint32 => "UINT32",
        SynDataType::Int64 => "INT64",
        SynDataType::Uint64 => "UINT64",
    }
}

/// The engine's run precision as one lower-case word (`fp8`, `bf16`).
pub(super) fn precision_word(facts: &RunFacts) -> &'static str {
    if facts.fp8_weights {
        "fp8"
    } else {
        "bf16"
    }
}

pub(super) fn jq(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn jopt(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.prec$}"),
        _ => "null".to_string(),
    }
}

/// The `precision` object of a model description.
fn precision_json(spec: &ModelSpec, facts: &RunFacts, dt: Option<&StepDtypes>) -> String {
    let matrix = dt.and_then(|d| d.matrix).map(dtype_label);
    let head = dt.and_then(|d| d.head).map(dtype_label);
    let mut weights = match (matrix, head) {
        (Some(m), Some(h)) if m != h => format!("{m} matrices, {h} embedding and readout"),
        (Some(m), _) => m.to_string(),
        // a family whose graphs are not described here: the engine's run flag is all we know
        (None, _) => precision_word(facts).to_uppercase(),
    };
    if spec.fp8_quantized && !facts.fp8_weights {
        weights.push_str(" (FP8 checkpoint dequantised at load)");
    }
    let mut out = format!("{{\"weights\":{}", jq(&weights));
    if let Some(kv) = dt.and_then(|d| d.kv) {
        out.push_str(&format!(",\"kv\":{}", jq(dtype_label(kv))));
    }
    if let Some(st) = dt.and_then(|d| d.state) {
        out.push_str(&format!(",\"state\":{}", jq(dtype_label(st))));
    }
    if let Some(m) = facts.drafter.as_ref().and_then(|d| d.matrix) {
        out.push_str(&format!(",\"drafter\":{}", jq(dtype_label(m))));
    }
    out.push('}');
    out
}

/// The model description (`model_desc`): what the model is and how it runs.
pub(super) fn model_desc_json(
    id: &str,
    checkpoint: &str,
    spec: &ModelSpec,
    n_ctx: u64,
    facts: &RunFacts,
    dt: Option<&StepDtypes>,
) -> String {
    let n_lin = spec.n_linear_layers();
    let n_full = spec.n_full_attention_layers();
    let mut out = format!(
        "{{\"v\":1,\"name\":{},\"checkpoint\":{},\"arch\":{},\"family\":{},\"kind\":{},\
         \"layers\":{},\"hidden\":{},\"attention\":{{\"full\":{n_full},\"linear\":{n_lin}}},\
         \"n_ctx\":{n_ctx},\"precision\":{}",
        jq(id),
        jq(checkpoint),
        jq(&spec.model_type),
        jq(&format!("{:?}", spec.family)),
        jq(if spec.is_moe() { "moe" } else { "dense" }),
        spec.n_layers,
        spec.hidden,
        precision_json(spec, facts, dt),
    );
    if spec.is_moe() {
        out.push_str(&format!(
            ",\"moe\":{{\"experts\":{},\"top_k\":{},\"expert_width\":{},\"form\":{}",
            spec.n_experts,
            spec.experts_per_tok,
            spec.moe_intermediate,
            jq(if facts.moe_routed {
                "routed"
            } else {
                "dense-masked"
            }),
        ));
        // every expert is a device tensor in this engine; its size comes from the step graph
        match dt.map(|d| d.mlp_bytes).filter(|&b| b > 0) {
            Some(b) => out.push_str(&format!(
                ",\"placement\":{{\"mode\":\"card\",\"card_gb\":{:.2},\"host_gb\":0}}}}",
                b as f64 / 1e9
            )),
            None => out.push_str(",\"placement\":{\"mode\":\"card\"}}"),
        }
    }
    let drafters: Vec<String> = facts
        .drafter
        .iter()
        .filter_map(|d| d.desc_name().map(jq))
        .collect();
    out.push_str(&format!(",\"drafters\":[{}]", drafters.join(",")));
    if let Some(d) = &facts.drafter {
        let ks: Vec<String> = d.draft_k.iter().map(u64::to_string).collect();
        out.push_str(&format!(
            ",\"drafter\":{{\"kind\":{},\"draft_k\":[{}],\"samples\":{}",
            jq(&d.kind),
            ks.join(","),
            d.samples
        ));
        if let Some(c) = &d.checkpoint {
            out.push_str(&format!(",\"checkpoint\":{}", jq(c)));
        }
        if let Some(t) = &d.tree {
            out.push_str(&format!(",\"tree\":{}", jq(t)));
        }
        out.push('}');
    }
    out.push_str(&format!(",\"features\":{{\"vision\":{}}}", facts.vision));
    if let Some(h) = facts.hbm_bytes {
        out.push_str(&format!(
            ",\"hbm_gb\":{{\"weights\":{:.3},\"kv\":{:.3},\"state\":{:.3},\"scratch\":{:.3},\"workspace\":{:.3}}}",
            h[0] as f64 / 1e9,
            h[1] as f64 / 1e9,
            h[2] as f64 / 1e9,
            h[3] as f64 / 1e9,
            h[4] as f64 / 1e9,
        ));
    }
    if !facts.from_engine {
        // not resident: this is the configuration the next load will use
        out.push_str(",\"planned\":true");
    }
    out.push('}');
    out
}

/// Cumulative timings of a request, as every stream chunk and the final message carry them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Timings {
    pub prompt_n: u64,
    /// The engine's prefill time: its own clock from the start of the prefill to the first token.
    pub prompt_ms: f64,
    /// Tokens generated so far, the first one included.
    pub predicted_n: u64,
    /// First token on the host to the newest launch's tokens on the host.
    pub predicted_ms: f64,
    /// Decode launches; `None` when the engine reports no launches (the field is left out).
    pub launches: Option<u64>,
    pub draft_n: u64,
    pub draft_n_accepted: u64,
}

impl Timings {
    pub fn json(&self) -> String {
        let mut out = format!(
            "{{\"cache_n\":0,\"prompt_n\":{},\"prompt_ms\":{:.2}",
            self.prompt_n, self.prompt_ms
        );
        if self.prompt_ms > 0.0 && self.prompt_n > 0 {
            out.push_str(&format!(
                ",\"prompt_per_token_ms\":{:.3},\"prompt_per_second\":{:.2}",
                self.prompt_ms / self.prompt_n as f64,
                1e3 * self.prompt_n as f64 / self.prompt_ms
            ));
        }
        out.push_str(&format!(
            ",\"predicted_n\":{},\"predicted_ms\":{:.2}",
            self.predicted_n, self.predicted_ms
        ));
        if self.predicted_ms > 0.0 && self.predicted_n > 0 {
            out.push_str(&format!(
                ",\"predicted_per_token_ms\":{:.3},\"predicted_per_second\":{:.2}",
                self.predicted_ms / self.predicted_n as f64,
                1e3 * self.predicted_n as f64 / self.predicted_ms
            ));
        }
        if let Some(l) = self.launches {
            out.push_str(&format!(
                ",\"launches\":{l},\"draft_n\":{},\"draft_n_accepted\":{}",
                self.draft_n, self.draft_n_accepted
            ));
        }
        out.push('}');
        out
    }
}

/// The `spec` section of a request that drafted: what the drafts did.
pub(super) fn spec_section_json(t: &Timings) -> Option<String> {
    if t.draft_n == 0 {
        return None;
    }
    let steps = t.launches.unwrap_or(0);
    let per_step = if steps > 0 {
        // decode tokens per launch: the first token came from the prefill's readout
        Some(t.predicted_n.saturating_sub(1) as f64 / steps as f64)
    } else {
        None
    };
    Some(format!(
        "{{\"drafted\":{},\"accepted\":{},\"rate\":{:.3},\"steps\":{steps},\"tok_per_step\":{}}}",
        t.draft_n,
        t.draft_n_accepted,
        t.draft_n_accepted as f64 / t.draft_n as f64,
        jopt(per_step, 3),
    ))
}

/// The `record` of a `live` event: the request's rolling numbers and the engine's `profile`.
pub(super) fn live_record_json(
    phase: &str,
    method: &str,
    sum: Option<&RequestSummary>,
    hbm_bytes: Option<[u64; 5]>,
    profile: Option<&str>,
) -> String {
    let mut out = format!("{{\"phase\":{},\"method\":{}", jq(phase), jq(method));
    if let Some(s) = sum {
        out.push_str(&format!(
            ",\"engine_method\":{},\"prompt_tokens\":{},\"gen_tokens\":{},\"launches\":{}",
            jq(&s.method),
            s.prompt_tokens
                .map_or("null".to_string(), |p| p.to_string()),
            s.gen_tokens,
            s.launches,
        ));
        out.push_str(&format!(
            ",\"decode_tok_s\":{},\"decode_tok_s_run\":{},\"ttft_ms\":{}",
            jopt(s.decode_tok_s, 2),
            jopt(s.decode_tok_s_run, 2),
            jopt(s.ttft_ms, 1),
        ));
        out.push_str(&format!(
            ",\"tok_per_launch\":{},\"ms_per_launch\":{},\"accept_rate\":{}",
            jopt(s.tok_per_launch, 3),
            jopt(s.ms_per_launch, 3),
            jopt(s.accept_rate, 3),
        ));
    }
    if let Some(h) = hbm_bytes {
        out.push_str(&format!(
            ",\"hbm\":{{\"weights_gb\":{:.3},\"kv_gb\":{:.3},\"state_gb\":{:.3},\"scratch_gb\":{:.3},\"other_gb\":{:.3}}}",
            h[0] as f64 / 1e9,
            h[1] as f64 / 1e9,
            h[2] as f64 / 1e9,
            h[3] as f64 / 1e9,
            h[4] as f64 / 1e9,
        ));
    }
    out.push_str(&format!(",\"profile\":{}}}", profile.unwrap_or("null")));
    out
}

/// The `vision` section of a request's details: what its images and videos were, and where the
/// time before the first token went. `host_ms` is decoding and preparation on the host (before
/// the request took the card); the tower's times are inside the prefill (`prefill_ms`), and
/// `text_prefill_ms` is the prefill without them.
pub fn vision_section_json(
    c: &crate::vision::media::MediaCounts,
    host_ms: f64,
    tower: Option<&crate::vision::VisionReport>,
    prefill_ms: f64,
) -> String {
    let mut out = format!(
        "{{\"images\":{},\"videos\":{},\"frames\":{},\"inputs\":{},\"patches\":{},\"tokens\":{},\"host_ms\":{host_ms:.1}",
        c.images, c.videos, c.frames, c.inputs, c.patches, c.tokens
    );
    if let Some(v) = tower {
        let in_prefill = v.prep_ms + v.encode_ms + v.compile_ms + v.rope_ms;
        out.push_str(&format!(
            ",\"upload_ms\":{:.1},\"tower_ms\":{:.1},\"compile_ms\":{:.1},\"rope_ms\":{:.1},\"rope_rows\":{},\
             \"text_prefill_ms\":{:.1}",
            v.prep_ms,
            v.encode_ms,
            v.compile_ms,
            v.rope_ms,
            v.rope_rows,
            (prefill_ms - in_prefill).max(0.0),
        ));
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    fn dspine(ks: &[u64]) -> DrafterFacts {
        DrafterFacts {
            kind: "dspine".into(),
            draft_k: ks.to_vec(),
            samples: true,
            tree: None,
            matrix: Some(SynDataType::Bf16),
            checkpoint: Some("Model-DSpine-b16".into()),
        }
    }

    fn spec_27b(fp8: bool) -> ModelSpec {
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

    #[test]
    fn method_labels_follow_the_drafter_and_the_request() {
        assert_eq!(method_label(None, false), "0");
        assert_eq!(method_label(None, true), "0");
        assert_eq!(method_label(Some(&dspine(&[15, 7])), false), "ds-ap");
        assert_eq!(method_label(Some(&dspine(&[15, 7])), true), "ds-ap");
        assert_eq!(method_label(Some(&dspine(&[15])), false), "ds15p");
        let mut df = dspine(&[7]);
        df.kind = "dflash2".into();
        assert_eq!(method_label(Some(&df), false), "df7p");
        let mut mtp = dspine(&[4]);
        mtp.kind = "mtp".into();
        mtp.samples = false;
        assert_eq!(method_label(Some(&mtp), false), "4");
        assert_eq!(method_label(Some(&mtp), true), "0", "a greedy-only drafter");
        mtp.draft_k = vec![8, 5, 4, 3, 2];
        assert_eq!(method_label(Some(&mtp), false), "adaptive");

        assert_eq!(
            method_set(Some(&dspine(&[15, 7]))),
            ("ds-ap".into(), "ds-ap".into(), "ds-ap".into(), vec![])
        );
        assert_eq!(
            method_set(Some(&mtp)),
            (
                "adaptive".into(),
                "0".into(),
                "adaptive,0".into(),
                vec!["adaptive".to_string()]
            )
        );
        assert_eq!(method_set(None).2, "0");
    }

    #[test]
    fn aseq_has_one_digit_per_launch_and_sums_to_the_decode_tokens() {
        let rec = |emitted: u16, accepted: u16| LaunchRec {
            decode_ms: 0.0,
            emitted,
            drafted: 15,
            accepted,
        };
        // the last launch accepted 6 drafts but the budget cut it after 2 tokens
        let log = [rec(1, 0), rec(16, 15), rec(8, 7), rec(40, 31), rec(2, 6)];
        let s = aseq(&log);
        assert_eq!(s, "0f7v1");
        assert_eq!(s.len(), log.len());
        let bars: u32 = s.chars().take(3).map(|c| c.to_digit(32).unwrap() + 1).sum();
        assert_eq!(bars, 1 + 16 + 8);
        assert_eq!(aseq(&[]), "");
    }

    #[test]
    fn model_description_states_the_run_precision_and_the_drafter() {
        let spec = spec_27b(true);
        let dt = crate::engine::qwen3_traffic::step_dtypes(&spec).unwrap();
        let facts = RunFacts {
            from_engine: true,
            fp8_weights: true,
            moe_routed: false,
            drafter: Some(dspine(&[15, 7])),
            vision: false,
            hbm_bytes: Some([
                40_000_000_000,
                9_000_000_000,
                2_000_000_000,
                1_000_000_000,
                500_000_000,
            ]),
        };
        let v = json::parse_json(&model_desc_json(
            "Qwen3.8-27B-FP8",
            "Qwen3.8-27B-FP8",
            &spec,
            131072,
            &facts,
            Some(&dt),
        ))
        .expect("valid JSON");
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
        assert_eq!(s("kind").as_deref(), Some("dense"));
        assert_eq!(s("arch").as_deref(), Some("qwen3_5"));
        assert_eq!(v.get("n_ctx").and_then(|x| x.as_u64()), Some(131072));
        let att = v.get("attention").unwrap();
        assert_eq!(att.get("full").and_then(|x| x.as_u64()), Some(16));
        assert_eq!(att.get("linear").and_then(|x| x.as_u64()), Some(48));
        let p = v.get("precision").unwrap();
        let w = p.get("weights").and_then(|x| x.as_str()).unwrap();
        assert!(
            w.starts_with("FP8 "),
            "the first word names the matrices: {w}"
        );
        assert!(w.contains("BF16 embedding"));
        assert_eq!(p.get("kv").and_then(|x| x.as_str()), Some("BF16"));
        assert_eq!(p.get("state").and_then(|x| x.as_str()), Some("F32"));
        assert_eq!(p.get("drafter").and_then(|x| x.as_str()), Some("BF16"));
        let d = v.get("drafters").and_then(|x| x.as_array()).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].as_str(), Some("dspine"));
        assert!(v.get("moe").is_none());
        assert!(v.get("planned").is_none());
        assert!(v.get("hbm_gb").is_some());

        // not resident, no drafter, BF16 run of an FP8 checkpoint
        let mut spec_b = spec_27b(false);
        spec_b.fp8_quantized = true;
        let dt_b = crate::engine::qwen3_traffic::step_dtypes(&spec_b).unwrap();
        let planned = RunFacts {
            from_engine: false,
            fp8_weights: false,
            moe_routed: false,
            drafter: None,
            vision: false,
            hbm_bytes: None,
        };
        let v = json::parse_json(&model_desc_json(
            "m",
            "m",
            &spec_b,
            4096,
            &planned,
            Some(&dt_b),
        ))
        .unwrap();
        assert_eq!(v.get("planned").and_then(|x| x.as_bool()), Some(true));
        assert_eq!(
            v.get("drafters").and_then(|x| x.as_array()).unwrap().len(),
            0
        );
        assert!(v.get("drafter").is_none() && v.get("hbm_gb").is_none());
        let w = v
            .get("precision")
            .and_then(|p| p.get("weights"))
            .and_then(|x| x.as_str())
            .unwrap();
        assert!(w.starts_with("BF16") && w.contains("dequantised"));
        assert!(v.get("precision").unwrap().get("drafter").is_none());
    }

    #[test]
    fn timings_and_sections_leave_out_what_was_not_measured() {
        let t = Timings {
            prompt_n: 100,
            prompt_ms: 50.0,
            predicted_n: 41,
            predicted_ms: 400.0,
            launches: Some(10),
            draft_n: 110,
            draft_n_accepted: 30,
        };
        let v = json::parse_json(&t.json()).unwrap();
        assert_eq!(v.get("launches").and_then(|x| x.as_u64()), Some(10));
        assert_eq!(v.get("draft_n_accepted").and_then(|x| x.as_u64()), Some(30));
        assert!(
            (v.get("prompt_per_second").and_then(|x| x.as_f64()).unwrap() - 2000.0).abs() < 1e-6
        );
        let s = json::parse_json(&spec_section_json(&t).unwrap()).unwrap();
        assert_eq!(s.get("drafted").and_then(|x| x.as_u64()), Some(110));
        assert!((s.get("tok_per_step").and_then(|x| x.as_f64()).unwrap() - 4.0).abs() < 1e-9);
        assert!((s.get("rate").and_then(|x| x.as_f64()).unwrap() - 30.0 / 110.0).abs() < 1e-3);

        // an engine without a launch feed, before the first token
        let t0 = Timings {
            prompt_n: 100,
            ..Timings::default()
        };
        let v = json::parse_json(&t0.json()).unwrap();
        for absent in [
            "launches",
            "draft_n",
            "prompt_per_second",
            "predicted_per_second",
        ] {
            assert!(v.get(absent).is_none(), "{absent}");
        }
        assert!(
            spec_section_json(&t0).is_none(),
            "a request that drafted nothing has no spec section"
        );

        let rec = json::parse_json(&live_record_json("decode", "0", None, None, None)).unwrap();
        assert!(rec.get("decode_tok_s").is_none() && rec.get("hbm").is_none());
        assert!(matches!(rec.get("profile"), Some(json::JsonValue::Null)));
    }
}
