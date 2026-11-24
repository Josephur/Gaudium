//! [kv8] FP8 K/V caches of the trunk's attention (`--kv-cache fp8`).
//!
//! What changes against the BF16 graphs, layer by layer (the rules are `verified/kv_cache.bend`,
//! the table's layout and sources `core/kv_cache.rs`):
//! - the q block, the K / V caches and the softmax's P are `fp8_143`, so the QK and PV GEMMs run
//!   on FP8 operands with FP32 accumulation (raw sums: the kernels apply the scales);
//! - `attn_prep` and `attn_softmax_chunk` take the layer's scale table `L<l>_kv8` as one more,
//!   last, input (the glue picks their `-DKV8` builds from the FP8 tensors);
//! - the V projection's rows are stored in the FP8 V byte order (`kv_wv_row`), so the PV GEMM
//!   gives the attention output in the channel order the BF16 path has: everything after it is
//!   unchanged.
//!
//! Only the trunk's caches follow `--kv-cache` (`kv_layer_dtype`): an MTP head's cache and the
//! drafters' context caches stay BF16. The dtype is a property of each cache tensor, so a plan
//! that lays caches out per lane allocates either kind from the same graph shapes.

use super::*;
use crate::core::kv_cache::{
    check_fp8_decode, check_fp8_shape, default_table_path, kv_dtype_name, KvCalibration,
    Kv8Layer, KvDecode, KV8_TABLE_LEN,
};
use crate::verified::{kv_is_fp8, kv_layer_dtype, KvDtype};

/// Layer suffix of a scale table: `L<l>_kv8` (the prefill's generic `L_kv8`).
pub const KV8_SUFFIX: &str = "kv8";

/// Kernel ELFs every FP8 K/V load uses beyond the BF16 set (decode step and chain verify).
pub const KV8_ELFS: &[&str] = &["GAUDIUM_APREP8_ELF", "GAUDIUM_ASOFTC8_ELF"];
/// The prefill tiles' fast-exp softmax build on FP8 P.
pub const KV8_PREFILL_ELFS: &[&str] = &["GAUDIUM_ASOFTCX8_ELF"];
/// The draft-tree verify's builds.
pub const KV8_TREE_ELFS: &[&str] = &["GAUDIUM_APREPT8_ELF", "GAUDIUM_ASOFTCT8_ELF"];

pub fn is_kv8_suffix(suffix: &str) -> bool {
    suffix == KV8_SUFFIX
}

/// A trunk layer prefix: `L<l>_`, or the prefill's generic `L_`.
fn is_trunk_prefix(lpfx: &str) -> bool {
    lpfx.strip_prefix('L')
        .and_then(|r| r.strip_suffix('_'))
        .is_some_and(|d| d.bytes().all(|c| c.is_ascii_digit()))
}

/// The cache dtype of one attention layer and the tensors its graph needs for it.
#[derive(Debug, Clone, Copy)]
pub struct AttnKv {
    /// Element type of the q block, the K / V caches and P.
    pub dt: SynDataType,
    /// The layer's scale table (FP8 only): the last input of `attn_prep` and of the softmax.
    pub kv8: Option<TensorId>,
}

impl AttnKv {
    /// The attention layer of prefix `lpfx` in `spec`'s graphs.
    pub fn of(gb: &mut GraphBuilder, spec: &ModelSpec, lpfx: &str) -> Self {
        let d = kv_layer_dtype(is_trunk_prefix(lpfx), spec.kv_dtype);
        if kv_is_fp8(d) {
            let t = gb.get_or_add_persistent_tensor(
                format!("{lpfx}{KV8_SUFFIX}"),
                SynDataType::Float,
                &[KV8_TABLE_LEN],
            );
            Self {
                dt: SynDataType::Fp8_143,
                kv8: Some(t),
            }
        } else {
            Self {
                dt: SynDataType::Bf16,
                kv8: None,
            }
        }
    }

    /// `ins` with the scale table appended (FP8 layers).
    pub fn inputs(&self, ins: &[TensorId]) -> Vec<TensorId> {
        let mut v = ins.to_vec();
        v.extend(self.kv8);
        v
    }
}

/// The decode form a speculative configuration reads the trunk's cache with.
pub fn decode_form(sc: Option<&spec::SpecConfig>) -> (KvDecode, &'static str) {
    match sc.map(|s| &s.method) {
        None => (KvDecode::KdPlain, "K=0"),
        Some(spec::SpecMethod::DSpine { .. }) => (KvDecode::KdChain, "the DSpine drafter"),
        Some(spec::SpecMethod::Oracle { .. }) => (KvDecode::KdChain, "oracle drafts"),
        Some(spec::SpecMethod::OracleTree { .. }) => (KvDecode::KdTree, "a draft tree"),
        Some(spec::SpecMethod::Mtp { tree: Some(_), .. }) => (KvDecode::KdTree, "the MTP tree"),
        Some(spec::SpecMethod::Mtp { .. }) => (KvDecode::KdOther, "the MTP drafter"),
        Some(spec::SpecMethod::DFlash2 { .. }) => (KvDecode::KdOther, "the DFlash2 drafter"),
        Some(spec::SpecMethod::Assistant { .. }) => (KvDecode::KdOther, "an assistant drafter"),
    }
}

/// Settles the trunk's K/V dtype of a load into `spec.kv_dtype` (before any graph is built) and
/// returns the calibration its FP8 scale tables are made from, with the file it came from.
pub fn resolve(
    cfg: &Qwen3Config,
    spec: &mut ModelSpec,
) -> Result<Option<(KvCalibration, PathBuf)>, String> {
    spec.kv_dtype = KvDtype::KvBf16;
    if !kv_is_fp8(cfg.kv_dtype) {
        return Ok(None);
    }
    let refuse = |why: String| format!("--kv-cache fp8: {why}");
    if cfg.kv_calibrate || cfg.fp8_calibrate {
        return Err(refuse(
            "a calibration load measures the BF16 caches (leave --kv-cache at bf16)".into(),
        ));
    }
    if spec.is_moe() || spec.n_linear_layers() == 0 {
        return Err(refuse(format!(
            "the FP8 attention is wired for the dense hybrid Qwen3.5 / 3.8 models only (this model: {:?}{})",
            spec.family,
            if spec.is_moe() { ", mixture of experts" } else { "" }
        )));
    }
    check_fp8_shape(spec.head_dim, spec.n_kv_heads).map_err(refuse)?;
    let (form, what) = decode_form(cfg.spec.as_ref());
    check_fp8_decode(form, what).map_err(refuse)?;
    let path = cfg
        .kv_scales
        .clone()
        .unwrap_or_else(|| default_table_path(&cfg.model_dir));
    if !path.exists() {
        return Err(refuse(format!(
            "no K/V scale table at {} (create it with `gaudium --kv-calibrate --model {}`, or name one with --kv-scales)",
            path.display(),
            cfg.model_dir.display()
        )));
    }
    let cal = KvCalibration::load(&path)?;
    for l in (0..spec.n_layers as usize).filter(|&l| spec.is_full_attention_layer(l)) {
        let a = cal.layers.get(&l).ok_or_else(|| {
            refuse(format!(
                "the K/V scale table {} has no layer {l} (made for another model?)",
                path.display()
            ))
        })?;
        if a.v.len() as u64 != spec.n_kv_heads {
            return Err(refuse(format!(
                "the K/V scale table {} lists {} heads for layer {l}, the model has {}",
                path.display(),
                a.v.len(),
                spec.n_kv_heads
            )));
        }
    }
    spec.kv_dtype = KvDtype::KvFp8;
    if cfg.verbose {
        eprintln!(
            "[gaudium] K/V cache: {} (scale table {})",
            kv_dtype_name(spec.kv_dtype),
            path.display()
        );
    }
    Ok(Some((cal, path)))
}

/// The device bytes of layer `l`'s scale table (`L<l>_kv8`), from the layer's norm weights
/// (`qn_w` / `kn_w`, as packed for the kernels) and its calibrated V maxima.
pub fn table_bytes(
    spec: &ModelSpec,
    cal: &KvCalibration,
    l: usize,
    qn_w: &[u8],
    kn_w: &[u8],
) -> Result<Vec<u8>, String> {
    let max_abs = |b: &[u8]| {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs())
            .fold(0.0f32, f32::max)
    };
    let a = cal
        .layers
        .get(&l)
        .ok_or_else(|| format!("K/V scale table: no layer {l}"))?;
    let layer = Kv8Layer::new(max_abs(qn_w), max_abs(kn_w), &a.v, spec.head_dim)
        .map_err(|e| format!("K/V scale table, layer {l}: {e}"))?;
    Ok(layer.table_bytes())
}

/// Largest |x| of a run of BF16 values.
fn bf16_max_abs(b: &[u8]) -> f32 {
    b.chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]]) & 0x7fff) << 16))
        .fold(0.0f32, f32::max)
}

impl Qwen3Engine {
    /// Calibration load: reads the trunk's caches back for the first `rows` positions (whole
    /// chunks: a row past them holds an earlier calibration prompt's values or zero) and folds
    /// the per-head maxima of K and V into the running calibration.
    pub fn kv_measure(&mut self, rows: u64) -> Result<(), String> {
        let Some(mut cal) = self.kv_calib.take() else {
            return Err("not a K/V calibration load".into());
        };
        let r = self.kv_measure_into(&mut cal, rows);
        self.kv_calib = Some(cal);
        r
    }

    fn kv_measure_into(&mut self, cal: &mut KvCalibration, rows: u64) -> Result<(), String> {
        let nkv = self.spec.n_kv_heads as usize;
        let head_bytes = (KV_CHUNK * self.spec.head_dim * 2) as usize; // one {head_dim, C} batch
        let chunks = rows.div_ceil(KV_CHUNK).clamp(1, self.max_chunks) as usize;
        let piece = self.stage_io.len() - self.stage_io.len() % 2;
        unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
            .map_err(|e| format!("sync before the K/V read-back: {e}"))?;
        for l in (0..self.spec.n_layers as usize).filter(|&l| self.spec.is_full_attention_layer(l)) {
            let mut amax = [vec![0.0f32; nkv], vec![0.0f32; nkv]];
            for (kind, name) in ["kcache", "vcache"].iter().enumerate() {
                let (base, bytes) = self.kv_addr[&format!("L{l}_{name}")];
                for ch in 0..chunks {
                    for g in 0..nkv {
                        let off = ((ch * nkv + g) * head_bytes) as u64;
                        if off + head_bytes as u64 > bytes {
                            return Err(format!("L{l}_{name}: chunk {ch} lies past the cache"));
                        }
                        let mut done = 0usize;
                        while done < head_bytes {
                            let n = piece.min(head_bytes - done);
                            self.dev
                                .copy_d2h_pinned_sync(base + off + done as u64, &mut self.stage_io, n)
                                .map_err(|e| format!("read back L{l}_{name}: {e}"))?;
                            let m = bf16_max_abs(&self.stage_io.as_slice()[..n]);
                            amax[kind][g] = amax[kind][g].max(m);
                            done += n;
                        }
                    }
                }
            }
            cal.observe(l, &amax[0], &amax[1]);
        }
        Ok(())
    }

    /// The running K/V calibration of a calibration load (`None`: not one).
    pub fn kv_calibration(&self) -> Option<KvCalibration> {
        self.kv_calib.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::qwen3::prefill::{build_attn_tile_graph, PrefillOpts, TileForm};

    fn spec_27b(kv: KvDtype) -> ModelSpec {
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
        let mut s = ModelSpec::from_config_json(cfg).unwrap();
        s.kv_dtype = kv;
        s
    }

    fn tensor(gb: &GraphBuilder, name: &str) -> Option<(SynDataType, Vec<u64>, u64)> {
        gb.persistent_tensor_specs()
            .into_iter()
            .find(|t| t.0 == name)
            .map(|t| (t.1, t.2, t.3))
    }

    fn node_inputs(gb: &GraphBuilder, name: &str) -> Vec<String> {
        gb.node_io_views()
            .into_iter()
            .find(|n| n.name == name)
            .map(|n| n.inputs.iter().map(|t| t.name.to_string()).collect())
            .unwrap_or_default()
    }

    /// The decode step and the chain verify: FP8 q / caches / P, the table as the last input of
    /// the prep and of the softmax, half the cache bytes; the DeltaNet layers are untouched.
    #[test]
    fn step_graphs_carry_fp8_caches_and_their_tables() {
        let (b, f) = (spec_27b(KvDtype::KvBf16), spec_27b(KvDtype::KvFp8));
        for (gb16, gb8) in [
            (
                build_decode_step_graph(&b, 8192, 2, &[]),
                build_decode_step_graph(&f, 8192, 2, &[]),
            ),
            (
                spec::build_spec_step_graph(&b, 8192, 2, 8, 8, None, &[]),
                spec::build_spec_step_graph(&f, 8192, 2, 8, 8, None, &[]),
            ),
        ] {
            let (dt, dims, bytes) = tensor(&gb8, "L3_kcache").unwrap();
            assert_eq!((dt, dims), (SynDataType::Fp8_143, vec![256, KV_CHUNK, 8]));
            assert_eq!(bytes * 2, tensor(&gb16, "L3_kcache").unwrap().2);
            assert_eq!(tensor(&gb8, "L3_vcache").unwrap().0, SynDataType::Fp8_143);
            assert_eq!(tensor(&gb8, "L3_qb").unwrap().0, SynDataType::Fp8_143);
            assert_eq!(tensor(&gb8, "L3_P").unwrap().0, SynDataType::Fp8_143);
            assert_eq!(
                tensor(&gb8, "L3_kv8").unwrap(),
                (SynDataType::Float, vec![KV8_TABLE_LEN], KV8_TABLE_LEN * 4)
            );
            assert!(tensor(&gb16, "L3_kv8").is_none());
            assert!(tensor(&gb8, "L0_kv8").is_none(), "a DeltaNet layer has no K/V table");
            let prep = node_inputs(&gb8, "L3_attn_prep");
            assert_eq!(prep.len(), 9);
            assert_eq!(prep.last().map(String::as_str), Some("L3_kv8"));
            let soft = node_inputs(&gb8, "L3_softmax");
            assert_eq!(soft, ["L3_scores", "cursor", "L3_kv8"]);
            assert_eq!(node_inputs(&gb16, "L3_attn_prep").len(), 8);
            // a different recipe: the fingerprint covers the dtypes
            assert_ne!(gb16.fingerprint("d"), gb8.fingerprint("d"));
            // the FP8 tables are layer weights, bound like the norms
            assert!(is_weight_or_table("L3_kv8"));
        }
    }

    /// The prefill tiles, plain and fast-exp with chunk groups: FP8 caches and P with the generic
    /// table `L_kv8`; the half-width core (a BF16-cache form) is off.
    #[test]
    fn prefill_tiles_carry_fp8_caches_and_drop_the_half_width_core() {
        let f = spec_27b(KvDtype::KvFp8);
        let b = spec_27b(KvDtype::KvBf16);
        let form = TileForm::resolve(&PrefillOpts::default(), &f);
        assert_eq!(form.s16, 0);
        // 256-row tiles only (the 512-row FP8 tile is wrong past one chunk on the card)
        assert_eq!(form.tile_candidates(&f), &[256]);
        assert_eq!(TileForm::resolve(&PrefillOpts::default(), &b).tile_candidates(&b)[0], 512);
        assert!(form.extra_elfs(&f).contains(&"GAUDIUM_ASOFTCX8_ELF"));
        assert!(!form.extra_elfs(&b).contains(&"GAUDIUM_ASOFTCX8_ELF"));
        for form in [TileForm::PLAIN, form, TileForm { s16: 2, ..form }] {
            let att = build_attn_tile_graph(&f, 256, 262144, 64, &form, &[], &[]);
            assert_eq!(tensor(&att, "L_kcache").unwrap().0, SynDataType::Fp8_143);
            assert_eq!(tensor(&att, "L_vcache").unwrap().0, SynDataType::Fp8_143);
            assert!(tensor(&att, "L_kv8").is_some());
            assert!(is_generic_layer_weight("L_kv8"));
            let softmaxes: Vec<_> = att
                .node_io_views()
                .into_iter()
                .filter(|n| n.guid == "gaudium_attn_softmax_chunk")
                .collect();
            assert!(!softmaxes.is_empty());
            for n in &softmaxes {
                assert_eq!(n.inputs.last().map(|t| t.name), Some("L_kv8"));
            }
            assert!(att
                .node_io_views()
                .iter()
                .all(|n| n.guid != "gaudium_attn_softmax16"));
        }
    }

    /// Every channel of a head lands on one row of the same 128-row block, for both dtypes (a
    /// bijection: checked over the whole 256-channel domain), and the FP8 order is the BF16 order
    /// followed by the softmax's P byte order.
    #[test]
    fn v_rows_are_a_permutation_inside_each_block() {
        use crate::verified::{kv8_vpos, kv_wv_row, lane256};
        for d in [KvDtype::KvBf16, KvDtype::KvFp8] {
            let mut seen = [false; 256];
            for c in 0..256u64 {
                let r = kv_wv_row(c, 256, d);
                assert!(r < 256 && r / 128 == c / 128, "{d:?}: channel {c} -> row {r}");
                assert!(!seen[r as usize], "{d:?}: row {r} taken twice");
                seen[r as usize] = true;
            }
            let mut seen = [false; 128];
            for c in 0..128u64 {
                let r = kv_wv_row(c, 128, d) as usize;
                assert!(r < 128 && !seen[r]);
                seen[r] = true;
            }
        }
        for c in 0..256u64 {
            assert_eq!(kv_wv_row(c, 256, KvDtype::KvFp8), kv8_vpos(lane256(c)));
            assert_eq!(kv_wv_row(c, 256, KvDtype::KvBf16), lane256(c));
        }
    }

    #[test]
    fn only_trunk_prefixes_follow_the_setting() {
        for p in ["L0_", "L63_", "L_"] {
            assert!(is_trunk_prefix(p), "{p}");
        }
        for p in ["M_", "D0_", "L3", "Lx_", "G_", ""] {
            assert!(!is_trunk_prefix(p), "{p}");
        }
    }

    #[test]
    fn bf16_maxima() {
        let v: Vec<u8> = [0x3f80u16, 0xc040, 0x0000, 0x8000]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(bf16_max_abs(&v), 3.0); // 1.0, -3.0, +0, -0
    }
}
