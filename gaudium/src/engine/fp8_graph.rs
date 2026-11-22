//! FP8 (`fp8_143`) weight GEMMs on the Gaudi 2 MME: graph builder, tensor naming, and the
//! activation-scale calibrator.
//!
//! Defined numerics (per output channel, static per-site activation casts):
//!
//! ```text
//! xq {K, T} fp8  = convert_to_fp8_bf16(x {K, T} bf16, cs {1} f32)      cs = 2^k, round half to even
//! y  {N, T} bf16 = fp8_gemm_bf16(xq, W8 {K, N} fp8, sa {1} f32, sB {N} f32)
//!                = (xq . W8^T) * sa * sB[n]                            sa = 2^-k, sB[n] = 2^e_n
//! ```
//!
//! - `W8` holds one power-of-two scale per output channel (`<weight>_scale`), so the dequantising
//!   multiply is exact.
//! - A GEMM input tensor is a *site*: it is cast once (`<site>_xq`) and every GEMM reading it shares
//!   that cast (`W_gate` and `W_up` share the post-attention norm). The site's cast scale pair is
//!   the device tensors `<site>_cs` / `<site>_sa`, bound like weights and filled from a
//!   calibration table (`<site> <k> <amax> <rms>` per line, [`crate::fp8::load_act_scales`]).
//! - The device cast multiplies in bf16 before rounding, so `cs` must be a power of two.
//!
//! Which matrices are FP8 is a property of the graph under construction
//! ([`GraphBuilder::fp8_weights`], set from `ModelSpec::fp8_weights`) and of the weight's name
//! ([`is_fp8_matrix`]): the projection and MLP matrices of every decoder layer. Embeddings, the
//! language-model head, norms, convolution taps, the DeltaNet gate projection and MoE routers stay
//! BF16 / f32.

use std::collections::BTreeMap;

use crate::fp8::{bf16_to_f32, ActScale};
use crate::model_spec::ModelSpec;
use crate::synapse::device::HpuDevice;
use crate::synapse::ffi::SynDataType;
use crate::synapse::graph::{GraphBuilder, TensorId};
use crate::synapse::ops::{as_param_bytes, ConvertToFp8Params, Fp8GemmV2Params};

/// `CAST_ROUND_HALF_NE` of `ns_ConvertToFp8::Params::round_mode`.
const CAST_ROUND_HALF_NE: u32 = 0;
/// Headroom factor of the static activation scales: one binade above the calibrated maximum.
pub const ACT_MARGIN: f32 = 2.0;

/// Device precision of the projection and MLP weight matrices, chosen at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightPrecision {
    /// FP8 for a block-FP8 checkpoint, BF16 for a BF16 checkpoint.
    #[default]
    Auto,
    /// BF16 GEMMs everywhere; a block-FP8 checkpoint is dequantised at load.
    Bf16,
    /// Per-output-channel FP8 GEMMs with static activation casts (needs the model's
    /// activation-scale table).
    Fp8,
}

impl WeightPrecision {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "bf16" => Some(Self::Bf16),
            "fp8" => Some(Self::Fp8),
            _ => None,
        }
    }
}

/// Weight-name suffixes of the matrices stored as FP8 when a graph runs FP8 weights.
pub const FP8_MATRIX_SUFFIXES: [&str; 8] = [
    "W_qkvz", "W_o", "W_gate", "W_up", "W_down", "W_qg", "W_k", "W_v",
];

/// Splits a layer tensor name into its layer index and suffix: `"L12_W_o"` -> `(Some(12), "W_o")`,
/// the generic per-layer prefill form `"L_W_o"` -> `(None, "W_o")`. `None` for any other name.
pub fn split_layer_name(name: &str) -> Option<(Option<usize>, &str)> {
    // the second-half body of a two-body prefill recipe names its (generic) layer `G_`
    if let Some(suffix) = name.strip_prefix("G_") {
        return Some((None, suffix));
    }
    let rest = name.strip_prefix('L')?;
    let sep = rest.find('_')?;
    let (digits, suffix) = (&rest[..sep], &rest[sep + 1..]);
    if digits.is_empty() {
        return Some((None, suffix));
    }
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((digits.parse().ok(), suffix))
}

/// Is `name` (`L<l>_<what>` or `L_<what>`) one of the matrices an FP8 graph stores as `fp8_143`?
pub fn is_fp8_matrix(name: &str) -> bool {
    split_layer_name(name).is_some_and(|(_, what)| FP8_MATRIX_SUFFIXES.contains(&what))
}

/// Is `suffix` (the part after `L<l>_`) an FP8 auxiliary device tensor: a per-output-channel
/// weight scale (`<W>_scale`) or a site's activation cast scale (`<site>_cs`, `<site>_sa`)?
pub fn is_fp8_aux_suffix(suffix: &str) -> bool {
    if let Some(w) = suffix.strip_suffix("_scale") {
        return FP8_MATRIX_SUFFIXES.contains(&w);
    }
    suffix.ends_with("_cs") || suffix.ends_with("_sa")
}

/// Activation site of a cast-scale tensor name: `"L3_h2_cs"` -> `("L3_h2", true)`,
/// `"L3_h2_sa"` -> `("L3_h2", false)`.
pub fn site_of_cast_scale(name: &str) -> Option<(&str, bool)> {
    if let Some(site) = name.strip_suffix("_cs") {
        return Some((site, true));
    }
    name.strip_suffix("_sa").map(|site| (site, false))
}

/// Appends `y {N, T} = W {K, N}^T . x {K, T}` with an FP8 weight matrix: the site cast (created
/// once per input tensor) and the `fp8_gemm_bf16` node. Tensor names follow the BF16 builder, so
/// an FP8 graph differs from its BF16 twin only by the `_xq`, `_cs`, `_sa` and `_scale` tensors.
pub fn add_linear_fp8(
    gb: &mut GraphBuilder,
    x: TensorId,
    w_name: &str,
    k: u64,
    n: u64,
    t: u64,
    y_name: &str,
) -> TensorId {
    let site = gb.tensor_name(x).to_owned();
    let cs = gb.get_or_add_persistent_tensor(format!("{site}_cs"), SynDataType::Float, &[1]);
    let sa = gb.get_or_add_persistent_tensor(format!("{site}_sa"), SynDataType::Float, &[1]);
    let xq_name = format!("{site}_xq");
    let xq = match gb.find_persistent_tensor(&xq_name) {
        Some(id) => id,
        None => {
            let xq = gb.add_persistent_tensor(xq_name, SynDataType::Fp8_143, &[k, t]);
            let cp = ConvertToFp8Params {
                round_mode: CAST_ROUND_HALF_NE,
            };
            gb.add_node(
                "convert_to_fp8_bf16",
                format!("{site}_cast"),
                &[x, cs],
                &[xq],
                as_param_bytes(&cp),
            );
            xq
        }
    };
    let w = gb.get_or_add_persistent_tensor(w_name, SynDataType::Fp8_143, &[k, n]);
    let sb = gb.get_or_add_persistent_tensor(format!("{w_name}_scale"), SynDataType::Float, &[n]);
    let y = gb.add_persistent_tensor(y_name, SynDataType::Bf16, &[n, t]);
    let gp = Fp8GemmV2Params {
        transpose_a: 0,
        transpose_b: 1,
        is_hw_aligned: 0,
        _pad: 0,
        scale_method_hash_id: 0,
    };
    gb.add_node(
        "fp8_gemm_bf16",
        format!("{y_name}_gemm"),
        &[xq, w, sa, sb],
        &[y],
        as_param_bytes(&gp),
    );
    y
}

/// Running maxima of the GEMM input sites of a calibration pass (BF16 graphs, prefill only).
///
/// After each prefill layer launch the engine hands the layer's site buffers to
/// [`ActCalibrator::observe`]; a site's scale is `k = floor(log2(240 / (amax * ACT_MARGIN)))`.
#[derive(Debug, Default)]
pub struct ActCalibrator {
    sites: BTreeMap<String, ActScale>,
    rows_seen: u64,
    host: Vec<u8>,
}

impl ActCalibrator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds bf16 values (little-endian bytes) into `site`.
    pub fn observe_bytes(&mut self, site: &str, bf16_le: &[u8]) {
        let mut amax = 0.0f32;
        let mut ss = 0.0f64;
        let mut n = 0u64;
        for &ch in bf16_le.as_chunks::<2>().0 {
            let v = bf16_to_f32(u16::from_le_bytes(ch));
            amax = amax.max(v.abs());
            ss += f64::from(v) * f64::from(v);
            n += 1;
        }
        let rms = if n > 0 {
            (ss / n as f64).sqrt() as f32
        } else {
            0.0
        };
        let e = self.sites.entry(site.to_owned()).or_default();
        e.amax = e.amax.max(amax);
        e.rms = e.rms.max(rms);
        e.k = ActScale::compute_k(e.amax, ACT_MARGIN);
    }

    /// Reads the bf16 buffer `{features, rows}` at `hbm_addr` back from the card and folds it into
    /// `site`. The caller has synchronised the compute stream.
    pub fn observe(
        &mut self,
        dev: &HpuDevice,
        site: &str,
        hbm_addr: u64,
        features: u64,
        rows: u64,
    ) -> Result<(), String> {
        let bytes = (features * rows * 2) as usize;
        let mut host = std::mem::take(&mut self.host);
        host.resize(bytes, 0);
        let res = dev
            .copy_d2h_sync(hbm_addr, &mut host[..bytes])
            .map_err(|e| format!("calibration read of site {site}: {e}"));
        if res.is_ok() {
            self.observe_bytes(site, &host[..bytes]);
        }
        self.host = host;
        res
    }

    /// Folds the last `tail` features of every row of the buffer [`Self::observe`] read last
    /// (`{features, rows}`) into `site`: a sub-block of a wide activation as a site of its own.
    pub fn observe_tail_of_last(&mut self, site: &str, features: u64, tail: u64) {
        let host = std::mem::take(&mut self.host);
        let (f, tl) = (features as usize * 2, tail as usize * 2);
        if f > 0 && tl <= f {
            let mut sub = Vec::with_capacity(host.len() / f * tl);
            for row in host.chunks_exact(f) {
                sub.extend_from_slice(&row[f - tl..]);
            }
            self.observe_bytes(site, &sub);
        }
        self.host = host;
    }

    /// Counts the prompt rows a pass contributed (reported with the table).
    pub fn add_rows(&mut self, rows: u64) {
        self.rows_seen += rows;
    }

    pub fn rows_seen(&self) -> u64 {
        self.rows_seen
    }

    /// The calibration table gathered so far.
    pub fn table(&self) -> BTreeMap<String, ActScale> {
        self.sites.clone()
    }
}

/// Reads the four GEMM input sites of prefill layer `l` (`h`, `core`, `h2`, `m`: the first
/// `rows` rows of the layer recipe's scratch buffers) into the calibrator. The caller has
/// synchronised the compute stream after the layer's launch.
pub(crate) fn calibrate_layer(
    cal: &mut ActCalibrator,
    dev: &HpuDevice,
    spec: &ModelSpec,
    scratch: &BTreeMap<String, u64>,
    l: usize,
    rows: u64,
) -> Result<(), String> {
    let core = if spec.is_full_attention_layer(l) {
        spec.n_heads * spec.head_dim
    } else {
        spec.gdn_v_heads * spec.gdn_v_dim
    };
    let m = if spec.is_moe() {
        crate::engine::moe::wide_inter(spec)
    } else {
        spec.intermediate
    };
    for (site, features) in [
        ("h", spec.hidden),
        ("core", core),
        ("h2", spec.hidden),
        ("m", m),
    ] {
        let key = format!("L_{site}");
        let &addr = scratch
            .get(&key)
            .ok_or_else(|| format!("calibration: prefill scratch has no buffer '{key}'"))?;
        cal.observe(dev, &format!("L{l}_{site}"), addr, features, rows)?;
        if site == "m" && spec.is_moe() {
            // the shared expert's block of the wide product: the routed form's down-GEMM site
            cal.observe_tail_of_last(&format!("L{l}_ms"), m, spec.moe_intermediate);
        }
    }
    if l == 0 {
        cal.add_rows(rows);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::f32_to_bf16_rne;

    #[test]
    fn layer_names_split_and_classify() {
        assert_eq!(split_layer_name("L12_W_o"), Some((Some(12), "W_o")));
        assert_eq!(split_layer_name("L_W_down"), Some((None, "W_down")));
        assert_eq!(split_layer_name("G_W_down"), Some((None, "W_down")));
        assert!(is_fp8_matrix("G_W_gate"));
        assert_eq!(split_layer_name("Lx_W_o"), None);
        assert_eq!(split_layer_name("embed_table"), None);
        assert!(is_fp8_matrix("L3_W_qkvz"));
        assert!(is_fp8_matrix("L_W_v"));
        assert!(!is_fp8_matrix("L3_W_ba"));
        assert!(!is_fp8_matrix("L3_W_router"));
        assert!(!is_fp8_matrix("W_lm"));
        assert!(is_fp8_aux_suffix("W_gate_scale"));
        assert!(is_fp8_aux_suffix("h2_cs"));
        assert!(is_fp8_aux_suffix("core_sa"));
        assert!(!is_fp8_aux_suffix("h2_xq"));
        assert!(!is_fp8_aux_suffix("cs_in"));
        assert!(!is_fp8_aux_suffix("W_ba_scale"));
        assert_eq!(site_of_cast_scale("L3_h2_cs"), Some(("L3_h2", true)));
        assert_eq!(site_of_cast_scale("L3_m_sa"), Some(("L3_m", false)));
        assert_eq!(site_of_cast_scale("L3_m"), None);
    }

    #[test]
    fn fp8_linear_shares_one_cast_per_site() {
        let mut gb = GraphBuilder::new("t");
        let h2 = gb.add_persistent_tensor("L0_h2", SynDataType::Bf16, &[64, 4]);
        let g = add_linear_fp8(&mut gb, h2, "L0_W_gate", 64, 128, 4, "L0_gate");
        let u = add_linear_fp8(&mut gb, h2, "L0_W_up", 64, 128, 4, "L0_up");
        assert_ne!(g, u);
        let specs = gb.persistent_tensor_specs();
        let count = |n: &str| specs.iter().filter(|s| s.0 == n).count();
        assert_eq!(count("L0_h2_xq"), 1);
        assert_eq!(count("L0_h2_cs"), 1);
        assert_eq!(count("L0_h2_sa"), 1);
        assert_eq!(count("L0_W_gate_scale"), 1);
        assert_eq!(count("L0_W_up_scale"), 1);
        let xq = specs.iter().find(|s| s.0 == "L0_h2_xq").unwrap();
        assert_eq!(xq.1, SynDataType::Fp8_143);
        assert_eq!(xq.3, 64 * 4);
        let w = specs.iter().find(|s| s.0 == "L0_W_gate").unwrap();
        assert_eq!((w.1, w.3), (SynDataType::Fp8_143, 64 * 128));
    }

    #[test]
    fn calibrator_takes_the_running_maximum_with_one_binade_of_headroom() {
        let mut c = ActCalibrator::new();
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| f32_to_bf16_rne(x).to_le_bytes())
                .collect()
        };
        c.observe_bytes("L0_h", &to_bytes(&[1.0, -30.0, 2.0]));
        c.observe_bytes("L0_h", &to_bytes(&[0.5, 3.0]));
        let t = c.table();
        let a = t["L0_h"];
        assert_eq!(a.amax, 30.0);
        // 240 / (30 * 2) = 4 -> k = 2
        assert_eq!(a.k, 2);
        assert_eq!(a.cs(), 4.0);
        assert_eq!(a.sa(), 0.25);
    }
}
