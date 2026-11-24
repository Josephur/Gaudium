//! K/V cache element type and the FP8 scale table of the trunk's attention: the Rust shell of
//! `verified/kv_cache.bend`.
//!
//! The FP8 attention kernels (`attn_prep` / `attn_softmax_chunk` / `attn_softmax_fresh` /
//! `kv_commit`, `-DKV8` builds) store q, K, V and the softmax's P as Gaudi 2 `fp8_143` (E4M3, IEEE
//! style: largest finite value 240, exponent 15 is Inf / NaN) and run the QK / PV GEMMs on FP8
//! operands with FP32 accumulation. Every tensor is scaled into the format by a power of two
//! before its conversion, so a multiplier changes no value but the ones it pushes past 240
//! (clipped) or below 2^-9 (lost). One table per layer, `L<l>_kv8` f32 {32}, carries them:
//!
//! | index | value |
//! |---|---|
//! | `g` (0..4) | q multiplier of kv group `g`, the attention scale 1/sqrt(head_dim) folded in |
//! | `4 + g` | K multiplier |
//! | `8 + g` | V multiplier |
//! | `12 + g` | P multiplier |
//! | 16 | score scale: 1 / (q x K multiplier), the softmax's unit of a raw QK sum |
//! | 17 | stats scale: P x V multiplier, which the combine's division by the sums cancels |
//!
//! Where the multipliers come from:
//! - **q and K**: from a bound the RMSNorm in front of them guarantees. A normed row has
//!   `sum(x_c^2) = head_dim` (the epsilon only lowers it), so `|x_c| <= sqrt(head_dim)`, also for
//!   the pairs RoPE rotates; times the norm weight, `|k_c| <= sqrt(head_dim) * max|kn_w|` and
//!   `|q_c / sqrt(head_dim)| <= max|qn_w|`. Neither can saturate, whatever the input (text or
//!   image), and the values a bound-derived multiplier leaves below the FP8 floor are thousands
//!   of times smaller than a typical element.
//! - **V**: no such bound exists (no norm after the projection), so V is calibrated: the largest
//!   |V| per layer and kv head over a set of prompts (`gaudium --kv-calibrate`), with one bit of
//!   margin. A layer takes one V multiplier, the smallest of its heads': the kernels' statistics
//!   carry one P x V scale per layer, and keeping P's multiplier the same for every group keeps P
//!   (the tensor with the widest range, all of it in [0, 1]) as far from the FP8 floor as it gets.
//! - **P**: its largest value is exactly 1 (the row maximum's exponential), so it takes the largest
//!   power of two the format holds at 1.0 without a margin.
//!
//! The multipliers are powers of two, so their products are exact in f32 and the two scales cancel
//! them exactly. The exponent rules (no clipping, the cap, the layer's V multiplier) are the
//! verified functions; this module only reads and writes their inputs and outputs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use crate::verified::{KvDecode, KvDtype};
use crate::verified::{
    kv8_decode_ok, kv8_fit, kv8_min4, kv8_shape_ok, kv_elem_bytes, kv_is_fp8, kv_row_bytes,
};

/// The kernels' FP8 vector: one 256-channel cache row.
pub const KV8_ROW: u64 = 256;
/// Kv groups the table holds per kind.
pub const KV8_SLOTS: u64 = 4;
/// Entries of one layer's table (`L<l>_kv8` f32 {32}; the kernels read 0..18).
pub const KV8_TABLE_LEN: u64 = 32;
/// File name of a model's K/V scale table in its `<model_dir>-gaudium-pc` directory.
pub const KV_TABLE_FILE: &str = "kv_scales.txt";

/// f32 exponent bias: a field e stands for 2^(e - 127).
const F32_BIAS: u64 = 127;
/// Exponent field of the Gaudi 2 `fp8_143` maximum (240 = 1.875 * 2^7).
const FP8_143_MAX_FIELD: u64 = 134;
/// The fit rule's `top`: the format's largest exponent field plus the bias.
const FP8_TOP: u64 = FP8_143_MAX_FIELD + F32_BIAS;
/// Margin bits of a bound (q, K: covers the kernel's f32 rounding of the norm).
const MARGIN_BOUND: u64 = 1;
/// Margin bits of a calibrated maximum (V: a later input may exceed what calibration saw).
const MARGIN_CALIBRATED: u64 = 1;
/// P's maximum is exactly 1.0: no margin.
const MARGIN_EXACT: u64 = 0;
/// No multiplier above 2^24 (a head calibrated as all zero is not blown up past that).
const CAP_FIELD: u64 = F32_BIAS + 24;

/// Parses a `--kv-cache` value.
pub fn parse_kv_dtype(s: &str) -> Option<KvDtype> {
    match s.trim().to_ascii_lowercase().as_str() {
        "bf16" => Some(KvDtype::KvBf16),
        "fp8" => Some(KvDtype::KvFp8),
        _ => None,
    }
}

/// The `--kv-cache` spelling of a dtype.
pub fn kv_dtype_name(d: KvDtype) -> &'static str {
    if kv_is_fp8(d) {
        "fp8"
    } else {
        "bf16"
    }
}

/// Bytes of one cache element.
pub fn elem_bytes(d: KvDtype) -> u64 {
    kv_elem_bytes(d)
}

/// Bytes of one position of one layer's K (or V) cache.
pub fn row_bytes(heads: u64, head_dim: u64, d: KvDtype) -> u64 {
    kv_row_bytes(heads, head_dim, d)
}

/// The FP8 attention kernels take this head shape.
pub fn check_fp8_shape(head_dim: u64, n_kv_heads: u64) -> Result<(), String> {
    if kv8_shape_ok(head_dim, n_kv_heads, KV8_ROW, KV8_SLOTS) {
        Ok(())
    } else {
        Err(format!(
            "FP8 K/V needs {KV8_ROW}-channel heads and 1..={KV8_SLOTS} kv heads (this model: {head_dim}-channel heads, {n_kv_heads} kv heads)"
        ))
    }
}

/// The decode form may run on FP8 K/V.
pub fn check_fp8_decode(form: KvDecode, what: &str) -> Result<(), String> {
    if kv8_decode_ok(form) {
        Ok(())
    } else {
        Err(format!(
            "FP8 K/V has not been validated with {what}: its tokens are not yet known to match FP8-K/V K=0 (run it with --kv-cache bf16, or without the drafter)"
        ))
    }
}

/// Default location of a model's K/V scale table.
pub fn default_table_path(model_dir: &Path) -> PathBuf {
    let name = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    model_dir
        .with_file_name(format!("{name}-gaudium-pc"))
        .join(KV_TABLE_FILE)
}

// ============================================================================
// The calibration file
// ============================================================================

/// Largest |K| and |V| per kv head of one layer, as calibration saw them in the BF16 caches.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerAmax {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// A model's K/V calibration: per full-attention layer and kv head, the largest magnitudes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvCalibration {
    pub layers: BTreeMap<usize, LayerAmax>,
    /// Free-text header lines (what was measured, on what).
    pub notes: Vec<String>,
}

impl KvCalibration {
    /// Folds one more measurement of layer `l` in (per-head maxima).
    pub fn observe(&mut self, l: usize, k: &[f32], v: &[f32]) {
        let e = self.layers.entry(l).or_insert_with(|| LayerAmax {
            k: vec![0.0; k.len()],
            v: vec![0.0; v.len()],
        });
        for (a, &b) in e.k.iter_mut().zip(k) {
            *a = a.max(b);
        }
        for (a, &b) in e.v.iter_mut().zip(v) {
            *a = a.max(b);
        }
    }

    /// The file's text.
    pub fn to_text(&self) -> String {
        let mut s = String::from(
            "# Gaudium K/V calibration: per full-attention layer and kv head, the largest |K| and |V|\n# of the trunk's BF16 caches over the calibration prompts (gaudium --kv-calibrate).\n",
        );
        for n in &self.notes {
            s.push_str("# ");
            s.push_str(n);
            s.push('\n');
        }
        for (l, a) in &self.layers {
            s.push_str(&format!("layer {l} k"));
            for x in &a.k {
                s.push_str(&format!(" {x:e}"));
            }
            s.push_str(" v");
            for x in &a.v {
                s.push_str(&format!(" {x:e}"));
            }
            s.push('\n');
        }
        s
    }

    /// Parses a file's text.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut out = Self::default();
        for (no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(n) = line.strip_prefix('#') {
                out.notes.push(n.trim().to_string());
                continue;
            }
            let bad = |why: &str| format!("line {}: {why}: '{line}'", no + 1);
            let mut it = line.split_whitespace();
            if it.next() != Some("layer") {
                return Err(bad("expected 'layer <l> k <amax..> v <amax..>'"));
            }
            let l: usize = it
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| bad("bad layer index"))?;
            if it.next() != Some("k") {
                return Err(bad("expected 'k'"));
            }
            let (mut k, mut v) = (Vec::new(), Vec::new());
            let mut in_v = false;
            for tok in it {
                if tok == "v" && !in_v {
                    in_v = true;
                    continue;
                }
                let x: f32 = tok.parse().map_err(|_| bad("bad number"))?;
                if !x.is_finite() || x < 0.0 {
                    return Err(bad("a maximum is negative or not finite"));
                }
                if in_v {
                    v.push(x);
                } else {
                    k.push(x);
                }
            }
            if !in_v || k.is_empty() || k.len() != v.len() {
                return Err(bad("expected as many V maxima as K maxima"));
            }
            if out.layers.insert(l, LayerAmax { k, v }).is_some() {
                return Err(bad("layer listed twice"));
            }
        }
        if out.layers.is_empty() {
            return Err("no layer in the table".into());
        }
        Ok(out)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read K/V scale table {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("K/V scale table {}: {e}", path.display()))
    }

    pub fn store(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, self.to_text()).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// A digest of what the multipliers are made from (the engine identity of the host tier).
    pub fn id(&self) -> String {
        let mut text = String::new();
        for (l, a) in &self.layers {
            text.push_str(&format!("{l}:{:?}:{:?};", a.k, a.v));
        }
        let d = crate::cache_ram::digest128(text.as_bytes());
        format!("{:016x}{:016x}", d[0], d[1])
    }
}

// ============================================================================
// One layer's table
// ============================================================================

/// The f32 exponent field of a positive finite maximum (0 for zero, which the cap then bounds).
fn field_of(x: f32) -> Result<u64, String> {
    if !x.is_finite() || x < 0.0 {
        return Err(format!("a maximum of {x} has no FP8 multiplier"));
    }
    Ok(u64::from((x.to_bits() >> 23) & 0xff))
}

/// 2^(e - 127) as an f32, which must be a normal number.
fn pow2_of(e: u64) -> Result<f32, String> {
    if !(1..=254).contains(&e) {
        return Err(format!("multiplier 2^{} is outside f32", e as i64 - 127));
    }
    Ok(f32::from_bits((e as u32) << 23))
}

/// Multipliers of one layer (the table's distinct values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kv8Layer {
    /// q multiplier with the attention scale folded in.
    pub q: f32,
    pub k: f32,
    pub v: f32,
    pub p: f32,
    /// 1 / (q x K multiplier, without the attention scale).
    pub score: f32,
    /// P x V multiplier.
    pub stats: f32,
}

impl Kv8Layer {
    /// The multipliers of a layer whose norm weights reach `max_qn` / `max_kn` (absolute, as the
    /// kernels read them), whose heads' calibrated V maxima are `v_amax` (1..=4 heads) and whose
    /// heads have `head_dim` channels (a power of four: the attention scale is a power of two).
    pub fn new(max_qn: f32, max_kn: f32, v_amax: &[f32], head_dim: u64) -> Result<Self, String> {
        if v_amax.is_empty() || v_amax.len() as u64 > KV8_SLOTS {
            return Err(format!("{} V maxima for one layer", v_amax.len()));
        }
        let root = (head_dim as f64).sqrt();
        if root.fract() != 0.0 || !(root as u64).is_power_of_two() {
            return Err(format!("head_dim {head_dim}: 1/sqrt(head_dim) is not a power of two"));
        }
        let root = root as f32;
        // q / sqrt(head_dim) <= max|qn_w|, K <= sqrt(head_dim) * max|kn_w| (module docs)
        let eq = kv8_fit(field_of(max_qn)?, FP8_TOP, MARGIN_BOUND, CAP_FIELD);
        let ek = kv8_fit(field_of(max_kn * root)?, FP8_TOP, MARGIN_BOUND, CAP_FIELD);
        let mut ev = Vec::with_capacity(4);
        for &a in v_amax {
            ev.push(kv8_fit(field_of(a)?, FP8_TOP, MARGIN_CALIBRATED, CAP_FIELD));
        }
        while ev.len() < 4 {
            ev.push(ev[0]);
        }
        let ev = kv8_min4(ev[0], ev[1], ev[2], ev[3]);
        let ep = kv8_fit(field_of(1.0)?, FP8_TOP, MARGIN_EXACT, CAP_FIELD);
        let (mq, mk, mv, mp) = (pow2_of(eq)?, pow2_of(ek)?, pow2_of(ev)?, pow2_of(ep)?);
        let l = Self {
            q: mq / root,
            k: mk,
            v: mv,
            p: mp,
            score: 1.0 / (mq * mk),
            stats: mp * mv,
        };
        for x in [l.q, l.score, l.stats] {
            if !x.is_normal() {
                return Err(format!("an FP8 K/V scale of {x} is not a normal f32"));
            }
        }
        Ok(l)
    }

    /// The device table (`L<l>_kv8` f32 {32}, little-endian bytes).
    pub fn table_bytes(&self) -> Vec<u8> {
        let mut t = [0.0f32; KV8_TABLE_LEN as usize];
        for g in 0..KV8_SLOTS as usize {
            t[g] = self.q;
            t[4 + g] = self.k;
            t[8 + g] = self.v;
            t[12 + g] = self.p;
        }
        t[16] = self.score;
        t[17] = self.stats;
        t.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_spellings_round_trip() {
        for d in [KvDtype::KvBf16, KvDtype::KvFp8] {
            assert_eq!(parse_kv_dtype(kv_dtype_name(d)), Some(d));
        }
        assert_eq!(parse_kv_dtype("FP8"), Some(KvDtype::KvFp8));
        assert_eq!(parse_kv_dtype("fp8e5"), None);
        assert_eq!(elem_bytes(KvDtype::KvBf16), 2);
        assert_eq!(elem_bytes(KvDtype::KvFp8), 1);
        // the 27B: 4 kv heads of 256 channels
        assert_eq!(row_bytes(4, 256, KvDtype::KvBf16), 2048);
        assert_eq!(row_bytes(4, 256, KvDtype::KvFp8), 1024);
    }

    #[test]
    fn shape_rule_takes_the_hybrid_heads_and_refuses_the_rest() {
        assert!(check_fp8_shape(256, 4).is_ok()); // Qwen3.8-27B
        assert!(check_fp8_shape(256, 2).is_ok()); // the 35B MoE's heads
        assert!(check_fp8_shape(128, 8).is_err()); // plain Qwen3: 128-channel heads
        assert!(check_fp8_shape(256, 8).is_err());
        assert!(check_fp8_shape(256, 0).is_err());
    }

    #[test]
    fn decode_forms_today() {
        assert!(check_fp8_decode(KvDecode::KdPlain, "K=0").is_ok());
        assert!(check_fp8_decode(KvDecode::KdChain, "DSpine").is_ok());
        assert!(check_fp8_decode(KvDecode::KdTree, "a draft tree").is_err());
        assert!(check_fp8_decode(KvDecode::KdOther, "MTP").is_err());
    }

    #[test]
    fn calibration_text_round_trips() {
        let mut c = KvCalibration::default();
        c.notes.push("model test".into());
        c.observe(3, &[10.0, 12.5, 9.0, 11.0], &[3.0, 4.0, 2.5, 5.0]);
        c.observe(3, &[11.0, 12.0, 9.5, 10.0], &[3.5, 3.0, 2.0, 6.0]);
        c.observe(7, &[1.0, 2.0, 3.0, 4.0], &[0.5, 0.25, 0.125, 1.0]);
        let back = KvCalibration::parse(&c.to_text()).unwrap();
        assert_eq!(back.layers, c.layers);
        assert_eq!(back.layers[&3].k, vec![11.0, 12.5, 9.5, 11.0]);
        assert_eq!(back.layers[&3].v, vec![3.5, 4.0, 2.5, 6.0]);
        assert_eq!(back.id(), c.id());
        assert!(KvCalibration::parse("layer 3 k 1 2 v 1\n").is_err());
        assert!(KvCalibration::parse("layer 3 k 1 v -1\n").is_err());
        assert!(KvCalibration::parse("# only notes\n").is_err());
        assert!(KvCalibration::parse("layer 3 k 1 v 1\nlayer 3 k 1 v 1\n").is_err());
    }

    /// Today's rule on the 27B's numbers: the multipliers and that nothing it is built from clips.
    #[test]
    fn layer_multipliers_fit_and_cancel() {
        // layer 3 of Qwen3.8-27B: max|1 + q_norm| 1.480, max|1 + k_norm| 1.754
        let l = Kv8Layer::new(1.480, 1.754, &[3.0, 5.5, 2.0, 4.0], 256).unwrap();
        // q / 16 <= 1.48 -> x64 (94.7 < 128); K <= 28.06 -> x4 (112.3 < 128);
        // V: the largest head (5.5) -> x16 (88 < 128); P: 1.0 -> x128
        assert_eq!(l.q, 64.0 / 16.0);
        assert_eq!(l.k, 4.0);
        assert_eq!(l.v, 16.0);
        assert_eq!(l.p, 128.0);
        assert_eq!(l.score, 1.0 / 256.0);
        assert_eq!(l.stats, 2048.0);
        for (amax, m) in [(1.480 * 16.0, l.q), (1.754 * 16.0, l.k), (5.5, l.v), (1.0, l.p)] {
            assert!(amax * m <= 240.0, "{amax} x {m} clips");
        }
        // the score scale undoes q x K exactly (powers of two)
        assert_eq!(l.q * 16.0 * l.k * l.score, 1.0);
        let b = l.table_bytes();
        assert_eq!(b.len(), 128);
        let at = |i: usize| f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!((at(0), at(3), at(4), at(8), at(12), at(16), at(17)), (4.0, 4.0, 4.0, 16.0, 128.0, 1.0 / 256.0, 2048.0));
        assert_eq!(at(18), 0.0);
        // a head measured as all zero is bounded by the cap, and the others still decide
        assert!(Kv8Layer::new(1.0, 1.0, &[0.0], 256).unwrap().v <= (1u32 << 24) as f32);
        assert_eq!(Kv8Layer::new(1.0, 1.0, &[0.0, 5.5], 256).unwrap().v, 16.0);
        assert!(Kv8Layer::new(1.0, 1.0, &[], 256).is_err());
        assert!(Kv8Layer::new(1.0, 1.0, &[1.0], 128).is_err());
        assert!(Kv8Layer::new(f32::NAN, 1.0, &[1.0], 256).is_err());
    }
}
