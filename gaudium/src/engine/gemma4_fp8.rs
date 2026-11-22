//! FP8 device weights for the Gemma 4 engine, quantised from the BF16 checkpoint at load.
//!
//! The graphs' numerics are the shared FP8 GEMM form of `engine/fp8_graph.rs`: one power-of-two
//! scale per output channel of a weight matrix, one static power-of-two cast scale per GEMM input
//! site. This packer sits in front of the BF16 [`Gemma4Packer`] and produces the device tensors
//! an FP8 graph adds:
//!
//! | device tensor | content |
//! |---|---|
//! | `L<l>_<W>` (`W_qg`, `W_k`, `W_v`, `W_o`, `W_gate`, `W_up`, `W_down`) | `fp8_143` codes `{K, N}` |
//! | `L<l>_<W>_scale` | the `N` per-output-channel scales, f32 |
//! | `L<l>_<site>_cs` / `_sa` (`h`, `core`, `h2`, `m`) | the site's cast scale `2^k` / `2^-k`, f32 |
//!
//! Quantisation is row-wise over the device rows, so it is taken *after* the BF16 packer put the
//! projection rows in lane order: every output channel keeps the scale of its own row. What stays
//! BF16: the tied embedding / readout table, the per-layer-input matrices, every norm weight and
//! the drafter (its drafts never decide a token).
//!
//! The four sites of a layer are calibrated on the model's own BF16 graphs
//! ([`calibrate_g4_layer`], `gaudium --fp8-calibrate`).

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::engine::fp8_graph::{
    site_of_cast_scale, split_layer_name, ActCalibrator, FP8_MATRIX_SUFFIXES,
};
use crate::engine::gemma4::{G4Dims, G4NormMode};
use crate::engine::gemma4_packer::Gemma4Packer;
use crate::fp8::{cache_load_fp8, cache_store_fp8, quantize_rows_bf16, ActScale};
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::safetensors::SafetensorsStore;
use crate::synapse::device::HpuDevice;

/// `(output channels N, inputs K)` of layer `l`'s matrix `what` (device layout `{K, N}`).
pub fn g4_matrix_shape(d: &G4Dims, l: usize, what: &str) -> Option<(u64, u64)> {
    let (nkv, dh) = d.heads(d.is_global(l));
    let inter = d.layer_inter(l);
    match what {
        "W_qg" => Some((d.nh * dh, d.hid)),
        "W_k" | "W_v" => Some((nkv * dh, d.hid)),
        "W_o" => Some((d.hid, d.nh * dh)),
        "W_gate" | "W_up" => Some((inter, d.hid)),
        "W_down" => Some((d.hid, inter)),
        _ => None,
    }
}

/// The FP8 matrices layer `l` has: a layer without keys of its own has no `W_k` / `W_v`, a global
/// layer of a `k = v` model no `W_v`.
pub fn g4_layer_matrices(d: &G4Dims, l: usize) -> Vec<&'static str> {
    let mut v = vec!["W_qg"];
    if !d.is_shared(l) {
        v.push("W_k");
        if !(d.is_global(l) && d.keqv) {
            v.push("W_v");
        }
    }
    v.extend(["W_o", "W_gate", "W_up", "W_down"]);
    v
}

/// Quantises every FP8 matrix of a Gemma 4 BF16 checkpoint into `cache_dir` (`<tensor>.w8` +
/// `<tensor>.scale`), on the host alone: an FP8 load that names the directory then reads the
/// codes instead of quantising while it holds the card. Matrices already cached are kept.
pub fn prepack_fp8_cache(model_dir: &Path, cache_dir: &Path) -> Result<G4Fp8Stats, String> {
    let mut spec = ModelSpec::from_config_file(model_dir.join("config.json"))
        .map_err(|e| format!("load config.json: {e}"))?;
    spec.validate()
        .map_err(|e| format!("verified ModelSpec validation failed: {e}"))?;
    if spec.family != ModelFamily::Gemma4 {
        return Err(format!(
            "--fp8-pack quantises Gemma 4 BF16 checkpoints; this model is {:?}",
            spec.family
        ));
    }
    spec.fp8_weights = true;
    let dims = G4Dims::from_spec(&spec, 1, G4NormMode::Row)?;
    let store = SafetensorsStore::open(model_dir).map_err(|e| format!("open safetensors: {e}"))?;
    let base = Gemma4Packer::new(&store, &spec, &dims);
    let packer = G4Fp8Packer::new(&base, &dims, BTreeMap::new(), Some(cache_dir.to_path_buf()));
    for l in 0..dims.n_layers {
        for what in g4_layer_matrices(&dims, l) {
            packer.pack(&format!("L{l}_{what}"))?;
        }
    }
    Ok(packer.stats())
}

/// Counters of one FP8 load.
#[derive(Debug, Clone, Copy, Default)]
pub struct G4Fp8Stats {
    pub matrices: usize,
    pub from_cache: usize,
    pub codes: u64,
    pub seconds: f64,
}

/// Device-weight source of a Gemma 4 load: the BF16 packer, plus the FP8 tensors when the graphs
/// run FP8 weights.
pub struct G4Fp8Packer<'a, 'p> {
    base: &'p Gemma4Packer<'a>,
    dims: &'p G4Dims,
    act: BTreeMap<String, ActScale>,
    cache_dir: Option<PathBuf>,
    threads: usize,
    /// scales of the matrix quantised last, for the `<W>_scale` request that follows it
    memo: RefCell<Option<(String, Vec<f32>)>>,
    stats: RefCell<G4Fp8Stats>,
}

impl<'a, 'p> G4Fp8Packer<'a, 'p> {
    pub fn new(
        base: &'p Gemma4Packer<'a>,
        dims: &'p G4Dims,
        act: BTreeMap<String, ActScale>,
        cache_dir: Option<PathBuf>,
    ) -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 16);
        Self {
            base,
            dims,
            act,
            cache_dir,
            threads,
            memo: RefCell::new(None),
            stats: RefCell::new(G4Fp8Stats::default()),
        }
    }

    pub fn stats(&self) -> G4Fp8Stats {
        *self.stats.borrow()
    }

    /// Codes and scales of `L<l>_<what>`: from the cache directory, else quantised now.
    fn quantised(&self, name: &str, l: usize, what: &str) -> Result<(Vec<u8>, Vec<f32>), String> {
        let (n, k) = g4_matrix_shape(self.dims, l, what)
            .ok_or_else(|| format!("'{name}' is not an FP8 matrix of this model"))?;
        let (n, k) = (n as usize, k as usize);
        let t0 = Instant::now();
        if let Some(dir) = &self.cache_dir {
            if let Some(hit) = cache_load_fp8(dir, name, n, k)
                .map_err(|e| format!("read FP8 cache {}: {e}", dir.display()))?
            {
                let mut st = self.stats.borrow_mut();
                st.matrices += 1;
                st.from_cache += 1;
                st.codes += (n * k) as u64;
                st.seconds += t0.elapsed().as_secs_f64();
                return Ok(hit);
            }
        }
        let bf = self.base.pack_weight(name)?;
        if bf.len() != n * k * 2 {
            return Err(format!(
                "weight '{name}' has {} bytes, expected BF16 {n} x {k}",
                bf.len()
            ));
        }
        let vals: Vec<u16> = bf
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect();
        drop(bf);
        let (w8, scale, _) = quantize_rows_bf16(&vals, n, k, self.threads);
        if let Some(dir) = &self.cache_dir {
            cache_store_fp8(dir, name, &w8, &scale)
                .map_err(|e| format!("write FP8 cache {}: {e}", dir.display()))?;
        }
        let mut st = self.stats.borrow_mut();
        st.matrices += 1;
        st.codes += (n * k) as u64;
        st.seconds += t0.elapsed().as_secs_f64();
        Ok((w8, scale))
    }

    /// Packs device weight tensor `name` into its exact HBM byte image.
    pub fn pack(&self, name: &str) -> Result<Cow<'a, [u8]>, String> {
        if !self.dims.fp8 {
            return self.base.pack_weight(name);
        }
        let Some((Some(l), what)) = split_layer_name(name) else {
            return self.base.pack_weight(name);
        };
        if what.ends_with("_cs") || what.ends_with("_sa") {
            let (site, is_cs) = site_of_cast_scale(name).expect("suffix checked");
            let a = self.act.get(site).ok_or_else(|| {
                format!("the FP8 activation-scale table has no site '{site}' (tensor '{name}')")
            })?;
            let v = if is_cs { a.cs() } else { a.sa() };
            return Ok(Cow::Owned(v.to_le_bytes().to_vec()));
        }
        if let Some(w) = what.strip_suffix("_scale") {
            if FP8_MATRIX_SUFFIXES.contains(&w) {
                let wname = &name[..name.len() - "_scale".len()];
                let memo = self.memo.borrow_mut().take();
                let scale = match memo {
                    Some((m, s)) if m == wname => s,
                    _ => self.quantised(wname, l, w)?.1,
                };
                let mut out = Vec::with_capacity(scale.len() * 4);
                for s in scale {
                    out.extend_from_slice(&s.to_le_bytes());
                }
                return Ok(Cow::Owned(out));
            }
        }
        if FP8_MATRIX_SUFFIXES.contains(&what) {
            let (w8, scale) = self.quantised(name, l, what)?;
            *self.memo.borrow_mut() = Some((name.to_owned(), scale));
            return Ok(Cow::Owned(w8));
        }
        self.base.pack_weight(name)
    }
}

/// Reads the four GEMM input sites of prefill layer `l` (`h`, `core`, `h2`, `m`: the first `rows`
/// rows of the generic layer recipe's scratch buffers) into the calibrator. The caller has
/// synchronised the compute stream after the layer's launch.
pub(crate) fn calibrate_g4_layer(
    cal: &mut ActCalibrator,
    dev: &HpuDevice,
    d: &G4Dims,
    scratch: &BTreeMap<String, u64>,
    l: usize,
    rows: u64,
) -> Result<(), String> {
    let (_, dh) = d.heads(d.is_global(l));
    for (site, features) in [
        ("h", d.hid),
        ("core", d.nh * dh),
        ("h2", d.hid),
        ("m", d.layer_inter(l)),
    ] {
        let key = format!("L_{site}");
        let &addr = scratch
            .get(&key)
            .ok_or_else(|| format!("calibration: prefill scratch has no buffer '{key}'"))?;
        cal.observe(dev, &format!("L{l}_{site}"), addr, features, rows)?;
    }
    if l == 0 {
        cal.add_rows(rows);
    }
    Ok(())
}
