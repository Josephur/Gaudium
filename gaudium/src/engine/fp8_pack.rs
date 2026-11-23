//! Weight packer extension for block-FP8 checkpoints, FP8 device weights and MoE wide MLPs.
//!
//! [`ExtPacker`] sits in front of the BF16 [`WeightPacker`](crate::engine::packer::WeightPacker)
//! and produces the device tensors it does not know:
//!
//! | device tensor | content |
//! |---|---|
//! | `L<l>_<W>` of an FP8 graph | `fp8_143` codes `{K, N}`, one power-of-two scale per output channel |
//! | `L<l>_<W>_scale` | those scales, f32 `{N}` |
//! | `L<l>_<W>` of a BF16 graph, block-FP8 checkpoint | the checkpoint dequantised: `bf16(code * block scale)` |
//! | `L<l>_<site>_cs` / `_sa` | the site's static activation cast scale `2^k` / `2^-k`, f32 `{1}` |
//! | `L<l>_W_router`, `L<l>_W_sgate` | MoE router `[NE, H]` and shared-expert gate `[1, H]`, BF16 |
//! | MoE `L<l>_W_gate` / `W_up` / `W_down` | the wide MLP assembled from the per-expert tensors |
//!
//! Checkpoint format (official block-FP8): an `F8_E4M3` matrix `[rows, cols]` (OCP e4m3fn: the
//! exponent field 15 is finite, up to 448) with a BF16 `<name>_scale_inv` of
//! `[ceil(rows / 128), ceil(cols / 128)]`; `W[i, j] = f32(code[i, j]) * f32(scale[i / 128, j / 128])`,
//! exact in f32.
//!
//! Per-output-channel requantisation (the FP8 graphs): for device row `n` (an output channel, in
//! the packer's row layout) `s_n = 2^ceil(log2(max_k |W[n, k]| / 240))` and
//! `W8[n, k] = e4m3_rne(W[n, k] / s_n)` on the Gaudi 2 grid (max finite 240). The quantisation is
//! row-wise, so it commutes with every row permutation the kernels need (`W_qg` query / gate
//! split, `W_v` lane order). A MoE `W_down` row is the concatenation over the expert blocks, so
//! its one scale covers every expert's columns.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::thread;

use crate::engine::fp8_graph::{
    site_of_cast_scale, split_layer_name, WeightPrecision, FP8_MATRIX_SUFFIXES,
};
use crate::engine::qwen3::Qwen3Config;
use crate::fp8::{
    bf16_to_f32, cache_map_fp8, cache_store_fp8, e4m3_to_f32, f32_to_bf16_rne, load_act_scales,
    pow2_ceil, ActScale, FP8_MAX,
};
use crate::model_spec::ModelSpec;
use crate::safetensors::SafetensorsStore;
use crate::verified::{lane128, lane256};

/// Checkpoint block-scale granularity along both axes.
const BLOCK: usize = 128;

/// Round-to-nearest-even `f32 -> fp8_143` code, bit-identical to [`crate::fp8::f32_to_e4m3`]
/// (integer mantissa rounding instead of float rounding calls: this runs once per weight).
#[inline(always)]
pub fn e4m3_rne(x: f32) -> u8 {
    let u = x.to_bits();
    let sign = ((u >> 24) & 0x80) as u8;
    let a = u & 0x7FFF_FFFF;
    if a > 0x7F80_0000 {
        return 0x7F | sign; // NaN
    }
    if a >= 0x4378_0000 {
        return 0x77 | sign; // |x| >= 248 (and infinities) saturate to 240
    }
    let e = (a >> 23) as i32 - 127;
    if e < -6 {
        // below 2^-6: multiples of 2^-9; the product is exact (or far below half a step)
        let v = f32::from_bits(a) * 512.0;
        let iv = v as u32;
        let fr = v - iv as f32;
        let m = iv + u32::from(fr > 0.5 || (fr == 0.5 && (iv & 1) == 1));
        if m >= 8 {
            return sign | (1 << 3);
        }
        return sign | m as u8;
    }
    let frac = a & 0x007F_FFFF;
    let q = frac >> 20;
    let rem = frac & 0x000F_FFFF;
    let mut mant = q + u32::from(rem > 0x8_0000 || (rem == 0x8_0000 && (q & 1) == 1));
    let mut be = e + 7;
    if mant == 8 {
        mant = 0;
        be += 1;
    }
    if be > 15 || (be == 15 && mant > 6) {
        return 0x77 | sign;
    }
    sign | ((be as u8) << 3) | mant as u8
}

fn decode_lut() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = e4m3_to_f32(i as u8);
    }
    t
}

/// One checkpoint matrix `[rows, cols]`: BF16 values, or FP8 codes with their BF16 block scales.
#[derive(Clone, Copy)]
struct Src<'a> {
    data: &'a [u8],
    scale: Option<&'a [u8]>,
    cols: usize,
    scale_cols: usize,
}

impl Src<'_> {
    fn is_fp8(&self) -> bool {
        self.scale.is_some()
    }

    /// Row `r` as exact f32 values. Returns `false` when the row holds a NaN code.
    #[inline]
    fn row_f32(&self, r: usize, lut: &[f32; 256], out: &mut [f32]) -> bool {
        debug_assert_eq!(out.len(), self.cols);
        match self.scale {
            None => {
                let row = &self.data[r * self.cols * 2..(r + 1) * self.cols * 2];
                for (o, &ch) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                    *o = bf16_to_f32(u16::from_le_bytes(ch));
                }
                true
            }
            Some(sc) => {
                let q = &self.data[r * self.cols..(r + 1) * self.cols];
                let srow = &sc[(r / BLOCK) * self.scale_cols * 2..];
                let mut nan = false;
                for (kb, (qb, ob)) in q.chunks(BLOCK).zip(out.chunks_mut(BLOCK)).enumerate() {
                    let s = bf16_to_f32(u16::from_le_bytes([srow[2 * kb], srow[2 * kb + 1]]));
                    for (o, &c) in ob.iter_mut().zip(qb) {
                        nan |= (c & 0x7F) == 0x7F;
                        *o = lut[c as usize] * s;
                    }
                }
                !nan
            }
        }
    }
}

/// How the rows of one device matrix `{K, N}` (HF `[N, K]`) come from the checkpoint.
enum Layout<'a> {
    /// device row `n` = row `map[n].1` of `srcs[map[n].0]`
    Rows {
        srcs: Vec<Src<'a>>,
        map: Vec<(u32, u32)>,
    },
    /// device row `n` = row `n` of every block, concatenated (the MoE wide `W_down`)
    Blocks { srcs: Vec<Src<'a>> },
}

struct MatPlan<'a> {
    n: usize,
    k: usize,
    layout: Layout<'a>,
}

impl MatPlan<'_> {
    fn any_fp8(&self) -> bool {
        match &self.layout {
            Layout::Rows { srcs, .. } | Layout::Blocks { srcs } => srcs.iter().any(Src::is_fp8),
        }
    }

    #[inline]
    fn row_f32(&self, n: usize, lut: &[f32; 256], out: &mut [f32]) -> bool {
        match &self.layout {
            Layout::Rows { srcs, map } => {
                let (s, r) = map[n];
                srcs[s as usize].row_f32(r as usize, lut, out)
            }
            Layout::Blocks { srcs } => {
                let mut ok = true;
                let mut off = 0usize;
                for s in srcs {
                    ok &= s.row_f32(n, lut, &mut out[off..off + s.cols]);
                    off += s.cols;
                }
                ok
            }
        }
    }
}

/// Counters of one load (reported by the engine after the upload).
#[derive(Debug, Clone, Copy, Default)]
pub struct PackStats {
    pub fp8_matrices: usize,
    pub fp8_from_cache: usize,
    pub fp8_codes: u64,
    pub dequantised_matrices: usize,
    pub pack_seconds: f64,
}

/// Packer extension for FP8 checkpoints / FP8 device weights / MoE wide MLPs (module docs).
pub struct ExtPacker<'a> {
    store: &'a SafetensorsStore,
    spec: &'a ModelSpec,
    act: BTreeMap<String, ActScale>,
    cache_dir: Option<PathBuf>,
    threads: usize,
    lut: [f32; 256],
    /// per-channel scales of the matrix quantised last, for the `<W>_scale` request that follows
    memo: RefCell<Option<(String, Vec<f32>)>>,
    stats: RefCell<PackStats>,
    /// Mapped cache files whose bytes were handed out by `pack`: append-only, unmapped on drop.
    maps: RefCell<Vec<crate::safetensors::MappedShard>>,
    /// The cache files this load took a matrix from, or wrote one into: name, size and
    /// modification time of each, as they were when the load used them.
    cache_used: RefCell<std::collections::BTreeSet<String>>,
}

/// The four GEMM input sites of a decoder layer.
pub const LAYER_SITES: [&str; 4] = ["h", "core", "h2", "m"];

/// Loads a calibration table and checks that it is this model's: exactly the sites
/// `L<l>_{h, core, h2, m}` of its layers must be present.
pub fn load_act_table(path: &Path, spec: &ModelSpec) -> Result<BTreeMap<String, ActScale>, String> {
    let act = load_act_scales(path).map_err(|e| {
        format!(
            "FP8 weights need this model's activation-scale table: cannot read {} ({e}). \
             Create it with `gaudium --fp8-calibrate` (a BF16-graph prefill pass over a few prompts).",
            path.display()
        )
    })?;
    let down = if spec.is_moe() {
        crate::engine::moe::down_site(spec)
    } else {
        "m"
    };
    for l in 0..spec.n_layers {
        for s in ["h", "core", "h2", down] {
            let site = format!("L{l}_{s}");
            if !act.contains_key(&site) {
                return Err(format!(
                    "{} is not this model's activation-scale table: site {site} is missing ({} sites, {} layers)",
                    path.display(),
                    act.len(),
                    spec.n_layers
                ));
            }
        }
    }
    let foreign = format!("L{}_h", spec.n_layers);
    if act.contains_key(&foreign) {
        return Err(format!(
            "{} is not this model's activation-scale table: it has sites of more than {} layers",
            path.display(),
            spec.n_layers
        ));
    }
    Ok(act)
}

impl<'a> ExtPacker<'a> {
    /// `act`: the calibration table (empty for BF16 graphs); `cache_dir`: optional directory of
    /// quantised matrices (`<tensor>.w8` + `<tensor>.scale`, device row layout).
    pub fn new(
        store: &'a SafetensorsStore,
        spec: &'a ModelSpec,
        act: BTreeMap<String, ActScale>,
        cache_dir: Option<PathBuf>,
    ) -> Self {
        let threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 16);
        Self {
            store,
            spec,
            act,
            cache_dir,
            threads,
            lut: decode_lut(),
            memo: RefCell::new(None),
            stats: RefCell::new(PackStats::default()),
            maps: RefCell::new(Vec::new()),
            cache_used: RefCell::new(std::collections::BTreeSet::new()),
        }
    }

    pub fn stats(&self) -> PackStats {
        *self.stats.borrow()
    }

    /// Notes the two cache files of matrix `name` as this load used them.
    fn note_cache_files(&self, dir: &Path, name: &str) {
        for ext in ["w8", "scale"] {
            let file = format!("{name}.{ext}");
            let md = std::fs::metadata(dir.join(&file)).ok();
            let mtime = md
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            self.cache_used.borrow_mut().insert(format!(
                "{file}:{}:{mtime}",
                md.as_ref().map_or(0, |m| m.len())
            ));
        }
    }

    /// What this load's FP8 matrices came from, for the identity of the engine it loads: the
    /// number of cache files it used and a digest of their names, sizes and modification
    /// times (`none` for a load without a cache directory, whose matrices are quantised from
    /// the checkpoint every time). A cached matrix is uploaded as it is found, so another file
    /// under the same name is another engine.
    pub fn cache_files_id(&self) -> String {
        let used = self.cache_used.borrow();
        if used.is_empty() {
            return "none".into();
        }
        let text: Vec<&str> = used.iter().map(String::as_str).collect();
        let d = crate::cache_ram::digest128(text.join(",").as_bytes());
        format!("{}:{:016x}{:016x}", used.len(), d[0], d[1])
    }

    /// One line on what this load packed (verbose loads).
    pub fn log_summary(&self, verbose: bool) {
        let st = self.stats();
        if !verbose || (st.fp8_matrices == 0 && st.dequantised_matrices == 0) {
            return;
        }
        eprintln!(
            "[gaudium] weight packer: {} FP8 matrices ({:.2} G codes, {} from the cache{}), {} block-FP8 matrices dequantised to BF16; {:.1} s of host packing on {} threads",
            st.fp8_matrices,
            st.fp8_codes as f64 / 1e9,
            st.fp8_from_cache,
            match &self.cache_dir {
                Some(d) => format!(" {}", d.display()),
                None => String::new(),
            },
            st.dequantised_matrices,
            st.pack_seconds,
            self.threads
        );
    }

    fn src(&self, full: &str, rows: u64, cols: u64) -> Result<Src<'a>, String> {
        let t = self
            .store
            .get(full)
            .map_err(|e| format!("missing safetensors weight '{full}': {e}"))?;
        if t.shape != [rows, cols] {
            return Err(format!(
                "weight '{full}' has shape {:?}, expected [{rows}, {cols}]",
                t.shape
            ));
        }
        let data = self
            .store
            .tensor_bytes(full)
            .map_err(|e| format!("read '{full}': {e}"))?
            .ok_or_else(|| format!("weight '{full}' is lazy/unmapped"))?;
        match t.dtype.as_str() {
            "BF16" => Ok(Src {
                data,
                scale: None,
                cols: cols as usize,
                scale_cols: 0,
            }),
            "F8_E4M3" => {
                let sname = format!("{full}_scale_inv");
                let st = self
                    .store
                    .get(&sname)
                    .map_err(|e| format!("block-FP8 weight '{full}' has no '{sname}': {e}"))?;
                let want = [rows.div_ceil(BLOCK as u64), cols.div_ceil(BLOCK as u64)];
                if st.dtype != "BF16" || st.shape != want {
                    return Err(format!(
                        "'{sname}' is {} {:?}, expected BF16 {:?}",
                        st.dtype, st.shape, want
                    ));
                }
                let scale = self
                    .store
                    .tensor_bytes(&sname)
                    .map_err(|e| format!("read '{sname}': {e}"))?
                    .ok_or_else(|| format!("'{sname}' is lazy/unmapped"))?;
                Ok(Src {
                    data,
                    scale: Some(scale),
                    cols: cols as usize,
                    scale_cols: want[1] as usize,
                })
            }
            other => Err(format!(
                "weight '{full}' has dtype {other}, expected BF16 or F8_E4M3"
            )),
        }
    }

    /// The checkpoint rows behind device matrix `L<l>_<what>` (the BF16 packer's row layouts).
    fn plan(&self, l: usize, what: &str, chunked_attn: bool) -> Result<MatPlan<'a>, String> {
        let m = self.spec;
        let lp = format!("{}layers.{l}.", m.text_prefix);
        let hid = m.hidden;
        let attn = m.is_full_attention_layer(l);
        let ident = |name: String, n: u64, k: u64| -> Result<MatPlan<'a>, String> {
            let s = self.src(&name, n, k)?;
            Ok(MatPlan {
                n: n as usize,
                k: k as usize,
                layout: Layout::Rows {
                    srcs: vec![s],
                    map: (0..n as u32).map(|i| (0, i)).collect(),
                },
            })
        };
        let moe = m.is_moe();
        match what {
            // routed form: the device MLP is the shared expert alone
            "W_gate" if moe && m.moe_routed => ident(
                format!("{lp}mlp.shared_expert.gate_proj.weight"),
                m.moe_intermediate,
                hid,
            ),
            "W_up" if moe && m.moe_routed => ident(
                format!("{lp}mlp.shared_expert.up_proj.weight"),
                m.moe_intermediate,
                hid,
            ),
            "W_down" if moe && m.moe_routed => ident(
                format!("{lp}mlp.shared_expert.down_proj.weight"),
                hid,
                m.moe_intermediate,
            ),
            // routed form, stacked experts {H, 2 EF, NE}: per expert the gate rows, then the up rows
            "EW12" if moe => {
                let (ne, ef) = (m.n_experts, m.moe_intermediate);
                let mut srcs = Vec::with_capacity(2 * ne as usize);
                let mut map = Vec::with_capacity((2 * ne * ef) as usize);
                for e in 0..ne {
                    for (j, pj) in ["gate_proj", "up_proj"].iter().enumerate() {
                        srcs.push(self.src(
                            &format!("{lp}mlp.experts.{e}.{pj}.weight"),
                            ef,
                            hid,
                        )?);
                        for i in 0..ef as u32 {
                            map.push(((2 * e) as u32 + j as u32, i));
                        }
                    }
                }
                Ok(MatPlan {
                    n: (2 * ne * ef) as usize,
                    k: hid as usize,
                    layout: Layout::Rows { srcs, map },
                })
            }
            // routed form, stacked experts {EF, H, NE}: per expert its down matrix [H][EF]
            "EW3" if moe => {
                let (ne, ef) = (m.n_experts, m.moe_intermediate);
                let mut srcs = Vec::with_capacity(ne as usize);
                let mut map = Vec::with_capacity((ne * hid) as usize);
                for e in 0..ne {
                    srcs.push(self.src(
                        &format!("{lp}mlp.experts.{e}.down_proj.weight"),
                        hid,
                        ef,
                    )?);
                    for r in 0..hid as u32 {
                        map.push((e as u32, r));
                    }
                }
                Ok(MatPlan {
                    n: (ne * hid) as usize,
                    k: ef as usize,
                    layout: Layout::Rows { srcs, map },
                })
            }
            "W_gate" | "W_up" if moe => {
                let pj = if what == "W_gate" {
                    "gate_proj"
                } else {
                    "up_proj"
                };
                let (ne, ef) = (m.n_experts, m.moe_intermediate);
                let mut srcs = Vec::with_capacity(ne as usize + 1);
                for e in 0..ne {
                    srcs.push(self.src(&format!("{lp}mlp.experts.{e}.{pj}.weight"), ef, hid)?);
                }
                srcs.push(self.src(&format!("{lp}mlp.shared_expert.{pj}.weight"), ef, hid)?);
                let mut map = Vec::with_capacity(((ne + 1) * ef) as usize);
                for b in 0..=ne as u32 {
                    for i in 0..ef as u32 {
                        map.push((b, i));
                    }
                }
                Ok(MatPlan {
                    n: ((ne + 1) * ef) as usize,
                    k: hid as usize,
                    layout: Layout::Rows { srcs, map },
                })
            }
            "W_down" if moe => {
                let (ne, ef) = (m.n_experts, m.moe_intermediate);
                let mut srcs = Vec::with_capacity(ne as usize + 1);
                for e in 0..ne {
                    srcs.push(self.src(
                        &format!("{lp}mlp.experts.{e}.down_proj.weight"),
                        hid,
                        ef,
                    )?);
                }
                srcs.push(self.src(&format!("{lp}mlp.shared_expert.down_proj.weight"), hid, ef)?);
                Ok(MatPlan {
                    n: hid as usize,
                    k: ((ne + 1) * ef) as usize,
                    layout: Layout::Blocks { srcs },
                })
            }
            "W_gate" => ident(format!("{lp}mlp.gate_proj.weight"), m.intermediate, hid),
            "W_up" => ident(format!("{lp}mlp.up_proj.weight"), m.intermediate, hid),
            "W_down" => ident(format!("{lp}mlp.down_proj.weight"), hid, m.intermediate),
            "W_qkvz" if !attn => {
                let (qkv, z) = (m.gdn_qkv_dim(), m.gdn_v_heads * m.gdn_v_dim);
                let a = self.src(&format!("{lp}linear_attn.in_proj_qkv.weight"), qkv, hid)?;
                let b = self.src(&format!("{lp}linear_attn.in_proj_z.weight"), z, hid)?;
                let mut map: Vec<(u32, u32)> = (0..qkv as u32).map(|i| (0, i)).collect();
                map.extend((0..z as u32).map(|i| (1, i)));
                Ok(MatPlan {
                    n: (qkv + z) as usize,
                    k: hid as usize,
                    layout: Layout::Rows {
                        srcs: vec![a, b],
                        map,
                    },
                })
            }
            "W_o" if !attn => ident(
                format!("{lp}linear_attn.out_proj.weight"),
                hid,
                m.gdn_v_heads * m.gdn_v_dim,
            ),
            "W_qg" if attn => {
                let (nh, dh, qg) = (m.n_heads, m.head_dim, m.qg_dim());
                let s = self.src(&format!("{lp}self_attn.q_proj.weight"), qg, hid)?;
                let mut map = vec![(0u32, 0u32); qg as usize];
                if m.attn_gate {
                    // per-head [Q_h | G_h] -> [Q_0 .. Q_{NH-1} | G_0 .. G_{NH-1}]
                    for h in 0..nh {
                        for c in 0..dh {
                            map[(dh * h + c) as usize] = (0, (2 * dh * h + c) as u32);
                            map[(nh * dh + dh * h + c) as usize] =
                                (0, (2 * dh * h + dh + c) as u32);
                        }
                    }
                } else {
                    for (i, e) in map.iter_mut().enumerate() {
                        *e = (0, i as u32);
                    }
                }
                Ok(MatPlan {
                    n: qg as usize,
                    k: hid as usize,
                    layout: Layout::Rows { srcs: vec![s], map },
                })
            }
            "W_k" if attn => ident(format!("{lp}self_attn.k_proj.weight"), m.kv_dim(), hid),
            "W_v" if attn => {
                let (nkv, dh, kvd) = (m.n_kv_heads, m.head_dim, m.kv_dim());
                let s = self.src(&format!("{lp}self_attn.v_proj.weight"), kvd, hid)?;
                let mut map = vec![(0u32, 0u32); kvd as usize];
                for g in 0..nkv {
                    for c in 0..dh {
                        let dc = if !chunked_attn {
                            c
                        } else if dh == 128 {
                            lane128(c)
                        } else {
                            lane256(c)
                        };
                        map[(g * dh + dc) as usize] = (0, (g * dh + c) as u32);
                    }
                }
                Ok(MatPlan {
                    n: kvd as usize,
                    k: hid as usize,
                    layout: Layout::Rows { srcs: vec![s], map },
                })
            }
            "W_o" if attn => ident(
                format!("{lp}self_attn.o_proj.weight"),
                hid,
                m.n_heads * m.head_dim,
            ),
            _ => Err(format!(
                "no checkpoint layout for layer {l} matrix '{what}' (full attention: {attn})"
            )),
        }
    }

    /// Runs `f(row, row bytes, scratch) -> per-row value` over the rows of a matrix on the
    /// worker threads; `false` from any row aborts with an error naming NaN codes.
    fn par_rows<F>(
        &self,
        plan: &MatPlan<'_>,
        row_bytes: usize,
        dst: &mut [u8],
        per_row: &mut [f32],
        f: F,
    ) -> bool
    where
        F: Fn(usize, &mut [u8], &mut [f32]) -> (f32, bool) + Sync,
    {
        let n = plan.n;
        let threads = self.threads.clamp(1, n.max(1));
        let rows_per = n.div_ceil(threads);
        let mut all_ok = true;
        thread::scope(|s| {
            let mut handles = Vec::new();
            for (ti, (dchunk, pchunk)) in dst
                .chunks_mut(rows_per * row_bytes)
                .zip(per_row.chunks_mut(rows_per))
                .enumerate()
            {
                let f = &f;
                let k = plan.k;
                handles.push(s.spawn(move || {
                    let mut scratch = vec![0.0f32; k];
                    let mut ok = true;
                    for (i, (drow, p)) in dchunk
                        .chunks_mut(row_bytes)
                        .zip(pchunk.iter_mut())
                        .enumerate()
                    {
                        let (v, row_ok) = f(ti * rows_per + i, drow, &mut scratch);
                        *p = v;
                        ok &= row_ok;
                    }
                    ok
                }));
            }
            for h in handles {
                all_ok &= h.join().unwrap_or(false);
            }
        });
        all_ok
    }

    /// Per-output-channel requantisation of a planned matrix: `(codes [n * k], scales [n])`.
    fn quantise(&self, name: &str, plan: &MatPlan<'_>) -> Result<(Vec<u8>, Vec<f32>), String> {
        let (n, k) = (plan.n, plan.k);
        let mut w8 = vec![0u8; n * k];
        let mut scale = vec![1.0f32; n];
        let lut = &self.lut;
        let ok = self.par_rows(plan, k, &mut w8, &mut scale, |row, dst, scratch| {
            let ok = plan.row_f32(row, lut, scratch);
            let mut amax = 0.0f32;
            for &v in scratch.iter() {
                amax = amax.max(v.abs());
            }
            let s = if amax > 0.0 {
                pow2_ceil(amax / FP8_MAX)
            } else {
                1.0
            };
            let inv = 1.0 / s;
            for (d, &v) in dst.iter_mut().zip(scratch.iter()) {
                *d = e4m3_rne(v * inv);
            }
            (s, ok)
        });
        if !ok {
            return Err(format!("NaN FP8 codes in the checkpoint rows of {name}"));
        }
        Ok((w8, scale))
    }

    /// A planned matrix as bf16 (a block-FP8 source dequantised, round to nearest even).
    fn dequantise(&self, name: &str, plan: &MatPlan<'_>) -> Result<Vec<u8>, String> {
        let (n, k) = (plan.n, plan.k);
        let mut out = vec![0u8; n * k * 2];
        let mut unused = vec![0.0f32; n];
        let lut = &self.lut;
        let ok = self.par_rows(plan, k * 2, &mut out, &mut unused, |row, dst, scratch| {
            let ok = plan.row_f32(row, lut, scratch);
            for (d, &v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(scratch.iter()) {
                *d = f32_to_bf16_rne(v).to_le_bytes();
            }
            (0.0, ok)
        });
        if !ok {
            return Err(format!("NaN FP8 codes in the checkpoint rows of {name}"));
        }
        Ok(out)
    }

    /// The FP8 codes and scales of `name`, from the cache directory when it holds them.
    ///
    /// A cached matrix is mapped, not read: its codes stay in the page cache until the upload
    /// copies them into staging memory.
    fn fp8_matrix(
        &'a self,
        name: &str,
        plan: &MatPlan<'_>,
    ) -> Result<(Cow<'a, [u8]>, Vec<f32>), String> {
        // cache entry name: the routed form's shared-only MLP matrices have their own entries
        let shared_only = self.spec.moe_routed
            && ["_W_gate", "_W_up", "_W_down"]
                .iter()
                .any(|s| name.ends_with(s));
        let entry = if shared_only {
            format!("{name}_shared")
        } else {
            name.to_owned()
        };
        let name = entry.as_str();
        if let Some(dir) = &self.cache_dir {
            match cache_map_fp8(dir, name, plan.n, plan.k) {
                Ok(Some((map, scale))) => {
                    let mut st = self.stats.borrow_mut();
                    st.fp8_matrices += 1;
                    st.fp8_from_cache += 1;
                    st.fp8_codes += (plan.n * plan.k) as u64;
                    let (ptr, len) = (map.as_bytes().as_ptr(), map.as_bytes().len());
                    self.maps.borrow_mut().push(map);
                    self.note_cache_files(dir, name);
                    // SAFETY: the mapping's address does not move with the `Vec`, `maps` is
                    // append-only, and it is unmapped only when `self` is dropped, which the
                    // `&'a self` borrow puts after every use of the returned slice.
                    let codes: &'a [u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
                    return Ok((Cow::Borrowed(codes), scale));
                }
                Ok(None) => {}
                Err(e) => return Err(format!("read FP8 cache entry {name}: {e}")),
            }
        }
        let (w8, scale) = self.quantise(name, plan)?;
        if let Some(dir) = &self.cache_dir {
            cache_store_fp8(dir, name, &w8, &scale)
                .map_err(|e| format!("write FP8 cache entry {name} in {}: {e}", dir.display()))?;
            // the files a later load will find: this load and that one are the same engine
            self.note_cache_files(dir, name);
        }
        let mut st = self.stats.borrow_mut();
        st.fp8_matrices += 1;
        st.fp8_codes += (plan.n * plan.k) as u64;
        Ok((Cow::Owned(w8), scale))
    }

    /// Packs device tensor `name` when it is one of this extension's (module docs); `None` hands
    /// the name to the BF16 packer. `chunked_attn`: the BF16 packer's `W_v` lane-order switch.
    pub fn pack(&'a self, name: &str, chunked_attn: bool) -> Option<Result<Cow<'a, [u8]>, String>> {
        let (layer, suffix) = match split_layer_name(name) {
            Some((Some(l), s)) => (l, s),
            _ => return None,
        };
        let m = self.spec;
        let t0 = std::time::Instant::now();

        // MoE router and shared-expert gate: never quantised
        if m.is_moe() && matches!(suffix, "W_router" | "W_sgate") {
            let lp = format!("{}layers.{layer}.", m.text_prefix);
            let (full, rows) = if suffix == "W_router" {
                (format!("{lp}mlp.gate.weight"), m.n_experts)
            } else {
                (format!("{lp}mlp.shared_expert_gate.weight"), 1)
            };
            return Some(self.src(&full, rows, m.hidden).and_then(|s| {
                if s.is_fp8() {
                    Err(format!("'{full}' is quantised: expected a BF16 tensor"))
                } else {
                    Ok(Cow::Borrowed(s.data))
                }
            }));
        }

        // static activation cast scales of the FP8 graphs
        if m.fp8_weights {
            if let Some((site, is_cs)) = site_of_cast_scale(name) {
                return Some(match self.act.get(site) {
                    Some(a) => {
                        let v = if is_cs { a.cs() } else { a.sa() };
                        Ok(Cow::Owned(v.to_le_bytes().to_vec()))
                    }
                    None => Err(format!("no activation scale for site {site}")),
                });
            }
        }

        // per-output-channel scales: memoised by the matrix request that precedes them
        if let Some(base) = suffix.strip_suffix("_scale") {
            if !m.fp8_weights || !FP8_MATRIX_SUFFIXES.contains(&base) {
                return None;
            }
            let wname = &name[..name.len() - "_scale".len()];
            let to_bytes =
                |s: &[f32]| -> Vec<u8> { s.iter().flat_map(|v| v.to_le_bytes()).collect() };
            if let Some((mname, sc)) = self.memo.borrow().as_ref() {
                if mname == wname {
                    return Some(Ok(Cow::Owned(to_bytes(sc))));
                }
            }
            let res = self
                .plan(layer, base, chunked_attn)
                .and_then(|plan| self.fp8_matrix(wname, &plan))
                .map(|(_, sc)| Cow::Owned(to_bytes(&sc)));
            self.stats.borrow_mut().pack_seconds += t0.elapsed().as_secs_f64();
            return Some(res);
        }

        // routed MoE form: the stacked expert tensors are always BF16
        if m.is_moe() && matches!(suffix, "EW12" | "EW3") {
            let res = self
                .plan(layer, suffix, chunked_attn)
                .and_then(|plan| self.dequantise(name, &plan))
                .map(Cow::Owned);
            let mut st = self.stats.borrow_mut();
            st.dequantised_matrices += 1;
            st.pack_seconds += t0.elapsed().as_secs_f64();
            return Some(res);
        }

        if !FP8_MATRIX_SUFFIXES.contains(&suffix) {
            return None;
        }
        let plan = match self.plan(layer, suffix, chunked_attn) {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let wide_mlp = m.is_moe() && matches!(suffix, "W_gate" | "W_up" | "W_down");
        let res = if m.fp8_weights {
            self.fp8_matrix(name, &plan).map(|(w8, sc)| {
                *self.memo.borrow_mut() = Some((name.to_owned(), sc));
                w8
            })
        } else if plan.any_fp8() || wide_mlp {
            let r = self.dequantise(name, &plan).map(Cow::Owned);
            self.stats.borrow_mut().dequantised_matrices += 1;
            r
        } else {
            return None; // a BF16 matrix of a BF16 graph: the BF16 packer's own path
        };
        self.stats.borrow_mut().pack_seconds += t0.elapsed().as_secs_f64();
        Some(res)
    }
}

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// The activation-scale table a load reads: the configured file, `GAUDIUM_FP8_ACT`, or
/// `<model_dir>-gaudium-pc/act_scales_lc.txt`.
pub fn act_path_of(cfg: &Qwen3Config) -> PathBuf {
    act_path_with(cfg, &env_path)
}

/// [`act_path_of`] with the environment given (`env(name)`: the path a variable holds).
pub fn act_path_with(cfg: &Qwen3Config, env: &dyn Fn(&str) -> Option<PathBuf>) -> PathBuf {
    cfg.fp8_act
        .clone()
        .or_else(|| env("GAUDIUM_FP8_ACT"))
        .unwrap_or_else(|| default_fp8_dir(&cfg.model_dir).join(ACT_TABLE_FILE))
}

/// The optional directory of cached quantised matrices: the configured one or `GAUDIUM_FP8_CACHE`.
pub fn cache_dir_of(cfg: &Qwen3Config) -> Option<PathBuf> {
    cfg.fp8_cache
        .clone()
        .or_else(|| env_path("GAUDIUM_FP8_CACHE"))
}

/// The activation-scale table a load parsed, for the identity of the engine it loads: the
/// number of sites and a digest of every site's entry (`none` for the empty table of a BF16
/// load). It is taken from the table in memory, the one the graphs are built with, so it is
/// the same wherever the file was found (the setting, `GAUDIUM_FP8_ACT`, the default place),
/// and another one as soon as a scale differs.
pub fn act_table_id(act: &BTreeMap<String, ActScale>) -> String {
    if act.is_empty() {
        return "none".into();
    }
    let mut text = String::new();
    for (site, a) in act {
        text.push_str(&format!(
            "{site} {} {:08x} {:08x}\n",
            a.k,
            a.amax.to_bits(),
            a.rms.to_bits()
        ));
    }
    let d = crate::cache_ram::digest128(text.as_bytes());
    format!("{}:{:016x}{:016x}", act.len(), d[0], d[1])
}

/// Resolves the run precision of a load into `spec.fp8_weights` and returns the calibration
/// table of an FP8 load (empty for BF16 graphs). Runs before the device is opened, so a missing
/// or foreign table costs no card time.
pub(crate) fn resolve_precision(
    cfg: &Qwen3Config,
    spec: &mut ModelSpec,
) -> Result<BTreeMap<String, ActScale>, String> {
    let fp8 = match cfg.precision {
        WeightPrecision::Auto => spec.fp8_quantized && !cfg.fp8_calibrate,
        WeightPrecision::Bf16 => false,
        WeightPrecision::Fp8 => true,
    };
    if fp8 && cfg.fp8_calibrate {
        return Err(
            "an FP8 calibration pass measures the BF16 graphs: run it with precision bf16 or auto"
                .into(),
        );
    }
    spec.fp8_weights = fp8;
    if cfg.moe_routed && !spec.is_moe() {
        return Err("the routed MoE form needs a mixture-of-experts model".into());
    }
    // a calibration pass always measures the dense-masked graphs: its table serves both forms
    spec.moe_routed = cfg.moe_routed && !cfg.fp8_calibrate;
    if cfg.verbose && spec.is_moe() {
        eprintln!(
            "[gaudium] MoE form: {}",
            if spec.moe_routed {
                "routed (stacked BF16 experts through the vendor expert-select node; shared expert dense)"
            } else {
                "dense-masked (one wide MLP per layer: every expert block computed, router mask applied)"
            }
        );
    }
    if !fp8 {
        if cfg.verbose {
            eprintln!(
                "[gaudium] precision: BF16 weights{}{}",
                if spec.fp8_quantized {
                    " (block-FP8 checkpoint dequantised at load)"
                } else {
                    ""
                },
                if cfg.fp8_calibrate {
                    ", FP8 CALIBRATION pass"
                } else {
                    ""
                }
            );
        }
        return Ok(BTreeMap::new());
    }
    let path = act_path_of(cfg);
    let act = load_act_table(&path, spec)?;
    if cfg.verbose {
        eprintln!(
            "[gaudium] precision: FP8 weights per output channel{}, {} activation sites from {}",
            if spec.fp8_quantized {
                " (block-FP8 checkpoint requantised)"
            } else {
                " (BF16 checkpoint quantised)"
            },
            act.len(),
            path.display()
        );
    }
    Ok(act)
}

/// Default directory of a model's FP8 load files: `<model_dir>-gaudium-pc` next to the model.
pub fn default_fp8_dir(model_dir: &Path) -> PathBuf {
    let mut s = model_dir
        .as_os_str()
        .to_string_lossy()
        .trim_end_matches('/')
        .to_owned();
    s.push_str("-gaudium-pc");
    PathBuf::from(s)
}

/// File name of the activation-scale table inside an FP8 directory.
pub const ACT_TABLE_FILE: &str = "act_scales_lc.txt";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::f32_to_e4m3;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn fast_encoder_matches_the_reference_encoder() {
        // every fp8 grid value, its neighbours, midpoints, and a pseudo-random sweep
        let mut vals: Vec<f32> = vec![
            0.0,
            -0.0,
            240.0,
            244.0,
            247.99,
            248.0,
            1e9,
            -1e9,
            1.0 / 512.0,
            1.0 / 1024.0,
            3.0 / 1024.0,
            15.0 / 1024.0,
            1.0 / 64.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e-30,
        ];
        for c in 0u16..256 {
            let v = e4m3_to_f32(c as u8);
            if v.is_nan() {
                continue;
            }
            let n = e4m3_to_f32((c as u8).wrapping_add(1));
            vals.push(v);
            vals.push(f32::from_bits(v.to_bits().wrapping_add(1)));
            vals.push(f32::from_bits(v.to_bits().wrapping_sub(1)));
            if !n.is_nan() {
                vals.push((v + n) * 0.5);
                vals.push(f32::from_bits(((v + n) * 0.5).to_bits().wrapping_add(1)));
                vals.push(f32::from_bits(((v + n) * 0.5).to_bits().wrapping_sub(1)));
            }
        }
        let mut x = 0x1234_5678_9abc_def0u64;
        for _ in 0..2_000_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // magnitudes from 2^-14 to 2^9, both signs
            let e = (x % 24) as i32 - 14;
            let frac = ((x >> 8) & 0x7F_FFFF) as u32;
            let sign = ((x >> 40) & 1) as u32;
            vals.push(f32::from_bits(
                (sign << 31) | (((e + 127) as u32) << 23) | frac,
            ));
        }
        for v in vals {
            assert_eq!(
                e4m3_rne(v),
                f32_to_e4m3(v),
                "value {v:e} ({:#x})",
                v.to_bits()
            );
        }
        assert_eq!(e4m3_rne(f32::NAN) & 0x7F, 0x7F);
    }

    fn st_bytes(entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut hdr = String::from("{");
        let mut payload = Vec::new();
        for (i, (name, dtype, shape, data)) in entries.iter().enumerate() {
            if i > 0 {
                hdr.push(',');
            }
            let shape_s: Vec<String> = shape.iter().map(u64::to_string).collect();
            hdr.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                shape_s.join(","),
                payload.len(),
                payload.len() + data.len()
            ));
            payload.extend_from_slice(data);
        }
        hdr.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        out.extend_from_slice(hdr.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn tiny_moe_spec() -> ModelSpec {
        // 2 routed experts of width 128 at hidden 128: wide MLP of 3 blocks = 384 channels
        let cfg = r#"{
          "architectures": ["Qwen3_5MoeForConditionalGeneration"],
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 128, "num_hidden_layers": 4, "num_attention_heads": 16,
            "num_key_value_heads": 2, "head_dim": 256, "vocab_size": 1024,
            "full_attention_interval": 4, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 128
          }
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    #[test]
    fn wide_moe_matrices_dequantise_and_requantise_from_block_fp8() {
        let hid = 128usize;
        let ef = 128usize;
        let mut x = 7u32;
        let mut rnd = move || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 24) as u8
        };
        // codes away from NaN (0x7F / 0xFF)
        let mut codes = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    let c = rnd();
                    if c & 0x7F == 0x7F {
                        c & 0xF0
                    } else {
                        c
                    }
                })
                .collect()
        };
        let scale_of = |v: f32| f32_to_bf16_rne(v).to_le_bytes().to_vec();
        let mut entries: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
        let lp = "model.language_model.layers.0.mlp.";
        let block_scales = [0.0078125f32, 0.01171875, 0.005859375]; // exact in bf16, not all pow2
        for (b, who) in ["experts.0.", "experts.1.", "shared_expert."]
            .iter()
            .enumerate()
        {
            for pj in ["gate_proj", "up_proj", "down_proj"] {
                entries.push((
                    format!("{lp}{who}{pj}.weight"),
                    "F8_E4M3",
                    vec![128, 128],
                    codes(hid * ef),
                ));
                entries.push((
                    format!("{lp}{who}{pj}.weight_scale_inv"),
                    "BF16",
                    vec![1, 1],
                    scale_of(block_scales[b]),
                ));
            }
        }
        entries.push((
            format!("{lp}gate.weight"),
            "BF16",
            vec![2, 128],
            vec![0u8; 2 * 128 * 2],
        ));
        entries.push((
            format!("{lp}shared_expert_gate.weight"),
            "BF16",
            vec![1, 128],
            vec![1u8; 128 * 2],
        ));
        let refs: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = entries
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.clone()))
            .collect();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("gaudium_fp8pack_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.safetensors"), st_bytes(&refs)).unwrap();
        let store = SafetensorsStore::open(&dir).unwrap();

        let get = |n: &str| -> Vec<u8> {
            store
                .tensor_bytes(&format!("{lp}{n}"))
                .unwrap()
                .unwrap()
                .to_vec()
        };

        // ---- BF16 graphs: the checkpoint dequantised, wide layout ----
        let spec = tiny_moe_spec();
        let p = ExtPacker::new(&store, &spec, BTreeMap::new(), None);
        let wg = p.pack("L0_W_gate", true).unwrap().unwrap();
        assert_eq!(wg.len(), 3 * ef * hid * 2);
        let wd = p.pack("L0_W_down", true).unwrap().unwrap();
        assert_eq!(wd.len(), hid * 3 * ef * 2);
        let bf = |b: &[u8], i: usize| bf16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
        let want = |code: u8, s: f32| bf16_to_f32(f32_to_bf16_rne(e4m3_to_f32(code) * s));
        let g1 = get("experts.1.gate_proj.weight");
        let sh = get("shared_expert.gate_proj.weight");
        // W_gate row (block 1, i = 5), column 9 = expert 1's gate row 5
        assert_eq!(
            bf(&wg, (ef + 5) * hid + 9),
            want(g1[5 * hid + 9], block_scales[1])
        );
        // block 2 = the shared expert
        assert_eq!(
            bf(&wg, (2 * ef + 127) * hid + 3),
            want(sh[127 * hid + 3], block_scales[2])
        );
        // W_down row n = concat over blocks of each block's row n
        let d0 = get("experts.0.down_proj.weight");
        let ds = get("shared_expert.down_proj.weight");
        assert_eq!(
            bf(&wd, 17 * 3 * ef + 40),
            want(d0[17 * ef + 40], block_scales[0])
        );
        assert_eq!(
            bf(&wd, 17 * 3 * ef + 2 * ef + 40),
            want(ds[17 * ef + 40], block_scales[2])
        );
        assert!(p.pack("L0_W_gate_scale", true).is_none());
        assert_eq!(
            p.pack("L0_W_router", true).unwrap().unwrap().len(),
            2 * 128 * 2
        );
        assert_eq!(p.pack("L0_W_sgate", true).unwrap().unwrap().len(), 128 * 2);
        assert!(p.pack("L0_w_in", true).is_none());
        assert!(p.pack("embed_table", true).is_none());
        assert_eq!(p.stats().dequantised_matrices, 2);

        // ---- FP8 graphs: per-output-channel requantisation, cached ----
        let mut spec8 = tiny_moe_spec();
        spec8.fp8_weights = true;
        let mut act = BTreeMap::new();
        act.insert(
            "L0_h2".to_string(),
            ActScale {
                k: 3,
                amax: 12.0,
                rms: 1.0,
            },
        );
        let cache = dir.join("pc");
        let p8 = ExtPacker::new(&store, &spec8, act.clone(), Some(cache.clone()));
        let w8 = p8.pack("L0_W_down", true).unwrap().unwrap();
        let sc = p8.pack("L0_W_down_scale", true).unwrap().unwrap();
        assert_eq!(w8.len(), hid * 3 * ef);
        assert_eq!(sc.len(), hid * 4);
        for n in [0usize, 17, 127] {
            // exact f32 row, its amax, the pow2 scale and the codes
            let mut row = vec![0.0f32; 3 * ef];
            for (b, name) in ["experts.0.", "experts.1.", "shared_expert."]
                .iter()
                .enumerate()
            {
                let q = get(&format!("{name}down_proj.weight"));
                for k in 0..ef {
                    row[b * ef + k] = e4m3_to_f32(q[n * ef + k]) * block_scales[b];
                }
            }
            let amax = row.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let s = pow2_ceil(amax / FP8_MAX);
            let got_s = f32::from_le_bytes(sc[4 * n..4 * n + 4].try_into().unwrap());
            assert_eq!(got_s, s);
            assert!(s.log2().fract() == 0.0);
            for k in 0..3 * ef {
                assert_eq!(w8[n * 3 * ef + k], f32_to_e4m3(row[k] / s));
                // a requantised weight is within half a grid step of the checkpoint value
                let back = e4m3_to_f32(w8[n * 3 * ef + k]) * s;
                assert!((back - row[k]).abs() <= 0.0625 * row[k].abs().max(s * 0.015625) + 1e-12);
            }
        }
        assert_eq!(
            p8.pack("L0_h2_cs", true).unwrap().unwrap().as_ref(),
            &8.0f32.to_le_bytes()
        );
        assert_eq!(
            p8.pack("L0_h2_sa", true).unwrap().unwrap().as_ref(),
            &0.125f32.to_le_bytes()
        );
        assert!(p8.pack("L0_m_cs", true).unwrap().is_err());
        assert_eq!(p8.stats().fp8_from_cache, 0);
        // second packer: served from the cache, identical bytes
        let p8b = ExtPacker::new(&store, &spec8, act.clone(), Some(cache.clone()));
        let sc2 = p8b.pack("L0_W_down_scale", true).unwrap().unwrap();
        let w8b = p8b.pack("L0_W_down", true).unwrap().unwrap();
        assert_eq!(w8b.as_ref(), w8.as_ref());
        assert_eq!(sc2.as_ref(), sc.as_ref());
        assert_eq!(p8b.stats().fp8_from_cache, 2);
        // The load that wrote the cache and the load that read it used the same files: one
        // identity. A packer without a cache has none to name.
        let id = p8.cache_files_id();
        assert!(id.starts_with("2:"), "{id}");
        assert_eq!(p8b.cache_files_id(), id);
        let plain = ExtPacker::new(&store, &spec8, BTreeMap::new(), None);
        assert_eq!(plain.cache_files_id(), "none");
        // A cached matrix is uploaded as it is found. One that was replaced (same name, same
        // size, other codes) gives the load that uses it another identity.
        drop((w8b, sc2));
        drop(p8b);
        let codes = cache.join("L0_W_down.w8");
        let mut other = fs::read(&codes).unwrap();
        other[0] ^= 0x55;
        std::thread::sleep(std::time::Duration::from_millis(5));
        fs::write(&codes, &other).unwrap();
        let p8c = ExtPacker::new(&store, &spec8, act.clone(), Some(cache.clone()));
        let w8c = p8c.pack("L0_W_down", true).unwrap().unwrap();
        assert_eq!(w8c.as_ref(), &other[..]);
        assert_ne!(p8c.cache_files_id(), id);
        drop(w8c);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn act_table_must_cover_exactly_the_models_layers() {
        let spec = tiny_moe_spec();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("gaudium_act_{nanos}.txt"));
        let mut txt = String::new();
        for l in 0..4 {
            for s in LAYER_SITES {
                txt.push_str(&format!("L{l}_{s} 2 30 1.5\n"));
            }
        }
        fs::write(&path, &txt).unwrap();
        let t = load_act_table(&path, &spec).unwrap();
        assert_eq!(t.len(), 16);
        assert_eq!(t["L3_m"].k, 2);
        fs::write(&path, format!("{txt}L4_h 1 50 2\n")).unwrap();
        assert!(load_act_table(&path, &spec).is_err());
        fs::write(&path, txt.replace("L2_core 2 30 1.5\n", "")).unwrap();
        assert!(load_act_table(&path, &spec).is_err());
        fs::remove_file(&path).unwrap();
        assert!(load_act_table(&path, &spec).is_err());
        assert_eq!(
            default_fp8_dir(Path::new("/models/M-FP8/")),
            PathBuf::from("/models/M-FP8-gaudium-pc")
        );
    }
}
