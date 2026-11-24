//! Prefill tile recipes of the Qwen3 / Qwen3.5 engine: their throughput forms, the scratch plan
//! they share, and the per-launch profile of the tile loop.
//!
//! A prompt is prefilled tile by tile through generic single-layer recipes (one launch per layer,
//! bound to that layer's weights, caches and states). What bounds a tile on Gaudi 2 differs per
//! layer kind, so each kind has its own form here:
//!
//! - **Gated DeltaNet layers** (`mme`): the serial per-row recurrence kernel keeps one TPC busy
//!   per value head while the MME idles. The chunked form runs 64-row chunks as batched MME GEMMs
//!   (`O = (G q - P W) S + P U`, `S' = gL S + (tail k)^T (U - W S)`): the TPCs only build the
//!   chunk coefficients and invert the 16x16 diagonal blocks, the MME finishes the inverse with
//!   an exact Neumann series and carries the state through three small nodes per chunk. The state
//!   enters and leaves in the decode kernels' blob layout, so decode continues from it unchanged.
//!   The conv prep and the gated norm run row-parallel, one special-function table per kernel
//!   (`rows = 2`), instead of one index-space point per channel block.
//! - **Full-attention layers** (`fastexp`, `groups`): the chunk softmax is VPU-bound, so the tile
//!   recipes use the build with a 13-operation `exp` and an unmasked path for history chunks, and
//!   split the chunk loop into groups (`QK_g -> softmax_g -> PV_g`) so the MME multiplies one
//!   group while the TPCs sweep another.
//! - **Half-width core** (`s16`): a score written as BF16 in 1/512 octaves halves both the bytes
//!   the QK GEMM writes and the vectors the softmax sweeps (128 lanes instead of 64); `s16 = 2`
//!   also stores `P` as FP8. It changes the numerics of the prefill attention (a score keeps 8
//!   significant bits), so it is the default only where the weights are FP8 already.
//!
//! Every lever is a field of [`PrefillOpts`] (`--prefill k=v,..` or `GAUDIUM_PREFILL`); with all
//! of them off the engine builds the plain per-layer recipes of `qwen3.rs`, bit for bit.
//!
//! Scratch: a tile recipe's activations are persistent tensors (the compiler never has to place
//! them), and only one tile recipe runs at a time, so all of them are laid out in ONE planned
//! region sized by the largest recipe ([`TileRt::plan_bytes`]): nothing is allocated per recipe.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use super::{
    add_linear_bf16, add_rmsnorm, append_post_and_mlp, build_embed_graph,
    build_prefill_attn_layer_graph, build_prefill_linear_layer_graph, is_generic_layer_weight,
    Qwen3Config, KV_CHUNK,
};
use crate::kernels::KernelEnv;
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::synapse::ffi::{SynDataType, SynapseApi};
use crate::synapse::graph::{CompiledRecipe, GraphBuilder, TensorId};
use crate::synapse::ops::{as_param_bytes, u32_params, ConvertToFp8Params, TransposeParams};
use crate::synapse::recipe_cache::RecipeCache;
use crate::verified::align_up;

/// Rows of one DeltaNet chunk of the MME recurrence (the solve kernel's blocks are 16 rows; 64
/// rows is the largest chunk whose coefficient rows fit the TPC's local memory beside a table).
pub const GDN_CHUNK: u64 = 64;
/// Widest prefill tile the engine plans for.
pub const TILE_MAX: u64 = 512;
/// Tile widths tried in order when the width is not fixed: the widest whose scratch fits.
const TILE_CANDIDATES: [u64; 2] = [512, 256];
/// `CAST_ROUND_HALF_NE` of `ns_ConvertToFp8::Params::round_mode`.
const CAST_ROUND_HALF_NE: u32 = 0;

/// The prefill levers as the user sets them (`--prefill`, `GAUDIUM_PREFILL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillOpts {
    /// Rows of a full tile (multiple of 64, at most [`TILE_MAX`]); 0 = the widest that fits.
    pub tile: u64,
    /// DeltaNet recurrence as chunked MME GEMMs.
    pub mme: bool,
    /// Conv prep / gated norm kernels of the MME form: 0 = one point per channel block,
    /// 1 = row-parallel (two tables per kernel), 2 = row-parallel, one table per kernel.
    pub rows: u32,
    /// Tile attention recipes use the fast-`exp` chunk softmax and the unrolled combine.
    pub fastexp: bool,
    /// Chunk groups of a tile attention layer (1 = one QK / softmax / PV node each).
    pub groups: u64,
    /// Half-width attention core: 0 = f32 scores, 1 = BF16 scores, 2 = BF16 scores and FP8 `P`;
    /// `None` = by model (2 for dense FP8 weights, else 0).
    pub s16: Option<u32>,
    /// Synchronise after every prefill launch and report the device time per recipe class.
    pub profile: bool,
    /// Report the host-side split of the tile loop (uploads, launch calls, waiting) per request.
    pub stats: bool,
    /// Measurement: after the first request, time every half-layer recipe form over this many
    /// launches (0 = off). The engine serves nothing afterwards.
    pub half_probe: u32,
    /// Two tiles in flight: full tiles run pairwise through two-body recipes (the second half of
    /// one tile's layer with the first half of the next tile's), so the card overlaps one tile's
    /// MME work with the other's TPC work. The same nodes on the same rows: the same bits.
    pub wave: bool,
}

impl Default for PrefillOpts {
    fn default() -> Self {
        Self {
            tile: 0,
            mme: true,
            rows: 2,
            fastexp: true,
            groups: 16,
            s16: None,
            profile: false,
            stats: false,
            half_probe: 0,
            wave: true,
        }
    }
}

impl PrefillOpts {
    /// The plain per-layer recipes of `qwen3.rs` on 256-row tiles.
    pub fn plain() -> Self {
        Self {
            tile: 256,
            mme: false,
            rows: 0,
            fastexp: false,
            groups: 1,
            s16: Some(0),
            profile: false,
            stats: false,
            half_probe: 0,
            wave: false,
        }
    }

    /// Parses `k=v[,k=v..]`: `tile`, `mme`, `rows`, `fastexp`, `groups`, `s16`, `wave`, and the bare
    /// words `plain` (every lever off, 256-row tiles), `profile` and `stats`. Later entries win.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut o = Self::default();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (k, v) = match part.split_once('=') {
                Some((k, v)) => (k.trim(), Some(v.trim())),
                None => (part, None),
            };
            let num = |what: &str| -> Result<u64, String> {
                v.ok_or_else(|| format!("prefill option '{what}' needs a value"))?
                    .parse::<u64>()
                    .map_err(|_| format!("prefill option '{part}': not a number"))
            };
            match k {
                "plain" => {
                    let (profile, stats, half_probe) = (o.profile, o.stats, o.half_probe);
                    o = Self::plain();
                    o.profile = profile;
                    o.stats = stats;
                    o.half_probe = half_probe;
                }
                "halfprobe" => o.half_probe = num("halfprobe")?.min(10_000) as u32,
                "profile" => o.profile = v.is_none_or(|v| v != "0"),
                "stats" => o.stats = v.is_none_or(|v| v != "0"),
                "tile" => {
                    let t = num("tile")?;
                    if t != 0 && (t % 64 != 0 || t > TILE_MAX) {
                        return Err(format!(
                            "prefill tile {t}: must be a multiple of 64 up to {TILE_MAX} (0 = automatic)"
                        ));
                    }
                    o.tile = t;
                }
                "mme" => o.mme = num("mme")? != 0,
                "wave" => o.wave = num("wave")? != 0,
                "rows" => {
                    let r = num("rows")?;
                    if r > 2 {
                        return Err(format!("prefill rows {r}: 0, 1 or 2"));
                    }
                    o.rows = r as u32;
                }
                "fastexp" => o.fastexp = num("fastexp")? != 0,
                "groups" => o.groups = num("groups")?.max(1),
                "s16" => {
                    let s = num("s16")?;
                    if s > 2 {
                        return Err(format!("prefill s16 {s}: 0, 1 or 2"));
                    }
                    o.s16 = Some(s as u32);
                }
                other => {
                    return Err(format!(
                        "unknown prefill option '{other}' (tile, mme, rows, fastexp, groups, s16, wave, plain, profile, stats, halfprobe)"
                    ))
                }
            }
        }
        Ok(o)
    }

    /// The options of a load: `Qwen3Config::prefill`, else `GAUDIUM_PREFILL`, else the defaults.
    pub fn from_config(cfg: &Qwen3Config) -> Result<Self, String> {
        let mut o = match &cfg.prefill {
            Some(s) => Self::parse(s)?,
            None => match std::env::var("GAUDIUM_PREFILL") {
                Ok(s) => Self::parse(&s).map_err(|e| format!("GAUDIUM_PREFILL: {e}"))?,
                Err(_) => Self::default(),
            },
        };
        // an explicit narrower `Qwen3Config::prefill_tile` still caps the tile
        if cfg.prefill_tile < TILE_MAX && (o.tile == 0 || o.tile > cfg.prefill_tile) {
            o.tile = cfg.prefill_tile.max(1);
        }
        Ok(o)
    }
}

/// The recipe forms one model's tiles are built in ([`PrefillOpts`] resolved against the model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileForm {
    pub mme: bool,
    pub rows: u32,
    pub fastexp: bool,
    pub groups: u64,
    pub s16: u32,
}

impl TileForm {
    pub const PLAIN: Self = Self {
        mme: false,
        rows: 0,
        fastexp: false,
        groups: 1,
        s16: 0,
    };

    /// Resolves the levers for `spec`. The throughput forms are built for the dense families
    /// whose kernel shapes were measured: plain Qwen3 (attention levers) and the dense hybrid
    /// (all levers). MoE models keep the plain recipes.
    pub fn resolve(o: &PrefillOpts, spec: &ModelSpec) -> Self {
        let dense_hybrid = spec.family == ModelFamily::Qwen35Dense
            && !spec.is_moe()
            && spec.head_dim == 256
            && spec.gdn_k_heads == 16
            && spec.gdn_v_heads == 48;
        let plain_qwen3 = spec.family == ModelFamily::Qwen3
            && spec.n_linear_layers() == 0
            && !spec.is_moe()
            && spec.head_dim == 128;
        if !dense_hybrid && !plain_qwen3 {
            return Self::PLAIN;
        }
        let mme = o.mme && dense_hybrid && spec.n_linear_layers() > 0;
        let fastexp = o.fastexp;
        let s16_default = if dense_hybrid && spec.fp8_weights {
            2
        } else {
            0
        };
        // [kv8] the half-width core is a BF16-cache form
        let s16 = if fastexp && dense_hybrid && !crate::verified::kv_is_fp8(spec.kv_dtype) {
            o.s16.unwrap_or(s16_default)
        } else {
            0
        };
        Self {
            mme,
            rows: if mme { o.rows } else { 0 },
            fastexp,
            groups: if fastexp { o.groups.max(1) } else { 1 },
            s16,
        }
    }

    /// Kernel ELFs the tile recipes of this form use beyond the engine's base set.
    pub fn extra_elfs(&self, spec: &ModelSpec) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.mme {
            v.extend([
                "GAUDIUM_GATEC_ELF",
                "GAUDIUM_CHUNK_ELF",
                "GAUDIUM_SOLVE_ELF",
                "GAUDIUM_RELAYS_ELF",
                "GAUDIUM_RELAY_ELF",
            ]);
            match self.rows {
                0 => {}
                1 => v.extend(["GAUDIUM_PREPS2_ELF", "GAUDIUM_POST2_ELF"]),
                _ => v.extend([
                    "GAUDIUM_PREPA_ELF",
                    "GAUDIUM_PREPB_ELF",
                    "GAUDIUM_POSTA_ELF",
                    "GAUDIUM_POSTB_ELF",
                ]),
            }
        }
        if self.fastexp && crate::verified::kv_is_fp8(spec.kv_dtype) {
            v.extend_from_slice(super::kv8::KV8_PREFILL_ELFS); // [kv8]
        }
        if self.fastexp {
            v.push("GAUDIUM_ASOFTCX_ELF");
            v.push(if spec.head_dim == 128 {
                "GAUDIUM_A128COMBX_ELF"
            } else {
                "GAUDIUM_ACOMBX_ELF"
            });
        }
        if self.s16 > 0 {
            v.push(if self.s16 >= 2 {
                "GAUDIUM_ASOFT16P8_ELF"
            } else {
                "GAUDIUM_ASOFT16_ELF"
            });
            v.extend(["GAUDIUM_AVNAT_ELF", "GAUDIUM_AQS16_ELF"]);
        }
        v
    }

    /// Tile widths tried when the width is not fixed, widest first. The 512-row tile was
    /// measured with the tile forms only, so a model on the serial DeltaNet kernels or the
    /// plain softmax keeps the 256-row tile those recipes were validated at.
    pub fn tile_candidates(&self, spec: &ModelSpec) -> &'static [u64] {
        if crate::verified::kv_is_fp8(spec.kv_dtype) {
            // [kv8] measured on the card (SynapseAI 1.24, issue #63): the 512-row FP8 attention
            // tile gives NaN rows once it reads more than one cache chunk (any softmax build,
            // with or without chunk groups); 256-row tiles are correct at every chunk count
            return &TILE_CANDIDATES[1..];
        }
        if self.fastexp && (self.mme || spec.n_linear_layers() == 0) {
            &TILE_CANDIDATES
        } else {
            &TILE_CANDIDATES[1..]
        }
    }

    fn label(&self) -> String {
        format!(
            "DeltaNet {} | attention softmax {}, {} chunk group(s), {}",
            if self.mme {
                match self.rows {
                    0 => "chunked on the MME (per-block prep/post)",
                    1 => "chunked on the MME (row-parallel prep/post)",
                    _ => "chunked on the MME (row-parallel one-table prep/post)",
                }
            } else {
                "serial on the TPCs"
            },
            if self.fastexp { "fast-exp" } else { "plain" },
            self.groups,
            match self.s16 {
                0 => "f32 scores",
                1 => "BF16 scores",
                _ => "BF16 scores + FP8 P",
            }
        )
    }
}

// ============================================================================
// Graph builders
// ============================================================================

/// Sizes of the chunk groups of an `nch`-chunk attention core split into at most `groups`.
pub fn chunk_groups(nch: u64, groups: u64) -> Vec<u64> {
    let ng = groups.clamp(1, nch.max(1));
    let per = nch.div_ceil(ng);
    let mut out = Vec::new();
    let mut c0 = 0;
    while c0 < nch {
        let n = per.min(nch - c0);
        out.push(n);
        c0 += n;
    }
    out
}

/// Views of `src {d0, d1, NKV * nch}` per chunk group along the outermost axis (no copies).
#[allow(clippy::too_many_arguments)]
fn group_views(
    gb: &mut GraphBuilder,
    src: TensorId,
    name: &str,
    dtype: SynDataType,
    d0: u64,
    d1: u64,
    nkv: u64,
    gsz: &[u64],
) -> Vec<TensorId> {
    if gsz.len() == 1 {
        return vec![src];
    }
    let parts: Vec<TensorId> = gsz
        .iter()
        .enumerate()
        .map(|(g, &n)| gb.add_intermediate(format!("{name}_v{g}"), dtype, &[d0, d1, nkv * n]))
        .collect();
    gb.op_split(format!("{name}_split"), src, &parts, 2);
    parts
}

/// Appends one full-attention layer in its tile form (fast-`exp` softmax, chunk groups,
/// optionally the half-width core). Tensor names shared with the plain layer keep their meaning.
#[allow(clippy::too_many_arguments)]
fn append_attn_layer_tile(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    cursor: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lpfx: &str,
    t: u64,
    nch: u64,
    form: &TileForm,
) -> TensorId {
    let core = append_attn_first_tile(gb, spec, x, cursor, rope_c, rope_s, lpfx, t, nch, form);
    append_second_half(gb, spec, x, core, lpfx, t)
}

/// The second half of a layer, the same for both layer kinds: the output projection of the
/// attention / DeltaNet rows `core`, the residual add, the post norm and the MLP.
fn append_second_half(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    core: TensorId,
    lpfx: &str,
    t: u64,
) -> TensorId {
    let core_dim = spec.n_heads * spec.head_dim;
    let attn = add_linear_bf16(
        gb,
        core,
        &format!("{lpfx}W_o"),
        core_dim,
        spec.hidden,
        t,
        &format!("{lpfx}attn"),
    );
    append_post_and_mlp(gb, spec, x, attn, lpfx, t, false, None).0
}

/// The first half of a full-attention tile layer: everything up to the attention rows `core`
/// (`{heads * head_dim, t}` BF16), which it returns. Writes the tile's K / V rows.
#[allow(clippy::too_many_arguments)]
fn append_attn_first_tile(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    cursor: TensorId,
    rope_c: TensorId,
    rope_s: TensorId,
    lpfx: &str,
    t: u64,
    nch: u64,
    form: &TileForm,
) -> TensorId {
    let hid = spec.hidden;
    let nh = spec.n_heads;
    let nkv = spec.n_kv_heads;
    let dh = spec.head_dim;
    let r = (nh / nkv) * t;
    let b = nkv * nch;
    gb.set_fp8_weights(spec.fp8_weights);

    let (g_aprep, g_acomb, g_apost) = if dh == 128 {
        (
            "gaudium_attn128_prep",
            "gaudium_attn128_combine",
            "gaudium_attn128_post",
        )
    } else {
        (
            "gaudium_attn_prep",
            "gaudium_attn_combine",
            "gaudium_attn_post_f32",
        )
    };

    let h = add_rmsnorm(gb, spec, x, &format!("{lpfx}w_in"), &format!("{lpfx}h"), t);
    let qg = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_qg"),
        hid,
        spec.qg_dim(),
        t,
        &format!("{lpfx}qg"),
    );
    let k = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_k"),
        hid,
        spec.kv_dim(),
        t,
        &format!("{lpfx}k"),
    );
    let v = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_v"),
        hid,
        spec.kv_dim(),
        t,
        &format!("{lpfx}v"),
    );

    let qn_w = gb.get_or_add_persistent_tensor(format!("{lpfx}qn_w"), SynDataType::Float, &[dh]);
    let kn_w = gb.get_or_add_persistent_tensor(format!("{lpfx}kn_w"), SynDataType::Float, &[dh]);
    // [kv8] the cache dtype of this layer (q block, caches and P) and its scale table
    let kv = super::kv8::AttnKv::of(gb, spec, lpfx);
    let qb = gb.add_persistent_tensor(format!("{lpfx}qb"), kv.dt, &[dh, r, b]);
    let kcache =
        gb.add_persistent_tensor(format!("{lpfx}kcache"), kv.dt, &[dh, KV_CHUNK, b]);
    let vcache =
        gb.add_persistent_tensor(format!("{lpfx}vcache"), kv.dt, &[dh, KV_CHUNK, b]);
    gb.add_node(
        g_aprep,
        format!("{lpfx}attn_prep"),
        &kv.inputs(&[qg, k, v, qn_w, kn_w, rope_c, rope_s, cursor]),
        &[qb, kcache, vcache],
        &u32_params(&[0u32]),
    );

    let ao = gb.add_persistent_tensor(format!("{lpfx}ao"), SynDataType::Float, &[dh, r, nkv]);
    // the half-width core reads BF16 caches (`TileForm::resolve` turns it off for FP8 K/V)
    let s16 = form.s16 > 0 && kv.kv8.is_none();
    let gsz = chunk_groups(nch, form.groups);
    if gsz.len() == 1 && !s16 {
        // one group: the plain node chain with the tile builds of the softmax and the combine
        let scores = gb.add_persistent_tensor(
            format!("{lpfx}scores"),
            SynDataType::Float,
            &[KV_CHUNK, r, b],
        );
        gb.op_batch_gemm(format!("{lpfx}qk"), qb, kcache, scores, false, true);
        let pm = gb.add_persistent_tensor(format!("{lpfx}P"), kv.dt, &[KV_CHUNK, r, b]);
        let stats =
            gb.add_persistent_tensor(format!("{lpfx}stats"), SynDataType::Float, &[64, 2, r, b]);
        gb.add_node(
            "gaudium_attn_softmax_chunk",
            format!("{lpfx}softmax"),
            &kv.inputs(&[scores, cursor]),
            &[pm, stats],
            &u32_params(&[t as u32, nch as u32, 0, 0, 1]),
        );
        let ao_part =
            gb.add_persistent_tensor(format!("{lpfx}ao_part"), SynDataType::Float, &[dh, r, b]);
        gb.op_batch_gemm(format!("{lpfx}pv"), pm, vcache, ao_part, false, false);
        gb.add_node(
            g_acomb,
            format!("{lpfx}combine"),
            &[ao_part, stats],
            &[ao],
            &u32_params(&[nch as u32, 1]),
        );
    } else {
        // half-width core: queries in the score unit of the 128-lane softmax, V chunks in the
        // key order of its P. Kept persistent (scratch) so no recipe's workspace grows with the
        // context.
        let p8 = s16 && form.s16 >= 2;
        let (qsrc, vsrc, vdt) = if s16 {
            let qs =
                gb.add_persistent_tensor(format!("{lpfx}qs16"), SynDataType::Bf16, &[dh, r, b]);
            gb.add_node(
                "gaudium_attn_qs16",
                format!("{lpfx}qs16"),
                &[qb],
                &[qs],
                &[],
            );
            let vn = gb.add_persistent_tensor(
                format!("{lpfx}vnat"),
                SynDataType::Bf16,
                &[dh, KV_CHUNK, b],
            );
            gb.add_node(
                "gaudium_attn_vnat",
                format!("{lpfx}vnat"),
                &[vcache],
                &[vn],
                &u32_params(&[u32::from(p8)]),
            );
            if p8 {
                // P and the PV GEMM's V as FP8: V / 2 through the vendor cast (unit scale)
                let v8 = gb.add_persistent_tensor(
                    format!("{lpfx}vnat8"),
                    SynDataType::Fp8_143,
                    &[dh, KV_CHUNK, b],
                );
                let one = gb.get_or_add_persistent_tensor("fp8_one", SynDataType::Float, &[1]);
                let cp = ConvertToFp8Params {
                    round_mode: CAST_ROUND_HALF_NE,
                };
                gb.add_node(
                    "convert_to_fp8_bf16",
                    format!("{lpfx}vnat_cast"),
                    &[vn, one],
                    &[v8],
                    as_param_bytes(&cp),
                );
                (qs, v8, SynDataType::Fp8_143)
            } else {
                (qs, vn, SynDataType::Bf16)
            }
        } else {
            (qb, vcache, kv.dt)
        };
        let sdt = if s16 {
            SynDataType::Bf16
        } else {
            SynDataType::Float
        };
        let pdt = if p8 {
            SynDataType::Fp8_143
        } else {
            kv.dt
        };
        let qv = group_views(
            gb,
            qsrc,
            &format!("{lpfx}qb"),
            if s16 { SynDataType::Bf16 } else { kv.dt },
            dh,
            r,
            nkv,
            &gsz,
        );
        let kvw = group_views(
            gb,
            kcache,
            &format!("{lpfx}kc"),
            kv.dt,
            dh,
            KV_CHUNK,
            nkv,
            &gsz,
        );
        let vv = group_views(gb, vsrc, &format!("{lpfx}vc"), vdt, dh, KV_CHUNK, nkv, &gsz);
        let mut aops = Vec::with_capacity(gsz.len());
        let mut sts = Vec::with_capacity(gsz.len());
        let mut c0 = 0u64;
        for (g, &n) in gsz.iter().enumerate() {
            let gp = format!("{lpfx}g{g}_");
            let bg = nkv * n;
            let sc = gb.add_persistent_tensor(format!("{gp}scores"), sdt, &[KV_CHUNK, r, bg]);
            gb.op_batch_gemm(format!("{gp}qk"), qv[g], kvw[g], sc, false, true);
            let pg = gb.add_persistent_tensor(format!("{gp}P"), pdt, &[KV_CHUNK, r, bg]);
            let stg =
                gb.add_persistent_tensor(format!("{gp}stats"), SynDataType::Float, &[64, 2, r, bg]);
            let base = (c0 * KV_CHUNK) as u32;
            if s16 {
                gb.add_node(
                    "gaudium_attn_softmax16",
                    format!("{gp}softmax"),
                    &[sc, cursor],
                    &[pg, stg],
                    &u32_params(&[t as u32, n as u32, 0, base]),
                );
            } else {
                gb.add_node(
                    "gaudium_attn_softmax_chunk",
                    format!("{gp}softmax"),
                    &kv.inputs(&[sc, cursor]),
                    &[pg, stg],
                    &u32_params(&[t as u32, n as u32, 0, 0, 1, base]),
                );
            }
            let ag =
                gb.add_persistent_tensor(format!("{gp}ao_part"), SynDataType::Float, &[dh, r, bg]);
            gb.op_batch_gemm(format!("{gp}pv"), pg, vv[g], ag, false, false);
            aops.push(ag);
            sts.push(stg);
            c0 += n;
        }
        let comb = u32_params(&[nch as u32, 1]);
        if gsz.len() > 1 {
            let ao_part =
                gb.add_persistent_tensor(format!("{lpfx}ao_part"), SynDataType::Float, &[dh, r, b]);
            let stats = gb.add_persistent_tensor(
                format!("{lpfx}stats"),
                SynDataType::Float,
                &[64, 2, r, b],
            );
            gb.op_concat(format!("{lpfx}ao_cat"), &aops, ao_part, 2);
            gb.op_concat(format!("{lpfx}st_cat"), &sts, stats, 3);
            gb.add_node(
                g_acomb,
                format!("{lpfx}combine"),
                &[ao_part, stats],
                &[ao],
                &comb,
            );
        } else {
            gb.add_node(
                g_acomb,
                format!("{lpfx}combine"),
                &[aops[0], sts[0]],
                &[ao],
                &comb,
            );
        }
    }

    let core = gb.add_persistent_tensor(format!("{lpfx}core"), SynDataType::Bf16, &[nh * dh, t]);
    gb.add_node(g_apost, format!("{lpfx}attn_post"), &[ao, qg], &[core], &[]);
    core
}

/// The chunked gated-delta recurrence of one layer from the matrix state `s0` over `nchk` chunks
/// of `c` rows: coefficients on the TPCs, inverse and products on the MME, then the serial state
/// chain. Returns `(X {128, 2c, hv, nchk}, state after the last chunk)`; rows `< c` of chunk `j`
/// of `X` are the outputs of rows `j*c ..`.
#[allow(clippy::too_many_arguments)]
fn append_chunk_core(
    gb: &mut GraphBuilder,
    lpfx: &str,
    c: u64,
    nchk: u64,
    hv: u64,
    s0: TensorId,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    gate: TensorId,
) -> (TensorId, TensorId) {
    const D: u64 = 128;
    let f32t = SynDataType::Float;
    let tp = c * nchk;
    let a = |gb: &mut GraphBuilder, n: &str, dims: &[u64]| {
        gb.add_persistent_tensor(format!("{lpfx}{n}"), f32t, dims)
    };
    let cc = [c, c, hv, nchk];

    let gbt = a(gb, "gbT", &[tp, 128]);
    let tpar = TransposeParams {
        permutation: [1, 0, 0, 0, 0],
        tensor_dim: 2,
    };
    gb.add_node(
        "transpose",
        format!("{lpfx}gbT"),
        &[gate],
        &[gbt],
        as_param_bytes(&tpar),
    );
    let kk = a(gb, "KK", &cc);
    let qk = a(gb, "QK", &cc);
    gb.op_batch_gemm(format!("{lpfx}KK"), k, k, kk, false, true);
    gb.op_batch_gemm(format!("{lpfx}QK"), q, k, qk, false, true);

    let m3 = a(gb, "M3", &[c, 2 * c, hv, nchk]);
    let an = a(gb, "An", &cc);
    let gb2 = a(gb, "gb2", &[c, 2, hv, nchk]);
    let gq0 = a(gb, "GQ0", &[D, 2 * c, hv, nchk]);
    let ak = a(gb, "AK", &[D, 2 * c, hv, nchk]);
    let gl = a(gb, "gL", &[1, 1, hv, nchk]);
    gb.add_node(
        "gaudium_gdn_chunk",
        format!("{lpfx}chunk"),
        &[gbt, kk, qk, q, k],
        &[m3, an, gb2, gq0, ak, gl],
        &[],
    );

    // (I + A)^-1 diag(beta): 16x16 diagonal blocks on the TPC, the rest by the exact Neumann
    // series of the block-strictly-lower remainder on the MME
    let d0n = a(gb, "D0n", &cc);
    let dn = a(gb, "Dn", &cc);
    let dg = a(gb, "Dg", &cc);
    let lp = a(gb, "Lp", &cc);
    let id = a(gb, "Id", &cc);
    gb.add_node(
        "gaudium_gdn_solve",
        format!("{lpfx}solve"),
        &[an, gb2],
        &[d0n, dn, dg, lp, id],
        &[],
    );
    let gemm3 = |gb: &mut GraphBuilder,
                 n: String,
                 x: TensorId,
                 y: TensorId,
                 z: TensorId,
                 out: TensorId,
                 ta: bool| {
        let p = crate::synapse::ops::GemmParams::new(ta, false);
        gb.add_node("batch_gemm", n, &[x, y, z], &[out], as_param_bytes(&p));
    };
    let y = a(gb, "Y", &cc);
    gb.op_batch_gemm(format!("{lpfx}Y"), d0n, lp, y, false, false);
    let mut n = a(gb, "N1", &cc);
    gemm3(gb, format!("{lpfx}N1"), d0n, lp, id, n, false);
    for it in 2..(c / 16) {
        let n2 = a(gb, &format!("N{it}"), &cc);
        gemm3(gb, format!("{lpfx}N{it}"), y, n, id, n2, false);
        n = n2;
    }
    let mw = a(gb, "Mw", &cc);
    let mun = a(gb, "Mun", &cc);
    gb.op_batch_gemm(format!("{lpfx}Mun"), n, dn, mun, false, false);
    gb.op_batch_gemm(format!("{lpfx}Mw"), n, dg, mw, false, false);

    // chunk-batched products
    let m2 = a(gb, "M2", &[c, 2 * c, hv, nchk]);
    let un = a(gb, "Un", &[D, c, hv, nchk]);
    let cx = a(gb, "CX", &[D, 2 * c, hv, nchk]);
    let ax = a(gb, "AX", &[D, 2 * c, hv, nchk]);
    gb.op_batch_gemm(format!("{lpfx}M2"), m3, mw, m2, false, false);
    gb.op_batch_gemm(format!("{lpfx}Un"), mun, v, un, false, false);
    gb.op_batch_gemm(format!("{lpfx}CX"), m3, un, cx, false, false);
    gemm3(gb, format!("{lpfx}AX"), m2, k, gq0, ax, false);

    // per-chunk operands: views along the outermost axis
    let split = |gb: &mut GraphBuilder, src: TensorId, n: &str, dims: &[u64]| -> Vec<TensorId> {
        let parts: Vec<TensorId> = (0..nchk)
            .map(|j| gb.add_intermediate(format!("{lpfx}{n}_{j}"), f32t, dims))
            .collect();
        gb.op_split(format!("{lpfx}{n}_split"), src, &parts, 3);
        parts
    };
    let axj = split(gb, ax, "AXj", &[D, 2 * c, hv, 1]);
    let cxj = split(gb, cx, "CXj", &[D, 2 * c, hv, 1]);
    let akj = split(gb, ak, "AKj", &[D, 2 * c, hv, 1]);
    let glj = split(gb, gl, "gLj", &[1, 1, hv, 1]);

    // the serial chain over chunks: three nodes per chunk
    let mut xs = Vec::with_capacity(nchk as usize);
    let mut s = s0;
    for j in 0..nchk as usize {
        let xj = gb.add_intermediate(format!("{lpfx}X_{j}"), f32t, &[D, 2 * c, hv, 1]);
        gemm3(gb, format!("{lpfx}X_{j}"), axj[j], s, cxj[j], xj, false);
        let sd = a(gb, &format!("mSd_{j}"), &[D, D, hv, 1]);
        gb.add_node(
            "mult_fwd_f32",
            format!("{lpfx}Sd_{j}"),
            &[s, glj[j]],
            &[sd],
            &[],
        );
        let sn = a(gb, &format!("mS{}", j + 1), &[D, D, hv, 1]);
        gemm3(gb, format!("{lpfx}Supd_{j}"), akj[j], xj, sd, sn, true);
        xs.push(xj);
        s = sn;
    }
    let x_all = a(gb, "X", &[D, 2 * c, hv, nchk]);
    gb.op_concat(format!("{lpfx}Xcat"), &xs, x_all, 3);
    (x_all, s)
}

/// Measurement only: the MME side of the chunk recurrence (Gram products, Neumann series,
/// chunk-batched products, serial state chain) on its own, every operand a bound tensor of data
/// type `dt`, so the card can time the same GEMM shapes in f32 and in BF16. The TPC kernels that
/// build the coefficients are left out; the values do not matter for the timing.
pub fn build_chunk_gemm_probe(spec: &ModelSpec, t: u64, dt: SynDataType, chain_only: bool) -> GraphBuilder {
    const D: u64 = 128;
    let c = GDN_CHUNK;
    let nchk = t.div_ceil(c);
    let hv = spec.gdn_v_heads;
    let tag = if dt == SynDataType::Float { "f32" } else { "bf16" };
    let mut gb = GraphBuilder::new(format!(
        "g_qwen_probe_chunk{}_{tag}_T{t}",
        if chain_only { "chain" } else { "gemm" }
    ));
    let a = |gb: &mut GraphBuilder, n: &str, dims: &[u64]| {
        gb.add_persistent_tensor(format!("P_{n}"), dt, dims)
    };
    let gemm3 = |gb: &mut GraphBuilder, n: &str, x: TensorId, y: TensorId, z: TensorId, out: TensorId, ta: bool| {
        let p = crate::synapse::ops::GemmParams::new(ta, false);
        gb.add_node("batch_gemm", format!("P_{n}"), &[x, y, z], &[out], as_param_bytes(&p));
    };
    let cc = [c, c, hv, nchk];
    let wide = [D, 2 * c, hv, nchk];
    let (ax, cx, ak, gl);
    if chain_only {
        ax = a(&mut gb, "AX", &wide);
        cx = a(&mut gb, "CX", &wide);
    } else {
        let q = a(&mut gb, "q", &[D, c, hv, nchk]);
        let k = a(&mut gb, "k", &[D, c, hv, nchk]);
        let v = a(&mut gb, "v", &[D, c, hv, nchk]);
        let kk = a(&mut gb, "KK", &cc);
        let qk = a(&mut gb, "QK", &cc);
        gb.op_batch_gemm("P_KK", k, k, kk, false, true);
        gb.op_batch_gemm("P_QK", q, k, qk, false, true);
        let (d0n, dn, dg, lp, id) = (
            a(&mut gb, "D0n", &cc),
            a(&mut gb, "Dn", &cc),
            a(&mut gb, "Dg", &cc),
            a(&mut gb, "Lp", &cc),
            a(&mut gb, "Id", &cc),
        );
        let y = a(&mut gb, "Y", &cc);
        gb.op_batch_gemm("P_Y", d0n, lp, y, false, false);
        let mut n = a(&mut gb, "N1", &cc);
        gemm3(&mut gb, "N1", d0n, lp, id, n, false);
        for it in 2..(c / 16) {
            let n2 = a(&mut gb, &format!("N{it}"), &cc);
            gemm3(&mut gb, &format!("N{it}"), y, n, id, n2, false);
            n = n2;
        }
        let mw = a(&mut gb, "Mw", &cc);
        let mun = a(&mut gb, "Mun", &cc);
        gb.op_batch_gemm("P_Mun", n, dn, mun, false, false);
        gb.op_batch_gemm("P_Mw", n, dg, mw, false, false);
        let m3 = a(&mut gb, "M3", &[c, 2 * c, hv, nchk]);
        let gq0 = a(&mut gb, "GQ0", &wide);
        let m2 = a(&mut gb, "M2", &[c, 2 * c, hv, nchk]);
        let un = a(&mut gb, "Un", &[D, c, hv, nchk]);
        cx = a(&mut gb, "CX", &wide);
        ax = a(&mut gb, "AX", &wide);
        gb.op_batch_gemm("P_M2", m3, mw, m2, false, false);
        gb.op_batch_gemm("P_Un", mun, v, un, false, false);
        gb.op_batch_gemm("P_CX", m3, un, cx, false, false);
        gemm3(&mut gb, "AX", m2, k, gq0, ax, false);
    }
    ak = a(&mut gb, "AK", &wide);
    gl = a(&mut gb, "gL", &[1, 1, hv, nchk]);
    let split = |gb: &mut GraphBuilder, src: TensorId, n: &str, dims: &[u64]| -> Vec<TensorId> {
        let parts: Vec<TensorId> = (0..nchk)
            .map(|j| gb.add_intermediate(format!("P_{n}_{j}"), dt, dims))
            .collect();
        gb.op_split(format!("P_{n}_split"), src, &parts, 3);
        parts
    };
    let axj = split(&mut gb, ax, "AXj", &[D, 2 * c, hv, 1]);
    let cxj = split(&mut gb, cx, "CXj", &[D, 2 * c, hv, 1]);
    let akj = split(&mut gb, ak, "AKj", &[D, 2 * c, hv, 1]);
    let glj = split(&mut gb, gl, "gLj", &[1, 1, hv, 1]);
    let mut s = a(&mut gb, "S0", &[D, D, hv, 1]);
    let mul = if dt == SynDataType::Float { "mult_fwd_f32" } else { "mult_fwd_bf16" };
    for j in 0..nchk as usize {
        let xj = a(&mut gb, &format!("X_{j}"), &[D, 2 * c, hv, 1]);
        gemm3(&mut gb, &format!("X_{j}"), axj[j], s, cxj[j], xj, false);
        let sd = a(&mut gb, &format!("Sd_{j}"), &[D, D, hv, 1]);
        gb.add_node(mul, format!("P_Sd_{j}"), &[s, glj[j]], &[sd], &[]);
        let sn = a(&mut gb, &format!("S{}", j + 1), &[D, D, hv, 1]);
        gemm3(&mut gb, &format!("Supd_{j}"), akj[j], xj, sd, sn, true);
        s = sn;
    }
    gb
}

/// Appends one Gated DeltaNet layer with the chunked MME recurrence. The interface to the engine
/// is the plain layer's: `(x_in, S_in, cs_in) -> (out, S_out, cs_out)`, states in the decode
/// kernels' blob layout.
fn append_linear_layer_mme(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    slot: TensorId,
    lpfx: &str,
    t: u64,
    form: &TileForm,
) -> TensorId {
    let core = append_linear_first_mme(gb, spec, x, slot, lpfx, t, form);
    let attn = add_linear_bf16(
        gb,
        core,
        &format!("{lpfx}W_o"),
        spec.gdn_v_heads * spec.gdn_v_dim,
        spec.hidden,
        t,
        &format!("{lpfx}attn"),
    );
    append_post_and_mlp(gb, spec, x, attn, lpfx, t, false, None).0
}

/// The first half of a Gated DeltaNet tile layer with the chunked MME recurrence: everything up
/// to the gated, normed recurrence rows `core` (`{value heads * value dim, t}` BF16), which it
/// returns. Reads and writes the layer's recurrent and conv state.
fn append_linear_first_mme(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    slot: TensorId,
    lpfx: &str,
    t: u64,
    form: &TileForm,
) -> TensorId {
    let hid = spec.hidden;
    let hv = spec.gdn_v_heads;
    let hk = spec.gdn_k_heads;
    let dl = spec.gdn_v_dim;
    let qkv = spec.gdn_qkv_dim();
    let hvp = spec.gdn_hvp();
    let f32t = SynDataType::Float;
    let c = GDN_CHUNK;
    let nchk = t.div_ceil(c);
    let tp = c * nchk;
    gb.set_fp8_weights(spec.fp8_weights);

    let h = add_rmsnorm(gb, spec, x, &format!("{lpfx}w_in"), &format!("{lpfx}h"), t);
    let qkvz = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_qkvz"),
        hid,
        spec.gdn_qkvz_dim(),
        t,
        &format!("{lpfx}qkvz"),
    );
    let ba = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_ba"),
        hid,
        spec.gdn_ba_dim(),
        t,
        &format!("{lpfx}ba"),
    );

    let conv_w = gb.get_or_add_persistent_tensor(format!("{lpfx}conv_w"), f32t, &[qkv, 4]);
    let conv_b = gb.get_or_add_persistent_tensor(format!("{lpfx}conv_b"), f32t, &[qkv]);
    let cs_in = gb.add_persistent_tensor(format!("{lpfx}cs_in"), f32t, &[qkv, 3, 1]);
    let cs_out = gb.add_persistent_tensor(format!("{lpfx}cs_out"), f32t, &[qkv, 3]);
    let cs_ckpt = gb.add_persistent_tensor(format!("{lpfx}cs_ckpt"), f32t, &[qkv, 3, 1]);
    let a_log = gb.get_or_add_persistent_tensor(format!("{lpfx}A_log"), f32t, &[hvp]);
    let dt_bias = gb.get_or_add_persistent_tensor(format!("{lpfx}dt_bias"), f32t, &[hvp]);
    let norm_w = gb.get_or_add_persistent_tensor(format!("{lpfx}norm_w"), f32t, &[dl]);
    let s_in = gb.add_persistent_tensor(format!("{lpfx}S_in"), f32t, &[64, 256, hv, 1]);
    let s_out = gb.add_persistent_tensor(format!("{lpfx}S_out"), f32t, &[64, 256, hv]);

    // q / k / v in the chunked layout: row t at {., t % c, head, t / c}; pad rows are zero
    let q = gb.add_persistent_tensor(format!("{lpfx}q"), f32t, &[dl, c, hv, nchk]);
    let k = gb.add_persistent_tensor(format!("{lpfx}k"), f32t, &[dl, c, hv, nchk]);
    let v = gb.add_persistent_tensor(format!("{lpfx}v"), f32t, &[dl, c, hv, nchk]);
    // flags: bit5 chunked layout, bit6 row-parallel build, chunk rows in the high half
    let pf = 32u32 | ((c as u32) << 16) | if form.rows >= 1 { 64 } else { 0 };
    if form.rows >= 2 {
        let y = gb.add_intermediate(format!("{lpfx}y"), f32t, &[128, t, 2 * hk]);
        gb.add_node(
            "gaudium_gdn_prepa",
            format!("{lpfx}gdn_prepa"),
            &[qkvz, conv_w, cs_in, conv_b, slot],
            &[y, v, cs_out, cs_ckpt],
            &u32_params(&[pf]),
        );
        gb.add_node(
            "gaudium_gdn_prepb",
            format!("{lpfx}gdn_prepb"),
            &[y],
            &[q, k],
            &u32_params(&[c as u32]),
        );
    } else {
        gb.add_node(
            "gaudium_gdn_prep",
            format!("{lpfx}gdn_prep"),
            &[qkvz, conv_w, cs_in, conv_b, slot],
            &[q, k, v, cs_out, cs_ckpt],
            &u32_params(&[pf]),
        );
    }

    // gate: lanes 0..63 the chunk-local cumulative log decay, 64..127 beta
    let gate = gb.add_persistent_tensor(format!("{lpfx}gb"), f32t, &[128, tp]);
    gb.add_node(
        "gaudium_gdn_gate",
        format!("{lpfx}gdn_gate"),
        &[ba, a_log, dt_bias],
        &[gate],
        &u32_params(&[c as u32]),
    );

    // state blob -> matrix, the chunk recurrence, matrix -> blob
    let s0 = gb.add_persistent_tensor(format!("{lpfx}mS0"), f32t, &[128, 128, hv, 1]);
    gb.add_node(
        "gaudium_gdn_relayout",
        format!("{lpfx}relay_in"),
        &[s_in, slot],
        &[s0],
        &u32_params(&[0]),
    );
    let (x_all, s_last) = append_chunk_core(gb, lpfx, c, nchk, hv, s0, q, k, v, gate);
    gb.add_node(
        "gaudium_gdn_relayout",
        format!("{lpfx}relay_out"),
        &[s_last],
        &[s_out],
        &u32_params(&[1]),
    );

    let core = gb.add_persistent_tensor(format!("{lpfx}core"), SynDataType::Bf16, &[hv * dl, t]);
    if form.rows >= 2 {
        let rn = gb.add_intermediate(format!("{lpfx}rn"), f32t, &[64, t, hv]);
        gb.add_node(
            "gaudium_gdn_posta",
            format!("{lpfx}gdn_posta"),
            &[x_all],
            &[rn],
            &u32_params(&[c as u32]),
        );
        gb.add_node(
            "gaudium_gdn_postb",
            format!("{lpfx}gdn_postb"),
            &[x_all, qkvz, norm_w, rn],
            &[core],
            &u32_params(&[c as u32]),
        );
    } else {
        gb.add_node(
            "gaudium_gdn_post",
            format!("{lpfx}gdn_post"),
            &[x_all, qkvz, norm_w],
            &[core],
            &u32_params(&[pf]),
        );
    }
    core
}

/// The single-layer full-attention tile graph of `form` (`t` rows over `nch` cache chunks).
/// `elf_base` are the engine's base kernel hashes (the plain form's fingerprint), `elf_tile` the
/// hashes of the tile forms' kernels.
pub fn build_attn_tile_graph(
    spec: &ModelSpec,
    t: u64,
    lmax: u64,
    nch: u64,
    form: &TileForm,
    elf_base: &[(String, String)],
    elf_tile: &[(String, String)],
) -> GraphBuilder {
    if !form.fastexp {
        return build_prefill_attn_layer_graph(spec, t, lmax, nch, elf_base);
    }
    let tag = match form.s16 {
        0 => "x",
        1 => "h",
        _ => "h8",
    };
    let mut gb = GraphBuilder::new(format!(
        "g_qwen_attn{tag}_T{t}_C{nch}_G{}",
        chunk_groups(nch, form.groups).len()
    ));
    for (k, h) in elf_base.iter().chain(elf_tile) {
        gb.record_kernel_elf_hash(k, h);
    }
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[spec.hidden, t]);
    let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
    let rope_c = gb.add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    let _ = append_attn_layer_tile(
        &mut gb, spec, x_in, cursor, rope_c, rope_s, "L_", t, nch, form,
    );
    gb
}

/// The single-layer Gated DeltaNet tile graph of `form` (`t` rows).
pub fn build_lin_tile_graph(
    spec: &ModelSpec,
    t: u64,
    form: &TileForm,
    elf_base: &[(String, String)],
    elf_tile: &[(String, String)],
) -> GraphBuilder {
    if !form.mme {
        return build_prefill_linear_layer_graph(spec, t, elf_base);
    }
    let mut gb = GraphBuilder::new(format!("g_qwen_linm{}_T{t}", form.rows));
    for (k, h) in elf_base.iter().chain(elf_tile) {
        gb.record_kernel_elf_hash(k, h);
    }
    let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[spec.hidden, t]);
    let slot = gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]);
    let _ = append_linear_layer_mme(&mut gb, spec, x_in, slot, "L_", t, form);
    gb
}

// ============================================================================
// Half-layer graphs: two tiles in flight
// ============================================================================
//
// A tile layer is a serial chain that alternates between the engine classes: its first half
// (the projections, then the attention core or the DeltaNet recurrence) keeps the TPCs busy
// most of the time, its second half (output projection and MLP) the MME. Inside one launch the
// card runs nodes of different engine classes at once when no tensor connects them, so a recipe
// that holds the SECOND half of one tile's layer and the FIRST half of the next tile's layer
// lets the big weight GEMMs of the one run under the kernels of the other. The two bodies are
// built exactly like the halves of the full-layer recipes (same nodes, same shapes: the same
// bits), then their nodes are put in the creation order a cost model of the two engine queues
// picks ([`weave_order`]).

/// The first half a half-layer graph carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FirstHalf {
    /// Gated DeltaNet layer (chunked MME recurrence).
    Lin,
    /// Full-attention layer over this many KV chunks.
    Attn(u64),
}

/// Layer prefix of the second-half body of a half-layer graph (the first half keeps `L_`).
pub const SECOND_PFX: &str = "G_";

/// Engine class a node keeps busy, for the weave's cost model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Eng {
    Mme,
    Tpc,
}

/// Estimated device time of a node in microseconds and its engine class. The figures are node
/// spans of a 512-row FP8 tile on the card, scaled by the rows; a node the table does not know
/// counts as a short TPC node. Only the ORDER of the two bodies' nodes depends on these numbers.
fn node_cost(guid: &str, name: &str, t: u64) -> (Eng, f64) {
    let scale = t as f64 / 512.0;
    let mme = matches!(guid, "gemm" | "batch_gemm" | "fp8_gemm_bf16");
    // strip the layer prefix (`L_` / `G_`) and a chunk-group prefix (`g3_`)
    let base = name.split_once('_').map_or(name, |(_, r)| r);
    let base = match base.split_once('_') {
        Some((g, r)) if g.len() > 1 && g.starts_with('g') && g[1..].bytes().all(|b| b.is_ascii_digit()) => r,
        _ => base,
    };
    let us = if mme {
        match base {
            "qkvz_gemm" => 104.0,
            "gate_gemm" => 111.0,
            "up_gemm" => 156.0,
            "mo_gemm" => 116.0,
            "attn_gemm" => 45.0,
            "qg_gemm" => 83.0,
            "k_gemm" => 42.0,
            "v_gemm" => 19.0,
            "qk" => 66.0,
            "pv" => 35.0,
            "CX" | "M2" => 31.0,
            "QK" | "AX" => 25.0,
            b if b.starts_with("X_") => 13.0,
            b if b.starts_with("Supd_") => 6.0,
            _ => 16.0,
        }
    } else {
        match base {
            "attn_prep" => 149.0,
            "softmax" => 141.0,
            "solve" => 72.0,
            "combine" => 62.0,
            "gdn_prepa" => 60.0,
            "chunk" => 50.0,
            "silu" | "mul" => 32.0,
            "gdn_posta" | "gdn_postb" => 34.0,
            "gdn_prepb" | "attn_post" => 26.0,
            "relay_in" | "relay_out" | "gdn_gate" | "vnat" | "qs16" => 18.0,
            _ => 6.0,
        }
    };
    (if mme { Eng::Mme } else { Eng::Tpc }, us * scale)
}

/// Creation order for the nodes of two independent bodies `a` and `b` (each a list of node
/// indices in its own creation order): the order a dispatcher would start them in if every
/// engine class ran one node at a time and each body ran its nodes one after another. The card
/// executes each engine's queue in order, so this order lets a body's MME node start while the
/// other body is on the TPCs instead of waiting behind its next MME node.
fn weave_order(cost: &dyn Fn(usize) -> (Eng, f64), a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut order = Vec::with_capacity(a.len() + b.len());
    let (mut ia, mut ib) = (0usize, 0usize);
    let (mut ready_a, mut ready_b) = (0.0f64, 0.0f64);
    let mut free = [0.0f64; 2];
    let slot = |e: Eng| usize::from(e == Eng::Tpc);
    while ia < a.len() || ib < b.len() {
        let start = |i: usize, list: &[usize], ready: f64| -> Option<(f64, Eng, f64)> {
            list.get(i).map(|&n| {
                let (e, us) = cost(n);
                (ready.max(free[slot(e)]), e, us)
            })
        };
        let (sa, sb) = (start(ia, a, ready_a), start(ib, b, ready_b));
        let take_a = match (sa, sb) {
            (Some(x), Some(y)) => x.0 <= y.0,
            (Some(_), None) => true,
            _ => false,
        };
        if take_a {
            let (s, e, us) = sa.expect("a has a node");
            ready_a = s + us;
            free[slot(e)] = s + us;
            order.push(a[ia]);
            ia += 1;
        } else {
            let (s, e, us) = sb.expect("b has a node");
            ready_b = s + us;
            free[slot(e)] = s + us;
            order.push(b[ib]);
            ib += 1;
        }
    }
    order
}

/// Puts the nodes of a two-body half-layer graph in the order `seq` describes and hands that
/// order to the compiler as the user execution order (one group: with the compiler's soft
/// control edges this fixes each engine's queue order without cross-engine waits).
///
/// `seq` is a `,`-separated list of runs. `B:name` = "the nodes of body `B` (`G` = second-half
/// body, `F` = first-half body) not placed yet, in creation order, up to and including node
/// `name`" (`name` without the layer prefix; `*` = the rest of the body); `B=name` = that node
/// alone (to keep the nodes of one compiler bundle together when the builder created another
/// node between them). Nodes no run names follow in creation order. A measurement / tuning entry point: the production orders are
/// fixed sequences chosen from card timings.
pub fn apply_seq_order(gb: &mut GraphBuilder, seq: &str) -> Result<(), String> {
    // `~` first: a partial order. Only the nodes the runs name get a position; the compiler
    // places the others itself (it keeps its own interleave of, say, the chunk groups of an
    // attention core, which a full order would serialise).
    let (partial, seq) = match seq.strip_prefix('~') {
        Some(rest) => (true, rest),
        None => (false, seq),
    };
    let names: Vec<String> = gb.node_io_views().iter().map(|n| n.name.to_string()).collect();
    let body_of = |i: usize| if names[i].starts_with(SECOND_PFX) { 'G' } else { 'F' };
    let mut emitted = vec![false; names.len()];
    let mut order: Vec<usize> = Vec::with_capacity(names.len());
    for run in seq.split(',').map(str::trim).filter(|r| !r.is_empty()) {
        // `B:name` = body B's remaining nodes up to and including `name`; `B=name` = that node alone
        let (body, name, single) = match (run.split_once(':'), run.split_once('=')) {
            (Some((b, n)), _) => (b, n, false),
            (None, Some((b, n))) => (b, n, true),
            _ => return Err(format!("weave sequence run '{run}': expected B:name or B=name")),
        };
        let (bc, pfx) = match body {
            "G" => ('G', SECOND_PFX),
            "F" => ('F', "L_"),
            _ => return Err(format!("weave sequence run '{run}': body must be F or G")),
        };
        let want = format!("{pfx}{name}");
        let target = if name == "*" {
            None
        } else {
            Some(
                (0..names.len())
                    .find(|&i| !emitted[i] && body_of(i) == bc && names[i] == want)
                    .ok_or_else(|| {
                        format!("weave sequence run '{run}': no node {want} left in body {body}")
                    })?,
            )
        };
        for (i, done) in emitted.iter_mut().enumerate() {
            if *done || body_of(i) != bc {
                continue;
            }
            let take = match (single, target) {
                (true, Some(t)) => i == t,
                (_, Some(t)) => i <= t,
                (_, None) => true,
            };
            if take {
                *done = true;
                order.push(i);
            }
        }
    }
    let placed = order.len();
    order.extend((0..names.len()).filter(|&i| !emitted[i]));
    if !gb.reorder_nodes(&order) {
        return Err("weave sequence: not a permutation".into());
    }
    if partial {
        for i in 0..placed {
            gb.set_node_exec_order(crate::synapse::graph::NodeId(i), 0, i as u32);
        }
    } else {
        gb.set_user_exec_order(Some(0));
    }
    Ok(())
}

/// How the nodes of a two-body half-layer graph are ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weave {
    /// The order [`weave_order`] picks.
    Cost,
    /// The second-half body's nodes, then the first-half body's.
    SecondFirst,
    /// The first-half body's nodes, then the second-half body's.
    FirstFirst,
}

/// A half-layer tile graph of `t` rows: the first half `first` of a layer (prefix `L_`, the
/// names of the full-layer graph; its result `L_core` is a tensor the tile loop binds), the
/// second half of a layer (prefix [`SECOND_PFX`]: `G_x_in`, `G_core` -> `G_out`), or both as two
/// independent bodies.
#[allow(clippy::too_many_arguments)]
pub fn build_half_graph(
    spec: &ModelSpec,
    t: u64,
    lmax: u64,
    first: Option<FirstHalf>,
    second: bool,
    weave: Weave,
    form: &TileForm,
    elf_base: &[(String, String)],
    elf_tile: &[(String, String)],
) -> GraphBuilder {
    let ftag = match first {
        None => String::new(),
        Some(FirstHalf::Lin) => format!("_Fl{}", form.rows),
        Some(FirstHalf::Attn(nch)) => format!(
            "_Fa{}_C{nch}_G{}",
            form.s16,
            chunk_groups(nch, form.groups).len()
        ),
    };
    let wtag = match (first.is_some() && second, weave) {
        (false, _) | (true, Weave::Cost) => "",
        (true, Weave::SecondFirst) => "_wG",
        (true, Weave::FirstFirst) => "_wF",
    };
    let mut gb = GraphBuilder::new(format!(
        "g_qwen_half{}{ftag}_T{t}{wtag}",
        if second { "_G" } else { "" }
    ));
    for (k, h) in elf_base.iter().chain(elf_tile) {
        gb.record_kernel_elf_hash(k, h);
    }
    let n0 = gb.node_count();
    if second {
        let core_dim = spec.n_heads * spec.head_dim;
        gb.set_fp8_weights(spec.fp8_weights);
        let x = gb.add_persistent_tensor(
            format!("{SECOND_PFX}x_in"),
            SynDataType::Bf16,
            &[spec.hidden, t],
        );
        let core = gb.add_persistent_tensor(
            format!("{SECOND_PFX}core"),
            SynDataType::Bf16,
            &[core_dim, t],
        );
        let _ = append_second_half(&mut gb, spec, x, core, SECOND_PFX, t);
    }
    let n1 = gb.node_count();
    match first {
        None => {}
        Some(FirstHalf::Lin) => {
            let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[spec.hidden, t]);
            let slot = gb.get_or_add_persistent_tensor("zero_slot", SynDataType::Int32, &[1]);
            let _ = append_linear_first_mme(&mut gb, spec, x_in, slot, "L_", t, form);
        }
        Some(FirstHalf::Attn(nch)) => {
            let x_in = gb.add_persistent_tensor("x_in", SynDataType::Bf16, &[spec.hidden, t]);
            let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
            let rope_c = gb.add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
            let rope_s = gb.add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
            let _ = append_attn_first_tile(
                &mut gb, spec, x_in, cursor, rope_c, rope_s, "L_", t, nch, form,
            );
        }
    }
    let n2 = gb.node_count();
    if n1 > n0 && n2 > n1 {
        let (g, f): (Vec<usize>, Vec<usize>) = ((n0..n1).collect(), (n1..n2).collect());
        let order: Vec<usize> = match weave {
            Weave::SecondFirst => (0..n2).collect(),
            Weave::FirstFirst => (0..n0).chain(n1..n2).chain(n0..n1).collect(),
            Weave::Cost => {
                let costs: Vec<(Eng, f64)> = gb
                    .node_io_views()
                    .iter()
                    .map(|n| node_cost(n.guid, n.name, t))
                    .collect();
                let woven = weave_order(&|i| costs[i], &f, &g);
                (0..n0).chain(woven).collect()
            }
        };
        let ok = gb.reorder_nodes(&order);
        debug_assert!(ok, "weave is a permutation");
    }
    gb
}

/// Is `name` a tensor of a half-layer graph that the tile loop binds itself, beyond those of a
/// full-layer graph: the hand-over rows between the halves and the second body's rows and
/// weights.
fn is_half_bound(name: &str) -> bool {
    name == "L_core"
        || name.strip_prefix(SECOND_PFX).is_some_and(|s| {
            matches!(s, "x_in" | "core" | "out") || is_generic_layer_weight(&format!("L_{s}"))
        })
}

/// Offsets of a half-layer graph's scratch tensors in the shared region, and the bytes they span.
pub(super) fn half_scratch_layout(gb: &GraphBuilder) -> (BTreeMap<String, u64>, u64) {
    let mut map = BTreeMap::new();
    let mut off = 0u64;
    for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
        if is_loop_bound(&name) || is_half_bound(&name) {
            continue;
        }
        map.insert(name, off);
        off += align_up(bytes.max(256), 256);
    }
    (map, off)
}

// ============================================================================
// Two tiles in flight: the wavefront recipes
// ============================================================================
//
// A full-layer recipe is a serial chain: on a 512-row FP8 tile the MME is busy 62 % of a DeltaNet
// layer and the TPCs 42 %, one after the other. The card runs each engine's queue in the order
// the compiler gives it and makes no engine wait for the other beyond the data flow, so a recipe
// that holds half-steps of TWO tiles overlaps them when each queue alternates between the
// bodies. The compiler's own order keeps a body together; a user execution order
// (`GraphBuilder::set_user_exec_order`) changes each queue's order and adds no waits (the
// compiler's control edges are soft). Measured on the card (FP8, 512 rows): DeltaNet first half
// + second half 1134 us against 1267 us apart, attention 986 against 1090.
//
// Which half-steps share a launch is the verified schedule (`verified/prefill_wave.bend`); the
// node order inside a recipe is the sequence below, chosen with a per-engine queue model of the
// card and confirmed by card timings.

/// The bodies of one wavefront recipe: the first half of a layer of one tile and / or the second
/// half of a layer of the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaveKey {
    pub first: Option<FirstHalf>,
    pub second: bool,
}

/// Node order of a two-body recipe (see [`apply_seq_order`]), from the nodes the first half has
/// in this tile form. The second-half body is placed as its two GEMM clusters (output projection
/// + gate; up + activation + down); the compiler keeps each cluster together.
fn wave_seq(gb: &GraphBuilder, first: FirstHalf) -> Result<String, String> {
    let has = |name: &str| gb.node_io_views().iter().any(|n| n.name == format!("L_{name}"));
    let need = |names: &[&'static str]| -> Result<&'static str, String> {
        names
            .iter()
            .copied()
            .find(|n| has(n))
            .ok_or_else(|| format!("wavefront order: none of the nodes {names:?} in the first half"))
    };
    match first {
        FirstHalf::Lin => {
            let prep = need(&["gdn_prepa", "gdn_prep"])?;
            need(&["h_norm"])?;
            need(&["solve"])?;
            Ok(format!(
                "F:h_norm,F=ba_gemm,F:qkvz_gemm,G:attn_gemm,F={prep},G:gate_gemm,F:solve,G:*,F:*"
            ))
        }
        FirstHalf::Attn(_) => {
            let qk = need(&["g0_qk", "qk"])?;
            let sm = need(&["g0_softmax", "softmax"])?;
            let groups = (0..)
                .take_while(|g| has(&format!("g{g}_softmax")))
                .count();
            let head = "F:v_gemm,G:attn_gemm,F=attn_prep,G:gate_gemm";
            if groups <= 1 {
                // one chunk group: the second body's up / activation / down cluster after the
                // softmax, under the PV GEMM, the combine and the post kernel (card: 885 us
                // against 976 us with the cluster before the softmax, 1090 us apart)
                let pre = ["vnat", "qs16"].into_iter().find(|n| has(n));
                let pre = pre.map(|n| format!("F:{n},")).unwrap_or_default();
                Ok(format!("{head},{pre}F={qk},F={sm},G:*,F:*"))
            } else {
                // several chunk groups: the compiler's own order of the core (every QK GEMM,
                // then the softmaxes, then the PV GEMMs) pipelines the groups, and a full order
                // would serialise them (card, 4 groups: 1914 us against 1806 us apart). So only
                // the second body and a few nodes of the first get a position: the cluster
                // follows the softmaxes of up to 8 groups (card, 4 groups: 1686 us after all
                // four, 1705 us after two, 1789 us after one) and of half of them beyond that
                // (a 131072-token prefill, up to 16 groups: 32.7 s after half, 33.6 s after
                // all, 34.2 s with one tile in flight).
                let after = if groups <= 8 { groups } else { groups / 2 };
                let mid: String = (0..after).map(|g| format!("F=g{g}_softmax,")).collect();
                Ok(format!("~{head},{mid}G=up_gemm,G:*"))
            }
        }
    }
}

/// The wavefront recipe `key` of `t` rows: a half-layer graph, in the wavefront's node order
/// when it holds two bodies.
pub fn build_wave_graph(
    spec: &ModelSpec,
    t: u64,
    lmax: u64,
    key: WaveKey,
    form: &TileForm,
    elf_base: &[(String, String)],
    elf_tile: &[(String, String)],
) -> Result<GraphBuilder, String> {
    let mut gb = build_half_graph(
        spec,
        t,
        lmax,
        key.first,
        key.second,
        Weave::SecondFirst,
        form,
        elf_base,
        elf_tile,
    );
    if let (Some(first), true) = (key.first, key.second) {
        let seq = wave_seq(&gb, first)?;
        apply_seq_order(&mut gb, &seq)?;
        let name = gb.recipe_name().replace("g_qwen_half_G", "g_qwen_wave").replace("_wG", "");
        gb.set_recipe_name(name);
    }
    Ok(gb)
}

/// Device buffers of the wavefront beyond the single-tile loop's: per buffer set the layer
/// input / output rows, the hand-over rows between a layer's halves, the tile's ids and cursor,
/// and the drafter's tap rows. Set 0 reuses the single-tile loop's rows, ids, cursor and taps.
#[derive(Debug, Clone, Default)]
pub(super) struct WaveBufs {
    pub x: [[u64; 2]; 2],
    pub core: [u64; 2],
    pub ids: [u64; 2],
    pub cursor: [u64; 2],
    /// Tap rows of set 1 (set 0 uses the drafter's own).
    pub taps1: Vec<u64>,
}

// ============================================================================
// Scratch plan
// ============================================================================

/// Is `name` a tile-recipe tensor the tile loop binds itself (input / output rows, caches,
/// states, tables, weights)? Everything else is scratch laid out in the shared region.
fn is_loop_bound(name: &str) -> bool {
    matches!(
        name,
        "x_in"
            | "L_out"
            | "cursor"
            | "rope_c"
            | "rope_s"
            | "norm_ones"
            | "zero_slot"
            | "fp8_one"
            | "L_kcache"
            | "L_vcache"
            | "L_S_in"
            | "L_S_out"
            | "L_cs_in"
            | "L_cs_out"
    ) || is_generic_layer_weight(name)
}

/// Offsets of a tile graph's scratch tensors in the shared region, and the bytes they span.
fn scratch_layout(gb: &GraphBuilder) -> (BTreeMap<String, u64>, u64) {
    let mut map = BTreeMap::new();
    let mut off = 0u64;
    for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
        if is_loop_bound(&name) {
            continue;
        }
        map.insert(name, off);
        off += align_up(bytes.max(256), 256);
    }
    (map, off)
}

/// Per-request timers of the tile loop.
#[derive(Debug, Default)]
pub(super) struct TileTimers {
    pub tiles: u64,
    pub launches: u64,
    /// Host time in the ids / cursor uploads, in the launch calls (pools, checks, enqueue) and
    /// waiting for the tile's last launch.
    pub upload_ms: f64,
    pub submit_ms: f64,
    pub wait_ms: f64,
    /// Profile runs: device time per recipe class (launch call to its synchronisation).
    classes: BTreeMap<String, (u64, f64)>,
}

impl TileTimers {
    pub(super) fn class(&mut self, name: &str, ms: f64) {
        match self.classes.get_mut(name) {
            Some(e) => {
                e.0 += 1;
                e.1 += ms;
            }
            None => {
                self.classes.insert(name.to_string(), (1, ms));
            }
        }
    }
}

/// Tile-recipe state of one loaded engine: the forms, the tile width, the scratch region and
/// the scratch layout of every compiled tile recipe.
pub(super) struct TileRt {
    pub opts: PrefillOpts,
    pub form: TileForm,
    tile: u64,
    elf_tile: Vec<(String, String)>,
    scratch_base: u64,
    scratch_bytes: u64,
    /// Device address of the f32 constant 1.0 the FP8 cast of the half-width core reads.
    const_one: u64,
    attn_layout: HashMap<(u64, u64), BTreeMap<String, u64>>,
    lin_layout: HashMap<u64, BTreeMap<String, u64>>,
    pub timers: TileTimers,
    /// Wavefront: drafter tap count the two-tile loop has to carry (`None`: this load cannot use
    /// the wavefront), its recipes with their scratch layouts, and its buffers.
    wave_taps: Option<usize>,
    /// Wavefront recipes are compiled (the engine holds them, keyed like `wave_layout`).
    wave_on: bool,
    wave_layout: HashMap<WaveKey, BTreeMap<String, u64>>,
    pub(super) wave_bufs: Option<WaveBufs>,
}

impl TileRt {
    /// Resolves the forms for `spec` and fingerprints the kernels they add.
    pub(super) fn new(
        opts: PrefillOpts,
        spec: &ModelSpec,
        kenv: &KernelEnv,
        elf_base: &[(String, String)],
    ) -> Result<Self, String> {
        let form = TileForm::resolve(&opts, spec);
        let mut elf_tile = Vec::new();
        for (k, h) in kenv.fingerprint_hashes(&form.extra_elfs(spec))? {
            if !elf_base.iter().any(|(bk, _)| *bk == k) {
                elf_tile.push((k, h));
            }
        }
        Ok(Self {
            opts,
            form,
            tile: if opts.tile > 0 {
                opts.tile
            } else {
                form.tile_candidates(spec)[0]
            },
            elf_tile,
            scratch_base: 0,
            scratch_bytes: 0,
            const_one: 0,
            attn_layout: HashMap::new(),
            lin_layout: HashMap::new(),
            timers: TileTimers::default(),
            wave_taps: None,
            wave_on: false,
            wave_layout: HashMap::new(),
            wave_bufs: None,
        })
    }

    /// Does this load want the wavefront recipes? The two-tile recipes exist for the dense
    /// hybrid's tile forms (chunked DeltaNet, fast-exp attention) whose two layer kinds hand the
    /// same row width to the output projection, with a DeltaNet layer first (the single first
    /// half the wavefront starts with).
    pub(super) fn wave_wanted(&self, spec: &ModelSpec) -> bool {
        self.opts.wave
            && self.wave_taps.is_some()
            && !self.opts.profile
            && self.form.mme
            && self.form.fastexp
            && spec.n_layers >= 2
            && !spec.is_full_attention_layer(0)
            && spec.n_heads * spec.head_dim == spec.gdn_v_heads * spec.gdn_v_dim
    }

    /// Tells the tile loop how many drafter tap rows a tile carries (`None`: the loaded drafter
    /// has no two-tile form, so the wavefront stays off).
    pub(super) fn set_wave_taps(&mut self, taps: Option<usize>) {
        self.wave_taps = taps;
    }

    /// Bytes of the wavefront's own buffers at `tile` rows (0 when it is not wanted).
    pub(super) fn wave_buf_bytes(&self, spec: &ModelSpec, tile: u64) -> u64 {
        if !self.wave_wanted(spec) {
            return 0;
        }
        let a = |n: u64| align_up(n.max(256), 256);
        let x = a(spec.hidden * tile * 2);
        let core = a(spec.n_heads * spec.head_dim * tile * 2);
        let taps = self.wave_taps.unwrap_or(0) as u64;
        2 * x + 2 * core + a(tile * 4) + 256 + taps * x
    }

    /// Lays the wavefront's buffers out in the region planned for them (`base`, sized by
    /// [`Self::wave_buf_bytes`] at `tile` rows); set 0 takes the single-tile loop's buffers.
    pub(super) fn set_wave_addresses(
        &mut self,
        spec: &ModelSpec,
        tile: u64,
        base: u64,
        x0: [u64; 2],
        ids0: u64,
        cursor0: u64,
    ) {
        if base == 0 || !self.wave_on {
            return;
        }
        let a = |n: u64| align_up(n.max(256), 256);
        let x = a(spec.hidden * tile * 2);
        let core = a(spec.n_heads * spec.head_dim * tile * 2);
        let mut at = base;
        let mut take = |n: u64| {
            let p = at;
            at += n;
            p
        };
        let x1 = [take(x), take(x)];
        let core2 = [take(core), take(core)];
        let ids1 = take(a(tile * 4));
        let cursor1 = take(256);
        let taps1 = (0..self.wave_taps.unwrap_or(0)).map(|_| take(x)).collect();
        self.wave_bufs = Some(WaveBufs {
            x: [x0, x1],
            core: core2,
            ids: [ids0, ids1],
            cursor: [cursor0, cursor1],
            taps1,
        });
    }

    /// The wavefront recipe keys of a load: the single halves that start and end a wavefront and
    /// the two-body recipes of both layer kinds (one per KV chunk count for the attention half).
    fn wave_keys(spec: &ModelSpec, max_chunks: u64) -> Vec<WaveKey> {
        let mut keys = vec![
            WaveKey { first: Some(FirstHalf::Lin), second: false },
            WaveKey { first: None, second: true },
            WaveKey { first: Some(FirstHalf::Lin), second: true },
        ];
        if spec.n_full_attention_layers() > 0 {
            keys.extend((1..=max_chunks).map(|nch| WaveKey {
                first: Some(FirstHalf::Attn(nch)),
                second: true,
            }));
        }
        keys
    }

    /// Is the wavefront ready to run (its recipes compiled and its buffers placed)?
    pub(super) fn wave_ready(&self) -> bool {
        self.wave_on && self.wave_bufs.is_some()
    }

    /// Scratch bindings (and the cast constant) of the wavefront recipe `key`.
    pub(super) fn wave_pool(&self, key: WaveKey) -> BTreeMap<String, u64> {
        self.pool_of(self.wave_layout.get(&key))
    }

    /// Rows of a full tile (the widest candidate until [`Self::settle`] chose).
    pub(super) fn tile(&self) -> u64 {
        self.tile
    }

    /// `(base address, bytes)` of the planned scratch region.
    pub(super) fn scratch_region(&self) -> (u64, u64) {
        (self.scratch_base, self.scratch_bytes)
    }

    /// Hashes of the kernels the tile forms add to the engine's base set.
    pub(super) fn elf_tile(&self) -> &[(String, String)] {
        &self.elf_tile
    }

    /// Device address of the f32 constant 1.0 (None: no recipe of this form reads it).
    pub(super) fn const_one_addr(&self) -> Option<u64> {
        (self.const_one != 0).then_some(self.const_one)
    }

    /// Bytes of the `fp8_one` constant buffer (0 when no recipe reads it).
    pub(super) fn const_bytes(&self) -> u64 {
        if self.form.s16 >= 2 {
            256
        } else {
            0
        }
    }

    pub(super) fn attn_graph(
        &self,
        spec: &ModelSpec,
        t: u64,
        lmax: u64,
        nch: u64,
        elf_base: &[(String, String)],
    ) -> GraphBuilder {
        build_attn_tile_graph(spec, t, lmax, nch, &self.form, elf_base, &self.elf_tile)
    }

    pub(super) fn lin_graph(
        &self,
        spec: &ModelSpec,
        t: u64,
        elf_base: &[(String, String)],
    ) -> GraphBuilder {
        build_lin_tile_graph(spec, t, &self.form, elf_base, &self.elf_tile)
    }

    /// The scratch buffers `(name, bytes)` of the tile recipe whose scratch is the largest at
    /// `tile` rows (the one `plan_bytes` sizes the region for), for the itemised memory report.
    pub(super) fn plan_items(
        &self,
        spec: &ModelSpec,
        tile: u64,
        lmax: u64,
        max_chunks: u64,
    ) -> Vec<(String, u64)> {
        let items = |gb: &GraphBuilder, half: bool| -> (u64, Vec<(String, u64)>) {
            let v: Vec<(String, u64)> = gb
                .persistent_tensor_specs()
                .into_iter()
                .filter(|(name, ..)| !is_loop_bound(name) && !(half && is_half_bound(name)))
                .map(|(name, _dt, _dims, bytes)| (name, align_up(bytes.max(256), 256)))
                .collect();
            (v.iter().map(|x| x.1).sum(), v)
        };
        let mut best: (u64, Vec<(String, u64)>) = (0, Vec::new());
        let mut take = |c: (u64, Vec<(String, u64)>)| {
            if c.0 > best.0 {
                best = c;
            }
        };
        if spec.n_full_attention_layers() > 0 {
            for nch in 1..=max_chunks {
                take(items(&self.attn_graph(spec, tile, lmax, nch, &[]), false));
            }
        }
        if spec.n_linear_layers() > 0 {
            take(items(&self.lin_graph(spec, tile, &[]), false));
        }
        if self.wave_wanted(spec) {
            for key in Self::wave_keys(spec, max_chunks) {
                if let Ok(gb) = build_wave_graph(spec, tile, lmax, key, &self.form, &[], &self.elf_tile)
                {
                    take(items(&gb, true));
                }
            }
        }
        best.1
    }

    /// Scratch bytes of the largest tile recipe at `tile` rows: every chunk count up to
    /// `max_chunks` of the attention layer, and the DeltaNet layer.
    pub(super) fn plan_bytes(
        &self,
        spec: &ModelSpec,
        tile: u64,
        lmax: u64,
        max_chunks: u64,
    ) -> u64 {
        let mut bytes = 0u64;
        if spec.n_full_attention_layers() > 0 {
            for nch in 1..=max_chunks {
                let gb = self.attn_graph(spec, tile, lmax, nch, &[]);
                bytes = bytes.max(scratch_layout(&gb).1);
            }
        }
        if spec.n_linear_layers() > 0 {
            bytes = bytes.max(scratch_layout(&self.lin_graph(spec, tile, &[])).1);
        }
        if self.wave_wanted(spec) {
            for key in Self::wave_keys(spec, max_chunks) {
                if let Ok(gb) = build_wave_graph(spec, tile, lmax, key, &self.form, &[], &self.elf_tile)
                {
                    bytes = bytes.max(half_scratch_layout(&gb).1);
                }
            }
        }
        bytes
    }

    /// Chooses the tile width (the widest candidate for which `fits(scratch_bytes, workspace)`
    /// holds; a fixed width is taken as is), compiles that width's full-tile recipes into the
    /// engine's maps and returns `(scratch bytes, largest workspace)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn settle(
        &mut self,
        spec: &ModelSpec,
        lmax: u64,
        max_chunks: u64,
        elf_base: &[(String, String)],
        api: &std::sync::Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
        attn: &mut HashMap<(u64, u64), CompiledRecipe>,
        lin: &mut HashMap<u64, CompiledRecipe>,
        embed: &mut HashMap<u64, CompiledRecipe>,
        wave: &mut HashMap<WaveKey, CompiledRecipe>,
        verbose: bool,
        fits: &dyn Fn(u64, u64) -> bool,
    ) -> Result<(u64, u64), String> {
        let widest = self.form.tile_candidates(spec)[0];
        let candidates: Vec<u64> = if self.opts.tile > 0 {
            // [kv8] an FP8 K/V load takes no tile wider than its candidates (`tile_candidates`)
            if crate::verified::kv_is_fp8(spec.kv_dtype) && self.opts.tile > widest {
                eprintln!(
                    "[gaudium] prefill tile {} -> {widest} rows: FP8 K/V prefill tiles are at most {widest} rows",
                    self.opts.tile
                );
                vec![widest]
            } else {
                vec![self.opts.tile]
            }
        } else {
            self.form.tile_candidates(spec).to_vec()
        };
        let mut chosen = None;
        for (i, &tile) in candidates.iter().enumerate() {
            let last = i + 1 == candidates.len();
            let scratch = self.plan_bytes(spec, tile, lmax, max_chunks);
            if !last && !fits(scratch, 0) {
                continue;
            }
            // the full-tile recipes of this width, one attention recipe per chunk count a prompt
            // can reach: their workspaces are part of the plan, and no request pays for a
            // full-tile recipe inside its time to first token
            let t_build = Instant::now();
            let mut ws = 0u64;
            let mut built: Vec<((u64, u64), CompiledRecipe)> = Vec::new();
            if spec.n_full_attention_layers() > 0 && tile > 1 {
                for nch in 1..=max_chunks {
                    let gb = self.attn_graph(spec, tile, lmax, nch, elf_base);
                    let r = gb
                        .compile(std::sync::Arc::clone(api), cache, driver_version)
                        .map_err(|e| format!("compile prefill attn layer T={tile} C={nch}: {e}"))?;
                    ws = ws.max(r.ws_size);
                    built.push(((tile, nch), r));
                }
            }
            if !last && !fits(scratch, ws) {
                continue;
            }
            if verbose && !built.is_empty() {
                eprintln!(
                    "[gaudium] prefill attention recipes T={tile} C=1..{max_chunks}: {} from the cache, {} compiled, {:.2}s, largest ws={:.2} MiB",
                    built.iter().filter(|b| b.1.from_cache).count(),
                    built.iter().filter(|b| !b.1.from_cache).count(),
                    t_build.elapsed().as_secs_f64(),
                    ws as f64 / (1024.0 * 1024.0)
                );
            }
            for (key, r) in built {
                let gb = self.attn_graph(spec, key.0, lmax, key.1, &[]);
                self.attn_layout.insert(key, scratch_layout(&gb).0);
                attn.insert(key, r);
            }
            if spec.n_linear_layers() > 0 && tile > 1 {
                let gb = self.lin_graph(spec, tile, elf_base);
                let r = gb
                    .compile(std::sync::Arc::clone(api), cache, driver_version)
                    .map_err(|e| format!("compile prefill lin layer T={tile}: {e}"))?;
                if verbose {
                    eprintln!(
                        "[gaudium] recipe {} cache_hit={} ws={:.2} MiB",
                        r.name,
                        r.from_cache,
                        r.ws_size as f64 / (1024.0 * 1024.0)
                    );
                }
                ws = ws.max(r.ws_size);
                self.lin_layout.insert(tile, scratch_layout(&gb).0);
                lin.insert(tile, r);
            }
            if tile > 1 {
                let r = build_embed_graph(spec, tile)
                    .compile(std::sync::Arc::clone(api), cache, driver_version)
                    .map_err(|e| format!("compile embed T={tile}: {e}"))?;
                ws = ws.max(r.ws_size);
                embed.insert(tile, r);
            }
            // the wavefront recipes of this width (a compile or order failure leaves the
            // single-tile loop in charge: the wavefront is an overlap of the same nodes)
            wave.clear();
            self.wave_on = false;
            self.wave_layout.clear();
            if self.wave_wanted(spec) && tile > 1 {
                let t_wave = Instant::now();
                let mut built = Vec::new();
                let mut err = None;
                for key in Self::wave_keys(spec, max_chunks) {
                    let made = build_wave_graph(spec, tile, lmax, key, &self.form, elf_base, &self.elf_tile)
                        .and_then(|gb| {
                            let layout = half_scratch_layout(&gb);
                            gb.compile(std::sync::Arc::clone(api), cache, driver_version)
                                .map(|r| (r, layout))
                                .map_err(|e| format!("compile {}: {e}", gb.recipe_name()))
                        });
                    match made {
                        Ok((r, (layout, bytes))) if bytes <= scratch => built.push((key, r, layout)),
                        Ok((r, (_, bytes))) => {
                            err = Some(format!("{} needs {bytes} B of scratch, {scratch} B are planned", r.name));
                            break;
                        }
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                match err {
                    None => {
                        if verbose {
                            eprintln!(
                                "[gaudium] prefill wavefront recipes T={tile}: {} from the cache, {} compiled, {:.2}s",
                                built.iter().filter(|b| b.1.from_cache).count(),
                                built.iter().filter(|b| !b.1.from_cache).count(),
                                t_wave.elapsed().as_secs_f64()
                            );
                        }
                        for (key, r, layout) in built {
                            ws = ws.max(r.ws_size);
                            self.wave_layout.insert(key, layout);
                            wave.insert(key, r);
                        }
                        self.wave_on = true;
                    }
                    Some(e) => eprintln!(
                        "[gaudium] prefill wavefront off (single-tile recipes are used): {e}"
                    ),
                }
            }
            chosen = Some((tile, scratch, ws));
            break;
        }
        let (tile, scratch, ws) = chosen.expect("at least one tile candidate");
        self.tile = tile;
        self.scratch_bytes = scratch;
        if verbose {
            eprintln!(
                "[gaudium] prefill: tile {tile} rows, {} | scratch {:.2} GiB",
                self.form.label(),
                scratch as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            if self.wave_on {
                eprintln!("[gaudium] prefill: full tiles run two in flight (wavefront recipes)");
            }
        }
        Ok((scratch, ws))
    }

    /// Records where the planned scratch region (and the constant buffer) were allocated.
    pub(super) fn set_addresses(&mut self, scratch_base: u64, const_one: u64) {
        self.scratch_base = scratch_base;
        self.const_one = const_one;
    }

    /// Fills the constant buffer after the allocations are sealed.
    pub(super) fn finish_load(
        &self,
        dev: &crate::synapse::device::HpuDevice,
    ) -> Result<(), String> {
        if self.const_one != 0 {
            dev.copy_h2d_sync(&1.0f32.to_le_bytes(), self.const_one)
                .map_err(|e| format!("upload prefill constant: {e}"))?;
        }
        Ok(())
    }

    /// Registers the scratch layout of a newly built attention tile recipe; refuses a recipe
    /// whose scratch would not fit the planned region (nothing is allocated after sealing).
    pub(super) fn register_attn(
        &mut self,
        gb: &GraphBuilder,
        t: u64,
        nch: u64,
    ) -> Result<(), String> {
        let (map, bytes) = scratch_layout(gb);
        self.check_fits("attention", t, bytes)?;
        self.attn_layout.insert((t, nch), map);
        Ok(())
    }

    /// Registers the scratch layout of a newly built DeltaNet tile recipe.
    pub(super) fn register_lin(&mut self, gb: &GraphBuilder, t: u64) -> Result<(), String> {
        let (map, bytes) = scratch_layout(gb);
        self.check_fits("DeltaNet", t, bytes)?;
        self.lin_layout.insert(t, map);
        Ok(())
    }

    fn check_fits(&self, what: &str, t: u64, bytes: u64) -> Result<(), String> {
        if bytes > self.scratch_bytes {
            return Err(format!(
                "prefill {what} recipe of {t} rows needs {bytes} B of scratch, the plan holds {} B",
                self.scratch_bytes
            ));
        }
        Ok(())
    }

    fn pool_of(&self, layout: Option<&BTreeMap<String, u64>>) -> BTreeMap<String, u64> {
        let mut pool = BTreeMap::new();
        if let Some(l) = layout {
            for (name, &off) in l {
                pool.insert(name.clone(), self.scratch_base + off);
            }
        }
        if self.const_one != 0 {
            pool.insert("fp8_one".to_string(), self.const_one);
        }
        pool
    }

    /// Scratch bindings of the attention tile recipe `(t, nch)`.
    pub(super) fn attn_pool(&self, t: u64, nch: u64) -> BTreeMap<String, u64> {
        self.pool_of(self.attn_layout.get(&(t, nch)))
    }

    /// Scratch bindings of the DeltaNet tile recipe of `t` rows.
    pub(super) fn lin_pool(&self, t: u64) -> BTreeMap<String, u64> {
        self.pool_of(self.lin_layout.get(&t))
    }

    /// Starts the timers of a request.
    pub(super) fn begin_request(&mut self) {
        self.timers = TileTimers::default();
    }

    /// Prints the timers of the request that just prefilled (`profile` / `stats` only).
    pub(super) fn report(&self, prompt_tokens: usize, ttft_ms: f64) {
        if !self.opts.profile && !self.opts.stats {
            return;
        }
        let t = &self.timers;
        eprintln!(
            "[prefill] {prompt_tokens} tok, {} tiles of {} rows, {} launches in {ttft_ms:.1} ms: host uploads {:.1} ms, launch calls {:.1} ms, waiting {:.1} ms",
            t.tiles, self.tile, t.launches, t.upload_ms, t.submit_ms, t.wait_ms
        );
        if self.opts.profile {
            let total: f64 = t.classes.values().map(|e| e.1).sum();
            for (name, (n, ms)) in &t.classes {
                eprintln!(
                    "[prefill]   {name:<24} {n:>6} launches {ms:>10.1} ms  {:>8.3} ms each  {:>5.1} %",
                    ms / (*n as f64).max(1.0),
                    100.0 * ms / total.max(1e-9)
                );
            }
        }
    }
}

/// A clock for the tile loop's timers.
pub(super) fn now() -> Instant {
    Instant::now()
}

/// Milliseconds since `t0`.
pub(super) fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_spec() -> ModelSpec {
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
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4
          }
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    #[test]
    fn options_parse_and_reject() {
        assert_eq!(PrefillOpts::parse("").unwrap(), PrefillOpts::default());
        let o = PrefillOpts::parse("tile=256, groups=8,s16=1,profile").unwrap();
        assert_eq!(
            (o.tile, o.groups, o.s16, o.profile),
            (256, 8, Some(1), true)
        );
        let p = PrefillOpts::parse("stats,plain").unwrap();
        assert!(p.stats && !p.mme && !p.fastexp && p.tile == 256);
        assert!(PrefillOpts::parse("tile=100").is_err());
        assert!(PrefillOpts::parse("tile=1024").is_err());
        assert!(PrefillOpts::parse("s16=3").is_err());
        assert!(PrefillOpts::parse("warp=9").is_err());
    }

    #[test]
    fn chunk_groups_cover_every_chunk_once() {
        assert_eq!(chunk_groups(1, 16), vec![1]);
        assert_eq!(chunk_groups(8, 16), vec![1; 8]);
        assert_eq!(chunk_groups(17, 16), vec![2, 2, 2, 2, 2, 2, 2, 2, 1]);
        assert_eq!(chunk_groups(20, 16), vec![2; 10]);
        assert_eq!(chunk_groups(64, 16), vec![4; 16]);
        assert_eq!(chunk_groups(64, 1), vec![64]);
        for nch in 1..=64u64 {
            for g in [1u64, 4, 8, 16] {
                let v = chunk_groups(nch, g);
                assert_eq!(v.iter().sum::<u64>(), nch);
                assert!(v.len() as u64 <= g);
            }
        }
    }

    #[test]
    fn forms_resolve_per_model_and_precision() {
        let mut spec = dense_spec();
        let f = TileForm::resolve(&PrefillOpts::default(), &spec);
        assert!(f.mme && f.fastexp && f.rows == 2 && f.groups == 16 && f.s16 == 0);
        spec.fp8_weights = true;
        assert_eq!(TileForm::resolve(&PrefillOpts::default(), &spec).s16, 2);
        assert_eq!(
            TileForm::resolve(&PrefillOpts::parse("s16=0").unwrap(), &spec).s16,
            0
        );
        assert_eq!(
            TileForm::resolve(&PrefillOpts::plain(), &spec),
            TileForm::PLAIN
        );
        // the wide tile only with the forms it was measured with
        assert_eq!(f.tile_candidates(&spec), &[512, 256]);
        assert_eq!(TileForm::PLAIN.tile_candidates(&spec), &[256]);
        let serial = TileForm::resolve(&PrefillOpts::parse("mme=0").unwrap(), &spec);
        assert_eq!(serial.tile_candidates(&spec), &[256]);
    }

    #[test]
    fn wave_graphs_hold_both_halves_in_a_user_order() {
        for fp8 in [false, true] {
            let mut spec = dense_spec();
            spec.fp8_weights = fp8;
            let form = TileForm::resolve(&PrefillOpts::default(), &spec);
            for first in [FirstHalf::Lin, FirstHalf::Attn(1), FirstHalf::Attn(5)] {
                let f = build_half_graph(&spec, 512, 8192, Some(first), false, Weave::Cost, &form, &[], &[]);
                let g = build_half_graph(&spec, 512, 8192, None, true, Weave::Cost, &form, &[], &[]);
                let key = WaveKey { first: Some(first), second: true };
                let w = build_wave_graph(&spec, 512, 8192, key, &form, &[], &[]).unwrap();
                // the two halves, node for node
                assert_eq!(w.node_count(), f.node_count() + g.node_count());
                let mut names: Vec<String> =
                    w.node_io_views().iter().map(|n| n.name.to_string()).collect();
                let mut want: Vec<String> = f
                    .node_io_views()
                    .iter()
                    .chain(g.node_io_views().iter())
                    .map(|n| n.name.to_string())
                    .collect();
                names.sort();
                want.sort();
                assert_eq!(names, want);
                assert!(w.recipe_name().starts_with("g_qwen_wave"), "{}", w.recipe_name());
                // the order is part of what compiles: not the plain two-body recipe's key
                let plain = build_half_graph(
                    &spec, 512, 8192, Some(first), true, Weave::SecondFirst, &form, &[], &[],
                );
                assert_ne!(w.fingerprint("d"), plain.fingerprint("d"));
            }
            // the single halves at the ends of a wavefront are the half-layer recipes
            let key = WaveKey { first: Some(FirstHalf::Lin), second: false };
            let a = build_wave_graph(&spec, 512, 8192, key, &form, &[], &[]).unwrap();
            let b = build_half_graph(
                &spec, 512, 8192, Some(FirstHalf::Lin), false, Weave::SecondFirst, &form, &[], &[],
            );
            assert_eq!(a.fingerprint("d"), b.fingerprint("d"));
        }
    }

    #[test]
    fn seq_order_places_runs_and_single_nodes() {
        let mut spec = dense_spec();
        spec.fp8_weights = true;
        let form = TileForm::resolve(&PrefillOpts::default(), &spec);
        let mut gb = build_half_graph(
            &spec, 512, 8192, Some(FirstHalf::Lin), true, Weave::SecondFirst, &form, &[], &[],
        );
        let n = gb.node_count();
        apply_seq_order(&mut gb, "F:h_norm,F=ba_gemm,G:attn_gemm,F:*").unwrap();
        let names: Vec<String> = gb.node_io_views().iter().map(|x| x.name.to_string()).collect();
        assert_eq!(names.len(), n);
        assert_eq!(&names[..2], ["L_h_norm", "L_ba_gemm"]);
        let wo = names.iter().position(|x| x == "G_attn_gemm").unwrap();
        let qkvz = names.iter().position(|x| x == "L_qkvz_gemm").unwrap();
        assert!(wo < qkvz, "the second body's run comes before the rest of the first");
        // nodes no run names keep their creation order at the end
        assert!(names.iter().position(|x| x == "G_mo_gemm").unwrap() > qkvz);
        assert!(apply_seq_order(&mut gb, "F:no_such_node").is_err());
        assert!(apply_seq_order(&mut gb, "X:h_norm").is_err());
    }

    #[test]
    fn wave_option_parses_and_plain_turns_it_off() {
        assert!(PrefillOpts::default().wave);
        assert!(!PrefillOpts::parse("wave=0").unwrap().wave);
        assert!(!PrefillOpts::parse("plain").unwrap().wave);
        assert!(PrefillOpts::parse("plain,wave=1").unwrap().wave);
    }

    #[test]
    fn plain_form_builds_the_plain_recipes() {
        let spec = dense_spec();
        let elfs = vec![("K".to_string(), "h".to_string())];
        let a = build_attn_tile_graph(&spec, 256, 8192, 2, &TileForm::PLAIN, &elfs, &[]);
        let b = build_prefill_attn_layer_graph(&spec, 256, 8192, 2, &elfs);
        assert_eq!(a.fingerprint("d"), b.fingerprint("d"));
        let a = build_lin_tile_graph(&spec, 256, &TileForm::PLAIN, &elfs, &[]);
        let b = build_prefill_linear_layer_graph(&spec, 256, &elfs);
        assert_eq!(a.fingerprint("d"), b.fingerprint("d"));
    }

    #[test]
    fn tile_graphs_keep_the_loop_interface_and_plan_scratch() {
        let spec = dense_spec();
        let form = TileForm::resolve(&PrefillOpts::default(), &spec);
        let lin = build_lin_tile_graph(&spec, 512, &form, &[], &[]);
        let names: Vec<String> = lin
            .persistent_tensor_specs()
            .into_iter()
            .map(|s| s.0)
            .collect();
        for n in [
            "x_in",
            "L_out",
            "L_S_in",
            "L_S_out",
            "L_cs_in",
            "L_cs_out",
            "zero_slot",
        ] {
            assert!(names.iter().any(|x| x == n), "missing {n}");
        }
        // calibration sites keep their names
        for n in ["L_h", "L_core", "L_h2", "L_m"] {
            assert!(names.iter().any(|x| x == n), "missing {n}");
        }
        // 8 chunks of 64 rows: three nodes per chunk in the serial chain
        let nodes = lin.node_io_views();
        assert_eq!(nodes.iter().filter(|n| n.guid == "mult_fwd_f32").count(), 8);
        // a remainder tile pads to whole chunks
        let lin70 = build_lin_tile_graph(&spec, 70, &form, &[], &[]);
        assert!(lin70
            .persistent_tensor_specs()
            .iter()
            .any(|s| s.0 == "L_gb" && s.2 == vec![128, 128]));

        let att = build_attn_tile_graph(&spec, 256, 262144, 64, &form, &[], &[]);
        let nodes = att.node_io_views();
        assert_eq!(
            nodes
                .iter()
                .filter(|n| n.guid == "gaudium_attn_softmax_chunk")
                .count(),
            16
        );
        // grouping never changes the scratch a chunk count needs by more than the concat copies
        let one = TileForm { groups: 1, ..form };
        let att1 = build_attn_tile_graph(&spec, 256, 262144, 64, &one, &[], &[]);
        let (_, grouped) = scratch_layout(&att);
        let (_, single) = scratch_layout(&att1);
        assert!(grouped >= single && grouped < single + single / 8);

        // half-width core: BF16 scores and FP8 P halve the big buffers
        let h8 = TileForm { s16: 2, ..form };
        let att8 = build_attn_tile_graph(&spec, 256, 262144, 64, &h8, &[], &[]);
        assert!(scratch_layout(&att8).1 < grouped * 3 / 4);
        assert!(att8
            .persistent_tensor_specs()
            .iter()
            .any(|s| s.0 == "fp8_one"));
        assert!(!scratch_layout(&att8).0.contains_key("fp8_one"));
    }
}

