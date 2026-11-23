//! Image input of the Inkling 2 engine: the checkpoint's vision tower on the card, its host
//! preprocessing, and the hand-over of the image features to the text prefill.
//!
//! What the tower is (the checkpoint's `vision_config` and processor, read from the reference
//! implementation): no resize; an image is padded at its bottom and right to whole 40 x 40 tiles
//! (the pad value is -1 on the 0..255 scale) and normalised `(x - 127.5) / 127.5`; one image
//! token per tile; each tile is four 20 x 20 patches of 1200 values (y, x, channel) through a
//! bias-free linear to 1024; a learned 32 x 32 position table resampled bilinearly to the patch
//! grid; a class token in front of each image; a pre-LayerNorm; 23 pre-norm encoder layers with
//! layer scale, 16 heads of 64 with biases, a 2-D axial rotary code on interleaved pairs, an exact
//! GELU MLP 1024-4096-1024; no final norm; the class token dropped; four patches of a tile
//! concatenated (4096) through linear, exact GELU, linear to one 2304-wide embedding per token,
//! which replaces one placeholder row of the prompt.
//!
//! How it runs on the chip:
//!
//! - **Three recipes per bucket, launched per image.** The embedding recipe (one MME GEMM over the
//!   patch rows plus the position rows, then the pre-norm), one generic encoder-layer recipe bound
//!   at launch to each layer's weights (`vb_*` -> `VL<l>_*`), and the merger recipe. An image runs
//!   in a sequence of `1 + 4 * b` rows where `b` is the image's token count rounded up a ladder of
//!   bucket sizes (`ikv_round`): images of nearby sizes share their recipes, and the padding rows
//!   are taken out of attention by a per-key bias of -1e30 (finite, so a row never turns NaN).
//! - **Attention is one batched product per layer.** At the largest planned image (1024 tokens,
//!   4097 rows) the f32 scores of all 16 heads are 1.07 GB: the matrix engine computes them in one
//!   `batch_gemm` with the heads as the batch, the softmax runs in f32 on the TPCs, and the
//!   probabilities go back to the matrix engine in BF16 for `P . V`. The attention scale 1/8 is a
//!   power of two and is folded exactly into the query weights; the pair rotation of the rotary
//!   code is folded into a second copy of the query / key weights (`qr`, `kr`: rows permuted and
//!   negated, also exact), so the rotary product is two f32 multiplies and an add.
//! - **The class token costs nothing extra.** Column 0 of the patch rows is zero (the patch
//!   linear has no bias) and column 0 of the position rows holds `class + class position`, so the
//!   embedding GEMM and its add write the class row as well.
//! - **The merger reads the layer output in place.** Its input is bound one column into the last
//!   layer's buffer (the class row skipped), and a `{4096, b}` view of `{1024, 4 b}` is exactly the
//!   four patches of each tile side by side. It writes its rows straight into the request's
//!   feature buffer at the image's offset.
//! - **The hand-over is one recipe variant.** A prefill tile with placeholder rows runs
//!   `g_ik_embed_vis_T<t>` instead of the plain embedding recipe: the normed token embedding times
//!   a per-row keep flag (1 text, 0 image) plus a gather from the feature buffer (column 0 stays
//!   zero, which text rows read). Both terms are exact, so a text row is bit for bit the plain
//!   embedding and an image row bit for bit its feature row (the features bypass the norm).
//!
//! Every buffer is sized up front from the configured largest image and request
//! (`--vision-max-patches`, `--vision-max-tokens`) and joins the engine's sealed plan; an image or
//! a request beyond them is refused with the numbers. Decisions (the tile grid, the token limit,
//! the bucket, the sequence rows, live keys, feature columns and row selection) come from
//! `verified/inkling2_vision.bend`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use super::{select_exact_bindings, IkDims, Inkling2Engine};
use crate::core::vision::EmbRows;
use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};
use crate::json::{parse_json, JsonValue};
use crate::safetensors::SafetensorsStore;
use crate::synapse::device::{HbmArena, HpuDevice, PinnedBuf};
use crate::synapse::ffi::{SynDataType, SynapseApi};
use crate::synapse::graph::{CompiledRecipe, GraphBuilder, PreparedLaunch, TensorId};
use crate::synapse::ops::{as_param_bytes, TransposeParams};
use crate::synapse::recipe_cache::RecipeCache;
use crate::verified::{
    align_up, ikv_bucket, ikv_feat_cols, ikv_feat_row, ikv_feat_write_ok, ikv_key_live, ikv_round,
    ikv_rows_match, ikv_seq_rows, ikv_side, ikv_token_cap, ikv_tokens_fit, HbmCat,
};
use crate::vision::preprocess::{PreparedImage, RgbImage};
use crate::vision::{ImageProcessor, VisionReport, VisionSpec};

/// The reference configuration's image placeholder when `config.json` names none.
pub const DEFAULT_IMAGE_TOKEN_ID: i32 = 200_054;
const DEFAULT_IMAGE_TOKEN: &str = "<|unused_200054|>";
/// The template's marker in front of an image's placeholder.
const CONTENT_IMAGE: &str = "<|content_image|>";
/// Additive bias of a padding key: far below any score, finite so a softmax row never sees
/// `-inf - -inf`.
const DEAD_KEY_BIAS: f32 = -1.0e30;
/// Pinned host stage for the per-image inputs (uploaded in pieces of this size).
const STAGE_BYTES: usize = 16 * 1024 * 1024;

// ============================================================================
// Configuration
// ============================================================================

fn vision_config(root: &JsonValue) -> Option<&JsonValue> {
    root.get("vision_config")
        .filter(|v| v.as_object().is_some())
}

/// Whether a `config.json` describes an Inkling 2 vision tower.
pub fn is_inkling2_vision(config_json: &str) -> bool {
    parse_json(config_json).ok().is_some_and(|root| {
        vision_config(&root)
            .and_then(|vc| vc.get("model_type"))
            .and_then(JsonValue::as_str)
            == Some("inkling2_vision")
    })
}

/// Geometry of the tower, from `vision_config` (missing keys take the reference
/// configuration's defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IkvDims {
    pub hid: u64,
    pub layers: u64,
    pub nh: u64,
    pub dh: u64,
    pub inter: u64,
    pub patch: u64,
    pub merge: u64,
    pub grid: u64,
    pub proj: u64,
    pub out: u64,
    pub channels: u64,
    pub eps: f32,
    pub theta: f64,
}

impl IkvDims {
    pub fn from_config_json(config_json: &str) -> Result<Self, String> {
        let root = parse_json(config_json).map_err(|e| format!("config.json: {e}"))?;
        let vc = vision_config(&root).ok_or("config.json has no vision_config")?;
        let u = |k: &str, d: u64| vc.get(k).and_then(JsonValue::as_u64).unwrap_or(d);
        let hid = u("hidden_size", 1024);
        let ratio = vc
            .get("mlp_ratio")
            .and_then(JsonValue::as_f64)
            .unwrap_or(4.0);
        let inter = vc
            .get("intermediate_size")
            .and_then(JsonValue::as_u64)
            .unwrap_or((hid as f64 * ratio) as u64);
        let nh = u("num_attention_heads", 16);
        let theta = vc
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(JsonValue::as_f64)
            .or_else(|| vc.get("rope_theta").and_then(JsonValue::as_f64))
            .unwrap_or(10_000.0);
        let d = Self {
            hid,
            layers: u("num_hidden_layers", 23),
            nh,
            dh: if nh > 0 { hid / nh } else { 0 },
            inter,
            patch: vc
                .get("native_patch_size")
                .or_else(|| vc.get("patch_size"))
                .and_then(JsonValue::as_u64)
                .unwrap_or(20),
            merge: vc
                .get("adapter_merge_size")
                .or_else(|| vc.get("spatial_merge_size"))
                .and_then(JsonValue::as_u64)
                .unwrap_or(2),
            grid: u("posemb_grid_size", 32),
            proj: u("projection_hidden_dim", 4096),
            out: u("text_hidden_size", 4608),
            channels: u("num_channels", 3),
            eps: vc
                .get("layer_norm_eps")
                .and_then(JsonValue::as_f64)
                .unwrap_or(1e-5) as f32,
            theta,
        };
        if vc.get("projector_bias").and_then(JsonValue::as_bool) == Some(true) {
            return Err(
                "vision_config.projector_bias: a merger with biases is not supported".into(),
            );
        }
        if let Some(act) = vc.get("hidden_act").and_then(JsonValue::as_str) {
            if act != "gelu" {
                return Err(format!(
                    "vision_config.hidden_act {act}: only the exact GELU is supported"
                ));
            }
        }
        d.validate()?;
        Ok(d)
    }

    fn validate(&self) -> Result<(), String> {
        if self.hid == 0 || self.layers == 0 || self.nh == 0 || self.inter == 0 || self.proj == 0 {
            return Err("vision_config: sizes must be positive".into());
        }
        if self.hid % self.nh != 0 || self.dh % 4 != 0 {
            return Err(format!(
                "vision_config: hidden_size {} / heads {} must give a head width divisible by 4 (two rotary axes of pairs)",
                self.hid, self.nh
            ));
        }
        if self.merge != 2 {
            return Err(format!(
                "vision_config: merge size {} (2 x 2 only)",
                self.merge
            ));
        }
        if self.channels != 3 || self.patch == 0 || self.grid < 2 {
            return Err(
                "vision_config: 3 channels, a positive patch and a position grid of at least 2"
                    .into(),
            );
        }
        Ok(())
    }

    /// Patches that make one token.
    pub fn ppt(&self) -> u64 {
        self.merge * self.merge
    }

    /// Pixel side of one tile (one token).
    pub fn tile(&self) -> u64 {
        self.patch * self.merge
    }

    /// Values of one patch row.
    pub fn row_len(&self) -> u64 {
        self.channels * self.patch * self.patch
    }

    /// Width of one merger input row.
    pub fn merged(&self) -> u64 {
        self.hid * self.ppt()
    }
}

/// The server-side description of the tower (image processor, placeholder) for the media
/// pipeline: `VisionSpec` with the tiled processor.
pub fn vision_spec(config_json: &str, processor_json: Option<&str>) -> Result<VisionSpec, String> {
    let d = IkvDims::from_config_json(config_json)?;
    let root = parse_json(config_json).map_err(|e| format!("config.json: {e}"))?;
    let image_token_id = root
        .get("image_token_id")
        .and_then(JsonValue::as_i64)
        .map_or(DEFAULT_IMAGE_TOKEN_ID, |v| v as i32);
    let mut mean = [0.5f64; 3];
    let mut std = [0.5f64; 3];
    let mut token = DEFAULT_IMAGE_TOKEN.to_string();
    if let Some(text) = processor_json {
        let p = parse_json(text).map_err(|e| format!("processor_config.json: {e}"))?;
        if let Some(t) = p.get("image_token").and_then(JsonValue::as_str) {
            token = t.to_string();
        }
        let ip = p.get("image_processor").unwrap_or(&p);
        let three = |k: &str| -> Option<[f64; 3]> {
            let a = ip.get(k)?.as_array()?;
            if a.len() != 3 {
                return None;
            }
            Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?])
        };
        if let Some(v) = three("image_mean") {
            mean = v;
        }
        if let Some(v) = three("image_std") {
            std = v;
        }
        if let Some(size) = ip.get("size") {
            for k in ["height", "width"] {
                if let Some(v) = size.get(k).and_then(JsonValue::as_u64) {
                    if v != d.tile() {
                        return Err(format!(
                            "processor_config.json size.{k} {v} is not the tower's tile ({} pixels)",
                            d.tile()
                        ));
                    }
                }
            }
        }
    }
    if std.contains(&0.0) {
        return Err("image_std holds a zero".into());
    }
    let s = VisionSpec {
        processor: ImageProcessor::Tiles,
        image_block: format!("{CONTENT_IMAGE}{token}"),
        depth: d.layers,
        hidden: d.hid,
        n_heads: d.nh,
        intermediate: d.inter,
        in_channels: d.channels,
        patch: d.patch,
        // the patch linear takes one frame (`num_channels * patch^2` inputs)
        temporal: 1,
        merge: d.merge,
        out_hidden: d.out,
        n_pos: d.grid * d.grid,
        gelu_tanh: false,
        gelu_bf16: false,
        image_token_id,
        video_token_id: -1,
        vision_start_id: -1,
        vision_end_id: -1,
        mrope_section: [0; 3],
        min_pixels: 1,
        max_pixels: u64::MAX,
        image_mean: mean,
        image_std: std,
        video: None,
    };
    s.validate()?;
    Ok(s)
}

// ============================================================================
// Host preprocessing
// ============================================================================

/// Tiles of an image: `(rows, cols)`.
pub fn tile_grid(width: usize, height: usize, tile: u64) -> (u64, u64) {
    (ikv_side(height as u64, tile), ikv_side(width as u64, tile))
}

/// The padded `(height, width)` of a `width x height` image (whole tiles).
pub fn padded_size(width: usize, height: usize, vs: &VisionSpec) -> Result<(usize, usize), String> {
    if width == 0 || height == 0 {
        return Err("image has zero size".into());
    }
    let tile = vs.patch * vs.merge;
    let (rows, cols) = tile_grid(width, height, tile);
    Ok(((rows * tile) as usize, (cols * tile) as usize))
}

/// f32 mean and std on the 0..255 scale (the processor folds its rescale into them).
fn mean_std_255(vs: &VisionSpec) -> ([f32; 3], [f32; 3]) {
    let mut mean = [0f32; 3];
    let mut std = [0f32; 3];
    for c in 0..3 {
        mean[c] = (vs.image_mean[c] * (1.0 / (1.0 / 255.0))) as f32;
        std[c] = (vs.image_std[c] * (1.0 / (1.0 / 255.0))) as f32;
    }
    (mean, std)
}

/// Walks the padded image in patch-row order, calling `put(value_on_0_255_scale, channel)`:
/// tiles row-major, a tile's four patches row-major, a patch's pixels row-major, the channels of
/// a pixel. Pixels outside the image are the pad value -1.
fn for_each_tile_value(
    img: &RgbImage,
    vs: &VisionSpec,
    mut put: impl FnMut(f32, usize),
) -> (usize, usize) {
    let (p, m) = (vs.patch as usize, vs.merge as usize);
    let tile = p * m;
    let (rows, cols) = tile_grid(img.width, img.height, tile as u64);
    let (rows, cols) = (rows as usize, cols as usize);
    for tr in 0..rows {
        for tc in 0..cols {
            for mh in 0..m {
                for mw in 0..m {
                    let y0 = tr * tile + mh * p;
                    let x0 = tc * tile + mw * p;
                    for py in 0..p {
                        let y = y0 + py;
                        for px in 0..p {
                            let x = x0 + px;
                            if y < img.height && x < img.width {
                                let at = (y * img.width + x) * 3;
                                for c in 0..3 {
                                    put(f32::from(img.data[at + c]), c);
                                }
                            } else {
                                for c in 0..3 {
                                    put(-1.0, c);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (rows * m, cols * m)
}

/// The processor's `pixel_values` in patch-row order (f32, before the tower's BF16 cast) with the
/// patch grid `(rows, cols)`.
pub fn tile_rows_f32(img: &RgbImage, vs: &VisionSpec) -> Result<(usize, usize, Vec<f32>), String> {
    let (mean, std) = mean_std_255(vs);
    let mut out = Vec::new();
    let (gh, gw) = for_each_tile_value(img, vs, |v, c| out.push((v - mean[c]) / std[c]));
    Ok((gh, gw, out))
}

/// A tower input: BF16 patch rows of the padded image.
pub fn prepare_tiles(img: &RgbImage, vs: &VisionSpec) -> Result<PreparedImage, String> {
    let (rh, rw) = padded_size(img.width, img.height, vs)?;
    let (mean, std) = mean_std_255(vs);
    let row_len = vs.patch_row() as usize;
    let mut pixels =
        Vec::with_capacity((rh / vs.patch as usize) * (rw / vs.patch as usize) * row_len);
    let (grid_h, grid_w) = for_each_tile_value(img, vs, |v, c| {
        pixels.push(f32_to_bf16_rne((v - mean[c]) / std[c]))
    });
    Ok(PreparedImage {
        src_width: img.width,
        src_height: img.height,
        width: rw,
        height: rh,
        grid_h,
        grid_w,
        row_len,
        pixels,
    })
}

/// `(row, col)` in the patch grid of patch `k` in patch-row order (tile raster, then the tile's
/// 2 x 2 patches).
pub fn patch_pos(k: usize, grid_w: usize, merge: usize) -> (usize, usize) {
    let mm = merge * merge;
    let tok = k / mm;
    let within = k % mm;
    let tiles_w = grid_w / merge;
    (
        (tok / tiles_w) * merge + within / merge,
        (tok % tiles_w) * merge + within % merge,
    )
}

/// One axis of the bilinear resample of a `side`-long table to `size` positions (half-pixel
/// centres, border clamped): the two taps and their weights at position `i`, in f32 as the
/// reference computes them.
pub fn axis_taps(i: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let src = (i as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let fl = src.floor();
    let mut taps = [0usize; 2];
    let mut w = [0f32; 2];
    for o in 0..2 {
        let raw = fl as i64 + o as i64;
        taps[o] = raw.clamp(0, side as i64 - 1) as usize;
        let dist = (src - fl - o as f32).abs();
        w[o] = (1.0 - dist).max(0.0);
    }
    (taps, w)
}

/// The four table rows and weights of every patch of a `grid_h x grid_w` patch grid, in patch-row
/// order: taps `(h0 w0, h0 w1, h1 w0, h1 w1)`, index `h * side + w`.
pub fn interp_taps(
    grid_h: usize,
    grid_w: usize,
    side: usize,
    merge: usize,
) -> Vec<([usize; 4], [f32; 4])> {
    (0..grid_h * grid_w)
        .map(|k| {
            let (r, c) = patch_pos(k, grid_w, merge);
            let (ht, hw) = axis_taps(r, grid_h, side);
            let (wt, ww) = axis_taps(c, grid_w, side);
            let mut idx = [0usize; 4];
            let mut wt4 = [0f32; 4];
            for a in 0..2 {
                for b in 0..2 {
                    idx[a * 2 + b] = ht[a] * side + wt[b];
                    wt4[a * 2 + b] = hw[a] * ww[b];
                }
            }
            (idx, wt4)
        })
        .collect()
}

/// Resampled position rows (BF16, `hid` per patch) from the BF16 table `[side^2][hid]`: the
/// weighted sum of the four rows in f32, rounded once.
pub fn pos_rows(table: &[u16], hid: usize, taps: &[([usize; 4], [f32; 4])]) -> Vec<u16> {
    let mut out = Vec::with_capacity(taps.len() * hid);
    for (idx, w) in taps {
        for j in 0..hid {
            let mut acc = 0f32;
            for t in 0..4 {
                acc += bf16_to_f32(table[idx[t] * hid + j]) * w[t];
            }
            out.push(f32_to_bf16_rne(acc));
        }
    }
    out
}

/// Inverse frequencies of one rotary axis: `1 / theta^(2i / (dh / 2))` for `i < dh / 4`, in f32.
pub fn inv_freq(dh: u64, theta: f64) -> Vec<f32> {
    let spatial = (dh / 2) as f32;
    (0..dh / 4)
        .map(|i| {
            let e = (2 * i) as f32 / spatial;
            (1.0f64 / theta.powf(f64::from(e))) as f32
        })
        .collect()
}

/// Rotary position `(h, w)` of every sequence row: the class row `(0, 0)`, then the patches at
/// `(row + 1, col + 1)` in patch-row order.
pub fn rope_positions(grid_h: usize, grid_w: usize, merge: usize) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(1 + grid_h * grid_w);
    out.push((0, 0));
    for k in 0..grid_h * grid_w {
        let (r, c) = patch_pos(k, grid_w, merge);
        out.push((r as u32 + 1, c as u32 + 1));
    }
    out
}

/// `cos` and `sin` of one row's head (`dh` values): the width angles then the height angles,
/// each repeated for the two channels of its pair.
pub fn rope_row(pos: (u32, u32), inv: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut cs = Vec::with_capacity(inv.len() * 4);
    let mut sn = Vec::with_capacity(inv.len() * 4);
    for axis in [pos.1, pos.0] {
        for &f in inv {
            let a = axis as f32 * f;
            let (s, c) = (f64::from(a).sin() as f32, f64::from(a).cos() as f32);
            cs.extend([c, c]);
            sn.extend([s, s]);
        }
    }
    (cs, sn)
}

// ============================================================================
// Graphs
// ============================================================================

/// `ns_LayerNormKernel::Params` (`layer_norm_fwd_bf16`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LayerNormParams {
    eps_valid: u8,
    _pad: [u8; 3],
    eps: f32,
}

/// `ns_GeluKernel::Params` (`gelu_fwd_f32`); 0 = the exact (erf) form.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GeluParams {
    approximation: u8,
}

/// `ns_Softmax::Params` (`softmax_fwd_f32`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SoftmaxParams {
    dim: i32,
}

/// `ns_CastKernel::Params`; 0 = round half to even.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CastParams {
    round_mode: u32,
}

const _: () = {
    assert!(std::mem::size_of::<LayerNormParams>() == 8);
    assert!(std::mem::size_of::<GeluParams>() == 1);
    assert!(std::mem::size_of::<SoftmaxParams>() == 4);
    assert!(std::mem::size_of::<CastParams>() == 4);
};

/// Weight kinds of one encoder layer: `vb_<kind>` in the generic recipe, `VL<l>_<kind>` on the
/// card.
pub const LAYER_WEIGHTS: &[&str] = &[
    "n1_w", "n1_b", "q_W", "q_b", "qr_W", "qr_b", "k_W", "k_b", "kr_W", "kr_b", "v_W", "v_b",
    "o_W", "o_b", "ls1", "n2_w", "n2_b", "fc1_W", "fc1_b", "fc2_W", "fc2_b", "ls2",
];

/// Tensors bound per launch to the layer buffers, the feature buffer or the text prefill.
const CHAIN_TENSORS: &[&str] = &["v_x0", "vb_x_in", "vb_x_out", "vm_x_in", "v_feat"];
/// Tensors of the hand-over recipe that belong to the text engine.
const TEXT_TENSORS: &[&str] = &["ids", "embed", "w_embed", "x0g", "x0"];

struct Tb<'a> {
    gb: &'a mut GraphBuilder,
    d: &'a IkvDims,
    n: u64,
}

impl Tb<'_> {
    fn act(&mut self, name: impl Into<String>, dt: SynDataType, dims: &[u64]) -> TensorId {
        self.gb.add_persistent_tensor(name, dt, dims)
    }

    /// `y {nout, cols} = W {k, nout}^T . x {k, cols}` (+ `b {nout, 1}`).
    fn lin(
        &mut self,
        x: TensorId,
        w: &str,
        k: u64,
        nout: u64,
        cols: u64,
        bias: bool,
        y: &str,
    ) -> TensorId {
        let bf = SynDataType::Bf16;
        let wt = self
            .gb
            .get_or_add_persistent_tensor(format!("{w}_W"), bf, &[k, nout]);
        if !bias {
            let o = self.act(y, bf, &[nout, cols]);
            self.gb.op_gemm(format!("{y}_gemm"), x, wt, o, false, true);
            return o;
        }
        let b = self
            .gb
            .get_or_add_persistent_tensor(format!("{w}_b"), bf, &[nout, 1]);
        let g = self.act(format!("{y}_mm"), bf, &[nout, cols]);
        let o = self.act(y, bf, &[nout, cols]);
        self.gb.op_gemm(format!("{y}_gemm"), x, wt, g, false, true);
        self.gb.op_add_bf16(format!("{y}_bias"), g, b, o);
        o
    }

    /// LayerNorm over the feature dimension (weight and bias in f32).
    fn ln(&mut self, x: TensorId, w: &str, c: u64, cols: u64, y: &str) -> TensorId {
        let f = SynDataType::Float;
        let bf = SynDataType::Bf16;
        let b = self
            .gb
            .get_or_add_persistent_tensor(format!("{w}_b"), f, &[c]);
        let g = self
            .gb
            .get_or_add_persistent_tensor(format!("{w}_w"), f, &[c]);
        let x4 = self
            .gb
            .add_intermediate(format!("{y}_x4"), bf, &[c, cols, 1, 1]);
        let y4 = self
            .gb
            .add_intermediate(format!("{y}_y4"), bf, &[c, cols, 1, 1]);
        let mean = self
            .gb
            .add_intermediate(format!("{y}_mean"), f, &[1, cols, 1, 1]);
        let rstd = self
            .gb
            .add_intermediate(format!("{y}_rstd"), f, &[1, cols, 1, 1]);
        let o = self.act(y, bf, &[c, cols]);
        self.gb.op_reshape(format!("{y}_r4"), x, x4);
        let p = LayerNormParams {
            eps_valid: 1,
            _pad: [0; 3],
            eps: self.d.eps,
        };
        self.gb.add_node(
            "layer_norm_fwd_bf16",
            format!("{y}_ln"),
            &[x4, b, g],
            &[y4, mean, rstd],
            as_param_bytes(&p),
        );
        self.gb.op_reshape(format!("{y}_r2"), y4, o);
        o
    }

    fn cast(&mut self, x: TensorId, to: SynDataType, dims: &[u64], y: &str) -> TensorId {
        let o = self.act(y, to, dims);
        let guid = if to == SynDataType::Float {
            "cast_bf16_to_f32"
        } else {
            "cast_f32_to_bf16"
        };
        let p = CastParams { round_mode: 0 };
        self.gb
            .add_node(guid, format!("{y}_cast"), &[x], &[o], as_param_bytes(&p));
        o
    }

    /// Exact GELU in f32 on BF16 rows, rounded back once.
    fn gelu(&mut self, x: TensorId, c: u64, cols: u64, y: &str) -> TensorId {
        let f = SynDataType::Float;
        let xf = self.cast(x, f, &[c, cols], &format!("{y}_xf"));
        let of = self.act(format!("{y}_of"), f, &[c, cols]);
        let aux = self.act(format!("{y}_aux"), f, &[c, cols]);
        let p = GeluParams { approximation: 0 };
        self.gb.add_node(
            "gelu_fwd_f32",
            format!("{y}_gelu"),
            &[xf],
            &[of, aux],
            as_param_bytes(&p),
        );
        self.cast(of, SynDataType::Bf16, &[c, cols], y)
    }

    fn bin(
        &mut self,
        guid: &str,
        a: TensorId,
        b: TensorId,
        dt: SynDataType,
        dims: &[u64],
        y: &str,
    ) -> TensorId {
        let o = self.act(y, dt, dims);
        self.gb
            .add_node(guid, format!("{y}_{guid}"), &[a, b], &[o], &[]);
        o
    }

    fn swap_heads() -> TransposeParams {
        TransposeParams {
            permutation: [0, 2, 1, 0, 0],
            tensor_dim: 3,
        }
    }

    /// `{hid, n}` -> `{dh, nh, n}` -> `{dh, n, nh}`: heads become the batch dimension.
    fn heads(&mut self, x: TensorId, y: &str) -> TensorId {
        let (dh, nh, n) = (self.d.dh, self.d.nh, self.n);
        let r = self.act(format!("{y}_r3"), SynDataType::Bf16, &[dh, nh, n]);
        self.gb.op_reshape(format!("{y}_reshape"), x, r);
        let t = self.act(format!("{y}_h"), SynDataType::Bf16, &[dh, n, nh]);
        let p = Self::swap_heads();
        self.gb.add_node(
            "transpose",
            format!("{y}_transpose"),
            &[r],
            &[t],
            as_param_bytes(&p),
        );
        t
    }

    /// `{dh, n, nh}` -> `{dh, nh, n}` -> `{hid, n}`.
    fn unheads(&mut self, t: TensorId, y: &str) -> TensorId {
        let (dh, nh, n, h) = (self.d.dh, self.d.nh, self.n, self.d.hid);
        let r = self.act(format!("{y}_t3"), SynDataType::Bf16, &[dh, nh, n]);
        let p = Self::swap_heads();
        self.gb.add_node(
            "transpose",
            format!("{y}_transpose"),
            &[t],
            &[r],
            as_param_bytes(&p),
        );
        let o = self.act(y, SynDataType::Bf16, &[h, n]);
        self.gb.op_reshape(format!("{y}_reshape"), r, o);
        o
    }

    /// Rotary product in f32: `e = x * cos + xr * sin` (`xr` = the pair-rotated rows from the
    /// folded weights), rounded to BF16, heads to the batch dimension.
    fn rope_heads(
        &mut self,
        x: TensorId,
        xr: TensorId,
        cs: TensorId,
        sn: TensorId,
        y: &str,
    ) -> TensorId {
        let f = SynDataType::Float;
        let dims = [self.d.hid, self.n];
        let xf = self.cast(x, f, &dims, &format!("{y}_f"));
        let xrf = self.cast(xr, f, &dims, &format!("{y}_rf"));
        let a = self.bin("mult_fwd_f32", xf, cs, f, &dims, &format!("{y}_xc"));
        let b = self.bin("mult_fwd_f32", xrf, sn, f, &dims, &format!("{y}_xs"));
        let e = self.bin("add_fwd_f32", a, b, f, &dims, &format!("{y}_e"));
        let eb = self.cast(e, SynDataType::Bf16, &dims, &format!("{y}_eb"));
        self.heads(eb, y)
    }
}

/// `g_ikv_embed_S<n>`: patch rows and position rows -> embeddings (`v_emb`) -> pre-norm (`v_x0`).
pub fn build_embed_graph(d: &IkvDims, n: u64) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_ikv_embed_S{n}"));
    let bf = SynDataType::Bf16;
    let mut b = Tb { gb: &mut gb, d, n };
    let pix = b.act("v_pix", bf, &[d.row_len(), n]);
    let pos = b.act("v_pos", bf, &[d.hid, n]);
    let pe = b.lin(pix, "VW_patch", d.row_len(), d.hid, n, false, "v_pe");
    let emb = b.bin("add_fwd_bf16", pe, pos, bf, &[d.hid, n], "v_emb");
    b.ln(emb, "VW_pre", d.hid, n, "v_x0");
    gb
}

/// `g_ikv_block_S<n>`: one generic pre-norm encoder layer (`vb_x_in` -> `vb_x_out`).
pub fn build_block_graph(d: &IkvDims, n: u64) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_ikv_block_S{n}"));
    let (bf, f) = (SynDataType::Bf16, SynDataType::Float);
    let (h, nh, dh) = (d.hid, d.nh, d.dh);
    let mut b = Tb { gb: &mut gb, d, n };
    let x = b.act("vb_x_in", bf, &[h, n]);
    let cs = b.act("v_cos", f, &[h, n]);
    let sn = b.act("v_sin", f, &[h, n]);
    let kb = b.act("v_kbias", f, &[n, 1, 1]);

    let h1 = b.ln(x, "vb_n1", h, n, "vb_h1");
    let q = b.lin(h1, "vb_q", h, h, n, true, "vb_q");
    let qr = b.lin(h1, "vb_qr", h, h, n, true, "vb_qr");
    let k = b.lin(h1, "vb_k", h, h, n, true, "vb_k");
    let kr = b.lin(h1, "vb_kr", h, h, n, true, "vb_kr");
    let v = b.lin(h1, "vb_v", h, h, n, true, "vb_v");
    let qh = b.rope_heads(q, qr, cs, sn, "vb_qe");
    let kh = b.rope_heads(k, kr, cs, sn, "vb_ke");
    let vh = b.heads(v, "vb_vh");

    let scores = b.act("vb_scores", f, &[n, n, nh]);
    b.gb.op_batch_gemm("vb_qk", qh, kh, scores, false, true);
    let masked = b.bin("add_fwd_f32", scores, kb, f, &[n, n, nh], "vb_masked");
    let pf = b.act("vb_pf", f, &[n, n, nh]);
    let sp = SoftmaxParams { dim: 0 };
    b.gb.add_node(
        "softmax_fwd_f32",
        "vb_softmax",
        &[masked],
        &[pf],
        as_param_bytes(&sp),
    );
    let pb = b.cast(pf, bf, &[n, n, nh], "vb_pb");
    let ao = b.act("vb_ao", bf, &[dh, n, nh]);
    b.gb.op_batch_gemm("vb_pv", pb, vh, ao, false, false);
    let a = b.unheads(ao, "vb_attn");
    let o = b.lin(a, "vb_o", h, h, n, true, "vb_o");
    let ls1 = b.gb.get_or_add_persistent_tensor("vb_ls1", bf, &[h, 1]);
    let os = b.bin("mult_fwd_bf16", o, ls1, bf, &[h, n], "vb_os");
    let x2 = b.bin("add_fwd_bf16", x, os, bf, &[h, n], "vb_x2");

    let h2 = b.ln(x2, "vb_n2", h, n, "vb_h2");
    let f1 = b.lin(h2, "vb_fc1", h, d.inter, n, true, "vb_f1");
    let g = b.gelu(f1, d.inter, n, "vb_g");
    let f2 = b.lin(g, "vb_fc2", d.inter, h, n, true, "vb_f2");
    let ls2 = b.gb.get_or_add_persistent_tensor("vb_ls2", bf, &[h, 1]);
    let fs = b.bin("mult_fwd_bf16", f2, ls2, bf, &[h, n], "vb_fs");
    b.bin("add_fwd_bf16", x2, fs, bf, &[h, n], "vb_x_out");
    gb
}

/// `g_ikv_merge_T<tb>`: the patch rows of `tb` tokens (`vm_x_in {hid, 4 tb}`, bound behind the
/// class row) -> features `v_feat {out, tb}`.
pub fn build_merge_graph(d: &IkvDims, tb: u64) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_ikv_merge_T{tb}"));
    let bf = SynDataType::Bf16;
    let mut b = Tb {
        gb: &mut gb,
        d,
        n: tb * d.ppt(),
    };
    let x = b.act("vm_x_in", bf, &[d.hid, tb * d.ppt()]);
    let r = b.act("vm_r", bf, &[d.merged(), tb]);
    b.gb.op_reshape("vm_reshape", x, r);
    let f1 = b.lin(r, "VM_in", d.merged(), d.proj, tb, false, "vm_f1");
    let g = b.gelu(f1, d.proj, tb, "vm_g");
    b.lin(g, "VM_out", d.proj, d.out, tb, false, "v_feat");
    gb
}

/// `g_ik_embed_vis_T<t>`: the text prefill's embedding stage for a tile with image rows:
/// `x0 = norm(embed[ids]) * keep + feat[fidx]`.
pub fn build_text_embed_graph(
    td: &IkDims,
    feat_cols: u64,
    t: u64,
    elf_hashes: &[(String, String)],
) -> GraphBuilder {
    let mut gb = super::ik_graph(format!("g_ik_embed_vis_T{t}"), elf_hashes);
    let (bf, i32t) = (SynDataType::Bf16, SynDataType::Int32);
    let ids = gb.add_persistent_tensor("ids", i32t, &[t]);
    let table = gb.get_or_add_persistent_tensor("embed", bf, &[td.hid, td.vocab_in]);
    let w = gb.get_or_add_persistent_tensor("w_embed", bf, &[td.hid]);
    let e = gb.add_persistent_tensor("x0g", bf, &[td.hid, t]);
    let xn = gb.add_persistent_tensor("vis_xn", bf, &[td.hid, t]);
    gb.op_gather_bf16("embed_gather", table, ids, e, 1);
    gb.add_node(
        "gaudium_ga_rmsnorm",
        "embed_norm",
        &[e, w],
        &[xn],
        &td.eps.to_le_bytes(),
    );
    let feat = gb.add_persistent_tensor("vis_feat", bf, &[td.hid, feat_cols]);
    let fidx = gb.add_persistent_tensor("vis_fidx", i32t, &[t]);
    let fg = gb.add_persistent_tensor("vis_fg", bf, &[td.hid, t]);
    gb.op_gather_bf16("vis_gather", feat, fidx, fg, 1);
    let keep = gb.add_persistent_tensor("vis_keep", bf, &[1, t]);
    let xk = gb.add_persistent_tensor("vis_xk", bf, &[td.hid, t]);
    gb.op_mult_bf16("vis_keep_mult", xn, keep, xk);
    let x0 = gb.add_persistent_tensor("x0", bf, &[td.hid, t]);
    gb.op_add_bf16("vis_select", xk, fg, x0);
    gb
}

fn generic_layer_weight(name: &str) -> Option<&str> {
    let kind = name.strip_prefix("vb_")?;
    LAYER_WEIGHTS.contains(&kind).then_some(kind)
}

fn align256(n: u64) -> u64 {
    align_up(n.max(256), 256)
}

// ============================================================================
// Plan
// ============================================================================

/// What the tower holds on the card, from the graph descriptions alone.
#[derive(Debug, Clone, PartialEq)]
pub struct IkvPlan {
    pub d: IkvDims,
    /// Largest image, in tokens, and its sequence rows.
    pub cap: u64,
    pub max_rows: u64,
    /// Image tokens one request may hold, and the feature buffer's columns.
    pub max_tokens: u64,
    pub feat_cols: u64,
    pub weights: BTreeMap<String, u64>,
    pub acts: BTreeMap<String, u64>,
    pub xbuf_bytes: u64,
    pub feat_bytes: u64,
}

impl IkvPlan {
    /// Plans the tower for images of up to `max_patches` patches and requests of up to
    /// `max_tokens` image tokens, beside a text prefill of at most `tile` rows.
    pub fn new(
        d: IkvDims,
        text: &IkDims,
        max_patches: u64,
        max_tokens: u64,
        tile: u64,
    ) -> Result<Self, String> {
        if d.out != text.hid {
            return Err(format!(
                "vision text_hidden_size {} != the decoder's hidden size {}",
                d.out, text.hid
            ));
        }
        let cap = ikv_token_cap(max_patches, d.ppt());
        if cap == 0 {
            return Err(format!(
                "--vision-max-patches {max_patches} holds no image token ({} patches each)",
                d.ppt()
            ));
        }
        let max_tokens = max_tokens.max(cap);
        let max_rows = ikv_seq_rows(cap, d.ppt());
        let feat_cols = ikv_feat_cols(max_tokens, cap);
        let mut plan = Self {
            d,
            cap,
            max_rows,
            max_tokens,
            feat_cols,
            weights: BTreeMap::new(),
            acts: BTreeMap::new(),
            xbuf_bytes: d.hid * max_rows * 2,
            feat_bytes: text.hid * feat_cols * 2,
        };
        let graphs = [
            build_embed_graph(&d, max_rows),
            build_block_graph(&d, max_rows),
            build_merge_graph(&d, cap),
            build_text_embed_graph(text, feat_cols, tile, &[]),
        ];
        for gb in &graphs {
            for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
                if CHAIN_TENSORS.contains(&name.as_str())
                    || TEXT_TENSORS.contains(&name.as_str())
                    || name == "vis_feat"
                {
                    continue;
                }
                if let Some(kind) = generic_layer_weight(&name) {
                    for l in 0..d.layers {
                        plan.weights.insert(format!("VL{l}_{kind}"), bytes);
                    }
                } else if name.starts_with("VW_") || name.starts_with("VM_") {
                    plan.weights.insert(name, bytes);
                } else {
                    let e = plan.acts.entry(name).or_insert(0);
                    *e = (*e).max(bytes);
                }
            }
        }
        Ok(plan)
    }

    pub fn weights_bytes(&self) -> u64 {
        self.weights.values().map(|&b| align256(b)).sum()
    }

    pub fn act_bytes(&self) -> u64 {
        self.acts.values().map(|&b| align256(b)).sum::<u64>()
            + 2 * align256(self.xbuf_bytes)
            + align256(self.feat_bytes)
    }

    pub fn total(&self) -> u64 {
        self.weights_bytes() + self.act_bytes()
    }
}

// ============================================================================
// Weights
// ============================================================================

fn ckpt<'a>(store: &'a SafetensorsStore, name: &str, shape: &[u64]) -> Result<&'a [u8], String> {
    let full = format!("model.visual.{name}");
    let t = store
        .get(&full)
        .map_err(|e| format!("missing checkpoint tensor '{full}': {e}"))?;
    if t.dtype != "BF16" || t.shape != shape {
        return Err(format!(
            "checkpoint tensor '{full}' is {} {:?}, expected BF16 {shape:?}",
            t.dtype, t.shape
        ));
    }
    store
        .tensor_bytes(&full)
        .map_err(|e| format!("read '{full}': {e}"))?
        .ok_or_else(|| format!("checkpoint tensor '{full}' is not mapped"))
}

fn bf16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// BF16 values times a power of two (exact unless a value leaves the normal range).
fn scale_pow2(v: &[u16], s: f32) -> Vec<u16> {
    v.iter()
        .map(|&x| f32_to_bf16_rne(bf16_to_f32(x) * s))
        .collect()
}

/// Output rows `[out][in]` (or a bias `[out]`, `width` 1) with each interleaved pair rotated:
/// row `2j` = -row `2j + 1`, row `2j + 1` = row `2j` (the rotary code's pair rotation, folded
/// into the weights; negation and moves are exact).
fn rotate_pairs(v: &[u16], width: usize) -> Vec<u16> {
    let rows = v.len() / width;
    let mut out = vec![0u16; v.len()];
    for r in (0..rows).step_by(2) {
        for c in 0..width {
            let (a, b) = (v[r * width + c], v[(r + 1) * width + c]);
            out[r * width + c] = b ^ 0x8000;
            out[(r + 1) * width + c] = a;
        }
    }
    out
}

/// The device bytes of tower weight `name`.
pub fn pack_weight(store: &SafetensorsStore, d: &IkvDims, name: &str) -> Result<Vec<u8>, String> {
    let (h, inter) = (d.hid, d.inter);
    let f32_of = |n: &str, len: u64| -> Result<Vec<u8>, String> {
        let v = bf16s(ckpt(store, n, &[len])?);
        Ok(f32_bytes(
            &v.iter().map(|&x| bf16_to_f32(x)).collect::<Vec<_>>(),
        ))
    };
    match name {
        "VW_patch_W" => {
            return Ok(ckpt(
                store,
                "backbone.embeddings.patch_embedding.weight",
                &[h, d.row_len()],
            )?
            .to_vec())
        }
        "VW_pre_w" => return f32_of("backbone.ln_pre.weight", h),
        "VW_pre_b" => return f32_of("backbone.ln_pre.bias", h),
        "VM_in_W" => return Ok(ckpt(store, "proj_in.weight", &[d.proj, d.merged()])?.to_vec()),
        "VM_out_W" => return Ok(ckpt(store, "proj_out.weight", &[d.out, d.proj])?.to_vec()),
        _ => {}
    }
    let (l, kind) = name
        .strip_prefix("VL")
        .and_then(|r| r.split_once('_'))
        .and_then(|(i, k)| i.parse::<u64>().ok().map(|i| (i, k)))
        .ok_or_else(|| format!("no packer for tower tensor '{name}'"))?;
    let p = format!("backbone.layers.{l}.");
    let raw = |s: &str, shape: &[u64]| ckpt(store, &format!("{p}{s}"), shape);
    let scale = 1.0 / (d.dh as f32).sqrt();
    if scale.log2().fract() != 0.0 {
        return Err(format!(
            "head width {}: the attention scale is not a power of two",
            d.dh
        ));
    }
    Ok(match kind {
        "n1_w" => f32_of(&format!("{p}ln_1.weight"), h)?,
        "n1_b" => f32_of(&format!("{p}ln_1.bias"), h)?,
        "n2_w" => f32_of(&format!("{p}ln_2.weight"), h)?,
        "n2_b" => f32_of(&format!("{p}ln_2.bias"), h)?,
        "q_W" => u16_bytes(&scale_pow2(
            &bf16s(raw("self_attn.q_proj.weight", &[h, h])?),
            scale,
        )),
        "q_b" => u16_bytes(&scale_pow2(
            &bf16s(raw("self_attn.q_proj.bias", &[h])?),
            scale,
        )),
        "qr_W" => u16_bytes(&rotate_pairs(
            &scale_pow2(&bf16s(raw("self_attn.q_proj.weight", &[h, h])?), scale),
            h as usize,
        )),
        "qr_b" => u16_bytes(&rotate_pairs(
            &scale_pow2(&bf16s(raw("self_attn.q_proj.bias", &[h])?), scale),
            1,
        )),
        "k_W" => raw("self_attn.k_proj.weight", &[h, h])?.to_vec(),
        "k_b" => raw("self_attn.k_proj.bias", &[h])?.to_vec(),
        "kr_W" => u16_bytes(&rotate_pairs(
            &bf16s(raw("self_attn.k_proj.weight", &[h, h])?),
            h as usize,
        )),
        "kr_b" => u16_bytes(&rotate_pairs(
            &bf16s(raw("self_attn.k_proj.bias", &[h])?),
            1,
        )),
        "v_W" => raw("self_attn.v_proj.weight", &[h, h])?.to_vec(),
        "v_b" => raw("self_attn.v_proj.bias", &[h])?.to_vec(),
        "o_W" => raw("self_attn.out_proj.weight", &[h, h])?.to_vec(),
        "o_b" => raw("self_attn.out_proj.bias", &[h])?.to_vec(),
        "ls1" => raw("ls_1.gamma", &[h])?.to_vec(),
        "ls2" => raw("ls_2.gamma", &[h])?.to_vec(),
        "fc1_W" => raw("mlp.c_fc.weight", &[inter, h])?.to_vec(),
        "fc1_b" => raw("mlp.c_fc.bias", &[inter])?.to_vec(),
        "fc2_W" => raw("mlp.c_proj.weight", &[h, inter])?.to_vec(),
        "fc2_b" => raw("mlp.c_proj.bias", &[h])?.to_vec(),
        other => return Err(format!("no packer for tower tensor kind '{other}'")),
    })
}

/// The learned position table `[grid^2][hid]` and the class row `class + class position`
/// (BF16, rounded once from the f32 sum).
pub fn host_tables(store: &SafetensorsStore, d: &IkvDims) -> Result<(Vec<u16>, Vec<u16>), String> {
    let table = bf16s(ckpt(
        store,
        "backbone.embeddings.position_embedding",
        &[d.grid * d.grid, d.hid],
    )?);
    let ce = bf16s(ckpt(
        store,
        "backbone.embeddings.class_embedding",
        &[d.hid],
    )?);
    let cp = bf16s(ckpt(
        store,
        "backbone.embeddings.cls_position_embedding",
        &[d.hid],
    )?);
    let cls = ce
        .iter()
        .zip(&cp)
        .map(|(&a, &b)| f32_to_bf16_rne(bf16_to_f32(a) + bf16_to_f32(b)))
        .collect();
    Ok((table, cls))
}

// ============================================================================
// Runtime
// ============================================================================

struct BucketRecipes {
    embed: CompiledRecipe,
    block: CompiledRecipe,
    merge: CompiledRecipe,
    /// One prepared launch per layer (weights and ping-pong buffers fixed).
    layers: Vec<PreparedLaunch>,
}

/// The images of the request being served and where their features are.
struct ActiveRequest {
    images: Vec<PreparedImage>,
    tokens: u64,
}

/// The tower of one engine: plan, device buffers, recipes per bucket, request state.
pub struct IkVision {
    spec: VisionSpec,
    plan: IkvPlan,
    text_hid: u64,
    image_token_id: i32,
    precompile: Vec<u64>,
    dump_dir: Option<std::path::PathBuf>,

    // recipes and pinned memory (dropped before the device by the engine's field order)
    sets: HashMap<u64, BucketRecipes>,
    text_embed: HashMap<u64, CompiledRecipe>,
    stage: Option<PinnedBuf>,

    weights_addr: BTreeMap<String, u64>,
    act_addr: BTreeMap<String, u64>,
    xbuf: [u64; 2],
    feat: u64,
    pos_table: Vec<u16>,
    cls_row: Vec<u16>,

    pending: Option<Vec<PreparedImage>>,
    active: Option<ActiveRequest>,
    last_report: Option<VisionReport>,
    last_tokens: u64,
}

impl IkVision {
    /// Plans the tower when the configuration asks for image input and the checkpoint has a
    /// tower (`None`: text only). Nothing is compiled or allocated here.
    pub fn plan(cfg: &crate::engine::EngineConfig, text: &IkDims) -> Result<Option<Self>, String> {
        if cfg.vision_max_patches == 0 {
            return Ok(None);
        }
        let config = std::fs::read_to_string(cfg.model_dir.join("config.json"))
            .map_err(|e| format!("read config.json: {e}"))?;
        if !is_inkling2_vision(&config) {
            return Ok(None);
        }
        let proc = std::fs::read_to_string(cfg.model_dir.join("processor_config.json")).ok();
        let spec = vision_spec(&config, proc.as_deref())?;
        let d = IkvDims::from_config_json(&config)?;
        let plan = IkvPlan::new(
            d,
            text,
            cfg.vision_max_patches,
            cfg.vision_max_tokens,
            super::IK_MAX_PREFILL_TILE,
        )?;
        let mut pre: Vec<u64> = cfg
            .vision_precompile
            .iter()
            .map(|&p| ikv_bucket(p / d.ppt(), ikv_round(p / d.ppt()), plan.cap))
            .filter(|&b| b > 0)
            .collect();
        pre.push(plan.cap);
        pre.sort_unstable();
        pre.dedup();
        Ok(Some(Self {
            image_token_id: spec.image_token_id,
            spec,
            plan,
            text_hid: text.hid,
            precompile: pre,
            dump_dir: std::env::var_os("GAUDIUM_IKV_DUMP").map(std::path::PathBuf::from),
            sets: HashMap::new(),
            text_embed: HashMap::new(),
            stage: None,
            weights_addr: BTreeMap::new(),
            act_addr: BTreeMap::new(),
            xbuf: [0; 2],
            feat: 0,
            pos_table: Vec::new(),
            cls_row: Vec::new(),
            pending: None,
            active: None,
            last_report: None,
            last_tokens: 0,
        }))
    }

    pub fn spec(&self) -> &VisionSpec {
        &self.spec
    }

    pub fn plan_bytes(&self) -> (u64, u64) {
        (self.plan.weights_bytes(), self.plan.act_bytes())
    }

    /// Compiles the recipes of the planned buckets (the largest image among them), so the shared
    /// workspace covers them. Returns the largest workspace.
    pub fn compile_planned(
        &mut self,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver: &str,
        verbose: bool,
    ) -> Result<u64, String> {
        let mut ws = 0u64;
        for &b in &self.precompile.clone() {
            let (e, k, m) = self.compile_bucket(api, cache, driver, b, verbose)?;
            ws = ws.max(e.ws_size).max(k.ws_size).max(m.ws_size);
            self.sets.insert(
                b,
                BucketRecipes {
                    embed: e,
                    block: k,
                    merge: m,
                    layers: Vec::new(),
                },
            );
        }
        Ok(ws)
    }

    fn compile_bucket(
        &self,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver: &str,
        b: u64,
        verbose: bool,
    ) -> Result<(CompiledRecipe, CompiledRecipe, CompiledRecipe), String> {
        let d = &self.plan.d;
        let n = ikv_seq_rows(b, d.ppt());
        let mut out = Vec::with_capacity(3);
        for gb in [
            build_embed_graph(d, n),
            build_block_graph(d, n),
            build_merge_graph(d, b),
        ] {
            let t0 = Instant::now();
            let r = gb
                .compile(Arc::clone(api), cache, driver)
                .map_err(|e| format!("compile {}: {e}", gb.recipe_name()))?;
            if verbose {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                    r.name,
                    r.from_cache,
                    r.ws_size as f64 / (1024.0 * 1024.0),
                    t0.elapsed().as_secs_f64()
                );
            }
            out.push(r);
        }
        let m = out.pop().expect("three recipes");
        let k = out.pop().expect("three recipes");
        let e = out.pop().expect("three recipes");
        Ok((e, k, m))
    }

    /// Allocates the tower's buffers (before the engine seals its memory).
    pub fn allocate(&mut self, dev: &mut HpuDevice) -> Result<(), String> {
        let mut warena = dev
            .alloc_arena(
                self.plan.weights_bytes().max(256),
                HbmCat::CatWeights,
                "ikv_weights_arena",
            )
            .map_err(|e| format!("alloc vision weights arena: {e}"))?;
        let mut aarena = dev
            .alloc_arena(
                self.plan.act_bytes().max(256),
                HbmCat::CatScratch,
                "ikv_act_arena",
            )
            .map_err(|e| format!("alloc vision act arena: {e}"))?;
        let sub = |a: &mut HbmArena, bytes: u64| -> Result<u64, String> {
            a.suballoc(bytes.max(256), 256)
                .map(|(addr, _)| addr)
                .map_err(|e| format!("vision arena suballoc: {e}"))
        };
        for (name, &bytes) in &self.plan.weights {
            self.weights_addr
                .insert(name.clone(), sub(&mut warena, bytes)?);
        }
        for (name, &bytes) in &self.plan.acts {
            self.act_addr.insert(name.clone(), sub(&mut aarena, bytes)?);
        }
        self.xbuf = [
            sub(&mut aarena, self.plan.xbuf_bytes)?,
            sub(&mut aarena, self.plan.xbuf_bytes)?,
        ];
        self.feat = sub(&mut aarena, self.plan.feat_bytes)?;
        Ok(())
    }

    /// Binds the recipes to the shared workspace, uploads the weights, zeroes the feature buffer
    /// (its column 0 is what text rows read) and keeps the host tables.
    pub fn upload(
        &mut self,
        dev: &HpuDevice,
        store: &SafetensorsStore,
        stage: &mut PinnedBuf,
    ) -> Result<u64, String> {
        for set in self.sets.values_mut() {
            for r in [&mut set.embed, &mut set.block, &mut set.merge] {
                r.bind_shared_workspace(dev)
                    .map_err(|e| format!("bind {} workspace: {e}", r.name))?;
            }
        }
        let mut bytes = 0u64;
        for (name, &addr) in &self.weights_addr {
            let data = pack_weight(store, &self.plan.d, name)?;
            if data.len() as u64 != self.plan.weights[name] {
                return Err(format!(
                    "tower weight '{name}' packed size {} != graph tensor size {}",
                    data.len(),
                    self.plan.weights[name]
                ));
            }
            dev.upload_h2d_chunked(&data, addr, stage)
                .map_err(|e| format!("upload tower weight '{name}': {e}"))?;
            bytes += data.len() as u64;
        }
        let (table, cls) = host_tables(store, &self.plan.d)?;
        self.pos_table = table;
        self.cls_row = cls;
        dev.zero_hbm_compute(self.feat, align_up(self.plan.feat_bytes, 4))
            .map_err(|e| format!("zero the feature buffer: {e}"))?;
        // SAFETY: the compute stream is owned by `dev` and alive for its lifetime.
        unsafe { dev.sync_stream(dev.compute_stream()) }
            .map_err(|e| format!("sync zero fill: {e}"))?;
        self.stage = Some(
            dev.alloc_pinned(STAGE_BYTES)
                .map_err(|e| format!("alloc vision stage: {e}"))?,
        );
        Ok(bytes)
    }

    /// Takes the images of the next request after checking each against the plan.
    pub fn set_request_images(&mut self, images: Vec<PreparedImage>) -> Result<(), String> {
        let d = self.plan.d;
        let mut rows = EmbRows::new(self.plan.max_tokens);
        for (k, im) in images.iter().enumerate() {
            let m = d.merge as usize;
            if im.row_len as u64 != d.row_len()
                || im.grid_h % m != 0
                || im.grid_w % m != 0
                || im.pixels.len() != im.grid_h * im.grid_w * im.row_len
            {
                return Err(format!("image {}: not a tower input of this model", k + 1));
            }
            let (tr, tc) = ((im.grid_h / m) as u64, (im.grid_w / m) as u64);
            if !ikv_tokens_fit(tr, tc, self.plan.cap) {
                return Err(format!(
                    "image {}: {}x{} pixels make {} image tokens ({tr} x {tc} tiles of {} pixels); this engine's vision tower was planned for images of at most {} tokens (--vision-max-patches {})",
                    k + 1,
                    im.src_width,
                    im.src_height,
                    tr * tc,
                    d.tile(),
                    self.plan.cap,
                    self.plan.cap * d.ppt()
                ));
            }
            rows.take(tr * tc).map_err(|(need, cap)| {
                format!("the request's images need {need} image tokens; this engine takes {cap} per request")
            })?;
        }
        self.pending = Some(images);
        Ok(())
    }

    /// Matches the pending images with the prompt's placeholder rows and makes them the current
    /// request (no pending images: a text request).
    pub fn begin_request(&mut self, prompt_ids: &[i32], step_prefill: bool) -> Result<(), String> {
        self.active = None;
        let Some(images) = self.pending.take() else {
            self.last_report = None;
            return Ok(());
        };
        if images.is_empty() {
            self.last_report = None;
            return Ok(());
        }
        if step_prefill {
            return Err(
                "GAUDIUM_IK_STEP_PREFILL prefills through the decode step, which takes no image"
                    .into(),
            );
        }
        let m = self.plan.d.merge as usize;
        let tokens: u64 = images
            .iter()
            .map(|i| ((i.grid_h / m) * (i.grid_w / m)) as u64)
            .sum();
        let placeholders = prompt_ids
            .iter()
            .filter(|&&t| t == self.image_token_id)
            .count() as u64;
        if !ikv_rows_match(placeholders, tokens) {
            return Err(format!(
                "the prompt has {placeholders} image placeholder rows (id {}); its {} image(s) have {tokens} tokens",
                self.image_token_id,
                images.len()
            ));
        }
        self.active = Some(ActiveRequest { images, tokens });
        Ok(())
    }

    /// Image tokens of the last request with images.
    pub fn last_tokens(&self) -> u64 {
        self.last_tokens
    }

    pub fn last_report(&self) -> Option<VisionReport> {
        self.last_report.clone()
    }

    /// Feature rows of the last image request (`tokens x out` BF16).
    pub fn read_features(&self, dev: &HpuDevice) -> Result<Vec<u8>, String> {
        if self.last_tokens == 0 {
            return Err("no image request has run on this engine".into());
        }
        let row = self.text_hid * 2;
        let mut out = vec![0u8; (self.last_tokens * row) as usize];
        dev.copy_d2h_sync(self.feat + row, &mut out)
            .map_err(|e| format!("read image features: {e}"))?;
        Ok(out)
    }

    /// The per-image host inputs of a sequence of `n` rows: patch rows, position rows (class row
    /// first), rotary tables and the key bias.
    fn inputs(&self, im: &PreparedImage, n: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let d = &self.plan.d;
        let (hid, rl, m) = (d.hid as usize, d.row_len() as usize, d.merge as usize);
        let n = n as usize;
        let np = im.grid_h * im.grid_w;
        let tokens = (np / (m * m)) as u64;
        // patch rows behind a zero class column, zero padding columns
        let mut pix = vec![0u16; rl * n];
        pix[rl..rl * (1 + np)].copy_from_slice(&im.pixels);
        // class row, the resampled table, zero padding
        let mut pos = vec![0u16; hid * n];
        pos[..hid].copy_from_slice(&self.cls_row);
        let taps = interp_taps(im.grid_h, im.grid_w, d.grid as usize, m);
        let rows = pos_rows(&self.pos_table, hid, &taps);
        pos[hid..hid * (1 + np)].copy_from_slice(&rows);
        // rotary tables, the same for every head; padding rows at (0, 0)
        let inv = inv_freq(d.dh, d.theta);
        let positions = rope_positions(im.grid_h, im.grid_w, m);
        let dh = d.dh as usize;
        let mut cs = vec![0f32; hid * n];
        let mut sn = vec![0f32; hid * n];
        for r in 0..n {
            let p = positions.get(r).copied().unwrap_or((0, 0));
            let (c, s) = rope_row(p, &inv);
            for h in 0..d.nh as usize {
                cs[r * hid + h * dh..r * hid + (h + 1) * dh].copy_from_slice(&c);
                sn[r * hid + h * dh..r * hid + (h + 1) * dh].copy_from_slice(&s);
            }
        }
        let kb: Vec<f32> = (0..n as u64)
            .map(|j| {
                if ikv_key_live(j, tokens, d.ppt()) {
                    0.0
                } else {
                    DEAD_KEY_BIAS
                }
            })
            .collect();
        (
            u16_bytes(&pix),
            u16_bytes(&pos),
            f32_bytes(&cs),
            f32_bytes(&sn),
            f32_bytes(&kb),
        )
    }
}

impl Inkling2Engine {
    /// The tower's image settings when the engine was loaded with image input.
    pub fn vision_spec(&self) -> Option<&VisionSpec> {
        self.vision.as_ref().map(IkVision::spec)
    }

    /// Hands the images of the next `generate_cancellable` call to the engine, in prompt order.
    pub fn set_request_images(&mut self, images: Vec<PreparedImage>) -> Result<(), String> {
        match self.vision.as_mut() {
            Some(v) => v.set_request_images(images),
            None => Err("this engine was loaded without its vision tower (text only)".into()),
        }
    }

    /// Vision timings of the last request (`None` if it had no images).
    pub fn last_vision_report(&self) -> Option<VisionReport> {
        self.vision.as_ref().and_then(IkVision::last_report)
    }

    /// The image features of the last image request read back from HBM (`tokens x 2304` BF16).
    pub fn read_vision_embeddings(&self) -> Result<Vec<u8>, String> {
        match self.vision.as_ref() {
            Some(v) => v.read_features(&self.dev),
            None => Err("this engine was loaded without its vision tower (text only)".into()),
        }
    }

    /// The placeholder token and, for each image of the current request in prompt order, a
    /// fingerprint of its tower input and its number of rows (prefix reuse compares image rows
    /// by these; a text request, or an engine without a tower: no images, and a token no
    /// prompt holds).
    pub(super) fn ikv_request_keys(&self) -> (i32, Vec<(u64, u64)>) {
        use std::hash::{Hash, Hasher};
        let Some(vis) = self.vision.as_ref() else {
            return (-1, Vec::new());
        };
        let m = vis.plan.d.merge as usize;
        let images = vis.active.as_ref().map_or_else(Vec::new, |req| {
            req.images
                .iter()
                .map(|im| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    (im.grid_h, im.grid_w).hash(&mut h);
                    im.pixels.hash(&mut h);
                    (h.finish(), ((im.grid_h / m) * (im.grid_w / m)) as u64)
                })
                .collect()
        });
        (vis.image_token_id, images)
    }

    /// Runs the tower over the current request's images, writing their features into the
    /// feature buffer in prompt order (prefill time: it counts in the time to first token).
    pub(super) fn ikv_run_tower(&mut self) -> Result<(), String> {
        let Some(vis) = self.vision.as_mut() else {
            return Ok(());
        };
        let Some(req) = vis.active.take() else {
            return Ok(());
        };
        let res = self.ikv_encode(&req);
        if let Some(vis) = self.vision.as_mut() {
            vis.active = Some(req);
        }
        res
    }

    fn ikv_ensure_bucket(&mut self, b: u64) -> Result<f64, String> {
        let vis = self.vision.as_mut().expect("vision tower");
        if !vis.sets.contains_key(&b) {
            let t0 = Instant::now();
            let (mut e, mut k, mut m) = vis.compile_bucket(
                self.dev.api(),
                &self.cache,
                &self.driver_version,
                b,
                self.verbose,
            )?;
            for r in [&mut e, &mut k, &mut m] {
                r.bind_shared_workspace(&self.dev)
                    .map_err(|err| format!("bind {} workspace: {err}", r.name))?;
            }
            vis.sets.insert(
                b,
                BucketRecipes {
                    embed: e,
                    block: k,
                    merge: m,
                    layers: Vec::new(),
                },
            );
            return Ok(t0.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(0.0)
    }

    fn ikv_encode(&mut self, req: &ActiveRequest) -> Result<(), String> {
        let mut report = VisionReport::default();
        report.images = req.images.len();
        report.image_tokens = req.tokens;
        let mut off = 1u64; // column 0 stays zero
        for (k, im) in req.images.iter().enumerate() {
            let vis = self.vision.as_ref().expect("vision tower");
            let d = vis.plan.d;
            let m = d.merge as usize;
            let tokens = ((im.grid_h / m) * (im.grid_w / m)) as u64;
            let b = ikv_bucket(tokens, ikv_round(tokens), vis.plan.cap);
            let n = ikv_seq_rows(b, d.ppt());
            if !ikv_feat_write_ok(off, b, vis.plan.feat_cols) {
                return Err(format!(
                    "image {}: its features (columns {off}..{}) do not fit the feature buffer of {} columns",
                    k + 1,
                    off + b,
                    vis.plan.feat_cols
                ));
            }
            report.compile_ms += self.ikv_ensure_bucket(b)?;

            // host inputs and their upload
            let t_prep = Instant::now();
            let vis = self.vision.as_mut().expect("vision tower");
            let (pix, pos, cs, sn, kb) = vis.inputs(im, n);
            let mut stage = vis.stage.take().ok_or("vision stage is not allocated")?;
            let ups = [
                (&pix, "v_pix"),
                (&pos, "v_pos"),
                (&cs, "v_cos"),
                (&sn, "v_sin"),
                (&kb, "v_kbias"),
            ];
            let mut up_res = Ok(());
            for (data, name) in ups {
                let addr = vis.act_addr[name];
                if let Err(e) = self.dev.upload_h2d_chunked(data, addr, &mut stage) {
                    up_res = Err(format!("upload {name}: {e}"));
                    break;
                }
            }
            vis.stage = Some(stage);
            up_res?;
            report.prep_ms += t_prep.elapsed().as_secs_f64() * 1000.0;

            // embedding, the layers, the merger
            let t_enc = Instant::now();
            let vis = self.vision.as_mut().expect("vision tower");
            let set = vis.sets.get_mut(&b).expect("bucket compiled above");
            let mut pool = vis.act_addr.clone();
            for (name, &addr) in &vis.weights_addr {
                pool.insert(name.clone(), addr);
            }
            pool.insert("v_x0".into(), vis.xbuf[0]);
            let bind = select_exact_bindings(&set.embed, &pool)?;
            set.embed
                .launch_checked(&self.dev, &bind)
                .map_err(|e| format!("launch tower embedding: {e}"))?;
            if set.layers.is_empty() {
                for l in 0..d.layers {
                    let mut p = vis.act_addr.clone();
                    for kind in LAYER_WEIGHTS {
                        p.insert(
                            format!("vb_{kind}"),
                            vis.weights_addr[&format!("VL{l}_{kind}")],
                        );
                    }
                    p.insert("vb_x_in".into(), vis.xbuf[(l % 2) as usize]);
                    p.insert("vb_x_out".into(), vis.xbuf[((l + 1) % 2) as usize]);
                    let exact = select_exact_bindings(&set.block, &p)?;
                    set.layers.push(
                        set.block
                            .prepare_launch(&exact)
                            .map_err(|e| format!("prepare tower layer {l}: {e}"))?,
                    );
                }
            }
            let dump = vis.dump_dir.clone();
            let dumps: &[u64] = &[0, 11, 22];
            if let Some(dir) = &dump {
                self.ikv_sync("tower embedding")?;
                let vis = self.vision.as_ref().expect("vision tower");
                let live = 1 + tokens * d.ppt();
                ikv_dump(
                    &self.dev,
                    dir,
                    &format!("img{k}_emb"),
                    vis.act_addr["v_emb"],
                    live,
                    d.hid,
                )?;
                ikv_dump(
                    &self.dev,
                    dir,
                    &format!("img{k}_pre"),
                    vis.xbuf[0],
                    live,
                    d.hid,
                )?;
            }
            for l in 0..d.layers {
                let vis = self.vision.as_ref().expect("vision tower");
                let set = &vis.sets[&b];
                set.block
                    .launch_prepared(&self.dev, &set.layers[l as usize])
                    .map_err(|e| format!("launch tower layer {l}: {e}"))?;
                if let (Some(dir), true) = (&dump, dumps.contains(&l)) {
                    let out = vis.xbuf[((l + 1) % 2) as usize];
                    self.ikv_sync("tower layer")?;
                    ikv_dump(
                        &self.dev,
                        dir,
                        &format!("img{k}_l{l:02}"),
                        out,
                        1 + tokens * d.ppt(),
                        d.hid,
                    )?;
                }
            }
            let vis = self.vision.as_ref().expect("vision tower");
            let set = &vis.sets[&b];
            let last = vis.xbuf[(d.layers % 2) as usize];
            let mut pool = vis.act_addr.clone();
            pool.insert("VM_in_W".into(), vis.weights_addr["VM_in_W"]);
            pool.insert("VM_out_W".into(), vis.weights_addr["VM_out_W"]);
            pool.insert("vm_x_in".into(), last + d.hid * 2);
            pool.insert("v_feat".into(), vis.feat + off * vis.text_hid * 2);
            let bind = select_exact_bindings(&set.merge, &pool)?;
            set.merge
                .launch_checked(&self.dev, &bind)
                .map_err(|e| format!("launch tower merger: {e}"))?;
            self.ikv_sync("vision tower")?;
            report.encode_ms += t_enc.elapsed().as_secs_f64() * 1000.0;
            report.patches += tokens * d.ppt();
            if let Some(dir) = &dump {
                let vis = self.vision.as_ref().expect("vision tower");
                ikv_dump(
                    &self.dev,
                    dir,
                    &format!("img{k}_feat"),
                    vis.feat + off * vis.text_hid * 2,
                    tokens,
                    vis.text_hid,
                )?;
            }
            if self.verbose {
                eprintln!(
                    "[gaudium] vision image {}: {}x{} px -> {tokens} tokens, bucket {b} ({n} rows)",
                    k + 1,
                    im.src_width,
                    im.src_height
                );
            }
            off += tokens;
        }
        let vis = self.vision.as_mut().expect("vision tower");
        vis.last_tokens = req.tokens;
        vis.last_report = Some(report);
        // the debug dump covers the first image request of the process only (its syncs would
        // distort the timings of the ones after it)
        vis.dump_dir = None;
        Ok(())
    }

    fn ikv_sync(&self, what: &str) -> Result<(), String> {
        // SAFETY: the compute stream is owned by `self.dev` and alive for its lifetime.
        unsafe { self.dev.sync_stream(self.dev.compute_stream()) }
            .map_err(|e| format!("sync {what}: {e}"))
    }

    /// The embedding stage of a prefill tile `[pos, pos + t)` when it holds image rows: the
    /// hand-over recipe instead of the plain one. `Ok(false)`: the tile has no image row (the
    /// caller launches the plain recipe).
    pub(super) fn ikv_embed_tile(
        &mut self,
        prompt_ids: &[i32],
        pos: usize,
        t: u64,
    ) -> Result<bool, String> {
        let Some(vis) = self.vision.as_ref() else {
            return Ok(false);
        };
        if vis.active.is_none() {
            return Ok(false);
        }
        let img = vis.image_token_id;
        // the tile may run more rows than the prompt has left (padding rows past the cursor,
        // id 0): those are text rows
        let tv = (prompt_ids.len() - pos.min(prompt_ids.len())).min(t as usize);
        let tile = &prompt_ids[pos..pos + tv];
        if !tile.contains(&img) {
            return Ok(false);
        }
        // the rank of each placeholder row in the whole prompt
        let mut rank = prompt_ids[..pos].iter().filter(|&&x| x == img).count() as u64;
        let mut fidx = Vec::with_capacity(t as usize * 4);
        let mut keep = Vec::with_capacity(t as usize * 2);
        for &id in tile.iter().chain(std::iter::repeat_n(&0, t as usize - tv)) {
            let is_img = id == img;
            let col = ikv_feat_row(is_img, rank);
            if is_img {
                rank += 1;
            }
            fidx.extend((col as i32).to_le_bytes());
            keep.extend(f32_to_bf16_rne(if is_img { 0.0 } else { 1.0 }).to_le_bytes());
        }
        if !self.vision.as_ref().unwrap().text_embed.contains_key(&t) {
            let vis = self.vision.as_ref().unwrap();
            let gb = build_text_embed_graph(&self.dims, vis.plan.feat_cols, t, &self.elf_hashes);
            let r = self.compile(&gb, &format!("image embed T={t}"))?;
            self.vision.as_mut().unwrap().text_embed.insert(t, r);
        }
        let vis = self.vision.as_mut().unwrap();
        let mut stage = vis.stage.take().ok_or("vision stage is not allocated")?;
        let res = self
            .dev
            .upload_h2d_chunked(&fidx, vis.act_addr["vis_fidx"], &mut stage)
            .and_then(|()| {
                self.dev
                    .upload_h2d_chunked(&keep, vis.act_addr["vis_keep"], &mut stage)
            });
        vis.stage = Some(stage);
        res.map_err(|e| format!("upload image row selection: {e}"))?;
        let vis = self.vision.as_ref().unwrap();
        let mut pool = self.pf_scratch.clone();
        for (k, &v) in &vis.act_addr {
            pool.insert(k.clone(), v);
        }
        for name in ["embed", "w_embed"] {
            pool.insert(name.to_string(), self.weights_addr[name]);
        }
        pool.insert("ids".into(), self.pf_ids);
        pool.insert("x0".into(), self.src_buf(0));
        pool.insert("vis_feat".into(), vis.feat);
        let r = &vis.text_embed[&t];
        let bind = select_exact_bindings(r, &pool)?;
        r.launch_checked(&self.dev, &bind)
            .map_err(|e| format!("launch image embed T={t}: {e}"))?;
        Ok(true)
    }
}

/// Debug dump of `rows` BF16 rows of width `width` at `addr` to `<dir>/<name>.bf16`.
fn ikv_dump(
    dev: &HpuDevice,
    dir: &std::path::Path,
    name: &str,
    addr: u64,
    rows: u64,
    width: u64,
) -> Result<(), String> {
    let mut buf = vec![0u8; (rows * width * 2) as usize];
    dev.copy_d2h_sync(addr, &mut buf)
        .map_err(|e| format!("read {name}: {e}"))?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = dir.join(format!("{name}.bf16"));
    std::fs::write(&path, &buf).map_err(|e| format!("write {}: {e}", path.display()))
}

#[cfg(test)]
#[path = "inkling2_vision_tests.rs"]
mod tests;
