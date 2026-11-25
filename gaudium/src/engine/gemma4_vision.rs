//! Image input of the Gemma 4 engine: the checkpoint's image path on the card, its host
//! preprocessing, and the hand-over of the soft tokens to the text prefill.
//!
//! The four sizes take an image one of two ways, read from `vision_config.model_type`:
//!
//! - **A tower** (`gemma4_vision`: E2B, E4B, 31B). 16 x 16 patches through a bias-free linear,
//!   plus a position row that is the sum of a column row and a row row of a learned table; then
//!   pre-norm encoder layers with four RMS norms each, per-head q / k norms with a weight and a
//!   weightless v norm, a 2-D rotary code (the first half of a head turns with the column, the
//!   second with the row), attention scale 1, every patch seeing every patch, a gated tanh-GELU
//!   MLP. The small sizes clamp the input and the output of every encoder projection to four
//!   stored scalars; the 31B standardises the pooled rows with two stored vectors. After the
//!   encoder: the mean of every `pool x pool` block of patches times `sqrt(hidden)`, a weightless
//!   RMS norm and a projection to the decoder's width.
//! - **No tower** (`gemma4_unified_vision`: the 12B). A soft token is a function of its own
//!   `pool x pool` patches (48 x 48 pixels): LayerNorm, a linear with bias, LayerNorm, plus the
//!   sum of its column and row position rows, LayerNorm, a weightless RMS norm, a projection.
//!
//! Both read the same processor: the image resized (antialiased bicubic on 8-bit pixels, aspect
//! kept) to the largest size of whole 48-pixel squares within the soft-token budget, pixels
//! divided by 255, no normalisation. The soft tokens replace the embeddings of the image's
//! placeholder rows (not scaled like a token embedding). The decoder's positions stay
//! `0, 1, 2, ...`; the 12B and the 31B (`use_bidirectional_attention: "vision"`) let the rows of
//! one image attend each other both ways in their sliding layers (`prefill units`, below).
//!
//! Decisions (the resized size and its token bound, the units and the extension of a row's window)
//! come from `verified/gemma4_vision.bend`; the feature buffer's columns and the row selection of
//! the embedding stage are the family-independent ones of `verified/inkling2_vision.bend`.

use crate::fp8::f32_to_bf16_rne;
use crate::json::{parse_json, JsonValue};
use crate::verified::{
    g4v_grid_ok, g4v_pool_count, g4v_pool_x, g4v_pool_y, g4v_rope_freq, g4v_rope_row_axis, g4v_rot_quarter,
    g4v_row_values, g4v_size_ok, g4v_soft_budget, g4v_table_holds, g4v_target_h, g4v_target_w, g4v_token_patches,
    g4v_token_pixels, g4v_token_side, g4v_tokens, g4v_tower_ok, g4v_unified_ok, g4v_width_ok,
};
use crate::vision::preprocess::{resize_rgb, PreparedImage, RgbImage};
use crate::vision::{ImageProcessor, VisionSpec};

#[path = "gemma4_vision_graphs.rs"]
pub mod graphs;
#[path = "gemma4_vision_run.rs"]
pub mod run;
#[cfg(test)]
#[path = "gemma4_vision_tests.rs"]
mod tests;

/// The family's image text: the template writes the pad, the processor puts the begin and end
/// tokens around it (read from `tokenizer_config.json` when there is one).
const IMAGE_TOKEN: &str = "<|image|>";
const BOI_TOKEN: &str = "<|image>";
const EOI_TOKEN: &str = "<image|>";

// ============================================================================
// Configuration
// ============================================================================

fn vision_config(root: &JsonValue) -> Option<&JsonValue> {
    root.get("vision_config")
        .filter(|v| v.as_object().is_some())
}

fn vision_type(root: &JsonValue) -> Option<&str> {
    vision_config(root)?.get("model_type")?.as_str()
}

/// Whether a `config.json` describes a Gemma 4 image path (a tower or the unified embedder).
pub fn is_gemma4_vision(config_json: &str) -> bool {
    parse_json(config_json).ok().is_some_and(|root| {
        matches!(
            vision_type(&root),
            Some("gemma4_vision" | "gemma4_unified_vision")
        )
    })
}

/// Which of the two image paths a checkpoint has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G4vKind {
    /// Encoder layers over the patches, then pooling and the projection.
    Tower,
    /// One soft token per merged patch, no attention between them.
    Unified,
}

/// Geometry of the image path, from `vision_config` and the decoder's width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct G4vDims {
    pub kind: G4vKind,
    /// Tower width, or the unified embedder's width (`mm_embed_dim`).
    pub hid: u64,
    pub layers: u64,
    pub nh: u64,
    pub dh: u64,
    pub inter: u64,
    pub patch: u64,
    pub pool: u64,
    /// Rows of the position table per axis.
    pub npos: u64,
    /// Decoder width the projection writes.
    pub out: u64,
    /// Clamped encoder projections (`use_clipped_linears`).
    pub clipped: bool,
    /// Pooled rows standardised with `std_bias` / `std_scale`.
    pub standardize: bool,
    pub eps: f32,
    pub theta: f64,
    /// The input sizes `with_sizes` derives (side of a soft token's square, its patches, the
    /// values of one input row).
    side: u64,
    ppt: u64,
    row_len: u64,
}

impl G4vDims {
    pub fn from_config_json(config_json: &str) -> Result<Self, String> {
        let root = parse_json(config_json).map_err(|e| format!("config.json: {e}"))?;
        let vc = vision_config(&root).ok_or("config.json has no vision_config")?;
        let text = root.get("text_config").unwrap_or(&root);
        let out = text
            .get("hidden_size")
            .and_then(JsonValue::as_u64)
            .ok_or("text_config.hidden_size is missing")?;
        let u = |key: &str| vc.get(key).and_then(JsonValue::as_u64);
        let need = |key: &str| u(key).ok_or_else(|| format!("vision_config.{key} is missing"));
        let eps = vc
            .get("rms_norm_eps")
            .and_then(JsonValue::as_f64)
            .unwrap_or(1e-6) as f32;
        let d = match vision_type(&root) {
            Some("gemma4_vision") => {
                let hid = need("hidden_size")?;
                let nh = need("num_attention_heads")?;
                let dh = u("head_dim").unwrap_or(if nh > 0 { hid / nh } else { 0 });
                if u("num_key_value_heads").is_some_and(|k| k != nh) {
                    return Err("vision_config: grouped key / value heads in the tower are not supported".into());
                }
                if vc.get("attention_bias").and_then(JsonValue::as_bool) == Some(true) {
                    return Err("vision_config.attention_bias: the tower's projections have no bias here".into());
                }
                if let Some(act) = vc.get("hidden_activation").and_then(JsonValue::as_str) {
                    if act != "gelu_pytorch_tanh" {
                        return Err(format!("vision_config.hidden_activation {act}: the tower runs tanh-GELU"));
                    }
                }
                let theta = vc
                    .get("rope_parameters")
                    .and_then(|r| r.get("rope_theta"))
                    .and_then(JsonValue::as_f64)
                    .unwrap_or(100.0);
                Self {
                    kind: G4vKind::Tower,
                    hid,
                    layers: need("num_hidden_layers")?,
                    nh,
                    dh,
                    inter: need("intermediate_size")?,
                    patch: need("patch_size")?,
                    pool: need("pooling_kernel_size")?,
                    npos: need("position_embedding_size")?,
                    out,
                    clipped: vc
                        .get("use_clipped_linears")
                        .and_then(JsonValue::as_bool)
                        .unwrap_or(false),
                    standardize: vc
                        .get("standardize")
                        .and_then(JsonValue::as_bool)
                        .unwrap_or(false),
                    eps,
                    theta,
                    side: 0,
                    ppt: 0,
                    row_len: 0,
                }
            }
            Some("gemma4_unified_vision") => {
                let patch = need("patch_size")?;
                let pool = need("pooling_kernel_size")?;
                let hid = need("mm_embed_dim")?;
                // the merged patch and the projection agree with the graph (the verified guard;
                // `None`: the sizes overflow)
                let (mp, proj) = (u("model_patch_size"), u("output_proj_dims"));
                let agree = g4v_unified_ok(
                    mp.is_some(),
                    mp.unwrap_or(0),
                    patch,
                    pool,
                    proj.is_some(),
                    proj.unwrap_or(0),
                    hid,
                );
                if agree != Some(true) {
                    return Err(format!(
                        "vision_config: model_patch_size {mp:?} must be patch_size x pooling_kernel_size ({patch} x {pool}), output_proj_dims {proj:?} must be mm_embed_dim ({hid})"
                    ));
                }
                Self {
                    kind: G4vKind::Unified,
                    hid,
                    layers: 0,
                    nh: 0,
                    dh: 0,
                    inter: 0,
                    patch,
                    pool,
                    npos: need("mm_posemb_size")?,
                    out,
                    clipped: false,
                    standardize: false,
                    eps,
                    theta: 0.0,
                    side: 0,
                    ppt: 0,
                    row_len: 0,
                }
            }
            other => return Err(format!("vision_config.model_type {other:?} is not a Gemma 4 image path")),
        };
        let d = d.with_sizes()?;
        d.validate()?;
        Ok(d)
    }

    /// The derived input sizes, from the verified total functions: a config whose sizes overflow
    /// 64 bits is refused here, before anything multiplies them.
    fn with_sizes(mut self) -> Result<Self, String> {
        let merged = self.kind == G4vKind::Unified;
        let sizes = (
            g4v_token_side(self.patch, self.pool),
            g4v_token_patches(self.pool),
            g4v_row_values(merged, self.patch, self.pool),
        );
        let (Some(side), Some(ppt), Some(row_len)) = sizes else {
            return Err(format!(
                "vision_config: patch_size {} and pooling_kernel_size {} give input sizes past 64 bits",
                self.patch, self.pool
            ));
        };
        (self.side, self.ppt, self.row_len) = (side, ppt, row_len);
        Ok(self)
    }

    /// Shape rules the graphs and the host layout rely on (the verified guards).
    pub fn validate(&self) -> Result<(), String> {
        if !g4v_grid_ok(self.patch, self.pool, self.npos) {
            return Err("vision_config: patch, pooling and position table sizes must be positive".into());
        }
        if !g4v_width_ok(self.hid) || !g4v_width_ok(self.out) {
            return Err(format!(
                "vision widths {} / {} must be whole blocks of 128 channels (the family's norm kernels)",
                self.hid, self.out
            ));
        }
        if self.kind == G4vKind::Tower {
            if !g4v_tower_ok(self.layers, self.inter) {
                return Err("vision_config: a tower needs layers and an MLP".into());
            }
            crate::core::gemma4::validate_gemma4_vision_heads(self.nh, self.dh, self.hid)
                .map_err(|e| format!("vision_config: {e:?}"))?;
        } else if !g4v_width_ok(self.row_len()) {
            return Err(format!(
                "a merged patch of {} values: the LayerNorm kernel needs whole blocks of 128",
                self.row_len()
            ));
        }
        Ok(())
    }

    /// Pixel side of one soft token's square (`g4v_token_side`).
    pub fn side(&self) -> u64 {
        self.side
    }

    /// Patches that make one soft token (`g4v_token_patches`).
    pub fn ppt(&self) -> u64 {
        self.ppt
    }

    /// Values of one input row: a patch (tower) or a merged patch (unified), channel last
    /// (`g4v_row_values`).
    pub fn row_len(&self) -> u64 {
        self.row_len
    }
}

/// Values of one prepared input row of a checkpoint's processor. For Gemma 4 the verified
/// `g4v_row_values`, which never overflows for a spec `VisionSpec::validate` took (0 marks one it
/// did not take).
pub fn row_len(vs: &VisionSpec) -> usize {
    match vs.processor {
        ImageProcessor::Gemma4 { merged } => {
            g4v_row_values(merged, vs.patch, vs.merge).map_or(0, |v| v as usize)
        }
        _ => vs.patch_row() as usize,
    }
}

fn token_text(tok: Option<&JsonValue>, key: &str, default: &str) -> String {
    tok.and_then(|t| t.get(key))
        .and_then(JsonValue::as_str)
        .unwrap_or(default)
        .to_string()
}

/// The vision settings of a Gemma 4 checkpoint: `config.json`, the image part of
/// `processor_config.json` (the budget `max_soft_tokens`, sizes checked against the model), and
/// the image token texts of `tokenizer_config.json`.
pub fn vision_spec(
    config_json: &str,
    processor_json: Option<&str>,
    tokenizer_json: Option<&str>,
) -> Result<VisionSpec, String> {
    let d = G4vDims::from_config_json(config_json)?;
    let root = parse_json(config_json).map_err(|e| format!("config.json: {e}"))?;
    let id = |key: &str| -> Result<i32, String> {
        root.get(key)
            .and_then(JsonValue::as_i64)
            .map(|v| v as i32)
            .ok_or_else(|| format!("config.json: {key} is missing"))
    };
    let mut max_soft = root
        .get("vision_soft_tokens_per_image")
        .and_then(JsonValue::as_u64)
        .or_else(|| {
            vision_config(&root).and_then(|vc| {
                vc.get("default_output_length")
                    .or_else(|| vc.get("num_soft_tokens"))
                    .and_then(JsonValue::as_u64)
            })
        })
        .unwrap_or(280);
    let mut rescale = 1.0 / 255.0;
    if let Some(text) = processor_json {
        let p = parse_json(text).map_err(|e| format!("processor_config.json: {e}"))?;
        let ip = p.get("image_processor").unwrap_or(&p);
        if let Some(v) = ip.get("max_soft_tokens").and_then(JsonValue::as_u64) {
            max_soft = v;
        }
        for (key, model) in [("patch_size", d.patch), ("pooling_kernel_size", d.pool)] {
            if let Some(v) = ip.get(key).and_then(JsonValue::as_u64) {
                if v != model {
                    return Err(format!(
                        "processor_config.json {key}={v} disagrees with vision_config ({model})"
                    ));
                }
            }
        }
        if ip.get("do_normalize").and_then(JsonValue::as_bool) == Some(true) {
            return Err("processor_config.json: a normalising Gemma 4 image processor is not supported".into());
        }
        if ip.get("do_rescale").and_then(JsonValue::as_bool) == Some(false) {
            rescale = 1.0;
        } else if let Some(v) = ip.get("rescale_factor").and_then(JsonValue::as_f64) {
            rescale = v;
        }
        if ip.get("resample").and_then(JsonValue::as_u64).is_some_and(|r| r != 3) {
            return Err("processor_config.json: only bicubic resampling (resample 3) is implemented".into());
        }
    }
    let tok = match tokenizer_json {
        Some(t) => Some(parse_json(t).map_err(|e| format!("tokenizer_config.json: {e}"))?),
        None => None,
    };
    let pad = token_text(tok.as_ref(), "image_token", IMAGE_TOKEN);
    let boi = token_text(tok.as_ref(), "boi_token", BOI_TOKEN);
    let eoi = token_text(tok.as_ref(), "eoi_token", EOI_TOKEN);
    // pixels of a soft token's square (`with_sizes` took the same sizes, so this is no refusal in
    // practice)
    let px = g4v_token_pixels(d.patch, d.pool)
        .ok_or("vision_config: a soft token's square has more pixels than 64 bits hold")?;
    let s = VisionSpec {
        processor: ImageProcessor::Gemma4 {
            merged: d.kind == G4vKind::Unified,
        },
        image_block: pad.clone(),
        image_text: format!("{boi}{pad}{eoi}"),
        rescale,
        depth: d.layers,
        hidden: d.hid,
        n_heads: d.nh,
        intermediate: d.inter,
        in_channels: 3,
        patch: d.patch,
        temporal: 1,
        merge: d.pool,
        out_hidden: d.out,
        n_pos: d.npos,
        gelu_tanh: true,
        gelu_bf16: false,
        image_token_id: id("image_token_id")?,
        // video input of this family is not implemented: the model takes none here
        video_token_id: -1,
        vision_start_id: id("boi_token_id")?,
        vision_end_id: id("eoi_token_id")?,
        mrope_section: [0; 3],
        min_pixels: px,
        max_pixels: max_soft.saturating_mul(px),
        image_mean: [0.0; 3],
        image_std: [1.0; 3],
        video: None,
    };
    s.validate()?;
    Ok(s)
}

// ============================================================================
// Host preprocessing (the checkpoint's processor)
// ============================================================================

/// Soft tokens the budget `vs.max_pixels` holds.
pub fn soft_budget(vs: &VisionSpec) -> u64 {
    g4v_soft_budget(vs.max_pixels, vs.resize_factor())
}

/// The resized `(height, width)` of a `width x height` image: the processor's aspect-preserving
/// size (its floating-point steps here, exactly as the reference evaluates them; the integer rule
/// and the token bound in the verified core).
pub fn target_size(width: usize, height: usize, vs: &VisionSpec) -> Result<(usize, usize), String> {
    if width == 0 || height == 0 {
        return Err("image has zero size".into());
    }
    let side = vs.resize_factor();
    let max_soft = soft_budget(vs);
    let target_px = max_soft * side * side;
    let (h, w) = (height as u64, width as u64);
    let factor = (target_px as f64 / (h * w) as f64).sqrt();
    let floor_side = |ideal: f64| ((ideal / side as f64).floor() as u64) * side;
    let th0 = floor_side(factor * h as f64);
    let tw0 = floor_side(factor * w as f64);
    let max_side = max_soft * side;
    let th = g4v_target_h(th0, tw0, h / w, side, max_side);
    let tw = g4v_target_w(th0, tw0, w / h, side, max_side);
    if !g4v_size_ok(th, tw, vs.patch, vs.merge, max_soft) {
        return Err(format!(
            "an image of {width}x{height} pixels has no size within {max_soft} soft tokens of {side}x{side} pixels"
        ));
    }
    Ok((th as usize, tw as usize))
}

/// Soft tokens of an image resized to `rh x rw` pixels.
pub fn image_tokens(rh: usize, rw: usize, vs: &VisionSpec) -> u64 {
    g4v_tokens(rh as u64, rw as u64, vs.patch, vs.merge)
}

/// The processor's value of pixel byte `v`: the byte times the rescale factor, in f32.
#[inline]
fn rescaled(v: u8, s: f32) -> f32 {
    f32::from(v) * s
}

/// The tower's input value of a processor value: `2 (x - 0.5)` in f32, rounded once to BF16.
#[inline]
fn tower_value(x: f32) -> u16 {
    f32_to_bf16_rne(2.0 * (x - 0.5))
}

/// Walks the patches of a `rh x rw` resized image (`patch` pixels square) in the processor's
/// raster order, calling `put(patch row, patch column, y, x)` for every value's pixel in the
/// patch's own order (y, x; the channel is the innermost index of a row).
fn raster_patches(rh: usize, rw: usize, p: usize, mut put: impl FnMut(usize, usize)) {
    for gy in 0..rh / p {
        for gx in 0..rw / p {
            for py in 0..p {
                for px in 0..p {
                    put(gy * p + py, gx * p + px);
                }
            }
        }
    }
}

/// The patch grid positions in pooling order: block by block of `pool x pool` patches (blocks
/// in reading order, the order of the soft tokens; a block's patches in reading order), from the
/// verified `g4v_pool_y` / `g4v_pool_x`. The tower does not depend on the order of its patches
/// (each carries its own position); in this order the mean of a block is one product with a
/// 0 / 1 matrix. `pool` is positive (`g4v_grid_ok`).
pub fn pooling_order(gh: usize, gw: usize, pool: usize) -> Vec<(usize, usize)> {
    let (nby, nbx, p) = ((gh / pool) as u64, (gw / pool) as u64, pool as u64);
    (0..g4v_pool_count(nby, nbx, p))
        .map(|i| (g4v_pool_y(i, nbx, p) as usize, g4v_pool_x(i, nbx, p) as usize))
        .collect()
}

fn resize_for(img: &RgbImage, vs: &VisionSpec) -> Result<(usize, usize, Vec<u8>), String> {
    let (rh, rw) = target_size(img.width, img.height, vs)?;
    let px = if (rw, rh) == (img.width, img.height) {
        img.data.clone()
    } else {
        resize_rgb(&img.data, img.width, img.height, rw, rh)
    };
    Ok((rh, rw, px))
}

/// The processor's `pixel_values` rows (f32, before any rounding): for the tower one row per
/// patch in raster order (y, x, channel); for the unified form one row per merged patch in raster
/// order of the merged grid (its `pool * patch` square, y, x, channel). Checked against stored
/// reference dumps.
pub fn pixel_rows_f32(img: &RgbImage, vs: &VisionSpec) -> Result<(usize, usize, Vec<f32>), String> {
    let (rh, rw, px) = resize_for(img, vs)?;
    let s = vs.rescale as f32;
    let p = vs.patch as usize;
    let unit = match vs.processor {
        ImageProcessor::Gemma4 { merged: true } => vs.resize_factor() as usize,
        _ => p,
    };
    let mut out = Vec::with_capacity(rh * rw * 3);
    raster_patches(rh, rw, unit, |y, x| {
        let at = (y * rw + x) * 3;
        for c in 0..3 {
            out.push(rescaled(px[at + c], s));
        }
    });
    Ok((rh / p, rw / p, out))
}

/// Full host pipeline of one image: resize, rescale, the BF16 input rows of the image path.
/// The tower takes `bf16(2 (x - 0.5))` patch rows in pooling order; the unified embedder takes
/// `bf16(x)` merged rows in raster order.
pub fn prepare_image(img: &RgbImage, vs: &VisionSpec) -> Result<PreparedImage, String> {
    let (rh, rw, px) = resize_for(img, vs)?;
    let s = vs.rescale as f32;
    let p = vs.patch as usize;
    let pool = vs.merge as usize;
    let (gh, gw) = (rh / p, rw / p);
    let merged = matches!(vs.processor, ImageProcessor::Gemma4 { merged: true });
    let mut pixels = Vec::with_capacity(rh * rw * 3);
    // one value per byte: the BF16 input value of every byte (rounded once)
    let mut lut = [0u16; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        let x = rescaled(v as u8, s);
        *slot = if merged {
            f32_to_bf16_rne(x)
        } else {
            tower_value(x)
        };
    }
    if merged {
        raster_patches(rh, rw, p * pool, |y, x| {
            let at = (y * rw + x) * 3;
            for c in 0..3 {
                pixels.push(lut[px[at + c] as usize]);
            }
        });
    } else {
        for (gy, gx) in pooling_order(gh, gw, pool) {
            for py in 0..p {
                let at = ((gy * p + py) * rw + gx * p) * 3;
                for v in &px[at..at + p * 3] {
                    pixels.push(lut[*v as usize]);
                }
            }
        }
    }
    let row_len = row_len(vs);
    debug_assert!(row_len > 0 && pixels.len().is_multiple_of(row_len));
    Ok(PreparedImage {
        src_width: img.width,
        src_height: img.height,
        width: rw,
        height: rh,
        grid_h: gh,
        grid_w: gw,
        row_len,
        merge: pool,
        pixels,
    })
}

/// Position rows of the tower's patches in pooling order: `bf16(column row + row row)` of the
/// table `[2][npos][hid]` (axis 0 the columns, axis 1 the rows), summed in f32 and rounded once
/// (the reference adds the two BF16 rows in BF16).
pub fn tower_positions(table: &[u16], npos: usize, hid: usize, gh: usize, gw: usize, pool: usize) -> Result<Vec<u16>, String> {
    if !g4v_table_holds(gh as u64, gw as u64, npos as u64) {
        return Err(format!(
            "an image of {gh} x {gw} patches: the position table has {npos} rows per axis"
        ));
    }
    let bf = crate::fp8::bf16_to_f32;
    let mut out = Vec::with_capacity(gh * gw * hid);
    for (y, x) in pooling_order(gh, gw, pool) {
        let tx = &table[x * hid..(x + 1) * hid];
        let ty = &table[(npos + y) * hid..(npos + y + 1) * hid];
        for c in 0..hid {
            out.push(f32_to_bf16_rne(bf(tx[c]) + bf(ty[c])));
        }
    }
    Ok(out)
}

/// Position rows of the unified embedder's merged patches in raster order: `bf16(pos[x][0] +
/// pos[y][1])` of the table `[npos][2][hid]`.
pub fn merged_positions(table: &[u16], npos: usize, hid: usize, mh: usize, mw: usize) -> Result<Vec<u16>, String> {
    if !g4v_table_holds(mh as u64, mw as u64, npos as u64) {
        return Err(format!(
            "an image of {mh} x {mw} merged patches: the position table has {npos} rows per axis"
        ));
    }
    let bf = crate::fp8::bf16_to_f32;
    let mut out = Vec::with_capacity(mh * mw * hid);
    for y in 0..mh {
        for x in 0..mw {
            let ax = &table[(x * 2) * hid..(x * 2 + 1) * hid];
            let ay = &table[(y * 2 + 1) * hid..(y * 2 + 2) * hid];
            for c in 0..hid {
                out.push(f32_to_bf16_rne(bf(ax[c]) + bf(ay[c])));
            }
        }
    }
    Ok(out)
}

/// The tower's rotation tables `{dh, nh N}` (BF16, as the reference casts them) of its patches
/// in pooling order: channel `k` of a head turns with the patch's column or row
/// (`g4v_rope_row_axis`: the first half of the head the column, the second the row) by the
/// frequency `g4v_rope_freq` selects; inside a half, channels `j` and `j + dh / 4` are a pair
/// with the frequency `theta^(-2 j / (dh / 2))` (in f32, as the reference computes it). Returns
/// `(cos, sin)`, each patch's row repeated for every head.
pub fn tower_rope(d: &G4vDims, gh: usize, gw: usize) -> (Vec<u16>, Vec<u16>) {
    let dh = d.dh as usize;
    let quart = g4v_rot_quarter(d.dh);
    // the reference's frequencies, `theta^(-2 j / (dh / 2))` over the places of a quarter
    let inv: Vec<f32> = (0..quart)
        .map(|j| 1.0f32 / (d.theta as f32).powf((2 * j) as f32 / (dh / 2) as f32))
        .collect();
    let order = pooling_order(gh, gw, d.pool as usize);
    let nh = d.nh as usize;
    let mut cs = Vec::with_capacity(order.len() * nh * dh);
    let mut sn = Vec::with_capacity(order.len() * nh * dh);
    let mut c1 = vec![0u16; dh];
    let mut s1 = vec![0u16; dh];
    for &(y, x) in &order {
        for k in 0..dh {
            let pos = if g4v_rope_row_axis(k as u64, quart) { y } else { x } as f32;
            let ang = pos * inv[g4v_rope_freq(k as u64, quart) as usize];
            c1[k] = f32_to_bf16_rne(ang.cos());
            s1[k] = f32_to_bf16_rne(ang.sin());
        }
        for _ in 0..nh {
            cs.extend_from_slice(&c1);
            sn.extend_from_slice(&s1);
        }
    }
    (cs, sn)
}

