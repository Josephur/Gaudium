//! The vision tower of an Inkling 2 model given as a GGUF file: the projector file beside the
//! model (`mmproj*.gguf`), its geometry from the file's `clip.vision.*` metadata, and its tensors
//! presented to the tower's weight packer under the publisher's names, shapes and element order
//! ([`TowerSource`]), so a projector file and the publisher's checkpoint give one set of device
//! tensors.
//!
//! How the projector file relates to the checkpoint (from the files themselves and the public
//! converter's description of its transforms):
//! - Every tensor keeps the checkpoint's row-major shape (GGUF lists the dimensions innermost
//!   first) except the patch embedding: the checkpoint's `[out, y * x * channel]` linear is stored
//!   as a convolution kernel `[out, channel, y, x]`.
//! - Large matrices are BF16 as in the checkpoint; the embeddings, norms, biases and layer
//!   scales are stored as F32 (here: the checkpoint's BF16 values widened, so they round back
//!   exactly).
//! - The layer scales (`ls_1.gamma`, `ls_2.gamma`) are `ls1.weight`, `ls2.weight`; the class
//!   token and its position are `v.class_embd` and `v.class_position_embd.weight`; the merger's
//!   two linears are `mm.1.weight` and `mm.2.weight`.
//! - The file does not carry the rotary base, the colour channels or the image placeholder:
//!   the family fixes them ([`super::FAMILY`]); the placeholder's id comes from the model's own
//!   vocabulary.
//!
//! Which file is used: the `mmproj*.gguf` files of the model's directory in name order, folded
//! through the verified rule (`verified/inkling2_mmproj.bend`: a file matches when its projector
//! type is the family's and its projection width is the model's hidden size; among matching
//! files the best precision rank wins, the first in name order on a tie). No candidate: the
//! model is served text only. Candidates of which none matches: refused with every file's reason.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{bf16s, IkvDims, TowerSource, FAMILY};
use crate::engine::inkling2::gguf_source;
use crate::gguf::dequant::{rows_bf16, rows_f32};
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::model_spec::ModelSpec;
use crate::verified::{
    mmp_fit, mmp_matches, mmp_outcome, mmp_rank, mmp_step, MmpFit, MmpOutcome, MmpPick,
    MmpPrecision,
};
use crate::vision::VisionSpec;

/// The tower of a GGUF model: the projector file it is read from and its geometry.
#[derive(Debug, Clone)]
pub struct ModelTower {
    pub projector: PathBuf,
    pub spec: VisionSpec,
    pub dims: IkvDims,
}

/// The `mmproj*.gguf` files beside the model at `model` (a GGUF file, or a directory of one),
/// in name order.
pub fn candidates(model: &Path) -> Vec<PathBuf> {
    let dir = if model.is_dir() {
        model.to_path_buf()
    } else {
        match model.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        }
    };
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    let n = n.to_ascii_lowercase();
                    n.starts_with("mmproj") && n.ends_with(".gguf")
                })
        })
        .collect();
    out.sort();
    out
}

/// The precision class of a projector file: the type that holds most of its values.
pub fn precision(g: &Gguf) -> MmpPrecision {
    let mut by_type: Vec<(GgmlType, u64)> = Vec::new();
    for t in g.tensors() {
        match by_type.iter_mut().find(|(ty, _)| *ty == t.ty) {
            Some((_, n)) => *n = n.saturating_add(t.n_elems),
            None => by_type.push((t.ty, t.n_elems)),
        }
    }
    let bulk = by_type.iter().max_by_key(|(_, n)| *n).map(|(ty, _)| *ty);
    match bulk {
        Some(GgmlType::BF16) => MmpPrecision::MmpBf16,
        Some(GgmlType::F32) => MmpPrecision::MmpF32,
        Some(GgmlType::F16) => MmpPrecision::MmpF16,
        _ => MmpPrecision::MmpOther,
    }
}

fn projector_type(g: &Gguf) -> &str {
    g.get_str("clip.vision.projector_type").unwrap_or("")
}

fn projection_dim(g: &Gguf) -> Option<u64> {
    g.get_u64("clip.vision.projection_dim")
}

/// One candidate's verdict for a model of hidden size `hidden`: whether it matches, its rank,
/// and why not when it does not.
fn verdict(g: &Gguf, hidden: u64) -> (bool, u64, Option<String>) {
    let family = projector_type(g) == FAMILY.projector_type;
    let rank = mmp_rank(precision(g));
    let Some(proj) = projection_dim(g) else {
        return (false, rank, Some("no clip.vision.projection_dim".into()));
    };
    // whether it matches is what the search folds (`mmp_matches`); why not is the same rule's
    // reason (`mmp_fit`, which agrees with it: `mmp_fit_serves_exactly`)
    let ok = mmp_matches(family, proj, hidden);
    (ok, rank, refusal(g, mmp_fit(family, proj, hidden), proj, hidden))
}

/// Why a projector file of width `proj` does not serve a model of hidden size `hidden`, in
/// words, from the verified reason `fit` (`None` when it serves it).
fn refusal(g: &Gguf, fit: MmpFit, proj: u64, hidden: u64) -> Option<String> {
    match fit {
        MmpFit::MmpServes => None,
        MmpFit::MmpOtherType => Some(format!(
            "projector type '{}', not '{}'",
            projector_type(g),
            FAMILY.projector_type
        )),
        MmpFit::MmpWidth => Some(format!(
            "projection width {proj}, the model's hidden size is {hidden}"
        )),
    }
}

/// The projector file the model at `model` (hidden size `hidden`) reads its tower from.
/// `Ok(None)`: no candidate beside it (text only).
pub fn find_projector(model: &Path, hidden: u64) -> Result<Option<PathBuf>, String> {
    let cands = candidates(model);
    let mut best = MmpPick::MmpNone;
    let mut why = Vec::new();
    for (i, c) in cands.iter().enumerate() {
        let (ok, rank) = match Gguf::open(c) {
            Ok(g) => {
                let (ok, rank, reason) = verdict(&g, hidden);
                if let Some(r) = reason {
                    why.push(format!("{}: {r}", c.display()));
                }
                (ok, rank)
            }
            Err(e) => {
                why.push(format!("{}: {e}", c.display()));
                (false, 0)
            }
        };
        best = mmp_step(ok, best, i as u64, rank);
    }
    // the chosen file, text only, or a refusal: the verified outcome of the search
    match mmp_outcome(cands.len() as u64, best) {
        MmpOutcome::MmpUse(i) => cands
            .get(i as usize)
            .cloned()
            .map(Some)
            .ok_or_else(|| format!("projector choice {i} of {} files", cands.len())),
        MmpOutcome::MmpTextOnly => Ok(None),
        MmpOutcome::MmpNoMatch => Err(format!(
            "no projector file beside the model matches it:\n  {}",
            why.join("\n  ")
        )),
    }
}

fn shape_of(g: &Gguf, name: &str) -> Result<Vec<u64>, String> {
    Ok(g.get(name).map_err(|e| e.to_string())?.shape_row_major())
}

fn mul(a: u64, b: u64, what: &str) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("projector file: {what} overflows"))
}

/// The tower's geometry from a projector file's metadata, checked against its own tensors.
pub fn projector_dims(g: &Gguf) -> Result<IkvDims, String> {
    if projector_type(g) != FAMILY.projector_type {
        return Err(format!(
            "projector type '{}' is not this family's '{}'",
            projector_type(g),
            FAMILY.projector_type
        ));
    }
    let need = |k: &str| -> Result<u64, String> {
        g.get_u64(&format!("clip.vision.{k}"))
            .ok_or_else(|| format!("projector metadata clip.vision.{k} is missing or not a number"))
    };
    let hid = need("embedding_length")?;
    let layers = need("block_count")?;
    let nh = need("attention.head_count")?;
    let inter = need("feed_forward_length")?;
    let patch = need("patch_size")?;
    let merge = need("spatial_merge_size")?;
    let image = need("image_size")?;
    let out = need("projection_dim")?;
    let eps = g
        .get_f64("clip.vision.attention.layer_norm_epsilon")
        .ok_or("projector metadata clip.vision.attention.layer_norm_epsilon is missing")?;
    if patch == 0 || !image.is_multiple_of(patch) {
        return Err(format!(
            "projector metadata: image_size {image} is not a whole number of {patch}-pixel patches"
        ));
    }
    let grid = image / patch;
    let ch = FAMILY.channels;
    let check = |name: &str, want: Vec<u64>| -> Result<(), String> {
        let got = shape_of(g, name)?;
        if got != want {
            return Err(format!(
                "projector tensor '{name}' has shape {got:?}, the metadata gives {want:?}"
            ));
        }
        Ok(())
    };
    check("v.patch_embd.weight", vec![hid, ch, patch, patch])?;
    check(
        "v.position_embd.weight",
        vec![mul(grid, grid, "the position grid")?, hid],
    )?;
    check("v.class_embd", vec![hid])?;
    check("v.class_position_embd.weight", vec![hid])?;
    check("v.blk.0.ffn_up.weight", vec![inter, hid])?;
    let merged = mul(hid, mul(merge, merge, "the merge")?, "the merged width")?;
    let m1 = shape_of(g, "mm.1.weight")?;
    let proj = match m1.as_slice() {
        [p, w] if *w == merged => *p,
        _ => {
            return Err(format!(
                "projector tensor 'mm.1.weight' has shape {m1:?}, expected [*, {merged}] ({merge} x {merge} patches of {hid})"
            ))
        }
    };
    check("mm.2.weight", vec![out, proj])?;
    let present = (0..=layers)
        .take_while(|l| g.tensor(&format!("v.blk.{l}.attn_q.weight")).is_some())
        .count() as u64;
    if present != layers {
        return Err(format!(
            "projector metadata block_count is {layers}, the file holds {present} layers"
        ));
    }
    let d = IkvDims {
        hid,
        layers,
        nh,
        dh: if nh > 0 { hid / nh } else { 0 },
        inter,
        patch,
        merge,
        grid,
        proj,
        out,
        channels: ch,
        eps: eps as f32,
        theta: FAMILY.rope_theta,
    };
    d.validate()?;
    Ok(d)
}

/// The id of token `text` in a GGUF model's vocabulary.
fn token_id(main: &Gguf, text: &str) -> Result<i32, String> {
    let toks = main
        .kv("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .ok_or("the model's GGUF metadata has no tokenizer.ggml.tokens")?;
    let at = toks
        .byte_strings()
        .position(|s| s == text.as_bytes())
        .ok_or_else(|| format!("the model's vocabulary has no image placeholder {text}"))?;
    i32::try_from(at).map_err(|_| format!("image placeholder id {at} is out of range"))
}

fn three(g: &Gguf, key: &str) -> Result<[f64; 3], String> {
    let Some(v) = g.kv(key) else {
        return Ok([0.5; 3]);
    };
    let a = v
        .as_array()
        .and_then(|a| a.to_f64())
        .filter(|a| a.len() == 3)
        .ok_or_else(|| format!("projector metadata {key} is not three numbers"))?;
    Ok([a[0], a[1], a[2]])
}

/// The image spec and geometry of the tower in projector file `proj` for the model `main` of
/// hidden size `hidden`: a file the search would refuse (another projector type, or a projection
/// that is not the model's width) is refused for the same reason, in the same words.
pub fn tower_spec(main: &Gguf, proj: &Gguf, hidden: u64) -> Result<(VisionSpec, IkvDims), String> {
    if let (_, _, Some(why)) = verdict(proj, hidden) {
        return Err(why);
    }
    // a file the verified rule serves is of the family's type with the model's width
    // (`mmp_fit_serves_exactly`), so the geometry's own type check and width agree with it
    let d = projector_dims(proj)?;
    let id = token_id(main, FAMILY.image_token)?;
    let mean = three(proj, "clip.vision.image_mean")?;
    let std = three(proj, "clip.vision.image_std")?;
    let spec = super::tiled_spec(&d, id, FAMILY.image_token, mean, std)?;
    Ok((spec, d))
}

fn hidden_of(main: &Gguf) -> Result<u64, String> {
    main.arch_kv("embedding_length")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            format!(
                "GGUF metadata key {}.embedding_length is missing",
                main.arch()
            )
        })
}

/// The tower of the GGUF model `main` opened from `model`: `None` when no projector file lies
/// beside it (or the model is of another family).
pub fn model_tower(model: &Path, main: &Gguf) -> Result<Option<ModelTower>, String> {
    if !gguf_source::is_family(main) {
        return Ok(None);
    }
    let hidden = hidden_of(main)?;
    let Some(projector) = find_projector(model, hidden)? else {
        return Ok(None);
    };
    let g = Gguf::open(&projector).map_err(|e| e.to_string())?;
    let (spec, dims) =
        tower_spec(main, &g, hidden).map_err(|e| format!("{}: {e}", projector.display()))?;
    Ok(Some(ModelTower {
        projector,
        spec,
        dims,
    }))
}

/// The image spec of the GGUF model at `model` (an error when it has no tower).
pub fn model_vision_spec(model: &Path) -> Result<VisionSpec, String> {
    let main = Gguf::open(model).map_err(|e| format!("open GGUF: {e}"))?;
    if !gguf_source::is_family(&main) {
        return Err(format!(
            "a GGUF model of architecture '{}' takes no images",
            main.arch()
        ));
    }
    model_tower(model, &main)?.map(|t| t.spec).ok_or_else(|| {
        format!(
            "no projector file (mmproj*.gguf) beside {}: this model takes no images",
            model.display()
        )
    })
}

/// The model spec of the GGUF model at `model` when a projector file may give it a tower (one
/// lies beside it: which one, and whether it fits, is decided when the tower is loaded).
pub fn model_spec_with_tower(model: &Path) -> Option<ModelSpec> {
    let main = Gguf::open(model).ok()?;
    if !gguf_source::is_family(&main) || candidates(model).is_empty() {
        return None;
    }
    let mut spec = ModelSpec::from_config_json(&gguf_source::config_json(&main).ok()?).ok()?;
    spec.has_vision = true;
    Some(spec)
}

// ============================================================================
// The projector file as a tower source
// ============================================================================

/// A projector file as a [`TowerSource`].
pub struct ProjectorSource {
    g: Arc<Gguf>,
}

/// The projector file's name of checkpoint tower tensor `name` (below `model.visual.`), for
/// every tensor that keeps its shape and element order.
fn tensor_name(name: &str) -> Option<String> {
    Some(match name {
        "backbone.embeddings.class_embedding" => "v.class_embd".into(),
        "backbone.embeddings.cls_position_embedding" => "v.class_position_embd.weight".into(),
        "backbone.embeddings.position_embedding" => "v.position_embd.weight".into(),
        "backbone.ln_pre.weight" => "v.pre_ln.weight".into(),
        "backbone.ln_pre.bias" => "v.pre_ln.bias".into(),
        "proj_in.weight" => "mm.1.weight".into(),
        "proj_out.weight" => "mm.2.weight".into(),
        _ => {
            let (l, what) = name.strip_prefix("backbone.layers.")?.split_once('.')?;
            l.parse::<u64>().ok()?;
            let tail = match what {
                "ln_1.weight" => "ln1.weight",
                "ln_1.bias" => "ln1.bias",
                "ln_2.weight" => "ln2.weight",
                "ln_2.bias" => "ln2.bias",
                "ls_1.gamma" => "ls1.weight",
                "ls_2.gamma" => "ls2.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "self_attn.out_proj.weight" => "attn_out.weight",
                "self_attn.out_proj.bias" => "attn_out.bias",
                "mlp.c_fc.weight" => "ffn_up.weight",
                "mlp.c_fc.bias" => "ffn_up.bias",
                "mlp.c_proj.weight" => "ffn_down.weight",
                "mlp.c_proj.bias" => "ffn_down.bias",
                _ => return None,
            };
            format!("v.blk.{l}.{tail}")
        }
    })
}

const PATCH_EMBEDDING: &str = "backbone.embeddings.patch_embedding.weight";

/// The checkpoint's patch rows `[out][y][x][channel]` from the kernel `[out][channel][y][x]`.
fn patch_rows(kernel: &[u16], out: usize, ch: usize, p: usize) -> Vec<u16> {
    let row = ch * p * p;
    let mut v = vec![0u16; kernel.len()];
    for o in 0..out {
        for c in 0..ch {
            for y in 0..p {
                for x in 0..p {
                    v[o * row + (y * p + x) * ch + c] = kernel[o * row + (c * p + y) * p + x];
                }
            }
        }
    }
    v
}

fn u16_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl ProjectorSource {
    pub fn new(g: Arc<Gguf>) -> Self {
        Self { g }
    }

    pub fn open(path: &Path) -> Result<Self, String> {
        Ok(Self::new(Arc::new(
            Gguf::open(path).map_err(|e| format!("open projector file: {e}"))?,
        )))
    }

    fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.g.get(name).map_err(|e| e.to_string())
    }

    /// Every value of `t` as BF16 (a BF16 tensor's bytes as stored).
    fn all_bf16(&self, t: &TensorInfo) -> Result<Cow<'_, [u8]>, String> {
        if t.ty == GgmlType::BF16 {
            return Ok(Cow::Borrowed(self.g.tensor_data(t)));
        }
        let mut v = vec![0u16; t.n_elems as usize];
        rows_bf16(&self.g, t, 0, t.rows(), None, &mut v).map_err(|e| e.to_string())?;
        Ok(Cow::Owned(u16_le(&v)))
    }
}

impl TowerSource for ProjectorSource {
    fn bf16(&self, name: &str, shape: &[u64]) -> Result<Cow<'_, [u8]>, String> {
        if name == PATCH_EMBEDDING {
            let [out, row] = *shape else {
                return Err(format!("'{name}': the patch linear is 2-D, not {shape:?}"));
            };
            let t = self.tensor("v.patch_embd.weight")?;
            let got = t.shape_row_major();
            let (ch, p) = match got.as_slice() {
                [o, c, py, px] if *o == out && py == px && c * py * px == row => (*c, *py),
                _ => {
                    return Err(format!(
                        "projector tensor 'v.patch_embd.weight' has shape {got:?}, not a kernel of {out} x {row} values"
                    ))
                }
            };
            let k = bf16s(&self.all_bf16(t)?);
            return Ok(Cow::Owned(u16_le(&patch_rows(
                &k,
                out as usize,
                ch as usize,
                p as usize,
            ))));
        }
        let gname = tensor_name(name)
            .ok_or_else(|| format!("no projector tensor is known for tower tensor '{name}'"))?;
        let t = self.tensor(&gname)?;
        if t.shape_row_major() != shape {
            return Err(format!(
                "projector tensor '{gname}' has shape {:?}, expected {shape:?}",
                t.shape_row_major()
            ));
        }
        self.all_bf16(t)
    }

    fn f32s(&self, name: &str, len: u64) -> Result<Vec<f32>, String> {
        let gname = tensor_name(name)
            .ok_or_else(|| format!("no projector tensor is known for tower tensor '{name}'"))?;
        let t = self.tensor(&gname)?;
        if t.shape_row_major() != [len] {
            return Err(format!(
                "projector tensor '{gname}' has shape {:?}, expected [{len}]",
                t.shape_row_major()
            ));
        }
        let mut v = vec![0f32; len as usize];
        rows_f32(&self.g, t, 0, t.rows(), &mut v).map_err(|e| e.to_string())?;
        Ok(v)
    }
}

#[cfg(test)]
#[path = "inkling2_vision_gguf_tests.rs"]
mod tests;
