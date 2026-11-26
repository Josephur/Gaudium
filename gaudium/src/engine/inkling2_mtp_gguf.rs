//! The block drafter of an Inkling 2 model given as a GGUF file: the drafter's own GGUF file
//! beside the model, its contract from that file's metadata, and its tensors presented to the
//! drafter packer under the checkpoint's names, shapes and element order ([`GgufMtp`]), so a
//! drafter file and the publisher's `mtp.safetensors` give one set of device tensors.
//!
//! Two layouts of drafter file exist (architecture `dflash` both):
//! - **Plain** (it stores `dspark_fuse.weight`): every checkpoint tensor under its own name,
//!   matrices BF16 as in the checkpoint, norms and biases widened to F32 (they round back
//!   exactly). Every device tensor is byte-identical to the checkpoint's.
//! - **Folded** (it stores `blk.0.ctx_mix.weight` and `markov_w2.weight`): matrices of the layers
//!   in a block format, and pre-computed forms instead of four checkpoint tensors: the mixes'
//!   softmax weights per row (`ctx_mix`, `markov_ctx_mix`) instead of `fuse_weight`, the mask
//!   row once per mask position (`mask_embd`), the recurrent head's latent map already folded
//!   through the unembedding (`markov_w2`: weight and bias) instead of `hidden_out`, and the
//!   token table `markov_w1` without the padding rows. Its `sliding_window` counts the anchor's
//!   own position (one more than the context positions a windowed layer reads).
//!   The file does not carry what its pre-computed tensors were made from, so they are used as
//!   stored (the matching on-card fold is skipped) and the padding rows are zero (no draft and
//!   no anchor is a padding id).
//!
//! Which file is used: the `.gguf` files beside the model, then those of a `dspark/` directory
//! beside it (each in name order; the model's own file and projector files are left out),
//! folded through the verified rule (`verified/inkling2_mtpfile.bend`, `mtpf_step`): a file
//! matches when it is a block drafter of one of the two layouts and its hidden size and
//! vocabulary are the model's; among matching files the best precision rank wins, the first in
//! that order on a tie. Why a file does not serve the model is the same rule's (`mtpf_fit`),
//! both in the search's refusal and when the chosen file's contract is read.

use super::*;
use crate::gguf::dequant::{rows_bf16_par, rows_f32};
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::verified::{
    ik_bf16_bytes, mtpf_fit, mtpf_rank, mtpf_rows_fit, mtpf_step, MmpPick, MtpfFit, MtpfPrecision,
};
use std::path::{Path, PathBuf};

/// The architecture name of a drafter file.
const ARCH: &str = "dflash";
/// The directory beside the model that may hold drafter files.
pub const DRAFTER_DIR: &str = "dspark";

/// The two drafter file layouts (module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Plain,
    Folded,
}

/// The layout of a drafter file of this family, `None` for any other file.
pub fn layout(g: &Gguf) -> Option<Layout> {
    if g.arch() != ARCH || g.tensor("joint_proj.weight").is_none() {
        return None;
    }
    if g.tensor("dspark_fuse.weight").is_some() {
        Some(Layout::Plain)
    } else if g.tensor("blk.0.ctx_mix.weight").is_some() && g.tensor("markov_w2.weight").is_some() {
        Some(Layout::Folded)
    } else {
        None
    }
}

fn dir_of(model: &Path) -> PathBuf {
    if model.is_dir() {
        model.to_path_buf()
    } else {
        match model.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        }
    }
}

/// The directories searched for a drafter file of the model at `model`, in order.
pub fn search_dirs(model: &Path) -> [PathBuf; 2] {
    let d = dir_of(model);
    [d.clone(), d.join(DRAFTER_DIR)]
}

/// The candidate drafter files of the model at `model` (a GGUF file, or a directory of one):
/// every `.gguf` file of [`search_dirs`], each directory in name order, without the model's own
/// file and the projector files (`mmproj*`).
pub fn candidates(model: &Path) -> Vec<PathBuf> {
    let own = std::fs::canonicalize(model).ok();
    let mut out = Vec::new();
    for dir in search_dirs(model) {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut here: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                        let n = n.to_ascii_lowercase();
                        n.ends_with(".gguf") && !n.starts_with("mmproj")
                    })
                    && (own.is_none() || std::fs::canonicalize(p).ok() != own)
            })
            .collect();
        here.sort();
        out.extend(here);
    }
    out
}

/// The precision class of a drafter file: the type that holds most of its values.
pub fn precision(g: &Gguf) -> MtpfPrecision {
    let mut by_type: Vec<(GgmlType, u64)> = Vec::new();
    for t in g.tensors() {
        match by_type.iter_mut().find(|(ty, _)| *ty == t.ty) {
            Some((_, n)) => *n = n.saturating_add(t.n_elems),
            None => by_type.push((t.ty, t.n_elems)),
        }
    }
    match by_type.iter().max_by_key(|(_, n)| *n).map(|(ty, _)| *ty) {
        Some(GgmlType::BF16) => MtpfPrecision::MtpfBf16,
        Some(GgmlType::F32) => MtpfPrecision::MtpfF32,
        Some(GgmlType::Q8_0) => MtpfPrecision::MtpfQ8,
        _ => MtpfPrecision::MtpfOther,
    }
}

/// The hidden size a GGUF file states (`<arch>.embedding_length`).
fn hidden_of(g: &Gguf) -> Option<u64> {
    g.arch_kv("embedding_length").and_then(|v| v.as_u64())
}

/// The vocabulary a GGUF file states: its unpadded vocabulary size, else its token list.
fn vocab_of(g: &Gguf) -> Option<u64> {
    g.arch_kv("vocab_size_unpadded")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            g.kv("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
        })
}

/// The model the drafter must serve: its hidden size and its (unpadded) vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub hid: u64,
    pub vocab: u64,
}

/// What the selection rule reads of one candidate: whether it is a block drafter of the
/// family's layout, its hidden size, its vocabulary and its preference rank.
fn facts(g: &Gguf) -> (bool, u64, u64, u64) {
    (
        layout(g).is_some(),
        hidden_of(g).unwrap_or(0),
        vocab_of(g).unwrap_or(0),
        mtpf_rank(precision(g)),
    )
}

/// Why a candidate does not serve the model `t` (`None`: it does), from the facts the search
/// folds through `mtpf_step`.
fn why_not(g: &Gguf, t: Target) -> Option<String> {
    let (drafter, hid, vocab, _) = facts(g);
    refusal(
        g,
        mtpf_fit(drafter, hid, t.hid, vocab, t.vocab),
        hid,
        vocab,
        t,
    )
}

/// The words for the verified reason `fit` a file of hidden size `hid` and vocabulary `vocab`
/// does not serve the model `t` for (`None`: it serves it). Both numbers are in every size
/// refusal.
fn refusal(g: &Gguf, fit: MtpfFit, hid: u64, vocab: u64, t: Target) -> Option<String> {
    match fit {
        MtpfFit::MtpfServes => None,
        MtpfFit::MtpfNotDrafter => Some(format!(
            "not a block drafter of this family (architecture '{}')",
            g.arch()
        )),
        MtpfFit::MtpfHidden => Some(format!("hidden size {hid}, the model's is {}", t.hid)),
        MtpfFit::MtpfVocab => Some(format!(
            "vocabulary of {vocab} tokens, the model's has {}",
            t.vocab
        )),
    }
}

/// The drafter file of the model at `model` (serving `t`), or a refusal that says where it
/// looked and why each file found there does not serve the model.
pub fn find_drafter(model: &Path, t: Target) -> Result<PathBuf, String> {
    let cands = candidates(model);
    let mut best = MmpPick::MmpNone;
    let mut why = Vec::new();
    for (i, c) in cands.iter().enumerate() {
        let (drafter, hid, vocab, rank) = match Gguf::open(c) {
            Ok(g) => {
                if let Some(r) = why_not(&g, t) {
                    why.push(format!("{}: {r}", c.display()));
                }
                facts(&g)
            }
            Err(e) => {
                why.push(format!("{}: {e}", c.display()));
                (false, 0, 0, 0)
            }
        };
        best = mtpf_step(drafter, hid, t.hid, vocab, t.vocab, best, i as u64, rank);
    }
    let [d0, d1] = search_dirs(model);
    match best {
        MmpPick::MmpAt(i, _) => cands
            .get(i as usize)
            .cloned()
            .ok_or_else(|| format!("drafter choice {i} of {} files", cands.len())),
        MmpPick::MmpNone if cands.is_empty() => Err(format!(
            "Inkling 2: no drafter file (a block drafter GGUF) beside the GGUF model: looked in {} and {}",
            d0.display(),
            d1.display()
        )),
        MmpPick::MmpNone => Err(format!(
            "Inkling 2: no drafter file beside the GGUF model serves it (looked in {} and {}):\n  {}",
            d0.display(),
            d1.display(),
            why.join("\n  ")
        )),
    }
}

fn shape_of(g: &Gguf, name: &str) -> Result<Vec<u64>, String> {
    Ok(g.get(name).map_err(|e| e.to_string())?.shape_row_major())
}

/// The drafter's contract from a drafter file's metadata, checked against its own tensors and
/// against the model `t` (both numbers in every refusal).
pub fn drafter_config(g: &Gguf, path: &Path, t: Target) -> Result<MtpConfig, String> {
    let bad = |what: String| format!("Inkling 2 drafter file {}: {what}", path.display());
    let need = |k: &str| -> Result<u64, String> {
        g.arch_kv(k)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| bad(format!("metadata {ARCH}.{k} is missing or not a number")))
    };
    let lay = layout(g);
    // the sizes are read as the contract of a file of the family's layout, where a missing one is
    // an error of its own; a file of another layout is refused for that whatever it states
    // (`mtpf_fit_not_drafter_exactly`), so its sizes are not read
    let (hid, vocab) = match lay {
        Some(_) => (
            need("embedding_length")?,
            vocab_of(g).ok_or_else(|| bad("no vocabulary size and no token list".into()))?,
        ),
        None => (0, 0),
    };
    let fit = mtpf_fit(lay.is_some(), hid, t.hid, vocab, t.vocab);
    if let Some(why) = refusal(g, fit, hid, vocab, t) {
        return Err(bad(why));
    }
    // a file the verified rule serves is one of the family's layout (`mtpf_fit_serves_exactly`)
    let lay = lay.ok_or_else(|| {
        bad(format!(
            "not a block drafter of this family (architecture '{}')",
            g.arch()
        ))
    })?;
    let layers = need("block_count")?;
    let present = (0..=layers)
        .take_while(|l| g.tensor(&format!("blk.{l}.attn_q.weight")).is_some())
        .count() as u64;
    if present != layers {
        return Err(bad(format!(
            "metadata block_count is {layers}, the file holds {present} layers"
        )));
    }
    for biased in ["attn_q.bias", "attn_output.bias"] {
        if g.tensor(&format!("blk.0.{biased}")).is_some() {
            return Err(bad(format!(
                "it has {biased} (the drafter is built without biases)"
            )));
        }
    }
    if need("attention.key_length")? != DH {
        return Err(bad(
            "key_length must be 128 (fixed in the drafter kernels)".into()
        ));
    }
    // the recurrent head: [state | previous token | row | context] -> [gate | candidate | latent]
    let jp = shape_of(g, "joint_proj.weight")?;
    let rank = match jp.as_slice() {
        [r3, cols] if r3 % 3 == 0 && *cols == 2 * (r3 / 3) + 2 * hid => r3 / 3,
        _ => {
            return Err(bad(format!(
                "joint_proj.weight has shape {jp:?}, not [3 r, 2 r + 2 x {hid}]"
            )))
        }
    };
    if let Some(r) = g.arch_kv("dspark.markov_rank").and_then(|v| v.as_u64()) {
        if r != rank {
            return Err(bad(format!(
                "metadata markov_rank is {r}, joint_proj.weight gives {rank}"
            )));
        }
    }
    let block = need("block_size")?;
    if lay == Layout::Folded {
        let m = shape_of(g, "mask_embd.weight")?;
        if m != [block.saturating_sub(1), hid] {
            return Err(bad(format!(
                "block_size {block} needs {} mask rows of {hid}, mask_embd.weight has shape {m:?}",
                block.saturating_sub(1)
            )));
        }
    }
    let taps = g
        .arch_kv("target_layers")
        .and_then(|v| v.as_array())
        .and_then(|a| a.to_i64())
        .ok_or_else(|| bad(format!("metadata {ARCH}.target_layers is missing")))?
        .into_iter()
        .map(|x| u64::try_from(x).map_err(|_| bad(format!("target layer {x}"))))
        .collect::<Result<Vec<u64>, String>>()?;
    let pattern = g
        .arch_kv("attention.sliding_window_pattern")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            bad(format!(
                "metadata {ARCH}.attention.sliding_window_pattern is missing"
            ))
        })?;
    let local = (0..pattern.len())
        .map(|k| pattern.get(k).and_then(|v| v.as_bool()))
        .collect::<Option<Vec<bool>>>()
        .ok_or_else(|| bad("sliding_window_pattern is not a list of flags".into()))?;
    if local.len() as u64 != layers {
        return Err(bad(format!(
            "sliding_window_pattern names {} layers, block_count is {layers}",
            local.len()
        )));
    }
    let stored_window = need("attention.sliding_window")?;
    let window = match lay {
        Layout::Plain => stored_window,
        // the folded layout counts the anchor's own position
        Layout::Folded => stored_window
            .checked_sub(1)
            .ok_or_else(|| bad("sliding_window 0".into()))?,
    };
    let c = MtpConfig {
        hid,
        nh: need("attention.head_count")?,
        nkv: need("attention.head_count_kv")?,
        ff: need("feed_forward_length")?,
        local,
        window,
        block,
        rank,
        theta: g
            .arch_kv("rope.freq_base")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| bad(format!("metadata {ARCH}.rope.freq_base is missing")))?,
        eps: g
            .arch_kv("attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32,
        taps,
        weights: MtpWeights::Gguf(path.to_path_buf()),
    };
    c.validate().map_err(bad)
}

/// The main GGUF model's [`Target`].
pub fn target_of(main: &Gguf) -> Result<Target, String> {
    Ok(Target {
        hid: hidden_of(main).ok_or_else(|| {
            format!(
                "GGUF metadata key {}.embedding_length is missing",
                main.arch()
            )
        })?,
        vocab: vocab_of(main).ok_or_else(|| {
            format!(
                "GGUF metadata has no {}.vocab_size_unpadded and no token list",
                main.arch()
            )
        })?,
    })
}

/// The drafter of the GGUF model `main` opened from `model`: the file found beside it and the
/// contract read from that file.
pub fn model_drafter(model: &Path, main: &Gguf) -> Result<MtpConfig, String> {
    let t = target_of(main)?;
    let path = find_drafter(model, t)?;
    let g = Gguf::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    drafter_config(&g, &path, t)
}

// ============================================================================
// The drafter file as a drafter source
// ============================================================================

/// A drafter GGUF file as a [`MtpSource`].
pub struct GgufMtp {
    g: Gguf,
    lay: Layout,
}

/// The file's name of checkpoint tensor `name` in layout `lay`, for every tensor the file keeps
/// in the checkpoint's shape and element order (GGUF lists dimensions innermost first).
fn tensor_name(lay: Layout, name: &str) -> Option<String> {
    let plain = lay == Layout::Plain;
    Some(match name {
        "fuse_weight" if plain => "dspark_fuse.weight".into(),
        "mask_embed" if plain => "mask_embd.weight".into(),
        "markov_head.dflare_norm.weight" if plain => "dspark_norm.weight".into(),
        "markov_head.dflare_norm.weight" => "markov_ctx_norm.weight".into(),
        "markov_head.hidden_out.weight" if plain => "dspark_hidden_out.weight".into(),
        "markov_head.hidden_out.bias" if plain => "dspark_hidden_out.bias".into(),
        "markov_head.markov_w1.weight" => "markov_w1.weight".into(),
        "markov_head.rnn_proj.weight" => "joint_proj.weight".into(),
        "markov_head.rnn_proj.bias" => "joint_proj.bias".into(),
        "norm.weight" => "output_norm.weight".into(),
        _ => {
            let (l, what) = name.strip_prefix("layers.")?.split_once('.')?;
            l.parse::<u64>().ok()?;
            let tail = match what {
                "dflare_norm.weight" if plain => "dflare_norm.weight",
                "dflare_norm.weight" => "ctx_norm.weight",
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.k_proj_ctx.weight" => "attn_k_ctx.weight",
                "self_attn.v_proj_ctx.weight" => "attn_v_ctx.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "self_attn.k_norm_ctx.weight" => "attn_k_ctx_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                _ => return None,
            };
            format!("blk.{l}.{tail}")
        }
    })
}

fn u16_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Threads for decoding a block-format matrix.
fn decode_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(16))
}

impl GgufMtp {
    pub fn open(path: &Path) -> Result<Self, String> {
        let g =
            Gguf::open(path).map_err(|e| format!("open drafter file {}: {e}", path.display()))?;
        let lay = layout(&g).ok_or_else(|| {
            format!(
                "{}: not a block drafter of this family (architecture '{}')",
                path.display(),
                g.arch()
            )
        })?;
        Ok(Self { g, lay })
    }

    pub fn layout(&self) -> Layout {
        self.lay
    }

    fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.g.get(name).map_err(|e| format!("drafter file: {e}"))
    }

    /// Every value of `t` as BF16 (a BF16 tensor's bytes as stored, any other type rounded to
    /// nearest even).
    fn all_bf16(&self, t: &TensorInfo) -> Result<Cow<'_, [u8]>, String> {
        if t.ty == GgmlType::BF16 {
            return Ok(Cow::Borrowed(self.g.tensor_data(t)));
        }
        let mut v = vec![0u16; t.n_elems as usize];
        rows_bf16_par(&self.g, t, 0, t.rows(), None, decode_threads(), &mut v)
            .map_err(|e| e.to_string())?;
        Ok(Cow::Owned(u16_le(&v)))
    }

    fn all_f32(&self, name: &str, len: u64) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?;
        if t.shape_row_major() != [len] {
            return Err(format!(
                "drafter tensor '{name}' has shape {:?}, expected [{len}]",
                t.shape_row_major()
            ));
        }
        let mut v = vec![0f32; len as usize];
        rows_f32(&self.g, t, 0, t.rows(), &mut v).map_err(|e| e.to_string())?;
        Ok(v)
    }

    /// Table `name` of rows of `cols` values stored with as many or fewer rows than the device
    /// tensor's `cap`, of which the drafter reads the first `need`: BF16, the rows past the
    /// file's zero.
    fn table(&self, name: &str, cols: u64, need: u64, cap: u64) -> Result<Vec<u8>, String> {
        let t = self.tensor(name)?;
        let shape = t.shape_row_major();
        let rows = match shape.as_slice() {
            [r, c] if *c == cols => *r,
            [r] if cols == 1 => *r,
            _ => {
                return Err(format!(
                    "drafter tensor '{name}' has shape {shape:?}, expected rows of {cols}"
                ))
            }
        };
        if !mtpf_rows_fit(rows, need, cap) {
            return Err(format!(
                "drafter tensor '{name}' has {rows} rows: the drafter reads {need}, the device tensor holds {cap}"
            ));
        }
        // the device tensor's bytes; `None`: its shape (from the model's config) overflows u64
        let bytes = ik_bf16_bytes(cap, cols)
            .and_then(|b| usize::try_from(b).ok())
            .ok_or_else(|| {
                format!("drafter tensor '{name}': a table of {cap} rows of {cols} values overflows")
            })?;
        let mut out = self.all_bf16(t)?.into_owned();
        out.resize(bytes, 0);
        Ok(out)
    }
}

impl MtpSource for GgufMtp {
    fn bf16(&self, name: &str, shape: &[u64]) -> Result<Cow<'_, [u8]>, String> {
        let gname = tensor_name(self.lay, name)
            .ok_or_else(|| format!("no drafter file tensor is known for '{name}'"))?;
        let t = self.tensor(&gname)?;
        if t.shape_row_major() == shape {
            return self.all_bf16(t);
        }
        // a token table without the padding rows (never read: no id reaches them)
        if let [cap, cols] = *shape {
            if name == "markov_head.markov_w1.weight" {
                let need = self.vocab_out()?;
                return Ok(Cow::Owned(self.table(&gname, cols, need, cap)?));
            }
        }
        Err(format!(
            "drafter tensor '{gname}' has shape {:?}, expected {shape:?}",
            t.shape_row_major()
        ))
    }

    fn stored(&self, name: &str, g: &MtpGeom) -> Result<Option<Vec<u8>>, String> {
        if self.lay != Layout::Folded {
            return Ok(None);
        }
        let c = &g.c;
        Ok(Some(match name {
            "mtw_alpha" => {
                let mut a = Vec::with_capacity(16);
                for l in 0..3 {
                    a.extend(self.all_f32(&format!("blk.{l}.ctx_mix.weight"), 4)?);
                }
                a.extend(self.all_f32("markov_ctx_mix.weight", 4)?);
                f32_bytes(&a)
            }
            "mtw_mask" => {
                let t = self.tensor("mask_embd.weight")?;
                if t.shape_row_major() != [c.block - 1, c.hid] {
                    return Err(format!(
                        "drafter tensor 'mask_embd.weight' has shape {:?}, expected [{}, {}]",
                        t.shape_row_major(),
                        c.block - 1,
                        c.hid
                    ));
                }
                self.all_bf16(t)?.into_owned()
            }
            "mtw_fold" => self.table("markov_w2.weight", c.rank, g.vocab_out, g.vocab_pad)?,
            "mtw_bf" => self.table("markov_w2.bias", 1, g.vocab_out, g.vocab_pad)?,
            _ => return Ok(None),
        }))
    }

    fn folds_on_card(&self) -> bool {
        self.lay == Layout::Plain
    }
}

impl GgufMtp {
    /// The vocabulary the drafter proposes from (ids it can read rows of).
    fn vocab_out(&self) -> Result<u64, String> {
        vocab_of(&self.g).ok_or_else(|| "drafter file: no vocabulary size and no token list".into())
    }
}

#[cfg(test)]
#[path = "inkling2_mtp_gguf_tests.rs"]
mod tests;
