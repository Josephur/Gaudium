//! The Gemma 4 image path on the card: its share of the engine's up-front plan, its weights,
//! the per-image launch chain into the request's feature buffer, and the request state that the
//! text prefill reads (feature column and keep flag of every prompt row, image blocks).
//!
//! Every buffer is sized from the configured largest image and request (`--vision-max-patches`,
//! `--vision-max-tokens`) and joins the engine's sealed plan: the tower's activations at its
//! largest patch count, two chain buffers, the input rows of one image. The feature buffer is the
//! decoder's own prefill tensor `vis_feat` (column 0 zero, the text rows' column; then one column
//! per image token of the request in prompt order, then room for the unified recipe's fixed rows).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use super::graphs::{
    build_block_graph, build_patch_graph, build_pool_graph, build_unified_graph, is_weight,
    BOUNDS, CHAIN_TENSORS, CLAMPED, LAYER_WEIGHTS,
};
use super::{merged_positions, tower_positions, tower_rope, G4vDims, G4vKind};
use crate::fp8::bf16_to_f32;
use crate::safetensors::SafetensorsStore;
use crate::synapse::device::{HpuDevice, PinnedBuf};
use crate::synapse::ffi::SynapseApi;
use crate::synapse::graph::{CompiledRecipe, GraphBuilder};
use crate::synapse::recipe_cache::RecipeCache;
use crate::verified::{
    g4_lane, ikv_feat_cols, ikv_feat_next, ikv_feat_row, ikv_feat_start, ikv_feat_write_safe,
    g4v_add_part, g4v_image_budget, g4v_pad_cols, g4v_recipe_rows, g4v_request_tokens, ikv_rows_match,
    vis_grid_fits, vis_rows_fit,
};
use crate::vision::preprocess::PreparedImage;
use crate::vision::{VisionReport, VisionSpec};

/// Pinned host stage for the per-image inputs (uploaded in pieces of this size).
const STAGE_BYTES: usize = 16 * 1024 * 1024;

/// Sizes the image path is planned for.
#[derive(Debug, Clone)]
pub struct G4vGeo {
    pub d: G4vDims,
    pub spec: VisionSpec,
    /// Tower: the largest image in patches. Unified: the rows of its one recipe (the soft tokens
    /// of the largest image).
    pub max_n: u64,
    /// Image tokens one request may hold.
    pub max_tokens: u64,
    /// Columns of the feature buffer.
    pub feat_cols: u64,
}

impl G4vGeo {
    /// The image path of the model at `cfg.model_dir` when the engine is loaded with image input
    /// (`cfg.vision_max_patches > 0` and a checkpoint with an image path), else `None`.
    pub fn plan(cfg: &crate::engine::EngineConfig) -> Result<Option<Self>, String> {
        if cfg.vision_max_patches == 0 {
            return Ok(None);
        }
        let cfg_text = std::fs::read_to_string(cfg.model_dir.join("config.json"))
            .map_err(|e| format!("read config.json: {e}"))?;
        if !super::is_gemma4_vision(&cfg_text) {
            return Ok(None);
        }
        let d = G4vDims::from_config_json(&cfg_text)?;
        let mut spec = VisionSpec::from_model_dir(&cfg.model_dir)?;
        let ppt = d.ppt();
        // the tower holds images of at most `--vision-max-patches` patches: the checkpoint's
        // budget, lowered to the whole tokens the plan holds
        let budget = g4v_image_budget(super::soft_budget(&spec), cfg.vision_max_patches, ppt);
        if budget == 0 {
            return Err(format!(
                "--vision-max-patches {} holds no soft token of {ppt} patches",
                cfg.vision_max_patches
            ));
        }
        let unified = d.kind == G4vKind::Unified;
        // the processor's pixel budget lowered to the image the plan holds (the server's rule)
        spec.max_pixels = crate::core::vision::image_pixel_cap(
            spec.max_pixels,
            g4v_recipe_rows(false, budget, ppt),
            d.patch,
        );
        let max_tokens = g4v_request_tokens(cfg.vision_max_tokens, budget);
        let (max_n, cap) = (
            g4v_recipe_rows(unified, budget, ppt),
            g4v_pad_cols(unified, budget),
        );
        let feat_cols = ikv_feat_cols(max_tokens, cap)
            .ok_or("the feature buffer's column count overflows")?;
        Ok(Some(Self {
            d,
            spec,
            max_n,
            max_tokens,
            feat_cols,
        }))
    }

    /// The image path's recipes for `n` (tower: patches of one image; unified: its fixed rows).
    pub fn graphs(&self, n: u64, elf: &[(String, String)]) -> Vec<GraphBuilder> {
        match self.d.kind {
            G4vKind::Tower => vec![
                build_patch_graph(&self.d, n, elf),
                build_block_graph(&self.d, n, elf),
                build_pool_graph(&self.d, n, elf),
            ],
            G4vKind::Unified => vec![build_unified_graph(&self.d, n, elf)],
        }
    }

    /// Exact device bytes of the image path, from its largest recipes.
    pub fn plan_bytes(&self) -> G4vPlan {
        let mut p = G4vPlan::default();
        for gb in self.graphs(self.max_n, &[]) {
            for (name, _dt, _dims, bytes) in gb.persistent_tensor_specs() {
                if CHAIN_TENSORS.contains(&name.as_str()) {
                    continue;
                }
                if is_weight(&name) {
                    if let Some(kind) = name.strip_prefix("vb_") {
                        for l in 0..self.d.layers {
                            p.weights.insert(format!("V{l}_{kind}"), bytes);
                        }
                    } else {
                        p.weights.insert(name, bytes);
                    }
                } else {
                    let e = p.acts.entry(name).or_insert(0);
                    *e = (*e).max(bytes);
                }
            }
        }
        p.x_bytes = match self.d.kind {
            G4vKind::Tower => self.d.hid * self.max_n * 2,
            G4vKind::Unified => 0,
        };
        p
    }
}

/// Device bytes of the image path (the feature buffer is the decoder's prefill tensor).
#[derive(Debug, Clone, Default)]
pub struct G4vPlan {
    pub weights: BTreeMap<String, u64>,
    pub acts: BTreeMap<String, u64>,
    /// Each of the two chain buffers of the tower's layers.
    pub x_bytes: u64,
}

impl G4vPlan {
    /// Arena bytes of the weights: every buffer as its sub-allocation takes it (`g4v_add_part`).
    pub fn weights_bytes(&self) -> u64 {
        self.weights.values().fold(0, |t, &b| g4v_add_part(t, b))
    }

    /// Arena bytes of the activations and the two chain buffers.
    pub fn act_bytes(&self) -> u64 {
        let acts = self.acts.values().fold(0, |t, &b| g4v_add_part(t, b));
        if self.x_bytes == 0 {
            return acts;
        }
        g4v_add_part(g4v_add_part(acts, self.x_bytes), self.x_bytes)
    }
}

// ============================================================================
// Weights
// ============================================================================

fn bf16_of<'a>(
    store: &'a SafetensorsStore,
    name: &str,
    count: u64,
) -> Result<std::borrow::Cow<'a, [u8]>, String> {
    let meta = store
        .get(name)
        .map_err(|e| format!("vision weight {name}: {e}"))?;
    if meta.dtype != "BF16" || meta.nbytes as u64 != count * 2 {
        return Err(format!(
            "vision weight {name}: {} of {} bytes, expected {count} BF16 values",
            meta.dtype, meta.nbytes
        ));
    }
    store
        .tensor_data(name)
        .map_err(|e| format!("vision weight {name}: {e}"))
}

fn bf16_vals(b: &[u8]) -> impl Iterator<Item = u16> + '_ {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]))
}

fn f32_bytes(v: impl Iterator<Item = f32>) -> Vec<u8> {
    v.flat_map(|x| x.to_le_bytes()).collect()
}

fn u16_bytes(v: impl Iterator<Item = u16>) -> Vec<u8> {
    v.flat_map(|x| x.to_le_bytes()).collect()
}

/// The checkpoint tensor behind a per-layer kind, and how many values it has.
fn layer_tensor(d: &G4vDims, kind: &str) -> Option<(&'static str, u64)> {
    let (h, dh, i) = (d.hid, d.dh, d.inter);
    Some(match kind {
        "in_w" => ("input_layernorm.weight", h),
        "pa_w" => ("post_attention_layernorm.weight", h),
        "pfin_w" => ("pre_feedforward_layernorm.weight", h),
        "pf_w" => ("post_feedforward_layernorm.weight", h),
        "qn_w" => ("self_attn.q_norm.weight", dh),
        "kn_w" => ("self_attn.k_norm.weight", dh),
        "q_W" => ("self_attn.q_proj.linear.weight", h * h),
        "k_W" => ("self_attn.k_proj.linear.weight", h * h),
        "v_W" => ("self_attn.v_proj.linear.weight", h * h),
        "o_W" => ("self_attn.o_proj.linear.weight", h * h),
        "gate_W" => ("mlp.gate_proj.linear.weight", i * h),
        "up_W" => ("mlp.up_proj.linear.weight", i * h),
        "down_W" => ("mlp.down_proj.linear.weight", h * i),
        _ => return None,
    })
}

fn proj_module(p: &str) -> Option<&'static str> {
    Some(match p {
        "q" => "self_attn.q_proj",
        "k" => "self_attn.k_proj",
        "v" => "self_attn.v_proj",
        "o" => "self_attn.o_proj",
        "gate" => "mlp.gate_proj",
        "up" => "mlp.up_proj",
        "down" => "mlp.down_proj",
        _ => return None,
    })
}

fn bound_field(b: &str) -> Option<&'static str> {
    Some(match b {
        "imin" => "input_min",
        "imax" => "input_max",
        "omin" => "output_min",
        "omax" => "output_max",
        _ => return None,
    })
}

/// f32 values of a BF16 vector in the converted-lane order of `gaudium_ga_lnorm` (within each 128
/// channels, lane `i < 64` is channel `2 i`, lane `64 + i` channel `2 i + 1`).
fn lanes(vals: &[u16]) -> Vec<f32> {
    let mut out = vec![0f32; vals.len()];
    for (c, &v) in vals.iter().enumerate() {
        let c = c as u64;
        out[((c / 128) * 128 + g4_lane(c)) as usize] = bf16_to_f32(v);
    }
    out
}

/// Device bytes of the image-path weight `name` (`expected` bytes, from the plan).
pub fn pack_weight(store: &SafetensorsStore, geo: &G4vGeo, name: &str) -> Result<Vec<u8>, String> {
    let d = &geo.d;
    let (h, dh) = (d.hid, d.dh);
    let one = crate::fp8::f32_to_bf16_rne(1.0);
    let vt = "model.vision_tower.";
    let ve = "model.vision_embedder.";
    let out = match name {
        "V_patch_W" => bf16_of(store, &format!("{vt}patch_embedder.input_proj.weight"), h * d.row_len())?.into_owned(),
        "V_proj_W" | "U_proj_W" => bf16_of(store, "model.embed_vision.embedding_projection.weight", d.out * h)?.into_owned(),
        "V_one" | "U_one" => u16_bytes(std::iter::repeat_n(one, h as usize)),
        "V_oned" => u16_bytes(std::iter::repeat_n(one, dh as usize)),
        "V_rot_W" => {
            // out[o] = -in[o + dh/4] in the first quarter of each half, +in[o - dh/4] in the second
            let (half, quart) = (dh / 2, dh / 4);
            let mut w = vec![0u16; (dh * dh) as usize];
            for o in 0..dh {
                let lo = (o % half) < quart;
                let i = if lo { o + quart } else { o - quart };
                w[(o * dh + i) as usize] = crate::fp8::f32_to_bf16_rne(if lo { -1.0 } else { 1.0 });
            }
            u16_bytes(w.into_iter())
        }
        "V_pool_W" => {
            let p2 = d.ppt();
            let mut w = vec![0u16; (h * p2 * h) as usize];
            for o in 0..h {
                for j in 0..p2 {
                    w[(o * (h * p2) + j * h + o) as usize] = one;
                }
            }
            u16_bytes(w.into_iter())
        }
        "V_pool_s" => f32_bytes(std::iter::once(1.0 / d.ppt() as f32)),
        "V_root" => f32_bytes(std::iter::once((h as f64).sqrt() as f32)),
        "V_std_nb" => f32_bytes(bf16_vals(&bf16_of(store, &format!("{vt}std_bias"), h)?).map(|v| -bf16_to_f32(v))),
        "V_std_s" => f32_bytes(bf16_vals(&bf16_of(store, &format!("{vt}std_scale"), h)?).map(bf16_to_f32)),
        "U_dense_W" => bf16_of(store, &format!("{ve}patch_dense.weight"), h * d.row_len())?.into_owned(),
        "U_dense_b" => f32_bytes(bf16_vals(&bf16_of(store, &format!("{ve}patch_dense.bias"), h)?).map(bf16_to_f32)),
        "U_ln1_g" | "U_ln1_b" | "U_ln2_g" | "U_ln2_b" | "U_ln3_g" | "U_ln3_b" => {
            let (src, c) = match &name[..6] {
                "U_ln1_" => ("patch_ln1", d.row_len()),
                "U_ln2_" => ("patch_ln2", h),
                _ => ("pos_norm", h),
            };
            let field = if name.ends_with("_g") { "weight" } else { "bias" };
            let v: Vec<u16> = bf16_vals(&bf16_of(store, &format!("{ve}{src}.{field}"), c)?).collect();
            f32_bytes(lanes(&v).into_iter())
        }
        _ => {
            // V<l>_<kind>: one encoder layer's weight or clamp bound
            let rest = name
                .strip_prefix('V')
                .ok_or_else(|| format!("no vision weight named {name}"))?;
            let us = rest.find('_').ok_or_else(|| format!("no vision weight named {name}"))?;
            let l: u64 = rest[..us]
                .parse()
                .map_err(|_| format!("no vision weight named {name}"))?;
            let kind = &rest[us + 1..];
            let base = format!("{vt}encoder.layers.{l}.");
            if let Some((src, count)) = layer_tensor(d, kind) {
                let raw = bf16_of(store, &format!("{base}{src}"), count)?;
                if kind == "qn_w" || kind == "kn_w" {
                    f32_bytes(bf16_vals(&raw).map(bf16_to_f32))
                } else {
                    raw.into_owned()
                }
            } else {
                let (p, b) = kind
                    .rsplit_once('_')
                    .ok_or_else(|| format!("no vision weight named {name}"))?;
                let (module, field) = proj_module(p)
                    .zip(bound_field(b))
                    .ok_or_else(|| format!("no vision weight named {name}"))?;
                let v = bf16_of(store, &format!("{base}{module}.{field}"), 1)?;
                let bound = u16::from_le_bytes([v[0], v[1]]);
                u16_bytes(std::iter::repeat_n(bound, 128))
            }
        }
    };
    Ok(out)
}

/// Every per-layer weight name the tower binds (the plan's map names them all).
pub fn layer_kinds(d: &G4vDims) -> Vec<String> {
    let mut v: Vec<String> = LAYER_WEIGHTS.iter().map(|s| s.to_string()).collect();
    if d.clipped {
        for p in CLAMPED {
            for b in BOUNDS {
                v.push(format!("{p}_{b}"));
            }
        }
    }
    v
}

// ============================================================================
// Runtime
// ============================================================================

/// An image request between `set_request_images` and the end of its prefill.
#[derive(Debug, Clone, Default)]
pub struct VisRequest {
    /// Feature column of every prompt row (0: a text row).
    pub fidx: Vec<i32>,
    /// Image blocks of the prompt: `(first row, rows)`, in order.
    pub blocks: Vec<(usize, usize)>,
}

pub struct G4Vision {
    pub geo: G4vGeo,
    elf: Vec<(String, String)>,
    /// Recipes per size (tower: patch, block, pool; unified: the one recipe).
    recipes: HashMap<u64, Vec<CompiledRecipe>>,
    pub w_addr: BTreeMap<String, u64>,
    pub act_addr: BTreeMap<String, u64>,
    pub xbuf: [u64; 2],
    /// The feature buffer (the decoder's `vis_feat`), `out x feat_cols` BF16.
    pub feat: u64,
    /// Position table kept on the host (a patch's two rows are summed there).
    pos_table: Vec<u16>,
    stage: Option<PinnedBuf>,
    pending: Vec<PreparedImage>,
    pub req: Option<VisRequest>,
    report: Option<VisionReport>,
    last_tokens: u64,
}

impl G4Vision {
    pub fn new(geo: G4vGeo, elf: Vec<(String, String)>) -> Self {
        Self {
            geo,
            elf,
            recipes: HashMap::new(),
            w_addr: BTreeMap::new(),
            act_addr: BTreeMap::new(),
            xbuf: [0; 2],
            feat: 0,
            pos_table: Vec::new(),
            stage: None,
            pending: Vec::new(),
            req: None,
            report: None,
            last_tokens: 0,
        }
    }

    pub fn spec(&self) -> &VisionSpec {
        &self.geo.spec
    }

    pub fn elf(&self) -> &[(String, String)] {
        &self.elf
    }

    /// Compiles the recipes of the largest planned size before the workspace is sized (returns
    /// their largest workspace).
    pub fn compile_planned(
        &mut self,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver: &str,
    ) -> Result<u64, String> {
        let n = self.geo.max_n;
        let mut v = Vec::new();
        for gb in self.geo.graphs(n, &self.elf) {
            v.push(
                gb.compile(Arc::clone(api), cache, driver)
                    .map_err(|e| format!("compile vision recipe for {n}: {e}"))?,
            );
        }
        let ws = v.iter().map(|r| r.ws_size).max().unwrap_or(0);
        self.recipes.insert(n, v);
        Ok(ws)
    }

    /// Binds the precompiled recipes to the sealed shared workspace.
    pub fn bind_workspaces(&mut self, dev: &HpuDevice) -> Result<(), String> {
        for rs in self.recipes.values_mut() {
            for r in rs.iter_mut() {
                r.bind_shared_workspace(dev)
                    .map_err(|e| format!("bind {} ws: {e}", r.name))?;
            }
        }
        Ok(())
    }

    /// Assigns the plan's buffers (allocated by the engine) and keeps the host tables.
    pub fn attach(
        &mut self,
        store: &SafetensorsStore,
        dev: &HpuDevice,
        w_addr: BTreeMap<String, u64>,
        act_addr: BTreeMap<String, u64>,
        xbuf: [u64; 2],
        feat: u64,
    ) -> Result<(), String> {
        let d = &self.geo.d;
        let (table, count) = match d.kind {
            G4vKind::Tower => (
                "model.vision_tower.patch_embedder.position_embedding_table",
                2 * d.npos * d.hid,
            ),
            G4vKind::Unified => ("model.vision_embedder.pos_embedding", d.npos * 2 * d.hid),
        };
        self.pos_table = bf16_vals(&bf16_of(store, table, count)?).collect();
        self.w_addr = w_addr;
        self.act_addr = act_addr;
        self.xbuf = xbuf;
        self.feat = feat;
        self.stage = Some(
            dev.alloc_pinned(STAGE_BYTES)
                .map_err(|e| format!("alloc vision stage: {e}"))?,
        );
        // column 0 of the feature buffer is what text rows read: zero, and never written again
        let row = self.geo.d.out * 2;
        dev.copy_h2d_sync(&vec![0u8; row as usize], feat)
            .map_err(|e| format!("zero the feature buffer's text column: {e}"))?;
        Ok(())
    }

    /// Takes the images of the next request, in prompt order.
    pub fn set_request_images(&mut self, images: Vec<PreparedImage>) -> Result<(), String> {
        let tokens: u64 = images.iter().map(|i| i.n_tokens() as u64).sum();
        if !vis_rows_fit(0, tokens, self.geo.max_tokens) {
            return Err(format!(
                "the request's images hold {tokens} image tokens; this engine is planned for {} (--vision-max-tokens)",
                self.geo.max_tokens
            ));
        }
        let max_patches = match self.geo.d.kind {
            G4vKind::Tower => self.geo.max_n,
            G4vKind::Unified => self.geo.max_n * self.geo.d.ppt(),
        };
        for im in &images {
            if !vis_grid_fits(im.grid_h as u64, im.grid_w as u64, max_patches) || im.pixels.is_empty() {
                return Err(format!(
                    "an image of {} patches: the image path is planned for {max_patches}",
                    im.n_patches()
                ));
            }
        }
        self.pending = images;
        Ok(())
    }

    pub fn last_report(&self) -> Option<VisionReport> {
        self.report.clone()
    }

    /// Lays the request's images over its prompt: the pad rows must be the images' tokens in
    /// order (one run per image). Takes the pending images (also when it fails).
    pub fn begin_request(&mut self, prompt_ids: &[i32]) -> Result<Vec<PreparedImage>, String> {
        let images = std::mem::take(&mut self.pending);
        self.req = None;
        self.report = None;
        let pad = self.geo.spec.image_token_id;
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, &t) in prompt_ids.iter().enumerate() {
            if t == pad {
                match runs.last_mut() {
                    Some((s, n)) if *s + *n == i => *n += 1,
                    _ => runs.push((i, 1)),
                }
            }
        }
        if images.is_empty() {
            if !runs.is_empty() {
                return Err("the prompt holds image placeholders but the request has no image".into());
            }
            return Ok(images);
        }
        let rows: usize = runs.iter().map(|r| r.1).sum();
        let feats: u64 = images.iter().map(|i| i.n_tokens() as u64).sum();
        if !ikv_rows_match(rows as u64, feats) || runs.len() != images.len() {
            return Err(format!(
                "the prompt has {} image placeholder runs of {rows} rows for {} images of {feats} tokens",
                runs.len(),
                images.len()
            ));
        }
        for (k, (r, im)) in runs.iter().zip(&images).enumerate() {
            if !ikv_rows_match(r.1 as u64, im.n_tokens() as u64) {
                return Err(format!(
                    "image {k}: {} placeholder rows for {} tokens",
                    r.1,
                    im.n_tokens()
                ));
            }
        }
        let mut fidx = vec![ikv_feat_row(false, 0) as i32; prompt_ids.len()];
        let mut rank = 0u64;
        for &(s, n) in &runs {
            for row in fidx.iter_mut().skip(s).take(n) {
                *row = ikv_feat_row(true, rank) as i32;
                rank += 1;
            }
        }
        self.req = Some(VisRequest { fidx, blocks: runs });
        Ok(images)
    }

    /// Runs the image path over the request's images into the feature buffer (synced).
    pub fn encode(
        &mut self,
        dev: &HpuDevice,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver: &str,
        images: &[PreparedImage],
    ) -> Result<(), String> {
        let mut rep = VisionReport {
            images: images.len(),
            ..VisionReport::default()
        };
        let mut off = ikv_feat_start();
        for im in images {
            let n = match self.geo.d.kind {
                G4vKind::Tower => im.n_patches() as u64,
                G4vKind::Unified => self.geo.max_n,
            };
            let t0 = Instant::now();
            if !self.recipes.contains_key(&n) {
                let mut v = Vec::new();
                for gb in self.geo.graphs(n, &self.elf) {
                    let mut r = gb
                        .compile(Arc::clone(api), cache, driver)
                        .map_err(|e| format!("compile vision recipe for {n}: {e}"))?;
                    if r.ws_size > dev.shared_workspace().1 {
                        return Err(format!(
                            "vision recipe {} needs a workspace of {} bytes, more than the planned {}",
                            r.name,
                            r.ws_size,
                            dev.shared_workspace().1
                        ));
                    }
                    r.bind_shared_workspace(dev)
                        .map_err(|e| format!("bind {} ws: {e}", r.name))?;
                    v.push(r);
                }
                self.recipes.insert(n, v);
            }
            rep.compile_ms += t0.elapsed().as_secs_f64() * 1000.0;
            let tokens = im.n_tokens() as u64;
            let cols = match self.geo.d.kind {
                G4vKind::Tower => tokens,
                G4vKind::Unified => self.geo.max_n,
            };
            if !ikv_feat_write_safe(off, cols, self.geo.feat_cols) {
                return Err(format!(
                    "image features at column {off} (+{cols}) leave the feature buffer of {} columns",
                    self.geo.feat_cols
                ));
            }
            let t1 = Instant::now();
            self.upload_inputs(dev, im)?;
            rep.prep_ms += t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            self.launch_chain(dev, n, off)?;
            // SAFETY: the compute stream is owned by `dev` and alive for its lifetime.
            unsafe { dev.sync_stream(dev.compute_stream()) }
                .map_err(|e| format!("sync vision: {e}"))?;
            rep.encode_ms += t2.elapsed().as_secs_f64() * 1000.0;
            rep.patches += im.n_patches() as u64;
            rep.image_tokens += tokens;
            off = ikv_feat_next(off, tokens);
        }
        self.last_tokens = rep.image_tokens;
        self.report = Some(rep);
        Ok(())
    }

    fn upload(&mut self, dev: &HpuDevice, name: &str, bytes: &[u8]) -> Result<(), String> {
        let addr = *self
            .act_addr
            .get(name)
            .ok_or_else(|| format!("no vision buffer {name}"))?;
        let stage = self.stage.as_mut().ok_or("vision stage missing")?;
        dev.upload_h2d_chunked(bytes, addr, stage)
            .map_err(|e| format!("upload {name}: {e}"))
    }

    fn upload_inputs(&mut self, dev: &HpuDevice, im: &PreparedImage) -> Result<(), String> {
        let d = self.geo.d;
        let px: Vec<u8> = im.pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
        match d.kind {
            G4vKind::Tower => {
                let pos = tower_positions(
                    &self.pos_table,
                    d.npos as usize,
                    d.hid as usize,
                    im.grid_h,
                    im.grid_w,
                    d.pool as usize,
                )?;
                let (cs, sn) = tower_rope(&d, im.grid_h, im.grid_w);
                self.upload(dev, "v_pix", &px)?;
                self.upload(dev, "v_pos", &u16_bytes(pos.into_iter()))?;
                self.upload(dev, "v_cos", &u16_bytes(cs.into_iter()))?;
                self.upload(dev, "v_sin", &u16_bytes(sn.into_iter()))?;
            }
            G4vKind::Unified => {
                let pool = d.pool as usize;
                let (mh, mw) = (im.grid_h / pool, im.grid_w / pool);
                let pos = merged_positions(&self.pos_table, d.npos as usize, d.hid as usize, mh, mw)?;
                // the fixed rows of the recipe: the image's rows, zero rows behind them
                let r = self.geo.max_n as usize;
                let mut pxr = vec![0u8; r * d.row_len() as usize * 2];
                pxr[..px.len()].copy_from_slice(&px);
                let pb = u16_bytes(pos.into_iter());
                let mut posr = vec![0u8; r * d.hid as usize * 2];
                posr[..pb.len()].copy_from_slice(&pb);
                self.upload(dev, "u_px", &pxr)?;
                self.upload(dev, "u_pos", &posr)?;
            }
        }
        Ok(())
    }

    fn pool_for(&self, extra: &[(&str, u64)]) -> BTreeMap<String, u64> {
        let mut pool = self.act_addr.clone();
        for (k, &v) in &self.w_addr {
            pool.insert(k.clone(), v);
        }
        for &(k, v) in extra {
            pool.insert(k.to_string(), v);
        }
        pool
    }

    fn launch(&self, dev: &HpuDevice, r: &CompiledRecipe, pool: &BTreeMap<String, u64>) -> Result<(), String> {
        let bind = crate::engine::gemma4::select_exact_bindings(r, pool)?;
        r.launch_checked(dev, &bind)
            .map_err(|e| format!("launch {}: {e}", r.name))
    }

    /// The image's launches into the feature buffer at column `off`.
    fn launch_chain(&self, dev: &HpuDevice, n: u64, off: u64) -> Result<(), String> {
        let rs = &self.recipes[&n];
        let feat = self.feat + off * self.geo.d.out * 2;
        match self.geo.d.kind {
            G4vKind::Unified => {
                let pool = self.pool_for(&[("v_feat", feat)]);
                self.launch(dev, &rs[0], &pool)
            }
            G4vKind::Tower => {
                let d = &self.geo.d;
                let pool = self.pool_for(&[("v_x0", self.xbuf[0])]);
                self.launch(dev, &rs[0], &pool)?;
                let kinds = layer_kinds(d);
                for l in 0..d.layers as usize {
                    let mut pool = self.pool_for(&[
                        ("vb_x_in", self.xbuf[l % 2]),
                        ("vb_x_out", self.xbuf[(l + 1) % 2]),
                    ]);
                    for k in &kinds {
                        let a = *self
                            .w_addr
                            .get(&format!("V{l}_{k}"))
                            .ok_or_else(|| format!("no tower weight V{l}_{k}"))?;
                        pool.insert(format!("vb_{k}"), a);
                    }
                    self.launch(dev, &rs[1], &pool)?;
                }
                let pool = self.pool_for(&[
                    ("vp_x_in", self.xbuf[d.layers as usize % 2]),
                    ("v_feat", feat),
                ]);
                self.launch(dev, &rs[2], &pool)
            }
        }
    }

    /// The soft tokens of the last image request (`tokens x out` BF16), read back from HBM.
    pub fn read_features(&self, dev: &HpuDevice) -> Result<Vec<u8>, String> {
        if self.last_tokens == 0 {
            return Err("no image request has run on this engine".into());
        }
        let row = self.geo.d.out * 2;
        let mut out = vec![0u8; (self.last_tokens * row) as usize];
        dev.copy_d2h_sync(self.feat + row, &mut out)
            .map_err(|e| format!("read image features: {e}"))?;
        Ok(out)
    }
}
