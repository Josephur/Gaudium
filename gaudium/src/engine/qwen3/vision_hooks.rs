//! Image-input hooks of `Qwen3Engine` (kept apart from the text engine).
//!
//! A request with images does three extra things, all before the first prefill tile:
//! 1. lays the prompt out (image-pad spans, M-RoPE positions),
//! 2. rewrites the decoder's slot-indexed RoPE rows from the first image slot on,
//! 3. runs the vision tower per image, leaving the merged embeddings in HBM (an image whose rows
//!    all lie inside a prefix reused from the previous request is skipped: `engine/prefix_cache.rs`).
//!
//! During prefill, each tile's image-pad rows are overwritten on the device with those
//! embeddings right after the tile's embedding gather. Decode needs nothing: it reads the RoPE
//! table by KV slot. A text-only request only restores rows a previous image request changed.
//!
//! For speculative methods: `last_prompt_layout()` gives the per-row position table and the
//! image span list (lookup must not match inside a span; positions past the prompt are
//! `slot + delta`), `vision_embeddings_addr()` the HBM buffer drafters must read image-slot
//! embeddings from (row `ImageSpan::emb_row + offset`), never the image-pad token embedding.

use std::time::Instant;

use super::{Qwen3Config, Qwen3Engine};
use crate::model_spec::ModelSpec;
use crate::vision::mrope::layout_prompt;
use crate::vision::vit::{VisionTower, ROPE_ROW_MARGIN};
use crate::vision::{PreparedImage, PromptLayout, VisionReport, VisionSpec};

/// Request-scoped vision state of the engine.
#[derive(Default)]
pub(super) struct VisionRequestState {
    /// Images of the next `generate` call (taken by it).
    pending: Vec<PreparedImage>,
    last_report: Option<VisionReport>,
    last_layout: Option<PromptLayout>,
}

/// Plans the vision tower when the configuration asks for image input.
pub(super) fn plan_vision(
    spec: &ModelSpec,
    cfg: &Qwen3Config,
) -> Result<Option<VisionTower>, String> {
    let max_pre = cfg.vision_precompile.iter().copied().max().unwrap_or(0);
    let max_patches = cfg.vision_max_patches.max(max_pre);
    if max_patches == 0 {
        return Ok(None);
    }
    if !spec.has_vision {
        return Err(
            "image input was requested but this checkpoint has no vision tower (config.json has no vision_config)".into(),
        );
    }
    let mut vs = VisionSpec::from_model_dir(&cfg.model_dir)?;
    vs.gelu_bf16 = cfg.vision_gelu_bf16;
    VisionTower::plan(
        vs,
        spec.hidden,
        max_patches,
        cfg.vision_max_tokens,
        &cfg.vision_precompile,
    )
    .map(Some)
}

impl Qwen3Engine {
    /// Vision geometry of the loaded tower (`None`: the engine was loaded without image input).
    pub fn vision_spec(&self) -> Option<&VisionSpec> {
        self.vision.as_ref().map(VisionTower::spec)
    }

    /// Largest image (patches) and most image tokens per request the tower was planned for.
    pub fn vision_limits(&self) -> Option<(u64, u64)> {
        self.vision
            .as_ref()
            .map(|v| (v.max_patches(), v.max_tokens()))
    }

    /// Hands the images of the NEXT `generate` call to the engine, in prompt order. The prompt
    /// must hold one expanded image-pad run per image (`vision::mrope::expand_image_pads`).
    pub fn set_request_images(&mut self, images: Vec<PreparedImage>) -> Result<(), String> {
        let Some(tower) = self.vision.as_ref() else {
            return Err(
                "this engine was loaded without image input (vision_max_patches = 0)".into(),
            );
        };
        for (k, img) in images.iter().enumerate() {
            if img.n_patches() as u64 > tower.max_patches() {
                return Err(format!(
                    "image {k} has {} patches; the vision tower was planned for {} (--vision-max-patches)",
                    img.n_patches(),
                    tower.max_patches()
                ));
            }
        }
        self.vision_req.pending = images;
        Ok(())
    }

    /// Vision timings of the last request (`None` if it had no images).
    pub fn last_vision_report(&self) -> Option<&VisionReport> {
        self.vision_req.last_report.as_ref()
    }

    /// Positions and image spans of the last image request.
    pub fn last_prompt_layout(&self) -> Option<&PromptLayout> {
        self.vision_req.last_layout.as_ref()
    }

    /// HBM address of the image embeddings `{HID, tokens}` BF16 of the current request.
    pub fn vision_embeddings_addr(&self) -> Option<u64> {
        self.vision.as_ref().map(VisionTower::embeddings_addr)
    }

    /// Reads the image embeddings of the last image request back from HBM: `tokens x HID`
    /// BF16 values, rows in prompt order (validation against a reference tower). The rows of an
    /// image that a resumed request did not encode again hold what an earlier request left.
    pub fn read_vision_embeddings(&self) -> Result<Vec<u8>, String> {
        let (Some(tower), Some(layout)) = (self.vision.as_ref(), self.last_prompt_layout()) else {
            return Err("no image request has run on this engine".into());
        };
        let mut out = vec![0u8; layout.image_tokens() * (self.spec.hidden as usize) * 2];
        self.dev
            .copy_d2h_sync(tower.embeddings_addr(), &mut out)
            .map_err(|e| format!("read image embeddings D2H: {e}"))?;
        Ok(out)
    }

    /// Start of a request, before its prefix plan: takes the images handed over and lays the
    /// prompt out (image-pad spans, M-RoPE positions). `None`: a text-only request, or an
    /// engine without image input. Nothing touches the device here.
    pub(super) fn vision_prepare(
        &mut self,
        prompt_ids: &[i32],
    ) -> Result<Option<(Vec<PreparedImage>, PromptLayout)>, String> {
        let images = std::mem::take(&mut self.vision_req.pending);
        self.vision_req.last_report = None;
        self.vision_req.last_layout = None;
        let Some(tower) = self.vision.as_ref() else {
            return Ok(None);
        };
        if images.is_empty() {
            return Ok(None);
        }
        let grids: Vec<(usize, usize)> = images.iter().map(|i| (i.grid_h, i.grid_w)).collect();
        let layout = layout_prompt(prompt_ids, &grids, tower.spec())?;
        if layout.image_tokens() as u64 > tower.max_tokens() {
            return Err(format!(
                "the prompt holds {} image tokens; the vision embedding buffer was planned for {}",
                layout.image_tokens(),
                tower.max_tokens()
            ));
        }
        Ok(Some((images, layout)))
    }

    /// Start of a request, after its prefix plan: installs the request's RoPE rows and runs the
    /// tower when the request has images, otherwise restores rows left by an earlier image
    /// request. `resume`: the prefill starts at this row; an image whose rows all lie before it
    /// is not encoded again (no tile of this request reads its embeddings: its rows are among
    /// the reused ones, which the verified rule only allows for the same picture at the same
    /// place). Everything is synchronized on return (success or error).
    pub(super) fn vision_begin_request(
        &mut self,
        prepared: Option<(Vec<PreparedImage>, PromptLayout)>,
        prompt_len: usize,
        max_new: u64,
        resume: usize,
    ) -> Result<Option<PromptLayout>, String> {
        let Some(tower) = self.vision.as_mut() else {
            return Ok(None);
        };
        let rope_c = self.weights_addr["rope_c"];
        let rope_s = self.weights_addr["rope_s"];
        let theta = self.spec.rope_theta;
        let rot = self.spec.rotary_dim as usize;
        let Some((images, layout)) = prepared else {
            tower.apply_request_rope(&self.dev, rope_c, rope_s, theta, rot, None, 0)?;
            return Ok(None);
        };

        let dev = &self.dev;
        let (cache, drv) = (&self.cache, self.driver_version.as_str());
        let max_ctx = self.max_ctx as usize;
        let run = |tower: &mut VisionTower| -> Result<VisionReport, String> {
            let t_rope = Instant::now();
            let row_hi = (prompt_len + max_new as usize + ROPE_ROW_MARGIN).min(max_ctx);
            let rope_rows =
                tower.apply_request_rope(dev, rope_c, rope_s, theta, rot, Some(&layout), row_hi)?;
            let rope_ms = t_rope.elapsed().as_secs_f64() * 1000.0;

            let compile0 = tower.compile_ms();
            let t_enc = Instant::now();
            let mut prep_ms = 0.0;
            let mut reused_images = 0usize;
            for (img, span) in images.iter().zip(&layout.spans) {
                if span.end() <= resume {
                    reused_images += 1;
                    continue;
                }
                prep_ms += tower.encode_image(dev, cache, drv, img, span.emb_row)?;
            }
            tower.sync(dev)?;
            let compile_ms = tower.compile_ms() - compile0;
            let total_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
            Ok(VisionReport {
                images: images.len(),
                reused_images,
                patches: images.iter().map(|i| i.n_patches() as u64).sum(),
                image_tokens: layout.image_tokens() as u64,
                prep_ms,
                encode_ms: (total_ms - prep_ms - compile_ms).max(0.0),
                compile_ms,
                rope_ms,
                rope_rows,
                rope_delta: layout.delta,
            })
        };
        match run(tower) {
            Ok(report) => {
                self.vision_req.last_report = Some(report);
                self.vision_req.last_layout = Some(layout.clone());
                Ok(Some(layout))
            }
            Err(e) => {
                // never return with tower launches queued
                let _ = tower.sync(dev);
                Err(e)
            }
        }
    }

    /// Overwrites the image-pad rows of the prefill tile `[pos, pos + tile_len)` at `x0_addr`
    /// (`{HID, T}` BF16, feature-fastest, so a span is one contiguous run) with the tower's
    /// embeddings. Enqueued on the compute stream right after the tile's embedding gather.
    pub(super) fn vision_splice_tile(
        &self,
        layout: &PromptLayout,
        pos: usize,
        tile_len: usize,
        x0_addr: u64,
    ) -> Result<(), String> {
        let Some(tower) = self.vision.as_ref() else {
            return Ok(());
        };
        let row_bytes = self.spec.hidden * 2;
        for (emb_row, slot, n) in layout.overlaps(pos, pos + tile_len) {
            let src = tower.embeddings_addr() + (emb_row as u64) * row_bytes;
            let dst = x0_addr + ((slot - pos) as u64) * row_bytes;
            if let Err(e) = self.dev.copy_d2d_compute(src, (n as u64) * row_bytes, dst) {
                // the tile's embedding gather is already queued: drain before reporting
                // SAFETY: the compute stream belongs to `self.dev`.
                let _ = unsafe { self.dev.sync_stream(self.dev.compute_stream()) };
                return Err(format!(
                    "copy image embeddings into prefill tile at slot {slot}: {e}"
                ));
            }
        }
        Ok(())
    }
}
