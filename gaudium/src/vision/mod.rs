//! Image and video input for the Qwen3.5/3.8 vision-language family on Intel Gaudi 2, and the
//! model-independent parts (decoders, request media, budgets) that the Inkling 2 family's tower
//! (`engine::inkling2::vision`, `ImageProcessor::Tiles`) shares.
//!
//! - `decode`: the crate's own image file decoders (JPEG, PNG, GIF, WebP, BMP), bit-exact with
//!   the reference image processor's decoder; animations as frames.
//! - `media`: the media of one request (images, animations, frame lists) from encoded bytes to
//!   tower inputs: limits, video frame sampling and timing, prompt placeholders.
//! - `preprocess`: host image preparation (RGB8 in, smart resize, antialiased bicubic
//!   resampling, normalisation, patch rows in 2x2 merge order, BF16), for stills and for the
//!   temporal patches of a video.
//! - `mrope`: prompt layout (image-pad spans, interleaved M-RoPE positions) and the slot-indexed
//!   RoPE table rows of a vision request. RoPE positions and KV slots diverge after an image.
//! - `packer`: vision tower weights (BF16 / F32, `rotate_half` folded into extra q / k rows),
//!   bilinear position embeddings and the tower's 2-D RoPE tables.
//! - `vit`: the tower's three recipes per patch count (patch embedding, one generic block bound
//!   to each block's weights, merger), its up-front HBM plan, and the per-image launch chain.
//!
//! The decoder is untouched: image-pad rows of a prefill tile are overwritten on the device with
//! the merger's output, and the request's RoPE rows are rewritten so that the table stays
//! indexed by KV slot. Text-only requests launch exactly what they launched before.

use std::path::Path;

use crate::json::{parse_json, JsonValue};

pub mod cli;
pub mod decode;
pub mod media;
pub mod mrope;
pub mod packer;
pub mod preprocess;
pub mod vit;

pub use mrope::{ImageSpan, PromptLayout};
pub use preprocess::{PreparedImage, RgbImage};

/// How a checkpoint's image processor turns a picture into tower inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageProcessor {
    /// Resized to a pixel budget (`preprocess::smart_resize`), temporal frames per patch row,
    /// patch rows laid out channel, frame, y, x (the Qwen3.5 / 3.8 family).
    Resize,
    /// Never resized: padded at the bottom and right to whole `patch * merge` tiles, one token
    /// per tile, patch rows laid out y, x, channel (the Inkling 2 family,
    /// `engine::inkling2::vision`).
    Tiles,
}

/// Vision tower geometry, special token IDs and image-processor settings of one checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionSpec {
    pub processor: ImageProcessor,
    /// The chat template's text for one image (its pad is expanded to the image's tokens).
    pub image_block: String,
    pub depth: u64,
    pub hidden: u64,
    pub n_heads: u64,
    pub intermediate: u64,
    pub in_channels: u64,
    pub patch: u64,
    pub temporal: u64,
    pub merge: u64,
    /// Decoder embedding width the merger projects to.
    pub out_hidden: u64,
    /// Rows of the learned position table (a square grid of `pos_side() ^ 2` rows).
    pub n_pos: u64,
    /// `gelu_pytorch_tanh` in the blocks' MLP (the merger always uses the exact form).
    pub gelu_tanh: bool,
    /// Evaluate GELU with the BF16 kernel instead of in F32 (a few percent less tower time,
    /// about a percent of accuracy per block; changes the compiled graphs). Off by default.
    pub gelu_bf16: bool,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_id: i32,
    pub vision_end_id: i32,
    /// Interleaved M-RoPE frequency counts of the (t, h, w) sections.
    pub mrope_section: [u64; 3],
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub image_mean: [f64; 3],
    pub image_std: [f64; 3],
    /// The checkpoint's video processor settings; `None`: it takes no video.
    pub video: Option<VideoSpec>,
}

/// Video processor settings of a checkpoint (`video_preprocessor_config.json`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoSpec {
    /// Pixel budget of a whole video: frames x height x width of the resized frames.
    pub min_pixels: u64,
    pub max_pixels: u64,
    /// Frames per second of video time the model is shown.
    pub fps: f64,
    /// Fewest and most frames kept of one video (never more than it has).
    pub min_frames: u64,
    pub max_frames: u64,
}

impl VideoSpec {
    /// Reads `video_preprocessor_config.json`; a missing key keeps the processor's own default.
    pub fn from_json(text: &str, vs: &VisionSpec) -> Result<Self, String> {
        let p = parse_json(text).map_err(|e| format!("video_preprocessor_config.json: {e}"))?;
        let mut v = Self {
            min_pixels: 128 * 32 * 32,
            max_pixels: 32 * 32 * 768,
            fps: 2.0,
            min_frames: 4,
            max_frames: 768,
        };
        if let Some(size) = p.get("size") {
            if let Some(x) = obj_u64(size, "shortest_edge") {
                v.min_pixels = x;
            }
            if let Some(x) = obj_u64(size, "longest_edge") {
                v.max_pixels = x;
            }
        }
        if let Some(x) = obj_u64(&p, "min_pixels") {
            v.min_pixels = x;
        }
        if let Some(x) = obj_u64(&p, "max_pixels") {
            v.max_pixels = x;
        }
        if let Some(x) = p.get("fps").and_then(JsonValue::as_f64) {
            v.fps = x;
        }
        if let Some(x) = obj_u64(&p, "min_frames") {
            v.min_frames = x;
        }
        if let Some(x) = obj_u64(&p, "max_frames") {
            v.max_frames = x;
        }
        for (key, model) in [
            ("patch_size", vs.patch),
            ("temporal_patch_size", vs.temporal),
            ("merge_size", vs.merge),
        ] {
            if let Some(x) = obj_u64(&p, key) {
                if x != model {
                    return Err(format!(
                        "video_preprocessor_config.json {key}={x} disagrees with vision_config ({model})"
                    ));
                }
            }
        }
        for (key, model) in [("image_mean", vs.image_mean), ("image_std", vs.image_std)] {
            if let Some(x) = obj_f64x3(&p, key) {
                if x != model {
                    return Err(format!(
                        "video_preprocessor_config.json {key} differs from the image processor's"
                    ));
                }
            }
        }
        if !(v.fps.is_finite() && v.fps > 0.0) || v.max_frames == 0 || v.max_pixels == 0 {
            return Err(
                "video_preprocessor_config.json: fps, max_frames and the pixel budget must be positive".into(),
            );
        }
        Ok(v)
    }
}

fn obj_u64(o: &JsonValue, key: &str) -> Option<u64> {
    o.get(key).and_then(JsonValue::as_u64)
}

fn obj_f64x3(o: &JsonValue, key: &str) -> Option<[f64; 3]> {
    let a = o.get(key)?.as_array()?;
    if a.len() != 3 {
        return None;
    }
    Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?])
}

impl VisionSpec {
    /// Reads `config.json` (required), `preprocessor_config.json` (optional) and
    /// `video_preprocessor_config.json` (optional: without it, or without a video pad token,
    /// the checkpoint takes no video) of a model dir.
    pub fn from_model_dir(dir: &Path) -> Result<Self, String> {
        let cfg = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("read config.json: {e}"))?;
        let pre = std::fs::read_to_string(dir.join("preprocessor_config.json")).ok();
        if crate::engine::inkling2::vision::is_inkling2_vision(&cfg) {
            let proc = std::fs::read_to_string(dir.join("processor_config.json")).ok();
            return crate::engine::inkling2::vision::vision_spec(&cfg, proc.as_deref());
        }
        let mut s = Self::from_json(&cfg, pre.as_deref())?;
        if s.video_token_id >= 0 {
            if let Ok(text) = std::fs::read_to_string(dir.join("video_preprocessor_config.json")) {
                s.video = Some(VideoSpec::from_json(&text, &s)?);
            }
        }
        Ok(s)
    }

    pub fn from_json(config_json: &str, preprocessor_json: Option<&str>) -> Result<Self, String> {
        if crate::engine::inkling2::vision::is_inkling2_vision(config_json) {
            return crate::engine::inkling2::vision::vision_spec(config_json, preprocessor_json);
        }
        let root = parse_json(config_json).map_err(|e| format!("config.json: {e}"))?;
        let vc = root
            .get("vision_config")
            .filter(|v| v.as_object().is_some())
            .ok_or("config.json has no vision_config: this checkpoint takes no images")?;
        let need = |key: &str| -> Result<u64, String> {
            obj_u64(vc, key).ok_or_else(|| format!("vision_config.{key} is missing"))
        };
        let tok = |key: &str| -> Result<i32, String> {
            root.get(key)
                .and_then(JsonValue::as_i64)
                .map(|v| v as i32)
                .ok_or_else(|| format!("config.json: {key} is missing"))
        };
        if let Some(ds) = vc
            .get("deepstack_visual_indexes")
            .and_then(JsonValue::as_array)
        {
            if !ds.is_empty() {
                return Err(
                    "vision_config.deepstack_visual_indexes is not empty: DeepStack injection into decoder layers is not implemented".into(),
                );
            }
        }
        let text = root.get("text_config").unwrap_or(&root);
        let mut mrope_section = [0u64; 3];
        let sec = text
            .get("rope_parameters")
            .and_then(|r| r.get("mrope_section"))
            .and_then(JsonValue::as_array);
        match sec {
            Some(a) if a.len() == 3 => {
                for (d, v) in mrope_section.iter_mut().zip(a) {
                    *d = v
                        .as_u64()
                        .ok_or("rope_parameters.mrope_section: not integers")?;
                }
            }
            _ => {
                return Err(
                    "text_config.rope_parameters.mrope_section (3 entries) is missing".into(),
                )
            }
        }

        let mut s = Self {
            processor: ImageProcessor::Resize,
            image_block: media::IMAGE_BLOCK.to_string(),
            depth: need("depth")?,
            hidden: need("hidden_size")?,
            n_heads: need("num_heads")?,
            intermediate: need("intermediate_size")?,
            in_channels: obj_u64(vc, "in_channels").unwrap_or(3),
            patch: need("patch_size")?,
            temporal: obj_u64(vc, "temporal_patch_size").unwrap_or(2),
            merge: obj_u64(vc, "spatial_merge_size").unwrap_or(2),
            out_hidden: need("out_hidden_size")?,
            n_pos: need("num_position_embeddings")?,
            gelu_tanh: vc.get("hidden_act").and_then(JsonValue::as_str)
                == Some("gelu_pytorch_tanh"),
            gelu_bf16: false,
            image_token_id: tok("image_token_id")?,
            video_token_id: root
                .get("video_token_id")
                .and_then(JsonValue::as_i64)
                .map_or(-1, |v| v as i32),
            vision_start_id: tok("vision_start_token_id")?,
            vision_end_id: tok("vision_end_token_id")?,
            mrope_section,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            video: None,
        };
        if let Some(text) = preprocessor_json {
            let p = parse_json(text).map_err(|e| format!("preprocessor_config.json: {e}"))?;
            if let Some(size) = p.get("size") {
                if let Some(v) = obj_u64(size, "shortest_edge") {
                    s.min_pixels = v;
                }
                if let Some(v) = obj_u64(size, "longest_edge") {
                    s.max_pixels = v;
                }
            }
            if let Some(v) = obj_u64(&p, "min_pixels") {
                s.min_pixels = v;
            }
            if let Some(v) = obj_u64(&p, "max_pixels") {
                s.max_pixels = v;
            }
            if let Some(v) = obj_f64x3(&p, "image_mean") {
                s.image_mean = v;
            }
            if let Some(v) = obj_f64x3(&p, "image_std") {
                s.image_std = v;
            }
            for (key, model) in [
                ("patch_size", s.patch),
                ("temporal_patch_size", s.temporal),
                ("merge_size", s.merge),
            ] {
                if let Some(v) = obj_u64(&p, key) {
                    if v != model {
                        return Err(format!(
                            "preprocessor_config.json {key}={v} disagrees with vision_config ({model})"
                        ));
                    }
                }
            }
        }
        s.validate()?;
        Ok(s)
    }

    /// Shape rules the graphs, the merge order and the M-RoPE layout rely on.
    pub fn validate(&self) -> Result<(), String> {
        if self.depth == 0 || self.hidden == 0 || self.n_heads == 0 || self.intermediate == 0 {
            return Err("vision_config: depth, hidden_size, num_heads and intermediate_size must be positive".into());
        }
        if !self.hidden.is_multiple_of(self.n_heads) || !self.head_dim().is_multiple_of(4) {
            return Err(format!(
                "vision_config: hidden_size {} / num_heads {} must give a head width divisible by 4 (2-D rotary pairs)",
                self.hidden, self.n_heads
            ));
        }
        if self.merge != 2 {
            return Err(format!(
                "vision_config.spatial_merge_size {} is not supported (2x2 merger only)",
                self.merge
            ));
        }
        if self.temporal == 0 || self.patch == 0 || self.in_channels != 3 {
            return Err(
                "vision_config: patch sizes must be positive and in_channels must be 3".into(),
            );
        }
        if self.processor == ImageProcessor::Tiles {
            // no resize and no M-RoPE: the tile rule and the plain token positions of the
            // decoder need neither the pixel budget nor the frequency sections
            if self.temporal != 1 {
                return Err("a tiled image processor takes one frame per patch".into());
            }
            return Ok(());
        }
        let side = self.pos_side();
        if side * side != self.n_pos || side < 2 {
            return Err(format!(
                "vision_config.num_position_embeddings {} is not a square grid",
                self.n_pos
            ));
        }
        if self.mrope_section.iter().sum::<u64>() == 0 {
            return Err("mrope_section sums to zero".into());
        }
        if self.image_std.contains(&0.0) {
            return Err("image_std holds a zero".into());
        }
        Ok(())
    }

    pub fn head_dim(&self) -> u64 {
        self.hidden / self.n_heads
    }

    /// Values of one patch row: channels x temporal frames x patch x patch.
    pub fn patch_row(&self) -> u64 {
        self.in_channels * self.temporal * self.patch * self.patch
    }

    /// Width of one merged row entering the merger MLP.
    pub fn merged_width(&self) -> u64 {
        self.hidden * self.merge * self.merge
    }

    /// Patches that make one decoder token.
    pub fn patches_per_token(&self) -> u64 {
        self.merge * self.merge
    }

    pub fn pos_side(&self) -> u64 {
        (self.n_pos as f64).sqrt().round() as u64
    }

    /// Pixel side an image dimension is resized to a multiple of.
    pub fn resize_factor(&self) -> u64 {
        self.patch * self.merge
    }
}

/// Per-request vision timings and sizes (kept apart from the decoder's prefill / decode numbers).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisionReport {
    pub images: usize,
    /// Images the tower did not run for: all their rows lie inside the prefix a request took
    /// over from the one before it.
    pub reused_images: usize,
    pub patches: u64,
    pub image_tokens: u64,
    /// Host input preparation and uploads (pixels, position rows, tower RoPE tables).
    pub prep_ms: f64,
    /// Tower launches until the last image's embeddings are in HBM (synced).
    pub encode_ms: f64,
    /// Tower recipes compiled (or loaded from the recipe cache) on first use of an image size.
    pub compile_ms: f64,
    /// M-RoPE table rows computed and uploaded for the request.
    pub rope_ms: f64,
    pub rope_rows: u64,
    /// `position(slot) = slot + rope_delta` for every slot at or after the prompt end.
    pub rope_delta: i64,
}

#[cfg(test)]
pub(crate) mod test_util {
    /// `config.json` of the Qwen3.8-27B checkpoint reduced to the fields the vision code reads.
    pub const QWEN38_27B_CONFIG: &str = r#"{
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "image_token_id": 248056,
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {
          "mrope_interleaved": true,
          "mrope_section": [11, 11, 10],
          "partial_rotary_factor": 0.25,
          "rope_theta": 10000000
        }
      },
      "video_token_id": 248057,
      "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "intermediate_size": 4304,
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 5120,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
      },
      "vision_end_token_id": 248054,
      "vision_start_token_id": 248053
    }"#;

    pub const QWEN38_27B_PREPROCESSOR: &str = r#"{
      "size": {"longest_edge": 16777216, "shortest_edge": 65536},
      "patch_size": 16,
      "temporal_patch_size": 2,
      "merge_size": 2,
      "image_mean": [0.5, 0.5, 0.5],
      "image_std": [0.5, 0.5, 0.5]
    }"#;

    pub const QWEN38_27B_VIDEO_PREPROCESSOR: &str = r#"{
      "size": {"longest_edge": 25165824, "shortest_edge": 4096},
      "patch_size": 16,
      "temporal_patch_size": 2,
      "merge_size": 2,
      "image_mean": [0.5, 0.5, 0.5],
      "image_std": [0.5, 0.5, 0.5]
    }"#;

    pub fn qwen38_vision_spec() -> super::VisionSpec {
        let mut s =
            super::VisionSpec::from_json(QWEN38_27B_CONFIG, Some(QWEN38_27B_PREPROCESSOR)).unwrap();
        s.video = Some(super::VideoSpec::from_json(QWEN38_27B_VIDEO_PREPROCESSOR, &s).unwrap());
        s
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::*;
    use super::*;

    #[test]
    fn qwen38_vision_spec_parses_and_validates() {
        let s = qwen38_vision_spec();
        assert_eq!(
            (s.depth, s.hidden, s.n_heads, s.intermediate),
            (27, 1152, 16, 4304)
        );
        assert_eq!(
            (s.head_dim(), s.patch_row(), s.merged_width()),
            (72, 1536, 4608)
        );
        assert_eq!(
            (s.pos_side(), s.resize_factor(), s.patches_per_token()),
            (48, 32, 4)
        );
        assert_eq!(s.image_token_id, 248_056);
        assert_eq!(
            (s.vision_start_id, s.vision_end_id, s.video_token_id),
            (248_053, 248_054, 248_057)
        );
        assert_eq!(s.mrope_section, [11, 11, 10]);
        assert_eq!((s.min_pixels, s.max_pixels), (65_536, 16_777_216));
        assert!(s.gelu_tanh);
        // the video processor: its own pixel budget, the processor's defaults for the rest
        let v = s.video.unwrap();
        assert_eq!((v.min_pixels, v.max_pixels), (4_096, 25_165_824));
        assert_eq!((v.fps, v.min_frames, v.max_frames), (2.0, 4, 768));
        let bad = QWEN38_27B_VIDEO_PREPROCESSOR.replace("\"patch_size\": 16", "\"patch_size\": 14");
        assert!(VideoSpec::from_json(&bad, &s).is_err());
        assert!(VideoSpec::from_json(r#"{"fps": 0}"#, &s).is_err());
        let d = VideoSpec::from_json("{}", &s).unwrap();
        assert_eq!((d.min_pixels, d.max_pixels), (131_072, 786_432));
    }

    #[test]
    fn text_only_and_deepstack_configs_are_refused() {
        let text_only = r#"{"model_type": "qwen3", "hidden_size": 4096}"#;
        assert!(VisionSpec::from_json(text_only, None).is_err());
        let deep = QWEN38_27B_CONFIG.replace(
            "\"deepstack_visual_indexes\": []",
            "\"deepstack_visual_indexes\": [8, 16]",
        );
        assert!(VisionSpec::from_json(&deep, None)
            .unwrap_err()
            .contains("DeepStack"));
        let merge3 =
            QWEN38_27B_CONFIG.replace("\"spatial_merge_size\": 2", "\"spatial_merge_size\": 3");
        assert!(VisionSpec::from_json(&merge3, None).is_err());
    }

    /// Checks the host pipeline against a directory of reference prompt records (set
    /// `GAUDIUM_VISION_REF_DIR`; run with `--ignored`). Each `<name>.json` holds
    /// `prompt_token_ids`, `images[{grid_thw, pixels, span}]`, `mrope` (3 x P) and `rope_delta`
    /// from the reference processor; next to it `<name>_img<k>.ppm` (the input image) and
    /// `<name>_img<k>.f32` (the processor's patch rows). Pixels must match bit for bit.
    #[test]
    #[ignore = "needs GAUDIUM_VISION_REF_DIR with reference prompt records"]
    fn reference_prompt_records_match_bit_for_bit() {
        use crate::json::parse_json;
        use crate::vision::mrope::{expand_image_pads, layout_prompt};
        use crate::vision::preprocess::patch_rows_f32;

        let dir = std::path::PathBuf::from(
            std::env::var("GAUDIUM_VISION_REF_DIR").expect("GAUDIUM_VISION_REF_DIR is not set"),
        );
        let vs = qwen38_vision_spec();
        let ints = |v: &JsonValue| -> Vec<i64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect()
        };
        let mut checked = 0usize;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if !name.ends_with(".json") || name.matches('.').count() != 1 {
                continue;
            }
            let rec = parse_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let Some(images) = rec.get("images").and_then(JsonValue::as_array) else {
                continue;
            };
            let ids: Vec<i32> = ints(rec.get("prompt_token_ids").unwrap())
                .into_iter()
                .map(|v| v as i32)
                .collect();
            let mut grids = Vec::new();
            for (k, im) in images.iter().enumerate() {
                let g = ints(im.get("grid_thw").unwrap());
                let stem = name.trim_end_matches(".json");
                let rgb = RgbImage::load_ppm(&dir.join(format!("{stem}_img{k}.ppm"))).unwrap();
                let (gh, gw, rows) = patch_rows_f32(&rgb, &vs).unwrap();
                assert_eq!(
                    (1, gh as i64, gw as i64),
                    (g[0], g[1], g[2]),
                    "{name} image {k} grid"
                );
                let want =
                    std::fs::read(dir.join(im.get("pixels").unwrap().as_str().unwrap())).unwrap();
                let got: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
                assert!(
                    got == want,
                    "{name} image {k}: patch rows differ from the reference"
                );
                grids.push((gh, gw));
            }
            let layout = layout_prompt(&ids, &grids, &vs).unwrap();
            let mrope = rec.get("mrope").unwrap().as_array().unwrap();
            for sec in 0..3 {
                let want = ints(&mrope[sec]);
                let got: Vec<i64> = layout.pos3.iter().map(|p| i64::from(p[sec])).collect();
                assert_eq!(got, want, "{name} section {sec}");
            }
            assert_eq!(
                layout.delta,
                rec.get("rope_delta").unwrap().as_i64().unwrap(),
                "{name}"
            );
            for (span, im) in layout.spans.iter().zip(images) {
                let s = ints(im.get("span").unwrap());
                assert_eq!(
                    (span.start as i64, span.len as i64),
                    (s[0], s[1]),
                    "{name} span"
                );
            }
            // collapsing every pad run to one placeholder and expanding again gives the prompt
            let mut collapsed: Vec<i32> = Vec::new();
            for &t in &ids {
                if t != vs.image_token_id || collapsed.last() != Some(&t) {
                    collapsed.push(t);
                }
            }
            let tokens: Vec<usize> = layout.spans.iter().map(|s| s.len).collect();
            assert_eq!(
                expand_image_pads(&collapsed, &tokens, vs.image_token_id).unwrap(),
                ids
            );
            checked += 1;
        }
        assert!(
            checked > 0,
            "no reference prompt records in {}",
            dir.display()
        );
    }
}
