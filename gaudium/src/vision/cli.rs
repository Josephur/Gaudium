//! Command-line glue for image prompts: load and prepare the images, put the vision
//! placeholders into the prompt, and size the tower for exactly these images.

use std::path::Path;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::engine::{EngineConfig, GenEngine, GenParams, GenReport};
use crate::vision::mrope::expand_image_pads;
use crate::vision::preprocess::{prepare_image, PreparedImage, RgbImage};
use crate::vision::{VisionReport, VisionSpec};

/// The chat template's block for one image; the pad is expanded to the image's token count.
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Images of one command-line request, prepared for the tower.
pub struct CliImages {
    pub spec: VisionSpec,
    pub images: Vec<PreparedImage>,
}

/// Loads `(is_raw, arg)` images (`--image <file>`: JPEG, PNG, GIF, WebP, BMP or binary PPM,
/// decoded by the crate's own decoders and turned by its EXIF orientation, exactly as the
/// server decodes a request's image; `--image-raw WxH:file`) and prepares them. With a patch
/// cap, larger images are resized down to it. `None` when no image was given.
pub fn prepare_cli_images(
    model_dir: &Path,
    args: &[(bool, String)],
    max_patches: u64,
) -> Result<Option<CliImages>, String> {
    if args.is_empty() {
        return Ok(None);
    }
    let mut spec = VisionSpec::from_model_dir(model_dir)?;
    if max_patches > 0 {
        let cap = max_patches.saturating_mul(spec.patch * spec.patch);
        if cap < spec.min_pixels {
            return Err(format!(
                "--vision-max-patches {max_patches} is below the smallest image this model takes ({} patches)",
                spec.min_pixels.div_ceil(spec.patch * spec.patch)
            ));
        }
        spec.max_pixels = spec.max_pixels.min(cap);
    }
    let mut images = Vec::with_capacity(args.len());
    for (raw, arg) in args {
        let rgb = if *raw {
            RgbImage::load_raw_rgb(arg)?
        } else {
            let bytes = std::fs::read(arg).map_err(|e| format!("read {arg}: {e}"))?;
            let limits = crate::core::vision::MediaLimits::default();
            limits
                .check_bytes(bytes.len())
                .map_err(|e| format!("{arg}: {e}"))?;
            let decode_limits = crate::vision::decode::DecodeLimits {
                max_pixels: limits.max_pixels,
            };
            crate::vision::decode::decode_image(&bytes, &decode_limits)
                .map_err(|e| format!("{arg}: {e}"))?
        };
        images.push(prepare_image(&rgb, &spec).map_err(|e| format!("{arg}: {e}"))?);
    }
    Ok(Some(CliImages { spec, images }))
}

impl CliImages {
    /// Puts one placeholder block per image in front of the user text, as the chat template
    /// does for `[image..., text]` content, unless the text already places them itself.
    pub fn with_placeholders(&self, text: &str) -> String {
        if text.contains(self.spec.image_block.as_str()) || text.contains("<|image_pad|>") {
            return text.to_string();
        }
        let mut s = self.spec.image_block.repeat(self.images.len());
        s.push_str(text);
        s
    }

    /// Expands every image-pad placeholder of the tokenised prompt to its image's tokens.
    pub fn expand_prompt(&self, ids: &[i32]) -> Result<Vec<i32>, String> {
        let tokens: Vec<usize> = self.images.iter().map(PreparedImage::n_tokens).collect();
        expand_image_pads(ids, &tokens, self.spec.image_token_id)
    }

    /// Plans the tower for exactly these images (or the larger explicit patch cap).
    pub fn configure(&self, cfg: &mut EngineConfig, max_patches: u64) {
        let ns: Vec<u64> = self.images.iter().map(|i| i.n_patches() as u64).collect();
        cfg.vision_max_patches = max_patches.max(ns.iter().copied().max().unwrap_or(0));
        cfg.vision_max_tokens = self.images.iter().map(|i| i.n_tokens() as u64).sum();
        cfg.vision_precompile = ns;
    }
}

/// One status line with the vision share of TTFT kept apart from the decoder's prefill.
pub fn format_vision_report(v: &VisionReport, r: &GenReport) -> String {
    let vision_ms = v.prep_ms + v.encode_ms + v.compile_ms + v.rope_ms;
    let text_ms = (r.ttft_ms - vision_ms).max(1e-6);
    format!(
        "[gaudium] vision: {} image(s), {} patches -> {} tokens | host prep+upload {:.1} ms, tower {:.1} ms, recipes {:.1} ms, M-RoPE rows {} in {:.1} ms (delta {}) | prefill without vision: {:.1} ms ({:.1} tok/s)",
        v.images,
        v.patches,
        v.image_tokens,
        v.prep_ms,
        v.encode_ms,
        v.compile_ms,
        v.rope_rows,
        v.rope_ms,
        v.rope_delta,
        text_ms,
        (r.prompt_tokens as f64) / (text_ms / 1000.0)
    )
}

/// Parses comma / whitespace separated token IDs.
pub fn parse_ids(text: &str) -> Result<Vec<i32>, String> {
    text.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<i32>()
                .map_err(|_| format!("invalid token id '{s}'"))
        })
        .collect()
}

fn timing_line(tag: &str, r: &GenReport) -> String {
    format!(
        "[gaudium] {tag}: prefill {} tok in {:.1} ms ({:.1} tok/s) | decode: {} tok in {:.1} ms ({:.2} tok/s, {:.2} ms/step) | eos={}",
        r.prompt_tokens,
        r.ttft_ms,
        r.prefill_tok_s,
        r.generated_tokens.len().saturating_sub(1),
        r.decode_ms,
        r.decode_tok_s,
        r.ms_per_step,
        r.stopped_by_eos
    )
}

/// Measurement and state checks after the main request, in the same process: the same request
/// again `repeat - 1` times (recipes and tables warm), then optionally a text-only request,
/// which must give the tokens it gives in a process that never saw an image.
pub fn run_extra_requests(
    engine: &mut dyn GenEngine,
    images: Option<&[PreparedImage]>,
    prompt_ids: &[i32],
    params: &GenParams,
    repeat: usize,
    after_ids_file: Option<&str>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    for run in 1..repeat {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        if let Some(imgs) = images {
            engine.set_request_images(imgs.to_vec())?;
        }
        let r = engine.generate_cancellable(prompt_ids, params, Some(cancel), &mut |_| {})?;
        eprintln!("REPEAT{run} TOKENS: {:?}", r.generated_tokens);
        eprintln!("{}", timing_line(&format!("repeat {run}"), &r));
        if let Some(v) = engine.last_vision_report() {
            eprintln!("{}", format_vision_report(&v, &r));
        }
    }
    if let Some(path) = after_ids_file {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read --after-ids-file {path}: {e}"))?;
        let ids = parse_ids(&text)?;
        let r = engine.generate_cancellable(&ids, params, Some(cancel), &mut |_| {})?;
        eprintln!("AFTER TOKENS: {:?}", r.generated_tokens);
        eprintln!("{}", timing_line("text after image", &r));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::test_util::qwen38_vision_spec;

    fn solid(w: usize, h: usize) -> PreparedImage {
        let img = RgbImage::from_raw(w, h, vec![128u8; w * h * 3]).unwrap();
        prepare_image(&img, &qwen38_vision_spec()).unwrap()
    }

    #[test]
    fn placeholders_prompt_expansion_and_tower_sizing() {
        let cli = CliImages {
            spec: qwen38_vision_spec(),
            images: vec![solid(256, 256), solid(448, 320)],
        };
        let text = cli.with_placeholders("What is this?");
        assert_eq!(
            text,
            format!("{IMAGE_PLACEHOLDER}{IMAGE_PLACEHOLDER}What is this?")
        );
        // a prompt that places its own image is left alone
        let own = "Look: <|vision_start|><|image_pad|><|vision_end|> and say.";
        assert_eq!(cli.with_placeholders(own), own);

        let ids = [1, 248_053, 248_056, 248_054, 248_053, 248_056, 248_054, 7];
        let out = cli.expand_prompt(&ids).unwrap();
        assert_eq!(out.len(), ids.len() - 2 + 64 + 140);
        assert_eq!(out.iter().filter(|&&t| t == 248_056).count(), 204);

        let mut cfg = EngineConfig::new("m".into(), "k".into());
        cli.configure(&mut cfg, 0);
        assert_eq!(cfg.vision_max_patches, 560);
        assert_eq!(cfg.vision_max_tokens, 204);
        assert_eq!(cfg.vision_precompile, vec![256, 560]);
        cli.configure(&mut cfg, 4096);
        assert_eq!(cfg.vision_max_patches, 4096);
    }
}
