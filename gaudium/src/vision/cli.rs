//! Command-line glue for image prompts: load and prepare the images, put the vision
//! placeholders into the prompt, and size the tower for exactly these images.

use std::path::Path;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::engine::{EngineConfig, GenEngine, GenParams, GenReport};
use crate::verified::{vis_plan_raise, vis_request_tokens};
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
        // a model whose processor writes more than the template's block (begin and end tokens
        // around the pad) gets that text in place of each block
        let block = self.spec.image_block.as_str();
        let full = if self.spec.image_text.is_empty() {
            block
        } else {
            self.spec.image_text.as_str()
        };
        if text.contains(full) || text.contains("<|image_pad|>") {
            return text.to_string();
        }
        if text.contains(block) {
            return text.replace(block, full);
        }
        let mut s = full.repeat(self.images.len());
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
        configure_requests(&[self], cfg, max_patches);
    }
}

/// Expands the image-pad placeholders of every request that has images (`images[r]` belongs
/// to `prompts[r]`) to its images' token runs.
pub fn expand_requests(
    prompts: &mut [Vec<i32>],
    images: &[Option<CliImages>],
) -> Result<(), String> {
    if prompts.len() != images.len() {
        return Err(format!(
            "{} requests but images for {}",
            prompts.len(),
            images.len()
        ));
    }
    for (ids, imgs) in prompts.iter_mut().zip(images) {
        if let Some(v) = imgs {
            *ids = v.expand_prompt(ids)?;
        }
    }
    Ok(())
}

/// Plans the tower for the images of several requests of one run: its largest image (or the
/// larger explicit patch cap), the image tokens of the request that holds the most, and a
/// recipe for every image size. The counts and the plan are the verified
/// `vis_request_tokens` / `vis_plan_raise` (`verified/vision_budget.bend`).
pub fn configure_requests(requests: &[&CliImages], cfg: &mut EngineConfig, max_patches: u64) {
    let mut ns: Vec<u64> = Vec::new();
    let (mut patches, mut tokens) = (max_patches, 0u64);
    for r in requests {
        let mut request = 0u64;
        for i in &r.images {
            let n = i.n_patches() as u64;
            if !ns.contains(&n) {
                ns.push(n);
            }
            patches = vis_plan_raise(patches, n);
            request = vis_request_tokens(request, i.n_tokens() as u64);
        }
        tokens = vis_plan_raise(tokens, request);
    }
    cfg.vision_max_patches = patches;
    cfg.vision_max_tokens = tokens;
    cfg.vision_precompile = ns;
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

/// Measurement and state checks after the requests, in the same process: the first request
/// again `repeat - 1` times under the first sampler setting (recipes and tables warm), then
/// optionally a text-only request (`after`) once per sampler setting, in setting order, which
/// must give the tokens it gives on an engine that never saw an image. Each run prints its
/// tokens as `REPEAT<n> TOKENS:` / `AFTER TOKENS:` lines.
pub fn run_extra_requests(
    engine: &mut dyn GenEngine,
    images: Option<&[PreparedImage]>,
    prompt_ids: &[i32],
    settings: &[GenParams],
    repeat: usize,
    after: Option<&[i32]>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let Some(params) = settings.first() else {
        return Ok(());
    };
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
    if let Some(ids) = after {
        for (k, p) in settings.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            let r = engine.generate_cancellable(ids, p, Some(cancel), &mut |_| {})?;
            eprintln!("AFTER TOKENS: {:?}", r.generated_tokens);
            let tag = match k {
                0 => "text after image".to_string(),
                _ => format!("text after image, sampler setting {k}"),
            };
            eprintln!("{}", timing_line(&tag, &r));
        }
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

        // several requests: the largest image, the request with the most image tokens, every
        // image size once
        let one = CliImages {
            spec: qwen38_vision_spec(),
            images: vec![solid(256, 256)],
        };
        configure_requests(&[&one, &cli], &mut cfg, 0);
        assert_eq!(cfg.vision_max_patches, 560);
        assert_eq!(cfg.vision_max_tokens, 204);
        assert_eq!(cfg.vision_precompile, vec![256, 560]);
        configure_requests(&[&one], &mut cfg, 0);
        assert_eq!((cfg.vision_max_patches, cfg.vision_max_tokens), (256, 64));

        // per request: a text request keeps its ids, an image request gets its own runs
        let text = vec![1, 2, 3];
        let with_one = vec![1, 248_053, 248_056, 248_054, 7];
        let mut prompts = vec![text.clone(), with_one.clone(), ids.to_vec()];
        let images = vec![None, Some(one), Some(cli)];
        expand_requests(&mut prompts, &images).unwrap();
        assert_eq!(prompts[0], text);
        assert_eq!(prompts[1].len(), with_one.len() - 1 + 64);
        assert_eq!(prompts[2].len(), ids.len() - 2 + 64 + 140);
        // a request whose placeholders do not match its images is refused
        let mut bad = vec![text.clone()];
        assert!(expand_requests(&mut bad, &images[1..2]).is_err());
        assert!(expand_requests(&mut prompts, &images[..1]).is_err());
    }
}
