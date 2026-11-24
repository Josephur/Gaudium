//! Model weight packers and Intel Gaudi 2 execution engines.

use std::sync::atomic::AtomicBool;

use crate::model_spec::{ModelFamily, ModelSpec};

pub mod fp8_graph;
pub mod fp8_pack;
pub mod gemma4;
pub mod gemma4_fp8;
pub mod gemma4_packer;
pub mod inkling2;
pub mod mem_report;
pub mod moe;
pub mod packer;
pub mod qwen3;
pub mod qwen3_traffic;
pub mod sampler;
pub mod step_opts;

/// Engine-independent load configuration (model dir, kernels, context capacity, recipe cache).
pub type EngineConfig = qwen3::Qwen3Config;
/// Engine-independent generation parameters (sampling, limits, EOS handling).
pub type GenParams = qwen3::Qwen3GenParams;
/// Engine-independent generation report (prefill and decode timings kept separate).
pub type GenReport = qwen3::Qwen3GenReport;

/// Common interface of every live Gaudi 2 text-generation engine, so the CLI and the server
/// dispatch on the model family instead of naming a concrete engine type.
pub trait GenEngine: Send {
    fn spec(&self) -> &ModelSpec;
    fn max_ctx(&self) -> u64;
    /// Exact 5-category HBM ledger bytes `[weights, kv, state, scratch, workspace]`.
    fn hbm_bytes(&self) -> [u64; 5];
    /// Runs prefill + decode, calling `on_token` per generated token. Implementations check
    /// `cancel` only between synced launches so no Synapse launch is left queued.
    fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        cancel: Option<&AtomicBool>,
        on_token: &mut dyn FnMut(i32),
    ) -> Result<GenReport, String>;

    /// Reader of the engine's measured live `profile` block (`None`: the engine reports none).
    fn live_profile(&self) -> Option<crate::live_report::ProfileHandle> {
        None
    }

    /// The speculative runtime the engine was loaded with (`None`: every request is `K = 0`).
    fn spec_info(&self) -> Option<qwen3::spec::readout::SpecInfo> {
        None
    }

    /// Wall time of the device state reset that preceded the last request, in ms. It runs
    /// before the prefill timer starts, so it is not part of the report's `ttft_ms`.
    fn last_reset_ms(&self) -> f64 {
        0.0
    }

    /// Vision geometry when the engine was loaded with image input (`None` otherwise).
    fn vision_spec(&self) -> Option<&crate::vision::VisionSpec> {
        None
    }
    /// Hands the images of the NEXT `generate_cancellable` call to the engine, in prompt order
    /// (the prompt holds one expanded image-pad run per image).
    fn set_request_images(
        &mut self,
        _images: Vec<crate::vision::PreparedImage>,
    ) -> Result<(), String> {
        Err("this engine has no image input".into())
    }
    /// Vision timings of the last request (`None` if it had no images).
    fn last_vision_report(&self) -> Option<crate::vision::VisionReport> {
        None
    }
    /// Image embeddings of the last image request read back from HBM (`tokens x hidden` BF16).
    fn read_vision_embeddings(&self) -> Result<Vec<u8>, String> {
        Err("this engine has no image input".into())
    }
    /// Turns the loaded drafter off (`false`) or on again for the requests that follow, so one
    /// process can produce a request's non-speculative stream and its speculative one.
    fn set_speculative(&mut self, _on: bool) -> Result<(), String> {
        Err("this engine cannot switch speculative decoding per request".into())
    }
    /// FP8 activation-scale table gathered by a calibration load (`EngineConfig::fp8_calibrate`):
    /// the running maxima over every prefill run so far. `None`: not a calibration load.
    fn fp8_calibration(&self) -> Option<std::collections::BTreeMap<String, crate::fp8::ActScale>> {
        None
    }
    /// K/V calibration load (`EngineConfig::kv_calibrate`): reads the trunk's caches back after a
    /// request whose prompt had `rows` positions and folds their per-head maxima in.
    fn kv_measure(&mut self, _rows: u64) -> Result<(), String> {
        Err("this engine has no K/V calibration pass".into())
    }
    /// The running K/V calibration of a calibration load (`None`: not one).
    fn kv_calibration(&self) -> Option<crate::core::kv_cache::KvCalibration> {
        None
    }
    /// Prefix reuse across requests: the positions the engine keeps to resume a prefill from
    /// and what its requests took over (`None`: this engine prefills every prompt whole).
    fn prefix_stats(&self) -> Option<qwen3::prefix_cache::PrefixStats> {
        None
    }
    /// Work for the time between two requests (the host-RAM tier's save). `waiting`: how many
    /// requests wait for the card now; the work gives way once one does.
    fn idle_work(&mut self, _waiting: &dyn Fn() -> u64) {}
    /// Called before the engine is dropped to release the card: what is worth keeping goes to
    /// host memory, and nothing is left queued on the device.
    fn before_release(&mut self) {}
    /// The host-RAM tier of this engine: measured rates, entry sizes, the last request
    /// (`None`: it has none).
    fn host_tier_facts(&self) -> Option<qwen3::host_tier::TierFacts> {
        None
    }
}

impl GenEngine for qwen3::Qwen3Engine {
    fn spec(&self) -> &ModelSpec {
        qwen3::Qwen3Engine::spec(self)
    }
    fn max_ctx(&self) -> u64 {
        qwen3::Qwen3Engine::max_ctx(self)
    }
    fn hbm_bytes(&self) -> [u64; 5] {
        qwen3::Qwen3Engine::hbm_bytes(self)
    }
    fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        cancel: Option<&AtomicBool>,
        on_token: &mut dyn FnMut(i32),
    ) -> Result<GenReport, String> {
        qwen3::Qwen3Engine::generate_cancellable(self, prompt_ids, params, cancel, on_token)
    }
    fn live_profile(&self) -> Option<crate::live_report::ProfileHandle> {
        Some(qwen3::Qwen3Engine::live_profile(self))
    }
    fn spec_info(&self) -> Option<qwen3::spec::readout::SpecInfo> {
        qwen3::Qwen3Engine::spec_info(self)
    }
    fn last_reset_ms(&self) -> f64 {
        qwen3::Qwen3Engine::last_reset_ms(self)
    }
    fn vision_spec(&self) -> Option<&crate::vision::VisionSpec> {
        qwen3::Qwen3Engine::vision_spec(self)
    }
    fn set_request_images(
        &mut self,
        images: Vec<crate::vision::PreparedImage>,
    ) -> Result<(), String> {
        qwen3::Qwen3Engine::set_request_images(self, images)
    }
    fn last_vision_report(&self) -> Option<crate::vision::VisionReport> {
        qwen3::Qwen3Engine::last_vision_report(self).cloned()
    }
    fn read_vision_embeddings(&self) -> Result<Vec<u8>, String> {
        qwen3::Qwen3Engine::read_vision_embeddings(self)
    }
    fn fp8_calibration(&self) -> Option<std::collections::BTreeMap<String, crate::fp8::ActScale>> {
        qwen3::Qwen3Engine::fp8_calibration(self)
    }
    fn kv_measure(&mut self, rows: u64) -> Result<(), String> {
        qwen3::Qwen3Engine::kv_measure(self, rows)
    }
    fn kv_calibration(&self) -> Option<crate::core::kv_cache::KvCalibration> {
        qwen3::Qwen3Engine::kv_calibration(self)
    }
    fn prefix_stats(&self) -> Option<qwen3::prefix_cache::PrefixStats> {
        qwen3::Qwen3Engine::prefix_stats(self)
    }
    fn idle_work(&mut self, waiting: &dyn Fn() -> u64) {
        qwen3::Qwen3Engine::idle_work(self, waiting)
    }
    fn before_release(&mut self) {
        qwen3::Qwen3Engine::before_release(self)
    }
    fn host_tier_facts(&self) -> Option<qwen3::host_tier::TierFacts> {
        qwen3::Qwen3Engine::host_tier_facts(self)
    }
}

impl GenEngine for gemma4::Gemma4Engine {
    fn spec(&self) -> &ModelSpec {
        gemma4::Gemma4Engine::spec(self)
    }
    fn max_ctx(&self) -> u64 {
        gemma4::Gemma4Engine::max_ctx(self)
    }
    fn hbm_bytes(&self) -> [u64; 5] {
        gemma4::Gemma4Engine::hbm_bytes(self)
    }
    fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        cancel: Option<&AtomicBool>,
        on_token: &mut dyn FnMut(i32),
    ) -> Result<GenReport, String> {
        gemma4::Gemma4Engine::generate_cancellable(self, prompt_ids, params, cancel, on_token)
    }
    fn live_profile(&self) -> Option<crate::live_report::ProfileHandle> {
        Some(gemma4::Gemma4Engine::live_profile(self))
    }
    fn spec_info(&self) -> Option<qwen3::spec::readout::SpecInfo> {
        gemma4::Gemma4Engine::spec_info(self)
    }
    fn fp8_calibration(&self) -> Option<std::collections::BTreeMap<String, crate::fp8::ActScale>> {
        gemma4::Gemma4Engine::fp8_calibration(self)
    }
    fn set_speculative(&mut self, on: bool) -> Result<(), String> {
        gemma4::Gemma4Engine::set_speculative(self, on)
    }
}

impl GenEngine for inkling2::Inkling2Engine {
    fn spec(&self) -> &ModelSpec {
        inkling2::Inkling2Engine::spec(self)
    }
    fn max_ctx(&self) -> u64 {
        inkling2::Inkling2Engine::max_ctx(self)
    }
    fn hbm_bytes(&self) -> [u64; 5] {
        inkling2::Inkling2Engine::hbm_bytes(self)
    }
    fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        cancel: Option<&AtomicBool>,
        on_token: &mut dyn FnMut(i32),
    ) -> Result<GenReport, String> {
        inkling2::Inkling2Engine::generate_cancellable(self, prompt_ids, params, cancel, on_token)
    }
    fn live_profile(&self) -> Option<crate::live_report::ProfileHandle> {
        Some(inkling2::Inkling2Engine::live_profile(self))
    }
    fn prefix_stats(&self) -> Option<qwen3::prefix_cache::PrefixStats> {
        inkling2::Inkling2Engine::prefix_stats(self)
    }
    fn spec_info(&self) -> Option<qwen3::spec::readout::SpecInfo> {
        inkling2::Inkling2Engine::spec_info(self)
    }
    fn vision_spec(&self) -> Option<&crate::vision::VisionSpec> {
        inkling2::Inkling2Engine::vision_spec(self)
    }
    fn set_request_images(
        &mut self,
        images: Vec<crate::vision::PreparedImage>,
    ) -> Result<(), String> {
        inkling2::Inkling2Engine::set_request_images(self, images)
    }
    fn last_vision_report(&self) -> Option<crate::vision::VisionReport> {
        inkling2::Inkling2Engine::last_vision_report(self)
    }
    fn read_vision_embeddings(&self) -> Result<Vec<u8>, String> {
        inkling2::Inkling2Engine::read_vision_embeddings(self)
    }
}

/// Loads the engine matching the model family declared in `<model_dir>/config.json`.
pub fn load_engine(cfg: &EngineConfig) -> Result<Box<dyn GenEngine>, String> {
    if crate::gguf::looks_like_gguf(&cfg.model_dir) {
        // a GGUF file names its family in its metadata
        let g = crate::gguf::Gguf::open(&cfg.model_dir).map_err(|e| format!("open GGUF: {e}"))?;
        if inkling2::gguf_source::is_family(&g) {
            drop(g);
            return Ok(Box::new(inkling2::Inkling2Engine::load(cfg)?));
        }
        return Err(format!(
            "no live Gaudi 2 engine loads a GGUF file of architecture '{}' yet",
            g.arch()
        ));
    }
    let spec = ModelSpec::from_config_file(cfg.model_dir.join("config.json"))
        .map_err(|e| format!("load config.json: {e}"))?;
    match spec.family {
        ModelFamily::Qwen3 | ModelFamily::Qwen35Dense | ModelFamily::Qwen35Moe => {
            Ok(Box::new(qwen3::Qwen3Engine::load(cfg)?))
        }
        ModelFamily::Gemma4 => Ok(Box::new(gemma4::Gemma4Engine::load(cfg)?)),
        ModelFamily::Inkling2 => Ok(Box::new(inkling2::Inkling2Engine::load(cfg)?)),
        other => Err(format!(
            "no live Gaudi 2 engine is wired up for model family {other:?} yet"
        )),
    }
}

/// Fills the recipe cache for `cfg` and prints its device memory plan without opening the card
/// (`qwen3::Qwen3Engine::build_cache`). Available for the Qwen3 / Qwen3.5 / Qwen3.8 engine.
pub fn build_cache(cfg: &EngineConfig) -> Result<(), String> {
    let spec = ModelSpec::from_config_file(cfg.model_dir.join("config.json"))
        .map_err(|e| format!("load config.json: {e}"))?;
    match spec.family {
        ModelFamily::Qwen3 | ModelFamily::Qwen35Dense | ModelFamily::Qwen35Moe => {
            qwen3::Qwen3Engine::build_cache(cfg)
        }
        other => Err(format!(
            "--build-cache is not available for model family {other:?} yet"
        )),
    }
}

/// The model's own context limit (`max_position_embeddings` in `config.json`, looked up in
/// `text_config` for multimodal checkpoints, or a GGUF file's `<arch>.context_length`); `None`
/// when the model does not state one (`ModelText::max_positions`).
pub fn model_max_positions(model_dir: &std::path::Path) -> Option<u64> {
    crate::model_text::ModelText::open(model_dir)
        .ok()?
        .max_positions()
}

/// Clamps a requested context capacity to what the model supports, in whole KV chunks
/// (never below one chunk). Used where one `--max-ctx` applies to several models.
pub fn clamp_ctx_to_model(requested: u64, model_dir: &std::path::Path) -> u64 {
    let chunk = qwen3::KV_CHUNK;
    let want = crate::verified::align_up(requested.max(chunk), chunk);
    match model_max_positions(model_dir) {
        Some(max_pos) => want.min((max_pos / chunk).max(1) * chunk),
        None => want,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_ctx_respects_model_limit_in_whole_chunks() {
        let dir = std::env::temp_dir().join(format!("gaudium-clamp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"text_config":{"max_position_embeddings":40960}}"#,
        )
        .unwrap();
        assert_eq!(model_max_positions(&dir), Some(40960));
        assert_eq!(clamp_ctx_to_model(262144, &dir), 40960);
        assert_eq!(clamp_ctx_to_model(16384, &dir), 16384);
        assert_eq!(clamp_ctx_to_model(1, &dir), 4096);
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert_eq!(clamp_ctx_to_model(262144, &dir), 262144);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
