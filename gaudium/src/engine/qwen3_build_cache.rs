//! Recipe pre-compilation without the card (a child module of `engine::qwen3`).
//!
//! Compiling a graph needs the SynapseAI graph compiler and the kernel glue, not the device: only
//! `synInitialize` is called here, never a device acquire, so this runs beside a process that owns
//! the card. It walks the compile phase of `Qwen3Engine::load` with the same builders, kernel
//! fingerprints and runtime `prepare` functions, so every recipe it writes is the recipe a load
//! with the same options looks up. Recipes with the vendor expert-select node take tens of
//! seconds each to compile on the host (measured with the routed MoE form), and a large context
//! has one decode-step recipe per KV chunk count: compiled here, they cost no card time.
//!
//! Scope of this pass: the mixture-of-experts family (the other families compile in seconds).

use super::*;

/// Compiles into the recipe cache every recipe `Qwen3Engine::load(cfg)` compiles, the full-tile
/// prefill recipes and the drafter's own recipes of every KV chunk count, and the speculative
/// step recipes of the chunk counts in `spec_chunks` (a load compiles only the first and the last
/// chunk count of a large context; the others on first use inside a request).
pub fn build_cache(cfg: &Qwen3Config, spec_chunks: &[u64]) -> Result<(), String> {
    let t_start = Instant::now();
    if cfg.no_cache {
        return Err("--build-cache writes the recipe cache: drop --no-cache".into());
    }
    let mut spec = ModelSpec::from_config_file(cfg.model_dir.join("config.json"))
        .map_err(|e| format!("load config.json: {e}"))?;
    spec.validate()
        .map_err(|e| format!("verified ModelSpec validation failed: {e}"))?;
    crate::engine::fp8_pack::resolve_precision(cfg, &mut spec)?;
    if !(spec.family == ModelFamily::Qwen35Moe && crate::engine::moe::supported_shape(&spec)) {
        return Err(format!(
            "--build-cache covers the Qwen3.5 mixture-of-experts family in this build (got {:?})",
            spec.family
        ));
    }
    let required_elfs = crate::engine::moe::QWEN35_MOE_ELFS;
    let has_gdn = spec.n_linear_layers() > 0;
    let max_ctx = align_up(cfg.max_ctx.max(KV_CHUNK), KV_CHUNK);
    let max_chunks = max_ctx / KV_CHUNK;
    let prefill_tile = cfg.prefill_tile.clamp(1, MAX_PREFILL_TILE);

    let kenv = KernelEnv::configure_from_dir(&cfg.kernels_dir)?;
    let elf_hashes = kenv.fingerprint_hashes(required_elfs)?;
    let api = Arc::new(SynapseApi::load_default().map_err(|e| format!("load libSynapse.so: {e}"))?);
    let driver_version = api.driver_version().unwrap_or_else(|_| "unknown".into());
    // SAFETY: the first Synapse lifecycle call of this process; no device is acquired.
    let st = unsafe { (api.syn_initialize)() };
    if st != 0 {
        return Err(format!("synInitialize failed with status {st}"));
    }
    let cache = match &cfg.cache_dir {
        Some(dir) => RecipeCache::at_dir(dir),
        None => RecipeCache::from_env(),
    };
    eprintln!(
        "[gaudium] build-cache: no device opened; driver={driver_version} family={:?} max_ctx={max_ctx} ({max_chunks} chunks)",
        spec.family
    );

    let mut n = [0usize; 2]; // [compiled, already cached]
    let res = (|| -> Result<(), String> {
        let mut one = |gb: GraphBuilder| -> Result<CompiledRecipe, String> {
            let t0 = Instant::now();
            let r = gb
                .compile(Arc::clone(&api), &cache, &driver_version)
                .map_err(|e| format!("compile {}: {e}", r_name(&gb)))?;
            n[usize::from(r.from_cache)] += 1;
            if cfg.verbose || !r.from_cache {
                eprintln!(
                    "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                    r.name,
                    r.from_cache,
                    r.ws_size as f64 / (1024.0 * 1024.0),
                    t0.elapsed().as_secs_f64()
                );
            }
            Ok(r)
        };
        // the compile phase of `Qwen3Engine::load`, in its order
        one(build_head_t1_graph(&spec))?;
        for nch in 1..=max_chunks {
            one(build_decode_step_graph(&spec, max_ctx, nch, &elf_hashes))?;
        }
        if prefill_tile > 1 {
            one(build_prefill_attn_layer_graph(
                &spec,
                prefill_tile,
                max_ctx,
                1,
                &elf_hashes,
            ))?;
            if has_gdn {
                one(build_prefill_linear_layer_graph(
                    &spec,
                    prefill_tile,
                    &elf_hashes,
                ))?;
            }
            one(build_embed_graph(&spec, prefill_tile))?;
        }
        // the attention prefill recipe of every later chunk count (first use is inside a request)
        if prefill_tile > 1 {
            for nch in 2..=max_chunks {
                one(build_prefill_attn_layer_graph(
                    &spec,
                    prefill_tile,
                    max_ctx,
                    nch,
                    &elf_hashes,
                ))?;
            }
        }
        moe_rt::RoutedPrefill::prepare(&spec, prefill_tile, &api, &cache, &driver_version, true)?;
        let samp_eager: Vec<u64> = if cfg.spec.is_none() && max_chunks <= 8 {
            (1..=max_chunks).collect()
        } else if max_chunks > 1 {
            vec![1, max_chunks]
        } else {
            vec![1]
        };
        let samp = samp_rt::SampRt::prepare(
            &spec,
            &kenv,
            &elf_hashes,
            max_ctx,
            &samp_eager,
            &api,
            &cache,
            &driver_version,
            true,
        )?;
        if let Some(sc) = &cfg.spec {
            let rt = spec::SpecRuntime::prepare(
                sc,
                &spec,
                &kenv,
                &elf_hashes,
                max_ctx,
                max_chunks,
                prefill_tile,
                &api,
                &cache,
                &driver_version,
                true,
                samp.is_some(),
                // the chunk counts between the first and the last are chosen below
                false,
                &cfg.step,
            )?;
            // the drafter's own recipes of every chunk count (cheap; first use is inside a request)
            for nch in 1..=max_chunks {
                for gb in rt.lazy_drafter_graphs(&spec, prefill_tile, nch) {
                    one(gb)?;
                }
            }
            // later chunk counts of the speculative steps (a load compiles them on first use)
            let (keys, sampled_family) = rt.loaded_step_keys();
            for &nch in spec_chunks {
                if nch == 0 || nch > max_chunks {
                    return Err(format!(
                        "--build-cache-chunks {nch}: this context has 1..={max_chunks} chunks"
                    ));
                }
                for &key in &keys {
                    one(rt.lazy_step_graph(&spec, nch, key, false))?;
                    if sampled_family {
                        one(rt.lazy_step_graph(&spec, nch, key, true))?;
                    }
                }
            }
        }
        Ok(())
    })();
    // SAFETY: every recipe handle of this function is dropped; no device was acquired.
    unsafe {
        (api.syn_destroy)();
    }
    res?;
    eprintln!(
        "[gaudium] build-cache: done in {:.1}s ({} compiled, {} already cached among the listed recipes; the sampler and speculative runtimes report theirs above)",
        t_start.elapsed().as_secs_f64(),
        n[0],
        n[1]
    );
    Ok(())
}

fn r_name(gb: &GraphBuilder) -> String {
    gb.recipe_name().to_string()
}
