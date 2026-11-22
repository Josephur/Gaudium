//! Routed-MoE prefill runtime of `Qwen3Engine` (a child module of `engine::qwen3`, so it reads the
//! engine's private recipe maps and scratch tables without widening them).
//!
//! In the routed form a prefill tile runs, per layer, the layer's own recipe (attention or Gated
//! DeltaNet, post-attention norm, router, gated shared expert) and then ONE generic
//! routed-experts recipe (`engine::moe::build_prefill_routed_graph`): the vendor expert-select
//! node over the tile's rows and the layer tail. Keeping the expert-select node out of the layer
//! recipes matters on Gaudi 2 because that node is expensive to compile (it expands into one
//! weight multiplexer pair per row and routed expert) and the attention layer recipe exists once
//! per KV chunk count: a 262,144-token context has 64 of those and one routed-experts recipe.
//!
//! The recipe is built for a few padded row counts (`engine::moe::routed_prefill_rows`), so a new
//! prompt-tail length compiles nothing. Rows past the tile hold whatever an earlier tile left in
//! the layer scratch (zeroed once at load, so they are always valid ids and finite values); their
//! results are written past the tile's rows of the output buffer and never read.

use super::*;
use crate::engine::moe;

/// Routed-experts prefill recipes by padded row count. Empty and inert unless the engine runs
/// the routed MoE form.
#[derive(Default)]
pub(super) struct RoutedPrefill {
    on: bool,
    recipes: HashMap<u64, CompiledRecipe>,
    max_ws: u64,
}

impl RoutedPrefill {
    pub(super) fn max_ws(&self) -> u64 {
        self.max_ws
    }

    /// Before the HBM plan: compiles the recipe of the full prefill tile (the widest one, whose
    /// workspace bounds every narrower one's).
    pub(super) fn prepare(
        spec: &ModelSpec,
        prefill_tile: u64,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
        verbose: bool,
    ) -> Result<Self, String> {
        if !(spec.is_moe() && spec.moe_routed) {
            return Ok(Self::default());
        }
        let mut rt = Self {
            on: true,
            ..Self::default()
        };
        let rows = moe::routed_prefill_rows(prefill_tile, prefill_tile);
        let t0 = Instant::now();
        let r = moe::build_prefill_routed_graph(spec, rows)
            .compile(Arc::clone(api), cache, driver_version)
            .map_err(|e| format!("compile routed-experts prefill T={rows}: {e}"))?;
        if verbose {
            eprintln!(
                "[gaudium] recipe {} cache_hit={} ws={:.2} MiB ({:.2}s)",
                r.name,
                r.from_cache,
                r.ws_size as f64 / (1024.0 * 1024.0),
                t0.elapsed().as_secs_f64()
            );
        }
        rt.max_ws = r.ws_size;
        rt.recipes.insert(rows, r);
        Ok(rt)
    }

    /// After the allocation seal: binds the workspace and zeroes the layer scratch rows the
    /// padded recipes read beyond a short tile.
    pub(super) fn finish(
        &mut self,
        dev: &HpuDevice,
        spec: &ModelSpec,
        prefill_tile: u64,
        scratch: [&BTreeMap<String, u64>; 2],
        stage_zero: &PinnedBuf,
    ) -> Result<(), String> {
        if !self.on {
            return Ok(());
        }
        for r in self.recipes.values_mut() {
            r.bind_shared_workspace(dev)
                .map_err(|e| format!("bind {} ws: {e}", r.name))?;
        }
        let hid = spec.hidden;
        let ek = spec.experts_per_tok;
        for map in scratch {
            for (_, src) in moe::ROUTED_PREFILL_INPUTS {
                let Some(&addr) = map.get(src) else {
                    continue; // a model without this layer kind
                };
                let row_bytes = match src {
                    "L_rid" => ek * 4,
                    "L_rwb" => ek * 2,
                    _ => hid * 2,
                };
                dev.zero_hbm_chunked(addr, row_bytes * prefill_tile, stage_zero)
                    .map_err(|e| format!("zero routed prefill scratch {src}: {e}"))?;
            }
        }
        Ok(())
    }
}

/// Makes the routed-experts recipe of a tile of `tile_len` rows launchable (compiled into the
/// shared workspace on first use of its padded row count). No-op unless the engine runs the
/// routed MoE form.
pub(super) fn ensure(eng: &mut Qwen3Engine, tile_len: u64) -> Result<(), String> {
    if !eng.moe_rx.on {
        return Ok(());
    }
    let rows = moe::routed_prefill_rows(tile_len, eng.prefill_tile);
    if eng.moe_rx.recipes.contains_key(&rows) {
        return Ok(());
    }
    let mut r = moe::build_prefill_routed_graph(&eng.spec, rows)
        .compile(Arc::clone(eng.dev.api()), &eng.cache, &eng.driver_version)
        .map_err(|e| format!("compile routed-experts prefill T={rows}: {e}"))?;
    r.bind_shared_workspace(&eng.dev)
        .map_err(|e| format!("bind routed-experts prefill T={rows} ws: {e}"))?;
    eng.moe_rx.recipes.insert(rows, r);
    Ok(())
}

impl RoutedPrefill {
    /// After layer `l`'s prefill recipe of a tile: the layer's routed experts and tail, written
    /// to `xout`. `scratch` = the scratch table of the layer's kind. No-op unless the engine runs
    /// the routed MoE form.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_layer(
        &self,
        dev: &HpuDevice,
        scratch: &BTreeMap<String, u64>,
        weights_addr: &BTreeMap<String, u64>,
        prefill_tile: u64,
        l: usize,
        tile_len: u64,
        xout: u64,
    ) -> Result<(), String> {
        if !self.on {
            return Ok(());
        }
        let rows = moe::routed_prefill_rows(tile_len, prefill_tile);
        let recipe = self
            .recipes
            .get(&rows)
            .ok_or_else(|| format!("routed-experts prefill recipe T={rows} was not prepared"))?;
        let mut pool = BTreeMap::new();
        for (name, src) in moe::ROUTED_PREFILL_INPUTS {
            let &addr = scratch
                .get(src)
                .ok_or_else(|| format!("routed prefill: layer scratch has no '{src}'"))?;
            pool.insert(name.to_string(), addr);
        }
        for key in ["EW12", "EW3"] {
            let real = format!("L{l}_{key}");
            let &addr = weights_addr
                .get(&real)
                .ok_or_else(|| format!("routed prefill: no stacked expert tensor '{real}'"))?;
            pool.insert(format!("L_{key}"), addr);
        }
        pool.insert("L_out".to_string(), xout);
        let bind = select_exact_bindings(recipe, &pool)?;
        recipe
            .launch_checked(dev, &bind)
            .map_err(|e| format!("launch routed experts of prefill layer {l} T={rows}: {e}"))
    }
}
