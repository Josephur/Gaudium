//! Prefill with two tiles in flight (the wavefront): a child module of `engine::qwen3`.
//!
//! A tile walks the layers as a chain of half-steps (first half of layer 0, second half of layer
//! 0, first half of layer 1, ...). The single-tile loop launches one whole layer at a time, and
//! on the card that layer is a serial chain: the MME idles while the TPCs build the DeltaNet
//! coefficients or sweep the softmax, the TPCs idle under the weight GEMMs. Here the full tiles
//! of a prompt run in pairs: the second tile of a pair trails the first by one half-step, so
//! every launch holds the second half of one tile's layer and the first half of the other's, in
//! a recipe whose node order lets each engine alternate between the two bodies
//! (`prefill::build_wave_graph`).
//!
//! The two bodies are the halves of the single-tile recipes, node for node, on the same rows:
//! the states, caches and rows this loop leaves are the ones the single-tile loop leaves.
//!
//! Which half-step runs in which launch, and which tiles pair up, is the verified schedule
//! (`verified/prefill_wave.bend`: `wave_tiles`, `wave_step`, `wave_set`); this module only turns
//! it into bindings and launches. Tiles the schedule does not take (an odd full tile, the
//! partial last tile) stay with the single-tile loop, which continues where this one stops.

use super::prefill::{FirstHalf, WaveBufs, WaveKey, SECOND_PFX};
use crate::vision::PromptLayout;
use super::*;
use crate::verified::{wave_set, wave_step, wave_tiles};

/// Generic per-layer weight suffixes a half-layer recipe may bind (both layer kinds).
const LAYER_WEIGHTS: [&str; 18] = [
    "w_in", "w_post", "W_qg", "W_k", "W_v", "qn_w", "kn_w", "W_o", "W_gate", "W_up", "W_down",
    "W_qkvz", "W_ba", "conv_w", "conv_b", "norm_w", "A_log", "dt_bias",
];

/// What the wavefront did for a request.
pub(super) struct WaveEnd {
    /// Tiles it prefilled (the first `tiles` full tiles from the row the prefill started at).
    pub tiles: usize,
    /// Device rows of the last of them after the last layer.
    pub final_x: u64,
    pub cancelled: bool,
}

/// Buffer set of tile `tile` of the wavefront.
fn set_of(tile: usize) -> usize {
    wave_set(tile % 2 == 1) as usize
}

/// Input and output rows of layer `l` for a tile on buffer set `set` (the drafter's tapped
/// layers write their tap rows instead, and the next layer reads them from there).
fn layer_io(eng: &Qwen3Engine, bufs: &WaveBufs, set: usize, l: usize) -> (u64, u64) {
    let (xin, xout) = if l.is_multiple_of(2) {
        (bufs.x[set][0], bufs.x[set][1])
    } else {
        (bufs.x[set][1], bufs.x[set][0])
    };
    match &eng.specdec {
        Some(sd) if set == 0 => sd.prefill_io(l, xin, xout),
        Some(sd) => sd.prefill_io_set1(l, xin, xout, &bufs.taps1),
        None => (xin, xout),
    }
}

/// The bindings every launch of recipe `key` shares: its scratch and the constant tables.
fn base_pool(eng: &Qwen3Engine, key: WaveKey) -> BTreeMap<String, u64> {
    let mut pool = eng.pf.wave_pool(key);
    for name in ["zero_slot", "norm_ones", "rope_c", "rope_s"] {
        if let Some(&a) = eng.weights_addr.get(name) {
            pool.insert(name.to_string(), a);
        }
    }
    pool
}

/// Binds layer `l`'s weights under the generic prefix `pfx` (`L_` / `G_`).
fn bind_weights(eng: &Qwen3Engine, pool: &mut BTreeMap<String, u64>, l: usize, pfx: &str) {
    for suffix in LAYER_WEIGHTS {
        if let Some(&a) = eng.weights_addr.get(&format!("L{l}_{suffix}")) {
            pool.insert(format!("{pfx}{suffix}"), a);
        }
    }
    // [moe/fp8 hook] FP8 weight scales and site cast scales of layer l
    if pfx == "L_" {
        crate::engine::moe::bind_ext_layer_weights(&eng.weights_addr, l, pool);
    } else {
        let mut ext = BTreeMap::new();
        crate::engine::moe::bind_ext_layer_weights(&eng.weights_addr, l, &mut ext);
        for (k, a) in ext {
            pool.insert(format!("{pfx}{}", &k[2..]), a);
        }
    }
}

/// Binds the first half of layer `l` of tile `tile` (body prefix `L_`). `start` = whole tiles
/// of the prompt before the wavefront's first tile.
fn bind_first(
    eng: &Qwen3Engine,
    bufs: &WaveBufs,
    pool: &mut BTreeMap<String, u64>,
    start: usize,
    tile: usize,
    l: usize,
    gdn_parity: usize,
) -> Result<(), String> {
    let set = set_of(tile);
    let (xin, _) = layer_io(eng, bufs, set, l);
    pool.insert("x_in".into(), xin);
    pool.insert("L_core".into(), bufs.core[set]);
    pool.insert("cursor".into(), bufs.cursor[set]);
    bind_weights(eng, pool, l, "L_");
    if eng.spec.is_full_attention_layer(l) {
        for (k, name) in [("L_kcache", "kcache"), ("L_vcache", "vcache")] {
            let &(a, _) = eng
                .kv_addr
                .get(&format!("L{l}_{name}"))
                .ok_or_else(|| format!("prefill wavefront: layer {l} has no {name}"))?;
            pool.insert(k.into(), a);
        }
    } else {
        // the state alternates between its two buffers from tile to tile, as in the
        // single-tile loop; a tile at a kept position reads or writes that position's slot
        let st = eng
            .gdn_state
            .get(&l)
            .ok_or_else(|| format!("prefill wavefront: layer {l} has no recurrent state"))?;
        let p = (gdn_parity + tile) % 2;
        let io = prefix_cache::state_io(
            st,
            &eng.prefix_slots,
            &eng.prefix_plan,
            eng.prefill_tile,
            l,
            (start + tile) as u64,
            p,
        )?;
        pool.insert("L_S_in".into(), io.s_in);
        pool.insert("L_S_out".into(), io.s_out);
        pool.insert("L_cs_in".into(), io.cs_in);
        pool.insert("L_cs_out".into(), io.cs_out);
    }
    Ok(())
}

/// Binds the second half of layer `l` of tile `tile` (body prefix `G_`).
fn bind_second(
    eng: &Qwen3Engine,
    bufs: &WaveBufs,
    pool: &mut BTreeMap<String, u64>,
    tile: usize,
    l: usize,
) {
    let set = set_of(tile);
    let (xin, xout) = layer_io(eng, bufs, set, l);
    pool.insert(format!("{SECOND_PFX}x_in"), xin);
    pool.insert(format!("{SECOND_PFX}core"), bufs.core[set]);
    pool.insert(format!("{SECOND_PFX}out"), xout);
    bind_weights(eng, pool, l, SECOND_PFX);
}

/// Starts tile `tile`: waits until the tile that last used its buffer set is done, uploads its
/// ids and cursor, launches the embedding and splices the vision rows.
fn begin_tile(
    eng: &mut Qwen3Engine,
    bufs: &WaveBufs,
    prompt_ids: &[i32],
    start: usize,
    tile: usize,
    vis_layout: Option<&PromptLayout>,
    pf_launch: &mut u64,
) -> Result<(), String> {
    let t = eng.prefill_tile as usize;
    let set = set_of(tile);
    let pos = (start + tile) * t;
    let t_wait = prefill::now();
    unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
        .map_err(|e| format!("sync before prefill tile at pos={pos}: {e}"))?;
    eng.pf.timers.wait_ms += prefill::ms_since(t_wait);

    let t_up = prefill::now();
    let tile_bytes = t * 4;
    for (k, &id) in prompt_ids[pos..pos + t].iter().enumerate() {
        eng.stage_io.as_mut_slice()[k * 4..k * 4 + 4].copy_from_slice(&id.to_le_bytes());
    }
    let cur_off = tile_bytes.next_multiple_of(64);
    eng.stage_io.as_mut_slice()[cur_off..cur_off + 4].copy_from_slice(&(pos as i32).to_le_bytes());
    eng.dev
        .copy_h2d_pinned_batch_sync(
            &eng.stage_io,
            &[(0, tile_bytes, bufs.ids[set]), (cur_off, 4, bufs.cursor[set])],
        )
        .map_err(|e| format!("upload prefill ids / cursor: {e}"))?;
    eng.pf.timers.upload_ms += prefill::ms_since(t_up);

    let t_submit = prefill::now();
    let mut emb_pool = BTreeMap::new();
    emb_pool.insert("embed_table".to_string(), eng.weights_addr["embed_table"]);
    emb_pool.insert("ids".to_string(), bufs.ids[set]);
    emb_pool.insert("x0".to_string(), bufs.x[set][0]);
    let emb_recipe = eng
        .embed_recipes
        .get(&(t as u64))
        .ok_or_else(|| format!("prefill wavefront: no embed recipe of {t} rows"))?;
    let emb_bind = select_exact_bindings(emb_recipe, &emb_pool)?;
    prof_rt::before_prefill_launch(&mut eng.prof, &eng.dev, *pf_launch)?;
    *pf_launch += 1;
    emb_recipe
        .launch_checked(&eng.dev, &emb_bind)
        .map_err(|e| format!("launch embed T={t}: {e}"))?;
    if let Some(layout) = vis_layout {
        // image-pad rows of this tile take the tower's embeddings (device copy)
        eng.vision_splice_tile(layout, pos, t, bufs.x[set][0])?;
    }
    eng.pf.timers.submit_ms += prefill::ms_since(t_submit);
    Ok(())
}

/// Ends tile `tile` (its last half-step is launched): the drafter's context pass over the
/// tile's tap rows, and the progress record.
fn end_tile(
    eng: &mut Qwen3Engine,
    bufs: &WaveBufs,
    start: usize,
    tile: usize,
) -> Result<(), String> {
    let t = eng.prefill_tile;
    let pos = (start + tile) as u64 * t;
    let n_layers = eng.spec.n_layers;
    if eng.specdec.is_some() {
        if set_of(tile) == 0 {
            spec::prefill_tile_hook(eng, pos, t)?;
        } else {
            let mut over = BTreeMap::new();
            over.insert("ctx_cursor".to_string(), bufs.cursor[1]);
            for (i, &a) in bufs.taps1.iter().enumerate() {
                over.insert(format!("df_tap{i}"), a);
            }
            spec::prefill_tile_hook_set1(eng, pos, t, &over)?;
        }
    }
    eng.pf.timers.tiles += 1;
    eng.pf.timers.launches += n_layers + 1;
    let end_pos = pos + t;
    let nch = end_pos.div_ceil(KV_CHUNK).clamp(1, eng.max_chunks);
    let has_gdn = !eng.gdn_state.is_empty();
    eng.live.prefill_tile(end_pos, n_layers + 1, (t << 16) | nch, || {
        // [prefill] the byte model of the tile forms (the wavefront launches the same nodes)
        let attn = (eng.spec.n_full_attention_layers() > 0)
            .then(|| eng.pf.attn_graph(&eng.spec, t, eng.max_ctx, nch, &[]));
        let lin = has_gdn.then(|| eng.pf.lin_graph(&eng.spec, t, &[]));
        qwen3_traffic::prefill_tile_of(&eng.spec, t, nch, attn.as_ref(), lin.as_ref())
    });
    Ok(())
}

/// One launch of the wavefront: its first-half body and its second-half body, each
/// `(tile, layer)`.
type Launch = [Option<(usize, usize)>; 2];

/// The launches of `n_wave` tiles of `n_layers` layers, in order: the verified schedule
/// (`wave_step`) laid out as a table. A launch holds at most one first half and one second half;
/// a schedule that put two of a kind into one launch is refused (the recipes hold one of each).
fn launch_table(n_wave: usize, n_layers: usize) -> Result<Vec<Launch>, String> {
    let period = 2 * n_layers as u64;
    if n_wave == 0 || period == 0 {
        return Ok(Vec::new());
    }
    let step_of = |tile: usize, half: u64| wave_step((tile / 2) as u64, tile % 2 == 1, half, period);
    let n_launch = (0..n_wave).map(|i| step_of(i, period - 1)).max().unwrap_or(0) as usize + 1;
    let mut table: Vec<Launch> = vec![[None; 2]; n_launch];
    for tile in 0..n_wave {
        for half in 0..period {
            let slot = &mut table[step_of(tile, half) as usize][(half % 2) as usize];
            if slot.is_some() {
                return Err(format!(
                    "prefill wavefront: two tiles claim the same half of launch {}",
                    step_of(tile, half)
                ));
            }
            *slot = Some((tile, (half / 2) as usize));
        }
    }
    Ok(table)
}

/// Prefills the tiles of `prompt_ids` that the verified schedule runs two in flight, from row
/// `resume` (a whole number of tiles: 0, or the kept position a request resumes from).
/// `Ok(None)`: this request uses the single-tile loop from there.
pub(super) fn run(
    eng: &mut Qwen3Engine,
    prompt_ids: &[i32],
    resume: usize,
    gdn_parity: usize,
    vis_layout: Option<&PromptLayout>,
    is_cancelled: &impl Fn() -> bool,
    pf_launch: &mut u64,
) -> Result<Option<WaveEnd>, String> {
    let t = eng.prefill_tile;
    let n_layers = eng.spec.n_layers as usize;
    let ready = eng.pf.wave_ready()
        && !eng.wave_recipes.is_empty()
        && eng.calib.is_none()
        && !eng.pf.opts.profile
        && t > 0;
    let full = ((prompt_ids.len().saturating_sub(resume)) as u64)
        .checked_div(t)
        .unwrap_or(0);
    let n_wave = wave_tiles(full, ready) as usize;
    let Some(bufs) = eng.pf.wave_bufs.clone().filter(|_| n_wave > 0) else {
        return Ok(None);
    };
    let start = (resume as u64).checked_div(t).unwrap_or(0) as usize;
    let table = launch_table(n_wave, n_layers)?;
    // every state binding of these tiles, checked before the first launch
    for tile in 0..n_wave {
        for (&l, st) in &eng.gdn_state {
            prefix_cache::state_io(
                st,
                &eng.prefix_slots,
                &eng.prefix_plan,
                t,
                l,
                (start + tile) as u64,
                (gdn_parity + tile) % 2,
            )?;
        }
    }

    let mut pools: HashMap<WaveKey, BTreeMap<String, u64>> = HashMap::new();
    let mut done = 0usize;
    for bodies in &table {
        let [first, second] = *bodies;
        if let Some((tile, 0)) = first {
            if is_cancelled() {
                unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
                    .map_err(|e| format!("sync cancelled prefill: {e}"))?;
                return Ok(Some(WaveEnd {
                    tiles: done,
                    final_x: 0,
                    cancelled: true,
                }));
            }
            begin_tile(eng, &bufs, prompt_ids, start, tile, vis_layout, pf_launch)?;
        }
        let t_submit = prefill::now();
        let key = WaveKey {
            first: first.map(|(tile, l)| {
                if eng.spec.is_full_attention_layer(l) {
                    let end_pos = ((start + tile) as u64 + 1) * t;
                    FirstHalf::Attn(end_pos.div_ceil(KV_CHUNK).clamp(1, eng.max_chunks))
                } else {
                    FirstHalf::Lin
                }
            }),
            second: second.is_some(),
        };
        let pool = pools.entry(key).or_insert_with(|| base_pool(eng, key));
        if let Some((tile, l)) = first {
            bind_first(eng, &bufs, pool, start, tile, l, gdn_parity)?;
        }
        if let Some((tile, l)) = second {
            bind_second(eng, &bufs, pool, tile, l);
        }
        let recipe = eng
            .wave_recipes
            .get(&key)
            .ok_or_else(|| format!("prefill wavefront: no recipe for {key:?}"))?;
        let bind = select_exact_bindings(recipe, pool)?;
        prof_rt::before_prefill_launch(&mut eng.prof, &eng.dev, *pf_launch)?;
        *pf_launch += 1;
        recipe
            .launch_checked(&eng.dev, &bind)
            .map_err(|e| format!("launch prefill wavefront {}: {e}", recipe.name))?;
        eng.pf.timers.submit_ms += prefill::ms_since(t_submit);
        if let Some((tile, l)) = second {
            if l + 1 == n_layers {
                let t_end = prefill::now();
                end_tile(eng, &bufs, start, tile)?;
                eng.pf.timers.submit_ms += prefill::ms_since(t_end);
                done += 1;
            }
        }
    }
    let t_wait = prefill::now();
    unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
        .map_err(|e| format!("sync prefill wavefront: {e}"))?;
    eng.pf.timers.wait_ms += prefill::ms_since(t_wait);
    if done != n_wave {
        return Err(format!(
            "prefill wavefront: {done} of {n_wave} tiles finished their last layer"
        ));
    }
    let final_x = layer_io(eng, &bufs, set_of(n_wave - 1), n_layers - 1).1;
    Ok(Some(WaveEnd {
        tiles: n_wave,
        final_x,
        cancelled: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_table_runs_every_half_once_in_prompt_order() {
        for (n_wave, n_layers) in [(2usize, 1usize), (2, 4), (4, 3), (6, 64)] {
            let table = launch_table(n_wave, n_layers).unwrap();
            // where each half-step of each tile runs
            let mut at = vec![vec![usize::MAX; 2 * n_layers]; n_wave];
            for (s, launch) in table.iter().enumerate() {
                for (kind, body) in launch.iter().enumerate() {
                    if let Some((tile, layer)) = *body {
                        assert_eq!(at[tile][2 * layer + kind], usize::MAX, "a half-step ran twice");
                        at[tile][2 * layer + kind] = s;
                    }
                }
                // two bodies of one launch belong to different tiles
                if let [Some((a, _)), Some((b, _))] = *launch {
                    assert_ne!(a, b);
                }
            }
            for tile in 0..n_wave {
                for h in 0..2 * n_layers {
                    assert_ne!(at[tile][h], usize::MAX, "tile {tile} half-step {h} never ran");
                    // a tile's half-steps run in order
                    if h > 0 {
                        assert!(at[tile][h] > at[tile][h - 1]);
                    }
                    // and after the tile in front of it ran the same half-step
                    if tile > 0 {
                        assert!(at[tile][h] > at[tile - 1][h]);
                    }
                }
            }
            // today's schedule: one launch per half-step pair, plus the single halves at both ends
            assert_eq!(table.len(), n_wave * n_layers + 1);
            assert_eq!(table[0], [Some((0, 0)), None]);
            assert_eq!(table[table.len() - 1], [None, Some((n_wave - 1, n_layers - 1))]);
        }
        assert!(launch_table(0, 4).unwrap().is_empty());
    }

    #[test]
    fn tiles_in_flight_use_different_buffer_sets() {
        for tile in 0..8 {
            assert_ne!(set_of(tile), set_of(tile + 1));
            assert!(set_of(tile) < 2);
        }
    }
}
