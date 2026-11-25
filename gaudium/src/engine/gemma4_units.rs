//! The prefill of a prompt with images (child module of `gemma4`).
//!
//! - **Image rows** take their soft tokens: a tile with image rows runs the embedding stage
//!   `g_g4_embed_vis_T<t>` (the token embedding times a keep flag, plus a gather from the feature
//!   buffer), and its ids carry the pad id at image rows (what the small sizes' per-layer inputs
//!   gather there, as the reference does).
//! - **Two-way image blocks** (the sizes with `use_bidirectional_attention: "vision"`): in a
//!   sliding layer a row of an image block attends the rest of its block as well. A tile whose
//!   block runs into later tiles prefills with them as one *unit*, layer by layer: at every
//!   sliding layer the later tiles' keys go into the ring first (`g_g4_skv_T<t>`), then every tile
//!   of the unit runs the layer with the ring softmax that extends each row's window to the end of
//!   its block (`g_g4_sx_T<t>`, `ext`). Global layers stay causal and run tile by tile. A unit
//!   never holds more rows than the ring keeps beside the window (`g4u_fits`, checked at load for
//!   the largest unit), so the ring stays the text ring and the decode step is unchanged. A tile of
//!   a unit is final when every block over it ends inside the unit (`g4u_final`); the next unit
//!   starts at the first tile that is not.
//!
//! Positions stay `0, 1, 2, ...` in this family (no multi-axis positions): a row's RoPE position
//! is its KV slot. Text-only requests never come here and launch what they launched before.

use std::time::Instant;

use super::recipes::{embed_recipe, layer_recipes, pf_graph, tile_chunks, tile_has_image, PfRecipe};
use super::vision::graphs::build_embed_vis_graph;
use super::*;
use crate::verified::{
    g4u_cross, g4u_ext, g4u_final, g4u_fits, g4u_in_block, g4u_join, g4u_overlaps, g4u_two_way, g4u_unit_tiles, g4v_add_part,
    ikv_feat_row,
};

/// The vision recipes of the decoder, by kind and tile rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum VisRecipe {
    Embed,
    SlidingExt,
    SlidingKv,
}

/// What the decoder's plan holds for image input.
#[derive(Debug, Clone)]
pub(super) struct VisLoad {
    pub geo: vision::run::G4vGeo,
    /// Image rows attend their block both ways in the sliding layers.
    pub bidir: bool,
    /// Tiles a unit may hold (1 without two-way blocks).
    pub unit_tiles: u64,
}

impl VisLoad {
    /// The decoder side of the image path for tiles of `tile` rows, or the reason the
    /// configuration cannot take images.
    pub fn new(geo: vision::run::G4vGeo, spec: &ModelSpec, d: &G4Dims, tile: u64) -> Result<Self, String> {
        let bidir = spec
            .gemma4
            .as_ref()
            .is_some_and(|g| g.bidirectional_vision);
        let mut unit_tiles = 1;
        if bidir {
            if d.first_shared < d.n_layers {
                return Err("two-way image blocks with layers that share keys are not implemented".into());
            }
            // the tiles the largest block reaches from any row of a tile (its first tile included)
            let block = super::vision::soft_budget(&geo.spec);
            unit_tiles = g4u_unit_tiles(tile, block);
            if !g4u_fits(d.ring, d.window, unit_tiles * tile) {
                return Err(format!(
                    "an image block of {block} rows reaches {unit_tiles} prefill tiles of {tile} rows: with a window of {} that needs more than the {}-slot ring",
                    d.window, d.ring
                ));
            }
        }
        Ok(Self {
            geo,
            bidir,
            unit_tiles,
        })
    }

    /// The decoder graphs of the image path at the planned tile (for the plan's scratch sizes).
    pub fn sizing_graphs(&self, d: &G4Dims, tile: u64, lmax: u64) -> Vec<GraphBuilder> {
        let mut v = vec![build_embed_vis_graph(d, self.geo.feat_cols, tile, &[])];
        if self.bidir {
            v.push(build_sliding_vis_graph(d, tile, lmax, false, &[]));
            v.push(build_sliding_vis_graph(d, tile, lmax, true, &[]));
        }
        v
    }
}

/// Arena bytes of the buffers of `extra` unit tiles beyond the first: two layer buffers of
/// `x_bytes`, ids and extension table of `ids_bytes`, a cursor (each as its sub-allocation takes
/// it, `g4v_add_part`).
pub(super) fn unit_bytes(extra: u64, x_bytes: u64, ids_bytes: u64) -> u64 {
    let mut t = 0u64;
    for _ in 0..extra {
        for b in [x_bytes, x_bytes, ids_bytes, PF_CURSOR_BYTES, ids_bytes] {
            t = g4v_add_part(t, b);
        }
    }
    t
}

/// One prefill tile of an image request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Tile {
    pub pos: u64,
    pub t: u64,
}

impl Tile {
    pub fn end(&self) -> u64 {
        self.pos + self.t
    }
}

/// The prompt's tiles of at most `tile` rows.
pub(super) fn split_tiles(len: u64, tile: u64) -> Vec<Tile> {
    let mut tiles = Vec::new();
    let mut pos = 0u64;
    while pos < len {
        let t = (len - pos).min(tile);
        tiles.push(Tile { pos, t });
        pos += t;
    }
    tiles
}

/// The largest end of the image blocks (`(first row, rows)`) over the rows `[a, b)` (0: none).
pub(super) fn block_end(blocks: &[(usize, usize)], a: u64, b: u64) -> u64 {
    blocks
        .iter()
        .filter(|&&(s, n)| g4u_overlaps(s as u64, n as u64, a, b))
        .map(|&(s, n)| (s + n) as u64)
        .max()
        .unwrap_or(0)
}

/// One prefill unit: its tiles (indices), the end of its rows, whether it runs the two-way
/// form, and the tile the next unit starts at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Unit {
    pub tiles: Vec<usize>,
    pub end: u64,
    pub two_way: bool,
    pub next: usize,
}

/// The unit that starts at tile `i`: the tile, and the later tiles an image block runs into
/// (`g4u_cross`, while the plan holds buffers for them, `g4u_join`); it runs the two-way form
/// when a block lies over one of its tiles (`g4u_two_way`). The next unit starts at the first of
/// its tiles that is not final (`g4u_final`), else behind it. Refused when even the first tile
/// is not final (a block longer than a unit reaches).
pub(super) fn plan_unit(
    tiles: &[Tile],
    blocks: &[(usize, usize)],
    i: usize,
    bidir: bool,
    max_tiles: u64,
) -> Result<Unit, String> {
    let mut unit = vec![i];
    while let Some(&last) = unit.last() {
        let end = tiles[last].end();
        // the blocks over the unit's last row (a tile holds at least one row)
        let cross = g4u_cross(bidir, last as u64 + 1, tiles.len() as u64, block_end(blocks, end - 1, end), end);
        if !g4u_join(cross, unit.len() as u64, max_tiles) {
            break;
        }
        unit.push(last + 1);
    }
    let end = tiles[*unit.last().expect("one tile")].end();
    let over = unit
        .iter()
        .map(|&j| block_end(blocks, tiles[j].pos, tiles[j].end()))
        .max()
        .unwrap_or(0);
    let two_way = g4u_two_way(bidir, over);
    let next = unit
        .iter()
        .copied()
        .find(|&j| !g4u_final(two_way, block_end(blocks, tiles[j].pos, tiles[j].end()), end))
        .unwrap_or(unit[unit.len() - 1] + 1);
    if next == i {
        return Err(format!(
            "an image block from row {} runs past the {max_tiles} tiles a prefill unit holds",
            tiles[i].pos
        ));
    }
    Ok(Unit {
        tiles: unit,
        end,
        two_way,
        next,
    })
}

/// Positions after its own that the row at `p` attends in a unit ending at `unit_end`.
pub(super) fn row_ext(blocks: &[(usize, usize)], p: u64, unit_end: u64) -> u64 {
    let blk = blocks
        .iter()
        .find(|&&(s, n)| g4u_in_block(s as u64, n as u64, p));
    g4u_ext(blk.is_some(), p, blk.map_or(0, |&(s, n)| (s + n) as u64), unit_end)
}

/// Buffers of the `j`-th tile of a unit: its two layer buffers, ids, cursor, extension table.
#[derive(Debug, Clone, Copy)]
pub(super) struct TileBufs {
    pub x: [u64; 2],
    pub ids: u64,
    pub cursor: u64,
    pub ext: u64,
}

impl Gemma4Engine {
    fn unit_bufs(&self, j: usize) -> TileBufs {
        if j == 0 {
            // `ext` is planned whenever two-way blocks are (`VisLoad::sizing_graphs`); 0 (none)
            // is refused by `prefill_vis` before it uploads a table
            TileBufs {
                x: self.pf_x,
                ids: self.pf_ids,
                cursor: self.pf_cursor,
                ext: self.pf_scratch.get("ext").copied().unwrap_or(0),
            }
        } else {
            self.pf_unit[j - 1]
        }
    }

    pub(super) fn ensure_vis(&mut self, kind: VisRecipe, t: u64) -> Result<(), String> {
        if self.vis_recipes.contains_key(&(kind, t)) {
            return Ok(());
        }
        let v = self.vision.as_ref().ok_or("no image input")?;
        let gb = pf_graph(
            &self.dims,
            self.max_ctx,
            &self.elf_hashes,
            Some((v.geo.feat_cols, v.elf())),
            PfRecipe::Vis(kind, t),
        )?;
        let r = self.compile(&gb, &format!("vision {kind:?} T={t}"))?;
        self.vis_recipes.insert((kind, t), r);
        Ok(())
    }

    /// The binding pool of layer `l` of a prefill tile (its weights under their generic names,
    /// the K / V it attends, the RoPE tables) with the tile's buffers.
    fn layer_pool(&self, l: usize, x_in: u64, x_out: u64, b: &TileBufs) -> BTreeMap<String, u64> {
        let d = &self.dims;
        let glob = d.is_global(l);
        let mut pool = self.pf_scratch.clone();
        pool.insert("x_in".into(), x_in);
        pool.insert("L_out".into(), x_out);
        pool.insert("cursor".into(), b.cursor);
        pool.insert("ext".into(), b.ext);
        let (rc, rs) = rope_names(glob);
        pool.insert(rc.into(), self.weights_addr[rc]);
        pool.insert(rs.into(), self.weights_addr[rs]);
        let lp = format!("L{l}_");
        for (name, &addr) in self
            .weights_addr
            .range(lp.clone()..)
            .take_while(|(name, _)| name.starts_with(&lp))
        {
            pool.insert(format!("L_{}", &name[lp.len()..]), addr);
        }
        let src = d.kv_source(l);
        let (kk, vk) = kv_names(glob);
        pool.insert(format!("L_{kk}"), self.kv_addr[&format!("L{src}_{kk}")].0);
        pool.insert(format!("L_{vk}"), self.kv_addr[&format!("L{src}_{vk}")].0);
        pool
    }

    fn launch_pool(&self, r: &CompiledRecipe, pool: &BTreeMap<String, u64>, what: &str) -> Result<(), String> {
        let bind = select_exact_bindings(r, pool)?;
        r.launch_checked(&self.dev, &bind)
            .map_err(|e| format!("launch {what}: {e}"))
    }

    /// The embedding stage of tile `tl` into `b.x[0]`: the plain recipe for a text tile, the
    /// vision recipe for a tile with image rows (ids with the pad id there, feature columns,
    /// keep flags). Synced (the next tile's inputs reuse the staging buffers).
    fn embed_tile(&mut self, ids: &[i32], fidx: &[i32], tl: Tile, b: &TileBufs) -> Result<f64, String> {
        let (p0, t) = (tl.pos as usize, tl.t as usize);
        let text_col = ikv_feat_row(false, 0) as i32;
        let has_img = tile_has_image(fidx, tl);
        let t_rec = Instant::now();
        self.ensure_pf(embed_recipe(has_img, tl.t))?;
        let recipe_ms = t_rec.elapsed().as_secs_f64() * 1000.0;
        let pad = self.spec.gemma4.as_ref().map_or(0, |g| g.pad_id.max(0));
        {
            let st = self.stage_io.as_mut_slice();
            for k in 0..t {
                let id = if fidx[p0 + k] != text_col { pad } else { ids[p0 + k] };
                st[k * 4..k * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
        }
        self.dev
            .copy_h2d_pinned_sync(&self.stage_io, t * 4, b.ids)
            .map_err(|e| format!("upload prefill ids: {e}"))?;
        self.write_i32_scalar(b.cursor, tl.pos as i32)?;
        let mut pool = self.pf_scratch.clone();
        for name in GLOBAL_WEIGHTS {
            if let Some(&addr) = self.weights_addr.get(*name) {
                pool.insert((*name).to_string(), addr);
            }
        }
        pool.insert("ids".into(), b.ids);
        pool.insert("x0".into(), b.x[0]);
        pool.insert("ple_out".into(), self.pf_ple);
        if has_img {
            let mut fb = Vec::with_capacity(t * 4);
            let mut kb = Vec::with_capacity(t * 2);
            let one = crate::fp8::f32_to_bf16_rne(1.0);
            for &f in &fidx[p0..p0 + t] {
                fb.extend_from_slice(&f.to_le_bytes());
                kb.extend_from_slice(&(if f == text_col { one } else { 0u16 }).to_le_bytes());
            }
            self.dev
                .copy_h2d_sync(&fb, self.pf_scratch["vis_fidx"])
                .map_err(|e| format!("upload feature columns: {e}"))?;
            self.dev
                .copy_h2d_sync(&kb, self.pf_scratch["vis_keep"])
                .map_err(|e| format!("upload keep flags: {e}"))?;
            let r = &self.vis_recipes[&(VisRecipe::Embed, tl.t)];
            self.launch_pool(r, &pool, &format!("vision embed T={t}"))?;
        } else {
            let r = &self.embed_recipes[&tl.t];
            self.launch_pool(r, &pool, &format!("embed T={t}"))?;
        }
        self.sync_compute("prefill embedding")?;
        Ok(recipe_ms)
    }

    /// Prefill of an image request. Returns `(rows of the last tile, address of its last layer's
    /// output, first-use recipe time)`, or `None` when cancelled.
    pub(super) fn prefill_vis(
        &mut self,
        prompt_ids: &[i32],
        is_cancelled: &impl Fn() -> bool,
    ) -> Result<Option<(u64, u64, f64)>, String> {
        let req = self
            .vision
            .as_ref()
            .and_then(|v| v.req.clone())
            .ok_or("an image prefill without an image request")?;
        let n_layers = self.dims.n_layers;
        let kinds = self.dims.kinds();
        let tiles = split_tiles(prompt_ids.len() as u64, self.prefill_tile);
        let bidir = self.vis_load.as_ref().is_some_and(|v| v.bidir);
        let max_tiles = self.vis_load.as_ref().map_or(1, |v| v.unit_tiles);
        let mut recipe_ms = 0.0f64;
        let mut out_addr = 0u64;
        let mut i = 0usize;
        while i < tiles.len() {
            if is_cancelled() {
                return Ok(None);
            }
            let Unit {
                tiles: unit,
                end: unit_end,
                two_way,
                next,
            } = plan_unit(&tiles, &req.blocks, i, bidir, max_tiles)?;
            for (k, &j) in unit.iter().enumerate() {
                let b = self.unit_bufs(k);
                recipe_ms += self.embed_tile(prompt_ids, &req.fidx, tiles[j], &b)?;
            }
            let t_rec = Instant::now();
            for &j in &unit {
                let (t, nch) = (tiles[j].t, tile_chunks(tiles[j].end(), self.max_chunks));
                for r in layer_recipes(&kinds, t, nch, two_way) {
                    self.ensure_pf(r)?;
                }
            }
            recipe_ms += t_rec.elapsed().as_secs_f64() * 1000.0;
            if !two_way {
                // a text tile (or a size whose image rows attend causally): the plain launches
                let tl = tiles[i];
                let nch = tile_chunks(tl.end(), self.max_chunks);
                for l in 0..n_layers {
                    self.launch_prefill_layer(l, tl.t, nch)?;
                }
                self.sync_compute("prefill tile")?;
                out_addr = self.prefill_io(n_layers - 1).1;
                self.live.prefill_progress(tl.end());
                i = next;
                continue;
            }
            // two-way blocks: every row's extension, then the unit layer by layer
            for (k, &j) in unit.iter().enumerate() {
                let tl = tiles[j];
                let mut ext = Vec::with_capacity(tl.t as usize * 4);
                for p in tl.pos..tl.end() {
                    ext.extend_from_slice(&(row_ext(&req.blocks, p, unit_end) as i32).to_le_bytes());
                }
                let b = self.unit_bufs(k);
                if b.ext == 0 {
                    return Err("the plan holds no extension table for two-way image blocks".into());
                }
                self.dev
                    .copy_h2d_sync(&ext, b.ext)
                    .map_err(|e| format!("upload block extensions: {e}"))?;
            }
            for l in 0..n_layers {
                let glob = self.dims.is_global(l);
                if !glob {
                    for (k, &j) in unit.iter().enumerate().skip(1) {
                        let b = self.unit_bufs(k);
                        let pool = self.layer_pool(l, b.x[l % 2], b.x[(l + 1) % 2], &b);
                        let r = &self.vis_recipes[&(VisRecipe::SlidingKv, tiles[j].t)];
                        self.launch_pool(r, &pool, &format!("block keys layer {l}"))?;
                    }
                }
                for (k, &j) in unit.iter().enumerate() {
                    let tl = tiles[j];
                    let b = self.unit_bufs(k);
                    let pool = self.layer_pool(l, b.x[l % 2], b.x[(l + 1) % 2], &b);
                    let r = if glob {
                        let nch = tile_chunks(tl.end(), self.max_chunks);
                        &self.layer_recipes[&(G4LayerKind::Global, tl.t, nch)]
                    } else {
                        &self.vis_recipes[&(VisRecipe::SlidingExt, tl.t)]
                    };
                    self.launch_pool(r, &pool, &format!("image prefill layer {l} at {}", tl.pos))?;
                }
            }
            self.sync_compute("image prefill unit")?;
            let last = unit[unit.len() - 1];
            if next > last {
                let b = self.unit_bufs(unit.len() - 1);
                out_addr = b.x[n_layers % 2];
            }
            self.live.prefill_progress(tiles[next - 1].end());
            i = next;
        }
        let last_t = tiles.last().map_or(0, |t| t.t);
        Ok(Some((last_t, out_addr, recipe_ms)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the planner over a prompt and checks what the prefill relies on: every unit fits the
    /// ring, never holds more tiles than planned, makes progress, every row's extension stays in
    /// its unit and inside its own block, and every tile is eventually run in a unit where it
    /// is final.
    fn walk(len: u64, tile: u64, blocks: &[(usize, usize)], ring: u64, w: u64) -> Vec<Unit> {
        let tiles = split_tiles(len, tile);
        let block = blocks.iter().map(|b| b.1 as u64).max().unwrap_or(1);
        let max_tiles = g4u_unit_tiles(tile, block);
        assert!(g4u_fits(ring, w, max_tiles * tile));
        let mut done = vec![false; tiles.len()];
        let mut units = Vec::new();
        let mut i = 0;
        while i < tiles.len() {
            let u = plan_unit(&tiles, blocks, i, true, max_tiles).unwrap();
            assert!(u.tiles.len() as u64 <= max_tiles);
            assert!(u.next > i);
            let rows = u.end - tiles[u.tiles[0]].pos;
            assert!(g4u_fits(ring, w, rows));
            for &j in &u.tiles {
                for p in tiles[j].pos..tiles[j].end() {
                    let e = row_ext(blocks, p, u.end);
                    assert!(p + e < u.end);
                    if let Some(&(s, n)) = blocks.iter().find(|&&(s, n)| s as u64 <= p && p < (s + n) as u64) {
                        assert!(p + e < (s + n) as u64);
                        // a final tile's rows see their whole block
                        if j < u.next {
                            assert_eq!(p + e + 1, (s + n) as u64, "row {p}");
                        }
                    } else {
                        assert_eq!(e, 0);
                    }
                }
                if j < u.next {
                    done[j] = true;
                }
            }
            i = u.next;
            units.push(u);
        }
        assert!(done.iter().all(|&d| d));
        units
    }

    #[test]
    fn one_image_over_a_tile_boundary_is_one_unit() {
        // 298 rows: a block of 260 rows from row 5 crosses the boundary of 256-row tiles
        let u = walk(298, 256, &[(5, 260)], 2048, 1024);
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tiles, vec![0, 1]);
        assert!(u[0].two_way);
    }

    #[test]
    fn small_tiles_and_back_to_back_images() {
        // blocks over three 128-row tiles each, and two images with one text row between
        walk(1200, 128, &[(100, 280), (382, 280), (700, 266)], 2048, 1024);
        walk(1200, 16, &[(3, 280), (290, 280)], 2048, 1024);
        let u = walk(600, 256, &[(10, 200)], 2048, 1024);
        // a block inside the first tile: units of one tile
        assert!(u.iter().all(|u| u.tiles.len() == 1));
    }

    #[test]
    fn a_text_prompt_never_runs_two_way() {
        let u = walk(1000, 256, &[], 2048, 1024);
        assert!(u.iter().all(|u| !u.two_way && u.tiles.len() == 1));
    }

    #[test]
    fn causal_sizes_move_tile_by_tile_over_a_block_boundary() {
        // a size whose image rows attend causally: one tile per unit, every tile final, also
        // when an image block runs over the tile's end, and whatever tile buffers are planned
        let tiles = split_tiles(700, 256);
        let blocks = [(200, 280)];
        for max_tiles in [1, 3] {
            let mut i = 0;
            let mut seen = 0;
            while i < tiles.len() {
                let u = plan_unit(&tiles, &blocks, i, false, max_tiles).unwrap();
                assert_eq!(u.tiles, vec![i]);
                assert!(!u.two_way);
                assert_eq!(u.next, i + 1);
                i = u.next;
                seen += 1;
            }
            assert_eq!(seen, tiles.len());
        }
    }

    #[test]
    fn unit_tiles_cover_the_largest_block() {
        for (t, b) in [(256, 280), (256, 1), (128, 280), (16, 280), (1, 0), (1, 280), (512, 280)] {
            let n = g4u_unit_tiles(t, b);
            assert!(n >= 1);
            // a block of b rows from the last row of a tile ends inside n tiles
            assert!(t - 1 + b <= n * t, "t {t} b {b}: {n} tiles");
        }
    }

    #[test]
    fn a_block_longer_than_the_planned_unit_is_refused() {
        let tiles = split_tiles(2000, 128);
        assert!(plan_unit(&tiles, &[(10, 900)], 0, true, 3).is_err());
    }
}
