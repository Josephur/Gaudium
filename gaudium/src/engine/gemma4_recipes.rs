//! The decoder's prefill recipes a request launches, and `--build-cache` (child module of
//! `gemma4`).
//!
//! A load compiles the recipes every request needs (the readout head, the first decode step, the
//! drafter's first steps, the image path at its largest size). Other recipes depend on a
//! request: the rows of its last prefill tile, the tiles that hold image rows, the units of
//! two-way image blocks, the chunk count its context reaches, the patch count of each image.
//! They compile on first use. The prefill picks them through the functions below, and so does
//! `--build-cache`, which compiles them for the requests on its command line without the card.

use std::collections::BTreeSet;

use super::units::{plan_unit, split_tiles, Tile, VisLoad, VisRecipe};
use super::vision::graphs::build_embed_vis_graph;
use super::vision::run::G4Vision;
use super::*;
use crate::engine::BuildRequest;
use crate::verified::ikv_feat_row;

/// A recipe of the decoder's prefill, by kind and tile rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum PfRecipe {
    /// The embedding stage of a text tile.
    Embed(u64),
    /// The final norm over the last tile.
    Final(u64),
    /// A layer kind's recipe: `(kind, rows, chunks a global layer reads)`.
    Layer(G4LayerKind, u64, u64),
    /// The image path's decoder recipes (the embedding of image rows, the two-way sliding layer
    /// and its key write).
    Vis(VisRecipe, u64),
}

/// The key a load keeps a layer recipe under: a sliding layer reads no global chunks, so one
/// recipe serves every chunk count.
pub(super) fn layer_key(kind: G4LayerKind, t: u64, nch: u64) -> (G4LayerKind, u64, u64) {
    (kind, t, if kind.is_global() { nch } else { 0 })
}

impl PfRecipe {
    /// The key a load keeps the recipe under (`layer_key` for a layer).
    fn key(self) -> Self {
        match self {
            Self::Layer(kind, t, nch) => {
                let (k, t, n) = layer_key(kind, t, nch);
                Self::Layer(k, t, n)
            }
            r => r,
        }
    }
}

/// Chunks of the global caches a prefill tile that ends at row `end` reads (the verified
/// `g4u_tile_chunks`: the chunks of its keys, at least one, at most the cache's).
pub(super) fn tile_chunks(end: u64, max_chunks: u64) -> u64 {
    crate::verified::g4u_tile_chunks(end, G4_CHUNK, max_chunks)
}

/// The embedding stage of a tile of `t` rows: the image one when it holds image rows.
pub(super) fn embed_recipe(has_img: bool, t: u64) -> PfRecipe {
    if has_img {
        PfRecipe::Vis(VisRecipe::Embed, t)
    } else {
        PfRecipe::Embed(t)
    }
}

/// The layer recipes of a tile of `t` rows that reads `nch` chunks: one per layer kind, except
/// that a two-way unit runs its sliding layers with the image recipes (the key write and the
/// extended softmax) instead of the plain one.
pub(super) fn layer_recipes(kinds: &[G4LayerKind], t: u64, nch: u64, two_way: bool) -> Vec<PfRecipe> {
    let mut v: Vec<PfRecipe> = kinds
        .iter()
        .filter(|&&k| !(two_way && k == G4LayerKind::Sliding))
        .map(|&k| PfRecipe::Layer(k, t, nch))
        .collect();
    if two_way {
        v.push(PfRecipe::Vis(VisRecipe::SlidingExt, t));
        v.push(PfRecipe::Vis(VisRecipe::SlidingKv, t));
    }
    v
}

/// Whether the tile `tl` holds image rows (`fidx`: every prompt row's feature column).
pub(super) fn tile_has_image(fidx: &[i32], tl: Tile) -> bool {
    let text_col = ikv_feat_row(false, 0) as i32;
    let (p0, t) = (tl.pos as usize, tl.t as usize);
    fidx[p0..p0 + t].iter().any(|&f| f != text_col)
}

/// The graph of a prefill recipe (`elf`: the decoder's kernel fingerprints; `vis`: the feature
/// buffer's columns and the image path's fingerprints, which its decoder recipes carry too).
pub(super) fn pf_graph(
    d: &G4Dims,
    lmax: u64,
    elf: &[(String, String)],
    vis: Option<(u64, &[(String, String)])>,
    r: PfRecipe,
) -> Result<GraphBuilder, String> {
    Ok(match r {
        PfRecipe::Embed(t) => build_embed_graph(d, t, elf),
        PfRecipe::Final(t) => build_final_graph(d, t, elf),
        PfRecipe::Layer(kind, t, nch) => build_layer_graph(d, kind, t, lmax, nch, elf),
        PfRecipe::Vis(kind, t) => {
            let (cols, vis_elf) = vis.ok_or("an image recipe without image input")?;
            let mut e = elf.to_vec();
            e.extend_from_slice(vis_elf);
            match kind {
                VisRecipe::Embed => build_embed_vis_graph(d, cols, t, &e),
                VisRecipe::SlidingExt => build_sliding_vis_graph(d, t, lmax, false, &e),
                VisRecipe::SlidingKv => build_sliding_vis_graph(d, t, lmax, true, &e),
            }
        }
    })
}

/// The decode-step buckets of a context of `max_chunks` chunks (every rung a decode can reach).
pub(super) fn step_buckets(max_chunks: u64) -> Vec<u64> {
    let mut v: Vec<u64> = (1..=max_chunks).map(|n| step_bucket(n, max_chunks)).collect();
    v.dedup();
    v
}

/// What the decoder's prefill planning of a request needs to know.
pub(super) struct PrefillShape<'a> {
    pub kinds: &'a [G4LayerKind],
    pub tile: u64,
    pub max_chunks: u64,
    /// The image path's decoder side and the pad id of image rows (`None`: text only).
    pub vis: Option<(&'a VisLoad, i32)>,
}

/// The prefill recipes of one request: every tile's embedding and layer recipes, the units of
/// its two-way image blocks, and the final norm over its last tile. Added to `out`; the tile
/// rows go to `rows` (a block drafter's context pass covers each tile).
pub(super) fn request_recipes(
    s: &PrefillShape,
    prompt: &[i32],
    has_images: bool,
    out: &mut BTreeSet<PfRecipe>,
    rows: &mut BTreeSet<u64>,
) -> Result<(), String> {
    let tiles = split_tiles(prompt.len() as u64, s.tile);
    let Some(last) = tiles.last() else {
        return Ok(());
    };
    out.insert(PfRecipe::Final(last.t));
    match (has_images, s.vis) {
        (false, _) => {
            for tl in &tiles {
                out.insert(embed_recipe(false, tl.t));
                out.extend(layer_recipes(s.kinds, tl.t, tile_chunks(tl.end(), s.max_chunks), false));
                rows.insert(tl.t);
            }
        }
        (true, Some((vl, pad))) => {
            let blocks = super::vision::run::pad_runs(prompt, pad);
            let fidx = super::vision::run::feature_rows(prompt.len(), &blocks);
            let mut i = 0usize;
            while i < tiles.len() {
                let u = plan_unit(&tiles, &blocks, i, vl.bidir, vl.unit_tiles)?;
                for &j in &u.tiles {
                    let tl = tiles[j];
                    out.insert(embed_recipe(tile_has_image(&fidx, tl), tl.t));
                    out.extend(layer_recipes(s.kinds, tl.t, tile_chunks(tl.end(), s.max_chunks), u.two_way));
                }
                i = u.next;
            }
        }
        (true, None) => return Err("a request with images for a load without image input".into()),
    }
    Ok(())
}

/// What a `--build-cache` run compiles beside the load's own recipes.
pub(super) struct BuildCtx<'a> {
    pub api: &'a Arc<SynapseApi>,
    pub cache: &'a RecipeCache,
    pub driver: &'a str,
    pub dims: &'a G4Dims,
    pub max_ctx: u64,
    pub max_chunks: u64,
    pub tile: u64,
    pub elf: &'a [(String, String)],
    pub vision: Option<(&'a G4Vision, &'a VisLoad)>,
    pub specrt: Option<&'a spec::G4SpecRt>,
    /// The shared workspace the load plans (a further recipe that needs more is reported).
    pub ws_bytes: u64,
    pub verbose: bool,
}

/// `--build-cache`: compiles into the recipe cache the recipes a load of this configuration
/// compiles on first use, without the card:
/// - the decode steps of every chunk bucket (the plain step, and a drafter's steps and first
///   blocks);
/// - the full-tile prefill recipes of every chunk count (and the two-way image recipes of a full
///   tile);
/// - the recipes of the given requests: their last tiles, their image tiles and units, the image
///   path at their images' sizes.
pub(super) fn build_further(c: &BuildCtx, requests: &[BuildRequest]) -> Result<(), String> {
    let t0 = Instant::now();
    let d = c.dims;
    let kinds = d.kinds();
    let shape = PrefillShape {
        kinds: &kinds,
        tile: c.tile,
        max_chunks: c.max_chunks,
        vis: c.vision.map(|(v, vl)| (vl, v.spec().image_token_id)),
    };
    let mut pf: BTreeSet<PfRecipe> = BTreeSet::new();
    let mut rows: BTreeSet<u64> = BTreeSet::new();
    // the configuration: full tiles at every chunk count
    pf.insert(PfRecipe::Final(c.tile));
    pf.insert(embed_recipe(false, c.tile));
    rows.insert(c.tile);
    for nch in 1..=c.max_chunks {
        pf.extend(layer_recipes(&kinds, c.tile, nch, false));
        if let Some((_, vl)) = c.vision {
            pf.insert(embed_recipe(true, c.tile));
            if vl.bidir {
                pf.extend(layer_recipes(&kinds, c.tile, nch, true));
            }
        }
    }
    // the requests
    let mut sizes: BTreeSet<u64> = BTreeSet::new();
    for r in requests {
        request_recipes(&shape, r.prompt, !r.images.is_empty(), &mut pf, &mut rows)?;
        if let Some((v, _)) = c.vision {
            sizes.extend(r.images.iter().map(|im| v.image_rows(im)));
        }
    }
    let (mut hits, mut built, mut ws) = (0u64, 0u64, 0u64);
    let mut note = |r: &CompiledRecipe| {
        if r.from_cache {
            hits += 1;
        } else {
            built += 1;
        }
        ws = ws.max(r.ws_size);
        if c.verbose {
            eprintln!("[gaudium] recipe {} cache_hit={} ws={:.2} MiB", r.name, r.from_cache, mib(r.ws_size));
        }
    };
    let compile = |gb: GraphBuilder, what: &str| -> Result<CompiledRecipe, String> {
        gb.compile(Arc::clone(c.api), c.cache, c.driver)
            .map_err(|e| format!("compile {what}: {e}"))
    };
    let vis_elf = c.vision.map(|(v, _)| (v.geo.feat_cols, v.elf()));
    let mut seen: BTreeSet<PfRecipe> = BTreeSet::new();
    for &r in &pf {
        if seen.insert(r.key()) {
            note(&compile(pf_graph(d, c.max_ctx, c.elf, vis_elf, r)?, &format!("{r:?}"))?);
        }
    }
    let buckets = step_buckets(c.max_chunks);
    for &b in &buckets {
        note(&compile(build_step_graph(d, c.max_ctx, b, c.elf), &format!("decode step ({b} chunks)"))?);
    }
    if let Some(rt) = c.specrt {
        let rows: Vec<u64> = rows.iter().copied().collect();
        spec::build_cache_recipes(rt, d, c.max_ctx, &buckets, &rows, c.api, c.cache, c.driver, &mut note)?;
    }
    if let Some((v, _)) = c.vision {
        for &n in &sizes {
            for gb in v.geo.graphs(n, v.elf()) {
                note(&compile(gb, &format!("vision recipe for {n}"))?);
            }
        }
    }
    eprintln!(
        "[gaudium] build-cache: {} further recipes ({} prefill, {} decode buckets{}, {} image sizes): {hits} from the cache, {built} compiled ({:.1}s); largest workspace {:.2} MiB (the load plans {:.2} MiB)",
        hits + built,
        seen.len(),
        buckets.len(),
        if c.specrt.is_some() { " with the drafter's" } else { "" },
        sizes.len(),
        t0.elapsed().as_secs_f64(),
        mib(ws),
        mib(c.ws_bytes)
    );
    if ws > c.ws_bytes {
        eprintln!(
            "[gaudium] build-cache: a further recipe needs a workspace of {ws} bytes, more than the {} bytes the load plans: a request that launches it is refused",
            c.ws_bytes
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_text_request_takes_its_tiles_and_its_last_tile_s_final_norm() {
        let kinds = [G4LayerKind::Sliding, G4LayerKind::Global];
        let s = PrefillShape {
            kinds: &kinds,
            tile: 256,
            max_chunks: 2,
            vis: None,
        };
        let (mut out, mut rows) = (BTreeSet::new(), BTreeSet::new());
        request_recipes(&s, &vec![7; 300], false, &mut out, &mut rows).unwrap();
        let want: BTreeSet<PfRecipe> = [
            PfRecipe::Embed(256),
            PfRecipe::Embed(44),
            PfRecipe::Layer(G4LayerKind::Sliding, 256, 1),
            PfRecipe::Layer(G4LayerKind::Global, 256, 1),
            PfRecipe::Layer(G4LayerKind::Sliding, 44, 1),
            PfRecipe::Layer(G4LayerKind::Global, 44, 1),
            PfRecipe::Final(44),
        ]
        .into_iter()
        .collect();
        assert_eq!(out, want);
        assert_eq!(rows, [44, 256].into_iter().collect());
        // a tile past the first chunk reads two
        let (mut out, mut rows) = (BTreeSet::new(), BTreeSet::new());
        request_recipes(&s, &vec![7; 4097], false, &mut out, &mut rows).unwrap();
        assert!(out.contains(&PfRecipe::Layer(G4LayerKind::Global, 1, 2)));
        assert!(out.contains(&PfRecipe::Layer(G4LayerKind::Global, 256, 1)));
        assert!(out.contains(&PfRecipe::Final(1)));
    }

    #[test]
    fn a_two_way_unit_runs_its_sliding_layers_with_the_image_recipes() {
        let kinds = [G4LayerKind::Sliding, G4LayerKind::Global];
        assert_eq!(
            layer_recipes(&kinds, 64, 3, true),
            vec![
                PfRecipe::Layer(G4LayerKind::Global, 64, 3),
                PfRecipe::Vis(VisRecipe::SlidingExt, 64),
                PfRecipe::Vis(VisRecipe::SlidingKv, 64),
            ]
        );
        assert_eq!(layer_recipes(&kinds, 64, 3, false).len(), 2);
        // a sliding layer is kept under one key whatever the chunk count
        assert_eq!(PfRecipe::Layer(G4LayerKind::Sliding, 8, 5).key(), PfRecipe::Layer(G4LayerKind::Sliding, 8, 0));
        assert_eq!(PfRecipe::Layer(G4LayerKind::Global, 8, 5).key(), PfRecipe::Layer(G4LayerKind::Global, 8, 5));
    }

    #[test]
    fn a_tile_reads_the_chunks_of_its_keys() {
        assert_eq!(tile_chunks(1, 64), 1);
        assert_eq!(tile_chunks(G4_CHUNK, 64), 1);
        assert_eq!(tile_chunks(G4_CHUNK + 1, 64), 2);
        assert_eq!(tile_chunks(64 * G4_CHUNK, 64), 64);
        // never past the cache, never none
        assert_eq!(tile_chunks(65 * G4_CHUNK, 64), 64);
        assert_eq!(tile_chunks(0, 64), 1);
        for end in [1, 255, 4095, 4096, 4097, 8192, 8193, 100_000] {
            assert_eq!(tile_chunks(end, 64), end.div_ceil(G4_CHUNK).clamp(1, 64));
        }
    }

    #[test]
    fn the_step_buckets_are_the_ladder_s_rungs() {
        assert_eq!(step_buckets(1), vec![1]);
        assert_eq!(step_buckets(8), vec![1, 2, 3, 4, 6, 8]);
        assert_eq!(step_buckets(64), vec![1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]);
    }
}
