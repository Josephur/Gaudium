//! Measurement of the half-layer tile recipes on the card (`--prefill halfprobe=<launches>`): a
//! child module of `engine::qwen3`.
//!
//! After a request has prefilled (so the tile buffers, the states and the caches hold real rows),
//! every half-layer form is launched `n` times back to back and timed between two stream
//! synchronisations: the first half of a DeltaNet layer and of an attention layer alone, the
//! second half alone, and both as two independent bodies of one recipe in each node order. The
//! question it answers is how much of two bodies' device time the card overlaps.
//!
//! The probe overwrites the tile buffers and the first layers' states, so the engine must not
//! serve another request afterwards (the CLI runs it after its last request).

use super::prefill::{
    build_attn_tile_graph, build_half_graph, build_lin_tile_graph, half_scratch_layout, FirstHalf,
    TileForm, Weave, SECOND_PFX,
};
use super::*;

/// Generic per-layer weight suffixes a tile recipe may bind.
const LAYER_WEIGHTS: [&str; 16] = [
    "w_in", "w_post", "W_qg", "W_k", "W_v", "qn_w", "kn_w", "W_o", "W_gate", "W_up", "W_down",
    "W_qkvz", "W_ba", "conv_w", "conv_b", "norm_w",
];
const LAYER_WEIGHTS_GDN: [&str; 2] = ["A_log", "dt_bias"];

/// One recipe of the probe: (label, first half, second half, node order, compiler settings, whole
/// layer in one body, recipe name suffix).
type Case = (String, Option<FirstHalf>, bool, Weave, Vec<(String, String)>, bool, String);

/// The probe's recipes for this model and tile form. Graph-compiler settings to try on the
/// DeltaNet forms come from `GAUDIUM_HALFPROBE_GC`: sets separated by `;`, each a `,`-separated
/// list of `NAME=value`.
fn cases(spec: &ModelSpec, form: &TileForm) -> (Vec<Case>, Vec<(&'static str, FirstHalf)>) {
    let has_lin = (0..spec.n_layers as usize).any(|l| !spec.is_full_attention_layer(l));
    let has_attn = (0..spec.n_layers as usize).any(|l| spec.is_full_attention_layer(l));
    let gc_sets: Vec<Vec<(String, String)>> = std::env::var("GAUDIUM_HALFPROBE_GC")
        .unwrap_or_default()
        .split(';')
        .map(|set| {
            set.split(',')
                .filter_map(|kv| kv.trim().split_once('='))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect::<Vec<_>>()
        })
        .filter(|s| !s.is_empty())
        .collect();

    let mut all: Vec<Case> = Vec::new();
    all.push(("second half alone".into(), None, true, Weave::Cost, Vec::new(), false, String::new()));
    let mut firsts: Vec<(&'static str, FirstHalf)> = Vec::new();
    if has_lin && form.mme {
        firsts.push(("DeltaNet", FirstHalf::Lin));
    }
    if has_attn && form.fastexp {
        firsts.push(("attention", FirstHalf::Attn(1)));
    }
    for (name, f) in &firsts {
        all.push((
            format!("{name} first half alone"),
            Some(*f),
            false,
            Weave::Cost,
            Vec::new(),
            false,
            String::new(),
        ));
        for (wn, w) in [
            ("cost-ordered", Weave::Cost),
            ("second body first", Weave::SecondFirst),
            ("first body first", Weave::FirstFirst),
        ] {
            all.push((
                format!("{name} first half + second half, {wn}"),
                Some(*f),
                true,
                w,
                Vec::new(),
                false,
                String::new(),
            ));
        }
    }
    // whole layers of each kind, then every compiler-setting set on the whole layers and on the
    // second half (the MLP), which both layer kinds share
    let mut wholes: Vec<(&'static str, FirstHalf)> = Vec::new();
    if has_lin && form.mme {
        wholes.push(("DeltaNet", FirstHalf::Lin));
    }
    if has_attn && form.fastexp {
        wholes.push(("attention", FirstHalf::Attn(1)));
    }
    for (name, f) in &wholes {
        all.push((format!("{name} whole layer"), Some(*f), true, Weave::Cost, Vec::new(), true, String::new()));
    }
    for (i, set) in gc_sets.iter().enumerate() {
        let tag: Vec<String> = set.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let tag = tag.join(",");
        for (name, f) in &wholes {
            all.push((
                format!("{name} whole layer [{tag}]"),
                Some(*f),
                true,
                Weave::Cost,
                set.clone(),
                true,
                format!("_gc{i}"),
            ));
        }
        all.push((
            format!("second half alone [{tag}]"),
            None,
            true,
            Weave::Cost,
            set.clone(),
            false,
            format!("_gc{i}"),
        ));
    }
    (all, firsts)
}

/// The graph of one case.
#[allow(clippy::too_many_arguments)]
fn case_graph(
    case: &Case,
    spec: &ModelSpec,
    t: u64,
    max_ctx: u64,
    form: &TileForm,
    elf_hashes: &[(String, String)],
    elf_tile: &[(String, String)],
) -> GraphBuilder {
    let (_, first, second, weave, gc, whole, sfx) = case;
    let mut gb = if *whole {
        match first {
            Some(FirstHalf::Attn(nch)) => {
                let mut g = build_attn_tile_graph(spec, t, max_ctx, *nch, form, elf_hashes, elf_tile);
                g.set_recipe_name(format!("g_qwen_probe_attn_T{t}{sfx}"));
                g
            }
            _ => {
                let mut g = build_lin_tile_graph(spec, t, form, elf_hashes, elf_tile);
                g.set_recipe_name(format!("g_qwen_probe_lin_T{t}{sfx}"));
                g
            }
        }
    } else {
        let mut g =
            build_half_graph(spec, t, max_ctx, *first, *second, *weave, form, elf_hashes, elf_tile);
        if !sfx.is_empty() {
            let name = format!("{}{sfx}", g.recipe_name());
            g.set_recipe_name(name);
        }
        g
    };
    for (k, v) in gc {
        gb.set_gc_setting(k.clone(), v.clone());
    }
    gb
}

/// Compiles every probe recipe WITHOUT the card and without the recipe cache (so the compiler
/// really runs and writes whatever its own dump and log settings ask for), and prints each one's
/// node count and workspace. Used by `--build-cache --prefill halfprobe=<n>`.
#[allow(clippy::too_many_arguments)]
pub(super) fn compile_only(
    spec: &ModelSpec,
    t: u64,
    max_ctx: u64,
    form: &TileForm,
    elf_hashes: &[(String, String)],
    elf_tile: &[(String, String)],
    api: &Arc<SynapseApi>,
    driver_version: &str,
) -> Result<(), String> {
    let (all, _) = cases(spec, form);
    let cache = RecipeCache::disabled();
    eprintln!("[halfprobe] compile only: tile {t} rows, {} recipes, recipe cache off", all.len());
    for case in &all {
        let gb = case_graph(case, spec, t, max_ctx, form, elf_hashes, elf_tile);
        let nodes = gb.node_count();
        let t0 = Instant::now();
        let r = gb
            .compile(Arc::clone(api), &cache, driver_version)
            .map_err(|e| format!("halfprobe: compile {}: {e}", gb.recipe_name()))?;
        eprintln!(
            "[halfprobe] compiled {} ({}): {nodes} nodes, workspace {:.2} MiB, {:.2}s",
            r.name,
            case.0,
            r.ws_size as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

pub(super) fn run(eng: &mut Qwen3Engine, n: u32) -> Result<(), String> {
    let t = eng.prefill_tile;
    let spec = eng.spec.clone();
    if spec.n_heads * spec.head_dim != spec.gdn_v_heads * spec.gdn_v_dim && spec.n_linear_layers() > 0
    {
        return Err("halfprobe: the two layer kinds' output projections differ in width".into());
    }
    let (base, total) = eng.pf.scratch_region();
    // the hand-over rows live at the top of the planned scratch region for this measurement
    let align256 = |n: u64| n.next_multiple_of(256);
    let core_bytes = align256(spec.n_heads * spec.head_dim * t * 2);
    let out_bytes = align256(spec.hidden * t * 2);
    let reserve = 2 * core_bytes + out_bytes;
    if total < reserve {
        return Err("halfprobe: the planned tile scratch is too small".into());
    }
    let top = base + total - reserve;
    let (l_core, g_core, g_out) = (top, top + core_bytes, top + 2 * core_bytes);

    let lin_layer = (0..spec.n_layers as usize).find(|&l| !spec.is_full_attention_layer(l));
    let attn_layer = (0..spec.n_layers as usize).find(|&l| spec.is_full_attention_layer(l));
    let form = eng.pf.form;
    let elf_tile = eng.pf.elf_tile().to_vec();
    let (all, firsts) = cases(&spec, &form);

    eprintln!(
        "[halfprobe] tile {t} rows, {n} launches per recipe (device time between two stream syncs)"
    );
    let mut results: Vec<(String, f64)> = Vec::new();
    for case in all {
        let gb = case_graph(&case, &spec, t, eng.max_ctx, &form, &eng.elf_hashes, &elf_tile);
        let (label, first, second, _, _, whole, _) = case;
        let (layout, bytes) = half_scratch_layout(&gb);
        if bytes > total - reserve {
            return Err(format!(
                "halfprobe: '{label}' needs {bytes} B of scratch, {} B are planned",
                total - reserve
            ));
        }
        let mut r = gb
            .compile(Arc::clone(eng.dev.api()), &eng.cache, &eng.driver_version)
            .map_err(|e| format!("halfprobe: compile {}: {e}", gb.recipe_name()))?;
        r.bind_shared_workspace(&eng.dev)
            .map_err(|e| format!("halfprobe: workspace of {}: {e}", r.name))?;

        let mut pool: BTreeMap<String, u64> = BTreeMap::new();
        for (name, off) in layout {
            pool.insert(name, base + off);
        }
        for name in ["zero_slot", "norm_ones", "rope_c", "rope_s"] {
            if let Some(&a) = eng.weights_addr.get(name) {
                pool.insert(name.to_string(), a);
            }
        }
        if let Some(a) = eng.pf.const_one_addr() {
            pool.insert("fp8_one".into(), a);
        }
        pool.insert("x_in".into(), eng.pf_x0);
        pool.insert("L_out".into(), g_out);
        pool.insert("cursor".into(), eng.pf_cursor);
        pool.insert("L_core".into(), l_core);
        pool.insert(format!("{SECOND_PFX}x_in"), eng.pf_x1);
        pool.insert(format!("{SECOND_PFX}core"), g_core);
        pool.insert(format!("{SECOND_PFX}out"), g_out);
        // the first half takes the weights, state and caches of the first layer of its kind, the
        // second half the weights of the first layer
        let lf = match first {
            Some(FirstHalf::Attn(_)) => attn_layer,
            Some(FirstHalf::Lin) => lin_layer,
            None => None,
        };
        let bind_layer = |pool: &mut BTreeMap<String, u64>, l: usize, pfx: &str| {
            for suffix in LAYER_WEIGHTS.iter().chain(&LAYER_WEIGHTS_GDN) {
                if let Some(&a) = eng.weights_addr.get(&format!("L{l}_{suffix}")) {
                    pool.insert(format!("{pfx}{suffix}"), a);
                }
            }
            let mut ext = BTreeMap::new();
            crate::engine::moe::bind_ext_layer_weights(&eng.weights_addr, l, &mut ext);
            for (k, a) in ext {
                pool.insert(format!("{pfx}{}", &k[2..]), a);
            }
        };
        if let Some(l) = lf {
            bind_layer(&mut pool, l, "L_");
            if let Some(st) = eng.gdn_state.get(&l) {
                pool.insert("L_S_in".into(), st.s[0]);
                pool.insert("L_S_out".into(), st.s[1]);
                pool.insert("L_cs_in".into(), st.cs[0]);
                pool.insert("L_cs_out".into(), st.cs[1]);
            }
            for (k, name) in [("L_kcache", "kcache"), ("L_vcache", "vcache")] {
                if let Some(&(a, _)) = eng.kv_addr.get(&format!("L{l}_{name}")) {
                    pool.insert(k.into(), a);
                }
            }
        }
        if second && !whole {
            bind_layer(&mut pool, 0, SECOND_PFX);
        }
        let bind = select_exact_bindings(&r, &pool)?;
        // one launch to settle, then the timed run
        r.launch_checked(&eng.dev, &bind)
            .map_err(|e| format!("halfprobe: launch {}: {e}", r.name))?;
        unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
            .map_err(|e| format!("halfprobe: sync: {e}"))?;
        let t0 = Instant::now();
        for _ in 0..n {
            r.launch_checked(&eng.dev, &bind)
                .map_err(|e| format!("halfprobe: launch {}: {e}", r.name))?;
        }
        unsafe { eng.dev.sync_stream(eng.dev.compute_stream()) }
            .map_err(|e| format!("halfprobe: sync: {e}"))?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n.max(1));
        eprintln!("[halfprobe] {us:>9.1} us  {label}  ({})", r.name);
        results.push((label, us));
    }
    // overlap summary: two bodies in one recipe against the sum of the two alone
    let alone = |key: &str| {
        results
            .iter()
            .find(|(l, _)| l == key)
            .map(|x| x.1)
            .unwrap_or(0.0)
    };
    for (l, us) in &results {
        if let Some((base, _)) = l.split_once(" [") {
            let b = alone(base);
            if b > 0.0 {
                eprintln!("[halfprobe] {l}: {us:.1} us against {b:.1} us = {:+.1} %", 100.0 * (us - b) / b);
            }
        }
    }
    let g = alone("second half alone");
    for (name, _) in &firsts {
        let f = alone(&format!("{name} first half alone"));
        for (l, us) in results.iter().filter(|(l, _)| l.starts_with(&format!("{name} first half + "))) {
            eprintln!(
                "[halfprobe] {l}: {us:.1} us against {:.1} us apart = {:.1} % saved",
                f + g,
                100.0 * (f + g - us) / (f + g).max(1e-9)
            );
        }
    }
    Ok(())
}
