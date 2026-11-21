//! Gaudium CLI entry point (Verified Core + Host Subsystems + Intel Gaudi 2 SynapseAI Shell).

use std::collections::BTreeMap;
use std::env;
use std::path::Path;
use std::process;
use std::sync::Arc;

use gaudium::core::budget::{advertised_served_ctx, SequenceBudget};
use gaudium::core::hbm::{check_hbm_fit, usable_hbm_bytes, HbmLedgerGuard};
use gaudium::core::kv_pool::{BlockPoolTracker, BLOCKS_PER_SLAB, BLOCK_BYTES};
use gaudium::core::lifecycle::{DeviceGuard, ExpertSlotGuard, RecipeGuard, RequestGuard};
use gaudium::core::shapes::{validate_gdn, validate_gqa, validate_hidden};
use gaudium::fp8::{bf16_to_f32, f32_to_bf16_rne};
use gaudium::model_spec::ModelSpec;
use gaudium::safetensors::SafetensorsStore;
use gaudium::synapse::device::HpuDevice;
use gaudium::synapse::ffi::{SynDataType, SynapseApi};
use gaudium::synapse::graph::GraphBuilder;
use gaudium::synapse::recipe_cache::RecipeCache;
use gaudium::verified::{DevEvent, HbmCat, ReqEvent};

fn inspect_model(dir: &Path) -> Result<(), String> {
    let cfg_path = dir.join("config.json");
    let spec = ModelSpec::from_config_file(&cfg_path).map_err(|e| e.to_string())?;
    spec.validate().map_err(|e| e.to_string())?;
    let store = SafetensorsStore::open(dir).map_err(|e| e.to_string())?;
    println!(
        "OK: {:?} ({}) — hidden={}, vocab={}, layers={} (full_attn={}), heads={}/{}, dh={}, fp8={}, vision={}, tensors={}",
        spec.family,
        spec.architecture,
        spec.hidden,
        spec.vocab,
        spec.n_layers,
        spec.n_full_attention_layers(),
        spec.n_heads,
        spec.n_kv_heads,
        spec.head_dim,
        spec.fp8_quantized,
        spec.has_vision,
        store.len()
    );
    Ok(())
}

/// Live hardware end-to-end smoke test on Intel Gaudi 2 (HL-225):
///   1. Dynamically loads `libSynapse.so` and opens Gaudi 2 through verified `DeviceGuard`.
///   2. Checks HBM capacity, compiles a BF16 `add_fwd_bf16` graph via `GraphBuilder` (destroys graph immediately).
///   3. Allocates HBM buffers + shared workspace, seals `AllocGuard` (proves post-seal alloc fails).
///   4. Runs H2D DMA -> `launch_checked` on Gaudi 2 TPC -> D2H DMA -> verifies exact BF16 output.
///   5. Drains all streams cleanly on drop.
fn hpu_smoke() -> Result<(), String> {
    let api = Arc::new(SynapseApi::load_default().map_err(|e| e.to_string())?);
    let drv_ver = api.driver_version().map_err(|e| e.to_string())?;
    let mut dev = HpuDevice::open_gaudi2(Arc::clone(&api)).map_err(|e| e.to_string())?;
    let (free_b, total_b) = dev.memory_info().map_err(|e| e.to_string())?;

    let n = 128usize;
    let buf_bytes = (n * 2) as u64; // BF16
    let mut gb = GraphBuilder::new("gaudium_smoke_add_bf16");
    let ta = gb.add_persistent_tensor("a", SynDataType::Bf16, &[n as u64]);
    let tb = gb.add_persistent_tensor("b", SynDataType::Bf16, &[n as u64]);
    let tc = gb.add_persistent_tensor("out", SynDataType::Bf16, &[n as u64]);
    gb.add_node("add_fwd_bf16", "add0", &[ta, tb], &[tc], &[]);

    let cache = RecipeCache::from_env();
    let mut recipe = gb
        .compile(Arc::clone(&api), &cache, &drv_ver)
        .map_err(|e| e.to_string())?;

    let ws_plan = recipe.ws_size.max(4096);
    let planned_total = 3 * buf_bytes + ws_plan;
    dev.check_plan_fits(planned_total, 64 << 20)
        .map_err(|e| e.to_string())?;

    let addr_a = dev
        .dmalloc(buf_bytes, HbmCat::CatOther, "smoke_a")
        .map_err(|e| e.to_string())?;
    let addr_b = dev
        .dmalloc(buf_bytes, HbmCat::CatOther, "smoke_b")
        .map_err(|e| e.to_string())?;
    let addr_out = dev
        .dmalloc(buf_bytes, HbmCat::CatOther, "smoke_out")
        .map_err(|e| e.to_string())?;
    dev.seal_with_shared_workspace(ws_plan)
        .map_err(|e| e.to_string())?;

    // Verify AllocGuard actively blocks post-seal allocations
    if dev
        .dmalloc(128, HbmCat::CatOther, "illegal_post_seal")
        .is_ok()
    {
        return Err("AllocGuard failed to refuse post-seal dmalloc!".into());
    }

    recipe
        .bind_shared_workspace(&dev)
        .map_err(|e| e.to_string())?;

    let host_a: Vec<u16> = (0..n).map(|i| f32_to_bf16_rne(i as f32)).collect();
    let host_b: Vec<u16> = (0..n)
        .map(|i| f32_to_bf16_rne(10.0 + (i as f32) * 0.5))
        .collect();
    let bytes_a: Vec<u8> = host_a.iter().flat_map(|x| x.to_le_bytes()).collect();
    let bytes_b: Vec<u8> = host_b.iter().flat_map(|x| x.to_le_bytes()).collect();

    dev.copy_h2d_sync(&bytes_a, addr_a)
        .map_err(|e| e.to_string())?;
    dev.copy_h2d_sync(&bytes_b, addr_b)
        .map_err(|e| e.to_string())?;

    let mut bindings = BTreeMap::new();
    bindings.insert("a".to_owned(), addr_a);
    bindings.insert("b".to_owned(), addr_b);
    bindings.insert("out".to_owned(), addr_out);

    let ev0 = dev.create_event(true).map_err(|e| e.to_string())?;
    let ev1 = dev.create_event(true).map_err(|e| e.to_string())?;
    ev0.record_compute(&dev).map_err(|e| e.to_string())?;
    recipe
        .launch_checked(&dev, &bindings)
        .map_err(|e| e.to_string())?;
    ev1.record_compute(&dev).map_err(|e| e.to_string())?;
    ev1.synchronize().map_err(|e| e.to_string())?;
    let ns = gaudium::synapse::device::DeviceEvent::elapsed_ns(&ev0, &ev1).unwrap_or(0);

    let mut out_bytes = vec![0u8; bytes_a.len()];
    dev.copy_d2h_sync(addr_out, &mut out_bytes)
        .map_err(|e| e.to_string())?;

    for i in 0..n {
        let got_bits = u16::from_le_bytes([out_bytes[2 * i], out_bytes[2 * i + 1]]);
        let got = bf16_to_f32(got_bits);
        let expected = bf16_to_f32(f32_to_bf16_rne(
            bf16_to_f32(host_a[i]) + bf16_to_f32(host_b[i]),
        ));
        if got != expected {
            return Err(format!(
                "mismatch at lane {i}: got {got}, expected {expected}"
            ));
        }
    }

    println!(
        "GAUDI 2 HPU SMOKE PASS: driver={}, HBM={:.2}/{:.2} GiB free, recipe_ws={} B, cache_hit={}, device_time={:.1} us, 128/128 BF16 lanes bit-exact!",
        drv_ver,
        (free_b as f64) / (1u64 << 30) as f64,
        (total_b as f64) / (1u64 << 30) as f64,
        recipe.ws_size,
        recipe.from_cache,
        (ns as f64) / 1e3
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 3 && args[1] == "--inspect-model" {
        if let Err(e) = inspect_model(Path::new(&args[2])) {
            eprintln!("FAIL: {e}");
            process::exit(1);
        }
        return;
    }
    if args.len() == 2 && args[1] == "--hpu-smoke" {
        if let Err(e) = hpu_smoke() {
            eprintln!("HPU SMOKE FAIL: {e}");
            process::exit(1);
        }
        return;
    }

    // Smoke-check Slice 1 + Slice 2 verified core
    let mut dev = DeviceGuard::new();
    dev.transition(DevEvent::EvSynInit).expect("syn init");
    dev.transition(DevEvent::EvDevAcquire).expect("acquire");
    dev.transition(DevEvent::EvStreamsCreate).expect("streams");
    dev.transition(DevEvent::EvSealAlloc).expect("seal");

    let mut rec = RecipeGuard::new();
    rec.compile();
    rec.bind_workspace(256 << 20, 512 << 20, true)
        .expect("workspace fits");

    let mut slot = ExpertSlotGuard::new();
    slot.begin_fill().expect("fill");
    slot.mark_loaded().expect("loaded");
    slot.publish().expect("publish");

    let mut req = RequestGuard::new();
    req.apply(ReqEvent::EvReqStartPrefill).expect("prefill");
    req.apply(ReqEvent::EvReqPrefillDone).expect("decode");
    req.apply(ReqEvent::EvReqFinishStop).expect("stop");

    let mut pool = BlockPoolTracker::new(BLOCKS_PER_SLAB);
    pool.alloc(7).expect("alloc 7 blocks");

    let gib = 1u64 << 30;
    let usable = usable_hbm_bytes(96 * gib, 4 * gib, gib);
    let mut ledger = HbmLedgerGuard::new();
    ledger.record(HbmCat::CatWeights, 54 * gib);
    ledger.record(HbmCat::CatKv, 24 * gib);
    let margin = check_hbm_fit(ledger.total_bytes(), usable).expect("HBM fits");

    let budget = SequenceBudget::admit(1024, Some(4096), 262_144).expect("prompt admitted");
    validate_gqa(24, 4, 256).expect("Qwen3.8-27B GQA");
    validate_gdn(16, 48, 4).expect("Qwen3.8-27B GDN");
    validate_hidden(5120, 248_320).expect("Qwen3.8-27B hidden/vocab");

    println!(
        "Gaudium RustyBends v1 core OK:\n  \
         Slice 1: dev={:?}, recipe={:?}, slot={:?}, req={:?}\n  \
         Slice 2: pool={}/{}x{}MiB, hbm_margin={}GiB, served_ctx={}, seq_rem={}",
        dev.state(),
        rec.state(),
        slot.state(),
        req.state(),
        pool.allocated(),
        pool.capacity(),
        BLOCK_BYTES >> 20,
        margin >> 30,
        advertised_served_ctx(262_144),
        budget.remaining()
    );
}
