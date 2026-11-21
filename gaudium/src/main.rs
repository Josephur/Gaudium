//! Gaudium CLI entry point (Verified Core + Host Engine Subsystems).

use std::env;
use std::path::Path;
use std::process;

use gaudium::core::budget::{advertised_served_ctx, SequenceBudget};
use gaudium::core::hbm::{check_hbm_fit, usable_hbm_bytes, HbmLedgerGuard};
use gaudium::core::kv_pool::{BlockPoolTracker, BLOCKS_PER_SLAB, BLOCK_BYTES};
use gaudium::core::lifecycle::{DeviceGuard, ExpertSlotGuard, RecipeGuard, RequestGuard};
use gaudium::core::shapes::{validate_gdn, validate_gqa, validate_hidden};
use gaudium::model_spec::ModelSpec;
use gaudium::safetensors::SafetensorsStore;
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 3 && args[1] == "--inspect-model" {
        if let Err(e) = inspect_model(Path::new(&args[2])) {
            eprintln!("FAIL: {e}");
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
