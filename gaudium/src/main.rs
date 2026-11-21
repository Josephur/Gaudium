//! Gaudium CLI entry point (Verified Core + Host Subsystems + Intel Gaudi 2 SynapseAI Shell).

use std::collections::BTreeMap;
use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;

use gaudium::core::budget::{advertised_served_ctx, SequenceBudget};
use gaudium::core::hbm::{check_hbm_fit, usable_hbm_bytes, HbmLedgerGuard};
use gaudium::core::kv_pool::{BlockPoolTracker, BLOCKS_PER_SLAB, BLOCK_BYTES};
use gaudium::core::lifecycle::{DeviceGuard, ExpertSlotGuard, RecipeGuard, RequestGuard};
use gaudium::core::shapes::{validate_gdn, validate_gqa, validate_hidden};
use gaudium::engine::qwen3::{Qwen3Config, Qwen3Engine, Qwen3GenParams, KV_CHUNK};
use gaudium::fp8::{bf16_to_f32, f32_to_bf16_rne};
use gaudium::model_spec::ModelSpec;
use gaudium::safetensors::SafetensorsStore;
use gaudium::synapse::device::HpuDevice;
use gaudium::synapse::ffi::{SynDataType, SynapseApi};
use gaudium::synapse::graph::GraphBuilder;
use gaudium::synapse::recipe_cache::RecipeCache;
use gaudium::tokenizer::{format_qwen3_chat, StreamDecoder, Tokenizer};
use gaudium::verified::{DevEvent, HbmCat, ReqEvent};

fn print_usage() {
    println!(
        "Gaudium — Intel Gaudi 2 Inference Engine (RustyBends v1)\n\
         \n\
         Usage:\n\
         \x20 gaudium                                   Run verified-core self-check\n\
         \x20 gaudium --inspect-model <dir>             Validate config.json & Safetensors\n\
         \x20 gaudium --tokenize <model_dir> <text>     Encode & round-trip decode via tokenizer.json\n\
         \x20 gaudium --hpu-smoke                       Run live Gaudi 2 BF16 add smoke test\n\
         \x20 gaudium --generate [options]              Run live Gaudi 2 LLM generation\n\
         \n\
         Generate options:\n\
         \x20 --model <dir>         Model directory (or GAUDIUM_MODEL)\n\
         \x20 --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)\n\
         \x20 --prompt <text>       Input prompt text\n\
         \x20 --prompt-ids <csv>    Raw comma-separated prompt token IDs (skips tokenizer)\n\
         \x20 --chat                Wrap --prompt in Qwen3 ChatML template\n\
         \x20 --no-think            Disable thinking block (with --chat)\n\
         \x20 --system <text>       System prompt (with --chat)\n\
         \x20 --max-new <N>         Maximum tokens to generate (default: 64)\n\
         \x20 --max-ctx <N>         Context capacity, multiple of 4096 (default: 4096)\n\
         \x20 --temperature <F>     Sampling temperature; 0 = greedy (default: 0.0)\n\
         \x20 --top-p <F>           Nucleus sampling p (default: 1.0)\n\
         \x20 --top-k <N>           Top-k sampling cap (default: 0 = off)\n\
         \x20 --min-p <F>           Min-p sampling threshold (default: 0.0)\n\
         \x20 --seed <N>            Sampling RNG seed (default: 0)\n\
         \x20 --no-eos              Do not stop early on EOS token\n\
         \x20 --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)\n\
         \x20 --no-cache            Disable on-disk compiled recipe cache\n\
         \x20 --dump-tokens         Print generated token IDs to stderr\n\
         \x20 --verbose             Print HBM, recipe compile, and timing diagnostics"
    );
}

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

fn run_tokenize(dir: &Path, text: &str) -> Result<(), String> {
    let tok = Tokenizer::from_model_dir(dir).map_err(|e| e.to_string())?;
    let ids = tok.encode(text, true);
    let decoded = tok.decode(&ids, false);
    println!("vocab={} tokens={}", tok.vocab_size(), ids.len());
    println!("ids={ids:?}");
    println!("decoded={decoded:?}");
    Ok(())
}

fn run_generate(args: &[String]) -> Result<(), String> {
    let mut model_dir: Option<PathBuf> = env::var_os("GAUDIUM_MODEL").map(PathBuf::from);
    let mut kernels_dir: Option<PathBuf> = env::var_os("GAUDIUM_KERNELS").map(PathBuf::from);
    let mut prompt: Option<String> = None;
    let mut prompt_ids_raw: Option<String> = None;
    let mut chat = false;
    let mut no_think = false;
    let mut system_prompt: Option<String> = None;
    let mut max_ctx = KV_CHUNK;
    let mut cache_dir: Option<PathBuf> = None;
    let mut no_cache = false;
    let mut dump_tokens = false;
    let mut verbose = false;
    let mut params = Qwen3GenParams::default();

    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: &mut usize, name: &str| -> Result<String, String> {
            *idx += 1;
            args.get(*idx)
                .cloned()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        match flag {
            "--generate" => {}
            "--model" => model_dir = Some(PathBuf::from(next_val(&mut i, "--model")?)),
            "--kernels" => kernels_dir = Some(PathBuf::from(next_val(&mut i, "--kernels")?)),
            "--prompt" => prompt = Some(next_val(&mut i, "--prompt")?),
            "--prompt-ids" => prompt_ids_raw = Some(next_val(&mut i, "--prompt-ids")?),
            "--chat" => chat = true,
            "--no-think" => no_think = true,
            "--system" => system_prompt = Some(next_val(&mut i, "--system")?),
            "--max-new" => {
                params.max_new = next_val(&mut i, "--max-new")?
                    .parse()
                    .map_err(|_| "invalid --max-new".to_string())?;
            }
            "--max-ctx" => {
                max_ctx = next_val(&mut i, "--max-ctx")?
                    .parse()
                    .map_err(|_| "invalid --max-ctx".to_string())?;
            }
            "--temperature" => {
                params.temperature = next_val(&mut i, "--temperature")?
                    .parse()
                    .map_err(|_| "invalid --temperature".to_string())?;
            }
            "--top-p" => {
                params.top_p = next_val(&mut i, "--top-p")?
                    .parse()
                    .map_err(|_| "invalid --top-p".to_string())?;
            }
            "--top-k" => {
                params.top_k = next_val(&mut i, "--top-k")?
                    .parse()
                    .map_err(|_| "invalid --top-k".to_string())?;
            }
            "--min-p" => {
                params.min_p = next_val(&mut i, "--min-p")?
                    .parse()
                    .map_err(|_| "invalid --min-p".to_string())?;
            }
            "--seed" => {
                params.seed = next_val(&mut i, "--seed")?
                    .parse()
                    .map_err(|_| "invalid --seed".to_string())?;
            }
            "--no-eos" => params.stop_on_eos = false,
            "--cache-dir" => cache_dir = Some(PathBuf::from(next_val(&mut i, "--cache-dir")?)),
            "--no-cache" => no_cache = true,
            "--dump-tokens" => dump_tokens = true,
            "--verbose" | "-v" => verbose = true,
            other => return Err(format!("unknown argument '{other}'")),
        }
        i += 1;
    }

    let model_dir =
        model_dir.ok_or_else(|| "missing --model <dir> (or GAUDIUM_MODEL env var)".to_string())?;
    let kernels_dir = kernels_dir
        .ok_or_else(|| "missing --kernels <dir> (or GAUDIUM_KERNELS env var)".to_string())?;

    let tok = Tokenizer::from_model_dir(&model_dir).map_err(|e| e.to_string())?;
    let prompt_ids: Vec<i32> = if let Some(csv) = prompt_ids_raw {
        csv.split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                s.trim()
                    .parse::<i32>()
                    .map_err(|_| format!("invalid token id '{s}'"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let raw_prompt = prompt.unwrap_or_else(|| "The capital of France is".to_string());
        let formatted = if chat {
            format_qwen3_chat(&raw_prompt, system_prompt.as_deref(), !no_think)
        } else {
            raw_prompt
        };
        tok.encode(&formatted, true)
    };

    if verbose {
        eprintln!(
            "[gaudium] prompt_tokens={} ids={:?}",
            prompt_ids.len(),
            &prompt_ids[..prompt_ids.len().min(32)]
        );
    }

    let mut cfg = Qwen3Config::new(model_dir, kernels_dir);
    cfg.max_ctx = max_ctx;
    cfg.cache_dir = cache_dir;
    cfg.no_cache = no_cache;
    cfg.verbose = verbose;

    let mut engine = Qwen3Engine::load(&cfg)?;

    let mut decoder = StreamDecoder::new(&tok);
    let stdout = io::stdout();
    let report = engine.generate(&prompt_ids, &params, |id| {
        if !tok.is_special(id) {
            let chunk = decoder.push(id);
            if !chunk.is_empty() {
                let mut lock = stdout.lock();
                let _ = lock.write_all(chunk.as_bytes());
                let _ = lock.flush();
            }
        }
    })?;

    let tail = decoder.flush();
    if !tail.is_empty() {
        print!("{tail}");
    }
    println!();

    if dump_tokens {
        eprintln!("TOKENS: {:?}", report.generated_tokens);
    }
    eprintln!(
        "[gaudium] prefill: {} tok in {:.1} ms ({:.1} tok/s) | decode: {} tok in {:.1} ms ({:.2} tok/s, {:.2} ms/step) | eos={}",
        report.prompt_tokens,
        report.ttft_ms,
        report.prefill_tok_s,
        report.generated_tokens.len().saturating_sub(1),
        report.decode_ms,
        report.decode_tok_s,
        report.ms_per_step,
        report.stopped_by_eos
    );
    Ok(())
}

/// Live hardware end-to-end smoke test on Intel Gaudi 2 (HL-225).
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
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }
    if args.len() == 3 && args[1] == "--inspect-model" {
        if let Err(e) = inspect_model(Path::new(&args[2])) {
            eprintln!("FAIL: {e}");
            process::exit(1);
        }
        return;
    }
    if args.len() == 4 && args[1] == "--tokenize" {
        if let Err(e) = run_tokenize(Path::new(&args[2]), &args[3]) {
            eprintln!("TOKENIZE FAIL: {e}");
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
    if args.iter().any(|a| a == "--generate") {
        if let Err(e) = run_generate(&args[1..]) {
            eprintln!("GENERATE FAIL: {e}");
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
