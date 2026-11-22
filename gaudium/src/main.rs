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
use gaudium::engine::fp8_graph::WeightPrecision;
use gaudium::engine::fp8_pack::{default_fp8_dir, ACT_TABLE_FILE};
use gaudium::engine::qwen3::spec::{OracleRequest, SpecConfig, SpecMethod};
use gaudium::engine::qwen3::{Qwen3Config, Qwen3GenParams, KV_CHUNK};
use gaudium::engine::{load_engine, GenEngine};
use gaudium::fp8::{bf16_to_f32, f32_to_bf16_rne, store_act_scales};
use gaudium::model_spec::ModelSpec;
use gaudium::safetensors::SafetensorsStore;
use gaudium::server::{parse_model_arg, run_server, ModelRunOptions, ServerConfig};
use gaudium::synapse::device::HpuDevice;
use gaudium::synapse::ffi::{SynDataType, SynapseApi};
use gaudium::synapse::graph::GraphBuilder;
use gaudium::synapse::recipe_cache::RecipeCache;
use gaudium::tokenizer::{
    format_gemma4_chat, format_qwen3_chat, gemma4_template_closes_thought, StreamDecoder, Tokenizer,
};
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
         \x20 gaudium --build-cache [generate options]  Compile every recipe of that configuration into\n\
         \x20                                           the recipe cache and print its exact HBM plan,\n\
         \x20                                           without opening the card (Qwen engines)\n\
         \x20 gaudium --serve [options]                 Start OpenAI HTTP/SSE + Gaudium Web UI server\n\
         \x20 gaudium --fp8-calibrate [options]         Measure a model's FP8 activation scales (BF16 graphs)\n\
         \x20 gaudium --fp8-pack --model <dir> --fp8-cache <dir>\n\
         \x20                                           Quantise a Gemma 4 BF16 checkpoint's matrices into an\n\
         \x20                                           FP8 cache directory (host only, no card)\n\
         \n\
         Serve options:\n\
         \x20 --host <addr>         Bind address (default: 0.0.0.0)\n\
         \x20 --port <N>            Bind port (default: 8090)\n\
         \x20 --model <dir|id=dir>  Model directory (repeatable for multi-model hot-swap)\n\
         \x20 --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)\n\
         \x20 --ui-dir <dir>        Gaudium Web UI bundle directory (or GAUDIUM_UI_DIR)\n\
         \x20 --max-ctx <N>         Context capacity, multiple of 4096 (default: 4096)\n\
         \x20 --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)\n\
         \x20 --no-cache            Disable on-disk compiled recipe cache\n\
         \x20 --lazy                Start without holding Gaudi 2; load on first chat/start\n\
         \x20 --idle-park <secs>    Auto-release Gaudi 2 after N seconds idle (0=off, default: 300)\n\
         \x20 --queue-config <file> Card-queue settings, re-read on every turn (or GAUDIUM_QUEUE_CONFIG;\n\
         \x20                       default: queue.conf in the parent of GAUDI_LIVE_DIR, the card-lock\n\
         \x20                       state dir). Keys: group_max_wait_min (default 45), group_max_skips\n\
         \x20                       (default 3): a request for the loaded model may run ahead of older\n\
         \x20                       requests until the oldest has waited that long or been passed over\n\
         \x20                       that many times\n\
         \x20 --draft-model <id=dir> Drafter checkpoint for model <id>: speculative decoding, output\n\
         \x20                       token-identical to K=0 (repeatable, one per model). The kind\n\
         \x20                       follows the directory name: a block drafter when it contains\n\
         \x20                       dflash (7 drafts per launch), a model's own drafter when it\n\
         \x20                       ends in -assistant (4 drafts), otherwise DSpine. <id>=mtp\n\
         \x20                       drafts with the model's own MTP head (no checkpoint; 4 drafts,\n\
         \x20                       3 in the routed mixture-of-experts form)\n\
         \x20 --moe-form <id=f>     Mixture-of-experts model <id>: dense (default) or routed (faster\n\
         \x20                       decode, more device memory, its own token stream)\n\
         \x20 --model-ctx <id=N>    Context size for model <id> instead of --max-ctx\n\
         \x20 --draft-k <N[,N..]>   DSpine drafts verified per launch (default: 7,15 adaptive)\n\
         \x20 --fp8-act <id=file>   FP8 activation-scale table for FP8 model <id> (repeatable)\n\
         \x20 --fp8-cache <id=dir>  Cache directory for model <id>'s quantised FP8 matrices\n\
         \x20 --api-key <key>       Optional Bearer / X-Api-Key auth token\n\
         \x20 --verbose             Print server, HBM, and recipe diagnostics\n\
         \n\
         Live status environment variables:\n\
         \x20 GAUDI_LIVE_DIR        Directory for live dashboard JSON records (GAUDI_LIVE=0 disables)\n\
         \n\
         Engine environment variables:\n\
         \x20 GAUDIUM_REQUEST_RESET Device state zeroed before each request: auto (default), dirty\n\
         \x20                       or full (see --request-reset under Generate options)\n\
         \x20 GAUDIUM_UPLOAD_DIRECT 0 = never read weight files with direct I/O at load (default:\n\
         \x20                       weight bytes that are not in the page cache are read directly)\n\
         Qwen prefill environment variables:\n\
         \x20 GAUDIUM_PREFILL       Prefill levers `k=v,..` when --prefill is not given (see --prefill\n\
         \x20                       under Generate options). Changes the compiled tile recipes; with\n\
         \x20                       BF16 weights streams may differ between settings at exact BF16\n\
         \x20                       logit ties, s16 > 0 is not token-identical to s16=0.\n\
         \n\
         Gemma 4 environment variables:\n\
         \x20 GAUDIUM_G4_NORM       Layer RMS-norm node: row | tile | vendor (default: tile for sizes with\n\
         \x20                       per-layer inputs, else vendor). Changes compiled graphs; streams may\n\
         \x20                       differ between settings at exact BF16 logit ties.\n\
         \n\
         Generate options:\n\
         \x20 --model <dir>         Model directory (or GAUDIUM_MODEL)\n\
         \x20 --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)\n\
         \x20 --prompt <text>       Input prompt text\n\
         \x20 --prompt-ids <csv>    Raw comma-separated prompt token IDs (skips tokenizer)\n\
         \x20 --prompt-file <path>  Read the prompt text from a file (long-context prompts)\n\
         \x20 --prompt-ids-file <p> Read comma/whitespace-separated prompt token IDs from a file\n\
         \x20                       (repeatable: each file is one request on the same loaded engine)\n\
         \x20 --also-prompt-ids-file <p>  Same as a further --prompt-ids-file\n\
         \x20 --also-prompt-file <p>  One more text request on the same loaded engine (repeatable;\n\
         \x20                       formatted like --prompt, after the token-ID requests)\n\
         \x20 --chat                Wrap --prompt in the model's chat template (Qwen3 ChatML / Gemma 4 turns)\n\
         \x20 --no-think            Disable thinking block (with --chat)\n\
         \x20 --system <text>       System prompt (with --chat)\n\
         \x20 --max-new <N>         Maximum tokens to generate (default: 64)\n\
         \x20 --max-ctx <N>         Context capacity, multiple of 4096 (default: 4096)\n\
         \x20 --temperature <F>     Sampling temperature; 0 = greedy (default: 0.0)\n\
         \x20 --top-p <F>           Nucleus sampling p (default: 1.0)\n\
         \x20 --top-k <N>           Top-k sampling cap (default: 0 = off)\n\
         \x20 --min-p <F>           Min-p sampling threshold (default: 0.0)\n\
         \x20 --seed <N>            Sampling RNG seed (default: 0)\n\
         \x20                       Sampled requests sample on the card (the sampler is part of the\n\
         \x20                       recipes; the token at a position is a function of the prefix,\n\
         \x20                       the seed and the position, so speculative runs equal K=0)\n\
         \x20 --sampling <T[:top_k[:top_p[:min_p[:seed[:coupled[:check]]]]]]>  One more sampler\n\
         \x20                       setting (repeatable; T = 0 is greedy): every request runs once\n\
         \x20                       per listed setting, in place of --temperature / --top-k / --top-p /\n\
         \x20                       --min-p / --seed / --coupled / --sample-check\n\
         \x20 --coupled <0|1>       Sampled speculative requests: DSpine drafts are drawn with the\n\
         \x20                       verify's position noise on the drafter's own candidate scores\n\
         \x20                       (default 1). Tokens never change, only draft acceptance\n\
         \x20 --draft-trunc <0|1>   Coupled drafts also apply top-k / top-p / min-p to the drafter's\n\
         \x20                       candidates (default 1; 0 = temperature and noise only)\n\
         \x20 --sample-check        Debug: sample every row again on the host from the card's own\n\
         \x20                       logits (bit-exact CPU twin; also the coupled walk) and report\n\
         \x20                       rows / mismatches. One launch at a time: not a timing run\n\
         \x20 --compare-k0          Run every request first on the plain non-speculative K=0 loop\n\
         \x20                       of the same engine (same seed and sampler settings), then as\n\
         \x20                       configured (speculatively with --spec, enqueue-ahead K=0\n\
         \x20                       without), and report identity or the first divergence\n\
         \x20 --profile-steps <A-B[@R][:file]>  Device trace of decode launches A..B (0-based) of\n\
         \x20                       request R (default 0; with --compare-k0 the K=0 reference of a\n\
         \x20                       prompt is the request before its speculative run): prints the\n\
         \x20                       launch split by node family and engine, and the idle time by\n\
         \x20                       node transition; :file also writes the raw events (tab-separated).\n\
         \x20                       Repeatable. Recipes compile fresh (recipe cache off); a profiled\n\
         \x20                       run is not a timing result\n\
         \x20 --prefill-warm <N>    Load the prefill recipes of prompt-tail lengths 1..N with the\n\
         \x20                       engine (compiling them on a recipe-cache miss) instead of inside\n\
         \x20                       the first request of each prompt length (default: 0 = first use)\n\
         \x20 --no-eos              Do not stop early on EOS token\n\
         \x20 --image <file.ppm>    Image input, binary PPM (P6); repeatable, in prompt order\n\
         \x20 --image-raw <WxH:f>   Image input, raw RGB8 file with explicit size; repeatable\n\
         \x20 --vision-max-patches <N>  Largest image the vision tower is planned for, in\n\
         \x20                       16x16 patches (default: the largest --image; larger images\n\
         \x20                       are resized down to fit)\n\
         \x20 --vision-gelu-bf16    Evaluate the vision tower's GELU in BF16 (default: F32);\n\
         \x20                       slightly faster, slightly less accurate image embeddings\n\
         \x20 --dump-vision-emb <p> Write the image embeddings (tokens x hidden, raw BF16) to a file\n\
         \x20 --dump-prompt-ids     Print the final prompt token IDs to stderr and continue\n\
         \x20 --repeat <N>          Run the same request N times in one process (warm timings)\n\
         \x20 --after-ids-file <p>  Then run a text-only request with these prompt token IDs\n\
         \x20                       (checks that an image request leaves no state behind)\n\
         \x20 --spec <method>       Speculative decoding: dspine or dflash2 (on-card block drafters),\n\
         \x20                       mtp (the target's own MTP head, hybrid Qwen3.5/3.8, dense or\n\
         \x20                       mixture-of-experts, either --moe-form), assistant (a Gemma 4\n\
         \x20                       size's own drafter checkpoint) or oracle (test: drafts from\n\
         \x20                       --oracle-file); absent = non-speculative K=0.\n\
         \x20                       A Gemma 4 model reads the drafter kind from the checkpoint:\n\
         \x20                       --spec dflash2 --draft-model <dir> runs a DFlashDraftModel\n\
         \x20                       block drafter (one --draft-k width, default 7).\n\
         \x20                       Chain modes are token-identical to K=0 (sampled: same seed)\n\
         \x20 --draft-model <dir>   Drafter checkpoint directory (with --spec dspine / dflash2;\n\
         \x20                       assistant: default <model_dir>-assistant)\n\
         \x20 --draft-k <N[,N..]>   Drafts verified per launch, 1..31 (default: 15; dflash2: 1..7,\n\
         \x20                       default 7; mtp: at most 15, default 4; assistant: default 4);\n\
         \x20                       several values let the engine pick a width per step from\n\
         \x20                       acceptance (mtp: every step then drafts the widest listed block)\n\
         \x20 --sweep-draft-k       Measurement: with several --draft-k widths, run every request once\n\
         \x20                       per listed width as a fixed-width run, then once with the width\n\
         \x20                       policy (one engine load for a whole width sweep)\n\
         \x20 --tree <shape>        Verify a draft tree per launch (with --spec mtp or oracle):\n\
         \x20                       D = spine depth, D:j1-j2@r1-r2,.. adds the rank r1..r2\n\
         \x20                       candidates (2 = runner-up) at depths j1..j2, e.g. 4:1-2@2.\n\
         \x20                       Opt-in: a leaf row may differ from K=0 in the last bit, so the\n\
         \x20                       stream can fork at an exact BF16 tie; with --spec mtp an\n\
         \x20                       --oracle-file is a reference stream whose first fork is reported\n\
         \x20 --k0-first            Same as --compare-k0\n\
         \x20 --draft-vocab <N>     Draft readouts over N token ids only: the lowest ids plus the\n\
         \x20                       control-token block at the lowest EOS id (multiple of 128; 0 =\n\
         \x20                       the full vocabulary). Default: 98304 for dspine / dflash2 (or\n\
         \x20                       GAUDIUM_DRAFT_VOCAB), the full vocabulary for mtp. Drafts are\n\
         \x20                       guesses: the output is unchanged, only acceptance can move\n\
         \x20 --draft-fp8 <0|1>     DSpine drafter layers as FP8 weight GEMMs (default, or\n\
         \x20                       GAUDIUM_DRAFT_FP8: on when the drafter's activation-scale table\n\
         \x20                       exists). The verify stays as it is: the output is unchanged\n\
         \x20 --draft-fp8-act <f>   The drafter's activation-scale table (or GAUDIUM_DRAFT_FP8_ACT;\n\
         \x20                       default <draft dir>-gaudium-pc/act_scales_ds.txt)\n\
         \x20 --draft-fp8-calibrate <f>  Calibration run for that table: the drafter runs BF16, one\n\
         \x20                       launch at a time, its GEMM input sites are read back after every\n\
         \x20                       launch and the table is written to <f> after each request\n\
         \x20 --width-sweep         Measurement: every request runs once per fixed verify width of\n\
         \x20                       --draft-k, then once with the width policy (compare the\n\
         \x20                       policy against the best fixed width per prompt)\n\
         \x20 --gdn-fast-rows <0|1> Checkpoint form of --gdn-verify: run the rows with the kernel\n\
         \x20                       build that takes its per-row gate scalars as vector loads\n\
         \x20                       (bit-identical; default, or GAUDIUM_GDN_FAST_ROWS: on when that\n\
         \x20                       kernel and the step glue library are in the kernels directory)\n\
         \x20 --k0-pipeline <0|1>   Greedy K=0 requests of the dense models decode enqueue-ahead:\n\
         \x20                       the next step is queued while the current one runs, its token\n\
         \x20                       fed back on the card (default 1, or GAUDIUM_K0_PIPELINE; same\n\
         \x20                       recipes and tokens as 0, without the host round trip per token)\n\
         \x20 --gdn-verify <m>      Gated DeltaNet state across the verify rows of a speculative\n\
         \x20                       step (dspine / dflash2 / mtp chains; or GAUDIUM_GDN_VERIFY):\n\
         \x20                       checkpoint (default: one state checkpoint per row) or replay\n\
         \x20                       (no checkpoints: the next step re-applies the accepted rows'\n\
         \x20                       updates); both are token-identical to K=0. mme: the rows read\n\
         \x20                       the state through MME GEMMs and commit in one closed form:\n\
         \x20                       faster per row, NOT token-identical (another fp32 summation\n\
         \x20                       order: a stream may fork at a near-tie; check with --compare-k0)\n\
         \x20 --oracle-file <p>     Known continuation token IDs (with --spec oracle; repeatable,\n\
         \x20                       one per request in order)\n\
         \x20 --oracle-corrupt <csv> Generated-token indices whose oracle draft is made wrong\n\
         \x20                       (applies to the --oracle-file options that follow it; with\n\
         \x20                       --tree the leaf of that depth carries the true token)\n\
         \x20 --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)\n\
         \x20 --no-cache            Disable on-disk compiled recipe cache\n\
         \x20 --build-cache         Compile every recipe an engine load with these options uses\n\
         \x20                       into the recipe cache, print the exact HBM plan and exit,\n\
         \x20                       WITHOUT opening the card (runs beside a process that owns it)\n\
         \x20 --build-cache-chunks <csv>  With --build-cache and --spec on a mixture-of-experts\n\
         \x20                       model: compile the speculative step recipes of exactly these KV\n\
         \x20                       chunk counts (ceil(position / 4096)) instead of all of them;\n\
         \x20                       routed-form steps take tens of seconds each\n\
         \x20 --precision <p>       Weight precision: auto (default: FP8 for a block-FP8 checkpoint,\n\
         \x20                       else BF16), bf16 (an FP8 checkpoint is dequantised at load), fp8\n\
         \x20                       (a BF16 checkpoint, Gemma 4 included, is quantised at load per\n\
         \x20                       output channel; needs the model's table from --fp8-calibrate.\n\
         \x20                       FP8 is a precision of its own: its K=0 stream is not the BF16 one).\n\
         \x20                       A Gemma 4 model given --fp8-act runs FP8 under auto\n\
         \x20 --fp8-act <file>      FP8 activation-scale table (or GAUDIUM_FP8_ACT; default:\n\
         \x20                       <model_dir>-gaudium-pc/act_scales_lc.txt)\n\
         \x20 --fp8-cache <dir>     Cache directory for the quantised FP8 matrices (or GAUDIUM_FP8_CACHE)\n\
         \x20 --moe-form <f>        MoE expert execution: dense (default: one masked wide MLP per layer)\n\
         \x20                       or routed (stacked BF16 experts, only the routed ones are read;\n\
         \x20                       faster decode, more device memory, its own token stream: the\n\
         \x20                       two forms round differently)\n\
         \x20 --request-reset <m>   Device state zeroed before each request: auto (default: the\n\
         \x20                       DeltaNet start state; KV rows past the cursor are masked),\n\
         \x20                       dirty (also the KV rows earlier requests wrote) or full\n\
         \x20                       (or GAUDIUM_REQUEST_RESET; same output in every mode)\n\
         \x20 --prefill <k=v,..>    Prefill levers of the Qwen engines (default: all on): tile=N rows\n\
         \x20                       per tile (multiple of 64 up to 512; 0 = the widest that fits),\n\
         \x20                       mme=0|1 DeltaNet recurrence as chunked MME GEMMs, rows=0|1|2 its\n\
         \x20                       prep / norm kernels, fastexp=0|1 fast-exp tile softmax, groups=N\n\
         \x20                       attention chunk groups, s16=0|1|2 half-width attention core (BF16\n\
         \x20                       scores; 2: + FP8 P; default 2 with dense FP8 weights, else 0: it\n\
         \x20                       changes the prefill numerics), plain = every lever off, stats /\n\
         \x20                       profile = print the tile loop's host split / per-recipe times\n\
         \x20 --dump-tokens         Print generated token IDs to stderr\n\
         \x20 --force-tokens-file <p> Teacher-force this reference stream (comma/whitespace token IDs)\n\
         \x20                       and report every position where greedy K=0 disagrees, with logits\n\
         \x20 --verbose             Print HBM, recipe compile, and timing diagnostics\n\
         \n\
         FP8 calibration options (prefill only, BF16 graphs; writes `<site> <k> <amax> <rms>` lines):\n\
         \x20 --model <dir>         Model directory (or GAUDIUM_MODEL)\n\
         \x20 --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)\n\
         \x20 --prompt-ids-file <p> Calibration prompt as token IDs (repeatable)\n\
         \x20 --prompt-file <path>  Calibration prompt as text (repeatable)\n\
         \x20 --out <file>          Table to write (default: <model_dir>-gaudium-pc/act_scales_lc.txt)\n\
         \x20 --max-ctx <N>         Context capacity, multiple of 4096 (default: 4096)\n\
         \x20 --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)\n\
         \x20 --no-cache            Disable on-disk compiled recipe cache\n\
         \x20 --verbose             Print HBM, recipe compile, and per-site diagnostics"
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
    let mut extra_prompt_ids_raw: Vec<String> = Vec::new();
    let mut extra_prompt_texts: Vec<String> = Vec::new();
    let mut chat = false;
    let mut no_think = false;
    let mut system_prompt: Option<String> = None;
    let mut max_ctx = KV_CHUNK;
    let mut cache_dir: Option<PathBuf> = None;
    let mut no_cache = false;
    let mut dump_tokens = false;
    let mut verbose = false;
    let mut precision = WeightPrecision::Auto;
    let mut fp8_act: Option<PathBuf> = None;
    let mut fp8_cache: Option<PathBuf> = None;
    let mut moe_routed = false;
    let mut prefill_opts: Option<String> = None;
    let mut params = Qwen3GenParams::default();
    let mut image_args: Vec<(bool, String)> = Vec::new(); // (raw RGB8?, path / WxH:path)
    let mut vision_max_patches = 0u64;
    let mut dump_prompt_ids = false;
    let mut vision_gelu_bf16 = false;
    let mut dump_vision_emb: Option<String> = None;
    let mut repeat = 1usize;
    let mut after_ids_file: Option<String> = None;
    let mut spec_method: Option<String> = None;
    let mut draft_model: Option<PathBuf> = None;
    let mut draft_k: Vec<u64> = Vec::new();
    let mut oracle_streams: Vec<(String, Vec<usize>)> = Vec::new();
    let mut oracle_corrupt: Vec<usize> = Vec::new();
    let mut tree_spec: Option<String> = None;
    let mut draft_vocab: u64 = 0;
    let mut samplings: Vec<String> = Vec::new();
    let mut compare_k0 = false;
    let mut sweep_draft_k = false;
    let mut build_cache = false;
    let mut build_cache_chunks: Vec<u64> = Vec::new();
    let mut prefill_warm = 0u64;
    let mut request_reset: Option<String> = None;
    let mut profile_windows: Vec<gaudium::synapse::profiler::ProfileWindow> = Vec::new();
    let mut draft_vocab_set = false;
    let mut draft_fp8: Option<bool> = None;
    let mut draft_fp8_act: Option<PathBuf> = None;
    let mut draft_fp8_calibrate: Option<PathBuf> = None;
    let mut gdn_verify: Option<gaudium::engine::step_opts::GdnVerify> = None;
    let mut k0_pipeline: Option<bool> = None;
    let mut gdn_fast_rows: Option<bool> = None;
    let mut width_sweep = false;

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
            "--build-cache" => build_cache = true,
            "--model" => model_dir = Some(PathBuf::from(next_val(&mut i, "--model")?)),
            "--kernels" => kernels_dir = Some(PathBuf::from(next_val(&mut i, "--kernels")?)),
            "--prompt" => prompt = Some(next_val(&mut i, "--prompt")?),
            "--prompt-ids" => prompt_ids_raw = Some(next_val(&mut i, "--prompt-ids")?),
            "--prompt-file" => {
                let path = next_val(&mut i, "--prompt-file")?;
                prompt = Some(
                    std::fs::read_to_string(&path)
                        .map_err(|e| format!("read --prompt-file {path}: {e}"))?,
                );
            }
            "--prompt-ids-file" => {
                let path = next_val(&mut i, "--prompt-ids-file")?;
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("read --prompt-ids-file {path}: {e}"))?;
                if prompt_ids_raw.is_some() {
                    extra_prompt_ids_raw.push(text);
                } else {
                    prompt_ids_raw = Some(text);
                }
            }
            "--also-prompt-file" => {
                let path = next_val(&mut i, "--also-prompt-file")?;
                extra_prompt_texts.push(
                    std::fs::read_to_string(&path)
                        .map_err(|e| format!("read --also-prompt-file {path}: {e}"))?,
                );
            }
            "--also-prompt-ids-file" => {
                let path = next_val(&mut i, "--also-prompt-ids-file")?;
                extra_prompt_ids_raw.push(
                    std::fs::read_to_string(&path)
                        .map_err(|e| format!("read --also-prompt-ids-file {path}: {e}"))?,
                );
            }
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
            "--sampling" => samplings.push(next_val(&mut i, "--sampling")?),
            "--coupled" => {
                params.coupled = parse_switch(&next_val(&mut i, "--coupled")?, "--coupled")?
            }
            "--draft-trunc" => {
                params.draft_trunc =
                    parse_switch(&next_val(&mut i, "--draft-trunc")?, "--draft-trunc")?
            }
            "--sample-check" => params.sample_check = true,
            "--compare-k0" => compare_k0 = true,
            "--sweep-draft-k" => sweep_draft_k = true,
            "--build-cache-chunks" => {
                build_cache_chunks = next_val(&mut i, "--build-cache-chunks")?
                    .split(',')
                    .map(|s| s.trim().parse::<u64>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "invalid --build-cache-chunks".to_string())?;
            }
            "--profile-steps" => {
                profile_windows.push(gaudium::synapse::profiler::ProfileWindow::parse(
                    &next_val(&mut i, "--profile-steps")?,
                )?)
            }
            "--prefill-warm" => {
                prefill_warm = next_val(&mut i, "--prefill-warm")?
                    .parse()
                    .map_err(|_| "invalid --prefill-warm".to_string())?;
            }
            "--no-eos" => params.stop_on_eos = false,
            "--image" => image_args.push((false, next_val(&mut i, "--image")?)),
            "--image-raw" => image_args.push((true, next_val(&mut i, "--image-raw")?)),
            "--vision-max-patches" => {
                vision_max_patches = next_val(&mut i, "--vision-max-patches")?
                    .parse()
                    .map_err(|_| "invalid --vision-max-patches".to_string())?;
            }
            "--dump-prompt-ids" => dump_prompt_ids = true,
            "--vision-gelu-bf16" => vision_gelu_bf16 = true,
            "--dump-vision-emb" => dump_vision_emb = Some(next_val(&mut i, "--dump-vision-emb")?),
            "--repeat" => {
                repeat = next_val(&mut i, "--repeat")?
                    .parse()
                    .map_err(|_| "invalid --repeat".to_string())?;
            }
            "--after-ids-file" => after_ids_file = Some(next_val(&mut i, "--after-ids-file")?),
            "--spec" => spec_method = Some(next_val(&mut i, "--spec")?),
            "--k0-first" => compare_k0 = true,
            "--tree" => tree_spec = Some(next_val(&mut i, "--tree")?),
            "--draft-fp8" => {
                draft_fp8 = Some(parse_switch(
                    &next_val(&mut i, "--draft-fp8")?,
                    "--draft-fp8",
                )?)
            }
            "--draft-fp8-act" => {
                draft_fp8_act = Some(PathBuf::from(next_val(&mut i, "--draft-fp8-act")?))
            }
            "--draft-fp8-calibrate" => {
                draft_fp8_calibrate =
                    Some(PathBuf::from(next_val(&mut i, "--draft-fp8-calibrate")?))
            }
            "--width-sweep" => width_sweep = true,
            "--gdn-fast-rows" => {
                gdn_fast_rows = Some(parse_switch(
                    &next_val(&mut i, "--gdn-fast-rows")?,
                    "--gdn-fast-rows",
                )?)
            }
            "--k0-pipeline" => {
                k0_pipeline = Some(parse_switch(
                    &next_val(&mut i, "--k0-pipeline")?,
                    "--k0-pipeline",
                )?)
            }
            "--gdn-verify" => {
                let v = next_val(&mut i, "--gdn-verify")?;
                gdn_verify = Some(
                    gaudium::engine::step_opts::GdnVerify::parse(&v)
                        .ok_or_else(|| format!("--gdn-verify '{v}': checkpoint, replay or mme"))?,
                )
            }
            "--draft-vocab" => {
                draft_vocab_set = true;
                draft_vocab = next_val(&mut i, "--draft-vocab")?
                    .parse()
                    .map_err(|_| "invalid --draft-vocab".to_string())?
            }
            "--draft-model" => {
                draft_model = Some(PathBuf::from(next_val(&mut i, "--draft-model")?))
            }
            "--draft-k" => {
                draft_k = next_val(&mut i, "--draft-k")?
                    .split(',')
                    .map(|s| s.trim().parse::<u64>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "invalid --draft-k".to_string())?;
            }
            "--oracle-file" => {
                let path = next_val(&mut i, "--oracle-file")?;
                oracle_streams.push((
                    std::fs::read_to_string(&path)
                        .map_err(|e| format!("read --oracle-file {path}: {e}"))?,
                    oracle_corrupt.clone(),
                ));
            }
            "--oracle-corrupt" => {
                oracle_corrupt = next_val(&mut i, "--oracle-corrupt")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "invalid --oracle-corrupt".to_string())?;
            }
            "--cache-dir" => cache_dir = Some(PathBuf::from(next_val(&mut i, "--cache-dir")?)),
            "--no-cache" => no_cache = true,
            "--precision" => {
                let s = next_val(&mut i, "--precision")?;
                precision = WeightPrecision::parse(&s)
                    .ok_or_else(|| format!("invalid --precision '{s}' (auto, bf16, fp8)"))?;
            }
            "--fp8-act" => fp8_act = Some(PathBuf::from(next_val(&mut i, "--fp8-act")?)),
            "--fp8-cache" => fp8_cache = Some(PathBuf::from(next_val(&mut i, "--fp8-cache")?)),
            "--moe-form" => {
                moe_routed = match next_val(&mut i, "--moe-form")?.as_str() {
                    "dense" => false,
                    "routed" => true,
                    other => return Err(format!("invalid --moe-form '{other}' (dense, routed)")),
                };
            }
            "--request-reset" => request_reset = Some(next_val(&mut i, "--request-reset")?),
            "--prefill" => {
                let s = next_val(&mut i, "--prefill")?;
                gaudium::engine::qwen3::prefill::PrefillOpts::parse(&s)
                    .map_err(|e| format!("--prefill: {e}"))?;
                prefill_opts = Some(s);
            }
            "--dump-tokens" => dump_tokens = true,
            "--force-tokens-file" => {
                let path = next_val(&mut i, "--force-tokens-file")?;
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("read --force-tokens-file {path}: {e}"))?;
                let ids = text
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        s.parse::<i32>()
                            .map_err(|_| format!("invalid forced token id '{s}'"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                params.force_tokens = Some(ids);
            }
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
    if let Some(im_end) = tok.token_id("<|im_end|>") {
        params.extra_eos_ids.push(im_end);
    }
    // Image input: prepare on the host first, so the tower is planned for these image sizes.
    let vision_in =
        gaudium::vision::cli::prepare_cli_images(&model_dir, &image_args, vision_max_patches)?;
    let prompt = match (&vision_in, prompt) {
        (Some(v), p) if prompt_ids_raw.is_none() => {
            Some(v.with_placeholders(p.as_deref().unwrap_or("Describe this image.")))
        }
        (_, p) => p,
    };
    let parse_ids_csv = |csv: &str| -> Result<Vec<i32>, String> {
        csv.split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                s.trim()
                    .parse::<i32>()
                    .map_err(|_| format!("invalid token id '{s}'"))
            })
            .collect::<Result<Vec<_>, _>>()
    };
    // a text prompt as token ids, through the chat template when `--chat` is given
    let encode_text = |raw_prompt: String| -> Vec<i32> {
        let formatted = if chat && tok.token_id("<|turn>").is_some() {
            let template =
                std::fs::read_to_string(model_dir.join("chat_template.jinja")).unwrap_or_default();
            format_gemma4_chat(
                &raw_prompt,
                system_prompt.as_deref(),
                !no_think,
                gemma4_template_closes_thought(&template),
            )
        } else if chat {
            format_qwen3_chat(&raw_prompt, system_prompt.as_deref(), !no_think)
        } else {
            raw_prompt
        };
        tok.encode(&formatted, true)
    };
    let prompt_ids: Vec<i32> = if let Some(csv) = prompt_ids_raw {
        parse_ids_csv(&csv)?
    } else {
        encode_text(prompt.unwrap_or_else(|| "The capital of France is".to_string()))
    };
    // `--also-prompt-file`: more text requests for the same loaded engine
    let extra_text_ids: Vec<Vec<i32>> = extra_prompt_texts.into_iter().map(encode_text).collect();
    let prompt_ids = match &vision_in {
        Some(v) => v.expand_prompt(&prompt_ids)?,
        None => prompt_ids,
    };
    if dump_prompt_ids {
        eprintln!("PROMPT_IDS: {prompt_ids:?}");
    }

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
    cfg.precision = precision;
    cfg.fp8_act = fp8_act;
    cfg.fp8_cache = fp8_cache;
    cfg.moe_routed = moe_routed;
    cfg.prefill_warm_rows = prefill_warm;
    cfg.request_reset = request_reset;
    if let Some(v) = draft_fp8 {
        cfg.step.draft_fp8 = Some(v);
    }
    if draft_fp8_act.is_some() {
        cfg.step.draft_fp8_act = draft_fp8_act;
    }
    cfg.step.draft_fp8_calibrate = draft_fp8_calibrate;
    if let Some(v) = gdn_verify {
        cfg.step.gdn_verify = v;
    }
    if let Some(v) = k0_pipeline {
        cfg.step.k0_pipeline = v;
    }
    if gdn_fast_rows.is_some() {
        cfg.step.gdn_fast_rows = gdn_fast_rows;
    }
    if draft_vocab_set && matches!(spec_method.as_deref(), Some("dspine" | "dflash2")) {
        // the block drafters' readout rows (the MTP head takes its own below)
        cfg.step.draft_vocab = Some(draft_vocab);
        draft_vocab = 0;
    }
    if !profile_windows.is_empty() {
        // The profiler template must be in the environment before the device opens, and node
        // names reach the trace only from recipes compiled in this process.
        gaudium::synapse::profiler::setup_env();
        cfg.no_cache = true;
        cfg.step.profile = profile_windows;
    }
    cfg.prefill = prefill_opts;
    if let Some(v) = &vision_in {
        v.configure(&mut cfg, vision_max_patches);
        cfg.vision_gelu_bf16 = vision_gelu_bf16;
    }
    let sweep_widths: Vec<u64> = draft_k.clone();
    if let Some(method) = spec_method {
        if draft_k.is_empty() {
            draft_k.push(match method.as_str() {
                "dflash2" => 7,
                "mtp" | "assistant" => 4,
                _ => 15,
            });
        }
        // a draft tree: one step verifies all its nodes (rows = nodes, so `nodes - 1` drafts)
        let tree = match &tree_spec {
            Some(t) if method == "mtp" || method == "oracle" => {
                let shape = gaudium::tree::parse_tree_spec(t).map_err(|e| e.to_string())?;
                draft_k = vec![(shape.n_nodes() as u64).saturating_sub(1)];
                Some(shape)
            }
            Some(_) => return Err("--tree needs --spec mtp or --spec oracle".into()),
            None => None,
        };
        let known_streams = || -> Result<Vec<OracleRequest>, String> {
            let mut requests = Vec::new();
            for (csv, corrupt) in &oracle_streams {
                requests.push(OracleRequest {
                    stream: parse_ids_csv(csv)?,
                    corrupt: corrupt.clone(),
                });
            }
            Ok(requests)
        };
        if draft_vocab > 0 && method != "mtp" {
            return Err("--draft-vocab needs --spec mtp, dspine or dflash2".into());
        }
        let method = match method.as_str() {
            "mtp" => SpecMethod::Mtp {
                reference: if tree.is_some() {
                    known_streams()?.into_iter().map(|r| r.stream).collect()
                } else {
                    Vec::new()
                },
                tree,
                draft_vocab,
            },
            "oracle" if tree.is_some() => SpecMethod::OracleTree {
                requests: known_streams()?,
                shape: tree.expect("checked"),
            },
            "dspine" => SpecMethod::DSpine {
                draft_model: draft_model
                    .ok_or_else(|| "--spec dspine needs --draft-model <dir>".to_string())?,
            },
            "dflash2" => SpecMethod::DFlash2 {
                draft_model: draft_model
                    .ok_or_else(|| "--spec dflash2 needs --draft-model <dir>".to_string())?,
            },
            "assistant" => SpecMethod::Assistant {
                // a size's drafter is published beside it as `<model>-assistant`
                draft_model: draft_model.unwrap_or_else(|| {
                    let mut name = cfg
                        .model_dir
                        .as_os_str()
                        .to_string_lossy()
                        .trim_end_matches('/')
                        .to_owned();
                    name.push_str("-assistant");
                    PathBuf::from(name)
                }),
            },
            "oracle" => {
                let mut requests = Vec::new();
                for (csv, corrupt) in &oracle_streams {
                    requests.push(OracleRequest {
                        stream: parse_ids_csv(csv)?,
                        corrupt: corrupt.clone(),
                    });
                }
                SpecMethod::Oracle { requests }
            }
            other => {
                return Err(format!(
                    "unknown --spec method '{other}' (dspine, dflash2, mtp, assistant, oracle)"
                ))
            }
        };
        cfg.spec = Some(SpecConfig { method, draft_k });
    } else if draft_model.is_some()
        || !draft_k.is_empty()
        || !oracle_streams.is_empty()
        || tree_spec.is_some()
        || draft_vocab > 0
    {
        return Err(
            "--draft-model / --draft-k / --oracle-file / --tree / --draft-vocab need --spec <method>"
                .into(),
        );
    }

    // `--build-cache`: compile this configuration's recipes without the card (SynapseAI is
    // initialized, the device is never acquired), then exit. With `--build-cache-chunks` the
    // speculative step recipes of exactly those KV chunk counts are compiled (the routed
    // mixture-of-experts steps take tens of seconds each); without it, every recipe a load of
    // this configuration uses.
    if build_cache {
        if !build_cache_chunks.is_empty() {
            return gaudium::engine::qwen3::build_cache(&cfg, &build_cache_chunks);
        }
        return gaudium::engine::build_cache(&cfg);
    }
    if !build_cache_chunks.is_empty() {
        return Err("--build-cache-chunks needs --build-cache".into());
    }

    // Stop requests (SIGINT/SIGTERM) end generation between synced launches, never mid-launch.
    gaudium::signals::install();
    let mut engine: Box<dyn GenEngine> = load_engine(&cfg)?;
    if let Some(v) = &vision_in {
        engine.set_request_images(v.images.clone())?;
    }

    // Every `--prompt-ids-file` after the first is one more request served by the same loaded engine.
    let mut all_prompts: Vec<Vec<i32>> = vec![prompt_ids];
    for csv in &extra_prompt_ids_raw {
        all_prompts.push(parse_ids_csv(csv)?);
    }
    all_prompts.extend(extra_text_ids);
    // Sampler settings every request runs under (`--sampling`, else the single base setting).
    let mut settings: Vec<Qwen3GenParams> = Vec::new();
    for raw in &samplings {
        settings.push(parse_sampling(raw, &params)?);
    }
    if settings.is_empty() {
        settings.push(params.clone());
    }
    // `--sweep-draft-k`: every setting once per listed width (fixed), then with the width policy
    if sweep_draft_k {
        let widths: Vec<u64> = cfg.spec.as_ref().map_or(Vec::new(), |s| s.draft_k.clone());
        if widths.len() < 2 {
            return Err("--sweep-draft-k needs --spec with several --draft-k widths".into());
        }
        let base = std::mem::take(&mut settings);
        for p in base {
            for &k in &widths {
                let mut fixed = p.clone();
                fixed.spec_k = Some(k);
                settings.push(fixed);
            }
            settings.push(p);
        }
    }
    let multi = all_prompts.len() * settings.len() > 1 || width_sweep;
    // `--compare-k0`: the K=0 stream of a (request, sampler setting); settings that differ only
    // in how drafts are drawn share it.
    type K0Key = (usize, u32, u64, u32, u32, u64, bool);
    let mut k0_streams: Vec<(K0Key, Vec<i32>)> = Vec::new();
    // one run per (request, sampler setting); `--width-sweep`: once per fixed verify width of
    // `--draft-k`, then once with the width policy
    let mut forced: Vec<Option<u64>> = Vec::new();
    if width_sweep {
        forced.extend(sweep_widths.iter().map(|&k| Some(k)));
    }
    forced.push(None);
    let runs: Vec<(usize, usize, Option<u64>)> = (0..all_prompts.len())
        .flat_map(|p| (0..settings.len()).map(move |s| (p, s)))
        .flat_map(|(p, s)| forced.iter().map(move |&f| (p, s, f)))
        .collect();
    for (prompt_idx, set_idx, force_k) in runs {
        let mut run_params = settings[set_idx].clone();
        run_params.force_draft_k = force_k;
        let (prompt_ids, params) = (&all_prompts[prompt_idx], &run_params);
        if width_sweep {
            match force_k {
                Some(k) => eprintln!("[gaudium] width sweep: fixed verify width K={k}"),
                None => eprintln!("[gaudium] width sweep: width policy"),
            }
        }
        if gaudium::signals::term_requested() {
            break;
        }
        if multi {
            eprintln!(
                "[gaudium] request {prompt_idx}: prompt_tokens={} sampling: {}{}",
                prompt_ids.len(),
                sampling_label(params),
                match params.spec_k {
                    Some(k) => format!(" | fixed draft width K={k}"),
                    None if sweep_draft_k => " | width policy".to_string(),
                    None => String::new(),
                }
            );
        }
        // `--compare-k0`: the reference stream first, on the same engine's non-speculative path
        let k0_key: K0Key = (
            prompt_idx,
            params.temperature.max(0.0).to_bits(),
            params.top_k,
            params.top_p.to_bits(),
            params.min_p.to_bits(),
            if params.temperature > 0.0 {
                params.seed
            } else {
                0
            },
            params.sample_check,
        );
        let k0_known = k0_streams
            .iter()
            .find(|(k, _)| *k == k0_key)
            .map(|(_, v)| v.clone());
        let k0_ref: Option<Vec<i32>> = if let Some(known) = k0_known.filter(|_| compare_k0) {
            Some(known)
        } else if compare_k0 {
            let mut p0 = params.clone();
            p0.non_speculative = true;
            // the reference is the plain K=0 loop, also for a non-speculative engine (whose
            // greedy requests otherwise decode enqueue-ahead)
            p0.k0_plain = true;
            if let Some(v) = vision_in.as_ref().filter(|_| prompt_idx == 0) {
                engine.set_request_images(v.images.clone())?;
            }
            // An engine that switches speculation per engine rather than per request (Gemma 4)
            // is switched around the reference run; the others honour `non_speculative`.
            let switched = engine.set_speculative(false).is_ok();
            let r0 = engine.generate_cancellable(
                prompt_ids,
                &p0,
                Some(gaudium::signals::term_flag()),
                &mut |_| {},
            );
            if switched {
                engine.set_speculative(true)?;
            }
            let r0 = r0?;
            eprintln!(
                "[gaudium] K=0 reference: {} | prefill {:.1} ms | decode: {} tok in {:.1} ms ({:.2} tok/s, {:.2} ms/step){}",
                r0.method,
                r0.ttft_ms,
                r0.generated_tokens.len().saturating_sub(1),
                r0.decode_ms,
                r0.decode_tok_s,
                r0.ms_per_step,
                check_note(&r0),
            );
            if gaudium::signals::term_requested() {
                break;
            }
            k0_streams.push((k0_key, r0.generated_tokens.clone()));
            Some(r0.generated_tokens)
        } else {
            None
        };
        // the images belong to the first prompt: every run of it gets them again
        if let Some(v) = vision_in.as_ref().filter(|_| prompt_idx == 0) {
            engine.set_request_images(v.images.clone())?;
        }
        let mut decoder = StreamDecoder::new(&tok);
        let stdout = io::stdout();
        let report = engine.generate_cancellable(
            prompt_ids,
            params,
            Some(gaudium::signals::term_flag()),
            &mut |id| {
                if !tok.is_special(id) {
                    let chunk = decoder.push(id);
                    if !chunk.is_empty() {
                        let mut lock = stdout.lock();
                        let _ = lock.write_all(chunk.as_bytes());
                        let _ = lock.flush();
                    }
                }
            },
        )?;

        let tail = decoder.flush();
        if !tail.is_empty() {
            print!("{tail}");
        }
        println!();

        if dump_tokens {
            eprintln!("TOKENS: {:?}", report.generated_tokens);
        }
        if params.force_tokens.is_some() {
            eprintln!(
                "[gaudium] teacher-forced {} positions: {} disagree with greedy K=0",
                report.generated_tokens.len(),
                report.forks.len()
            );
            for f in &report.forks {
                eprintln!(
                    "FORK @{}: engine {} (logit {:.4}) vs forced {} (logit {:.4}) gap {:.4}",
                    f.pos,
                    f.engine_token,
                    f.engine_logit,
                    f.forced_token,
                    f.forced_logit,
                    f.engine_logit - f.forced_logit
                );
            }
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
        if report.ttft_recipe_ms >= 0.05 {
            eprintln!(
                "[gaudium] prefill includes {:.1} ms loading / compiling recipes used for the first time in this process",
                report.ttft_recipe_ms
            );
        }
        if !report.method.is_empty() {
            eprintln!(
                "[gaudium] method: {} | launches: {} | tokens/launch: {:.3} | ms/launch: {:.2}{}{}",
                report.method,
                report.launches,
                report.tokens_per_launch,
                report.ms_per_step,
                if report.drafted > 0 {
                    format!(
                        " | drafts accepted: {}/{} ({:.1}%)",
                        report.draft_accepted,
                        report.drafted,
                        100.0 * report.draft_accepted as f64 / report.drafted as f64
                    )
                } else {
                    String::new()
                },
                if report.accept_per_pos.is_empty() {
                    String::new()
                } else {
                    format!(
                        " | accept/pos: {}",
                        report
                            .accept_per_pos
                            .iter()
                            .map(|p| format!("{p:.2}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                }
            );
        }
        if report.sample_check_rows > 0 {
            eprintln!("[gaudium] sample-check:{}", check_note(&report));
        }
        if let Some(k0) = &k0_ref {
            let got = &report.generated_tokens;
            match k0.iter().zip(got.iter()).position(|(a, b)| a != b) {
                None if k0.len() == got.len() => eprintln!(
                    "[gaudium] IDENTITY vs K=0 ({}): identical, {} tokens",
                    sampling_label(params),
                    got.len()
                ),
                None => eprintln!(
                    "[gaudium] IDENTITY vs K=0 ({}): LENGTH DIFFERS (K=0 {} tokens, speculative {}), common prefix identical",
                    sampling_label(params),
                    k0.len(),
                    got.len()
                ),
                Some(i) => {
                    eprintln!(
                        "[gaudium] IDENTITY vs K=0 ({}): FIRST DIVERGENCE at generated index {i}: K=0 {} vs speculative {} (of {} / {} tokens)",
                        sampling_label(params),
                        k0[i],
                        got[i],
                        k0.len(),
                        got.len()
                    );
                    // Margin trace of a greedy fork: the plain K=0 loop teacher-forced along the
                    // run's own stream up to the fork, which records both tokens' logits there.
                    if params.temperature <= 0.0 && i > 0 {
                        let mut pt = params.clone();
                        pt.non_speculative = true;
                        pt.k0_plain = true;
                        pt.force_draft_k = None;
                        pt.max_new = i as u64 + 1;
                        pt.stop_on_eos = false;
                        pt.force_tokens = Some(got[..=i].to_vec());
                        if let Some(v) = vision_in.as_ref().filter(|_| prompt_idx == 0) {
                            engine.set_request_images(v.images.clone())?;
                        }
                        let rt = engine.generate_cancellable(
                            prompt_ids,
                            &pt,
                            Some(gaudium::signals::term_flag()),
                            &mut |_| {},
                        )?;
                        match rt.forks.iter().find(|f| f.pos == i) {
                            Some(f) => eprintln!(
                                "[gaudium] FORK TRACE at generated index {i}: K=0 token {} logit {:.4} vs the run's token {} logit {:.4} (margin {:.4} BF16 logits; {} earlier disagreements on the forced prefix)",
                                f.engine_token,
                                f.engine_logit,
                                f.forced_token,
                                f.forced_logit,
                                f.engine_logit - f.forced_logit,
                                rt.forks.iter().filter(|x| x.pos < i).count()
                            ),
                            None => eprintln!(
                                "[gaudium] FORK TRACE at generated index {i}: the teacher-forced K=0 pass agreed with the run's token ({} disagreements elsewhere)",
                                rt.forks.len()
                            ),
                        }
                    }
                }
            }
        }
        // Images belong to the first request only (reported once).
        if prompt_idx == 0 && set_idx == 0 {
            if let Some(v) = engine.last_vision_report() {
                eprintln!(
                    "{}",
                    gaudium::vision::cli::format_vision_report(&v, &report)
                );
            }
            if let Some(path) = &dump_vision_emb {
                let emb = engine.read_vision_embeddings()?;
                std::fs::write(path, &emb)
                    .map_err(|e| format!("write --dump-vision-emb {path}: {e}"))?;
                eprintln!(
                    "[gaudium] wrote {} bytes of image embeddings to {path}",
                    emb.len()
                );
            }
        }
    }
    let params = settings[0].clone();
    gaudium::vision::cli::run_extra_requests(
        engine.as_mut(),
        vision_in.as_ref().map(|v| v.images.as_slice()),
        &all_prompts[0],
        &params,
        repeat,
        after_ids_file.as_deref(),
        gaudium::signals::term_flag(),
    )?;
    Ok(())
}

fn parse_switch(v: &str, flag: &str) -> Result<bool, String> {
    match v {
        "0" | "off" | "false" => Ok(false),
        "1" | "on" | "true" => Ok(true),
        other => Err(format!("invalid {flag} '{other}' (0 or 1)")),
    }
}

/// One `--sampling` value `T[:top_k[:top_p[:min_p[:seed[:coupled[:check]]]]]]` on top of the
/// base parameters (omitted fields: top_k 0, top_p 1, min_p 0, the base seed, coupling and
/// sample check).
fn parse_sampling(raw: &str, base: &Qwen3GenParams) -> Result<Qwen3GenParams, String> {
    let bad = || {
        format!("invalid --sampling '{raw}' (T[:top_k[:top_p[:min_p[:seed[:coupled[:check]]]]]])")
    };
    let f: Vec<&str> = raw.split(':').collect();
    if f.is_empty() || f.len() > 7 {
        return Err(bad());
    }
    let mut p = base.clone();
    p.temperature = f[0].trim().parse().map_err(|_| bad())?;
    p.top_k = match f.get(1) {
        Some(v) => v.trim().parse().map_err(|_| bad())?,
        None => 0,
    };
    p.top_p = match f.get(2) {
        Some(v) => v.trim().parse().map_err(|_| bad())?,
        None => 1.0,
    };
    p.min_p = match f.get(3) {
        Some(v) => v.trim().parse().map_err(|_| bad())?,
        None => 0.0,
    };
    if let Some(v) = f.get(4) {
        p.seed = v.trim().parse().map_err(|_| bad())?;
    }
    if let Some(v) = f.get(5) {
        p.coupled = parse_switch(v.trim(), "--sampling coupled")?;
    }
    if let Some(v) = f.get(6) {
        p.sample_check = parse_switch(v.trim(), "--sampling check")?;
    }
    Ok(p)
}

fn sampling_label(p: &Qwen3GenParams) -> String {
    if p.temperature > 0.0 {
        format!(
            "T={} top_k={} top_p={} min_p={} seed={} coupled={}{}",
            p.temperature,
            p.top_k,
            p.top_p,
            p.min_p,
            p.seed,
            p.coupled as u8,
            if p.sample_check { " sample-check" } else { "" }
        )
    } else {
        "greedy".to_string()
    }
}

fn check_note(r: &gaudium::engine::GenReport) -> String {
    if r.sample_check_rows == 0 {
        return String::new();
    }
    format!(
        " {} rows sampled again on the host, {} mismatches",
        r.sample_check_rows, r.sample_check_mismatches
    )
}

/// FP8 calibration pass: prefill of every given prompt on the BF16 graphs, recording the amax of
/// each GEMM input site, then one table of static power-of-two activation scales.
fn run_fp8_calibrate(args: &[String]) -> Result<(), String> {
    let mut model_dir: Option<PathBuf> = env::var_os("GAUDIUM_MODEL").map(PathBuf::from);
    let mut kernels_dir: Option<PathBuf> = env::var_os("GAUDIUM_KERNELS").map(PathBuf::from);
    let mut id_files: Vec<String> = Vec::new();
    let mut text_files: Vec<String> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut max_ctx = KV_CHUNK;
    let mut cache_dir: Option<PathBuf> = None;
    let mut no_cache = false;
    let mut verbose = false;

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
            "--fp8-calibrate" => {}
            "--model" => model_dir = Some(PathBuf::from(next_val(&mut i, "--model")?)),
            "--kernels" => kernels_dir = Some(PathBuf::from(next_val(&mut i, "--kernels")?)),
            "--prompt-ids-file" => id_files.push(next_val(&mut i, "--prompt-ids-file")?),
            "--prompt-file" => text_files.push(next_val(&mut i, "--prompt-file")?),
            "--out" => out = Some(PathBuf::from(next_val(&mut i, "--out")?)),
            "--max-ctx" => {
                max_ctx = next_val(&mut i, "--max-ctx")?
                    .parse()
                    .map_err(|_| "invalid --max-ctx".to_string())?;
            }
            "--cache-dir" => cache_dir = Some(PathBuf::from(next_val(&mut i, "--cache-dir")?)),
            "--no-cache" => no_cache = true,
            "--verbose" | "-v" => verbose = true,
            other => return Err(format!("unknown --fp8-calibrate argument '{other}'")),
        }
        i += 1;
    }
    let model_dir =
        model_dir.ok_or_else(|| "missing --model <dir> (or GAUDIUM_MODEL env var)".to_string())?;
    let kernels_dir = kernels_dir
        .ok_or_else(|| "missing --kernels <dir> (or GAUDIUM_KERNELS env var)".to_string())?;
    let out = out.unwrap_or_else(|| default_fp8_dir(&model_dir).join(ACT_TABLE_FILE));

    let mut prompts: Vec<(String, Vec<i32>)> = Vec::new();
    for path in &id_files {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("read --prompt-ids-file {path}: {e}"))?;
        let ids = raw
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                s.trim()
                    .parse::<i32>()
                    .map_err(|_| format!("invalid token id '{s}' in {path}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        prompts.push((path.clone(), ids));
    }
    if !text_files.is_empty() {
        let tok = Tokenizer::from_model_dir(&model_dir).map_err(|e| e.to_string())?;
        for path in &text_files {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("read --prompt-file {path}: {e}"))?;
            prompts.push((path.clone(), tok.encode(&text, true)));
        }
    }
    if prompts.is_empty() {
        return Err("--fp8-calibrate needs at least one --prompt-ids-file or --prompt-file".into());
    }

    let mut cfg = Qwen3Config::new(model_dir, kernels_dir);
    cfg.max_ctx = max_ctx;
    cfg.cache_dir = cache_dir;
    cfg.no_cache = no_cache;
    cfg.verbose = verbose;
    cfg.precision = WeightPrecision::Bf16;
    cfg.fp8_calibrate = true;

    gaudium::signals::install();
    let mut engine: Box<dyn GenEngine> = load_engine(&cfg)?;
    let params = Qwen3GenParams {
        max_new: 1,
        stop_on_eos: false,
        ..Qwen3GenParams::default()
    };
    let mut rows = 0usize;
    for (name, ids) in &prompts {
        if gaudium::signals::term_requested() {
            return Err("cancelled before every prompt was measured: no table written".into());
        }
        let report = engine.generate_cancellable(
            ids,
            &params,
            Some(gaudium::signals::term_flag()),
            &mut |_| {},
        )?;
        if report.stopped_by_cancel {
            return Err("cancelled before every prompt was measured: no table written".into());
        }
        rows += ids.len();
        eprintln!(
            "[calibrate] {name}: {} prompt tokens, prefill {:.1} ms",
            ids.len(),
            report.ttft_ms
        );
    }
    let table = engine
        .fp8_calibration()
        .ok_or_else(|| "this model's engine has no FP8 calibration pass".to_string())?;
    if table.is_empty() {
        return Err("calibration measured no site (prompts of one token run no prefill)".into());
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    store_act_scales(&out, &table).map_err(|e| format!("write {}: {e}", out.display()))?;
    if verbose {
        for (site, a) in &table {
            eprintln!("[calibrate] {site} k={} amax={} rms={}", a.k, a.amax, a.rms);
        }
    }
    let (kmin, kmax) = table.values().fold((i32::MAX, i32::MIN), |(lo, hi), a| {
        (lo.min(a.k), hi.max(a.k))
    });
    println!(
        "FP8 CALIBRATION OK: {} sites from {} prompts ({} rows), k in [{kmin}, {kmax}] -> {}",
        table.len(),
        prompts.len(),
        rows,
        out.display()
    );
    Ok(())
}

/// Host-only FP8 pre-pack: quantises a Gemma 4 checkpoint's matrices into `--fp8-cache <dir>`.
fn run_fp8_pack(args: &[String]) -> Result<(), String> {
    let mut model_dir: Option<PathBuf> = env::var_os("GAUDIUM_MODEL").map(PathBuf::from);
    let mut cache: Option<PathBuf> = env::var_os("GAUDIUM_FP8_CACHE").map(PathBuf::from);
    let mut i = 0usize;
    while i < args.len() {
        let val = |i: usize| {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("missing value for {}", args[i]))
        };
        match args[i].as_str() {
            "--fp8-pack" => {}
            "--model" => {
                model_dir = Some(PathBuf::from(val(i)?));
                i += 1;
            }
            "--fp8-cache" => {
                cache = Some(PathBuf::from(val(i)?));
                i += 1;
            }
            other => return Err(format!("unknown --fp8-pack argument '{other}'")),
        }
        i += 1;
    }
    let model_dir =
        model_dir.ok_or_else(|| "missing --model <dir> (or GAUDIUM_MODEL env var)".to_string())?;
    let cache = cache
        .ok_or_else(|| "missing --fp8-cache <dir> (or GAUDIUM_FP8_CACHE env var)".to_string())?;
    let st = gaudium::engine::gemma4_fp8::prepack_fp8_cache(&model_dir, &cache)?;
    println!(
        "FP8 PACK OK: {} matrices ({:.2} G codes), {} already cached, {:.1} s -> {}",
        st.matrices,
        st.codes as f64 / 1e9,
        st.from_cache,
        st.seconds,
        cache.display()
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

fn run_serve(args: &[String]) -> Result<(), String> {
    let mut host = env::var("GAUDIUM_SERVE_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let mut port: u16 = env::var("GAUDIUM_SERVE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8090);
    let mut models = Vec::new();
    let mut kernels_dir: Option<PathBuf> = env::var_os("GAUDIUM_KERNELS").map(PathBuf::from);
    let mut ui_dir: Option<PathBuf> = env::var_os("GAUDIUM_UI_DIR").map(PathBuf::from);
    let mut max_ctx = KV_CHUNK;
    let mut cache_dir: Option<PathBuf> = None;
    let mut no_cache = false;
    let mut lazy = false;
    let mut idle_park_secs = 300u64;
    // the card-queue settings file: flag, then env, then `queue.conf` in the card-lock state directory
    let mut queue_config: Option<PathBuf> = env::var_os("GAUDIUM_QUEUE_CONFIG")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("GAUDI_LIVE_DIR")
                .map(PathBuf::from)
                .and_then(|d| d.parent().map(|p| p.join("queue.conf")))
        });
    let mut api_key: Option<String> = env::var("GAUDIUM_API_KEY").ok().filter(|s| !s.is_empty());
    let mut verbose = false;
    let mut per_model: Vec<(String, ModelRunOptions)> = Vec::new();
    let mut draft_k: Vec<u64> = Vec::new();
    // `<id>=<path>` options are collected per served model id.
    fn model_opt<'a>(
        per_model: &'a mut Vec<(String, ModelRunOptions)>,
        raw: &str,
        flag: &str,
    ) -> Result<(&'a mut ModelRunOptions, PathBuf), String> {
        let (id, path) = raw
            .split_once('=')
            .filter(|(id, path)| !id.is_empty() && !path.is_empty())
            .ok_or_else(|| format!("{flag} expects <model id>=<path>, got '{raw}'"))?;
        let idx = match per_model.iter().position(|(k, _)| k == id) {
            Some(i) => i,
            None => {
                per_model.push((id.to_string(), ModelRunOptions::default()));
                per_model.len() - 1
            }
        };
        Ok((&mut per_model[idx].1, PathBuf::from(path)))
    }

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
            "--serve" => {}
            "--draft-model" => {
                let raw = next_val(&mut i, "--draft-model")?;
                let (opts, path) = model_opt(&mut per_model, &raw, "--draft-model")?;
                if path.as_os_str() == "mtp" {
                    opts.own_mtp = true;
                } else {
                    opts.draft_model = Some(path);
                }
            }
            "--moe-form" => {
                let raw = next_val(&mut i, "--moe-form")?;
                let (opts, form) = model_opt(&mut per_model, &raw, "--moe-form")?;
                opts.moe_routed = match form.to_str() {
                    Some("routed") => true,
                    Some("dense") => false,
                    _ => {
                        return Err(format!(
                            "--moe-form expects <model id>=dense|routed, got '{raw}'"
                        ))
                    }
                };
            }
            "--model-ctx" => {
                let raw = next_val(&mut i, "--model-ctx")?;
                let (opts, n) = model_opt(&mut per_model, &raw, "--model-ctx")?;
                opts.max_ctx = Some(
                    n.to_str()
                        .and_then(|s| s.parse::<u64>().ok())
                        .filter(|&n| n > 0)
                        .ok_or_else(|| {
                            format!("--model-ctx expects <model id>=<tokens>, got '{raw}'")
                        })?,
                );
            }
            "--fp8-act" => {
                let raw = next_val(&mut i, "--fp8-act")?;
                let (opts, path) = model_opt(&mut per_model, &raw, "--fp8-act")?;
                opts.fp8_act = Some(path);
            }
            "--fp8-cache" => {
                let raw = next_val(&mut i, "--fp8-cache")?;
                let (opts, path) = model_opt(&mut per_model, &raw, "--fp8-cache")?;
                opts.fp8_cache = Some(path);
            }
            "--draft-k" => {
                draft_k = next_val(&mut i, "--draft-k")?
                    .split(',')
                    .map(|s| s.trim().parse::<u64>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "invalid --draft-k".to_string())?;
            }
            "--host" => host = next_val(&mut i, "--host")?,
            "--port" => {
                let s = next_val(&mut i, "--port")?;
                port = s.parse().map_err(|_| format!("invalid --port '{s}'"))?;
            }
            "--model" => {
                let s = next_val(&mut i, "--model")?;
                models.push(parse_model_arg(&s));
            }
            "--kernels" => kernels_dir = Some(PathBuf::from(next_val(&mut i, "--kernels")?)),
            "--ui-dir" => ui_dir = Some(PathBuf::from(next_val(&mut i, "--ui-dir")?)),
            "--max-ctx" => {
                let s = next_val(&mut i, "--max-ctx")?;
                max_ctx = s.parse().map_err(|_| format!("invalid --max-ctx '{s}'"))?;
            }
            "--cache-dir" => cache_dir = Some(PathBuf::from(next_val(&mut i, "--cache-dir")?)),
            "--no-cache" => no_cache = true,
            "--lazy" => lazy = true,
            "--idle-park" => {
                let s = next_val(&mut i, "--idle-park")?;
                idle_park_secs = s
                    .parse()
                    .map_err(|_| format!("invalid --idle-park '{s}'"))?;
            }
            "--queue-config" => {
                queue_config = Some(PathBuf::from(next_val(&mut i, "--queue-config")?))
            }
            "--api-key" => api_key = Some(next_val(&mut i, "--api-key")?),
            "--verbose" => verbose = true,
            other => return Err(format!("unknown --serve flag '{other}'")),
        }
        i += 1;
    }

    if models.is_empty() {
        if let Ok(m) = env::var("GAUDIUM_MODEL") {
            if !m.is_empty() {
                models.push(parse_model_arg(&m));
            }
        }
    }
    let kernels_dir = kernels_dir.ok_or("missing --kernels <dir> (or GAUDIUM_KERNELS)")?;
    if ui_dir.is_none() {
        let default_ui = PathBuf::from("ui");
        if default_ui.is_dir() {
            ui_dir = Some(default_ui);
        }
    }

    run_server(ServerConfig {
        host,
        port,
        models,
        kernels_dir,
        ui_dir,
        max_ctx,
        cache_dir,
        no_cache,
        lazy,
        idle_park_secs,
        queue_config,
        api_key,
        verbose,
        per_model,
        draft_k,
    })
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
    if args.iter().any(|a| a == "--fp8-pack") {
        if let Err(e) = run_fp8_pack(&args[1..]) {
            eprintln!("FP8 PACK FAIL: {e}");
            process::exit(1);
        }
        return;
    }
    if args.iter().any(|a| a == "--fp8-calibrate") {
        if let Err(e) = run_fp8_calibrate(&args[1..]) {
            eprintln!("FP8 CALIBRATE FAIL: {e}");
            process::exit(1);
        }
        return;
    }
    if args
        .iter()
        .any(|a| a == "--generate" || a == "--build-cache")
    {
        if let Err(e) = run_generate(&args[1..]) {
            eprintln!("GENERATE FAIL: {e}");
            process::exit(1);
        }
        return;
    }
    if args.iter().any(|a| a == "--serve") {
        if let Err(e) = run_serve(&args[1..]) {
            eprintln!("SERVE FAIL: {e}");
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
