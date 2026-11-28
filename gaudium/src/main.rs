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
use gaudium::server::{
    parse_model_arg, run_server, ModelRunOptions, ProxyOptions, ServerConfig,
};
use gaudium::synapse::device::HpuDevice;
use gaudium::synapse::ffi::{SynDataType, SynapseApi};
use gaudium::synapse::graph::GraphBuilder;
use gaudium::synapse::recipe_cache::RecipeCache;
use gaudium::synapse::recipe_trim;
use gaudium::tokenizer::{
    format_gemma4_chat, format_qwen3_chat, gemma4_template_closes_thought, StreamDecoder,
};
use gaudium::verified::{DevEvent, HbmCat, ReqEvent};

/// Most kept prefix positions `--prefix-cache` accepts (a lane plan multiplies them by the lanes
/// and the slot size in checked arithmetic; far more than any card holds).
const MAX_PREFIX_SLOTS: u64 = 4096;

/// The line `gaudium --version` prints: the crate version, then the commit the binary was built
/// from when the build was given one in `GAUDIUM_BUILD_COMMIT`.
fn version_line() -> String {
    let version = env!("CARGO_PKG_VERSION");
    match option_env!("GAUDIUM_BUILD_COMMIT") {
        Some(commit) if !commit.trim().is_empty() => format!("gaudium {version} ({})", commit.trim()),
        _ => format!("gaudium {version}"),
    }
}

fn print_usage() {
    let store_n = gaudium::server::DEFAULT_RESPONSE_STORE_ENTRIES;
    let store_mb = gaudium::server::DEFAULT_RESPONSE_STORE_MIB;
    println!(
        "Gaudium — Intel Gaudi 2 Inference Engine (RustyBends v1)\n\
         \n\
         Usage:\n\
         \x20 gaudium                                   Run verified-core self-check\n\
         \x20 gaudium --inspect-model <dir|file.gguf>   Validate config.json & Safetensors, or a GGUF\n\
         \x20                                           model (a .gguf file, any shard of a split one,\n\
         \x20                                           or a directory of them without config.json):\n\
         \x20                                           metadata, tensors and weights per type\n\
         \x20 gaudium --tokenize <model_dir> <text>     Encode & round-trip decode via tokenizer.json\n\
         \x20                                           (or a GGUF model's tokenizer metadata)\n\
         \x20 gaudium --hpu-smoke                       Run live Gaudi 2 BF16 add smoke test\n\
         \x20 gaudium --probe-copy [GiB]                Measure copies between the card and host memory\n\
         \x20                                           (both directions, through the pinned transfer ring\n\
         \x20                                           the host-RAM prefix cache uses; opens the card;\n\
         \x20                                           default: 2 GiB of device memory)\n\
         \x20 gaudium --apply-template <model_dir> <request.json|->\n\
         \x20                                           Print the prompt a chat request renders to with\n\
         \x20                                           the model's own chat template (host only; a GGUF\n\
         \x20                                           model: its tokenizer.chat_template)\n\
         \x20 gaudium --generate [options]              Run live Gaudi 2 LLM generation\n\
         \x20 gaudium --build-cache [generate options]  Compile every recipe of that configuration into\n\
         \x20                                           the recipe cache and print its exact HBM plan,\n\
         \x20                                           without opening the card (Qwen and Gemma 4\n\
         \x20                                           engines)\n\
         \x20 gaudium --serve [options]                 Start OpenAI HTTP/SSE + Gaudium Web UI server\n\
         \x20 gaudium --fp8-calibrate [options]         Measure a model's FP8 activation scales (BF16 graphs)\n\
         \x20 gaudium --kv-calibrate [options]          Measure the K/V maxima a model's FP8 K/V scale tables\n\
         \x20                                           are made from (BF16 caches, read back per prompt)\n\
         \x20 gaudium --vision-suite <suite.json> [options] [-- <generate options>]\n\
         \x20                                           Vision regression suite: every model and\n\
         \x20                                           speculative method of the suite file with its\n\
         \x20                                           fixed images (gaudium/tests/vision_suite), one\n\
         \x20                                           --generate process per row, one after another\n\
         \x20                                           (this process never opens the card)\n\
         \x20 gaudium --trim-cache --cache-max-gb <N> [options]\n\
         \x20                                           Remove the least recently used recipes of the\n\
         \x20                                           recipe cache until it fits N GiB (host only;\n\
         \x20                                           safe beside a server using the directory)\n\
         \x20 gaudium --make-cert <dir> [options]       Make the server's HTTPS certificate (host only): a\n\
         \x20                                           local certificate authority, made once and trusted\n\
         \x20                                           once on each device, and a server certificate\n\
         \x20                                           signed by it, for --tls-cert and --tls-key\n\
         \x20 gaudium --fp8-pack --model <dir> --fp8-cache <dir>\n\
         \x20                                           Quantise a Gemma 4 BF16 checkpoint's matrices into an\n\
         \x20                                           FP8 cache directory (host only, no card)\n\
         \x20 gaudium --help | -h                       Print this text\n\
         \x20 gaudium --version | -V                    Print the version (and the commit of a release build)\n\
         \n\
         Serve options:\n\
         \x20 --host <addr>         Bind address (or GAUDIUM_SERVE_HOST; default: 0.0.0.0)\n\
         \x20 --port <N>            Bind port (or GAUDIUM_SERVE_PORT; default: 8090)\n\
         \x20 --model <dir|id=dir>  Model directory (repeatable for multi-model hot-swap); a GGUF\n\
         \x20                       file (or a directory of one GGUF model) serves its own tokenizer,\n\
         \x20                       chat template, context limit and sampling defaults from its metadata\n\
         \x20 --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)\n\
         \x20 --ui-dir <dir>        Gaudium Web UI bundle directory (or GAUDIUM_UI_DIR)\n\
         \x20 --max-ctx <N>         Context capacity, multiple of 4096 (default: 4096)\n\
         \x20 --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)\n\
         \x20 --no-cache            Disable on-disk compiled recipe cache\n\
         \x20 --cache-max-gb <N>    Trim the recipe cache to N GiB (decimals allowed) in the\n\
         \x20                       background: once the first model load is done (at start with\n\
         \x20                       --lazy), then every --cache-trim-every minutes, least recently\n\
         \x20                       used recipes first (see --trim-cache; default: never trimmed).\n\
         \x20                       Recipes this server has loaded or compiled are never removed\n\
         \x20 --cache-keep-days <D> With --cache-max-gb: recipes used within D days are kept even\n\
         \x20                       over the budget (default: 1; decimals allowed)\n\
         \x20 --cache-trim-every <M> With --cache-max-gb: minutes between trims (default: 60)\n\
         \x20 --profile-steps <A-B[:file]>  Device trace of the batched launches A..B (0-based,\n\
         \x20                       counted from the load) of a model served with --parallel\n\
         \x20                       lanes: prints each launch's split by node family (a slot's\n\
         \x20                       nodes fold together as s#_) and engine; :file also writes the\n\
         \x20                       raw events. Repeatable. The batched recipes compile fresh\n\
         \x20                       (the others load from the recipe cache); a profiled run is\n\
         \x20                       not a timing result\n\
         \x20 --lazy                Start without holding Gaudi 2; load on first chat/start\n\
         \x20 --build-cache         With --serve: compile every recipe the served models load into\n\
         \x20                       the recipe cache and exit, without opening the card and without\n\
         \x20                       listening (safe beside a running server). Each model takes the\n\
         \x20                       engine configuration a load of it takes, so a start after it\n\
         \x20                       compiles nothing. Run it with the server's own options after any\n\
         \x20                       change to kernels, graphs or contexts\n\
         \x20 --build-cache-only <id>  The same for one served model id (repeatable)\n\
         \x20 --idle-park <secs>    Auto-release Gaudi 2 after N seconds idle (0=off, default: 300)\n\
         \x20 --prefix-cache <N>    Prefix reuse across requests: a request whose prompt starts like\n\
         \x20                       the one before it prefills only the rest, with the same output\n\
         \x20                       as a whole prefill. N = positions of the previous prompt whose\n\
         \x20                       recurrent state is kept on the card to resume from (default: 8;\n\
         \x20                       0 = off; at most 4096; 0.15 GiB each for a 48-layer DeltaNet\n\
         \x20                       model). The\n\
         \x20                       context is planned first: a load keeps as many as fit beside\n\
         \x20                       it and says so. A request with cache_prompt: false prefills\n\
         \x20                       its whole prompt. Dense Qwen3.5 / Qwen3.8 models without a\n\
         \x20                       drafter or with DSpine\n\
         \x20                       (N positions), and Inkling 2 (any N > 0: the rows on the card\n\
         \x20                       are resumed while the bounded rings still hold what the rest\n\
         \x20                       of the prompt reads, and a copy of the rings taken when each\n\
         \x20                       prompt is in, 0.6 GiB, covers a long generation after it)\n\
         \x20 --parallel <N>        Requests a model runs at once (default: 1; at most 16): each in a\n\
         \x20                       lane of its own (its own K/V rows, recurrent state and prefix\n\
         \x20                       reuse), the decoding lanes stepped together in one launch that\n\
         \x20                       reads the weights once. Every stream is token-identical to the\n\
         \x20                       same request run alone. Further requests wait in arrival order;\n\
         \x20                       a request with images runs alone; one with sampling penalties,\n\
         \x20                       logit_bias or structured output is refused (they apply one\n\
         \x20                       request at a time, at K = 0). Each lane holds the full\n\
         \x20                       context (--max-ctx / --model-ctx): a load that cannot plan N\n\
         \x20                       lanes of it is refused with the numbers. A request goes to the\n\
         \x20                       free lane where it takes over the most of its prompt for the\n\
         \x20                       least of another conversation's rows. Dense Qwen3 /\n\
         \x20                       Qwen3.5 / Qwen3.8 models without a drafter or with a DSpine\n\
         \x20                       drafter whose verify rows are replayed or checkpointed:\n\
         \x20                       lanes take the replay form (no per-row state copies, about\n\
         \x20                       2 GiB less per lane on the 27B) unless GAUDIUM_GDN_VERIFY\n\
         \x20                       names a form. With replay each lane verifies at a width of\n\
         \x20                       its own, the narrowest or the widest --draft-k, chosen every\n\
         \x20                       step from its own acceptance and the measured step times;\n\
         \x20                       with checkpoint every lane steps at the narrowest. Other\n\
         \x20                       models are\n\
         \x20                       served with one lane (the log says why). /props total_slots\n\
         \x20                       and /slots show the lanes. With --cache-ram every lane saves\n\
         \x20                       to and restores from the one host-RAM store (an entry one\n\
         \x20                       lane saved is restored into any lane)\n\
         \x20 --cache-ram <GB>      Host memory for prefix reuse when the card no longer holds a\n\
         \x20                       conversation: after another conversation, another model, or\n\
         \x20                       the idle release (--idle-park). What a request could resume\n\
         \x20                       from (K/V rows of the prompt, the state at its kept positions)\n\
         \x20                       is copied to host memory after the response, and copied back\n\
         \x20                       instead of prefilled when that is faster by the engine's own\n\
         \x20                       measured rates; the output is that of a whole prefill. GB may\n\
         \x20                       be a decimal; 0 = off. Default: a quarter of the host memory\n\
         \x20                       available at start (the smallest of what the host and the\n\
         \x20                       control groups above the process have left; version 1 and 2\n\
         \x20                       memory controllers are read). No default is taken, and the\n\
         \x20                       cache is off, when a memory limit of the process cannot be\n\
         \x20                       read: name a budget to turn it on there. Memory is taken as\n\
         \x20                       prefixes are saved and given back when they leave; a full\n\
         \x20                       cache evicts the entry used least recently. A budget that does\n\
         \x20                       not fit in what is available is refused at start with the\n\
         \x20                       numbers. While a model is loaded, 256 MiB of pinned transfer\n\
         \x20                       buffers come on top, and 128 MiB for a moment at each load;\n\
         \x20                       all of it is asked for against what the host has left at that\n\
         \x20                       moment, and an engine that cannot have it serves without the\n\
         \x20                       cache. Needs\n\
         \x20                       --prefix-cache > 0; about 64 KiB per token and 0.15 GiB per\n\
         \x20                       kept position for a 16-attention-layer, 48-DeltaNet-layer model\n\
         \x20 --response-store <N>  Responses POST /v1/responses keeps in host memory for GET\n\
         \x20                       /v1/responses/{{id}}, its input_items and previous_response_id\n\
         \x20                       (default {store_n}; 0 = off: a request with\n\
         \x20                       previous_response_id is refused). A request with store: false\n\
         \x20                       is never kept, nor one deleted while it is generated (the\n\
         \x20                       DELETE cancels it). When the store is full the oldest responses\n\
         \x20                       leave first; a response larger than the whole store is not kept.\n\
         \x20                       Response ids carry 128 random bits\n\
         \x20 --response-store-mb <MiB>\n\
         \x20                       Bytes the response store may hold, in MiB (default\n\
         \x20                       {store_mb}). Each response is held as serialised JSON (its\n\
         \x20                       response object and its conversation's input items) and\n\
         \x20                       counts those bytes, its id three times and 256 bytes of\n\
         \x20                       overhead. The request contexts of the responses being\n\
         \x20                       generated are counted apart and bounded to this size plus the\n\
         \x20                       256 MiB request limit; a request past that bound gets a 503\n\
         \x20 --queue-config <file> Card-queue settings, re-read on every turn (or GAUDIUM_QUEUE_CONFIG;\n\
         \x20                       default: queue.conf in the parent of GAUDI_LIVE_DIR, the card-lock\n\
         \x20                       state dir). Keys: group_max_wait_min (default 45), group_max_skips\n\
         \x20                       (default 3): a request for the loaded model may run ahead of older\n\
         \x20                       requests until one of them has waited that long or been passed over\n\
         \x20                       that many times (a request a --parallel lane session admits\n\
         \x20                       counts too: the session then lets its lanes drain and hands\n\
         \x20                       the card on)\n\
         \x20 --draft-model <id=dir> Drafter checkpoint for model <id>: speculative decoding, output\n\
         \x20                       token-identical to K=0 (repeatable, one per model). The kind\n\
         \x20                       follows the directory name: a block drafter when it contains\n\
         \x20                       dflash (7 drafts per launch), a model's own drafter when it\n\
         \x20                       ends in -assistant (4 drafts), otherwise DSpine. <id>=mtp\n\
         \x20                       drafts with the model's own MTP head (no checkpoint; 4 drafts,\n\
         \x20                       3 in the routed mixture-of-experts form; an Inkling 2 model:\n\
         \x20                       its block drafter, 7 drafts, as --spec mtp finds it).\n\
         \x20                       A request with presence / frequency / repetition penalties\n\
         \x20                       or logit_bias decodes at K=0 (its tokens are picked on the\n\
         \x20                       host from the read-back logits, over every generated token,\n\
         \x20                       reasoning included). So does structured output (chat\n\
         \x20                       response_format json_object / json_schema, function tools\n\
         \x20                       with \"strict\": true, Messages output_config.format and strict\n\
         \x20                       tools, Responses text.format and strict function tools): each\n\
         \x20                       answer token is masked to what the schema admits;\n\
         \x20                       the reasoning stays free\n\
         \x20 --moe-form <id=f>     Mixture-of-experts model <id>: dense (default) or routed (faster\n\
         \x20                       decode, more device memory, its own token stream)\n\
         \x20 --model-ctx <id=N>    Context size for model <id> instead of --max-ctx\n\
         \x20 --effort-levels <id=l,..>  Reasoning-effort levels of model <id> instead of the ones\n\
         \x20                       read from its chat template (each one the template renders\n\
         \x20                       without raising): levels of minimal, low, medium, high,\n\
         \x20                       xhigh, max, or none for no effort setting. /props advertises\n\
         \x20                       them; a request's reasoning_effort goes to the nearest one\n\
         \x20 --precision <id=p>    Weight precision of model <id>: auto (default), bf16 or fp8 (a\n\
         \x20                       BF16 checkpoint is quantised at load per output channel; give\n\
         \x20                       its activation-scale table with --fp8-act)\n\
         \x20 --draft-k <N[,N..]>   DSpine drafts verified per launch (default: 7,15 adaptive)\n\
         \x20 --no-vision           Load no vision tower: every model is served text only. By default\n\
         \x20                       a model whose checkpoint has a vision tower takes images (JPEG,\n\
         \x20                       PNG, GIF, WebP, BMP as data: URLs) and videos (an animated GIF /\n\
         \x20                       WebP, or a list of frames with their rate) in chat requests.\n\
         \x20                       A GGUF model's tower is a projector file (mmproj*.gguf) beside\n\
         \x20                       it; with none there it is served text only.\n\
         \x20                       Drafters that run with image input: DSpine, a dense Qwen\n\
         \x20                       model's DFlash2 drafter and MTP head, Inkling 2's own drafter\n\
         \x20                       and Gemma 4's -assistant drafter; a model with another drafter\n\
         \x20                       is served text only (Gemma 4 takes images, no video)\n\
         \x20 --vision-max-patches <N>  Largest image the vision tower is planned for, in the\n\
         \x20                       model's patches (default: 4096; 16x16 pixels for Qwen3.5 / 3.8:\n\
         \x20                       1024x1024 pixels, larger images are resized down to it; 20x20\n\
         \x20                       for Inkling 2, which never resizes: an image of more than N/4\n\
         \x20                       tiles of 40x40 pixels is refused; Gemma 4: an image is resized\n\
         \x20                       to whole 48x48-pixel soft tokens of 9 patches, at most N/9 and\n\
         \x20                       at most the checkpoint's own budget). The tower's device memory\n\
         \x20                       grows with N^2 (about 160 bytes x N^2: 2.5 GiB at 4096)\n\
         \x20 --vision-max-tokens <N>  Image tokens one request may hold, all images and video\n\
         \x20                       frames together (default: 16384; 10 KiB of device memory each)\n\
         \x20 --model-vision <id=N|off>  The vision tower of model <id> instead of the two options\n\
         \x20                       above: its largest image in patches, or off for text only\n\
         \x20 --fp8-act <id=file>   FP8 activation-scale table for FP8 model <id> (repeatable)\n\
         \x20 --fp8-cache <id=dir>  Cache directory for model <id>'s quantised FP8 matrices\n\
         \x20 --kv-cache <id=d>     K/V cache of model <id>: bf16 (default) or fp8 (half the bytes per\n\
         \x20                       position, so a longer context fits; its K=0 stream is its own and\n\
         \x20                       the reference of its drafter; dense Qwen3.5 / 3.8 hybrids, K=0 or\n\
         \x20                       DSpine; needs the scale table from --kv-calibrate)\n\
         \x20 --kv-scales <id=file> K/V calibration of model <id> (default:\n\
         \x20                       <model_dir>-gaudium-pc/kv_scales.txt)\n\
         \x20 --api-key <key>       Optional Bearer / X-Api-Key auth token (or GAUDIUM_API_KEY)\n\
         \x20                       Only the Web UI's files, GET /health and OPTIONS go without it\n\
         \x20 --a2ui                Offer A2UI 1.0 interfaces (or GAUDIUM_A2UI=1): a chat request that\n\
         \x20                       declares its renderer in the field gaudium_a2ui (true, or an\n\
         \x20                       object with supportedCatalogIds) gets the instructions that teach\n\
         \x20                       the model to write <a2ui-json> interface blocks in its reply;\n\
         \x20                       the Web UI then offers interactive interfaces in its settings.\n\
         \x20                       GET /gaudium/a2ui shows what is offered and the instructions.\n\
         \x20                       Requests that declare nothing are unchanged\n\
         \x20 --a2ui-prompt <file>  Read the A2UI instructions from <file> instead of the built-in\n\
         \x20                       text (or GAUDIUM_A2UI_PROMPT), and re-read it whenever it\n\
         \x20                       changes: edits apply to the next request, no restart. A file\n\
         \x20                       that is missing, empty, over 256 KiB, not UTF-8 text, or that\n\
         \x20                       never names the <a2ui-json> tags is not used: the last good\n\
         \x20                       version (or the built-in text) stays, the reason is logged and\n\
         \x20                       shown under instructions_source in GET /gaudium/a2ui\n\
         \x20 --tls-cert <file>     Serve HTTPS as well: the PEM certificate (with its chain) for\n\
         \x20 --tls-key <file>      the server, and its unencrypted PEM private key. HTTP and HTTPS\n\
         \x20                       then share the port, and a plain-HTTP request is answered with a\n\
         \x20                       redirect to https on the same host and port (see --tls-http).\n\
         \x20                       Opened over https the Web UI may use the microphone and the\n\
         \x20                       clipboard from other machines. TLS comes from the system's\n\
         \x20                       OpenSSL 3 (libssl.so.3), loaded when asked for: without it the\n\
         \x20                       server starts, says that HTTPS is not offered, and serves plain\n\
         \x20                       HTTP. Behind a front end that ends TLS itself give neither option\n\
         \x20 --tls-http <mode>     What plain-HTTP requests get while HTTPS is served: redirect\n\
         \x20                       (default) sends clients on other machines to https with a 308\n\
         \x20                       (method and body kept; a missing or malformed Host header is a\n\
         \x20                       400) and answers clients on this machine's loopback interface\n\
         \x20                       (local tools; their traffic never leaves the machine); strict\n\
         \x20                       redirects every client, loopback included; allow answers plain\n\
         \x20                       HTTP from every client as well (tools elsewhere that cannot\n\
         \x20                       speak HTTPS)\n\
         \x20 --hsts-max-age <secs>  max-age of the Strict-Transport-Security header every response\n\
         \x20                       over HTTPS carries (default: 86400, one day; at most 63072000).\n\
         \x20                       A browser that saw it uses https for this host name, on every\n\
         \x20                       port, until it expires; 0 tells browsers to forget it. No\n\
         \x20                       includeSubDomains, no preload. On --port 80 the default is 0\n\
         \x20                       and a nonzero value is refused (browsers would send http://\n\
         \x20                       links to port 443)\n\
         \x20 --mcp-proxy           Offer the MCP proxy (/cors-proxy) to the Web UI: the server\n\
         \x20                       makes the requests to an MCP server instead of the browser (for\n\
         \x20                       servers without CORS headers, servers only this machine\n\
         \x20                       reaches, and plain-http servers used from an https page). Off\n\
         \x20                       by default. It needs the targets it may reach (the two options\n\
         \x20                       below); requests need the API key when one is set; a page of\n\
         \x20                       another site cannot drive it, and a request must name this\n\
         \x20                       server (see --server-name). GET /gaudium/proxy shows its\n\
         \x20                       settings and last decisions\n\
         \x20 --mcp-proxy-allow <list>  Targets the proxy may reach, comma separated (repeatable):\n\
         \x20                       host, host:port, *.domain, an address, an address range\n\
         \x20                       (192.0.2.0/24), or * for any target (* needs --api-key). Each\n\
         \x20                       address a target resolves to must be public, and the\n\
         \x20                       connection goes to the address that was checked\n\
         \x20 --mcp-proxy-allow-private <list>  Targets that may be reached although their address\n\
         \x20                       is loopback, private or link-local: the same forms without *\n\
         \x20                       (localhost:8931, 127.0.0.1, 192.168.0.0/16). An entry here\n\
         \x20                       also allows the target, so an MCP server on this machine\n\
         \x20                       needs this option only\n\
         \x20 --server-name <list>  Names this server is opened by, comma separated (repeatable),\n\
         \x20                       for the MCP proxy: its requests must name the server (Host and\n\
         \x20                       any X-Forwarded-Host or Forwarded host) by an address,\n\
         \x20                       localhost, a name given here, or a DNS name of the HTTPS\n\
         \x20                       certificate; any other name is refused, so a page of another\n\
         \x20                       site cannot point its own name at this server (DNS rebinding).\n\
         \x20                       Give the name the Web UI is opened by, and a front end's\n\
         \x20                       public name\n\
         \x20 --mcp-proxy-max-mb <N>  Largest response the proxy passes on, in MiB (default: 64)\n\
         \x20 --mcp-proxy-timeout <secs>  Longest silence of a target, or wait on a page that\n\
         \x20                       stopped reading, before the proxy gives up (default: 600).\n\
         \x20                       An idle event stream gets a comment line every 20 s so\n\
         \x20                       that a front end does not cut it\n\
         \x20 --mcp-proxy-ca <file>  PEM certificate authorities the proxy trusts for https targets\n\
         \x20                       besides the system's (an MCP server with a private certificate)\n\
         \x20 --verbose             Print server, HBM, and recipe diagnostics\n\
         \x20 --debug-endpoints     Serve the debug endpoints: GET /gaudium/requests lists the\n\
         \x20                       last 64 chat requests (job, reply id, endpoint, model,\n\
         \x20                       streamed or not, still running or not, and the newest request\n\
         \x20                       event: phase, timings, token counts; never prompt or reply\n\
         \x20                       text). Off by default: the path answers 404. Needs the API key\n\
         \x20                       when one is set\n\
         \n\
         Live status environment variables:\n\
         \x20 GAUDI_LIVE_DIR        Directory for live dashboard JSON records (GAUDI_LIVE=0 disables).\n\
         \x20                       A directory that does not exist turns the records off\n\
         \x20 GAUDIUM_LIVE_ROOT     Used when GAUDI_LIVE_DIR is unset: the records go to\n\
         \x20                       <root>/.cardlock/live\n\
         \x20 GAUDI_LIVE_TAG        First part of the record's file name, <tag>.<pid>.json (default:\n\
         \x20                       the host name)\n\
         \x20 GAUDI_CARD_OWNER      Owner name written in the record and in the card lock's holder\n\
         \x20                       file (default: server/<engine>)\n\
         \x20 GAUDI_CARDLOCK_DIR    Card-lock directory that gets the holder file and the lock log\n\
         \x20                       (default: the parent of a records directory named `live`)\n\
         \x20 GAUDI_CARD_NO_HBM_REPORT  1 = do not publish the device memory in use to\n\
         \x20                       /dev/shm/mem_usage_accel<N>\n\
         \x20 GAUDI_CARD_ACCEL_INDEX  The <N> of that file (default: 0)\n\
         \n\
         Engine environment variables:\n\
         \x20 GAUDIUM_REQUEST_RESET Device state zeroed before each request: auto (default), dirty\n\
         \x20                       or full (see --request-reset under Generate options)\n\
         \x20 GAUDIUM_UPLOAD_DIRECT 0 = never read weight files with direct I/O at load (default:\n\
         \x20                       weight bytes that are not in the page cache are read directly)\n\
         \x20 GAUDIUM_ACQUIRE_RETRIES  Attempts to acquire the card at start, 2 s apart (default: 10)\n\
         \x20 GAUDIUM_SYNAPSE_LIB   The SynapseAI library to load, tried before\n\
         \x20                       /usr/lib/habanalabs/libSynapse.so and the loader's search path\n\
         \x20 GAUDIUM_<KERNEL>_ELF  One TPC kernel's ELF file, read by the graph-compiler glue. When\n\
         \x20                       unset, the engine sets it to that kernel's file in the kernels\n\
         \x20                       directory\n\
         \x20 GAUDIUM_W4_GLUE_DEBUG Set (any value) = the graph-compiler glue of the MoE W4 kernels\n\
         \x20                       prints the tensor shapes of a node it refuses\n\
         Read when the engine is built, not when it runs:\n\
         \x20 GAUDIUM_BUILD_COMMIT  The commit `gaudium --version` prints after the version (release\n\
         \x20                       builds set it; unset = the version alone)\n\
         SDK environment variables the engine reads and completes:\n\
         \x20 GC_KERNEL_PATH        The engine puts its glue libraries first, keeps the entries\n\
         \x20                       already there and adds the SDK's TPC kernel library\n\
         \x20 HABANA_SCAL_BIN_PATH / HABANA_PLUGINS_LIB_PATH\n\
         \x20                       Left as set; when unset, set to the SDK's usual directories if\n\
         \x20                       those exist\n\
         \x20 HABANA_PROFILE        The SDK profiler's settings. --profile-steps sets its own when\n\
         \x20                       this is unset\n\
         Measurement switches of the graph compiler (not for serving; each is part of the recipe\n\
         fingerprints it touches):\n\
         \x20 GAUDIUM_GC_SET        `NAME=value[,NAME=value..]`: graph-compiler settings applied to\n\
         \x20                       the graphs this process compiles (set for the compile, then\n\
         \x20                       restored); GAUDIUM_GC_ONLY limits them to recipes whose name\n\
         \x20                       starts with one of its `|`-separated prefixes; GAUDIUM_GC_TRACE\n\
         \x20                       prints each setting's previous value\n\
         \x20 GAUDIUM_HALFPROBE_GC  Sets of such settings (sets separated by `;`) that --prefill\n\
         \x20                       halfprobe=N times on the tile layers, each against the default\n\
         \x20 GAUDIUM_HALFPROBE_SEQS_LIN / GAUDIUM_HALFPROBE_SEQS_ATTN\n\
         \x20                       `name=order[;name=order..]`: node orders of the two-body\n\
         \x20                       (DeltaNet / attention first half + second half) recipes for\n\
         \x20                       halfprobe to time. An order is `B:node` runs (body F or G up to\n\
         \x20                       and including the node, `*` = the rest) and `B=node` single\n\
         \x20                       nodes, `,`-separated; a leading `~` orders only the named nodes\n\
         \x20 GAUDIUM_HALFPROBE_NCH KV chunks the probe's attention recipes read (default 1; the\n\
         \x20                       prompt must have filled them)\n\
         \x20 GAUDIUM_HALFPROBE_CHUNK  1 = halfprobe also times the DeltaNet chunk GEMMs alone with\n\
         \x20                       f32 and with BF16 operands\n\
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
         Inkling 2 environment variables (debugging; none changes a served stream unless noted):\n\
         \x20 GAUDIUM_IK_AMAX_BF16  1: decode the 4-bit expert matrices with their per-expert scale terms\n\
         \x20                       rounded to BF16 first, as the publisher's reference loader ends up\n\
         \x20                       doing (for comparisons with it; changes the weights slightly)\n\
         \x20 GAUDIUM_IK_APPS       Load and run only the first N layer applications (kernel checks)\n\
         \x20 GAUDIUM_IK_STEP_PREFILL  1: prefill the prompt one row at a time with the decode step\n\
         \x20 GAUDIUM_IK_DUMP       Directory: after the prompt, every tensor of the one-row decode step and\n\
         \x20                       the head of every K / V state is written there as <name>.bin\n\
         \x20 GAUDIUM_IK_VERIFY_CHECK  N (2..64): after a greedy request its tokens are run again in passes\n\
         \x20                       of N rows and every row's logits are compared with the one-row step's\n\
         \x20 GAUDIUM_IKV_DUMP      Directory: the vision tower writes, per image k of the first image\n\
         \x20                       request, img<k>_emb / _pre / _l00 / _l11 / _l22 (sequence rows x 1024)\n\
         \x20                       and img<k>_feat (tokens x 2304) as raw BF16 (it syncs after each one)\n\
         \x20 GAUDIUM_IK_MTP_DUMP   Directory: after the first block of the drafter (--spec mtp), every\n\
         \x20                       drafter activation and the head of its context state as <name>.bin\n\
         \n\
         Generate options:\n\
         \x20 --model <dir>         Model directory (or GAUDIUM_MODEL); or a GGUF file (Inkling 2)\n\
         \x20 --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)\n\
         \x20 --prompt <text>       Input prompt text\n\
         \x20 --prompt-ids <csv>    Raw comma-separated prompt token IDs (skips tokenizer)\n\
         \x20 --prompt-file <path>  Read the prompt text from a file (long-context prompts)\n\
         \x20 --prompt-ids-file <p> Read comma/whitespace-separated prompt token IDs from a file\n\
         \x20                       (repeatable: each file is one request on the same loaded engine)\n\
         \x20 --also-prompt-ids-file <p>  Same as a further --prompt-ids-file\n\
         \x20 --also-prompt-file <p>  One more text request on the same loaded engine (repeatable;\n\
         \x20                       formatted like --prompt, after the token-ID requests)\n\
         \x20 --chat                Wrap --prompt in the model's chat template (Qwen3 ChatML / Gemma 4 turns;\n\
         \x20                       other families: the model's own template, reasoning off with --no-think)\n\
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
         \x20                       without), and report identity or the first divergence (greedy:\n\
         \x20                       with both tokens' logits at every position of the run's stream\n\
         \x20                       where K=0 would have chosen otherwise, from a teacher-forced replay)\n\
         \x20 --profile-steps <[p]A-B[@R][:file]>  Device trace of decode launches A..B (0-based; with\n\
         \x20                       a leading p: of prefill launches A..B, counted over the tile\n\
         \x20                       loop's embed and layer launches) of\n\
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
         \x20 --image <file>        Image input: JPEG, PNG, GIF (first frame), WebP, BMP or binary\n\
         \x20                       PPM, decoded as the server decodes a request's image;\n\
         \x20                       repeatable, in prompt order. An image belongs to the first\n\
         \x20                       request, or to the further request (--also-prompt-file,\n\
         \x20                       --also-prompt-ids-file, a repeated --prompt-ids-file) it follows\n\
         \x20 --image-raw <WxH:f>   Image input, raw RGB8 file with explicit size; repeatable\n\
         \x20 --vision-max-patches <N>  Largest image the vision tower is planned for, in the\n\
         \x20                       model's patches (default: the largest --image; Qwen3.5 / 3.8\n\
         \x20                       resize larger images down to fit, Inkling 2 refuses them)\n\
         \x20 --vision-gelu-bf16    Evaluate the vision tower's GELU in BF16 (default: F32);\n\
         \x20                       slightly faster, slightly less accurate image embeddings\n\
         \x20 --dump-vision-emb <p> Write the image embeddings (tokens x hidden, raw BF16) to a file\n\
         \x20 --dump-prompt-ids     Print the final prompt token IDs to stderr and continue\n\
         \x20 --repeat <N>          Run the same request N times in one process (warm timings)\n\
         \x20 --after-ids-file <p>  Then run a text-only request with these prompt token IDs, once\n\
         \x20                       per sampler setting (checks that an image request leaves no\n\
         \x20                       state behind: its AFTER TOKENS lines must equal the tokens of\n\
         \x20                       the same request on an engine that never saw an image)\n\
         \x20 --after-prompt-file <p>  The same with a text prompt, formatted like --prompt\n\
         \x20 --spec <method>       Speculative decoding: dspine or dflash2 (on-card block drafters),\n\
         \x20                       mtp (the target's own MTP head, hybrid Qwen3.5/3.8, dense or\n\
         \x20                       mixture-of-experts, either --moe-form), assistant (a Gemma 4\n\
         \x20                       size's own drafter checkpoint) or oracle (test: drafts from\n\
         \x20                       --oracle-file); absent = non-speculative K=0.\n\
         \x20                       A Gemma 4 model reads the drafter kind from the checkpoint:\n\
         \x20                       --spec dflash2 --draft-model <dir> runs a DFlashDraftModel\n\
         \x20                       block drafter (one --draft-k width, default 7).\n\
         \x20                       An Inkling 2 model: mtp drafts with the checkpoint's block\n\
         \x20                       drafter (mtp.safetensors beside the model; up to 7 drafts per\n\
         \x20                       block, pass --draft-k 7 for all of them). A GGUF model reads\n\
         \x20                       it from a drafter GGUF beside the model file or in a dspark/\n\
         \x20                       directory there (BF16 before Q8_0 when several match).\n\
         \x20                       Chain modes are token-identical to K=0 (sampled: same seed)\n\
         \x20 --draft-model <dir>   Drafter checkpoint directory (with --spec dspine / dflash2;\n\
         \x20                       assistant: default <model_dir>-assistant)\n\
         \x20 --draft-k <N[,N..]>   Drafts verified per launch, 1..31 (default: 15; dflash2: 1..7,\n\
         \x20                       default 7; mtp: at most 15, default 4; assistant: default 4);\n\
         \x20                       several values let the engine pick a width per step from\n\
         \x20                       acceptance (mtp: every step then drafts the widest listed block;\n\
         \x20                       --parallel lanes: each lane at the narrowest or the widest\n\
         \x20                       listed in the replay form, else at the narrowest)\n\
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
         \x20                       the full vocabulary; or GAUDIUM_DRAFT_VOCAB). Default: the full\n\
         \x20                       vocabulary for dspine (as an FP8 copy of the readout when the\n\
         \x20                       drafter's FP8 table has its site) and for mtp, 98304 for\n\
         \x20                       dflash2. A token outside the subset is never drafted (a large\n\
         \x20                       share of the tokens outside English text and code); the subset\n\
         \x20                       reads fewer bytes per block. Drafts are guesses: the output is\n\
         \x20                       unchanged, only acceptance and the step time can move\n\
         \x20 --draft-fp8 <0|1>     DSpine drafter layers as FP8 weight GEMMs (default, or\n\
         \x20                       GAUDIUM_DRAFT_FP8: on when the drafter's activation-scale table\n\
         \x20                       exists). The verify stays as it is: the output is unchanged\n\
         \x20 --draft-fp8-act <f>   The drafter's activation-scale table (or GAUDIUM_DRAFT_FP8_ACT;\n\
         \x20                       default <draft dir>-gaudium-pc/act_scales_ds.txt)\n\
         \x20 --draft-fp8-calibrate <f>  Calibration run for that table: the drafter runs BF16, one\n\
         \x20                       launch at a time, its GEMM input sites are read back after every\n\
         \x20                       launch and the table is written to <f> after each request\n\
         \x20 --draft-ring <0|1>    DSpine context caches as rings of the checkpoint's sliding window\n\
         \x20                       instead of one row per context position: the drafter's share of\n\
         \x20                       a step then does not grow with --max-ctx (default, or\n\
         \x20                       GAUDIUM_DRAFT_RING: on when the ring kernels and the step glue\n\
         \x20                       library are in the kernels directory). Drafts only: the output\n\
         \x20                       is unchanged\n\
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
         \x20                       WITHOUT opening the card (runs beside a process that owns it).\n\
         \x20                       Gemma 4: also the decode steps of every context bucket, the\n\
         \x20                       full-tile prefill of every chunk count, and what the given\n\
         \x20                       prompts and images launch (their last tiles, image tiles and\n\
         \x20                       image sizes); it plans the widest prefill tile\n\
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
         \x20 --kv-cache <d>        K/V cache: bf16 (default) or fp8 (FP8 q / K / V / P in the trunk's\n\
         \x20                       attention, half the K/V bytes; a precision of its own: its K=0\n\
         \x20                       stream is the reference of FP8-K/V speculation. Dense Qwen3.5 / 3.8\n\
         \x20                       hybrids, K=0 or DSpine; needs the table from --kv-calibrate;\n\
         \x20                       its prefill runs 256-row tiles)\n\
         \x20 --kv-scales <file>    K/V calibration the FP8 scale tables are made from (default:\n\
         \x20                       <model_dir>-gaudium-pc/kv_scales.txt)\n\
         \x20 --moe-form <f>        MoE expert execution: dense (default: one masked wide MLP per layer)\n\
         \x20                       or routed (stacked BF16 experts, only the routed ones are read;\n\
         \x20                       faster decode, more device memory, its own token stream: the\n\
         \x20                       two forms round differently)\n\
         \x20 --request-reset <m>   Device state zeroed before each request: auto (default: the\n\
         \x20                       DeltaNet start state; KV rows past the cursor are masked),\n\
         \x20                       dirty (also the KV rows earlier requests wrote) or full\n\
         \x20                       (or GAUDIUM_REQUEST_RESET; same output in every mode)\n\
         \x20 --prefix-cache <N>    Prefix reuse between the requests of one run (default: 0 = off,\n\
         \x20                       every request prefills its whole prompt): keep the recurrent\n\
         \x20                       state at N positions of the previous prompt (see --prefix-cache\n\
         \x20                       under Server options)\n\
         \x20 --prefill <k=v,..>    Prefill levers of the Qwen engines (default: all on): tile=N rows\n\
         \x20                       per tile (multiple of 64 up to 512; 0 = the widest that fits),\n\
         \x20                       mme=0|1 DeltaNet recurrence as chunked MME GEMMs, rows=0|1|2 its\n\
         \x20                       prep / norm kernels, fastexp=0|1 fast-exp tile softmax, groups=N\n\
         \x20                       attention chunk groups, s16=0|1|2 half-width attention core (BF16\n\
         \x20                       scores; 2: + FP8 P; default 2 with dense FP8 weights, else 0: it\n\
         \x20                       changes the prefill numerics), wave=0|1 full tiles two in flight\n\
         \x20                       (the halves of two tiles' layers share a launch so the MME and\n\
         \x20                       the TPCs overlap; same nodes, same output; needs the mme and\n\
         \x20                       fastexp forms), plain = every lever off, stats /\n\
         \x20                       profile = print the tile loop's host split / per-recipe times,\n\
         \x20                       halfprobe=N = measurement: after the last request, time whole\n\
         \x20                       tile layers, their halves and two halves in one recipe over N\n\
         \x20                       launches each (with --build-cache: compile them without the\n\
         \x20                       card and the recipe cache, for the compiler's own dumps)\n\
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
         \x20 --verbose             Print HBM, recipe compile, and per-site diagnostics\n\
         \n\
         K/V calibration options (prefill only, BF16 caches; writes `layer <l> k <amax..> v <amax..>`\n\
         lines, the largest |K| and |V| per kv head; FP8 K/V takes its V multipliers from them):\n\
         \x20 --model <dir>         Model directory (or GAUDIUM_MODEL)\n\
         \x20 --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)\n\
         \x20 --prompt-ids-file <p> Calibration prompt as token IDs (repeatable)\n\
         \x20 --prompt-file <path>  Calibration prompt as text (repeatable)\n\
         \x20 --out <file>          Table to write (default: <model_dir>-gaudium-pc/kv_scales.txt)\n\
         \x20 --max-ctx <N>         Context capacity, multiple of 4096 (default and least: what the\n\
         \x20                       longest prompt needs, with the engine's reserve)\n\
         \x20 --precision <p>       Weight precision of the measuring load, as the served one: auto\n\
         \x20                       (default), bf16 or fp8\n\
         \x20 --fp8-act <file>      FP8 activation-scale table of an fp8 load\n\
         \x20 --fp8-cache <dir>     Cache directory for the quantised FP8 matrices\n\
         \x20 --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)\n\
         \x20 --no-cache            Disable on-disk compiled recipe cache\n\
         \x20 --verbose             Print HBM, recipe compile, and per-layer diagnostics\n\
         \n\
         Cache trim options (--trim-cache; an entry is a <name>-<hash>.recipe file and its .sha256\n\
         sidecar, removed together; every other file in the directory is left alone and not counted;\n\
         each cache hit sets the sidecar's modification time, so recency needs no access times):\n\
         \x20 --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)\n\
         \x20 --cache-max-gb <N>    Budget: GiB of cache entries left afterwards (decimals allowed;\n\
         \x20                       0 = every entry outside the keep window)\n\
         \x20 --cache-keep-days <D> Entries used within D days stay even over the budget (default: 1;\n\
         \x20                       an entry dated ahead of the clock counts as just used)\n\
         \x20 --dry-run             Print what would be freed, remove nothing\n\
         \n\
         Certificate options (--make-cert <dir>: writes ca.crt and ca.key, the authority, when the\n\
         directory has neither of them, and server.crt (the chain) and server.key, signed by it; the\n\
         keys get mode 0600 and no file is replaced without --cert-overwrite; prints the SHA-256\n\
         fingerprints and how to trust ca.crt on each kind of device; docs/mcp-proxy-https.md):\n\
         \x20 --cert-names <list>   Names and addresses the Web UI is opened by, comma separated\n\
         \x20                       (repeatable): host names (gaudium.lan) and IPv4 or IPv6 addresses.\n\
         \x20                       localhost, 127.0.0.1 and ::1 are always added. Default: this\n\
         \x20                       machine's host name and the address of its default route. A new\n\
         \x20                       authority may sign only for these names (and names under them),\n\
         \x20                       loopback and local-network addresses\n\
         \x20 --cert-days <N>       Days the server certificate is valid (default and most: 825, the\n\
         \x20                       longest Apple's systems accept; never past the end of the\n\
         \x20                       authority, which is valid for 10 years)\n\
         \x20 --cert-overwrite      Replace the server certificate and key that are there (a new name\n\
         \x20                       or address, or a renewal). The authority is never replaced: for a\n\
         \x20                       name it does not cover, move ca.crt and ca.key away to make a new\n\
         \x20                       one, and trust it again on each device\n\
         \n\
         Vision suite options (a row = one model and one method of the suite file; per row the\n\
         text request, every image case and the text request again, greedy and seeded sampled,\n\
         each checked for token identity to its K=0 reference, a right answer, and the text\n\
         request after the images giving the fresh engine's tokens; exit status 0 = all passed):\n\
         \x20 --models <dir>        Directory of the suite's relative model names and of {{models}}\n\
         \x20 --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)\n\
         \x20 --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)\n\
         \x20 --out <dir>           Prompts, each row's command line and logs, report.json\n\
         \x20                       (default: a new directory under the system temp directory)\n\
         \x20 --only <row>          Run only this model, or model/method (repeatable)\n\
         \x20 --build-cache         Compile every selected row's recipes into the recipe cache\n\
         \x20                       without the card (--build-cache rows) instead of running them\n\
         \x20 -- <options>          Further --generate options for every row"
    );
}

/// `gaudium --vision-suite <suite.json> [options] [-- <generate options>]`.
fn run_vision_suite(args: &[String]) -> Result<bool, String> {
    use gaudium::vision::suite::{run, SuiteOptions};
    let mut o = SuiteOptions::default();
    let mut out: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < args.len() {
        let next = |i: &mut usize, name: &str| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        match args[i].as_str() {
            "--vision-suite" => o.suite = PathBuf::from(next(&mut i, "--vision-suite")?),
            "--models" => o.models_dir = Some(PathBuf::from(next(&mut i, "--models")?)),
            "--kernels" => o.kernels = Some(PathBuf::from(next(&mut i, "--kernels")?)),
            "--cache-dir" => o.cache_dir = Some(PathBuf::from(next(&mut i, "--cache-dir")?)),
            "--out" => out = Some(PathBuf::from(next(&mut i, "--out")?)),
            "--only" => o.only.push(next(&mut i, "--only")?),
            "--build-cache" => o.build_cache = true,
            "--" => {
                o.extra = args[i + 1..].to_vec();
                break;
            }
            other => return Err(format!("unknown --vision-suite option '{other}'")),
        }
        i += 1;
    }
    if o.suite.as_os_str().is_empty() {
        return Err("--vision-suite <suite.json>".into());
    }
    o.out_dir = out.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("gaudium-vision-suite-{}", process::id()))
    });
    run(&o)
}

fn inspect_model(dir: &Path) -> Result<(), String> {
    if gaudium::gguf::looks_like_gguf(dir) {
        println!("{}", gaudium::gguf::inspect::inspect(dir).map_err(|e| e.to_string())?);
        return Ok(());
    }
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

/// The families whose `--generate --chat` prompt has a built-in form (the others render the
/// model's own chat template).
fn has_builtin_chat_form(family: gaudium::model_spec::ModelFamily) -> bool {
    use gaudium::model_spec::ModelFamily as F;
    match family {
        F::Qwen3 | F::Qwen35Dense | F::Qwen35Moe | F::Gemma4 | F::Gemma4Assistant => true,
        F::Inkling2 => false,
    }
}

fn run_tokenize(dir: &Path, text: &str) -> Result<(), String> {
    let tok = gaudium::model_text::ModelText::open(dir)?.tokenizer()?;
    let ids = tok.encode(text, true);
    let decoded = tok.decode(&ids, false);
    println!("vocab={} tokens={}", tok.vocab_size(), ids.len());
    println!("ids={ids:?}");
    println!("decoded={decoded:?}");
    Ok(())
}

/// The request of the command line an `--image` belongs to: the first one, or the further
/// token-ID / text request (`--also-prompt-ids-file`, a repeated `--prompt-ids-file`,
/// `--also-prompt-file`) it follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliRequest {
    First,
    Ids(usize),
    Text(usize),
}

impl CliRequest {
    /// Index in the run's request list: request 0, the token-ID requests, then the text ones.
    fn index(self, n_ids: usize) -> usize {
        match self {
            CliRequest::First => 0,
            CliRequest::Ids(k) => 1 + k,
            CliRequest::Text(k) => 1 + n_ids + k,
        }
    }
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
    let mut kv_dtype = gaudium::verified::KvDtype::KvBf16;
    let mut kv_scales: Option<PathBuf> = None;
    let mut moe_routed = false;
    let mut prefill_opts: Option<String> = None;
    let mut params = Qwen3GenParams::default();
    // (request, raw RGB8?, path / WxH:path): an image belongs to request 0 unless it follows a
    // flag that adds a further request
    let mut image_args: Vec<(CliRequest, bool, String)> = Vec::new();
    let mut image_req = CliRequest::First;
    let mut vision_max_patches = 0u64;
    let mut dump_prompt_ids = false;
    let mut vision_gelu_bf16 = false;
    let mut dump_vision_emb: Option<String> = None;
    let mut repeat = 1usize;
    let mut after_ids_file: Option<String> = None;
    let mut after_prompt: Option<String> = None;
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
    let mut prefix_slots = 0u64;
    let mut profile_windows: Vec<gaudium::synapse::profiler::ProfileWindow> = Vec::new();
    let mut draft_vocab_set = false;
    let mut draft_fp8: Option<bool> = None;
    let mut draft_fp8_act: Option<PathBuf> = None;
    let mut draft_fp8_calibrate: Option<PathBuf> = None;
    let mut gdn_verify: Option<gaudium::engine::step_opts::GdnVerify> = None;
    let mut k0_pipeline: Option<bool> = None;
    let mut gdn_fast_rows: Option<bool> = None;
    let mut draft_ring: Option<bool> = None;
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
                    image_req = CliRequest::Ids(extra_prompt_ids_raw.len() - 1);
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
                image_req = CliRequest::Text(extra_prompt_texts.len() - 1);
            }
            "--also-prompt-ids-file" => {
                let path = next_val(&mut i, "--also-prompt-ids-file")?;
                extra_prompt_ids_raw.push(
                    std::fs::read_to_string(&path)
                        .map_err(|e| format!("read --also-prompt-ids-file {path}: {e}"))?,
                );
                image_req = CliRequest::Ids(extra_prompt_ids_raw.len() - 1);
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
            "--image" => image_args.push((image_req, false, next_val(&mut i, "--image")?)),
            "--image-raw" => image_args.push((image_req, true, next_val(&mut i, "--image-raw")?)),
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
            "--after-prompt-file" => {
                let path = next_val(&mut i, "--after-prompt-file")?;
                after_prompt = Some(
                    std::fs::read_to_string(&path)
                        .map_err(|e| format!("read --after-prompt-file {path}: {e}"))?,
                );
            }
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
            "--draft-ring" => {
                draft_ring = Some(parse_switch(
                    &next_val(&mut i, "--draft-ring")?,
                    "--draft-ring",
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
            "--kv-cache" => {
                let s = next_val(&mut i, "--kv-cache")?;
                kv_dtype = gaudium::core::kv_cache::parse_kv_dtype(&s)
                    .ok_or_else(|| format!("invalid --kv-cache '{s}' (bf16, fp8)"))?;
            }
            "--kv-scales" => kv_scales = Some(PathBuf::from(next_val(&mut i, "--kv-scales")?)),
            "--moe-form" => {
                moe_routed = match next_val(&mut i, "--moe-form")?.as_str() {
                    "dense" => false,
                    "routed" => true,
                    other => return Err(format!("invalid --moe-form '{other}' (dense, routed)")),
                };
            }
            "--request-reset" => request_reset = Some(next_val(&mut i, "--request-reset")?),
            "--prefix-cache" => {
                let s = next_val(&mut i, "--prefix-cache")?;
                prefix_slots = s
                    .parse()
                    .ok()
                    .filter(|&n| gaudium::verified::ln_prefix_arg_ok(n, MAX_PREFIX_SLOTS))
                    .ok_or_else(|| {
                        format!("invalid --prefix-cache '{s}' (0..={MAX_PREFIX_SLOTS} positions)")
                    })?;
            }
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

    // the model's text resources: a checkpoint directory or a GGUF file
    let text = gaudium::model_text::ModelText::open(&model_dir)?;
    let tok = text.tokenizer()?;
    // `--chat`: the family's built-in chat form, or the model's own template (`mt_cli_chat`)
    let builtin_chat = text.spec().map_or(true, |s| has_builtin_chat_form(s.family));
    let one_turn = if gaudium::verified::mt_cli_wants_template(chat, builtin_chat) {
        gaudium::server::OneTurnTemplate::load(&text)?
    } else {
        None
    };
    let chat_form = gaudium::verified::mt_cli_chat(
        tok.token_id("<|turn>").is_some(),
        builtin_chat,
        one_turn.is_some(),
    );
    let gemma_closes = text
        .chat_template_source()
        .is_some_and(|t| gemma4_template_closes_thought(&t));
    drop(text);
    if let Some(im_end) = tok.token_id("<|im_end|>") {
        params.extra_eos_ids.push(im_end);
    }
    // Image input: prepare on the host first, so the tower is planned for these image sizes.
    // `request_images[r]`: the images of request r (request 0, the further token-ID requests in
    // order, then the further text requests in order: the order of `all_prompts` below).
    if after_ids_file.is_some() && after_prompt.is_some() {
        return Err("give --after-ids-file or --after-prompt-file, not both".into());
    }
    let n_ids = extra_prompt_ids_raw.len();
    let n_requests = 1 + n_ids + extra_prompt_texts.len();
    let mut request_images: Vec<Option<gaudium::vision::cli::CliImages>> = Vec::new();
    for r in 0..n_requests {
        let args: Vec<(bool, String)> = image_args
            .iter()
            .filter(|(q, _, _)| q.index(n_ids) == r)
            .map(|(_, raw, a)| (*raw, a.clone()))
            .collect();
        request_images.push(gaudium::vision::cli::prepare_cli_images(
            &model_dir,
            &args,
            vision_max_patches,
        )?);
    }
    let vision_in = request_images[0].as_ref();
    let prompt = match (vision_in, prompt) {
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
    // a text prompt as token ids, through the chat form when `--chat` is given
    let encode_text = |raw_prompt: String| -> Result<Vec<i32>, String> {
        use gaudium::verified::MtChatForm;
        let sys = system_prompt.as_deref();
        let formatted = match (chat, chat_form) {
            (false, _) => raw_prompt,
            (true, MtChatForm::MtChatGemma) => {
                format_gemma4_chat(&raw_prompt, sys, !no_think, gemma_closes)
            }
            (true, MtChatForm::MtChatChatml) => format_qwen3_chat(&raw_prompt, sys, !no_think),
            (true, MtChatForm::MtChatTemplate) => one_turn
                .as_ref()
                .ok_or("--chat: the model's chat template was not loaded")?
                .render(&raw_prompt, sys, !no_think)?,
            (true, MtChatForm::MtChatNone) => {
                return Err("--chat: the model has no chat template".into())
            }
        };
        Ok(tok.encode(&formatted, true))
    };
    let prompt_ids: Vec<i32> = if let Some(csv) = prompt_ids_raw {
        parse_ids_csv(&csv)?
    } else {
        encode_text(prompt.unwrap_or_else(|| "The capital of France is".to_string()))?
    };
    // `--also-prompt-file`: more text requests for the same loaded engine
    let extra_text_ids: Vec<Vec<i32>> = extra_prompt_texts
        .into_iter()
        .enumerate()
        .map(|(k, text)| match &request_images[1 + n_ids + k] {
            Some(v) => encode_text(v.with_placeholders(&text)),
            None => encode_text(text),
        })
        .collect::<Result<_, _>>()?;
    // `--after-prompt-file`: a text request after all others, formatted like --prompt
    let after_ids: Option<Vec<i32>> = match (&after_prompt, &after_ids_file) {
        (Some(text), _) => Some(encode_text(text.clone())?),
        (None, Some(path)) => Some(gaudium::vision::cli::parse_ids(
            &std::fs::read_to_string(path)
                .map_err(|e| format!("read --after-ids-file {path}: {e}"))?,
        )?),
        (None, None) => None,
    };
    let prompt_ids = match vision_in {
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
    cfg.kv_dtype = kv_dtype;
    cfg.kv_scales = kv_scales;
    cfg.moe_routed = moe_routed;
    cfg.prefill_warm_rows = prefill_warm;
    cfg.request_reset = request_reset;
    cfg.prefix_slots = prefix_slots;
    if let Some(v) = draft_fp8 {
        cfg.step.draft_fp8 = Some(v);
    }
    if draft_fp8_act.is_some() {
        cfg.step.draft_fp8_act = draft_fp8_act;
    }
    cfg.step.draft_fp8_calibrate = draft_fp8_calibrate;
    if let Some(v) = gdn_verify {
        cfg.step.gdn_verify = v;
        cfg.step.gdn_verify_asked = true;
    }
    if let Some(v) = k0_pipeline {
        cfg.step.k0_pipeline = v;
    }
    if gdn_fast_rows.is_some() {
        cfg.step.gdn_fast_rows = gdn_fast_rows;
    }
    if draft_ring.is_some() {
        cfg.step.draft_ring = draft_ring;
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
    let with_images: Vec<&gaudium::vision::cli::CliImages> = request_images.iter().flatten().collect();
    if !with_images.is_empty() {
        gaudium::vision::cli::configure_requests(&with_images, &mut cfg, vision_max_patches);
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

    // The requests, built on the host before anything is loaded (a malformed one fails here,
    // also under --build-cache). Every `--prompt-ids-file` after the first is one more request served by the same loaded engine.
    let mut all_prompts: Vec<Vec<i32>> = vec![prompt_ids];
    for csv in &extra_prompt_ids_raw {
        all_prompts.push(parse_ids_csv(csv)?);
    }
    all_prompts.extend(extra_text_ids);
    // a further request's image-pad placeholders become its images' token runs
    gaudium::vision::cli::expand_requests(&mut all_prompts[1..], &request_images[1..])?;

    // `--build-cache`: compile this configuration's recipes without the card (SynapseAI is
    // initialized, the device is never acquired), then exit. With `--build-cache-chunks` the
    // speculative step recipes of exactly those KV chunk counts are compiled (the routed
    // mixture-of-experts steps take tens of seconds each); without it, every recipe a load of
    // this configuration uses.
    if build_cache {
        if !build_cache_chunks.is_empty() {
            return gaudium::engine::qwen3::build_cache(&cfg, &build_cache_chunks);
        }
        // the requests (in `all_prompts` order, then the --after request), for an engine whose
        // recipes depend on them
        let none: &[gaudium::vision::PreparedImage] = &[];
        let requests: Vec<gaudium::engine::BuildRequest> = all_prompts
            .iter()
            .enumerate()
            .map(|(r, p)| gaudium::engine::BuildRequest {
                prompt: p,
                images: request_images[r].as_ref().map_or(none, |v| &v.images),
            })
            .chain(after_ids.iter().map(|p| gaudium::engine::BuildRequest {
                prompt: p,
                images: none,
            }))
            .collect();
        return gaudium::engine::build_cache(&cfg, &requests);
    }
    if !build_cache_chunks.is_empty() {
        return Err("--build-cache-chunks needs --build-cache".into());
    }

    // Stop requests (SIGINT/SIGTERM) end generation between synced launches, never mid-launch.
    gaudium::signals::install();
    let mut engine: Box<dyn GenEngine> = load_engine(&cfg)?;
    if let Some(v) = vision_in {
        engine.set_request_images(v.images.clone())?;
    }

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
            if let Some(v) = &request_images[prompt_idx] {
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
        // a request's images: every run of it gets them again
        if let Some(v) = &request_images[prompt_idx] {
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
                    // run's whole stream, which records both tokens' logits at every position
                    // where K=0 would have chosen otherwise (the first fork and any later one).
                    if params.temperature <= 0.0 && i > 0 {
                        let mut pt = params.clone();
                        pt.non_speculative = true;
                        pt.k0_plain = true;
                        pt.force_draft_k = None;
                        pt.max_new = got.len() as u64;
                        pt.stop_on_eos = false;
                        pt.force_tokens = Some(got.clone());
                        if let Some(v) = &request_images[prompt_idx] {
                            engine.set_request_images(v.images.clone())?;
                        }
                        let rt = engine.generate_cancellable(
                            prompt_ids,
                            &pt,
                            Some(gaudium::signals::term_flag()),
                            &mut |_| {},
                        )?;
                        for f in &rt.forks {
                            eprintln!(
                                "[gaudium] FORK TRACE at generated index {}: K=0 token {} logit {:.4} vs the run's token {} logit {:.4} (margin {:.4} BF16 logits)",
                                f.pos,
                                f.engine_token,
                                f.engine_logit,
                                f.forced_token,
                                f.forced_logit,
                                f.engine_logit - f.forced_logit,
                            );
                        }
                        if !rt.forks.iter().any(|f| f.pos == i) {
                            eprintln!(
                                "[gaudium] FORK TRACE at generated index {i}: the teacher-forced K=0 pass agreed with the run's token"
                            );
                        }
                        if !gaudium::signals::term_requested() {
                            eprintln!(
                                "[gaudium] FORK TRACE done: the run's {} tokens replayed on the K=0 loop, {} disagreement(s)",
                                rt.generated_tokens.len(),
                                rt.forks.len()
                            );
                        }
                    }
                }
            }
        }
        // The vision report of each request with images (once, under its first sampler setting).
        if set_idx == 0 && request_images[prompt_idx].is_some() {
            if let Some(v) = engine.last_vision_report() {
                eprintln!(
                    "{}",
                    gaudium::vision::cli::format_vision_report(&v, &report)
                );
            }
            if let Some(path) = dump_vision_emb.as_ref().filter(|_| prompt_idx == 0) {
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
    gaudium::vision::cli::run_extra_requests(
        engine.as_mut(),
        vision_in.map(|v| v.images.as_slice()),
        &all_prompts[0],
        &settings,
        repeat,
        after_ids.as_deref(),
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
        let tok = gaudium::model_text::ModelText::open(&model_dir)?.tokenizer()?;
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

/// K/V calibration pass: a cold prefill of every given prompt on BF16 caches, after each of which
/// the trunk's caches are read back for their per-head maxima (`core/kv_cache.rs`); writes the
/// table an `--kv-cache fp8` load builds its scale tables from.
fn run_kv_calibrate(args: &[String]) -> Result<(), String> {
    let mut model_dir: Option<PathBuf> = env::var_os("GAUDIUM_MODEL").map(PathBuf::from);
    let mut kernels_dir: Option<PathBuf> = env::var_os("GAUDIUM_KERNELS").map(PathBuf::from);
    let mut id_files: Vec<String> = Vec::new();
    let mut text_files: Vec<String> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut max_ctx: Option<u64> = None;
    let mut precision = WeightPrecision::Auto;
    let mut fp8_act: Option<PathBuf> = None;
    let mut fp8_cache: Option<PathBuf> = None;
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
            "--kv-calibrate" => {}
            "--model" => model_dir = Some(PathBuf::from(next_val(&mut i, "--model")?)),
            "--kernels" => kernels_dir = Some(PathBuf::from(next_val(&mut i, "--kernels")?)),
            "--prompt-ids-file" => id_files.push(next_val(&mut i, "--prompt-ids-file")?),
            "--prompt-file" => text_files.push(next_val(&mut i, "--prompt-file")?),
            "--out" => out = Some(PathBuf::from(next_val(&mut i, "--out")?)),
            "--max-ctx" => {
                max_ctx = Some(
                    next_val(&mut i, "--max-ctx")?
                        .parse()
                        .map_err(|_| "invalid --max-ctx".to_string())?,
                );
            }
            "--precision" => {
                let s = next_val(&mut i, "--precision")?;
                precision = WeightPrecision::parse(&s)
                    .ok_or_else(|| format!("invalid --precision '{s}' (auto, bf16, fp8)"))?;
            }
            "--fp8-act" => fp8_act = Some(PathBuf::from(next_val(&mut i, "--fp8-act")?)),
            "--fp8-cache" => fp8_cache = Some(PathBuf::from(next_val(&mut i, "--fp8-cache")?)),
            "--cache-dir" => cache_dir = Some(PathBuf::from(next_val(&mut i, "--cache-dir")?)),
            "--no-cache" => no_cache = true,
            "--verbose" | "-v" => verbose = true,
            other => return Err(format!("unknown --kv-calibrate argument '{other}'")),
        }
        i += 1;
    }
    let model_dir =
        model_dir.ok_or_else(|| "missing --model <dir> (or GAUDIUM_MODEL env var)".to_string())?;
    let kernels_dir = kernels_dir
        .ok_or_else(|| "missing --kernels <dir> (or GAUDIUM_KERNELS env var)".to_string())?;
    let out = out.unwrap_or_else(|| gaudium::core::kv_cache::default_table_path(&model_dir));
    let prompts = calibration_prompts(&model_dir, &id_files, &text_files)?;
    if prompts.is_empty() {
        return Err("--kv-calibrate needs at least one --prompt-ids-file or --prompt-file".into());
    }
    let longest = prompts
        .iter()
        .map(|(_, ids)| ids.len() as u64)
        .max()
        .unwrap_or(1);

    let mut cfg = Qwen3Config::new(model_dir.clone(), kernels_dir);
    // room for the longest prompt, its one generated token and the engine's reserve (the
    // engine rounds it up to whole chunks)
    let need = gaudium::core::budget::ctx_for_prompt(longest);
    cfg.max_ctx = max_ctx.unwrap_or(need).max(need);
    cfg.cache_dir = cache_dir;
    cfg.no_cache = no_cache;
    cfg.verbose = verbose;
    cfg.precision = precision;
    cfg.fp8_act = fp8_act;
    cfg.fp8_cache = fp8_cache;
    cfg.kv_calibrate = true;

    gaudium::signals::install();
    let mut engine: Box<dyn GenEngine> = load_engine(&cfg)?;
    let params = Qwen3GenParams {
        max_new: 1,
        stop_on_eos: false,
        cache_prompt: false,
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
        engine.kv_measure(ids.len() as u64)?;
        rows += ids.len();
        eprintln!(
            "[kv-calibrate] {name}: {} prompt tokens, prefill {:.1} ms",
            ids.len(),
            report.ttft_ms
        );
    }
    let mut table = engine
        .kv_calibration()
        .ok_or_else(|| "this model's engine has no K/V calibration pass".to_string())?;
    let model_name = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    table.notes.push(format!(
        "model {model_name}, {} prompts, {rows} rows, weights {:?}",
        prompts.len(),
        precision
    ));
    table.store(&out)?;
    if verbose {
        for (l, a) in &table.layers {
            eprintln!("[kv-calibrate] layer {l}: K {:?} V {:?}", a.k, a.v);
        }
    }
    println!(
        "KV CALIBRATION OK: {} layers from {} prompts ({} rows) -> {}",
        table.layers.len(),
        prompts.len(),
        rows,
        out.display()
    );
    Ok(())
}

/// The prompts of a calibration run: token-id files and text files (encoded with the model's
/// tokenizer), each with the path it came from.
fn calibration_prompts(
    model_dir: &Path,
    id_files: &[String],
    text_files: &[String],
) -> Result<Vec<(String, Vec<i32>)>, String> {
    let mut prompts: Vec<(String, Vec<i32>)> = Vec::new();
    for path in id_files {
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
        let tok = gaudium::model_text::ModelText::open(model_dir)?.tokenizer()?;
        for path in text_files {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("read --prompt-file {path}: {e}"))?;
            prompts.push((path.clone(), tok.encode(&text, true)));
        }
    }
    Ok(prompts)
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

/// A non-negative number, decimals allowed (GiB, days).
fn parse_amount(s: &str, flag: &str) -> Result<f64, String> {
    s.parse()
        .ok()
        .filter(|v: &f64| v.is_finite() && *v >= 0.0)
        .ok_or_else(|| format!("invalid {flag} '{s}' (a number >= 0)"))
}

/// `gaudium --trim-cache --cache-max-gb <N> [--cache-dir <dir>] [--cache-keep-days <D>] [--dry-run]`:
/// `Ok(false)` when a removal failed.
fn run_trim_cache(args: &[String]) -> Result<bool, String> {
    let mut dir: Option<PathBuf> = None;
    let mut max_gb: Option<f64> = None;
    let mut keep_days = recipe_trim::DEFAULT_KEEP_DAYS;
    let mut dry_run = false;
    let mut i = 0usize;
    while i < args.len() {
        let next_val = |idx: &mut usize, name: &str| -> Result<String, String> {
            *idx += 1;
            args.get(*idx)
                .cloned()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        match args[i].as_str() {
            "--trim-cache" => {}
            "--cache-dir" => dir = Some(PathBuf::from(next_val(&mut i, "--cache-dir")?)),
            "--cache-max-gb" => {
                max_gb = Some(parse_amount(&next_val(&mut i, "--cache-max-gb")?, "--cache-max-gb")?)
            }
            "--cache-keep-days" => {
                keep_days = parse_amount(&next_val(&mut i, "--cache-keep-days")?, "--cache-keep-days")?
            }
            "--dry-run" => dry_run = true,
            other => return Err(format!("unknown --trim-cache flag '{other}'")),
        }
        i += 1;
    }
    let dir = dir
        .or_else(|| RecipeCache::from_env().root().map(Path::to_path_buf))
        .ok_or("missing --cache-dir <dir> (or GAUDIUM_RECIPE_CACHE)")?;
    let max_gb = max_gb.ok_or("missing --cache-max-gb <N>")?;
    let opts = recipe_trim::TrimOptions::from_gib_days(max_gb, keep_days, dry_run)?;
    let rep = recipe_trim::trim(&dir, &opts).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in &rep.errors {
        eprintln!("{e}");
    }
    println!("{}", rep.summary(&dir, &opts));
    Ok(rep.errors.is_empty())
}

/// `gaudium --make-cert <dir> [--cert-names <list>] [--cert-days <N>] [--cert-overwrite]`.
fn run_make_cert(args: &[String]) -> Result<(), String> {
    use gaudium::tls_cert::{
        detect_names, make_cert, parse_names, trust_steps, MakeCertOptions, CA_CERT, CA_DAYS,
        CA_KEY, DEFAULT_SERVER_DAYS, MAX_SERVER_DAYS, SERVER_CERT, SERVER_KEY,
    };
    let mut dir: Option<PathBuf> = None;
    let mut names = Vec::new();
    let mut names_given = false;
    let mut days = DEFAULT_SERVER_DAYS;
    let mut overwrite = false;
    let mut i = 0usize;
    while i < args.len() {
        let next_val = |idx: &mut usize, name: &str| -> Result<String, String> {
            *idx += 1;
            args.get(*idx)
                .cloned()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        match args[i].as_str() {
            "--make-cert" => dir = Some(PathBuf::from(next_val(&mut i, "--make-cert")?)),
            "--cert-names" => {
                names.extend(parse_names(&next_val(&mut i, "--cert-names")?)?);
                names_given = true;
            }
            "--cert-days" => {
                let v = next_val(&mut i, "--cert-days")?;
                days = match v.parse::<u64>() {
                    Ok(d) if (1..=MAX_SERVER_DAYS).contains(&d) => d,
                    _ => {
                        return Err(format!(
                            "--cert-days must be a whole number from 1 to {MAX_SERVER_DAYS} (got '{v}')"
                        ))
                    }
                };
            }
            "--cert-overwrite" => overwrite = true,
            other => return Err(format!("unknown --make-cert option '{other}'")),
        }
        i += 1;
    }
    let dir = dir.ok_or("--make-cert <dir>: the directory for the certificate files")?;
    let dir = std::path::absolute(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if !names_given {
        names = detect_names();
        let shown: Vec<String> = names.iter().map(ToString::to_string).collect();
        println!(
            "No --cert-names given: using this machine's name and address ({}). Give --cert-names to name others.\n",
            if shown.is_empty() { "none found".into() } else { shown.join(", ") }
        );
    }
    let r = make_cert(&MakeCertOptions {
        dir: dir.clone(),
        names: names.clone(),
        days,
        overwrite,
        ca_days: CA_DAYS,
    })?;
    let shown: Vec<String> = r.names.iter().map(ToString::to_string).collect();
    if r.made_ca {
        println!("Made a new certificate authority and a server certificate in {}:", dir.display());
    } else {
        println!(
            "Made a new server certificate in {}, signed by the authority already there{}:",
            dir.display(),
            if r.replaced_server { " (the old server files are replaced)" } else { "" }
        );
    }
    println!(
        "  {CA_CERT}      the authority: trust it once on each device that opens the Web UI (below)\n\
         \x20 {CA_KEY}      its key (mode 0600): it stays on this machine, never copy it\n\
         \x20 {SERVER_CERT}  the server's certificate, then the authority's (for --tls-cert)\n\
         \x20 {SERVER_KEY}  the server's key (mode 0600, for --tls-key)\n\
         \n\
         Server certificate\n\
         \x20 names        {}\n\
         \x20 valid until  {} ({} days)\n\
         \x20 SHA-256      {}\n\
         Certificate authority (a device shows this fingerprint when the certificate is installed)\n\
         \x20 name         {}\n\
         \x20 valid until  {}\n\
         \x20 SHA-256      {}\n",
        shown.join(", "),
        r.server.not_after,
        r.server_days,
        r.server.sha256,
        r.ca.subject,
        r.ca.not_after,
        r.ca.sha256,
    );
    for n in names.iter().filter(|n| n.looks_public()) {
        println!(
            "Note: {n} looks like a public name or address. A server that is reached from the internet under it\n\
             is better served by a certificate from a public authority (or a TLS front end): this authority is\n\
             trusted only by the devices it is installed on.\n"
        );
    }
    let host = names.first().map(ToString::to_string).unwrap_or_else(|| "localhost".into());
    let host = if host.contains(':') { format!("[{host}]") } else { host };
    println!(
        "Start the server with\n\
         \x20 gaudium --serve ... --tls-cert {} --tls-key {}\n\
         and open https://{host}:<port>/ (plain HTTP on the same port is sent there; see --tls-http).\n\
         \n\
         Trust the authority once on each device; every server certificate it signs is then trusted:\n\
         {}\n\
         \n\
         A new name or address, or the renewal before the date above: run the same command with\n\
         --cert-overwrite. The authority stays, so the devices need nothing new.",
        r.server.path.display(),
        r.server_key.display(),
        trust_steps(&r.ca.path),
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
    let mut a2ui =
        env::var("GAUDIUM_A2UI").is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "on"));
    let mut a2ui_prompt: Option<PathBuf> = env::var_os("GAUDIUM_A2UI_PROMPT")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let mut verbose = false;
    let mut debug_endpoints = false;
    let mut mcp_proxy = ProxyOptions::default();
    let mut tls_cert: Option<PathBuf> = None;
    let mut tls_key: Option<PathBuf> = None;
    let mut tls_http = gaudium::core::listener::PlainHttpMode::LsHttpRedirect;
    let mut hsts_max_age: Option<u64> = None;
    let mut per_model: Vec<(String, ModelRunOptions)> = Vec::new();
    let mut draft_k: Vec<u64> = Vec::new();
    let mut build_cache = false;
    let mut build_cache_only: Vec<String> = Vec::new();
    let mut vision = true;
    let mut vision_max_patches = gaudium::server::DEFAULT_VISION_MAX_PATCHES;
    let mut vision_max_tokens = gaudium::server::DEFAULT_VISION_MAX_TOKENS;
    let mut prefix_slots = gaudium::engine::qwen3::prefix_cache::DEFAULT_SLOTS;
    let mut cache_ram_bytes: Option<u64> = None;
    let mut response_store = gaudium::server::ResponseStoreLimits::default();
    let mut parallel = 1u64;
    let mut cache_max_gb: Option<f64> = None;
    let mut cache_keep_days: Option<f64> = None;
    let mut cache_trim_every: Option<u64> = None;
    let mut profile: Vec<gaudium::synapse::profiler::ProfileWindow> = Vec::new();
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
            "--precision" => {
                let raw = next_val(&mut i, "--precision")?;
                let (opts, p) = model_opt(&mut per_model, &raw, "--precision")?;
                opts.precision =
                    Some(p.to_str().and_then(WeightPrecision::parse).ok_or_else(|| {
                        format!("--precision expects <model id>=auto|bf16|fp8, got '{raw}'")
                    })?);
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
            "--effort-levels" => {
                let raw = next_val(&mut i, "--effort-levels")?;
                let (opts, list) = model_opt(&mut per_model, &raw, "--effort-levels")?;
                opts.effort_levels = Some(
                    gaudium::core::effort::EffortLevels::parse(list.to_str().unwrap_or(""))
                        .map_err(|e| format!("--effort-levels: {e}"))?,
                );
            }
            "--no-vision" => vision = false,
            "--vision-max-patches" => {
                let s = next_val(&mut i, "--vision-max-patches")?;
                vision_max_patches = s
                    .parse::<u64>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("invalid --vision-max-patches '{s}'"))?;
            }
            "--vision-max-tokens" => {
                let s = next_val(&mut i, "--vision-max-tokens")?;
                vision_max_tokens = s
                    .parse::<u64>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("invalid --vision-max-tokens '{s}'"))?;
            }
            "--model-vision" => {
                let raw = next_val(&mut i, "--model-vision")?;
                let (opts, v) = model_opt(&mut per_model, &raw, "--model-vision")?;
                opts.vision_patches = Some(match v.to_str() {
                    Some("off") => 0,
                    other => other
                        .and_then(|s| s.parse::<u64>().ok())
                        .filter(|&n| n > 0)
                        .ok_or_else(|| {
                            format!("--model-vision expects <model id>=<patches>|off, got '{raw}'")
                        })?,
                });
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
            "--kv-cache" => {
                let raw = next_val(&mut i, "--kv-cache")?;
                let (opts, d) = model_opt(&mut per_model, &raw, "--kv-cache")?;
                opts.kv_dtype = Some(
                    d.to_str()
                        .and_then(gaudium::core::kv_cache::parse_kv_dtype)
                        .ok_or_else(|| {
                            format!("--kv-cache expects <model id>=bf16|fp8, got '{raw}'")
                        })?,
                );
            }
            "--kv-scales" => {
                let raw = next_val(&mut i, "--kv-scales")?;
                let (opts, path) = model_opt(&mut per_model, &raw, "--kv-scales")?;
                opts.kv_scales = Some(path);
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
            "--cache-max-gb" => {
                cache_max_gb = Some(parse_amount(&next_val(&mut i, "--cache-max-gb")?, "--cache-max-gb")?)
            }
            "--cache-keep-days" => {
                cache_keep_days =
                    Some(parse_amount(&next_val(&mut i, "--cache-keep-days")?, "--cache-keep-days")?)
            }
            "--cache-trim-every" => {
                let s = next_val(&mut i, "--cache-trim-every")?;
                cache_trim_every = Some(
                    s.parse()
                        .ok()
                        .filter(|&m: &u64| m > 0)
                        .ok_or_else(|| format!("invalid --cache-trim-every '{s}' (minutes, > 0)"))?,
                );
            }
            "--profile-steps" => {
                profile.push(gaudium::synapse::profiler::ProfileWindow::parse(&next_val(
                    &mut i,
                    "--profile-steps",
                )?)?)
            }
            "--lazy" => lazy = true,
            "--build-cache" => build_cache = true,
            "--build-cache-only" => {
                build_cache = true;
                build_cache_only.push(next_val(&mut i, "--build-cache-only")?);
            }
            "--idle-park" => {
                let s = next_val(&mut i, "--idle-park")?;
                idle_park_secs = s
                    .parse()
                    .map_err(|_| format!("invalid --idle-park '{s}'"))?;
            }
            "--prefix-cache" => {
                let s = next_val(&mut i, "--prefix-cache")?;
                prefix_slots = s
                    .parse()
                    .ok()
                    .filter(|&n| gaudium::verified::ln_prefix_arg_ok(n, MAX_PREFIX_SLOTS))
                    .ok_or_else(|| {
                        format!("invalid --prefix-cache '{s}' (0..={MAX_PREFIX_SLOTS} positions)")
                    })?;
            }
            "--parallel" => {
                let s = next_val(&mut i, "--parallel")?;
                parallel = s
                    .parse()
                    .ok()
                    .filter(|&n| gaudium::verified::ln_lanes_arg_ok(n, gaudium::engine::MAX_LANES))
                    .ok_or_else(|| {
                        format!(
                            "invalid --parallel '{s}' (1 to {} lanes)",
                            gaudium::engine::MAX_LANES
                        )
                    })?;
            }
            "--cache-ram" => {
                // GiB, whole or decimal ("8", "0.5"); 0 turns the host tier off
                let s = next_val(&mut i, "--cache-ram")?;
                let gib: f64 = s
                    .parse()
                    .ok()
                    .filter(|g: &f64| g.is_finite() && *g >= 0.0)
                    .ok_or_else(|| format!("invalid --cache-ram '{s}' (GiB, 0 = off)"))?;
                cache_ram_bytes = Some((gib * (1u64 << 30) as f64) as u64);
            }
            "--response-store" => {
                let s = next_val(&mut i, "--response-store")?;
                response_store.entries = s
                    .parse()
                    .map_err(|_| format!("invalid --response-store '{s}' (responses, 0 = off)"))?;
            }
            "--response-store-mb" => {
                let s = next_val(&mut i, "--response-store-mb")?;
                response_store.bytes = s
                    .parse::<u64>()
                    .ok()
                    .and_then(|mb| mb.checked_mul(1 << 20))
                    .ok_or_else(|| format!("invalid --response-store-mb '{s}' (MiB)"))?;
            }
            "--queue-config" => {
                queue_config = Some(PathBuf::from(next_val(&mut i, "--queue-config")?))
            }
            "--api-key" => api_key = Some(next_val(&mut i, "--api-key")?),
            "--a2ui" => a2ui = true,
            "--a2ui-prompt" => {
                a2ui_prompt = Some(PathBuf::from(next_val(&mut i, "--a2ui-prompt")?))
            }
            "--tls-cert" => tls_cert = Some(PathBuf::from(next_val(&mut i, "--tls-cert")?)),
            "--tls-key" => tls_key = Some(PathBuf::from(next_val(&mut i, "--tls-key")?)),
            "--tls-http" => {
                use gaudium::core::listener::PlainHttpMode as M;
                let s = next_val(&mut i, "--tls-http")?;
                tls_http = match s.as_str() {
                    "redirect" => M::LsHttpRedirect,
                    "allow" => M::LsHttpAllow,
                    "strict" => M::LsHttpStrict,
                    _ => {
                        return Err(format!(
                            "invalid --tls-http '{s}' (expected redirect, allow or strict)"
                        ))
                    }
                };
            }
            "--hsts-max-age" => {
                let s = next_val(&mut i, "--hsts-max-age")?;
                hsts_max_age = Some(
                    s.parse()
                        .ok()
                        .filter(|&n: &u64| n <= 63_072_000)
                        .ok_or_else(|| {
                            format!("invalid --hsts-max-age '{s}' (seconds, 0 to 63072000)")
                        })?,
                );
            }
            "--mcp-proxy" => mcp_proxy.enabled = true,
            "--mcp-proxy-allow" => mcp_proxy
                .allow
                .push(next_val(&mut i, "--mcp-proxy-allow")?),
            "--mcp-proxy-allow-private" => mcp_proxy
                .allow_private
                .push(next_val(&mut i, "--mcp-proxy-allow-private")?),
            "--server-name" => mcp_proxy
                .server_names
                .push(next_val(&mut i, "--server-name")?),
            "--mcp-proxy-max-mb" => {
                let s = next_val(&mut i, "--mcp-proxy-max-mb")?;
                mcp_proxy.max_response_mb = s
                    .parse()
                    .map_err(|_| format!("invalid --mcp-proxy-max-mb '{s}'"))?;
            }
            "--mcp-proxy-timeout" => {
                let s = next_val(&mut i, "--mcp-proxy-timeout")?;
                mcp_proxy.idle_timeout_s = s
                    .parse()
                    .map_err(|_| format!("invalid --mcp-proxy-timeout '{s}'"))?;
            }
            "--mcp-proxy-ca" => {
                mcp_proxy.ca_file = Some(PathBuf::from(next_val(&mut i, "--mcp-proxy-ca")?))
            }
            "--verbose" => verbose = true,
            "--debug-endpoints" => debug_endpoints = true,
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

    let cache_trim = match cache_max_gb {
        None => {
            if cache_keep_days.is_some() || cache_trim_every.is_some() {
                return Err("--cache-keep-days and --cache-trim-every need --cache-max-gb".into());
            }
            None
        }
        Some(gb) => {
            let dir = if no_cache {
                None
            } else {
                cache_dir
                    .clone()
                    .or_else(|| RecipeCache::from_env().root().map(Path::to_path_buf))
            };
            let dir = dir.ok_or(
                "--cache-max-gb needs the recipe cache on (--cache-dir or GAUDIUM_RECIPE_CACHE, \
                 without --no-cache)",
            )?;
            let keep = cache_keep_days.unwrap_or(recipe_trim::DEFAULT_KEEP_DAYS);
            let every = cache_trim_every.unwrap_or(recipe_trim::DEFAULT_TRIM_EVERY_MIN);
            Some(recipe_trim::ServerTrim {
                dir,
                opts: recipe_trim::TrimOptions::from_gib_days(gb, keep, false)?,
                every: std::time::Duration::from_secs(every.saturating_mul(60)),
            })
        }
    };

    let hsts_max_age =
        gaudium::server::hsts_max_age_for(tls_cert.is_some(), port, hsts_max_age)?;

    if !profile.is_empty() {
        // as for --generate: the template before the device opens (the batched lane recipes
        // compile in the process, the others load from the recipe cache)
        gaudium::synapse::profiler::setup_env();
    }
    let cfg = ServerConfig {
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
        a2ui,
        a2ui_prompt,
        verbose,
        debug_endpoints,
        per_model,
        draft_k,
        vision,
        vision_max_patches,
        vision_max_tokens,
        prefix_slots,
        mcp_proxy,
        tls_cert,
        tls_key,
        tls_http,
        hsts_max_age,
        cache_ram_bytes,
        parallel,
        cache_trim,
        profile,
        response_store,
    };
    // `--serve ... --build-cache`: compile the served models' recipes without the card, then exit
    if build_cache {
        return gaudium::server::build_cache(&cfg, &build_cache_only);
    }
    run_server(cfg)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V") {
        println!("{}", version_line());
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
    if args.len() == 4 && args[1] == "--apply-template" {
        // the prompt a chat request renders to with the model's own template (host only)
        let run = || -> Result<String, String> {
            let body = if args[3] == "-" {
                let mut s = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut s)
                    .map_err(|e| format!("read stdin: {e}"))?;
                s
            } else {
                std::fs::read_to_string(&args[3]).map_err(|e| format!("read {}: {e}", args[3]))?
            };
            gaudium::server::apply_chat_template(Path::new(&args[2]), &body)
        };
        match run() {
            Ok(prompt) => print!("{prompt}"),
            Err(e) => {
                eprintln!("APPLY TEMPLATE FAIL: {e}");
                process::exit(1);
            }
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
    if (args.len() == 2 || args.len() == 3) && args[1] == "--probe-copy" {
        let gib = match args.get(2).map(|s| s.parse::<u64>()) {
            None => 2,
            Some(Ok(g)) if g > 0 => g,
            _ => {
                eprintln!("COPY PROBE FAIL: --probe-copy expects a whole number of GiB");
                process::exit(1);
            }
        };
        if let Err(e) = gaudium::synapse::xfer::probe(gib) {
            eprintln!("COPY PROBE FAIL: {e}");
            process::exit(1);
        }
        return;
    }
    if args.len() >= 3 && args[1] == "--vision-suite" {
        match run_vision_suite(&args[1..]) {
            Ok(true) => return,
            Ok(false) => process::exit(1),
            Err(e) => {
                eprintln!("VISION SUITE FAIL: {e}");
                process::exit(2);
            }
        }
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
    if args.iter().any(|a| a == "--kv-calibrate") {
        if let Err(e) = run_kv_calibrate(&args[1..]) {
            eprintln!("KV CALIBRATE FAIL: {e}");
            process::exit(1);
        }
        return;
    }
    if args.iter().any(|a| a == "--make-cert") {
        if let Err(e) = run_make_cert(&args[1..]) {
            eprintln!("MAKE CERT FAIL: {e}");
            process::exit(1);
        }
        return;
    }
    if args.iter().any(|a| a == "--trim-cache") {
        match run_trim_cache(&args[1..]) {
            Ok(true) => return,
            Ok(false) => process::exit(1),
            Err(e) => {
                eprintln!("TRIM CACHE FAIL: {e}");
                process::exit(2);
            }
        }
    }
    let serving = args.iter().any(|a| a == "--serve");
    if !serving
        && args
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
