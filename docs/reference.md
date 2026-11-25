# Gaudium reference: command line and environment

This page lists every command-line flag of the `gaudium` binary and every environment
variable the engine reads.

The first section is the text `gaudium --help` prints, copied here by a test so that it can be
read without building the engine. The same test (`gaudium/tests/reference_docs.rs`, part of
`cargo test`) reads the engine's sources and fails when:

- the code compares an argument with a flag that `--help` does not name, or `--help` names a
  flag the code no longer has;
- the engine reads an environment variable that `--help` does not name, or `--help` names a
  `GAUDIUM_*` / `GAUDI_*` variable nothing reads any more;
- a variable that only tests read is missing from the last section of this page;
- the copy of the help text below differs from what the binary prints.

To add a flag or a variable: write its line in `print_usage` (`gaudium/src/main.rs`), then run
`GAUDIUM_UPDATE_REFERENCE=1 cargo test --test reference_docs` to refresh the copy below, and
commit both.

## `gaudium --help`

<!-- help:begin (written by the reference_docs test; do not edit by hand) -->
```text
Gaudium — Intel Gaudi 2 Inference Engine (RustyBends v1)

Usage:
  gaudium                                   Run verified-core self-check
  gaudium --inspect-model <dir|file.gguf>   Validate config.json & Safetensors, or a GGUF
                                            model (a .gguf file, any shard of a split one,
                                            or a directory of them without config.json):
                                            metadata, tensors and weights per type
  gaudium --tokenize <model_dir> <text>     Encode & round-trip decode via tokenizer.json
                                            (or a GGUF model's tokenizer metadata)
  gaudium --hpu-smoke                       Run live Gaudi 2 BF16 add smoke test
  gaudium --probe-copy [GiB]                Measure copies between the card and host memory
                                            (both directions, through the pinned transfer ring
                                            the host-RAM prefix cache uses; opens the card;
                                            default: 2 GiB of device memory)
  gaudium --apply-template <model_dir> <request.json|->
                                            Print the prompt a chat request renders to with
                                            the model's own chat template (host only; a GGUF
                                            model: its tokenizer.chat_template)
  gaudium --generate [options]              Run live Gaudi 2 LLM generation
  gaudium --build-cache [generate options]  Compile every recipe of that configuration into
                                            the recipe cache and print its exact HBM plan,
                                            without opening the card (Qwen and Gemma 4
                                            engines)
  gaudium --serve [options]                 Start OpenAI HTTP/SSE + Gaudium Web UI server
  gaudium --fp8-calibrate [options]         Measure a model's FP8 activation scales (BF16 graphs)
  gaudium --kv-calibrate [options]          Measure the K/V maxima a model's FP8 K/V scale tables
                                            are made from (BF16 caches, read back per prompt)
  gaudium --vision-suite <suite.json> [options] [-- <generate options>]
                                            Vision regression suite: every model and
                                            speculative method of the suite file with its
                                            fixed images (gaudium/tests/vision_suite), one
                                            --generate process per row, one after another
                                            (this process never opens the card)
  gaudium --trim-cache --cache-max-gb <N> [options]
                                            Remove the least recently used recipes of the
                                            recipe cache until it fits N GiB (host only;
                                            safe beside a server using the directory)
  gaudium --make-cert <dir> [options]       Make the server's HTTPS certificate (host only): a
                                            local certificate authority, made once and trusted
                                            once on each device, and a server certificate
                                            signed by it, for --tls-cert and --tls-key
  gaudium --fp8-pack --model <dir> --fp8-cache <dir>
                                            Quantise a Gemma 4 BF16 checkpoint's matrices into an
                                            FP8 cache directory (host only, no card)
  gaudium --help | -h                       Print this text
  gaudium --version | -V                    Print the version (and the commit of a release build)

Serve options:
  --host <addr>         Bind address (or GAUDIUM_SERVE_HOST; default: 0.0.0.0)
  --port <N>            Bind port (or GAUDIUM_SERVE_PORT; default: 8090)
  --model <dir|id=dir>  Model directory (repeatable for multi-model hot-swap); a GGUF
                        file (or a directory of one GGUF model) serves its own tokenizer,
                        chat template, context limit and sampling defaults from its metadata
  --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)
  --ui-dir <dir>        Gaudium Web UI bundle directory (or GAUDIUM_UI_DIR)
  --max-ctx <N>         Context capacity, multiple of 4096 (default: 4096)
  --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)
  --no-cache            Disable on-disk compiled recipe cache
  --cache-max-gb <N>    Trim the recipe cache to N GiB (decimals allowed) in the
                        background: once the first model load is done (at start with
                        --lazy), then every --cache-trim-every minutes, least recently
                        used recipes first (see --trim-cache; default: never trimmed).
                        Recipes this server has loaded or compiled are never removed
  --cache-keep-days <D> With --cache-max-gb: recipes used within D days are kept even
                        over the budget (default: 1; decimals allowed)
  --cache-trim-every <M> With --cache-max-gb: minutes between trims (default: 60)
  --profile-steps <A-B[:file]>  Device trace of the batched launches A..B (0-based,
                        counted from the load) of a model served with --parallel
                        lanes: prints each launch's split by node family (a slot's
                        nodes fold together as s#_) and engine; :file also writes the
                        raw events. Repeatable. The batched recipes compile fresh
                        (the others load from the recipe cache); a profiled run is
                        not a timing result
  --lazy                Start without holding Gaudi 2; load on first chat/start
  --build-cache         With --serve: compile every recipe the served models load into
                        the recipe cache and exit, without opening the card and without
                        listening (safe beside a running server). Each model takes the
                        engine configuration a load of it takes, so a start after it
                        compiles nothing. Run it with the server's own options after any
                        change to kernels, graphs or contexts
  --build-cache-only <id>  The same for one served model id (repeatable)
  --idle-park <secs>    Auto-release Gaudi 2 after N seconds idle (0=off, default: 300)
  --prefix-cache <N>    Prefix reuse across requests: a request whose prompt starts like
                        the one before it prefills only the rest, with the same output
                        as a whole prefill. N = positions of the previous prompt whose
                        recurrent state is kept on the card to resume from (default: 8;
                        0 = off; at most 4096; 0.15 GiB each for a 48-layer DeltaNet
                        model). The
                        context is planned first: a load keeps as many as fit beside
                        it and says so. A request with cache_prompt: false prefills
                        its whole prompt. Dense Qwen3.5 / Qwen3.8 models without a
                        drafter or with DSpine
                        (N positions), and Inkling 2 (any N > 0: the rows on the card
                        are resumed while the bounded rings still hold what the rest
                        of the prompt reads, and a copy of the rings taken when each
                        prompt is in, 0.6 GiB, covers a long generation after it)
  --parallel <N>        Requests a model runs at once (default: 1; at most 16): each in a
                        lane of its own (its own K/V rows, recurrent state and prefix
                        reuse), the decoding lanes stepped together in one launch that
                        reads the weights once. Every stream is token-identical to the
                        same request run alone. Further requests wait in arrival order;
                        a request with images runs alone. Each lane holds the full
                        context (--max-ctx / --model-ctx): a load that cannot plan N
                        lanes of it is refused with the numbers. A request goes to the
                        free lane where it takes over the most of its prompt for the
                        least of another conversation's rows. Dense Qwen3 /
                        Qwen3.5 / Qwen3.8 models without a drafter or with a DSpine
                        drafter whose verify rows are replayed or checkpointed:
                        lanes take the replay form (no per-row state copies, about
                        2 GiB less per lane on the 27B) unless GAUDIUM_GDN_VERIFY
                        names a form. With replay each lane verifies at a width of
                        its own, the narrowest or the widest --draft-k, chosen every
                        step from its own acceptance and the measured step times;
                        with checkpoint every lane steps at the narrowest. Other
                        models are
                        served with one lane (the log says why). /props total_slots
                        and /slots show the lanes. With --cache-ram every lane saves
                        to and restores from the one host-RAM store (an entry one
                        lane saved is restored into any lane)
  --cache-ram <GB>      Host memory for prefix reuse when the card no longer holds a
                        conversation: after another conversation, another model, or
                        the idle release (--idle-park). What a request could resume
                        from (K/V rows of the prompt, the state at its kept positions)
                        is copied to host memory after the response, and copied back
                        instead of prefilled when that is faster by the engine's own
                        measured rates; the output is that of a whole prefill. GB may
                        be a decimal; 0 = off. Default: a quarter of the host memory
                        available at start (the smallest of what the host and the
                        control groups above the process have left; version 1 and 2
                        memory controllers are read). No default is taken, and the
                        cache is off, when a memory limit of the process cannot be
                        read: name a budget to turn it on there. Memory is taken as
                        prefixes are saved and given back when they leave; a full
                        cache evicts the entry used least recently. A budget that does
                        not fit in what is available is refused at start with the
                        numbers. While a model is loaded, 256 MiB of pinned transfer
                        buffers come on top, and 128 MiB for a moment at each load;
                        all of it is asked for against what the host has left at that
                        moment, and an engine that cannot have it serves without the
                        cache. Needs
                        --prefix-cache > 0; about 64 KiB per token and 0.15 GiB per
                        kept position for a 16-attention-layer, 48-DeltaNet-layer model
  --queue-config <file> Card-queue settings, re-read on every turn (or GAUDIUM_QUEUE_CONFIG;
                        default: queue.conf in the parent of GAUDI_LIVE_DIR, the card-lock
                        state dir). Keys: group_max_wait_min (default 45), group_max_skips
                        (default 3): a request for the loaded model may run ahead of older
                        requests until one of them has waited that long or been passed over
                        that many times (a request a --parallel lane session admits
                        counts too: the session then lets its lanes drain and hands
                        the card on)
  --draft-model <id=dir> Drafter checkpoint for model <id>: speculative decoding, output
                        token-identical to K=0 (repeatable, one per model). The kind
                        follows the directory name: a block drafter when it contains
                        dflash (7 drafts per launch), a model's own drafter when it
                        ends in -assistant (4 drafts), otherwise DSpine. <id>=mtp
                        drafts with the model's own MTP head (no checkpoint; 4 drafts,
                        3 in the routed mixture-of-experts form; an Inkling 2 model:
                        its block drafter, 7 drafts, as --spec mtp finds it)
  --moe-form <id=f>     Mixture-of-experts model <id>: dense (default) or routed (faster
                        decode, more device memory, its own token stream)
  --model-ctx <id=N>    Context size for model <id> instead of --max-ctx
  --effort-levels <id=l,..>  Reasoning-effort levels of model <id> instead of the ones
                        read from its chat template (each one the template renders
                        without raising): levels of minimal, low, medium, high,
                        xhigh, max, or none for no effort setting. /props advertises
                        them; a request's reasoning_effort goes to the nearest one
  --precision <id=p>    Weight precision of model <id>: auto (default), bf16 or fp8 (a
                        BF16 checkpoint is quantised at load per output channel; give
                        its activation-scale table with --fp8-act)
  --draft-k <N[,N..]>   DSpine drafts verified per launch (default: 7,15 adaptive)
  --no-vision           Load no vision tower: every model is served text only. By default
                        a model whose checkpoint has a vision tower takes images (JPEG,
                        PNG, GIF, WebP, BMP as data: URLs) and videos (an animated GIF /
                        WebP, or a list of frames with their rate) in chat requests.
                        A GGUF model's tower is a projector file (mmproj*.gguf) beside
                        it; with none there it is served text only.
                        Drafters that run with image input: DSpine, a dense Qwen
                        model's DFlash2 drafter and MTP head, Inkling 2's own drafter
                        and Gemma 4's -assistant drafter; a model with another drafter
                        is served text only (Gemma 4 takes images, no video)
  --vision-max-patches <N>  Largest image the vision tower is planned for, in the
                        model's patches (default: 4096; 16x16 pixels for Qwen3.5 / 3.8:
                        1024x1024 pixels, larger images are resized down to it; 20x20
                        for Inkling 2, which never resizes: an image of more than N/4
                        tiles of 40x40 pixels is refused; Gemma 4: an image is resized
                        to whole 48x48-pixel soft tokens of 9 patches, at most N/9 and
                        at most the checkpoint's own budget). The tower's device memory
                        grows with N^2 (about 160 bytes x N^2: 2.5 GiB at 4096)
  --vision-max-tokens <N>  Image tokens one request may hold, all images and video
                        frames together (default: 16384; 10 KiB of device memory each)
  --model-vision <id=N|off>  The vision tower of model <id> instead of the two options
                        above: its largest image in patches, or off for text only
  --fp8-act <id=file>   FP8 activation-scale table for FP8 model <id> (repeatable)
  --fp8-cache <id=dir>  Cache directory for model <id>'s quantised FP8 matrices
  --kv-cache <id=d>     K/V cache of model <id>: bf16 (default) or fp8 (half the bytes per
                        position, so a longer context fits; its K=0 stream is its own and
                        the reference of its drafter; dense Qwen3.5 / 3.8 hybrids, K=0 or
                        DSpine; needs the scale table from --kv-calibrate)
  --kv-scales <id=file> K/V calibration of model <id> (default:
                        <model_dir>-gaudium-pc/kv_scales.txt)
  --api-key <key>       Optional Bearer / X-Api-Key auth token (or GAUDIUM_API_KEY)
  --a2ui                Offer A2UI 1.0 interfaces (or GAUDIUM_A2UI=1): a chat request that
                        declares its renderer in the field gaudium_a2ui (true, or an
                        object with supportedCatalogIds) gets the instructions that teach
                        the model to write <a2ui-json> interface blocks in its reply;
                        the Web UI then offers interactive interfaces in its settings.
                        GET /gaudium/a2ui shows what is offered and the instructions.
                        Requests that declare nothing are unchanged
  --a2ui-prompt <file>  Read the A2UI instructions from <file> instead of the built-in
                        text (or GAUDIUM_A2UI_PROMPT), and re-read it whenever it
                        changes: edits apply to the next request, no restart. A file
                        that is missing, empty, over 256 KiB, not UTF-8 text, or that
                        never names the <a2ui-json> tags is not used: the last good
                        version (or the built-in text) stays, the reason is logged and
                        shown under instructions_source in GET /gaudium/a2ui
  --tls-cert <file>     Serve HTTPS as well: the PEM certificate (with its chain) for
  --tls-key <file>      the server, and its unencrypted PEM private key. HTTP and HTTPS
                        then share the port, and a plain-HTTP request is answered with a
                        redirect to https on the same host and port (see --tls-http).
                        Opened over https the Web UI may use the microphone and the
                        clipboard from other machines. TLS comes from the system's
                        OpenSSL 3 (libssl.so.3), loaded when asked for: without it the
                        server starts, says that HTTPS is not offered, and serves plain
                        HTTP. Behind a front end that ends TLS itself give neither option
  --tls-http <mode>     What plain-HTTP requests get while HTTPS is served: redirect
                        (default) sends clients on other machines to https with a 308
                        (method and body kept; a missing or malformed Host header is a
                        400) and answers clients on this machine's loopback interface
                        (local tools; their traffic never leaves the machine); strict
                        redirects every client, loopback included; allow answers plain
                        HTTP from every client as well (tools elsewhere that cannot
                        speak HTTPS)
  --hsts-max-age <secs>  max-age of the Strict-Transport-Security header every response
                        over HTTPS carries (default: 86400, one day; at most 63072000).
                        A browser that saw it uses https for this host name, on every
                        port, until it expires; 0 tells browsers to forget it. No
                        includeSubDomains, no preload
  --mcp-proxy           Offer the MCP proxy (/cors-proxy) to the Web UI: the server
                        makes the requests to an MCP server instead of the browser (for
                        servers without CORS headers, servers only this machine
                        reaches, and plain-http servers used from an https page). Off
                        by default. It needs the targets it may reach (the two options
                        below); requests need the API key when one is set; a page of
                        another site cannot drive it. GET /gaudium/proxy shows its
                        settings and last decisions
  --mcp-proxy-allow <list>  Targets the proxy may reach, comma separated (repeatable):
                        host, host:port, *.domain, an address, an address range
                        (192.0.2.0/24), or * for any target (* needs --api-key). Each
                        address a target resolves to must be public, and the
                        connection goes to the address that was checked
  --mcp-proxy-allow-private <list>  Targets that may be reached although their address
                        is loopback, private or link-local: the same forms without *
                        (localhost:8931, 127.0.0.1, 192.168.0.0/16). An entry here
                        also allows the target, so an MCP server on this machine
                        needs this option only
  --mcp-proxy-max-mb <N>  Largest response the proxy passes on, in MiB (default: 64)
  --mcp-proxy-timeout <secs>  Longest silence of a target, or wait on a page that
                        stopped reading, before the proxy gives up (default: 600).
                        An idle event stream gets a comment line every 20 s so
                        that a front end does not cut it
  --mcp-proxy-ca <file>  PEM certificate authorities the proxy trusts for https targets
                        besides the system's (an MCP server with a private certificate)
  --verbose             Print server, HBM, and recipe diagnostics

Live status environment variables:
  GAUDI_LIVE_DIR        Directory for live dashboard JSON records (GAUDI_LIVE=0 disables).
                        A directory that does not exist turns the records off
  GAUDIUM_LIVE_ROOT     Used when GAUDI_LIVE_DIR is unset: the records go to
                        <root>/.cardlock/live
  GAUDI_LIVE_TAG        First part of the record's file name, <tag>.<pid>.json (default:
                        the host name)
  GAUDI_CARD_OWNER      Owner name written in the record and in the card lock's holder
                        file (default: server/<engine>)
  GAUDI_CARDLOCK_DIR    Card-lock directory that gets the holder file and the lock log
                        (default: the parent of a records directory named `live`)
  GAUDI_CARD_NO_HBM_REPORT  1 = do not publish the device memory in use to
                        /dev/shm/mem_usage_accel<N>
  GAUDI_CARD_ACCEL_INDEX  The <N> of that file (default: 0)

Engine environment variables:
  GAUDIUM_REQUEST_RESET Device state zeroed before each request: auto (default), dirty
                        or full (see --request-reset under Generate options)
  GAUDIUM_UPLOAD_DIRECT 0 = never read weight files with direct I/O at load (default:
                        weight bytes that are not in the page cache are read directly)
  GAUDIUM_ACQUIRE_RETRIES  Attempts to acquire the card at start, 2 s apart (default: 10)
  GAUDIUM_SYNAPSE_LIB   The SynapseAI library to load, tried before
                        /usr/lib/habanalabs/libSynapse.so and the loader's search path
  GAUDIUM_<KERNEL>_ELF  One TPC kernel's ELF file, read by the graph-compiler glue. When
                        unset, the engine sets it to that kernel's file in the kernels
                        directory
  GAUDIUM_W4_GLUE_DEBUG Set (any value) = the graph-compiler glue of the MoE W4 kernels
                        prints the tensor shapes of a node it refuses
Read when the engine is built, not when it runs:
  GAUDIUM_BUILD_COMMIT  The commit `gaudium --version` prints after the version (release
                        builds set it; unset = the version alone)
SDK environment variables the engine reads and completes:
  GC_KERNEL_PATH        The engine puts its glue libraries first, keeps the entries
                        already there and adds the SDK's TPC kernel library
  HABANA_SCAL_BIN_PATH / HABANA_PLUGINS_LIB_PATH
                        Left as set; when unset, set to the SDK's usual directories if
                        those exist
  HABANA_PROFILE        The SDK profiler's settings. --profile-steps sets its own when
                        this is unset
Measurement switches of the graph compiler (not for serving; each is part of the recipe
fingerprints it touches):
  GAUDIUM_GC_SET        `NAME=value[,NAME=value..]`: graph-compiler settings applied to
                        the graphs this process compiles (set for the compile, then
                        restored); GAUDIUM_GC_ONLY limits them to recipes whose name
                        starts with one of its `|`-separated prefixes; GAUDIUM_GC_TRACE
                        prints each setting's previous value
  GAUDIUM_HALFPROBE_GC  Sets of such settings (sets separated by `;`) that --prefill
                        halfprobe=N times on the tile layers, each against the default
  GAUDIUM_HALFPROBE_SEQS_LIN / GAUDIUM_HALFPROBE_SEQS_ATTN
                        `name=order[;name=order..]`: node orders of the two-body
                        (DeltaNet / attention first half + second half) recipes for
                        halfprobe to time. An order is `B:node` runs (body F or G up to
                        and including the node, `*` = the rest) and `B=node` single
                        nodes, `,`-separated; a leading `~` orders only the named nodes
  GAUDIUM_HALFPROBE_NCH KV chunks the probe's attention recipes read (default 1; the
                        prompt must have filled them)
  GAUDIUM_HALFPROBE_CHUNK  1 = halfprobe also times the DeltaNet chunk GEMMs alone with
                        f32 and with BF16 operands
Qwen prefill environment variables:
  GAUDIUM_PREFILL       Prefill levers `k=v,..` when --prefill is not given (see --prefill
                        under Generate options). Changes the compiled tile recipes; with
                        BF16 weights streams may differ between settings at exact BF16
                        logit ties, s16 > 0 is not token-identical to s16=0.

Gemma 4 environment variables:
  GAUDIUM_G4_NORM       Layer RMS-norm node: row | tile | vendor (default: tile for sizes with
                        per-layer inputs, else vendor). Changes compiled graphs; streams may
                        differ between settings at exact BF16 logit ties.

Inkling 2 environment variables (debugging; none changes a served stream unless noted):
  GAUDIUM_IK_AMAX_BF16  1: decode the 4-bit expert matrices with their per-expert scale terms
                        rounded to BF16 first, as the publisher's reference loader ends up
                        doing (for comparisons with it; changes the weights slightly)
  GAUDIUM_IK_APPS       Load and run only the first N layer applications (kernel checks)
  GAUDIUM_IK_STEP_PREFILL  1: prefill the prompt one row at a time with the decode step
  GAUDIUM_IK_DUMP       Directory: after the prompt, every tensor of the one-row decode step and
                        the head of every K / V state is written there as <name>.bin
  GAUDIUM_IK_VERIFY_CHECK  N (2..64): after a greedy request its tokens are run again in passes
                        of N rows and every row's logits are compared with the one-row step's
  GAUDIUM_IKV_DUMP      Directory: the vision tower writes, per image k of the first image
                        request, img<k>_emb / _pre / _l00 / _l11 / _l22 (sequence rows x 1024)
                        and img<k>_feat (tokens x 2304) as raw BF16 (it syncs after each one)
  GAUDIUM_IK_MTP_DUMP   Directory: after the first block of the drafter (--spec mtp), every
                        drafter activation and the head of its context state as <name>.bin

Generate options:
  --model <dir>         Model directory (or GAUDIUM_MODEL); or a GGUF file (Inkling 2)
  --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)
  --prompt <text>       Input prompt text
  --prompt-ids <csv>    Raw comma-separated prompt token IDs (skips tokenizer)
  --prompt-file <path>  Read the prompt text from a file (long-context prompts)
  --prompt-ids-file <p> Read comma/whitespace-separated prompt token IDs from a file
                        (repeatable: each file is one request on the same loaded engine)
  --also-prompt-ids-file <p>  Same as a further --prompt-ids-file
  --also-prompt-file <p>  One more text request on the same loaded engine (repeatable;
                        formatted like --prompt, after the token-ID requests)
  --chat                Wrap --prompt in the model's chat template (Qwen3 ChatML / Gemma 4 turns;
                        other families: the model's own template, reasoning off with --no-think)
  --no-think            Disable thinking block (with --chat)
  --system <text>       System prompt (with --chat)
  --max-new <N>         Maximum tokens to generate (default: 64)
  --max-ctx <N>         Context capacity, multiple of 4096 (default: 4096)
  --temperature <F>     Sampling temperature; 0 = greedy (default: 0.0)
  --top-p <F>           Nucleus sampling p (default: 1.0)
  --top-k <N>           Top-k sampling cap (default: 0 = off)
  --min-p <F>           Min-p sampling threshold (default: 0.0)
  --seed <N>            Sampling RNG seed (default: 0)
                        Sampled requests sample on the card (the sampler is part of the
                        recipes; the token at a position is a function of the prefix,
                        the seed and the position, so speculative runs equal K=0)
  --sampling <T[:top_k[:top_p[:min_p[:seed[:coupled[:check]]]]]]>  One more sampler
                        setting (repeatable; T = 0 is greedy): every request runs once
                        per listed setting, in place of --temperature / --top-k / --top-p /
                        --min-p / --seed / --coupled / --sample-check
  --coupled <0|1>       Sampled speculative requests: DSpine drafts are drawn with the
                        verify's position noise on the drafter's own candidate scores
                        (default 1). Tokens never change, only draft acceptance
  --draft-trunc <0|1>   Coupled drafts also apply top-k / top-p / min-p to the drafter's
                        candidates (default 1; 0 = temperature and noise only)
  --sample-check        Debug: sample every row again on the host from the card's own
                        logits (bit-exact CPU twin; also the coupled walk) and report
                        rows / mismatches. One launch at a time: not a timing run
  --compare-k0          Run every request first on the plain non-speculative K=0 loop
                        of the same engine (same seed and sampler settings), then as
                        configured (speculatively with --spec, enqueue-ahead K=0
                        without), and report identity or the first divergence (greedy:
                        with both tokens' logits at every position of the run's stream
                        where K=0 would have chosen otherwise, from a teacher-forced replay)
  --profile-steps <[p]A-B[@R][:file]>  Device trace of decode launches A..B (0-based; with
                        a leading p: of prefill launches A..B, counted over the tile
                        loop's embed and layer launches) of
                        request R (default 0; with --compare-k0 the K=0 reference of a
                        prompt is the request before its speculative run): prints the
                        launch split by node family and engine, and the idle time by
                        node transition; :file also writes the raw events (tab-separated).
                        Repeatable. Recipes compile fresh (recipe cache off); a profiled
                        run is not a timing result
  --prefill-warm <N>    Load the prefill recipes of prompt-tail lengths 1..N with the
                        engine (compiling them on a recipe-cache miss) instead of inside
                        the first request of each prompt length (default: 0 = first use)
  --no-eos              Do not stop early on EOS token
  --image <file>        Image input: JPEG, PNG, GIF (first frame), WebP, BMP or binary
                        PPM, decoded as the server decodes a request's image;
                        repeatable, in prompt order. An image belongs to the first
                        request, or to the further request (--also-prompt-file,
                        --also-prompt-ids-file, a repeated --prompt-ids-file) it follows
  --image-raw <WxH:f>   Image input, raw RGB8 file with explicit size; repeatable
  --vision-max-patches <N>  Largest image the vision tower is planned for, in the
                        model's patches (default: the largest --image; Qwen3.5 / 3.8
                        resize larger images down to fit, Inkling 2 refuses them)
  --vision-gelu-bf16    Evaluate the vision tower's GELU in BF16 (default: F32);
                        slightly faster, slightly less accurate image embeddings
  --dump-vision-emb <p> Write the image embeddings (tokens x hidden, raw BF16) to a file
  --dump-prompt-ids     Print the final prompt token IDs to stderr and continue
  --repeat <N>          Run the same request N times in one process (warm timings)
  --after-ids-file <p>  Then run a text-only request with these prompt token IDs, once
                        per sampler setting (checks that an image request leaves no
                        state behind: its AFTER TOKENS lines must equal the tokens of
                        the same request on an engine that never saw an image)
  --after-prompt-file <p>  The same with a text prompt, formatted like --prompt
  --spec <method>       Speculative decoding: dspine or dflash2 (on-card block drafters),
                        mtp (the target's own MTP head, hybrid Qwen3.5/3.8, dense or
                        mixture-of-experts, either --moe-form), assistant (a Gemma 4
                        size's own drafter checkpoint) or oracle (test: drafts from
                        --oracle-file); absent = non-speculative K=0.
                        A Gemma 4 model reads the drafter kind from the checkpoint:
                        --spec dflash2 --draft-model <dir> runs a DFlashDraftModel
                        block drafter (one --draft-k width, default 7).
                        An Inkling 2 model: mtp drafts with the checkpoint's block
                        drafter (mtp.safetensors beside the model; up to 7 drafts per
                        block, pass --draft-k 7 for all of them). A GGUF model reads
                        it from a drafter GGUF beside the model file or in a dspark/
                        directory there (BF16 before Q8_0 when several match).
                        Chain modes are token-identical to K=0 (sampled: same seed)
  --draft-model <dir>   Drafter checkpoint directory (with --spec dspine / dflash2;
                        assistant: default <model_dir>-assistant)
  --draft-k <N[,N..]>   Drafts verified per launch, 1..31 (default: 15; dflash2: 1..7,
                        default 7; mtp: at most 15, default 4; assistant: default 4);
                        several values let the engine pick a width per step from
                        acceptance (mtp: every step then drafts the widest listed block;
                        --parallel lanes: each lane at the narrowest or the widest
                        listed in the replay form, else at the narrowest)
  --sweep-draft-k       Measurement: with several --draft-k widths, run every request once
                        per listed width as a fixed-width run, then once with the width
                        policy (one engine load for a whole width sweep)
  --tree <shape>        Verify a draft tree per launch (with --spec mtp or oracle):
                        D = spine depth, D:j1-j2@r1-r2,.. adds the rank r1..r2
                        candidates (2 = runner-up) at depths j1..j2, e.g. 4:1-2@2.
                        Opt-in: a leaf row may differ from K=0 in the last bit, so the
                        stream can fork at an exact BF16 tie; with --spec mtp an
                        --oracle-file is a reference stream whose first fork is reported
  --k0-first            Same as --compare-k0
  --draft-vocab <N>     Draft readouts over N token ids only: the lowest ids plus the
                        control-token block at the lowest EOS id (multiple of 128; 0 =
                        the full vocabulary; or GAUDIUM_DRAFT_VOCAB). Default: the full
                        vocabulary for dspine (as an FP8 copy of the readout when the
                        drafter's FP8 table has its site) and for mtp, 98304 for
                        dflash2. A token outside the subset is never drafted (a large
                        share of the tokens outside English text and code); the subset
                        reads fewer bytes per block. Drafts are guesses: the output is
                        unchanged, only acceptance and the step time can move
  --draft-fp8 <0|1>     DSpine drafter layers as FP8 weight GEMMs (default, or
                        GAUDIUM_DRAFT_FP8: on when the drafter's activation-scale table
                        exists). The verify stays as it is: the output is unchanged
  --draft-fp8-act <f>   The drafter's activation-scale table (or GAUDIUM_DRAFT_FP8_ACT;
                        default <draft dir>-gaudium-pc/act_scales_ds.txt)
  --draft-fp8-calibrate <f>  Calibration run for that table: the drafter runs BF16, one
                        launch at a time, its GEMM input sites are read back after every
                        launch and the table is written to <f> after each request
  --draft-ring <0|1>    DSpine context caches as rings of the checkpoint's sliding window
                        instead of one row per context position: the drafter's share of
                        a step then does not grow with --max-ctx (default, or
                        GAUDIUM_DRAFT_RING: on when the ring kernels and the step glue
                        library are in the kernels directory). Drafts only: the output
                        is unchanged
  --width-sweep         Measurement: every request runs once per fixed verify width of
                        --draft-k, then once with the width policy (compare the
                        policy against the best fixed width per prompt)
  --gdn-fast-rows <0|1> Checkpoint form of --gdn-verify: run the rows with the kernel
                        build that takes its per-row gate scalars as vector loads
                        (bit-identical; default, or GAUDIUM_GDN_FAST_ROWS: on when that
                        kernel and the step glue library are in the kernels directory)
  --k0-pipeline <0|1>   Greedy K=0 requests of the dense models decode enqueue-ahead:
                        the next step is queued while the current one runs, its token
                        fed back on the card (default 1, or GAUDIUM_K0_PIPELINE; same
                        recipes and tokens as 0, without the host round trip per token)
  --gdn-verify <m>      Gated DeltaNet state across the verify rows of a speculative
                        step (dspine / dflash2 / mtp chains; or GAUDIUM_GDN_VERIFY):
                        checkpoint (default: one state checkpoint per row) or replay
                        (no checkpoints: the next step re-applies the accepted rows'
                        updates); both are token-identical to K=0. mme: the rows read
                        the state through MME GEMMs and commit in one closed form:
                        faster per row, NOT token-identical (another fp32 summation
                        order: a stream may fork at a near-tie; check with --compare-k0)
  --oracle-file <p>     Known continuation token IDs (with --spec oracle; repeatable,
                        one per request in order)
  --oracle-corrupt <csv> Generated-token indices whose oracle draft is made wrong
                        (applies to the --oracle-file options that follow it; with
                        --tree the leaf of that depth carries the true token)
  --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)
  --no-cache            Disable on-disk compiled recipe cache
  --build-cache         Compile every recipe an engine load with these options uses
                        into the recipe cache, print the exact HBM plan and exit,
                        WITHOUT opening the card (runs beside a process that owns it).
                        Gemma 4: also the decode steps of every context bucket, the
                        full-tile prefill of every chunk count, and what the given
                        prompts and images launch (their last tiles, image tiles and
                        image sizes); it plans the widest prefill tile
  --build-cache-chunks <csv>  With --build-cache and --spec on a mixture-of-experts
                        model: compile the speculative step recipes of exactly these KV
                        chunk counts (ceil(position / 4096)) instead of all of them;
                        routed-form steps take tens of seconds each
  --precision <p>       Weight precision: auto (default: FP8 for a block-FP8 checkpoint,
                        else BF16), bf16 (an FP8 checkpoint is dequantised at load), fp8
                        (a BF16 checkpoint, Gemma 4 included, is quantised at load per
                        output channel; needs the model's table from --fp8-calibrate.
                        FP8 is a precision of its own: its K=0 stream is not the BF16 one).
                        A Gemma 4 model given --fp8-act runs FP8 under auto
  --fp8-act <file>      FP8 activation-scale table (or GAUDIUM_FP8_ACT; default:
                        <model_dir>-gaudium-pc/act_scales_lc.txt)
  --fp8-cache <dir>     Cache directory for the quantised FP8 matrices (or GAUDIUM_FP8_CACHE)
  --kv-cache <d>        K/V cache: bf16 (default) or fp8 (FP8 q / K / V / P in the trunk's
                        attention, half the K/V bytes; a precision of its own: its K=0
                        stream is the reference of FP8-K/V speculation. Dense Qwen3.5 / 3.8
                        hybrids, K=0 or DSpine; needs the table from --kv-calibrate;
                        its prefill runs 256-row tiles)
  --kv-scales <file>    K/V calibration the FP8 scale tables are made from (default:
                        <model_dir>-gaudium-pc/kv_scales.txt)
  --moe-form <f>        MoE expert execution: dense (default: one masked wide MLP per layer)
                        or routed (stacked BF16 experts, only the routed ones are read;
                        faster decode, more device memory, its own token stream: the
                        two forms round differently)
  --request-reset <m>   Device state zeroed before each request: auto (default: the
                        DeltaNet start state; KV rows past the cursor are masked),
                        dirty (also the KV rows earlier requests wrote) or full
                        (or GAUDIUM_REQUEST_RESET; same output in every mode)
  --prefix-cache <N>    Prefix reuse between the requests of one run (default: 0 = off,
                        every request prefills its whole prompt): keep the recurrent
                        state at N positions of the previous prompt (see --prefix-cache
                        under Server options)
  --prefill <k=v,..>    Prefill levers of the Qwen engines (default: all on): tile=N rows
                        per tile (multiple of 64 up to 512; 0 = the widest that fits),
                        mme=0|1 DeltaNet recurrence as chunked MME GEMMs, rows=0|1|2 its
                        prep / norm kernels, fastexp=0|1 fast-exp tile softmax, groups=N
                        attention chunk groups, s16=0|1|2 half-width attention core (BF16
                        scores; 2: + FP8 P; default 2 with dense FP8 weights, else 0: it
                        changes the prefill numerics), wave=0|1 full tiles two in flight
                        (the halves of two tiles' layers share a launch so the MME and
                        the TPCs overlap; same nodes, same output; needs the mme and
                        fastexp forms), plain = every lever off, stats /
                        profile = print the tile loop's host split / per-recipe times,
                        halfprobe=N = measurement: after the last request, time whole
                        tile layers, their halves and two halves in one recipe over N
                        launches each (with --build-cache: compile them without the
                        card and the recipe cache, for the compiler's own dumps)
  --dump-tokens         Print generated token IDs to stderr
  --force-tokens-file <p> Teacher-force this reference stream (comma/whitespace token IDs)
                        and report every position where greedy K=0 disagrees, with logits
  --verbose             Print HBM, recipe compile, and timing diagnostics

FP8 calibration options (prefill only, BF16 graphs; writes `<site> <k> <amax> <rms>` lines):
  --model <dir>         Model directory (or GAUDIUM_MODEL)
  --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)
  --prompt-ids-file <p> Calibration prompt as token IDs (repeatable)
  --prompt-file <path>  Calibration prompt as text (repeatable)
  --out <file>          Table to write (default: <model_dir>-gaudium-pc/act_scales_lc.txt)
  --max-ctx <N>         Context capacity, multiple of 4096 (default: 4096)
  --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)
  --no-cache            Disable on-disk compiled recipe cache
  --verbose             Print HBM, recipe compile, and per-site diagnostics

K/V calibration options (prefill only, BF16 caches; writes `layer <l> k <amax..> v <amax..>`
lines, the largest |K| and |V| per kv head; FP8 K/V takes its V multipliers from them):
  --model <dir>         Model directory (or GAUDIUM_MODEL)
  --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)
  --prompt-ids-file <p> Calibration prompt as token IDs (repeatable)
  --prompt-file <path>  Calibration prompt as text (repeatable)
  --out <file>          Table to write (default: <model_dir>-gaudium-pc/kv_scales.txt)
  --max-ctx <N>         Context capacity, multiple of 4096 (default and least: what the
                        longest prompt needs, with the engine's reserve)
  --precision <p>       Weight precision of the measuring load, as the served one: auto
                        (default), bf16 or fp8
  --fp8-act <file>      FP8 activation-scale table of an fp8 load
  --fp8-cache <dir>     Cache directory for the quantised FP8 matrices
  --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)
  --no-cache            Disable on-disk compiled recipe cache
  --verbose             Print HBM, recipe compile, and per-layer diagnostics

Cache trim options (--trim-cache; an entry is a <name>-<hash>.recipe file and its .sha256
sidecar, removed together; every other file in the directory is left alone and not counted;
each cache hit sets the sidecar's modification time, so recency needs no access times):
  --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)
  --cache-max-gb <N>    Budget: GiB of cache entries left afterwards (decimals allowed;
                        0 = every entry outside the keep window)
  --cache-keep-days <D> Entries used within D days stay even over the budget (default: 1;
                        an entry dated ahead of the clock counts as just used)
  --dry-run             Print what would be freed, remove nothing

Certificate options (--make-cert <dir>: writes ca.crt and ca.key, the authority, when the
directory has neither of them, and server.crt (the chain) and server.key, signed by it; the
keys get mode 0600 and no file is replaced without --cert-overwrite; prints the SHA-256
fingerprints and how to trust ca.crt on each kind of device; docs/mcp-proxy-https.md):
  --cert-names <list>   Names and addresses the Web UI is opened by, comma separated
                        (repeatable): host names (gaudium.lan) and IPv4 or IPv6 addresses.
                        localhost, 127.0.0.1 and ::1 are always added. Default: this
                        machine's host name and the address of its default route. A new
                        authority may sign only for these names (and names under them),
                        loopback and local-network addresses
  --cert-days <N>       Days the server certificate is valid (default and most: 825, the
                        longest Apple's systems accept; never past the end of the
                        authority, which is valid for 10 years)
  --cert-overwrite      Replace the server certificate and key that are there (a new name
                        or address, or a renewal). The authority is never replaced: for a
                        name it does not cover, move ca.crt and ca.key away to make a new
                        one, and trust it again on each device

Vision suite options (a row = one model and one method of the suite file; per row the
text request, every image case and the text request again, greedy and seeded sampled,
each checked for token identity to its K=0 reference, a right answer, and the text
request after the images giving the fresh engine's tokens; exit status 0 = all passed):
  --models <dir>        Directory of the suite's relative model names and of {models}
  --kernels <dir>       Built TPC-C kernels & glue dir (or GAUDIUM_KERNELS)
  --cache-dir <dir>     Recipe cache directory (or GAUDIUM_RECIPE_CACHE)
  --out <dir>           Prompts, each row's command line and logs, report.json
                        (default: a new directory under the system temp directory)
  --only <row>          Run only this model, or model/method (repeatable)
  --build-cache         Compile every selected row's recipes into the recipe cache
                        without the card (--build-cache rows) instead of running them
  -- <options>          Further --generate options for every row
```
<!-- help:end -->

## Variables read only by the tests

These are not settings of the engine. A test that needs one is marked `#[ignore]` and says so,
or reads it to rewrite a file.

<!-- test-variables:begin -->

| Variable | Read by | Meaning |
|---|---|---|
| `GAUDIUM_INKLING2_DIR` | `tokenizer::tests::inkling2_ids_match_the_reference`, `server::chat::tests::inkling2_template_renders_like_jinja2`, `server::chat::tests::inkling2_requests_render_through_the_model_template`, `engine::inkling2::vision::tests::pos_rows_match_the_reference` (`--ignored`) | An Inkling 2 model directory: its `tokenizer.json` must encode and decode the cases of `gaudium/tests/data/tokenizer/inkling2_cases.json` exactly, its `chat_template.jinja` must render the contexts of `gaudium/tests/data/chat/inkling2_renders.json` as Jinja2 does, and the vision tower's position rows and class row resampled from its checkpoint must match `gaudium/tests/data/vision/inkling2/fixture.json` bit for bit (all written by the generator scripts next to them). |
| `GAUDIUM_IMAGE_REF_DIR` | the decoder tests in `src/vision/decode/` (`--ignored`) | A directory of image files with the raw RGB dump the reference image library wrote for each; every file must decode to the same bytes. |
| `GAUDIUM_IMAGE_PIPELINE_REF_DIR` | `vision::media::tests::reference_pipeline_records_match_bit_for_bit` (`--ignored`) | A directory written by `gaudium/tests/data/vision/make_pipeline_refs.py`: image files with the patch rows of the checkpoint's own image processor, which must match bit for bit. |
| `GAUDIUM_G4V_MODEL` | `engine::gemma4::vision::tests::reference_pixel_rows_match_bit_for_bit` (`--ignored`) | The Gemma 4 model directory whose processor settings the check uses (with `GAUDIUM_G4V_REF_DIR`). |
| `GAUDIUM_G4V_REF_DIR` | `engine::gemma4::vision::tests::reference_pixel_rows_match_bit_for_bit` (`--ignored`) | A directory of Gemma 4 image records (`tests/data/vision/gemma4/make_gemma4_refs.py`): the processor's pixel rows must match them bit for bit. |
| `GAUDIUM_VISION_REF_DIR` | `vision::tests::reference_prompt_records_match_bit_for_bit` (`--ignored`) | A directory of reference prompt records (token ids, image grids, positions, patch rows) that the prompt layout and the image preprocessing must reproduce. |
| `GAUDIUM_GGUF_REF_DIR` | `gguf::tests::reference_gguf_files_open_and_decode` (`--ignored`) | A directory of published GGUF files (and its `dspark/` subdirectory): every file must open, the Inkling 2 Mini model must have its 744 tensors with architecture `inkling2_loop` and NVFP4 weights, every NVFP4 tensor its per-expert `.scale`, and one row of every type must decode to finite numbers. |
| `GAUDIUM_INKLING2_GGUF` | `engine::inkling2::gguf_source::tests::gguf_and_checkpoint_pack_the_same_device_tensors`, `tokenizer::gguf::tests::inkling2_gguf_tokenizer_matches_the_checkpoint`, `server::chat::tests::inkling2_gguf_renders_like_the_checkpoint`, `engine::inkling2::vision::gguf_tower::tests::projector_and_checkpoint_pack_the_same_tower` (`--ignored`, with `GAUDIUM_INKLING2_DIR`) | A GGUF file of the same Inkling 2 model as the checkpoint directory `GAUDIUM_INKLING2_DIR`: both must pack byte-identical device tensors, the tokenizer built from the file's metadata must equal the one of `tokenizer.json` (every token, every merge, the fixture cases), and the file's chat template must render the Jinja2 fixture contexts and a set of chat requests to the same prompts and ids as the checkpoint's. The projector file beside it must give the checkpoint's tower geometry, image spec and every tower device tensor byte for byte. |
| `GAUDIUM_UPDATE_REFERENCE` | `tests/reference_docs.rs` | `1` = rewrite the copy of the help text on this page from the binary instead of failing when it differs. |

<!-- test-variables:end -->
