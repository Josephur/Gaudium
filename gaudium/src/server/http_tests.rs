//! The chat endpoints over real sockets, with a scripted engine in place of the card: a request
//! goes through the HTTP parser, the model's chat template, the turn queue, the output parser
//! and the reply writer exactly as in service; only the token source is canned.

use std::collections::VecDeque;
use std::io::{Read, Write};

use super::*;
use crate::engine::{GenParams, GenReport};

const TOOLS: &str = r#""tools":[{"type":"function","function":{"name":"get_weather","description":"Weather.","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}},{"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{}}}}]"#;

/// The tower inputs of each request a scripted engine ran: `(grid rows, grid columns, values)`.
type SeenImages = Arc<Mutex<Vec<Vec<(usize, usize, usize)>>>>;

/// Emits the next canned reply token by token and records the prompt it was given.
struct ScriptedEngine {
    spec: ModelSpec,
    tok: Arc<Tokenizer>,
    replies: Arc<Mutex<VecDeque<String>>>,
    prompts: Arc<Mutex<Vec<String>>>,
    /// Tower inputs handed over for the next request: `(grid rows, grid columns, values)` each.
    pending_images: Vec<(usize, usize, usize)>,
    /// The tower inputs each request ran with.
    images: SeenImages,
    /// Prefix reuse as an engine reports it: the prompt of the request before, and the counters.
    /// A request takes over the rows it shares with that prompt, in whole 4-row tiles and never
    /// the last row.
    held: Vec<i32>,
    prefix: crate::engine::qwen3::prefix_cache::PrefixStats,
    /// The host-RAM tier as an engine reports it, and the calls the server made to it.
    tier: Arc<Mutex<ScriptedTier>>,
}

/// What the scripted engine says about its host tier, and what the server asked of it.
#[derive(Default)]
struct ScriptedTier {
    /// Reports of the next requests, in order (none: the engine has no host tier).
    next: VecDeque<crate::engine::qwen3::host_tier::TierReport>,
    last: Option<crate::engine::qwen3::host_tier::TierReport>,
    /// `idle_work` calls, with the number of waiting requests each saw.
    idle_waiting: Vec<u64>,
    releases: u64,
}

impl GenEngine for ScriptedEngine {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn set_request_images(
        &mut self,
        images: Vec<crate::vision::PreparedImage>,
    ) -> Result<(), String> {
        self.pending_images = images
            .iter()
            .map(|i| (i.grid_h, i.grid_w, i.pixels.len()))
            .collect();
        Ok(())
    }

    fn last_vision_report(&self) -> Option<crate::vision::VisionReport> {
        let last = self.images.lock().unwrap().last().cloned()?;
        (!last.is_empty()).then(|| crate::vision::VisionReport {
            images: last.len(),
            reused_images: 0,
            patches: last.iter().map(|g| (g.0 * g.1) as u64).sum(),
            image_tokens: last.iter().map(|g| (g.0 * g.1 / 4) as u64).sum(),
            prep_ms: 1.0,
            encode_ms: 20.0,
            compile_ms: 0.0,
            rope_ms: 2.0,
            rope_rows: 100,
            rope_delta: -3,
        })
    }

    fn max_ctx(&self) -> u64 {
        16384
    }

    fn prefix_stats(&self) -> Option<crate::engine::qwen3::prefix_cache::PrefixStats> {
        Some(self.prefix)
    }

    fn idle_work(&mut self, waiting: &dyn Fn() -> u64) {
        self.tier.lock().unwrap().idle_waiting.push(waiting());
    }

    fn before_release(&mut self) {
        self.tier.lock().unwrap().releases += 1;
    }

    fn host_tier_facts(&self) -> Option<crate::engine::qwen3::host_tier::TierFacts> {
        use crate::engine::qwen3::host_tier::{Rates, TierFacts};
        self.tier.lock().unwrap().last.map(|last| TierFacts {
            rates: Rates {
                to_host: 9 << 30,
                to_device: 14 << 30,
                prefill_rows: 7000,
            },
            bytes_per_row: 65536,
            bytes_per_state: 150 << 20,
            ring_bytes: 175 << 20,
            last,
        })
    }

    fn hbm_bytes(&self) -> [u64; 5] {
        [0; 5]
    }

    fn generate_cancellable(
        &mut self,
        prompt_ids: &[i32],
        params: &GenParams,
        cancel: Option<&AtomicBool>,
        on_token: &mut dyn FnMut(i32),
    ) -> Result<GenReport, String> {
        self.prompts
            .lock()
            .unwrap()
            .push(self.tok.decode(prompt_ids, false));
        self.images
            .lock()
            .unwrap()
            .push(std::mem::take(&mut self.pending_images));
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .ok_or("no scripted reply")?;
        let ids = self.tok.encode(&reply, true);
        let common = prompt_ids
            .iter()
            .zip(&self.held)
            .take_while(|(a, b)| a == b)
            .count();
        let reused = if params.cache_prompt {
            common.min(prompt_ids.len() - 1) / 4 * 4
        } else {
            0
        };
        self.held = prompt_ids.to_vec();
        self.prefix.requests += 1;
        self.prefix.hits += u64::from(reused > 0);
        self.prefix.reused_rows += reused as u64;
        self.prefix.last_prompt_rows = prompt_ids.len() as u64;
        self.prefix.last_reused_rows = reused as u64;
        self.prefix.last_lcp = common as u64;
        let host_tier = {
            let mut t = self.tier.lock().unwrap();
            let r = t.next.pop_front();
            if r.is_some() {
                t.last = r;
            }
            r
        };
        let mut rep = GenReport {
            prompt_tokens: prompt_ids.len(),
            method: "scripted".into(),
            reused_tokens: reused as u64,
            prefix_common: common as u64,
            host_tier,
            ..GenReport::default()
        };
        for id in ids {
            // as an engine: cancellation is seen between launches
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                rep.stopped_by_cancel = true;
                return Ok(rep);
            }
            if rep.generated_tokens.len() as u64 >= params.max_new {
                return Ok(rep);
            }
            // as an engine: an end-of-sequence id of the request ends the generation
            if params.extra_eos_ids.contains(&id) {
                rep.generated_tokens.push(id);
                break;
            }
            rep.generated_tokens.push(id);
            on_token(id);
        }
        rep.stopped_by_eos = true;
        Ok(rep)
    }
}

struct Harness {
    state: Arc<AppState>,
    replies: Arc<Mutex<VecDeque<String>>>,
    prompts: Arc<Mutex<Vec<String>>>,
    images: SeenImages,
    tier: Arc<Mutex<ScriptedTier>>,
}

/// A second model family for the tests: reasoning and tool calls between special tokens, the
/// call form of a `response_template`.
const CALL_FORM_TEMPLATE: &str = "{{- bos_token }}{% if tools %}<|turn>system\n{% for t in tools %}<|tool>{{ t.function.name }}<tool|>{% endfor %}<turn|>\n{% endif %}\
{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}<|turn>model\n";
const CALL_FORM_CONFIG: &str = r#"{"bos_token": "<bos>", "response_template": {"fields": {
    "thinking": {"open": "<|channel>thought\n", "close": "<channel|>"},
    "tool_calls": {"open_pattern": "<\\|tool_call>call:(?P<name>\\w+)", "close": "<tool_call|>", "content": "json",
                   "content_args": {"string_delims": [["<|\"|>", "<|\"|>"]]}}}}}"#;

fn harness() -> Harness {
    harness_with_cache(None)
}

/// The harness with a host-RAM tier store in the server (`--cache-ram`), or without one.
fn harness_with_cache(host_cache: Option<Arc<crate::cache_ram::SharedCache>>) -> Harness {
    let spec = ModelSpec::from_config_json(
        r#"{"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text",
            "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320,
            "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
            "head_dim": 256, "full_attention_interval": 4,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_conv_kernel_dim": 4}}"#,
    )
    .unwrap();
    let tok = Arc::new(crate::tokenizer::test_tokenizer(&[
        ("<|im_start|>", true),
        ("<|im_end|>", true),
        ("<bos>", true),
        ("<|turn>", true),
        ("<turn|>", true),
        ("<|channel>", true),
        ("<channel|>", true),
        ("<|tool_call>", true),
        ("<tool_call|>", true),
        ("<|\"|>", true),
        ("<|tool_response>", true),
        ("<|tool>", true),
        ("<tool|>", true),
        ("<|vision_start|>", true),
        ("<|vision_end|>", true),
        ("<|image_pad|>", true),
        ("<|video_pad|>", true),
    ]));
    let model = |id: &str, template: &str, config: Option<&str>| {
        let config = config.map(|c| json::parse_json(c).unwrap());
        let (chat, note) = ChatTemplate::from_parts(template, config.as_ref()).unwrap();
        assert!(note.is_none(), "{note:?}");
        RegisteredModel {
            id: id.into(),
            dir: PathBuf::from("/nonexistent"),
            spec: spec.clone(),
            format: chat.format.clone(),
            chat: Some(chat),
            tokenizer: Arc::clone(&tok),
            extra_eos_ids: Vec::new(),
            max_ctx: 16384,
            gemma_closes_thought: false,
            gen_defaults: SamplerDefaults::FALLBACK,
            vision: None,
        }
    };
    let replies = Arc::new(Mutex::new(VecDeque::new()));
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let images = Arc::new(Mutex::new(Vec::new()));
    let tier = Arc::new(Mutex::new(ScriptedTier::default()));
    let engine = ScriptedEngine {
        spec: spec.clone(),
        tok: Arc::clone(&tok),
        replies: Arc::clone(&replies),
        prompts: Arc::clone(&prompts),
        pending_images: Vec::new(),
        images: Arc::clone(&images),
        held: Vec::new(),
        prefix: crate::engine::qwen3::prefix_cache::PrefixStats {
            slots: 8,
            slot_bytes: 1024,
            ..Default::default()
        },
        tier: Arc::clone(&tier),
    };
    let state = Arc::new(AppState {
        cfg: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            models: Vec::new(),
            kernels_dir: PathBuf::new(),
            ui_dir: None,
            max_ctx: 16384,
            cache_dir: None,
            no_cache: true,
            lazy: true,
            idle_park_secs: 0,
            queue_config: None,
            api_key: None,
            a2ui: true,
            a2ui_prompt: None,
            verbose: false,
            per_model: Vec::new(),
            draft_k: Vec::new(),
            vision: true,
            vision_max_patches: DEFAULT_VISION_MAX_PATCHES,
            vision_max_tokens: DEFAULT_VISION_MAX_TOKENS,
            prefix_slots: 0,
            mcp_proxy: ProxyOptions::default(),
            tls_cert: None,
            tls_key: None,
            cache_ram_bytes: Some(0),
        },
        models: vec![model("tags", chat::TEST_TEMPLATE, None)],
        slot: Mutex::new(EngineSlot {
            engine: Some(Box::new(engine)),
            resident_idx: Some(0),
            lifecycle: Lifecycle::Ready(0),
            detail: String::new(),
            since: Instant::now(),
            last_activity: Instant::now(),
            admin_parked: false,
            hold: false,
            active_requests: 0,
            waiting: vec![0],
            switches: 0,
            last_switch_s: None,
            last_warmup_s: None,
            last_live_json: None,
            last_request_json: None,
            hbm_bytes: None,
            facts: None,
            prefix: None,
            host_tier: None,
            idle_due: false,
        }),
        card_op: Mutex::new(()),
        cancels: Mutex::new(HashMap::new()),
        next_req_id: AtomicU64::new(1),
        hub: EventHub::new(),
        dtypes: Mutex::new(HashMap::new()),
        turns: TurnQueue::default(),
        proxy: proxy::McpProxy::disabled(),
        tls: None,
        host_cache,
    });
    let _ = model; // (one resident model per harness)
    Harness {
        state,
        replies,
        prompts,
        images,
        tier,
    }
}

/// The same harness with the call-form family as its model.
fn call_form_harness() -> Harness {
    let h = harness();
    let m = &h.state.models[0];
    let config = json::parse_json(CALL_FORM_CONFIG).unwrap();
    let (chat, note) = ChatTemplate::from_parts(CALL_FORM_TEMPLATE, Some(&config)).unwrap();
    assert!(note.is_none(), "{note:?}");
    let model = RegisteredModel {
        id: "callform".into(),
        dir: m.dir.clone(),
        spec: m.spec.clone(),
        format: chat.format.clone(),
        chat: Some(chat),
        tokenizer: Arc::clone(&m.tokenizer),
        extra_eos_ids: vec![m.tokenizer.token_id("<|tool_response>").unwrap()],
        max_ctx: m.max_ctx,
        gemma_closes_thought: false,
        gen_defaults: m.gen_defaults,
        vision: None,
    };
    let engine = h.state.slot.lock().unwrap().engine.take();
    let state = Arc::new(AppState {
        cfg: h.state.cfg.clone(),
        models: vec![model],
        slot: Mutex::new(EngineSlot {
            engine,
            resident_idx: Some(0),
            lifecycle: Lifecycle::Ready(0),
            detail: String::new(),
            since: Instant::now(),
            last_activity: Instant::now(),
            admin_parked: false,
            hold: false,
            active_requests: 0,
            waiting: vec![0],
            switches: 0,
            last_switch_s: None,
            last_warmup_s: None,
            last_live_json: None,
            last_request_json: None,
            hbm_bytes: None,
            facts: None,
            prefix: None,
            host_tier: None,
            idle_due: false,
        }),
        card_op: Mutex::new(()),
        cancels: Mutex::new(HashMap::new()),
        next_req_id: AtomicU64::new(1),
        hub: EventHub::new(),
        dtypes: Mutex::new(HashMap::new()),
        turns: TurnQueue::default(),
        proxy: proxy::McpProxy::disabled(),
        tls: None,
        host_cache: None,
    });
    Harness {
        state,
        replies: h.replies,
        prompts: h.prompts,
        images: h.images,
        tier: h.tier,
    }
}

/// A chat template with the content macro of a vision-language family: a content list writes
/// one placeholder block per image / video part, numbered when `add_vision_id` is set.
const VISION_TEMPLATE: &str = r#"{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- set image_count.value = image_count.value + 1 %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- set video_count.value = video_count.value + 1 %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{%- endmacro %}
{%- for m in messages %}
    {%- if m.role == 'tool' %}
        {{- '<|im_start|>user\n<tool_response>\n' ~ render_content(m.content) ~ '\n</tool_response><|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>' ~ m.role ~ '\n' ~ render_content(m.content, m.role == 'system') ~ '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

/// The harness with a vision-language model: the template above, and a tower planned for
/// images of `max_patches` patches and `max_tokens` image tokens per request.
fn vision_harness(max_patches: u64, max_tokens: u64) -> Harness {
    let h = harness();
    let m = &h.state.models[0];
    let (chat, _) = ChatTemplate::from_parts(VISION_TEMPLATE, None).unwrap();
    let mut vs = crate::vision::test_util::qwen38_vision_spec();
    let id = |t: &str| m.tokenizer.token_id(t).unwrap();
    vs.image_token_id = id("<|image_pad|>");
    vs.video_token_id = id("<|video_pad|>");
    vs.vision_start_id = id("<|vision_start|>");
    vs.vision_end_id = id("<|vision_end|>");
    let model = RegisteredModel {
        id: "vl".into(),
        dir: m.dir.clone(),
        spec: m.spec.clone(),
        format: chat.format.clone(),
        chat: Some(chat),
        tokenizer: Arc::clone(&m.tokenizer),
        extra_eos_ids: Vec::new(),
        max_ctx: m.max_ctx,
        gemma_closes_thought: false,
        gen_defaults: m.gen_defaults,
        vision: Some(
            VisionServe::new(vs, max_patches, max_tokens, MediaLimits::default()).unwrap(),
        ),
    };
    let engine = h.state.slot.lock().unwrap().engine.take();
    let state = Arc::new(AppState {
        cfg: h.state.cfg.clone(),
        models: vec![model],
        slot: Mutex::new(EngineSlot {
            engine,
            resident_idx: Some(0),
            lifecycle: Lifecycle::Ready(0),
            detail: String::new(),
            since: Instant::now(),
            last_activity: Instant::now(),
            admin_parked: false,
            hold: false,
            active_requests: 0,
            waiting: vec![0],
            switches: 0,
            last_switch_s: None,
            last_warmup_s: None,
            last_live_json: None,
            last_request_json: None,
            hbm_bytes: None,
            facts: None,
            prefix: None,
            host_tier: None,
            idle_due: false,
        }),
        card_op: Mutex::new(()),
        cancels: Mutex::new(HashMap::new()),
        next_req_id: AtomicU64::new(1),
        hub: EventHub::new(),
        dtypes: Mutex::new(HashMap::new()),
        turns: TurnQueue::default(),
        proxy: proxy::McpProxy::disabled(),
        tls: None,
        host_cache: None,
    });
    Harness {
        state,
        replies: h.replies,
        prompts: h.prompts,
        images: h.images,
        tier: h.tier,
    }
}

/// A `data:` URL of a synthetic image (binary PPM, which needs no codec).
fn image_url(w: usize, h: usize, k: usize) -> String {
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut f = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            f.extend([(x * 255 / w) as u8, (y * 255 / h) as u8, (k * 40) as u8]);
        }
    }
    let mut out = String::from("data:image/x-portable-pixmap;base64,");
    for c in f.chunks(3) {
        let v = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(B64[(v >> 18) as usize] as char);
        out.push(B64[((v >> 12) & 63) as usize] as char);
        out.push(if c.len() > 1 {
            B64[((v >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            B64[(v & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

impl Harness {
    fn script(&self, reply: &str) {
        self.replies.lock().unwrap().push_back(reply.to_string());
    }

    fn last_prompt(&self) -> String {
        self.prompts
            .lock()
            .unwrap()
            .last()
            .cloned()
            .unwrap_or_default()
    }

    /// One HTTP exchange with the server's connection handler: `(status, body)`, the body with
    /// its transfer chunks joined.
    fn post(&self, path: &str, body: &str) -> (u16, String) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::clone(&self.state);
        let server = thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            handle_connection(state, s);
        });
        let mut c = TcpStream::connect(addr).unwrap();
        write!(
            c,
            "POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut raw = Vec::new();
        c.read_to_end(&mut raw).unwrap();
        server.join().unwrap();
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("a response head");
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut rest = &raw[split + 4..];
        if !head.contains("Transfer-Encoding: chunked") {
            return (status, String::from_utf8_lossy(rest).into_owned());
        }
        let mut out = Vec::new();
        loop {
            let eol = rest
                .windows(2)
                .position(|w| w == b"\r\n")
                .expect("a chunk size line");
            let n = usize::from_str_radix(std::str::from_utf8(&rest[..eol]).unwrap().trim(), 16)
                .unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&rest[eol + 2..eol + 2 + n]);
            rest = &rest[eol + 2 + n + 2..];
        }
        (status, String::from_utf8(out).unwrap())
    }

    fn chat(&self, extra: &str) -> (u16, JsonValue) {
        let body = format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"Weather in Oslo?\"}}]{extra}}}"
        );
        let (status, text) = self.post("/v1/chat/completions", &body);
        (
            status,
            json::parse_json(&text).unwrap_or_else(|e| panic!("{e}: {text}")),
        )
    }
}

fn s<'a>(v: &'a JsonValue, path: &[&str]) -> &'a str {
    at(v, path)
        .as_str()
        .unwrap_or_else(|| panic!("{path:?} is not a string in {}", v.dump()))
}

fn at<'a>(v: &'a JsonValue, path: &[&str]) -> &'a JsonValue {
    path.iter().fold(v, |cur, key| match key.parse::<usize>() {
        Ok(i) => &cur
            .as_array()
            .unwrap_or_else(|| panic!("{key} of {}", cur.dump()))[i],
        Err(_) => cur
            .get(key)
            .unwrap_or_else(|| panic!("no '{key}' in {}", cur.dump())),
    })
}

/// A chat-completion stream put back together.
#[derive(Default)]
struct ChatStream {
    reasoning: String,
    content: String,
    /// `(id, name, arguments)` per call index.
    calls: Vec<(String, String, String)>,
    finish: String,
    done: bool,
}

fn collect_chat_stream(body: &str) -> ChatStream {
    let mut st = ChatStream::default();
    for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            st.done = true;
            continue;
        }
        let chunk = json::parse_json(line).unwrap();
        let choice = at(&chunk, &["choices", "0"]);
        if let Some(f) = choice.get("finish_reason").and_then(JsonValue::as_str) {
            st.finish = f.to_string();
        }
        let delta = at(choice, &["delta"]);
        let text = |key: &str| delta.get(key).and_then(JsonValue::as_str).unwrap_or("");
        st.reasoning.push_str(text("reasoning_content"));
        st.content.push_str(text("content"));
        for tc in delta
            .get("tool_calls")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
        {
            let i = tc.get("index").and_then(JsonValue::as_u64).unwrap() as usize;
            if let Some(id) = tc.get("id").and_then(JsonValue::as_str) {
                // a call's first fragment: the next index, with its type and name
                assert_eq!(i, st.calls.len(), "a header for a new index");
                assert_eq!(s(tc, &["type"]), "function");
                let name = s(tc, &["function", "name"]).to_string();
                st.calls.push((id.to_string(), name, String::new()));
            }
            assert!(i < st.calls.len(), "a fragment before its header");
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(JsonValue::as_str)
                .unwrap_or("");
            st.calls[i].2.push_str(args);
        }
    }
    st
}

const CALL_REPLY: &str = "I should look it up.\n</think>\n\nLet me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo \"S\"\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>";

#[test]
fn chat_completion_returns_tool_calls() {
    let h = harness();
    h.script(CALL_REPLY);
    let (status, r) = h.chat(&format!(",{TOOLS}"));
    assert_eq!(status, 200, "{}", r.dump());
    // the tools reached the model through its template
    assert!(h
        .last_prompt()
        .contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\""));
    assert!(h
        .last_prompt()
        .ends_with("<|im_start|>assistant\n<think>\n"));
    let msg = at(&r, &["choices", "0", "message"]);
    assert_eq!(s(msg, &["reasoning_content"]), "I should look it up.");
    assert_eq!(s(msg, &["content"]), "Let me check.");
    let calls = at(msg, &["tool_calls"]).as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(s(&calls[0], &["type"]), "function");
    assert!(s(&calls[0], &["id"]).starts_with("call_"));
    assert_eq!(s(&calls[0], &["function", "name"]), "get_weather");
    assert_eq!(
        s(&calls[0], &["function", "arguments"]),
        "{\"city\":\"Oslo \\\"S\\\"\",\"days\":2}"
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "tool_calls");
    assert_eq!(at(&r, &["gaudium", "tool_calls"]).as_u64(), Some(1));

    // a reply that is only a call has no content; without tools the same text is plain content
    h.script("</think>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>");
    let (_, r) = h.chat(&format!(",{TOOLS}"));
    assert_eq!(
        at(&r, &["choices", "0", "message", "content"]),
        &JsonValue::Null
    );
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "arguments"
            ]
        ),
        "{}"
    );
    h.script("</think>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>");
    let (_, r) = h.chat("");
    assert!(s(&r, &["choices", "0", "message", "content"]).starts_with("<tool_call>"));
    assert!(at(&r, &["choices", "0", "message"])
        .get("tool_calls")
        .is_none());
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "stop");
}

#[test]
fn chat_completion_streams_tool_call_fragments() {
    let h = harness();
    h.script(&format!(
        "{CALL_REPLY}\n<tool_call>\n<function=now>\n</function>\n</tool_call>"
    ));
    let (status, body) = h.post(
        "/v1/chat/completions",
        &format!("{{\"stream\":true,\"messages\":[{{\"role\":\"user\",\"content\":\"Weather?\"}}],{TOOLS}}}"),
    );
    assert_eq!(status, 200);
    let st = collect_chat_stream(&body);
    assert_eq!(st.reasoning, "I should look it up.");
    assert_eq!(st.content, "Let me check.");
    assert_eq!(st.calls.len(), 2, "{:?}", st.calls);
    assert_eq!(
        (st.calls[0].1.as_str(), st.calls[0].2.as_str()),
        ("get_weather", "{\"city\":\"Oslo \\\"S\\\"\",\"days\":2}")
    );
    assert_eq!(
        (st.calls[1].1.as_str(), st.calls[1].2.as_str()),
        ("now", "{}")
    );
    assert_ne!(st.calls[0].0, st.calls[1].0);
    assert_eq!(st.finish, "tool_calls");
    assert!(st.done);
    // a string argument streams before the call is complete: more than one fragment chunk
    let fragments = body.matches("\"tool_calls\":[").count();
    assert!(
        fragments > 3,
        "{fragments} chunks carried tool-call fragments"
    );
}

#[test]
fn tool_results_go_back_through_the_template() {
    let h = harness();
    h.script("</think>\n\nIt is sunny in Oslo.");
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":"Weather in Oslo?"}},
            {{"role":"assistant","content":"","tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"get_weather","arguments":"{{\"city\":\"Oslo\",\"days\":2}}"}}}}]}},
            {{"role":"tool","tool_call_id":"call_1","content":"sunny, 4 C"}}],{TOOLS}}}"#
    );
    let (status, text) = h.post("/v1/chat/completions", &body);
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(
        s(&r, &["choices", "0", "message", "content"]),
        "It is sunny in Oslo."
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "stop");
    assert!(h.last_prompt().contains(
        "<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\nsunny, 4 C\n</tool_response><|im_end|>\n"
    ));
}

#[test]
fn tool_choice_parallel_calls_and_stop_strings() {
    let h = harness();
    // a named choice: the call's opening ends the prompt, the model writes the rest
    h.script("<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>");
    let (_, r) = h.chat(&format!(",{TOOLS},\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}"));
    assert!(h
        .last_prompt()
        .ends_with("<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n"));
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "name"
            ]
        ),
        "get_weather"
    );
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "arguments"
            ]
        ),
        "{\"city\":\"Oslo\"}"
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "tool_calls");
    // required: the model names the function
    h.script("<function=now>\n</function>\n</tool_call>");
    let (_, r) = h.chat(&format!(",{TOOLS},\"tool_choice\":\"required\""));
    assert!(h.last_prompt().ends_with("</think>\n\n<tool_call>\n"));
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "name"
            ]
        ),
        "now"
    );
    // none: the tools are not shown
    h.script("</think>\n\nNo tools here.");
    let (_, r) = h.chat(&format!(",{TOOLS},\"tool_choice\":\"none\""));
    assert!(!h.last_prompt().contains("<tools>"));
    assert_eq!(
        s(&r, &["choices", "0", "message", "content"]),
        "No tools here."
    );
    // refused before the card is asked
    let before = h.prompts.lock().unwrap().len();
    let (status, r) = h.chat(",\"tool_choice\":\"required\"");
    assert_eq!(status, 400);
    assert!(s(&r, &["error", "message"]).contains("non-empty 'tools'"));
    assert_eq!(h.prompts.lock().unwrap().len(), before);

    // parallel calls off: the first call ends the generation
    h.script("</think>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call> and more text");
    let (_, r) = h.chat(&format!(",{TOOLS},\"parallel_tool_calls\":false"));
    assert_eq!(
        at(&r, &["choices", "0", "message", "tool_calls"])
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "tool_calls");
    assert!(
        (at(&r, &["usage", "completion_tokens"]).as_u64().unwrap() as usize) < 80,
        "the generation was cut"
    );

    // a stop string ends the reply and is not part of it
    h.script("</think>\n\none two STOP three");
    let (_, r) = h.chat(",\"stop\":[\"STOP\"]");
    assert_eq!(s(&r, &["choices", "0", "message", "content"]), "one two ");
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "stop");
    // the token limit
    h.script("</think>\n\nabcdefghijklmnopqrstuvwxyz");
    let (_, r) = h.chat(",\"max_tokens\":16");
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "length");
}

#[test]
fn special_token_tags_and_the_call_form() {
    let h = call_form_harness();
    h.script("<|channel>thought\nNeed the tool.<channel|>On it.<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,days:<|\"|>2<|\"|>}<tool_call|><|tool_response>never shown");
    let (status, r) = h.chat(&format!(",{TOOLS}"));
    assert_eq!(status, 200, "{}", r.dump());
    assert!(h
        .last_prompt()
        .starts_with("<bos><|turn>system\n<|tool>get_weather<tool|>"));
    let msg = at(&r, &["choices", "0", "message"]);
    assert_eq!(s(msg, &["reasoning_content"]), "Need the tool.");
    assert_eq!(s(msg, &["content"]), "On it.");
    assert_eq!(
        s(msg, &["tool_calls", "0", "function", "name"]),
        "get_weather"
    );
    // the schema types `days` as an integer
    assert_eq!(
        s(msg, &["tool_calls", "0", "function", "arguments"]),
        "{\"city\":\"Oslo\",\"days\":2}"
    );
    assert_eq!(s(&r, &["choices", "0", "finish_reason"]), "tool_calls");
    // a forced call in this form
    h.script("city:<|\"|>Rome<|\"|>}<tool_call|>");
    let (_, r) = h.chat(&format!(",{TOOLS},\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}"));
    assert!(h
        .last_prompt()
        .ends_with("<|turn>model\n<|tool_call>call:get_weather{"));
    assert_eq!(
        s(
            &r,
            &[
                "choices",
                "0",
                "message",
                "tool_calls",
                "0",
                "function",
                "arguments"
            ]
        ),
        "{\"city\":\"Rome\"}"
    );
}

const MESSAGES_TOOLS: &str = r#""tools":[{"name":"get_weather","description":"Weather.","input_schema":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}]"#;

#[test]
fn messages_endpoint_returns_content_blocks() {
    let h = harness();
    h.script(CALL_REPLY);
    let (status, text) = h.post(
        "/v1/messages",
        &format!("{{\"model\":\"any\",\"max_tokens\":512,\"system\":\"Be brief.\",{MESSAGES_TOOLS},\"messages\":[{{\"role\":\"user\",\"content\":\"Weather in Oslo?\"}}]}}"),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(
        (s(&r, &["type"]), s(&r, &["role"]), s(&r, &["stop_reason"])),
        ("message", "assistant", "tool_use")
    );
    assert!(s(&r, &["id"]).starts_with("msg_"));
    let blocks = at(&r, &["content"]).as_array().unwrap();
    let kinds: Vec<&str> = blocks.iter().map(|b| s(b, &["type"])).collect();
    assert_eq!(kinds, ["thinking", "text", "tool_use"]);
    assert_eq!(s(&blocks[0], &["thinking"]), "I should look it up.");
    assert_eq!(
        anthropic::signature_reasoning(s(&blocks[0], &["signature"])).as_deref(),
        Some("I should look it up.")
    );
    assert_eq!(s(&blocks[1], &["text"]), "Let me check.");
    assert!(s(&blocks[2], &["id"]).starts_with("toolu_"));
    assert_eq!(
        at(&blocks[2], &["input"]).dump(),
        "{\"city\":\"Oslo \\\"S\\\"\",\"days\":2}"
    );
    assert!(at(&r, &["usage", "input_tokens"]).as_u64().unwrap() > 0);
    assert!(at(&r, &["usage", "output_tokens"]).as_u64().unwrap() > 0);
    assert!(h
        .last_prompt()
        .contains("<|im_start|>system\nBe brief.<|im_end|>"));

    // the tool result goes back; a stop sequence ends the next reply
    h.script("</think>\n\nSunny. END of story");
    let (status, text) = h.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":512,"stop_sequences":["END"],{MESSAGES_TOOLS},"messages":[
                {{"role":"user","content":"Weather in Oslo?"}},
                {{"role":"assistant","content":[{{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{{"city":"Oslo"}}}}]}},
                {{"role":"user","content":[{{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}}]}}]}}"#
        ),
    );
    assert_eq!(status, 200, "{text}");
    let r = json::parse_json(&text).unwrap();
    assert_eq!(
        (s(&r, &["stop_reason"]), s(&r, &["stop_sequence"])),
        ("stop_sequence", "END")
    );
    assert_eq!(s(&r, &["content", "0", "text"]), "Sunny. ");
    assert!(h
        .last_prompt()
        .contains("<tool_response>\nsunny\n</tool_response>"));

    // errors in this API's shape
    let (status, text) = h.post(
        "/v1/messages",
        r#"{"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"x"}]}"#,
    );
    let r = json::parse_json(&text).unwrap();
    assert_eq!(
        (status, s(&r, &["type"]), s(&r, &["error", "type"])),
        (400, "error", "invalid_request_error")
    );
}

#[test]
fn messages_endpoint_streams_blocks_that_rebuild_the_message() {
    let h = harness();
    h.script(CALL_REPLY);
    let (status, body) = h.post(
        "/v1/messages",
        &format!("{{\"stream\":true,\"max_tokens\":512,{MESSAGES_TOOLS},\"messages\":[{{\"role\":\"user\",\"content\":\"Weather?\"}}]}}"),
    );
    assert_eq!(status, 200);
    let events: Vec<(String, JsonValue)> = body
        .split("\n\n")
        .filter(|e| !e.is_empty())
        .map(|e| {
            let (name, data) = e.split_once('\n').unwrap();
            (
                name.strip_prefix("event: ").unwrap().to_string(),
                json::parse_json(data.strip_prefix("data: ").unwrap()).unwrap(),
            )
        })
        .collect();
    assert_eq!(events.first().unwrap().0, "message_start");
    assert_eq!(events.last().unwrap().0, "message_stop");
    let count = |name: &str| events.iter().filter(|(n, _)| n == name).count();
    assert_eq!(
        (count("content_block_start"), count("content_block_stop")),
        (3, 3)
    );
    // each block's deltas rebuild its text
    let joined = |index: u64, kind: &str, field: &str| -> String {
        events
            .iter()
            .filter(|(n, d)| {
                n == "content_block_delta"
                    && d.get("index").and_then(JsonValue::as_u64) == Some(index)
            })
            .filter_map(|(_, d)| {
                d.get("delta")
                    .filter(|x| x.get("type").and_then(JsonValue::as_str) == Some(kind))
            })
            .map(|x| s(x, &[field]).to_string())
            .collect()
    };
    assert_eq!(
        joined(0, "thinking_delta", "thinking"),
        "I should look it up."
    );
    assert_eq!(joined(1, "text_delta", "text"), "Let me check.");
    assert_eq!(
        joined(2, "input_json_delta", "partial_json"),
        "{\"city\":\"Oslo \\\"S\\\"\",\"days\":2}"
    );
    let start = &events
        .iter()
        .filter(|(n, _)| n == "content_block_start")
        .nth(2)
        .unwrap()
        .1;
    assert_eq!(s(start, &["content_block", "name"]), "get_weather");
    let delta = &events.iter().find(|(n, _)| n == "message_delta").unwrap().1;
    assert_eq!(s(delta, &["delta", "stop_reason"]), "tool_use");
    assert!(at(delta, &["usage", "output_tokens"]).as_u64().unwrap() > 0);
}

#[test]
fn template_and_token_count_endpoints_and_props() {
    let h = harness();
    let body = format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],{TOOLS}}}");
    let (status, text) = h.post("/apply-template", &body);
    assert_eq!(status, 200);
    let prompt = json::parse_json(&text).unwrap();
    assert!(
        s(&prompt, &["prompt"]).contains("<tools>")
            && s(&prompt, &["prompt"]).ends_with("<think>\n")
    );
    let (_, text) = h.post("/gaudium/count_tokens", &body);
    let with_tools = at(&json::parse_json(&text).unwrap(), &["n_prompt_tokens"])
        .as_u64()
        .unwrap();
    let (_, text) = h.post(
        "/gaudium/count_tokens",
        r#"{"messages":[{"role":"user","content":"Hi"}]}"#,
    );
    let without = at(&json::parse_json(&text).unwrap(), &["n_prompt_tokens"])
        .as_u64()
        .unwrap();
    assert!(
        with_tools > without + 100,
        "{with_tools} vs {without}: the tools count"
    );
    let (status, text) = h.post(
        "/v1/messages/count_tokens",
        &format!("{{{MESSAGES_TOOLS},\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}"),
    );
    assert_eq!(status, 200, "{text}");
    assert!(
        at(&json::parse_json(&text).unwrap(), &["input_tokens"])
            .as_u64()
            .unwrap()
            > without
    );

    let props = json::parse_json(&h.state.props_json(0, true)).unwrap();
    let caps = at(&props, &["chat_template_caps"]);
    for key in [
        "supports_tools",
        "supports_tool_calls",
        "supports_parallel_tool_calls",
    ] {
        assert_eq!(
            caps.get(key).and_then(JsonValue::as_bool),
            Some(true),
            "{key}"
        );
    }
}

#[test]
fn a2ui_declaration_adds_the_instructions_to_the_system_turn() {
    let h = harness();
    // a request that declares nothing renders as before
    h.script("Hello.");
    let (status, _) = h.chat("");
    assert_eq!(status, 200);
    assert!(!h.last_prompt().contains("a2ui"));

    // a declared renderer: the instructions open the conversation as a system turn
    h.script("A form helps.\n</think>\n\nHere.\n<a2ui-json>\n[]\n</a2ui-json>");
    let (status, reply) = h.chat(",\"gaudium_a2ui\":true");
    assert_eq!(status, 200, "{reply:?}");
    let prompt = h.last_prompt();
    assert!(
        prompt.starts_with("<|im_start|>system\n# Interactive interfaces (A2UI 1.0)"),
        "{prompt}"
    );
    assert!(prompt.contains("<|im_end|>\n<|im_start|>user\nWeather in Oslo?"));
    // the model's block reaches the client inside the ordinary content
    assert!(s(&reply, &["choices", "0", "message", "content"]).contains("<a2ui-json>"));

    // the same prompt is what the template and token-count endpoints see
    let body = r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],"gaudium_a2ui":{"1.0":{"supportedCatalogIds":["https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"]}}}"#;
    let (status, text) = h.post("/apply-template", body);
    assert_eq!(status, 200, "{text}");
    let rendered = json::parse_json(&text).unwrap();
    let rendered = s(&rendered, &["prompt"]);
    assert!(rendered.starts_with("<|im_start|>system\nBe brief.\n\n# Interactive interfaces"));
    assert_eq!(rendered.matches("<|im_start|>system").count(), 1);
    let (_, text) = h.post("/gaudium/count_tokens", body);
    let with = at(&json::parse_json(&text).unwrap(), &["n_prompt_tokens"])
        .as_u64()
        .unwrap();
    let (_, text) = h.post(
        "/gaudium/count_tokens",
        r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]}"#,
    );
    let without = at(&json::parse_json(&text).unwrap(), &["n_prompt_tokens"])
        .as_u64()
        .unwrap();
    assert!(with > without + 500, "{with} vs {without}");

    // a malformed declaration is refused; the Messages endpoint takes the field too
    let (status, _) = h.chat(",\"gaudium_a2ui\":\"yes\"");
    assert_eq!(status, 400);
    h.script("Hello.");
    let (status, text) = h.post(
        "/v1/messages",
        r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],"gaudium_a2ui":true}"#,
    );
    assert_eq!(status, 200, "{text}");
    assert!(h
        .last_prompt()
        .contains("# Interactive interfaces (A2UI 1.0)"));

    let props = json::parse_json(&h.state.props_json(0, true)).unwrap();
    assert_eq!(at(&props, &["a2ui", "enabled"]).as_bool(), Some(true));
    assert_eq!(s(&props, &["a2ui", "request_field"]), "gaudium_a2ui");
    assert!(at(&props, &["a2ui"]).get("instructions").is_none());
}

/// A model load and the off-card recipe build take one engine configuration.
#[test]
fn served_engine_configuration_follows_the_server_options() {
    let mut opts = ModelRunOptions::default();
    opts.draft_model = Some(PathBuf::from("/models/Target-DSpine-b16"));
    opts.precision = Some(crate::engine::fp8_graph::WeightPrecision::Fp8);
    opts.max_ctx = Some(131_072);
    let mut own = ModelRunOptions::default();
    own.own_mtp = true;
    let cfg = ServerConfig {
        host: "127.0.0.1".into(),
        port: 0,
        models: Vec::new(),
        kernels_dir: PathBuf::from("/kernels"),
        ui_dir: None,
        max_ctx: 262_144,
        cache_dir: Some(PathBuf::from("/cache")),
        no_cache: false,
        lazy: true,
        idle_park_secs: 0,
        queue_config: None,
        api_key: None,
        a2ui: false,
        a2ui_prompt: None,
        verbose: false,
        per_model: vec![("a".into(), opts), ("moe".into(), own)],
        draft_k: Vec::new(),
        vision: true,
        vision_max_patches: DEFAULT_VISION_MAX_PATCHES,
        vision_max_tokens: DEFAULT_VISION_MAX_TOKENS,
        prefix_slots: 5,
        mcp_proxy: ProxyOptions::default(),
        tls_cert: None,
        tls_key: None,
        cache_ram_bytes: Some(0),
    };
    // a DSpine drafter: the adaptive pair of widths, the model's precision and cache
    let (q, drafter) = engine_config_of(&cfg, "a", Path::new("/models/Target"), 131_072);
    // the kept positions the server was started with
    assert_eq!(q.prefix_slots, 5);
    assert_eq!(
        drafter.as_deref(),
        Some(Path::new("/models/Target-DSpine-b16"))
    );
    assert_eq!(q.max_ctx, 131_072);
    assert_eq!(q.cache_dir.as_deref(), Some(Path::new("/cache")));
    assert_eq!(q.precision, crate::engine::fp8_graph::WeightPrecision::Fp8);
    let spec = q.spec.expect("a drafter is planned");
    assert_eq!(spec.draft_k, vec![7, 15]);
    // the model's own MTP head: no checkpoint, four drafts
    let (q, drafter) = engine_config_of(&cfg, "moe", Path::new("/models/Moe"), 262_144);
    assert!(drafter.is_none());
    assert_eq!(q.spec.expect("own MTP head").draft_k, vec![4]);
    // a model without options: no drafter, the server's kernels and cache
    let (q, drafter) = engine_config_of(&cfg, "plain", Path::new("/models/Plain"), 4096);
    assert!(drafter.is_none() && q.spec.is_none());
    assert_eq!(q.kernels_dir, PathBuf::from("/kernels"));
    // a configuration without models, or an id it does not serve, is refused before anything
    // is compiled
    assert!(build_cache(&cfg, &[]).is_err());
    assert!(build_cache(&cfg, &["nope".to_string()]).is_err());
}

/// The host-RAM tier as a client sees it: where a request's reused rows came from and what was
/// copied for it, the store and the engine's measured rates in `/props`, and the two moments
/// the server gives the engine for its saves (after a response, before a release).
/// A connection that ends while a chat request still holds the card must leave that request's
/// idle work pending: the request's own connection does it once the card is free.
#[test]
fn idle_work_stays_pending_until_a_connection_can_start_it() {
    use crate::engine::qwen3::host_tier::TierReport;
    let cache = Arc::new(crate::cache_ram::SharedCache::new(8 << 30, 256 << 20, false, false));
    let h = harness_with_cache(Some(Arc::clone(&cache)));
    h.tier.lock().unwrap().next.push_back(TierReport::default());
    h.script("First.");
    assert_eq!(h.chat("").0, 200);
    let idles = || h.tier.lock().unwrap().idle_waiting.len();
    let due = || h.state.slot.lock().unwrap().idle_due;
    assert_eq!((idles(), due()), (1, false));
    // A chat request is at its end: it has left idle work pending and still holds the card.
    h.state.slot.lock().unwrap().idle_due = true;
    let card = h.state.card_op.lock().unwrap();
    // Another connection (a token count: no card) comes and goes meanwhile. It cannot start
    // the work, and it must not mark it done.
    let (status, text) = h.post(
        "/gaudium/count_tokens",
        r#"{"messages":[{"role":"user","content":"Weather in Oslo?"}]}"#,
    );
    assert_eq!(status, 200, "{text}");
    assert_eq!((idles(), due()), (1, true));
    // The chat request lets go of the card and its connection ends: now the work is done, once.
    drop(card);
    h.state.idle_work();
    assert_eq!((idles(), due()), (2, false));
    // nothing is pending: a connection that ends starts nothing
    let (status, _) = h.post(
        "/gaudium/count_tokens",
        r#"{"messages":[{"role":"user","content":"Weather in Oslo?"}]}"#,
    );
    assert_eq!(status, 200);
    assert_eq!((idles(), due()), (2, false));
    // work that is pending while a request uses the engine waits for that request too
    {
        let mut s = h.state.slot.lock().unwrap();
        s.idle_due = true;
        s.active_requests = 1;
    }
    h.state.idle_work();
    assert_eq!((idles(), due()), (2, true));
    h.state.slot.lock().unwrap().active_requests = 0;
    h.state.idle_work();
    assert_eq!((idles(), due()), (3, false));
}

#[test]
fn the_host_tier_is_reported_and_given_its_idle_time() {
    use crate::engine::qwen3::host_tier::{Tier, TierReport};
    let cache = Arc::new(crate::cache_ram::SharedCache::new(8 << 30, 256 << 20, false, false));
    let h = harness_with_cache(Some(Arc::clone(&cache)));
    let u = |v: &JsonValue, path: &[&str]| at(v, path).as_u64().unwrap();
    // a first request: everything prefilled
    h.tier.lock().unwrap().next.push_back(TierReport::default());
    h.script("First.");
    let (status, v) = h.chat("");
    assert_eq!(status, 200, "{}", v.dump());
    assert_eq!(at(&v, &["gaudium", "prefix", "tier"]).as_str(), Some("prefilled"));
    assert_eq!(at(&v, &["gaudium", "prefix", "host", "restore_bytes"]).as_u64(), Some(0));
    // the engine got its idle time after the response, with no request waiting
    assert_eq!(h.tier.lock().unwrap().idle_waiting, vec![0]);
    // a second request whose rows came back from host memory, after it paid for a save
    h.tier.lock().unwrap().next.push_back(TierReport {
        tier: Tier::Host,
        restore_ms: 26.9,
        restore_bytes: 407_502_848,
        restore_rows: 1024,
        save_ms: 3.5,
        save_bytes: 1 << 20,
    });
    h.script("Second.");
    let (_, v) = h.chat("");
    assert!(u(&v, &["timings", "cache_n"]) > 0);
    assert_eq!(at(&v, &["gaudium", "prefix", "tier"]).as_str(), Some("host"));
    let host = at(&v, &["gaudium", "prefix", "host"]);
    assert_eq!(u(host, &["restore_bytes"]), 407_502_848);
    assert_eq!(u(host, &["restore_tokens"]), 1024);
    assert_eq!(u(host, &["save_bytes"]), 1 << 20);
    assert!((at(host, &["restore_ms"]).as_f64().unwrap() - 26.9).abs() < 1e-6);
    assert_eq!(h.tier.lock().unwrap().idle_waiting.len(), 2);
    // /props: the store, and the resident engine's rates and entry sizes
    let props = json::parse_json(&h.state.props_json(0, true)).unwrap();
    let cr = at(&props, &["cache_ram"]);
    assert_eq!(u(cr, &["budget_bytes"]), 8 << 30);
    assert_eq!((u(cr, &["held_bytes"]), u(cr, &["entries"])), (0, 0));
    assert_eq!(u(cr, &["engine", "bytes_per_token"]), 65536);
    assert_eq!(u(cr, &["engine", "to_device_bytes_per_s"]), 14 << 30);
    assert_eq!(u(cr, &["engine", "prefill_tokens_per_s"]), 7000);
    assert_eq!(at(cr, &["engine", "last", "tier"]).as_str(), Some("host"));
    // a request with `cache_prompt: false` on the Messages endpoint reaches the engine as one
    h.script("Third.");
    let (status, text) = h.post(
        "/v1/messages",
        r#"{"model":"m","max_tokens":16,"cache_prompt":false,"messages":[{"role":"user","content":"Weather in Oslo?"}]}"#,
    );
    assert_eq!(status, 200, "{text}");
    let m = json::parse_json(&text).unwrap();
    assert_eq!(u(&m, &["usage", "cache_read_input_tokens"]), 0);
    // the release of the card: the engine is told first
    assert_eq!(h.tier.lock().unwrap().releases, 0);
    h.state.park(false, "test").unwrap();
    assert_eq!(h.tier.lock().unwrap().releases, 1);
    // without a store the server gives no idle work and reports no cache
    let plain = harness();
    plain.script("Plain.");
    let (_, v) = plain.chat("");
    assert_eq!(at(&v, &["gaudium", "prefix", "tier"]).as_str(), Some("prefilled"));
    assert!(v.get("gaudium").and_then(|g| g.get("prefix")).and_then(|p| p.get("host")).is_none());
    assert!(plain.tier.lock().unwrap().idle_waiting.is_empty());
    let props = json::parse_json(&plain.state.props_json(0, true)).unwrap();
    assert!(matches!(at(&props, &["cache_ram"]), JsonValue::Null));
    // rows reused by an engine that names no tier came from the device
    plain.script("Again.");
    let (_, v) = plain.chat("");
    assert_eq!(at(&v, &["gaudium", "prefix", "tier"]).as_str(), Some("device"));
}

/// Prefix reuse as a client sees it: the rows an engine took over from the request before are
/// reported apart from the rows it prefilled, in the reply, in `/props` and in the Messages
/// usage, and `cache_prompt: false` reaches the engine.
#[test]
fn prefix_reuse_is_reported_to_the_client() {
    let h = harness();
    let u = |v: &JsonValue, path: &[&str]| at(v, path).as_u64().unwrap();
    // a first request has nothing to take over
    h.script("First.");
    let (status, v) = h.chat("");
    assert_eq!(status, 200, "{}", v.dump());
    let prompt = u(&v, &["usage", "prompt_tokens"]);
    assert_eq!(u(&v, &["timings", "cache_n"]), 0);
    assert_eq!(u(&v, &["timings", "prompt_n"]), prompt);
    assert_eq!(u(&v, &["usage", "prompt_tokens_details", "cached_tokens"]), 0);
    assert_eq!(at(&v, &["gaudium", "prefix", "hit"]).as_bool(), Some(false));
    // the same prompt again: all but its last rows are taken over
    h.script("Second.");
    let (_, v) = h.chat("");
    let cached = u(&v, &["timings", "cache_n"]);
    assert!(cached > 0 && cached < prompt, "{cached} of {prompt}");
    assert_eq!(cached + u(&v, &["timings", "prompt_n"]), prompt);
    assert_eq!(
        u(&v, &["usage", "prompt_tokens_details", "cached_tokens"]),
        cached
    );
    assert_eq!(u(&v, &["usage", "prompt_tokens"]), prompt);
    assert_eq!(at(&v, &["gaudium", "prefix", "hit"]).as_bool(), Some(true));
    assert_eq!(u(&v, &["gaudium", "prefix", "reused_tokens"]), cached);
    assert_eq!(u(&v, &["gaudium", "prefix", "common_tokens"]), prompt);
    // /props: what the engine keeps and what its requests took over
    let props = json::parse_json(&h.state.props_json(0, true)).unwrap();
    let pc = at(&props, &["prefix_cache"]);
    assert_eq!(u(pc, &["positions"]), 8);
    assert_eq!((u(pc, &["requests"]), u(pc, &["hits"])), (2, 1));
    assert_eq!(u(pc, &["reused_tokens"]), cached);
    assert_eq!(at(pc, &["last", "hit"]).as_bool(), Some(true));
    assert_eq!(u(pc, &["last", "reused_tokens"]), cached);
    assert_eq!(u(pc, &["last", "prompt_tokens"]), prompt);
    // a request that asks for a whole prefill gets one
    h.script("Third.");
    let (_, v) = h.chat(",\"cache_prompt\":false");
    assert_eq!(u(&v, &["timings", "cache_n"]), 0);
    assert_eq!(u(&v, &["timings", "prompt_n"]), prompt);
    assert_eq!(at(&v, &["gaudium", "prefix", "hit"]).as_bool(), Some(false));
    // the stream carries the split in its timings too
    h.script("Fourth.");
    let (_, text) = h.post(
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"Weather in Oslo?"}],"stream":true}"#,
    );
    let last = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .rfind(|l| l.contains("\"usage\""))
        .expect("a final chunk with usage");
    let fin = json::parse_json(last).unwrap();
    assert_eq!(u(&fin, &["timings", "cache_n"]), cached);
    assert_eq!(u(&fin, &["timings", "prompt_n"]), prompt - cached);
    // the Messages API: reused rows are cache reads, the rest is input
    h.script("Fifth.");
    let body = r#"{"model":"tags","max_tokens":64,"messages":[{"role":"user","content":"Weather in Oslo?"}]}"#;
    h.post("/v1/messages", body);
    h.script("Sixth.");
    let (status, text) = h.post("/v1/messages", body);
    assert_eq!(status, 200, "{text}");
    let m = json::parse_json(&text).unwrap();
    let read = u(&m, &["usage", "cache_read_input_tokens"]);
    assert!(read > 0);
    assert_eq!(
        read + u(&m, &["usage", "input_tokens"]),
        u(&m, &["timings", "cache_n"]) + u(&m, &["timings", "prompt_n"])
    );
}

/// An image in a chat request: decoded, prepared and handed to the engine; its pad run in the
/// prompt has the image's token count, and the reply's details say what the tower did.
#[test]
fn chat_image_reaches_the_engine_with_its_pad_run() {
    let h = vision_harness(4096, 4096);
    h.script("A gradient.");
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"{}"}}}},{{"type":"text","text":"What is this?"}}]}}]}}"#,
        image_url(256, 256, 1)
    );
    let (status, text) = h.post("/v1/chat/completions", &body);
    assert_eq!(status, 200, "{text}");
    let v = json::parse_json(&text).unwrap();
    assert_eq!(
        s(&v, &["choices", "0", "message", "content"]),
        "A gradient."
    );
    // 256 x 256 pixels: 16 x 16 patches, 64 tokens
    let prompt = h.last_prompt();
    assert_eq!(prompt.matches("<|image_pad|>").count(), 64);
    assert!(prompt.contains("<|vision_start|><|image_pad|>"));
    assert!(
        prompt.ends_with("<|vision_end|>What is this?<|im_end|>\n<|im_start|>assistant\n"),
        "{prompt}"
    );
    assert_eq!(
        h.images.lock().unwrap().last().unwrap(),
        &vec![(16, 16, 256 * 1536)]
    );
    let vis = at(&v, &["gaudium", "vision"]);
    assert_eq!(at(vis, &["images"]).as_u64(), Some(1));
    assert_eq!(at(vis, &["patches"]).as_u64(), Some(256));
    assert_eq!(at(vis, &["tokens"]).as_u64(), Some(64));
    assert_eq!(at(vis, &["tower_ms"]).as_f64(), Some(20.0));
    assert!(at(&v, &["usage", "prompt_tokens"]).as_u64().unwrap() > 64);
    // a text-only request afterwards hands the engine no image
    h.script("Hello.");
    let (status, _) = h.chat("");
    assert_eq!(status, 200);
    assert!(h.images.lock().unwrap().last().unwrap().is_empty());
}

/// Two images and a video in one request, in the order of their parts; the video is written
/// into the prompt as timed temporal patches.
#[test]
fn chat_video_frames_become_timed_patches_between_images() {
    let h = vision_harness(4096, 4096);
    h.script("ok");
    let frames: Vec<String> = (0..7)
        .map(|k| format!("\"{}\"", image_url(96, 64, k)))
        .collect();
    let body = format!(
        r#"{{"chat_template_kwargs":{{"add_vision_id":true}},"messages":[{{"role":"user","content":[
            {{"type":"image_url","image_url":{{"url":"{a}"}}}},
            {{"type":"video","video":{{"frames":[{f}],"fps":2}}}},
            {{"type":"text","text":"Compare."}},
            {{"type":"image","image":"{b}"}}]}}]}}"#,
        a = image_url(256, 256, 0),
        b = image_url(448, 320, 2),
        f = frames.join(","),
    );
    let (status, text) = h.post("/v1/chat/completions", &body.replace('\n', " "));
    assert_eq!(status, 200, "{text}");
    let prompt = h.last_prompt();
    let pads = "<|video_pad|>".repeat(6);
    let video: String = ["0.2", "1.2", "2.2", "3.0"]
        .iter()
        .map(|t| format!("<{t} seconds><|vision_start|>{pads}<|vision_end|>"))
        .collect();
    assert!(
        prompt.contains(&format!(
            "Video 1: {video}Compare.Picture 2: <|vision_start|>"
        )),
        "{prompt}"
    );
    assert!(prompt.contains("Picture 1: <|vision_start|><|image_pad|>"));
    assert_eq!(prompt.matches("<|image_pad|>").count(), 64 + 140);
    // the engine gets the inputs in prompt order: image, four temporal patches, image
    let got = h.images.lock().unwrap().last().unwrap().clone();
    let grids: Vec<(usize, usize)> = got.iter().map(|g| (g.0, g.1)).collect();
    assert_eq!(
        grids,
        vec![(16, 16), (4, 6), (4, 6), (4, 6), (4, 6), (20, 28)]
    );
    let v = json::parse_json(&text).unwrap();
    let vis = at(&v, &["gaudium", "vision"]);
    assert_eq!(
        ["images", "videos", "frames", "inputs", "tokens"].map(|k| at(vis, &[k]).as_u64().unwrap()),
        [2, 1, 7, 6, 64 + 24 + 140]
    );
}

/// The prompt size of a request with images comes from the files' headers, and is the size the
/// request then runs with.
#[test]
fn count_tokens_counts_image_tokens_without_running() {
    let h = vision_harness(4096, 4096);
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"{}"}}}},{{"type":"text","text":"Hi"}}]}}]}}"#,
        image_url(448, 320, 1)
    );
    let (status, text) = h.post("/gaudium/count_tokens", &body);
    assert_eq!(status, 200, "{text}");
    let c = json::parse_json(&text).unwrap();
    assert_eq!(at(&c, &["image_tokens"]).as_u64(), Some(140));
    assert_eq!(at(&c, &["images"]).as_u64(), Some(1));
    let counted = at(&c, &["n_prompt_tokens"]).as_u64().unwrap();
    assert!(c.get("tokens").is_none());
    h.script("ok");
    let (status, text) = h.post("/v1/chat/completions", &body);
    assert_eq!(status, 200, "{text}");
    let v = json::parse_json(&text).unwrap();
    assert_eq!(at(&v, &["usage", "prompt_tokens"]).as_u64(), Some(counted));
    assert!(at(&v, &["gaudium"]).get("tokens").is_none());
    // on request: the prompt's ids from the count, the generated ids from the chat
    let with = body.replacen(
        '{',
        r#"{"return_tokens":true,"gaudium_return_tokens":true,"#,
        1,
    );
    let (_, text) = h.post("/gaudium/count_tokens", &with);
    let c = json::parse_json(&text).unwrap();
    let ids = at(&c, &["tokens"]).as_array().unwrap();
    assert_eq!(ids.len() as u64, counted);
    let pad = h.state.models[0]
        .tokenizer
        .token_id("<|image_pad|>")
        .unwrap() as i64;
    assert_eq!(ids.iter().filter(|t| t.as_i64() == Some(pad)).count(), 140);
    h.script("ok");
    let (_, text) = h.post("/v1/chat/completions", &with);
    let v = json::parse_json(&text).unwrap();
    assert_eq!(at(&v, &["gaudium", "tokens"]).as_array().unwrap().len(), 2);
}

/// What a request with media is refused for, before it takes the card.
#[test]
fn media_requests_are_refused_with_the_reason() {
    let h = vision_harness(4096, 100);
    let part = |p: &str| format!(r#"{{"messages":[{{"role":"user","content":[{p}]}}]}}"#);
    let cases = [
        (
            r#"{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}"#.to_string(),
            "image 1: only data: URLs are accepted",
        ),
        (
            r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#.to_string(),
            "image 1: not an image this server decodes",
        ),
        (
            r#"{"type":"image_url","image_url":{}}"#.to_string(),
            "needs image_url.url",
        ),
        (
            r#"{"type":"video","video":{"frames":[]}}"#.to_string(),
            "at least one frame",
        ),
        (
            r#"{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAAIGZ0eXBpc29tAAAAAA=="}}"#.to_string(),
            "video 1: this is a video file, which the server does not decode",
        ),
        (
            // 448 x 320 gives 140 tokens: more than the 100 this tower takes per request
            format!(r#"{{"type":"image_url","image_url":{{"url":"{}"}}}}"#, image_url(448, 320, 0)),
            "need 140 image tokens; this server takes 100",
        ),
        (
            r#"{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}"#.to_string(),
            "audio input is not supported",
        ),
    ];
    for (p, want) in &cases {
        let (status, text) = h.post("/v1/chat/completions", &part(p));
        assert_eq!(status, 400, "{p}: {text}");
        assert!(text.contains(want), "{p}: expected '{want}' in {text}");
    }
    // nothing reached the engine
    assert!(h.prompts.lock().unwrap().is_empty());
    // a system message with an image is refused by the template itself
    let sys = format!(
        r#"{{"messages":[{{"role":"system","content":[{{"type":"image_url","image_url":{{"url":"{}"}}}}]}},{{"role":"user","content":"x"}}]}}"#,
        image_url(64, 64, 0)
    );
    let (status, text) = h.post("/v1/chat/completions", &sys);
    assert_eq!(status, 400);
    assert!(
        text.contains("System message cannot contain images"),
        "{text}"
    );
    // a model served without a tower says so
    let text_only = harness();
    let img = format!(
        r#"{{"type":"image_url","image_url":{{"url":"{}"}}}}"#,
        image_url(64, 64, 0)
    );
    let (status, text) = text_only.post("/v1/chat/completions", &part(&img));
    assert_eq!(status, 400);
    assert!(text.contains("takes no images or videos"), "{text}");
}

/// `/v1/messages`: image blocks of a user turn and of a tool result reach the tower.
#[test]
fn messages_api_images_reach_the_engine() {
    let h = vision_harness(4096, 4096);
    h.script("Seen.");
    let data = image_url(256, 256, 3);
    let b64 = data.split_once(',').unwrap().1;
    let body = format!(
        r#"{{"model":"vl","max_tokens":64,"messages":[
            {{"role":"user","content":[{{"type":"text","text":"Look: "}},{{"type":"image","source":{{"type":"base64","media_type":"image/x-portable-pixmap","data":"{b64}"}}}}]}},
            {{"role":"assistant","content":[{{"type":"tool_use","id":"toolu_1","name":"shot","input":{{}}}}]}},
            {{"role":"user","content":[{{"type":"tool_result","tool_use_id":"toolu_1","content":[{{"type":"text","text":"screen:"}},{{"type":"image","source":{{"type":"base64","media_type":"image/x-portable-pixmap","data":"{b64}"}}}}]}}]}}]}}"#
    );
    let (status, text) = h.post("/v1/messages", &body.replace('\n', " "));
    assert_eq!(status, 200, "{text}");
    let prompt = h.last_prompt();
    assert_eq!(prompt.matches("<|image_pad|>").count(), 128);
    assert!(prompt.contains("Look: <|vision_start|>"), "{prompt}");
    assert!(
        prompt.contains("<tool_response>\nscreen:<|vision_start|>"),
        "{prompt}"
    );
    assert_eq!(h.images.lock().unwrap().last().unwrap().len(), 2);
    // a URL source is a remote image: refused
    let remote = r#"{"model":"vl","max_tokens":8,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/x.png"}}]}]}"#;
    let (status, text) = h.post("/v1/messages", remote);
    assert_eq!(status, 400);
    assert!(text.contains("only data: URLs are accepted"), "{text}");
    // the count endpoint sees the same prompt
    let (status, text) = h.post("/v1/messages/count_tokens", &body.replace('\n', " "));
    assert_eq!(status, 200, "{text}");
    assert!(
        at(&json::parse_json(&text).unwrap(), &["input_tokens"])
            .as_u64()
            .unwrap()
            > 128
    );
}

/// `/props` and the model list say what the model takes.
#[test]
fn props_report_the_modalities_truthfully() {
    let h = vision_harness(4096, 2048);
    let p = json::parse_json(&h.state.props_json(0, false)).unwrap();
    assert_eq!(at(&p, &["modalities", "vision"]).as_bool(), Some(true));
    assert_eq!(at(&p, &["modalities", "video"]).as_bool(), Some(true));
    assert_eq!(at(&p, &["vision", "max_patches"]).as_u64(), Some(4096));
    assert_eq!(at(&p, &["vision", "max_tokens"]).as_u64(), Some(2048));
    assert_eq!(at(&p, &["vision", "max_pixels"]).as_u64(), Some(1_048_576));
    assert_eq!(
        at(&p, &["vision", "video", "max_frames"]).as_u64(),
        Some(768)
    );
    assert!(h.state.hello_json().contains("\"vision\":true"));
    let text_only = harness();
    let p = json::parse_json(&text_only.state.props_json(0, false)).unwrap();
    assert_eq!(at(&p, &["modalities", "vision"]).as_bool(), Some(false));
    assert_eq!(at(&p, &["modalities", "video"]).as_bool(), Some(false));
    assert!(matches!(at(&p, &["vision"]), JsonValue::Null));
    assert!(text_only.state.hello_json().contains("\"vision\":false"));
}

/// Which served models get a vision tower, and what a load of them is configured with.
#[test]
fn vision_tower_follows_the_checkpoint_the_drafter_and_the_options() {
    let dir = std::env::temp_dir().join(format!("gaudium-vis-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        crate::vision::test_util::QWEN38_27B_CONFIG,
    )
    .unwrap();
    std::fs::write(
        dir.join("preprocessor_config.json"),
        crate::vision::test_util::QWEN38_27B_PREPROCESSOR,
    )
    .unwrap();
    std::fs::write(
        dir.join("video_preprocessor_config.json"),
        crate::vision::test_util::QWEN38_27B_VIDEO_PREPROCESSOR,
    )
    .unwrap();
    let opt = |f: &dyn Fn(&mut ModelRunOptions)| {
        let mut o = ModelRunOptions::default();
        f(&mut o);
        o
    };
    let mut cfg = ServerConfig {
        host: "127.0.0.1".into(),
        port: 0,
        models: Vec::new(),
        kernels_dir: PathBuf::from("/kernels"),
        ui_dir: None,
        max_ctx: 262_144,
        cache_dir: None,
        no_cache: false,
        lazy: true,
        idle_park_secs: 0,
        queue_config: None,
        api_key: None,
        a2ui: false,
        a2ui_prompt: None,
        verbose: false,
        per_model: vec![
            (
                "ds".into(),
                opt(&|o| o.draft_model = Some(PathBuf::from("/m/T-DSpine-b16"))),
            ),
            ("mtp".into(), opt(&|o| o.own_mtp = true)),
            (
                "block".into(),
                opt(&|o| o.draft_model = Some(PathBuf::from("/m/T-DFlash2"))),
            ),
            ("off".into(), opt(&|o| o.vision_patches = Some(0))),
            ("big".into(), opt(&|o| o.vision_patches = Some(8192))),
            ("odd".into(), opt(&|o| o.vision_patches = Some(4098))),
            ("tiny".into(), opt(&|o| o.vision_patches = Some(64))),
        ],
        draft_k: Vec::new(),
        vision: true,
        vision_max_patches: DEFAULT_VISION_MAX_PATCHES,
        vision_max_tokens: DEFAULT_VISION_MAX_TOKENS,
        prefix_slots: 0,
        mcp_proxy: ProxyOptions::default(),
        tls_cert: None,
        tls_key: None,
        cache_ram_bytes: Some(0),
    };
    // plain decode and a DSpine drafter: the server-wide tower
    for id in ["plain", "ds"] {
        assert_eq!(
            vision_plan_of(&cfg, id, &dir, 262_144),
            Some((4096, 16_384)),
            "{id}"
        );
    }
    // drafters that are not validated with image input: text only
    assert_eq!(vision_plan_of(&cfg, "mtp", &dir, 262_144), None);
    assert_eq!(vision_plan_of(&cfg, "block", &dir, 262_144), None);
    // the model's own setting wins over the server's
    assert_eq!(vision_plan_of(&cfg, "off", &dir, 262_144), None);
    assert_eq!(
        vision_plan_of(&cfg, "big", &dir, 262_144),
        Some((8192, 16_384))
    );
    // the image tokens of a request never exceed the served context
    assert_eq!(
        vision_plan_of(&cfg, "plain", &dir, 4096),
        Some((4096, 4096 - 128))
    );
    // what a load and the off-card build are configured with
    let (q, _) = engine_config_of(&cfg, "ds", &dir, 262_144);
    assert_eq!((q.vision_max_patches, q.vision_max_tokens), (4096, 16_384));
    assert_eq!(q.vision_precompile, vec![4096]);
    assert!(
        q.vision_cache_sizes.contains(&4032) && q.vision_cache_sizes.iter().all(|&n| n <= 4096)
    );
    let (q, _) = engine_config_of(&cfg, "mtp", &dir, 262_144);
    assert_eq!((q.vision_max_patches, q.vision_max_tokens), (0, 0));
    assert!(q.vision_precompile.is_empty() && q.vision_cache_sizes.is_empty());
    // what requests may carry
    let serve = vision_serve_of(&cfg, "ds", &dir, 262_144).unwrap().unwrap();
    assert_eq!(
        (serve.max_patches, serve.spec.max_pixels),
        (4096, 1_048_576)
    );
    assert!(serve.spec.video.is_some());
    assert!(vision_serve_of(&cfg, "off", &dir, 262_144)
        .unwrap()
        .is_none());
    // a tower that cannot be loaded is a configuration error at start-up
    assert!(vision_serve_of(&cfg, "odd", &dir, 262_144)
        .unwrap_err()
        .contains("multiple of 4"));
    assert!(vision_serve_of(&cfg, "tiny", &dir, 262_144)
        .unwrap_err()
        .contains("smallest image"));
    // --no-vision: no tower unless the model asks for one itself
    cfg.vision = false;
    assert_eq!(vision_plan_of(&cfg, "plain", &dir, 262_144), None);
    assert_eq!(
        vision_plan_of(&cfg, "big", &dir, 262_144),
        Some((8192, 16_384))
    );
    // a checkpoint without a tower, or a missing one, has none
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "qwen3", "hidden_size": 4096}"#,
    )
    .unwrap();
    cfg.vision = true;
    assert_eq!(vision_plan_of(&cfg, "plain", &dir, 262_144), None);
    assert_eq!(
        vision_plan_of(&cfg, "plain", Path::new("/nonexistent"), 262_144),
        None
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------------------------------------------
// The MCP proxy route and the HTTPS listener, through the server's connection handler
// ------------------------------------------------------------------------------------------------

/// The harness with the MCP proxy, the HTTPS listener and the API key set as given.
fn net_harness(
    proxy: proxy::McpProxy,
    tls: Option<crate::tls::TlsServer>,
    api_key: Option<&str>,
) -> Harness {
    let Harness {
        state,
        replies,
        prompts,
        images,
        tier,
    } = harness();
    let mut st = Arc::try_unwrap(state).ok().expect("the harness owns its state");
    st.proxy = proxy;
    st.tls = tls;
    st.cfg.api_key = api_key.map(str::to_string);
    Harness {
        state: Arc::new(st),
        replies,
        prompts,
        images,
        tier,
    }
}

/// A proxy that may reach the loopback address (where the tests' target listens).
fn loopback_proxy(keyed: bool) -> proxy::McpProxy {
    let opts = ProxyOptions {
        enabled: true,
        allow_private: vec!["127.0.0.1".into()],
        ..ProxyOptions::default()
    };
    proxy::McpProxy::new(&opts, keyed).unwrap()
}

/// A target on a loopback port that answers every request with `body` as JSON and records the
/// head of each request it got.
fn json_target(body: &'static str) -> (u16, Arc<Mutex<Vec<String>>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    thread::spawn(move || {
        for tcp in l.incoming() {
            let Ok(mut tcp) = tcp else { break };
            let mut buf = Vec::new();
            let mut b = [0u8; 2048];
            while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                match tcp.read(&mut b) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&b[..n]),
                }
            }
            log.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&buf).to_ascii_lowercase());
            let _ = write!(
                tcp,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: s-1\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (port, seen)
}

/// Serves the next `n` connections with the server's connection handler.
fn serve(state: &Arc<AppState>, n: usize) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::clone(state);
    thread::spawn(move || {
        for _ in 0..n {
            let Ok((s, _)) = listener.accept() else { break };
            let st = Arc::clone(&state);
            thread::spawn(move || handle_connection(st, s));
        }
    });
    addr
}

/// `(status, head in lower case, body with its transfer chunks joined)` of a raw response.
fn split_response(raw: &[u8]) -> (u16, String, String) {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("no response head in {:?}", String::from_utf8_lossy(raw)));
    let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
    let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut rest = &raw[split + 4..];
    if !head.contains("transfer-encoding: chunked") {
        return (status, head, String::from_utf8_lossy(rest).into_owned());
    }
    let mut out = Vec::new();
    while let Some(eol) = rest.windows(2).position(|w| w == b"\r\n") {
        let n = usize::from_str_radix(std::str::from_utf8(&rest[..eol]).unwrap().trim(), 16)
            .unwrap();
        if n == 0 {
            break;
        }
        out.extend_from_slice(&rest[eol + 2..eol + 2 + n]);
        rest = &rest[eol + 2 + n + 2..];
    }
    (status, head, String::from_utf8_lossy(&out).into_owned())
}

/// One plain-HTTP exchange: `head` is the request line and header lines, `body` its body.
fn plain_exchange(state: &Arc<AppState>, head: &str, body: &str) -> (u16, String, String) {
    let addr = serve(state, 1);
    let mut c = TcpStream::connect(addr).unwrap();
    write!(c, "{head}\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut raw = Vec::new();
    let _ = c.read_to_end(&mut raw);
    split_response(&raw)
}

fn proxy_path(port: u16) -> String {
    format!("/cors-proxy?url=http%3A%2F%2F127.0.0.1%3A{port}%2Fmcp")
}

#[test]
fn props_report_the_proxy_truthfully_and_the_route_says_when_it_is_off() {
    let h = harness();
    let props = json::parse_json(&h.state.props_json(0, true)).unwrap();
    assert_eq!(at(&props, &["cors_proxy_enabled"]).as_bool(), Some(false));
    let (status, _, body) = plain_exchange(
        &h.state,
        "GET /cors-proxy?url=http%3A%2F%2F192.0.2.1%2F HTTP/1.1\r\nHost: localhost",
        "",
    );
    assert_eq!(status, 403);
    assert!(body.contains("feature_disabled") && body.contains("mcp proxy: "), "{body}");

    let h = net_harness(loopback_proxy(false), None, None);
    let (status, _, body) = plain_exchange(&h.state, "GET /props HTTP/1.1\r\nHost: localhost", "");
    assert_eq!(status, 200);
    let props = json::parse_json(&body).unwrap();
    assert_eq!(at(&props, &["cors_proxy_enabled"]).as_bool(), Some(true));

    // GET, POST and DELETE reach the target through the route
    let (port, seen) = json_target("{\"ok\":true}");
    for method in ["GET", "POST", "DELETE"] {
        let (status, head, body) = plain_exchange(
            &h.state,
            &format!("{method} {} HTTP/1.1\r\nHost: localhost", proxy_path(port)),
            "",
        );
        assert_eq!((status, body.as_str()), (200, "{\"ok\":true}"), "{method}");
        assert!(head.contains("mcp-session-id: s-1"), "{head}");
        assert!(!head.contains("access-control-allow-origin"), "{head}");
    }
    let seen = seen.lock().unwrap();
    assert!(seen[0].starts_with("get /mcp http/1.1"), "{}", seen[0]);
    assert!(seen[1].starts_with("post /mcp http/1.1"));
    assert!(seen[2].starts_with("delete /mcp http/1.1"));
    drop(seen);

    let (status, _, body) =
        plain_exchange(&h.state, "GET /gaudium/proxy HTTP/1.1\r\nHost: localhost", "");
    assert_eq!(status, 200);
    let v = json::parse_json(&body).unwrap();
    assert_eq!(at(&v, &["counts", "proxied"]).as_f64(), Some(3.0));
    assert_eq!(s(&v, &["allow_private", "0"]), "127.0.0.1");
}

#[test]
fn the_proxy_route_is_behind_the_api_key_and_never_forwards_it() {
    let h = net_harness(loopback_proxy(true), None, Some("server-key"));
    let (port, seen) = json_target("{\"ok\":true}");
    let path = proxy_path(port);

    let (status, _, body) =
        plain_exchange(&h.state, &format!("POST {path} HTTP/1.1\r\nHost: localhost"), "{}");
    assert_eq!(status, 401);
    assert!(body.contains("authentication_error"), "{body}");
    let (status, _, _) = plain_exchange(
        &h.state,
        &format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer wrong"),
        "{}",
    );
    assert_eq!(status, 401);
    assert!(seen.lock().unwrap().is_empty());
    let (status, _, _) =
        plain_exchange(&h.state, "GET /gaudium/proxy HTTP/1.1\r\nHost: localhost", "");
    assert_eq!(status, 401);

    let (status, _, body) = plain_exchange(
        &h.state,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer server-key\r\n\
             x-gaudium-proxy-header-authorization: Bearer mcp-token\r\n\
             x-gaudium-proxy-header-content-type: application/json"
        ),
        "{}",
    );
    assert_eq!((status, body.as_str()), (200, "{\"ok\":true}"));
    let seen = seen.lock().unwrap();
    assert!(seen[0].contains("authorization: bearer mcp-token"), "{}", seen[0]);
    assert!(!seen[0].contains("server-key"), "{}", seen[0]);
}

/// Behind a front end that ends TLS and forwards plain HTTP (a tunnel, a reverse proxy, a
/// forwarded router port), the server gets a `Host` it has never heard of and a page whose
/// origin is https. Nothing is configured with that name, and every answer is the one a direct
/// request gets.
#[test]
fn a_request_through_a_tls_front_end_gets_the_answers_of_a_direct_one() {
    let h = net_harness(loopback_proxy(false), None, None);
    let (port, seen) = json_target("{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}");
    let direct = "Host: 127.0.0.1:8090";
    let fronted = "Host: chat.example.com\r\nX-Forwarded-Proto: https\r\n\
                   X-Forwarded-For: 198.51.100.7\r\nX-Forwarded-Host: chat.example.com\r\n\
                   Origin: https://chat.example.com\r\nSec-Fetch-Site: same-origin";
    // a front end that rewrites Host to its upstream's name and adds no forwarding header
    let rewritten = "Host: 127.0.0.1:8090\r\nX-Forwarded-Proto: https\r\n\
                     Origin: https://chat.example.com\r\nSec-Fetch-Site: same-origin";
    // an older browser on the same route (no fetch-site statement): Origin against Host
    let bare = "Host: chat.example.com\r\nX-Forwarded-Proto: https\r\nOrigin: https://chat.example.com";

    let proxied = format!("POST {} HTTP/1.1", proxy_path(port));
    let chat = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";
    for (line, body) in [
        ("GET /health HTTP/1.1", ""),
        ("GET /props HTTP/1.1", ""),
        ("GET /v1/models HTTP/1.1", ""),
        ("GET /gaudium/a2ui HTTP/1.1", ""),
        (proxied.as_str(), "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}"),
        // a target that is not listed (an address: nothing is looked up)
        ("DELETE /cors-proxy?url=http%3A%2F%2F192.0.2.1%2F HTTP/1.1", ""),
        ("POST /tokenize HTTP/1.1", "{\"content\":\"Hello there\"}"),
    ] {
        let (d_status, _, d_body) = plain_exchange(&h.state, &format!("{line}\r\n{direct}"), body);
        for (name, headers) in [("fronted", fronted), ("rewritten", rewritten), ("bare", bare)] {
            let (status, head, got) = plain_exchange(&h.state, &format!("{line}\r\n{headers}"), body);
            assert_eq!((status, &got), (d_status, &d_body), "{line} ({name})");
            // no answer sends the browser anywhere by name
            assert!(!head.contains("location:"), "{line} ({name}): {head}");
            assert!(!head.contains("chat.example.com"), "{line} ({name}): {head}");
        }
    }
    // a chat request runs the same way (its answer carries an id and timings of its own)
    for headers in [direct, fronted] {
        h.script("</think>\n\nHello.");
        let (status, _, body) = plain_exchange(
            &h.state,
            &format!("POST /v1/chat/completions HTTP/1.1\r\n{headers}"),
            chat,
        );
        assert_eq!(status, 200, "{body}");
        let v = json::parse_json(&body).unwrap();
        assert_eq!(s(&v, &["choices", "0", "message", "content"]), "Hello.");
    }
    // the proxied request reached the target each time, and the target saw none of the front
    // end's headers
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4);
    for head in seen.iter() {
        assert!(head.contains(&format!("host: 127.0.0.1:{port}")), "{head}");
        assert!(!head.contains("x-forwarded") && !head.contains("chat.example.com"), "{head}");
    }
}

/// One exchange over TLS with the engine's own client: `(status, head, body)`.
fn tls_exchange(
    client: &crate::tls::TlsClient,
    addr: std::net::SocketAddr,
    name: &str,
    head: &str,
    body: &str,
) -> (u16, String, String) {
    let mut c = client
        .connect(TcpStream::connect(addr).unwrap(), name)
        .expect("a TLS connection to the server");
    write!(c, "{head}\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut raw = Vec::new();
    let _ = c.read_to_end(&mut raw);
    split_response(&raw)
}

#[test]
fn http_and_https_share_the_port() {
    let Some(tc) =
        crate::tls::test_util::test_cert(&["DNS:localhost", "DNS:gaudium.example.com", "IP:127.0.0.1"])
    else {
        return;
    };
    let server = crate::tls::TlsServer::new(&tc.cert, &tc.key).unwrap();
    let h = net_harness(loopback_proxy(false), Some(server), None);
    let client = crate::tls::TlsClient::new(Some(&tc.cert)).unwrap();
    let (port, _) = json_target("{\"ok\":true}");

    // every connection below goes to ONE listening port
    let addr = serve(&h.state, 16);
    let plain = |head: &str, body: &str| {
        let mut c = TcpStream::connect(addr).unwrap();
        write!(c, "{head}\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut raw = Vec::new();
        let _ = c.read_to_end(&mut raw);
        split_response(&raw)
    };

    // plain HTTP goes on working beside HTTPS
    let (status, _, body) = plain("GET /health HTTP/1.1\r\nHost: localhost", "");
    assert_eq!((status, body.as_str()), (200, "{\"status\":\"ok\"}"));

    // HTTPS under each name the page may be opened by: this machine, a LAN name, an address
    for name in ["localhost", "gaudium.example.com", "127.0.0.1"] {
        let (status, _, body) = tls_exchange(
            &client,
            addr,
            name,
            &format!("GET /props HTTP/1.1\r\nHost: {name}"),
            "",
        );
        assert_eq!(status, 200, "{name}");
        let props = json::parse_json(&body).unwrap();
        assert_eq!(at(&props, &["cors_proxy_enabled"]).as_bool(), Some(true), "{name}");
    }

    // the proxy from an HTTPS page to a plain-http MCP server (what a browser cannot do itself)
    let (status, head, body) = tls_exchange(
        &client,
        addr,
        "localhost",
        &format!(
            "POST {} HTTP/1.1\r\nHost: localhost:8090\r\nOrigin: https://localhost:8090\r\n\
             Sec-Fetch-Site: same-origin\r\nx-gaudium-proxy-header-content-type: application/json",
            proxy_path(port)
        ),
        "{}",
    );
    assert_eq!((status, body.as_str()), (200, "{\"ok\":true}"));
    assert!(head.contains("mcp-session-id: s-1"), "{head}");

    // a streamed chat completion over TLS (the stream writer holds a second handle on the
    // connection)
    h.script("</think>\n\nHello over TLS.");
    let (status, _, body) = tls_exchange(
        &client,
        addr,
        "localhost",
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json",
        "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}",
    );
    assert_eq!(status, 200, "{body}");
    let st = collect_chat_stream(&body);
    assert_eq!(st.content, "Hello over TLS.");
    assert!(st.done);

    // a client that does not know the certificate's issuer refuses the server
    let strict = crate::tls::TlsClient::new(None).unwrap();
    assert!(strict
        .connect(TcpStream::connect(addr).unwrap(), "localhost")
        .is_err());

    // another program's TLS client, when there is one
    let curl = std::process::Command::new("curl")
        .args(["--silent", "--show-error", "--max-time", "10", "--cacert"])
        .arg(&tc.cert)
        .arg(format!("https://localhost:{}/health", addr.port()))
        .output();
    match curl {
        Ok(out) => assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "{\"status\":\"ok\"}",
            "curl: {}",
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(e) => eprintln!("skipped the curl check: curl is not available ({e})"),
    }
}

#[test]
fn a_tls_client_gets_no_http_answer_from_a_server_without_a_certificate() {
    let Some(tc) = crate::tls::test_util::test_cert(&["DNS:localhost"]) else {
        return;
    };
    let h = harness();
    let addr = serve(&h.state, 2);
    let client = crate::tls::TlsClient::new(Some(&tc.cert)).unwrap();
    // the handshake is not answered: the connection closes
    assert!(client
        .connect(TcpStream::connect(addr).unwrap(), "localhost")
        .is_err());
    // and the server goes on serving plain HTTP
    let mut c = TcpStream::connect(addr).unwrap();
    write!(c, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
    let mut raw = Vec::new();
    let _ = c.read_to_end(&mut raw);
    assert_eq!(split_response(&raw).0, 200);
}

#[test]
fn the_https_listener_starts_only_with_both_files_and_says_why_not() {
    let mut cfg = harness().state.cfg.clone();
    assert!(start_tls(&cfg).unwrap().is_none());
    cfg.tls_cert = Some(PathBuf::from("/nonexistent/cert.pem"));
    let e = start_tls(&cfg).err().unwrap();
    assert!(e.contains("--tls-cert and --tls-key go together"), "{e}");
    cfg.tls_key = Some(PathBuf::from("/nonexistent/key.pem"));
    if crate::tls::availability().is_err() {
        // no TLS library on this machine: the server starts, plain HTTP only
        assert!(start_tls(&cfg).unwrap().is_none());
        return;
    }
    let e = start_tls(&cfg).err().unwrap();
    assert!(e.contains("HTTPS listener") && e.contains("/nonexistent/cert.pem"), "{e}");
    let Some(tc) = crate::tls::test_util::test_cert(&["DNS:localhost"]) else {
        return;
    };
    cfg.tls_cert = Some(tc.cert.clone());
    cfg.tls_key = Some(tc.key.clone());
    assert!(start_tls(&cfg).unwrap().is_some());
    // the error for a wrong key names the file, never what is in it
    cfg.tls_key = Some(tc.cert.clone());
    let e = start_tls(&cfg).err().unwrap();
    assert!(e.contains("private key file"), "{e}");
    assert!(!e.contains("BEGIN") && !e.contains("MII"), "{e}");
}
