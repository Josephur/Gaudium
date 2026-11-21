// The server API: OpenAI-compatible endpoints and Gaudium extensions.
import { readSSE, newSegment, applyChunk } from './stream.js';

let apiKey = '';
export const setApiKey = (k) => { apiKey = k || ''; };
export const authHeaders = (extra = {}) => (apiKey ? { ...extra, Authorization: `Bearer ${apiKey}` } : extra);

export class HttpError extends Error {
  constructor(status, body) {
    const e = body && body.error ? body.error : {};
    super(e.message || `HTTP ${status}`);
    this.status = status; this.type = e.type || ''; this.body = body;
  }
}

async function getJson(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: authHeaders(opts.headers || {}) });
  let body = null;
  try { body = await r.json(); } catch { body = null; }
  if (!r.ok) throw new HttpError(r.status, body);
  return body;
}

export const getProps = () => getJson('/props');
export const getEngine = () => getJson('/gaudium/engine');
export const getUiInfo = () => getJson('/gaudium/ui');
export const getLive = () => getJson('/gaudium/live');
export const engineAction = (action) => getJson('/gaudium/engine', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }) });
// the prompt size of a chat body without running it (gaudium-server's /gaudium/count_tokens; docs/ui.md 10) and of a text
export const countTokens = (body) => getJson('/gaudium/count_tokens', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
export const tokenize = (content) => getJson('/tokenize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content }) });
export const skipReasoning = (id) => getJson('/v1/chat/completions/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'reasoning_end', id }) });

// Build the request body from the UI settings, including Gaudium's optional gaudium_method selector.
export function buildBody(messages, s, tools) {
  const b = {
    messages, stream: true, timings_per_token: true, return_progress: true, sse_ping_interval: 1,
    reasoning_format: 'auto', cache_prompt: true,
  };
  if (s.thinking === 'off') b.chat_template_kwargs = { enable_thinking: false };
  else { b.chat_template_kwargs = { enable_thinking: true }; if (s.effort && s.effort !== 'default') b.reasoning_effort = s.effort; }
  const num = (v) => (v === '' || v == null ? null : Number(v));
  for (const k of ['temperature', 'top_k', 'top_p', 'min_p']) { const v = num(s[k]); if (v != null && isFinite(v)) b[k] = v; }
  const seed = num(s.seed); if (seed != null && isFinite(seed) && seed >= 0) b.seed = seed;
  const mt = num(s.max_tokens); if (mt != null && isFinite(mt) && mt > 0) b.max_tokens = mt;
  if (s.method && s.method !== 'default') b.gaudium_method = s.method;
  if (tools && tools.length) b.tools = tools;
  if (s.custom && s.custom.trim()) { try { Object.assign(b, JSON.parse(s.custom)); } catch { /* shown as invalid in settings */ } }
  return b;
}

// Stream one completion. onUpdate(seg, info) after every chunk; resolves with the finished segment.
export async function streamChat(body, { signal, onUpdate, onOpen } = {}) {
  const seg = newSegment();
  seg.sentAt = performance.now();
  const r = await fetch('/v1/chat/completions', { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify(body), signal });
  if (!r.ok) {
    let j = null; try { j = await r.json(); } catch { j = null; }
    throw new HttpError(r.status, j);
  }
  if (onOpen) onOpen(seg);
  await readSSE(r, (data) => {
    const res = applyChunk(seg, data, performance.now());
    if (onUpdate) onUpdate(seg, res);
  }, signal);
  return seg;
}

// The Gaudium event stream (one EventSource; reconnects by itself; closed while the tab is hidden)
export function connectEvents(onEvent, onState) {
  let es = null, closed = false, poll = null;
  const open = () => {
    if (closed || es) return;
    if (apiKey) { startPolling(); return; }   // EventSource cannot send the key: poll /gaudium/live instead
    es = new EventSource('/gaudium/events');
    es.onopen = () => onState && onState('open');
    es.onerror = () => { onState && onState('error'); if (es && es.readyState === 2) { es = null; setTimeout(open, 3000); } };
    es.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch { /* ignore */ } };
  };
  const startPolling = () => {
    if (poll) return;
    const tick = async () => {
      try { const s = await getLive(); onState && onState('poll'); for (const k of ['hello', 'engine', 'request', 'live', 'switch', 'sections']) if (s[k]) onEvent(s[k]); } catch { onState && onState('error'); }
    };
    tick(); poll = setInterval(tick, 1000);
  };
  const close = () => { if (es) { es.close(); es = null; } if (poll) { clearInterval(poll); poll = null; } };
  const vis = () => { if (document.hidden) close(); else open(); };
  document.addEventListener('visibilitychange', vis);
  open();
  return { close() { closed = true; close(); document.removeEventListener('visibilitychange', vis); }, reopen() { close(); open(); } };
}
