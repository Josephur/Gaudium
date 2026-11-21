// A small MCP client in the browser: the browser talks to MCP servers and runs the tool loop.
// Streamable HTTP transport (JSON-RPC over POST; answers as JSON or as an SSE stream), protocol 2025-06-18.
// "Use server proxy": the request goes to Gaudium's /cors-proxy?url=… (forwarded headers
// carry the prefix x-gaudium-proxy-header-), for servers that send no CORS headers or are reachable only from the server.
import { createSSEParser } from './stream.js';
import { authHeaders } from './api.js';

const PROTO = '2025-06-18';
const PFX = 'x-gaudium-proxy-header-';

export class McpClient {
  constructor(server) {
    this.s = server; this.session = null; this.nextId = 1; this.tools = []; this.info = null; this.protocol = PROTO;
  }
  target() { return this.s.proxy ? `/cors-proxy?url=${encodeURIComponent(this.s.url)}` : this.s.url; }
  headers(extra = {}) {
    const h = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', ...extra };
    const own = parseHeaders(this.s.headers);
    if (this.session) own['Mcp-Session-Id'] = this.session;
    if (this.info) own['MCP-Protocol-Version'] = this.protocol;
    // through the proxy only prefixed headers reach the MCP server, so Content-Type and Accept are forwarded too (without them
    // the upstream POST went out as text/plain and mcp-searxng answered 415, 2026-11-10); Authorization (our API key) stays ours
    if (this.s.proxy) { const out = authHeaders({ 'Content-Type': h['Content-Type'] }); for (const [k, v] of Object.entries({ ...h, ...own })) out[PFX + k.toLowerCase()] = v; return out; }
    return { ...h, ...own };
  }
  async rpc(method, params, { notify = false, timeoutMs = 60000, signal } = {}) {
    const id = notify ? undefined : this.nextId++;
    const body = notify ? { jsonrpc: '2.0', method, params } : { jsonrpc: '2.0', id, method, params };
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    if (signal) signal.addEventListener('abort', () => ctl.abort(), { once: true });
    try {
      const r = await fetch(this.target(), { method: 'POST', headers: this.headers(), body: JSON.stringify(body), signal: ctl.signal });
      const sid = r.headers.get('Mcp-Session-Id'); if (sid) this.session = sid;
      if (notify) return null;
      if (!r.ok) { const txt = await r.text().catch(() => ''); const e = new Error(`${this.s.name || this.s.url}: HTTP ${r.status}${txt ? ': ' + txt.slice(0, 200) : ''}`); e.status = r.status; e.body = txt; throw e; }
      const ct = r.headers.get('Content-Type') || '';
      let msg = null;
      if (ct.includes('text/event-stream')) {
        const dec = new TextDecoder(); const reader = r.body.getReader();
        const p = createSSEParser((e) => { try { const j = JSON.parse(e.data); if (j.id === id) msg = j; } catch { /* not ours */ } });
        while (!msg) { const { value, done } = await reader.read(); if (done) break; p.feed(dec.decode(value, { stream: true })); }
        p.end(); try { await reader.cancel(); } catch { /* closed */ }
      } else msg = await r.json();
      if (!msg) throw new Error('no answer from the MCP server');
      if (msg.error) throw new Error(msg.error.message || JSON.stringify(msg.error));
      return msg.result;
    } finally { clearTimeout(t); }
  }
  async connect() {
    const res = await this.rpc('initialize', { protocolVersion: PROTO, capabilities: {}, clientInfo: { name: 'gaudium-ui', version: '1.0' } }, { timeoutMs: 20000 });
    this.info = res; this.protocol = res.protocolVersion || PROTO;
    await this.rpc('notifications/initialized', {}, { notify: true }).catch(() => {});
    const tools = [];
    let cursor;
    for (let i = 0; i < 20; i++) { const r = await this.rpc('tools/list', cursor ? { cursor } : {}); tools.push(...(r.tools || [])); cursor = r.nextCursor; if (!cursor) break; }
    this.tools = tools;
    return { info: res, tools };
  }
  // end the session (Streamable HTTP: DELETE with the session id). Servers that do not allow it answer 405; either way it is gone here.
  async close() {
    if (!this.session) return;
    const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 5000);
    try { await fetch(this.target(), { method: 'DELETE', headers: this.headers(), signal: ctl.signal }); } catch { /* best effort */ } finally { clearTimeout(t); this.session = null; }
  }
  async call(name, args, signal) {
    const r = await this.rpc('tools/call', { name, arguments: args || {} }, { timeoutMs: 300000, signal });
    return { text: resultText(r), isError: !!r.isError, raw: r };
  }
}

// The fetch threw (a TypeError: "Failed to fetch", "NetworkError…", "Load failed"): the page got no answer it may read.
export const isNetworkError = (e) => e instanceof TypeError || /Failed to fetch|NetworkError|Load failed/i.test(String(e && e.message));

// What to tell the person when the browser could not read an answer from an MCP server. A page cannot tell apart an origin
// refusal, a missing CORS header and an unreachable host (all three surface as the same TypeError), so the likely causes are
// listed, most likely first. MCP servers on the official TypeScript SDK check the Origin header (DNS-rebinding protection) and
// answer 403 without CORS headers, which the browser reports exactly like an unreachable server (mcp-searxng 2.3.0, 2026-11-08).
// probe: what the same request through gaudium-server's proxy gave (it sends no browser Origin): {ok} | {status, message} |
// {unreachable, message} (the proxy could not connect) | {proxyError, message} (the proxy refused, e.g. its address guard);
// null when there was no probe.
export function connectFailureText({ url, origin, proxyEnabled, viaProxy = false, probe = null }) {
  if (viaProxy) return `The page could not reach gaudium-server's proxy for ${url}. Check that gaudium-server is running.`;
  const lines = [`The browser got no answer it may read from ${url}. Likely causes, most likely first:`,
    `1. The server refuses this page's origin (${origin}). Add ${origin} to its allowed origins (for mcp-searxng: MCP_HTTP_ALLOWED_ORIGINS).`,
    '2. The server sends no CORS headers, so the browser hides its answer.',
    '3. The server cannot be reached from this browser (address, port, firewall).'];
  if (probe && probe.ok) lines.push('Through gaudium-server\'s proxy it answers, so it is up: cause 1 or 2. Allow the origin, or use the server proxy for it.');
  else if (probe && probe.status) lines.push(`Through gaudium-server's proxy it answers HTTP ${probe.status}${probe.message ? `: ${probe.message}` : ''}.`);
  else if (probe && probe.unreachable) lines.push(`gaudium-server's proxy cannot reach it either${probe.message ? ` (${probe.message})` : ''}: cause 3 is likely.`);
  else if (probe && probe.proxyError) lines.push(`gaudium-server's proxy did not try it: ${probe.message}.`);
  lines.push(proxyEnabled ? 'Or tick "Use the server proxy" for this server: gaudium-server then connects instead of the browser.'
    : 'Or start gaudium-server with --mcp-proxy and use its proxy: it connects instead of the browser.');
  return lines.join('\n');
}

// "Key: value" lines or a JSON object
export function parseHeaders(s) {
  if (!s) return {};
  if (typeof s === 'object') return { ...s };
  const t = s.trim(); if (!t) return {};
  if (t.startsWith('{')) { try { return JSON.parse(t); } catch { return {}; } }
  const out = {};
  for (const line of t.split('\n')) { const i = line.indexOf(':'); if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim(); }
  return out;
}

// a tools/call result as the text the model sees
export function resultText(r) {
  if (!r) return '';
  if (r.structuredContent && !(r.content || []).length) return JSON.stringify(r.structuredContent);
  return (r.content || []).map((c) => c.type === 'text' ? c.text : c.type === 'resource' ? (c.resource && (c.resource.text || c.resource.uri)) || '' : `[${c.type}${c.mimeType ? ' ' + c.mimeType : ''}]`).join('\n');
}

// OpenAI tool definitions for the request; names are kept as the server gives them (the first server wins a clash)
export function toOpenAiTools(entries) {
  const seen = new Set(); const out = [];
  for (const { tool } of entries) {
    if (seen.has(tool.name)) continue; seen.add(tool.name);
    out.push({ type: 'function', function: { name: tool.name, description: tool.description || '', parameters: tool.inputSchema || { type: 'object', properties: {} } } });
  }
  return out;
}

// built-in tools that run in the page
export const BUILTIN = {
  get_datetime: {
    tool: { name: 'get_datetime', description: 'Get the current date and time in the user\'s time zone.', inputSchema: { type: 'object', properties: {} } },
    run: () => { const d = new Date(); return `${d.toString()} (ISO ${d.toISOString()}, time zone ${Intl.DateTimeFormat().resolvedOptions().timeZone})`; },
  },
};
