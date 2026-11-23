// The application: state, the send / stream / tool loop, branches, conversations, the Gaudium event stream, shortcuts.
import { html, cx, createStore } from './ui/h.js';
import { useEffect, useRef, useState, useMemo } from 'preact/hooks';
import { Icon } from './ui/icons.js';
import { Sidebar } from './ui/sidebar.js';
import { Composer } from './ui/composer.js';
import { UserMessage, AssistantMessage, LiveDock } from './ui/message.js';
import { useFollow } from './ui/follow.js';
import { SettingsDialog, CardDialog, ShortcutsDialog, ConfirmDialog, KeyDialog } from './ui/dialogs.js';
import { onLibsReady } from './lib/markdown.js';
import * as api from './lib/api.js';
import * as db from './lib/db.js';
import { McpClient, toOpenAiTools, BUILTIN, isNetworkError, connectFailureText } from './lib/mcp.js';
import { newConversation, addChild, update, activePath, pathTo, siblings, switchSibling, removeSubtree, titleFrom, toApiMessages, uid,
  insertAbove, spliceOut, compactableCount, compactionMessages, displayCut, countMessages, isCompaction, nodeSummary, usableSummary, SUMMARY_PREFILL } from './lib/tree.js';
import { finalStats, methodInfo, greedyOnly, setDfName, fmtInt, contextUsed } from './lib/stats.js';
import { modelDesc, modelFacts, precisionShort, hasDrafter, dfFamilyName } from './lib/model.js';
import { reserveTokens, compactPercent, projection, summaryMaxTokens, usedLive, estimateMessages, estimateText, longestText, elideMessage, setVisionLimits } from './lib/context.js';
import { visionLimits } from './lib/media.js';
import { newSegment } from './lib/stream.js';
import { blocksOfPath, syncRuntime, wrapMessage, errorMessage, adoptCalls, callIsA2ui, RENDER_TOOL } from './lib/a2ui.js';
import { loadSettings, saveSettings, applyTheme, effectiveDark, loadDockCollapsed, saveDockCollapsed } from './lib/settings.js';

const settings0 = loadSettings();
applyTheme(settings0.theme);
api.setApiKey(settings0.apiKey);

const fresh = newConversation();
export const store = createStore({
  settings: settings0, props: null, propsError: null, needKey: false, keyError: '',
  convs: [], conv: fresh.conv, messages: fresh.messages, saved: false,
  streaming: null, tick: 0,
  hello: null, engine: null, engineAt: 0, live: null, events: 'connecting',
  query: '', railOpen: window.innerWidth > 900, dialog: null, confirm: null, toast: null, mcp: {},
  searchKey: 0, focusKey: 0, renaming: null, now: Date.now(), themeKey: effectiveDark() ? 'd' : 'l', dockCollapsed: loadDockCollapsed(),
  ctxBase: null,     // {convId, key, used, exact}: what the open chat's next turn occupies before the draft (the ring; docs/ui.md 10)
  expandCut: null,   // the compaction node whose earlier messages are shown
});
const S = () => store.get();
const set = (p) => store.set(p);

// ------------------------------------------------------------------------------------------------- conversations
const cache = new Map();   // convId -> {conv, messages}: the open one and any that is streaming
const saveTimers = new Map();
function current() { const s = S(); return { conv: s.conv, messages: s.messages }; }
function getConv(id) { if (id === S().conv.id) return current(); return cache.get(id); }
function commit(conv, messages, { save = true, now = true } = {}) {
  const c = now ? { ...conv, updated: Date.now() } : conv;
  cache.set(c.id, { conv: c, messages });
  if (S().conv.id === c.id) set({ conv: c, messages, saved: S().saved || save });
  if (save) scheduleSave(c.id);
}
function scheduleSave(id, ms = 400) {
  clearTimeout(saveTimers.get(id));
  saveTimers.set(id, setTimeout(async () => {
    const x = cache.get(id); if (!x) return;
    try { await db.saveConversation(x.conv, x.messages); refreshList(); } catch (e) { toast(`Could not save the chat: ${e.message}`); }
  }, ms));
}
async function refreshList() { set({ convs: await db.listConversations() }); }
function mutate(convId, fn, opts) {
  const x = getConv(convId); if (!x) return;
  const r = fn(x.conv, x.messages);
  commit(r.conv || x.conv, r.messages || x.messages, opts);
}

async function openConv(id) {
  if (S().conv.id === id) { closeRailOnPhone(); return; }
  let x = cache.get(id);
  if (!x) {
    const c = S().convs.find((k) => k.id === id); const m = await db.loadTree(id);
    if (!c || !m) { toast('That chat could not be loaded.'); return; }
    x = { conv: c, messages: m }; cache.set(id, x);
  }
  set({ conv: x.conv, messages: x.messages, saved: true, focusKey: S().focusKey + 1 });
  try { localStorage.setItem('gaudium-ui.last', id); } catch { /* no storage */ }
  closeRailOnPhone();
}
function newChat() {
  const n = newConversation(); cache.set(n.conv.id, n);
  set({ conv: n.conv, messages: n.messages, saved: false, focusKey: S().focusKey + 1 });
  closeRailOnPhone();
}
function closeRailOnPhone() { if (window.innerWidth <= 900) set({ railOpen: false }); }

function toast(msg) { set({ toast: msg }); clearTimeout(toast.t); toast.t = setTimeout(() => set({ toast: null }), 3600); }

function download(name, obj) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 1)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
const stamp = () => new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');

// ------------------------------------------------------------------------------------------------- settings
function setSetting(k, v) {
  const s = { ...S().settings, [k]: v };
  saveSettings(s);
  set({ settings: s });
  if (k === 'theme') { applyTheme(v); set({ themeKey: effectiveDark() ? 'd' : 'l' }); }
  if (k === 'apiKey') { api.setApiKey(v); }
  if (k === 'mcpServers') { const ids = new Set(v.map((x) => x.id)); for (const id of [...mcpClients.keys()]) if (!ids.has(id)) mcpClients.delete(id); }
}

// ------------------------------------------------------------------------------------------------- MCP + tools
const mcpClients = new Map();
async function mcpConnect(srv) {
  set({ mcp: { ...S().mcp, [srv.id]: { status: 'connecting', tools: [] } } });
  const c = new McpClient(srv);
  try {
    const r = await c.connect(); mcpClients.set(srv.id, c);
    set({ mcp: { ...S().mcp, [srv.id]: { status: 'ok', tools: r.tools, server: r.info && r.info.serverInfo } } });
  } catch (e) {
    mcpClients.delete(srv.id);
    let error = e.message;
    if (isNetworkError(e)) {
      const proxyEnabled = !!(S().props && S().props.cors_proxy_enabled);
      let probe = null;
      if (proxyEnabled && !srv.proxy) probe = await mcpProbe(srv);   // once, through the server: it sends no browser Origin
      error = connectFailureText({ url: srv.url, origin: window.location.origin, proxyEnabled, viaProxy: !!srv.proxy, probe });
    }
    set({ mcp: { ...S().mcp, [srv.id]: { status: 'error', tools: [], error } } });
  }
}
// the server's real answer through gaudium-server's proxy (status and message), for the error text of a failed browser connect
// The probe's initialize opens a session on the server; it is closed again (MCP session DELETE, through the proxy) so a failed
// connect leaves nothing open there.
async function mcpProbe(srv) {
  const c = new McpClient({ ...srv, proxy: true });
  try { await c.rpc('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'gaudium-ui', version: '1.0' } }, { timeoutMs: 8000 }); return { ok: true }; }
  catch (e) {
    if (!e.status) return isNetworkError(e) ? null : { unreachable: true, message: String(e.message).slice(0, 160) };
    let m = e.body || ''; try { const j = JSON.parse(m); m = (j.error && (j.error.message || j.error)) || m; } catch { /* plain text */ }
    m = String(m).slice(0, 160);
    if (/^proxy error: /.test(m)) return { unreachable: true, message: m.replace(/^proxy error: /, '') };   // the proxy's own answers
    if (/^mcp proxy: /.test(m)) return { proxyError: true, message: m.replace(/^mcp proxy: /, '') };
    return { status: e.status, message: m };
  } finally { await c.close(); }
}
function collectTools() {
  const s = S().settings; const entries = []; const map = new Map();
  if (s.builtinTools) for (const [n, b] of Object.entries(BUILTIN)) { entries.push({ tool: b.tool }); map.set(n, { builtin: b }); }
  for (const srv of s.mcpServers) {
    if (!srv.enabled) continue; const c = mcpClients.get(srv.id); if (!c) continue;
    for (const t of c.tools) { if ((srv.disabledTools || []).includes(t.name) || map.has(t.name)) continue; entries.push({ tool: t }); map.set(t.name, { client: c, server: srv.name }); }
  }
  // beside other tools, the interface is a tool too (lib/a2ui.js RENDER_TOOL); alone, the model writes the block
  if (entries.length && s.a2ui && a2uiOffered() && !map.has(RENDER_TOOL.name)) entries.push({ tool: RENDER_TOOL });
  return { defs: toOpenAiTools(entries), map };
}
const asks = new Map();   // tool call id -> resolve(choice)
function toolAnswer(id, choice) { const r = asks.get(id); if (r) { asks.delete(id); r(choice); } }

// ------------------------------------------------------------------------------------------------- streaming
let raf = 0;
function bump() { if (raf) return; raf = requestAnimationFrame(() => { raf = 0; set({ tick: S().tick + 1 }); }); }

function predictedGreedy(snap) {
  const p = S().props; const def = p && p.default_generation_settings && p.default_generation_settings.params;
  const t = snap.temperature === '' || snap.temperature == null ? (def ? def.temperature : 1) : Number(snap.temperature);
  return !(t > 0);
}
// the engine's methods and precision: from its live state (they exist once it has started), else from the hello
function engineInfo() {
  const s = S(); const i = (s.engine && s.engine.info) || {}; const h = s.hello || {}; const hm = h.methods || {};
  return { greedy: i.greedy || hm.greedy || '', sampled: i.sampled || hm.sampled || '', loaded: i.methods || hm.loaded || '', precision: i.precision || h.precision || '' };
}
// The methods the loaded engine runs, or null while it has not said. A saved method the engine does not run (picked on
// another model: DSpine on the 27B, then Flash-Next loaded) is treated as unset, so it is never listed or sent; the saved
// choice itself stays, and comes back when a model that runs it is loaded.
function loadedMethods() {
  const l = engineInfo().loaded;
  return l ? [...new Set(String(l).replace(/\(.*?\)/g, '').split(/[\s,/]+/).filter(Boolean))] : null;
}
function effSettings(s) {
  if (!s || !s.method || s.method === 'default') return s;
  const l = loadedMethods();
  return l && !l.includes(s.method) ? { ...s, method: 'default' } : s;
}
// A greedy-only method picked while the request would sample (docs/ui.md 3.1). The engine refuses that, so:
// - temperature left empty (the server's default, T 1.0): picking the method is the choice, the request runs it greedy
//   (temperature 0), with a "greedy" chip;
// - an explicit temperature > 0: sampling is the choice, the server runs the sampled default instead
//   (gaudium.method_fallback), with a "sampled: MTP adaptive" chip.
// The readout says which, from the send on. null: nothing to say.
function methodPlan(s) {
  s = effSettings(s);
  if (!s || !s.method || s.method === 'default') return null;
  const h = S().hello; const list = h && h.methods && h.methods.greedy_only;
  if (!greedyOnly(s.method, list) || predictedGreedy(s)) return null;
  if (s.temperature === '' || s.temperature == null) return { kind: 'greedy', method: s.method };
  return { kind: 'fallback', method: s.method, used: engineInfo().sampled || 'adaptive' };
}
const forcedGreedy = (s) => { const p = methodPlan(s); return !!(p && p.kind === 'greedy'); };
// the settings a request is built from: temperature 0 when the plan is greedy
// (interactive interfaces are declared only to a server that offers them: it refuses the declaration otherwise)
const a2uiOffered = () => !!(S().props && S().props.a2ui && S().props.a2ui.enabled);
function sendSettings() {
  let s = effSettings(S().settings);
  if (s.a2ui && !a2uiOffered()) s = { ...s, a2ui: false };
  return forcedGreedy(s) ? { ...s, temperature: 0 } : s;
}
function defaultMethod(snap) {
  const e = engineInfo(); const g = snap && snap.greedy != null ? snap.greedy : predictedGreedy(snap || {});
  return g ? e.greedy : e.sampled;
}
function requestSnapshot() {
  const s = effSettings(S().settings);
  const snap = { temperature: s.temperature, top_k: s.top_k, top_p: s.top_p, min_p: s.min_p, seed: s.seed, max_tokens: s.max_tokens, method: s.method === 'default' ? '' : s.method, thinking: s.thinking, effort: s.effort };
  const plan = methodPlan(s);
  if (plan) snap.methodPlan = plan;
  if (plan && plan.kind === 'greedy') { snap.forcedGreedy = true; snap.temperature = 0; }
  snap.greedy = predictedGreedy(snap);
  return snap;
}
function newAssistant() { return { id: uid('a'), role: 'assistant', segments: [], created: Date.now(), request: requestSnapshot() }; }

function friendlyError(e, { canCompact = true } = {}) {
  if (e && e.name === 'AbortError') return null;
  if (e instanceof api.HttpError) {
    const b = e.body && e.body.error;
    if (e.status === 401 || e.status === 403 && e.type === 'authentication_error') { set({ needKey: true, keyError: 'The key was not accepted.' }); return { message: 'The server needs an API key.', code: e.status }; }
    if (b && b.type === 'exceed_context_size_error') return { message: `This conversation (${fmtInt(b.n_prompt_tokens)} tokens) is longer than the context (${fmtInt(b.n_ctx)} tokens).`, code: 400, overflow: true, canCompact,
      hint: canCompact ? 'Compact the conversation and send it again, or start a new chat.' : 'Nothing before this message is left to compact: it is too long for the context by itself. Shorten it or remove an attachment.' };
    if (e.status === 503) return { message: e.message, code: 503, hint: 'The engine is parked or loading. Open “Engine & Model” to see its state and start it.' };
    return { message: e.message, code: e.status };
  }
  return { message: /Failed to fetch|NetworkError|network/i.test(String(e && e.message)) ? 'The connection to the server was lost.' : String((e && e.message) || e), hint: 'Check that gaudium-server is running (gaudium-serve status).' };
}

function finishSegment(seg, live) {
  const g = { ...seg };
  g.stats = finalStats(seg);
  g.live = live && live.rec ? live.rec : null;
  delete g.samples; delete g.progress; delete g.done;
  if (g.gaudium && g.gaudium.tokens) { g.gaudium = { ...g.gaudium }; delete g.gaudium.tokens; }   // ids are not needed in the browser
  return g;
}

// One streamed completion into `targetId` (an assistant turn, or a compaction node): appends the finished segment.
// kind 'compaction': a summary (marked on the segment; the prefill the request continued from is put back in front of it).
async function streamPass(st, convId, targetId, body, { kind = null, compacted = 0, canCompact = true } = {}) {
  const signal = st.abort.signal;
  const mark = (g) => { if (g && kind) { g.kind = kind; g.prefill = SUMMARY_PREFILL; g.compacted = compacted; } return g; };
  st.seg = mark(newSegment()); st.started = false; st.switchStep = null; st.rec = null; st.sections = null; st.phase = 'sending'; st.kind = kind; bump();
  let seg = null, err = null;
  try {
    seg = await api.streamChat(body, {
      signal,
      onUpdate: (g, res) => {
        mark(g); st.seg = g; st.started = true;
        const inR = !!g.reasoning && !g.content && !g.toolCalls.length;
        if (st.inReasoning && !inR && g.reasoning && g.reasoningMs == null && g.firstTokenAt != null) g.reasoningMs = performance.now() - g.firstTokenAt;
        st.inReasoning = inR;
        st.phase = g.firstTokenAt == null ? 'prefill' : kind ? 'compacting' : g.toolCalls.length ? 'tool call' : inR ? 'thinking' : 'answering';
        if (g.gaudium && g.gaudium.switch_step != null && g.gaudium.switch_step >= 0) st.switchStep = g.gaudium.switch_step;
        if (g.progress && g.firstTokenAt == null) st.progressText = `${fmtInt(g.progress.processed)} of ${fmtInt(g.progress.total)} tokens${g.progress.cache ? ` (${fmtInt(g.progress.cache)} reused)` : ''}`;
        const u = usedLive(g); if (u != null) st.ctxUsed = u;   // the ring follows the stream
        bump();
      },
    });
  } catch (e) { err = e; seg = st.seg; }
  const aborted = signal.aborted;
  if (seg && !seg.finish && aborted) seg.finish = 'cancelled';
  if (err && !aborted) seg = { ...(seg || newSegment()), error: friendlyError(err, { canCompact }) };
  if (seg && !aborted && !seg.error && !seg.done && !seg.finish) seg.error = { message: 'The stream ended before the reply was complete.', hint: 'The engine may have stopped; see “Engine & Model”.' };
  mark(seg);
  const hasAny = !!(seg && (seg.content || seg.reasoning || seg.toolCalls.length || seg.error || seg.timings));
  const segIndex = (getConv(convId).messages[targetId].segments || []).length;
  if (hasAny) {
    if (seg.toolCalls.length && !aborted) {   // an interface sent as a tool call is an interface all the same (lib/a2ui.js)
      const ad = adoptCalls(seg.content, seg.toolCalls, new Set(a2uiRt ? a2uiRt.surfaces.keys() : []));
      if (ad.moved) { seg.content = ad.content; seg.toolCalls = ad.calls; if (!ad.calls.length && seg.finish === 'tool_calls') seg.finish = 'stop'; }
    }
    if (seg.toolCalls.length) for (const c of seg.toolCalls) c.status = aborted ? 'skipped' : 'pending';
    const fin = finishSegment(seg, st);
    if (kind && fin.content) fin.content = SUMMARY_PREFILL + fin.content;
    delete fin.prefill;
    mutate(convId, (c, m) => ({ messages: update(m, targetId, (n) => ({ segments: [...(n.segments || []), fin] })) }));
  }
  if (hasAny && !hasDecodeProfile(st.rec)) pendingLive = { convId, msgId: targetId, seg: segIndex, job: st.job, until: Date.now() + 6000 };   // the engine's final record may land after the reply
  if (seg && seg.firstTokenAt != null) st.prevSeg = seg;   // the docked readout keeps this turn's numbers while tools run
  st.seg = newSegment(); st.kind = null;
  return { seg, aborted, segIndex, hasAny, overflow: !!(seg && seg.error && seg.error.overflow) };
}

function newStreamState(convId, msgId, ctl, snap) {
  const base = S().ctxBase;
  return { convId, msgId, seg: newSegment(), started: false, inReasoning: false, phase: 'sending', switchStep: null, rec: null, sections: null, abort: ctl,
    method: snap.method || defaultMethod(snap), kind: null, ctxUsed: base && base.convId === convId ? base.used : null };
}
// the stream ended: the ring keeps the last live number until the next count of the new state arrives (no flash back)
function endStream(convId, st) {
  set({ streaming: null, ...(S().conv.id === convId && st.ctxUsed != null ? { ctxBase: { convId, key: 'pending', used: st.ctxUsed, exact: false } } : {}) });
  scheduleSave(convId, 0); scheduleCtx(0);
}

async function run(convId, aid, { compactFirst = false } = {}) {
  const ctl = new AbortController();
  const snap = (getConv(convId).messages[aid] || {}).request || requestSnapshot();
  const st = newStreamState(convId, aid, ctl, snap);
  set({ streaming: st });
  const s = S().settings;
  let retried = false;
  try {
    for (let turn = 0; turn < Math.max(1, s.maxToolTurns); turn++) {
      // room for this pass: compact first when the projection passes the threshold (or "Compact and retry" asked for it)
      const room = await ensureRoom(st, convId, aid, { force: compactFirst });
      compactFirst = false;
      st.msgId = aid;
      if (room === 'aborted' || ctl.signal.aborted) { markStopped(convId, aid); break; }
      const { messages } = getConv(convId);
      const path = pathTo(messages, aid);
      const tools = collectTools();
      const body = api.buildBody(toApiMessages(path, apiOpts()), sendSettings(), tools.defs);
      const can = canCompactFor(convId, aid);
      const r = await streamPass(st, convId, aid, body, { canCompact: can });
      if (r.overflow && !r.aborted && !retried && can && S().settings.autoCompact) {   // the server said there is no room: compact once, send again
        retried = true; compactFirst = true;
        if (r.hasAny) mutate(convId, (c, m) => ({ messages: update(m, aid, (n) => ({ segments: n.segments.filter((_, i) => i !== r.segIndex) })) }));
        turn--; continue;
      }
      const seg = r.seg;
      if (r.aborted || !seg || seg.error || seg.finish !== 'tool_calls' || !seg.toolCalls.length) break;
      st.phase = 'running tools'; st.started = true; bump();
      await runTools(convId, aid, r.segIndex, tools.map, ctl.signal);
      if (ctl.signal.aborted) break;
    }
  } finally {
    endStream(convId, st);
  }
}

// ------------------------------------------------------------------------------------------------- compaction (docs/ui.md 10)
const apiOpts = () => ({ system: S().settings.system, sendReasoning: S().settings.sendReasoning });
const nCtxNow = () => { const s = S(); return (s.hello && s.hello.n_ctx) || (s.props && s.props.default_generation_settings && s.props.default_generation_settings.n_ctx) || 0; };
const canCount = () => !!(S().hello && S().hello.api && S().hello.api.count_tokens);
const outputSeg = (g) => !g.error && g.kind !== 'compaction' && !!(String(g.content || '').trim() || (g.toolCalls || []).length);

// The prompt size of `msgs` as the next request would send them: counted by the server (exact: same template, tools and
// image spans), else estimated (an older server without /gaudium/count_tokens).
async function countPrompt(msgs, defs, extra = null) {
  if (canCount()) {
    const b = api.buildBody(msgs, sendSettings(), defs); delete b.stream;
    if (extra) Object.assign(b, extra);
    try { const r = await api.countTokens(b); return { n: r.n_prompt_tokens, nCtx: r.n_ctx, exact: true }; } catch { /* estimate */ }
  }
  return { n: estimateMessages(msgs, defs), nCtx: nCtxNow(), exact: false };
}

// what a compaction for this turn would cover: the passes of a tool loop so far (in-turn), else the history before its message
function compactionTarget(messages, aid) {
  const a = messages[aid]; const u = a && messages[a.parent];
  const inTurn = !!a && (a.segments || []).some(outputSeg);
  return { inTurn, target: inTurn ? pathTo(messages, aid) : (u && u.parent ? pathTo(messages, u.parent) : []) };
}
function canCompactFor(convId, aid) { const x = getConv(convId); return !!x && compactableCount(compactionTarget(x.messages, aid).target) > 0; }

// Before a pass: 'ok' (room, or nothing to compact) | 'aborted'. force: compact whatever the projection says.
async function ensureRoom(st, convId, aid, { force = false } = {}) {
  const s = S().settings; const nCtx = nCtxNow();
  if (!force && (!s.autoCompact || !nCtx)) return 'ok';
  const { messages } = getConv(convId);
  const { inTurn, target } = compactionTarget(messages, aid);
  if (!compactableCount(target)) return 'ok';   // nothing to compact: send as is (the server says so if it does not fit)
  if (!force) {
    const percent = compactPercent(s), reserve = reserveTokens(s, nCtx), tools = collectTools();
    const msgs = toApiMessages(pathTo(messages, aid), apiOpts());
    // a cheap estimate first; the exact count only when it comes near the threshold (the count uploads the images too)
    if (estimateMessages(msgs, tools.defs) + reserve < 0.6 * ((nCtx * percent) / 100)) return 'ok';
    st.phase = 'sending'; bump();
    const c = await countPrompt(msgs, tools.defs);
    if (st.abort.signal.aborted) return 'aborted';
    if (!projection({ history: c.n, reserve, nCtx: c.nCtx || nCtx, percent }).over) return 'ok';
  }
  return compactInto(st, convId, aid, { inTurn, target, auto: !force });
}

function newCompaction(path, extra = {}) {
  return { id: uid('k'), role: 'compaction', segments: [], created: Date.now(), compacted: countMessages(path), request: requestSnapshot(), ...extra };
}
async function compactInto(st, convId, aid, { inTurn, target, auto }) {
  let targetId = aid;
  if (!inTurn) {   // a marker between the turns: above this turn's user message
    const c = newCompaction(target, { auto });
    mutate(convId, (cv, m) => ({ messages: insertAbove(m, m[aid].parent, c) }));
    targetId = c.id;
  }
  st.msgId = targetId;
  const r = await summarize(st, convId, targetId, target, { compacted: countMessages(target) });
  st.msgId = aid;
  return r.aborted ? 'aborted' : 'ok';
}

// Stream the summary of `srcPath` into `targetId`. The request is the conversation exactly as the chat sends it (same system,
// tools and template kwargs, so the engine reuses it as a cached prefix) + the instruction + an assistant opening it continues.
async function summarize(st, convId, targetId, srcPath, { focus = '', compacted = 0 } = {}) {
  const nCtx = nCtxNow(); const maxT = summaryMaxTokens(nCtx); const tools = collectTools();
  st.kind = 'compaction'; st.phase = 'sending'; bump();
  const cont = { continue_final_message: true, add_generation_prompt: false };
  const msgs = await fitForSummary(compactionMessages(srcPath, { ...apiOpts(), focus }), tools.defs, nCtx ? nCtx - maxT : Infinity, cont, st.abort.signal);
  if (st.abort.signal.aborted) { markStopped(convId, targetId); return { aborted: true }; }
  const body = api.buildBody(msgs, sendSettings(), tools.defs);
  Object.assign(body, cont, { max_tokens: maxT, gaudium_turn_prefill: false });   // its own turn is never reused: no background prefill
  return streamPass(st, convId, targetId, body, { kind: 'compaction', compacted, canCompact: false });
}

// The summarisation request must itself leave room for the summary: drop the reasoning first (the answers carry the facts),
// then shorten the longest texts in their middle. Data is not touched: only this one request is shortened.
async function fitForSummary(msgs, defs, budget, extra, signal) {
  let m = msgs; let c = await countPrompt(m, defs, extra);
  if (c.n <= budget || signal.aborted) return m;
  m = m.map((x) => { if (!x.reasoning_content) return x; const { reasoning_content: _r, ...y } = x; return y; });
  c = await countPrompt(m, defs, extra);
  for (let i = 0; i < 12 && c.n > budget && !signal.aborted; i++) {
    const k = longestText(m, 2); if (k < 0) break;
    m = m.map((x, j) => (j === k ? elideMessage(x, Math.max(0.2, (0.9 * budget) / c.n)) : x));
    c = await countPrompt(m, defs, extra);
  }
  return m;
}

function markStopped(convId, id) {
  const x = getConv(convId); if (!x || !x.messages[id] || (x.messages[id].segments || []).length) return;
  mutate(convId, (c, m) => ({ messages: update(m, id, { segments: [{ content: '', reasoning: '', toolCalls: [], finish: 'cancelled' }] }) }));
}

// Manual: "Compact conversation" (the ring's panel), /compact [focus], Ctrl+Shift+L
async function compactNow(focus = '') {
  if (S().streaming) { toast('A reply is streaming: wait for it or stop it, then compact.'); return; }
  const { conv, messages } = ensureSaved();
  const path = activePath(messages, conv.root);
  if (!compactableCount(path)) { toast(path.length ? 'Nothing new to compact since the last summary.' : 'Nothing to compact yet.'); return; }
  const c = newCompaction(path, { manual: true, ...(focus ? { focus } : {}) });
  commit(conv, addChild(messages, path[path.length - 1].id, c));
  const ctl = new AbortController();
  const st = newStreamState(conv.id, c.id, ctl, requestSnapshot());
  set({ streaming: st, expandCut: null });
  try { await summarize(st, conv.id, c.id, path, { focus, compacted: countMessages(path) }); }
  finally { endStream(conv.id, st); }
}

// ------------------------------------------------------------------------------------------------- the context ring
// The open chat's next turn before the draft: counted by the server for the shown path (+ an empty user message: the next
// turn's wrapper), cached by the path's state, stored with the chat (conv.ctx), else the last measured context.
const ctxCache = new Map();
let ctxTimer = 0, ctxSeq = 0;
const hashStr = (t) => { let h = 0; for (let i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) | 0; return (h >>> 0).toString(36); };
function ctxKeyOf(conv, path) {
  const leaf = path[path.length - 1]; const s = S().settings;
  return [conv.id, leaf ? leaf.id : '-', leaf ? (leaf.segments || []).length : 0, hashStr(s.system || ''), s.thinking, s.effort, s.sendReasoning ? 1 : 0, s.a2ui && a2uiOffered() ? 1 : 0,
    collectTools().defs.map((t) => t.function && t.function.name).join(',')].join('|');
}
function measuredContext(path) {
  for (let i = path.length - 1; i >= 0; i--) {
    const n = path[i];
    if (isCompaction(n)) { const g = nodeSummary(n); if (g) return estimateText(g.content) + estimateText(S().settings.system) + 120; continue; }
    if (n.role !== 'assistant') continue;
    const segs = n.segments || [];
    for (let j = segs.length - 1; j >= 0; j--) {
      const g = segs[j];
      if (g.kind === 'compaction') { if (usableSummary(g)) return estimateText(g.content) + 400; continue; }
      if (g.timings && !g.error) return contextUsed(g.timings);
    }
  }
  return 0;
}
function nextTurnMessages(path) {
  const msgs = toApiMessages(path, apiOpts());
  const last = msgs[msgs.length - 1];
  if (!last || last.role === 'assistant' || last.role === 'system') msgs.push({ role: 'user', content: '' });
  return msgs;
}
function scheduleCtx(ms = 150) { clearTimeout(ctxTimer); ctxTimer = setTimeout(() => { refreshCtx().catch(() => { /* the ring keeps its value */ }); }, ms); }
async function refreshCtx() {
  const s = S();
  if (s.streaming && s.streaming.convId === s.conv.id) return;
  const { conv, messages } = current();
  const path = activePath(messages, conv.root);
  const key = ctxKeyOf(conv, path);
  const setBase = (used, exact) => { const b = S().ctxBase; if (!b || b.key !== key || b.used !== used || b.exact !== exact) set({ ctxBase: { convId: conv.id, key, used, exact } }); };
  if (!path.length) { setBase(0, true); return; }
  const hit = ctxCache.get(key) || (conv.ctx && conv.ctx.key === key ? { used: conv.ctx.used, exact: true } : null);
  if (hit) { ctxCache.set(key, hit); setBase(hit.used, hit.exact); return; }
  setBase(measuredContext(path), false);
  const seq = ++ctxSeq;
  const r = await countPrompt(nextTurnMessages(path), collectTools().defs);
  if (seq !== ctxSeq || S().conv.id !== conv.id || (S().streaming && S().streaming.convId === conv.id) || !r.exact) return;
  ctxCache.set(key, { used: r.n, exact: true });
  setBase(r.n, true);
  mutate(conv.id, (c) => ({ conv: { ...c, ctx: { key, used: r.n } } }), { now: false });
}
const countDraft = async (text) => { try { const r = await api.tokenize(text); return Array.isArray(r.tokens) ? r.tokens.length : null; } catch { return null; } };

async function runTools(convId, aid, segIndex, map, signal) {
  const calls = getConv(convId).messages[aid].segments[segIndex].toolCalls;
  const patch = (k, p) => mutate(convId, (c, m) => ({ messages: update(m, aid, (n) => ({ segments: n.segments.map((g, i) => (i !== segIndex ? g : { ...g, toolCalls: g.toolCalls.map((x, j) => (j === k ? { ...x, ...p } : x)) })) })) }));
  for (let k = 0; k < calls.length; k++) {
    if (signal.aborted) { patch(k, { status: 'skipped' }); continue; }
    const c = calls[k]; const prov = map.get(c.name);
    if (!prov) { patch(k, { status: 'error', error: callIsA2ui(c.name) ? 'The interface could not be read: pass surfaceId and components (a JSON array of component objects, one with the id "root").' : `No connected tool is called ${c.name}.` }); continue; }
    const server = prov.server || 'built-in';
    patch(k, { server });
    let args = {};
    try { args = c.arguments ? JSON.parse(c.arguments) : {}; } catch { patch(k, { status: 'error', error: 'The model wrote arguments that are not valid JSON.' }); continue; }
    const key = `${server}/${c.name}`;
    if (S().settings.askTools && !S().settings.allowedTools.includes(key)) {
      patch(k, { status: 'ask' });
      const choice = await new Promise((resolve) => { asks.set(c.id || `${segIndex}-${k}`, resolve); signal.addEventListener('abort', () => resolve('abort'), { once: true }); });
      if (choice === 'abort') { patch(k, { status: 'skipped' }); continue; }
      if (choice === 'deny') { patch(k, { status: 'denied' }); continue; }
      if (choice === 'always') setSetting('allowedTools', [...S().settings.allowedTools, key]);
    }
    patch(k, { status: 'running' });
    const t0 = performance.now();
    try {
      if (prov.builtin) patch(k, { status: 'done', result: await prov.builtin.run(args), ms: performance.now() - t0 });
      else { const r = await prov.client.call(c.name, args, signal); patch(k, { status: r.isError ? 'error' : 'done', result: r.text, error: r.isError ? r.text || 'the tool reported an error' : undefined, ms: performance.now() - t0 }); }
    } catch (e) { patch(k, { status: 'error', error: String(e.message || e), ms: performance.now() - t0 }); }
  }
}

// ------------------------------------------------------------------------------------------------- user actions
function ensureSaved() { const { conv, messages } = current(); cache.set(conv.id, { conv, messages }); return { conv, messages }; }
function send(text, atts) {
  if (S().streaming) return;
  const cmd = /^\/compact(?:\s+([\s\S]*))?$/.exec(String(text || '').trim());   // /compact [what to keep in particular]
  if (cmd && !atts.length) { compactNow((cmd[1] || '').trim()); return; }
  let { conv, messages } = ensureSaved();
  const path = activePath(messages, conv.root);
  const parent = path.length ? path[path.length - 1].id : conv.root;
  const u = { id: uid('u'), role: 'user', content: text, attachments: atts, created: Date.now() };
  messages = addChild(messages, parent, u);
  if (conv.title === 'New chat') conv = { ...conv, title: titleFrom(text || (atts[0] && atts[0].name)) };
  const a = newAssistant(); messages = addChild(messages, u.id, a);
  commit(conv, messages);
  try { localStorage.setItem('gaudium-ui.last', conv.id); } catch { /* no storage */ }
  run(conv.id, a.id);
}
// The interfaces (A2UI surfaces) of the open chat: the blocks of the shown path, applied in order (lib/a2ui.js).
let a2uiRt = null;
function a2uiFor(path, st) {
  a2uiRt = syncRuntime(a2uiRt, blocksOfPath(path, st ? { id: st.msgId, seg: st.seg } : null));
  return a2uiRt;
}
const act = {
  // a pressed button of an interface, and a report of what was wrong with one: each is a user turn the model answers
  a2uiSend: (msg) => send(wrapMessage(msg), []),
  a2uiReport: (errors) => send(errors.slice(0, 6).map((e) => wrapMessage(errorMessage(e))).join('\n'), []),
  copy: async (t) => { try { await navigator.clipboard.writeText(t); toast('Copied.'); } catch { toast('The browser refused the clipboard.'); } },
  nav: (id, d) => { const { conv, messages } = current(); commit(conv, switchSibling(messages, id, d), { now: false }); },
  regenerate: (aid) => {
    if (S().streaming) return;
    const { conv, messages } = current(); const n = messages[aid]; if (!n) return;
    const a = newAssistant(); commit(conv, addChild(messages, n.parent, a)); run(conv.id, a.id);
  },
  editUser: (id, text) => {
    if (S().streaming) return;
    const { conv, messages } = current(); const n = messages[id]; if (!n) return;
    const u = { id: uid('u'), role: 'user', content: text, attachments: n.attachments || [], created: Date.now() };
    let m = addChild(messages, n.parent, u); const a = newAssistant(); m = addChild(m, u.id, a);
    commit(conv, m); run(conv.id, a.id);
  },
  editAssistant: (id, text) => {
    const { conv, messages } = current(); const n = messages[id]; if (!n) return;
    const a = { id: uid('a'), role: 'assistant', created: Date.now(), request: n.request, segments: [{ content: text, reasoning: '', toolCalls: [], finish: 'stop', edited: true }] };
    commit(conv, addChild(messages, n.parent, a));
  },
  remove: (id) => set({ confirm: { title: 'Delete this message?', body: 'It is removed with every reply and branch below it.', confirm: 'Delete', danger: true, onConfirm: () => { const { conv, messages } = current(); commit(conv, removeSubtree(messages, id)); } } }),
  toolAnswer,
  // after a 400 exceed_context_size_error: drop the failed pass, compact (above this turn, or the tool loop so far), run again
  compactRetry: (aid) => {
    if (S().streaming) return;
    const { conv, messages } = current(); const a = messages[aid]; if (!a) return;
    const segs = a.segments || []; let k = segs.length;
    while (k > 0 && !outputSeg(segs[k - 1]) && segs[k - 1].kind !== 'compaction') k--;
    commit(conv, update(messages, aid, { segments: segs.slice(0, k) }));
    run(conv.id, aid, { compactFirst: true });
  },
  undoCompact: (id) => set({ confirm: { title: 'Undo this compaction?', body: 'The summary is removed and the requests send the whole conversation again. The messages after it stay where they are.', confirm: 'Undo compaction', onConfirm: () => { const { conv, messages } = current(); commit(conv, spliceOut(messages, id)); } } }),
  toggleCut: (id) => set({ expandCut: S().expandCut === id ? null : id }),
  skipReasoning: async () => { const st = S().streaming; if (!st || !st.seg || !st.seg.id) return; try { const r = await api.skipReasoning(st.seg.id); if (!r.success) toast(r.message || 'Could not skip the reasoning.'); } catch (e) { toast(e.message); } },
};
function stop() { const st = S().streaming; if (st) st.abort.abort(); }

const railAct = {
  newChat, open: openConv,
  del: (id) => set({ confirm: { title: 'Delete this chat?', body: 'It is removed from this browser. Export it first to keep a copy.', confirm: 'Delete chat', danger: true, onConfirm: async () => {
    await db.deleteConversation(id); cache.delete(id); if (S().conv.id === id) newChat(); refreshList(); } } }),
  rename: (id, title) => { const x = getConv(id) || { conv: S().convs.find((c) => c.id === id) }; if (!x || !x.conv) return;
    if (x.messages) commit({ ...x.conv, title }, x.messages); else db.loadTree(id).then((m) => { cache.set(id, { conv: { ...x.conv, title }, messages: m }); scheduleSave(id, 0); }); },
  renaming: null, renamed: () => set({ renaming: null }),
  importFile: async (f) => { try { const n = await db.importConversations(JSON.parse(await f.text())); await refreshList(); toast(`Imported ${n} chat${n === 1 ? '' : 's'}.`); } catch (e) { toast(e.message || 'That file could not be imported.'); } },
  exportAll: async () => { download(`gaudium-chats-${stamp()}.json`, await db.exportConversations()); },
  openCard: () => set({ dialog: 'card' }), openSettings: () => set({ dialog: 'settings' }),
};
async function exportCurrent() {
  const { conv } = current(); if (!S().saved) { toast('Nothing to export yet.'); return; }
  scheduleSave(conv.id, 0); await new Promise((r) => setTimeout(r, 60));
  download(`gaudium-${conv.title.replace(/[^\w-]+/g, '-').slice(0, 40)}-${stamp()}.json`, await db.exportConversations([conv.id]));
}

// ------------------------------------------------------------------------------------------------- server + events
async function loadProps() {
  for (let i = 0; ; i++) {
    try { const p = await api.getProps(); set({ props: p, propsError: null, needKey: false }); return; }
    catch (e) {
      if (e.status === 401 || e.status === 403) { set({ needKey: true }); return; }
      set({ propsError: e.status === 503 ? 'The server is loading the model…' : `Cannot reach the server (${e.message}).` });
      await new Promise((r) => setTimeout(r, Math.min(5000, 1000 + i * 500)));
    }
  }
}
let events = null;
const hasDecodeProfile = (r) => !!(r && r.profile && r.profile.hbm);
let pendingLive = null;   // {convId, msgId, seg, job, until}: a finished reply without a profile yet
function onEvent(ev) {
  const st = S().streaming;
  switch (ev.type) {
    case 'hello': set({ hello: ev }); break;
    case 'engine': set({ engine: ev, engineAt: Date.now() }); break;
    case 'live':
      set({ live: ev });
      // a record counts for a reply only when it carries the decode profile (HBM flow, step split): the prefill-phase record, and the
      // record of the background turn re-render that follows a reply, have a prefill-only profile and used to overwrite the good one
      if (st && ev.record && (st.job == null || ev.job === st.job) && (hasDecodeProfile(ev.record) || !hasDecodeProfile(st.rec))) { st.rec = ev.record; bump(); }
      else if (pendingLive && hasDecodeProfile(ev.record) && (pendingLive.job == null || ev.job === pendingLive.job) && Date.now() < pendingLive.until) {   // the engine's record arrived after the reply ended
        const p = pendingLive; pendingLive = null;
        mutate(p.convId, (c, m) => ({ messages: update(m, p.msgId, (n) => ({ segments: n.segments.map((g, i) => (i === p.seg && !hasDecodeProfile(g.live) ? { ...g, live: ev.record } : g)) })) }), { now: false });
      }
      break;
    case 'switch': if (st && (st.job == null || ev.job === st.job)) { st.switchStep = ev.step; bump(); } break;
    // running totals of the request's optional sections (moe, ple, qsa; docs/server.md 7.8): the readout's model panel follows them
    case 'sections': if (st && (st.job == null || ev.job === st.job)) { st.sections = ev; bump(); } break;
    case 'request': if (st && !ev.background && (ev.phase === 'queued' || ev.phase === 'prefill')) st.job = ev.job; break;
    default: break;
  }
}

function onKey(e) {
  const mod = e.ctrlKey || e.metaKey; const k = e.key.toLowerCase();
  const typing = /^(input|textarea|select)$/i.test((e.target && e.target.tagName) || '');
  if (mod && e.shiftKey && k === 'o') { e.preventDefault(); newChat(); }
  else if (mod && !e.shiftKey && k === 'k') { e.preventDefault(); set({ railOpen: true, searchKey: S().searchKey + 1 }); }
  else if (mod && k === ',') { e.preventDefault(); set({ dialog: 'settings' }); }
  else if (mod && e.shiftKey && k === 'e') { e.preventDefault(); if (S().saved) set({ railOpen: true, renaming: S().conv.id }); }
  else if (mod && e.shiftKey && k === 's') { e.preventDefault(); exportCurrent(); }
  else if (mod && e.shiftKey && k === 'd') { e.preventDefault(); if (S().saved) railAct.del(S().conv.id); }
  else if (mod && e.shiftKey && k === 'g') { e.preventDefault(); set({ dialog: 'card' }); }
  else if (mod && e.shiftKey && k === 'l') { e.preventDefault(); compactNow(); }
  else if (mod && !e.shiftKey && k === 'b') { e.preventDefault(); set({ railOpen: !S().railOpen }); }
  else if (mod && (k === '/' || k === '?')) { e.preventDefault(); set({ dialog: 'shortcuts' }); }
  else if (!mod && !typing && k === '/' && !S().dialog) { e.preventDefault(); set({ focusKey: S().focusKey + 1 }); }
  else if (k === 'escape' && S().streaming && !S().dialog) { stop(); }
}

export async function boot() {
  document.addEventListener('keydown', onKey);
  onLibsReady(() => set({ tick: S().tick + 1, libs: (S().libs || 0) + 1 }));
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  mq.addEventListener('change', () => set({ themeKey: effectiveDark() ? 'd' : 'l' }));
  setInterval(() => { if (S().streaming || (S().engine && S().engine.state !== 'ready') || S().dialog === 'card') set({ now: Date.now() }); }, 1000);
  document.addEventListener('click', (e) => {   // code-block copy buttons (rendered markdown)
    const b = e.target.closest && e.target.closest('[data-copy]'); if (!b) return;
    const code = b.closest('.code').querySelector('code');
    navigator.clipboard.writeText(code ? code.innerText : '').then(() => { b.textContent = 'Copied'; setTimeout(() => { b.textContent = 'Copy'; }, 1200); }, () => toast('The browser refused the clipboard.'));
  });
  events = api.connectEvents(onEvent, (state) => set({ events: state }));
  loadProps();
  const list = await db.listConversations();
  set({ convs: list });
  let last = null; try { last = localStorage.getItem('gaudium-ui.last'); } catch { last = null; }
  if (last && list.find((c) => c.id === last)) openConv(last);
  for (const srv of S().settings.mcpServers) if (srv.enabled) mcpConnect(srv);
}

// ------------------------------------------------------------------------------------------------- views
// what the readout will show for this model, in one sentence (from its description; the older wording when there is none)
function helloText(desc) {
  const parts = ['tokens per second'];
  if (!desc.known || desc.drafters.length) parts.push('tokens per decode step and how many drafts were accepted', desc.known && !hasDrafter(desc, 'df') && !hasDrafter(desc, 'ds') ? 'which head proposes them' : 'which drafter is proposing them');
  if (desc.moe) parts.push(desc.moe.placement && desc.moe.placement.mode !== 'card' ? 'where the experts run (card or host) and the host work per step' : 'where the experts live');
  parts.push("how much of the card's HBM bandwidth each step uses");
  return `Every reply shows what the engine is doing while it writes: ${parts.slice(0, -1).join(', ')}, and ${parts[parts.length - 1]}.`;
}
function Hello({ hello, props, onPick, desc }) {
  const ei = engineInfo();
  const n = hello ? hello.n_ctx : props && props.default_generation_settings && props.default_generation_settings.n_ctx;
  const starters = [
    ['Explain speculative decoding to a new engineer', 'with a small worked example'],
    ['Write a Python function that merges overlapping intervals', hasDrafter(desc, 'df') ? `code: ${dfFamilyName(desc)} drafts run long here` : 'code'],
    ['Tell me a short fable about a lighthouse keeper', 'story'],
    ['Derive the closed form of the sum 1 + 2 + … + n', 'math, rendered'],
  ];
  const facts = modelFacts(desc);
  return html`<div class="hello">
    <h1>${desc.name || 'The model'} on one Gaudi 2, with an engine built for the chip.</h1>
    <p>${helloText(desc)}</p>
    <div class="facts">${hello ? html`${ei.precision || desc.precision.weights ? html`<span>${precisionShort(desc, ei.precision) || ei.precision}</span>` : null}<span>${fmtInt(n)} tokens of context</span>${ei.greedy ? html`<span>greedy: ${methodInfo(ei.greedy).label}</span>` : null}${ei.sampled ? html`<span>sampled: ${methodInfo(ei.sampled).label}</span>` : null}${facts.map((f) => html`<span>${f}</span>`)}${hello.vision ? html`<span>vision on</span>` : null}${hello.backend === 'mock' ? html`<span>mock replay</span>` : null}` : null}</div>
    <div class="starters">${starters.map(([t, s]) => html`<button class="starter" onClick=${() => onPick(t)}>${t}<small>${s}</small></button>`)}</div>
  </div>`;
}

function CardPill({ engine, hello, events, onClick, now, engineAt, precision }) {
  const e = engine || {};
  const st = e.state || (events === 'error' ? 'offline' : 'connecting');
  const elapsed = Math.round((e.state_for_s || 0) + Math.max(0, (now - (engineAt || now)) / 1000));
  const text = { ready: e.mock && !e.available ? 'mock replay' : 'card ready', parked: 'card parked', queued: 'waiting for the card', loading: `warming up ${elapsed} s`, stopping: 'releasing the card', failed: 'engine failed', offline: 'no event stream', connecting: 'connecting' }[st] || st;
  const cls = st === 'ready' ? 'ok' : st === 'failed' || st === 'offline' ? 'bad' : 'warn';
  return html`<button class="cardpill" onClick=${onClick} title="Engine & Model (Ctrl+Shift+G)">
    <span class=${cx('dot', cls, (st === 'queued' || st === 'loading' || st === 'stopping') && 'pulse')}></span>
    <span class="label">${text}</span>${precision && st === 'ready' ? html`<span class="mono">${precision}</span>` : null}
  </button>`;
}

export function App() {
  const [, force] = useState(0);
  useEffect(() => store.sub(() => force((x) => x + 1)), []);
  const s = S();
  const scroller = useRef(null);
  const path = activePath(s.messages, s.conv.root);
  const st = s.streaming && s.streaming.convId === s.conv.id ? s.streaming : null;
  const busy = !!s.streaming;
  const ei = engineInfo();
  const desc = useMemo(() => modelDesc(s.hello, s.engine), [s.hello, s.engine && s.engine.info]);
  setDfName(dfFamilyName(desc));   // the df family's name in every method label below (DFlash or DFlash2: the drafter this engine loaded)
  const precision = precisionShort(desc, ei.precision);
  // only what the loaded engine runs; before it reports, the saved choice stays listed so the picker keeps its value
  const methods = useMemo(() => {
    const l = loadedMethods();
    if (l) return l;
    return s.settings.method && s.settings.method !== 'default' ? [s.settings.method] : [];
  }, [ei.loaded, s.settings.method]);
  const effSet = useMemo(() => effSettings(s.settings), [s.settings, ei.loaded]);
  const efforts = (s.props && s.props.chat_template_caps && s.props.chat_template_caps.reasoning_effort_levels) || ['low', 'medium', 'xhigh'];
  const vision = !s.props || !s.props.modalities || s.props.modalities.vision !== false;
  const audio = !!(s.props && s.props.modalities && s.props.modalities.audio);   // only when the server advertises audio support
  const video = vision && !!(s.props && s.props.modalities && s.props.modalities.video);   // animations and video files as video parts
  const limits = useMemo(() => { setVisionLimits(s.props); return visionLimits(s.props); }, [s.props]);
  const a2ui = a2uiFor(path, st);
  const view = useMemo(() => ({ settings: s.settings, precision, desc, themeKey: s.themeKey + (s.libs || 0), mock: s.hello && s.hello.backend === 'mock', defaultMethod, nCtx: s.hello ? s.hello.n_ctx : 0, a2ui }), [s.settings, s.hello, s.themeKey, s.libs, precision, desc, a2ui]);
  const liveView = st ? { ...view, engine: s.engine, engineAt: s.engineAt, now: s.now, engineReady: !s.engine || s.engine.state === 'ready', canSkip: !!(s.hello ? s.hello.supports_inject : true) } : view;
  // the chat follows the stream while the reader is at the bottom (a new chat or a new turn starts there)
  const streamedHere = useRef(false);   // the open chat had a stream in the last render
  const glide = streamedHere.current && !st; streamedHere.current = !!st;
  const onScroll = useFollow(scroller, { key: `${s.conv.id}:${path.length}`, glide });
  const toolCount = collectTools().defs.length;
  // the context ring: live from the stream, else the counted next turn of this chat (+ the draft, added by the composer)
  const nCtx = nCtxNow();
  const ctxKey = ctxKeyOf(s.conv, path);
  useEffect(() => { if (!st) scheduleCtx(); }, [ctxKey, !!st, s.hello && s.hello.api && s.hello.api.count_tokens]);
  const base = s.ctxBase && s.ctxBase.convId === s.conv.id ? s.ctxBase : null;
  // (while streaming in steps of 0.1 % of the context: the same arc to the eye, far fewer DOM writes than one per chunk)
  const liveStep = Math.max(1, Math.ceil(nCtx / 1000));
  const ring = { nCtx, used: st && st.ctxUsed != null ? Math.floor(st.ctxUsed / liveStep) * liveStep : base ? base.used : 0, live: !!st, exact: !!(base && base.exact),
    canCompact: !busy && compactableCount(path) > 0, auto: s.settings.autoCompact, at: compactPercent(s.settings) };
  const cutAt = displayCut(path);
  const cutId = cutAt > 0 ? path[cutAt].id : null;
  const expanded = cutId != null && s.expandCut === cutId;
  const renderMsg = (m) => {
    const sib = siblings(s.messages, m.id);
    return m.role === 'user' ? html`<${UserMessage} key=${m.id} m=${m} sib=${sib} act=${act} busy=${busy} />`
      : html`<${AssistantMessage} key=${m.id} m=${m} sib=${sib} act=${act} busy=${busy} live=${st && st.msgId === m.id ? { ...st } : null} view=${st && st.msgId === m.id ? liveView : view} />`;
  };
  return html`<div class=${cx('shell', s.railOpen ? 'rail-open' : 'rail-closed')}>
    <${Sidebar} convs=${s.convs} curId=${s.saved ? s.conv.id : null} query=${s.query} setQuery=${(q) => set({ query: q })} act=${{ ...railAct, renaming: s.renaming }} searchKey=${s.searchKey} />
    <div class="scrim" onClick=${() => set({ railOpen: false })}></div>
    <main class="main">
      <header class="topbar">
        <button class="icon-btn" title="Sidebar (Ctrl+B)" aria-label="Toggle the sidebar" onClick=${() => set({ railOpen: !s.railOpen })}><${Icon} name="sidebar" /></button>
        <div class="title">${s.saved ? s.conv.title : 'New chat'}<small>${s.hello ? s.hello.model : ''}</small></div>
        <${CardPill} engine=${s.engine} hello=${s.hello} events=${s.events} now=${s.now} engineAt=${s.engineAt} precision=${precision} onClick=${() => set({ dialog: 'card' })} />
        <button class="icon-btn" title=${`Theme: ${s.settings.theme}`} aria-label="Switch theme" onClick=${() => setSetting('theme', s.settings.theme === 'system' ? (effectiveDark() ? 'light' : 'dark') : s.settings.theme === 'dark' ? 'light' : 'dark')}><${Icon} name=${effectiveDark() ? 'sun' : 'moon'} /></button>
        <button class="icon-btn" title="New chat (Ctrl+Shift+O)" aria-label="New chat" onClick=${newChat}><${Icon} name="plus" /></button>
      </header>
      <div class="scroller" ref=${scroller} onScroll=${onScroll}>
        <div class="column">
          ${s.propsError ? html`<div class="err"><b>Server</b>: ${s.propsError}</div>` : null}
          ${!path.length ? html`<${Hello} hello=${s.hello} props=${s.props} desc=${desc} onPick=${(t) => send(t, [])} />` : null}
          ${cutId ? html`<div class="compact-divider" key="cut-divider"><button class="compact-toggle" aria-expanded=${expanded} onClick=${() => act.toggleCut(cutId)}
              title=${expanded ? 'Fold the earlier messages again' : 'Show the messages the summary replaces (they stay in this chat; requests send only the summary)'}>
              <${Icon} name=${expanded ? 'up' : 'down'} />${countMessages(path.slice(0, cutAt))} earlier messages compacted<span class="sep">·</span>${expanded ? 'hide' : 'show'}</button></div>` : null}
          ${cutId && expanded ? html`<div class="compacted" key="cut-earlier">${path.slice(0, cutAt).map(renderMsg)}</div>` : null}
          ${(cutId ? path.slice(cutAt) : path).map(renderMsg)}
        </div>
      </div>
      ${st && s.settings.telemetry ? html`<${LiveDock} live=${{ ...st }} m=${s.messages[st.msgId]} view=${liveView} collapsed=${s.dockCollapsed}
        onToggle=${() => { saveDockCollapsed(!s.dockCollapsed); set({ dockCollapsed: !s.dockCollapsed }); }} />` : null}
      <${Composer} settings=${effSet} defaultMethodName=${defaultMethod(effSet)} setSetting=${setSetting} busy=${busy} onSend=${send} onStop=${stop} methods=${methods} efforts=${efforts} vision=${vision} audio=${audio} video=${video} limits=${limits}
        greedyOnlyList=${(s.hello && s.hello.methods && s.hello.methods.greedy_only) || null} methodPlan=${methodPlan(s.settings)} docked=${!!(st && s.settings.telemetry && !s.dockCollapsed)}
        toolCount=${toolCount} notify=${toast} focusKey=${s.focusKey} draftKey=${s.conv.id} modelName=${desc.name}
        ring=${ring} countDraft=${countDraft} onCompact=${() => compactNow()} openContextSettings=${() => set({ dialog: 'settings', settingsTab: 'context' })} />
    </main>
    ${s.dialog === 'settings' ? html`<${SettingsDialog} tab0=${s.settingsTab} settings=${effSet} vision=${vision} video=${video} set=${setSetting} onClose=${() => set({ dialog: null, settingsTab: null })} props=${s.props} methods=${methods} efforts=${efforts}
        greedyMethod=${ei.greedy} sampledMethod=${ei.sampled}
        greedyOnlyList=${(s.hello && s.hello.methods && s.hello.methods.greedy_only) || null}
        mcp=${s.mcp} proxyEnabled=${!!(s.props && s.props.cors_proxy_enabled)}
        mcpAct=${{ connect: mcpConnect, deleteAll: () => set({ confirm: { title: 'Delete all chats?', body: 'Every chat in this browser is removed. Export them first to keep a copy.', confirm: 'Delete all', danger: true, onConfirm: async () => { await db.deleteAll(); cache.clear(); newChat(); refreshList(); } } }) }} />` : null}
    ${s.dialog === 'card' ? html`<${CardDialog} engine=${s.engine} hello=${s.hello} live=${s.live} now=${s.now} engineAt=${s.engineAt} desc=${desc} props=${s.props} onClose=${() => set({ dialog: null })}
        onAction=${async (a, m) => { try { const r = await api.engineAction(a, m); toast(r.message || a); } catch (e) { toast(e.message); } }} />` : null}
    ${s.dialog === 'shortcuts' ? html`<${ShortcutsDialog} onClose=${() => set({ dialog: null })} />` : null}
    ${s.confirm ? html`<${ConfirmDialog} ...${s.confirm} onClose=${() => set({ confirm: null })} />` : null}
    ${s.needKey ? html`<${KeyDialog} error=${s.keyError} onSave=${(k) => { setSetting('apiKey', k); set({ needKey: false, keyError: '' }); loadProps(); if (events) events.reopen(); }} />` : null}
    ${s.toast ? html`<div class="toast" role="status">${s.toast}</div>` : null}
  </div>`;
}
