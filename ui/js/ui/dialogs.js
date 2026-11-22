// Dialogs: settings (incl. MCP servers), the engine and card, keyboard shortcuts, confirmation, the API key.
import { html, cx } from './h.js';
import { useState, useEffect, useRef } from 'preact/hooks';
import { Icon } from './icons.js';
import { methodInfo, greedyOnly, fmtInt, fmt1, fmtGBs } from '../lib/stats.js';
import { placementOf, drafterNames } from '../lib/model.js';
import { uid } from '../lib/tree.js';

function Dialog({ title, onClose, children, foot, narrow, tabs }) {
  const ref = useRef(null);
  useEffect(() => {
    const prev = document.activeElement;
    const f = ref.current && ref.current.querySelector('input, select, textarea, button:not(.dlg-close)');
    if (f) f.focus();
    const k = (e) => { if (e.key === 'Escape') { e.stopPropagation(); onClose(); } };
    window.addEventListener('keydown', k, true);
    return () => { window.removeEventListener('keydown', k, true); if (prev && prev.focus) prev.focus(); };
  }, []);
  return html`<div class="overlay" onMouseDown=${(e) => { if (e.target === e.currentTarget) onClose(); }}>
    <div class=${cx('dialog', narrow && 'narrow')} role="dialog" aria-modal="true" aria-label=${title} ref=${ref}>
      <div class="dlg-head"><h2>${title}</h2><button class="icon-btn dlg-close" title="Close (Esc)" onClick=${onClose}><${Icon} name="x" /></button></div>
      ${tabs || null}
      <div class="dlg-body">${children}</div>
      ${foot ? html`<div class="dlg-foot">${foot}</div>` : null}
    </div>
  </div>`;
}

const Field = ({ label, help, children, id }) => html`<div class="field">${label ? html`<label for=${id}>${label}</label>` : null}${children}${help ? html`<div class="help">${help}</div>` : null}</div>`;

function Seg({ value, options, onChange, label }) {
  return html`<div class="seg-ctl" role="radiogroup" aria-label=${label}>${options.map(([v, t]) => html`<button type="button" role="radio" aria-checked=${value === v} class=${value === v ? 'on' : ''} onClick=${() => onChange(v)}>${t}</button>`)}</div>`;
}

export function SettingsDialog({ settings, vision = true, set, onClose, props, methods, efforts, mcp, mcpAct, proxyEnabled, tab0, greedyOnlyList, greedyMethod, sampledMethod }) {
  const [tab, setTab] = useState(tab0 || 'general');
  const s = settings;
  const def = (props && props.default_generation_settings && props.default_generation_settings.params) || {};
  const num = (k, label, help, step = 'any', ph) => html`<${Field} id=${'s-' + k} label=${label} help=${help}>
    <input id=${'s-' + k} type="number" step=${step} value=${s[k]} placeholder=${ph != null ? `server default: ${ph}` : 'server default'} onInput=${(e) => set(k, e.target.value)} /></${Field}>`;
  let customOk = true; if (s.custom.trim()) { try { JSON.parse(s.custom); } catch { customOk = false; } }
  const T = [['general', 'General'], ['sampling', 'Sampling'], ['reasoning', 'Reasoning'], ['context', 'Context'], ['tools', 'Tools and MCP'], ['advanced', 'Advanced']];
  const tabs = html`<div class="tabs" role="tablist">${T.map(([k, t]) => html`<button role="tab" class=${cx('tab', tab === k && 'on')} aria-selected=${tab === k} onClick=${() => setTab(k)}>${t}</button>`)}</div>`;
  return html`<${Dialog} title="Settings" onClose=${onClose} tabs=${tabs} foot=${html`<span class="notice" style="margin-right:auto">Saved in this browser as you change them.</span><button class="btn primary" onClick=${onClose}>Done</button>`}>
    ${tab === 'general' ? html`
      <${Field} id="s-system" label="System prompt" help="Sent first in every chat. Leave empty for none.">
        <textarea id="s-system" rows="4" value=${s.system} onInput=${(e) => set('system', e.target.value)} style="font-family:var(--font-ui);font-size:14px"></textarea></${Field}>
      <div class="field"><span class="lbl">Theme</span><${Seg} label="Theme" value=${s.theme} onChange=${(v) => set('theme', v)} options=${[['system', 'System'], ['light', 'Light'], ['dark', 'Dark']]} /></div>
      <div class="grid2">
        ${vision ? html`<${Field} id="s-pdf" label="PDFs" help="Text uses the PDF's text layer; images send up to 8 pages to the vision tower (a scan has no text layer and always goes as images).">
          <select id="s-pdf" value=${s.pdfMode} onChange=${(e) => set('pdfMode', e.target.value)}><option value="text">as text</option><option value="images">as page images</option></select></${Field}>`
          : html`<div class="field"><span class="lbl">PDFs</span><div class="help">Sent as text: the loaded model has no vision.</div></div>`}
        ${num('pasteToFile', 'Long paste becomes a file at', 'Characters; 0 keeps every paste in the message.', '1')}
      </div>
      <label class="check"><input type="checkbox" checked=${s.telemetry} onChange=${(e) => set('telemetry', e.target.checked)} /><span>Show the engine readout while a reply streams (speed, tokens per step, HBM flow, the step split)</span></label>
      <${Field} id="s-key" label="API key" help="Only if the server was started with --api-key. Kept in this browser.">
        <input id="s-key" type="password" value=${s.apiKey} autocomplete="off" onInput=${(e) => set('apiKey', e.target.value)} /></${Field}>` : null}
    ${tab === 'sampling' ? html`
      <p class="notice" style="margin:0">Empty fields use the server's defaults (the model's generation_config: T ${def.temperature ?? 1}, top-k ${def.top_k ?? 20}, top-p ${def.top_p ?? 0.95}). Temperature 0 is greedy: the engine then uses its greedy method${greedyMethod ? ` (${methodInfo(greedyMethod).label})` : ''}, otherwise its sampled method${sampledMethod ? ` (${methodInfo(sampledMethod).label})` : ''}.</p>
      <div class="grid2">
        ${num('temperature', 'Temperature', '0 = greedy.', '0.05', def.temperature)}
        ${num('top_k', 'Top-k', '0 = off.', '1', def.top_k)}
        ${num('top_p', 'Top-p', '1 = off.', '0.01', def.top_p)}
        ${num('min_p', 'Min-p', '0 = off.', '0.01', def.min_p)}
        ${num('seed', 'Seed', 'Empty = a new random seed per message. The same seed gives the same reply, with or without speculation.', '1')}
        ${num('max_tokens', 'Max tokens', 'Empty = until the context is full.', '1')}
      </div>
      <${Field} id="s-method" label="Speculative method" help="Which of the engine's loaded runs serves the request (gaudium_method). The default picks the greedy or the sampled default. Greedy-only runs (external drafters, trees, auto) run greedy (temperature 0) when picked, shown by a “greedy” chip next to the method.">
        ${methods.length > 1 ? html`<select id="s-method" value=${s.method} onChange=${(e) => set('method', e.target.value)}><option value="default">engine default</option>${methods.map((m) => html`<option value=${m}>${methodInfo(m).label} (${m})${greedyOnly(m, greedyOnlyList) ? ' · greedy only' : ''}</option>`)}</select>` : html`<div class="notice">This model's engine runs one method: ${methodInfo(methods[0] || greedyMethod || sampledMethod || '').label}.</div>`}</${Field}>` : null}
    ${tab === 'reasoning' ? html`
      <div class="field"><span class="lbl">Thinking</span><${Seg} label="Thinking" value=${s.thinking} onChange=${(v) => set('thinking', v)} options=${[['on', 'On'], ['off', 'Off']]} /><div class="help">Off renders the template with enable_thinking: false.</div></div>
      <div class="field"><span class="lbl">Effort</span><${Seg} label="Effort" value=${s.effort} onChange=${(v) => set('effort', v)} options=${[['default', 'Default'], ...efforts.map((l) => [l, l[0].toUpperCase() + l.slice(1)])]} /></div>
      <div class="field"><span class="lbl">Reasoning block</span><${Seg} label="Reasoning block" value=${s.showReasoning} onChange=${(v) => set('showReasoning', v)} options=${[['stream', 'Open while thinking'], ['always', 'Always open'], ['never', 'Folded']]} /></div>
      <label class="check"><input type="checkbox" checked=${s.sendReasoning} onChange=${(e) => set('sendReasoning', e.target.checked)} /><span>Send earlier reasoning back with the conversation (the chat template renders it; the engine's prefix cache then reuses the whole conversation)</span></label>` : null}
    ${tab === 'context' ? html`
      <p class="notice" style="margin:0">The ring around the send button shows the context the next turn would use: the conversation as it is sent (counted by the server), plus the message you are typing. Green up to 80 %, yellow up to 90 %, red above.</p>
      <label class="check"><input type="checkbox" id="s-autocompact" checked=${s.autoCompact} onChange=${(e) => set('autoCompact', e.target.checked)} /><span>Compact automatically: before a turn (or the next pass of a tool loop) that would pass the threshold, the model first summarises the conversation, and the turn is sent with the summary in place of the earlier messages. A request the server refuses as too long is compacted and sent again once.</span></label>
      <div class="grid2">
        <${Field} id="s-compactat" label="Threshold (% of the context)" help=${`The conversation + your message + room for the reply (Max tokens, else 4,096 tokens). 50 to 99; default 90.`}>
          <input id="s-compactat" type="number" min="50" max="99" step="1" value=${s.compactAt} onInput=${(e) => set('compactAt', e.target.value)} /></${Field}>
      </div>
      <p class="notice" style="margin:0">Compact by hand at any time: click the ring, type <code>/compact</code> (optionally followed by what to keep in particular), or press Ctrl+Shift+L. The earlier messages stay in the chat, folded under a divider; undo a compaction from its summary.</p>` : null}
    ${tab === 'tools' ? html`<${McpPanel} s=${s} set=${set} mcp=${mcp} act=${mcpAct} proxyEnabled=${proxyEnabled} />` : null}
    ${tab === 'advanced' ? html`
      <${Field} id="s-custom" label="Extra request JSON" help=${customOk ? 'Merged into every /v1/chat/completions body last, e.g. {"stop": ["###"]}.' : html`<span style="color:var(--bad)">Not valid JSON: it is not sent.</span>`}>
        <textarea id="s-custom" rows="5" value=${s.custom} placeholder='{"stop": ["###"]}' onInput=${(e) => set('custom', e.target.value)}></textarea></${Field}>
      <div class="field"><span class="lbl">Danger zone</span><div><button class="btn danger" onClick=${mcpAct.deleteAll}><${Icon} name="trash" />Delete all chats in this browser</button></div></div>` : null}
  </${Dialog}>`;
}

function McpPanel({ s, set, mcp, act, proxyEnabled }) {
  const [f, setF] = useState({ name: '', url: '', headers: '', proxy: false });
  const add = () => {
    if (!/^https?:\/\//.test(f.url.trim())) return;
    const srv = { id: uid('s'), name: f.name.trim() || new URL(f.url.trim()).host, url: f.url.trim(), headers: f.headers, proxy: f.proxy && proxyEnabled, enabled: true, disabledTools: [] };
    set('mcpServers', [...s.mcpServers, srv]); setF({ name: '', url: '', headers: '', proxy: false }); act.connect(srv);
  };
  const upd = (id, patch) => set('mcpServers', s.mcpServers.map((x) => (x.id === id ? { ...x, ...patch } : x)));
  return html`
    <p class="notice" style="margin:0">MCP servers run in your browser: the page connects to each server (Streamable HTTP), lists its tools and offers them to the model; when the model calls one, the page runs it and sends the result back.${proxyEnabled ? ' This server offers its guarded proxy (/cors-proxy) for servers your browser cannot reach directly.' : ' The server proxy is off (start the server with --mcp-proxy to offer it).'}</p>
    <div class="grid2">
      <label class="check"><input type="checkbox" checked=${s.builtinTools} onChange=${(e) => set('builtinTools', e.target.checked)} /><span>Built-in tool get_datetime</span></label>
      <label class="check"><input type="checkbox" checked=${s.askTools} onChange=${(e) => set('askTools', e.target.checked)} /><span>Ask before running a tool</span></label>
    </div>
    ${s.mcpServers.map((x) => { const st = mcp[x.id] || {}; return html`<div class="srv">
      <div class="srv-head">
        <label class="check" style="flex:none"><input type="checkbox" checked=${x.enabled} onChange=${(e) => { upd(x.id, { enabled: e.target.checked }); if (e.target.checked) act.connect({ ...x, enabled: true }); }} aria-label=${`Use ${x.name}`} /></label>
        <span class="nm">${x.name}</span><span class="url">${x.url}${x.proxy ? ' · via server proxy' : ''}</span>
        <span class=${cx('badge', st.status === 'ok' && 'ok', st.status === 'error' && 'bad', st.status === 'connecting' && 'warn')}>${st.status === 'ok' ? `${st.tools.length} tools` : st.status || 'not connected'}</span>
        <button class="btn" onClick=${() => act.connect(x)}>Reconnect</button>
        <button class="icon-btn" title=${`Remove ${x.name}`} onClick=${() => set('mcpServers', s.mcpServers.filter((y) => y.id !== x.id))}><${Icon} name="trash" /></button>
      </div>
      ${st.error ? html`<div class="srv-err" role="alert">${st.error}</div>` : null}
      ${st.tools && st.tools.length ? html`<div class="srv-tools">${st.tools.map((t) => { const off = (x.disabledTools || []).includes(t.name); return html`<button class=${cx('chip', !off && 'on')} title=${t.description || t.name} aria-pressed=${!off}
          onClick=${() => upd(x.id, { disabledTools: off ? x.disabledTools.filter((n) => n !== t.name) : [...(x.disabledTools || []), t.name] })}>${t.name}</button>`; })}</div>` : null}
    </div>`; })}
    <div class="srv">
      <div class="lbl" style="font-weight:600">Add an MCP server</div>
      <div class="grid2">
        <${Field} id="m-url" label="URL"><input id="m-url" type="url" placeholder="http://127.0.0.1:8931/mcp" value=${f.url} onInput=${(e) => setF({ ...f, url: e.target.value })} /></${Field}>
        <${Field} id="m-name" label="Name"><input id="m-name" type="text" placeholder="optional" value=${f.name} onInput=${(e) => setF({ ...f, name: e.target.value })} /></${Field}>
      </div>
      <${Field} id="m-h" label="Headers" help="One per line (Authorization: Bearer …), or a JSON object."><textarea id="m-h" rows="2" value=${f.headers} onInput=${(e) => setF({ ...f, headers: e.target.value })}></textarea></${Field}>
      <div class="composer-row">
        <label class="check" title=${proxyEnabled ? '' : 'The server was started without --mcp-proxy'}><input type="checkbox" disabled=${!proxyEnabled} checked=${f.proxy && proxyEnabled} onChange=${(e) => setF({ ...f, proxy: e.target.checked })} /><span>Use the server proxy</span></label>
        <span class="grow"></span><button class="btn primary" disabled=${!/^https?:\/\//.test(f.url.trim())} onClick=${add}><${Icon} name="plus" />Add and connect</button>
      </div>
    </div>`;
}

export function CardDialog({ engine, hello, live, onClose, onAction, now, engineAt, desc }) {
  const pl = desc ? placementOf(desc) : null;
  const e = engine || {};
  const info = e.info || (hello && hello.engine_info) || {};
  const prof = live && live.record && live.record.profile;
  const st = e.state || 'unknown';
  const elapsed = (e.state_for_s || 0) + Math.max(0, (now - (engineAt || now)) / 1000);
  const label = { ready: 'Ready', parked: 'Parked: the card is free for other jobs', queued: 'Waiting for the card', loading: 'Warming up the card', stopping: 'Stopping', failed: 'Failed' }[st] || st;
  const cls = st === 'ready' ? 'ok' : st === 'failed' ? 'bad' : 'warn';
  return html`<${Dialog} title="Engine and card" onClose=${onClose} narrow foot=${e.available ? html`
      <button class="btn" onClick=${() => onAction(e.hold ? 'unhold' : 'hold')} title="Hold: keep the card while idle, even when other jobs wait">${e.hold ? 'Stop holding' : 'Hold the card'}</button>
      ${st === 'ready' ? html`<button class="btn" onClick=${() => onAction('release')} title="Free the card now; the next message takes it again">Release the card</button>` : null}
      ${st === 'parked' || st === 'failed' ? html`<button class="btn primary" onClick=${() => onAction('start')}>Start the engine</button>` : null}` : null}>
    <div class="card-state"><span class=${cx('dot', cls, (st === 'queued' || st === 'loading') && 'pulse')}></span>${label}</div>
    <p class="notice" style="margin:0">${e.mock ? 'Mock backend: recorded card runs replayed with their step timing; no card is used.' : `The engine gives the card to other jobs after ${Math.round(e.idle_yield_s || 60)} s idle while one waits${(e.expected_ready_s || e.last_start_to_ready_s) ? `, and takes about ${Math.round(e.expected_ready_s || e.last_start_to_ready_s)} s to come back (this model, measured at its last start)` : ''}.`}</p>
    <div class="card-sec"><h3>Card</h3><dl class="kv">
      <dt>state</dt><dd>${st} for ${fmtInt(elapsed)} s${e.detail ? ` · ${e.detail}` : ''}</dd>
      <dt>holder</dt><dd>${e.holder || 'nobody'}${e.owner && e.holder === e.owner ? ' (this server)' : ''}</dd>
      <dt>queue</dt><dd>${e.waiting_jobs != null ? `${e.waiting_jobs} other job(s) waiting` : '–'}</dd>
      <dt>warm-up</dt><dd>${e.last_warmup_s != null ? `${fmt1(e.last_warmup_s)} s, card taken to ready (last start)` : `about ${Math.round(e.expected_ready_s || 36)} s`}</dd>
      <dt>last start</dt><dd>${e.last_start_to_ready_s != null ? `${fmt1(e.last_start_to_ready_s)} s from the start command to ready (incl. the wait in the queue)` : '–'}</dd>
      <dt>policy</dt><dd>${e.hold ? 'holding' : `yield after ${e.idle_yield_s ?? 60} s idle when a job waits`}${e.admin_parked ? ' · parked by the admin' : ''}</dd>
      ${e.last_error ? html`<dt>last error</dt><dd style="color:var(--bad)">${e.last_error}</dd>` : null}
    </dl></div>
    <div class="card-sec"><h3>Model</h3><dl class="kv">
      ${Array.isArray(e.models) && e.models.length > 1 ? html`<dt>switch</dt><dd style="display:flex;gap:6px;flex-wrap:wrap">${e.models.map((m) => html`<button class=${cx('chip', (m.id === e.resident || m.resident) && 'on')} onClick=${() => onAction('load', m.id)} title=${`Switch to ${m.id} (${m.state})`}>${m.id} · ${m.state}</button>`)}</dd>` : null}
      <dt>model</dt><dd>${(desc && desc.known && desc.name) || (hello && hello.model) || info.model || '–'}${desc && desc.arch ? html` <span class="notice">(${desc.arch}${desc.layers ? `, ${desc.layers} layers` : ''})</span>` : null}</dd>
      ${desc && desc.known ? html`<dt>kind</dt><dd>${desc.moe ? `MoE: ${desc.moe.experts} experts, top-${desc.moe.topK}, expert width ${desc.moe.width}${desc.moe.form ? ` (${desc.moe.form} form)` : ''}` : 'dense'}</dd>` : null}
      ${pl ? html`<dt>experts</dt><dd>${pl.headline} · ${pl.text}${pl.layersText ? html`<br /><span class="notice">layer${pl.mode === 'host' ? 's' : ':slots'} ${pl.layersText}${pl.promoteEvery ? ` · hot-expert promotion every ${pl.promoteEvery} steps` : ''}</span>` : null}</dd>` : null}
      ${desc && desc.known && desc.drafters.length ? html`<dt>drafters</dt><dd>${drafterNames(desc).map((n) => (n === 'MTP head' && desc.precision.mtp ? `MTP head (${desc.precision.mtp})` : n)).join(', ')}</dd>` : null}
      ${desc && desc.known && (desc.features.qsa || desc.features.ple || desc.features.hc) ? html`<dt>features</dt><dd>${[desc.features.qsa ? `QSA sparse attention (dense below ${fmtInt(desc.features.qsa.denseBelow)} visible tokens)` : null, desc.features.ple ? `n-gram embedding before layer ${desc.features.ple.layer}, gathered on the host` : null, desc.features.hc ? `${desc.features.hc.streams} hyper-connection streams` : null].filter(Boolean).join(' · ')}</dd>` : null}
      <dt>precision</dt><dd>${(desc && desc.precision.weights) || info.precision || (hello && hello.precision) || '–'}${desc && desc.precision.weights && info.precision ? ` · engine ${info.precision}` : ''}${desc && desc.precision.kv ? ` · KV ${desc.precision.kv}` : ''}</dd>
      <dt>context</dt><dd>${fmtInt((hello && hello.n_ctx) || info.n_ctx)} tokens</dd>
      <dt>greedy</dt><dd>${methodInfo(info.greedy || (hello && hello.methods && hello.methods.greedy)).label} (${info.greedy || (hello && hello.methods && hello.methods.greedy) || '–'})</dd>
      <dt>sampled</dt><dd>${methodInfo(info.sampled || (hello && hello.methods && hello.methods.sampled)).label} (${info.sampled || (hello && hello.methods && hello.methods.sampled) || '–'})</dd>
      <dt>loaded</dt><dd>${info.methods || (hello && hello.methods && hello.methods.loaded) || '–'}</dd>
      <dt>vision</dt><dd>${hello && hello.vision ? `on${info.vision_max_patches ? ` (up to ${fmtInt(info.vision_max_patches)} patches)` : ''}` : 'off'}</dd>
      <dt>build</dt><dd>${info.build || (hello && hello.build) || '–'}</dd>
      ${info.startup_s ? html`<dt>engine start</dt><dd>${fmt1(info.startup_s)} s</dd>` : null}
    </dl></div>
    ${prof ? html`<div class="card-sec"><h3>Last profile (level 0)</h3><dl class="kv">
      <dt>HBM flow</dt><dd>${fmtGBs(prof.hbm && prof.hbm.gbps)} (${prof.hbm ? Math.round((100 * prof.hbm.gbps) / 2450) : '–'}% of spec)</dd>
      <dt>step</dt><dd>${fmt1(prof.step_ms)} ms, device busy ${fmt1(prof.busy_pct)}%</dd>
      <dt>host</dt><dd>${prof.host ? `enqueue ${fmtInt(prof.host.enqueue_us)} µs, synLaunch ${fmtInt(prof.host.launch_us)} µs` : '–'}</dd>
      ${live.source === 'mock' ? html`<dt>source</dt><dd>mock: modelled, not measured</dd>` : null}
    </dl></div>` : null}
  </${Dialog}>`;
}

export function ShortcutsDialog({ onClose }) {
  const K = (...k) => html`<span>${k.map((x) => html`<kbd>${x}</kbd>`)}</span>`;
  const rows = [['New chat', K('Ctrl', 'Shift', 'O')], ['Search chats', K('Ctrl', 'K')], ['Settings', K('Ctrl', ',')], ['Rename this chat', K('Ctrl', 'Shift', 'E')],
    ['Export this chat', K('Ctrl', 'Shift', 'S')], ['Delete this chat', K('Ctrl', 'Shift', 'D')], ['Toggle the sidebar', K('Ctrl', 'B')], ['Focus the message box', K('/')],
    ['Send', K('Enter')], ['New line', K('Shift', 'Enter')], ['Stop the reply', K('Esc')], ['Engine and card', K('Ctrl', 'Shift', 'G')], ['Compact the conversation', K('Ctrl', 'Shift', 'L')], ['These shortcuts', K('Ctrl', '/')]];
  return html`<${Dialog} title="Keyboard shortcuts" onClose=${onClose} narrow>
    <div class="shortcuts">${rows.map(([a, b]) => html`<span>${a}</span>${b}`)}</div>
    <p class="notice" style="margin:0">On a Mac, Cmd works in place of Ctrl.</p>
  </${Dialog}>`;
}

export function ConfirmDialog({ title, body, confirm, onConfirm, onClose, danger }) {
  return html`<${Dialog} title=${title} onClose=${onClose} narrow foot=${html`<button class="btn" onClick=${onClose}>Cancel</button><button class=${cx('btn', danger ? 'danger' : 'primary')} onClick=${() => { onConfirm(); onClose(); }}>${confirm}</button>`}>
    <p style="margin:0">${body}</p>
  </${Dialog}>`;
}

export function KeyDialog({ onSave, error }) {
  const [k, setK] = useState('');
  return html`<${Dialog} title="This server needs an API key" onClose=${() => {}} narrow foot=${html`<button class="btn primary" onClick=${() => onSave(k)}>Save and connect</button>`}>
    <p style="margin:0">gaudium-server was started with --api-key. Enter the key; it is kept in this browser.</p>
    ${error ? html`<p style="margin:0;color:var(--bad)">${error}</p>` : null}
    <div class="field"><label for="k-in">API key</label><input id="k-in" type="password" value=${k} onInput=${(e) => setK(e.target.value)} onKeyDown=${(e) => { if (e.key === 'Enter') onSave(k); }} /></div>
  </${Dialog}>`;
}
