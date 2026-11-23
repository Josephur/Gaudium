// One message: the user's bubble, or the assistant's turn (reasoning, answer, tool calls, the readout, actions, branches).
import { html, cx, memo } from './h.js';
import { useState, useRef, useEffect } from 'preact/hooks';
import { Icon } from './icons.js';
import { renderMarkdown } from '../lib/markdown.js';
import { Readout, WarmUp } from './readout.js';
import { useFollow } from './follow.js';
import { methodInfo, familyAt, liveStats, finalStats, stepsFromSamples, fmtInt, fmt1, fmt2, fmtMs, fmtPct, fmtBytes } from '../lib/stats.js';
import { requestSections } from '../lib/model.js';
import { parseArgs } from '../lib/stream.js';
import { splitContent, hasA2ui, readUserTurn, actionLabel } from '../lib/a2ui.js';
import { A2uiBlock, A2Icon } from './a2ui.js';

function Markdown({ text, cls = 'answer' }) {
  return html`<div class=${cls} dangerouslySetInnerHTML=${{ __html: renderMarkdown(text) }}></div>`;
}

// A reply's text: Markdown, with the model's interface blocks (A2UI) drawn where they stand. segKey: this segment's key
// among the conversation's blocks (lib/a2ui.js blocksOfPath).
function Content({ text, live, segKey, rt, act, busy }) {
  if (!live && !hasA2ui(text)) return html`<${Markdown} text=${text} />`;
  let n = 0;
  return splitContent(text, { live }).map((p, i) => (p.t === 'text' ? html`<${Markdown} key=${`t${i}`} text=${p.text} />`
    : html`<${A2uiBlock} key=${`b${i}`} rt=${rt} blockKey=${`${segKey}:${n++}`} raw=${p.raw} closed=${p.closed} live=${live} onAction=${act.a2uiSend} onReport=${act.a2uiReport} busy=${busy} />`));
}

// A user turn made by pressing a button of an interface: what was pressed, with the message it sent one click away.
function ActionTurn({ turn }) {
  const [open, setOpen] = useState(false);
  return html`<div class="bubble a2-action"><${A2Icon} name=${turn[0].error ? 'warning' : 'send'} /><span>${turn.map(actionLabel).join(' · ')}</span>
      <button type="button" class="a2-link" aria-expanded=${open} onClick=${() => setOpen(!open)}>${open ? 'hide' : 'details'}</button></div>
    ${open ? html`<pre class="a2-json a2-action-json">${JSON.stringify(turn.length === 1 ? turn[0] : turn, null, 2)}</pre>` : null}`;
}

function Branch({ sib, onNav }) {
  if (!sib || sib.count < 2) return null;
  return html`<span class="branch" aria-label="Branches">
    <button class="icon-btn" title="Previous branch" disabled=${sib.index === 0} onClick=${() => onNav(-1)}><${Icon} name="left" /></button>
    <span>${sib.index + 1}/${sib.count}</span>
    <button class="icon-btn" title="Next branch" disabled=${sib.index >= sib.count - 1} onClick=${() => onNav(1)}><${Icon} name="right" /></button>
  </span>`;
}

export function Attachment({ a, big, onRemove }) {
  const ext = (a.name.split('.').pop() || a.kind).slice(0, 4);
  if (a.kind === 'audio') return html`<div class=${cx('att audio', a.warn && 'warn')}>
    <div class="meta"><div title=${a.name}>${a.name}</div><audio controls preload="metadata" src=${a.dataUrl} aria-label=${`Play ${a.name}`}></audio><small>${a.note || fmtBytes(a.size)}</small>
      ${a.warn ? html`<small class="att-warn" role="note">${a.warn.text}</small>` : null}</div>
    ${onRemove ? html`<button class="icon-btn x" title=${`Remove ${a.name}`} onClick=${onRemove}><${Icon} name="x" /></button>` : null}
  </div>`;
  if (a.kind === 'video') return html`<${VideoAttachment} a=${a} big=${big} onRemove=${onRemove} />`;
  return html`<div class=${cx('att', big && a.kind === 'image' && 'big')}>
    ${a.kind === 'image' ? html`<img src=${a.dataUrl} alt=${a.name} />` : html`<span class="ficon">${a.kind === 'pdf' ? 'pdf' : ext}</span>`}
    ${!(big && a.kind === 'image') ? html`<div class="meta"><div>${a.name}</div><small>${a.note || fmtBytes(a.size)}</small></div>` : null}
    ${onRemove ? html`<button class="icon-btn x" title=${`Remove ${a.name}`} onClick=${onRemove}><${Icon} name="x" /></button>` : null}
  </div>`;
}

// A video: an animation plays as the image it is; a video file shows the frames that were sent, stepping through them
// at their own rate while the pointer is over it (or it has the focus), so what the model is shown can be checked.
function VideoAttachment({ a, big, onRemove }) {
  const frames = Array.isArray(a.frames) ? a.frames : null;
  const [i, setI] = useState(0);
  const [play, setPlay] = useState(false);
  useEffect(() => {
    if (!play || !frames || frames.length < 2) return undefined;
    const h = setInterval(() => setI((k) => (k + 1) % frames.length), Math.max(120, Math.min(600, 1000 / (a.fps || 2))));
    return () => clearInterval(h);
  }, [play, frames && frames.length]);
  const n = frames ? frames.length : a.frames || 0;
  const src = a.animation ? a.dataUrl : frames && frames.length ? frames[Math.min(i, frames.length - 1)] : null;
  const label = a.animation ? `Animation ${a.name}, ${n} frames` : `Video ${a.name}, ${n} frames sent`;
  return html`<div class=${cx('att video', big && 'big')} tabindex=${frames && frames.length > 1 ? 0 : null} aria-label=${label}
      onMouseEnter=${() => setPlay(true)} onMouseLeave=${() => { setPlay(false); setI(0); }} onFocus=${() => setPlay(true)} onBlur=${() => { setPlay(false); setI(0); }}>
    <span class="vthumb">${src ? html`<img src=${src} alt=${a.name} />` : html`<span class="ficon">vid</span>`}
      <span class="vbadge">${a.animation ? 'GIF' : frames && play ? `${i + 1}/${n}` : '▶'}</span></span>
    <div class="meta"><div title=${a.name}>${a.name}</div><small>${a.note || fmtBytes(a.size)}</small></div>
    ${onRemove ? html`<button class="icon-btn x" title=${`Remove ${a.name}`} onClick=${onRemove}><${Icon} name="x" /></button>` : null}
  </div>`;
}

function Edit({ initial, onSave, onCancel, label }) {
  const [t, setT] = useState(initial);
  const ref = useRef(null);
  useEffect(() => { const e = ref.current; if (e) { e.focus(); e.style.height = 'auto'; e.style.height = e.scrollHeight + 'px'; } }, []);
  return html`<div class="composer" style="width:100%">
    <textarea ref=${ref} value=${t} rows="2" aria-label="Edit message" onInput=${(e) => { setT(e.target.value); e.target.style.height = 'auto'; e.target.style.height = e.target.scrollHeight + 'px'; }}
      onKeyDown=${(e) => { if (e.key === 'Escape') onCancel(); if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); onSave(t); } }}></textarea>
    <div class="composer-row"><span class="grow notice">Saving makes a new branch; the old one stays in the navigator.</span>
      <button class="btn" onClick=${onCancel}>Cancel</button><button class="btn primary" onClick=${() => onSave(t)}>${label}</button></div>
  </div>`;
}

export const UserMessage = memo(function UserMessage({ m, sib, act, busy }) {
  const [editing, setEditing] = useState(false);
  const turn = readUserTurn(m.content);   // a pressed button of an interface, not typed text
  return html`<div class="msg user" data-id=${m.id}>
    ${m.attachments && m.attachments.length ? html`<div class="atts">${m.attachments.map((a) => html`<${Attachment} a=${a} big />`)}</div>` : null}
    ${editing ? html`<${Edit} initial=${m.content || ''} label="Send" onCancel=${() => setEditing(false)} onSave=${(t) => { setEditing(false); act.editUser(m.id, t); }} />`
      : turn.length ? html`<${ActionTurn} turn=${turn} />` : (m.content ? html`<div class="bubble">${m.content}</div>` : null)}
    <div class="msg-meta">
      <${Branch} sib=${sib} onNav=${(d) => act.nav(m.id, d)} />
      <button class="icon-btn" title="Copy" onClick=${() => act.copy(m.content || '')}><${Icon} name="copy" /></button>
      ${turn.length ? null : html`<button class="icon-btn" title="Edit (makes a branch)" disabled=${busy} onClick=${() => setEditing(true)}><${Icon} name="edit" /></button>`}
      <button class="icon-btn" title="Delete this message and what follows" disabled=${busy} onClick=${() => act.remove(m.id)}><${Icon} name="trash" /></button>
    </div>
  </div>`;
});

function ToolCard({ c, act, live }) {
  const [open, setOpen] = useState(false);
  const args = parseArgs(c.arguments);
  const st = c.status || 'done';
  const label = { streaming: 'writing the call…', ask: 'waiting for you', running: 'running…', done: c.ms != null ? `done · ${fmtMs(c.ms)}` : 'done', error: 'failed', denied: 'declined', skipped: 'not run' }[st] || st;
  const showBody = open || st === 'ask' || st === 'error';
  return html`<div class=${cx('tool', st === 'error' && 'error')}>
    <div class="tool-head">
      <span class="icon-btn" style="width:20px;height:20px"><${Icon} name="tool" /></span>
      <span class="name">${c.name || '…'}</span><span class="where">${c.server ? `on ${c.server}` : ''}</span>
      <span class="st">${st === 'running' || st === 'streaming' ? html`<span class="dot warn pulse"></span>` : st === 'done' ? html`<span class="dot ok"></span>` : st === 'error' ? html`<span class="dot bad"></span>` : null}${label}</span>
      <button class="icon-btn" title=${showBody ? 'Hide details' : 'Show arguments and result'} aria-expanded=${showBody} onClick=${() => setOpen(!open)}><${Icon} name=${showBody ? 'x' : 'chev'} /></button>
    </div>
    ${showBody ? html`<div class="tool-body">
      <div class="tool-sec"><h4>arguments</h4><pre>${args.ok ? JSON.stringify(args.value, null, 2) : String(c.arguments || '')}</pre></div>
      ${c.result != null || c.error ? html`<div class="tool-sec"><h4>${c.error ? 'error' : 'result'}</h4><pre>${c.error || (typeof c.result === 'string' ? c.result : JSON.stringify(c.result, null, 2))}</pre></div>` : null}
    </div>` : null}
    ${st === 'ask' ? html`<div class="tool-ask"><span>Run <b>${c.name}</b>${c.server ? ` on ${c.server}` : ''} with these arguments?</span>
      <button class="btn" onClick=${() => act.toolAnswer(c.id, 'deny')}>Decline</button>
      <button class="btn" onClick=${() => act.toolAnswer(c.id, 'always')}>Always allow</button>
      <button class="btn primary" onClick=${() => act.toolAnswer(c.id, 'run')}>Run</button></div>` : null}
  </div>`;
}

function Thinking({ text, live, inReasoning, pref, onSkip, ms }) {
  const auto = pref === 'always' ? true : pref === 'never' ? false : live && inReasoning;
  const [user, setUser] = useState(null);
  const open = user == null ? auto : user;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  // the body is a box of its own (max-height): while the reasoning streams, it follows its newest line like the chat does
  const body = useRef(null);
  const onScroll = useFollow(body, { active: !!live, near: 24 });
  const label = live && inReasoning ? 'Thinking' : ms >= 100 ? `Thought for ${fmtMs(ms)}` : 'Reasoning';
  return html`<details class=${cx('think', live && inReasoning && 'live')} open=${open} onToggle=${(e) => { if (e.target.open !== open) setUser(e.target.open); }}>
    <summary><${Icon} name="chev" /><span class="label">${label}</span><span class="grow mono" style="color:var(--muted);font-size:11px">${fmtInt(words)} words</span>
      ${live && inReasoning && onSkip ? html`<button class="btn" style="height:26px" onClick=${(e) => { e.preventDefault(); onSkip(); }} title="End the reasoning now and answer"><${Icon} name="skip" />Skip reasoning</button>` : null}
    </summary>
    ${open ? html`<div class="think-body" ref=${body} onScroll=${onScroll}><${Markdown} text=${text} cls="answer" /></div>` : null}
  </details>`;
}

// A compaction summary (docs/ui.md 10): open while it streams, folded after (it can be long); marked as a summary, not a reply.
function Summary({ g, live, compacted, inTurn, auto }) {
  const [user, setUser] = useState(null);
  const open = user == null ? !!live : user;
  const body = useRef(null);
  const onScroll = useFollow(body, { active: !!live, near: 24 });
  const text = (live && g.prefill ? g.prefill : '') + (g.content || '');
  const n = (g.stats && g.stats.n) || (g.timings && g.timings.predicted_n) || 0;
  const label = live ? 'Compacting the conversation' : inTurn ? 'Compacted during the tool loop' : auto ? 'Compaction summary (automatic)' : 'Compaction summary';
  return html`<details class=${cx('summary', live && 'live')} open=${open} onToggle=${(e) => { if (e.target.open !== open) setUser(e.target.open); }}>
    <summary><${Icon} name="chev" /><span class="sum-mark"><${Icon} name="compact" /></span><span class="label">${label}</span>
      <span class="grow mono meta">${compacted ? `${fmtInt(compacted)} messages` : ''}${n ? ` · ${fmtInt(n)} tokens` : ''}</span></summary>
    ${open ? html`<div class="summary-body" ref=${body} onScroll=${onScroll}><${Markdown} text=${text} cls="answer" /></div>` : null}
  </details>`;
}

function StatsLine({ s, open, onToggle, info, nCtx }) {
  if (!s || (!s.n && !s.promptN)) return null;
  return html`<div class="stats-line">
    <span><span class=${`dot ${info.maxTok === 1 ? 'none' : info.family}`} style="display:inline-block;margin-right:5px"></span>${info.short || info.label}${s.switchStep != null ? ` → MTP tree @${s.switchStep + 1}` : ''}</span>
    <span><b>${fmt1(s.tps)}</b> tok/s</span>
    ${s.ttftMs != null ? html`<span>TTFT <b>${fmtInt(s.ttftMs)}</b> ms</span>` : null}
    <span><b>${fmtInt(s.n)}</b> tokens</span>
    ${s.perStep ? html`<span><b>${fmt1(s.perStep)}</b> tok/step</span>` : null}
    ${s.accept != null ? html`<span>accept <b>${fmtPct(s.accept)}</b></span>` : null}
    ${s.cacheN > 0 ? html`<span class="reuse" title="Prompt tokens the engine reused from its prefix cache instead of prefilling them">reused ${fmtInt(s.cacheN)}</span>` : null}
    ${nCtx ? html`<span title="Context used: prompt + reused + generated tokens, of the engine's context">context <b>${fmtInt(s.promptN + s.cacheN + s.n)}</b>/${fmtInt(nCtx)}</span>` : null}
    <button class="more" onClick=${onToggle} aria-expanded=${open}>${open ? 'hide readout' : 'readout'}</button>
  </div>`;
}

// the request's sampling as the readout shows it (from the server's details, else the settings it was sent with)
// why the method differs from the one picked (docs/ui.md 3.1): the server ran a sampled request on its sampled default
// (gaudium.method_fallback), or the UI ran a greedy-only method greedy
const methodNoteOf = (seg, snap) => {
  const f = seg && seg.gaudium && seg.gaudium.method_fallback;
  if (f) return { text: `${methodInfo(f.asked).label} is greedy-only: sampled with ${methodInfo(f.used).label}`, title: f.why };
  const p = snap && snap.methodPlan;
  if (p && p.kind === 'fallback') return { text: `${methodInfo(p.method).label} is greedy-only: sampled with ${methodInfo(p.used).label}`, title: 'the server runs a sampled request on its sampled method (the temperature is set)' };
  if (snap && snap.forcedGreedy) return { text: `greedy: ${methodInfo(snap.method).label} is greedy-only`, title: `${methodInfo(snap.method).label} does not sample, so this request ran at temperature 0. Pick MTP adaptive or the engine default to sample.` };
  return null;
};
const samplingOf = (seg, snap) => (seg && seg.gaudium && seg.gaudium.sampling) || (snap ? { temperature: snap.temperature, seed: snap.seed, greedy: snap.greedy } : null);

export const AssistantMessage = memo(function AssistantMessage({ m, sib, act, live, busy, view }) {
  const [editing, setEditing] = useState(false);
  const [openRo, setOpenRo] = useState(false);
  const segs = m.segments || [];
  const isLive = !!live;
  // the reply that just streamed keeps its readout open (the live readout turns into the past one in place, not a one-line summary)
  // (open in the very render that ends the stream, not one effect later: the docked panel and this readout swap in one frame)
  // (not a compaction summary: its readout stays one click away, the summary card is what matters there)
  const isNode = m.role === 'compaction';
  const wasLive = useRef(false);
  const roOpen = openRo || (wasLive.current && !isLive && !isNode);
  useEffect(() => { if (wasLive.current && !isLive && !isNode) setOpenRo(true); wasLive.current = isLive; }, [isLive]);
  const all = isLive ? [...segs, live.seg] : segs;
  const snap = m.request || {};
  const methodOf = (seg) => methodInfo((seg && seg.gaudium && seg.gaudium.method) || (isLive && live.method) || snap.method || view.defaultMethod(snap));
  const lastDone = [...segs].reverse().find((g) => g.stats && (g.stats.n || g.stats.promptN));
  const passes = segs.filter((g) => g.stats && g.stats.n);
  const passName = (g, i) => (g.kind === 'compaction' ? 'summary' : `pass ${i + 1}`);
  const [pass, setPass] = useState(null);
  const selPass = pass == null || pass >= passes.length ? passes.length - 1 : pass;
  const text = all.filter((g) => isNode || g.kind !== 'compaction').map((g) => g.content || '').join('\n\n');
  const waiting = isLive && !live.started;
  return html`<div class=${cx('msg assistant', isNode && 'compaction')} data-id=${m.id}>
    ${all.map((g, i) => {
      const liveSeg = isLive && i === all.length - 1;
      const sum = isNode || g.kind === 'compaction';
      const overflow = g.error && g.error.overflow && !liveSeg && i === all.length - 1;
      return html`
        ${sum && (g.content || liveSeg) ? html`<${Summary} g=${g} live=${liveSeg} compacted=${isNode ? m.compacted : g.compacted} inTurn=${!isNode} auto=${isNode && m.auto} />` : null}
        ${!sum && g.reasoning ? html`<${Thinking} text=${g.reasoning} live=${liveSeg} inReasoning=${liveSeg && !g.content && !(g.toolCalls || []).length && live.inReasoning} pref=${view.settings.showReasoning}
            onSkip=${liveSeg && view.canSkip ? () => act.skipReasoning() : null} ms=${g.reasoningMs} />` : null}
        ${!sum && g.content ? html`<${Content} text=${g.content} live=${liveSeg} segKey=${`${m.id}:${i}`} rt=${view.a2ui} act=${act} busy=${busy} />` : null}
        ${sum ? null : (g.toolCalls || []).map((c) => html`<${ToolCard} c=${c} act=${act} live=${liveSeg} />`)}
        ${g.error ? html`<div class="err"><b>${g.error.code ? `Error ${g.error.code}` : 'Error'}</b>: ${sum ? 'the summary failed: ' : ''}${g.error.message}${g.error.hint && !sum ? html`<div class="notice">${g.error.hint}</div>` : null}
          ${overflow && !isNode && g.error.canCompact !== false && act.compactRetry ? html`<div class="err-actions"><button class="btn primary" disabled=${busy} onClick=${() => act.compactRetry(m.id)}><${Icon} name="compact" />Compact and retry</button></div>` : null}</div>` : null}
        ${!liveSeg && g.finish === 'cancelled' ? html`<div class="notice">${sum ? 'Stopped: this summary is not used; the requests send the whole conversation.' : 'Stopped.'}</div>` : null}
        ${!liveSeg && g.finish === 'length' ? html`<div class="notice">${sum ? 'The summary reached its token budget and is used as it is.' : 'Stopped at the token limit.'}</div>` : null}`;
    })}
    ${waiting && !view.settings.telemetry ? (view.engineReady ? html`<div class="notice"><span class="dot warn pulse" style="display:inline-block;margin-right:8px"></span>Prefilling ${live.progressText || ''}</div>` : html`<${WarmUp} engine=${view.engine} receivedAt=${view.engineAt} now=${view.now} />`) : null}
    ${!isLive && lastDone ? html`<${StatsLine} s=${lastDone.stats} nCtx=${isNode ? 0 : view.nCtx} info=${methodOf(lastDone)} open=${roOpen} onToggle=${() => setOpenRo(!roOpen)} />` : null}
    ${!isLive && roOpen && passes.length ? (() => {
      // one readout per reply: a tool loop has one generation pass per model turn; show one at a time (the last by default)
      const g = passes[selPass];
      const chips = passes.length > 1 ? html`<span class="ro-passes" role="group" aria-label="Generation passes of this reply">${passes.map((_, i) =>
        html`<button class=${cx('ro-pass', i === selPass && 'on')} aria-pressed=${i === selPass} title=${passes[i].kind === 'compaction' ? 'The summary written in the middle of the tool loop' : i < passes.length - 1 ? 'This pass ended in a tool call' : 'The final pass'} onClick=${() => setPass(i)}>${passName(passes[i], i)}</button>`)}</span>` : null;
      return html`<${Readout} mode="past" st=${g.stats} info=${methodOf(g)} switchStep=${g.stats.switchStep} rec=${g.live}
        steps=${g.stats.steps || []} sampling=${samplingOf(g, snap)} note=${methodNoteOf(g, snap)} precision=${view.precision} themeKey=${view.themeKey} mock=${g.stats.mock} tool=${chips}
        desc=${view.desc} sec=${requestSections(g.gaudium)} />`;
    })() : null}
    ${editing ? html`<${Edit} initial=${text} label="Save as a branch" onCancel=${() => setEditing(false)} onSave=${(t) => { setEditing(false); act.editAssistant(m.id, t); }} />` : null}
    ${!isLive && isNode ? html`<div class="msg-meta">
      <${Branch} sib=${sib} onNav=${(d) => act.nav(m.id, d)} />
      <button class="icon-btn" title="Copy the summary" onClick=${() => act.copy(text)}><${Icon} name="copy" /></button>
      <button class="icon-btn" title="Undo this compaction (the summary goes; every message stays)" disabled=${busy} onClick=${() => act.undoCompact(m.id)}><${Icon} name="undo" /></button>
    </div>` : null}
    ${!isLive && !isNode ? html`<div class="msg-meta">
      <${Branch} sib=${sib} onNav=${(d) => act.nav(m.id, d)} />
      <button class="icon-btn" title="Copy" onClick=${() => act.copy(text)}><${Icon} name="copy" /></button>
      <button class="icon-btn" title="Regenerate (makes a branch)" disabled=${busy} onClick=${() => act.regenerate(m.id)}><${Icon} name="refresh" /></button>
      <button class="icon-btn" title="Edit (makes a branch)" disabled=${busy} onClick=${() => setEditing(true)}><${Icon} name="edit" /></button>
      <button class="icon-btn" title="Delete this reply and what follows" disabled=${busy} onClick=${() => act.remove(m.id)}><${Icon} name="trash" /></button>
    </div>` : null}
  </div>`;
});

// The prompt is being read (prefill): shown in the docked readout's box until the first token.
function PrefillWait({ seg, text }) {
  const p = seg && seg.progress;
  const pct = p && p.total > 0 ? Math.min(100, (100 * p.processed) / p.total) : null;
  return html`<section class="warm" role="status" aria-live="polite">
    <div class="warm-head"><span class="dot warn pulse"></span>Reading the prompt</div>
    <p>The engine prefills the whole prompt before it writes the first token; the numbers start with that token.</p>
    <div class=${cx('progress', pct == null && 'indet')}><div style=${pct == null ? '' : `width:${pct}%`}></div></div>
    <div class="mono">${text || 'sending'}</div>
  </section>`;
}

const WAIT_TEXT = { parked: 'taking the card', queued: 'waiting for the card', loading: 'warming up the card', stopping: 'the engine is stopping', failed: 'the engine failed' };

// The live readout while a request is in flight, docked between the chat and the message box (not in the scrolling flow, so it
// stays put while the reply grows). The same box from the send to the last token: the wait for the card or the prefill lie over
// it, then the numbers. Nothing in it changes its size (fixed slots, tabular digits). collapsed: a one-line summary.
export function LiveDock({ live, m, view, collapsed, onToggle }) {
  const seg = (live.phase === 'running tools' && live.prevSeg) || live.seg || {};
  const snap = (m && m.request) || {};
  const info = methodInfo((seg.gaudium && seg.gaudium.method) || live.method || snap.method || view.defaultMethod(snap));
  const st = liveStats(seg);
  const steps = stepsFromSamples(seg.samples || []);
  const pre = live.phase === 'sending' || live.phase === 'prefill';
  const warm = pre && !view.engineReady;
  const phase = warm ? 'waiting' : live.phase;
  const toggle = html`<button class="icon-btn ro-toggle" title=${collapsed ? 'Show the full readout' : 'Shrink the readout to one line'}
    aria-label=${collapsed ? 'Show the full readout' : 'Shrink the readout to one line'} aria-expanded=${!collapsed} onClick=${onToggle}><${Icon} name=${collapsed ? 'up' : 'down'} /></button>`;
  if (collapsed) {
    const e = view.engine || {};
    const elapsed = e.state ? (e.state_for_s || 0) + Math.max(0, (view.now - (view.engineAt || view.now)) / 1000) : 0;
    const fam = info.maxTok === 1 ? 'none' : familyAt(info, Math.max(0, steps.length - 1), live.switchStep);
    return html`<div class="dock collapsed"><div class="dock-line" role="status" aria-label="Engine readout">
      ${pre ? html`<span class="dot warn pulse"></span><span class="what">${warm ? `${WAIT_TEXT[e.state] || e.state || 'sending'} · ${fmtInt(elapsed)} s` : `reading the prompt${live.progressText ? ` · ${live.progressText}` : ''}`}</span>`
        : html`<span class="what nums"><span class="meth"><span class=${`dot ${fam}`}></span>${info.short || info.label}<span class="sw">${live.switchStep != null && live.switchStep >= 0 ? ' → MTP tree' : ''}</span></span>
          <span><b>${fmt1(st.tps)}</b> tok/s</span>${info.maxTok === 1 ? html`<span><b>${st.msPerStep ? fmt1(st.msPerStep) : '–'}</b> ms/step</span>` : html`<span><b>${fmt2(st.perStep)}</b> tok/step</span><span class="acc">accept <b>${fmtPct(st.accept)}</b></span>`}</span>`}
      <span class="grow"></span><span class="ro-phase">${phase}</span>${toggle}
    </div></div>`;
  }
  const cover = warm ? html`<${WarmUp} engine=${view.engine} receivedAt=${view.engineAt} now=${view.now} />` : pre ? html`<${PrefillWait} seg=${seg} text=${live.progressText} />` : null;
  return html`<div class="dock"><${Readout} mode="live" st=${st} info=${info} switchStep=${live.switchStep} rec=${live.rec} steps=${steps}
    sampling=${samplingOf(seg, snap)} note=${methodNoteOf(seg, snap)} precision=${view.precision} phase=${phase} themeKey=${view.themeKey} mock=${view.mock} tool=${toggle} cover=${cover}
    desc=${view.desc} sec=${requestSections(seg.gaudium, live.sections)} /></div>`;
}

export { finalStats };
