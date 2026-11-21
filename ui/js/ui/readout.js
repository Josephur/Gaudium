// The engine readout: what only Gaudium can show, live while a reply streams and kept with the message afterwards.
// Numbers come from the reply's own stream (timings per token, the engine's final details); HBM flow and the step split come
// from the engine's level-0 profile (docs/profiling.md) via /gaudium/events; the card state from the same stream.
import { html, cx } from './h.js';
import { useEffect, useRef, useState } from 'preact/hooks';
import { methodInfo, familyAt, familyName, fmt1, fmt2, fmtInt, fmtMs, fmtPct, fmtGBs } from '../lib/stats.js';
import { placementOf, isSpeculative, mtpHeadLabel } from '../lib/model.js';

const HBM_SPEC = 2450, HBM_BEST = 2240;   // GB/s: Gaudi 2 HBM2E spec (tier B) and the best measured on this card (docs/profiling.md)

function cssVar(el, name) { return getComputedStyle(el).getPropertyValue(name).trim() || '#888'; }

// A step's drafter by name: familyName (stats.js; the df family is DFlash or DFlash2, from the engine's model description). `auto`
// exists only with DFlash2 (the Qwen family), so its steps say DFlash2.
const stepFamilyName = (info, fam) => (info.auto && fam === 'df' ? 'DFlash2' : familyName(fam));
// The chart's scale: the most tokens one step of the chosen method can commit (methodInfo().maxTok: DSpine 16, DFlash/DFlash2 8, MTP adaptive 9,
// a fixed K its K + 1), `auto` and unknown labels 8; never below what the reply actually reached. Four grid lines.
export function chartScale(info, steps) {
  const seen = Math.ceil(Math.max(0, ...(steps || [])));
  const maxY = Math.max((info && info.maxTok) || 8, seen, 2);
  const grid = Math.max(1, Math.ceil(maxY / 4));
  return { maxY, grid };
}
// Tokens committed per decode step, one bar per step, coloured by the drafter that proposed it.
export function StepChart({ steps, info, switchStep, live, themeKey }) {
  const ref = useRef(null);
  const [tip, setTip] = useState(null);
  const geo = useRef({ n: 0, bw: 1, start: 0, per: 1 });
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth || 600, H = c.clientHeight || 52;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const col = { df: cssVar(c, '--df'), ds: cssVar(c, '--ds'), mtp: cssVar(c, '--mtp'), grid: cssVar(c, '--line'), base: cssVar(c, '--line2'), muted: cssVar(c, '--muted'), fg: cssVar(c, '--fg') };
    const { maxY, grid } = chartScale(info, steps);
    const top = 4, bot = H - 2, ph = bot - top;
    // faint grid: 4 lines up to the method's widest step
    g.strokeStyle = col.grid; g.lineWidth = 1;
    for (let y = grid; y <= maxY; y += grid) { const yy = Math.round(bot - (y / maxY) * ph) + 0.5; g.beginPath(); g.moveTo(0, yy); g.lineTo(W, yy); g.stroke(); }
    // live: the newest steps that fit at 3 px + 1 px gap; finished: the whole reply, bucketed when it is longer than the width
    const n = steps.length;
    let start = 0, per = 1, bw = 3, gap = 1;
    const cap = Math.floor(W / (bw + gap));
    if (live) { start = Math.max(0, n - cap); }
    else if (n > cap) { per = Math.ceil(n / cap); bw = Math.max(1, Math.floor(W / Math.ceil(n / per)) - 1); }
    else if (n > 0) { bw = Math.max(2, Math.min(10, Math.floor(W / n) - 1)); }
    geo.current = { n, bw, gap, start, per };
    let x = 0;
    for (let i = start; i < n; i += per) {
      let v = 0, k = 0; for (let j = i; j < Math.min(n, i + per); j++) { v += steps[j]; k++; }
      v /= k;
      const fam = familyAt(info, i, switchStep);
      const h = Math.max(1.5, (v / maxY) * ph);
      g.fillStyle = col[fam];
      g.beginPath();
      if (g.roundRect) g.roundRect(x, bot - h, bw, h, [Math.min(2, bw / 2), Math.min(2, bw / 2), 0, 0]); else g.rect(x, bot - h, bw, h);
      g.fill();
      x += bw + gap;
    }
    // the switch of `auto`: a dashed line where the MTP tree takes over
    if (switchStep != null && switchStep >= start && switchStep < n) {
      const sx = Math.round(((switchStep - start) / per) * (bw + gap)) - 1 + 0.5;
      g.strokeStyle = col.fg; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(sx, 0); g.lineTo(sx, H); g.stroke(); g.setLineDash([]);
    }
    g.strokeStyle = col.base; g.beginPath(); g.moveTo(0, bot + 0.5); g.lineTo(W, bot + 0.5); g.stroke();
  }, [steps.length, steps[steps.length - 1], switchStep, live, themeKey, info.family, info.auto, info.maxTok]);
  const onMove = (e) => {
    const c = ref.current; if (!c) return;
    const r = c.getBoundingClientRect(); const { n, bw, gap, start, per } = geo.current;
    const k = Math.floor((e.clientX - r.left) / (bw + gap)); const i = start + k * per;
    if (i < 0 || i >= n) { setTip(null); return; }
    let v = 0, m = 0; for (let j = i; j < Math.min(n, i + per); j++) { v += steps[j]; m++; } v /= m;
    const fam = stepFamilyName(info, familyAt(info, i, switchStep));
    setTip({ x: Math.min(Math.max(e.clientX - r.left, 60), r.width - 60), text: per > 1 ? `steps ${i + 1}–${Math.min(n, i + per)} · ${fmt2(v)} tok/step · ${fam}` : `step ${i + 1} · ${Math.round(v * 100) / 100} tokens · ${fam}` });
  };
  return html`<div class="ro-chart" onMouseMove=${onMove} onMouseLeave=${() => setTip(null)}>
    <canvas ref=${ref} role="img" aria-label=${`Tokens per decode step for ${steps.length} steps`}></canvas>
    ${tip ? html`<div class="tip" style=${`left:${tip.x}px`}>${tip.text}</div>` : null}
  </div>`;
}

function Num({ k, v, unit, hero }) {
  return html`<div class=${cx('ro-num', hero && 'hero')}><div class="k">${k}</div><div class="v">${v}${unit ? html`<small>${unit}</small>` : null}</div></div>`;
}

function MethodHead({ info, switchStep, desc, spec }) {
  const sw = switchStep != null && switchStep >= 0;
  const head = spec && info.family === 'mtp' && desc ? mtpHeadLabel(desc) : '';
  if (info.auto) {
    return html`<span class="meth"><span class="dot df"></span>DFlash2</span>
      ${sw ? html`<span class="arrow">→</span><span class="sw"><span class="dot mtp"></span>MTP tree from step ${switchStep + 1}</span>` : html`<span class="notice">auto: switches to the MTP tree if DFlash2 drops below 3.5 tokens/step over 8 steps</span>`}`;
  }
  return html`<span class="meth"><span class=${`dot ${spec ? info.family : 'none'}`}></span>${info.label}${head ? html`<span class="notice"> · ${head}</span>` : null}</span>`;
}

// What the model adds to the readout (docs/server.md 7.8), from its description and this request's optional sections: where the
// experts live and the host work per step (MoE), QSA's regime, the n-gram rows. Nothing at all for a model without them.
export function ModelPanel({ desc, sec, mode }) {
  const pl = desc ? placementOf(desc) : null;
  const m = sec && sec.moe;
  const f = (desc && desc.features) || {};
  const chips = [];
  if (sec && sec.qsa) chips.push([sec.qsa.sparse ? 'QSA sparse' : 'QSA dense', `${fmtInt(sec.qsa.visible)} visible tokens; attention is dense below ${fmtInt(sec.qsa.denseBelow)}, then reads the selected blocks only`]);
  else if (f.qsa) chips.push([`QSA above ${fmtInt(f.qsa.denseBelow)}`, `sparse attention: dense while a query sees fewer than ${fmtInt(f.qsa.denseBelow)} tokens, then a ${fmtInt(f.qsa.budget)}-token budget`]);
  if (sec && sec.ple) chips.push([`n-gram rows: host${sec.ple.msPerStep != null ? ` ${fmt2(sec.ple.msPerStep)} ms/step` : ''}`, 'the n-gram embedding is gathered on the host for every step (the time includes the wait for the previous step)']);
  else if (f.ple) chips.push(['n-gram rows: host', `the n-gram embedding (before layer ${f.ple.layer}) is gathered on the host for every step`]);
  if (f.hc && f.hc.streams) chips.push([`${f.hc.streams} HC streams`, `hyper-connections: the residual is ${f.hc.streams} streams wide`]);
  if (!pl && !chips.length) return null;
  const hybrid = pl && pl.mode !== 'card';
  const cache = hybrid && pl.cache ? pl.cache : null;   // the expert cache: the right bar shows its hit rate, not the host twin's work
  const c = m && m.cache;
  const cacheTitle = c ? `Means per decode step over this reply. ${fmtPct(c.hitRate)} of the ${fmtInt(c.routed)} expert assignments in the ${pl.layers} cached layers found their expert on the card; per step ${fmt1(c.missedPerStep)} missed, ${fmt1(c.waitsPerStep)} layers waited at their miss gate for the copy, ${fmt1(c.fillsPerStep)} experts were copied from host RAM to the card (${fmtInt(c.evictions)} evictions in all)` : '';
  const hostTitle = m ? `${fmt1(m.hostRowsPerStep)} expert-rows on the host per step over ${fmt1(m.boundariesPerStep)} layer boundaries; ${fmt2(m.hostMsPerPart)} ms of host work each (routing down, the twin, the result up); the boundaries take ${fmt1(m.boundaryMsPerStep)} ms per step including the wait for each device segment${m.promotions ? `; ${m.promotions} hot-expert promotions, ${fmt2(m.promoMsEach)} ms each` : ''}` : '';
  return html`<div class="ro-model">
    ${pl ? html`<div class="ro-bars">
      <div class="ro-bar">
        <div class="k"><span>experts <b>${pl.mode === 'card' ? 'all on the card' : pl.share != null ? `${Math.round(100 * pl.share)}% card` : pl.mode}</b></span><span title=${pl.text}>${pl.mode === 'card' ? pl.text : pl.short}</span></div>
        <div class="meter" title=${`${pl.headline}${pl.layersText ? ` (layer:slots ${pl.layersText})` : ''}; ${pl.experts} experts per layer, top-${pl.topK}`}><div class="fill" style=${`width:${pl.share != null ? 100 * pl.share : 0}%;background:var(--seg2)`}></div></div>
        <div class="seg-legend"><span>${pl.experts} experts · top-${pl.topK}</span>${hybrid && pl.layersText ? html`<span title=${`layer:slots ${pl.layersText}`}>${pl.layers} ${pl.mode === 'host' ? 'host' : 'hybrid'} layer${pl.layers === 1 ? '' : 's'}</span>` : null}${pl.promoteEvery ? html`<span>promotion every ${pl.promoteEvery} steps</span>` : null}</div>
      </div>
      ${cache ? html`<div class="ro-bar">
        <div class="k"><span>expert cache <b>${c && c.hitRate != null ? `${fmtPct(c.hitRate)} hit` : '–'}</b></span><span title=${cacheTitle}>${c ? `${fmtInt(Math.round(c.waitsPerStep))}/${pl.layers} layers wait · ${fmtInt(Math.round(c.fillsPerStep))} copies` : (mode === 'live' ? 'waiting for totals' : 'not reported')}</span></div>
        <div class="meter" title=${cacheTitle}><div class="fill" style=${`width:${c && c.hitRate != null ? 100 * c.hitRate : 0}%;background:var(--seg3)`}></div></div>
        <div class="seg-legend"><span title=${cacheTitle}>per step (mean) · misses ${c ? fmtInt(Math.round(c.missedPerStep)) : '–'}</span><span>${cache.policy || 'lru'} slots · ${cache.reserve} kept free per layer</span></div>
      </div>` : html`<div class="ro-bar">
        <div class="k"><span>host per step <b>${!hybrid ? 'none' : m && m.hostMsPerStep != null ? `${fmt1(m.hostMsPerStep)} ms` : '–'}</b></span><span title=${hostTitle}>${!hybrid ? 'every expert on the card' : m ? `${fmt1(m.hostRowsPerStep)} rows · ${fmt1(m.boundariesPerStep)} cuts` : (mode === 'live' ? 'waiting for totals' : 'not reported')}</span></div>
        <div class="segbar" role="img" aria-label="Host work per step" title=${hostTitle}>${hybrid && m && m.boundaryMsPerStep ? html`<div style=${`flex-grow:${Math.max(0.0001, m.hostMsPerStep || 0)};background:var(--seg3)`}></div><div style=${`flex-grow:${Math.max(0.0001, m.boundaryMsPerStep - (m.hostMsPerStep || 0))};background:var(--line2)`}></div>` : html`<div style="flex-grow:1;background:var(--sunk)"></div>`}</div>
        ${hybrid ? html`<div class="seg-legend"><span><i style="background:var(--seg3)"></i>host work ${m ? fmt1(m.hostMsPerStep) : '–'}</span><span><i style="background:var(--line2)"></i>device wait ${m && m.boundaryMsPerStep != null ? fmt1(Math.max(0, m.boundaryMsPerStep - (m.hostMsPerStep || 0))) : '–'}</span><span>promotions ${m ? fmtInt(m.promotions) : '–'}</span></div>` : null}
      </div>`}
    </div>` : null}
    ${chips.length ? html`<div class="ro-chips">${chips.map(([t, title]) => html`<span class="ro-chip" title=${title}>${t}</span>`)}</div>` : null}
  </div>`;
}

// live: a streaming segment's stats; past: the stored summary. rec: the engine's live record (HBM, split), may be null.
// tool: an optional control at the end of the head (the dock's collapse toggle). cover: optional content laid over the body
// (the wait for the card or the prefill) while the box keeps its size: the numbers sit under it until the first token.
// desc: the model description (lib/model.js modelDesc), sec: this request's sections (requestSections); both optional.
export function Readout({ mode, st, info, switchStep, rec, steps, sampling, note, precision, phase, themeKey, mock, tool, cover, desc, sec }) {
  const spec = isSpeculative(info, st, sec);
  const hostRows = sec && sec.moe && sec.moe.hostWork ? sec.moe.hostRowsPerStep : null;
  const prof = rec && rec.profile;
  const hbm = prof && prof.hbm;
  const split = prof && prof.split_ms;
  const stepMs = (prof && prof.step_ms) || st.msPerStep;
  const idle = prof && prof.step_ms && prof.dev_ms ? Math.max(0, prof.step_ms - prof.dev_ms) : 0;
  const parts = split ? [['verify', split.verify, 'var(--seg1)'], ['drafts', split.drafts, 'var(--seg2)'], ['readout', split.readout, 'var(--seg3)'], ['other', (split.other || 0) + idle, 'var(--seg4)']] : null;
  // live: the legend's line is there from the start (dashes until the profile arrives), so the box does not grow when it lands
  const legend = parts || (mode === 'live' ? [['verify', null, 'var(--seg1)'], ['drafts', null, 'var(--seg2)'], ['readout', null, 'var(--seg3)'], ['other', null, 'var(--seg4)']] : null);
  const samp = sampling || {};
  const right = [precision, samp.greedy === false || Number(samp.temperature) > 0 ? `T ${samp.temperature === '' || samp.temperature == null ? '(server default)' : samp.temperature}` : 'greedy', samp.seed != null && samp.seed !== '' ? `seed ${samp.seed}` : (samp.greedy === false ? 'random seed' : null), mock ? 'mock replay' : null].filter(Boolean).join(' · ');
  return html`<section class=${cx('readout', mode === 'past' ? 'past' : 'live', cover && 'covered')} aria-label="Engine readout">
    <div class="ro-head">
      <${MethodHead} info=${info} switchStep=${switchStep} desc=${desc} spec=${spec} />
      <span class="right">${phase ? html`<span class="ro-phase">${phase}</span> · ` : null}${right}</span>${tool || null}
    </div>
    <div class="ro-nums">
      <${Num} hero k="decode" v=${fmt1(mode === 'live' ? st.tps : st.tps)} unit="tok/s" />
      <${Num} k="ttft" v=${st.ttftMs == null ? '–' : fmtInt(st.ttftMs)} unit="ms" />
      <${Num} k="prefill" v=${fmtInt(st.prefillTps)} unit="tok/s" />
      ${spec ? html`<${Num} k="per step" v=${fmt2(st.perStep)} unit="tok" />
      <${Num} k="accepted" v=${fmtPct(st.accept)} unit="" />` : html`<${Num} k="one step" v=${st.msPerStep ? fmt1(st.msPerStep) : '–'} unit="ms" />
      ${hostRows != null ? html`<${Num} k="host rows" v=${fmt1(hostRows)} unit="/step" />` : html`<${Num} k="steps" v=${fmtInt(st.launches)} unit="" />`}`}
    </div>
    ${spec ? html`<div>
      <${StepChart} steps=${steps} info=${info} switchStep=${switchStep} live=${mode === 'live'} themeKey=${themeKey} />
      <div class="ro-axis"><span class="ro-legend">${info.auto || info.family === 'df' ? html`<span><i style="background:var(--df)"></i>${stepFamilyName(info, 'df')}</span>` : null}${info.family === 'ds' ? html`<span><i style="background:var(--ds)"></i>DSpine</span>` : null}${info.auto || info.family === 'mtp' ? html`<span><i style="background:var(--mtp)"></i>MTP</span>` : null}<span>${(() => { const { maxY, grid } = chartScale(info, steps); const g = []; for (let y = grid; y <= maxY; y += grid) g.push(y); return `tokens per step (grid ${g.join('/')}, max ${maxY})`; })()}</span></span><span>${fmtInt(steps.length)} steps</span></div>
    </div>` : null}
    <div class="ro-bars">
      <div class="ro-bar">
        <div class="k"><span>HBM flow ${hbm ? html`<b>${fmtGBs(hbm.gbps)}</b>` : '–'}</span><span>${hbm ? `${Math.round((100 * hbm.gbps) / HBM_SPEC)}% of 2.45 TB/s · ${fmt1(hbm.gb_per_step)} GB/step` : (mode === 'live' ? 'waiting for the profile' : 'no profile recorded')}</span></div>
        <div class="meter" title="dashed line: 2.24 TB/s, the best measured on this card"><div class="fill" style=${`width:${hbm ? Math.min(100, (100 * hbm.gbps) / HBM_SPEC) : 0}%`}></div><div class="tick" style=${`left:${(100 * HBM_BEST) / HBM_SPEC}%`}></div></div>
      </div>
      <div class="ro-bar">
        <div class="k"><span>one step <b>${stepMs ? `${fmt2(stepMs)} ms` : '–'}</b></span><span>${prof && prof.busy_pct != null ? `device busy ${fmt1(prof.busy_pct)}%` : ''}</span></div>
        <div class="segbar" role="img" aria-label="Step split">${parts ? parts.map(([k, v, c]) => html`<div style=${`flex-grow:${Math.max(0.0001, v || 0)};background:${c}`} title=${`${k} ${fmt2(v)} ms`}></div>`) : html`<div style="flex-grow:1;background:var(--sunk)"></div>`}</div>
        ${legend ? html`<div class="seg-legend">${legend.map(([k, v, c]) => html`<span><i style=${`background:${c}`}></i>${k} ${fmt1(v)}</span>`)}</div>` : null}
      </div>
    </div>
    <${ModelPanel} desc=${desc} sec=${sec} mode=${mode} />
    <div class="ro-foot">
      ${note ? html`<span class="ro-note" title=${note.title || note.text}>${note.text}</span>` : null}
      ${st.cacheN > 0 ? html`<span class="reuse">reused <b>${fmtInt(st.cacheN)}</b> tokens · prefill ${fmtMs(st.promptMs)}</span>` : html`<span>prompt <b>${fmtInt(st.promptN)}</b> tokens in ${fmtMs(st.promptMs)}</span>`}
      <span><b>${fmtInt(st.n)}</b> tokens in ${fmtMs(st.ms)}</span>
      ${st.waitMs > 800 ? html`<span title="Time the request waited for the card before the engine took it">waited ${fmtMs(st.waitMs)} for the card</span>` : null}
      <span><b>${fmtInt(st.launches)}</b> launches${st.msPerStep ? ` · ${fmt1(st.msPerStep)} ms each` : ''}</span>
      ${st.vitMs ? html`<span>vision tower ${fmtMs(st.vitMs)}</span>` : null}
      ${prof && prof.split_basis && mode === 'past' ? html`<span title=${prof.split_basis}>split: modelled from HBM bytes</span>` : null}
    </div>
    ${cover ? html`<div class="ro-cover">${cover}</div>` : null}
  </section>`;
}

// The card is not ready: say what is happening and how long it takes, instead of a bare spinner.
export function WarmUp({ engine, receivedAt, now }) {
  const st = engine ? engine.state : 'unknown';
  const elapsed = engine ? (engine.state_for_s || 0) + Math.max(0, (now - (receivedAt || now)) / 1000) : 0;
  const expect = (engine && engine.expected_ready_s) || 36;
  let title, body, pct = null;
  if (st === 'parked') { title = 'Taking the card'; body = 'The engine is parked: it gave the card to other jobs while the chat was idle. It is being started for this message.'; }
  else if (st === 'queued') {
    const holder = engine.holder && engine.holder !== engine.owner ? engine.holder : null;
    title = holder ? `Waiting for the card: ${holder} has it` : 'Waiting for the card';
    body = holder ? `Another job holds the Gaudi 2. This message runs as soon as it releases the card${engine.waiting_jobs ? ` (${engine.waiting_jobs} job${engine.waiting_jobs > 1 ? 's' : ''} in the queue)` : ''}, then the engine takes about ${Math.round(expect)} s to start.` : `The card queue is handing the card over. Then the engine starts, about ${Math.round(expect)} s.`;
  } else if (st === 'loading') {
    title = 'Warming up the card';
    body = `Loading the weights and compiled recipes onto the Gaudi 2. A start takes about ${Math.round(expect)} s with the checkpoint in the page cache (about 3 min from disk).`;
    pct = Math.min(97, (100 * elapsed) / Math.max(1, expect));
  } else if (st === 'stopping') { title = 'The engine is stopping'; body = 'It is handing the card back; this message starts it again right after.'; }
  else if (st === 'failed') { title = 'The engine failed'; body = (engine && engine.last_error) || 'See gaudium-serve log.'; }
  else { title = 'Sending'; body = 'Waiting for the server.'; }
  return html`<section class="warm" role="status" aria-live="polite">
    <div class="warm-head"><span class=${cx('dot', st === 'failed' ? 'bad' : 'warn', st !== 'failed' && 'pulse')}></span>${title}</div>
    <p>${body}</p>
    <div class=${cx('progress', pct == null && 'indet')}><div style=${pct == null ? '' : `width:${pct}%`}></div></div>
    <div class="mono">${st} · ${fmtInt(elapsed)} s${st === 'loading' ? ` of ~${Math.round(expect)} s` : ''}${engine && engine.mock ? ' · mock card' : ''}</div>
  </section>`;
}

export { methodInfo };
