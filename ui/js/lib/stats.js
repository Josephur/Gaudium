// Live and per-message numbers from the stream's own timings (timings_per_token) and the engine's final details.
// Pure (no DOM), unit-tested. Server-clock quantities (predicted_ms, prompt_ms) are preferred over client timing.

// The name of the 'df' family: the block drafter the engine loaded, from its model description (model.js dfFamilyName: "DFlash" for
// --drafter dflash, "DFlash2" for dflash2 and for an engine that does not say). One model per page: the app sets it from the hello.
let DF_NAME = 'DFlash2';
export function setDfName(name) { DF_NAME = name || 'DFlash2'; }
export function familyName(fam) { return fam === 'df' ? DF_NAME : fam === 'ds' ? 'DSpine' : 'MTP'; }

// Speculative method labels (the engine's run labels, docs/server.md 7.4) -> what a person recognises, and the drafter family
// that colours a step: 'df' = DFlash or DFlash2 (familyName), 'ds' = DSpine, 'mtp' = MTP (chain, adaptive or tree). `auto` starts on
// DFlash2 and may switch once (it exists only with DFlash2 and the MTP tree, the Qwen family). A run that drafts nothing ("0", "df0p",
// "ds0p") is plain decoding, whatever drafter its process holds.
// maxTok: the most tokens one step of that method can commit (its widest verify chain + the bonus token); the step chart's scale.
// DSpine drafts a block of 15 (16 per step), DFlash/DFlash2 7 (8), the MTP adaptive set tops out at K = 8 (9); null = not known from the label.
export const PLAIN_DECODE = 'plain decode (no drafter)';
export function methodInfo(label) {
  const m = String(label || '').replace(/\s*\(mock.*\)$/, '');
  const base = m.split('@')[0];
  if (!base) return { label: 'engine default', family: 'mtp', base: '' };
  if (/^(?:df|ds)?0p?$/.test(base)) return { label: PLAIN_DECODE, short: 'plain decode', family: base.startsWith('df') ? 'df' : base.startsWith('ds') ? 'ds' : 'mtp', plain: true, maxTok: 1, base };
  if (base === 'auto') return { label: 'auto (DFlash2 → MTP tree)', short: 'auto', family: 'df', auto: true, maxTok: 8, base };
  if (/^ds-a/.test(base)) { const t = /\+L(\d+)/.exec(base); return { label: t ? `DSpine adaptive + ${t[1]} leaves` : 'DSpine adaptive', short: 'DSpine', family: 'ds', maxTok: 16, base }; }
  if (/^ds\d/.test(base)) { const k = parseInt(base.slice(2), 10), t = /\+L(\d+)/.exec(base); return { label: `DSpine K=${k}${t ? ` + ${t[1]} leaves` : ''}`, short: 'DSpine', family: 'ds', maxTok: k + 1, base }; }
  if (/^ds/.test(base)) return { label: `DSpine ${base.slice(2).replace(/^[-:]/, '')}`.trim(), short: 'DSpine', family: 'ds', maxTok: 16, base };
  if (/^df-a/.test(base)) { const t = /\+L(\d+)/.exec(base); return { label: `${DF_NAME} adaptive${t ? ` + ${t[1]} leaves` : ''}`, short: DF_NAME, family: 'df', maxTok: 8, base }; }
  if (/^df\d/.test(base)) { const k = parseInt(base.slice(2), 10), t = /\+L(\d+)/.exec(base); return { label: `${DF_NAME} K=${k}${t ? ` + ${t[1]} leaves` : ''}`, short: DF_NAME, family: 'df', maxTok: t ? 8 : k + 1, base }; }
  if (/^df/.test(base)) return { label: `${DF_NAME} ${base.slice(2).replace(/^[-:]/, '')}`.trim(), short: DF_NAME, family: 'df', maxTok: 8, base };
  if (base === 'adaptive' || /^a:/.test(base)) return { label: 'MTP adaptive', short: 'MTP', family: 'mtp', maxTok: 9, base };
  if (/^T/.test(base) || /tree/i.test(base)) return { label: `MTP tree ${base}`, short: 'MTP tree', family: 'mtp', base };
  if (/^c\d/.test(base)) return { label: `certified MTP K=${base.slice(1)}`, short: 'MTP', family: 'mtp', maxTok: (parseInt(base.slice(1), 10) || 0) + 1 || null, base };
  if (/^\d+$/.test(base)) return { label: `MTP K=${base}`, short: 'MTP', family: 'mtp', maxTok: parseInt(base, 10) + 1, base };
  return { label: base, short: base, family: 'mtp', base };
}

// Greedy-only runs. `list`: the server's hello.methods.greedy_only, which the engine computes from its own rule (gaudium_engine.cpp
// sampled_ok; the server's label rule only for an older engine or the mock): with a list, a run is greedy-only exactly when the list
// names it (a Gemma 4 engine's df0p / df7p sample). Without one (no hello yet), the old label rule: only the plain MTP chain ("<K>")
// and MTP adaptive sample.
export function greedyOnly(label, list) {
  const b = String(label || '').replace(/\s*\(mock.*\)$/, '').split('@')[0].trim();
  if (!b || b === 'default') return false;
  if (Array.isArray(list)) return list.includes(b);
  return !(b === 'adaptive' || /^\d+$/.test(b));
}

// The drafter of step i (0-based) for a method and an optional switch step (the engine's: the MTP tree runs from that step on)
export function familyAt(info, i, switchStep) {
  if (info.auto) return switchStep != null && switchStep >= 0 && i >= switchStep ? 'mtp' : 'df';
  return info.family;
}

// Per-step committed tokens from the engine's accept sequence ("0123..." accepted drafts per step, base 32)
export function stepsFromAseq(aseq) {
  const out = [];
  for (const c of String(aseq || '')) { const v = parseInt(c, 32); if (!Number.isNaN(v)) out.push(v + 1); }
  return out;
}

// Per-step tokens from timing samples while streaming: each sample is cumulative; a chunk may cover several steps
// (text held back by the UTF-8 decoder or a stop-string match), which is spread evenly over them.
export function stepsFromSamples(samples) {
  const out = [];
  let prev = null;
  for (const s of samples) {
    if (prev && s.launches > prev.launches) {
      const dl = s.launches - prev.launches, dn = s.n - prev.n;
      for (let k = 0; k < dl; k++) out.push(dn / dl);
    }
    if (!prev || s.launches >= prev.launches) prev = s;
  }
  return out;
}

// TTFT counts from the moment the engine took the request (its first chunk): a wait for the card is reported apart (waitMs)
function ttft(seg) {
  if (seg.firstTokenAt == null) return null;
  const t0 = seg.startedAt != null ? seg.startedAt : seg.sentAt;
  return t0 == null ? null : seg.firstTokenAt - t0;
}
function waitMs(seg) { return seg.startedAt != null && seg.sentAt != null ? Math.round(seg.startedAt - seg.sentAt) : null; }

// Live numbers for the strip. windowMs: the decode-rate window on the server clock.
export function liveStats(seg, windowMs = 1500) {
  const S = seg.samples || [];
  const t = seg.timings || {};
  const last = S[S.length - 1];
  const r = {
    ttftMs: ttft(seg), waitMs: waitMs(seg),
    promptN: t.prompt_n || 0, cacheN: t.cache_n || 0, promptMs: t.prompt_ms || 0,
    prefillTps: t.prompt_ms > 0 ? (1000 * (t.prompt_n || 0)) / t.prompt_ms : null,
    n: t.predicted_n || 0, ms: t.predicted_ms || 0,
    tps: null, tpsAvg: t.predicted_ms > 0 ? (1000 * t.predicted_n) / t.predicted_ms : null,
    perStep: null, perStepAvg: t.launches > 0 ? (t.predicted_n || 0) / t.launches : null,
    accept: null, acceptAvg: t.draft_n > 0 ? (t.draft_n_accepted || 0) / t.draft_n : null,
    launches: t.launches || 0, msPerStep: t.launches > 0 ? (t.predicted_ms || 0) / t.launches : null,
  };
  if (last) {
    let i = S.length - 1;
    while (i > 0 && last.ms - S[i - 1].ms <= windowMs) i--;
    const a = S[i];
    if (a && last.ms > a.ms) {
      r.tps = (1000 * (last.n - a.n)) / (last.ms - a.ms);
      if (last.launches > a.launches) r.perStep = (last.n - a.n) / (last.launches - a.launches);
      if (last.drafts > a.drafts) r.accept = (last.acc - a.acc) / (last.drafts - a.drafts);
    }
  }
  if (r.tps == null) r.tps = r.tpsAvg;
  if (r.perStep == null) r.perStep = r.perStepAvg;
  if (r.accept == null) r.accept = r.acceptAvg;
  return r;
}

// One finished segment's summary (stored with the message; also what the footer shows)
export function finalStats(seg) {
  const t = seg.timings || {};
  const g = seg.gaudium || {};
  const steps = g.aseq ? stepsFromAseq(g.aseq) : stepsFromSamples(seg.samples || []);
  const prefix = g.prefix || null;
  return {
    method: g.method || null, mock: !!g.mock,
    ttftMs: ttft(seg) == null ? null : Math.round(ttft(seg)), waitMs: waitMs(seg),
    promptN: t.prompt_n || 0, cacheN: t.cache_n || 0, promptMs: t.prompt_ms || 0,
    prefillTps: t.prompt_ms > 0 ? (1000 * (t.prompt_n || 0)) / t.prompt_ms : null,
    n: t.predicted_n || 0, ms: t.predicted_ms || 0,
    tps: t.predicted_ms > 0 ? (1000 * t.predicted_n) / t.predicted_ms : null,
    launches: t.launches || g.steps || 0,
    perStep: (t.launches || g.steps) ? (t.predicted_n || 0) / (t.launches || g.steps) : null,
    accept: t.draft_n > 0 ? (t.draft_n_accepted || 0) / t.draft_n : null,
    msPerStep: g.ms_per_step || (t.launches > 0 ? t.predicted_ms / t.launches : null),
    switchStep: g.switch_step != null && g.switch_step >= 0 ? g.switch_step : null,
    vitMs: g.vit_ms || null, prefixReuse: prefix ? prefix.reuse : null, prefixVitSkipped: prefix ? prefix.vit_skipped : null,
    steps,
  };
}

// Context use: prompt_n + cache_n + predicted_n
export function contextUsed(t) { return t ? (t.prompt_n || 0) + (t.cache_n || 0) + (t.predicted_n || 0) : 0; }

// --- formatting (fixed decimals keep the strip's digits from jumping) -----------------------------------------------
const nf0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
export const fmtInt = (x) => (x == null || !isFinite(x) ? '–' : nf0.format(Math.round(x)));
export const fmt1 = (x) => (x == null || !isFinite(x) ? '–' : x.toFixed(1));
export const fmt2 = (x) => (x == null || !isFinite(x) ? '–' : x.toFixed(2));
export const fmtPct = (x) => (x == null || !isFinite(x) ? '–' : `${Math.round(100 * x)}%`);
export function fmtMs(ms) {
  if (ms == null || !isFinite(ms)) return '–';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
  const m = Math.floor(ms / 60000); return `${m} min ${Math.round((ms % 60000) / 1000)} s`;
}
export function fmtGBs(gbps) {
  if (gbps == null || !isFinite(gbps)) return '–';
  return gbps >= 1000 ? `${(gbps / 1000).toFixed(2)} TB/s` : `${Math.round(gbps)} GB/s`;
}
export function fmtBytes(n) {
  if (n == null) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
export function fmtAgo(ts, now = Date.now()) {
  const s = Math.max(0, Math.round((now - ts) / 1000));
  if (s < 60) return 'now';
  if (s < 3600) return `${Math.floor(s / 60)} min`;
  if (s < 86400) return `${Math.floor(s / 3600)} h`;
  const d = Math.floor(s / 86400); return d < 30 ? `${d} d` : new Date(ts).toLocaleDateString();
}
