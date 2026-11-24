// What the model is, and the optional sections of one request (docs/server.md 7.8). Pure (no DOM), unit-tested.
//
// The engine describes its model once, in its ready state (`model_desc`: arch, dense or MoE, precision per part, drafters, context,
// features); the server passes it on in the hello (`hello.model_desc`) and in the engine state (`engine.info.model_desc`). Each
// request's details (the final chunk's `gaudium`) may carry sections: `spec`, `moe`, `ple`, `qsa`, `prefix`; the HBM flow is the
// live record's `profile.hbm`. The readout shows a panel only when its section (or the description) exists, so an engine or a mock
// that sends only the older fields shows exactly what it showed before.

// The description, normalised; `known` is false when no engine or profile has described the model (an older engine, a parked one
// the server has never seen, the plain mock): then only the server's model name is known.
export function modelDesc(hello, engine) {
  const raw = (engine && engine.info && engine.info.model_desc) || (hello && hello.model_desc) || (hello && hello.engine_info && hello.engine_info.model_desc) || null;
  const d = raw && typeof raw === 'object' ? raw : {};
  const known = !!raw;
  const m = d.moe && typeof d.moe === 'object' ? d.moe : null;
  const p = m && m.placement && typeof m.placement === 'object' ? m.placement : null;
  const f = d.features && typeof d.features === 'object' ? d.features : {};
  const pr = d.precision && typeof d.precision === 'object' ? d.precision : {};
  return {
    known,
    name: d.name || (hello && hello.model) || '',
    arch: d.arch || '',
    kind: d.kind || (known ? (m ? 'moe' : 'dense') : ''),
    layers: d.layers || null,
    attention: d.attention || null,
    nCtx: d.n_ctx || (hello && hello.n_ctx) || 0,
    precision: { weights: pr.weights || '', kv: pr.kv || '', mtp: pr.mtp || '' },
    drafters: Array.isArray(d.drafters) ? d.drafters : [],
    moe: m ? {
      experts: m.experts || 0, topK: m.top_k || 0, width: m.expert_width || 0, form: m.form || '', w8Layers: m.w8_layers || 0,
      placement: p ? {
        mode: p.mode || 'card', cardGb: num(p.card_gb), hostGb: num(p.host_gb), promoteEvery: p.promote_every || 0,
        // the expert cache (docs/notes/moe-expert-cache.md): device slots filled on a miss; null when the engine has no cache section
        cache: p.cache && typeof p.cache === 'object' ? { on: !!p.cache.on, policy: p.cache.policy || '', alloc: p.cache.alloc || '', gate: !!p.cache.gate, upload: !!p.cache.upload, reserve: p.cache.reserve || 0 } : null,
        layers: Array.isArray(p.layers) ? p.layers.filter((x) => Array.isArray(x) && x.length >= 2).map(([l, s]) => ({ layer: l, slots: s })) : [],
      } : null,
    } : null,
    features: {
      vision: f.vision == null ? null : !!f.vision,
      qsa: f.qsa && typeof f.qsa === 'object' ? { denseBelow: f.qsa.dense_below || 0, budget: f.qsa.budget || 0, block: f.qsa.block || 0, heads: f.qsa.heads || 0 } : null,
      ple: f.ple && typeof f.ple === 'object' ? { layer: f.ple.layer, width: f.ple.width || 0, ngram: f.ple.ngram || 0 } : null,
      hc: f.hc && typeof f.hc === 'object' ? { streams: f.hc.streams || 0, rank: f.hc.rank || 0 } : null,
      prefixSlots: f.prefix_slots == null ? null : f.prefix_slots,
    },
    mock: !!d.mock,
  };
}
const num = (x) => (typeof x === 'number' && isFinite(x) ? x : null);

// The drafter families an engine has loaded, by name: 'mtp' (the model's own MTP head) and the external drafters. The name of a
// family comes from the method label (stats.methodInfo); the description adds what only it knows (the MTP head's precision).
const DRAFTER_NAME = { mtp: 'MTP head', dflash: 'DFlash', dflash2: 'DFlash2', dspine: 'DSpine' };
const FAMILY_DRAFTERS = { df: ['dflash', 'dflash2'], ds: ['dspine'], mtp: ['mtp'] };
export function drafterNames(desc) { return (desc.drafters || []).map((x) => DRAFTER_NAME[x] || x); }
export function hasDrafter(desc, fam) {   // fam: 'df' | 'ds' | 'mtp' (stats.js families); unknown description: as before (all may exist)
  if (!desc.known) return true;
  return (FAMILY_DRAFTERS[fam] || [fam]).some((k) => desc.drafters.includes(k));
}
// The name of the 'df' family for this engine (stats.setDfName): "DFlash" when it loaded the DFlash block drafter (--drafter dflash),
// else "DFlash2" (its dflash2 drafter, or a description that does not say)
export function dfFamilyName(desc) { return desc && desc.known && desc.drafters.includes('dflash') && !desc.drafters.includes('dflash2') ? 'DFlash' : 'DFlash2'; }
// "Q8_0 head" for an MTP method when the description names the head's precision
export function mtpHeadLabel(desc) { return desc.precision.mtp ? `${desc.precision.mtp} head` : ''; }

// A short precision tag for the readout's head: GGUF weights -> their main type ("IQ4_XS"), else the first word ("bf16", "fp8g")
// The engine's precision (`fallback`, e.g. "fp8g") says how the dense weights run; the description says what the file holds.
// A GGUF share list ("BF16 0.0 %, IQ3_S 61.4 %, ...") names its largest type, not its first; an FP8 engine adds "FP8".
export function precisionShort(desc, fallback = '') {
  const w = desc.precision.weights;
  const eng = fallback ? String(fallback).split(' ')[0] : '';
  const fp8 = /^fp8/i.test(eng);
  if (!w) return eng;
  if (/^GGUF\b/.test(w)) {
    let best = null, top = -1;
    for (const m of String(w).matchAll(/([A-Za-z0-9_]+)\s+([\d.]+)\s*%/g)) if (+m[2] > top) { top = +m[2]; best = m[1]; }
    if (!best) { const g = /^GGUF\s+(?:[\w.-]*\s+\()?([A-Za-z0-9_]+)/.exec(w); best = g ? g[1] : 'GGUF'; }
    return fp8 ? `${best} · FP8` : best;
  }
  return fp8 && !/fp8/i.test(w) ? 'FP8' : String(w).split(' ')[0];
}

const plural = (n, one, many) => `${n} ${n === 1 ? one : many || `${one}s`}`;
const gb = (x) => (x == null ? '–' : `${x >= 10 ? x.toFixed(1) : x.toFixed(2)} GB`);

// Where the experts live, for the MoE panel and the hello page. null: a dense model (or no description).
export function placementOf(desc) {
  const m = desc.moe; if (!m) return null;
  const p = m.placement || { mode: 'card', cardGb: null, hostGb: null, layers: [], promoteEvery: 0, cache: null };
  const cache = p.mode === 'hybrid' && p.cache && p.cache.on ? p.cache : null;
  // with the expert cache the promoter runs only under its 'promoter' policy; the default LRU cache never promotes
  const promoteEvery = cache && cache.policy !== 'promoter' ? 0 : p.promoteEvery;
  const total = p.cardGb != null && p.hostGb != null ? p.cardGb + p.hostGb : null;
  const share = total ? p.cardGb / total : p.mode === 'card' ? 1 : null;
  let headline;
  if (p.mode === 'hybrid' && cache) headline = `expert cache: ${plural(p.layers.length, 'layer')} keep part of their experts in card slots, filled from host RAM on a miss`;
  else if (p.mode === 'hybrid') headline = `hybrid: ${plural(p.layers.length, 'layer')} share their experts with the host`;
  else if (p.mode === 'host') headline = `${plural(p.layers.length, 'layer')} with every expert on the host`;
  else headline = `all ${m.experts} experts of every layer on the card`;
  const layersText = p.layers.length ? p.layers.map((x) => (p.mode === 'host' ? `${x.layer}` : `${x.layer}:${x.slots}`)).join(' ') : '';
  const short = `${gb(p.cardGb)} card${p.hostGb ? ` · ${gb(p.hostGb)} host` : ''}`;
  return { mode: p.mode, cardGb: p.cardGb, hostGb: p.hostGb, share, headline, layersText, layers: p.layers.length, promoteEvery, cache, short,
    experts: m.experts, topK: m.topK, form: m.form, text: `${gb(p.cardGb)} on the card${p.hostGb ? ` · ${gb(p.hostGb)} on the host` : ''}`,
    split: expertsSplit(p) };
}

// Whether the experts are held in more than one place: the card and host RAM today (hybrid or host layers, the expert cache), the
// card and another device later (any placement mode other than 'card'). Only then do the readout's placement bars say something;
// with every expert on the card they are left out.
export function expertsSplit(placement) {
  const p = placement;
  if (!p || typeof p !== 'object') return false;
  return (p.mode || 'card') !== 'card' || (typeof p.hostGb === 'number' && p.hostGb > 0);
}

// Short facts for the hello page and the card dialog: what this model has that the readout will show.
export function modelFacts(desc) {
  const out = [];
  if (desc.moe) out.push(`MoE: ${desc.moe.experts} experts, top-${desc.moe.topK}`);
  const pl = placementOf(desc);
  if (pl) out.push(pl.mode === 'hybrid' ? `experts: ${gb(pl.cardGb)} card + ${gb(pl.hostGb)} host` : pl.mode === 'host' ? 'experts partly on the host' : 'every expert on the card');
  const dn = drafterNames(desc);
  if (dn.length) out.push(`drafters: ${dn.map((n) => (n === 'MTP head' && desc.precision.mtp ? `MTP head (${desc.precision.mtp})` : n)).join(', ')}`);
  if (desc.features.qsa) out.push(`QSA sparse attention above ${desc.features.qsa.denseBelow.toLocaleString('en-US')} tokens`);
  if (desc.features.ple) out.push('n-gram embedding on the host');
  return out;
}

// The sections of one request, per step. `details`: the request's `gaudium` object (final), `running`: the newest running totals
// (/gaudium/events "sections") while it streams. Each field is null when its section is absent.
export function requestSections(details, running) {
  const g = details || {};
  const r = running || {};
  const moe = g.moe || r.moe || null, ple = g.ple || r.ple || null, qsa = g.qsa || r.qsa || null, spec = g.spec || null, prefix = g.prefix || null;
  const per = (x, n) => (n > 0 && x != null ? x / n : null);
  return {
    moe: moe ? {
      steps: moe.steps || 0,
      hostRowsPerStep: per(moe.host_rows, moe.steps), hostPartsPerStep: per(moe.host_parts, moe.steps),
      hostMsPerStep: per(moe.host_ms, moe.steps), hostMsPerPart: per(moe.host_ms, moe.host_parts),
      boundariesPerStep: per(moe.boundaries, moe.steps), boundaryMsPerStep: per(moe.boundary_ms, moe.steps),
      promotions: moe.promotions || 0, promoMsEach: per(moe.promo_ms, moe.promotions),
      hostWork: (moe.host_rows || 0) > 0 || (moe.boundaries || 0) > 0,
      // the expert cache's totals: routed = expert assignments in the gated layers, missed = those not on the card when routed,
      // gate_waits = layers that waited at their miss gate, fills = experts copied to the card, evictions = slots given up
      cache: moe.cache && typeof moe.cache === 'object' ? {
        routed: moe.cache.routed || 0, missed: moe.cache.missed || 0,
        hitRate: moe.cache.routed > 0 ? 1 - (moe.cache.missed || 0) / moe.cache.routed : null,
        waitsPerStep: per(moe.cache.gate_waits, moe.steps), fillsPerStep: per(moe.cache.fills, moe.steps),
        missedPerStep: per(moe.cache.missed, moe.steps), evictions: moe.cache.evictions || 0,
      } : null,
    } : null,
    ple: ple ? { steps: ple.steps || 0, msPerStep: per(ple.ms, ple.steps) } : null,
    qsa: qsa ? { visible: qsa.visible || 0, denseBelow: qsa.dense_below || 0, sparse: !!qsa.sparse } : null,
    spec: spec ? { drafted: spec.drafted || 0, accepted: spec.accepted || 0, rate: spec.rate == null ? null : spec.rate, tokPerStep: spec.tok_per_step == null ? null : spec.tok_per_step } : null,
    prefix: prefix ? { reuse: prefix.reuse || 0 } : null,
  };
}

// Does this request speculate? The final spec section says so; else the method label (K=0 commits one token per step); unknown
// labels count as speculative (the panels stay as they were).
export function isSpeculative(info, st, sections) {
  if (sections && sections.spec) return sections.spec.drafted > 0;
  if (st && st.accept != null) return true;
  return !(info && info.maxTok === 1);
}

// The reasoning-effort levels the model's chat template accepts, low to high (/props chat_template_caps). Empty: the template has no
// effort setting, or the server does not say; the effort picker is then hidden.
export function effortLevels(props) {
  const c = props && props.chat_template_caps;
  if (!c || c.supports_reasoning_effort !== true || !Array.isArray(c.reasoning_effort_levels)) return [];
  return c.reasoning_effort_levels.filter((l) => typeof l === 'string' && l);
}
// The effort a request is built with: a saved level this model does not offer is the template's default (the saved setting stays,
// for a model that has it).
export function effortFor(effort, levels) { return effort && effort !== 'default' && levels.includes(effort) ? effort : 'default'; }
