// A2UI 1.0 surfaces drawn with the page's own components (the basic catalog). The protocol, data binding, functions and
// checks are in lib/a2ui.js; this file only turns a surface's components into elements and user input into data writes
// and action messages. Nothing the model sends is run as code or inserted as HTML (text goes through the sanitized
// Markdown renderer; URLs are limited to web, mail and media schemes).
import { Component } from 'preact';
import { html, cx } from './h.js';
import { useState, useEffect, useRef } from 'preact/hooks';
import { renderMarkdown } from '../lib/markdown.js';
import { resolve, toText, toBool, isBinding, bindingPath, resolvePath, getAt, runChecks, actionMessage, localAction, parseBlock, safeUrl } from '../lib/a2ui.js';

// ------------------------------------------------------------------------------------------------- icons
// The catalog's icon names as 24x24 line drawings ("f:" = filled).
const circle = 'M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18z';
const dot = (x, y) => `M${x} ${y - 1.2}a1.2 1.2 0 1 0 0 2.4a1.2 1.2 0 0 0 0-2.4z`;
const calendar = 'M4 6h16v14H4zM4 10h16M8 3v4M16 3v4';
const heart = 'M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.5A4 4 0 0 1 19 10c0 5.5-7 10-7 10z';
const star = 'M12 3.5l2.6 5.4l5.9.8l-4.3 4.1l1 5.9L12 16.9l-5.2 2.8l1-5.9L3.5 9.7l5.9-.8z';
const bell = 'M6 17v-6a6 6 0 0 1 12 0v6l1.5 2h-15zM10 21h4';
const eye = 'M2.5 12s3.5-6.5 9.5-6.5s9.5 6.5 9.5 6.5s-3.5 6.5-9.5 6.5S2.5 12 2.5 12zM12 9a3 3 0 1 0 0 6a3 3 0 0 0 0-6z';
const speaker = 'M4 9.5h3.5L12 6v12l-4.5-3.5H4z';
const handset = 'M6 3h3l2 5l-2.5 1.5a11 11 0 0 0 6 6L16 13l5 2v3a2 2 0 0 1-2 2A16 16 0 0 1 3 5a2 2 0 0 1 2-2z';
const ICONS = {
  accountCircle: `${circle}M12 7.5a3 3 0 1 0 0 6a3 3 0 0 0 0-6zM6.2 18.2a6.5 6.5 0 0 1 11.6 0`,
  add: 'M12 5v14M5 12h14', arrowBack: 'M19 12H5M11 6l-6 6l6 6', arrowForward: 'M5 12h14M13 6l6 6l-6 6',
  attachFile: 'M16.5 6.5V16a4.5 4.5 0 0 1-9 0V6a3 3 0 0 1 6 0v9.5a1.5 1.5 0 0 1-3 0V7',
  calendarToday: calendar, call: handset, camera: 'M4 7h3l2-3h6l2 3h3v12H4zM12 10a3.5 3.5 0 1 0 0 7a3.5 3.5 0 0 0 0-7z',
  check: 'M5 12.5l4.5 4.5L19 7.5', close: 'M6 6l12 12M18 6L6 18', delete: 'M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13M10 11v6M14 11v6',
  download: 'M12 4v11M7 11l5 5l5-5M5 20h14', edit: 'M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4', event: `${calendar}M13.5 13.5h3v3h-3z`,
  error: `${circle}M12 7v6M12 16.5v.5`, fastForward: 'M4 6l8 6l-8 6zM13 6l8 6l-8 6z', favorite: `f:${heart}`, favoriteOff: heart,
  folder: 'M3 6h6l2 2h10v11H3z', help: `${circle}M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7M12 16.5v.5`,
  home: 'M4 11l8-7l8 7M6 10v10h4v-6h4v6h4V10', info: `${circle}M12 11v6M12 7.5v.5`,
  locationOn: 'M12 21s-6-6.5-6-11a6 6 0 0 1 12 0c0 4.5-6 11-6 11zM12 7.5a2.5 2.5 0 1 0 0 5a2.5 2.5 0 0 0 0-5z',
  lock: 'M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 0 1 7 0v3', lockOpen: 'M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 0 1 6.8-1',
  mail: 'M3 6h18v12H3zM3 7l9 6l9-6', menu: 'M4 7h16M4 12h16M4 17h16',
  moreVert: `f:${dot(12, 5.5)}${dot(12, 12)}${dot(12, 18.5)}`, moreHoriz: `f:${dot(5.5, 12)}${dot(12, 12)}${dot(18.5, 12)}`,
  notifications: bell, notificationsOff: `${bell}M4 4l16 16`, pause: 'M8 5v14M16 5v14', payment: 'M3 6h18v12H3zM3 10h18M6 15h4',
  person: 'M12 4a4 4 0 1 0 0 8a4 4 0 0 0 0-8zM4.5 20a7.5 7.5 0 0 1 15 0',
  phone: 'M8 2h8a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zM11 18.5h2',
  photo: 'M4 5h16v14H4zM4 16l5-5l4 4l3-3l4 4M15 8.5v.5', play: 'M7 5l12 7l-12 7z', print: 'M7 9V4h10v5M7 17H5V9h14v8h-2M7 14h10v6H7z',
  refresh: 'M19 12a7 7 0 1 1-2.1-5M19 4v4h-4', rewind: 'M20 6l-8 6l8 6zM11 6l-8 6l8 6z',
  search: 'M10.5 4a6.5 6.5 0 1 0 0 13a6.5 6.5 0 0 0 0-13zM15.5 15.5L20 20', send: 'M4 20l17-8L4 4l2.5 8zM6.5 12H13',
  settings: 'M12 9a3 3 0 1 0 0 6a3 3 0 0 0 0-6zM12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1',
  share: 'M17 4a2.5 2.5 0 1 0 0 5a2.5 2.5 0 0 0 0-5zM7 9.5a2.5 2.5 0 1 0 0 5a2.5 2.5 0 0 0 0-5zM17 15a2.5 2.5 0 1 0 0 5a2.5 2.5 0 0 0 0-5zM9.2 11l5.6-3.2M9.2 13l5.6 3.2',
  shoppingCart: `M3 4h3l2.5 11H18l2-8H7${dot(9.5, 19.5)}${dot(17, 19.5)}`, skipNext: 'M6 5l10 7l-10 7zM18 5v14', skipPrevious: 'M18 5L8 12l10 7zM6 5v14',
  star: `f:${star}`, starHalf: `${star}M12 3.5v13.4`, starOff: star, stop: 'M6 6h12v12H6z', upload: 'M12 20V9M7 13l5-5l5 5M5 4h14',
  visibility: eye, visibilityOff: `${eye}M4 4l16 16`, volumeDown: `${speaker}M15.5 9.5a3.5 3.5 0 0 1 0 5`, volumeMute: speaker,
  volumeOff: `${speaker}M16 9.5l5 5M21 9.5l-5 5`, volumeUp: `${speaker}M15.5 9.5a3.5 3.5 0 0 1 0 5M18 7a7 7 0 0 1 0 10`,
  warning: 'M12 4l9.5 16h-19zM12 10v5M12 17.5v.5',
};
const PATH_OK = /^[MmLlHhVvCcSsQqTtAaZz0-9\s,.\-eE+]{1,4000}$/;

export function A2Icon({ name }) {
  let d = '', filled = false, label = '';
  if (name && typeof name === 'object' && typeof name.svgPath === 'string') { if (PATH_OK.test(name.svgPath)) d = name.svgPath; filled = true; } else {
    label = toText(name);
    const v = ICONS[label] || '';
    filled = v.startsWith('f:'); d = filled ? v.slice(2) : v;
  }
  if (!d) return html`<span class="a2-icon missing" role="img" aria-label=${label || 'icon'} title=${label}>•</span>`;
  return html`<span class="a2-icon" role="img" aria-label=${label || 'icon'} title=${label}><svg viewBox="0 0 24 24" fill=${filled ? 'currentColor' : 'none'} stroke=${filled ? 'none' : 'currentColor'} stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d=${d} /></svg></span>`;
}

// ------------------------------------------------------------------------------------------------- helpers
const JUSTIFY = { start: 'flex-start', center: 'center', end: 'flex-end', spaceBetween: 'space-between', spaceAround: 'space-around', spaceEvenly: 'space-evenly', stretch: 'stretch' };
const ALIGN = { start: 'flex-start', center: 'center', end: 'flex-end', stretch: 'stretch' };
const PLAIN = /^[^*_`#[\]<>|~\\$\n]*$/;

const ctxOf = (env, scope, index) => ({ data: env.surface.data, scope, index });
const val = (env, scope, index, v) => resolve(v, ctxOf(env, scope, index));

// A component's value property as [value, set]: bound to the data model, else kept in the component.
function useValue(env, scope, index, prop, fallback) {
  const [local, setLocal] = useState(undefined);
  if (isBinding(prop)) {
    const path = bindingPath(prop, scope);
    const v = getAt(env.surface.data, path);
    return [v === undefined ? fallback : v, (x) => env.rt.setData(env.surface.id, path, x)];
  }
  const v = local !== undefined ? local : val(env, scope, index, prop);
  return [v === undefined ? fallback : v, setLocal];
}

// The checks of an input: its message shows once the user has been in the field (or pressed a button of the surface).
function useChecks(env, c, scope, index) {
  const key = `${c.id}@${scope}`;
  const r = runChecks(c, ctxOf(env, scope, index));
  const shown = !r.valid && (env.surface.allTouched || env.surface.touched.has(key));
  return { ...r, shown, touch: () => env.rt.touch(env.surface.id, key) };
}
const CheckNote = ({ ck }) => (ck.shown && ck.messages.length ? html`<div class="a2-check" role="alert">${ck.messages.join(' ')}</div>` : null);

// the children of a container: a list of ids, or one template drawn per element of a data list
function childList(env, c, scope, index, depth, wrap) {
  const ch = c.children;
  const one = (id, sc, ix, key) => {
    const child = env.surface.components.get(id);
    const node = html`<${Node} key=${key} env=${env} id=${id} scope=${sc} index=${ix} depth=${depth + 1} />`;
    return wrap ? wrap(node, child, key) : node;
  };
  if (Array.isArray(ch)) return ch.filter((id) => typeof id === 'string').map((id, i) => one(id, scope, index, `${id}:${i}`));
  if (ch && typeof ch === 'object' && typeof ch.componentId === 'string') {
    const base = resolvePath(ch.path, scope);
    const list = getAt(env.surface.data, base);
    const keys = Array.isArray(list) ? list.map((_, i) => i) : list && typeof list === 'object' ? Object.keys(list) : [];
    return keys.slice(0, 500).map((k, i) => one(ch.componentId, `${base === '/' ? '' : base}/${String(k).replace(/~/g, '~0').replace(/\//g, '~1')}`, i, `t:${k}`));
  }
  return [];
}
// a child of a Row or Column takes its share of the free space when it has a weight
const weighted = (node, child, key) => (child && typeof child.weight === 'number' && child.weight > 0
  ? html`<div key=${key} class="a2-weight" style=${`flex:${child.weight} 1 0`}>${node}</div>` : node);

// ------------------------------------------------------------------------------------------------- components
function TextC({ env, c, scope, index }) {
  const t = toText(val(env, scope, index, c.text));
  const cls = cx('a2-text', c.variant === 'caption' && 'caption');
  if (PLAIN.test(t)) return html`<div class=${cls}>${t}</div>`;
  return html`<div class=${cx(cls, 'answer')} dangerouslySetInnerHTML=${{ __html: renderMarkdown(t) }}></div>`;
}

function ImageC({ env, c, scope, index }) {
  const url = safeUrl(toText(val(env, scope, index, c.url)), ['http:', 'https:', 'data:']);
  const alt = toText(val(env, scope, index, c.description));
  const [bad, setBad] = useState(false);
  useEffect(() => setBad(false), [url]);
  const fit = { contain: 'contain', cover: 'cover', fill: 'fill', none: 'none', scaleDown: 'scale-down' }[c.fit] || 'cover';
  const variant = ['icon', 'avatar', 'smallFeature', 'mediumFeature', 'largeFeature', 'header'].includes(c.variant) ? c.variant : 'mediumFeature';
  if (!url || bad) return html`<div class=${cx('a2-image', variant, 'missing')} role="img" aria-label=${alt || 'image'}><${A2Icon} name="photo" /><span>${alt || (url ? 'The image did not load' : 'No image')}</span></div>`;
  return html`<img class=${cx('a2-image', variant)} src=${url} alt=${alt} style=${`object-fit:${fit}`} loading="lazy" referrerpolicy="no-referrer" onError=${() => setBad(true)} />`;
}

function MediaC({ env, c, scope, index }) {
  const url = safeUrl(toText(val(env, scope, index, c.url)), ['http:', 'https:', 'data:']);
  const desc = toText(val(env, scope, index, c.description));
  if (!url) return html`<div class="a2-image missing mediumFeature"><span>No media</span></div>`;
  if (c.component === 'Video') {
    const poster = safeUrl(toText(val(env, scope, index, c.posterUrl)), ['http:', 'https:', 'data:']);
    return html`<video class="a2-video" controls preload="metadata" src=${url} poster=${poster || undefined}></video>`;
  }
  return html`<div class="a2-audio">${desc ? html`<div class="a2-text caption">${desc}</div>` : null}<audio controls preload="metadata" src=${url} aria-label=${desc || 'Audio'}></audio></div>`;
}

function Box({ env, c, scope, index, depth }) {
  const row = c.component === 'Row';
  const style = `justify-content:${JUSTIFY[c.justify] || 'flex-start'};align-items:${ALIGN[c.align] || (row ? 'center' : 'stretch')}`;
  return html`<div class=${row ? 'a2-row' : 'a2-col'} style=${style}>${childList(env, c, scope, index, depth, weighted)}</div>`;
}

function ListC({ env, c, scope, index, depth }) {
  const horizontal = c.direction === 'horizontal';
  return html`<div class=${cx('a2-list', horizontal && 'h')} style=${`align-items:${ALIGN[c.align] || 'stretch'}`} role="list">
    ${childList(env, c, scope, index, depth, (node, child, key) => html`<div key=${key} class="a2-item" role="listitem">${node}</div>`)}</div>`;
}

function TabsC({ env, c, scope, index, depth }) {
  const tabs = (Array.isArray(c.tabs) ? c.tabs : []).filter((t) => t && typeof t === 'object');
  const [sel, setSel] = useState(0);
  const cur = Math.min(sel, Math.max(0, tabs.length - 1));
  return html`<div class="a2-tabs">
    <div class="tabs" role="tablist">${tabs.map((t, i) => html`<button type="button" role="tab" class=${cx('tab', i === cur && 'on')} aria-selected=${i === cur} onClick=${() => setSel(i)}>${toText(val(env, scope, index, t.title))}</button>`)}</div>
    ${tabs[cur] ? html`<div class="a2-tab-body" role="tabpanel"><${Node} key=${`tab:${cur}`} env=${env} id=${tabs[cur].child} scope=${scope} index=${index} depth=${depth + 1} /></div>` : null}
  </div>`;
}

function ModalC({ env, c, scope, index, depth }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return undefined;
    const k = (e) => { if (e.key === 'Escape') { e.stopPropagation(); setOpen(false); } };
    window.addEventListener('keydown', k, true);
    return () => window.removeEventListener('keydown', k, true);
  }, [open]);
  // the trigger opens the dialog; a Button used as the trigger does not also send its action
  return html`<span class="a2-modal-trigger" onClickCapture=${(e) => { e.preventDefault(); e.stopPropagation(); setOpen(true); }}>
      <${Node} env=${env} id=${c.trigger} scope=${scope} index=${index} depth=${depth + 1} /></span>
    ${open ? html`<div class="overlay a2-overlay" onMouseDown=${(e) => { if (e.target === e.currentTarget) setOpen(false); }}>
      <div class="dialog narrow a2-dialog" role="dialog" aria-modal="true">
        <div class="a2-dialog-close"><button type="button" class="icon-btn" title="Close (Esc)" aria-label="Close" onClick=${() => setOpen(false)}><${A2Icon} name="close" /></button></div>
        <div class="dlg-body"><${Node} env=${env} id=${c.content} scope=${scope} index=${index} depth=${depth + 1} /></div>
      </div></div>` : null}`;
}

function ButtonC({ env, c, scope, index, depth }) {
  const ctx = ctxOf(env, scope, index);
  const ck = runChecks(c, ctx);
  const sends = !!(c.action && c.action.event);
  const blocked = sends && (env.busy || env.readonly);
  const press = () => {
    env.rt.touchAll(env.surface.id);
    if (!ck.valid || blocked) return;
    const local = localAction(c, ctx);
    if (local && local.openUrl) { window.open(local.openUrl, '_blank', 'noopener,noreferrer'); return; }
    const msg = actionMessage(env.surface, c, ctx);
    if (msg && env.onAction) env.onAction(msg);
  };
  const why = !ck.valid ? ck.messages.join(' ') : blocked ? (env.readonly ? 'Open the newest branch of this chat to use this interface' : 'Wait for the reply to finish') : '';
  const variant = c.variant === 'primary' ? 'primary' : c.variant === 'borderless' ? 'borderless' : '';
  return html`<span class="a2-button-wrap">
    <button type="button" class=${cx('btn a2-button', variant)} disabled=${!ck.valid || blocked} aria-disabled=${!ck.valid || blocked} title=${why} onClick=${press}>
      <${Node} env=${env} id=${c.child} scope=${scope} index=${index} depth=${depth + 1} /></button>
    ${!ck.valid && ck.messages.length && env.surface.allTouched ? html`<span class="a2-check" role="alert">${ck.messages.join(' ')}</span>` : null}
  </span>`;
}

function TextFieldC({ env, c, scope, index }) {
  const [v, set] = useValue(env, scope, index, c.value, '');
  const ck = useChecks(env, c, scope, index);
  const label = toText(val(env, scope, index, c.label));
  const ph = toText(val(env, scope, index, c.placeholder));
  const id = `a2-${env.surface.id}-${c.id}-${scope}`;
  const number = c.variant === 'number';
  const onInput = (e) => { const t = e.target.value; set(number && t.trim() !== '' && isFinite(+t) ? +t : t); };
  const common = { id, value: toText(v), placeholder: ph, onInput, onBlur: ck.touch, 'aria-invalid': ck.shown, disabled: env.readonly };
  return html`<div class=${cx('field a2-field', ck.shown && 'bad')}>
    ${label ? html`<label for=${id}>${label}</label>` : null}
    ${c.variant === 'longText' ? html`<textarea rows="3" ...${common}></textarea>`
      : html`<input type=${number ? 'number' : c.variant === 'obscured' ? 'password' : 'text'} step=${number ? 'any' : undefined} autocomplete="off" ...${common} />`}
    <${CheckNote} ck=${ck} />
  </div>`;
}

function CheckBoxC({ env, c, scope, index }) {
  const [v, set] = useValue(env, scope, index, c.value, false);
  const ck = useChecks(env, c, scope, index);
  return html`<div class="a2-field"><label class="check a2-checkbox"><input type="checkbox" checked=${toBool(v)} disabled=${env.readonly} onChange=${(e) => { set(e.target.checked); ck.touch(); }} />
    <span>${toText(val(env, scope, index, c.label))}</span></label><${CheckNote} ck=${ck} /></div>`;
}

function ChoicePickerC({ env, c, scope, index }) {
  const [raw, set] = useValue(env, scope, index, c.value, []);
  const ck = useChecks(env, c, scope, index);
  const [q, setQ] = useState('');
  const sel = Array.isArray(raw) ? raw.map(String) : raw == null || raw === '' ? [] : [String(raw)];
  const multi = c.variant === 'multipleSelection';
  const label = toText(val(env, scope, index, c.label));
  const opts = (Array.isArray(c.options) ? c.options : []).filter((o) => o && typeof o === 'object').map((o) => ({ value: String(o.value), label: toText(val(env, scope, index, o.label)) || String(o.value) }));
  const shown = c.filterable && q.trim() ? opts.filter((o) => o.label.toLowerCase().includes(q.trim().toLowerCase())) : opts;
  const pick = (value) => { set(multi ? (sel.includes(value) ? sel.filter((x) => x !== value) : [...sel, value]) : [value]); ck.touch(); };
  const name = `a2-${env.surface.id}-${c.id}-${scope}`;
  return html`<fieldset class=${cx('a2-field a2-choice', ck.shown && 'bad')} disabled=${env.readonly}>
    ${label ? html`<legend>${label}</legend>` : null}
    ${c.filterable ? html`<input type="search" class="a2-filter" placeholder="Filter" value=${q} aria-label=${`Filter ${label || 'options'}`} onInput=${(e) => setQ(e.target.value)} />` : null}
    ${c.displayStyle === 'chips'
      ? html`<div class="a2-chips" role=${multi ? 'group' : 'radiogroup'}>${shown.map((o) => html`<button type="button" class=${cx('a2-chip', sel.includes(o.value) && 'on')} role=${multi ? 'checkbox' : 'radio'} aria-checked=${sel.includes(o.value)} onClick=${() => pick(o.value)}>${o.label}</button>`)}</div>`
      : html`<div class="a2-options">${shown.map((o) => html`<label class="check"><input type=${multi ? 'checkbox' : 'radio'} name=${name} checked=${sel.includes(o.value)} onChange=${() => pick(o.value)} /><span>${o.label}</span></label>`)}</div>`}
    ${c.filterable && !shown.length ? html`<div class="a2-text caption">No option matches.</div>` : null}
    <${CheckNote} ck=${ck} />
  </fieldset>`;
}

function SliderC({ env, c, scope, index }) {
  const min = typeof c.min === 'number' ? c.min : 0;
  const max = typeof c.max === 'number' && c.max > min ? c.max : min + 100;
  const [raw, set] = useValue(env, scope, index, c.value, min);
  const ck = useChecks(env, c, scope, index);
  const v = typeof raw === 'number' ? raw : isFinite(+raw) && raw !== '' ? +raw : min;
  const step = Number.isInteger(c.steps) && c.steps > 0 ? (max - min) / c.steps : Number.isInteger(min) && Number.isInteger(max) ? 1 : 'any';
  const label = toText(val(env, scope, index, c.label));
  const id = `a2-${env.surface.id}-${c.id}-${scope}`;
  return html`<div class="field a2-field a2-slider">
    <div class="a2-slider-head">${label ? html`<label for=${id}>${label}</label>` : html`<span></span>`}<output for=${id}>${Number.isInteger(v) ? v : Math.round(v * 100) / 100}</output></div>
    <input id=${id} type="range" min=${min} max=${max} step=${step} value=${Math.min(max, Math.max(min, v))} disabled=${env.readonly} onInput=${(e) => { set(+e.target.value); ck.touch(); }} />
    <${CheckNote} ck=${ck} />
  </div>`;
}

function DateTimeC({ env, c, scope, index }) {
  const [raw, set] = useValue(env, scope, index, c.value, '');
  const ck = useChecks(env, c, scope, index);
  const type = c.enableDate && c.enableTime ? 'datetime-local' : c.enableTime ? 'time' : 'date';
  // the stored value is ISO 8601; the field shows the part it edits
  const shape = (t) => { const s = toText(t); return type === 'date' ? s.slice(0, 10) : type === 'time' ? (/T/.test(s) ? s.split('T')[1] : s).slice(0, 5) : s.replace(/(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$/, '').slice(0, 16); };
  const label = toText(val(env, scope, index, c.label));
  const id = `a2-${env.surface.id}-${c.id}-${scope}`;
  return html`<div class=${cx('field a2-field', ck.shown && 'bad')}>
    ${label ? html`<label for=${id}>${label}</label>` : null}
    <input id=${id} type=${type} value=${shape(raw)} min=${shape(val(env, scope, index, c.min)) || undefined} max=${shape(val(env, scope, index, c.max)) || undefined}
      disabled=${env.readonly} onInput=${(e) => set(e.target.value)} onBlur=${ck.touch} />
    <${CheckNote} ck=${ck} />
  </div>`;
}

const TYPES = {
  Text: TextC, Image: ImageC, Video: MediaC, AudioPlayer: MediaC, Row: Box, Column: Box, List: ListC, Tabs: TabsC, Modal: ModalC, Button: ButtonC,
  TextField: TextFieldC, CheckBox: CheckBoxC, ChoicePicker: ChoicePickerC, Slider: SliderC, DateTimeInput: DateTimeC,
  Icon: ({ env, c, scope, index }) => html`<${A2Icon} name=${val(env, scope, index, c.name)} />`,
  Card: ({ env, c, scope, index, depth }) => html`<div class="a2-card"><${Node} env=${env} id=${c.child} scope=${scope} index=${index} depth=${depth + 1} /></div>`,
  Divider: ({ c }) => html`<div class=${cx('a2-divider', c.axis === 'vertical' && 'v')} role="separator"></div>`,
};

// One component of the surface, by id. A child that is not defined (yet) draws nothing; a component cannot hold itself.
function Node({ env, id, scope = '', index = 0, depth = 0 }) {
  const c = typeof id === 'string' ? env.surface.components.get(id) : null;
  if (!c || depth > 40) return null;
  const T = TYPES[c.component];
  if (!T) return null;
  const acc = c['@accessibility'];
  if (acc && toBool(val(env, scope, index, acc.hidden))) return html`<div aria-hidden="true"><${T} env=${env} c=${c} scope=${scope} index=${index} depth=${depth} /></div>`;
  return html`<${T} env=${env} c=${c} scope=${scope} index=${index} depth=${depth} />`;
}

// ------------------------------------------------------------------------------------------------- a surface and a block
// a fault in drawing one surface must not take the chat down with it
class Guard extends Component {
  constructor() { super(); this.state = { failed: null }; }
  componentDidCatch(e) { this.setState({ failed: String((e && e.message) || e) }); }
  render(p, s) { return s.failed ? html`<div class="err"><b>Interface</b>: it could not be drawn (${s.failed}).</div>` : p.children; }
}

// re-render when the runtime changes (a block applied, an input written)
function useRuntime(rt) {
  const [, force] = useState(0);
  useEffect(() => (rt ? rt.subscribe(() => force((x) => x + 1)) : undefined), [rt]);
}

function pretty(raw) {
  const r = parseBlock(raw);
  return r.error ? String(raw || '').trim() : JSON.stringify(r.messages, null, 2);
}

function SurfaceView({ rt, surface, onAction, busy, readonly }) {
  const env = { rt, surface, onAction, busy, readonly };
  if (!surface.components.has('root')) return html`<div class="a2-wait"><span class="dot warn pulse"></span>Waiting for the interface's root component…</div>`;
  return html`<div class="a2-surface" data-a2ui-surface=${surface.id}><${Guard}><${Node} env=${env} id="root" /></${Guard}></div>`;
}

// One interface block of a reply. It draws the surfaces this block is the latest to touch; a surface that a later block
// touched again is drawn there, and this block points to it.
export function A2uiBlock({ rt, blockKey, raw, closed, live, onAction, onReport, busy }) {
  useRuntime(rt);
  const [showJson, setShowJson] = useState(false);
  const ref = useRef(null);
  if (!closed) {
    if (live) return html`<div class="a2-wait" role="status"><span class="dot warn pulse"></span>Building the interface…</div>`;
    return html`<div class="a2-block broken"><div class="err"><b>Interface</b>: the reply ended inside an interface block, so it was not drawn.</div><pre class="a2-json">${String(raw || '').trim()}</pre></div>`;
  }
  const done = rt && rt.blocks.get(blockKey);
  if (!done) return null;
  const res = done.result;
  if (res.parseError) return html`<div class="a2-block broken"><div class="err"><b>Interface</b>: ${res.parseError}
      ${onReport ? html`<div class="err-actions"><button type="button" class="btn" disabled=${busy} onClick=${() => onReport([{ code: 'VALIDATION_FAILED', surfaceId: '', path: '', message: res.parseError }])}>Ask the model to fix it</button></div>` : null}</div>
    <pre class="a2-json">${String(raw || '').trim()}</pre></div>`;
  const ids = [...new Set([...res.created, ...res.touched])];
  const jump = (id) => { const el = document.querySelector(`[data-a2ui-surface="${CSS.escape(id)}"]`); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' }); };
  return html`<div class="a2-block" ref=${ref}>
    ${ids.map((id) => {
      const s = rt.surfaces.get(id);
      if (!s) return html`<div class="a2-stub" key=${id}>This interface was closed later in the chat.</div>`;
      if (s.anchor !== blockKey) return html`<div class="a2-stub" key=${id}>This interface was updated later in the chat. <button type="button" class="a2-link" onClick=${() => jump(id)}>Show it</button></div>`;
      return html`<${SurfaceView} key=${id} rt=${rt} surface=${s} onAction=${onAction} busy=${busy} />`;
    })}
    ${res.deleted.map((id) => html`<div class="a2-stub" key=${`del:${id}`}>The interface “${id}” was closed.</div>`)}
    ${res.errors.length ? html`<details class="a2-problems"><summary>${res.errors.length === 1 ? 'The model made 1 mistake in this interface' : `The model made ${res.errors.length} mistakes in this interface`}</summary>
      <ul>${res.errors.slice(0, 12).map((e) => html`<li><code>${e.path || '/'}</code> ${e.message}</li>`)}</ul>
      ${onReport ? html`<button type="button" class="btn" disabled=${busy} onClick=${() => onReport(res.errors)}>Ask the model to fix it</button>` : null}</details>` : null}
    <div class="a2-foot"><span title=${res.repaired ? 'The model left a bracket out or put one too many; the JSON was balanced before it was read' : ''}>Interface${ids.length ? ` · ${ids.join(', ')}` : ''}${res.repaired ? ' · brackets repaired' : ''}</span>
      <button type="button" class="a2-link" aria-expanded=${showJson} onClick=${() => setShowJson(!showJson)}>${showJson ? 'Hide the JSON' : 'Show the JSON'}</button></div>
    ${showJson ? html`<pre class="a2-json">${pretty(raw)}</pre>` : null}
  </div>`;
}

