// A2UI 1.0 (agent-to-user-interface) on the client: finding the model's interface blocks in a reply, checking and applying
// their messages to surfaces, data binding, the basic catalog's functions and checks, and the messages sent back.
// Pure (no DOM): unit-tested in ui/test/a2ui.test.mjs. The components that draw a surface are in ui/a2ui.js.
//
// The wire format between the model and this page: the model writes `<a2ui-json>` ... `</a2ui-json>` blocks inside its
// ordinary content, each holding a JSON array of A2UI messages; a pressed button goes back as a user turn holding one
// renderer-to-agent message in the same tags. The server adds the instructions that teach the model this when a request
// declares `gaudium_a2ui` (see /gaudium/a2ui).

export const A2UI_VERSION = '1.0';
export const BASIC_CATALOG = 'https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json';
export const OPEN_TAG = '<a2ui-json>';
export const CLOSE_TAG = '</a2ui-json>';
const MAX_DEPTH = 64;

// ------------------------------------------------------------------------------------------------- blocks in a reply
// Splits a reply into text and interface blocks: [{t: 'text', text} | {t: 'a2ui', raw, closed}].
// live: the reply is still streaming, so a tail that may be the start of a tag is held back and an open block stays open.
// Backticks or a code fence written around a tag are dropped (a model that "quotes" its block still gets it drawn).
export function splitContent(content, { live = false } = {}) {
  const src = String(content || '');
  if (src.indexOf(OPEN_TAG) < 0) { const t = live ? src.slice(0, src.length - heldTail(src)) : src; return t ? [{ t: 'text', text: t }] : []; }
  const parts = [];
  const pushText = (t, { before = false, after = false } = {}) => {
    if (before) t = t.replace(/(?:^|\n)[ \t]*`{1,3}[a-zA-Z]*[ \t]*\n?[ \t]*$/, (m) => (m.startsWith('\n') ? '\n' : ''));   // ```json or ` right before a tag
    if (after) t = t.replace(/^[ \t]*\n?[ \t]*`{1,3}[ \t]*(?=\n|$)/, '');                                                // ``` or ` right after one
    if (t.trim()) parts.push({ t: 'text', text: t });
  };
  let i = 0, afterBlock = false;
  for (;;) {
    const o = src.indexOf(OPEN_TAG, i);
    if (o < 0) break;
    pushText(src.slice(i, o), { before: true, after: afterBlock });
    const c = src.indexOf(CLOSE_TAG, o + OPEN_TAG.length);
    if (c < 0) { parts.push({ t: 'a2ui', raw: src.slice(o + OPEN_TAG.length), closed: false }); return parts; }
    parts.push({ t: 'a2ui', raw: src.slice(o + OPEN_TAG.length, c), closed: true });
    i = c + CLOSE_TAG.length; afterBlock = true;
  }
  let rest = src.slice(i);
  if (live) rest = rest.slice(0, rest.length - heldTail(rest));
  pushText(rest, { after: afterBlock });
  return parts;
}

// How many characters at the end of `text` may still be the start of the opening tag (they are not shown yet).
export function heldTail(text) {
  const max = Math.min(text.length, OPEN_TAG.length - 1);
  for (let n = max; n > 0; n--) if (OPEN_TAG.startsWith(text.slice(text.length - n))) return n;
  return 0;
}

export const hasA2ui = (content) => String(content || '').indexOf(OPEN_TAG) >= 0;

// A model's most common slip in deeply nested JSON is a closing bracket too few or too many. This balances the brackets
// of `text` (outside strings): a closer that does not match what is open first closes what stands in its way, a closer
// with nothing to close is dropped, what is still open at the end is closed, and a comma before a closer goes. Where an
// object expects a property name and an object or a list opens instead (`{"a":{"b":1},{"d":2}`), closers are missing
// before that comma: the objects are closed back to the list the new element belongs to.
export function balanceJson(text) {
  const t = String(text || '');
  const stack = []; let out = '', inStr = false, wantKey = false;
  const close = { '{': '}', '[': ']' };
  const dropComma = () => { out = out.replace(/,\s*$/, ''); };
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (inStr) { out += c; if (c === '\\' && i + 1 < t.length) out += t[++i]; else if (c === '"') inStr = false; continue; }
    if (c === '"') { inStr = true; wantKey = false; out += c; continue; }
    if (c === '{' || c === '[') {
      if (wantKey && stack.includes('[')) {
        out = out.replace(/,\s*$/, '');
        while (stack[stack.length - 1] !== '[') out += close[stack.pop()];
        out += ',';
      }
      stack.push(c); out += c; wantKey = false; continue;
    }
    if (c === '}' || c === ']') {
      const want = c === '}' ? '{' : '[';
      if (!stack.includes(want)) continue;
      dropComma();
      while (stack[stack.length - 1] !== want) out += close[stack.pop()];
      stack.pop(); out += c; wantKey = false; continue;
    }
    if (c === ',') wantKey = stack[stack.length - 1] === '{';
    else if (!/\s/.test(c)) wantKey = false;
    out += c;
  }
  if (inStr) out += '"';
  dropComma();
  while (stack.length) out += close[stack.pop()];
  return out;
}

// The messages of one block: {messages} or {error}; `repaired` when the brackets had to be balanced first. A block is a
// JSON array of messages; one message object and one message per line are read too, and a code fence inside the tags
// is dropped.
export function parseBlock(raw) {
  let t = String(raw || '').trim();
  t = t.replace(/^`{1,3}[a-zA-Z]*\s*/, '').replace(/\s*`{1,3}$/, '').trim();
  if (!t) return { messages: [] };
  const asList = (v) => (Array.isArray(v) ? v : v && typeof v === 'object' ? [v] : null);
  let first = null;
  try {
    const list = asList(JSON.parse(t));
    if (!list) return { error: 'The block is not a JSON array of messages.' };
    return { messages: list };
  } catch (e) { first = e; }
  const lines = t.split('\n').map((l) => l.trim().replace(/,$/, '')).filter(Boolean);
  if (lines.length > 1) {
    try { return { messages: lines.flatMap((l) => asList(JSON.parse(l)) || []) }; } catch { /* not one message per line */ }
  }
  try {
    const list = asList(JSON.parse(balanceJson(t)));
    if (list) return { messages: list, repaired: true };
  } catch { /* reported below */ }
  return { error: `The block is not valid JSON (${String(first.message || first).slice(0, 160)}).` };
}

// A model that has tools at hand sometimes sends its interface as a call of a tool named after the tag instead of
// writing the block. `callIsA2ui` knows such a call by its name; `blockFromCall` finds the messages in its arguments
// (the array itself, one message, or either of them under any argument name, also as a JSON string) and returns them
// as a block for the reply's text, or null when the arguments hold no message. A call named after a message
// (`createSurface`) carries that message's body as its arguments.
const MESSAGE_KEYS = ['createSurface', 'updateComponents', 'updateDataModel', 'deleteSurface'];
const messageKey = (name) => MESSAGE_KEYS.find((k) => k.toLowerCase() === String(name || '').trim().replace(/[-_ ]/g, '').toLowerCase());
// Beside other tools the interface is offered as a tool of its own (a model with tools at hand reaches for one, and loops
// over the others when there is none for this). The call never runs: `adoptCalls` turns it into the block it stands for.
export const RENDER_TOOL = {
  name: 'render_a2ui',
  description: 'Shows an interactive interface (A2UI) to the user. Same as writing an <a2ui-json> block: do one or the other for an interface, never both. '
    + 'Creates the surface `surfaceId` from `components` (the flat component list, one with id "root") and `dataModel`, as the system instructions describe for createSurface. '
    + 'To change a surface that exists, pass its surfaceId again with the components to add or replace and/or the new dataModel. After the call your turn ends and the user sees the interface.',
  inputSchema: { type: 'object', properties: { surfaceId: { type: 'string' }, components: { type: 'array', items: { type: 'object' } }, dataModel: { type: 'object' } }, required: ['surfaceId', 'components'] },
};
export const callIsA2ui = (name) => /^(?:render[-_ ]?)?a2ui(?:[-_ ]?json)?$/i.test(String(name || '').trim()) || !!messageKey(name);
const isMessage = (v) => !!v && typeof v === 'object' && !Array.isArray(v) && MESSAGE_KEYS.some((k) => v[k] && typeof v[k] === 'object');
function messagesIn(v, depth) {
  if (depth > 4 || v == null) return null;
  if (typeof v === 'string') {
    if (!/[[{]/.test(v)) return null;
    const inner = v.replace(new RegExp(`${OPEN_TAG}|${CLOSE_TAG}`, 'g'), '');
    const r = parseBlock(inner);
    return r.messages ? messagesIn(r.messages, depth + 1) : null;
  }
  if (typeof v !== 'object') return null;
  if (isMessage(v)) return [v];
  if (Array.isArray(v)) {
    if (v.some(isMessage)) return v.filter(isMessage);
    for (const x of v) { const m = messagesIn(x, depth + 1); if (m) return m; }
    return null;
  }
  for (const k of Object.keys(v)) { const m = messagesIn(v[k], depth + 1); if (m) return m; }
  return null;
}
export function blockFromCall(args, name, known) {
  let msgs = messagesIn(typeof args === 'string' ? args : args || null, 0);
  if (!msgs) {
    let body = args;
    if (typeof body === 'string') { try { body = JSON.parse(body); } catch { try { body = JSON.parse(balanceJson(body)); } catch { body = null; } } }
    const key = messageKey(name);
    if (body && typeof body === 'object' && !Array.isArray(body) && typeof body.surfaceId === 'string') {
      if (key) msgs = [{ version: '1.0', [key]: body }];
      else {   // the render tool: a new surface, or components and data for one that exists
        const list = (v) => { if (typeof v === 'string') { const r = parseBlock(v); return r.messages || null; } return Array.isArray(v) ? v : null; };
        const obj = (v) => { if (typeof v === 'string') { try { v = JSON.parse(balanceJson(v)); } catch { return undefined; } } return v && typeof v === 'object' && !Array.isArray(v) ? v : undefined; };
        const components = list(body.components); const dataModel = obj(body.dataModel !== undefined ? body.dataModel : body.data);
        const id = body.surfaceId;
        if (known && known.has(id)) {
          msgs = [];
          if (components && components.length) msgs.push({ version: '1.0', updateComponents: { surfaceId: id, components } });
          if (dataModel) msgs.push({ version: '1.0', updateDataModel: { surfaceId: id, value: dataModel } });
        } else if (components && components.length) msgs = [{ version: '1.0', createSurface: { surfaceId: id, catalogId: BASIC_CATALOG, components, ...(dataModel ? { dataModel } : {}) } }];
      }
    }
  }
  return msgs && msgs.length ? `${OPEN_TAG}\n${JSON.stringify(msgs)}\n${CLOSE_TAG}` : null;
}
// The calls of a finished reply: those that are an interface move into the text. `known` is the set of surface ids that
// exist already; a surface a block of this same reply creates counts as known too (a call for it updates it). When the
// reply already holds a block, an interface call with nothing to draw is a model that did both: the call is dropped
// (counted in `moved` and `dropped`) instead of being answered with an error that starts another turn.
// Returns {content, calls, moved, dropped}.
export function adoptCalls(content, calls, known) {
  let text = String(content || ''), moved = 0, dropped = 0; const rest = [];
  const ids = new Set(known || []);
  const learn = (src) => {
    for (const p of splitContent(src)) {
      if (p.t !== 'a2ui') continue;
      for (const m of parseBlock(p.raw).messages || []) if (m && m.createSurface && typeof m.createSurface.surfaceId === 'string') ids.add(m.createSurface.surfaceId);
    }
  };
  learn(text);
  for (const c of calls || []) {
    const isUi = callIsA2ui(c && c.name);
    const b = isUi ? blockFromCall(c.arguments, c.name, ids) : null;
    if (!b) { if (isUi && hasA2ui(text)) { moved++; dropped++; } else rest.push(c); continue; }
    text = `${text.replace(/\s+$/, '')}${text.trim() ? '\n\n' : ''}${b}`; moved++;
    learn(b);
  }
  return { content: text, calls: rest, moved, dropped };
}

// ------------------------------------------------------------------------------------------------- JSON Pointer (RFC 6901)
const unesc = (s) => s.replace(/~1/g, '/').replace(/~0/g, '~');
export function pointerTokens(path) {
  const p = String(path == null ? '' : path);
  if (p === '' || p === '/') return [];
  return (p[0] === '/' ? p.slice(1) : p).split('/').map(unesc);
}
// A binding's path in a scope: `/a/b` is absolute; anything else is relative to the scope (a template's current element).
export function resolvePath(path, scope = '') {
  const p = String(path == null ? '' : path);
  if (p[0] === '/') return p;
  if (p === '' || p === '.') return scope || '/';
  return `${scope}/${p.replace(/^\.\//, '')}`;
}
export function getAt(data, path) {
  let v = data;
  for (const k of pointerTokens(path)) {
    if (v == null || typeof v !== 'object') return undefined;
    v = v[k];
  }
  return v;
}
// Sets the value at `path` (creating what is missing on the way); `undefined` or null removes the key. Returns the new root.
export function setAt(data, path, value) {
  const keys = pointerTokens(path);
  const remove = value === undefined || value === null;
  if (!keys.length) return remove ? {} : value;
  const root = data && typeof data === 'object' ? data : {};
  let v = root;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    if (v[k] == null || typeof v[k] !== 'object') {
      if (remove) return root;
      v[k] = /^\d+$/.test(keys[i + 1]) ? [] : {};
    }
    v = v[k];
  }
  const last = keys[keys.length - 1];
  if (remove) { if (Array.isArray(v) && /^\d+$/.test(last)) v.splice(+last, 1); else delete v[last]; } else v[last] = value;
  return root;
}

// ------------------------------------------------------------------------------------------------- the basic catalog
// Per component type: the properties it needs, the properties that name one child and those that name a list of children.
export const COMPONENTS = {
  Text: { need: ['text'] }, Image: { need: ['url'] }, Icon: { need: ['name'] }, Video: { need: ['url'] }, AudioPlayer: { need: ['url'] },
  Row: { need: ['children'], lists: ['children'] }, Column: { need: ['children'], lists: ['children'] }, List: { need: ['children'], lists: ['children'] },
  Card: { need: ['child'], one: ['child'] }, Tabs: { need: ['tabs'] }, Modal: { need: ['trigger', 'content'], one: ['trigger', 'content'] },
  Divider: { need: [] }, Button: { need: ['child', 'action'], one: ['child'] }, TextField: { need: ['label', 'value'] },
  CheckBox: { need: ['label', 'value'] }, ChoicePicker: { need: ['options', 'value'] }, Slider: { need: ['value', 'max'] }, DateTimeInput: { need: ['value'] },
};
export const INPUTS = new Set(['TextField', 'CheckBox', 'ChoicePicker', 'Slider', 'DateTimeInput']);

// The ids a component refers to (static children only; a template's component is listed with `template: true`).
export function childRefs(c) {
  const spec = COMPONENTS[c.component] || {};
  const out = [];
  for (const k of spec.one || []) if (typeof c[k] === 'string') out.push({ id: c[k], prop: k });
  for (const k of spec.lists || []) {
    const v = c[k];
    if (Array.isArray(v)) v.forEach((id, i) => { if (typeof id === 'string') out.push({ id, prop: `${k}/${i}` }); });
    else if (v && typeof v === 'object' && typeof v.componentId === 'string') out.push({ id: v.componentId, prop: `${k}/componentId`, template: true });
  }
  if (c.component === 'Tabs' && Array.isArray(c.tabs)) c.tabs.forEach((t, i) => { if (t && typeof t.child === 'string') out.push({ id: t.child, prop: `tabs/${i}/child` }); });
  return out;
}

// ------------------------------------------------------------------------------------------------- surfaces
// The interfaces of one conversation. Messages are applied in order; a surface keeps its components (by id), its data
// model, and `anchor`: the key of the latest block that touched it (where it is drawn).
export class A2uiRuntime {
  constructor() { this.surfaces = new Map(); this.log = []; this.blocks = new Map(); this.version = 0; this.subs = new Set(); }
  subscribe(fn) { this.subs.add(fn); return () => this.subs.delete(fn); }
  bump() { this.version++; for (const f of [...this.subs]) f(this.version); }
  // (a block is applied while the page renders: the components hear of it right after, not in the middle of that render)
  bumpSoon() { this.version++; if (this.pending) return; this.pending = true; queueMicrotask(() => { this.pending = false; for (const f of [...this.subs]) f(this.version); }); }

  // Applies the block `raw` under `key` (once). Returns what the block did: {created, touched, deleted, errors, parseError}.
  applyBlock(key, raw) {
    const done = this.blocks.get(key);
    if (done && done.raw === raw) return done.result;
    const result = { created: [], touched: [], deleted: [], errors: [], parseError: null };
    const parsed = parseBlock(raw);
    if (parsed.error) result.parseError = parsed.error;
    else { result.repaired = !!parsed.repaired; parsed.messages.forEach((m, i) => this.applyMessage(m, key, result, i)); }
    for (const id of new Set([...result.created, ...result.touched])) {
      const s = this.surfaces.get(id);
      if (s) { s.anchor = key; result.errors.push(...validateSurface(s)); }
    }
    this.blocks.set(key, { raw, result });
    this.log.push({ key, raw });
    this.bumpSoon();
    return result;
  }

  applyMessage(m, key, result, index = 0) {
    const fail = (surfaceId, path, message, code = 'VALIDATION_FAILED') => result.errors.push({ code, surfaceId: surfaceId || '', path, message });
    if (!m || typeof m !== 'object' || Array.isArray(m)) return fail('', `/${index}`, 'A message must be a JSON object.');
    const kind = ['createSurface', 'updateComponents', 'updateDataModel', 'deleteSurface'].find((k) => m[k] && typeof m[k] === 'object');
    if (!kind) {
      if (m.callRendererFunction || m.agentFunctionResponse) return fail('', `/${index}`, 'callRendererFunction and agentFunctionResponse are not supported by this client.');
      return fail('', `/${index}`, 'A message needs one of createSurface, updateComponents, updateDataModel, deleteSurface.');
    }
    const p = m[kind]; const id = p.surfaceId;
    if (typeof id !== 'string' || !id) return fail('', `/${index}/${kind}/surfaceId`, 'surfaceId must be a non-empty string.');
    if (kind === 'createSurface') {
      if (this.surfaces.has(id)) return fail(id, `/${index}/createSurface/surfaceId`, `The surface "${id}" already exists: use updateComponents or updateDataModel, or a new surfaceId.`);
      if (p.catalogId != null && p.catalogId !== BASIC_CATALOG) fail(id, `/${index}/createSurface/catalogId`, `Unknown catalog "${String(p.catalogId).slice(0, 120)}": drawn with the basic catalog.`);
      const s = { id, catalogId: BASIC_CATALOG, components: new Map(), data: {}, sendDataModel: p.sendDataModel === true, origin: key, anchor: key, touched: new Set() };
      this.surfaces.set(id, s); result.created.push(id);
      if (p.components != null) this.putComponents(s, p.components, result, `/${index}/createSurface/components`);
      if (p.dataModel != null) {
        if (typeof p.dataModel === 'object' && !Array.isArray(p.dataModel)) s.data = clone(p.dataModel);
        else fail(id, `/${index}/createSurface/dataModel`, 'dataModel must be an object.');
      }
      return undefined;
    }
    const s = this.surfaces.get(id);
    if (!s) return fail(id, `/${index}/${kind}/surfaceId`, `There is no surface "${id}" (createSurface comes first).`);
    if (kind === 'deleteSurface') { this.surfaces.delete(id); result.deleted.push(id); return undefined; }
    result.touched.push(id);
    if (kind === 'updateComponents') return this.putComponents(s, p.components, result, `/${index}/updateComponents/components`);
    if (!('value' in p)) return fail(id, `/${index}/updateDataModel/value`, 'updateDataModel needs "value" (null deletes).');
    const path = p.path == null || p.path === '' ? '/' : String(p.path);
    if (path === '/') s.data = p.value && typeof p.value === 'object' && !Array.isArray(p.value) ? clone(p.value) : {};
    else s.data = setAt(s.data, path, clone(p.value));
    return undefined;
  }

  putComponents(s, list, result, at) {
    const fail = (path, message) => result.errors.push({ code: 'VALIDATION_FAILED', surfaceId: s.id, path, message });
    if (!Array.isArray(list)) return fail(at, 'components must be a list.');
    list.forEach((c, i) => {
      const here = `${at}/${i}`;
      if (!c || typeof c !== 'object' || Array.isArray(c)) return fail(here, 'A component must be an object (children are named by id, never nested).');
      if (typeof c.id !== 'string' || !c.id) return fail(`${here}/id`, 'A component needs a string "id".');
      const spec = COMPONENTS[c.component];
      if (!spec) return fail(`${here}/component`, `Unknown component type "${String(c.component).slice(0, 60)}".`);
      for (const k of spec.need) if (c[k] === undefined) fail(`${here}/${k}`, `${c.component} "${c.id}" needs "${k}".`);
      for (const k of [...(spec.one || []), ...(spec.lists || [])]) {
        const v = c[k];
        if (v && typeof v === 'object' && !Array.isArray(v) && v.component) fail(`${here}/${k}`, `"${k}" of "${c.id}" holds a component object: give the child its own entry in the list and put its id here.`);
        if (Array.isArray(v) && v.some((x) => typeof x !== 'string')) fail(`${here}/${k}`, `"${k}" of "${c.id}" must list component ids (strings).`);
      }
      s.components.set(c.id, c);
      return undefined;
    });
    return undefined;
  }

  // The user changed an input: write it to the surface's data model.
  setData(surfaceId, path, value) {
    const s = this.surfaces.get(surfaceId);
    if (!s) return;
    s.data = setAt(s.data, path, value);
    this.bump();
  }
  touch(surfaceId, key) { const s = this.surfaces.get(surfaceId); if (s && !s.touched.has(key)) { s.touched.add(key); this.bump(); } }
  touchAll(surfaceId) { const s = this.surfaces.get(surfaceId); if (s && !s.allTouched) { s.allTouched = true; this.bump(); } }
}

const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));

// What is wrong with a surface as a whole: no root, a child that does not exist, a component inside itself.
export function validateSurface(s) {
  const errs = [];
  const fail = (path, message) => errs.push({ code: 'VALIDATION_FAILED', surfaceId: s.id, path, message });
  if (!s.components.size) return errs;
  if (!s.components.has('root')) fail('/components', 'No component has the id "root".');
  for (const c of s.components.values()) {
    for (const r of childRefs(c)) if (!s.components.has(r.id)) fail(`/components/${c.id}/${r.prop}`, `"${c.id}" refers to "${r.id}", which is not defined.`);
  }
  const state = new Map();   // 1 = on the path, 2 = done
  const walk = (id) => {
    if (state.get(id) === 2 || !s.components.has(id)) return;
    if (state.get(id) === 1) { fail(`/components/${id}`, `"${id}" contains itself.`); return; }
    state.set(id, 1);
    for (const r of childRefs(s.components.get(id))) walk(r.id);
    state.set(id, 2);
  };
  walk('root');
  return errs;
}

// ------------------------------------------------------------------------------------------------- dynamic values
// ctx: {data, scope, index, action}. scope: the JSON Pointer of the template element being drawn ('' at the root);
// index: its position; action: the value is resolved for a user gesture (openUrl is allowed only then).
const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
export const isBinding = (v) => isObj(v) && (typeof v['@path'] === 'string' || (typeof v.path === 'string' && Object.keys(v).length === 1));
export const isCall = (v) => isObj(v) && (typeof v['@call'] === 'string' || (typeof v.call === 'string' && Object.keys(v).every((k) => k === 'call' || k === 'args' || k === 'returnType')));
export const bindingPath = (v, scope = '') => (isBinding(v) ? resolvePath(v['@path'] != null ? v['@path'] : v.path, scope) : null);

export function resolve(v, ctx, depth = 0) {
  if (depth > MAX_DEPTH) return undefined;
  if (v === null || typeof v !== 'object') return v;
  if (Array.isArray(v)) return v.map((x) => resolve(x, ctx, depth + 1));
  if (isBinding(v)) return getAt(ctx.data, bindingPath(v, ctx.scope));
  if (isCall(v)) return callFunction(v['@call'] != null ? v['@call'] : v.call, v.args || {}, ctx, depth + 1);
  const out = {};
  for (const [k, x] of Object.entries(v)) out[k.startsWith('@@') ? k.slice(1) : k] = resolve(x, ctx, depth + 1);
  return out;
}

export function toText(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'object') { try { return JSON.stringify(v); } catch { return ''; } }
  return String(v);
}
// The coercion of anything to a boolean (the protocol's table).
export function toBool(v) {
  if (typeof v === 'boolean') return v;
  if (isObj(v) && typeof v.valid === 'boolean') return v.valid;
  if (typeof v === 'string') return v.toLowerCase() === 'true';
  if (typeof v === 'number') return v !== 0 && !Number.isNaN(v);
  return false;
}
const toNum = (v) => (typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' && isFinite(+v) ? +v : NaN);
const empty = (v) => v === null || v === undefined || v === '' || v === false || (Array.isArray(v) && !v.length) || (typeof v === 'string' && !v.trim());

// A pattern of the protocol's regex dialect that is safe to run, or null (too long, a construct that is not allowed, or
// a shape that can backtrack without bound).
export function safeRegex(pattern) {
  const p = String(pattern == null ? '' : pattern);
  if ([...p].length > 256) return null;
  if (/\\[1-9]|\\k<|\(\?<[=!]|\(\?<[A-Za-z]|\\[pP]\{|\\c[A-Za-z]|\(\?[a-z-]+[:)]/.test(p)) return null;
  // a quantified group that itself holds an unbounded quantifier, or two adjacent unbounded atoms of one class
  if (/\((?:[^()\\]|\\.)*(?:[+*]|\{\d+,\})(?:[^()\\]|\\.)*\)(?:[+*]|\{\d+,\d*\})/.test(p)) return null;
  if (/\((?:[^()|\\]|\\.)+\|(?:[^()\\]|\\.)+\)[+*]/.test(p) && /\((\w)[^|]*\|\1/.test(p)) return null;
  if (/(\\[dws]|\.)[+*]\\s\*\1[+*]|(\\[dws]|\.)[+*]\2[+*]/.test(p)) return null;
  try { return new RegExp(p, 'u'); } catch { return null; }
}

const two = (n) => String(n).padStart(2, '0');
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
// A date or time in a Unicode date pattern (the tokens the catalog lists; text in single quotes is literal).
export function formatDate(value, format) {
  let d;
  if (value instanceof Date) d = value;
  else if (typeof value === 'number') d = new Date(value < 1e11 ? value * 1000 : value);
  else {
    const s = String(value == null ? '' : value).trim();
    if (!s) return '';
    d = /^\d{2}:\d{2}(:\d{2})?$/.test(s) ? new Date(`1970-01-01T${s}`) : /^\d{4}-\d{2}-\d{2}$/.test(s) ? new Date(`${s}T00:00:00`) : new Date(s);
  }
  if (!d || Number.isNaN(d.getTime())) return toText(value);
  const h = d.getHours();
  const tok = {
    yyyy: () => String(d.getFullYear()), yy: () => two(d.getFullYear() % 100), y: () => String(d.getFullYear()),
    MMMM: () => MONTHS[d.getMonth()], MMM: () => MONTHS[d.getMonth()].slice(0, 3), MM: () => two(d.getMonth() + 1), M: () => String(d.getMonth() + 1),
    dd: () => two(d.getDate()), d: () => String(d.getDate()), EEEE: () => DAYS[d.getDay()], EEE: () => DAYS[d.getDay()].slice(0, 3), EE: () => DAYS[d.getDay()].slice(0, 3), E: () => DAYS[d.getDay()].slice(0, 3),
    HH: () => two(h), H: () => String(h), hh: () => two(h % 12 || 12), h: () => String(h % 12 || 12), mm: () => two(d.getMinutes()), m: () => String(d.getMinutes()),
    ss: () => two(d.getSeconds()), s: () => String(d.getSeconds()), a: () => (h < 12 ? 'AM' : 'PM'),
  };
  return String(format || 'yyyy-MM-dd').replace(/'([^']*)'|yyyy|yy|y|MMMM|MMM|MM|M|dd|d|EEEE|EEE|EE|E|HH|H|hh|h|mm|m|ss|s|a/g, (m, lit) => (lit != null ? lit : tok[m]()));
}

function numberFormat(value, { decimals, grouping, currency } = {}) {
  const n = toNum(value);
  if (Number.isNaN(n)) return toText(value);
  const o = { useGrouping: grouping !== false };
  const d = toNum(decimals);
  if (!Number.isNaN(d)) { o.minimumFractionDigits = Math.max(0, Math.min(20, d)); o.maximumFractionDigits = o.minimumFractionDigits; }
  if (currency) { o.style = 'currency'; o.currency = String(currency).toUpperCase(); }
  try { return new Intl.NumberFormat(undefined, o).format(n); } catch { return String(n); }
}

const invalid = (message) => (message ? { valid: false, message } : { valid: false });
const VALID = { valid: true };
// The catalog's functions. Each gets its arguments already resolved (formatString resolves its own placeholders).
const FUNCTIONS = {
  required: (a) => (empty(a.value) ? invalid() : VALID),
  email: (a) => (/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(toText(a.value)) ? VALID : invalid()),
  regex: (a) => { const t = toText(a.value); const re = safeRegex(a.pattern); return re && [...t].length <= 4096 && re.test(t) ? VALID : invalid(); },
  length: (a) => { const n = [...toText(a.value)].length; return (a.min == null || n >= toNum(a.min)) && (a.max == null || n <= toNum(a.max)) ? VALID : invalid(); },
  numeric: (a) => { const n = toNum(a.value); return !Number.isNaN(n) && (a.min == null || n >= toNum(a.min)) && (a.max == null || n <= toNum(a.max)) ? VALID : invalid(); },
  and: (a) => (Array.isArray(a.values) ? a.values : []).every(toBool),
  or: (a) => (Array.isArray(a.values) ? a.values : []).some(toBool),
  not: (a) => !toBool(a.value),
  formatNumber: (a) => numberFormat(a.value, a),
  formatCurrency: (a) => numberFormat(a.value, { ...a, currency: toText(a.currency) || 'USD' }),
  formatDate: (a) => formatDate(a.value, toText(a.format)),
  pluralize: (a) => {
    const n = toNum(a.value);
    let cat = 'other';
    try { cat = new Intl.PluralRules().select(Number.isNaN(n) ? 0 : n); } catch { cat = n === 1 ? 'one' : 'other'; }
    if (n === 0 && a.zero != null) cat = 'zero';
    return toText(a[cat] != null ? a[cat] : a.other);
  },
};
export const FUNCTION_NAMES = [...Object.keys(FUNCTIONS), 'formatString', 'openUrl', '@index'];

export function callFunction(name, args, ctx, depth = 0) {
  if (depth > MAX_DEPTH) return undefined;
  if (name === '@index') { const off = toNum(resolve(isObj(args) ? args.offset : undefined, ctx, depth + 1)); return (ctx.index || 0) + (Number.isNaN(off) ? 0 : off); }
  const a = {};
  for (const [k, v] of Object.entries(isObj(args) ? args : {})) a[k] = resolve(v, ctx, depth + 1);
  if (name === 'formatString') return formatString(toText(a.value), ctx, depth + 1);
  if (name === 'openUrl') return ctx.action ? { openUrl: toText(a.url) } : undefined;   // only from a user gesture
  const f = FUNCTIONS[name];
  return f ? f(a) : undefined;
}

// `${...}` placeholders in a text: a data path (`${/a/b}`, `${name}`) or a function call with named arguments
// (`${formatDate(value:${/when}, format:'MMM d')}`); `\${` is a literal `${`.
export function formatString(text, ctx, depth = 0) {
  if (depth > MAX_DEPTH) return '';
  let out = '';
  for (let i = 0; i < text.length;) {
    if (text[i] === '\\' && text.startsWith('${', i + 1)) { out += '${'; i += 3; continue; }
    if (text.startsWith('${', i)) {
      const end = closing(text, i + 2);
      if (end < 0) { out += text.slice(i); break; }
      out += toText(evalExpr(text.slice(i + 2, end), ctx, depth + 1));
      i = end + 1; continue;
    }
    out += text[i++];
  }
  return out;
}
// the index of the `}` that closes a placeholder opened before `from` (nested placeholders and quotes are skipped)
function closing(s, from) {
  let depth = 0, quote = '';
  for (let i = from; i < s.length; i++) {
    const c = s[i];
    if (quote) { if (c === '\\') i++; else if (c === quote) quote = ''; continue; }
    if (c === '\'' || c === '"') quote = c;
    else if (c === '$' && s[i + 1] === '{') { depth++; i++; } else if (c === '}') { if (!depth) return i; depth--; }
  }
  return -1;
}
function splitTop(s) {
  const out = []; let depth = 0, quote = '', cur = '';
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (quote) { cur += c; if (c === '\\' && i + 1 < s.length) cur += s[++i]; else if (c === quote) quote = ''; continue; }
    if (c === '\'' || c === '"') { quote = c; cur += c; } else if (c === '(' || c === '{') { depth++; cur += c; } else if (c === ')' || c === '}') { depth--; cur += c; } else if (c === ',' && !depth) { out.push(cur); cur = ''; } else cur += c;
  }
  if (cur.trim()) out.push(cur);
  return out;
}
function evalExpr(expr, ctx, depth) {
  const e = expr.trim();
  if (depth > MAX_DEPTH) return '';
  const call = /^(@?[A-Za-z_][\w]*)\s*\(([\s\S]*)\)$/.exec(e);
  if (call) {
    const args = {};
    for (const part of splitTop(call[2])) {
      const c = part.indexOf(':');
      if (c < 0) continue;
      args[part.slice(0, c).trim()] = evalArg(part.slice(c + 1).trim(), ctx, depth + 1);
    }
    return callFunction(call[1], args, ctx, depth + 1);
  }
  return getAt(ctx.data, resolvePath(e, ctx.scope));
}
function evalArg(a, ctx, depth) {
  if (a.startsWith('${') && a.endsWith('}')) return evalExpr(a.slice(2, -1), ctx, depth);
  if ((a[0] === '\'' || a[0] === '"') && a[a.length - 1] === a[0]) return a.slice(1, -1).replace(/\\(.)/g, '$1');
  if (a === 'true') return true;
  if (a === 'false') return false;
  if (a === 'null') return null;
  if (a !== '' && isFinite(+a)) return +a;
  return a[0] === '/' ? getAt(ctx.data, a) : a;
}

// ------------------------------------------------------------------------------------------------- checks
// The checks of a component: {valid, messages}. A check's condition is a binding or a function call; a false one fails
// with its own message, else the check's.
export function runChecks(component, ctx) {
  const list = component && (component['@checks'] || component.checks);
  const messages = [];
  if (!Array.isArray(list)) return { valid: true, messages };
  for (const rule of list) {
    if (!isObj(rule)) continue;
    const r = resolve(rule.condition, ctx);
    if (toBool(r)) continue;
    const msg = (isObj(r) && typeof r.message === 'string' && r.message) || (typeof rule.message === 'string' ? rule.message : '');
    messages.push(msg);
  }
  return { valid: !messages.length, messages: messages.filter(Boolean), failed: messages.length };
}

// ------------------------------------------------------------------------------------------------- back to the model
// The message a pressed button sends (its context resolved in the button's scope), or null when the action is local.
export function actionMessage(surface, component, ctx, now = new Date()) {
  const ev = component && component.action && component.action.event;
  if (!isObj(ev) || typeof ev.name !== 'string') return null;
  const action = { name: ev.name, surfaceId: surface.id, sourceComponentId: component.id, timestamp: now.toISOString() };
  const um = toText(resolve(ev.userMessage, ctx));
  if (um) action.userMessage = um;
  const context = resolve(isObj(ev.context) ? ev.context : {}, ctx);
  action.context = isObj(context) ? JSON.parse(JSON.stringify(context, (k, v) => (v === undefined ? null : v))) : {};
  if (surface.sendDataModel) action.metadata = { extensions: { gaudium_data_model: clone(surface.data) } };
  return { version: A2UI_VERSION, action };
}
// What a local action does: {openUrl} for a link that may be opened, else null.
export function localAction(component, ctx) {
  const fc = component && component.action && component.action.functionCall;
  if (!isCall(fc)) return null;
  const r = resolve(fc, { ...ctx, action: true });
  if (isObj(r) && typeof r.openUrl === 'string' && safeUrl(r.openUrl, ['http:', 'https:', 'mailto:', 'tel:'])) return { openUrl: r.openUrl };
  return null;
}
export const errorMessage = (e) => ({ version: A2UI_VERSION, error: { code: e.code || 'VALIDATION_FAILED', surfaceId: e.surfaceId || '', path: e.path || '', message: e.message || '' } });
// A renderer-to-agent message as the text of a user turn.
export const wrapMessage = (msg) => `${OPEN_TAG}\n${JSON.stringify(msg)}\n${CLOSE_TAG}`;
// The renderer-to-agent messages of a user turn made by this page ([] for an ordinary message).
export function readUserTurn(content) {
  const t = String(content || '').trim();
  if (!t.startsWith(OPEN_TAG)) return [];
  const out = [];
  for (const p of splitContent(t)) {
    if (p.t !== 'a2ui' || !p.closed) return [];
    const r = parseBlock(p.raw);
    if (r.error) return [];
    out.push(...r.messages.filter((m) => isObj(m) && (isObj(m.action) || isObj(m.error))));
  }
  return out;
}
// A short line for a turn that carries an action ("Place order").
export function actionLabel(msg) {
  if (!msg) return '';
  if (msg.error) return 'Reported a problem with the interface';
  const a = msg.action || {};
  if (a.userMessage) return a.userMessage;
  const words = String(a.name || 'action').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_\-.]+/g, ' ').trim().toLowerCase();
  return words ? words[0].toUpperCase() + words.slice(1) : 'Action';
}

// A URL the page may load or open: an absolute URL of one of the allowed schemes.
export function safeUrl(url, schemes = ['http:', 'https:']) {
  const u = String(url || '').trim();
  if (!u) return '';
  if (/^data:/i.test(u)) return schemes.includes('data:') && /^data:(image|audio|video)\/[a-z0-9.+-]+;base64,/i.test(u) && !/^data:image\/svg/i.test(u) ? u : '';
  try { const p = new URL(u); return schemes.includes(p.protocol) ? u : ''; } catch { return ''; }
}

// ------------------------------------------------------------------------------------------------- a conversation's blocks
// The closed interface blocks of a path, in order: [{key, raw}]. extra: the segment that is streaming on `extra.id`.
export function blocksOfPath(path, extra = null) {
  const out = [];
  const add = (id, si, content, live) => {
    if (!hasA2ui(content)) return;
    let bi = 0;
    for (const p of splitContent(content, { live })) if (p.t === 'a2ui') { if (p.closed) out.push({ key: `${id}:${si}:${bi}`, raw: p.raw }); bi++; }
  };
  for (const m of path) {
    if (m.role !== 'assistant') continue;
    const segs = m.segments || [];
    segs.forEach((g, si) => { if (g.kind !== 'compaction') add(m.id, si, g.content, false); });
    if (extra && extra.id === m.id && extra.seg && extra.seg.kind !== 'compaction') add(m.id, segs.length, extra.seg.content, true);
  }
  return out;
}

// Brings `rt` up to the blocks of the shown path. Blocks are applied once, in order; when the path no longer starts with
// what was applied (another branch, an edit, another chat) a fresh runtime replays it. Returns the runtime to use.
export function syncRuntime(rt, blocks) {
  const same = rt && rt.log.length <= blocks.length && rt.log.every((b, i) => b.key === blocks[i].key && b.raw === blocks[i].raw);
  const r = same ? rt : new A2uiRuntime();
  for (let i = r.log.length; i < blocks.length; i++) r.applyBlock(blocks[i].key, blocks[i].raw);
  return r;
}
