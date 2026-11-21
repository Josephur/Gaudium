// Conversations in the browser (IndexedDB "gaudium-ui"): "convs" holds the list (title, times, search text), "trees" the
// message tree of each. Falls back to memory when IndexedDB is unavailable (private windows): the UI still works.
// Schema versions and their migrations: lib/migrate.js (the database version is the schema version).
import { searchText, uid } from './tree.js';
import { SCHEMA, migrateTree, migrateRecord } from './migrate.js';

const DB = 'gaudium-ui', VER = SCHEMA;
let dbp = null;
const mem = { convs: new Map(), trees: new Map() };

function open() {
  if (dbp) return dbp;
  dbp = new Promise((resolve) => {
    try {
      const r = indexedDB.open(DB, VER);
      r.onupgradeneeded = (e) => {
        const d = r.result;
        if (!d.objectStoreNames.contains('convs')) d.createObjectStore('convs', { keyPath: 'id' });
        if (!d.objectStoreNames.contains('trees')) d.createObjectStore('trees', { keyPath: 'id' });
        if (e.oldVersion >= 1 && e.oldVersion < SCHEMA) {   // migrate every stored tree in the upgrade transaction (all or nothing)
          const cur = r.transaction.objectStore('trees').openCursor();
          cur.onsuccess = () => { const c = cur.result; if (!c) return; c.update(migrateRecord(c.value)); c.continue(); };
        }
      };
      r.onsuccess = () => { const d = r.result; d.onversionchange = () => { d.close(); dbp = null; }; resolve(d); };   // a newer tab upgrades: let it
      r.onerror = () => resolve(null);
      // another tab still holds the old version open: wait for it (it closes on reload); after 8 s work from memory
      r.onblocked = () => { setTimeout(() => resolve(null), 8000); };
    } catch { resolve(null); }
  });
  return dbp;
}

function tx(d, stores, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = d.transaction(stores, mode);
    const out = fn(t);
    t.oncomplete = () => resolve(out && out.result !== undefined ? out.result : out);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export async function listConversations() {
  const d = await open();
  if (!d) return [...mem.convs.values()].sort((a, b) => b.updated - a.updated);
  const all = await tx(d, ['convs'], 'readonly', (t) => t.objectStore('convs').getAll());
  return (all || []).sort((a, b) => b.updated - a.updated);
}

export async function loadTree(id) {
  const d = await open();
  if (!d) return mem.trees.get(id) || null;
  const r = await tx(d, ['trees'], 'readonly', (t) => t.objectStore('trees').get(id));
  return r ? migrateRecord(r).messages : null;
}

export async function saveConversation(conv, messages) {
  const c = { ...conv, search: searchText(conv, messages) };
  const d = await open();
  if (!d) { mem.convs.set(c.id, c); mem.trees.set(c.id, messages); return c; }
  await tx(d, ['convs', 'trees'], 'readwrite', (t) => { t.objectStore('convs').put(c); t.objectStore('trees').put({ id: c.id, schema: SCHEMA, messages }); });
  return c;
}

export async function deleteConversation(id) {
  const d = await open();
  if (!d) { mem.convs.delete(id); mem.trees.delete(id); return; }
  await tx(d, ['convs', 'trees'], 'readwrite', (t) => { t.objectStore('convs').delete(id); t.objectStore('trees').delete(id); });
}

export async function deleteAll() {
  const d = await open();
  if (!d) { mem.convs.clear(); mem.trees.clear(); return; }
  await tx(d, ['convs', 'trees'], 'readwrite', (t) => { t.objectStore('convs').clear(); t.objectStore('trees').clear(); });
}

// ------------------------------------------------------------------------------------------------- export / import
export async function exportConversations(ids) {
  const convs = (await listConversations()).filter((c) => !ids || ids.includes(c.id));
  const out = [];
  for (const c of convs) { const { search, ...conv } = c; out.push({ conv, messages: await loadTree(c.id) }); }
  return { format: 'gaudium-ui', version: SCHEMA, exported: new Date().toISOString(), conversations: out };
}

// Accepts our export and legacy web UI exports (one {conv, messages} or an array of them): every conversation gets a
// new id, so importing twice never overwrites.
export async function importConversations(data) {
  const items = data && data.format === 'gaudium-ui' ? data.conversations : Array.isArray(data) ? data : data && data.conv ? [data] : null;
  if (!items) throw new Error('This file is not a conversation export.');
  let n = 0;
  for (const it of items) {
    const r = data.format === 'gaudium-ui' ? fromOurs(it, data.version || 1) : fromLegacyExport(it);
    if (!r) continue;
    await saveConversation(r.conv, r.messages); n++;
  }
  return n;
}

function fromOurs(it, version) {
  if (!it || !it.conv || !it.messages) return null;
  const { ctx, ...conv } = it.conv;   // ctx is keyed by the old id: counted again
  return { conv: { ...conv, id: uid('c') }, messages: migrateTree(it.messages, version) };
}

// Legacy UI export (DatabaseConversation + DatabaseMessage): {conv:{id,name,currNode,lastModified}, messages:[{id,convId,type,role,
// content,reasoningContent|thinking,parent,children,timestamp,extra:[{type,name,base64Url|content}],toolCalls,timings}]}
function fromLegacyExport(it) {
  if (!it || !it.conv || !Array.isArray(it.messages)) return null;
  const msgs = {};
  let root = null;
  for (const m of it.messages) {
    const role = m.type === 'root' || m.role === 'root' ? 'root' : m.role;
    const node = { id: m.id, parent: m.parent || null, children: [...(m.children || [])], active: null, role, created: m.timestamp || Date.now() };
    if (role === 'root') root = node.id;
    else if (role === 'user' || role === 'system') {
      node.role = 'user'; node.content = m.content || '';
      node.attachments = (m.extra || []).map((x) => x.type === 'imageFile' || x.type === 'IMAGE' ? { kind: 'image', name: x.name || 'image', dataUrl: x.base64Url } :
        { kind: 'text', name: x.name || 'file', text: x.content || '' }).filter((a) => a.dataUrl || a.text != null);
    } else if (role === 'assistant') {
      let calls = [];
      try { calls = typeof m.toolCalls === 'string' ? JSON.parse(m.toolCalls) : (m.toolCalls || []); } catch { calls = []; }
      node.segments = [{ content: m.content || '', reasoning: m.reasoningContent || m.thinking || '', toolCalls: calls.map((c) => ({ id: c.id, name: c.function && c.function.name, arguments: c.function && c.function.arguments, status: 'done' })), timings: m.timings || null, finish: 'stop' }];
    } else if (role === 'tool') {
      node.role = 'tool'; node.content = m.content || ''; node.toolCallId = m.toolCallId;
    }
    msgs[node.id] = node;
  }
  // fold separate tool messages into the preceding assistant segment's call results, and the following assistant
  // message into the same turn (our turns are one node with segments)
  for (const n of Object.values(msgs)) {
    if (n.role !== 'tool') continue;
    const p = msgs[n.parent];
    if (p && p.role === 'assistant') {
      const seg = p.segments[p.segments.length - 1];
      const c = seg.toolCalls.find((x) => x.id === n.toolCallId) || seg.toolCalls.find((x) => x.result == null);
      if (c) c.result = n.content;
      for (const ch of n.children) { const a = msgs[ch]; if (a && a.role === 'assistant') { p.segments.push(...a.segments); p.children = a.children.slice(); for (const g of a.children) if (msgs[g]) msgs[g].parent = p.id; delete msgs[ch]; } }
      p.children = p.children.filter((x) => x !== n.id);
      delete msgs[n.id];
    }
  }
  if (!root) {   // no root in the export: make one above the parentless nodes
    root = uid('r'); msgs[root] = { id: root, parent: null, children: [], active: null, role: 'root', created: 0 };
    for (const n of Object.values(msgs)) if (n.id !== root && !n.parent) { n.parent = root; msgs[root].children.push(n.id); }
  }
  for (const n of Object.values(msgs)) n.children = n.children.filter((c) => msgs[c]);
  // mark the exported current branch active
  let cur = it.conv.currNode && msgs[it.conv.currNode] ? msgs[it.conv.currNode] : null;
  while (cur && cur.parent && msgs[cur.parent]) { msgs[cur.parent].active = cur.id; cur = msgs[cur.parent]; }
  const now = Date.now();
  return { conv: { id: uid('c'), title: it.conv.name || 'Imported chat', created: it.conv.lastModified || now, updated: it.conv.lastModified || now, root }, messages: msgs };
}
