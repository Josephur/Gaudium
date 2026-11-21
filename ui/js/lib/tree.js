// The conversation tree: every edit or regenerate adds a sibling branch; `active` on a node picks the child shown.
// Pure (no DOM), unit-tested. A conversation is {id, title, root, ...}; messages are a map id -> node:
//   {id, parent, children: [ids], active: childId|null, role: 'root'|'user'|'assistant', created, ...payload}

export function uid(prefix = 'm') {
  const r = (globalThis.crypto && crypto.getRandomValues) ? Array.from(crypto.getRandomValues(new Uint8Array(8)), (b) => b.toString(16).padStart(2, '0')).join('') : Math.random().toString(16).slice(2);
  return `${prefix}_${Date.now().toString(36)}${r}`;
}

export function newConversation(now = Date.now()) {
  const root = { id: uid('r'), parent: null, children: [], active: null, role: 'root', created: now };
  const conv = { id: uid('c'), title: 'New chat', created: now, updated: now, root: root.id, search: '' };
  return { conv, messages: { [root.id]: root } };
}

// add `node` under `parentId` and make it the active branch there (returns a new messages map)
export function addChild(messages, parentId, node) {
  const p = messages[parentId];
  if (!p) throw new Error(`no parent ${parentId}`);
  const n = { children: [], active: null, ...node, parent: parentId };
  return { ...messages, [parentId]: { ...p, children: [...p.children, n.id], active: n.id }, [n.id]: n };
}

export function update(messages, id, patch) {
  const m = messages[id];
  if (!m) return messages;
  return { ...messages, [id]: { ...m, ...(typeof patch === 'function' ? patch(m) : patch) } };
}

// the shown path from the root: following `active` (else the newest child) down to a leaf
export function activePath(messages, rootId) {
  const out = [];
  let n = messages[rootId];
  const seen = new Set();
  while (n && !seen.has(n.id)) {
    seen.add(n.id);
    if (n.role !== 'root') out.push(n);
    const next = n.active && messages[n.active] ? n.active : n.children[n.children.length - 1];
    n = next ? messages[next] : null;
  }
  return out;
}

// the path from the root to `id` (inclusive), root excluded
export function pathTo(messages, id) {
  const out = [];
  let n = messages[id];
  const seen = new Set();
  while (n && !seen.has(n.id)) { seen.add(n.id); if (n.role !== 'root') out.unshift(n); n = n.parent ? messages[n.parent] : null; }
  return out;
}

export function siblings(messages, id) {
  const n = messages[id];
  if (!n || !n.parent) return { index: 0, count: 1, ids: [id] };
  const ids = messages[n.parent].children;
  return { index: Math.max(0, ids.indexOf(id)), count: ids.length, ids };
}

// show sibling `delta` steps away from `id` (the branch navigator); makes every node on its path active
export function switchSibling(messages, id, delta) {
  const s = siblings(messages, id);
  const j = s.index + delta;
  if (j < 0 || j >= s.count) return messages;
  const n = messages[id];
  return update(messages, n.parent, { active: s.ids[j] });
}

// delete a node and its subtree; the parent shows its previous child (or none)
export function removeSubtree(messages, id) {
  const n = messages[id];
  if (!n || !n.parent) return messages;
  const out = { ...messages };
  const stack = [id];
  while (stack.length) { const k = stack.pop(); const m = out[k]; if (!m) continue; stack.push(...m.children); delete out[k]; }
  const p = out[n.parent];
  const ch = p.children.filter((c) => c !== id);
  out[n.parent] = { ...p, children: ch, active: p.active === id ? (ch[ch.length - 1] || null) : p.active };
  return out;
}

// text for the conversation search (titles + message text, capped)
export function searchText(conv, messages) {
  let s = conv.title || '';
  for (const m of Object.values(messages)) {
    if (m.role === 'user') s += '\n' + (m.content || '');
    else if (m.role === 'assistant') for (const g of m.segments || []) s += '\n' + (g.content || '');
    if (s.length > 40000) break;
  }
  return s.slice(0, 40000).toLowerCase();
}

export function titleFrom(text) {
  const t = String(text || '').replace(/\s+/g, ' ').trim();
  if (!t) return 'New chat';
  return t.length > 64 ? t.slice(0, 61).replace(/\s+\S*$/, '') + '…' : t;
}

// ------------------------------------------------------------------------------------- compaction (docs/ui.md 10)
// A compaction is a marker on the path whose summary replaces everything before it in the requests (the messages stay in
// the tree, shown collapsed). Two shapes:
//   - a node {role: 'compaction', segments: [summary segment], compacted: N} between two turns (manual, before a send,
//     "Compact and retry" on a first pass); its children are the turns that follow;
//   - a segment {kind: 'compaction', ...} inside an assistant turn, between two passes of a tool loop; the requests after it
//     send the summary, that turn's user message again (the model needs a query) and the passes after the summary.
// A summary counts only when it is usable (finished, no error, text past the prefill); a stopped or failed one is ignored.
export const SUMMARY_PREFILL = '## Goal';
export const isCompaction = (n) => !!n && n.role === 'compaction';
export const usableSummary = (g) => !!g && !g.error && (g.finish === 'stop' || g.finish === 'length') && String(g.content || '').trim().length > SUMMARY_PREFILL.length;
export const nodeSummary = (n) => (isCompaction(n) ? (n.segments || []).find(usableSummary) || null : null);

// the last usable compaction on a path: {at: path index, seg: segment index (in-turn) | null, summary, compacted} | null
export function lastCut(path) {
  for (let i = path.length - 1; i >= 0; i--) {
    const n = path[i];
    if (isCompaction(n)) { const g = nodeSummary(n); if (g) return { at: i, seg: null, summary: g.content, compacted: n.compacted || 0, node: n }; }
    else if (n.role === 'assistant') {
      const segs = n.segments || [];
      for (let j = segs.length - 1; j >= 0; j--) if (segs[j].kind === 'compaction' && usableSummary(segs[j])) return { at: i, seg: j, summary: segs[j].content, compacted: segs[j].compacted || 0, node: n };
    }
  }
  return null;
}
// the shown path's divider: the index of the last usable compaction node (in-turn summaries do not hide messages), else -1
export function displayCut(path) {
  for (let i = path.length - 1; i >= 0; i--) if (isCompaction(path[i]) && nodeSummary(path[i])) return i;
  return -1;
}
const isTurn = (n) => n.role === 'user' || n.role === 'assistant';
// user and assistant messages on a path (what "N earlier messages" counts)
export const countMessages = (nodes) => nodes.filter(isTurn).length;
const hasOutput = (g) => !g.error && g.kind !== 'compaction' && (String(g.content || '').trim() || (g.toolCalls || []).length);
// How much a compaction of `path` would replace: the replies with output since the last summary. 0 = nothing to compact.
export function compactableCount(path) {
  const cut = lastCut(path);
  let n = 0;
  const from = cut ? (cut.seg == null ? cut.at + 1 : cut.at) : 0;
  for (let i = from; i < path.length; i++) {
    const m = path[i];
    if (m.role !== 'assistant') continue;
    const segs = m.segments || [];
    const start = cut && cut.seg != null && i === cut.at ? cut.seg + 1 : 0;
    if (segs.slice(start).some(hasOutput)) n++;
  }
  return n;
}

// The text the model gets in place of the compacted messages: appended to the system message (renders cleanly with this
// template whatever follows; a fabricated user/assistant pair would put words in the model's mouth). docs/ui.md 10.
export function summarySystem(system, summary, compacted) {
  const note = `The conversation so far was compacted to save context. The summary below replaces the ${compacted ? `${compacted} earlier messages` : 'earlier messages'}: `
    + 'treat it as what was said and done, and continue the conversation from it. The user can still read the earlier messages; you only have this summary.\n\n'
    + `<conversation_summary>\n${String(summary || '').trim()}\n</conversation_summary>`;
  return system && system.trim() ? `${system}\n\n${note}` : note;
}

// The summarisation request's last two messages: the instruction, and the assistant's opening (continued with
// continue_final_message, which also starts the answer past an empty think block: thinking off without changing the system
// block, so the engine reuses the whole conversation as a cached prefix).
export function compactionInstruction(focus = '') {
  return [
    'Compact this conversation: write a faithful, structured summary of everything above, so that we can continue from the summary alone.',
    'It replaces the conversation. Anything you leave out is lost, so keep everything needed to continue the work.',
    'Use these sections, in this order, and skip a section only when it has nothing:',
    '## Goal: what the user wants, in their words where it matters.',
    '## Decisions and constraints: what was agreed, chosen or ruled out, and why.',
    '## Facts and results: names, numbers, findings, answers already given.',
    '## Code and files: file paths, functions, commands, settings, and the code the work depends on (verbatim when short).',
    '## Tool work: the tools called, what they returned, and what is still pending.',
    '## Open questions and next steps: what remains, and the user\'s latest request.',
    'Prefer exact names, values and short quotes to paraphrase. Do not add anything that was not in the conversation. Write only the summary.',
    focus && focus.trim() ? `Pay particular attention to: ${focus.trim()}` : '',
  ].filter(Boolean).join('\n');
}
export function compactionMessages(path, opts = {}) {
  const msgs = toApiMessages(path, opts);
  msgs.push({ role: 'user', content: compactionInstruction(opts.focus) });
  msgs.push({ role: 'assistant', content: SUMMARY_PREFILL });
  return msgs;
}

// put `node` between `id` and its parent (it takes id's place among the parent's children; id becomes its only child)
export function insertAbove(messages, id, node) {
  const n = messages[id];
  if (!n || !n.parent) throw new Error(`cannot insert above ${id}`);
  const p = messages[n.parent];
  const c = { ...node, children: [id], active: id, parent: p.id };
  return { ...messages, [p.id]: { ...p, children: p.children.map((x) => (x === id ? c.id : x)), active: p.active === id ? c.id : p.active }, [c.id]: c, [id]: { ...n, parent: c.id } };
}
// remove a node but keep what hangs below it: its children take its place under its parent (undo a compaction)
export function spliceOut(messages, id) {
  const n = messages[id];
  if (!n || !n.parent) return messages;
  const p = messages[n.parent];
  const out = { ...messages };
  delete out[id];
  for (const c of n.children) if (out[c]) out[c] = { ...out[c], parent: p.id };
  const children = p.children.flatMap((x) => (x === id ? n.children : [x]));
  const active = p.active === id ? (n.active && n.children.includes(n.active) ? n.active : n.children[n.children.length - 1] || null) : p.active;
  out[p.id] = { ...p, children, active };
  return out;
}

// ------------------------------------------------------------------------------------- OpenAI messages for the request
// path: the nodes from the root to the last user node (inclusive) or to an assistant node being continued.
// An assistant turn is a list of segments; a segment that ended in tool calls becomes an assistant message with tool_calls
// followed by one tool message per call, so a tool loop resends exactly what happened.
// After a compaction only its summary (in the system message) and what follows it are sent.
export function toApiMessages(path, { system = '', sendReasoning = true } = {}) {
  const out = [];
  const cut = lastCut(path);
  const sys = cut ? summarySystem(system, cut.summary, cut.compacted) : system;
  if (sys && sys.trim()) out.push({ role: 'system', content: sys });
  let rest = path;
  if (cut && cut.seg == null) rest = path.slice(cut.at + 1);
  else if (cut) {   // in-turn: that turn's user message again, then the passes after the summary
    let u = cut.at - 1; while (u >= 0 && path[u].role !== 'user') u--;
    rest = [...(u >= 0 ? [path[u]] : []), { ...path[cut.at], segments: (path[cut.at].segments || []).slice(cut.seg + 1) }, ...path.slice(cut.at + 1)];
  }
  for (const m of rest) {
    if (m.role === 'user') out.push({ role: 'user', content: userContent(m) });
    else if (m.role === 'assistant') {
      for (const g of m.segments || []) {
        if (g.kind === 'compaction') continue;
        if (g.error && !g.content && !(g.toolCalls || []).length) continue;
        const a = { role: 'assistant', content: g.content || '' };
        if (sendReasoning && g.reasoning) a.reasoning_content = g.reasoning;
        const calls = (g.toolCalls || []).filter((c) => c.name);
        if (calls.length) a.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments || '{}' } }));
        if (!a.content && !a.tool_calls && !a.reasoning_content) continue;
        out.push(a);
        for (const c of calls) out.push({ role: 'tool', tool_call_id: c.id, content: toolResultText(c) });
      }
    }
  }
  return out;
}

export function toolResultText(c) {
  if (c.result == null) return c.error ? `Error: ${c.error}` : (c.status === 'denied' ? 'The user declined to run this tool.' : '');
  return typeof c.result === 'string' ? c.result : JSON.stringify(c.result);
}

// user message content: plain text, or OpenAI parts when there are images or audio clips (images and PDF pages first, then the
// clips as input_audio parts carrying their WAV, then the text, if any); text files and PDFs go in as text blocks
export function userContent(m) {
  const atts = m.attachments || [];
  let text = m.content || '';
  const files = atts.filter((a) => a.kind === 'text' || (a.kind === 'pdf' && a.text != null));
  if (files.length) {
    const blocks = files.map((a) => `File: ${a.name}\n\`\`\`\n${a.text}\n\`\`\``).join('\n\n');
    text = text ? `${blocks}\n\n${text}` : blocks;
  }
  const imgs = atts.filter((a) => a.kind === 'image' && a.dataUrl).concat(...atts.filter((a) => a.kind === 'pdf' && a.pages).map((a) => a.pages.map((p) => ({ dataUrl: p }))));
  const clips = atts.filter((a) => a.kind === 'audio' && a.dataUrl);
  if (!imgs.length && !clips.length) return text;
  return [...imgs.map((a) => ({ type: 'image_url', image_url: { url: a.dataUrl } })),
    ...clips.map((a) => ({ type: 'input_audio', input_audio: { data: a.dataUrl.slice(a.dataUrl.indexOf(',') + 1), format: 'wav' } })),
    ...(text ? [{ type: 'text', text }] : [])];   // a message of only media is sent as only its media: no empty text part
}
