// The conversations rail: new chat, search, the list (grouped by age), rename / delete, import / export, links.
import { html, cx } from './h.js';
import { useState, useRef, useEffect } from 'preact/hooks';
import { Icon, Mark } from './icons.js';

function group(ts, now) {
  const d = (now - ts) / 86400000;
  const today = new Date(now); today.setHours(0, 0, 0, 0);
  if (ts >= today.getTime()) return 'Today';
  if (ts >= today.getTime() - 86400000) return 'Yesterday';
  if (d < 7) return 'Previous 7 days';
  if (d < 30) return 'Previous 30 days';
  return 'Older';
}

function Row({ c, cur, act }) {
  const [edit, setEdit] = useState(false);
  const [name, setName] = useState(c.title);
  const ref = useRef(null);
  useEffect(() => { if (edit && ref.current) { ref.current.focus(); ref.current.select(); } }, [edit]);
  useEffect(() => { if (act.renaming === c.id) { setName(c.title); setEdit(true); act.renamed(); } }, [act.renaming]);
  const done = (ok) => { setEdit(false); if (ok && name.trim() && name !== c.title) act.rename(c.id, name.trim()); else setName(c.title); };
  return html`<div class=${cx('conv', cur && 'cur')} role="listitem" tabindex=${edit ? -1 : 0} aria-current=${cur ? 'true' : null} onClick=${() => !edit && act.open(c.id)} title=${c.title}
    onKeyDown=${(e) => { if (!edit && e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); act.open(c.id); } }}>
    ${edit ? html`<input ref=${ref} value=${name} aria-label="Conversation name" onInput=${(e) => setName(e.target.value)} onBlur=${() => done(true)}
        onKeyDown=${(e) => { if (e.key === 'Enter') done(true); if (e.key === 'Escape') done(false); }} onClick=${(e) => e.stopPropagation()} />`
      : html`<span class="name">${c.title}</span>
        <button class="icon-btn" title="Rename" onClick=${(e) => { e.stopPropagation(); setName(c.title); setEdit(true); }}><${Icon} name="edit" /></button>
        <button class="icon-btn" title="Delete" onClick=${(e) => { e.stopPropagation(); act.del(c.id); }}><${Icon} name="trash" /></button>`}
  </div>`;
}

export function Sidebar({ convs, curId, query, setQuery, act, searchKey }) {
  const sref = useRef(null);
  useEffect(() => { if (searchKey && sref.current) sref.current.focus(); }, [searchKey]);
  const q = query.trim().toLowerCase();
  const list = q ? convs.filter((c) => (c.search || c.title.toLowerCase()).includes(q)) : convs;
  const now = Date.now();
  const groups = [];
  for (const c of list) { const g = group(c.updated, now); let G = groups[groups.length - 1]; if (!G || G.name !== g) groups.push(G = { name: g, items: [] }); G.items.push(c); }
  const imp = useRef(null);
  return html`<aside class="rail" aria-label="Conversations">
    <div class="brand"><${Mark} />gaudium<small>Gaudi 2</small></div>
    <div class="rail-actions"><button class="btn primary" onClick=${act.newChat} title="New chat (Ctrl+Shift+O)"><${Icon} name="plus" />New chat</button></div>
    <div class="search"><${Icon} name="search" /><input ref=${sref} id="conv-search" type="search" placeholder="Search chats" aria-label="Search chats" value=${query} onInput=${(e) => setQuery(e.target.value)} onKeyDown=${(e) => { if (e.key === 'Escape') { setQuery(''); e.target.blur(); } }} /></div>
    <div class="convs" role="list">
      ${!list.length ? html`<div class="empty-list">${q ? `No chat contains “${query}”.` : 'Your chats appear here. They are stored in this browser only.'}</div>` : null}
      ${groups.map((g) => html`<div class="conv-group">${g.name}</div>${g.items.map((c) => html`<${Row} key=${c.id} c=${c} cur=${c.id === curId} act=${act} />`)}`)}
    </div>
    <div class="rail-foot">
      <input ref=${imp} type="file" accept="application/json,.json" hidden id="import-input" onChange=${(e) => { if (e.target.files[0]) act.importFile(e.target.files[0]); e.target.value = ''; }} />
      <button class="link" onClick=${() => imp.current.click()}><${Icon} name="upload" />Import chats</button>
      <button class="link" onClick=${act.exportAll}><${Icon} name="download" />Export all chats</button>
      <button class="link" onClick=${act.openCard}><${Icon} name="chip" />Engine and card</button>
      <button class="link" onClick=${act.openSettings}><${Icon} name="gear" />Settings</button>
    </div>
  </aside>`;
}
