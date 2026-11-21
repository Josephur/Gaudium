// Stored-conversation schema and its migrations (IndexedDB "gaudium-ui" and the export files). Pure, unit-tested.
//
//   1  2026-11-07  convs {id, title, created, updated, root, search}, trees {id, messages}
//   2  2026-11-08  compaction (docs/ui.md 10): nodes may have role 'compaction', assistant segments kind 'compaction', a
//                  conversation may carry ctx {key, used} (its last counted context). Tree records carry `schema`.
//                  Migration: a reply that failed on the context limit before v2 ("This conversation (N tokens) is longer
//                  than the context ...") is marked `overflow`, so it offers "Compact and retry" like a new one.
// Nothing is removed or rewritten otherwise: a v1 tree is a valid v2 tree.
export const SCHEMA = 2;

const OLD_OVERFLOW = /is longer than the context/;

export function migrateTree(messages, from = 1) {
  if (!messages || typeof messages !== 'object' || from >= SCHEMA) return messages;
  const out = {};
  for (const [id, n] of Object.entries(messages)) {
    if (n && n.role === 'assistant' && Array.isArray(n.segments) && n.segments.some((g) => g && g.error && !g.error.overflow && g.error.code === 400 && OLD_OVERFLOW.test(g.error.message || ''))) {
      out[id] = { ...n, segments: n.segments.map((g) => (g && g.error && g.error.code === 400 && OLD_OVERFLOW.test(g.error.message || '') ? { ...g, error: { ...g.error, overflow: true, canCompact: true } } : g)) };
    } else out[id] = n;
  }
  return out;
}

// a stored tree record {id, messages, schema?} at the current schema
export function migrateRecord(rec) {
  if (!rec) return rec;
  const from = rec.schema || 1;
  return from >= SCHEMA ? rec : { ...rec, schema: SCHEMA, messages: migrateTree(rec.messages, from) };
}
