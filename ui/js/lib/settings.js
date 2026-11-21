// UI settings, kept in localStorage (per browser). Empty sampling fields = the server's defaults (generation_config.json).
export const DEFAULTS = {
  apiKey: '',
  system: '',
  theme: 'system',            // system | light | dark
  thinking: 'on',             // on | off
  effort: 'default',          // default | low | medium | xhigh (the template's levels, /props)
  showReasoning: 'stream',    // stream: open while it streams, then folds | always | never
  temperature: '', top_k: '', top_p: '', min_p: '', seed: '', max_tokens: '',
  method: 'default',          // gaudium_method: default (greedy -> engine greedy default, sampled -> sampled default) or a loaded run
  sendReasoning: true,        // send earlier turns' reasoning back (the template re-renders it; prefix reuse matches)
  pdfMode: 'text',            // text | images
  pasteToFile: 2500,          // pasted text longer than this becomes a text attachment (0 = never)
  telemetry: true,            // the live strip under a streaming reply
  builtinTools: true,         // get_datetime
  askTools: true,             // ask before running a tool
  maxToolTurns: 12,
  autoCompact: true,          // compact the conversation before a turn that would pass compactAt % of the context (docs/ui.md 10)
  compactAt: 90,              // percent of the context: the conversation + the draft + the reply's reserve
  custom: '',                 // extra request JSON (merged last)
  mcpServers: [],             // [{id, name, url, headers, proxy, enabled, disabledTools: []}]
  allowedTools: [],           // tools that run without asking ("server/tool")
};

const KEY = 'gaudium-ui.settings.v1';
export function loadSettings() {
  try { const s = JSON.parse(localStorage.getItem(KEY) || '{}'); return { ...DEFAULTS, ...s }; } catch { return { ...DEFAULTS }; }
}
export function saveSettings(s) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage blocked: settings last for this tab */ }
}
export function applyTheme(theme) {
  const r = document.documentElement;
  if (theme === 'light' || theme === 'dark') r.setAttribute('data-theme', theme); else r.removeAttribute('data-theme');
}
export function effectiveDark() {
  const t = document.documentElement.getAttribute('data-theme');
  if (t) return t === 'dark';
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

// the docked live readout shrunk to one line (a per-browser view choice, apart from the request settings)
const DOCK_KEY = 'gaudium-ui.dock.collapsed';
export function loadDockCollapsed() { try { return localStorage.getItem(DOCK_KEY) === '1'; } catch { return false; } }
export function saveDockCollapsed(v) { try { localStorage.setItem(DOCK_KEY, v ? '1' : '0'); } catch { /* storage blocked: lasts for this tab */ } }
