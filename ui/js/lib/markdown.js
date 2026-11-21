// Markdown -> sanitized HTML: marked (GFM), DOMPurify, KaTeX for $…$ / $$…$$ / \(…\) / \[…\], highlight.js for code.
// KaTeX and highlight.js load on first need; until then math shows as source and code unhighlighted, and `onReady`
// listeners re-render once they arrive. Output is cached by input text (streaming re-renders the growing message).
import { Marked, Renderer } from '../../vendor/marked/marked.esm.js';
import DOMPurify from '../../vendor/dompurify/purify.es.mjs';

let katex = null, hljs = null, katexLoading = null, hljsLoading = null;
const listeners = new Set();
export const onLibsReady = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
const ready = () => { cache.clear(); for (const fn of listeners) fn(); };

function needKatex() {
  if (katex || katexLoading) return;
  const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = '/ui/vendor/katex/katex.min.css'; document.head.appendChild(css);
  katexLoading = import('../../vendor/katex/katex.mjs').then((m) => { katex = m.default || m; ready(); }).catch(() => {});
}
function needHljs() {
  if (hljs || hljsLoading) return;
  hljsLoading = import('../../vendor/hljs/highlight.min.js').then((m) => { hljs = m.default || m; ready(); }).catch(() => {});
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const marked = new Marked({ gfm: true, breaks: false });
marked.use({
  renderer: {
    code({ text, lang }) {
      const l = (lang || '').trim().split(/\s+/)[0].toLowerCase();
      let body = esc(text);
      let shown = l;
      needHljs();
      if (hljs) {
        try {
          if (l && hljs.getLanguage(l)) body = hljs.highlight(text, { language: l, ignoreIllegals: true }).value;
          else if (!l && text.length < 8000) { const a = hljs.highlightAuto(text); if (a.relevance >= 5) { body = a.value; shown = a.language; } }   // unlabelled fence: guess
        } catch { /* plain */ }
      }
      return `<div class="code"><div class="code-bar"><span class="code-lang">${esc(shown || 'text')}</span><button type="button" class="code-copy" data-copy>Copy</button></div><pre><code class="hljs${l ? ' language-' + esc(l) : ''}">${body}</code></pre></div>`;
    },
    table(token) {   // tables scroll inside their own box on narrow screens
      return `<div class="table-wrap">${Renderer.prototype.table.call(this, token)}</div>`;
    },
  },
});

DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A' && node.getAttribute('href')) { node.setAttribute('target', '_blank'); node.setAttribute('rel', 'noopener noreferrer'); }
});

// math spans outside code -> placeholders (so marked cannot mangle _ and \), restored after parsing
const MATH_RE = /(\$\$[\s\S]+?\$\$)|(\\\[[\s\S]+?\\\])|(\\\([\s\S]+?\\\))|(\$(?=[^\s$])(?:[^$\n\\]|\\.){1,400}?(?<=[^\s$])\$(?!\d))/g;
function protectMath(src, store) {
  // split off fenced code and inline code
  const parts = src.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g);
  return parts.map((p, i) => (i % 2 ? p : p.replace(MATH_RE, (m) => {
    const display = m.startsWith('$$') || m.startsWith('\\[');
    const tex = m.startsWith('$$') ? m.slice(2, -2) : m.startsWith('\\[') || m.startsWith('\\(') ? m.slice(2, -2) : m.slice(1, -1);
    store.push({ tex, display });
    return `GMATH${store.length - 1}X`;
  }))).join('');
}
function renderMath({ tex, display }) {
  needKatex();
  if (!katex) return `<span class="math-src">${esc(display ? `$$${tex}$$` : `$${tex}$`)}</span>`;
  try { return katex.renderToString(tex, { displayMode: display, throwOnError: false, output: 'htmlAndMathml', strict: 'ignore' }); }
  catch { return `<code>${esc(tex)}</code>`; }
}

const cache = new Map();
export function renderMarkdown(src) {
  const key = src || '';
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  const store = [];
  let html = marked.parse(protectMath(key, store));
  if (store.length) html = html.replace(/GMATH(\d+)X/g, (_, i) => renderMath(store[+i]));
  html = DOMPurify.sanitize(html, { USE_PROFILES: { html: true, mathml: true, svg: true }, ADD_ATTR: ['target', 'data-copy'] });
  if (cache.size > 300) cache.delete(cache.keys().next().value);
  cache.set(key, html);
  return html;
}

// plain-text escape for places that show raw text
export const escapeHtml = esc;
