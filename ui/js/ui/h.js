// Preact + htm: components are plain functions returning html`...` (no build step, no JSX).
import { h, render, Fragment, Component } from 'preact';
import htm from 'htm';
export { h, render, Fragment };
export const html = htm.bind(h);

// a tiny store: one state object, immutable updates, subscribers
export function createStore(init) {
  let s = init;
  const subs = new Set();
  return {
    get: () => s,
    set(patch) { s = typeof patch === 'function' ? { ...s, ...patch(s) } : { ...s, ...patch }; for (const f of subs) f(s); },
    sub(f) { subs.add(f); return () => subs.delete(f); },
  };
}

export const cx = (...a) => a.filter(Boolean).join(' ');

// memo: re-render only when a prop changed (shallow), like preact/compat's memo
export function memo(fn) {
  class Memo extends Component {
    shouldComponentUpdate(next) { for (const k in next) if (next[k] !== this.props[k]) return true; for (const k in this.props) if (!(k in next)) return true; return false; }
    render(p) { return h(fn, p); }
  }
  Memo.displayName = fn.name || 'Memo';
  return Memo;
}
