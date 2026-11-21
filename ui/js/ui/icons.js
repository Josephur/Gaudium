// Inline icons (24-unit stroke paths), so the bundle needs no icon font.
import { html } from './h.js';

const P = {
  plus: 'M12 5v14M5 12h14',
  search: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4.2-4.2',
  menu: 'M4 7h16M4 12h16M4 17h16',
  sidebar: 'M4 5h16v14H4zM9 5v14',
  gear: 'M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM19.4 13a7.5 7.5 0 0 0 0-2l2-1.6-2-3.4-2.4 1a7.6 7.6 0 0 0-1.7-1L15 3.5h-4l-.4 2.5a7.6 7.6 0 0 0-1.7 1l-2.4-1-2 3.4L6.6 11a7.5 7.5 0 0 0 0 2l-2 1.6 2 3.4 2.4-1c.5.4 1.1.7 1.7 1l.3 2.5h4l.4-2.5c.6-.3 1.2-.6 1.7-1l2.4 1 2-3.4z',
  send: 'M12 19V5M5 12l7-7 7 7',
  stop: 'M7 7h10v10H7z',
  copy: 'M9 9h10v10H9zM5 15V5h10',
  edit: 'M4 20h4L19 9l-4-4L4 16zM14 6l4 4',
  refresh: 'M20 11a8 8 0 0 0-14.6-4.5L4 8M4 4v4h4M4 13a8 8 0 0 0 14.6 4.5L20 16M20 20v-4h-4',
  trash: 'M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13',
  left: 'M15 6l-6 6 6 6', right: 'M9 6l6 6-6 6', chev: 'M9 6l6 6-6 6', down: 'M6 9l6 6 6-6', up: 'M6 15l6-6 6 6',
  clip: 'M8 12.5l6.4-6.4a3 3 0 0 1 4.2 4.2l-8 8a5 5 0 0 1-7-7l7.6-7.6',
  mic: 'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7',
  x: 'M6 6l12 12M18 6L6 18',
  sun: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4',
  moon: 'M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z',
  monitor: 'M3 5h18v11H3zM8 20h8M12 16v4',
  download: 'M12 4v11M7 10l5 5 5-5M5 20h14', upload: 'M12 20V9M7 14l5-5 5 5M5 4h14',
  tool: 'M14.7 6.3a4 4 0 0 0-5.4 5.2L4 16.8V20h3.2l5.3-5.3a4 4 0 0 0 5.2-5.4l-2.6 2.6-2.6-.5-.5-2.6z',
  chip: 'M8 8h8v8H8zM10 3v3M14 3v3M10 18v3M14 18v3M3 10h3M3 14h3M18 10h3M18 14h3',
  keyboard: 'M3 6h18v12H3zM7 10h.01M11 10h.01M15 10h.01M7 14h10',
  link: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1',
  brain: 'M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V5a2 2 0 0 0-3-1zM15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1',
  skip: 'M6 6l8 6-8 6zM17 6v12',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  pin: 'M12 17v4M8 3h8l-1 6 3 3v2H6v-2l3-3z',
  compact: 'M4 12h16M12 3v6M9 6l3 3 3-3M12 21v-6M9 18l3-3 3 3',
  undo: 'M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3',
};

export function Icon({ name, title }) {
  return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden=${title ? undefined : 'true'} role=${title ? 'img' : undefined}>${title ? html`<title>${title}</title>` : null}<path d=${P[name] || ''} /></svg>`;
}

export function Mark() {
  return html`<svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="var(--ink)"/><rect x="6" y="17" width="4" height="9" rx="1.5" fill="var(--df)"/><rect x="12" y="9" width="4" height="17" rx="1.5" fill="var(--df)"/><rect x="18" y="13" width="4" height="13" rx="1.5" fill="var(--mtp)"/><rect x="24" y="6" width="4" height="20" rx="1.5" fill="var(--mtp)"/></svg>`;
}
