// Follow the bottom of a scrolling box while it grows (a streaming reply, the live reasoning): stay pinned while the reader is
// at the bottom, let go as soon as they scroll up by any amount, and pin again when they come back to the bottom.
// - Up: the reader's intent is read from the input itself (wheel up, a finger dragging the text down, PageUp / ArrowUp / Home,
//   a press on the scrollbar), not only from scroll events. A scroll event can arrive after a render already pinned the box
//   back to the bottom (the stream renders many times a second), which erased the reader's move: a trackpad or a smooth wheel
//   moving a few px per event could not get away (measured headless 2026-11-08: 30 x 8 px wheel steps, snapped back each time).
// - Back down: a move down is measured from the position at the previous scroll event (a layout effect may already have read the
//   new position), and "at the bottom" is judged against the heights of the last 300 ms of renders, so text that landed between
//   the reader's scroll and its event does not leave them unpinned a few lines short of the end.
// Runs in a layout effect, so the box is scrolled before the frame is painted (no grow-then-jump flicker).
import { useRef, useLayoutEffect, useEffect } from 'preact/hooks';

const reduceMotion = () => { try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch { return false; } };
const UP_KEYS = new Set(['PageUp', 'ArrowUp', 'Home']);

// glide: this render ends a stream; the docked readout leaves and the finished one lands under the reply. Instead of jumping
// to the new bottom, the view stays where it is for this frame (the reply's text does not move) and then glides down.
export function useFollow(ref, { active = true, key = '', near = 80, glide = false } = {}) {
  const s = useRef(null);
  if (!s.current) s.current = { pinned: true, top: 0, evTop: 0, sh: 0, hist: [], ch: 0, key, el: null, off: null, glideUntil: 0, touchY: null };
  const f = s.current;
  useLayoutEffect(() => {
    const el = ref.current;
    if (el !== f.el) attach(f, el);
    if (!el) return;
    if (f.key !== key) { f.key = key; f.pinned = true; }   // another chat, or a new turn: start at the bottom
    if (glide && !reduceMotion()) f.glideUntil = performance.now() + 700;
    if (f.pinned) {
      // (a render during the glide, e.g. the list refresh after the save, keeps gliding instead of cutting it short)
      if (active) { if (performance.now() < f.glideUntil) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }); else { el.scrollTop = el.scrollHeight; f.evTop = el.scrollTop; } }
    } else if (el.scrollHeight - el.scrollTop - el.clientHeight <= 1) {
      f.pinned = true;   // the content shrank under a reader who had scrolled up (the reasoning folds) and left them at the bottom
    }
    const now = performance.now();
    f.hist.push([now, el.scrollHeight]); while (f.hist.length > 1 && now - f.hist[0][0] > 300) f.hist.shift();
    f.top = el.scrollTop; f.sh = el.scrollHeight; f.ch = el.clientHeight;
  });
  useEffect(() => () => attach(f, null), []);
  return (e) => {
    const el = e.currentTarget;
    const up = el.scrollTop < f.top - 1 && el.scrollHeight >= f.sh && el.clientHeight <= f.ch;   // not a clamp after a shrink
    const bottom = el.scrollTop + el.clientHeight;
    const down = el.scrollTop > f.evTop + 1;
    const recent = Math.min(el.scrollHeight, ...f.hist.filter(([t]) => performance.now() - t < 300).map((x) => x[1]));   // the height the reader saw
    if (up) f.pinned = false;
    else if (el.scrollHeight - bottom < near || (down && bottom >= recent - near)) f.pinned = true;
    f.top = el.scrollTop; f.evTop = el.scrollTop; f.sh = Math.max(f.sh, el.scrollHeight); f.ch = el.clientHeight;
  };
}

// an element between the event's target and the box that would take this upward scroll itself (the reasoning box, a tool result)
function innerTakesUp(t, box) {
  for (let n = t; n && n !== box; n = n.parentElement) {
    if (n.scrollTop > 0 && n.scrollHeight > n.clientHeight + 1) { const o = getComputedStyle(n).overflowY; if (o === 'auto' || o === 'scroll') return true; }
  }
  return false;
}

function attach(f, el) {
  if (f.off) { f.off(); f.off = null; }
  f.el = el;
  if (!el) return;
  const letGo = () => { if (el.scrollTop > 0) f.pinned = false; };   // (nothing to scroll up to: stay pinned)
  const wheel = (e) => { if (e.deltaY < 0 && !innerTakesUp(e.target, el)) letGo(); };
  const tstart = (e) => { f.touchY = e.touches.length === 1 ? e.touches[0].clientY : null; };
  const tmove = (e) => { if (f.touchY != null && e.touches.length === 1 && e.touches[0].clientY > f.touchY + 2 && !innerTakesUp(e.target, el)) letGo(); };
  const key = (e) => { if (UP_KEYS.has(e.key) && !/^(input|textarea|select)$/i.test(e.target.tagName)) letGo(); };
  const press = (e) => { if (e.target === el && e.offsetX > el.clientWidth) letGo(); };   // a press on the scrollbar
  const o = { passive: true };
  el.addEventListener('wheel', wheel, o); el.addEventListener('touchstart', tstart, o); el.addEventListener('touchmove', tmove, o);
  el.addEventListener('keydown', key); el.addEventListener('pointerdown', press, o);
  f.off = () => { el.removeEventListener('wheel', wheel, o); el.removeEventListener('touchstart', tstart, o); el.removeEventListener('touchmove', tmove, o);
    el.removeEventListener('keydown', key); el.removeEventListener('pointerdown', press, o); };
}
