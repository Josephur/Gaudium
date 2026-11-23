// Context use: the ring around the send button, the projection that triggers automatic compaction, and the fallback
// estimates. Pure (no DOM), unit-tested. docs/ui.md 10.
import { fmtInt, contextUsed } from './stats.js';
import { estimateAudio, audioSecondsOfBase64 } from './audio.js';
import { imageTokens, videoTokens, visionLimits } from './media.js';

// The ring's colour: muted green up to 80 % inclusive, yellow above 80 % up to 90 % inclusive, red above 90 %.
// Integer maths, so exactly 80 % and exactly 90 % land on the lower band whatever the context size.
export function ringLevel(used, nCtx) {
  if (!(nCtx > 0)) return 'ok';
  const u = Math.max(0, used || 0);
  if (u * 10 <= nCtx * 8) return 'ok';
  if (u * 10 <= nCtx * 9) return 'warn';
  return 'bad';
}

// Percent in words: rounded up, so the number agrees with the colour (80 % is green, 80.03 % reads 81 % and is yellow) and
// errs on the side of less room left
export const ringPct = (used, nCtx) => (nCtx > 0 ? Math.ceil((100 * Math.max(0, used || 0)) / nCtx) : 0);

// The arc's fill, 0..1 (an overflowing request shows a full ring; the words say by how much)
export const ringFrac = (used, nCtx) => (nCtx > 0 ? Math.min(1, Math.max(0, (used || 0) / nCtx)) : 0);

// The words for the tooltip and the aria-label (colour is never the only signal)
export const ringLabel = (used, nCtx) => `Context ${fmtInt(used || 0)} of ${fmtInt(nCtx)} tokens (${ringPct(used, nCtx)}%)`;

// Room kept for the reply when deciding whether the next turn fits: the max-tokens setting, else 4,096 tokens
// (a quarter of a small context at most, so a test server with a tiny --ctx-size still has room for a prompt).
export function reserveTokens(settings, nCtx) {
  const mt = Number(settings && settings.max_tokens);
  if (mt > 0 && isFinite(mt)) return Math.min(Math.floor(mt), nCtx || mt);
  return Math.min(4096, Math.max(16, Math.floor((nCtx || 0) / 4)));
}

// The automatic-compaction threshold, percent of the context (setting compactAt; clamped to 50..99)
export function compactPercent(settings) {
  const p = Number(settings && settings.compactAt);
  return isFinite(p) && p > 0 ? Math.min(99, Math.max(50, Math.round(p))) : 90;
}

// The next turn's projection: the conversation as it would be sent + the draft + the reply's reserve, against the threshold.
// over = compact before sending. fits = the prompt itself still fits the context (the server's own rule: prompt + 1 <= n_ctx).
export function projection({ history = 0, draft = 0, reserve = 0, nCtx = 0, percent = 90 }) {
  const prompt = history + draft;
  const projected = prompt + reserve;
  const limit = Math.floor((nCtx * percent) / 100);
  return { prompt, projected, limit, over: nCtx > 0 && projected > limit, fits: nCtx > 0 && prompt + 1 <= nCtx };
}

// The summary's own token budget: about a twelfth of the context, 256..6,144 tokens
export const summaryMaxTokens = (nCtx) => Math.max(256, Math.min(6144, Math.floor((nCtx || 0) / 12)));

// Live context of a streaming segment: the server's timings (prompt + reused + generated), else the prefill's total
export function usedLive(seg) {
  if (!seg) return null;
  if (seg.timings) return contextUsed(seg.timings);
  if (seg.progress && seg.progress.total > 0) return seg.progress.total;
  return null;
}

// --- estimates, only when the server cannot count (an older gaudium-server without /gaudium/count_tokens) ----------------
// The model's tokenizer gives 3.4 characters per token on English prose and on code (measured with /tokenize on docs/ui.md:
// 27,410 chars -> 8,007 tokens, and on js/app.js: 33,549 -> 9,846); an image is about one token per 32x32 pixels after
// the processor's resize (patch 16, merge 2).
// Text outside ASCII (CJK above all) is counted one token per character, so the estimate errs high there, not low.
export function estimateText(text) {
  const s = String(text || '');
  let wide = 0;
  for (let i = 0; i < s.length; i++) if (s.charCodeAt(i) > 0x7f) wide++;
  return Math.ceil((s.length - wide) / 3.4) + wide;
}
// The server's vision tower caps an image (/props `vision`: the largest image in pixels, the image tokens of a request);
// the page tells this module once it has the props.
let LIMITS = visionLimits(null);
export function setVisionLimits(props) { LIMITS = visionLimits(props); }
export function estimateImage(width, height) { return imageTokens(width, height, LIMITS); }
// a video attachment ({frames: count or list, fps, duration, width, height}) or a video content part
export function estimateVideo(a) {
  const frames = Array.isArray(a.frames) ? a.frames.length : a.frames || 1;
  return videoTokens({ frames, fps: a.fps, durationMs: a.duration > 0 ? a.duration * 1000 : 0, width: a.width, height: a.height }, LIMITS).tokens;
}
// a video part of a request: its frame list with their rate; an animation's frames are not known from the part (taken as
// a short clip at the model's fewest frames)
function estimateVideoPart(p) {
  const v = p.video || p.input_video || p.video_url || {};
  const frames = Array.isArray(v) ? v : v.frames;
  return frames ? estimateVideo({ frames, fps: v.fps }) : estimateVideo({ frames: 8, fps: 2 });
}
const isVideoPart = (p) => p.type === 'video' || p.type === 'input_video' || p.type === 'video_url';
export function estimateMessages(msgs, tools) {
  let n = 8 + (tools && tools.length ? estimateText(JSON.stringify(tools)) + 300 : 0);
  for (const m of msgs || []) {
    n += 6;
    if (typeof m.content === 'string') n += estimateText(m.content);
    else if (Array.isArray(m.content)) for (const p of m.content) n += p.type === 'text' ? estimateText(p.text) : p.type === 'input_audio' ? estimateAudio(audioSecondsOfBase64(p.input_audio && p.input_audio.data)) : isVideoPart(p) ? estimateVideoPart(p) : estimateImage(p.width, p.height);
    if (m.reasoning_content) n += estimateText(m.reasoning_content) + 4;
    if (m.tool_calls) n += estimateText(JSON.stringify(m.tool_calls)) + 10;
  }
  return n;
}

// --- shortening the summarisation request when the conversation is too long to summarise in one request ---------------------
const textLen = (m) => (typeof m.content === 'string' ? m.content.length : Array.isArray(m.content) ? m.content.reduce((a, p) => a + (p.type === 'text' ? String(p.text || '').length : 0), 0) : 0);
// the index of the message with the longest text, the last `keep` messages (the instruction and the opening) excluded; the
// system message (it may hold an earlier summary) only when nothing else is left. -1 = nothing long enough to shorten.
export function longestText(msgs, keep = 2) {
  let best = -1, len = 400;
  for (let i = 0; i < msgs.length - keep; i++) { if (msgs[i].role === 'system') continue; const l = textLen(msgs[i]); if (l > len) { best = i; len = l; } }
  if (best < 0 && msgs[0] && msgs[0].role === 'system' && textLen(msgs[0]) > 400 && msgs.length > keep) best = 0;
  return best;
}
// keep `ratio` of a message's text: its start and its end, with a note of what was left out in between
export function elideText(t, ratio) {
  const s = String(t || '');
  const keep = Math.max(200, Math.floor(s.length * ratio));
  if (keep >= s.length) return s;
  const head = Math.floor(keep * 0.6), tail = keep - head;
  return `${s.slice(0, head)}\n\n[… ${fmtInt(s.length - keep)} characters of this message left out so the conversation fits the summary request …]\n\n${s.slice(s.length - tail)}`;
}
export function elideMessage(m, ratio) {
  if (typeof m.content === 'string') return { ...m, content: elideText(m.content, ratio) };
  if (Array.isArray(m.content)) return { ...m, content: m.content.map((p) => (p.type === 'text' ? { ...p, text: elideText(p.text, ratio) } : p)) };
  return m;
}
