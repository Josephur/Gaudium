// Server-sent events and the chat-completion chunk reducer. Pure (no DOM): unit-tested in src/ui/test/.
//
// The wire format is OpenAI-compatible SSE: `data: <json>\n\n` lines ending in `data: [DONE]`, keep-alive
// comments `:\n\n`, an error after the stream started as `data: {"error": ...}`. Gaudium adds `gaudium` (the engine's
// per-request details: method, steps, accept sequence, switch step, prefix reuse) to the final chunk; nothing else changes.

// Incremental SSE parser: feed() text in arbitrary pieces; onEvent({event, data, id}) per complete event.
export function createSSEParser(onEvent) {
  let buf = '';
  let ev = { event: 'message', data: [], id: null };
  const flush = () => {
    if (ev.data.length) onEvent({ event: ev.event, data: ev.data.join('\n'), id: ev.id });
    ev = { event: 'message', data: [], id: ev.id };
  };
  const line = (l) => {
    if (l === '') return flush();
    if (l[0] === ':') return;                       // comment / keep-alive
    const c = l.indexOf(':');
    const field = c < 0 ? l : l.slice(0, c);
    let value = c < 0 ? '' : l.slice(c + 1);
    if (value[0] === ' ') value = value.slice(1);
    if (field === 'data') ev.data.push(value);
    else if (field === 'event') ev.event = value;
    else if (field === 'id') ev.id = value;
  };
  return {
    feed(text) {
      buf += text;
      let i;
      while ((i = buf.search(/\r\n|\r|\n/)) >= 0) {
        const nl = buf[i] === '\r' && buf[i + 1] === '\n' ? 2 : 1;
        line(buf.slice(0, i));
        buf = buf.slice(i + nl);
      }
    },
    end() { if (buf) { line(buf); buf = ''; } flush(); },
  };
}

// A fresh accumulator for one streamed completion (one "segment" of an assistant turn).
export function newSegment() {
  return {
    reasoning: '', content: '', toolCalls: [], finish: null, timings: null, gaudium: null, usage: null,
    error: null, id: null, progress: null, samples: [], firstTokenAt: null, sentAt: null, startedAt: null, done: false,
  };
}

// Apply one parsed SSE `data` payload to the segment. Returns the segment (mutated) and whether text changed.
// `now` = client clock in ms (performance.now()), used for TTFT and the live sample list.
export function applyChunk(seg, data, now = 0) {
  if (data === '[DONE]') { seg.done = true; return { seg, changed: false, done: true }; }
  let j;
  try { j = JSON.parse(data); } catch { return { seg, changed: false, bad: true }; }
  if (j.error) { seg.error = { message: j.error.message || String(j.error), code: j.error.code, type: j.error.type }; return { seg, changed: true, error: true }; }
  let changed = false;
  if (seg.startedAt == null) seg.startedAt = now;   // the first chunk: the engine took the request (after any wait for the card)
  if (j.id && !seg.id) seg.id = j.id;
  if (j.prompt_progress) seg.progress = j.prompt_progress;
  const ch = j.choices && j.choices[0];
  if (ch) {
    const d = ch.delta || ch.message || {};
    if (typeof d.reasoning_content === 'string' && d.reasoning_content) { seg.reasoning += d.reasoning_content; changed = true; }
    if (typeof d.content === 'string' && d.content) { seg.content += d.content; changed = true; }
    if (Array.isArray(d.tool_calls)) {
      for (const tc of d.tool_calls) {
        const idx = typeof tc.index === 'number' ? tc.index : seg.toolCalls.length;
        let t = seg.toolCalls[idx];
        if (!t) t = seg.toolCalls[idx] = { id: '', name: '', arguments: '', status: 'streaming' };
        if (tc.id) t.id = tc.id;
        if (tc.function) {
          if (tc.function.name) t.name += t.name && t.name === tc.function.name ? '' : tc.function.name;
          if (typeof tc.function.arguments === 'string') t.arguments += tc.function.arguments;
          else if (tc.function.arguments && typeof tc.function.arguments === 'object') t.arguments += JSON.stringify(tc.function.arguments);
        }
      }
      changed = true;
    }
    if (ch.finish_reason) seg.finish = ch.finish_reason;
  }
  if (changed && seg.firstTokenAt == null) seg.firstTokenAt = now;
  if (j.timings) {
    seg.timings = j.timings;
    seg.samples.push(sampleOf(j.timings, now));
  }
  if (j.gaudium) seg.gaudium = j.gaudium;
  if (j.usage) seg.usage = j.usage;
  return { seg, changed };
}

export function sampleOf(t, now) {
  return {
    t: now, n: t.predicted_n || 0, ms: t.predicted_ms || 0, launches: t.launches || 0,
    drafts: t.draft_n || 0, acc: t.draft_n_accepted || 0, prompt_n: t.prompt_n || 0, prompt_ms: t.prompt_ms || 0, cache_n: t.cache_n || 0,
  };
}

// Tool-call arguments: the model's JSON string (complete once the stream ends). Returns {value, ok}.
export function parseArgs(s) {
  if (!s || !s.trim()) return { value: {}, ok: true };
  try { return { value: JSON.parse(s), ok: true }; } catch { return { value: s, ok: false }; }
}

// Read a fetch Response body as SSE, calling onData(dataString) per event; resolves when the stream ends.
export async function readSSE(response, onData, signal) {
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  const p = createSSEParser((e) => onData(e.data, e));
  try {
    for (;;) {
      if (signal && signal.aborted) break;
      const { value, done } = await reader.read();
      if (done) break;
      p.feed(dec.decode(value, { stream: true }));
    }
    p.feed(dec.decode());
    p.end();
  } finally {
    try { reader.releaseLock(); } catch { /* already released */ }
  }
}
