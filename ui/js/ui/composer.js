// The message box: text, attachments (button, paste, drop), microphone recording (models with audio input), thinking / effort /
// method, send and stop, and the context ring around the send button (docs/ui.md 10).
import { html, cx } from './h.js';
import { useState, useRef, useEffect } from 'preact/hooks';
import { Icon } from './icons.js';
import { Attachment } from './message.js';
import { readAttachment, textAttachment, VIDEO_TYPES } from '../lib/attachments.js';
import { methodInfo, greedyOnly, fmtInt } from '../lib/stats.js';
import { userContent } from '../lib/tree.js';
import { ringLevel, ringFrac, ringLabel, estimateText, estimateImage, estimateVideo } from '../lib/context.js';
import { estimateAudio, fmtDuration, recordUnavailable, startRecording, recordingToClip, AUDIO_MAX_S, SILENT_DB } from '../lib/audio.js';

const FILE_TYPES = 'application/pdf,text/*,.md,.json,.csv,.py,.js,.ts,.c,.cpp,.h,.hpp,.rs,.go,.java,.sh,.yaml,.yml,.toml,.sql,.html,.css,.xml,.log';
const AUDIO_TYPES = 'audio/*,.wav,.mp3,.m4a,.aac,.ogg,.oga,.opus,.flac,.weba';

// The microphone: a button, then (while recording) the elapsed time, a live input meter and stop / discard. Stops by itself at the
// clip limit. A clip that comes out silent or very quiet carries a warning (lib/audio.js levelWarning), shown on it and as a toast.
// On a page where browsers give no microphone (plain HTTP on an address that is not localhost) the button is dimmed: its tooltip,
// and a click, say why.
const METER_FLOOR = -60;   // dBFS at the meter's left end
function Recorder({ onClip, notify }) {
  const [rec, setRec] = useState(null);   // {handle, t0}
  const [tick, setTick] = useState(0);
  const why = recordUnavailable();
  const stopRef = useRef(null);
  const lvl = useRef({ cur: -Infinity, max: -Infinity, shown: -Infinity });   // peak since the last tick, loudest so far
  useEffect(() => {
    if (!rec) return undefined;
    const h = setInterval(() => {
      const L = lvl.current; L.shown = L.cur; L.cur = -Infinity; setTick((n) => n + 1);
      if ((Date.now() - rec.t0) / 1000 >= AUDIO_MAX_S) stopRef.current && stopRef.current();
    }, 100);
    return () => clearInterval(h);
  }, [rec]);
  useEffect(() => () => { if (stopRef.current) stopRef.current(true); }, []);   // leaving the composer: drop a running recording
  const opening = useRef(false);
  const start = async () => {
    if (opening.current) return;   // a second click while the browser asks for the microphone
    opening.current = true;
    lvl.current = { cur: -Infinity, max: -Infinity, shown: -Infinity };
    try {
      const handle = await startRecording({ onLevel: (db) => { const L = lvl.current; if (db > L.cur) L.cur = db; if (db > L.max) L.max = db; } });
      const t0 = Date.now(); setRec({ handle, t0 });
      stopRef.current = async (drop) => {
        stopRef.current = null; setRec(null);
        if (drop) { handle.cancel(); return; }
        const pcm = await handle.stop();
        const full = (Date.now() - t0) / 1000 >= AUDIO_MAX_S;
        let clip;
        try { clip = await recordingToClip(pcm, `recording-${new Date(t0).toTimeString().slice(0, 8).replace(/:/g, '')}.wav`); }
        catch (e) { notify(e.message || String(e)); return; }
        onClip(clip);
        if (clip.warn) notify(clip.warn.text);
        else if (full) notify(`Recording stopped at ${fmtDuration(AUDIO_MAX_S)}, the longest clip this model takes.`);
      };
    } catch (e) { notify(e.message || String(e)); }
    finally { opening.current = false; }
  };
  if (rec) {
    const s = (Date.now() - rec.t0) / 1000, L = lvl.current;
    const frac = Math.max(0, Math.min(1, (L.shown - METER_FLOOR) / -METER_FLOOR));
    const dead = s > 2 && !(L.max >= SILENT_DB);   // two seconds without a sound: say so while it records
    return html`<span class=${cx('rec-chip', dead && 'dead')} role="status" aria-live="off" data-tick=${tick}><span class="rec-dot" aria-hidden="true"></span>${fmtDuration(Math.floor(s))}
      <span class="rec-meter" title=${dead ? 'No sound from the microphone' : 'Input level'} aria-hidden="true"><span style=${`width:${(frac * 100).toFixed(0)}%`}></span></span>
      ${dead ? html`<span class="rec-dead">no sound</span>` : null}
      <button class="icon-btn" title="Stop and attach the recording" aria-label="Stop recording" onClick=${() => stopRef.current && stopRef.current()}><${Icon} name="stop" /></button>
      <button class="icon-btn" title="Discard the recording" aria-label="Discard recording" onClick=${() => stopRef.current && stopRef.current(true)}><${Icon} name="x" /></button></span>`;
  }
  return html`<button class="icon-btn" id="record-btn" aria-disabled=${why ? 'true' : undefined} title=${why || `Record from the microphone (up to ${fmtDuration(AUDIO_MAX_S)})`}
    aria-label=${why ? `Record audio (unavailable: ${why})` : 'Record audio'} onClick=${() => (why ? notify(why) : start())}><${Icon} name="mic" /></button>`;
}

// The ring: an arc over a faint track, clockwise from the top, used / n_ctx; colour by band, the words in the tooltip.
// It is drawn around the 36 px button inside the same 36 px slot (it overhangs into the composer's padding): nothing moves.
const RING_R = 20.5, RING_C = 2 * Math.PI * RING_R;
function CtxRing({ used, nCtx, live, open, onOpen }) {
  const label = ringLabel(used, nCtx);
  const lvl = ringLevel(used, nCtx);
  return html`<button type="button" class=${cx('ctx-ring', lvl, live && 'live')} data-level=${lvl} data-used=${used} data-nctx=${nCtx}
    title=${live ? label : `${label}. Click for details and compaction.`} aria-label=${label} aria-haspopup="dialog" aria-expanded=${!!open} onClick=${onOpen}>
    <svg viewBox="0 0 46 46" aria-hidden="true"><circle class="track" cx="23" cy="23" r=${RING_R} />
      <circle class="arc" cx="23" cy="23" r=${RING_R} stroke-dasharray=${`${RING_C} ${RING_C}`} stroke-dashoffset=${RING_C * (1 - ringFrac(used, nCtx))} transform="rotate(-90 23 23)" /></svg>
  </button>`;
}

function CtxPanel({ ring, used, draft, onCompact, onSettings, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    const down = (e) => { if (ref.current && !ref.current.contains(e.target) && !(e.target.closest && e.target.closest('.ctx-ring'))) onClose(); };
    const key = (e) => { if (e.key === 'Escape') { e.stopPropagation(); onClose(); } };
    document.addEventListener('mousedown', down); window.addEventListener('keydown', key, true);
    return () => { document.removeEventListener('mousedown', down); window.removeEventListener('keydown', key, true); };
  }, []);
  const lvl = ringLevel(used, ring.nCtx);
  return html`<div class="ctx-pop" role="dialog" aria-label="Context use" ref=${ref}>
    <div class="ctx-pop-head"><span class=${`ctx-dot ${lvl}`}></span><b>${ringLabel(used, ring.nCtx)}</b></div>
    <dl class="kv">
      <dt>${ring.live ? 'this request' : 'conversation'}</dt><dd>${fmtInt(ring.used)}${ring.exact || ring.live ? '' : ' (estimate)'}</dd>
      ${ring.live ? null : html`<dt>draft</dt><dd>${fmtInt(draft)}</dd>`}
      <dt>left</dt><dd>${fmtInt(Math.max(0, ring.nCtx - used))}</dd>
    </dl>
    <p class="notice">${ring.auto ? `Compacts by itself before a turn that would pass ${ring.at}% of the context, with room left for the reply.` : 'Automatic compaction is off.'}${' '}
      <button class="linkish" onClick=${onSettings}>Settings</button></p>
    <div class="ctx-pop-foot">
      <button class="btn" disabled=${!ring.canCompact} onClick=${() => { onClose(); onCompact(); }} title=${ring.canCompact ? 'Summarise the conversation so far; later requests send the summary instead (Ctrl+Shift+L, or type /compact)' : ring.live ? 'Wait for the reply to finish' : 'Nothing to compact yet'}><${Icon} name="compact" />Compact conversation</button>
      <span class="notice"><kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>L</kbd> · /compact</span>
    </div>
  </div>`;
}

export function Composer({ settings, defaultMethodName, setSetting, busy, onSend, onStop, methods, efforts, vision, audio, video, limits, toolCount, notify, focusKey, draftKey, ring, countDraft, onCompact, openContextSettings, greedyOnlyList, methodPlan, docked, modelName }) {
  const [text, setText] = useState(() => { try { return sessionStorage.getItem('gaudium-ui.draft.' + draftKey) || ''; } catch { return ''; } });
  const [atts, setAtts] = useState([]);
  const [drag, setDrag] = useState(false);
  const [reading, setReading] = useState(0);
  const [progress, setProgress] = useState(null);   // {name, done, total}: the frames of a video being read
  const ta = useRef(null), file = useRef(null);
  const [draftN, setDraftN] = useState(0);
  const [panel, setPanel] = useState(false);
  const draftSeq = useRef(0);
  // the draft's tokens for the ring: its text (with text files and PDFs as the request sends them) counted by the server once
  // typing has paused for 500 ms, never while a reply streams; images estimated from their size, audio from its length
  useEffect(() => {
    const seq = ++draftSeq.current;
    const t = userContent({ content: text, attachments: atts });
    const str = typeof t === 'string' ? t : (t.find((p) => p.type === 'text') || {}).text || '';
    const imgs = atts.reduce((a, x) => a + (x.kind === 'image' ? estimateImage(x.width, x.height) : x.kind === 'video' ? estimateVideo(x) : x.kind === 'audio' ? estimateAudio(x.duration) : x.kind === 'pdf' && x.pages ? x.pages.length * 1500 : 0), 0);
    if (/^\/compact\b/.test(text.trim()) || (!str.trim() && !imgs)) { setDraftN(0); return undefined; }
    if (busy) return undefined;
    const h = setTimeout(async () => {
      let n = str.trim() && countDraft ? await countDraft(str) : 0;
      if (n == null) n = estimateText(str);
      if (seq === draftSeq.current) setDraftN(n + imgs);
    }, 500);
    return () => clearTimeout(h);
  }, [text, atts, busy]);
  const rng = ring || { nCtx: 0, used: 0 };
  const used = rng.live ? rng.used : rng.used + draftN;
  // the draft grows up to 40 % of the window; while the full readout is docked above it, up to 20 % (a 40-line draft at 1280x800
  // left the chat about 80 px). It shrinks when the dock appears and grows back on the next edit, not when the reply ends.
  const size = () => { const e = ta.current; if (e) { e.style.height = 'auto'; e.style.height = Math.min(e.scrollHeight, window.innerHeight * (docked ? 0.2 : 0.4)) + 'px'; } };
  useEffect(() => { if (ta.current) ta.current.focus(); }, [focusKey]);
  useEffect(() => { try { const d = sessionStorage.getItem('gaudium-ui.draft.' + draftKey); setText(d || ''); } catch { setText(''); } setAtts([]); }, [draftKey]);
  useEffect(size, [text]);
  useEffect(() => { if (docked) size(); }, [docked]);
  const saveDraft = (t) => { try { sessionStorage.setItem('gaudium-ui.draft.' + draftKey, t); } catch { /* no storage */ } };
  const add = async (files) => {
    for (const f of files) {
      setReading((n) => n + 1);
      try {
        const a = await readAttachment(f, { pdfMode: settings.pdfMode, vision, audio, video, limits,
          videoFps: Number(settings.videoFps) || 2, videoMaxFrames: Number(settings.videoMaxFrames) || 64, videoMaxSide: Number(settings.videoMaxSide) || 0,
          onProgress: (done, total) => setProgress({ name: f.name, done, total }) });
        setAtts((x) => [...x, a]); if (a.warn) notify(`${a.name}: ${a.warn.text}`);
      }
      catch (e) { notify(e.message || String(e)); }
      finally { setReading((n) => n - 1); setProgress(null); }
    }
  };
  const send = () => {
    if (busy || reading) return;
    if (!text.trim() && !atts.length) return;
    onSend(text, atts); setText(''); setAtts([]); saveDraft('');
  };
  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
    else if (e.key === 'Escape' && busy) { e.preventDefault(); onStop(); }
  };
  const onPaste = (e) => {
    const items = [...(e.clipboardData ? e.clipboardData.files : [])];
    if (items.length) { e.preventDefault(); add(items); return; }
    const t = e.clipboardData && e.clipboardData.getData('text/plain');
    if (t && settings.pasteToFile > 0 && t.length > settings.pasteToFile) {
      e.preventDefault(); setAtts((x) => [...x, textAttachment(`pasted-${x.length + 1}.txt`, t)]); notify(`Long paste (${t.length.toLocaleString()} characters) attached as a text file.`);
    }
  };
  const thinkOn = settings.thinking !== 'off';
  return html`<div class="composer-wrap">
    <div class=${cx('composer', drag && 'drag')}
      onDragOver=${(e) => { e.preventDefault(); setDrag(true); }} onDragLeave=${() => setDrag(false)}
      onDrop=${(e) => { e.preventDefault(); setDrag(false); add([...e.dataTransfer.files]); }}>
      ${atts.length || reading ? html`<div class="atts">${atts.map((a, i) => html`<${Attachment} a=${a} onRemove=${() => setAtts(atts.filter((_, j) => j !== i))} />`)}${reading ? html`<span class="notice" role="status">${progress ? `reading ${progress.name}: frame ${progress.done} of ${progress.total}…` : `reading ${reading} file${reading > 1 ? 's' : ''}…`}</span>` : null}</div>` : null}
      <textarea ref=${ta} id="composer-input" rows="1" placeholder=${`Message ${modelName || 'the model'} on the Gaudi 2`} aria-label="Message" value=${text}
        onInput=${(e) => { setText(e.target.value); saveDraft(e.target.value); }} onKeyDown=${onKey} onPaste=${onPaste}></textarea>
      <div class="composer-row">
        <input ref=${file} type="file" multiple hidden id="file-input" accept=${[vision && 'image/*', vision && video && VIDEO_TYPES, audio && AUDIO_TYPES, FILE_TYPES].filter(Boolean).join(',')}
          onChange=${(e) => { add([...e.target.files]); e.target.value = ''; }} />
        <button class="icon-btn" title=${`Attach ${[vision && 'images', vision && video && 'videos or animations', audio && 'audio clips', 'text files or PDFs'].filter(Boolean).join(', ')}`} onClick=${() => file.current.click()}><${Icon} name="clip" /></button>
        ${audio ? html`<${Recorder} onClip=${(a) => setAtts((x) => [...x, a])} notify=${notify} />` : null}
        <button class=${cx('chip', thinkOn && 'on')} title="Thinking: the model reasons before it answers" aria-pressed=${thinkOn} onClick=${() => setSetting('thinking', thinkOn ? 'off' : 'on')}><${Icon} name="brain" />${thinkOn ? 'Thinking' : 'No thinking'}</button>
        ${thinkOn ? html`<label class="select-chip" title="Reasoning effort"><span class="sr">Reasoning effort</span>
          <select value=${settings.effort} onChange=${(e) => setSetting('effort', e.target.value)}>
            <option value="default">effort: default</option>${efforts.map((l) => html`<option value=${l}>effort: ${l}</option>`)}
          </select></label>` : null}
        ${methods.length > 1 ? html`<label class="select-chip" title="Speculative method (gaudium_method)"><span class="sr">Speculative method</span>
          <select class="method-pick" value=${settings.method} onChange=${(e) => setSetting('method', e.target.value)}>
            <option value="default">${defaultMethodName ? `method: ${methodInfo(defaultMethodName).label} (default)` : 'method: engine default'}</option>
            ${methods.map((m) => html`<option value=${m}>${methodInfo(m).label}${greedyOnly(m, greedyOnlyList) ? ' · greedy' : ''}</option>`)}
          </select></label>`
          : methods.length === 1 || defaultMethodName ? html`<span class="badge" title="The only speculative method this model's engine runs">${methodInfo(methods[0] || defaultMethodName).label}</span>` : null}
        ${methodPlan ? html`<span class="greedy-chip" title=${methodPlan.kind === 'greedy'
          ? `${methodInfo(methodPlan.method).label} is greedy-only: these requests run it at temperature 0 (greedy). Set a temperature to sample (then on ${methodInfo(methodPlan.used || 'adaptive').label}), or pick MTP adaptive.`
          : `${methodInfo(methodPlan.method).label} is greedy-only and the temperature is set to ${settings.temperature}: these requests sample with ${methodInfo(methodPlan.used).label} instead. Clear the temperature (or set 0) to run ${methodInfo(methodPlan.method).label}.`}>${methodPlan.kind === 'greedy' ? 'greedy' : `sampled: ${methodInfo(methodPlan.used).label}`}</span>` : null}
        ${toolCount ? html`<span class="badge" title="Tools offered to the model"><${Icon} name="tool" />${toolCount}</span>` : null}
        <span class="grow"></span>
        <span class="send-slot">
          ${rng.nCtx ? html`<${CtxRing} used=${used} nCtx=${rng.nCtx} live=${rng.live} open=${panel} onOpen=${() => setPanel(!panel)} />` : null}
          ${busy ? html`<button class="send" title="Stop (Esc)" onClick=${onStop}><${Icon} name="stop" /></button>`
            : html`<button class="send" title="Send (Enter)" disabled=${(!text.trim() && !atts.length) || reading > 0} onClick=${send}><${Icon} name="send" /></button>`}
          ${panel && rng.nCtx ? html`<${CtxPanel} ring=${rng} used=${used} draft=${draftN} onCompact=${onCompact} onSettings=${() => { setPanel(false); openContextSettings(); }} onClose=${() => setPanel(false)} />` : null}
        </span>
      </div>
    </div>
    <div class="hint"><kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> new line · <kbd>Esc</kbd> stop · <kbd>/compact</kbd> summarise · <kbd>Ctrl</kbd>+<kbd>/</kbd> shortcuts</div>
  </div>`;
}
