// Audio clips for a model with audio input (/props modalities.audio). The server takes WAV only, so every clip is converted
// here: the browser decodes whatever it can play (mp3, m4a, ogg, opus, flac, webm, wav; decodeAudioData), the channels are
// averaged to mono, the clip is resampled to 16 kHz (what the model's processor reads; a WAV at 16 kHz gives the reference's
// features exactly, docs/server.md "Audio") and written as 16-bit PCM WAV. Microphone recordings are captured as raw samples
// (pcm-capture.worklet.js) and take the same path from a WAV at the device's rate. Every clip gets its levels and, when it is
// silent or very quiet, a warning. The pure parts (mono, resample, WAV, levels, limits, estimates) are unit-tested in node.

export const AUDIO_RATE = 16000;
// the server's longest clip: 7500 soft tokens of 40 ms (five minutes; the engine's --audio-max-tokens). It cuts a longer clip
// there without saying so, so the UI refuses one instead (and a recording stops by itself at the limit).
export const AUDIO_MAX_S = 300;
const TOKEN_SAMPLES = 640;   // one soft token per 40 ms (640 samples at 16 kHz)

const AUDIO_EXT = /\.(wav|wave|mp3|m4a|aac|ogg|oga|opus|flac|webm|weba|mka|aif|aiff|caf|amr|3gp)$/i;
export function isAudioFile(file) {
  if (file.type && file.type.startsWith('audio/')) return true;
  return (!file.type || file.type === 'application/octet-stream') && AUDIO_EXT.test(file.name || '');
}

// m:ss (m:ss.t under a minute), e.g. 0:03.5, 1:07
export function fmtDuration(s) {
  if (!(s >= 0)) return '';
  if (s < 60) { const t = Math.round(s * 10) / 10; return `0:${t < 10 ? '0' : ''}${t.toFixed(1)}`; }
  const r = Math.round(s), m = Math.floor(r / 60), ss = r % 60;
  return `${m}:${ss < 10 ? '0' : ''}${ss}`;
}

// the clip's prompt tokens: the begin marker, one soft token per 40 ms, the end marker
export function estimateAudio(seconds) {
  if (!(seconds > 0)) return 2;
  return Math.ceil(Math.round(Math.min(seconds, AUDIO_MAX_S) * AUDIO_RATE) / TOKEN_SAMPLES) + 2;
}
// the same from a stored part (base64 of a 16-bit mono WAV at 16 kHz with a 44-byte header, as this file writes them)
export function audioSecondsOfBase64(b64) {
  const n = String(b64 || '').replace(/^data:[^,]*,/, '');
  const bytes = Math.floor(n.length * 3 / 4) - (n.endsWith('==') ? 2 : n.endsWith('=') ? 1 : 0);
  return Math.max(0, bytes - 44) / 2 / AUDIO_RATE;
}

export function checkDuration(seconds, max = AUDIO_MAX_S) {
  if (seconds > max + 1e-6) return `The clip is ${fmtDuration(seconds)} long; this model takes at most ${fmtDuration(max)} a clip. Cut it shorter and attach it again.`;
  if (!(seconds > 0)) return 'The clip is empty.';
  return null;
}

// channels (Float32Arrays of one length) -> one channel, their average (what the server does with a multi-channel WAV)
export function mixMono(channels) {
  if (channels.length === 1) return channels[0];
  const n = channels[0].length, out = new Float32Array(n), k = channels.length;
  for (const c of channels) for (let i = 0; i < n; i++) out[i] += c[i];
  for (let i = 0; i < n; i++) out[i] /= k;
  return out;
}

// windowed-sinc resampling (Blackman window, 16 zero crossings a side, the cut-off at the lower rate's Nyquist), the kernel read
// from a table: only used when the browser decodes at a rate of its own (one that refuses a 16 kHz context)
export function resample(x, from, to) {
  if (from === to) return x;
  const ratio = to / from, n = Math.max(1, Math.round(x.length * ratio)), out = new Float32Array(n);
  const fc = Math.min(1, ratio), half = 16 / fc, RES = 64, tab = new Float32Array(Math.ceil(half * RES) + 2);
  for (let i = 0; i < tab.length; i++) {
    const d = i / RES; if (d > half) break;
    const w = 0.42 + 0.5 * Math.cos(Math.PI * d / half) + 0.08 * Math.cos(2 * Math.PI * d / half);
    tab[i] = w * (d === 0 ? 1 : Math.sin(Math.PI * fc * d) / (Math.PI * fc * d));
  }
  for (let i = 0; i < n; i++) {
    const t = i / ratio;
    let acc = 0, wsum = 0;
    for (let j = Math.ceil(t - half), e = Math.floor(t + half); j <= e; j++) {
      const k = tab[Math.round(Math.abs(t - j) * RES)];
      wsum += k; if (j >= 0 && j < x.length) acc += x[j] * k;
    }
    out[i] = wsum ? acc / wsum : 0;
  }
  return out;
}

// mono float samples -> a 16-bit PCM WAV file
export function encodeWav(samples, rate = AUDIO_RATE) {
  const n = samples.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, Math.round(s < 0 ? s * 32768 : s * 32767), true); }
  return buf;
}

export function base64OfBuffer(buf) {
  const b = new Uint8Array(buf); let s = '';
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(s);
}

// bytes of any audio file the browser can play -> {samples (mono Float32 at 16 kHz), seconds}
export async function decodeToMono16k(arrayBuffer) {
  const AC = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
  if (!AC) throw new Error('This browser has no Web Audio decoder, so it cannot convert audio files.');
  let ctx;
  try { ctx = new AC(1, 1, AUDIO_RATE); } catch { ctx = new AC(1, 1, 44100); }   // a browser that refuses 16 kHz: resample below
  let ab;
  try { ab = await ctx.decodeAudioData(arrayBuffer.slice(0)); }
  catch { throw new Error('the browser cannot decode this audio file (try WAV, MP3, M4A, OGG or FLAC)'); }
  const ch = []; for (let c = 0; c < ab.numberOfChannels; c++) ch.push(ab.getChannelData(c));
  const mono = resample(mixMono(ch), ab.sampleRate, AUDIO_RATE);
  return { samples: mono, seconds: mono.length / AUDIO_RATE };
}

// Levels of a mono clip in dBFS: its peak, its RMS, and the RMS of its loudest 100 ms (what says whether anyone spoke: a clip of
// speech has its loudest 100 ms within about 20 dB of full scale, a muted or far-away microphone does not)
export function clipLevels(x, rate = AUDIO_RATE) {
  const db = (v) => (v > 0 ? 20 * Math.log10(v) : -Infinity);
  let peak = 0, sum = 0;
  for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; sum += x[i] * x[i]; }
  const w = Math.max(1, Math.round(rate / 10));
  let loud = 0;
  for (let i = 0; i < x.length; i += w) {
    const end = Math.min(x.length, i + w);
    if (end - i < w / 2 && i > 0) break;   // a short tail is not a window of its own
    let e = 0; for (let j = i; j < end; j++) e += x[j] * x[j];
    loud = Math.max(loud, e / (end - i));
  }
  return { peak: db(peak), rms: db(x.length ? Math.sqrt(sum / x.length) : 0), loud: db(Math.sqrt(loud)) };
}

// The model does not say "I heard nothing": a silent or near-silent clip gets an invented answer. So the UI warns, by the loudest
// 100 ms: below SILENT_DB there is nothing but the noise floor of a muted or disconnected input; below QUIET_DB the speech is
// about 30 dB under a normal recording (speech recorded at a normal level has its loudest 100 ms near -15 to -5 dBFS)
export const SILENT_DB = -60, QUIET_DB = -45;
const fmtDb = (v) => (v === -Infinity ? '-∞' : Math.round(v).toString()).replace('-', '−');
export function levelWarning(lv, liftDb = 0) {
  if (!lv) return null;
  if (!(lv.loud >= SILENT_DB)) return { kind: 'silent', text: `Silent (loudest ${fmtDb(lv.loud)} dBFS): the microphone gave no sound. The model will not say it heard nothing; it may invent an answer.` };
  if (lv.loud < QUIET_DB) return { kind: 'quiet', text: `Very quiet (loudest ${fmtDb(lv.loud)} dBFS${liftDb > 0 ? `, raised by ${Math.round(liftDb)} dB` : ''}): the model may not hear the words and invent an answer. Record closer to the microphone or louder.` };
  return null;
}

// A quiet clip that is not silent is raised before it is sent: the peak to -1 dBFS, by at most 40 dB, when the loudest 100 ms
// is below -20 dBFS (normal speech has it near -15 to -5). Measured on the live Gemma 4 12B (2026-11-19, T = 0, "Transcribe this
// audio clip."): the 11 s reference speech clip at -30 dB was answered with only the speech's attribution; raised by 31 dB, with
// the words and the attribution. A silent clip is not raised (that would only lift the noise floor). Returns the gain in dB.
export const LIFT_BELOW_DB = -20, LIFT_PEAK_DB = -1, LIFT_MAX_DB = 40;
export function liftGain(lv) {
  if (!lv || !(lv.loud >= SILENT_DB) || lv.loud >= LIFT_BELOW_DB || !(lv.peak > -Infinity)) return 0;
  return Math.max(0, Math.min(LIFT_MAX_DB, LIFT_PEAK_DB - lv.peak));
}

// a File or Blob -> the attachment the conversation keeps: {kind 'audio', name, mime 'audio/wav', size, dataUrl, duration,
// levels, warn}; trim: cut a clip that is too long at the limit (a recording) instead of refusing it (a file)
export async function readAudio(file, { max = AUDIO_MAX_S, name, trim = false } = {}) {
  const label = name || file.name || 'audio';
  let dec;
  try { dec = await decodeToMono16k(await file.arrayBuffer()); }
  catch (e) { throw new Error(`${label}: ${e.message || e}`); }
  if (trim && dec.seconds > max) dec = { samples: dec.samples.subarray(0, Math.floor(max * AUDIO_RATE)), seconds: Math.floor(max * AUDIO_RATE) / AUDIO_RATE };
  const why = checkDuration(dec.seconds, max);
  if (why) throw new Error(`${label}: ${why}`);
  const levels = clipLevels(dec.samples), lift = liftGain(levels), warn = levelWarning(levels, lift);
  let samples = dec.samples;
  if (lift > 0) { const g = 10 ** (lift / 20); samples = Float32Array.from(samples, (v) => v * g); }
  const wav = encodeWav(samples, AUDIO_RATE);
  return { kind: 'audio', name: label, mime: 'audio/wav', size: wav.byteLength, duration: dec.seconds, levels, ...(lift > 0 ? { liftDb: lift } : {}), ...(warn ? { warn } : {}),
    dataUrl: 'data:audio/wav;base64,' + base64OfBuffer(wav), note: `${fmtDuration(dec.seconds)} · 16 kHz mono${lift > 0 ? ` · raised ${Math.round(lift)} dB` : ''}` };
}

// Microphone recording needs a secure context: browsers hide getUserMedia on a plain-HTTP page that is not localhost.
// null = recording works here; else the reason, in words for someone who is not a web developer
export function recordUnavailable(g = globalThis) {
  if (!g.isSecureContext) return 'Browsers only let a web page use the microphone when it is opened over HTTPS (or as localhost on this computer), and this page is plain HTTP: open the chat at an https:// address to record, or attach an audio file instead.';
  if (!g.navigator || !g.navigator.mediaDevices || !g.navigator.mediaDevices.getUserMedia) return 'This browser does not offer the microphone to web pages. Attach an audio file instead.';
  if (!(g.AudioContext || g.webkitAudioContext)) return 'This browser has no Web Audio, so it cannot record here. Attach an audio file instead.';
  return null;
}

// Recording: the microphone's PCM through an AudioWorklet (a ScriptProcessor where there is none), at the device's rate (44.1 or
// 48 kHz). No MediaRecorder: its Opus or AAC would be decoded again by the browser, and that is where browsers differ (a WebM
// from MediaRecorder has no duration; fragmented MP4 is decoded differently); raw samples are the same in every browser.
// Echo cancellation and noise suppression are off (nothing plays while recording, and both can eat quiet or the first words of
// speech); automatic gain control is on (it lifts a quiet voice to a normal level).
export const MIC_CONSTRAINTS = { channelCount: { ideal: 1 }, echoCancellation: false, noiseSuppression: false, autoGainControl: true };

// start recording; returns {rate, stop(): Promise<{samples, rate}>, cancel()}; onLevel(peak dBFS of the last block) while it runs.
// Throws with a readable message when the microphone is refused.
export async function startRecording({ onLevel } = {}) {
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS }); }
  catch (e) {
    throw new Error(e && e.name === 'NotAllowedError' ? 'Microphone access was refused (the browser or the system blocks it).'
      : e && e.name === 'NotFoundError' ? 'No microphone was found.' : `The microphone could not be opened: ${e && e.message || e}`);
  }
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ctx = new AC();
  const chunks = [];
  const take = (b) => {
    chunks.push(b);
    if (onLevel) { let p = 0; for (let i = 0; i < b.length; i++) { const a = Math.abs(b[i]); if (a > p) p = a; } onLevel(p > 0 ? 20 * Math.log10(p) : -Infinity); }
  };
  let node, flushed = null;
  try {
    if (ctx.state === 'suspended') await ctx.resume();
    const src = ctx.createMediaStreamSource(stream);
    if (ctx.audioWorklet && globalThis.AudioWorkletNode) {
      await ctx.audioWorklet.addModule(new URL('./pcm-capture.worklet.js', import.meta.url));
      node = new AudioWorkletNode(ctx, 'gaudium-pcm-capture');
      flushed = new Promise((resolve) => { node.port.onmessage = (e) => (e.data === 'done' ? resolve() : take(e.data)); });
    } else {
      node = ctx.createScriptProcessor(4096, 2, 1);
      node.onaudioprocess = (e) => {
        const ib = e.inputBuffer, k = ib.numberOfChannels, out = new Float32Array(ib.length);
        for (let c = 0; c < k; c++) { const d = ib.getChannelData(c); for (let i = 0; i < out.length; i++) out[i] += d[i] / k; }
        take(out);
      };
    }
    const mute = ctx.createGain(); mute.gain.value = 0;   // pulls the graph without playing the microphone back
    src.connect(node); node.connect(mute); mute.connect(ctx.destination);
  } catch (e) { stream.getTracks().forEach((t) => t.stop()); ctx.close(); throw new Error(`The microphone could not be recorded: ${e && e.message || e}`); }
  const end = async () => {
    if (flushed) { node.port.postMessage('flush'); await Promise.race([flushed, new Promise((r) => setTimeout(r, 1000))]); }
    try { node.disconnect(); } catch { /* gone */ }
    stream.getTracks().forEach((t) => t.stop()); await ctx.close();
  };
  return {
    rate: ctx.sampleRate,
    stop: async () => {
      await end();
      const n = chunks.reduce((a, c) => a + c.length, 0), samples = new Float32Array(n);
      let o = 0; for (const c of chunks) { samples.set(c, o); o += c.length; }
      return { samples, rate: ctx.sampleRate };
    },
    cancel: () => { chunks.length = 0; end(); },
  };
}

// a finished recording -> its attachment: written as a WAV at the device's rate, then through the same path as a file (the
// browser's own resampler to 16 kHz, levels, warning)
export function recordingToClip({ samples, rate }, name) {
  const blob = new Blob([encodeWav(samples, rate)], { type: 'audio/wav' });
  return readAudio(blob, { name, trim: true });
}
