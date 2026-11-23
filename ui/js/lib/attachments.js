// Files the user attaches: images (sent to the vision tower), animated images and video files (sent as a video: models
// with video input), text files (sent as text), PDFs (text via pdf.js, or page images when they have no text layer or the
// setting says so), audio clips (models with audio input; converted to a 16 kHz mono WAV by lib/audio.js).
// The server decodes PNG, JPEG, GIF, WebP and BMP itself, so those go as the file the user picked (the model sees the
// pixels a reference pipeline would see); other image types (AVIF, HEIC, SVG, ...) are converted to PNG here first.
// An animated GIF / WebP goes as the file too, as a video part: the server takes its frames and their times.
// A video file is not decoded by the server: this page draws its frames from a <video> element (the browser's decoder) and
// sends them as JPEG images with their rate.
import { isAudioFile, readAudio } from './audio.js';
import { animationInfo, planFrames, frameSize, fmtDuration } from './media.js';
const SERVER_IMAGE = ['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/bmp'];
const VIDEO_EXT = /\.(mp4|m4v|mov|webm|mkv|ogv|avi|mpe?g|3gp)$/i;
export const VIDEO_TYPES = 'video/*,.mp4,.m4v,.mov,.webm,.mkv,.ogv';
export const isVideoFile = (file) => (file.type || '').startsWith('video/') || (!file.type && VIDEO_EXT.test(file.name));
const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|jsonl|ya?ml|toml|ini|cfg|conf|log|xml|html?|css|scss|js|mjs|cjs|ts|tsx|jsx|py|rb|go|rs|c|cc|cpp|cxx|h|hpp|hh|java|kt|swift|m|mm|cs|php|pl|lua|r|jl|sh|bash|zsh|fish|ps1|sql|tex|bib|rst|adoc|org|diff|patch|cmake|mk|makefile|dockerfile|gradle|proto|vue|svelte|asm|s)$/i;

const readAs = (file, how) => new Promise((resolve, reject) => {
  const r = new FileReader();
  r.onload = () => resolve(r.result); r.onerror = () => reject(r.error);
  if (how === 'text') r.readAsText(file); else if (how === 'buffer') r.readAsArrayBuffer(file); else r.readAsDataURL(file);
});

async function toPng(dataUrl) {
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = () => rej(new Error('the browser cannot decode this image')); img.src = dataUrl; });
  const c = document.createElement('canvas'); c.width = img.naturalWidth || 512; c.height = img.naturalHeight || 512;
  c.getContext('2d').drawImage(img, 0, 0);
  return { dataUrl: c.toDataURL('image/png'), width: c.width, height: c.height };
}

// width x height (the context ring's estimate of an image's tokens); null when the browser cannot decode it
async function imageSize(dataUrl) {
  try { const img = new Image(); img.src = dataUrl; await img.decode(); return { width: img.naturalWidth, height: img.naturalHeight }; } catch { return null; }
}

const once = (el, ok, bad, ms, what) => new Promise((resolve, reject) => {
  let timer = null;
  const done = (f, v) => { clearTimeout(timer); el.removeEventListener(ok, onOk); el.removeEventListener(bad, onBad); f(v); };
  const onOk = () => done(resolve);
  const onBad = () => done(reject, new Error(what));
  timer = setTimeout(() => done(reject, new Error(what)), ms);
  el.addEventListener(ok, onOk); el.addEventListener(bad, onBad);
});

// A video file as frames: `fps` pictures per second of video time (at most `maxFrames`, a longer video more thinly), each
// drawn at the video's size with its longer side held to `maxSide`, as JPEG. The browser must be able to play the file.
// onProgress(done, total) is called as the frames are drawn.
export async function readVideo(file, { fps = 2, maxFrames = 64, maxSide = 768, onProgress = null } = {}) {
  const url = URL.createObjectURL(file);
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto'; v.crossOrigin = 'anonymous';
  const cannot = `${file.name}: this browser cannot play the video (its container or codec), so its frames cannot be read. Convert it to MP4 (H.264) or WebM, or attach an animated GIF.`;
  try {
    v.src = url;
    if (v.readyState < 1) await once(v, 'loadedmetadata', 'error', 20000, cannot);
    if (!(v.videoWidth > 0 && v.videoHeight > 0)) throw new Error(`${file.name}: the file has no video track.`);
    let duration = v.duration;
    if (!Number.isFinite(duration)) {
      // a stream without a length in its header: seeking far past the end makes the browser find the real end
      v.currentTime = 1e9; await once(v, 'seeked', 'error', 20000, cannot);
      duration = v.duration; v.currentTime = 0;
      if (!Number.isFinite(duration)) throw new Error(`${file.name}: the video's length cannot be read.`);
    }
    const plan = planFrames(duration, { fps, maxFrames });
    const size = frameSize(v.videoWidth, v.videoHeight, maxSide);
    const c = document.createElement('canvas'); c.width = size.width; c.height = size.height;
    const g = c.getContext('2d');
    const frames = [];
    for (let i = 0; i < plan.times.length; i++) {
      const t = plan.times[i];
      if (Math.abs(v.currentTime - t) > 1e-4 || v.readyState < 2) { v.currentTime = t; await once(v, 'seeked', 'error', 20000, cannot); }
      g.drawImage(v, 0, 0, size.width, size.height);
      frames.push(c.toDataURL('image/jpeg', 0.9));
      if (onProgress) onProgress(i + 1, plan.times.length);
    }
    const thin = plan.fps < fps - 1e-9;
    return {
      kind: 'video', name: file.name, mime: file.type || 'video/*', size: file.size,
      frames, fps: plan.fps, duration, width: size.width, height: size.height, srcWidth: v.videoWidth, srcHeight: v.videoHeight,
      note: `video · ${fmtDuration(duration)} · ${frames.length} frames${thin ? ` (one every ${(1 / plan.fps).toFixed(1)} s)` : ''}`,
    };
  } finally { v.removeAttribute('src'); v.load(); URL.revokeObjectURL(url); }
}

export function isProbablyText(file) {
  if (file.type.startsWith('text/')) return true;
  if (['application/json', 'application/xml', 'application/javascript', 'application/x-sh', 'application/sql', 'application/toml', 'application/yaml'].includes(file.type)) return true;
  return TEXT_EXT.test(file.name) || /^(makefile|dockerfile|readme|license|changelog)$/i.test(file.name);
}

let pdfjs = null;
async function loadPdf() {
  if (pdfjs) return pdfjs;
  pdfjs = await import('../../vendor/pdfjs/pdf.min.mjs');
  pdfjs.GlobalWorkerOptions.workerSrc = '/ui/vendor/pdfjs/pdf.worker.min.mjs';
  return pdfjs;
}

async function readPdf(file, mode, vision = true) {
  if (!vision) mode = 'text';   // page images need a vision tower
  const lib = await loadPdf();
  const doc = await lib.getDocument({ data: new Uint8Array(await readAs(file, 'buffer')), isEvalSupported: false }).promise;
  const pages = [];
  let text = '';
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const tc = await page.getTextContent();
    const t = tc.items.map((i) => i.str + (i.hasEOL ? '\n' : '')).join('');
    text += (p > 1 ? '\n\n' : '') + t;
  }
  const att = { kind: 'pdf', name: file.name, mime: file.type, size: file.size, numPages: doc.numPages };
  if (!vision && !text.trim()) throw new Error('This PDF has no text layer (a scan), and the loaded model has no vision to read page images.');
  if (mode === 'images' || !text.trim()) {
    for (let p = 1; p <= Math.min(doc.numPages, 8); p++) {
      const page = await doc.getPage(p);
      const vp = page.getViewport({ scale: 1.5 });
      const c = document.createElement('canvas'); c.width = vp.width; c.height = vp.height;
      await page.render({ canvasContext: c.getContext('2d'), viewport: vp }).promise;
      pages.push(c.toDataURL('image/png'));
    }
    att.pages = pages; att.note = text.trim() ? `${pages.length} page images` : `no text layer: ${pages.length} page images`;
  } else { att.text = text; att.note = `${doc.numPages} pages as text`; }
  return att;
}

export async function readAttachment(file, { pdfMode = 'text', vision = true, audio = false, video = false, videoFps = 2, videoMaxFrames = 64, videoMaxSide = 768, limits = null, onProgress = null } = {}) {
  if (file.type === 'application/pdf' || /\.pdf$/i.test(file.name)) return readPdf(file, pdfMode, vision);
  if (isVideoFile(file)) {
    if (!vision) throw new Error('The loaded model has no vision, so it cannot watch videos.');
    if (!video) throw new Error(`${file.name}: the loaded model takes images but no video.`);
    return readVideo(file, { fps: videoFps, maxFrames: videoMaxFrames, maxSide: videoMaxSide, onProgress });
  }
  if (isAudioFile(file)) {
    if (!audio) throw new Error(`${file.name}: the loaded model has no audio input, so it cannot hear audio clips.`);
    return readAudio(file);
  }
  if (file.type.startsWith('image/')) {
    if (!vision) throw new Error('The loaded model has no vision, so it cannot read images.');
    if (limits && file.size > limits.maxFileBytes) throw new Error(`${file.name} is ${(file.size / 1048576).toFixed(1)} MB; the server takes files up to ${Math.round(limits.maxFileBytes / 1048576)} MB.`);
    let dataUrl = await readAs(file, 'url');
    let mime = file.type;
    let dim = null;
    if (!SERVER_IMAGE.includes(mime)) { const p = await toPng(dataUrl); dataUrl = p.dataUrl; mime = 'image/png'; dim = p; }
    else {
      // an animated GIF / WebP is a video to a model that takes video; to any other model it is its first frame
      const anim = mime === 'image/gif' || mime === 'image/webp' ? animationInfo(new Uint8Array(await readAs(file, 'buffer'))) : null;
      if (anim && video) {
        return { kind: 'video', animation: true, name: file.name, mime, size: file.size, dataUrl, frames: anim.frames, duration: anim.durationMs / 1000,
          width: anim.width, height: anim.height, note: `animation · ${anim.frames} frames · ${fmtDuration(anim.durationMs / 1000)}` };
      }
      dim = await imageSize(dataUrl);
      if (anim) return { kind: 'image', name: file.name, mime, size: file.size, dataUrl, note: 'animation: the first frame only (this model takes no video)', ...(dim ? { width: dim.width, height: dim.height } : {}) };
    }
    return { kind: 'image', name: file.name, mime, size: file.size, dataUrl, ...(dim ? { width: dim.width, height: dim.height } : {}) };
  }
  if (isProbablyText(file)) {
    if (file.size > 4 * 1048576) throw new Error(`${file.name} is larger than 4 MB of text.`);
    return { kind: 'text', name: file.name, mime: file.type || 'text/plain', size: file.size, text: await readAs(file, 'text') };
  }
  throw new Error(`${file.name}: only images, ${video ? 'videos, ' : ''}${audio ? 'audio clips, ' : ''}text files and PDFs can be attached.`);
}

export function textAttachment(name, text) {
  return { kind: 'text', name, mime: 'text/plain', size: new Blob([text]).size, text };
}
