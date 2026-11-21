// Files the user attaches: images (sent to the vision tower), text files (sent as text), PDFs (text via pdf.js, or page
// images when they have no text layer or the setting says so), audio clips (models with audio input; converted to a 16 kHz
// mono WAV by lib/audio.js). The server decodes PNG, JPEG, GIF and BMP; other image types (WebP, AVIF, SVG, ...) are
// converted to PNG here first.
import { isAudioFile, readAudio } from './audio.js';
const SERVER_IMAGE = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp'];
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

export async function readAttachment(file, { pdfMode = 'text', vision = true, audio = false } = {}) {
  if (file.type === 'application/pdf' || /\.pdf$/i.test(file.name)) return readPdf(file, pdfMode, vision);
  if (isAudioFile(file)) {
    if (!audio) throw new Error(`${file.name}: the loaded model has no audio input, so it cannot hear audio clips.`);
    return readAudio(file);
  }
  if (file.type.startsWith('image/')) {
    if (!vision) throw new Error('The loaded model has no vision, so it cannot read images.');
    let dataUrl = await readAs(file, 'url');
    let mime = file.type;
    let dim = null;
    if (!SERVER_IMAGE.includes(mime)) { const p = await toPng(dataUrl); dataUrl = p.dataUrl; mime = 'image/png'; dim = p; }
    else dim = await imageSize(dataUrl);
    return { kind: 'image', name: file.name, mime, size: file.size, dataUrl, ...(dim ? { width: dim.width, height: dim.height } : {}) };
  }
  if (isProbablyText(file)) {
    if (file.size > 4 * 1048576) throw new Error(`${file.name} is larger than 4 MB of text.`);
    return { kind: 'text', name: file.name, mime: file.type || 'text/plain', size: file.size, text: await readAs(file, 'text') };
  }
  throw new Error(`${file.name}: only images, ${audio ? 'audio clips, ' : ''}text files and PDFs can be attached.`);
}

export function textAttachment(name, text) {
  return { kind: 'text', name, mime: 'text/plain', size: new Blob([text]).size, text };
}
