// Moving pictures: what an animated image file holds (read from its bytes, no decoding), which frames of a video file the
// browser extracts for the model, and how many tokens a video costs. No DOM here (unit-tested in node: test/media.test.mjs).
//
// The server decodes no video container. A video is sent as its frames: an animated GIF / WebP goes as the file itself
// (the server decodes its frames), a video file as frames this page draws from a <video> element (lib/attachments.js).

const u16le = (b, o) => b[o] | (b[o + 1] << 8);
const u24le = (b, o) => b[o] | (b[o + 1] << 8) | (b[o + 2] << 16);
const u32le = (b, o) => (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0;
const tag = (b, o) => String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);

// GIF: the logical screen and, per image, the delay of the graphic control extension before it (hundredths of a second)
function gifInfo(b) {
  if (b.length < 13) return null;
  const width = u16le(b, 6), height = u16le(b, 8);
  let o = 13;
  if (b[10] & 0x80) o += 3 << ((b[10] & 7) + 1);   // global colour table
  let frames = 0, durationMs = 0, delay = 0;
  const skipBlocks = () => { while (o < b.length && b[o] !== 0) o += b[o] + 1; o += 1; };
  while (o < b.length) {
    const id = b[o];
    if (id === 0x3b) break;                           // trailer
    if (id === 0x21) {                                // extension
      if (b[o + 1] === 0xf9 && o + 6 < b.length) delay = u16le(b, o + 4) * 10;
      o += 2; skipBlocks();
    } else if (id === 0x2c) {                         // image descriptor
      if (o + 10 > b.length) break;
      const flags = b[o + 9];
      o += 10;
      if (flags & 0x80) o += 3 << ((flags & 7) + 1);  // local colour table
      o += 1;                                         // LZW minimum code size
      skipBlocks();
      frames += 1; durationMs += delay; delay = 0;
    } else break;                                     // not a block: stop at what was read
  }
  return frames ? { type: 'gif', width, height, frames, durationMs } : null;
}

// WebP: the canvas of the extended header and one ANMF chunk per frame (its duration in milliseconds)
function webpInfo(b) {
  if (b.length < 30 || tag(b, 12) !== 'VP8X' || !(b[20] & 0x02)) return null;   // no animation flag: a still image
  const width = u24le(b, 24) + 1, height = u24le(b, 27) + 1;
  let o = 12, frames = 0, durationMs = 0;
  while (o + 8 <= b.length) {
    const size = u32le(b, o + 4);
    if (tag(b, o) === 'ANMF' && o + 24 <= b.length) { frames += 1; durationMs += u24le(b, o + 8 + 12); }
    o += 8 + size + (size & 1);
  }
  return frames ? { type: 'webp', width, height, frames, durationMs } : null;
}

// {type, width, height, frames, durationMs} of an animated GIF / WebP with more than one frame; null for anything else
// (a still image, another format, a damaged file: those are sent as images)
export function animationInfo(bytes) {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let info = null;
  try {
    if (b.length > 6 && tag(b, 0) === 'GIF8') info = gifInfo(b);
    else if (b.length > 16 && tag(b, 0) === 'RIFF' && tag(b, 8) === 'WEBP') info = webpInfo(b);
  } catch { info = null; }
  return info && info.frames > 1 ? info : null;
}

// The moments of a video file that become frames: `fps` frames per second of video time from its start, at most `maxFrames`
// (a longer video is sampled more thinly, evenly, and the rate sent with the frames is the thinner one). Frame i is the
// picture at i / fps seconds, which is where the server puts it on the model's time axis.
export function planFrames(duration, { fps = 2, maxFrames = 64 } = {}) {
  const d = Number(duration);
  if (!(d > 0) || !Number.isFinite(d)) return { fps, times: [0] };
  let rate = fps > 0 ? fps : 2;
  const most = Math.max(1, Math.floor(maxFrames) || 1);
  let n = Math.max(1, Math.ceil(d * rate - 1e-9));
  if (n > most) { n = most; rate = n / d; }
  const times = [];
  for (let i = 0; i < n; i++) times.push(Math.min(i / rate, Math.max(0, d - 0.001)));
  return { fps: rate, times };
}

// The size a frame is drawn at: the video's own, scaled down so that its longer side is at most `maxSide` (0 = as it is)
export function frameSize(width, height, maxSide) {
  const w = Math.max(1, Math.round(width)), h = Math.max(1, Math.round(height));
  const long = Math.max(w, h);
  if (!(maxSide > 0) || long <= maxSide) return { width: w, height: h };
  const k = maxSide / long;
  return { width: Math.max(1, Math.round(w * k)), height: Math.max(1, Math.round(h * k)) };
}

// What the server tells about its vision tower (/props `vision`), with the values of the default build when it tells nothing
export function visionLimits(props) {
  const v = (props && props.vision) || {};
  const video = v.video || null;
  return {
    maxPixels: v.max_pixels || 1048576,          // one image, after the processor's resize
    maxTokens: v.max_tokens || 16384,            // all images and frames of a request
    minPixels: v.min_pixels || 65536,
    maxFileBytes: v.max_file_bytes || 64 * 1048576,
    maxRequestBytes: v.max_request_bytes || 256 * 1048576,
    video: video ? { fps: video.fps || 2, minFrames: video.min_frames || 4, maxFrames: video.max_frames || 768, maxPixels: video.max_pixels || 25165824 } : null,
  };
}

// An image's tokens: one per 32 x 32 pixels after the processor's resize (patch 16, merge 2), plus its two markers
export function imageTokens(width, height, limits) {
  const L = limits || visionLimits(null);
  if (!(width > 0 && height > 0)) return Math.round(L.maxPixels / 1024 / 2) + 2;
  const px = Math.min(L.maxPixels, Math.max(L.minPixels, width * height));
  return Math.max(4, Math.round(px / 1024)) + 2;
}

// A video's tokens. The server keeps `fps` (its own rate, 2) frames per second of video time, between its fewest and most
// frames; two frames make one temporal patch; every patch costs its picture's tokens, its two markers and its time label
// (about 6 tokens: "<12.5 seconds>"). All frames share one pixel budget, a single frame is held to the largest image, and
// a video never takes more than the request's image tokens.
export function videoTokens({ frames, fps, durationMs, width, height }, limits) {
  const L = limits || visionLimits(null);
  const v = L.video || { fps: 2, minFrames: 4, maxFrames: 768, maxPixels: 25165824 };
  const total = Math.max(1, frames | 0);
  const seconds = durationMs > 0 ? durationMs / 1000 : fps > 0 ? total / fps : total / 24;
  const kept = Math.min(Math.max(Math.floor(seconds * v.fps), v.minFrames), v.maxFrames, total);
  const patches = Math.ceil(kept / 2);
  const w = width > 0 ? width : 640, h = height > 0 ? height : 360;
  const perFrame = Math.min(w * h, L.maxPixels, v.maxPixels / (patches * 2));
  const each = Math.max(1, Math.round(perFrame / 1024));
  const picture = Math.min(patches * each, L.maxTokens);
  return { kept, patches, tokens: picture + patches * 8 };
}

// "0:07", "1:23", "1:02:03"
export function fmtDuration(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}` : `${m}:${String(r).padStart(2, '0')}`;
}

// The content parts of a video attachment (lib/tree.js): an animation is its file, a video file its frames with their rate
export function videoPart(a) {
  if (a.animation) return { type: 'video_url', video_url: { url: a.dataUrl } };
  return { type: 'video', video: { frames: a.frames, fps: a.fps } };
}
