// Unit tests of the moving-picture helpers (no DOM): node --test ui/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../js/lib/media.js';
import { userContent } from '../js/lib/tree.js';
import { estimateMessages, estimateVideo, estimateImage, setVisionLimits } from '../js/lib/context.js';

// a GIF of `delays.length` frames of 1x1 pixels (hundredths of a second each)
function gif(delays, { w = 3, h = 2, comment = false } = {}) {
  const b = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61, w & 255, w >> 8, h & 255, h >> 8, 0x80, 0, 0, 0, 0, 0, 255, 255, 255];
  if (comment) b.push(0x21, 0xfe, 3, 0x61, 0x62, 0x63, 0);
  for (const d of delays) {
    b.push(0x21, 0xf9, 4, 0, d & 255, d >> 8, 0, 0);            // graphic control extension
    b.push(0x2c, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 2, 0x44, 0x01, 0); // image descriptor + LZW data
  }
  b.push(0x3b);
  return new Uint8Array(b);
}
function chunk(tag, body) {
  const n = body.length;
  return [...tag].map((c) => c.charCodeAt(0)).concat([n & 255, (n >> 8) & 255, (n >> 16) & 255, 0], body, n & 1 ? [0] : []);
}
// an animated WebP container of frames with the given durations (the frame payloads are empty: only the headers are read)
function webp(durations, { w = 40, h = 30, animated = true } = {}) {
  const vp8x = [animated ? 0x02 : 0, 0, 0, 0, (w - 1) & 255, ((w - 1) >> 8) & 255, 0, (h - 1) & 255, ((h - 1) >> 8) & 255, 0];
  let body = chunk('VP8X', vp8x).concat(chunk('ANIM', [0, 0, 0, 0, 0, 0]));
  for (const d of durations) body = body.concat(chunk('ANMF', [0, 0, 0, 0, 0, 0, (w - 1) & 255, 0, 0, (h - 1) & 255, 0, 0, d & 255, (d >> 8) & 255, (d >> 16) & 255, 0]));
  const size = body.length + 4;
  return new Uint8Array([0x52, 0x49, 0x46, 0x46, size & 255, (size >> 8) & 255, (size >> 16) & 255, 0, 0x57, 0x45, 0x42, 0x50].concat(body));
}

test('an animated GIF is told from a still one by its frames', () => {
  assert.deepEqual(M.animationInfo(gif([10, 10, 25])), { type: 'gif', width: 3, height: 2, frames: 3, durationMs: 450 });
  assert.deepEqual(M.animationInfo(gif([4, 4], { comment: true, w: 300, h: 200 })), { type: 'gif', width: 300, height: 200, frames: 2, durationMs: 80 });
  assert.equal(M.animationInfo(gif([10])), null);                  // one frame: an image
  assert.equal(M.animationInfo(gif([])), null);
  assert.equal(M.animationInfo(new Uint8Array([1, 2, 3])), null);
  assert.equal(M.animationInfo(new Uint8Array(0)), null);
  // a truncated file never throws; what was read counts
  const g = gif([10, 10, 10, 10]);
  for (let n = 0; n < g.length; n++) assert.doesNotThrow(() => M.animationInfo(g.slice(0, n)));
  assert.equal(M.animationInfo(g.slice(0, g.length - 12)).frames, 3);   // the last image's header is cut
});

test('an animated WebP is read from its container', () => {
  assert.deepEqual(M.animationInfo(webp([100, 100, 40])), { type: 'webp', width: 40, height: 30, frames: 3, durationMs: 240 });
  assert.equal(M.animationInfo(webp([100])), null);
  assert.equal(M.animationInfo(webp([100, 100], { animated: false })), null);
  const w = webp([50, 50, 50]);
  for (let n = 0; n < w.length; n++) assert.doesNotThrow(() => M.animationInfo(w.slice(0, n)));
});

test('the frames drawn from a video file follow its length, the rate and the limit', () => {
  assert.deepEqual(M.planFrames(3, { fps: 2, maxFrames: 64 }), { fps: 2, times: [0, 0.5, 1, 1.5, 2, 2.5] });
  assert.deepEqual(M.planFrames(0.4, { fps: 2, maxFrames: 64 }), { fps: 2, times: [0] });
  // a long video: the limit, spread over its whole length, with the thinner rate
  const long = M.planFrames(120, { fps: 2, maxFrames: 60 });
  assert.equal(long.times.length, 60);
  assert.equal(long.fps, 0.5);
  assert.equal(long.times[59], 118);
  // frame i is at i / fps, and no moment is past the end
  const p = M.planFrames(10.3, { fps: 2, maxFrames: 64 });
  assert.equal(p.times.length, 21);
  p.times.forEach((t, i) => { assert.ok(Math.abs(t - i / p.fps) < 1e-9); assert.ok(t < 10.3); });
  assert.deepEqual(M.planFrames(NaN), { fps: 2, times: [0] });
  assert.deepEqual(M.planFrames(Infinity), { fps: 2, times: [0] });
  assert.equal(M.planFrames(5, { fps: 0, maxFrames: 0 }).times.length, 1);
});

test('a drawn frame keeps its shape under the longer-side limit', () => {
  assert.deepEqual(M.frameSize(1920, 1080, 768), { width: 768, height: 432 });
  assert.deepEqual(M.frameSize(1080, 1920, 768), { width: 432, height: 768 });
  assert.deepEqual(M.frameSize(640, 360, 768), { width: 640, height: 360 });
  assert.deepEqual(M.frameSize(1920, 1080, 0), { width: 1920, height: 1080 });
});

test('image and video token estimates follow the server limits', () => {
  const L = M.visionLimits({ vision: { max_pixels: 1048576, max_tokens: 16384, min_pixels: 65536, video: { fps: 2, min_frames: 4, max_frames: 768, max_pixels: 25165824 } } });
  assert.equal(M.imageTokens(1024, 1024, L), 1026);
  assert.equal(M.imageTokens(4000, 3000, L), 1026);                 // resized down to the tower's largest image
  assert.equal(M.imageTokens(64, 64, L), 66);                       // resized up to the smallest
  assert.equal(M.visionLimits(null).maxPixels, 1048576);
  // 10 s at 2 frames/s: 20 frames, 10 temporal patches of 640x360 (225 tokens) + their markers and time labels
  const v = M.videoTokens({ frames: 20, fps: 2, width: 640, height: 360 }, L);
  assert.deepEqual([v.kept, v.patches, v.tokens], [20, 10, 2250 + 80]);
  // a 30 frames/s animation of 3 s: 6 frames are kept
  assert.equal(M.videoTokens({ frames: 90, durationMs: 3000, width: 320, height: 240 }, L).kept, 6);
  // a long video shares the model's pixel budget among its frames: 25,165,824 / 400 pixels each, 61 tokens a patch
  const big = M.videoTokens({ frames: 400, fps: 2, width: 1024, height: 1024 }, L);
  assert.deepEqual([big.patches, big.tokens], [200, 200 * 61 + 200 * 8]);
  // and never estimates above the request's image tokens (plus its labels)
  const tight = M.videoTokens({ frames: 400, fps: 2, width: 1024, height: 1024 }, { ...L, maxTokens: 4096 });
  assert.equal(tight.tokens, 4096 + tight.patches * 8);
  setVisionLimits({ vision: { max_pixels: 262144, max_tokens: 4096, min_pixels: 65536 } });
  assert.equal(estimateImage(2000, 2000), 258);
  setVisionLimits(null);
});

test('video attachments become video parts, after the images and before the text', () => {
  const frames = ['data:image/jpeg;base64,AA', 'data:image/jpeg;base64,BB'];
  const m = { content: 'What happens?', attachments: [
    { kind: 'video', name: 'clip.mp4', frames, fps: 2, duration: 1, width: 64, height: 64 },
    { kind: 'image', name: 'a.png', dataUrl: 'data:image/png;base64,CC' },
    { kind: 'video', animation: true, name: 'a.gif', dataUrl: 'data:image/gif;base64,DD', frames: 12, duration: 1.2 },
  ] };
  assert.deepEqual(userContent(m), [
    { type: 'image_url', image_url: { url: 'data:image/png;base64,CC' } },
    { type: 'video', video: { frames, fps: 2 } },
    { type: 'video_url', video_url: { url: 'data:image/gif;base64,DD' } },
    { type: 'text', text: 'What happens?' },
  ]);
  assert.equal(M.fmtDuration(7), '0:07');
  assert.equal(M.fmtDuration(83.4), '1:23');
  assert.equal(M.fmtDuration(3723), '1:02:03');
  // estimates count a video part by its frames
  const est = estimateMessages([{ role: 'user', content: userContent(m) }]);
  assert.ok(est > estimateVideo(m.attachments[0]) && est < 3000, String(est));
});
