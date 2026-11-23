// Unit tests of the readout numbers taken from a request's final details (no DOM): node --test ui/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { finalStats, reuseSource } from '../js/lib/stats.js';

const seg = (prefix, cacheN = 1024) => ({
  timings: { prompt_n: 510, cache_n: cacheN, prompt_ms: 107.5, predicted_n: 51, predicted_ms: 300, launches: 7 },
  gaudium: { method: 'dspine', prefix },
});

test('the readout says where reused tokens came from', () => {
  // copied back from host memory by the --cache-ram tier
  const host = finalStats(seg({ hit: true, reused_tokens: 1024, common_tokens: 1534, tier: 'host', host: { tier: 'host', restore_ms: 26.9, restore_bytes: 407502848 } }));
  assert.equal(host.prefixTier, 'host');
  assert.equal(host.hostRestoreMs, 26.9);
  assert.equal(reuseSource(host).text, 'from host memory');
  assert.match(reuseSource(host).title, /copied back from host memory in 27 ms/);
  // still on the card
  const dev = finalStats(seg({ hit: true, reused_tokens: 1024, common_tokens: 1534, tier: 'device' }));
  assert.equal(reuseSource(dev).text, 'still on the card');
  assert.equal(dev.hostRestoreMs, null);
});

test('nothing reused, or an engine that names no tier: no source is shown', () => {
  assert.equal(reuseSource(finalStats(seg({ hit: false, reused_tokens: 0, tier: 'prefilled' }, 0))), null);
  assert.equal(reuseSource(finalStats(seg({ hit: true, reused_tokens: 1024 }))), null);
  assert.equal(reuseSource(finalStats(seg(null))), null);
  assert.equal(reuseSource(null), null);
});
