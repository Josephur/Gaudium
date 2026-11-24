// Unit tests of the model description helpers the readout's model panel reads (no DOM): node --test ui/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { modelDesc, placementOf, expertsSplit } from '../js/lib/model.js';

const moe = (placement) => modelDesc({ model_desc: { name: 'm', moe: { experts: 128, top_k: 8, placement } } });

test('every expert on the card: no placement bars', () => {
  for (const p of [{ mode: 'card' }, { mode: 'card', card_gb: 49.6, host_gb: 0 }, undefined]) {
    const pl = placementOf(moe(p));
    assert.ok(pl, 'a MoE model still has a placement');
    assert.equal(pl.split, false);
    assert.equal(expertsSplit(moe(p).moe.placement), false);
  }
});

test('experts split between the card and another place: the bars are shown', () => {
  const hybrid = placementOf(moe({ mode: 'hybrid', card_gb: 30, host_gb: 20, layers: [[3, 64], [4, 64]] }));
  assert.equal(hybrid.split, true);
  assert.equal(Math.round(100 * hybrid.share), 60);
  // the expert cache is a hybrid placement too
  assert.equal(placementOf(moe({ mode: 'hybrid', card_gb: 30, host_gb: 20, cache: { on: true, policy: 'lru' }, layers: [[1, 8]] })).split, true);
  assert.equal(placementOf(moe({ mode: 'host', card_gb: 0, host_gb: 49, layers: [[0, 0]] })).split, true);
  // a placement over several devices that a later engine may report is not keyed on today's names
  assert.equal(placementOf(moe({ mode: 'devices', card_gb: 25, host_gb: 0 })).split, true);
  // host memory holding experts counts whatever the mode says
  assert.equal(expertsSplit({ mode: 'card', hostGb: 2.5 }), true);
});

test('a dense model or no description has no placement', () => {
  assert.equal(placementOf(modelDesc({ model_desc: { name: 'd' } })), null);
  assert.equal(expertsSplit(null), false);
  assert.equal(expertsSplit(undefined), false);
});
