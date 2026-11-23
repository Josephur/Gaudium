// Unit tests of the effort picker's levels (no DOM): node --test ui/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { effortLevels, effortFor } from '../js/lib/model.js';
import { buildBody } from '../js/lib/api.js';

const caps = (supports, levels) => ({ chat_template_caps: { supports_reasoning_effort: supports, reasoning_effort_levels: levels } });

test('the picker offers the levels the server advertises, and none without an effort setting', () => {
  assert.deepEqual(effortLevels(caps(true, ['low', 'medium', 'xhigh'])), ['low', 'medium', 'xhigh']);
  // a template without an effort setting (Gemma 4): no picker
  assert.deepEqual(effortLevels(caps(false, [])), []);
  // the server does not say: no picker rather than guessed levels
  assert.deepEqual(effortLevels(null), []);
  assert.deepEqual(effortLevels({}), []);
  assert.deepEqual(effortLevels(caps(true, undefined)), []);
  assert.deepEqual(effortLevels(caps(false, ['low'])), []);
  assert.deepEqual(effortLevels(caps(true, ['low', 7, ''])), ['low']);
});

test('a request carries only a level the model has', () => {
  const levels = ['low', 'medium', 'xhigh'];
  assert.equal(effortFor('medium', levels), 'medium');
  assert.equal(effortFor('high', levels), 'default');
  assert.equal(effortFor('low', []), 'default');
  assert.equal(effortFor('default', levels), 'default');
  const body = (effort) => buildBody([], { thinking: 'on', effort: effortFor(effort, levels) });
  assert.equal(body('xhigh').reasoning_effort, 'xhigh');
  assert.equal(body('max').reasoning_effort, undefined);
  assert.deepEqual(body('max').chat_template_kwargs, { enable_thinking: true });
});
