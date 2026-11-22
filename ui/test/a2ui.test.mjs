// Unit tests of the A2UI client core (no DOM): node --test ui/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as A from '../js/lib/a2ui.js';

const block = (msgs) => `${A.OPEN_TAG}\n${JSON.stringify(msgs)}\n${A.CLOSE_TAG}`;
const create = (surfaceId, components, dataModel = {}) => ({ version: '1.0', createSurface: { surfaceId, catalogId: A.BASIC_CATALOG, components, dataModel } });

test('a reply splits into text and interface blocks', () => {
  assert.deepEqual(A.splitContent('just text'), [{ t: 'text', text: 'just text' }]);
  assert.deepEqual(A.splitContent(''), []);
  const p = A.splitContent(`Pick one:\n\n${block([])}\n\nDone.`);
  assert.deepEqual(p.map((x) => x.t), ['text', 'a2ui', 'text']);
  assert.equal(p[0].text, 'Pick one:\n\n');
  assert.equal(p[1].closed, true);
  assert.deepEqual(A.parseBlock(p[1].raw), { messages: [] });
  // two blocks, nothing between them
  assert.deepEqual(A.splitContent(block([]) + '\n' + block([])).map((x) => x.t), ['a2ui', 'a2ui']);
  // an open block at the end of a stream stays open
  const open = A.splitContent('Here:\n<a2ui-json>\n[{"version"', { live: true });
  assert.deepEqual(open.map((x) => [x.t, x.closed]), [['text', undefined], ['a2ui', false]]);
});

test('a tag that is still arriving is held back, and quoting around a block is dropped', () => {
  for (let n = 1; n < A.OPEN_TAG.length; n++) {
    const part = `Sure.\n${A.OPEN_TAG.slice(0, n)}`;
    assert.deepEqual(A.splitContent(part, { live: true }), [{ t: 'text', text: 'Sure.\n' }], part);
    assert.equal(A.splitContent(part)[0].text, part);   // a finished reply shows what it holds
  }
  assert.equal(A.heldTail('a < b'), 0);
  assert.equal(A.heldTail('x <a2'), 3);
  const fenced = A.splitContent('Form:\n```json\n<a2ui-json>\n[]\n</a2ui-json>\n```\nAfter.');
  assert.deepEqual(fenced, [{ t: 'text', text: 'Form:\n' }, { t: 'a2ui', raw: '\n[]\n', closed: true }, { t: 'text', text: '\nAfter.' }]);
  const ticks = A.splitContent('`<a2ui-json>`[]`</a2ui-json>`');
  assert.deepEqual(ticks.map((x) => x.t), ['a2ui']);
  assert.deepEqual(A.parseBlock(ticks[0].raw), { messages: [] });
});

test('a block is a list of messages, in several spellings', () => {
  assert.equal(A.parseBlock('[{"a":1},{"b":2}]').messages.length, 2);
  assert.equal(A.parseBlock('{"a":1}').messages.length, 1);
  assert.equal(A.parseBlock('{"a":1}\n{"b":2}').messages.length, 2);
  assert.equal(A.parseBlock('```json\n[{"a":1}]\n```').messages.length, 1);
  assert.equal(A.parseBlock('[{"a":1},').repaired, true);
  assert.match(A.parseBlock('7').error, /not a JSON array/);
});

test('a bracket too few or too many is repaired', () => {
  const good = '[{"a":{"b":[1,{"c":"x]}"}]}},{"d":2}]';
  assert.equal(A.balanceJson(good), good);
  for (const bad of ['[{"a":{"b":[1,{"c":"x]}"}]}},{"d":2}', '[{"a":{"b":[1,{"c":"x]}"}]}}},{"d":2}]', '[{"a":{"b":[1,{"c":"x]}"}]}},{"d":2},]', '[{"a":{"b":[1,{"c":"x]}"}}},{"d":2}]']) {
    const r = A.parseBlock(bad);
    assert.equal(r.repaired, true, bad);
    assert.equal(r.messages.length, 2, bad);
    assert.deepEqual(r.messages[1], { d: 2 }, bad);
  }
  assert.equal(A.parseBlock(good).repaired, undefined);
  assert.match(A.parseBlock('[{"a" 1}]').error, /not valid JSON/);   // not a bracket slip: reported
  assert.match(A.parseBlock('[{"a":{"b":1},{"d":2}]').error, /not valid JSON/);   // a bracket missing in the middle cannot be placed: reported
  const rt = new A.A2uiRuntime();
  const r = rt.applyBlock('k', '[{"version":"1.0","createSurface":{"surfaceId":"s","components":[{"id":"root","component":"Divider"}],"dataModel":{"a":{"b":1}}}]');
  assert.deepEqual([r.created, r.repaired, r.errors], [['s'], true, []]);
  assert.deepEqual(rt.surfaces.get('s').data, { a: { b: 1 } });
});

test('JSON pointers read and write the data model', () => {
  const d = { user: { name: 'Ada', tags: ['x', 'y'] }, 'a/b': 1 };
  assert.equal(A.getAt(d, '/user/name'), 'Ada');
  assert.equal(A.getAt(d, '/user/tags/1'), 'y');
  assert.equal(A.getAt(d, '/a~1b'), 1);
  assert.equal(A.getAt(d, '/nope/deeper'), undefined);
  assert.equal(A.getAt(d, '/'), d);
  assert.equal(A.resolvePath('name', '/users/2'), '/users/2/name');
  assert.equal(A.resolvePath('/company', '/users/2'), '/company');
  assert.equal(A.resolvePath('.', '/users/2'), '/users/2');
  let m = A.setAt({}, '/form/email', 'a@b.co');
  assert.deepEqual(m, { form: { email: 'a@b.co' } });
  m = A.setAt(m, '/list/0/done', true);
  assert.deepEqual(m.list, [{ done: true }]);
  m = A.setAt(m, '/form/email', null);
  assert.deepEqual(m.form, {});
  assert.deepEqual(A.setAt(m, '/', { fresh: 1 }), { fresh: 1 });
});

test('messages build, update and delete surfaces', () => {
  const rt = new A.A2uiRuntime();
  const r = rt.applyBlock('k1', JSON.stringify([create('s1', [
    { id: 'root', component: 'Column', children: ['t'] }, { id: 't', component: 'Text', text: { '@path': '/msg' } }], { msg: 'hi' })]));
  assert.deepEqual([r.created, r.errors, r.parseError], [['s1'], [], null]);
  const s = rt.surfaces.get('s1');
  assert.equal(s.anchor, 'k1');
  assert.equal(A.resolve(s.components.get('t').text, { data: s.data, scope: '' }), 'hi');
  // applying the same block again changes nothing
  assert.equal(rt.applyBlock('k1', rt.log[0].raw), r);
  assert.equal(rt.log.length, 1);

  const u = rt.applyBlock('k2', JSON.stringify([
    { version: '1.0', updateDataModel: { surfaceId: 's1', path: '/msg', value: 'bye' } },
    { version: '1.0', updateComponents: { surfaceId: 's1', components: [{ id: 'extra', component: 'Divider' }] } }]));
  assert.deepEqual([u.touched, u.errors], [['s1', 's1'], []]);
  assert.equal(s.data.msg, 'bye');
  assert.equal(s.anchor, 'k2');   // the surface is drawn where it was last touched
  assert.equal(s.components.size, 3);

  rt.applyBlock('k3', JSON.stringify([{ version: '1.0', updateDataModel: { surfaceId: 's1', value: { all: 'new' } } }]));
  assert.deepEqual(s.data, { all: 'new' });
  const d = rt.applyBlock('k4', JSON.stringify([{ version: '1.0', deleteSurface: { surfaceId: 's1' } }]));
  assert.deepEqual(d.deleted, ['s1']);
  assert.equal(rt.surfaces.size, 0);
});

test('what is wrong with a block is reported with a path', () => {
  const rt = new A.A2uiRuntime();
  const codes = (r) => r.errors.map((e) => `${e.path}: ${e.message}`).join('\n');
  assert.match(rt.applyBlock('a', '[{"oops" 1').parseError, /not valid JSON/);
  let r = rt.applyBlock('b', JSON.stringify([{ version: '1.0', updateDataModel: { surfaceId: 'none', value: 1 } }, { nothing: true }, 5]));
  assert.match(codes(r), /There is no surface "none"/);
  assert.match(codes(r), /needs one of createSurface/);
  assert.match(codes(r), /must be a JSON object/);
  r = rt.applyBlock('c', JSON.stringify([create('s', [
    { id: 'root', component: 'Column', children: ['ghost', 'btn', 'loop'] },
    { id: 'btn', component: 'Button', child: { component: 'Text', text: 'nested' } },
    { id: 'loop', component: 'Card', child: 'root' },
    { id: 'x', component: 'Sparkle' }, { component: 'Text', text: 'no id' }, { id: 'nt', component: 'Text' }])]));
  const text = codes(r);
  for (const want of [/"root" refers to "ghost"/, /holds a component object/, /Button "btn" needs "action"/, /Unknown component type "Sparkle"/, /needs a string "id"/, /Text "nt" needs "text"/, /contains itself/]) assert.match(text, want);
  assert.ok(r.errors.every((e) => e.code === 'VALIDATION_FAILED' && e.surfaceId === 's'));
  r = rt.applyBlock('d', JSON.stringify([create('s', [{ id: 'root', component: 'Divider' }])]));
  assert.match(codes(r), /already exists/);
  r = rt.applyBlock('e', JSON.stringify([create('noroot', [{ id: 'a', component: 'Divider' }])]));
  assert.match(codes(r), /No component has the id "root"/);
});

test('dynamic values: bindings, scopes, calls and literal objects', () => {
  const data = { company: 'Acme', users: [{ name: 'Ada', n: 3 }, { name: 'Bo', n: 1 }], total: 1234.5, when: '2026-01-16T14:30:00' };
  const root = { data, scope: '' };
  const item = { data, scope: '/users/1', index: 1 };
  assert.equal(A.resolve({ '@path': 'name' }, item), 'Bo');
  assert.equal(A.resolve({ '@path': '/company' }, item), 'Acme');
  assert.equal(A.resolve({ path: '/company' }, root), 'Acme');              // the earlier spelling of a binding
  assert.deepEqual(A.resolve({ path: '/company', label: 'x' }, root), { path: 'Acme'.length ? '/company' : '', label: 'x' });   // a literal object
  assert.deepEqual(A.resolve({ a: [{ '@path': '/total' }], '@@type': 'x' }, root), { a: [1234.5], '@type': 'x' });
  assert.equal(A.resolve({ '@call': '@index', args: { offset: 1 } }, item), 2);
  assert.equal(A.resolve({ '@call': 'formatString', args: { value: 'Hi ${name} of ${/company} (#${@index(offset: 1)})' } }, item), 'Hi Bo of Acme (#2)');
  assert.equal(A.formatString('cost \\${5} and ${/missing}!', root), 'cost ${5} and !');
  assert.equal(A.formatString("${formatDate(value:${/when}, format:'MMM d, yyyy')} at ${formatDate(value:${/when}, format:'h:mm a')}", root), 'Jan 16, 2026 at 2:30 PM');
  assert.equal(A.formatString('${/users}', { data: { users: [1] }, scope: '' }), '[1]');
  assert.equal(A.resolve({ '@call': 'pluralize', args: { value: { '@path': 'n' }, one: 'item', other: 'items' } }, item), 'item');
  assert.equal(A.resolve({ '@call': 'formatNumber', args: { value: 1234.5, decimals: 1, grouping: false } }, root), new Intl.NumberFormat(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1, useGrouping: false }).format(1234.5));
  assert.match(A.resolve({ '@call': 'formatCurrency', args: { value: { '@path': '/total' }, currency: 'USD' } }, root), /1.?234[.,]50/);
  assert.equal(A.formatDate('2026-01-16', 'EEEE, d MMMM yy'), 'Friday, 16 January 26');
  assert.equal(A.formatDate('14:05', 'HH:mm'), '14:05');
  assert.equal(A.resolve({ '@call': 'nope', args: {} }, root), undefined);
  assert.equal(A.resolve({ '@call': 'openUrl', args: { url: 'https://example.com' } }, root), undefined);   // never outside a gesture
});

test('checks and the boolean coercion', () => {
  const ctx = { data: { form: { email: 'a@b.co', name: '', age: 17, terms: true, pick: [] } }, scope: '' };
  const call = (name, args) => ({ '@call': name, args });
  const p = (path) => ({ '@path': path });
  assert.deepEqual(A.runChecks({ '@checks': [{ condition: call('required', { value: p('/form/name') }), message: 'Name is required.' }] }, ctx), { valid: false, messages: ['Name is required.'], failed: 1 });
  assert.equal(A.runChecks({ '@checks': [{ condition: call('email', { value: p('/form/email') }) }] }, ctx).valid, true);
  assert.equal(A.runChecks({ '@checks': [{ condition: call('required', { value: p('/form/pick') }) }] }, ctx).valid, false);
  assert.equal(A.runChecks({ '@checks': [{ condition: call('numeric', { value: p('/form/age'), min: 18 }), message: 'Adults only.' }] }, ctx).messages[0], 'Adults only.');
  assert.equal(A.runChecks({ '@checks': [{ condition: call('length', { value: 'abcd', min: 2, max: 3 }) }] }, ctx).failed, 1);
  const both = call('and', { values: [p('/form/terms'), call('or', { values: [call('required', { value: p('/form/name') }), call('required', { value: p('/form/email') })] })] });
  assert.equal(A.runChecks({ '@checks': [{ condition: both }] }, ctx).valid, true);
  assert.equal(A.runChecks({ '@checks': [{ condition: call('not', { value: both }), message: 'no' }] }, ctx).valid, false);
  assert.equal(A.runChecks({ '@checks': [{ condition: p('/form/terms') }] }, ctx).valid, true);
  assert.equal(A.runChecks({}, ctx).valid, true);
  for (const [v, want] of [[true, true], ['TRUE', true], ['hello', false], ['', false], [0, false], [2, true], [null, false], [{ valid: true }, true], [{ valid: false }, false], [[1], false], [{}, false]]) assert.equal(A.toBool(v), want, JSON.stringify(v));
});

test('regular expressions run only when they are safe', () => {
  const ok = (pattern, value) => A.callFunction('regex', { value, pattern }, { data: {}, scope: '' }).valid;
  assert.equal(ok('^[a-z]+@[a-z]+\\.com$', 'ada@acme.com'), true);
  assert.equal(ok('^(?=.*\\d).{8,}$', 'passw0rdx'), true);
  assert.equal(ok('\\w+\\s+\\w+', 'two words'), true);
  assert.equal(ok('^\\d{5}$', '1234'), false);
  for (const bad of ['(a+)+$', '(a|ab)*c', '\\d+\\s*\\d+', '(\\w)\\1', '(?<=a)b', '(?<n>a)', '\\p{L}+', 'x'.repeat(257), '(']) assert.equal(A.safeRegex(bad), null, bad);
  assert.equal(ok('a', 'a'.repeat(5000)), false);   // a very long value never matches
});

test('a pressed button becomes a message for the model', () => {
  const rt = new A.A2uiRuntime();
  rt.applyBlock('k', JSON.stringify([create('order', [
    { id: 'root', component: 'Button', child: 'l', action: { event: { name: 'placeOrder', userMessage: { '@call': 'formatString', args: { value: 'Ordered a ${/size/0}' } }, context: { size: { '@path': '/size' }, item: 'latte', missing: { '@path': '/nope' } } } } },
    { id: 'l', component: 'Text', text: 'Go' }], { size: ['large'] })]));
  const s = rt.surfaces.get('order');
  const m = A.actionMessage(s, s.components.get('root'), { data: s.data, scope: '' }, new Date('2026-06-02T08:57:23Z'));
  assert.deepEqual(m, { version: '1.0', action: { name: 'placeOrder', surfaceId: 'order', sourceComponentId: 'root', timestamp: '2026-06-02T08:57:23.000Z', userMessage: 'Ordered a large', context: { size: ['large'], item: 'latte', missing: null } } });
  const turn = A.wrapMessage(m);
  assert.deepEqual(A.readUserTurn(turn), [m]);
  assert.deepEqual(A.readUserTurn('hello <a2ui-json>'), []);
  assert.deepEqual(A.readUserTurn(`${turn}\nand some words`), []);
  assert.equal(A.actionLabel(m), 'Ordered a large');
  assert.equal(A.actionLabel({ action: { name: 'place_orderNow' } }), 'Place order now');
  // the whole data model rides along when the surface asks for it
  s.sendDataModel = true;
  assert.deepEqual(A.actionMessage(s, s.components.get('root'), { data: s.data, scope: '' }).action.metadata, { extensions: { gaudium_data_model: { size: ['large'] } } });
  // a local action: only links of a safe scheme
  const link = (url) => A.localAction({ action: { functionCall: { '@call': 'openUrl', args: { url } } } }, { data: { u: 'https://example.com/a' }, scope: '' });
  assert.deepEqual(link({ '@path': '/u' }), { openUrl: 'https://example.com/a' });
  assert.equal(link('javascript:alert(1)'), null);
  assert.equal(A.localAction({ action: { event: { name: 'x' } } }, { data: {}, scope: '' }), null);
  assert.equal(A.actionMessage(s, { id: 'b', action: { functionCall: { '@call': 'openUrl', args: {} } } }, { data: {}, scope: '' }), null);
  assert.equal(A.safeUrl('data:image/png;base64,AAAA', ['https:', 'data:']), 'data:image/png;base64,AAAA');
  assert.equal(A.safeUrl('data:text/html;base64,AAAA', ['https:', 'data:']), '');
  assert.deepEqual(A.errorMessage({ surfaceId: 's', path: '/x', message: 'bad' }), { version: '1.0', error: { code: 'VALIDATION_FAILED', surfaceId: 's', path: '/x', message: 'bad' } });
});

test('the surfaces follow the shown path', () => {
  const b1 = block([create('one', [{ id: 'root', component: 'Divider' }])]);
  const b2 = block([{ version: '1.0', updateDataModel: { surfaceId: 'one', path: '/x', value: 1 } }]);
  const path = [{ id: 'u1', role: 'user', content: 'hi' }, { id: 'a1', role: 'assistant', segments: [{ content: `Here.\n${b1}` }, { kind: 'compaction', content: b1 }] }];
  let blocks = A.blocksOfPath(path);
  assert.deepEqual(blocks.map((b) => b.key), ['a1:0:0']);
  const rt = A.syncRuntime(null, blocks);
  assert.equal(rt.surfaces.get('one').anchor, 'a1:0:0');
  rt.setData('one', '/typed', 'kept');
  // a streaming segment: an open block is not applied, a closed one is, under the key it keeps once the segment is stored
  const next = { id: 'a2', role: 'assistant', segments: [] };
  blocks = A.blocksOfPath([...path, next], { id: 'a2', seg: { content: b2.slice(0, 40) } });
  assert.equal(A.syncRuntime(rt, blocks), rt);
  assert.equal(rt.surfaces.get('one').data.x, undefined);
  blocks = A.blocksOfPath([...path, next], { id: 'a2', seg: { content: b2 } });
  assert.equal(A.syncRuntime(rt, blocks), rt);
  assert.deepEqual(rt.surfaces.get('one').data, { typed: 'kept', x: 1 });
  assert.equal(rt.surfaces.get('one').anchor, 'a2:0:0');
  assert.deepEqual(A.blocksOfPath([...path, { ...next, segments: [{ content: b2 }] }]), blocks);
  // another branch: a fresh runtime replays the path
  const other = A.syncRuntime(rt, A.blocksOfPath(path));
  assert.notEqual(other, rt);
  assert.deepEqual(other.surfaces.get('one').data, {});
});

test('the example in the server instructions is a valid interface', () => {
  const text = readFileSync(new URL('../../gaudium/src/server/a2ui_prompt.md', import.meta.url), 'utf8');
  const parts = A.splitContent(text.slice(text.lastIndexOf('## Example')));
  const rt = new A.A2uiRuntime();
  const r = rt.applyBlock('doc', parts.find((p) => p.t === 'a2ui').raw);
  assert.deepEqual([r.created, r.errors, r.parseError], [['coffee_order_1'], [], null]);
  const s = rt.surfaces.get('coffee_order_1');
  const send = s.components.get('send');
  assert.equal(A.runChecks(send, { data: s.data, scope: '' }).valid, false);   // no name yet: the button is disabled
  rt.setData(s.id, '/order/name', 'Ada');
  assert.equal(A.runChecks(send, { data: s.data, scope: '' }).valid, true);
  assert.deepEqual(A.actionMessage(s, send, { data: s.data, scope: '' }).action.context, { size: ['small'], name: 'Ada', extras: [{ label: 'Oat milk', chosen: false }, { label: 'Extra shot', chosen: false }] });
  for (const name of [...text.matchAll(/`(\w+)` \{/g)].map((m) => m[1])) assert.ok(A.COMPONENTS[name] || A.FUNCTION_NAMES.includes(name), `${name} is described but not drawn`);
});
