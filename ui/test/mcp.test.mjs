// Unit tests of the MCP client's request shaping and failure texts (no DOM, no network): node --test ui/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { McpClient, connectFailureText, isMixedContent, parseHeaders } from '../js/lib/mcp.js';
import { setApiKey } from '../js/lib/api.js';

test('through the proxy the target is a relative path and only prefixed headers carry the request', () => {
  setApiKey('server-key');
  const c = new McpClient({ url: 'http://192.0.2.7:8931/mcp?a=1&b=2', proxy: true, headers: 'Authorization: Bearer mcp-token\nX-Tenant: t1' });
  // relative: the page reaches the proxy on whatever origin it was opened from (a LAN name, a tunnel, a front end)
  assert.equal(c.target(), '/cors-proxy?url=http%3A%2F%2F192.0.2.7%3A8931%2Fmcp%3Fa%3D1%26b%3D2');
  c.session = 'session-1';
  const h = c.headers();
  // the server's own key stays unprefixed: it authenticates the page to gaudium-server and is not forwarded
  assert.equal(h.Authorization, 'Bearer server-key');
  assert.equal(h['x-gaudium-proxy-header-authorization'], 'Bearer mcp-token');
  assert.equal(h['x-gaudium-proxy-header-x-tenant'], 't1');
  assert.equal(h['x-gaudium-proxy-header-content-type'], 'application/json');
  assert.equal(h['x-gaudium-proxy-header-accept'], 'application/json, text/event-stream');
  assert.equal(h['x-gaudium-proxy-header-mcp-session-id'], 'session-1');
  for (const k of Object.keys(h)) assert.ok(/^(Authorization|Content-Type|x-gaudium-proxy-header-[a-z0-9-]+)$/.test(k), k);
  setApiKey('');

  const direct = new McpClient({ url: 'https://mcp.example.com/mcp', proxy: false, headers: { 'X-Tenant': 't1' } });
  assert.equal(direct.target(), 'https://mcp.example.com/mcp');
  assert.deepEqual(direct.headers(), { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', 'X-Tenant': 't1' });
});

test('header text parses as lines or as JSON', () => {
  assert.deepEqual(parseHeaders('A: 1\nB: x: y\n\nnot a header'), { A: '1', B: 'x: y' });
  assert.deepEqual(parseHeaders('{"A":"1"}'), { A: '1' });
  assert.deepEqual(parseHeaders(''), {});
});

test('an https page and a plain-http server: the text names the cause and the way out', () => {
  assert.equal(isMixedContent('https://chat.example.com', 'http://192.0.2.7:8931/mcp'), true);
  assert.equal(isMixedContent('https://chat.example.com', 'https://mcp.example.com/mcp'), false);
  assert.equal(isMixedContent('http://192.0.2.5:8090', 'http://192.0.2.7:8931/mcp'), false);
  // this computer's own loopback names are exempt in browsers
  for (const u of ['http://localhost:8931/mcp', 'http://127.0.0.1:8931/mcp', 'http://[::1]:8931/', 'http://localhost']) assert.equal(isMixedContent('https://localhost:8090', u), false, u);
  assert.equal(isMixedContent('https://chat.example.com', 'http://localhost.example.com/mcp'), true);

  const base = { url: 'http://192.0.2.7:8931/mcp', origin: 'https://chat.example.com' };
  const on = connectFailureText({ ...base, proxyEnabled: true, probe: { ok: true } });
  assert.match(on, /open over HTTPS and http:\/\/192\.0\.2\.7:8931\/mcp is plain HTTP/);
  assert.match(on, /Through gaudium-server's proxy it answers\./);
  assert.match(on, /Tick "Use the server proxy"/);
  assert.doesNotMatch(on, /allowed origins/);
  const off = connectFailureText({ ...base, proxyEnabled: false });
  assert.match(off, /Start gaudium-server with --mcp-proxy/);
  const refused = connectFailureText({ ...base, proxyEnabled: true, probe: { proxyError: true, message: 'target 192.0.2.7:8931 is not in the allow-list' } });
  assert.match(refused, /proxy did not try it: target 192\.0\.2\.7:8931 is not in the allow-list\./);
  const down = connectFailureText({ ...base, proxyEnabled: true, probe: { unreachable: true, message: 'cannot connect to 192.0.2.7:8931' } });
  assert.match(down, /proxy cannot reach it \(cannot connect to 192\.0\.2\.7:8931\)/);
});

test('a same-scheme failure keeps the list of likely causes', () => {
  const t = connectFailureText({ url: 'http://192.0.2.7:8931/mcp', origin: 'http://192.0.2.5:8090', proxyEnabled: true, probe: { status: 403, message: 'Forbidden' } });
  assert.match(t, /Likely causes, most likely first/);
  assert.match(t, /Through gaudium-server's proxy it answers HTTP 403: Forbidden\./);
  assert.match(connectFailureText({ url: 'http://x/', origin: 'http://y', proxyEnabled: true, viaProxy: true }), /could not reach gaudium-server's proxy/);
});
