# The MCP proxy and HTTPS

The Web UI's MCP client runs in the browser: the page connects to each MCP server, lists its
tools and runs the calls the model makes. Three things can keep a browser from an MCP server:

- the server sends no CORS headers, so the browser hides its answers;
- only the Gaudium machine reaches it (a server on that machine's loopback interface, or on a
  network the browser is not on);
- the page is open over HTTPS and the server is plain HTTP: a browser does not let an HTTPS
  page call a plain-HTTP address.

For these, the Gaudium server offers a proxy: the page sends the request to its own server
(`/cors-proxy`), and the server makes it. This page describes the proxy, its guards, and the
HTTPS listener that lets the Web UI be opened over `https://`.

Sources: `gaudium/src/server/proxy.rs`, `gaudium/src/server/conn.rs`, `gaudium/src/tls.rs`;
the decisions are in `gaudium/verified/mcp_proxy.bend` and `gaudium/verified/tls_listener.bend`.

## Turning it on

The proxy is off by default. `--mcp-proxy` turns it on, and it needs the targets it may reach:

| MCP server | Flags |
|---|---|
| on the Gaudium machine, port 8931 | `--mcp-proxy --mcp-proxy-allow-private localhost:8931` |
| on the local network | `--mcp-proxy --mcp-proxy-allow-private 192.168.0.0/16` (or a host: `nas.example.com:8931`) |
| public, by name | `--mcp-proxy --mcp-proxy-allow mcp.example.com,*.tools.example.com` |
| any public server | `--mcp-proxy --mcp-proxy-allow '*' --api-key <key>` |

Both list options take comma-separated entries and may be repeated. An entry is a host
(`mcp.example.com`), a host and port (`mcp.example.com:8931`), every name under a domain
(`*.example.com`), an address (`192.0.2.7`, `[2001:db8::1]:8931`) or an address range
(`192.0.2.0/24`, `fd00::/8`). `*` (any target) is accepted in `--mcp-proxy-allow` only.

- `--mcp-proxy-allow` names targets whose addresses must be public.
- `--mcp-proxy-allow-private` names targets that may be reached although their address is
  loopback, private or link-local. An entry here also allows the target, so a local MCP server
  needs this option only. `*` and a `/0` range are not accepted here: private targets are named.

In the Web UI, tick "Use the server proxy" for an MCP server (Settings, Tools). `/props` reports
`cors_proxy_enabled`, and the page offers the checkbox only when it is true.

Notes on names and addresses:

- A name is judged by every address it resolves to. `localhost` resolves to `127.0.0.1` and
  `::1`: list it by name (`localhost` or `localhost:8931`), or write `127.0.0.1` in the URL and
  in the list. Listing only `127.0.0.1` and using `http://localhost:8931/` in the URL is refused,
  because `::1` is then outside the list.
- A name that no host entry matches is still looked up when the lists hold an address range: it
  is allowed when every address it resolves to lies inside the ranges. Without a range entry an
  unlisted name is refused before any lookup.

## What the proxy does with a request

`GET | POST | DELETE /cors-proxy?url=<absolute http(s) URL>` (GET is the event stream of
Streamable HTTP, DELETE ends a session).

- Only request headers named `x-gaudium-proxy-header-<name>` go to the target, as `<name>`.
  Everything else of the request stays with the server: its own `Authorization` / `X-Api-Key`
  (the server's API key) and its cookies are never forwarded.
- The target's status, headers and body come back as they arrive; an event stream is passed on
  event by event, not collected. `Mcp-Session-Id` reaches the page.
- Redirects are followed by the proxy itself, each through every check below. A relative
  `Location` is resolved against the whole URL of the request that got it (RFC 3986,
  section 5): `?query` keeps the path, `#fragment` keeps path and query, `.` and `..` segments
  are removed.
- A request the proxy refuses is answered with a JSON error whose message starts with
  `mcp proxy: `; a target that could not be reached, or did not answer properly, with one that
  starts with `proxy error: ` (status 502).
- `GET /gaudium/proxy` (behind the API key) shows the settings, counters (proxied, refused,
  failed, and `stalled`: answers ended because the page stopped reading them) and the last 50
  decisions. Every decision is also a `[mcp-proxy]` line in the server's log. URLs are logged
  and shown without their query and fragment, which may hold a credential, and without a user
  part (`user:password@`), also when the URL is refused or malformed.

## Threat model and guards

A route that fetches a URL named by the request is a forwarder inside the operator's network.
Without guards, whoever reaches the server's port could make it request services that only the
server's machine reaches (other hosts of the network, loopback-only ports, the instance-metadata
endpoints of hosted machines) and read the answers. Name lookup adds a trap: a name that
resolves to a public address when it is checked and to `127.0.0.1` when it is connected to.

| Guard | How |
|---|---|
| Off by default | `--mcp-proxy`; otherwise 403 `feature_disabled` and `cors_proxy_enabled: false` |
| Targets are named | no list: the server does not start with `--mcp-proxy`. A list that allows any target (`*`, a `/0` range) needs `--api-key` |
| API key | when the server has one, the route needs it like every API route |
| Not for other sites' pages | a request another site's page sent is refused (see below) |
| URL form | `http` and `https` only; no user name or password in the URL; a valid host and port; no control characters |
| Addresses checked after lookup | every address of the target is classed: public, loopback, private (RFC 1918, shared address space, unique-local, documentation and benchmarking ranges), link-local (with the metadata endpoints), or never (unspecified, multicast, reserved). Non-public needs `--mcp-proxy-allow-private`; "never" is refused even when listed. An IPv6 address that carries an IPv4 one (IPv4-mapped, the `64:ff9b::/96` translation prefix, 6to4) is classed by that IPv4 address |
| One refused address refuses the name | a name with a public and a private address is refused as a whole |
| No second lookup | the connection goes to an address that was checked; `Host` and the TLS server name stay the name |
| Redirects | followed by the proxy, at most 3, each new target through all checks. A redirect to another origin takes no header along that may carry a credential (only content negotiation), nor does any later hop of the chain, even one back to the origin the page named; a POST becomes a GET without its body on 301 / 302 / 303 |
| Limits | request body 8 MiB (413); response `--mcp-proxy-max-mb` (default 64 MiB): a longer one is refused when its length is declared, cut otherwise; connect 10 s; silence of the target `--mcp-proxy-timeout` (default 600 s); one exchange 1 h at most. A write to the page waits no longer than what is left of these: a page that takes nothing of the answer for `--mcp-proxy-timeout`, or an exchange past 1 h, ends the exchange, closes the connection to the target and counts as `stalled` |
| Headers | to the target: only the prefixed ones, without the ones that frame the connection (`Host`, `Content-Length`, `Connection`, ...). To the page: not the target's cookies, cross-origin grants, redirect target or page policies; the answer carries `Content-Security-Policy: sandbox` and `X-Content-Type-Options: nosniff`, so a target's HTML opened from the proxy's address runs without the Gaudium page's rights |
| HTTPS targets | verified against the system's trust store (plus `--mcp-proxy-ca <file>`), with the host name checked; without the TLS library they are refused, never fetched in the clear |

**Another site's page.** A page of any site, open in the same browser as the Gaudium Web UI,
can make the browser send requests to the Gaudium server. The proxy refuses them without
knowing its own public name: it looks only at the request. A browser states where a request
comes from in `Sec-Fetch-Site` (page script cannot set it): `same-origin` and `none` (typed by
the person) pass, every other value is refused. Browsers send this header to HTTPS pages'
servers and to `localhost`. Without it, the request's `Origin`, when there is one, must name the
host the request was sent to: its own `Host`, or the host a front end names in
`X-Forwarded-Host` / `Forwarded: host=`. A client that is not a browser sends neither header
and is not affected.

**Not covered.** An allowed MCP server that is itself hostile (its tools run in the user's
browser, with the page's permission prompts); whoever holds the API key; and, on a server
without an API key opened over plain HTTP from another machine, a request that a page of another
site makes the browser send without an `Origin` (a plain GET, whose answer that page cannot
read): it reaches the listed targets like any request from the network does. Set `--api-key` to
close that.

## HTTPS

```
gaudium --serve ... --tls-cert /path/to/cert.pem --tls-key /path/to/key.pem
```

`--tls-cert` is the server's PEM certificate with its chain, `--tls-key` its unencrypted PEM
private key. HTTP and HTTPS then share the port: `http://host:port` and `https://host:port`
both answer. The first byte of a connection says which protocol the client speaks (a TLS
handshake record, or the first letter of an HTTP request), so existing clients and saved URLs go
on working and no second port has to be opened. Without the two options the server is plain HTTP
as before.

TLS comes from the system's OpenSSL 3 (`libssl.so.3`), loaded when it is asked for; the engine
links no TLS library. On a machine without it the server starts, says that HTTPS is not
offered, and serves plain HTTP; HTTPS targets of the proxy are refused with a message that names
the missing library. A certificate without its key, or files the library does not accept, stop
the start with the reason. The key file is read by the library: the server keeps its path only,
and neither logs nor `/props` nor error messages hold key material.

The certificate decides under which names a browser accepts the server: `localhost` for the same
machine, the machine's name or address on the local network, a public DNS name. Getting one is
not part of the server (a certificate from a public authority for a public name; a self-signed
one or one from a private authority for local use).

### Behind a front end that ends TLS

A reverse proxy, a tunnel or a forwarded router port may end TLS and pass plain HTTP on. Give
neither TLS option then. Nothing in the server is configured with the public name: the Web UI
uses relative paths only, the server builds no URL from a name, no check pins `Host`, and the
proxy's site rule compares the request with itself (above).

A front end usually cuts a connection that carries nothing for a while (60 to 100 s are common
limits). The server's own event stream sends a comment line every 10 s; the proxy sends one
(`: keep-alive`) every 20 s on a target's event stream that is idle. A comment is written only
where a line starts, so it never lands inside a line of the target's stream. A target that takes
longer than the front end's limit to send the head of its answer (a tool call that answers as
plain JSON after minutes) cannot be helped this way: use an MCP server that answers such calls as
an event stream, or raise the front end's limit.
