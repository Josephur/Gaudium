# Slice 8 — Verified HTTPS Listener Core
# Formally specifies and proves the decisions of the server's HTTPS listener:
#   - Whether the server serves HTTPS (a certificate and a key were given, the TLS library was
#     found, the files were accepted), refuses its configuration, or serves plain HTTP only
#   - Which protocol one accepted connection speaks. HTTP and HTTPS share the port: the first
#     byte a client sends says which of the two it is
#   - What a request gets: an answer, or (plain HTTP on a server that serves TLS) a redirect to
#     HTTPS; and which connections carry the Strict-Transport-Security header

import Base

# ============================================================================
# 1. Starting the Listener
# ============================================================================

type LsStart is Data:
  LsStOff{}
  LsStHalf{}
  LsStNoLib{}
  LsStBadFiles{}
  LsStOn{}

# cert, key: the certificate file and the key file were given
# lib_ok:    the TLS library was found and has what is needed
# files_ok:  the library accepted the two files (they parse and belong together)
def ls_start(cert: Bool, key: Bool, lib_ok: Bool, files_ok: Bool) -> LsStart:
  match cert key:
    case False{} False{}:
      LsStOff{}
    case True{} False{}:
      LsStHalf{}
    case False{} True{}:
      LsStHalf{}
    case True{} True{}:
      match lib_ok:
        case False{}:
          LsStNoLib{}
        case True{}:
          match files_ok:
            case False{}:
              LsStBadFiles{}
            case True{}:
              LsStOn{}

# The server answers TLS connections.
def ls_start_serves_tls(s: LsStart) -> Bool:
  match s:
    case LsStOn{}:
      True{}
    case _:
      False{}

# The configuration is refused (the server does not start with it): one file without the other,
# or files the library does not accept.
def ls_start_refuses(s: LsStart) -> Bool:
  match s:
    case LsStHalf{}:
      True{}
    case LsStBadFiles{}:
      True{}
    case _:
      False{}

# The server starts and says that HTTPS was asked for and is not offered (no TLS library on this
# machine).
def ls_start_warns(s: LsStart) -> Bool:
  match s:
    case LsStNoLib{}:
      True{}
    case _:
      False{}

# LAW 1.1: Without the two files the listener is plain HTTP, whatever the machine has.
law ls_start_off_without_files:
  for l: Bool
  for f: Bool
  {ls_start(False{}, False{}, l, f) == LsStOff{} : LsStart}

def ls_start_off_without_files(l, f):
  {==}

# LAW 1.2: TLS is never served without the library.
law ls_start_needs_library:
  for +c: Bool
  for +k: Bool
  for f: Bool
  {ls_start_serves_tls(ls_start(c, k, False{}, f)) == False{} : Bool}

def ls_start_needs_library(c, k, f):
  match c k:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# LAW 1.3: TLS is never served with files the library did not accept.
law ls_start_needs_good_files:
  for +c: Bool
  for +k: Bool
  for +l: Bool
  {ls_start_serves_tls(ls_start(c, k, l, False{})) == False{} : Bool}

def ls_start_needs_good_files(c, k, l):
  match c k l:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# LAW 1.4: A certificate without its key, or a key without its certificate, never serves TLS.
law ls_start_cert_alone_never_serves:
  for l: Bool
  for f: Bool
  {ls_start_serves_tls(ls_start(True{}, False{}, l, f)) == False{} : Bool}

def ls_start_cert_alone_never_serves(l, f):
  {==}

law ls_start_key_alone_never_serves:
  for l: Bool
  for f: Bool
  {ls_start_serves_tls(ls_start(False{}, True{}, l, f)) == False{} : Bool}

def ls_start_key_alone_never_serves(l, f):
  {==}

# LAW 1.5: HTTPS that was asked for and is not served is never silent: the configuration is
# refused or the server says so.
law ls_start_asked_never_silent:
  for +l: Bool
  for +f: Bool
  {Bool.or(ls_start_serves_tls(ls_start(True{}, True{}, l, f)), Bool.or(ls_start_refuses(ls_start(True{}, True{}, l, f)), ls_start_warns(ls_start(True{}, True{}, l, f)))) == True{} : Bool}

def ls_start_asked_never_silent(l, f):
  match l f:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# LAW 1.6: A configuration is never both served and refused.
law ls_start_serve_xor_refuse:
  for +s: LsStart
  {Bool.and(ls_start_serves_tls(s), ls_start_refuses(s)) == False{} : Bool}

def ls_start_serve_xor_refuse(s):
  match s:
    case LsStOff{}: {==}
    case LsStHalf{}: {==}
    case LsStNoLib{}: {==}
    case LsStBadFiles{}: {==}
    case LsStOn{}: {==}

# ============================================================================
# 2. The Protocol of One Connection
# ============================================================================

type LsProto is Data:
  LsPlain{}
  LsTls{}
  LsDrop{}

# A TLS connection opens with a handshake record, whose first byte is 22; no HTTP request line
# starts with that byte.
def ls_is_hello(first_byte: Nat) -> Bool:
  Nat.is_eq(first_byte, 22n)

# hello:     ls_is_hello of the first byte the client sent
# tls_ready: the server serves TLS
def ls_proto(hello: Bool, tls_ready: Bool) -> LsProto:
  match hello:
    case False{}:
      LsPlain{}
    case True{}:
      match tls_ready:
        case True{}:
          LsTls{}
        case False{}:
          LsDrop{}

def ls_proto_is_tls(p: LsProto) -> Bool:
  match p:
    case LsTls{}:
      True{}
    case _:
      False{}

def ls_proto_is_plain(p: LsProto) -> Bool:
  match p:
    case LsPlain{}:
      True{}
    case _:
      False{}

# LAW 2.1: A handshake record is recognised.
law ls_hello_byte_recognised:
  {ls_is_hello(22n) == True{} : Bool}

def ls_hello_byte_recognised():
  {==}

# LAW 2.2: A TLS handshake is never handed to the HTTP parser.
law ls_hello_never_plain:
  for +t: Bool
  {ls_proto_is_plain(ls_proto(True{}, t)) == False{} : Bool}

def ls_hello_never_plain(t):
  match t:
    case True{}: {==}
    case False{}: {==}

# LAW 2.3: A server that does not serve TLS never starts a handshake.
law ls_no_tls_without_listener:
  for +h: Bool
  {ls_proto_is_tls(ls_proto(h, False{})) == False{} : Bool}

def ls_no_tls_without_listener(h):
  match h:
    case True{}: {==}
    case False{}: {==}

# LAW 2.4: What is not a handshake is never answered with one.
law ls_plain_never_handshakes:
  for t: Bool
  {ls_proto_is_tls(ls_proto(False{}, t)) == False{} : Bool}

def ls_plain_never_handshakes(t):
  {==}

# ============================================================================
# 3. Plain HTTP Beside HTTPS
# ============================================================================

# What --tls-http asks for plain-HTTP requests while the server serves TLS.
type LsHttpMode is Data:
  LsHttpRedirect{}
  LsHttpAllow{}
  LsHttpStrict{}

# LsHttpRedirect: clients on other machines are sent to HTTPS; clients on this machine's
#                 loopback interface may be answered over plain HTTP (their traffic stays on the
#                 machine)
# LsHttpAllow:    every client is answered over plain HTTP as well
# LsHttpStrict:   every client is sent to HTTPS, loopback clients included

type LsPlainAct is Data:
  LsPlainServe{}
  LsPlainRedirect{}

# What a request on a connection of protocol p gets. A TLS connection is always answered; the
# rest is about plain HTTP.
# p:         the connection's protocol (ls_proto)
# tls_ready: the server serves TLS (ls_start_serves_tls)
# mode:      --tls-http
# loopback:  the client connected over this machine's loopback interface
def ls_plain_act(p: LsProto, tls_ready: Bool, mode: LsHttpMode, loopback: Bool) -> LsPlainAct:
  match p:
    case LsTls{}:
      LsPlainServe{}
    case _:
      match tls_ready:
        case False{}:
          LsPlainServe{}
        case True{}:
          match mode:
            case LsHttpAllow{}:
              LsPlainServe{}
            case LsHttpStrict{}:
              LsPlainRedirect{}
            case LsHttpRedirect{}:
              match loopback:
                case True{}:
                  LsPlainServe{}
                case False{}:
                  LsPlainRedirect{}

# The request is answered (not redirected).
def ls_plain_served(a: LsPlainAct) -> Bool:
  match a:
    case LsPlainServe{}:
      True{}
    case LsPlainRedirect{}:
      False{}

# The response carries Strict-Transport-Security. Only a TLS connection may: a browser ignores the
# header over plain HTTP, and a plain answer that set it could be forged by anyone on the path.
def ls_sends_hsts(p: LsProto) -> Bool:
  match p:
    case LsTls{}:
      True{}
    case _:
      False{}

# LAW 3.1
# promise: a server that does not serve TLS answers every plain-HTTP request, whatever --tls-http says and wherever the client is
# harm: with HTTPS asked for and not offered (no TLS library), every client would be sent to a port that does not speak TLS and could reach nothing
# catches: the mode looked at before whether TLS is served (strict mode redirects on a server without TLS)
law ls_plain_served_without_tls:
  for +p: LsProto
  for m: LsHttpMode
  for l: Bool
  {ls_plain_served(ls_plain_act(p, False{}, m, l)) == True{} : Bool}

def ls_plain_served_without_tls(p, m, l):
  match p:
    case LsPlain{}: {==}
    case LsTls{}: {==}
    case LsDrop{}: {==}

# LAW 3.2
# promise: with TLS served and plain HTTP not allowed, a client on another machine is never answered over plain HTTP
# harm: prompts, answers and the API key would cross the network in the clear although HTTPS is offered
# catches: the loopback test inverted (remote clients answered in the clear, loopback clients redirected)
law ls_plain_remote_redirected:
  {ls_plain_served(ls_plain_act(LsPlain{}, True{}, LsHttpRedirect{}, False{})) == False{} : Bool}

def ls_plain_remote_redirected():
  {==}

# LAW 3.3
# promise: in strict mode no plain-HTTP request is answered while TLS is served, loopback clients included
# catches: the loopback exemption applied in strict mode as well
law ls_plain_strict_redirects_all:
  for +l: Bool
  {ls_plain_served(ls_plain_act(LsPlain{}, True{}, LsHttpStrict{}, l)) == False{} : Bool}

def ls_plain_strict_redirects_all(l):
  match l:
    case True{}: {==}
    case False{}: {==}

# LAW 3.4
# promise: when plain HTTP is allowed beside TLS, every client is answered over plain HTTP
# harm: local tooling that cannot speak HTTPS, run from another machine, would stop working although the opt-out was given
# catches: the allow mode treated as the default mode (clients on other machines still redirected)
law ls_plain_allow_serves_all:
  for +l: Bool
  {ls_plain_served(ls_plain_act(LsPlain{}, True{}, LsHttpAllow{}, l)) == True{} : Bool}

def ls_plain_allow_serves_all(l):
  match l:
    case True{}: {==}
    case False{}: {==}

# LAW 3.5
# promise: a request that came over TLS is always answered, never redirected, whatever --tls-http says and wherever the client is
# harm: an HTTPS client would be sent to the address it already uses, again and again, and reach nothing
# catches: the connection's protocol ignored (strict mode redirects HTTPS requests too)
law ls_tls_request_served:
  for t: Bool
  for m: LsHttpMode
  for l: Bool
  {ls_plain_served(ls_plain_act(LsTls{}, t, m, l)) == True{} : Bool}

def ls_tls_request_served(t, m, l):
  {==}

# LAW 3.6
# promise: Strict-Transport-Security is never sent on a connection that is not TLS
# catches: the header added to every response, plain HTTP included
law ls_hsts_only_over_tls:
  {Bool.or(ls_sends_hsts(LsPlain{}), ls_sends_hsts(LsDrop{})) == False{} : Bool}

def ls_hsts_only_over_tls():
  {==}

# LAW 3.7
# promise: every response sent over TLS carries Strict-Transport-Security
# harm: a browser that once reached the server over HTTPS would go on trying plain HTTP first, where the first request can be read or redirected by anyone on the path
# catches: the header sent on no connection at all (the TLS case answered False)
law ls_hsts_on_tls:
  {ls_sends_hsts(LsTls{}) == True{} : Bool}

def ls_hsts_on_tls():
  {==}
