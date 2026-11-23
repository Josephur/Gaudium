# Slice 8 — Verified HTTPS Listener Core
# Formally specifies and proves the decisions of the server's HTTPS listener:
#   - Whether the server serves HTTPS (a certificate and a key were given, the TLS library was
#     found, the files were accepted), refuses its configuration, or serves plain HTTP only
#   - Which protocol one accepted connection speaks. HTTP and HTTPS share the port: the first
#     byte a client sends says which of the two it is

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
