# Verified API-Key Route Gate
# Formally specifies and proves which HTTP requests the server answers when it was started with an
# API key:
#   - What kind of route a request reaches, from what the shell's route table found for it, whether
#     its path is one the API owns, and whether it only reads
#   - Whether the request is answered or refused (401) for that kind, whether a key is configured,
#     and whether the request carries it
#
# Closed by default: a request the route table does not list needs the key, unless it is a read
# outside the API's paths (a Web UI file). Only the Web UI's files and the liveness probe are
# served without the key; every route that reads server state, runs inference, or operates the
# server needs it.

import Base

# ============================================================================
# 1. Route Kinds
# ============================================================================

type RouteKind is Data:
  RouteUiAsset{}
  RouteLiveness{}
  RouteStatus{}
  RouteInference{}
  RouteAdmin{}
  RouteUnlisted{}

# RouteUiAsset:   a file of the Web UI (the browser fetches them without the key)
# RouteLiveness:  the liveness probe (`GET /health`)
# RouteStatus:    reads server, model or engine state
# RouteInference: takes or returns text or tokens (completions, messages, tokenization, templates)
# RouteAdmin:     operates the server (loads and unloads models, parks the engine, stops replies,
#                 forwards to MCP servers)
# RouteUnlisted:  the route table lists no route for the request

# The kind of route a request reaches.
# listed:   the kind the shell's route table gives the request's method and path (RouteUnlisted
#           when the table lists none)
# api_path: the path is one the API owns (its namespaces, or a path the table lists under another
#           method)
# read:     the method only reads (GET or HEAD)
def route_kind_of(listed: RouteKind, api_path: Bool, read: Bool) -> RouteKind:
  match listed:
    case RouteUnlisted{}:
      match api_path:
        case True{}:
          RouteUnlisted{}
        case False{}:
          match read:
            case True{}:
              RouteUiAsset{}
            case False{}:
              RouteUnlisted{}
    case _:
      listed

# A route of this kind is served without the key.
def route_kind_public(k: RouteKind) -> Bool:
  match k:
    case RouteUiAsset{}:
      True{}
    case RouteLiveness{}:
      True{}
    case _:
      False{}

# ============================================================================
# 2. The Key Gate
# ============================================================================

# The request is answered (False: it is refused with 401).
# listed, api_path, read: as for route_kind_of
# key_set: the server was started with an API key
# key_ok:  the request carries the configured key (true when none is configured)
def route_admit(listed: RouteKind, api_path: Bool, read: Bool, key_set: Bool, key_ok: Bool) -> Bool:
  match key_set:
    case False{}:
      True{}
    case True{}:
      match key_ok:
        case True{}:
          True{}
        case False{}:
          route_kind_public(route_kind_of(listed, api_path, read))

# LAW 1
# promise: with an API key configured, an inference route is never answered without it
# harm: anyone who can reach the port could run prompts on the card and read what it writes
# catches: the inference kind counted among the public kinds
law route_inference_needs_key:
  for a: Bool
  for r: Bool
  {route_admit(RouteInference{}, a, r, True{}, False{}) == False{} : Bool}

def route_inference_needs_key(a, r):
  {==}

# LAW 2
# promise: with an API key configured, a route that operates the server is never answered without it
# harm: anyone who can reach the port could unload the model, park the engine, stop other clients' replies or use the server to reach MCP servers
# catches: the admin kind counted among the public kinds
law route_admin_needs_key:
  for a: Bool
  for r: Bool
  {route_admit(RouteAdmin{}, a, r, True{}, False{}) == False{} : Bool}

def route_admin_needs_key(a, r):
  {==}

# LAW 3
# promise: with an API key configured, a route that reads server state is never answered without it
# harm: recent requests, model names and engine state would be visible to anyone who can reach the port
# catches: every read-only route served without the key (status counted among the public kinds)
law route_status_needs_key:
  for a: Bool
  for r: Bool
  {route_admit(RouteStatus{}, a, r, True{}, False{}) == False{} : Bool}

def route_status_needs_key(a, r):
  {==}

# LAW 4
# promise: with an API key configured, a request on a path the API owns that the route table does not list is never answered without it
# harm: a route added to the router and forgotten in the route table would be served without the key
# catches: every unlisted read served as a Web UI file, the API's paths not looked at
law route_unlisted_api_path_needs_key:
  for r: Bool
  {route_admit(RouteUnlisted{}, True{}, r, True{}, False{}) == False{} : Bool}

def route_unlisted_api_path_needs_key(r):
  {==}

# LAW 5
# promise: with an API key configured, a request that is not a read and that the route table does not list is never answered without it, whatever its path
# harm: a new POST route under a path outside the API's namespaces, forgotten in the route table, would run without the key
# catches: unlisted requests open by default (an unlisted request outside the API's paths served whatever its method)
law route_unlisted_write_needs_key:
  for +a: Bool
  {route_admit(RouteUnlisted{}, a, False{}, True{}, False{}) == False{} : Bool}

def route_unlisted_write_needs_key(a):
  match a:
    case True{}: {==}
    case False{}: {==}

# LAW 6
# promise: a request that carries the configured API key is never refused by the key gate, whatever route it reaches
# harm: clients holding the key would be locked out of the routes the key exists to open
# catches: the request's key result ignored (every non-public route refused while a key is configured)
law route_valid_key_admits:
  for k: RouteKind
  for a: Bool
  for r: Bool
  {route_admit(k, a, r, True{}, True{}) == True{} : Bool}

def route_valid_key_admits(k, a, r):
  {==}

# LAW 7
# promise: a server started without an API key answers every request, whatever route it reaches and whatever the request carries
# harm: a server run without a key would refuse its own Web UI and every client
# catches: whether a key is configured ignored (a request without a key refused on a server that has none)
law route_no_key_configured_admits:
  for k: RouteKind
  for a: Bool
  for r: Bool
  for v: Bool
  {route_admit(k, a, r, False{}, v) == True{} : Bool}

def route_no_key_configured_admits(k, a, r, v):
  {==}
