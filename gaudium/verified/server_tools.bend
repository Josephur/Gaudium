# Server-side tools of a request: the tools the server runs itself while it answers (the
# Messages MCP connector first; hosted tools and the Responses `mcp` tools use the same loop).
# Formally specifies and proves:
#   - Whether a request that names tool servers runs, or is refused (the MCP proxy is off, too
#     many servers)
#   - Where a call the model makes goes: back to the client, to the server's own runner, or a
#     name of both (the request is refused before it runs)
#   - The tool loop: after a generation round, whether the server runs the round's calls and
#     whether it generates again, within a bound on the rounds
#   - The bounds on the calls run per request, on the size of one tool result put back into the
#     prompt, and on the pages of a server's tool list
#   - The request's output-token limit over the rounds: what each round may generate, and when
#     the reply stops because the limit is spent

import Base
import nat_lemmas.bend as L

# ============================================================================
# 1. Admission
# ============================================================================

type SvAdmit is Data:
  SvNone{}
  SvRun{}
  SvProxyOff{}
  SvForeign{}
  SvTooMany{}

# The request names at most `most` servers.
def sv_admit_fits(fits: Bool) -> SvAdmit:
  match fits:
    case True{}:
      SvRun{}
    case False{}:
      SvTooMany{}

# asked:    tool servers the request names
# proxy_on: the server offers the MCP proxy (its outbound connections go through the proxy's
#           allow rules, so without it there is nothing they may reach)
# site_ok:  px_site_ok of the request that names the servers (a page of another site, open in a
#           browser that can reach the server, may not drive the proxy through this route either)
# most:     servers one request may name
def sv_admit(asked: Nat, proxy_on: Bool, site_ok: Bool, most: Nat) -> SvAdmit:
  match asked:
    case 0n:
      SvNone{}
    case 1n+ap:
      match proxy_on:
        case False{}:
          SvProxyOff{}
        case True{}:
          match site_ok:
            case False{}:
              SvForeign{}
            case True{}:
              sv_admit_fits(Nat.is_le(1n+ap, most))

# The request runs with its tool servers.
def sv_admit_runs(a: SvAdmit) -> Bool:
  match a:
    case SvRun{}:
      True{}
    case _:
      False{}

# The request names no tool server (it runs as any other request).
def sv_admit_is_none(a: SvAdmit) -> Bool:
  match a:
    case SvNone{}:
      True{}
    case _:
      False{}

def sv_fits_past(+m: Nat, +d: Nat) -> {sv_admit_runs(sv_admit_fits(Nat.is_le(Nat.add(m, 1n+d), m))) == False{} : Bool}:
  match m:
    case 0n: {==}
    case 1n+mp:
      sv_fits_past(mp, d)

def sv_fits_within(+n: Nat, +d: Nat) -> {sv_admit_runs(sv_admit_fits(Nat.is_le(n, Nat.add(n, d)))) == True{} : Bool}:
  match n:
    case 0n:
      match d:
        case 0n: {==}
        case 1n+dp: {==}
    case 1n+np:
      sv_fits_within(np, d)

def sv_fits_not_none(+b: Bool) -> {sv_admit_is_none(sv_admit_fits(b)) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# LAW 1.1: Without the MCP proxy no tool server is ever reached.
# promise: a request that names tool servers on a server without the MCP proxy never runs them
# harm: the server would connect to client-named addresses with none of the proxy's allow rules
# catches: the proxy check dropped (sv_admit answers SvRun whenever the count fits)
law sv_proxy_off_never_runs:
  for +n: Nat
  for +s: Bool
  for +m: Nat
  {sv_admit_runs(sv_admit(n, False{}, s, m)) == False{} : Bool}

def sv_proxy_off_never_runs(n, s, m):
  match n:
    case 0n: {==}
    case 1n+np: {==}

# LAW 1.2: A page of another site never drives the tool servers.
# promise: a request that names tool servers and comes from a page of another site never runs them
# harm: any web page open in a browser that can reach the server could make it call allow-listed
#       (even private) MCP servers and read their tool lists and results
# catches: the site input ignored (sv_admit answers sv_admit_fits whenever the proxy is on)
law sv_foreign_never_runs:
  for +n: Nat
  for +p: Bool
  for +m: Nat
  {sv_admit_runs(sv_admit(n, p, False{}, m)) == False{} : Bool}

def sv_foreign_never_runs(n, p, m):
  match n:
    case 0n: {==}
    case 1n+np:
      match p:
        case False{}: {==}
        case True{}: {==}

# LAW 1.3: A request that names more servers than allowed never runs them.
# promise: a request naming more than `most` tool servers is never run
# harm: one request could fan out to any number of outbound connections
# catches: the count check off by one (Nat.is_le(ap, most) in place of Nat.is_le(1n+ap, most))
law sv_over_limit_never_runs:
  for +m: Nat
  for +d: Nat
  for +p: Bool
  for +s: Bool
  {sv_admit_runs(sv_admit(Nat.add(m, 1n+d), p, s, m)) == False{} : Bool}

def sv_over_limit_never_runs(m, d, p, s):
  match m:
    case 0n:
      match p s:
        case False{} False{}: {==}
        case False{} True{}: {==}
        case True{} False{}: {==}
        case True{} True{}: {==}
    case 1n+mp:
      match p s:
        case False{} False{}: {==}
        case False{} True{}: {==}
        case True{} False{}: {==}
        case True{} True{}:
          sv_fits_past(1n+mp, d)

# LAW 1.4: Named servers are never silently dropped: the request runs with them or is refused.
# promise: a request that names at least one tool server is never treated as naming none
# catches: a refused request answered as SvNone (the servers ignored and the request run without them)
law sv_asked_never_dropped:
  for +n: Nat
  for +p: Bool
  for +s: Bool
  for +m: Nat
  {sv_admit_is_none(sv_admit(1n+n, p, s, m)) == False{} : Bool}

def sv_asked_never_dropped(n, p, s, m):
  match p s:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}:
      sv_fits_not_none(Nat.is_le(1n+n, m))

# LAW 1.5: With the proxy on, a request of the server's own site within the limit runs.
# promise: with the MCP proxy on, a request that passes the site rule and names between one and `most` servers runs them
# catches: sv_admit answers SvTooMany for every count (the connector never runs)
law sv_within_limit_runs:
  for +n: Nat
  for +d: Nat
  {sv_admit_runs(sv_admit(1n+n, True{}, True{}, Nat.add(1n+n, d))) == True{} : Bool}

def sv_within_limit_runs(n, d):
  sv_fits_within(1n+n, d)

# LAW 1.6: A request that names no tool server is never refused for it.
# promise: a request naming no tool server runs as any other request, whatever the proxy setting and wherever it comes from
# harm: every request without tool servers would be refused
# catches: the no-server case answers SvProxyOff (sv_admit refuses a request that asked for nothing)
law sv_unasked_runs_as_before:
  for +p: Bool
  for +s: Bool
  for +m: Nat
  {sv_admit_is_none(sv_admit(0n, p, s, m)) == True{} : Bool}

def sv_unasked_runs_as_before(p, s, m):
  {==}

# ============================================================================
# 2. Where a Call Goes
# ============================================================================

type SvRoute is Data:
  SvClient{}
  SvServer{}
  SvClash{}

# server_named: the name is one of the server-side tools of the request
# client_named: the name is one of the client's own tools
# A name of neither goes back to the client, as any call of an unknown tool does.
def sv_route(server_named: Bool, client_named: Bool) -> SvRoute:
  match server_named:
    case True{}:
      match client_named:
        case True{}:
          SvClash{}
        case False{}:
          SvServer{}
    case False{}:
      SvClient{}

# The server runs the call.
def sv_route_server(r: SvRoute) -> Bool:
  match r:
    case SvServer{}:
      True{}
    case _:
      False{}

# The name is both: the request is refused before it runs.
def sv_route_clash(r: SvRoute) -> Bool:
  match r:
    case SvClash{}:
      True{}
    case _:
      False{}

# LAW 2.1: A call of a client tool is never run by the server.
# promise: a call whose name is one of the client's tools is never run server-side
# harm: the client's call (and its arguments) would go to a remote tool server
# catches: sv_route answers SvServer for every server-side name, whatever client_named says
law sv_client_tool_never_server:
  for +s: Bool
  {sv_route_server(sv_route(s, True{})) == False{} : Bool}

def sv_client_tool_never_server(s):
  match s:
    case True{}: {==}
    case False{}: {==}

# LAW 2.2: A name of no server-side tool is never run by the server.
# promise: a call whose name is not one of the request's server-side tools is never run server-side
# catches: sv_route answers SvServer for every name the client does not own
law sv_unknown_never_server:
  for +c: Bool
  {sv_route_server(sv_route(False{}, c)) == False{} : Bool}

def sv_unknown_never_server(c):
  match c:
    case True{}: {==}
    case False{}: {==}

# LAW 2.3: A name of both kinds is refused, never resolved to one of them.
# promise: a tool name that is both a client tool and a server-side tool is a clash
# catches: the clash case answers SvClient (the server-side tool silently shadowed)
law sv_both_names_clash:
  {sv_route_clash(sv_route(True{}, True{})) == True{} : Bool}

def sv_both_names_clash():
  {==}

# LAW 2.4: A server-side tool the client does not also define runs on the server.
# promise: a call of a server-side tool that is not a client tool is run server-side
# catches: sv_route sends every call back to the client
law sv_server_tool_runs:
  {sv_route_server(sv_route(True{}, False{})) == True{} : Bool}

def sv_server_tool_runs():
  {==}

# LAW 2.5: Only a name of both kinds is a clash.
# promise: a name that is not one of the client's tools is never a clash
# harm: every request with tool servers would be refused as a clash
# catches: sv_route_clash answers True{} for every route
law sv_no_clash_without_client:
  for +s: Bool
  {sv_route_clash(sv_route(s, False{})) == False{} : Bool}

def sv_no_clash_without_client(s):
  match s:
    case True{}: {==}
    case False{}: {==}

# ============================================================================
# 3. The Tool Loop
# ============================================================================

type SvNext is Data:
  SvStop{}
  SvRunGoOn{}
  SvRunStop{}
  SvRunPause{}

# Rounds are left: generate again; none: stop, and the client may resume the turn.
def sv_next_room(room: Bool) -> SvNext:
  match room:
    case True{}:
      SvRunGoOn{}
    case False{}:
      SvRunPause{}

# After a generation round. ended: the round ended where the model ended it (end of sequence,
# the call limit), not cut by the length limit, a stop string or a cancel; server / client: the
# round's calls of server-side and of client tools; done: rounds generated so far, this one
# included; most: rounds one request may generate.
#   SvStop      the reply ends with the round's own stop reason
#   SvRunGoOn   the server runs its calls and generates again
#   SvRunStop   the server runs its calls and stops: client calls wait for the client
#   SvRunPause  the server runs its calls and stops: no round is left
def sv_next(ended: Bool, server: Nat, client: Nat, done: Nat, most: Nat) -> SvNext:
  match ended:
    case False{}:
      SvStop{}
    case True{}:
      match server:
        case 0n:
          SvStop{}
        case 1n+sp:
          match client:
            case 1n+cp:
              SvRunStop{}
            case 0n:
              sv_next_room(Nat.is_lt(done, most))

# The server runs the round's server-side calls.
def sv_runs(n: SvNext) -> Bool:
  match n:
    case SvStop{}:
      False{}
    case _:
      True{}

# The reply stops for the client's own calls (`tool_use`), not because the rounds ran out.
def sv_waits_client(n: SvNext) -> Bool:
  match n:
    case SvRunStop{}:
      True{}
    case _:
      False{}

# Another round is generated.
def sv_goes_on(n: SvNext) -> Bool:
  match n:
    case SvRunGoOn{}:
      True{}
    case _:
      False{}

def sv_room_past(+m: Nat, +d: Nat) -> {sv_goes_on(sv_next_room(Nat.is_lt(Nat.add(m, d), m))) == False{} : Bool}:
  match m d:
    case 0n 0n: {==}
    case 0n 1n+dp: {==}
    case 1n+mp _:
      sv_room_past(mp, d)

def sv_room_left(+n: Nat, +d: Nat) -> {sv_goes_on(sv_next_room(Nat.is_lt(n, Nat.add(n, 1n+d)))) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      sv_room_left(np, d)

def sv_room_waits(+b: Bool) -> {sv_waits_client(sv_next_room(b)) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def sv_room_runs(+b: Bool) -> {sv_runs(sv_next_room(b)) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

# LAW 3.1: No round is generated past the bound.
# promise: once `most` rounds were generated, the loop never generates another
# harm: a model that keeps calling tools would hold the request (and the card's turns) forever
# catches: sv_next_room ignores the room check and always answers SvRunGoOn
law sv_never_past_rounds:
  for +e: Bool
  for +s: Nat
  for +c: Nat
  for +m: Nat
  for +d: Nat
  {sv_goes_on(sv_next(e, s, c, Nat.add(m, d), m)) == False{} : Bool}

def sv_never_past_rounds(e, s, c, m, d):
  match e:
    case False{}: {==}
    case True{}:
      match s:
        case 0n: {==}
        case 1n+sp:
          match c:
            case 1n+cp: {==}
            case 0n:
              sv_room_past(m, d)

# LAW 3.2: A round with a client call never goes on: the reply stops for the client.
# promise: a round in which the model called a client tool never generates another round
# catches: the client-call branch answers sv_next_room (client calls left unanswered while generation goes on)
law sv_client_call_stops:
  for +e: Bool
  for +s: Nat
  for +c: Nat
  for +n: Nat
  for +m: Nat
  {sv_goes_on(sv_next(e, s, 1n+c, n, m)) == False{} : Bool}

def sv_client_call_stops(e, s, c, n, m):
  match e:
    case False{}: {==}
    case True{}:
      match s:
        case 0n: {==}
        case 1n+sp: {==}

# LAW 3.3: A round without server-side calls runs nothing and never goes on.
# promise: a round in which the model called no server-side tool neither runs tools nor generates again
# catches: the no-call branch answers sv_next_room (the loop regenerates with nothing new in the prompt)
law sv_no_server_call_never_runs:
  for +e: Bool
  for +c: Nat
  for +n: Nat
  for +m: Nat
  {sv_runs(sv_next(e, 0n, c, n, m)) == False{} : Bool}

def sv_no_server_call_never_runs(e, c, n, m):
  match e:
    case False{}: {==}
    case True{}: {==}

# LAW 3.4: A round the model did not end runs nothing.
# promise: a round cut by the length limit, a stop string or a cancel never runs its calls
# harm: a half-written call, or one the client cancelled, would reach a remote tool server
# catches: the ended flag ignored (sv_next starts at the server-call match)
law sv_cut_round_never_runs:
  for +s: Nat
  for +c: Nat
  for +n: Nat
  for +m: Nat
  {sv_runs(sv_next(False{}, s, c, n, m)) == False{} : Bool}

def sv_cut_round_never_runs(s, c, n, m):
  {==}

# LAW 3.5: The server-side calls of a round the model ended are always run, also when the reply
# stops there, so every server-side call in the reply has its result.
# promise: in a round the model ended, its server-side calls are run whether or not the loop goes on
# catches: the client-call branch answers SvStop (server calls of that round left without results)
law sv_server_calls_always_run:
  for +s: Nat
  for +c: Nat
  for +n: Nat
  for +m: Nat
  {sv_runs(sv_next(True{}, 1n+s, c, n, m)) == True{} : Bool}

def sv_server_calls_always_run(s, c, n, m):
  match c:
    case 0n:
      sv_room_runs(Nat.is_lt(n, m))
    case 1n+cp: {==}

# LAW 3.6: With rounds left, a round of server-side calls only goes on.
# promise: a round the model ended with server-side calls and no client call generates again while rounds are left
# catches: sv_next_room answers SvRunPause whatever the room (the loop never continues)
law sv_room_goes_on:
  for +s: Nat
  for +n: Nat
  for +d: Nat
  {sv_goes_on(sv_next(True{}, 1n+s, 0n, n, Nat.add(n, 1n+d))) == True{} : Bool}

def sv_room_goes_on(s, n, d):
  sv_room_left(n, d)

# LAW 3.7: A round the model ended with client calls stops for the client.
# promise: a round the model ended with client calls stops waiting for the client, also when it ran server-side calls
# harm: the client would be told to resume a turn whose own tool calls it never answered
# catches: the client-call branch answers SvRunPause (stopped as if the rounds ran out)
law sv_client_calls_wait:
  for +s: Nat
  for +c: Nat
  for +n: Nat
  for +m: Nat
  {sv_waits_client(sv_next(True{}, 1n+s, 1n+c, n, m)) == True{} : Bool}

def sv_client_calls_wait(s, c, n, m):
  {==}

# LAW 3.8: Without client calls the reply never waits for the client.
# promise: a round without client calls never stops as if client calls were waiting
# harm: the client would be asked for results of calls that are not in the reply
# catches: sv_next_room answers SvRunStop when no round is left
law sv_no_client_never_waits:
  for +e: Bool
  for +s: Nat
  for +n: Nat
  for +m: Nat
  {sv_waits_client(sv_next(e, s, 0n, n, m)) == False{} : Bool}

def sv_no_client_never_waits(e, s, n, m):
  match e:
    case False{}: {==}
    case True{}:
      match s:
        case 0n: {==}
        case 1n+sp:
          sv_room_waits(Nat.is_lt(n, m))

# ============================================================================
# 4. Bounds: Calls, Results, Tool-List Pages
# ============================================================================

# Calls of a round the server runs, with `done` run before in this request and `most` allowed
# per request; the rest are answered with an error result.
def sv_calls_take(done: Nat, asked: Nat, most: Nat) -> Nat:
  Nat.min(asked, Nat.sub(most, done))

def sv_min_add(+a: Nat, +r: Nat) -> {Nat.min(a, Nat.add(a, r)) == a : Nat}:
  match a:
    case 0n: {==}
    case 1n+ap:
      %sv_min_add(ap, r) : {1n+Nat.min(ap, Nat.add(ap, r)) == 1n+_ : Nat}
      {==}

# LAW 4.1: Never more calls than the round asked for.
# promise: the calls run in a round never exceed the calls the model made in it
# catches: sv_calls_take answers the room left (Nat.sub(most, done)) alone
law sv_take_within_asked:
  for +d: Nat
  for +a: Nat
  for +m: Nat
  {Nat.is_le(sv_calls_take(d, a, m), a) == True{} : Bool}

def sv_take_within_asked(d, a, m):
  L.nat_min_le_l(a, Nat.sub(m, d))

# LAW 4.2: Never more calls than are left of the request's allowance.
# promise: the calls run in a round never exceed what is left of the per-request call limit
# harm: one request could make any number of outbound calls
# catches: sv_calls_take answers `asked` alone
law sv_take_within_room:
  for +d: Nat
  for +a: Nat
  for +m: Nat
  {Nat.is_le(sv_calls_take(d, a, m), Nat.sub(m, d)) == True{} : Bool}

def sv_take_within_room(d, a, m):
  L.nat_min_le_r(a, Nat.sub(m, d))

# LAW 4.3: Calls that fit are all run.
# promise: when the round's calls fit in what is left of the limit, every one of them runs
# catches: sv_calls_take off by one (Nat.sub(Nat.sub(most, done), 1n) as the room)
law sv_take_all_that_fit:
  for +d: Nat
  for +a: Nat
  for +r: Nat
  {sv_calls_take(d, a, Nat.add(d, Nat.add(a, r))) == a : Nat}

def sv_take_all_that_fit(d, a, r):
  %Equal.sym(Nat, Nat.sub(Nat.add(d, Nat.add(a, r)), d), Nat.add(a, r), L.nat_add_sub_cancel_l(d, Nat.add(a, r))) : {Nat.min(a, _) == a : Nat}
  sv_min_add(a, r)

# Bytes of one tool result put back into the prompt.
def sv_result_keep(len: Nat, cap: Nat) -> Nat:
  Nat.min(len, cap)

# LAW 4.4: A result never goes into the prompt above the cap.
# promise: the bytes of a tool result kept for the prompt never exceed the cap
# harm: a large tool answer would fill the context, or the host's memory
# catches: sv_result_keep answers `len` alone
law sv_result_within_cap:
  for +l: Nat
  for +c: Nat
  {Nat.is_le(sv_result_keep(l, c), c) == True{} : Bool}

def sv_result_within_cap(l, c):
  L.nat_min_le_r(l, c)

# LAW 4.5: Never more bytes than the result has.
# promise: the bytes kept of a tool result never exceed its length
# catches: sv_result_keep answers the cap alone (a short result read past its end)
law sv_result_within_len:
  for +l: Nat
  for +c: Nat
  {Nat.is_le(sv_result_keep(l, c), l) == True{} : Bool}

def sv_result_within_len(l, c):
  L.nat_min_le_l(l, c)

# LAW 4.6: A result that fits is kept whole.
# promise: a tool result no longer than the cap is kept whole
# catches: sv_result_keep off by one (Nat.min(len, Nat.pred(cap)))
law sv_result_fits_whole:
  for +l: Nat
  for +d: Nat
  {sv_result_keep(l, Nat.add(l, d)) == l : Nat}

def sv_result_fits_whole(l, d):
  sv_min_add(l, d)

# Whether another page of a server's tool list is asked for: the server gave a cursor, and
# fewer than `most` pages were read.
def sv_list_more(cursor: Bool, pages: Nat, most: Nat) -> Bool:
  match cursor:
    case False{}:
      False{}
    case True{}:
      Nat.is_lt(pages, most)

def sv_lt_add_false(+m: Nat, +d: Nat) -> {Nat.is_lt(Nat.add(m, d), m) == False{} : Bool}:
  match m d:
    case 0n 0n: {==}
    case 0n 1n+dp: {==}
    case 1n+mp _:
      sv_lt_add_false(mp, d)

def sv_lt_add_succ(+n: Nat, +d: Nat) -> {Nat.is_lt(n, Nat.add(n, 1n+d)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+np:
      sv_lt_add_succ(np, d)

# LAW 4.7: Never more pages than the bound.
# promise: once `most` pages of a tool list were read, no further page is asked for
# harm: a server that always answers with a cursor would hold the request forever
# catches: sv_list_more answers the cursor alone
law sv_list_within_pages:
  for +c: Bool
  for +m: Nat
  for +d: Nat
  {sv_list_more(c, Nat.add(m, d), m) == False{} : Bool}

def sv_list_within_pages(c, m, d):
  match c:
    case False{}: {==}
    case True{}:
      sv_lt_add_false(m, d)

# LAW 4.8: A list without a cursor has ended.
# promise: a tool-list page without a cursor is the last one asked for
# catches: sv_list_more answers the page check alone (the last page asked for again)
law sv_list_ends_without_cursor:
  for +p: Nat
  for +m: Nat
  {sv_list_more(False{}, p, m) == False{} : Bool}

def sv_list_ends_without_cursor(p, m):
  {==}

# LAW 4.9: With pages left, a cursor is followed.
# promise: a tool list with a cursor is read on while pages are left
# catches: sv_list_more answers False{} always (only the first page is read)
law sv_list_follows_cursor:
  for +n: Nat
  for +d: Nat
  {sv_list_more(True{}, n, Nat.add(n, 1n+d)) == True{} : Bool}

def sv_list_follows_cursor(n, d):
  sv_lt_add_succ(n, d)

# ============================================================================
# 5. The request's output-token limit across rounds
# ============================================================================

# The output tokens the next round may generate: what is left of the request's limit `most`
# after the `used` tokens of the rounds before.
def sv_tokens_left(most: Nat, used: Nat) -> Nat:
  Nat.sub(most, used)

# Whether the loop generates another round: the round's decision goes on (`sv_goes_on`), and the
# request has output tokens left. `limited`: the request set an output limit; `left`: what is
# left of it (`sv_tokens_left`).
def sv_round_goes_on(n: SvNext, limited: Bool, left: Nat) -> Bool:
  match limited:
    case False{}:
      sv_goes_on(n)
    case True{}:
      Bool.and(sv_goes_on(n), Nat.is_gt(left, 0n))

# The round's decision goes on, but the request's output limit is spent: the reply stops there,
# for the limit (`max_tokens`).
def sv_budget_spent(n: SvNext, limited: Bool, left: Nat) -> Bool:
  match limited:
    case False{}:
      False{}
    case True{}:
      Bool.and(sv_goes_on(n), Nat.is_eq(left, 0n))

# LAW 5.1: A round is never given more output tokens than are left of the request's limit.
# promise: a round is never given more output tokens than are left of the request's limit after the rounds before
# harm: one reply could generate many times the max_tokens its client asked for
# catches: sv_tokens_left answers `most` alone (every round given the whole limit)
law sv_left_within_limit:
  for +m: Nat
  for +u: Nat
  {Nat.is_le(sv_tokens_left(m, u), Nat.sub(m, u)) == True{} : Bool}

def sv_left_within_limit(m, u):
  L.nat_le_refl(Nat.sub(m, u))

# LAW 5.2: A round is given everything that is left.
# promise: the next round is given at least every output token left of the request's limit
# harm: a round given less is cut by its own length limit, and the reply ends with max_tokens while the client's limit is far from spent
# catches: sv_tokens_left capped at one token (Nat.min(Nat.sub(most, used), 1n))
law sv_left_covers_unspent:
  for +m: Nat
  for +u: Nat
  {Nat.is_le(Nat.sub(m, u), sv_tokens_left(m, u)) == True{} : Bool}

def sv_left_covers_unspent(m, u):
  L.nat_le_refl(Nat.sub(m, u))

# LAW 5.3: No round once the limit is spent.
# promise: once the request's output-token limit is spent, the loop never generates another round
# harm: a round given no limit (0 means none to the chat path) would generate to the end of the context
# catches: sv_round_goes_on answers sv_goes_on(n) alone (the limit ignored)
law sv_spent_never_goes_on:
  for +n: SvNext
  {sv_round_goes_on(n, True{}, 0n) == False{} : Bool}

def sv_spent_never_goes_on(n):
  match n:
    case SvStop{}: {==}
    case SvRunGoOn{}: {==}
    case SvRunStop{}: {==}
    case SvRunPause{}: {==}

# LAW 5.4: The limit never makes a round the round's own decision stops.
# promise: the loop never generates a round that the round's own decision stops, whatever is left of the limit
# catches: sv_round_goes_on answers the tokens-left check alone (the round decision ignored)
law sv_round_within_rules:
  for +n: SvNext
  for +b: Bool
  for +l: Nat
  {Bool.and(sv_round_goes_on(n, b, l), Bool.not(sv_goes_on(n))) == False{} : Bool}

def sv_round_within_rules(n, b, l):
  match n:
    case SvRunGoOn{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}
    case SvStop{}:
      match b:
        case False{}: {==}
        case True{}: {==}
    case SvRunStop{}:
      match b:
        case False{}: {==}
        case True{}: {==}
    case SvRunPause{}:
      match b:
        case False{}: {==}
        case True{}: {==}

# LAW 5.5: With tokens left, the loop goes on as the round decided.
# promise: a round the round's decision continues generates again while output tokens are left of the limit
# catches: sv_round_goes_on off by one (Nat.is_gt(left, 1n): the last token left never used)
law sv_left_goes_on:
  for +b: Bool
  for +l: Nat
  {sv_round_goes_on(SvRunGoOn{}, b, 1n+l) == True{} : Bool}

def sv_left_goes_on(b, l):
  match b:
    case False{}: {==}
    case True{}: {==}

# LAW 5.6: Without a limit, the loop goes on as the round decided.
# promise: a request without an output limit generates its rounds as the round decision says, without a token bound
# catches: sv_round_goes_on ignores `limited` (a request without a limit stops after its first round)
law sv_unlimited_goes_on:
  for +l: Nat
  {sv_round_goes_on(SvRunGoOn{}, False{}, l) == True{} : Bool}

def sv_unlimited_goes_on(l):
  {==}

# LAW 5.7: The reply stops for the limit only when the limit is set and spent.
# promise: the reply stops for the output limit only when the request set one and none of it is left
# catches: sv_budget_spent answers sv_goes_on(n) alone (max_tokens while tokens are left)
law sv_spent_only_when_spent:
  for +n: SvNext
  for +b: Bool
  for +l: Nat
  {Bool.and(sv_budget_spent(n, b, l), Bool.or(Bool.not(b), Nat.is_gt(l, 0n))) == False{} : Bool}

def sv_spent_only_when_spent(n, b, l):
  match n:
    case SvStop{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}
    case SvRunGoOn{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}
    case SvRunStop{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}
    case SvRunPause{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}

# LAW 5.8: A round that stops on its own keeps its stop reason.
# promise: a round whose own decision stops (client calls waiting, no round left, nothing to run) never stops for the output limit instead
# catches: sv_budget_spent answers the tokens-left check alone (tool_use or pause_turn reported as max_tokens)
law sv_spent_keeps_stop_reason:
  for +n: SvNext
  for +b: Bool
  for +l: Nat
  {Bool.and(sv_budget_spent(n, b, l), Bool.not(sv_goes_on(n))) == False{} : Bool}

def sv_spent_keeps_stop_reason(n, b, l):
  match n:
    case SvStop{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}
    case SvRunGoOn{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}
    case SvRunStop{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}
    case SvRunPause{}:
      match b:
        case False{}: {==}
        case True{}:
          match l:
            case 0n: {==}
            case 1n+lp: {==}

# LAW 5.9: A spent limit is named as the stop reason.
# promise: a reply whose round decision goes on but whose output limit is spent stops for the limit
# catches: sv_budget_spent answers False{} always (a spent limit reported as pause_turn)
law sv_spent_says_so:
  {sv_budget_spent(SvRunGoOn{}, True{}, 0n) == True{} : Bool}

def sv_spent_says_so():
  {==}

# ============================================================================
# 6. Which Calls of a Round Run, and Whether the Reply Goes On
# ============================================================================

# Whether call `i` (0 first) of a round's server-side calls is sent to its server. whole: the model
# finished the call (the parser did not close it at the end of generation); cancelled: the reply
# was cancelled, or its client went away; take: the calls the round may run (sv_calls_take).
# A call that does not run is answered with an error result.
def sv_call_runs(whole: Bool, cancelled: Bool, i: Nat, take: Nat) -> Bool:
  Bool.and(Bool.and(whole, Bool.not(cancelled)), Nat.is_lt(i, take))

# LAW 6.1: A call the model never finished never runs.
# promise: a call the parser closed at the end of generation (not finished by the model) is never sent to its server
# harm: a call whose arguments were cut mid-value would run remotely with the partial value
# catches: the whole flag ignored (Bool.and(Bool.not(cancelled), Nat.is_lt(i, take)))
law sv_repaired_call_never_runs:
  for +c: Bool
  for +i: Nat
  for +t: Nat
  {sv_call_runs(False{}, c, i, t) == False{} : Bool}

def sv_repaired_call_never_runs(c, i, t):
  match c:
    case True{}: {==}
    case False{}: {==}

# LAW 6.2: No call runs once the reply is cancelled.
# promise: once the reply is cancelled (or its client is gone), no further call is sent to any server
# harm: a cancelled request would go on causing side effects on remote tool servers
# catches: the cancelled flag ignored (Bool.and(whole, Nat.is_lt(i, take)))
law sv_cancelled_call_never_runs:
  for +w: Bool
  for +i: Nat
  for +t: Nat
  {sv_call_runs(w, True{}, i, t) == False{} : Bool}

def sv_cancelled_call_never_runs(w, i, t):
  match w:
    case True{}: {==}
    case False{}: {==}

# LAW 6.3: A whole call within the take runs while the reply lives.
# promise: a call the model finished, inside the round's take, of a reply not cancelled, is run
# catches: sv_call_runs answers False{} always (no server-side call ever runs)
law sv_whole_live_call_runs:
  for +i: Nat
  for +d: Nat
  {sv_call_runs(True{}, False{}, i, Nat.add(i, 1n+d)) == True{} : Bool}

def sv_whole_live_call_runs(i, d):
  match i:
    case 0n: {==}
    case 1n+ip:
      sv_whole_live_call_runs(ip, d)

# LAW 6.4: No call past the round's take runs.
# promise: a call at or past the number of calls the round may run is never sent
# harm: a round could exceed the per-request call limit by one call
# catches: the index bound off by one (Nat.is_le(i, take))
law sv_call_past_take_never_runs:
  for +w: Bool
  for +c: Bool
  for +t: Nat
  for +d: Nat
  {sv_call_runs(w, c, Nat.add(t, d), t) == False{} : Bool}

def sv_call_past_take_never_runs(w, c, t, d):
  match w c:
    case True{} False{}:
      match t d:
        case 0n 0n: {==}
        case 0n 1n+dp: {==}
        case 1n+tp _:
          sv_call_past_take_never_runs(True{}, False{}, tp, d)
    case True{} True{}: {==}
    case False{} _: {==}

# After a round's calls ran: another round is generated only when the round's decision and the
# output limit let it (sv_round_goes_on) and the reply was not cancelled.
def sv_goes_on_live(cancelled: Bool, n: SvNext, limited: Bool, left: Nat) -> Bool:
  Bool.and(Bool.not(cancelled), sv_round_goes_on(n, limited, left))

# LAW 6.5: A cancelled reply generates no further round.
# promise: once the reply is cancelled (or its client is gone), the loop never generates another round
# harm: a cancel sent with the reply's id would be ignored after the first round, and the card would keep generating for nobody
# catches: the cancelled flag ignored (sv_goes_on_live answers sv_round_goes_on(n, limited, left))
law sv_cancelled_never_goes_on:
  for +n: SvNext
  for +b: Bool
  for +l: Nat
  {sv_goes_on_live(True{}, n, b, l) == False{} : Bool}

def sv_cancelled_never_goes_on(n, b, l):
  {==}

# LAW 6.6: A live reply goes on as the round decided.
# promise: a reply not cancelled, without an output limit, whose round decision goes on generates another round
# catches: sv_goes_on_live answers False{} always (the loop never reaches a second round)
law sv_live_round_goes_on:
  {sv_goes_on_live(False{}, SvRunGoOn{}, False{}, 0n) == True{} : Bool}

def sv_live_round_goes_on():
  {==}

# LAW 6.7: A live reply never goes on past what the round decided.
# promise: the loop never generates a round that the round's own decision or the output limit refuses, cancelled or not
# harm: a reply would run past its round bound or its max_tokens while nobody cancelled it
# catches: the round decision dropped (sv_goes_on_live answers Bool.not(cancelled) alone)
law sv_live_within_rules:
  for +c: Bool
  for +n: SvNext
  for +b: Bool
  for +l: Nat
  {Bool.and(sv_goes_on_live(c, n, b, l), Bool.not(sv_round_goes_on(n, b, l))) == False{} : Bool}

def sv_live_within_rules(c, n, b, l):
  match c:
    case True{}: {==}
    case False{}:
      match n:
        case SvRunGoOn{}:
          match b:
            case False{}: {==}
            case True{}:
              match l:
                case 0n: {==}
                case 1n+lp: {==}
        case SvStop{}:
          match b:
            case False{}: {==}
            case True{}: {==}
        case SvRunStop{}:
          match b:
            case False{}: {==}
            case True{}: {==}
        case SvRunPause{}:
          match b:
            case False{}: {==}
            case True{}: {==}
