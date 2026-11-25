# Certificate Setup Core (`gaudium --make-cert`)
# Formally specifies and proves the decisions of the command that makes the server's HTTPS
# certificate. A directory holds a small local certificate authority (its certificate and its key)
# and the server's certificate and key, signed by that authority:
#   - Which files one run writes. The authority is made when neither of its files exists and
#     reused when both do; a run never replaces it, never pairs a server certificate with half an
#     authority, and replaces the server's two files only when the operator asked for that
#   - How long the server certificate is valid: never longer than asked, never past the end of
#     the authority that signs it, and never zero days

import Base
import nat_lemmas.bend as L

# ============================================================================
# 1. Which Files a Run Writes
# ============================================================================

type CgPlan is Data:
  CgHalfCa{}
  CgServerExists{}
  CgReuseCa{}
  CgNewCa{}

# The server's two files may be written: neither exists, or the operator asked for them to be
# replaced.
def cg_server_free(srv_cert: Bool, srv_key: Bool, overwrite: Bool) -> Bool:
  Bool.or(overwrite, Bool.not(Bool.or(srv_cert, srv_key)))

def cg_plan_server(free: Bool, write: CgPlan) -> CgPlan:
  match free:
    case True{}:
      write
    case False{}:
      CgServerExists{}

# ca_cert, ca_key:   the authority's certificate file / key file exists
# srv_cert, srv_key: the server's certificate file / key file exists
# overwrite:         the operator asked for the server's files to be replaced
def cg_plan(ca_cert: Bool, ca_key: Bool, srv_cert: Bool, srv_key: Bool, overwrite: Bool) -> CgPlan:
  match ca_cert ca_key:
    case True{} False{}:
      CgHalfCa{}
    case False{} True{}:
      CgHalfCa{}
    case True{} True{}:
      cg_plan_server(cg_server_free(srv_cert, srv_key, overwrite), CgReuseCa{})
    case False{} False{}:
      cg_plan_server(cg_server_free(srv_cert, srv_key, overwrite), CgNewCa{})

# The run writes the server's certificate and key.
def cg_writes_server(p: CgPlan) -> Bool:
  match p:
    case CgReuseCa{}:
      True{}
    case CgNewCa{}:
      True{}
    case _:
      False{}

# The run makes the authority (writes its certificate and key).
def cg_writes_ca(p: CgPlan) -> Bool:
  match p:
    case CgNewCa{}:
      True{}
    case _:
      False{}

# LAW 1.1
# promise: a run never writes the authority's files while either of them exists, whatever the overwrite flag says
# harm: every device that trusts the authority would stop trusting the server, and its key would be lost
# catches: the overwrite flag also applied to the authority (a new one made over the existing files)
law cg_ca_never_replaced:
  for +a: Bool
  for +b: Bool
  for +c: Bool
  for +d: Bool
  for +f: Bool
  {Bool.and(cg_writes_ca(cg_plan(a, b, c, d, f)), Bool.or(a, b)) == False{} : Bool}

def cg_ca_never_replaced(a, b, c, d, f):
  match a b c d f:
    case True{} True{} True{} True{} True{}: {==}
    case True{} True{} True{} True{} False{}: {==}
    case True{} True{} True{} False{} True{}: {==}
    case True{} True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{} True{}: {==}
    case True{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} False{} True{}: {==}
    case True{} True{} False{} False{} False{}: {==}
    case True{} False{} True{} True{} True{}: {==}
    case True{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} False{} True{}: {==}
    case True{} False{} True{} False{} False{}: {==}
    case True{} False{} False{} True{} True{}: {==}
    case True{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} False{} True{}: {==}
    case True{} False{} False{} False{} False{}: {==}
    case False{} True{} True{} True{} True{}: {==}
    case False{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} False{} False{}: {==}
    case False{} True{} False{} True{} True{}: {==}
    case False{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} False{} False{}: {==}
    case False{} False{} True{} True{} True{}: {==}
    case False{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} False{} True{}: {==}
    case False{} False{} True{} False{} False{}: {==}
    case False{} False{} False{} True{} True{}: {==}
    case False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} False{} False{}: {==}

# LAW 1.2
# promise: without the overwrite flag a run never writes the server's files while either of them exists
# harm: a working certificate and key would be replaced without being asked
# catches: only the certificate file looked at (an existing server key ignored)
law cg_server_kept_without_overwrite:
  for +a: Bool
  for +b: Bool
  for +c: Bool
  for +d: Bool
  {Bool.and(cg_writes_server(cg_plan(a, b, c, d, False{})), Bool.or(c, d)) == False{} : Bool}

def cg_server_kept_without_overwrite(a, b, c, d):
  match a b c d:
    case True{} True{} True{} True{}: {==}
    case True{} True{} True{} False{}: {==}
    case True{} True{} False{} True{}: {==}
    case True{} True{} False{} False{}: {==}
    case True{} False{} True{} True{}: {==}
    case True{} False{} True{} False{}: {==}
    case True{} False{} False{} True{}: {==}
    case True{} False{} False{} False{}: {==}
    case False{} True{} True{} True{}: {==}
    case False{} True{} True{} False{}: {==}
    case False{} True{} False{} True{}: {==}
    case False{} True{} False{} False{}: {==}
    case False{} False{} True{} True{}: {==}
    case False{} False{} True{} False{}: {==}
    case False{} False{} False{} True{}: {==}
    case False{} False{} False{} False{}: {==}

# LAW 1.3
# promise: a server certificate is never made when the authority has only one of its two files
# harm: a certificate signed by a key that does not match the authority the devices trust
# catches: an authority whose key file is missing reused as if it were whole
law cg_half_ca_refused:
  for +a: Bool
  for +b: Bool
  for +c: Bool
  for +d: Bool
  for +f: Bool
  {Bool.and(Bool.xor(a, b), cg_writes_server(cg_plan(a, b, c, d, f))) == False{} : Bool}

def cg_half_ca_refused(a, b, c, d, f):
  match a b c d f:
    case True{} True{} True{} True{} True{}: {==}
    case True{} True{} True{} True{} False{}: {==}
    case True{} True{} True{} False{} True{}: {==}
    case True{} True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{} True{}: {==}
    case True{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} False{} True{}: {==}
    case True{} True{} False{} False{} False{}: {==}
    case True{} False{} True{} True{} True{}: {==}
    case True{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} False{} True{}: {==}
    case True{} False{} True{} False{} False{}: {==}
    case True{} False{} False{} True{} True{}: {==}
    case True{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} False{} True{}: {==}
    case True{} False{} False{} False{} False{}: {==}
    case False{} True{} True{} True{} True{}: {==}
    case False{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} False{} False{}: {==}
    case False{} True{} False{} True{} True{}: {==}
    case False{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} False{} False{}: {==}
    case False{} False{} True{} True{} True{}: {==}
    case False{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} False{} True{}: {==}
    case False{} False{} True{} False{} False{}: {==}
    case False{} False{} False{} True{} True{}: {==}
    case False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} False{} False{}: {==}

# LAW 1.4
# promise: a server certificate is only written when its authority is made in the same run or both of the authority's files exist
# harm: a server certificate signed by an authority that is not kept, so no device can be told to trust it
# catches: an empty directory planned as reusing an authority it does not have
law cg_server_needs_signer:
  for +a: Bool
  for +b: Bool
  for +c: Bool
  for +d: Bool
  for +f: Bool
  {Bool.and(cg_writes_server(cg_plan(a, b, c, d, f)), Bool.not(Bool.or(cg_writes_ca(cg_plan(a, b, c, d, f)), Bool.and(a, b)))) == False{} : Bool}

def cg_server_needs_signer(a, b, c, d, f):
  match a b c d f:
    case True{} True{} True{} True{} True{}: {==}
    case True{} True{} True{} True{} False{}: {==}
    case True{} True{} True{} False{} True{}: {==}
    case True{} True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{} True{}: {==}
    case True{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} False{} True{}: {==}
    case True{} True{} False{} False{} False{}: {==}
    case True{} False{} True{} True{} True{}: {==}
    case True{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} False{} True{}: {==}
    case True{} False{} True{} False{} False{}: {==}
    case True{} False{} False{} True{} True{}: {==}
    case True{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} False{} True{}: {==}
    case True{} False{} False{} False{} False{}: {==}
    case False{} True{} True{} True{} True{}: {==}
    case False{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} False{} False{}: {==}
    case False{} True{} False{} True{} True{}: {==}
    case False{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} False{} False{}: {==}
    case False{} False{} True{} True{} True{}: {==}
    case False{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} False{} True{}: {==}
    case False{} False{} True{} False{} False{}: {==}
    case False{} False{} False{} True{} True{}: {==}
    case False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} False{} False{}: {==}

# LAW 1.5
# promise: when the server's files are absent and the authority is whole or absent, a run writes them
# catches: a run refused whenever the authority already exists (renewal impossible)
law cg_free_dir_writes:
  for +a: Bool
  for +f: Bool
  {cg_writes_server(cg_plan(a, a, False{}, False{}, f)) == True{} : Bool}

def cg_free_dir_writes(a, f):
  match a f:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# LAW 1.6
# promise: with the overwrite flag the server's files are replaced whenever the authority is whole or absent
# catches: the overwrite flag ignored (existing server files always refused)
law cg_overwrite_replaces:
  for +a: Bool
  for +c: Bool
  for +d: Bool
  {cg_writes_server(cg_plan(a, a, c, d, True{})) == True{} : Bool}

def cg_overwrite_replaces(a, c, d):
  match a c d:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

# ============================================================================
# 2. How Long the Server Certificate Is Valid
# ============================================================================

type CgValid is Data:
  CgNoDaysAsked{}
  CgCaSpent{}
  CgValidFor{days: Nat}

# asked:   whole days the operator asked the server certificate to be valid for
# ca_left: whole days the authority that signs it is still valid from now
def cg_validity(asked: Nat, ca_left: Nat) -> CgValid:
  match asked ca_left:
    case 0n _:
      CgNoDaysAsked{}
    case 1n+a 0n:
      CgCaSpent{}
    case 1n+a 1n+l:
      CgValidFor{Nat.min(1n+a, 1n+l)}

# The days the certificate that is made is valid for (0 when none is made).
def cg_valid_days(v: CgValid) -> Nat:
  match v:
    case CgValidFor{d}:
      d
    case _:
      0n

# No certificate is made.
def cg_valid_refused(v: CgValid) -> Bool:
  match v:
    case CgValidFor{d}:
      False{}
    case _:
      True{}

# LAW 2.1
# promise: the server certificate is never valid for longer than the operator asked
# catches: Nat.max in place of Nat.min
law cg_days_within_asked:
  for +a: Nat
  for +l: Nat
  {Nat.is_le(cg_valid_days(cg_validity(a, l)), a) == True{} : Bool}

def cg_days_within_asked(a, l):
  match a l:
    case 0n _:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+lp:
      L.nat_min_le_l(1n+ap, 1n+lp)

# LAW 2.2
# promise: the server certificate never outlives the authority that signs it
# harm: browsers would refuse the server once the authority has expired, while its certificate still looks valid
# catches: the authority's remaining days ignored (the asked days used as they are)
law cg_days_within_ca:
  for +a: Nat
  for +l: Nat
  {Nat.is_le(cg_valid_days(cg_validity(a, l)), l) == True{} : Bool}

def cg_days_within_ca(a, l):
  match a l:
    case 0n _:
      L.nat_le_zero(l)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+lp:
      L.nat_min_le_r(1n+ap, 1n+lp)

# LAW 2.3
# promise: a server certificate that is made is valid for at least one day
# catches: a day taken off as a margin, so that a one-day request gives a certificate that is already expired
law cg_made_days_positive:
  for +a: Nat
  for +l: Nat
  {Bool.or(cg_valid_refused(cg_validity(a, l)), Nat.is_lt(0n, cg_valid_days(cg_validity(a, l)))) == True{} : Bool}

def cg_made_days_positive(a, l):
  match a l:
    case 0n _:
      {==}
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+lp:
      {==}

# LAW 2.4
# promise: the validity decision never refuses a request for at least one day whose authority has at least a whole day left
# harm: the command would refuse every certificate although nothing stands in its way
# catches: every request refused (cg_validity always CgNoDaysAsked{})
law cg_room_makes_certificate:
  for +a: Nat
  for +l: Nat
  {cg_valid_refused(cg_validity(1n+a, 1n+l)) == False{} : Bool}

def cg_room_makes_certificate(a, l):
  {==}
