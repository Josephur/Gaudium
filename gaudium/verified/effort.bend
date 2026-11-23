# Slice — Verified Reasoning-Effort Mapping
# Formally specifies and proves how a request's reasoning effort meets the levels a model's chat
# template accepts:
#   - The levels a template accepts, as a set over one ordered effort scale
#   - The plan for a request: reasoning off, one of the offered levels, reasoning at the
#     template's own default, or a refusal
#   - The levels a model advertises (the same plan, asked for each level of the scale)
#
# The effort scale, low to high, as ranks: 0n minimal, 1n low, 2n medium, 3n high, 4n xhigh,
# 5n max. Level names are text and stay in the Rust shell (`core::effort`), which turns a name
# into its rank and a rank back into its name; this file sees only ranks.

import Base

# ============================================================================
# 1. Types
# ============================================================================

# The levels of the scale a template accepts for `reasoning_effort`, one flag per rank.
type EffortSet is Data:
  EfSet{minimal: Bool, low: Bool, medium: Bool, high: Bool, xhigh: Bool, max: Bool}

# What a request asks for.
type EffortAsk is Data:
  EaOff{}             # "none": no reasoning
  EaLevel{rank: Nat}  # a level of the scale
  EaOther{}           # a name that is not on the scale

# What the prompt gets.
type EffortPlan is Data:
  EpThinkOff{}        # reasoning off
  EpLevel{rank: Nat}  # reasoning on, the template is given this level
  EpThinkOnly{}       # reasoning on, no level (the template's own default)
  EpRefuse{}          # the request is refused

# ============================================================================
# 2. The offered set
# ============================================================================

def effort_has_at(a: Bool, b: Bool, c: Bool, d: Bool, e: Bool, f: Bool, r: Nat) -> Bool:
  match r:
    case 0n:
      a
    case 1n+r1:
      match r1:
        case 0n:
          b
        case 1n+r2:
          match r2:
            case 0n:
              c
            case 1n+r3:
              match r3:
                case 0n:
                  d
                case 1n+r4:
                  match r4:
                    case 0n:
                      e
                    case 1n+r5:
                      match r5:
                        case 0n:
                          f
                        case 1n+_:
                          False{}

# The template accepts the level of rank `r` (a rank past the scale is never offered).
def effort_has(s: EffortSet, r: Nat) -> Bool:
  match s:
    case EfSet{a, b, c, d, e, f}:
      effort_has_at(a, b, c, d, e, f, r)

# The template accepts at least one level (it has an effort setting).
def effort_any(s: EffortSet) -> Bool:
  match s:
    case EfSet{a, b, c, d, e, f}:
      Bool.or(a, Bool.or(b, Bool.or(c, Bool.or(d, Bool.or(e, f)))))

# ============================================================================
# 3. The plan for a request
# ============================================================================

# One step of the search: the rank above, else the rank below, else what the next step finds.
def effort_pick(up: Bool, down: Bool, a: Nat, b: Nat, rest: Nat) -> Nat:
  match up:
    case True{}:
      a
    case False{}:
      match down:
        case True{}:
          b
        case False{}:
          rest

# The nearest rank to `r` at distance `d` or more that the template accepts, searched outwards
# (the higher one first on a tie); `r` itself when the fuel runs out.
def effort_search(fuel: Nat, +s: EffortSet, +r: Nat, +d: Nat) -> Nat:
  match fuel:
    case 0n:
      r
    case 1n+f:
      effort_pick(effort_has(s, Nat.add(r, d)), Bool.and(Nat.is_le(d, r), effort_has(s, Nat.sub(r, d))), Nat.add(r, d), Nat.sub(r, d), effort_search(f, s, r, 1n+d))

# A candidate level reaches the template only when the template accepts it.
def effort_guard(b: Bool, c: Nat) -> EffortPlan:
  match b:
    case True{}:
      EpLevel{c}
    case False{}:
      EpThinkOnly{}

def effort_near(+s: EffortSet, +c: Nat) -> EffortPlan:
  effort_guard(effort_has(s, c), c)

# A level of the scale: itself when offered, else the nearest offered one.
def effort_level_with(b: Bool, +s: EffortSet, +r: Nat) -> EffortPlan:
  match b:
    case True{}:
      EpLevel{r}
    case False{}:
      effort_near(s, effort_search(5n, s, r, 1n))

# A name off the scale: refused by a model that has levels, plain reasoning by one that has none.
def effort_other_with(b: Bool) -> EffortPlan:
  match b:
    case True{}:
      EpRefuse{}
    case False{}:
      EpThinkOnly{}

def effort_plan(+s: EffortSet, a: EffortAsk) -> EffortPlan:
  match a:
    case EaOff{}:
      EpThinkOff{}
    case EaLevel{+r}:
      effort_level_with(effort_has(s, r), s, r)
    case EaOther{}:
      effort_other_with(effort_any(s))

# ============================================================================
# 4. Observers
# ============================================================================

# A plan's level is one the template accepts.
def effort_plan_ok(s: EffortSet, p: EffortPlan) -> Bool:
  match p:
    case EpLevel{c}:
      effort_has(s, c)
    case _:
      True{}

def effort_plan_is_level(p: EffortPlan) -> Bool:
  match p:
    case EpLevel{c}:
      True{}
    case _:
      False{}

# The plan gives the template the level of rank `r`.
def effort_plan_is(p: EffortPlan, r: Nat) -> Bool:
  match p:
    case EpLevel{c}:
      Nat.is_eq(c, r)
    case _:
      False{}

def effort_plan_is_off(p: EffortPlan) -> Bool:
  match p:
    case EpThinkOff{}:
      True{}
    case _:
      False{}

# The level of the scale a model advertises: asking for it keeps it.
def effort_advertised(+s: EffortSet, +r: Nat) -> Bool:
  effort_plan_is(effort_plan(s, EaLevel{r}), r)

# ============================================================================
# 5. Helper lemmas (erased before Rust codegen)
# ============================================================================

def effort_eq_refl(n: Nat) -> {Nat.is_eq(n, n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      effort_eq_refl(p)

def effort_has_empty(r: Nat) -> {effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, r) == False{} : Bool}:
  match r:
    case 0n: {==}
    case 1n+r1:
      match r1:
        case 0n: {==}
        case 1n+r2:
          match r2:
            case 0n: {==}
            case 1n+r3:
              match r3:
                case 0n: {==}
                case 1n+r4:
                  match r4:
                    case 0n: {==}
                    case 1n+r5:
                      match r5:
                        case 0n: {==}
                        case 1n+_: {==}

def effort_guard_ok(+s: EffortSet, +c: Nat, b: Bool, h: {effort_has(s, c) == b : Bool}) -> {effort_plan_ok(s, effort_guard(b, c)) == True{} : Bool}:
  match b:
    case True{}:
      h
    case False{}:
      {==}

def effort_level_ok(+s: EffortSet, +r: Nat, b: Bool, h: {effort_has(s, r) == b : Bool}) -> {effort_plan_ok(s, effort_level_with(b, s, r)) == True{} : Bool}:
  match b:
    case True{}:
      h
    case False{}:
      effort_guard_ok(s, effort_search(5n, s, r, 1n), effort_has(s, effort_search(5n, s, r, 1n)), {==})

def effort_other_ok(+s: EffortSet, b: Bool) -> {effort_plan_ok(s, effort_other_with(b)) == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def effort_empty_guard(+c: Nat, b: Bool, h: {effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, c) == b : Bool}) -> {effort_plan_is_level(effort_guard(b, c)) == False{} : Bool}:
  match b:
    case True{}:
      Equal.trans(Bool, True{}, effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, c), False{}, Equal.sym(Bool, effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, c), True{}, h), effort_has_empty(c))
    case False{}:
      {==}

def effort_empty_level(+r: Nat, b: Bool, h: {effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, r) == b : Bool}) -> {effort_plan_is_level(effort_level_with(b, EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, r)) == False{} : Bool}:
  match b:
    case True{}:
      Equal.trans(Bool, True{}, effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, r), False{}, Equal.sym(Bool, effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, r), True{}, h), effort_has_empty(r))
    case False{}:
      effort_empty_guard(effort_search(5n, EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, r, 1n), effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, effort_search(5n, EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, r, 1n)), {==})

def effort_kept_with(+s: EffortSet, +r: Nat, b: Bool) -> {Bool.or(Bool.not(b), effort_plan_is(effort_level_with(b, s, r), r)) == True{} : Bool}:
  match b:
    case True{}:
      effort_eq_refl(r)
    case False{}:
      {==}

def effort_guard_thinks(+c: Nat, b: Bool) -> {effort_plan_is_off(effort_guard(b, c)) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def effort_level_thinks(+s: EffortSet, +r: Nat, b: Bool) -> {effort_plan_is_off(effort_level_with(b, s, r)) == False{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      effort_guard_thinks(effort_search(5n, s, r, 1n), effort_has(s, effort_search(5n, s, r, 1n)))

# ============================================================================
# 6. Laws
# ============================================================================

# Whatever is asked, a level that reaches the template is one the template accepts.
law effort_plan_offered:
  for +s: EffortSet
  for +a: EffortAsk
  {effort_plan_ok(s, effort_plan(s, a)) == True{} : Bool}

def effort_plan_offered(s, a):
  match a:
    case EaOff{}:
      {==}
    case EaLevel{+r}:
      effort_level_ok(s, r, effort_has(s, r), {==})
    case EaOther{}:
      effort_other_ok(s, effort_any(s))

# A template without an effort setting is never given a level.
law effort_no_levels_no_level:
  for +a: EffortAsk
  {effort_plan_is_level(effort_plan(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, a)) == False{} : Bool}

def effort_no_levels_no_level(a):
  match a:
    case EaOff{}:
      {==}
    case EaLevel{+r}:
      effort_empty_level(r, effort_has(EfSet{False{}, False{}, False{}, False{}, False{}, False{}}, r), {==})
    case EaOther{}:
      {==}

# Asking for a level the template accepts gives exactly that level.
law effort_offered_kept:
  for +s: EffortSet
  for +r: Nat
  {Bool.or(Bool.not(effort_has(s, r)), effort_plan_is(effort_plan(s, EaLevel{r}), r)) == True{} : Bool}

def effort_offered_kept(s, r):
  effort_kept_with(s, r, effort_has(s, r))

# Asking for a level never turns reasoning off.
law effort_level_ask_thinks:
  for +s: EffortSet
  for +r: Nat
  {effort_plan_is_off(effort_plan(s, EaLevel{r})) == False{} : Bool}

def effort_level_ask_thinks(s, r):
  effort_level_thinks(s, r, effort_has(s, r))

# Every level the template accepts is advertised.
law effort_offered_advertised:
  for +s: EffortSet
  for +r: Nat
  {Bool.or(Bool.not(effort_has(s, r)), effort_advertised(s, r)) == True{} : Bool}

def effort_offered_advertised(s, r):
  effort_kept_with(s, r, effort_has(s, r))
