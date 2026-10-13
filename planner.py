"""TestWise planner (Phase 2): contingent planning and deterministic graph search.

Part A - contingent planner (``solve_contingent_plan``)
-------------------------------------------------------
Exact outcome-dependent (contingent) planning by memoized expectimax over
finite planning states s = (beliefs b, performed investigations, remaining
budget).  Terminal classification value A(b) = max_i b_i.  Bellman recurrence:

    V(s) = max{ A(b),
                max_{t in avail(s)} [ -lambda * c_t + sum_o P(o|b,t) V(s_{t,o}) ] }

* ``avail(s)``: investigations that are unperformed, permitted and affordable.
* Stopping is always available and yields A(b).
* lambda = ``PlannerSettings.cost_penalty_lambda`` in CLASSIFICATION-UTILITY
  units per cost unit.  It is NOT bits per cost unit; information gain (bits)
  is a separate quantity and never enters this objective.
* ``PlannerSettings.confidence_threshold`` is NOT used here: the planner may
  stop earlier or later than the threshold if that has greater expected utility.
* Outcome probabilities come from ``model.outcome_probability`` and successor
  beliefs from ``model.update_beliefs``; no Bayes formula is duplicated.

Budget arithmetic (exact, no tolerance): affordability and budget updates use
exact rational arithmetic on the decimal value of each float (its shortest
round-trip ``repr``), so 0.3 - 0.1 is exactly 0.2 and a cost is affordable iff
``cost <= remaining_budget`` exactly.  ``PlanningState.remaining_budget`` is the
unrounded actual budget; it is never rounded for caching.  The next-state
budget is the nearest double to the exact difference, and affordability is
always judged against that stored float.

Memoization key (exact within one solve): (evidence, performed, remaining
budget, permitted), where evidence is the frozenset of (investigation,
outcome) pairs observed on the path from the root.  Beliefs are a function of
the root beliefs and the evidence set (Bayesian updates commute), and the legal
actions are a function of performed / budget / permitted, so equal keys mean
mathematically identical planning states.  No belief or budget rounding is
used.  Different orders reaching the same evidence set share one entry; states
whose beliefs merely look similar are never merged.  The only residual
numerical effect is ordinary floating-point evaluation noise (~1e-16): the
cached node's beliefs were computed along whichever path reached it first.

Tie-breaking (deterministic): stopping wins ties (within ``TIE_EPS``); among
investigations the earlier one in model declaration order wins ties.  Derived
bound (exact arithmetic, ignoring ~1e-16 floating-point error): at each node the
chosen option is within TIE_EPS of the best option, so by induction on the
remaining decision depth k the reported utility of the returned policy tree is
within k * TIE_EPS of the true optimum (k <= number of investigations, so
<= 6e-12 for the 6-investigation default model).

Search limit: at most ``max_state_expansions`` distinct states are expanded
(cache hits are free).  If more are needed the result is INCOMPLETE: no
root action, no utility and no policy tree is reported (never a greedy
fallback presented as optimal).

Part B - deterministic graph search (BFS, DFS, A*) on a ``SearchProblem``.
The contingent planner is NOT A*; they are unrelated algorithms.

No UI, network or randomness is used here.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from fractions import Fraction
from dataclasses import dataclass, field
from enum import Enum
from typing import (Callable, Generic, Hashable, Iterable, Optional, TypeVar)

from config import Investigation, ModelConfig, PlannerSettings
from model import BeliefState, outcome_probability, update_beliefs

TIE_EPS: float = 1e-12   # utility ties: stop preferred, then earlier test (see docstring)


class PlanningError(ValueError):
    """Invalid planning state, settings, or identifiers."""


# --------------------------------------------------------------------------- #
# Planning state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlanningState:
    """Observable decision state (contains no hidden ground truth).

    beliefs: current BeliefState (probabilities[hypothesis_id]).
    performed: IDs of investigations already done (cannot be repeated).
    remaining_budget: unspent budget in cost units (finite, >= 0).
    permitted: optional whitelist of investigation IDs the decision maker may
        use; ``None`` means all investigations are permitted.
    """
    beliefs: BeliefState
    performed: frozenset[str] = frozenset()
    remaining_budget: float = 0.0
    permitted: Optional[frozenset[str]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.beliefs, BeliefState):
            raise PlanningError("beliefs must be a BeliefState")
        object.__setattr__(self, "performed", frozenset(self.performed))
        if self.permitted is not None:
            object.__setattr__(self, "permitted", frozenset(self.permitted))
        b = self.remaining_budget
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or b < 0:
            raise PlanningError(f"remaining_budget must be a finite number >= 0, got {b!r}")


def make_initial_state(beliefs: BeliefState, settings: PlannerSettings,
                       permitted: Optional[Iterable[str]] = None) -> PlanningState:
    """Initial state: nothing performed, budget = settings.initial_budget."""
    return PlanningState(beliefs, frozenset(), float(settings.initial_budget),
                         None if permitted is None else frozenset(permitted))


def _exact(x: float) -> Fraction:
    """Exact rational value of the decimal shown by ``repr(float(x))``."""
    return Fraction(repr(float(x)))


def is_affordable(cost: float, remaining_budget: float) -> bool:
    """True iff ``cost <= remaining_budget`` EXACTLY (no tolerance, no rounding).

    Comparison is done on the decimal values of the floats (see module
    docstring), so a cost 1e-16 above the remaining budget is NOT affordable.
    """
    return _exact(cost) <= _exact(remaining_budget)


def available_investigations(state: PlanningState, model: ModelConfig) -> list[Investigation]:
    """Unperformed, permitted, affordable investigations in model order."""
    _check_state_ids(state, model)
    return [inv for inv in model.investigations
            if inv.id not in state.performed
            and (state.permitted is None or inv.id in state.permitted)
            and is_affordable(inv.cost, state.remaining_budget)]


def advance_state(state: PlanningState, test_id: str, outcome: str, model: ModelConfig) -> PlanningState:
    """Successor state after performing ``test_id`` and observing ``outcome``.

    Beliefs are updated with ``model.update_beliefs`` (which rejects impossible
    outcomes), the test is added to ``performed`` and its cost is deducted.
    """
    if test_id in state.performed:
        raise PlanningError(f"investigation {test_id!r} was already performed")
    try:
        inv = model.investigation(test_id)
    except KeyError:
        raise PlanningError(f"unknown investigation {test_id!r}") from None
    if not is_affordable(inv.cost, state.remaining_budget):
        raise PlanningError(f"investigation {test_id!r} (cost {inv.cost}) exceeds remaining budget "
                            f"{state.remaining_budget}")
    new_beliefs = update_beliefs(state.beliefs, test_id, outcome, model)
    # Exact decimal subtraction; nonnegative because the cost was just checked.
    remaining = float(_exact(state.remaining_budget) - _exact(inv.cost))
    return PlanningState(new_beliefs, state.performed | {test_id}, remaining, state.permitted)


def _check_state_ids(state: PlanningState, model: ModelConfig) -> None:
    known = {i.id for i in model.investigations}
    bad = set(state.performed) - known
    if bad:
        raise PlanningError(f"unknown performed investigation IDs: {sorted(bad)}")
    if state.permitted is not None and set(state.permitted) - known:
        raise PlanningError(f"unknown permitted investigation IDs: {sorted(set(state.permitted) - known)}")


def terminal_value(beliefs: BeliefState) -> float:
    """A(b) = max_i b_i, the probability of a correct best-guess classification."""
    return max(beliefs.probabilities.values())


def terminal_decision(beliefs: BeliefState, model: ModelConfig) -> str:
    """Hypothesis ID attaining A(b); ties go to the earlier hypothesis in model order."""
    best, best_p = None, -1.0
    for h in model.hypothesis_ids():
        if beliefs.probabilities[h] > best_p:
            best, best_p = h, beliefs.probabilities[h]
    return best  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Policy tree and result types
# --------------------------------------------------------------------------- #
class PlanStatus(str, Enum):
    """COMPLETE: search finished, result is optimal for the recurrence.
    INCOMPLETE_EXPANSION_LIMIT: limit hit; nothing is claimed optimal."""
    COMPLETE = "complete"
    INCOMPLETE_EXPANSION_LIMIT = "incomplete_expansion_limit"


@dataclass(frozen=True)
class OutcomeBranch:
    """One chance branch of a chosen investigation.

    reachable=False means P(outcome | beliefs) is exactly 0.0 (or underflowed
    to 0.0); then posterior and child are None and nothing is fabricated.
    """
    outcome: str
    probability: float
    reachable: bool
    posterior: Optional[BeliefState]
    child: Optional["PolicyNode"]


@dataclass(frozen=True)
class PolicyNode:
    """A decision point of the policy tree.

    beliefs / performed (sorted) / remaining_budget: the state.
    terminal_decision, terminal_value: best-guess hypothesis and A(b) if stopped here.
    action: chosen investigation ID, or None meaning STOP.
    value: V(s), expected net utility (classification-utility units).
    action_values: (test_id, net utility) for every available investigation.
    branches: outcome branches of ``action`` (empty when stopping).
    Nodes of equivalent states are shared (the tree is a DAG in memory).
    """
    beliefs: BeliefState
    performed: tuple[str, ...]
    remaining_budget: float
    terminal_decision: str
    terminal_value: float
    action: Optional[str]
    value: float
    action_values: tuple[tuple[str, float], ...]
    branches: tuple[OutcomeBranch, ...]


@dataclass(frozen=True)
class PlanningResult:
    """Result of ``solve_contingent_plan``.

    status: COMPLETE or INCOMPLETE_EXPANSION_LIMIT.
    root_action: investigation ID to perform first; None with
        stops_at_root=True means stop immediately; None with stops_at_root=False
        means NO decision is available (incomplete search).
    expected_utility: V(root) (None if incomplete).
    stop_utility: A(b) at the root; always a valid utility of stopping.
    policy_tree: root PolicyNode (None if incomplete).
    expanded_states: number of distinct states expanded.
    reason: human-readable explanation of the status.
    """
    status: PlanStatus
    root_action: Optional[str]
    stops_at_root: bool
    expected_utility: Optional[float]
    stop_utility: float
    policy_tree: Optional[PolicyNode]
    expanded_states: int
    max_state_expansions: int
    reason: str

    @property
    def is_optimal(self) -> bool:
        """True only when the search completed."""
        return self.status is PlanStatus.COMPLETE


class _LimitReached(Exception):
    pass


def _validate_settings(settings: PlannerSettings) -> None:
    lam, lim = settings.cost_penalty_lambda, settings.max_state_expansions
    if isinstance(lam, bool) or not isinstance(lam, (int, float)) or not math.isfinite(lam) or lam < 0:
        raise PlanningError(f"cost_penalty_lambda must be a finite number >= 0, got {lam!r}")
    if isinstance(lim, bool) or not isinstance(lim, int) or lim < 1:
        raise PlanningError(f"max_state_expansions must be an integer >= 1, got {lim!r}")


def solve_contingent_plan(initial_state: PlanningState, model: ModelConfig,
                          settings: PlannerSettings) -> PlanningResult:
    """Solve the contingent-planning recurrence exactly (see module docstring).

    Raises PlanningError for invalid settings/state and the Phase 1 errors
    (UnknownIdentifierError, BeliefError) for invalid beliefs.
    """
    _validate_settings(settings)
    _check_state_ids(initial_state, model)
    lam, limit = float(settings.cost_penalty_lambda), settings.max_state_expansions
    cache: dict[tuple, PolicyNode] = {}
    expansions = 0

    def solve(s: PlanningState, evidence: frozenset) -> PolicyNode:
        nonlocal expansions
        k = (evidence, s.performed, s.remaining_budget, s.permitted)
        hit = cache.get(k)
        if hit is not None:
            return hit
        if expansions >= limit:
            raise _LimitReached
        expansions += 1
        stop_v = terminal_value(s.beliefs)
        best_action: Optional[str] = None
        best_value = stop_v
        best_branches: tuple[OutcomeBranch, ...] = ()
        action_values: list[tuple[str, float]] = []
        for inv in available_investigations(s, model):
            probs = outcome_probability(s.beliefs, inv.id, model)
            branches: list[OutcomeBranch] = []
            expected = 0.0
            for o in inv.outcomes:
                p = probs[o]
                if p == 0.0:
                    branches.append(OutcomeBranch(o, 0.0, False, None, None))
                    continue
                nxt = advance_state(s, inv.id, o, model)
                child = solve(nxt, evidence | {(inv.id, o)})
                expected += p * child.value
                branches.append(OutcomeBranch(o, p, True, nxt.beliefs, child))
            q = -lam * inv.cost + expected
            action_values.append((inv.id, q))
            if q > best_value + TIE_EPS:
                best_action, best_value, best_branches = inv.id, q, tuple(branches)
        node = PolicyNode(s.beliefs, tuple(sorted(s.performed)), s.remaining_budget,
                          terminal_decision(s.beliefs, model), stop_v, best_action,
                          best_value, tuple(action_values), best_branches)
        cache[k] = node
        return node

    root_stop = terminal_value(initial_state.beliefs)
    try:
        root = solve(initial_state, frozenset())
    except _LimitReached:
        return PlanningResult(
            PlanStatus.INCOMPLETE_EXPANSION_LIMIT, None, False, None, root_stop, None,
            expansions, limit,
            f"State-expansion limit of {limit} reached after {expansions} expansions; the search did "
            f"not finish, so no optimal action or utility is reported (stopping would give "
            f"{root_stop:.6g}, which is only a lower bound).")
    return PlanningResult(
        PlanStatus.COMPLETE, root.action, root.action is None, root.value, root_stop, root,
        expansions, limit,
        f"Search completed with {expansions} expanded states; the result is optimal for the recurrence.")


# --------------------------------------------------------------------------- #
# Deterministic graph search
# --------------------------------------------------------------------------- #
S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


@dataclass(frozen=True)
class SearchProblem(Generic[S, A]):
    """Minimal graph-search contract.

    initial_state: hashable start state.
    successors(state) -> iterable of (action, next_state, edge_cost); order is
        the deterministic expansion order.  edge_cost must be finite and >= 0.
    is_goal(state) -> bool.
    """
    initial_state: S
    successors: Callable[[S], Iterable[tuple[A, S, float]]]
    is_goal: Callable[[S], bool]


def unit_cost_problem(initial_state: S, neighbors: Callable[[S], Iterable[S]],
                      is_goal: Callable[[S], bool]) -> SearchProblem[S, S]:
    """Convenience: unweighted problem (every edge costs 1; action = next state)."""
    return SearchProblem(initial_state, lambda s: [(n, n, 1.0) for n in neighbors(s)], is_goal)


class SearchStatus(str, Enum):
    FOUND = "found"
    UNREACHABLE = "unreachable"        # search space exhausted, no goal
    LIMIT_REACHED = "limit_reached"    # max_expansions hit; goal not proven unreachable


@dataclass(frozen=True)
class SearchResult(Generic[S, A]):
    """Outcome of a search.

    path: states from start to goal inclusive (empty unless FOUND).
    actions: actions along the path (len(path) - 1 items).
    path_cost: sum of edge costs on the path (None unless FOUND).
    expanded_states: number of expansions (a state's successors were generated);
        a goal state that is detected is not counted as expanded.
    expansion_order: states in the order expanded (A* may list a state twice).
    reopened_states: A* only: expansions of states that had been expanded before.
    """
    status: SearchStatus
    path: tuple[S, ...]
    actions: tuple[A, ...]
    path_cost: Optional[float]
    expanded_states: int
    expansion_order: tuple[S, ...]
    reopened_states: int = 0

    @property
    def found(self) -> bool:
        return self.status is SearchStatus.FOUND


def _check_cost(c: float) -> float:
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c < 0:
        raise ValueError(f"edge cost must be a finite number >= 0, got {c!r}")
    return float(c)


def _build(parent: dict, goal, order: list, status: SearchStatus, reopened: int = 0,
           cost: Optional[float] = None) -> SearchResult:
    states, actions, total = [goal], [], 0.0
    cur = goal
    while parent[cur] is not None:
        prev, act, c = parent[cur]
        states.append(prev)
        actions.append(act)
        total += c
        cur = prev
    states.reverse()
    actions.reverse()
    return SearchResult(status, tuple(states), tuple(actions),
                        total if cost is None else cost, len(order), tuple(order), reopened)


def _fail(status: SearchStatus, order: list, reopened: int = 0) -> SearchResult:
    return SearchResult(status, (), (), None, len(order), tuple(order), reopened)


def breadth_first_search(problem: SearchProblem, max_expansions: Optional[int] = None) -> SearchResult:
    """BFS: returns a path with the minimum NUMBER OF EDGES (edge costs ignored for
    the search but summed into path_cost).  Goal test happens when a state is
    generated.  Start == goal returns a one-state path, cost 0, 0 expansions."""
    start = problem.initial_state
    parent: dict = {start: None}
    order: list = []
    if problem.is_goal(start):
        return _build(parent, start, order, SearchStatus.FOUND)
    queue = deque([start])
    while queue:
        if max_expansions is not None and len(order) >= max_expansions:
            return _fail(SearchStatus.LIMIT_REACHED, order)
        state = queue.popleft()
        order.append(state)
        for action, nxt, cost in problem.successors(state):
            c = _check_cost(cost)
            if nxt in parent:
                continue
            parent[nxt] = (state, action, c)
            if problem.is_goal(nxt):
                return _build(parent, nxt, order, SearchStatus.FOUND)
            queue.append(nxt)
    return _fail(SearchStatus.UNREACHABLE, order)


def depth_first_search(problem: SearchProblem, max_expansions: Optional[int] = None) -> SearchResult:
    """DFS (preorder, first successor explored first); a visited set handles
    cycles.  Does not guarantee a shortest or cheapest path.  Goal test happens
    when a state is taken from the stack.  Start == goal: 0 expansions."""
    stack: list = [(problem.initial_state, None)]
    parent: dict = {}
    order: list = []
    while stack:
        state, via = stack.pop()
        if state in parent:
            continue
        parent[state] = via
        if problem.is_goal(state):
            return _build(parent, state, order, SearchStatus.FOUND)
        if max_expansions is not None and len(order) >= max_expansions:
            return _fail(SearchStatus.LIMIT_REACHED, order)
        order.append(state)
        succ = [(a, n, _check_cost(c)) for a, n, c in problem.successors(state)]
        for action, nxt, c in reversed(succ):
            if nxt not in parent:
                stack.append((nxt, (state, action, c)))
    return _fail(SearchStatus.UNREACHABLE, order)


def a_star_search(problem: SearchProblem, heuristic: Callable[[S], float],
                  max_expansions: Optional[int] = None) -> SearchResult:
    """A* with nonnegative edge costs and a nonnegative ``heuristic(state)``.

    Returns a minimum-cost path if the heuristic is admissible (never
    overestimates the true remaining cost) and the search completes; consistency
    is NOT required: a state reached by a cheaper path after being expanded is
    re-opened (counted in ``reopened_states``).  Stale queue entries are
    skipped, goal test happens when a state is popped, ties are FIFO.
    Start == goal: cost 0, 0 expansions.  ``heuristic`` is not checked for
    admissibility; negative or non-finite values raise ValueError.
    """
    def h(s) -> float:
        v = heuristic(s)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"heuristic must be finite and >= 0, got {v!r} for {s!r}")
        return float(v)

    start = problem.initial_state
    best_g: dict = {start: 0.0}
    parent: dict = {start: None}
    heap: list = [(h(start), 0, 0.0, start)]
    counter = 1
    order: list = []
    expanded_once: set = set()
    reopened = 0
    while heap:
        _, _, g, state = heapq.heappop(heap)
        if g > best_g[state]:
            continue  # stale entry: a cheaper path to this state was found later
        if problem.is_goal(state):
            return _build(parent, state, order, SearchStatus.FOUND, reopened, cost=g)
        if max_expansions is not None and len(order) >= max_expansions:
            return _fail(SearchStatus.LIMIT_REACHED, order, reopened)
        order.append(state)
        if state in expanded_once:
            reopened += 1
        expanded_once.add(state)
        for action, nxt, cost in problem.successors(state):
            ng = g + _check_cost(cost)
            if ng < best_g.get(nxt, math.inf):
                best_g[nxt] = ng
                parent[nxt] = (state, action, float(cost))
                heapq.heappush(heap, (ng + h(nxt), counter, ng, nxt))
                counter += 1
    return _fail(SearchStatus.UNREACHABLE, order, reopened)
