"""TestWise policy agent (Phase 2).

The agent sees only a ``PlanningState`` (beliefs, performed investigations,
remaining budget, permitted set) and the model; it never receives or infers a
hidden ground-truth hypothesis.

Policies
--------
fixed_order            first affordable, unperformed investigation in the
                       effective fixed order (``AgentSettings.fixed_order``
                       first, then any unlisted ones in model declaration order;
                       default = model declaration order).
cheapest_first         lowest cost; ties -> earlier in model declaration order.
greedy_information_gain  highest expected information gain PER COST UNIT
                       (bits / cost); ties -> higher total gain, then earlier
                       in model order.  Zero-cost tests: ratio is +inf if gain
                       > 0 else 0.0.
contingent_optimal     delegates to ``planner.solve_contingent_plan``; stops when
                       the planner says stopping has the greater expected
                       utility.  If the plan is incomplete (expansion limit) the
                       decision is ``no_decision`` - never a silent fallback.

Baselines stop when max posterior >= ``confidence_threshold`` or no permitted
affordable investigation remains.  Units are kept separate everywhere:
cost (cost units), entropy (bits), information gain (bits), information gain per
cost (bits per cost unit), expected net utility (classification-utility units).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from config import ModelConfig, PlannerSettings
from model import entropy, expected_information_gain, outcome_probability
from planner import (PlanningError, PlanningResult, PlanningState,
                     available_investigations, solve_contingent_plan,
                     terminal_value)

PolicyName = Literal["fixed_order", "cheapest_first", "greedy_information_gain", "contingent_optimal"]
POLICY_NAMES: tuple[str, ...] = ("fixed_order", "cheapest_first", "greedy_information_gain",
                                 "contingent_optimal")
StopReason = Literal["confidence_threshold_met", "no_affordable_investigations", "continue"]
DecisionKind = Literal["investigate", "stop", "no_decision"]

THRESHOLD_EPS: float = 1e-12


@dataclass(frozen=True)
class AgentSettings:
    """Agent settings.  Planner parameters (budget, lambda, confidence threshold,
    expansion limit) are NOT duplicated: they live in ``planner``.

    fixed_order: optional investigation IDs for the fixed_order policy.
    """
    planner: PlannerSettings
    fixed_order: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.planner, PlannerSettings):
            raise ValueError("AgentSettings.planner must be a PlannerSettings")
        if self.fixed_order is not None:
            object.__setattr__(self, "fixed_order", tuple(self.fixed_order))


@dataclass(frozen=True)
class CandidateAction:
    """An available investigation with its separately labelled metrics.

    cost: cost units.  expected_information_gain_bits: bits.
    information_gain_per_cost: bits per cost unit (inf if cost == 0 and gain > 0).
    outcome_probabilities: (outcome_id, P(outcome | beliefs)) pairs.
    """
    test_id: str
    cost: float
    expected_information_gain_bits: float
    information_gain_per_cost: float
    outcome_probabilities: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class StopDecision:
    """Result of ``should_stop``.  max_posterior = A(b), a probability."""
    stop: bool
    reason: StopReason
    max_posterior: float
    confidence_threshold: float


@dataclass(frozen=True)
class Explanation:
    """Structured explanation; every number carries its unit in its name.

    entropy_bits: Shannon entropy of current beliefs (bits).
    max_posterior: A(b) = probability of the best-guess hypothesis (probability).
    remaining_budget: cost units.
    candidates: all available investigations with cost / gain (bits) / gain per cost.
    selected: the chosen candidate (None when stopping or undecided).
    stop_reason: why the policy stopped (None otherwise).
    planner_status, expected_net_utility, stop_utility, planner_expanded_states:
        contingent_optimal only.  Utilities are in classification-utility units
        (probability of a correct classification minus lambda * cost); they are
        not bits.
    """
    policy: str
    summary: str
    entropy_bits: float
    max_posterior: float
    remaining_budget: float
    candidates: tuple[CandidateAction, ...]
    selected: Optional[CandidateAction] = None
    stop_reason: Optional[str] = None
    planner_status: Optional[str] = None
    expected_net_utility: Optional[float] = None
    stop_utility: Optional[float] = None
    planner_expanded_states: Optional[int] = None


@dataclass(frozen=True)
class Decision:
    """kind: "investigate" (test_id set), "stop", or "no_decision" (contingent
    plan incomplete; the caller must not treat it as a stop)."""
    kind: DecisionKind
    test_id: Optional[str]
    policy: str
    explanation: Explanation


def get_candidate_actions(state: PlanningState, model: ModelConfig) -> list[CandidateAction]:
    """Affordable, permitted, unperformed investigations in model order, with metrics."""
    out = []
    for inv in available_investigations(state, model):
        gain = expected_information_gain(state.beliefs, inv.id, model)
        if inv.cost > 0:
            ratio = gain / inv.cost
        else:
            ratio = math.inf if gain > 0 else 0.0
        probs = outcome_probability(state.beliefs, inv.id, model)
        out.append(CandidateAction(inv.id, float(inv.cost), gain, ratio,
                                   tuple((o, probs[o]) for o in inv.outcomes)))
    return out


def should_stop(state: PlanningState, settings: AgentSettings,
                model: Optional[ModelConfig] = None) -> StopDecision:
    """Baseline stopping rule.

    Stops if max posterior >= confidence_threshold.  If ``model`` is supplied
    (an extension of the minimal signature, needed to know costs) it also stops
    when no permitted affordable investigation remains.
    """
    thr = settings.planner.confidence_threshold
    a = terminal_value(state.beliefs)
    if a >= thr - THRESHOLD_EPS:
        return StopDecision(True, "confidence_threshold_met", a, thr)
    if model is not None and not available_investigations(state, model):
        return StopDecision(True, "no_affordable_investigations", a, thr)
    return StopDecision(False, "continue", a, thr)


def _effective_fixed_order(model: ModelConfig, settings: AgentSettings) -> list[str]:
    declared = [i.id for i in model.investigations]
    if settings.fixed_order is None:
        return declared
    order = list(settings.fixed_order)
    if len(set(order)) != len(order) or set(order) - set(declared):
        raise ValueError(f"fixed_order must contain unique known investigation IDs, got {order}")
    return order + [t for t in declared if t not in order]


def _base_explanation(policy: str, state: PlanningState, model: ModelConfig,
                      candidates: list[CandidateAction], **kw) -> Explanation:
    return Explanation(policy=policy, summary=kw.pop("summary", ""), entropy_bits=entropy(state.beliefs),
                       max_posterior=terminal_value(state.beliefs),
                       remaining_budget=state.remaining_budget, candidates=tuple(candidates), **kw)


def choose_action(state: PlanningState, model: ModelConfig, policy: PolicyName,
                  settings: AgentSettings) -> Decision:
    """Choose the next action under ``policy`` (see module docstring)."""
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICY_NAMES}")
    if not isinstance(settings, AgentSettings):
        raise ValueError("settings must be an AgentSettings")
    candidates = get_candidate_actions(state, model)

    if policy == "contingent_optimal":
        return _contingent(state, model, settings, candidates)

    sd = should_stop(state, settings, model)
    if sd.stop:
        ex = _base_explanation(policy, state, model, candidates, stop_reason=sd.reason, summary=(
            f"Stop: {sd.reason.replace('_', ' ')} (max posterior {sd.max_posterior:.4f}, "
            f"threshold {sd.confidence_threshold:.4f})."))
        return Decision("stop", None, policy, ex)

    by_id = {c.test_id: c for c in candidates}
    index = {i.id: n for n, i in enumerate(model.investigations)}
    if policy == "fixed_order":
        sel = next(by_id[t] for t in _effective_fixed_order(model, settings) if t in by_id)
        why = "first affordable unperformed investigation in the fixed order"
    elif policy == "cheapest_first":
        sel = min(candidates, key=lambda c: (c.cost, index[c.test_id]))
        why = "lowest cost among affordable unperformed investigations"
    else:  # greedy_information_gain
        sel = min(candidates, key=lambda c: (-c.information_gain_per_cost,
                                             -c.expected_information_gain_bits, index[c.test_id]))
        why = "highest expected information gain per cost unit"
    ex = _base_explanation(policy, state, model, candidates, selected=sel, summary=(
        f"Chose {sel.test_id} ({why}): cost {sel.cost:g}, expected information gain "
        f"{sel.expected_information_gain_bits:.4f} bits, {sel.information_gain_per_cost:.4f} bits per cost unit."))
    return Decision("investigate", sel.test_id, policy, ex)


def _contingent(state: PlanningState, model: ModelConfig, settings: AgentSettings,
                candidates: list[CandidateAction]) -> Decision:
    res: PlanningResult = solve_contingent_plan(state, model, settings.planner)
    common = dict(planner_status=res.status.value, stop_utility=res.stop_utility,
                  planner_expanded_states=res.expanded_states)
    if not res.is_optimal:
        ex = _base_explanation("contingent_optimal", state, model, candidates,
                               summary="No decision: " + res.reason, **common)
        return Decision("no_decision", None, "contingent_optimal", ex)
    if res.stops_at_root:
        ex = _base_explanation("contingent_optimal", state, model, candidates,
                               stop_reason="planner_stop_optimal", expected_net_utility=res.expected_utility,
                               summary=(f"Stop: stopping (utility {res.stop_utility:.4f}) is at least as good "
                                        f"as any investigation plan (best expected net utility "
                                        f"{res.expected_utility:.4f}, classification-utility units)."),
                               **common)
        return Decision("stop", None, "contingent_optimal", ex)
    sel = next(c for c in candidates if c.test_id == res.root_action)
    ex = _base_explanation("contingent_optimal", state, model, candidates, selected=sel,
                           expected_net_utility=res.expected_utility,
                           summary=(f"Chose {sel.test_id}: optimal first step of the contingent plan, expected "
                                    f"net utility {res.expected_utility:.4f} (classification-utility units) vs "
                                    f"{res.stop_utility:.4f} for stopping now. For reference, this test costs "
                                    f"{sel.cost:g} and has expected information gain "
                                    f"{sel.expected_information_gain_bits:.4f} bits."),
                           **common)
    return Decision("investigate", sel.test_id, "contingent_optimal", ex)
