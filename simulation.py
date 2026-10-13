"""TestWise simulation engine (Phase 0C).

This module implements sequential simulation of Bayesian investigation episodes.
It maintains a strict information boundary: ground truth hypotheses exist only
in the simulator's internal state and offline episode results; they are NEVER
exposed to the agent or included in public planning states or decisions.

Reproducibility is guaranteed via dedicated SHA-256 derived seed streams for:
1. Ground truth sampling: P(hypothesis) from model priors.
2. Potential outcome sampling: P(outcome | hypothesis) for each (seed, hypothesis, test_id).
This ensures counterfactual consistency: a test produces the exact same outcome
for a given synthetic case regardless of what other tests were performed or their order.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, Sequence

from config import Investigation, ModelConfig
from model import BeliefState, entropy, initial_beliefs, validate_model
from planner import (
    PlanningState,
    advance_state,
    available_investigations,
    is_affordable,
    terminal_decision,
    terminal_value,
    _exact,
)
from agent import (
    POLICY_NAMES,
    THRESHOLD_EPS,
    AgentSettings,
    Decision,
    PolicyName,
    choose_action,
)


class SimulationError(ValueError):
    """Raised when simulation parameters or state transitions are invalid."""


def _derive_seed(seed: int, domain: str, *keys: str) -> int:
    """Derive a deterministic 64-bit integer seed using SHA-256."""
    payload = f"{domain}:{seed}:" + ":".join(keys)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_hidden_hypothesis(model: ModelConfig, seed: int) -> str:
    """Sample a hidden hypothesis from model priors using a dedicated seed stream."""
    stream_seed = _derive_seed(seed, "hidden_hypothesis")
    rng = random.Random(stream_seed)
    hids = list(model.hypothesis_ids())
    weights = [model.priors[h] for h in hids]
    return rng.choices(hids, weights=weights, k=1)[0]


def sample_potential_outcome(
    model: ModelConfig,
    seed: int,
    hidden_hypothesis: str,
    test_id: str,
) -> str:
    """Sample an outcome from P(outcome | hidden_hypothesis) using a stable dedicated seed.

    The seed is derived strictly from (seed, hidden_hypothesis, test_id), ensuring that
    for a given episode and hidden truth, test_id produces the exact same outcome
    regardless of what other tests were performed or the order they were performed in.
    """
    inv = model.investigation(test_id)
    if hidden_hypothesis not in inv.likelihoods:
        raise SimulationError(f"Hypothesis {hidden_hypothesis!r} not in investigation {test_id!r}")

    stream_seed = _derive_seed(seed, "potential_outcome", hidden_hypothesis, test_id)
    rng = random.Random(stream_seed)
    outcomes = list(inv.outcomes)
    weights = [inv.likelihoods[hidden_hypothesis][o] for o in outcomes]
    return rng.choices(outcomes, weights=weights, k=1)[0]


@dataclass(frozen=True)
class InvestigationRecord:
    """Immutable record of an investigation executed during an episode.

    Contains no hidden ground truth.
    """
    step_index: int
    test_id: str
    outcome: str
    cost: float
    prior_beliefs: BeliefState
    posterior_beliefs: BeliefState
    remaining_budget: float
    decision: Decision


@dataclass(frozen=True)
class EpisodeResult:
    """Immutable final result of a simulation episode.

    Contains all public trajectory data, metrics, and ground truth for offline evaluation.
    """
    seed: int
    policy: str
    hidden_hypothesis: str
    initial_budget: float
    total_cost: float
    remaining_budget: float
    history: tuple[InvestigationRecord, ...]
    final_beliefs: BeliefState
    predicted_hypothesis: str
    final_confidence: float
    final_entropy: float
    is_resolved: bool
    is_completed: bool
    is_correct: bool
    termination_reason: str
    stop_reason: Optional[str]
    performed_investigations: tuple[str, ...]
    planner_expansion_counts: tuple[int, ...]
    confidence_threshold: float


class Episode:
    """Active simulation episode tracking state transitions.

    Maintains internal hidden truth strictly isolated from the agent-facing state.
    """

    def __init__(
        self,
        model: ModelConfig,
        seed: int,
        budget: float,
        hidden_hypothesis: str,
        permitted: Optional[Iterable[str]] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self.seed = seed
        self.initial_budget = float(budget)
        self._hidden_hypothesis = hidden_hypothesis
        self.state = PlanningState(
            beliefs=initial_beliefs(model),
            performed=frozenset(),
            remaining_budget=float(budget),
            permitted=frozenset(permitted) if permitted is not None else None,
        )
        self.history: list[InvestigationRecord] = []
        self.is_terminal: bool = False
        self.termination_reason: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self.policy: Optional[str] = None
        self.max_steps = len(model.investigations) if max_steps is None else max_steps

    @property
    def remaining_budget(self) -> float:
        return self.state.remaining_budget

    @property
    def current_beliefs(self) -> BeliefState:
        return self.state.beliefs

    @property
    def performed(self) -> frozenset[str]:
        return self.state.performed

    def to_result(
        self,
        model: ModelConfig,
        settings: AgentSettings,
        policy: Optional[str] = None,
    ) -> EpisodeResult:
        """Convert a terminated episode into an immutable EpisodeResult."""
        if not self.is_terminal:
            raise SimulationError("Cannot produce EpisodeResult for an episode that is not yet terminal")

        pol = policy or self.policy or "unknown"
        exact_total_cost = sum((_exact(rec.cost) for rec in self.history), Fraction(0))
        total_cost = float(exact_total_cost)

        final_beliefs = self.state.beliefs
        pred = terminal_decision(final_beliefs, model)
        conf = terminal_value(final_beliefs)
        ent = entropy(final_beliefs)
        thr = settings.planner.confidence_threshold

        resolved = conf >= thr - THRESHOLD_EPS
        completed = self.termination_reason != "planner_incomplete"
        correct = pred == self._hidden_hypothesis

        performed_ids = tuple(rec.test_id for rec in self.history)
        planner_expansions = tuple(
            rec.decision.explanation.planner_expanded_states
            for rec in self.history
            if rec.decision.explanation.planner_expanded_states is not None
        )

        return EpisodeResult(
            seed=self.seed,
            policy=pol,
            hidden_hypothesis=self._hidden_hypothesis,
            initial_budget=self.initial_budget,
            total_cost=total_cost,
            remaining_budget=self.state.remaining_budget,
            history=tuple(self.history),
            final_beliefs=final_beliefs,
            predicted_hypothesis=pred,
            final_confidence=conf,
            final_entropy=ent,
            is_resolved=resolved,
            is_completed=completed,
            is_correct=correct,
            termination_reason=self.termination_reason or "unknown",
            stop_reason=self.stop_reason,
            performed_investigations=performed_ids,
            planner_expansion_counts=planner_expansions,
            confidence_threshold=thr,
        )


def create_episode(
    model: ModelConfig,
    seed: int,
    budget: float,
    permitted: Optional[Iterable[str]] = None,
    max_steps: Optional[int] = None,
) -> Episode:
    """Create a reproducible simulation episode with a sampled hidden hypothesis.

    Validates model, seed, budget, and permitted options.
    """
    validate_model(model)

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SimulationError(f"seed must be a non-negative integer, got {seed!r}")

    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget < 0
    ):
        raise SimulationError(f"budget must be a finite number >= 0, got {budget!r}")

    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise SimulationError(f"max_steps must be a positive integer, got {max_steps!r}")

    if permitted is not None:
        known = {i.id for i in model.investigations}
        bad = set(permitted) - known
        if bad:
            raise SimulationError(f"unknown permitted investigation IDs: {sorted(bad)}")

    hidden_hypothesis = sample_hidden_hypothesis(model, seed)
    return Episode(
        model=model,
        seed=seed,
        budget=budget,
        hidden_hypothesis=hidden_hypothesis,
        permitted=permitted,
        max_steps=max_steps,
    )


def step_episode(
    episode: Episode,
    model: ModelConfig,
    policy: PolicyName,
    settings: AgentSettings,
) -> Episode:
    """Execute at most one agent action and return the updated episode.

    The agent receives only the public PlanningState; it never sees the hidden truth.
    """
    if episode.is_terminal:
        raise SimulationError("Cannot step an episode that is already terminal")

    if policy not in POLICY_NAMES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {POLICY_NAMES}")

    if not isinstance(settings, AgentSettings):
        raise ValueError(f"settings must be an AgentSettings, got {type(settings).__name__}")

    episode.policy = policy

    # Guard against exceeding maximum allowed investigation steps
    if len(episode.history) >= episode.max_steps:
        episode.is_terminal = True
        episode.termination_reason = "step_limit"
        episode.stop_reason = "step_limit"
        return episode

    decision = choose_action(episode.state, model, policy, settings)

    if decision.kind == "stop":
        episode.is_terminal = True
        if decision.explanation.stop_reason == "no_affordable_investigations":
            episode.termination_reason = "no_legal_actions"
        else:
            episode.termination_reason = "policy_stop"
        episode.stop_reason = decision.explanation.stop_reason
        return episode

    if decision.kind == "no_decision":
        episode.is_terminal = True
        episode.termination_reason = "planner_incomplete"
        episode.stop_reason = decision.explanation.stop_reason or "incomplete_expansion_limit"
        return episode

    if decision.kind == "investigate":
        test_id = decision.test_id
        if test_id is None:
            raise SimulationError("Decision kind 'investigate' must specify test_id")

        inv = model.investigation(test_id)
        if test_id in episode.state.performed:
            raise SimulationError(f"Investigation {test_id!r} was already performed")
        if episode.state.permitted is not None and test_id not in episode.state.permitted:
            raise SimulationError(f"Investigation {test_id!r} is not in permitted set")
        if not is_affordable(inv.cost, episode.state.remaining_budget):
            raise SimulationError(
                f"Investigation {test_id!r} (cost {inv.cost}) exceeds remaining budget {episode.state.remaining_budget}"
            )

        outcome = sample_potential_outcome(model, episode.seed, episode._hidden_hypothesis, test_id)
        new_state = advance_state(episode.state, test_id, outcome, model)

        record = InvestigationRecord(
            step_index=len(episode.history),
            test_id=test_id,
            outcome=outcome,
            cost=float(inv.cost),
            prior_beliefs=episode.state.beliefs,
            posterior_beliefs=new_state.beliefs,
            remaining_budget=new_state.remaining_budget,
            decision=decision,
        )
        episode.history.append(record)
        episode.state = new_state

        if len(episode.history) >= episode.max_steps:
            episode.is_terminal = True
            episode.termination_reason = "step_limit"
            episode.stop_reason = "step_limit"

        return episode

    raise SimulationError(f"Unexpected decision kind {decision.kind!r}")


def run_episode(
    model: ModelConfig,
    policy: PolicyName,
    settings: AgentSettings,
    seed: int,
    budget: float,
    permitted: Optional[Iterable[str]] = None,
    max_steps: Optional[int] = None,
) -> EpisodeResult:
    """Execute a complete simulation episode and return an immutable EpisodeResult."""
    episode = create_episode(
        model=model,
        seed=seed,
        budget=budget,
        permitted=permitted,
        max_steps=max_steps,
    )
    while not episode.is_terminal:
        step_episode(episode, model, policy, settings)
    return episode.to_result(model, settings, policy)
