"""TestWise Bayesian model (Phase 1). No UI dependencies.

Conventions: probabilities in [0, 1]; entropy and information gain in bits
(log base 2). Hypothesis and outcome IDs are the canonical IDs from config.
Invalid input is rejected, never silently normalized or repaired.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from config import (PROBABILITY_TOLERANCE, ConfigError, Investigation,
                    ModelConfig)

# Smallest positive *normal* double. Below this, products of probabilities lose
# precision (subnormals) or underflow, so update_beliefs switches to a
# log-space computation instead of dividing by a degraded evidence value.
_NORMAL_MIN: float = sys.float_info.min


class UnknownIdentifierError(ValueError):
    """Unknown hypothesis, investigation, or outcome ID."""


class ImpossibleObservationError(ValueError):
    """Observation has EXACTLY zero probability: no hypothesis with P(h) > 0
    assigns it P(o | h) > 0. Tiny but positive probabilities are not errors."""


class BeliefError(ValueError):
    """Malformed belief state."""


@dataclass(frozen=True)
class BeliefState:
    """Belief distribution over hypotheses (FROZEN CONTRACT).

    Access: ``beliefs.probabilities[hypothesis_id]`` -> float, keyed by the
    canonical hypothesis IDs of the ModelConfig. All later modules must pass
    and return this type, never bare dicts.

    Immutability: the constructor stores a read-only copy (MappingProxyType),
    so later changes to the caller's dict have no effect and item assignment
    raises TypeError. Instances are hashable (hash of the sorted items), so
    they may be used as memoization keys. To "modify", build a new BeliefState.
    Values are checked against a model by the functions in this module.
    """
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, Mapping):
            raise BeliefError(f"probabilities must be a mapping, got {type(self.probabilities).__name__}")
        object.__setattr__(self, "probabilities", MappingProxyType(dict(self.probabilities)))

    def __hash__(self) -> int:
        return hash(frozenset(self.probabilities.items()))


def initial_beliefs(model: ModelConfig) -> BeliefState:
    """BeliefState equal to the model's priors (a copy; the model is unchanged)."""
    return BeliefState(dict(model.priors))


def _is_prob(x: object) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and 0.0 <= x <= 1.0)


def _check_distribution(name: str, dist: Mapping[str, float], keys: tuple[str, ...]) -> None:
    if set(dist.keys()) != set(keys):
        raise ConfigError(f"{name}: keys {sorted(dist)} do not match expected {sorted(keys)}")
    for k, v in dist.items():
        if not _is_prob(v):
            raise ConfigError(f"{name}: probability for '{k}' must be a finite number in [0, 1], got {v!r}")
    total = math.fsum(dist.values())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ConfigError(f"{name}: probabilities sum to {total!r}, expected 1 (tol {PROBABILITY_TOLERANCE})")


def _check_unique_ids(kind: str, ids: list[str]) -> None:
    for i in ids:
        if not isinstance(i, str) or not i.strip():
            raise ConfigError(f"{kind} ID must be a non-empty string, got {i!r}")
    if len(set(ids)) != len(ids):
        raise ConfigError(f"duplicate {kind} IDs: {ids}")


def validate_model(model: ModelConfig) -> None:
    """Raise ConfigError if the model is malformed. Does not modify the model."""
    if not isinstance(model, ModelConfig):
        raise ConfigError("model must be a ModelConfig")
    hids = list(model.hypothesis_ids())
    if not hids:
        raise ConfigError("model must have at least one hypothesis")
    _check_unique_ids("hypothesis", hids)
    if not model.investigations:
        raise ConfigError("model must have at least one investigation")
    _check_unique_ids("investigation", [i.id for i in model.investigations])
    _check_distribution("priors", model.priors, tuple(hids))
    for inv in model.investigations:
        if not (isinstance(inv.cost, (int, float)) and not isinstance(inv.cost, bool)
                and math.isfinite(inv.cost) and inv.cost >= 0):
            raise ConfigError(f"investigation '{inv.id}': cost must be a finite number >= 0, got {inv.cost!r}")
        if len(inv.outcomes) < 1:
            raise ConfigError(f"investigation '{inv.id}': needs at least one outcome")
        _check_unique_ids(f"outcome of '{inv.id}'", list(inv.outcomes))
        if set(inv.likelihoods.keys()) != set(hids):
            raise ConfigError(f"investigation '{inv.id}': likelihood rows {sorted(inv.likelihoods)} "
                              f"do not match hypotheses {sorted(hids)}")
        for h in hids:
            _check_distribution(f"likelihood[{inv.id}][{h}]", inv.likelihoods[h], inv.outcomes)


def _validate_beliefs(beliefs: BeliefState, model: ModelConfig) -> None:
    probs = beliefs.probabilities
    hids = model.hypothesis_ids()
    unknown = set(probs) - set(hids)
    if unknown:
        raise UnknownIdentifierError(f"unknown hypothesis IDs in beliefs: {sorted(unknown)}")
    try:
        _check_distribution("beliefs", probs, hids)
    except ConfigError as e:
        raise BeliefError(str(e)) from None


def _get_investigation(model: ModelConfig, test_id: str) -> Investigation:
    try:
        return model.investigation(test_id)
    except KeyError:
        raise UnknownIdentifierError(f"unknown investigation ID: {test_id!r}") from None


def outcome_probability(beliefs: BeliefState, test_id: str, model: ModelConfig) -> dict[str, float]:
    """Return P(outcome) = sum_h P(h) P(outcome | h) for each outcome of the test."""
    inv = _get_investigation(model, test_id)
    _validate_beliefs(beliefs, model)
    return {o: math.fsum(beliefs.probabilities[h] * inv.likelihoods[h][o]
                         for h in model.hypothesis_ids())
            for o in inv.outcomes}


def update_beliefs(beliefs: BeliefState, test_id: str, outcome: str, model: ModelConfig) -> BeliefState:
    """Bayes' rule: P(h | o) = P(h) P(o | h) / P(o).

    Raises UnknownIdentifierError for unknown test/outcome and
    ImpossibleObservationError only if P(o) is exactly zero. Tiny positive
    probabilities are accepted; below the smallest normal double the posterior
    is computed in log space (see _NORMAL_MIN).
    """
    inv = _get_investigation(model, test_id)
    if outcome not in inv.outcomes:
        raise UnknownIdentifierError(f"unknown outcome {outcome!r} for investigation {test_id!r}; "
                                     f"valid: {list(inv.outcomes)}")
    _validate_beliefs(beliefs, model)
    hids = model.hypothesis_ids()
    prior = {h: beliefs.probabilities[h] for h in hids}
    like = {h: inv.likelihoods[h][outcome] for h in hids}
    joint = {h: prior[h] * like[h] for h in hids}
    evidence = math.fsum(joint.values())
    if evidence >= _NORMAL_MIN:  # ordinary case: direct Bayes' rule
        return BeliefState({h: j / evidence for h, j in joint.items()})
    # P(o) is exactly zero iff no hypothesis has both P(h) > 0 and P(o|h) > 0.
    support = [h for h in hids if prior[h] > 0.0 and like[h] > 0.0]
    if not support:
        raise ImpossibleObservationError(
            f"outcome {outcome!r} of {test_id!r} has probability exactly 0 under current beliefs")
    # Possible but tiny (products subnormal or underflowed): compute in log
    # space, shifting by the maximum, so the posterior stays accurate.
    logs = {h: math.log(prior[h]) + math.log(like[h]) for h in support}
    top = max(logs.values())
    weights = {h: math.exp(v - top) for h, v in logs.items()}
    total = math.fsum(weights.values())
    return BeliefState({h: weights.get(h, 0.0) / total for h in hids})


def _entropy_of(probs) -> float:
    return -math.fsum(p * math.log2(p) for p in probs if p > 0.0)


def entropy(beliefs: BeliefState) -> float:
    """Shannon entropy in bits, H = -sum p log2 p, with 0 log 0 = 0."""
    probs = list(beliefs.probabilities.values())
    if any(not _is_prob(p) for p in probs):
        raise BeliefError("beliefs must be finite probabilities in [0, 1]")
    if abs(math.fsum(probs) - 1.0) > PROBABILITY_TOLERANCE:
        raise BeliefError("beliefs must sum to 1")
    return max(0.0, _entropy_of(probs))  # guards against -0.0


def expected_information_gain(beliefs: BeliefState, test_id: str, model: ModelConfig) -> float:
    """EIG in bits: H(prior) - sum_o P(o) H(posterior | o); mutual information I(H; O)."""
    p_out = outcome_probability(beliefs, test_id, model)
    expected_post = 0.0
    for o, po in p_out.items():
        if po == 0.0:
            # Contributes nothing: either impossible, or so small (< ~1e-308,
            # underflowed) that po * H(posterior) is below double precision.
            continue
        expected_post += po * entropy(update_beliefs(beliefs, test_id, o, model))
    gain = entropy(beliefs) - expected_post
    if gain < -PROBABILITY_TOLERANCE:
        raise ArithmeticError(f"negative information gain {gain!r}; model inconsistent")
    return max(0.0, gain)
