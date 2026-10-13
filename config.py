"""TestWise configuration (Phase 1).

All probabilities here are ILLUSTRATIVE SYNTHETIC values for an educational
simulation. They are not medical evidence and carry no clinical meaning.
No UI framework is imported anywhere in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

# Absolute tolerance for "sums to one" checks on priors and likelihood rows.
PROBABILITY_TOLERANCE: float = 1e-9


class ConfigError(ValueError):
    """Raised when configuration or model data is malformed."""


@dataclass(frozen=True)
class Hypothesis:
    """A candidate explanation. `id` is the canonical stable identifier."""
    id: str
    label: str


@dataclass(frozen=True)
class Investigation:
    """A test that can be performed.

    id: canonical stable identifier.
    label: human-readable name.
    cost: nonnegative cost in abstract units.
    outcomes: ordered canonical outcome IDs (unique within this investigation).
    likelihoods: likelihoods[hypothesis_id][outcome_id] = P(outcome | hypothesis).
        Every hypothesis must have a row over exactly `outcomes`, summing to 1.
    """
    id: str
    label: str
    cost: float
    outcomes: tuple[str, ...]
    likelihoods: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        if isinstance(self.outcomes, (str, bytes)) or not isinstance(self.outcomes, (tuple, list)):
            raise ConfigError(
                f"investigation '{self.id}': outcomes must be a tuple or list of strings, got {type(self.outcomes).__name__}"
            )
        object.__setattr__(self, "outcomes", tuple(self.outcomes))

        if not isinstance(self.likelihoods, Mapping):
            raise ConfigError(
                f"investigation '{self.id}': likelihoods must be a mapping, got {type(self.likelihoods).__name__}"
            )
        frozen_likelihoods = {}
        for h, row in self.likelihoods.items():
            if not isinstance(row, Mapping):
                raise ConfigError(
                    f"investigation '{self.id}': likelihood row for '{h}' must be a mapping, got {type(row).__name__}"
                )
            frozen_likelihoods[h] = MappingProxyType(dict(row))
        object.__setattr__(self, "likelihoods", MappingProxyType(frozen_likelihoods))


@dataclass(frozen=True)
class ModelConfig:
    """The Bayesian model: hypotheses, prior P(h), and investigations."""
    hypotheses: tuple[Hypothesis, ...]
    priors: Mapping[str, float]
    investigations: tuple[Investigation, ...]

    def __post_init__(self) -> None:
        if isinstance(self.hypotheses, (str, bytes)) or not isinstance(self.hypotheses, (tuple, list)):
            raise ConfigError(
                f"hypotheses must be a tuple or list, got {type(self.hypotheses).__name__}"
            )
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))

        if not isinstance(self.priors, Mapping):
            raise ConfigError(
                f"priors must be a mapping, got {type(self.priors).__name__}"
            )
        object.__setattr__(self, "priors", MappingProxyType(dict(self.priors)))

        if isinstance(self.investigations, (str, bytes)) or not isinstance(self.investigations, (tuple, list)):
            raise ConfigError(
                f"investigations must be a tuple or list, got {type(self.investigations).__name__}"
            )
        object.__setattr__(self, "investigations", tuple(self.investigations))

    def hypothesis_ids(self) -> tuple[str, ...]:
        return tuple(h.id for h in self.hypotheses)

    def investigation(self, test_id: str) -> Investigation:
        for inv in self.investigations:
            if inv.id == test_id:
                return inv
        raise KeyError(test_id)


@dataclass(frozen=True)
class PlannerSettings:
    """Planner configuration parameters."""
    initial_budget: float = 10.0
    confidence_threshold: float = 0.90
    cost_penalty_lambda: float = 0.05
    max_state_expansions: int = 10_000


@dataclass(frozen=True)
class EvaluationSettings:
    """Evaluation configuration parameters."""
    num_episodes: int = 200


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""
    model: ModelConfig
    app_title: str = "TestWise"
    schema_version: int = 1
    planner: PlannerSettings = field(default_factory=PlannerSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    random_seed: int = 42


def build_default_model() -> ModelConfig:
    """Illustrative 4-hypothesis, 6-investigation model (synthetic values)."""
    hyps = (
        Hypothesis("cond_a", "Condition A"),
        Hypothesis("cond_b", "Condition B"),
        Hypothesis("cond_c", "Condition C"),
        Hypothesis("cond_d", "Condition D"),
    )
    priors = {"cond_a": 0.40, "cond_b": 0.30, "cond_c": 0.20, "cond_d": 0.10}

    def inv(id_, label, cost, outcomes, rows):
        lik = {h.id: dict(zip(outcomes, row)) for h, row in zip(hyps, rows)}
        return Investigation(id_, label, cost, tuple(outcomes), lik)

    invs = (
        inv("quick_screen", "Quick screen", 1.0, ("negative", "positive"),
            [(0.70, 0.30), (0.40, 0.60), (0.30, 0.70), (0.20, 0.80)]),
        inv("marker_panel", "Marker panel", 2.0, ("low", "normal", "high"),
            [(0.05, 0.25, 0.70), (0.20, 0.60, 0.20),
             (0.60, 0.30, 0.10), (0.25, 0.50, 0.25)]),
        inv("imaging", "Imaging", 3.0, ("clear", "ambiguous", "positive"),
            [(0.80, 0.15, 0.05), (0.10, 0.30, 0.60),
             (0.20, 0.50, 0.30), (0.10, 0.20, 0.70)]),
        inv("specialist_review", "Specialist review", 5.0, ("negative", "positive"),
            [(0.95, 0.05), (0.10, 0.90), (0.90, 0.10), (0.50, 0.50)]),
        inv("history_check", "History check", 2.0, ("absent", "present"),
            [(0.60, 0.40), (0.55, 0.45), (0.50, 0.50), (0.45, 0.55)]),
        inv("extended_panel", "Extended panel", 4.0,
            ("normal", "abnormal_x", "abnormal_y"),
            [(0.70, 0.20, 0.10), (0.10, 0.60, 0.30),
             (0.15, 0.25, 0.60), (0.30, 0.35, 0.35)]),
    )
    return ModelConfig(hyps, priors, invs)


def build_default_app_config() -> AppConfig:
    """Default AppConfig wrapping the default synthetic model."""
    return AppConfig(model=build_default_model())


def validate_config(config: AppConfig) -> None:
    """Validate an AppConfig; raise ConfigError on any problem. Never repairs."""
    from model import validate_model  # local import avoids a circular import

    if not isinstance(config, AppConfig):
        raise ConfigError("config must be an AppConfig")
    if not isinstance(config.app_title, str) or not config.app_title.strip():
        raise ConfigError("app_title must be a non-empty string")
    if not isinstance(config.schema_version, int) or isinstance(config.schema_version, bool) \
            or config.schema_version < 1:
        raise ConfigError("schema_version must be a positive integer")

    validate_model(config.model)

    if not isinstance(config.planner, PlannerSettings):
        raise ConfigError("planner must be a PlannerSettings")
    p = config.planner
    cheapest = min((i.cost for i in config.model.investigations), default=0.0)
    if (isinstance(p.initial_budget, bool)
            or not isinstance(p.initial_budget, (int, float))
            or not math.isfinite(p.initial_budget)
            or p.initial_budget < cheapest):
        raise ConfigError(
            f"initial_budget must be a finite number >= cheapest test ({cheapest}), got {p.initial_budget!r}"
        )
    if (isinstance(p.confidence_threshold, bool)
            or not isinstance(p.confidence_threshold, (int, float))
            or not math.isfinite(p.confidence_threshold)
            or not (0.0 < p.confidence_threshold <= 1.0)):
        raise ConfigError(
            f"confidence_threshold must be a finite number in (0, 1], got {p.confidence_threshold!r}"
        )
    if (isinstance(p.cost_penalty_lambda, bool)
            or not isinstance(p.cost_penalty_lambda, (int, float))
            or not math.isfinite(p.cost_penalty_lambda)
            or p.cost_penalty_lambda < 0.0):
        raise ConfigError(
            f"cost_penalty_lambda must be a finite number >= 0, got {p.cost_penalty_lambda!r}"
        )
    if (isinstance(p.max_state_expansions, bool)
            or not isinstance(p.max_state_expansions, int)
            or p.max_state_expansions < 1):
        raise ConfigError(
            f"max_state_expansions must be an integer >= 1, got {p.max_state_expansions!r}"
        )

    if not isinstance(config.evaluation, EvaluationSettings):
        raise ConfigError("evaluation must be an EvaluationSettings")
    e = config.evaluation
    if (isinstance(e.num_episodes, bool)
            or not isinstance(e.num_episodes, int)
            or not (1 <= e.num_episodes <= 1_000_000)):
        raise ConfigError(
            f"num_episodes must be an integer in [1, 1000000], got {e.num_episodes!r}"
        )

    if (isinstance(config.random_seed, bool)
            or not isinstance(config.random_seed, int)
            or config.random_seed < 0):
        raise ConfigError(
            f"random_seed must be an integer >= 0, got {config.random_seed!r}"
        )
