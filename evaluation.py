"""TestWise evaluation engine (Phase 0D).

This module implements deterministic, reproducible comparative evaluation of
the four TestWise decision policies on common synthetic cases.

Fair comparison design:
1. Common cases: for a given evaluation seed and case index, all policies receive
   the exact same synthetic case (same hidden hypothesis and same potential
   investigation outcomes).
2. Policy independence: each policy-case evaluation runs in an isolated, fresh
   episode without sharing mutable state or mutating global RNG state.
3. Stable CSV serialization: exports episode traces to UTF-8 CSV with structured
   history serialized as valid JSON.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from config import AppConfig, validate_config
from agent import POLICY_NAMES, AgentSettings, PolicyName
from planner import _exact
from simulation import EpisodeResult, InvestigationRecord, run_episode

CSV_COLUMNS: tuple[str, ...] = (
    "seed",
    "policy",
    "hidden_hypothesis",
    "is_completed",
    "termination_reason",
    "stop_reason",
    "is_resolved",
    "is_correct",
    "predicted_hypothesis",
    "final_confidence",
    "initial_budget",
    "total_cost",
    "remaining_budget",
    "num_investigations",
    "final_entropy",
    "planner_expansion_count",
    "investigation_history",
)


def derive_case_seed(base_seed: int, case_idx: int) -> int:
    """Derive a deterministic, process-independent 64-bit integer seed for a case."""
    payload = f"case:{base_seed}:{case_idx}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class MetricsSummary:
    """Aggregated evaluation metrics for a policy.

    Rates and means return None when the relevant denominator is zero.
    """
    policy: str
    attempted_episodes: int
    completed_episodes: int
    incomplete_episodes: int
    error_episodes: int
    resolution_rate: Optional[float]
    completed_accuracy: Optional[float]
    overall_accuracy: Optional[float]
    resolved_accuracy: Optional[float]
    mean_cost: Optional[float]
    mean_investigations: Optional[float]
    mean_final_entropy_bits: Optional[float]
    budget_violations: int
    total_planner_expansions: Optional[int]
    mean_planner_expansions: Optional[float]


@dataclass(frozen=True)
class EvaluationReport:
    """Complete comparative evaluation report across policies."""
    seed: int
    num_episodes: int
    policies: tuple[str, ...]
    total_runs: int
    summaries: Mapping[str, MetricsSummary]
    all_results: tuple[EpisodeResult, ...]
    model_hypotheses: tuple[str, ...]
    model_investigations: tuple[str, ...]
    initial_budget: float
    confidence_threshold: float


def summarize_results(
    results: Sequence[EpisodeResult],
    policy: Optional[str] = None,
) -> MetricsSummary:
    """Compute aggregated metrics from a collection of EpisodeResult records.

    Denominator definitions:
    - resolution_rate: resolved episodes / attempted episodes
    - completed_accuracy: correct completed episodes / completed episodes
    - overall_accuracy: correct completed episodes / attempted episodes
      (incomplete episodes count as incorrect in the denominator)
    - resolved_accuracy: correct resolved episodes / resolved episodes
    - mean_cost: total cost across attempted episodes / attempted episodes
    - mean_investigations: total investigations / attempted episodes
    - mean_final_entropy_bits: average entropy across episodes with finite entropy
    """
    n = len(results)
    if policy is not None:
        pol = policy
    elif n > 0 and all(r.policy == results[0].policy for r in results):
        pol = results[0].policy
    elif n > 0:
        pol = "mixed"
    else:
        pol = "empty"

    if n == 0:
        return MetricsSummary(
            policy=pol,
            attempted_episodes=0,
            completed_episodes=0,
            incomplete_episodes=0,
            error_episodes=0,
            resolution_rate=None,
            completed_accuracy=None,
            overall_accuracy=None,
            resolved_accuracy=None,
            mean_cost=None,
            mean_investigations=None,
            mean_final_entropy_bits=None,
            budget_violations=0,
            total_planner_expansions=None,
            mean_planner_expansions=None,
        )

    completed = [r for r in results if r.is_completed]
    num_completed = len(completed)
    num_incomplete = n - num_completed

    num_error = sum(1 for r in results if r.termination_reason == "error")
    resolved = [r for r in results if r.is_resolved]
    num_resolved = len(resolved)

    resolution_rate = num_resolved / n

    if num_completed > 0:
        completed_accuracy = sum(1 for r in completed if r.is_correct) / num_completed
    else:
        completed_accuracy = None

    overall_accuracy = sum(1 for r in completed if r.is_correct) / n

    if num_resolved > 0:
        resolved_accuracy = sum(1 for r in resolved if r.is_correct) / num_resolved
    else:
        resolved_accuracy = None

    mean_cost = sum(r.total_cost for r in results) / n
    mean_investigations = sum(len(r.history) for r in results) / n

    valid_entropies = [
        r.final_entropy for r in results
        if r.final_entropy is not None and math.isfinite(r.final_entropy)
    ]
    if valid_entropies:
        mean_final_entropy = sum(valid_entropies) / len(valid_entropies)
    else:
        mean_final_entropy = None

    budget_violations = sum(
        1 for r in results
        if _exact(r.total_cost) > _exact(r.initial_budget)
    )

    has_expansions = any(len(r.planner_expansion_counts) > 0 for r in results)
    if has_expansions:
        total_expansions = sum(sum(r.planner_expansion_counts) for r in results)
        mean_expansions = total_expansions / n
    else:
        total_expansions = None
        mean_expansions = None

    return MetricsSummary(
        policy=pol,
        attempted_episodes=n,
        completed_episodes=num_completed,
        incomplete_episodes=num_incomplete,
        error_episodes=num_error,
        resolution_rate=resolution_rate,
        completed_accuracy=completed_accuracy,
        overall_accuracy=overall_accuracy,
        resolved_accuracy=resolved_accuracy,
        mean_cost=mean_cost,
        mean_investigations=mean_investigations,
        mean_final_entropy_bits=mean_final_entropy,
        budget_violations=budget_violations,
        total_planner_expansions=total_expansions,
        mean_planner_expansions=mean_expansions,
    )


def evaluate_policies(
    config: AppConfig,
    policies: Sequence[PolicyName] = POLICY_NAMES,
) -> EvaluationReport:
    """Evaluate policies on common synthetic cases generated under config.random_seed.

    Validates config and policy collection, derives isolated cases deterministically,
    and returns a structured EvaluationReport.
    """
    validate_config(config)

    if isinstance(policies, (str, bytes)) or not isinstance(policies, Sequence):
        raise ValueError("policies must be a sequence of policy names")
    if len(policies) == 0:
        raise ValueError("policies sequence must not be empty")

    seen_policies = set()
    for p in policies:
        if p not in POLICY_NAMES:
            raise ValueError(f"Unknown policy {p!r}; must be one of {POLICY_NAMES}")
        if p in seen_policies:
            raise ValueError(f"Duplicate policy {p!r} in policies sequence")
        seen_policies.add(p)

    # Preserve deterministic canonical output ordering
    canonical_policies = tuple(p for p in POLICY_NAMES if p in seen_policies)

    model = config.model
    planner_settings = config.planner
    agent_settings = AgentSettings(planner=planner_settings)
    base_seed = config.random_seed
    num_episodes = config.evaluation.num_episodes
    budget = planner_settings.initial_budget

    all_results: list[EpisodeResult] = []
    policy_results: dict[str, list[EpisodeResult]] = {p: [] for p in canonical_policies}

    for case_idx in range(num_episodes):
        case_seed = derive_case_seed(base_seed, case_idx)
        for p in canonical_policies:
            res = run_episode(
                model=model,
                policy=p,
                settings=agent_settings,
                seed=case_seed,
                budget=budget,
            )
            all_results.append(res)
            policy_results[p].append(res)

    summaries = {
        p: summarize_results(policy_results[p], policy=p)
        for p in canonical_policies
    }

    return EvaluationReport(
        seed=base_seed,
        num_episodes=num_episodes,
        policies=canonical_policies,
        total_runs=len(all_results),
        summaries=summaries,
        all_results=tuple(all_results),
        model_hypotheses=model.hypothesis_ids(),
        model_investigations=tuple(i.id for i in model.investigations),
        initial_budget=budget,
        confidence_threshold=planner_settings.confidence_threshold,
    )


def _serialize_history(history: tuple[InvestigationRecord, ...]) -> str:
    """Serialize investigation trajectory as valid JSON."""
    serialized = []
    for rec in history:
        item = {
            "step_index": rec.step_index,
            "test_id": rec.test_id,
            "outcome": rec.outcome,
            "cost": rec.cost,
            "prior_beliefs": dict(rec.prior_beliefs.probabilities),
            "posterior_beliefs": dict(rec.posterior_beliefs.probabilities),
            "remaining_budget": rec.remaining_budget,
            "decision": {
                "kind": rec.decision.kind,
                "test_id": rec.decision.test_id,
                "policy": rec.decision.policy,
                "explanation": {
                    "summary": rec.decision.explanation.summary,
                    "entropy_bits": rec.decision.explanation.entropy_bits,
                    "max_posterior": rec.decision.explanation.max_posterior,
                    "remaining_budget": rec.decision.explanation.remaining_budget,
                    "stop_reason": rec.decision.explanation.stop_reason,
                    "planner_status": rec.decision.explanation.planner_status,
                    "expected_net_utility": rec.decision.explanation.expected_net_utility,
                    "stop_utility": rec.decision.explanation.stop_utility,
                    "planner_expanded_states": rec.decision.explanation.planner_expanded_states,
                },
            },
        }
        serialized.append(item)
    return json.dumps(serialized, separators=(",", ":"))


def _result_to_csv_row(res: EpisodeResult) -> list[str]:
    """Convert an EpisodeResult to a deterministic CSV row."""
    planner_expansions_str = (
        str(sum(res.planner_expansion_counts))
        if res.planner_expansion_counts
        else ""
    )
    return [
        str(res.seed),
        res.policy,
        res.hidden_hypothesis,
        str(res.is_completed),
        res.termination_reason,
        res.stop_reason or "",
        str(res.is_resolved),
        str(res.is_correct),
        res.predicted_hypothesis,
        f"{res.final_confidence:.8g}",
        f"{res.initial_budget:.8g}",
        f"{res.total_cost:.8g}",
        f"{res.remaining_budget:.8g}",
        str(len(res.history)),
        f"{res.final_entropy:.8g}",
        planner_expansions_str,
        _serialize_history(res.history),
    ]


def export_results_csv(
    results: Sequence[EpisodeResult],
    path: Union[str, Path],
) -> None:
    """Export tabular episode results to CSV with JSON history serialization."""
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for r in results:
            writer.writerow(_result_to_csv_row(r))
