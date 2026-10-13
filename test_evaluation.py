"""Unit tests for the TestWise evaluation engine (Phase 0D).

Validates:
1. All expected policy names are accepted.
2. Unknown policies are rejected.
3. Duplicate policies are rejected.
4. Invalid configurations are rejected.
5. Same configuration and inputs produce identical evaluation reports.
6. Evaluation order does not change the case truth/outcome for a given policy and case.
7. Each policy receives a fresh episode.
8. Metric denominators are correct.
9. Resolved-but-incorrect episodes are handled correctly.
10. Incomplete episodes count as incorrect in attempted-case accuracy.
11. Undefined rates/means return None where appropriate.
12. Empty-result behavior.
13. Budget violations are detected in summaries.
14. Planner expansion metrics are computed only from relevant records.
15. CSV header and column order are stable.
16. CSV fields correctly quote special characters.
17. History JSON round-trips without losing outcomes or posterior snapshots.
18. Empty CSV export still writes the header.
19. No evaluation metrics are hardcoded or fabricated.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path

from config import (
    AppConfig,
    ConfigError,
    EvaluationSettings,
    PlannerSettings,
    build_default_app_config,
    build_default_model,
)
from agent import POLICY_NAMES, CandidateAction, Decision, Explanation
from model import BeliefState
from simulation import EpisodeResult, InvestigationRecord
from evaluation import (
    CSV_COLUMNS,
    EvaluationReport,
    MetricsSummary,
    derive_case_seed,
    evaluate_policies,
    export_results_csv,
    summarize_results,
)

M = build_default_model()


def make_test_config(num_episodes: int = 2, seed: int = 42) -> AppConfig:
    """Build a lightweight AppConfig with small episode count for fast testing."""
    base = build_default_app_config()
    return dataclasses.replace(
        base,
        evaluation=EvaluationSettings(num_episodes=num_episodes),
        random_seed=seed,
    )


def make_dummy_result(
    seed: int = 1,
    policy: str = "fixed_order",
    hidden: str = "cond_a",
    pred: str = "cond_a",
    completed: bool = True,
    resolved: bool = True,
    correct: bool = True,
    cost: float = 1.0,
    budget: float = 10.0,
    entropy_val: float = 0.5,
    num_tests: int = 1,
    expansions: tuple[int, ...] = (),
    termination_reason: str = "policy_stop",
) -> EpisodeResult:
    """Construct a mock EpisodeResult with explicit fields for metric validation."""
    beliefs = BeliefState({"cond_a": 0.95, "cond_b": 0.05, "cond_c": 0.0, "cond_d": 0.0})
    ex = Explanation(
        policy=policy,
        summary="Test step",
        entropy_bits=entropy_val,
        max_posterior=0.95,
        remaining_budget=budget - cost,
        candidates=(),
    )
    dec = Decision(kind="investigate", test_id="quick_screen", policy=policy, explanation=ex)
    records = tuple(
        InvestigationRecord(
            step_index=i,
            test_id="quick_screen",
            outcome="positive",
            cost=cost / max(1, num_tests),
            prior_beliefs=beliefs,
            posterior_beliefs=beliefs,
            remaining_budget=budget - cost,
            decision=dec,
        )
        for i in range(num_tests)
    )
    return EpisodeResult(
        seed=seed,
        policy=policy,
        hidden_hypothesis=hidden,
        initial_budget=budget,
        total_cost=cost,
        remaining_budget=budget - cost,
        history=records,
        final_beliefs=beliefs,
        predicted_hypothesis=pred,
        final_confidence=0.95,
        final_entropy=entropy_val,
        is_resolved=resolved,
        is_completed=completed,
        is_correct=correct,
        termination_reason=termination_reason,
        stop_reason="confidence_threshold_met" if resolved else None,
        performed_investigations=tuple(r.test_id for r in records),
        planner_expansion_counts=expansions,
        confidence_threshold=0.90,
    )


class TestPolicyValidation(unittest.TestCase):
    def test_all_expected_policies_accepted(self):
        cfg = make_test_config(num_episodes=2)
        report = evaluate_policies(cfg, policies=POLICY_NAMES)
        self.assertIsInstance(report, EvaluationReport)
        self.assertEqual(report.policies, POLICY_NAMES)
        self.assertEqual(len(report.summaries), len(POLICY_NAMES))
        self.assertEqual(report.total_runs, 2 * len(POLICY_NAMES))

    def test_unknown_policy_rejected(self):
        cfg = make_test_config(num_episodes=1)
        with self.assertRaises(ValueError):
            evaluate_policies(cfg, policies=("fixed_order", "unknown_magic_policy"))

    def test_duplicate_policies_rejected(self):
        cfg = make_test_config(num_episodes=1)
        with self.assertRaises(ValueError):
            evaluate_policies(cfg, policies=("fixed_order", "fixed_order"))

    def test_empty_or_invalid_collection_rejected(self):
        cfg = make_test_config(num_episodes=1)
        with self.assertRaises(ValueError):
            evaluate_policies(cfg, policies=())
        with self.assertRaises(ValueError):
            evaluate_policies(cfg, policies="fixed_order")  # string instead of sequence


class TestConfigurationValidation(unittest.TestCase):
    def test_invalid_config_rejected(self):
        cfg = make_test_config()
        # Invalid title
        with self.assertRaises(ConfigError):
            evaluate_policies(dataclasses.replace(cfg, app_title=" "))
        # Invalid schema version
        with self.assertRaises(ConfigError):
            evaluate_policies(dataclasses.replace(cfg, schema_version=0))
        # Invalid seed
        with self.assertRaises(ConfigError):
            evaluate_policies(dataclasses.replace(cfg, random_seed=-1))
        # Invalid evaluation settings
        with self.assertRaises(ConfigError):
            evaluate_policies(dataclasses.replace(cfg, evaluation=None))


class TestFairComparisonAndIndependence(unittest.TestCase):
    def test_identical_reports_same_config(self):
        cfg = make_test_config(num_episodes=3, seed=100)
        pols = ("fixed_order", "cheapest_first")
        r1 = evaluate_policies(cfg, policies=pols)
        r2 = evaluate_policies(cfg, policies=pols)

        self.assertEqual(r1.seed, r2.seed)
        self.assertEqual(r1.total_runs, r2.total_runs)
        self.assertEqual(r1.summaries, r2.summaries)
        self.assertEqual(r1.all_results, r2.all_results)

    def test_evaluation_order_does_not_change_case_outcomes(self):
        cfg = make_test_config(num_episodes=3, seed=77)
        # Order A
        r_a = evaluate_policies(cfg, policies=("fixed_order", "cheapest_first"))
        # Order B (reversed)
        r_b = evaluate_policies(cfg, policies=("cheapest_first", "fixed_order"))

        # Both reports maintain canonical output order
        self.assertEqual(r_a.policies, r_b.policies)

        # For each policy, results must be identical regardless of input order
        for pol in ("fixed_order", "cheapest_first"):
            res_a = [res for res in r_a.all_results if res.policy == pol]
            res_b = [res for res in r_b.all_results if res.policy == pol]
            self.assertEqual(res_a, res_b)

    def test_each_policy_receives_fresh_episode(self):
        cfg = make_test_config(num_episodes=2, seed=88)
        pols = ("fixed_order", "cheapest_first")
        report = evaluate_policies(cfg, policies=pols)

        # Confirm all results have distinct objects
        seen_ids = set()
        for res in report.all_results:
            self.assertNotIn(id(res), seen_ids)
            seen_ids.add(id(res))


class TestMetricCalculationsAndDenominators(unittest.TestCase):
    def test_metric_denominators_and_formulas(self):
        # Create 4 specific mock cases:
        # Case 1: completed, resolved, correct, cost 2.0, 1 test, entropy 0.1
        c1 = make_dummy_result(seed=1, completed=True, resolved=True, correct=True, cost=2.0, num_tests=1, entropy_val=0.1)
        # Case 2: completed, resolved, INCORRECT (resolved but wrong), cost 3.0, 2 tests, entropy 0.2
        c2 = make_dummy_result(seed=2, completed=True, resolved=True, correct=False, cost=3.0, num_tests=2, entropy_val=0.2)
        # Case 3: completed, UNRESOLVED, incorrect, cost 1.0, 1 test, entropy 0.8
        c3 = make_dummy_result(seed=3, completed=True, resolved=False, correct=False, cost=1.0, num_tests=1, entropy_val=0.8)
        # Case 4: INCOMPLETE, unresolved, incorrect, cost 4.0, 2 tests, entropy 1.2
        c4 = make_dummy_result(seed=4, completed=False, resolved=False, correct=False, cost=4.0, num_tests=2, entropy_val=1.2,
                               termination_reason="planner_incomplete")

        summary = summarize_results([c1, c2, c3, c4], policy="test_policy")

        self.assertEqual(summary.attempted_episodes, 4)
        self.assertEqual(summary.completed_episodes, 3)
        self.assertEqual(summary.incomplete_episodes, 1)
        self.assertEqual(summary.error_episodes, 0)

        # Resolution rate: resolved (2) / attempted (4) = 0.5
        self.assertEqual(summary.resolution_rate, 0.5)

        # Completed accuracy: correct completed (1) / completed (3) = 1/3
        self.assertAlmostEqual(summary.completed_accuracy, 1 / 3, places=7)

        # Overall accuracy: correct completed (1) / attempted (4) = 0.25 (incomplete counts as incorrect)
        self.assertEqual(summary.overall_accuracy, 0.25)

        # Resolved accuracy: correct resolved (1) / resolved (2) = 0.5
        self.assertEqual(summary.resolved_accuracy, 0.5)

        # Mean cost: (2 + 3 + 1 + 4) / 4 = 2.5
        self.assertEqual(summary.mean_cost, 2.5)

        # Mean investigations: (1 + 2 + 1 + 2) / 4 = 1.5
        self.assertEqual(summary.mean_investigations, 1.5)

        # Mean final entropy: (0.1 + 0.2 + 0.8 + 1.2) / 4 = 0.575
        self.assertAlmostEqual(summary.mean_final_entropy_bits, 0.575, places=7)

        # Budget violations: 0
        self.assertEqual(summary.budget_violations, 0)

    def test_resolved_but_incorrect_handled_correctly(self):
        # 1 resolved-correct, 1 resolved-incorrect
        r1 = make_dummy_result(seed=1, completed=True, resolved=True, correct=True)
        r2 = make_dummy_result(seed=2, completed=True, resolved=True, correct=False)
        summary = summarize_results([r1, r2])

        self.assertEqual(summary.resolution_rate, 1.0)
        self.assertEqual(summary.resolved_accuracy, 0.5)
        self.assertEqual(summary.overall_accuracy, 0.5)

    def test_incomplete_counts_as_incorrect_in_overall_accuracy(self):
        # 1 completed correct, 1 incomplete
        r1 = make_dummy_result(seed=1, completed=True, resolved=True, correct=True)
        r2 = make_dummy_result(seed=2, completed=False, resolved=False, correct=False, termination_reason="planner_incomplete")
        summary = summarize_results([r1, r2])

        self.assertEqual(summary.completed_episodes, 1)
        self.assertEqual(summary.incomplete_episodes, 1)
        # completed accuracy = 1 / 1 = 1.0
        self.assertEqual(summary.completed_accuracy, 1.0)
        # overall accuracy = 1 / 2 = 0.5 (incomplete penalizes overall)
        self.assertEqual(summary.overall_accuracy, 0.5)

    def test_undefined_rates_return_none(self):
        # No resolved episodes
        r1 = make_dummy_result(completed=True, resolved=False, correct=False)
        summary = summarize_results([r1])
        self.assertIsNone(summary.resolved_accuracy)
        self.assertEqual(summary.resolution_rate, 0.0)

        # No completed episodes
        r_incomp = make_dummy_result(completed=False, resolved=False, correct=False)
        summary_incomp = summarize_results([r_incomp])
        self.assertIsNone(summary_incomp.completed_accuracy)
        self.assertEqual(summary_incomp.overall_accuracy, 0.0)

    def test_empty_results_summary(self):
        s = summarize_results([])
        self.assertEqual(s.attempted_episodes, 0)
        self.assertEqual(s.completed_episodes, 0)
        self.assertEqual(s.incomplete_episodes, 0)
        self.assertEqual(s.error_episodes, 0)
        self.assertIsNone(s.resolution_rate)
        self.assertIsNone(s.completed_accuracy)
        self.assertIsNone(s.overall_accuracy)
        self.assertIsNone(s.resolved_accuracy)
        self.assertIsNone(s.mean_cost)
        self.assertIsNone(s.mean_investigations)
        self.assertIsNone(s.mean_final_entropy_bits)
        self.assertEqual(s.budget_violations, 0)
        self.assertIsNone(s.total_planner_expansions)
        self.assertIsNone(s.mean_planner_expansions)

    def test_budget_violations_detected(self):
        # Result where cost (12.0) exceeds budget (10.0)
        viol = make_dummy_result(cost=12.0, budget=10.0)
        ok = make_dummy_result(cost=8.0, budget=10.0)
        summary = summarize_results([viol, ok])
        self.assertEqual(summary.budget_violations, 1)

    def test_planner_expansion_metrics_computed_properly(self):
        # Non-contingent (no expansions)
        r_base = make_dummy_result(expansions=())
        s_base = summarize_results([r_base])
        self.assertIsNone(s_base.total_planner_expansions)
        self.assertIsNone(s_base.mean_planner_expansions)

        # Contingent (has expansions)
        r_plan1 = make_dummy_result(expansions=(100, 50))
        r_plan2 = make_dummy_result(expansions=(200,))
        s_plan = summarize_results([r_plan1, r_plan2])
        self.assertEqual(s_plan.total_planner_expansions, 350)
        self.assertEqual(s_plan.mean_planner_expansions, 175.0)

    def test_metrics_are_not_hardcoded(self):
        # Verify dynamic change
        r1 = make_dummy_result(cost=1.0)
        r2 = make_dummy_result(cost=5.0)
        self.assertEqual(summarize_results([r1]).mean_cost, 1.0)
        self.assertEqual(summarize_results([r2]).mean_cost, 5.0)
        self.assertEqual(summarize_results([r1, r2]).mean_cost, 3.0)


class TestCSVExport(unittest.TestCase):
    def test_csv_header_and_column_order_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            res = make_dummy_result()
            export_results_csv([res], csv_path)

            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(tuple(header), CSV_COLUMNS)

    def test_empty_csv_writes_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            export_results_csv([], csv_path)

            with csv_path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                self.assertEqual(len(lines), 1)
                self.assertEqual(lines[0], ",".join(CSV_COLUMNS))

    def test_csv_special_characters_quoting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "quoted.csv"
            # Explanation with commas, quotes, and newlines
            ex = Explanation(
                policy="test",
                summary='Chose test: "high value", very good!\nLine 2.',
                entropy_bits=1.0,
                max_posterior=0.9,
                remaining_budget=5.0,
                candidates=(),
            )
            dec = Decision(kind="investigate", test_id="t1", policy="test", explanation=ex)
            rec = InvestigationRecord(
                step_index=0,
                test_id="t1",
                outcome="pos,positive",
                cost=1.0,
                prior_beliefs=BeliefState({"h1": 0.5, "h2": 0.5}),
                posterior_beliefs=BeliefState({"h1": 0.8, "h2": 0.2}),
                remaining_budget=4.0,
                decision=dec,
            )
            res = dataclasses.replace(make_dummy_result(), history=(rec,))
            export_results_csv([res], csv_path)

            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                raw_json = rows[0]["investigation_history"]
                parsed = json.loads(raw_json)
                self.assertEqual(parsed[0]["outcome"], "pos,positive")
                self.assertEqual(parsed[0]["decision"]["explanation"]["summary"],
                                 'Chose test: "high value", very good!\nLine 2.')

    def test_history_json_round_trips_completely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "roundtrip.csv"
            cfg = make_test_config(num_episodes=2, seed=42)
            report = evaluate_policies(cfg, policies=("fixed_order",))
            export_results_csv(report.all_results, csv_path)

            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), len(report.all_results))

                for original, row in zip(report.all_results, rows):
                    self.assertEqual(str(original.seed), row["seed"])
                    self.assertEqual(original.policy, row["policy"])
                    self.assertEqual(original.hidden_hypothesis, row["hidden_hypothesis"])
                    self.assertEqual(str(original.is_completed), row["is_completed"])
                    self.assertEqual(str(original.is_resolved), row["is_resolved"])
                    self.assertEqual(str(original.is_correct), row["is_correct"])

                    # Parse history
                    hist = json.loads(row["investigation_history"])
                    self.assertEqual(len(hist), len(original.history))
                    for o_rec, p_rec in zip(original.history, hist):
                        self.assertEqual(o_rec.step_index, p_rec["step_index"])
                        self.assertEqual(o_rec.test_id, p_rec["test_id"])
                        self.assertEqual(o_rec.outcome, p_rec["outcome"])
                        self.assertAlmostEqual(o_rec.cost, p_rec["cost"], places=7)
                        self.assertEqual(dict(o_rec.prior_beliefs.probabilities), p_rec["prior_beliefs"])
                        self.assertEqual(dict(o_rec.posterior_beliefs.probabilities), p_rec["posterior_beliefs"])

    def test_parent_directories_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "sub1" / "sub2" / "out.csv"
            self.assertFalse(nested_path.parent.exists())
            export_results_csv([], nested_path)
            self.assertTrue(nested_path.exists())


if __name__ == "__main__":
    unittest.main()
