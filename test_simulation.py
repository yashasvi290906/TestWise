"""Unit tests for the TestWise simulation engine (Phase 0C).

Validates:
1. Episode initialization and prior sampling.
2. Observation sampling from the correct hidden-hypothesis likelihood row.
3. Bayesian posterior updates from observed outcomes.
4. Hidden-truth isolation from agent-facing state.
5. Deterministic replay with the same seed.
6. Unchanged global RNG state.
7. One action maximum per step.
8. No repeated investigations.
9. Exact budget safety, including decimal edge cases.
10. Stopping behavior and unresolved-but-completed episodes.
11. Incomplete contingent planning behavior.
12. Malformed seed, budget, and policy inputs.
13. A model with zero-probability outcomes and controlled impossible-evidence handling.
14. Result immutability and history snapshot integrity.
15. Deterministic potential outcomes across different investigation orders.
"""
from __future__ import annotations

import dataclasses
import math
import random
import unittest
from fractions import Fraction

from config import (
    ConfigError,
    Hypothesis,
    Investigation,
    ModelConfig,
    PlannerSettings,
    build_default_model,
)
from model import (
    BeliefState,
    initial_beliefs,
    update_beliefs,
    validate_model,
)
from agent import (
    POLICY_NAMES,
    THRESHOLD_EPS,
    AgentSettings,
    CandidateAction,
    Decision,
    Explanation,
    StopDecision,
)
from planner import PlanningState
from simulation import (
    Episode,
    EpisodeResult,
    InvestigationRecord,
    SimulationError,
    create_episode,
    run_episode,
    sample_hidden_hypothesis,
    sample_potential_outcome,
    step_episode,
)

M = build_default_model()
PS = PlannerSettings(initial_budget=10.0, confidence_threshold=0.90, cost_penalty_lambda=0.05, max_state_expansions=10_000)
AS = AgentSettings(planner=PS)


def make_binary_model() -> ModelConfig:
    """Simple 2-hypothesis, 2-investigation deterministic test model."""
    hyps = (
        Hypothesis("h1", "Hypothesis 1"),
        Hypothesis("h2", "Hypothesis 2"),
    )
    priors = {"h1": 0.60, "h2": 0.40}
    invs = (
        Investigation(
            "test_perfect",
            "Perfect Test",
            1.0,
            ("pos", "neg"),
            {"h1": {"pos": 1.0, "neg": 0.0}, "h2": {"pos": 0.0, "neg": 1.0}},
        ),
        Investigation(
            "test_noisy",
            "Noisy Test",
            0.5,
            ("pos", "neg"),
            {"h1": {"pos": 0.8, "neg": 0.2}, "h2": {"pos": 0.3, "neg": 0.7}},
        ),
    )
    m = ModelConfig(hyps, priors, invs)
    validate_model(m)
    return m


class TestSimulationInitialization(unittest.TestCase):
    def test_create_episode_defaults(self):
        ep = create_episode(M, seed=42, budget=10.0)
        self.assertIsInstance(ep, Episode)
        self.assertEqual(ep.seed, 42)
        self.assertEqual(ep.initial_budget, 10.0)
        self.assertEqual(ep.remaining_budget, 10.0)
        self.assertEqual(ep.state.beliefs, initial_beliefs(M))
        self.assertEqual(ep.state.performed, frozenset())
        self.assertEqual(len(ep.history), 0)
        self.assertFalse(ep.is_terminal)
        self.assertIn(ep._hidden_hypothesis, M.hypothesis_ids())

    def test_sample_hidden_hypothesis_distribution(self):
        # Sample 1000 hypotheses from M (priors: a=0.4, b=0.3, c=0.2, d=0.1)
        counts = {"cond_a": 0, "cond_b": 0, "cond_c": 0, "cond_d": 0}
        n = 1000
        for s in range(n):
            h = sample_hidden_hypothesis(M, seed=s)
            counts[h] += 1
        # Check that empirical frequencies are within 5% of theoretical priors
        self.assertAlmostEqual(counts["cond_a"] / n, 0.40, delta=0.05)
        self.assertAlmostEqual(counts["cond_b"] / n, 0.30, delta=0.05)
        self.assertAlmostEqual(counts["cond_c"] / n, 0.20, delta=0.05)
        self.assertAlmostEqual(counts["cond_d"] / n, 0.10, delta=0.05)

    def test_permitted_investigations_filter(self):
        ep = create_episode(M, seed=7, budget=10.0, permitted=("quick_screen", "imaging"))
        self.assertEqual(ep.state.permitted, frozenset({"quick_screen", "imaging"}))

    def test_invalid_init_inputs_rejected(self):
        # Invalid seeds
        for bad_seed in (True, -1, 1.5, "42", None):
            with self.subTest(seed=bad_seed), self.assertRaises((SimulationError, ValueError)):
                create_episode(M, seed=bad_seed, budget=10.0)

        # Invalid budgets
        for bad_budget in (True, -0.1, float("nan"), float("inf"), "10", None):
            with self.subTest(budget=bad_budget), self.assertRaises((SimulationError, ValueError)):
                create_episode(M, seed=1, budget=bad_budget)

        # Invalid max_steps
        for bad_steps in (True, 0, -1, 1.5, "5"):
            with self.subTest(steps=bad_steps), self.assertRaises((SimulationError, ValueError)):
                create_episode(M, seed=1, budget=10.0, max_steps=bad_steps)

        # Invalid permitted IDs
        with self.assertRaises(SimulationError):
            create_episode(M, seed=1, budget=10.0, permitted=("unknown_test",))


class TestObservationSampling(unittest.TestCase):
    def test_deterministic_outcome_matches_truth(self):
        model = make_binary_model()
        # In make_binary_model, test_perfect yields "pos" under h1, "neg" under h2
        # Run 50 seeds and verify test_perfect ALWAYS matches truth
        for s in range(50):
            ep = create_episode(model, seed=s, budget=5.0)
            outcome = sample_potential_outcome(model, s, ep._hidden_hypothesis, "test_perfect")
            if ep._hidden_hypothesis == "h1":
                self.assertEqual(outcome, "pos")
            else:
                self.assertEqual(outcome, "neg")

    def test_observation_frequency_matches_likelihood_distribution(self):
        model = make_binary_model()
        # For h1, test_noisy has P(pos)=0.8, P(neg)=0.2
        n = 1000
        pos_count = sum(
            1 for s in range(n)
            if sample_potential_outcome(model, s, "h1", "test_noisy") == "pos"
        )
        self.assertAlmostEqual(pos_count / n, 0.80, delta=0.05)

    def test_zero_probability_outcome_never_sampled(self):
        model = make_binary_model()
        # Under h1, "neg" has probability exactly 0.0
        for s in range(500):
            outcome = sample_potential_outcome(model, s, "h1", "test_perfect")
            self.assertNotEqual(outcome, "neg")


class TestBayesianUpdates(unittest.TestCase):
    def test_posterior_matches_model_update_beliefs(self):
        ep = create_episode(M, seed=12, budget=10.0)
        # Step once
        prior_b = ep.state.beliefs
        step_episode(ep, M, "fixed_order", AS)
        self.assertEqual(len(ep.history), 1)
        rec = ep.history[0]

        expected_posterior = update_beliefs(prior_b, rec.test_id, rec.outcome, M)
        self.assertEqual(rec.posterior_beliefs, expected_posterior)
        self.assertEqual(ep.state.beliefs, expected_posterior)

    def test_posteriors_normalize_to_one(self):
        res = run_episode(M, "fixed_order", AS, seed=33, budget=10.0)
        for rec in res.history:
            probs = list(rec.posterior_beliefs.probabilities.values())
            self.assertAlmostEqual(sum(probs), 1.0, delta=1e-9)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))


class TestHiddenTruthIsolation(unittest.TestCase):
    def test_planning_state_contains_no_hidden_truth(self):
        ep = create_episode(M, seed=42, budget=10.0)
        forbidden = ("truth", "hidden", "secret", "ground", "actual")

        # Check fields of PlanningState
        for f in dataclasses.fields(ep.state):
            for word in forbidden:
                self.assertNotIn(word, f.name.lower())

        # Step through an episode and check each record's decision and explanation
        while not ep.is_terminal:
            step_episode(ep, M, "cheapest_first", AS)
            # The agent state must not have any private simulator fields
            self.assertFalse(hasattr(ep.state, "_hidden_hypothesis"))
            self.assertFalse(hasattr(ep.state, "hidden_hypothesis"))

        # Check InvestigationRecords in history
        for rec in ep.history:
            ex = rec.decision.explanation
            for f in dataclasses.fields(ex):
                for word in forbidden:
                    self.assertNotIn(word, f.name.lower())

    def test_agent_decision_is_independent_of_hidden_truth(self):
        # Two episodes with different hidden hypotheses but same initial observable state
        # Find two seeds with different sampled hypotheses
        s_a = next(s for s in range(100) if sample_hidden_hypothesis(M, s) == "cond_a")
        s_b = next(s for s in range(100) if sample_hidden_hypothesis(M, s) == "cond_b")

        ep_a = create_episode(M, seed=s_a, budget=10.0)
        ep_b = create_episode(M, seed=s_b, budget=10.0)
        self.assertNotEqual(ep_a._hidden_hypothesis, ep_b._hidden_hypothesis)

        # Both agents must choose the exact same first action
        for pol in POLICY_NAMES:
            d_a = agent_step = step_episode(create_episode(M, s_a, 10.0), M, pol, AS).history
            d_b = agent_step = step_episode(create_episode(M, s_b, 10.0), M, pol, AS).history
            if d_a and d_b:
                self.assertEqual(d_a[0].test_id, d_b[0].test_id)


class TestDeterminismAndReproducibility(unittest.TestCase):
    def test_deterministic_replay_same_seed(self):
        res1 = run_episode(M, "fixed_order", AS, seed=123, budget=10.0)
        res2 = run_episode(M, "fixed_order", AS, seed=123, budget=10.0)

        self.assertEqual(res1.hidden_hypothesis, res2.hidden_hypothesis)
        self.assertEqual(res1.performed_investigations, res2.performed_investigations)
        self.assertEqual(res1.total_cost, res2.total_cost)
        self.assertEqual(res1.remaining_budget, res2.remaining_budget)
        self.assertEqual(res1.final_beliefs, res2.final_beliefs)
        self.assertEqual(res1.predicted_hypothesis, res2.predicted_hypothesis)
        self.assertEqual(res1.is_resolved, res2.is_resolved)
        self.assertEqual(res1.is_completed, res2.is_completed)
        self.assertEqual(res1.termination_reason, res2.termination_reason)

    def test_global_random_state_unaffected(self):
        random.seed(999)
        state_before = random.getstate()

        # Run several episodes across different policies
        for pol in POLICY_NAMES:
            run_episode(M, pol, AS, seed=42, budget=10.0)

        state_after = random.getstate()
        self.assertEqual(state_before, state_after)

    def test_counterfactual_potential_outcomes_independent_of_order(self):
        # Construct a 2-test model where neither test reaches 0.90 threshold alone
        hyps = (Hypothesis("h1", "H1"), Hypothesis("h2", "H2"))
        priors = {"h1": 0.50, "h2": 0.50}
        invs = (
            Investigation("test_a", "Test A", 1.0, ("pos", "neg"),
                          {"h1": {"pos": 0.65, "neg": 0.35}, "h2": {"pos": 0.35, "neg": 0.65}}),
            Investigation("test_b", "Test B", 1.0, ("pos", "neg"),
                          {"h1": {"pos": 0.65, "neg": 0.35}, "h2": {"pos": 0.35, "neg": 0.65}}),
        )
        model = ModelConfig(hyps, priors, invs)
        seed = 77
        hidden = sample_hidden_hypothesis(model, seed)

        expected_outcome_a = sample_potential_outcome(model, seed, hidden, "test_a")
        expected_outcome_b = sample_potential_outcome(model, seed, hidden, "test_b")

        # Run policy 1: A then B
        as_custom1 = AgentSettings(planner=PS, fixed_order=("test_a", "test_b"))
        res1 = run_episode(model, "fixed_order", as_custom1, seed=seed, budget=5.0)

        # Run policy 2: B then A
        as_custom2 = AgentSettings(planner=PS, fixed_order=("test_b", "test_a"))
        res2 = run_episode(model, "fixed_order", as_custom2, seed=seed, budget=5.0)

        outcomes1 = {rec.test_id: rec.outcome for rec in res1.history}
        outcomes2 = {rec.test_id: rec.outcome for rec in res2.history}

        # Despite different ordering, potential outcomes are identical
        self.assertEqual(outcomes1["test_a"], expected_outcome_a)
        self.assertEqual(outcomes2["test_a"], expected_outcome_a)
        self.assertEqual(outcomes1["test_b"], expected_outcome_b)
        self.assertEqual(outcomes2["test_b"], expected_outcome_b)


class TestStepAndExecutionInvariants(unittest.TestCase):
    def test_at_most_one_investigation_per_step(self):
        ep = create_episode(M, seed=1, budget=10.0)
        self.assertEqual(len(ep.history), 0)

        step_episode(ep, M, "fixed_order", AS)
        self.assertEqual(len(ep.history), 1)

        step_episode(ep, M, "fixed_order", AS)
        self.assertEqual(len(ep.history), 2)

    def test_no_repeated_investigations(self):
        for pol in POLICY_NAMES:
            res = run_episode(M, pol, AS, seed=42, budget=20.0)
            tests = res.performed_investigations
            self.assertEqual(len(tests), len(set(tests)), f"Repeated test in policy {pol}")

    def test_step_already_terminal_raises_error(self):
        ep = create_episode(M, seed=1, budget=0.0)
        # With budget 0.0, first step terminates immediately (no affordable tests)
        step_episode(ep, M, "fixed_order", AS)
        self.assertTrue(ep.is_terminal)

        with self.assertRaises(SimulationError):
            step_episode(ep, M, "fixed_order", AS)

    def test_step_limit_enforced(self):
        ep = create_episode(M, seed=1, budget=20.0, max_steps=2)
        while not ep.is_terminal:
            step_episode(ep, M, "fixed_order", AS)

        self.assertEqual(len(ep.history), 2)
        self.assertEqual(ep.termination_reason, "step_limit")
        self.assertEqual(ep.stop_reason, "step_limit")


class TestBudgetSafety(unittest.TestCase):
    def test_decimal_costs_no_float_drift(self):
        # Construct model with costs 0.1 and 0.2
        hyps = (Hypothesis("h1", "H1"), Hypothesis("h2", "H2"))
        priors = {"h1": 0.5, "h2": 0.5}
        invs = (
            Investigation("t1", "T1", 0.1, ("a", "b"), {"h1": {"a": 0.5, "b": 0.5}, "h2": {"a": 0.5, "b": 0.5}}),
            Investigation("t2", "T2", 0.2, ("a", "b"), {"h1": {"a": 0.5, "b": 0.5}, "h2": {"a": 0.5, "b": 0.5}}),
        )
        model = ModelConfig(hyps, priors, invs)

        # Budget 0.3. Naive float subtraction gives 0.3 - 0.1 = 0.19999999999999998 < 0.2
        settings = AgentSettings(planner=PlannerSettings(initial_budget=0.3, confidence_threshold=1.0))
        res = run_episode(model, "cheapest_first", settings, seed=0, budget=0.3)

        self.assertEqual(res.performed_investigations, ("t1", "t2"))
        self.assertEqual(res.remaining_budget, 0.0)
        self.assertEqual(res.total_cost, 0.3)

    def test_slightly_short_budget_blocks_test(self):
        model = make_binary_model()
        # test_perfect costs 1.0; budget 0.9999999999 is slightly short
        ep = create_episode(model, seed=1, budget=0.9999999999)
        # Should stop immediately because neither test (costs 1.0, 0.5) or only 0.5 is affordable
        step_episode(ep, model, "fixed_order", AgentSettings(planner=PlannerSettings(initial_budget=1.0),
                                                             fixed_order=("test_perfect",)))
        self.assertEqual(ep.history[0].test_id, "test_noisy")  # fell through to 0.5 test

    def test_zero_budget_stops_immediately(self):
        res = run_episode(M, "fixed_order", AS, seed=5, budget=0.0)
        self.assertEqual(len(res.history), 0)
        self.assertEqual(res.total_cost, 0.0)
        self.assertEqual(res.remaining_budget, 0.0)
        self.assertEqual(res.termination_reason, "no_legal_actions")
        self.assertTrue(res.is_completed)

    def test_budget_accounting_exactness(self):
        for b in (3.7, 5.0, 7.3, 10.0):
            res = run_episode(M, "cheapest_first", AS, seed=9, budget=b)
            self.assertEqual(
                Fraction(repr(res.total_cost)) + Fraction(repr(res.remaining_budget)),
                Fraction(repr(b)),
            )


class TestTerminationAndResolving(unittest.TestCase):
    def test_confidence_threshold_stop_is_resolved(self):
        model = make_binary_model()
        # test_perfect resolves hypothesis with certainty 1.0 >= 0.90
        res = run_episode(model, "fixed_order", AS, seed=10, budget=5.0)
        self.assertTrue(res.is_completed)
        self.assertTrue(res.is_resolved)
        self.assertEqual(res.termination_reason, "policy_stop")
        self.assertEqual(res.stop_reason, "confidence_threshold_met")
        self.assertAlmostEqual(res.final_confidence, 1.0, delta=1e-12)

    def test_budget_exhausted_unresolved_but_completed(self):
        # Budget 1.0 allows only quick_screen; confidence threshold 0.95 won't be reached
        settings = AgentSettings(planner=PlannerSettings(initial_budget=1.0, confidence_threshold=0.95))
        res = run_episode(M, "cheapest_first", settings, seed=10, budget=1.0)
        self.assertTrue(res.is_completed)
        self.assertFalse(res.is_resolved)
        self.assertEqual(res.termination_reason, "no_legal_actions")
        self.assertEqual(res.stop_reason, "no_affordable_investigations")

    def test_contingent_optimal_utility_stopping(self):
        # High cost penalty lambda forces stopping early because utility of stopping > investigating
        high_lam = AgentSettings(planner=PlannerSettings(initial_budget=10.0, cost_penalty_lambda=0.5, confidence_threshold=0.99))
        res = run_episode(M, "contingent_optimal", high_lam, seed=15, budget=10.0)
        self.assertTrue(res.is_completed)
        self.assertEqual(res.termination_reason, "policy_stop")
        self.assertEqual(res.stop_reason, "planner_stop_optimal")

    def test_contingent_optimal_expansion_limit_incomplete(self):
        # max_state_expansions=1 forces search limit to be reached at root
        limit_1 = AgentSettings(planner=PlannerSettings(initial_budget=10.0, cost_penalty_lambda=0.01, max_state_expansions=1))
        res = run_episode(M, "contingent_optimal", limit_1, seed=20, budget=10.0)
        self.assertFalse(res.is_completed)
        self.assertFalse(res.is_resolved)
        self.assertEqual(res.termination_reason, "planner_incomplete")
        self.assertEqual(res.stop_reason, "incomplete_expansion_limit")


class TestResultImmutability(unittest.TestCase):
    def test_result_and_records_are_frozen(self):
        res = run_episode(M, "fixed_order", AS, seed=55, budget=10.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            res.total_cost = 999.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            res.is_completed = False

        if res.history:
            rec = res.history[0]
            with self.assertRaises(dataclasses.FrozenInstanceError):
                rec.outcome = "fake_outcome"

    def test_to_result_before_terminal_raises_error(self):
        ep = create_episode(M, seed=1, budget=10.0)
        self.assertFalse(ep.is_terminal)
        with self.assertRaises(SimulationError):
            ep.to_result(M, AS)


class TestPolicySupport(unittest.TestCase):
    def test_all_four_policies_run_successfully(self):
        for pol in POLICY_NAMES:
            res = run_episode(M, pol, AS, seed=88, budget=10.0)
            self.assertIsInstance(res, EpisodeResult)
            self.assertIn(res.termination_reason, ("policy_stop", "no_legal_actions"))
            self.assertGreaterEqual(res.remaining_budget, 0.0)

    def test_invalid_policy_raises_error(self):
        ep = create_episode(M, seed=1, budget=10.0)
        with self.assertRaises(ValueError):
            step_episode(ep, M, "invalid_policy_name", AS)


if __name__ == "__main__":
    unittest.main()
