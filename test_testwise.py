"""Phase 1 tests: configuration and Bayesian model (unittest only)."""
import dataclasses
import math
import unittest

from config import (AppConfig, ConfigError, EvaluationSettings, Investigation,
                    ModelConfig, PlannerSettings, build_default_app_config,
                    build_default_model, validate_config)
from agent import (POLICY_NAMES, AgentSettings, CandidateAction, Decision, Explanation,
                   StopDecision, choose_action, get_candidate_actions, should_stop)
from fractions import Fraction
from planner import (PlanningResult, is_affordable, available_investigations, PlanningState, PlanStatus, PlanningError, PolicyNode,
                     SearchProblem, SearchStatus, a_star_search, advance_state,
                     breadth_first_search, depth_first_search, make_initial_state,
                     solve_contingent_plan, unit_cost_problem)
from config import Hypothesis
from model import (BeliefState, BeliefError, ImpossibleObservationError,
                   UnknownIdentifierError, entropy, expected_information_gain,
                   initial_beliefs, outcome_probability, update_beliefs,
                   validate_model)

TOL = 1e-9
M = build_default_model()
PRIOR = BeliefState(dict(M.priors))
UNIFORM = BeliefState({h: 0.25 for h in M.hypothesis_ids()})


def lik_copy(test_id, model=M):
    """Mutable deep copy of an investigation's (read-only) likelihood table."""
    return {h: dict(r) for h, r in model.investigation(test_id).likelihoods.items()}


def with_inv(model, test_id, **changes):
    invs = tuple(dataclasses.replace(i, **changes) if i.id == test_id else i
                 for i in model.investigations)
    return dataclasses.replace(model, investigations=invs)


class TestDefaults(unittest.TestCase):
    def test_default_model_shape_and_validity(self):
        validate_model(M)
        self.assertEqual(len(M.hypotheses), 4)
        self.assertEqual(len(M.investigations), 6)
        self.assertAlmostEqual(sum(M.priors.values()), 1.0, delta=TOL)

    def test_default_app_config_valid(self):
        validate_config(build_default_app_config())

    def test_invalid_app_config(self):
        with self.assertRaises(ConfigError):
            validate_config(AppConfig(model=M, app_title=" "))
        with self.assertRaises(ConfigError):
            validate_config(AppConfig(model=M, schema_version=0))


class TestValidation(unittest.TestCase):
    def test_negative_prior(self):
        p = dict(M.priors); p["cond_a"] = -0.1; p["cond_b"] = 0.8
        with self.assertRaises(ConfigError):
            validate_model(dataclasses.replace(M, priors=p))

    def test_prior_total_not_one(self):
        p = dict(M.priors); p["cond_a"] = 0.5
        with self.assertRaises(ConfigError):
            validate_model(dataclasses.replace(M, priors=p))

    def test_prior_nan_and_missing_key(self):
        p = dict(M.priors); p["cond_a"] = float("nan")
        with self.assertRaises(ConfigError):
            validate_model(dataclasses.replace(M, priors=p))
        p = dict(M.priors); del p["cond_d"]
        with self.assertRaises(ConfigError):
            validate_model(dataclasses.replace(M, priors=p))

    def test_prior_within_tolerance_accepted(self):
        p = dict(M.priors); p["cond_a"] += 1e-11
        validate_model(dataclasses.replace(M, priors=p))

    def test_likelihood_row_not_normalized(self):
        lik = lik_copy("quick_screen")
        lik["cond_a"]["positive"] = 0.5
        with self.assertRaises(ConfigError):
            validate_model(with_inv(M, "quick_screen", likelihoods=lik))

    def test_likelihood_negative(self):
        lik = lik_copy("quick_screen")
        lik["cond_a"] = {"negative": 1.2, "positive": -0.2}
        with self.assertRaises(ConfigError):
            validate_model(with_inv(M, "quick_screen", likelihoods=lik))

    def test_likelihood_missing_hypothesis_or_outcome(self):
        lik = lik_copy("quick_screen")
        del lik["cond_d"]
        with self.assertRaises(ConfigError):
            validate_model(with_inv(M, "quick_screen", likelihoods=lik))
        lik = lik_copy("quick_screen")
        lik["cond_a"] = {"negative": 1.0}
        with self.assertRaises(ConfigError):
            validate_model(with_inv(M, "quick_screen", likelihoods=lik))

    def test_malformed_not_repaired(self):
        p = dict(M.priors); p["cond_a"] = 0.5
        bad = dataclasses.replace(M, priors=p)
        with self.assertRaises(ConfigError):
            validate_model(bad)
        self.assertEqual(bad.priors["cond_a"], 0.5)  # untouched

    def test_duplicate_ids_and_bad_cost(self):
        dup = dataclasses.replace(M, investigations=M.investigations + (M.investigations[0],))
        with self.assertRaises(ConfigError):
            validate_model(dup)
        with self.assertRaises(ConfigError):
            validate_model(with_inv(M, "imaging", cost=-1.0))
        with self.assertRaises(ConfigError):
            validate_model(with_inv(M, "imaging", outcomes=("clear", "clear", "positive")))


class TestUnknownIds(unittest.TestCase):
    def test_unknown_investigation(self):
        for fn in (lambda: outcome_probability(PRIOR, "nope", M),
                   lambda: update_beliefs(PRIOR, "nope", "positive", M),
                   lambda: expected_information_gain(PRIOR, "nope", M)):
            with self.assertRaises(UnknownIdentifierError):
                fn()

    def test_unknown_outcome(self):
        with self.assertRaises(UnknownIdentifierError):
            update_beliefs(PRIOR, "quick_screen", "maybe", M)

    def test_unknown_hypothesis_in_beliefs(self):
        b = BeliefState({**PRIOR.probabilities, "cond_z": 0.0})
        with self.assertRaises(UnknownIdentifierError):
            outcome_probability(b, "quick_screen", M)

    def test_malformed_beliefs(self):
        with self.assertRaises(BeliefError):
            outcome_probability(BeliefState({"cond_a": 0.5, "cond_b": 0.1,
                                             "cond_c": 0.1, "cond_d": 0.1}), "quick_screen", M)


class TestBayes(unittest.TestCase):
    def test_hand_example_1(self):
        # prior (.4,.3,.2,.1); quick_screen positive: joints .12,.18,.14,.08; P(o)=.52
        post = update_beliefs(PRIOR, "quick_screen", "positive", M).probabilities
        exp = {"cond_a": 3 / 13, "cond_b": 9 / 26, "cond_c": 7 / 26, "cond_d": 2 / 13}
        for h, v in exp.items():
            self.assertAlmostEqual(post[h], v, delta=TOL)
        self.assertAlmostEqual(outcome_probability(PRIOR, "quick_screen", M)["positive"], 0.52, delta=TOL)

    def test_hand_example_2(self):
        # uniform prior; specialist_review negative: joints .2375,.025,.225,.125; P(o)=.6125
        post = update_beliefs(UNIFORM, "specialist_review", "negative", M).probabilities
        exp = {"cond_a": 19 / 49, "cond_b": 2 / 49, "cond_c": 18 / 49, "cond_d": 10 / 49}
        for h, v in exp.items():
            self.assertAlmostEqual(post[h], v, delta=TOL)

    def test_normalization_and_nonnegativity_all_cases(self):
        for inv in M.investigations:
            po = outcome_probability(PRIOR, inv.id, M)
            self.assertAlmostEqual(sum(po.values()), 1.0, delta=TOL)
            for o in inv.outcomes:
                post = update_beliefs(PRIOR, inv.id, o, M).probabilities
                self.assertAlmostEqual(sum(post.values()), 1.0, delta=TOL)
                self.assertTrue(all(p >= 0 for p in post.values()))

    def test_sequential_update_order_independent(self):
        a = update_beliefs(update_beliefs(PRIOR, "quick_screen", "positive", M),
                           "imaging", "clear", M).probabilities
        b = update_beliefs(update_beliefs(PRIOR, "imaging", "clear", M),
                           "quick_screen", "positive", M).probabilities
        for h in a:
            self.assertAlmostEqual(a[h], b[h], delta=TOL)

    def test_impossible_observation(self):
        lik = lik_copy("quick_screen")
        lik["cond_a"] = {"negative": 1.0, "positive": 0.0}
        m2 = with_inv(M, "quick_screen", likelihoods=lik)
        b = BeliefState({"cond_a": 1.0, "cond_b": 0.0, "cond_c": 0.0, "cond_d": 0.0})
        with self.assertRaises(ImpossibleObservationError):
            update_beliefs(b, "quick_screen", "positive", m2)


class TestEntropyAndGain(unittest.TestCase):
    def test_certain_entropy_zero(self):
        b = BeliefState({"cond_a": 1.0, "cond_b": 0.0, "cond_c": 0.0, "cond_d": 0.0})
        self.assertAlmostEqual(entropy(b), 0.0, delta=TOL)

    def test_uniform_four_entropy_two_bits(self):
        self.assertAlmostEqual(entropy(UNIFORM), 2.0, delta=TOL)

    def test_known_entropy_half_quarter(self):
        b = BeliefState({"cond_a": 0.5, "cond_b": 0.25, "cond_c": 0.25, "cond_d": 0.0})
        self.assertAlmostEqual(entropy(b), 1.5, delta=TOL)

    def test_uninformative_test_zero_gain(self):
        row = {"negative": 0.3, "positive": 0.7}
        lik = {h: dict(row) for h in M.hypothesis_ids()}
        m2 = with_inv(M, "quick_screen", likelihoods=lik)
        validate_model(m2)
        self.assertAlmostEqual(expected_information_gain(PRIOR, "quick_screen", m2), 0.0, delta=TOL)

    def test_perfect_test_gain_equals_prior_entropy(self):
        outs = tuple(M.hypothesis_ids())
        lik = {h: {o: 1.0 if o == h else 0.0 for o in outs} for h in outs}
        inv = dataclasses.replace(M.investigations[0], id="oracle", outcomes=outs, likelihoods=lik)
        m2 = dataclasses.replace(M, investigations=(inv,))
        validate_model(m2)
        self.assertAlmostEqual(expected_information_gain(UNIFORM, "oracle", m2), 2.0, delta=TOL)

    def test_gain_nonnegative_and_bounded(self):
        beliefs = [PRIOR, UNIFORM,
                   BeliefState({"cond_a": 1.0, "cond_b": 0.0, "cond_c": 0.0, "cond_d": 0.0}),
                   update_beliefs(PRIOR, "imaging", "positive", M)]
        for b in beliefs:
            for inv in M.investigations:
                g = expected_information_gain(b, inv.id, M)
                self.assertGreaterEqual(g, -TOL)
                self.assertLessEqual(g, entropy(b) + TOL)

    def test_gain_matches_hand_calculation_quick_screen(self):
        # Independent computation from hand-derived marginals/posteriors.
        h = lambda ps: -sum(p * math.log2(p) for p in ps if p > 0)
        p_pos, p_neg = 0.52, 0.48
        pos = [3 / 13, 9 / 26, 7 / 26, 2 / 13]
        neg = [0.28 / 0.48, 0.12 / 0.48, 0.06 / 0.48, 0.02 / 0.48]
        expected = h([.4, .3, .2, .1]) - p_pos * h(pos) - p_neg * h(neg)
        self.assertAlmostEqual(expected_information_gain(PRIOR, "quick_screen", M), expected, delta=1e-9)


def tiny_model(row_a, row_b, test_id="quick_screen"):
    """Default model with cond_a / cond_b rows of quick_screen replaced."""
    lik = lik_copy(test_id)
    lik["cond_a"], lik["cond_b"] = row_a, row_b
    m2 = with_inv(M, test_id, likelihoods=lik)
    validate_model(m2)
    return m2


def beliefs_ab(pa, pb):
    return BeliefState({"cond_a": pa, "cond_b": pb, "cond_c": 0.0, "cond_d": 0.0})


class TestLowProbabilityEvidence(unittest.TestCase):
    def test_exact_zero_is_impossible(self):
        m2 = tiny_model({"negative": 1.0, "positive": 0.0}, {"negative": 1.0, "positive": 0.0})
        with self.assertRaises(ImpossibleObservationError):
            update_beliefs(beliefs_ab(0.5, 0.5), "quick_screen", "positive", m2)

    def test_zero_prior_on_only_supporting_hypothesis_is_impossible(self):
        m2 = tiny_model({"negative": 0.0, "positive": 1.0}, {"negative": 1.0, "positive": 0.0})
        # all prior mass on cond_b, which cannot produce "positive"
        with self.assertRaises(ImpossibleObservationError):
            update_beliefs(beliefs_ab(0.0, 1.0), "quick_screen", "positive", m2)

    def test_small_positive_1e_minus_15_accepted(self):
        m2 = tiny_model({"negative": 1.0 - 1e-15, "positive": 1e-15}, {"negative": 1.0, "positive": 0.0})
        b = beliefs_ab(0.5, 0.5)
        self.assertAlmostEqual(outcome_probability(b, "quick_screen", m2)["positive"], 5e-16, delta=1e-25)
        post = update_beliefs(b, "quick_screen", "positive", m2).probabilities
        self.assertEqual(post["cond_a"], 1.0)
        self.assertEqual(post["cond_b"], 0.0)

    def test_tiny_normal_range_1e_minus_300(self):
        m2 = tiny_model({"negative": 1.0, "positive": 1e-300}, {"negative": 1.0, "positive": 3e-300})
        post = update_beliefs(beliefs_ab(0.5, 0.5), "quick_screen", "positive", m2).probabilities
        self.assertAlmostEqual(post["cond_a"], 0.25, delta=1e-9)
        self.assertAlmostEqual(post["cond_b"], 0.75, delta=1e-9)

    def test_subnormal_evidence_uses_accurate_path(self):
        la, lb = 1e-320, 3e-320  # subnormal doubles; evidence is subnormal
        m2 = tiny_model({"negative": 1.0, "positive": la}, {"negative": 1.0, "positive": lb})
        post = update_beliefs(beliefs_ab(0.5, 0.5), "quick_screen", "positive", m2).probabilities
        self.assertAlmostEqual(post["cond_a"], la / (la + lb), delta=1e-9)
        self.assertAlmostEqual(math.fsum(post.values()), 1.0, delta=1e-12)

    def test_underflowing_product_still_possible(self):
        # P(h_a)=1e-200 and P(o|h_a)=1e-200: product 1e-400 underflows to 0.0 in
        # double precision, but the observation is mathematically possible.
        m2 = tiny_model({"negative": 1.0, "positive": 1e-200}, {"negative": 1.0, "positive": 0.0})
        b = BeliefState({"cond_a": 1e-200, "cond_b": 1.0, "cond_c": 0.0, "cond_d": 0.0})
        post = update_beliefs(b, "quick_screen", "positive", m2).probabilities
        self.assertEqual(post["cond_a"], 1.0)
        self.assertEqual(post["cond_b"], 0.0)
        g = expected_information_gain(b, "quick_screen", m2)
        self.assertTrue(math.isfinite(g) and g >= -1e-9)

    def test_normal_case_unchanged_by_safeguard(self):
        post = update_beliefs(PRIOR, "quick_screen", "positive", M).probabilities
        self.assertAlmostEqual(post["cond_a"], 3 / 13, delta=1e-12)


class TestAppConfigSettings(unittest.TestCase):
    def cfg(self, **kw):
        return dataclasses.replace(build_default_app_config(), **kw)

    def planner(self, **kw):
        return self.cfg(planner=PlannerSettings(**kw))

    def test_defaults_valid_and_documented_values(self):
        c = build_default_app_config()
        validate_config(c)
        self.assertEqual((c.planner.initial_budget, c.planner.confidence_threshold,
                          c.planner.cost_penalty_lambda, c.planner.max_state_expansions),
                         (10.0, 0.90, 0.05, 10_000))
        self.assertEqual((c.evaluation.num_episodes, c.random_seed), (200, 42))

    def test_boundary_values_accepted(self):
        validate_config(self.planner(initial_budget=1.0, confidence_threshold=1.0,
                                     cost_penalty_lambda=0, max_state_expansions=1))
        validate_config(self.cfg(evaluation=EvaluationSettings(num_episodes=1_000_000), random_seed=0))

    def test_invalid_budget(self):
        for v in (0, -1.0, float("nan"), float("inf"), True, "10", 0.5):  # 0.5 < cheapest cost 1.0
            with self.subTest(v=v), self.assertRaises(ConfigError):
                validate_config(self.planner(initial_budget=v))

    def test_invalid_threshold(self):
        for v in (0, -0.1, 1.0001, float("nan"), True):
            with self.subTest(v=v), self.assertRaises(ConfigError):
                validate_config(self.planner(confidence_threshold=v))

    def test_invalid_lambda(self):
        for v in (-0.01, float("nan"), float("inf"), False, None):
            with self.subTest(v=v), self.assertRaises(ConfigError):
                validate_config(self.planner(cost_penalty_lambda=v))

    def test_invalid_expansion_limit(self):
        for v in (0, -5, 10.0, True, None):
            with self.subTest(v=v), self.assertRaises(ConfigError):
                validate_config(self.planner(max_state_expansions=v))

    def test_invalid_seed(self):
        for v in (-1, 1.5, True, None):
            with self.subTest(v=v), self.assertRaises(ConfigError):
                validate_config(self.cfg(random_seed=v))

    def test_invalid_episodes(self):
        for v in (0, -1, 1_000_001, 10.0, True):
            with self.subTest(v=v), self.assertRaises(ConfigError):
                validate_config(self.cfg(evaluation=EvaluationSettings(num_episodes=v)))

    def test_wrong_nested_types(self):
        with self.assertRaises(ConfigError):
            validate_config(self.cfg(planner={"initial_budget": 10}))
        with self.assertRaises(ConfigError):
            validate_config(self.cfg(evaluation=None))


class TestImmutability(unittest.TestCase):
    def test_belief_state_read_only(self):
        b = BeliefState({"cond_a": 1.0})
        with self.assertRaises(TypeError):
            b.probabilities["cond_a"] = 0.5
        with self.assertRaises(dataclasses.FrozenInstanceError):
            b.probabilities = {}

    def test_belief_state_defensive_copy(self):
        src = dict(M.priors)
        b = BeliefState(src)
        src["cond_a"] = 0.0
        self.assertEqual(b.probabilities["cond_a"], 0.4)

    def test_belief_state_equality_and_hash(self):
        a, b = BeliefState(dict(M.priors)), BeliefState(dict(M.priors))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)
        self.assertNotEqual(a, UNIFORM)

    def test_belief_state_rejects_non_mapping(self):
        with self.assertRaises(BeliefError):
            BeliefState([("cond_a", 1.0)])

    def test_model_nested_mappings_read_only(self):
        with self.assertRaises(TypeError):
            M.priors["cond_a"] = 0.9
        inv = M.investigation("quick_screen")
        with self.assertRaises(TypeError):
            inv.likelihoods["cond_a"] = {}
        with self.assertRaises(TypeError):
            inv.likelihoods["cond_a"]["positive"] = 0.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            M.priors = {}
        self.assertIsInstance(M.hypotheses, tuple)
        self.assertIsInstance(inv.outcomes, tuple)

    def test_model_defensive_copy_of_inputs(self):
        pri = dict(M.priors)
        rows = lik_copy("quick_screen")
        outs = list(M.investigation("quick_screen").outcomes)
        inv = Investigation("t", "T", 1.0, outs, rows)
        m2 = ModelConfig(list(M.hypotheses), pri, [inv])
        validate_model(m2)
        pri["cond_a"] = 0.0
        rows["cond_a"]["positive"] = 0.0
        outs.append("extra")
        validate_model(m2)  # still valid: the model kept its own copies
        self.assertEqual(m2.priors["cond_a"], 0.4)
        self.assertEqual(len(m2.investigation("t").outcomes), 2)

    def test_malformed_container_types_rejected(self):
        with self.assertRaises(ConfigError):
            dataclasses.replace(M, priors=[0.4, 0.3, 0.2, 0.1])
        with self.assertRaises(ConfigError):
            Investigation("t", "T", 1.0, "ab", {})

    def test_updates_do_not_mutate_inputs(self):
        before = dict(PRIOR.probabilities)
        update_beliefs(PRIOR, "imaging", "positive", M)
        self.assertEqual(dict(PRIOR.probabilities), before)

    def test_initial_beliefs_equals_priors(self):
        self.assertEqual(initial_beliefs(M).probabilities, M.priors)


# =========================== Phase 2: graph search ===========================
def graph_problem(adj, start, goal):
    return unit_cost_problem(start, lambda s: adj.get(s, []), lambda s: s == goal)


def weighted_problem(edges, start, goal):
    return SearchProblem(start, lambda s: [(n, n, c) for n, c in edges.get(s, [])], lambda s: s == goal)


class TestSearch(unittest.TestCase):
    ADJ = {"S": ["A", "B"], "A": ["C"], "B": ["C", "G"], "C": ["G"]}

    def test_bfs_shortest_path_and_counts(self):
        r = breadth_first_search(graph_problem(self.ADJ, "S", "G"))
        self.assertTrue(r.found)
        self.assertEqual(r.path, ("S", "B", "G"))      # 2 edges; S-A-C-G has 3
        self.assertEqual(r.path_cost, 2.0)
        self.assertEqual(r.expansion_order, ("S", "A", "B"))  # G detected while expanding B
        self.assertEqual(r.expanded_states, 3)

    def test_bfs_min_edges_not_min_cost(self):
        edges = {"S": [("G", 10.0), ("A", 1.0)], "A": [("G", 1.0)]}
        r = breadth_first_search(weighted_problem(edges, "S", "G"))
        self.assertEqual(r.path, ("S", "G"))
        self.assertEqual(r.path_cost, 10.0)

    def test_dfs_deterministic_order_and_cycle(self):
        adj = {"S": ["A", "B"], "A": ["S", "C"], "C": ["A", "B"], "B": ["S"]}
        r = depth_first_search(graph_problem(adj, "S", "Z"))  # unreachable, cyclic
        self.assertEqual(r.status, SearchStatus.UNREACHABLE)
        self.assertEqual(r.expansion_order, ("S", "A", "C", "B"))
        self.assertEqual(r.expanded_states, 4)
        self.assertEqual(r.path, ())
        self.assertIsNone(r.path_cost)
        r2 = depth_first_search(graph_problem(adj, "S", "B"))
        self.assertEqual(r2.path, ("S", "A", "C", "B"))        # depth-first, not shortest
        self.assertEqual(r2.path_cost, 3.0)
        self.assertEqual(r2.expanded_states, 3)

    def test_start_is_goal(self):
        p = graph_problem(self.ADJ, "S", "S")
        for search in (breadth_first_search, depth_first_search,
                       lambda q: a_star_search(q, lambda s: 0.0)):
            r = search(p)
            self.assertTrue(r.found)
            self.assertEqual((r.path, r.actions, r.path_cost, r.expanded_states), (("S",), (), 0.0, 0))

    def test_unreachable_goal_all_algorithms(self):
        p = graph_problem(self.ADJ, "S", "Z")
        for search in (breadth_first_search, depth_first_search,
                       lambda q: a_star_search(q, lambda s: 0.0)):
            r = search(p)
            self.assertEqual(r.status, SearchStatus.UNREACHABLE)
            self.assertFalse(r.found)
            self.assertEqual(r.expanded_states, 5)  # S, A, B, C, G all reachable

    def test_a_star_optimal_with_competing_paths(self):
        edges = {"S": [("G", 10.0), ("A", 2.0)], "A": [("B", 2.0)], "B": [("G", 2.0)]}
        r = a_star_search(weighted_problem(edges, "S", "G"), lambda s: 0.0)
        self.assertEqual(r.path, ("S", "A", "B", "G"))
        self.assertEqual(r.path_cost, 6.0)
        self.assertEqual(r.actions, ("A", "B", "G"))

    def test_a_star_inconsistent_heuristic_reopens(self):
        # h is admissible (never above true remaining cost) but inconsistent:
        # h(A)=11 > c(A,C)+h(C)=1. C is first expanded via B at g=4, then must be
        # re-opened when A offers g=2. Without re-opening the cost would be 14.
        edges = {"S": [("A", 1.0), ("B", 1.0)], "B": [("C", 3.0)], "A": [("C", 1.0)], "C": [("G", 10.0)]}
        h = {"S": 0.0, "A": 11.0, "B": 0.0, "C": 0.0, "G": 0.0}.__getitem__
        r = a_star_search(weighted_problem(edges, "S", "G"), h)
        self.assertEqual(r.path, ("S", "A", "C", "G"))
        self.assertEqual(r.path_cost, 12.0)
        self.assertEqual(r.expansion_order, ("S", "B", "C", "A", "C"))
        self.assertEqual(r.reopened_states, 1)
        self.assertEqual(r.expanded_states, 5)

    def test_a_star_stale_entries_and_better_heuristic_expands_less(self):
        edges = {"S": [("A", 1.0), ("B", 5.0)], "A": [("B", 1.0)], "B": [("G", 1.0)]}
        r = a_star_search(weighted_problem(edges, "S", "G"), lambda s: 0.0)
        self.assertEqual((r.path, r.path_cost), (("S", "A", "B", "G"), 3.0))
        exact = {"S": 3.0, "A": 2.0, "B": 1.0, "G": 0.0}.__getitem__
        r2 = a_star_search(weighted_problem(edges, "S", "G"), exact)
        self.assertEqual(r2.path_cost, 3.0)
        self.assertLessEqual(r2.expanded_states, r.expanded_states)

    def test_invalid_costs_and_heuristic_rejected(self):
        bad = weighted_problem({"S": [("G", -1.0)]}, "S", "G")
        with self.assertRaises(ValueError):
            a_star_search(bad, lambda s: 0.0)
        with self.assertRaises(ValueError):
            breadth_first_search(bad)
        with self.assertRaises(ValueError):
            a_star_search(weighted_problem({"S": [("G", 1.0)]}, "S", "G"), lambda s: -1.0)

    def test_search_expansion_limit(self):
        r = breadth_first_search(graph_problem(self.ADJ, "S", "G"), max_expansions=1)
        self.assertEqual(r.status, SearchStatus.LIMIT_REACHED)
        self.assertFalse(r.found)


# ========================= Phase 2: contingent planning =========================
def mk_model(priors, tests):
    """tests: list of (id, cost, outcomes, {hypothesis: row of probabilities})."""
    hyps = tuple(Hypothesis(h, h) for h in priors)
    invs = tuple(Investigation(i, i, c, tuple(o), {h: dict(zip(o, row)) for h, row in rows.items()})
                 for i, c, o, rows in tests)
    m = ModelConfig(hyps, dict(priors), invs)
    validate_model(m)
    return m


def psettings(lam=0.1, budget=10.0, limit=10_000, thr=0.9):
    return PlannerSettings(initial_budget=budget, confidence_threshold=thr,
                           cost_penalty_lambda=lam, max_state_expansions=limit)


def start(model, settings):
    return make_initial_state(initial_beliefs(model), settings)


TINY = mk_model({"h1": 0.5, "h2": 0.5},
                [("T", 1.0, ("pos", "neg"), {"h1": (0.9, 0.1), "h2": (0.2, 0.8)})])

# T0 splits {h1,h2} | {h3,h4}; U separates h1/h2 (blind to h3/h4); W separates h3/h4.
SPLIT = mk_model({"h1": .25, "h2": .25, "h3": .25, "h4": .25}, [
    ("T0", 0.5, ("left", "right"), {"h1": (1, 0), "h2": (1, 0), "h3": (0, 1), "h4": (0, 1)}),
    ("U", 1.0, ("a", "b"), {"h1": (1, 0), "h2": (0, 1), "h3": (.5, .5), "h4": (.5, .5)}),
    ("W", 1.0, ("a", "b"), {"h1": (.5, .5), "h2": (.5, .5), "h3": (1, 0), "h4": (0, 1)}),
])

# T1 isolates h1; U separates h2 from h3 (uninformative on h1).
THREE = mk_model({"h1": 1 / 3, "h2": 1 / 3, "h3": 1 / 3}, [
    ("T1", 1.0, ("x", "y"), {"h1": (1, 0), "h2": (0, 1), "h3": (0, 1)}),
    ("U", 1.0, ("x", "y"), {"h1": (.5, .5), "h2": (1, 0), "h3": (0, 1)}),
])


def brute_force_value(model, lam, budget):
    """Independent check: unnormalized-joint recursion (no division, no model code).
    U(w) = max(max w, max_t[-lam*c*sum(w) + sum_o U(w * L(o|.))]); V(prior) = U(prior)."""
    def rec(w, done, left):
        best = max(w.values())
        total = sum(w.values())
        for inv in model.investigations:
            if inv.id in done or inv.cost > left + 1e-9:
                continue
            v = -lam * inv.cost * total
            for o in inv.outcomes:
                v += rec({h: w[h] * inv.likelihoods[h][o] for h in w}, done | {inv.id}, left - inv.cost)
            best = max(best, v)
        return best
    return rec(dict(model.priors), frozenset(), budget)


def walk(node, path=()):
    """Yield the list of investigations along every root-to-leaf path of a policy tree."""
    if node.action is None:
        yield path
        return
    for br in node.branches:
        if br.reachable:
            yield from walk(br.child, path + (node.action,))


def all_nodes(node):
    seen, stack = {}, [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen[id(n)] = n
        stack.extend(b.child for b in n.branches if b.child is not None)
    return list(seen.values())


class TestContingentPlanner(unittest.TestCase):
    def test_hand_calculated_root_action(self):
        # b=(.5,.5); T: joint pos=(.45,.10), neg=(.05,.40) -> P(pos)=.55, P(neg)=.45
        # E[A] = .45 + .40 = .85; net = .85 - 0.1*1 = .75 > stop .5
        s = psettings(lam=0.1)
        r = solve_contingent_plan(start(TINY, s), TINY, s)
        self.assertEqual(r.status, PlanStatus.COMPLETE)
        self.assertEqual(r.root_action, "T")
        self.assertFalse(r.stops_at_root)
        self.assertAlmostEqual(r.expected_utility, 0.75, delta=1e-12)
        self.assertAlmostEqual(r.stop_utility, 0.5, delta=1e-12)
        self.assertEqual(r.expanded_states, 3)  # root + two successors
        self.assertTrue(r.is_optimal)

    def test_chance_branches_and_successor_beliefs(self):
        s = psettings(lam=0.1)
        node = solve_contingent_plan(start(TINY, s), TINY, s).policy_tree
        br = {b.outcome: b for b in node.branches}
        self.assertAlmostEqual(br["pos"].probability, 0.55, delta=1e-12)
        self.assertAlmostEqual(br["neg"].probability, 0.45, delta=1e-12)
        self.assertAlmostEqual(br["pos"].posterior.probabilities["h1"], 9 / 11, delta=1e-12)
        self.assertAlmostEqual(br["neg"].posterior.probabilities["h2"], 8 / 9, delta=1e-12)
        self.assertAlmostEqual(br["pos"].child.terminal_value, 9 / 11, delta=1e-12)
        self.assertEqual(br["pos"].child.terminal_decision, "h1")
        self.assertEqual(br["neg"].child.terminal_decision, "h2")
        self.assertIsNone(br["pos"].child.action)  # T cannot be repeated -> must stop
        self.assertAlmostEqual(br["pos"].child.remaining_budget, 9.0, delta=1e-12)
        self.assertEqual(br["pos"].child.performed, ("T",))

    def test_stop_preferred_when_cost_too_high(self):
        # lambda=.4: investigating nets .85-.4=.45 < stop .5
        s = psettings(lam=0.4)
        r = solve_contingent_plan(start(TINY, s), TINY, s)
        self.assertTrue(r.stops_at_root)
        self.assertIsNone(r.root_action)
        self.assertAlmostEqual(r.expected_utility, 0.5, delta=1e-12)
        self.assertEqual(r.policy_tree.branches, ())
        self.assertEqual(r.expanded_states, 3)  # successors are still expanded to price the test

    def test_tie_prefers_stop(self):
        s = psettings(lam=0.35)  # .85 - .35 == .5
        r = solve_contingent_plan(start(TINY, s), TINY, s)
        self.assertTrue(r.stops_at_root)

    def test_planner_may_stop_before_confidence_threshold(self):
        # beliefs .5 < threshold .9, yet planner stops because lambda is high
        s = psettings(lam=0.4, thr=0.9)
        r = solve_contingent_plan(start(TINY, s), TINY, s)
        self.assertTrue(r.stops_at_root)

    def test_planner_may_continue_past_threshold(self):
        # beliefs (.95,.05) already >= threshold .9, but a free informative test still pays
        m = mk_model({"h1": .95, "h2": .05}, [("T", 0.0, ("x", "y"), {"h1": (0, 1), "h2": (1, 0)})])
        s = psettings(lam=0.1, thr=0.9)
        r = solve_contingent_plan(start(m, s), m, s)
        self.assertEqual(r.root_action, "T")           # perfect free test: utility 1.0 > .95
        self.assertAlmostEqual(r.expected_utility, 1.0, delta=1e-12)

    def test_different_outcomes_lead_to_different_next_tests(self):
        # Hand value: -0.1*0.5 (T0) + 1*(-0.1*1 + 1) = 0.85
        s = psettings(lam=0.1, budget=3.0)
        r = solve_contingent_plan(start(SPLIT, s), SPLIT, s)
        self.assertEqual(r.root_action, "T0")
        self.assertAlmostEqual(r.expected_utility, 0.85, delta=1e-12)
        br = {b.outcome: b for b in r.policy_tree.branches}
        self.assertEqual(br["left"].child.action, "U")
        self.assertEqual(br["right"].child.action, "W")
        self.assertAlmostEqual(br["left"].probability, 0.5, delta=1e-12)
        for b in br.values():
            self.assertAlmostEqual(b.child.value, 0.9, delta=1e-12)   # 1 - 0.1*1
        self.assertAlmostEqual(brute_force_value(SPLIT, 0.1, 3.0), 0.85, delta=1e-12)

    def test_budget_binds_on_every_branch(self):
        # Hand values: budget 2.0 -> -0.05 + 1/3*1 + 2/3*(1-0.05) = 11/12
        #              budget 1.5 -> one test only: 37/60
        s2 = psettings(lam=0.05, budget=2.0)
        r2 = solve_contingent_plan(start(THREE, s2), THREE, s2)
        self.assertAlmostEqual(r2.expected_utility, 11 / 12, delta=1e-12)
        self.assertEqual(max(len(p) for p in walk(r2.policy_tree)), 2)
        s15 = psettings(lam=0.05, budget=1.5)
        r15 = solve_contingent_plan(start(THREE, s15), THREE, s15)
        self.assertAlmostEqual(r15.expected_utility, 37 / 60, delta=1e-12)
        self.assertEqual(max(len(p) for p in walk(r15.policy_tree)), 1)
        for node in all_nodes(r15.policy_tree):
            self.assertGreaterEqual(node.remaining_budget, 0.0)

    def test_budget_never_exceeded_default_model(self):
        s = psettings(lam=0.02, budget=7.0)
        r = solve_contingent_plan(start(M, s), M, s)
        costs = {i.id: i.cost for i in M.investigations}
        for path in walk(r.policy_tree):
            self.assertLessEqual(sum(costs[t] for t in path), 7.0 + 1e-9)
            self.assertEqual(len(path), len(set(path)))       # no repeated investigation
        self.assertTrue(any(len(p) >= 2 for p in walk(r.policy_tree)))

    def test_zero_probability_outcome_is_unreachable(self):
        m = mk_model({"h1": .5, "h2": .5}, [("T", 1.0, ("pos", "neg", "never"),
                                            {"h1": (.9, .1, 0.0), "h2": (.2, .8, 0.0)})])
        s = psettings(lam=0.1)
        r = solve_contingent_plan(start(m, s), m, s)
        self.assertAlmostEqual(r.expected_utility, 0.75, delta=1e-12)
        never = next(b for b in r.policy_tree.branches if b.outcome == "never")
        self.assertFalse(never.reachable)
        self.assertEqual(never.probability, 0.0)
        self.assertIsNone(never.posterior)
        self.assertIsNone(never.child)
        self.assertEqual(r.expanded_states, 3)  # unreachable outcome expands nothing

    def test_tiny_positive_probability_branch_is_reachable(self):
        m = mk_model({"h1": .5, "h2": .5}, [("T", 1.0, ("pos", "neg", "rare"),
                                            {"h1": (.9, .1 - 1e-15, 1e-15), "h2": (.2, .8, 0.0)})])
        s = psettings(lam=0.1)
        r = solve_contingent_plan(start(m, s), m, s)
        self.assertEqual(r.root_action, "T")
        rare = next(b for b in r.policy_tree.branches if b.outcome == "rare")
        self.assertTrue(rare.reachable)
        self.assertAlmostEqual(rare.probability, 5e-16, delta=1e-25)
        self.assertEqual(rare.posterior.probabilities["h1"], 1.0)

    def test_expansion_limit_incomplete(self):
        s_full = psettings(lam=0.1)
        full = solve_contingent_plan(start(TINY, s_full), TINY, s_full)
        self.assertEqual(full.expanded_states, 3)
        at_limit = psettings(lam=0.1, limit=3)
        r = solve_contingent_plan(start(TINY, at_limit), TINY, at_limit)
        self.assertEqual(r.status, PlanStatus.COMPLETE)           # needing exactly the limit is fine
        below = psettings(lam=0.1, limit=2)
        r = solve_contingent_plan(start(TINY, below), TINY, below)
        self.assertEqual(r.status, PlanStatus.INCOMPLETE_EXPANSION_LIMIT)
        self.assertFalse(r.is_optimal)
        self.assertIsNone(r.root_action)
        self.assertFalse(r.stops_at_root)
        self.assertIsNone(r.expected_utility)
        self.assertIsNone(r.policy_tree)
        self.assertEqual(r.expanded_states, 2)
        self.assertIn("limit", r.reason)
        self.assertAlmostEqual(r.stop_utility, 0.5, delta=1e-12)  # only a lower bound

    def test_default_model_complete_and_matches_brute_force(self):
        for lam, budget in ((0.05, 6.0), (0.02, 4.0), (0.3, 10.0)):
            s = psettings(lam=lam, budget=budget)
            r = solve_contingent_plan(start(M, s), M, s)
            self.assertEqual(r.status, PlanStatus.COMPLETE)
            self.assertAlmostEqual(r.expected_utility, brute_force_value(M, lam, budget), delta=1e-9)
            self.assertGreaterEqual(r.expected_utility, r.stop_utility - 1e-12)

    def test_value_consistent_inside_tree(self):
        s = psettings(lam=0.05, budget=6.0)
        r = solve_contingent_plan(start(M, s), M, s)
        costs = {i.id: i.cost for i in M.investigations}
        for node in all_nodes(r.policy_tree):
            if node.action is None:
                self.assertAlmostEqual(node.value, node.terminal_value, delta=1e-12)
            else:
                expected = -0.05 * costs[node.action] + sum(
                    b.probability * b.child.value for b in node.branches if b.reachable)
                self.assertAlmostEqual(node.value, expected, delta=1e-9)
                self.assertAlmostEqual(sum(b.probability for b in node.branches), 1.0, delta=1e-9)

    def test_invalid_inputs(self):
        s = psettings()
        st = start(TINY, s)
        with self.assertRaises(PlanningError):
            solve_contingent_plan(st, TINY, PlannerSettings(max_state_expansions=0))
        with self.assertRaises(PlanningError):
            solve_contingent_plan(dataclasses.replace(st, performed=frozenset({"nope"})), TINY, s)
        with self.assertRaises(PlanningError):
            PlanningState(st.beliefs, frozenset(), -1.0)
        with self.assertRaises(PlanningError):
            advance_state(advance_state(st, "T", "pos", TINY), "T", "neg", TINY)

    def test_permitted_set_restricts_actions(self):
        s = psettings(lam=0.1, budget=3.0)
        st = make_initial_state(initial_beliefs(SPLIT), s, permitted={"U"})
        r = solve_contingent_plan(st, SPLIT, s)
        self.assertEqual(r.root_action, "U")
        self.assertTrue(all(set(p) <= {"U"} for p in walk(r.policy_tree)))

    def test_state_keys_belief_hash_consistency(self):
        a, b = BeliefState(dict(M.priors)), BeliefState(dict(M.priors))
        self.assertTrue(a == b and hash(a) == hash(b))
        s = psettings()
        x = advance_state(advance_state(start(M, s), "quick_screen", "positive", M), "imaging", "clear", M)
        y = advance_state(advance_state(start(M, s), "imaging", "clear", M), "quick_screen", "positive", M)
        self.assertEqual(x.performed, y.performed)
        self.assertEqual(x.remaining_budget, y.remaining_budget)
        for h in M.hypothesis_ids():
            self.assertAlmostEqual(x.beliefs.probabilities[h], y.beliefs.probabilities[h], delta=1e-12)


# ================================ Phase 2: agent ================================
PS = psettings(lam=0.1, budget=10.0, thr=0.9)
AS = AgentSettings(planner=PS)


def pstate(model=M, budget=10.0, performed=(), probs=None, settings=PS):
    b = initial_beliefs(model) if probs is None else BeliefState(probs)
    return PlanningState(b, frozenset(performed), budget)


# A: weak but cheap test, B: perfect but expensive. Uniform two hypotheses.
GAIN = mk_model({"h1": .5, "h2": .5}, [
    ("B", 1.0, ("x", "y"), {"h1": (1, 0), "h2": (0, 1)}),
    ("A", 0.25, ("x", "y"), {"h1": (.8, .2), "h2": (.2, .8)}),
])


class TestAgent(unittest.TestCase):
    def test_fixed_order_default_and_custom(self):
        d = choose_action(pstate(), M, "fixed_order", AS)
        self.assertEqual((d.kind, d.test_id), ("investigate", "quick_screen"))
        d = choose_action(pstate(performed=["quick_screen"]), M, "fixed_order", AS)
        self.assertEqual(d.test_id, "marker_panel")
        custom = AgentSettings(planner=PS, fixed_order=("imaging", "history_check"))
        self.assertEqual(choose_action(pstate(), M, "fixed_order", custom).test_id, "imaging")
        # imaging (cost 3) unaffordable with budget 2.5 -> next affordable in custom order
        self.assertEqual(choose_action(pstate(budget=2.5), M, "fixed_order", custom).test_id, "history_check")
        with self.assertRaises(ValueError):
            choose_action(pstate(), M, "fixed_order", AgentSettings(planner=PS, fixed_order=("zzz",)))

    def test_cheapest_first_with_deterministic_tie_break(self):
        self.assertEqual(choose_action(pstate(), M, "cheapest_first", AS).test_id, "quick_screen")
        d = choose_action(pstate(performed=["quick_screen"]), M, "cheapest_first", AS)
        self.assertEqual(d.test_id, "marker_panel")  # cost 2 tie with history_check -> model order
        d = choose_action(pstate(performed=["quick_screen", "marker_panel"]), M, "cheapest_first", AS)
        self.assertEqual(d.test_id, "history_check")
        self.assertEqual(choose_action(pstate(budget=2.5, performed=["quick_screen"]), M,
                                       "cheapest_first", AS).test_id, "marker_panel")

    def test_information_gain_per_cost_ranking(self):
        # Test A: P(x)=P(y)=.5, posterior (.8,.2) -> gain = 1 - H(.8) bits; cost .25
        h8 = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
        gain_a = 1 - h8                      # ~0.27807 bits
        st = pstate(GAIN)
        cands = {c.test_id: c for c in get_candidate_actions(st, GAIN)}
        self.assertAlmostEqual(cands["A"].expected_information_gain_bits, gain_a, delta=1e-9)
        self.assertAlmostEqual(cands["A"].information_gain_per_cost, gain_a / 0.25, delta=1e-9)
        self.assertAlmostEqual(cands["B"].expected_information_gain_bits, 1.0, delta=1e-9)
        self.assertAlmostEqual(cands["B"].information_gain_per_cost, 1.0, delta=1e-9)
        d = choose_action(st, GAIN, "greedy_information_gain", AgentSettings(planner=PS))
        self.assertEqual(d.test_id, "A")     # 1.112 bits/cost beats 1.0 although B has more total gain
        self.assertEqual(d.explanation.selected, cands["A"])

    def test_greedy_zero_cost_and_zero_gain_handling(self):
        m = mk_model({"h1": .5, "h2": .5}, [
            ("free", 0.0, ("x", "y"), {"h1": (1, 0), "h2": (0, 1)}),
            ("blind", 0.0, ("x", "y"), {"h1": (.5, .5), "h2": (.5, .5)})])
        cands = {c.test_id: c for c in get_candidate_actions(pstate(m), m)}
        self.assertEqual(cands["free"].information_gain_per_cost, math.inf)
        self.assertEqual(cands["blind"].information_gain_per_cost, 0.0)
        self.assertEqual(choose_action(pstate(m), m, "greedy_information_gain", AS).test_id, "free")

    def test_baseline_confidence_stopping(self):
        high = {"cond_a": 0.95, "cond_b": 0.03, "cond_c": 0.01, "cond_d": 0.01}
        for pol in ("fixed_order", "cheapest_first", "greedy_information_gain"):
            d = choose_action(pstate(probs=high), M, pol, AS)
            self.assertEqual(d.kind, "stop")
            self.assertEqual(d.explanation.stop_reason, "confidence_threshold_met")
            self.assertAlmostEqual(d.explanation.max_posterior, 0.95, delta=1e-12)
            self.assertIsNone(d.test_id)
        low = {"cond_a": 0.85, "cond_b": 0.05, "cond_c": 0.05, "cond_d": 0.05}
        self.assertEqual(choose_action(pstate(probs=low), M, "fixed_order", AS).kind, "investigate")
        exact = {"cond_a": 0.9, "cond_b": 0.1, "cond_c": 0.0, "cond_d": 0.0}
        sd = should_stop(pstate(probs=exact), AS)
        self.assertTrue(sd.stop)
        self.assertEqual(sd.reason, "confidence_threshold_met")
        self.assertFalse(should_stop(pstate(probs=low), AS).stop)

    def test_no_affordable_investigation_gives_valid_stop(self):
        st = pstate(budget=0.5)
        self.assertEqual(get_candidate_actions(st, M), [])
        for pol in POLICY_NAMES:
            d = choose_action(st, M, pol, AS)
            self.assertEqual((d.kind, d.test_id), ("stop", None))
            self.assertIsInstance(d, Decision)
        d = choose_action(st, M, "fixed_order", AS)
        self.assertEqual(d.explanation.stop_reason, "no_affordable_investigations")
        sd = should_stop(st, AS, M)
        self.assertEqual((sd.stop, sd.reason), (True, "no_affordable_investigations"))
        self.assertIsInstance(sd, StopDecision)

    def test_all_investigations_performed_stops(self):
        st = pstate(performed=[i.id for i in M.investigations])
        self.assertEqual(choose_action(st, M, "cheapest_first", AS).kind, "stop")

    def test_contingent_policy_delegates_to_planner(self):
        s = psettings(lam=0.1)
        agent_s = AgentSettings(planner=s)
        st = start(TINY, s)
        d = choose_action(st, TINY, "contingent_optimal", agent_s)
        plan = solve_contingent_plan(st, TINY, s)
        self.assertEqual((d.kind, d.test_id), ("investigate", plan.root_action))
        self.assertAlmostEqual(d.explanation.expected_net_utility, plan.expected_utility, delta=1e-12)
        self.assertEqual(d.explanation.planner_status, "complete")
        self.assertEqual(d.explanation.planner_expanded_states, plan.expanded_states)
        # SPLIT: contingent picks T0 although another test has higher information gain per cost
        s3 = psettings(lam=0.1, budget=3.0)
        d3 = choose_action(start(SPLIT, s3), SPLIT, "contingent_optimal", AgentSettings(planner=s3))
        self.assertEqual(d3.test_id, "T0")

    def test_contingent_stop_uses_utility_not_threshold(self):
        s = psettings(lam=0.4, thr=0.9)
        a = AgentSettings(planner=s)
        st = start(TINY, s)
        self.assertEqual(choose_action(st, TINY, "contingent_optimal", a).kind, "stop")
        self.assertEqual(choose_action(st, TINY, "contingent_optimal", a).explanation.stop_reason,
                         "planner_stop_optimal")
        # a threshold baseline with the same settings still investigates (0.5 < 0.9)
        self.assertEqual(choose_action(st, TINY, "fixed_order", a).kind, "investigate")

    def test_contingent_incomplete_plan_is_not_a_decision(self):
        s = psettings(lam=0.05, limit=1)
        d = choose_action(start(M, s), M, "contingent_optimal", AgentSettings(planner=s))
        self.assertEqual(d.kind, "no_decision")
        self.assertIsNone(d.test_id)
        self.assertEqual(d.explanation.planner_status, "incomplete_expansion_limit")

    def test_explanation_keeps_quantities_separate(self):
        d = choose_action(pstate(), M, "greedy_information_gain", AS)
        ex = d.explanation
        self.assertIsInstance(ex, Explanation)
        self.assertAlmostEqual(ex.entropy_bits, entropy(initial_beliefs(M)), delta=1e-12)
        self.assertAlmostEqual(ex.max_posterior, 0.4, delta=1e-12)
        self.assertEqual(ex.remaining_budget, 10.0)
        sel = ex.selected
        self.assertIsInstance(sel, CandidateAction)
        self.assertIn("bits", d.explanation.summary)
        self.assertIsNone(ex.expected_net_utility)  # baselines report no utility
        self.assertNotAlmostEqual(sel.cost, sel.expected_information_gain_bits, places=3)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            choose_action(pstate(), M, "random", AS)

    def test_agent_state_has_no_hidden_truth(self):
        import inspect
        forbidden = ("truth", "true_", "hidden", "ground", "label", "actual", "secret")
        for cls in (PlanningState, Decision, Explanation, CandidateAction, StopDecision, AgentSettings):
            for f in dataclasses.fields(cls):
                self.assertFalse(any(w in f.name.lower() for w in forbidden), (cls.__name__, f.name))
        self.assertEqual({f.name for f in dataclasses.fields(PlanningState)},
                         {"beliefs", "performed", "remaining_budget", "permitted"})
        self.assertEqual(list(inspect.signature(choose_action).parameters),
                         ["state", "model", "policy", "settings"])
        self.assertEqual(list(inspect.signature(should_stop).parameters), ["state", "settings", "model"])


# ===================== Phase 2 numerical safety (budget + cache keys) =====================
def exact(x):
    """Test-side exact decimal value of a float (independent of planner internals)."""
    return Fraction(repr(float(x)))


def noisy_pair_model(costs, outcomes_prob=0.7):
    """Two hypotheses, one weakly informative test per cost; more tests always help."""
    tests = [(name, c, ("x", "y"), {"h1": (outcomes_prob, 1 - outcomes_prob),
                                    "h2": (1 - outcomes_prob, outcomes_prob)})
             for name, c in costs]
    return mk_model({"h1": .5, "h2": .5}, tests)


def bit_model(spec):
    """Four hypotheses (uniform). spec: list of (id, cost, kind), kind 'hi' separates
    {h1,h2}|{h3,h4}, 'lo' separates {h1,h3}|{h2,h4}, 'blind' tells nothing.
    Stopping is worth 0.25; one informative test gives 0.5; one 'hi' plus one 'lo' test
    identify the hypothesis exactly (1.0); a 'blind' test or a repeat of the same kind adds nothing."""
    rows = {"hi": {"h1": (1, 0), "h2": (1, 0), "h3": (0, 1), "h4": (0, 1)},
            "lo": {"h1": (1, 0), "h2": (0, 1), "h3": (1, 0), "h4": (0, 1)},
            "blind": {h: (.5, .5) for h in ("h1", "h2", "h3", "h4")}}
    return mk_model({"h1": .25, "h2": .25, "h3": .25, "h4": .25},
                    [(i, c, ("x", "y"), rows[k]) for i, c, k in spec])


class TestBudgetSafety(unittest.TestCase):
    def test_affordability_is_exact_no_tolerance(self):
        self.assertTrue(is_affordable(1.0, 1.0))
        self.assertTrue(is_affordable(1.0, 1.0000000001))
        for rem in (0.9999999999, 0.9999999995, 1.0 - 1e-12, 0.9999999999999999):
            with self.subTest(remaining=rem):
                self.assertFalse(is_affordable(1.0, rem))

    def test_slightly_short_budget_blocks_investigation_everywhere(self):
        m = noisy_pair_model([("A", 1.0)])
        s = psettings(lam=0.0)
        for rem in (0.9999999999, 0.9999999999999999):
            with self.subTest(remaining=rem):
                st = PlanningState(initial_beliefs(m), frozenset(), rem)
                self.assertEqual(available_investigations(st, m), [])
                self.assertEqual(get_candidate_actions(st, m), [])
                with self.assertRaises(PlanningError):
                    advance_state(st, "A", "x", m)
                r = solve_contingent_plan(st, m, s)
                self.assertTrue(r.stops_at_root)
                self.assertEqual(r.expanded_states, 1)
                for pol in POLICY_NAMES:
                    d = choose_action(st, m, pol, AgentSettings(planner=s))
                    self.assertEqual((d.kind, d.test_id), ("stop", None))
        ok = PlanningState(initial_beliefs(m), frozenset(), 1.0)
        self.assertEqual([i.id for i in available_investigations(ok, m)], ["A"])

    def test_budget_is_not_rounded_in_state_transitions(self):
        m = noisy_pair_model([("A", 0.3333333333333)])
        st = PlanningState(initial_beliefs(m), frozenset(), 1.0)
        nxt = advance_state(st, "A", "x", m)
        self.assertEqual(exact(nxt.remaining_budget), exact(1.0) - exact(0.3333333333333))
        self.assertNotEqual(nxt.remaining_budget, round(nxt.remaining_budget, 9))

    def test_decimal_costs_do_not_suffer_float_drift(self):
        # naive floats: 0.3 - 0.1 = 0.19999999999999998 < 0.2 would wrongly block B after A
        m = bit_model([("A", 0.1, "hi"), ("B", 0.2, "lo")])
        st = PlanningState(initial_beliefs(m), frozenset(), 0.3)
        after_a = advance_state(st, "A", "x", m)
        self.assertEqual(after_a.remaining_budget, 0.2)
        self.assertEqual([i.id for i in available_investigations(after_a, m)], ["B"])
        after_b = advance_state(st, "B", "x", m)
        self.assertEqual(after_b.remaining_budget, 0.1)
        self.assertEqual(advance_state(after_b, "A", "y", m).remaining_budget, 0.0)
        s = psettings(lam=0.0, budget=0.3)
        r = solve_contingent_plan(make_initial_state(initial_beliefs(m), s), m, s)
        self.assertAlmostEqual(r.expected_utility, 1.0, delta=1e-12)   # both tests are affordable
        self.assertEqual(max(len(p) for p in walk(r.policy_tree)), 2)

    def test_boundary_budget_cannot_be_exceeded_on_any_branch(self):
        # Q ('hi') and R ('lo') each cost 1.0000000001 and only TOGETHER identify the
        # hypothesis (utility 1.0). P is useless. Q+R = 2.0000000002.
        # Budget 2.0000000001 is 1e-10 SHORT of Q+R, so the best exact plan is to stop
        # one informative test (0.5, not 1.0); after Q the remaining budget is exactly
        # 1.0, which agrees with 1.0000000001 to 9 decimals but cannot pay for R.
        m = bit_model([("P", 1.0, "blind"), ("Q", 1.0000000001, "hi"), ("R", 1.0000000001, "lo")])
        cost_of = {i.id: i.cost for i in m.investigations}
        short = psettings(lam=0.0, budget=2.0000000001)
        r = solve_contingent_plan(make_initial_state(initial_beliefs(m), short), m, short)
        self.assertAlmostEqual(r.expected_utility, 0.5, delta=1e-12)
        for path in walk(r.policy_tree):
            self.assertFalse({"Q", "R"} <= set(path), path)
            self.assertLessEqual(sum(exact(cost_of[t]) for t in path), exact(2.0000000001), path)
        st_q = advance_state(make_initial_state(initial_beliefs(m), short), "Q", "x", m)
        self.assertEqual(st_q.remaining_budget, 1.0)
        self.assertEqual([i.id for i in available_investigations(st_q, m)], ["P"])
        # With exactly Q+R available the plan uses both and reaches utility 1.0.
        enough = psettings(lam=0.0, budget=2.0000000002)
        r2 = solve_contingent_plan(make_initial_state(initial_beliefs(m), enough), m, enough)
        self.assertAlmostEqual(r2.expected_utility, 1.0, delta=1e-12)
        for path in walk(r2.policy_tree):
            self.assertLessEqual(sum(exact(cost_of[t]) for t in path), exact(2.0000000002))
        self.assertTrue(any({"Q", "R"} <= set(p) for p in walk(r2.policy_tree)))

    def test_total_spend_never_exceeds_initial_budget_exactly(self):
        costs = {i.id: i.cost for i in M.investigations}
        for budget in (5.7, 7.0, 4.3, 9.9):
            s = psettings(lam=0.01, budget=budget)
            r = solve_contingent_plan(start(M, s), M, s)
            for path in walk(r.policy_tree):
                self.assertLessEqual(sum(exact(costs[t]) for t in path), exact(budget), (budget, path))


class TestCacheKeys(unittest.TestCase):
    def test_equivalent_orders_share_states(self):
        # Two always-informative tests with all outcomes reachable:
        # states = root 1 + (A,o) 2 + (B,o) 2 + both-done 4 evidence sets = 9
        # (13 if A-then-B and B-then-A were not merged).
        m = noisy_pair_model([("A", 1.0), ("B", 1.0)])
        s = psettings(lam=0.0, budget=5.0)
        r = solve_contingent_plan(make_initial_state(initial_beliefs(m), s), m, s)
        self.assertEqual(r.expanded_states, 9)

    def test_near_identical_beliefs_are_not_merged(self):
        # Outcomes x / y of T give posteriors that differ by ~1e-13 (below the old
        # 12-decimal rounding). They are different evidence, so they must stay
        # separate nodes carrying their own beliefs.
        d = 1e-13
        m = mk_model({"h1": .5, "h2": .5}, [
            ("T", 0.0, ("x", "y"), {"h1": (0.5 + d, 0.5 - d), "h2": (0.5, 0.5)}),
            ("U", 0.0, ("a", "b"), {"h1": (1, 0), "h2": (0, 1)})])
        s = psettings(lam=0.0, budget=1.0)
        r = solve_contingent_plan(make_initial_state(initial_beliefs(m), s), m, s)
        self.assertEqual(r.root_action, "T")  # free; ties among equal-value actions go to the earlier test
        bx, by = r.policy_tree.branches
        self.assertIsNot(bx.child, by.child)
        self.assertNotEqual(bx.posterior, by.posterior)
        self.assertAlmostEqual(bx.posterior.probabilities["h1"], (0.5 + d) / (1 + d), delta=1e-15)
        self.assertAlmostEqual(by.posterior.probabilities["h1"], (0.5 - d) / (1 - d), delta=1e-15)
        self.assertGreater(bx.posterior.probabilities["h1"], by.posterior.probabilities["h1"])

    def test_zero_belief_is_not_merged_with_tiny_positive_belief(self):
        # Prior mass 0.0 vs 1e-14 on h2 must keep different reachable outcomes.
        m = mk_model({"h1": .5, "h2": .5}, [
            ("T", 0.0, ("x", "y"), {"h1": (1.0, 0.0), "h2": (0.0, 1.0)}),
            ("U", 0.0, ("a", "b"), {"h1": (1.0, 0.0), "h2": (0.5, 0.5)})])
        s = psettings(lam=0.0, budget=1.0)
        for p2, reachable in ((0.0, False), (1e-14, True)):
            b = BeliefState({"h1": 1.0 - p2, "h2": p2})
            node = solve_contingent_plan(PlanningState(b, frozenset(), 1.0), m, s).policy_tree
            if node.action == "U":
                bb = {x.outcome: x for x in node.branches}
                self.assertEqual(bb["b"].reachable, reachable)

    def test_planner_agrees_with_brute_force_on_sweep(self):
        worst = 0.0
        for lam in (0.0, 0.01, 0.05, 0.2):
            for budget in (1.0, 3.0, 5.5, 8.0):
                s = psettings(lam=lam, budget=budget)
                r = solve_contingent_plan(start(M, s), M, s)
                self.assertTrue(r.is_optimal)
                worst = max(worst, abs(r.expected_utility - brute_force_value(M, lam, budget)))
        self.assertLess(worst, 1e-9)


if __name__ == "__main__":
    unittest.main()
