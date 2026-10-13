"""Unit tests for the Bayesian Network module.

Verifies graph representation, DAG cycle validation, CPT normalization,
exact inference by enumeration, analytical numbers, and edge case handling.
"""
from __future__ import annotations

import unittest
from fractions import Fraction
import bayes_net


class BayesNetTests(unittest.TestCase):
    """Test suite for bayes_net.py."""

    def test_01_canonical_network_construction_and_validation(self) -> None:
        """Verify the 4-variable Alarm network builds cleanly and passes all validation."""
        net = bayes_net.build_canonical_alarm_net()
        self.assertEqual(len(net.nodes), 4)
        self.assertIn("Burglary", net.nodes)
        self.assertIn("Earthquake", net.nodes)
        self.assertIn("Alarm", net.nodes)
        self.assertIn("JohnCalls", net.nodes)

        topo = net.topological_sort()
        # In any valid topological sort, parents must precede their children
        self.assertLess(topo.index("Burglary"), topo.index("Alarm"))
        self.assertLess(topo.index("Earthquake"), topo.index("Alarm"))
        self.assertLess(topo.index("Alarm"), topo.index("JohnCalls"))

    def test_02_cycle_detection(self) -> None:
        """Verify that directed cycles in the network raise ValueError."""
        # Cycle: A -> B -> C -> A
        net = bayes_net.BayesNet()
        net.add_variable("A", (True, False), parents=("C",), cpt={(True,): {True: 0.5, False: 0.5}, (False,): {True: 0.5, False: 0.5}})
        net.add_variable("B", (True, False), parents=("A",), cpt={(True,): {True: 0.5, False: 0.5}, (False,): {True: 0.5, False: 0.5}})
        net.add_variable("C", (True, False), parents=("B",), cpt={(True,): {True: 0.5, False: 0.5}, (False,): {True: 0.5, False: 0.5}})

        with self.assertRaises(ValueError) as ctx:
            net.validate()
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_03_self_cycle_detection(self) -> None:
        """Verify that a node having itself as a parent is rejected."""
        net = bayes_net.BayesNet()
        net.add_variable("Selfie", (True, False), parents=("Selfie",), cpt={(True,): {True: 0.5, False: 0.5}, (False,): {True: 0.5, False: 0.5}})
        with self.assertRaises(ValueError) as ctx:
            net.validate()
        self.assertIn("cannot have itself as a parent", str(ctx.exception))

    def test_04_unknown_parent_rejection(self) -> None:
        """Verify that referencing a non-existent parent raises ValueError."""
        net = bayes_net.BayesNet()
        net.add_variable("Child", (True, False), parents=("GhostParent",), cpt={(True,): {True: 1.0, False: 0.0}})
        with self.assertRaises(ValueError) as ctx:
            net.validate()
        self.assertIn("unknown parent", str(ctx.exception))

    def test_05_invalid_state_spaces(self) -> None:
        """Verify that degenerate state spaces (<2 values or duplicates) raise ValueError."""
        with self.assertRaises(ValueError):
            bayes_net.BayesVariable("Unary", (True,))
        with self.assertRaises(ValueError):
            bayes_net.BayesVariable("Duplicate", (True, True))
        with self.assertRaises(ValueError):
            bayes_net.BayesVariable("", (True, False))

    def test_06_cpt_row_normalization_and_completeness(self) -> None:
        """Verify that non-normalized or incomplete CPT rows are caught by validation."""
        # Row does not sum to 1.0
        net = bayes_net.BayesNet()
        net.add_variable("BadProb", (True, False), parents=(), cpt={(): {True: 0.8, False: 0.8}})
        with self.assertRaises(ValueError) as ctx:
            net.validate()
        self.assertIn("does not sum to 1.0", str(ctx.exception))

        # Missing parent row
        net2 = bayes_net.BayesNet()
        net2.add_variable("P1", (True, False), parents=(), cpt={(): {True: 0.5, False: 0.5}})
        net2.add_variable("Child", (True, False), parents=("P1",), cpt={(True,): {True: 0.5, False: 0.5}})
        with self.assertRaises(ValueError) as ctx2:
            net2.validate()
        self.assertIn("expected 2 rows", str(ctx2.exception))

    def test_07_joint_probability_exact(self) -> None:
        """Verify exact joint probability matches manual analytical formula."""
        net = bayes_net.build_canonical_alarm_net()
        # Case: B=True, E=False, A=True, J=True
        # P(B=T)*P(E=F)*P(A=T|B=T,E=F)*P(J=T|A=T) = 0.001 * 0.998 * 0.94 * 0.90 = 0.000844308
        expected_joint = 0.001 * 0.998 * 0.94 * 0.90
        calc_joint = net.joint_probability({
            "Burglary": True,
            "Earthquake": False,
            "Alarm": True,
            "JohnCalls": True,
        })
        self.assertAlmostEqual(calc_joint, expected_joint, places=9)

    def test_08_marginal_alarm_probability(self) -> None:
        """Verify marginal P(Alarm=True) matches the exact analytical fraction 1258221 / 500000000."""
        net = bayes_net.build_canonical_alarm_net()
        dist = net.marginal("Alarm")
        expected_fraction = float(Fraction(1258221, 500000000))  # 0.002516442
        self.assertAlmostEqual(dist[True], expected_fraction, places=8)
        self.assertAlmostEqual(dist[False], 1.0 - expected_fraction, places=8)

    def test_09_conditional_johncalls_given_alarm(self) -> None:
        """Verify P(JohnCalls=True | Alarm=True) = 0.90."""
        net = bayes_net.build_canonical_alarm_net()
        dist = net.conditional("JohnCalls", {"Alarm": True})
        self.assertAlmostEqual(dist[True], 0.90, places=7)
        self.assertAlmostEqual(dist[False], 0.10, places=7)

    def test_10_conditional_burglary_given_johncalls(self) -> None:
        """Verify diagnostic posterior P(Burglary=True | JohnCalls=True) matches 8490170 / 521389757."""
        net = bayes_net.build_canonical_alarm_net()
        dist = net.conditional("Burglary", {"JohnCalls": True})
        expected = float(Fraction(8490170, 521389757))  # ~0.01628373
        self.assertAlmostEqual(dist[True], expected, places=7)

    def test_11_explaining_away_phenomenon(self) -> None:
        """Verify that learning Burglary=True explains away the Alarm, reducing P(Earthquake)."""
        net = bayes_net.build_canonical_alarm_net()
        p_e_alarm = net.conditional("Earthquake", {"Alarm": True})[True]
        p_e_alarm_burg = net.conditional("Earthquake", {"Alarm": True, "Burglary": True})[True]

        # Prior is 0.002; Alarm increases it to ~0.231; knowing Burglary reduces it back to ~0.002
        self.assertGreater(p_e_alarm, 0.20)
        self.assertAlmostEqual(p_e_alarm_burg, float(Fraction(95, 47001)), places=6)
        self.assertLess(p_e_alarm_burg, p_e_alarm)

    def test_12_query_variable_in_evidence(self) -> None:
        """Verify trivial distribution when the queried variable is part of the evidence."""
        net = bayes_net.build_canonical_alarm_net()
        dist = net.conditional("Alarm", {"Alarm": True, "JohnCalls": True})
        self.assertEqual(dist[True], 1.0)
        self.assertEqual(dist[False], 0.0)

    def test_13_impossible_evidence_handling(self) -> None:
        """Verify that impossible/contradictory evidence raises a clear ValueError."""
        # Create a network with deterministic constraint
        net = bayes_net.BayesNet()
        net.add_variable("A", (True, False), parents=(), cpt={(): {True: 0.5, False: 0.5}})
        net.add_variable("B", (True, False), parents=("A",), cpt={
            (True,): {True: 1.0, False: 0.0},
            (False,): {True: 1.0, False: 0.0},
        })  # B is always True
        net.validate()

        with self.assertRaises(ValueError) as ctx:
            net.conditional("A", {"B": False})
        self.assertIn("zero-probability evidence", str(ctx.exception))

    def test_14_independent_variables(self) -> None:
        """Verify that conditioning on an independent variable does not alter the posterior."""
        net = bayes_net.BayesNet()
        net.add_variable("X", ("H", "T"), parents=(), cpt={(): {"H": 0.4, "T": 0.6}})
        net.add_variable("Y", ("H", "T"), parents=(), cpt={(): {"H": 0.7, "T": 0.3}})
        net.validate()

        dist_x_prior = net.marginal("X")
        dist_x_given_y = net.conditional("X", {"Y": "H"})
        self.assertAlmostEqual(dist_x_given_y["H"], dist_x_prior["H"], places=7)
        self.assertAlmostEqual(dist_x_given_y["T"], dist_x_prior["T"], places=7)

    def test_15_demonstration_suite(self) -> None:
        """Verify that run_alarm_demonstration executes completely and returns structured output."""
        res = bayes_net.run_alarm_demonstration()
        self.assertEqual(res["network"], "Burglary-Earthquake-Alarm-JohnCalls")
        self.assertEqual(len(res["variables"]), 4)
        self.assertEqual(len(res["dag_edges"]), 3)
        self.assertEqual(len(res["queries"]), 4)
        for q in res["queries"]:
            self.assertIn("id", q)
            self.assertIn("query", q)
            self.assertIn("explanation", q)


if __name__ == "__main__":
    unittest.main()
