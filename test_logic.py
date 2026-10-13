"""Unit tests for the Propositional Logic module.

Verifies grammar parsing, operator precedence, truth-table generation,
satisfiability, tautologies, valid entailments, and fallacy detection.
"""
from __future__ import annotations

import unittest
import logic


class PropositionalLogicTests(unittest.TestCase):
    """Test suite for logic.py."""

    def test_01_all_connectives_and_constants(self) -> None:
        """Verify parsing and evaluation of all supported connectives."""
        assignment = {"P": True, "Q": False}

        # Not
        self.assertFalse(logic.parse_expr("~P").evaluate(assignment))
        self.assertTrue(logic.parse_expr("~Q").evaluate(assignment))

        # And
        self.assertFalse(logic.parse_expr("P & Q").evaluate(assignment))
        self.assertTrue(logic.parse_expr("P & ~Q").evaluate(assignment))

        # Or
        self.assertTrue(logic.parse_expr("P | Q").evaluate(assignment))
        self.assertFalse(logic.parse_expr("~P | Q").evaluate(assignment))

        # Implies
        # True -> False is False
        self.assertFalse(logic.parse_expr("P -> Q").evaluate(assignment))
        # False -> False is True
        self.assertTrue(logic.parse_expr("Q -> P").evaluate(assignment))

        # Iff
        self.assertFalse(logic.parse_expr("P <-> Q").evaluate(assignment))
        self.assertTrue(logic.parse_expr("P <-> P").evaluate(assignment))

        # Constants
        self.assertTrue(logic.parse_expr("True").evaluate(assignment))
        self.assertFalse(logic.parse_expr("False").evaluate(assignment))

    def test_02_operator_precedence(self) -> None:
        """Verify strict operator precedence: NOT > AND > OR > IMPLIES > IFF."""
        # 1. NOT binds tighter than AND: ~P & Q is (~P) & Q
        e1 = logic.parse_expr("~P & Q")
        self.assertIsInstance(e1, logic.And)
        self.assertIsInstance(e1.left, logic.Not)

        # 2. AND binds tighter than OR: P & Q | R is (P & Q) | R
        e2 = logic.parse_expr("P & Q | R")
        self.assertIsInstance(e2, logic.Or)
        self.assertIsInstance(e2.left, logic.And)

        # 3. OR binds tighter than IMPLIES: P | Q -> R is (P | Q) -> R
        e3 = logic.parse_expr("P | Q -> R")
        self.assertIsInstance(e3, logic.Implies)
        self.assertIsInstance(e3.left, logic.Or)

        # 4. IMPLIES is right-associative: P -> Q -> R is P -> (Q -> R)
        e4 = logic.parse_expr("P -> Q -> R")
        self.assertIsInstance(e4, logic.Implies)
        self.assertIsInstance(e4.right, logic.Implies)

        # 5. IMPLIES binds tighter than IFF: P -> Q <-> ~P | Q is (P -> Q) <-> (~P | Q)
        e5 = logic.parse_expr("P -> Q <-> ~P | Q")
        self.assertIsInstance(e5, logic.Iff)
        self.assertIsInstance(e5.left, logic.Implies)
        self.assertIsInstance(e5.right, logic.Or)

    def test_03_parentheses_override(self) -> None:
        """Verify parentheses properly override default operator precedence."""
        e1 = logic.parse_expr("~(P & Q)")
        self.assertIsInstance(e1, logic.Not)
        self.assertIsInstance(e1.child, logic.And)

        e2 = logic.parse_expr("P & (Q | R)")
        self.assertIsInstance(e2, logic.And)
        self.assertIsInstance(e2.right, logic.Or)

    def test_04_invalid_syntax_and_tokens_rejected(self) -> None:
        """Verify that malformed expressions raise clear ValueError."""
        with self.assertRaises(ValueError):
            logic.parse_expr("")  # Empty
        with self.assertRaises(ValueError):
            logic.parse_expr("P &")  # Trailing operator
        with self.assertRaises(ValueError):
            logic.parse_expr("(P | Q")  # Unbalanced open paren
        with self.assertRaises(ValueError):
            logic.parse_expr("P | Q)")  # Unbalanced closing paren
        with self.assertRaises(ValueError):
            logic.parse_expr("P @ Q")  # Illegal character

    def test_05_no_arbitrary_code_execution(self) -> None:
        """Verify that attempted Python code injections are strictly rejected by the lexer/parser."""
        malicious = [
            "__import__('os').system('ls')",
            "exec('x = 1')",
            "eval('2 + 2')",
            "open('/etc/passwd')",
            "lambda x: x",
        ]
        for code in malicious:
            with self.assertRaises(ValueError):
                logic.parse_expr(code)

    def test_06_truth_table_generation(self) -> None:
        """Verify truth table dimensions and values for simple formulas."""
        expr = logic.parse_expr("P & Q")
        table = logic.generate_truth_table(expr)
        self.assertEqual(table.variables, ["P", "Q"])
        self.assertEqual(table.total_rows, 4)
        # Exactly 1 row is True (where P=True and Q=True)
        true_rows = [r for r in table.rows if r["values"][0] is True]
        self.assertEqual(len(true_rows), 1)
        self.assertEqual(true_rows[0]["assignment"], {"P": True, "Q": True})

    def test_07_satisfiability_and_tautologies(self) -> None:
        """Verify satisfiability and tautology determination."""
        # Satisfiable
        sat, model = logic.is_satisfiable(logic.parse_expr("P | Q"))
        self.assertTrue(sat)
        self.assertIsNotNone(model)

        # Contradiction: unsatisfiable
        unsat, no_model = logic.is_satisfiable(logic.parse_expr("P & ~P"))
        self.assertFalse(unsat)
        self.assertIsNone(no_model)

        # Tautology: Law of Excluded Middle
        self.assertTrue(logic.is_tautology(logic.parse_expr("P | ~P")))
        # De Morgan Law equivalence
        self.assertTrue(logic.is_tautology(logic.parse_expr("~(P & Q) <-> (~P | ~Q)")))
        # Material implication equivalence
        self.assertTrue(logic.is_tautology(logic.parse_expr("(P -> Q) <-> (~P | Q)")))

        # Not a tautology
        self.assertFalse(logic.is_tautology(logic.parse_expr("P | Q")))

    def test_08_modus_ponens_valid_entailment(self) -> None:
        """Verify Modus Ponens: {P -> Q, P} |= Q."""
        res = logic.run_modus_ponens_demo()
        self.assertTrue(res.is_entailed)
        self.assertEqual(len(res.counterexamples), 0)
        self.assertEqual(res.satisfying_premise_models, 1)  # Only P=True, Q=True
        self.assertEqual(res.total_assignments, 4)

    def test_09_affirming_consequent_invalid_fallacy(self) -> None:
        """Verify Affirming the Consequent: {P -> Q, Q} |/= P is recognized as invalid."""
        res = logic.run_affirming_consequent_demo()
        self.assertFalse(res.is_entailed)
        self.assertEqual(len(res.counterexamples), 1)
        self.assertEqual(res.counterexamples[0], {"P": False, "Q": True})

    def test_10_modus_tollens_and_disjunctive_syllogism(self) -> None:
        """Verify additional canonical valid inferences."""
        # Modus Tollens: {P -> Q, ~Q} |= ~P
        mt = logic.check_entailment(premises=["P -> Q", "~Q"], conclusion="~P")
        self.assertTrue(mt.is_entailed)
        self.assertEqual(len(mt.counterexamples), 0)

        # Disjunctive Syllogism: {P | Q, ~P} |= Q
        ds = logic.check_entailment(premises=["P | Q", "~P"], conclusion="Q")
        self.assertTrue(ds.is_entailed)
        self.assertEqual(len(ds.counterexamples), 0)

    def test_11_max_variable_safeguard(self) -> None:
        """Verify that expressions exceeding maximum variable limit raise ValueError."""
        # 11 distinct variables: A, B, C, D, E, F, G, H, I, J, K
        large_expr = "A & B & C & D & E & F & G & H & I & J & K"
        with self.assertRaises(ValueError) as ctx:
            logic.generate_truth_table(logic.parse_expr(large_expr), max_variables=10)
        self.assertIn("exceeding the maximum safe limit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
