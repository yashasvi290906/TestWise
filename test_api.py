"""Integration and API tests for the TestWise web application.

Verifies end-to-end functionality, mathematical integrity, hidden truth isolation,
evaluation runs, CSV exports, and state management.
"""
from __future__ import annotations

import unittest
from starlette.testclient import TestClient

from app import app


class TestWiseApiIntegrationTests(unittest.TestCase):
    """Test suite covering the complete TestWise REST API."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_01_index_html_served(self) -> None:
        """Verify the root endpoint serves the single-page application."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("TestWise", response.text)
        self.assertIn("Executive Command Center", response.text)

    def test_02_get_config_metadata(self) -> None:
        """Verify model specification and planner defaults."""
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["app_title"], "TestWise")
        self.assertEqual(len(data["hypotheses"]), 4)
        self.assertEqual(len(data["investigations"]), 6)
        self.assertEqual(data["planner_defaults"]["initial_budget"], 10.0)
        self.assertEqual(data["planner_defaults"]["confidence_threshold"], 0.90)
        self.assertIn("greedy_information_gain", data["available_policies"])
        self.assertIn("contingent_optimal", data["available_policies"])

    def test_03_start_episode_and_hidden_truth_isolation(self) -> None:
        """Verify episode starts cleanly and hidden ground truth is strictly concealed."""
        response = self.client.post(
            "/api/episode/start",
            json={
                "seed": 42,
                "budget": 10.0,
                "policy": "greedy_information_gain",
                "confidence_threshold": 0.90,
            },
        )
        self.assertEqual(response.status_code, 200)
        ep = response.json()

        self.assertIn("session_id", ep)
        self.assertEqual(ep["seed"], 42)
        self.assertEqual(ep["remaining_budget"], 10.0)
        self.assertFalse(ep["is_terminal"])
        self.assertEqual(len(ep["candidates"]), 6)

        # CRITICAL SECURITY/PEDAGOGICAL ASSERTION:
        # Hidden truth MUST BE None during active episode
        self.assertIsNone(ep["hidden_truth"])

    def test_04_step_episode_and_state_transition(self) -> None:
        """Verify autonomous stepping advances state and deducts budget."""
        start_res = self.client.post(
            "/api/episode/start",
            json={"seed": 42, "budget": 10.0, "policy": "greedy_information_gain"},
        )
        session_id = start_res.json()["session_id"]

        step_res = self.client.post("/api/episode/step", json={"session_id": session_id})
        self.assertEqual(step_res.status_code, 200)
        stepped = step_res.json()

        self.assertEqual(stepped["steps_taken"], 1)
        self.assertLess(stepped["remaining_budget"], 10.0)
        self.assertEqual(len(stepped["history"]), 1)

        rec = stepped["history"][0]
        self.assertIn(rec["test_id"], {"quick_screen", "marker_panel", "imaging", "specialist_review", "history_check", "extended_panel"})
        self.assertIn("outcome", rec)
        self.assertIn("prior_beliefs", rec)
        self.assertIn("posterior_beliefs", rec)

    def test_05_manual_investigation_selection(self) -> None:
        """Verify student can manually select and execute a specific test."""
        start_res = self.client.post(
            "/api/episode/start",
            json={"seed": 42, "budget": 10.0, "policy": "cheapest_first"},
        )
        session_id = start_res.json()["session_id"]

        # Manually perform quick_screen (cost 1.0)
        inv_res = self.client.post(
            "/api/episode/investigate",
            json={"session_id": session_id, "test_id": "quick_screen"},
        )
        self.assertEqual(inv_res.status_code, 200)
        state = inv_res.json()

        self.assertEqual(state["steps_taken"], 1)
        self.assertEqual(state["remaining_budget"], 9.0)
        self.assertEqual(state["history"][0]["test_id"], "quick_screen")

    def test_06_episode_completion_reveals_ground_truth(self) -> None:
        """Verify ground truth is only unlocked when episode reaches terminal state."""
        start_res = self.client.post(
            "/api/episode/start",
            json={"seed": 42, "budget": 10.0, "policy": "greedy_information_gain"},
        )
        session_id = start_res.json()["session_id"]

        # Step until terminal
        for _ in range(10):
            step_res = self.client.post("/api/episode/step", json={"session_id": session_id})
            ep = step_res.json()
            if ep["is_terminal"]:
                break

        self.assertTrue(ep["is_terminal"])
        self.assertIsNotNone(ep["hidden_truth"])
        self.assertIn("hypothesis_id", ep["hidden_truth"])
        self.assertIn("is_correct", ep["hidden_truth"])
        self.assertIsInstance(ep["hidden_truth"]["is_correct"], bool)

    def test_07_solve_contingent_plan(self) -> None:
        """Verify Bellman contingent planner produces valid policy tree."""
        response = self.client.post(
            "/api/plan/solve",
            json={"budget": 10.0, "confidence_threshold": 0.90},
        )
        self.assertEqual(response.status_code, 200)
        plan = response.json()

        self.assertEqual(plan["status"], "complete")
        self.assertGreater(plan["expanded_states"], 0)
        self.assertIsNotNone(plan["policy_tree"])
        self.assertIn("action", plan["policy_tree"])
        self.assertGreater(len(plan["policy_tree"]["branches"]), 0)

    def test_08_run_evaluation_and_export_csv(self) -> None:
        """Verify batch multi-policy evaluation and deterministic CSV generation."""
        eval_res = self.client.post(
            "/api/evaluation/run",
            json={
                "num_episodes": 4,
                "seed": 42,
                "budget": 10.0,
                "confidence_threshold": 0.90,
            },
        )
        self.assertEqual(eval_res.status_code, 200)
        report = eval_res.json()

        self.assertEqual(report["num_episodes"], 4)
        self.assertEqual(report["total_runs"], 16)
        self.assertEqual(len(report["summaries"]), 4)
        self.assertIn("greedy_information_gain", report["summaries"])

        report_id = report["report_id"]
        csv_res = self.client.get(f"/api/evaluation/export?report_id={report_id}")
        self.assertEqual(csv_res.status_code, 200)
        self.assertEqual(csv_res.headers["content-type"], "text/csv; charset=utf-8")

        csv_text = csv_res.text
        self.assertIn("seed", csv_text)
        self.assertIn("policy", csv_text)
        self.assertIn("is_correct", csv_text)
        self.assertIn("greedy_information_gain", csv_text)

    def test_09_algorithm_lab_budget_curves(self) -> None:
        """Verify empirical accuracy vs budget sweeps."""
        response = self.client.get("/api/algorithm-lab/budget-curve?seed=42&episodes=5")
        self.assertEqual(response.status_code, 200)
        curve = response.json()

        self.assertIn("budgets", curve)
        self.assertIn("curve_data", curve)
        self.assertEqual(len(curve["curve_data"]), 4)

    def test_10_error_handling_and_input_validation(self) -> None:
        """Verify API cleanly rejects invalid requests with standard 400/404 codes."""
        # Unknown policy
        bad_pol = self.client.post(
            "/api/episode/start",
            json={"seed": 42, "budget": 10.0, "policy": "nonexistent_policy"},
        )
        self.assertEqual(bad_pol.status_code, 400)

        # Negative budget
        bad_bgt = self.client.post(
            "/api/episode/start",
            json={"seed": 42, "budget": -5.0},
        )
        self.assertEqual(bad_bgt.status_code, 422)  # Pydantic validation error

        # Unknown session ID
        unknown_sess = self.client.post(
            "/api/episode/step",
            json={"session_id": "nonexistent-id-12345"},
        )
        self.assertEqual(unknown_sess.status_code, 404)

    def test_11_concurrent_stepping_safety(self) -> None:
        """Verify concurrent requests on the same session do not corrupt state or history."""
        import concurrent.futures

        start_res = self.client.post(
            "/api/episode/start",
            json={"seed": 42, "budget": 10.0, "policy": "greedy_information_gain"},
        )
        session_id = start_res.json()["session_id"]

        def do_step():
            return self.client.post("/api/episode/step", json={"session_id": session_id})

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(do_step) for _ in range(4)]
            results = [f.result() for f in futures]

        # All requests must return 200 without race condition or crash
        for r in results:
            self.assertEqual(r.status_code, 200)

        # Final state check
        final_state = self.client.get(f"/api/episode/state?session_id={session_id}").json()
        self.assertLessEqual(final_state["steps_taken"], final_state["max_steps"])
        self.assertEqual(len(final_state["history"]), final_state["steps_taken"])
        # History step indices must be strictly unique and sequential
        step_indices = [rec["step_index"] for rec in final_state["history"]]
        self.assertEqual(step_indices, list(range(len(step_indices))))

    def test_12_search_algorithms_demo_endpoint(self) -> None:
        """Verify the /api/search/demo endpoint demonstrates BFS, DFS, and A* accurately."""
        res = self.client.get("/api/search/demo")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("problem", data)
        self.assertIn("algorithms", data)

        algos = data["algorithms"]
        self.assertIn("breadth_first_search", algos)
        self.assertIn("depth_first_search", algos)
        self.assertIn("a_star_search", algos)

        bfs = algos["breadth_first_search"]
        self.assertEqual(bfs["path"], ["A", "B", "D", "G"])
        self.assertEqual(bfs["path_cost"], 9.0)
        self.assertEqual(bfs["expanded_states"], 4)

        dfs = algos["depth_first_search"]
        self.assertEqual(dfs["path"], ["A", "B", "D", "G"])
        self.assertEqual(dfs["expanded_states"], 3)

        astar = algos["a_star_search"]
        self.assertEqual(astar["path"], ["A", "C", "E", "G"])
        self.assertEqual(astar["path_cost"], 7.0)
        self.assertEqual(astar["expanded_states"], 4)

    def test_13_session_bounds_eviction(self) -> None:
        """Verify active sessions are bounded to avoid memory exhaustion."""
        from app import adapter
        old_limit = adapter.max_sessions
        adapter.max_sessions = 5

        session_ids = []
        for i in range(7):
            res = self.client.post(
                "/api/episode/start",
                json={"seed": i, "budget": 10.0, "policy": "cheapest_first"},
            )
            session_ids.append(res.json()["session_id"])

        self.assertLessEqual(len(adapter.active_sessions), 5)
        # Oldest session should be evicted
        self.assertNotIn(session_ids[0], adapter.active_sessions)
        # Newest session should be present
        self.assertIn(session_ids[-1], adapter.active_sessions)

        adapter.max_sessions = old_limit

    def test_14_bayes_net_spec_and_demos(self) -> None:
        """Verify Bayesian network specification and demonstration queries."""
        # 1. Spec endpoint
        spec_res = self.client.get("/api/bayes-net/spec")
        self.assertEqual(spec_res.status_code, 200)
        spec = spec_res.json()
        self.assertEqual(len(spec["nodes"]), 4)
        self.assertEqual(len(spec["edges"]), 3)

        # 2. Demo endpoint
        demo_res = self.client.get("/api/bayes-net/demo")
        self.assertEqual(demo_res.status_code, 200)
        demo = demo_res.json()
        self.assertEqual(len(demo["queries"]), 4)
        self.assertEqual(demo["queries"][0]["id"], "p_alarm")

    def test_15_bayes_net_query_and_errors(self) -> None:
        """Verify exact Bayesian inference query and error handling."""
        # Valid query: P(Burglary | JohnCalls=True)
        res = self.client.post(
            "/api/bayes-net/query",
            json={"query_var": "Burglary", "evidence": {"JohnCalls": True}},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["query_var"], "Burglary")
        self.assertAlmostEqual(data["p_true"], 0.0162837, places=5)
        self.assertIn("exact enumeration computes", data["explanation"])

        # Unknown variable error -> 400
        res_bad = self.client.post(
            "/api/bayes-net/query",
            json={"query_var": "NonExistentVar", "evidence": {}},
        )
        self.assertEqual(res_bad.status_code, 400)

    def test_16_logic_truth_table_and_safeguards(self) -> None:
        """Verify truth table generation, tautology detection, and syntax error handling."""
        # 1. Valid formula
        res = self.client.post(
            "/api/logic/truth-table",
            json={"expression": "P -> Q"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["variables"], ["P", "Q"])
        self.assertEqual(data["total_rows"], 4)
        self.assertFalse(data["is_tautology"])
        self.assertTrue(data["is_satisfiable"])

        # 2. Tautology
        res_taut = self.client.post(
            "/api/logic/truth-table",
            json={"expression": "P | ~P"},
        )
        self.assertEqual(res_taut.status_code, 200)
        self.assertTrue(res_taut.json()["is_tautology"])

        # 3. Syntax error -> 400
        res_err = self.client.post(
            "/api/logic/truth-table",
            json={"expression": "P & @#$"},
        )
        self.assertEqual(res_err.status_code, 400)

    def test_17_logic_entailment_and_demos(self) -> None:
        """Verify semantic entailment checking and canonical logic demonstrations."""
        # 1. Valid entailment: Modus Ponens
        res_mp = self.client.post(
            "/api/logic/entailment",
            json={"premises": ["P -> Q", "P"], "conclusion": "Q"},
        )
        self.assertEqual(res_mp.status_code, 200)
        mp_data = res_mp.json()
        self.assertTrue(mp_data["is_entailed"])
        self.assertEqual(len(mp_data["counterexamples"]), 0)

        # 2. Invalid fallacy: Affirming the Consequent
        res_ac = self.client.post(
            "/api/logic/entailment",
            json={"premises": ["P -> Q", "Q"], "conclusion": "P"},
        )
        self.assertEqual(res_ac.status_code, 200)
        ac_data = res_ac.json()
        self.assertFalse(ac_data["is_entailed"])
        self.assertEqual(len(ac_data["counterexamples"]), 1)
        self.assertEqual(ac_data["counterexamples"][0], {"P": False, "Q": True})

        # 3. Demos endpoint
        res_demo = self.client.get("/api/logic/demos")
        self.assertEqual(res_demo.status_code, 200)
        demos = res_demo.json()
        self.assertIn("modus_ponens", demos)
        self.assertIn("affirming_the_consequent", demos)


if __name__ == "__main__":
    unittest.main()

