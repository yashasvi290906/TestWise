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

    def test_18_episode_reset_preserves_session_id_and_restores_state(self) -> None:
        """Verify complete start -> step -> reset -> step lifecycle preserves session ID and state."""
        # 1. Start episode
        start_res = self.client.post(
            "/api/episode/start",
            json={"seed": 42, "budget": 10.0, "policy": "greedy_information_gain"},
        )
        self.assertEqual(start_res.status_code, 200)
        orig_session_id = start_res.json()["session_id"]

        # 2. Step once
        step1_res = self.client.post("/api/episode/step", json={"session_id": orig_session_id})
        self.assertEqual(step1_res.status_code, 200)
        s1 = step1_res.json()
        self.assertEqual(s1["steps_taken"], 1)
        self.assertLess(s1["remaining_budget"], 10.0)
        self.assertEqual(len(s1["history"]), 1)

        # 3. Reset episode
        reset_res = self.client.post("/api/episode/reset", json={"session_id": orig_session_id})
        self.assertEqual(reset_res.status_code, 200)
        r_state = reset_res.json()

        # Contract verification: Session ID preserved, budget restored, history cleared
        self.assertEqual(r_state["session_id"], orig_session_id)
        self.assertEqual(r_state["remaining_budget"], 10.0)
        self.assertEqual(r_state["cumulative_cost"], 0.0)
        self.assertEqual(r_state["steps_taken"], 0)
        self.assertEqual(len(r_state["history"]), 0)
        self.assertFalse(r_state["is_terminal"])
        self.assertIsNone(r_state["hidden_truth"])

        # 4. Step after reset operates on the freshly reset episode
        step2_res = self.client.post("/api/episode/step", json={"session_id": orig_session_id})
        self.assertEqual(step2_res.status_code, 200)
        s2 = step2_res.json()
        self.assertEqual(s2["steps_taken"], 1)
        self.assertLess(s2["remaining_budget"], 10.0)
        self.assertEqual(len(s2["history"]), 1)

    def test_19_reset_unknown_session_returns_404(self) -> None:
        """Verify resetting an unknown or expired session returns HTTP 404."""
        res = self.client.post("/api/episode/reset", json={"session_id": "nonexistent-sess-id"})
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_20_cors_configuration_allows_explicit_origins(self) -> None:
        """Verify CORS headers respond to explicit allowed origins without wildcard credentials."""
        res = self.client.get(
            "/api/config",
            headers={"Origin": "http://localhost:8000"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("access-control-allow-origin"), "http://localhost:8000")
        self.assertEqual(res.headers.get("access-control-allow-credentials"), "true")

        # Disallowed origin should not receive Access-Control-Allow-Origin
        res_disallowed = self.client.get(
            "/api/config",
            headers={"Origin": "http://untrusted-site.org"},
        )
        self.assertNotEqual(res_disallowed.headers.get("access-control-allow-origin"), "http://untrusted-site.org")

    def test_21_concurrent_registry_access_safety(self) -> None:
        """Verify concurrent episode creations and lookups do not corrupt session registry."""
        import concurrent.futures

        def start_and_step(seed_val: int):
            start = self.client.post(
                "/api/episode/start",
                json={"seed": seed_val, "budget": 10.0, "policy": "cheapest_first"},
            )
            sid = start.json()["session_id"]
            step = self.client.post("/api/episode/step", json={"session_id": sid})
            return start.status_code, step.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(start_and_step, i + 100) for i in range(12)]
            results = [f.result() for f in futures]

        for s_code, st_code in results:
            self.assertEqual(s_code, 200)
            self.assertEqual(st_code, 200)

    def test_22_independent_sessions_progress_without_registry_lock_serialization(self) -> None:
        """Prove that independent sessions progress in parallel and are not serialized by registry lock.

        Uses deterministic synchronization primitives (threading.Event) to establish
        that Session B starts and completes an episode step while Session A is held paused
        inside its own session lock.
        """
        import threading
        import simulation

        res_a = self.client.post("/api/episode/start", json={"seed": 101, "budget": 10.0})
        res_b = self.client.post("/api/episode/start", json={"seed": 202, "budget": 10.0})
        sid_a = res_a.json()["session_id"]
        sid_b = res_b.json()["session_id"]

        session_a_in_lock = threading.Event()
        session_a_can_exit = threading.Event()
        session_b_completed = threading.Event()

        orig_step = simulation.step_episode

        def instrumented_step(ep, m, pol, sett):
            # Only intercept Session A's step
            if getattr(ep, "seed", None) == 101:
                session_a_in_lock.set()
                session_a_can_exit.wait(timeout=5.0)
            return orig_step(ep, m, pol, sett)

        simulation.step_episode = instrumented_step
        try:
            thread_a_result = []

            def worker_a():
                res = self.client.post("/api/episode/step", json={"session_id": sid_a})
                thread_a_result.append(res.status_code)

            t_a = threading.Thread(target=worker_a)
            t_a.start()

            # Wait until Session A has acquired its lock and paused inside step
            self.assertTrue(session_a_in_lock.wait(timeout=3.0), "Session A failed to enter lock")

            # At this exact moment: Session A holds session_a.lock and is paused.
            # If adapter held registry_lock across the step, calling step on Session B
            # would block on registry_lock.
            # Now invoke step on Session B:
            res_b_step = self.client.post("/api/episode/step", json={"session_id": sid_b})
            self.assertEqual(res_b_step.status_code, 200)
            self.assertEqual(res_b_step.json()["steps_taken"], 1)
            session_b_completed.set()

            # Unpause Session A
            session_a_can_exit.set()
            t_a.join(timeout=3.0)

            self.assertEqual(thread_a_result, [200])
            self.assertTrue(session_b_completed.is_set())
        finally:
            simulation.step_episode = orig_step
            session_a_can_exit.set()

    def test_23_racing_step_and_reset_deterministic_behavior(self) -> None:
        """Prove that racing reset and step operations on the same session have deterministic, valid outcomes."""
        import concurrent.futures

        res = self.client.post("/api/episode/start", json={"seed": 42, "budget": 10.0})
        sid = res.json()["session_id"]

        def do_step():
            return self.client.post("/api/episode/step", json={"session_id": sid})

        def do_reset():
            return self.client.post("/api/episode/reset", json={"session_id": sid})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_step = executor.submit(do_step)
            f_reset = executor.submit(do_reset)
            r_step = f_step.result()
            r_reset = f_reset.result()

        self.assertEqual(r_step.status_code, 200)
        self.assertEqual(r_reset.status_code, 200)

        # State check: Regardless of whether step or reset won the race,
        # the state must be mathematically consistent (never corrupt or torn).
        final_state = self.client.get(f"/api/episode/state?session_id={sid}").json()
        self.assertIn(final_state["steps_taken"], (0, 1))
        self.assertEqual(len(final_state["history"]), final_state["steps_taken"])
        if final_state["steps_taken"] == 0:
            self.assertEqual(final_state["remaining_budget"], 10.0)
        else:
            self.assertLess(final_state["remaining_budget"], 10.0)

    def test_24_eviction_and_in_flight_operation_safety(self) -> None:
        """Prove that in-flight operations safely complete if a session is evicted after lookup.

        Crucially verifies that the response explicitly signals eviction (is_active: False,
        is_evicted: True, X-Session-Evicted: true) so no successful in-flight response falsely
        implies the session remains available in the registry.
        """
        import threading
        from app import adapter
        import simulation

        res = self.client.post("/api/episode/start", json={"seed": 55, "budget": 10.0})
        self.assertEqual(res.status_code, 200)
        start_payload = res.json()
        sid = start_payload["session_id"]
        self.assertTrue(start_payload["is_active"])
        self.assertFalse(start_payload["is_evicted"])

        # Safely retain session reference before eviction
        session = adapter._lookup_session(sid)
        self.assertFalse(session.is_evicted)

        # Evict the session under registry lock while reference is held
        evicted = adapter.evict_session(sid)
        self.assertTrue(evicted)
        self.assertTrue(session.is_evicted)

        # Subsequent lookup must fail with KeyError (HTTP 404)
        with self.assertRaises(KeyError):
            adapter._lookup_session(sid)
        not_found_res = self.client.get(f"/api/episode/state?session_id={sid}")
        self.assertEqual(not_found_res.status_code, 404)

        # In-flight operation on retained reference completes cleanly under session.lock
        with session.lock:
            simulation.step_episode(session.episode, adapter.model, session.policy, session.settings)
            state_dict = session.to_dict()

        self.assertEqual(state_dict["steps_taken"], 1)
        self.assertLess(state_dict["remaining_budget"], 10.0)
        # Verify explicit notification that session was evicted during operation
        self.assertFalse(state_dict["is_active"])
        self.assertTrue(state_dict["is_evicted"])

        # Also verify end-to-end HTTP behavior: in-flight HTTP step during eviction
        res2 = self.client.post("/api/episode/start", json={"seed": 88, "budget": 10.0})
        sid2 = res2.json()["session_id"]

        step_in_progress = threading.Event()
        can_finish_step = threading.Event()
        http_step_result = []

        orig_step = simulation.step_episode

        def intercept_step(ep, m, pol, sett):
            if getattr(ep, "seed", None) == 88:
                step_in_progress.set()
                can_finish_step.wait(timeout=5.0)
            return orig_step(ep, m, pol, sett)

        simulation.step_episode = intercept_step
        try:
            def client_worker():
                r = self.client.post("/api/episode/step", json={"session_id": sid2})
                http_step_result.append(r)

            t = threading.Thread(target=client_worker)
            t.start()

            # Wait for step to begin
            self.assertTrue(step_in_progress.wait(timeout=3.0))

            # Evict session while step is actively executing
            adapter.evict_session(sid2)

            # Let step finish
            can_finish_step.set()
            t.join(timeout=3.0)

            # In-flight HTTP request finished with 200, but explicitly declares eviction
            self.assertEqual(len(http_step_result), 1)
            step_r = http_step_result[0]
            self.assertEqual(step_r.status_code, 200)
            self.assertFalse(step_r.json()["is_active"])
            self.assertTrue(step_r.json()["is_evicted"])
            self.assertEqual(step_r.headers.get("x-session-evicted"), "true")

            # Subsequent requests for this session immediately receive 404
            next_step_r = self.client.post("/api/episode/step", json={"session_id": sid2})
            self.assertEqual(next_step_r.status_code, 404)
        finally:
            simulation.step_episode = orig_step
            can_finish_step.set()

    def test_25_snapshot_consistency_during_active_state_transitions(self) -> None:
        """Prove that snapshot reads are always consistent with completed transitions and never see torn state."""
        import threading
        from app import adapter

        res = self.client.post("/api/episode/start", json={"seed": 77, "budget": 10.0})
        sid = res.json()["session_id"]

        stop_flag = threading.Event()
        read_errors = []

        def reader():
            while not stop_flag.is_set():
                snap = adapter.get_episode_state(sid)
                if len(snap["history"]) != snap["steps_taken"]:
                    read_errors.append("History length does not match steps_taken")
                if not (abs((snap["cumulative_cost"] + snap["remaining_budget"]) - snap["initial_budget"]) < 1e-6):
                    read_errors.append("Cost and budget mismatch")
                if snap["is_terminal"] and snap["hidden_truth"] is None:
                    read_errors.append("Terminal episode missing hidden truth")
                if not snap["is_terminal"] and snap["hidden_truth"] is not None:
                    read_errors.append("Active episode leaked hidden truth")

        t_reader = threading.Thread(target=reader)
        t_reader.start()

        for _ in range(6):
            self.client.post("/api/episode/step", json={"session_id": sid})

        stop_flag.set()
        t_reader.join(timeout=3.0)

        self.assertEqual(read_errors, [])

    def test_26_fifo_registry_capacity_eviction(self) -> None:
        """Verify FIFO eviction when registry capacity exceeds max_sessions."""
        from app import adapter
        with adapter.registry_lock:
            saved_sessions = dict(adapter.active_sessions)
            adapter.active_sessions.clear()
            orig_max = adapter.max_sessions
            adapter.max_sessions = 5
        try:
            sids = []
            for i in range(5):
                r = self.client.post("/api/episode/start", json={"seed": i, "budget": 10.0})
                sids.append(r.json()["session_id"])

            for sid in sids:
                res = self.client.get(f"/api/episode/state?session_id={sid}")
                self.assertEqual(res.status_code, 200)

            # Start 6th episode -> oldest (sids[0]) must be evicted
            r6 = self.client.post("/api/episode/start", json={"seed": 99, "budget": 10.0})
            sid6 = r6.json()["session_id"]

            res_old = self.client.get(f"/api/episode/state?session_id={sids[0]}")
            self.assertEqual(res_old.status_code, 404)

            # Subsequent 4 and 6th remain valid
            for sid in sids[1:] + [sid6]:
                res = self.client.get(f"/api/episode/state?session_id={sid}")
                self.assertEqual(res.status_code, 200)
        finally:
            with adapter.registry_lock:
                adapter.active_sessions.clear()
                adapter.active_sessions.update(saved_sessions)
                adapter.max_sessions = orig_max


if __name__ == "__main__":
    unittest.main()


