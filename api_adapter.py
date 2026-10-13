"""TestWise API Adapter Layer.

Thin, explicit, tested adapter bridging the Python core modules (config, model,
planner, agent, simulation, evaluation) with the web API without modifying any
underlying Bayesian or decision algorithms.
"""
from __future__ import annotations

import io
import math
import threading
import uuid
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, Union

import agent
import bayes_net
import config
import evaluation
import logic
import model
import planner
import simulation


def serialize_value(v: Any) -> Any:
    """Recursively convert custom dataclasses, Fractions, and mappings into JSON-serializable types."""
    if isinstance(v, Fraction):
        return float(v)
    if isinstance(v, (dict, MappingProxyType)):
        return {str(k): serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [serialize_value(x) for x in v]
    if hasattr(v, "__dataclass_fields__"):
        return {k: serialize_value(getattr(v, k)) for k in v.__dataclass_fields__}
    if hasattr(v, "value"):  # Enum support
        return v.value
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return v


def get_hyp_label(model: config.ModelConfig, hyp_id: Optional[str]) -> str:
    """Safely resolve human-readable label for a hypothesis ID."""
    if not hyp_id:
        return ""
    for h in model.hypotheses:
        if h.id == hyp_id:
            return h.label
    return str(hyp_id)


def get_inv_label(model: config.ModelConfig, test_id: Optional[str]) -> str:
    """Safely resolve human-readable label for an investigation ID."""
    if not test_id:
        return ""
    for inv in model.investigations:
        if inv.id == test_id:
            return inv.label
    return str(test_id)


class EpisodeSession:
    """Encapsulates a live or completed episode with state isolation."""

    def __init__(
        self,
        session_id: str,
        episode: simulation.Episode,
        policy: agent.PolicyName,
        settings: agent.AgentSettings,
        model: config.ModelConfig,
    ) -> None:
        self.session_id = session_id
        self.episode = episode
        self.policy = policy
        self.settings = settings
        self.model = model
        self.initial_budget = float(episode.initial_budget)
        self.seed = int(episode.seed)
        self.lock = threading.Lock()

    def to_dict(self) -> Dict[str, Any]:
        """Convert current episode state to frontend-safe dictionary.

        CRITICAL: Never leaks hidden truth while episode is active.
        """
        ep = self.episode
        m = self.model
        current_state = ep.state

        # Calculate Shannon entropy and max posterior for the current belief
        ent = model.entropy(current_state.beliefs)
        probs = current_state.beliefs.probabilities
        max_p = max(probs.values()) if probs else 0.0
        best_hyp = (
            max(probs.items(), key=lambda kv: kv[1])[0]
            if probs
            else None
        )

        # Get candidate actions for current state
        candidates = agent.get_candidate_actions(current_state, m)

        # Format history records
        history_list = []
        cumulative_cost = 0.0
        for rec in ep.history:
            cumulative_cost += rec.cost
            rec_dict = {
                "step_index": rec.step_index,
                "test_id": rec.test_id,
                "test_label": m.investigation(rec.test_id).label if rec.test_id in {i.id for i in m.investigations} else rec.test_id,
                "outcome": rec.outcome,
                "cost": rec.cost,
                "prior_beliefs": dict(rec.prior_beliefs.probabilities),
                "posterior_beliefs": dict(rec.posterior_beliefs.probabilities),
                "remaining_budget": rec.remaining_budget,
                "decision_kind": rec.decision.kind,
                "decision_summary": rec.decision.explanation.summary if rec.decision and rec.decision.explanation else "",
            }
            history_list.append(rec_dict)

        # Build response payload
        payload = {
            "session_id": self.session_id,
            "seed": self.seed,
            "policy": self.policy,
            "initial_budget": self.initial_budget,
            "remaining_budget": current_state.remaining_budget,
            "cumulative_cost": self.initial_budget - current_state.remaining_budget,
            "is_terminal": ep.is_terminal,
            "termination_reason": ep.termination_reason,
            "stop_reason": ep.stop_reason,
            "current_beliefs": dict(probs),
            "current_entropy_bits": ent,
            "current_max_posterior": max_p,
            "predicted_hypothesis": best_hyp,
            "predicted_label": get_hyp_label(m, best_hyp),
            "performed_investigations": list(current_state.performed),
            "steps_taken": len(ep.history),
            "max_steps": ep.max_steps,
            "confidence_threshold": self.settings.planner.confidence_threshold,
            "candidates": [
                {
                    "test_id": c.test_id,
                    "label": m.investigation(c.test_id).label,
                    "cost": c.cost,
                    "expected_information_gain_bits": c.expected_information_gain_bits,
                    "information_gain_per_cost": c.information_gain_per_cost,
                    "outcome_probabilities": [
                        {"outcome": out, "probability": prob}
                        for out, prob in c.outcome_probabilities
                    ],
                    "is_affordable": planner.is_affordable(c.cost, current_state.remaining_budget),
                }
                for c in candidates
            ],
            "history": history_list,
        }

        # Hidden ground truth is STRICTLY revealed ONLY after the episode has reached terminal state
        if ep.is_terminal:
            hidden = ep._hidden_hypothesis
            payload["hidden_truth"] = {
                "hypothesis_id": hidden,
                "label": get_hyp_label(m, hidden),
                "is_correct": (best_hyp == hidden),
            }
        else:
            payload["hidden_truth"] = None

        return payload


class TestWiseAdapter:
    """Central adapter providing clean, validated operations over TestWise core."""

    def __init__(self, app_config: Optional[config.AppConfig] = None) -> None:
        self.app_config = app_config or config.build_default_app_config()
        self.model = self.app_config.model
        self.active_sessions: Dict[str, EpisodeSession] = {}
        self.cached_evaluations: Dict[str, evaluation.EvaluationReport] = {}
        self.max_sessions: int = 100

    def get_config_metadata(self) -> Dict[str, Any]:
        """Return static model specifications, hypotheses, investigations, and parameter bounds."""
        cfg = self.app_config
        m = self.model
        cheapest = min((i.cost for i in m.investigations), default=0.0)
        most_expensive = max((i.cost for i in m.investigations), default=0.0)

        return {
            "app_title": cfg.app_title,
            "schema_version": cfg.schema_version,
            "random_seed": cfg.random_seed,
            "hypotheses": [
                {"id": h.id, "label": h.label, "prior": m.priors[h.id]}
                for h in m.hypotheses
            ],
            "investigations": [
                {
                    "id": inv.id,
                    "label": inv.label,
                    "cost": inv.cost,
                    "outcomes": list(inv.outcomes),
                    "likelihoods": {
                        h: {out: inv.likelihoods[h][out] for out in inv.outcomes}
                        for h in [h.id for h in m.hypotheses]
                    },
                }
                for inv in m.investigations
            ],
            "planner_defaults": {
                "initial_budget": cfg.planner.initial_budget,
                "confidence_threshold": cfg.planner.confidence_threshold,
                "cost_penalty_lambda": cfg.planner.cost_penalty_lambda,
                "max_state_expansions": cfg.planner.max_state_expansions,
                "min_cost": cheapest,
                "max_cost": most_expensive,
            },
            "evaluation_defaults": {
                "num_episodes": cfg.evaluation.num_episodes,
            },
            "available_policies": list(agent.POLICY_NAMES),
        }

    def start_episode(
        self,
        seed: int = 42,
        budget: float = 10.0,
        policy: agent.PolicyName = "greedy_information_gain",
        confidence_threshold: float = 0.90,
        cost_penalty_lambda: float = 0.05,
        max_state_expansions: int = 10000,
    ) -> Dict[str, Any]:
        """Start a new episode with deterministic seed and isolated session state."""
        if policy not in agent.POLICY_NAMES:
            raise ValueError(f"Unknown policy {policy!r}; must be one of {agent.POLICY_NAMES}")

        p_settings = config.PlannerSettings(
            initial_budget=float(budget),
            confidence_threshold=float(confidence_threshold),
            cost_penalty_lambda=float(cost_penalty_lambda),
            max_state_expansions=int(max_state_expansions),
        )
        a_settings = agent.AgentSettings(planner=p_settings)

        ep = simulation.create_episode(self.model, seed=int(seed), budget=float(budget))
        session_id = str(uuid.uuid4())
        session = EpisodeSession(session_id, ep, policy, a_settings, self.model)
        
        # Evict oldest session if capacity reached
        if len(self.active_sessions) >= self.max_sessions:
            oldest_id = next(iter(self.active_sessions))
            del self.active_sessions[oldest_id]

        self.active_sessions[session_id] = session

        return session.to_dict()

    def step_episode(self, session_id: str) -> Dict[str, Any]:
        """Execute one autonomous step using the episode's configured policy."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise KeyError(f"Episode session {session_id!r} not found")

        with session.lock:
            if session.episode.is_terminal:
                return session.to_dict()

            simulation.step_episode(
                session.episode,
                self.model,
                session.policy,
                session.settings,
            )
            return session.to_dict()

    def investigate_test(self, session_id: str, test_id: str) -> Dict[str, Any]:
        """Manually execute a specific investigation chosen by the user in the workbench."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise KeyError(f"Episode session {session_id!r} not found")

        ep = session.episode
        if ep.is_terminal:
            raise ValueError("Cannot investigate: episode is already terminal")

        m = self.model
        if test_id not in {i.id for i in m.investigations}:
            raise ValueError(f"Unknown investigation ID {test_id!r}")

        with session.lock:
            ep = session.episode
            if ep.is_terminal:
                raise ValueError("Cannot investigate: episode is already terminal")

            inv = m.investigation(test_id)
            if test_id in ep.state.performed:
                raise ValueError(f"Investigation {test_id!r} was already performed in this episode")

            if not planner.is_affordable(inv.cost, ep.state.remaining_budget):
                raise ValueError(
                    f"Investigation cost ({inv.cost}) exceeds remaining budget ({ep.state.remaining_budget})"
                )

            # Sample outcome using simulator logic
            outcome = simulation.sample_potential_outcome(m, ep.seed, ep._hidden_hypothesis, test_id)
            new_state = planner.advance_state(ep.state, test_id, outcome, m)

            stop_decision = agent.should_stop(new_state, session.settings, m)

            manual_decision = agent.Decision(
                kind="investigate",
                test_id=test_id,
                policy="manual_override",
                explanation=agent.Explanation(
                    policy="manual_override",
                    summary=f"Manually conducted test {inv.label} (cost ${inv.cost:.2f})",
                    entropy_bits=model.entropy(ep.state.beliefs),
                    max_posterior=max(ep.state.beliefs.probabilities.values()),
                    remaining_budget=ep.state.remaining_budget,
                    candidates=(),
                    selected=None,
                ),
            )

            record = simulation.InvestigationRecord(
                step_index=len(ep.history),
                test_id=test_id,
                outcome=outcome,
                cost=float(inv.cost),
                prior_beliefs=ep.state.beliefs,
                posterior_beliefs=new_state.beliefs,
                remaining_budget=new_state.remaining_budget,
                decision=manual_decision,
            )
            ep.history.append(record)
            ep.state = new_state

            if stop_decision.stop:
                ep.is_terminal = True
                ep.stop_reason = stop_decision.reason
                ep.termination_reason = "threshold_met" if stop_decision.reason == "confidence_threshold_met" else "no_legal_actions"
            elif len(ep.history) >= ep.max_steps:
                ep.is_terminal = True
                ep.stop_reason = "step_limit"
                ep.termination_reason = "step_limit"

            return session.to_dict()

    def reset_episode(self, session_id: str) -> Dict[str, Any]:
        """Reset the episode to its initial state using identical parameters."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise KeyError(f"Episode session {session_id!r} not found")

        with session.lock:
            return self.start_episode(
                seed=session.seed,
                budget=session.initial_budget,
                policy=session.policy,
                confidence_threshold=session.settings.planner.confidence_threshold,
                cost_penalty_lambda=session.settings.planner.cost_penalty_lambda,
                max_state_expansions=session.settings.planner.max_state_expansions,
            )

    def get_episode_state(self, session_id: str) -> Dict[str, Any]:
        """Retrieve state for an existing episode."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise KeyError(f"Episode session {session_id!r} not found")
        return session.to_dict()

    def solve_contingent_plan(
        self,
        budget: float = 10.0,
        confidence_threshold: float = 0.90,
        cost_penalty_lambda: float = 0.05,
        max_state_expansions: int = 10000,
        custom_beliefs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run genuine Bellman contingent planning to generate the complete policy tree."""
        m = self.model
        if custom_beliefs:
            beliefs = model.BeliefState(custom_beliefs)
            model.validate_model(m)  # Ensure consistency
        else:
            beliefs = model.initial_beliefs(m)

        p_settings = config.PlannerSettings(
            initial_budget=float(budget),
            confidence_threshold=float(confidence_threshold),
            cost_penalty_lambda=float(cost_penalty_lambda),
            max_state_expansions=int(max_state_expansions),
        )

        state = planner.make_initial_state(beliefs, p_settings)
        planning_result = planner.solve_contingent_plan(state, m, p_settings)

        # Build clean serialized representation of the tree
        def format_node(node: Optional[planner.PolicyNode]) -> Optional[Dict[str, Any]]:
            if node is None:
                return None

            branches = []
            for b in node.branches:
                branches.append(
                    {
                        "outcome": b.outcome,
                        "probability": float(b.probability),
                        "reachable": b.reachable,
                        "posterior": dict(b.posterior.probabilities) if b.posterior else None,
                        "child": format_node(b.child),
                    }
                )

            return {
                "performed": list(node.performed),
                "remaining_budget": node.remaining_budget,
                "terminal_decision": node.terminal_decision,
                "terminal_label": get_hyp_label(m, node.terminal_decision),
                "terminal_value": float(node.terminal_value),
                "action": node.action,
                "action_label": get_inv_label(m, node.action),
                "cost": m.investigation(node.action).cost if node.action in {i.id for i in m.investigations} else 0.0,
                "value": float(node.value),
                "action_values": [
                    {"action": a, "value": float(val)} for a, val in node.action_values
                ],
                "branches": branches,
            }

        return {
            "status": planning_result.status.value,
            "root_action": planning_result.root_action,
            "stops_at_root": planning_result.stops_at_root,
            "expected_utility": float(planning_result.expected_utility) if planning_result.expected_utility is not None else None,
            "stop_utility": float(planning_result.stop_utility),
            "expanded_states": planning_result.expanded_states,
            "max_state_expansions": planning_result.max_state_expansions,
            "reason": planning_result.reason,
            "policy_tree": format_node(planning_result.policy_tree),
        }

    def run_evaluation(
        self,
        num_episodes: int = 20,
        seed: int = 42,
        policies: Optional[List[str]] = None,
        budget: float = 10.0,
        confidence_threshold: float = 0.90,
    ) -> Dict[str, Any]:
        """Execute real multi-policy evaluation across synthetic episodes."""
        p_list = tuple(policies) if policies else tuple(agent.POLICY_NAMES)
        for p in p_list:
            if p not in agent.POLICY_NAMES:
                raise ValueError(f"Unknown policy {p!r}")

        eval_cfg = config.AppConfig(
            model=self.model,
            app_title=self.app_config.app_title,
            schema_version=self.app_config.schema_version,
            planner=config.PlannerSettings(
                initial_budget=float(budget),
                confidence_threshold=float(confidence_threshold),
                cost_penalty_lambda=self.app_config.planner.cost_penalty_lambda,
                max_state_expansions=self.app_config.planner.max_state_expansions,
            ),
            evaluation=config.EvaluationSettings(num_episodes=int(num_episodes)),
            random_seed=int(seed),
        )

        report = evaluation.evaluate_policies(eval_cfg, policies=p_list)
        report_id = str(uuid.uuid4())
        self.cached_evaluations[report_id] = report

        # Serialize summaries and individual episode results
        serialized_summaries = {}
        for pol_name, summary in report.summaries.items():
            serialized_summaries[pol_name] = {
                "policy": summary.policy,
                "attempted_episodes": summary.attempted_episodes,
                "completed_episodes": summary.completed_episodes,
                "incomplete_episodes": summary.incomplete_episodes,
                "error_episodes": summary.error_episodes,
                "resolution_rate": summary.resolution_rate,
                "overall_accuracy": summary.overall_accuracy,
                "resolved_accuracy": summary.resolved_accuracy,
                "mean_cost": summary.mean_cost,
                "mean_investigations": summary.mean_investigations,
                "mean_final_entropy_bits": summary.mean_final_entropy_bits,
                "budget_violations": summary.budget_violations,
                "mean_planner_expansions": summary.mean_planner_expansions,
            }

        episodes_list = []
        for ep_res in report.all_results:
            episodes_list.append(
                {
                    "seed": ep_res.seed,
                    "policy": ep_res.policy,
                    "hidden_hypothesis": ep_res.hidden_hypothesis,
                    "hidden_label": get_hyp_label(self.model, ep_res.hidden_hypothesis),
                    "predicted_hypothesis": ep_res.predicted_hypothesis,
                    "predicted_label": get_hyp_label(self.model, ep_res.predicted_hypothesis),
                    "total_cost": ep_res.total_cost,
                    "remaining_budget": ep_res.remaining_budget,
                    "final_confidence": ep_res.final_confidence,
                    "final_entropy": ep_res.final_entropy,
                    "is_resolved": ep_res.is_resolved,
                    "is_completed": ep_res.is_completed,
                    "is_correct": ep_res.is_correct,
                    "termination_reason": ep_res.termination_reason,
                    "investigations_count": len(ep_res.performed_investigations),
                    "performed_investigations": list(ep_res.performed_investigations),
                }
            )

        return {
            "report_id": report_id,
            "seed": report.seed,
            "num_episodes": report.num_episodes,
            "policies": list(report.policies),
            "total_runs": report.total_runs,
            "summaries": serialized_summaries,
            "episodes": episodes_list,
        }

    def export_csv(self, report_id: str) -> str:
        """Export cached evaluation report as raw CSV text."""
        import os
        import tempfile
        report = self.cached_evaluations.get(report_id)
        if not report:
            raise KeyError(f"Evaluation report {report_id!r} not found")

        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False, encoding="utf-8") as tmp:
            tmp_path = tmp.name

        try:
            evaluation.export_results_csv(report.all_results, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def run_budget_curve(
        self,
        policies: Optional[List[str]] = None,
        budgets: Optional[List[float]] = None,
        seed: int = 42,
        episodes_per_point: int = 15,
    ) -> Dict[str, Any]:
        """Compute real multi-budget evaluation curves for the Algorithm Lab."""
        p_list = tuple(policies) if policies else tuple(agent.POLICY_NAMES)
        b_list = budgets or [3.0, 5.0, 8.0, 12.0, 18.0]

        curve_data: Dict[str, List[Dict[str, Any]]] = {p: [] for p in p_list}

        for b in b_list:
            eval_cfg = config.AppConfig(
                model=self.model,
                app_title=self.app_config.app_title,
                schema_version=self.app_config.schema_version,
                planner=config.PlannerSettings(
                    initial_budget=float(b),
                    confidence_threshold=0.90,
                    cost_penalty_lambda=0.05,
                    max_state_expansions=5000,
                ),
                evaluation=config.EvaluationSettings(num_episodes=int(episodes_per_point)),
                random_seed=int(seed),
            )
            rep = evaluation.evaluate_policies(eval_cfg, policies=p_list)
            for p in p_list:
                s = rep.summaries[p]
                curve_data[p].append(
                    {
                        "budget": float(b),
                        "overall_accuracy": s.overall_accuracy or 0.0,
                        "resolved_accuracy": s.resolved_accuracy or 0.0,
                        "mean_cost": s.mean_cost or 0.0,
                        "mean_tests": s.mean_investigations or 0.0,
                    }
                )

        return {
            "budgets": b_list,
            "policies": list(p_list),
            "curve_data": curve_data,
        }

    def run_search_demo(self) -> Dict[str, Any]:
        """Demonstrate standalone BFS, DFS, and A* search algorithms on a standard graph problem."""
        graph = {
            "A": [("to_B", "B", 1.0), ("to_C", "C", 4.0)],
            "B": [("to_D", "D", 2.0), ("to_E", "E", 5.0)],
            "C": [("to_E", "E", 1.0)],
            "D": [("to_G", "G", 6.0)],
            "E": [("to_G", "G", 2.0)],
            "G": [],
        }
        h_admissible = {"A": 4.0, "B": 4.0, "C": 3.0, "D": 6.0, "E": 2.0, "G": 0.0}
        problem = planner.SearchProblem(
            initial_state="A",
            successors=lambda s: graph.get(s, []),
            is_goal=lambda s: s == "G",
        )

        bfs_res = planner.breadth_first_search(problem)
        dfs_res = planner.depth_first_search(problem)
        astar_res = planner.a_star_search(problem, lambda s: h_admissible[s])

        return {
            "problem": {
                "initial_state": "A",
                "goal_state": "G",
                "graph_edges": [
                    {"from": u, "to": v, "action": a, "cost": c}
                    for u, edges in graph.items()
                    for a, v, c in edges
                ],
                "admissible_heuristic": h_admissible,
            },
            "algorithms": {
                "breadth_first_search": {
                    "status": bfs_res.status.value,
                    "path": list(bfs_res.path),
                    "actions": list(bfs_res.actions),
                    "path_cost": bfs_res.path_cost,
                    "expanded_states": bfs_res.expanded_states,
                    "expansion_order": list(bfs_res.expansion_order),
                    "guarantee": "Minimum number of edge steps (unweighted shortest path)",
                },
                "depth_first_search": {
                    "status": dfs_res.status.value,
                    "path": list(dfs_res.path),
                    "actions": list(dfs_res.actions),
                    "path_cost": dfs_res.path_cost,
                    "expanded_states": dfs_res.expanded_states,
                    "expansion_order": list(dfs_res.expansion_order),
                    "guarantee": "Deep-first exploration (not guaranteed shortest or cheapest)",
                },
                "a_star_search": {
                    "status": astar_res.status.value,
                    "path": list(astar_res.path),
                    "actions": list(astar_res.actions),
                    "path_cost": astar_res.path_cost,
                    "expanded_states": astar_res.expanded_states,
                    "expansion_order": list(astar_res.expansion_order),
                    "reopened_states": astar_res.reopened_states,
                    "guarantee": "Optimal minimum-cost path under admissible heuristic",
                },
            },
        }

    def get_bayes_net_spec(self) -> Dict[str, Any]:
        """Return variables, DAG structure, and CPT tables for the canonical Alarm network."""
        net = bayes_net.build_canonical_alarm_net()
        nodes_data = []
        for name in net.topological_sort():
            node = net.nodes[name]
            cpt_rows = []
            for p_tuple, dist in node.cpt.items():
                p_dict = dict(zip(node.parents, p_tuple)) if node.parents else {}
                cpt_rows.append({
                    "parents": p_dict,
                    "probabilities": {str(val): prob for val, prob in dist.items()},
                })
            nodes_data.append({
                "name": node.name,
                "values": [str(v) for v in node.values],
                "parents": list(node.parents),
                "cpt": cpt_rows,
            })
        return {
            "name": "Burglary-Earthquake-Alarm-JohnCalls",
            "nodes": nodes_data,
            "edges": [
                {"from": "Burglary", "to": "Alarm"},
                {"from": "Earthquake", "to": "Alarm"},
                {"from": "Alarm", "to": "JohnCalls"},
            ],
        }

    def query_bayes_net(self, query_var: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Perform exact probabilistic inference on the canonical Alarm network."""
        net = bayes_net.build_canonical_alarm_net()
        clean_evidence: Dict[str, Any] = {}
        for k, v in evidence.items():
            if isinstance(v, str):
                if v.lower() in ("true", "t", "1"):
                    clean_evidence[k] = True
                elif v.lower() in ("false", "f", "0"):
                    clean_evidence[k] = False
                else:
                    clean_evidence[k] = v
            else:
                clean_evidence[k] = v

        dist = net.enumeration_ask(query_var, clean_evidence)
        p_true = float(dist.get(True, 0.0))
        p_false = float(dist.get(False, 0.0))

        ev_str = ", ".join(f"{k}={v}" for k, v in clean_evidence.items()) if clean_evidence else "None (Prior Marginal)"
        explanation = (
            f"Under evidence [{ev_str}], exact enumeration computes "
            f"P({query_var}=True) = {p_true:.6f} ({p_true * 100:.2f}%) and "
            f"P({query_var}=False) = {p_false:.6f} ({p_false * 100:.2f}%)."
        )

        return {
            "query_var": query_var,
            "evidence": clean_evidence,
            "distribution": {str(k): float(v) for k, v in dist.items()},
            "p_true": p_true,
            "p_false": p_false,
            "explanation": explanation,
        }

    def get_bayes_net_demos(self) -> Dict[str, Any]:
        """Return standard textbook demonstration queries."""
        return bayes_net.run_alarm_demonstration()

    def generate_logic_truth_table(self, expression: str) -> Dict[str, Any]:
        """Safely parse expression and generate complete truth table with satisfiability & tautology checks."""
        expr = logic.parse_expr(expression)
        table = logic.generate_truth_table(expr)
        is_taut = logic.is_tautology(expr)
        is_sat, sat_model = logic.is_satisfiable(expr)

        return {
            "expression": expression,
            "parsed": str(expr),
            "prefix_ast": expr.to_prefix(),
            "variables": table.variables,
            "column_headers": table.column_headers,
            "rows": table.rows,
            "total_rows": table.total_rows,
            "is_tautology": is_taut,
            "is_satisfiable": is_sat,
            "satisfying_model": sat_model,
        }

    def check_logic_entailment(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """Check semantic entailment KB |= alpha with counterexample extraction."""
        result = logic.check_entailment(premises=premises, conclusion=conclusion)
        return {
            "is_entailed": result.is_entailed,
            "premises": result.premises,
            "conclusion": result.conclusion,
            "variables": result.variables,
            "total_assignments": result.total_assignments,
            "satisfying_premise_models": result.satisfying_premise_models,
            "counterexamples": result.counterexamples,
            "explanation": result.explanation,
            "truth_table": {
                "variables": result.truth_table.variables,
                "column_headers": result.truth_table.column_headers,
                "rows": result.truth_table.rows,
                "total_rows": result.truth_table.total_rows,
            },
        }

    def get_logic_demos(self) -> Dict[str, Any]:
        """Return Modus Ponens and Affirming the Consequent demonstration results."""
        mp = logic.run_modus_ponens_demo()
        ac = logic.run_affirming_consequent_demo()
        return {
            "modus_ponens": {
                "name": "Modus Ponens (Valid Rule of Inference)",
                "premises": mp.premises,
                "conclusion": mp.conclusion,
                "is_entailed": mp.is_entailed,
                "counterexamples": mp.counterexamples,
                "explanation": mp.explanation,
            },
            "affirming_the_consequent": {
                "name": "Affirming the Consequent (Fallacy / Invalid)",
                "premises": ac.premises,
                "conclusion": ac.conclusion,
                "is_entailed": ac.is_entailed,
                "counterexamples": ac.counterexamples,
                "explanation": ac.explanation,
            },
        }

