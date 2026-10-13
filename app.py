"""TestWise Web Application Server.

FastAPI application providing REST endpoints and serving the modern,
glossy scientific frontend interface.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
import api_adapter

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="TestWise",
    description="Interactive Scientific Workbench for Bayesian Investigations & Contingent Planning",
    version="1.0.0",
)

# Configure CORS safely for local educational development and frontend origins
_cors_origins_env = os.environ.get("TESTWISE_ALLOWED_ORIGINS", "")
if _cors_origins_env.strip():
    allowed_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global adapter instance
adapter = api_adapter.TestWiseAdapter()


# Pydantic request models
class StartEpisodeRequest(BaseModel):
    seed: int = Field(default=42, ge=0)
    budget: float = Field(default=10.0, gt=0.0)
    policy: str = Field(default="greedy_information_gain")
    confidence_threshold: float = Field(default=0.90, gt=0.0, le=1.0)
    cost_penalty_lambda: float = Field(default=0.05, ge=0.0)
    max_state_expansions: int = Field(default=10000, ge=1)


class StepEpisodeRequest(BaseModel):
    session_id: str


class InvestigateRequest(BaseModel):
    session_id: str
    test_id: str


class SolvePlanRequest(BaseModel):
    budget: float = Field(default=10.0, gt=0.0)
    confidence_threshold: float = Field(default=0.90, gt=0.0, le=1.0)
    cost_penalty_lambda: float = Field(default=0.05, ge=0.0)
    max_state_expansions: int = Field(default=10000, ge=1)


class RunEvaluationRequest(BaseModel):
    num_episodes: int = Field(default=20, ge=1, le=500)
    seed: int = Field(default=42, ge=0)
    policies: Optional[List[str]] = None
    budget: float = Field(default=10.0, gt=0.0)
    confidence_threshold: float = Field(default=0.90, gt=0.0, le=1.0)


class BayesNetQueryRequest(BaseModel):
    query_var: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class LogicTruthTableRequest(BaseModel):
    expression: str


class LogicEntailmentRequest(BaseModel):
    premises: List[str] = Field(default_factory=list)
    conclusion: str


# REST Endpoints
@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    """Retrieve model configuration, investigation specs, and application defaults."""
    try:
        return adapter.get_config_metadata()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/episode/start")
def start_episode(req: StartEpisodeRequest) -> Dict[str, Any]:
    """Start a new simulation episode with deterministic seed and isolated session state."""
    if req.policy not in agent.POLICY_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policy {req.policy!r}; must be one of {list(agent.POLICY_NAMES)}",
        )
    try:
        return adapter.start_episode(
            seed=req.seed,
            budget=req.budget,
            policy=req.policy,  # type: ignore[arg-type]
            confidence_threshold=req.confidence_threshold,
            cost_penalty_lambda=req.cost_penalty_lambda,
            max_state_expansions=req.max_state_expansions,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/episode/step")
def step_episode(req: StepEpisodeRequest) -> Dict[str, Any]:
    """Execute one autonomous decision step in the episode under its policy."""
    try:
        return adapter.step_episode(req.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/episode/investigate")
def investigate_test(req: InvestigateRequest) -> Dict[str, Any]:
    """Execute a specific investigation selected manually by the user in the workbench."""
    try:
        return adapter.investigate_test(req.session_id, req.test_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/episode/reset")
def reset_episode(req: StepEpisodeRequest) -> Dict[str, Any]:
    """Reset the current episode back to prior beliefs with full budget restored."""
    try:
        return adapter.reset_episode(req.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/episode/state")
def get_episode_state(session_id: str = Query(...)) -> Dict[str, Any]:
    """Retrieve the current state of an active or completed episode."""
    try:
        return adapter.get_episode_state(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/plan/solve")
def solve_plan(req: SolvePlanRequest) -> Dict[str, Any]:
    """Solve the complete Bellman contingent policy tree using the planner module."""
    try:
        return adapter.solve_contingent_plan(
            budget=req.budget,
            confidence_threshold=req.confidence_threshold,
            cost_penalty_lambda=req.cost_penalty_lambda,
            max_state_expansions=req.max_state_expansions,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/evaluation/run")
def run_evaluation(req: RunEvaluationRequest) -> Dict[str, Any]:
    """Run real multi-policy evaluation across synthetic episodes."""
    try:
        return adapter.run_evaluation(
            num_episodes=req.num_episodes,
            seed=req.seed,
            policies=req.policies,
            budget=req.budget,
            confidence_threshold=req.confidence_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/evaluation/export")
def export_evaluation_csv(report_id: str = Query(...)) -> Response:
    """Download the raw CSV export generated by evaluation.py."""
    try:
        csv_content = adapter.export_csv(report_id)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=testwise_evaluation_{report_id[:8]}.csv"},
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/algorithm-lab/budget-curve")
def get_budget_curve(
    seed: int = Query(default=42, ge=0),
    episodes: int = Query(default=15, ge=5, le=50),
) -> Dict[str, Any]:
    """Compute real accuracy vs budget curves across all 4 policies."""
    try:
        return adapter.run_budget_curve(seed=seed, episodes_per_point=episodes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/search/demo")
def get_search_demo() -> Dict[str, Any]:
    """Demonstrate standalone BFS, DFS, and A* search algorithms."""
    try:
        return adapter.run_search_demo()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# AI Concepts Lab: Bayesian Network Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/bayes-net/spec")
def get_bayes_net_spec() -> Dict[str, Any]:
    """Retrieve DAG structure, variables, and CPT tables for the canonical Alarm network."""
    try:
        return adapter.get_bayes_net_spec()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bayes-net/query")
def query_bayes_net(req: BayesNetQueryRequest) -> Dict[str, Any]:
    """Perform exact inference by enumeration on the Bayesian network."""
    try:
        return adapter.query_bayes_net(req.query_var, req.evidence)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bayes-net/demo")
def get_bayes_net_demo() -> Dict[str, Any]:
    """Retrieve standard textbook demonstration queries (Marginal, Diagnostic, Explaining Away)."""
    try:
        return adapter.get_bayes_net_demos()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# AI Concepts Lab: Propositional Logic Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/logic/truth-table")
def generate_logic_truth_table(req: LogicTruthTableRequest) -> Dict[str, Any]:
    """Safely parse formula and generate complete truth table with tautology and satisfiability checks."""
    try:
        return adapter.generate_logic_truth_table(req.expression)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/logic/entailment")
def check_logic_entailment(req: LogicEntailmentRequest) -> Dict[str, Any]:
    """Check semantic entailment KB |= alpha and extract counterexample models if invalid."""
    try:
        return adapter.check_logic_entailment(req.premises, req.conclusion)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logic/demos")
def get_logic_demos() -> Dict[str, Any]:
    """Retrieve Modus Ponens (valid) and Affirming the Consequent (fallacy) demonstrations."""
    try:
        return adapter.get_logic_demos()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve Single-Page App
@app.get("/")
def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend static index.html not found")
    return FileResponse(index_path)


# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
