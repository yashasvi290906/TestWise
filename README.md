# TestWise: Bayesian Investigation & Contingent Planning Workbench

TestWise is an interactive scientific workbench and decision-theoretic AI testbed designed to demonstrate how an intelligent agent reasons under uncertainty, investigates diagnostic hypotheses under a limited resource budget, updates Bayesian beliefs from stochastic evidence, and decides contingently whether to perform another investigation or terminate.

---

## 1. Project Overview & Core Problem

In real-world problem-solving (such as troubleshooting engineering failures, scientific experimentation, or diagnostic triaging), an agent rarely possesses full observability. Instead:
- The true underlying state (**hypothesis**) is unknown.
- Gathering evidence requires running **investigations** (tests, probes, experiments).
- Each investigation incurs an **abstract cost** and yields a noisy, probabilistic observation.
- The agent operates under a finite **budget constraint**.

TestWise demonstrates the formal sequential decision process: how to balance the **information gain** of tests against their **cost**, and how to plan outcome-contingent decision trees using backward induction (Bellman optimality).

---

## 2. Sequential Bayesian Investigation Workflow

At each discrete decision step $t$:
1. **Current Belief State**: The agent maintains a probability distribution over hypotheses:
   $$b_t = (P(H_1), P(H_2), \dots, P(H_K)), \quad \sum_{i=1}^K P(H_i) = 1.0$$
2. **Stopping Evaluation**: The agent evaluates whether the maximum posterior probability meets or exceeds a configured confidence threshold $\tau$ (e.g., $\max_i P(H_i) \ge 0.90$), or if the remaining budget is insufficient to run any unperformed test. If so, the episode terminates.
3. **Action Selection**: If not stopping, the agent chooses an unperformed investigation $a$ from the set of candidate actions according to its policy.
4. **Outcome Realization**: An outcome $o \in \Omega_a$ is observed according to likelihood model $P(o \mid h^*)$, where $h^*$ is the hidden hypothesis.
5. **Bayesian Belief Update**: Beliefs are updated via Bayes' rule:
   $$P(H_i \mid o) = \frac{P(o \mid H_i) P(H_i)}{\sum_j P(o \mid H_j) P(H_j)}$$
6. **Budget Deduction**: The investigation cost $c(a)$ is subtracted from the remaining budget:
   $$B_{t+1} = B_t - c(a)$$

---

## 3. Hidden Ground Truth Isolation vs. Observable Planning State

A fundamental architectural guarantee in TestWise is the **strict isolation of hidden ground truth**:
- **Hidden Ground Truth ($h^*$)**: When an episode is instantiated with a deterministic seed, a synthetic true hypothesis $h^*$ is sampled from prior base rates. This ground truth is stored privately in `Episode._hidden_hypothesis`.
- **Observable Planning State**: The agent, planner, policies, and API responses have access **only** to the observable `BeliefState`, the remaining budget, and the history of performed tests and observed outcomes.
- **Pedagogical Security**: The hidden hypothesis is never leaked into decision rationales or active episode API responses. It is strictly revealed in the workbench only after the episode reaches a terminal state, enabling students to audit whether the final classification was correct.

---

## 4. Decision Policies

TestWise provides four distinct decision policies to demonstrate the algorithmic trade-offs between heuristic myopia and contingent dynamic programming:

| Policy | Horizon | Cost Awareness | Mechanism |
| :--- | :--- | :--- | :--- |
| **Fixed Order** | Static | None | Executes tests in a predetermined, hardcoded sequence regardless of intermediate belief updates or costs. |
| **Cheapest First** | 1-step Greedy | Cost Only | Prioritizes available affordable tests with the lowest cost, ignoring their diagnostic discriminatory power. |
| **Greedy Info Gain** | 1-step Greedy | Info / Cost Ratio | Evaluates expected Shannon information gain (entropy reduction) per cost unit and greedily selects the most efficient immediate test. |
| **Contingent Optimal** | Full Lookahead | Bellman Optimal | Builds a complete contingent policy tree using recursive backward induction over belief-budget states. |

---

## 5. Bellman Contingent-Planning Approach

The contingent planner (`planner.py`) models investigation as a Markov Decision Process (MDP) over continuous belief simplex and discrete budget states:

### Stopping Value (Terminal Decision)
At state $s = (b, B)$, if the agent chooses to stop and classify:
$$V_{\text{stop}}(s) = \max_{h \in \mathcal{H}} b(h)$$

### Continuation Value (Investigation)
If the agent performs investigation $a$ with cost $c(a) \le B$:
$$Q(s, a) = - \lambda \cdot c(a) + \sum_{o \in \Omega_a} P(o \mid b) \cdot V^*(s_{a, o})$$
where $s_{a, o} = (\text{BayesUpdate}(b, a, o), B - c(a))$ and $\lambda$ is the cost penalty parameter.

### Bellman Recurrence
$$V^*(s) = \max \left( V_{\text{stop}}(s), \max_{a \in \mathcal{A}(s)} Q(s, a) \right)$$

Backward induction produces a **contingent policy tree** where subsequent investigations branch conditionally upon each possible outcome of prior tests.

---

## 6. Budget Constraints and Cost Penalty ($\lambda$)

- **Budget Units**: All budgets and investigation costs are defined in abstract **cost units** (e.g., test costs ranging from 1.0 to 5.0 units with a default initial budget of 10.0 units). TestWise does not use arbitrary currency symbols.
- **Exact Budget Arithmetic**: Budget verification uses exact floating arithmetic checks (`is_affordable(cost, budget)`) preventing illegal test execution.
- **Cost Penalty ($\lambda$)**: The parameter $\lambda$ (default $0.05$) converts cost units into decision utility penalty units. It controls how much diagnostic confidence the agent is willing to sacrifice to conserve resources. Higher $\lambda$ values favor early stopping; lower $\lambda$ values encourage exhaustive testing.

---

## 7. Information Gain (Bits) vs. Decision Utility

TestWise highlights a key theoretical distinction in AI:
- **Expected Information Gain (EIG)**: Measured in Shannon entropy bits:
  $$\text{EIG}(a) = H(b) - \mathbb{E}_{o}[H(b_{a, o})]$$
  EIG quantifies pure uncertainty reduction without considering misclassification costs or task payoffs.
- **Decision Utility ($Q$-value)**: Measures the expected terminal classification accuracy penalized by resource expenditure.
A test with high Shannon entropy reduction may not change the optimal classification decision, whereas a targeted test with lower entropy reduction might cleanly separate the two most competitive hypotheses.

---

## 8. Evaluation Metrics & Precise Denominators

The evaluation engine (`evaluation.py`) aggregates performance across reproducible batch runs using mathematically defensible denominators:

- **Attempted Episodes ($N$)**: Total episodes initiated.
- **Completed Episodes ($N_{\text{comp}}$)**: Episodes that terminated via a legitimate stopping condition (`threshold_met`, `no_legal_actions`, `policy_stop`).
- **Incomplete Episodes ($N_{\text{incomp}}$)**: Episodes terminated prematurely (e.g., search expansion cutoffs).
- **Resolution Rate**: $\frac{N_{\text{resolved}}}{N}$ (episodes meeting the confidence threshold).
- **Completed Accuracy**: $\frac{\text{Correct Completed Episodes}}{N_{\text{comp}}}$.
- **Overall Accuracy**: $\frac{\text{Correct Completed Episodes}}{N}$ (incomplete episodes penalize overall accuracy).
- **Resolved Accuracy**: $\frac{\text{Correct Resolved Episodes}}{N_{\text{resolved}}}$.
- **Mean Investigation Cost**: Total cost across attempted episodes divided by $N$.
- **Mean Investigations**: Total tests conducted divided by $N$.
- **Mean Final Entropy**: Average entropy across episodes with finite entropy.
- **Budget Violations**: Count of episodes where cost exceeded initial budget (must strictly equal 0).
- **Mean Planner Expansions**: Total search node expansions divided by **episodes with planner activity** (episodes that actually invoked the search planner).

---

## 9. AI Concepts Lab: Bayesian Networks & Propositional Logic

Beyond sequential diagnosis, TestWise includes two standalone syllabus modules (`bayes_net.py` and `logic.py`):

1. **Bayesian Networks**:
   - Directed Acyclic Graph (DAG) representation with Conditional Probability Tables (CPTs).
   - Canonical 4-variable Alarm network ($\text{Burglary}, \text{Earthquake} \to \text{Alarm} \to \text{JohnCalls}$).
   - Exact probabilistic inference by Enumeration (`enumeration_ask`) supporting marginal, diagnostic, and explaining-away queries without numerical approximation or external libraries.
2. **Propositional Logic**:
   - Safe recursive-descent parser building Abstract Syntax Trees without `eval` or `exec`.
   - Complete truth table generator with automatic tautology and satisfiability verification.
   - Semantic entailment checker ($KB \models \alpha$) that detects entailment and extracts counterexample truth assignments if the inference is invalid.

---

## 10. Classical Search Demonstrations

TestWise includes graph-search demonstrations (`api_adapter.py: run_search_demo`) covering:
- **Breadth-First Search (BFS)** (FIFO queue, shortest path by edge count).
- **Depth-First Search (DFS)** (LIFO stack, path exploration).
- **A* Search** (Priority queue with path cost $g(n)$ and admissible heuristic $h(n)$).

These demonstrations relate directly to standard AI search syllabi and showcase state-space traversal concepts used in the contingent planner.

---

## 11. Architecture & Repository Structure

```
D:\TestWise\
├── config.py             # Configuration dataclasses, default model parameters, validation
├── model.py              # Hypotheses, investigations, prior/likelihood distributions, entropy
├── planner.py            # Bellman backward induction, PolicyNode tree, budget pruning
├── agent.py              # Decision policies (fixed_order, cheapest_first, greedy, contingent)
├── simulation.py         # Deterministic episode simulation, hidden-truth isolation
├── evaluation.py         # Multi-policy batch benchmarking, metrics summaries, CSV export
├── bayes_net.py          # Bayesian Network DAG, CPT representation, exact inference
├── logic.py              # Propositional logic AST, truth table generator, entailment
├── api_adapter.py        # Central orchestrator, thread-safe session registry, serialization
├── app.py                # FastAPI REST API, CORS middleware, static file serving
├── requirements.txt      # Direct runtime and testing dependencies
├── .github/workflows/    # GitHub Actions automated continuous integration
├── docs/                 # In-depth architectural reports and syllabus specifications
│   ├── AI_CONCEPTS_LAB.md
│   └── ENGINEERING_REPAIR_REPORT.md
├── static/               # Vanilla JavaScript, CSS, and SVG scientific workbench interface
│   ├── index.html
│   ├── style.css
│   └── app.js
├── test_testwise.py      # Core engine unit tests (103 tests)
├── test_simulation.py    # Simulation lifecycle tests (30 tests)
├── test_evaluation.py    # Evaluation metric and CSV tests (22 tests)
├── test_api.py           # REST API integration & concurrency tests (26 tests)
├── test_bayes_net.py     # Bayesian Network inference tests (16 tests)
└── test_logic.py         # Propositional logic parsing & entailment tests (10 tests)
```

---

## 12. Installation, Startup, and Testing

### Prerequisites
- Python 3.10 or higher.

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yashasvi290906/TestWise.git
cd TestWise
pip install -r requirements.txt
```

### Running the Workbench Server
Start the local FastAPI application server:
```bash
py -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

### Running the Complete Test Suite
Run the full test suite with warning enforcement:
```bash
py -W error -m unittest discover -s . -p "test*.py"
```
*Current test suite: 202 tests, 0 failures, 0 errors.*

---

## 13. Example Workflows

### Workflow 1: Stepping Through an Episode
1. Navigate to the **Executive Command Center** or **Investigation Workbench**.
2. Set Seed = `42`, Budget = `10.0` units, Policy = `Greedy Information Gain`.
3. Click **Start Episode**. Observe initial priors and available candidate investigations.
4. Click **Step Policy →** to execute one test, observe the evidence, and watch the Bayesian belief update in real-time.
5. Click **Reset Episode** to verify that the episode returns in-place to initial priors with the full budget restored.

### Workflow 2: Inspecting a Contingent Plan Tree
1. Open the **Contingent-Plan Inspector**.
2. Click **Recompute Contingent Plan**.
3. Inspect the SVG decision tree showing root action and conditional outcome branches.
4. Click any decision node or use the inspector breadcrumb links to navigate deeper sub-decisions.
5. Review candidate $Q$-values and verify the provenance of the optimal action.

### Workflow 3: Running Policy Benchmarks & Exporting CSV
1. Open the **Evaluation & Benchmarking** panel.
2. Select Episode Count = `20` and click **Run Comparative Evaluation**.
3. Compare overall accuracy, resolution rate, and mean cost across all 4 policies.
4. Click **Export Results CSV** to download a structured dataset for offline analysis.

### Workflow 4: Exploring Bayesian Networks & Logic
1. Switch to the **AI Concepts Lab**.
2. Under Bayesian Networks, query $P(\text{Burglary} \mid \text{JohnCalls}=\text{True})$ to observe exact diagnostic reasoning.
3. Under Propositional Logic, test Modus Ponens ($P \to Q, P \models Q$) or test fallacies like Affirming the Consequent to see counterexample truth assignments.

---

## 14. Known Limitations

- **Synthetic Data**: Investigations, likelihood matrices, and hypotheses are synthetic benchmark constructs calibrated for educational exploration; they are not drawn from real clinical or industrial EHR datasets.
- **Search Expansion Limits**: When budgets are large ($>25$ units), full backward induction can exceed the configured state expansion limit (`max_state_expansions`). In such scenarios, the planner transparently marks the tree as `incomplete` and returns the best heuristic action rather than a proven optimal tree.
- **Local Deployment**: The session manager uses in-memory process state with thread synchronization and FIFO eviction; it is designed for single-node local educational demos, not multi-node clustered persistence.

---

## 15. Responsible Scientific Use

TestWise is strictly an **educational and scientific simulation workbench** for studying Bayesian sequential decision-making, information theory, and contingent dynamic programming. It is **not** a clinical diagnostic medical device, health advisor, or production safety system. All decisions, costs, and likelihoods are synthetic and should be interpreted solely in the context of AI curriculum and decision-theoretic benchmarks.
