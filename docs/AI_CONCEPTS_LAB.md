# AI Concepts Lab — Bayesian Networks & Propositional Logic

This document details the theory, mathematical specifications, algorithms, REST API contracts, and viva voce defense notes for the two syllabus extensions in **TestWise**:

1. **Bayesian Networks (`bayes_net.py`)**: Directed Acyclic Graph (DAG) representation, Conditional Probability Tables (CPTs), and exact probabilistic inference by Enumeration.
2. **Propositional Logic (`logic.py`)**: Abstract Syntax Tree (AST), recursive-descent parsing without `eval`/`exec`, exhaustive truth tables, satisfiability, tautology, and semantic entailment checking ($KB \models \alpha$) with counterexample detection.

---

## 1. Bayesian Networks (`bayes_net.py`)

### 1.1 Mathematical Formulation

A Bayesian Network is a directed acyclic graph (DAG) where nodes represent discrete random variables $X_1, X_2, \dots, X_n$ and directed edges represent direct probabilistic dependencies.

#### Joint Probability Distribution
By the chain rule of probability combined with the conditional independence assumptions encoded by the DAG structure, the full joint probability distribution factorizes into the product of local conditional distributions:

$$P(X_1 = x_1, X_2 = x_2, \dots, X_n = x_n) = \prod_{i=1}^n P(X_i = x_i \mid \text{Parents}(X_i))$$

#### Conditional Query via Enumeration
To evaluate the posterior probability distribution of a query variable $X$ given observed evidence $\mathbf{e}$:

$$P(X \mid \mathbf{e}) = \frac{P(X, \mathbf{e})}{P(\mathbf{e})} = \alpha \sum_{\mathbf{y}} P(X, \mathbf{e}, \mathbf{y})$$

where:
- $\mathbf{y}$ denotes the set of unobserved (hidden) variables in the network.
- $\alpha = \frac{1}{\sum_x P(X=x, \mathbf{e})}$ is the normalization constant.
- The summation enumerates over the Cartesian product of the finite domains of all hidden variables $\mathbf{y}$.

### 1.2 Algorithm: Exact Inference by Enumeration

`bayes_net.py` implements the classic recursive **Enumeration-Ask** algorithm (Russell & Norvig, *Artificial Intelligence: A Modern Approach*, Chapter 14):

```python
def enumeration_ask(query_var, evidence, net):
    # 1. Topological sort of variables: [X_1, X_2, ..., X_n]
    # 2. For each value v in domain(query_var):
    #        extended_e = evidence extended with {query_var: v}
    #        Q[v] = enumerate_all(topological_vars, extended_e)
    # 3. Normalize Q such that sum(Q.values()) == 1.0
```

#### Recursive Worker `enumerate_all(vars, assignment)`:
- **Base Case:** If `vars` is empty, return $1.0$.
- **Instantiated Variable:** If $Y = \text{vars}[0]$ is instantiated in `assignment`:
  $$\text{return } P(Y = \text{assignment}[Y] \mid \text{Parents}(Y)) \times \text{enumerate\_all}(\text{vars}[1:], \text{assignment})$$
- **Hidden Variable:** If $Y = \text{vars}[0]$ is unobserved:
  $$\text{return } \sum_{y \in \text{Domain}(Y)} P(Y = y \mid \text{Parents}(Y)) \times \text{enumerate\_all}(\text{vars}[1:], \text{assignment} \cup \{Y: y\})$$

### 1.3 Complexity Analysis
- **Time Complexity:** $O(d^n)$ in the worst case, where $d$ is the maximum domain cardinality (for binary variables $d=2$) and $n$ is the number of variables.
- **Space Complexity:** $O(n)$ space for the recursion call stack (equal to the number of variables in topological order).
- **Pedagogical Significance:** Enumeration guarantees exact mathematical precision without sampling variance (unlike Monte Carlo or Particle Filtering) and requires zero external libraries.

### 1.4 Canonical Demonstration: 4-Variable Alarm Network

Topology:
- $\text{Burglary} \to \text{Alarm}$
- $\text{Earthquake} \to \text{Alarm}$
- $\text{Alarm} \to \text{JohnCalls}$

#### Textbook Parameters (Russell & Norvig standard):
- $P(B = \text{True}) = 0.001$, $P(B = \text{False}) = 0.999$
- $P(E = \text{True}) = 0.002$, $P(E = \text{False}) = 0.998$
- CPT $P(A = \text{True} \mid B, E)$:
  - $P(A \mid B, E) = 0.95$
  - $P(A \mid B, \neg E) = 0.94$
  - $P(A \mid \neg B, E) = 0.29$
  - $P(A \mid \neg B, \neg E) = 0.001$
- CPT $P(J = \text{True} \mid A)$:
  - $P(J \mid A) = 0.90$
  - $P(J \mid \neg A) = 0.05$

#### Analytical Verification of Canonical Queries:
1. **Marginal Alarm Probability:**
   $$P(A = \text{True}) = \frac{1258221}{500000000} \approx 0.002516442 \quad (0.25\%)$$
2. **Direct Conditional Lookup:**
   $$P(J = \text{True} \mid A = \text{True}) = 0.9000 \quad (90.0\%)$$
3. **Diagnostic Inference:**
   $$P(B = \text{True} \mid J = \text{True}) = \frac{8490170}{521389757} \approx 0.01628373 \quad (1.63\%)$$
   *(Prior of Burglary rises 16-fold from $0.001$ to $0.01628$ upon hearing John's phone call).*
4. **Explaining Away Phenomenon:**
   - Prior: $P(E = \text{True}) = 0.002$.
   - Knowing Alarm sounded: $P(E = \text{True} \mid A = \text{True}) \approx 0.2310$ (23.1%).
   - Learning Burglary occurred: $P(E = \text{True} \mid A = \text{True}, B = \text{True}) = \frac{95}{47001} \approx 0.00202$ (0.20%).
   *(The confirmed burglary fully "explains away" the alarm, returning the earthquake posterior back down to its base rate).*

---

## 2. Propositional Logic (`logic.py`)

### 2.1 Grammar and Precedence

The propositional language supports variables, truth constants, and standard connectives:

| Operator | Syntax | Precedence | Associativity | Mathematical Notation |
| :--- | :--- | :--- | :--- | :--- |
| **Parentheses** | `( ... )` | 1 (Highest) | N/A | $( \dots )$ |
| **Negation** | `~`, `!`, `not` | 2 | Right | $\neg P$ |
| **Conjunction** | `&`, `and` | 3 | Left | $P \land Q$ |
| **Disjunction** | `\|`, `or` | 4 | Left | $P \lor Q$ |
| **Implication** | `->`, `implies` | 5 | Right | $P \implies Q$ |
| **Biconditional** | `<->`, `iff` | 6 (Lowest) | Left | $P \iff Q$ |

### 2.2 Parser Implementation: Safe Recursive Descent

`logic.py` implements a hand-crafted recursive-descent parser.
- **Zero Code Execution:** Does **not** use Python's `eval()` or `exec()`.
- **AST Classes:** `Var`, `Constant`, `Not`, `And`, `Or`, `Implies`, `Iff`.
- **Lexical Safety:** Strict regex tokenization rejects non-logical tokens, arbitrary Python constructs, and injection strings with a `ValueError`.

### 2.3 Semantic Entailment & Model Checking

#### Definition of Semantic Entailment ($KB \models \alpha$):
A Knowledge Base $KB = \{P_1, P_2, \dots, P_m\}$ semantically entails a conclusion $\alpha$ if and only if in every model (truth assignment $m$) where every premise evaluates to True, the conclusion $\alpha$ also evaluates to True:

$$KB \models \alpha \iff M(KB) \subseteq M(\alpha)$$

where $M(KB) = \{m \mid \forall P_i \in KB, \, P_i(m) = \text{True}\}$.

#### Counterexample Detection:
If there exists any truth assignment $m^*$ such that:
$$P_1(m^*) = \text{True} \land P_2(m^*) = \text{True} \land \dots \land P_m(m^*) = \text{True} \land \alpha(m^*) = \text{False}$$
then entailment fails ($KB \not\models \alpha$), and $m^*$ is returned as a **counterexample model**.

### 2.4 Complexity and Safeguards
- **Exhaustive Truth Table:** For $k$ variables, generates $2^k$ truth assignments.
- **Combinatorial Guard:** To prevent server memory exhaustion or DoS attacks, a hard limit of `MAX_EXHAUSTIVE_VARIABLES = 10` ($2^{10} = 1024$ rows) is enforced. Formulars with $> 10$ variables are safely rejected.

### 2.5 Canonical Demonstrations

1. **Modus Ponens (Valid Inference Rule):**
   - Premises: $P \implies Q$, $P$
   - Conclusion: $Q$
   - Total Models: $4$.
   - Premise Models: $1$ (where $P = \text{True}, Q = \text{True}$).
   - Conclusion in Premise Models: True.
   - Result: **Valid Entailment ($KB \models Q$), 0 Counterexamples**.

2. **Affirming the Consequent (Fallacy / Invalid Inference):**
   - Premises: $P \implies Q$, $Q$
   - Conclusion: $P$
   - Total Models: $4$.
   - Premise Models: $2$ ($\{P: \text{True}, Q: \text{True}\}$ and $\{P: \text{False}, Q: \text{True}\}$).
   - Result: **Invalid ($KB \not\models P$). Counterexample: $\{P: \text{False}, Q: \text{True}\}$**.

---

## 3. REST API Contracts

### Bayesian Network Endpoints

#### `GET /api/bayes-net/spec`
Returns network metadata, nodes, values, parent lists, and complete conditional probability tables.

#### `POST /api/bayes-net/query`
Executes exact inference by enumeration.
- **Request:**
  ```json
  {
    "query_var": "Burglary",
    "evidence": { "JohnCalls": true }
  }
  ```
- **Response:**
  ```json
  {
    "query_var": "Burglary",
    "evidence": { "JohnCalls": true },
    "distribution": { "True": 0.0162837, "False": 0.983716 },
    "p_true": 0.0162837,
    "p_false": 0.983716,
    "explanation": "Under evidence [JohnCalls=True], exact enumeration computes P(Burglary=True) = 0.016284 (1.63%) and P(Burglary=False) = 0.983716 (98.37%)."
  }
  ```

#### `GET /api/bayes-net/demo`
Executes all four canonical textbook queries and returns structured comparative results.

---

### Propositional Logic Endpoints

#### `POST /api/logic/truth-table`
Generates complete truth table and computes semantic properties.
- **Request:**
  ```json
  { "expression": "(P -> Q) <-> (~P | Q)" }
  ```
- **Response:**
  ```json
  {
    "expression": "(P -> Q) <-> (~P | Q)",
    "parsed": "((P -> Q) <-> (~P | Q))",
    "prefix_ast": "(iff (implies (var P) (var Q)) (or (not (var P)) (var Q)))",
    "variables": ["P", "Q"],
    "total_rows": 4,
    "is_tautology": true,
    "is_satisfiable": true,
    "satisfying_model": { "P": false, "Q": false },
    "rows": [ ... ]
  }
  ```

#### `POST /api/logic/entailment`
Evaluates $KB \models \alpha$ and extracts counterexamples.
- **Request:**
  ```json
  {
    "premises": ["P -> Q", "Q"],
    "conclusion": "P"
  }
  ```
- **Response:**
  ```json
  {
    "is_entailed": false,
    "premises": ["(P -> Q)", "Q"],
    "conclusion": "P",
    "variables": ["P", "Q"],
    "total_assignments": 4,
    "satisfying_premise_models": 2,
    "counterexamples": [{ "P": false, "Q": true }],
    "explanation": "Entailment FAILS (KB |/= alpha). Found 1 counterexample assignment(s) where all premises are True but conclusion 'P' is False: [{'P': False, 'Q': True}].",
    "truth_table": { ... }
  }
  ```

#### `GET /api/logic/demos`
Returns canonical demonstrations for Modus Ponens and Affirming the Consequent.

---

## 4. Viva Voce Defense Guide

### Topic A: Bayesian Networks

**Q1: Why is exact inference in Bayesian Networks generally NP-hard?**
> *Answer:* Computing marginal or conditional probabilities in an arbitrary Bayesian network is equivalent to solving the 3-SAT problem (counting satisfying assignments for a boolean formula). In poly-tree networks (singly connected graphs without undirected loops), inference can be accomplished in linear time using belief propagation. However, in multiply connected networks, exact inference requires summing over all combinations of hidden variables, which is $O(d^n)$ in the worst case.

**Q2: What is "Explaining Away" (Berkson's Paradox) in Bayesian Networks?**
> *Answer:* Explaining away is a phenomenon that occurs when two independent causes share a common effect (a V-structure, $C_1 \to E \leftarrow C_2$). While $C_1$ and $C_2$ are marginally independent ($P(C_1, C_2) = P(C_1)P(C_2)$), conditioning on their common effect $E$ (or its descendants) makes them conditionally dependent. Observing that $C_1$ occurred provides an explanation for $E$, which diminishes or "explains away" the probability that $C_2$ was responsible.

**Q3: How does the Bayesian Network in `bayes_net.py` differ from the diagnostic engine in `model.py`?**
> *Answer:* `model.py` uses a single categorical latent hypothesis variable ($H \in \{H_1, H_2, H_3, H_4\}$) with conditional test observation likelihoods $P(O \mid H)$. This is a naive bipartite structure. In contrast, `bayes_net.py` supports general Directed Acyclic Graphs with arbitrary discrete variables, multiple parents, intermediate sensor nodes (e.g. Alarm between Burglary/Earthquake and JohnCalls), and variable conditioning.

---

### Topic B: Propositional Logic

**Q1: What is the formal difference between logical equivalence ($\equiv$), material implication ($\implies$), and semantic entailment ($\models$)?**
> *Answer:*
> - **Material implication ($\implies$):** A logical connective inside the object language that forms a sentence ($P \implies Q$), which is False only when $P$ is True and $Q$ is False.
> - **Logical equivalence ($\equiv$):** A metalanguage relation stating that two sentences have identical truth values in every model ($\alpha \equiv \beta$ iff $\alpha \models \beta$ and $\beta \models \alpha$).
> - **Semantic entailment ($\models$):** A semantic relation between a set of sentences (the KB) and a conclusion ($\alpha$), stating that $\alpha$ is True in every model where all KB sentences are True.

**Q2: Why is Affirming the Consequent a fallacy? Give the exact truth-table counterexample.**
> *Answer:* Affirming the Consequent assumes that because $P \implies Q$ is true and $Q$ is observed, $P$ must be true. However, material implication $P \implies Q$ is vacuously true whenever $P$ is False (regardless of $Q$). Thus, in the assignment $\{P: \text{False}, Q: \text{True}\}$, the premise $P \implies Q$ evaluates to True (False $\implies$ True is True), and the premise $Q$ is True, yet the conclusion $P$ is False. This single counterexample invalidates the inference.

**Q3: Why did we implement a recursive-descent parser instead of using Python's `eval()`?**
> *Answer:* Using `eval()` exposes the system to Arbitrary Code Execution (Remote Code Execution vulnerability), allowing an attacker or student to execute system commands, access the filesystem, or crash the process. A recursive-descent parser creates a strictly typed Abstract Syntax Tree (AST), strictly enforces formal grammar and operator precedence, and safely rejects any non-logical token or syntax aberration.
