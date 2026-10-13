"""Bayesian Network module for exact probabilistic inference.

Provides a clean, standalone, pedagogically transparent implementation of:
1. Directed Acyclic Graph (DAG) representation for discrete random variables.
2. Conditional Probability Tables (CPTs) with strict row normalization.
3. Graph validation (cycle detection via topological sort, completeness).
4. Exact probabilistic inference by Enumeration (Russell & Norvig, AIMA Chapter 14).
5. Canonical 4-variable Alarm network (Burglary, Earthquake, Alarm, JohnCalls).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class BayesVariable:
    """Discrete random variable in a Bayesian Network."""
    name: str
    values: Tuple[Any, ...] = (True, False)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Variable name must be a non-empty string.")
        if len(self.values) < 2:
            raise ValueError(f"Variable {self.name!r} must have at least 2 distinct values.")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Variable {self.name!r} has duplicate domain values: {self.values}")


@dataclass
class BayesNode:
    """A node in a Bayesian network with its variable, parents, and CPT."""
    variable: BayesVariable
    parents: Tuple[str, ...] = ()
    # CPT maps parent_values_tuple -> {var_value: probability}
    # For root nodes (parents=()), cpt key is ()
    cpt: Dict[Tuple[Any, ...], Dict[Any, float]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.variable.name

    @property
    def values(self) -> Tuple[Any, ...]:
        return self.variable.values

    def get_probability(self, value: Any, parent_assignment: Dict[str, Any]) -> float:
        """Look up P(Variable = value | parents)."""
        parent_tuple = tuple(parent_assignment[p] for p in self.parents)
        if parent_tuple not in self.cpt:
            raise KeyError(
                f"Parent configuration {dict(zip(self.parents, parent_tuple))} not found in CPT for node {self.name!r}"
            )
        distribution = self.cpt[parent_tuple]
        if value not in distribution:
            raise KeyError(f"Value {value!r} not in distribution for node {self.name!r}")
        return distribution[value]


class BayesNet:
    """Bayesian Network with exact inference by enumeration."""

    def __init__(self) -> None:
        self.nodes: Dict[str, BayesNode] = {}

    def add_node(self, node: BayesNode) -> None:
        """Add a BayesNode to the network."""
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name!r} already exists in the network.")
        self.nodes[node.name] = node

    def add_variable(
        self,
        name: str,
        values: Sequence[Any],
        parents: Sequence[str] = (),
        cpt: Optional[Dict[Tuple[Any, ...], Dict[Any, float]]] = None,
    ) -> BayesNode:
        """Convenience method to construct and register a node."""
        node = BayesNode(
            variable=BayesVariable(name=name, values=tuple(values)),
            parents=tuple(parents),
            cpt=cpt or {},
        )
        self.add_node(node)
        return node

    def validate(self) -> None:
        """Validate network structure, parents, absence of cycles, and CPT completeness."""
        # 1. Check all parents are known
        for name, node in self.nodes.items():
            for p in node.parents:
                if p not in self.nodes:
                    raise ValueError(f"Node {name!r} references unknown parent {p!r}")
                if p == name:
                    raise ValueError(f"Node {name!r} cannot have itself as a parent.")

        # 2. Cycle detection via topological sort
        self.topological_sort()

        # 3. CPT normalization and completeness
        for name, node in self.nodes.items():
            parent_domains = [self.nodes[p].values for p in node.parents]
            expected_parent_tuples = list(itertools.product(*parent_domains)) if parent_domains else [()]

            if len(node.cpt) != len(expected_parent_tuples):
                raise ValueError(
                    f"Node {name!r} CPT has {len(node.cpt)} rows; expected {len(expected_parent_tuples)} rows."
                )

            for parent_tuple in expected_parent_tuples:
                if parent_tuple not in node.cpt:
                    raise ValueError(
                        f"Node {name!r} missing CPT row for parent configuration: {parent_tuple}"
                    )
                row = node.cpt[parent_tuple]
                # Check completeness over domain
                for v in node.values:
                    if v not in row:
                        raise ValueError(f"Node {name!r} row {parent_tuple} missing value {v!r}")
                    p_val = row[v]
                    if not (0.0 <= p_val <= 1.0):
                        raise ValueError(
                            f"Node {name!r} invalid probability {p_val} for value {v!r} under parents {parent_tuple}"
                        )
                # Check sum equals 1.0 within floating tolerance
                total_prob = sum(row[v] for v in node.values)
                if abs(total_prob - 1.0) > 1e-5:
                    raise ValueError(
                        f"Node {name!r} CPT row for {parent_tuple} does not sum to 1.0 (sum = {total_prob:.6f})"
                    )

    def topological_sort(self) -> List[str]:
        """Compute a topological ordering of variables; raises ValueError if cycles exist."""
        in_degree: Dict[str, int] = {name: 0 for name in self.nodes}
        children: Dict[str, List[str]] = {name: [] for name in self.nodes}

        for name, node in self.nodes.items():
            for p in node.parents:
                if p in children:
                    children[p].append(name)
                    in_degree[name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for child in children[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            unvisited = [name for name, deg in in_degree.items() if deg > 0]
            raise ValueError(f"Cycle detected in Bayesian Network involving variables: {unvisited}")

        return order

    def joint_probability(self, assignment: Dict[str, Any]) -> float:
        """Compute the exact joint probability P(X1=x1, ..., Xn=xn) = prod_i P(Xi | parents(Xi))."""
        for name in self.nodes:
            if name not in assignment:
                raise ValueError(f"Complete assignment required; missing variable {name!r}")
        prob = 1.0
        for name, node in self.nodes.items():
            val = assignment[name]
            p = node.get_probability(val, assignment)
            prob *= p
        return prob

    def enumeration_ask(
        self,
        query_var: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[Any, float]:
        """Compute the conditional distribution P(query_var | evidence) via exact Enumeration.
        
        Follows Russell & Norvig, Artificial Intelligence: A Modern Approach, Chapter 14.
        """
        if query_var not in self.nodes:
            raise KeyError(f"Query variable {query_var!r} not in network.")

        evidence = dict(evidence or {})
        for ev_var, ev_val in evidence.items():
            if ev_var not in self.nodes:
                raise KeyError(f"Evidence variable {ev_var!r} not in network.")
            if ev_val not in self.nodes[ev_var].values:
                raise ValueError(
                    f"Evidence value {ev_val!r} is not in domain {self.nodes[ev_var].values} for variable {ev_var!r}"
                )

        query_node = self.nodes[query_var]

        # Trivial case: query variable is already in evidence
        if query_var in evidence:
            assigned_val = evidence[query_var]
            return {v: 1.0 if v == assigned_val else 0.0 for v in query_node.values}

        topo_order = self.topological_sort()

        # Compute unnormalized joint query-evidence probabilities
        distribution: Dict[Any, float] = {}
        for val in query_node.values:
            extended_evidence = dict(evidence)
            extended_evidence[query_var] = val
            distribution[val] = self._enumerate_all(topo_order, extended_evidence)

        # Normalize
        total = sum(distribution.values())
        if total <= 0.0 or abs(total) < 1e-15:
            raise ValueError(
                f"Contradictory or zero-probability evidence: P(evidence) = 0.0 under evidence {evidence}"
            )

        return {val: p / total for val, p in distribution.items()}

    def _enumerate_all(self, variables: Sequence[str], assignment: Dict[str, Any]) -> float:
        """Recursive worker for enumeration over variables."""
        if not variables:
            return 1.0

        y = variables[0]
        rest = variables[1:]
        node = self.nodes[y]

        if y in assignment:
            # y is instantiated
            val = assignment[y]
            p = node.get_probability(val, assignment)
            return p * self._enumerate_all(rest, assignment)
        else:
            # y is hidden/unobserved: sum over domain
            total = 0.0
            for val in node.values:
                extended = dict(assignment)
                extended[y] = val
                p = node.get_probability(val, extended)
                total += p * self._enumerate_all(rest, extended)
            return total

    def marginal(self, query_var: str) -> Dict[Any, float]:
        """Compute the prior marginal distribution P(query_var)."""
        return self.enumeration_ask(query_var, evidence={})

    def conditional(self, query_var: str, evidence: Dict[str, Any]) -> Dict[Any, float]:
        """Compute the conditional distribution P(query_var | evidence)."""
        return self.enumeration_ask(query_var, evidence=evidence)


def build_canonical_alarm_net() -> BayesNet:
    """Build the canonical 4-variable Burglary, Earthquake, Alarm, JohnCalls Bayesian network.
    
    Structure:
      Burglary -> Alarm
      Earthquake -> Alarm
      Alarm -> JohnCalls
    
    Textbook demonstration probabilities:
      P(B=T) = 0.001, P(B=F) = 0.999
      P(E=T) = 0.002, P(E=F) = 0.998
      P(A=T | B=T, E=T) = 0.95
      P(A=T | B=T, E=F) = 0.94
      P(A=T | B=F, E=T) = 0.29
      P(A=T | B=F, E=F) = 0.001
      P(J=T | A=T) = 0.90
      P(J=T | A=F) = 0.05
    """
    net = BayesNet()

    # 1. Burglary (B)
    net.add_variable(
        name="Burglary",
        values=(True, False),
        parents=(),
        cpt={
            (): {True: 0.001, False: 0.999}
        },
    )

    # 2. Earthquake (E)
    net.add_variable(
        name="Earthquake",
        values=(True, False),
        parents=(),
        cpt={
            (): {True: 0.002, False: 0.998}
        },
    )

    # 3. Alarm (A) with parents (Burglary, Earthquake)
    net.add_variable(
        name="Alarm",
        values=(True, False),
        parents=("Burglary", "Earthquake"),
        cpt={
            (True, True): {True: 0.95, False: 0.05},
            (True, False): {True: 0.94, False: 0.06},
            (False, True): {True: 0.29, False: 0.71},
            (False, False): {True: 0.001, False: 0.999},
        },
    )

    # 4. JohnCalls (J) with parent Alarm
    net.add_variable(
        name="JohnCalls",
        values=(True, False),
        parents=("Alarm",),
        cpt={
            (True,): {True: 0.90, False: 0.10},
            (False,): {True: 0.05, False: 0.95},
        },
    )

    net.validate()
    return net


def run_alarm_demonstration() -> Dict[str, Any]:
    """Execute standard demonstration queries on the canonical Alarm network."""
    net = build_canonical_alarm_net()

    # Query 1: Probability of an Alarm P(Alarm)
    p_alarm = net.marginal("Alarm")

    # Query 2: Probability of John calling given an Alarm P(JohnCalls | Alarm=True)
    p_john_given_alarm = net.conditional("JohnCalls", {"Alarm": True})

    # Query 3: Posterior probability of Burglary given that John calls P(Burglary | JohnCalls=True)
    p_burglary_given_john = net.conditional("Burglary", {"JohnCalls": True})

    # Query 4: Explaining away: P(Earthquake | Alarm=True) vs P(Earthquake | Alarm=True, Burglary=True)
    p_earthquake_alarm = net.conditional("Earthquake", {"Alarm": True})
    p_earthquake_alarm_burglary = net.conditional("Earthquake", {"Alarm": True, "Burglary": True})

    return {
        "network": "Burglary-Earthquake-Alarm-JohnCalls",
        "variables": list(net.nodes.keys()),
        "dag_edges": [
            {"from": "Burglary", "to": "Alarm"},
            {"from": "Earthquake", "to": "Alarm"},
            {"from": "Alarm", "to": "JohnCalls"},
        ],
        "queries": [
            {
                "id": "p_alarm",
                "description": "Marginal probability of Alarm sounding",
                "query": "Alarm",
                "evidence": {},
                "distribution": {str(k): v for k, v in p_alarm.items()},
                "p_true": p_alarm[True],
                "explanation": (
                    "Sum of joint probabilities across all parent combinations (B, E). "
                    f"P(Alarm=True) = {p_alarm[True]:.6f} (~0.25%)."
                ),
            },
            {
                "id": "p_john_given_alarm",
                "description": "Probability of John calling given that the Alarm sounded",
                "query": "JohnCalls",
                "evidence": {"Alarm": True},
                "distribution": {str(k): v for k, v in p_john_given_alarm.items()},
                "p_true": p_john_given_alarm[True],
                "explanation": (
                    "Direct conditional lookup from JohnCalls CPT given parent Alarm=True. "
                    f"P(JohnCalls=True | Alarm=True) = {p_john_given_alarm[True]:.4f} (90.0%)."
                ),
            },
            {
                "id": "p_burglary_given_john",
                "description": "Posterior probability of Burglary given that John calls",
                "query": "Burglary",
                "evidence": {"JohnCalls": True},
                "distribution": {str(k): v for k, v in p_burglary_given_john.items()},
                "p_true": p_burglary_given_john[True],
                "explanation": (
                    f"Diagnostic inference: prior P(B) rises from 0.001 to {p_burglary_given_john[True]:.5f} (~1.63%), "
                    "a 16-fold increase due to the reported phone call."
                ),
            },
            {
                "id": "explaining_away",
                "description": "Explaining Away: P(Earthquake | Alarm) vs P(Earthquake | Alarm, Burglary)",
                "query": "Earthquake",
                "evidence_a": {"Alarm": True},
                "p_true_a": p_earthquake_alarm[True],
                "evidence_b": {"Alarm": True, "Burglary": True},
                "p_true_b": p_earthquake_alarm_burglary[True],
                "explanation": (
                    f"Given Alarm=True, P(Earthquake) rises from 0.002 to {p_earthquake_alarm[True]:.4f} (~23.1%). "
                    f"However, learning that Burglary=True 'explains away' the alarm, dropping P(Earthquake) "
                    f"back to {p_earthquake_alarm_burglary[True]:.4f} (~0.20%)."
                ),
            },
        ],
    }
