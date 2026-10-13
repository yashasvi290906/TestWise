"""Propositional Logic module for syntax parsing, truth tables, and semantic entailment.

Provides a clean, standalone, pedagogically transparent implementation of:
1. AST expressions: Var, Not, And, Or, Implies, Iff, Constant.
2. Recursive-descent parser (strictly no eval/exec; safe grammar and precedence).
3. Truth table generation with combinatorial explosion safeguards.
4. Model checking: Satisfiability, Tautology, and Semantic Entailment.
5. Canonical demonstrations: Modus Ponens vs. Affirming the Consequent.
"""
from __future__ import annotations

import itertools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Maximum variables permitted for exhaustive truth-table enumeration (2^10 = 1024 models)
MAX_EXHAUSTIVE_VARIABLES = 10


# -----------------------------------------------------------------------------
# AST Representation
# -----------------------------------------------------------------------------

class Expr(ABC):
    """Abstract base class for propositional logic expressions."""

    @abstractmethod
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """Evaluate expression under a complete or partial truth assignment."""
        pass

    @abstractmethod
    def variables(self) -> Set[str]:
        """Return the set of free propositional variable names in this expression."""
        pass

    @abstractmethod
    def to_prefix(self) -> str:
        """Normalized prefix representation for canonical debugging."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Expr):
            return False
        return self.to_prefix() == other.to_prefix()

    def __hash__(self) -> int:
        return hash(self.to_prefix())


@dataclass(frozen=True)
class Var(Expr):
    """Propositional variable (e.g., P, Q, Alarm)."""
    name: str

    def __post_init__(self) -> None:
        if not self.name or not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", self.name):
            raise ValueError(f"Invalid variable identifier: {self.name!r}")

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        if self.name not in assignment:
            raise KeyError(f"Variable {self.name!r} not provided in truth assignment: {assignment}")
        return bool(assignment[self.name])

    def variables(self) -> Set[str]:
        return {self.name}

    def to_prefix(self) -> str:
        return f"(var {self.name})"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant(Expr):
    """Boolean constant True or False."""
    val: bool

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        return self.val

    def variables(self) -> Set[str]:
        return set()

    def to_prefix(self) -> str:
        return "(true)" if self.val else "(false)"

    def __str__(self) -> str:
        return "True" if self.val else "False"


@dataclass(frozen=True)
class Not(Expr):
    """Logical negation (~ or NOT)."""
    child: Expr

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        return not self.child.evaluate(assignment)

    def variables(self) -> Set[str]:
        return self.child.variables()

    def to_prefix(self) -> str:
        return f"(not {self.child.to_prefix()})"

    def __str__(self) -> str:
        if isinstance(self.child, (Var, Constant)):
            return f"~{self.child}"
        return f"~({self.child})"


@dataclass(frozen=True)
class And(Expr):
    """Logical conjunction (& or AND)."""
    left: Expr
    right: Expr

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        return self.left.evaluate(assignment) and self.right.evaluate(assignment)

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()

    def to_prefix(self) -> str:
        return f"(and {self.left.to_prefix()} {self.right.to_prefix()})"

    def __str__(self) -> str:
        return f"({self.left} & {self.right})"


@dataclass(frozen=True)
class Or(Expr):
    """Logical disjunction (| or OR)."""
    left: Expr
    right: Expr

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        return self.left.evaluate(assignment) or self.right.evaluate(assignment)

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()

    def to_prefix(self) -> str:
        return f"(or {self.left.to_prefix()} {self.right.to_prefix()})"

    def __str__(self) -> str:
        return f"({self.left} | {self.right})"


@dataclass(frozen=True)
class Implies(Expr):
    """Logical implication (-> or IMPLIES)."""
    left: Expr
    right: Expr

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        # P -> Q is equivalent to ~P | Q
        return (not self.left.evaluate(assignment)) or self.right.evaluate(assignment)

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()

    def to_prefix(self) -> str:
        return f"(implies {self.left.to_prefix()} {self.right.to_prefix()})"

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"


@dataclass(frozen=True)
class Iff(Expr):
    """Logical biconditional (<-> or IFF)."""
    left: Expr
    right: Expr

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        # P <-> Q is True iff P and Q have identical truth values
        return self.left.evaluate(assignment) == self.right.evaluate(assignment)

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()

    def to_prefix(self) -> str:
        return f"(iff {self.left.to_prefix()} {self.right.to_prefix()})"

    def __str__(self) -> str:
        return f"({self.left} <-> {self.right})"


# -----------------------------------------------------------------------------
# Lexer and Recursive-Descent Parser
# -----------------------------------------------------------------------------

TOKEN_SPEC = [
    ("IFF", r"<->|<=>|iff\b|IFF\b"),
    ("IMPLIES", r"->|=>|implies\b|IMPLIES\b"),
    ("OR", r"\||v\b|V\b|or\b|OR\b"),
    ("AND", r"&|\^|and\b|AND\b"),
    ("NOT", r"~|!|not\b|NOT\b"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("TRUE", r"True\b|TRUE\b|true\b|\b1\b"),
    ("FALSE", r"False\b|FALSE\b|false\b|\b0\b"),
    ("IDENT", r"[A-Za-z][A-Za-z0-9_]*"),
    ("WS", r"\s+"),
    ("MISMATCH", r"."),
]

TOKEN_REGEX = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int


def tokenize(text: str) -> List[Token]:
    """Tokenize an input expression string safely without eval or exec."""
    tokens: List[Token] = []
    for mo in TOKEN_REGEX.finditer(text):
        kind = mo.lastgroup
        value = mo.group()
        pos = mo.start()
        if kind == "WS":
            continue
        elif kind == "MISMATCH":
            raise ValueError(f"Unexpected character {value!r} at position {pos}")
        assert kind is not None
        tokens.append(Token(kind, value, pos))
    tokens.append(Token("EOF", "", len(text)))
    return tokens


class Parser:
    """Recursive-descent parser for propositional formulas.
    
    Precedence (lowest to highest):
      1. IFF (<->)
      2. IMPLIES (->, right-associative)
      3. OR (|)
      4. AND (&)
      5. NOT (~, prefix)
      6. Primary (ident, boolean constant, parenthesized expression)
    """

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, expected: Optional[str] = None) -> Token:
        tok = self.current()
        if expected and tok.kind != expected:
            raise ValueError(
                f"Expected {expected} at position {tok.position}, got {tok.kind} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    def parse(self) -> Expr:
        if self.current().kind == "EOF":
            raise ValueError("Cannot parse empty expression.")
        expr = self.parse_iff()
        if self.current().kind != "EOF":
            tok = self.current()
            raise ValueError(f"Unexpected token {tok.value!r} at position {tok.position}")
        return expr

    def parse_iff(self) -> Expr:
        left = self.parse_implies()
        while self.current().kind == "IFF":
            self.consume("IFF")
            right = self.parse_implies()
            left = Iff(left, right)
        return left

    def parse_implies(self) -> Expr:
        left = self.parse_or()
        if self.current().kind == "IMPLIES":
            self.consume("IMPLIES")
            # Right-associative: P -> Q -> R is P -> (Q -> R)
            right = self.parse_implies()
            return Implies(left, right)
        return left

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.current().kind == "OR":
            self.consume("OR")
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_not()
        while self.current().kind == "AND":
            self.consume("AND")
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self) -> Expr:
        if self.current().kind == "NOT":
            self.consume("NOT")
            child = self.parse_not()
            return Not(child)
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.current()
        if tok.kind == "IDENT":
            self.consume("IDENT")
            return Var(tok.value)
        elif tok.kind == "TRUE":
            self.consume("TRUE")
            return Constant(True)
        elif tok.kind == "FALSE":
            self.consume("FALSE")
            return Constant(False)
        elif tok.kind == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_iff()
            self.consume("RPAREN")
            return expr
        else:
            raise ValueError(
                f"Unexpected token {tok.value!r} of kind {tok.kind} at position {tok.position}"
            )


def parse_expr(text: str) -> Expr:
    """Safely parse a text string into an AST Expr."""
    tokens = tokenize(text)
    parser = Parser(tokens)
    return parser.parse()


# -----------------------------------------------------------------------------
# Truth Table and Semantic Entailment Engine
# -----------------------------------------------------------------------------

@dataclass
class TruthTableResult:
    """Complete generated truth table."""
    variables: List[str]
    column_headers: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int


@dataclass
class EntailmentResult:
    """Result of checking premises |= conclusion."""
    is_entailed: bool
    premises: List[str]
    conclusion: str
    variables: List[str]
    total_assignments: int
    satisfying_premise_models: int
    counterexamples: List[Dict[str, bool]]
    explanation: str
    truth_table: TruthTableResult


def generate_truth_table(
    expressions: Union[Expr, Sequence[Expr]],
    headers: Optional[Sequence[str]] = None,
    max_variables: int = MAX_EXHAUSTIVE_VARIABLES,
) -> TruthTableResult:
    """Generate truth table for one or more expressions."""
    if isinstance(expressions, Expr):
        expr_list = [expressions]
    else:
        expr_list = list(expressions)

    # Collect all unique variables
    all_vars: Set[str] = set()
    for e in expr_list:
        all_vars |= e.variables()
    sorted_vars = sorted(all_vars)

    if len(sorted_vars) > max_variables:
        raise ValueError(
            f"Expression contains {len(sorted_vars)} variables, exceeding the maximum safe limit "
            f"of {max_variables} variables (2^{len(sorted_vars)} = {2 ** len(sorted_vars)} rows)."
        )

    col_headers = list(headers) if headers else [str(e) for e in expr_list]

    rows: List[Dict[str, Any]] = []
    # Cartesian product over False, True for each variable
    for combo in itertools.product([False, True], repeat=len(sorted_vars)):
        assignment = dict(zip(sorted_vars, combo))
        row_vals: List[bool] = [e.evaluate(assignment) for e in expr_list]
        rows.append({
            "assignment": assignment,
            "values": row_vals,
        })

    return TruthTableResult(
        variables=sorted_vars,
        column_headers=col_headers,
        rows=rows,
        total_rows=len(rows),
    )


def is_satisfiable(
    expr: Expr,
    max_variables: int = MAX_EXHAUSTIVE_VARIABLES,
) -> Tuple[bool, Optional[Dict[str, bool]]]:
    """Check if there exists at least one truth assignment under which expr evaluates to True."""
    table = generate_truth_table(expr, max_variables=max_variables)
    for row in table.rows:
        if row["values"][0] is True:
            return True, row["assignment"]
    return False, None


def is_tautology(
    expr: Expr,
    max_variables: int = MAX_EXHAUSTIVE_VARIABLES,
) -> bool:
    """Check if expr is True under all possible truth assignments."""
    table = generate_truth_table(expr, max_variables=max_variables)
    return all(row["values"][0] is True for row in table.rows)


def check_entailment(
    premises: Sequence[Union[str, Expr]],
    conclusion: Union[str, Expr],
    max_variables: int = MAX_EXHAUSTIVE_VARIABLES,
) -> EntailmentResult:
    """Check semantic entailment: KB |= alpha.
    
    Definition: Premises entail conclusion iff in all models where every premise
    evaluates to True, the conclusion also evaluates to True.
    """
    prem_exprs = [parse_expr(p) if isinstance(p, str) else p for p in premises]
    conc_expr = parse_expr(conclusion) if isinstance(conclusion, str) else conclusion

    all_exprs = prem_exprs + [conc_expr]
    headers = [str(p) for p in prem_exprs] + [str(conc_expr)]
    table = generate_truth_table(all_exprs, headers=headers, max_variables=max_variables)

    premise_models = 0
    counterexamples: List[Dict[str, bool]] = []

    for row in table.rows:
        assignment = row["assignment"]
        premise_vals = row["values"][: len(prem_exprs)]
        conclusion_val = row["values"][-1]

        # Check if this assignment is a model of all premises
        if all(premise_vals):
            premise_models += 1
            if not conclusion_val:
                counterexamples.append(assignment)

    is_entailed = (len(counterexamples) == 0)

    if is_entailed:
        explanation = (
            f"Entailment HOLDS (KB |= alpha). In all {premise_models} model(s) where every premise "
            f"evaluates to True, the conclusion {str(conc_expr)!r} is also True. There are 0 counterexamples."
        )
    else:
        explanation = (
            f"Entailment FAILS (KB |/= alpha). Found {len(counterexamples)} counterexample assignment(s) "
            f"where all premises are True but conclusion {str(conc_expr)!r} is False: {counterexamples}."
        )

    return EntailmentResult(
        is_entailed=is_entailed,
        premises=[str(p) for p in prem_exprs],
        conclusion=str(conc_expr),
        variables=table.variables,
        total_assignments=table.total_rows,
        satisfying_premise_models=premise_models,
        counterexamples=counterexamples,
        explanation=explanation,
        truth_table=table,
    )


# -----------------------------------------------------------------------------
# Canonical Demonstrations
# -----------------------------------------------------------------------------

def run_modus_ponens_demo() -> EntailmentResult:
    """Demonstrate Modus Ponens: P -> Q, P |= Q (Valid Entailment)."""
    return check_entailment(
        premises=["P -> Q", "P"],
        conclusion="Q",
    )


def run_affirming_consequent_demo() -> EntailmentResult:
    """Demonstrate Affirming the Consequent: P -> Q, Q |/= P (Invalid Fallacy)."""
    return check_entailment(
        premises=["P -> Q", "Q"],
        conclusion="P",
    )
