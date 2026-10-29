"""
csp_model.py - Generic Constraint Satisfaction Problem (CSP) Formulation
========================================================================
Author: Member 3 (CSP Solver Engine)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Generic representation of a Constraint Satisfaction Problem:
  - Variables: X = {X1, X2, ..., Xn}
  - Domains: D = {D1, D2, ..., Dn}
  - Constraints: C = {C1, C2, ..., Cm}
Follows the standard textbook formulation from AIMA Chapter 6.
"""

from typing import List, Dict, Any, Callable, Optional, Set


class Constraint:
    """
    Abstract constraint over a subset of CSP variables.
    condition_fn: takes (assignment_dict) and returns bool.
    """

    def __init__(self, variables: List[str], condition_fn: Callable[[Dict[str, Any]], bool], name: str = ""):
        self.variables = variables
        self.condition_fn = condition_fn
        self.name = name or "Constraint"

    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """Returns True if all constrained variables are assigned and satisfy condition."""
        for v in self.variables:
            if v not in assignment:
                return True  # Vacuously true for unassigned variables
        return self.condition_fn(assignment)


class CSP:
    """
    Generic Constraint Satisfaction Problem container.
    """

    def __init__(self, variables: List[str], domains: Dict[str, List[Any]]):
        self.variables = list(variables)
        self.domains = {v: list(d) for v, d in domains.items()}
        self.constraints: Dict[str, List[Constraint]] = {v: [] for v in self.variables}
        self.all_constraints: List[Constraint] = []

    def add_constraint(self, constraint: Constraint) -> None:
        """Attaches a constraint to all its participating variables."""
        self.all_constraints.append(constraint)
        for var in constraint.variables:
            if var in self.constraints:
                self.constraints[var].append(constraint)

    def is_consistent(self, var: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """Checks if assigning var = value violates any constraints given current assignment."""
        temp_assignment = dict(assignment)
        temp_assignment[var] = value
        for c in self.constraints.get(var, []):
            if not c.is_satisfied(temp_assignment):
                return False
        return True

    def is_complete(self, assignment: Dict[str, Any]) -> bool:
        """Returns True if all variables are legally assigned."""
        return len(assignment) == len(self.variables)
