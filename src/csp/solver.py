"""
solver.py - Classroom CSP Solvers (AC-3, Backtracking, MRV, LCV, Forward Checking)
==================================================================================
Author: Member 3 (CSP Solver Engine)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Explicit implementations of classical CSP algorithms taught in class:
1. AC-3 (Arc Consistency 3) - Preprocessing and in-search domain pruning.
2. Backtracking Search - Depth-first state-space search with constraint propagation.
3. Minimum Remaining Values (MRV) Heuristic - Variable selection ("fail-first").
4. Least Constraining Value (LCV) Heuristic - Value ordering ("fail-last").
5. Forward Checking - Domain filtering after each variable assignment.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque
from .csp_model import CSP, Constraint


class CSPSolver:
    """Classroom CSP Solver supporting AC-3, Forward Checking, MRV, and LCV."""

    def __init__(self, csp: CSP):
        self.csp = csp

    def ac3(self, domains: Optional[Dict[str, List[Any]]] = None) -> Tuple[bool, Dict[str, List[Any]]]:
        """
        AC-3 Arc Consistency Algorithm.
        Enforces arc consistency across all binary constraints.
        Returns: (is_consistent, pruned_domains)
        """
        curr_domains = {v: list(d) for v, d in (domains or self.csp.domains).items()}

        # Queue of arcs (Xi, Xj) for binary constraints
        queue = deque()
        for c in self.csp.all_constraints:
            if len(c.variables) == 2:
                xi, xj = c.variables[0], c.variables[1]
                queue.append((xi, xj, c))
                queue.append((xj, xi, c))

        while queue:
            xi, xj, constraint = queue.popleft()
            revised, curr_domains = self._revise(curr_domains, xi, xj, constraint)

            if revised:
                if len(curr_domains[xi]) == 0:
                    return False, curr_domains  # Domain wipe-out -> Infeasible

                # Re-check arcs pointing to Xi
                for c2 in self.csp.constraints.get(xi, []):
                    if len(c2.variables) == 2:
                        xk = c2.variables[0] if c2.variables[1] == xi else c2.variables[1]
                        if xk != xj:
                            queue.append((xk, xi, c2))

        return True, curr_domains

    def _revise(
        self,
        domains: Dict[str, List[Any]],
        xi: str,
        xj: str,
        constraint: Constraint
    ) -> Tuple[bool, Dict[str, List[Any]]]:
        """
        Revises domain of Xi with respect to Xj.
        Deletes x in D(Xi) if no y in D(Xj) satisfies the constraint.
        """
        revised = False
        new_d_xi = []

        for x in domains[xi]:
            # Check if there exists some y in D(Xj) consistent with x
            satisfied = False
            for y in domains[xj]:
                if constraint.is_satisfied({xi: x, xj: y}):
                    satisfied = True
                    break
            if satisfied:
                new_d_xi.append(x)
            else:
                revised = True

        domains[xi] = new_d_xi
        return revised, domains

    def select_unassigned_variable_mrv(
        self,
        assignment: Dict[str, Any],
        domains: Dict[str, List[Any]]
    ) -> str:
        """
        MRV (Minimum Remaining Values) heuristic with Degree Heuristic tie-breaker.
        Selects unassigned variable with smallest domain size.
        """
        unassigned = [v for v in self.csp.variables if v not in assignment]
        # Sort key: (domain_size, -degree)
        return min(
            unassigned,
            key=lambda v: (len(domains[v]), -len(self.csp.constraints.get(v, [])))
        )

    def order_domain_values_lcv(
        self,
        var: str,
        assignment: Dict[str, Any],
        domains: Dict[str, List[Any]]
    ) -> List[Any]:
        """
        LCV (Least Constraining Value) heuristic.
        Orders values by number of conflicts/reductions they cause in neighboring variables.
        """
        values = domains[var]
        if len(values) <= 1:
            return values

        def count_conflicts(val: Any) -> int:
            conflicts = 0
            temp_assign = dict(assignment)
            temp_assign[var] = val
            for nbr_var in self.csp.variables:
                if nbr_var not in assignment and nbr_var != var:
                    for nbr_val in domains[nbr_var]:
                        temp_assign[nbr_var] = nbr_val
                        for c in self.csp.constraints.get(var, []):
                            if nbr_var in c.variables and not c.is_satisfied(temp_assign):
                                conflicts += 1
                        del temp_assign[nbr_var]
            return conflicts

        return sorted(values, key=count_conflicts)

    def forward_check(
        self,
        var: str,
        val: Any,
        assignment: Dict[str, Any],
        domains: Dict[str, List[Any]]
    ) -> Tuple[bool, Dict[str, List[Any]]]:
        """
        Forward Checking: Prunes inconsistent values from neighbors of var.
        Returns: (is_legal, updated_domains)
        """
        new_domains = {v: list(d) for v, d in domains.items()}
        new_domains[var] = [val]

        temp_assign = dict(assignment)
        temp_assign[var] = val

        for c in self.csp.constraints.get(var, []):
            for neighbor in c.variables:
                if neighbor not in assignment and neighbor != var:
                    pruned_domain = []
                    for nbr_val in new_domains[neighbor]:
                        temp_assign[neighbor] = nbr_val
                        if c.is_satisfied(temp_assign):
                            pruned_domain.append(nbr_val)
                        del temp_assign[neighbor]

                    if len(pruned_domain) == 0:
                        return False, new_domains  # Inconsistency detected
                    new_domains[neighbor] = pruned_domain

        return True, new_domains

    def backtrack_search(
        self,
        use_ac3: bool = True,
        use_mrv: bool = True,
        use_lcv: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Main Backtracking Search procedure with AC-3 preprocessing and Forward Checking.
        """
        domains = {v: list(d) for v, d in self.csp.domains.items()}

        if use_ac3:
            consistent, domains = self.ac3(domains)
            if not consistent:
                return None  # Problem is unsatisfiable before search starts

        return self._backtrack({}, domains, use_mrv, use_lcv)

    def _backtrack(
        self,
        assignment: Dict[str, Any],
        domains: Dict[str, List[Any]],
        use_mrv: bool,
        use_lcv: bool
    ) -> Optional[Dict[str, Any]]:
        """Recursive backtracking step."""
        if self.csp.is_complete(assignment):
            return assignment

        var = self.select_unassigned_variable_mrv(assignment, domains) if use_mrv else [
            v for v in self.csp.variables if v not in assignment
        ][0]

        values = self.order_domain_values_lcv(var, assignment, domains) if use_lcv else domains[var]

        for val in values:
            if self.csp.is_consistent(var, val, assignment):
                assignment[var] = val

                # Forward check consistency
                fc_legal, new_domains = self.forward_check(var, val, assignment, domains)
                if fc_legal:
                    result = self._backtrack(assignment, new_domains, use_mrv, use_lcv)
                    if result is not None:
                        return result

                del assignment[var]

        return None
