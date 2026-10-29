"""
factor.py - Pure Python Factor Operations for Exact Bayesian Inference
======================================================================
Author: Member 7 (Probabilistic Inference Engine)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Implements factor representations and operations for Variable Elimination:
1. Pointwise Product (Factor Multiplication) - Combines probability distributions.
2. Marginalization (Summing Out) - Eliminates hidden variables.
3. Evidence Reduction - Instantiates observed evidence.
4. Normalization - Ensures posterior probabilities sum to 1.0.
Grounded in AIMA Chapter 14 (Probabilistic Reasoning).
"""

from typing import List, Dict, Tuple, Any, Set
import math


class Factor:
    """
    Multidimensional discrete probability table over a subset of random variables.
    Values are indexed by tuple of assigned string values: tuple(var_assignments) -> prob.
    """

    def __init__(self, variables: List[str], values: Dict[Tuple[str, ...], float]):
        self.variables = list(variables)
        self.values = dict(values)

    def pointwise_product(self, other: "Factor", domains: Dict[str, List[str]]) -> "Factor":
        """
        Multiplies two factors: psi_3(X, Y, Z) = psi_1(X, Y) * psi_2(Y, Z).
        Union of variables, matching values on common variables.
        """
        combined_vars = list(self.variables)
        for v in other.variables:
            if v not in combined_vars:
                combined_vars.append(v)

        new_values: Dict[Tuple[str, ...], float] = {}

        # Recursive Cartesian product over combined variables
        def build_assignments(idx: int, current_assign: Dict[str, str]):
            if idx == len(combined_vars):
                # Look up in self
                self_key = tuple(current_assign[v] for v in self.variables)
                other_key = tuple(current_assign[v] for v in other.variables)
                val_self = self.values.get(self_key, 0.0)
                val_other = other.values.get(other_key, 0.0)

                combined_key = tuple(current_assign[v] for v in combined_vars)
                new_values[combined_key] = val_self * val_other
                return

            v_curr = combined_vars[idx]
            for val in domains.get(v_curr, []):
                current_assign[v_curr] = val
                build_assignments(idx + 1, current_assign)
                del current_assign[v_curr]

        build_assignments(0, {})
        return Factor(combined_vars, new_values)

    def marginalize(self, var_to_sum_out: str) -> "Factor":
        """
        Sums out a variable: tau(X, Z) = sum_y psi(X, y, Z).
        """
        if var_to_sum_out not in self.variables:
            return self

        new_vars = [v for v in self.variables if v != var_to_sum_out]
        var_idx = self.variables.index(var_to_sum_out)
        new_values: Dict[Tuple[str, ...], float] = {}

        for key, prob in self.values.items():
            reduced_key = tuple(val for i, val in enumerate(key) if i != var_idx)
            new_values[reduced_key] = new_values.get(reduced_key, 0.0) + prob

        return Factor(new_vars, new_values)

    def reduce_evidence(self, evidence: Dict[str, str]) -> "Factor":
        """
        Restricts factor rows to match instantiated evidence variables.
        """
        relevant_evidence = {v: evidence[v] for v in self.variables if v in evidence}
        if not relevant_evidence:
            return self

        new_values = {}
        for key, prob in self.values.items():
            matches = True
            for i, var in enumerate(self.variables):
                if var in relevant_evidence and key[i] != relevant_evidence[var]:
                    matches = False
                    break
            if matches:
                new_values[key] = prob

        return Factor(self.variables, new_values)

    def normalize(self) -> Dict[str, float]:
        """
        Normalizes a 1-variable factor to produce a probability distribution summing to 1.0.
        """
        if len(self.variables) != 1:
            raise ValueError(f"Can only normalize 1-variable factor, has {self.variables}")

        total = sum(self.values.values())
        if total <= 0.0:
            # Uniform fallback
            n = len(self.values)
            return {k[0]: 1.0 / n for k in self.values}

        return {k[0]: round(prob / total, 4) for k, prob in self.values.items()}
