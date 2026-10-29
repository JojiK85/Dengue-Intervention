"""
inference.py - Exact Variable Elimination Algorithm
===================================================
Author: Member 7 (Probabilistic Inference Engine)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Implements the classical Variable Elimination exact inference algorithm:
  P(Query | Evidence) = alpha * sum_{Hidden} prod_{i} P(Node_i | Parents(Node_i), Evidence)
Eliminates variables sequentially by computing pointwise factor products
and marginalizing out non-query variables.
"""

from typing import Dict, List, Any, Optional, Set
from .factor import Factor
from .bayes_net import DengueBayesianNetwork


class VariableElimination:
    """Exact probabilistic inference engine for discrete Bayesian Networks."""

    def __init__(self, bayes_net: DengueBayesianNetwork):
        self.bn = bayes_net

    def _node_to_factor(self, node_name: str, evidence: Dict[str, str]) -> Factor:
        """Constructs an initial factor from a BayesNode CPT and reduces evidence."""
        node = self.bn.nodes[node_name]
        var_list = node.parents + [node.name]
        factor_values = {}

        for parent_tuple, dist in node.cpt.items():
            for val, prob in dist.items():
                full_key = parent_tuple + (val,)
                factor_values[full_key] = prob

        factor = Factor(var_list, factor_values)
        return factor.reduce_evidence(evidence)

    def query(
        self,
        query_var: str,
        evidence: Optional[Dict[str, str]] = None,
        elimination_order: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculates exact posterior distribution P(query_var | evidence).
        Uses min-degree heuristic if elimination_order is omitted.
        """
        evidence = evidence or {}

        # 1. Initialize factors from all network nodes
        factors = [self._node_to_factor(name, evidence) for name in self.bn.nodes]

        # 2. Determine hidden variables to eliminate
        observed_or_query = set(evidence.keys()) | {query_var}
        hidden_vars = [name for name in self.bn.nodes if name not in observed_or_query]

        if elimination_order is None:
            # Min-degree heuristic: eliminate variables appearing in fewest factors first
            hidden_vars = sorted(
                hidden_vars,
                key=lambda v: sum(1 for f in factors if v in f.variables)
            )
        else:
            hidden_vars = [v for v in elimination_order if v in hidden_vars]

        # 3. Sequentially eliminate hidden variables
        for var in hidden_vars:
            factors_with_var = [f for f in factors if var in f.variables]
            factors_without_var = [f for f in factors if var not in f.variables]

            if not factors_with_var:
                continue

            # Pointwise product of all factors containing var
            product_factor = factors_with_var[0]
            for next_factor in factors_with_var[1:]:
                product_factor = product_factor.pointwise_product(next_factor, self.bn.domains)

            # Marginalize (sum out) the hidden variable
            marginalized = product_factor.marginalize(var)
            factors = factors_without_var + [marginalized]

        # 4. Multiply remaining factors containing query_var
        final_factor = factors[0]
        for next_factor in factors[1:]:
            final_factor = final_factor.pointwise_product(next_factor, self.bn.domains)

        # 5. Marginalize out any lingering non-query variables and normalize
        for v in list(final_factor.variables):
            if v != query_var:
                final_factor = final_factor.marginalize(v)

        return final_factor.normalize()
