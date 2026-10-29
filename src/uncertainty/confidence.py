"""
confidence.py - Shannon Entropy Confidence Quantification
==========================================================
Author: Member 8 (Epidemiological Risk & Under-Reporting)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Quantifies epistemic certainty of posterior probability distributions using Shannon Entropy:
  H(X) = - sum_{x in X} P(x) * log2(P(x))
Normalized Confidence Score:
  C(X) = 1.0 - (H(X) / log2(|X|))
where C(X) in [0.0, 1.0]. High entropy -> Low confidence; Sharp distribution -> High confidence.
"""

import math
from typing import Dict


def compute_shannon_entropy(dist: Dict[str, float]) -> float:
    """Computes Shannon entropy in bits."""
    entropy = 0.0
    for prob in dist.values():
        if prob > 1e-9:
            entropy -= prob * math.log2(prob)
    return entropy


def compute_confidence_score(dist: Dict[str, float]) -> float:
    """
    Computes normalized confidence score in [0.0, 1.0].
    Returns 1.0 for deterministic distributions and 0.0 for uniform uncertainty.
    """
    num_outcomes = len(dist)
    if num_outcomes <= 1:
        return 1.0

    max_entropy = math.log2(num_outcomes)
    actual_entropy = compute_shannon_entropy(dist)
    normalized = 1.0 - (actual_entropy / max_entropy)
    return round(max(0.0, min(1.0, normalized)), 3)
