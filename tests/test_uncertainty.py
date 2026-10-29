"""
test_uncertainty.py - Automated Unit Tests for Module 3 (Bayesian Network)
==========================================================================
Author: Member 7 (Inference Engine) & Member 8 (Risk Estimator)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar
"""

import pytest
from src.uncertainty.factor import Factor
from src.uncertainty.bayes_net import DengueBayesianNetwork
from src.uncertainty.inference import VariableElimination
from src.uncertainty.confidence import compute_shannon_entropy, compute_confidence_score
from src.uncertainty.risk_estimator import ZoneRiskEstimator


def test_cpt_probabilities_sum_to_one():
    bn = DengueBayesianNetwork()
    for name, node in bn.nodes.items():
        for cond, dist in node.cpt.items():
            prob_sum = sum(dist.values())
            assert 0.99 <= prob_sum <= 1.01, f"Node {name} CPT condition {cond} violates probability axiom"


def test_factor_operations():
    # Factor 1: P(A)
    f1 = Factor(["A"], {("T",): 0.6, ("F",): 0.4})
    # Factor 2: P(B | A)
    f2 = Factor(["A", "B"], {
        ("T", "T"): 0.8, ("T", "F"): 0.2,
        ("F", "T"): 0.1, ("F", "F"): 0.9
    })
    domains = {"A": ["T", "F"], "B": ["T", "F"]}

    f_prod = f1.pointwise_product(f2, domains)
    assert len(f_prod.values) == 4
    assert pytest.approx(f_prod.values[("T", "T")], 1e-4) == 0.48

    # Marginalize out A: P(B) = sum_A P(A, B)
    f_marg = f_prod.marginalize("A")
    assert f_marg.variables == ["B"]
    p_b_true = f_marg.values[("T",)]
    # P(B=T) = 0.6*0.8 + 0.4*0.1 = 0.48 + 0.04 = 0.52
    assert pytest.approx(p_b_true, 1e-4) == 0.52


def test_variable_elimination_evidence_updating():
    bn = DengueBayesianNetwork()
    ve = VariableElimination(bn)

    # Prior risk without evidence
    prior = ve.query("TrueRisk", {})
    assert "High" in prior and "Low" in prior
    assert pytest.approx(sum(prior.values()), 1e-3) == 1.0

    # Under monsoon conditions (High rainfall + Optimal temp + High humidity)
    monsoon_posterior = ve.query("TrueRisk", {
        "Rainfall": "High",
        "Temperature": "Optimal",
        "Humidity": "High"
    })
    # P(High) must increase significantly under monsoon conditions
    assert monsoon_posterior["High"] > prior["High"]


def test_under_reporting_detection():
    estimator = ZoneRiskEstimator()
    # Slum sector with poor clinic access, heavy rain, and high mosquito density
    raw_ev = {
        "rainfall_mm": 60.0,
        "temperature_c": 30.0,
        "humidity_pct": 82.0,
        "breteau_index": 38.0,
        "clinic_access_ratio": 0.25,
        "daily_reported_cases": 2.0,  # Artificially low reported count!
        "mobility_influx": 0.30
    }
    res = estimator.estimate_zone_risk("ZONE_SLUM", raw_ev)
    # True risk should be High and under-reporting flag must be True
    assert res["underreporting_detected"] is True
    assert res["p_high"] >= 0.50
    assert res["alert_level"] in ["ORANGE", "RED"]


def test_shannon_entropy_confidence_bounds():
    # Deterministic distribution -> Entropy = 0, Confidence = 1.0
    det_dist = {"High": 1.0, "Low": 0.0}
    assert compute_shannon_entropy(det_dist) == 0.0
    assert compute_confidence_score(det_dist) == 1.0

    # Uniform distribution -> Entropy = 1.0 bit, Confidence = 0.0
    unif_dist = {"High": 0.5, "Low": 0.5}
    assert pytest.approx(compute_confidence_score(unif_dist), 0.01) == 0.0
