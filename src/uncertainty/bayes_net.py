"""
bayes_net.py - Bayesian Network DAG & Conditional Probability Tables
=====================================================================
Author: Member 7 (Probabilistic Inference Engine)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Builds the DAG topology and Conditional Probability Tables (CPTs)
modeling urban dengue risk under incomplete observation.
Grounded in M-SDT (Roy et al. 2026) and DengueGNN (GulMohamed et al. 2026).
"""

from typing import Dict, List, Tuple, Any, Optional
from src.core.utils import setup_logger

logger = setup_logger("bayes_net")


class BayesNode:
    """Represents a discrete random variable in the Bayesian Network."""

    def __init__(self, name: str, values: List[str], parents: Optional[List[str]] = None):
        self.name = name
        self.values = list(values)
        self.parents = list(parents or [])
        self.cpt: Dict[Tuple[str, ...], Dict[str, float]] = {}

    def set_cpt(self, cpt_dict: Dict[Tuple[str, ...], Dict[str, float]]) -> None:
        """
        Sets conditional probability distribution table.
        cpt_dict maps: tuple(parent_values) -> {var_val: prob}
        Verifies that sum of probabilities for each condition is 1.0.
        """
        for parent_vals, dist in cpt_dict.items():
            prob_sum = sum(dist.values())
            if not (0.99 <= prob_sum <= 1.01):
                raise ValueError(
                    f"CPT for node {self.name} under condition {parent_vals} does not sum to 1.0 (sums to {prob_sum})"
                )
        self.cpt = cpt_dict


class DengueBayesianNetwork:
    """Directed Acyclic Graph (DAG) and CPT container for dengue epidemiological risk."""

    def __init__(self):
        self.nodes: Dict[str, BayesNode] = {}
        self.domains: Dict[str, List[str]] = {}
        self.build_default_network()

    def add_node(self, node: BayesNode) -> None:
        """Adds a verified node to the DAG."""
        self.nodes[node.name] = node
        self.domains[node.name] = node.values

    def build_default_network(self) -> None:
        """
        Constructs the epidemiological DAG:
          Rainfall, Temperature, Humidity -> BreteauIndex
          BreteauIndex, MobilityInflux -> TrueRisk
          TrueRisk, HealthcareAccess -> ReportedCases
          TrueRisk -> HospitalStrain
        """
        # 1. Root: Rainfall
        node_rain = BayesNode("Rainfall", ["Low", "High"])
        node_rain.set_cpt({(): {"Low": 0.60, "High": 0.40}})
        self.add_node(node_rain)

        # 2. Root: Temperature
        node_temp = BayesNode("Temperature", ["Optimal", "Extreme"])
        node_temp.set_cpt({(): {"Optimal": 0.75, "Extreme": 0.25}})
        self.add_node(node_temp)

        # 3. Root: Humidity
        node_humid = BayesNode("Humidity", ["Normal", "High"])
        node_humid.set_cpt({(): {"Normal": 0.55, "High": 0.45}})
        self.add_node(node_humid)

        # 4. Root: Mobility Influx
        node_mob = BayesNode("MobilityInflux", ["Low", "High"])
        node_mob.set_cpt({(): {"Low": 0.65, "High": 0.35}})
        self.add_node(node_mob)

        # 5. Root: Healthcare Access
        node_access = BayesNode("HealthcareAccess", ["Low", "High"])
        node_access.set_cpt({(): {"Low": 0.40, "High": 0.60}})
        self.add_node(node_access)

        # 6. Child: Breteau / Vector Index (Parents: Rainfall, Temperature, Humidity)
        node_bi = BayesNode("BreteauIndex", ["Low", "High"], ["Rainfall", "Temperature", "Humidity"])
        # Format: (Rainfall, Temperature, Humidity)
        bi_cpt = {
            ("Low", "Optimal", "Normal"): {"Low": 0.85, "High": 0.15},
            ("Low", "Optimal", "High"): {"Low": 0.70, "High": 0.30},
            ("Low", "Extreme", "Normal"): {"Low": 0.95, "High": 0.05},
            ("Low", "Extreme", "High"): {"Low": 0.88, "High": 0.12},
            ("High", "Optimal", "Normal"): {"Low": 0.40, "High": 0.60},
            ("High", "Optimal", "High"): {"Low": 0.15, "High": 0.85},  # Monsoon peak breeding
            ("High", "Extreme", "Normal"): {"Low": 0.75, "High": 0.25},
            ("High", "Extreme", "High"): {"Low": 0.45, "High": 0.55},
        }
        node_bi.set_cpt(bi_cpt)
        self.add_node(node_bi)

        # 7. Intermediate: True Outbreak Risk (Parents: BreteauIndex, MobilityInflux)
        node_risk = BayesNode("TrueRisk", ["Low", "Moderate", "High"], ["BreteauIndex", "MobilityInflux"])
        # Format: (BreteauIndex, MobilityInflux)
        risk_cpt = {
            ("Low", "Low"): {"Low": 0.80, "Moderate": 0.16, "High": 0.04},
            ("Low", "High"): {"Low": 0.55, "Moderate": 0.35, "High": 0.10},
            ("High", "Low"): {"Low": 0.20, "Moderate": 0.45, "High": 0.35},
            ("High", "High"): {"Low": 0.05, "Moderate": 0.25, "High": 0.70},  # Peak contagion
        }
        node_risk.set_cpt(risk_cpt)
        self.add_node(node_risk)

        # 8. Observed: Reported Cases (Parents: TrueRisk, HealthcareAccess)
        # Low HealthcareAccess leads to under-reporting!
        node_cases = BayesNode("ReportedCases", ["Low", "Moderate", "High"], ["TrueRisk", "HealthcareAccess"])
        # Format: (TrueRisk, HealthcareAccess)
        cases_cpt = {
            ("Low", "Low"): {"Low": 0.95, "Moderate": 0.04, "High": 0.01},
            ("Low", "High"): {"Low": 0.90, "Moderate": 0.08, "High": 0.02},
            ("Moderate", "Low"): {"Low": 0.70, "Moderate": 0.25, "High": 0.05},  # Hidden moderate cases!
            ("Moderate", "High"): {"Low": 0.15, "Moderate": 0.70, "High": 0.15},
            ("High", "Low"): {"Low": 0.45, "Moderate": 0.40, "High": 0.15},     # Under-reporting hotspot!
            ("High", "High"): {"Low": 0.05, "Moderate": 0.20, "High": 0.75},    # Clear detection
        }
        node_cases.set_cpt(cases_cpt)
        self.add_node(node_cases)

        # 9. Consequence: Hospital Strain (Parents: TrueRisk)
        node_hosp = BayesNode("HospitalStrain", ["Normal", "High"], ["TrueRisk"])
        hosp_cpt = {
            ("Low",): {"Normal": 0.92, "High": 0.08},
            ("Moderate",): {"Normal": 0.65, "High": 0.35},
            ("High",): {"Normal": 0.18, "High": 0.82},
        }
        node_hosp.set_cpt(hosp_cpt)
        self.add_node(node_hosp)
