import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd

from src.city_state import CityState, get_outputs_dir, setup_logger

logger = setup_logger("module3_uncertainty")

class Factor:
    def __init__(self, variables: List[str], values: Dict[Tuple[str, ...], float]):
        self.variables = list(variables)
        self.values = dict(values)

    def pointwise_product(self, other: "Factor", domains: Dict[str, List[str]]) -> "Factor":
        combined_vars = list(self.variables)
        for v in other.variables:
            if v not in combined_vars:
                combined_vars.append(v)

        new_values: Dict[Tuple[str, ...], float] = {}

        def build_assignments(idx: int, current_assign: Dict[str, str]):
            if idx == len(combined_vars):
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
        if len(self.variables) != 1:
            raise ValueError(f"Can only normalize 1-variable factor, has {self.variables}")
        total = sum(self.values.values())
        if total <= 0.0:
            n = len(self.values)
            return {k[0]: 1.0 / n for k in self.values}
        return {k[0]: round(prob / total, 4) for k, prob in self.values.items()}

class BayesNode:
    def __init__(self, name: str, values: List[str], parents: Optional[List[str]] = None):
        self.name = name
        self.values = list(values)
        self.parents = list(parents or [])
        self.cpt: Dict[Tuple[str, ...], Dict[str, float]] = {}

    def set_cpt(self, cpt_dict: Dict[Tuple[str, ...], Dict[str, float]]) -> None:
        for parent_vals, dist in cpt_dict.items():
            prob_sum = sum(dist.values())
            if not (0.99 <= prob_sum <= 1.01):
                raise ValueError(f"CPT for node {self.name} under {parent_vals} does not sum to 1.0 (sums to {prob_sum})")
        self.cpt = cpt_dict

class DengueBayesianNetwork:
    def __init__(self):
        self.nodes: Dict[str, BayesNode] = {}
        self.domains: Dict[str, List[str]] = {}
        self.build_default_network()

    def add_node(self, node: BayesNode) -> None:
        self.nodes[node.name] = node
        self.domains[node.name] = node.values

    def build_default_network(self) -> None:
        node_rain = BayesNode("Rainfall", ["Low", "High"])
        node_rain.set_cpt({(): {"Low": 0.60, "High": 0.40}})
        self.add_node(node_rain)

        node_temp = BayesNode("Temperature", ["Optimal", "Extreme"])
        node_temp.set_cpt({(): {"Optimal": 0.75, "Extreme": 0.25}})
        self.add_node(node_temp)

        node_humid = BayesNode("Humidity", ["Normal", "High"])
        node_humid.set_cpt({(): {"Normal": 0.55, "High": 0.45}})
        self.add_node(node_humid)

        node_mob = BayesNode("MobilityInflux", ["Low", "High"])
        node_mob.set_cpt({(): {"Low": 0.65, "High": 0.35}})
        self.add_node(node_mob)

        node_access = BayesNode("HealthcareAccess", ["Low", "High"])
        node_access.set_cpt({(): {"Low": 0.40, "High": 0.60}})
        self.add_node(node_access)

        node_bi = BayesNode("BreteauIndex", ["Low", "High"], ["Rainfall", "Temperature", "Humidity"])
        bi_cpt = {
            ("Low", "Optimal", "Normal"): {"Low": 0.85, "High": 0.15},
            ("Low", "Optimal", "High"): {"Low": 0.70, "High": 0.30},
            ("Low", "Extreme", "Normal"): {"Low": 0.95, "High": 0.05},
            ("Low", "Extreme", "High"): {"Low": 0.88, "High": 0.12},
            ("High", "Optimal", "Normal"): {"Low": 0.40, "High": 0.60},
            ("High", "Optimal", "High"): {"Low": 0.15, "High": 0.85},
            ("High", "Extreme", "Normal"): {"Low": 0.75, "High": 0.25},
            ("High", "Extreme", "High"): {"Low": 0.45, "High": 0.55},
        }
        node_bi.set_cpt(bi_cpt)
        self.add_node(node_bi)

        node_risk = BayesNode("TrueRisk", ["Low", "Moderate", "High"], ["BreteauIndex", "MobilityInflux"])
        risk_cpt = {
            ("Low", "Low"): {"Low": 0.80, "Moderate": 0.16, "High": 0.04},
            ("Low", "High"): {"Low": 0.55, "Moderate": 0.35, "High": 0.10},
            ("High", "Low"): {"Low": 0.20, "Moderate": 0.45, "High": 0.35},
            ("High", "High"): {"Low": 0.05, "Moderate": 0.25, "High": 0.70},
        }
        node_risk.set_cpt(risk_cpt)
        self.add_node(node_risk)

        node_cases = BayesNode("ReportedCases", ["Low", "Moderate", "High"], ["TrueRisk", "HealthcareAccess"])
        cases_cpt = {
            ("Low", "Low"): {"Low": 0.95, "Moderate": 0.04, "High": 0.01},
            ("Low", "High"): {"Low": 0.90, "Moderate": 0.08, "High": 0.02},
            ("Moderate", "Low"): {"Low": 0.70, "Moderate": 0.25, "High": 0.05},
            ("Moderate", "High"): {"Low": 0.15, "Moderate": 0.70, "High": 0.15},
            ("High", "Low"): {"Low": 0.45, "Moderate": 0.40, "High": 0.15},
            ("High", "High"): {"Low": 0.05, "Moderate": 0.20, "High": 0.75},
        }
        node_cases.set_cpt(cases_cpt)
        self.add_node(node_cases)

        node_hosp = BayesNode("HospitalStrain", ["Normal", "High"], ["TrueRisk"])
        hosp_cpt = {
            ("Low",): {"Normal": 0.92, "High": 0.08},
            ("Moderate",): {"Normal": 0.65, "High": 0.35},
            ("High",): {"Normal": 0.18, "High": 0.82},
        }
        node_hosp.set_cpt(hosp_cpt)
        self.add_node(node_hosp)

class VariableElimination:
    def __init__(self, bayes_net: DengueBayesianNetwork):
        self.bn = bayes_net

    def _node_to_factor(self, node_name: str, evidence: Dict[str, str]) -> Factor:
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
        evidence = evidence or {}
        factors = [self._node_to_factor(name, evidence) for name in self.bn.nodes]
        observed_or_query = set(evidence.keys()) | {query_var}
        hidden_vars = [name for name in self.bn.nodes if name not in observed_or_query]

        if elimination_order is None:
            hidden_vars = sorted(hidden_vars, key=lambda v: sum(1 for f in factors if v in f.variables))
        else:
            hidden_vars = [v for v in elimination_order if v in hidden_vars]

        for var in hidden_vars:
            factors_with_var = [f for f in factors if var in f.variables]
            factors_without_var = [f for f in factors if var not in f.variables]
            if not factors_with_var:
                continue
            product_factor = factors_with_var[0]
            for next_factor in factors_with_var[1:]:
                product_factor = product_factor.pointwise_product(next_factor, self.bn.domains)
            marginalized = product_factor.marginalize(var)
            factors = factors_without_var + [marginalized]

        final_factor = factors[0]
        for next_factor in factors[1:]:
            final_factor = final_factor.pointwise_product(next_factor, self.bn.domains)

        for v in list(final_factor.variables):
            if v != query_var:
                final_factor = final_factor.marginalize(v)
        return final_factor.normalize()

def compute_shannon_entropy(dist: Dict[str, float]) -> float:
    entropy = 0.0
    for prob in dist.values():
        if prob > 1e-9:
            entropy -= prob * math.log2(prob)
    return entropy

def compute_confidence_score(dist: Dict[str, float]) -> float:
    num_outcomes = len(dist)
    if num_outcomes <= 1:
        return 1.0
    max_entropy = math.log2(num_outcomes)
    actual_entropy = compute_shannon_entropy(dist)
    normalized = 1.0 - (actual_entropy / max_entropy)
    return round(max(0.0, min(1.0, normalized)), 3)

class ZoneRiskEstimator:
    def __init__(self, bayes_net: Optional[DengueBayesianNetwork] = None):
        self.bn = bayes_net or DengueBayesianNetwork()
        self.infer_engine = VariableElimination(self.bn)

    @staticmethod
    def discretize_evidence(raw_features: Dict[str, Any]) -> Dict[str, str]:
        evidence: Dict[str, str] = {}
        if "rainfall_mm" in raw_features:
            evidence["Rainfall"] = "High" if float(raw_features["rainfall_mm"]) >= 35.0 else "Low"
        if "temperature_c" in raw_features:
            t = float(raw_features["temperature_c"])
            evidence["Temperature"] = "Optimal" if (24.0 <= t <= 34.0) else "Extreme"
        if "humidity_pct" in raw_features:
            evidence["Humidity"] = "High" if float(raw_features["humidity_pct"]) >= 70.0 else "Normal"
        if "breteau_index" in raw_features:
            evidence["BreteauIndex"] = "High" if float(raw_features["breteau_index"]) >= 20.0 else "Low"
        if "mobility_influx" in raw_features:
            evidence["MobilityInflux"] = "High" if float(raw_features["mobility_influx"]) >= 0.20 else "Low"
        if "clinic_access_ratio" in raw_features:
            evidence["HealthcareAccess"] = "High" if float(raw_features["clinic_access_ratio"]) >= 0.50 else "Low"
        if "daily_reported_cases" in raw_features:
            c = float(raw_features["daily_reported_cases"])
            if c >= 15.0:
                evidence["ReportedCases"] = "High"
            elif c >= 6.0:
                evidence["ReportedCases"] = "Moderate"
            else:
                evidence["ReportedCases"] = "Low"
        return evidence

    def estimate_zone_risk(self, zone_id: str, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        discrete_ev = self.discretize_evidence(raw_features)
        posterior_risk = self.infer_engine.query("TrueRisk", discrete_ev)
        p_high = posterior_risk.get("High", 0.0)
        p_mod = posterior_risk.get("Moderate", 0.0)
        p_low = posterior_risk.get("Low", 0.0)

        reported_discrete = discrete_ev.get("ReportedCases", "Low")
        underreporting = (p_high >= 0.50 or (p_high + p_mod >= 0.75)) and (reported_discrete == "Low")

        if p_high >= 0.60:
            alert = "RED"
        elif p_high >= 0.35 or p_mod >= 0.50:
            alert = "ORANGE"
        elif p_mod >= 0.30:
            alert = "YELLOW"
        else:
            alert = "GREEN"

        confidence = compute_confidence_score(posterior_risk)
        raw_cases = float(raw_features.get("daily_reported_cases", 0.0))
        mult = float(raw_features.get("underreporting_factor", 3.0 if underreporting else 1.8))
        est_true_cases = max(raw_cases, round(raw_cases * mult, 1))

        return {
            "zone_id": zone_id, "evidence": discrete_ev, "posterior_risk": posterior_risk,
            "p_high": p_high, "p_mod": p_mod, "p_low": p_low, "alert_level": alert,
            "underreporting_detected": underreporting, "confidence_score": confidence,
            "estimated_true_cases": est_true_cases
        }

    def evaluate_city_surveillance(self, city_state: CityState, output_dir: Optional[Path] = None) -> pd.DataFrame:
        if output_dir is None:
            output_dir = get_outputs_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for zid, snap in city_state.zone_snapshots.items():
            zone_row = city_state.zones[city_state.zones["zone_id"] == zid] if not city_state.zones.empty else pd.DataFrame()
            access = float(zone_row["clinic_access_ratio"].values[0]) if not zone_row.empty else 0.5
            mult = float(zone_row["underreporting_factor"].values[0]) if not zone_row.empty else 2.5

            raw_ev = {
                "rainfall_mm": snap.rainfall_mm,
                "temperature_c": snap.temp_c,
                "humidity_pct": snap.humidity_pct,
                "breteau_index": snap.breteau_index,
                "daily_reported_cases": snap.confirmed_cases,
                "clinic_access_ratio": access,
                "underreporting_factor": mult
            }
            res = self.estimate_zone_risk(zid, raw_ev)
            snap.risk_score = res["p_high"]
            snap.alert_level = res["alert_level"]
            snap.underreporting_flag = res["underreporting_detected"]
            snap.confidence_score = res["confidence_score"]
            snap.estimated_true_cases = res["estimated_true_cases"]

            records.append({
                "zone_id": zid, "confirmed_cases": snap.confirmed_cases,
                "estimated_true_cases": snap.estimated_true_cases,
                "breteau_index": snap.breteau_index,
                "P_Risk_High": round(res["p_high"], 3), "P_Risk_Mod": round(res["p_mod"], 3),
                "P_Risk_Low": round(res["p_low"], 3), "alert_level": res["alert_level"],
                "underreporting_flag": res["underreporting_detected"], "confidence": res["confidence_score"]
            })

        df = pd.DataFrame(records)
        csv_path = output_dir / "zone_posterior_risks.csv"
        df.to_csv(csv_path, index=False)
        return df

    def demonstrate_dynamic_belief_updating(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        if output_dir is None:
            output_dir = get_outputs_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        stages = [
            ("1_Prior_No_Evidence", {}),
            ("2_After_Heavy_Rainfall", {"Rainfall": "High"}),
            ("3_After_Larval_Breteau_Spike", {"Rainfall": "High", "BreteauIndex": "High"}),
            ("4_After_Mobility_Commuter_Flux", {"Rainfall": "High", "BreteauIndex": "High", "MobilityInflux": "High"}),
            ("5_After_Clinical_Case_Confirmation", {"Rainfall": "High", "BreteauIndex": "High", "MobilityInflux": "High", "ReportedCases": "High"})
        ]
        records = []
        for name, ev in stages:
            dist = self.infer_engine.query("TrueRisk", ev)
            conf = compute_confidence_score(dist)
            records.append({
                "evidence_stage": name, "evidence_vars": str(list(ev.keys())),
                "P_High": dist.get("High", 0.0), "P_Moderate": dist.get("Moderate", 0.0),
                "P_Low": dist.get("Low", 0.0), "confidence_score": conf
            })

        df = pd.DataFrame(records)
        csv_path = output_dir / "belief_updating_log.csv"
        df.to_csv(csv_path, index=False)
        return df
