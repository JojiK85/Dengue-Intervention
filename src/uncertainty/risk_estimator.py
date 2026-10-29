"""
risk_estimator.py - Zone Risk Estimator & Under-Reporting Detector
==================================================================
Author: Member 8 (Epidemiological Risk & Under-Reporting)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Assesses true municipal outbreak risk and detects latent under-reporting.
Grounded in M-SDT (Roy et al. 2026, Ahmedabad municipal study):
  Decouples the latent epidemiological state TrueRisk from biased clinical reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from src.core.city_state import CityState
from src.core.utils import get_outputs_dir, setup_logger
from .bayes_net import DengueBayesianNetwork
from .inference import VariableElimination
from .confidence import compute_confidence_score

logger = setup_logger("risk_estimator")


class ZoneRiskEstimator:
    """Estimates true outbreak risk, detects under-reporting, and computes confidence scores."""

    def __init__(self, bayes_net: Optional[DengueBayesianNetwork] = None):
        self.bn = bayes_net or DengueBayesianNetwork()
        self.infer_engine = VariableElimination(self.bn)

    @staticmethod
    def discretize_evidence(raw_features: Dict[str, Any]) -> Dict[str, str]:
        """Maps continuous real-world metrics into discrete network evidence."""
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
        """
        Runs Variable Elimination to compute true posterior risk and detect under-reporting.
        """
        discrete_ev = self.discretize_evidence(raw_features)
        posterior_risk = self.infer_engine.query("TrueRisk", discrete_ev)
        p_high = posterior_risk.get("High", 0.0)
        p_mod = posterior_risk.get("Moderate", 0.0)
        p_low = posterior_risk.get("Low", 0.0)

        # Under-reporting detection (M-SDT Roy et al. 2026):
        # High underlying risk (> 50%) but Low reported cases in low-access settlement
        reported_discrete = discrete_ev.get("ReportedCases", "Low")
        underreporting = (p_high >= 0.50 or (p_high + p_mod >= 0.75)) and (reported_discrete == "Low")

        # Multi-tier Alert Level
        if p_high >= 0.60:
            alert = "RED"
        elif p_high >= 0.35 or p_mod >= 0.50:
            alert = "ORANGE"
        elif p_mod >= 0.30:
            alert = "YELLOW"
        else:
            alert = "GREEN"

        confidence = compute_confidence_score(posterior_risk)

        # Expected true cases calculation
        raw_cases = float(raw_features.get("daily_reported_cases", 0.0))
        mult = float(raw_features.get("underreporting_factor", 3.0 if underreporting else 1.8))
        est_true_cases = max(raw_cases, round(raw_cases * mult, 1))

        return {
            "zone_id": zone_id,
            "evidence": discrete_ev,
            "posterior_risk": posterior_risk,
            "p_high": p_high,
            "p_mod": p_mod,
            "p_low": p_low,
            "alert_level": alert,
            "underreporting_detected": underreporting,
            "confidence_score": confidence,
            "estimated_true_cases": est_true_cases
        }

    def evaluate_city_surveillance(
        self,
        city_state: CityState,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Evaluates all zones in the CityState, updates snapshots, and exports CSV.
        """
        if output_dir is None:
            output_dir = get_outputs_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for zid, snap in city_state.zone_snapshots.items():
            # Extract features from snapshot and zone demographic table
            zone_row = city_state.zones[city_state.zones["zone_id"] == zid]
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

            # Update live digital-twin state
            snap.risk_score = res["p_high"]
            snap.alert_level = res["alert_level"]
            snap.underreporting_flag = res["underreporting_detected"]
            snap.confidence_score = res["confidence_score"]
            snap.estimated_true_cases = res["estimated_true_cases"]

            records.append({
                "zone_id": zid,
                "confirmed_cases": snap.confirmed_cases,
                "estimated_true_cases": snap.estimated_true_cases,
                "breteau_index": snap.breteau_index,
                "P_Risk_High": round(res["p_high"], 3),
                "P_Risk_Mod": round(res["p_mod"], 3),
                "P_Risk_Low": round(res["p_low"], 3),
                "alert_level": res["alert_level"],
                "underreporting_flag": res["underreporting_detected"],
                "confidence": res["confidence_score"]
            })

        df = pd.DataFrame(records)
        csv_path = output_dir / "zone_posterior_risks.csv"
        df.to_csv(csv_path, index=False)
        return df

    def demonstrate_dynamic_belief_updating(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Demonstrates dynamic sequential belief updating as new evidence arrives.
        Step 1: Prior (no evidence)
        Step 2: Heavy rainfall observed
        Step 3: High Breteau larval index confirmed
        Step 4: Commuter mobility influx observed
        Step 5: High clinic fever reports arrive
        """
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
                "evidence_stage": name,
                "evidence_vars": str(list(ev.keys())),
                "P_High": dist.get("High", 0.0),
                "P_Moderate": dist.get("Moderate", 0.0),
                "P_Low": dist.get("Low", 0.0),
                "confidence_score": conf
            })

        df = pd.DataFrame(records)
        csv_path = output_dir / "belief_updating_log.csv"
        df.to_csv(csv_path, index=False)
        return df
