"""
conflict_explainer.py - CSP Shortage Explainer & Relaxed Allocator
==================================================================
Author: Member 4 (Resource Allocation & Diagnostics)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

When municipal resources are insufficient to satisfy all hard constraints simultaneously
(e.g., severe kit shortages, unavailable field teams), standard CSP returns FAILURE.
This module:
1. Explains why the problem is unsatisfiable by pinpointing violated constraints.
2. Computes a prioritized Relaxed Best-Effort Allocation based on Bayesian Outbreak Risk.
"""

from typing import Dict, List, Any, Tuple
from src.core.utils import setup_logger

logger = setup_logger("conflict_explainer")


class ConflictExplainer:
    """Diagnoses CSP inconsistencies and produces transparent relaxation plans."""

    @staticmethod
    def diagnose_unsatisfiability(
        zone_risks: Dict[str, str],
        demands: Dict[str, Dict[str, int]],
        available_resources: Dict[str, int]
    ) -> List[str]:
        """
        Analyzes resource budgets and minimum requirements to generate human-readable explanations.
        """
        reasons = []
        total_team_demand = sum(d.get("teams", 0) for d in demands.values())
        total_kit_demand = sum(d.get("kits", 0) for d in demands.values())
        total_bed_demand = sum(d.get("beds", 0) for d in demands.values())

        avail_teams = available_resources.get("teams", 0)
        avail_kits = available_resources.get("kits", 0)
        avail_beds = available_resources.get("beds", 0)

        if total_team_demand > avail_teams:
            reasons.append(
                f"Global Team Shortage: Wards demand {total_team_demand} field inspection teams, "
                f"but only {avail_teams} squads are available (deficit: {total_team_demand - avail_teams})."
            )

        if total_kit_demand > avail_kits:
            reasons.append(
                f"Diagnostic Kit Stockout: Total municipal requirement of {total_kit_demand} test kits "
                f"exceeds inventory of {avail_kits} kits (deficit: {total_kit_demand - avail_kits})."
            )

        if total_bed_demand > avail_beds:
            reasons.append(
                f"Hospital Bed Saturation: Requested surge capacity of {total_bed_demand} beds "
                f"exceeds available free beds of {avail_beds} (deficit: {total_bed_demand - avail_beds})."
            )

        # Check high-risk unserved wards
        high_risk_zones = [z for z, r in zone_risks.items() if r in ["High", "Critical", "ORANGE", "RED"]]
        if len(high_risk_zones) > avail_teams:
            reasons.append(
                f"Coverage Infeasibility: {len(high_risk_zones)} wards are High/Critical risk, "
                f"but only {avail_teams} teams exist, making 100% high-risk coverage mathematically impossible."
            )

        return reasons

    @staticmethod
    def solve_relaxed_best_effort(
        zone_risks: Dict[str, str],
        available_teams: int,
        total_kits: int,
        total_beds: int
    ) -> Dict[str, Dict[str, int]]:
        """
        Computes a greedy priority allocation when hard CSP is unsatisfiable.
        Sorts zones by outbreak priority: Critical > High > Medium > Low.
        """
        priority_weights = {"RED": 4, "Critical": 4, "ORANGE": 3, "High": 3, "YELLOW": 2, "Medium": 2, "GREEN": 1, "Low": 1}
        sorted_zones = sorted(
            zone_risks.keys(),
            key=lambda z: priority_weights.get(zone_risks[z], 1),
            reverse=True
        )

        rem_teams = max(0, available_teams)
        rem_kits = max(0, total_kits)
        rem_beds = max(0, total_beds)

        allocation: Dict[str, Dict[str, int]] = {}

        for z in sorted_zones:
            risk = zone_risks[z]
            weight = priority_weights.get(risk, 1)

            # Assign team if available and high/critical risk
            t_alloc = 0
            if weight >= 3 and rem_teams > 0:
                t_alloc = 1
                rem_teams -= 1

            # Proportional kit allocation
            k_alloc = 0
            if rem_kits > 0:
                target_k = 50 if weight >= 3 else 20
                k_alloc = min(rem_kits, target_k)
                rem_kits -= k_alloc

            # Proportional bed allocation
            b_alloc = 0
            if rem_beds > 0 and weight >= 3:
                target_b = 10 if weight >= 4 else 5
                b_alloc = min(rem_beds, target_b)
                rem_beds -= b_alloc

            allocation[z] = {
                "teams": t_alloc,
                "kits": k_alloc,
                "beds": b_alloc
            }

        return allocation
