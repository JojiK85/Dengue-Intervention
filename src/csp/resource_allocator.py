"""
resource_allocator.py - Dengue Municipal Resource Allocator & Conflict Diagnostician
=====================================================================================
Author: Member 4 (Resource Allocation & Diagnostics)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Formulates municipal dengue resource allocation as a formal Constraint Satisfaction Problem:
- Field teams assigned to inspection/larviciding
- Diagnostic test kits distributed to clinics
- Surge hospital beds mobilized
- Hard budget and capacity constraints
- When infeasible, returns conflict diagnostics and a relaxed best-effort plan.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.core.city_state import CityState
from src.core.utils import get_outputs_dir, setup_logger
from .csp_model import CSP, Constraint
from .solver import CSPSolver
from .conflict_explainer import ConflictExplainer

logger = setup_logger("resource_allocator")


@dataclass
class AllocationResult:
    """Consolidated outcome of the CSP allocation run."""
    status: str                         # FEASIBLE, INFEASIBLE_RELAXED, ERROR
    allocation: Dict[str, Dict[str, int]]
    conflicts: List[str]
    is_relaxed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "allocation": self.allocation,
            "conflicts": self.conflicts,
            "is_relaxed": self.is_relaxed
        }


class DengueResourceAllocator:
    """Manages CSP resource allocation across municipal wards."""

    def __init__(self, city_state: CityState):
        self.city_state = city_state

    def allocate_resources(
        self,
        zone_risks: Optional[Dict[str, str]] = None,
        available_teams: int = 5,
        total_kits: int = 350,
        total_beds: int = 80
    ) -> AllocationResult:
        """
        Builds and solves the resource allocation CSP.
        If infeasible, diagnoses constraints and executes relaxed best-effort allocation.
        """
        zones = self.city_state.get_zone_ids()
        if zone_risks is None:
            zone_risks = {
                zid: self.city_state.zone_snapshots[zid].alert_level
                for zid in zones
            }

        # Demands estimation
        demands = {}
        for z in zones:
            risk = zone_risks.get(z, "GREEN")
            if risk in ["RED", "Critical", "ORANGE", "High"]:
                demands[z] = {"teams": 1, "kits": 75, "beds": 15}
            elif risk in ["YELLOW", "Medium"]:
                demands[z] = {"teams": 0, "kits": 35, "beds": 5}
            else:
                demands[z] = {"teams": 0, "kits": 15, "beds": 0}

        avail_dict = {"teams": available_teams, "kits": total_kits, "beds": total_beds}
        total_team_demand = sum(d["teams"] for d in demands.values())
        total_kit_demand = sum(d["kits"] for d in demands.values())
        total_bed_demand = sum(d["beds"] for d in demands.values())

        # If demands strictly exceed available inventory, hard CSP is infeasible
        if total_team_demand > available_teams or total_kit_demand > total_kits or total_bed_demand > total_beds:
            logger.warning("Resource shortage detected! Triggering conflict diagnosis.")
            reasons = ConflictExplainer.diagnose_unsatisfiability(zone_risks, demands, avail_dict)
            relaxed = ConflictExplainer.solve_relaxed_best_effort(
                zone_risks, available_teams, total_kits, total_beds
            )
            return AllocationResult(
                status="INFEASIBLE_RELAXED",
                allocation=relaxed,
                conflicts=reasons,
                is_relaxed=True
            )

        # Build formal CSP
        variables = []
        domains = {}
        for z in zones:
            v_team = f"team_{z}"
            v_kit = f"kit_{z}"
            v_bed = f"bed_{z}"
            variables.extend([v_team, v_kit, v_bed])

            # Discrete domain values
            req_t = demands[z]["teams"]
            req_k = demands[z]["kits"]
            req_b = demands[z]["beds"]

            domains[v_team] = [req_t, 0] if req_t > 0 else [0]
            domains[v_kit] = [req_k, max(0, req_k - 20)]
            domains[v_bed] = [req_b, max(0, req_b - 5)]

        csp = CSP(variables, domains)

        # Global budget constraints
        def global_team_limit(assignment: Dict[str, Any]) -> bool:
            curr_teams = sum(val for k, val in assignment.items() if k.startswith("team_"))
            return curr_teams <= available_teams

        def global_kit_limit(assignment: Dict[str, Any]) -> bool:
            curr_kits = sum(val for k, val in assignment.items() if k.startswith("kit_"))
            return curr_kits <= total_kits

        def global_bed_limit(assignment: Dict[str, Any]) -> bool:
            curr_beds = sum(val for k, val in assignment.items() if k.startswith("bed_"))
            return curr_beds <= total_beds

        csp.add_constraint(Constraint([v for v in variables if v.startswith("team_")], global_team_limit, "GlobalTeamLimit"))
        csp.add_constraint(Constraint([v for v in variables if v.startswith("kit_")], global_kit_limit, "GlobalKitLimit"))
        csp.add_constraint(Constraint([v for v in variables if v.startswith("bed_")], global_bed_limit, "GlobalBedLimit"))

        # High risk coverage constraints
        for z in zones:
            if demands[z]["teams"] > 0:
                def make_high_risk_req(v_t):
                    return lambda a: a.get(v_t, 0) >= 1
                csp.add_constraint(Constraint([f"team_{z}"], make_high_risk_req(f"team_{z}"), f"HighRiskTeamReq_{z}"))

        solver = CSPSolver(csp)
        assignment = solver.backtrack_search(use_ac3=True, use_mrv=True, use_lcv=True)

        if assignment:
            formatted: Dict[str, Dict[str, int]] = {}
            for z in zones:
                formatted[z] = {
                    "teams": int(assignment.get(f"team_{z}", 0)),
                    "kits": int(assignment.get(f"kit_{z}", 0)),
                    "beds": int(assignment.get(f"bed_{z}", 0))
                }
            return AllocationResult(
                status="FEASIBLE",
                allocation=formatted,
                conflicts=[],
                is_relaxed=False
            )
        else:
            reasons = ConflictExplainer.diagnose_unsatisfiability(zone_risks, demands, avail_dict)
            relaxed = ConflictExplainer.solve_relaxed_best_effort(
                zone_risks, available_teams, total_kits, total_beds
            )
            return AllocationResult(
                status="INFEASIBLE_RELAXED",
                allocation=relaxed,
                conflicts=reasons,
                is_relaxed=True
            )

    def save_allocation_results(
        self,
        res: AllocationResult,
        filename: str = "csp_allocation.json"
    ) -> Path:
        """Saves allocation results and conflict diagnostics to outputs/tables/."""
        output_dir = get_outputs_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)
        return file_path
