import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

from src.city_state import CityState, get_outputs_dir, setup_logger

logger = setup_logger("module1_csp")

class Constraint:
    def __init__(self, variables: List[str], condition_fn: Callable[[Dict[str, Any]], bool], name: str = ""):
        self.variables = variables
        self.condition_fn = condition_fn
        self.name = name or "Constraint"

    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        for v in self.variables:
            if v not in assignment:
                return True
        return self.condition_fn(assignment)

class CSP:
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]]):
        self.variables = list(variables)
        self.domains = {v: list(d) for v, d in domains.items()}
        self.constraints: Dict[str, List[Constraint]] = {v: [] for v in self.variables}
        self.all_constraints: List[Constraint] = []

    def add_constraint(self, constraint: Constraint) -> None:
        self.all_constraints.append(constraint)
        for var in constraint.variables:
            if var in self.constraints:
                self.constraints[var].append(constraint)

    def is_consistent(self, var: str, value: Any, assignment: Dict[str, Any]) -> bool:
        temp_assignment = dict(assignment)
        temp_assignment[var] = value
        for c in self.constraints.get(var, []):
            if not c.is_satisfied(temp_assignment):
                return False
        return True

    def is_complete(self, assignment: Dict[str, Any]) -> bool:
        return len(assignment) == len(self.variables)

class CSPSolver:
    def __init__(self, csp: CSP):
        self.csp = csp

    def ac3(self, domains: Optional[Dict[str, List[Any]]] = None) -> Tuple[bool, Dict[str, List[Any]]]:
        curr_domains = {v: list(d) for v, d in (domains or self.csp.domains).items()}
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
                    return False, curr_domains
                for c2 in self.csp.constraints.get(xi, []):
                    if len(c2.variables) == 2:
                        xk = c2.variables[0] if c2.variables[1] == xi else c2.variables[1]
                        if xk != xj:
                            queue.append((xk, xi, c2))
        return True, curr_domains

    def _revise(self, domains: Dict[str, List[Any]], xi: str, xj: str, constraint: Constraint) -> Tuple[bool, Dict[str, List[Any]]]:
        revised = False
        new_d_xi = []
        for x in domains[xi]:
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

    def select_unassigned_variable_mrv(self, assignment: Dict[str, Any], domains: Dict[str, List[Any]]) -> str:
        unassigned = [v for v in self.csp.variables if v not in assignment]
        return min(unassigned, key=lambda v: (len(domains[v]), -len(self.csp.constraints.get(v, []))))

    def order_domain_values_lcv(self, var: str, assignment: Dict[str, Any], domains: Dict[str, List[Any]]) -> List[Any]:
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

    def forward_check(self, var: str, val: Any, assignment: Dict[str, Any], domains: Dict[str, List[Any]]) -> Tuple[bool, Dict[str, List[Any]]]:
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
                        return False, new_domains
                    new_domains[neighbor] = pruned_domain
        return True, new_domains

    def backtrack_search(self, use_ac3: bool = True, use_mrv: bool = True, use_lcv: bool = True) -> Optional[Dict[str, Any]]:
        domains = {v: list(d) for v, d in self.csp.domains.items()}
        if use_ac3:
            consistent, domains = self.ac3(domains)
            if not consistent:
                return None
        return self._backtrack({}, domains, use_mrv, use_lcv)

    def _backtrack(self, assignment: Dict[str, Any], domains: Dict[str, List[Any]], use_mrv: bool, use_lcv: bool) -> Optional[Dict[str, Any]]:
        if self.csp.is_complete(assignment):
            return assignment

        var = self.select_unassigned_variable_mrv(assignment, domains) if use_mrv else [v for v in self.csp.variables if v not in assignment][0]
        values = self.order_domain_values_lcv(var, assignment, domains) if use_lcv else domains[var]

        for val in values:
            if self.csp.is_consistent(var, val, assignment):
                assignment[var] = val
                fc_legal, new_domains = self.forward_check(var, val, assignment, domains)
                if fc_legal:
                    result = self._backtrack(assignment, new_domains, use_mrv, use_lcv)
                    if result is not None:
                        return result
                del assignment[var]
        return None

class ConflictExplainer:
    @staticmethod
    def diagnose_unsatisfiability(
        zone_risks: Dict[str, str],
        demands: Dict[str, Dict[str, int]],
        available_resources: Dict[str, int]
    ) -> List[str]:
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
        priority_weights = {"RED": 4, "Critical": 4, "ORANGE": 3, "High": 3, "YELLOW": 2, "Medium": 2, "GREEN": 1, "Low": 1}
        sorted_zones = sorted(zone_risks.keys(), key=lambda z: priority_weights.get(zone_risks[z], 1), reverse=True)

        rem_teams = max(0, available_teams)
        rem_kits = max(0, total_kits)
        rem_beds = max(0, total_beds)
        allocation: Dict[str, Dict[str, int]] = {}

        for z in sorted_zones:
            weight = priority_weights.get(zone_risks[z], 1)
            t_alloc = 1 if (weight >= 3 and rem_teams > 0) else 0
            if t_alloc:
                rem_teams -= 1

            k_alloc = 0
            if rem_kits > 0:
                target_k = 50 if weight >= 3 else 20
                k_alloc = min(rem_kits, target_k)
                rem_kits -= k_alloc

            b_alloc = 0
            if rem_beds > 0 and weight >= 3:
                target_b = 10 if weight >= 4 else 5
                b_alloc = min(rem_beds, target_b)
                rem_beds -= b_alloc

            allocation[z] = {"teams": t_alloc, "kits": k_alloc, "beds": b_alloc}
        return allocation

@dataclass
class AllocationResult:
    status: str
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
    def __init__(self, city_state: CityState):
        self.city_state = city_state

    def allocate_resources(
        self,
        zone_risks: Optional[Dict[str, str]] = None,
        available_teams: int = 5,
        total_kits: int = 350,
        total_beds: int = 80
    ) -> AllocationResult:
        zones = self.city_state.get_zone_ids()
        if zone_risks is None:
            zone_risks = {zid: self.city_state.zone_snapshots[zid].alert_level for zid in zones}

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

        if total_team_demand > available_teams or total_kit_demand > total_kits or total_bed_demand > total_beds:
            logger.warning("Resource shortage detected! Triggering conflict diagnosis.")
            reasons = ConflictExplainer.diagnose_unsatisfiability(zone_risks, demands, avail_dict)
            relaxed = ConflictExplainer.solve_relaxed_best_effort(zone_risks, available_teams, total_kits, total_beds)
            return AllocationResult("INFEASIBLE_RELAXED", relaxed, reasons, True)

        variables = []
        domains = {}
        for z in zones:
            v_team, v_kit, v_bed = f"team_{z}", f"kit_{z}", f"bed_{z}"
            variables.extend([v_team, v_kit, v_bed])
            req_t, req_k, req_b = demands[z]["teams"], demands[z]["kits"], demands[z]["beds"]
            domains[v_team] = [req_t, 0] if req_t > 0 else [0]
            domains[v_kit] = [req_k, max(0, req_k - 20)]
            domains[v_bed] = [req_b, max(0, req_b - 5)]

        csp = CSP(variables, domains)
        csp.add_constraint(Constraint(
            [v for v in variables if v.startswith("team_")],
            lambda a: sum(val for k, val in a.items() if k.startswith("team_")) <= available_teams,
            "GlobalTeamLimit"
        ))
        csp.add_constraint(Constraint(
            [v for v in variables if v.startswith("kit_")],
            lambda a: sum(val for k, val in a.items() if k.startswith("kit_")) <= total_kits,
            "GlobalKitLimit"
        ))
        csp.add_constraint(Constraint(
            [v for v in variables if v.startswith("bed_")],
            lambda a: sum(val for k, val in a.items() if k.startswith("bed_")) <= total_beds,
            "GlobalBedLimit"
        ))

        for z in zones:
            if demands[z]["teams"] > 0:
                def make_high_risk_req(v_t):
                    return lambda a: a.get(v_t, 0) >= 1
                csp.add_constraint(Constraint([f"team_{z}"], make_high_risk_req(f"team_{z}"), f"HighRiskTeamReq_{z}"))

        solver = CSPSolver(csp)
        assignment = solver.backtrack_search(use_ac3=True, use_mrv=True, use_lcv=True)

        if assignment:
            formatted = {}
            for z in zones:
                formatted[z] = {
                    "teams": int(assignment.get(f"team_{z}", 0)),
                    "kits": int(assignment.get(f"kit_{z}", 0)),
                    "beds": int(assignment.get(f"bed_{z}", 0))
                }
            return AllocationResult("FEASIBLE", formatted, [], False)
        else:
            reasons = ConflictExplainer.diagnose_unsatisfiability(zone_risks, demands, avail_dict)
            relaxed = ConflictExplainer.solve_relaxed_best_effort(zone_risks, available_teams, total_kits, total_beds)
            return AllocationResult("INFEASIBLE_RELAXED", relaxed, reasons, True)

    def save_allocation_results(self, res: AllocationResult, filename: str = "csp_allocation.json") -> Path:
        output_dir = get_outputs_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)
        return file_path
