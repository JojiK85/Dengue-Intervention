"""
test_csp.py - Automated Unit Tests for Module 1 (Constraint Satisfaction Problems)
==================================================================================
Author: Member 3 (CSP Solver) & Member 4 (Resource Allocator)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar
"""

import pytest
from src.core.city_state import CityState
from src.csp.csp_model import CSP, Constraint
from src.csp.solver import CSPSolver
from src.csp.resource_allocator import DengueResourceAllocator


def test_generic_csp_ac3_and_backtracking():
    # Simple Map Coloring problem: WA, NT, SA with Red, Green, Blue
    variables = ["WA", "NT", "SA"]
    domains = {v: ["Red", "Green", "Blue"] for v in variables}
    csp = CSP(variables, domains)

    def diff(v1, v2):
        return lambda a: a.get(v1) != a.get(v2)

    csp.add_constraint(Constraint(["WA", "NT"], diff("WA", "NT")))
    csp.add_constraint(Constraint(["WA", "SA"], diff("WA", "SA")))
    csp.add_constraint(Constraint(["NT", "SA"], diff("NT", "SA")))

    solver = CSPSolver(csp)
    consistent, pruned_domains = solver.ac3()
    assert consistent is True

    assignment = solver.backtrack_search(use_ac3=True, use_mrv=True, use_lcv=True)
    assert assignment is not None
    assert assignment["WA"] != assignment["NT"]
    assert assignment["WA"] != assignment["SA"]
    assert assignment["NT"] != assignment["SA"]


def test_feasible_municipal_resource_allocation():
    city = CityState()
    allocator = DengueResourceAllocator(city)

    # Moderate demand with generous resources
    zone_risks = {z: "GREEN" for z in city.get_zone_ids()}
    zone_risks["ZONE_NORTH"] = "ORANGE"

    res = allocator.allocate_resources(
        zone_risks=zone_risks,
        available_teams=6,
        total_kits=400,
        total_beds=80
    )
    assert res.status == "FEASIBLE"
    assert res.is_relaxed is False
    assert len(res.conflicts) == 0
    assert "ZONE_NORTH" in res.allocation
    assert res.allocation["ZONE_NORTH"]["teams"] >= 1


def test_shortage_conflict_diagnosis_and_relaxation():
    city = CityState()
    allocator = DengueResourceAllocator(city)

    # Severe shortage: 5 high risk wards but only 1 team and 30 kits
    zone_risks = {z: "RED" for z in city.get_zone_ids()}

    res = allocator.allocate_resources(
        zone_risks=zone_risks,
        available_teams=1,
        total_kits=30,
        total_beds=20
    )
    assert res.status == "INFEASIBLE_RELAXED"
    assert res.is_relaxed is True
    assert len(res.conflicts) >= 1
    # Check that explanation specifically mentions team or kit deficit
    explanation_text = " ".join(res.conflicts).lower()
    assert "shortage" in explanation_text or "deficit" in explanation_text or "exceeds" in explanation_text
    # Relaxed allocation should still assign resources prioritized by risk
    assert sum(a["teams"] for a in res.allocation.values()) <= 1


def test_zero_resource_edge_case():
    city = CityState()
    allocator = DengueResourceAllocator(city)
    zone_risks = {z: "RED" for z in city.get_zone_ids()}

    res = allocator.allocate_resources(
        zone_risks=zone_risks,
        available_teams=0,
        total_kits=0,
        total_beds=0
    )
    assert res.is_relaxed is True
    # Zero resources allocated across the board without crashing
    for a in res.allocation.values():
        assert a["teams"] == 0
        assert a["kits"] == 0
        assert a["beds"] == 0
