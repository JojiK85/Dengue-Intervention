"""
test_search.py - Automated Unit Tests for Module 1 (Search Algorithms)
======================================================================
Author: Member 2 (Search & Geospatial Routing)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar
"""

import pytest
import networkx as nx
from src.core.city_state import CityState
from src.search.algorithms import (
    breadth_first_search,
    depth_first_search,
    uniform_cost_search,
    a_star_search
)
from src.search.heuristics import haversine_admissible_heuristic
from src.search.evaluator import SearchEvaluator


@pytest.fixture
def sample_city() -> CityState:
    city = CityState()
    return city


def test_bfs_shortest_path(sample_city):
    G = sample_city.infra_graph
    res = breadth_first_search(G, "ZONE_CENTRAL", "ZONE_NORTH")
    assert res.path[0] == "ZONE_CENTRAL"
    assert res.path[-1] == "ZONE_NORTH"
    assert res.nodes_expanded > 0
    assert res.cost > 0


def test_dfs_path_finding(sample_city):
    G = sample_city.infra_graph
    res = depth_first_search(G, "ZONE_CENTRAL", "ZONE_SOUTH")
    assert res.path[0] == "ZONE_CENTRAL"
    assert res.path[-1] == "ZONE_SOUTH"
    assert len(res.path) >= 2


def test_ucs_and_astar_optimality(sample_city):
    G = sample_city.infra_graph
    ucs_res = uniform_cost_search(G, "ZONE_SOUTH", "HOSP_CIVIL_CENTRAL")
    astar_res = a_star_search(G, "ZONE_SOUTH", "HOSP_CIVIL_CENTRAL")

    assert ucs_res.path[-1] == "HOSP_CIVIL_CENTRAL"
    assert astar_res.path[-1] == "HOSP_CIVIL_CENTRAL"
    # A* and UCS must produce the exact same optimal cost
    assert pytest.approx(astar_res.cost, 0.01) == ucs_res.cost
    # A* should expand fewer or equal nodes than UCS due to admissible heuristic
    assert astar_res.nodes_expanded <= ucs_res.nodes_expanded


def test_heuristic_admissibility(sample_city):
    """Verifies h(n) <= h*(n) for all pairs of nodes."""
    G = sample_city.infra_graph
    zones = sample_city.get_zone_ids()
    hospitals = sample_city.get_hospital_ids()

    for z in zones:
        for h in hospitals:
            h_val = haversine_admissible_heuristic(G, z, h)
            true_res = uniform_cost_search(G, z, h)
            if true_res.path:
                assert h_val <= true_res.cost + 1e-4, f"Heuristic inadmissible for {z} -> {h}"


def test_hospital_routing_with_capacity(sample_city):
    evaluator = SearchEvaluator(sample_city)
    route_info = evaluator.route_patient_to_best_hospital("ZONE_NORTH", min_beds_required=2)
    assert "assigned_hospital" in route_info
    assert route_info["available_beds"] >= 2
    assert len(route_info["path"]) >= 2


def test_hospital_routing_diverts_on_saturation(sample_city):
    G = sample_city.infra_graph
    # Saturate the closest hospital
    closest_hosp = "HOSP_NORTH_SUBDIV"
    G.nodes[closest_hosp]["available_beds"] = 0

    evaluator = SearchEvaluator(sample_city)
    route_info = evaluator.route_patient_to_best_hospital("ZONE_NORTH", min_beds_required=1)
    # Must divert to another hospital with available beds
    assert route_info["assigned_hospital"] != closest_hosp
    assert route_info["available_beds"] >= 1


def test_cluster_detection(sample_city):
    sample_city.zone_snapshots["ZONE_NORTH"].confirmed_cases = 30.0
    sample_city.zone_snapshots["ZONE_CENTRAL"].confirmed_cases = 25.0
    evaluator = SearchEvaluator(sample_city)
    clusters = evaluator.detect_high_risk_clusters(case_threshold=20.0)
    assert len(clusters) >= 1
    # Check that connected high risk wards are clustered together
    flat = [n for c in clusters for n in c]
    assert "ZONE_NORTH" in flat
