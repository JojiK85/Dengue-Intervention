import pytest
import numpy as np
import networkx as nx

from src.city_state import CityState
from src.module1_search import (
    breadth_first_search,
    depth_first_search,
    uniform_cost_search,
    a_star_search,
    haversine_admissible_heuristic,
    SearchEvaluator
)
from src.module1_csp import CSP, Constraint, CSPSolver, DengueResourceAllocator
from src.module2_planning import (
    create_dengue_action_catalog,
    CPMTask,
    CriticalPathAnalyzer,
    InterventionPlanner,
    OnlineExecutionMonitor
)
from src.module3_uncertainty import (
    Factor,
    DengueBayesianNetwork,
    VariableElimination,
    ZoneRiskEstimator,
    compute_shannon_entropy,
    compute_confidence_score
)
from src.coordinator import AdaptiveDengueCoordinator
from src.scenario_runner import ScenarioRunner

@pytest.fixture
def sample_city() -> CityState:
    return CityState()

@pytest.fixture
def coordinator_fixture() -> AdaptiveDengueCoordinator:
    return AdaptiveDengueCoordinator(CityState())

# ==========================================
# MODULE 1: SEARCH ALGORITHM TESTS (7 tests)
# ==========================================

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
    assert pytest.approx(astar_res.cost, 0.01) == ucs_res.cost
    assert astar_res.nodes_expanded <= ucs_res.nodes_expanded

def test_heuristic_admissibility(sample_city):
    G = sample_city.infra_graph
    zones = sample_city.get_zone_ids()
    hospitals = sample_city.get_hospital_ids()
    for z in zones:
        for h in hospitals:
            h_val = haversine_admissible_heuristic(G, z, h)
            true_res = uniform_cost_search(G, z, h)
            if true_res.path:
                assert h_val <= true_res.cost + 1e-4

def test_hospital_routing_with_capacity(sample_city):
    evaluator = SearchEvaluator(sample_city)
    route_info = evaluator.route_patient_to_best_hospital("ZONE_NORTH", min_beds_required=2)
    assert "assigned_hospital" in route_info
    assert route_info["available_beds"] >= 2
    assert len(route_info["path"]) >= 2

def test_hospital_routing_diverts_on_saturation(sample_city):
    G = sample_city.infra_graph
    closest_hosp = "HOSP_NORTH_SUBDIV"
    G.nodes[closest_hosp]["available_beds"] = 0
    evaluator = SearchEvaluator(sample_city)
    route_info = evaluator.route_patient_to_best_hospital("ZONE_NORTH", min_beds_required=1)
    assert route_info["assigned_hospital"] != closest_hosp
    assert route_info["available_beds"] >= 1

def test_cluster_detection(sample_city):
    sample_city.zone_snapshots["ZONE_NORTH"].confirmed_cases = 30.0
    sample_city.zone_snapshots["ZONE_CENTRAL"].confirmed_cases = 25.0
    evaluator = SearchEvaluator(sample_city)
    clusters = evaluator.detect_high_risk_clusters(case_threshold=20.0)
    assert len(clusters) >= 1
    flat = [n for c in clusters for n in c]
    assert "ZONE_NORTH" in flat

# ==========================================
# MODULE 1: CSP RESOURCE ALLOCATION (4 tests)
# ==========================================

def test_generic_csp_ac3_and_backtracking():
    variables = ["WA", "NT", "SA"]
    domains = {v: ["Red", "Green", "Blue"] for v in variables}
    csp = CSP(variables, domains)
    def diff(v1, v2):
        return lambda a: a.get(v1) != a.get(v2)
    csp.add_constraint(Constraint(["WA", "NT"], diff("WA", "NT")))
    csp.add_constraint(Constraint(["WA", "SA"], diff("WA", "SA")))
    csp.add_constraint(Constraint(["NT", "SA"], diff("NT", "SA")))
    solver = CSPSolver(csp)
    consistent, _ = solver.ac3()
    assert consistent is True
    assignment = solver.backtrack_search(use_ac3=True, use_mrv=True, use_lcv=True)
    assert assignment is not None
    assert assignment["WA"] != assignment["NT"]
    assert assignment["WA"] != assignment["SA"]
    assert assignment["NT"] != assignment["SA"]

def test_feasible_municipal_resource_allocation(sample_city):
    allocator = DengueResourceAllocator(sample_city)
    zone_risks = {z: "GREEN" for z in sample_city.get_zone_ids()}
    zone_risks["ZONE_NORTH"] = "ORANGE"
    res = allocator.allocate_resources(zone_risks=zone_risks, available_teams=6, total_kits=400, total_beds=80)
    assert res.status == "FEASIBLE"
    assert res.is_relaxed is False
    assert len(res.conflicts) == 0
    assert res.allocation["ZONE_NORTH"]["teams"] >= 1

def test_shortage_conflict_diagnosis_and_relaxation(sample_city):
    allocator = DengueResourceAllocator(sample_city)
    zone_risks = {z: "RED" for z in sample_city.get_zone_ids()}
    res = allocator.allocate_resources(zone_risks=zone_risks, available_teams=1, total_kits=30, total_beds=20)
    assert res.status == "INFEASIBLE_RELAXED"
    assert res.is_relaxed is True
    assert len(res.conflicts) >= 1
    explanation_text = " ".join(res.conflicts).lower()
    assert "shortage" in explanation_text or "deficit" in explanation_text or "exceeds" in explanation_text
    assert sum(a["teams"] for a in res.allocation.values()) <= 1

def test_zero_resource_edge_case(sample_city):
    allocator = DengueResourceAllocator(sample_city)
    zone_risks = {z: "RED" for z in sample_city.get_zone_ids()}
    res = allocator.allocate_resources(zone_risks=zone_risks, available_teams=0, total_kits=0, total_beds=0)
    assert res.is_relaxed is True
    for a in res.allocation.values():
        assert a["teams"] == 0
        assert a["kits"] == 0
        assert a["beds"] == 0

# ==========================================
# MODULE 2: PLANNING & CPM TESTS (5 tests)
# ==========================================

def test_action_preconditions_and_effects():
    catalog = create_dengue_action_catalog("ZONE_CENTRAL")
    surv = catalog["surveillance"]
    inspect = catalog["field_inspection"]
    state = {"zone_active_ZONE_CENTRAL"}
    assert surv.is_applicable(state, "Clear") is True
    assert inspect.is_applicable(state, "Clear") is False
    state = surv.apply(state)
    assert inspect.is_applicable(state, "Clear") is True
    state = inspect.apply(state)
    assert "breeding_sites_identified_ZONE_CENTRAL" in state

def test_weather_constraint_blocks_chemical_spraying():
    catalog = create_dengue_action_catalog("ZONE_CENTRAL")
    spray = catalog["spraying"]
    state = {"vector_density_confirmed_ZONE_CENTRAL"}
    assert spray.is_applicable(state, "Clear") is True
    assert spray.is_applicable(state, "Rain") is False
    assert spray.is_applicable(state, "Monsoon") is False

def test_cpm_critical_path_and_slack():
    tasks = [
        CPMTask("A", 2.0, []),
        CPMTask("B", 3.0, ["A"]),
        CPMTask("C", 1.0, ["A"]),
        CPMTask("D", 2.0, ["B", "C"])
    ]
    scheduled, crit_path, duration = CriticalPathAnalyzer.compute_critical_path(tasks)
    assert duration == 7.0
    assert "A" in crit_path
    assert "B" in crit_path
    assert "D" in crit_path
    task_c = next(t for t in scheduled if t.task_id == "C")
    assert task_c.slack > 0
    assert task_c.is_critical is False

def test_planner_and_parallel_stage_partitioning():
    planner = InterventionPlanner(["ZONE_NORTH", "ZONE_SOUTH"])
    initial_state = {"zone_active_ZONE_NORTH", "zone_active_ZONE_SOUTH"}
    goals = {
        "ZONE_NORTH": {"surveillance_active_ZONE_NORTH", "breeding_containers_cleared_ZONE_NORTH"},
        "ZONE_SOUTH": {"surveillance_active_ZONE_SOUTH"}
    }
    plan = planner.build_multi_zone_plan(initial_state, goals, "Clear")
    assert len(plan) >= 3
    stages = planner.analyze_parallel_stages(plan)
    assert len(stages) >= 2
    assert stages[0]["total_cost_inr"] > 0

def test_online_replanning_disruptions():
    planner = InterventionPlanner(["ZONE_NORTH"])
    initial_plan = planner.generate_plan_for_zone(
        "ZONE_NORTH", {"zone_active_ZONE_NORTH"}, {"adult_vectors_reduced_ZONE_NORTH"}
    )
    stages = planner.analyze_parallel_stages(initial_plan)
    monitor = OnlineExecutionMonitor(["ZONE_NORTH"])
    trace = monitor.execute_with_monitoring(
        stages, disruptions=[{"stage_id": 1, "type": "heavy_rainfall", "zone": "ZONE_NORTH"}]
    )
    assert trace["replanning_events_count"] == 1
    assert "suspended outdoor" in trace["replanning_events"][0]["remedial_action"].lower()

# ==========================================
# MODULE 3: UNCERTAINTY TESTS (5 tests)
# ==========================================

def test_cpt_probabilities_sum_to_one():
    bn = DengueBayesianNetwork()
    for name, node in bn.nodes.items():
        for cond, dist in node.cpt.items():
            prob_sum = sum(dist.values())
            assert 0.99 <= prob_sum <= 1.01

def test_factor_operations():
    f1 = Factor(["A"], {("T",): 0.6, ("F",): 0.4})
    f2 = Factor(["A", "B"], {
        ("T", "T"): 0.8, ("T", "F"): 0.2,
        ("F", "T"): 0.1, ("F", "F"): 0.9
    })
    domains = {"A": ["T", "F"], "B": ["T", "F"]}
    f_prod = f1.pointwise_product(f2, domains)
    assert len(f_prod.values) == 4
    assert pytest.approx(f_prod.values[("T", "T")], 1e-4) == 0.48
    f_marg = f_prod.marginalize("A")
    assert f_marg.variables == ["B"]
    assert pytest.approx(f_marg.values[("T",)], 1e-4) == 0.52

def test_variable_elimination_evidence_updating():
    bn = DengueBayesianNetwork()
    ve = VariableElimination(bn)
    prior = ve.query("TrueRisk", {})
    assert "High" in prior and "Low" in prior
    assert pytest.approx(sum(prior.values()), 1e-3) == 1.0
    monsoon_posterior = ve.query("TrueRisk", {
        "Rainfall": "High", "Temperature": "Optimal", "Humidity": "High"
    })
    assert monsoon_posterior["High"] > prior["High"]

def test_under_reporting_detection():
    estimator = ZoneRiskEstimator()
    raw_ev = {
        "rainfall_mm": 60.0, "temperature_c": 30.0, "humidity_pct": 82.0,
        "breteau_index": 38.0, "clinic_access_ratio": 0.25,
        "daily_reported_cases": 2.0, "mobility_influx": 0.30
    }
    res = estimator.estimate_zone_risk("ZONE_SLUM", raw_ev)
    assert res["underreporting_detected"] is True
    assert res["p_high"] >= 0.50
    assert res["alert_level"] in ["ORANGE", "RED"]

def test_shannon_entropy_confidence_bounds():
    det_dist = {"High": 1.0, "Low": 0.0}
    assert compute_shannon_entropy(det_dist) == 0.0
    assert compute_confidence_score(det_dist) == 1.0
    unif_dist = {"High": 0.5, "Low": 0.5}
    assert pytest.approx(compute_confidence_score(unif_dist), 0.01) == 0.0

# ==========================================
# SCENARIOS AND INTEGRATION TESTS (3 tests)
# ==========================================

def test_all_eight_scenarios_execute_successfully(coordinator_fixture):
    runner = ScenarioRunner(coordinator_fixture)
    df = runner.run_all_scenarios()
    assert len(df) == 8
    for _, row in df.iterrows():
        assert row["system_status"] == "SUCCESS"
        assert len(str(row["key_metric"])) > 0

def test_full_coordinator_day_execution(coordinator_fixture):
    day_res = coordinator_fixture.run_day(1)
    assert day_res["day"] == 1
    assert "city_summary" in day_res
    assert day_res["csp_status"] in ["FEASIBLE", "INFEASIBLE_RELAXED"]
    assert "critical_path" in day_res
    assert "sitrep" in day_res
    assert len(day_res["sitrep"]) > 100

def test_longitudinal_simulation_pipeline(coordinator_fixture):
    history = coordinator_fixture.run_simulation_pipeline(num_days=5)
    assert len(history) == 5
    assert coordinator_fixture.city_state.current_day == 5
    summary = coordinator_fixture.city_state.get_summary_snapshot()
    assert summary["day"] == 5
    assert summary["total_zones"] == 6
