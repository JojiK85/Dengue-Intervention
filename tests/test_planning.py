"""
test_planning.py - Automated Unit Tests for Module 2 (Planning & Replanning)
============================================================================
Author: Member 5 (Planning) & Member 6 (CPM & Replanning)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar
"""

import pytest
from src.planning.actions import ActionSchema, create_dengue_action_catalog
from src.planning.cpm import CPMTask, CriticalPathAnalyzer
from src.planning.planner import InterventionPlanner
from src.planning.execution_monitor import OnlineExecutionMonitor


def test_action_preconditions_and_effects():
    catalog = create_dengue_action_catalog("ZONE_CENTRAL")
    surv = catalog["surveillance"]
    inspect = catalog["field_inspection"]

    # Initial state
    state = {"zone_active_ZONE_CENTRAL"}
    assert surv.is_applicable(state, "Clear") is True
    assert inspect.is_applicable(state, "Clear") is False  # Missing surveillance_active

    state = surv.apply(state)
    assert inspect.is_applicable(state, "Clear") is True

    state = inspect.apply(state)
    assert "breeding_sites_identified_ZONE_CENTRAL" in state
    assert "vector_density_confirmed_ZONE_CENTRAL" in state


def test_weather_constraint_blocks_chemical_spraying():
    catalog = create_dengue_action_catalog("ZONE_CENTRAL")
    spray = catalog["spraying"]
    state = {"vector_density_confirmed_ZONE_CENTRAL"}

    assert spray.is_applicable(state, "Clear") is True
    # In rain/monsoon, spraying is not applicable to prevent chemical washout
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

    # Path A -> B -> D duration is 2 + 3 + 2 = 7.0
    assert duration == 7.0
    assert "A" in crit_path
    assert "B" in crit_path
    assert "D" in crit_path
    # Task C is non-critical with positive slack
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
    # Verify stages have positive resource accounting
    assert stages[0]["total_cost_inr"] > 0


def test_online_replanning_disruptions():
    planner = InterventionPlanner(["ZONE_NORTH"])
    initial_plan = planner.generate_plan_for_zone(
        "ZONE_NORTH", {"zone_active_ZONE_NORTH"}, {"adult_vectors_reduced_ZONE_NORTH"}
    )
    stages = planner.analyze_parallel_stages(initial_plan)

    monitor = OnlineExecutionMonitor(["ZONE_NORTH"])
    # Disruption: heavy rainfall
    trace = monitor.execute_with_monitoring(
        stages, disruptions=[{"stage_id": 1, "type": "heavy_rainfall", "zone": "ZONE_NORTH"}]
    )
    assert trace["replanning_events_count"] == 1
    assert "suspended outdoor" in trace["replanning_events"][0]["remedial_action"].lower()
