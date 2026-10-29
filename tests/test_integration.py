"""
test_integration.py - Full System Integration Tests
===================================================
Author: Member 3 & Member 7
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar
"""

import pytest
from src.core.city_state import CityState
from src.coordinator.system_coordinator import AdaptiveDengueCoordinator


@pytest.fixture
def coordinator_fixture() -> AdaptiveDengueCoordinator:
    city = CityState()
    return AdaptiveDengueCoordinator(city)


def test_full_coordinator_day_execution(coordinator_fixture):
    day_res = coordinator_fixture.run_day(1)
    assert day_res["day"] == 1
    assert "city_summary" in day_res
    assert day_res["csp_status"] in ["FEASIBLE", "INFEASIBLE_RELAXED"]
    assert "critical_path" in day_res
    assert "rl_action_selected" in day_res
    assert "sitrep" in day_res
    assert len(day_res["sitrep"]) > 100


def test_longitudinal_simulation_pipeline(coordinator_fixture):
    # Run 5 consecutive days
    history = coordinator_fixture.run_simulation_pipeline(num_days=5)
    assert len(history) == 5
    assert coordinator_fixture.city_state.current_day == 5
    summary = coordinator_fixture.city_state.get_summary_snapshot()
    assert summary["day"] == 5
    assert summary["total_zones"] == 6
