"""
test_scenarios.py - Automated Tests for 8 Mandatory Public Health Scenarios
============================================================================
Author: Member 4 & Member 8
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar
"""

import pytest
from src.coordinator.system_coordinator import AdaptiveDengueCoordinator
from src.scenarios.scenario_runner import ScenarioRunner


def test_all_eight_scenarios_execute_successfully():
    coordinator = AdaptiveDengueCoordinator()
    runner = ScenarioRunner(coordinator)
    df = runner.run_all_scenarios()

    assert len(df) == 8
    # Every scenario must complete with SUCCESS status
    for _, row in df.iterrows():
        assert row["system_status"] == "SUCCESS", f"Scenario {row['scenario_id']} failed"
        assert len(str(row["key_metric"])) > 0
