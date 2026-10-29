"""
planner.py - Forward State-Space & Concurrent Intervention Planner
===================================================================
Author: Member 5 (Causal Planning & Formal PDDL)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Plans multi-zone public-health operations:
1. Forward state-space search satisfying STRIPS causal preconditions.
2. Concurrent stage partitioning (parallel vs sequential actions).
3. Integrates Critical Path Method (CPM) to identify scheduling bottlenecks.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import pandas as pd

from src.core.utils import get_outputs_dir, setup_logger
from .actions import ActionSchema, create_dengue_action_catalog
from .cpm import CPMTask, CriticalPathAnalyzer

logger = setup_logger("planner")


class InterventionPlanner:
    """Generates goal-directed public-health response plans."""

    def __init__(self, target_zones: List[str]):
        self.target_zones = target_zones
        self.catalogs: Dict[str, Dict[str, ActionSchema]] = {
            z: create_dengue_action_catalog(z) for z in target_zones
        }

    def generate_plan_for_zone(
        self,
        zone: str,
        initial_state: Set[str],
        goal_state: Set[str],
        weather: str = "Clear"
    ) -> List[ActionSchema]:
        """
        Generates a valid sequence of actions transitioning initial_state to goal_state.
        """
        catalog = self.catalogs.get(zone, {})
        current_state = set(initial_state)
        plan: List[ActionSchema] = []

        # Canonical public-health progression:
        # surveillance -> testing / inspection -> source_reduction / spraying -> hospital_prep -> follow_up
        action_order = [
            "surveillance",
            "laboratory_testing",
            "field_inspection",
            "public_awareness",
            "mosquito_source_reduction",
            "spraying",
            "hospital_preparation",
            "follow_up_assessment"
        ]

        for act_key in action_order:
            if goal_state.issubset(current_state):
                break
            action = catalog.get(act_key)
            if action and action.is_applicable(current_state, weather):
                current_state = action.apply(current_state)
                plan.append(action)

        return plan

    def build_multi_zone_plan(
        self,
        initial_state: Set[str],
        goals_by_zone: Dict[str, Set[str]],
        weather: str = "Clear"
    ) -> List[ActionSchema]:
        """Synthesizes actions across all municipal zones."""
        full_plan: List[ActionSchema] = []
        for z in self.target_zones:
            zone_goals = goals_by_zone.get(z, set())
            zone_plan = self.generate_plan_for_zone(z, initial_state, zone_goals, weather)
            full_plan.extend(zone_plan)
        return full_plan

    def analyze_parallel_stages(self, plan: List[ActionSchema]) -> List[Dict[str, Any]]:
        """
        Partitions a flat list of actions into concurrent execution stages.
        Actions in the same stage can be executed concurrently without causal dependencies.
        """
        stages: List[List[ActionSchema]] = []
        produced_effects: Set[str] = set()

        for action in plan:
            placed = False
            # Try to place in earliest compatible stage
            for stage in stages:
                # Check for resource conflict (e.g., both require teams and team pool is limited)
                team_conflict = (action.required_teams > 0 and
                                 sum(a.required_teams for a in stage) >= 4)
                # Check causal dependency with other actions in this stage
                causal_conflict = any(
                    not action.preconditions.isdisjoint(a.add_effects) or
                    not a.preconditions.isdisjoint(action.add_effects)
                    for a in stage
                )

                if not team_conflict and not causal_conflict:
                    stage.append(action)
                    placed = True
                    break

            if not placed:
                stages.append([action])

        formatted_stages = []
        for i, stage in enumerate(stages, 1):
            formatted_stages.append({
                "stage_id": i,
                "concurrent_actions_count": len(stage),
                "total_teams_required": sum(a.required_teams for a in stage),
                "total_kits_required": sum(a.required_kits for a in stage),
                "total_cost_inr": sum(a.cost_inr for a in stage),
                "actions": [a.to_dict() for a in stage]
            })

        return formatted_stages

    def build_cpm_schedule(self, plan: List[ActionSchema]) -> Tuple[List[CPMTask], List[str], float]:
        """
        Constructs a CPM task network from action preconditions and add effects,
        then calculates early/late schedules and the critical path.
        """
        tasks: List[CPMTask] = []
        action_producers: Dict[str, str] = {}  # effect -> task_id

        # First pass: identify which action produces which effect
        for i, act in enumerate(plan, 1):
            tid = f"T{i}_{act.name}"
            for eff in act.add_effects:
                action_producers[eff] = tid

        # Second pass: establish dependencies
        for i, act in enumerate(plan, 1):
            tid = f"T{i}_{act.name}"
            deps = []
            for pre in act.preconditions:
                if pre in action_producers and action_producers[pre] != tid:
                    deps.append(action_producers[pre])

            tasks.append(CPMTask(
                task_id=tid,
                duration=float(act.duration_days),
                dependencies=deps,
                zone=act.zone,
                action_type=act.name.split("_")[0]
            ))

        return CriticalPathAnalyzer.compute_critical_path(tasks)

    def export_plan_artifacts(
        self,
        stages: List[Dict[str, Any]],
        cpm_tasks: List[CPMTask],
        critical_path: List[str],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Exports planning schedules and CPM tables to CSV and JSON."""
        if output_dir is None:
            output_dir = get_outputs_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        stages_json = output_dir / "planning_initial_stages.json"
        with open(stages_json, "w", encoding="utf-8") as f:
            json.dump(stages, f, indent=2)

        cpm_records = [t.to_dict() for t in cpm_tasks]
        cpm_df = pd.DataFrame(cpm_records)
        cpm_csv = output_dir / "cpm_schedule_table.csv"
        cpm_df.to_csv(cpm_csv, index=False)

        cpm_json = output_dir / "cpm_critical_path.json"
        with open(cpm_json, "w", encoding="utf-8") as f:
            json.dump({
                "critical_path_tasks": critical_path,
                "total_tasks": len(cpm_tasks),
                "tasks": cpm_records
            }, f, indent=2)

        return {"stages": stages_json, "cpm_csv": cpm_csv, "cpm_json": cpm_json}
