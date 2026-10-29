"""
execution_monitor.py - Online Execution Monitor & Dynamic Replanner
===================================================================
Author: Member 6 (Temporal Planning, CPM & Replanning)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Monitors plan execution step-by-step and performs automated dynamic replanning
when environmental or operational disruptions occur:
  1. Case surge: Triggers emergency triage and immediate spraying.
  2. Team outage: Re-sequences tasks across active squads.
  3. Heavy rainfall: Suspends outdoor adulticide fogging to avoid chemical washout;
     pivots to indoor source reduction and drainage clearance.
  4. Weaker efficacy: Escalates to combined follow-up sweeps.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from src.core.utils import get_outputs_dir, setup_logger
from .actions import ActionSchema, create_dengue_action_catalog

logger = setup_logger("execution_monitor")


class OnlineExecutionMonitor:
    """Monitors live plan execution and triggers automated schedule repair."""

    def __init__(self, target_zones: List[str]):
        self.target_zones = target_zones
        self.catalogs = {z: create_dengue_action_catalog(z) for z in target_zones}

    def execute_with_monitoring(
        self,
        initial_stages: List[Dict[str, Any]],
        disruptions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes plan stages sequentially. If a disruption occurs at stage k,
        dynamically repairs the remaining schedule.
        """
        disruptions = disruptions or []
        disruption_map = {d.get("stage_id", -1): d for d in disruptions}

        execution_trace = []
        replanning_events = []
        active_stages = list(initial_stages)
        stage_idx = 0

        while stage_idx < len(active_stages):
            current_stage = active_stages[stage_idx]
            current_id = current_stage["stage_id"]

            # Check if an unexpected disruption occurred at this stage
            if current_id in disruption_map:
                event = disruption_map[current_id]
                event_type = event.get("type", "unknown")
                target_zone = event.get("zone", self.target_zones[0])
                logger.info(f"Disruption detected at stage {current_id}: {event_type} in {target_zone}")

                replan_action, reason, repaired_stages = self._replan(
                    event_type, target_zone, active_stages[stage_idx:]
                )

                replanning_events.append({
                    "stage_id": current_id,
                    "event_type": event_type,
                    "target_zone": target_zone,
                    "reason": reason,
                    "remedial_action": replan_action
                })

                # Replace future stages with repaired schedule
                active_stages = active_stages[:stage_idx] + repaired_stages
                current_stage = active_stages[stage_idx]

            # Mark actions as COMPLETED
            executed_actions = []
            for act in current_stage["actions"]:
                act_record = dict(act)
                act_record["status"] = "COMPLETED"
                executed_actions.append(act_record)

            execution_trace.append({
                "stage_id": current_id,
                "executed_actions": executed_actions
            })
            stage_idx += 1

        return {
            "total_stages_executed": len(execution_trace),
            "replanning_events_count": len(replanning_events),
            "replanning_events": replanning_events,
            "final_execution_trace": execution_trace
        }

    def _replan(
        self,
        event_type: str,
        zone: str,
        remaining_stages: List[Dict[str, Any]]
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Synthesizes dynamic schedule repairs based on disruption type."""
        catalog = self.catalogs.get(zone, {})

        if event_type == "surge":
            reason = f"Sudden surge in clinical fever reports in {zone} requires urgent capacity expansion."
            remedy = f"Inserted emergency hospital preparation and immediate vector spraying for {zone}."
            new_stage = {
                "stage_id": remaining_stages[0]["stage_id"],
                "concurrent_actions_count": 2,
                "total_teams_required": 1,
                "total_kits_required": 50,
                "total_cost_inr": 20500.0,
                "actions": [
                    catalog["hospital_preparation"].to_dict(),
                    catalog["spraying"].to_dict()
                ]
            }
            # Shift remaining stages down
            repaired = [new_stage]
            for s in remaining_stages:
                s_copy = dict(s)
                s_copy["stage_id"] += 1
                repaired.append(s_copy)
            return remedy, reason, repaired

        elif event_type == "heavy_rainfall":
            reason = f"Heavy monsoon rainfall (85mm) washes out chemical insecticides, rendering outdoor fogging ineffective."
            remedy = f"Suspended outdoor adulticide spraying; redirected teams to indoor source reduction and water drainage clearance."
            repaired = []
            for s in remaining_stages:
                filtered_actions = []
                for act in s["actions"]:
                    if "chemical_spraying" in act["name"]:
                        # Replace spraying with indoor source reduction
                        filtered_actions.append(catalog["mosquito_source_reduction"].to_dict())
                    else:
                        filtered_actions.append(act)
                s_copy = dict(s)
                s_copy["actions"] = filtered_actions
                repaired.append(s_copy)
            return remedy, reason, repaired

        elif event_type == "team_unavailable":
            reason = f"Inspection team vehicle breakdown in {zone} reduced available workforce."
            remedy = f"Sequentialized parallel inspection tasks across consecutive stages to match available squads."
            repaired = []
            for s in remaining_stages:
                s_copy = dict(s)
                # Cap teams at 1 per stage
                if s_copy["total_teams_required"] > 1:
                    s_copy["total_teams_required"] = 1
                repaired.append(s_copy)
            return remedy, reason, repaired

        elif event_type == "weak_efficacy":
            reason = f"Follow-up larval survey in {zone} shows Breteau index remains elevated (>20)."
            remedy = f"Escalated intervention: scheduled secondary combined larviciding and public awareness sweep."
            new_stage = {
                "stage_id": remaining_stages[-1]["stage_id"] + 1,
                "concurrent_actions_count": 2,
                "total_teams_required": 1,
                "total_kits_required": 0,
                "total_cost_inr": 9000.0,
                "actions": [
                    catalog["mosquito_source_reduction"].to_dict(),
                    catalog["public_awareness"].to_dict()
                ]
            }
            repaired = list(remaining_stages) + [new_stage]
            return remedy, reason, repaired

        return "No action taken", "Unknown event type", remaining_stages

    def save_replan_log(self, trace_result: Dict[str, Any], filename: str = "planning_replan_log.json") -> Path:
        """Saves execution monitoring and replanning logs to disk."""
        output_dir = get_outputs_dir() / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trace_result, f, indent=2)
        return file_path
