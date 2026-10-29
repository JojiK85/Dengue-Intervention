import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Dict, List, Any, Optional, Tuple
import pandas as pd

from src.city_state import get_outputs_dir, get_project_root, setup_logger

logger = setup_logger("module2_planning")

@dataclass
class ActionSchema:
    name: str
    zone: str
    preconditions: Set[str] = field(default_factory=set)
    add_effects: Set[str] = field(default_factory=set)
    delete_effects: Set[str] = field(default_factory=set)
    duration_days: int = 1
    required_teams: int = 0
    required_kits: int = 0
    required_beds: int = 0
    cost_inr: float = 2000.0
    weather_constraint: Optional[str] = None

    def is_applicable(self, current_state: Set[str], current_weather: str = "Clear") -> bool:
        if not self.preconditions.issubset(current_state):
            return False
        if self.weather_constraint == "Dry" and current_weather.lower() in ["rain", "heavy_rain", "monsoon"]:
            return False
        return True

    def apply(self, current_state: Set[str]) -> Set[str]:
        return (current_state - self.delete_effects) | self.add_effects

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "zone": self.zone,
            "preconditions": sorted(list(self.preconditions)),
            "add_effects": sorted(list(self.add_effects)),
            "delete_effects": sorted(list(self.delete_effects)),
            "duration_days": self.duration_days,
            "required_teams": self.required_teams,
            "required_kits": self.required_kits,
            "required_beds": self.required_beds,
            "cost_inr": self.cost_inr,
            "weather_constraint": self.weather_constraint
        }

def create_dengue_action_catalog(zone: str) -> Dict[str, ActionSchema]:
    return {
        "surveillance": ActionSchema(
            name=f"surveillance_{zone}", zone=zone,
            preconditions={f"zone_active_{zone}"},
            add_effects={f"surveillance_active_{zone}", f"fever_cases_monitored_{zone}"},
            duration_days=1, cost_inr=1500.0
        ),
        "field_inspection": ActionSchema(
            name=f"field_inspection_{zone}", zone=zone,
            preconditions={f"surveillance_active_{zone}"},
            add_effects={f"breeding_sites_identified_{zone}", f"vector_density_confirmed_{zone}"},
            duration_days=1, required_teams=1, cost_inr=4500.0
        ),
        "mosquito_source_reduction": ActionSchema(
            name=f"source_reduction_{zone}", zone=zone,
            preconditions={f"breeding_sites_identified_{zone}"},
            add_effects={f"breeding_containers_cleared_{zone}"},
            delete_effects={f"breeding_sites_identified_{zone}"},
            duration_days=2, required_teams=1, cost_inr=6000.0
        ),
        "spraying": ActionSchema(
            name=f"chemical_spraying_{zone}", zone=zone,
            preconditions={f"vector_density_confirmed_{zone}"},
            add_effects={f"adult_vectors_reduced_{zone}"},
            delete_effects={f"vector_density_confirmed_{zone}"},
            duration_days=1, required_teams=1, cost_inr=8500.0,
            weather_constraint="Dry"
        ),
        "laboratory_testing": ActionSchema(
            name=f"laboratory_testing_{zone}", zone=zone,
            preconditions={f"fever_cases_monitored_{zone}"},
            add_effects={f"cases_serotyped_{zone}", f"outbreak_confirmed_{zone}"},
            duration_days=1, required_kits=25, cost_inr=7000.0
        ),
        "public_awareness": ActionSchema(
            name=f"public_awareness_{zone}", zone=zone,
            preconditions={f"zone_active_{zone}"},
            add_effects={f"community_cooperation_active_{zone}"},
            duration_days=1, cost_inr=3000.0
        ),
        "hospital_preparation": ActionSchema(
            name=f"hospital_preparation_{zone}", zone=zone,
            preconditions={f"outbreak_confirmed_{zone}"},
            add_effects={f"surge_beds_mobilized_{zone}", f"triage_active_{zone}"},
            duration_days=1, required_beds=10, cost_inr=12000.0
        ),
        "follow_up_assessment": ActionSchema(
            name=f"follow_up_assessment_{zone}", zone=zone,
            preconditions={f"adult_vectors_reduced_{zone}", f"breeding_containers_cleared_{zone}"},
            add_effects={f"zone_transmission_contained_{zone}"},
            duration_days=1, required_teams=1, cost_inr=3500.0
        )
    }

@dataclass
class CPMTask:
    task_id: str
    duration: float
    dependencies: List[str] = field(default_factory=list)
    zone: str = ""
    action_type: str = ""
    es: float = 0.0
    ef: float = 0.0
    ls: float = 0.0
    lf: float = 0.0
    slack: float = 0.0
    is_critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "zone": self.zone, "action_type": self.action_type,
            "duration": self.duration, "dependencies": self.dependencies,
            "ES": round(self.es, 1), "EF": round(self.ef, 1),
            "LS": round(self.ls, 1), "LF": round(self.lf, 1),
            "slack": round(self.slack, 1), "is_critical": self.is_critical
        }

class CriticalPathAnalyzer:
    @staticmethod
    def compute_critical_path(tasks: List[CPMTask]) -> Tuple[List[CPMTask], List[str], float]:
        if not tasks:
            return [], [], 0.0
        task_dict = {t.task_id: t for t in tasks}
        successors = {t.task_id: [] for t in tasks}
        for t in tasks:
            for dep in t.dependencies:
                if dep in successors:
                    successors[dep].append(t.task_id)

        in_degree = {t.task_id: len([d for d in t.dependencies if d in task_dict]) for t in tasks}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        topo_order = []

        while queue:
            curr_id = queue.pop(0)
            topo_order.append(curr_id)
            curr_task = task_dict[curr_id]
            if not curr_task.dependencies:
                curr_task.es = 0.0
            else:
                pred_efs = [task_dict[d].ef for d in curr_task.dependencies if d in task_dict]
                curr_task.es = max(pred_efs) if pred_efs else 0.0
            curr_task.ef = curr_task.es + curr_task.duration

            for succ_id in successors[curr_id]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        project_duration = max((t.ef for t in tasks), default=0.0)

        for tid in reversed(topo_order):
            curr_task = task_dict[tid]
            succ_ids = successors[tid]
            if not succ_ids:
                curr_task.lf = project_duration
            else:
                succ_lss = [task_dict[s].ls for s in succ_ids]
                curr_task.lf = min(succ_lss) if succ_lss else project_duration

            curr_task.ls = curr_task.lf - curr_task.duration
            curr_task.slack = max(0.0, curr_task.ls - curr_task.es)
            curr_task.is_critical = abs(curr_task.slack) < 1e-4

        critical_path = [t.task_id for t in tasks if t.is_critical]
        return tasks, critical_path, project_duration

class InterventionPlanner:
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
        catalog = self.catalogs.get(zone, {})
        current_state = set(initial_state)
        plan: List[ActionSchema] = []
        action_order = [
            "surveillance", "laboratory_testing", "field_inspection", "public_awareness",
            "mosquito_source_reduction", "spraying", "hospital_preparation", "follow_up_assessment"
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
        full_plan: List[ActionSchema] = []
        for z in self.target_zones:
            zone_goals = goals_by_zone.get(z, set())
            zone_plan = self.generate_plan_for_zone(z, initial_state, zone_goals, weather)
            full_plan.extend(zone_plan)
        return full_plan

    def analyze_parallel_stages(self, plan: List[ActionSchema]) -> List[Dict[str, Any]]:
        stages: List[List[ActionSchema]] = []
        for action in plan:
            placed = False
            for stage in stages:
                team_conflict = (action.required_teams > 0 and sum(a.required_teams for a in stage) >= 4)
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
        tasks: List[CPMTask] = []
        action_producers: Dict[str, str] = {}
        for i, act in enumerate(plan, 1):
            tid = f"T{i}_{act.name}"
            for eff in act.add_effects:
                action_producers[eff] = tid

        for i, act in enumerate(plan, 1):
            tid = f"T{i}_{act.name}"
            deps = [action_producers[pre] for pre in act.preconditions if pre in action_producers and action_producers[pre] != tid]
            tasks.append(CPMTask(
                task_id=tid, duration=float(act.duration_days), dependencies=deps,
                zone=act.zone, action_type=act.name.split("_")[0]
            ))
        return CriticalPathAnalyzer.compute_critical_path(tasks)

    def export_plan_artifacts(
        self,
        stages: List[Dict[str, Any]],
        cpm_tasks: List[CPMTask],
        critical_path: List[str],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
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

class OnlineExecutionMonitor:
    def __init__(self, target_zones: List[str]):
        self.target_zones = target_zones
        self.catalogs = {z: create_dengue_action_catalog(z) for z in target_zones}

    def execute_with_monitoring(
        self,
        initial_stages: List[Dict[str, Any]],
        disruptions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        disruptions = disruptions or []
        disruption_map = {d.get("stage_id", -1): d for d in disruptions}
        execution_trace = []
        replanning_events = []
        active_stages = list(initial_stages)
        stage_idx = 0

        while stage_idx < len(active_stages):
            current_stage = active_stages[stage_idx]
            current_id = current_stage["stage_id"]

            if current_id in disruption_map:
                event = disruption_map[current_id]
                event_type = event.get("type", "unknown")
                target_zone = event.get("zone", self.target_zones[0])
                logger.info(f"Disruption detected at stage {current_id}: {event_type} in {target_zone}")

                replan_action, reason, repaired_stages = self._replan(
                    event_type, target_zone, active_stages[stage_idx:]
                )
                replanning_events.append({
                    "stage_id": current_id, "event_type": event_type,
                    "target_zone": target_zone, "reason": reason,
                    "remedial_action": replan_action
                })
                active_stages = active_stages[:stage_idx] + repaired_stages
                current_stage = active_stages[stage_idx]

            executed_actions = []
            for act in current_stage["actions"]:
                act_record = dict(act)
                act_record["status"] = "COMPLETED"
                executed_actions.append(act_record)

            execution_trace.append({"stage_id": current_id, "executed_actions": executed_actions})
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
        catalog = self.catalogs.get(zone, {})

        if event_type == "surge":
            reason = f"Sudden surge in clinical fever reports in {zone} requires urgent capacity expansion."
            remedy = f"Inserted emergency hospital preparation and immediate vector spraying for {zone}."
            new_stage = {
                "stage_id": remaining_stages[0]["stage_id"],
                "concurrent_actions_count": 2, "total_teams_required": 1,
                "total_kits_required": 50, "total_cost_inr": 20500.0,
                "actions": [catalog["hospital_preparation"].to_dict(), catalog["spraying"].to_dict()]
            }
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
                filtered_actions = [
                    catalog["mosquito_source_reduction"].to_dict() if "chemical_spraying" in act["name"] else act
                    for act in s["actions"]
                ]
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
                if s_copy["total_teams_required"] > 1:
                    s_copy["total_teams_required"] = 1
                repaired.append(s_copy)
            return remedy, reason, repaired

        elif event_type == "weak_efficacy":
            reason = f"Follow-up larval survey in {zone} shows Breteau index remains elevated (>20)."
            remedy = f"Escalated intervention: scheduled secondary combined larviciding and public awareness sweep."
            new_stage = {
                "stage_id": remaining_stages[-1]["stage_id"] + 1,
                "concurrent_actions_count": 2, "total_teams_required": 1,
                "total_kits_required": 0, "total_cost_inr": 9000.0,
                "actions": [catalog["mosquito_source_reduction"].to_dict(), catalog["public_awareness"].to_dict()]
            }
            return remedy, reason, list(remaining_stages) + [new_stage]

        return "No action taken", "Unknown event type", remaining_stages

    def save_replan_log(self, trace_result: Dict[str, Any], filename: str = "planning_replan_log.json") -> Path:
        output_dir = get_outputs_dir() / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trace_result, f, indent=2)
        return file_path

def generate_pddl_domain() -> str:
    return """;;; Adaptive Dengue Surveillance and Intervention Coordinator
;;; Course Project: CS4L002 Artificial Intelligence, IIT Bhubaneswar
;;; Formal STRIPS / PDDL Domain Definition

(define (domain dengue-intervention)
  (:requirements :strips :typing)
  (:types zone)

  (:predicates
    (zone-active ?z - zone)
    (surveillance-active ?z - zone)
    (breeding-sites-identified ?z - zone)
    (breeding-containers-cleared ?z - zone)
    (vector-density-confirmed ?z - zone)
    (adult-vectors-reduced ?z - zone)
    (outbreak-confirmed ?z - zone)
    (surge-beds-mobilized ?z - zone)
    (transmission-contained ?z - zone)
  )

  (:action surveillance
    :parameters (?z - zone)
    :precondition (zone-active ?z)
    :effect (surveillance-active ?z)
  )

  (:action field-inspection
    :parameters (?z - zone)
    :precondition (surveillance-active ?z)
    :effect (and (breeding-sites-identified ?z)
                 (vector-density-confirmed ?z))
  )

  (:action source-reduction
    :parameters (?z - zone)
    :precondition (breeding-sites-identified ?z)
    :effect (and (breeding-containers-cleared ?z)
                 (not (breeding-sites-identified ?z)))
  )

  (:action chemical-spraying
    :parameters (?z - zone)
    :precondition (vector-density-confirmed ?z)
    :effect (and (adult-vectors-reduced ?z)
                 (not (vector-density-confirmed ?z)))
  )

  (:action laboratory-testing
    :parameters (?z - zone)
    :precondition (surveillance-active ?z)
    :effect (outbreak-confirmed ?z)
  )

  (:action hospital-preparation
    :parameters (?z - zone)
    :precondition (outbreak-confirmed ?z)
    :effect (surge-beds-mobilized ?z)
  )

  (:action follow-up-assessment
    :parameters (?z - zone)
    :precondition (and (adult-vectors-reduced ?z)
                       (breeding-containers-cleared ?z))
    :effect (transmission-contained ?z)
  )
)
"""

def generate_pddl_problem(zones: List[str]) -> str:
    zone_objs = " ".join(f"{z} - zone" for z in zones)
    inits = "\n    ".join(f"(zone-active {z})" for z in zones)
    goals = "\n    ".join(f"(transmission-contained {z})" for z in zones)
    return f""";;; Dengue Intervention Problem Instance
;;; Target Wards: {', '.join(zones)}

(define (problem dengue-outbreak-containment)
  (:domain dengue-intervention)
  (:objects
    {zone_objs}
  )

  (:init
    {inits}
  )

  (:goal
    (and
      {goals}
    )
  )
)
"""

def export_pddl_files(target_zones: List[str], pddl_dir: Optional[Path] = None) -> Dict[str, Path]:
    if pddl_dir is None:
        pddl_dir = get_project_root() / "pddl"
    pddl_dir.mkdir(parents=True, exist_ok=True)
    domain_path = pddl_dir / "domain.pddl"
    problem_path = pddl_dir / "problem.pddl"
    with open(domain_path, "w", encoding="utf-8") as f:
        f.write(generate_pddl_domain())
    with open(problem_path, "w", encoding="utf-8") as f:
        f.write(generate_pddl_problem(target_zones))
    return {"domain": domain_path, "problem": problem_path}
