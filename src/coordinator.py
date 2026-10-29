import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.city_state import CityState, PlanStep, RoutingResult, get_outputs_dir, setup_logger
from src.module1_search import SearchEvaluator
from src.module1_csp import DengueResourceAllocator, AllocationResult
from src.module2_planning import InterventionPlanner, OnlineExecutionMonitor
from src.module3_uncertainty import ZoneRiskEstimator
from src.module4_rl import DengueSimulationEnv, TabularQLearningAgent

logger = setup_logger("coordinator")

def generate_deterministic_sitrep(
    city_state: CityState,
    day: int,
    csp_status: str,
    critical_path: List[str]
) -> str:
    summary = city_state.get_summary_snapshot()
    high_risk = summary["high_risk_zones"]
    underreported = summary["underreporting_zones"]

    sitrep_lines = [
        f"# MUNICIPAL DENGUE SITUATION REPORT (SITREP) - DAY {day:02d}",
        f"**Jurisdiction:** Urban Municipal Corporation | **Surveillance Cycle:** Longitudinal 30-Day Simulation",
        f"**Deterministic Decision-Support Engine:** Modules 1-4 (Search, CSP, Planning, Bayes Net, MDP/RL)",
        "---",
        "### 1. Epidemiological Summary",
        f"- **Total Clinically Reported Cases:** {summary['total_confirmed_cases']}",
        f"- **Estimated True Infection Burden:** {summary['total_estimated_true_cases']} (accounting for under-reporting multipliers)",
        f"- **Hospital Surge Bed Occupancy:** {summary['total_beds_occupied']} occupied",
        f"- **Diagnostic Test Kits Remaining:** {summary['total_diagnostic_kits_remaining']}",
        f"- **High-Risk Wards (Alert ORANGE/RED):** {', '.join(high_risk) if high_risk else 'None'}",
        f"- **Latent Under-Reporting Alerts:** {', '.join(underreported) if underreported else 'None'}",
        "",
        "### 2. Algorithmic Decisions & Resource Directives",
        f"- **Module 3 (Bayesian Network):** Exact Variable Elimination inferred true outbreak risks; detected {len(underreported)} under-reported sectors.",
        f"- **Module 1 (CSP Allocator):** Allocation status: **{csp_status}**. Enforced strict team travel times and high-risk coverage constraints.",
        f"- **Module 2 (Intervention Planning):** Critical path activities governing schedule: `{', '.join(critical_path[:3])}`.",
        "",
        "### 3. Public Health Guidance & Preventative Advisories",
    ]

    for zid in high_risk:
        snap = city_state.zone_snapshots[zid]
        sitrep_lines.extend([
            f"**Ward {zid} (Alert {snap.alert_level}, True Risk {snap.risk_score:.2f}, Confidence {snap.confidence_score:.2f}):**",
            f"  - Vector Control Directive: Immediate adulticide spraying and source reduction authorized.",
            f"  - Clinical Guidance: Mobilize triage beds at nearest referral facility; distribute diagnostic kits.",
            f"  - Citizen Advisory: Empty stagnant water coolers and overhead containers within 48 hours."
        ])

    if underreported:
        sitrep_lines.append("")
        sitrep_lines.append("**SPECIAL SURVEILLANCE DIRECTIVE (Under-Reporting Hotspots):**")
        for zid in underreported:
            sitrep_lines.append(
                f"  - Ward {zid} exhibits low clinic reporting despite high Breteau larval index. "
                f"Active door-to-door fever screening squads dispatched to prevent unobserved contagion."
            )

    sitrep_lines.append("---")
    sitrep_lines.append("*Report generated deterministically by the Adaptive Dengue Surveillance Coordinator.*")
    return "\n".join(sitrep_lines)

class AdaptiveDengueCoordinator:
    def __init__(self, city_state: Optional[CityState] = None):
        self.city_state = city_state or CityState()
        self.risk_estimator = ZoneRiskEstimator()
        self.search_evaluator = SearchEvaluator(self.city_state)
        self.csp_allocator = DengueResourceAllocator(self.city_state)

        target_zones = self.city_state.get_zone_ids()
        self.planner = InterventionPlanner(target_zones)
        self.monitor = OnlineExecutionMonitor(target_zones)

        self.rl_env = DengueSimulationEnv(num_zones=len(target_zones), horizon_steps=30, seed=42)
        self.rl_agent = TabularQLearningAgent(seed=42)
        self.rl_agent.train(self.rl_env, num_episodes=50)

    def run_day(
        self,
        day: int,
        weather_override: Optional[Dict[str, float]] = None,
        resource_override: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        self.city_state.current_day = day
        logger.info(f"=== Commencing Decision Pipeline for Day {day:02d} ===")

        if not self.city_state.weather_data.empty:
            day_weather = self.city_state.weather_data[self.city_state.weather_data["day"] == day]
            for _, row in day_weather.iterrows():
                zid = str(row["zone_id"])
                if zid in self.city_state.zone_snapshots:
                    snap = self.city_state.zone_snapshots[zid]
                    snap.rainfall_mm = float(row.get("rainfall_mm", 10.0))
                    snap.temp_c = float(row.get("temperature_c", 29.0))
                    snap.humidity_pct = float(row.get("humidity_pct", 72.0))
                    snap.breteau_index = float(row.get("breteau_index", 15.0))
                    snap.confirmed_cases = float(row.get("daily_reported_cases", 5.0))

        if weather_override:
            for snap in self.city_state.zone_snapshots.values():
                if "rainfall_mm" in weather_override:
                    snap.rainfall_mm = weather_override["rainfall_mm"]
                if "temp_c" in weather_override:
                    snap.temp_c = weather_override["temp_c"]

        self.risk_estimator.evaluate_city_surveillance(self.city_state)
        zone_risks = {zid: snap.alert_level for zid, snap in self.city_state.zone_snapshots.items()}

        avail_teams = resource_override.get("teams", 5) if resource_override else 5
        avail_kits = resource_override.get("kits", 350) if resource_override else 350
        avail_beds = resource_override.get("beds", 80) if resource_override else 80

        csp_result: AllocationResult = self.csp_allocator.allocate_resources(
            zone_risks=zone_risks,
            available_teams=avail_teams,
            total_kits=avail_kits,
            total_beds=avail_beds
        )

        for zid, alloc in csp_result.allocation.items():
            if zid in self.city_state.zone_snapshots:
                self.city_state.zone_snapshots[zid].teams_active = alloc.get("teams", 0)
                self.city_state.zone_snapshots[zid].kits_remaining = alloc.get("kits", 0)

        routing_actions = []
        for zid, snap in self.city_state.zone_snapshots.items():
            if snap.alert_level in ["RED", "ORANGE"]:
                route_info = self.search_evaluator.route_patient_to_best_hospital(zid, min_beds_required=2)
                routing_actions.append(route_info)

        goals_by_zone = {}
        for zid, snap in self.city_state.zone_snapshots.items():
            if snap.alert_level in ["RED", "ORANGE"]:
                goals_by_zone[zid] = {
                    f"surveillance_active_{zid}",
                    f"breeding_containers_cleared_{zid}",
                    f"adult_vectors_reduced_{zid}",
                    f"surge_beds_mobilized_{zid}"
                }
            else:
                goals_by_zone[zid] = {f"surveillance_active_{zid}"}

        initial_state = {f"zone_active_{zid}" for zid in self.city_state.get_zone_ids()}
        current_weather = "Rain" if any(s.rainfall_mm >= 50.0 for s in self.city_state.zone_snapshots.values()) else "Clear"

        multi_zone_plan = self.planner.build_multi_zone_plan(initial_state, goals_by_zone, current_weather)
        stages = self.planner.analyze_parallel_stages(multi_zone_plan)
        cpm_tasks, critical_path, project_duration = self.planner.build_cpm_schedule(multi_zone_plan)

        current_rl_state = self.rl_env.get_discrete_state()
        rl_action = self.rl_agent.select_action(current_rl_state, evaluate=True)
        next_rl_state, reward, done, rl_info = self.rl_env.step(rl_action)

        for snap in self.city_state.zone_snapshots.values():
            snap.rl_action = DengueSimulationEnv.ACTION_NAMES[rl_action]
            snap.rl_reward = round(reward, 2)

        sitrep_text = generate_deterministic_sitrep(self.city_state, day, csp_result.status, critical_path)

        daily_summary = {
            "day": day,
            "city_summary": self.city_state.get_summary_snapshot(),
            "csp_status": csp_result.status,
            "csp_is_relaxed": csp_result.is_relaxed,
            "csp_conflicts": csp_result.conflicts,
            "routing_actions": routing_actions,
            "planning_stages_count": len(stages),
            "critical_path": critical_path,
            "project_duration_days": project_duration,
            "rl_action_selected": DengueSimulationEnv.ACTION_NAMES[rl_action],
            "rl_step_reward": round(reward, 2),
            "sitrep": sitrep_text
        }

        log_dir = get_outputs_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"city_state_day{day:02d}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(daily_summary, f, indent=2)

        return daily_summary

    def run_simulation_pipeline(self, num_days: int = 30) -> List[Dict[str, Any]]:
        logger.info(f"Starting Longitudinal Municipal Simulation for {num_days} Days...")
        pipeline_history = []
        for d in range(1, num_days + 1):
            day_result = self.run_day(d)
            pipeline_history.append(day_result)
        logger.info("Longitudinal simulation complete.")
        return pipeline_history
