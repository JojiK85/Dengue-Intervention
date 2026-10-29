"""
main.py - Master Command-Line Interface (CLI) Demonstrator
==========================================================
Author: Group 9 (Undergraduate Project Team, IIT Bhubaneswar)
Course: CS4L002 Artificial Intelligence (Semester Project)

Master CLI executing:
1. Synthetic data generation and digital-twin initialization.
2. Search algorithm benchmarks (BFS, DFS, UCS, A*).
3. CSP resource allocation and shortage diagnostics.
4. Classical planning, concurrent stages, and Critical Path Method (CPM).
5. Exact Bayesian inference, under-reporting detection, and Shannon entropy.
6. G-SEIR simulation, MDP value iteration, Q-Learning, and SARSA.
7. Execution of all 8 mandatory public-health stress scenarios.
8. Generation of publication-grade Matplotlib visualizations.
"""

import sys
import argparse
from pathlib import Path

# Ensure final_project is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from data.generator import generate_all_synthetic_data
from src.core.city_state import CityState
from src.core.utils import setup_logger, get_outputs_dir
from src.search.evaluator import SearchEvaluator
from src.csp.resource_allocator import DengueResourceAllocator
from src.planning.planner import InterventionPlanner
from src.planning.execution_monitor import OnlineExecutionMonitor
from src.uncertainty.risk_estimator import ZoneRiskEstimator
from src.rl.baseline_comparison import run_all_rl_experiments
from src.coordinator.system_coordinator import AdaptiveDengueCoordinator
from src.scenarios.scenario_runner import ScenarioRunner
from src.visualizer.plotter import generate_all_plots

logger = setup_logger("master_main")


def print_banner():
    banner = """
================================================================================
   ADAPTIVE DENGUE SURVEILLANCE & INTERVENTION COORDINATOR (GROUP 9)
   Course: CS4L002 Artificial Intelligence | IIT Bhubaneswar
   Core Modules: Search (1), CSP (1), Planning (2), Bayes Net (3), MDP/RL (4)
================================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Dengue Surveillance Coordinator Master Runner")
    parser.add_argument("--days", type=int, default=30, help="Number of longitudinal simulation days (default: 30)")
    parser.add_argument("--scenarios", action="store_true", help="Execute all 8 mandatory public health stress scenarios")
    parser.add_argument("--benchmark", action="store_true", help="Execute complete algorithm benchmarks for Search and RL")
    parser.add_argument("--plots", action="store_true", help="Generate all publication-grade Matplotlib plots")
    parser.add_argument("--all", action="store_true", help="Execute full pipeline: 30 days + 8 scenarios + benchmarks + plots")

    args = parser.parse_args()
    print_banner()

    # Step 1: Ensure synthetic data exists
    logger.info("[Step 1/6] Verifying and generating synthetic municipal surveillance data...")
    generate_all_synthetic_data()

    # Step 2: Initialize Digital Twin
    logger.info("[Step 2/6] Initializing CityState Digital-Twin and Coordinator...")
    city = CityState()
    coordinator = AdaptiveDengueCoordinator(city)

    # Step 3: Run Longitudinal Simulation Loop
    run_days = args.days if not args.all else 30
    logger.info(f"[Step 3/6] Running Longitudinal Municipal Simulation for {run_days} Days...")
    history = coordinator.run_simulation_pipeline(num_days=run_days)
    logger.info(f"Day {run_days} reached. Latest status: {city.get_summary_snapshot()}")

    # Step 4: Run 8 Mandatory Public Health Scenarios
    if args.scenarios or args.all:
        logger.info("[Step 4/6] Executing 8 Mandatory Public-Health Stress Scenarios...")
        runner = ScenarioRunner(coordinator)
        scenarios_df = runner.run_all_scenarios()
        print("\n--- Scenario Evaluation Summary ---")
        print(scenarios_df[["scenario_id", "name", "system_status", "key_metric"]].to_string(index=False))

    # Step 5: Run Detailed Algorithmic Benchmarks
    if args.benchmark or args.all:
        logger.info("[Step 5/6] Running Search and RL Comparative Benchmarks...")
        search_eval = SearchEvaluator(city)
        search_df = search_eval.run_benchmark_comparison()

        risk_eval = ZoneRiskEstimator()
        risk_eval.demonstrate_dynamic_belief_updating()

        rl_df = run_all_rl_experiments()
        print("\n--- RL Policy Comparison ---")
        print(rl_df.to_string(index=False))

    # Step 6: Generate Publication-Grade Visualizations
    if args.plots or args.all:
        logger.info("[Step 6/6] Generating Publication-Grade Visualizations (Pure Matplotlib)...")
        plots = generate_all_plots(city)
        logger.info(f"Generated {len(plots)} visualization plots in outputs/plots/")

    logger.info("Master Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    # If no flags provided, run full integrated demonstration by default
    if len(sys.argv) == 1:
        sys.argv.append("--all")
    main()
