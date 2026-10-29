import sys
import argparse
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.city_state import CityState, setup_logger, get_outputs_dir
from src.data_generator import generate_all_synthetic_data
from src.module1_search import SearchEvaluator
from src.module1_csp import DengueResourceAllocator
from src.module2_planning import InterventionPlanner, OnlineExecutionMonitor
from src.module3_uncertainty import ZoneRiskEstimator
from src.module4_rl import run_all_rl_experiments
from src.coordinator import AdaptiveDengueCoordinator
from src.scenario_runner import ScenarioRunner
from src.visualizer import generate_all_plots

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

    logger.info("[Step 1/6] Verifying and generating synthetic municipal surveillance data...")
    generate_all_synthetic_data()

    logger.info("[Step 2/6] Initializing CityState Digital-Twin and Coordinator...")
    city = CityState()
    coordinator = AdaptiveDengueCoordinator(city)

    run_days = args.days if not args.all else 30
    logger.info(f"[Step 3/6] Running Longitudinal Municipal Simulation for {run_days} Days...")
    history = coordinator.run_simulation_pipeline(num_days=run_days)
    logger.info(f"Day {run_days} reached. Latest status: {city.get_summary_snapshot()}")

    if args.scenarios or args.all:
        logger.info("[Step 4/6] Executing 8 Mandatory Public-Health Stress Scenarios...")
        runner = ScenarioRunner(coordinator)
        scenarios_df = runner.run_all_scenarios()
        print("\n--- Scenario Evaluation Summary ---")
        print(scenarios_df[["scenario_id", "name", "system_status", "key_metric"]].to_string(index=False))

    if args.benchmark or args.all:
        logger.info("[Step 5/6] Running Search and RL Comparative Benchmarks...")
        search_eval = SearchEvaluator(city)
        search_df = search_eval.run_benchmark_comparison()

        risk_eval = ZoneRiskEstimator()
        risk_eval.demonstrate_dynamic_belief_updating()

        rl_df = run_all_rl_experiments()
        print("\n--- RL Policy Comparison ---")
        print(rl_df.to_string(index=False))

    if args.plots or args.all:
        logger.info("[Step 6/6] Generating Publication-Grade Visualizations (Pure Matplotlib)...")
        plots = generate_all_plots(city)
        logger.info(f"Generated {len(plots)} visualization plots in outputs/plots/")

    logger.info("Master Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--all")
    main()
