"""
Adaptive Dengue Surveillance and Intervention Coordinator
CS4L002 Artificial Intelligence - IIT Bhubaneswar
Consolidated Core Package
"""

from .city_state import CityState, ZoneSnapshot, PlanStep, RoutingResult, setup_logger, get_project_root, get_data_dir, get_outputs_dir
from .data_generator import generate_all_synthetic_data, ingest_opendengue_raw
from .module1_search import SearchResult, breadth_first_search, depth_first_search, uniform_cost_search, a_star_search, haversine_admissible_heuristic, SearchEvaluator
from .module1_csp import Constraint, CSP, CSPSolver, ConflictExplainer, AllocationResult, DengueResourceAllocator
from .module2_planning import ActionSchema, create_dengue_action_catalog, CPMTask, CriticalPathAnalyzer, InterventionPlanner, OnlineExecutionMonitor, generate_pddl_domain, generate_pddl_problem, export_pddl_files
from .module3_uncertainty import Factor, BayesNode, DengueBayesianNetwork, VariableElimination, ZoneRiskEstimator, compute_shannon_entropy, compute_confidence_score
from .coordinator import AdaptiveDengueCoordinator, generate_deterministic_sitrep
from .scenario_runner import ScenarioRunner
from .visualizer import generate_all_plots
