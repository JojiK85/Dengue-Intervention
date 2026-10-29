"""Search package: Spatial Graph, BFS, DFS, UCS, A*, and evaluators."""
from .heuristics import haversine_admissible_heuristic, zero_heuristic
from .algorithms import (
    SearchResult,
    breadth_first_search,
    depth_first_search,
    uniform_cost_search,
    a_star_search
)
from .graph_model import SpatialCityGraph
from .evaluator import SearchEvaluator

__all__ = [
    "SearchResult",
    "breadth_first_search",
    "depth_first_search",
    "uniform_cost_search",
    "a_star_search",
    "haversine_admissible_heuristic",
    "zero_heuristic",
    "SpatialCityGraph",
    "SearchEvaluator"
]
