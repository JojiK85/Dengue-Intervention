"""
heuristics.py - Admissible Heuristics for A* Search
===================================================
Author: Member 2 (Search & Geospatial Routing)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Provides admissible and consistent heuristic estimators for spatial pathfinding.
Theory:
  A heuristic h(n) is admissible if:
    h(n) <= h*(n) for all nodes n, where h*(n) is the true optimal cost to goal.
  A heuristic h(n) is consistent (monotonic) if:
    h(n) <= c(n, a, n') + h(n') for all transitions n -> n' with step cost c(n, a, n').

For travel time search on a road network:
  h(n) = haversine_distance(n, goal) / v_max
Because physical straight-line distance is always <= road distance, and
actual vehicle speed is always <= v_max (0.75 km/min = 45 km/h in urban traffic),
h(n) is provably admissible and consistent.
"""

from typing import Dict, Any, Tuple
import networkx as nx
from src.core.utils import haversine_distance_km, MAX_TRANSIT_SPEED_KM_MIN


def haversine_admissible_heuristic(
    graph: nx.Graph,
    node: str,
    goal: str,
    max_speed_km_min: float = MAX_TRANSIT_SPEED_KM_MIN
) -> float:
    """
    Computes lower bound on road travel time from node to goal in minutes.
    Admissibility guarantee: straight-line distance / max_speed <= true travel time.
    """
    if node == goal:
        return 0.0

    node_data = graph.nodes.get(node, {})
    goal_data = graph.nodes.get(goal, {})

    if "lat" not in node_data or "lat" not in goal_data:
        return 0.0

    lat1, lon1 = float(node_data["lat"]), float(node_data["lon"])
    lat2, lon2 = float(goal_data["lat"]), float(goal_data["lon"])

    dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
    return dist_km / max_speed_km_min


def zero_heuristic(graph: nx.Graph, node: str, goal: str) -> float:
    """Zero heuristic reduces A* search to Uniform-Cost Search."""
    return 0.0
