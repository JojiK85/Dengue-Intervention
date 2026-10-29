"""
algorithms.py - Explicit Classroom Search Algorithms (BFS, DFS, UCS, A*)
========================================================================
Author: Member 2 (Search & Geospatial Routing)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Explicit Python implementations of foundational graph search algorithms:
1. Breadth-First Search (BFS) - Explores shallowest nodes first; optimal for unit step cost.
2. Depth-First Search (DFS) - Explores deepest nodes first; useful for transmission pathway exploration.
3. Uniform-Cost Search (UCS) - Priority queue ordered by path cost g(n); optimal for arbitrary positive costs.
4. A* Search - Priority queue ordered by evaluation function f(n) = g(n) + h(n); optimally efficient.

Written explicitly with clear loops and comments for student viva explanation.
"""

import time
import heapq
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
import networkx as nx

from .heuristics import haversine_admissible_heuristic


@dataclass
class SearchResult:
    """Standardized result schema for all graph search algorithms."""
    algorithm: str
    path: List[str]
    cost: float
    nodes_expanded: int
    execution_time_us: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "path": self.path,
            "cost": round(self.cost, 3),
            "nodes_expanded": self.nodes_expanded,
            "execution_time_us": round(self.execution_time_us, 1)
        }


def breadth_first_search(
    graph: nx.Graph,
    start: str,
    goal: str,
    weight_key: str = "travel_time_min"
) -> SearchResult:
    """
    Breadth-First Search (BFS) using a FIFO queue.
    Guarantees the minimum number of hops/edges to the goal node.
    """
    t0 = time.perf_counter()
    if start not in graph or goal not in graph:
        return SearchResult("BFS", [], float("inf"), 0, (time.perf_counter() - t0) * 1e6)

    if start == goal:
        return SearchResult("BFS", [start], 0.0, 1, (time.perf_counter() - t0) * 1e6)

    # Queue stores: (current_node, [path_so_far], cumulative_cost)
    queue = deque([(start, [start], 0.0)])
    visited: Set[str] = {start}
    nodes_expanded = 0

    while queue:
        current, path, cost = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            return SearchResult("BFS", path, cost, nodes_expanded, elapsed_us)

        for neighbor in sorted(graph.neighbors(current)):
            if neighbor not in visited:
                visited.add(neighbor)
                edge_weight = float(graph[current][neighbor].get(weight_key, 1.0))
                queue.append((neighbor, path + [neighbor], cost + edge_weight))

    elapsed_us = (time.perf_counter() - t0) * 1e6
    return SearchResult("BFS", [], float("inf"), nodes_expanded, elapsed_us)


def depth_first_search(
    graph: nx.Graph,
    start: str,
    goal: str,
    weight_key: str = "travel_time_min"
) -> SearchResult:
    """
    Depth-First Search (DFS) using an explicit LIFO stack.
    Explores along each branch as deep as possible before backtracking.
    """
    t0 = time.perf_counter()
    if start not in graph or goal not in graph:
        return SearchResult("DFS", [], float("inf"), 0, (time.perf_counter() - t0) * 1e6)

    if start == goal:
        return SearchResult("DFS", [start], 0.0, 1, (time.perf_counter() - t0) * 1e6)

    # Stack stores: (current_node, [path_so_far], cumulative_cost)
    stack: List[Tuple[str, List[str], float]] = [(start, [start], 0.0)]
    visited: Set[str] = set()
    nodes_expanded = 0

    while stack:
        current, path, cost = stack.pop()
        nodes_expanded += 1

        if current == goal:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            return SearchResult("DFS", path, cost, nodes_expanded, elapsed_us)

        if current not in visited:
            visited.add(current)
            # Push in reverse sorted order so alphabetical order is popped first
            for neighbor in sorted(graph.neighbors(current), reverse=True):
                if neighbor not in visited and neighbor not in path:
                    edge_weight = float(graph[current][neighbor].get(weight_key, 1.0))
                    stack.append((neighbor, path + [neighbor], cost + edge_weight))

    elapsed_us = (time.perf_counter() - t0) * 1e6
    return SearchResult("DFS", [], float("inf"), nodes_expanded, elapsed_us)


def uniform_cost_search(
    graph: nx.Graph,
    start: str,
    goal: str,
    weight_key: str = "travel_time_min"
) -> SearchResult:
    """
    Uniform-Cost Search (UCS) / Dijkstra's algorithm.
    Uses a priority queue ordered by path cost g(n).
    Guarantees finding the path with minimum total cost for non-negative weights.
    """
    t0 = time.perf_counter()
    if start not in graph or goal not in graph:
        return SearchResult("UCS", [], float("inf"), 0, (time.perf_counter() - t0) * 1e6)

    if start == goal:
        return SearchResult("UCS", [start], 0.0, 1, (time.perf_counter() - t0) * 1e6)

    # Priority queue stores: (g_cost, tiebreaker, current_node, path)
    counter = 0
    pq: List[Tuple[float, int, str, List[str]]] = [(0.0, counter, start, [start])]
    best_costs: Dict[str, float] = {start: 0.0}
    nodes_expanded = 0

    while pq:
        g_cost, _, current, path = heapq.heappop(pq)
        nodes_expanded += 1

        if current == goal:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            return SearchResult("UCS", path, g_cost, nodes_expanded, elapsed_us)

        if g_cost > best_costs.get(current, float("inf")):
            continue

        for neighbor in sorted(graph.neighbors(current)):
            edge_weight = float(graph[current][neighbor].get(weight_key, 1.0))
            new_cost = g_cost + edge_weight

            if new_cost < best_costs.get(neighbor, float("inf")):
                best_costs[neighbor] = new_cost
                counter += 1
                heapq.heappush(pq, (new_cost, counter, neighbor, path + [neighbor]))

    elapsed_us = (time.perf_counter() - t0) * 1e6
    return SearchResult("UCS", [], float("inf"), nodes_expanded, elapsed_us)


def a_star_search(
    graph: nx.Graph,
    start: str,
    goal: str,
    heuristic_fn: Optional[Callable[[nx.Graph, str, str], float]] = None,
    weight_key: str = "travel_time_min"
) -> SearchResult:
    """
    A* Search using evaluation function f(n) = g(n) + h(n).
    When h(n) is admissible and consistent, A* is provably optimal and expands
    the minimal number of nodes necessary.
    """
    t0 = time.perf_counter()
    if heuristic_fn is None:
        heuristic_fn = haversine_admissible_heuristic

    if start not in graph or goal not in graph:
        return SearchResult("A*", [], float("inf"), 0, (time.perf_counter() - t0) * 1e6)

    if start == goal:
        return SearchResult("A*", [start], 0.0, 1, (time.perf_counter() - t0) * 1e6)

    counter = 0
    h_start = heuristic_fn(graph, start, goal)
    # PQ stores: (f_score, h_score, tiebreaker, current_node, g_cost, path)
    pq: List[Tuple[float, float, int, str, float, List[str]]] = [
        (h_start, h_start, counter, start, 0.0, [start])
    ]
    best_g_costs: Dict[str, float] = {start: 0.0}
    nodes_expanded = 0

    while pq:
        f_cost, _, _, current, g_cost, path = heapq.heappop(pq)
        nodes_expanded += 1

        if current == goal:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            return SearchResult("A*", path, g_cost, nodes_expanded, elapsed_us)

        if g_cost > best_g_costs.get(current, float("inf")):
            continue

        for neighbor in sorted(graph.neighbors(current)):
            edge_weight = float(graph[current][neighbor].get(weight_key, 1.0))
            new_g = g_cost + edge_weight

            if new_g < best_g_costs.get(neighbor, float("inf")):
                best_g_costs[neighbor] = new_g
                h_val = heuristic_fn(graph, neighbor, goal)
                new_f = new_g + h_val
                counter += 1
                heapq.heappush(pq, (new_f, h_val, counter, neighbor, new_g, path + [neighbor]))

    elapsed_us = (time.perf_counter() - t0) * 1e6
    return SearchResult("A*", [], float("inf"), nodes_expanded, elapsed_us)
