import time
import heapq
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
import networkx as nx
import pandas as pd

from src.city_state import CityState, RoutingResult, haversine_distance_km, MAX_TRANSIT_SPEED_KM_MIN, get_outputs_dir, setup_logger

logger = setup_logger("module1_search")

@dataclass
class SearchResult:
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

def haversine_admissible_heuristic(
    graph: nx.Graph,
    node: str,
    goal: str,
    max_speed_km_min: float = MAX_TRANSIT_SPEED_KM_MIN
) -> float:
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
    return 0.0

def breadth_first_search(
    graph: nx.Graph,
    start: str,
    goal: str,
    weight_key: str = "travel_time_min"
) -> SearchResult:
    t0 = time.perf_counter()
    if start not in graph or goal not in graph:
        return SearchResult("BFS", [], float("inf"), 0, (time.perf_counter() - t0) * 1e6)
    if start == goal:
        return SearchResult("BFS", [start], 0.0, 1, (time.perf_counter() - t0) * 1e6)

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
    t0 = time.perf_counter()
    if start not in graph or goal not in graph:
        return SearchResult("DFS", [], float("inf"), 0, (time.perf_counter() - t0) * 1e6)
    if start == goal:
        return SearchResult("DFS", [start], 0.0, 1, (time.perf_counter() - t0) * 1e6)

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
    t0 = time.perf_counter()
    if start not in graph or goal not in graph:
        return SearchResult("UCS", [], float("inf"), 0, (time.perf_counter() - t0) * 1e6)
    if start == goal:
        return SearchResult("UCS", [start], 0.0, 1, (time.perf_counter() - t0) * 1e6)

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
    t0 = time.perf_counter()
    if heuristic_fn is None:
        heuristic_fn = haversine_admissible_heuristic
    if start not in graph or goal not in graph:
        return SearchResult("A*", [], float("inf"), 0, (time.perf_counter() - t0) * 1e6)
    if start == goal:
        return SearchResult("A*", [start], 0.0, 1, (time.perf_counter() - t0) * 1e6)

    counter = 0
    h_start = heuristic_fn(graph, start, goal)
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

class SearchEvaluator:
    def __init__(self, city_state: CityState):
        self.city_state = city_state
        self.infra_graph = city_state.infra_graph
        self.mobility_graph = city_state.mobility_graph

    def route_patient_to_best_hospital(self, origin_zone: str, min_beds_required: int = 1) -> Dict[str, Any]:
        hospitals = []
        for n, d in self.infra_graph.nodes(data=True):
            if d.get("node_type") == "hospital":
                avail = int(d.get("available_beds", 0))
                hospitals.append((n, avail))

        candidates = [h for h, avail in hospitals if avail >= min_beds_required]
        if not candidates:
            logger.warning("All hospitals saturated! Saturated triage mode activated.")
            candidates = [h for h, _ in hospitals]

        best_result: Optional[SearchResult] = None
        best_hospital: Optional[str] = None

        for hosp in candidates:
            res = a_star_search(self.infra_graph, origin_zone, hosp)
            if res.path and (best_result is None or res.cost < best_result.cost):
                best_result = res
                best_hospital = hosp

        if best_result and best_hospital:
            routing_record = RoutingResult(
                algorithm="A*",
                source=origin_zone,
                destination=best_hospital,
                path=best_result.path,
                cost_min=best_result.cost,
                distance_km=best_result.cost * 0.55,
                nodes_expanded=best_result.nodes_expanded,
                execution_time_us=best_result.execution_time_us
            )
            self.city_state.record_routing_result(routing_record)
            return {
                "origin_zone": origin_zone,
                "assigned_hospital": best_hospital,
                "available_beds": self.infra_graph.nodes[best_hospital].get("available_beds", 0),
                "path": best_result.path,
                "travel_time_min": round(best_result.cost, 2),
                "nodes_expanded": best_result.nodes_expanded
            }
        return {"error": f"No accessible hospital found from {origin_zone}"}

    def route_field_team(self, depot_id: str, target_zones: List[str]) -> Dict[str, Any]:
        if not target_zones:
            return {"depot": depot_id, "route": [depot_id], "total_travel_time_min": 0.0}

        current = depot_id
        unvisited = list(target_zones)
        complete_path = [current]
        total_time = 0.0

        while unvisited:
            best_next = None
            best_cost = float("inf")
            best_subpath = []
            for target in unvisited:
                res = a_star_search(self.infra_graph, current, target)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_next = target
                    best_subpath = res.path
            if best_next is None:
                break
            total_time += best_cost
            complete_path.extend(best_subpath[1:])
            current = best_next
            unvisited.remove(best_next)

        return {
            "depot": depot_id,
            "target_zones": target_zones,
            "complete_route": complete_path,
            "total_travel_time_min": round(total_time, 2)
        }

    def detect_high_risk_clusters(self, case_threshold: float = 15.0) -> List[List[str]]:
        high_risk_nodes = set()
        for zid, snap in self.city_state.zone_snapshots.items():
            if snap.confirmed_cases >= case_threshold or snap.alert_level in ["ORANGE", "RED"]:
                high_risk_nodes.add(zid)

        subgraph = self.infra_graph.subgraph(high_risk_nodes)
        clusters = []
        visited = set()

        for node in subgraph.nodes():
            if node not in visited:
                cluster = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)
                    for nbr in subgraph.neighbors(curr):
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                clusters.append(sorted(cluster))
        return clusters

    def trace_transmission_pathways(self, seed_zone: str, min_flux: float = 0.15) -> List[List[str]]:
        pathways = []
        def dfs_path(curr: str, current_path: List[str]):
            if len(current_path) >= 4:
                pathways.append(list(current_path))
                return
            has_successor = False
            for succ in self.mobility_graph.successors(curr):
                flux = float(self.mobility_graph[curr][succ].get("weight", 0.0))
                if flux >= min_flux and succ not in current_path:
                    has_successor = True
                    dfs_path(succ, current_path + [succ])
            if not has_successor and len(current_path) > 1:
                pathways.append(list(current_path))

        dfs_path(seed_zone, [seed_zone])
        return pathways

    def run_benchmark_comparison(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        if output_dir is None:
            output_dir = get_outputs_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        zones = self.city_state.get_zone_ids()
        hospitals = self.city_state.get_hospital_ids()
        records = []

        for z in zones:
            for h in hospitals:
                bfs_res = breadth_first_search(self.infra_graph, z, h)
                dfs_res = depth_first_search(self.infra_graph, z, h)
                ucs_res = uniform_cost_search(self.infra_graph, z, h)
                astar_res = a_star_search(self.infra_graph, z, h)

                for res in [bfs_res, dfs_res, ucs_res, astar_res]:
                    records.append({
                        "origin": z,
                        "destination": h,
                        "algorithm": res.algorithm,
                        "path_length_hops": len(res.path) - 1 if res.path else 0,
                        "path_cost_min": round(res.cost, 2),
                        "nodes_expanded": res.nodes_expanded,
                        "runtime_us": round(res.execution_time_us, 1)
                    })

        df = pd.DataFrame(records)
        csv_path = output_dir / "search_benchmarks.csv"
        df.to_csv(csv_path, index=False)

        json_path = output_dir / "search_evaluation_summary.json"
        summary = {
            "total_benchmark_queries": len(records),
            "average_nodes_expanded": {
                algo: round(float(df[df["algorithm"] == algo]["nodes_expanded"].mean()), 2)
                for algo in ["BFS", "DFS", "UCS", "A*"]
            },
            "average_runtime_us": {
                algo: round(float(df[df["algorithm"] == algo]["runtime_us"].mean()), 2)
                for algo in ["BFS", "DFS", "UCS", "A*"]
            }
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved search benchmarks to {csv_path} and {json_path}")
        return df
