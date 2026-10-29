"""
evaluator.py - Search Evaluations, Patient Routing, and Clustering
===================================================================
Author: Member 2 (Search & Geospatial Routing)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Executes real public-health routing workflows:
1. Patient Ambulance Diversion: routes patients to nearest hospital with free beds.
2. Field Team Vehicle Routing: multi-target inspection route generation.
3. High-Risk Cluster Detection: connected components on transmission graph using BFS.
4. Transmission Pathway Tracing: contagion pathway discovery using DFS on mobility flux.
5. Search Algorithm Benchmarks: comparative table comparing BFS, DFS, UCS, A*.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import networkx as nx

from src.core.city_state import CityState, RoutingResult
from src.core.utils import get_outputs_dir, setup_logger
from .algorithms import (
    breadth_first_search,
    depth_first_search,
    uniform_cost_search,
    a_star_search,
    SearchResult
)
from .heuristics import haversine_admissible_heuristic

logger = setup_logger("search_evaluator")


class SearchEvaluator:
    """Orchestrates pathfinding, hospital diversion, and cluster detection."""

    def __init__(self, city_state: CityState):
        self.city_state = city_state
        self.infra_graph = city_state.infra_graph
        self.mobility_graph = city_state.mobility_graph

    def route_patient_to_best_hospital(
        self,
        origin_zone: str,
        min_beds_required: int = 1
    ) -> Dict[str, Any]:
        """
        Diverts an acute patient/ambulance to the nearest hospital that has available beds.
        Uses A* Search to find the optimal path in terms of road transit time.
        """
        hospitals = []
        for n, d in self.infra_graph.nodes(data=True):
            if d.get("node_type") == "hospital":
                avail = int(d.get("available_beds", 0))
                hospitals.append((n, avail))

        # Check hospitals that satisfy bed capacity constraint
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
                distance_km=best_result.cost * 0.55,  # Estimated distance
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

    def route_field_team(
        self,
        depot_id: str,
        target_zones: List[str]
    ) -> Dict[str, Any]:
        """
        Generates a multi-stop inspection or spraying route using nearest-neighbor A*.
        """
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
        """
        Finds connected clusters of high-risk wards using BFS traversal on the spatial graph.
        """
        high_risk_nodes = set()
        for zid, snap in self.city_state.zone_snapshots.items():
            if snap.confirmed_cases >= case_threshold or snap.alert_level in ["ORANGE", "RED"]:
                high_risk_nodes.add(zid)

        subgraph = self.infra_graph.subgraph(high_risk_nodes)
        clusters = []
        visited = set()

        for node in subgraph.nodes():
            if node not in visited:
                # Run BFS on subgraph
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

    def trace_transmission_pathways(
        self,
        seed_zone: str,
        min_flux: float = 0.15
    ) -> List[List[str]]:
        """
        Traces potential contagion transmission pathways using DFS on commuter mobility graph.
        """
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
        """
        Benchmarks BFS, DFS, UCS, and A* across all origin-destination pairs.
        Exports results to CSV and JSON.
        """
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
