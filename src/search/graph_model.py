"""
graph_model.py - Spatial Graph Representation & Helpers
========================================================
Author: Member 2 (Search & Geospatial Routing)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Provides helper abstractions over the NetworkX dual-layer graphs for Module 1.
"""

from typing import Dict, List, Any, Optional
import networkx as nx
import pandas as pd

from src.core.city_state import CityState


class SpatialCityGraph:
    """Convenience wrapper around CityState dual-layer graphs for search operations."""

    def __init__(self, city_state: CityState):
        self.city_state = city_state
        self.infra_graph: nx.Graph = city_state.infra_graph
        self.mobility_graph: nx.DiGraph = city_state.mobility_graph

    def get_zone_nodes(self) -> List[str]:
        return [n for n, d in self.infra_graph.nodes(data=True) if d.get("node_type") == "zone"]

    def get_hospital_nodes(self) -> List[str]:
        return [n for n, d in self.infra_graph.nodes(data=True) if d.get("node_type") == "hospital"]

    def get_depot_nodes(self) -> List[str]:
        return [n for n, d in self.infra_graph.nodes(data=True) if d.get("node_type") == "depot"]

    def get_edge_weight(self, u: str, v: str, key: str = "travel_time_min") -> float:
        if self.infra_graph.has_edge(u, v):
            return float(self.infra_graph[u][v].get(key, 1.0))
        return float("inf")
