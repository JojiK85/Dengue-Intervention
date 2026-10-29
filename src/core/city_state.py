"""
city_state.py - Shared Digital-Twin State Object
=================================================
Author: Member 1 (Lead Architect & Core State)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

The CityState object serves as the single source of truth for the entire pipeline.
Every upstream module updates this shared digital-twin state; every downstream module reads from it.
Dual-layer NetworkX graph:
  - mobility_graph: Directed graph weighted by commuter mobility flux (DengueGNN gravity model).
  - infra_graph: Undirected graph weighted by physical road distance and travel time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import networkx as nx
import pandas as pd

from .utils import get_data_dir, haversine_distance_km, travel_time_minutes, setup_logger

logger = setup_logger("city_state")


@dataclass
class ZoneSnapshot:
    """Live state of an urban municipal ward/zone for a specific simulation day."""
    zone_id: str
    day: int = 1
    # Epidemiological state
    confirmed_cases: float = 0.0
    estimated_true_cases: float = 0.0
    cumulative_cases: float = 0.0
    breteau_index: float = 15.0          # Positive containers per 100 inspected
    adult_vector_density: float = 1.0   # Mosquito count per trap
    # Environmental factors
    rainfall_mm: float = 10.0
    temp_c: float = 29.0
    humidity_pct: float = 72.0
    # Healthcare & municipal resource state
    beds_occupied: int = 0
    beds_capacity: int = 50
    kits_remaining: int = 100
    teams_active: int = 0
    # Module 3 probabilistic outputs
    risk_score: float = 0.20            # P(TrueRisk = High | Evidence)
    alert_level: str = "GREEN"          # GREEN, YELLOW, ORANGE, RED
    underreporting_flag: bool = False
    confidence_score: float = 0.85
    # Module 4 RL outputs
    rl_action: Optional[str] = "Routine_Monitoring"
    rl_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanStep:
    """A discrete public-health intervention action scheduled by Module 2."""
    step_id: int
    day: int
    zone_id: str
    action: str
    duration_days: int = 1
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    required_teams: int = 1
    required_kits: int = 0
    required_beds: int = 0
    cost_inr: float = 5000.0
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, REPLANNED, SKIPPED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingResult:
    """Result of pathfinding algorithms executed in Module 1."""
    algorithm: str
    source: str
    destination: str
    path: List[str]
    cost_min: float
    distance_km: float
    nodes_expanded: int
    execution_time_us: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CityState:
    """
    Central digital-twin state engine coordinating all AI modules.
    Maintains dual NetworkX graphs, zone snapshots, clinical histories, and resources.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or get_data_dir()
        self.current_day: int = 1

        # Dual-layer graphs
        self.mobility_graph: nx.DiGraph = nx.DiGraph()
        self.infra_graph: nx.Graph = nx.Graph()

        # DataFrames loaded from storage
        self.zones: pd.DataFrame = pd.DataFrame()
        self.infrastructure: pd.DataFrame = pd.DataFrame()
        self.weather_data: pd.DataFrame = pd.DataFrame()

        # Dynamic state structures
        self.zone_snapshots: Dict[str, ZoneSnapshot] = {}
        self.plan_steps: List[PlanStep] = []
        self.routing_results: List[RoutingResult] = []
        self.execution_logs: List[Dict[str, Any]] = []

        # Load initial datasets
        self.load_city_data()

    def load_city_data(self) -> None:
        """Loads static ward demographic and infrastructure tables if they exist."""
        zones_file = self.data_dir / "zones_demographics.csv"
        infra_file = self.data_dir / "city_infrastructure.csv"
        edges_file = self.data_dir / "spatial_network_edges.csv"
        mob_file = self.data_dir / "mobility_commuter_flux.csv"

        if zones_file.exists():
            self.zones = pd.read_csv(zones_file)
            for _, row in self.zones.iterrows():
                zid = str(row["zone_id"])
                self.zone_snapshots[zid] = ZoneSnapshot(
                    zone_id=zid,
                    day=self.current_day,
                    confirmed_cases=float(row.get("initial_cases", 5)),
                    estimated_true_cases=float(row.get("initial_cases", 5)) * float(row.get("underreporting_factor", 2.5)),
                    cumulative_cases=float(row.get("initial_cases", 5)),
                    breteau_index=float(row.get("initial_breteau", 18.0)),
                    beds_occupied=int(row.get("initial_hospital_beds_occupied", 10)),
                    beds_capacity=int(row.get("hospital_bed_capacity", 50)),
                    kits_remaining=int(row.get("initial_diagnostic_kits", 100))
                )
                # Add node to infra graph
                self.infra_graph.add_node(
                    zid,
                    node_type="zone",
                    name=row.get("zone_name", zid),
                    lat=float(row.get("lat", 20.2961)),
                    lon=float(row.get("lon", 85.8245)),
                    population=int(row.get("population", 75000)),
                    vulnerability=str(row.get("vulnerability_tier", "Medium"))
                )
                self.mobility_graph.add_node(
                    zid,
                    population=int(row.get("population", 75000))
                )

        if infra_file.exists():
            self.infrastructure = pd.read_csv(infra_file)
            for _, row in self.infrastructure.iterrows():
                fid = str(row["facility_id"])
                self.infra_graph.add_node(
                    fid,
                    node_type=str(row.get("facility_type", "hospital")),
                    name=str(row.get("facility_name", fid)),
                    lat=float(row.get("lat", 20.2961)),
                    lon=float(row.get("lon", 85.8245)),
                    total_beds=int(row.get("total_beds", 100)),
                    available_beds=int(row.get("available_beds", 30)),
                    diagnostic_kits=int(row.get("diagnostic_kits", 200)),
                    inspection_teams=int(row.get("inspection_teams", 2)),
                    spraying_units=int(row.get("spraying_units", 2))
                )

        if edges_file.exists():
            edges_df = pd.read_csv(edges_file)
            for _, row in edges_df.iterrows():
                u, v = str(row["source"]), str(row["target"])
                d_km = float(row["distance_km"])
                t_min = float(row.get("travel_time_min", travel_time_minutes(d_km)))
                self.infra_graph.add_edge(u, v, distance_km=d_km, travel_time_min=t_min)

        if mob_file.exists():
            mob_df = pd.read_csv(mob_file)
            for _, row in mob_df.iterrows():
                u, v = str(row["source"]), str(row["target"])
                flux = float(row["commuter_flux_probability"])
                self.mobility_graph.add_edge(u, v, weight=flux)

    def get_zone_ids(self) -> List[str]:
        """Returns sorted list of municipal zone identifiers."""
        return sorted(list(self.zone_snapshots.keys()))

    def get_hospital_ids(self) -> List[str]:
        """Returns list of hospital facility identifiers in the infrastructure graph."""
        hospitals = []
        for n, data in self.infra_graph.nodes(data=True):
            if data.get("node_type") == "hospital":
                hospitals.append(n)
        return hospitals

    def advance_day(self) -> int:
        """Advances the city simulation clock by 1 day."""
        self.current_day += 1
        for snap in self.zone_snapshots.values():
            snap.day = self.current_day
        return self.current_day

    def record_routing_result(self, res: RoutingResult) -> None:
        """Stores a search pathfinding benchmark."""
        self.routing_results.append(res)

    def record_plan_step(self, step: PlanStep) -> None:
        """Stores a scheduled or executed plan action."""
        self.plan_steps.append(step)

    def get_summary_snapshot(self) -> Dict[str, Any]:
        """Returns a consolidated dictionary summary of city status."""
        total_confirmed = sum(s.confirmed_cases for s in self.zone_snapshots.values())
        total_estimated = sum(s.estimated_true_cases for s in self.zone_snapshots.values())
        total_beds_occ = sum(s.beds_occupied for s in self.zone_snapshots.values())
        total_kits = sum(s.kits_remaining for s in self.zone_snapshots.values())
        high_risk_zones = [zid for zid, s in self.zone_snapshots.items() if s.alert_level in ["ORANGE", "RED"]]
        underreporting_zones = [zid for zid, s in self.zone_snapshots.items() if s.underreporting_flag]

        return {
            "day": self.current_day,
            "total_zones": len(self.zone_snapshots),
            "total_confirmed_cases": round(total_confirmed, 1),
            "total_estimated_true_cases": round(total_estimated, 1),
            "total_beds_occupied": total_beds_occ,
            "total_diagnostic_kits_remaining": total_kits,
            "high_risk_zones": high_risk_zones,
            "underreporting_zones": underreporting_zones
        }
