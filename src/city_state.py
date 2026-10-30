from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import networkx as nx
import pandas as pd

EARTH_RADIUS_KM: float = 6371.0
MAX_TRANSIT_SPEED_KM_MIN: float = 0.75

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_data_dir() -> Path:
    data_dir = get_project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_outputs_dir() -> Path:
    outputs_dir = get_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c

def travel_time_minutes(distance_km: float, speed_km_min: float = 0.50) -> float:
    if speed_km_min <= 0:
        raise ValueError("Speed must be positive")
    return distance_km / speed_km_min

def setup_logger(name: str = "dengue_coordinator") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

logger = setup_logger("city_state")

@dataclass
class ZoneSnapshot:
    zone_id: str
    day: int = 1
    confirmed_cases: float = 0.0
    estimated_true_cases: float = 0.0
    cumulative_cases: float = 0.0
    breteau_index: float = 15.0
    adult_vector_density: float = 1.0
    rainfall_mm: float = 10.0
    temp_c: float = 29.0
    humidity_pct: float = 72.0
    beds_occupied: int = 0
    beds_capacity: int = 50
    kits_remaining: int = 100
    teams_active: int = 0
    risk_score: float = 0.20
    alert_level: str = "GREEN"
    underreporting_flag: bool = False
    confidence_score: float = 0.85

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PlanStep:
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
    status: str = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RoutingResult:
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
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or get_data_dir()
        self.current_day: int = 1
        self.mobility_graph: nx.DiGraph = nx.DiGraph()
        self.infra_graph: nx.Graph = nx.Graph()
        self.zones: pd.DataFrame = pd.DataFrame()
        self.infrastructure: pd.DataFrame = pd.DataFrame()
        self.weather_data: pd.DataFrame = pd.DataFrame()
        self.zone_snapshots: Dict[str, ZoneSnapshot] = {}
        self.plan_steps: List[PlanStep] = []
        self.routing_results: List[RoutingResult] = []
        self.execution_logs: List[Dict[str, Any]] = []
        self.load_city_data()

    def load_city_data(self) -> None:
        zones_file = self.data_dir / "zones_demographics.csv"
        infra_file = self.data_dir / "city_infrastructure.csv"
        edges_file = self.data_dir / "spatial_network_edges.csv"
        mob_file = self.data_dir / "mobility_commuter_flux.csv"
        weather_file = self.data_dir / "weather_surveillance.csv"

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

        if weather_file.exists():
            self.weather_data = pd.read_csv(weather_file)

    def get_zone_ids(self) -> List[str]:
        return sorted(list(self.zone_snapshots.keys()))

    def get_hospital_ids(self) -> List[str]:
        return [n for n, data in self.infra_graph.nodes(data=True) if data.get("node_type") == "hospital"]

    def advance_day(self) -> int:
        self.current_day += 1
        for snap in self.zone_snapshots.values():
            snap.day = self.current_day
        return self.current_day

    def record_routing_result(self, res: RoutingResult) -> None:
        self.routing_results.append(res)

    def record_plan_step(self, step: PlanStep) -> None:
        self.plan_steps.append(step)

    def get_summary_snapshot(self) -> Dict[str, Any]:
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
