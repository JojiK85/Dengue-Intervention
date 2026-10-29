"""Core package: CityState digital twin, dataclasses, and utilities."""
from .city_state import CityState, ZoneSnapshot, PlanStep, RoutingResult
from .utils import (
    haversine_distance_km,
    travel_time_minutes,
    get_project_root,
    get_data_dir,
    get_outputs_dir,
    setup_logger
)

__all__ = [
    "CityState",
    "ZoneSnapshot",
    "PlanStep",
    "RoutingResult",
    "haversine_distance_km",
    "travel_time_minutes",
    "get_project_root",
    "get_data_dir",
    "get_outputs_dir",
    "setup_logger"
]
