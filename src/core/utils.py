"""
utils.py - Core Utilities and Geospatial Mathematics
===================================================
Author: Member 1 (Lead Architect & Core State)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Provides:
- Haversine geodetic distance formula for realistic physical transit times.
- Admissible travel time calculations for A* search heuristics.
- Robust filesystem path resolution (avoids FileNotFoundError regardless of cwd).
- Standardized project-wide logging setup.
"""

import math
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# Earth radius in kilometers
EARTH_RADIUS_KM: float = 6371.0

# Maximum emergency transit speed in urban traffic (km/min)
# 45 km/h = 0.75 km/min
MAX_TRANSIT_SPEED_KM_MIN: float = 0.75


def get_project_root() -> Path:
    """Returns the absolute Path to the 'final_project' directory."""
    # utils.py is in final_project/src/core/
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Returns the absolute Path to the 'final_project/data' directory."""
    data_dir = get_project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_outputs_dir() -> Path:
    """Returns the absolute Path to the 'final_project/outputs' directory."""
    outputs_dir = get_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates using the Haversine formula.
    Grounded in spherical trigonometry:
      a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
      c = 2 * atan2(sqrt(a), sqrt(1-a))
      d = R * c
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def travel_time_minutes(distance_km: float, speed_km_min: float = 0.50) -> float:
    """
    Converts physical distance to travel time in minutes.
    Default speed 30 km/h = 0.5 km/min.
    """
    if speed_km_min <= 0:
        raise ValueError("Speed must be positive")
    return distance_km / speed_km_min


def setup_logger(name: str = "dengue_coordinator") -> logging.Logger:
    """Configures a clean student-readable logger for the coordinator system."""
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
