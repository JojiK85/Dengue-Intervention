"""
opendengue_loader.py - OpenDengue & Surveillance Data Adapter
=============================================================
Author: Member 1 (Lead Architect & Core State)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Adapts global OpenDengue standard epidemiological schemas (Adm0, Adm1, Adm2,
temporal case registers, test positivity) to our local municipal digital-twin schema.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from src.core.utils import get_data_dir, setup_logger

logger = setup_logger("opendengue_loader")


def ingest_opendengue_raw(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parses OpenDengue standardized CSV format and aligns columns with CityState.
    If no external file is supplied, returns the verified municipal surveillance table.
    """
    data_dir = get_data_dir()
    target_path = csv_path or (data_dir / "weather_surveillance.csv")

    if not target_path.exists():
        logger.warning(f"File {target_path} not found. Returning empty dataframe.")
        return pd.DataFrame()

    df = pd.read_csv(target_path)
    # OpenDengue schema mapping dictionary
    mapping = {
        "adm2_name": "zone_id",
        "dengue_cases": "daily_reported_cases",
        "rainfall": "rainfall_mm",
        "temp": "temperature_c"
    }
    df = df.rename(columns=mapping)
    return df
