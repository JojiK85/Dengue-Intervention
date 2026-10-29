import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.city_state import haversine_distance_km, travel_time_minutes, get_data_dir, setup_logger

logger = setup_logger("data_generator")

SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)

def generate_all_synthetic_data(data_dir: Optional[Path] = None) -> Dict[str, Path]:
    if data_dir is None:
        data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    zones_data = [
        {
            "zone_id": "ZONE_CENTRAL",
            "zone_name": "Central Commercial Ward",
            "lat": 20.2961, "lon": 85.8245,
            "population": 95000, "area_sq_km": 6.5,
            "vulnerability_tier": "Low",
            "clinic_access_ratio": 0.88,
            "underreporting_factor": 2.1,
            "initial_cases": 18,
            "initial_breteau": 14.5,
            "hospital_bed_capacity": 60,
            "initial_hospital_beds_occupied": 28,
            "initial_diagnostic_kits": 120
        },
        {
            "zone_id": "ZONE_NORTH",
            "zone_name": "Northern Dense Settlement",
            "lat": 20.3200, "lon": 85.8150,
            "population": 120000, "area_sq_km": 8.2,
            "vulnerability_tier": "High",
            "clinic_access_ratio": 0.42,
            "underreporting_factor": 4.2,
            "initial_cases": 24,
            "initial_breteau": 32.0,
            "hospital_bed_capacity": 45,
            "initial_hospital_beds_occupied": 35,
            "initial_diagnostic_kits": 80
        },
        {
            "zone_id": "ZONE_SOUTH",
            "zone_name": "Southern Industrial & Peri-Urban",
            "lat": 20.2650, "lon": 85.8350,
            "population": 85000, "area_sq_km": 11.0,
            "vulnerability_tier": "High",
            "clinic_access_ratio": 0.35,
            "underreporting_factor": 4.5,
            "initial_cases": 8,
            "initial_breteau": 28.5,
            "hospital_bed_capacity": 30,
            "initial_hospital_beds_occupied": 22,
            "initial_diagnostic_kits": 60
        },
        {
            "zone_id": "ZONE_EAST",
            "zone_name": "Eastern Residential Suburb",
            "lat": 20.2850, "lon": 85.8650,
            "population": 70000, "area_sq_km": 7.4,
            "vulnerability_tier": "Medium",
            "clinic_access_ratio": 0.65,
            "underreporting_factor": 3.0,
            "initial_cases": 12,
            "initial_breteau": 18.0,
            "hospital_bed_capacity": 40,
            "initial_hospital_beds_occupied": 15,
            "initial_diagnostic_kits": 90
        },
        {
            "zone_id": "ZONE_WEST",
            "zone_name": "Western Institutional / University Area",
            "lat": 20.3050, "lon": 85.7800,
            "population": 65000, "area_sq_km": 9.0,
            "vulnerability_tier": "Low",
            "clinic_access_ratio": 0.82,
            "underreporting_factor": 2.2,
            "initial_cases": 6,
            "initial_breteau": 12.0,
            "hospital_bed_capacity": 35,
            "initial_hospital_beds_occupied": 10,
            "initial_diagnostic_kits": 100
        },
        {
            "zone_id": "ZONE_SLUM_LAKE",
            "zone_name": "Lakefront Informal Settlement",
            "lat": 20.2750, "lon": 85.8100,
            "population": 55000, "area_sq_km": 4.1,
            "vulnerability_tier": "High",
            "clinic_access_ratio": 0.28,
            "underreporting_factor": 4.8,
            "initial_cases": 4,
            "initial_breteau": 38.0,
            "hospital_bed_capacity": 20,
            "initial_hospital_beds_occupied": 16,
            "initial_diagnostic_kits": 40
        }
    ]
    zones_df = pd.DataFrame(zones_data)
    zones_path = data_dir / "zones_demographics.csv"
    zones_df.to_csv(zones_path, index=False)

    infra_data = [
        {
            "facility_id": "HOSP_CIVIL_CENTRAL",
            "facility_name": "Bhubaneswar Central Civil Hospital",
            "facility_type": "hospital",
            "lat": 20.2970, "lon": 85.8230,
            "total_beds": 120, "available_beds": 25,
            "diagnostic_kits": 350,
            "inspection_teams": 3,
            "spraying_units": 2
        },
        {
            "facility_id": "HOSP_NORTH_SUBDIV",
            "facility_name": "Northern Sub-Divisional Hospital",
            "facility_type": "hospital",
            "lat": 20.3250, "lon": 85.8180,
            "total_beds": 60, "available_beds": 8,
            "diagnostic_kits": 150,
            "inspection_teams": 2,
            "spraying_units": 1
        },
        {
            "facility_id": "HOSP_SOUTH_GENERAL",
            "facility_name": "South Urban Community Hospital",
            "facility_type": "hospital",
            "lat": 20.2600, "lon": 85.8400,
            "total_beds": 50, "available_beds": 5,
            "diagnostic_kits": 100,
            "inspection_teams": 1,
            "spraying_units": 1
        },
        {
            "facility_id": "DEPOT_CENTRAL_VECTOR",
            "facility_name": "Central Vector Control Headquarters",
            "facility_type": "depot",
            "lat": 20.2920, "lon": 85.8200,
            "total_beds": 0, "available_beds": 0,
            "diagnostic_kits": 500,
            "inspection_teams": 6,
            "spraying_units": 4
        },
        {
            "facility_id": "LAB_MUNICIPAL_PCR",
            "facility_name": "Regional Molecular Diagnostic Lab",
            "facility_type": "testing_lab",
            "lat": 20.3010, "lon": 85.8300,
            "total_beds": 0, "available_beds": 0,
            "diagnostic_kits": 800,
            "inspection_teams": 0,
            "spraying_units": 0
        }
    ]
    infra_df = pd.DataFrame(infra_data)
    infra_path = data_dir / "city_infrastructure.csv"
    infra_df.to_csv(infra_path, index=False)

    all_nodes = {z["zone_id"]: (z["lat"], z["lon"]) for z in zones_data}
    all_nodes.update({f["facility_id"]: (f["lat"], f["lon"]) for f in infra_data})

    edges_list = [
        ("ZONE_CENTRAL", "ZONE_NORTH"),
        ("ZONE_CENTRAL", "ZONE_SOUTH"),
        ("ZONE_CENTRAL", "ZONE_EAST"),
        ("ZONE_CENTRAL", "ZONE_WEST"),
        ("ZONE_CENTRAL", "ZONE_SLUM_LAKE"),
        ("ZONE_NORTH", "ZONE_WEST"),
        ("ZONE_SOUTH", "ZONE_EAST"),
        ("ZONE_SOUTH", "ZONE_SLUM_LAKE"),
        ("ZONE_EAST", "ZONE_NORTH"),
        ("HOSP_CIVIL_CENTRAL", "ZONE_CENTRAL"),
        ("HOSP_CIVIL_CENTRAL", "DEPOT_CENTRAL_VECTOR"),
        ("HOSP_CIVIL_CENTRAL", "LAB_MUNICIPAL_PCR"),
        ("HOSP_NORTH_SUBDIV", "ZONE_NORTH"),
        ("HOSP_NORTH_SUBDIV", "ZONE_WEST"),
        ("HOSP_SOUTH_GENERAL", "ZONE_SOUTH"),
        ("HOSP_SOUTH_GENERAL", "ZONE_SLUM_LAKE"),
        ("DEPOT_CENTRAL_VECTOR", "ZONE_CENTRAL"),
        ("DEPOT_CENTRAL_VECTOR", "ZONE_SLUM_LAKE"),
        ("LAB_MUNICIPAL_PCR", "ZONE_CENTRAL"),
        ("LAB_MUNICIPAL_PCR", "ZONE_EAST")
    ]

    edges_data = []
    for u, v in edges_list:
        lat1, lon1 = all_nodes[u]
        lat2, lon2 = all_nodes[v]
        dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
        road_km = round(dist_km * 1.22, 2)
        tt_min = round(travel_time_minutes(road_km, speed_km_min=0.55), 1)
        edges_data.append({
            "source": u,
            "target": v,
            "distance_km": road_km,
            "travel_time_min": tt_min
        })

    edges_df = pd.DataFrame(edges_data)
    edges_path = data_dir / "spatial_network_edges.csv"
    edges_df.to_csv(edges_path, index=False)

    mobility_data = []
    for z1 in zones_data:
        zid1 = z1["zone_id"]
        p1 = z1["population"]
        lat1, lon1 = z1["lat"], z1["lon"]
        raw_fluxes = []
        for z2 in zones_data:
            zid2 = z2["zone_id"]
            if zid1 == zid2:
                raw_fluxes.append((zid2, 0.0))
                continue
            p2 = z2["population"]
            lat2, lon2 = z2["lat"], z2["lon"]
            d = max(0.8, haversine_distance_km(lat1, lon1, lat2, lon2))
            gravity_val = (p1 ** 0.55 * p2 ** 0.55) / (d ** 1.4)
            raw_fluxes.append((zid2, gravity_val))

        total_flux = sum(f[1] for f in raw_fluxes)
        for zid2, raw_val in raw_fluxes:
            prob = round(raw_val / total_flux, 4) if total_flux > 0 else 0.0
            mobility_data.append({
                "source": zid1,
                "target": zid2,
                "commuter_flux_probability": prob
            })

    mob_df = pd.DataFrame(mobility_data)
    mob_path = data_dir / "mobility_commuter_flux.csv"
    mob_df.to_csv(mob_path, index=False)

    weather_data = []
    for day in range(1, 31):
        if 11 <= day <= 15:
            base_rain = 75.0 + np.random.uniform(-10.0, 20.0)
            base_temp = 27.0 + np.random.uniform(-1.0, 1.5)
            base_humid = 88.0 + np.random.uniform(-2.0, 5.0)
        elif 20 <= day <= 24:
            base_rain = 5.0 + np.random.uniform(0.0, 10.0)
            base_temp = 32.5 + np.random.uniform(-1.0, 2.0)
            base_humid = 78.0 + np.random.uniform(-3.0, 4.0)
        else:
            base_rain = 15.0 + np.random.exponential(12.0)
            base_temp = 29.5 + np.random.normal(0, 1.5)
            base_humid = 72.0 + np.random.normal(0, 5.0)

        for z in zones_data:
            zid = z["zone_id"]
            tier = z["vulnerability_tier"]
            tier_mult = 1.35 if tier == "High" else (1.05 if tier == "Medium" else 0.85)

            breteau = z["initial_breteau"] * tier_mult + (base_rain * 0.18) + np.random.normal(0, 2.0)
            breteau = max(4.0, min(80.0, round(breteau, 1)))

            true_risk_factor = (breteau / 20.0) * (base_humid / 70.0)
            daily_true_infections = int(np.random.poisson(max(1.0, true_risk_factor * 8.0)))
            daily_reported = int(max(0, round(daily_true_infections / z["underreporting_factor"])))

            weather_data.append({
                "day": day,
                "zone_id": zid,
                "rainfall_mm": round(base_rain, 1),
                "temperature_c": round(base_temp, 1),
                "humidity_pct": round(base_humid, 1),
                "breteau_index": breteau,
                "adult_vector_index": round(breteau * 0.08, 2),
                "daily_reported_cases": daily_reported,
                "daily_true_cases": daily_true_infections,
                "citizen_fever_calls": int(np.random.poisson(daily_reported * 2.2 + 2))
            })

    weather_df = pd.DataFrame(weather_data)
    weather_path = data_dir / "weather_surveillance.csv"
    weather_df.to_csv(weather_path, index=False)

    return {
        "zones": zones_path,
        "infrastructure": infra_path,
        "edges": edges_path,
        "mobility": mob_path,
        "weather": weather_path
    }

def ingest_opendengue_raw(csv_path: Optional[Path] = None) -> pd.DataFrame:
    data_dir = get_data_dir()
    target_path = csv_path or (data_dir / "weather_surveillance.csv")
    if not target_path.exists():
        logger.warning(f"File {target_path} not found. Returning empty dataframe.")
        return pd.DataFrame()

    df = pd.read_csv(target_path)
    mapping = {
        "adm2_name": "zone_id",
        "dengue_cases": "daily_reported_cases",
        "rainfall": "rainfall_mm",
        "temp": "temperature_c"
    }
    return df.rename(columns=mapping)

if __name__ == "__main__":
    paths = generate_all_synthetic_data()
    print("Generated synthetic data files:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
