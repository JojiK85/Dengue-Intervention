"""
dashboard.py - Interactive Streamlit Public Health Dashboard
=============================================================
Author: Member 1 (Lead Architect) & Member 6 (Planning)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Provides an interactive GUI for public-health administrators:
  - Daily simulation clock control (Days 1 to 30)
  - Real-time ward epidemiological risk cards
  - Hospital bed saturation gauges & ambulance diversion
  - CPM Critical Path intervention schedule
  - Search algorithm benchmark comparison
  - Reinforcement Learning policy action monitoring
  - Deterministic public-health situation reports (SITREPs)
"""

import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np

from src.core.city_state import CityState
from src.core.utils import get_outputs_dir
from src.coordinator.system_coordinator import AdaptiveDengueCoordinator
from src.visualizer.plotter import generate_all_plots


@st.cache_resource
def get_coordinator() -> AdaptiveDengueCoordinator:
    """Initializes and caches the master system coordinator."""
    city = CityState()
    coord = AdaptiveDengueCoordinator(city)
    return coord


def main():
    st.set_page_config(
        page_title="Adaptive Dengue Surveillance Coordinator",
        page_icon="🦟",
        layout="wide"
    )

    st.title("🦟 Adaptive Dengue Surveillance & Intervention Coordinator")
    st.caption("Group 9 | B.Tech CSE Artificial Intelligence Course Project | IIT Bhubaneswar")

    coordinator = get_coordinator()
    city = coordinator.city_state

    # Sidebar
    st.sidebar.header("🕹️ Simulation Controls")
    sim_day = st.sidebar.slider("Select Simulation Day", min_value=1, max_value=30, value=city.current_day)
    stress_scenario = st.sidebar.selectbox(
        "Inject Stress Scenario",
        ["None (Normal)", "Sudden Case Surge", "Diagnostic Kit Shortage", "Heavy Rainfall Washout", "Hospital Bed Saturation"]
    )

    weather_override = None
    res_override = None

    if stress_scenario == "Sudden Case Surge":
        city.zone_snapshots["ZONE_NORTH"].confirmed_cases = 75.0
        st.sidebar.warning("Injected 300% Case Surge in ZONE_NORTH!")
    elif stress_scenario == "Diagnostic Kit Shortage":
        res_override = {"kits": 30, "teams": 5, "beds": 80}
        st.sidebar.error("Injected Kit Stockout: Only 30 kits available!")
    elif stress_scenario == "Heavy Rainfall Washout":
        weather_override = {"rainfall_mm": 85.0}
        st.sidebar.info("Injected 85mm Torrential Monsoon Downpour!")
    elif stress_scenario == "Hospital Bed Saturation":
        city.infra_graph.nodes["HOSP_CIVIL_CENTRAL"]["available_beds"] = 0
        st.sidebar.error("Central Civil Hospital 100% Saturated!")

    if st.sidebar.button("▶️ Execute Municipal Cycle"):
        with st.spinner(f"Executing Decision Cycle for Day {sim_day}..."):
            day_result = coordinator.run_day(sim_day, weather_override, res_override)
            st.sidebar.success(f"Day {sim_day} Executed Successfully!")

    # Summary Metrics
    summary = city.get_summary_snapshot()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Confirmed Cases", f"{summary['total_confirmed_cases']:.0f}")
    col2.metric("Est. True Burden", f"{summary['total_estimated_true_cases']:.0f}", delta=f"+{summary['total_estimated_true_cases'] - summary['total_confirmed_cases']:.0f} unobserved")
    col3.metric("Beds Occupied", f"{summary['total_beds_occupied']}")
    col4.metric("Kits Remaining", f"{summary['total_diagnostic_kits_remaining']}")
    col5.metric("High-Risk Wards", f"{len(summary['high_risk_zones'])}")

    st.markdown("---")

    # Tabs for modules
    tab_surv, tab_search_csp, tab_plan, tab_rl, tab_sitrep = st.tabs([
        "📊 Module 3: Uncertainty & Surveillance",
        "🗺️ Module 1: Search & CSP Allocator",
        "⏱️ Module 2: Intervention Planning & CPM",
        "🤖 Module 4: MDP & RL Control",
        "📜 Deterministic SITREP"
    ])

    with tab_surv:
        st.subheader("Urban Municipal Wards - Risk & Under-Reporting Status")
        records = []
        for zid, snap in city.zone_snapshots.items():
            records.append({
                "Ward ID": zid,
                "Alert Level": snap.alert_level,
                "Confirmed": snap.confirmed_cases,
                "Est. True Cases": snap.estimated_true_cases,
                "Breteau Index": snap.breteau_index,
                "P(High Risk)": f"{snap.risk_score:.3f}",
                "Confidence": f"{snap.confidence_score:.2f}",
                "Under-Reporting Alert": "🚨 ALERT" if snap.underreporting_flag else "Normal"
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)

    with tab_search_csp:
        st.subheader("Module 1: CSP Allocation & Spatial Routing")
        c1, c2 = st.columns(2)
        with c1:
            st.write("#### CSP Resource Allocation Status")
            alloc_data = []
            for zid, snap in city.zone_snapshots.items():
                alloc_data.append({
                    "Ward": zid,
                    "Inspection Teams": snap.teams_active,
                    "Kits Allocated": snap.kits_remaining
                })
            st.dataframe(pd.DataFrame(alloc_data), use_container_width=True)

        with c2:
            st.write("#### Hospital Bed Saturation & Diversion")
            hosp_data = []
            for n, d in city.infra_graph.nodes(data=True):
                if d.get("node_type") == "hospital":
                    hosp_data.append({
                        "Hospital": d.get("name", n),
                        "Total Beds": d.get("total_beds", 0),
                        "Available Beds": d.get("available_beds", 0)
                    })
            st.dataframe(pd.DataFrame(hosp_data), use_container_width=True)

    with tab_plan:
        st.subheader("Module 2: Activity-On-Node Critical Path Method (CPM)")
        st.write("Forward/backward pass schedule identifying critical path interventions:")
        cpm_csv_path = get_outputs_dir() / "tables" / "cpm_schedule_table.csv"
        if cpm_csv_path.exists():
            st.dataframe(pd.read_csv(cpm_csv_path), use_container_width=True)
        else:
            st.info("Execute municipal cycle to populate CPM schedule.")

    with tab_rl:
        st.subheader("Module 4: Reinforcement Learning & Policy Control")
        st.write(f"**Current Action Executed on G-SEIR Environment:** `{city.zone_snapshots['ZONE_CENTRAL'].rl_action}`")
        st.write(f"**Step Reward:** `{city.zone_snapshots['ZONE_CENTRAL'].rl_reward}`")
        rl_csv_path = get_outputs_dir() / "tables" / "rl_policy_comparison.csv"
        if rl_csv_path.exists():
            st.write("#### Policy Benchmark Comparison")
            st.dataframe(pd.read_csv(rl_csv_path), use_container_width=True)

    with tab_sitrep:
        st.subheader("Deterministic Public Health Situation Report (100% Python, No LLMs)")
        sitrep_text = generate_deterministic_sitrep(city, sim_day, "FEASIBLE", ["T1_surveillance", "T3_field_inspect"])
        st.markdown(f"```markdown\n{sitrep_text}\n```")


if __name__ == "__main__":
    main()
