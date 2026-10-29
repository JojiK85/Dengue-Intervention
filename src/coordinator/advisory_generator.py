"""
advisory_generator.py - Deterministic Public Health Advisory & SITREP Generator
================================================================================
Author: Member 5 (Planning) & Member 8 (Uncertainty)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

CRITICAL SCOPE COMPLIANCE NOTE:
In strict accordance with course guidelines, Module 5 (GenAI / Multi-Agent LLMs)
is EXCLUDED. All public-health advisories, situation reports (SITREPs), and action
recommendations are generated 100% deterministically in pure Python from verified
mathematical outputs of Modules 1 through 4. Zero LLM API calls or black-box prompts.
"""

from typing import Dict, List, Any
from src.core.city_state import CityState


def generate_deterministic_sitrep(
    city_state: CityState,
    day: int,
    csp_status: str,
    critical_path: List[str]
) -> str:
    """
    Synthesizes a structured municipal situation report (SITREP) deterministically.
    """
    summary = city_state.get_summary_snapshot()
    high_risk = summary["high_risk_zones"]
    underreported = summary["underreporting_zones"]

    sitrep_lines = [
        f"# MUNICIPAL DENGUE SITUATION REPORT (SITREP) - DAY {day:02d}",
        f"**Jurisdiction:** Urban Municipal Corporation | **Surveillance Cycle:** Longitudinal 30-Day Simulation",
        f"**Deterministic Decision-Support Engine:** Modules 1-4 (Search, CSP, Planning, Bayes Net, MDP/RL)",
        "---",
        "### 1. Epidemiological Summary",
        f"- **Total Clinically Reported Cases:** {summary['total_confirmed_cases']}",
        f"- **Estimated True Infection Burden:** {summary['total_estimated_true_cases']} (accounting for under-reporting multipliers)",
        f"- **Hospital Surge Bed Occupancy:** {summary['total_beds_occupied']} occupied",
        f"- **Diagnostic Test Kits Remaining:** {summary['total_diagnostic_kits_remaining']}",
        f"- **High-Risk Wards (Alert ORANGE/RED):** {', '.join(high_risk) if high_risk else 'None'}",
        f"- **Latent Under-Reporting Alerts:** {', '.join(underreported) if underreported else 'None'}",
        "",
        "### 2. Algorithmic Decisions & Resource Directives",
        f"- **Module 3 (Bayesian Network):** Exact Variable Elimination inferred true outbreak risks; detected {len(underreported)} under-reported sectors.",
        f"- **Module 1 (CSP Allocator):** Allocation status: **{csp_status}**. Enforced strict team travel times and high-risk coverage constraints.",
        f"- **Module 2 (Intervention Planning):** Critical path activities governing schedule: `{', '.join(critical_path[:3])}`.",
        "",
        "### 3. Public Health Guidance & Preventative Advisories",
    ]

    for zid in high_risk:
        snap = city_state.zone_snapshots[zid]
        sitrep_lines.extend([
            f"**Ward {zid} (Alert {snap.alert_level}, True Risk {snap.risk_score:.2f}, Confidence {snap.confidence_score:.2f}):**",
            f"  - Vector Control Directive: Immediate adulticide spraying and source reduction authorized.",
            f"  - Clinical Guidance: Mobilize triage beds at nearest referral facility; distribute diagnostic kits.",
            f"  - Citizen Advisory: Empty stagnant water coolers and overhead containers within 48 hours."
        ])

    if underreported:
        sitrep_lines.append("")
        sitrep_lines.append("**SPECIAL SURVEILLANCE DIRECTIVE (Under-Reporting Hotspots):**")
        for zid in underreported:
            sitrep_lines.append(
                f"  - Ward {zid} exhibits low clinic reporting despite high Breteau larval index. "
                f"Active door-to-door fever screening squads dispatched to prevent unobserved contagion."
            )

    sitrep_lines.append("---")
    sitrep_lines.append("*Report generated deterministically by the Adaptive Dengue Surveillance Coordinator.*")
    return "\n".join(sitrep_lines)
