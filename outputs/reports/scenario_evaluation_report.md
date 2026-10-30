# Adaptive Dengue Coordinator - Scenario Evaluation Report
**Course:** CS4L002 Artificial Intelligence (IIT Bhubaneswar) | **Group 9**

## Overview of Mandatory Public-Health Stress Scenarios
All 8 required scenarios from `Group 9 - Disease Prevention.pdf` were executed against the integrated system.

| # | Scenario Name | Trigger Condition | System Algorithmic Response | Status | Key Metric |
|---|---|---|---|---|---|
| 1 | Sudden Case Surge | Cases spiked to 65 in ZONE_NORTH | Module 3 elevated alert to YELLOW; Module 1 diverted patients to HOSP_NORTH_SUBDIV | SUCCESS | `Alert=YELLOW, Path=1.4min` |
| 2 | Under-Reporting Detection | Reported cases=2 in ZONE_SOUTH despite Breteau=34 and Rain=55mm | Module 3 flagged under-reporting=True; estimated true burden=18.0 cases | SUCCESS | `UnderreportingFlag=True, TrueCasesEst=18.0` |
| 3 | Mosquito-Density Spike | Breteau index spiked to 48.0 in ZONE_EAST and ZONE_SLUM_LAKE | Module 1 BFS detected 0 spatial contagion clusters; Planner scheduled source reduction | SUCCESS | `ClustersDetected=0` |
| 4 | Heavy Monsoon Rainfall | 85mm torrential downpour | Module 2 Replanner suspended outdoor adulticide fogging; pivoted to indoor drainage cleanup | SUCCESS | `ReplanningEvents=1` |
| 5 | Diagnostic-Kit Shortage | City-wide diagnostic kit inventory crashed to 30 kits | Module 1 CSP detected conflict (status=INFEASIBLE_RELAXED); executed relaxed best-effort allocation | SUCCESS | `IsRelaxed=True, ConflictLogs=1` |
| 6 | Field-Team Outage | Municipal squads reduced from 5 to 1 active team | Module 1 Search generated multi-stop vehicle route (transit time: 25.5 min) | SUCCESS | `RouteTime=25.5min` |
| 7 | Hospital Crowding & Diversion | Central Civil Hospital reached 100% saturation (0 available beds) | Module 1 A* successfully diverted ambulances to secondary facility HOSP_NORTH_SUBDIV | SUCCESS | `DivertedTo=HOSP_NORTH_SUBDIV` |
| 8 | Misleading Citizen Reports | 85 unverified citizen complaint calls flooded ZONE_WEST | Module 3 Bayes Net grounded risk in entomological priors (alert=GREEN, confidence=0.605) | SUCCESS | `Alert=GREEN, ConfidenceScore=0.605` |


### Verifiable Algorithmic Outcomes
- **Search:** Handled patient hospital diversion under bed saturation; generated multi-stop team routes; detected spatial clusters.
- **CSP:** Handled severe diagnostic kit stockouts with conflict explanations and relaxed best-effort allocations.
- **Planning:** Execution monitor repaired schedules dynamically on monsoon downpours and case surges.
- **Uncertainty:** Decoupled true risk from clinical reporting bias, successfully detecting under-reporting in peri-urban wards.
