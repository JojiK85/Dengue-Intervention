"""
actions.py - STRIPS / PDDL Action Schemas for Public-Health Interventions
========================================================================
Author: Member 5 (Causal Planning & Formal PDDL)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Defines formal classical action schemas for all eight public-health actions:
  Action = <Name, Preconditions, AddList, DeleteList, Duration, ResourceRequirements>
Grounded in AIMA Chapter 11 (Classical Planning).
"""

from dataclasses import dataclass, field
from typing import Set, Dict, List, Any, Optional


@dataclass
class ActionSchema:
    """Classical STRIPS action operator representation."""
    name: str
    zone: str
    preconditions: Set[str] = field(default_factory=set)
    add_effects: Set[str] = field(default_factory=set)
    delete_effects: Set[str] = field(default_factory=set)
    duration_days: int = 1
    required_teams: int = 0
    required_kits: int = 0
    required_beds: int = 0
    cost_inr: float = 2000.0
    weather_constraint: Optional[str] = None  # e.g., "Dry" for outdoor spraying

    def is_applicable(self, current_state: Set[str], current_weather: str = "Clear") -> bool:
        """Returns True if all preconditions are satisfied and weather permits."""
        if not self.preconditions.issubset(current_state):
            return False
        if self.weather_constraint == "Dry" and current_weather.lower() in ["rain", "heavy_rain", "monsoon"]:
            return False
        return True

    def apply(self, current_state: Set[str]) -> Set[str]:
        """Applies add and delete lists to produce the successor state."""
        return (current_state - self.delete_effects) | self.add_effects

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "zone": self.zone,
            "preconditions": sorted(list(self.preconditions)),
            "add_effects": sorted(list(self.add_effects)),
            "delete_effects": sorted(list(self.delete_effects)),
            "duration_days": self.duration_days,
            "required_teams": self.required_teams,
            "required_kits": self.required_kits,
            "required_beds": self.required_beds,
            "cost_inr": self.cost_inr,
            "weather_constraint": self.weather_constraint
        }


def create_dengue_action_catalog(zone: str) -> Dict[str, ActionSchema]:
    """
    Creates the complete catalog of all 8 public-health actions for a specific zone:
    1. surveillance
    2. field_inspection
    3. mosquito_source_reduction
    4. spraying
    5. laboratory_testing
    6. public_awareness
    7. hospital_preparation
    8. follow_up_assessment
    """
    return {
        "surveillance": ActionSchema(
            name=f"surveillance_{zone}",
            zone=zone,
            preconditions={f"zone_active_{zone}"},
            add_effects={f"surveillance_active_{zone}", f"fever_cases_monitored_{zone}"},
            delete_effects=set(),
            duration_days=1,
            required_teams=0,
            required_kits=0,
            cost_inr=1500.0
        ),
        "field_inspection": ActionSchema(
            name=f"field_inspection_{zone}",
            zone=zone,
            preconditions={f"surveillance_active_{zone}"},
            add_effects={f"breeding_sites_identified_{zone}", f"vector_density_confirmed_{zone}"},
            delete_effects=set(),
            duration_days=1,
            required_teams=1,
            required_kits=0,
            cost_inr=4500.0
        ),
        "mosquito_source_reduction": ActionSchema(
            name=f"source_reduction_{zone}",
            zone=zone,
            preconditions={f"breeding_sites_identified_{zone}"},
            add_effects={f"breeding_containers_cleared_{zone}"},
            delete_effects={f"breeding_sites_identified_{zone}"},
            duration_days=2,
            required_teams=1,
            cost_inr=6000.0
        ),
        "spraying": ActionSchema(
            name=f"chemical_spraying_{zone}",
            zone=zone,
            preconditions={f"vector_density_confirmed_{zone}"},
            add_effects={f"adult_vectors_reduced_{zone}"},
            delete_effects={f"vector_density_confirmed_{zone}"},
            duration_days=1,
            required_teams=1,
            cost_inr=8500.0,
            weather_constraint="Dry"  # Heavy rainfall washes away insecticide
        ),
        "laboratory_testing": ActionSchema(
            name=f"laboratory_testing_{zone}",
            zone=zone,
            preconditions={f"fever_cases_monitored_{zone}"},
            add_effects={f"cases_serotyped_{zone}", f"outbreak_confirmed_{zone}"},
            delete_effects=set(),
            duration_days=1,
            required_kits=25,
            cost_inr=7000.0
        ),
        "public_awareness": ActionSchema(
            name=f"public_awareness_{zone}",
            zone=zone,
            preconditions={f"zone_active_{zone}"},
            add_effects={f"community_cooperation_active_{zone}"},
            delete_effects=set(),
            duration_days=1,
            cost_inr=3000.0
        ),
        "hospital_preparation": ActionSchema(
            name=f"hospital_preparation_{zone}",
            zone=zone,
            preconditions={f"outbreak_confirmed_{zone}"},
            add_effects={f"surge_beds_mobilized_{zone}", f"triage_active_{zone}"},
            delete_effects=set(),
            duration_days=1,
            required_beds=10,
            cost_inr=12000.0
        ),
        "follow_up_assessment": ActionSchema(
            name=f"follow_up_assessment_{zone}",
            zone=zone,
            preconditions={f"adult_vectors_reduced_{zone}", f"breeding_containers_cleared_{zone}"},
            add_effects={f"zone_transmission_contained_{zone}"},
            delete_effects=set(),
            duration_days=1,
            required_teams=1,
            cost_inr=3500.0
        )
    }
