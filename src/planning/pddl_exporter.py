"""
pddl_exporter.py - Formal PDDL Domain and Problem Exporter
==========================================================
Author: Member 5 (Causal Planning & Formal PDDL)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Generates formal Planning Domain Definition Language (PDDL 2.1) specifications
for the municipal dengue intervention domain.
"""

from pathlib import Path
from typing import List, Dict, Any

from src.core.utils import get_project_root


def generate_pddl_domain() -> str:
    """Returns the formal PDDL domain string."""
    return """;;; Adaptive Dengue Surveillance and Intervention Coordinator
;;; Course Project: CS4L002 Artificial Intelligence, IIT Bhubaneswar
;;; Formal STRIPS / PDDL Domain Definition

(define (domain dengue-intervention)
  (:requirements :strips :typing)
  (:types zone)

  (:predicates
    (zone-active ?z - zone)
    (surveillance-active ?z - zone)
    (breeding-sites-identified ?z - zone)
    (breeding-containers-cleared ?z - zone)
    (vector-density-confirmed ?z - zone)
    (adult-vectors-reduced ?z - zone)
    (outbreak-confirmed ?z - zone)
    (surge-beds-mobilized ?z - zone)
    (transmission-contained ?z - zone)
  )

  (:action surveillance
    :parameters (?z - zone)
    :precondition (zone-active ?z)
    :effect (surveillance-active ?z)
  )

  (:action field-inspection
    :parameters (?z - zone)
    :precondition (surveillance-active ?z)
    :effect (and (breeding-sites-identified ?z)
                 (vector-density-confirmed ?z))
  )

  (:action source-reduction
    :parameters (?z - zone)
    :precondition (breeding-sites-identified ?z)
    :effect (and (breeding-containers-cleared ?z)
                 (not (breeding-sites-identified ?z)))
  )

  (:action chemical-spraying
    :parameters (?z - zone)
    :precondition (vector-density-confirmed ?z)
    :effect (and (adult-vectors-reduced ?z)
                 (not (vector-density-confirmed ?z)))
  )

  (:action laboratory-testing
    :parameters (?z - zone)
    :precondition (surveillance-active ?z)
    :effect (outbreak-confirmed ?z)
  )

  (:action hospital-preparation
    :parameters (?z - zone)
    :precondition (outbreak-confirmed ?z)
    :effect (surge-beds-mobilized ?z)
  )

  (:action follow-up-assessment
    :parameters (?z - zone)
    :precondition (and (adult-vectors-reduced ?z)
                       (breeding-containers-cleared ?z))
    :effect (transmission-contained ?z)
  )
)
"""


def generate_pddl_problem(zones: List[str]) -> str:
    """Returns the formal PDDL problem string for target zones."""
    zone_objs = " ".join(f"{z} - zone" for z in zones)
    inits = "\n    ".join(f"(zone-active {z})" for z in zones)
    goals = "\n    ".join(f"(transmission-contained {z})" for z in zones)

    return f""";;; Dengue Intervention Problem Instance
;;; Target Wards: {', '.join(zones)}

(define (problem dengue-outbreak-containment)
  (:domain dengue-intervention)
  (:objects
    {zone_objs}
  )

  (:init
    {inits}
  )

  (:goal
    (and
      {goals}
    )
  )
)
"""


def export_pddl_files(target_zones: List[str], pddl_dir: Path = None) -> Dict[str, Path]:
    """Writes domain.pddl and problem.pddl to the pddl/ directory."""
    if pddl_dir is None:
        pddl_dir = get_project_root() / "pddl"
    pddl_dir.mkdir(parents=True, exist_ok=True)

    domain_path = pddl_dir / "domain.pddl"
    problem_path = pddl_dir / "problem.pddl"

    with open(domain_path, "w", encoding="utf-8") as f:
        f.write(generate_pddl_domain())

    with open(problem_path, "w", encoding="utf-8") as f:
        f.write(generate_pddl_problem(target_zones))

    return {"domain": domain_path, "problem": problem_path}
