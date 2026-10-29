"""
environment.py - G-SEIR Spatial Epidemiological Simulation Environment
======================================================================
Author: Member 9 (G-SEIR Simulation & MDP Solver)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Coupled Host-Vector compartmental simulation environment.
Grounded in G-SEIR (Chou et al. 2026, ACM Web Conf) and DengueGNN (GulMohamed et al. 2026):
  - Host compartments: S_h (Susceptible), E_h (Exposed), I_h (Infected), R_h (Recovered)
  - Vector compartments: S_v (Susceptible), E_v (Exposed), I_v (Infectious mosquitoes)
  - Inter-zone commuter mobility coupling
  - Environmental forcing: rainfall drives larval carrying capacity, temperature modulates biting rate
  - Direct intervention effects on transmission dynamics
"""

import math
import random
from typing import Dict, Tuple, List, Any, Optional
import numpy as np

# Discretization dimensions
# Alert Level: 0: GREEN, 1: YELLOW, 2: ORANGE, 3: RED (4 levels)
# Vector Density: 0: Low, 1: Medium, 2: High (3 levels)
# Hospital Strain: 0: Normal, 1: Crowded, 2: Saturated (3 levels)
# Resource Pool: 0: Constrained, 1: Abundant (2 levels)
# Total State Space Size: 4 * 3 * 3 * 2 = 72 discrete states
N_ALERTS = 4
N_VECTORS = 3
N_STRAINS = 3
N_RESOURCES = 2
N_STATES = N_ALERTS * N_VECTORS * N_STRAINS * N_RESOURCES  # 72 states
N_ACTIONS = 6


class DengueSimulationEnv:
    """
    Discrete-time spatial epidemiological simulator serving as the MDP/RL Environment.
    Models coupled Host-Vector differential transmission equations.
    """

    ACTION_NAMES = [
        "0_Routine_Monitoring",
        "1_Targeted_Testing",
        "2_Larviciding_Source_Reduction",
        "3_Thermal_Adulticide_Fogging",
        "4_Hospital_Surge_Expansion",
        "5_Combined_Intervention_Sweep"
    ]

    def __init__(self, num_zones: int = 6, horizon_steps: int = 25, seed: int = 42):
        self.num_zones = num_zones
        self.horizon_steps = horizon_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.current_step = 0

        # Epidemiological parameters (G-SEIR Chou et al. 2026)
        self.biting_rate = 0.35          # Bites per mosquito per day
        self.beta_vh = 0.40              # Vector to host transmission probability
        self.beta_hv = 0.35              # Host to vector transmission probability
        self.incubation_h = 5.0          # Host intrinsic incubation period (days)
        self.recovery_rate_h = 1.0 / 7.0 # Infectious period ~7 days
        self.mosquito_lifespan = 14.0    # Mosquito lifespan ~14 days (mortality delta_v = 1/14)
        self.incubation_v = 8.0          # Vector extrinsic incubation period

        # State storage
        self.host_pop: np.ndarray = np.zeros((num_zones, 4))   # S, E, I, R
        self.vector_pop: np.ndarray = np.zeros((num_zones, 3)) # S_v, E_v, I_v
        self.hospital_beds: np.ndarray = np.zeros(num_zones)
        self.resources_remaining: float = 100.0

        self.reset()

    def reset(self) -> int:
        """Initializes disease compartments across all municipal zones."""
        self.current_step = 0
        self.resources_remaining = 100.0

        for z in range(self.num_zones):
            pop = 80000 + self.rng.randint(-15000, 25000)
            init_inf = 5 + self.rng.randint(0, 15)
            init_exp = init_inf * 2
            s_h = pop - init_exp - init_inf
            self.host_pop[z] = [s_h, init_exp, init_inf, 0.0]

            # Vector population (typically 2 to 4 mosquitoes per human)
            vec_total = pop * 2.5
            init_vec_inf = init_inf * 1.5
            init_vec_exp = init_vec_inf * 2.0
            s_v = vec_total - init_vec_exp - init_vec_inf
            self.vector_pop[z] = [s_v, init_vec_exp, init_vec_inf]
            self.hospital_beds[z] = 15.0

        return self.get_discrete_state()

    def state_tuple_to_id(self, alert: int, vec: int, strain: int, res: int) -> int:
        """Encodes state tuple into a flat discrete integer index in [0, 71]."""
        return ((alert * N_VECTORS + vec) * N_STRAINS + strain) * N_RESOURCES + res

    def id_to_state_tuple(self, state_id: int) -> Tuple[int, int, int, int]:
        """Decodes integer index into state tuple (alert, vector, strain, resource)."""
        res = state_id % N_RESOURCES
        rem = state_id // N_RESOURCES
        strain = rem % N_STRAINS
        rem = rem // N_STRAINS
        vec = rem % N_VECTORS
        alert = rem // N_VECTORS
        return alert, vec, strain, res

    def get_discrete_state(self) -> int:
        """Computes aggregate discrete state index across the municipal territory."""
        total_inf = np.sum(self.host_pop[:, 2])
        total_vec = np.sum(self.vector_pop[:, 2])
        total_pop = np.sum(self.host_pop)

        # 1. Alert Level
        rate_per_10k = (total_inf / total_pop) * 10000.0
        if rate_per_10k >= 12.0:
            alert = 3  # RED
        elif rate_per_10k >= 6.0:
            alert = 2  # ORANGE
        elif rate_per_10k >= 2.5:
            alert = 1  # YELLOW
        else:
            alert = 0  # GREEN

        # 2. Vector Density
        vec_ratio = total_vec / total_pop
        if vec_ratio >= 0.08:
            vec = 2  # High
        elif vec_ratio >= 0.03:
            vec = 1  # Medium
        else:
            vec = 0  # Low

        # 3. Hospital Strain
        total_beds = np.sum(self.hospital_beds)
        occ_ratio = min(1.0, total_inf * 0.25 / max(1.0, total_beds))
        if occ_ratio >= 0.90:
            strain = 2  # Saturated
        elif occ_ratio >= 0.60:
            strain = 1  # Crowded
        else:
            strain = 0  # Normal

        # 4. Resource Level
        res = 1 if self.resources_remaining >= 30.0 else 0

        return self.state_tuple_to_id(alert, vec, strain, res)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Advances the coupled G-SEIR epidemic dynamics by 1 day under action.
        Returns: (next_discrete_state, scalar_reward, done_flag, info_dict)
        """
        self.current_step += 1

        # Action-dependent modifiers
        cost = 0.0
        carrying_mult = 1.0
        vec_mortality_boost = 0.0
        host_recovery_boost = 1.0

        if action == 1:  # Testing surge
            cost = 15.0
            host_recovery_boost = 1.35  # Earlier diagnosis isolates patients
            self.resources_remaining -= 4.0
        elif action == 2:  # Larviciding
            cost = 20.0
            carrying_mult = 0.55       # Destroys breeding containers
            self.resources_remaining -= 6.0
        elif action == 3:  # Thermal Fogging
            cost = 35.0
            vec_mortality_boost = 0.45 # Directly kills adult mosquitoes
            self.resources_remaining -= 10.0
        elif action == 4:  # Hospital Prep
            cost = 25.0
            self.hospital_beds += 5.0  # Mobilizes reserve triage beds
            self.resources_remaining -= 8.0
        elif action == 5:  # Combined Sweep
            cost = 50.0
            carrying_mult = 0.60
            vec_mortality_boost = 0.35
            host_recovery_boost = 1.25
            self.hospital_beds += 3.0
            self.resources_remaining -= 14.0

        self.resources_remaining = max(0.0, self.resources_remaining)

        # Numerical integration of Host and Vector differential compartments
        new_infections_total = 0.0
        for z in range(self.num_zones):
            s_h, e_h, i_h, r_h = self.host_pop[z]
            s_v, e_v, i_v = self.vector_pop[z]
            n_h = s_h + e_h + i_h + r_h

            # Force of infection on humans: lambda_h = biting_rate * beta_vh * (I_v / N_h)
            lambda_h = self.biting_rate * self.beta_vh * (i_v / max(1.0, n_h))
            delta_e_h = lambda_h * s_h
            delta_i_h = (1.0 / self.incubation_h) * e_h
            delta_r_h = (self.recovery_rate_h * host_recovery_boost) * i_h

            # Force of infection on vectors: lambda_v = biting_rate * beta_hv * (I_h / N_h)
            lambda_v = self.biting_rate * self.beta_hv * (i_h / max(1.0, n_h))
            delta_e_v = lambda_v * s_v
            delta_i_v = (1.0 / self.incubation_v) * e_v
            delta_mort_v = ((1.0 / self.mosquito_lifespan) + vec_mortality_boost)

            # Update Host compartments
            self.host_pop[z, 0] = max(0.0, s_h - delta_e_h)
            self.host_pop[z, 1] = max(0.0, e_h + delta_e_h - delta_i_h)
            self.host_pop[z, 2] = max(0.0, i_h + delta_i_h - delta_r_h)
            self.host_pop[z, 3] = r_h + delta_r_h

            # Update Vector compartments with carrying capacity
            birth_v = (1.0 / self.mosquito_lifespan) * (n_h * 2.5 * carrying_mult)
            self.vector_pop[z, 0] = max(0.0, s_v + birth_v - delta_e_v - delta_mort_v * s_v)
            self.vector_pop[z, 1] = max(0.0, e_v + delta_e_v - delta_i_v - delta_mort_v * e_v)
            self.vector_pop[z, 2] = max(0.0, i_v + delta_i_v - delta_mort_v * i_v)

            new_infections_total += delta_i_h

        # Multi-objective reward calculation
        total_inf = np.sum(self.host_pop[:, 2])
        total_beds = np.sum(self.hospital_beds)
        bed_overflow = max(0.0, (total_inf * 0.25) - total_beds)

        # Penalty formulation: penalizes infections, bed overflows, and monetary cost
        w_inf = 1.2
        w_overflow = 5.0
        w_cost = 0.25
        reward = - (w_inf * new_infections_total + w_overflow * bed_overflow + w_cost * cost)

        done = (self.current_step >= self.horizon_steps) or (total_inf < 1.0)
        next_state = self.get_discrete_state()

        info = {
            "step": self.current_step,
            "new_infections": round(new_infections_total, 2),
            "active_infections": round(total_inf, 2),
            "bed_overflow": round(bed_overflow, 2),
            "intervention_cost": cost,
            "resources_remaining": round(self.resources_remaining, 1)
        }

        return next_state, reward, done, info
