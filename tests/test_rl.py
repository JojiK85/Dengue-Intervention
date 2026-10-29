"""
test_rl.py - Automated Unit Tests for Module 4 (G-SEIR Environment & RL/MDP)
============================================================================
Author: Member 9 (G-SEIR Environment) & Member 10 (RL Agents)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar
"""

import pytest
import numpy as np
from src.rl.environment import DengueSimulationEnv, N_STATES, N_ACTIONS
from src.rl.mdp_solver import ModelBasedMDPSolver
from src.rl.q_learning import TabularQLearningAgent
from src.rl.sarsa import SARSAAgent


def test_g_seir_compartment_conservation():
    env = DengueSimulationEnv(num_zones=4, horizon_steps=10, seed=42)
    s = env.reset()
    total_pop_start = np.sum(env.host_pop)

    # Step through simulation
    for _ in range(5):
        _, _, done, _ = env.step(1)
        if done:
            break

    total_pop_end = np.sum(env.host_pop)
    # Total human population must be conserved (closed urban cohort S+E+I+R)
    assert pytest.approx(total_pop_start, 1e-2) == total_pop_end


def test_action_impact_on_mosquito_mortality():
    env_routine = DengueSimulationEnv(num_zones=4, horizon_steps=5, seed=42)
    env_spray = DengueSimulationEnv(num_zones=4, horizon_steps=5, seed=42)

    env_routine.reset()
    env_spray.reset()

    # Step routine monitoring vs chemical spraying
    env_routine.step(0)
    env_spray.step(3)

    vec_routine = np.sum(env_routine.vector_pop)
    vec_spray = np.sum(env_spray.vector_pop)

    # Adulticide spraying must reduce vector population significantly
    assert vec_spray < vec_routine


def test_discrete_state_encoding_decoding():
    env = DengueSimulationEnv(seed=42)
    # Test all possible discrete states
    for alert in range(4):
        for vec in range(3):
            for strain in range(3):
                for res in range(2):
                    sid = env.state_tuple_to_id(alert, vec, strain, res)
                    assert 0 <= sid < N_STATES
                    a_dec, v_dec, s_dec, r_dec = env.id_to_state_tuple(sid)
                    assert (a_dec, v_dec, s_dec, r_dec) == (alert, vec, strain, res)


def test_mdp_value_iteration_convergence():
    solver = ModelBasedMDPSolver(num_states=10, num_actions=4)
    # Simple transition model
    for s in range(10):
        for a in range(4):
            solver.P[s, a, min(9, s + 1)] = 0.8
            solver.P[s, a, max(0, s - 1)] = 0.2
            solver.R[s, a] = -float(s)

    V, policy, it_count = solver.value_iteration(max_iterations=200)
    assert it_count < 200
    assert len(V) == 10
    assert len(policy) == 10


def test_q_learning_and_sarsa_training():
    env = DengueSimulationEnv(num_zones=4, horizon_steps=10, seed=42)
    q_agent = TabularQLearningAgent(seed=42)
    q_rewards = q_agent.train(env, num_episodes=15)
    assert len(q_rewards) == 15
    assert q_agent.epsilon < 1.0  # Verify epsilon decay

    sarsa_agent = SARSAAgent(seed=42)
    sarsa_rewards = sarsa_agent.train(env, num_episodes=15)
    assert len(sarsa_rewards) == 15
    assert sarsa_agent.epsilon < 1.0
