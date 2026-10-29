"""
mdp_solver.py - Model-Based MDP Solver (Value Iteration & Policy Iteration)
===========================================================================
Author: Member 9 (G-SEIR Simulation & MDP Solver)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Solves the Bellman Optimality Equations for the discretized MDP:
1. Value Iteration:
     V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [ R(s,a) + gamma * V_k(s') ]
2. Policy Iteration:
     Alternates between Policy Evaluation and Policy Improvement.
Grounded in AIMA Chapter 17 (Making Complex Decisions).
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np

from .environment import N_STATES, N_ACTIONS, DengueSimulationEnv


class ModelBasedMDPSolver:
    """Model-based dynamic programming solver using Bellman Optimality equations."""

    def __init__(self, num_states: int = N_STATES, num_actions: int = N_ACTIONS, gamma: float = 0.90, theta: float = 1e-4):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.theta = theta

        # Model matrices: P(s' | s, a) and R(s, a)
        self.P = np.zeros((num_states, num_actions, num_states))
        self.R = np.zeros((num_states, num_actions))

    def estimate_model_from_env(self, num_episodes: int = 150, horizon: int = 25) -> None:
        """
        Samples the simulation environment across episodes to build empirical
        transition probability and expected reward models.
        """
        counts = np.zeros((self.num_states, self.num_actions, self.num_states))
        rewards = np.zeros((self.num_states, self.num_actions))
        action_counts = np.zeros((self.num_states, self.num_actions))

        env = DengueSimulationEnv(num_zones=6, horizon_steps=horizon, seed=42)

        for ep in range(num_episodes):
            s = env.reset()
            done = False
            while not done:
                a = env.rng.randint(0, self.num_actions)
                s_next, r, done, _ = env.step(a)
                counts[s, a, s_next] += 1
                rewards[s, a] += r
                action_counts[s, a] += 1
                s = s_next

        # Normalize counts into probabilities
        for s in range(self.num_states):
            for a in range(self.num_actions):
                tot = action_counts[s, a]
                if tot > 0:
                    self.P[s, a, :] = counts[s, a, :] / tot
                    self.R[s, a] = rewards[s, a] / tot
                else:
                    # Self-loop default
                    self.P[s, a, s] = 1.0
                    self.R[s, a] = -10.0

    def value_iteration(self, max_iterations: int = 500) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Bellman Value Iteration until convergence ||V_{k+1} - V_k||_inf < theta.
        Returns: (V_star, optimal_policy, iterations_count)
        """
        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states, dtype=int)

        for it in range(1, max_iterations + 1):
            delta = 0.0
            new_V = np.zeros(self.num_states)

            for s in range(self.num_states):
                # Q(s, a) = R(s, a) + gamma * sum_{s'} P(s'|s,a) * V(s')
                q_vals = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    q_vals[a] = self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V)

                best_val = np.max(q_vals)
                delta = max(delta, abs(best_val - V[s]))
                new_V[s] = best_val
                policy[s] = int(np.argmax(q_vals))

            V = new_V
            if delta < self.theta:
                return V, policy, it

        return V, policy, max_iterations

    def policy_iteration(self, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Policy Iteration: alternates between Policy Evaluation and Policy Improvement.
        """
        policy = np.zeros(self.num_states, dtype=int)
        V = np.zeros(self.num_states)

        for it in range(1, max_iterations + 1):
            # 1. Policy Evaluation
            for _ in range(50):
                for s in range(self.num_states):
                    a = policy[s]
                    V[s] = self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V)

            # 2. Policy Improvement
            policy_stable = True
            for s in range(self.num_states):
                old_action = policy[s]
                q_vals = [
                    self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V)
                    for a in range(self.num_actions)
                ]
                new_action = int(np.argmax(q_vals))
                if new_action != old_action:
                    policy_stable = False
                policy[s] = new_action

            if policy_stable:
                return V, policy, it

        return V, policy, max_iterations
