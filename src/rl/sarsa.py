"""
sarsa.py - Model-Free On-Policy SARSA Agent
===========================================
Author: Member 10 (RL Agents, Baselines & Benchmarking)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Implements On-Policy Temporal Difference SARSA (State-Action-Reward-State-Action):
  Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
Contrasts with off-policy Q-Learning by using the actual next action taken under epsilon-greedy.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from .environment import N_STATES, N_ACTIONS, DengueSimulationEnv


class SARSAAgent:
    """On-Policy SARSA Agent learning intervention policies."""

    def __init__(
        self,
        num_states: int = N_STATES,
        num_actions: int = N_ACTIONS,
        alpha: float = 0.15,
        gamma: float = 0.90,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985,
        seed: int = 42
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.RandomState(seed)

        self.Q = np.zeros((num_states, num_actions))

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and self.rng.rand() < self.epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.Q[state, :]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> None:
        """On-policy SARSA update rule."""
        target = r if done else (r + self.gamma * self.Q[s_next, a_next])
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: DengueSimulationEnv, num_episodes: int = 250) -> List[float]:
        """Trains the SARSA agent across episodes."""
        rewards_history = []

        for ep in range(num_episodes):
            s = env.reset()
            a = self.select_action(s, evaluate=False)
            ep_reward = 0.0
            done = False

            while not done:
                s_next, r, done, _ = env.step(a)
                a_next = self.select_action(s_next, evaluate=False)
                self.update(s, a, r, s_next, a_next, done)
                s = s_next
                a = a_next
                ep_reward += r

            self.decay_epsilon()
            rewards_history.append(ep_reward)

        return rewards_history

    def get_learned_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
