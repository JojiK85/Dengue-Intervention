import json
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Callable, Union
import numpy as np
import pandas as pd

from src.city_state import get_outputs_dir, setup_logger

logger = setup_logger("module4_rl")

N_ALERTS = 4
N_VECTORS = 3
N_STRAINS = 3
N_RESOURCES = 2
N_STATES = N_ALERTS * N_VECTORS * N_STRAINS * N_RESOURCES  # 72 states
N_ACTIONS = 6

class DengueSimulationEnv:
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

        self.biting_rate = 0.35
        self.beta_vh = 0.40
        self.beta_hv = 0.35
        self.incubation_h = 5.0
        self.recovery_rate_h = 1.0 / 7.0
        self.mosquito_lifespan = 14.0
        self.incubation_v = 8.0

        self.host_pop: np.ndarray = np.zeros((num_zones, 4))
        self.vector_pop: np.ndarray = np.zeros((num_zones, 3))
        self.hospital_beds: np.ndarray = np.zeros(num_zones)
        self.resources_remaining: float = 100.0
        self.reset()

    def reset(self) -> int:
        self.current_step = 0
        self.resources_remaining = 100.0
        for z in range(self.num_zones):
            pop = 80000 + self.rng.randint(-15000, 25000)
            init_inf = 5 + self.rng.randint(0, 15)
            init_exp = init_inf * 2
            s_h = pop - init_exp - init_inf
            self.host_pop[z] = [s_h, init_exp, init_inf, 0.0]

            vec_total = pop * 2.5
            init_vec_inf = init_inf * 1.5
            init_vec_exp = init_vec_inf * 2.0
            s_v = vec_total - init_vec_exp - init_vec_inf
            self.vector_pop[z] = [s_v, init_vec_exp, init_vec_inf]
            self.hospital_beds[z] = 15.0
        return self.get_discrete_state()

    def state_tuple_to_id(self, alert: int, vec: int, strain: int, res: int) -> int:
        return ((alert * N_VECTORS + vec) * N_STRAINS + strain) * N_RESOURCES + res

    def id_to_state_tuple(self, state_id: int) -> Tuple[int, int, int, int]:
        res = state_id % N_RESOURCES
        rem = state_id // N_RESOURCES
        strain = rem % N_STRAINS
        rem = rem // N_STRAINS
        vec = rem % N_VECTORS
        alert = rem // N_VECTORS
        return alert, vec, strain, res

    def get_discrete_state(self) -> int:
        total_inf = np.sum(self.host_pop[:, 2])
        total_vec = np.sum(self.vector_pop[:, 2])
        total_pop = np.sum(self.host_pop)

        rate_per_10k = (total_inf / total_pop) * 10000.0
        if rate_per_10k >= 12.0:
            alert = 3
        elif rate_per_10k >= 6.0:
            alert = 2
        elif rate_per_10k >= 2.5:
            alert = 1
        else:
            alert = 0

        vec_ratio = total_vec / total_pop
        if vec_ratio >= 0.08:
            vec = 2
        elif vec_ratio >= 0.03:
            vec = 1
        else:
            vec = 0

        total_beds = np.sum(self.hospital_beds)
        occ_ratio = min(1.0, total_inf * 0.25 / max(1.0, total_beds))
        if occ_ratio >= 0.90:
            strain = 2
        elif occ_ratio >= 0.60:
            strain = 1
        else:
            strain = 0

        res = 1 if self.resources_remaining >= 30.0 else 0
        return self.state_tuple_to_id(alert, vec, strain, res)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        self.current_step += 1
        cost = 0.0
        carrying_mult = 1.0
        vec_mortality_boost = 0.0
        host_recovery_boost = 1.0

        if action == 1:
            cost = 15.0
            host_recovery_boost = 1.35
            self.resources_remaining -= 4.0
        elif action == 2:
            cost = 20.0
            carrying_mult = 0.55
            self.resources_remaining -= 6.0
        elif action == 3:
            cost = 35.0
            vec_mortality_boost = 0.45
            self.resources_remaining -= 10.0
        elif action == 4:
            cost = 25.0
            self.hospital_beds += 5.0
            self.resources_remaining -= 8.0
        elif action == 5:
            cost = 50.0
            carrying_mult = 0.60
            vec_mortality_boost = 0.35
            host_recovery_boost = 1.25
            self.hospital_beds += 3.0
            self.resources_remaining -= 14.0

        self.resources_remaining = max(0.0, self.resources_remaining)
        new_infections_total = 0.0

        for z in range(self.num_zones):
            s_h, e_h, i_h, r_h = self.host_pop[z]
            s_v, e_v, i_v = self.vector_pop[z]
            n_h = s_h + e_h + i_h + r_h

            lambda_h = self.biting_rate * self.beta_vh * (i_v / max(1.0, n_h))
            delta_e_h = lambda_h * s_h
            delta_i_h = (1.0 / self.incubation_h) * e_h
            delta_r_h = (self.recovery_rate_h * host_recovery_boost) * i_h

            lambda_v = self.biting_rate * self.beta_hv * (i_h / max(1.0, n_h))
            delta_e_v = lambda_v * s_v
            delta_i_v = (1.0 / self.incubation_v) * e_v
            delta_mort_v = ((1.0 / self.mosquito_lifespan) + vec_mortality_boost)

            self.host_pop[z, 0] = max(0.0, s_h - delta_e_h)
            self.host_pop[z, 1] = max(0.0, e_h + delta_e_h - delta_i_h)
            self.host_pop[z, 2] = max(0.0, i_h + delta_i_h - delta_r_h)
            self.host_pop[z, 3] = r_h + delta_r_h

            birth_v = (1.0 / self.mosquito_lifespan) * (n_h * 2.5 * carrying_mult)
            self.vector_pop[z, 0] = max(0.0, s_v + birth_v - delta_e_v - delta_mort_v * s_v)
            self.vector_pop[z, 1] = max(0.0, e_v + delta_e_v - delta_i_v - delta_mort_v * e_v)
            self.vector_pop[z, 2] = max(0.0, i_v + delta_i_v - delta_mort_v * i_v)

            new_infections_total += delta_i_h

        total_inf = np.sum(self.host_pop[:, 2])
        total_beds = np.sum(self.hospital_beds)
        bed_overflow = max(0.0, (total_inf * 0.25) - total_beds)

        reward = - (1.2 * new_infections_total + 5.0 * bed_overflow + 0.25 * cost)
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

class ModelBasedMDPSolver:
    def __init__(self, num_states: int = N_STATES, num_actions: int = N_ACTIONS, gamma: float = 0.90, theta: float = 1e-4):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.theta = theta
        self.P = np.zeros((num_states, num_actions, num_states))
        self.R = np.zeros((num_states, num_actions))

    def estimate_model_from_env(self, num_episodes: int = 150, horizon: int = 25) -> None:
        counts = np.zeros((self.num_states, self.num_actions, self.num_states))
        rewards = np.zeros((self.num_states, self.num_actions))
        action_counts = np.zeros((self.num_states, self.num_actions))
        env = DengueSimulationEnv(num_zones=6, horizon_steps=horizon, seed=42)

        for _ in range(num_episodes):
            s = env.reset()
            done = False
            while not done:
                a = env.rng.randint(0, self.num_actions)
                s_next, r, done, _ = env.step(a)
                counts[s, a, s_next] += 1
                rewards[s, a] += r
                action_counts[s, a] += 1
                s = s_next

        for s in range(self.num_states):
            for a in range(self.num_actions):
                tot = action_counts[s, a]
                if tot > 0:
                    self.P[s, a, :] = counts[s, a, :] / tot
                    self.R[s, a] = rewards[s, a] / tot
                else:
                    self.P[s, a, s] = 1.0
                    self.R[s, a] = -10.0

    def value_iteration(self, max_iterations: int = 500) -> Tuple[np.ndarray, np.ndarray, int]:
        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states, dtype=int)

        for it in range(1, max_iterations + 1):
            delta = 0.0
            new_V = np.zeros(self.num_states)
            for s in range(self.num_states):
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
        policy = np.zeros(self.num_states, dtype=int)
        V = np.zeros(self.num_states)

        for it in range(1, max_iterations + 1):
            for _ in range(50):
                for s in range(self.num_states):
                    a = policy[s]
                    V[s] = self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V)

            policy_stable = True
            for s in range(self.num_states):
                old_action = policy[s]
                q_vals = [self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V) for a in range(self.num_actions)]
                new_action = int(np.argmax(q_vals))
                if new_action != old_action:
                    policy_stable = False
                policy[s] = new_action

            if policy_stable:
                return V, policy, it
        return V, policy, max_iterations

class TabularQLearningAgent:
    def __init__(
        self, num_states: int = N_STATES, num_actions: int = N_ACTIONS,
        alpha: float = 0.15, gamma: float = 0.90, epsilon: float = 1.0,
        epsilon_min: float = 0.05, epsilon_decay: float = 0.985, seed: int = 42
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
        if not evaluate and self.rng.rand() < self.epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.Q[state, :]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        target = r if done else (r + self.gamma * np.max(self.Q[s_next, :]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: DengueSimulationEnv, num_episodes: int = 250) -> List[float]:
        rewards_history = []
        for _ in range(num_episodes):
            s = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                a = self.select_action(s, evaluate=False)
                s_next, r, done, _ = env.step(a)
                self.update(s, a, r, s_next, done)
                s = s_next
                ep_reward += r
            self.decay_epsilon()
            rewards_history.append(ep_reward)
        return rewards_history

    def get_learned_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

class SARSAAgent:
    def __init__(
        self, num_states: int = N_STATES, num_actions: int = N_ACTIONS,
        alpha: float = 0.15, gamma: float = 0.90, epsilon: float = 1.0,
        epsilon_min: float = 0.05, epsilon_decay: float = 0.985, seed: int = 42
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
        if not evaluate and self.rng.rand() < self.epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.Q[state, :]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> None:
        target = r if done else (r + self.gamma * self.Q[s_next, a_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: DengueSimulationEnv, num_episodes: int = 250) -> List[float]:
        rewards_history = []
        for _ in range(num_episodes):
            s = env.reset()
            a = self.select_action(s, evaluate=False)
            ep_reward = 0.0
            done = False
            while not done:
                s_next, r, done, _ = env.step(a)
                a_next = self.select_action(s_next, evaluate=False)
                self.update(s, a, r, s_next, a_next, done)
                s, a = s_next, a_next
                ep_reward += r
            self.decay_epsilon()
            rewards_history.append(ep_reward)
        return rewards_history

    def get_learned_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

def policy_no_intervention(state_id: int) -> int:
    return 0

def policy_random(state_id: int) -> int:
    return int(np.random.randint(0, N_ACTIONS))

def policy_greedy_heuristic(state_id: int) -> int:
    res = state_id % 2
    rem = state_id // 2
    strain = rem % 3
    rem = rem // 3
    vec = rem % 3
    alert = rem // 3
    if alert >= 2:
        return 3
    elif vec >= 2:
        return 2
    elif strain >= 2:
        return 4
    return 0

def evaluate_policy(
    policy: Union[np.ndarray, Callable[[int], int]],
    num_episodes: int = 30,
    horizon: int = 25,
    seed: int = 42
) -> Dict[str, float]:
    env = DengueSimulationEnv(num_zones=6, horizon_steps=horizon, seed=seed)
    total_rewards, total_infections, total_overflows, total_costs = [], [], [], []

    for _ in range(num_episodes):
        s = env.reset()
        ep_rew, ep_inf, ep_over, ep_cost = 0.0, 0.0, 0.0, 0.0
        done = False
        while not done:
            a = int(policy[s]) if isinstance(policy, np.ndarray) else int(policy(s))
            s_next, r, done, info = env.step(a)
            ep_rew += r
            ep_inf += info["new_infections"]
            ep_over += info["bed_overflow"]
            ep_cost += info["intervention_cost"]
            s = s_next
        total_rewards.append(ep_rew)
        total_infections.append(ep_inf)
        total_overflows.append(ep_over)
        total_costs.append(ep_cost)

    return {
        "mean_episode_reward": round(float(np.mean(total_rewards)), 2),
        "std_episode_reward": round(float(np.std(total_rewards)), 2),
        "mean_infections": round(float(np.mean(total_infections)), 2),
        "mean_bed_overflow": round(float(np.mean(total_overflows)), 2),
        "mean_intervention_cost": round(float(np.mean(total_costs)), 2)
    }

def run_all_rl_experiments(output_dir: Optional[Path] = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = get_outputs_dir() / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training Model-Based MDP Solver (Value Iteration)...")
    mdp_solver = ModelBasedMDPSolver()
    mdp_solver.estimate_model_from_env(num_episodes=100)
    v_star, mdp_policy, it_count = mdp_solver.value_iteration()

    logger.info("Training Model-Free Tabular Q-Learning Agent...")
    env_train = DengueSimulationEnv(num_zones=6, horizon_steps=25, seed=42)
    q_agent = TabularQLearningAgent(seed=42)
    q_rewards = q_agent.train(env_train, num_episodes=250)

    logger.info("Training Model-Free SARSA Agent...")
    env_train_sarsa = DengueSimulationEnv(num_zones=6, horizon_steps=25, seed=42)
    sarsa_agent = SARSAAgent(seed=42)
    sarsa_rewards = sarsa_agent.train(env_train_sarsa, num_episodes=250)

    logger.info("Evaluating policies across 30 test episodes...")
    results = {
        "No_Intervention": evaluate_policy(policy_no_intervention),
        "Random_Policy": evaluate_policy(policy_random),
        "Greedy_Heuristic": evaluate_policy(policy_greedy_heuristic),
        "MDP_Value_Iteration": evaluate_policy(mdp_policy),
        "Q_Learning_Agent": evaluate_policy(q_agent.get_learned_policy()),
        "SARSA_Agent": evaluate_policy(sarsa_agent.get_learned_policy())
    }

    records = []
    for pol_name, metrics in results.items():
        records.append({
            "Policy": pol_name,
            "Mean_Reward": metrics["mean_episode_reward"],
            "Reward_Std": metrics["std_episode_reward"],
            "Mean_Infections": metrics["mean_infections"],
            "Mean_Bed_Overflow": metrics["mean_bed_overflow"],
            "Mean_Cost_INR": metrics["mean_intervention_cost"]
        })

    df = pd.DataFrame(records)
    csv_path = output_dir / "rl_policy_comparison.csv"
    df.to_csv(csv_path, index=False)

    json_path = output_dir / "rl_evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "policies": results,
            "mdp_iterations": it_count,
            "q_learning_final_epsilon": round(q_agent.epsilon, 4),
            "sarsa_final_epsilon": round(sarsa_agent.epsilon, 4)
        }, f, indent=2)

    logger.info(f"Saved RL benchmark comparison to {csv_path} and {json_path}")
    return df
