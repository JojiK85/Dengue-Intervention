"""
baseline_comparison.py - Public Health Baseline Policies & RL Evaluation
========================================================================
Author: Member 10 (RL Agents, Baselines & Benchmarking)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Compares learned RL/MDP policies against three classical public-health baselines:
1. No-Intervention Policy: Routine passive monitoring only.
2. Random Policy: Stochastic action selection.
3. Greedy Highest-Reported-Cases Heuristic: Immediate adulticide spraying if cases surge.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
import numpy as np
import pandas as pd

from src.core.utils import get_outputs_dir, setup_logger
from .environment import DengueSimulationEnv, N_ACTIONS
from .mdp_solver import ModelBasedMDPSolver
from .q_learning import TabularQLearningAgent
from .sarsa import SARSAAgent

logger = setup_logger("baseline_comparison")


def policy_no_intervention(state_id: int) -> int:
    """Always selects action 0 (Routine Monitoring)."""
    return 0


def policy_random(state_id: int) -> int:
    """Selects a random legal action uniformly."""
    return np.random.randint(0, N_ACTIONS)


def policy_greedy_heuristic(state_id: int) -> int:
    """
    Public-health heuristic:
    If alert is RED or ORANGE, deploy adulticide spraying (action 3).
    If vector density is High, deploy larviciding (action 2).
    Otherwise, maintain routine monitoring (action 0).
    """
    res = state_id % 2
    rem = state_id // 2
    strain = rem % 3
    rem = rem // 3
    vec = rem % 3
    alert = rem // 3

    if alert >= 2:
        return 3  # Spraying
    elif vec >= 2:
        return 2  # Larviciding
    elif strain >= 2:
        return 4  # Hospital prep
    return 0


def evaluate_policy(
    policy: Union[np.ndarray, Callable[[int], int]],
    num_episodes: int = 30,
    horizon: int = 25,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluates a policy over multiple test episodes and returns mean metrics.
    """
    env = DengueSimulationEnv(num_zones=6, horizon_steps=horizon, seed=seed)
    total_rewards = []
    total_infections = []
    total_overflows = []
    total_costs = []

    for ep in range(num_episodes):
        s = env.reset()
        ep_rew = 0.0
        ep_inf = 0.0
        ep_over = 0.0
        ep_cost = 0.0
        done = False

        while not done:
            if isinstance(policy, np.ndarray):
                a = int(policy[s])
            else:
                a = int(policy(s))

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
    """
    Runs full training and comparative benchmark across all agents and baselines.
    Exports results to CSV and JSON.
    """
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

    # Evaluate all 6 approaches under identical testing environments
    logger.info("Evaluating policies across 30 test episodes...")
    results = {}
    results["No_Intervention"] = evaluate_policy(policy_no_intervention)
    results["Random_Policy"] = evaluate_policy(policy_random)
    results["Greedy_Heuristic"] = evaluate_policy(policy_greedy_heuristic)
    results["MDP_Value_Iteration"] = evaluate_policy(mdp_policy)
    results["Q_Learning_Agent"] = evaluate_policy(q_agent.get_learned_policy())
    results["SARSA_Agent"] = evaluate_policy(sarsa_agent.get_learned_policy())

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
