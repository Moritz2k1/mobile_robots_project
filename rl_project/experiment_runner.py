# experiment_runner.py

import os
import json
import torch.nn as nn
from train_rl import train_dqn, train_ddpg

# 🔹 Optimal hyperparameters
hyperparameter_set = {
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "batch_size": 64,
    "hidden_layers": 2,
    "hidden_units": 128,
    "activation_fn": nn.ReLU
}

# 🔹 Experiment execution
def run_experiments():
    os.makedirs("experiment_results", exist_ok=True)

    # DQN Experiment
    print("Running DQN with optimal hyperparameters...")
    results_dqn = train_dqn(**hyperparameter_set)
    with open("experiment_results/dqn_optimal.json", "w") as f:
        json.dump(results_dqn, f)

    # DDPG Experiment
    print("Running DDPG with optimal hyperparameters...")
    results_ddpg = train_ddpg(**hyperparameter_set)
    with open("experiment_results/ddpg_optimal.json", "w") as f:
        json.dump(results_ddpg, f)

if __name__ == "__main__":
    run_experiments()
