# plot_optimal_results.py

import json
import matplotlib.pyplot as plt

# 🔹 Load the results
with open('experiment_results/dqn_optimal.json', 'r') as f:
    dqn_results = json.load(f)

with open('experiment_results/ddpg_optimal.json', 'r') as f:
    ddpg_results = json.load(f)

# 🔹 Plot the results
plt.plot(dqn_results["episode_rewards"], label='DQN Optimal')
plt.plot(ddpg_results["episode_rewards"], label='DDPG Optimal')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Optimal DQN vs DDPG Performance')
plt.legend()
plt.show()
