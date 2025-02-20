# train_rl.py

import torch
import torch.nn as nn
import torch.optim as optim

# 🔹 Helper function to create neural network dynamically
def create_network(input_size, output_size, hidden_layers=2, hidden_units=128, activation_fn=nn.ReLU):
    layers = [nn.Linear(input_size, hidden_units), activation_fn()]
    for _ in range(hidden_layers - 1):
        layers.append(nn.Linear(hidden_units, hidden_units))
        layers.append(activation_fn())
    layers.append(nn.Linear(hidden_units, output_size))
    return nn.Sequential(*layers)

# 🔹 DQN Model Training Function
def train_dqn(learning_rate=0.0005, gamma=0.99, batch_size=64, hidden_layers=2, hidden_units=128, activation_fn=nn.ReLU):
    print(f"Training DQN | LR: {learning_rate}, Gamma: {gamma}, Batch: {batch_size}, Layers: {hidden_layers}, Units: {hidden_units}")
    model = create_network(4, 2, hidden_layers, hidden_units, activation_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    rewards = [i * 10 for i in range(1, 21)]  # Example rewards for testing
    return {"episode_rewards": rewards}

# 🔹 DDPG Model Training Function
def train_ddpg(learning_rate=0.0005, gamma=0.99, batch_size=64, hidden_layers=2, hidden_units=128, activation_fn=nn.ReLU):
    print(f"Training DDPG | LR: {learning_rate}, Gamma: {gamma}, Batch: {batch_size}, Layers: {hidden_layers}, Units: {hidden_units}")
    actor = create_network(4, 2, hidden_layers, hidden_units, activation_fn)
    critic = create_network(4 + 2, 1, hidden_layers, hidden_units, activation_fn)
    optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    rewards = [i * 5 for i in range(1, 21)]  # Example rewards for testing
    return {"episode_rewards": rewards}
