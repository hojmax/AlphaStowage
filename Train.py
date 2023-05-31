import torch
import torch.nn as nn
import torch.optim as optim
from FloodEnv import FloodEnv
from NeuralNetwork import NeuralNetwork
import Node
import numpy as np
import wandb


def loss_fn(pred_values, values, pred_probs, probs):
    value_error = torch.mean((values - pred_values) ** 2)
    cross_entropy = -torch.mean(torch.sum(probs * torch.log(pred_probs), dim=1))
    scaling = 0.5
    loss = scaling * value_error + cross_entropy
    return loss


def train_network(network, data, batch_size, n_batches):
    if len(data) < batch_size:
        batch_size = len(data)

    sum_loss = 0

    for _ in range(n_batches):
        batch_indices = np.random.choice(len(all_data), batch_size, replace=False)
        batch = [all_data[i] for i in batch_indices]
        state_batch = torch.stack([x[0].squeeze(0) for x in batch])
        prob_batch = torch.stack([torch.tensor(x[1]) for x in batch])
        value_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32)

        pred_prob_batch, pred_value_batch = network(state_batch)
        loss = loss_fn(pred_value_batch, value_batch, pred_prob_batch, prob_batch)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        sum_loss += loss.item()

    return sum_loss / n_batches


# Training parameters
l2_weight_reg = 1e-4
batch_size = 8
batches_per_episode = 16
learning_rate = 1e-3
training_iterations = 10000
width = 5
height = 5
n_colors = 3
nn_blocks = 3
c_puct = 1
temperature = 1
search_iterations = 100
all_data = []
max_data = 1e5

config = {
    "l2_weight_reg": l2_weight_reg,
    "batch_size": batch_size,
    "batches_per_episode": batches_per_episode,
    "learning_rate": learning_rate,
    "training_iterations": training_iterations,
    "width": width,
    "height": height,
    "n_colors": n_colors,
    "nn_blocks": nn_blocks,
    "c_puct": c_puct,
    "temperature": temperature,
    "search_iterations": search_iterations,
    "max_data": max_data,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NeuralNetwork(n_colors=n_colors, width=width, height=height, n_blocks=nn_blocks)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_weight_reg)

wandb.init(
    entity="hojmax",
    project="bachelor",
    config=config,
)

for i in range(training_iterations):
    episode_data = []
    env = FloodEnv(width, height, n_colors)
    while not env.is_terminal():
        probabilities = Node.alphago_zero_search(
            env, net, search_iterations, c_puct, temperature
        )
        episode_data.append((env.get_tensor_state(), probabilities))
        action = np.random.choice(env.n_colors, p=probabilities)
        env.step(action)

    value = env.value

    for state, probabilities in episode_data:
        all_data.append((state, probabilities, value))

    if len(all_data) > max_data:
        all_data = all_data[-max_data:]

    avg_loss = train_network(net, all_data, batch_size, batches_per_episode)

    wandb.log({"loss": avg_loss, "value": value, "episode": i})
