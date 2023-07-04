import torch
import torch.nn as nn
import torch.optim as optim
from FloodEnv import FloodEnv
from NeuralNetwork import NeuralNetwork
import Node
import numpy as np
import wandb
import json


def loss_fn(pred_values, values, pred_probs, probs, value_scaling):
    value_error = torch.mean(torch.square(values - pred_values))
    cross_entropy = (
        -torch.sum(probs.flatten() * torch.log(pred_probs.flatten())) / probs.shape[0]
    )
    loss = value_scaling * value_error + cross_entropy
    return loss


def train_network(network, data, batch_size, n_batches, optimizer, value_scaling):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(data) < batch_size:
        batch_size = len(data)

    sum_loss = 0

    for _ in range(n_batches):
        batch_indices = np.random.choice(len(data), batch_size, replace=False)
        batch = [data[i] for i in batch_indices]
        state_batch = torch.stack([x[0].squeeze(0) for x in batch]).to(device)
        # Data Augmentation: random shuffling of color channels
        permutation = torch.randperm(state_batch.shape[1])
        state_batch = state_batch[:, permutation, :, :]
        prob_batch = torch.stack([torch.tensor(x[1]) for x in batch]).to(device)
        prob_batch = prob_batch[:, permutation]

        value_batch = torch.tensor([[x[2]] for x in batch], dtype=torch.float32).to(
            device
        )

        pred_prob_batch, pred_value_batch = network(state_batch)
        loss = loss_fn(
            pred_value_batch, value_batch, pred_prob_batch, prob_batch, value_scaling
        )
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        sum_loss += loss.item()

    return sum_loss / n_batches


if __name__ == "__main__":
    with_wandb = True
    all_data = []

    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NeuralNetwork(config)
    net.to(device)
    optimizer = optim.Adam(
        net.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["l2_weight_reg"],
    )

    if with_wandb:
        wandb.init(
            entity="hojmax",
            project="bachelor",
            config=config,
        )

    net.train()
    for i in range(config["train"]["n_iterations"]):
        episode_data = []
        env = FloodEnv(
            config["env"]["width"], config["env"]["height"], config["env"]["n_colors"]
        )
        while not env.is_terminal():
            _, probabilities = Node.alphago_zero_search(
                env,
                net,
                config["mcts"]["search_iterations"],
                config["mcts"]["c_puct"],
                config["mcts"]["temperature"],
            )
            episode_data.append((env.get_tensor_state(), probabilities))
            action = np.random.choice(env.n_colors, p=probabilities)
            env.step(action)

        value = env.value
        for i, (state, probabilities) in enumerate(episode_data):
            all_data.append((state, probabilities, value + i))

        if len(all_data) > config["train"]["max_data"]:
            all_data = all_data[-config["train"]["max_data"] :]

        avg_loss = train_network(
            net,
            all_data,
            config["train"]["batch_size"],
            config["train"]["batches_per_episode"],
            optimizer,
            config["train"]["value_scaling"],
        )
        if with_wandb:
            wandb.log({"loss": avg_loss, "value": value, "episode": i})

    torch.save(net.state_dict(), "model.pt")

    if with_wandb:
        wandb.save("model.pt")

        wandb.finish()
