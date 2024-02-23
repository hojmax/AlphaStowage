import torch
import torch.nn as nn
import torch.optim as optim
from FloodEnv import FloodEnv, n_lookahead_run_episode
from NeuralNetwork import NeuralNetwork
import Node
import numpy as np
import wandb
import json
import argparse
from tqdm import tqdm
from Node import remove_pruning


def loss_fn(pred_value, value, pred_prob, prob, value_scaling):
    value_error = torch.mean(torch.square(value - pred_value))
    cross_entropy = (
        -torch.sum(prob.flatten() * torch.log(pred_prob.flatten())) / prob.shape[0]
    )
    loss = value_scaling * value_error + cross_entropy
    return loss, value_error, cross_entropy


def optimize_network(
    pred_value, value, pred_prob, prob, optimizer, scheduler, value_scaling
):
    loss, value_loss, cross_entropy = loss_fn(
        pred_value=pred_value,
        value=value,
        pred_prob=pred_prob,
        prob=prob,
        value_scaling=value_scaling,
    )
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss.item(), value_loss.item(), cross_entropy.item()


def train_batch(
    network, buffer, batch_size, optimizer, scheduler, value_scaling, device
):
    state, prob, value = buffer.sample(batch_size)
    state = state.to(device)
    prob = prob.to(device)
    value = value.to(device)
    pred_prob, pred_value = network(state)
    loss, value_loss, cross_entropy = optimize_network(
        pred_value=pred_value,
        value=value,
        pred_prob=pred_prob,
        prob=prob,
        optimizer=optimizer,
        scheduler=scheduler,
        value_scaling=value_scaling,
    )
    return (
        loss,
        value_loss,
        cross_entropy,
    )


def play_episode(env, net, config, device, deterministic=False):
    episode_data = []
    reused_tree = None

    while not env.is_terminal():
        reused_tree, probabilities = Node.alpha_zero_search(
            env,
            net,
            config["mcts"]["search_iterations"],
            config["mcts"]["c_puct"],
            config["mcts"]["temperature"],
            config["mcts"]["dirichlet_weight"],
            config["mcts"]["dirichlet_alpha"],
            device,
            reused_tree,
        )
        episode_data.append((env.get_tensor_state(), probabilities))
        if deterministic:
            action = np.argmax(probabilities)
        else:
            action = np.random.choice(env.n_colors, p=probabilities)

        env.step(action)
        reused_tree = reused_tree.children[action]
        remove_pruning(reused_tree)

    output_data = []
    real_value = env.value
    for i, (state, probabilities) in enumerate(episode_data):
        output_data.append((state, probabilities, real_value + i))

    return output_data, real_value


def create_testset(config):
    testset = []
    for i in range(config["eval"]["testset_size"]):
        np.random.seed(i)
        env = FloodEnv(
            config["env"]["width"], config["env"]["height"], config["env"]["n_colors"]
        )
        solution = n_lookahead_run_episode(env.copy(), config["eval"]["n_lookahead"])
        testset.append((env, solution))
    return testset


def test_network(net, testset, config, device):
    with torch.no_grad():
        net.eval()
        avg_error = 0

        for env, solution in testset:
            _, value = play_episode(env.copy(), net, config, device, deterministic=True)
            avg_error += value - solution

        avg_error /= len(testset)
        net.train()
        return avg_error


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def get_config():
    with open("config.json", "r") as f:
        return json.load(f)


def get_optimizer(model, config):
    return optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["l2_weight_reg"],
    )


def get_scheduler(optimizer, config):
    return optim.lr_scheduler.StepLR(
        optimizer,
        config["train"]["scheduler_step_size_in_batches"],
        config["train"]["scheduler_gamma"],
    )


# "testset_size": 100,
# Testset 1 created in 0.39 seconds
# Average score: -8.07

# Testset 2 created in 0.81 seconds
# Average score: -7.55

# Testset 3 created in 1.80 seconds
# Average score: -7.43

# Testset 4 created in 3.95 seconds
# Average score: -7.35

# Testset 5 created in 7.92 seconds
# Average score: -7.26

# Testset 6 created in 14.59 seconds
# Average score: -7.24

# Testset 7 created in 21.82 seconds
# Average score: -7.22

# Testset 8 created in 28.14 seconds
# Average score: -7.20

# Testset 9 created in 29.44 seconds
# Average score: -7.19
