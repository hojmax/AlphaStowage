import torch
import torch.optim as optim
import Node
import numpy as np
import json
from MPSPEnv import Env
from Node import (
    remove_all_pruning,
    get_torch_obs,
    close_envs_in_tree,
    TruncatedEpisodeError,
)
import random
import warnings
import wandb
from NeuralNetwork import NeuralNetwork


def loss_fn(pred_value, value, pred_prob, prob, config):
    value_error = torch.mean(torch.square(value - pred_value))
    cross_entropy = (
        -torch.sum(prob.flatten() * torch.log(pred_prob.flatten())) / prob.shape[0]
    )
    loss = config["train"]["value_scaling"] * value_error + cross_entropy
    return loss, value_error, cross_entropy


def optimize_model(pred_value, value, pred_prob, prob, optimizer, scheduler, config):
    loss, value_loss, cross_entropy = loss_fn(
        pred_value=pred_value,
        value=value,
        pred_prob=pred_prob,
        prob=prob,
        config=config,
    )
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss.item(), value_loss.item(), cross_entropy.item()


def train_batch(model, buffer, optimizer, scheduler, config):
    bay, flat_T, prob, value = buffer.sample(config["train"]["batch_size"])
    bay = bay.to(model.device)
    flat_T = flat_T.to(model.device)
    prob = prob.to(model.device)
    value = value.to(model.device)

    pred_prob, pred_value = model(bay, flat_T)
    loss, value_loss, cross_entropy = optimize_model(
        pred_value=pred_value,
        value=value,
        pred_prob=pred_prob,
        prob=prob,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    return (
        loss,
        value_loss,
        cross_entropy,
    )


def get_action(
    probabilities: torch.Tensor, deterministic: bool, config: dict, env: Env
) -> int:
    if deterministic:
        action = torch.argmax(probabilities).item()
    else:
        action = np.random.choice(2 * config["env"]["C"], p=probabilities)

    action = action if action < env.C else action + env.C - config["env"]["C"]

    return action


def close_other_branches(reused_tree: Node, action: int, env: Env) -> None:
    for a in range(2 * env.C):
        if a != action and a in reused_tree.children:
            close_envs_in_tree(reused_tree.children[a])


def update_tree(reused_tree: Node, action: int, env: Env) -> None:
    close_other_branches(reused_tree, action, env)

    reused_tree.env.close()
    reused_tree = reused_tree.children[action]
    reused_tree.parent = None
    reused_tree.prior_prob = None
    remove_all_pruning(reused_tree)
    return reused_tree


def add_value_to_observations(observations, final_value):
    output = []

    for i, sample in enumerate(observations):
        output.append(sample + [torch.tensor(final_value + i)])

    return output


def play_episode(env, net, config, device, deterministic=False):
    if deterministic:
        np.random.seed(0)

    observations = []
    reused_tree = None
    transposition_table = {}

    while not env.terminal:
        reused_tree, probabilities, transposition_table = Node.alpha_zero_search(
            env,
            net,
            device,
            config,
            reused_tree,
            transposition_table,
        )
        bay, flat_T = get_torch_obs(env, config)
        observations.append([bay, flat_T, probabilities])

        action = get_action(probabilities, deterministic, config, env)
        env.step(action)

        reused_tree = update_tree(reused_tree, action, env)

    close_envs_in_tree(reused_tree)

    final_value = -env.moves_to_solve
    reshuffles = env.total_reward
    observations = add_value_to_observations(observations, final_value)

    return observations, final_value, reshuffles


def get_env(config):
    return Env(
        config["env"]["R"],
        config["env"]["C"],
        config["env"]["N"],
        # random.choice(range(6, config["env"]["R"] + 1, 2)),
        # random.choice(range(2, config["env"]["C"] + 1, 2)),
        # random.choice(range(4, config["env"]["N"] + 1, 2)),
        skip_last_port=True,
        take_first_action=True,
        strict_mask=True,
    )


def create_testset(config):
    testset = []
    for i in range(config["eval"]["testset_size"]):
        np.random.seed(i)
        env = get_env(config)
        env.reset(i)
        testset.append(env)
    return testset


def test_network(model, testset, config):
    was_training = model.training

    with torch.no_grad():
        model.eval()
        avg_error = 0
        avg_reshuffles = 0

        for env in testset:
            copy_env = env.copy()
            try:
                _, value, reshuffles = play_episode(
                    copy_env, model, config, model.device, deterministic=True
                )
                avg_error += value
                avg_reshuffles += reshuffles
            except TruncatedEpisodeError:
                warnings.warn("Episode was truncated during evaluation.")
                avg_error += -1e9
                avg_reshuffles += -1e9

            copy_env.close()

        avg_error /= len(testset)
        avg_reshuffles /= len(testset)

    model.train(mode=was_training)

    return avg_error, avg_reshuffles


def save_model(model: NeuralNetwork, config: dict, i: int) -> None:
    model_path = f"model{i}.pt"
    torch.save(model.state_dict(), model_path)

    if config["train"]["log_wandb"]:
        wandb.save(model_path)


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def get_config(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)

    return config


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
