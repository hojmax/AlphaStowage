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


class BaselinePolicy:
    def __init__(self, C, N):
        self.C = C
        self.N = N

    def predict(self, one_hot_bay, deterministic=True):
        """Place the container in the rightmost non-filled column."""
        j = self.C - 1

        while j >= 1:
            can_drop = True
            for h in range(self.N - 1):
                if one_hot_bay[h, 0, j] != 0:
                    can_drop = False
                    break
            if can_drop:
                break
            j -= 1

        return j, {}


def train_batch(
    network, buffer, batch_size, optimizer, scheduler, value_scaling, device
):
    bay, flat_T, prob, value = buffer.sample(batch_size)
    bay = bay.to(device)
    flat_T = flat_T.to(device)
    prob = prob.to(device)
    value = value.to(device)
    pred_prob, pred_value = network(bay, flat_T)
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


def baseline_policy(env, config):
    policy = BaselinePolicy(env.C, env.N)
    action, _ = policy.predict(env.one_hot_bay)
    probabilities = torch.zeros(2 * config["env"]["C"]) + 0.1 / (
        2 * config["env"]["C"] - 1
    )
    probabilities[action] = 0.9
    return action, probabilities


def play_episode(env, net, config, device, deterministic=False):
    if deterministic:
        np.random.seed(13)

    episode_data = []
    reused_tree = None
    transposition_table = {}

    while not env.terminal:
        if config["train"]["use_baseline_policy"]:
            action, probabilities = baseline_policy(env, config)
        else:
            reused_tree, probabilities, transposition_table = Node.alpha_zero_search(
                env,
                net,
                config["mcts"]["search_iterations"],
                config["mcts"]["c_puct"],
                config["mcts"]["temperature"],
                config["mcts"]["dirichlet_weight"],
                config["mcts"]["dirichlet_alpha"],
                device,
                config,
                reused_tree,
                transposition_table,
            )
            if deterministic:
                action = torch.argmax(probabilities).item()
            else:
                action = np.random.choice(2 * config["env"]["C"], p=probabilities)

        bay, flat_T = get_torch_obs(env, config)
        episode_data.append([bay, flat_T, probabilities])
        env.step(action)

        if reused_tree != None:
            # close the other branches
            for a in range(2 * env.C):
                if a != action and a in reused_tree.children:
                    close_envs_in_tree(reused_tree.children[a])
            reused_tree.env.close()

            reused_tree = reused_tree.children[action]
            reused_tree.parent = None
            reused_tree.prior_prob = None
            remove_all_pruning(reused_tree)

    if reused_tree:
        close_envs_in_tree(reused_tree)

    output_data = []
    for i, sample in enumerate(episode_data):
        output_data.append(sample + [torch.tensor(-env.moves_to_solve + i)])

    return output_data, -env.moves_to_solve, env.total_reward


def get_env(config):
    return Env(
        random.choice(range(6, config["env"]["R"] + 1, 2)),
        random.choice(range(2, config["env"]["C"] + 1, 2)),
        random.choice(range(4, config["env"]["N"] + 1, 2)),
        skip_last_port=True,
        take_first_action=True,
        strict_mask=True,
    )


def create_testset(config):
    testset = []
    for i in range(config["eval"]["testset_size"]):
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
                avg_error += -env.N * env.R * env.C
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


def get_config():
    with open("config.json", "r") as f:
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
