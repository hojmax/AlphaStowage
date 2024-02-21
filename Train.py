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


def loss_fn(pred_value, value, pred_prob, prob, value_scaling):
    value_error = torch.mean(torch.square(value - pred_value))
    cross_entropy = (
        -torch.sum(prob.flatten() * torch.log(pred_prob.flatten())) / prob.shape[0]
    )
    loss = value_scaling * value_error + cross_entropy
    return loss, value_error, cross_entropy


def get_batch(data, batch_size):
    batch_indices = np.random.choice(len(data), batch_size, replace=False)
    batch = [data[i] for i in batch_indices]
    state_batch = torch.stack([x[0].squeeze(0) for x in batch])
    # Data Augmentation: random shuffling of color channels
    permutation = torch.randperm(state_batch.shape[1])
    state_batch = state_batch[:, permutation, :, :]
    prob_batch = torch.stack([torch.tensor(x[1]) for x in batch]).float()
    prob_batch = prob_batch[:, permutation]
    value_batch = torch.tensor([[x[2]] for x in batch], dtype=torch.float32)

    return state_batch, prob_batch, value_batch


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


def train_network(
    network, data, batch_size, n_batches, optimizer, scheduler, value_scaling, device
):
    if len(data) < batch_size:
        batch_size = len(data)

    sum_loss = 0
    sum_value_loss = 0
    sum_cross_entropy = 0

    for _ in range(n_batches):
        state, prob, value = get_batch(data, batch_size)
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

        sum_loss += loss
        sum_value_loss += value_loss
        sum_cross_entropy += cross_entropy

    return (
        sum_loss / n_batches,
        sum_value_loss / n_batches,
        sum_cross_entropy / n_batches,
    )


def play_episode(env, net, config, device, deterministic=False):
    episode_data = []

    while not env.is_terminal():
        _, probabilities = Node.alpha_zero_search(
            env,
            net,
            config["mcts"]["search_iterations"],
            config["mcts"]["c_puct"],
            config["mcts"]["temperature"],
            config["mcts"]["dirichlet_weight"],
            config["mcts"]["dirichlet_alpha"],
            device,
            deterministic=deterministic,
        )
        episode_data.append((env.get_tensor_state(), probabilities))
        if deterministic:
            action = np.argmax(probabilities)
        else:
            action = np.random.choice(env.n_colors, p=probabilities)
        env.step(action)

    output_data = []
    real_value = env.value
    for i, (state, probabilities) in enumerate(episode_data):
        output_data.append((state, probabilities, real_value + i))

    return output_data, real_value


def create_testset(config):
    testset = []
    for i in range(100):
        np.random.seed(i)
        env = FloodEnv(
            config["env"]["width"], config["env"]["height"], config["env"]["n_colors"]
        )
        solution = n_lookahead_run_episode(env.copy(), config["eval"]["n_lookahead"])
        testset.append((env.copy(), solution))
    return testset


def test_network(net, testset, config, device):
    with torch.no_grad():
        net.eval()
        avg_error = 0

        for env, solution in testset:
            _, value = play_episode(env.copy(), net, config, device, deterministic=True)
            avg_error += value - solution

        avg_error /= len(testset)
        return avg_error


def merge_dicts(a, b):
    """
    Recursively merges dictionary b into dictionary a. Prefers the values of a.

    :param a: Target dictionary where the merge results will be stored.
    :param b: Source dictionary from which keys and values are taken if they don't exist in a.
    """
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    with open("config.json", "r") as f:
        local_config = json.load(f)

    if args.load_run is not None:
        run_path = args.load_run
        api = wandb.Api()
        run = api.run(run_path)
        file = run.file(args.model_path)
        file.download(replace=True)
        config = run.config
        merge_dicts(config, local_config)
    else:
        config = local_config

    net = NeuralNetwork(config)

    if args.load_run is not None:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)

    testset = create_testset(config)
    optimizer = optim.Adam(
        net.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["l2_weight_reg"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        config["train"]["scheduler_step_size"] * config["train"]["batches_per_episode"],
        config["train"]["scheduler_gamma"],
    )
    all_data = []

    wandb.init(
        entity="hojmax",
        project="bachelor",
        config=config,
    )

    net.train()
    best_score = float("-inf")
    best_model = None

    for i in tqdm(range(int(config["train"]["n_iterations"]))):
        if (i + 1) % config["train"]["test_interval"] == 0:
            relative_score = test_network(net, testset, config, device)
            if relative_score > best_score:
                model_path = f"model{i}.pt"
                best_score = relative_score
                best_model = model_path
                torch.save(net.state_dict(), model_path)
                wandb.save(model_path)
            else:
                # reload best model
                net.load_state_dict(torch.load(best_model))
                net.train()
            wandb.log({"relative_score": relative_score, "episode": i})
            net.train()

        env = FloodEnv(
            config["env"]["width"], config["env"]["height"], config["env"]["n_colors"]
        )
        episode_data, episode_value = play_episode(env, net, config, device)

        all_data.extend(episode_data)

        if len(all_data) > config["train"]["max_data"]:
            all_data = all_data[-config["train"]["max_data"] :]

        avg_loss, avg_value_loss, avg_cross_entropy = train_network(
            net,
            all_data,
            config["train"]["batch_size"],
            config["train"]["batches_per_episode"],
            optimizer,
            scheduler,
            config["train"]["value_scaling"],
            device,
        )

        wandb.log(
            {
                "loss": avg_loss,
                "value_loss": avg_value_loss,
                "cross_entropy": avg_cross_entropy,
                "value": episode_value,
                "episode": i,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    torch.save(net.state_dict(), "model.pt")

    wandb.save("model.pt")

    wandb.finish()


# Things to add:
# - L2 weight regularization
# - Other implementations
#   - https://github.com/geochri/AlphaZero_Chess/blob/master/src/MCTS_chess.py
