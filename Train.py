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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    return args


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def download_model_and_get_merged_config(run_path, model_path, local_config):
    api = wandb.Api()
    run = api.run(run_path)
    file = run.file(model_path)
    file.download(replace=True)
    config = run.config
    merge_dicts(config, local_config)


def get_config():
    with open("config.json", "r") as f:
        return json.load(f)


def get_model_and_config(load_run, model_path, local_config):
    if load_run is not None:
        config = download_model_and_get_merged_config(
            load_run, model_path, local_config
        )
    else:
        config = local_config

    model = NeuralNetwork(config)

    if load_run is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    return model, config


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


def run_training_loop(
    model,
    config,
    device,
    testset,
    optimizer,
    scheduler,
):
    all_data = []

    model.train()
    best_score = float("-inf")
    best_model = None
    best_optimizer = None
    for i in tqdm(range(int(config["train"]["n_iterations"]))):
        should_test = (i + 1) % config["train"]["test_interval"] == 0
        if should_test:
            relative_score = test_network(model, testset, config, device)
            if relative_score > best_score:
                model_path = f"model{i}.pt"
                optimizer_path = f"optimizer{i}.pt"
                best_score = relative_score
                best_model = model_path
                best_optimizer = optimizer_path
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                wandb.save(model_path)
            else:
                # reload best model
                model.load_state_dict(torch.load(best_model))
                optimizer.load_state_dict(torch.load(best_optimizer))
                model.train()
            wandb.log({"relative_score": relative_score, "episode": i})
            model.train()

        env = FloodEnv(
            config["env"]["width"], config["env"]["height"], config["env"]["n_colors"]
        )
        episode_data, episode_value = play_episode(env, model, config, device)

        all_data.extend(episode_data)

        if len(all_data) > config["train"]["max_data"]:
            all_data = all_data[-config["train"]["max_data"] :]

        avg_loss, avg_value_loss, avg_cross_entropy = train_network(
            model,
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


if __name__ == "__main__":
    args = get_args()
    device = get_device()
    local_config = get_config()
    model, config = get_model_and_config(args.load_run, args.model_path, local_config)
    testset = create_testset(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    wandb.init(
        entity="hojmax",
        project="bachelor",
        config=config,
    )

    run_training_loop(
        model,
        config,
        device,
        testset,
        optimizer,
        scheduler,
    )

    torch.save(model.state_dict(), "model.pt")

    wandb.save("model.pt")

    wandb.finish()
