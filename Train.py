import torch
import torch.nn as nn
import torch.optim as optim
from FloodEnv import FloodEnv, n_lookahead_run_episode
from NeuralNetwork import NeuralNetwork
import Node
import numpy as np
import wandb
import json
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


def play_episode(env, net, config, device):
    episode_data = []

    while not env.is_terminal():
        _, probabilities = Node.alpha_zero_search(
            env,
            net,
            config["mcts"]["search_iterations"],
            config["mcts"]["c_puct"],
            config["mcts"]["temperature"],
            device,
        )
        episode_data.append((env.get_tensor_state(), probabilities))
        action = np.random.choice(env.n_colors, p=probabilities)
        env.step(action)

    output_data = []
    real_value = env.value
    for i, (state, probabilities) in enumerate(episode_data):
        output_data.append((state, probabilities, real_value + i))

    return output_data, real_value


def extend_and_handle_duplicates(all_data, episode_data):
    for state, probabilities, value in episode_data:
        removalIndex = None
        shouldAdd = True
        for i, (state2, _, value2) in enumerate(all_data):
            if torch.equal(state, state2):
                if value < value2:
                    removalIndex = i
                else:
                    shouldAdd = False
                break

        if removalIndex is not None:
            all_data.pop(removalIndex)
        if shouldAdd:
            all_data.append((state, probabilities, value))


def create_testset(config):
    testset = []
    for i in range(10):
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
            _, value = play_episode(env, net, config, device)
            avg_error += value - solution

        avg_error /= len(testset)
        return avg_error


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    testset = create_testset(config)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    net = NeuralNetwork(config)
    net.to(device)
    optimizer = optim.Adam(
        net.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["l2_weight_reg"],
    )
    # exponential decay
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, config["train"]["learning_rate_decay"]
    )
    all_data = []

    wandb.init(
        entity="hojmax",
        project="bachelor",
        config=config,
    )

    net.train()

    for i in tqdm(range(int(config["train"]["n_iterations"]))):
        if (i + 1) % config["train"]["test_interval"] == 0:
            avg_error = test_network(net, testset, config, device)
            wandb.log({"test_error": avg_error, "episode": i})
            net.train()

        env = FloodEnv(
            config["env"]["width"], config["env"]["height"], config["env"]["n_colors"]
        )
        episode_data, episode_value = play_episode(env, net, config, device)
        extend_and_handle_duplicates(all_data, episode_data)

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
