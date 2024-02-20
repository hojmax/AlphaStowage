import wandb
import json
from tqdm import tqdm
import torch
from create_supervised_data import MyDataset
from torch.utils.data import DataLoader
from NeuralNetwork import NeuralNetwork
import torch.optim as optim


def optimize_network(
    pred_value, value, pred_prob, prob, optimizer, value_scaling
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

    return loss.item(), value_loss.item(), cross_entropy.item()


def loss_fn(pred_value, value, pred_prob, prob, value_scaling):
    value_error = torch.mean(torch.square(value - pred_value))
    cross_entropy = (
        -torch.sum(prob.flatten() * torch.log(pred_prob.flatten())) / prob.shape[0]
    )
    loss = value_scaling * value_error + cross_entropy
    return loss, value_error, cross_entropy


if __name__ == "__main__":
    # load dataset.pt
    dataset = torch.load("dataset.pt")
    total_samples = len(dataset)
    train_size = int(total_samples * 0.8)
    test_size = total_samples - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # load config.json
    with open("config.json", "r") as f:
        config = json.load(f)
    # load model
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


    all_data = []

    wandb.init(
        entity="hojmax",
        project="bachelor",
        config=config,
    )

    net.train()
    n_epochs = 1000
    for i in tqdm(range(n_epochs)):
        epoch_loss = 0
        epoch_value_loss = 0
        epoch_cross_entropy = 0
        for data in train_loader:
            state, action, value = data
            state = state.to(device)
            action = action.to(device)
            value = value.to(device)
            pred_prob, pred_value = net(state)
            loss, value_loss, cross_entropy = optimize_network(
                pred_value,
                value,
                pred_prob,
                action,
                optimizer,
                config["train"]["value_scaling"],
            )
            epoch_loss += loss
            epoch_value_loss += value_loss
            epoch_cross_entropy += cross_entropy
        epoch_loss /= len(train_loader)
        epoch_value_loss /= len(train_loader)
        epoch_cross_entropy /= len(train_loader)
        wandb.log(
            {
                "loss": epoch_loss,
                "value_loss": epoch_value_loss,
                "cross_entropy": epoch_cross_entropy,
            }
        )
        net.eval()
        test_loss = 0
        test_value_loss = 0
        test_cross_entropy = 0
        with torch.no_grad():
            for data in test_loader:
                state, action, value = data
                state = state.to(device)
                action = action.to(device)
                value = value.to(device)
                pred_prob, pred_value = net(state)
                loss, value_loss, cross_entropy = loss_fn(
                    pred_value=pred_value,
                    value=value,
                    pred_prob=pred_prob,
                    prob=action,
                    value_scaling=config["train"]["value_scaling"],
                )
                test_loss += loss
                test_value_loss += value_loss
                test_cross_entropy += cross_entropy
            test_loss /= len(test_loader)
            test_value_loss /= len(test_loader)
            test_cross_entropy /= len(test_loader)
            wandb.log(
                {
                    "test_loss": test_loss,
                    "test_value_loss": test_value_loss,
                    "test_cross_entropy": test_cross_entropy,
                }
            )
