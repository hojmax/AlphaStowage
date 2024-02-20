import numpy as np
from itertools import product
import torch
from torch.utils.data import Dataset
import json
from FloodEnv import bfs_solver
from FloodEnv import FloodEnv
from torch.nn.functional import one_hot


def get_all_possible_non_terminal_boards(width, height, n_colors):
    combinations = list(product(list(range(n_colors)), repeat=width * height))
    arrays = [
        np.array(combination).reshape((width, height))
        for combination in combinations
        if len(set(combination)) > 1  # Remove terminal states
    ]
    return arrays


class MyDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    config["env"]["width"] = 3
    data = get_all_possible_non_terminal_boards(
        config["env"]["width"], config["env"]["height"], config["env"]["n_colors"]
    )
    tuples = []

    for board in data:
        env = FloodEnv(
            config["env"]["width"], config["env"]["height"], config["env"]["n_colors"]
        )
        env.reset(board)
        best_value, best_action = bfs_solver(env)
        # one hot encode best action
        best_action = one_hot(torch.tensor(best_action), config["env"]["n_colors"])
        tuples.append(
            (
                env.get_tensor_state().squeeze(0),
                best_action,
                torch.tensor(best_value).unsqueeze(0),
            )
        )

    dataset = MyDataset(tuples)

    torch.save(dataset, "dataset.pt")
