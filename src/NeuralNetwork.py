import torch
import torch.nn as nn
import math


class Convolutional_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.silu = nn.SiLU()
        self.batch = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch(out)
        out = self.silu(out)
        return out


class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.silu = nn.SiLU()
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.batch2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out = out + x
        out = self.silu(out)
        return out


class NeuralNetwork(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.env_config = config["env"]
        nn_config = config["nn"]
        input_channels = 2

        tower1 = []
        tower1.append(
            Convolutional_Block(
                input_channels,
                nn_config["hidden_channels"],
                nn_config["hidden_kernel_size"],
                stride=nn_config["hidden_stride"],
            )
        )
        for _ in range(math.floor(nn_config["blocks"] / 2)):
            tower1.append(
                Residual_Block(
                    nn_config["hidden_channels"],
                    nn_config["hidden_channels"],
                    nn_config["hidden_kernel_size"],
                    nn_config["hidden_stride"],
                )
            )

        self.tower1 = nn.Sequential(*tower1)

        tower2 = []
        for _ in range(math.ceil(nn_config["blocks"] / 2)):
            tower2.append(
                Residual_Block(
                    nn_config["hidden_channels"],
                    nn_config["hidden_channels"],
                    nn_config["hidden_kernel_size"],
                    nn_config["hidden_stride"],
                )
            )

        self.tower2 = nn.Sequential(*tower2)

        self.flat_T_reshaper = nn.Sequential(
            nn.Linear(
                self.env_config["N"] * (self.env_config["N"] - 1) // 2,
                self.env_config["R"] * self.env_config["C"],
            ),
            nn.Sigmoid(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=nn_config["hidden_channels"],
                out_channels=nn_config["policy_channels"],
                kernel_size=nn_config["policy_kernel_size"],
                stride=nn_config["policy_stride"],
            ),
            nn.SiLU(),
            nn.BatchNorm2d(nn_config["policy_channels"]),
            nn.Flatten(),
            nn.Linear(
                nn_config["policy_channels"]
                * self.env_config["R"]
                * self.env_config["C"],
                2 * self.env_config["R"] * self.env_config["C"],
            ),
        )
        self.softmax = nn.Softmax(dim=1)

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=nn_config["hidden_channels"],
                out_channels=nn_config["value_channels"],
                kernel_size=nn_config["value_kernel_size"],
                stride=nn_config["value_stride"],
            ),
            nn.BatchNorm2d(nn_config["value_channels"]),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(
                nn_config["value_channels"]
                * self.env_config["R"]
                * self.env_config["C"],
                nn_config["value_hidden"],
            ),
            nn.SiLU(),
            nn.Linear(nn_config["value_hidden"], 1),
        )
        self.containers_left_embedding = nn.Sequential(
            nn.Linear(1, nn_config["embedding_hidden_size"]),
            nn.SiLU(),
            nn.Linear(nn_config["embedding_hidden_size"], nn_config["hidden_channels"]),
        )
        self.mask_embedding = nn.Sequential(
            nn.Linear(
                2 * self.env_config["R"] * self.env_config["C"],
                nn_config["embedding_hidden_size"],
            ),
            nn.SiLU(),
            nn.Linear(nn_config["embedding_hidden_size"], nn_config["hidden_channels"]),
        )

    def forward(self, bay, flat_T, containers_left, mask):
        flat_T = self.flat_T_reshaper(flat_T)
        flat_T = flat_T.view(-1, 1, self.env_config["R"], self.env_config["C"])
        if bay.dim() == 2:
            bay = bay.unsqueeze(0).unsqueeze(0)

        x = torch.cat([bay, flat_T], dim=1)
        tower1 = self.tower1(x)
        containers_left_embedding = (
            self.containers_left_embedding(containers_left).unsqueeze(-1).unsqueeze(-1)
        )
        mask_embedding = self.mask_embedding(mask).unsqueeze(-1).unsqueeze(-1)
        tower1 = tower1 + containers_left_embedding + mask_embedding
        tower2 = self.tower2(tower1)

        logits = self.policy_head(tower2)
        logits = logits - (1 - mask) * 1e9  # mask out invalid moves
        policy = self.softmax(logits)

        value = self.value_head(tower2)

        return policy, value, logits
