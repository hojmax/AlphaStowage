import torch
import torch.nn as nn
import torch.optim as optim


class Convulutional_Block(nn.Module):
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
    def __init__(self, config):
        super().__init__()
        self.env_config = config["env"]
        nn_config = config["nn"]
        input_channels = self.env_config["N"]

        layers = []
        layers.append(
            Convulutional_Block(
                input_channels,
                nn_config["hidden_channels"],
                nn_config["hidden_kernel_size"],
                stride=nn_config["hidden_stride"],
            )
        )
        for _ in range(nn_config["blocks"]):
            layers.append(
                Residual_Block(
                    nn_config["hidden_channels"],
                    nn_config["hidden_channels"],
                    nn_config["hidden_kernel_size"],
                    nn_config["hidden_stride"],
                )
            )

        self.layers = nn.Sequential(*layers)

        self.flat_T_reshaper = nn.Linear(
            self.env_config["N"] * (self.env_config["N"] - 1) // 2,
            self.env_config["R"] * self.env_config["C"],
        )

        self.sigmoid = nn.Sigmoid()

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=nn_config["hidden_channels"],
                out_channels=nn_config["policy_channels"],
                kernel_size=nn_config["policy_kernel_size"],
                stride=nn_config["policy_stride"],
            ),
            nn.BatchNorm2d(nn_config["policy_channels"]),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(
                nn_config["policy_channels"]
                * self.env_config["R"]
                * self.env_config["C"],
                2 * self.env_config["C"],
            ),
        )

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

    def forward(self, bay, flat_T, mask):
        flat_T = self.flat_T_reshaper(flat_T)
        flat_T = self.sigmoid(flat_T)
        flat_T = flat_T.view(-1, 1, self.env_config["R"], self.env_config["C"])
        x = torch.cat([bay, flat_T], dim=1)
        out = self.layers(x)
        policy = self.policy_head(out)
        policy = policy - (1 - mask) * 1e9
        policy = torch.softmax(policy, dim=-1)
        value = self.value_head(out)
        return policy, value
