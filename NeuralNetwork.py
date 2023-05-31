import torch
import torch.nn as nn


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
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch(out)
        out = self.relu(out)
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
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch(out)
        out = out + x
        out = self.relu(out)
        return out


class NeuralNetwork(nn.Module):
    def __init__(self, n_colors, width, height, n_blocks):
        super().__init__()
        input_channels = 1
        hidden_channels = 8
        hidden_kernel_size = 3
        hidden_stride = 1
        policy_channels = 2
        policy_kernel_size = 1
        policy_stride = 1
        value_channels = 1
        value_kernel_size = 1
        value_stride = 1
        value_hidden = 256

        layers = []
        layers.append(
            Convulutional_Block(
                input_channels,
                hidden_channels,
                hidden_kernel_size,
                stride=hidden_stride,
            )
        )
        for _ in range(n_blocks):
            layers.append(
                Residual_Block(
                    hidden_channels, hidden_channels, hidden_kernel_size, hidden_stride
                )
            )

        self.layers = nn.Sequential(*layers)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=policy_channels,
                kernel_size=policy_kernel_size,
                stride=policy_stride,
            ),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_channels * width * height, n_colors),
            nn.Softmax(dim=1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=value_channels,
                kernel_size=value_kernel_size,
                stride=value_stride,
            ),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_channels * width * height, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x):
        out = self.layers(x)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


# Testing the network
if __name__ == "__main__":
    n_colors = 3
    width = 5
    height = 5
    n_blocks = 3
    net = NeuralNetwork(
        n_colors=n_colors, width=width, height=height, n_blocks=n_blocks
    )
    # batch size, channels, width, height
    x = torch.randint(0, n_colors, (1, 1, width, height), dtype=torch.float32)
    print(x)
    policy, value = net(x)
    print(policy)
    print(value)
