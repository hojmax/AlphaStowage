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
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.batch2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out = out + x
        out = self.relu(out)
        return out


class NeuralNetwork(nn.Module):
    def __init__(self, n_colors, width, height, config):
        super().__init__()
        input_channels = n_colors

        layers = []
        layers.append(
            Convulutional_Block(
                input_channels,
                config["hidden_channels"],
                config["hidden_kernel_size"],
                stride=config["hidden_stride"],
            )
        )
        for _ in range(config["blocks"]):
            layers.append(
                Residual_Block(
                    config["hidden_channels"],
                    config["hidden_channels"],
                    config["hidden_kernel_size"],
                    config["hidden_stride"],
                )
            )

        self.layers = nn.Sequential(*layers)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config["hidden_channels"],
                out_channels=config["policy_channels"],
                kernel_size=config["policy_kernel_size"],
                stride=config["policy_stride"],
            ),
            nn.BatchNorm2d(config["policy_channels"]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config["policy_channels"] * width * height, n_colors),
            nn.Softmax(dim=1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config["hidden_channels"],
                out_channels=config["value_channels"],
                kernel_size=config["value_kernel_size"],
                stride=config["value_stride"],
            ),
            nn.BatchNorm2d(config["value_channels"]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                config["value_channels"] * width * height, config["value_hidden"]
            ),
            nn.ReLU(),
            nn.Linear(config["value_hidden"], 1),
        )

    def forward(self, x):
        out = self.layers(x)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


# Testing the network
if __name__ == "__main__":
    from Train import train_network, loss_fn

    n_colors = 3
    width = 5
    height = 5
    n_blocks = 3
    learning_rate = 0.01
    l2_weight_reg = 0.0001
    net = NeuralNetwork(
        n_colors=n_colors, width=width, height=height, n_blocks=n_blocks
    )
    net.train()
    optimizer = optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=l2_weight_reg,
    )
    n_fake_samples = 10
    soft_max = nn.Softmax(dim=1)
    # set torch seed
    torch.manual_seed(0)
    fake_training_data = [
        (
            torch.randint(0, n_colors, (1, 1, width, height), dtype=torch.float32),
            soft_max(torch.rand(1, n_colors, dtype=torch.float32)).numpy(),
            torch.randint(0, n_colors, (1,)),
        )
        for _ in range(n_fake_samples)
    ]
    print("REAL input", fake_training_data[0][0])
    print("REAL input2", fake_training_data[1][0])
    print("REAL", fake_training_data[0][1], fake_training_data[0][2])
    print("REAL2", fake_training_data[1][1], fake_training_data[1][2])
    for i in range(1000):
        if i % 100 == 0:
            with torch.no_grad():
                net.eval()
                print(net(fake_training_data[0][0]))
                print(net(fake_training_data[1][0]))
                net.train()
        loss = train_network(net, fake_training_data, n_fake_samples, 1, optimizer)

        if i % 100 == 0:
            print(f"Loss: {loss}")
