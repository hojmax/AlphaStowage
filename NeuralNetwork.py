import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence


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


class SequenceEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config["nn"]["embedding_dim"]
        self.container_embedding = torch.nn.Linear(1, self.embedding_dim)
        self.T_embedding = torch.nn.Linear(1, self.embedding_dim)
        self.token_embedding = torch.nn.Embedding(4, self.embedding_dim)

    def forward(self, bay, T):
        COL_STOP = 0
        BAY_STOP = 1
        NEXT_PORT = 2
        T_STOP = 3

        bay = bay.squeeze(1)
        T = T.squeeze(1)

        batch_size, _, R = bay.shape
        bay_flat = bay.view(batch_size, -1, 1)
        T_flat = T.view(batch_size, -1, 1)

        bay_flat_embedding = self.container_embedding(bay_flat)
        T_flat_embedding = self.T_embedding(T_flat)

        # We need to insert COL_STOP tokens and handle batches properly
        bay_with_col = []
        for b in range(batch_size):
            bay_seq = []
            for i in range(bay_flat_embedding.shape[1]):
                bay_seq.append(bay_flat_embedding[b, i])
                if ((i + 1) % R) == 0 and i + 1 != bay_flat_embedding.shape[1]:
                    token = self.token_embedding(
                        torch.tensor([COL_STOP], dtype=torch.int).to(bay.device)
                    )
                    bay_seq.append(token.squeeze(0))
            bay_with_col.append(torch.stack(bay_seq))

        # Insert BAY_STOP for each sequence in the batch
        for b in range(batch_size):
            bay_with_col[b] = torch.cat(
                (
                    bay_with_col[b],
                    self.token_embedding(
                        torch.tensor([BAY_STOP], dtype=torch.int).to(bay.device)
                    )
                    .squeeze(0)
                    .unsqueeze(0),
                ),
                dim=0,
            )

        # Handle T_flat_embedding similarly
        T_with_tokens = []
        for b in range(batch_size):
            T_seq = []
            N = int((1 + (1 + 8 * T_flat_embedding.shape[1]) ** 0.5) / 2)
            counter = 0
            for i in range(T_flat_embedding.shape[1]):
                T_seq.append(T_flat_embedding[b, i])
                counter += 1
                if counter == (N - 1) and (i + 1) != T_flat_embedding.shape[1]:
                    token = self.token_embedding(
                        torch.tensor([NEXT_PORT], dtype=torch.int).to(T.device)
                    )
                    T_seq.append(token.squeeze(0))
                    counter = 0
                    N -= 1
            T_with_tokens.append(torch.stack(T_seq))

        # Insert T_STOP for each sequence in the batch
        for b in range(batch_size):
            T_with_tokens[b] = torch.cat(
                (
                    T_with_tokens[b],
                    self.token_embedding(
                        torch.tensor([T_STOP], dtype=torch.int).to(T.device)
                    )
                    .squeeze(0)
                    .unsqueeze(0),
                ),
                dim=0,
            )

        # Concatenate all
        final_sequence = [
            torch.cat((bay_seq, T_seq), dim=0)
            for bay_seq, T_seq in zip(bay_with_col, T_with_tokens)
        ]

        # Ensure final_sequence is a proper batched tensor
        final_sequence = torch.nn.utils.rnn.pad_sequence(
            final_sequence, batch_first=True
        )

        return final_sequence


class NeuralNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.env_config = config["env"]
        self.sequence_embedder = SequenceEmbedder(config)
        nn_config = config["nn"]
        self.embedding_dim = nn_config["embedding_dim"]

        encoder_layer_config = config.get("transformer_encoder", {})
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=encoder_layer_config.get("nhead", 8),
            dim_feedforward=encoder_layer_config.get("dim_feedforward", 2048),
            dropout=encoder_layer_config.get("dropout", 0.1),
        )

        # Stack encoder layers to form the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layer_config.get("num_layers", 8),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            ),
            nn.BatchNorm2d(1),
            nn.SiLU(),
            nn.Linear(
                self.embedding_dim,
                2 * self.env_config["C"],
            ),
        )

        self.value_head = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            ),
            nn.BatchNorm2d(1),
            nn.SiLU(),
            nn.Linear(
                self.embedding_dim,
                1,
            ),
        )

    def forward(self, bay, T, mask):
        sequence = self.sequence_embedder(bay, T)

        transformer_output = self.transformer_encoder(sequence)

        sequence_embedding = transformer_output.mean(dim=1).unsqueeze(0).unsqueeze(0)

        policy = self.policy_head(sequence_embedding)

        value = self.value_head(sequence_embedding)

        # print(policy.shape, value.shape)

        return policy, value
