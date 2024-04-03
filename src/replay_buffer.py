import numpy as np
import ray
from shared_storage import SharedStorage


@ray.remote
class ReplayBuffer:
    def __init__(self, config):
        self.max_size = config["train"]["buffer_size"]
        self.index = 0
        self.buffer = {}

        # Fix random generator seed
        np.random.seed(config["train"]["seed"])

    def extend(
        self,
        game_history: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        index = self.index % self.max_size
        self.buffer[index] = game_history
        self.index += 1

    def sample(self, batch_size: int) -> tuple:
        indices = np.random.choice(self.index, batch_size)
        batch = [self.buffer[i] for i in indices]

        return batch

    def get_buffer(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)
