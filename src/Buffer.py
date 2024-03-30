import threading
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, config):
        self.max_size = config["train"]["buffer_size"]
        self.lock = threading.Lock()
        self.episode = 0
        self.ptr = 0
        self.size = 0

        bay_size = (self.max_size, 1, config["env"]["R"], config["env"]["C"])
        flat_T_size = (
            self.max_size,
            config["env"]["N"] * (config["env"]["N"] - 1) // 2,
        )
        prob_size = (self.max_size, 2 * config["env"]["C"])
        value_size = (self.max_size, 1)
        self.bay = torch.zeros(bay_size, dtype=torch.float32)
        self.flat_T = torch.zeros(flat_T_size, dtype=torch.float32)
        self.prob = torch.zeros(prob_size, dtype=torch.float32)
        self.value = torch.zeros(value_size, dtype=torch.float32)

    def increment_episode(self) -> int:
        with self.lock:
            self.episode += 1
            return self.episode

    def extend(
        self,
        bay: torch.Tensor,
        flat_T: torch.Tensor,
        prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        with self.lock:
            self.bay[self.ptr] = bay
            self.flat_T[self.ptr] = flat_T
            self.prob[self.ptr] = prob
            self.value[self.ptr] = value
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple:
        with self.lock:
            indices = np.random.choice(self.size, batch_size, replace=False)
            return (
                self.bay[indices],
                self.flat_T[indices],
                self.prob[indices],
                self.value[indices],
            )

    def __len__(self):
        return self.size
