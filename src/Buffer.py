import numpy as np
import torch
import torch.multiprocessing as mp


class ReplayBuffer:
    def __init__(self, config):
        self.max_size = config["train"]["buffer_size"]
        self.lock = mp.Lock()
        self.ptr = mp.Value("i", 0)
        self.size = mp.Value("i", 0)

        bay_size = (self.max_size, 1, config["env"]["R"], config["env"]["C"])
        flat_T_size = (
            self.max_size,
            config["env"]["N"] * (config["env"]["N"] - 1) // 2,
        )
        prob_size = (self.max_size, 2 * config["env"]["C"])
        value_size = (self.max_size, 1)

        self.bay = torch.zeros(bay_size, dtype=torch.float32).share_memory_()
        self.flat_T = torch.zeros(flat_T_size, dtype=torch.float32).share_memory_()
        self.prob = torch.zeros(prob_size, dtype=torch.float32).share_memory_()
        self.value = torch.zeros(value_size, dtype=torch.float32).share_memory_()

    def extend(
        self,
        bay: torch.Tensor,
        flat_T: torch.Tensor,
        prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        with self.lock:
            self.bay[self.ptr.value] = bay
            self.flat_T[self.ptr.value] = flat_T
            self.prob[self.ptr.value] = prob
            self.value[self.ptr.value] = value
            self.ptr.value = (self.ptr.value + 1) % self.max_size
            self.size.value = min(self.size.value + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple:
        with self.lock:
            indices = np.random.choice(self.size.value, batch_size, replace=False)
            return (
                self.bay[indices],
                self.flat_T[indices],
                self.prob[indices],
                self.value[indices],
            )

    def __len__(self):
        return self.size.value
