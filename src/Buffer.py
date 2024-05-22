import numpy as np
import torch
import torch.multiprocessing as mp
import warnings


class ReplayBuffer:
    def __init__(self, config):
        self.max_size = config["replay_buffer"]["max_size"]
        self.lock = mp.Lock()
        self.ptr = mp.Value("i", 0)
        self.size = mp.Value("i", 0)

        (
            self.bay,
            self.flat_T,
            self.prob,
            self.value,
            self.containers_left,
            self.mask,
        ) = self._create_buffers(self.max_size, config)

        if config["replay_buffer"]["checkpoint_path"]:
            self.load_from_disk(config)

        self.config = config

    def _create_buffers(self, max_size, config):
        bay_size = (max_size, 1, config["env"]["R"], config["env"]["C"])
        flat_T_size = (
            max_size,
            config["env"]["N"] * (config["env"]["N"] - 1) // 2,
        )
        prob_size = (max_size, 2 * config["env"]["R"] * config["env"]["C"])
        value_size = (max_size, 1)
        container_left_size = (max_size, 1)
        mask_size = (max_size, 2 * config["env"]["R"] * config["env"]["C"])

        bay = torch.zeros(bay_size, dtype=torch.float32).share_memory_()
        flat_T = torch.zeros(flat_T_size, dtype=torch.float32).share_memory_()
        prob = torch.zeros(prob_size, dtype=torch.float32).share_memory_()
        value = torch.zeros(value_size, dtype=torch.float32).share_memory_()
        containers_left = torch.zeros(
            container_left_size, dtype=torch.float32
        ).share_memory_()
        mask = torch.zeros(mask_size, dtype=torch.float32).share_memory_()

        return bay, flat_T, prob, value, containers_left, mask

    def load_from_disk(self, config):
        try:
            data = torch.load(config["replay_buffer"]["checkpoint_path"])
            self.bay = data["bay"]
            self.flat_T = data["flat_T"]
            self.prob = data["prob"]
            self.value = data["value"]
            self.containers_left = data["containers_left"]
            self.mask = data["mask"]
            self.ptr.value = data["ptr"]
            self.size.value = data["size"]
        except FileNotFoundError:
            warnings.warn(
                f"Could not find file at {config['replay_buffer']['checkpoint_path']}"
            )

    def save_to_disk(self):
        data = {
            "bay": self.bay,
            "flat_T": self.flat_T,
            "prob": self.prob,
            "value": self.value,
            "containers_left": self.containers_left,
            "mask": self.mask,
            "ptr": self.ptr.value,
            "size": self.size.value,
        }
        torch.save(data, f"replay_buffer_checkpoint.pt")

    def extend(
        self,
        observations: list,
    ) -> None:
        with self.lock:
            for obs in observations:
                (bay, flat_T, prob, containers_left, mask, value) = obs
                self.bay[self.ptr.value] = bay
                self.flat_T[self.ptr.value] = flat_T
                self.prob[self.ptr.value] = prob
                self.value[self.ptr.value] = value
                self.containers_left[self.ptr.value] = containers_left
                self.mask[self.ptr.value] = mask
                self.ptr.value = (self.ptr.value + 1) % self.max_size
                self.size.value = min(self.size.value + 1, self.max_size)

                if (
                    self.ptr.value
                    % self.config["train"]["save_buffer_every_n_observations"]
                    == 0
                ):
                    self.save_to_disk()

            print(f"Buffer size: {self.size.value}")

    def sample(self, batch_size: int) -> tuple:
        if self.size.value < batch_size:
            batch_size = self.size.value

        with self.lock:
            indices = np.random.choice(self.size.value, batch_size, replace=False)
            return (
                self.bay[indices],
                self.flat_T[indices],
                self.prob[indices],
                self.value[indices],
                self.containers_left[indices],
                self.mask[indices],
            )

    def __len__(self):
        return self.size.value
