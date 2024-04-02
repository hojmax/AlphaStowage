from Train import play_episode, get_env
import torch
from Node import TruncatedEpisodeError
import warnings
from Buffer import ReplayBuffer
from Logging import log_episode, init_wandb_run
import torch.multiprocessing as mp
import numpy as np
from NeuralNetwork import NeuralNetwork


class SelfPlay:

    def __init__(self, config: dict, checkpoint_path: str | None = None, seed: int = 0):
        self.config = config
        self.stop_event = mp.Event()
        self.update_event = mp.Event()
        self.model = NeuralNetwork(config)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.to("cpu")
        self.model.eval()
        torch.manual_seed(seed)
        np.random.seed(seed)

    def selfplay_loop(
        self, stop_event: mp.Event, update_event: mp.Event, buffer: ReplayBuffer
    ) -> None:

        if self.config["train"]["log_wandb"]:
            init_wandb_run(self.config)

        i = 0
        while not stop_event.is_set():
            if update_event.is_set():
                self.model.load_state_dict(
                    torch.load("shared_model.pt", map_location="cpu")
                )
                update_event.clear()

            env = get_env(self.config)
            env.reset(np.random.randint(1e9))

            try:
                print(f"playing episode {i}...")
                observations, final_value, final_reshuffles = play_episode(
                    env, self.model, self.config, "cpu", deterministic=False
                )
                print(f"finished episode {i}.")
                i += 1
                env.close()
            except TruncatedEpisodeError:
                warnings.warn("Episode was truncated in training.")
                env.close()
                continue

            for bay, flat_T, prob, value in observations:
                buffer.extend(bay, flat_T, prob, value)

            log_episode(
                buffer.increment_episode(),
                final_value,
                final_reshuffles,
                self.config,
            )
