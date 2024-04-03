from train import play_episode, get_env
import torch
from node import TruncatedEpisodeError
import warnings
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from logger import log_episode
import numpy as np
from network import NeuralNetwork
import ray


@ray.remote
class SelfPlay:
    def __init__(self, config: dict, checkpoint: dict, seed: int = 0):
        self.config = config
        self.model = NeuralNetwork(config)
        self.model.set_weights(checkpoint["best_weights"])
        self.model.to("cpu")
        self.model.eval()
        torch.manual_seed(seed)
        np.random.seed(seed)

    def self_play_loop(
        self, shared_storage: SharedStorage, replay_buffer: ReplayBuffer
    ) -> None:

        while ray.get(shared_storage.get_info.remote("training_step")) < self.config[
            "train"
        ]["training_steps"] and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(
                ray.get(shared_storage.get_info.remote("best_weights"))
            )

            env = get_env(self.config)
            env.reset(np.random.randint(1e9))

            try:
                observations, final_value, final_reshuffles = play_episode(
                    env, self.model, self.config, "cpu", deterministic=False
                )
                for bay, flat_T, prob, value in observations:
                    replay_buffer.extend.remote((bay, flat_T, prob, value))

                env.close()
            except TruncatedEpisodeError:
                warnings.warn("Episode was truncated in training.")
                env.close()
                continue

            log_episode(
                ray.get(shared_storage.get_info.remote("num_played_games")),
                final_value,
                final_reshuffles,
                self.config,
            )

            shared_storage.increment_info.remote("num_played_games")
