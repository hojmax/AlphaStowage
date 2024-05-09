import torch
import numpy as np
import random
from MPSPEnv import Env
from Node import TruncatedEpisodeError
from Buffer import ReplayBuffer
from multiprocessing.connection import Connection
from multiprocessing import Queue
from EpisodePlayer import EpisodePlayer


class InferenceProcess:
    def __init__(
        self,
        seed: int,
        buffer: ReplayBuffer,
        conn: Connection,
        log_episode_queue: Queue,
        config: dict,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.buffer = buffer
        self.conn = conn
        self.log_episode_queue = log_episode_queue
        self.config = config
        self.episode_count = 0

    def loop(self):
        while True:
            self.config["inference"]["can_only_add"] = (
                self.episode_count
                < self.config["inference"]["n_episodes_with_only_add"]
            )
            env = self._get_env()

            try:
                player = EpisodePlayer(env, self.conn, self.config, deterministic=False)
                (
                    observations,
                    value,
                    reshuffles,
                    remove_fraction,
                    avg_options_considered,
                ) = player.run_episode()
            except (TruncatedEpisodeError, KeyError):
                continue
            finally:
                env.close()

            for obs in observations:
                self.buffer.extend(*obs)

            self.log_episode_queue.put(
                {
                    "value": value,
                    "reshuffles": reshuffles,
                    "remove_fraction": remove_fraction,
                    "avg_options_considered": avg_options_considered,
                }
            )
            self.episode_count += 1

    def _get_env(self) -> Env:
        env = Env(
            random.choice(range(6, self.config["env"]["R"] + 1, 2)),
            random.choice(range(2, self.config["env"]["C"] + 1, 2)),
            random.choice(range(4, self.config["env"]["N"] + 1, 2)),
            skip_last_port=True,
            take_first_action=True,
            strict_mask=True,
            speedy=True,
        )
        env.reset(np.random.randint(1e9))
        return env
