import torch
import numpy as np
import random
from PaddedEnv import PaddedEnv
from MPSPEnv import Env
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
        self.seed = seed
        self.conn = conn
        self.log_episode_queue = log_episode_queue
        self.config = config
        self.episode_count = 0

    def loop(self):
        games = []
        while True:
            env = self._get_env()

            player = EpisodePlayer(env, self.conn, self.config, deterministic=False)
            (
                observations,
                value,
                reshuffles,
                remove_fraction,
            ) = player.run_episode()

            self.buffer.extend(observations)

            self.log_episode_queue.put(
                {
                    "value": value,
                    "reshuffles": reshuffles,
                    "remove_fraction": remove_fraction,
                }
            )
            self.episode_count += 1
            games.append(reshuffles)
            if self.episode_count % 10 == 0:
                print(
                    f"Process: {self.seed} - Episode {self.episode_count} - Mean reshuffles: {np.mean(games)}"
                )
                games = []
            env.close()

    def _get_env(self) -> Env:
        env = PaddedEnv(
            R=random.choice(range(6, self.config["env"]["R"] + 1, 2)),
            C=random.choice(range(2, self.config["env"]["C"] + 1, 2)),
            N=random.choice(range(4, self.config["env"]["N"] + 1, 2)),
            max_C=self.config["env"]["C"],
            max_R=self.config["env"]["R"],
            max_N=self.config["env"]["N"],
            auto_move=True,
            speedy=True,
        )
        env.reset(np.random.randint(1e9))
        return env
