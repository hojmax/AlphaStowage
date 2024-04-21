import torch
import numpy as np
import random
from MPSPEnv import Env
from MCTS import TruncatedEpisodeError
from Buffer import ReplayBuffer
from multiprocessing.connection import Connection
from multiprocessing import Queue
import time
from EpisodePlayer import EpisodePlayer
import gc


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

    def loop(self):
        while True:
            env = self._get_env()
            start = time.time()

            try:
                player = EpisodePlayer(env, self.conn, self.config, deterministic=False)
                observations, value, reshuffles, removes = player.run_episode()
            except TruncatedEpisodeError:
                continue
            finally:
                env.close()

            for obs in observations:
                self.buffer.extend(*obs)

            seconds = time.time() - start
            self.log_episode_queue.put(
                {
                    "value": value,
                    "reshuffles": reshuffles,
                    "seconds/episode": seconds,
                    "removes/episode": removes,
                }
            )
            del env, player, observations
            gc.collect()

    def _get_env(self) -> Env:
        env = Env(
            random.choice(range(6, self.config["env"]["R"] + 1, 2)),
            random.choice(range(2, self.config["env"]["C"] + 1, 2)),
            random.choice(range(4, self.config["env"]["N"] + 1, 2)),
            skip_last_port=True,
            take_first_action=True,
            strict_mask=True,
        )
        env.reset(np.random.randint(1e9))
        return env
