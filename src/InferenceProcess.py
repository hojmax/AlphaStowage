import torch
import numpy as np
import random
from MPSPEnv import Env
from Node import TruncatedEpisodeError
from Buffer import ReplayBuffer
from multiprocessing.connection import Connection
from multiprocessing import Queue
import time
from Node import (
    remove_all_pruning,
    close_envs_in_tree,
    Node,
    get_np_obs,
    alpha_zero_search,
)


def get_action(
    probabilities: torch.Tensor, deterministic: bool, config: dict, env: Env
) -> int:
    if deterministic:
        action = torch.argmax(probabilities).item()
    else:
        action = np.random.choice(2 * config["env"]["C"], p=probabilities)

    action = action if action < env.C else action + env.C - config["env"]["C"]

    return action


def close_other_branches(reused_tree: Node, action: int, env: Env) -> None:
    for a in range(2 * env.C):
        if a != action and a in reused_tree.children:
            close_envs_in_tree(reused_tree.children[a])


def update_tree(reused_tree: Node, action: int, env: Env) -> None:
    close_other_branches(reused_tree, action, env)

    reused_tree.env.close()
    reused_tree = reused_tree.children[action]
    reused_tree.parent = None
    reused_tree.prior_prob = None
    remove_all_pruning(reused_tree)
    return reused_tree


def add_value_to_observations(observations, final_value):
    output = []

    for i, sample in enumerate(observations):
        output.append(sample + [torch.tensor(final_value + i)])

    return output


def play_episode(env, conn, config, deterministic=False):
    if deterministic:
        np.random.seed(0)

    observations = []
    reused_tree = None
    transposition_table = {}

    while not env.terminal:
        reused_tree, probabilities, transposition_table = alpha_zero_search(
            env,
            conn,
            config,
            reused_tree,
            transposition_table,
        )
        bay, flat_T = get_np_obs(env, config)
        bay = torch.tensor(bay, dtype=torch.float32)
        flat_T = torch.tensor(flat_T, dtype=torch.float32)
        observations.append([bay, flat_T, probabilities])

        action = get_action(probabilities, deterministic, config, env)
        env.step(action)

        reused_tree = update_tree(reused_tree, action, env)

    close_envs_in_tree(reused_tree)

    final_value = -env.moves_to_solve
    reshuffles = env.total_reward
    observations = add_value_to_observations(observations, final_value)

    del transposition_table, reused_tree
    return observations, final_value, reshuffles


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
                observations, value, reshuffles = play_episode(
                    env, self.conn, self.config, deterministic=False
                )
            except TruncatedEpisodeError:
                continue
            finally:
                env.close()

            for obs in observations:
                self.buffer.extend(*obs)

            seconds = time.time() - start
            self.log_episode_queue.put(
                {"value": value, "reshuffles": reshuffles, "seconds/episode": seconds}
            )

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
