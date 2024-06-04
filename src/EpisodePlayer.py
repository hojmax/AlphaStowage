from MCTS import (
    close_envs_in_tree,
    alpha_zero_search,
)
from MPSPEnv import Env
from multiprocessing.connection import Connection
import torch
import numpy as np
from min_max import MinMaxStats


class EpisodePlayer:
    def __init__(
        self,
        env: Env,
        conn: Connection,
        config: dict,
        deterministic: bool,
    ) -> None:
        self.env = env
        self.conn = conn
        self.config = config
        self.deterministic = deterministic
        self.observations = []
        self.reused_tree = None
        self.transposition_table = {}
        self.n_removes = 0
        self.min_max_stats = MinMaxStats()

        if self.deterministic:
            np.random.seed(0)

    def run_episode(self):
        placed = []
        while not self.env.terminated:
            action = self._get_action()
            if action >= self.env.C * self.env.R:
                self.n_removes += 1
            placed.append(self.env.containers_placed)
            self.env.step(action)

        self._cleanup(placed)

        return (
            self.observations,
            -self.env.containers_placed,
            self.env.total_reward,
            self.n_removes / len(placed) if len(placed) > 0 else 0,
        )

    def _cleanup(self, placed: list[int]) -> None:
        if self.reused_tree is not None:
            close_envs_in_tree(self.reused_tree)

        self._add_value_to_observations(placed)

    def _add_observation(self, probabilities: torch.Tensor, env: Env) -> None:
        self.observations.append(
            [
                torch.tensor(env.bay),
                torch.tensor(env.flat_T),
                probabilities,
                torch.tensor([env.containers_left]),
                torch.tensor(env.mask),
            ]
        )

    def _get_action(self):
        probabilities, self.reused_tree, self.transposition_table = alpha_zero_search(
            self.env,
            self.conn,
            self.config,
            self.min_max_stats,
            self.reused_tree,
            self.transposition_table,
        )
        self._add_observation(probabilities, self.env)
        action = torch.argmax(probabilities).item()
        self._update_tree(action)
        return action

    def _close_other_branches(self, action: int) -> None:
        for key in self.reused_tree.children.keys():
            if key != action:
                close_envs_in_tree(self.reused_tree.children[key])

    def _update_tree(self, action: int) -> None:
        self._close_other_branches(action)

        self.reused_tree.env.close()
        self.reused_tree = self.reused_tree.children[action]
        self.reused_tree.parent = None
        self.reused_tree.prior_prob = None

    def _add_value_to_observations(self, placed: list[int]) -> None:
        overall_placed = -self.env.containers_placed
        for i in range(len(self.observations)):
            self.observations[i] += [torch.tensor(overall_placed + placed[i])]
