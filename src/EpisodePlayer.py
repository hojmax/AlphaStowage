from MCTS import (
    remove_all_pruning,
    close_envs_in_tree,
    get_np_obs,
    alpha_zero_search,
)
from MPSPEnv import Env
from multiprocessing.connection import Connection
import torch
import numpy as np


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
        self.total_options_considered = 0

        if self.deterministic:
            np.random.seed(0)

    def run_episode(self):
        actions = []
        while not self.env.terminal:
            action = self._get_action()
            if action >= self.env.C:
                self.n_removes += 1
            actions.append(action)
            self.env.step(action)

        self._cleanup(actions, self.env.history)

        return (
            self.observations,
            self.final_value,
            self.reshuffles,
            self.n_removes / len(actions),
            self.total_options_considered / len(actions),
        )

    def _cleanup(self, actions: list[int], history: np.ndarray) -> None:
        close_envs_in_tree(self.reused_tree)

        self.final_value = -self.env.moves_to_solve
        self.reshuffles = self.env.total_reward

        self._add_value_to_observations(actions)
        self._add_history_to_observations(history)

    def _add_observation(self, probabilities: torch.Tensor, env: Env) -> None:
        bay, flat_T = get_np_obs(env, self.config)
        self.observations.append(
            [
                torch.tensor(bay),
                torch.tensor(flat_T),
                probabilities,
                torch.tensor([env.containers_left]),
            ]
        )

    def _update_considered_options(self, probabilities: torch.Tensor) -> None:
        n_options_considered = torch.sum(probabilities > 0).item()
        self.total_options_considered += n_options_considered

    def _get_action(self):
        probabilities, self.reused_tree, self.transposition_table = alpha_zero_search(
            self.env,
            self.conn,
            self.config,
            self.reused_tree,
            self.transposition_table,
        )
        self._update_considered_options(probabilities)
        self._add_observation(probabilities, self.env)
        action = self._probs_to_action(probabilities)
        self._update_tree(action)

        return action

    def _probs_to_action(self, probabilities: torch.Tensor) -> int:
        if self.deterministic:
            action = torch.argmax(probabilities).item()
        else:
            action = np.random.choice(2 * self.config["env"]["C"], p=probabilities)

        action = (
            action
            if action < self.env.C
            else action + self.env.C - self.config["env"]["C"]
        )

        return action

    def _close_other_branches(self, action: int) -> None:
        for a in range(2 * self.env.C):
            if a != action and a in self.reused_tree.children:
                close_envs_in_tree(self.reused_tree.children[a])

    def _update_tree(self, action: int) -> None:
        self._close_other_branches(action)

        self.reused_tree.env.close()
        self.reused_tree = self.reused_tree.children[action]
        self.reused_tree.parent = None
        self.reused_tree.prior_prob = None
        remove_all_pruning(self.reused_tree)

    def _add_value_to_observations(self, actions: list[int]) -> None:
        cummulative_removes = 0
        offset = 1 if self.env.take_first_action else 0
        for i in range(len(self.observations)):
            value = self.final_value + i - cummulative_removes + offset
            self.observations[i] += [torch.tensor(value)]

            if actions[i] >= self.env.C:
                cummulative_removes += 1

    def _add_history_to_observations(self, history: np.ndarray) -> None:
        history = np.pad(
            history,
            (
                (0, len(history)),
                (0, self.config["env"]["R"] - history.shape[1]),
                (0, self.config["env"]["C"] - history.shape[2]),
            ),
            mode="constant",
            constant_values=0,
        )

        offset = 1 if self.env.take_first_action else 0

        for i in range(len(self.observations)):
            self.observations[i] += [torch.tensor(history[i + offset])]
