from MPSPEnv import Env
from multiprocessing.connection import Connection
import torch
import numpy as np
from mcgs import MCGS, Node


class GPUModel:
    def __init__(self, conn: Connection) -> None:
        self.conn = conn

    def __call__(
        self, bay: torch.Tensor, flat_T: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.conn.send([bay, flat_T])
        return self.conn.recv()


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
        self.tree = Node(self.env.copy())
        self.transposition_table = {}
        self.n_removes = 0
        self.total_options_considered = 0
        self.mcgs = MCGS(
            GPUModel(conn),
            c_puct=config["mcts"]["c_puct_constant"] * env.N * env.R * env.C,
            dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
            dirichlet_weight=config["mcts"]["dirichlet_weight"],
        )

        if self.deterministic:
            np.random.seed(0)

    def run_episode(self):
        actions = []
        while not self.env.terminal:
            action = self._get_action()

            actions.append(action)
            self.env.step(action)
            self.tree = self.tree.children_and_edge_visits[action][0]

        self.final_value = -self.env.moves_to_solve
        self.reshuffles = self.env.total_reward

        self._add_value_to_observations(actions)

        return (
            self.observations,
            self.final_value,
            self.reshuffles,
            self.n_removes / len(actions),
            self.total_options_considered / len(actions),
        )

    def _add_observation(self, probabilities: torch.Tensor, env: Env) -> None:
        bay, flat_T = env.bay, env.flat_T
        self.observations.append(
            [torch.tensor(bay), torch.tensor(flat_T), probabilities]
        )

    def _update_considered_options(self, probabilities: torch.Tensor) -> None:
        n_options_considered = torch.sum(probabilities > 0).item()
        self.total_options_considered += n_options_considered

    def _get_action(self):
        self.mcgs.run(
            self.tree, search_iterations=self.config["mcts"]["search_iterations"]
        )
        action_probs = torch.zeros(len(self.tree.P), dtype=torch.float64)
        for i, (_, edge_visits) in self.tree.children_and_edge_visits.items():
            value = np.power(edge_visits, 1 / self.config["mcts"]["temperature"])
            action_probs[i] = value

        probs = action_probs / torch.sum(action_probs)
        self._update_considered_options(probs)
        self._add_observation(probs, self.env)
        action = self._probs_to_action(probs)
        return action

    def _probs_to_action(self, probabilities: torch.Tensor) -> int:
        if self.deterministic:
            action = torch.argmax(probabilities).item()
        else:
            action = np.random.choice(len(probabilities), p=probabilities)

        return action

    def _add_value_to_observations(self, actions: list[int]) -> None:
        cummulative_removes = 0
        offset = 1 if self.env.take_first_action else 0
        for i in range(len(self.observations)):
            value = self.final_value + i - cummulative_removes + offset
            self.observations[i] += [torch.tensor(value)]

            if actions[i] >= self.env.C:
                cummulative_removes += 1
