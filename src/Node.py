import numpy as np
from MPSPEnv import Env


class TruncatedEpisodeError(Exception):
    pass


class Node:
    def __init__(
        self,
        env: Env,
        config: dict,
        prior_prob: float = None,
        estimated_value: float = 0,
        parent: "Node" = None,
        depth: int = 0,
        action: int = None,
    ) -> None:
        self._env = env
        self.pruned = False
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = None
        self.estimated_value = estimated_value
        self.prior_prob = prior_prob
        self.children = {}
        self.parent = parent
        self.depth = depth
        self.c_puct = self.get_c_puct(env, config)
        self._uct = None
        self.children_pruned = 0
        self.needed_action = action

    def get_c_puct(self, env: Env, config: dict) -> float:
        return config["mcts"]["c_puct_constant"] * env.remaining_ports * env.R * env.C

    @property
    def env(self) -> Env:
        if self.needed_action is not None:
            self._env.step(self.needed_action)
            self.needed_action = None

        return self._env

    def close(self):
        self._env.close()

    @property
    def Q(self) -> float:
        return (
            self.mean_action_value
            if self.mean_action_value is not None
            else self.estimated_value
        )

    @property
    def U(self) -> float:
        return (
            self.c_puct
            * self.prior_prob
            * np.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )

    @property
    def uct(self) -> float:
        if self._uct == None:
            self._uct = self.Q + self.U

        return self._uct

    def prune(self) -> None:
        if self.parent == None:
            raise TruncatedEpisodeError

        if not self.pruned:
            self.parent.children_pruned += 1

        self.pruned = True

    def unprune(self) -> None:
        if self.parent != None and self.pruned:
            self.parent.children_pruned -= 1

        self.pruned = False

    def increment_value(self, value: float) -> None:
        self.total_action_value += value
        self.visit_count += 1
        self.mean_action_value = self.total_action_value / self.visit_count

        self._uct = None
        for child in self.children.values():
            child._uct = None

    @property
    def no_valid_children(self) -> bool:
        return self.children_pruned == len(self.children)

    def get_valid_children(self) -> list["Node"]:
        return [child for child in self.children.values() if not child.pruned]

    def add_child(
        self, action: int, new_env: Env, prior: float, state_value: float, config: dict
    ) -> None:
        self.children[action] = Node(
            env=new_env,
            config=config,
            prior_prob=prior,
            estimated_value=state_value,
            parent=self,
            depth=self.depth + 1,
            action=action,
        )

    def select_child(self) -> "Node":
        return max(self.get_valid_children(), key=lambda x: x.uct)

    def __str__(self) -> str:
        output = f"{self.env.bay}\n{self.env.T}\nN={self.visit_count}, Q={self.Q:.2f}\nMoves={self.env.moves_to_solve}"
        if self.prior_prob is not None:
            output += f" P={self.prior_prob:.2f}\nQ+U={self.uct:.2f}"
        if self.pruned:
            output = "pruned\n" + output
        return output
