import numpy as np
from MPSPEnv import Env


class Node:
    def __init__(
        self,
        env: Env,
        c_puct: float,
        prior_prob: float = None,
        estimated_value: float = 0,
        parent: "Node" = None,
        depth: int = 0,
    ) -> None:
        self.env = env
        self.pruned = False
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = None
        self.estimated_value = estimated_value
        self.prior_prob = prior_prob
        self.children = {}
        self.parent = parent
        self.depth = depth
        self.c_puct = c_puct

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

    def increment_value(self, value: float) -> None:
        self.total_action_value += value
        self.visit_count += 1
        self.mean_action_value = self.total_action_value / self.visit_count

    @property
    def uct(self) -> float:
        return self.Q + self.U

    @property
    def valid_children(self) -> list["Node"]:
        return [child for child in self.children.values() if not child.pruned]

    def select_child(self) -> "Node":
        return max(self.valid_children, key=lambda x: x.uct)

    def __str__(self) -> str:
        output = f"{self.env.bay}\n{self.env.T}\nN={self.visit_count}, Q={self.Q:.2f}\nMoves={self.env.moves_to_solve}"
        if self.prior_prob is not None:
            output += f" P={self.prior_prob:.2f}\nQ+U={self.uct:.2f}"
        if self.pruned:
            output = "pruned\n" + output
        return output
