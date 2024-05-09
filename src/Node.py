import numpy as np
from MPSPEnv import Env
import warnings


class TruncatedEpisodeError(Exception):
    pass


class Node:
    def __init__(
        self,
        env: Env,
        prior_prob: float = 0,
        estimated_value: float = 0,
        parent: "Node" = None,
        depth: int = 0,
        action: int = None,
        c_puct_constant: float = 1,
        dirichlet_weight: float = 0.25,
        dirichlet_alpha: float = 1,
    ) -> None:
        self.Q = 0
        self.N = 0

        # action -> (child, number of edge_visits)
        self.children_and_edge_visits: dict[int, tuple["Node", int]] = {}

        self._env = env
        self._pruned = False

        self.total_utility = 0
        self.estimated_value = np.float16(estimated_value)
        self.prior_prob = np.float16(prior_prob)
        self.children = {}
        self.parent = parent
        self.depth = depth
        self.children_pruned = 0
        self.needed_action = action
        self.c_puct_constant = c_puct_constant
        self.c_puct = np.float16(c_puct_constant * env.N * env.R * env.C)
        self.dirichlet_weight = np.float16(dirichlet_weight)
        self.dirichlet_alpha = np.float16(dirichlet_alpha)

    def add_noise(self) -> None:

        if len(self.children) == 0:
            return

        for _, child in self.children.items():
            noise = np.random.dirichlet([self.dirichlet_alpha])
            child.prior_prob = np.float16(
                noise
            ) * self.dirichlet_weight + child.prior_prob * (
                np.float16(1) - self.dirichlet_weight
            )

    @property
    def env(self) -> Env:
        if self.needed_action is not None:
            self._env.step(self.needed_action)
            self.needed_action = None

        return self._env

    def close(self):
        self._env.close()

    def prune(self) -> None:
        if self.parent == None:
            raise TruncatedEpisodeError

        if not self._pruned:
            self.parent.children_pruned += 1

        self._pruned = True

    def unprune(self) -> None:
        if self.parent != None and self._pruned:
            self.parent.children_pruned -= 1

        self._pruned = False

    @property
    def no_valid_children(self) -> bool:
        return self.children_pruned == len(self.children)

    def get_valid_children(self) -> list["Node"]:
        return [child for child in self.children.values() if not child._pruned]

    def add_child(
        self, action: int, new_env: Env, prior: float, state_value: float
    ) -> None:
        new_depth = self.depth + 1 if action < new_env.C else self.depth
        self.children[action] = Node(
            env=new_env,
            prior_prob=prior,
            estimated_value=state_value,
            parent=self,
            depth=new_depth,
            action=action,
            c_puct_constant=self.c_puct_constant,
            dirichlet_weight=self.dirichlet_weight,
            dirichlet_alpha=self.dirichlet_alpha,
        )
