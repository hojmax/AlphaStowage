import numpy as np
from MPSPEnv import Env
import warnings


class TruncatedEpisodeError(Exception):
    pass


class Node:
    def __init__(self, game_state: Env) -> None:
        self.Q = 0
        self.N = 0
        self.best_depth = np.inf
        self.U: float | None = None
        self._P: np.ndarray | None = None
        self._children_and_edge_visits: dict[int, tuple["Node", int]] = {}

        self.game_state = game_state

    @property
    def children_and_edge_visits(self) -> dict[int, tuple["Node", int]]:
        """action -> (child, number of edge_visits)"""

        # Filter out children where there is a quicker path to the same state
        valid_children = {}
        for action, (child, edge_visits) in self._children_and_edge_visits.items():
            if child.best_depth > self.best_depth:
                valid_children[action] = (child, edge_visits)

        return valid_children

    def add_child(self, action: int, child: "Node") -> None:
        self._children_and_edge_visits[action] = (child, 0)

    def increment_visits(self, action: int) -> None:
        self._children_and_edge_visits[action] = (
            self._children_and_edge_visits[action][0],
            self._children_and_edge_visits[action][1] + 1,
        )

    @property
    def P(self) -> np.ndarray | None:
        """Policy distribution."""
        if self._P is None:
            return None

        # Filter out actions that has a quicker path to the same state
        quickest_path = np.ones(len(self._P))
        for action, (child, _) in self._children_and_edge_visits.items():
            if child.best_depth <= self.best_depth:
                quickest_path[action] = 0

        P = self._P * quickest_path
        P *= self.game_state.mask
        P /= P.sum()
        return P

    @P.setter
    def P(self, value):
        self._P = value
