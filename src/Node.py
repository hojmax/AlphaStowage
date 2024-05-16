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
        self._U: float | None = None
        self._P: np.ndarray | None = None
        self._children_and_edge_visits: dict[int, tuple["Node", int]] = {}

        self.game_state = game_state

    @property
    def children_and_edge_visits(self) -> dict[int, tuple["Node", int]]:
        """action -> (child, number of edge_visits)"""

        # Filter out children where there is a quicker path to the same state
        valid_children = {}
        for action, (child, edge_visits) in self._children_and_edge_visits.items():
            has_valid_children = child.P is not None and child.P.sum() > 0
            is_quickest_path = child.best_depth > self.best_depth
            if has_valid_children and is_quickest_path:
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
            is_quickest_path = child.best_depth > self.best_depth
            if not is_quickest_path:
                quickest_path[action] = 0

        P = self._P * quickest_path
        P *= self.game_state.mask
        if P.sum() != 0:
            P /= P.sum()

        return P

    @P.setter
    def P(self, value):
        self._P = value

    @property
    def U(self) -> float:
        return self._U - self.best_depth if self._U else None

    @U.setter
    def U(self, value: float) -> None:
        self._U = value
