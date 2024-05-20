import numpy as np
from MPSPEnv import Env


class TruncatedEpisodeError(Exception):
    pass


class Node:
    def __init__(self, game_state: Env) -> None:
        self.Q = 0
        self.N = 0
        self.best_depth = np.inf
        self.U: float | None = None
        self._P: np.ndarray | None = None

        # action -> (child, number of edge_visits)
        self.children_and_edge_visits: dict[int, tuple["Node", int]] = {}

        self.game_state = game_state

    def add_child(self, action: int, child: "Node") -> None:
        self.children_and_edge_visits[action] = (child, 0)

    def increment_visits(self, action: int) -> None:
        self.children_and_edge_visits[action] = (
            self.children_and_edge_visits[action][0],
            self.children_and_edge_visits[action][1] + 1,
        )

    @property
    def P(self) -> np.ndarray | None:
        """Policy distribution."""
        if self._P is None:
            return None

        return self._P * self.game_state.mask

    @P.setter
    def P(self, value):
        self._P = value
