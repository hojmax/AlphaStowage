import numpy as np
from MPSPEnv import Env
import warnings


class TruncatedEpisodeError(Exception):
    pass


class Node:
    def __init__(self, game_state: Env) -> None:
        self.Q = 0
        self.N = 0
        self.U = None
        self.P = None

        # action -> (child, P(n,a), number of edge_visits)
        # child is None if the action edge hasn't been explored
        self.children_and_edge_visits: dict[int, tuple["Node" | None, int]] = {}

        self.game_state = game_state
