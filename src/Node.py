import numpy as np
from MPSPEnv import Env
import warnings
from min_max import MinMaxStats


class Node:
    def __init__(
        self,
        env: Env,
        config: dict,
        prior_prob: float = 0,
        parent: "Node" = None,
        depth: int = 0,
        action: int = None,
    ) -> None:
        self._env = env
        self.config = config
        self.visit_count = np.float16(0)
        self.total_action_value = 0
        self.prior_prob = np.float16(prior_prob)
        self.children = {}
        self.parent = parent
        self.depth = depth
        self.needed_action = action
        self._Q = None
        self._U = None

    def get_c_puct(self, env: Env, config: dict) -> float:
        return config["mcts"]["c_puct_constant"] * env.N * env.R * env.C

    def add_noise(self) -> None:
        noise = np.random.dirichlet(
            np.full(len(self.children), self.config["mcts"]["dirichlet_alpha"])
        )
        weight = np.float16(self.config["mcts"]["dirichlet_weight"])

        for child_index, child in enumerate(self.children.values()):
            child.prior_prob = np.float16(
                noise[child_index]
            ) * weight + child.prior_prob * (np.float16(1) - weight)
            child.clear_cache()

    @property
    def env(self) -> Env:
        if self.needed_action is not None:
            self._env.step(self.needed_action)
            self.needed_action = None

        return self._env

    def close(self):
        self._env.close()

    @property
    def Q(self) -> np.float16:
        if self._Q:
            return self._Q

        if self.visit_count == 0:
            return 0
        else:
            return np.float16(self.total_action_value / np.float32(self.visit_count))

    @property
    def U(self) -> np.float16:
        if self._U:
            return self._U

        return (
            self.c_puct
            * self.prior_prob
            * np.sqrt(self.parent.visit_count, dtype=np.float16)
            / (np.float16(1) + self.visit_count)
        )

    @property
    def c_puct(self) -> float:
        base = self.config["mcts"]["c_puct_base"]
        init = self.config["mcts"]["c_puct_init"]
        return np.log((self.parent.visit_count + base + 1) / base) + init

    def increment_value(self, value: float) -> None:
        if self.total_action_value == None:
            self.total_action_value = np.float32(value)
        else:
            self.total_action_value += np.float32(value)

        if self.visit_count < np.finfo(np.float16).max:
            self.visit_count += np.float16(1)
        else:
            warnings.warn("visit count overflow")

        self.clear_cache()

    def add_child(self, action: int, new_env: Env, prior: float, config: dict) -> None:
        new_depth = self.depth + 1 if action < new_env.C else self.depth
        self.children[action] = Node(
            env=new_env,
            config=config,
            prior_prob=prior,
            parent=self,
            depth=new_depth,
            action=action,
        )

    def select_child(self, min_max_stats: MinMaxStats) -> "Node":
        best_child = None
        best_uct = 0

        for child in self.children.values():
            Q = min_max_stats.normalize(child.Q)
            uct = Q + child.U
            if uct > best_uct:
                best_child = child
                best_uct = uct

        return best_child

    def __str__(self) -> str:
        output = f"{self.env.bay_store.ndarray}\n{self.env.T}\nN={self.visit_count}, Q={self.Q:.2f}\nleft={self.env.containers_left}, placed={self.env.containers_placed}"
        if self.parent != None:
            output += f"\nP={self.prior_prob:.2f},  U={self.U:.2f}"
        return output

    def clear_cache(self):
        self._Q = None
        self._U = None
        for child in self.children.values():
            child._U = None
