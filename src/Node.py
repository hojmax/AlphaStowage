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
        self.visit_count = np.int32(0)
        self.total_action_value = None
        self.prior_prob = np.float16(prior_prob)
        self.children = {}
        self.parent = parent
        self.depth = depth
        self.needed_action = action
        self._Q = None
        self._U = None
        self.estimate = None

    def add_noise(self) -> None:
        alpha = (
            0.03
            * 2
            * self.config["env"]["C"]
            * self.config["env"]["R"]
            / len(self.children)
        )
        noise = np.random.dirichlet(np.full(len(self.children), alpha))
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
            return None
        else:
            return np.float16(self.total_action_value / self.visit_count)

    @property
    def U(self) -> np.float16:
        if self._U:
            return self._U

        return (
            self.c_puct
            * self.prior_prob
            * np.sqrt(self.parent.visit_count, dtype=np.float32)
            / (np.float16(1) + self.visit_count)
        )

    @property
    def c_puct(self) -> float:
        base = np.float16(self.config["mcts"]["c_puct_base"])
        init = np.float16(self.config["mcts"]["c_puct_init"])
        return np.log((self.parent.visit_count + base + np.float16(1)) / base) + init

    def increment_value(self, value: float) -> None:
        if self.total_action_value == None:
            self.estimate = value
            self.total_action_value = np.float32(value)
        else:
            self.total_action_value += np.float32(value)

        self.visit_count += np.int32(1)

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
        best_uct = -np.inf

        for child in self.children.values():
            Q = min_max_stats.normalize(
                child.Q if child.Q != None else min_max_stats.minimum
            )
            uct = Q + child.U
            if uct > best_uct:
                best_child = child
                best_uct = uct

        if best_child == None:
            return list(self.children.values())[0]
        else:
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
