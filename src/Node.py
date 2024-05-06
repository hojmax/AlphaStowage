import numpy as np
from MPSPEnv import Env
import warnings


class TruncatedEpisodeError(Exception):
    pass


class Node:
    def __init__(
        self,
        env: Env,
        config: dict,
        prior_prob: float = 0,
        estimated_value: float = 0,
        parent: "Node" = None,
        depth: int = 0,
        action: int = None,
    ) -> None:
        self._env = env
        self.config = config
        self._pruned = False
        self.visit_count = np.float16(0)
        self.total_action_value = None
        self.estimated_value = np.float16(estimated_value)
        self.prior_prob = np.float16(prior_prob)
        self.children = {}
        self.parent = parent
        self.depth = depth
        self.c_puct = np.float16(self.get_c_puct(env, config))
        self._uct = None
        self.children_pruned = 0
        self.needed_action = action

    def get_c_puct(self, env: Env, config: dict) -> float:
        return config["mcts"]["c_puct_constant"] * env.N * env.R * env.C

    def add_noise(self) -> None:
        noise = np.random.dirichlet(
            np.full(self._env.C * 2, self.config["mcts"]["dirichlet_alpha"])
        )
        weight = np.float16(self.config["mcts"]["dirichlet_weight"])

        for action, child in self.children.items():
            child.prior_prob = np.float16(noise[action]) * weight + child.prior_prob * (
                np.float16(1) - weight
            )
            child._uct = None

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
        if self.total_action_value == None:
            return self.estimated_value
        else:
            return np.float16(self.total_action_value / np.float32(self.visit_count))

    @property
    def U(self) -> np.float16:
        return (
            self.c_puct
            * self.prior_prob
            * np.sqrt(self.parent.visit_count, dtype=np.float16)
            / (np.float16(1) + self.visit_count)
        )

    @property
    def uct(self) -> np.float16:
        if self._uct == None:
            self._uct = self.Q + self.U

        return self._uct

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

    def increment_value(self, value: float) -> None:
        if self.total_action_value == None:
            self.total_action_value = np.float32(value)
        else:
            self.total_action_value += np.float32(value)

        if self.visit_count < np.finfo(np.float16).max:
            self.visit_count += np.float16(1)
        else:
            warnings.warn("visit count overflow")

        self._uct = None
        for child in self.children.values():
            child._uct = None

    @property
    def no_valid_children(self) -> bool:
        return self.children_pruned == len(self.children)

    def get_valid_children(self) -> list["Node"]:
        return [child for child in self.children.values() if not child._pruned]

    def add_child(
        self, action: int, new_env: Env, prior: float, state_value: float, config: dict
    ) -> None:
        new_depth = self.depth + 1 if action < new_env.C else self.depth
        self.children[action] = Node(
            env=new_env,
            config=config,
            prior_prob=prior,
            estimated_value=state_value,
            parent=self,
            depth=new_depth,
            action=action,
        )

    def select_child(self) -> "Node":
        best_child = None

        for child in self.get_valid_children():
            if best_child is None or child.uct > best_child.uct:
                best_child = child

        return best_child
