from MPSPEnv import Env
from MPSPEnv.c_interface import c_lib
import numpy as np


class PaddedEnv(Env):

    def __init__(
        self,
        R: int,
        C: int,
        N: int,
        skip_last_port: bool = False,
        take_first_action: bool = False,
        strict_mask: bool = False,
        speedy: bool = False,
        max_C: int = 10,
        max_R: int = 10,
        max_N: int = 10,
    ) -> None:
        super().__init__(
            R, C, N, skip_last_port, take_first_action, strict_mask, speedy
        )
        self.max_C = max_C
        self.max_R = max_R
        self.max_N = max_N

    def copy(self, track_history: bool = True) -> "PaddedEnv":
        new_env = PaddedEnv(
            self.R,
            self.C,
            self.N,
            self.skip_last_port,
            self.take_first_action,
            self.strict_mask,
            self.speedy,
            self.max_C,
            self.max_R,
            self.max_N,
        )
        new_env._env = c_lib.copy_env(self._env, int(track_history))
        new_env.total_reward = self.total_reward
        new_env.action_probs = self.action_probs
        new_env.terminal = self.terminal
        new_env._port_tracker = self._port_tracker
        new_env.reshuffles_per_port = self.reshuffles_per_port
        new_env.steps_taken = self.steps_taken
        new_env._set_stores()

        return new_env

    def step(self, action: int):
        if action >= self.max_C:
            action = action - (self.max_C - self.C)

        super().step(action)

    @property
    def mask(self) -> np.ndarray:
        mask = self.mask_store.ndarray.copy()

        add_mask = np.pad(
            mask[: self.C], (0, self.max_C - self.C), mode="constant", constant_values=0
        )
        remove_mask = np.pad(
            mask[self.C :], (0, self.max_C - self.C), mode="constant", constant_values=0
        )
        return np.concatenate([add_mask, remove_mask])

    @property
    def bay(self) -> np.ndarray:
        bay = self.bay_store.ndarray.copy()
        bay = bay.astype(np.float32)
        bay = bay / self.remaining_ports
        bay = np.pad(
            bay,
            ((0, self.max_R - self.R), (0, self.max_C - self.C)),
            mode="constant",
            constant_values=-1,
        )
        return bay

    @property
    def flat_T(self) -> np.ndarray:
        T = self.T
        T = T.astype(np.float32)
        T = np.pad(
            T,
            ((0, self.max_N - self.N), (0, self.max_N - self.N)),
            mode="constant",
            constant_values=0,
        )
        i, j = np.triu_indices(n=T.shape[0], k=1)
        flat_T = T[i, j]
        flat_T = flat_T / (self.R * self.C)
        return flat_T
