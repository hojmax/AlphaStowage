from MPSPEnv import Env
from MPSPEnv.c_interface import c_lib
import numpy as np


class PaddedEnv(Env):

    def __init__(
        self,
        R: int,
        C: int,
        N: int,
        max_C: int,
        max_R: int,
        max_N: int,
        auto_move: bool = False,
        speedy: bool = False,
    ) -> None:
        super().__init__(R, C, N, auto_move, speedy)
        self.max_C = max_C
        self.max_R = max_R
        self.max_N = max_N

    def copy(self) -> "PaddedEnv":
        new_env = PaddedEnv(
            self.R,
            self.C,
            self.N,
            self.max_C,
            self.max_R,
            self.max_N,
            self.auto_move,
            self.speedy,
        )
        new_env._env = c_lib.copy_env(self._env)
        new_env._set_stores()

        return new_env

    def step(self, action: int):
        col = (action // self.max_R) % self.max_C
        n_containers = action % self.max_R + 1
        is_add = action < self.max_R * self.max_C
        unpacked_action = (
            col * self.R + n_containers - 1 + (1 - is_add) * self.R * self.C
        )
        super().step(unpacked_action)

    @property
    def mask(self) -> np.ndarray:
        mask = self.mask_store.ndarray.copy()
        mask = mask.astype(np.float32)
        mask = np.reshape(mask, (2, self.C, self.R))
        mask = np.pad(
            mask,
            ((0, 0), (0, self.max_C - self.C), (0, self.max_R - self.R)),
            "constant",
            constant_values=0,
        )
        return mask.flatten()

    @property
    def bay(self) -> np.ndarray:
        bay = self.bay_store.ndarray.copy()
        bay = bay.astype(np.float32)

        if self.remaining_ports > 0:
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
