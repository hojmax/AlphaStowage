from MPSPEnv import Env


class BaselinePolicy:
    def __init__(self, C, N):
        self.C = C
        self.N = N

    def predict(self, one_hot_bay):
        """Place the container in the rightmost non-filled column."""
        j = self.C - 1

        while j >= 1:
            can_drop = True
            for h in range(self.N - 1):
                if one_hot_bay[h, 0, j] != 0:
                    can_drop = False
                    break
            if can_drop:
                break
            j -= 1

        return j


def run_episode(env):
    while not env.terminal:
        action = baseline_policy.predict(env.one_hot_bay)
        env.step(action)
    return env.moves_to_solve


repeats = 100

for R in range(6, 12 + 1, 2):
    for C in range(2, 12 + 1, 2):
        for N in range(4, 16 + 1, 2):
            baseline_policy = BaselinePolicy(C, N)
            total = 0
            for i in range(repeats):
                env = Env(R=R, C=C, N=N, skip_last_port=True, strict_mask=True)
                env.reset()
                total += run_episode(env)
            print(f"R={R}, C={C}, N={N}, Avg={total / repeats}")
