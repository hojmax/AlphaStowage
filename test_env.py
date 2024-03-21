from MPSPEnv import Env
from Train import baseline_policy
from matplotlib import pyplot as plt
import numpy as np

# hardest_transportation = None
# hardest_score = float("-inf")
# repeats = 100000
# moves = []
# np.random.seed(0)
# for i in range(repeats):
#     env = Env(R=6, C=2, N=6, skip_last_port=True)
#     env.reset()
#     transportation = env.T

#     while not env.terminal:
#         action, probabilities = baseline_policy(env)
#         env.step(action)

#     env.close()
#     moves.append(env.moves_to_solve)
#     if env.moves_to_solve > hardest_score:
#         hardest_score = env.moves_to_solve
#         hardest_transportation = transportation

# print(hardest_score)
# print(hardest_transportation)

# # show histogram
# plt.hist(moves)
# plt.show()

# dirichlet_alpha = 1
# probabilities = np.array([0.5, 0.5])
# noise = np.random.dirichlet(np.zeros_like(probabilities) + dirichlet_alpha)
# print(noise)

env = Env(R=6, C=2, N=6, skip_last_port=True)
env.reset_to_transportation(
    np.array(
        [
            [0, 10, 0, 0, 0, 2],
            [0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
)

while not env.terminal:
    action, probabilities = baseline_policy(env)
    env.print()
    env.step(action)


env.print()
print("Moves used:", env.moves_to_solve)
env.close()
