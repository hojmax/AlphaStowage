from MPSPEnv import Env
from sb3_contrib.ppo_mask import MaskablePPO

env = Env(R=8, C=4, N=5, skip_last_port=True)
env.reset()

env.step(0)

env.print()

env.close()
