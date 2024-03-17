from MPSPEnv import Env

env = Env(R=8, C=4, N=5, skip_last_port=True)
env.reset()

env.step(0)

env.print()

env.close()
