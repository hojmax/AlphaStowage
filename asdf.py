from MPSPEnv import Env


def create_testset(config):
    testset = []
    for i in range(config["eval"]["testset_size"]):
        env = Env(
            config["env"]["R"],
            config["env"]["C"],
            config["env"]["N"],
            skip_last_port=True,
            take_first_action=True,
            strict_mask=True,
        )
        env.reset(i)
        testset.append(env)
    return testset


config = {"env": {"R": 6, "C": 2, "N": 6}, "eval": {"testset_size": 100}}
envs = create_testset(config)

for e in range(5):
    envs[e].print()

for env in envs:
    env.close()
