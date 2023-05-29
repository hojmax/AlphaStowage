from FloodEnv import FloodEnv
from NeuralNetwork import NeuralNetwork
import Node
import numpy as np

training_iterations = 1000
width = 5
height = 5
n_colors = 3
nn_blocks = 3
c_puct = 1
temperature = 1
search_iterations = 100
net = NeuralNetwork(n_colors=n_colors, width=width, height=height, n_blocks=nn_blocks)
all_data = []

for i in range(training_iterations):
    episode_data = []
    env = FloodEnv(width, height, n_colors)

    while not env.is_terminal():
        probabilities = Node.alphago_zero_search(
            env, net, search_iterations, c_puct, temperature
        )
        episode_data.append((env.get_tensor_state(), probabilities))
        action = np.random.choice(env.n_colors, p=probabilities)
        env.step(action)

    value = env.get_value()

    for state, probabilities in episode_data:
        all_data.append((state, probabilities, value))
