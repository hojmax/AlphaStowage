import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from NeuralNetwork import NeuralNetwork
from MPSPEnv import Env
from Node import alpha_zero_search
from Train import get_config
import wandb


def _draw_tree_recursive(graph, node):
    for action, child in node.children.items():
        graph.add_node(
            str(hash(child)),
            label=str(child),
        )
        graph.add_edge(str(hash(node)), str(hash(child)), label=str(action))
        _draw_tree_recursive(graph, child)


def draw_tree(node):
    graph = nx.DiGraph()
    graph.add_node(str(hash(node)), label=str(node))

    _draw_tree_recursive(graph, node)

    pos = graphviz_layout(graph, prog="dot")
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")

    nx.draw(
        graph,
        pos,
        labels=labels,
        with_labels=True,
        node_size=4000,
        font_size=9,
        node_color="#00000000",
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)

    plt.gcf().set_size_inches(12, 7)
    plt.show()


run_path = "hojmax/multi-thread/32kirnc9"
model_path = "model26000.pt"
api = wandb.Api()
run = api.run(run_path)
file = run.file(model_path)
file.download(replace=True)
config = run.config

net = NeuralNetwork(config=config)
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()
# net = NeuralNetwork(config)

config = get_config()


env = Env(
    config["env"]["R"],
    config["env"]["C"],
    config["env"]["N"],
    skip_last_port=True,
    take_first_action=True,
    strict_mask=True,
)
env.reset_to_transportation(
    np.array(
        [
            [0, 2, 4, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 4],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
)
env.step(0)
env.step(0)
env.step(1)
env.step(0)
env.step(0)
# env.step(2)
# env.step(1)
# env.step(0)
# env.step(0)
# episode_data = play_episode(env, net, config, "cpu", deterministic=True)
# print(episode_data)
# for e in episode_data[0]:
#     print(e[0][0])
#     print(e[1])
#     print(e[2])
#     print()
for i in range(1, 100):
    np.random.seed(13)
    root, probs, transposition_table = alpha_zero_search(
        env,
        net,
        i,
        config["mcts"]["c_puct"],
        config["mcts"]["temperature"],
        config["mcts"]["dirichlet_weight"],
        config["mcts"]["dirichlet_alpha"],
        device="cpu",
    )
    draw_tree(root)

env.close()
