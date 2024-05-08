import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from NeuralNetwork import NeuralNetwork
from MPSPEnv import Env
from MCTS import get_np_obs
from main import get_config
import wandb
import os
import warnings
import pandas as pd
from main import start_process_loop, PretrainedModel
from GPUProcess import GPUProcess
import torch.multiprocessing as mp
from MCTS import alpha_zero_search
import time


def _draw_tree_recursive(graph, node):
    for action, child in node.children.items():
        graph.add_node(
            str(hash(child)),
            label=str(child),
        )
        graph.add_edge(str(hash(node)), str(hash(child)), label=str(action))
        _draw_tree_recursive(graph, child)


def draw_tree(node):
    graph = nx.DiGraph(ratio="fill", size="10,10")
    graph.add_node(str(hash(node)), label=str(node))

    _draw_tree_recursive(graph, node)

    pos = graphviz_layout(graph, prog="dot")
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")

    fig = plt.figure(1, figsize=(24, 17), dpi=60)

    nx.draw(
        graph,
        pos,
        labels=labels,
        with_labels=True,
        node_size=60,
        font_size=9,
        node_color="#00000000",
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)

    # plt.gcf().set_size_inches(12, 7)
    plt.show()


def run_search(iters, env, conn, config):
    np.random.seed(0)
    config["mcts"]["search_iterations"] = iters
    probabilities, reused_tree, transposition_table = alpha_zero_search(
        env, conn, config
    )
    draw_tree(reused_tree)
    print("search probs", probabilities)


class BaselinePolicy:
    def __init__(self, env):
        self.C = env.C
        self.N = env.N

    def predict(self, env):
        """Place the container in the rightmost non-filled column."""
        j = self.C - 1

        while j >= 1:
            if env.mask[j]:
                return j
            j -= 1

        return j


if __name__ == "__main__":
    print("Started...")
    env = Env(
        R=10,
        C=12,
        N=16,
        skip_last_port=True,
        take_first_action=True,
        strict_mask=True,
        speedy=True,
    )
    env.reset_to_transportation(
        np.array(
            [
                [0, 12, 19, 11, 8, 8, 4, 1, 1, 43, 1, 2, 0, 0, 10, 0],
                [0, 0, 2, 2, 0, 1, 0, 0, 0, 3, 0, 1, 0, 0, 0, 3],
                [0, 0, 0, 4, 0, 2, 0, 4, 2, 0, 0, 3, 1, 1, 2, 2],
                [0, 0, 0, 0, 3, 3, 2, 2, 0, 0, 0, 0, 2, 0, 3, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 4, 0, 1, 0, 3],
                [0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 5, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 4, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 3, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 4, 0, 13, 5, 6],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 5, 10, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 9, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 25],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
    )
    baseline_policy = BaselinePolicy(env)
    action = baseline_policy.predict(env)

    while not env.remaining_ports == 9:
        action = baseline_policy.predict(env)
        env.step(action)

    for _ in range(19):
        action = baseline_policy.predict(env)
        env.step(action)

    print(env.bay)
    print(env.T)

    config = get_config("config.json")
    gpu_update_event = mp.Event()
    inference_pipes = [mp.Pipe()]

    gpu_process = mp.Process(
        target=start_process_loop,
        args=(
            GPUProcess,
            inference_pipes,
            gpu_update_event,
            "mps",
            PretrainedModel(
                local_model="07-05.pt", wandb_model="", artifact="", wandb_run=""
            ),
            config,
        ),
    )
    gpu_process.start()
    print("GPU Process Started...")

    _, conn = inference_pipes[0]
    for i in range(1, 50):
        run_search(i, env, conn, config)

    env.close()


# env = Env(
#     12,
#     6,
#     14,
#     skip_last_port=True,
#     take_first_action=True,
#     strict_mask=True,
#     speedy=True,
# )
# env.reset_to_transportation(
#     np.array(
#         [
#             [0, 0, 3, 7, 4, 6, 0, 27, 0, 8, 10, 5, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
#             [0, 0, 0, 0, 1, 0, 2, 1, 1, 0, 0, 1, 1, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 1, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 3],
#             [0, 0, 0, 0, 0, 0, 0, 0, 7, 3, 2, 13, 4, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 2, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8, 4, 3],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 15, 2],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 21],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#         dtype=np.int32,
#     )
# )
