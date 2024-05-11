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
from Node import Node


def _draw_tree_recursive(graph, node):
    for action, child in node.children.items():
        graph.add_node(
            str(hash(child)),
            label=str(child),
        )
        graph.add_edge(str(hash(node)), str(hash(child)), label=str(action))
        _draw_tree_recursive(graph, child)


def node_to_str(node: Node) -> str:
    output = f"{node.env.bay}\n{node.env.T}\nN={node.visit_count},Q={node.Q:.2f}\nMoves={node.env.moves_to_solve}"
    if node.prior_prob is not None and node.parent != None:
        output += f" P={node.prior_prob:.2f}\nU={node.U},Q+U={node.uct:.2f}"
    if node._pruned:
        output = "pruned\n" + output
    return output


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
        R=8,
        C=4,
        N=6,
        skip_last_port=True,
        take_first_action=True,
        strict_mask=True,
        speedy=True,
        should_reorder=False,
    )
    env.reset_to_transportation(
        np.array(
            [
                [0, 14, 9, 2, 4, 3],
                [0, 0, 2, 6, 1, 5],
                [0, 0, 0, 2, 9, 0],
                [0, 0, 0, 0, 2, 8],
                [0, 0, 0, 0, 0, 16],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
    )
    env.step(0)
    env.step(1)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(6)
    env.step(1)
    config = get_config("local_config.json")
    device = torch.device("mps")
    model = NeuralNetwork(config, device)
    model.load_state_dict(torch.load("shared_model.pt", map_location=device))
    bceloss = torch.nn.BCELoss()

    bay_input, flat_T_input = get_np_obs(env, config)
    bay_input = torch.tensor(bay_input).unsqueeze(0).unsqueeze(0).float()
    flat_T_input = torch.tensor(flat_T_input).unsqueeze(0).float()
    containers_left = torch.tensor([[env.containers_left]]).float()
    print(bay_input.device, flat_T_input.device, containers_left.device, model.device)
    print(bay_input.shape, flat_T_input.shape, containers_left.shape)

    with torch.no_grad():
        model.eval()
        print(env.bay)
        print(bay_input)
        print(env.T)
        print(flat_T_input)
        policy, value, reshuffle = model(bay_input, flat_T_input, containers_left)
        policy = policy.squeeze()
        value = value.squeeze()
        print("policy", policy)
        print("value", value)
        print("reshuffle", reshuffle)
    example1 = torch.zeros_like(reshuffle, device=device)
    print(example1)
    example1[0, 0, 1, 1] = 1
    print(
        bceloss(
            reshuffle.to(device),
            torch.zeros_like(reshuffle, device=device),
        )
    )
    # print(
    #     bceloss(
    #         torch.zeros_like(reshuffle, device=device),
    #         torch.zeros_like(reshuffle, device=device),
    #     )
    # )
    print(
        bceloss(
            reshuffle.to(device),
            example1,
        )
    )
    # print(
    #     bceloss(
    #         torch.ones_like(reshuffle, device=device),
    #         torch.zeros_like(reshuffle, device=device),
    #     )
    # )

    # gpu_update_event = mp.Event()
    # inference_pipes = [mp.Pipe()]

    # gpu_process = mp.Process(
    #     target=start_process_loop,
    #     args=(
    #         GPUProcess,
    #         inference_pipes,
    #         gpu_update_event,
    #         "mps",
    #         PretrainedModel(
    #             local_model="shared_model_inv.pt",
    #             wandb_model="",
    #             artifact="",
    #             wandb_run="",
    #         ),
    #         config,
    #     ),
    # )
    # gpu_process.start()
    # print("GPU Process Started...")

    # _, conn = inference_pipes[0]
    # for i in range(1, 50):
    #     run_search(i, env, conn, config)

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
