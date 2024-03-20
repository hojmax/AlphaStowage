import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from NeuralNetwork import NeuralNetwork
from MPSPEnv import Env
from Node import alpha_zero_search, get_torch_obs
from Train import get_config, test_network, create_testset
import wandb
import os
import pandas as pd


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


def get_benchmarking_data(path):
    """Go through all files in the directory and return a list of dictionaries with:
    N, R, C, seed, transportation_matrix, paper_result"""

    output = []
    df = pd.read_excel(
        os.path.join(path, "paper_results.xlsx"),
    )

    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r") as f:
                lines = f.readlines()
                N = int(lines[0].split(": ")[1])
                R = int(lines[1].split(": ")[1])
                C = int(lines[2].split(": ")[1])
                seed = int(lines[3].split(": ")[1])

                paper_result = df[
                    (df["N"] == N)
                    & (df["R"] == R)
                    & (df["C"] == C)
                    & (df["seed"] == seed)
                ]["res"].values

                assert len(paper_result) == 1
                paper_result = paper_result[0]

                output.append(
                    {
                        "N": N,
                        "R": R,
                        "C": C,
                        "seed": seed,
                        "transportation_matrix": np.loadtxt(lines[4:], dtype=np.int32),
                        "paper_result": paper_result,
                    }
                )

    return output


def transform_benchmarking_data(data):
    testset = []
    for instance in data:
        env = Env(
            instance["R"],
            instance["C"],
            instance["N"],
            skip_last_port=True,
            take_first_action=True,
            strict_mask=True,
        )
        env.reset_to_transportation(instance["transportation_matrix"])
        testset.append(env)
    return testset


run_path = "alphastowage/AlphaStowage/2gqbl328"
model_path = "model360000.pt"
api = wandb.Api()
run = api.run(run_path)
file = run.file(model_path)
file.download(replace=True)
config = run.config

net = NeuralNetwork(config=config)
net.load_state_dict(torch.load(model_path, map_location="cpu"))

config = get_config()
config["use_baseline_policy"] = False
test_set = create_testset(config)
avg_error, avg_reshuffles = test_network(net, test_set, config, "cpu")
print("Random testset:")
print("Eval Moves:", avg_error, "Eval Reshuffles:", avg_reshuffles)


data = get_benchmarking_data("/Users/axelhojmark/Desktop/rl-mpsp-benchmark/set_2")
data = [
    e
    for e in data
    if e["N"] == config["env"]["N"]
    and e["R"] == config["env"]["R"]
    and e["C"] == config["env"]["C"]
]
data = transform_benchmarking_data(data)

avg_error, avg_reshuffles = test_network(net, data, config, "cpu")
print("Benchmarking testset:")
print("Eval Moves:", avg_error, "Eval Reshuffles:", avg_reshuffles)

# env = Env(
#     config["env"]["R"],
#     config["env"]["C"],
#     config["env"]["N"],
#     skip_last_port=True,
#     take_first_action=True,
#     strict_mask=True,
# )
# env.reset_to_transportation(
#     np.array(
#         [
#             [0, 10, 0, 0, 0, 2],
#             [0, 0, 5, 5, 0, 0],
#             [0, 0, 0, 0, 5, 0],
#             [0, 0, 0, 0, 0, 5],
#             [0, 0, 0, 0, 0, 2],
#             [0, 0, 0, 0, 0, 0],
#         ],
#         dtype=np.int32,
#     )
# )

# root, probs, transposition_table = alpha_zero_search(
#     env,
#     net,
#     100,
#     config["mcts"]["c_puct"],
#     config["mcts"]["temperature"],
#     config["mcts"]["dirichlet_weight"],
#     config["mcts"]["dirichlet_alpha"],
#     device="cpu",
# )
# env.print()
# torch.set_printoptions(precision=3, sci_mode=False)
# bay, flat_t, mask = get_torch_obs(env)
# probabilities, state_value = net(bay, flat_t, mask)
# print("Net Probs:", probabilities, "Net Value:", state_value)
# print("MCTS Probs:", probs)

# for i in range(99, 100):
#     np.random.seed(13)
#     root, probs, transposition_table = alpha_zero_search(
#         env,
#         net,
#         i,
#         config["mcts"]["c_puct"],
#         config["mcts"]["temperature"],
#         config["mcts"]["dirichlet_weight"],
#         config["mcts"]["dirichlet_alpha"],
#         device="cpu",
#     )
#     draw_tree(root)

# env.close()
