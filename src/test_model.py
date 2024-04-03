import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from NeuralNetwork import NeuralNetwork
from MPSPEnv import Env
from Node import alpha_zero_search, get_torch_obs
from Train import get_config, test_network, create_testset, play_episode, get_action
import wandb
import os
import pandas as pd
from main import PretrainedModel


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


def test_on_benchmark(conn, config):
    testset = get_benchmarking_data("benchmark/set_2")
    testset = [e for e in testset if e["N"] == 6 and e["R"] == 6 and e["C"] == 2]
    testset = transform_benchmarking_data(testset)
    avg_error, avg_reshuffles = test_network(conn, testset, config)
    print("Average Error:", avg_error, "Average Reshuffles:", avg_reshuffles)


def get_pretrained_model(pretrained: PretrainedModel):
    api = wandb.Api()
    run = api.run(pretrained["wandb_run"])
    file = run.file(pretrained["wandb_model"])
    file.download(replace=True)
    config = run.config
    config["train"]["can_only_add"] = False

    model = NeuralNetwork(config=config, device="cpu")
    model.load_state_dict(torch.load(pretrained["wandb_model"], map_location="cpu"))

    return model


if __name__ == "__main__":
    pretrained = PretrainedModel(
        wandb_run="alphastowage/AlphaStowage/l3wodtt2", wandb_model="model116000.pt"
    )
    print("Pretrained Model:", pretrained)
    config = get_config("config.json")
    model = get_pretrained_model(pretrained)
    test_on_benchmark(model, config)

    # env = Env(
    #     6,
    #     4,
    #     4,
    #     skip_last_port=True,
    #     take_first_action=True,
    #     strict_mask=True,
    # )
    # env.reset()
    # env.step(0)
    # env.step(0)
    # env.step(0)
    # env.step(0)
    # env.step(0)
    # config["mcts"]["search_iterations"] = 100
    # root, probs, transposition_table = alpha_zero_search(env, net, "cpu", config)

    # for _ in range(1000):
    #     print(get_action(probs, False, config, env), end=" ")
    # print()
    # bay, flat_t = get_torch_obs(env, config)
    # probabilities, state_value = net(bay, flat_t)
    # print("Bay:", bay, "Flat T:", flat_t)
    # print("Bay:", env.bay, "T:", env.T)
    # print("Net Probs:", probabilities, "Net Value:", state_value)
    # print("MCTS Probs:", probs)
    # draw_tree(root)
    # # config["mcts"]["search_iterations"] = 100
    # # root, probs, transposition_table = alpha_zero_search(env, net, "cpu", config)
    # # print("MCTS Probs 2:", probs)
    # env.close()

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
#             [0, 0, 0, 0, 0, 5],
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
#     config=config,
# )
# env.print()
# torch.set_printoptions(precision=3, sci_mode=False)
# bay, flat_t = get_torch_obs(env)
# probabilities, state_value = net(bay, flat_t)
# print("Net Probs:", probabilities, "Net Value:", state_value)
# print("MCTS Probs:", probs)


# output_data, real_value, reshuffles = play_episode(
#     env, net, config, "cpu", deterministic=True
# )

# print("Real Value:", real_value, "Reshuffles:", reshuffles)
# for e in output_data:
#     with open("test.txt", "a") as f:
#         f.write(str(e[0][0].numpy()))
#         f.write("\n")
#         f.write(str(e[0][1].numpy()))
#         f.write("\n")
#         f.write(str(e[1].numpy()))
#         f.write("\n")
#         f.write(str(e[2]))
#         f.write("\n")
#         f.write("\n")

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
#         config=config,
#     )
#     draw_tree(root)

# env.close()
