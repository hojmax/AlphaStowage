import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from main import get_config
from main import start_process_loop, PretrainedModel
from GPUProcess import GPUProcess
import torch.multiprocessing as mp
from MCTS import alpha_zero_search
from Node import Node
from PaddedEnv import PaddedEnv


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

    return output


def draw_tree(node):
    graph = nx.DiGraph(ratio="fill", size="10,10")
    graph.add_node(str(hash(node)), label=str(node))

    _draw_tree_recursive(graph, node)

    pos = graphviz_layout(graph, prog="dot")
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")

    fig = plt.figure(1, figsize=(24, 14), dpi=60)

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


if __name__ == "__main__":
    print("Started...")
    env = PaddedEnv(
        R=8, C=4, N=6, max_R=8, max_C=4, max_N=6, speedy=True, auto_move=True
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
    config = get_config("local_config.json")

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
                local_model="shared_model.pt",
                wandb_model="",
                artifact="",
                wandb_run="",
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
