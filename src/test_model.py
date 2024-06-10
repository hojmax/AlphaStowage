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

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import plotly.graph_objects as go
from min_max import MinMaxStats


def _draw_tree_recursive(
    graph: nx.DiGraph,
    node: Node,
    nodes: set[Node],
    node_colors: dict,
    node_data: dict,
    edge_data: dict,
):
    for (
        action,
        child,
    ) in node.children.items():  # Assuming edge_value is the value to check

        if child in nodes or child is None:
            continue

        if child.visit_count == 0:
            continue

        graph.add_edge(str(hash(node)), str(hash(child)))

        graph.add_node(str(hash(child)))

        nodes.add(child)
        node_colors[str(hash(child))] = child.Q  # Store the Q value
        node_data[str(hash(child))] = {
            "Q": child.Q,
            "U": child.estimated_value,
            "Placed": child.env.containers_placed,
            "N": child.visit_count,
            "P": child.prior_prob,
            "Action": action_to_string(action, child.env.max_R, child.env.max_C),
        }  # Store additional data
        _draw_tree_recursive(graph, child, nodes, node_colors, node_data, edge_data)


def draw_tree(node: Node):
    nodes = set([node])
    graph = nx.MultiDiGraph()
    graph.add_node(str(hash(node)))

    node_colors = {str(hash(node)): node.Q}
    node_data = {
        str(hash(node)): {
            "Q": node.Q,
            "U": node.estimated_value,
            "Placed": node.env.containers_placed,
            "N": node.visit_count,
        }
    }
    edge_data = {}

    _draw_tree_recursive(graph, node, nodes, node_colors, node_data, edge_data)

    pos = graphviz_layout(graph, prog="dot")

    edge_x = []
    edge_y = []
    edge_hovertext = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="gray"),
        hovertext=edge_hovertext,
        hoverinfo="text",
        mode="lines+markers",
        marker=dict(
            size=1,
            color="gray",
            symbol="arrow-bar-up",  # Symbol to represent the direction
        ),
    )

    node_x = []
    node_y = []
    node_hovertext = []
    node_Q_values = []

    for node in graph.nodes():
        hover_text = ""
        for key, value in node_data[node].items():
            hover_text += f"{key}: {value}, "
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_hovertext.append(hover_text)
        node_Q_values.append(node_data[node]["Q"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=node_hovertext,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_Q_values,
            size=10,
            colorbar=dict(
                thickness=15, title="Q value", xanchor="left", titleside="right"
            ),
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )
    fig.show()


# def _draw_tree_recursive(graph, node):
#     for action, child in node.children.items():
#         graph.add_node(
#             str(hash(child)),
#             label=node_to_str(child),
#         )
#         graph.add_edge(
#             str(hash(node)),
#             str(hash(child)),
#             label=action_to_string(action, node.env.max_R, node.env.max_C)
#             + f" {child.prior_prob:.2f}",
#         )
#         _draw_tree_recursive(graph, child)


# def node_to_str(node: Node) -> str:
#     # output = f"{node.env.bay}\n{node.env.T}\nN={node.visit_count},Q={node.Q:.2f}\nMoves={node.env.moves_to_solve}"
#     # if node.prior_prob is not None and node.parent != None:
#     #     output += f" P={node.prior_prob:.2f}\nU={node.U},Q+U={node.uct:.2f}"

#     output = f"N={node.visit_count},Q={node.Q:.2f}"

#     return output


def action_to_string(action, R, C):
    if action >= R * C:
        return f"r{action % R + 1}c{(action - R * C) // R}"
    else:
        return f"a{action % R + 1}c{action // R}"


# def draw_tree(node):
#     graph = nx.DiGraph(ratio="fill", size="10,10")
#     graph.add_node(str(hash(node)), label=node_to_str(node))

#     _draw_tree_recursive(graph, node)

#     pos = graphviz_layout(graph, prog="dot")
#     labels = nx.get_node_attributes(graph, "label")
#     edge_labels = nx.get_edge_attributes(graph, "label")

#     fig = plt.figure(1, figsize=(24, 14), dpi=60)

#     nx.draw(
#         graph,
#         pos,
#         labels=labels,
#         with_labels=True,
#         node_size=60,
#         font_size=9,
#         node_color="#00000000",
#     )
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)

#     # plt.gcf().set_size_inches(12, 7)
#     plt.show()


def run_search(iters, env, conn, config, min_max_stats):
    np.random.seed(0)
    config["mcts"]["search_iterations"] = iters
    probabilities, reused_tree, transposition_table = alpha_zero_search(
        env, conn, config, min_max_stats
    )
    draw_tree(reused_tree)
    print("search probs", probabilities)


if __name__ == "__main__":
    print("Started...")
    env = PaddedEnv(
        R=10, C=10, N=10, max_R=12, max_C=12, max_N=16, speedy=True, auto_move=True
    )
    env.reset()
    # env.reset_to_transportation(
    #     np.array(
    #         [
    #             [0, 14, 9, 2, 4, 3],
    #             [0, 0, 2, 6, 1, 5],
    #             [0, 0, 0, 2, 9, 0],
    #             [0, 0, 0, 0, 2, 8],
    #             [0, 0, 0, 0, 0, 16],
    #             [0, 0, 0, 0, 0, 0],
    #         ],
    #         dtype=np.int32,
    #     )
    # )
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
                local_model=config["wandb"]["local_model"],
                wandb_model="",
                artifact=config["wandb"]["artifact"],
                wandb_run=config["wandb"]["pretrained_run"],
            ),
            config,
        ),
    )
    gpu_process.start()
    print("GPU Process Started...")

    _, conn = inference_pipes[0]
    min_max_stats = MinMaxStats()
    # for i in range(1, 50):
    #     run_search(i, env, conn, config, min_max_stats)
    run_search(800, env, conn, config, min_max_stats)

    env.close()
