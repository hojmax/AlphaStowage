import numpy as np
from FloodEnv import FloodEnv
from NeuralNetwork import NeuralNetwork
import torch
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
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

    plt.gcf().set_size_inches(10, 7)
    plt.show()


class Node:
    def __init__(self, env, prior_prob, parent=None):
        self.env = env
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior_prob = prior_prob
        self.children = {}
        self.parent = parent

    def uct(self, cpuct, total_visit_count):
        return self.mean_action_value + cpuct * self.prior_prob * np.sqrt(
            total_visit_count
        ) / (1 + self.visit_count)

    def __str__(self):
        if self.prior_prob is None:
            return f"{self.env}\nN={self.visit_count}, Q={self.mean_action_value:.2f}"
        return f"{self.env}\nN={self.visit_count}, Q={self.mean_action_value:.2f}, P={self.prior_prob:.2f}"


def select(node, cpuct):
    total_visit_count = node.visit_count
    return max(node.children.values(), key=lambda x: x.uct(cpuct, total_visit_count))


def expand_and_evaluate(node, neural_network):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if node.env.is_terminal():
        return node.env.value
    with torch.no_grad():
        probabilities, state_value = neural_network(
            node.env.get_tensor_state().to(device)
        )
        probabilities = probabilities.detach().cpu().numpy().squeeze()
        state_value = state_value.detach().cpu().numpy().squeeze()

    for i in range(node.env.n_colors):
        if not node.env.valid_actions[i]:
            continue
        action = i
        prob = probabilities[action]
        new_env = node.env.copy()
        new_env.step(action)
        node.children[action] = Node(new_env, prob, parent=node)

    return state_value


def backup(node, value):
    node.visit_count += 1
    node.total_action_value += value
    node.mean_action_value = node.total_action_value / node.visit_count

    if node.parent is not None:
        backup(node.parent, value)


def get_tree_probs(node, temperature):
    action_probs = []
    for i in range(node.env.n_colors):
        if i in node.children:
            action_probs.append(np.power(node.children[i].visit_count, 1 / temperature))
        else:
            action_probs.append(0)
    action_probs /= np.sum(action_probs)
    return action_probs


def alphago_zero_search(root_env, neural_network, num_simulations, cpuct, temperature):
    root_node = Node(root_env, None)

    for i in range(num_simulations):
        node = root_node

        while node.children:
            node = select(node, cpuct)

        state_value = expand_and_evaluate(node, neural_network)
        backup(node, state_value)

    return root_node, get_tree_probs(root_node, temperature)


# Testing the tree search
if __name__ == "__main__":
    run_path = "hojmax/bachelor/tkek63zs"
    api = wandb.Api()
    run = api.run(run_path)
    file = run.file("model.pt")
    file.download(replace=True)
    config = run.config

    net = NeuralNetwork(
        n_colors=config["n_colors"],
        width=config["width"],
        height=config["height"],
        config=config["nn"],
    )
    net.load_state_dict(torch.load("model.pt"))
    net.eval()

    env = FloodEnv(
        n_colors=config["n_colors"],
        width=config["width"],
        height=config["height"],
    )
    env.reset(
        np.array(
            [
                [0, 1, 1],
                [2, 1, 1],
                [0, 2, 2],
            ]
        )
    )
    for i in range(1, 10):
        root, probs = alphago_zero_search(
            env,
            net,
            i,
            config["c_puct"],
            config["temperature"],
        )
        print(probs)
        draw_tree(root)
