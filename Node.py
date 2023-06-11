import numpy as np
from FloodEnv import FloodEnv
from NeuralNetwork import NeuralNetwork
import torch
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


def _draw_tree_recursive(graph, node):
    for action, child in node.children.items():
        graph.add_node(
            str(hash(child)),
            label=f"{np.array2string(child.env.state, separator=',')}\nN={child.visit_count}, Q={child.mean_action_value:.2f}, P={child.prior_prob:.2f}",
        )
        graph.add_edge(str(hash(node)), str(hash(child)), label=str(action))
        _draw_tree_recursive(graph, child)


def draw_tree(node):
    graph = nx.DiGraph()
    graph.add_node(
        str(hash(node)),
        label=f"{np.array2string(node.env.state, separator=',')}\nN={node.visit_count}, Q={node.mean_action_value:.2f}, P={node.prior_prob}",
    )

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
        return f"Node(N={self.visit_count}, Q={self.mean_action_value}, P={self.prior_prob}, Children={list(self.children.keys())})\n{self.env}\n"


def select(node, cpuct):
    total_visit_count = node.visit_count
    return max(node.children.values(), key=lambda x: x.uct(cpuct, total_visit_count))


def expand_and_evaluate(node, neural_network):
    if node.env.is_terminal():
        return node.env.value
    with torch.no_grad():
        probabilities, state_value = neural_network(node.env.get_tensor_state())
        probabilities = probabilities.detach().numpy().squeeze()
        state_value = state_value.detach().numpy().squeeze()

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
    width = 3
    height = 3
    n_colors = 3
    nn_blocks = 3

    np.random.seed(2)
    env = FloodEnv(width, height, n_colors)

    class FakeNet:
        def __init__(self, n_colors):
            self.n_colors = n_colors

        def __call__(self, state):
            return torch.ones(n_colors) / n_colors, torch.zeros(1)

    net = FakeNet(n_colors)
    c_puct = 1
    temperature = 1
    print(env)
    root, probs = alphago_zero_search(env, net, 100, 1, 1)
    draw_tree(root)
