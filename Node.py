import numpy as np


class Node:
    def __init__(self, state, prior_prob):
        self.state = state
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior_prob = prior_prob
        self.children = {}

    def uct(self, cpuct, total_visit_count):
        return self.mean_action_value + cpuct * self.prior_prob * np.sqrt(
            total_visit_count
        ) / (1 + self.visit_count)


def select(node, cpuct):
    total_visit_count = sum(child.visit_count for child in node.children.values())
    return max(node.children.items(), key=lambda x: x[1].uct(cpuct, total_visit_count))


def expand_and_evaluate(node, neural_network):
    actions, probabilities, state_value = neural_network.predict(node.state)
    for a, p in zip(actions, probabilities):
        node.children[a] = Node(a, p)
    return state_value


def backup(node, value):
    node.visit_count += 1
    node.total_action_value += value
    node.mean_action_value = node.total_action_value / node.visit_count
    if node.state.is_terminal():
        return
    for child in node.children.values():
        child.mean_action_value = backup(child, child.state, -value)


def play(node, temperature):
    action_probs = [
        np.power(child.visit_count, 1 / temperature) for child in node.children.values()
    ]
    action_probs /= np.sum(action_probs)
    action_index = np.random.choice(len(node.children), p=action_probs)
    return list(node.children.keys())[action_index]


def alphago_zero_search(
    root_state, neural_network, num_simulations, cpuct, temperature
):
    root_node = Node(root_state, 1)

    for i in range(num_simulations):
        node = root_node
        while node.children:
            action, node = select(node, cpuct)
        state_value = expand_and_evaluate(node, neural_network)
        backup(node, state_value)

    return play(root_node, temperature)
