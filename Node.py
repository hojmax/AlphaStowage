import numpy as np


class Node:
    def __init__(self, state, prior_prob, parent=None):
        self.state = state
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


def select(node, cpuct):
    total_visit_count = sum(child.visit_count for child in node.children.values())
    return max(node.children.values(), key=lambda x: x.uct(cpuct, total_visit_count))


def expand_and_evaluate(node, neural_network):
    if node.env.is_terminal():
        return node.env.value

    probabilities, state_value = neural_network.predict(node.env.state)

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
            node = select(node, cpuct)
        state_value = expand_and_evaluate(node, neural_network)
        backup(node, state_value)

    return play(root_node, temperature)
