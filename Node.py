import numpy as np
import torch


class Node:
    def __init__(self, env, prior_prob=None, estimated_value=0, parent=None, depth=0):
        self.env = env
        self.pruned = False
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = None
        self.estimated_value = estimated_value
        self.prior_prob = prior_prob
        self.children = {}
        self.parent = parent
        self.depth = depth

    def uct(self, cpuct):
        exploration_term = (
            cpuct
            * self.prior_prob
            * np.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        value_term = (
            self.mean_action_value
            if self.mean_action_value is not None
            else self.estimated_value
        )
        return value_term + exploration_term

    def __str__(self):
        if self.prior_prob is None:
            return f"{self.env.bay}\n{self.env.T}\nN={self.visit_count}, Q={self.mean_action_value:.2f}"
        output = f"{self.env.bay}\n{self.env.T}\nN={self.visit_count}, Q={self.mean_action_value:.2f}, P={self.prior_prob:.2f}, Q+U={self.uct(1):.2f}"
        if self.pruned:
            output = "(pruned) " + output
        return output


def select(node, cpuct):
    valid_children = [child for child in node.children.values() if not child.pruned]
    return max(valid_children, key=lambda x: x.uct(cpuct))


def get_torch_obs(env):
    bay = torch.from_numpy(env.one_hot_bay).unsqueeze(0).float()
    flat_t = torch.from_numpy(env.flat_T).unsqueeze(0).float()
    mask = torch.from_numpy(env.action_masks()).unsqueeze(0).float()
    return bay, flat_t, mask


def run_network(node, neural_network, device):
    with torch.no_grad():
        bay, flat_t, mask = get_torch_obs(node.env)
        probabilities, state_value = neural_network(
            bay.to(device), flat_t.to(device), mask.to(device)
        )
        probabilities = probabilities.detach().cpu().numpy().squeeze()
        state_value = state_value.detach().cpu().numpy().squeeze()
    return probabilities, state_value


def get_prob_and_value(node, neural_network, device, transposition_table):
    if node.env in transposition_table:
        probabilities, state_value = transposition_table[node.env]
    else:
        probabilities, state_value = run_network(node.env, neural_network, device)
        transposition_table[node.env] = (probabilities, state_value)

    return probabilities, state_value - node.depth  # Counting the already made moves


def add_dirichlet_noise(probabilities, dirichlet_weight, dirichlet_alpha):
    """Add dirichlet noise to prior probs for more exploration"""
    noise = np.random.dirichlet(np.zeros_like(probabilities) + dirichlet_alpha)
    probabilities = (1 - dirichlet_weight) * probabilities + dirichlet_weight * noise
    return probabilities


def is_root_node(node):
    return node.parent == None


def expand_node(
    node, neural_network, dirichlet_weight, dirichlet_alpha, device, transposition_table
):
    probabilities, state_value = get_prob_and_value(
        node.env, neural_network, device, transposition_table
    )

    if is_root_node(node):
        probabilities = add_dirichlet_noise(
            probabilities, dirichlet_weight, dirichlet_alpha
        )

    add_children(probabilities, state_value, node)

    return state_value


def evaluate(
    node, neural_network, dirichlet_weight, dirichlet_alpha, device, transposition_table
):
    if node.env.terminal:
        return -node.env.moves_to_solve
    else:
        state_value = expand_node(
            node,
            neural_network,
            dirichlet_weight,
            dirichlet_alpha,
            device,
            transposition_table,
        )
        return state_value


def add_children(probabilities, state_value, node):
    for action in range(2 * node.env.C):
        is_legal = node.env.action_masks()[action]
        if not is_legal:
            continue
        new_env = node.env.copy()
        new_env.step(action)
        node.children[action] = Node(
            env=new_env,
            prior_prob=probabilities[action],
            estimated_value=state_value,
            parent=node,
            depth=node.depth + 1,
        )


def backup(node, value):
    node.visit_count += 1
    node.total_action_value += value
    node.mean_action_value = node.total_action_value / node.visit_count

    if not is_root_node(node):
        backup(node.parent, value)


def remove_all_pruning(node):
    node.pruned = False
    for child in node.children.values():
        remove_all_pruning(child)


def get_tree_probs(node, temperature):
    action_probs = [
        (
            np.power(node.children[i].visit_count, 1 / temperature)
            if i in node.children
            else 0
        )
        for i in range(2 * node.env.C)
    ]
    return torch.tensor(action_probs) / sum(action_probs)


def prune_and_move_back_up(node):
    node.pruned = True
    return node.parent


def find_leaf(root_node, cpuct, best_score):
    node = root_node

    while node.children:
        try:
            node = select(node, cpuct)
        except ValueError:
            # In case all children are pruned
            node = prune_and_move_back_up(node)

        if not node.env.terminal and node.env.moves_to_solve <= best_score:
            node = prune_and_move_back_up(node)

    return node


def alpha_zero_search(
    root_env,
    neural_network,
    num_simulations,
    cpuct,
    temperature,
    dirichlet_weight,
    dirichlet_alpha,
    device,
    reused_tree=None,
    transposition_table={},
):
    root_node = reused_tree if reused_tree else Node(root_env)
    best_score = float("-inf")

    for _ in range(num_simulations):
        node = find_leaf(root_node, cpuct, best_score)

        state_value = evaluate(
            node,
            neural_network,
            dirichlet_weight,
            dirichlet_alpha,
            device,
            transposition_table,
        )

        if node.env.terminal:
            best_score = max(best_score, state_value)

        backup(node, state_value)

    return root_node, get_tree_probs(root_node, temperature), transposition_table
