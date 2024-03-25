import numpy as np
import torch


class TruncatedEpisodeError(Exception):
    pass


class Node:
    def __init__(
        self, env, prior_prob=None, estimated_value=0, parent=None, depth=0, cpuct=1
    ):
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
        self.cpuct = cpuct

    @property
    def Q(self):
        return (
            self.mean_action_value
            if self.mean_action_value is not None
            else self.estimated_value
        )

    @property
    def U(self):
        return (
            self.cpuct
            * self.prior_prob
            * np.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )

    def increment_value(self, value):
        self.total_action_value += value
        self.visit_count += 1
        self.mean_action_value = self.total_action_value / self.visit_count

    @property
    def uct(self):
        return self.Q + self.U

    def __str__(self):
        output = f"{self.env.bay}\n{self.env.T}\nN={self.visit_count}, Q={self.Q:.2f}, Moves={self.env.moves_to_solve}"
        if self.prior_prob is not None:
            output += f" P={self.prior_prob:.2f}, Q+U={self.uct:.2f}"
        if self.pruned:
            output = "pruned\n" + output
        return output


def select(node, cpuct):
    valid_children = [child for child in node.children.values() if not child.pruned]

    return max(valid_children, key=lambda x: x.uct)


def get_torch_obs(env):
    bay = torch.from_numpy(env.one_hot_bay).unsqueeze(0).float()
    summed_over_ports = bay.sum(dim=1, keepdim=True)
    bay = summed_over_ports / bay.size(1)
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
        probabilities, state_value = run_network(node, neural_network, device)
        transposition_table[node.env] = (probabilities, state_value)

    return probabilities, state_value - node.depth  # Counting the already made moves


def add_dirichlet_noise(probabilities, dirichlet_weight, dirichlet_alpha):
    """Add dirichlet noise to prior probs for more exploration"""
    noise = np.random.dirichlet(np.zeros_like(probabilities) + dirichlet_alpha)
    probabilities = (1 - dirichlet_weight) * probabilities + dirichlet_weight * noise
    return probabilities


def is_root(node):
    return node.parent == None


def expand_node(
    node,
    neural_network,
    dirichlet_weight,
    dirichlet_alpha,
    device,
    transposition_table,
    cpuct,
):
    probabilities, state_value = get_prob_and_value(
        node, neural_network, device, transposition_table
    )

    if is_root(node):
        probabilities = add_dirichlet_noise(
            probabilities, dirichlet_weight, dirichlet_alpha
        )

    add_children(probabilities, state_value, node, cpuct)

    return state_value


def close_envs_in_tree(node):
    if node.env:
        node.env.close()
    for child in node.children.values():
        close_envs_in_tree(child)


def evaluate(
    node,
    neural_network,
    dirichlet_weight,
    dirichlet_alpha,
    device,
    transposition_table,
    cpuct,
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
            cpuct,
        )
        return state_value


def add_children(probabilities, state_value, node, cpuct):
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
            cpuct=cpuct,
        )


def backup(node, value):
    node.increment_value(value)

    if not is_root(node):
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

    if is_root(node):
        raise TruncatedEpisodeError

    return node.parent


def should_prune(node, best_score):
    n_valid_children = len(
        [child for child in node.children.values() if not child.pruned]
    )
    moves_upper_bound = node.env.N * node.env.C * node.env.R
    return (
        n_valid_children == 0
        or -node.env.moves_to_solve < best_score
        or -node.env.moves_to_solve <= -moves_upper_bound
    )


def find_leaf(root_node, cpuct, best_score):
    node = root_node

    while node.children:
        if should_prune(node, best_score):
            node = prune_and_move_back_up(node)
        else:
            node = select(node, cpuct)

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
    root_node = reused_tree if reused_tree else Node(root_env.copy())
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
            cpuct,
        )

        backup(node, state_value)

        if node.env.terminal:
            best_score = max(best_score, state_value)

    return root_node, get_tree_probs(root_node, temperature), transposition_table
