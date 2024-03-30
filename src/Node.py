import numpy as np
import torch
from MPSPEnv import Env


class TruncatedEpisodeError(Exception):
    pass


class Node:
    def __init__(
        self, env, c_puct, prior_prob=None, estimated_value=0, parent=None, depth=0
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
        self.c_puct = c_puct

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
            self.c_puct
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

    @property
    def valid_children(self):
        return [child for child in self.children.values() if not child.pruned]

    def __str__(self):
        output = f"{self.env.bay}\n{self.env.T}\nN={self.visit_count}, Q={self.Q:.2f}, Moves={self.env.moves_to_solve}"
        if self.prior_prob is not None:
            output += f" P={self.prior_prob:.2f}, Q+U={self.uct:.2f}"
        if self.pruned:
            output = "pruned\n" + output
        return output

    def select_child(self):
        return max(self.valid_children, key=lambda x: x.uct)


def get_torch_bay(env, config):
    bay = env.bay
    bay = bay / env.remaining_ports
    bay = np.pad(
        bay,
        ((0, config["env"]["R"] - env.R), (0, config["env"]["C"] - env.C)),
        mode="constant",
        constant_values=-1,
    )
    bay = torch.from_numpy(bay).unsqueeze(0).unsqueeze(0).float()
    return bay


def get_torch_flat_T(env, config):
    T = env.T
    T = np.pad(
        T,
        ((0, config["env"]["N"] - env.N), (0, config["env"]["N"] - env.N)),
        mode="constant",
        constant_values=0,
    )
    i, j = np.triu_indices(n=T.shape[0], k=1)
    flat_T = T[i, j]
    flat_T = flat_T / (env.R * env.C)
    flat_T = torch.from_numpy(flat_T).unsqueeze(0).float()
    return flat_T


def get_torch_obs(env: Env, config):
    bay = get_torch_bay(env, config)
    flat_T = get_torch_flat_T(env, config)
    return bay, flat_T


def run_network(node, neural_network, device, config):
    with torch.no_grad():
        bay, flat_t = get_torch_obs(node.env, config)
        probabilities, state_value = neural_network(bay.to(device), flat_t.to(device))
        probabilities = probabilities.detach().cpu().numpy().squeeze()
        state_value = state_value.detach().cpu().numpy().squeeze()
    return probabilities, state_value


def get_prob_and_value(node, neural_network, device, transposition_table, config):
    if node.env in transposition_table:
        probabilities, state_value = transposition_table[node.env]
    else:
        probabilities, state_value = run_network(node, neural_network, device, config)
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
    device,
    transposition_table,
    config,
):
    probabilities, state_value = get_prob_and_value(
        node, neural_network, device, transposition_table, config
    )

    if is_root(node):
        probabilities = add_dirichlet_noise(
            probabilities,
            config["mcts"]["dirichlet_weight"],
            config["mcts"]["dirichlet_alpha"],
        )

    add_children(probabilities, state_value, node, config)

    return state_value


def close_envs_in_tree(node):
    if node.env:
        node.env.close()
    for child in node.children.values():
        close_envs_in_tree(child)


def evaluate(
    node,
    neural_network,
    device,
    transposition_table,
    config,
):
    if node.env.terminal:
        return -node.env.moves_to_solve
    else:
        state_value = expand_node(
            node,
            neural_network,
            device,
            transposition_table,
            config,
        )
        return state_value


def add_children(probabilities, state_value, node, config):
    possible_actions = (
        range(node.env.C) if config["train"]["can_only_add"] else range(2 * node.env.C)
    )
    for action in possible_actions:
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
            c_puct=config["mcts"]["c_puct"],
        )


def backup(node, value):
    node.increment_value(value)

    if not is_root(node):
        backup(node.parent, value)


def remove_all_pruning(node):
    node.pruned = False
    for child in node.children.values():
        remove_all_pruning(child)


def get_tree_probs(node, config):
    action_probs = torch.zeros(2 * config["env"]["C"], dtype=torch.float64)

    for i in node.children:
        value = np.power(
            node.children[i].visit_count, 1 / config["mcts"]["temperature"]
        )
        index = i if i < node.env.C else i + config["env"]["C"] - node.env.C
        action_probs[index] = value

    return action_probs / torch.sum(action_probs)


def prune_and_move_back_up(node):
    node.pruned = True

    if is_root(node):
        raise TruncatedEpisodeError

    return node.parent


def should_prune(node: Node, best_score: float) -> bool:
    moves_upper_bound = node.env.N * node.env.C * node.env.R
    return (
        len(node.valid_children) == 0
        or -node.env.moves_to_solve < best_score
        or -node.env.moves_to_solve <= -moves_upper_bound
    )


def find_leaf(root_node, best_score):
    node = root_node

    while node.children:
        if should_prune(node, best_score):
            node = prune_and_move_back_up(node)
        else:
            node = node.select_child()

    return node


def alpha_zero_search(
    root_env,
    neural_network,
    device,
    config,
    reused_tree=None,
    transposition_table={},
):
    root_node = (
        reused_tree if reused_tree else Node(root_env.copy(), config["mcts"]["c_puct"])
    )
    best_score = float("-inf")

    for _ in range(config["mcts"]["search_iterations"]):
        node = find_leaf(root_node, best_score)

        state_value = evaluate(
            node,
            neural_network,
            device,
            transposition_table,
            config,
        )

        backup(node, state_value)

        if node.env.terminal:
            best_score = max(best_score, state_value)

    return (
        root_node,
        get_tree_probs(root_node, config),
        transposition_table,
    )
