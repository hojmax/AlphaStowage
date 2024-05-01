import numpy as np
import torch
from MPSPEnv import Env
from multiprocessing.connection import Connection
from Node import Node


def get_np_bay(env: Env, config: dict) -> np.ndarray:
    bay = env.bay
    bay = bay.astype(np.float32)
    bay = bay / env.remaining_ports
    bay = np.pad(
        bay,
        ((0, config["env"]["R"] - env.R), (0, config["env"]["C"] - env.C)),
        mode="constant",
        constant_values=-1,
    )
    return bay


def get_np_flat_T(env: Env, config: dict) -> np.ndarray:
    T = env.T
    T = T.astype(np.float32)
    T = np.pad(
        T,
        ((0, config["env"]["N"] - env.N), (0, config["env"]["N"] - env.N)),
        mode="constant",
        constant_values=0,
    )
    i, j = np.triu_indices(n=T.shape[0], k=1)
    flat_T = T[i, j]
    flat_T = flat_T / (env.R * env.C)
    return flat_T


def get_np_obs(env: Env, config: dict) -> tuple[np.ndarray, np.ndarray]:
    bay = get_np_bay(env, config)
    flat_T = get_np_flat_T(env, config)
    return bay, flat_T


def run_network(
    node: Node, conn: Connection, config: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    conn.send(get_np_obs(node.env, config))
    probabilities, value = conn.recv()
    return torch.tensor(probabilities), torch.tensor(value)


def get_prob_and_value(
    node: Node,
    conn: Connection,
    transposition_table: dict[Env, tuple[np.ndarray, np.ndarray]],
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if node.env in transposition_table:
        probabilities, state_value = transposition_table[node.env]
    else:
        probabilities, state_value = run_network(node, conn, config)
        transposition_table[node.env] = (probabilities, state_value)

    return probabilities, state_value - node.depth  # Counting the already made moves


def add_dirichlet_noise(
    probabilities: np.ndarray, dirichlet_weight: float, dirichlet_alpha: float
) -> np.ndarray:
    """Add dirichlet noise to prior probabilities for more exploration"""
    noise = np.random.dirichlet(np.zeros_like(probabilities) + dirichlet_alpha)
    probabilities = (1 - dirichlet_weight) * probabilities + dirichlet_weight * noise
    return probabilities


def is_root(node: Node) -> bool:
    return node.parent == None


def expand_node(
    node: Node,
    conn: Connection,
    transposition_table: dict[Env, tuple[np.ndarray, np.ndarray]],
    config: dict,
) -> np.ndarray:

    probabilities, state_value = get_prob_and_value(
        node, conn, transposition_table, config
    )

    if is_root(node):
        probabilities = add_dirichlet_noise(
            probabilities,
            config["mcts"]["dirichlet_weight"],
            config["mcts"]["dirichlet_alpha"],
        )

    add_children(probabilities, state_value, node, config)

    return state_value


def close_envs_in_tree(node: Node) -> None:
    node.close()

    for child in node.children.values():
        close_envs_in_tree(child)


def evaluate(
    node: Node,
    conn: Connection,
    transposition_table: dict[Env, tuple[np.ndarray, np.ndarray]],
    config: dict,
) -> float:
    if node.env.terminal:
        return -node.env.moves_to_solve
    else:
        state_value = expand_node(
            node,
            conn,
            transposition_table,
            config,
        )
        return state_value


def add_children(
    probabilities: np.ndarray, state_value: float, node: Node, config: dict
) -> None:
    possible_actions = (
        range(node.env.C)
        if config["inference"]["can_only_add"]
        else range(2 * node.env.C)
    )
    for action in possible_actions:
        is_legal = node.env.action_masks()[action]
        if not is_legal:
            continue
        prior = (
            probabilities[action]
            if action < node.env.C
            else probabilities[action + config["env"]["C"] - node.env.C]
        )
        node.add_child(action, node.env.copy(), prior, state_value, config)


def backup(node: Node, value: float) -> None:
    node.increment_value(value)

    if not is_root(node):
        backup(node.parent, value)


def remove_all_pruning(node: Node) -> None:
    node.unprune()

    for child in node.children.values():
        remove_all_pruning(child)


def get_tree_probs(node: Node, config: dict) -> torch.Tensor:
    action_probs = torch.zeros(2 * config["env"]["C"], dtype=torch.float64)

    for i in node.children:
        value = np.power(
            node.children[i].visit_count, 1 / config["mcts"]["temperature"]
        )
        index = i if i < node.env.C else i + config["env"]["C"] - node.env.C
        action_probs[index] = value

    return action_probs / torch.sum(action_probs)


def prune_and_move_back_up(node: Node) -> Node:
    node.prune()

    return node.parent


def should_prune(node: Node, best_score: float, moves_upper_bound: int) -> bool:
    moves_upper_bound = node.env.N * node.env.C * node.env.R
    return (
        node.no_valid_children
        or -node.env.moves_to_solve < best_score
        or -node.env.moves_to_solve <= -moves_upper_bound
    )


def find_leaf(root_node: Node, best_score: float, moves_upper_bound: int) -> Node:
    node = root_node

    while node.children:
        if should_prune(node, best_score, moves_upper_bound):
            node = prune_and_move_back_up(node)
        else:
            node = node.select_child()

    return node


def alpha_zero_search(
    root_env: Env,
    conn: Connection,
    config: dict,
    reused_tree: Node = None,
    transposition_table: dict[Env, tuple[np.ndarray, np.ndarray]] = {},
) -> tuple[torch.Tensor, Node, dict[Env, tuple[np.ndarray, np.ndarray]]]:
    root_node = reused_tree if reused_tree else Node(root_env.copy(), config)
    moves_upper_bound = root_env.N * root_env.C * root_env.R
    best_score = float("-inf")
    for _ in range(config["mcts"]["search_iterations"]):
        node = find_leaf(root_node, best_score, moves_upper_bound)

        state_value = evaluate(
            node,
            conn,
            transposition_table,
            config,
        )

        backup(node, state_value)

        if node.env.terminal:
            best_score = max(best_score, state_value)

    return (
        get_tree_probs(root_node, config),
        root_node,
        transposition_table,
    )