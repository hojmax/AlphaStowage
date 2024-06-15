import numpy as np
import torch
from MPSPEnv import Env
from multiprocessing.connection import Connection
from Node import Node
from min_max import MinMaxStats


def run_network(node: Node, conn: Connection) -> tuple[np.ndarray, np.ndarray]:
    conn.send(
        (
            node.env.bay,
            node.env.flat_T,
            np.array([node.env.containers_left], dtype=np.float32),
            node.env.mask,
        )
    )
    probabilities, value = conn.recv()
    return probabilities, value


def get_prob_and_value(
    node: Node,
    conn: Connection,
    transposition_table: dict[Env, tuple[np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, float]:
    if node.env in transposition_table:
        probabilities, state_value = transposition_table[node.env]
    else:
        probabilities, state_value = run_network(node, conn)
        transposition_table[node.env] = (probabilities, state_value)

    return (probabilities, state_value.item() - node.env.containers_placed)


def is_root(node: Node) -> bool:
    return node.parent == None


def expand_node(
    node: Node,
    conn: Connection,
    transposition_table: dict[Env, tuple[np.ndarray, np.ndarray]],
    config: dict,
) -> float:

    probabilities, state_value = get_prob_and_value(node, conn, transposition_table)
    add_children(probabilities, node, config)

    if is_root(node):
        node.add_noise()

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
    if node.env.terminated:
        return -node.env.containers_placed
    else:
        state_value = expand_node(
            node,
            conn,
            transposition_table,
            config,
        )
        return state_value


def add_children(probabilities: np.ndarray, node: Node, config: dict) -> None:
    for action in range(2 * node.env.R * node.env.C):
        if not node.env.mask[action]:
            continue

        node.add_child(
            action=action,
            new_env=node.env.copy(),
            prior=probabilities[action],
            config=config,
        )


def backup(node: Node, value: float, min_reward: int | None) -> None:
    node.increment_value(value)

    if not is_root(node):
        # If an optimal path is found, the visit count of the parent node is set to the maximum possible value
        if min_reward and node.parent.env.total_reward == min_reward:
            node.visit_count = np.iinfo(np.int32).max
            node.total_action_value = (
                -(node.env.containers_placed + node.env.containers_left)
                * node.visit_count
            )  # Done to have the right value for the Q value

        backup(node.parent, value, min_reward)


def get_tree_probs(node: Node, config: dict) -> torch.Tensor:
    action_probs = torch.zeros(
        2 * config["env"]["R"] * config["env"]["C"], dtype=torch.float64
    )

    for action in node.children.keys():
        action_probs[action] = np.power(
            node.children[action].visit_count, 1 / config["mcts"]["temperature"]
        )

    return action_probs / torch.sum(action_probs)


def is_leaf_node(node: Node) -> bool:
    return len(node.children) == 0


def find_leaf(root_node: Node, min_max_stats: MinMaxStats) -> tuple[Node, int | None]:
    node = root_node

    min_reward = node.env.total_reward

    while not is_leaf_node(node):
        node = node.select_child(min_max_stats)
        min_reward = min(min_reward, node.env.total_reward)

    return node, min_reward if node.env.terminated else None


def get_new_root_node(root_env: Env, reused_tree: Node, config: dict) -> Node:
    if reused_tree is not None:
        if len(reused_tree.children) > 0:
            reused_tree.add_noise()

        return reused_tree
    else:
        return Node(root_env.copy(), config)


def alpha_zero_search(
    root_env: Env,
    conn: Connection,
    config: dict,
    min_max_stats: MinMaxStats,
    reused_tree: Node = None,
    transposition_table: dict[Env, tuple[np.ndarray, np.ndarray]] = {},
) -> tuple[torch.Tensor, Node, dict[Env, tuple[np.ndarray, np.ndarray]]]:
    root_node = get_new_root_node(root_env, reused_tree, config)

    found_optimal_path = False

    for _ in range(config["mcts"]["search_iterations"]):
        node, min_reward = find_leaf(root_node, min_max_stats)

        state_value = evaluate(
            node,
            conn,
            transposition_table,
            config,
        )

        min_max_stats.update(state_value)

        backup(node, state_value, min_reward)

        is_optimal_path = (
            node.env.terminated and node.env.total_reward == root_node.env.total_reward
        )  # The optimal path if the path that has the same reward as the root node
        if is_optimal_path:
            found_optimal_path = True
            break

    return (
        get_tree_probs(root_node, config),
        root_node,
        transposition_table,
        found_optimal_path,
    )
