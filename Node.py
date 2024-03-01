import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from MPSPEnv import Env
import json
from NeuralNetwork import NeuralNetwork


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

    plt.gcf().set_size_inches(12, 7)
    plt.show()


class Node:
    def __init__(self, env, prior_prob=None, estimated_value=0, parent=None, depth=0):
        self.env = env
        self.pruned = False
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = estimated_value
        self.prior_prob = prior_prob
        self.children = {}
        self.parent = parent
        self.depth = depth

    def uct(self, cpuct):
        return self.mean_action_value + cpuct * self.prior_prob * np.sqrt(
            self.parent.visit_count
        ) / (1 + self.visit_count)

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


def expand_and_evaluate(
    node, neural_network, dirichlet_weight, dirichlet_alpha, device, transposition_table
):
    if node.env.terminal:
        return -(node.env.containers_placed + node.env.containers_left)

    if node.env in transposition_table:
        probabilities, state_value = transposition_table[node.env]
    else:
        with torch.no_grad():
            bay, flat_t, mask = get_torch_obs(node.env)
            probabilities, state_value = neural_network(
                bay.to(device), flat_t.to(device), mask.to(device)
            )
            probabilities = probabilities.detach().cpu().numpy().squeeze()
            state_value = state_value.detach().cpu().numpy().squeeze() - node.depth
            transposition_table[node.env] = (probabilities, state_value)

    is_root_node = node.parent == None
    if is_root_node:
        # Add dirichlet noise to prior probs for more exploration
        noise = np.random.dirichlet(np.zeros_like(probabilities) + dirichlet_alpha)
        probabilities = (
            1 - dirichlet_weight
        ) * probabilities + dirichlet_weight * noise

    for i in range(2 * node.env.C):
        if not node.env.action_masks()[i]:
            continue
        action = i
        prob = probabilities[action]
        new_env = node.env.copy()
        next_depth = node.depth + 1
        new_env.step(action)
        node.children[action] = Node(
            env=new_env,
            prior_prob=prob,
            estimated_value=state_value,
            parent=node,
            depth=next_depth,
        )

    return state_value


def backup(node, value):
    node.visit_count += 1
    node.total_action_value += value
    node.mean_action_value = node.total_action_value / node.visit_count

    if node.parent is not None:
        backup(node.parent, value)


def remove_pruning(node):
    node.pruned = False
    for child in node.children.values():
        remove_pruning(child)


def get_tree_probs(node, temperature):
    action_probs = []
    for i in range(2 * node.env.C):
        if i in node.children:
            action_probs.append(np.power(node.children[i].visit_count, 1 / temperature))
        else:
            action_probs.append(0)
    action_probs = torch.tensor(action_probs)
    action_probs /= action_probs.sum()
    return action_probs


def reset_action_value(node):
    node.mean_action_value = 0


def backtrack(node):
    node.pruned = True
    return node.parent


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
    if reused_tree == None:
        root_node = Node(root_env)
    else:
        root_node = reused_tree

    best_depth = float("inf")

    for i in range(num_simulations):
        node = root_node
        depth = 0

        while node.children:
            try:
                node = select(node, cpuct)
                depth += 1
            except ValueError:
                # In case all children are pruned
                node = backtrack(node)
                depth -= 1

            if not node.env.terminal and depth >= best_depth:
                node = backtrack(node)
                depth -= 1

        reset_action_value(node)
        state_value = expand_and_evaluate(
            node,
            neural_network,
            dirichlet_weight,
            dirichlet_alpha,
            device,
            transposition_table,
        )

        if node.env.terminal:
            best_depth = min(best_depth, abs(state_value))

        backup(node, state_value)

    return root_node, get_tree_probs(root_node, temperature), transposition_table


if __name__ == "__main__":
    # run_path = "hojmax/multi-thread/9h0s1ig7"
    # model_path = "model10800.pt"
    # api = wandb.Api()
    # run = api.run(run_path)
    # file = run.file(model_path)
    # file.download(replace=True)
    # config = run.config

    # net = NeuralNetwork(config=config)
    # net.load_state_dict(torch.load(model_path, map_location="cpu"))
    # net.eval()

    # load config.json
    with open("config.json") as f:
        config = json.load(f)

    # class FakeNet:
    #     def __init__(self):
    #         pass

    #     def __call__(self, x, y, z):
    #         return torch.ones(1, 2 * config["env"]["C"]) / (
    #             2 * config["env"]["C"]
    #         ), -torch.ones(1, 1)

    # net = FakeNet()
    net = NeuralNetwork(config)

    env = Env(
        config["env"]["R"],
        config["env"]["C"],
        config["env"]["N"],
        skip_last_port=True,
        take_first_action=True,
        strict_mask=True,
    )
    env.reset()

    for i in range(1, 100):
        np.random.seed(11)
        root, probs, transposition_table = alpha_zero_search(
            env,
            net,
            i,
            config["mcts"]["c_puct"],
            config["mcts"]["temperature"],
            config["mcts"]["dirichlet_weight"],
            config["mcts"]["dirichlet_alpha"],
            device="cpu",
        )
        print(probs)
        draw_tree(root)
