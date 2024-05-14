import torch
import numpy as np
from MPSPEnv import Env
from Node import Node


class MCGS:
    """Monte Carlo Graph Search algorithm. The algorithm is used to estimate the utility of a given state by simulating n times"""

    def __init__(
        self,
        model: torch.nn.Module,
        c_puct: float = 1,
        dirichlet_weight: float = 0.03,
        dirichlet_alpha: float = 0.25,
    ):
        self.model = model
        self.nodes_by_hash: dict[Env, Node] = {}
        self.c_puct = c_puct
        self.dirichlet_weight = dirichlet_weight
        self.dirichlet_alpha = dirichlet_alpha

    def run(
        self,
        root: Node,
        add_exploration_noise: bool = True,
        search_iterations: int = 100,
    ) -> int:
        """Run the Monte Carlo Graph Search algorithm from the root node for a given number of iterations."""

        if add_exploration_noise:
            self._add_noise(root)

        self._find_best_depths(root)

        for _ in range(search_iterations):
            search_path = self._find_leaf(root)
            node = search_path[-1]
            self._evaluate(node)
            self._backpropagate(search_path)

        return root

    def _find_best_depths(self, root: Node) -> None:
        """Do BFS to find the best depth for each node"""
        queue = [(root, 0)]
        nodes = set([root])
        reverse_queue = [root]
        while queue:
            node, depth = queue.pop(0)
            node.best_depth = min(node.best_depth, depth)
            for child, _ in node._children_and_edge_visits.values():
                if child in nodes:
                    continue
                nodes.add(child)
                queue.append((child, depth + 1))
                reverse_queue.append(child)
        while reverse_queue:
            node = reverse_queue.pop()
            self._update_Q_and_N(node)

    def _add_noise(self, node: Node) -> None:
        """Add noise to the policy of a node."""
        if node.P is None:  # Node hasn't been evaluated yet. Don't add noise
            return

        n = len(node.P)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)

        node.P = (1 - self.dirichlet_weight) * node.P + self.dirichlet_weight * noise

    def _backpropagate(self, search_path: list[Node]) -> None:
        """Backpropagate the utility of the leaf node up the search path."""

        for i, node in enumerate(reversed(search_path)):
            self._update_Q_and_N(node)
            depth = len(search_path) - i - 1
            node.best_depth = min(depth, node.best_depth)

    def _update_Q_and_N(self, node: Node) -> None:
        """Update the average utility Q and number of visits N of a node."""
        if node.U is None:  # Node hasn't been evaluated yet
            return

        children_and_edge_visits = node.children_and_edge_visits.values()
        node.N = 1 + sum(edge_visits for (_, edge_visits) in children_and_edge_visits)
        node.Q = (1 / node.N) * (
            node.U
            + sum(
                (child.Q - 1) * edge_visits  # -1 = account for minimize moves
                for (child, edge_visits) in children_and_edge_visits
            )
        )

    def _find_leaf(self, node: Node) -> list[Node]:
        search_path = [node]
        while node.U and not node.game_state.terminal:  # Has been evaluated
            action = self._select_action(node)

            if action in node.children_and_edge_visits:
                child, _ = node.children_and_edge_visits[action]
            else:

                state = node.game_state.copy()
                state.step(action)
                if state in self.nodes_by_hash:
                    child = self.nodes_by_hash[state]
                else:
                    child = Node(state)
                    self.nodes_by_hash[state] = child

                node.add_child(action, child)

            node.increment_visits(action)
            node = child
            search_path.append(node)

        return search_path

    def _select_action(self, node: Node) -> int:
        """Select an action based on the revised PUCT formula."""
        assert (
            node.P is not None and node.U is not None
        ), "Node has not been evaluated yet."

        n = len(node.P)
        Q = np.zeros(n)
        N = np.zeros(n)
        for a, (child, edge_visits) in node.children_and_edge_visits.items():
            Q[a] = child.Q
            N[a] = edge_visits

        U = self.c_puct * node.P * np.sqrt(node.N) / (1 + N)

        Q_plus_U = Q + U
        Q_plus_U[Q_plus_U == 0] = -np.inf  # Q values are negative

        action = np.argmax(Q_plus_U)
        return action

    def _evaluate(self, node: Node) -> None:
        """Calculate the utility of an unexplored node."""
        if node.game_state.terminal:
            node.U = -node.game_state.moves_to_solve
        else:
            policy, value = self._run_model(node.game_state)
            node.P = policy * node.game_state.mask
            node.U = value

    def _run_model(self, env: Env) -> tuple[np.ndarray, float]:
        """Estimate the utility and policy of a given state based on the neural network."""
        bay, flat_T = env.bay, env.flat_T

        bay = torch.tensor(bay)
        flat_T = torch.tensor(flat_T)
        bay = bay.unsqueeze(0).unsqueeze(0)
        flat_T = flat_T.unsqueeze(0)

        with torch.no_grad():
            policy, state_value = self.model(bay, flat_T)
            policy = policy.detach().cpu().numpy().squeeze()
            state_value = state_value.item()

        return policy, state_value
