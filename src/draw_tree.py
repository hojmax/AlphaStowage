from Node import Node
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import plotly.graph_objects as go

def _draw_tree_recursive(
    graph: nx.DiGraph,
    node: Node,
    nodes: set[Node],
    node_colors: dict,
    node_data: dict,
    edge_data: dict,
):
    for action, (
        child,
        edge_visits,
    ) in (
        node.children_and_edge_visits.items()
    ):  # Assuming edge_value is the value to check
        color = "green" if edge_visits < node.game_state.C else "red"
        graph.add_edge(
            str(hash(node)), str(hash(child)), label=str(action), color=color
        )
        edge_data[(str(hash(node)), str(hash(child)))] = {
            "action": action,
            "edge_visits": edge_visits,
        }
        if child in nodes or child is None:
            continue

        graph.add_node(str(hash(child)))

        nodes.add(child)
        node_colors[str(hash(child))] = child.Q  # Store the Q value
        node_data[str(hash(child))] = {
            "U": child.U,
            "Q": child.Q,
            "N": child.N,
            "best_depth": child.best_depth,
        }  # Store additional data
        _draw_tree_recursive(graph, child, nodes, node_colors, node_data, edge_data)


def draw_tree(node: Node):
    nodes = set([node])
    graph = nx.MultiDiGraph()
    graph.add_node(str(hash(node)))

    node_colors = {str(hash(node)): node.Q}
    node_data = {str(hash(node)): {"U": node.U, "Q": node.Q}}
    edge_data = {}

    _draw_tree_recursive(graph, node, nodes, node_colors, node_data, edge_data)

    pos = graphviz_layout(graph, prog="dot")

    edge_x = []
    edge_y = []
    edge_hovertext = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_dict = edge_data[(edge[0], edge[1])]
        edge_hovertext.append(f"Action: {edge_dict["action"]}, Edge visits: {edge_dict["edge_visits"]}")
        edge_hovertext.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="gray"),
        hovertext=edge_hovertext,
        hoverinfo="text",
        mode="lines+markers",
        marker=dict(
            size=1,
            color="gray",
            symbol="arrow-bar-up",  # Symbol to represent the direction
        ),
    )

    node_x = []
    node_y = []
    node_hovertext = []
    node_Q_values = []

    for node in graph.nodes():
        hover_text = ""
        for key, value in node_data[node].items():
            hover_text += f"{key}: {value}, "
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_hovertext.append(hover_text)
        node_Q_values.append(node_data[node]["Q"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=node_hovertext,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_Q_values,
            size=10,
            colorbar=dict(
                thickness=15, title="Q value", xanchor="left", titleside="right"
            ),
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )
    fig.show()
