from graphviz import Digraph

from .base_backend import VisualizationBackend


class GraphvizBackend(VisualizationBackend):
    def __init__(self):
        self.graph = Digraph(comment="Model Visualization")

    def create_node(self, node_id: str, label: str, shape: str = "box"):
        self.graph.node(node_id, label, shape=shape)

    def create_edge(self, from_node: str, to_node: str, style: str = "solid"):
        self.graph.edge(from_node, to_node, style=style)

    def render(self, filename: str, format: str = "svg"):
        self.graph.render(filename, format=format, cleanup=True)
        print(f"Graph saved as {filename}.{format}")

    def view(self, wait=False):
        self.graph.view(cleanup=True, quiet=True, quiet_view=True)
        if wait:
            input("Press Enter to continue...")

    def clear(self):
        self.graph.clear()
