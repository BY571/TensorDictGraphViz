from contextlib import contextmanager

from graphviz import Digraph

from .base_backend import VisualizationBackend


class GraphvizBackend(VisualizationBackend):
    def __init__(self):
        self.graph = Digraph(comment="Model Visualization")
        self._stack = [self.graph]

    @property
    def _current(self):
        return self._stack[-1]

    def create_node(self, node_id: str, label: str, shape: str = "box", **attrs):
        self._current.node(node_id, label, shape=shape, **attrs)

    def create_edge(self, from_node: str, to_node: str, **attrs):
        self._current.edge(from_node, to_node, **attrs)

    def set_graph_attr(self, **attrs):
        self._current.attr(**attrs)

    @contextmanager
    def subgraph(self, name: str, **attrs):
        with self._current.subgraph(name=name) as s:
            if attrs:
                s.attr(**attrs)
            self._stack.append(s)
            try:
                yield
            finally:
                self._stack.pop()

    def render(self, filename: str, format: str = "svg"):
        self.graph.render(filename, format=format, cleanup=True)
        print(f"Graph saved as {filename}.{format}")

    def view(self, wait=False):
        self.graph.view(cleanup=True, quiet=True, quiet_view=True)
        if wait:
            input("Press Enter to continue...")

    def clear(self):
        self.graph.clear()
        self._stack = [self.graph]
