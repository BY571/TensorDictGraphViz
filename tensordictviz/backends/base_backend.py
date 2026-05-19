"""Backend abstract base class.

A backend produces a graph in some output format (Graphviz, HTML, ...).
The visualizer is backend-agnostic: it calls ``create_node``, ``create_edge``,
``subgraph`` (as a context manager), and ``set_graph_attr`` against the
current scope.
"""

from abc import ABC, abstractmethod


class VisualizationBackend(ABC):
    @abstractmethod
    def create_node(self, node_id: str, label: str, shape: str = "box", **attrs):
        pass

    @abstractmethod
    def create_edge(self, from_node: str, to_node: str, **attrs):
        pass

    @abstractmethod
    def set_graph_attr(self, **attrs):
        """Set attributes on the current graph/subgraph scope."""

    @abstractmethod
    def subgraph(self, name: str, **attrs):
        """Context manager that scopes create_node/create_edge to a new subgraph."""

    @abstractmethod
    def render(self, filename: str, format: str = "svg"):
        pass

    @abstractmethod
    def render_svg_string(self) -> str:
        """Return the current graph as an in-memory SVG string (for Jupyter)."""

    @abstractmethod
    def view(self, wait: bool = False):
        pass

    @abstractmethod
    def clear(self):
        pass
