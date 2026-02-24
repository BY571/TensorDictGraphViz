# tensordictviz/backends/base_backend.py

from abc import ABC, abstractmethod
from contextlib import contextmanager


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
        pass

    @abstractmethod
    def subgraph(self, name: str, **attrs):
        """Context manager that scopes create_node/create_edge to a new subgraph."""
        pass

    @abstractmethod
    def render(self, filename: str, format: str = "svg"):
        pass

    @abstractmethod
    def view(self):
        pass

    @abstractmethod
    def clear(self):
        pass
