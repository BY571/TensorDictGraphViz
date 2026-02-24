# tensordictviz/backends/base_backend.py

from abc import ABC, abstractmethod

class VisualizationBackend(ABC):
    @abstractmethod
    def create_node(self, node_id: str, label: str, shape: str):
        pass

    @abstractmethod
    def create_edge(self, from_node: str, to_node: str, style: str):
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
