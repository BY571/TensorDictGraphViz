from contextlib import contextmanager

from .base_backend import VisualizationBackend


class ExcalidrawBackend(VisualizationBackend):
    def __init__(self):
        self.elements = []

    def create_node(self, node_id: str, label: str, shape: str = "rectangle", **attrs):
        # Implementation for Excalidraw JSON structure
        pass

    def create_edge(self, from_node: str, to_node: str, **attrs):
        # Implementation for Excalidraw JSON structure
        pass

    def set_graph_attr(self, **attrs):
        # No-op for now
        pass

    @contextmanager
    def subgraph(self, name: str, **attrs):
        # No-op context manager for now
        yield

    def render(self, filename: str, format: str = "json"):
        # Convert elements to Excalidraw format and save
        pass

    def view(self):
        # Optionally, open the Excalidraw file in a viewer
        pass

    def clear(self):
        self.elements.clear()
