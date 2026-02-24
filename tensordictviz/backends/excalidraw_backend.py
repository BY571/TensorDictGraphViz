from .base_backend import VisualizationBackend  # Import the base class


class ExcalidrawBackend(VisualizationBackend):
    def __init__(self):
        self.elements = []

    def create_node(self, node_id: str, label: str, shape: str = "rectangle"):
        # Implementation for Excalidraw JSON structure
        pass

    def create_edge(self, from_node: str, to_node: str, style: str = "solid"):
        # Implementation for Excalidraw JSON structure
        pass

    def render(self, filename: str, format: str = "json"):
        # Convert elements to Excalidraw format and save
        pass

    def view(self):
        # Optionally, open the Excalidraw file in a viewer
        pass

    def clear(self):
        self.elements.clear()
