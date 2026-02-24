from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

# import VisualizationBackend  # Import the base class
from .backends import GraphvizBackend  # Import GraphvizBackend
from tensordict.nn import TensorDictModule, TensorDictSequential

DARK_THEME = {
    "bg": "#1a1a2e",
    "module_fill": "#16213e",
    "module_border": "#0f3460",
    "module_text": "#e0e0e0",
    "key_input": "#2d6a4f",
    "key_intermediate": "#b8860b",
    "key_output": "#1a5276",
    "key_text": "#e0e0e0",
    "edge_internal": "#e94560",
    "edge_key": "#53a8b6",
    "cluster_border": "#333366",
    "cluster_fill": "#1a1a2e",
    "font": "Helvetica",
}


def _format_key(key):
    """Stringify a TensorDict key (which may be a string or a tuple of strings)."""
    if isinstance(key, tuple):
        return ".".join(key)
    return str(key)


def _format_keys(keys):
    """Stringify a list of TensorDict keys, joining them with ', '."""
    return ", ".join(_format_key(k) for k in keys)


def _join_keys(keys, sep="_"):
    """Join a list of TensorDict keys into a single ID string."""
    return sep.join(_format_key(k) for k in keys)


class ModelVisualizer:
    def __init__(
        self,
        model: Optional[
            Union[nn.Module, nn.Sequential, TensorDictModule, TensorDictSequential]
        ] = None,
        backend="graphviz",
    ):
        self.model = model
        self.backend = self._select_backend(backend)

    def _select_backend(self, backend_name):
        if backend_name == "graphviz":
            return GraphvizBackend()
        elif backend_name == "excalidraw":
            # Placeholder for an ExcalidrawBackend implementation
            raise NotImplementedError("Excalidraw backend is not yet implemented.")
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

    def visualize(self, render: bool = True, model: Optional[Any] = None, detail: str = "compact"):
        if model is not None:
            self.model = model
        elif self.model is None:
            raise ValueError("No model provided for visualization.")

        if isinstance(self.model, nn.Sequential) and not isinstance(self.model, TensorDictSequential):
            self._visualize_sequential(self.model)
        elif isinstance(self.model, (TensorDictModule, TensorDictSequential)):
            self._visualize_td_sequential(self.model, detail=detail)
        elif isinstance(self.model, nn.Module):
            self._visualize_generic_module(self.model)
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}. "
                            f"Expected nn.Sequential, TensorDictSequential, TensorDictModule, or nn.Module.")

        if render:
            self.backend.render("model_visualization")

    def _visualize_sequential(self, model):
        prev_node = "input"
        self.backend.create_node(prev_node, "Input", shape="ellipse")

        for i, layer in enumerate(model):
            layer_name = f"layer_{i}"
            label = self._get_layer_label(layer)
            self.backend.create_node(layer_name, label)
            self.backend.create_edge(prev_node, layer_name)
            prev_node = layer_name

        self.backend.create_node("output", "Output", shape="ellipse")
        self.backend.create_edge(prev_node, "output")

    def _visualize_module(self, model, parent_name):
        first_node = None
        prev_node = None

        for i, (name, layer) in enumerate(model.named_children()):
            layer_name = f"{parent_name}_layer_{i}"
            label = self._get_layer_label(layer)

            self.backend.create_node(layer_name, label, shape="box")
            if prev_node:
                self.backend.create_edge(prev_node, layer_name)
            else:
                first_node = layer_name
            prev_node = layer_name

        return first_node, prev_node

    def _visualize_td_module(self, td_module, index):
        in_keys = _format_keys(td_module.in_keys) if td_module.in_keys else "None"
        out_keys = _format_keys(td_module.out_keys) if td_module.out_keys else "None"

        module_name = f"TDModule_{index}"

        with self.backend.subgraph(name=f"cluster_{module_name}",
                                   label=module_name, style="filled",
                                   color="lightgrey", rankdir="TB"):
            # Entry node
            entry_node_name = f"{module_name}_entry"
            self.backend.create_node(entry_node_name, f"In Key: {in_keys}",
                                     shape="box", style="filled", fillcolor="white")

            # Internal module subgraph
            with self.backend.subgraph(name=f"cluster_{module_name}_internal",
                                       label="Internal Module", style="filled",
                                       color="white"):
                inner_module = getattr(td_module, "module", None)
                if inner_module is not None:
                    first_internal_node, last_internal_node = self._visualize_module(
                        inner_module, module_name)
                else:
                    first_internal_node = last_internal_node = None

                if first_internal_node is None:
                    dummy_name = f"{module_name}_internal_dummy"
                    label = type(td_module).__name__ if inner_module is None else "Empty Module"
                    self.backend.create_node(dummy_name, label, shape="box")
                    first_internal_node = last_internal_node = dummy_name

            # Exit node
            exit_node_name = f"{module_name}_exit"
            self.backend.create_node(exit_node_name, f"Out Key: {out_keys}",
                                     shape="box", style="filled", fillcolor="white")

            # Connect nodes
            self.backend.create_edge(entry_node_name, first_internal_node)
            self.backend.create_edge(last_internal_node, exit_node_name)

        return entry_node_name, exit_node_name

    def _visualize_td_sequential(self, model, detail="compact"):
        self.backend.set_graph_attr(rankdir="TB", splines="ortho")

        seq_label = "TensorDictSequential" if isinstance(model, TensorDictSequential) else "TensorDictModule"
        with self.backend.subgraph(name="cluster_td_sequential",
                                   label=seq_label, style="filled", color="white"):
            input_nodes = {}
            output_nodes = {}

            # Handle both TensorDictSequential and single TensorDictModule
            modules = model if isinstance(model, TensorDictSequential) else [model]

            # First pass: Create all nodes
            for i, td_module in enumerate(modules):
                entry_node, exit_node = self._visualize_td_module(td_module, i)

                # Group modules by input keys
                in_keys = tuple(td_module.in_keys)
                if in_keys not in input_nodes:
                    input_nodes[in_keys] = []
                input_nodes[in_keys].append(entry_node)

                # Group modules by output keys
                out_keys = tuple(td_module.out_keys)
                if out_keys not in output_nodes:
                    output_nodes[out_keys] = []
                output_nodes[out_keys].append(exit_node)

            # Second pass: Connect nodes
            for in_keys, entries in input_nodes.items():
                input_key = _join_keys(in_keys)
                input_node = f"input_{input_key}"
                self.backend.create_node(input_node, f"Input\n{_format_keys(in_keys)}",
                                         shape="box", style="filled", fillcolor="lightblue")
                for entry in entries:
                    self.backend.create_edge(input_node, entry, style="dotted",
                                             color="lightblue", penwidth="0.5")

            for out_keys, exits in output_nodes.items():
                output_key = _join_keys(out_keys)
                output_node = f"output_{output_key}"
                self.backend.create_node(output_node, f"Output\n{_format_keys(out_keys)}",
                                         shape="box", style="filled", fillcolor="lightblue")
                for exit_n in exits:
                    self.backend.create_edge(exit_n, output_node, style="dotted",
                                             color="lightblue", penwidth="0.5")

            # Connect outputs to inputs based on key matching (only for TensorDictSequential)
            if isinstance(model, TensorDictSequential):
                for out_keys, exits in output_nodes.items():
                    for in_keys, entries in input_nodes.items():
                        if set(out_keys) & set(in_keys):
                            out_node = f"output_{_join_keys(out_keys)}"
                            in_node = f"input_{_join_keys(in_keys)}"
                            self.backend.create_edge(out_node, in_node, style="dotted",
                                                     color="lightblue", penwidth="0.5")

    def _visualize_generic_module(self, model):
        self.backend.set_graph_attr(rankdir="TB", splines="ortho")

        with self.backend.subgraph(name="cluster_generic_module",
                                   label="Generic Module", style="filled",
                                   color="white"):
            # Internal module subgraph
            with self.backend.subgraph(name="cluster_generic_module_internal",
                                       label="Internal Module", style="filled",
                                       color="white"):
                first_internal_node, last_internal_node = self._visualize_module(
                    model, "generic_module")

                if first_internal_node is None:
                    dummy_name = "generic_module_internal_dummy"
                    self.backend.create_node(dummy_name, "Empty Module", shape="box")
                    first_internal_node = last_internal_node = dummy_name

            # Input and output nodes
            self.backend.create_node("input", "Input", shape="box",
                                     style="filled", fillcolor="lightblue")
            self.backend.create_node("output", "Output", shape="box",
                                     style="filled", fillcolor="lightblue")

            # Connect nodes
            self.backend.create_edge("input", first_internal_node, style="dotted",
                                     color="lightblue", penwidth="0.5")
            self.backend.create_edge(last_internal_node, "output", style="dotted",
                                     color="lightblue", penwidth="0.5")

    def _get_layer_label(self, layer):
        if isinstance(layer, nn.Linear):
            return f"Linear\nIn: {layer.in_features}, Out: {layer.out_features}"
        elif isinstance(layer, nn.Conv2d):
            return f"Conv2D\nIn: {layer.in_channels}, Out: {layer.out_channels}\nKernel: {layer.kernel_size}"
        elif isinstance(layer, nn.Dropout):
            return f"Dropout\nP: {layer.p}"
        # Add more layer types as needed
        return type(layer).__name__

    def _get_layer_summary(self, layer):
        """One-line compact summary of a layer for compact mode."""
        if isinstance(layer, nn.Linear):
            return f"Linear({layer.in_features}\u2192{layer.out_features})"
        elif isinstance(layer, nn.Conv2d):
            return f"Conv2d({layer.in_channels}\u2192{layer.out_channels})"
        elif isinstance(layer, nn.Dropout):
            return f"Drop({layer.p})"
        return type(layer).__name__

    def _get_module_summary(self, module):
        """Chain summary like 'Linear(4\u21925) \u2192 ReLU \u2192 Linear(5\u21923)'."""
        parts = []
        for name, layer in module.named_children():
            parts.append(self._get_layer_summary(layer))
        return " \u2192 ".join(parts) if parts else type(module).__name__

    def view(self, wait=False):
        self.backend.view(wait)

    def clear(self):
        self.backend.clear()
