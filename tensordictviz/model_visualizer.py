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
        T = DARK_THEME
        first_node = None
        prev_node = None

        for i, (name, layer) in enumerate(model.named_children()):
            layer_name = f"{parent_name}_layer_{i}"
            label = self._get_layer_label(layer)

            self.backend.create_node(layer_name, label, shape="box",
                                     style="filled,rounded",
                                     fillcolor=T["module_fill"],
                                     fontcolor=T["module_text"],
                                     fontname=T["font"])
            if prev_node:
                self.backend.create_edge(prev_node, layer_name,
                                         color=T["edge_internal"])
            else:
                first_node = layer_name
            prev_node = layer_name

        return first_node, prev_node

    def _visualize_td_module(self, td_module, index, detail="compact"):
        if detail == "compact":
            return self._visualize_td_module_compact(td_module, index)
        else:
            return self._visualize_td_module_full(td_module, index)

    def _visualize_td_module_compact(self, td_module, index):
        """Render a TDModule as a single styled node."""
        T = DARK_THEME
        module_name = f"TDModule_{index}"

        inner_module = getattr(td_module, "module", None)
        if inner_module is not None:
            summary = self._get_module_summary(inner_module)
        else:
            summary = type(td_module).__name__

        label = f"{module_name}\n{summary}"
        node_id = f"module_{index}"

        self.backend.create_node(
            node_id, label,
            shape="box", style="filled,rounded",
            fillcolor=T["module_fill"],
            color=T["module_border"],
            fontcolor=T["module_text"],
            fontname=T["font"],
        )

        return node_id, node_id  # both entry and exit are the same node

    def _visualize_td_module_full(self, td_module, index):
        """Render a TDModule as an expanded cluster with individual layers."""
        T = DARK_THEME
        in_keys = _format_keys(td_module.in_keys) if td_module.in_keys else "None"
        out_keys = _format_keys(td_module.out_keys) if td_module.out_keys else "None"

        module_name = f"TDModule_{index}"

        with self.backend.subgraph(name=f"cluster_{module_name}",
                                   label=module_name, style="filled",
                                   color=T["cluster_border"],
                                   fillcolor=T["module_fill"],
                                   fontname=T["font"],
                                   fontcolor=T["module_text"]):
            entry_node_name = f"{module_name}_entry"
            self.backend.create_node(entry_node_name, f"In: {in_keys}",
                                     shape="box", style="filled,rounded",
                                     fillcolor=T["key_input"],
                                     fontcolor=T["key_text"],
                                     fontname=T["font"])

            with self.backend.subgraph(name=f"cluster_{module_name}_internal",
                                       label="Layers", style="filled",
                                       color=T["cluster_border"],
                                       fillcolor="#0d1b2a",
                                       fontname=T["font"],
                                       fontcolor=T["module_text"]):
                inner_module = getattr(td_module, "module", None)
                if inner_module is not None:
                    first_internal_node, last_internal_node = self._visualize_module(
                        inner_module, module_name)
                else:
                    first_internal_node = last_internal_node = None

                if first_internal_node is None:
                    dummy_name = f"{module_name}_internal_dummy"
                    label = type(td_module).__name__ if inner_module is None else "Empty Module"
                    self.backend.create_node(dummy_name, label, shape="box",
                                             style="filled,rounded",
                                             fillcolor=T["module_fill"],
                                             fontcolor=T["module_text"],
                                             fontname=T["font"])
                    first_internal_node = last_internal_node = dummy_name

            exit_node_name = f"{module_name}_exit"
            self.backend.create_node(exit_node_name, f"Out: {out_keys}",
                                     shape="box", style="filled,rounded",
                                     fillcolor=T["key_output"],
                                     fontcolor=T["key_text"],
                                     fontname=T["font"])

            self.backend.create_edge(entry_node_name, first_internal_node,
                                     color=T["edge_internal"])
            self.backend.create_edge(last_internal_node, exit_node_name,
                                     color=T["edge_internal"])

        return entry_node_name, exit_node_name

    def _visualize_td_sequential(self, model, detail="compact"):
        T = DARK_THEME
        self.backend.set_graph_attr(
            rankdir="TB", splines="ortho",
            bgcolor=T["bg"], fontname=T["font"], fontcolor=T["module_text"],
        )

        seq_label = ("TensorDictSequential"
                     if isinstance(model, TensorDictSequential)
                     else "TensorDictModule")
        with self.backend.subgraph(
                name="cluster_td_sequential", label=seq_label,
                style="filled", color=T["cluster_border"],
                fillcolor=T["cluster_fill"],
                fontname=T["font"], fontcolor=T["module_text"]):

            modules = (model if isinstance(model, TensorDictSequential)
                       else [model])

            # Per-key tracking
            produced_by = {}  # formatted_key -> [exit_node_id, ...]
            consumed_by = {}  # formatted_key -> [entry_node_id, ...]

            # First pass: create module nodes
            for i, td_module in enumerate(modules):
                entry_node, exit_node = self._visualize_td_module(
                    td_module, i, detail=detail)

                for key in td_module.in_keys:
                    fk = _format_key(key)
                    consumed_by.setdefault(fk, []).append(entry_node)

                for key in td_module.out_keys:
                    fk = _format_key(key)
                    produced_by.setdefault(fk, []).append(exit_node)

            # Second pass: create single key nodes, preserving insertion order
            all_keys = list(dict.fromkeys(
                [_format_key(k) for m in modules for k in m.in_keys] +
                [_format_key(k) for m in modules for k in m.out_keys]
            ))

            for key in all_keys:
                is_produced = key in produced_by
                is_consumed = key in consumed_by

                if is_produced and is_consumed:
                    fillcolor = T["key_intermediate"]
                elif is_produced:
                    fillcolor = T["key_output"]
                else:
                    fillcolor = T["key_input"]

                key_node_id = f"key_{key}"
                self.backend.create_node(
                    key_node_id, key,
                    shape="ellipse", style="filled",
                    fillcolor=fillcolor,
                    fontcolor=T["key_text"],
                    fontname=T["font"],
                )

                for exit_node in produced_by.get(key, []):
                    self.backend.create_edge(
                        exit_node, key_node_id,
                        color=T["edge_key"], penwidth="1.5")

                for entry_node in consumed_by.get(key, []):
                    self.backend.create_edge(
                        key_node_id, entry_node,
                        color=T["edge_key"], penwidth="1.5")

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
