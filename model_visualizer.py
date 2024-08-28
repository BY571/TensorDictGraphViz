from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

# import VisualizationBackend  # Import the base class
from backends import GraphvizBackend  # Import GraphvizBackend
from tensordict.nn import TensorDictModule, TensorDictSequential


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

    def visualize(self, render: bool = True, model: Optional[Any] = None):
        if model is not None:
            self.model = model
        elif self.model is None:
            raise ValueError("No model provided for visualization.")

        if isinstance(self.model, nn.Sequential) and not isinstance(self.model, TensorDictSequential):
            self._visualize_sequential(self.model)
        elif isinstance(self.model, (TensorDictModule, TensorDictSequential)):
            self._visualize_td_sequential(self.model)
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

    def _visualize_module(self, model, parent_name, subgraph):
        first_node = None
        prev_node = None

        for i, (name, layer) in enumerate(model.named_children()):
            layer_name = f"{parent_name}_layer_{i}"
            label = self._get_layer_label(layer)
            
            subgraph.node(layer_name, label, shape="box")
            if prev_node:
                subgraph.edge(prev_node, layer_name)
            else:
                first_node = layer_name
            prev_node = layer_name

        return first_node, prev_node

    def _visualize_td_module(self, td_module, subgraph, index):
        in_keys = ", ".join(td_module.in_keys) if td_module.in_keys else "None"
        out_keys = ", ".join(td_module.out_keys) if td_module.out_keys else "None"

        module_name = f"TDModule_{index}"

        with subgraph.subgraph(name=f"cluster_{module_name}") as s:
            s.attr(label=module_name, style="filled", color="lightgrey", rankdir="TB")
            
            # Entry node
            entry_node_name = f"{module_name}_entry"
            s.node(entry_node_name, f"In Key: {in_keys}", shape="box", style="filled", fillcolor="white")
            
            # Internal module subgraph
            with s.subgraph(name=f"cluster_{module_name}_internal") as internal:
                internal.attr(label="Internal Module", style="filled", color="white")
                
                first_internal_node, last_internal_node = self._visualize_module(td_module.module, module_name, internal)
                
                if first_internal_node is None:
                    # If the internal module is empty, create a dummy node
                    dummy_name = f"{module_name}_internal_dummy"
                    internal.node(dummy_name, "Empty Module", shape="box")
                    first_internal_node = last_internal_node = dummy_name

            # Exit node
            exit_node_name = f"{module_name}_exit"
            s.node(exit_node_name, f"Out Key: {out_keys}", shape="box", style="filled", fillcolor="white")
            
            # Connect nodes
            s.edge(entry_node_name, first_internal_node)
            s.edge(last_internal_node, exit_node_name)

        return entry_node_name, exit_node_name

    def _visualize_td_sequential(self, model):
        self.backend.graph.attr(rankdir="TB", splines="ortho")
        
        with self.backend.graph.subgraph(name="cluster_td_sequential") as c:
            c.attr(label="TensorDictSequential" if isinstance(model, TensorDictSequential) else "TensorDictModule", style="filled", color="white")
            
            input_nodes = {}
            output_nodes = {}

            # Handle both TensorDictSequential and single TensorDictModule
            modules = model if isinstance(model, TensorDictSequential) else [model]

            # First pass: Create all nodes
            for i, td_module in enumerate(modules):
                entry_node, exit_node = self._visualize_td_module(td_module, c, i)
                
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
                input_key = "_".join(in_keys)
                input_node = f"input_{input_key}"
                c.node(input_node, f"Input\n{', '.join(in_keys)}", shape="box", style="filled", fillcolor="lightblue")
                for entry in entries:
                    c.edge(input_node, entry, style="dotted", color="lightblue", penwidth="0.5")

            for out_keys, exits in output_nodes.items():
                output_key = "_".join(out_keys)
                output_node = f"output_{output_key}"
                c.node(output_node, f"Output\n{', '.join(out_keys)}", shape="box", style="filled", fillcolor="lightblue")
                for exit in exits:
                    c.edge(exit, output_node, style="dotted", color="lightblue", penwidth="0.5")

            # Connect outputs to inputs based on key matching (only for TensorDictSequential)
            if isinstance(model, TensorDictSequential):
                for out_keys, exits in output_nodes.items():
                    for in_keys, entries in input_nodes.items():
                        if set(out_keys) & set(in_keys):  # If there's any overlap in keys
                            out_node = f"output_{'_'.join(out_keys)}"
                            in_node = f"input_{'_'.join(in_keys)}"
                            c.edge(out_node, in_node, style="dotted", color="lightblue", penwidth="0.5")

    def _visualize_generic_module(self, model):
        self.backend.graph.attr(rankdir="TB", splines="ortho")
        
        with self.backend.graph.subgraph(name="cluster_generic_module") as c:
            c.attr(label="Generic Module", style="filled", color="white")
            
            # Internal module subgraph
            with c.subgraph(name="cluster_generic_module_internal") as internal:
                internal.attr(label="Internal Module", style="filled", color="white")
                
                first_internal_node, last_internal_node = self._visualize_module(model, "generic_module", internal)
                
                if first_internal_node is None:
                    # If the internal module is empty, create a dummy node
                    dummy_name = "generic_module_internal_dummy"
                    internal.node(dummy_name, "Empty Module", shape="box")
                    first_internal_node = last_internal_node = dummy_name

            # Input and output nodes
            c.node("input", "Input", shape="box", style="filled", fillcolor="lightblue")
            c.node("output", "Output", shape="box", style="filled", fillcolor="lightblue")
            
            # Connect nodes
            c.edge("input", first_internal_node, style="dotted", color="lightblue", penwidth="0.5")
            c.edge(last_internal_node, "output", style="dotted", color="lightblue", penwidth="0.5")

    def _get_layer_label(self, layer):
        if isinstance(layer, nn.Linear):
            return f"Linear\nIn: {layer.in_features}, Out: {layer.out_features}"
        elif isinstance(layer, nn.Conv2d):
            return f"Conv2D\nIn: {layer.in_channels}, Out: {layer.out_channels}\nKernel: {layer.kernel_size}"
        elif isinstance(layer, nn.Dropout):
            return f"Dropout\nP: {layer.p}"
        # Add more layer types as needed
        return type(layer).__name__

    def view(self, wait=False):
        self.backend.view(wait)

    def clear(self):
        self.backend.clear()
