"""Dispatch + rendering for nn.Sequential, TensorDictModule, TensorDictSequential,
and generic nn.Module.

The visualizer is intentionally thin — heavy lifting lives in:

- ``layer_registry`` for per-layer label/summary formatting,
- ``shape_inference`` for fake-forward-pass shape capture,
- ``themes`` for color/font presets,
- ``backends`` for the actual rendering API.

This module wires those together and produces dataflow graphs that show
which modules share inputs, which produce intermediate keys, and which
recurrent state keys feed back through LSTM/GRU layers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch.nn as nn

from .backends import GraphvizBackend
from .layer_registry import get_label as _layer_label
from .layer_registry import get_summary as _layer_summary
from .shape_inference import ShapeInferer
from .themes import resolve_theme

# --- Optional integrations: tensordict / torchrl ---------------------------

try:
    from tensordict.nn import (
        ProbabilisticTensorDictModule,
        TensorDictModule,
        TensorDictSequential,
    )

    _HAS_TD = True
except ImportError:
    _HAS_TD = False
    TensorDictModule = TensorDictSequential = None  # type: ignore
    ProbabilisticTensorDictModule = None  # type: ignore

try:
    from torchrl.modules.tensordict_module.rnn import GRUModule, LSTMModule

    _RECURRENT_TYPES: Tuple[type, ...] = (LSTMModule, GRUModule)
except ImportError:
    _RECURRENT_TYPES = ()


# --- Key helpers -----------------------------------------------------------


def _format_key(key) -> str:
    if isinstance(key, tuple):
        return ".".join(key)
    return str(key)


def _format_keys(keys) -> str:
    return ", ".join(_format_key(k) for k in keys)


def _join_keys(keys, sep: str = "_") -> str:
    return sep.join(_format_key(k) for k in keys)


def _shape_label(shape: Optional[Iterable[int]]) -> str:
    """Render a shape for a key node label.

    Drops the batch dim. Empty result returns "" so the caller can omit it.
    """
    if shape is None:
        return ""
    rest = list(shape[1:]) if len(shape) > 1 else list(shape)
    if not rest:
        return ""
    return "[" + ", ".join(str(d) for d in rest) + "]"


# --- Module classification --------------------------------------------------


def _is_probabilistic(td_module) -> bool:
    return (
        _HAS_TD
        and ProbabilisticTensorDictModule is not None
        and isinstance(td_module, ProbabilisticTensorDictModule)
    )


def _is_recurrent(td_module) -> bool:
    if _RECURRENT_TYPES and isinstance(td_module, _RECURRENT_TYPES):
        return True
    # Name-based fallback if torchrl isn't installed but module came from there.
    name = type(td_module).__name__
    return name in {"LSTMModule", "GRUModule"}


def _module_kind_label(td_module, index: int) -> str:
    """Short identifier shown above the module's body (e.g. 'LSTMModule_0')."""
    cls = type(td_module).__name__
    if cls in {"TensorDictModule", "TensorDictSequential"}:
        return f"TDModule_{index}"
    return f"{cls}_{index}"


def _state_keys(modules) -> set:
    """Return formatted keys that appear in BOTH in_keys and out_keys of one module.

    These are recurrent/state keys (e.g., LSTM hidden states) — they get a
    distinct style so the loop-back nature is visible.
    """
    states: set = set()
    for m in modules:
        ins = {_format_key(k) for k in m.in_keys}
        outs = {_format_key(k) for k in m.out_keys}
        states |= ins & outs
    return states


# --- Top-level convenience -------------------------------------------------


def visualize(
    model,
    *,
    backend: str = "graphviz",
    render: bool = False,
    detail: str = "compact",
    theme: Union[str, dict] = "light",
    sample_input=None,
    show_legend: bool = True,
):
    """One-liner: build a ModelVisualizer, visualize, and return it.

    Returns the visualizer so callers can chain ``.view()``, ``.save()``,
    or use Jupyter's auto-rendering via ``_repr_svg_``.
    """
    viz = ModelVisualizer(model=model, backend=backend)
    viz.visualize(
        render=render,
        detail=detail,
        theme=theme,
        sample_input=sample_input,
        show_legend=show_legend,
    )
    return viz


# --- ModelVisualizer --------------------------------------------------------


class ModelVisualizer:
    def __init__(
        self,
        model=None,
        backend: str = "graphviz",
    ):
        self.model = model
        self.backend = self._select_backend(backend)
        self.theme: Dict[str, str] = resolve_theme("light")
        self.key_shapes: Dict[str, Tuple[int, ...]] = {}
        self._state_keys: set = set()

    # -- backend / lifecycle ------------------------------------------------

    def _select_backend(self, backend_name):
        if backend_name == "graphviz":
            return GraphvizBackend()
        raise ValueError(f"Unsupported backend: {backend_name}")

    def view(self, wait=False):
        self.backend.view(wait)

    def save(self, path: str, format: str = "svg") -> str:
        """Render to disk. Returns the output path (without the format suffix)."""
        self.backend.render(path, format=format)
        return path

    def clear(self):
        self.backend.clear()

    # Jupyter integration: auto-display in notebooks.
    def _repr_svg_(self):
        if hasattr(self.backend, "render_svg_string"):
            return self.backend.render_svg_string()
        return None

    # -- main entry point ---------------------------------------------------

    def visualize(
        self,
        render: bool = True,
        model: Optional[Any] = None,
        detail: str = "compact",
        theme: Union[str, dict] = "light",
        sample_input=None,
        show_legend: bool = True,
    ):
        if model is not None:
            self.model = model
        elif self.model is None:
            raise ValueError("No model provided for visualization.")

        self.theme = resolve_theme(theme)
        self.key_shapes = ShapeInferer(self.model, sample_input=sample_input).infer()

        if isinstance(self.model, nn.Sequential) and not (
            _HAS_TD and isinstance(self.model, TensorDictSequential)
        ):
            self._visualize_sequential(self.model)
            modules: List[Any] = []
        elif _HAS_TD and isinstance(
            self.model, (TensorDictModule, TensorDictSequential)
        ):
            modules = (
                list(self.model)
                if isinstance(self.model, TensorDictSequential)
                else [self.model]
            )
            self._state_keys = _state_keys(modules)
            self._visualize_td_sequential(self.model, detail=detail)
        elif isinstance(self.model, nn.Module):
            self._visualize_generic_module(self.model)
            modules = []
        else:
            raise TypeError(
                f"Unsupported model type: {type(self.model)}. "
                f"Expected nn.Sequential, TensorDictSequential, "
                f"TensorDictModule, or nn.Module."
            )

        if show_legend:
            self._add_legend(include_state=bool(self._state_keys))

        if render:
            self.backend.render("model_visualization")

    # -- Path 1: plain nn.Sequential ---------------------------------------

    def _visualize_sequential(self, model):
        T = self.theme
        self._apply_graph_attrs()

        prev_node = "input"
        self.backend.create_node(
            prev_node,
            "Input",
            shape=T["key_shape"],
            style="filled",
            fillcolor=T["key_input"],
            color=T["key_input_border"],
            fontcolor=T["key_text"],
            fontname=T["font"],
        )

        for i, layer in enumerate(model):
            layer_name = f"layer_{i}"
            self.backend.create_node(
                layer_name,
                _layer_label(layer),
                shape=T["module_shape"],
                style=self._module_style(),
                fillcolor=T["module_fill"],
                color=T["module_border"],
                fontcolor=T["module_text"],
                fontname=T["font"],
            )
            self.backend.create_edge(
                prev_node,
                layer_name,
                color=T["edge_internal"],
                penwidth=T["edge_penwidth"],
                **self._edge_shape_attrs(prev_node),
            )
            prev_node = layer_name

        self.backend.create_node(
            "output",
            "Output",
            shape=T["key_shape"],
            style="filled",
            fillcolor=T["key_output"],
            color=T["key_output_border"],
            fontcolor=T["key_text"],
            fontname=T["font"],
        )
        self.backend.create_edge(
            prev_node,
            "output",
            color=T["edge_internal"],
            penwidth=T["edge_penwidth"],
            **self._edge_shape_attrs(prev_node),
        )

    # -- Path 2: TensorDictModule / TensorDictSequential --------------------

    def _visualize_td_sequential(self, model, detail: str = "compact"):
        T = self.theme
        self._apply_graph_attrs()

        seq_label = type(model).__name__

        with self.backend.subgraph(
            name="cluster_td_sequential",
            label=seq_label,
            style="filled",
            color=T["cluster_border"],
            fillcolor=T["cluster_fill"],
            fontname=T["font"],
            fontcolor=T["module_text"],
        ):
            modules = (
                list(model) if isinstance(model, TensorDictSequential) else [model]
            )

            produced_by: Dict[str, List[str]] = {}
            consumed_by: Dict[str, List[str]] = {}

            # Pass 1: module nodes + edge endpoints
            for i, td_module in enumerate(modules):
                entry_node, exit_node = self._visualize_td_module(
                    td_module, i, detail=detail
                )

                for key in td_module.in_keys:
                    consumed_by.setdefault(_format_key(key), []).append(entry_node)
                for key in td_module.out_keys:
                    produced_by.setdefault(_format_key(key), []).append(exit_node)

            # Pass 2: dedupe + emit key nodes in first-mentioned order
            all_keys = list(
                dict.fromkeys(
                    [_format_key(k) for m in modules for k in m.in_keys]
                    + [_format_key(k) for m in modules for k in m.out_keys]
                )
            )

            for key in all_keys:
                self._emit_key_node(key, produced_by, consumed_by)

    def _visualize_td_module(self, td_module, index: int, detail: str = "compact"):
        if detail == "compact":
            return self._visualize_td_module_compact(td_module, index)
        return self._visualize_td_module_full(td_module, index)

    def _visualize_td_module_compact(self, td_module, index: int):
        T = self.theme
        kind = _module_kind_label(td_module, index)

        if _is_probabilistic(td_module):
            summary = self._probabilistic_summary(td_module)
            fillcolor = T["probabilistic_fill"]
            border = T["probabilistic_border"]
        else:
            inner_module = getattr(td_module, "module", None)
            inner_module = inner_module or self._unwrap_torchrl_inner(td_module)
            if inner_module is not None:
                summary = self._module_summary(inner_module)
            else:
                summary = type(td_module).__name__
            fillcolor = T["module_fill"]
            border = T["module_border"]

        if _is_recurrent(td_module):
            summary = f"recurrent\n{summary}"

        label = f"{kind}\n{summary}"
        node_id = f"module_{index}"
        self.backend.create_node(
            node_id,
            label,
            shape=T["module_shape"],
            style=self._module_style(),
            fillcolor=fillcolor,
            color=border,
            fontcolor=T["module_text"],
            fontname=T["font"],
            penwidth="2" if _is_probabilistic(td_module) else "1",
        )
        return node_id, node_id

    def _visualize_td_module_full(self, td_module, index: int):
        T = self.theme
        kind = _module_kind_label(td_module, index)
        in_keys = _format_keys(td_module.in_keys) if td_module.in_keys else "None"
        out_keys = _format_keys(td_module.out_keys) if td_module.out_keys else "None"

        is_prob = _is_probabilistic(td_module)
        cluster_fill = T["probabilistic_fill"] if is_prob else T["module_fill"]
        cluster_border = T["probabilistic_border"] if is_prob else T["cluster_border"]

        with self.backend.subgraph(
            name=f"cluster_{kind}",
            label=kind,
            style="filled",
            color=cluster_border,
            fillcolor=cluster_fill,
            fontname=T["font"],
            fontcolor=T["module_text"],
        ):
            entry_node = f"{kind}_entry"
            self.backend.create_node(
                entry_node,
                f"In: {in_keys}",
                shape=T["module_shape"],
                style=self._module_style(),
                fillcolor=T["key_input"],
                color=T["key_input_border"],
                fontcolor=T["key_text"],
                fontname=T["font"],
            )

            with self.backend.subgraph(
                name=f"cluster_{kind}_internal",
                label="Layers",
                style="filled",
                color=T["cluster_border"],
                fillcolor=T["cluster_inner_fill"],
                fontname=T["font"],
                fontcolor=T["module_text"],
            ):
                inner_module = getattr(td_module, "module", None) or self._unwrap_torchrl_inner(
                    td_module
                )
                if is_prob:
                    dummy = f"{kind}_internal_dummy"
                    self.backend.create_node(
                        dummy,
                        self._probabilistic_summary(td_module),
                        shape=T["module_shape"],
                        style=self._module_style(),
                        fillcolor=T["probabilistic_fill"],
                        color=T["probabilistic_border"],
                        fontcolor=T["module_text"],
                        fontname=T["font"],
                        penwidth="2",
                    )
                    first_internal_node = last_internal_node = dummy
                elif inner_module is not None:
                    first_internal_node, last_internal_node = self._visualize_module(
                        inner_module, kind
                    )
                else:
                    first_internal_node = last_internal_node = None

                if first_internal_node is None:
                    dummy = f"{kind}_internal_dummy"
                    label = (
                        type(td_module).__name__
                        if inner_module is None
                        else "Empty Module"
                    )
                    self.backend.create_node(
                        dummy,
                        label,
                        shape=T["module_shape"],
                        style=self._module_style(),
                        fillcolor=T["module_fill"],
                        color=T["module_border"],
                        fontcolor=T["module_text"],
                        fontname=T["font"],
                    )
                    first_internal_node = last_internal_node = dummy

            exit_node = f"{kind}_exit"
            self.backend.create_node(
                exit_node,
                f"Out: {out_keys}",
                shape=T["module_shape"],
                style=self._module_style(),
                fillcolor=T["key_output"],
                color=T["key_output_border"],
                fontcolor=T["key_text"],
                fontname=T["font"],
            )

            self.backend.create_edge(
                entry_node, first_internal_node, color=T["edge_internal"],
                penwidth=T["edge_penwidth"],
            )
            self.backend.create_edge(
                last_internal_node, exit_node, color=T["edge_internal"],
                penwidth=T["edge_penwidth"],
            )

        return entry_node, exit_node

    def _visualize_module(self, model, parent_name: str):
        T = self.theme
        first_node = None
        prev_node = None
        for i, (_name, layer) in enumerate(model.named_children()):
            layer_name = f"{parent_name}_layer_{i}"
            self.backend.create_node(
                layer_name,
                _layer_label(layer),
                shape=T["module_shape"],
                style=self._module_style(),
                fillcolor=T["module_fill"],
                color=T["module_border"],
                fontcolor=T["module_text"],
                fontname=T["font"],
            )
            if prev_node:
                self.backend.create_edge(
                    prev_node, layer_name, color=T["edge_internal"],
                    penwidth=T["edge_penwidth"],
                )
            else:
                first_node = layer_name
            prev_node = layer_name
        return first_node, prev_node

    def _emit_key_node(self, key: str, produced_by, consumed_by):
        T = self.theme
        is_produced = key in produced_by
        is_consumed = key in consumed_by
        is_state = key in self._state_keys

        if is_state:
            fillcolor = T["key_state"]
            border_color = T["key_state_border"]
            border_style = "dashed"
        elif is_produced and is_consumed:
            fillcolor = T["key_intermediate"]
            border_color = T["key_intermediate_border"]
            border_style = "solid"
        elif is_produced:
            fillcolor = T["key_output"]
            border_color = T["key_output_border"]
            border_style = "solid"
        else:
            fillcolor = T["key_input"]
            border_color = T["key_input_border"]
            border_style = "solid"

        shape_str = _shape_label(self.key_shapes.get(key))
        # Fallback: pull a dim from the first/last Linear if shape inference came up empty.
        if not shape_str:
            shape_str = self._fallback_shape_label_for_key(key, produced_by, consumed_by)

        label = f"{key} {shape_str}".rstrip() if shape_str else key

        key_node_id = f"key_{key}"
        edge_color = T["edge_state"] if is_state else T["edge_key"]
        edge_style = "dashed" if is_state else "solid"

        self.backend.create_node(
            key_node_id,
            label,
            shape=T["key_shape"],
            style=f"filled,{border_style}" if border_style != "solid" else "filled",
            fillcolor=fillcolor,
            color=border_color,
            fontcolor=T["key_text"],
            fontname=T["font"],
        )

        for exit_node in produced_by.get(key, []):
            self.backend.create_edge(
                exit_node,
                key_node_id,
                color=edge_color,
                style=edge_style,
                penwidth=T["edge_penwidth"],
            )
        for entry_node in consumed_by.get(key, []):
            self.backend.create_edge(
                key_node_id,
                entry_node,
                color=edge_color,
                style=edge_style,
                penwidth=T["edge_penwidth"],
            )

    # -- Path 3: generic nn.Module -----------------------------------------

    def _visualize_generic_module(self, model):
        T = self.theme
        self._apply_graph_attrs()

        with self.backend.subgraph(
            name="cluster_generic_module",
            label="Module",
            style="filled",
            color=T["cluster_border"],
            fillcolor=T["cluster_fill"],
            fontname=T["font"],
            fontcolor=T["module_text"],
        ):
            with self.backend.subgraph(
                name="cluster_generic_module_internal",
                label="Layers",
                style="filled",
                color=T["cluster_border"],
                fillcolor=T["cluster_inner_fill"],
                fontname=T["font"],
                fontcolor=T["module_text"],
            ):
                first_internal_node, last_internal_node = self._visualize_module(
                    model, "generic_module"
                )
                if first_internal_node is None:
                    dummy = "generic_module_internal_dummy"
                    self.backend.create_node(
                        dummy,
                        "Empty Module",
                        shape=T["module_shape"],
                        style=self._module_style(),
                        fillcolor=T["module_fill"],
                        color=T["module_border"],
                        fontcolor=T["module_text"],
                        fontname=T["font"],
                    )
                    first_internal_node = last_internal_node = dummy

            in_shape = _shape_label(self.key_shapes.get("input"))
            out_shape = _shape_label(self.key_shapes.get("output"))
            self.backend.create_node(
                "input",
                f"Input {in_shape}".rstrip() if in_shape else "Input",
                shape=T["key_shape"],
                style="filled",
                fillcolor=T["key_input"],
                color=T["key_input_border"],
                fontcolor=T["key_text"],
                fontname=T["font"],
            )
            self.backend.create_node(
                "output",
                f"Output {out_shape}".rstrip() if out_shape else "Output",
                shape=T["key_shape"],
                style="filled",
                fillcolor=T["key_output"],
                color=T["key_output_border"],
                fontcolor=T["key_text"],
                fontname=T["font"],
            )
            self.backend.create_edge(
                "input", first_internal_node, color=T["edge_key"],
                penwidth=T["edge_penwidth"],
            )
            self.backend.create_edge(
                last_internal_node, "output", color=T["edge_key"],
                penwidth=T["edge_penwidth"],
            )

    # -- Legend ------------------------------------------------------------

    def _add_legend(self, include_state: bool = False):
        T = self.theme
        with self.backend.subgraph(
            name="cluster_legend",
            label="Legend",
            style=self._module_style(),
            color=T["legend_border"],
            fillcolor=T["legend_fill"],
            fontname=T["font"],
            fontcolor=T["module_text"],
            fontsize="10",
        ):
            # Each entry: (node_id, label, fill, border, shape, style, fontcolor).
            # Shapes and borders track the theme so the legend always matches
            # the graph it explains.
            key_shape = T["key_shape"]
            entries = [
                ("legend_input", "Input key", T["key_input"], T["key_input_border"],
                 key_shape, "solid", T["key_text"]),
                ("legend_inter", "Intermediate", T["key_intermediate"],
                 T["key_intermediate_border"], key_shape, "solid", T["key_text"]),
                ("legend_output", "Output key", T["key_output"], T["key_output_border"],
                 key_shape, "solid", T["key_text"]),
            ]
            if include_state:
                entries.append(
                    ("legend_state", "Recurrent state", T["key_state"],
                     T["key_state_border"], key_shape, "dashed", T["key_text"])
                )
            entries.append(
                ("legend_module", "Module", T["module_fill"], T["module_border"],
                 T["module_shape"], "module", T["module_text"])
            )

            for node_id, label, fill, border, shape, style, fontcolor in entries:
                if style == "module":
                    node_style = self._module_style()
                elif style == "dashed":
                    node_style = "filled,dashed"
                else:
                    node_style = "filled"
                self.backend.create_node(
                    node_id,
                    label,
                    shape=shape,
                    style=node_style,
                    fillcolor=fill,
                    color=border,
                    fontcolor=fontcolor,
                    fontname=T["font"],
                    fontsize="10",
                )

            # Invisible edges to vertically stack the legend entries.
            for a, b in zip(entries, entries[1:]):
                self.backend.create_edge(a[0], b[0], style="invis")

    # -- Helpers -----------------------------------------------------------

    def _module_style(self) -> str:
        """Module-node style string, honoring the theme's corner setting."""
        return "filled,rounded" if self.theme.get("module_rounded", True) else "filled"

    def _apply_graph_attrs(self):
        """Set canvas + layout attributes from the active theme."""
        T = self.theme
        self.backend.set_graph_attr(
            rankdir=T["rankdir"],
            splines=T["splines"],
            bgcolor=T["bg"],
            fontname=T["font"],
            fontcolor=T["module_text"],
            nodesep=T["nodesep"],
            ranksep=T["ranksep"],
        )

    def _edge_shape_attrs(self, source_node: str) -> Dict[str, str]:
        """Edge attrs labelling an edge with the shape of the tensor leaving
        ``source_node``. Empty dict when no shape was captured."""
        label = _shape_label(self.key_shapes.get(source_node))
        if not label:
            return {}
        return {
            "label": label,
            "fontsize": "9",
            "fontcolor": self.theme["module_text"],
        }

    def _unwrap_torchrl_inner(self, td_module):
        """Some torchrl modules expose their core nn.Module under a different attr."""
        for attr in ("lstm", "gru", "rnn"):
            inner = getattr(td_module, attr, None)
            if isinstance(inner, nn.Module):
                return inner
        return None

    def _probabilistic_summary(self, td_module) -> str:
        dist_cls = getattr(td_module, "distribution_class", None)
        dist_name = dist_cls.__name__ if dist_cls is not None else "?"
        ins = _format_keys(td_module.in_keys) if td_module.in_keys else "?"
        outs = _format_keys(td_module.out_keys) if td_module.out_keys else "?"
        return f"Probabilistic\nDist: {dist_name}\n({ins}) → ({outs})"

    def _module_summary(self, module) -> str:
        parts = [_layer_summary(layer) for _, layer in module.named_children()]
        return " → ".join(parts) if parts else type(module).__name__

    # Back-compat wrappers (the test suite calls these directly).
    def _get_layer_label(self, layer):
        return _layer_label(layer)

    def _get_layer_summary(self, layer):
        return _layer_summary(layer)

    def _get_module_summary(self, module):
        return self._module_summary(module)

    def _get_first_linear(self, module):
        for child in module.modules():
            if isinstance(child, nn.Linear):
                return child
        return None

    def _get_last_linear(self, module):
        last = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                last = child
        return last

    def _fallback_shape_label_for_key(self, key, produced_by, consumed_by):
        """When ShapeInferer returns nothing, peek at adjacent Linear layers."""
        # If key is produced by a module, use that module's last Linear out_features.
        # Else if consumed, use the first Linear's in_features.
        # We don't have direct module references at this point, so we re-walk.
        if not _HAS_TD or self.model is None:
            return ""
        if not isinstance(self.model, (TensorDictModule, TensorDictSequential)):
            return ""
        modules = (
            list(self.model)
            if isinstance(self.model, TensorDictSequential)
            else [self.model]
        )
        for m in modules:
            inner = getattr(m, "module", None) or self._unwrap_torchrl_inner(m)
            if inner is None:
                continue
            if key in [_format_key(k) for k in m.out_keys]:
                last = self._get_last_linear(inner)
                if last is not None:
                    return f"[{last.out_features}]"
            if key in [_format_key(k) for k in m.in_keys]:
                first = self._get_first_linear(inner)
                if first is not None:
                    return f"[{first.in_features}]"
        return ""
