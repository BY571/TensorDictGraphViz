# CLAUDE.md

## Project

**tensordictviz** — Visualize neural network architectures built with TorchRL's `TensorDictModule` / `TensorDictSequential`. Renders graph diagrams showing each module's input/output keys, internal layers, and **real** tensor shapes captured by a fake forward pass.

`TensorDictSequential` wires modules together by matching output keys to input keys (a dataflow graph), but this "key-plumbing" is invisible in `print(model)`. This tool makes the data flow explicit: which modules share inputs, which produce intermediate keys, where recurrent state loops back.

## Architecture

```
tensordictviz/
├── __init__.py              # Public API: visualize, ModelVisualizer, register_layer, THEMES, LAYER_REGISTRY
├── model_visualizer.py      # Core dispatch + per-path visualization + top-level visualize()
├── shape_inference.py       # ShapeInferer — fake forward pass, captures shape per key
├── layer_registry.py        # 20+ default layer formatters + @register_layer for extensions
├── themes.py                # "light" / "dark" / "print" presets + dict overrides
└── backends/
    ├── __init__.py
    ├── base_backend.py      # VisualizationBackend ABC (+ render_svg_string)
    └── graphviz_backend.py  # Graphviz Digraph implementation
```

**Flow:** `visualize(model)` → `ModelVisualizer.visualize()` runs `ShapeInferer` to capture key shapes, dispatches on model type (`nn.Sequential`, `TensorDictModule`, `TensorDictSequential`, generic `nn.Module`), calls the matching `_visualize_*` method which builds nodes/edges through `self.backend`, then optionally adds the legend.

**Backend interface** (`VisualizationBackend`): `create_node`, `create_edge`, `set_graph_attr`, `subgraph` (context manager), `render`, `render_svg_string` (for Jupyter), `view`, `clear`.

## Development

```bash
uv sync --extra torch --extra dev   # all deps
uv run pytest                       # 94 tests
uv run python examples/gallery.py   # render gallery to /tmp/tviz_gallery/
```

## Task Backlog

| Priority | Task | Effort |
|----------|------|--------|
| ~~**P0**~~ | ~~Fix nested key crash~~ — Done | ~~Small~~ |
| ~~**P0**~~ | ~~Fix `ProbabilisticTensorDictModule` crash~~ — Done | ~~Small~~ |
| ~~**P1**~~ | ~~Add basic tests~~ — Done (94 tests across two files) | ~~Medium~~ |
| ~~**P2**~~ | ~~Fix backend abstraction leak~~ — Done | ~~Medium~~ |
| ~~**P3**~~ | ~~Improve graph aesthetics~~ — Done | ~~Medium~~ |
| ~~**P2**~~ | ~~Real shape inference~~ — Done: `ShapeInferer` runs a fake forward pass, falls back to static heuristic on failure | ~~Medium~~ |
| ~~**P2**~~ | ~~Broader layer coverage~~ — Done: ~30 layer types in registry, extensible via `@register_layer` | ~~Medium~~ |
| ~~**P2**~~ | ~~TorchRL-aware rendering~~ — Done: ProbabilisticTDM highlighted, recurrent state keys detected and rendered with dashed yellow style | ~~Medium~~ |
| ~~**P2**~~ | ~~Ease of use~~ — Done: top-level `visualize()`, `_repr_svg_` for Jupyter, `.save()`, themes, on-diagram legend | ~~Small~~ |
| ~~**P3**~~ | ~~Clean up Excalidraw stub~~ — Done: removed | ~~Small~~ |
| ~~**P4**~~ | ~~Update README with current API and fresh example images~~ — Done | ~~Small~~ |
| **P2** | Recursive nested `TensorDictSequential` (Phase B) | Medium |
| **P3** | Interactive HTML backend with pan/zoom/hover (Phase B) | Large |
