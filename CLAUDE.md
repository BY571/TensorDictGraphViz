# CLAUDE.md

## Project

**tensordictviz** — Visualize neural network architectures built with TorchRL's `TensorDictModule` / `TensorDictSequential`. Renders graph diagrams showing each module's input/output keys and internal layers using Graphviz.

TorchRL's `TensorDictSequential` wires modules together by matching output keys to input keys (a dataflow graph), but this "key-plumbing" is invisible in `print(model)`. This tool makes the data flow explicit: which modules share inputs, which produce intermediate keys consumed downstream, etc.

## Architecture

```
tensordictviz/
├── __init__.py              # Public API: ModelVisualizer, GraphvizBackend, VisualizationBackend
├── model_visualizer.py      # Core visualizer — dispatches to backends based on model type
└── backends/
    ├── __init__.py
    ├── base_backend.py      # VisualizationBackend ABC
    ├── graphviz_backend.py  # Graphviz (Digraph) implementation
    └── excalidraw_backend.py  # Stub — not yet implemented
```

**ModelVisualizer** detects the model type (nn.Sequential, TensorDictModule, TensorDictSequential, generic nn.Module) and calls the appropriate `_visualize_*` method. The backend interface (`VisualizationBackend`) provides `create_node`, `create_edge`, `render`, `view`, `clear`.

## Development

```bash
# Install with uv (core only — graphviz)
uv sync

# Install with torch/tensordict for full functionality
uv sync --extra torch

# Install dev dependencies
uv sync --extra dev

# Run example
uv run python examples/example.py

# Run tests
uv run pytest
```

## Task Backlog

| Priority | Task | Effort |
|----------|------|--------|
| ~~**P0**~~ | ~~Fix nested key crash~~ — Done: added `_format_key`/`_format_keys`/`_join_keys` helpers | ~~Small~~ |
| ~~**P0**~~ | ~~Fix `ProbabilisticTensorDictModule` crash~~ — Done: guarded `.module` access with `getattr` | ~~Small~~ |
| ~~**P1**~~ | ~~Add basic tests~~ — Done: 30 tests in `tests/test_model_visualizer.py` (3 viz paths, nested keys, ProbabilisticTDM, error handling, helpers) | ~~Medium~~ |
| **P2** | Fix backend abstraction leak — `_visualize_td_sequential` bypasses backend interface, calls `self.backend.graph` directly | Medium |
| **P2** | Support nested `TensorDictSequential` (recursive visualization) | Medium |
| **P3** | Improve graph aesthetics (cluster rank alignment, arrow routing, final output keys) | Medium |
| **P3** | Clean up Excalidraw backend stub or remove it | Small |
| **P4** | Update README with current API and fresh example images | Small |
