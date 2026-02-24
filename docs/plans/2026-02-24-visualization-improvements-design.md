# Visualization Improvements Design

## Context

The current visualization has redundant key nodes (separate Input/Output pairs per key), generic module names ("TDModule_0"), no detail control, and dated aesthetics. This redesign makes the diagrams cleaner, more informative, and visually appealing.

## Design

### 1. Single key nodes

Replace the `Input_X` + `Output_X` node pair with one node per unique key, sitting between producer and consumer modules.

Key classification by role:
- **Input-only keys**: no producer in the graph (e.g., `observation` — comes from environment)
- **Intermediate keys**: produced by one module, consumed by another (e.g., `latentspace`)
- **Output-only keys**: no consumer in the graph (e.g., `action1`, `action2`)

Each role gets a distinct color.

### 2. Collapsed modules by default

New `detail` parameter on `visualize()`:
- `detail="compact"` (default): each TDModule is a single node with an HTML label showing module name + layer summary chain (e.g., `Linear(4,5) → ReLU → Linear(5,3)`)
- `detail="full"`: current behavior with individual layer nodes inside a cluster subgraph

### 3. Dark theme

| Element | Style |
|---------|-------|
| Background | `#1a1a2e` (deep navy) |
| Module nodes | `#16213e` fill, `#e0e0e0` text, `#0f3460` border, rounded |
| Key nodes | pill-shaped (ellipse) |
| Input keys | `#2d6a4f` fill (forest green) |
| Intermediate keys | `#b8860b` fill (dark amber) |
| Output keys | `#1a5276` fill (teal blue) |
| Edges (internal) | `#e94560` (coral) |
| Edges (key flow) | `#53a8b6` (cyan), slightly thicker |
| Font | Helvetica, `#e0e0e0` |
| Cluster borders | `#333366` |

### 4. API

```python
viz = ModelVisualizer(model=actor, backend="graphviz")
viz.visualize(
    render=True,
    detail="compact",   # or "full"
)
```

No breaking changes. Default behavior improves without requiring new arguments.

### 5. Out of scope

- Interactive HTML output (future backend)
- Tensor shape annotations (requires forward pass with sample data)
- Nested TensorDictSequential (separate P2 task)
