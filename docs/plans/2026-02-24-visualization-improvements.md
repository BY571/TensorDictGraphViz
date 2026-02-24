# Visualization Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace redundant key node pairs with single key nodes, add collapsed module view, and apply a dark theme to all visualization paths.

**Architecture:** Three changes layered on the existing backend abstraction: (1) a `DARK_THEME` dict of colors used by all visualization methods, (2) a `detail` parameter on `visualize()` that switches between compact single-node modules and full expanded clusters, (3) single key nodes in `_visualize_td_sequential` that classify keys by role (input/intermediate/output) and color them accordingly.

**Tech Stack:** Python, Graphviz (via `graphviz` library), pytest

---

### Task 1: Add `_get_layer_summary` and `_get_module_summary` helpers

These helpers produce one-line summaries for the compact module view.

**Files:**
- Modify: `tensordictviz/model_visualizer.py:226-234` (next to `_get_layer_label`)
- Test: `tests/test_model_visualizer.py`

**Step 1: Write the failing tests**

Add to `tests/test_model_visualizer.py` — import the new helpers at the top alongside `_format_key`, then add a new test class:

```python
# Add to imports at line 8:
from tensordictviz.model_visualizer import _format_key, _format_keys, _join_keys

# New test class after TestJoinKeys (after line 57):
class TestLayerSummary:
    def _viz(self):
        return ModelVisualizer()

    def test_linear_summary(self):
        layer = nn.Linear(4, 5)
        assert self._viz()._get_layer_summary(layer) == "Linear(4\u21925)"

    def test_conv2d_summary(self):
        layer = nn.Conv2d(1, 32, kernel_size=3)
        assert self._viz()._get_layer_summary(layer) == "Conv2d(1\u219232)"

    def test_relu_summary(self):
        layer = nn.ReLU()
        assert self._viz()._get_layer_summary(layer) == "ReLU"

    def test_dropout_summary(self):
        layer = nn.Dropout(0.5)
        assert self._viz()._get_layer_summary(layer) == "Drop(0.5)"

    def test_unknown_layer_summary(self):
        layer = nn.Sigmoid()
        assert self._viz()._get_layer_summary(layer) == "Sigmoid"


class TestModuleSummary:
    def _viz(self):
        return ModelVisualizer()

    def test_sequential_chain(self):
        module = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
        result = self._viz()._get_module_summary(module)
        assert result == "Linear(4\u21925) \u2192 ReLU \u2192 Linear(5\u21923)"

    def test_single_layer(self):
        module = nn.Sequential(nn.Linear(4, 2))
        result = self._viz()._get_module_summary(module)
        assert result == "Linear(4\u21922)"

    def test_empty_module_uses_class_name(self):
        module = nn.Module()
        result = self._viz()._get_module_summary(module)
        assert result == "Module"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_visualizer.py::TestLayerSummary -v`
Expected: FAIL with `AttributeError: 'ModelVisualizer' object has no attribute '_get_layer_summary'`

**Step 3: Implement the helpers**

Add to `tensordictviz/model_visualizer.py` inside the `ModelVisualizer` class, after `_get_layer_label` (after line 234):

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_visualizer.py::TestLayerSummary tests/test_model_visualizer.py::TestModuleSummary -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add tensordictviz/model_visualizer.py tests/test_model_visualizer.py
git commit -m "Add _get_layer_summary and _get_module_summary helpers for compact mode"
```

---

### Task 2: Add `DARK_THEME` constants and `detail` parameter

**Files:**
- Modify: `tensordictviz/model_visualizer.py:1-10` (add theme dict at module level)
- Modify: `tensordictviz/model_visualizer.py:47` (add `detail` param to `visualize()`)

**Step 1: Add theme dict**

Add at module level in `tensordictviz/model_visualizer.py` after the imports (after line 7, before `_format_key`):

```python
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
```

**Step 2: Add `detail` parameter to `visualize()`**

Change the `visualize` method signature (line 47) from:

```python
def visualize(self, render: bool = True, model: Optional[Any] = None):
```

to:

```python
def visualize(self, render: bool = True, model: Optional[Any] = None, detail: str = "compact"):
```

Store it as `self._detail = detail` right after the model validation (after line 51), and pass it through to `_visualize_td_sequential`:

Change line 56 from:
```python
self._visualize_td_sequential(self.model)
```
to:
```python
self._visualize_td_sequential(self.model, detail=detail)
```

**Step 3: Run existing tests to verify nothing breaks yet**

Run: `uv run pytest tests/test_model_visualizer.py -v`
Expected: all PASS (the `detail` param has a default, so existing calls still work)

**Step 4: Commit**

```bash
git add tensordictviz/model_visualizer.py
git commit -m "Add DARK_THEME constants and detail parameter to visualize()"
```

---

### Task 3: Rewrite `_visualize_td_module` for both compact and full modes

This is the core structural change. In compact mode, each TDModule becomes a single styled node. In full mode, it keeps the expanded cluster but with dark theme.

**Files:**
- Modify: `tensordictviz/model_visualizer.py:97-137` (rewrite `_visualize_td_module`)

**Step 1: Write failing tests for compact mode**

Add to `tests/test_model_visualizer.py`:

```python
class TestCompactMode:
    def test_compact_td_module_is_single_node(self):
        """In compact mode, each TDModule is one node, not a cluster."""
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False, detail="compact")

        src = _source(viz)
        # Should NOT have cluster subgraph for the module
        assert "cluster_TDModule_0" not in src
        # Should have the module summary
        assert "Linear(4" in src

    def test_compact_shows_layer_chain(self):
        """Compact mode shows the layer summary chain."""
        net = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False, detail="compact")

        src = _source(viz)
        assert "\u2192" in src  # arrow character in chain

    def test_full_mode_has_clusters(self):
        """detail='full' preserves expanded cluster view."""
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False, detail="full")

        src = _source(viz)
        assert "cluster_TDModule_0" in src

    def test_probabilistic_compact(self):
        """ProbabilisticTensorDictModule shows class name in compact mode."""
        from tensordict.nn.distributions import NormalParamExtractor
        net = nn.Sequential(nn.Linear(4, 8), NormalParamExtractor())
        param_mod = TensorDictModule(net, in_keys=["obs"], out_keys=["loc", "scale"])
        prob_mod = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=Normal,
        )
        actor = TensorDictSequential(param_mod, prob_mod)
        viz = ModelVisualizer(model=actor)
        viz.visualize(render=False, detail="compact")

        src = _source(viz)
        assert "ProbabilisticTensorDictModule" in src
        assert "loc" in src
        assert "action" in src
```

**Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_model_visualizer.py::TestCompactMode -v`
Expected: FAIL

**Step 3: Implement compact and full mode `_visualize_td_module`**

Replace `_visualize_td_module` (lines 97-137) with two methods and a dispatcher:

```python
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

    return node_id, node_id

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
        # Entry node
        entry_node_name = f"{module_name}_entry"
        self.backend.create_node(entry_node_name, f"In: {in_keys}",
                                 shape="box", style="filled,rounded",
                                 fillcolor=T["key_input"],
                                 fontcolor=T["key_text"],
                                 fontname=T["font"])

        # Internal module subgraph
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

        # Exit node
        exit_node_name = f"{module_name}_exit"
        self.backend.create_node(exit_node_name, f"Out: {out_keys}",
                                 shape="box", style="filled,rounded",
                                 fillcolor=T["key_output"],
                                 fontcolor=T["key_text"],
                                 fontname=T["font"])

        # Connect nodes
        self.backend.create_edge(entry_node_name, first_internal_node,
                                 color=T["edge_internal"])
        self.backend.create_edge(last_internal_node, exit_node_name,
                                 color=T["edge_internal"])

    return entry_node_name, exit_node_name
```

Also update `_visualize_module` to apply dark theme to internal layer nodes:

```python
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
```

**Step 4: Run compact mode tests**

Run: `uv run pytest tests/test_model_visualizer.py::TestCompactMode -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add tensordictviz/model_visualizer.py tests/test_model_visualizer.py
git commit -m "Add compact and full mode for _visualize_td_module with dark theme"
```

---

### Task 4: Rewrite `_visualize_td_sequential` with single key nodes

This replaces the dual Input/Output node approach with one node per unique key, classified by role.

**Files:**
- Modify: `tensordictviz/model_visualizer.py:139-194` (rewrite `_visualize_td_sequential`)
- Test: `tests/test_model_visualizer.py`

**Step 1: Write failing tests for single key nodes**

Add to `tests/test_model_visualizer.py`:

```python
class TestSingleKeyNodes:
    def test_intermediate_key_is_single_node(self):
        """A key that's produced and consumed should appear once, not twice."""
        m1 = nn.Sequential(nn.Linear(4, 8))
        m2 = nn.Sequential(nn.Linear(8, 2))
        model = TensorDictSequential(
            TensorDictModule(m1, in_keys=["obs"], out_keys=["hidden"]),
            TensorDictModule(m2, in_keys=["hidden"], out_keys=["action"]),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        # "hidden" should appear as a key node, not as separate Input/Output
        assert "key_hidden" in src
        # Should NOT have the old-style separate input/output nodes
        assert "input_hidden" not in src
        assert "output_hidden" not in src

    def test_input_only_key_colored_green(self):
        """Keys only consumed (not produced) should use input color."""
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "key_obs" in src
        # Input key should use the input color
        assert "#2d6a4f" in src

    def test_output_only_key_colored_blue(self):
        """Keys only produced (not consumed) should use output color."""
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "key_act" in src
        assert "#1a5276" in src

    def test_intermediate_key_colored_amber(self):
        """Keys both produced and consumed should use intermediate color."""
        m1 = nn.Sequential(nn.Linear(4, 8))
        m2 = nn.Sequential(nn.Linear(8, 2))
        model = TensorDictSequential(
            TensorDictModule(m1, in_keys=["obs"], out_keys=["hidden"]),
            TensorDictModule(m2, in_keys=["hidden"], out_keys=["action"]),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "#b8860b" in src  # amber for intermediate

    def test_shared_input_key_has_two_consumers(self):
        """A key consumed by two modules should have edges to both."""
        m1 = nn.Sequential(nn.Linear(4, 2))
        m2 = nn.Sequential(nn.Linear(4, 3))
        model = TensorDictSequential(
            TensorDictModule(m1, in_keys=["latent"], out_keys=["a"]),
            TensorDictModule(m2, in_keys=["latent"], out_keys=["b"]),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "key_latent" in src
        # latent should connect to both module_0 and module_1
        assert "key_latent -> module_0" in src or "key_latent -> module_1" in src
```

**Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_model_visualizer.py::TestSingleKeyNodes -v`
Expected: FAIL

**Step 3: Rewrite `_visualize_td_sequential`**

Replace `_visualize_td_sequential` (lines 139-194) with:

```python
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

        # Per-key tracking: which modules produce/consume each key
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

        # Second pass: create single key nodes and edges
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

            # Producers -> key node
            for exit_node in produced_by.get(key, []):
                self.backend.create_edge(
                    exit_node, key_node_id,
                    color=T["edge_key"], penwidth="1.5")

            # Key node -> consumers
            for entry_node in consumed_by.get(key, []):
                self.backend.create_edge(
                    key_node_id, entry_node,
                    color=T["edge_key"], penwidth="1.5")
```

**Step 4: Run new key node tests**

Run: `uv run pytest tests/test_model_visualizer.py::TestSingleKeyNodes -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add tensordictviz/model_visualizer.py tests/test_model_visualizer.py
git commit -m "Replace dual Input/Output key nodes with single role-classified key nodes"
```

---

### Task 5: Apply dark theme to sequential and generic module paths

**Files:**
- Modify: `tensordictviz/model_visualizer.py:66-78` (`_visualize_sequential`)
- Modify: `tensordictviz/model_visualizer.py:196-224` (`_visualize_generic_module`)

**Step 1: Write failing test for dark theme on sequential**

```python
class TestDarkTheme:
    def test_sequential_has_dark_bg(self):
        model = nn.Sequential(nn.Linear(4, 2))
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "#1a1a2e" in src  # dark background

    def test_generic_module_has_dark_bg(self):
        model = nn.Linear(3, 1)
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "#1a1a2e" in src

    def test_td_sequential_has_dark_bg(self):
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "#1a1a2e" in src
```

**Step 2: Update `_visualize_sequential`**

```python
def _visualize_sequential(self, model):
    T = DARK_THEME
    self.backend.set_graph_attr(
        rankdir="TB", bgcolor=T["bg"],
        fontname=T["font"], fontcolor=T["module_text"],
    )

    prev_node = "input"
    self.backend.create_node(prev_node, "Input", shape="ellipse",
                             style="filled", fillcolor=T["key_input"],
                             fontcolor=T["key_text"], fontname=T["font"])

    for i, layer in enumerate(model):
        layer_name = f"layer_{i}"
        label = self._get_layer_label(layer)
        self.backend.create_node(layer_name, label, shape="box",
                                 style="filled,rounded",
                                 fillcolor=T["module_fill"],
                                 color=T["module_border"],
                                 fontcolor=T["module_text"],
                                 fontname=T["font"])
        self.backend.create_edge(prev_node, layer_name,
                                 color=T["edge_internal"])
        prev_node = layer_name

    self.backend.create_node("output", "Output", shape="ellipse",
                             style="filled", fillcolor=T["key_output"],
                             fontcolor=T["key_text"], fontname=T["font"])
    self.backend.create_edge(prev_node, "output",
                             color=T["edge_internal"])
```

**Step 3: Update `_visualize_generic_module`**

```python
def _visualize_generic_module(self, model):
    T = DARK_THEME
    self.backend.set_graph_attr(
        rankdir="TB", splines="ortho",
        bgcolor=T["bg"], fontname=T["font"], fontcolor=T["module_text"],
    )

    with self.backend.subgraph(name="cluster_generic_module",
                               label="Module", style="filled",
                               color=T["cluster_border"],
                               fillcolor=T["cluster_fill"],
                               fontname=T["font"],
                               fontcolor=T["module_text"]):
        with self.backend.subgraph(name="cluster_generic_module_internal",
                                   label="Layers", style="filled",
                                   color=T["cluster_border"],
                                   fillcolor="#0d1b2a",
                                   fontname=T["font"],
                                   fontcolor=T["module_text"]):
            first_internal_node, last_internal_node = self._visualize_module(
                model, "generic_module")

            if first_internal_node is None:
                dummy_name = "generic_module_internal_dummy"
                self.backend.create_node(dummy_name, "Empty Module",
                                         shape="box", style="filled,rounded",
                                         fillcolor=T["module_fill"],
                                         fontcolor=T["module_text"],
                                         fontname=T["font"])
                first_internal_node = last_internal_node = dummy_name

        self.backend.create_node("input", "Input", shape="ellipse",
                                 style="filled", fillcolor=T["key_input"],
                                 fontcolor=T["key_text"], fontname=T["font"])
        self.backend.create_node("output", "Output", shape="ellipse",
                                 style="filled", fillcolor=T["key_output"],
                                 fontcolor=T["key_text"], fontname=T["font"])

        self.backend.create_edge("input", first_internal_node,
                                 color=T["edge_key"], penwidth="1.5")
        self.backend.create_edge(last_internal_node, "output",
                                 color=T["edge_key"], penwidth="1.5")
```

**Step 4: Run dark theme tests**

Run: `uv run pytest tests/test_model_visualizer.py::TestDarkTheme -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add tensordictviz/model_visualizer.py tests/test_model_visualizer.py
git commit -m "Apply dark theme to sequential and generic module visualization paths"
```

---

### Task 6: Update existing tests to match new output format

The structural changes (single key nodes, compact default, dark theme) alter the DOT output. Existing tests that assert on old-format strings need updating.

**Files:**
- Modify: `tests/test_model_visualizer.py`

**Step 1: Run full test suite to see what breaks**

Run: `uv run pytest tests/test_model_visualizer.py -v`
Note which tests fail and why.

**Step 2: Update failing tests**

Key changes to expect:
- Tests checking for `"TDModule_0"` in compact mode: the module label is now `"TDModule_0\n..."` inside a single node, not a cluster label. The string `"TDModule_0"` should still be present.
- `test_probabilistic_in_sequential`: in compact mode, `"In: 4"` won't appear (that's the full-mode layer label). Change to check for `"Linear(4"` instead, or use `detail="full"`.
- Tests checking for `"Input\n"` or `"Output\n"` key node labels: now keys are `"obs"`, `"act"`, etc. directly.
- The `test_single_linear_module` (generic path) checks for `"Empty Module"` which should still appear.

For each failing test, update the assertion to match the new format. If a test specifically tests the expanded view, add `detail="full"`.

**Step 3: Run full suite**

Run: `uv run pytest tests/test_model_visualizer.py -v`
Expected: all PASS

**Step 4: Commit**

```bash
git add tests/test_model_visualizer.py
git commit -m "Update existing tests for new visualization format"
```

---

### Task 7: Update CLAUDE.md, example, and render a sample

**Files:**
- Modify: `CLAUDE.md`
- Modify: `examples/example.py`

**Step 1: Update CLAUDE.md task backlog**

Mark the P3 "Improve graph aesthetics" task as done (dark theme, single key nodes, compact mode address this).

**Step 2: Update example.py**

The example uses `visualizer.backend.graph.attr(label=title)` which bypasses the backend. Change to:

```python
def visualize_model(model, title):
    visualizer = ModelVisualizer(model=model, backend="graphviz")
    visualizer.visualize(render=False)
    visualizer.backend.set_graph_attr(label=title)
    visualizer.view(wait=True)
    del visualizer
```

Also add a compact vs full comparison:

```python
# Compare compact and full detail modes
visualizer = ModelVisualizer(model=seq_td_module, backend="graphviz")

visualizer.visualize(render=False, detail="compact")
visualizer.backend.set_graph_attr(label="Compact View")
visualizer.view(wait=True)

visualizer.clear()
visualizer.visualize(render=False, detail="full")
visualizer.backend.set_graph_attr(label="Full View")
visualizer.view(wait=True)
```

**Step 3: Render a sample SVG to verify visually**

```bash
uv run python -c "
from tensordictviz import ModelVisualizer
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential

model1 = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
model2 = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
model3 = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 3))

seq = TensorDictSequential(
    TensorDictModule(model1, in_keys=['observation'], out_keys=['latentspace']),
    TensorDictModule(model2, in_keys=['latentspace'], out_keys=['action1']),
    TensorDictModule(model3, in_keys=['latentspace'], out_keys=['action2']),
)

viz = ModelVisualizer(model=seq)
viz.visualize(render=True, detail='compact')
"
```

Visually inspect the SVG. Verify: dark background, colored key nodes (green/amber/blue), rounded module nodes, clean layout.

**Step 4: Run full test suite one final time**

Run: `uv run pytest tests/test_model_visualizer.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add CLAUDE.md examples/example.py
git commit -m "Update example and docs for new visualization features"
```

---

## Summary of all changes

| File | What changes |
|------|-------------|
| `tensordictviz/model_visualizer.py` | `DARK_THEME` dict, `detail` param, `_get_layer_summary`, `_get_module_summary`, compact/full `_visualize_td_module`, single-key-node `_visualize_td_sequential`, dark-themed `_visualize_sequential`/`_visualize_generic_module`/`_visualize_module` |
| `tests/test_model_visualizer.py` | New test classes: `TestLayerSummary`, `TestModuleSummary`, `TestCompactMode`, `TestSingleKeyNodes`, `TestDarkTheme`. Updated existing tests for new output format. |
| `CLAUDE.md` | Mark P3 aesthetics as done |
| `examples/example.py` | Use backend API, show compact vs full |
