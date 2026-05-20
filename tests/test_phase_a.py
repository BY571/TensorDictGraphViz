"""Tests for Phase A improvements: shape inference, layer registry,
themes, top-level visualize(), and TorchRL-aware rendering."""

import pytest
import torch
import torch.nn as nn
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    TensorDictModule,
    TensorDictSequential,
)
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal

from tensordictviz import LAYER_REGISTRY, ModelVisualizer, THEMES, register_layer, visualize
from tensordictviz.layer_registry import get_label, get_summary
from tensordictviz.model_visualizer import _shape_label
from tensordictviz.shape_inference import ShapeInferer
from tensordictviz.themes import resolve_theme


def _source(viz):
    return viz.backend.graph.source


# ---------------------------------------------------------------------------
# Top-level visualize()
# ---------------------------------------------------------------------------


class TestTopLevelVisualize:
    def test_returns_visualizer(self):
        model = nn.Sequential(nn.Linear(4, 2))
        viz = visualize(model)
        assert isinstance(viz, ModelVisualizer)

    def test_does_not_render_by_default(self, tmp_path):
        """Top-level visualize() should default render=False so notebooks don't write files."""
        # If it tried to render, it would print "Graph saved as ..." and create a file.
        model = nn.Sequential(nn.Linear(4, 2))
        viz = visualize(model)
        assert "Linear" in _source(viz)

    def test_kwargs_passed_through(self):
        model = TensorDictModule(nn.Linear(4, 2), in_keys=["obs"], out_keys=["act"])
        viz = visualize(model, detail="full", theme="dark")
        assert viz.theme["bg"] == "#1e1e2e"
        assert "cluster_TDModule_0" in _source(viz)


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


class TestThemes:
    def test_six_presets_available(self):
        assert set(THEMES.keys()) == {
            "light", "dark", "print", "blueprint", "editorial", "vivid",
        }

    def test_resolve_light_default(self):
        t = resolve_theme(None)
        assert t["bg"] == "white"

    def test_resolve_named(self):
        assert resolve_theme("dark")["bg"] == "#1e1e2e"
        assert resolve_theme("print")["bg"] == "white"
        assert resolve_theme("blueprint")["bg"] == "#0d2137"

    def test_resolve_dict_overrides_light(self):
        t = resolve_theme({"bg": "#abcdef"})
        assert t["bg"] == "#abcdef"
        assert t["module_fill"] == THEMES["light"]["module_fill"]

    def test_every_theme_resolves_to_complete_schema(self):
        """Every preset, merged onto LIGHT, must define all schema keys."""
        required = set(resolve_theme("light").keys())
        for name in THEMES:
            resolved = resolve_theme(name)
            assert set(resolved.keys()) == required, f"{name} missing keys"

    def test_structural_keys_present(self):
        t = resolve_theme("blueprint")
        for key in ("rankdir", "splines", "key_shape", "module_shape",
                    "module_rounded", "nodesep", "ranksep"):
            assert key in t

    def test_blueprint_is_structurally_distinct(self):
        t = resolve_theme("blueprint")
        assert t["key_shape"] == "box"
        assert t["module_rounded"] is False
        assert t["font"] == "Courier New"

    def test_partial_theme_inherits_structure(self):
        """A dict override that only sets a color keeps LIGHT's structure."""
        t = resolve_theme({"bg": "#000000"})
        assert t["key_shape"] == "ellipse"
        assert t["module_rounded"] is True

    def test_module_rounded_override_changes_style(self):
        model = nn.Sequential(nn.Linear(4, 2))
        sharp = visualize(model, theme={"module_rounded": False})
        assert "filled,rounded" not in _source(sharp)
        rounded = visualize(model, theme={"module_rounded": True})
        assert "rounded" in _source(rounded)

    def test_unknown_theme_raises(self):
        with pytest.raises(ValueError, match="Unknown theme"):
            resolve_theme("solarized")

    def test_theme_applied_to_graph(self):
        model = nn.Sequential(nn.Linear(4, 2))
        viz = visualize(model, theme="dark")
        assert "#1e1e2e" in _source(viz)

    def test_resolve_returns_copy(self):
        t = resolve_theme("light")
        t["bg"] = "tampered"
        assert THEMES["light"]["bg"] == "white"


# ---------------------------------------------------------------------------
# Layer registry
# ---------------------------------------------------------------------------


class TestLayerRegistry:
    def test_linear_in_registry(self):
        assert nn.Linear in LAYER_REGISTRY

    def test_conv_layers_in_registry(self):
        for t in (nn.Conv1d, nn.Conv2d, nn.Conv3d):
            assert t in LAYER_REGISTRY

    def test_rnn_layers_in_registry(self):
        for t in (nn.LSTM, nn.GRU, nn.RNN):
            assert t in LAYER_REGISTRY

    def test_norm_layers_in_registry(self):
        for t in (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm):
            assert t in LAYER_REGISTRY

    def test_pool_layers_in_registry(self):
        for t in (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d):
            assert t in LAYER_REGISTRY

    def test_attention_in_registry(self):
        assert nn.MultiheadAttention in LAYER_REGISTRY

    def test_lstm_summary_format(self):
        layer = nn.LSTM(16, 32, num_layers=2)
        assert "LSTM" in get_summary(layer)
        assert "16" in get_summary(layer)
        assert "32" in get_summary(layer)

    def test_mha_summary_format(self):
        layer = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        s = get_summary(layer)
        assert "MHA" in s and "64" in s and "h=4" in s

    def test_unknown_layer_falls_back_to_class_name(self):
        class WeirdLayer(nn.Module):
            pass

        assert get_summary(WeirdLayer()) == "WeirdLayer"
        assert get_label(WeirdLayer()) == "WeirdLayer"

    def test_register_layer_decorator(self):
        class CustomLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = 42

        @register_layer(CustomLayer)
        def _fmt(layer):
            return (f"Custom\nfoo: {layer.foo}", f"Custom({layer.foo})")

        try:
            assert get_summary(CustomLayer()) == "Custom(42)"
            assert "foo: 42" in get_label(CustomLayer())
        finally:
            del LAYER_REGISTRY[CustomLayer]

    def test_normal_param_extractor_registered(self):
        layer = NormalParamExtractor()
        assert "NormalParam" in get_summary(layer) or "NormalParam" in get_label(layer)


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------


class TestShapeInference:
    def test_linear_chain(self):
        model = TensorDictSequential(
            TensorDictModule(nn.Linear(4, 8), in_keys=["obs"], out_keys=["hidden"]),
            TensorDictModule(nn.Linear(8, 2), in_keys=["hidden"], out_keys=["action"]),
        )
        shapes = ShapeInferer(model).infer()
        assert shapes["obs"] == (2, 4)
        assert shapes["hidden"] == (2, 8)
        assert shapes["action"] == (2, 2)

    def test_conv2d(self):
        cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        model = TensorDictModule(cnn, in_keys=["pixels"], out_keys=["features"])
        shapes = ShapeInferer(model).infer()
        # pixels: (B, 3, 32, 32), features: (B, 16, 32, 32)
        assert shapes["pixels"][1] == 3
        assert shapes["features"][1] == 16

    def test_embedding(self):
        net = nn.Sequential(nn.Embedding(50, 12))
        model = TensorDictModule(net, in_keys=["tokens"], out_keys=["embeds"])
        shapes = ShapeInferer(model).infer()
        # Embedding embeds shape: (B, seq_len=16, 12)
        assert shapes["embeds"][-1] == 12

    def test_user_sample_input_dict(self):
        model = TensorDictModule(nn.Linear(7, 3), in_keys=["x"], out_keys=["y"])
        shapes = ShapeInferer(model, sample_input={"x": torch.randn(5, 7)}).infer()
        assert shapes["x"] == (5, 7)
        assert shapes["y"] == (5, 3)

    def test_unknown_input_returns_empty(self):
        class WeirdLayer(nn.Module):
            def forward(self, x):
                return x

        model = TensorDictModule(WeirdLayer(), in_keys=["obs"], out_keys=["out"])
        # No shape-bearing layer => can't construct fake input => empty result.
        shapes = ShapeInferer(model).infer()
        assert shapes == {}

    def test_forward_failure_swallowed_with_warning(self):
        class FailingLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 2)

            def forward(self, x):
                raise RuntimeError("boom")

        model = TensorDictModule(FailingLayer(), in_keys=["obs"], out_keys=["out"])
        with pytest.warns(UserWarning, match="shape inference failed"):
            shapes = ShapeInferer(model).infer()
        assert shapes == {}

    def test_sequential_per_layer_shapes(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        shapes = ShapeInferer(model).infer()
        assert shapes["input"] == (2, 4)
        assert shapes["layer_0"] == (2, 8)  # after Linear(4, 8)
        assert shapes["layer_1"] == (2, 8)  # after ReLU
        assert shapes["layer_2"] == (2, 2)  # after Linear(8, 2)

    def test_generic_module_input_output_shapes(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(6, 3)

            def forward(self, x):
                return self.fc(x)

        shapes = ShapeInferer(M()).infer()
        assert shapes["input"] == (2, 6)
        assert shapes["output"] == (2, 3)

    def test_sequential_tensor_sample_input(self):
        model = nn.Sequential(nn.Linear(7, 3))
        shapes = ShapeInferer(model, sample_input=torch.randn(5, 7)).infer()
        assert shapes["input"] == (5, 7)
        assert shapes["layer_0"] == (5, 3)

    def test_sequential_edge_labels_in_source(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        viz = visualize(model)
        src = _source(viz)
        # Running shape flows onto each edge as a label (batch dim dropped).
        assert 'label="[4]"' in src  # input -> layer_0
        assert 'label="[8]"' in src  # layer_0 -> layer_1
        assert 'label="[2]"' in src  # layer_2 -> output


# ---------------------------------------------------------------------------
# _shape_label
# ---------------------------------------------------------------------------


class TestShapeLabel:
    def test_drops_batch_dim(self):
        assert _shape_label((2, 4)) == "[4]"
        assert _shape_label((2, 3, 32, 32)) == "[3, 32, 32]"

    def test_none_is_empty(self):
        assert _shape_label(None) == ""

    def test_single_dim_kept(self):
        # batch-only shape — show what we have rather than nothing.
        assert _shape_label((10,)) == "[10]"


# ---------------------------------------------------------------------------
# Key node shape in label
# ---------------------------------------------------------------------------


class TestKeyShapeInLabel:
    def test_shape_appears_on_key_nodes(self):
        model = TensorDictSequential(
            TensorDictModule(nn.Linear(4, 8), in_keys=["obs"], out_keys=["hidden"]),
            TensorDictModule(nn.Linear(8, 2), in_keys=["hidden"], out_keys=["action"]),
        )
        viz = visualize(model)
        src = _source(viz)
        # The visualizer drops the batch dim and renders [last_dims]
        assert "obs [4]" in src
        assert "hidden [8]" in src
        assert "action [2]" in src


# ---------------------------------------------------------------------------
# TorchRL-aware: ProbabilisticTensorDictModule
# ---------------------------------------------------------------------------


class TestProbabilisticRendering:
    def _actor(self):
        net = nn.Sequential(nn.Linear(4, 8), NormalParamExtractor())
        param_mod = TensorDictModule(net, in_keys=["obs"], out_keys=["loc", "scale"])
        prob_mod = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=Normal,
        )
        return TensorDictSequential(param_mod, prob_mod)

    def test_distribution_name_in_label(self):
        viz = visualize(self._actor())
        src = _source(viz)
        assert "Probabilistic" in src
        assert "Normal" in src

    def test_distinct_fill_in_compact(self):
        """ProbabilisticTDM uses the theme's probabilistic_fill color."""
        viz = visualize(self._actor())
        src = _source(viz)
        prob_fill = THEMES["light"]["probabilistic_fill"]
        assert prob_fill in src

    def test_full_mode_distinct_cluster(self):
        viz = visualize(self._actor(), detail="full")
        src = _source(viz)
        assert "ProbabilisticTensorDictModule_1" in src or "Probabilistic" in src


# ---------------------------------------------------------------------------
# Subclasses of TensorDictSequential use their real class name
# ---------------------------------------------------------------------------


class TestSequentialSubclassLabel:
    def test_subclass_name_in_cluster_label(self):
        class MyActor(TensorDictSequential):
            pass

        m = MyActor(
            TensorDictModule(nn.Linear(4, 2), in_keys=["obs"], out_keys=["act"])
        )
        viz = visualize(m)
        assert "MyActor" in _source(viz)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


class TestLegend:
    def test_legend_present_by_default(self):
        model = TensorDictModule(nn.Linear(4, 2), in_keys=["obs"], out_keys=["act"])
        viz = visualize(model)
        src = _source(viz)
        assert "cluster_legend" in src
        assert "Input key" in src
        assert "Output key" in src

    def test_legend_can_be_disabled(self):
        model = TensorDictModule(nn.Linear(4, 2), in_keys=["obs"], out_keys=["act"])
        viz = visualize(model, show_legend=False)
        assert "cluster_legend" not in _source(viz)

    def test_state_legend_only_when_state_keys_present(self):
        """No state keys => no 'Recurrent state' entry in legend."""
        model = TensorDictModule(nn.Linear(4, 2), in_keys=["obs"], out_keys=["act"])
        viz = visualize(model)
        # The "Recurrent state" label appears only when a state key was detected.
        assert "Recurrent state" not in _source(viz)


# ---------------------------------------------------------------------------
# Recurrent state-key detection
# ---------------------------------------------------------------------------


class TestStateKeys:
    def test_self_loop_key_classified_as_state(self):
        """A key in both in_keys AND out_keys of the same module is a state key."""

        class FakeRecurrent(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

            def forward(self, x, h):
                return self.lin(x), h

        td = TensorDictModule(
            FakeRecurrent(),
            in_keys=["obs", "hidden"],
            out_keys=["features", "hidden"],
        )
        # Manually compose a sequential so produced/consumed includes the same module twice
        viz = visualize(td, show_legend=True)
        src = _source(viz)
        # The state color should appear (key_state)
        assert THEMES["light"]["key_state"] in src
        assert "Recurrent state" in src


# ---------------------------------------------------------------------------
# Save + _repr_svg_
# ---------------------------------------------------------------------------


class TestSaveAndRepr:
    def test_save_writes_file(self, tmp_path):
        model = nn.Sequential(nn.Linear(4, 2))
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)
        path = str(tmp_path / "out")
        viz.save(path, format="svg")
        assert (tmp_path / "out.svg").exists()

    def test_repr_svg_returns_string(self):
        model = nn.Sequential(nn.Linear(4, 2))
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)
        svg = viz._repr_svg_()
        assert isinstance(svg, str)
        assert svg.startswith("<?xml") or svg.startswith("<svg")
