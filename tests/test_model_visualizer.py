import pytest
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn import ProbabilisticTensorDictModule
from torch.distributions import Normal

from tensordictviz import ModelVisualizer
from tensordictviz.model_visualizer import _format_key, _format_keys, _join_keys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source(viz):
    """Return the DOT source string from a visualizer's backend."""
    return viz.backend.graph.source


# ---------------------------------------------------------------------------
# _format_key / _format_keys / _join_keys
# ---------------------------------------------------------------------------

class TestFormatKey:
    def test_string_key(self):
        assert _format_key("observation") == "observation"

    def test_tuple_key(self):
        assert _format_key(("agents", "observation")) == "agents.observation"

    def test_deeply_nested_key(self):
        assert _format_key(("a", "b", "c")) == "a.b.c"


class TestFormatKeys:
    def test_all_string_keys(self):
        assert _format_keys(["obs", "action"]) == "obs, action"

    def test_all_tuple_keys(self):
        assert _format_keys([("a", "obs"), ("a", "act")]) == "a.obs, a.act"

    def test_mixed_keys(self):
        assert _format_keys(["obs", ("agents", "act")]) == "obs, agents.act"

    def test_empty(self):
        assert _format_keys([]) == ""


class TestJoinKeys:
    def test_default_sep(self):
        assert _join_keys(["obs", "action"]) == "obs_action"

    def test_tuple_keys(self):
        assert _join_keys([("a", "obs")]) == "a.obs"

    def test_custom_sep(self):
        assert _join_keys(["x", "y"], sep="-") == "x-y"


# ---------------------------------------------------------------------------
# ModelVisualizer construction & error paths
# ---------------------------------------------------------------------------

class TestModelVisualizerInit:
    def test_default_backend_is_graphviz(self):
        viz = ModelVisualizer()
        assert viz.backend is not None
        assert hasattr(viz.backend, "graph")  # GraphvizBackend has .graph

    def test_unsupported_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            ModelVisualizer(backend="matplotlib")

    def test_excalidraw_backend_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ModelVisualizer(backend="excalidraw")

    def test_no_model_raises_on_visualize(self):
        viz = ModelVisualizer()
        with pytest.raises(ValueError, match="No model provided"):
            viz.visualize(render=False)

    def test_unsupported_model_type_raises(self):
        viz = ModelVisualizer()
        with pytest.raises(TypeError, match="Unsupported model type"):
            viz.visualize(render=False, model="not a module")


# ---------------------------------------------------------------------------
# Path 1: nn.Sequential
# ---------------------------------------------------------------------------

class TestVisualizeSequential:
    def test_simple_sequential(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "Input" in src
        assert "Output" in src
        # Linear layers should show dimensions
        assert "In: 4" in src
        assert "Out: 8" in src
        assert "ReLU" in src

    def test_single_layer_sequential(self):
        model = nn.Sequential(nn.Linear(3, 1))
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "Input" in src
        assert "Output" in src
        assert "In: 3" in src

    def test_conv_layers(self):
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "Conv2D" in src
        assert "In: 1" in src
        assert "Out: 32" in src

    def test_dropout_layer(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "Dropout" in src
        assert "P: 0.5" in src


# ---------------------------------------------------------------------------
# Path 2: TensorDictModule / TensorDictSequential
# ---------------------------------------------------------------------------

class TestVisualizeTensorDict:
    def test_single_td_module(self):
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["observation"], out_keys=["action"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "observation" in src
        assert "action" in src
        assert "TDModule_0" in src

    def test_td_sequential(self):
        m1 = nn.Sequential(nn.Linear(4, 8))
        m2 = nn.Sequential(nn.Linear(8, 2))
        model = TensorDictSequential(
            TensorDictModule(m1, in_keys=["obs"], out_keys=["hidden"]),
            TensorDictModule(m2, in_keys=["hidden"], out_keys=["action"]),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "obs" in src
        assert "hidden" in src
        assert "action" in src
        assert "TDModule_0" in src
        assert "TDModule_1" in src

    def test_td_sequential_shared_input_key(self):
        """Two modules reading from the same input key."""
        m1 = nn.Sequential(nn.Linear(4, 2))
        m2 = nn.Sequential(nn.Linear(4, 3))
        model = TensorDictSequential(
            TensorDictModule(m1, in_keys=["latent"], out_keys=["a"]),
            TensorDictModule(m2, in_keys=["latent"], out_keys=["b"]),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        # Both modules share the "latent" input
        assert "latent" in src
        assert "TDModule_0" in src
        assert "TDModule_1" in src

    def test_model_passed_via_visualize(self):
        """Test passing model through visualize() instead of constructor."""
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer()
        viz.visualize(render=False, model=model)

        src = _source(viz)
        assert "obs" in src
        assert "act" in src


# ---------------------------------------------------------------------------
# Path 3: Generic nn.Module
# ---------------------------------------------------------------------------

class TestVisualizeGenericModule:
    def test_custom_module(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(8, 2)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        model = MyModule()
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "In: 4" in src
        assert "Out: 8" in src
        assert "In: 8" in src
        assert "Out: 2" in src

    def test_single_linear_module(self):
        """A plain nn.Linear has no children, so gets the 'Empty Module' dummy."""
        model = nn.Linear(3, 1)
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "Empty Module" in src


# ---------------------------------------------------------------------------
# P0 edge cases: nested keys
# ---------------------------------------------------------------------------

class TestNestedKeys:
    def test_tuple_keys_on_td_module(self):
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(
            net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "action")],
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "agents.observation" in src
        assert "agents.action" in src

    def test_mixed_string_and_tuple_keys(self):
        m1 = nn.Sequential(nn.Linear(4, 3))
        m2 = nn.Sequential(nn.Linear(3, 2))
        model = TensorDictSequential(
            TensorDictModule(m1, in_keys=[("agents", "obs")], out_keys=["latent"]),
            TensorDictModule(m2, in_keys=["latent"], out_keys=[("agents", "act")]),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "agents.obs" in src
        assert "latent" in src
        assert "agents.act" in src

    def test_deeply_nested_keys(self):
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(
            net,
            in_keys=[("env", "agents", "obs")],
            out_keys=[("env", "agents", "act")],
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "env.agents.obs" in src
        assert "env.agents.act" in src


# ---------------------------------------------------------------------------
# P0 edge case: ProbabilisticTensorDictModule
# ---------------------------------------------------------------------------

class TestProbabilisticModule:
    def test_probabilistic_in_sequential(self):
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
        viz.visualize(render=False)

        src = _source(viz)
        # param_mod should show its internal layers
        assert "In: 4" in src
        # prob_mod should show its class name (no .module)
        assert "ProbabilisticTensorDictModule" in src
        assert "loc" in src
        assert "scale" in src
        assert "action" in src


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_graph(self):
        model = nn.Sequential(nn.Linear(4, 2))
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)
        assert "Linear" in _source(viz)

        viz.clear()
        # After clear the graph body should be empty
        assert "Linear" not in _source(viz)


# ---------------------------------------------------------------------------
# Backend subgraph stack integrity
# ---------------------------------------------------------------------------

class TestSubgraphStack:
    def test_stack_returns_to_root_after_visualize(self):
        """After visualize(), the backend stack should be back at the root graph."""
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        # Stack should contain only the root graph
        assert len(viz.backend._stack) == 1
        assert viz.backend._stack[0] is viz.backend.graph

    def test_stack_returns_to_root_after_generic_module(self):
        """Stack integrity for the generic module path."""
        model = nn.Linear(3, 1)
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        assert len(viz.backend._stack) == 1
        assert viz.backend._stack[0] is viz.backend.graph

    def test_clear_resets_stack(self):
        """clear() should reset the stack to [root graph]."""
        viz = ModelVisualizer(model=nn.Sequential(nn.Linear(4, 2)))
        viz.visualize(render=False)
        viz.clear()

        assert len(viz.backend._stack) == 1
        assert viz.backend._stack[0] is viz.backend.graph


# ---------------------------------------------------------------------------
# _get_layer_summary / _get_module_summary
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Compact mode vs full mode
# ---------------------------------------------------------------------------

class TestCompactMode:
    def test_compact_td_module_is_single_node(self):
        """In compact mode, each TDModule is one node, not a cluster."""
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False, detail="compact")

        src = _source(viz)
        assert "cluster_TDModule_0" not in src
        assert "Linear(4" in src

    def test_compact_shows_layer_chain(self):
        """Compact mode shows the layer summary chain."""
        net = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False, detail="compact")

        src = _source(viz)
        assert "\u2192" in src

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


# ---------------------------------------------------------------------------
# Single key nodes
# ---------------------------------------------------------------------------

class TestSingleKeyNodes:
    def test_intermediate_key_is_single_node(self):
        m1 = nn.Sequential(nn.Linear(4, 8))
        m2 = nn.Sequential(nn.Linear(8, 2))
        model = TensorDictSequential(
            TensorDictModule(m1, in_keys=["obs"], out_keys=["hidden"]),
            TensorDictModule(m2, in_keys=["hidden"], out_keys=["action"]),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "key_hidden" in src
        assert "input_hidden" not in src
        assert "output_hidden" not in src

    def test_input_only_key_colored_green(self):
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "key_obs" in src
        assert "#2d6a4f" in src

    def test_output_only_key_colored_blue(self):
        net = nn.Sequential(nn.Linear(4, 2))
        model = TensorDictModule(net, in_keys=["obs"], out_keys=["act"])
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "key_act" in src
        assert "#1a5276" in src

    def test_intermediate_key_colored_amber(self):
        m1 = nn.Sequential(nn.Linear(4, 8))
        m2 = nn.Sequential(nn.Linear(8, 2))
        model = TensorDictSequential(
            TensorDictModule(m1, in_keys=["obs"], out_keys=["hidden"]),
            TensorDictModule(m2, in_keys=["hidden"], out_keys=["action"]),
        )
        viz = ModelVisualizer(model=model)
        viz.visualize(render=False)

        src = _source(viz)
        assert "#b8860b" in src

    def test_shared_input_key_connects_to_both_modules(self):
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
