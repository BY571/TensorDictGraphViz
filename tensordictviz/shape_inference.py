"""Shape inference for TensorDict-based and plain PyTorch models.

The visualizer uses this to annotate key nodes with their actual tensor
shape (last dim, plus full shape on demand). We build a fake input tensor
for each leaf input key by introspecting the first shape-bearing layer of
the consuming module, run a forward pass under ``no_grad``, and walk the
resulting TensorDict to capture every key's shape.

If anything goes wrong (unknown layer types, shape mismatch, side effects)
we emit a warning and return ``{}`` — the visualizer falls back to the
older static Linear-only heuristic so the user still gets a graph.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union


def _format_key(key) -> str:
    if isinstance(key, tuple):
        return ".".join(key)
    return str(key)


def _normalize_key(key) -> Tuple[str, ...]:
    """Convert any key form (str or tuple) to a canonical tuple."""
    if isinstance(key, tuple):
        return tuple(key)
    return (str(key),)


class ShapeInferer:
    """Capture real tensor shapes by running a fake forward pass.

    Parameters
    ----------
    model : the model to introspect.
    sample_input : optional TensorDict-like dict or TensorDict. If provided,
        used directly instead of building a fake input.
    batch_size : batch dim for fake tensors. Default 2 to keep BatchNorm
        operations valid (BN errors on batch=1 in train mode).
    """

    def __init__(self, model, sample_input=None, batch_size: int = 2):
        self.model = model
        self.sample_input = sample_input
        self.batch_size = batch_size

    # -- public ------------------------------------------------------------

    def infer(self) -> Dict[str, Any]:
        """Return ``{formatted_key: shape_tuple}`` for every key we could capture."""
        try:
            import torch
        except ImportError:
            return {}

        try:
            from tensordict.nn import TensorDictModule, TensorDictSequential
        except ImportError:
            TensorDictModule = TensorDictSequential = None  # type: ignore

        if TensorDictModule is not None and isinstance(
            self.model, (TensorDictModule, TensorDictSequential)
        ):
            return self._infer_td(torch)
        return {}

    # -- TensorDict path ---------------------------------------------------

    def _infer_td(self, torch) -> Dict[str, Any]:
        try:
            from tensordict import TensorDict
            from tensordict.nn import TensorDictSequential
        except ImportError:
            return {}

        try:
            td_in = self._prepare_input(torch, TensorDict, TensorDictSequential)
            if td_in is None:
                return {}

            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    td_out = self.model(td_in.clone(recurse=True))
            finally:
                if was_training:
                    self.model.train()

            shapes: Dict[str, Any] = {}
            for fk, shape in self._walk_shapes(td_in):
                shapes.setdefault(fk, shape)
            for fk, shape in self._walk_shapes(td_out):
                shapes[fk] = shape
            return shapes
        except Exception as e:  # noqa: BLE001 - we want to swallow all failures
            warnings.warn(
                f"tensordictviz: shape inference failed ({type(e).__name__}: {e}). "
                "Falling back to static heuristic.",
                stacklevel=2,
            )
            return {}

    def _prepare_input(self, torch, TensorDict, TensorDictSequential):
        """Use a user-provided sample, or synthesize one from layer introspection."""
        if self.sample_input is not None:
            if isinstance(self.sample_input, TensorDict):
                return self.sample_input
            if isinstance(self.sample_input, dict):
                # Infer batch dim from the first tensor in the user's dict so
                # we don't clobber their intended batch size.
                first = next(iter(self.sample_input.values()), None)
                batch = [first.shape[0]] if first is not None and first.ndim > 0 else []
                return TensorDict(dict(self.sample_input), batch_size=batch)
            raise TypeError(
                f"sample_input must be TensorDict or dict, got {type(self.sample_input).__name__}"
            )

        modules = self._list_modules()
        input_keys = self._leaf_input_keys(modules)

        fake: Dict[Any, Any] = {}
        for key in input_keys:
            tensor = self._fake_tensor_for_key(torch, key, modules)
            if tensor is None:
                return None
            fake[key] = tensor

        return TensorDict(fake, batch_size=[self.batch_size])

    def _list_modules(self) -> List[Any]:
        try:
            from tensordict.nn import TensorDictSequential
        except ImportError:
            return [self.model]
        if isinstance(self.model, TensorDictSequential):
            return list(self.model)
        return [self.model]

    def _leaf_input_keys(self, modules) -> List[Any]:
        """Keys consumed but never produced upstream — the model's true inputs."""
        produced: set = set()
        consumed: List[Any] = []
        seen_consumed: set = set()
        for m in modules:
            for k in m.in_keys:
                norm = _normalize_key(k)
                if norm not in produced and norm not in seen_consumed:
                    consumed.append(k)
                    seen_consumed.add(norm)
            for k in m.out_keys:
                produced.add(_normalize_key(k))
        # Filter out anything that ends up produced by a later module.
        return [k for k in consumed if _normalize_key(k) not in produced]

    def _fake_tensor_for_key(self, torch, key, modules):
        """Find the first module that consumes this key, dispatch on its layer type."""
        target = _normalize_key(key)
        for m in modules:
            in_norms = [_normalize_key(k) for k in m.in_keys]
            if target not in in_norms:
                continue
            inner = getattr(m, "module", None)
            if inner is None:
                continue
            tensor = self._fake_tensor_for_module(torch, inner)
            if tensor is not None:
                return tensor
        return None

    def _fake_tensor_for_module(self, torch, module):
        """Walk the module tree for the first layer we know how to feed."""
        import torch.nn as nn

        B = self.batch_size
        for child in module.modules():
            if isinstance(child, nn.Linear):
                return torch.randn(B, child.in_features)
            if isinstance(child, nn.Bilinear):
                return torch.randn(B, child.in1_features)
            if isinstance(child, nn.Conv1d):
                return torch.randn(B, child.in_channels, 32)
            if isinstance(child, nn.Conv2d):
                return torch.randn(B, child.in_channels, 32, 32)
            if isinstance(child, nn.Conv3d):
                return torch.randn(B, child.in_channels, 8, 8, 8)
            if isinstance(child, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                spatial = {
                    nn.ConvTranspose1d: (16,),
                    nn.ConvTranspose2d: (16, 16),
                    nn.ConvTranspose3d: (4, 4, 4),
                }[type(child)]
                return torch.randn(B, child.in_channels, *spatial)
            if isinstance(child, nn.Embedding):
                return torch.randint(
                    0, child.num_embeddings, (B, 16), dtype=torch.long
                )
            if isinstance(child, (nn.LSTM, nn.GRU, nn.RNN)):
                return torch.randn(B, 8, child.input_size)
            if isinstance(child, nn.MultiheadAttention):
                return torch.randn(B, 8, child.embed_dim)
            if isinstance(child, (nn.LSTMCell, nn.GRUCell, nn.RNNCell)):
                return torch.randn(B, child.input_size)
            if isinstance(child, nn.TransformerEncoderLayer):
                return torch.randn(B, 8, child.self_attn.embed_dim)
            if isinstance(child, nn.TransformerDecoderLayer):
                return torch.randn(B, 8, child.self_attn.embed_dim)
        return None

    def _walk_shapes(self, td):
        """Yield ``(formatted_key, shape_tuple)`` for every tensor leaf in ``td``."""
        try:
            items = td.items(include_nested=True, leaves_only=True)
        except TypeError:
            items = td.items()
        for key, value in items:
            shape = getattr(value, "shape", None)
            if shape is None:
                continue
            yield _format_key(key), tuple(shape)
