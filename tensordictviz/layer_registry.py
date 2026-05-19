"""Registry of layer label/summary formatters.

Each layer type maps to a function returning ``(long_label, short_summary)``.
The long label is multi-line and used in detail="full"; the short summary is
one-liner used in detail="compact" chains like
``Linear(4 → 8) → ReLU → Linear(8 → 2)``.

Users can register custom formatters via the ``register_layer`` decorator:

    @register_layer(MyLayer)
    def _fmt(layer):
        return ("MyLayer\\nfoo=...", f"MyLayer({layer.foo})")
"""

from typing import Callable, Dict, Tuple, Type

import torch.nn as nn


# A formatter takes the layer instance and returns (long_label, short_summary).
LayerFormatter = Callable[[nn.Module], Tuple[str, str]]

LAYER_REGISTRY: Dict[Type[nn.Module], LayerFormatter] = {}


def register_layer(layer_type: Type[nn.Module]):
    """Decorator: register a (long_label, short_summary) formatter for a layer."""

    def decorator(fn: LayerFormatter) -> LayerFormatter:
        LAYER_REGISTRY[layer_type] = fn
        return fn

    return decorator


def get_label(layer: nn.Module) -> str:
    """Multi-line label used in detail='full'."""
    fn = LAYER_REGISTRY.get(type(layer))
    if fn is not None:
        return fn(layer)[0]
    return type(layer).__name__


def get_summary(layer: nn.Module) -> str:
    """One-line summary used in detail='compact' chains."""
    fn = LAYER_REGISTRY.get(type(layer))
    if fn is not None:
        return fn(layer)[1]
    return type(layer).__name__


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------

_ARROW = "→"


def _kfmt(k):
    """Format a kernel/stride/padding tuple compactly: (3, 3) -> '3x3'."""
    if isinstance(k, int):
        return str(k)
    return "x".join(str(x) for x in k)


# --- Linear / bilinear ------------------------------------------------------


@register_layer(nn.Linear)
def _linear(l):
    return (
        f"Linear\nIn: {l.in_features}, Out: {l.out_features}",
        f"Linear({l.in_features}{_ARROW}{l.out_features})",
    )


@register_layer(nn.Bilinear)
def _bilinear(l):
    return (
        f"Bilinear\nIn1: {l.in1_features}, In2: {l.in2_features}, Out: {l.out_features}",
        f"Bilinear({l.in1_features},{l.in2_features}{_ARROW}{l.out_features})",
    )


# --- Conv -------------------------------------------------------------------


def _conv_fmt(name):
    def fmt(l):
        long = (
            f"{name}\nCh: {l.in_channels}{_ARROW}{l.out_channels}\n"
            f"Kernel: {_kfmt(l.kernel_size)}  Stride: {_kfmt(l.stride)}  "
            f"Pad: {_kfmt(l.padding)}"
        )
        short = f"{name}({l.in_channels}{_ARROW}{l.out_channels}, k={_kfmt(l.kernel_size)})"
        return long, short

    return fmt


for _layer_t, _name in [
    (nn.Conv1d, "Conv1d"),
    (nn.Conv2d, "Conv2d"),
    (nn.Conv3d, "Conv3d"),
    (nn.ConvTranspose1d, "ConvT1d"),
    (nn.ConvTranspose2d, "ConvT2d"),
    (nn.ConvTranspose3d, "ConvT3d"),
]:
    LAYER_REGISTRY[_layer_t] = _conv_fmt(_name)


# --- Normalisation ----------------------------------------------------------


def _batchnorm_fmt(name):
    def fmt(l):
        return (
            f"{name}\nFeatures: {l.num_features}",
            f"{name}({l.num_features})",
        )

    return fmt


for _layer_t, _name in [
    (nn.BatchNorm1d, "BN1d"),
    (nn.BatchNorm2d, "BN2d"),
    (nn.BatchNorm3d, "BN3d"),
    (nn.InstanceNorm1d, "IN1d"),
    (nn.InstanceNorm2d, "IN2d"),
    (nn.InstanceNorm3d, "IN3d"),
]:
    LAYER_REGISTRY[_layer_t] = _batchnorm_fmt(_name)


@register_layer(nn.LayerNorm)
def _layernorm(l):
    return (
        f"LayerNorm\nShape: {tuple(l.normalized_shape)}",
        f"LayerNorm({_kfmt(tuple(l.normalized_shape))})",
    )


@register_layer(nn.GroupNorm)
def _groupnorm(l):
    return (
        f"GroupNorm\nGroups: {l.num_groups}, Channels: {l.num_channels}",
        f"GroupNorm({l.num_groups},{l.num_channels})",
    )


# --- Pooling ----------------------------------------------------------------


def _pool_fmt(name):
    def fmt(l):
        return (
            f"{name}\nKernel: {_kfmt(l.kernel_size)}  Stride: {_kfmt(l.stride)}",
            f"{name}(k={_kfmt(l.kernel_size)})",
        )

    return fmt


def _adaptive_pool_fmt(name):
    def fmt(l):
        return (
            f"{name}\nOut: {_kfmt(l.output_size)}",
            f"{name}({_kfmt(l.output_size)})",
        )

    return fmt


for _layer_t, _name in [
    (nn.MaxPool1d, "MaxPool1d"),
    (nn.MaxPool2d, "MaxPool2d"),
    (nn.MaxPool3d, "MaxPool3d"),
    (nn.AvgPool1d, "AvgPool1d"),
    (nn.AvgPool2d, "AvgPool2d"),
    (nn.AvgPool3d, "AvgPool3d"),
]:
    LAYER_REGISTRY[_layer_t] = _pool_fmt(_name)

for _layer_t, _name in [
    (nn.AdaptiveMaxPool1d, "AdaptMaxPool1d"),
    (nn.AdaptiveMaxPool2d, "AdaptMaxPool2d"),
    (nn.AdaptiveMaxPool3d, "AdaptMaxPool3d"),
    (nn.AdaptiveAvgPool1d, "AdaptAvgPool1d"),
    (nn.AdaptiveAvgPool2d, "AdaptAvgPool2d"),
    (nn.AdaptiveAvgPool3d, "AdaptAvgPool3d"),
]:
    LAYER_REGISTRY[_layer_t] = _adaptive_pool_fmt(_name)


# --- Embedding --------------------------------------------------------------


@register_layer(nn.Embedding)
def _embedding(l):
    return (
        f"Embedding\nNum: {l.num_embeddings}, Dim: {l.embedding_dim}",
        f"Embed({l.num_embeddings},{l.embedding_dim})",
    )


@register_layer(nn.EmbeddingBag)
def _embedding_bag(l):
    return (
        f"EmbeddingBag\nNum: {l.num_embeddings}, Dim: {l.embedding_dim}",
        f"EmbedBag({l.num_embeddings},{l.embedding_dim})",
    )


# --- Recurrent --------------------------------------------------------------


def _rnn_fmt(name):
    def fmt(l):
        bidi = " bi" if getattr(l, "bidirectional", False) else ""
        return (
            f"{name}\nIn: {l.input_size}, Hidden: {l.hidden_size}\n"
            f"Layers: {l.num_layers}{bidi}",
            f"{name}({l.input_size}{_ARROW}{l.hidden_size}, n={l.num_layers}{bidi})",
        )

    return fmt


for _layer_t, _name in [
    (nn.LSTM, "LSTM"),
    (nn.GRU, "GRU"),
    (nn.RNN, "RNN"),
]:
    LAYER_REGISTRY[_layer_t] = _rnn_fmt(_name)


def _rnncell_fmt(name):
    def fmt(l):
        return (
            f"{name}\nIn: {l.input_size}, Hidden: {l.hidden_size}",
            f"{name}({l.input_size}{_ARROW}{l.hidden_size})",
        )

    return fmt


for _layer_t, _name in [
    (nn.LSTMCell, "LSTMCell"),
    (nn.GRUCell, "GRUCell"),
    (nn.RNNCell, "RNNCell"),
]:
    LAYER_REGISTRY[_layer_t] = _rnncell_fmt(_name)


# --- Attention / Transformer ------------------------------------------------


@register_layer(nn.MultiheadAttention)
def _mha(l):
    return (
        f"MultiheadAttention\nEmbed: {l.embed_dim}, Heads: {l.num_heads}",
        f"MHA(d={l.embed_dim}, h={l.num_heads})",
    )


@register_layer(nn.TransformerEncoderLayer)
def _txenc(l):
    d_model = l.self_attn.embed_dim
    heads = l.self_attn.num_heads
    return (
        f"TransformerEncoderLayer\nd_model: {d_model}, Heads: {heads}",
        f"TxEnc(d={d_model}, h={heads})",
    )


@register_layer(nn.TransformerDecoderLayer)
def _txdec(l):
    d_model = l.self_attn.embed_dim
    heads = l.self_attn.num_heads
    return (
        f"TransformerDecoderLayer\nd_model: {d_model}, Heads: {heads}",
        f"TxDec(d={d_model}, h={heads})",
    )


# --- Dropout ----------------------------------------------------------------


def _dropout_fmt(name):
    def fmt(l):
        return (f"{name}\nP: {l.p}", f"{name}({l.p})")

    return fmt


for _layer_t, _name in [
    (nn.Dropout, "Dropout"),
    (nn.Dropout1d, "Dropout1d"),
    (nn.Dropout2d, "Dropout2d"),
    (nn.Dropout3d, "Dropout3d"),
    (nn.AlphaDropout, "AlphaDropout"),
]:
    LAYER_REGISTRY[_layer_t] = _dropout_fmt(_name)


# --- Reshape ----------------------------------------------------------------


@register_layer(nn.Flatten)
def _flatten(l):
    return (
        f"Flatten\nDims: {l.start_dim}..{l.end_dim}",
        f"Flatten({l.start_dim},{l.end_dim})",
    )


@register_layer(nn.Unflatten)
def _unflatten(l):
    return (
        f"Unflatten\nDim: {l.dim}, Shape: {l.unflattened_size}",
        f"Unflatten({l.dim},{l.unflattened_size})",
    )


@register_layer(nn.Identity)
def _identity(_):
    return ("Identity", "Identity")


# --- Activations (short names, no useful attrs) -----------------------------


def _activation_fmt(name):
    def fmt(_):
        return (name, name)

    return fmt


for _layer_t, _name in [
    (nn.ReLU, "ReLU"),
    (nn.ReLU6, "ReLU6"),
    (nn.LeakyReLU, "LeakyReLU"),
    (nn.PReLU, "PReLU"),
    (nn.ELU, "ELU"),
    (nn.SELU, "SELU"),
    (nn.GELU, "GELU"),
    (nn.SiLU, "SiLU"),
    (nn.Mish, "Mish"),
    (nn.Sigmoid, "Sigmoid"),
    (nn.Tanh, "Tanh"),
    (nn.Hardtanh, "Hardtanh"),
    (nn.Softplus, "Softplus"),
    (nn.Softmax, "Softmax"),
    (nn.LogSoftmax, "LogSoftmax"),
]:
    LAYER_REGISTRY[_layer_t] = _activation_fmt(_name)


# --- Optional: tensordict / torchrl extras (registered if importable) -------

try:
    from tensordict.nn.distributions import NormalParamExtractor

    @register_layer(NormalParamExtractor)
    def _normal_param_extractor(_):
        return (
            "NormalParamExtractor\n(splits last dim → loc, scale)",
            "NormalParam(→loc,scale)",
        )

except ImportError:
    pass
