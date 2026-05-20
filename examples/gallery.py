"""Gallery of diverse architectures for visual testing.

Renders every example to ``/tmp/tviz_gallery/`` as PNG. Pass ``--open`` to
auto-open them. The gallery showcases shape inference, broad layer
coverage, themes, and TorchRL-aware rendering.
"""

import os
import sys

import torch
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    TensorDictModule,
    TensorDictSequential,
)
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Normal

from tensordictviz import visualize

OUTPUT_DIR = "/tmp/tviz_gallery"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def render(model, name, *, detail="compact", theme="light", sample_input=None):
    viz = visualize(
        model, render=False, detail=detail, theme=theme, sample_input=sample_input
    )
    path = f"{OUTPUT_DIR}/{name}"
    viz.backend.render(path, format="png")
    print(f"  {name}.png")
    return path + ".png"


# --- 1. Fan-out: 1 input → 2 outputs ---------------------------------------
encoder = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
head_a = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
head_b = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 3))

fan_out = TensorDictSequential(
    TensorDictModule(encoder, in_keys=["observation"], out_keys=["latent"]),
    TensorDictModule(head_a, in_keys=["latent"], out_keys=["action"]),
    TensorDictModule(head_b, in_keys=["latent"], out_keys=["value"]),
)

# --- 2. Fan-in: 2 inputs → 1 output -----------------------------------------
embed_img = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
embed_text = nn.Sequential(nn.Linear(128, 32), nn.ReLU())


class Fuse(nn.Module):
    """Concatenate the image and text embeddings, then project to an action."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 16)
        self.act = nn.ReLU()
        self.head = nn.Linear(16, 4)

    def forward(self, image_emb, text_emb):
        x = torch.cat([image_emb, text_emb], dim=-1)
        return self.head(self.act(self.proj(x)))


fan_in = TensorDictSequential(
    TensorDictModule(embed_img, in_keys=["image"], out_keys=["image_emb"]),
    TensorDictModule(embed_text, in_keys=["text"], out_keys=["text_emb"]),
    TensorDictModule(Fuse(), in_keys=["image_emb", "text_emb"], out_keys=["action"]),
)

# --- 3. CNN-based feature extractor + MLP head ------------------------------
cnn = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
)
mlp = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))

cnn_model = TensorDictSequential(
    TensorDictModule(cnn, in_keys=["pixels"], out_keys=["features"]),
    TensorDictModule(mlp, in_keys=["features"], out_keys=["action"]),
)

# --- 4. Deep chain ----------------------------------------------------------
deep_chain = TensorDictSequential(
    TensorDictModule(
        nn.Sequential(nn.Linear(10, 64), nn.ReLU()),
        in_keys=["obs"],
        out_keys=["h1"],
    ),
    TensorDictModule(
        nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1)),
        in_keys=["h1"],
        out_keys=["h2"],
    ),
    TensorDictModule(
        nn.Sequential(nn.Linear(64, 32), nn.ReLU()),
        in_keys=["h2"],
        out_keys=["h3"],
    ),
    TensorDictModule(
        nn.Sequential(nn.Linear(32, 4)),
        in_keys=["h3"],
        out_keys=["action"],
    ),
)

# --- 5. Diamond -------------------------------------------------------------
enc = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
left = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
right = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
merge = nn.Sequential(nn.Linear(8, 4))

diamond = TensorDictSequential(
    TensorDictModule(enc, in_keys=["state"], out_keys=["hidden"]),
    TensorDictModule(left, in_keys=["hidden"], out_keys=["left_out"]),
    TensorDictModule(right, in_keys=["hidden"], out_keys=["right_out"]),
    TensorDictModule(merge, in_keys=["left_out"], out_keys=["action"]),
)

# --- 6. Single TDModule -----------------------------------------------------
single_td = TensorDictModule(
    nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3)),
    in_keys=["observation"],
    out_keys=["action"],
)

# --- 7. Nested keys ---------------------------------------------------------
nested_keys = TensorDictSequential(
    TensorDictModule(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "hidden")],
    ),
    TensorDictModule(
        nn.Sequential(nn.Linear(8, 2)),
        in_keys=[("agents", "hidden")],
        out_keys=[("agents", "action")],
    ),
)

# --- 8. Plain nn.Sequential -------------------------------------------------
plain_seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

# --- 9. Probabilistic policy (Normal) ---------------------------------------
prob_net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 4), NormalParamExtractor())
param_mod = TensorDictModule(prob_net, in_keys=["observation"], out_keys=["loc", "scale"])
prob_mod = ProbabilisticTensorDictModule(
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=Normal,
)
prob_actor = TensorDictSequential(param_mod, prob_mod)

# --- 10. Token embedding + MLP ----------------------------------------------
embed_model = TensorDictSequential(
    TensorDictModule(
        nn.Sequential(nn.Embedding(100, 16)),
        in_keys=["tokens"],
        out_keys=["embeds"],
    ),
    TensorDictModule(
        nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4)),
        in_keys=["embeds"],
        out_keys=["logits"],
    ),
)

# --- 11. Mini "RNN-ish" module with a self-loop state key -------------------
class StatefulCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, x, h):
        h_next = self.cell(x, h)
        return h_next, h_next


stateful = TensorDictModule(
    StatefulCell(8, 16),
    in_keys=["obs", "hidden"],
    out_keys=["features", "hidden"],
)

# --- 12. Attention head -----------------------------------------------------
attn_net = nn.Sequential(nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True))
# MultiheadAttention takes (query, key, value) so a plain TDModule won't directly
# wrap it; show as a generic module so the registry's MHA label shows up.
attn_demo = nn.Sequential(
    nn.Linear(16, 32),
    nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True),
)

# --- 13. LSTM (registry coverage) -------------------------------------------
lstm_seq = nn.Sequential(nn.LSTM(input_size=8, hidden_size=16, num_layers=2))


# Render all
print(f"Rendering gallery to {OUTPUT_DIR}/\n")

files = []
files.append(render(fan_out, "01_fan_out"))
files.append(render(fan_in, "02_fan_in"))
files.append(render(cnn_model, "03_cnn"))
files.append(render(deep_chain, "04_deep_chain"))
files.append(render(diamond, "05_diamond"))
files.append(render(single_td, "06_single_td"))
files.append(render(nested_keys, "07_nested_keys"))
files.append(render(plain_seq, "08_plain_sequential"))
files.append(render(fan_out, "09_fan_out_full", detail="full"))
files.append(render(prob_actor, "10_probabilistic_actor"))
files.append(render(embed_model, "11_embedding"))
files.append(render(stateful, "12_stateful"))
files.append(render(attn_demo, "13_attention"))
files.append(render(lstm_seq, "14_lstm"))

# Theme showcase on the fan_out model.
files.append(render(fan_out, "15_fan_out_dark", theme="dark"))
files.append(render(fan_out, "16_fan_out_print", theme="print"))

print(f"\nDone! {len(files)} images in {OUTPUT_DIR}/")

if "--open" in sys.argv:
    for f in files:
        os.system(f"xdg-open {f} &")
