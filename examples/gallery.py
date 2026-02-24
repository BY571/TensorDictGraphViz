"""Gallery of diverse architectures for visual testing."""

from tensordictviz import ModelVisualizer
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
import sys
import os

OUTPUT_DIR = "/tmp/tviz_gallery"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def render(model, name, detail="compact"):
    viz = ModelVisualizer(model=model, backend="graphviz")
    viz.visualize(render=False, detail=detail)
    path = f"{OUTPUT_DIR}/{name}"
    viz.backend.render(path, format="png")
    print(f"  {name}.png")
    return path + ".png"


# --- 1. Fan-out: 1 input → 2 outputs ---
encoder = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
head_a = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
head_b = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 3))

fan_out = TensorDictSequential(
    TensorDictModule(encoder, in_keys=["observation"], out_keys=["latent"]),
    TensorDictModule(head_a, in_keys=["latent"], out_keys=["action"]),
    TensorDictModule(head_b, in_keys=["latent"], out_keys=["value"]),
)

# --- 2. Fan-in: 2 inputs → 1 output ---
embed_img = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
embed_text = nn.Sequential(nn.Linear(128, 32), nn.ReLU())
fuse = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))

fan_in = TensorDictSequential(
    TensorDictModule(embed_img, in_keys=["image"], out_keys=["image_emb"]),
    TensorDictModule(embed_text, in_keys=["text"], out_keys=["text_emb"]),
    TensorDictModule(fuse, in_keys=["image_emb"], out_keys=["action"]),
)

# --- 3. CNN-based ---
cnn = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
)
mlp = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))

cnn_model = TensorDictSequential(
    TensorDictModule(cnn, in_keys=["pixels"], out_keys=["features"]),
    TensorDictModule(mlp, in_keys=["features"], out_keys=["action"]),
)

# --- 4. Deep chain: 4 modules in series ---
deep_chain = TensorDictSequential(
    TensorDictModule(
        nn.Sequential(nn.Linear(10, 64), nn.ReLU()),
        in_keys=["obs"], out_keys=["h1"],
    ),
    TensorDictModule(
        nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1)),
        in_keys=["h1"], out_keys=["h2"],
    ),
    TensorDictModule(
        nn.Sequential(nn.Linear(64, 32), nn.ReLU()),
        in_keys=["h2"], out_keys=["h3"],
    ),
    TensorDictModule(
        nn.Sequential(nn.Linear(32, 4)),
        in_keys=["h3"], out_keys=["action"],
    ),
)

# --- 5. Diamond: shared intermediate consumed by two, both feed into final ---
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

# --- 6. Single TDModule ---
single_td = TensorDictModule(
    nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3)),
    in_keys=["observation"],
    out_keys=["action"],
)

# --- 7. Nested keys ---
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

# --- 8. Plain nn.Sequential ---
plain_seq = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)
)


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

# Also render fan_out in full detail for comparison
files.append(render(fan_out, "09_fan_out_full", detail="full"))

print(f"\nDone! {len(files)} images in {OUTPUT_DIR}/")

if "--open" in sys.argv:
    for f in files:
        os.system(f"xdg-open {f} &")
