"""Quick demo of the tensordictviz top-level API.

Renders the same model in compact/full modes and across themes, opening
each in the system viewer.
"""

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from tensordictviz import visualize

encoder = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
head_a = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
head_b = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 3))

policy = TensorDictSequential(
    TensorDictModule(encoder, in_keys=["observation"], out_keys=["latent"]),
    TensorDictModule(head_a, in_keys=["latent"], out_keys=["action"]),
    TensorDictModule(head_b, in_keys=["latent"], out_keys=["value"]),
)

single_td = TensorDictModule(
    nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3)),
    in_keys=["observation"],
    out_keys=["action"],
)

plain = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))


def show(model, title: str, **kwargs):
    viz = visualize(model, render=False, **kwargs)
    viz.backend.set_graph_attr(label=title)
    viz.view(wait=True)


show(plain, "Plain nn.Sequential")
show(single_td, "Single TensorDictModule")
show(policy, "TensorDictSequential (compact, light)")
show(policy, "TensorDictSequential (full, dark)", detail="full", theme="dark")
