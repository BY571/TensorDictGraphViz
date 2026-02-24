from tensordictviz import ModelVisualizer
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential

# Define models
model1 = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
model2 = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
model3 = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 3))

# Create TensorDictSequential model
seq_td_module = TensorDictSequential(
    TensorDictModule(model1, in_keys=["observation"], out_keys=["latentspace"]),
    TensorDictModule(model2, in_keys=["latentspace"], out_keys=["action1"]),
    TensorDictModule(model3, in_keys=["latentspace"], out_keys=["action2"]),
)

# Create a single TensorDictModule
td_model = TensorDictModule(model1, in_keys=["observation"], out_keys=["action"])

# Create a simple PyTorch Sequential model
single_torch_model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))

# Function to visualize and display a model
def visualize_model(model, title, detail="compact"):
    visualizer = ModelVisualizer(model=model, backend="graphviz")
    visualizer.visualize(render=False, detail=detail)
    visualizer.backend.set_graph_attr(label=title)
    visualizer.view(wait=True)
    del visualizer

# Visualize different models
visualize_model(single_torch_model, "PyTorch Sequential Model")
visualize_model(td_model, "Single TensorDictModule")
visualize_model(seq_td_module, "TensorDictSequential Model (Compact)")

# Compare compact vs full detail
visualize_model(seq_td_module, "TensorDictSequential Model (Full)", detail="full")
