from model_visualizer import ModelVisualizer
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential

model1 = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
model2 = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
model3 = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 3))

seq_module2 = TensorDictSequential(
    TensorDictModule(model1, in_keys=["observation"], out_keys=["latentspace"]),
    TensorDictModule(model2, in_keys=["latentspace"], out_keys=["action1"]),
    TensorDictModule(model3, in_keys=["latentspace"], out_keys=["action2"]),
)


# Now you can create a visualizer instance and choose a backend
#visualizer = ModelVisualizer(model=model1, backend="graphviz")
visualizer = ModelVisualizer(model=seq_module2, backend="graphviz")
# Or, if you need to interact directly with a specific backend:
# graphviz_backend = GraphvizBackend()

visualizer.visualize(render=False)
visualizer.view()
#visualizer.clear()

# visualizer = ModelVisualizer(model=seq_module2, backend="graphviz")
# visualizer.visualize(render=False)
# visualizer.view()