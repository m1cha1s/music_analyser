import torch
from torchviz import make_dot
import numpy as np
from sys import argv

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

samples = np.load("samples.npy").astype("float32")

m = torch.load(argv[1], weights_only=False, map_location=torch.device(device)).to(device)

x = torch.stack([torch.from_numpy(samples[0]).to(device)])
x.unsqueeze_(1)

y = m(x)

print(y)

graph = make_dot(y[0], dict(m.named_parameters()))
graph.render(filename="model_structure", format="png", cleanup=True)
