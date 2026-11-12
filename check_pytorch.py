"""Check all three notebook convolution APIs against identical weights."""
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_num_threads(2)
torch.manual_seed(42)
ns = dict(torch=torch, nn=nn, F=F, np=np, dtype=torch.float32, device=torch.device('cpu'))
notebook = json.loads((Path(__file__).parent/'assignment2/PyTorch.ipynb').read_text(encoding='utf-8'))
for index in (10, 12, 14, 28, 30):
    exec(''.join(notebook['cells'][index]['source']), ns)
model = ns['ThreeLayerConvNet'](3, 4, 5, 10)
x = torch.randn(3, 3, 32, 32)
params = [model.conv1.weight, model.conv1.bias, model.conv2.weight,
          model.conv2.bias, model.fc.weight.T, model.fc.bias]
expected = ns['three_layer_convnet'](x, params)
assert expected.shape == (3, 10)
assert torch.allclose(model(x), expected, atol=1e-5)
sequential = nn.Sequential(model.conv1, nn.ReLU(), model.conv2, nn.ReLU(),
                           nn.Flatten(), model.fc)
assert torch.allclose(sequential(x), expected, atol=1e-5)
F.cross_entropy(expected, torch.tensor([0, 1, 2])).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('Barebones / Module / Sequential equivalence and gradients PASS')
