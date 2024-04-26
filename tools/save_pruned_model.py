import onnx
import torch
from datetime import datetime
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch_pruning as tp


model = models.get_model(name="resnet18", pretrained=False)
model.load_state_dict(torch.load("../checkpoints/resnet18-5c106cde.pth"))
model.to("cpu")

example_input = torch.randn([1, 3, 224, 224])

importance_class = getattr(tp.importance, 'MagnitudeImportance')
imp = importance_class(p=2)

ignored_layers = []
# DO NOT prune the final classifier!!
for m in model.modules():
    if isinstance(m, torch.nn.Linear):
        ignored_layers.append(m)
        
pruner_class = getattr(tp.pruner, 'MagnitudePruner')
pruner = pruner_class(
    model,
    example_input,
    importance=imp,
    iterative_steps=1,
    ch_sparsity=0.1,
    ignored_layers=ignored_layers
)
        
pruner.step()

model.eval()

print(model)

checkPoint = {'model': model, 
              'state_dick': model.state_dict()}

torch.save(checkPoint, "../checkpoints/resnet18_pruned_01.pth")
