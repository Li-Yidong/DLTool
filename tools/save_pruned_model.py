import onnx
import torch
from datetime import datetime
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch_pruning as tp


ori_model_arc = "resnet152"
pruned_model_path = "Z:/work_files/YuTaKa_assy/CpythonTest/CpythonTest/CpythonTest/anomaly_classify_upper_20240718.pt"
pruning_rate = 0.8
class_num = 2
save_path = "Z:/work_files/YuTaKa_assy/CpythonTest/CpythonTest/CpythonTest/anomaly_classify_upper_20240718_checkpoint.pt"

model = models.get_model(name=ori_model_arc, pretrained=True)

n_feats: int = model.fc.in_features
model.fc = nn.Linear(n_feats, class_num)

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
    ch_sparsity=pruning_rate,
    ignored_layers=ignored_layers
)

pruner.step()

model.load_state_dict(torch.load(pruned_model_path))

model.eval()

print(model)

checkPoint = {'model': model, 
              'state_dict': model.state_dict()}

torch.save(checkPoint, save_path)
