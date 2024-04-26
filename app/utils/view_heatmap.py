import sys

sys.path.append("../../../")
from tools.heat_map_pruned import generate_heatmap
import cv2
import numpy as np


def view_heatmap(config: dict) -> np.array:

    heatmap = generate_heatmap(model_architecture="resnet18", 
                               model_path=config["model_path"], 
                            image_path=config["image_path"],
                            input_shape=config["input_shape"],
                            classes_num=config["classes_num"])
    
    out = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return out