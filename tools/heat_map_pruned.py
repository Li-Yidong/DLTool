import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
from math import ceil
from functools import partial
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchvision import models
from torchcam.utils import overlay_mask
from torchcam.methods import (
    CAM,
    GradCAM,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
    GradCAMpp,
    SmoothGradCAMpp,
)
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)
from typing import List, Union, Optional
import numpy as np
import torch_pruning as tp
import cv2


def load_checkpoint(modelPath):
    checkpoint = torch.load(modelPath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dick'])

    model.eval()

    return model


def load_image(image_path: str, input_shape: List[int] = [224, 224]) -> Tensor:
    img_pil: Image.Image = Image.open(image_path)

    transformer: Compose = Compose([Resize(input_shape), ToTensor()])
    img_tensor: Tensor = transformer(img_pil)

    return img_tensor.unsqueeze(0)


def get_all_candidate_layers(
    model: nn.Module, input_shape: tuple[int, ...] = (3, 224, 224)
) -> List[str]:
    module_mode: bool = model.training
    model.eval()

    output_shapes: List[tuple[Optional[str], tuple[int, ...]]] = []

    def _record_output_shape(
        module: nn.Module, input: Tensor, output: Tensor, name: Optional[str] = None
    ) -> None:
        output_shapes.append((name, output.shape))

    hook_handles: List[RemovableHandle] = []
    for n, m in model.named_modules():
        if type(m) in {nn.Sequential, nn.Conv2d}:
            hook_handles.append(
                m.register_forward_hook(partial(_record_output_shape, name=n))
            )

    with torch.no_grad():
        _ = model(
            torch.zeros((1, *input_shape), device=next(model.parameters()).data.device)
        )

    for handle in hook_handles:
        handle.remove()

    model.training = module_mode

    candidate_layers: List[str] = []
    for layer_name, output_shape in output_shapes:
        if (
            len(output_shape) == (len(input_shape) + 1)
            and any(v != 1 for v in output_shape[2:])
            and len(layer_name.split(".")) <= 2  # ! depth of the module name
        ):
            candidate_layers.append(layer_name)

    return candidate_layers


def generate_heatmap(
    model_architecture: str,
    model_path: str,
    image_path: str,
    classes_num: int = 2,
    old_version: bool = False,
    input_shape: List[int] = [224, 224],
    target_layer: Optional[Union[List[str], str]] = None,
    all_layers: bool = False,
    cam: Union[
        CAM, GradCAM, LayerCAM, ScoreCAM, XGradCAM, GradCAMpp, SmoothGradCAMpp
    ] = GradCAM,
) -> List[tuple[str, Image.Image]]:
    
    model = load_checkpoint(model_path)

    # Create a cam_extractor by target method
    cam_extractor: Union[  # ! warning here, about target_layer being None
        CAM, GradCAM, LayerCAM, ScoreCAM, GradCAMpp, SmoothGradCAMpp
    ] = cam(
        model,
        target_layer=target_layer
        if not all_layers
        else get_all_candidate_layers(model),
    )

    # load image and get output
    inputs: Tensor = load_image(image_path, input_shape=input_shape)
    out: Tensor = model(inputs)

    # Use cam_extractor function to forward output of model and prediction result, get heatmap
    cams: List[Tensor] = cam_extractor(out.squeeze(0).argmax().item(), out)

    # Put heatmap on resource image and save it to images list
    image: Image.Image
    for name, c in zip(cam_extractor.target_names, cams):
        res: Image.Image = overlay_mask(
            F.to_pil_image(inputs.squeeze(0)),
            F.to_pil_image(c.squeeze(0), mode="F"),
            alpha=0.6,
        )
        image = res

    np_image = np.array(image)
    # Convert RGB to BGR
    np_image = np_image[:, :, ::-1].copy()

    return np_image


if __name__ == "__main__":
    cam_image: np.array = generate_heatmap(
        "resnet18",
        "./checkpoints/resnet18_pruned_01.pth",
        "./imgs/dog.jpg",
        classes_num=1000,
        cam=GradCAM,
        old_version=True,
    )

    cv2.imshow("cam", cam_image)
    cv2.waitKey(0)
