import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
from cv2 import Mat
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

from utils import set_classes_num


def load_pretrained_model(
    architecture: str, model_path: str, classes_num: int = 2, old_version: bool = False
) -> nn.Module:
    model = models.get_model(name=architecture, weights=None)
    model = set_classes_num(model, architecture, classes_num, old_version=old_version)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    return model


def load_image(image_path: str, input_shape: List[int] = [224, 224]) -> Tensor:
    img: Mat = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, ::-1]
    img_pil: Image.Image = Image.fromarray(img)

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
    """
    Generate heatmap overlaid on the input image to visualize where the model is focusing its attention.

    Parameters:
        model_architecture (str): The architecture of the model to be used.
        model_path (str): The path to the model weights file.
        image_path (str): The path to the input image for which the heatmap is to be generated.
        classes_num (int, optional): Number of classes in the model. Defaults to 2.
        old_version (bool, optional): Whether to use an older version of the model. Defaults to False.
        input_shape (List[int], optional): Shape of the input image expected by the model. Defaults to [224, 224].
        target_layer (Optional[Union[List[str], str]], optional): Name of the target layer or list of layer names 
            to visualize. If None, uses default layer(s) of the model. Defaults to None.
        all_layers (bool, optional): Whether to use all candidate layers to generate heatmap. Defaults to False.
        cam (Union[...], optional): Class Activation Map (CAM) method to be used. Defaults to GradCAM.

    Returns:
        List[tuple[str, Image.Image]]: List of tuples containing the name of the CAM method and the 
            image with heatmap overlay.
    """

    # Load model and turn to evaluate mode
    model: nn.Module = load_pretrained_model(
        model_architecture, model_path, classes_num=classes_num, old_version=old_version
    ).eval()

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
    images: List[tuple[str, Image.Image]] = []
    for name, c in zip(cam_extractor.target_names, cams):
        res: Image.Image = overlay_mask(
            F.to_pil_image(inputs.squeeze(0)),
            F.to_pil_image(c.squeeze(0), mode="F"),
            alpha=0.6,
        )
        images.append((name, res))

    return images


def show_heatmap(
    heatmap: List[tuple[str, Image.Image]],
    iterate: bool = False,
    columns: int = 4,
) -> None:
    if iterate:
        for name, img in heatmap:
            plt.imshow(img)
            plt.axis("off")
            plt.title(name)
            plt.show()
    else:
        n_rows: int = ceil(len(heatmap) / columns)
        _, axes = plt.subplots(n_rows, columns, figsize=(12, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, (name, img) in enumerate(heatmap):
            ax = axes[i // columns, i % columns]
            ax.imshow(img)
            ax.set_title(name)

        for ax in axes.flatten():
            ax.axis("off")

        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cam_images: List[tuple[str, Image.Image]] = generate_heatmap(
        "resnet18",
        "./checkpoints/resnet18-5c106cde.pth",
        "./imgs/dog.jpg",
        classes_num=1000,
        cam=GradCAM,
        old_version=True,
    )

    show_heatmap(cam_images)
