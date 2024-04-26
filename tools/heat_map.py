import torch
from torchvision import utils, models, datasets, transforms
import torch.nn as nn
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet152
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image


# Run with forward() function
# module: Pointed module
# inputs: input of module's forward() function
# outputs: output of module's forward() function
def forward_hook(module, inputs, outputs):
    global feature
    # To handle multiple inputs, it seems that 'inputs' is wrapped in a tuple.
    # In this case, there is only one 'input,' so [0] is specified.
    # The module where the hook function is registered is assumed to be the layer that computes feature maps
    # (in this case, 'model.features'). To obtain the 'outputs' ,
    # which are the results of computing 'model.features,' you can simply get it directly.
    feature = outputs[0]


# Run with forward() function
# module: Pointed module
# inputs: input of module's forward() function
# outputs: output of module's forward() function
def backward_hook(module, grad_inputs, grad_outputs):
    global feature_grad
    # The module where the hook function is registered is expected to be 'model.features,' similar to a forward hook.
    # Since we want the gradients of the feature maps, you should directly obtain 'grad_outputs',
    # which corresponds to the gradient results of 'model.features.'
    feature_grad = grad_outputs[0]


# Convert to heatmap
def toHeatmap(x):
    x = (x*255).reshape(-1)
    cm = plt.get_cmap('jet')
    x = np.array([cm(int(np.round(xi)))[:3] for xi in x])
    return x.reshape(224, 224, 3)


imagePath = "/home/hanju/PycharmProjects/HanjuCNN/20231010_145318_NG_Src2_1_polar_lut_img.jpg"
modelPath = "/home/hanju/PycharmProjects/HanjuCNN/checkpoints/Assy_models/pt_checkpoints/Assy_Resnet152_1012_188.pt"
target_layers = ["layer4", "avgpool", "fc"]

Labels = ["NG", "OK"]

# Load image
test_image = cv2.imread(imagePath)
# BGR to RGB
test_image_converted = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# Convert opencv image to PIL image
pil_test_image = Image.fromarray(test_image_converted)

# Transformer
transformer = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
]
)
test_image_tensor = transformer(pil_test_image)
test_image_tensor = test_image_tensor.unsqueeze(0)  # (3, 224, 224) -> (1, 3, 224, 224)

# Load model
model = models.get_model("resnet152", pretrained=False)
model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
model.fc.out_features = 2       # Pre-trained model's output size is 2
print(model)

# Register forward and backward hook into layer4(If want to see heat map of another layer, change layer here)
model.layer4.register_forward_hook(forward_hook)
model.layer4.register_backward_hook(backward_hook)

model.eval()

predict = model(test_image_tensor)      # Forward predict
predict_index = torch.argmax(predict)   # Get the output index

predict[0][predict_index].backward()    # Use output index to backward

print(f"Predict result: {Labels[predict_index]}")

feature_vec = feature_grad.view(2048, 7*7)      # torch.Size([2048, 49])

# Calculate every colum's average. (Here is 2048)
# Calculate α in the paper
alpha = torch.mean(feature_vec, axis=1)     # torch.Size([2048])

# Delete batch size
# (1x512x7x7) -> (512x7x7)
feature = feature.squeeze(0)

L = F.relu(torch.sum(feature*alpha.view(-1, 1, 1), 0))
L = L.detach().numpy()

# Normalize to 0-1
L_min = np.min(L)
L_max = np.max(L - L_min)
L = (L - L_min)/L_max

# Resize to the same size with transformed image
L = cv2.resize(L, (224, 224))

# Convert to heat map
img2 = toHeatmap(L)

# squeeze(0): (1, 3, 224, 224) -> (3, 224, 224); permute(1,2,0): (3, 224, 224) -> (224, 224, 3)
img1 = test_image_tensor.squeeze(0).permute(1,2,0)

alpha = 0.5
grad_cam_image = img1*alpha + img2*(1-alpha)

# save heat map
numpy_grad_cam_image = grad_cam_image.detach().numpy()
arr_img = np.float32(numpy_grad_cam_image)
arr_img = arr_img * 255
numpy_grad_cam_image = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("heat_map.jpg", numpy_grad_cam_image)

plt.imshow(grad_cam_image)
plt.show()
