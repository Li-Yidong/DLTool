import torch
import utils
import argparse


def namespace_to_dict(namespace):
    dictionary = vars(namespace)
    return dictionary


def export_onnx_model(model, output_path, input_shape):
    # define dummy input
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(model,
                      dummy_input,
                      output_path,
                      export_params=True,
                      opset_version=11,
                      verbose=False,
                      input_names=["input"],
                      output_names=["output"])


# create comment parser
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--classes_num', type=int, default=2)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--onnx_path', type=str, default='')
parser.add_argument('--use_official_model', type=bool, default=False)
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--input_shape', default=(1, 3, 224, 224))

args = parser.parse_args()

# convert argparse.Namespace to dictionary
config = namespace_to_dict(args)

# load pt model
model = utils.model_loader(config)
model.eval()

# get input shape and output path
output_path = args.onnx_path
input_shape = args.input_shape

# export onnx model
export_onnx_model(model, output_path, input_shape)
