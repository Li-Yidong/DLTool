import torch
from tools import utils
import argparse


# create comment parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_version', type=str, default='DLTool')
parser.add_argument('--input_model_path', type=str, default="./checkpoints/resnet18-5c106cde.pth")
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--classes_num', type=int, default=1000)
parser.add_argument('--output_model_path', type=str, default="./checkpoints/resnet18-5c106cde.onnx")
parser.add_argument('--input_shape', default=(1, 3, 224, 224))


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


def export_inference_model(config: dict):
    model_version = config['model_version']
    input_model_path = config['input_model_path']
    model_attributes = config['model']
    classes_num = config['classes_num']
    output_model_path = config['output_model_path']
    input_shape = config['input_shape']
    
    custom_checkpoint: bool = False
    if (model_version == 'DLTool'):
        custom_checkpoint = True

    checkpoint_config = {'model': model_attributes, 
                         'classes_num': classes_num,
                         'use_official_model': False,
                         'custom_checkpoint': custom_checkpoint,
                         'pretrained': True,
                         'pretrained_model': input_model_path}

    # Load checkpoint
    checkpoint = utils.checkpoint_loader(checkpoint_config)
    
    model = checkpoint['model']
    model.state_dict().update(checkpoint['state_dict'])

    # Export ONNX model
    export_onnx_model(model, output_model_path, input_shape)

    print('Export ONNX model finished!')


def run_export(config: dict):
    
    flatten_config = {'model_version': config['input_model_config']['model_version'],
                      'input_model_path': config['input_model_config']['input_model_path'],
                      'model': config['input_model_config']['model'],
                      'classes_num': config['input_model_config']['classes_num'],
                      'output_model_path': config['output_model_config']['output_model_path'],
                      'input_shape': config['output_model_config']['input_shape']}
    
    # export inference model
    export_inference_model(flatten_config)


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # convert argparse.Namespace to dictionary
    config = namespace_to_dict(args)

    # export inference model
    export_inference_model(config)
