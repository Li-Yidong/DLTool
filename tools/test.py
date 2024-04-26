from tools import utils
import os
import sys
import torchvision.utils
import onnxruntime
import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms, models
import onnx
from datetime import datetime
from tqdm import tqdm


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def test_pt(config):
    # get dataloader
    test_dataloader = utils.dataloader(config=config['dataset_config'])

    # get model
    model = utils.model_loader(config['model_config'])
    model.to(config['device'])
    model.eval()

    # set target
    target = config['target']

    # declare parameters
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    test_correct = 0
    test_n = 0

    # save predict result
    predict_results = []

    t0 = datetime.now()

    # test
    for image, labels in test_dataloader:
        # send images and labels to device
        if config['device'] == 'cuda:0':
            image, labels = image.to('cuda:0'), labels.to('cuda:0')

        label = labels.item()

        # forward
        outputs = model(image)

        # send output to cpu
        if config['device'] == 'cuda:0':
            outputs = outputs.to('cpu')

        _, predicted = torch.max(outputs, dim=1)
        predicted = predicted.item()
        predict_results.append(predicted)

        # statistical prediction result
        if label == target and predicted == target:
            true_positives += 1
        elif label == target and predicted != target:
            false_negatives += 1
        elif label != target and predicted == target:
            false_positives += 1

        # count result
        if label == predicted:
            test_correct += 1

        test_n += 1

    utils.save_wrong_images(predict_results, test_dataloader, target, config['class_names'])

    # calculate recall and precision and accurate
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    accurate = test_correct / test_n

    dt = datetime.now() - t0
    print(f'Duration: {dt}', f'Test accurate: {accurate:.4f}', f'Recall: {recall: .4f}', f'Precision: {precision: .4f}')


def test_pt_checkpoint(config):
    # get dataloader
    test_dataloader = utils.dataloader(config=config['dataset_config'])

    # get model
    checkpoint = utils.checkpoint_loader(config['model_config'])
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(config['device'])
    model.eval()

    # declare parameters
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    test_correct = 0
    test_n = 0

    # save predict result
    predict_results = []

    # set target
    target = config['target']

    t0 = datetime.now()

    # test
    for image, labels in tqdm(test_dataloader, desc="Testing"):
        # send images and labels to device
        if config['device'] == 'cuda:0':
            image, labels = image.to('cuda:0'), labels.to('cuda:0')

        label = labels.item()

        # forward
        outputs = model(image)

        # send output to cpu
        if config['device'] == 'cuda:0':
            outputs = outputs.to('cpu')

        _, predicted = torch.max(outputs, dim=1)
        predicted = predicted.item()
        predict_results.append(predicted)

        # statistical prediction result
        if label == target and predicted == target:
            true_positives += 1
        elif label == target and predicted != target:
            false_negatives += 1
        elif label != target and predicted == target:
            false_positives += 1

        # count result
        if label == predicted:
            test_correct += 1

        test_n += 1

    utils.save_wrong_images(predict_results, test_dataloader, target, config['class_names'])

    # calculate recall and precision and accurate
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    accurate = test_correct / test_n

    dt = datetime.now() - t0
    print(f'Duration: {dt}', f'Test accurate: {accurate:.4f}', f'Recall: {recall: .4f}', f'Precision: {precision: .4f}')


def test_onnx(config):
    # get device
    device = config['device']

    # get dataloader
    test_dataloader = utils.dataloader(config=config['dataset_config'])

    # get model
    model_config = config['model_config']
    onet_session = onnxruntime.InferenceSession(model_config['pretrained_model'])

    # Set providers
    providers = ['CUDAExecutionProvider'] if device == 'cuda:0' else []
    onet_session.set_providers(providers)

    # set target
    target = config['target']

    # declare parameters
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    test_correct = 0
    test_n = 0

    # save predict result
    predict_results = []

    t0 = datetime.now()

    # test
    for image, labels in test_dataloader:
        if config['device'] == 'cuda:0':
            image, labels = image.to('cuda:0'), labels.to('cuda:0')

        label = labels.item()

        inputs = {onet_session.get_inputs()[0].name: to_numpy(image)}
        outs = onet_session.run(None, inputs)

        # get inference result
        preds = outs[0]

        # transform to tensor
        output_data_tensor = torch.tensor(preds).to(device)

        # softmax
        tensor_probabilities = torch.nn.functional.softmax(output_data_tensor, dim=1)

        probabilities = tensor_probabilities.cpu().numpy()
        prob = probabilities.flatten()

        # get result
        predicted = np.argmax(prob)

        predict_results.append(predicted)

        # statistical prediction result
        if label == target and predicted == target:
            true_positives += 1
        elif label == target and predicted != target:
            false_negatives += 1
        elif label != target and predicted == target:
            false_positives += 1

        if label == predicted:
            test_correct += 1

        test_n += 1

    utils.save_wrong_images(predict_results, test_dataloader, target, config['class_names'])

    # calculate recall and precision and accurate
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    accurate = test_correct / test_n

    dt = datetime.now() - t0
    print(f'Duration: {dt}', f'Test accurate: {accurate:.4f}', f'Recall: {recall: .4f}', f'Precision: {precision: .4f}')


