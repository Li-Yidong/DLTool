import onnx
import torch
import numpy as np
from datetime import datetime
from torchvision import datasets, transforms, models
# import torch_pruning as tp
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tools import utils
from torch import Tensor
from sklearn.metrics import classification_report
from typing import List, Callable
from tqdm import tqdm

SEPARATOR: str = ''.join(['-'] * 24)

def export_onnx_model(model, output_path, input_shape, device):
    # define dummy input
    dummy_input = torch.randn(input_shape, device=device)

    torch.onnx.export(model,
                      dummy_input,
                      output_path,
                      export_params=True,
                      opset_version=11,
                      verbose=False,
                      input_names=["input"],
                      output_names=["output"])


# get dataloader
def get_dataset(config):
    train_dataloader = utils.dataloader(config=config['train_data_config'])
    val_dataloader = utils.dataloader(config=config['val_data_config'])

    return train_dataloader, val_dataloader


# get optimizer
def get_optimizer(parameters, config):
    optimizer_class = getattr(optim, config['optimizer'])
    if config['optimizer'] == 'SGD':
        optimizer = optimizer_class(parameters, lr=config['learning_rate'], momentum=config['momentum'])
        return optimizer
    else:
        optimizer = optimizer_class(parameters, lr=config['learning_rate'])
        return optimizer


def get_loss_func(config):
    loss_class = getattr(nn, config['criterion'])
    loss_func = loss_class()
    return loss_func


def f1_cal(outputs, tar, targets):
    tp = 0
    fn = 0
    fp = 0
    outs = outputs.cpu().detach().numpy()
    target = targets.cpu().detach().numpy()
    num_rows, num_cols = outs.shape
    for i in range(num_rows):
        out = outs[i]
        if target[i] == tar:
            if np.argmax(out) == tar:
                tp = tp + 1
            else:
                fn = fn + 1
        if np.argmax(out) == tar:
            if target[i] != tar:
                fp = fp + 1
    return tp, fn, fp


def model_evaluate(model, val_loader, criterion, config, device):
    val_loss = []
    val_outputs: List[int]; val_targets: List[int]
    val_outputs, val_targets = [], []
    for inputs, targets in val_loader:
        inputs: Tensor; targets: Tensor
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss.append(loss.item())

        y_pred: Tensor
        _, y_pred = torch.max(outputs, 1)
        val_outputs += y_pred.cpu().detach().tolist()
        val_targets += targets.cpu().detach().tolist()

    val_loss = np.mean(val_loss)

    return val_loss, classification_report(val_targets, val_outputs, digits=4, output_dict=True, zero_division=0.0)


def save_model(model, config, it, device):
    save_config = config['save_config']

    model_save_name = save_config['save_name']
    model_save_path = save_config['save_path']

    if 'save_onnx_per_epoch' in save_config:
        if save_config['save_onnx_per_epoch']:
            onnx_save_name = model_save_name + "_" + str(it) + ".onnx"
            onnx_save_path = model_save_path + onnx_save_name

            export_onnx_model(model,
                              output_path=onnx_save_path,
                              input_shape=save_config['input_shape'],
                              device=device)
        else:
            onnx_save_name = model_save_name + ".onnx"
            onnx_save_path = model_save_path + onnx_save_name

            export_onnx_model(model,
                              output_path=onnx_save_path,
                              input_shape=save_config['input_shape'],
                              device=device)
    else:
        if save_config['save_per_epoch']:
            pt_save_name = model_save_name + "_" + str(it) + ".pt"
            pt_path = model_save_path + pt_save_name
            torch.save(model.state_dict(), pt_path)
        else:
            pt_save_name = model_save_name + ".pt"
            pt_path = model_save_path + pt_save_name
            torch.save(model.state_dict(), pt_path)


def format_print(report: dict, precision: int=4):
    stringify_dict: Callable =\
        lambda dic, p: ', '.join([f'{key}: {round(val, p)}' for key, val in dic.items()])
    _ = report.pop('accuracy')
    macro_avg: dict = report.pop('macro avg')
    weighted_avg: dict = report.pop('weighted avg')
    for key, val in report.items():
        key: str; val: dict
        print(f'Class {key}:', stringify_dict(val, precision))
    print('Macro Avg:', stringify_dict(macro_avg, precision))
    print('Weighted Avg:', stringify_dict(weighted_avg, precision))


def format_print_dict(report: dict, precision: int=4):
    accuracy: float = report.pop('accuracy')
    macro_avg: dict = report.pop('macro avg')
    weighted_avg: dict = report.pop('weighted avg')
    cnames: List[str] = report.keys()
    cprecisions: List[float] = [report[cname]['precision'] for cname in cnames]
    crecalls: List[float] = [report[cname]['recall'] for cname in cnames]
    cf1s: List[float] = [report[cname]['f1-score'] for cname in cnames]
    csupport: List[int] = [report[cname]['support'] for cname in cnames]

    rows = zip(cnames, cprecisions, crecalls, cf1s, csupport)

    headers = ['precision', 'recall', 'f1-score', 'support']
    longest_last_line_heading: str = 'weighted avg'
    name_width: int = max(len(cn) for cn in cnames)
    width: int = max(name_width, len(longest_last_line_heading), precision)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    rep = head_fmt.format("", *headers, width=width)
    rep += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for row in rows:
        rep += row_fmt.format(*row, width=width, digits=precision)
    rep += "\n"
    row_fmt_accuracy = (
        "{:>{width}s} "
        + " {:>9.{digits}}" * 2
        + " {:>9.{digits}f}"
        + " {:>9}\n"
    )
    rep += row_fmt_accuracy.format(
        'accuracy', '', '', accuracy, sum(csupport), width=width, digits=precision
    ) # ! variable accuracy is not correct
    rep += row_fmt.format('macro_avg', *macro_avg.values(), width=width, digits=precision)
    rep += row_fmt.format('weighted_avg', *weighted_avg.values(), width=width, digits=precision)
    print(rep)
    print(SEPARATOR)


def batch_gd(model, 
             device, 
             criterion, 
             optimizer, 
             train_loader, 
             val_loader, 
             epochs, 
             config):
    model.to(device)

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    weight_decay = 0

    best_f1 = float('-inf')
    best_loss = float('inf')
    train_outputs: List[int]; train_targets: List[int]
    train_outputs, train_targets = [], []
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in tqdm(train_loader, desc=f"Epoch: {it} Tranning"):
            # move data to device
            inputs: Tensor; targets: Tensor
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradents
            optimizer.zero_grad()

            # forward pass
            outputs: Tensor = model(inputs)
            loss = criterion(outputs, targets)

            y_pred: Tensor
            _, y_pred = torch.max(outputs, 1)
            train_outputs += y_pred.cpu().detach().tolist()
            train_targets += targets.cpu().detach().tolist()

            # calculate regularization of normalized parameters
            l2_regularization = 0
            for param in model.parameters():
                l2_regularization += torch.norm(param)

            # Backward adn optimize
            loss += weight_decay * l2_regularization
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss: float = np.mean(train_loss)
        train_report: dict = classification_report(train_targets, train_outputs, digits=4, output_dict=True, zero_division=0.0)
        model.eval()

        # model evaluate
        val_loss: float; val_report: dict
        val_loss, val_report = model_evaluate(model, 
                                              val_loader, 
                                              criterion, 
                                              config['Hyperparams_config'], 
                                              device)

        # save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        avg_loss = (train_loss + val_loss) / 2

        # f1 score towards the target class
        f1: float = val_report[str(config['Hyperparams_config']['target'])]['f1-score']

        dt = datetime.now() - t0
        print(
            f'Epoch {it + 1}\{epochs}, Train Loss: {train_loss:.4f}, '
            f'Test Loss: {val_loss:.4f}, Duration: {dt}, TrainAccuracy: {train_report["accuracy"]:.4f}, '
            f'ValAccuracy: {val_report["accuracy"]:.4f}'
        )
        format_print_dict(train_report)
        format_print_dict(val_report)

        # save models
        save_model(model, config, it, device)

        # save best model
        if f1 > best_f1:
            torch.save(model.state_dict(), f"{config['save_config']['save_path']}/best_checkpoint.pt")
            best_f1 = f1
        elif f1 == best_f1:
            if avg_loss < best_loss:
                torch.save(model.state_dict(), f"{config['save_config']['save_path']}/best_checkpoint.pt")
                best_loss = avg_loss

    # save model
    save_model(model, config, 0, device)

    return train_losses, val_losses


def batch_gd_ver2(model, 
             device, 
             criterion, 
             optimizer, 
             train_loader, 
             val_loader, 
             epochs, 
             config):
    
    model.to(device)

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    weight_decay = 0

    best_f1 = float('-inf')
    best_loss = float('inf')
    train_outputs: List[int]; train_targets: List[int]
    train_outputs, train_targets = [], []
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in tqdm(train_loader, desc=f"Epoch: {it} Tranning"):
            # move data to device
            inputs: Tensor; targets: Tensor
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradents
            optimizer.zero_grad()

            # forward pass
            outputs: Tensor = model(inputs)
            loss = criterion(outputs, targets)

            y_pred: Tensor
            _, y_pred = torch.max(outputs, 1)
            train_outputs += y_pred.cpu().detach().tolist()
            train_targets += targets.cpu().detach().tolist()

            # calculate regularization of normalized parameters
            l2_regularization = 0
            for param in model.parameters():
                l2_regularization += torch.norm(param)

            # Backward adn optimize
            loss += weight_decay * l2_regularization
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss: float = np.mean(train_loss)
        train_report: dict = classification_report(train_targets, 
                                                   train_outputs, 
                                                   digits=4, 
                                                   output_dict=True, 
                                                   zero_division=0.0)
        model.eval()

        # model evaluate
        val_loss: float; val_report: dict
        val_loss, val_report = model_evaluate(model, 
                                              val_loader, 
                                              criterion, 
                                              config['Hyperparams_config'], 
                                              device)

        # save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        avg_loss = (train_loss + val_loss) / 2

        # f1 score towards the target class
        f1: float = val_report[str(config['Hyperparams_config']['target'])]['f1-score']

        dt = datetime.now() - t0
        print(
            f'Epoch {it + 1}\{epochs}, Train Loss: {train_loss:.4f}, '
            f'Test Loss: {val_loss:.4f}, Duration: {dt}, TrainAccuracy: {train_report["accuracy"]:.4f}, '
            f'ValAccuracy: {val_report["accuracy"]:.4f}'
        )
        format_print_dict(train_report)
        format_print_dict(val_report)

        # save models
        utils.save_checkpoint(model, config, it, device)

    # save model
    utils.save_checkpoint(model, config, 0, device)

    return train_losses, val_losses


def model_test(model, train_dataloader, val_dataloader, device):
    # train accuracy
    train_outputs: List[int]; train_targets: List[int]
    train_outputs, train_targets = [], []
    for inputs, targets in train_dataloader:
        # move data to GPU
        inputs: Tensor; targets: Tensor;
        inputs, targets = inputs.to(device), targets.to(device)

        # forward pass
        outputs = model(inputs)

        # get the prediction
        # torch.max returns both max and argmax
        predictions: Tensor
        _, predictions = torch.max(outputs, 1)
        train_outputs += predictions.cpu().detach().tolist()
        train_targets += targets.cpu().detach().tolist()

    val_outputs: List[int]; val_targets: List[int]
    val_outputs, val_targets = [], []
    for inputs, targets in val_dataloader:
        # move data to GPU
        inputs: Tensor; targets: Tensor
        inputs, targets = inputs.to(device), targets.to(device)

        # forward pass
        outputs = model(inputs)

        # get the prediction
        # torch.max returns both max and argmax
        predictions: Tensor
        _, predictions = torch.max(outputs, 1)
        val_outputs += predictions.cpu().detach().tolist()
        val_targets += targets.cpu().detach().tolist()

    print('Training Report')
    print(classification_report(train_targets, train_outputs, digits=4, zero_division=0.0))
    print(SEPARATOR)
    print('\nValidation Report')
    print(classification_report(val_targets, val_outputs, digits=4, zero_division=0.0))
    print(SEPARATOR)


def train(config):
    # get device
    device = torch.device(config['device'])

    # get dataloader
    train_dataloader, val_dataloader = get_dataset(config=config['dataset_config'])

    # get model
    model = utils.model_loader(config=config['model_config'])
    print(model)

    hyperparams_config = config['Hyperparams_config']
    # get optimizer and loss function
    optimizer = get_optimizer(model.parameters(), config=hyperparams_config)
    criterion = get_loss_func(config=hyperparams_config)

    # trainer
    epochs = hyperparams_config['epochs']
    train_losses, val_losses = batch_gd(model, 
                                        device, 
                                        criterion, 
                                        optimizer, 
                                        train_dataloader, 
                                        val_dataloader, 
                                        epochs, config)

    # calculate accuracy
    model_test(model, 
               train_dataloader, 
               val_dataloader, 
               device)

    # Plot the train loss and test loss per iteration
    if config['graph_mode']:
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='validation loss')
        plt.legend()
        plt.show()


def train_checkpoint(config):
    # get device
    device = torch.device(config['device'])

    # get dataloader
    train_dataloader, val_dataloader = get_dataset(config=config['dataset_config'])

    # get checkpoint
    checkpoint = utils.checkpoint_loader(config=config['model_config'])
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    print(model)

    hyperparams_config = config['Hyperparams_config']
    # get optimizer and loss function
    optimizer = get_optimizer(model.parameters(), config=hyperparams_config)
    criterion = get_loss_func(config=hyperparams_config)

    # trainer
    epochs = hyperparams_config['epochs']
    train_losses, val_losses = batch_gd_ver2(model, 
                                            device, 
                                            criterion, 
                                            optimizer, 
                                            train_dataloader, 
                                            val_dataloader, 
                                            epochs, 
                                            config)

    # calculate accuracy
    model_test(model, 
               train_dataloader, 
               val_dataloader, 
               device)

    # Plot the train loss and test loss per iteration
    if config['graph_mode']:
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='validation loss')
        plt.legend()
        plt.show()
