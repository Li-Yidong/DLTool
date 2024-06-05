import onnx
import torch
from datetime import datetime
from torchvision import datasets, transforms, models
# import torch_pruning as tp
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tools import utils
from tqdm import tqdm


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
    tar = config['target']
    TP = 0
    FN = 0
    FP = 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        tp, fn, fp = f1_cal(outputs, tar=tar, targets=targets)
        loss = criterion(outputs, targets)
        val_loss.append(loss.item())

        TP = TP + tp
        FN = FN + fn
        FP = FP + fp

    val_loss = np.mean(val_loss)

    recall = 0
    precision = 0
    if (TP + FN) != 0:
        recall += TP / (TP + FN)
    if (TP + FP) != 0:
        precision += TP / (TP + FP)

    if recall + precision == 0:
        f1_score = 0
    else:
        f1_score = 2 * (recall * precision) / (recall + precision)
    return val_loss, recall, precision, f1_score


def save_model(model, config, it, device, pruning_percentage, pruning_step):
    save_config = config['save_config']

    model_save_name = save_config['save_name']
    model_save_path = save_config['save_path']

    pruning_percentage_str = str(pruning_percentage)
    pruning_percentage_str_without_dot = pruning_percentage_str.replace(".", "")
    if save_config['save_onnx_per_epoch']:
        onnx_save_name = model_save_name + "_" + str(it) + "_" + pruning_percentage_str_without_dot + "_" + str(pruning_step) + ".onnx"
        onnx_save_path = model_save_path + onnx_save_name

        export_onnx_model(model,
                          output_path=onnx_save_path,
                          input_shape=save_config['input_shape'],
                          device=device)
    else:
        onnx_save_name = model_save_name + "_" + pruning_percentage_str_without_dot + ".onnx"
        onnx_save_path = model_save_path + onnx_save_name

        export_onnx_model(model,
                          output_path=onnx_save_path,
                          input_shape=save_config['input_shape'],
                          device=device)



def batch_gd(model, device, criterion, optimizer, train_loader, val_loader, epochs,
             config, pruning_percentage, pruning_step):
    model.to(device)

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    weight_decay = 0

    best_f1 = 0.0
    best_loss = 1.0
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in tqdm(train_loader, desc=f"Epoch: {it} Tranning"):
            # move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradents
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # calculate regularization of normalized parameters
            l2_regularization = 0
            for param in model.parameters():
                l2_regularization += torch.norm(param)

            # Backward adn optimize
            loss += weight_decay * l2_regularization
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        model.eval()

        # model evaluate
        val_loss, recall, precision, f1_score = model_evaluate(model, val_loader, criterion,
                                                               config['Hyperparams_config'], device)

        # save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        avg_loss = (train_loss + val_loss) / 2

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}\{epochs}, Train Loss: {train_loss:.4f}, \
                    Test Loss: {val_loss:.4f}, Duration: {dt}, Recall: {recall:.4f}, Precision: {precision:.4f}')

        # save models
        # save_model(model, config, it, device, pruning_percentage, pruning_step)
        
        # save checkpoints
        utils.save_checkpoint(model, config, it, device, pruning_percentage, pruning_step)

        # save best model
        if f1_score > best_f1:
            torch.save(model.state_dict(), "Output/checkpoints/best_checkpoint.pt")
            best_f1 = f1_score
        elif f1_score == best_f1:
            if avg_loss < best_loss:
                torch.save(model.state_dict(), "Output/checkpoints/best_checkpoint.pt")
                best_loss = avg_loss

    # save model
    # save_model(model, config, 0, device, pruning_percentage, pruning_step)

    utils.save_checkpoint(model, config, 0, device, pruning_percentage, pruning_step)

    return train_losses, val_losses


def model_test(model, train_dataloader, val_dataloader, device):
    # train accuracy
    n_correct = 0
    n_total = 0
    for inputs, targets in train_dataloader:
        # move data to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # forward pass
        outputs = model(inputs)

        # get the prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    train_acc = n_correct / n_total

    # test accuracy
    n_correct = 0
    n_total = 0
    n = 0
    ep = 1
    for inputs, targets in val_dataloader:
        # move data to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # forward pass
        outputs = model(inputs)

        # get the prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]
        ep = ep + 1

    test_acc = n_correct / n_total

    print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

