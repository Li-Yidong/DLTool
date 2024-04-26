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
import torch_pruning as tp
from tools import train
from tools import pruning_train

def get_importance(config):
    importance_fam = {
        'TaylorImportance',
        'MagnitudeImportance',
        'BNScaleImportance',
        'RandomImportance'
    }

    importance_class = getattr(tp.importance, config['features_importance'])
    imp = importance_class(p=2)
    return imp


def get_dataset(config):
    train_dataloader = utils.dataloader(config=config['train_data_config'])
    val_dataloader = utils.dataloader(config=config['val_data_config'])

    return train_dataloader, val_dataloader


def get_pruner(model, example_input, class_num, config):
    imp = get_importance(config)
    ignored_layers = []

    # DO NOT prune the final classifier!!
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == class_num:
            ignored_layers.append(m)

    iterative_steps = config['iterative_steps']
    ch_sparsity = config['channel_sparsity']

    pruner_fam = {
        'MetaPruner',
        'MagnitudePruner',
        'BNScalePruner',
        'GroupNormPruner',
    }

    pruner_class = getattr(tp.pruner, config['pruner'])
    pruner = pruner_class(
        model,
        example_input,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=ch_sparsity,
        ignored_layers=ignored_layers
    )

    return pruner


def prune_train(model, epochs, config, pruning_percentage, pruning_step):
    # get device
    device = torch.device(config['device'])

    # get dataloader
    train_dataloader, val_dataloader = get_dataset(config=config['dataset_config'])

    hyperparams_config = config['Hyperparams_config']
    # get optimizer and loss function
    optimizer = train.get_optimizer(model.parameters(), config=hyperparams_config)
    criterion = train.get_loss_func(config=hyperparams_config)

    # trainer
    # train_losses, val_losses = train.batch_gd(model, device, criterion, optimizer, train_dataloader, val_dataloader, epochs,
    #                                     config)

    # trainer
    train_losses, val_losses = pruning_train.batch_gd(model, device, criterion, optimizer, train_dataloader,
                                                      val_dataloader, epochs, config,
                                                      pruning_percentage, pruning_step)

    # calculate accuracy
    pruning_train.model_test(model, train_dataloader, val_dataloader, device)


def pruning(config):
    # load model
    model_config = config['model_config']
    print(model_config)
    model = utils.model_loader(model_config)

    # load dataset
    data_config = config['dataset_config']
    train_dataloader, val_dataloader = get_dataset(data_config)

    # get pruner configration
    pruner_config = config['pruner_config']

    # get example input
    example_input = torch.randn(pruner_config['input_shape'])

    # get class count
    classes_num = model_config['classes_num']

    # get pruner
    pruner = get_pruner(model, example_input, classes_num, pruner_config)

    # get base operators and parameters
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_input)

    iterative_steps = pruner_config['iterative_steps']
    hyperparams_config = config['Hyperparams_config']

    # start pruning
    for i in range(iterative_steps):
        if config['device'] == 'cuda:0' and i > 0:
            model.to('cpu')

        if i > 0:
            model.load_state_dict(torch.load("Output/checkpoints/best_checkpoint.pt"))

        pruner.step()

        # get operators and parameters after 1 step
        macs, params = tp.utils.count_ops_and_params(model, example_input)
        print(model)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i + 1, iterative_steps, base_params / 1e6, params / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )

        if i == iterative_steps - 1:
            # train model
            prune_train(model, hyperparams_config['last_epochs'], config, pruner_config["channel_sparsity"], i)
        else:
            # train model
            prune_train(model, hyperparams_config['epochs'], config, pruner_config["channel_sparsity"], i)





















