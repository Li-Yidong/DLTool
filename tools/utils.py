import os
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import models, datasets, transforms


def dataloader(config):
    #######################################################################
    # config sample

    # dataset_config = {
    #     # dataset
    #     'data': 'datasets/Assy_CNN_2classes/train',
    #     'batch_size': 32,
    #     'shuffle': True,
    #
    #     # data augmentation
    #     'Resize': [224, 224],
    #     'RandomRotation': 5,
    #     'ColorJitter': {
    #         'brightness': .5,
    #         'hue': .3
    #     },
    #     'RandomInvert': True,
    #     'RandomHorizontalFlip': True,
    #     'RandomVerticalFlip': True
    # }
    ########################################################################

    # default data augmentation
    default_transformers = [
    ]

    # check configuration directory
    if 'Resize' in config:
        size = config['Resize']
        default_transformers.append(transforms.Resize(size))
    else:
        default_transformers.append(transforms.Resize([224, 224]))

    if 'RandomRotation' in config:
        degree = config['RandomRotation']
        default_transformers.append(transforms.RandomRotation(degrees=degree))

    if 'ColorJitter' in config:
        params = config['ColorJitter']
        if 'brightness' in params:
            brightness = params['brightness']
            if 'hue' in params:
                hue = params['hue']
                default_transformers.append(transforms.ColorJitter(brightness=brightness, hue=hue))
            else:
                default_transformers.append(transforms.ColorJitter(brightness=brightness))

    if 'RandomInvert' in config:
        if config['RandomInvert']:
            default_transformers.append(transforms.RandomInvert())

    if 'RandomHorizontalFlip' in config:
        if config['RandomHorizontalFlip']:
            default_transformers.append(transforms.RandomHorizontalFlip())

    if 'RandomVerticalFlip' in config:
        if config['RandomVerticalFlip']:
            default_transformers.append(transforms.RandomVerticalFlip())

    default_transformers.append(transforms.ToTensor())

    data_transform = transforms.Compose(default_transformers)

    # import data
    data = config['data']
    dataset = datasets.ImageFolder(
        data,
        transform=data_transform
    )

    # Data loader (automatically generates batches in training loop & takes care of shuffling)
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return data_loader


def set_classes_num(
    model: nn.Module, model_name: str, classes_num: int, old_version: bool = False
):
    vgg_fam: set = {"vgg16", "vgg19"}
    resnet_fam: set = { "resnet" + str(x) for x in [18, 34, 50, 101, 152] }
    efficientnet_fam: set = { "efficientnet_b" + str(x) for x in range(8) }

    if model_name in vgg_fam:
        if old_version:
            model.classifier[6].out_features = classes_num
        else:
            n_feats: int = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(n_feats, classes_num)
    elif model_name in resnet_fam:
        if old_version:
            model.fc.out_features = classes_num
        else:
            n_feats: int = model.fc.in_features
            model.fc = nn.Linear(n_feats, classes_num)
    elif model_name in efficientnet_fam:
        if old_version:
            model.classifier[1].out_features = classes_num
        else:
            n_feats: int = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(n_feats, classes_num)
    else:
        raise Exception("Unrecognized model architecture")

    return model


def model_loader(config: dict):
    #######################################################################
    # config sample

    # model_config = {
    #     'model': 'vgg19',
    #     'classes_num': 2,
    #     'use_official_model': True,
    #     'pretrained': False,
    #     'pretrained_model': ''
    # }
    ########################################################################

    classes_num = config['classes_num']

    # get model name
    if 'model' in config:
        model_name: str = config['model']

        if 'use_official_model' in config:
            if config['use_official_model']:
                model = models.get_model(
                    name=model_name,
                    weights=models.get_model_weights(model_name).DEFAULT,
                )
                model = set_classes_num(
                    model,
                    model_name,
                    classes_num,
                    old_version=config.get('old_version', False),
                )

                return model
            else:
                if 'pretrained' in config:
                    if config['pretrained']:
                        model = models.get_model(
                            name=model_name,
                            weights=models.get_model_weights(model_name).DEFAULT,
                        )
                        model = set_classes_num(
                            model,
                            model_name,
                            classes_num,
                            old_version=config.get('old_version', False),
                        )
                        model.load_state_dict(torch.load(config['pretrained_model']))

                        return model
    else:
        print('You need to define a model!!')


def checkpoint_loader(config: dict):
    #######################################################################
    # config sample

    # model_config = {
    #     'model': 'vgg19',
    #     'classes_num': 2,
    #     'use_official_model': True,
    #     'custom_checkpoint': False,
    #     'pretrained': False,
    #     'pretrained_model': ''
    # }
    ########################################################################

    if (config['pretrained'] == False):
        model = model_loader(config)

        checkpoint = {'model': model,
                    'state_dict': model.state_dict(),
                    'pruning_percentage': 0,
                    'iteration': 0,
                    'pruning_step': 0}
        return checkpoint
    else:
        checkpoint = torch.load(config['pretrained_model'])
        return checkpoint


def merge_dicts(dict1, dict2):
    merged_dict = dict(dict1)
    for key, value in dict2.items():
        if key in merged_dict and isinstance(merged_dict[key], dict) and isinstance(value, dict):
            merged_dict[key] = merge_dicts(merged_dict[key], value)
        else:
            merged_dict[key] = value
    return merged_dict


# todo: change folder-name to variable?
def save_wrong_images(predict_results, data_loader, target, class_names):
    # get date and current time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # create folder
    folder_name = f"Output/test_{current_time}"
    os.mkdir(folder_name)

    for i, (image, label) in enumerate(data_loader):
        predict_result = predict_results[i]

        if label == target and predict_result != target:
            img_name = f"{class_names[target]}_to_{class_names[predict_result]}_{i}.jpg"
            image = image.squeeze(0)
            image_path = os.path.join(folder_name, img_name)
            transforms.ToPILImage()(image).save(image_path)
        elif predict_result == target and label != target:
            img_name = f"{class_names[label]}_to_{class_names[predict_result]}_{i}.jpg"
            image = image.squeeze(0)
            image_path = os.path.join(folder_name, img_name)
            transforms.ToPILImage()(image).save(image_path)


def save_checkpoint(model, config, it, device, pruning_percentage=0, pruning_step=0):
    save_config = config['save_config']

    model_save_name = save_config['save_name']
    model_save_path = save_config['save_path']

    pruning_percentage_str = str(pruning_percentage)
    pruning_percentage_str_without_dot = pruning_percentage_str.replace(".", "")

    if (device=='cuda: 0'):
        model.to('cpu')
    
    # Generate checkpoint
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'pruning_percentage': pruning_percentage,
                  'iteration': it,
                  'pruning_step': pruning_step}

    # Save checkpoint
    if save_config['save_per_epoch']:
        checkpoint_save_name = model_save_name + "_" + str(it) + "_" + pruning_percentage_str_without_dot + "_" + str(pruning_step) + ".pt"
        checkpoint_save_path = model_save_path + checkpoint_save_name

        torch.save(checkpoint, checkpoint_save_path)
    else:
        checkpoint_save_name = model_save_name + "_" + pruning_percentage_str_without_dot + ".pt"
        checkpoint_save_path = model_save_path + checkpoint_save_name

        torch.save(checkpoint, checkpoint_save_path)
