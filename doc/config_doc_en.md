# Configuration Dictionary

## 1. Keywords and explanations related to training

The default configuration for the training setion is as follows: 

```
# train configuration dictionary
    'train_config': {
        # device
        'device': 'cuda:0',
        'graph_mode': True,

        # save mode
        'save_config': {
            'save_per_epoch': True,
            'save_path': 'Output/checkpoints/',
            'save_name': 'test'
        },

        # dataset configuration dictionary
        'dataset_config': {
            'train_data_config': {
                'data': 'datasets/Assy_CNN_2classes/train',
                'batch_size': 32,
                'shuffle': True,

                # data augmentation
                'Resize': [224, 224],
                'RandomRotation': 5,
                'ColorJitter': {
                    'brightness': .5,
                    'hue': .3
                },
                'RandomInvert': True,
                'RandomHorizontalFlip': True,
                'RandomVerticalFlip': True
            },
            'val_data_config': {
                'data': 'datasets/Assy_CNN_2classes/val',
                'batch_size': 32,
                'shuffle': False,
            }
        },

        # model config
        'model_config': {
            'model': 'resnet18',
            'classes_num': 2,
            'use_official_model': True,
            'pretrained': False,
            'pretrained_model': ''
        },

        # Hyperparameters configuration dictionary
        'Hyperparams_config': {
            'epochs': 300,
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'target': 0
        }

    },
```

### 1.1 `save_config`

- Default: 

```
'save_config': {
    'save_per_epoch': True,
    'save_path': 'Output/checkpoints/',
    'save_name': 'test'
},
```

- Description

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|save_per_epoch|True/False            |`True`: Model will be save after each epoch; `False`: Only save the model of last epoch|
|save_path     |{Your model save path}|Location of saving the model|
|save_name     |{Your model save name}|Saving name of model|

### 1.2 `dataset_config`

- Default: 
  
```
'dataset_config': {
    'train_data_config': {
        'data': 'datasets/Assy_CNN_2classes/train',
        'batch_size': 32,
        'shuffle': True,

        # data augmentation
        'Resize': [224, 224],
        'RandomRotation': 5,
        'ColorJitter': {
            'brightness': .5,
            'hue': .3
        },
        'RandomInvert': True,
        'RandomHorizontalFlip': True,
        'RandomVerticalFlip': True
    },
    'val_data_config': {
        'data': 'datasets/Assy_CNN_2classes/val',
        'batch_size': 32,
        'shuffle': False,
    }
}
```

- `train_data_config`

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|data          |{Your train dataset path}|Training data        |
|batch_size    |32                    |The number of images taken from one iteration. A lager number will result in faster training speed|
|shuffle       |True/False            |`True`: Images taken by randomly. |
|Resize        |[224, 224]            |Resize the images to the specified size|
|RandomRotation |5                    |Randomly rotate the captured images by 5 degrees|
|ColorJitter   |`brightness`: .5, `hue`: .3 | `brightness`: Randomly convert the taken images to 50% ~ 150% brightness; `hue`: Randomly change the hue of the images to 1 ± 30% of the original image (70% to 130%).|
|RandomInvert  |True/False            |`True`: Randomly invert the color|
|RandomHorizontalFlip |True/False     |`True`: Randomly flip the images horizontally|
|RandomVerticalFlip   |True/False     |`True`: Randomly flip the images vertically (upside down).|

- `val_data_config`

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|data          |{Your validation dataset path}|Validation data    |
|batch_size    |32                    |The number of images taken from one iteration. A lager number will result in faster training speed|
|shuffle       |True/False            |`True`: Images taken by randomly. |

### 1.3 `model_config`

- Default: 

```
'model_config': {
    'model': 'resnet18',
    'classes_num': 2,
    'use_official_model': True,
    'pretrained': False,
    'pretrained_model': ''
},
```

- Intruduction: 

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|model         |{Your modual name}    |All modules in this package are provided by `PyTorch` . There are modules availeble：**`VGG`**:`vgg16`, `vgg19`; **`ResNet`**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`; **`EfficientNet`**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7` |
|classes_num  |{Your classes number}  |It must be same with train data|
|use_official_model |True/False      |`True`: Use the pretrained model provided by `PyTorch` do transform learning|
|pretrained   |True/False            |`True`: Use the pretrained model by ourselves do transform learning |
|pretrained_model | {Your pretrained model} | If you do transform learning using your own pretrained model, you should add model path. |

### 1.4 `Hyperparams_config`

- Default

```
'Hyperparams_config': {
    'epochs': 300,
    'criterion': 'CrossEntropyLoss',
    'optimizer': 'SGD',
    'learning_rate': 0.01,
    'momentum': 0.9,
    'target': 0
}
```

- Intruduction

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|epochs       |300                   |Iterations of trainning   |
|criterion    |CrossEntropyLoss      |Loss function. Options can choose: `CrossEntropyLoss`, `BCEWithLogitsLoss`, `BCELoss`, `NLLLoss` |
|optimizer    |SGD                   |To reducing the loss of trainning, optimize the model by feedback. Methods can choose：`SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad` |
|learning_rate |0.01                 |Determines how small or large the updates to the model's parameters are during training. Smaller learning rate results in finer updates, while larger learning rate leads to more significant updates. 
|momentum     |0.9                   |`SGD` specific parameter: It optimizes the parameters as if a ball is rolling on the function's surface.|
|target       |0　　　　　　　　　　　 |Index of the class of interest. Recall and Precision will be calculated based on this index.|


## 2. Keywords and explanations related to model validation

```
    'test_config': {
        'test_mode': {
            'pt_model_test': True,
            'onnx_model_test': False
        },

        # dataset configuration dictionary
        'dataset_config': {
            'data': 'datasets/Assy_CNN_2classes/val',
            'batch_size': 1,
            'shuffle': False,
            'Resize': [224, 224]
        },

        # model config
        'model_config': {
            'model': 'resnet18',
            'classes_num': 2,
            'use_official_model': False,
            'pretrained': True,
            'pretrained_model': ''
        },

        'device': 'cpu',

        'target': 0,
        'wrong_image_save': True,
        'class_names': ['NG', 'OK']
    },
```

### 2.1 `test_mode`

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|pt_model_test |True / False          |`True`: Test `.pt` model|
|onnx_model_test |True / False          |`True`: Test `.onnx` model|

>Note: Please ensure that these two values do not simutaneously become `True` and `False`. 

### 2.2 `dataset_config`

- Default

```
'dataset_config': {
    'data': 'datasets/Assy_CNN_2classes/val',
    'batch_size': 1,
    'shuffle': False,
    'Resize': [224, 224]
},
```

- Intruduction

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|data          |{Your test dataset path} |テストしたいデータセット|
|batch_size    |1                     |`.pt` model：Adjust this value by RAM and GPU；`.onnx` model： Please ensure this value is same as the input size with model|
|shuffle       |True / False          |`True`: Images taken by randomly. |
|Resize        |[224, 224]            |Same size with you train the model|

### 2.3 `model_config`

- Default:

```
'model_config': {
    'model': 'resnet18',
    'classes_num': 2,
    'use_official_model': False,
    'pretrained': True,
    'pretrained_model': ''
},
```

- Intruduction

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|model         |'resnet18'            |Same module with `.pt` model(Only for `.pt` model)|
|classes_num   |2                     |Same module with `.pt` model(Only for `.pt` model)|
|use_official_model | False           |Definitly `False`         |
|pretrained    |True                  |Definitly `True`          |
|pretrained_model | {Your model path} |Path of the model for testing|

### 2.4 The others

- Default:

```
'device': 'cpu',

'target': 0,
'wrong_image_save': True,
'class_names': ['NG', 'OK']
```

- Intruduction

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|device        |cpu / cuda:0          |Set device to do process |
|target        |0                     |Index of the class of interest. Recall and Precision will be calculated based on this index.|
|wrong_image_save |True / False       |`True`: Save wrong images of inference|
|class_names  |['NG', 'OK']           |Classes of dataset to be tested|

## 3. Keywords and explanations related to model pruning

```
'pruning_config': {
        # save configuration dictionary
        'save_config': {
            'save_per_epoch': False,
            'save_onnx_per_epoch': True,
            'save_path': 'Output/checkpoints/',
            'save_name': 'prune_test',
            'input_shape': (1, 3, 224, 224)
        },

        # dataset configuration dictionary
        'dataset_config': {
            'train_data_config': {
                'data': 'datasets/Assy_CNN_2classes/train',
                'batch_size': 32,
                'shuffle': True,

                # data augmentation
                'Resize': [224, 224],
                'RandomRotation': 5,
                'ColorJitter': {
                    'brightness': .5,
                    'hue': .3
                },
                'RandomInvert': True,
                'RandomHorizontalFlip': True,
                'RandomVerticalFlip': True
            },
            'val_data_config': {
                'data': 'datasets/Assy_CNN_2classes/val',
                'batch_size': 32,
                'shuffle': False,
            }
        },

        # model config
        'model_config': {
            'model': 'resnet18',
            'classes_num': 2,
            'use_official_model': False,
            'pretrained': True,
            'pretrained_model': ''
        },

        # Hyperparameters configuration dictionary
        'Hyperparams_config': {
            'epochs': 10,
            'last_epochs': 20,
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'target': 0,

        },

        # pruner configration dictionary
        'pruner_config': {
            'input_shape': (1, 3, 224, 224),
            'features_importance': 'MagnitudeImportance',
            'pruner': 'MagnitudePruner',
            'iterative_steps': 1,
            'channel_sparsity': 0.8,
        },

        'device': 'cuda:0',
    }
```

### 3.1 `save_config`

- Default: 

```
'save_config': {
    'save_per_epoch': False,
    'save_onnx_per_epoch': True,
    'save_path': 'Output/checkpoints/',
    'save_name': 'prune_test',
    'input_shape': (1, 3, 224, 224)
},
```

- Description

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|save_per_epoch|True/False            |`True`: Model will be save after each epoch; `False`: Only save the model of last epoch|
|save_onnx_per_epoch|True/False            |`True`: .onnx Model will be save after each epoch; `False`: Only save the .onnx model of last epoch|
|save_path     |{Your model save path}|Location of saving the model|
|save_name     |{Your model save name}|File name of model|
|input_shape     |{Your model input shape}|Your `.onnx` model's input shape, should be same as `.pt` model's input shape|


### 3.2 `dataset_config`

same as [1.2](#12-dataset_config)

### 3.3 `model_config`

- Default: 

```
'model_config': {
    'model': 'resnet18',
    'classes_num': 2,
    'use_official_model': False,
    'pretrained': True,
    'pretrained_model': ''
},
```

- Description

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|model         |'resnet18'            |Same module with `.pt` model(Only for `.pt` model)|
|classes_num   |2                     |Same module with `.pt` model(Only for `.pt` model)|
|use_official_model | False           |Definitly `False`         |
|pretrained    |True                  |Definitly `True`          |
|pretrained_model | {Your model path} |Path of the model for pruning|


### 3.4 `Hyperparams_config`

same as [1.4](#14-hyperparams_config)

### 3.5 `pruner_config`

- Default: 

```
'pruner_config': {
    'input_shape': (1, 3, 224, 224),
    'features_importance': 'MagnitudeImportance',
    'pruner': 'MagnitudePruner',
    'iterative_steps': 1,
    'channel_sparsity': 0.8,
}
```

- Discreption: 

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|input_shape   |(B, C, H, W)          |Dummy input for claculating the feature inportance. It should be the same as the input shape of youe saved model|
|features_importance | MagnitudeImportance | The way to calculate the model features' importance. There are several ways you can choose from: `TaylorImportance`, `MagnitudeImportance`, `BNScaleImportance`, `RandomImportance`. For details, please refer to PyTorch's documentation. |
|iterative_steps | {Your pruning steps}| The iteration steps you use to achieve the desired separation. |
|channel_sparsity | {Your desired separation} | The desired separation |


### 3.6 The others

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|device        |cuda:0/cpu            |Using a device to train. |
