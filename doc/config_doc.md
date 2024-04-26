# Configuration Dictionary

## 1. 学習に関するキーワード及び説明

学習の部分に関するデフォルト配置は以下になる：

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

主に作業工程で`save_config`, `dataset_config`, `model_config`及び`Hyperparams_config`で分けられる.

### 1.1 `save_config`

- 主な使い方

```
'save_config': {
    'save_per_epoch': True,
    'save_path': 'Output/checkpoints/',
    'save_name': 'test'
},
```

- 説明

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|save_per_epoch|True/False            |`True`だとモデルが`1epoch`ずつ保存される. `False`だと最後の`epoch`のモデルだけ保存される.|
|save_path     |{Your model save path}|モデルの保存場所である.|
|save_name     |{Your model save name}|モデルの保存ネームである.|

### 1.2 `dataset_config`

- 主な使い方
  
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
|data          |{Your train dataset path}|教師データ        |
|batch_size    |32                    |一回でデータから取る画像数，GPUのメモリーによる変換する．数字が大きくなると学習スピードも速くなる|
|shuffle       |True/False            |`True`だとランダムでデータセットからデータを取る．`False`だと順番で取る|
|Resize        |[224, 224]            |画像を指定されたサイズでResizeする．|
|RandomRotation |5                    |取られた画像をランダムに`5`度でRotateする．|
|ColorJitter   |`brightness`: .5, `hue`: .3 | `brightness`: 画像の明るさをランダムに元画像の1 +- 50%に変化する(50% ~ 150%)．`hue`: 画像の色相をランダムに元画像の1 +- 30%に変化する(70% ~ 130%)．|
|RandomInvert  |True/False            |`True`: 画像をランダムにカラーリバーサルする．|
|RandomHorizontalFlip |True/False     |`True`: 画像をランダムに左右フリップする. |
|RandomVerticalFlip   |True/False     |`True`: 画像をランダムに上下フリップする. |

- `val_data_config`

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|data          |{Your validation dataset path}|バリデーションデータ        |
|batch_size    |32                    |一回でデータから取る画像数，GPUのメモリーによる変換する．数字が大きくなると学習スピードも速くなる|
|shuffle       |True/False            |`True`だとランダムでデータセットからデータを取る．`False`だと順番で取る|

### 1.3 `model_config`

- 主な使い方

```
'model_config': {
    'model': 'resnet18',
    'classes_num': 2,
    'use_official_model': True,
    'pretrained': False,
    'pretrained_model': ''
},
```

- 説明

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|model         |{Your modual name}    |本パッケージのモジュールが全部`Pytorch`で提供された.使えモジュールは以下となる：**`VGG`**:`vgg16`, `vgg19`; **`ResNet`**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`; **`EfficientNet`**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7` |
|classes_num  |{Your classes number}  |教師データのクラス数が同じすべき|
|use_official_model |True/False      |`True`: `Pytorch`が提供するモデルを使用する．|
|pretrained   |True/False            |`True`: 自分の転移学習用のモデルを使用する． |
|pretrained_model | {Your pretrained model} | 自分の転移学習用のモデルを使用する場合，モデルパスを渡さないといけない|

### 1.4 `Hyperparams_config`

- 主な使い方

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

- 説明

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|epochs       |300                   |学習回数            |
|criterion    |CrossEntropyLoss      |損失関数. 選択できる損失関数: `CrossEntropyLoss`, `BCEWithLogitsLoss`, `BCELoss`, `NLLLoss` |
|optimizer    |SGD                   |学習損失を抑えるために，フィードバック最適化手法である. 選択できる手法：`SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad` |
|learning_rate |0.01                 |学習間隔である. 小さくなるほど学習が細かくなる．
|momentum     |0.9                   |`SGD`専用パラメータ：関数平面上をボールが転がるように最適化されます|
|target       |0　　　　　　　　　　　 |注目したいクラスのインデックス．そのインデックスによるRecallとPrecisionを計算する．|


## 2. モデルの検証に関するキーワード及び説明

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

主に作業工程で`test_mode`, `dataset_config`及び`model_config`で分けられる.

### 2.1 `test_mode`

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|pt_model_test |True / False          |`True`: テストしたいモデルは`.pt`モデル|
|onnx_model_test |True / False          |`True`: テストしたいモデルは`.onnx`モデル|

>注意：この２つのValueを同時に`True`か`False`にならないようにしてください.

### 2.2 `dataset_config`

- 主な使い方

```
'dataset_config': {
    'data': 'datasets/Assy_CNN_2classes/val',
    'batch_size': 1,
    'shuffle': False,
    'Resize': [224, 224]
},
```

- 説明

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|data          |{Your test dataset path} |テストしたいデータセット|
|batch_size    |1                     |`.pt`モデル：GPUかRAMで変化していい；`.onnx`モデル：`input_size`に対応してください.|
|shuffle       |True / False          |`True`だとランダムでデータセットからデータを取る．`False`だと順番で取る|
|Resize        |[224, 224]            |モデルを訓練された時の入力サイズと同様|

### 2.3 `model_config`

- 主な使い方

```
'model_config': {
    'model': 'resnet18',
    'classes_num': 2,
    'use_official_model': False,
    'pretrained': True,
    'pretrained_model': ''
},
```

- 説明

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|model         |'resnet18'            |`.pt`モデルを訓練された時と同様(`.pt`モデル専用)|
|classes_num   |2                     |`.pt`モデルを訓練された時と同様(`.pt`モデル専用)|
|use_official_model | False           |絶対`False`         |
|pretrained    |True                  |絶対`True`          |
|pretrained_model | {Your model path} |検証したいモデルのパス|

### 2.4 他のパラメータ

- 主な使い方

```
'device': 'cpu',

'target': 0,
'wrong_image_save': True,
'class_names': ['NG', 'OK']
```

- 説明

|Key           |Value                 |Description         |
| -------------|----------------------|--------------------|
|device        |cpu / cuda:0          |使いたい設備         |
|target        |0                     |着目したいクラスのインデックス，このインデックスによるRecallとPresicionを計算する|
|wrong_image_save |True / False       |`True`: 設置された`target`による正解と違う判断された画像を保存する. |
|class_names  |['NG', 'OK']           |テストされるデータセットのクラス分け|

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

[1.2](#12-dataset_config)と同様

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
|use_official_model | False           |絶対 `False`         |
|pretrained    |True                  |絶対 `True`          |
|pretrained_model | {Your model path} |Path of the model for pruning|


### 3.4 `Hyperparams_config`

[1.4](#14-hyperparams_config)と同様

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