# configuration dictionary
config = {
    # choose mode
    'mode': {
        'train': True,
        'test': False,
        'pruning': False
    },

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

    # test configuration dictionary
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

    # pruning configuration dictionary
    'pruning_config': {
        # save configuration dictionary
        'save_config': {
            'save_per_epoch': False,
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
    },
    
    # heatmap viewer configuration dictionary
    'heatmapViewer': {
        'image_path': '',
        'cam': 'GradCAM',
        'classes_num': 2,
        'input_shape': [224, 224],
    },
    
    # Export configuration dictionary
    'exportInferenceModel': {
        'input_model_config': {
            'model_version': 'DLTool',
            'input_model_path': '',
            'model': "resnet18",
            'classes_num': 2,
        },
        'output_model_config': {
            'output_model_path': '',
            'input_shape': (1, 3, 224, 224)
        }
    }
}


