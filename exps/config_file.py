config = {
    # choose mode
    'mode': {
        'train': True,
        'test': False,
        'pruning': False
    },

    # train configuration dictionary
    'train_config': {
        'device': 'cpu',

        # dataset configuration dictionary
        'dataset_config': {
            'batch_size': 8
        },
    },


    # test configuration dictionary
    'test_config': {
        'test_mode': {
            'pt_model_test': True,
            'onnx_model_test': False
        },

        # dataset configuration dictionary
        'dataset_config': {
            'data': 'datasets/Assy_CNN_2classes/test',
            'Resize': [224, 224]
        },

        # model config
        'model_config': {
            'model': 'resnet18',
            'classes_num': 2,
            'pretrained_model': 'checkpoints/pt_checkpoints/Yutaka_230522_config_file.pt'
        },

        'device': 'cuda:0',

        'target': 0,
        'wrong_image_save': True,
        'class_names': ['NG', 'OK']
    }
}














