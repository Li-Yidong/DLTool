config = {
    # choose mode
    'mode': {
        'train': False,
        'test': True,
        'pruning': False
    },

    # test configuration dictionary
    'test_config': {
        'test_mode': {
            'pt_model_test': False,
            'onnx_model_test': True
        },

        # dataset configuration dictionary
        'dataset_config': {
            'data': 'datasets/Assy_CNN_2classes/test',
            'Resize': [224, 224]
        },

        # model config
        'model_config': {
            'pretrained_model': 'checkpoints/onnx_checkpoints/Yutaka_230522_config_file.onnx'
        },

        'device': 'cpu',

        'target': 0,
        'wrong_image_save': True,
        'class_names': ['NG', 'OK']
    }
}














