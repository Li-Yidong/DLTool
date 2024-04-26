config = {
    # choose mode
    'mode': {
        'train': False,
        'test': False,
        'pruning': True
    },

    # pruning configuration dictionary
    'pruning_config': {
        # save configuration dictionary
        'save_config': {
            'save_per_epoch': False,
            'save_onnx_per_epoch': True,
            'save_path': 'Output/checkpoints/',
            'save_name': 'prune_test'
        },

        # model config
        'model_config': {
            'pretrained_model': 'checkpoints/pt_checkpoints/Yutaka_230522_config_file.pt'
        },

        # pruner configration dictionary
        'pruner_config': {
            'iterative_steps': 8,
        },

        'device': 'cuda:0',
    }

}
