from tools import train
from tools import test
from tools import utils
from tools import default_config
from tools import pruning_tool
from tools import export_inference_model
from tools import launch_heatmap_viewer


def launch(customize_config):
    # default configuration
    config = default_config.config
    # merge customize configuration with default configuration
    merged_config = utils.merge_dicts(config, customize_config)

    # launch mode
    mode = merged_config['mode']

    # choose launch function
    if mode['train']:
        train.train_checkpoint(merged_config['train_config'])
    elif mode['test']:
        test_config = merged_config['test_config']
        test_mode = test_config['test_mode']
        if test_mode['pt_model_test']:
            test.test_pt_checkpoint(test_config)
        elif test_mode['onnx_model_test']:
            test.test_onnx(test_config)
    elif mode['pruning']:
        pruning_tool.pruning(merged_config['pruning_config'])
    elif mode['export']:
        export_inference_model.run_export(merged_config['exportInferenceModel'])
