from tools import launch
from exps import config_file
from exps import prune_config
# from exps import onnx_test
# from exps import Towa_fingerprint
# from exps import snapring_burr_config_file
# from exps import snapring_burr_prune_config
from exps.Yutaka_Assy import Bite_obejects_0731


# config = config_file.config
# config = prune_config.config
# config = Bite_obejects_0714.config

# Connector Assy
config = Bite_obejects_0731.config

# snapring_burr
# config = snapring_burr_config_file.config
# config = snapring_burr_prune_config.config

launch.launch(config)

