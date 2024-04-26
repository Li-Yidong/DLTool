"""app/utils/run.py"""
import sys

sys.path.append("../../../")
from tools.launch import launch


class CustomLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        pass


def run(config: dict) -> None:
    save_path: str = ""
    save_name: str = ""
    if config["mode"]["train"] or config["mode"]["test"]:
        save_path = config["train_config"]["save_config"]["save_path"]
        save_name = config["train_config"]["save_config"]["save_name"]
    elif config["mode"]["pruning"]:
        save_path = config["pruning_config"]["save_config"]["save_path"]
        save_name = config["pruning_config"]["save_config"]["save_name"]
    elif config["mode"]["export"]:
        save_path = "./Outputs/"
        save_name = "export"
    elif config["mode"]["heatmap"]:
        return "heatmap"
    else:
        raise Exception("No mode chosen for running the file")
    log_file = CustomLogger(f"{save_path}{save_name}.out")
    sys.stdout = log_file
    try:
        launch(config)
    finally:
        sys.stdout = log_file.terminal
        log_file.log_file.close()
