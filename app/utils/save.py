"""app/utils/save.py"""
import json


def save_config(path: str, data: dict):
    if not path:
        return
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
