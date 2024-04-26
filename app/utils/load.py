"""app/utils/load.py"""
import json


# def have_same_keys(dict_a: dict, dict_b: dict) -> bool:
#     if set(dict_a.keys()) != set(dict_b.keys()):
#         return False
#     else:
#         for key_a, val_a in dict_a.items():
#             if type(val_a) == type(dict_b[key_a]):
#                 # ? check list length, type of element
#                 if isinstance(val_a, dict) and not have_same_keys(val_a, dict_b[key_a]):
#                     return False
#             else:
#                 return False
#         return True


# def load_config(path: str, config_w: dict) -> dict:
#     load_json: dict
#     with open(path, "r", encoding="utf-8") as f:
#         load_json = json.load(f)
#     if not have_same_keys(config_w, load_json):
#         print("Cannot read file because it has different structures")  # ? raise error
#         return config_w
#     return load_json


def load_config(path: str, config_w: dict) -> dict:
    load_json: dict
    with open(path, "r", encoding="utf-8") as f:
        load_json = json.load(f)

    merged_config = {}

    for key, value in config_w.items():
        if key in load_json:
            if isinstance(value, dict) and isinstance(load_json[key], dict):
                merged_config[key] = load_config_recursive(value, load_json[key])
            else:
                merged_config[key] = load_json[key]

    print("Merged config successfully!!")
    return merged_config


def load_config_recursive(config_w: dict, load_json: dict) -> dict:
    merged_config = {}
    for key, value in config_w.items():
        if key in load_json:
            if isinstance(value, dict) and isinstance(load_json[key], dict):
                merged_config[key] = load_config_recursive(value, load_json[key])
            else:
                merged_config[key] = load_json[key]
    return merged_config