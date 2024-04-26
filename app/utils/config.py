"""app/utils/config.py"""
import os
import json
from dotenv import load_dotenv
from configparser import ConfigParser
from typing import List, Dict

load_dotenv()


class AppConfig:
    """
    Configuration File
    """

    APP_NAME: List[str] = []
    APP_VERSION: str = ""
    APP_TITLE: List[str] = []
    WINDOW_WIDTH: int = None
    WINDOW_HEIGHT: int = None
    CONFIG_UI: dict = {}
    LANG_MAP: Dict[str, int] = {
        "EN": 0,
        "JP": 1,
    }
    CUR_LANG: int = None

    @classmethod
    def initialize(cls) -> None:
        """
        Performs initialization
        - Loads settings from a file
        """
        ui_path, ui_path_abs = os.getenv("PATH_UI"), os.getenv(
            "PATH_UI_ABS"
        )
        cfg_path, cfg_path_abs = os.getenv("PATH_CFG"), os.getenv(
            "PATH_CFG_ABS"
        )

        try:
            with open(
                ui_path if ui_path else ui_path_abs, "r", encoding="utf-8"
            ) as file:
                cls.CONFIG_UI = json.load(file)
        except Exception as e:
            raise Exception("UI file not found") from e

        cfg: ConfigParser = ConfigParser()
        configs: List[str] = cfg.read([cfg_path, cfg_path_abs], encoding="utf-8")
        if not configs:
            raise Exception("Config file not found") from FileNotFoundError()
        else:
            cls.APP_NAME = json.loads(cfg.get("app", "name"))
            cls.APP_VERSION = cfg.get("app", "version")
            cls.APP_TITLE = json.loads(cfg.get("app", "title"))
            cls.WINDOW_WIDTH = int(cfg.get("window", "width"))
            cls.WINDOW_HEIGHT = int(cfg.get("window", "height"))
            cls.CUR_LANG = cls.LANG_MAP.get(cfg.get("language", "gui_lang"), 0)

    @classmethod
    def save_config_ini(cls) -> None:
        cfg_path, cfg_path_abs = os.getenv("PATH_CFG", default=""), os.getenv(
            "PATH_CFG_ABS", default=""
        )
        paths: List[str] = [cfg_path, cfg_path_abs]
        cfg: ConfigParser = ConfigParser()
        configs: List[str] = cfg.read(paths, encoding="utf-8")
        for i, c in enumerate(configs):
            cfg["language"]["gui_lang"] = {v: k for k, v in cls.LANG_MAP.items()}[
                cls.CUR_LANG
            ]
            if paths[i]:
                with open(paths[i], 'w', encoding="utf-8") as file:
                    cfg.write(file)
