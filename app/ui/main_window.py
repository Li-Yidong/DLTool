"""app/ui/main_window.py"""
from copy import deepcopy
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QTextEdit,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QFileDialog,
    QStackedWidget,
)
from PyQt6.QtGui import QFont, QScreen
from typing import Dict, List, Any, Union

from .widgets.dropdown import Dropdown
from .widgets.path_select import PathSelect
from .widgets.input_field import InputField
from .widgets.switch_lang import SwitchLang
from ..utils.config import AppConfig
from ..utils.helper import (
    stringify_key_chain,
    dump_json,
    stringify_comment,
    print_json,
    InputTypes,
)
from ..utils.save import save_config
from ..utils.load import load_config
from ..utils.run import run
from ..ui.heatmap_view_window import HeatmapViewerWindow


VIEW_BUTTON_LABEL: List[str] = ["View", "View"]  # ? put in AppConfig
LOAD_BUTTON_LABEL: List[str] = ["Load", "Load"]
SAVE_BUTTON_LABEL: List[str] = ["Save", "Save"]
RUN_BUTTON_LABEL: List[str] = ["Run", "Run"]
SECTION_LABEL: List[str] = ["Section", "Section"]


class MainWindow(QMainWindow):
    """
    MainWindow
    """

    FONT_LEVELS: Dict[int, QFont] = {  # ? alternative to differentiate between levels
        1: lambda x: x.setStyleSheet("font-weight: 900;"),
        2: lambda x: x.setStyleSheet("font-weight: 800;"),
        3: lambda x: x.setStyleSheet("font-weight: 700;"),
    }

    def __init__(self) -> None:
        """
        Initializes the main window
        """
        super().__init__()

        self.pages_l: Dict[str, QHBoxLayout] = {}
        self.pages_w: Dict[str, QWidget] = {}
        self.nav_buttons: Dict[str, QPushButton] = {}
        self.other_buttons: Dict[str, QPushButton] = {}
        self.keys_delete: List[List[str]] = []

        # window settings
        self.setWindowTitle(
            f"{AppConfig.APP_NAME[AppConfig.CUR_LANG]} {AppConfig.APP_VERSION}"
        )
        self.setMinimumSize(AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)

        # move to secondary display if exists
        monitors: List[QScreen] = QScreen.virtualSiblings(self.screen())
        monitor: QScreen = monitors[-1]
        self.setGeometry(
            monitor.geometry().center().x() - AppConfig.WINDOW_WIDTH // 2,
            monitor.geometry().center().y() - AppConfig.WINDOW_HEIGHT // 2,
            AppConfig.WINDOW_WIDTH,
            AppConfig.WINDOW_HEIGHT,
        )

        # top bar
        layout_top: QHBoxLayout = QHBoxLayout()
        layout_top_text: QHBoxLayout = QHBoxLayout()
        layout_top_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout_top_buttons: QHBoxLayout = QHBoxLayout()
        layout_top_buttons.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.top_text: QLabel = QLabel(AppConfig.APP_TITLE[AppConfig.CUR_LANG])
        self.top_text.setStyleSheet("QLabel {font-size: 18pt;}")
        layout_top_text.addWidget(self.top_text)
        widget_top_left: QWidget = QWidget()
        widget_top_left.setLayout(layout_top_text)

        self.button_lang: SwitchLang = SwitchLang(checked=AppConfig.CUR_LANG)
        self.button_view: QPushButton = QPushButton(
            VIEW_BUTTON_LABEL[AppConfig.CUR_LANG]
        )
        self.button_save: QPushButton = QPushButton(
            SAVE_BUTTON_LABEL[AppConfig.CUR_LANG]
        )
        self.button_load: QPushButton = QPushButton(
            LOAD_BUTTON_LABEL[AppConfig.CUR_LANG]
        )
        self.button_run: QPushButton = QPushButton(RUN_BUTTON_LABEL[AppConfig.CUR_LANG])
        layout_top_buttons.addWidget(self.button_lang)
        layout_top_buttons.addWidget(self.button_view)
        layout_top_buttons.addWidget(self.button_save)
        layout_top_buttons.addWidget(self.button_load)
        layout_top_buttons.addWidget(self.button_run)
        widget_top_right: QWidget = QWidget()
        widget_top_right.setLayout(layout_top_buttons)

        # view button and view page
        key_nav_view: str = "view_working_config"
        self.nav_buttons[key_nav_view] = self.button_view
        layout_view: QVBoxLayout = QVBoxLayout()
        layout_view.setAlignment(Qt.AlignmentFlag.AlignTop)
        view_text_area: QTextEdit = QTextEdit()
        view_text_area.setReadOnly(True)
        layout_view.addWidget(view_text_area)
        self.button_view.clicked.connect(
            lambda: view_text_area.setText(dump_json(self.config_working["start"], 4))
        )
        self.button_view.clicked.connect(lambda: self.button_nav_clicked(key_nav_view))
        self.pages_l[key_nav_view] = layout_view

        layout_top.addWidget(widget_top_left)
        layout_top.addWidget(widget_top_right)

        # lang switch
        self.button_lang.switchClicked.connect(self.update_lang)

        # save button
        key_btn_save: str = "save_working_config"
        self.button_save.clicked.connect(self.button_save_clicked)
        self.other_buttons[key_btn_save] = self.button_save

        # load button
        key_btn_load: str = "load_working_config"
        self.button_load.clicked.connect(self.button_load_clicked)
        self.other_buttons[key_btn_load] = self.button_load

        # run button and run page
        key_btn_run: str = "run_launch_program"
        self.button_run.clicked.connect(lambda: self.button_run_clicked(key_btn_run))
        self.other_buttons[key_btn_run] = self.button_run

        # stacked pages widget
        self.widget_pages: QStackedWidget = QStackedWidget()

        # sidebar
        self.widget_sidebar: QWidget = QWidget()
        self.layout_sidebar = QVBoxLayout(self.widget_sidebar)
        self.layout_sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.config_default: Dict[str, dict] = dict({"start": AppConfig.CONFIG_UI})
        self.config_working: Dict[str, dict] = dict(
            {"start": deepcopy(AppConfig.CONFIG_UI)}
        )
        self.dfs_dynamic_gui(key_chain=["start"])
        for kc in self.keys_delete:
            self.delete_config_key(self.config_working, kc)
        for i, (k, l) in enumerate(self.pages_l.items()):
            widget_placeholder: QWidget = QWidget()
            widget_placeholder.setLayout(l)
            widget_scrollarea: QScrollArea = QScrollArea()
            widget_scrollarea.setWidgetResizable(True)
            widget_scrollarea.setWidget(widget_placeholder)
            widget_scrollarea.horizontalScrollBar().setEnabled(False)
            self.pages_w[k] = widget_scrollarea
            self.widget_pages.insertWidget(i, widget_scrollarea)

        # main
        layout_main: QHBoxLayout = QHBoxLayout()
        layout_main.addWidget(self.widget_sidebar)
        layout_main.addWidget(self.widget_pages)

        widget_main: QWidget = QWidget()
        widget_main.setLayout(layout_main)

        # whole
        layout_whole: QVBoxLayout = QVBoxLayout()
        widget_top: QWidget = QWidget()
        widget_top.setFixedHeight(75)
        widget_top.setLayout(layout_top)
        layout_whole.addWidget(widget_top)
        layout_whole.addWidget(widget_main)

        widget_whole: QWidget = QWidget()
        widget_whole.setLayout(layout_whole)

        self.setCentralWidget(widget_whole)
        self.heatmap_viewer_window = HeatmapViewerWindow()

    def update_config(self, dic: dict, key_chain: List[str], val: Any) -> None:
        cur: dict = dic
        for key in key_chain[:-1]:
            cur = cur[key]
        cur[key_chain[-1]] = val

    def delete_config_key(self, dic: dict, key_chain: List[str]) -> None:
        cur: dict = dic
        for key in key_chain[:-1]:
            cur = cur.get(key, {})
        del cur[key_chain[-1]]

    def dfs_dynamic_gui(self, key_chain: List[str], level: int = -2) -> None:
        cur: dict = self.config_working
        for key in key_chain:
            cur = cur.get(key, {})
        if isinstance(cur, dict) and key_chain[-1] != InputTypes.PYQT.value:
            for key in cur.keys():
                self.dfs_dynamic_gui(key_chain=[*key_chain, key], level=level + 1)
        elif key_chain[-1] == InputTypes.PYQT.value:
            if cur["type"] != InputTypes.DICT.value:
                prev_key: str = stringify_key_chain(key_chain, 1, -(1 + level))
                cur_key: str = stringify_key_chain(key_chain, 1, -1)
                if cur["type"] == InputTypes.BOOLEAN.value:
                    cur_checkbox: QCheckBox = QCheckBox(cur["name"][AppConfig.CUR_LANG])
                    cur_checkbox.setChecked(cur["default"])
                    cur_checkbox.stateChanged.connect(
                        lambda: self.update_config(
                            self.config_working,
                            key_chain[:-1],
                            cur_checkbox.isChecked(),
                        )
                    )
                    cur_checkbox.setToolTip(cur["comment"][AppConfig.CUR_LANG])
                    cur_checkbox.setObjectName(cur_key)
                    self.pages_l[prev_key].addWidget(cur_checkbox)
                elif cur["type"] == InputTypes.DROPDOWN.value:
                    cur_dropdown: Dropdown = Dropdown(
                        cur["name"][AppConfig.CUR_LANG], cur["choices"], cur["default"]
                    )
                    cur_dropdown.currentTextChanged.connect(
                        lambda: self.update_config(
                            self.config_working,
                            key_chain[:-1],
                            cur_dropdown.current_text,
                        )
                    )
                    cur_dropdown.setToolTip(cur["comment"][AppConfig.CUR_LANG])
                    cur_dropdown.setObjectName(cur_key)
                    self.pages_l[prev_key].addWidget(cur_dropdown)
                elif cur["type"] == InputTypes.PATH.value:
                    cur_dialog: PathSelect = PathSelect(
                        cur["name"][AppConfig.CUR_LANG],
                        cur["default"],
                        target=cur["target"],
                    )
                    cur_dialog.directoryEntered.connect(
                        lambda: self.update_config(
                            self.config_working, key_chain[:-1], cur_dialog.folder_path
                        )
                    )
                    cur_dialog.setToolTip(cur["comment"][AppConfig.CUR_LANG])
                    cur_dialog.setObjectName(cur_key)
                    self.pages_l[prev_key].addWidget(cur_dialog)
                else:
                    # cur['type'] == 'StrInput' | 'IntInput' | 'FltInput' |
                    #   'ShpInput' | 'SizeInput'
                    cur_input: InputField = InputField(
                        cur["name"][AppConfig.CUR_LANG],
                        cur["default"],
                        input_type=cur["type"],
                    )
                    cur_input.textChanged.connect(
                        lambda: self.update_config(
                            self.config_working, key_chain[:-1], cur_input.cur_val
                        )
                    )
                    cur_input.setToolTip(cur["comment"][AppConfig.CUR_LANG])
                    cur_input.setObjectName(cur_key)
                    self.pages_l[prev_key].addWidget(cur_input)

                self.update_config(self.config_working, key_chain[:-1], cur["default"])
            else:  # ? could differentiate between levels
                # ! could just change to key_chain[1],
                # !     but this is for if another navbar is needed
                root_key: str = stringify_key_chain(key_chain, 1, -(1 + level))
                cur_key: str = stringify_key_chain(key_chain, 1, None)
                if level == 0:
                    cur_layout: QVBoxLayout = QVBoxLayout()
                    cur_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
                    cur_button: QPushButton = QPushButton(
                        cur["name"][AppConfig.CUR_LANG]
                    )
                    cur_button.clicked.connect(
                        lambda: self.button_nav_clicked(root_key)
                    )
                    cur_button.setToolTip(cur["comment"][AppConfig.CUR_LANG])
                    cur_button.setObjectName(root_key)
                    self.nav_buttons[root_key] = cur_button
                    self.layout_sidebar.addWidget(cur_button)
                    self.pages_l[root_key] = cur_layout
                else:  # ? could add tabs
                    cur_label: QLabel = QLabel(
                        f'{cur["name"][AppConfig.CUR_LANG]} {SECTION_LABEL[AppConfig.CUR_LANG]}'
                    )
                    cur_label.setToolTip(cur["comment"][AppConfig.CUR_LANG])
                    cur_label.setObjectName(cur_key)
                    self.FONT_LEVELS[level](cur_label)
                    self.pages_l[root_key].addWidget(cur_label)
                self.keys_delete.append(key_chain)

    def dfs_update_gui(self, key_chain: List[str], level: int = -2) -> None:
        cur: dict = self.config_working
        for key in key_chain:
            cur = cur.get(key, {})
        if isinstance(cur, dict):
            for key in cur.keys():
                self.dfs_update_gui(key_chain=[*key_chain, key], level=level + 1)
        else:
            cur_key: str = stringify_key_chain(key_chain, 1, None)
            ref_widget: Union[
                QCheckBox, Dropdown, PathSelect, InputField
            ] = self.findChild(QWidget, cur_key)
            if isinstance(ref_widget, QCheckBox):
                ref_widget.setChecked(cur)
            else:
                ref_widget: Union[Dropdown, PathSelect, InputField]
                ref_widget.update_val(cur)

    def update_lang(self):
        AppConfig.CUR_LANG = int(not AppConfig.CUR_LANG)
        self.setWindowTitle(
            f"{AppConfig.APP_NAME[AppConfig.CUR_LANG]} {AppConfig.APP_VERSION}"
        )
        self.top_text.setText(AppConfig.APP_TITLE[AppConfig.CUR_LANG])
        self.button_view.setText(VIEW_BUTTON_LABEL[AppConfig.CUR_LANG])
        self.button_save.setText(SAVE_BUTTON_LABEL[AppConfig.CUR_LANG])
        self.button_load.setText(LOAD_BUTTON_LABEL[AppConfig.CUR_LANG])
        self.button_run.setText(RUN_BUTTON_LABEL[AppConfig.CUR_LANG])

        self.dfs_update_lang(key_chain=["start"])

    def dfs_update_lang(self, key_chain: List[str], level: int = -2) -> None:
        cur: dict = self.config_default
        for key in key_chain:
            cur = cur.get(key, {})
        if isinstance(cur, dict) and key_chain[-1] != InputTypes.PYQT.value:
            for key in cur.keys():
                self.dfs_update_lang(key_chain=[*key_chain, key], level=level + 1)
        elif key_chain[-1] == InputTypes.PYQT.value:
            if cur["type"] != InputTypes.DICT.value:
                cur_key: str = stringify_key_chain(key_chain, 1, -1)
                ref_widget: Union[
                    QCheckBox, Dropdown, PathSelect, InputField
                ] = self.findChild(QWidget, cur_key)
                if isinstance(ref_widget, QCheckBox):
                    ref_widget.setText(cur["name"][AppConfig.CUR_LANG])
                else:
                    ref_widget: Union[Dropdown, PathSelect, InputField]
                    ref_widget.update_language(cur["name"][AppConfig.CUR_LANG])
                ref_widget.setToolTip(cur["comment"][AppConfig.CUR_LANG])
            else:
                root_key: str = stringify_key_chain(key_chain, 1, -(1 + level))
                cur_key: str = stringify_key_chain(key_chain, 1, None)
                if level == 0:
                    ref_button: QPushButton = self.findChild(QPushButton, root_key)
                    ref_button.setText(cur["name"][AppConfig.CUR_LANG])
                    ref_button.setToolTip(cur["comment"][AppConfig.CUR_LANG])
                else:
                    ref_label: QLabel = self.findChild(QLabel, cur_key)
                    ref_label.setText(
                        f'{cur["name"][AppConfig.CUR_LANG]} {SECTION_LABEL[AppConfig.CUR_LANG]}'
                    )
                    ref_label.setToolTip(cur["comment"][AppConfig.CUR_LANG])

    def closeEvent(self, event: QEvent):
        AppConfig.save_config_ini()
        event.accept()

    def button_nav_clicked(self, key: str) -> None:
        self.widget_pages.setCurrentWidget(self.pages_w[key])
        keys = self.nav_buttons.keys()
        self.nav_buttons[key].setStyleSheet("QPushButton {background-color: yellow;}")
        for k in keys:
            if k == key:
                continue
            self.nav_buttons[k].setStyleSheet("")

    def button_save_clicked(self) -> None:
        dialog: QFileDialog = QFileDialog(self)
        fpath: str = dialog.getSaveFileName(
            None, "Create new file", filter="JSON files (*.json)"
        )[0]
        if fpath:
            save_config(fpath, self.config_working["start"])

    def button_load_clicked(self) -> None:
        dialog: QFileDialog = QFileDialog(self)
        fpath: str = dialog.getOpenFileName(
            None, "Choose .json file", filter="JSON files (*.json)"
        )[0]
        if fpath:
            self.config_working["start"] = load_config(
                fpath, self.config_working["start"]
            )
            self.dfs_update_gui(key_chain=["start"])

    def button_run_clicked(self, key: str) -> None:
        self.window().showNormal()
        self.window().showMinimized()
        buttons = [*self.nav_buttons.values(), *self.other_buttons.values()]
        for button in buttons:
            button.setDisabled(True)
            button.setStyleSheet("")
        self.other_buttons[key].setStyleSheet("QPushButton {background-color: yellow;}")
        
        status: str = ""
        try:
            status = run(self.config_working["start"])
        except Exception as e:
            raise Exception("Error when running using the config file")

        # Show heatmap viewer window
        if (status == "heatmap"):
            self.show_heatmap_viewer_window(self.config_working["start"])

        for button in buttons:
            button.setDisabled(False)
            button.setStyleSheet("")
        self.window().showNormal()

    def button_debug_clicked(self) -> None:
        print("Debug button clicked")

    def show_heatmap_viewer_window(self, config: dict):
        self.heatmap_viewer_window.show()
        self.heatmap_viewer_window.show_list(config)
        self.heatmap_viewer_window.left_widget.update()
