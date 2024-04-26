from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, 
                             QHBoxLayout,
                             QSplitter,
                             QListWidget,
                             QGraphicsView,
                             QGraphicsScene)
from PyQt6.QtGui import QPixmap, QImage
from ..utils.config import AppConfig
import os
from ..utils.view_heatmap import view_heatmap
from tools import default_config
from tools import utils

class HeatmapViewerWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        
        self.setWindowTitle("Heatmap Viewer")
        self.setFixedSize(AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)

        self.layout = QHBoxLayout(self)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)

        # Left side is a list widget
        self.left_widget = QListWidget(parent=splitter)
        self.left_widget.setObjectName('left_widget')
        
        # Left widget style
        self.left_widget.setContentsMargins(5, 5, 5, 5)
        self.left_widget.setStyleSheet("QWidget#left_widget{border: 1px solid black}")

        # Get file list
        self.left_widget.itemDoubleClicked.connect(self.show_image)

        # Right side is a image viewer
        self.right_widget = QGraphicsView(parent=splitter)
        self.right_widget.setObjectName('right_widget')
        self.right_widget.setContentsMargins(5, 5, 5, 5)
        self.right_widget.setStyleSheet("QWidget#right_widget{border: 1px solid black}")

        splitter.addWidget(self.left_widget)
        splitter.addWidget(self.right_widget)

        splitter.setSizes([AppConfig.WINDOW_WIDTH * 0.2, AppConfig.WINDOW_WIDTH * 0.8])

        self.layout.addWidget(splitter)

    def get_images(self, image_path: str):
        image_file_list: list = []
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_file_list.append(file)
                    print(file)
                    

        return image_file_list
    

    def show_list(self, config: dict):
        self.configuration = config["heatmapViewer"]
        # Add file list
        self.image_path = self.configuration['image_path']
        image_file_list = self.get_images(self.image_path)
        self.left_widget.addItems(image_file_list)


    def show_image(self):
        # Get the current item
        self.item = self.left_widget.currentItem()

        if self.item is not None:
            # Generate heatmap
            image_name = self.item.text()
            image_path = os.path.join(self.image_path, image_name)
            self.configuration["image_path"] = image_path
            
            # Generate heatmap
            heatmap = view_heatmap(self.configuration)

            # Get the dimensions of the heatmap
            x = heatmap.shape[1]
            y = heatmap.shape[0]

            # Convert the heatmap into a QImage format
            frame = QImage(heatmap, x, y, QImage.Format.Format_RGB888)
            # Create a QPixmap from the QImage
            pix = QPixmap.fromImage(frame)

            # Create a new graphics scene and add the QPixmap to the scene
            scene = QGraphicsScene()
            scene.addPixmap(pix)
            # Update the graphics scene displayed in the right-side window
            self.right_widget.setScene(scene)
        