from PyQt5 import QtCore, QtWidgets
import os
from pathlib import Path


class ImageButton(QtWidgets.QPushButton):
    def __init__(self, icon_path, parent=None):
        super().__init__(parent)
        icon_path = icon_path.replace('\\', '/')
        self.setStyleSheet(f"""
            QPushButton {{
                border: none; 
                border-image: url({icon_path}) 0 0 0 0 stretch stretch;
            }}
            QPushButton:hover {{
                border-image: url({icon_path}) 0 0 0 0 stretch stretch;
                border: 3px solid #999999;
            }}
        """)


class CardWidget(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal()

    def __init__(self, title, image_path: Path, parent=None):
        super(CardWidget, self).__init__(parent)
        # creating a push button
        path = os.path.normpath(image_path)
        print(path)
        button = ImageButton(path, self)
        # Set button size policy
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        # Set layout for CardWidget
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(button)

        # Connect button to signal
        button.clicked.connect(self.clicked)

    def resizeEvent(self, event):
        # Resize the widget to keep the aspect ratio of the image
        width = self.width()
        new_height = width * 64 // 177  # Adjust height based on width
        self.resize(width, int(new_height))


class ChooseAppDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget=None):
        super(ChooseAppDialog, self).__init__(parent)
        self.setWindowTitle("Choose an application")

        hbox = QtWidgets.QHBoxLayout()
        file_path = Path(os.path.abspath(os.path.dirname(__file__)))
        demo_image_path = file_path / "demo_card.png"
        viewer_image_path = file_path / "viewer_card.png"
        demo_card = CardWidget("DepthAI Demo", demo_image_path)
        viewer_card = CardWidget("DepthAI Viewer", viewer_image_path)
        hbox.addWidget(demo_card)
        hbox.addWidget(viewer_card)
        self.setLayout(hbox)

        demo_card.clicked.connect(self.runDemo)
        viewer_card.clicked.connect(self.runViewer)

        # Get screen dimensions
        screen = QtWidgets.QApplication.instance().primaryScreen()
        screen_size = screen.size()
        self.resize(screen_size.width() // 2, screen_size.height() // 4)

    @QtCore.pyqtSlot()
    def runDemo(self):
        self.reject()

    @QtCore.pyqtSlot()
    def runViewer(self):
        self.accept()
