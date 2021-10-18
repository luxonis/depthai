# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path

from PySide6.QtCore import QStringListModel, QUrl
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QApplication


class DemoQtGui:
    def __init__(self):
        self.app = QApplication()
        self.view = QQuickView()
        self.view.setResizeMode(QQuickView.SizeRootObjectToView)
        self.my_model = QStringListModel()
        self.my_model.setStringList(["test1", "test2"])
        self.view.setInitialProperties({"myModel": self.my_model})
        self.qml_file = Path(__file__).parent / "view.qml"
        self.view.setSource(QUrl.fromLocalFile(self.qml_file.resolve()))
        if self.view.status() == QQuickView.Error:
            sys.exit(-1)
        self.view.show()
        self.app.exec()
        del self.view

    def show_splash(self):
        pass

if __name__ == "__main__":
    DemoQtGui()
