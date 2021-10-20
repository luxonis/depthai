# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle

# To be used on the @QmlElement decorator
# (QML_IMPORT_MINOR_VERSION is optional)
QML_IMPORT_NAME = "dai.gui"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class Bridge(QObject):
    @Slot(bool)
    def toggleSubpixel(self, state):
        print("STATE: {}".format(state))

    @Slot(int)
    def setDisparityConfidenceThreshold(self, value):
        print("setting new threshold: {}".format(value))


class DemoQtGui:
    def __init__(self, device):
        self.app = QGuiApplication()
        self.engine = QQmlApplicationEngine()
        self.qml_file = Path(__file__).parent / "view.qml"
        with self.qml_file.open('rb') as f:
            additional = b"import dai.gui 1.0\n"
            self.engine.loadData(additional + f.read())
        if not self.engine.rootObjects():
            sys.exit(-1)
        sys.exit(self.app.exec())

if __name__ == "__main__":
    # pm = PipelineManager()
    # pm.createColorCam(xout=True)
    # with dai.Device(pm.pipeline) as device:
    #     DemoQtGui(device)
    DemoQtGui(None)
