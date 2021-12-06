import os
import subprocess
import sys
from pathlib import Path

def createNewArgs(args):
    def removeArg(name, withValue=True):
        if name in sys.argv:
            idx = sys.argv.index(name)
            if withValue:
                del sys.argv[idx + 1]
            del sys.argv[idx]

    removeArg("-gt")
    removeArg("--guiType")
    removeArg("--noSupervisor")
    return sys.argv[2:] + ["--noSupervisor", "--guiType", args.guiType]


class Supervisor:
    def runDemo(self, args):
        new_env = os.environ.copy()
        if args.guiType == "qt":
            from PyQt5.QtCore import QLibraryInfo
            new_env["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
            new_env["QT_QUICK_BACKEND"] = "software"
        new_env["DEPTHAI_INSTALL_SIGNAL_HANDLER"] = "0"
        args.noSupervisor = True

        try:
            new_args = createNewArgs(args)
            subprocess.check_call(sys.argv[:2] + new_args)
        except subprocess.CalledProcessError as ex:
            args.guiType = "cv"
            new_args = createNewArgs(args)
            subprocess.check_call(sys.argv[:2] + new_args)

    def checkQtAvailability(self):
        try:
            from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QApplication
            from PyQt5.QtGui import QPixmap
            from PyQt5.QtCore import Qt, QTimer

            app = QApplication([sys.argv[0]])

            label = QLabel()
            px = QPixmap(str(Path(__file__).parent / "logo.png"))
            label.setPixmap(px)
            label.resize(200, 200)
            label.setWindowFlags(Qt.FramelessWindowHint)
            label.setAttribute(Qt.WA_TranslucentBackground)
            label.show()
            QTimer.singleShot(10, label.close)
            exit_code = app.exec_()
            return exit_code == 0
        except:
            raise
            return False