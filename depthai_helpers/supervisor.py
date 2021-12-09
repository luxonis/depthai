import importlib.util
import os
import subprocess
import sys
import time

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
            new_env["LD_LIBRARY_PATH"] = QLibraryInfo.location(QLibraryInfo.LibrariesPath)
        new_env["DEPTHAI_INSTALL_SIGNAL_HANDLER"] = "0"
        args.noSupervisor = True

        try:
            new_args = createNewArgs(args)
            subprocess.check_call(sys.argv[:2] + new_args, env=new_env)
        except subprocess.CalledProcessError as ex:
            if args.guiType != "qt":
                raise
            print("Error while running demo script... {}".format(ex))
            print("Waiting 5s for the device to be discoverable again...")
            time.sleep(5)
            args.guiType = "cv"
            new_args = createNewArgs(args)
            subprocess.check_call(sys.argv[:2] + new_args, env=new_env)

    def checkQtAvailability(self):
        return importlib.util.find_spec("PyQt5") is not None
