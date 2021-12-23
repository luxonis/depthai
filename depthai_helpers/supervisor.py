import importlib.util
import os
import subprocess
import sys
import time
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
    return sys.argv[1:] + ["--noSupervisor", "--guiType", args.guiType]


class Supervisor:
    def runDemo(self, args):
        repo_root = Path(__file__).parent.parent
        args.noSupervisor = True
        new_args = createNewArgs(args)
        env = os.environ.copy()

        if args.guiType == "qt":
            new_env = env.copy()
            new_env["QT_QUICK_BACKEND"] = "software"
            new_env["LD_LIBRARY_PATH"] = str(Path(importlib.util.find_spec("PyQt5").origin).parent / "Qt5/lib")
            new_env["DEPTHAI_INSTALL_SIGNAL_HANDLER"] = "0"
            try:
                subprocess.check_call(' '.join([f'"{sys.executable}"', "depthai_demo.py"] + new_args), env=new_env, shell=True, cwd=repo_root)
            except subprocess.CalledProcessError as ex:
                print("Error while running demo script... {}".format(ex))
                print("Waiting 5s for the device to be discoverable again...")
                time.sleep(5)
                args.guiType = "cv"
        if args.guiType == "cv":
            new_env = env.copy()
            new_env["DEPTHAI_INSTALL_SIGNAL_HANDLER"] = "0"
            new_args = createNewArgs(args)
            subprocess.check_call(' '.join([f'"{sys.executable}"', "depthai_demo.py"] + new_args), env=new_env, shell=True, cwd=repo_root)

    def checkQtAvailability(self):
        return importlib.util.find_spec("PyQt5") is not None
