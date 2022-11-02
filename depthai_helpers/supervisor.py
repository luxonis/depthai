import atexit
import importlib.util
import os
import signal
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
    child = None

    def __init__(self):
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        atexit.register(self.cleanup)

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
                cmd = ' '.join([f'"{sys.executable}"', "depthai_demo.py"] + new_args)
                self.child = subprocess.Popen(cmd, shell=True, env=new_env, cwd=str(repo_root.resolve()))
                self.child.communicate()
                if self.child.returncode != 0:
                    raise subprocess.CalledProcessError(self.child.returncode, cmd)
            except subprocess.CalledProcessError as ex:
                print("Error while running demo script... {}".format(ex))
                print("Waiting 5s for the device to be discoverable again...")
                time.sleep(5)
                args.guiType = "cv"
        if args.guiType == "cv":
            new_env = env.copy()
            new_env["DEPTHAI_INSTALL_SIGNAL_HANDLER"] = "0"
            new_args = createNewArgs(args)
            cmd = ' '.join([f'"{sys.executable}"', "depthai_demo.py"] + new_args)
            self.child = subprocess.Popen(cmd, shell=True, env=new_env, cwd=str(repo_root.resolve()))
            self.child.communicate()

    def checkQtAvailability(self):
        return importlib.util.find_spec("PyQt5") is not None

    def cleanup(self, *args, **kwargs):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(1)
            except subprocess.TimeoutExpired:
                pass



