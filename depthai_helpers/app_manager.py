import os
import subprocess
import sys
from pathlib import Path

initEnv = os.environ.copy()


class App:
    def __init__(self, appName, appPath=None, appRequirements=None, appEntrypoint=None):
        self.appName = appName
        self.appPath = appPath or Path(__file__).parent.parent / "apps" / self.appName
        self.venvPath = self.appPath / "venv"
        self.appPip = str(self.venvPath / "bin" / "pip")
        self.appInterpreter = str(self.venvPath / "bin" / "python")
        self.appRequirements = appRequirements or self.appPath / "requirements.txt"
        self.appEntrypoint = appEntrypoint or self.appPath / "main.py"

    def createVenv(self, force=False):
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', '-h'], env=initEnv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print("Error accessing \"venv\" module! Please try to install \"python3-venv\" or see oficial docs here - https://docs.python.org/3/library/venv.html", file=sys.stderr)
            raise

        if not force and Path(self.appInterpreter).exists():
            print("Existing venv found.")
        else:
            print("Creating venv...")
            subprocess.check_call(' '.join([sys.executable, '-m', 'venv', str(self.venvPath)]), env=initEnv, cwd=self.appPath)
        print("Installing requirements...")
        subprocess.check_call(' '.join([self.appPip, 'install', '-U', "pip"]), env=initEnv, shell=True, cwd=self.appPath)
        subprocess.check_call(' '.join([self.appPip, 'install', '-r', str(self.appRequirements)]), env=initEnv, shell=True, cwd=self.appPath)

    def runApp(self):
        subprocess.check_call(' '.join([self.appInterpreter, str(self.appEntrypoint)]), env=initEnv, shell=True, cwd=self.appPath)

