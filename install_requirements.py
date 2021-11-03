#!/usr/bin/env python3
import platform
import subprocess
import sys

thisPlatform = platform.machine()

# https://stackoverflow.com/a/58026969/5494277
# Check if in virtual environment
in_venv = getattr(sys, "real_prefix", getattr(sys, "base_prefix", sys.prefix)) != sys.prefix
pip_call = [sys.executable, "-m", "pip"]
pip_installed = True
pip_install = pip_call + ["install", "-U"]
pip_package_install = pip_install + ["--prefer-binary"]

try:
    subprocess.check_call(pip_call + ["--version"])
except subprocess.CalledProcessError as ex:
    pip_installed = False

if not pip_installed:
    err_str = "Issues with \"pip\" package detected! Follow the official instructions to install - https://pip.pypa.io/en/stable/installation/"
    raise RuntimeError(err_str)

if sys.version_info[0] != 3:
    raise RuntimeError("Demo script requires Python 3 to run (detected: Python {})".format(sys.version_info[0]))

if platform.machine() == "arm64" and platform.system() == "Darwin":
    err_str = "There are no prebuilt wheels for M1 processors. Please open the following link for a solution - https://discuss.luxonis.com/d/69-running-depthai-on-apple-m1-based-macs"
    raise RuntimeError(err_str)

is_pi = thisPlatform.startswith("arm")
prebuiltWheelsPythonVersion = [7,9]
if is_pi and sys.version_info[1] not in prebuiltWheelsPythonVersion:
    print("[WARNING] There are no prebuilt wheels for Python 3.{} for OpenCV, building process on this device may be long and unstable".format(sys.version_info[1]))

if not in_venv:
    pip_install.append("--user")
    pip_package_install.append("--user")

subprocess.check_call(pip_install + ["pip", "-U"])

subprocess.check_call(pip_call + ["uninstall", "opencv-python", "opencv-contrib-python", "--yes"])
subprocess.check_call(pip_call + ["uninstall", "depthai", "--yes"])
subprocess.check_call(pip_package_install + ["-r", "requirements.txt"])

try:
    subprocess.check_call(pip_package_install + ["-r", "requirements-optional.txt"], stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError as ex:
    print("Optional dependencies were not installed. This is not an error.")


if thisPlatform == "aarch64":
    # try to import opencv, numpy in a subprocess, since it might fail with illegal instruction
    # if it was previously installed w/ pip without setting OPENBLAS_CORE_TYPE=ARMV8 env variable
    opencvInstalledProperly = False
    try:
        subprocess.check_call([sys.executable, "-c", "import numpy, cv2;"])
        opencvInstalledProperly = True
    except subprocess.CalledProcessError as ex:
        opencvInstalledProperly = False

    if not opencvInstalledProperly:
        from os import environ
        OPENBLAS_CORE_TYPE = environ.get('OPENBLAS_CORE_TYPE')
        if OPENBLAS_CORE_TYPE != 'ARMV8':
            WARNING='\033[1;5;31m'
            RED='\033[91m'
            LINE_CL='\033[0m'
            SUGGESTION='echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc && source ~/.bashrc'
            print(f'{WARNING}WARNING:{LINE_CL} Need to set OPENBLAS_CORE_TYPE environment variable, otherwise opencv will fail with illegal instruction.')
            print(f'Run: {RED}{SUGGESTION}{LINE_CL}')
