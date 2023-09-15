#!/usr/bin/env python3
import platform
import subprocess
import sys
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
this_platform = platform.machine()

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

is_pi = this_platform.startswith("arm")
prebuiltWheelsPythonVersion = [7,9]
if is_pi and sys.version_info[1] not in prebuiltWheelsPythonVersion:

    print("[WARNING] There are no prebuilt wheels for Python 3.{} for OpenCV, building process on this device may be long and unstable".format(sys.version_info[1]))

if not in_venv:
    pip_install.append("--user")
    pip_package_install.append("--user")

subprocess.check_call(pip_install + ["pip", "-U"])

subprocess.check_call(pip_call + ["uninstall", "opencv-python", "opencv-contrib-python", "--yes"])
subprocess.check_call(pip_call + ["uninstall", "depthai", "--yes"])
subprocess.check_call(pip_package_install + ["-r", "requirements.txt"], cwd=script_directory)

try:
    subprocess.check_call(pip_package_install + ["-r", "requirements-optional.txt"], cwd=script_directory, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError as ex:
    print("Optional dependencies were not installed. This is not an error.")
