#!/usr/bin/env python3

import subprocess
import sys

# https://stackoverflow.com/a/58026969/5494277
in_venv = getattr(sys, "real_prefix", getattr(sys, "base_prefix", sys.prefix)) != sys.prefix
pip_call = [sys.executable, "-m", "pip"]
pip_install = pip_call + ["install"]

if not in_venv:
    pip_install.append("--user")

subprocess.check_call([*pip_install, "pip", "-U"])
# temporary workaroud for issue between main and develop
subprocess.check_call([*pip_call, "uninstall", "depthai", "--yes"])
subprocess.check_call([*pip_install, "-r", "requirements.txt"])

try:
    subprocess.check_call([*pip_install, "-r", "requirements-optional.txt"], stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError as ex:
    print(f"Optional dependencies were not installed. This is not an error.")
