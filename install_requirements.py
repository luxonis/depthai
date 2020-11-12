#!/usr/bin/env python3

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pip", "-U"])
# temporary workaroud for issue between main and develop
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "depthai", "--yes"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
