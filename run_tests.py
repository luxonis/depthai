#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path
import sys

env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).parent)
result = subprocess.run([sys.executable, "./tests/tests_runner.py"], env=env)
exit(result.returncode)