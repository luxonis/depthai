#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path

env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).parent)
result = subprocess.run(["python3", "./tests/tests_runner.py"], env=env)
exit(result.returncode)