#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path
import sys

env = os.environ.copy()
curr_dir = Path(__file__).parent
env["PYTHONPATH"] = str(curr_dir)
result = subprocess.run([sys.executable, curr_dir / Path("./tests/tests_runner.py"), "--reruns", "1"], env=env)
exit(result.returncode)
