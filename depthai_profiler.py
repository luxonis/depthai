#!/usr/bin/env python3

#depthai function profiler
import subprocess
import sys
import numpy as np

#this is a debugging tool, that's why it's not added to requirements.txt
try:
    import snakeviz
except ImportError:
    raise ImportError('\033[1;5;31m snakeviz module not found, run: \033[0m python3 -m pip install snakeviz ')

if __name__ == "__main__":
    output_profile_file = 'depthai.prof'
    cmd = ["python3", "-m", "cProfile", "-o", output_profile_file, "-s", "tottime", "depthai_demo.py"]
    cmd = np.concatenate((cmd, sys.argv[1:]))
    print(cmd)

    subprocess.run(cmd)
    subprocess.run(["snakeviz", output_profile_file])
