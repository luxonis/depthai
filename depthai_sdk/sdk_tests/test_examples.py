import subprocess
import sys
import time
from pathlib import Path

import cv2

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def test_examples():
    python_executable = Path(sys.executable)
    for example in EXAMPLES_DIR.rglob("**/*.py"):
        print(f"Running example: {example.name}")

        result = subprocess.Popen(f"{python_executable} {example}", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env={"DISPLAY": ""}, shell=True)

        time.sleep(5)
        result.kill()
        time.sleep(5)
        print('Stderr: ', result.stderr.read().decode())

        # if result.returncode and result.returncode != 0:
        #     assert False, f"{example} raised an exception: {result.stderr}"

        cv2.destroyAllWindows()
