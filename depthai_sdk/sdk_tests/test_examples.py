import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import pytest

EXAMPLES_DIR = Path(__file__).parents[1] / 'examples'

# Create a temporary directory for the tests
Path('/tmp/depthai_sdk_tests').mkdir(exist_ok=True)
os.chdir('/tmp/depthai_sdk_tests')


@pytest.mark.parametrize('example', list(EXAMPLES_DIR.rglob("**/*.py")))
def test_examples(example):
    print(f"Running {example}")
    python_executable = Path(sys.executable)
    result = subprocess.Popen(f"{python_executable} {example}",
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              env={
                                  'DISPLAY': '',
                                  'PYTHONPATH': f'{os.environ["PYTHONPATH"]}:{EXAMPLES_DIR.parent}'
                              },
                              shell=True)

    time.sleep(5)
    result.kill()
    time.sleep(5)
    print('Stderr: ', result.stderr.read().decode())

    if result.returncode and result.returncode != 0:
        assert False, f"{example} raised an exception: {result.stderr}"

    cv2.destroyAllWindows()
