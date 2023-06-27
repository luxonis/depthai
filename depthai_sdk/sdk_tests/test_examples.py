import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
from pyvirtualdisplay import Display

EXAMPLES_DIR = Path(__file__).parents[1] / 'examples'
os.environ['DISPLAY'] = ''  # Hide the display

# Create a temporary directory for the tests
Path('/tmp/depthai_sdk_tests').mkdir(exist_ok=True)
os.chdir('/tmp/depthai_sdk_tests')


def test_examples():
    python_executable = Path(sys.executable)
    print(list(EXAMPLES_DIR.rglob("**/*.py")))
    print(EXAMPLES_DIR.absolute())
    for example in [list(EXAMPLES_DIR.rglob("**/*.py"))[0]]:
        print(f"Running example: {example.name}")
        print(f'{EXAMPLES_DIR.parent}:{EXAMPLES_DIR.parent}/depthai_sdk')
        with Display(visible=False, size=(800, 600)):
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

            # if result.returncode and result.returncode != 0:
            #     assert False, f"{example} raised an exception: {result.stderr}"

            cv2.destroyAllWindows()
