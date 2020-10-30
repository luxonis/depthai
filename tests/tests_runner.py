import argparse
import os
from pathlib import Path
from consts.resource_paths import nn_resource_path
from depthai_helpers.cli_utils import PrintColors
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--reruns", type=int, default=0)
args = parser.parse_args()


def print_result(name, success):
    if success:
        print(f"{PrintColors.BLACK_BG_BLUE}Test {name}: {PrintColors.BLACK_BG_GREEN}Passed!{PrintColors.ENDC}")
    else:
        print(f"{PrintColors.BLACK_BG_BLUE}Test {name}: {PrintColors.BLACK_BG_RED}Failed!{PrintColors.ENDC}")


def rel_path(path):
    return str((Path(__file__).parent / Path(path)).resolve().absolute())


class TestRunner:
    results = {}

    def test(self, name, cmd, run_nr=0):
        exit_code = os.system(cmd)
        success = exit_code == 0
        self.results[name] = success
        print_result(name, success=success)
        if not success:
            if run_nr < args.reruns:
                print(f"\033[1;34;40mTest {name}: \033[1;33;40mRerun...\033[0m")
                self.test(name, cmd, run_nr=run_nr + 1)
            else:
                raise RuntimeError("Test failed: " + cmd)

    def test_streams(self, streams):
        cmd = sys.executable + " " + rel_path('./test_executor.py') + ' -s ' + streams
        name = ','.join(streams.split(' '))
        self.test(name, cmd)

    def test_cnn(self, cnn):
        cmd = sys.executable + " " + rel_path('./test_executor.py') + ' -s previewout metaout -cnn ' + cnn
        self.test(cnn, cmd)

    def print_summary(self):
        for name in self.results:
            print_result(name, self.results[name])


if __name__ == "__main__":
    runner = TestRunner()
    runner.test_streams("meta_d2h metaout")
    runner.test_streams("previewout")
    runner.test_streams("left")
    runner.test_streams("right")
    runner.test_streams("depth")
    runner.test_streams("disparity")
    runner.test_streams("disparity_color")
    runner.test_streams("object_tracker")

    cnnlist = os.listdir(nn_resource_path)
    for cnn in cnnlist:
        if os.path.isdir(nn_resource_path + cnn):
            runner.test_cnn(cnn)

    runner.print_summary()
