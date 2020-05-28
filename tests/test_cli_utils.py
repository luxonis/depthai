import unittest

from argparse import Namespace
import numpy as np

from depthai_helpers.cli_utils import parse_args, cli_print, PrintColors


class TestCliUtils(unittest.TestCase):

    def test_parse_args_default(self):
        options = parse_args()
        assert isinstance(options, Namespace)
        assert np.isclose(options.baseline, 9.0)
        assert np.isclose(options.field_of_view, 71.86)
        assert np.isclose(options.rgb_field_of_view, 68.7938)
        assert np.isclose(options.rgb_baseline, 2.0)
        assert options.swap_lr is True

    def test_cli_print_valid(self):
        cli_print("some message", PrintColors.WARNING)

    def test_cli_print_input_not_valid(self):
        with self.assertRaises(ValueError):
            cli_print("some message", "Not a valid type")


if __name__ == "__main__":

    unittest.main()
