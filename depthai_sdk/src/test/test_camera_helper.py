from depthai_sdk.components.camera_helper import *
import unittest


class TestUtils(unittest.TestCase):

    def test_closest_ispscale_1(self):
        scale = getClosestIspScale((4056, 3040), width=3560)
        self.assertEqual(scale, [7, 8, 7, 8])

    def test_closest_ispscale_2(self):
        scale = getClosestIspScale((4056, 3040), height=1715)
        self.assertEqual(scale, [13, 23, 13, 23])

    def test_closest_ispscale_3(self):
        scale = getClosestIspScale((4056, 3040), width=3560, videoEncoder=True)
        self.assertEqual(scale, [12, 13, 9, 10])

    def test_closest_ispscale_4(self):
        scale = getClosestIspScale((4056, 3040), height=2500, videoEncoder=True)
        self.assertEqual(scale, [13, 16, 14, 17])

    def test_closest_ispscale_5(self):
        scale = getClosestIspScale((4056, 3040), width=2900, videoEncoder=True)
        self.assertEqual(scale, [13, 16, 9, 11])
