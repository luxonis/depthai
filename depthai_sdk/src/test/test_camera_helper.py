from depthai_sdk.components.camera_helper import *
import unittest
import depthai as dai

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

    def test_sensor_type_1(self):
        self.assertEqual(cameraSensorType('imx378'), dai.node.ColorCamera)

    def test_sensor_type_2(self):
        self.assertEqual(cameraSensorType('imx577'), dai.node.ColorCamera)

    def test_sensor_type_3(self):
        self.assertEqual(cameraSensorType('ov9282'), dai.node.MonoCamera)

    def test_sensor_res_1(self):
        self.assertEqual(cameraSensorResolution('imx378'), dai.ColorCameraProperties.SensorResolution.THE_12_MP)

    def test_sensor_res_2(self):
        self.assertEqual(cameraSensorResolution('imx214'), dai.ColorCameraProperties.SensorResolution.THE_13_MP)

    def test_sensor_res_3(self):
        self.assertEqual(cameraSensorResolution('ov9282'), dai.MonoCameraProperties.SensorResolution.THE_800_P)

    def test_sensor_res_size_1(self):
        self.assertEqual(cameraSensorResolutionSize('imx378'), (4056, 3040))

    def test_sensor_res_size_2(self):
        self.assertEqual(cameraSensorResolutionSize('imx214'), (4208, 3120))

    def test_sensor_res_size_3(self):
        self.assertEqual(cameraSensorResolutionSize('ov9282'), (1280, 800))
