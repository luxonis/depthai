import unittest
import depthai as dai
from depthai_sdk.managers import PipelineManager


class TestPipeline(unittest.TestCase):

    def test_Init1(self):
        """Testing if everything is set to None when using Init with no arguments"""
        test = PipelineManager()
        self.assertIsNone(test.openvinoVersion)
        self.assertEqual(test.poeQuality, 100)
        self.assertIsNone(test._depthConfigInputQueue)
        self.assertIsNone(test._rgbConfigInputQueue)
        self.assertIsNone(test._leftConfigInputQueue)
        self.assertIsNone(test._rightConfigInputQueue)

    def test_Init2(self):
        """Testing if everything is set to None when using Init with opevino as argument"""
        test = PipelineManager(openvinoVersion=dai.OpenVINO.Version(5))
        self.assertEqual(test.openvinoVersion, dai.OpenVINO.Version(5))
        self.assertEqual(test.poeQuality, 100)
        self.assertIsNone(test._depthConfigInputQueue)
        self.assertIsNone(test._rgbConfigInputQueue)
        self.assertIsNone(test._leftConfigInputQueue)
        self.assertIsNone(test._rightConfigInputQueue)

    def test_createColorCam1(self):
        """Testing if createColorCam with xout as argument works correctly"""
        test = PipelineManager()
        test.createColorCam(xout=True)
        self.assertIsNotNone(test.nodes.xoutRgb)

    def test_createColorCam2(self):
        """Testing if createColorCam with xoutVideo as argument works correctly"""
        test = PipelineManager()
        test.createColorCam(xoutVideo=True)
        self.assertIsNotNone(test.nodes.xoutRgbVideo)

    def test_createColorCam3(self):
        """Testing if createColorCam with xoutStill as argument works correctly"""
        test = PipelineManager()
        test.createColorCam(xoutStill=True)
        self.assertIsNotNone(test.nodes.xoutRgbStill)

    def test_createColorCam4(self):
        """Testing if createColorCam without arguments works correctly"""
        test = PipelineManager()
        test.createColorCam()
        with self.assertRaises(AttributeError):
            x = test.nodes.xoutRgb
        with self.assertRaises(AttributeError):
            x = test.nodes.xoutRgbVideo
        with self.assertRaises(AttributeError):
            x = test.nodes.xoutRgbStill
        self.assertIsNotNone(test.nodes.xinRgbControl)
        self.assertIsNotNone(test.nodes.camRgb)

    # TODO also test for every other attribute

    def test_createLeftCam1(self):
        """Testing if createLeftCam with xout as argument works correctly"""
        test = PipelineManager()
        test.createLeftCam(xout=True)
        self.assertIsNotNone(test.nodes.xoutLeft)

    def test_createLeftCam2(self):
        """Testing if createLeftCam without arguments works correctly"""
        test = PipelineManager()
        test.createLeftCam()
        with self.assertRaises(AttributeError):
            x = test.nodes.xoutLeft
        self.assertIsNotNone(test.nodes.monoLeft)
        self.assertIsNotNone(test.nodes.xinLeftControl)

    def test_createRightCam1(self):
        """Testing if createLeftCam with xout as argument works correctly"""
        test = PipelineManager()
        test.createRightCam(xout=True)
        self.assertIsNotNone(test.nodes.xoutRight)

    def test_createRightCam2(self):
        """Testing if createLeftCam without arguments works correctly"""
        test = PipelineManager()
        test.createRightCam()
        with self.assertRaises(AttributeError):
            x = test.nodes.xoutRight
        self.assertIsNotNone(test.nodes.monoRight)
        self.assertIsNotNone(test.nodes.xinRightControl)
