import unittest
from pathlib import Path
from depthai_sdk.managers import EncodingManager, PipelineManager
from depthai_sdk import Previews
import depthai as dai
import os
unittest.TestLoader.sortTestMethodsUsing = None


class TestEncodingManager(unittest.TestCase):

    def test_Init1(self):
        """Testing init with an empty dict and a real path"""
        test = EncodingManager(encodeConfig={}, encodeOutput=Path(""))
        self.assertIsNotNone(test)

    def  test_Init2(self):
        """Testing init with an empty dict and a false path"""
        with self.assertRaises(RuntimeError):
            EncodingManager(encodeConfig={}, encodeOutput=Path("/NotARealPath"))

    def test_Init3(self):
        """Testing if everything in init is stored correctly if used with every attribute"""
        test = EncodingManager(encodeConfig={Previews.color.name: 30}, encodeOutput=Path(""))
        self.assertDictEqual(test.encodeConfig, {Previews.color.name: 30})
        self.assertEqual(test.encodeOutput, Path(""))

    def test_CreateEncoders1(self):
        """Testing createEncoders with a valid pipeline"""
        pm = PipelineManager()
        pm.createColorCam()
        test = EncodingManager({Previews.color.name: 30}, Path(""))
        test.createEncoders(pm)
        self.assertTrue("color" in test._encodingNodes)

    def test_CreateEncoders2(self):
        """Testing createEncoders with a valid pipeline(all nodes)"""
        pm = PipelineManager()
        pm.createColorCam()
        pm.createLeftCam()
        pm.createRightCam()
        test = EncodingManager({
            Previews.color.name: 30,
            Previews.left.name: 30,
            Previews.right.name: 30}, Path(""))
        test.createEncoders(pm)
        self.assertTrue("color" in test._encodingNodes and
                        "left" in test._encodingNodes and
                        "right" in test._encodingNodes)

    def test_CreateDefaultQueues1(self):
        """Testing createDefaultQueues with a valid pipeline"""
        pm = PipelineManager()
        pm.createColorCam()
        test = EncodingManager({Previews.color.name: 30}, Path(""))
        test.createEncoders(pm)
        with dai.Device(pm.pipeline) as device:
            test.createDefaultQueues(device)
        self.assertEqual(len(test._encodingQueues), 1)
        self.assertTrue("color" in test._encodingQueues)
        self.assertTrue("color" in test._encodingFiles)

    def test_CreateDefaultQueues2(self):
        """Testing createDefaultQueues with a valid pipeline(all nodes)"""
        pm = PipelineManager()
        pm.createColorCam()
        pm.createLeftCam()
        pm.createRightCam()
        test = EncodingManager({
            Previews.color.name: 30,
            Previews.left.name: 30,
            Previews.right.name: 30}, Path(""))
        test.createEncoders(pm)
        with dai.Device(pm.pipeline) as device:
            test.createDefaultQueues(device)
        self.assertEqual(len(test._encodingQueues), 3)
        self.assertTrue("color" in test._encodingQueues and
                        "left" in test._encodingQueues and
                        "right" in test._encodingQueues)
        self.assertTrue("color" in test._encodingFiles and
                        "left" in test._encodingFiles and
                        "right" in test._encodingFiles)

    def test_close1(self):
        """Testing close with a valid pipeline, if closed correctly the file will be deleted (files are in .h264)"""
        pm = PipelineManager()
        pm.createColorCam()
        test = EncodingManager({Previews.color.name: 30}, Path(""))
        test.createEncoders(pm)
        with dai.Device(pm.pipeline) as device:
            test.createDefaultQueues(device)
            test.close()
        os.remove("color.h264")

    def test_close2(self):
        """Testing close with a valid pipeline, if closed correctly the files will be deleted (files are in .h264)"""
        pm = PipelineManager()
        pm.createColorCam()
        pm.createLeftCam()
        pm.createRightCam()
        test = EncodingManager({
            Previews.color.name: 30,
            Previews.left.name: 30,
            Previews.right.name: 30}, Path(""))
        test.createEncoders(pm)
        with dai.Device(pm.pipeline) as device:
            test.createDefaultQueues(device)
            test.close()
        os.remove("color.h264")
        os.remove("left.h264")
        os.remove("right.h264")
