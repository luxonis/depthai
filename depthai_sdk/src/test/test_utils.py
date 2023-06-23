import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytube
from depthai_sdk import utils
import unittest
import depthai as dai


class TestUtils(unittest.TestCase):

    #############################
    # Test if cosDis is correct #
    #############################

    def test_blob_download(self):
        blob_path = utils.getBlob('https://model-blobs.fra1.digitaloceanspaces.com/cars_yolo_v4_tiny_openvino_2021.3_6shave.blob')
        model = dai.OpenVINO.Blob(str(blob_path))

        # Get first input layer (only one)
        nn_input: dai.TensorInfo = next(iter(model.networkInputs.values()))
        self.assertEqual(nn_input.dims, [512, 320, 3, 1])

    def test_cos1(self):
        """2D vector test"""
        self.assertEqual(utils.cosDist((3, 4), (4, 3)), 0.96)

    def test_cos2(self):
        """2D vector test"""
        self.assertEqual(utils.cosDist((7, 1), (5, 5)), 0.7999999999999999)

    def test_cos3(self):
        """3D vector test"""
        self.assertEqual(utils.cosDist((3, 4, 0), (4, 4, 2)), 0.9333333333333333)

    def test_cos4(self):
        """Incorrect arguments test"""
        self.assertTrue(np.isnan(utils.cosDist(0, 0)))

    #######################################
    # Test if frameNorm is done correctly #
    #######################################

    def test_frameNorm1(self):
        """Testing if function returns correct number of values"""
        frame = cv2.imread("./data/logo.png")
        self.assertEqual(len(utils.frameNorm(frame, [100, 200, 100, 200])), 4)

    ###############################################
    # Testing if toPlanar works correctly #
    ###############################################

    def test_toPlanar1(self):
        """Testing with no arguments"""
        frame = cv2.imread("./data/logo.png")
        self.assertEqual(utils.toPlanar(frame).shape, (3, 515, 510))

    def test_toPlanar2(self):
        """Testing with arguments"""
        frame = cv2.imread("./data/logo.png")
        self.assertEqual(utils.toPlanar(frame, (200, 200)).shape, (3, 200, 200))

    ##########################################
    # Test if toTensorResult works correctly #
    ##########################################

    def test_toTensorResult1(self):
        """Testing with empty NNdata object"""
        self.assertEqual(utils.toTensorResult(dai.NNData()), {})

    ###################################
    # Test if merge is done correctly #
    ###################################

    def test_merge1(self):
        """Testing 2 different dictionaries"""
        dict1 = {"1": "value", "2": "value"}
        dict2 = {"3": "value", "4": "value"}
        self.assertDictEqual(utils.merge(dict1, dict2), {"1": "value", "2": "value", "3": "value", "4": "value"})

    def test_merge2(self):
        """Testing 2 same dictionaries"""
        dict1 = {"1": "value", "2": "value"}
        self.assertDictEqual(utils.merge(dict1, dict1), {"1": "value", "2": "value"})

    ########################################
    # Test if loadModule is done correctly #
    ########################################

    def test_loadModule1(self):
        """Testing a valid file"""
        self.assertNotEqual(utils.loadModule(Path("../depthai_sdk/utils.py")), None)

    ###########################################
    # Test if downloadYTVideo works correctly #
    ###########################################

    def test_downloadYTVideo1(self):
        """Testing with a valid url"""
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            utils.downloadYTVideo("https://www.youtube.com/watch?v=UImAcR7Ke_c", tmp_path)
            self.assertTrue((tmp_path / "OAK-D PoE unboxing.mp4").exists())

    def test_downloadYTVideo2(self):
        """Testing with an invalid url"""
        with self.assertRaisesRegex(pytube.exceptions.RegexMatchError, "regex_search"):
            utils.downloadYTVideo("https://www.youtube.com/watch?v=_", "./data/")

    #############################
    # Test if cropToAspectRatio #
    #############################

    def test_cropToAspectRatio1(self):
        """Testing with a valid frame and a valid size (ratio 2:1, h:w)"""
        frame = np.full((100, 100), 1, dtype=int)
        self.assertEqual(utils.cropToAspectRatio(frame, (2, 1)).shape, (50, 100))

    def test_cropToAspectRatio2(self):
        """Testing with a valid frame and a valid size (ratio 1:2, h:w)"""
        frame = np.full((100, 100), 1, dtype=int)
        self.assertEqual(utils.cropToAspectRatio(frame, (1, 2)).shape, (100, 50))

    def test_cropToAspectRatio3(self):
        """Testing with an invalid frame (0, 0) and a valid size (ratio 1:2, h:w)"""
        frame = np.ndarray((0, 0))
        with self.assertRaises(ZeroDivisionError):
            frame = utils.cropToAspectRatio(frame, (1, 2)).shape, (100, 50)

    def test_cropToAspectRatio4(self):
        """Testing with a valid frame and a valid size (ratio 1.5:2.5, h:w)"""
        frame = np.full((100, 100), 1, dtype=int)
        self.assertEqual(utils.cropToAspectRatio(frame, (1.5, 2.5)).shape, (100, 60))

    ##############################################
    # Testing if resizeLetterbox works correctly #
    ##############################################

    def test_resizeLetterbox1(self):
        frame = cv2.imread("./data/logo.png")
        self.assertEqual(utils.resizeLetterbox(frame, (1000, 1000)).shape, (1000, 1000, 3))

    ############################################
    # Test if createBlankFrame works correctly #
    ############################################

    def test_createBlankFrame1(self):
        self.assertEqual(utils.createBlankFrame(300, 200, (255, 0, 0)).shape, (200, 300, 3))

