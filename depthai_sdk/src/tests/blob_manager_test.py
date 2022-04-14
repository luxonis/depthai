import unittest
from pathlib import Path
from depthai_sdk.managers import BlobManager


class TestBlobManager(unittest.TestCase):

    def test_Init1(self):
        """Testing init with no arguments"""
        test = BlobManager()
        self.assertIsNotNone(test)

    def test_Init2(self):
        """Testing init with a valid blobPath as argument"""
        test = BlobManager(blobPath=Path("./test_data/face-detection-retail-0004_openvino_2021.2_4shave.blob"))
        self.assertIsNotNone(test)

    def test_Init3(self):
        """Testing init with an invalid blobPath as argument"""
        with self.assertRaises(RuntimeError):
            test = BlobManager(blobPath=Path("notRealPath"))

    def test_Init4(self):
        """Testing init with a valid configPath as argument"""
        test = BlobManager(configPath=Path("./test_data/custom_model.json"))
        self.assertIsNotNone(test)

    def test_Init5(self):
        """Testing init with an invalid configPath as argument"""
        with self.assertRaises(RuntimeError):
            test = BlobManager(configPath=Path("notRealPath"))

    def test_Init6(self):
        """Testing init with a valid zooName as argument"""
        test = BlobManager(zooName="megadepth")
        self.assertIsNotNone(test)

    def test_Init7(self):
        """Testing init with an invalid zooName as argument"""
        with self.assertRaises(RuntimeError):
            test = BlobManager(configPath=Path("notRealPath"))

    def test_Init8(self):
        """Testing with a valid zooDir as argument"""
        test = BlobManager(zooDir=Path("./test_data/zooDir"))
        self.assertIsNotNone(test)

    def test_Init9(self):
        """Testing with an invalid zooDir as argument"""
        with self.assertRaises(RuntimeError):
            test = BlobManager(configPath=Path("notRealPath"))


