import unittest
from pathlib import Path

import blobconverter

from depthai_sdk.managers import BlobManager


class TestBlobManager(unittest.TestCase):

    def test_Init1(self):
        """Testing init with no arguments"""
        test = BlobManager()
        self.assertIsNotNone(test)

    def test_Init2(self):
        """Testing init with a valid blobPath as argument"""
        blob_path = blobconverter.from_zoo("mobilenet-ssd")
        test = BlobManager(blobPath=blob_path)
        self.assertEqual(test.getBlob(), blob_path)

    def test_Init3(self):
        """Testing init with an invalid blobPath as argument"""
        with self.assertRaises(RuntimeError):
            test = BlobManager(blobPath=Path("notRealPath"))

    def test_Init4(self):
        """Testing init with a valid configPath as argument"""
        test = BlobManager(configPath=Path("./data/custom_model.json"))
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
        test = BlobManager(zooDir=Path("./data"))
        self.assertIsNotNone(test)

    def test_Init9(self):
        """Testing with an invalid zooDir as argument"""
        with self.assertRaises(RuntimeError):
            test = BlobManager(configPath=Path("notRealPath"))


