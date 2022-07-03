import array
from pathlib import Path
import os
import cv2
import types
import depthai as dai
import datetime
from .utils import *

class Replay:
    disabledStreams = []
    _streamTypes = ['color', 'left', 'right', 'depth'] # Available types to stream back to the camera
    _fileTypes = ['color', 'left', 'right', 'disparity', 'depth']
    _supportedExtensions = ['.mjpeg', '.avi', '.mp4', '.h265', '.h264', '.bag', '.mcap']
    _inputQueues = dict() # dai.InputQueue dictionary for each stream
    _seqNum = 0 # Frame sequence number, added to each imgFrame
    _start = datetime.datetime.now() # For frame timestamp
    _now = datetime.datetime.now()
    _colorSize = None
    _keepAR = True # By default crop image as needed to keep the aspect ratio
    _xins = [] # XLinkIn stream names

    def __init__(self, path: str):
        """
        Helper file to replay recorded depthai stream. It reads from recorded files (mjpeg/avi/mp4/h265/h264/bag)
        and sends frames back to OAK camera to replay the scene, including depth reconstruction from 2 synced mono
        streams.
    
        Args:
            path (str): Path to the recording folder
        """
        self.path = Path(path).resolve().absolute()

        self.frames = dict() # Frames read from Readers
        self.imgFrames = dict() # Last frame sent to the device

        self.readers = dict()
        def readFile(filePath: Path) -> None:
            file = os.path.basename(filePath)
            (name, extension) = os.path.splitext(file)
            if extension in self._supportedExtensions:
                if extension == '.bag':
                    from .readers.rosbag_reader import RosbagReader
                    self.readers[name] = RosbagReader(filePath)
                elif extension == '.mcap':
                    from .readers.mcap_reader import McapReader
                    self.readers[name] = McapReader(filePath)
                elif name in self._fileTypes:
                    # For .mjpeg / .h265 / .mp4 files
                    from .readers.videocap_reader import VideoCapReader
                    self.readers[name] = VideoCapReader(filePath)
                else:
                    print(f"Found and skipped an unsupported file name: '{file}'.")    
            elif file == 'calib.json':
                self._calibData = dai.CalibrationHandler(filePath)
            else:
                print(f"Found and skipped an unknown file, extension: '{extension}'.")

        if self.path.is_dir(): # Provided path is a folder
            for fileName in os.listdir(path):
                filePath = self.path / fileName
                if filePath.is_file(): readFile(filePath)
        else: # Provided path is a file
            readFile(self.path)

        if len(self.readers) == 0:
            raise RuntimeError("Path invalid - no recordings found.")

    def setResizeColor(self, size: tuple):
        """
        Resize color frames prior to sending them to the device.

        Args:
            size (tuple(width, heigth)): Size of color frames that are sent to the camera
        """
        self._colorSize = size

    def keepAspectRatio(self, keepAspectRatio: bool):
        """
        Used when we want to resize color frames before sending them to the host. By default,
        this is set to True, so frames are cropped to keep the original aspect ratio.
        """
        self._keepAR = keepAspectRatio

    def disableStream(self, streamName: str, disableReading: bool = False):
        """
        Disable sending a recorded stream to the device.

        Args:
            streamName(str): Name of the stream to disable (eg. 'left', 'color', 'depth', etc.)
            disableReading (bool, Optional): Also disable reading frames from the file
        """
        # if streamName not in self.readers:
        #     print(f"There's no stream '{streamName}' available!")
        #     return
        if disableReading:
            self.readers[streamName].close()
            # Remove the stream from the dict
            self.readers.pop(streamName, None)

        self.disabledStreams.append(streamName)

    def sendFrames(self):
        """
        Reads and sends recorded frames from all enabled streams to the OAK camera.

        Returns:
            bool: True if successful, otherwise False.
        """
        if not self._readFrames():
            return False # End of the recording
        self._now = datetime.datetime.now()
        for name in self.frames:
            imgFrame = self._createImgFrame(name, self.frames[name])
            # Save the imgFrame
            self.imgFrames[name] = imgFrame

            # Don't send these frames to the OAK camera
            if name in self.disabledStreams: continue

            # Send an imgFrame to the OAK camera
            self._inputQueues[name].send(imgFrame)
        
        self._seqNum += 1
        return True

    def initPipeline(self):
        """
        Prepares the pipeline for replaying. It creates XLinkIn nodes and sets up StereoDepth node.
        Returns:
            pipeline, nodes
        """
        pipeline = dai.Pipeline()
        pipeline.setCalibrationData(self._calibData)
        nodes = types.SimpleNamespace()

        def createXIn(p: dai.Pipeline, name: str):
            xin = p.create(dai.node.XLinkIn)
            xin.setMaxDataSize(self._getMaxSize(name))
            xin.setStreamName(name + '_in')
            self._xins.append(name)
            print()
            return xin

        for _, reader in self.readers.items():
            for name in reader.getStreams():
                if name not in self.disabledStreams:
                    setattr(nodes, name, createXIn(pipeline, name))

        # Create StereoDepth node
        if hasattr(nodes, 'left') and hasattr(nodes, 'right'):
            nodes.stereo = pipeline.create(dai.node.StereoDepth)
            nodes.stereo.setInputResolution(self._getShape('left'))

            nodes.left.out.link(nodes.stereo.left)
            nodes.right.out.link(nodes.stereo.right)

        return pipeline, nodes

    def createQueues(self, device: dai.Device):
        """
        Creates input queue for each enabled stream
        
        Args:
            device (dai.Device): Device to which we will stream frames
        """
        for name in self._xins:
            print('creating out queue for ', name)
            self._inputQueues[name] = device.getInputQueue(name + '_in')

    def getStreams(self) -> array:
        streams = []
        for _, reader in self.readers.items():
            [streams.append(name) for name in reader.getStreams()]
        return streams

    def _resizeColor(self, frame: cv2.Mat) -> cv2.Mat:
        if self._colorSize is None:
            # No resizing needed
            return frame

        if not self._keepAR:
            # No need to keep aspect ratio, image will be squished
            return cv2.resize(frame, self._colorSize)

        cropped = cropToAspectRatio(frame, self._colorSize)
        return cv2.resize(cropped, self._colorSize)

    def _createNewFrame(self, cvFrame) -> dai.ImgFrame:
        imgFrame = dai.ImgFrame()
        imgFrame.setData(cvFrame)
        imgFrame.setTimestamp(self._now - self._start)
        imgFrame.setSequenceNum(self._seqNum)
        shape = cvFrame.shape[::-1]
        imgFrame.setWidth(shape[0])
        imgFrame.setHeight(shape[1])
        return imgFrame

    def _createImgFrame(self, name: str, cvFrame: cv2.Mat) -> dai.ImgFrame:
        imgFrame: dai.ImgFrame = None
        if name == 'color':
            # Resize/crop color frame as specified by the user
            cvFrame = self._resizeColor(cvFrame)
            # cv2 reads frames in interleaved format, and most networks expect planar by default
            cvFrame = toPlanar(cvFrame)
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.BGR888p)
            imgFrame.setInstanceNum(dai.CameraBoardSocket.RGB)
        elif name == 'left' or name == 'right':
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.RAW8)
            imgFrame.setInstanceNum(getattr(dai.CameraBoardSocket, name.upper()))
        elif name == 'depth':
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.RAW16)

        return imgFrame

    def _readFrames(self) -> bool:
        """
        Reads frames from all Readers.
        
        Returns:
            bool: True if successful, otherwise False.
        """
        self.frames = dict()
        for name in self.readers:
            if 1 < len(self.readers[name].getStreams()): # Read all frames (one of each)
                frames = self.readers[name].read()
                for name, frame in frames.items():
                    self.frames[name] = frame
            else:
                self.frames[name] = self.readers[name].read() # Read a frame
            
            if self.frames[name] is False:
                return False # No more frames!
            
            # Compress 3-plane frame to a single plane
            if name in ["left", "right", "disparity"] and len(self.frames[name].shape) == 3:
                self.frames[name] = self.frames[name][:,:,0] # All 3 planes are the same
        return True

    def _getMaxSize(self, name: str):
        """
        Used when setting XLinkIn nodes, so they consume the least amount of memory needed.
        """
        size = self._getShape(name)
        bytes_per_pixel = 1
        if name == 'color': bytes_per_pixel = 3
        elif name == 'depth': bytes_per_pixel = 2 # 16bit
        return size[0] * size[1] * bytes_per_pixel

    def _getShape(self, name: str) -> tuple:
        """
        Get shape of a stream
        """
        for _, reader in self.readers.items():
            if name in reader.getStreams():
                return reader.getShape(name)


    def close(self):
        """
        Closes all video readers.
        """
        for name in self.readers:
            self.readers[name].close()