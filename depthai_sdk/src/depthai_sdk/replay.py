from pathlib import Path
import os
import cv2
import depthai as dai
from numpy import isin
from .utils import *
from typing import Dict, Optional, Tuple, List, Any, Union
from .readers.abstract_reader import AbstractReader
from time import monotonic
import time
from threading import Thread


class Replay:
    disabledStreams: List[str] = []
    readers: Dict[str, AbstractReader] = dict()
    # Nodes
    left: dai.node.XLinkIn = None
    right: dai.node.XLinkIn = None
    color: dai.node.XLinkIn = None
    stereo: dai.node.StereoDepth = None

    frames: Dict[str, Any] = dict()  # Cv2 frames read from Readers
    imgFrames: Dict[str, dai.ImgFrame] = dict()  # Last frame sent to the device

    _streamTypes = ['color', 'left', 'right', 'depth']  # Available types to stream back to the camera
    _fileTypes = ['color', 'left', 'right', 'disparity', 'depth']
    _videoExt = ['.mjpeg', '.avi', '.mp4', '.h265', '.h264', '.mpg', '.webm']
    _imageExt = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm',
                 '.pnm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']
    _inputQueues = dict()  # dai.InputQueue dictionary for each stream
    _seqNum = 0  # Frame sequence number, added to each imgFrame
    _now: monotonic = None
    _colorSize: Optional[Tuple[int, int]] = None
    _keepAR = True  # By default crop image as needed to keep the aspect ratio
    _xins = []  # XLinkIn stream names
    _pause = False
    _calibData = None
    fps: float = 30.0
    thread: Thread = None
    _stop: bool = False

    def __init__(self, path: str):
        """
        Helper file to replay recorded depthai stream. It reads from recorded files (mjpeg/avi/mp4/h265/h264/bag)
        and sends frames back to OAK camera to replay the scene, including depth reconstruction from 2 synced mono
        streams.
    
        Args:
            path (str): Path to the recording folder, file, depthai-recording name, or url to video/YouTube/image
        """
        self.path = self._get_path(path)

        def read_file(file_path: Path) -> None:
            str_path = str(file_path)
            file = os.path.basename(file_path)
            (name, extension) = os.path.splitext(file)
            if extension == '.bag':
                from .readers.rosbag_reader import RosbagReader
                self.readers[name] = RosbagReader(str_path)
            elif extension == '.mcap':
                from .readers.mcap_reader import McapReader
                self.readers[name] = McapReader(str_path)
            elif extension in self._imageExt:
                from .readers.image_reader import ImageReader
                self.readers[name] = ImageReader(str_path)
            elif extension in self._videoExt:
                # For .mjpeg / .h265 / .mp4 files
                from .readers.videocap_reader import VideoCapReader
                self.readers[name] = VideoCapReader(str_path)
            elif file == 'calib.json':
                self._calibData = dai.CalibrationHandler(str_path)
            else:
                print(f"Found and skipped an unknown file, extension: '{extension}'.")

        if self.path.is_dir():  # Provided path is a folder
            for fileName in os.listdir(self.path):
                file_path = self.path / fileName
                if file_path.is_file():
                    read_file(file_path)
        else:  # Provided path is a file
            read_file(self.path)

        if len(self.readers) == 0:
            raise RuntimeError("Path invalid - no recordings found.")

    def _get_path(self, path: str) -> Path:
        """
        Either use local depthai-recording, YT link, mp4 url
        @param path: depthai-recording path.
        @return: Replay module
        """
        if isUrl(path):
            if isYoutubeLink(path):
                # Overwrite source - so Replay class can use it
                return downloadYTVideo(path)
            else:
                # TODO: download video/image(s) from the internet
                raise NotImplementedError("Only YouTube video download is currently supported!")

        # TODO: check if absolute or relative
        if Path(path).is_file() or Path(path).is_dir():
            return Path(path)

        recordingName: str = path
        # Check if we have it stored locally
        path: Path = getLocalRecording(recordingName)
        if path is not None:
            return path
        # Try to download from the server
        dic = getAvailableRecordings()
        if recordingName in dic:
            arr = dic[recordingName]
            print("Downloading depthai recording '{}' from Luxonis' servers, in total {:.2f} MB".format(recordingName,
                                                                                                  arr[1] / 1e6))
            path = downloadRecording(recordingName, arr[0])
            return path
        else:
            raise ValueError(f"DepthAI recording '{recordingName}' was not found on the server!")

    def togglePause(self):
        """
        Toggle pausing of sending frames to the OAK camera.
        """
        self._pause = not self._pause

    def setResizeColor(self, size: Tuple[int, int]):
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

    def setFps(self, fps: float):
        """
        Sets frequency at which Replay module will send frames to the camera. Default 30FPS.
        """
        self.fps = fps

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

    def sendFrames(self, cb=None) -> bool:
        """
        Reads and sends recorded frames from all enabled streams to the OAK camera.

        Returns:
            bool: True if successful, otherwise False.
        """
        if not self._pause:  # If replaying is paused, don't read new frames
            if not self._readFrames():
                return False  # End of the recording

        self._now = monotonic()
        for name in self.frames:
            imgFrame = self._createImgFrame(name, self.frames[name])
            # Save the imgFrame
            self.imgFrames[name] = imgFrame
            if cb:  # callback
                cb(name, imgFrame)

            # Don't send these frames to the OAK camera
            if name in self.disabledStreams: continue

            # Send an imgFrame to the OAK camera
            self._inputQueues[name].send(imgFrame)

        self._seqNum += 1
        return True

    def initPipeline(self,
                     pipeline: Optional[dai.Pipeline] = None
                     ) -> dai.Pipeline:
        """
        Prepares the pipeline for replaying. It creates XLinkIn nodes and sets up StereoDepth node.
        Returns:
            pipeline, nodes
        """
        if pipeline is None:  # Create pipeline if not passed
            pipeline = dai.Pipeline()

        if self._calibData is not None:
            pipeline.setCalibrationData(self._calibData)

        def createXIn(p: dai.Pipeline, name: str):
            xin = p.create(dai.node.XLinkIn)
            xin.setMaxDataSize(self._getMaxSize(name))
            xin.setNumFrames(4)
            xin.setStreamName(name + '_in')
            self._xins.append(name)
            return xin

        for _, reader in self.readers.items():
            for name in reader.getStreams():
                if name not in self.disabledStreams:
                    if name.upper() == 'LEFT':
                        self.left = createXIn(pipeline, name)
                    elif name.upper() == 'RIGHT':
                        self.right = createXIn(pipeline, name)
                    elif name.upper() == 'COLOR':
                        self.color = createXIn(pipeline, name)
                    else:
                        pass  # Not implemented

        # Create StereoDepth node
        if self.left and self.right:
            self.stereo = pipeline.create(dai.node.StereoDepth)
            self.stereo.setInputResolution(self.getShape('left'))

            if self.color:  # Enable RGB-depth alignment
                self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
                if self._colorSize is not None:
                    self.stereo.setOutputSize(*self._colorSize)
                else:
                    self.stereo.setOutputSize(*self.getShape('color'))

            self.left.out.link(self.stereo.left)
            self.right.out.link(self.stereo.right)

        return pipeline

    def start(self, cb):
        """
        Start sending frames to the OAK device on a new thread
        """
        self.thread = Thread(target=self.run, args=(cb,))
        self.thread.start()

    def run(self, cb):
        delay = 1.0 / self.fps
        while True:
            time.sleep(delay)
            if not self.sendFrames(cb): break
            if self._stop: break
        print('Replay `run` thread stopped')

    def createQueues(self, device: dai.Device):
        """
        Creates input queue for each enabled stream
        
        Args:
            device (dai.Device): Device to which we will stream frames
        """
        for name in self._xins:
            self._inputQueues[name] = device.getInputQueue(name + '_in', maxSize=1)

    def getStreams(self) -> List[str]:
        streams: List[str] = []
        for _, reader in self.readers.items():
            [streams.append(name) for name in reader.getStreams()]
        return streams

    def _resizeColor(self, frame):
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
        imgFrame.setTimestamp(self._now)
        imgFrame.setSequenceNum(self._seqNum)
        shape = cvFrame.shape[::-1]
        imgFrame.setWidth(shape[0])
        imgFrame.setHeight(shape[1])
        return imgFrame

    def _createImgFrame(self, name: str, cvFrame) -> dai.ImgFrame:
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
            if 1 < len(self.readers[name].getStreams()):  # Read all frames (one of each)
                frames = self.readers[name].read()
                for name, frame in frames.items():
                    self.frames[name] = frame
            else:
                self.frames[name] = self.readers[name].read()  # Read a frame

            if self.frames[name] is False:
                return False  # No more frames!

            # Compress 3-plane frame to a single plane
            if name in ["left", "right", "disparity"] and len(self.frames[name].shape) == 3:
                self.frames[name] = self.frames[name][:, :, 0]  # All 3 planes are the same
        return True

    def _getMaxSize(self, name: str) -> int:
        """
        Used when setting XLinkIn nodes, so they consume the least amount of memory needed.
        """
        size = self.getShape(name)
        bytes_per_pixel = 1
        if name == 'color':
            bytes_per_pixel = 3
        elif name == 'depth':
            bytes_per_pixel = 2  # 16bit
        return size[0] * size[1] * bytes_per_pixel

    def getShape(self, streamName: str) -> Tuple[int, int]:
        """
        Get shape of a stream
        """
        for _, reader in self.readers.items():
            if streamName in reader.getStreams():
                return reader.getShape(streamName)

    def close(self):
        """
        Closes all video readers.
        """
        for name in self.readers:
            self.readers[name].close()
        self._stop = True
        if self.thread:
            self.thread.join()
