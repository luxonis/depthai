import os
import time
from threading import Thread
from time import monotonic

import depthai as dai

from depthai_sdk.readers.abstract_reader import AbstractReader
from depthai_sdk.utils import *

_fileTypes = ['color', 'left', 'right', 'disparity', 'depth']
_videoExt = ['.mjpeg', '.avi', '.mp4', '.h265', '.h264', '.webm']
_imageExt = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm',
             '.pnm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']


class Replay:
    def __init__(self, path: str):
        """
        Helper file to replay recorded depthai stream. It reads from recorded files (mjpeg/avi/mp4/h265/h264/bag)
        and sends frames back to OAK camera to replay the scene, including depth reconstruction from 2 synced mono
        streams.
    
        Args:
            path (str): Path to the recording folder
        """
        self.path = self._get_path(path)

        self.disabledStreams: List[str] = []
        # Nodes
        self.left: Optional[dai.node.XLinkIn] = None
        self.right: Optional[dai.node.XLinkIn] = None
        self.color: Optional[dai.node.XLinkIn] = None

        self._inputQueues = dict()  # dai.InputQueue dictionary for each stream
        self._seqNum = 0  # Frame sequence number, added to each imgFrame
        self._now: monotonic = None
        self._colorSize = None
        self._keepAR = True  # By default, crop image as needed to keep the aspect ratio
        self._pause = False
        self._calibData = None

        self.fps: float = 30.0
        self.thread: Optional[Thread] = None
        self._stop: bool = False  # Stop the thread that's sending frames to the OAK camera

        self.xins: List[str] = []  # Name of XLinkIn streams

        self.reader: Optional[AbstractReader] = None
        self.frames: Dict[str, np.ndarray] = dict()  # Frames read from Readers
        self.imgFrames: Dict[str, dai.ImgFrame] = dict()  # Last frame sent to the device

        def cntFilesExt(path: Path, ext: Union[str, List[str]]) -> int:
            def fileWithExt(file: str) -> bool:
                if isinstance(ext, List):
                    return os.path.splitext(file)[1] in ext
                elif isinstance(ext, str):
                    return file.endswith(ext)
                else:
                    raise ValueError('ext should be either str or List[str]!')

            return [fileWithExt(f) for f in os.listdir(str(path))].count(True)

        if self.path.is_dir():  # Provided path is a folder
            if 0 < cntFilesExt(self.path, _imageExt):
                from .readers.image_reader import ImageReader
                self.reader = ImageReader(self.path)
            elif 0 < cntFilesExt(self.path, _videoExt):
                from .readers.videocap_reader import VideoCapReader
                self.reader = VideoCapReader(self.path)
            elif cntFilesExt(self.path, '.bag') == 1:
                from .readers.rosbag_reader import RosbagReader
                self.reader = RosbagReader(self.path)
            elif cntFilesExt(self.path, '.mcap') == 1:
                from .readers.mcap_reader import McapReader
                self.reader = McapReader(self.path)
            elif cntFilesExt(self.path, '.db3') == 1:
                from .readers.db3_reader import Db3Reader
                self.reader = Db3Reader(self.path)
            else:
                raise RuntimeError("Path invalid - no recordings found.")

            # Read calibration file
            calib_file = self.path / 'calib.json'
            if calib_file.exists():
                self._calibData = dai.CalibrationHandler(str(calib_file))

        else:  # Provided path is a file
            if self.path.suffix in _videoExt:
                from .readers.videocap_reader import VideoCapReader
                self.reader = VideoCapReader(self.path)
            elif self.path.suffix in _imageExt:
                from .readers.image_reader import ImageReader
                self.reader = ImageReader(self.path)
            else:
                raise NotImplementedError('Please select folder')

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
                return downloadContent(path)

        if Path(path).resolve().exists():
            return Path(path).resolve()

        recording_name: str = path
        # Check if we have it stored locally
        path: Path = getLocalRecording(recording_name)
        if path is not None:
            return path

        # Try to download from the server
        dic = getAvailableRecordings()
        if recording_name in dic:
            arr = dic[recording_name]
            print("Downloading depthai recording '{}' from Luxonis' servers, in total {:.2f} MB".format(recording_name,
                                                                                                        arr[1] / 1e6))
            path = downloadRecording(recording_name, arr[0])
            return path
        else:
            raise ValueError(f"DepthAI recording '{recording_name}' was not found on the server!")

    def togglePause(self):
        """
        Toggle pausing of sending frames to the OAK camera.
        """
        self._pause = not self._pause
        print("PAUSE", self._pause)

    def setFps(self, fps: float):
        """
        Sets frequency at which Replay module will send frames to the camera. Default 30FPS.
        """
        self.fps = fps

    def getFps(self) -> float:
        return self.fps

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
            self.reader.disableStream(streamName)

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

    def initPipeline(self, pipeline: dai.Pipeline = None):
        """
        Prepares the pipeline for replaying. It creates XLinkIn nodes and sets up StereoDepth node.
        Returns: dai.Pipeline
        """
        if pipeline is None:  # Create pipeline if not passed
            pipeline = dai.Pipeline()

        if self._calibData is not None:
            pipeline.setCalibrationData(self._calibData)

        def createXIn(p: dai.Pipeline, name: str):
            xin = p.create(dai.node.XLinkIn)
            xin.setMaxDataSize(self._getMaxSize(name))
            xin.setStreamName(name + '_in')
            self.xins.append(name)
            return xin

        for name in self.reader.getStreams():
            if name not in self.disabledStreams:
                xin = createXIn(pipeline, name)
                if name.upper() == 'LEFT':
                    self.left = xin
                elif name.upper() == 'RIGHT':
                    self.right = xin
                elif name.upper() == 'COLOR':
                    self.color = xin
                else:
                    pass  # Not implemented

        return pipeline

    def initStereoDepth(self, stereo: dai.node.StereoDepth):
        streams = self.reader.getStreams()
        if 'left' not in streams or 'right' not in streams:
            raise Exception("Tried to init StereoDepth, but left/right streams aren't available!")
        stereo.setInputResolution(self.getShape('left'))

        if self.color:  # Enable RGB-depth alignment
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            if self._colorSize is not None:
                stereo.setOutputSize(*self._colorSize)
            else:
                stereo.setOutputSize(*self.getShape('color'))

        self.left.out.link(stereo.left)
        self.right.out.link(stereo.right)

    def start(self, cb):
        """
        Start sending frames to the OAK device on a new thread
        """
        self.thread = Thread(target=self.run, args=(cb,))
        self.thread.start()

    def run(self, cb):
        delay = 1.0 / self.fps
        while True:
            if not self.sendFrames(cb): break
            time.sleep(delay)
            if self._stop: break
        print('Replay `run` thread stopped')
        self._stop = True

    def createQueues(self, device: dai.Device):
        """
        Creates input queue for each enabled stream
        
        Args:
            device (dai.Device): Device to which we will stream frames
        """
        for name in self.xins:
            self._inputQueues[name] = device.getInputQueue(name + '_in')

    def getStreams(self) -> List[str]:
        streams: List[str] = []
        [streams.append(name) for name in self.reader.getStreams()]
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
        imgFrame: dai.ImgFrame
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
        frames = self.reader.read()
        if not frames:
            return False  # No more frames!

        for name, frame in frames.items():
            self.frames[name] = frame

        # Compress 3-plane frame to a single plane
        for name, frame in self.frames.items():
            if name in ["left", "right", "disparity"] and len(frame.shape) == 3:
                self.frames[name] = frame[:, :, 0]  # All 3 planes are the same
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

    def getShape(self, name: str) -> Tuple[int, int]:
        """
        Get shape of a stream
        """
        if name in self.reader.getStreams():
            return self.reader.getShape(name)

    def close(self):
        """
        Closes all video readers.
        """
        self._stop = True
        if self.thread:
            self.thread.join()
        self.reader.close()
