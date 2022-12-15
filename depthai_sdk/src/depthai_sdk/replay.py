import os
import time
from enum import IntEnum
from threading import Thread
from time import monotonic

import depthai as dai
import numpy as np

from depthai_sdk.readers.abstract_reader import AbstractReader
from depthai_sdk.utils import *

_fileTypes = ['color', 'left', 'right', 'disparity', 'depth']
_videoExt = ['.mjpeg', '.avi', '.mp4', '.h265', '.h264', '.webm']
_imageExt = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm',
             '.pnm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']

class ReplayStream:
    node: dai.node.XLinkIn
    stream_name: str # XLink stream name
    queue: dai.DataInputQueue # Input queue
    frame: np.ndarray # Last read frame from Reader (ndarray)
    imgFrame: dai.ImgFrame # Last read ImgFrame from Reader (dai.ImgFrame)
    shape: Tuple[int,int] # width, height
    disabled: bool
    size: int # bytes
    socket_num: dai.CameraBoardSocket = None

    def __init__(self):
        self.disabled = False
        self.stream_name = ''

    def get_socket(self) -> dai.CameraBoardSocket:
        if self.socket_num:
            return self.socket_num

        if 'color' in self.stream_name or 'rgb' in self.stream_name:
            return dai.CameraBoardSocket.RGB
        elif 'left' in self.stream_name:
            return dai.CameraBoardSocket.LEFT
        elif 'right' in self.stream_name:
            return dai.CameraBoardSocket.RIGHT

class Replay:
    streams: Dict[str, ReplayStream] = dict()

    _seqNum = 0  # Frame sequence number, added to each imgFrame
    _now: monotonic = None
    _colorSize = None
    _keepAR = True  # By default, crop image as needed to keep the aspect ratio
    _pause = False
    _calibData = None

    fps: float = 30.0
    thread: Thread = None
    _stop: bool = False  # Stop the thread that's sending frames to the OAK camera

    reader: AbstractReader = None

    def __init__(self, path: str):
        """
        Helper file to replay recorded depthai stream. It reads from recorded files (mjpeg/avi/mp4/h265/h264/bag)
        and sends frames back to OAK camera to replay the scene, including depth reconstruction from 2 synced mono
        streams.
    
        Args:
            path (str): Path to the recording folder
        """
        self.path = self._get_path(path)

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
            calibFile = self.path / 'calib.json'
            if calibFile.exists():
                self._calibData = dai.CalibrationHandler(str(calibFile))

        else:  # Provided path is a file
            if self.path.suffix in _videoExt:
                from .readers.videocap_reader import VideoCapReader
                self.reader = VideoCapReader(self.path)
            elif self.path.suffix in _imageExt:
                from .readers.image_reader import ImageReader
                self.reader = ImageReader(self.path)
            else:
                raise NotImplementedError('Please select folder')

        # Read all available streams
        for stream_name in self.reader.getStreams():
            stream = ReplayStream()
            stream.shape = self.reader.getShape(stream_name)
            stream.size = self.reader.get_message_size(stream_name)
            self.streams[stream_name] = stream

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

    def disableStream(self, stream_name: str, disableReading: bool = False):
        """
        Disable sending a recorded stream to the device.

        Args:
            streamName(str): Name of the stream to disable (eg. 'left', 'color', 'depth', etc.)
            disableReading (bool, Optional): Also disable reading frames from the file
        """
        if disableReading:
            self.reader.disableStream(stream_name)

        if stream_name not in self.streams:
            print(f"There's no stream '{stream_name}' available!")
            return

        self.streams[stream_name].disabled = True

    def specify_socket(self, stream_name: str, socket: dai.CameraBoardSocket):
        if stream_name not in self.streams:
            print(f"There's no stream '{stream_name}' available!")
            return

        self.streams[stream_name].socket_num = socket


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
        for stream_name, stream in self.streams.items():
            stream.imgFrame =self._createImgFrame(stream.frame, stream.get_socket())
            # Save the imgFrame
            if cb:  # callback
                cb(stream_name, stream.imgFrame)

            # Don't send these frames to the OAK camera
            if stream.disabled: continue

            # Send an imgFrame to the OAK camera
            stream.queue.send(stream.imgFrame)

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

        def createXIn(p: dai.Pipeline, xlink_stream_name: str, size: int):
            xin = p.create(dai.node.XLinkIn)
            xin.setMaxDataSize(size)
            xin.setStreamName(xlink_stream_name)
            return xin

        for name, stream in self.streams.items():
            if stream.disabled: continue

            stream.stream_name = name + '_in'
            stream.node = createXIn(pipeline, stream.stream_name, stream.size)

        return pipeline

    def initStereoDepth(self, stereo: dai.node.StereoDepth, left_name: str='left', right_name: str='right', align_to: str = 'color'):
        streams = self.reader.getStreams()
        if 'left' not in streams or 'right' not in streams:
            raise Exception("Tried to init StereoDepth, but left/right streams aren't available!")
        stereo.setInputResolution(self.getShape(left_name))

        if align_to:  # Enable RGB-depth alignment
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            if self._colorSize is not None:
                stereo.setOutputSize(*self._colorSize)
            else:
                stereo.setOutputSize(*self.getShape(align_to))

        self.streams[left_name].out.link(stereo.left)
        self.streams[right_name].out.link(stereo.right)

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
        for name, stream in self.streams.items():
            if stream.stream_name:
                stream.queue = device.getInputQueue(stream.stream_name)

    def getStreams(self) -> List[str]:
        return [name for name, stream in self.streams.items()]

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

    def _createImgFrame(self, cvFrame: np.ndarray, socket: dai.CameraBoardSocket) -> dai.ImgFrame:
        imgFrame: dai.ImgFrame = None

        if cvFrame.shape[-1] == 3:
            # Resize/crop color frame as specified by the user
            cvFrame = self._resizeColor(cvFrame)
            # cv2 reads frames in interleaved format, and most networks expect planar by default
            cvFrame = toPlanar(cvFrame)
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.BGR888p)
            imgFrame.setInstanceNum(socket)
        elif cvFrame.dtype == np.uint8:
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.RAW8)
            imgFrame.setInstanceNum(socket)
        elif cvFrame.dtype == np.uint16:
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.RAW16)

        return imgFrame

    def _readFrames(self) -> bool:
        """
        Reads frames from all Readers.
        
        Returns:
            bool: True if successful, otherwise False.
        """
        frames = self.reader.read()
        if not frames:
            return False  # No more frames!

        for name, frame in frames.items():
            self.streams[name].frame = frame

        # Compress 3-plane frame to a single plane
        # for name, frame in self.frames.items():
        #     if name in ["left", "right", "disparity"] and len(frame.shape) == 3:
        #         self.frames[name] = frame[:, :, 0]  # All 3 planes are the same
        return True


    def getShape(self, name: str) -> Tuple[int, int]:
        """
        Get shape of a stream
        """
        return self.streams[name].shape

    def close(self):
        """
        Closes all video readers.
        """
        self._stop = True
        if self.thread:
            self.thread.join()
        self.reader.close()
