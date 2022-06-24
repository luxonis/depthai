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
    _supportedExtensions = ['mjpeg', 'avi', 'mp4', 'h265', 'h264', 'bag']
    _inputQueues = dict() # dai.InputQueue dictionary for each stream
    _seqNum = 0 # Frame sequence number, added to each imgFrame
    _start = datetime.datetime.now() # For frame timestamp
    _now = datetime.datetime.now()
    _colorSize = None
    _keepAR = True # By default crop image as needed to keep the aspect ratio

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
        for file in os.listdir(path):
            if not '.' in file: continue # Folder
            name, extension = file.split('.')
            if name in self._fileTypes and extension in self._supportedExtensions:
                if extension == 'bag':
                    # For .bag files
                    from .video_readers.rosbag_reader import RosbagReader
                    self.readers[name] = RosbagReader(str(self.path / file))
                else:
                    # For .mjpeg / .h265 / .mp4 files
                    from .video_readers.videocap_reader import VideoCapReader
                    self.readers[name] = VideoCapReader(str(self.path / file))

        if len(self.readers) == 0:
            raise RuntimeError("There are no recordings in the folder specified.")

        # Load calibration data from the recording folder
        self._calibData = dai.CalibrationHandler(str(self.path / "calib.json"))

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
        if streamName not in self.readers:
            print(f"There's no stream '{streamName}' available!")
            return
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
            print(f"sending frame {name} seq {imgFrame.getSequenceNum()} ts {imgFrame.getTimestamp()}")

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
            return xin

        for name in self.readers:
            if name not in self.disabledStreams:
                setattr(nodes, name, createXIn(pipeline, name))

        if hasattr(nodes, 'left') and hasattr(nodes, 'right'): # Create StereoDepth node
            nodes.stereo = pipeline.create(dai.node.StereoDepth)
            nodes.stereo.setInputResolution(self.readers['left'].getShape())

            nodes.left.out.link(nodes.stereo.left)
            nodes.right.out.link(nodes.stereo.right)

        return pipeline, nodes

    def createQueues(self, device: dai.Device):
        """
        Creates input queue for each enabled stream
        
        Args:
            device (dai.Device): Device to which we will stream frames
        """
        for name in self.readers:
            if name in self._streamTypes and name not in self.disabledStreams:
                self._inputQueues[name] = device.getInputQueue(name+'_in')

    def _resizeColor(self, frame: cv2.Mat) -> cv2.Mat:
        if self._colorSize is None:
            # No resizing needed
            return frame

        if not self._keepAR:
            # No need to keep aspect ratio, image will be squished
            return cv2.resize(frame, self._colorSize)

        h = frame.shape[0]
        w = frame.shape[1]
        desired_ratio = self._colorSize[0] / self._colorSize[1]
        current_ratio = w / h

        # Crop width/heigth to match the aspect ratio needed by the NN
        if desired_ratio < current_ratio: # Crop width
            # Use full height, crop width
            new_w = (desired_ratio/current_ratio) * w
            crop = int((w - new_w) / 2)
            preview = frame[:, crop:w-crop]
        else: # Crop height
            # Use full width, crop height
            new_h = (current_ratio/desired_ratio) * h
            crop = int((h - new_h) / 2)
            preview = frame[crop:h-crop,:]

        return cv2.resize(preview, self._colorSize)

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
        Reads one frame from each Reader.
        
        Returns:
            bool: True if successful, otherwise False.
        """
        self.frames = dict()
        for name in self.readers:
            self.frames[name] = self.readers[name].read() # Read the frame
            
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
        size = self.readers[name].getShape()
        bytes_per_pixel = 1
        if name == 'color': bytes_per_pixel = 3
        elif name == 'depth': bytes_per_pixel = 2 # 16bit
        return size[0] * size[1] * bytes_per_pixel

    def close(self):
        """
        Closes all video readers.
        """
        for name in self.readers:
            self.readers[name].close()