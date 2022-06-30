#!/usr/bin/env python3
import array
from pathlib import Path
from queue import Queue
from threading import Thread
import depthai as dai
from enum import IntEnum

class EncodingQuality(IntEnum):
    BEST = 1 # Lossless MJPEG
    HIGH = 2 # MJPEG Quality=97 (default)
    MEDIUM = 3 # MJPEG Quality=93
    LOW = 4 # H265 BitrateKbps=10000

def _run(recorders, frameQ):
    """
    Start recording infinite loop
    """
    while True:
        try:
            frames = frameQ.get()
            if frames is None: # Terminate app
                break
            for name in frames:
                # Save all synced frames into files
                recorders[name].write(name, frames[name])
        except KeyboardInterrupt:
            break
    # Close all recorders - Can't use ExitStack with VideoWriter
    for n in recorders:
        recorders[n].close()
    print('Exiting store frame thread')

# class Recorder(IntEnum):
#     RAW = 1 # Save raw bitstream
#     MP4 = 2 # Containerize into mp4 file, requires `av` library

class Record():
    """
    This class records depthai streams from OAK cameras into different formats.
    Available formats: .h265, .mjpeg, .mp4, .mcap, .bag
    It will also save calibration .json, so depth reconstruction will 
    """

    save = ['color', 'left', 'right']
    _fps: int = 30
    _timelapse: int = -1
    quality = EncodingQuality.HIGH
    rotate = -1
    _preview: bool = False
    _mcap: bool = False
    _pointcloud: bool = False
    _mjpegQuality: int = None

    def __init__(self, path: Path, device: dai.Device):
        """
        Args:
            path (Path): Path to the recording folder
            device (dai.Device): OAK device object
        """
        self.device = device
        self.stereo = 1 < len(device.getConnectedCameras())
        self.mxid = device.getMxId()
        self.path = self._createFolder(path, self.mxid)

        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))


    def _getRecorders(self) -> dict:
        """
        Create recorders
        """
        recorders = dict()
        save = self.save.copy()

        if self._mcap:
            if self.quality == EncodingQuality.LOW or self.quality == EncodingQuality.BEST:
                raise Exception("MCAP only supports MEDIUM and HIGH quality!") # Foxglove Studio doesn't support Lossless MJPEG
            from .recorders.mcap_recorder import McapRecorder
            rec = McapRecorder(self.path, self.device)
            rec.setPointcloud(self._pointcloud)
            for name in save:
                recorders[name] = rec
            return recorders

        if 'depth' in save:
            from .recorders.rosbag_recorder import RosbagRecorder
            recorders['depth'] = RosbagRecorder(self.path, self.device, self.getSizes())
            save.remove('depth')

        if len(save) == 0: return recorders

        else:
            try:
                # Try importing av
                from .recorders.pyav_mp4_recorder import PyAvRecorder
                rec = PyAvRecorder(self.path, self.quality, self._fps)
            except:
                print("'av' library is not installed, depthai-record will save raw encoded streams.")
                from .recorders.raw_recorder import RawRecorder
                rec = RawRecorder(self.path, self.quality)
        # All other streams ("color", "left", "right", "disparity") will use
        # the same Raw/PyAv recorder
        for name in save:
            recorders[name] = rec
        return recorders

    def start(self):
        """
        Start recording process. This will create and start the pipeline,
        start recording threads, and initialize all queues.
        """
        if not self.stereo: # If device doesn't have stereo camera pair
            if "left" in self.save: self.save.remove("left")
            if "right" in self.save: self.save.remove("right")
            if "disparity" in self.save: self.save.remove("disparity")
            if "depth" in self.save: self.save.remove("depth")

        if self._preview: self.save.append('preview')

        if 0 < self._timelapse:
            self._fps = 5

        self.pipeline, self.nodes = self._createPipeline()

        self.frame_q = Queue(maxsize=20)
        self.process = Thread(target=_run, args=(self._getRecorders(), self.frame_q))
        self.process.start()

        self.device.startPipeline(self.pipeline)

        self.queues = []
        maxSize = 1 if 0 < self._timelapse else 10
        for stream in self.save:
            self.queues.append({
                'q': self.device.getOutputQueue(name=stream, maxSize=maxSize, blocking=False),
                'msgs': [],
                'name': stream,
                'mxid': self.mxid
            })

    def setFps(self, fps):
        """
        Sets FPS of all cameras. TODO: use SDK and its parser
        """
        self._fps = fps

    def setTimelapse(self, timelapseSec: int):
        """
        Sets number of seconds between each frame for the timelapse mode.
        """
        self._timelapse = timelapseSec
        
    def setQuality(self, quality):
        """
        Sets recording quality. Better recording quality consumes more disk space.
        """
        if type(quality) == str:
            self.quality = EncodingQuality[quality]
        elif type(quality) == int:
            self._mjpegQuality = quality
        else:
            raise Exception("Quality has to be either int or string!")

    def setPreview(self, preview: bool):
        """
        Whether we want to preview a color frame. TODO: use SDK and its parser to show streams that are specified by `-s`
        """
        self._preview = preview

    def setMcap(self, enable: bool):
        """
        Whether we want to record into MCAP file.
        """
        self._mcap = enable

    def setRecordStreams(self, save_streams: array):
        """
        Specify which streams to record to the disk on the host.
        """
        self.save = save_streams
        if "pointcloud" in self.save:
            self.save.remove("pointcloud")
            self._pointcloud = True
            # Only MCAP Pointcloud is supported
            self.setMcap(True)
            if "depth" not in self.save:
                # Depth is needed for Pointcloud
                self.save.append("depth")

    # def set_rotate(self, angle):
    #     """
    #     Available values for `angle`:
    #     - cv2.ROTATE_90_CLOCKWISE (0)
    #     - cv2.ROTATE_180 (1)
    #     - cv2.ROTATE_90_COUNTERCLOCKWISE (2)
    #     """
    #     raise Exception("Rotating not yet supported!")
    #     # Currently RealSense Viewer throws error "memory access violation". Debug.
    #     self.rotate = angle

    def getSizes(self):
        dict = {}
        if "color" in self.save: dict['color'] = self.nodes['color'].getVideoSize()
        if "right" in self.save: dict['right'] = self.nodes['right'].getResolutionSize()
        if "left" in self.save: dict['left'] = self.nodes['left'].getResolutionSize()
        if "disparity" in self.save: dict['disparity'] = self.nodes['left'].getResolutionSize()
        if "depth" in self.save: dict['depth'] = self.nodes['left'].getResolutionSize()
        return dict

    def _createFolder(self, path: Path, mxid: str) -> Path:
        """
        Creates recording folder
        """
        i = 0
        while True:
            i += 1
            recordings_path = path / f"{i}-{str(mxid)}"
            if not recordings_path.is_dir():
                recordings_path.mkdir(parents=True, exist_ok=False)
                return recordings_path

    def _createPipeline(self):
        """
        Creates depthai pipeline for recording
        """

        pipeline = dai.Pipeline()
        nodes = {}

        def create_mono(name):
            nodes[name] = pipeline.create(dai.node.MonoCamera)
            nodes[name].setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            socket = dai.CameraBoardSocket.LEFT if name == "left" else dai.CameraBoardSocket.RIGHT
            nodes[name].setBoardSocket(socket)
            nodes[name].setFps(self._fps)

        def stream_out(name, fps, out, noEnc=False):
            # Create XLinkOutputs for the stream
            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName(name)
            if noEnc:
                out.link(xout.input)
                return

            encoder = pipeline.create(dai.node.VideoEncoder)
            profile = dai.VideoEncoderProperties.Profile.H265_MAIN if self.quality == EncodingQuality.LOW else dai.VideoEncoderProperties.Profile.MJPEG
            encoder.setDefaultProfilePreset(fps, profile)

            if self.quality == EncodingQuality.BEST:
                encoder.setLossless(True)
            elif self.quality == EncodingQuality.HIGH:
                quality = 97 if self._mjpegQuality is None else self._mjpegQuality
                encoder.setQuality(quality)
            elif self.quality == EncodingQuality.MEDIUM:
                encoder.setQuality(93)
            elif self.quality == EncodingQuality.LOW:
                encoder.setBitrateKbps(10000)

            out.link(encoder.input)
            encoder.bitstream.link(xout.input)

        if "color" in self.save:
            nodes['color'] = pipeline.create(dai.node.ColorCamera)
            nodes['color'].setBoardSocket(dai.CameraBoardSocket.RGB)
            # RealSense Viewer expects RGB color order
            nodes['color'].setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            nodes['color'].setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            nodes['color'].setIspScale(1,2) # 1080P
            nodes['color'].setFps(self._fps)

            if self._preview:
                nodes['color'].setPreviewSize(640, 360)
                stream_out("preview", None, nodes['color'].preview, noEnc=True)

            # TODO change out to .isp instead of .video when ImageManip will support I420 -> NV12
            # Don't encode color stream if we save depth; as we will be saving color frames in rosbags as well
            stream_out("color", nodes['color'].getFps(), nodes['color'].video) #, noEnc='depth' in self.save)

        if True in (el in ["left", "disparity", "depth"] for el in self.save):
            create_mono("left")
            if "left" in self.save:
                stream_out("left", nodes['left'].getFps(), nodes['left'].out)

        if True in (el in ["right", "disparity", "depth"] for el in self.save):
            create_mono("right")
            if "right" in self.save:
                stream_out("right", nodes['right'].getFps(), nodes['right'].out)

        if True in (el in ["disparity", "depth"] for el in self.save):
            nodes['stereo'] = pipeline.create(dai.node.StereoDepth)

            nodes['stereo'].initialConfig.setConfidenceThreshold(255)
            nodes['stereo'].initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
            nodes['stereo'].setLeftRightCheck(True)
            nodes['stereo'].setExtendedDisparity(True)

            # if "disparity" not in self.save and "depth" in self.save:
            #     nodes['stereo'].setSubpixel(True) # For better depth visualization

            # if "depth" and "color" in self.save: # RGB depth alignment
            #     nodes['color'].setIspScale(1,3) # 4k -> 720P
            #     # For now, RGB needs fixed focus to properly align with depth.
            #     # This value was used during calibration
            #     nodes['color'].initialControl.setManualFocus(130)
            #     nodes['stereo'].setDepthAlign(dai.CameraBoardSocket.RGB)

            nodes['left'].out.link(nodes['stereo'].left)
            nodes['right'].out.link(nodes['stereo'].right)

            if "disparity" in self.save:
                stream_out("disparity", nodes['right'].getFps(), nodes['stereo'].disparity)
            if "depth" in self.save:
                stream_out('depth', None, nodes['stereo'].depth, noEnc=True)

        self.nodes = nodes
        self.pipeline = pipeline
        return pipeline, nodes

