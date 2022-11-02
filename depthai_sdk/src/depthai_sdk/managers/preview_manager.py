import math
from datetime import timedelta
from queue import Queue

import cv2
import depthai as dai

from ..previews import Previews, MouseClickTracker
import numpy as np


class PreviewManager:
    """
    Manager class that handles frames and displays them correctly.
    """

    #: dict: Contains name -> frame mapping that can be used to modify specific frames directly
    frames = {}

    def __init__(self, display=[], nnSource=None, colorMap=None, depthConfig=None, dispMultiplier=255/96, mouseTracker=False, decode=False, fpsHandler=None, createWindows=True):
        """
        Args:
            display (list, Optional): List of :obj:`depthai_sdk.Previews` objects representing the streams to display
            mouseTracker (bool, Optional): If set to :code:`True`, will enable mouse tracker on the preview windows that will display selected pixel value
            fpsHandler (depthai_sdk.fps.FPSHandler, Optional): if provided, will use fps handler to modify stream FPS and display it
            nnSource (str, Optional): Specifies NN source camera
            colorMap (cv2 color map, Optional): Color map applied on the depth frames
            decode (bool, Optional): If set to :code:`True`, will decode the received frames assuming they were encoded with MJPEG encoding
            dispMultiplier (float, Optional): Multiplier used for depth <-> disparity calculations (calculated on baseline and focal)
            depthConfig (depthai.StereoDepthConfig, optional): Configuration used for depth <-> disparity calculations
            createWindows (bool, Optional): If True, will create preview windows using OpenCV (enabled by default)
        """
        self.nnSource = nnSource
        if colorMap is not None:
            self.colorMap = colorMap
        else:
            self.colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
            self.colorMap[0] = [0, 0, 0]
        self.decode = decode
        self.dispMultiplier = dispMultiplier
        self._depthConfig = depthConfig
        self._fpsHandler = fpsHandler
        self._mouseTracker = MouseClickTracker() if mouseTracker else None
        self._display = display
        self._createWindows = createWindows
        self._rawFrames = {}

    def collectCalibData(self, device):
        """
        Collects calibration data and calculates :attr:`dispScaleFactor` accordingly

        Args:
            device (depthai.Device): Running device instance
        """

        calib = device.readCalibration()
        eeprom = calib.getEepromData()
        leftCam = calib.getStereoLeftCameraId()
        if leftCam != dai.CameraBoardSocket.AUTO and leftCam in eeprom.cameraData.keys():
            camInfo = eeprom.cameraData[leftCam]
            self.baseline = abs(camInfo.extrinsics.specTranslation.x * 10)  # cm -> mm
            self.fov = calib.getFov(calib.getStereoLeftCameraId())
            self.focal = (camInfo.width / 2) / (2. * math.tan(math.radians(self.fov / 2)))
        else:
            print("Warning: calibration data missing, using OAK-D defaults")
            self.baseline = 75
            self.fov = 71.86
            self.focal = 440
        self.dispScaleFactor = self.baseline * self.focal

    def createQueues(self, device, callback=None):
        """
        Create output queues for requested preview streams

        Args:
            device (depthai.Device): Running device instance
            callback (func, Optional): Function that will be executed with preview name once preview window was created
        """
        self.outputQueues = []
        for name in self._display:
            if self._createWindows:
                cv2.namedWindow(name)
            if callable(callback):
                callback(name)
            if self._createWindows and self._mouseTracker is not None:
                cv2.setMouseCallback(name, self._mouseTracker.selectPoint(name))
            if name not in (Previews.disparityColor.name, Previews.depth.name):  # generated on host
                self.outputQueues.append(device.getOutputQueue(name=name, maxSize=1, blocking=False))

        if Previews.disparityColor.name in self._display and Previews.disparity.name not in self._display:
            self.outputQueues.append(device.getOutputQueue(name=Previews.disparity.name, maxSize=1, blocking=False))
        if Previews.depth.name in self._display and Previews.depthRaw.name not in self._display:
            self.outputQueues.append(device.getOutputQueue(name=Previews.depthRaw.name, maxSize=1, blocking=False))

    def closeQueues(self):
        """
        Closes output queues for requested preview streams
        """

        for queue in self.outputQueues:
            queue.close()

    def _processFrame(self, frame, queueName):
        if self._fpsHandler is not None:
            self._fpsHandler.tick(queueName)
        return frame

    def _addRawFrame(self, frame, packet, name):
        if name in self._display:
            self._rawFrames[name] = frame

        if self._mouseTracker is not None:
            if name == Previews.disparity.name:
                rawFrame = packet.getFrame() if not self.decode else cv2.imdecode(packet.getData(), cv2.IMREAD_GRAYSCALE)
                self._mouseTracker.extractValue(Previews.disparity.name, rawFrame)
                self._mouseTracker.extractValue(Previews.disparityColor.name, rawFrame)
            if name == Previews.depthRaw.name:
                rawFrame = packet.getFrame()  # if not self.decode else cv2.imdecode(packet.getData(), cv2.IMREAD_UNCHANGED) TODO uncomment once depth encoding is possible
                self._mouseTracker.extractValue(Previews.depthRaw.name, rawFrame)
                self._mouseTracker.extractValue(Previews.depth.name, rawFrame)
            else:
                self._mouseTracker.extractValue(name, frame)

        if name == Previews.disparity.name and Previews.disparityColor.name in self._display:
            if self._fpsHandler is not None:
                self._fpsHandler.tick(Previews.disparityColor.name)
            self._rawFrames[Previews.disparityColor.name] = Previews.disparityColor.value(frame, self)

        if name == Previews.depthRaw.name and Previews.depth.name in self._display:
            if self._fpsHandler is not None:
                self._fpsHandler.tick(Previews.depth.name)
            self._rawFrames[Previews.depth.name] = Previews.depth.value(frame, self)


    def prepareFrames(self, blocking=False, callback=None):
        """
        This function consumes output queues' packets and parses them to obtain ready to use frames.
        To convert the frames from packets, this manager uses methods defined in :obj:`depthai_sdk.previews.PreviewDecoder`.

        Args:
            blocking (bool, Optional): If set to :code:`True`, will wait for a packet in each queue to be available
            callback (func, Optional): Function that will be executed once packet with frame has arrived
        """
        for queue in self.outputQueues:
            if blocking:
                packet = queue.get()
            else:
                packet = queue.tryGet()
            if packet is not None:
                frame = getattr(Previews, queue.getName()).value(packet, self)
                if frame is None:
                    print("[WARNING] Conversion of the {} frame has failed! (None value detected)".format(queue.getName()))
                    continue
                frame = self._processFrame(frame, queue.getName())
                if queue.getName() in self._display:
                    if callback is not None:
                        callback(frame, queue.getName())
                self._addRawFrame(frame, packet, queue.getName())

        for name in self._rawFrames:
            newFrame = self._rawFrames[name].copy()
            if name == Previews.depthRaw.name:
                newFrame = cv2.normalize(newFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            self.frames[name] = newFrame

    def showFrames(self, callback=None):
        """
        Displays stored frame onto preview windows.

        Args:
            callback (func, Optional): Function that will be executed right before :code:`cv2.imshow`
        """
        for name, frame in self.frames.items():
            if self._mouseTracker is not None:
                point = self._mouseTracker.points.get(name)
                value = self._mouseTracker.values.get(name)
                if point is not None:
                    cv2.circle(frame, point, 3, (255, 255, 255), -1)
                    cv2.putText(frame, str(value), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, str(value), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if self._fpsHandler is not None:
                self._fpsHandler.drawFps(frame, name)
            if callable(callback):
                newFrame = callback(frame, name)
                if newFrame is not None:
                    frame = newFrame
            if self._createWindows:
                cv2.imshow(name, frame)

    def has(self, name):
        """
        Determines whether manager has a frame assigned to specified preview

        Returns:
            bool: :code:`True` if contains a frame, :code:`False` otherwise
        """
        return name in self.frames

    def get(self, name):
        """
        Returns a frame assigned to specified preview

        Returns:
            numpy.ndarray: Resolved frame, will default to :code:`None` if not present
        """
        return self.frames.get(name, None)


class SyncedPreviewManager(PreviewManager):
    """
    Extension of the regular PreviewManager that allows to display all of the frames in sync
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seqPackets = {}
        self._packetsQ = None
        self._lastSeqs = {}
        self._syncedPackets = {}
        self.nnSyncSeq = None

    def __get_next_seq_packet(self, seqKey, name, defaultPacket):
        if seqKey not in self._seqPackets:
            return defaultPacket
        elif name not in self._seqPackets:
            return self.__get_next_seq_packet(seqKey + 1, name, defaultPacket)
        else:
            return self._seqPackets[seqKey][name]

    def prepareFrames(self, blocking=False, callback=None):
        """
        This overridden function serves the same purpose - to prepare ready to use frames - but before it provide any data
        it will sync all of the packets first, using their sequence number. So any frames retrievable from this class, after
        this method is called, will be in sync with each other.

        Args:
            blocking (bool, Optional): If set to :code:`True`, will wait for a packet in each queue to be available
            callback (func, Optional): Function that will be executed once packet with frame has arrived
        """

        for queue in self.outputQueues:
            if blocking:
                packet = queue.get()
            else:
                packet = queue.tryGet()
            if packet is not None:
                seq = packet.getSequenceNum()
                packets = self._seqPackets.get(seq, {})
                packets[queue.getName()] = packet
                self._seqPackets[seq] = packets
                self._lastSeqs[queue.getName()] = seq

                if len(packets) == len(self.outputQueues):
                    self._packetsQ = Queue(maxsize=100)
                    unsyncedSeq = sorted(list(filter(lambda itemSeq: itemSeq < seq, self._seqPackets.keys())))
                    for seqKey in unsyncedSeq:
                        unsynced = {
                            synced_name: self.__get_next_seq_packet(seqKey, synced_name, synced_packet)
                            for synced_name, synced_packet in packets.items()
                        }
                        self._packetsQ.put(unsynced)
                        del self._seqPackets[seqKey]

        if self._packetsQ is not None:
            try:
                packets = self._packetsQ.get_nowait()
            except:
                packets = None
            if packets is not None:
                self.nnSyncSeq = min(map(lambda packet: packet.getSequenceNum(), packets.values()))
                for name, packet in packets.items():
                    frame = getattr(Previews, name).value(packet, self)
                    if frame is None:
                        print("[WARNING] Conversion of the {} frame has failed! (None value detected)".format(name))
                        continue
                    frame = self._processFrame(frame, name)
                    if callback is not None:
                        callback(frame, name)
                    self._addRawFrame(frame, packet, name)

        for name in self._rawFrames:
            newFrame = self._rawFrames[name].copy()
            if name == Previews.depthRaw.name:
                newFrame = cv2.normalize(newFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            self.frames[name] = newFrame
