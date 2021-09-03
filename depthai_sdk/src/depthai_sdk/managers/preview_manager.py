import math

import cv2
import depthai as dai
from ..previews import Previews, MouseClickTracker


class PreviewManager:
    """
    Manager class that handles frames and displays them correctly.
    """

    #: dict: Contains name -> frame mapping that can be used to modify specific frames directly
    frames = {}

    def __init__(self, display=[], nn_source=None, colorMap=cv2.COLORMAP_JET, dispMultiplier=255/96, mouseTracker=False, lowBandwidth=False, scale=None, sync=False, fps_handler=None):
        """
        Args:
            display (list): List of :obj:`depthai_sdk.Previews` objects representing the streams to display
            mouseTracker (bool): If set to :code:`True`, will enable mouse tracker on the preview windows that will display selected pixel value
            fps_handler (depthai_sdk.fps.FPSHandler): if provided, will use fps handler to modify stream FPS and display it
            sync (bool): If set to :code:`True`, will assume that neural network source camera will not contain raw frame but scaled frame used by NN
            nn_source (str): Specifies NN source camera
            colorMap: Color map applied on the depth frames
            lowBandwidth (bool): If set to :code:`True`, will decode the received frames assuming they were encoded with MJPEG encoding
            scale (dict): Allows to scale down frames before preview. Useful when previewing e.g. 4K frames
            dispMultiplier (float): Value used for depth <-> disparity calculations
        """
        self.sync = sync
        self.nn_source = nn_source
        self.colorMap = colorMap
        self.lowBandwidth = lowBandwidth
        self.scale = scale
        self.dispMultiplier = dispMultiplier
        self._fps_handler = fps_handler
        self._mouse_tracker = MouseClickTracker() if mouseTracker else None
        self._display = display
        self._raw_frames = {}


    def collect_calib_data(self, device):
        """
        Collects calibration data and calculates :attr:`dispScaleFactor` accordingly

        Args:
            device (depthai.Device): Running device instance
        """

        calib = device.readCalibration()
        eeprom = calib.getEepromData()
        left_cam = calib.getStereoLeftCameraId()
        if left_cam != dai.CameraBoardSocket.AUTO:
            cam_info = eeprom.cameraData[left_cam]
            self.baseline = abs(cam_info.extrinsics.specTranslation.x * 10)  # cm -> mm
            self.fov = calib.getFov(calib.getStereoLeftCameraId())
            self.focal = (cam_info.width / 2) / (2. * math.tan(math.radians(self.fov / 2)))
        else:
            print("Warning: calibration data missing, using OAK-D defaults")
            self.baseline = 75
            self.fov = 71.86
            self.focal = 440
        self.dispScaleFactor = self.baseline * self.focal

    def create_queues(self, device, callback=None):
        """
        Create output queues for requested preview streams

        Args:
            device (depthai.Device): Running device instance
            callback (func): Function that will be executed with preview name once preview window was created
        """
        self.output_queues = []
        for name in self._display:
            cv2.namedWindow(name)
            if callable(callback):
                callback(name)
            if self._mouse_tracker is not None:
                cv2.setMouseCallback(name, self._mouse_tracker.select_point(name))
            if name not in (Previews.disparity_color.name, Previews.depth.name):  # generated on host
                self.output_queues.append(device.getOutputQueue(name=name, maxSize=1, blocking=False))

        if Previews.disparity_color.name in self._display and Previews.disparity.name not in self._display:
            self.output_queues.append(device.getOutputQueue(name=Previews.disparity.name, maxSize=1, blocking=False))
        if Previews.depth.name in self._display and Previews.depth_raw.name not in self._display:
            self.output_queues.append(device.getOutputQueue(name=Previews.depth_raw.name, maxSize=1, blocking=False))

    def prepare_frames(self, callback=None):
        """
        This function consumes output queues' packets and parses them to obtain ready to use frames.
        To convert the frames from packets, this manager uses methods defined in :obj:`depthai_sdk.previews.PreviewDecoder`.

        Args:
            callback (func): Function that will be executed once packet with frame has arrived
        """
        for queue in self.output_queues:
            packet = queue.tryGet()
            if packet is not None:
                if self._fps_handler is not None:
                    self._fps_handler.tick(queue.getName())
                frame = getattr(Previews, queue.getName()).value(packet, self)
                if frame is None:
                    print("[WARNING] Conversion of the {} frame has failed! (None value detected)".format(queue.getName()))
                    continue
                if self.scale is not None and queue.getName() in self.scale:
                    h, w = frame.shape[0:2]
                    frame = cv2.resize(frame, (int(w * self.scale[queue.getName()]), int(h * self.scale[queue.getName()])), interpolation=cv2.INTER_AREA)
                if queue.getName() in self._display:
                    if callback is not None:
                        callback(frame, queue.getName())
                    self._raw_frames[queue.getName()] = frame
                if self._mouse_tracker is not None:
                    if queue.getName() == Previews.disparity.name:
                        raw_frame = packet.getFrame() if not self.lowBandwidth else cv2.imdecode(packet.getData(), cv2.IMREAD_GRAYSCALE)
                        self._mouse_tracker.extract_value(Previews.disparity.name, raw_frame)
                        self._mouse_tracker.extract_value(Previews.disparity_color.name, raw_frame)
                    if queue.getName() == Previews.depth_raw.name:
                        raw_frame = packet.getFrame()  # if not self.lowBandwidth else cv2.imdecode(packet.getData(), cv2.IMREAD_UNCHANGED) TODO uncomment once depth encoding is possible
                        self._mouse_tracker.extract_value(Previews.depth_raw.name, raw_frame)
                        self._mouse_tracker.extract_value(Previews.depth.name, raw_frame)
                    else:
                        self._mouse_tracker.extract_value(queue.getName(), frame)

                if queue.getName() == Previews.disparity.name and Previews.disparity_color.name in self._display:
                    if self._fps_handler is not None:
                        self._fps_handler.tick(Previews.disparity_color.name)
                    self._raw_frames[Previews.disparity_color.name] = Previews.disparity_color.value(frame, self)

                if queue.getName() == Previews.depth_raw.name and Previews.depth.name in self._display:
                    if self._fps_handler is not None:
                        self._fps_handler.tick(Previews.depth.name)
                    self._raw_frames[Previews.depth.name] = Previews.depth.value(frame, self)

            for name in self._raw_frames:
                new_frame = self._raw_frames[name].copy()
                if name == Previews.depth_raw.name:
                    new_frame = cv2.normalize(new_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                self.frames[name] = new_frame

    def show_frames(self, callback=None):
        """
        Displays stored frame onto preview windows.

        Args:
            callback (func): Function that will be executed right before :code:`cv2.imshow`
        """
        for name, frame in self.frames.items():
            if self._mouse_tracker is not None:
                point = self._mouse_tracker.points.get(name)
                value = self._mouse_tracker.values.get(name)
                if point is not None:
                    cv2.circle(frame, point, 3, (255, 255, 255), -1)
                    cv2.putText(frame, str(value), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, str(value), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if callable(callback):
                frame = callback(frame, name)
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
