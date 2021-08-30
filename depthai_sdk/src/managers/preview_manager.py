import math

import cv2
import depthai as dai
from ..previews import Previews, MouseClickTracker


class PreviewManager:
    def __init__(self, display=[], nn_source=None, colorMap=cv2.COLORMAP_JET, dispMultiplier=255/96, mouseTracker=False, lowBandwidth=False, scale=None, sync=False, fps_handler=None):
        self.display = display
        self.frames = {}
        self.raw_frames = {}
        self.fps_handler = fps_handler
        self.nn_source = nn_source
        self.colorMap = colorMap
        self.lowBandwidth = lowBandwidth
        self.dispMultiplier = dispMultiplier
        self.mouse_tracker = MouseClickTracker() if mouseTracker else None
        self.scale = scale
        self.sync = sync

    def create_queues(self, device, callback=lambda *a, **k: None):
        if dai.CameraBoardSocket.LEFT in device.getConnectedCameras():
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
        self.output_queues = []
        for name in self.display:
            cv2.namedWindow(name)
            callback(name)
            if self.mouse_tracker is not None:
                cv2.setMouseCallback(name, self.mouse_tracker.select_point(name))
            if name not in (Previews.disparity_color.name, Previews.depth.name):  # generated on host
                self.output_queues.append(device.getOutputQueue(name=name, maxSize=1, blocking=False))

        if Previews.disparity_color.name in self.display and Previews.disparity.name not in self.display:
            self.output_queues.append(device.getOutputQueue(name=Previews.disparity.name, maxSize=1, blocking=False))
        if Previews.depth.name in self.display and Previews.depth_raw.name not in self.display:
            self.output_queues.append(device.getOutputQueue(name=Previews.depth_raw.name, maxSize=1, blocking=False))

    def prepare_frames(self, callback=lambda *a, **k: None):
        for queue in self.output_queues:
            packet = queue.tryGet()
            if packet is not None:
                if self.fps_handler is not None:
                    self.fps_handler.tick(queue.getName())
                frame = getattr(Previews, queue.getName()).value(packet, self)
                if frame is None:
                    print("[WARNING] Conversion of the {} frame has failed! (None value detected)".format(queue.getName()))
                    continue
                if self.scale is not None and queue.getName() in self.scale:
                    h, w = frame.shape[0:2]
                    frame = cv2.resize(frame, (int(w * self.scale[queue.getName()]), int(h * self.scale[queue.getName()])), interpolation=cv2.INTER_AREA)
                if queue.getName() in self.display:
                    callback(frame, queue.getName())
                    self.raw_frames[queue.getName()] = frame
                if self.mouse_tracker is not None:
                    if queue.getName() == Previews.disparity.name:
                        raw_frame = packet.getFrame() if not self.lowBandwidth else cv2.imdecode(packet.getData(), cv2.IMREAD_GRAYSCALE)
                        self.mouse_tracker.extract_value(Previews.disparity.name, raw_frame)
                        self.mouse_tracker.extract_value(Previews.disparity_color.name, raw_frame)
                    if queue.getName() == Previews.depth_raw.name:
                        raw_frame = packet.getFrame()  # if not self.lowBandwidth else cv2.imdecode(packet.getData(), cv2.IMREAD_UNCHANGED) TODO uncomment once depth encoding is possible
                        self.mouse_tracker.extract_value(Previews.depth_raw.name, raw_frame)
                        self.mouse_tracker.extract_value(Previews.depth.name, raw_frame)
                    else:
                        self.mouse_tracker.extract_value(queue.getName(), frame)

                if queue.getName() == Previews.disparity.name and Previews.disparity_color.name in self.display:
                    if self.fps_handler is not None:
                        self.fps_handler.tick(Previews.disparity_color.name)
                    self.raw_frames[Previews.disparity_color.name] = Previews.disparity_color.value(frame, self)

                if queue.getName() == Previews.depth_raw.name and Previews.depth.name in self.display:
                    if self.fps_handler is not None:
                        self.fps_handler.tick(Previews.depth.name)
                    self.raw_frames[Previews.depth.name] = Previews.depth.value(frame, self)

            for name in self.raw_frames:
                new_frame = self.raw_frames[name].copy()
                if name == Previews.depth_raw.name:
                    new_frame = cv2.normalize(new_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                self.frames[name] = new_frame

    def show_frames(self, callback=lambda *a, **k: None):
        for name, frame in self.frames.items():
            if self.mouse_tracker is not None:
                point = self.mouse_tracker.points.get(name)
                value = self.mouse_tracker.values.get(name)
                if point is not None:
                    cv2.circle(frame, point, 3, (255, 255, 255), -1)
                    cv2.putText(frame, str(value), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, str(value), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return_frame = callback(frame, name)  # Can be None, can be other frame e.g. after copy()
            cv2.imshow(name, return_frame if return_frame is not None else frame)

    def has(self, name):
        return name in self.frames

    def get(self, name):
        return self.frames.get(name, None)
