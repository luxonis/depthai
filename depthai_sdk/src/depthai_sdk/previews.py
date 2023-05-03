import enum
import math
from functools import partial

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from turbojpeg import TurboJPEG, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT, TJPF_GRAY
    turbo = TurboJPEG()
except:
    turbo = None


class PreviewDecoder:

    @staticmethod
    def jpegDecode(data, type):
        if turbo is not None:
            if type == cv2.IMREAD_GRAYSCALE:
                return turbo.decode(data, flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT, pixel_format=TJPF_GRAY)
            if type == cv2.IMREAD_UNCHANGED:
                return turbo.decode_to_yuv(data, flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)
            else:
                return turbo.decode(data, flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)
        else:
            return cv2.imdecode(data, type)

    @staticmethod
    def nnInput(packet, manager=None):
        """
        Produces NN passthough frame from raw data packet

        Args:
            packet (depthai.ImgFrame): Packet received from output queue
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        # if manager is not None and manager.decode: TODO change once passthrough frame type (8) is supported by VideoEncoder
        if False:
            frame = PreviewDecoder.jpegDecode(packet.getData(), cv2.IMREAD_COLOR)
        else:
            frame = packet.getCvFrame()
        if hasattr(manager, "nnSource") and manager.nnSource in (
                Previews.rectifiedLeft.name, Previews.rectifiedRight.name):
            frame = cv2.flip(frame, 1)
        return frame

    @staticmethod
    def color(packet, manager=None):
        """
        Produces color camera frame from raw data packet

        Args:
            packet (depthai.ImgFrame): Packet received from output queue
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        if manager is not None and manager.decode:
            return PreviewDecoder.jpegDecode(packet.getData(), cv2.IMREAD_COLOR)
        else:
            return packet.getCvFrame()

    @staticmethod
    def left(packet, manager=None):
        """
        Produces left camera frame from raw data packet

        Args:
            packet (depthai.ImgFrame): Packet received from output queue
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        if manager is not None and manager.decode:
            return PreviewDecoder.jpegDecode(packet.getData(), cv2.IMREAD_GRAYSCALE)
        else:
            return packet.getCvFrame()

    @staticmethod
    def right(packet, manager=None):
        """
        Produces right camera frame from raw data packet

        Args:
            packet (depthai.ImgFrame): Packet received from output queue
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        if manager is not None and manager.decode:
            return PreviewDecoder.jpegDecode(packet.getData(), cv2.IMREAD_GRAYSCALE)
        else:
            return packet.getCvFrame()

    @staticmethod
    def rectifiedLeft(packet, manager=None):
        """
        Produces rectified left frame (:obj:`depthai.node.StereoDepth.rectifiedLeft`) from raw data packet

        Args:
            packet (depthai.ImgFrame): Packet received from output queue
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        # if manager is not None and manager.decode:  # disabled to limit the memory usage
        if False:
            return PreviewDecoder.jpegDecode(packet.getData(), cv2.IMREAD_GRAYSCALE)
        else:
            return packet.getCvFrame()

    @staticmethod
    def rectifiedRight(packet, manager=None):
        """
        Produces rectified right frame (:obj:`depthai.node.StereoDepth.rectifiedRight`) from raw data packet

        Args:
            packet (depthai.ImgFrame): Packet received from output queue
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        # if manager is not None and manager.decode:  # disabled to limit the memory usage
        if False:
            return PreviewDecoder.jpegDecode(packet.getData(), cv2.IMREAD_GRAYSCALE)
        else:
            return packet.getCvFrame()

    @staticmethod
    def depthRaw(packet, manager=None):
        """
        Produces raw depth frame (:obj:`depthai.node.StereoDepth.depth`) from raw data packet

        Args:
            packet (depthai.ImgFrame): Packet received from output queue
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        # if manager is not None and manager.decode:  TODO change once depth frame type (14) is supported by VideoEncoder
        if False:
            return PreviewDecoder.jpegDecode(packet.getData(), cv2.IMREAD_UNCHANGED)
        else:
            return packet.getFrame()

    @staticmethod
    def depth(depthRaw, manager=None):
        """
        Produces depth frame from raw depth frame (converts to disparity and applies color map)

        Args:
            depthRaw (numpy.ndarray): OpenCV frame containing raw depth frame
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        if getattr(manager, "_depthConfig", None) is None:
            raise RuntimeError("Depth config has to be provided before decoding depth data")

        maxDisp = manager._depthConfig.getMaxDisparity()
        subpixelLevels = pow(2, manager._depthConfig.get().algorithmControl.subpixelFractionalBits)
        subpixel = manager._depthConfig.get().algorithmControl.enableSubpixel
        dispIntegerLevels = maxDisp if not subpixel else maxDisp / subpixelLevels
        dispScaleFactor = getattr(manager, "dispScaleFactor", None)
        if dispScaleFactor is None:
            baseline = getattr(manager, 'baseline', 75)  # mm
            fov = getattr(manager, 'fov', 71.86)
            focal = getattr(manager, 'focal', depthRaw.shape[1] / (2. * math.tan(math.radians(fov / 2))))
            dispScaleFactor = baseline * focal
            if manager is not None:
                setattr(manager, "dispScaleFactor", dispScaleFactor)
        with np.errstate(divide='ignore'):
            dispFrame = dispScaleFactor / depthRaw

        dispFrame = (dispFrame * 255. / dispIntegerLevels).astype(np.uint8)

        return PreviewDecoder.disparityColor(dispFrame, manager)

    @staticmethod
    def disparity(packet, manager=None):
        """
        Produces disparity frame (:obj:`depthai.node.StereoDepth.disparity`) from raw data packet

        Args:
            packet (depthai.ImgFrame): Packet received from output queue
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        if False:
            rawFrame = PreviewDecoder.jpegDecode(packet.getData(), cv2.IMREAD_GRAYSCALE)
        else:
            rawFrame = packet.getFrame()
        return (rawFrame * (manager.dispMultiplier if manager is not None else 255 / 96)).astype(np.uint8)

    @staticmethod
    def disparityColor(disparity, manager=None):
        """
        Applies color map to disparity frame

        Args:
            disparity (numpy.ndarray): OpenCV frame containing disparity frame
            manager (depthai_sdk.managers.PreviewManager, optional): PreviewManager instance

        Returns:
            numpy.ndarray: Ready to use OpenCV frame
        """
        return cv2.applyColorMap(disparity, manager.colorMap if manager is not None else cv2.COLORMAP_JET)


class Previews(enum.Enum):
    """
    Enum class, assigning preview name with decode function.

    Usually used as e.g. :code:`Previews.color.name` when specifying color preview name.

    Can be also used as e.g. :code:`Previews.color.value(packet)` to transform queue output packet to color camera frame
    """
    nnInput = partial(PreviewDecoder.nnInput)
    color = partial(PreviewDecoder.color)
    left = partial(PreviewDecoder.left)
    right = partial(PreviewDecoder.right)
    rectifiedLeft = partial(PreviewDecoder.rectifiedLeft)
    rectifiedRight = partial(PreviewDecoder.rectifiedRight)
    depthRaw = partial(PreviewDecoder.depthRaw)
    depth = partial(PreviewDecoder.depth)
    disparity = partial(PreviewDecoder.disparity)
    disparityColor = partial(PreviewDecoder.disparityColor)


class MouseClickTracker:
    """
    Class that allows to track the click events on preview windows and show pixel value of a frame in coordinates pointed
    by the user.

    Used internally by :obj:`depthai_sdk.managers.PreviewManager`
    """

    #: dict: Stores selected point position per frame
    points = {}
    #: dict: Stores values assigned to specific point per frame
    values = {}

    def selectPoint(self, name):
        """
        Returns callback function for :code:`cv2.setMouseCallback` that will update the selected point on mouse click
        event from frame.

        Usually used as

        .. code-block:: python

            mct = MouseClickTracker()
            # create preview window
            cv2.setMouseCallback(window_name, mct.select_point(window_name))

        Args:
            name (str): Name of the frame

        Returns:
            Callback function for :code:`cv2.setMouseCallback`
        """

        def cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                if self.points.get(name) == (x, y):
                    del self.points[name]
                    if name in self.values:
                        del self.values[name]
                else:
                    self.points[name] = (x, y)

        return cb

    def extractValue(self, name, frame: np.ndarray):
        """
        Extracts value from frame for a specific point

        Args:
            name (str): Name of the frame
        """
        point = self.points.get(name, None)
        if point is not None and frame is not None:
            if name in (Previews.depthRaw.name, Previews.depth.name):
                self.values[name] = "{}mm".format(frame[point[1]][point[0]])
            elif name in (Previews.disparityColor.name, Previews.disparity.name):
                self.values[name] = "{}px".format(frame[point[1]][point[0]])
            elif len(frame.shape) == 3:
                self.values[name] = "R:{},G:{},B:{}".format(*frame[point[1]][point[0]][::-1])
            elif len(frame.shape) == 2:
                self.values[name] = "Gray:{}".format(frame[point[1]][point[0]])
            else:
                self.values[name] = str(frame[point[1]][point[0]])
