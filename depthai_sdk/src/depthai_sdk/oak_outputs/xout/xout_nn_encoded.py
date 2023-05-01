from typing import Tuple, List

import depthai as dai

from depthai_sdk.classes.packets import FramePacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_h26x import XoutH26x
from depthai_sdk.oak_outputs.xout.xout_mjpeg import XoutMjpeg
from depthai_sdk.oak_outputs.xout.xout_nn import XoutNnResults


class XoutNnH26x(XoutNnResults, XoutH26x):
    name: str = "H26x NN Results"
    # Streams
    frames: StreamXout
    nn_results: StreamXout

    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 nn_results: StreamXout,
                 color: bool,
                 profile: dai.VideoEncoderProperties.Profile,
                 fps: float,
                 frame_shape: Tuple[int, ...]):
        self.nn_results = nn_results

        XoutH26x.__init__(self, frames, color, profile, fps, frame_shape)
        XoutNnResults.__init__(self, det_nn, frames, nn_results)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results]

    def visualize(self, packet: FramePacket):
        decoded_frame = XoutH26x.decode_frame(self, packet)
        if decoded_frame is None:
            return

        packet.frame = decoded_frame
        XoutNnResults.visualize(self, packet)


class XoutNnMjpeg(XoutNnResults, XoutMjpeg):
    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 nn_results: StreamXout,
                 color: bool,
                 lossless: bool,
                 fps: float,
                 frame_shape: Tuple[int, ...]):
        self.nn_results = nn_results
        XoutMjpeg.__init__(self, frames, color, lossless, fps, frame_shape)
        XoutNnResults.__init__(self, det_nn, frames, nn_results)

    def visualize(self, packet: FramePacket):
        packet.frame = XoutMjpeg.decode_frame(self, packet)
        XoutNnResults.visualize(self, packet)
