from typing import Tuple, List

import depthai as dai

from depthai_sdk.classes.packets import FramePacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_h26x import XoutH26x
from depthai_sdk.oak_outputs.xout.xout_mjpeg import XoutMjpeg
from depthai_sdk.oak_outputs.xout.xout_nn import XoutNnResults
from depthai_sdk.visualize.bbox import BoundingBox


class XoutNnH26x(XoutNnResults, XoutH26x):
    name: str = "H26x NN Results"
    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 nn_results: StreamXout,
                 color: bool,
                 profile: dai.VideoEncoderProperties.Profile,
                 bbox: BoundingBox):
        self.nn_results = nn_results

        XoutH26x.__init__(self, frames, color, profile)
        XoutNnResults.__init__(self, det_nn, frames, nn_results, bbox)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results]



class XoutNnMjpeg(XoutNnResults, XoutMjpeg):
    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 nn_results: StreamXout,
                 color: bool,
                 lossless: bool,
                 bbox: BoundingBox):
        self.nn_results = nn_results
        XoutMjpeg.__init__(self, frames, color, lossless)
        XoutNnResults.__init__(self, det_nn, frames, nn_results, bbox)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results]

