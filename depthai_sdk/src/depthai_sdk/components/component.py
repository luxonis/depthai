from abc import ABC, abstractmethod
from typing import Optional

import depthai as dai

from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, ReplayStream


class Component(ABC):
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """

    def forced_openvino_version(self) -> Optional[dai.OpenVINO.Version]:
        """
        Checks whether the component forces a specific OpenVINO version. Only used by NNComponent (which overrides this
        method). This function is called after Camera has been configured and right before we connect to the OAK camera.
        @return: Forced OpenVINO version (optional).
        """
        return None

    def on_pipeline_started(self, device: dai.Device) -> None:
        """
        This function gets called after the pipeline has been started. It is called from the main thread.
        It can be used to eg. initialize XlinkIn queues.
        """
        pass

    def _stream_name_ok(self, pipeline: dai.Pipeline, name: str) -> bool:
        # Check if there's already an XLinkOut stream with this name
        for node in pipeline.getAllNodes():
            if isinstance(node, dai.node.XLinkOut) and node.getStreamName() == name:
                return False
        return True

    def _create_xout(self, pipeline: dai.Pipeline, xout: XoutBase) -> XoutBase:
        for xstream in xout.xstreams():
            if not self._stream_name_ok(pipeline, xstream.name):
                continue

            if isinstance(xstream, ReplayStream):
                continue

            xlink = pipeline.createXLinkOut()
            xlink.setStreamName(xstream.name)
            xstream.stream.link(xlink.input)

        return xout
