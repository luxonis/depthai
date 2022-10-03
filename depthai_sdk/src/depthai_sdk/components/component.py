import depthai as dai
from typing import Optional, List, Callable
from ..oak_outputs.xout_base import XoutBase, ReplayStream
from abc import ABC, abstractmethod


class Component(ABC):
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """

    def _forced_openvino_version(self) -> Optional[dai.OpenVINO.Version]:
        """
        Checks whether the component forces a specific OpenVINO version. Only used by NNComponent (which overrides this
        method). This function is called after Camera has been configured and right before we connect to the OAK camera.
        @return: Forced OpenVINO version (optional).
        """
        return None

    @abstractmethod
    def out(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        """
        Main XLink output (to the host) from the component. Component is already initialized (_update_device_info()
        is called).
        @return:
        """
        raise NotImplementedError("Every component needs to include 'out()' method!")

    @abstractmethod
    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        """
        This function will be called after the app connects to the Device
        """
        raise NotImplementedError("Every component needs to include 'updateDeviceInfo()' method!")

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
