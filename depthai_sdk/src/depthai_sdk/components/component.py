import depthai as dai
from typing import Optional, List, Callable
from ..classes.xout_base import XoutBase, ReplayStream
from abc import ABC, abstractmethod

class Component(ABC):
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """
    xouts: List[str]

    def __init__(self):
        """
        On init, components should only parse and save passed settings. Pipeline building process
        should be done on when user starts the Camera.
        """
        self.xouts: List[str] = []

    def _forced_openvino_version(self) -> Optional[dai.OpenVINO.Version]:
        """
        Checks whether the component forces a specific OpenVINO version. Only used by NNComponent (which overrides this
        method). This function is called after Camera has been configured and right before we connect to the OAK camera.
        @return: Forced OpenVINO version (optional).
        """
        return None

    @abstractmethod
    def out(self, pipeline: dai.Pipeline, callback: Callable) -> XoutBase:
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

    def _create_xout(self, pipeline: dai.Pipeline, xout: XoutBase) -> XoutBase:
        for xstream in xout.xstreams():
            if xstream.name in self.xouts:
                continue

            if isinstance(xstream, ReplayStream):
                continue

            xlink = pipeline.createXLinkOut()
            xlink.setStreamName(xstream.name)
            xstream.stream.link(xlink.input)
            self.xouts.append(xstream.name)

        return xout