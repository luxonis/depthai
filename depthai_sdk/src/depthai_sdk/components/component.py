import depthai as dai
from typing import Optional, Union, Type, Dict, Tuple
import random
from ..replay import Replay
from abc import ABC, abstractmethod

class Component(ABC):
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """
    # nodes: List[dai.Node] # List of dai.nodes that this components abstracts

    # Camera object can loop through all components to get all XLinkOuts
    # Dict[str, Typle(Component type, dai msg type)]
    xouts: Dict[str, Tuple[Type, Type]]
    def __init__(self) -> None:
        """
        On init, components should only parse and save passed settings. Pipeline building process
        should be done on when user starts the Camera.
        """
        self.xouts = {}

    def forcedOpenVinoVersion(self) -> Optional[dai.OpenVINO.Version]:
        """
        Checks whether the component forces a specific OpenVINO version. Only used by NNComponent (which overrides this
        method). This function is called after Camera has been configured and right before we connect to the OAK camera.
        @return: Forced OpenVINO version (optional).
        """
        return None

    @abstractmethod
    def updateDeviceInfo(self, pipeline: dai.Pipeline, device: dai.Device):
        """
        This function will be called after the app connects to the Device
        """
        raise NotImplementedError("Every component needs to include 'updateDeviceInfo()' method!")

    def createXOut(self,
        pipeline: dai.Pipeline,
        type: Type,
        name: Union[str, bool],
        out: dai.Node.Output,
        depthaiMsg: Type,
        fpsLimit: Optional[float] = None) -> None:

        # If Replay we don't want to have XLinkIn->XLinkOut. Will read
        # frames directly from last ImgFrame sent by the Replay module.
        if isinstance(name, bool):
            name = f"__{str(type)}_{random.randint(100,999)}"

        if type == Replay:
            self.xouts[name] = (type, dai.ImgFrame)
            return
        
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(name)
        out.link(xout.input)

        if fpsLimit:
            xout.setFpsLimit(fpsLimit)

        self.xouts[name] = (type, depthaiMsg)
