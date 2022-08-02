import depthai as dai
from typing import Optional, Union, Type, Dict, Tuple
import random
from ..replay import Replay

class Component():
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """
    # nodes: List[dai.Node] # List of dai.nodes that this components abstracts

    # Camera object can loop through all components to get all XLinkOuts
    # Dict[str, Typle(Component type, dai msg type)]
    xouts: Dict[str, Tuple[Type, Type]]
    def __init__(self) -> None:
        self.xouts = {}

    def updateDeviceInfo(self, device: dai.Device):
        """
        This function will be called 
        """
        pass

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
