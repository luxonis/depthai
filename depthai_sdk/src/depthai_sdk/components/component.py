import depthai as dai
from typing import Any, List, Optional, Tuple, Union, Type, Dict
import random

class Component():
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """
    # nodes: List[dai.Node] # List of dai.nodes that this components abstracts

    # Camera object can loop through all components to get all XLinkOuts
    # Tuple[str, Component]
    xouts: Dict[str, Any]
    def __init__(self) -> None:
        self.xouts = {}

    def createXOut(self,
        pipeline: dai.Pipeline,
        type: Type,
        name: Union[str, bool],
        out: dai.Node.Output,
        fpsLimit: Optional[float] = None) -> None:
        
        xout = pipeline.create(dai.node.XLinkOut)
        if isinstance(name, bool):
            name = f"__{str(type)}_{random.randint(100,999)}"
        xout.setStreamName(name)
        out.link(xout.input)

        if fpsLimit:
            xout.setFpsLimit(fpsLimit)

        self.xouts[name] = type
