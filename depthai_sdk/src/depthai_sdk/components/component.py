from abc import ABC, abstractmethod
import depthai as dai
from typing import List, Optional, Tuple, Union, Type
import random

class Component(ABC):
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """
    # nodes: List[dai.Node] # List of dai.nodes that this components abstracts

    # Camera object can loop through all components to get all XLinkOuts
    # Tuple[str, Component]
    xouts: Tuple = {}

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
