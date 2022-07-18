from abc import ABC, abstractmethod
import depthai as dai
from typing import List, Optional, Tuple

class Component(ABC):
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """
    # nodes: List[dai.Node] # List of dai.nodes that this components abstracts

    # Camera object can loop through all components to get all XLinkOuts
    xouts: List[Tuple[str, dai.Node.Output]] = []

    def _createXOut(self,
        pipeline: dai.Pipeline,
        name: str,
        out: dai.Node.Output,
        fpsLimit: Optional[float] = None) -> None:
        
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(name)
        out.link(xout.input)

        if fpsLimit:
            xout.setFpsLimit(fpsLimit)

        self.xouts.append((name, out))


