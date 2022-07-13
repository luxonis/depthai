from abc import ABC, abstractmethod
import depthai as dai
from typing import List

class Component(ABC):
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """
    nodes: List[dai.Node] # List of dai.nodes that this components abstracts
