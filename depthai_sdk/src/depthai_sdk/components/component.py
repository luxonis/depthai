from abc import ABC
import depthai as dai


class Component(ABC):
    """
    SDK component is used as an abstraction to the current DepthAI API node or group of nodes.    
    """

    def on_pipeline_started(self, device: dai.Device) -> None:
        """
        This function gets called after the pipeline has been started. It is called from the main thread.
        It can be used to eg. initialize XlinkIn queues.
        """
        pass


    # So users can use:
    # packets: Dict[Packet] = q.get()
    # depthPacket = packets['depth']
    # depthPacket = packets[stereoComp]
    def __str__(self):
        return self.out.main.__str__()

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        if isinstance(other, Component):
            return str(self) == str(other)
        elif isinstance(other, str):
            return str(self) == other
        else:
            return False

class ComponentOutput(ABC):
    """
    Output of a component
    """
    def __init__(self, component: Component):
        """
        If user hasn't specified component's output name, we will
        generate one in Xout class
        """
        self.name = None
        self._comp = component

    def set_name(self, name: str) -> 'ComponentOutput':
        """
        Name component's output, which will be used for packet names. If not specified, it
        will be generated automatically after pipeline is started (after `oak.start()`) by
        combining all Xout Stream names (eg. "6_out;3_out").
        """
        self.name = name
        return self

    # So users can use:
    # packets: Dict[Packet] = q.get()
    # depthPacket = packets['depth']
    # depthPacket = packets[stereoComp.out.depth]
    def __str__(self):
        return self.name

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        if isinstance(other, ComponentOutput):
            return str(self) == str(other)
        elif isinstance(other, str):
            return str(self) == other
        else:
            return False

