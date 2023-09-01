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
