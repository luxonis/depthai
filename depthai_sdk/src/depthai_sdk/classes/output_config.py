from abc import abstractmethod
from typing import Optional, Callable, Union, Tuple, List

import depthai as dai

from ..visualize import NewVisualizer
from ..oak_outputs.xout import XoutFrames
from ..oak_outputs.xout_base import XoutBase
from ..record import Record


class VisualizeConfig:
    # TODO: support visualziation configs. eg. colors, fonts, locations where text in BBs is displayed,
    # BB rectangle config (transparency, rounded edges etc.)
    scale: Union[None, float, Tuple[int, int]]
    fps: bool

    def __init__(self, scale, fps):
        self.scale = scale
        self.fps = fps


class BaseConfig:
    @abstractmethod
    def setup(self, pipeline: dai.Pipeline, device, names: List[str]) -> XoutBase:
        raise NotImplementedError()


class OutputConfig(BaseConfig):
    """
    Saves callbacks/visualizers until the device is fully initialized. I'll admit it's not the cleanest solution.
    """
    visualizer: Optional[NewVisualizer]  # Visualization
    output: Callable  # Output of the component (a callback)
    callback: Callable  # Callback that gets called after syncing

    def __init__(self, output: Callable, callback: Callable, visualizer: NewVisualizer = None):
        self.output = output
        self.callback = callback
        self.visualizer = visualizer

    def find_new_name(self, name: str, names: List[str]):
        while True:
            arr = name.split(' ')
            num = arr[-1]
            if num.isnumeric():
                arr[-1] = str(int(num) + 1)
                name = " ".join(arr)
            else:
                name = f"{name} 2"
            if name not in names:
                return name

    def setup(self, pipeline: dai.Pipeline, device, names: List[str]) -> XoutBase:
        xoutbase: XoutBase = self.output(pipeline, device)
        xoutbase.setup_base(self.callback)

        if xoutbase.name in names:  # Stream name already exist, append a number to it
            xoutbase.name = self.find_new_name(xoutbase.name, names)
        names.append(xoutbase.name)

        if self.visualizer:
            xoutbase.setup_visualize(self.visualizer, xoutbase.name)

        return xoutbase


class RecordConfig(BaseConfig):
    rec: Record
    outputs: List[Callable]

    def __init__(self, outputs: List[Callable], rec: Record):
        self.outputs = outputs
        self.rec = rec

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, _) -> XoutBase:
        xouts: List[XoutFrames] = []
        for output in self.outputs:
            xoutbase: XoutFrames = output(pipeline, device)
            xoutbase.setup_base(None)
            xouts.append(xoutbase)

        self.rec.setup_base(None)
        self.rec.start(device, xouts)

        return self.rec
